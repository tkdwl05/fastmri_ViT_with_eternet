"""
ETER-ViT 학습 스크립트 v7 — DDP (2x RTX TITAN)
  - 단일 GPU v7 스크립트의 DDP 버전
  - effective BATCH_SIZE = BATCH_SIZE * world_size (16 * 2 = 32)

실행:
  torchrun --nproc_per_node=2 v7/main_train_eter_v7_ddp.py
"""

import os
import sys
import time
import datetime
import pytz

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import wandb
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'tools'))

from myConfig_choh_ETER_model_v7 import *
from u_choh_model_ETER_ViT import choh_ViT
from u_choh_model_ETER_ViT_v5 import choh_Decoder3_ETER_v5
from u_choh_SSIM import SSIM
from dataloader_h5_v5 import FastMRI_H5_Dataloader

NUM_EPOCHS         = int(os.environ.get('SANITY_NUM_EPOCHS', NUM_EPOCHS))
VAL_EVERY_N_EPOCHS = int(os.environ.get('SANITY_VAL_EVERY_N_EPOCHS', VAL_EVERY_N_EPOCHS))
NUM_VAL_FILES      = None


def skimage_ssim_batch(pred, target):
    p = pred.detach().float().cpu().numpy()
    t = target.detach().float().cpu().numpy()
    if p.ndim == 4:
        p, t = p[:, 0], t[:, 0]
    vals = []
    for i in range(p.shape[0]):
        dr = t[i].max() - t[i].min()
        if dr <= 0:
            continue
        vals.append(compare_ssim(t[i], p[i], data_range=dr))
    return float(np.mean(vals)) if vals else 0.0


def run_val_ddp(model, val_loader, criterion_l1, device, world_size):
    model.eval()
    all_ssim, all_psnr, all_nmse, all_l1 = [], [], [], []
    with torch.no_grad():
        for sample in val_loader:
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            with torch.amp.autocast("cuda"):
                out = model(data_in_img, data_in)
            out_f = out.float(); ref_f = data_ref.float()
            mse  = torch.mean((out_f - ref_f) ** 2)
            psnr = (20 * torch.log10(ref_f.max() / torch.sqrt(mse.clamp(min=1e-10)))).item()
            nmse = (torch.norm(out_f - ref_f) ** 2 / torch.norm(ref_f) ** 2).item()
            ssim = skimage_ssim_batch(out_f, ref_f)
            l1   = criterion_l1(out_f, ref_f).item()
            all_psnr.append(psnr); all_nmse.append(nmse); all_ssim.append(ssim); all_l1.append(l1)

    local = torch.tensor([sum(all_ssim), sum(all_psnr), sum(all_nmse), sum(all_l1),
                          float(len(all_ssim))], device=device)
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    s, p, nm, l1, n = local.tolist()
    model.train()
    return {
        'ssim': s / max(n, 1.0), 'psnr': p / max(n, 1.0),
        'nmse': nm / max(n, 1.0), 'l1':   l1 / max(n, 1.0),
    }


def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    rank       = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)

    def _print(*a, **kw):
        if is_main: print(*a, **kw)

    _print('====================================================')
    _print(f' [ETER-ViT v7 DDP] world_size={world_size}  effective_BS={BATCH_SIZE * world_size}')
    _print('====================================================')
    _print(f'Device per rank: {device}')
    _print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

    vit_choh = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1
    ).to(device)
    eter_decoder = choh_Decoder3_ETER_v5(
        encoder=vit_choh,
        eter_n_hori_hidden=NUM_ETER_HORI_HIDDEN,
        eter_n_vert_hidden=NUM_ETER_VERT_HIDDEN,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=DROPOUT,
    ).to(device)

    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        state = torch.load(RESUME_CKPT, map_location=device)
        eter_decoder.load_state_dict(state)
        _print(f"Resumed from: {RESUME_CKPT}")
        run_baseline = True
    else:
        _print("Scratch training (RESUME_CKPT is None)")
        run_baseline = False

    eter_decoder = DDP(eter_decoder, device_ids=[local_rank], find_unused_parameters=False)

    if is_main:
        num_params = sum(p.numel() for p in eter_decoder.parameters() if p.requires_grad)
        _print(f"모델 파라미터 수: {num_params / 1e6:.1f}M")

    criterion_l1   = nn.L1Loss()
    criterion_ssim_loss = SSIM().to(device)
    optimizer = torch.optim.Adam(
        eter_decoder.parameters(), lr=LEARNING_RATE_ADAM,
        weight_decay=LAMBDA_REGULAR_PER_PIXEL,
    )

    _print("\nFastMRI 데이터 파이프라인 (DDP sampler) 연결 중...")
    train_ds = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_train', num_files=None,
        augment=TRAIN_AUGMENT, augment_flip_p=TRAIN_AUGMENT_FLIP_P,
    )
    val_ds = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_val', num_files=NUM_VAL_FILES,
        random_mask=False, augment=False,
    )
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    trainloader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS_TRAIN, pin_memory=True,
        persistent_workers=True, prefetch_factor=PREFETCH_FACTOR,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, sampler=val_sampler,
        num_workers=NUM_WORKERS_VAL, pin_memory=True,
    )
    _print(f"Train samples (per rank): {len(train_sampler)}  total {len(train_ds)}")
    _print(f"Val   samples (per rank): {len(val_sampler)}  total {len(val_ds)}")

    steps_per_epoch = len(trainloader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    if is_main:
        wandb.init(
            project='ViT-MRI-Recon',
            name=f'ETER_v7_ddp_ws{world_size}_BS{BATCH_SIZE}_LR{LEARNING_RATE_ADAM}_EP{NUM_EPOCHS}',
            config={
                'model': 'ETER-ViT-v7-DDP',
                'world_size': world_size,
                'batch_size_per_rank': BATCH_SIZE,
                'effective_batch_size': BATCH_SIZE * world_size,
                'num_epochs': NUM_EPOCHS,
                'learning_rate': LEARNING_RATE_ADAM,
                'dropout': DROPOUT,
                'weight_decay': LAMBDA_REGULAR_PER_PIXEL,
                'ssim_weight': LAMBDA_SSIM_PER_PIXEL,
                'augment': TRAIN_AUGMENT,
                'augment_flip_p': TRAIN_AUGMENT_FLIP_P,
                'val_every_n_epochs': VAL_EVERY_N_EPOCHS,
                'earlystop_patience': EARLYSTOP_PATIENCE,
                'eter_hori_hidden': NUM_ETER_HORI_HIDDEN,
                'eter_vert_hidden': NUM_ETER_VERT_HIDDEN,
                'num_params': num_params,
                'train_samples_total': len(train_ds),
                'val_samples_total':   len(val_ds),
            },
        )

    scaler = torch.amp.GradScaler('cuda')
    eter_decoder.train()
    best_val_ssim = -1.0
    no_improve_val_count = 0
    early_stopped = False
    global_step = 0
    log_path = os.path.join(PATH_FOLDER, 'log_ddp.txt')

    if run_baseline:
        if is_main: tqdm.write('Resume baseline val 측정 중...')
        baseline = run_val_ddp(eter_decoder, val_loader, criterion_l1, device, world_size)
        best_val_ssim = baseline['ssim']
        if is_main:
            tqdm.write(f'  [Baseline]  SSIM={baseline["ssim"]:.4f} PSNR={baseline["psnr"]:.2f}')
            with open(log_path, 'a') as f:
                f.write(f'BASELINE val_ssim={baseline["ssim"]:.4f}\n')
            torch.save(eter_decoder.module.state_dict(),
                       os.path.join(PATH_FOLDER, 'eter_vit_best.pt'))
    elif is_main:
        with open(log_path, 'a') as f:
            f.write(f'DDP SCRATCH START ws={world_size} BS_per={BATCH_SIZE} '
                    f'effBS={BATCH_SIZE*world_size} LR={LEARNING_RATE_ADAM} EP={NUM_EPOCHS}\n')

    tic = time.time()
    epoch_bar = tqdm(range(NUM_EPOCHS), desc='전체 진행', unit='epoch', disable=not is_main)
    for epoch in epoch_bar:
        train_sampler.set_epoch(epoch)
        epoch_loss = epoch_ssim_c = 0.0
        batch_bar = tqdm(trainloader, desc=f'Epoch {epoch+1:3d}/{NUM_EPOCHS}',
                         leave=False, unit='batch', disable=not is_main)

        for sample in batch_bar:
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)

            with torch.amp.autocast("cuda"):
                out = eter_decoder(data_in_img, data_in)

            out_fp    = out.float()
            loss_l1   = criterion_l1(out_fp, data_ref)
            loss_ssim = 1 - criterion_ssim_loss(out_fp, data_ref)
            loss = loss_l1 + LAMBDA_SSIM_PER_PIXEL * loss_ssim

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(eter_decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                ssim_c = 1 - loss_ssim.item()
            global_step += 1
            epoch_loss   += loss.item()
            epoch_ssim_c += ssim_c
            if is_main:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/loss_l1': loss_l1.item(),
                    'train/loss_ssim': loss_ssim.item(),
                    'train/ssim_custom': ssim_c,
                    'train/lr': scheduler.get_last_lr()[0],
                }, step=global_step)
                batch_bar.set_postfix(
                    Loss=f'{loss.item():.4f}', SSIM_c=f'{ssim_c:.4f}',
                    LR=f'{scheduler.get_last_lr()[0]:.2e}',
                )

        n_batches = max(len(trainloader), 1)
        avg_loss   = epoch_loss   / n_batches
        avg_ssim_c = epoch_ssim_c / n_batches

        do_val = (epoch + 1) % VAL_EVERY_N_EPOCHS == 0
        if do_val:
            if is_main: tqdm.write(f'  [Val ep{epoch+1}] running on all ranks...')
            val_metrics = run_val_ddp(eter_decoder, val_loader, criterion_l1, device, world_size)
            if is_main:
                tqdm.write(f'  [Val]  SSIM={val_metrics["ssim"]:.4f} '
                           f'PSNR={val_metrics["psnr"]:.2f} NMSE={val_metrics["nmse"]:.4f}'
                           f' L1={val_metrics["l1"]:.4f}')
                wandb.log({
                    'val/ssim': val_metrics['ssim'], 'val/psnr': val_metrics['psnr'],
                    'val/nmse': val_metrics['nmse'], 'val/l1':   val_metrics['l1'],
                    'epoch': epoch + 1,
                    'epoch/train_loss': avg_loss,
                    'epoch/train_ssim_custom': avg_ssim_c,
                }, step=global_step)
                with open(log_path, 'a') as f:
                    f.write(f'Epoch {epoch+1}/{NUM_EPOCHS}  train_loss={avg_loss:.4f}'
                            f'  val_ssim={val_metrics["ssim"]:.4f}'
                            f'  val_psnr={val_metrics["psnr"]:.2f}\n')

                if val_metrics['ssim'] > best_val_ssim:
                    best_val_ssim = val_metrics['ssim']
                    torch.save(eter_decoder.module.state_dict(),
                               os.path.join(PATH_FOLDER, 'eter_vit_best.pt'))
                    tqdm.write(f'  [Best Ckpt] val_ssim {best_val_ssim:.4f}')
                    no_improve_val_count = 0
                else:
                    no_improve_val_count += 1
                    tqdm.write(f'  [No improvement {no_improve_val_count}/{EARLYSTOP_PATIENCE}]')

            stop_tensor = torch.tensor([no_improve_val_count], dtype=torch.long, device=device)
            dist.broadcast(stop_tensor, src=0)
            no_improve_val_count = int(stop_tensor.item())
            if no_improve_val_count >= EARLYSTOP_PATIENCE:
                if is_main: tqdm.write(f'  [EarlyStop] at epoch {epoch+1}')
                early_stopped = True
                break

        if is_main and (epoch + 1) % 5 == 0:
            torch.save(eter_decoder.module.state_dict(),
                       os.path.join(PATH_FOLDER, f'eter_vit_epoch_{epoch+1}.pt'))
            tqdm.write(f'  Checkpoint: eter_vit_epoch_{epoch+1}.pt')

    if is_main:
        toc = time.time()
        _print(f'\n학습 완료 (early_stopped={early_stopped})  소요: {toc - tic:.0f}s')
        _print(f'Best Val SSIM: {best_val_ssim:.4f}')
        wandb.finish()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
