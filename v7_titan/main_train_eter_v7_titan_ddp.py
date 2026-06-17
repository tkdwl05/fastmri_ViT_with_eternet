"""
ETER-ViT 학습 스크립트 v7_titan — DDP (2x TITAN RTX 24GB, scratch, 384×384)

  - Single-GPU 버전 (main_train_eter_v7_titan.py) 의 DDP 변환
  - effective BATCH_SIZE = BATCH_SIZE * world_size (default 4 * 2 = 8)
  - 평가/loss 는 brain_mask + weighted composite 동일 적용

실행:
  torchrun --nproc_per_node=2 v7_titan/main_train_eter_v7_titan_ddp.py
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

from myConfig_choh_ETER_model_v7_titan import *
from u_choh_model_ETER_ViT import choh_ViT
from u_choh_model_ETER_ViT_v7_titan import choh_Decoder3_ETER_v7_titan
from u_choh_SSIM import SSIM
from dataloader_h5_v5 import FastMRI_H5_Dataloader
from check_recon_env import check_env_for_model

NUM_EPOCHS         = int(os.environ.get('SANITY_NUM_EPOCHS', NUM_EPOCHS))
VAL_EVERY_N_EPOCHS = int(os.environ.get('SANITY_VAL_EVERY_N_EPOCHS', VAL_EVERY_N_EPOCHS))
BATCH_SIZE         = int(os.environ.get('SMOKE_BS', BATCH_SIZE))
NUM_VAL_FILES      = None


def skimage_ssim_batch_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple:
    """brain_mask 안에서만 skimage SSIM 평균 + sample count.

    DDP 에서 rank-local sum + count 로 반환 → 호출부에서 all_reduce 로 결합.
    """
    p = pred.detach().float().cpu().numpy()
    t = target.detach().float().cpu().numpy()
    m = mask.detach().float().cpu().numpy()
    if p.ndim == 4:
        p, t, m = p[:, 0], t[:, 0], m[:, 0]
    m_bool = m > 0.5
    s_sum, n = 0.0, 0
    for i in range(p.shape[0]):
        if not m_bool[i].any():
            continue
        t_in = t[i][m_bool[i]]
        dr = float(t_in.max() - t_in.min())
        if dr <= 0:
            continue
        _, ssim_map = compare_ssim(t[i], p[i], data_range=dr, full=True)
        s_sum += float(ssim_map[m_bool[i]].mean())
        n += 1
    return s_sum, n


def run_val_ddp(model, val_loader, device, world_size):
    """rank-local 누적 후 all_reduce 로 global mean. composite metric 동일 산식."""
    model.eval()
    s_ssim = s_psnr = s_nmse = s_l1 = 0.0
    n_ssim = n_other = 0
    with torch.no_grad():
        for sample in val_loader:
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            brain_mask  = sample['brain_mask'].float().to(device)

            with torch.amp.autocast('cuda'):
                out = model(data_in_img, data_in)

            out_f = out.float()
            ref_f = data_ref.float()
            m     = brain_mask
            m_sum = m.sum().clamp(min=1.0)

            diff_sq_sum = ((out_f - ref_f) ** 2 * m).sum()
            mse = diff_sq_sum / m_sum
            ref_max_in_mask = (ref_f * m).max().clamp(min=1e-10)
            psnr = (20 * torch.log10(ref_max_in_mask / torch.sqrt(mse.clamp(min=1e-10)))).item()
            ref_sq_sum = (ref_f ** 2 * m).sum().clamp(min=1e-10)
            nmse = (diff_sq_sum / ref_sq_sum).item()
            l1   = (((out_f - ref_f).abs() * m).sum() / m_sum).item()

            ssim_sum_b, n_b = skimage_ssim_batch_masked(out_f, ref_f, m)

            s_ssim += ssim_sum_b
            n_ssim += n_b
            s_psnr += psnr
            s_nmse += nmse
            s_l1   += l1
            n_other += 1

    local = torch.tensor(
        [s_ssim, float(n_ssim), s_psnr, s_nmse, s_l1, float(n_other)],
        device=device, dtype=torch.float64,
    )
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    g_s_ssim, g_n_ssim, g_s_psnr, g_s_nmse, g_s_l1, g_n_other = local.tolist()
    model.train()

    ssim_m = g_s_ssim / max(g_n_ssim, 1.0)
    psnr_m = g_s_psnr / max(g_n_other, 1.0)
    nmse_m = g_s_nmse / max(g_n_other, 1.0)
    l1_m   = g_s_l1   / max(g_n_other, 1.0)
    psnr_n = min(psnr_m, PSNR_NORM) / PSNR_NORM
    nmse_n = max(0.0, 1.0 - min(nmse_m, 1.0))
    composite = (COMPOSITE_W_SSIM * ssim_m
                 + COMPOSITE_W_PSNR * psnr_n
                 + COMPOSITE_W_NMSE * nmse_n)
    return {
        'ssim': ssim_m, 'psnr': psnr_m, 'nmse': nmse_m, 'l1': l1_m,
        'composite': composite,
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
        if is_main:
            print(*a, **kw)

    _print('====================================================')
    _print(f' [ETER-ViT v7_titan DDP] world_size={world_size}  '
           f'BS_per_rank={BATCH_SIZE}  effective_BS={BATCH_SIZE * world_size}')
    _print('====================================================')
    _print(f'rank={rank} local_rank={local_rank} device={device}')
    _print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

    if is_main and not check_env_for_model('eter', 'myConfig_choh_ETER_model_v7_titan', strict=True):
        dist.destroy_process_group()
        return

    vit_choh = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1,
    ).to(device)

    eter_decoder = choh_Decoder3_ETER_v7_titan(
        encoder=vit_choh,
        eter_n_hori_hidden=NUM_ETER_HORI_HIDDEN,
        eter_n_vert_hidden=NUM_ETER_VERT_HIDDEN,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=DROPOUT,
        unet_depth=ETER_UNET_DEPTH,
        unet_wf=ETER_UNET_WF,
    ).to(device)

    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        state = torch.load(RESUME_CKPT, map_location=device)
        eter_decoder.load_state_dict(state)
        _print(f"Resumed weights from: {RESUME_CKPT}")
        run_baseline = True
    else:
        _print("Scratch training (RESUME_CKPT is None)")
        run_baseline = False

    # find_unused_parameters=True: ViT-Base 의 일부 param 이 forward path 에 의해 활성화 안 됨 → DDP reduction sync 에러 회피
    eter_decoder = DDP(eter_decoder, device_ids=[local_rank], find_unused_parameters=True)

    if is_main:
        num_params = sum(p.numel() for p in eter_decoder.parameters() if p.requires_grad)
        _print(f"모델 파라미터 수: {num_params / 1e6:.1f}M")
    else:
        num_params = 0

    criterion_ssim_loss = SSIM().to(device)
    optimizer = torch.optim.Adam(
        eter_decoder.parameters(), lr=LEARNING_RATE_ADAM,
        weight_decay=LAMBDA_REGULAR_PER_PIXEL,
    )

    _print("\nFastMRI 데이터 파이프라인 (DDP sampler) 연결 중...")
    train_ds = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_train', num_files=None,
        target_size=IMAGE_SIZE[0],
        augment=TRAIN_AUGMENT, augment_flip_p=TRAIN_AUGMENT_FLIP_P,
    )
    val_ds = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_val', num_files=NUM_VAL_FILES,
        target_size=IMAGE_SIZE[0],
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
        val_ds, batch_size=max(1, BATCH_SIZE // 2), sampler=val_sampler,
        num_workers=NUM_WORKERS_VAL, pin_memory=True,
    )
    _print(f"Train: total {len(train_ds)} → per-rank {len(train_sampler)}  "
           f"steps_per_epoch (per-rank) {len(trainloader)}")
    _print(f"Val:   total {len(val_ds)}   → per-rank {len(val_sampler)}")

    steps_per_epoch = len(trainloader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    _print(f"Scheduler: CosineAnnealingLR T_max={total_steps} (={NUM_EPOCHS} epochs), eta_min=1e-6")
    _print(f"EarlyStopping: patience={EARLYSTOP_PATIENCE} val checks (composite 기준)")

    if is_main:
        wandb.init(
            project='ViT-MRI-Recon',
            name=f'ETER_v7_titan_ddp_ws{world_size}_BS{BATCH_SIZE}_LR{LEARNING_RATE_ADAM}_EP{NUM_EPOCHS}',
            config={
                'model': 'ETER-ViT-v7_titan-DDP',
                'world_size': world_size,
                'batch_size_per_rank': BATCH_SIZE,
                'effective_batch_size': BATCH_SIZE * world_size,
                'image_size': IMAGE_SIZE,
                'patch_size': PATCH_SIZE,
                'encoder_hidden': NUM_VIT_ENCODER_HIDDEN,
                'encoder_layers': NUM_VIT_ENCODER_LAYER,
                'eter_hori_hidden': NUM_ETER_HORI_HIDDEN,
                'eter_vert_hidden': NUM_ETER_VERT_HIDDEN,
                'unet_depth':       ETER_UNET_DEPTH,
                'unet_wf':          ETER_UNET_WF,
                'decoder_dim':    NUM_VIT_DECODER_DIM,
                'decoder_depth':  NUM_VIT_DECODER_DEPTH,
                'dropout':        DROPOUT,
                'weight_decay':   LAMBDA_REGULAR_PER_PIXEL,
                'augment':        TRAIN_AUGMENT,
                'augment_flip_p': TRAIN_AUGMENT_FLIP_P,
                'num_epochs':     NUM_EPOCHS,
                'learning_rate':  LEARNING_RATE_ADAM,
                'ssim_weight':    LAMBDA_SSIM_PER_PIXEL,
                'val_metric':       'skimage_ssim_masked',
                'earlystop_metric': 'composite',
                'earlystop_patience': EARLYSTOP_PATIENCE,
                'val_every_n_epochs': VAL_EVERY_N_EPOCHS,
                'composite_w_ssim': COMPOSITE_W_SSIM,
                'composite_w_psnr': COMPOSITE_W_PSNR,
                'composite_w_nmse': COMPOSITE_W_NMSE,
                'psnr_norm':        PSNR_NORM,
                'num_params':         num_params,
                'train_samples':      len(train_ds),
                'val_samples':        len(val_ds),
                'scheduler': 'CosineAnnealingLR',
                'T_max':     total_steps,
                'eta_min':   1e-6,
                'num_workers_train': NUM_WORKERS_TRAIN,
                'prefetch_factor':   PREFETCH_FACTOR,
            },
        )
        wandb.watch(eter_decoder, log='gradients', log_freq=200)

    _print(f"\n학습 시작 (총 {NUM_EPOCHS} 에폭, EarlyStop 가능)")
    scaler = torch.amp.GradScaler('cuda')
    eter_decoder.train()
    best_val_composite = -1.0
    best_val = {'ssim': None, 'psnr': None, 'nmse': None, 'l1': None, 'composite': None}
    no_improve_val_count = 0
    early_stopped = False
    tic = time.time()
    global_step = 0
    log_path = os.path.join(PATH_FOLDER, 'log_ddp.txt')

    if run_baseline:
        if is_main:
            tqdm.write('Resume baseline val 측정 중...')
        baseline = run_val_ddp(eter_decoder, val_loader, device, world_size)
        best_val_composite = baseline['composite']
        best_val = dict(baseline)
        if is_main:
            tqdm.write(
                f'  [Baseline]  Composite: {baseline["composite"]:.4f}'
                f'  SSIM_m: {baseline["ssim"]:.4f}'
                f'  PSNR: {baseline["psnr"]:.2f}dB  NMSE: {baseline["nmse"]:.4f}'
                f'  L1: {baseline["l1"]:.4f}'
            )
            with open(log_path, 'a') as f:
                f.write(
                    f'BASELINE (resume from {RESUME_CKPT}): '
                    f'val_composite={baseline["composite"]:.4f}'
                    f'  val_ssim_m={baseline["ssim"]:.4f}'
                    f'  val_psnr={baseline["psnr"]:.2f}'
                    f'  val_nmse={baseline["nmse"]:.4f}  val_l1={baseline["l1"]:.4f}\n'
                )
            torch.save(eter_decoder.module.state_dict(),
                       os.path.join(PATH_FOLDER, 'eter_vit_best.pt'))
    elif is_main:
        with open(log_path, 'a') as f:
            f.write(
                f'DDP SCRATCH START  ws={world_size}  BS_per={BATCH_SIZE}'
                f'  effBS={BATCH_SIZE*world_size}  LR={LEARNING_RATE_ADAM}  EPOCHS={NUM_EPOCHS}\n'
            )

    epoch_bar = tqdm(range(NUM_EPOCHS), desc='전체 진행', unit='epoch', disable=not is_main)
    for epoch in epoch_bar:
        train_sampler.set_epoch(epoch)
        epoch_loss = epoch_ssim = epoch_psnr = epoch_nmse = epoch_l1 = 0.0
        batch_bar = tqdm(trainloader, desc=f'Epoch {epoch+1:3d}/{NUM_EPOCHS}',
                         leave=False, unit='batch', disable=not is_main)

        for sample in batch_bar:
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            brain_mask  = sample['brain_mask'].float().to(device)

            with torch.amp.autocast('cuda'):
                out = eter_decoder(data_in_img, data_in)

            out_fp    = out.float()
            m_sum     = brain_mask.sum().clamp(min=1.0)
            loss_l1   = ((out_fp - data_ref).abs() * brain_mask).sum() / m_sum
            loss_ssim = 1 - criterion_ssim_loss(out_fp, data_ref, mask=brain_mask)
            loss = loss_l1 + LAMBDA_SSIM_PER_PIXEL * loss_ssim

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(eter_decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                out_f = out_fp.detach()
                ref_f = data_ref.float()
                m_t   = brain_mask
                diff_sq_sum_t = ((out_f - ref_f) ** 2 * m_t).sum()
                mse_val  = (diff_sq_sum_t / m_sum).item()
                ref_max_in_mask_t = (ref_f * m_t).max().clamp(min=1e-10).item()
                psnr_val = 20 * np.log10(ref_max_in_mask_t / max(np.sqrt(mse_val), 1e-10))
                ref_sq_sum_t = (ref_f ** 2 * m_t).sum().clamp(min=1e-10)
                nmse_val = (diff_sq_sum_t / ref_sq_sum_t).item()
                ssim_val_train = 1 - loss_ssim.item()

            global_step += 1
            epoch_loss += loss.item()
            epoch_ssim += ssim_val_train
            epoch_psnr += psnr_val
            epoch_nmse += nmse_val
            epoch_l1   += loss_l1.item()

            if is_main:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/loss_l1': loss_l1.item(),
                    'train/loss_ssim': loss_ssim.item(),
                    'train/ssim_custom': ssim_val_train,
                    'train/psnr': psnr_val,
                    'train/nmse': nmse_val,
                    'train/lr': scheduler.get_last_lr()[0],
                }, step=global_step)
                batch_bar.set_postfix(
                    Loss=f'{loss.item():.4f}', PSNR=f'{psnr_val:.2f}dB',
                    NMSE=f'{nmse_val:.4f}', SSIM_c=f'{ssim_val_train:.4f}',
                    LR=f'{scheduler.get_last_lr()[0]:.2e}',
                )

        n_batches = max(len(trainloader), 1)
        avg_loss       = epoch_loss / n_batches
        avg_train_ssim = epoch_ssim / n_batches
        avg_train_psnr = epoch_psnr / n_batches
        avg_train_nmse = epoch_nmse / n_batches
        avg_train_l1   = epoch_l1   / n_batches

        if is_main:
            wandb.log({
                'epoch': epoch + 1,
                'epoch/train_loss': avg_loss,
                'epoch/train_ssim_custom': avg_train_ssim,
                'epoch/train_psnr': avg_train_psnr,
                'epoch/train_nmse': avg_train_nmse,
                'epoch/train_l1': avg_train_l1,
            }, step=global_step)

        do_val = (epoch + 1) % VAL_EVERY_N_EPOCHS == 0

        if do_val:
            if is_main:
                tqdm.write(f'  [Val ep{epoch+1}] running on all ranks...')
            val_metrics = run_val_ddp(eter_decoder, val_loader, device, world_size)

            if is_main:
                tqdm.write(
                    f'  [Val]  Composite: {val_metrics["composite"]:.4f}'
                    f'  SSIM: {val_metrics["ssim"]:.4f}'
                    f'  PSNR: {val_metrics["psnr"]:.2f}dB'
                    f'  NMSE: {val_metrics["nmse"]:.4f}'
                    f'  L1: {val_metrics["l1"]:.4f}'
                )
                wandb.log({
                    'val/composite':   val_metrics['composite'],
                    'val/ssim_masked': val_metrics['ssim'],
                    'val/psnr_masked': val_metrics['psnr'],
                    'val/nmse_masked': val_metrics['nmse'],
                    'val/l1_masked':   val_metrics['l1'],
                }, step=global_step)
                epoch_bar.set_postfix(
                    train_ssim=f'{avg_train_ssim:.4f}',
                    val_ssim=f'{val_metrics["ssim"]:.4f}',
                    val_composite=f'{val_metrics["composite"]:.4f}',
                )
                with open(log_path, 'a') as f:
                    f.write(
                        f'Epoch {epoch+1}/{NUM_EPOCHS}'
                        f'  train_loss={avg_loss:.4f}'
                        f'  train_ssim_custom={avg_train_ssim:.4f}'
                        f'  val_composite={val_metrics["composite"]:.4f}'
                        f'  val_ssim_m={val_metrics["ssim"]:.4f}'
                        f'  val_psnr={val_metrics["psnr"]:.2f}'
                        f'  val_nmse={val_metrics["nmse"]:.4f}'
                        f'  val_l1={val_metrics["l1"]:.4f}\n'
                    )

                if val_metrics['composite'] > best_val_composite:
                    best_val_composite = val_metrics['composite']
                    best_val = dict(val_metrics)
                    best_ckpt_path = os.path.join(PATH_FOLDER, 'eter_vit_best.pt')
                    torch.save(eter_decoder.module.state_dict(), best_ckpt_path)
                    tqdm.write(
                        f'  [Best Ckpt] composite {best_val_composite:.4f}'
                        f' (SSIM_m {val_metrics["ssim"]:.4f},'
                        f' PSNR {val_metrics["psnr"]:.2f}dB,'
                        f' NMSE {val_metrics["nmse"]:.4f}) → {best_ckpt_path}'
                    )
                    no_improve_val_count = 0
                else:
                    no_improve_val_count += 1
                    tqdm.write(
                        f'  [No improvement] composite {val_metrics["composite"]:.4f}'
                        f' < best {best_val_composite:.4f}'
                        f'  (no-improve {no_improve_val_count}/{EARLYSTOP_PATIENCE})'
                    )
                wandb.log({'val/no_improve_count': no_improve_val_count}, step=global_step)

            stop_t = torch.tensor([no_improve_val_count], dtype=torch.long, device=device)
            dist.broadcast(stop_t, src=0)
            no_improve_val_count = int(stop_t.item())

            if no_improve_val_count >= EARLYSTOP_PATIENCE:
                if is_main:
                    tqdm.write(
                        f'  [EarlyStop] {no_improve_val_count} consecutive val checks'
                        f' without composite improvement → 학습 종료 (epoch {epoch+1})'
                    )
                    with open(log_path, 'a') as f:
                        f.write(f'EARLYSTOP at epoch {epoch+1}\n')
                early_stopped = True
                break
        else:
            if is_main:
                epoch_bar.set_postfix(train_ssim=f'{avg_train_ssim:.4f}', train_loss=f'{avg_loss:.4f}')
                with open(log_path, 'a') as f:
                    f.write(
                        f'Epoch {epoch+1}/{NUM_EPOCHS}'
                        f'  train_loss={avg_loss:.4f}'
                        f'  train_ssim_custom={avg_train_ssim:.4f}\n'
                    )

        if is_main and (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(PATH_FOLDER, f'eter_vit_epoch_{epoch+1}.pt')
            torch.save(eter_decoder.module.state_dict(), ckpt_path)
            tqdm.write(f'  Checkpoint 저장: {ckpt_path}')

    if is_main:
        toc = time.time()
        _print(f'\n학습 완료 (early_stopped={early_stopped})  소요 시간: {toc - tic:.0f}초')
        _print(f'Best Val Composite: {best_val_composite:.4f}')
        if best_val['ssim'] is not None:
            _print(
                f'  Best Val → Composite: {best_val["composite"]:.4f}'
                f'  SSIM_m: {best_val["ssim"]:.4f}'
                f'  PSNR: {best_val["psnr"]:.2f}dB'
                f'  NMSE: {best_val["nmse"]:.4f}  L1: {best_val["l1"]:.4f}'
            )
        _print(f'Best Checkpoint: {os.path.join(PATH_FOLDER, "eter_vit_best.pt")}')
        wandb.finish()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
