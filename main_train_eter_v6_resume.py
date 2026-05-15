"""
ETER-ViT 학습 스크립트 v6 (resume from epoch 90)

배경:
  - 원본 main_train_eter_v6.py 가 epoch 95 의 ckpt 저장 도중 디스크 풀로 실패 (2026-05-13).
  - 마지막 정상 저장 ckpt: logs/ETER_ViT_R4_brain320_v6/eter_vit_epoch_90.pt
  - 이 스크립트는 epoch 90 시점 weight 로부터 학습을 재개하여 epoch 91~200 을 마저 진행한다.

변경점 (main_train_eter_v6.py 대비):
  - RESUME_CKPT 를 ./logs/ETER_ViT_R4_brain320_v6/eter_vit_epoch_90.pt 로 override
  - START_EPOCH = 90 : 학습 루프 range(START_EPOCH, NUM_EPOCHS) 로 진행
  - scheduler 를 START_EPOCH * steps_per_epoch 만큼 fast-forward
  - best_val_ssim 초기화 = 0.8835 (epoch 85 의 기존 best); baseline 측정값으로 best.pt 갱신하지 않음
  - log.txt 는 append 모드, "RESUME from ep90" 한 줄 기록
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import torch
import torch.nn as nn
import numpy as np
import time
import datetime
import pytz
import wandb
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'configs'))
sys.path.append(os.path.join(current_dir, 'dataloaders'))
sys.path.append(os.path.join(current_dir, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(current_dir, 'tools'))

from myConfig_choh_ETER_model_v6 import *
from u_choh_model_ETER_ViT import choh_ViT
from u_choh_model_ETER_ViT_v5 import choh_Decoder3_ETER_v5
from u_choh_SSIM import SSIM
from dataloader_h5_v5 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader
from check_recon_env import check_env_for_model

NUM_VAL_FILES = None

RESUME_CKPT = './logs/ETER_ViT_R4_brain320_v6/eter_vit_epoch_90.pt'
START_EPOCH = 90
PREV_BEST_VAL_SSIM = 0.8835


def skimage_ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
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


def run_val(model, val_loader, criterion_l1, device):
    model.eval()
    all_ssim, all_psnr, all_nmse, all_l1 = [], [], [], []
    val_bar = tqdm(val_loader, desc='  Val', leave=False, unit='batch')
    with torch.no_grad():
        for sample in val_bar:
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)

            with torch.amp.autocast('cuda'):
                out = model(data_in_img, data_in)

            out_f = out.float()
            ref_f = data_ref.float()

            mse  = torch.mean((out_f - ref_f) ** 2)
            psnr = (20 * torch.log10(ref_f.max() / torch.sqrt(mse.clamp(min=1e-10)))).item()
            nmse = (torch.norm(out_f - ref_f) ** 2 / torch.norm(ref_f) ** 2).item()
            ssim = skimage_ssim_batch(out_f, ref_f)
            l1   = criterion_l1(out_f, ref_f).item()

            all_psnr.append(psnr)
            all_nmse.append(nmse)
            all_ssim.append(ssim)
            all_l1.append(l1)

            val_bar.set_postfix(SSIM=f'{ssim:.4f}', PSNR=f'{psnr:.2f}dB')

    model.train()
    return {
        'ssim': float(np.mean(all_ssim)),
        'psnr': float(np.mean(all_psnr)),
        'nmse': float(np.mean(all_nmse)),
        'l1':   float(np.mean(all_l1)),
    }


def main():
    print('====================================================')
    print(f' [ETER-ViT v6 RESUME] from epoch {START_EPOCH} → {NUM_EPOCHS}')
    print('====================================================')

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필수.")
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
    if not check_env_for_model('eter', 'myConfig_choh_ETER_model_v6', strict=True):
        return

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

    if not os.path.exists(RESUME_CKPT):
        raise FileNotFoundError(f"RESUME_CKPT not found: {RESUME_CKPT}")
    state = torch.load(RESUME_CKPT, map_location=device)
    eter_decoder.load_state_dict(state)
    print(f"\nResumed weights from: {RESUME_CKPT}")

    num_params = sum(p.numel() for p in eter_decoder.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {num_params / 1e6:.1f}M")

    criterion_l1   = nn.L1Loss()
    criterion_ssim_loss = SSIM().to(device)
    optimizer = torch.optim.Adam(
        eter_decoder.parameters(), lr=LEARNING_RATE_ADAM,
        weight_decay=LAMBDA_REGULAR_PER_PIXEL,
    )

    print("\nFastMRI 데이터 파이프라인 연결 중...")
    choh_data_train = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_train', num_files=None,
        augment=TRAIN_AUGMENT, augment_flip_p=TRAIN_AUGMENT_FLIP_P,
    )
    trainloader = DataLoader(
        choh_data_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    print(f"Train Dataloader 준비 완료! ({len(choh_data_train)} 샘플)")

    choh_data_val = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_val', num_files=NUM_VAL_FILES,
        random_mask=False, augment=False,
    )
    val_loader = DataLoader(
        choh_data_val, batch_size=4, shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"Val   Dataloader 준비 완료! ({len(choh_data_val)} 샘플)")

    steps_per_epoch = len(trainloader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    # fast-forward scheduler to the resume point
    skipped_steps = START_EPOCH * steps_per_epoch
    for _ in range(skipped_steps):
        scheduler.step()
    print(f"Scheduler: CosineAnnealingLR T_max={total_steps}, eta_min=1e-6, "
          f"fast-forwarded {skipped_steps} steps (epoch {START_EPOCH})")
    print(f"  current LR after fast-forward = {scheduler.get_last_lr()[0]:.2e}")
    print(f"EarlyStopping: patience={EARLYSTOP_PATIENCE} val checks (val_ssim 기준)")
    print(f"VAL_EVERY_N_EPOCHS = {VAL_EVERY_N_EPOCHS}")

    wandb.init(
        project='ViT-MRI-Recon',
        name=f'ETER_v6_resume_from_ep{START_EPOCH}_BS{BATCH_SIZE}_LR{LEARNING_RATE_ADAM}_EP{NUM_EPOCHS}',
        config={
            'model': 'ETER-ViT-v6-resume',
            'resume_from': RESUME_CKPT,
            'start_epoch': START_EPOCH,
            'prev_best_val_ssim': PREV_BEST_VAL_SSIM,
            'val_metric': 'skimage_ssim',
            'earlystop_metric': 'val_ssim',
            'earlystop_patience': EARLYSTOP_PATIENCE,
            'val_every_n_epochs': VAL_EVERY_N_EPOCHS,
            'dropout': DROPOUT,
            'weight_decay': LAMBDA_REGULAR_PER_PIXEL,
            'augment': TRAIN_AUGMENT,
            'augment_flip_p': TRAIN_AUGMENT_FLIP_P,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE_ADAM,
            'ssim_weight': LAMBDA_SSIM_PER_PIXEL,
            'image_size': IMAGE_SIZE,
            'patch_size': PATCH_SIZE,
            'encoder_hidden': NUM_VIT_ENCODER_HIDDEN,
            'encoder_layers': NUM_VIT_ENCODER_LAYER,
            'encoder_heads': NUM_VIT_ENCODER_HEAD,
            'eter_hori_hidden': NUM_ETER_HORI_HIDDEN,
            'eter_vert_hidden': NUM_ETER_VERT_HIDDEN,
            'decoder_dim': NUM_VIT_DECODER_DIM,
            'decoder_depth': NUM_VIT_DECODER_DEPTH,
            'num_params': num_params,
            'train_samples': len(choh_data_train),
            'val_samples': len(choh_data_val),
            'scheduler': 'CosineAnnealingLR',
            'T_max': total_steps,
            'eta_min': 1e-6,
        },
    )
    wandb.watch(eter_decoder, log='gradients', log_freq=100)

    print(f"\n학습 재개 (epoch {START_EPOCH+1}/{NUM_EPOCHS} 부터)")
    scaler = torch.amp.GradScaler('cuda')
    eter_decoder.train()
    best_val_ssim = PREV_BEST_VAL_SSIM
    best_val = {'ssim': PREV_BEST_VAL_SSIM, 'psnr': None, 'nmse': None, 'l1': None}
    no_improve_val_count = 0
    early_stopped = False
    tic = time.time()
    global_step = skipped_steps
    log_path = os.path.join(PATH_FOLDER, 'log.txt')

    # resume marker — append to existing log.txt
    with open(log_path, 'a') as f:
        f.write(
            f'RESUME (resume from {RESUME_CKPT} at epoch {START_EPOCH}):'
            f' prev_best_val_ssim={PREV_BEST_VAL_SSIM:.4f}\n'
        )

    tqdm.write('Resume baseline val 측정 중 (best.pt 갱신 안 함)...')
    baseline = run_val(eter_decoder, val_loader, criterion_l1, device)
    tqdm.write(
        f'  [Baseline at ep{START_EPOCH}]  SSIM: {baseline["ssim"]:.4f}'
        f'  PSNR: {baseline["psnr"]:.2f}dB  NMSE: {baseline["nmse"]:.4f}'
        f'  L1: {baseline["l1"]:.4f}'
    )
    with open(log_path, 'a') as f:
        f.write(
            f'BASELINE (resume from epoch_90.pt): '
            f'val_ssim={baseline["ssim"]:.4f}  val_psnr={baseline["psnr"]:.2f}'
            f'  val_nmse={baseline["nmse"]:.4f}  val_l1={baseline["l1"]:.4f}\n'
        )
    wandb.log({
        'baseline/ssim': baseline['ssim'],
        'baseline/psnr': baseline['psnr'],
        'baseline/nmse': baseline['nmse'],
        'baseline/l1':   baseline['l1'],
    }, step=global_step)

    epoch_bar = tqdm(range(START_EPOCH, NUM_EPOCHS), desc='전체 진행', unit='epoch')
    for epoch in epoch_bar:
        epoch_loss = epoch_ssim = epoch_psnr = epoch_nmse = epoch_l1 = 0.0
        batch_bar = tqdm(trainloader, desc=f'Epoch {epoch+1:3d}/{NUM_EPOCHS}', leave=False, unit='batch')

        for sample in batch_bar:
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)

            with torch.amp.autocast('cuda'):
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
                out_f = out_fp.detach()
                ref_f = data_ref.float()
                mse_val  = torch.mean((out_f - ref_f) ** 2)
                psnr_val = (20 * torch.log10(ref_f.max() / torch.sqrt(mse_val.clamp(min=1e-10)))).item()
                nmse_val = (torch.norm(out_f - ref_f) ** 2 / torch.norm(ref_f) ** 2).item()
                ssim_val_train = 1 - loss_ssim.item()

            global_step += 1
            wandb.log({
                'train/loss': loss.item(),
                'train/loss_l1': loss_l1.item(),
                'train/loss_ssim': loss_ssim.item(),
                'train/ssim_custom': ssim_val_train,
                'train/psnr': psnr_val,
                'train/nmse': nmse_val,
                'train/lr': scheduler.get_last_lr()[0],
            }, step=global_step)

            epoch_loss += loss.item()
            epoch_ssim += ssim_val_train
            epoch_psnr += psnr_val
            epoch_nmse += nmse_val
            epoch_l1   += loss_l1.item()
            batch_bar.set_postfix(
                Loss=f'{loss.item():.4f}', PSNR=f'{psnr_val:.2f}dB',
                NMSE=f'{nmse_val:.4f}', SSIM_c=f'{ssim_val_train:.4f}',
                LR=f'{scheduler.get_last_lr()[0]:.2e}',
            )

        n_batches = len(trainloader)
        avg_loss       = epoch_loss / n_batches
        avg_train_ssim = epoch_ssim / n_batches
        avg_train_psnr = epoch_psnr / n_batches
        avg_train_nmse = epoch_nmse / n_batches
        avg_train_l1   = epoch_l1   / n_batches

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
            tqdm.write(f'  [Val ep{epoch+1}] running...')
            val_metrics = run_val(eter_decoder, val_loader, criterion_l1, device)
            tqdm.write(
                f'  [Val]  SSIM: {val_metrics["ssim"]:.4f}'
                f'  PSNR: {val_metrics["psnr"]:.2f}dB'
                f'  NMSE: {val_metrics["nmse"]:.4f}'
                f'  L1: {val_metrics["l1"]:.4f}'
            )

            wandb.log({
                'val/ssim': val_metrics['ssim'],
                'val/psnr': val_metrics['psnr'],
                'val/nmse': val_metrics['nmse'],
                'val/l1': val_metrics['l1'],
            }, step=global_step)

            epoch_bar.set_postfix(
                train_ssim=f'{avg_train_ssim:.4f}',
                val_ssim=f'{val_metrics["ssim"]:.4f}',
                val_psnr=f'{val_metrics["psnr"]:.2f}dB',
            )

            with open(log_path, 'a') as f:
                f.write(
                    f'Epoch {epoch+1}/{NUM_EPOCHS}'
                    f'  train_loss={avg_loss:.4f}'
                    f'  train_ssim_custom={avg_train_ssim:.4f}'
                    f'  val_ssim={val_metrics["ssim"]:.4f}'
                    f'  val_psnr={val_metrics["psnr"]:.2f}'
                    f'  val_nmse={val_metrics["nmse"]:.4f}'
                    f'  val_l1={val_metrics["l1"]:.4f}\n'
                )

            if val_metrics['ssim'] > best_val_ssim:
                best_val_ssim = val_metrics['ssim']
                best_val = dict(val_metrics)
                best_ckpt_path = os.path.join(PATH_FOLDER, 'eter_vit_best.pt')
                torch.save(eter_decoder.state_dict(), best_ckpt_path)
                tqdm.write(f'  [Best Ckpt] val_ssim {best_val_ssim:.4f} → {best_ckpt_path}')
                no_improve_val_count = 0
            else:
                no_improve_val_count += 1
                tqdm.write(
                    f'  [No improvement] val_ssim {val_metrics["ssim"]:.4f}'
                    f' < best {best_val_ssim:.4f}'
                    f'  (no-improve {no_improve_val_count}/{EARLYSTOP_PATIENCE})'
                )

            wandb.log({'val/no_improve_count': no_improve_val_count}, step=global_step)

            if no_improve_val_count >= EARLYSTOP_PATIENCE:
                tqdm.write(
                    f'  [EarlyStop] {no_improve_val_count} consecutive val checks'
                    f' without val_ssim improvement → 학습 종료 (epoch {epoch+1})'
                )
                with open(log_path, 'a') as f:
                    f.write(f'EARLYSTOP at epoch {epoch+1} (no_improve_val_count={no_improve_val_count})\n')
                early_stopped = True
                break
        else:
            epoch_bar.set_postfix(train_ssim=f'{avg_train_ssim:.4f}', train_loss=f'{avg_loss:.4f}')
            with open(log_path, 'a') as f:
                f.write(
                    f'Epoch {epoch+1}/{NUM_EPOCHS}'
                    f'  train_loss={avg_loss:.4f}'
                    f'  train_ssim_custom={avg_train_ssim:.4f}\n'
                )

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(PATH_FOLDER, f'eter_vit_epoch_{epoch+1}.pt')
            torch.save(eter_decoder.state_dict(), ckpt_path)
            tqdm.write(f'  Checkpoint 저장: {ckpt_path}')

    toc = time.time()
    print(f'\n학습 완료 (early_stopped={early_stopped})  소요 시간: {toc - tic:.0f}초')
    print(f'Best Val SSIM: {best_val_ssim:.4f}')
    if best_val['ssim'] is not None:
        print(f'  Best Val → SSIM: {best_val["ssim"]:.4f}')
        if best_val.get('psnr') is not None:
            print(f'                PSNR: {best_val["psnr"]:.2f}dB'
                  f'  NMSE: {best_val["nmse"]:.4f}  L1: {best_val["l1"]:.4f}')
    print(f'Best Checkpoint: {os.path.join(PATH_FOLDER, "eter_vit_best.pt")}')
    wandb.finish()


if __name__ == '__main__':
    main()
