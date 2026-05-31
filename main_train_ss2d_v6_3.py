"""
SS2D-ViT 학습 스크립트 v6_2
  - Config:     myConfig_choh_SS2D_model_v6_3
  - Dataloader: dataloader_h5_v5 (v6 와 동일)
  - Model:      u_choh_model_SS2D_ViT_v4 (v6 와 동일)
  - 변경점 (v6 대비):
      * Loss 에 gradient(edge) loss term 추가 (mean-prediction blurring 처벌)
        loss = L1 + λ_ssim·(1-SSIM) + λ_grad·grad_l1
      * v6 best ckpt 부터 resume (state_dict 만, optimizer/scheduler 는 새로 시작)
      * NUM_EPOCHS 50 (fine-tune 짧게), LR 5e-5 → 5e-7 cosine
      * EARLYSTOP_PATIENCE 5
      * 나머지 (model / regularization / dataloader / DC block) 는 v6 와 동일
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
sys.path.append(os.path.join(current_dir, 'models', 'mamba_eternet'))
sys.path.append(os.path.join(current_dir, 'tools'))

from myConfig_choh_SS2D_model_v6_3 import *
from u_choh_model_ETER_ViT import choh_ViT
from u_choh_model_SS2D_ViT_v4 import choh_Decoder_SS2D_ViT
from u_choh_SSIM import SSIM
from dataloader_h5_v5 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader
from check_recon_env import check_env_for_model

NUM_VAL_FILES = None


def gradient_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Finite-difference gradient L1 loss on H/W axes.

    pred, target: (B, C, H, W). Returns scalar.
    """
    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_gt   = target[..., :, 1:] - target[..., :, :-1]
    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_gt   = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_pred, dx_gt) + F.l1_loss(dy_pred, dy_gt)


def skimage_ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """배치 평균 skimage SSIM (data_range = target.max() - target.min() 슬라이스별)."""
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
            mask_in     = sample['mask'].float().to(device)
            sens_in     = sample['sens'].float().to(device)

            with torch.amp.autocast("cuda"):
                out = model(data_in_img, data_in, mask_in, sens_in)

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
    print(' [SS2D-ViT v6_3] v6 best resume + gradient(edge) loss fine-tune')
    print('====================================================')

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필수.")
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
    if not check_env_for_model('ss2d', 'myConfig_choh_SS2D_model_v6_3', strict=True):
        return

    # 1. 인코더
    vit_choh = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1
    ).to(device)

    # 2. 디코더 (v6 동일)
    ss2d_decoder = choh_Decoder_SS2D_ViT(
        encoder=vit_choh,
        ss2d_d_inner=NUM_SS2D_D_INNER,
        ss2d_d_state=NUM_SS2D_D_STATE,
        ss2d_out_ch=NUM_SS2D_OUT_CH,
        decoder_dim=NUM_VIT_DECODER_DIM,
        decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD,
        decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=DROPOUT,
        dc_k_scale_ratio=DC_K_SCALE_RATIO,
        dc_init_alpha=DC_INIT_ALPHA,
    ).to(device)

    # 3. resume (v6 best → v6_2 시작점)
    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        state = torch.load(RESUME_CKPT, map_location=device)
        ss2d_decoder.load_state_dict(state)
        print(f"\nResumed weights from: {RESUME_CKPT}")
    else:
        print(f"\nWARNING: RESUME_CKPT not found ({RESUME_CKPT}) — 처음부터 학습")

    num_params = sum(p.numel() for p in ss2d_decoder.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {num_params / 1e6:.1f}M")

    # 4. 옵티마이저, 손실 (v6 + gradient loss)
    criterion_l1   = nn.L1Loss()
    criterion_ssim_loss = SSIM().to(device)
    optimizer = torch.optim.Adam(
        ss2d_decoder.parameters(),
        lr=LEARNING_RATE_ADAM,
        weight_decay=LAMBDA_REGULAR_PER_PIXEL,
    )

    # 5. 데이터셋
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
        optimizer, T_max=total_steps, eta_min=5e-7
    )
    print(f"Scheduler: CosineAnnealingLR T_max={total_steps} (={NUM_EPOCHS} epochs), eta_min=5e-7")
    print(f"EarlyStopping: patience={EARLYSTOP_PATIENCE} val checks (val_ssim 기준)")
    print(f"VAL_EVERY_N_EPOCHS = {VAL_EVERY_N_EPOCHS}")
    print(f"Loss: L1 + {LAMBDA_SSIM_PER_PIXEL}·(1-SSIM) + {LAMBDA_GRAD_PER_PIXEL}·grad_L1")

    # 6. wandb
    wandb.init(
        project='ViT-MRI-Recon',
        name=f'SS2D_v6_3_sharp_ablation_BS{BATCH_SIZE}_LR{LEARNING_RATE_ADAM}_EP{NUM_EPOCHS}',
        config={
            'model': 'SS2D-ViT-v6_2',
            'resume_from': RESUME_CKPT,
            'val_metric': 'skimage_ssim',
            'earlystop_metric': 'val_ssim',
            'earlystop_patience': EARLYSTOP_PATIENCE,
            'val_every_n_epochs': VAL_EVERY_N_EPOCHS,
            'dropout': DROPOUT,
            'weight_decay': LAMBDA_REGULAR_PER_PIXEL,
            'augment': TRAIN_AUGMENT,
            'augment_flip_p': TRAIN_AUGMENT_FLIP_P,
            'dc_k_scale_ratio': DC_K_SCALE_RATIO,
            'dc_init_alpha': DC_INIT_ALPHA,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE_ADAM,
            'ssim_weight': LAMBDA_SSIM_PER_PIXEL,
            'grad_weight': LAMBDA_GRAD_PER_PIXEL,
            'image_size': IMAGE_SIZE,
            'patch_size': PATCH_SIZE,
            'encoder_hidden': NUM_VIT_ENCODER_HIDDEN,
            'encoder_layers': NUM_VIT_ENCODER_LAYER,
            'encoder_heads': NUM_VIT_ENCODER_HEAD,
            'ss2d_d_inner': NUM_SS2D_D_INNER,
            'ss2d_d_state': NUM_SS2D_D_STATE,
            'decoder_dim': NUM_VIT_DECODER_DIM,
            'decoder_depth': NUM_VIT_DECODER_DEPTH,
            'num_params': num_params,
            'train_samples': len(choh_data_train),
            'val_samples': len(choh_data_val),
            'scheduler': 'CosineAnnealingLR',
            'T_max': total_steps,
            'eta_min': 5e-7,
        },
    )
    wandb.watch(ss2d_decoder, log='gradients', log_freq=100)

    # 7. 학습 루프
    print(f"\n학습 시작 (총 {NUM_EPOCHS} 에폭, EarlyStop 가능)")
    scaler = torch.amp.GradScaler('cuda')
    ss2d_decoder.train()
    best_val_ssim = -1.0
    best_val = {'ssim': None, 'psnr': None, 'nmse': None, 'l1': None}
    no_improve_val_count = 0
    early_stopped = False
    tic = time.time()
    global_step = 0
    log_path = os.path.join(PATH_FOLDER, 'log.txt')

    # 7a. baseline val (resume 시작점)
    tqdm.write('Resume baseline val 측정 중...')
    baseline = run_val(ss2d_decoder, val_loader, criterion_l1, device)
    best_val_ssim = baseline['ssim']
    best_val = dict(baseline)
    tqdm.write(
        f'  [Baseline]  SSIM: {baseline["ssim"]:.4f}'
        f'  PSNR: {baseline["psnr"]:.2f}dB  NMSE: {baseline["nmse"]:.4f}'
        f'  L1: {baseline["l1"]:.4f}'
    )
    with open(log_path, 'a') as f:
        f.write(
            f'BASELINE (resume from {RESUME_CKPT}): '
            f'val_ssim={baseline["ssim"]:.4f}  val_psnr={baseline["psnr"]:.2f}'
            f'  val_nmse={baseline["nmse"]:.4f}  val_l1={baseline["l1"]:.4f}\n'
        )
    torch.save(ss2d_decoder.state_dict(), os.path.join(PATH_FOLDER, 'ss2d_vit_best.pt'))

    epoch_bar = tqdm(range(NUM_EPOCHS), desc='전체 진행', unit='epoch')
    for epoch in epoch_bar:
        epoch_loss = epoch_ssim = epoch_psnr = epoch_nmse = epoch_l1 = epoch_grad = 0.0
        batch_bar = tqdm(trainloader, desc=f'Epoch {epoch+1:3d}/{NUM_EPOCHS}', leave=False, unit='batch')

        for sample in batch_bar:
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            mask_in     = sample['mask'].float().to(device)
            sens_in     = sample['sens'].float().to(device)

            with torch.amp.autocast("cuda"):
                out = ss2d_decoder(data_in_img, data_in, mask_in, sens_in)

            out_fp    = out.float()
            loss_l1   = criterion_l1(out_fp, data_ref)
            loss_ssim = 1 - criterion_ssim_loss(out_fp, data_ref)
            loss_grad = gradient_l1_loss(out_fp, data_ref)
            loss = (
                loss_l1
                + LAMBDA_SSIM_PER_PIXEL * loss_ssim
                + LAMBDA_GRAD_PER_PIXEL * loss_grad
            )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ss2d_decoder.parameters(), max_norm=1.0)
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
                'train/loss_grad': loss_grad.item(),
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
            epoch_grad += loss_grad.item()
            batch_bar.set_postfix(
                Loss=f'{loss.item():.4f}', PSNR=f'{psnr_val:.2f}dB',
                NMSE=f'{nmse_val:.4f}', SSIM_c=f'{ssim_val_train:.4f}',
                Grad=f'{loss_grad.item():.2e}',
                LR=f'{scheduler.get_last_lr()[0]:.2e}',
            )

        n_batches = len(trainloader)
        avg_loss       = epoch_loss / n_batches
        avg_train_ssim = epoch_ssim / n_batches
        avg_train_psnr = epoch_psnr / n_batches
        avg_train_nmse = epoch_nmse / n_batches
        avg_train_l1   = epoch_l1   / n_batches
        avg_train_grad = epoch_grad / n_batches

        wandb.log({
            'epoch': epoch + 1,
            'epoch/train_loss': avg_loss,
            'epoch/train_ssim_custom': avg_train_ssim,
            'epoch/train_psnr': avg_train_psnr,
            'epoch/train_nmse': avg_train_nmse,
            'epoch/train_l1': avg_train_l1,
            'epoch/train_grad': avg_train_grad,
        }, step=global_step)

        do_val = (epoch + 1) % VAL_EVERY_N_EPOCHS == 0

        if do_val:
            tqdm.write(f'  [Val ep{epoch+1}] running...')
            val_metrics = run_val(ss2d_decoder, val_loader, criterion_l1, device)
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
                    f'  train_grad={avg_train_grad:.2e}'
                    f'  train_ssim_custom={avg_train_ssim:.4f}'
                    f'  val_ssim={val_metrics["ssim"]:.4f}'
                    f'  val_psnr={val_metrics["psnr"]:.2f}'
                    f'  val_nmse={val_metrics["nmse"]:.4f}'
                    f'  val_l1={val_metrics["l1"]:.4f}\n'
                )

            if val_metrics['ssim'] > best_val_ssim:
                best_val_ssim = val_metrics['ssim']
                best_val = dict(val_metrics)
                best_ckpt_path = os.path.join(PATH_FOLDER, 'ss2d_vit_best.pt')
                torch.save(ss2d_decoder.state_dict(), best_ckpt_path)
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
                    f'  train_grad={avg_train_grad:.2e}'
                    f'  train_ssim_custom={avg_train_ssim:.4f}\n'
                )

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(PATH_FOLDER, f'ss2d_vit_epoch_{epoch+1}.pt')
            torch.save(ss2d_decoder.state_dict(), ckpt_path)
            tqdm.write(f'  Checkpoint 저장: {ckpt_path}')

    toc = time.time()
    print(f'\n학습 완료 (early_stopped={early_stopped})  소요 시간: {toc - tic:.0f}초')
    print(f'Best Val SSIM: {best_val_ssim:.4f}')
    if best_val['ssim'] is not None:
        print(f'  Best Val → SSIM: {best_val["ssim"]:.4f}  PSNR: {best_val["psnr"]:.2f}dB'
              f'  NMSE: {best_val["nmse"]:.4f}  L1: {best_val["l1"]:.4f}')
    print(f'Best Checkpoint: {os.path.join(PATH_FOLDER, "ss2d_vit_best.pt")}')
    wandb.finish()


if __name__ == '__main__':
    main()
