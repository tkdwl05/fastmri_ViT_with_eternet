"""
ETER-ViT 학습 스크립트 (SS2D-ViT와 비교용, 동일 조건)
  - 인코더: choh_ViT
  - 디코더: choh_Decoder3_ETER_skip_up_tail (원본 Bidirectional GRU)
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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'configs'))
sys.path.append(os.path.join(current_dir, 'dataloaders'))
sys.path.append(os.path.join(current_dir, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(current_dir, 'tools'))

from myConfig_choh_ETER_model import *
from u_choh_model_ETER_ViT import choh_ViT, choh_Decoder3_ETER_skip_up_tail
from u_choh_SSIM import SSIM
from dataloader_h5 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader
from check_recon_env import check_env_for_model

NUM_VAL_FILES = None  # val 전체 사용


def run_val(model, val_loader, criterion_l1, criterion_ssim, device):
    """val 데이터 전체를 순회하며 평균 SSIM, PSNR, NMSE, L1을 반환."""
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

            mse = torch.mean((out_f - ref_f) ** 2)
            psnr = (20 * torch.log10(ref_f.max() / torch.sqrt(mse))).item()
            nmse = (torch.norm(out_f - ref_f) ** 2 / torch.norm(ref_f) ** 2).item()
            ssim = criterion_ssim(out_f, ref_f).item()
            l1   = criterion_l1(out_f, ref_f).item()

            all_psnr.append(psnr)
            all_nmse.append(nmse)
            all_ssim.append(ssim)
            all_l1.append(l1)

            val_bar.set_postfix(
                SSIM=f'{ssim:.4f}',
                PSNR=f'{psnr:.2f}dB',
            )

    model.train()
    return {
        'ssim': np.mean(all_ssim),
        'psnr': np.mean(all_psnr),
        'nmse': np.mean(all_nmse),
        'l1':   np.mean(all_l1),
    }


def main():
    print('====================================================')
    print(' [ETER-ViT] ViT + Bidirectional GRU FastMRI Training')
    print('====================================================')

    if not torch.cuda.is_available():
        raise RuntimeError("이 실험은 CUDA GPU에서만 동작하도록 설계되었습니다. torch.cuda.is_available()가 False입니다.")
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
    if not check_env_for_model('eter', 'myConfig_choh_ETER_model', strict=True):
        return

    # 1. ViT 인코더 (fastMRI brain 320×320)
    vit_choh = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1
    ).to(device)

    # 2. ETER-Net 디코더 (원본 GRU)
    eter_decoder = choh_Decoder3_ETER_skip_up_tail(
        encoder=vit_choh,
        eter_n_hori_hidden=NUM_ETER_HORI_HIDDEN,
        eter_n_vert_hidden=NUM_ETER_VERT_HIDDEN,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT
    ).to(device)

    num_params = sum(p.numel() for p in eter_decoder.parameters() if p.requires_grad)
    print(f"\n모델 파라미터 수: {num_params / 1e6:.1f}M")
    print("모델 메모리 적재 완료")

    # 3. 옵티마이저, 손실함수 (스케줄러는 dataloader 생성 후)
    criterion_l1   = nn.L1Loss()
    criterion_ssim = SSIM().to(device)
    optimizer  = torch.optim.Adam(
        eter_decoder.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=LAMBDA_REGULAR_PER_PIXEL
    )

    # 4. 데이터셋 (SS2D와 동일 조건: 50 files)
    try:
        print("\nFastMRI 데이터 파이프라인 연결 중...")
        choh_data_train = FastMRI_H5_Dataloader('./fastMRI_data/multicoil_train', num_files=None)
        trainloader = DataLoader(choh_data_train, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        print(f"Train Dataloader 준비 완료! ({len(choh_data_train)} 샘플)")

        choh_data_val = FastMRI_H5_Dataloader('./fastMRI_data/multicoil_val', num_files=NUM_VAL_FILES)
        val_loader = DataLoader(choh_data_val, batch_size=4, shuffle=False,
                               num_workers=2, pin_memory=True)
        print(f"Val   Dataloader 준비 완료! ({len(choh_data_val)} 샘플)")
    except Exception as e:
        print(f"데이터셋 오류: {e}")
        return

    # CosineAnnealingWarmRestarts: 첫 cycle = 1 epoch, 그 다음 2, 4, 8, ... epoch
    steps_per_epoch = len(trainloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch, T_mult=2, eta_min=1e-6
    )
    print(f"Scheduler: CosineAnnealingWarmRestarts T_0={steps_per_epoch} (=1 epoch), T_mult=2, eta_min=1e-6")

    # 5. wandb 초기화
    wandb.init(
        project='ViT-MRI-Recon',
        name=f'ETER_BS{BATCH_SIZE}_LR{LEARNING_RATE_ADAM}_EP{NUM_EPOCHS}',
        config={
            'model': 'ETER-ViT',
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE_ADAM,
            'weight_decay': LAMBDA_REGULAR_PER_PIXEL,
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
            'scheduler': 'CosineAnnealingWarmRestarts',
            'T_0': steps_per_epoch,
            'T_mult': 2,
        },
    )
    wandb.watch(eter_decoder, log='gradients', log_freq=100)

    # 6. 학습 루프
    print(f"\n학습 시작 (총 {NUM_EPOCHS} 에폭)")
    scaler = torch.amp.GradScaler('cuda')
    eter_decoder.train()
    best_train_ssim = -1.0
    # val 각 지표별 best (composite score 계산용)
    best_val = {'ssim': None, 'psnr': None, 'nmse': None, 'l1': None}
    best_composite = -1.0
    tic = time.time()
    global_step = 0

    epoch_bar = tqdm(range(NUM_EPOCHS), desc='전체 진행', unit='epoch')
    for epoch in epoch_bar:
        epoch_loss = 0.0
        epoch_ssim = 0.0
        epoch_psnr = 0.0
        epoch_nmse = 0.0
        epoch_l1   = 0.0
        batch_bar = tqdm(trainloader, desc=f'Epoch {epoch+1:3d}/{NUM_EPOCHS}', leave=False, unit='batch')

        for sample in batch_bar:
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)

            with torch.amp.autocast('cuda'):
                out = eter_decoder(data_in_img, data_in)

            # 손실은 fp32 에서 계산 (SSIM 내부 제곱이 fp16 에서 overflow → NaN)
            out_fp = out.float()
            loss_l1   = criterion_l1(out_fp, data_ref)
            loss_ssim = 1 - criterion_ssim(out_fp, data_ref)
            loss = loss_l1 + LAMBDA_SSIM_PER_PIXEL * loss_ssim

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            with torch.no_grad():
                out_f = out_fp.detach()
                ref_f = data_ref.float()
                mse_val  = torch.mean((out_f - ref_f) ** 2)
                psnr_val = (20 * torch.log10(ref_f.max() / torch.sqrt(mse_val))).item()
                nmse_val = (torch.norm(out_f - ref_f) ** 2 / torch.norm(ref_f) ** 2).item()
                ssim_val = 1 - loss_ssim.item()

            global_step += 1
            wandb.log({
                'train/loss': loss.item(),
                'train/loss_l1': loss_l1.item(),
                'train/loss_ssim': loss_ssim.item(),
                'train/ssim': ssim_val,
                'train/psnr': psnr_val,
                'train/nmse': nmse_val,
                'train/lr': scheduler.get_last_lr()[0],
            }, step=global_step)

            epoch_loss += loss.item()
            epoch_ssim += ssim_val
            epoch_psnr += psnr_val
            epoch_nmse += nmse_val
            epoch_l1   += loss_l1.item()
            batch_bar.set_postfix(
                Loss=f'{loss.item():.4f}',
                PSNR=f'{psnr_val:.2f}dB',
                NMSE=f'{nmse_val:.4f}',
                SSIM=f'{ssim_val:.4f}',
                LR=f'{scheduler.get_last_lr()[0]:.2e}'
            )

        n_batches      = len(trainloader)
        avg_loss       = epoch_loss / n_batches
        avg_train_ssim = epoch_ssim / n_batches
        avg_train_psnr = epoch_psnr / n_batches
        avg_train_nmse = epoch_nmse / n_batches
        avg_train_l1   = epoch_l1   / n_batches

        log_path = os.path.join(PATH_FOLDER, 'log.txt')

        # train이 새로운 best를 달성했을 때만 val 실행
        if avg_train_ssim > best_train_ssim:
            best_train_ssim = avg_train_ssim
            tqdm.write(f'  [Train Best] Epoch {epoch+1}  Train SSIM {avg_train_ssim:.4f} → val 실행 중...')

            val_metrics = run_val(eter_decoder, val_loader, criterion_l1, criterion_ssim, device)
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
                    f'  train_ssim={avg_train_ssim:.4f} [BEST]'
                    f'  val_ssim={val_metrics["ssim"]:.4f}'
                    f'  val_psnr={val_metrics["psnr"]:.2f}'
                    f'  val_nmse={val_metrics["nmse"]:.4f}'
                    f'  val_l1={val_metrics["l1"]:.4f}\n'
                )

            # 첫 val이면 best 초기화 후 무조건 저장
            if best_val['ssim'] is None:
                best_val = {k: val_metrics[k] for k in best_val}
                composite = 1.0
            else:
                # 각 지표를 best 대비 비율로 정규화 (모두 높을수록 좋게 변환)
                # SSIM, PSNR: 높을수록 좋음 → 현재/best
                # NMSE, L1:   낮을수록 좋음 → best/현재
                composite = (
                    val_metrics['ssim'] / best_val['ssim'] +
                    best_val['nmse']   / val_metrics['nmse'] +
                    val_metrics['psnr'] / best_val['psnr'] +
                    best_val['l1']     / val_metrics['l1']
                ) / 4.0

            tqdm.write(
                f'  [Composite] {composite:.4f}'
                f'  (SSIM {val_metrics["ssim"]:.4f} vs best {best_val["ssim"] or 0:.4f}'
                f'  | NMSE {val_metrics["nmse"]:.4f} vs best {best_val["nmse"] or 0:.4f}'
                f'  | PSNR {val_metrics["psnr"]:.2f} vs best {best_val["psnr"] or 0:.2f}'
                f'  | L1 {val_metrics["l1"]:.4f} vs best {best_val["l1"] or 0:.4f})'
            )

            if composite > best_composite:
                best_composite = composite
                best_val = {k: val_metrics[k] for k in best_val}
                best_ckpt_path = os.path.join(PATH_FOLDER, 'eter_vit_best.pt')
                torch.save(eter_decoder.state_dict(), best_ckpt_path)
                tqdm.write(f'  [Best Ckpt] Composite {best_composite:.4f} → {best_ckpt_path}')
            else:
                tqdm.write(f'  [Overfit?] Train best지만 Composite {composite:.4f} < best {best_composite:.4f}')
        else:
            epoch_bar.set_postfix(train_ssim=f'{avg_train_ssim:.4f}', train_loss=f'{avg_loss:.4f}')

        # epoch 단위 요약 로깅
        wandb.log({
            'epoch': epoch + 1,
            'epoch/train_loss': avg_loss,
            'epoch/train_ssim': avg_train_ssim,
            'epoch/train_psnr': avg_train_psnr,
            'epoch/train_nmse': avg_train_nmse,
            'epoch/train_l1': avg_train_l1,
        }, step=global_step)

        if avg_train_ssim <= best_train_ssim:
            with open(log_path, 'a') as f:
                f.write(
                    f'Epoch {epoch+1}/{NUM_EPOCHS}'
                    f'  train_loss={avg_loss:.4f}'
                    f'  train_ssim={avg_train_ssim:.4f}\n'
                )

        # 정기 체크포인트 (5 에폭마다)
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(PATH_FOLDER, f'eter_vit_epoch_{epoch+1}.pt')
            torch.save(eter_decoder.state_dict(), ckpt_path)
            tqdm.write(f'  Checkpoint 저장: {ckpt_path}')

    toc = time.time()
    print(f'\n학습 완료! 소요 시간: {toc - tic:.0f}초')
    print(f'Best Train SSIM: {best_train_ssim:.4f}  |  Best Composite: {best_composite:.4f}')
    print(f'  Best Val → SSIM: {best_val["ssim"]:.4f}  PSNR: {best_val["psnr"]:.2f}dB  NMSE: {best_val["nmse"]:.4f}  L1: {best_val["l1"]:.4f}')
    print(f'Best Checkpoint: {os.path.join(PATH_FOLDER, "eter_vit_best.pt")}')
    wandb.finish()


if __name__ == '__main__':
    main()
