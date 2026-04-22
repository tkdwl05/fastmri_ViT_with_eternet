"""
SS2D-ViT 학습 스크립트 v4
  - 인코더: choh_ViT (v3 동일)
  - 디코더: u_choh_model_SS2D_ViT_v4.choh_Decoder_SS2D_ViT (dropout + complex + DC block)
  - 데이터로더: dataloader_h5_v4 (mask + sens 추가 반환)
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
sys.path.append(os.path.join(current_dir, 'models', 'mamba_eternet'))
sys.path.append(os.path.join(current_dir, 'tools'))

from myConfig_choh_SS2D_model_v4 import *
from u_choh_model_ETER_ViT import choh_ViT
from u_choh_model_SS2D_ViT_v4 import choh_Decoder_SS2D_ViT
from u_choh_SSIM import SSIM
from dataloader_h5_v4 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader
from check_recon_env import check_env_for_model

NUM_VAL_FILES = None  # val 전체 사용
VAL_EVERY_N_EPOCHS = 10  # train best가 아니더라도 N 에폭마다 val 실행


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
            mask_in     = sample['mask'].float().to(device)
            sens_in     = sample['sens'].float().to(device)

            with torch.amp.autocast("cuda"):
                out = model(data_in_img, data_in, mask_in, sens_in)

            out_f = out.float()
            ref_f = data_ref.float()

            mse = torch.mean((out_f - ref_f) ** 2)
            psnr = (20 * torch.log10(ref_f.max() / torch.sqrt(mse.clamp(min=1e-10)))).item()
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
    print(' [SS2D-ViT] ViT + SS2D(VMamba) FastMRI Training')
    print('====================================================')

    if not torch.cuda.is_available():
        raise RuntimeError("이 실험은 CUDA GPU에서만 동작하도록 설계되었습니다. torch.cuda.is_available()가 False입니다.")
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
    if not check_env_for_model('ss2d', 'myConfig_choh_SS2D_model_v4', strict=True):
        return

    # 1. ViT 인코더 (fastMRI brain 320×320)
    vit_choh = choh_ViT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN,
        depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD,
        mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)

    # 2. SS2D 디코더 v4 (dropout + complex 출력 + DC block)
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

    # 파라미터 수 출력
    num_params = sum(p.numel() for p in ss2d_decoder.parameters() if p.requires_grad)
    print(f"\n모델 파라미터 수: {num_params / 1e6:.1f}M")
    print("모델 메모리 적재 완료")

    # 3. 옵티마이저, 손실함수 (스케줄러는 dataloader 생성 후)
    criterion_l1   = nn.L1Loss()
    criterion_ssim = SSIM().to(device)
    optimizer  = torch.optim.Adam(
        ss2d_decoder.parameters(),
        lr=LEARNING_RATE_ADAM,
        weight_decay=LAMBDA_REGULAR_PER_PIXEL
    )

    # 4. 데이터셋
    try:
        print("\nFastMRI 데이터 파이프라인 연결 중...")
        choh_data_train = FastMRI_H5_Dataloader('./fastMRI_data/multicoil_train', num_files=None)
        trainloader = DataLoader(choh_data_train, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        print(f"Train Dataloader 준비 완료! ({len(choh_data_train)} 샘플)")

        choh_data_val = FastMRI_H5_Dataloader('./fastMRI_data/multicoil_val', num_files=NUM_VAL_FILES, random_mask=False)
        val_loader = DataLoader(choh_data_val, batch_size=4, shuffle=False,
                               num_workers=2, pin_memory=True)
        print(f"Val   Dataloader 준비 완료! ({len(choh_data_val)} 샘플)")
    except Exception as e:
        print(f"데이터셋 오류: {e}")
        return

    steps_per_epoch = len(trainloader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    print(f"Scheduler: CosineAnnealingLR T_max={total_steps} (={NUM_EPOCHS} epochs), eta_min=1e-6")

    # 5. wandb 초기화
    wandb.init(
        project='ViT-MRI-Recon',
        name=f'SS2D_v4_BS{BATCH_SIZE}_LR{LEARNING_RATE_ADAM}_EP{NUM_EPOCHS}',
        config={
            'model': 'SS2D-ViT-v4',
            'dropout': DROPOUT,
            'dc_k_scale_ratio': DC_K_SCALE_RATIO,
            'dc_init_alpha': DC_INIT_ALPHA,
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
            'ss2d_d_inner': NUM_SS2D_D_INNER,
            'ss2d_d_state': NUM_SS2D_D_STATE,
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
    wandb.watch(ss2d_decoder, log='gradients', log_freq=100)

    # 6. 학습 루프
    print(f"\n학습 시작 (총 {NUM_EPOCHS} 에폭)")
    scaler = torch.amp.GradScaler('cuda')
    ss2d_decoder.train()
    best_train_ssim = -1.0
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
            mask_in     = sample['mask'].float().to(device)
            sens_in     = sample['sens'].float().to(device)

            with torch.amp.autocast("cuda"):
                out = ss2d_decoder(data_in_img, data_in, mask_in, sens_in)

            # 손실은 fp32 에서 계산 (SSIM 내부 제곱이 fp16 에서 overflow → NaN)
            out_fp = out.float()
            loss_l1   = criterion_l1(out_fp, data_ref)
            loss_ssim = 1 - criterion_ssim(out_fp, data_ref)
            loss = loss_l1 + LAMBDA_SSIM_PER_PIXEL * loss_ssim

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ss2d_decoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ss2d_decoder.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            with torch.no_grad():
                out_f = out_fp.detach()
                ref_f = data_ref.float()
                mse_val  = torch.mean((out_f - ref_f) ** 2)
                psnr_val = (20 * torch.log10(ref_f.max() / torch.sqrt(mse_val.clamp(min=1e-10)))).item()
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

        # epoch 단위 요약 로깅
        wandb.log({
            'epoch': epoch + 1,
            'epoch/train_loss': avg_loss,
            'epoch/train_ssim': avg_train_ssim,
            'epoch/train_psnr': avg_train_psnr,
            'epoch/train_nmse': avg_train_nmse,
            'epoch/train_l1': avg_train_l1,
        }, step=global_step)

        # val 실행 조건: train best 갱신 또는 N 에폭마다 주기적 실행
        is_train_best = avg_train_ssim > best_train_ssim
        is_periodic_val = (epoch + 1) % VAL_EVERY_N_EPOCHS == 0

        if is_train_best:
            best_train_ssim = avg_train_ssim

        if is_train_best or is_periodic_val:
            reason = '[Train Best]' if is_train_best else f'[Periodic {VAL_EVERY_N_EPOCHS}ep]'
            tqdm.write(f'  {reason} Epoch {epoch+1}  Train SSIM {avg_train_ssim:.4f} → val 실행 중...')

            val_metrics = run_val(ss2d_decoder, val_loader, criterion_l1, criterion_ssim, device)
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
                    f'  train_ssim={avg_train_ssim:.4f}{" [BEST]" if is_train_best else ""}'
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
                best_ckpt_path = os.path.join(PATH_FOLDER, 'ss2d_vit_best.pt')
                torch.save(ss2d_decoder.state_dict(), best_ckpt_path)
                tqdm.write(f'  [Best Ckpt] Composite {best_composite:.4f} → {best_ckpt_path}')
            else:
                tqdm.write(f'  [No improvement] Composite {composite:.4f} < best {best_composite:.4f}')
        else:
            epoch_bar.set_postfix(train_ssim=f'{avg_train_ssim:.4f}', train_loss=f'{avg_loss:.4f}')
            with open(log_path, 'a') as f:
                f.write(
                    f'Epoch {epoch+1}/{NUM_EPOCHS}'
                    f'  train_loss={avg_loss:.4f}'
                    f'  train_ssim={avg_train_ssim:.4f}\n'
                )

        # 정기 체크포인트 (5 에폭마다)
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(PATH_FOLDER, f'ss2d_vit_epoch_{epoch+1}.pt')
            torch.save(ss2d_decoder.state_dict(), ckpt_path)
            tqdm.write(f'  Checkpoint 저장: {ckpt_path}')

    toc = time.time()
    print(f'\n학습 완료! 소요 시간: {toc - tic:.0f}초')
    print(f'Best Train SSIM: {best_train_ssim:.4f}  |  Best Composite: {best_composite:.4f}')
    print(f'  Best Val → SSIM: {best_val["ssim"]:.4f}  PSNR: {best_val["psnr"]:.2f}dB  NMSE: {best_val["nmse"]:.4f}  L1: {best_val["l1"]:.4f}')
    print(f'Best Checkpoint: {os.path.join(PATH_FOLDER, "ss2d_vit_best.pt")}')
    wandb.finish()


if __name__ == '__main__':
    main()
