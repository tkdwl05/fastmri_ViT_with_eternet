"""
ETER-ViT 학습 스크립트 v7_titan (single-GPU, TITAN RTX 24GB, scratch, 384×384)
  - Config:     v7_titan/configs/myConfig_choh_ETER_model_v7_titan
  - Dataloader: dataloaders/dataloader_h5_v5 (target_size=384)
  - Encoder:    choh_ViT (ViT-Base: dim=768, L=12, H=12, MLP=3072)
  - Decoder:    u_choh_model_ETER_ViT_v7_titan.choh_Decoder3_ETER_v7_titan
                  (v5 dropout 적용 디코더 + 최종합성 UNet_choh_skip 으로 교체)
  - 변경점 (v7 대비):
      * IMAGE_SIZE 320→384
      * ViT-Small → ViT-Base
      * 최종 합성 RefinementBlock → UNet_choh_skip(depth=3, wf=6)
      * BATCH_SIZE 16→4 (smoke 결과로 자동 조정)
"""

import os
import sys
import time
import datetime
import random
import pytz

import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader
from check_recon_env import check_env_for_model

NUM_EPOCHS         = int(os.environ.get('SANITY_NUM_EPOCHS', NUM_EPOCHS))
VAL_EVERY_N_EPOCHS = int(os.environ.get('SANITY_VAL_EVERY_N_EPOCHS', VAL_EVERY_N_EPOCHS))
BATCH_SIZE         = int(os.environ.get('SMOKE_BS', BATCH_SIZE))
NUM_VAL_FILES      = None


def save_checkpoint_atomic(obj, path):
    """tmp 파일에 쓴 뒤 os.replace 로 원자적 교체 — 저장 도중 프로세스가 죽어도
    기존 last.pt 가 깨지지 않도록 보장 (멀티데이 학습 중단 내성)."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def skimage_ssim_batch_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """배경 픽셀 제외하고 brain_mask 안에서만 skimage SSIM 평균."""
    p = pred.detach().float().cpu().numpy()
    t = target.detach().float().cpu().numpy()
    m = mask.detach().float().cpu().numpy()
    if p.ndim == 4:
        p, t, m = p[:, 0], t[:, 0], m[:, 0]
    m_bool = m > 0.5
    vals = []
    for i in range(p.shape[0]):
        if not m_bool[i].any():
            continue
        t_in = t[i][m_bool[i]]
        dr = float(t_in.max() - t_in.min())
        if dr <= 0:
            continue
        _, ssim_map = compare_ssim(t[i], p[i], data_range=dr, full=True)
        vals.append(float(ssim_map[m_bool[i]].mean()))
    return float(np.mean(vals)) if vals else 0.0


def run_val(model, val_loader, criterion_l1, device):
    """brain_mask 안에서 SSIM/PSNR/NMSE/L1 측정 + weighted composite 산출."""
    model.eval()
    all_ssim, all_psnr, all_nmse, all_l1 = [], [], [], []
    val_bar = tqdm(val_loader, desc='  Val', leave=False, unit='batch')
    with torch.no_grad():
        for sample in val_bar:
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
            mse  = diff_sq_sum / m_sum
            ref_max_in_mask = (ref_f * m).max().clamp(min=1e-10)
            psnr = (20 * torch.log10(ref_max_in_mask / torch.sqrt(mse.clamp(min=1e-10)))).item()
            ref_sq_sum = (ref_f ** 2 * m).sum().clamp(min=1e-10)
            nmse = (diff_sq_sum / ref_sq_sum).item()
            ssim = skimage_ssim_batch_masked(out_f, ref_f, m)
            l1   = (((out_f - ref_f).abs() * m).sum() / m_sum).item()

            all_psnr.append(psnr)
            all_nmse.append(nmse)
            all_ssim.append(ssim)
            all_l1.append(l1)
            val_bar.set_postfix(SSIM=f'{ssim:.4f}', PSNR=f'{psnr:.2f}dB')

    model.train()
    ssim_m = float(np.mean(all_ssim))
    psnr_m = float(np.mean(all_psnr))
    nmse_m = float(np.mean(all_nmse))
    l1_m   = float(np.mean(all_l1))
    psnr_n = min(psnr_m, PSNR_NORM) / PSNR_NORM
    nmse_n = max(0.0, 1.0 - min(nmse_m, 1.0))
    composite = (COMPOSITE_W_SSIM * ssim_m
                 + COMPOSITE_W_PSNR * psnr_n
                 + COMPOSITE_W_NMSE * nmse_n)
    return {
        'ssim':      ssim_m,
        'psnr':      psnr_m,
        'nmse':      nmse_m,
        'l1':        l1_m,
        'composite': composite,
    }


def main():
    print('====================================================')
    print(f' [ETER-ViT v7_titan] scratch + 384x384 + ViT-Base + UNet 후처리 복원  BS={BATCH_SIZE}')
    print('====================================================')

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필수.")
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
    if not check_env_for_model('eter', 'myConfig_choh_ETER_model_v7_titan', strict=True):
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

    # ── 재개 우선순위: ① full-state last.pt → ② weights-only RESUME_CKPT(레거시) → ③ scratch ──
    # optimizer/scheduler/scaler 생성 전이므로 여기선 model weight 만 load, 나머지 상태는 생성 후 복원.
    last_ckpt_path = os.path.join(PATH_FOLDER, 'eter_vit_last.pt')
    _full_state = None
    run_baseline = False
    if os.path.exists(last_ckpt_path):
        _full_state = torch.load(last_ckpt_path, map_location=device)
        eter_decoder.load_state_dict(_full_state['model'])
        resume_mode = 'full'
        print(f"\n[Resume:full] {last_ckpt_path} 발견 → epoch {_full_state['epoch']} 까지 완료,"
              f" 전체 상태(optimizer/scheduler/scaler/RNG) 복원 예정 (LR 연속)")
    elif RESUME_CKPT and os.path.exists(RESUME_CKPT):
        state = torch.load(RESUME_CKPT, map_location=device)
        eter_decoder.load_state_dict(state)
        resume_mode = 'warm'
        run_baseline = True
        print(f"\n[Resume:warm] weights-only from: {RESUME_CKPT} (레거시 warm-start)")
    else:
        resume_mode = 'scratch'
        print("\n[Scratch] 처음부터 학습 (last.pt 없음, RESUME_CKPT is None)")

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
        target_size=IMAGE_SIZE[0],
        augment=TRAIN_AUGMENT, augment_flip_p=TRAIN_AUGMENT_FLIP_P,
    )
    trainloader = DataLoader(
        choh_data_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS_TRAIN, pin_memory=True,
        persistent_workers=True, prefetch_factor=PREFETCH_FACTOR,
    )
    print(f"Train Dataloader 준비 완료! ({len(choh_data_train)} 샘플)")

    choh_data_val = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_val', num_files=NUM_VAL_FILES,
        target_size=IMAGE_SIZE[0],
        random_mask=False, augment=False,
    )
    val_loader = DataLoader(
        choh_data_val, batch_size=max(1, BATCH_SIZE // 2), shuffle=False,
        num_workers=NUM_WORKERS_VAL, pin_memory=True,
    )
    print(f"Val   Dataloader 준비 완료! ({len(choh_data_val)} 샘플)")

    steps_per_epoch = len(trainloader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    print(f"Scheduler: CosineAnnealingLR T_max={total_steps} (={NUM_EPOCHS} epochs), eta_min=1e-6")
    print(f"EarlyStopping: patience={EARLYSTOP_PATIENCE} val checks (val_composite 기준)")

    wandb.init(
        project='ViT-MRI-Recon',
        name=f'ETER_v7_titan_scratch_BS{BATCH_SIZE}_LR{LEARNING_RATE_ADAM}_EP{NUM_EPOCHS}',
        config={
            'model': 'ETER-ViT-v7_titan',
            'image_size': IMAGE_SIZE,
            'patch_size': PATCH_SIZE,
            'encoder_hidden': NUM_VIT_ENCODER_HIDDEN,
            'encoder_layers': NUM_VIT_ENCODER_LAYER,
            'encoder_heads':  NUM_VIT_ENCODER_HEAD,
            'encoder_mlp':    NUM_VIT_ENCODER_MLP_SIZE,
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
            'batch_size':     BATCH_SIZE,
            'num_epochs':     NUM_EPOCHS,
            'learning_rate':  LEARNING_RATE_ADAM,
            'ssim_weight':    LAMBDA_SSIM_PER_PIXEL,
            'val_metric':       'skimage_ssim',
            'earlystop_metric': 'val_composite',
            'earlystop_patience': EARLYSTOP_PATIENCE,
            'val_every_n_epochs': VAL_EVERY_N_EPOCHS,
            'num_params':         num_params,
            'train_samples':      len(choh_data_train),
            'val_samples':        len(choh_data_val),
            'scheduler': 'CosineAnnealingLR',
            'T_max':     total_steps,
            'eta_min':   1e-6,
            'num_workers_train': NUM_WORKERS_TRAIN,
            'prefetch_factor':   PREFETCH_FACTOR,
        },
    )
    wandb.watch(eter_decoder, log='gradients', log_freq=100)

    print(f"\n학습 시작 (총 {NUM_EPOCHS} 에폭, EarlyStop 가능)")
    scaler = torch.amp.GradScaler('cuda')
    eter_decoder.train()
    best_val_composite = -1.0
    best_val = {'ssim': None, 'psnr': None, 'nmse': None, 'l1': None, 'composite': None}
    no_improve_val_count = 0
    early_stopped = False
    tic = time.time()
    global_step = 0
    log_path = os.path.join(PATH_FOLDER, 'log.txt')

    # ── full-state 복원 (optimizer/scheduler/scaler/counter/RNG) — scheduler 생성 후 시점 ──
    start_epoch = 0
    if resume_mode == 'full':
        optimizer.load_state_dict(_full_state['optimizer'])
        scheduler.load_state_dict(_full_state['scheduler'])
        scaler.load_state_dict(_full_state['scaler'])
        best_val_composite   = _full_state['best_val_composite']
        best_val             = _full_state['best_val']
        no_improve_val_count = _full_state.get('no_improve_val_count', 0)
        global_step          = _full_state.get('global_step', 0)
        start_epoch          = _full_state['epoch']   # = 완료한 epoch 수 (다음 0-indexed epoch)
        rng = _full_state.get('rng', {})
        try:
            if rng.get('torch_cpu') is not None:
                torch.set_rng_state(rng['torch_cpu'])
            if rng.get('torch_cuda') is not None:
                torch.cuda.set_rng_state_all(rng['torch_cuda'])
            if rng.get('numpy') is not None:
                np.random.set_state(rng['numpy'])
            if rng.get('python') is not None:
                random.setstate(rng['python'])
        except Exception as e:
            print(f"  [Resume] RNG 복원 일부 실패(무시): {e}")
        cur_lr = scheduler.get_last_lr()[0]
        print(f"[Resume:full] 복원 완료 → start_epoch(0-idx)={start_epoch} "
              f"(epoch {start_epoch+1}/{NUM_EPOCHS} 부터), LR={cur_lr:.3e}, "
              f"best_composite={best_val_composite:.4f}, global_step={global_step}, "
              f"no_improve={no_improve_val_count}")
        with open(log_path, 'a') as f:
            f.write(f'RESUME from {last_ckpt_path}  start_epoch={start_epoch}  '
                    f'LR={cur_lr:.3e}  best_composite={best_val_composite:.4f}  '
                    f'global_step={global_step}\n')

    if run_baseline:
        tqdm.write('Resume baseline val 측정 중...')
        baseline = run_val(eter_decoder, val_loader, criterion_l1, device)
        best_val_composite = baseline['composite']
        best_val = dict(baseline)
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
        torch.save(eter_decoder.state_dict(), os.path.join(PATH_FOLDER, 'eter_vit_best.pt'))
    elif resume_mode == 'scratch':
        with open(log_path, 'a') as f:
            f.write(f'SCRATCH START  BATCH_SIZE={BATCH_SIZE}  LR={LEARNING_RATE_ADAM}  EPOCHS={NUM_EPOCHS}\n')
    # resume_mode == 'full': RESUME 라인은 위에서 이미 기록

    epoch_bar = tqdm(range(start_epoch, NUM_EPOCHS), desc='전체 진행', unit='epoch',
                     initial=start_epoch, total=NUM_EPOCHS)
    for epoch in epoch_bar:
        epoch_loss = epoch_ssim = epoch_psnr = epoch_nmse = epoch_l1 = 0.0
        batch_bar = tqdm(trainloader, desc=f'Epoch {epoch+1:3d}/{NUM_EPOCHS}', leave=False, unit='batch')

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
                'val/composite':   val_metrics['composite'],
                'val/ssim_masked': val_metrics['ssim'],
                'val/psnr_masked': val_metrics['psnr'],
                'val/nmse_masked': val_metrics['nmse'],
                'val/l1_masked':   val_metrics['l1'],
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
                torch.save(eter_decoder.state_dict(), best_ckpt_path)
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

            if no_improve_val_count >= EARLYSTOP_PATIENCE:
                tqdm.write(
                    f'  [EarlyStop] {no_improve_val_count} consecutive val checks'
                    f' without val_composite improvement → 학습 종료 (epoch {epoch+1})'
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

        # ── true-resume full-state (rolling, atomic) — 매 epoch 갱신, 중단 내성 ──
        save_checkpoint_atomic({
            'epoch': epoch + 1,                       # 완료한 epoch 수 (재개 시 start_epoch)
            'model': eter_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_composite': best_val_composite,
            'best_val': best_val,
            'no_improve_val_count': no_improve_val_count,
            'global_step': global_step,
            'rng': {
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all(),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            },
        }, os.path.join(PATH_FOLDER, 'eter_vit_last.pt'))

    toc = time.time()
    print(f'\n학습 완료 (early_stopped={early_stopped})  소요 시간: {toc - tic:.0f}초')
    print(f'Best Val Composite: {best_val_composite:.4f}')
    if best_val['ssim'] is not None:
        print(f'  Best Val → Composite: {best_val["composite"]:.4f}'
              f'  SSIM_m: {best_val["ssim"]:.4f}'
              f'  PSNR: {best_val["psnr"]:.2f}dB'
              f'  NMSE: {best_val["nmse"]:.4f}  L1: {best_val["l1"]:.4f}')
    print(f'Best Checkpoint: {os.path.join(PATH_FOLDER, "eter_vit_best.pt")}')
    wandb.finish()


if __name__ == '__main__':
    main()
