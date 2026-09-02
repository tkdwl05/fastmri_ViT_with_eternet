"""
v8 Pure ETER-Net 2×2 ablation — 단일 파라미터화 trainer.

환경변수로 4런 구동 (everything else identical):
  SEQ_MODEL = gru | ss2d | axial | pixelgru  (3·4번째 팔은 2026-09-02 추가 — 공정성 실험)
  USE_DC    = 0  | 1         (DC 축; DCBlock + complex head)
  SMOKE_BS, ACCUM_STEPS, SANITY_NUM_EPOCHS, SANITY_VAL_EVERY_N_EPOCHS 도 override 가능.

교수님 ETER-net(no ViT) 충실: [gru_h+gru_v | SS2D] → cat(aliased) → U-Net DFU → magnitude.
loss = masked L1 + (1 - masked SSIM). best/EarlyStop = masked composite.
true-resume(full-state {prefix}_last.pt) + atomic save + '학습 완료' sentinel (supervisor용).
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
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'rnn_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'attn_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'tools'))

from myConfig_pure_eter_v8 import *           # noqa: F401,F403  (공유 하이퍼파라미터)
from u_choh_SSIM import SSIM
from dataloader_h5_v5 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader
from check_recon_env import check_env_for_model

# ── 런 선택 (4런: SEQ_MODEL × USE_DC) ──
SEQ_MODEL = os.environ.get('SEQ_MODEL', 'gru').lower()
assert SEQ_MODEL in ('gru', 'ss2d', 'axial', 'pixelgru'), \
    f"SEQ_MODEL must be gru|ss2d|axial|pixelgru, got {SEQ_MODEL}"
USE_DC    = os.environ.get('USE_DC', '0') == '1'

# ── override ──
NUM_EPOCHS         = int(os.environ.get('SANITY_NUM_EPOCHS', NUM_EPOCHS))
VAL_EVERY_N_EPOCHS = int(os.environ.get('SANITY_VAL_EVERY_N_EPOCHS', VAL_EVERY_N_EPOCHS))
BATCH_SIZE         = int(os.environ.get('SMOKE_BS', BATCH_SIZE))
ACCUM_STEPS        = int(os.environ.get('ACCUM_STEPS', ACCUM_STEPS))
NUM_VAL_FILES      = None

# ── 시드 (멀티시드 공정성 실험용, 2026-09-01. SEED 미설정 시 기존 런과 동작 동일) ──
_SEED_ENV = os.environ.get('SEED')
if _SEED_ENV is not None:
    SEED = int(_SEED_ENV)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)          # CPU+CUDA 가중치 초기화·dropout 고정
    print(f"[seed] SEED={SEED} — weights/shuffle/aug/mask-offset 고정 "
          f"(fp16 atomics 로 비트단위 재현은 아님)")

_TAG        = 'DC' if USE_DC else 'noDC'
_RUN_SUFFIX = os.environ.get('RUN_SUFFIX', f"_s{_SEED_ENV}" if _SEED_ENV is not None else '')
_RUN_NAME   = f"PureETER_{SEQ_MODEL.upper()}_{_TAG}_R4_brain384_v8{_RUN_SUFFIX}"
PATH_FOLDER = os.path.join(_PROJECT_ROOT, 'logs', _RUN_NAME)
os.makedirs(PATH_FOLDER, exist_ok=True)
PREFIX      = f"pure_{SEQ_MODEL}"
DATA_TRAIN  = os.path.join(_PROJECT_ROOT, 'fastMRI_data', 'multicoil_train')
DATA_VAL    = os.path.join(_PROJECT_ROOT, 'fastMRI_data', 'multicoil_val')


def save_checkpoint_atomic(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def skimage_ssim_batch_masked(pred, target, mask):
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


def build_model(device):
    if SEQ_MODEL == 'gru':
        from u_pure_eternet_gru import PureETER_GRU
        model = PureETER_GRU(
            dim=IMAGE_SIZE[0], n_coil=N_COIL,
            n_hidden_1=N_HIDDEN_LRNN_1, n_hidden_2=N_HIDDEN_LRNN_2,
            unet_depth=UNET_DEPTH, unet_wf=UNET_WF,
            use_dc=USE_DC, dc_k_scale_ratio=DC_K_SCALE_RATIO, dc_init_alpha=DC_INIT_ALPHA,
        )
    elif SEQ_MODEL == 'axial':
        # 3번째 팔 (2026-09-02, docs/axial_transformer_arm_design.md) — 신규 파일만 추가
        from u_pure_eternet_axial import PureETER_AXIAL
        model = PureETER_AXIAL(
            n_coil=N_COIL, n_hidden_2=N_HIDDEN_LRNN_2,
            unet_depth=UNET_DEPTH, unet_wf=UNET_WF,
            axial_d_model=int(os.environ.get('AXIAL_D_MODEL', AXIAL_D_MODEL)),
            axial_n_pairs=int(os.environ.get('AXIAL_N_PAIRS', AXIAL_N_PAIRS)),
            axial_n_heads=int(os.environ.get('AXIAL_N_HEADS', AXIAL_N_HEADS)),
            use_dc=USE_DC, dc_k_scale_ratio=DC_K_SCALE_RATIO, dc_init_alpha=DC_INIT_ALPHA,
        )
    elif SEQ_MODEL == 'pixelgru':
        # 4번째 팔: 가중치 공유 재귀 (2026-09-02, 공정성 — 메커니즘 vs 파라미터화 분리)
        from u_pure_eternet_pixelgru import PureETER_PIXELGRU
        model = PureETER_PIXELGRU(
            n_coil=N_COIL, n_hidden_2=N_HIDDEN_LRNN_2,
            unet_depth=UNET_DEPTH, unet_wf=UNET_WF,
            pixelgru_hidden=int(os.environ.get('PIXELGRU_HIDDEN', PIXELGRU_HIDDEN)),
            use_dc=USE_DC, dc_k_scale_ratio=DC_K_SCALE_RATIO, dc_init_alpha=DC_INIT_ALPHA,
        )
    else:
        from u_pure_eternet_ss2d import PureETER_SS2D
        model = PureETER_SS2D(
            n_coil=N_COIL, n_hidden_2=N_HIDDEN_LRNN_2,
            unet_depth=UNET_DEPTH, unet_wf=UNET_WF,
            ss2d_d_inner=SS2D_D_INNER, ss2d_d_state=SS2D_D_STATE,
            use_dc=USE_DC, dc_k_scale_ratio=DC_K_SCALE_RATIO, dc_init_alpha=DC_INIT_ALPHA,
        )
    return model.to(device)


def run_val(model, val_loader, device):
    model.eval()
    all_ssim, all_psnr, all_nmse, all_l1 = [], [], [], []
    val_bar = tqdm(val_loader, desc='  Val', leave=False, unit='batch')
    with torch.no_grad():
        for sample in val_bar:
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            brain_mask  = sample['brain_mask'].float().to(device)
            mask        = sample['mask'].float().to(device)
            sens        = sample['sens'].float().to(device)

            with torch.amp.autocast('cuda'):
                out = model(data_in_img, data_in, mask, sens)

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
            all_psnr.append(psnr); all_nmse.append(nmse); all_ssim.append(ssim); all_l1.append(l1)
            val_bar.set_postfix(SSIM=f'{ssim:.4f}', PSNR=f'{psnr:.2f}dB')

    model.train()
    ssim_m = float(np.mean(all_ssim)); psnr_m = float(np.mean(all_psnr))
    nmse_m = float(np.mean(all_nmse)); l1_m = float(np.mean(all_l1))
    psnr_n = min(psnr_m, PSNR_NORM) / PSNR_NORM
    nmse_n = max(0.0, 1.0 - min(nmse_m, 1.0))
    composite = COMPOSITE_W_SSIM * ssim_m + COMPOSITE_W_PSNR * psnr_n + COMPOSITE_W_NMSE * nmse_n
    return {'ssim': ssim_m, 'psnr': psnr_m, 'nmse': nmse_m, 'l1': l1_m, 'composite': composite}


def main():
    print('====================================================')
    print(f' [Pure ETER-Net v8]  SEQ_MODEL={SEQ_MODEL}  USE_DC={USE_DC}  '
          f'BS={BATCH_SIZE}  ACCUM={ACCUM_STEPS}  (eff_bs={BATCH_SIZE*ACCUM_STEPS})')
    print(f'   run={_RUN_NAME}   logs={PATH_FOLDER}')
    print('====================================================')

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필수.")
    device = torch.device("cuda")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
    env_kind = 'ss2d' if (SEQ_MODEL == 'ss2d' or USE_DC) else 'eter'   # mamba_ssm 필요 여부
    if not check_env_for_model(env_kind, 'myConfig_pure_eter_v8', strict=True):
        return

    model = build_model(device)

    # ── 재개: ① full-state last.pt → ② scratch ──
    last_ckpt_path = os.path.join(PATH_FOLDER, f'{PREFIX}_last.pt')
    _full_state = None
    if os.path.exists(last_ckpt_path):
        _full_state = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(_full_state['model'])
        resume_mode = 'full'
        print(f"\n[Resume:full] {last_ckpt_path} → epoch {_full_state['epoch']} 완료, 전체 상태 복원 예정")
    else:
        resume_mode = 'scratch'
        print("\n[Scratch] 처음부터 학습")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {num_params / 1e6:.1f}M")

    criterion_ssim_loss = SSIM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_ADAM,
                                 weight_decay=LAMBDA_REGULAR_PER_PIXEL)

    print("\nFastMRI 데이터 파이프라인 연결 중...")
    choh_data_train = FastMRI_H5_Dataloader(
        DATA_TRAIN, num_files=None, target_size=IMAGE_SIZE[0],
        augment=TRAIN_AUGMENT, augment_flip_p=TRAIN_AUGMENT_FLIP_P,
    )
    _train_loader_kwargs = dict(batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS_TRAIN, pin_memory=True)
    if NUM_WORKERS_TRAIN > 0:
        _train_loader_kwargs.update(persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)
    if _SEED_ENV is not None:
        # dataset rng(마스크 offset·flip) 시드 + 워커별 독립 스트림 + 셔플 순서 고정.
        # (기존 무시드 경로와의 차이는 이 블록뿐 — persistent_workers fork 시 16워커가
        #  동일 rng 상태를 복제하던 문제도 worker_init_fn 재시드로 함께 해소)
        choh_data_train.rng = np.random.default_rng(SEED + 1)

        def _worker_init_fn(worker_id):
            ws = torch.initial_seed() % 2**32      # torch 가 워커마다 다른 base seed 부여
            info = torch.utils.data.get_worker_info()
            info.dataset.rng = np.random.default_rng(ws)
            np.random.seed(ws)
            random.seed(ws)

        _g = torch.Generator()
        _g.manual_seed(SEED)
        _train_loader_kwargs.update(worker_init_fn=_worker_init_fn, generator=_g)
    trainloader = DataLoader(choh_data_train, **_train_loader_kwargs)
    print(f"Train Dataloader 준비 완료! ({len(choh_data_train)} 샘플)")
    choh_data_val = FastMRI_H5_Dataloader(
        DATA_VAL, num_files=NUM_VAL_FILES, target_size=IMAGE_SIZE[0],
        random_mask=False, augment=False,
    )
    val_loader = DataLoader(
        choh_data_val, batch_size=max(1, BATCH_SIZE // 2), shuffle=False,
        num_workers=NUM_WORKERS_VAL, pin_memory=True,
    )
    print(f"Val   Dataloader 준비 완료! ({len(choh_data_val)} 샘플)")

    steps_per_epoch = len(trainloader)
    opt_steps_per_epoch = max(1, steps_per_epoch // ACCUM_STEPS)
    total_steps = opt_steps_per_epoch * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    print(f"Scheduler: CosineAnnealingLR T_max={total_steps} opt-steps "
          f"({opt_steps_per_epoch}/epoch × {NUM_EPOCHS}), eta_min=1e-6")
    print(f"EarlyStopping: patience={EARLYSTOP_PATIENCE} val checks (val_composite)")

    _wandb_tag = os.environ.get('WANDB_RUN_TAG', '')
    _wandb_id  = _RUN_NAME + (f'_{_wandb_tag}' if _wandb_tag else '')
    wandb.init(
        project='ViT-MRI-Recon', name=f'{_wandb_id}_BS{BATCH_SIZE}x{ACCUM_STEPS}',
        id=_wandb_id, resume='allow',   # per-arm id(+WANDB_RUN_TAG): online 실시간·supervisor 재시작 연속. rerun 태그로 새 run(step 충돌 회피)
        config={
            'track': 'v8_eter_pure', 'seq_model': SEQ_MODEL, 'use_dc': USE_DC,
            'image_size': IMAGE_SIZE, 'n_hidden_lrnn': N_HIDDEN_LRNN_2,
            'ss2d_d_inner': SS2D_D_INNER, 'ss2d_d_state': SS2D_D_STATE,
            'unet_depth': UNET_DEPTH, 'unet_wf': UNET_WF,
            'batch_size': BATCH_SIZE, 'accum_steps': ACCUM_STEPS,
            'eff_batch': BATCH_SIZE * ACCUM_STEPS, 'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE_ADAM, 'weight_decay': LAMBDA_REGULAR_PER_PIXEL,
            'ssim_weight': LAMBDA_SSIM_PER_PIXEL, 'num_params': num_params,
            'train_samples': len(choh_data_train), 'val_samples': len(choh_data_val),
            'earlystop_metric': 'val_composite', 'earlystop_patience': EARLYSTOP_PATIENCE,
            'val_every_n_epochs': VAL_EVERY_N_EPOCHS, 'scheduler': 'CosineAnnealingLR',
        },
    )

    print(f"\n학습 시작 (총 {NUM_EPOCHS} 에폭)")
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    best_val_composite = -1.0
    best_val = {'ssim': None, 'psnr': None, 'nmse': None, 'l1': None, 'composite': None}
    no_improve_val_count = 0
    early_stopped = False
    tic = time.time()
    global_step = 0
    consec_skip = 0          # 연속 non-finite loss batch 수 (anti-spin)
    total_skip = 0           # 전체 skip 누계 (진단)
    log_path = os.path.join(PATH_FOLDER, 'log.txt')

    start_epoch = 0
    if resume_mode == 'full':
        optimizer.load_state_dict(_full_state['optimizer'])
        scheduler.load_state_dict(_full_state['scheduler'])
        scaler.load_state_dict(_full_state['scaler'])
        best_val_composite   = _full_state['best_val_composite']
        best_val             = _full_state['best_val']
        no_improve_val_count = _full_state.get('no_improve_val_count', 0)
        global_step          = _full_state.get('global_step', 0)
        start_epoch          = _full_state['epoch']
        rng = _full_state.get('rng', {})
        try:
            if rng.get('torch_cpu') is not None:  torch.set_rng_state(rng['torch_cpu'])
            if rng.get('torch_cuda') is not None: torch.cuda.set_rng_state_all(rng['torch_cuda'])
            if rng.get('numpy') is not None:      np.random.set_state(rng['numpy'])
            if rng.get('python') is not None:     random.setstate(rng['python'])
        except Exception as e:
            print(f"  [Resume] RNG 복원 일부 실패(무시): {e}")
        print(f"[Resume:full] start_epoch={start_epoch}, LR={scheduler.get_last_lr()[0]:.3e}, "
              f"best_composite={best_val_composite:.4f}, global_step={global_step}")
        with open(log_path, 'a') as f:
            f.write(f'RESUME start_epoch={start_epoch} best_composite={best_val_composite:.4f}\n')
    else:
        with open(log_path, 'a') as f:
            f.write(f'SCRATCH START run={_RUN_NAME} BS={BATCH_SIZE} ACCUM={ACCUM_STEPS} '
                    f'LR={LEARNING_RATE_ADAM} EPOCHS={NUM_EPOCHS} params={num_params/1e6:.1f}M\n')

    epoch_bar = tqdm(range(start_epoch, NUM_EPOCHS), desc='전체 진행', unit='epoch',
                     initial=start_epoch, total=NUM_EPOCHS)
    for epoch in epoch_bar:
        epoch_loss = epoch_ssim = epoch_psnr = epoch_nmse = epoch_l1 = 0.0
        batch_bar = tqdm(trainloader, desc=f'Epoch {epoch+1:3d}/{NUM_EPOCHS}', leave=False, unit='batch')
        optimizer.zero_grad(set_to_none=True)

        for i, sample in enumerate(batch_bar):
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            brain_mask  = sample['brain_mask'].float().to(device)
            mask        = sample['mask'].float().to(device)
            sens        = sample['sens'].float().to(device)

            with torch.amp.autocast('cuda'):
                out = model(data_in_img, data_in, mask, sens)

            out_fp    = out.float()
            m_sum     = brain_mask.sum().clamp(min=1.0)
            loss_l1   = ((out_fp - data_ref).abs() * brain_mask).sum() / m_sum
            loss_ssim = 1 - criterion_ssim_loss(out_fp, data_ref, mask=brain_mask)
            loss      = loss_l1 + LAMBDA_SSIM_PER_PIXEL * loss_ssim

            # ── NaN/Inf-skip 가드 (self-heal + anti-spin) ──
            # 비정상 loss batch 는 backward/step 없이 건너뛴다(정상 batch 만 학습·통계 기여).
            # 연속 MAX_CONSEC_SKIP 초과 시 fail-fast(exit1) → supervisor 가 clean last.pt 에서 재개.
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                consec_skip += 1; total_skip += 1
                if consec_skip <= 3 or consec_skip % 50 == 0:
                    tqdm.write(f'  [NaN-skip] ep{epoch+1} batch{i} loss={loss.item()} '
                               f'consec={consec_skip} total={total_skip}')
                if consec_skip >= MAX_CONSEC_SKIP:
                    _msg = (f'FATAL: {consec_skip} consecutive non-finite loss (ep{epoch+1} batch{i}) '
                            f'→ fail-fast exit(1). [α clamp 후에도 재발 시 조사 필요]')
                    tqdm.write('  ' + _msg)
                    with open(log_path, 'a') as f:
                        f.write(_msg + '\n')
                    try:
                        wandb.finish(exit_code=1)
                    except Exception:
                        pass
                    sys.exit(1)
                continue
            consec_skip = 0

            scaler.scale(loss / ACCUM_STEPS).backward()
            do_step = ((i + 1) % ACCUM_STEPS == 0)
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # α clamp: overshoot(α>1) 물리 차단 = DC forward non-finite 근본원인 제거 (trainable 유지)
                if USE_DC:
                    with torch.no_grad():
                        model.dc.alpha.clamp_(DC_ALPHA_MIN, DC_ALPHA_MAX)

            with torch.no_grad():
                out_f = out_fp.detach(); ref_f = data_ref.float(); m_t = brain_mask
                diff_sq_sum_t = ((out_f - ref_f) ** 2 * m_t).sum()
                mse_val  = (diff_sq_sum_t / m_sum).item()
                ref_max_in_mask_t = (ref_f * m_t).max().clamp(min=1e-10).item()
                psnr_val = 20 * np.log10(ref_max_in_mask_t / max(np.sqrt(mse_val), 1e-10))
                nmse_val = (diff_sq_sum_t / (ref_f ** 2 * m_t).sum().clamp(min=1e-10)).item()
                ssim_val_train = 1 - loss_ssim.item()

            global_step += 1
            wandb.log({
                'train/loss': loss.item(), 'train/loss_l1': loss_l1.item(),
                'train/loss_ssim': loss_ssim.item(), 'train/psnr': psnr_val,
                'train/nmse': nmse_val, 'train/lr': scheduler.get_last_lr()[0],
            }, step=global_step)

            epoch_loss += loss.item(); epoch_ssim += ssim_val_train
            epoch_psnr += psnr_val; epoch_nmse += nmse_val; epoch_l1 += loss_l1.item()
            batch_bar.set_postfix(Loss=f'{loss.item():.4f}', PSNR=f'{psnr_val:.2f}dB',
                                  SSIM_c=f'{ssim_val_train:.4f}', LR=f'{scheduler.get_last_lr()[0]:.2e}')

        n_batches = len(trainloader)
        avg_loss = epoch_loss / n_batches; avg_train_ssim = epoch_ssim / n_batches
        wandb.log({'epoch': epoch + 1, 'epoch/train_loss': avg_loss,
                   'epoch/train_ssim_custom': avg_train_ssim,
                   'epoch/train_psnr': epoch_psnr / n_batches,
                   'epoch/train_l1': epoch_l1 / n_batches}, step=global_step)

        do_val = (epoch + 1) % VAL_EVERY_N_EPOCHS == 0
        if do_val:
            tqdm.write(f'  [Val ep{epoch+1}] running...')
            val_metrics = run_val(model, val_loader, device)
            tqdm.write(f'  [Val] composite={val_metrics["composite"]:.4f}  SSIM_m={val_metrics["ssim"]:.4f}'
                       f'  PSNR={val_metrics["psnr"]:.2f}dB  NMSE={val_metrics["nmse"]:.4f}  L1={val_metrics["l1"]:.4f}')
            wandb.log({'val/composite': val_metrics['composite'], 'val/ssim_masked': val_metrics['ssim'],
                       'val/psnr_masked': val_metrics['psnr'], 'val/nmse_masked': val_metrics['nmse'],
                       'val/l1_masked': val_metrics['l1']}, step=global_step)
            with open(log_path, 'a') as f:
                f.write(f'Epoch {epoch+1}/{NUM_EPOCHS}  train_loss={avg_loss:.4f}'
                        f'  val_composite={val_metrics["composite"]:.4f}  val_ssim_m={val_metrics["ssim"]:.4f}'
                        f'  val_psnr={val_metrics["psnr"]:.2f}  val_nmse={val_metrics["nmse"]:.4f}'
                        f'  val_l1={val_metrics["l1"]:.4f}\n')
            if val_metrics['composite'] > best_val_composite:
                best_val_composite = val_metrics['composite']; best_val = dict(val_metrics)
                save_checkpoint_atomic(model.state_dict(), os.path.join(PATH_FOLDER, f'{PREFIX}_best.pt'))
                tqdm.write(f'  [Best] composite {best_val_composite:.4f} → {PREFIX}_best.pt')
                no_improve_val_count = 0
            else:
                no_improve_val_count += 1
                tqdm.write(f'  [No improve] {no_improve_val_count}/{EARLYSTOP_PATIENCE}')
            if no_improve_val_count >= EARLYSTOP_PATIENCE:
                with open(log_path, 'a') as f:
                    f.write(f'EARLYSTOP at epoch {epoch+1}\n')
                early_stopped = True
                break
        else:
            with open(log_path, 'a') as f:
                f.write(f'Epoch {epoch+1}/{NUM_EPOCHS}  train_loss={avg_loss:.4f}\n')

        if (epoch + 1) % 5 == 0:
            save_checkpoint_atomic(model.state_dict(), os.path.join(PATH_FOLDER, f'{PREFIX}_epoch_{epoch+1}.pt'))

        save_checkpoint_atomic({
            'epoch': epoch + 1, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'best_val_composite': best_val_composite,
            'best_val': best_val, 'no_improve_val_count': no_improve_val_count,
            'global_step': global_step,
            'rng': {'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(),
                    'numpy': np.random.get_state(), 'python': random.getstate()},
        }, os.path.join(PATH_FOLDER, f'{PREFIX}_last.pt'))

    toc = time.time()
    print(f'\n학습 완료 (early_stopped={early_stopped})  소요: {toc - tic:.0f}초')
    print(f'Best Val Composite: {best_val_composite:.4f}')
    if best_val['ssim'] is not None:
        print(f'  Best → composite {best_val["composite"]:.4f}  SSIM_m {best_val["ssim"]:.4f}'
              f'  PSNR {best_val["psnr"]:.2f}dB  NMSE {best_val["nmse"]:.4f}  L1 {best_val["l1"]:.4f}')
    print(f'Best Checkpoint: {os.path.join(PATH_FOLDER, f"{PREFIX}_best.pt")}')
    wandb.finish()


if __name__ == '__main__':
    main()
