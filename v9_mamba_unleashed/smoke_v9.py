"""
v9 VRAM + 속도 스모크 — unleashed / radapt 두 변형 각각 full train-step(forward+loss+backward+
unscale+clip+step) 으로 BS auto-fallback + steady-state step time 측정.

- 랜덤 텐서(실 shape)로 mamba 커널·DC FFT·U-Net 실메모리/실속도 측정.
- 각 변형의 최대 가능 BS 를 <track>/runs/smoke_bs.txt 에 기록.
- step time → per-epoch 추정(N_train≈65000) → v8 SS2D 기준(BS8 1.23 s/batch ≈ 2.78h/ep) 대비 비교.

실행: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python v9_mamba_unleashed/smoke_v9.py
      SMOKE_BS_LIST=8,6,4,2 로 후보 조정 가능.
"""

import os
import sys
import gc
import time
import traceback

import torch

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_PROJECT_ROOT, 'v9_mamba_unleashed', 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'v9_mamba_radapt', 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'tools'))

from u_choh_SSIM import SSIM

N_TRAIN_REF = 65000   # v8 참고: ~8129 step × BS8. per-epoch 추정용(근사)
V8_SS2D_REF = "v8 SS2D: BS8, 1.23 s/batch ≈ 2.78 h/ep"
N_WARMUP = 2
N_MEASURE = 6


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _cfg_override(C):
    """env 로 d_inner/d_state/n_blocks/ckpt/downsample 오버라이드 (config 스윕용)."""
    di = int(os.environ.get('SMOKE_D_INNER', C.SS2D_D_INNER))
    ds = int(os.environ.get('SMOKE_D_STATE', C.SS2D_D_STATE))
    nb = int(os.environ.get('SMOKE_N_BLOCKS', C.SS2D_N_BLOCKS))
    ck = os.environ.get('SMOKE_CKPT', '1' if C.SS2D_USE_CHECKPOINT else '0') == '1'
    dn = int(os.environ.get('SMOKE_DOWNSAMPLE', getattr(C, 'SS2D_DOWNSAMPLE', 1)))
    return di, ds, nb, ck, dn


def build(variant, device):
    import myConfig_ss2d_v9 as CU
    if variant == 'unleashed':
        from u_pure_eternet_ss2d_v9 import PureETER_SS2D_V9
        di, ds, nb, ck, dn = _cfg_override(CU)
        m = PureETER_SS2D_V9(
            n_coil=CU.N_COIL, out_ch=CU.SS2D_OUT_CH, unet_depth=CU.UNET_DEPTH, unet_wf=CU.UNET_WF,
            ss2d_d_inner=di, ss2d_d_state=ds, ss2d_n_blocks=nb, ss2d_dropout=CU.SS2D_DROPOUT,
            ss2d_use_checkpoint=ck, ss2d_downsample=dn,
        )
        return m.to(device), CU
    import myConfig_ss2d_v9_radapt as CR
    from u_pure_eternet_ss2d_v9_radapt import PureETER_SS2D_V9_Radapt
    di, ds, nb, ck, dn = _cfg_override(CR)
    m = PureETER_SS2D_V9_Radapt(
        n_coil=CR.N_COIL, out_ch=CR.SS2D_OUT_CH, unet_depth=CR.UNET_DEPTH, unet_wf=CR.UNET_WF,
        ss2d_d_inner=di, ss2d_d_state=ds, ss2d_n_blocks=nb, ss2d_dropout=CR.SS2D_DROPOUT,
        ss2d_use_checkpoint=ck, mask_condition=CR.MASK_CONDITION,
        dc_k_scale_ratio=CR.DC_K_SCALE_RATIO, dc_init_alpha=CR.DC_INIT_ALPHA,
        ss2d_downsample=dn,
    )
    return m.to(device), CR


def try_one_bs(variant, bs):
    _free()
    try:
        device = torch.device('cuda')
        model, C = build(variant, device)
        model.train()
        H = W = C.IMAGE_SIZE[0]
        ssim_loss = SSIM().to(device)

        x_img = torch.randn(bs, C.INPUT_CHANNELS, H, W, device=device)
        x_ksp = torch.randn(bs, C.INPUT_CHANNELS, H, W, device=device)
        ref   = torch.randn(bs, 1, H, W, device=device).abs()
        mask  = torch.zeros(bs, 1, H, W, device=device); mask[..., ::4] = 1.0
        sens  = torch.randn(bs, C.INPUT_CHANNELS, H, W, device=device)
        bm    = torch.zeros(bs, 1, H, W, device=device); bm[..., H//5:4*H//5, W//5:4*W//5] = 1.0

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        step_times = []
        for it in range(N_WARMUP + N_MEASURE):
            torch.cuda.synchronize(); t0 = time.time()
            with torch.amp.autocast('cuda'):
                out = model(x_img, x_ksp, mask, sens)
                out_fp = out.float()
                m_sum = bm.sum().clamp(min=1.0)
                loss_l1 = ((out_fp - ref).abs() * bm).sum() / m_sum
                loss_ssim = 1 - ssim_loss(out_fp, ref, mask=bm)
                loss = loss_l1 + C.LAMBDA_SSIM_PER_PIXEL * loss_ssim
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            torch.cuda.synchronize()
            if it >= N_WARMUP:
                step_times.append(time.time() - t0)

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        s_per_batch = sum(step_times) / len(step_times)
        finite = bool(torch.isfinite(loss))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' in str(e).lower():
            _free(); return None
        traceback.print_exc(); raise
    del model
    _free()
    return peak_gb, s_per_batch, finite


def smoke_for(variant, bs_candidates):
    print(f'\n========== smoke: {variant} ==========', flush=True)
    for bs in bs_candidates:
        print(f'  [try] BS={bs} ...', flush=True)
        res = try_one_bs(variant, bs)
        if res is None:
            print(f'  [OOM] BS={bs}')
            continue
        peak, spb, finite = res
        s_per_sample = spb / bs
        ep_h = (N_TRAIN_REF * s_per_sample) / 3600.0
        fin = '' if finite else '  [WARN non-finite loss(랜덤입력 아티팩트 가능)]'
        print(f'  [OK]  BS={bs}  peak={peak:.2f} GB  {spb:.2f} s/batch  '
              f'({s_per_sample:.3f} s/sample → ~{ep_h:.2f} h/ep, ~{ep_h*50:.1f} h/50ep){fin}')
        return {'variant': variant, 'bs': bs, 'peak': peak, 's_per_batch': spb,
                's_per_sample': s_per_sample, 'ep_h': ep_h, 'finite': finite}
    print(f'  [FAIL] {variant}: 후보 BS 전부 OOM')
    return {'variant': variant, 'bs': None}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU 필수')
    print(f'GPU: {torch.cuda.get_device_name(0)}  '
          f'VRAM: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB')
    print(f'기준: {V8_SS2D_REF}  (N_train_ref={N_TRAIN_REF})')

    bs_env = os.environ.get('SMOKE_BS_LIST')
    bs_list = tuple(int(b) for b in bs_env.split(',')) if bs_env else (8, 6, 4, 2, 1)

    runs = {
        'unleashed': os.path.join(_PROJECT_ROOT, 'v9_mamba_unleashed', 'runs'),
        'radapt':    os.path.join(_PROJECT_ROOT, 'v9_mamba_radapt', 'runs'),
    }
    results = []
    for variant in ('unleashed', 'radapt'):
        r = smoke_for(variant, bs_list)
        results.append(r)
        if r['bs'] is not None:
            os.makedirs(runs[variant], exist_ok=True)
            with open(os.path.join(runs[variant], 'smoke_bs.txt'), 'w') as f:
                f.write(f"{r['bs']}\n")

    print('\n========== summary ==========')
    for r in results:
        if r['bs'] is None:
            print(f"  {r['variant']:10s}  FAIL (OOM)"); continue
        print(f"  {r['variant']:10s}  BS={r['bs']}  peak={r['peak']:.2f}GB  "
              f"{r['s_per_batch']:.2f} s/batch  ~{r['ep_h']:.2f} h/ep  ~{r['ep_h']*50:.1f} h/50ep  "
              f"→ smoke_bs.txt 기록")
    print(f"\n  v8 SS2D 기준 2.78 h/ep 대비 위 h/ep 비교 → 빠르면 epoch 상향 여지.")


if __name__ == '__main__':
    main()
