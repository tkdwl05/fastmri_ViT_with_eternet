"""
v8 Pure ETER-Net VRAM smoke — 4 variant {gru,ss2d}×{noDC,DC} 각각 full train-step
(forward + masked L1 + (1-SSIM) + backward + unscale + clip + step) 으로 BS auto-fallback.

공정 비교를 위해 4런 공통 BS = 4 variant 중 최소 가능 BS (GRU arm 이 구속조건).
결과를 v8_eter_pure/runs/smoke_bs.txt 에 기록 (chain 이 SMOKE_BS 로 주입).
GRU arm 의 autocast+backward 도 여기서 처음 실검증된다.
"""

import os
import sys
import gc
import time
import traceback

import torch

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))

import myConfig_pure_eter_v8 as C
from u_choh_SSIM import SSIM


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build(seq, use_dc, device):
    if seq == 'gru':
        from u_pure_eternet_gru import PureETER_GRU
        return PureETER_GRU(
            dim=C.IMAGE_SIZE[0], n_coil=C.N_COIL,
            n_hidden_1=C.N_HIDDEN_LRNN_1, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            use_dc=use_dc, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        ).to(device)
    from u_pure_eternet_ss2d import PureETER_SS2D
    return PureETER_SS2D(
        n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
        unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
        ss2d_d_inner=C.SS2D_D_INNER, ss2d_d_state=C.SS2D_D_STATE,
        use_dc=use_dc, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
    ).to(device)


def try_one_bs(seq, use_dc, bs):
    _free_cuda()
    try:
        device = torch.device('cuda')
        model = build(seq, use_dc, device)
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

        with torch.amp.autocast('cuda'):
            out = model(x_img, x_ksp, mask, sens)
            out_fp = out.float()
            m_sum = bm.sum().clamp(min=1.0)
            loss_l1 = ((out_fp - ref).abs() * bm).sum() / m_sum
            loss_ssim = 1 - ssim_loss(out_fp, ref, mask=bm)
            loss = loss_l1 + C.LAMBDA_SSIM_PER_PIXEL * loss_ssim

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' in str(e).lower():
            _free_cuda()
            return None
        traceback.print_exc()
        raise
    del model
    _free_cuda()
    return peak_gb


def smoke_for(seq, use_dc, bs_candidates):
    label = f"{seq}_{'DC' if use_dc else 'noDC'}"
    print(f'\n========== smoke: {label} ==========', flush=True)
    for bs in bs_candidates:
        print(f'  [try] BS={bs} ...', flush=True)
        t0 = time.time()
        peak = try_one_bs(seq, use_dc, bs)
        dt = time.time() - t0
        if peak is None:
            print(f'  [OOM] BS={bs} ({dt:.1f}s)')
            continue
        print(f'  [OK]  BS={bs}  peak={peak:.2f} GB  ({dt:.1f}s)')
        return label, bs, peak
    print(f'  [FAIL] {label}: BS=1 까지 OOM')
    return label, None, None


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU 필수')
    runs_dir = os.path.join(_HERE, 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB')

    bs_env = os.environ.get('SMOKE_BS_LIST')
    bs_list = tuple(int(b) for b in bs_env.split(',')) if bs_env else (4, 2, 1)

    results = []
    for use_dc in (False, True):
        for seq in ('gru', 'ss2d'):
            results.append(smoke_for(seq, use_dc, bs_list))

    print('\n========== smoke summary ==========')
    feasible = []
    for label, bs, peak in results:
        print(f'  {label:12s}  BS={bs}  peak={peak if peak is None else f"{peak:.2f} GB"}')
        if bs is not None:
            feasible.append(bs)
    if not feasible:
        print('  [FATAL] 어떤 variant 도 BS=1 불가 → capacity(H2/unet_depth) 조정 필요')
        sys.exit(2)
    common_bs = min(feasible)
    print(f'\n  → 공정 비교용 공통 BS = min = {common_bs}  (GRU arm 이 구속조건)')
    with open(os.path.join(runs_dir, 'smoke_bs.txt'), 'w') as f:
        f.write(f'{common_bs}\n')
    print(f'  기록: {os.path.join(runs_dir, "smoke_bs.txt")}')


if __name__ == '__main__':
    main()
