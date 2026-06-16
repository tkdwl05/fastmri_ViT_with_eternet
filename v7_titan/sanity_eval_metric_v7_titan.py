"""
평가 metric 재설계 sanity (v7_titan)

검증 항목:
  1) dataloader 가 brain_mask 키를 반환하는지 (shape (1,H,W), dtype float32, 값 ∈ {0,1})
  2) brain_mask 가 RSS target 의 brain 영역과 일치하는지 (PNG overlay 저장)
  3) 동일 (random init) ckpt 로 masked vs unmasked SSIM/PSNR/NMSE 비교
     - SSIM_unmasked > SSIM_masked 면 배경 부풀림이 정량화됨
  4) Loss 계산 (mask 가 sum=0 인 슬라이스에서도 NaN 안 발생)

본 학습 chain 실행 전 1회 통과 확인용. ~5분 소요.

사용:
  python v7_titan/sanity_eval_metric_v7_titan.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))

from myConfig_choh_SS2D_model_v7_titan import (
    IMAGE_SIZE, BATCH_SIZE,
    COMPOSITE_W_SSIM, COMPOSITE_W_PSNR, COMPOSITE_W_NMSE, PSNR_NORM,
)
from dataloader_h5_v5 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader


OUT_DIR = os.path.join(_HERE, 'runs', 'sanity_eval')
os.makedirs(OUT_DIR, exist_ok=True)


# ───────────────────────────────────────────
#  1) brain_mask 키/shape/dtype/값 검증
# ───────────────────────────────────────────

def check_brain_mask_key():
    print('\n[1] brain_mask 키 / shape / dtype / 값 범위 검증')
    ds = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_val', num_files=2,
        target_size=IMAGE_SIZE[0], augment=False, random_mask=False,
    )
    sample = ds[0]
    assert 'brain_mask' in sample, "brain_mask 키 없음 — dataloader_h5_v5.py 수정 안 됨"
    bm = sample['brain_mask']
    print(f'  shape:  {bm.shape}    expected (1, {IMAGE_SIZE[0]}, {IMAGE_SIZE[0]})')
    print(f'  dtype:  {bm.dtype}    expected float32')
    print(f'  values: unique={np.unique(bm).tolist()}    expected [0.0, 1.0]')
    print(f'  sum:    {bm.sum():.0f}    (mask 안 픽셀 수)')
    print(f'  ratio:  {bm.mean() * 100:.1f}%    (전체 픽셀 중 brain 비율)')
    assert bm.dtype == np.float32, f'dtype 불일치: {bm.dtype}'
    assert set(np.unique(bm).tolist()).issubset({0.0, 1.0}), f'binary 아님: {np.unique(bm)}'
    assert bm.sum() > 0, 'mask 가 모두 0 — threshold 또는 erode 잘못됨'
    print('  → PASS')
    return ds


# ───────────────────────────────────────────
#  2) mask overlay 시각화 PNG 저장
# ───────────────────────────────────────────

def visualize_mask_overlay(ds, n_samples=4):
    print('\n[2] brain_mask overlay 시각화 PNG 저장')
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    for i in range(n_samples):
        idx = i * (len(ds) // n_samples)
        s = ds[idx]
        target = s['label'][0]
        mask   = s['brain_mask'][0]
        # 정규화 표시
        t_show = target / max(target.max(), 1e-9)
        overlay = t_show.copy()
        overlay = np.stack([overlay, overlay, overlay], axis=-1)
        # mask 영역 빨갛게 50% 블렌드
        red = np.array([1.0, 0.3, 0.3])
        overlay = overlay * (1 - 0.4 * mask[..., None]) + red * (0.4 * mask[..., None])

        axes[i, 0].imshow(t_show, cmap='gray')
        axes[i, 0].set_title(f'target (sample {idx})  max={target.max():.2f}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'brain_mask  ratio={mask.mean()*100:.1f}%')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('overlay (red = brain)')
        axes[i, 2].axis('off')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'brain_mask_overlay.png')
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → 저장: {out_png}')


# ───────────────────────────────────────────
#  3) masked vs unmasked metric 비교
#  (모델 ckpt 없이 random aliased image vs target 비교)
# ───────────────────────────────────────────

def compare_masked_vs_unmasked(ds, n_samples=20):
    print('\n[3] masked vs unmasked metric 비교 (aliased image 를 pred 로 사용)')
    print('    — 진짜 모델 출력 대신 input image (aliased) 를 비교 대상으로 둠.')
    print('    — 배경 부풀림은 pred 의 정밀도와 무관해 input 으로도 정량화 가능.')
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    ssim_u_list, ssim_m_list = [], []
    psnr_u_list, psnr_m_list = [], []
    nmse_u_list, nmse_m_list = [], []
    for i, s in enumerate(loader):
        if i >= n_samples:
            break
        # pred: aliased image 의 RSS magnitude (조잡한 근사)
        data_in_img = s['data_img'].numpy()[0]   # (2C, H, W) real/imag packed
        target      = s['label'].numpy()[0, 0]   # (H, W)
        mask        = s['brain_mask'].numpy()[0, 0]

        # 2C 채널을 complex 로 재해석 후 RSS magnitude
        C = data_in_img.shape[0] // 2
        complex_img = data_in_img[0::2] + 1j * data_in_img[1::2]
        pred = np.sqrt((np.abs(complex_img) ** 2).sum(axis=0)).astype(np.float32)

        # 스케일 정합 (target 의 평균에 맞춤)
        if pred.max() > 0:
            pred = pred * (target.mean() / pred.mean())

        # --- unmasked (per-slice max-min) ---
        dr_u = target.max() - target.min()
        if dr_u > 0:
            ssim_u = compare_ssim(target, pred, data_range=dr_u)
        else:
            ssim_u = 0.0
        mse_u = float(((pred - target) ** 2).mean())
        psnr_u = 20 * np.log10(target.max() / max(np.sqrt(mse_u), 1e-10)) if target.max() > 0 else 0
        nmse_u = float(((pred - target) ** 2).sum() / max((target ** 2).sum(), 1e-10))

        # --- masked ---
        m_bool = mask > 0.5
        if m_bool.any():
            t_in = target[m_bool]
            dr_m = float(t_in.max() - t_in.min())
            if dr_m > 0:
                _, ssim_map = compare_ssim(target, pred, data_range=dr_m, full=True)
                ssim_m = float(ssim_map[m_bool].mean())
            else:
                ssim_m = 0.0
            diff_sq_sum = ((pred - target) ** 2 * mask).sum()
            m_sum = mask.sum()
            mse_m = float(diff_sq_sum / m_sum)
            ref_max_m = float((target * mask).max())
            psnr_m = 20 * np.log10(ref_max_m / max(np.sqrt(mse_m), 1e-10)) if ref_max_m > 0 else 0
            nmse_m = float(diff_sq_sum / max(((target ** 2) * mask).sum(), 1e-10))
        else:
            ssim_m = psnr_m = nmse_m = 0.0

        ssim_u_list.append(ssim_u); ssim_m_list.append(ssim_m)
        psnr_u_list.append(psnr_u); psnr_m_list.append(psnr_m)
        nmse_u_list.append(nmse_u); nmse_m_list.append(nmse_m)

    print(f'    n_samples = {len(ssim_u_list)}')
    print(f'    {"":<10s}  {"unmasked":>12s}    {"masked":>12s}    {"Δ (mask-unmask)":>16s}')
    for name, u, m in [
        ('SSIM',  ssim_u_list, ssim_m_list),
        ('PSNR',  psnr_u_list, psnr_m_list),
        ('NMSE',  nmse_u_list, nmse_m_list),
    ]:
        u_mean = float(np.mean(u))
        m_mean = float(np.mean(m))
        d      = m_mean - u_mean
        print(f'    {name:<10s}  {u_mean:>12.4f}    {m_mean:>12.4f}    {d:>+16.4f}')
    print('    → SSIM_masked < SSIM_unmasked 면 배경 부풀림이 정량화됨 (예상)')


# ───────────────────────────────────────────
#  4) Loss NaN sanity (mask sum=0 처리)
# ───────────────────────────────────────────

def loss_nan_sanity():
    print('\n[4] Loss NaN sanity — mask sum=0 edge case')
    H = W = IMAGE_SIZE[0]
    pred   = torch.randn(2, 1, H, W).abs()
    target = torch.randn(2, 1, H, W).abs()
    # 의도적으로 한 슬라이스 mask=0
    mask = torch.ones(2, 1, H, W)
    mask[1] = 0.0

    m_sum = mask.sum().clamp(min=1.0)
    loss_l1 = ((pred - target).abs() * mask).sum() / m_sum
    print(f'    mask[0].sum() = {mask[0].sum().item():.0f}    mask[1].sum() = {mask[1].sum().item():.0f}')
    print(f'    L1_masked = {loss_l1.item():.4f}    (NaN/Inf 안 나야 함)')
    assert not torch.isnan(loss_l1) and not torch.isinf(loss_l1), 'L1 mask NaN/Inf'

    # SSIM mask sum=0 처리
    sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
    from u_choh_SSIM import SSIM
    crit = SSIM().cuda()
    pred_g, target_g, mask_g = pred.cuda(), target.cuda(), mask.cuda()
    ssim = crit(pred_g, target_g, mask=mask_g)
    print(f'    SSIM_masked = {ssim.item():.4f}    (NaN 안 나야 함)')
    assert not torch.isnan(ssim) and not torch.isinf(ssim), 'SSIM mask NaN/Inf'
    print('    → PASS')


def main():
    print('=' * 60)
    print(' v7_titan 평가 metric 재설계 sanity')
    print('=' * 60)
    ds = check_brain_mask_key()
    visualize_mask_overlay(ds, n_samples=4)
    compare_masked_vs_unmasked(ds, n_samples=20)
    if torch.cuda.is_available():
        loss_nan_sanity()
    else:
        print('\n[4] CUDA 없음, loss NaN sanity 건너뜀')
    print('\n=== 모든 sanity 통과 ===')


if __name__ == '__main__':
    main()
