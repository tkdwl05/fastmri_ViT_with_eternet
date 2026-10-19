"""공식 정의 vs 마스크 정의 — 모델 간 볼륨 단위 짝 비교 (per_volume_<seq>.csv 두 개)."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

d = sys.argv[1]
a, b = sys.argv[2], sys.argv[3]          # 비교 모델, 기준 (Δ = a − b)
A = pd.read_csv(f'{d}/per_volume_{a}.csv').set_index('file')
B = pd.read_csv(f'{d}/per_volume_{b}.csv').set_index('file')
assert list(A.index) == list(B.index), '볼륨 불일치'
rows = [('ssim_mask', '마스크 SSIM (현행)', 1), ('ssim_mask_volrange', '마스크 SSIM, data_range 볼륨 단위', 1),
        ('ssim_off', '공식 SSIM', 1), ('ssim_off_in', '공식 SSIM map — 뇌 마스크 안', 1),
        ('ssim_off_out', '공식 SSIM map — 뇌 마스크 밖', 1), ('psnr_mask', '마스크 PSNR (dB)', 1),
        ('psnr_off', '공식 PSNR (dB)', 1), ('nmse_mask', '마스크 nMSE', -1), ('nmse_off', '공식 NMSE', -1)]
print(f'n = {len(A)} 볼륨, Δ = {a} − {b}')
print(f'| 지표 | {b} | {a} | Δ 평균 | {a} 가 나은 볼륨 | Wilcoxon p |')
print('|---|---|---|---|---|---|')
for k, name, sgn in rows:
    x, y = A[k].to_numpy(float), B[k].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    dd = x[ok] - y[ok]
    better = int(((sgn * dd) > 0).sum())
    p = wilcoxon(dd).pvalue if np.any(dd != 0) else float('nan')
    print(f'| {name} | {y[ok].mean():.4f}±{y[ok].std(ddof=1):.4f} | {x[ok].mean():.4f}±{x[ok].std(ddof=1):.4f} | '
          f'{dd.mean():+.5f} | {better}/{ok.sum()} ({100 * better / ok.sum():.1f}%) | {p:.2g} |')
fi = A['frac_in'].mean()
print(f'\n뇌 마스크 안 화소 비율(공식 틀, 테두리 3 px 제외) 평균 {fi:.3f}')
din = (A['ssim_off_in'] - B['ssim_off_in']).mean()
dout = (A['ssim_off_out'] - B['ssim_off_out']).mean()
print(f'공식 SSIM Δ 근사 분해: 안 {fi:.3f}×{din:+.5f} = {fi * din:+.5f}, 밖 {1 - fi:.3f}×{dout:+.5f} = {(1 - fi) * dout:+.5f}')
