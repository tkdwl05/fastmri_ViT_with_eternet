"""
brain_mask 알고리즘 random N sample 검증.

새 Otsu 알고리즘 적용한 dataloader 에서 random N sample 추출, mask 통계 분포 측정.

사용:
  python v7_titan/sanity_mask_random_n.py            # default N=500
  python v7_titan/sanity_mask_random_n.py 1000       # N=1000
  python v7_titan/sanity_mask_random_n.py 500 train  # train split 사용

출력:
  - mask ratio 분포 (히스토그램)
  - n_components 분포 (largest CC 효과 검증, 기대 = 거의 모두 1)
  - centroid / bbox 분포
  - 이상치 sample (ratio<10% 또는 >55%) idx 목록 + 통계
"""

import os
import sys
import time
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from scipy.ndimage import label as ndi_label

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))

from myConfig_choh_SS2D_model_v7_titan import IMAGE_SIZE
from dataloader_h5_v5 import FastMRI_H5_Dataloader


def histogram_text(values, bins, label):
    """텍스트 히스토그램. bins = (low, high, step)."""
    low, high, step = bins
    edges = np.arange(low, high + step, step)
    counts, _ = np.histogram(values, bins=edges)
    max_count = counts.max() if counts.max() > 0 else 1
    width = 40
    print(f'\n  {label}  (n={len(values)}, mean={np.mean(values):.3f}, median={np.median(values):.3f})')
    for i, c in enumerate(counts):
        bar = '#' * int(c / max_count * width)
        lo, hi = edges[i], edges[i + 1]
        print(f'    [{lo:6.2f} ~ {hi:6.2f})  {c:5d}  {bar}')


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    split = sys.argv[2] if len(sys.argv) > 2 else 'val'
    assert split in ('train', 'val')

    rng = np.random.default_rng(2026)

    print('=' * 60)
    print(f' brain_mask random {n_samples} sample 검증 (split={split})')
    print(f' 알고리즘: Otsu threshold + largest CC + erode 1px (no fill_holes)')
    print('=' * 60)

    t0 = time.time()
    ds = FastMRI_H5_Dataloader(
        f'./fastMRI_data/multicoil_{split}', num_files=None,
        target_size=IMAGE_SIZE[0], augment=False, random_mask=False,
    )
    N = len(ds)
    n_samples = min(n_samples, N)
    indices = rng.choice(N, size=n_samples, replace=False)
    print(f' 전체 dataset = {N}, sampled = {n_samples} (시드 2026)')
    print(f' dataset 로딩 시간: {time.time() - t0:.1f}s')

    ratios     = []
    n_comps    = []
    cy_list, cx_list = [], []
    bbox_hs, bbox_ws = [], []
    rss_max_list = []
    thr_list = []
    outliers_small = []   # ratio < 10%
    outliers_large = []   # ratio > 55%

    t1 = time.time()
    for k, idx in enumerate(indices):
        try:
            s = ds[int(idx)]
        except Exception as e:
            print(f'  [skip] idx={idx}: {type(e).__name__} {e}')
            continue
        mask = s['brain_mask'][0]
        label = s['label'][0]
        # val_amp_Y = 1e6 곱해진 상태 — 원본 RSS 의 1e6 배수. 분석엔 그대로 OK
        rss_max_list.append(float(label.max()))

        area = int(mask.sum())
        ratio = area / mask.size
        ratios.append(ratio)

        lbl_arr, n_cc = ndi_label(mask)
        n_comps.append(int(n_cc))

        if area > 0:
            ys, xs = np.where(mask > 0)
            cy_list.append(float(ys.mean()))
            cx_list.append(float(xs.mean()))
            bbox_hs.append(int(ys.max() - ys.min()))
            bbox_ws.append(int(xs.max() - xs.min()))
        else:
            cy_list.append(float('nan'))
            cx_list.append(float('nan'))
            bbox_hs.append(0)
            bbox_ws.append(0)

        if ratio < 0.10:
            outliers_small.append((int(idx), ratio, float(label.max()), int(n_cc)))
        elif ratio > 0.55:
            outliers_large.append((int(idx), ratio, float(label.max()), int(n_cc)))

        if (k + 1) % 50 == 0:
            elapsed = time.time() - t1
            eta = elapsed / (k + 1) * (n_samples - k - 1)
            print(f'  진행 {k+1}/{n_samples}  (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)')

    print(f'\n 총 처리 시간: {time.time() - t1:.1f}s ({(time.time() - t1) / n_samples * 1000:.1f}ms/sample)')

    print('\n────── 분포 ──────')
    histogram_text(np.array(ratios) * 100, (0, 100, 5), 'mask ratio (%)')
    n_comp_counter = Counter(n_comps)
    print('\n  n_components  분포 (largest CC 효과 검증, 기대=거의 모두 1)')
    for k in sorted(n_comp_counter.keys())[:10]:
        print(f'    n_components={k:3d}    {n_comp_counter[k]:5d} samples')
    histogram_text(cy_list, (0, IMAGE_SIZE[0], IMAGE_SIZE[0] / 20), 'centroid y')
    histogram_text(cx_list, (0, IMAGE_SIZE[0], IMAGE_SIZE[0] / 20), 'centroid x')
    histogram_text(bbox_hs, (0, IMAGE_SIZE[0], IMAGE_SIZE[0] / 20), 'bbox height')
    histogram_text(bbox_ws, (0, IMAGE_SIZE[0], IMAGE_SIZE[0] / 20), 'bbox width')

    print(f'\n────── 이상치 ──────')
    print(f'  ratio < 10% : {len(outliers_small)} samples ({len(outliers_small)/n_samples*100:.1f}%)')
    for idx, r, mx, nc in outliers_small[:10]:
        print(f'    idx={idx:6d}  ratio={r*100:.1f}%  rss_max={mx:.4g}  n_cc={nc}')
    if len(outliers_small) > 10:
        print(f'    ... + {len(outliers_small) - 10} more')

    print(f'\n  ratio > 55%: {len(outliers_large)} samples ({len(outliers_large)/n_samples*100:.1f}%)')
    for idx, r, mx, nc in outliers_large[:10]:
        print(f'    idx={idx:6d}  ratio={r*100:.1f}%  rss_max={mx:.4g}  n_cc={nc}')
    if len(outliers_large) > 10:
        print(f'    ... + {len(outliers_large) - 10} more')

    print('\n────── summary ──────')
    print(f'  ratio:        min={min(ratios)*100:.1f}% median={np.median(ratios)*100:.1f}% max={max(ratios)*100:.1f}%')
    print(f'  n_components 1 인 비율:  {n_comp_counter[1]/n_samples*100:.1f}%')
    print(f'  이상치 (ratio<10% or >55%):  {(len(outliers_small)+len(outliers_large))/n_samples*100:.1f}%')

    print(f'\n  통과 기준: ratio ∈ [15%, 50%] 이고 n_components == 1 인 비율이 95%+')
    pass_count = sum(
        1 for r, nc in zip(ratios, n_comps)
        if 0.15 <= r <= 0.50 and nc == 1
    )
    print(f'  통과 비율:  {pass_count}/{n_samples} = {pass_count/n_samples*100:.1f}%')


if __name__ == '__main__':
    main()
