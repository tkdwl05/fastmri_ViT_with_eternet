"""
brain_mask 알고리즘 전체 train + val (~73k slice) 검증.

dataloader_h5_v5.py 의 현재 brain_mask 알고리즘 적용 후 모든 slice 의 mask 통계 수집.

사용:
  python v7_titan/sanity_mask_full_check.py            # train + val 전체
  python v7_titan/sanity_mask_full_check.py val        # val 만
  python v7_titan/sanity_mask_full_check.py train      # train 만

출력 (stdout + 파일):
  - 진행률 (매 1000 sample)
  - 최종 분포 / 이상치 통계 / 통과 비율
  - 결과 파일: v7_titan/runs/sanity_eval/full_mask_check_{split}.txt
"""

import os
import sys
import time
import numpy as np
from collections import Counter
from scipy.ndimage import label as ndi_label

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))

from myConfig_choh_SS2D_model_v7_titan import IMAGE_SIZE
from dataloader_h5_v5 import FastMRI_H5_Dataloader

OUT_DIR = os.path.join(_HERE, 'runs', 'sanity_eval')
os.makedirs(OUT_DIR, exist_ok=True)


def histogram_text(values, bins, label, fp):
    low, high, step = bins
    edges = np.arange(low, high + step, step)
    counts, _ = np.histogram(values, bins=edges)
    max_count = counts.max() if counts.max() > 0 else 1
    width = 40
    line = f'\n  {label}  (n={len(values)}, mean={np.mean(values):.3f}, median={np.median(values):.3f})'
    print(line); fp.write(line + '\n')
    for i, c in enumerate(counts):
        bar = '#' * int(c / max_count * width)
        lo, hi = edges[i], edges[i + 1]
        line = f'    [{lo:6.2f} ~ {hi:6.2f})  {c:6d}  {bar}'
        print(line); fp.write(line + '\n')


def process_split(split, fp):
    print(f'\n==== {split} ====', flush=True)
    fp.write(f'\n==== {split} ====\n')

    t0 = time.time()
    ds = FastMRI_H5_Dataloader(
        f'./fastMRI_data/multicoil_{split}', num_files=None,
        target_size=IMAGE_SIZE[0], augment=False, random_mask=False,
    )
    N = len(ds)
    print(f' dataset 로딩 {time.time() - t0:.1f}s,  total = {N} slice', flush=True)
    fp.write(f' dataset = {N} slice\n')

    ratios     = []
    n_comps    = []
    bbox_hs, bbox_ws = [], []
    rss_max_list     = []
    outliers_small   = []
    outliers_large   = []

    t1 = time.time()
    for k in range(N):
        try:
            s = ds[k]
        except Exception as e:
            line = f'  [skip] idx={k}: {type(e).__name__} {e}'
            print(line, flush=True); fp.write(line + '\n')
            continue
        mask = s['brain_mask'][0]
        label = s['label'][0]
        rss_max_list.append(float(label.max()))

        area = int(mask.sum())
        ratio = area / mask.size
        ratios.append(ratio)

        _, n_cc = ndi_label(mask)
        n_comps.append(int(n_cc))

        if area > 0:
            ys, xs = np.where(mask > 0)
            bbox_hs.append(int(ys.max() - ys.min()))
            bbox_ws.append(int(xs.max() - xs.min()))
        else:
            bbox_hs.append(0)
            bbox_ws.append(0)

        if ratio < 0.10:
            outliers_small.append((k, ratio, float(label.max())))
        elif ratio > 0.55:
            outliers_large.append((k, ratio, float(label.max())))

        if (k + 1) % 2000 == 0 or (k + 1) == N:
            elapsed = time.time() - t1
            eta = elapsed / (k + 1) * (N - k - 1) if (k + 1) < N else 0
            line = f'  진행 {k+1}/{N} ({(k+1)/N*100:.1f}%)  elapsed {elapsed:.0f}s  ETA {eta:.0f}s'
            print(line, flush=True); fp.write(line + '\n')
            fp.flush()

    print(f'\n {split} 총 처리 시간: {time.time() - t1:.1f}s', flush=True)
    fp.write(f'\n {split} 총 처리 시간: {time.time() - t1:.1f}s\n')

    print(f'\n────── {split} 분포 ──────', flush=True); fp.write(f'\n────── {split} 분포 ──────\n')
    histogram_text(np.array(ratios) * 100, (0, 100, 5), 'mask ratio (%)', fp)
    nc_counter = Counter(n_comps)
    line = f'\n  n_components 1 인 비율:  {nc_counter[1] / len(n_comps) * 100:.2f}% ({nc_counter[1]}/{len(n_comps)})'
    print(line, flush=True); fp.write(line + '\n')
    histogram_text(bbox_hs, (0, IMAGE_SIZE[0], IMAGE_SIZE[0] / 20), 'bbox height', fp)
    histogram_text(bbox_ws, (0, IMAGE_SIZE[0], IMAGE_SIZE[0] / 20), 'bbox width', fp)

    line = f'\n  이상치 ratio<10% : {len(outliers_small)} ({len(outliers_small)/len(ratios)*100:.2f}%)'
    print(line, flush=True); fp.write(line + '\n')
    line = f'  이상치 ratio>55%: {len(outliers_large)} ({len(outliers_large)/len(ratios)*100:.2f}%)'
    print(line, flush=True); fp.write(line + '\n')

    pass_count = sum(
        1 for r, nc in zip(ratios, n_comps)
        if 0.15 <= r <= 0.50 and nc == 1
    )
    line = (
        f'\n  통과 (ratio∈[15%,50%] + n_components=1): '
        f'{pass_count}/{len(ratios)} = {pass_count/len(ratios)*100:.2f}%'
    )
    print(line, flush=True); fp.write(line + '\n')

    return {
        'split':       split,
        'N':           N,
        'ratios':      ratios,
        'n_comps':     n_comps,
        'pass_count':  pass_count,
        'outliers_s':  outliers_small,
        'outliers_l':  outliers_large,
    }


def main():
    splits = sys.argv[1:] if len(sys.argv) > 1 else ['val', 'train']
    out_path = os.path.join(OUT_DIR, f'full_mask_check_{"_".join(splits)}.txt')
    fp = open(out_path, 'w', buffering=1)

    print('=' * 60); fp.write('=' * 60 + '\n')
    print(' brain_mask 전체 검증')
    fp.write(' brain_mask 전체 검증\n')
    print(' algorithm: dataloader_h5_v5.py 현재 적용분 (F: Otsu × 0.4 + largest CC, no erode, no fill_holes)')
    fp.write(' algorithm: dataloader_h5_v5.py 현재 적용분 (F: Otsu × 0.4 + largest CC)\n')
    print('=' * 60); fp.write('=' * 60 + '\n')
    print(f' splits = {splits}    output = {out_path}')

    t_start = time.time()
    results = []
    for sp in splits:
        results.append(process_split(sp, fp))

    print('\n' + '=' * 60); fp.write('\n' + '=' * 60 + '\n')
    print('  최종 통합 요약'); fp.write('  최종 통합 요약\n')
    print('=' * 60); fp.write('=' * 60 + '\n')

    total_N    = sum(r['N'] for r in results)
    total_pass = sum(r['pass_count'] for r in results)
    line = f'\n  총 {total_N} slice 중 통과: {total_pass} ({total_pass/total_N*100:.2f}%)'
    print(line, flush=True); fp.write(line + '\n')

    for r in results:
        line = (
            f'    {r["split"]:5s}: {r["pass_count"]:6d}/{r["N"]:6d}  '
            f'({r["pass_count"]/r["N"]*100:.2f}%)  '
            f'이상치 작음 {len(r["outliers_s"])}  큼 {len(r["outliers_l"])}'
        )
        print(line, flush=True); fp.write(line + '\n')

    line = f'\n  총 소요 시간: {time.time() - t_start:.0f}초'
    print(line, flush=True); fp.write(line + '\n')

    fp.close()


if __name__ == '__main__':
    main()
