"""
Zero-filled 기준선 per-slice 평가 (CPU 전용, 모델 없음).

논문 표 관례(MMR-Mamba·HiFi-Mamba 등)의 "Zero-filling" 행을 본 연구 프로토콜과 **완전히 동일한
평가 프로토콜**(384² crop/pad · 16코일 · R=4 equispaced ACS 8% · brain-mask · eval_paired_v8_nodc 와
동일 지표식)로 계산한다. 재구성 = 언더샘플링된 k-space 의 iFFT 코일 영상 RSS(데이터로더의
`data_img` 그대로). 학습·GPU 불필요 → 실행 중인 GPU0 런에 영향 없음.

두 변형을 모두 기록한다:
  - raw     : 스케일 보정 없음 (R=4 undersampling 으로 강도가 ~1/R 로 낮아진 상태)
  - ls      : brain-mask 내 슬라이스별 최소제곱 강도 배율 보정(α=⟨r,g⟩/⟨r,r⟩) — 리더보드(leaderboard) 기준선
              (`eval_paired_baselines.py`)과 동일 처리. 논문 표에는 ls 를 쓰고 각주로 명시.

출력: results/eval/zero_filled/per_slice_zero_filled.csv (idx,file,slice_idx,{raw,ls}_{ssim,psnr,nmse,l1})
      results/eval/zero_filled/zero_filled_summary.md
실행: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 19 python v8_eter_pure/eval_zero_filled_v8.py
"""
import os
import sys
import csv
import argparse

import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
os.chdir(_PROJECT_ROOT)

import myConfig_pure_eter_v8 as C                      # noqa: E402
from dataloader_h5_v5 import FastMRI_H5_Dataloader     # noqa: E402

METRICS = ['ssim', 'psnr', 'nmse', 'l1']


def unpack_complex(packed):
    return packed[0::2].astype(np.float32) + 1j * packed[1::2].astype(np.float32)


def ls_scale(recon, gt, mask):
    m = mask > 0.5
    if not m.any():
        return recon
    r, g = recon[m], gt[m]
    denom = float((r * r).sum())
    if denom < 1e-12:
        return recon
    return (float((r * g).sum()) / denom) * recon


def slice_metrics_np(out, ref, m):
    """eval_paired_v8_nodc.slice_metrics 와 동일 공식 (composite 제외)."""
    m = m.astype(np.float32)
    m_sum = max(float(m.sum()), 1.0)
    diff_sq_sum = float(((out - ref) ** 2 * m).sum())
    mse = diff_sq_sum / m_sum
    ref_max_in_mask = max(float((ref * m).max()), 1e-10)
    psnr = float(20.0 * np.log10(ref_max_in_mask / np.sqrt(max(mse, 1e-10))))
    ref_sq_sum = max(float((ref ** 2 * m).sum()), 1e-10)
    nmse = diff_sq_sum / ref_sq_sum
    mb = m > 0.5
    ssim = 0.0
    if mb.any():
        t_in = ref[mb]
        dr = float(t_in.max() - t_in.min())
        if dr > 0:
            _, smap = compare_ssim(ref, out, data_range=dr, full=True)
            ssim = float(smap[mb].mean())
    l1 = float((np.abs(out - ref) * m).sum() / m_sum)
    return {'ssim': ssim, 'psnr': psnr, 'nmse': nmse, 'l1': l1}


def main():
    p = argparse.ArgumentParser(description='Zero-filled 기준선 per-slice 평가 (CPU)')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--out-dir', default='results/eval/zero_filled')
    p.add_argument('--max-samples', type=int, default=-1)
    args = p.parse_args()

    ds = FastMRI_H5_Dataloader(args.data_path, num_files=None, target_size=C.IMAGE_SIZE[0],
                               acceleration=4, center_fraction=0.08,
                               random_mask=False, augment=False)
    total = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'per_slice_zero_filled.csv')

    rows = []
    for idx in range(total):
        s = ds[idx]
        fp, si, _ = ds.samples[idx]
        img = unpack_complex(s['data_img'])                   # (16, H, W) complex, ×1e6
        zf = np.sqrt((np.abs(img) ** 2).sum(axis=0)).astype(np.float32)
        gt = s['label'][0].astype(np.float32)
        bm = s['brain_mask'][0].astype(np.float32)
        raw = slice_metrics_np(zf, gt, bm)
        ls = slice_metrics_np(ls_scale(zf, gt, bm), gt, bm)
        row = {'idx': idx, 'file': os.path.basename(fp), 'slice_idx': si}
        row.update({f'raw_{k}': raw[k] for k in METRICS})
        row.update({f'ls_{k}': ls[k] for k in METRICS})
        rows.append(row)
        if (idx + 1) % 500 == 0 or idx + 1 == total:
            print(f'  {idx + 1}/{total}  ls_ssim(mean so far)={np.mean([r["ls_ssim"] for r in rows]):.4f}',
                  flush=True)

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    files = np.array([r['file'] for r in rows])
    uf = np.unique(files)
    lines = ['# Zero-filled 기준선 (CPU, 모델 없음) — 본 연구 평가 프로토콜(평가 영역·해상도)',
             f'- 슬라이스 {len(rows):,} / 볼륨 {len(uf)} · R=4 equispaced ACS 8% · 384² · 16코일 · brain-masked',
             '- raw = 스케일 보정 없음, ls = brain-mask 내 슬라이스별 최소제곱 강도 배율 보정(리더보드(leaderboard) 기준선과 동일 처리)',
             '', '| 변형 | 단위 | SSIM | PSNR (dB) | nMSE (비율) | L1 (×10⁻⁶) |', '|---|---|---:|---:|---:|---:|']
    for var in ['raw', 'ls']:
        a = {k: np.array([r[f'{var}_{k}'] for r in rows]) for k in METRICS}
        lines.append(f'| {var} | slice 평균±표준편차(SD) | ' + ' | '.join(
            f'{a[k].mean():.4f}±{a[k].std():.4f}' if k in ('ssim', 'nmse') else f'{a[k].mean():.2f}±{a[k].std():.2f}'
            for k in METRICS) + ' |')
        v = {k: np.array([a[k][files == f].mean() for f in uf]) for k in METRICS}
        lines.append(f'| {var} | volume 평균±표준편차(SD) | ' + ' | '.join(
            f'{v[k].mean():.4f}±{v[k].std():.4f}' if k in ('ssim', 'nmse') else f'{v[k].mean():.2f}±{v[k].std():.2f}'
            for k in METRICS) + ' |')
    open(os.path.join(args.out_dir, 'zero_filled_summary.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
