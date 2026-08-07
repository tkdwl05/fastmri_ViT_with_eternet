"""
기준선(fastMRI brain leaderboard U-Net / E2E-VarNet) per-slice 평가 + v9 CSV 조인.

논문 §5.4 기준선 표용. `eval_paired_v9.py` 의 기준선 확장 — 사전학습 U-Net·VarNet 을
우리 val 파이프라인(같은 슬라이스·R4/cf0.08 mask·GT·brain mask·masked 지표)에서 추론하고,
`results/eval/v9_unleashed/per_slice_paired_v9.csv`(gru/ss2d/v9 per-slice, 7334) 와
(file, slice_idx) 키로 조인해 우위 슬라이스 비율 + Wilcoxon 을 산출한다.

베이스라인 캐비엇 (visualize_v7_titan_compare.py 와 동일):
  - U-Net/VarNet 출력은 자체 정규화 스케일 → 지표 계산 전 per-slice LS scale 로 GT 정합
    (우리 모델은 α≈1 이라 미적용 — 3모델 수치는 v9 CSV 재사용).
  - leaderboard 가중치는 전체 코일·native 해상도 학습 → 우리 16-coil·384 전처리와
    domain shift 존재. 절대 우열이 아닌 "동일 측정값에 대한 참고 기준선".
  - VarNet: k-space true ortho scale(~1e-4)이 sens 추정 발산을 유발했던 전례 → unit-max
    정규화로 완화(viz 스크립트 주석 참조). 잔여 non-finite 출력은 제외하되 개수를 보고.

실행 (GPU ~2h 예상; radapt 학습 중에는 실행 금지 — GPU0 단독 정책):
  python v8_eter_pure/eval_paired_baselines.py
CPU 스모크:
  CUDA_VISIBLE_DEVICES="" python v8_eter_pure/eval_paired_baselines.py --max-samples 2 --num-workers 0
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from scipy.stats import wilcoxon

from fastmri.models import VarNet, Unet

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))

import myConfig_pure_eter_v8 as C
from dataloader_h5_v5 import FastMRI_H5_Dataloader

METRICS = ['ssim', 'psnr', 'nmse', 'l1', 'composite']
LOWER_IS_BETTER = {'nmse', 'l1'}
CENTER_FRACTION = 0.08
ACCEL = 4
AMP_X_IMG = 1e6   # dataloader val_amp_X_img


def unpack_complex(packed):
    """(2C,H,W) packed real/imag (real=짝수 idx, imag=홀수 idx) → (C,H,W) complex."""
    return packed[0::2].astype(np.float32) + 1j * packed[1::2].astype(np.float32)


def ls_scale(recon, gt, mask):
    """brain-mask 안에서 α = ⟨recon,gt⟩/⟨recon,recon⟩ 최소제곱 scale 정합."""
    m = mask > 0.5
    if not m.any():
        return recon
    r = recon[m]
    g = gt[m]
    denom = float((r * r).sum())
    if denom < 1e-12:
        return recon
    return (float((r * g).sum()) / denom) * recon


def slice_metrics_np(out, ref, m):
    """eval_paired_v9.slice_metrics 와 동일 공식의 numpy 판 (out/ref/m: (H,W) float32)."""
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
    psnr_n = min(psnr, C.PSNR_NORM) / C.PSNR_NORM
    nmse_n = max(0.0, 1.0 - min(nmse, 1.0))
    composite = (C.COMPOSITE_W_SSIM * ssim + C.COMPOSITE_W_PSNR * psnr_n
                 + C.COMPOSITE_W_NMSE * nmse_n)
    return {'ssim': ssim, 'psnr': psnr, 'nmse': nmse, 'l1': l1, 'composite': composite}


def _load_state(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
    model.load_state_dict(state)
    return model.to(device).eval()


def run_unet(model, s, device):
    """zero-filled RSS → z-score(clamp ±6, fastmri 방식) → U-Net → unnorm. (visualize_v7_titan_compare 동일)"""
    img_c = unpack_complex(s['data_img']) / AMP_X_IMG        # (16,H,W) complex aliased image
    zf_rss = np.sqrt((np.abs(img_c) ** 2).sum(0)).astype(np.float32)
    t = torch.from_numpy(zf_rss).to(device)
    mean = t.mean()
    std = t.std().clamp(min=1e-8)
    x = ((t - mean) / std).clamp(-6.0, 6.0)[None, None]
    with torch.no_grad():
        out = model(x).squeeze()
    return (out * std + mean).float().cpu().numpy()


def run_varnet(model, s, device):
    """masked k-space(unit-max 정규화) + bool mask + n_low → VarNet RSS. (visualize_v7_titan_compare 동일)"""
    ksp_c = unpack_complex(s['data'])                        # (16,H,W) complex masked k-space
    ksp_c = ksp_c / (float(np.abs(ksp_c).max()) + 1e-12)     # sens 추정 발산(ortho ~1e-4 스케일) 완화
    H, W = ksp_c.shape[-2:]
    mk = torch.stack([torch.from_numpy(np.ascontiguousarray(ksp_c.real)),
                      torch.from_numpy(np.ascontiguousarray(ksp_c.imag))], dim=-1)
    mk = mk.unsqueeze(0).float().to(device)                  # (1,16,H,W,2)
    mask_arr = s['mask']
    mask_1d = mask_arr.reshape(-1, mask_arr.shape[-1])[0]    # (W,) undersample pattern
    mask_vn = torch.from_numpy(mask_1d > 0.5).view(1, 1, 1, W, 1).to(device)
    n_low = int(round(W * CENTER_FRACTION))
    with torch.no_grad():
        out = model(mk, mask_vn, num_low_frequencies=n_low)  # (1,H,W)
    return out.squeeze().float().cpu().numpy()


def load_v9_csv(path):
    """v9 조인 CSV → {(file, slice_idx): {gru_*, ss2d_*, v9_*}}."""
    d = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['file'], int(row['slice_idx']))
            d[key] = {f'{arm}_{k}': float(row[f'{arm}_{k}'])
                      for arm in ('gru', 'ss2d', 'v9') for k in METRICS}
    return d


def pair_table(title, a_name, a_vals, b_name, b_vals, total):
    """b 관점 우위 슬라이스 비율(proportion of slices favoring) 표."""
    lines = [f'## {title}', '',
             f'| 지표 | {a_name} mean±std | {b_name} mean±std | {b_name} 우위 슬라이스 비율 | {a_name} 우위 | tie | Wilcoxon p |',
             '|---|---|---|---|---|---|---|']
    for k in METRICS:
        a = np.array(a_vals[k])
        b = np.array(b_vals[k])
        if k in LOWER_IS_BETTER:
            b_wins = int((b < a).sum())
            a_wins = int((a < b).sum())
        else:
            b_wins = int((b > a).sum())
            a_wins = int((a > b).sum())
        ties = total - b_wins - a_wins
        rate = b_wins / total * 100 if total else float('nan')
        try:
            _s, pval = wilcoxon(b, a)
        except ValueError:
            pval = float('nan')
        lines.append(f'| {k} | {a.mean():.4f}±{a.std():.4f} | {b.mean():.4f}±{b.std():.4f} '
                     f'| {rate:.1f}% ({b_wins}/{total}) | {a_wins} | {ties} | {pval:.2e} |')
    return lines


def main():
    p = argparse.ArgumentParser(description='기준선(U-Net/VarNet leaderboard) per-slice 평가 + v9 CSV 조인')
    p.add_argument('--unet-ckpt', default='models/pretrained/brain_leaderboard_state_dict.pt')
    p.add_argument('--varnet-ckpt', default='models/pretrained/varnet_brain_leaderboard_state_dict.pt')
    p.add_argument('--v9-csv', default='results/eval/v9_unleashed/per_slice_paired_v9.csv')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--out-dir', default='results/eval/baselines_384')
    p.add_argument('--max-samples', type=int, default=-1, help='-1 = 전체 val set')
    p.add_argument('--num-workers', type=int, default=4)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 64)
    print(' 기준선 per-slice 평가 — U-Net / E2E-VarNet (leaderboard 사전학습)')
    print(f'  device={device}')
    print('=' * 64)
    if not torch.cuda.is_available():
        print('  [WARN] CUDA 없음 — CPU 진행 (VarNet 12-cascade 는 매우 느림; 스모크 용도)')

    print('\nv9 조인 CSV 로드 중...')
    v9 = load_v9_csv(args.v9_csv)
    print(f'  {args.v9_csv}: {len(v9)} 슬라이스')

    print('\n기준선 모델 로드 중...')
    unet = _load_state(Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0),
                       args.unet_ckpt, device)
    varnet = _load_state(VarNet(num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4),
                         args.varnet_ckpt, device)
    print(f'  U-Net : {args.unet_ckpt}')
    print(f'  VarNet: {args.varnet_ckpt}')

    ds = FastMRI_H5_Dataloader(args.data_path, num_files=None, target_size=C.IMAGE_SIZE[0],
                               acceleration=ACCEL, center_fraction=CENTER_FRACTION,
                               random_mask=False, augment=False)
    total = len(ds)
    if args.max_samples > 0:
        total = min(total, args.max_samples)
    print(f'\n평가 대상: {total} / {len(ds)} 슬라이스\n')

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'per_slice_baselines.csv')
    summary_path = os.path.join(args.out_dir, 'baseline_summary.md')

    vals = {m: {k: [] for k in METRICS} for m in ('unet', 'varnet', 'gru', 'ss2d', 'v9')}
    ours_vs_varnet = {m: {k: [] for k in METRICS} for m in ('ss2d', 'v9')}
    rows = []
    unmatched = 0
    varnet_nonfinite = 0

    it = iter(loader)
    for idx in tqdm(range(total), desc='baseline eval', unit='slice'):
        batch = next(it)
        s = {k: v[0].numpy() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        gt = s['label'].squeeze().astype(np.float32)          # (H,W)
        brain = s['brain_mask'].squeeze().astype(np.float32)  # (H,W)

        file_path, slice_idx, _ = ds.samples[idx]
        key = (os.path.basename(file_path), slice_idx)
        v9row = v9.get(key)
        if v9row is None:
            unmatched += 1
            continue

        row = {'idx': idx, 'file': key[0], 'slice_idx': slice_idx}

        u_out = run_unet(unet, s, device)
        um = slice_metrics_np(ls_scale(u_out, gt, brain), gt, brain)

        v_out = run_varnet(varnet, s, device)
        v_finite = bool(np.isfinite(v_out).all())
        if v_finite:
            vm = slice_metrics_np(ls_scale(v_out, gt, brain), gt, brain)
        else:
            varnet_nonfinite += 1
            vm = None

        for k in METRICS:
            row[f'unet_{k}'] = um[k]
            row[f'varnet_{k}'] = vm[k] if vm else ''
            for arm in ('gru', 'ss2d', 'v9'):
                row[f'{arm}_{k}'] = v9row[f'{arm}_{k}']
                vals[arm][k].append(v9row[f'{arm}_{k}'])
            vals['unet'][k].append(um[k])
            if vm:
                vals['varnet'][k].append(vm[k])
                for arm in ('ss2d', 'v9'):
                    ours_vs_varnet[arm][k].append(v9row[f'{arm}_{k}'])
        rows.append(row)

    matched = len(rows)
    fieldnames = ['idx', 'file', 'slice_idx']
    for m in ('unet', 'varnet', 'gru', 'ss2d', 'v9'):
        fieldnames += [f'{m}_{k}' for k in METRICS]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_varnet = len(vals['varnet']['ssim'])
    lines = ['# 기준선 per-slice 비교 — U-Net / E2E-VarNet (leaderboard) vs GRU / v8-SS2D / v9', '',
             f'- 평가 슬라이스: {total} / v9 CSV 조인 매칭: {matched} (미매칭 {unmatched})',
             f'- VarNet non-finite 출력: **{varnet_nonfinite}** 슬라이스 (VarNet 통계·paired 에서만 제외, CSV 에는 빈칸)',
             '- 기준선 출력은 per-slice LS scale 로 GT 정합 후 지표 계산 (우리 3모델은 α≈1, v9 CSV 재사용)',
             '- ⚠ leaderboard 가중치는 전체 코일·native 해상도 학습 → 16-coil·384 전처리와 domain shift.',
             '  절대 우열이 아닌 "동일 측정값에 대한 참고 기준선" (visualize_v7_titan_compare.py 캐비엇 동일)', '',
             '## 전체 평균 (matched 슬라이스; varnet 은 finite 만)', '',
             '| 모델 | ' + ' | '.join(METRICS) + ' |',
             '|---|' + '---|' * len(METRICS)]
    for m in ('unet', 'varnet', 'gru', 'ss2d', 'v9'):
        mv = vals[m]
        if len(mv['ssim']) == 0:
            continue
        lines.append(f'| {m} | ' + ' | '.join(f'{np.mean(mv[k]):.4f}' for k in METRICS) + ' |')
    lines.append('')
    if n_varnet:
        lines += pair_table(f'v9 vs VarNet (finite {n_varnet})', 'VarNet', vals['varnet'],
                            'v9', ours_vs_varnet['v9'], n_varnet) + ['']
        lines += pair_table(f'v8-SS2D vs VarNet (finite {n_varnet})', 'VarNet', vals['varnet'],
                            'v8-SS2D', ours_vs_varnet['ss2d'], n_varnet) + ['']
    lines += pair_table('v9 vs U-Net', 'U-Net', vals['unet'], 'v9', vals['v9'], matched)
    lines += ['', '(우위 슬라이스 비율 = proportion of slices favoring, probabilistic index — '
              'nmse/l1 은 낮을수록 승리. 논문 표기는 p<0.001 관례, 원값은 본 파일 보존)']

    msg = '\n'.join(lines)
    print('\n' + msg)
    with open(summary_path, 'w') as f:
        f.write(msg + '\n')

    print(f'\nCSV: {csv_path}')
    print(f'요약: {summary_path}')
    if unmatched:
        print(f'[WARN] v9 CSV 미매칭 {unmatched} 슬라이스 — dataset 정합 확인 필요')


if __name__ == '__main__':
    main()
