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
sys.path.append(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))

import myConfig_pure_eter_v8 as C
from dataloader_h5_v5 import FastMRI_H5_Dataloader
import native_protocol as NP

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
    """masked k-space + bool mask + n_low → VarNet RSS. (fastmri 공식 추론 규약 대조 완료)

    ⚠ zero-coil 채널 제거가 필수다. 데이터로더는 코일을 16채널로 zero-pad 하는데
    (val 464 파일 중 197개 = 슬라이스 3122/7334 = 42.6% 가 16코일 미만), VarNet 의
    SensitivityModel 은 코일마다 NormUnet 을 개별로 태우므로 all-zero 채널에서
    std=0 → (x-0)/0 = NaN 이 되고, divide_root_sum_of_squares 가 이를 전 코일로
    전파해 슬라이스 전체가 NaN 이 된다. 실측 확인: 4코일 원본 finite / 16채널
    zero-pad 입력 sens NaN 100%.

    unit-max 정규화는 안전장치로 유지하되 지표에는 영향이 없다 — VarNet 은
    NormUnet(mean/std 정규화 후 복원) + 선형 DC + RSS 구조라 양의 스칼라에 대해
    positively homogeneous: VarNet(a·k) = a·VarNet(k). 이후 per-slice LS scale
    정합까지 거치므로 스케일 선택은 결과를 바꾸지 않는다.
    """
    ksp_c = unpack_complex(s['data'])                        # (16,H,W) complex masked k-space
    keep = np.abs(ksp_c).reshape(ksp_c.shape[0], -1).sum(1) > 0
    ksp_c = ksp_c[keep]                                      # zero-pad 채널 제거 (NaN 방지)
    ksp_c = ksp_c / (float(np.abs(ksp_c).max()) + 1e-12)     # no-op (위 homogeneity) — 안전장치
    H, W = ksp_c.shape[-2:]
    mk = torch.stack([torch.from_numpy(np.ascontiguousarray(ksp_c.real)),
                      torch.from_numpy(np.ascontiguousarray(ksp_c.imag))], dim=-1)
    mk = mk.unsqueeze(0).float().to(device)                  # (1,C_keep,H,W,2)
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


def stratified_indices(ds, n, seed=0):
    """(contrast, 코일수 구간) 층화 표본 — 각 층에서 슬라이스 수에 비례 배분, 결정론적."""
    import h5py as _h5
    meta = {}
    strata = {}
    for i, (fp, si, _) in enumerate(ds.samples):
        if fp not in meta:
            with _h5.File(fp, 'r') as f:
                acq = f.attrs['acquisition']
                acq = acq.decode() if isinstance(acq, bytes) else str(acq)
                meta[fp] = (acq, int(f['kspace'].shape[1]))
        acq, coils = meta[fp]
        bucket = '<16' if coils < 16 else ('16' if coils == 16 else '>16')
        strata.setdefault((acq, bucket), []).append(i)

    total = len(ds.samples)
    rng = np.random.default_rng(seed)
    picked = []
    for key in sorted(strata):
        idxs = strata[key]
        k = max(1, int(round(n * len(idxs) / total)))
        k = min(k, len(idxs))
        picked += list(rng.choice(idxs, size=k, replace=False))
    picked = sorted(int(i) for i in picked)
    return picked, meta


def main():
    p = argparse.ArgumentParser(description='기준선(U-Net/VarNet leaderboard) per-slice 평가 + v9 CSV 조인')
    p.add_argument('--unet-ckpt', default='models/pretrained/brain_leaderboard_state_dict.pt')
    p.add_argument('--varnet-ckpt', default='models/pretrained/varnet_brain_leaderboard_state_dict.pt')
    p.add_argument('--v9-csv', default='results/eval/v9_unleashed/per_slice_paired_v9.csv')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--out-dir', default='results/eval/baselines_384')
    p.add_argument('--max-samples', type=int, default=-1, help='-1 = 전체 val set (앞에서부터)')
    p.add_argument('--sample-n', type=int, default=-1,
                   help='층화 표본 크기 (contrast × 코일수 구간). -1 = 미사용')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-native', action='store_true',
                   help='네이티브 프로토콜(전체 코일·native 해상도·공식 crop) 행 생략')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--torch-threads', type=int, default=0, help='>0 이면 torch CPU 스레드 제한')
    args = p.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    do_native = not args.no_native
    print('=' * 64)
    print(' 기준선 per-slice 평가 — U-Net / E2E-VarNet (leaderboard 사전학습)')
    print(f'  device={device}  native-protocol={do_native}  threads={torch.get_num_threads()}')
    print('=' * 64)
    if not torch.cuda.is_available():
        print('  [WARN] CUDA 없음 — CPU 진행 (VarNet 12-cascade 는 매우 느림)')

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

    meta = {}
    if args.sample_n > 0:
        indices, meta = stratified_indices(ds, args.sample_n, args.seed)
        print(f'\n층화 표본: {len(indices)} / {len(ds)} 슬라이스 (contrast × 코일수, seed={args.seed})')
    else:
        indices = list(range(len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)))
        print(f'\n평가 대상: {len(indices)} / {len(ds)} 슬라이스')

    subset = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'per_slice_baselines.csv')
    summary_path = os.path.join(args.out_dir, 'baseline_summary.md')

    ARMS = ['unet', 'varnet', 'varnet_natframe', 'unet_native', 'varnet_native', 'gru', 'ss2d', 'v9']
    fieldnames = ['idx', 'file', 'slice_idx', 'acquisition', 'coils']
    for _m in ARMS:
        fieldnames += [f'{_m}_{k}' for k in METRICS]
    fieldnames += ['native_error']
    csv_f = open(csv_path, 'w', newline='')          # 중단돼도 부분 결과가 남도록 슬라이스마다 flush
    csv_w = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction='ignore')
    csv_w.writeheader()
    vals = {m: {k: [] for k in METRICS} for m in ARMS}
    # 네이티브 VarNet 이 finite 인 슬라이스만 모은 우리 모델 값 (paired 비교용)
    paired_nat = {m: {k: [] for k in METRICS} for m in ('ss2d', 'v9', 'varnet_native')}
    rows = []
    unmatched = 0
    varnet_nonfinite = 0
    varnet_nat_nonfinite = 0

    for pos, batch in enumerate(tqdm(loader, total=len(indices), desc='baseline eval', unit='slice')):
        idx = indices[pos]
        s = {k: v[0].numpy() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        gt = s['label'].squeeze().astype(np.float32)          # (384,384)
        brain = s['brain_mask'].squeeze().astype(np.float32)

        file_path, slice_idx, _ = ds.samples[idx]
        key = (os.path.basename(file_path), slice_idx)
        v9row = v9.get(key)
        if v9row is None:
            unmatched += 1
            continue

        acq, coils = meta.get(file_path, ('', -1))
        row = {'idx': idx, 'file': key[0], 'slice_idx': slice_idx, 'acquisition': acq, 'coils': coils}
        per_slice = {}

        # ── (1) 우리 파이프라인 (16코일 절단 · 384 재-FFT · 384 프레임 지표)
        u_out = run_unet(unet, s, device)
        per_slice['unet'] = slice_metrics_np(ls_scale(u_out, gt, brain), gt, brain)

        v_out = run_varnet(varnet, s, device)
        if np.isfinite(v_out).all():
            per_slice['varnet'] = slice_metrics_np(ls_scale(v_out, gt, brain), gt, brain)
        else:
            varnet_nonfinite += 1

        # ── (2) 같은 재구성을 native recon 프레임으로 crop 한 지표 (프레임 효과 분리용)
        # ── (3) 네이티브 프로토콜: 전체 코일 · native k-space · 공식 crop
        if do_native:
            try:
                vn_out, vn_gt, vn_mask = NP.run_varnet_native(varnet, file_path, slice_idx, device,
                                                              CENTER_FRACTION, ACCEL)
                if np.isfinite(vn_out).all():
                    per_slice['varnet_native'] = slice_metrics_np(
                        ls_scale(vn_out, vn_gt, vn_mask), vn_gt, vn_mask)
                else:
                    varnet_nat_nonfinite += 1

                un_out, un_gt, un_mask = NP.run_unet_native(unet, file_path, slice_idx, device,
                                                            CENTER_FRACTION, ACCEL)
                per_slice['unet_native'] = slice_metrics_np(
                    ls_scale(un_out, un_gt, un_mask), un_gt, un_mask)

                if 'varnet' in per_slice:
                    shp = vn_gt.shape
                    v_c = NP.center_crop_np(v_out, shp)
                    g_c = NP.center_crop_np(gt, shp)
                    m_c = NP.center_crop_np(brain, shp)
                    per_slice['varnet_natframe'] = slice_metrics_np(ls_scale(v_c, g_c, m_c), g_c, m_c)
            except Exception as e:                     # 개별 슬라이스 실패는 건너뛰고 기록
                row['native_error'] = repr(e)[:120]

        # ── 우리 3모델 (v9 CSV 재사용, 384 프레임)
        for arm in ('gru', 'ss2d', 'v9'):
            per_slice[arm] = {k: v9row[f'{arm}_{k}'] for k in METRICS}

        for arm in ARMS:
            m = per_slice.get(arm)
            for k in METRICS:
                row[f'{arm}_{k}'] = m[k] if m else ''
            if m:
                for k in METRICS:
                    vals[arm][k].append(m[k])
        if 'varnet_native' in per_slice:
            for arm in ('ss2d', 'v9'):
                for k in METRICS:
                    paired_nat[arm][k].append(per_slice[arm][k])
            for k in METRICS:
                paired_nat['varnet_native'][k].append(per_slice['varnet_native'][k])
        rows.append(row)
        csv_w.writerow(row)
        csv_f.flush()

    csv_f.close()
    matched = len(rows)

    n_nat = len(vals['varnet_native']['ssim'])
    lines = ['# 기준선 per-slice 비교 — U-Net / E2E-VarNet (leaderboard) vs GRU / v8-SS2D / v9', '',
             f'- 평가 슬라이스: {matched} (미매칭 {unmatched})'
             + (f' · 층화 표본 n={args.sample_n} 요청, seed={args.seed}' if args.sample_n > 0 else ''),
             f'- VarNet non-finite: 우리 파이프라인 {varnet_nonfinite} · 네이티브 {varnet_nat_nonfinite} 슬라이스',
             '- 기준선 출력은 per-slice LS scale 로 GT 정합 후 지표 계산 (우리 3모델은 α≈1, v9 CSV 재사용)',
             '',
             '## 행 정의',
             '',
             '| 행 | 입력 | 프레임 | 비고 |',
             '|---|---|---|---|',
             '| `unet` / `varnet` | 우리 파이프라인(16코일 절단 · 384² 재-FFT) | 384² | 우리 모델과 **완전히 동일한 측정값** |',
             '| `varnet_natframe` | 위와 같은 재구성 | native recon crop | 프레임 효과만 분리 (`varnet` 과의 차이 = 프레임) |',
             '| `unet_native` / `varnet_native` | **공식 규약**(전체 코일 · native k-space · 헤더 crop) | native recon | leaderboard 가중치의 학습 조건에 가장 가까움 |',
             '| `gru` / `ss2d` / `v9` | 우리 파이프라인 | 384² | v9 per-slice CSV |',
             '',
             '- ⚠ **leaderboard 가중치는 train+val 합본으로 학습**(fastMRI 공식 README: "The leaderboard',
             '  model was trained where the `train` split included both the `train` and `val` splits from',
             '  the public data") — 즉 **본 검증셋 전체가 두 기준선의 학습 데이터**다. 기준선 수치는',
             '  낙관적으로 편향돼 있으며, 우리 모델(train 만 학습)과의 직접 우열 판정은 성립하지 않는다.',
             '',
             '## 전체 평균', '',
             '| 모델 | n | ' + ' | '.join(METRICS) + ' |',
             '|---|---|' + '---|' * len(METRICS)]
    for m in ARMS:
        mv = vals[m]
        if len(mv['ssim']) == 0:
            continue
        lines.append(f'| {m} | {len(mv["ssim"])} | '
                     + ' | '.join(f'{np.mean(mv[k]):.4f}' for k in METRICS) + ' |')
    lines.append('')

    n_varnet = len(vals['varnet']['ssim'])
    if n_varnet:
        ours = {arm: {k: [] for k in METRICS} for arm in ('ss2d', 'v9')}
        for r in rows:
            if r.get('varnet_ssim') != '':
                for arm in ('ss2d', 'v9'):
                    for k in METRICS:
                        ours[arm][k].append(r[f'{arm}_{k}'])
        lines += pair_table(f'v9 vs VarNet — 우리 파이프라인 (n={n_varnet})', 'VarNet', vals['varnet'],
                            'v9', ours['v9'], n_varnet) + ['']
        lines += pair_table(f'v8-SS2D vs VarNet — 우리 파이프라인 (n={n_varnet})', 'VarNet', vals['varnet'],
                            'v8-SS2D', ours['ss2d'], n_varnet) + ['']
    if n_nat:
        lines += pair_table(f'v9 vs VarNet — 네이티브 프로토콜 (n={n_nat})', 'VarNet_native',
                            paired_nat['varnet_native'], 'v9', paired_nat['v9'], n_nat) + ['']
        lines += pair_table(f'v8-SS2D vs VarNet — 네이티브 프로토콜 (n={n_nat})', 'VarNet_native',
                            paired_nat['varnet_native'], 'v8-SS2D', paired_nat['ss2d'], n_nat) + ['']
    lines += pair_table(f'v9 vs U-Net — 우리 파이프라인 (n={matched})', 'U-Net', vals['unet'],
                        'v9', vals['v9'], matched)
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
