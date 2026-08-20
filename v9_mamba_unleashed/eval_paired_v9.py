"""
v9 unleashed — per-slice 평가 + 기존 v8 per-slice CSV 조인 (v9 vs v8-SS2D / v9 vs GRU).

`v8_eter_pure/eval_paired_v8_nodc.py` 의 v9 확장. v9 추론만 새로 수행하고(단일 모델),
v8 두 arm 의 per-slice 수치는 이미 검증·저장된 `results/eval/v8_nodc/per_slice_paired.csv`
(전체 val 7334 슬라이스, 동일 val set·동일 지표 공식) 를 (file, slice_idx) 키로 재사용한다
— v8 재추론 불필요. paired win-rate + Wilcoxon 으로 v9 의 로그 우위(+0.0003 comp)가
슬라이스 단위에서도 성립하는지 검증한다.

sanity 앵커: v9 per-slice **ssim 평균**이 학습 로그 best ckpt(ep78) val_ssim_m(0.9145)과
일치해야 정상 — SSIM 은 학습 val 도 슬라이스 단위 계산이라 정확히 재현된다(실측 일치 확인).
composite/psnr 의 per-slice 평균은 학습 로그(0.9203/35.18)보다 낮게 나오는 것이 **정상**:
학습 val 은 BS=4 배치풀링(배치 공유 ref-max, MSE 합산 후 log)이라 절대값이 다르다.
v8 도 동일 오프셋(CSV composite 0.9104 vs 로그 0.9200) — paired 비교는 3모델 동일
프로토콜이므로 유효.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))

import myConfig_ss2d_v9 as C
from dataloader_h5_v5 import FastMRI_H5_Dataloader

METRICS = ['ssim', 'psnr', 'nmse', 'l1', 'composite']
LOWER_IS_BETTER = {'nmse', 'l1'}


def skimage_ssim_masked(pred, target, mask):
    """main_train_ss2d_v9.skimage_ssim_batch_masked 의 단일-슬라이스(배치=1) 버전."""
    p = pred.detach().float().cpu().numpy()
    t = target.detach().float().cpu().numpy()
    m = mask.detach().float().cpu().numpy()
    if p.ndim == 4:
        p, t, m = p[:, 0], t[:, 0], m[:, 0]
    m_bool = m > 0.5
    if not m_bool[0].any():
        return 0.0
    t_in = t[0][m_bool[0]]
    dr = float(t_in.max() - t_in.min())
    if dr <= 0:
        return 0.0
    _, ssim_map = compare_ssim(t[0], p[0], data_range=dr, full=True)
    return float(ssim_map[m_bool[0]].mean())


def slice_metrics(out, ref, mask):
    """main_train_ss2d_v9.run_val() 의 텐서 공식을 슬라이스(배치=1) 단위로 그대로 적용."""
    out_f = out.float()
    ref_f = ref.float()
    m = mask
    m_sum = m.sum().clamp(min=1.0)
    diff_sq_sum = ((out_f - ref_f) ** 2 * m).sum()
    mse = diff_sq_sum / m_sum
    ref_max_in_mask = (ref_f * m).max().clamp(min=1e-10)
    psnr = (20 * torch.log10(ref_max_in_mask / torch.sqrt(mse.clamp(min=1e-10)))).item()
    ref_sq_sum = (ref_f ** 2 * m).sum().clamp(min=1e-10)
    nmse = (diff_sq_sum / ref_sq_sum).item()
    ssim = skimage_ssim_masked(out_f, ref_f, m)
    l1 = (((out_f - ref_f).abs() * m).sum() / m_sum).item()
    psnr_n = min(psnr, C.PSNR_NORM) / C.PSNR_NORM
    nmse_n = max(0.0, 1.0 - min(nmse, 1.0))
    composite = C.COMPOSITE_W_SSIM * ssim + C.COMPOSITE_W_PSNR * psnr_n + C.COMPOSITE_W_NMSE * nmse_n
    return {'ssim': ssim, 'psnr': psnr, 'nmse': nmse, 'l1': l1, 'composite': composite}


def build_model(device):
    """main_train_ss2d_v9.build_model 과 동일 kwargs (config ground-truth)."""
    from u_pure_eternet_ss2d_v9 import PureETER_SS2D_V9
    model = PureETER_SS2D_V9(
        n_coil=C.N_COIL, out_ch=C.SS2D_OUT_CH,
        unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
        ss2d_d_inner=C.SS2D_D_INNER, ss2d_d_state=C.SS2D_D_STATE,
        ss2d_n_blocks=C.SS2D_N_BLOCKS, ss2d_dropout=C.SS2D_DROPOUT,
        ss2d_use_checkpoint=C.SS2D_USE_CHECKPOINT, ss2d_downsample=C.SS2D_DOWNSAMPLE,
    )
    return model.to(device)


def load_ckpt(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_v8_csv(path):
    """v8 per-slice CSV → {(file, slice_idx): {gru_*, ss2d_*}} (수치는 float 파싱)."""
    v8 = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['file'], int(row['slice_idx']))
            v8[key] = {f'{arm}_{k}': float(row[f'{arm}_{k}'])
                       for arm in ('gru', 'ss2d') for k in METRICS}
    return v8


def pair_table(title, a_name, a_vals, b_name, b_vals, total):
    """b(=v9) 관점 win-rate 표. a=상대(v8 arm)."""
    lines = [f'## {title}', '',
             f'| 지표 | {a_name} mean±std | {b_name} mean±std | {b_name} win-rate | {a_name} win | tie | Wilcoxon p |',
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
        win_rate = b_wins / total * 100
        try:
            _stat, pval = wilcoxon(b, a)
        except ValueError:
            pval = float('nan')
        lines.append(f'| {k} | {a.mean():.4f}±{a.std():.4f} | {b.mean():.4f}±{b.std():.4f} '
                     f'| {win_rate:.1f}% ({b_wins}/{total}) | {a_wins} | {ties} | {pval:.2e} |')
    return lines


def main():
    p = argparse.ArgumentParser(description='v9 unleashed per-slice 평가 + v8 CSV 조인 paired 비교')
    p.add_argument('--ckpt', default='logs/PureETER_SS2D_V9_unleashed_R4_brain384/ss2d_v9_best.pt')
    p.add_argument('--v8-csv', default='results/eval/v8_nodc/per_slice_paired.csv')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--out-dir', default='results/eval/v9_unleashed')
    p.add_argument('--max-samples', type=int, default=-1, help='-1 = 전체 val set')
    p.add_argument('--num-workers', type=int, default=4,
                   help='컨테이너 재생성(shm 128g) 후 multi-worker 안전 — v8 스크립트의 0 고정은 옛 shm 64MB 시절')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 64)
    print(' v9 unleashed — per-slice 평가 (+ v8 no-DC CSV 조인 paired 비교)')
    print(f'  device={device}')
    print('=' * 64)
    if not torch.cuda.is_available():
        print('  [WARN] CUDA 없음 — CPU 로 진행 (매우 느릴 수 있음)')

    print('\nv8 per-slice CSV 로드 중...')
    v8 = load_v8_csv(args.v8_csv)
    print(f'  {args.v8_csv}: {len(v8)} 슬라이스')

    print('\nv9 모델 로드 중...')
    model = load_ckpt(build_model(device), args.ckpt, device)
    print(f'  ckpt: {args.ckpt}')

    # v8 eval 과 동일 dataset 인자 (조인 정합의 전제)
    ds = FastMRI_H5_Dataloader(args.data_path, num_files=None, target_size=C.IMAGE_SIZE[0],
                               acceleration=4, center_fraction=0.08,
                               random_mask=False, augment=False)
    total = len(ds)
    if args.max_samples > 0:
        total = min(total, args.max_samples)
    print(f'\n평가 대상: {total} / {len(ds)} 슬라이스\n')

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'per_slice_paired_v9.csv')
    summary_path = os.path.join(args.out_dir, 'win_rate_summary_v9.md')

    v9_vals = {k: [] for k in METRICS}
    gru_vals = {k: [] for k in METRICS}
    ss2d_vals = {k: [] for k in METRICS}
    rows = []
    unmatched = 0

    it = iter(loader)
    with torch.no_grad():
        for idx in tqdm(range(total), desc='v9 eval', unit='slice'):
            sample = next(it)
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            brain_mask  = sample['brain_mask'].float().to(device)
            mask        = sample['mask'].float().to(device)
            sens        = sample['sens'].float().to(device)

            with torch.amp.autocast('cuda'):
                v9_out = model(data_in_img, data_in, mask, sens)

            vm = slice_metrics(v9_out, data_ref, brain_mask)

            file_path, slice_idx, _ = ds.samples[idx]
            key = (os.path.basename(file_path), slice_idx)
            row = {'idx': idx, 'file': key[0], 'slice_idx': slice_idx}

            v8row = v8.get(key)
            if v8row is None:
                unmatched += 1
                continue

            for k in METRICS:
                row[f'gru_{k}'] = v8row[f'gru_{k}']
                row[f'ss2d_{k}'] = v8row[f'ss2d_{k}']
                row[f'v9_{k}'] = vm[k]
                gru_vals[k].append(v8row[f'gru_{k}'])
                ss2d_vals[k].append(v8row[f'ss2d_{k}'])
                v9_vals[k].append(vm[k])
            row['winner_v9_vs_ss2d'] = ('v9' if vm['composite'] > v8row['ss2d_composite']
                                        else ('ss2d' if v8row['ss2d_composite'] > vm['composite'] else 'tie'))
            row['winner_v9_vs_gru'] = ('v9' if vm['composite'] > v8row['gru_composite']
                                       else ('gru' if v8row['gru_composite'] > vm['composite'] else 'tie'))
            rows.append(row)

    matched = len(rows)
    fieldnames = ['idx', 'file', 'slice_idx']
    for k in METRICS:
        fieldnames += [f'gru_{k}', f'ss2d_{k}', f'v9_{k}']
    fieldnames += ['winner_v9_vs_ss2d', 'winner_v9_vs_gru']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    v9_comp_mean = float(np.mean(v9_vals['composite'])) if matched else float('nan')
    v9_ssim_mean = float(np.mean(v9_vals['ssim'])) if matched else float('nan')
    lines = ['# v9 unleashed — per-slice paired 비교 (vs v8 no-DC, 동일 val·지표)', '',
             f'- 평가 슬라이스: {total} / 조인 매칭: {matched} (v8 CSV {len(v8)}, 미매칭 {unmatched})',
             f'- v9 ckpt: `{args.ckpt}` (best ep78)',
             f'- sanity: v9 per-slice ssim 평균 = **{v9_ssim_mean:.4f}** — 학습 로그 best ckpt(ep78) '
             f'val_ssim_m 0.9145 와 일치해야 정상(SSIM 은 양쪽 다 슬라이스 단위 계산). '
             f'composite/psnr 절대값은 per-slice vs 배치풀링(학습 val BS=4, 배치 공유 ref-max) 정의 차이로 '
             f'로그(0.9203/35.18)보다 낮게 보이는 것이 정상 — v8 도 동일 오프셋(CSV 0.9104 vs 로그 0.9200). '
             f'paired 비교는 3모델 동일 프로토콜로 유효. (v9 composite per-slice 평균 = {v9_comp_mean:.4f})', '']
    lines += pair_table('v9 vs v8-SS2D (no-DC)', 'v8-SS2D', ss2d_vals, 'v9', v9_vals, matched)
    lines += ['']
    lines += pair_table('v9 vs v8-GRU (no-DC)', 'GRU', gru_vals, 'v9', v9_vals, matched)
    lines += ['', '(win-rate 은 composite 이 아닌 각 지표 자체 기준 — nmse/l1 은 낮을수록 v9 승리)']

    msg = '\n'.join(lines)
    print('\n' + msg)
    with open(summary_path, 'w') as f:
        f.write(msg + '\n')

    print(f'\nCSV: {csv_path}')
    print(f'요약: {summary_path}')
    if unmatched:
        print(f'[WARN] v8 CSV 미매칭 {unmatched} 슬라이스 — dataset 정합 확인 필요')


if __name__ == '__main__':
    main()
