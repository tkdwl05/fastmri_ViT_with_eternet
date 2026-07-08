"""
v8 Pure ETER-Net — GRU vs SS2D (no-DC) per-slice paired 평가.

전체 val set(~7270 슬라이스)에서 두 체크포인트를 동시에 로드해 슬라이스 1회 순회하며
masked SSIM/PSNR/NMSE/L1/composite 을 슬라이스 단위로 계산한다(main_train_pure_v8.py
run_val() 과 동일 공식 — 두 모델 mean 이 학습 로그의 best_val_composite 와 근접해야
정상, 어긋나면 clone 버그 의심). paired win-rate + Wilcoxon signed-rank 로
"SS2D 완승"(docs/v8_eter_pure_rnn_vs_ss2d.md) 의 슬라이스 단위 근거를 보강한다.
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

import myConfig_pure_eter_v8 as C
from dataloader_h5_v5 import FastMRI_H5_Dataloader

METRICS = ['ssim', 'psnr', 'nmse', 'l1', 'composite']
LOWER_IS_BETTER = {'nmse', 'l1'}


def skimage_ssim_masked(pred, target, mask):
    """main_train_pure_v8.skimage_ssim_batch_masked 의 단일-슬라이스(배치=1) 버전."""
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
    """main_train_pure_v8.run_val() 의 텐서 공식을 슬라이스(배치=1) 단위로 그대로 적용."""
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


def build_model(seq_model, device):
    if seq_model == 'gru':
        from u_pure_eternet_gru import PureETER_GRU
        model = PureETER_GRU(
            dim=C.IMAGE_SIZE[0], n_coil=C.N_COIL,
            n_hidden_1=C.N_HIDDEN_LRNN_1, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF, use_dc=False,
        )
    else:
        from u_pure_eternet_ss2d import PureETER_SS2D
        model = PureETER_SS2D(
            n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            ss2d_d_inner=C.SS2D_D_INNER, ss2d_d_state=C.SS2D_D_STATE, use_dc=False,
        )
    return model.to(device)


def load_ckpt(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser(description='v8 pure ETER-Net GRU vs SS2D (no-DC) per-slice paired 평가')
    p.add_argument('--gru-ckpt', default='logs/PureETER_GRU_noDC_R4_brain384_v8/pure_gru_best.pt')
    p.add_argument('--ss2d-ckpt', default='logs/PureETER_SS2D_noDC_R4_brain384_v8/pure_ss2d_best.pt')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--out-dir', default='results/eval/v8_nodc')
    p.add_argument('--max-samples', type=int, default=-1, help='-1 = 전체 val set')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 64)
    print(' v8 Pure ETER-Net — GRU vs SS2D (no-DC) per-slice paired 평가')
    print(f'  device={device}')
    print('=' * 64)

    if not torch.cuda.is_available():
        print('  [WARN] CUDA 없음 — CPU 로 진행 (매우 느릴 수 있음)')

    print('\n모델 로드 중...')
    gru = load_ckpt(build_model('gru', device), args.gru_ckpt, device)
    ss2d = load_ckpt(build_model('ss2d', device), args.ss2d_ckpt, device)
    print(f'  GRU : {args.gru_ckpt}')
    print(f'  SS2D: {args.ss2d_ckpt}')

    ds = FastMRI_H5_Dataloader(args.data_path, num_files=None, target_size=C.IMAGE_SIZE[0],
                               acceleration=4, center_fraction=0.08,
                               random_mask=False, augment=False)
    total = len(ds)
    if args.max_samples > 0:
        total = min(total, args.max_samples)
    print(f'\n평가 대상: {total} / {len(ds)} 슬라이스\n')

    # num_workers=0: 이 컨테이너 /dev/shm 이 64MB 로 작아 multi-worker tensor 공유 시
    # bus error 발생 (실측). forward-only 단일 패스라 워커 병렬화 이득도 적어 0 으로 고정.
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'per_slice_paired.csv')
    summary_path = os.path.join(args.out_dir, 'win_rate_summary.md')

    gru_vals = {k: [] for k in METRICS}
    ss2d_vals = {k: [] for k in METRICS}
    rows = []

    it = iter(loader)
    with torch.no_grad():
        for idx in tqdm(range(total), desc='paired eval', unit='slice'):
            sample = next(it)
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            brain_mask  = sample['brain_mask'].float().to(device)
            mask        = sample['mask'].float().to(device)
            sens        = sample['sens'].float().to(device)

            with torch.amp.autocast('cuda'):
                gru_out = gru(data_in_img, data_in, mask, sens)
                ss2d_out = ss2d(data_in_img, data_in, mask, sens)

            gm = slice_metrics(gru_out, data_ref, brain_mask)
            sm = slice_metrics(ss2d_out, data_ref, brain_mask)

            file_path, slice_idx, _ = ds.samples[idx]
            row = {'idx': idx, 'file': os.path.basename(file_path), 'slice_idx': slice_idx}
            for k in METRICS:
                row[f'gru_{k}'] = gm[k]
                row[f'ss2d_{k}'] = sm[k]
                gru_vals[k].append(gm[k])
                ss2d_vals[k].append(sm[k])
            row['winner_composite'] = ('ss2d' if sm['composite'] > gm['composite']
                                       else ('gru' if gm['composite'] > sm['composite'] else 'tie'))
            rows.append(row)

    fieldnames = ['idx', 'file', 'slice_idx']
    for k in METRICS:
        fieldnames += [f'gru_{k}', f'ss2d_{k}']
    fieldnames.append('winner_composite')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    lines = ['# v8 Pure ETER-Net — GRU vs SS2D (no-DC) per-slice paired win-rate', '',
             f'슬라이스 수: {total} (val set 전체 = {len(ds)})', '',
             '| 지표 | GRU mean±std | SS2D mean±std | SS2D win-rate | GRU win | tie | Wilcoxon p |',
             '|---|---|---|---|---|---|---|']
    for k in METRICS:
        g = np.array(gru_vals[k])
        s = np.array(ss2d_vals[k])
        if k in LOWER_IS_BETTER:
            ss2d_wins = int((s < g).sum())
            gru_wins = int((g < s).sum())
        else:
            ss2d_wins = int((s > g).sum())
            gru_wins = int((g > s).sum())
        ties = total - ss2d_wins - gru_wins
        win_rate = ss2d_wins / total * 100
        try:
            _stat, pval = wilcoxon(s, g)
        except ValueError:
            pval = float('nan')
        lines.append(f'| {k} | {g.mean():.4f}±{g.std():.4f} | {s.mean():.4f}±{s.std():.4f} '
                     f'| {win_rate:.1f}% ({ss2d_wins}/{total}) | {gru_wins} | {ties} | {pval:.2e} |')

    lines += ['', '(win-rate 은 composite 이 아닌 각 지표 자체 기준 — 예: nmse/l1 은 낮을수록 SS2D 승리)']

    msg = '\n'.join(lines)
    print('\n' + msg)
    with open(summary_path, 'w') as f:
        f.write(msg + '\n')

    print(f'\nCSV: {csv_path}')
    print(f'요약: {summary_path}')


if __name__ == '__main__':
    main()
