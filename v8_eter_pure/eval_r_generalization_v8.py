"""
v8 Pure ETER-Net (no-DC) — 가속률(R) 일반화 cross-eval.

R4 로 학습한 GRU/SS2D(no-DC) 를 **재학습 없이** R∈{2,4,6,8}에서 평가해 일반화 곡선을
뽑는다. 질문(docs/v8_ss2d_kspace_domain_review.md §4): SS2D 의 R4 우위가 가속률 전반에서
유지/확대/축소되나? 어느 쪽이 더 완만하게 무너지나(§4.2-④ Mamba 전역성)?

메트릭 공식은 eval_paired_v8_nodc.slice_metrics 를 그대로 재사용(=R4 결과와 동일 좌표계).
모델은 R 무관이라 1회 로드 후 R 만 바꿔 dataloader 재생성. DC 축은 폐기(§7)라 no-DC 2모델만.
"""
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)

import eval_paired_v8_nodc as EP          # sys.path/env 셋업 + 재사용 함수
from eval_paired_v8_nodc import slice_metrics, build_model, load_ckpt, METRICS, LOWER_IS_BETTER

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = EP.C
FastMRI_H5_Dataloader = EP.FastMRI_H5_Dataloader


def mean_input_norm(data_path, R, cf, max_samples, stride=1):
    """R 별 입력 x_img 의 brain 영역 평균 |값| (모델 무접촉). R-불변 정규화 스케일 산출용."""
    ds = FastMRI_H5_Dataloader(data_path, num_files=None, target_size=C.IMAGE_SIZE[0],
                               acceleration=R, center_fraction=cf,
                               random_mask=False, augment=False)
    if stride > 1:
        ds = Subset(ds, list(range(0, len(ds), stride)))
    total = len(ds)
    if max_samples > 0:
        total = min(total, max_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    it = iter(loader); acc = 0.0; n = 0
    for _ in range(total):
        s = next(it)
        xi = s['data_img'].float().abs()
        bm = s['brain_mask'].float()
        acc += float((xi * bm).sum() / (bm.sum().clamp(min=1.0) * xi.shape[1])); n += 1
    return acc / max(n, 1)


def eval_at_R(gru, ss2d, data_path, R, cf, max_samples, device, scale=1.0, stride=1):
    ds = FastMRI_H5_Dataloader(data_path, num_files=None, target_size=C.IMAGE_SIZE[0],
                               acceleration=R, center_fraction=cf,
                               random_mask=False, augment=False)
    if stride > 1:
        ds = Subset(ds, list(range(0, len(ds), stride)))   # 파일 전반 대표샘플(첫 N 편향 제거)
    total = len(ds)
    if max_samples > 0:
        total = min(total, max_samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    gv = {k: [] for k in METRICS}
    sv = {k: [] for k in METRICS}
    it = iter(loader)
    with torch.no_grad():
        for _ in tqdm(range(total), desc=f'R={R}', unit='slice', leave=False):
            s = next(it)
            di  = s['data'].float().to(device)
            dii = s['data_img'].float().to(device)
            ref = s['label'].float().to(device)
            bm  = s['brain_mask'].float().to(device)
            mk  = s['mask'].float().to(device)
            sn  = s['sens'].float().to(device)
            if scale != 1.0:            # R-불변 정규화: 입력 magnitude 를 기준 R 스케일로
                dii = dii * scale; di = di * scale
            with torch.amp.autocast('cuda'):
                go = gru(dii, di, mk, sn)
                so = ss2d(dii, di, mk, sn)
            gm = slice_metrics(go, ref, bm)
            sm = slice_metrics(so, ref, bm)
            for k in METRICS:
                gv[k].append(gm[k]); sv[k].append(sm[k])
    out = {'R': R, 'n': total, 'gru': {}, 'ss2d': {}, 'ss2d_winrate': {}}
    for k in METRICS:
        g = np.array(gv[k]); s = np.array(sv[k])
        out['gru'][k] = float(g.mean()); out['ss2d'][k] = float(s.mean())
        wins = int((s < g).sum()) if k in LOWER_IS_BETTER else int((s > g).sum())
        out['ss2d_winrate'][k] = 100.0 * wins / total
    return out


def main():
    p = argparse.ArgumentParser(description='v8 no-DC GRU vs SS2D 가속률(R) 일반화 cross-eval')
    p.add_argument('--gru-ckpt', default='logs/PureETER_GRU_noDC_R4_brain384_v8/pure_gru_best.pt')
    p.add_argument('--ss2d-ckpt', default='logs/PureETER_SS2D_noDC_R4_brain384_v8/pure_ss2d_best.pt')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--out-dir', default='results/eval/v8_r_sweep')
    p.add_argument('--accels', default='2,4,6,8', help='쉼표구분 정수 R 목록')
    p.add_argument('--center-fraction', type=float, default=0.08)
    p.add_argument('--max-samples', type=int, default=400, help='-1=전체 val, 기본 400(빠른 트렌드)')
    p.add_argument('--r-invariant-norm', action='store_true',
                   help='R 별 입력 magnitude 를 --r-ref 스케일로 정규화(R2 교란=스케일 drift 분리 실험)')
    p.add_argument('--r-ref', type=int, default=4, help='정규화 기준 R (기본 4 = 학습 R)')
    p.add_argument('--stride', type=int, default=1,
                   help='>1 이면 전체 val 을 stride 간격으로 대표샘플(파일 전반, 첫 N 편향 제거). 예 4→~1834슬라이스')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accels = [int(x) for x in args.accels.split(',')]
    print('=' * 64)
    print(' v8 Pure ETER-Net (no-DC) — 가속률(R) 일반화 cross-eval')
    print(f'  device={device}  R={accels}  max_samples={args.max_samples}  cf={args.center_fraction}')
    print('=' * 64)

    print('\n모델 로드(1회, R 무관)...')
    gru = load_ckpt(build_model('gru', device), args.gru_ckpt, device)
    ss2d = load_ckpt(build_model('ss2d', device), args.ss2d_ckpt, device)

    scales = {R: 1.0 for R in accels}
    if args.r_invariant_norm:
        print(f'\nR-불변 정규화: R 별 입력 norm 산출 (기준 R={args.r_ref})...')
        norms = {R: mean_input_norm(args.data_path, R, args.center_fraction, args.max_samples, args.stride) for R in accels}
        ref = norms.get(args.r_ref, norms[accels[0]])
        scales = {R: ref / norms[R] for R in accels}
        for R in accels:
            print(f'  R={R}: mean|x_img|={norms[R]:.4g}  scale={scales[R]:.4f}')

    results = []
    for R in accels:
        r = eval_at_R(gru, ss2d, args.data_path, R, args.center_fraction, args.max_samples, device,
                      scale=scales[R], stride=args.stride)
        results.append(r)
        print(f"  R={R} (n={r['n']}): "
              f"comp GRU {r['gru']['composite']:.4f} / SS2D {r['ss2d']['composite']:.4f} "
              f"(Δ {r['ss2d']['composite']-r['gru']['composite']:+.4f}, SS2D win {r['ss2d_winrate']['composite']:.0f}%) | "
              f"ssim GRU {r['gru']['ssim']:.4f} / SS2D {r['ss2d']['ssim']:.4f} | "
              f"psnr GRU {r['gru']['psnr']:.2f} / SS2D {r['ss2d']['psnr']:.2f}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 표 ──
    L = ['# v8 Pure ETER-Net (no-DC) — 가속률(R) 일반화 cross-eval', '',
         f'R4 학습 모델을 재학습 없이 R∈{accels} 평가. n={results[0]["n"]}/R'
         + (f' (stride={args.stride} 대표샘플, 파일 전반)' if args.stride > 1
            else (' (전체 val set)' if args.max_samples <= 0 else ' (앞쪽 subsample)')),
         (f'**R-불변 정규화 ON** (기준 R={args.r_ref}): 입력 magnitude 를 R 무관하게 맞춤 → R 열화가 aliasing 순수효과인지 확인.'
          if args.r_invariant_norm else '(입력 정규화 없음 = val_amp 고정, 원본 파이프라인)'), '',
         '## composite (핵심)', '',
         '| R | GRU comp | SS2D comp | Δ(SS2D−GRU) | SS2D win% |', '|---:|---:|---:|---:|---:|']
    for r in results:
        d = r['ss2d']['composite'] - r['gru']['composite']
        L.append(f"| {r['R']} | {r['gru']['composite']:.4f} | {r['ss2d']['composite']:.4f} "
                 f"| {d:+.4f} | {r['ss2d_winrate']['composite']:.0f}% |")
    for k in ['ssim', 'psnr', 'nmse', 'l1']:
        L += ['', f'## {k}', '', f'| R | GRU {k} | SS2D {k} | Δ(SS2D−GRU) |', '|---:|---:|---:|---:|']
        for r in results:
            d = r['ss2d'][k] - r['gru'][k]
            fmt = '.4f' if k in ('ssim', 'nmse') else ('.2f' if k == 'psnr' else '.3f')
            L.append(f"| {r['R']} | {r['gru'][k]:{fmt}} | {r['ss2d'][k]:{fmt}} | {d:+{fmt}} |")
    table_path = os.path.join(args.out_dir, 'r_generalization_table.md')
    with open(table_path, 'w') as f:
        f.write('\n'.join(L) + '\n')

    # ── 곡선 ──
    Rs = [r['R'] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, k, ttl in zip(axes, ['ssim', 'psnr', 'nmse'],
                          ['masked SSIM', 'PSNR (dB)', 'NMSE (lower better)']):
        ax.plot(Rs, [r['gru'][k] for r in results], 'o-', label='GRU (668M)', color='#d62728')
        ax.plot(Rs, [r['ss2d'][k] for r in results], 's-', label='SS2D (31M)', color='#1f77b4')
        ax.set_xlabel('acceleration R'); ax.set_title(ttl); ax.set_xticks(Rs); ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f'v8 no-DC R 일반화 (R4 학습, 재학습 없음; n={results[0]["n"]}/R)')
    fig.tight_layout()
    curve_path = os.path.join(args.out_dir, 'r_generalization_curves.png')
    fig.savefig(curve_path, dpi=110, bbox_inches='tight')

    print('\n' + '\n'.join(L))
    print(f'\n표 : {table_path}')
    print(f'곡선: {curve_path}')


if __name__ == '__main__':
    main()
