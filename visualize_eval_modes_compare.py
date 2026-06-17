"""
평가 영역(evaluation region) 비교 시각화 — "배경 오차를 제외하는 것"의 영향을 한눈에 비교.

같은 recon 을 세 영역으로 채점/에러맵 작성:
  (1) full   : 전체 이미지 (배경 포함) — 배경 아티팩트가 보이고, raw SSIM 부풀림을 드러냄
  (2) narrow : brain_mask (Otsu×0.4 + largest CC) — 현행 v7_titan 평가
  (3) wide   : Otsu×(기본 0.3) + largest CC + dilation, narrow 와 합집합 — 경계 포함

recon 은 narrow brain-mask 기준 per-slice LS scale 로 정합해 고정하고, 평가 영역만 3가지로 바꾼다
(→ "평가 영역" 효과만 격리). 모델 추론은 visualize_v7_titan_compare 의 로더/러너를 재사용.

Layout (slice 당 4행×4열):
  Row 0: recon        | GT | U-Net | ETER | SS2D   (title = raw / nar / wid 의 SSIM·PSNR 3줄)
  Row 1: |err| full   | GT(full)  | 모델별 전체-이미지 에러맵 (배경 포함)
  Row 2: |err| narrow | GT×narrow | 모델별 현행 masked 에러맵
  Row 3: |err| wide   | GT×wide   | 모델별 wide-mask 에러맵
출력: results/vis/v7_titan_eval_modes/ (evalmodes_*.png + metrics_eval_modes.txt)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.ndimage import label as ndi_label, binary_dilation
from skimage.filters import threshold_otsu

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import visualize_v7_titan_compare as V   # 로더/러너/지표 재사용 (모델 추론 1회)


def brain_mask_from_gt(gt, otsu_factor, dilate_iters=0):
    """GT 에서 Otsu×factor + largest CC (+옵션 dilation) brain mask 생성.
    narrow(dataloader, Otsu×0.4)와 동일 방식이되 factor 만 다름."""
    m = float(gt.max())
    if m <= 0:
        return np.zeros_like(gt, dtype=np.float32)
    nz = gt[gt > 0]
    if nz.size > 100:
        try:
            thr = float(threshold_otsu(nz)) * otsu_factor
        except Exception:
            thr = 0.05 * m
    else:
        thr = 0.05 * m
    raw = gt > thr
    lbl, n = ndi_label(raw)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        out = (lbl == int(sizes.argmax()))
    else:
        out = raw
    if dilate_iters > 0:
        out = binary_dilation(out, iterations=dilate_iters)
    return out.astype(np.float32)


def metrics3(rec_adj, gt, ones, narrow, wide):
    return {
        'raw':    (V.masked_psnr(rec_adj, gt, ones),   V.masked_ssim(rec_adj, gt, ones)),
        'narrow': (V.masked_psnr(rec_adj, gt, narrow), V.masked_ssim(rec_adj, gt, narrow)),
        'wide':   (V.masked_psnr(rec_adj, gt, wide),   V.masked_ssim(rec_adj, gt, wide)),
    }


def main():
    p = argparse.ArgumentParser(description='평가 영역(full/narrow/wide) 비교 시각화')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--ss2d-ckpt', default='logs/SS2D_ViT_R4_brain384_v7_titan/ss2d_vit_best.pt')
    p.add_argument('--eter-ckpt', default='logs/ETER_ViT_R4_brain384_v7_titan/eter_vit_best.pt')
    p.add_argument('--unet-ckpt', default='models/pretrained/brain_leaderboard_state_dict.pt')
    p.add_argument('--out-dir', default='results/vis/v7_titan_eval_modes')
    p.add_argument('--num-samples', type=int, default=12)
    p.add_argument('--slice-indices', default=None,
                   help='쉼표구분 인덱스. 미지정 시 linspace(0,len-1,num_samples)')
    p.add_argument('--err-vmax-frac', type=float, default=0.10)
    p.add_argument('--wide-otsu', type=float, default=0.3, help='wide mask Otsu 계수 (narrow 는 0.4)')
    p.add_argument('--wide-dilate', type=int, default=5, help='wide mask dilation 반복')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 64)
    print(f' 평가 영역 비교: full / narrow(Otsu×0.4+CC) / wide(Otsu×{args.wide_otsu}+dil{args.wide_dilate})')
    print(f'  device={device}')
    print('=' * 64)

    h5 = V.FastMRI_H5_Dataloader(args.data_path, num_files=None, target_size=384,
                                 acceleration=V.ACCEL, center_fraction=V.CENTER_FRACTION,
                                 random_mask=False, augment=False)
    total = len(h5)
    if args.slice_indices:
        indices = [int(x) for x in args.slice_indices.split(',') if x.strip() != '']
        indices = [i for i in indices if 0 <= i < total]
    else:
        indices = np.linspace(0, total - 1, args.num_samples, dtype=int).tolist()
    print(f'총 val {total} → {len(indices)}개: {indices}\n')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 캐시: sample / GT / narrow / wide ──
    samp, gtc, narrowc, widec = {}, {}, {}, {}
    for idx in tqdm(indices, desc='cache', unit='img'):
        s = V.to_tensor_sample(h5[idx]); samp[idx] = s
        gt = s['label'].squeeze().numpy().astype(np.float32); gtc[idx] = gt
        narrow = s['brain_mask'].squeeze().numpy().astype(np.float32); narrowc[idx] = narrow
        wide = brain_mask_from_gt(gt, args.wide_otsu, args.wide_dilate)
        widec[idx] = np.maximum(wide, narrow)   # wide ⊇ narrow 보장

    # ── 모델 추론 (1회) → recon(LS-narrow 정합) + 3-region 지표 ──
    specs = [('U-Net',         args.unet_ckpt, V.load_unet,         V.run_unet),
             ('ETER v7_titan', args.eter_ckpt, V.load_eter_v7titan, V.run_eter),
             ('SS2D v7_titan', args.ss2d_ckpt, V.load_ss2d_v7titan, V.run_ss2d)]
    recon = {idx: {} for idx in indices}
    met = {idx: {} for idx in indices}
    for name, ckpt, loader, runner in specs:
        if not os.path.exists(ckpt):
            print(f'  [skip] {name}: {ckpt} 없음'); continue
        print(f'[{name}] load {ckpt}')
        model = loader(ckpt, device)
        for idx in tqdm(indices, desc=f'{name}', unit='img'):
            try:
                rec = runner(model, samp[idx], device)
            except Exception as e:
                tqdm.write(f'  [ERROR] {name} idx={idx}: {e}'); rec = np.zeros_like(gtc[idx])
            rec = np.nan_to_num(rec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            gt = gtc[idx]; narrow = narrowc[idx]
            rec_adj = V.ls_scale(rec, gt, narrow)
            recon[idx][name] = rec_adj
            met[idx][name] = metrics3(rec_adj, gt, np.ones_like(narrow), narrow, widec[idx])
        del model; torch.cuda.empty_cache()

    # ── 플로팅 (4행×4열) ──
    hot = plt.get_cmap('hot').copy(); hot.set_bad(color='black')
    em = args.err_vmax_frac
    cols = ['U-Net', 'ETER v7_titan', 'SS2D v7_titan']
    tc = {'U-Net': 'tab:orange', 'ETER v7_titan': 'tab:green', 'SS2D v7_titan': 'tab:blue'}
    rows_meta = []
    for idx in tqdm(indices, desc='plot', unit='img'):
        gt = gtc[idx]; narrow = narrowc[idx]; wide = widec[idx]
        gmax = max(float(gt.max()), 1e-8); gt_n = gt / gmax
        fig, ax = plt.subplots(4, 4, figsize=(20, 20))
        # col 0 — GT + 영역 레퍼런스
        ax[0, 0].imshow(gt_n, cmap='gray', vmin=0, vmax=1); ax[0, 0].set_title('GT', fontsize=12, fontweight='bold'); ax[0, 0].axis('off')
        ax[1, 0].imshow(gt_n, cmap='gray', vmin=0, vmax=1); ax[1, 0].set_title('region: full (no mask)', fontsize=10); ax[1, 0].axis('off')
        ax[2, 0].imshow(gt_n * narrow, cmap='gray', vmin=0, vmax=1); ax[2, 0].set_title('region: narrow (Otsu×0.4+CC)', fontsize=10); ax[2, 0].axis('off')
        ax[3, 0].imshow(gt_n * wide, cmap='gray', vmin=0, vmax=1); ax[3, 0].set_title(f'region: wide (Otsu×{args.wide_otsu}+dil{args.wide_dilate})', fontsize=10); ax[3, 0].axis('off')
        for c, name in enumerate(cols, start=1):
            if name not in recon[idx]:
                for r in range(4):
                    ax[r, c].axis('off')
                ax[0, c].set_title(f'{name}\n(N/A)', fontsize=10)
                continue
            ra = recon[idx][name]; rn = ra / gmax; mm = met[idx][name]
            ax[0, c].imshow(rn, cmap='gray', vmin=0, vmax=1)
            ax[0, c].set_title(
                f"{name}\nraw {mm['raw'][1]:.3f} / {mm['raw'][0]:.1f}dB\n"
                f"nar {mm['narrow'][1]:.3f} / {mm['narrow'][0]:.1f}dB\n"
                f"wid {mm['wide'][1]:.3f} / {mm['wide'][0]:.1f}dB",
                fontsize=9, color=tc[name])
            ax[0, c].axis('off')
            err = np.abs(rn - gt_n)
            im1 = ax[1, c].imshow(err, cmap=hot, vmin=0, vmax=em); ax[1, c].axis('off'); plt.colorbar(im1, ax=ax[1, c], fraction=0.046)
            im2 = ax[2, c].imshow(np.where(narrow > 0.5, err, np.nan), cmap=hot, vmin=0, vmax=em); ax[2, c].axis('off'); plt.colorbar(im2, ax=ax[2, c], fraction=0.046)
            im3 = ax[3, c].imshow(np.where(wide > 0.5, err, np.nan), cmap=hot, vmin=0, vmax=em); ax[3, c].axis('off'); plt.colorbar(im3, ax=ax[3, c], fraction=0.046)
        fig.suptitle(
            f'Sample #{idx} — 평가 영역 비교  (행: recon / |err| full / |err| narrow(현행) / |err| wide)\n'
            f'title = raw · nar · wid  의 SSIM / PSNR · err_vmax={em} · recon=narrow-LS 정합 고정',
            fontsize=12, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'evalmodes_{idx:04d}.png'), dpi=130, bbox_inches='tight')
        plt.close(fig)
        rows_meta.append((idx, {n: met[idx][n] for n in recon[idx]}))

    # ── 요약 ──
    present = [n for n, *_ in specs if any(n in m for _i, m in rows_meta)]
    lines = ['========== 평가 영역(full / narrow / wide) 비교 요약 ==========',
             f'슬라이스 {len(indices)}개: {indices}',
             f'narrow = Otsu×0.4+CC (dataloader brain_mask, 현행) · '
             f'wide = Otsu×{args.wide_otsu}+dilation{args.wide_dilate} (∪narrow) · recon = narrow-LS 정합',
             '',
             f'{"모델":>14s} | {"region":>7s} | {"SSIM (mean±std)":>18s} | {"PSNR dB (mean±std)":>20s}',
             f'{"-"*14} | {"-"*7} | {"-"*18} | {"-"*20}']
    for name in present:
        for region in ['raw', 'narrow', 'wide']:
            ss = [m[name][region][1] for _i, m in rows_meta if name in m and np.isfinite(m[name][region][1])]
            ps = [m[name][region][0] for _i, m in rows_meta if name in m and np.isfinite(m[name][region][0])]
            if not ss:
                continue
            lines.append(f'{name:>14s} | {region:>7s} | {np.mean(ss):>7.4f} ± {np.std(ss):<6.4f} | {np.mean(ps):>7.2f} ± {np.std(ps):<6.2f}')
        lines.append(f'{"-"*14} | {"-"*7} | {"-"*18} | {"-"*20}')
    lines += ['', '[읽는 법]',
              '- raw(full) SSIM 이 narrow/wide 보다 크게 높으면 = 배경(≈0)이 점수를 부풀린다는 직접 증거.',
              '- narrow = 뇌만 채점(현행 v7_titan), wide = 경계까지 포함. 세 값 비교로 "배경 제외" 영향 정량 확인.',
              '- 에러맵: row1(full) 은 배경 아티팩트까지 보임 / row2(narrow) 는 현행 / row3(wide) 는 경계 포함.']
    msg = '\n'.join(lines)
    print('\n' + msg)
    with open(os.path.join(args.out_dir, 'metrics_eval_modes.txt'), 'w') as f:
        f.write(msg + '\n')
    print(f'\n저장 완료: {args.out_dir}/  (PNG {len(indices)}장 + metrics_eval_modes.txt)')


if __name__ == '__main__':
    main()
