"""
visualize_diagnostic_v6.py

Visual-Metric Gap 진단용 시각화. `docs/visual_metric_gap_v6.md` 에 정리한
4 가지 원인을 한 번에 점검:

1. results/eval/eval_full_v6.csv 의 SS2D SSIM 기준 best/median/worst 슬라이스 자동 선정
2. brain-mask SSIM 추가 측정 (배경 제외)
3. full + zoom-in (160×160 center) 행 동시 표시
4. percentile windowing (display 1~99% / error map 0~99%)

출력: results/vis/vis_diagnostic_v6/ — 9 PNG + diagnostic_summary.txt
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import csv
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models import Unet

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'configs'))
sys.path.append(os.path.join(current_dir, 'dataloaders'))
sys.path.append(os.path.join(current_dir, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(current_dir, 'models', 'mamba_eternet'))


def center_crop(img, size):
    h, w = img.shape[-2:]
    sh, sw = (h - size) // 2, (w - size) // 2
    return img[..., sh:sh + size, sw:sw + size]


def crop_or_pad(data, shape):
    out = data
    for axis, target in zip((-2, -1), shape):
        cur = out.shape[axis]
        if cur == target:
            continue
        if cur > target:
            start = (cur - target) // 2
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(start, start + target)
            out = out[tuple(sl)]
        else:
            pad = target - cur
            pw = [(0, 0)] * out.ndim
            pw[axis] = (pad // 2, pad - pad // 2)
            out = np.pad(out, pw, mode='constant', constant_values=0)
    return out


def calc_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(target.max() / np.sqrt(mse))


def calc_ssim_raw(pred, target):
    dr = target.max() - target.min()
    if dr <= 0:
        return 0.0
    return float(compare_ssim(target, pred, data_range=dr))


def calc_ssim_masked(pred, target, mask):
    """brain mask 내부 평균 SSIM. data_range 는 brain 픽셀 기반."""
    if not mask.any():
        return 0.0
    dr = float(target[mask].max() - target[mask].min())
    if dr <= 0:
        return 0.0
    _, full_map = compare_ssim(target, pred, data_range=dr, full=True)
    return float(full_map[mask].mean())


def make_mask(gt, thr_ratio=0.05):
    return gt > (thr_ratio * gt.max())


def load_ss2d(ckpt, device):
    from myConfig_choh_SS2D_model_v4 import (
        IMAGE_SIZE, PATCH_SIZE, INPUT_CHANNELS,
        NUM_VIT_ENCODER_HIDDEN, NUM_VIT_ENCODER_LAYER,
        NUM_VIT_ENCODER_MLP_SIZE, NUM_VIT_ENCODER_HEAD,
        NUM_SS2D_D_INNER, NUM_SS2D_D_STATE, NUM_SS2D_OUT_CH,
        NUM_VIT_DECODER_DIM, NUM_VIT_DECODER_DEPTH,
        NUM_VIT_DECODER_HEAD, NUM_VIT_DECODER_DIM_HEAD,
        NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        DROPOUT, DC_K_SCALE_RATIO, DC_INIT_ALPHA,
    )
    from u_choh_model_ETER_ViT import choh_ViT
    from u_choh_model_SS2D_ViT_v4 import choh_Decoder_SS2D_ViT
    enc = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1)
    m = choh_Decoder_SS2D_ViT(
        encoder=enc,
        ss2d_d_inner=NUM_SS2D_D_INNER, ss2d_d_state=NUM_SS2D_D_STATE,
        ss2d_out_ch=NUM_SS2D_OUT_CH,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=DROPOUT, dc_k_scale_ratio=DC_K_SCALE_RATIO, dc_init_alpha=DC_INIT_ALPHA,
    )
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


def load_eter(ckpt, device):
    from myConfig_choh_ETER_model_v5 import (
        IMAGE_SIZE, PATCH_SIZE, INPUT_CHANNELS,
        NUM_VIT_ENCODER_HIDDEN, NUM_VIT_ENCODER_LAYER,
        NUM_VIT_ENCODER_MLP_SIZE, NUM_VIT_ENCODER_HEAD,
        NUM_ETER_HORI_HIDDEN, NUM_ETER_VERT_HIDDEN,
        NUM_VIT_DECODER_DIM, NUM_VIT_DECODER_DEPTH,
        NUM_VIT_DECODER_HEAD, NUM_VIT_DECODER_DIM_HEAD,
        NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        DROPOUT,
    )
    from u_choh_model_ETER_ViT import choh_ViT
    from u_choh_model_ETER_ViT_v5 import choh_Decoder3_ETER_v5
    enc = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1)
    m = choh_Decoder3_ETER_v5(
        encoder=enc,
        eter_n_hori_hidden=NUM_ETER_HORI_HIDDEN, eter_n_vert_hidden=NUM_ETER_VERT_HIDDEN,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=DROPOUT,
    )
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


def load_unet(ckpt, device):
    m = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return m.to(device).eval()


def select_slices(csv_path, n_each=3):
    """SS2D SSIM 기준 best/median/worst 각 n_each 개씩 선정."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append((int(row['idx']), float(row['ss2d_ssim'])))
    rows.sort(key=lambda x: x[1])
    n = len(rows)
    mid_start = n // 2 - n_each // 2
    selected = []
    for k in range(n_each):
        i, s = rows[n - 1 - k]
        selected.append(('best', i, s))
    for k in range(n_each):
        i, s = rows[mid_start + k]
        selected.append(('median', i, s))
    for k in range(n_each):
        i, s = rows[k]
        selected.append(('worst', i, s))
    return selected


def make_diagnostic_figure(slice_idx, tag, gt, ss2d, eter, unet,
                            metrics, save_path, zoom_size=160):
    """3행 × 4열: row0=full, row1=zoom, row2=brain mask + error zooms."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    vmin = float(np.percentile(gt, 1))
    vmax = float(np.percentile(gt, 99))

    titles = ['GT', 'SS2D-ViT v6', 'ETER-ViT v6', 'U-Net (PT)']
    images = [gt, ss2d, eter, unet]
    colors = ['black', 'tab:blue', 'tab:green', 'tab:orange']

    for j, (img, title, color) in enumerate(zip(images, titles, colors)):
        axes[0, j].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        if j == 0:
            sub = '\n(percentile 1~99% windowing)'
        else:
            m = metrics[j - 1]
            sub = (f"\nPSNR {m['psnr']:.2f}dB"
                   f"\nSSIM raw {m['ssim_raw']:.4f}  |  mask {m['ssim_mask']:.4f}")
        axes[0, j].set_title(title + sub, fontsize=10, color=color)
        axes[0, j].axis('off')

    zoom_imgs = [center_crop(im, zoom_size) for im in images]
    for j, (img, title, color) in enumerate(zip(zoom_imgs, titles, colors)):
        axes[1, j].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, j].set_title(f'{title}  zoom {zoom_size}×{zoom_size}',
                             fontsize=10, color=color)
        axes[1, j].axis('off')

    mask = make_mask(gt)
    axes[2, 0].imshow(center_crop(mask.astype(np.float32), zoom_size), cmap='gray')
    mask_ratio = float(mask.mean())
    axes[2, 0].set_title(f'Brain mask zoom\n(coverage {mask_ratio*100:.1f}% of full)',
                         fontsize=10)
    axes[2, 0].axis('off')

    errors = [np.abs(gt - im) for im in [ss2d, eter, unet]]
    err_zooms = [center_crop(e, zoom_size) for e in errors]
    all_err = np.concatenate([e.ravel() for e in err_zooms])
    err_vmax = float(np.percentile(all_err, 99))
    err_vmax = max(err_vmax, 1e-8)

    for j, (e, color, name) in enumerate(zip(err_zooms,
                                              ['tab:blue', 'tab:green', 'tab:orange'],
                                              ['SS2D', 'ETER', 'U-Net'])):
        im = axes[2, j + 1].imshow(e, cmap='hot', vmin=0, vmax=err_vmax)
        axes[2, j + 1].set_title(f'{name} |error| zoom\n(shared p99 scale)',
                                  fontsize=10, color=color)
        axes[2, j + 1].axis('off')
        plt.colorbar(im, ax=axes[2, j + 1], fraction=0.046)

    fig.suptitle(f'[{tag.upper()}] Sample #{slice_idx}',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ss2d-ckpt', default='logs/SS2D_ViT_R4_brain320_v6/ss2d_vit_best.pt')
    p.add_argument('--eter-ckpt', default='logs/ETER_ViT_R4_brain320_v6/eter_vit_best.pt')
    p.add_argument('--unet-ckpt', default='models/pretrained/brain_leaderboard_state_dict.pt')
    p.add_argument('--csv', default='results/eval/eval_full_v6.csv')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--output-dir', default='results/vis/vis_diagnostic_v6')
    p.add_argument('--n-each', type=int, default=3)
    p.add_argument('--zoom', type=int, default=160)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('====================================================')
    print(' Diagnostic Vis (v6): full + zoom + masked SSIM')
    print('====================================================')
    print(f'Device: {device}')

    print('\n슬라이스 선정 중...')
    slices = select_slices(args.csv, n_each=args.n_each)
    for tag, idx, ssim in slices:
        print(f'  [{tag:>6s}] idx={idx:5d}  ss2d_ssim={ssim:.4f}')

    print('\n모델 로드 중...')
    ss2d_m = load_ss2d(args.ss2d_ckpt, device)
    eter_m = load_eter(args.eter_ckpt, device)
    unet_m = load_unet(args.unet_ckpt, device)
    print('  3 모델 로드 완료.')

    print('\n데이터 파이프라인 준비 중...')
    mask_func = create_mask_for_mask_type('equispaced_fraction', [0.08], [4])
    unet_tr = T.UnetDataTransform(which_challenge='multicoil',
                                   mask_func=mask_func, use_seed=True)
    unet_ds = SliceDataset(root=args.data_path, transform=unet_tr, challenge='multicoil')

    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    h5_ds = FastMRI_H5_Dataloader(args.data_path, num_files=None,
                                   random_mask=False, augment=False)
    print(f'  unet_ds={len(unet_ds)}  h5_ds={len(h5_ds)}')

    os.makedirs(args.output_dir, exist_ok=True)

    summary_lines = [
        '========== Diagnostic Vis (v6) ==========',
        f'슬라이스 {len(slices)}개 (best {args.n_each} + median {args.n_each} + worst {args.n_each}, SS2D SSIM 기준)',
        '',
        f'{"tag":>7s} | {"idx":>5s} | {"model":>5s} | {"raw SSIM":>9s} | {"mask SSIM":>10s} | {"diff":>7s} | {"PSNR":>8s}',
        f'{"-"*7} | {"-"*5} | {"-"*5} | {"-"*9} | {"-"*10} | {"-"*7} | {"-"*8}',
    ]

    with torch.no_grad():
        for tag, idx, _ in tqdm(slices, desc='Diagnostic', unit='img'):
            image, target, mean, std, fname, slice_num, max_value = unet_ds[idx]
            u_out = unet_m(image.unsqueeze(0).unsqueeze(1).to(device)).squeeze().cpu()
            u_recon = (u_out * std + mean).numpy()
            u_gt = (target * std + mean).numpy()
            u_recon = crop_or_pad(u_recon, (320, 320))
            u_gt = crop_or_pad(u_gt, (320, 320))

            s = h5_ds[idx]
            d_in = torch.tensor(s['data']).unsqueeze(0).float().to(device)
            d_in_img = torch.tensor(s['data_img']).unsqueeze(0).float().to(device)
            d_ref = torch.tensor(s['label']).unsqueeze(0).float().to(device)
            m_in = torch.tensor(s['mask']).unsqueeze(0).float().to(device)
            sens_in = torch.tensor(s['sens']).unsqueeze(0).float().to(device)

            with torch.amp.autocast('cuda'):
                ss_out = ss2d_m(d_in_img, d_in, m_in, sens_in)
                et_out = eter_m(d_in_img, d_in)

            ss_recon = ss_out.squeeze().float().cpu().numpy()
            et_recon = et_out.squeeze().float().cpu().numpy()
            h5_gt = d_ref.squeeze().float().cpu().numpy()

            if h5_gt.max() > 0:
                ss_disp = ss_recon / h5_gt.max() * u_gt.max()
                et_disp = et_recon / h5_gt.max() * u_gt.max()
            else:
                ss_disp = ss_recon.copy()
                et_disp = et_recon.copy()

            mask_h5 = make_mask(h5_gt)
            mask_u = make_mask(u_gt)

            ss_psnr = calc_psnr(ss_recon, h5_gt)
            ss_ssim_raw = calc_ssim_raw(ss_recon, h5_gt)
            ss_ssim_mask = calc_ssim_masked(ss_recon, h5_gt, mask_h5)

            et_psnr = calc_psnr(et_recon, h5_gt)
            et_ssim_raw = calc_ssim_raw(et_recon, h5_gt)
            et_ssim_mask = calc_ssim_masked(et_recon, h5_gt, mask_h5)

            u_psnr = calc_psnr(u_recon, u_gt)
            u_ssim_raw = calc_ssim_raw(u_recon, u_gt)
            u_ssim_mask = calc_ssim_masked(u_recon, u_gt, mask_u)

            metrics = [
                {'psnr': ss_psnr, 'ssim_raw': ss_ssim_raw, 'ssim_mask': ss_ssim_mask},
                {'psnr': et_psnr, 'ssim_raw': et_ssim_raw, 'ssim_mask': et_ssim_mask},
                {'psnr': u_psnr,  'ssim_raw': u_ssim_raw,  'ssim_mask': u_ssim_mask},
            ]

            save_path = os.path.join(args.output_dir, f'{tag}_idx{idx:05d}.png')
            make_diagnostic_figure(idx, tag, u_gt, ss_disp, et_disp, u_recon,
                                    metrics, save_path, zoom_size=args.zoom)

            for name, m in zip(['SS2D', 'ETER', 'UNet'], metrics):
                diff = m['ssim_mask'] - m['ssim_raw']
                summary_lines.append(
                    f'{tag:>7s} | {idx:>5d} | {name:>5s} | '
                    f'{m["ssim_raw"]:>9.4f} | {m["ssim_mask"]:>10.4f} | '
                    f'{diff:+7.4f} | {m["psnr"]:>5.2f}dB'
                )

    print('\n' + '\n'.join(summary_lines))
    summary_path = os.path.join(args.output_dir, 'diagnostic_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f'\n저장 완료: {args.output_dir}/')


if __name__ == '__main__':
    main()
