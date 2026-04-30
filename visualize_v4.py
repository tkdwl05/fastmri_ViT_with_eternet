"""
SS2D-ViT v4 시각화 스크립트 (DC block + mask/sens 입력 지원)

원본 visualize.py는 v3 시그니처(model(img, ksp))만 지원하므로 v4 전용으로 분리.
ss2d_v4_changes.md §2 원칙 준수.

사용법:
  python visualize_v4.py
  python visualize_v4.py --ckpt logs/SS2D_ViT_R4_brain320_v4/ss2d_vit_best.pt --top_k 10

저장 위치 (기본):
  results/vis_ss2d_v4_<ckpt_basename>/
    ss2d_v4_best_01_idx{n}.png ...
    ss2d_v4_worst_01_idx{n}.png ...
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'configs'))
sys.path.append(os.path.join(current_dir, 'dataloaders'))
sys.path.append(os.path.join(current_dir, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(current_dir, 'models', 'mamba_eternet'))

from dataloader_h5_v4 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader
from u_choh_SSIM import SSIM


def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(target.max() / torch.sqrt(mse))).item()


def calc_nmse(pred, target):
    return (torch.norm(pred - target) ** 2 / torch.norm(target) ** 2).item()


def load_model_v4(ckpt_path: str, device):
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

    encoder = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1
    )
    model = choh_Decoder_SS2D_ViT(
        encoder=encoder,
        ss2d_d_inner=NUM_SS2D_D_INNER, ss2d_d_state=NUM_SS2D_D_STATE,
        ss2d_out_ch=NUM_SS2D_OUT_CH,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=DROPOUT,
        dc_k_scale_ratio=DC_K_SCALE_RATIO,
        dc_init_alpha=DC_INIT_ALPHA,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def aliased_to_sos(data_img_np: np.ndarray) -> np.ndarray:
    real = data_img_np[0::2]
    imag = data_img_np[1::2]
    return np.sqrt(np.sum(real ** 2 + imag ** 2, axis=0))


def compute_composite(metrics_list: list) -> np.ndarray:
    def minmax(arr, higher_better):
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.ones_like(arr) * 0.5
        norm = (arr - lo) / (hi - lo)
        return norm if higher_better else 1.0 - norm

    ssim_arr = np.array([m['ssim'] for m in metrics_list])
    psnr_arr = np.array([m['psnr'] for m in metrics_list])
    nmse_arr = np.array([m['nmse'] for m in metrics_list])
    l1_arr   = np.array([m['l1']   for m in metrics_list])

    return (
        minmax(ssim_arr, higher_better=True) +
        minmax(psnr_arr, higher_better=True) +
        minmax(nmse_arr, higher_better=False) +
        minmax(l1_arr,   higher_better=False)
    ) / 4.0


def save_figure(gt, aliased, recon, metrics, tag, rank, total, out_dir):
    error = np.abs(gt - recon)
    vmax     = np.percentile(gt, 99)
    vmax_err = np.percentile(error, 99)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.patch.set_facecolor('#1a1a1a')

    imgs   = [gt,    aliased, recon,  error]
    titles = ['Ground Truth', 'Before (Aliased)', 'After (Reconstructed)', 'Error Map |GT − Recon|']
    cmaps  = ['gray', 'gray',   'gray',  'hot']
    vmaxes = [vmax,   vmax,     vmax,    vmax_err]

    for ax, img, title, cmap, vm in zip(axes, imgs, titles, cmaps, vmaxes):
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vm)
        ax.set_title(title, fontsize=12, color='white', pad=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    m = metrics
    fig.suptitle(
        f'[SS2D-v4 {tag.upper()}  #{rank}]   Sample idx: {m["idx"]}   '
        f'(Rank {m["rank"]} / {total})\n'
        f'SSIM: {m["ssim"]:.4f}   '
        f'PSNR: {m["psnr"]:.2f} dB   '
        f'NMSE: {m["nmse"]:.4f}   '
        f'L1: {m["l1"]:.4f}   '
        f'Composite: {m["composite"]:.4f}',
        fontsize=11, color='white', y=1.02
    )

    plt.tight_layout()
    fname = os.path.join(out_dir, f'ss2d_v4_{tag}_{rank:02d}_idx{m["idx"]}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='logs/SS2D_ViT_R4_brain320_v4/ss2d_vit_best.pt')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default=None,
                        help='시각화 PNG 저장 폴더 (기본: results/vis_ss2d_v4_<ckpt_basename>/)')
    args = parser.parse_args()
    if args.out_dir is None:
        ckpt_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
        args.out_dir = os.path.join('results', f'vis_ss2d_v4_{ckpt_stem}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('====================================================')
    print(f' 시각화: SS2D-ViT v4 | {args.ckpt}')
    print('====================================================')
    print(f'Device: {device}')

    model        = load_model_v4(args.ckpt, device)
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIM().to(device)

    dataset = FastMRI_H5_Dataloader('./fastMRI_data/multicoil_val',
                                    num_files=None, random_mask=False)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=4, pin_memory=True)
    total   = len(dataset)
    print(f'Val 샘플 수: {total}\n')

    print('[ Pass 1 ] 전체 지표 계산 중...')
    all_metrics = []

    with torch.no_grad():
        for i, sample in enumerate(tqdm(loader, desc='지표 계산')):
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            mask_in     = sample['mask'].float().to(device)
            sens_in     = sample['sens'].float().to(device)

            with torch.amp.autocast('cuda'):
                out = model(data_in_img, data_in, mask_in, sens_in)

            out_f = out.float()
            ref_f = data_ref.float()

            all_metrics.append({
                'idx':  i,
                'ssim': criterion_ssim(out_f, ref_f).item(),
                'psnr': calc_psnr(out_f, ref_f),
                'nmse': calc_nmse(out_f, ref_f),
                'l1':   criterion_l1(out_f, ref_f).item(),
            })

    composite = compute_composite(all_metrics)
    for i, m in enumerate(all_metrics):
        m['composite'] = float(composite[i])

    sorted_desc = sorted(all_metrics, key=lambda x: x['composite'], reverse=True)
    for rank, m in enumerate(sorted_desc, 1):
        m['rank'] = rank

    best_list  = sorted_desc[:args.top_k]
    worst_list = list(reversed(sorted_desc[-args.top_k:]))

    selected_indices = {m['idx'] for m in best_list + worst_list}
    metrics_by_idx   = {m['idx']: m for m in all_metrics}

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n[ Pass 2 ] 선택된 {len(selected_indices)}개 샘플 시각화 중...')
    print(f'저장 경로: {out_dir}\n')

    saved = {'best': [], 'worst': []}

    with torch.no_grad():
        for i, sample in enumerate(tqdm(loader, desc='시각화')):
            if i not in selected_indices:
                continue

            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            mask_in     = sample['mask'].float().to(device)
            sens_in     = sample['sens'].float().to(device)

            with torch.amp.autocast('cuda'):
                out = model(data_in_img, data_in, mask_in, sens_in)

            gt_np      = data_ref[0, 0].cpu().float().numpy()
            aliased_np = aliased_to_sos(data_in_img[0].cpu().float().numpy())
            recon_np   = out[0, 0].cpu().float().numpy()
            m          = metrics_by_idx[i]

            if m in best_list:
                tag  = 'best'
                rank = best_list.index(m) + 1
            else:
                tag  = 'worst'
                rank = worst_list.index(m) + 1

            fname = save_figure(gt_np, aliased_np, recon_np, m, tag, rank, total, out_dir)
            saved[tag].append((rank, m, fname))

    print('\n========== Best 샘플 ==========')
    for rank, m, fname in sorted(saved['best'], key=lambda x: x[0]):
        print(
            f'  #{rank:2d}  idx={m["idx"]:5d}'
            f'  Composite={m["composite"]:.4f}'
            f'  SSIM={m["ssim"]:.4f}'
            f'  PSNR={m["psnr"]:.2f}dB'
            f'  NMSE={m["nmse"]:.4f}'
            f'  L1={m["l1"]:.4f}'
        )

    print('\n========== Worst 샘플 ==========')
    for rank, m, fname in sorted(saved['worst'], key=lambda x: x[0]):
        print(
            f'  #{rank:2d}  idx={m["idx"]:5d}'
            f'  Composite={m["composite"]:.4f}'
            f'  SSIM={m["ssim"]:.4f}'
            f'  PSNR={m["psnr"]:.2f}dB'
            f'  NMSE={m["nmse"]:.4f}'
            f'  L1={m["l1"]:.4f}'
        )

    print(f'\n총 {len(saved["best"]) + len(saved["worst"])}개 PNG 저장 완료: {out_dir}')


if __name__ == '__main__':
    main()
