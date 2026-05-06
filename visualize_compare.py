"""
모델 비교 시각화 스크립트: GT | SS2D-ViT v4 | ETER-ViT v4 | U-Net (Pre-trained)

동일 파일의 슬라이스에 대해 4열 비교 이미지 + 에러맵을 생성한다.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import fastmri
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models import Unet

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'configs'))
sys.path.append(os.path.join(current_dir, 'dataloaders'))
sys.path.append(os.path.join(current_dir, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(current_dir, 'models', 'mamba_eternet'))

from skimage.metrics import structural_similarity as compare_ssim

def center_crop_numpy(data: np.ndarray, shape: tuple):
    """크기가 큰 축은 center crop, 작은 축은 zero pad — 항상 shape 출력."""
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
            pad_before = pad // 2
            pad_after = pad - pad_before
            pad_widths = [(0, 0)] * out.ndim
            pad_widths[axis] = (pad_before, pad_after)
            out = np.pad(out, pad_widths, mode='constant', constant_values=0)
    return out

def calc_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(target.max() / np.sqrt(mse))

def calc_ssim(pred, target):
    data_range = target.max() - target.min()
    return compare_ssim(target, pred, data_range=data_range)

def load_ss2d_model(ckpt_path, device):
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
    print(f"  SS2D-ViT v4 로드 완료: {ckpt_path}")
    return model

def load_eter_model(ckpt_path, device):
    from myConfig_choh_ETER_model import (
        IMAGE_SIZE, PATCH_SIZE, INPUT_CHANNELS,
        NUM_VIT_ENCODER_HIDDEN, NUM_VIT_ENCODER_LAYER,
        NUM_VIT_ENCODER_MLP_SIZE, NUM_VIT_ENCODER_HEAD,
        NUM_ETER_HORI_HIDDEN, NUM_ETER_VERT_HIDDEN,
        NUM_VIT_DECODER_DIM, NUM_VIT_DECODER_DEPTH,
        NUM_VIT_DECODER_HEAD, NUM_VIT_DECODER_DIM_HEAD,
        NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
    )
    from u_choh_model_ETER_ViT import choh_ViT, choh_Decoder3_ETER_skip_up_tail

    encoder = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1
    )
    model = choh_Decoder3_ETER_skip_up_tail(
        encoder=encoder,
        eter_n_hori_hidden=NUM_ETER_HORI_HIDDEN,
        eter_n_vert_hidden=NUM_ETER_VERT_HIDDEN,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"  ETER-ViT 로드 완료: {ckpt_path}")
    return model

def load_unet_model(state_dict_path, device):
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"  U-Net 로드 완료: {state_dict_path}")
    return model

def make_comparison_figure(gt, ss2d_recon, eter_recon, unet_recon,
                           ss2d_psnr, ss2d_ssim,
                           eter_psnr, eter_ssim,
                           unet_psnr, unet_ssim,
                           sample_idx, save_path):
    ss2d_error = np.abs(gt - ss2d_recon)
    eter_error = np.abs(gt - eter_recon)
    unet_error = np.abs(gt - unet_recon)
    error_max = max(ss2d_error.max(), eter_error.max(), unet_error.max(), 1e-8) * 0.5

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    vmin, vmax = gt.min(), gt.max()

    axes[0, 0].imshow(gt, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('GT (Ground Truth)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(ss2d_recon, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'SS2D-ViT v4\nPSNR: {ss2d_psnr:.2f}dB | SSIM: {ss2d_ssim:.4f}',
                         fontsize=12, color='tab:blue')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(eter_recon, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'ETER-ViT v4\nPSNR: {eter_psnr:.2f}dB | SSIM: {eter_ssim:.4f}',
                         fontsize=12, color='tab:green')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(unet_recon, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 3].set_title(f'U-Net (Pre-trained)\nPSNR: {unet_psnr:.2f}dB | SSIM: {unet_ssim:.4f}',
                         fontsize=12, color='tab:orange')
    axes[0, 3].axis('off')

    axes[1, 0].set_visible(False)

    im1 = axes[1, 1].imshow(ss2d_error, cmap='hot', vmin=0, vmax=error_max)
    axes[1, 1].set_title('SS2D Error', fontsize=12, color='tab:blue')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)

    im2 = axes[1, 2].imshow(eter_error, cmap='hot', vmin=0, vmax=error_max)
    axes[1, 2].set_title('ETER Error', fontsize=12, color='tab:green')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    im3 = axes[1, 3].imshow(unet_error, cmap='hot', vmin=0, vmax=error_max)
    axes[1, 3].set_title('U-Net Error', fontsize=12, color='tab:orange')
    axes[1, 3].axis('off')
    plt.colorbar(im3, ax=axes[1, 3], fraction=0.046)

    fig.suptitle(f'Sample #{sample_idx}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='모델 비교 시각화')
    parser.add_argument('--unet-state-dict', type=str,
                        default='models/pretrained/brain_leaderboard_state_dict.pt')
    parser.add_argument('--ss2d-ckpt', type=str,
                        default='logs/SS2D_ViT_R4_brain320_v4/ss2d_vit_best.pt')
    parser.add_argument('--eter-ckpt', type=str,
                        default='logs/ETER_ViT_R4_brain320_v4/eter_vit_best.pt')
    parser.add_argument('--data-path', type=str,
                        default='./fastMRI_data/multicoil_val')
    parser.add_argument('--output-dir', type=str, default='results/vis_compare')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='시각화할 샘플 수')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('====================================================')
    print(' 시각화 비교: SS2D-ViT v4 vs ETER-ViT v4 vs U-Net (Pre-trained)')
    print('====================================================')

    print('\n모델 로드 중...')
    ss2d_model = load_ss2d_model(args.ss2d_ckpt, device)
    eter_model = load_eter_model(args.eter_ckpt, device)
    unet_model = load_unet_model(args.unet_state_dict, device)

    mask_func = create_mask_for_mask_type('equispaced_fraction', [0.08], [4])
    unet_transform = T.UnetDataTransform(which_challenge="multicoil", mask_func=mask_func, use_seed=True)
    unet_dataset = SliceDataset(root=args.data_path, transform=unet_transform, challenge="multicoil")

    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    ss2d_dataset = FastMRI_H5_Dataloader(args.data_path, num_files=None, random_mask=False, augment=False)

    total = min(len(unet_dataset), len(ss2d_dataset))
    indices = np.linspace(0, total - 1, args.num_samples, dtype=int).tolist()

    print(f"총 슬라이스: {total}, 시각화 대상: {len(indices)}개\n")
    os.makedirs(args.output_dir, exist_ok=True)

    all_ss2d_psnr, all_ss2d_ssim = [], []
    all_eter_psnr, all_eter_ssim = [], []
    all_unet_psnr, all_unet_ssim = [], []

    for idx in tqdm(indices, desc='시각화 생성', unit='img'):
        unet_sample = unet_dataset[idx]
        image, target, mean, std, fname, slice_num, max_value = unet_sample

        with torch.no_grad():
            unet_out = unet_model(image.unsqueeze(0).unsqueeze(1).to(device)).squeeze().cpu()

        unet_recon = (unet_out * std + mean).numpy()
        unet_gt = (target * std + mean).numpy()
        unet_recon = center_crop_numpy(unet_recon, (320, 320))
        unet_gt = center_crop_numpy(unet_gt, (320, 320))

        ss2d_sample = ss2d_dataset[idx]
        data_in = torch.tensor(ss2d_sample['data']).unsqueeze(0).float().to(device)
        data_in_img = torch.tensor(ss2d_sample['data_img']).unsqueeze(0).float().to(device)
        data_ref = torch.tensor(ss2d_sample['label']).unsqueeze(0).float().to(device)
        mask_in = torch.tensor(ss2d_sample['mask']).unsqueeze(0).float().to(device)
        sens_in = torch.tensor(ss2d_sample['sens']).unsqueeze(0).float().to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                ss2d_out = ss2d_model(data_in_img, data_in, mask_in, sens_in)
                eter_out = eter_model(data_in_img, data_in)

        ss2d_recon_raw = ss2d_out.squeeze().cpu().numpy()
        eter_recon_raw = eter_out.squeeze().cpu().numpy()
        ss2d_gt_raw = data_ref.squeeze().cpu().numpy()

        if ss2d_gt_raw.max() > 0:
            ss2d_recon_norm = ss2d_recon_raw / ss2d_gt_raw.max() * unet_gt.max()
            eter_recon_norm = eter_recon_raw / ss2d_gt_raw.max() * unet_gt.max()
        else:
            ss2d_recon_norm = ss2d_recon_raw
            eter_recon_norm = eter_recon_raw

        unet_psnr = calc_psnr(unet_recon, unet_gt)
        unet_ssim = calc_ssim(unet_recon, unet_gt)
        
        ss2d_psnr = calc_psnr(ss2d_recon_raw, ss2d_gt_raw)
        ss2d_ssim = calc_ssim(ss2d_recon_raw, ss2d_gt_raw)

        eter_psnr = calc_psnr(eter_recon_raw, ss2d_gt_raw)
        eter_ssim = calc_ssim(eter_recon_raw, ss2d_gt_raw)

        all_unet_psnr.append(unet_psnr)
        all_unet_ssim.append(unet_ssim)
        all_ss2d_psnr.append(ss2d_psnr)
        all_ss2d_ssim.append(ss2d_ssim)
        all_eter_psnr.append(eter_psnr)
        all_eter_ssim.append(eter_ssim)

        make_comparison_figure(
            gt=unet_gt,
            ss2d_recon=ss2d_recon_norm,
            eter_recon=eter_recon_norm,
            unet_recon=unet_recon,
            ss2d_psnr=ss2d_psnr,
            ss2d_ssim=ss2d_ssim,
            eter_psnr=eter_psnr,
            eter_ssim=eter_ssim,
            unet_psnr=unet_psnr,
            unet_ssim=unet_ssim,
            sample_idx=idx,
            save_path=os.path.join(args.output_dir, f'compare_{idx:04d}.png'),
        )

    summary = [
        '========== 모델 비교 요약 (선택 샘플) ==========',
        f'비교 샘플 수: {len(indices)}',
        '',
        f'{"모델":>15s} | {"PSNR(dB)":>12s} | {"SSIM":>12s}',
        f'{"-"*15} | {"-"*12} | {"-"*12}',
        f'{"SS2D-ViT v4":>15s} | {np.mean(all_ss2d_psnr):>6.2f}±{np.std(all_ss2d_psnr):<5.2f} | {np.mean(all_ss2d_ssim):>6.4f}±{np.std(all_ss2d_ssim):<6.4f}',
        f'{"ETER-ViT v4":>15s} | {np.mean(all_eter_psnr):>6.2f}±{np.std(all_eter_psnr):<5.2f} | {np.mean(all_eter_ssim):>6.4f}±{np.std(all_eter_ssim):<6.4f}',
        f'{"U-Net (PT)":>15s} | {np.mean(all_unet_psnr):>6.2f}±{np.std(all_unet_psnr):<5.2f} | {np.mean(all_unet_ssim):>6.4f}±{np.std(all_unet_ssim):<6.4f}',
    ]
    print('\n' + '\n'.join(summary))

    summary_path = os.path.join(args.output_dir, 'comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary) + '\n')

    print(f'\n시각화 저장 완료: {args.output_dir}/')

if __name__ == '__main__':
    main()
