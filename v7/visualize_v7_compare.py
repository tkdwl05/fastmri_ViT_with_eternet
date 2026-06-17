"""
v7 비교 시각화 스크립트
  - 기존 4-모델 비교 (GT | SS2D-v4 | ETER-v4 | U-Net) 를 그대로 유지하고
    맨 오른쪽에 v7 슬롯 1개를 추가 → 2x5 grid.
  - v7 슬롯은 --v7-model {ss2d,eter} 와 --v7-ckpt 로 지정.
  - 기존 visualize_compare.py 는 무변경 보존, 이 파일은 클론본.
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

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))                               # v7 configs
sys.path.append(os.path.join(_PROJECT_ROOT, 'configs'))                       # v4/v6 configs
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))

from skimage.metrics import structural_similarity as compare_ssim


def center_crop_numpy(data: np.ndarray, shape: tuple):
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


def load_ss2d_v4_model(ckpt_path, device):
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
    model = model.to(device); model.eval()
    print(f"  SS2D-ViT v4 로드 완료: {ckpt_path}")
    return model


def load_eter_v4_model(ckpt_path, device):
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
    model = model.to(device); model.eval()
    print(f"  ETER-ViT v4 로드 완료: {ckpt_path}")
    return model


def load_unet_model(state_dict_path, device):
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    model = model.to(device); model.eval()
    print(f"  U-Net 로드 완료: {state_dict_path}")
    return model


def load_v7_ss2d_model(ckpt_path, device):
    from myConfig_choh_SS2D_model_v7 import (
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
    state = torch.load(ckpt_path, map_location=device)
    if any(k.startswith('module.') for k in state.keys()):  # DDP ckpt
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device); model.eval()
    print(f"  SS2D-ViT v7 로드 완료: {ckpt_path}")
    return model


def load_v7_eter_model(ckpt_path, device):
    from myConfig_choh_ETER_model_v7 import (
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
    encoder = choh_ViT(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
        dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
        heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
        channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1
    )
    model = choh_Decoder3_ETER_v5(
        encoder=encoder,
        eter_n_hori_hidden=NUM_ETER_HORI_HIDDEN,
        eter_n_vert_hidden=NUM_ETER_VERT_HIDDEN,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=DROPOUT,
    )
    state = torch.load(ckpt_path, map_location=device)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device); model.eval()
    print(f"  ETER-ViT v7 로드 완료: {ckpt_path}")
    return model


def make_comparison_figure(gt, recons, errors, metrics, labels, colors,
                           sample_idx, save_path):
    """recons/errors/metrics/labels/colors 는 모두 같은 길이의 리스트.
    레이아웃: 2 rows × (1 + N) cols. 첫 컬럼은 GT (상) / hidden (하)."""
    n_models = len(recons)
    n_cols = 1 + n_models
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 11))

    vmin, vmax = gt.min(), gt.max()
    error_max = max(np.max(e) for e in errors) * 0.5 if errors else 1.0

    axes[0, 0].imshow(gt, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('GT (Ground Truth)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].set_visible(False)

    for i, (recon, err, m, label, color) in enumerate(
            zip(recons, errors, metrics, labels, colors)):
        col = 1 + i
        axes[0, col].imshow(recon, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'{label}\nPSNR: {m["psnr"]:.2f}dB | SSIM: {m["ssim"]:.4f}',
                                fontsize=12, color=color)
        axes[0, col].axis('off')

        im = axes[1, col].imshow(err, cmap='hot', vmin=0, vmax=error_max)
        axes[1, col].set_title(f'{label} Error', fontsize=12, color=color)
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)

    fig.suptitle(f'Sample #{sample_idx}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='v7 비교 시각화 (4-모델 baseline + v7 슬롯)')
    parser.add_argument('--unet-state-dict', type=str,
                        default='models/pretrained/brain_leaderboard_state_dict.pt')
    parser.add_argument('--ss2d-ckpt', type=str,
                        default='logs/SS2D_ViT_R4_brain320_v4/ss2d_vit_best.pt')
    parser.add_argument('--eter-ckpt', type=str,
                        default='logs/ETER_ViT_R4_brain320_v4/eter_vit_best.pt')
    parser.add_argument('--v7-model', type=str, default=None, choices=['ss2d', 'eter'],
                        help='v7 슬롯에 넣을 모델 종류 (생략 시 v7 슬롯 비활성)')
    parser.add_argument('--v7-ckpt', type=str, default=None, help='v7 ckpt 경로')
    parser.add_argument('--data-path', type=str, default='./fastMRI_data/multicoil_val')
    parser.add_argument('--output-dir', type=str, default='results/vis_compare_v7')
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('====================================================')
    print(' v7 비교 시각화: SS2D-v4 | ETER-v4 | U-Net | [+ v7 슬롯]')
    print('====================================================')

    print('\n베이스라인 모델 로드 중...')
    ss2d_model = load_ss2d_v4_model(args.ss2d_ckpt, device)
    eter_model = load_eter_v4_model(args.eter_ckpt, device)
    unet_model = load_unet_model(args.unet_state_dict, device)

    v7_model = None
    v7_label = None
    if args.v7_model and args.v7_ckpt:
        if args.v7_model == 'ss2d':
            v7_model = load_v7_ss2d_model(args.v7_ckpt, device)
            v7_label = 'SS2D-ViT v7'
        else:
            v7_model = load_v7_eter_model(args.v7_ckpt, device)
            v7_label = 'ETER-ViT v7'
    else:
        print('\nv7 슬롯 비활성 (--v7-model + --v7-ckpt 미지정)')

    mask_func = create_mask_for_mask_type('equispaced_fraction', [0.08], [4])
    unet_transform = T.UnetDataTransform(which_challenge="multicoil", mask_func=mask_func, use_seed=True)
    unet_dataset = SliceDataset(root=args.data_path, transform=unet_transform, challenge="multicoil")

    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    ss2d_dataset = FastMRI_H5_Dataloader(args.data_path, num_files=None, random_mask=False, augment=False)

    total = min(len(unet_dataset), len(ss2d_dataset))
    indices = np.linspace(0, total - 1, args.num_samples, dtype=int).tolist()
    print(f"총 슬라이스: {total}, 시각화 대상: {len(indices)}개\n")
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {'ss2d_v4': [], 'eter_v4': [], 'unet': [], 'v7': []}

    for idx in tqdm(indices, desc='시각화 생성', unit='img'):
        unet_sample = unet_dataset[idx]
        image, target, mean, std, fname, slice_num, max_value = unet_sample
        with torch.no_grad():
            unet_out = unet_model(image.unsqueeze(0).unsqueeze(1).to(device)).squeeze().cpu()
        unet_recon = center_crop_numpy((unet_out * std + mean).numpy(), (320, 320))
        unet_gt    = center_crop_numpy((target  * std + mean).numpy(), (320, 320))

        ss2d_sample = ss2d_dataset[idx]
        data_in     = torch.tensor(ss2d_sample['data']).unsqueeze(0).float().to(device)
        data_in_img = torch.tensor(ss2d_sample['data_img']).unsqueeze(0).float().to(device)
        data_ref    = torch.tensor(ss2d_sample['label']).unsqueeze(0).float().to(device)
        mask_in     = torch.tensor(ss2d_sample['mask']).unsqueeze(0).float().to(device)
        sens_in     = torch.tensor(ss2d_sample['sens']).unsqueeze(0).float().to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                ss2d_out = ss2d_model(data_in_img, data_in, mask_in, sens_in)
                eter_out = eter_model(data_in_img, data_in)
                v7_out = None
                if v7_model is not None:
                    if args.v7_model == 'ss2d':
                        v7_out = v7_model(data_in_img, data_in, mask_in, sens_in)
                    else:
                        v7_out = v7_model(data_in_img, data_in)

        ss2d_recon = ss2d_out.squeeze().cpu().numpy()
        eter_recon = eter_out.squeeze().cpu().numpy()
        v7_recon   = v7_out.squeeze().cpu().numpy() if v7_out is not None else None
        gt_raw     = data_ref.squeeze().cpu().numpy()

        if gt_raw.max() > 0:
            scale = unet_gt.max() / gt_raw.max()
            ss2d_recon_n = ss2d_recon * scale
            eter_recon_n = eter_recon * scale
            v7_recon_n   = v7_recon   * scale if v7_recon is not None else None
        else:
            ss2d_recon_n, eter_recon_n, v7_recon_n = ss2d_recon, eter_recon, v7_recon

        m_ss2d = {'psnr': calc_psnr(ss2d_recon, gt_raw), 'ssim': calc_ssim(ss2d_recon, gt_raw)}
        m_eter = {'psnr': calc_psnr(eter_recon, gt_raw), 'ssim': calc_ssim(eter_recon, gt_raw)}
        m_unet = {'psnr': calc_psnr(unet_recon, unet_gt), 'ssim': calc_ssim(unet_recon, unet_gt)}
        m_v7   = {'psnr': calc_psnr(v7_recon, gt_raw),   'ssim': calc_ssim(v7_recon, gt_raw)} if v7_recon is not None else None

        all_metrics['ss2d_v4'].append(m_ss2d)
        all_metrics['eter_v4'].append(m_eter)
        all_metrics['unet'].append(m_unet)
        if m_v7 is not None:
            all_metrics['v7'].append(m_v7)

        recons  = [ss2d_recon_n, eter_recon_n, unet_recon]
        errors  = [np.abs(unet_gt - r) for r in recons]
        metrics = [m_ss2d, m_eter, m_unet]
        labels  = ['SS2D-ViT v4', 'ETER-ViT v4', 'U-Net (Pre-trained)']
        colors  = ['tab:blue',     'tab:green',   'tab:orange']

        if v7_recon_n is not None:
            recons.append(v7_recon_n)
            errors.append(np.abs(unet_gt - v7_recon_n))
            metrics.append(m_v7)
            labels.append(v7_label)
            colors.append('tab:red')

        make_comparison_figure(
            gt=unet_gt,
            recons=recons, errors=errors, metrics=metrics,
            labels=labels, colors=colors,
            sample_idx=idx,
            save_path=os.path.join(args.output_dir, f'compare_v7_{idx:04d}.png'),
        )

    def _stat(lst, key):
        if not lst:
            return 'N/A'
        vals = [m[key] for m in lst]
        return f'{np.mean(vals):>6.4f}±{np.std(vals):<6.4f}' if key == 'ssim' else f'{np.mean(vals):>6.2f}±{np.std(vals):<5.2f}'

    summary = [
        '========== v7 비교 요약 (선택 샘플) ==========',
        f'비교 샘플 수: {len(indices)}',
        '',
        f'{"모델":>15s} | {"PSNR(dB)":>14s} | {"SSIM":>14s}',
        f'{"-"*15} | {"-"*14} | {"-"*14}',
        f'{"SS2D-ViT v4":>15s} | {_stat(all_metrics["ss2d_v4"], "psnr"):>14s} | {_stat(all_metrics["ss2d_v4"], "ssim"):>14s}',
        f'{"ETER-ViT v4":>15s} | {_stat(all_metrics["eter_v4"], "psnr"):>14s} | {_stat(all_metrics["eter_v4"], "ssim"):>14s}',
        f'{"U-Net (PT)":>15s} | {_stat(all_metrics["unet"], "psnr"):>14s} | {_stat(all_metrics["unet"], "ssim"):>14s}',
        f'{(v7_label or "v7 (off)"):>15s} | {_stat(all_metrics["v7"], "psnr"):>14s} | {_stat(all_metrics["v7"], "ssim"):>14s}',
    ]
    print('\n' + '\n'.join(summary))

    with open(os.path.join(args.output_dir, 'comparison_v7_summary.txt'), 'w') as f:
        f.write('\n'.join(summary) + '\n')

    print(f'\n시각화 저장 완료: {args.output_dir}/')


if __name__ == '__main__':
    main()
