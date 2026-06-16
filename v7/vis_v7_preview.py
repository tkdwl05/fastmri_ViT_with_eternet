"""
v7 미리보기 시각화: SS2D-ViT v7 vs ETER-ViT v7 vs GT
  - 24GB VRAM 환경 → 두 모델 동시 로드 (v6 의 mode 분리 patch 폐지)
  - vis_v6_preview.py 의 2x3 grid 그대로, ckpt 만 v7 로 교체
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
from skimage.metrics import structural_similarity as compare_ssim

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))


def calc_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(target.max() / np.sqrt(mse))


def calc_ssim(pred, target):
    dr = target.max() - target.min()
    return compare_ssim(target, pred, data_range=dr)


def _load_ddp_state(path, device):
    state = torch.load(path, map_location=device)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    return state


def load_ss2d_v7(ckpt_path, device):
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
        channels=INPUT_CHANNELS, dropout=0.0, emb_dropout=0.0,
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
    model.load_state_dict(_load_ddp_state(ckpt_path, device))
    model = model.to(device); model.eval()
    print(f'  SS2D-ViT v7 로드: {ckpt_path}')
    return model


def load_eter_v7(ckpt_path, device):
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
        channels=INPUT_CHANNELS, dropout=0.0, emb_dropout=0.0,
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
    model.load_state_dict(_load_ddp_state(ckpt_path, device))
    model = model.to(device); model.eval()
    print(f'  ETER-ViT v7 로드: {ckpt_path}')
    return model


def make_figure(gt, ss2d, eter, ms, me, idx, save_path):
    err_ss2d = np.abs(ss2d - gt)
    err_eter = np.abs(eter - gt)
    vmax = max(gt.max(), ss2d.max(), eter.max())
    err_max = max(err_ss2d.max(), err_eter.max())

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(gt, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 0].set_title('Ground Truth (RSS)', fontsize=11)
    axes[0, 0].axis('off')
    axes[0, 1].imshow(ss2d, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 1].set_title(f'SS2D-ViT v7\nPSNR {ms["psnr"]:.2f} dB | SSIM {ms["ssim"]:.4f}', fontsize=11, color='tab:blue')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(eter, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 2].set_title(f'ETER-ViT v7\nPSNR {me["psnr"]:.2f} dB | SSIM {me["ssim"]:.4f}', fontsize=11, color='tab:green')
    axes[0, 2].axis('off')

    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, f'sample idx={idx}', ha='center', va='center', fontsize=12)
    axes[1, 1].imshow(err_ss2d, cmap='hot', vmin=0, vmax=err_max)
    axes[1, 1].set_title('|SS2D v7 − GT|', fontsize=11, color='tab:blue')
    axes[1, 1].axis('off')
    axes[1, 2].imshow(err_eter, cmap='hot', vmin=0, vmax=err_max)
    axes[1, 2].set_title('|ETER v7 − GT|', fontsize=11, color='tab:green')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ss2d-ckpt', type=str,
                        default='logs/SS2D_ViT_R4_brain320_v7/ss2d_vit_best.pt')
    parser.add_argument('--eter-ckpt', type=str,
                        default='logs/ETER_ViT_R4_brain320_v7/eter_vit_best.pt')
    parser.add_argument('--data-path', type=str, default='./fastMRI_data/multicoil_val')
    parser.add_argument('--output-dir', type=str, default='results/vis_v7_preview')
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'===== v7 preview (device={device}) =====')

    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    ds = FastMRI_H5_Dataloader(args.data_path, num_files=None, random_mask=False, augment=False)
    total = len(ds)
    indices = np.linspace(0, total - 1, args.num_samples, dtype=int).tolist()
    print(f'총 {total} slices, 시각화 대상 {len(indices)}개')

    ss2d_model = load_ss2d_v7(args.ss2d_ckpt, device)
    eter_model = load_eter_v7(args.eter_ckpt, device)

    os.makedirs(args.output_dir, exist_ok=True)
    psnrs_s, ssims_s, psnrs_e, ssims_e = [], [], [], []

    for idx in tqdm(indices, desc='infer + plot'):
        s = ds[idx]
        data_in     = torch.tensor(s['data']).unsqueeze(0).float().to(device)
        data_in_img = torch.tensor(s['data_img']).unsqueeze(0).float().to(device)
        mask_in     = torch.tensor(s['mask']).unsqueeze(0).float().to(device)
        sens_in     = torch.tensor(s['sens']).unsqueeze(0).float().to(device)
        gt          = np.asarray(s['label']).squeeze()

        with torch.no_grad(), torch.amp.autocast('cuda'):
            ss2d_out = ss2d_model(data_in_img, data_in, mask_in, sens_in)
            eter_out = eter_model(data_in_img, data_in)

        ss2d = ss2d_out.squeeze().cpu().numpy()
        eter = eter_out.squeeze().cpu().numpy()

        ms = {'psnr': calc_psnr(ss2d, gt), 'ssim': calc_ssim(ss2d, gt)}
        me = {'psnr': calc_psnr(eter, gt), 'ssim': calc_ssim(eter, gt)}
        psnrs_s.append(ms['psnr']); ssims_s.append(ms['ssim'])
        psnrs_e.append(me['psnr']); ssims_e.append(me['ssim'])

        make_figure(gt, ss2d, eter, ms, me, idx,
                    os.path.join(args.output_dir, f'preview_v7_{idx:04d}.png'))

    summary = [
        '===== v7 preview 요약 =====',
        f'샘플 수: {len(indices)}',
        f'SS2D-ViT v7 : PSNR {np.mean(psnrs_s):.2f}±{np.std(psnrs_s):.2f} | SSIM {np.mean(ssims_s):.4f}±{np.std(ssims_s):.4f}',
        f'ETER-ViT v7 : PSNR {np.mean(psnrs_e):.2f}±{np.std(psnrs_e):.2f} | SSIM {np.mean(ssims_e):.4f}±{np.std(ssims_e):.4f}',
    ]
    print('\n' + '\n'.join(summary))
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(summary) + '\n')


if __name__ == '__main__':
    main()
