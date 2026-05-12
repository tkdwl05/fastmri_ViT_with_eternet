"""
SS2D-ViT v6 vs ETER-ViT v5 vs GT 비교 시각화 (UNet 제외, GPU 메모리 절약).
SS2D v6 학습 진행 중 미리보기용.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import gc
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'configs'))
sys.path.append(os.path.join(current_dir, 'dataloaders'))
sys.path.append(os.path.join(current_dir, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(current_dir, 'models', 'mamba_eternet'))


def calc_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(target.max() / np.sqrt(mse))


def calc_ssim(pred, target):
    dr = target.max() - target.min()
    return compare_ssim(target, pred, data_range=dr)


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
        channels=INPUT_CHANNELS, dropout=0.0, emb_dropout=0.0,
    )
    model = choh_Decoder_SS2D_ViT(
        encoder=encoder,
        ss2d_d_inner=NUM_SS2D_D_INNER,
        ss2d_d_state=NUM_SS2D_D_STATE,
        ss2d_out_ch=NUM_SS2D_OUT_CH,
        decoder_dim=NUM_VIT_DECODER_DIM,
        decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD,
        decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=DROPOUT,
        dc_k_scale_ratio=DC_K_SCALE_RATIO,
        dc_init_alpha=DC_INIT_ALPHA,
    )
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f'  SS2D-ViT v6 로드 완료: {ckpt_path}')
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
        channels=INPUT_CHANNELS, dropout=0.0, emb_dropout=0.0,
    )
    model = choh_Decoder3_ETER_skip_up_tail(
        encoder=encoder,
        eter_n_hori_hidden=NUM_ETER_HORI_HIDDEN,
        eter_n_vert_hidden=NUM_ETER_VERT_HIDDEN,
        decoder_dim=NUM_VIT_DECODER_DIM,
        decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD,
        decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f'  ETER-ViT v5 로드 완료: {ckpt_path}')
    return model


def make_figure(gt, ss2d, eter, ss2d_psnr, ss2d_ssim, eter_psnr, eter_ssim, idx, save_path):
    err_ss2d = np.abs(ss2d - gt)
    err_eter = np.abs(eter - gt)
    vmax = max(gt.max(), ss2d.max(), eter.max())
    err_max = max(err_ss2d.max(), err_eter.max())

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    axes[0, 0].imshow(gt, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 0].set_title('Ground Truth (RSS)', fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(ss2d, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 1].set_title(f'SS2D-ViT v6\nPSNR {ss2d_psnr:.2f} dB | SSIM {ss2d_ssim:.4f}', fontsize=11)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(eter, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 2].set_title(f'ETER-ViT v5\nPSNR {eter_psnr:.2f} dB | SSIM {eter_ssim:.4f}', fontsize=11)
    axes[0, 2].axis('off')

    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, f'sample idx={idx}', ha='center', va='center', fontsize=12)

    axes[1, 1].imshow(err_ss2d, cmap='hot', vmin=0, vmax=err_max)
    axes[1, 1].set_title('|SS2D − GT|', fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(err_eter, cmap='hot', vmin=0, vmax=err_max)
    axes[1, 2].set_title('|ETER − GT|', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def run_inference(mode, ckpt, data_path, num_samples, cache_dir, device):
    """단일 모델 추론을 별도 호출에서 실행. GT/sample idx 도 함께 .npz 로 저장."""
    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    ds = FastMRI_H5_Dataloader(data_path, num_files=None, random_mask=False, augment=False)
    total = len(ds)
    indices = np.linspace(0, total - 1, num_samples, dtype=int).tolist()
    print(f'[{mode}] 총 {total} slices, 추론 대상 {len(indices)}개')

    if mode == 'ss2d':
        model = load_ss2d_model(ckpt, device)
    elif mode == 'eter':
        model = load_eter_model(ckpt, device)
    else:
        raise ValueError(mode)

    outs = {}
    gts = {}
    for idx in tqdm(indices, desc=mode):
        s = ds[idx]
        data_in = torch.tensor(s['data']).unsqueeze(0).float().to(device)
        data_in_img = torch.tensor(s['data_img']).unsqueeze(0).float().to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            if mode == 'ss2d':
                mask_in = torch.tensor(s['mask']).unsqueeze(0).float().to(device)
                sens_in = torch.tensor(s['sens']).unsqueeze(0).float().to(device)
                out = model(data_in_img, data_in, mask_in, sens_in)
            else:
                out = model(data_in_img, data_in)
        outs[idx] = out.squeeze().cpu().numpy()
        if mode == 'ss2d':  # GT 는 한 번만 저장
            gts[idx] = np.asarray(s['label']).squeeze()

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(os.path.join(cache_dir, f'{mode}_outs.npz'),
             **{f'{k}': v for k, v in outs.items()})
    if mode == 'ss2d':
        np.savez(os.path.join(cache_dir, 'gts.npz'),
                 **{f'{k}': v for k, v in gts.items()})
    print(f'[{mode}] 저장 완료: {cache_dir}/{mode}_outs.npz')


def run_merge(cache_dir, output_dir):
    ss2d_npz = np.load(os.path.join(cache_dir, 'ss2d_outs.npz'))
    eter_npz = np.load(os.path.join(cache_dir, 'eter_outs.npz'))
    gt_npz   = np.load(os.path.join(cache_dir, 'gts.npz'))
    indices = sorted([int(k) for k in ss2d_npz.files])

    os.makedirs(output_dir, exist_ok=True)
    psnrs_s, ssims_s, psnrs_e, ssims_e = [], [], [], []
    for idx in tqdm(indices, desc='merge'):
        k = str(idx)
        gt = gt_npz[k]
        ss2d = ss2d_npz[k]
        eter = eter_npz[k]
        ps = calc_psnr(ss2d, gt); ss = calc_ssim(ss2d, gt)
        pe = calc_psnr(eter, gt); se = calc_ssim(eter, gt)
        psnrs_s.append(ps); ssims_s.append(ss)
        psnrs_e.append(pe); ssims_e.append(se)
        make_figure(gt, ss2d, eter, ps, ss, pe, se, idx,
                    os.path.join(output_dir, f'compare_{idx:04d}.png'))

    summary = [
        '===== SS2D v6 vs ETER v5 비교 (선택 샘플) =====',
        f'샘플 수: {len(indices)}',
        f'SS2D-ViT v6 : PSNR {np.mean(psnrs_s):.2f}±{np.std(psnrs_s):.2f} | SSIM {np.mean(ssims_s):.4f}±{np.std(ssims_s):.4f}',
        f'ETER-ViT v5 : PSNR {np.mean(psnrs_e):.2f}±{np.std(psnrs_e):.2f} | SSIM {np.mean(ssims_e):.4f}±{np.std(ssims_e):.4f}',
    ]
    print('\n' + '\n'.join(summary))
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(summary) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ss2d', 'eter', 'merge'], required=True,
                        help='ss2d/eter: 단일 모델 추론 후 .npz 저장 / merge: PNG 생성')
    parser.add_argument('--ss2d-ckpt', type=str,
                        default='logs/SS2D_ViT_R4_brain320_v6/ss2d_vit_best.pt')
    parser.add_argument('--eter-ckpt', type=str,
                        default='logs/ETER_ViT_R4_brain320_v5/eter_vit_best.pt')
    parser.add_argument('--data-path', type=str, default='./fastMRI_data/multicoil_val')
    parser.add_argument('--cache-dir', type=str, default='results/vis_compare_v6_partial/_cache')
    parser.add_argument('--output-dir', type=str, default='results/vis_compare_v6_partial')
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()

    if args.mode == 'merge':
        run_merge(args.cache_dir, args.output_dir)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'===== mode={args.mode} (device={device}) =====')
    ckpt = args.ss2d_ckpt if args.mode == 'ss2d' else args.eter_ckpt
    run_inference(args.mode, ckpt, args.data_path, args.num_samples,
                  args.cache_dir, device)


if __name__ == '__main__':
    main()
