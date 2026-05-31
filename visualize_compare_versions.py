"""
버전 비교 시각화 (v4 vs v6 vs v6_1 vs v6_2 vs v6_3) — 중간 슬라이스 위주.

동일한 슬라이스 인덱스로 SS2D 와 ETER 의 다섯 버전을 한 PNG 에 묶어서
v4 → v6 → v6_1 → v6_2 → v6_3 진화를 시각적으로 비교한다. U-Net 은 베이스라인으로 포함.

Layout (4x6 grid per slice):
    Row 0:  GT       | SS2D v4 | SS2D v6 | SS2D v6_1 | SS2D v6_2 | SS2D v6_3
    Row 1:  (blank)  | SS2D v4 err | ...                                     | SS2D v6_3 err
    Row 2:  UNet     | ETER v4 | ETER v6 | ETER v6_1 | ETER v6_2 | ETER v6_3
    Row 3:  UNet err | ETER v4 err | ...                                     | ETER v6_3 err
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
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0:
        return float('inf')
    return 20 * np.log10(target.max() / np.sqrt(mse))


def calc_ssim(pred, target):
    dr = float(target.max() - target.min())
    return compare_ssim(target, pred, data_range=dr)


def load_ss2d(ckpt, device):
    """SS2D v4/v6/v6_1 모두 같은 아키텍처 — v5 config + v4 model class 로 로드."""
    from myConfig_choh_SS2D_model_v5 import (
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
    enc = choh_ViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
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


def load_eter_v5plus(ckpt, device):
    """ETER v5/v6/v6_1 — Transformer dropout 추가된 v5 model class."""
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
    enc = choh_ViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
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


def load_eter_v4(ckpt, device):
    """ETER v4 — dropout 없는 base 클래스."""
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
    enc = choh_ViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1000,
                   dim=NUM_VIT_ENCODER_HIDDEN, depth=NUM_VIT_ENCODER_LAYER,
                   heads=NUM_VIT_ENCODER_HEAD, mlp_dim=NUM_VIT_ENCODER_MLP_SIZE,
                   channels=INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1)
    m = choh_Decoder3_ETER_skip_up_tail(
        encoder=enc,
        eter_n_hori_hidden=NUM_ETER_HORI_HIDDEN, eter_n_vert_hidden=NUM_ETER_VERT_HIDDEN,
        decoder_dim=NUM_VIT_DECODER_DIM, decoder_depth=NUM_VIT_DECODER_DEPTH,
        decoder_heads=NUM_VIT_DECODER_HEAD, decoder_dim_head=NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
    )
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


def load_unet(state_dict_path, device):
    m = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    m.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    return m.to(device).eval()


def run_ss2d(model, sample, device):
    d_in     = sample['data'].unsqueeze(0).float().to(device)
    d_in_img = sample['data_img'].unsqueeze(0).float().to(device)
    m_in     = sample['mask'].unsqueeze(0).float().to(device)
    s_in     = sample['sens'].unsqueeze(0).float().to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(d_in_img, d_in, m_in, s_in)
    return out.squeeze().cpu().numpy()


def run_eter(model, sample, device):
    d_in     = sample['data'].unsqueeze(0).float().to(device)
    d_in_img = sample['data_img'].unsqueeze(0).float().to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(d_in_img, d_in)
    return out.squeeze().cpu().numpy()


def to_tensor_sample(d):
    """numpy dict → torch tensor dict (DataLoader 없이 단건 사용)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)
        else:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description='버전 비교 시각화 (v4 vs v6 vs v6_1)')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--unet-ckpt', default='models/pretrained/brain_leaderboard_state_dict.pt')

    p.add_argument('--ss2d-v4',   default='logs/SS2D_ViT_R4_brain320_v4/ss2d_vit_best.pt')
    p.add_argument('--ss2d-v6',   default='logs/SS2D_ViT_R4_brain320_v6/ss2d_vit_best.pt')
    p.add_argument('--ss2d-v6_1', default='logs/SS2D_ViT_R4_brain320_v6_1/ss2d_vit_best.pt')
    p.add_argument('--ss2d-v6_2', default='logs/SS2D_ViT_R4_brain320_v6_2/ss2d_vit_best.pt')
    p.add_argument('--ss2d-v6_3', default='logs/SS2D_ViT_R4_brain320_v6_3/ss2d_vit_best.pt')

    p.add_argument('--eter-v4',   default='logs/ETER_ViT_R4_brain320_v4/eter_vit_best.pt')
    p.add_argument('--eter-v6',   default='logs/ETER_ViT_R4_brain320_v6/eter_vit_best.pt')
    p.add_argument('--eter-v6_1', default='logs/ETER_ViT_R4_brain320_v6_1/eter_vit_best.pt')
    p.add_argument('--eter-v6_2', default='logs/ETER_ViT_R4_brain320_v6_2/eter_vit_best.pt')
    p.add_argument('--eter-v6_3', default='logs/ETER_ViT_R4_brain320_v6_3/eter_vit_best.pt')

    p.add_argument('--out-dir', default='results/vis/aligned/vis_compare_versions_masked')
    p.add_argument('--num-samples', type=int, default=12,
                   help='뽑을 슬라이스 수 (기본 12 → linspace(0, total-1, 12) ≈ 0000, 0660, 1321, 1982, ...)')
    p.add_argument('--middle-frac', type=float, default=1.0,
                   help='슬라이스 sampling 범위 비율. 1.0 = 전체. 0.5 = 가운데 50% 범위 내 균등')
    p.add_argument('--err-vmax-frac', type=float, default=0.05,
                   help='에러맵 colormap vmax (정규화 단위, gt.max() 대비 비율). 기본 0.05 = 5%%')
    p.add_argument('--err-mask-thresh', type=float, default=0.05,
                   help='Brain mask threshold (gt.max() 대비 비율). 기본 0.05 → 5%% 미만 픽셀 배제 (background)')
    p.add_argument('--match-scale', action='store_true',
                   help='시각화 단계에서 per-slice least-squares scale matching (α·recon vs gt) — 모델의 magnitude bias 제거. 기본 off')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('====================================================')
    print(' Version Compare Visualization: SS2D / ETER (v4 vs v6 vs v6_1 vs v6_2 vs v6_3) + UNet')
    print('====================================================')

    mask_func = create_mask_for_mask_type('equispaced_fraction', [0.08], [4])
    unet_tr   = T.UnetDataTransform(which_challenge='multicoil', mask_func=mask_func, use_seed=True)
    unet_ds   = SliceDataset(root=args.data_path, transform=unet_tr, challenge='multicoil')

    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    h5_ds = FastMRI_H5_Dataloader(args.data_path, num_files=None, random_mask=False, augment=False)

    total = min(len(unet_ds), len(h5_ds))
    margin = int(total * (1 - args.middle_frac) / 2)
    lo, hi = margin, total - margin - 1
    indices = np.linspace(lo, hi, args.num_samples, dtype=int).tolist()
    print(f'\n총 슬라이스: {total}, 중간 영역 [{lo}, {hi}] 에서 {len(indices)}개 선택')
    print(f'선택 인덱스: {indices}\n')

    os.makedirs(args.out_dir, exist_ok=True)

    model_order = ['SS2D v4', 'SS2D v6', 'SS2D v6_1', 'SS2D v6_2', 'SS2D v6_3',
                   'ETER v4', 'ETER v6', 'ETER v6_1', 'ETER v6_2', 'ETER v6_3']

    # GPU 8GB 제약 — 모델 outer loop 로 한 번에 GPU 위 모델 1개만 유지.
    # (1) 슬라이스별 입력/GT 캐시 (CPU numpy) (2) UNet (3) 각 ViT 모델 순회 → recon 누적.
    print('슬라이스 입력 데이터 캐시 중...')
    h5_cache = {}      # idx -> sample dict
    gt_cache = {}      # idx -> (320,320) numpy
    unet_inputs = {}   # idx -> (image, target, mean, std)
    for idx in tqdm(indices, desc='데이터 캐시', unit='img'):
        h5_cache[idx] = to_tensor_sample(h5_ds[idx])
        gt_cache[idx] = h5_cache[idx]['label'].squeeze().numpy()
        image, target, mean, std, _fname, _sn, _mv = unet_ds[idx]
        unet_inputs[idx] = (image, target, mean, std)

    # UNet forward → CPU 캐시, 모델 free
    print(f'\nUNet 로드 및 forward: {args.unet_ckpt}')
    unet = load_unet(args.unet_ckpt, device)
    unet_recons = {}   # idx -> (recon, gt) numpy on (320,320) at fastMRI scale
    unet_metrics = {}  # idx -> (psnr, ssim)
    for idx in tqdm(indices, desc='UNet forward', unit='img'):
        image, target, mean, std = unet_inputs[idx]
        with torch.no_grad():
            unet_out = unet(image.unsqueeze(0).unsqueeze(1).to(device)).squeeze().cpu()
        unet_recon = crop_or_pad((unet_out * std + mean).numpy(), (320, 320))
        unet_gt    = crop_or_pad((target   * std + mean).numpy(), (320, 320))
        unet_recons[idx] = (unet_recon, unet_gt)
        unet_metrics[idx] = (calc_psnr(unet_recon, unet_gt), calc_ssim(unet_recon, unet_gt))
    del unet
    torch.cuda.empty_cache()

    # ViT 모델별 순회: 로드 → 슬라이스별 forward → 결과 캐시 → 모델 free
    model_specs = [
        ('SS2D v4',   args.ss2d_v4,   load_ss2d,         run_ss2d),
        ('SS2D v6',   args.ss2d_v6,   load_ss2d,         run_ss2d),
        ('SS2D v6_1', args.ss2d_v6_1, load_ss2d,         run_ss2d),
        ('SS2D v6_2', args.ss2d_v6_2, load_ss2d,         run_ss2d),
        ('SS2D v6_3', args.ss2d_v6_3, load_ss2d,         run_ss2d),
        ('ETER v4',   args.eter_v4,   load_eter_v4,      run_eter),
        ('ETER v6',   args.eter_v6,   load_eter_v5plus,  run_eter),
        ('ETER v6_1', args.eter_v6_1, load_eter_v5plus,  run_eter),
        ('ETER v6_2', args.eter_v6_2, load_eter_v5plus,  run_eter),
        ('ETER v6_3', args.eter_v6_3, load_eter_v5plus,  run_eter),
    ]
    recon_cache = {idx: {} for idx in indices}    # idx -> {model_name: recon}
    metric_cache = {idx: {} for idx in indices}   # idx -> {model_name: (psnr, ssim)}
    for name, ckpt, loader, runner in model_specs:
        print(f'\n[{name}] 로드: {ckpt}')
        model = loader(ckpt, device)
        for idx in tqdm(indices, desc=f'{name} forward', unit='img'):
            out = runner(model, h5_cache[idx], device)
            recon_cache[idx][name] = out
            metric_cache[idx][name] = (calc_psnr(out, gt_cache[idx]), calc_ssim(out, gt_cache[idx]))
        del model
        torch.cuda.empty_cache()

    # 플로팅 (GPU 사용 없음)
    # 에러맵 정책 (docs/error_map_v2_masked.md 참조):
    #  1) per-slice gt.max() 로 [0, 1] 정규화 (UNet 은 자기 GT 기준)
    #  2) brain mask = (gt_n > err_mask_thresh) — background 제거, 흰색
    #  3) --match-scale on 시: α = ⟨recon, gt⟩ / ⟨recon, recon⟩ (mask 안에서), recon ← α·recon
    print('\nPNG 생성 중...')
    hot_bad = plt.get_cmap('hot').copy()
    hot_bad.set_bad(color='white')

    def maybe_match_scale(recon_n, gt_n, mask):
        if not args.match_scale:
            return recon_n
        r = recon_n[mask]
        g = gt_n[mask]
        denom = float((r * r).sum())
        if denom < 1e-12:
            return recon_n
        return float((r * g).sum()) / denom * recon_n

    def masked_err(recon_n_adj, gt_n, mask):
        err = np.abs(recon_n_adj - gt_n)
        return np.where(mask, err, np.nan)

    rows = []
    for idx in tqdm(indices, desc='플롯', unit='img'):
        gt = gt_cache[idx]
        recons = recon_cache[idx]
        metrics = metric_cache[idx]
        unet_recon, unet_gt = unet_recons[idx]
        unet_psnr, unet_ssim = unet_metrics[idx]

        # 정규화 ([0, 1])
        gt_max = max(float(gt.max()), 1e-8)
        unet_gt_max = max(float(unet_gt.max()), 1e-8)
        gt_n = gt / gt_max
        unet_gt_n = unet_gt / unet_gt_max
        unet_recon_n = unet_recon / unet_gt_max
        recons_n = {k: v / gt_max for k, v in recons.items()}

        # Brain mask (per domain)
        brain = gt_n > args.err_mask_thresh
        unet_brain = unet_gt_n > args.err_mask_thresh

        # Optional scale match → 에러맵용 보정된 recon
        recons_n_adj  = {k: maybe_match_scale(v, gt_n, brain) for k, v in recons_n.items()}
        unet_recon_n_adj = maybe_match_scale(unet_recon_n, unet_gt_n, unet_brain)

        errs = {k: masked_err(v, gt_n, brain) for k, v in recons_n_adj.items()}
        unet_err = masked_err(unet_recon_n_adj, unet_gt_n, unet_brain)
        err_max = args.err_vmax_frac

        ss2d_cols = ['SS2D v4', 'SS2D v6', 'SS2D v6_1', 'SS2D v6_2', 'SS2D v6_3']
        eter_cols = ['ETER v4', 'ETER v6', 'ETER v6_1', 'ETER v6_2', 'ETER v6_3']

        fig, axes = plt.subplots(4, 6, figsize=(33, 22))

        # Row 0: GT + SS2D 5종 image (정규화된 [0,1] 공간)
        axes[0, 0].imshow(gt_n, cmap='gray', vmin=0.0, vmax=1.0)
        axes[0, 0].set_title('GT (Ground Truth)', fontsize=13, fontweight='bold')
        axes[0, 0].axis('off')
        for col, name in enumerate(ss2d_cols, start=1):
            axes[0, col].imshow(recons_n_adj[name], cmap='gray', vmin=0.0, vmax=1.0)
            ps, ss = metrics[name]
            axes[0, col].set_title(f'{name}\nPSNR: {ps:.2f}dB | SSIM: {ss:.4f}',
                                   fontsize=11, color='tab:blue')
            axes[0, col].axis('off')

        # Row 1: SS2D 5종 masked error map (col 0 비움)
        axes[1, 0].set_visible(False)
        for col, name in enumerate(ss2d_cols, start=1):
            im = axes[1, col].imshow(errs[name], cmap=hot_bad, vmin=0, vmax=err_max)
            axes[1, col].set_title(f'{name} masked |err|', fontsize=11, color='tab:blue')
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], fraction=0.046)

        # Row 2: UNet + ETER 5종 image
        axes[2, 0].imshow(unet_recon_n_adj, cmap='gray', vmin=0.0, vmax=1.0)
        axes[2, 0].set_title(f'U-Net (Pre-trained)\nPSNR: {unet_psnr:.2f}dB | SSIM: {unet_ssim:.4f}',
                             fontsize=11, color='tab:orange')
        axes[2, 0].axis('off')
        for col, name in enumerate(eter_cols, start=1):
            axes[2, col].imshow(recons_n_adj[name], cmap='gray', vmin=0.0, vmax=1.0)
            ps, ss = metrics[name]
            axes[2, col].set_title(f'{name}\nPSNR: {ps:.2f}dB | SSIM: {ss:.4f}',
                                   fontsize=11, color='tab:green')
            axes[2, col].axis('off')

        # Row 3: UNet error + ETER 5종 masked error map
        im_u = axes[3, 0].imshow(unet_err, cmap=hot_bad, vmin=0, vmax=err_max)
        axes[3, 0].set_title('U-Net masked |err|', fontsize=11, color='tab:orange')
        axes[3, 0].axis('off')
        plt.colorbar(im_u, ax=axes[3, 0], fraction=0.046)
        for col, name in enumerate(eter_cols, start=1):
            im = axes[3, col].imshow(errs[name], cmap=hot_bad, vmin=0, vmax=err_max)
            axes[3, col].set_title(f'{name} masked |err|', fontsize=11, color='tab:green')
            axes[3, col].axis('off')
            plt.colorbar(im, ax=axes[3, col], fraction=0.046)

        scale_note = ' [scale-matched]' if args.match_scale else ''
        fig.suptitle(
            f'Sample #{idx}  (normalized [0,1], brain-masked err_vmax={err_max:.3f},'
            f' mask>{args.err_mask_thresh:.3f}{scale_note}; metric=raw SSIM/PSNR)',
            fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        save_path = os.path.join(args.out_dir, f'compare_{idx:04d}.png')
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
        plt.close(fig)

        row = [idx] + [v for name in model_order for v in metrics[name]] + [unet_psnr, unet_ssim]
        rows.append(row)

    # 요약
    arr = np.array(rows)  # cols: idx, [PSNR,SSIM] × 10 models, u_p, u_s
    summary = [
        '========== 버전 비교 요약 (중간 슬라이스) ==========',
        f'샘플 수: {len(rows)}, 인덱스: {indices}',
        '',
        f'{"모델":>12s} | {"PSNR(dB)":>14s} | {"SSIM":>14s}',
        f'{"-"*12} | {"-"*14} | {"-"*14}',
    ]
    cols = [(name, 1 + 2 * i, 2 + 2 * i) for i, name in enumerate(model_order)]
    cols.append(('UNet', 1 + 2 * len(model_order), 2 + 2 * len(model_order)))
    for name, pi, si in cols:
        pm, ps = arr[:, pi].mean(), arr[:, pi].std()
        sm, ss = arr[:, si].mean(), arr[:, si].std()
        summary.append(f'{name:>12s} | {pm:>6.2f}±{ps:<6.2f} | {sm:>6.4f}±{ss:<6.4f}')
    msg = '\n'.join(summary)
    print('\n' + msg)
    with open(os.path.join(args.out_dir, 'comparison_summary.txt'), 'w') as f:
        f.write(msg + '\n')
    print(f'\n저장 완료: {args.out_dir}/')


if __name__ == '__main__':
    main()
