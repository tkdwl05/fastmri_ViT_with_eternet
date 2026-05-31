"""
SS2D-ViT v5 / ETER-ViT v5 / FastMRI Pre-trained U-Net 통합 평가

전체 val 셋 (7270 슬라이스) 에서 fastMRI 표준 SSIM (skimage data_range 기반) /
PSNR / NMSE / L1 을 모두 동일한 방식으로 측정한다. 학습 로그의 custom SSIM 은
val_range=None → L=1 고정으로 dynamic range 작은 fastMRI raw 값에서 부정확했음.

이 스크립트는 그 이슈를 우회하여 진짜 baseline 을 확정한다.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import argparse
import csv
import datetime
import pytz
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

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


def crop_or_pad(data: np.ndarray, shape: tuple) -> np.ndarray:
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


def calc_nmse(pred, target):
    return np.linalg.norm(pred - target) ** 2 / np.linalg.norm(target) ** 2


def calc_ssim(pred, target):
    return compare_ssim(target, pred, data_range=target.max() - target.min())


def load_ss2d(ckpt, device):
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


def load_unet(state_dict_path, device):
    m = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    m.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    return m.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ss2d-ckpt', default='logs/SS2D_ViT_R4_brain320_v5/ss2d_vit_best.pt')
    p.add_argument('--eter-ckpt', default='logs/ETER_ViT_R4_brain320_v5/eter_vit_best.pt')
    p.add_argument('--unet-ckpt', default='models/pretrained/brain_leaderboard_state_dict.pt')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--out-tag', default='full_v5')
    p.add_argument('--label', default='v5', help='모델 라벨 (예: v5, v6)')
    p.add_argument('--max-samples', type=int, default=-1, help='-1 = all')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('====================================================')
    print(f' Full Eval: SS2D-ViT {args.label} / ETER-ViT {args.label} / UNet (skimage SSIM)')
    print('====================================================')
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

    print('\n모델 로드 중...')
    ss2d = load_ss2d(args.ss2d_ckpt, device)
    eter = load_eter(args.eter_ckpt, device)
    unet = load_unet(args.unet_ckpt, device)
    print(f"  SS2D: {args.ss2d_ckpt}")
    print(f"  ETER: {args.eter_ckpt}")
    print(f"  UNet: {args.unet_ckpt}")

    mask_func = create_mask_for_mask_type('equispaced_fraction', [0.08], [4])
    unet_tr = T.UnetDataTransform(which_challenge='multicoil', mask_func=mask_func, use_seed=True)
    unet_ds = SliceDataset(root=args.data_path, transform=unet_tr, challenge='multicoil')

    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    h5_ds = FastMRI_H5_Dataloader(args.data_path, num_files=None, random_mask=False, augment=False)

    n_unet, n_h5 = len(unet_ds), len(h5_ds)
    if n_unet != n_h5:
        print(f"  주의: unet_ds={n_unet}, h5_ds={n_h5} 불일치 — min 으로 잘라서 사용")
    total = min(n_unet, n_h5)
    if args.max_samples > 0:
        total = min(total, args.max_samples)
    print(f"평가 대상: {total} 슬라이스\n")

    # (A) h5_ds prefetching: I/O 와 GPU forward 를 겹쳐서 처리.
    #     batch_size=1 유지하므로 forward 결과는 기존과 동일.
    h5_loader = DataLoader(
        h5_ds, batch_size=1, shuffle=False,
        num_workers=2, persistent_workers=True, pin_memory=False,
    )
    h5_iter = iter(h5_loader)

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, f'eval_{args.out_tag}.csv')
    summary_path = os.path.join(args.results_dir, f'eval_{args.out_tag}_summary.txt')

    rows = {'ss2d': [], 'eter': [], 'unet': []}
    csv_buffer = []          # (B) CSV row buffer
    FLUSH_EVERY = 100

    with open(csv_path, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        w.writerow(['idx',
                    'ss2d_psnr', 'ss2d_nmse', 'ss2d_ssim', 'ss2d_l1',
                    'eter_psnr', 'eter_nmse', 'eter_ssim', 'eter_l1',
                    'unet_psnr', 'unet_nmse', 'unet_ssim', 'unet_l1'])

        bar = tqdm(range(total), desc='Eval', unit='slice')
        with torch.no_grad():
            for idx in bar:
                # ── UNet ──
                image, target, mean, std, fname, slice_num, max_value = unet_ds[idx]
                u_out = unet(image.unsqueeze(0).unsqueeze(1).to(device)).squeeze().cpu()
                u_recon = (u_out * std + mean).numpy()
                u_gt = (target * std + mean).numpy()
                u_recon = crop_or_pad(u_recon, (320, 320))
                u_gt = crop_or_pad(u_gt, (320, 320))

                # ── SS2D / ETER 입력 (prefetched) ──
                s = next(h5_iter)
                # collate_fn 으로 이미 batch dim 1 추가됨 → unsqueeze 불필요
                d_in     = s['data'].float().to(device, non_blocking=True)
                d_in_img = s['data_img'].float().to(device, non_blocking=True)
                d_ref    = s['label'].float().to(device, non_blocking=True)
                m_in     = s['mask'].float().to(device, non_blocking=True)
                sens_in  = s['sens'].float().to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    ss_out = ss2d(d_in_img, d_in, m_in, sens_in)
                    et_out = eter(d_in_img, d_in)

                # (C) cpu().numpy() 1회 호출로 정리
                ss_recon = ss_out.squeeze().float().cpu().numpy()
                et_recon = et_out.squeeze().float().cpu().numpy()
                h5_gt    = d_ref.squeeze().float().cpu().numpy()

                # ── 지표 ──
                u_psnr = calc_psnr(u_recon, u_gt)
                u_nmse = calc_nmse(u_recon, u_gt)
                u_ssim = calc_ssim(u_recon, u_gt)
                u_l1   = float(np.mean(np.abs(u_recon - u_gt)))

                ss_psnr = calc_psnr(ss_recon, h5_gt)
                ss_nmse = calc_nmse(ss_recon, h5_gt)
                ss_ssim = calc_ssim(ss_recon, h5_gt)
                ss_l1   = float(np.mean(np.abs(ss_recon - h5_gt)))

                et_psnr = calc_psnr(et_recon, h5_gt)
                et_nmse = calc_nmse(et_recon, h5_gt)
                et_ssim = calc_ssim(et_recon, h5_gt)
                et_l1   = float(np.mean(np.abs(et_recon - h5_gt)))

                rows['ss2d'].append((ss_psnr, ss_nmse, ss_ssim, ss_l1))
                rows['eter'].append((et_psnr, et_nmse, et_ssim, et_l1))
                rows['unet'].append((u_psnr, u_nmse, u_ssim, u_l1))

                csv_buffer.append([idx,
                            f'{ss_psnr:.2f}', f'{ss_nmse:.6f}', f'{ss_ssim:.4f}', f'{ss_l1:.6f}',
                            f'{et_psnr:.2f}', f'{et_nmse:.6f}', f'{et_ssim:.4f}', f'{et_l1:.6f}',
                            f'{u_psnr:.2f}', f'{u_nmse:.6f}', f'{u_ssim:.4f}', f'{u_l1:.6f}'])

                if len(csv_buffer) >= FLUSH_EVERY:
                    w.writerows(csv_buffer)
                    csv_buffer.clear()

                if idx % 100 == 0:
                    bar.set_postfix(
                        SS2D=f'{ss_ssim:.3f}', ETER=f'{et_ssim:.3f}', UNet=f'{u_ssim:.3f}'
                    )

        if csv_buffer:
            w.writerows(csv_buffer)
            csv_buffer.clear()

    def stats(vals):
        a = np.asarray(vals)
        return a.mean(0), a.std(0)

    ss_mean, ss_std = stats(rows['ss2d'])
    et_mean, et_std = stats(rows['eter'])
    un_mean, un_std = stats(rows['unet'])

    lines = [
        '========== Full Eval (skimage SSIM) ==========',
        f'샘플 수: {total}',
        f'마스크: equispaced R=4, center=0.08',
        '',
        f'{"모델":>15s} | {"PSNR(dB)":>14s} | {"NMSE":>14s} | {"SSIM":>14s} | {"L1":>14s}',
        f'{"-"*15} | {"-"*14} | {"-"*14} | {"-"*14} | {"-"*14}',
        f'{f"SS2D-ViT {args.label}":>15s} | {ss_mean[0]:>6.2f}±{ss_std[0]:<6.2f} | {ss_mean[1]:>7.5f}±{ss_std[1]:<6.5f} | {ss_mean[2]:>6.4f}±{ss_std[2]:<6.4f} | {ss_mean[3]:>7.5f}±{ss_std[3]:<6.5f}',
        f'{f"ETER-ViT {args.label}":>15s} | {et_mean[0]:>6.2f}±{et_std[0]:<6.2f} | {et_mean[1]:>7.5f}±{et_std[1]:<6.5f} | {et_mean[2]:>6.4f}±{et_std[2]:<6.4f} | {et_mean[3]:>7.5f}±{et_std[3]:<6.5f}',
        f'{"UNet (PT)":>15s} | {un_mean[0]:>6.2f}±{un_std[0]:<6.2f} | {un_mean[1]:>7.5f}±{un_std[1]:<6.5f} | {un_mean[2]:>6.4f}±{un_std[2]:<6.4f} | {un_mean[3]:>7.5f}±{un_std[3]:<6.5f}',
    ]
    print('\n' + '\n'.join(lines))
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nCSV: {csv_path}')
    print(f'요약: {summary_path}')


if __name__ == '__main__':
    main()
