"""
v7 통합 평가: 기존 baseline (SS2D-v5, ETER-v5, U-Net) + v7 모델(들) 동시 평가.

기존 eval_full_compare.py 의 베이스라인 평가 코드를 그대로 유지하고 v7 SS2D/ETER ckpt
를 옵셔널로 추가해 같은 데이터셋에서 동일한 지표 (skimage SSIM, PSNR, NMSE, L1) 를 측정한다.
ckpt 가 지정된 v7 모델만 평가 결과에 포함되며, baseline 컬럼은 항상 출력된다.
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
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

import fastmri
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models import Unet

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))


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
    return float('inf') if mse == 0 else 20 * np.log10(target.max() / np.sqrt(mse))


def calc_nmse(pred, target):
    return np.linalg.norm(pred - target) ** 2 / np.linalg.norm(target) ** 2


def calc_ssim(pred, target):
    return compare_ssim(target, pred, data_range=target.max() - target.min())


def _load_state(path, device):
    state = torch.load(path, map_location=device)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    return state


def load_ss2d_v5(ckpt, device):
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
    m.load_state_dict(_load_state(ckpt, device))
    return m.to(device).eval()


def load_eter_v5(ckpt, device):
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
    m.load_state_dict(_load_state(ckpt, device))
    return m.to(device).eval()


def load_ss2d_v7(ckpt, device):
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
    m.load_state_dict(_load_state(ckpt, device))
    return m.to(device).eval()


def load_eter_v7(ckpt, device):
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
    m.load_state_dict(_load_state(ckpt, device))
    return m.to(device).eval()


def load_unet(state_dict_path, device):
    m = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    m.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    return m.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ss2d-v5-ckpt', default='logs/SS2D_ViT_R4_brain320_v5/ss2d_vit_best.pt')
    p.add_argument('--eter-v5-ckpt', default='logs/ETER_ViT_R4_brain320_v5/eter_vit_best.pt')
    p.add_argument('--unet-ckpt',    default='models/pretrained/brain_leaderboard_state_dict.pt')
    p.add_argument('--ss2d-v7-ckpt', default=None, help='v7 SS2D ckpt (optional)')
    p.add_argument('--eter-v7-ckpt', default=None, help='v7 ETER ckpt (optional)')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--out-tag', default='full_v7')
    p.add_argument('--max-samples', type=int, default=-1)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('====================================================')
    print(' Full Eval v7: baseline (v5/UNet) + 옵셔널 v7 모델')
    print('====================================================')
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

    print('\n베이스라인 모델 로드...')
    ss2d_v5 = load_ss2d_v5(args.ss2d_v5_ckpt, device); print(f"  SS2D v5: {args.ss2d_v5_ckpt}")
    eter_v5 = load_eter_v5(args.eter_v5_ckpt, device); print(f"  ETER v5: {args.eter_v5_ckpt}")
    unet    = load_unet(args.unet_ckpt, device);       print(f"  UNet  : {args.unet_ckpt}")

    ss2d_v7 = None
    if args.ss2d_v7_ckpt and os.path.exists(args.ss2d_v7_ckpt):
        ss2d_v7 = load_ss2d_v7(args.ss2d_v7_ckpt, device)
        print(f"  SS2D v7: {args.ss2d_v7_ckpt}")
    eter_v7 = None
    if args.eter_v7_ckpt and os.path.exists(args.eter_v7_ckpt):
        eter_v7 = load_eter_v7(args.eter_v7_ckpt, device)
        print(f"  ETER v7: {args.eter_v7_ckpt}")

    mask_func = create_mask_for_mask_type('equispaced_fraction', [0.08], [4])
    unet_tr = T.UnetDataTransform(which_challenge='multicoil', mask_func=mask_func, use_seed=True)
    unet_ds = SliceDataset(root=args.data_path, transform=unet_tr, challenge='multicoil')

    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    h5_ds = FastMRI_H5_Dataloader(args.data_path, num_files=None, random_mask=False, augment=False)

    total = min(len(unet_ds), len(h5_ds))
    if args.max_samples > 0:
        total = min(total, args.max_samples)
    print(f"평가 대상: {total} 슬라이스\n")

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path     = os.path.join(args.results_dir, f'eval_{args.out_tag}.csv')
    summary_path = os.path.join(args.results_dir, f'eval_{args.out_tag}_summary.txt')

    rows = {'ss2d_v5': [], 'eter_v5': [], 'unet': []}
    if ss2d_v7 is not None: rows['ss2d_v7'] = []
    if eter_v7 is not None: rows['eter_v7'] = []

    header = ['idx']
    for tag in ['ss2d_v5', 'eter_v5', 'unet']:
        header += [f'{tag}_psnr', f'{tag}_nmse', f'{tag}_ssim', f'{tag}_l1']
    if ss2d_v7 is not None:
        header += ['ss2d_v7_psnr', 'ss2d_v7_nmse', 'ss2d_v7_ssim', 'ss2d_v7_l1']
    if eter_v7 is not None:
        header += ['eter_v7_psnr', 'eter_v7_nmse', 'eter_v7_ssim', 'eter_v7_l1']

    with open(csv_path, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        w.writerow(header)
        bar = tqdm(range(total), desc='Eval', unit='slice')
        with torch.no_grad():
            for idx in bar:
                image, target, mean, std, fname, slice_num, max_value = unet_ds[idx]
                u_out = unet(image.unsqueeze(0).unsqueeze(1).to(device)).squeeze().cpu()
                u_recon = crop_or_pad((u_out * std + mean).numpy(), (320, 320))
                u_gt    = crop_or_pad((target * std + mean).numpy(), (320, 320))

                s = h5_ds[idx]
                d_in     = torch.tensor(s['data']).unsqueeze(0).float().to(device)
                d_in_img = torch.tensor(s['data_img']).unsqueeze(0).float().to(device)
                d_ref    = torch.tensor(s['label']).unsqueeze(0).float().to(device)
                m_in     = torch.tensor(s['mask']).unsqueeze(0).float().to(device)
                sens_in  = torch.tensor(s['sens']).unsqueeze(0).float().to(device)

                with torch.amp.autocast('cuda'):
                    ss5_out = ss2d_v5(d_in_img, d_in, m_in, sens_in)
                    et5_out = eter_v5(d_in_img, d_in)
                    ss7_out = ss2d_v7(d_in_img, d_in, m_in, sens_in) if ss2d_v7 is not None else None
                    et7_out = eter_v7(d_in_img, d_in)               if eter_v7 is not None else None

                ss5_recon = ss5_out.squeeze().float().cpu().numpy()
                et5_recon = et5_out.squeeze().float().cpu().numpy()
                h5_gt     = d_ref.squeeze().float().cpu().numpy()

                def metrics(recon, gt):
                    return (calc_psnr(recon, gt), calc_nmse(recon, gt),
                            calc_ssim(recon, gt), float(np.mean(np.abs(recon - gt))))

                ss5_m = metrics(ss5_recon, h5_gt)
                et5_m = metrics(et5_recon, h5_gt)
                u_m   = metrics(u_recon, u_gt)

                rows['ss2d_v5'].append(ss5_m)
                rows['eter_v5'].append(et5_m)
                rows['unet'].append(u_m)

                row_out = [idx]
                row_out += [f'{ss5_m[0]:.2f}', f'{ss5_m[1]:.6f}', f'{ss5_m[2]:.4f}', f'{ss5_m[3]:.6f}']
                row_out += [f'{et5_m[0]:.2f}', f'{et5_m[1]:.6f}', f'{et5_m[2]:.4f}', f'{et5_m[3]:.6f}']
                row_out += [f'{u_m[0]:.2f}',   f'{u_m[1]:.6f}',   f'{u_m[2]:.4f}',   f'{u_m[3]:.6f}']

                if ss7_out is not None:
                    ss7_recon = ss7_out.squeeze().float().cpu().numpy()
                    ss7_m = metrics(ss7_recon, h5_gt)
                    rows['ss2d_v7'].append(ss7_m)
                    row_out += [f'{ss7_m[0]:.2f}', f'{ss7_m[1]:.6f}', f'{ss7_m[2]:.4f}', f'{ss7_m[3]:.6f}']
                if et7_out is not None:
                    et7_recon = et7_out.squeeze().float().cpu().numpy()
                    et7_m = metrics(et7_recon, h5_gt)
                    rows['eter_v7'].append(et7_m)
                    row_out += [f'{et7_m[0]:.2f}', f'{et7_m[1]:.6f}', f'{et7_m[2]:.4f}', f'{et7_m[3]:.6f}']

                w.writerow(row_out)
                if idx % 100 == 0:
                    pf = {'SS2D5': f'{ss5_m[2]:.3f}', 'ETER5': f'{et5_m[2]:.3f}', 'UNet': f'{u_m[2]:.3f}'}
                    if ss2d_v7 is not None: pf['SS2D7'] = f'{rows["ss2d_v7"][-1][2]:.3f}'
                    if eter_v7 is not None: pf['ETER7'] = f'{rows["eter_v7"][-1][2]:.3f}'
                    bar.set_postfix(pf)

    def stats(vals):
        a = np.asarray(vals)
        return a.mean(0), a.std(0)

    def fmt_line(name, vals):
        mean, std = stats(vals)
        return (f'{name:>15s} | {mean[0]:>6.2f}±{std[0]:<6.2f} | '
                f'{mean[1]:>7.5f}±{std[1]:<6.5f} | {mean[2]:>6.4f}±{std[2]:<6.4f} | '
                f'{mean[3]:>7.5f}±{std[3]:<6.5f}')

    lines = [
        '========== Full Eval v7 (skimage SSIM) ==========',
        f'샘플 수: {total}',
        f'마스크: equispaced R=4, center=0.08',
        '',
        f'{"모델":>15s} | {"PSNR(dB)":>14s} | {"NMSE":>14s} | {"SSIM":>14s} | {"L1":>14s}',
        f'{"-"*15} | {"-"*14} | {"-"*14} | {"-"*14} | {"-"*14}',
        fmt_line('SS2D-ViT v5', rows['ss2d_v5']),
        fmt_line('ETER-ViT v5', rows['eter_v5']),
        fmt_line('UNet (PT)',   rows['unet']),
    ]
    if ss2d_v7 is not None: lines.append(fmt_line('SS2D-ViT v7', rows['ss2d_v7']))
    if eter_v7 is not None: lines.append(fmt_line('ETER-ViT v7', rows['eter_v7']))

    print('\n' + '\n'.join(lines))
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nCSV: {csv_path}')
    print(f'요약: {summary_path}')


if __name__ == '__main__':
    main()
