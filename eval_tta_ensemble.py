"""Tier 1 평가: Test-Time Augmentation (TTA) + v4/v6 앙상블

목적: 추가 학습 없이 SSIM / PSNR 향상 여지 측정.

내용:
  - Base: SS2D v4, SS2D v6, ETER v4, ETER v6, UNet (PT)
  - TTA: SS2D v6, ETER v6 — 4-way flip 평균
         (identity / H-flip / W-flip / HW-flip; image domain flip → FFT 로 k-space 재계산)
  - Ensemble: 출력 평균
         · SS2D v4 + SS2D v6
         · ETER v4 + ETER v6
         · SS2D v6 + ETER v6  (cross-architecture)
         · SS2D v6 TTA + ETER v6 TTA

검증:
  - --max-samples N 으로 부분 실행 (기본 500). 결과 만족시 -1 로 전체 7270 재실행.
  - skimage SSIM 으로 v6 풀평가 (results/eval/eval_full_v6_summary.txt) 와 같은 기준.
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

import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models import Unet

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'configs'))
sys.path.append(os.path.join(current_dir, 'dataloaders'))
sys.path.append(os.path.join(current_dir, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(current_dir, 'models', 'mamba_eternet'))


# ──────────────────────────────────────────────
#  Utility
# ──────────────────────────────────────────────

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


def calc_nmse(pred, target):
    return np.linalg.norm(pred - target) ** 2 / np.linalg.norm(target) ** 2


def calc_ssim(pred, target):
    return compare_ssim(target, pred, data_range=target.max() - target.min())


def fft2c_torch(x):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1),
    )


# ──────────────────────────────────────────────
#  Model loaders (v4 와 v6 둘 다 v5 config 의 model class import 가능
#  — DROPOUT, DC 등 hyperparam 만 다를 뿐 architecture 동일)
# ──────────────────────────────────────────────

def load_ss2d(ckpt, cfg_name, device):
    cfg = __import__(cfg_name)
    from u_choh_model_ETER_ViT import choh_ViT
    from u_choh_model_SS2D_ViT_v4 import choh_Decoder_SS2D_ViT
    enc = choh_ViT(
        image_size=cfg.IMAGE_SIZE, patch_size=cfg.PATCH_SIZE, num_classes=1000,
        dim=cfg.NUM_VIT_ENCODER_HIDDEN, depth=cfg.NUM_VIT_ENCODER_LAYER,
        heads=cfg.NUM_VIT_ENCODER_HEAD, mlp_dim=cfg.NUM_VIT_ENCODER_MLP_SIZE,
        channels=cfg.INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1,
    )
    m = choh_Decoder_SS2D_ViT(
        encoder=enc,
        ss2d_d_inner=cfg.NUM_SS2D_D_INNER, ss2d_d_state=cfg.NUM_SS2D_D_STATE,
        ss2d_out_ch=cfg.NUM_SS2D_OUT_CH,
        decoder_dim=cfg.NUM_VIT_DECODER_DIM, decoder_depth=cfg.NUM_VIT_DECODER_DEPTH,
        decoder_heads=cfg.NUM_VIT_DECODER_HEAD,
        decoder_dim_head=cfg.NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=cfg.NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=cfg.NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=cfg.NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        dropout=cfg.DROPOUT,
        dc_k_scale_ratio=cfg.DC_K_SCALE_RATIO, dc_init_alpha=cfg.DC_INIT_ALPHA,
    )
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


def load_eter(ckpt, cfg_name, device, decoder_class, with_dropout):
    cfg = __import__(cfg_name)
    from u_choh_model_ETER_ViT import choh_ViT
    enc = choh_ViT(
        image_size=cfg.IMAGE_SIZE, patch_size=cfg.PATCH_SIZE, num_classes=1000,
        dim=cfg.NUM_VIT_ENCODER_HIDDEN, depth=cfg.NUM_VIT_ENCODER_LAYER,
        heads=cfg.NUM_VIT_ENCODER_HEAD, mlp_dim=cfg.NUM_VIT_ENCODER_MLP_SIZE,
        channels=cfg.INPUT_CHANNELS, dropout=0.1, emb_dropout=0.1,
    )
    kwargs = dict(
        encoder=enc,
        eter_n_hori_hidden=cfg.NUM_ETER_HORI_HIDDEN,
        eter_n_vert_hidden=cfg.NUM_ETER_VERT_HIDDEN,
        decoder_dim=cfg.NUM_VIT_DECODER_DIM, decoder_depth=cfg.NUM_VIT_DECODER_DEPTH,
        decoder_heads=cfg.NUM_VIT_DECODER_HEAD,
        decoder_dim_head=cfg.NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden=cfg.NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail=cfg.NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear=cfg.NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
    )
    if with_dropout:
        kwargs['dropout'] = getattr(cfg, 'DROPOUT', 0.0)
    m = decoder_class(**kwargs)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


def load_unet(state_dict_path, device):
    m = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    m.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    return m.to(device).eval()


# ──────────────────────────────────────────────
#  TTA — flip inputs in image domain, FFT to get coherent k-space
# ──────────────────────────────────────────────

VAL_AMP_X_IMG = 1e6
VAL_AMP_X_KSP = 1e4


def apply_flip_inputs(data_img, mask, sens, hflip, wflip):
    """data_img/sens 는 image domain (packed real). flip 적용 후 FFT 로 data(k-space) 재계산."""
    # unpack
    img_c = (data_img[:, 0::2] + 1j * data_img[:, 1::2]) / VAL_AMP_X_IMG
    sens_c = sens[:, 0::2] + 1j * sens[:, 1::2]

    if hflip:
        img_c = torch.flip(img_c, dims=[-2])
        sens_c = torch.flip(sens_c, dims=[-2])
    if wflip:
        img_c = torch.flip(img_c, dims=[-1])
        sens_c = torch.flip(sens_c, dims=[-1])
        mask = torch.flip(mask, dims=[-1])

    # FFT 로 k-space 재계산 (Aliased image → masked k-space)
    ksp_c = fft2c_torch(img_c)

    def pack(c, scale):
        B, C, H, W = c.shape
        out = torch.empty(B, 2 * C, H, W, device=c.device, dtype=torch.float32)
        out[:, 0::2] = (c.real * scale).float()
        out[:, 1::2] = (c.imag * scale).float()
        return out

    data_img_f = pack(img_c, VAL_AMP_X_IMG)
    data_f = pack(ksp_c, VAL_AMP_X_KSP)
    sens_f = pack(sens_c, 1.0)
    return data_f, data_img_f, mask, sens_f


def unflip_output(out, hflip, wflip):
    if hflip:
        out = torch.flip(out, dims=[-2])
    if wflip:
        out = torch.flip(out, dims=[-1])
    return out


FLIP_CONFIGS = [(False, False), (True, False), (False, True), (True, True)]


def forward_ss2d(m, data_img, data, mask, sens):
    with torch.amp.autocast('cuda'):
        return m(data_img, data, mask, sens)


def forward_eter(m, data_img, data):
    with torch.amp.autocast('cuda'):
        return m(data_img, data)


def tta_ss2d(m, data_img, data, mask, sens):
    outs = []
    for hflip, wflip in FLIP_CONFIGS:
        if not hflip and not wflip:
            d_f, di_f, mk_f, sn_f = data, data_img, mask, sens
        else:
            d_f, di_f, mk_f, sn_f = apply_flip_inputs(data_img, mask, sens, hflip, wflip)
        o = forward_ss2d(m, di_f, d_f, mk_f, sn_f)
        outs.append(unflip_output(o, hflip, wflip).float())
    return torch.stack(outs).mean(0)


def tta_eter(m, data_img, data, mask, sens):
    outs = []
    for hflip, wflip in FLIP_CONFIGS:
        if not hflip and not wflip:
            d_f, di_f = data, data_img
        else:
            d_f, di_f, _, _ = apply_flip_inputs(data_img, mask, sens, hflip, wflip)
        o = forward_eter(m, di_f, d_f)
        outs.append(unflip_output(o, hflip, wflip).float())
    return torch.stack(outs).mean(0)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ss2d-v4-ckpt', default='logs/SS2D_ViT_R4_brain320_v4/ss2d_vit_best.pt')
    p.add_argument('--ss2d-v6-ckpt', default='logs/SS2D_ViT_R4_brain320_v6/ss2d_vit_best.pt')
    p.add_argument('--eter-v4-ckpt', default='logs/ETER_ViT_R4_brain320_v4/eter_vit_best.pt')
    p.add_argument('--eter-v6-ckpt', default='logs/ETER_ViT_R4_brain320_v6/eter_vit_best.pt')
    p.add_argument('--unet-ckpt', default='models/pretrained/brain_leaderboard_state_dict.pt')
    p.add_argument('--data-path', default='./fastMRI_data/multicoil_val')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--out-tag', default='tta_ensemble')
    p.add_argument('--max-samples', type=int, default=500, help='-1 = all')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('====================================================')
    print(' Tier 1: TTA + v4/v6 ensemble eval (skimage SSIM)')
    print('====================================================')
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

    from u_choh_model_ETER_ViT import choh_Decoder3_ETER_skip_up_tail
    from u_choh_model_ETER_ViT_v5 import choh_Decoder3_ETER_v5

    print('\n모델 로드 중...')
    ss2d_v4 = load_ss2d(args.ss2d_v4_ckpt, 'myConfig_choh_SS2D_model_v4', device)
    ss2d_v6 = load_ss2d(args.ss2d_v6_ckpt, 'myConfig_choh_SS2D_model_v6', device)
    eter_v4 = load_eter(args.eter_v4_ckpt, 'myConfig_choh_ETER_model',
                        device, choh_Decoder3_ETER_skip_up_tail, with_dropout=False)
    eter_v6 = load_eter(args.eter_v6_ckpt, 'myConfig_choh_ETER_model_v6',
                        device, choh_Decoder3_ETER_v5, with_dropout=True)
    unet = load_unet(args.unet_ckpt, device)
    print(f"  SS2D v4: {args.ss2d_v4_ckpt}")
    print(f"  SS2D v6: {args.ss2d_v6_ckpt}")
    print(f"  ETER v4: {args.eter_v4_ckpt}")
    print(f"  ETER v6: {args.eter_v6_ckpt}")
    print(f"  UNet:    {args.unet_ckpt}")

    # ── Datasets ──
    mask_func = create_mask_for_mask_type('equispaced_fraction', [0.08], [4])
    unet_tr = T.UnetDataTransform(which_challenge='multicoil', mask_func=mask_func, use_seed=True)
    unet_ds = SliceDataset(root=args.data_path, transform=unet_tr, challenge='multicoil')

    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    h5_ds = FastMRI_H5_Dataloader(args.data_path, num_files=None, random_mask=False, augment=False)

    n_unet, n_h5 = len(unet_ds), len(h5_ds)
    total = min(n_unet, n_h5)

    if args.max_samples > 0 and args.max_samples < total:
        rng = np.random.default_rng(args.seed)
        all_idx = rng.permutation(total)[:args.max_samples]
        all_idx.sort()
        eval_indices = all_idx.tolist()
    else:
        eval_indices = list(range(total))

    print(f"평가 대상: {len(eval_indices)} 슬라이스 (전체 {total})\n")

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, f'eval_{args.out_tag}.csv')
    summary_path = os.path.join(args.results_dir, f'eval_{args.out_tag}_summary.txt')

    # 평가 설정 목록 — 각 항목은 PSNR/NMSE/SSIM/L1 누적
    METHODS = [
        'ss2d_v4', 'ss2d_v6', 'ss2d_v6_tta',
        'eter_v4', 'eter_v6', 'eter_v6_tta',
        'unet',
        'ens_ss2d_v4_v6',
        'ens_eter_v4_v6',
        'ens_ss2d_eter_v6',
        'ens_ss2d_eter_v6_tta',
    ]
    rows = {k: [] for k in METHODS}

    def metrics(pred, target):
        return (calc_psnr(pred, target),
                calc_nmse(pred, target),
                calc_ssim(pred, target),
                float(np.mean(np.abs(pred - target))))

    csv_buffer = []
    with open(csv_path, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        header = ['idx']
        for k in METHODS:
            header += [f'{k}_psnr', f'{k}_nmse', f'{k}_ssim', f'{k}_l1']
        w.writerow(header)

        bar = tqdm(eval_indices, desc='Eval', unit='slice')
        with torch.no_grad():
            for idx in bar:
                # ── UNet ──
                image, target, mean, std, fname, slice_num, max_value = unet_ds[idx]
                u_out = unet(image.unsqueeze(0).unsqueeze(1).to(device)).squeeze().cpu()
                u_recon = (u_out * std + mean).numpy()
                u_gt = (target * std + mean).numpy()
                u_recon = crop_or_pad(u_recon, (320, 320))
                u_gt = crop_or_pad(u_gt, (320, 320))

                # ── h5 입력 (single sample) ──
                s = h5_ds[idx]
                d_in     = torch.from_numpy(s['data']).unsqueeze(0).to(device, non_blocking=True)
                d_in_img = torch.from_numpy(s['data_img']).unsqueeze(0).to(device, non_blocking=True)
                d_ref    = torch.from_numpy(s['label']).unsqueeze(0).to(device, non_blocking=True)
                m_in     = torch.from_numpy(s['mask']).unsqueeze(0).to(device, non_blocking=True)
                sens_in  = torch.from_numpy(s['sens']).unsqueeze(0).to(device, non_blocking=True)

                # ── Base forwards ──
                o_ss2d_v4 = forward_ss2d(ss2d_v4, d_in_img, d_in, m_in, sens_in)
                o_ss2d_v6 = forward_ss2d(ss2d_v6, d_in_img, d_in, m_in, sens_in)
                o_eter_v4 = forward_eter(eter_v4, d_in_img, d_in)
                o_eter_v6 = forward_eter(eter_v6, d_in_img, d_in)

                # ── TTA (v6 만) ──
                o_ss2d_v6_tta = tta_ss2d(ss2d_v6, d_in_img, d_in, m_in, sens_in)
                o_eter_v6_tta = tta_eter(eter_v6, d_in_img, d_in, m_in, sens_in)

                # ── Ensembles ──
                o_ens_ss2d_v4v6 = (o_ss2d_v4.float() + o_ss2d_v6.float()) * 0.5
                o_ens_eter_v4v6 = (o_eter_v4.float() + o_eter_v6.float()) * 0.5
                o_ens_se_v6     = (o_ss2d_v6.float() + o_eter_v6.float()) * 0.5
                o_ens_se_v6tta  = (o_ss2d_v6_tta + o_eter_v6_tta) * 0.5

                # ── 변환: 모든 outs 를 numpy 로 ──
                def to_np(o):
                    return o.squeeze().float().cpu().numpy()
                preds = {
                    'ss2d_v4':              to_np(o_ss2d_v4),
                    'ss2d_v6':              to_np(o_ss2d_v6),
                    'ss2d_v6_tta':          to_np(o_ss2d_v6_tta),
                    'eter_v4':              to_np(o_eter_v4),
                    'eter_v6':              to_np(o_eter_v6),
                    'eter_v6_tta':          to_np(o_eter_v6_tta),
                    'unet':                 u_recon,
                    'ens_ss2d_v4_v6':       to_np(o_ens_ss2d_v4v6),
                    'ens_eter_v4_v6':       to_np(o_ens_eter_v4v6),
                    'ens_ss2d_eter_v6':     to_np(o_ens_se_v6),
                    'ens_ss2d_eter_v6_tta': to_np(o_ens_se_v6tta),
                }
                gts = {k: u_gt if k == 'unet' else to_np(d_ref) for k in preds}

                row = [idx]
                postfix = {}
                for k in METHODS:
                    psnr, nmse, ssim, l1 = metrics(preds[k], gts[k])
                    rows[k].append((psnr, nmse, ssim, l1))
                    row += [f'{psnr:.2f}', f'{nmse:.6f}', f'{ssim:.4f}', f'{l1:.6f}']
                    if k in ('ss2d_v6', 'ss2d_v6_tta', 'ens_ss2d_eter_v6_tta'):
                        postfix[k] = f'{ssim:.3f}'

                csv_buffer.append(row)
                if len(csv_buffer) >= 50:
                    w.writerows(csv_buffer)
                    csv_buffer.clear()

                bar.set_postfix(**postfix)

        if csv_buffer:
            w.writerows(csv_buffer)
            csv_buffer.clear()

    # ── Summary ──
    def stats(vals):
        a = np.asarray(vals)
        return a.mean(0), a.std(0)

    label_map = {
        'ss2d_v4':              'SS2D-ViT v4',
        'ss2d_v6':              'SS2D-ViT v6',
        'ss2d_v6_tta':          'SS2D-ViT v6 (TTA)',
        'eter_v4':              'ETER-ViT v4',
        'eter_v6':              'ETER-ViT v6',
        'eter_v6_tta':          'ETER-ViT v6 (TTA)',
        'unet':                 'UNet (PT)',
        'ens_ss2d_v4_v6':       'Ens SS2D v4+v6',
        'ens_eter_v4_v6':       'Ens ETER v4+v6',
        'ens_ss2d_eter_v6':     'Ens SS2D+ETER v6',
        'ens_ss2d_eter_v6_tta': 'Ens SS2D+ETER v6 TTA',
    }

    lines = [
        '========== Tier 1: TTA + Ensemble Eval (skimage SSIM) ==========',
        f'샘플 수: {len(eval_indices)} / 전체 {total}',
        '마스크: equispaced R=4, center=0.08',
        f'시드: {args.seed}',
        '',
        f'{"모델":>25s} | {"PSNR(dB)":>14s} | {"NMSE":>14s} | {"SSIM":>14s} | {"L1":>14s}',
        f'{"-"*25} | {"-"*14} | {"-"*14} | {"-"*14} | {"-"*14}',
    ]
    for k in METHODS:
        mn, sd = stats(rows[k])
        lines.append(
            f'{label_map[k]:>25s} | {mn[0]:>6.2f}±{sd[0]:<6.2f} | '
            f'{mn[1]:>7.5f}±{sd[1]:<6.5f} | {mn[2]:>6.4f}±{sd[2]:<6.4f} | '
            f'{mn[3]:>7.5f}±{sd[3]:<6.5f}'
        )

    print('\n' + '\n'.join(lines))
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nCSV: {csv_path}')
    print(f'요약: {summary_path}')


if __name__ == '__main__':
    main()
