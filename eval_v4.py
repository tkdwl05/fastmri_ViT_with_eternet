"""
SS2D-ViT v4 평가 스크립트 (DC block + mask/sens 입력 지원)

기존 eval.py는 v3 시그니처(model(img, ksp))만 지원하므로 v4 전용으로 분리.
원본 파일은 건드리지 않는다는 ss2d_v4_changes.md §2 원칙 준수.

사용법:
  python eval_v4.py
  python eval_v4.py --ckpt logs/SS2D_ViT_R4_brain320_v4/ss2d_vit_epoch_195.pt

결과:
  results/eval_ss2d_v4_<ckpt_basename>.csv
  results/eval_ss2d_v4_<ckpt_basename>_summary.txt
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import csv
import datetime
import pytz
from tqdm.auto import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'configs'))
sys.path.append(os.path.join(current_dir, 'dataloaders'))
sys.path.append(os.path.join(current_dir, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(current_dir, 'models', 'mamba_eternet'))

from dataloader_h5_v4 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader
from u_choh_SSIM import SSIM


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_val = target.max()
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def nmse(pred, target):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='logs/SS2D_ViT_R4_brain320_v4/ss2d_vit_best.pt',
                        help='v4 체크포인트 경로')
    parser.add_argument('--results-dir', type=str,
                        default='results',
                        help='평가 결과 CSV/요약 저장 폴더 (프로젝트 컨벤션)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('====================================================')
    print(f' 평가: SS2D-ViT v4 | {args.ckpt}')
    print('====================================================')
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

    model = load_model_v4(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {n_params/1e6:.1f}M")

    val_dataset = FastMRI_H5_Dataloader('./fastMRI_data/multicoil_val',
                                        num_files=None, random_mask=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Val 샘플 수: {len(val_dataset)}\n")

    criterion_l1   = nn.L1Loss()
    criterion_ssim = SSIM().to(device)

    all_ssim, all_l1, all_psnr, all_nmse = [], [], [], []
    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_base = os.path.basename(args.ckpt)
    csv_path = os.path.join(args.results_dir, f'eval_ss2d_v4_{ckpt_base}.csv')
    summary_path = os.path.join(args.results_dir, f'eval_ss2d_v4_{ckpt_base}_summary.txt')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample_idx', 'psnr_db', 'nmse', 'ssim', 'l1'])

        eval_bar = tqdm(val_loader, desc='Eval v4', unit='batch')
        with torch.no_grad():
            for i, sample in enumerate(eval_bar):
                data_in     = sample['data'].float().to(device)
                data_in_img = sample['data_img'].float().to(device)
                data_ref    = sample['label'].float().to(device)
                mask_in     = sample['mask'].float().to(device)
                sens_in     = sample['sens'].float().to(device)

                with torch.amp.autocast('cuda'):
                    out = model(data_in_img, data_in, mask_in, sens_in)

                out_f = out.float()
                ref_f = data_ref.float()

                psnr_val = psnr(out_f, ref_f).item()
                nmse_val = nmse(out_f, ref_f)
                ssim_val = criterion_ssim(out_f, ref_f).item()
                l1_val   = criterion_l1(out_f, ref_f).item()

                all_psnr.append(psnr_val)
                all_nmse.append(nmse_val)
                all_ssim.append(ssim_val)
                all_l1.append(l1_val)

                writer.writerow([i, f'{psnr_val:.2f}', f'{nmse_val:.4f}',
                                 f'{ssim_val:.4f}', f'{l1_val:.4f}'])
                eval_bar.set_postfix(
                    PSNR=f'{psnr_val:.2f}dB',
                    NMSE=f'{nmse_val:.4f}',
                    SSIM=f'{ssim_val:.4f}',
                    L1=f'{l1_val:.4f}',
                )

    summary_lines = [
        '========== SS2D-ViT v4 평가 결과 ==========',
        f'체크포인트: {args.ckpt}',
        f'샘플 수: {len(all_psnr)}',
        f'PSNR : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB',
        f'NMSE : {np.mean(all_nmse):.4f} ± {np.std(all_nmse):.4f}',
        f'SSIM : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}',
        f'L1   : {np.mean(all_l1):.4f} ± {np.std(all_l1):.4f}',
    ]
    print('\n' + '\n'.join(summary_lines))
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f'\nCSV 저장: {csv_path}')
    print(f'요약 저장: {summary_path}')


if __name__ == '__main__':
    main()
