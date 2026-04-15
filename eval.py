"""
평가 스크립트: multicoil_val 데이터로 SSIM / PSNR / NMSE / L1 측정
(brain AXFLAIR test 셋에는 reconstruction_rss가 없어 정량 평가 불가 → val 사용)

사용법 (v2 — patch16, RefinementBlock):
  python eval.py --model ss2d --ckpt logs/SS2D_ViT_R4_brain320_v2/ss2d_vit_best.pt
  python eval.py --model eter --ckpt logs/ETER_ViT_R4_brain320_v2/eter_vit_best.pt

주요 지표:
  PSNR (dB)  : 픽셀 단위 정확도. 높을수록 좋음. 30+양호, 40+우수
  NMSE       : 정규화 MSE (fastMRI 공식 지표). 낮을수록 좋음
  SSIM       : 지각적 구조 유사도. 높을수록 좋음 (0~1)
  L1         : 픽셀 평균 절대 오차. 낮을수록 좋음
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

from dataloader_h5 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader
from u_choh_SSIM import SSIM


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_val = target.max()
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def nmse(pred, target):
    """Normalized MSE: ||pred-target||² / ||target||²  (낮을수록 좋음)"""
    return (torch.norm(pred - target) ** 2 / torch.norm(target) ** 2).item()


def load_model(model_type: str, ckpt_path: str, device):
    if model_type == 'ss2d':
        from myConfig_choh_SS2D_model import (
            IMAGE_SIZE, PATCH_SIZE, INPUT_CHANNELS,
            NUM_VIT_ENCODER_HIDDEN, NUM_VIT_ENCODER_LAYER,
            NUM_VIT_ENCODER_MLP_SIZE, NUM_VIT_ENCODER_HEAD,
            NUM_SS2D_D_INNER, NUM_SS2D_D_STATE, NUM_SS2D_OUT_CH,
            NUM_VIT_DECODER_DIM, NUM_VIT_DECODER_DEPTH,
            NUM_VIT_DECODER_HEAD, NUM_VIT_DECODER_DIM_HEAD,
            NUM_VIT_DECODER_DIM_MLP_HIDDEN,
            NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
            NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT,
        )
        from u_choh_model_ETER_ViT import choh_ViT
        from u_choh_model_SS2D_ViT import choh_Decoder_SS2D_ViT

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
            decoder_out_feat_size_final_linear=NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT
        )

    elif model_type == 'eter':
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
    else:
        raise ValueError(f"model_type은 'ss2d' 또는 'eter'이어야 합니다. 입력값: {model_type}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['ss2d', 'eter'],
                        help='평가할 모델 종류: ss2d 또는 eter')
    parser.add_argument('--ckpt',  type=str, required=True,
                        help='체크포인트 경로 (.pt 파일)')
    parser.add_argument('--legacy-scale', action='store_true',
                        help='이전 스케일 팩터(img=1e6, ksp=1e4, gt=1e6)로 학습된 체크포인트 평가 시 사용')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('====================================================')
    print(f' 평가: {args.model.upper()} | {args.ckpt}')
    print('====================================================')
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

    model = load_model(args.model, args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {n_params/1e6:.1f}M")

    # Val 데이터 (파일 수 제한 없음, reconstruction_rss 있음 → 정량 평가 가능)
    val_dataset = FastMRI_H5_Dataloader('./fastMRI_data/multicoil_val', num_files=None, random_mask=False)
    if args.legacy_scale:
        val_dataset.val_amp_X_img = 1e6
        val_dataset.val_amp_X_ksp = 1e4
        val_dataset.val_amp_Y     = 1e6
        print('[Legacy Scale] img=1e6, ksp=1e4, gt=1e6')
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Val 샘플 수: {len(val_dataset)}\n")

    criterion_l1   = nn.L1Loss()
    criterion_ssim = SSIM().to(device)

    all_ssim, all_l1, all_psnr, all_nmse = [], [], [], []
    os.makedirs('results', exist_ok=True)
    csv_path = f'results/eval_{args.model}_{os.path.basename(args.ckpt)}.csv'

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample_idx', 'psnr_db', 'nmse', 'ssim', 'l1'])

        eval_bar = tqdm(val_loader, desc='Eval', unit='batch')
        with torch.no_grad():
            for i, sample in enumerate(eval_bar):
                data_in     = sample['data'].float().to(device)
                data_in_img = sample['data_img'].float().to(device)
                data_ref    = sample['label'].float().to(device)

                with torch.amp.autocast('cuda'):
                    out = model(data_in_img, data_in)

                # float32로 복원 후 계산 (autocast로 dtype 변환된 경우 대비)
                out_f   = out.float()
                ref_f   = data_ref.float()

                psnr_val = psnr(out_f, ref_f).item()
                nmse_val = nmse(out_f, ref_f)
                ssim_val = criterion_ssim(out_f, ref_f).item()
                l1_val   = criterion_l1(out_f, ref_f).item()

                all_psnr.append(psnr_val)
                all_nmse.append(nmse_val)
                all_ssim.append(ssim_val)
                all_l1.append(l1_val)

                writer.writerow([i, f'{psnr_val:.2f}', f'{nmse_val:.4f}', f'{ssim_val:.4f}', f'{l1_val:.4f}'])
                eval_bar.set_postfix(
                    PSNR=f'{psnr_val:.2f}dB',
                    NMSE=f'{nmse_val:.4f}',
                    SSIM=f'{ssim_val:.4f}',
                    L1=f'{l1_val:.4f}',
                )

    print('\n========== 최종 평균 결과 ==========')
    print(f'  PSNR : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB   ← 픽셀 정확도')
    print(f'  NMSE : {np.mean(all_nmse):.4f} ± {np.std(all_nmse):.4f}       ← fastMRI 공식 지표')
    print(f'  SSIM : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}       ← 지각적 구조 유사도')
    print(f'  L1   : {np.mean(all_l1):.4f} ± {np.std(all_l1):.4f}')
    print(f'\nCSV 저장 완료: {csv_path}')


if __name__ == '__main__':
    main()
