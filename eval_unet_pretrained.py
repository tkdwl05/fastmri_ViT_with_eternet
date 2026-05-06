"""
FastMRI 공식 Pre-trained U-Net 평가 스크립트

FastMRI leaderboard 우승 U-Net (brain multicoil)을 로컬 val 데이터에서 평가하여
SS2D-ViT와 비교할 수 있는 동일 형식의 메트릭을 생성한다.

모델:  Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4)
가중치: brain_leaderboard_state_dict.pt
전처리: UnetDataTransform(multicoil, mask_func=R4 equispaced) — mask → iFFT → RSS → normalize

사용법:
  python eval_unet_pretrained.py
  python eval_unet_pretrained.py --state-dict models/pretrained/brain_leaderboard_state_dict.pt

결과:
  results/eval_unet_pretrained.csv
  results/eval_unet_pretrained_summary.txt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import csv
import datetime
import pytz
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

import fastmri
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models import Unet


# ──────────────────────────────────────────────
#  평가 지표 (denormalized 원본 스케일에서 계산)
# ──────────────────────────────────────────────

def calc_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR (dB). 높을수록 좋음."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    max_val = target.max()
    return 20 * np.log10(max_val / np.sqrt(mse))


def calc_nmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Normalized MSE. 낮을수록 좋음."""
    return np.linalg.norm(pred - target) ** 2 / np.linalg.norm(target) ** 2


def calc_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """SSIM (scikit-image). data_range를 자동 감지."""
    data_range = target.max() - target.min()
    return compare_ssim(target, pred, data_range=data_range)


def main():
    parser = argparse.ArgumentParser(description='FastMRI Pre-trained U-Net 평가')
    parser.add_argument('--state-dict', type=str,
                        default='models/pretrained/brain_leaderboard_state_dict.pt',
                        help='Pre-trained 가중치 경로')
    parser.add_argument('--data-path', type=str,
                        default='./fastMRI_data/multicoil_val',
                        help='val 데이터 경로')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='결과 저장 폴더')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda 또는 cpu')
    parser.add_argument('--acceleration', type=int, default=4,
                        help='가속 배율 (기본: R=4)')
    parser.add_argument('--center-fraction', type=float, default=0.08,
                        help='중앙 fraction (기본: 0.08)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('====================================================')
    print(' 평가: FastMRI Pre-trained U-Net (brain multicoil)')
    print('====================================================')
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

    # ── 1. 모델 로드 ──
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    state_dict = torch.load(args.state_dict, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {n_params / 1e6:.1f}M")
    print(f"가중치: {args.state_dict}")

    # ── 2. 데이터 로드 (R=4 equispaced 마스크 적용) ──
    mask_func = create_mask_for_mask_type(
        'equispaced_fraction',
        [args.center_fraction],
        [args.acceleration]
    )
    data_transform = T.UnetDataTransform(
        which_challenge="multicoil",
        mask_func=mask_func,
        use_seed=True,  # val에서는 고정 시드로 재현 가능하게
    )
    dataset = SliceDataset(
        root=args.data_path,
        transform=data_transform,
        challenge="multicoil",
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"Val 샘플 수: {len(dataset)}")
    print(f"마스크: equispaced_fraction R={args.acceleration}, center={args.center_fraction}")
    print()

    # ── 3. 평가 ──
    all_psnr, all_nmse, all_ssim, all_l1 = [], [], [], []
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, 'eval_unet_pretrained.csv')
    summary_path = os.path.join(args.results_dir, 'eval_unet_pretrained_summary.txt')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample_idx', 'psnr_db', 'nmse', 'ssim', 'l1'])

        eval_bar = tqdm(dataloader, desc='Eval U-Net', unit='batch')
        with torch.no_grad():
            for i, batch in enumerate(eval_bar):
                image, target, mean, std, fname, slice_num, max_value = batch

                # U-Net 추론 (normalized 입력)
                output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()

                # denormalize (원본 스케일 복원)
                mean_val = mean.unsqueeze(1).unsqueeze(2)
                std_val = std.unsqueeze(1).unsqueeze(2)
                output_denorm = (output * std_val + mean_val).numpy().squeeze()
                target_denorm = (target * std_val + mean_val).numpy().squeeze()

                # 지표 계산 (원본 스케일)
                psnr_val = calc_psnr(output_denorm, target_denorm)
                nmse_val = calc_nmse(output_denorm, target_denorm)
                ssim_val = calc_ssim(output_denorm, target_denorm)
                l1_val = np.mean(np.abs(output_denorm - target_denorm))

                all_psnr.append(psnr_val)
                all_nmse.append(nmse_val)
                all_ssim.append(ssim_val)
                all_l1.append(l1_val)

                writer.writerow([i, f'{psnr_val:.2f}', f'{nmse_val:.6f}',
                                 f'{ssim_val:.4f}', f'{l1_val:.6f}'])
                eval_bar.set_postfix(
                    PSNR=f'{psnr_val:.2f}dB',
                    NMSE=f'{nmse_val:.4f}',
                    SSIM=f'{ssim_val:.4f}',
                )

    # ── 4. 결과 요약 ──
    summary_lines = [
        '========== FastMRI Pre-trained U-Net 평가 결과 ==========',
        f'가중치: {args.state_dict}',
        f'모델: Unet(in=1, out=1, chans=256, pool=4)',
        f'파라미터 수: {n_params / 1e6:.1f}M',
        f'마스크: equispaced_fraction R={args.acceleration}, center={args.center_fraction}',
        f'샘플 수: {len(all_psnr)}',
        f'PSNR : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB',
        f'NMSE : {np.mean(all_nmse):.6f} ± {np.std(all_nmse):.6f}',
        f'SSIM : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}',
        f'L1   : {np.mean(all_l1):.6f} ± {np.std(all_l1):.6f}',
    ]
    print('\n' + '\n'.join(summary_lines))
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f'\nCSV 저장: {csv_path}')
    print(f'요약 저장: {summary_path}')


if __name__ == '__main__':
    main()
