import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import sys
import torch
import torch.nn as nn
import time
import datetime
import pytz

# ========================================================
# [Phase 2 적용] 모델 관련 모듈 임포트
# ========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'configs'))
sys.path.append(os.path.join(current_dir, 'dataloaders'))
sys.path.append(os.path.join(current_dir, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(current_dir, 'models', 'mae'))
sys.path.append(os.path.join(current_dir, 'models', 'vit_pytorch'))
# ========================================================
from myConfig_choh_model3 import *
from u_choh_model_ETER_ViT import choh_ViT
from u_choh_model_ETER_ViT import choh_Decoder3_ETER_skip_up_tail
from u_choh_SSIM import SSIM

# ========================================================
# [Phase 1 적용] 자체제작 원본 .h5 FastMRI Dataset 임포트
# (구버전의 에러나던 .mat 로더를 대체)
# ========================================================
from dataloader_h5 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader

def main():
    print('====================================================')
    print(' [통합 버전] ViT-ETER_net FastMRI_H5 Training')
    print('====================================================')

    # Device configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))

    # 1. 모델 아키텍처 정의 (ViT Encoder)
    vit_choh = choh_ViT(
        image_size = (384, 384),
        patch_size = (32, 32),
        num_classes = 1000,
        dim = NUM_VIT_ENCODER_HIDDEN,
        depth = NUM_VIT_ENCODER_LAYER,
        heads = NUM_VIT_ENCODER_HEAD,
        mlp_dim = NUM_VIT_ENCODER_MLP_SIZE,
        channels=32,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)

    # 2. 모델 아키텍처 정의 (ETER-net Decoder)
    choh_decoder = choh_Decoder3_ETER_skip_up_tail(
        encoder = vit_choh,
        eter_n_hori_hidden = NUM_ETER_HORI_HIDDEN,
        eter_n_vert_hidden = NUM_ETER_VERT_HIDDEN,
        decoder_dim = NUM_VIT_DECODER_DIM,
        decoder_depth = NUM_VIT_DECODER_DEPTH,
        decoder_heads= NUM_VIT_DECODER_HEAD,
        decoder_dim_head = NUM_VIT_DECODER_DIM_HEAD,
        decoder_dim_mlp_hidden = NUM_VIT_DECODER_DIM_MLP_HIDDEN,
        decoder_out_ch_up_tail = NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH,
        decoder_out_feat_size_final_linear =  NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT
    ).to(device)
    print("\n모델 구조 메모리 적재 완료")

    # 3. 옵티마이저, 스케줄러, 손실 함수 세팅
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIM().to(device)
    optimizer = torch.optim.Adam(choh_decoder.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=LAMBDA_REGULAR_PER_PIXEL)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=0)

    # 4. 데이터셋 세팅
    try:
        print("\nFastMRI 원본(.h5) 데이터 파이프라인 연결 중...")
        # 데이터가 너무 많으면 메모리가 터지므로 num_files를 조절하세요.
        choh_data_train = FastMRI_H5_Dataloader('./fastMRI_data/multicoil_train', num_files=None)
        trainloader = DataLoader(choh_data_train, batch_size=1, shuffle=True)
        print("Dataloader 준비 완료!")
    except Exception as e:
        print(f"데이터셋 연결 오류 발생: {e}")
        return

    # 5. 본격적인 Training Loop
    print("\n학습(Training) 루프를 시작합니다")
    num_epochs = 10  # 빠른 시작을 위한 기본 10 에폭
    lambda_ssim_per_pixel = LAMBDA_SSIM_PER_PIXEL

    scaler = torch.amp.GradScaler('cuda')
    choh_decoder.train()
    tic1 = time.time()

    for epoch in range(num_epochs):
        print(f'\n--- Epoch [{epoch+1}/{num_epochs}] ---')

        for i_batch, sample_batched in enumerate(trainloader):
            # H5 Loader에서 나온 데이터: data_in(Aliased K-space), data_in_img(Aliased Img), data_ref(GT Label)
            data_in = sample_batched['data'].type(torch.cuda.FloatTensor).to(device)
            data_in_img = sample_batched['data_img'].type(torch.cuda.FloatTensor).to(device)
            data_ref = sample_batched['label'].type(torch.cuda.FloatTensor).to(device)

            with torch.amp.autocast('cuda'):
                out = choh_decoder(data_in_img, data_in)

                # Loss 계산
                loss_pixel = criterion_l1(out, data_ref)
                loss_ssim = 1 - criterion_ssim(out, data_ref)
                loss = loss_pixel + lambda_ssim_per_pixel * loss_ssim

            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            print(f'Batch[{i_batch+1}/{len(trainloader)}] | Total Loss: {loss.item():.4f} | L1 pix: {loss_pixel.item():.4f} | 1-SSIM: {loss_ssim.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

        # 5 에폭마다 체크포인트 저장
        if (epoch+1) % 5 == 0:
            torch.save(choh_decoder.state_dict(), f'./models/choh_vit_eternet_epoch_{epoch+1}.pt')
            print(f'Checkpoint 저장 완료: models/choh_vit_eternet_epoch_{epoch+1}.pt')

    toc1 = time.time()
    print(f'\n학습 완료! 총 소요 시간: {toc1 - tic1:.2f}초')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT-ETER_net FastMRI training entrypoint')
    parser.add_argument(
        '--mode',
        choices=['legacy', 'eter', 'ss2d'],
        default='legacy',
        help='legacy: 기존 통합 스크립트 실행 / eter: main_train_eter.py 실행 / ss2d: main_train_ss2d.py 실행'
    )
    args, _ = parser.parse_known_args()

    if args.mode == 'legacy':
        main()
    elif args.mode == 'eter':
        from main_train_eter import main as _eter_main

        print('====================================================')
        print(' [통합 모드] main_train_eter.py 실행')
        print('====================================================')
        _eter_main()
    else:
        from main_train_ss2d import main as _ss2d_main

        print('====================================================')
        print(' [통합 모드] main_train_ss2d.py 실행')
        print('====================================================')
        _ss2d_main()
