"""
SS2D-ViT 모델 설정 (v6) — v5 resume + EarlyStop 기준 수정

v5 의 EarlyStop 이 ep 12에 너무 일찍 발동했고 (composite 4-metric 평균이 SSIM 단조
증가 경향과 어긋남), best ckpt 기준도 composite 였다. v6 는:
  - v5 의 가장 발전된 ckpt (epoch_10.pt, val SSIM 0.7257) 부터 resume
  - best 기준: val SSIM 단일 (composite 폐기)
  - patience: 5 val check → 10 val check (정체 판정에 더 보수적)
  - VAL_EVERY_N_EPOCHS = 5 (매 epoch val 폐지 → 학습 시간 절약)
  - 나머지 (data/regularization/augmentation/스케줄러/모델) 는 v5 와 동일
"""

import os

PATH_FOLDER = 'logs/SS2D_ViT_R4_brain320_v6/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (16, 16)
INPUT_CHANNELS = 32

# ── 학습 설정 ──
BATCH_SIZE = 4
NUM_EPOCHS = 200
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 3e-5
LAMBDA_SSIM_PER_PIXEL = 1.0

# ── 일반화 (v5 동일) ──
DROPOUT = 0.2
TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── v6 변경: EarlyStop ──
EARLYSTOP_PATIENCE = 10                  # val check 10회 정체 → ~50ep
VAL_EVERY_N_EPOCHS = 5                   # train_best 트리거 폐지

# ── v6 신규: resume ──
RESUME_CKPT = './logs/SS2D_ViT_R4_brain320_v5/ss2d_vit_epoch_10.pt'

# ── DC block (v5 동일) ──
DC_K_SCALE_RATIO = 100.0
DC_INIT_ALPHA = 1.0

# ── 인코더 ──
NUM_VIT_ENCODER_HIDDEN = 384
NUM_VIT_ENCODER_LAYER  = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD   = 6

# ── SS2D ──
NUM_SS2D_D_INNER = 64
NUM_SS2D_D_STATE = 16
NUM_SS2D_OUT_CH  = 20

# ── 디코더 ──
NUM_VIT_DECODER_HEAD              = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN    = 2048
NUM_VIT_DECODER_DIM               = 512
NUM_VIT_DECODER_DIM_HEAD          = 64
NUM_VIT_DECODER_DEPTH             = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 64
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 8
