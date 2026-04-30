"""
SS2D-ViT 모델 설정 (v4)
  - A: SS2D capacity 증설 (d_inner 32→64, d_state 8→16)
  - B: weight_decay 1e-7→1e-5, Transformer dropout=0.1
  - C: Data Consistency block (모델 출력 1ch magnitude → 2ch complex + 1-iter DC)
"""

import os

PATH_FOLDER = 'logs/SS2D_ViT_R4_brain320_v4/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 (fastMRI brain AXFLAIR 표준) ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (16, 16)
INPUT_CHANNELS = 32        # 16 coils × 2 (real/imag)

# ── 학습 설정 ──
BATCH_SIZE = 4    # v3 8 → v4 4 (d_inner=64 유지를 위해 backward 메모리 확보, 2026-04-27)
NUM_EPOCHS = 200
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 1e-5    # v3 1e-7 → v4 1e-5 (실질 regularization)
LAMBDA_SSIM_PER_PIXEL = 1.0

# v4 추가: dropout / DC block
DROPOUT = 0.1                       # Transformer decoder dropout
DC_K_SCALE_RATIO = 100.0            # val_amp_X_img(1e6) / val_amp_X_ksp(1e4) 비율, DC 내부 k-space 스케일 보정
DC_INIT_ALPHA = 1.0                 # DC mix 초기값 (1.0 = 샘플 위치 hard DC)

# ── 인코더 파라미터 (ViT-Small, 8GB GPU 기준) ──
NUM_VIT_ENCODER_HIDDEN = 384
NUM_VIT_ENCODER_LAYER  = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD   = 6

# ── SS2D 파라미터 (v4: capacity 증설) ──
NUM_SS2D_D_INNER = 64    # v3 32 → v4 64 (표현력 2배, 8GB 한계 내)
NUM_SS2D_D_STATE = 16    # v3 8  → v4 16 (장거리 의존성 포착)
NUM_SS2D_OUT_CH  = 20

# ── 디코더 파라미터 ──
NUM_VIT_DECODER_HEAD              = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN    = 2048
NUM_VIT_DECODER_DIM               = 512
NUM_VIT_DECODER_DIM_HEAD          = 64
NUM_VIT_DECODER_DEPTH             = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 64   # 256→64: patch 축소로 채널 줄여 메모리 절약
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 8    # 16→8: patch_size=16에서 1회 upsample
