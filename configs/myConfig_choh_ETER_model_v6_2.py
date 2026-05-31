"""
ETER-ViT 모델 설정 (v6_2) — v6_1 over-sharpening 회귀에 대한 λ_grad 완화

v6_1 결과 (vs v6, 7270 풀평가):
  PSNR  33.59 → 33.36 (-0.23)
  SSIM  0.8862 → 0.8865 (+0.0003, 실질적으로 동일)
  L1    8.21  → 8.46  (악화)
  NMSE  0.01143 → 0.01207 (악화)

SS2D v6_2 와 동일 처방: λ_grad 10.0 → 3.0, v6 best 에서 재시작.
"""

import os

PATH_FOLDER = 'logs/ETER_ViT_R4_brain320_v6_2/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (16, 16)
INPUT_CHANNELS = 32

# ── 학습 설정 ──
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE_ADAM = 5e-5
LAMBDA_REGULAR_PER_PIXEL = 3e-5
LAMBDA_SSIM_PER_PIXEL = 1.0
LAMBDA_GRAD_PER_PIXEL = 3.0              # v6_1 10.0 → v6_2 3.0

# ── 일반화 (v6_1 동일) ──
DROPOUT = 0.2
TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── EarlyStop ──
EARLYSTOP_PATIENCE = 5
VAL_EVERY_N_EPOCHS = 5

# ── v6_2: v6 best 로부터 resume ──
RESUME_CKPT = './logs/ETER_ViT_R4_brain320_v6/eter_vit_best.pt'

# ── 인코더 ──
NUM_VIT_ENCODER_HIDDEN   = 384
NUM_VIT_ENCODER_LAYER    = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD     = 6

# ── ETER GRU ──
NUM_ETER_HORI_HIDDEN = 6
NUM_ETER_VERT_HIDDEN = 6

# ── 디코더 ──
NUM_VIT_DECODER_HEAD               = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN     = 2048
NUM_VIT_DECODER_DIM                = 512
NUM_VIT_DECODER_DIM_HEAD           = 64
NUM_VIT_DECODER_DEPTH              = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 64
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 8
