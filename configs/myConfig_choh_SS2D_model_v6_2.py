"""
SS2D-ViT 모델 설정 (v6_2) — v6_1 over-sharpening 회귀에 대한 λ_grad 완화

v6_1 결과 (vs v6, 7270 풀평가):
  PSNR  34.81 → 34.18 (-0.63)
  SSIM  0.8913 → 0.8953 (+0.004, 미세 상승)
  L1    7.37  → 7.76  (악화)
  NMSE  0.00906 → 0.01018 (악화)

[docs/ss2d_eter_v6_1_changes.md](../docs/ss2d_eter_v6_1_changes.md) 의 fallback
시나리오 ("gradient loss 가 over-sharpening 을 유발하면 λ_grad 5.0 으로 줄여
재학습") 가 그대로 발생. v6_2 는 λ_grad 만 단일 변수로 줄여 over-sharpening
가설을 직접 검증한다.

변경점 (vs v6_1):
  - LAMBDA_GRAD_PER_PIXEL 10.0 → 3.0 (-70%)
  - RESUME_CKPT: v6_1 best 가 아닌 v6 best 에서 재시작
    (v6_1 의 over-edged state 누적 회피, 단일변수 비교 명확화)
  - 나머지 (epochs / LR / dropout / weight_decay / model / DC) 는 v6_1 동일
"""

import os

PATH_FOLDER = 'logs/SS2D_ViT_R4_brain320_v6_2/'
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
LAMBDA_GRAD_PER_PIXEL = 3.0              # v6_1 10.0 → v6_2 3.0 (-70%)

# ── 일반화 (v6_1 동일) ──
DROPOUT = 0.2
TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── EarlyStop (v6_1 동일) ──
EARLYSTOP_PATIENCE = 5
VAL_EVERY_N_EPOCHS = 5

# ── v6_2: v6 best 로부터 resume (v6_1 의 over-edged state 누적 회피) ──
RESUME_CKPT = './logs/SS2D_ViT_R4_brain320_v6/ss2d_vit_best.pt'

# ── DC block (v6 동일) ──
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
