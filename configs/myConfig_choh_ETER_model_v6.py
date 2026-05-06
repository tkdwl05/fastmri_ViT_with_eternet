"""
ETER-ViT 모델 설정 (v6) — v5 resume + EarlyStop 기준 수정

v5 의 EarlyStop 이 ep 9에 잘못 발동: SSIM 은 ep1→ep9 로 단조 증가 (0.6865 → 0.7421)
했는데 composite 기준으로는 ep 4 (0.7217) 가 best 로 잡혀 그 이후 no-improve.
ckpt 도 composite-best 시점 (ep 4) 만 저장됨.

v6:
  - v5 의 가장 발전된 ckpt (epoch_5.pt, val SSIM 0.7249) 부터 resume
  - best 기준: val SSIM 단일
  - patience: 5 → 10
  - VAL_EVERY_N_EPOCHS = 5
  - 나머지는 v5 동일
"""

import os

PATH_FOLDER = 'logs/ETER_ViT_R4_brain320_v6/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (16, 16)
INPUT_CHANNELS = 32

# ── 학습 설정 ──
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 3e-5
LAMBDA_SSIM_PER_PIXEL = 1.0

# ── 일반화 (v5 동일) ──
DROPOUT = 0.2
TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── v6 변경 ──
EARLYSTOP_PATIENCE = 10
VAL_EVERY_N_EPOCHS = 5

# ── v6 신규: resume ──
RESUME_CKPT = './logs/ETER_ViT_R4_brain320_v5/eter_vit_epoch_5.pt'

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
