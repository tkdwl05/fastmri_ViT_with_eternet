"""
ETER-ViT 모델 설정 (v7) — 24GB VRAM 환경, GRU capacity 복원, scratch 학습

v6 (8GB VRAM) 대비:
  - BATCH_SIZE 4 → 16 (v6 BS=8 OOM 으로 4 까지 강하했던 제약 해소)
  - GRU hidden 6 → 10 (원본 ETER-Net 기본값으로 복원)
  - RESUME_CKPT 폐기 (scratch 기준선)
  - PYTORCH_CUDA_ALLOC_CONF 환경변수 제거
  - 나머지 (ViT 인코더/디코더, dropout, augmentation, EarlyStop) 는 v6 동일
"""

import os

PATH_FOLDER = 'logs/ETER_ViT_R4_brain320_v7/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 (v6 동일) ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (16, 16)
INPUT_CHANNELS = 32

# ── 학습 설정 ──
BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 3e-5
LAMBDA_SSIM_PER_PIXEL = 1.0

# ── 일반화 (v6 동일) ──
DROPOUT = 0.2
TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── EarlyStop / val 주기 (v6 동일) ──
EARLYSTOP_PATIENCE = 10
VAL_EVERY_N_EPOCHS = 5

# ── v7: scratch 학습 ──
RESUME_CKPT = None

# ── DataLoader (24 core CPU 활용) ──
NUM_WORKERS_TRAIN = 16
NUM_WORKERS_VAL   = 4
PREFETCH_FACTOR   = 4

# ── 인코더 (v6 동일) ──
NUM_VIT_ENCODER_HIDDEN = 384
NUM_VIT_ENCODER_LAYER  = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD   = 6

# ── ETER GRU (v6: 6 → v7: 10, 원본 복원) ──
NUM_ETER_HORI_HIDDEN = 10
NUM_ETER_VERT_HIDDEN = 10

# ── 디코더 (v6 동일) ──
NUM_VIT_DECODER_HEAD              = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN    = 2048
NUM_VIT_DECODER_DIM               = 512
NUM_VIT_DECODER_DIM_HEAD          = 64
NUM_VIT_DECODER_DEPTH             = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 64
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 8
