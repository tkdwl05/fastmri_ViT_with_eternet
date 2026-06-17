"""
SS2D-ViT 모델 설정 (v7) — 24GB VRAM 환경, capacity 복원, scratch 학습

v6 (8GB VRAM) 대비:
  - BATCH_SIZE 4 → 16
  - RESUME_CKPT 폐기 (v6 ckpt 가 새 머신에 없음 + scratch 기준선 측정 목적)
  - PYTORCH_CUDA_ALLOC_CONF 환경변수 학습 스크립트에서 제거
  - 나머지 (DC block, SS2D dim, ViT 인코더/디코더, dropout, augmentation, EarlyStop)
    는 v6 동일 — 첫 단계는 BATCH_SIZE 만 증가시켜 fair baseline 확립
  - 추후 BATCH_SIZE 24 / SS2D d_state 32 등 추가 capacity 실험은 v7.1, v7.2 로 분리
"""

import os

PATH_FOLDER = 'logs/SS2D_ViT_R4_brain320_v7/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 (v6 동일) ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (16, 16)
INPUT_CHANNELS = 32

# ── 학습 설정 ──
BATCH_SIZE = 16                          # v6 의 4 → 16 (24GB VRAM 활용)
NUM_EPOCHS = 200
LEARNING_RATE_ADAM = 2e-4                # BS 증가에 따른 LR scaling 은 첫 epoch 결과 보고 결정
LAMBDA_REGULAR_PER_PIXEL = 3e-5
LAMBDA_SSIM_PER_PIXEL = 1.0

# ── 일반화 (v6 동일) ──
DROPOUT = 0.2
TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── EarlyStop / val 주기 (v6 동일) ──
EARLYSTOP_PATIENCE = 10
VAL_EVERY_N_EPOCHS = 5

# ── v7: scratch 학습 (resume 폐기) ──
RESUME_CKPT = None

# ── DC block (v6 동일) ──
DC_K_SCALE_RATIO = 100.0
DC_INIT_ALPHA = 1.0

# ── DataLoader (24 core CPU 활용) ──
NUM_WORKERS_TRAIN = 16
NUM_WORKERS_VAL   = 4
PREFETCH_FACTOR   = 4

# ── 인코더 (v6 동일) ──
NUM_VIT_ENCODER_HIDDEN = 384
NUM_VIT_ENCODER_LAYER  = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD   = 6

# ── SS2D (v6 동일) ──
NUM_SS2D_D_INNER = 64
NUM_SS2D_D_STATE = 16
NUM_SS2D_OUT_CH  = 20

# ── 디코더 (v6 동일) ──
NUM_VIT_DECODER_HEAD              = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN    = 2048
NUM_VIT_DECODER_DIM               = 512
NUM_VIT_DECODER_DIM_HEAD          = 64
NUM_VIT_DECODER_DEPTH             = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 64
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 8
