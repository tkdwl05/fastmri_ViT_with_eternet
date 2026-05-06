"""
ETER-ViT 모델 설정 (v5)
  - 분포 폭 확장: dataloader_h5_v5 사용 (320×320 외 파일도 흡수)
  - 일반화 강화: Transformer decoder dropout 0.0→0.2, weight_decay 1e-7→3e-5
  - H/V flip augmentation (train only)
  - EarlyStopping (val composite, patience=5 val check)
  - 아키텍처/스케줄러는 v4 그대로 유지 (모델은 v5 thin wrapper 로 dropout 만 주입)
"""

import os

PATH_FOLDER = 'logs/ETER_ViT_R4_brain320_v5/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (16, 16)
INPUT_CHANNELS = 32

# ── 학습 설정 ──
BATCH_SIZE = 8                       # ETER 는 DC block 없어 v4 와 동일하게 8
NUM_EPOCHS = 200
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 3e-5      # v4 1e-7 → v5 3e-5 (SS2D v5 와 동일 수준)
LAMBDA_SSIM_PER_PIXEL = 1.0

# ── v5 신규 ──
DROPOUT = 0.2                        # decoder Transformer dropout (v4 0.0 → v5 0.2)
EARLYSTOP_PATIENCE = 5
TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── 인코더 (v4 와 동일) ──
NUM_VIT_ENCODER_HIDDEN   = 384
NUM_VIT_ENCODER_LAYER    = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD     = 6

# ── ETER GRU (v4 와 동일) ──
NUM_ETER_HORI_HIDDEN = 6
NUM_ETER_VERT_HIDDEN = 6

# ── 디코더 (v4 와 동일) ──
NUM_VIT_DECODER_HEAD               = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN     = 2048
NUM_VIT_DECODER_DIM                = 512
NUM_VIT_DECODER_DIM_HEAD           = 64
NUM_VIT_DECODER_DEPTH              = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 64
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 8
