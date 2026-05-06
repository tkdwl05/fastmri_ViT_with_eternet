"""
SS2D-ViT 모델 설정 (v5)
  - 분포 폭 확장: dataloader가 320×320이 아닌 파일도 center-crop/pad로 흡수
  - 일반화 강화: Transformer dropout 0.1 → 0.2, weight_decay 1e-5 → 3e-5
  - H/V flip augmentation (train only)
  - EarlyStopping (val composite 기준, patience=5 val check)
  - 아키텍처/스케줄러/capacity는 v4 그대로 유지 → 모델 코드 변경 없음
"""

import os

PATH_FOLDER = 'logs/SS2D_ViT_R4_brain320_v5/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (16, 16)
INPUT_CHANNELS = 32        # 16 coils × 2 (real/imag)

# ── 학습 설정 ──
BATCH_SIZE = 4    # v4 와 동일 (8GB 메모리 제약)
NUM_EPOCHS = 200
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 3e-5    # v4 1e-5 → v5 3e-5
LAMBDA_SSIM_PER_PIXEL = 1.0

# ── v5 신규: 일반화/EarlyStopping/Augmentation ──
DROPOUT = 0.2                      # v4 0.1 → v5 0.2
EARLYSTOP_PATIENCE = 5             # 연속 val check N회 동안 composite 개선 없으면 정지
TRAIN_AUGMENT = True               # H/V flip random (train only)
TRAIN_AUGMENT_FLIP_P = 0.5         # 각 축별 flip 확률

# ── DC block (v4 와 동일) ──
DC_K_SCALE_RATIO = 100.0
DC_INIT_ALPHA = 1.0

# ── 인코더 (v4 와 동일) ──
NUM_VIT_ENCODER_HIDDEN = 384
NUM_VIT_ENCODER_LAYER  = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD   = 6

# ── SS2D (v4 와 동일) ──
NUM_SS2D_D_INNER = 64
NUM_SS2D_D_STATE = 16
NUM_SS2D_OUT_CH  = 20

# ── 디코더 (v4 와 동일) ──
NUM_VIT_DECODER_HEAD              = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN    = 2048
NUM_VIT_DECODER_DIM               = 512
NUM_VIT_DECODER_DIM_HEAD          = 64
NUM_VIT_DECODER_DEPTH             = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 64
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 8
