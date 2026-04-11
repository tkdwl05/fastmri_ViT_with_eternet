"""
ETER-ViT 비교용 설정 (fastMRI brain 320×320 표준, SS2D-ViT와 동일 조건)
  - 인코더: ViT-Small (SS2D 모델과 동일)
  - 디코더: ViT 트랜스포머 + Bidirectional GRU (원본 ETER-Net)
"""

import os

PATH_FOLDER = 'logs/ETER_ViT_R4_brain320/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 (fastMRI brain AXFLAIR 표준) ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (32, 32)      # 320/32 = 10 → 10×10 = 100 patches
INPUT_CHANNELS = 32        # 16 coils × 2 (real/imag)

# ── 학습 설정 (SS2D와 동일) ──
BATCH_SIZE = 8           # 1 → 8: gradient noise 감소, scheduler 사이클 정상화
NUM_EPOCHS = 200
LEARNING_RATE_ADAM = 2e-4   # BS 8배 증가에 맞춘 보수적 scale-up (1e-4 → 2e-4)
LAMBDA_REGULAR_PER_PIXEL = 1e-7
LAMBDA_SSIM_PER_PIXEL = 0.2

# ── 인코더 파라미터 (SS2D 모델과 동일) ──
NUM_VIT_ENCODER_HIDDEN   = 384
NUM_VIT_ENCODER_LAYER    = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD     = 6

# ── ETER-Net GRU 파라미터 ──
# hidden 크기 = image_size × NUM_ETER_HIDDEN → 메모리 주의
# 320 × 2 = 640 (8GB 한계 내, 기존 384×2=768보다 작음)
NUM_ETER_HORI_HIDDEN = 2
NUM_ETER_VERT_HIDDEN = 2

# ── 디코더 파라미터 (SS2D와 동일) ──
NUM_VIT_DECODER_HEAD               = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN     = 2048
NUM_VIT_DECODER_DIM                = 512
NUM_VIT_DECODER_DIM_HEAD           = 64
NUM_VIT_DECODER_DEPTH              = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 256
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 16
