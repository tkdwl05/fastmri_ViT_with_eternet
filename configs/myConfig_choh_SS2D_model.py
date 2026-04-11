"""
SS2D-ViT 모델 설정 (fastMRI brain 320×320 표준)
  - 인코더: ViT (원본과 동일 구조, 크기만 GPU에 맞게 조정)
  - 디코더: ViT 트랜스포머 + SS2D (Bidirectional GRU 대체)
"""

import os

PATH_FOLDER = 'logs/SS2D_ViT_R4_brain320/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 (fastMRI brain AXFLAIR 표준) ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (32, 32)      # 320/32 = 10 → 10×10 = 100 patches
INPUT_CHANNELS = 32        # 16 coils × 2 (real/imag)

# ── 학습 설정 ──
BATCH_SIZE = 8           # 1 → 8: gradient noise 감소, scheduler 사이클 정상화
NUM_EPOCHS = 200
LEARNING_RATE_ADAM = 2e-4   # BS 8배 증가에 맞춘 보수적 scale-up (1e-4 → 2e-4)
LAMBDA_REGULAR_PER_PIXEL = 1e-7
LAMBDA_SSIM_PER_PIXEL = 0.2

# ── 인코더 파라미터 (ViT-Small, 8GB GPU 기준) ──
# 원본 ViT-Base: hidden=768, layer=12, mlp=3072, head=12
NUM_VIT_ENCODER_HIDDEN = 384
NUM_VIT_ENCODER_LAYER  = 6
NUM_VIT_ENCODER_MLP_SIZE = 1536
NUM_VIT_ENCODER_HEAD   = 6

# ── SS2D 파라미터 (GRU 대체) ──
# SS2D: 이미지를 직접 2D로 처리하므로 hidden 크기가 이미지 크기에 무관
NUM_SS2D_D_INNER = 32    # SS2D 내부 차원 (메모리↑ = 표현력↑)
NUM_SS2D_D_STATE = 8     # SSM 상태 차원 N
NUM_SS2D_OUT_CH  = 20    # SS2D 출력 채널 (원본 2*eter_hidden=20과 동일)

# ── 디코더 파라미터 ──
NUM_VIT_DECODER_HEAD              = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN    = 2048
NUM_VIT_DECODER_DIM               = 512
NUM_VIT_DECODER_DIM_HEAD          = 64
NUM_VIT_DECODER_DEPTH             = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 256
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 16
