"""
ETER-ViT 모델 설정 (v7_titan) — TITAN RTX x2 24GB, 원본 ETER-Net 사양 복원

v7 (320×320, ViT-Small, RefinementBlock) 대비:
  - IMAGE_SIZE 320→384 (원본 ETER-Net 해상도)
  - ViT 인코더: dim 384→768, layers 6→12, heads 6→12, MLP 1536→3072 (ViT-Base)
  - GRU hidden 유지 (10)
  - 최종 합성: RefinementBlock(3 ResBlock) → UNet_choh_skip(depth=3, wf=6)
  - BATCH_SIZE: smoke 결과로 결정 (시작값 4)
  - 그 외 (dropout, weight_decay, EarlyStop, augment) 는 v7 동일
"""

import os

PATH_FOLDER = 'logs/ETER_ViT_R4_brain384_v7_titan/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 (v7 320 → v7_titan 384) ──
IMAGE_SIZE = (384, 384)
PATCH_SIZE = (16, 16)             # 384/16 = 24, num_patches = 576
INPUT_CHANNELS = 32

# ── 학습 설정 ──
BATCH_SIZE = 8                    # 2026-05-26 BS=4→8 (ETER smoke 14.7GB@BS=12, BS=8 더 안전)
NUM_EPOCHS = 50                   # 2026-05-26 200→50 (발열 운영 기간 단축)
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 3e-5
LAMBDA_SSIM_PER_PIXEL = 1.0

# ── 일반화 (v5~ 레시피) ──
DROPOUT = 0.2
TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── EarlyStop / val 주기 (v5~ 레시피) ──
EARLYSTOP_PATIENCE = 10
VAL_EVERY_N_EPOCHS = 5

# ── v7_titan: scratch 학습 ──
RESUME_CKPT = None

# ── DataLoader ──
NUM_WORKERS_TRAIN = 16
NUM_WORKERS_VAL   = 4
PREFETCH_FACTOR   = 4

# ── 인코더 (ViT-Base 복원) ──
NUM_VIT_ENCODER_HIDDEN = 768
NUM_VIT_ENCODER_LAYER  = 12
NUM_VIT_ENCODER_MLP_SIZE = 3072
NUM_VIT_ENCODER_HEAD   = 12

# ── ETER GRU (원본 ETER-Net 값 유지) ──
NUM_ETER_HORI_HIDDEN = 10
NUM_ETER_VERT_HIDDEN = 10

# ── U-Net 후처리 (원본 ETER-Net 복원) ──
ETER_UNET_DEPTH = 3
ETER_UNET_WF    = 6

# ── Brain mask (v7_titan: 배경 부풀림 차단용) ──
# 실제 mask 생성은 dataloaders/dataloader_h5_v5.py:243 에 하드코딩됨:
#   gt_rss > threshold_otsu(non_zero) * 0.4  →  largest CC keep (no erode, no fill_holes)
# (과거의 BRAIN_MASK_THRESHOLD=0.05 / ERODE_ITER=1 config 노출은 미사용이라 제거 — 73k 검증 후 §10-C 채택)

# ── Composite metric (best/EarlyStop 기준) ──
# composite = W_SSIM * SSIM_m + W_PSNR * min(PSNR,PSNR_NORM)/PSNR_NORM + W_NMSE * max(0, 1 - min(NMSE,1))
COMPOSITE_W_SSIM = 0.5
COMPOSITE_W_PSNR = 0.3
COMPOSITE_W_NMSE = 0.2
PSNR_NORM        = 40.0    # PSNR 정규화 분모 (dB)

# ── 디코더 (v7 동일, dim 512 유지) ──
NUM_VIT_DECODER_HEAD              = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN    = 2048
NUM_VIT_DECODER_DIM               = 512
NUM_VIT_DECODER_DIM_HEAD          = 64
NUM_VIT_DECODER_DEPTH             = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 64
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 8
