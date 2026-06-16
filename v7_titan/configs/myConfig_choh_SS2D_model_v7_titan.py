"""
SS2D-ViT 모델 설정 (v7_titan) — TITAN RTX x2 24GB, 384×384 해상도 + ViT-Base 인코더 + SS2D capacity 증대

v7 (320×320, ViT-Small, BS=16) 대비:
  - IMAGE_SIZE 320→384 (원본 ETER-Net 해상도)
  - ViT 인코더: dim 384→768, layers 6→12, heads 6→12, MLP 1536→3072 (ViT-Base)
  - SS2D: d_state 16→32, d_inner 64→128, out_ch 20→32
  - BATCH_SIZE: smoke 결과로 결정 (시작값 4, OOM 시 2→1 fallback)
  - 그 외 (DC block, dropout, weight_decay, EarlyStop, augment) 는 v7 동일
"""

import os

PATH_FOLDER = 'logs/SS2D_ViT_R4_brain384_v7_titan/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 데이터/입출력 크기 (v7 320 → v7_titan 384) ──
IMAGE_SIZE = (384, 384)
PATCH_SIZE = (16, 16)             # 384/16 = 24, num_patches = 576
INPUT_CHANNELS = 32

# ── 학습 설정 ──
BATCH_SIZE = 6                    # 2026-05-27 BS=8→6 (BS=8 single backward 중 OOM merge_norm 2.25GiB, 6 안전선)
NUM_EPOCHS = 50                   # 2026-05-26 200→50 (발열 운영 기간 단축)
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 3e-5
LAMBDA_SSIM_PER_PIXEL = 1.0

# ── 일반화 (v5~ 레시피) ──
DROPOUT = 0.2
TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── EarlyStop / val 주기 ──
# 2026-05-31: val 매 2 epoch 기록 (epoch 10 중간 점검용 granularity: ep 2/4/6/8/10 = 5점). NUM_EPOCHS=50 유지(T_max=50, ETER 와 동일 LR곡선).
# patience 50: VAL_EVERY=2 면 50ep 동안 val-check 가 25회뿐 → early-stop 은 50ep 내 사실상 비활성(ETER 와 동일, ETER 도 50ep 완주).
#   epoch10 점검 후 계속/중단은 수동 결정.
EARLYSTOP_PATIENCE = 50
VAL_EVERY_N_EPOCHS = 2

# ── v7_titan: scratch 학습 (2026-05-31) — ETER(scratch·단일GPU·50ep) 와 공정 비교 위해 RESUME_CKPT=None.
#    중단 내성은 main_train 의 true-resume(ss2d_vit_last.pt full-state) 가 담당하므로 여기는 항상 None.
#    (과거 epoch_10 weights-only warm-start 는 DDP head-start + LR 불연속 confounder 였음 → _ddp_archive 로 격리) ──
RESUME_CKPT = None

# ── DC block (v4~ 동일) ──
DC_K_SCALE_RATIO = 100.0
DC_INIT_ALPHA = 1.0

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

# ── DataLoader ──
NUM_WORKERS_TRAIN = 16
NUM_WORKERS_VAL   = 4
PREFETCH_FACTOR   = 4

# ── 인코더 (ViT-Base 복원) ──
NUM_VIT_ENCODER_HIDDEN = 768
NUM_VIT_ENCODER_LAYER  = 12
NUM_VIT_ENCODER_MLP_SIZE = 3072
NUM_VIT_ENCODER_HEAD   = 12

# ── SS2D (capacity 증대) ──
NUM_SS2D_D_INNER = 128
NUM_SS2D_D_STATE = 32
NUM_SS2D_OUT_CH  = 32

# ── 디코더 (v7 동일, dim 512 유지) ──
NUM_VIT_DECODER_HEAD              = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN    = 2048
NUM_VIT_DECODER_DIM               = 512
NUM_VIT_DECODER_DIM_HEAD          = 64
NUM_VIT_DECODER_DEPTH             = 6
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 64
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 8
