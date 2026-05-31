"""SS2D-ViT 설정 v6_3 — sharpness ablation (regularization 완화)

가설: v6 의 dropout=0.2 / weight_decay=3e-5 가 mean-prediction blurring 의 원인.
loss 자체는 v6 (L1 + (1-SSIM)) 그대로 두고 regularization 만 v4 수준 (dropout=0.1,
WD=1e-5) 으로 낮춰 sharpness 회복 여부 검증.

v6_2 (gradient loss λ_grad=3) 와 직교 실험. v6_2 가 loss 측에서 sharp 강제라면
v6_3 는 regularization 측에서 sharp 허용.

v6 best 부터 50ep fine-tune, LR 5e-5 → 5e-7 cosine.
"""
import os

PATH_FOLDER = 'logs/SS2D_ViT_R4_brain320_v6_3/'
PATH_FOLDER = './' + PATH_FOLDER
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

# ── 입출력 (v6 동일) ──
IMAGE_SIZE = (320, 320)
PATCH_SIZE = (16, 16)
INPUT_CHANNELS = 32

# ── 학습 (v6_2 동일 fine-tune 레짐, loss 만 v6 그대로) ──
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE_ADAM = 5e-5
LR_COSINE_MIN = 5e-7
LAMBDA_REGULAR_PER_PIXEL = 1e-5     # v6 3e-5 → v6_3 1e-5 (v4 수준 복귀)
LAMBDA_SSIM_PER_PIXEL = 1.0
LAMBDA_GRAD_PER_PIXEL = 0.0         # v6_3 은 gradient loss 사용 안 함 (v6_2 와 직교)

# ── 일반화 (sharpness 완화: dropout↓) ──
DROPOUT = 0.1                        # v6 0.2 → v6_3 0.1 (v4 수준 복귀)
TRAIN_AUGMENT = True                 # flip aug 유지
TRAIN_AUGMENT_FLIP_P = 0.5

# ── EarlyStop (v6_2 동일) ──
EARLYSTOP_PATIENCE = 5
VAL_EVERY_N_EPOCHS = 5

# ── Resume from v6 best ──
RESUME_CKPT = './logs/SS2D_ViT_R4_brain320_v6/ss2d_vit_best.pt'

# ── DC block (v6 동일) ──
DC_K_SCALE_RATIO = 100.0
DC_INIT_ALPHA = 1.0

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
