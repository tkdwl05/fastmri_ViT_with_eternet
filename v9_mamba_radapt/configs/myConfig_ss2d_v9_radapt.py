"""
v9 radapt (R 대응) — 설정.

언리시드 백본(SS2DStackV9)에 오퍼레이터식 R-일반화 요소를 얹은 변형:
  - 마스크(측정연산자) 명시 조건화 (MASK_CONDITION)
  - Data Consistency 측정 앵커 (DC_*)
  - multi-AR 학습 (AR_CHOICES 로 배치별 R 랜덤)

목표는 R4 절대품질이 아니라 **R∈{2,4,6,8} 전역에서 단일-R 모델보다 완만한 하락**(상대 robustness).
백본 용량은 언리시드와 동일하게 두어 "같은 Mamba 를 R-대응화" 효과를 격리한다.
val 은 R4 고정(best-ckpt 선택 기준을 언리시드/v8 과 비교 가능하게); R-sweep 은 별도 post-hoc 평가.
"""

import os

# ── 데이터/입출력 ──
IMAGE_SIZE = (384, 384)
INPUT_CHANNELS = 32
N_COIL = 16

# ── SS2D 백본 (언리시드와 동일 용량 + 동일 다운샘플) ──
SS2D_D_INNER   = 256
SS2D_D_STATE   = 32
SS2D_N_BLOCKS  = 3
SS2D_OUT_CH    = 64
SS2D_DROPOUT   = 0.05
SS2D_USE_CHECKPOINT = False
SS2D_DOWNSAMPLE = 3            # SSM 스캔을 128² coarse grid 에서 (언리시드와 동일 백본)

# ── R-대응 요소 ──
MASK_CONDITION = True                  # sampling mask 채널 concat (측정연산자 명시)
AR_CHOICES     = (2, 3, 4, 5, 6, 8)    # 학습 시 배치별 랜덤 R (multi-AR)
VAL_ACCELERATION = 4                   # val/best-ckpt 기준 R (R-sweep 은 별도)

# ── DC block ──
DC_K_SCALE_RATIO = 100.0
DC_INIT_ALPHA    = 1.0
DC_ALPHA_MIN     = 0.0                 # α clamp[0,1]: overshoot(→fp16 forward NaN) 물리 차단 (v8 진단)
DC_ALPHA_MAX     = 1.0
# DC-증폭 gradient 의 U-Net fp16 backward overflow 완화: GradScaler init_scale 낮춤
# (v8 진단: 기본 65536 은 과도 → scale≤8192 유한 확정). growth 는 그대로 두되 α clamp+NaN-skip 이 보강.
GRADSCALER_INIT_SCALE = 8192.0
MAX_CONSEC_SKIP = 300

# ── DFU (교수님 원본 U-Net) ──
UNET_DEPTH = 5
UNET_WF    = 6

# ── 학습 ──
BATCH_SIZE = 8                 # ds=3 스모크 기준. 스모크(SMOKE_BS)가 덮어씀
ACCUM_STEPS = 1
NUM_EPOCHS = 80                # ds=3 이 v8 보다 빠르므로 50→80 상향 (unleashed 와 정합)
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 3e-5    # weight_decay (Mamba A_log/D/dt_bias 는 no-WD 그룹 제외)
LAMBDA_SSIM_PER_PIXEL = 1.0

TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── EarlyStop / val ──
EARLYSTOP_PATIENCE = 40        # VAL_EVERY=2 → 80ep 무개선 → 사실상 80ep 완주
VAL_EVERY_N_EPOCHS = 2

# ── DataLoader ──
NUM_WORKERS_TRAIN = 16
NUM_WORKERS_VAL   = 4
PREFETCH_FACTOR   = 4

# ── Composite metric ──
COMPOSITE_W_SSIM = 0.5
COMPOSITE_W_PSNR = 0.3
COMPOSITE_W_NMSE = 0.2
PSNR_NORM = 40.0

# ── ckpt ──
RUN_NAME = 'PureETER_SS2D_V9_radapt_multiAR_brain384'
CKPT_PREFIX = 'ss2d_v9_radapt'


def path_folder() -> str:
    p = f"./logs/{RUN_NAME}/"
    os.makedirs(p, exist_ok=True)
    return p
