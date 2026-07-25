"""
v9 언리시드 Mamba(SS2D) — 설정.

v8_eter_pure 의 통제비교(GRU vs SS2D)와 **분리된 신규 트랙**. 재구성 품질 최대화가 목표라
v8 SS2D 의 조임(out_ch=20 강제, d_state=16, 단일 블록, 게이팅 없음)을 전부 해제한다.

데이터/파이프라인(384·R4·brain-mask·masked loss·composite)은 v8 과 정합해 절대 비교 가능하게 유지.
비교 레퍼런스: v8 SS2D best composite 0.9200 / ssim_m 0.9140, v8 GRU 0.9182 / 0.9126 (동일 val).
"""

import os

# ── 데이터/입출력 (v8 정합) ──
IMAGE_SIZE = (384, 384)
INPUT_CHANNELS = 32
N_COIL = 16

# ── SS2D 언리시드 용량 (v8 대비 확대) ──
SS2D_D_INNER   = 256      # v8 128 → 256 (SSM 폭 2×; GRU 668M 대비 여전히 소형)
SS2D_D_STATE   = 32       # v8 16 → 32 (Mamba 핵심 용량, v7_titan 수준)
SS2D_N_BLOCKS  = 3        # v8 (1) → 3 (residual 게이팅 스택 = 깊이)
SS2D_OUT_CH    = 64       # v8 20(강제) → 64 (병목 해제; 짝수 → n_hidden=(64+32)/2=48)
SS2D_DROPOUT   = 0.05     # v8 0 → 0.05 (확대 용량 정규화)
SS2D_USE_CHECKPOINT = False   # fp16 스캔으로 불필요 (메모리 여유)
# 다운샘플 front-end: SSM 스캔을 384/ds grid 에서 → 스캔 cost ∝ 면적 ds²× 절감.
# 스모크 실측: ds=3(128²) 풀용량 2.53 h/ep·BS8·16.7GB (v8 2.78 대비 빠름). SSM=전역문맥,
# U-Net=풀해상도 디테일 분업(VMamba 정석). stem 은 384²에서 k-space 전체를 먼저 봄(정보손실 없음).
SS2D_DOWNSAMPLE = 3

# ── DFU (교수님 원본 U-Net; v8 과 동일) ──
UNET_DEPTH = 5
UNET_WF    = 6

# ── 학습 (v8 정합) ──
BATCH_SIZE = 8                 # ds=3 스모크: BS8 @ 16.7GB. 스모크(SMOKE_BS)가 덮어씀
ACCUM_STEPS = 1                # eff_bs = BATCH_SIZE * ACCUM_STEPS
NUM_EPOCHS = 80                # ds=3 이 v8(2.78 h/ep)보다 빠르므로 50→80 상향 (사용자 요청)
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 3e-5   # weight_decay (Mamba A_log/D/dt_bias 는 no-WD 그룹으로 제외)
LAMBDA_SSIM_PER_PIXEL = 1.0

TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── EarlyStop / val (v8 정합) ──
EARLYSTOP_PATIENCE = 40        # VAL_EVERY=2 → 40 val-check(=80ep 무개선) → 사실상 80ep 완주
VAL_EVERY_N_EPOCHS = 2

RESUME_CKPT = None             # trainer full-state *_last.pt 가 중단내성 담당

# ── DataLoader (v8 정합) ──
NUM_WORKERS_TRAIN = 16
NUM_WORKERS_VAL   = 4
PREFETCH_FACTOR   = 4

# ── Composite metric (best / EarlyStop; v8 과 동일 정의) ──
COMPOSITE_W_SSIM = 0.5
COMPOSITE_W_PSNR = 0.3
COMPOSITE_W_NMSE = 0.2
PSNR_NORM = 40.0

# ── ckpt 디렉토리 / prefix ──
RUN_NAME = 'PureETER_SS2D_V9_unleashed_R4_brain384'
CKPT_PREFIX = 'ss2d_v9'


def path_folder() -> str:
    p = f"./logs/{RUN_NAME}/"
    os.makedirs(p, exist_ok=True)
    return p
