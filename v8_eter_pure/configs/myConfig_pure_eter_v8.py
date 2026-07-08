"""
v8 Pure ETER-Net 2×2 ablation — 공유 설정 (단일 config, 4런 공통).

4런 = {GRU, SS2D} × {no DC, DC}. 단일 trainer(main_train_pure_v8.py)가
환경변수 SEQ_MODEL(gru|ss2d) + USE_DC(0|1) 로 4런을 구동한다.
"everything else identical" 을 구조적으로 강제하기 위해 모든 하이퍼파라미터를
이 한 파일에서 공유한다.

교수님 ETER-net(no ViT) 충실: bi-RNN / SS2D → cat(aliased image) → U-Net DFU(depth=5, wf=6).
해상도 384, R=4, masked loss + composite metric (v7_titan 트랙과 정합).
"""

import os

# ── 데이터/입출력 ──
IMAGE_SIZE = (384, 384)
INPUT_CHANNELS = 32
N_COIL = 16

# ── 시퀀스 모델 공통 ──
# GRU hidden (원본 ETER-net model3 config "원본"=10; canonical model.py=12). 양 arm 동일 H2 사용.
N_HIDDEN_LRNN_1 = 10
N_HIDDEN_LRNN_2 = 10            # SS2D out_ch = 2*이 값 으로 강제 (= GRU out_v 채널)
# SS2D 내부 용량 (out_ch 는 위에서 강제되므로 자유 파라미터는 d_inner/d_state 뿐)
SS2D_D_INNER = 128
SS2D_D_STATE = 16

# ── DFU (교수님 원본 U-Net) ──
UNET_DEPTH = 5                  # 원본 model.py N_UNET_DEPTH=5
UNET_WF    = 6

# ── 학습 (4런 동일) ──
BATCH_SIZE = 2                  # smoke 가 SMOKE_BS 로 덮어씀 (GRU arm 이 구속조건)
ACCUM_STEPS = 1                 # effective batch 일치용 (4런 동일값). eff_bs = BATCH_SIZE * ACCUM_STEPS
NUM_EPOCHS = 50
LEARNING_RATE_ADAM = 2e-4
LAMBDA_REGULAR_PER_PIXEL = 3e-5
LAMBDA_SSIM_PER_PIXEL = 1.0

TRAIN_AUGMENT = True
TRAIN_AUGMENT_FLIP_P = 0.5

# ── EarlyStop / val ──
EARLYSTOP_PATIENCE = 50         # VAL_EVERY=2 → 25 val-check 뿐 → 사실상 50ep 완주
VAL_EVERY_N_EPOCHS = 2

RESUME_CKPT = None              # 중단내성은 trainer 의 full-state *_last.pt 가 담당

# ── DC block (DC arm 전용) ──
DC_K_SCALE_RATIO = 100.0
DC_INIT_ALPHA = 1.0

# ── DataLoader ──
NUM_WORKERS_TRAIN = 16
NUM_WORKERS_VAL   = 4
PREFETCH_FACTOR   = 4

# ── Composite metric (best / EarlyStop) ──
COMPOSITE_W_SSIM = 0.5
COMPOSITE_W_PSNR = 0.3
COMPOSITE_W_NMSE = 0.2
PSNR_NORM = 40.0


# ── 4런 ckpt 디렉토리 / prefix (SEQ_MODEL + USE_DC 로 선택) ──
def path_folder(seq_model: str, use_dc: bool) -> str:
    tag = 'DC' if use_dc else 'noDC'
    name = f"PureETER_{seq_model.upper()}_{tag}_R4_brain384_v8"
    p = f"./logs/{name}/"
    os.makedirs(p, exist_ok=True)
    return p


def ckpt_prefix(seq_model: str) -> str:
    return f"pure_{seq_model.lower()}"     # → pure_gru_best.pt / pure_ss2d_last.pt 등
