"""
Mamba-ETER v8fix 트레이너 (2026-10-12) — pod3 ETER-net(GRU) 런의 트레이너 main_train_pure_v8fix_ampguard.py
(origin/runpod/v8fix-pod3, md5 33cfc82d)를 복사해 **모델만** Mamba-ETER(models/pure_eternet/u_pure_eternet_mamba1d.py,
PureETER_Mamba1D — PureETER_GRU 의 데이터 흐름 그대로, nn.GRU 자리에 양방향 Mamba)로 바꾼 파일. 사용자 결정(10-12): pod2 에서
SEED 1·50 epoch, mamba_ssm 기본 초기화(outmatch 없음), Mamba h·v 두 단계 모두 fp32(fp32hv).

ETER-net 런과 같은 것: 레시피 DL·손실·optimizer 규칙·데이터 흐름(SEED+1000×epoch)·U-Net 초기값 복사(SAME_UNET_INIT)·검증·
기능 시험·건강 감시·체크포인트·AMP 보호 장치(GRAD-retry·배율 하한)·exit 코드.
바뀐 것:
  - 런 이름·체크포인트 접두사 = 'mamba1d' (logs/PureETER_MAMBA1D_noDC_R4_brain384_v8fix{RUN_SUFFIX}/, pure_mamba1d_*.pt).
    smoke_recipe_fix(R.*) 에는 라벨 'gru'(HARNESS_SEQ)를 넘긴다 — PureETER_Mamba1D 가 PureETER_GRU 의 _seq·forward 를 상속하고
    슬롯 이름 gru_h/gru_v 를 유지하므로 R 의 build_groups('gru')·run_validation('gru') 가 그대로 맞는다. 'mamba1d' 를 넘기면
    R.run_validation 이 시퀀스 출력 대신 0 을 넣고 시퀀스 시험을 조용히 빼므로(R.unet_input) 이름과 라벨을 분리했다.
  - 정밀도: gru_precision 대신 모델 생성 인자 fp32_stages=('h','v') (MAMBA_ETER_FP32, 기본·유일 허용값 hv). 그 단계 전체가
    autocast 밖 fp32, U-Net 은 다른 모델과 같은 fp16 autocast. 기준 ETER-net 은 gru_h 만 fp32(gru_v fp16) — 비대칭은 기록에 남긴다.
  - 시작 점검(어긋나면 exit 2 — 런처는 재시도하지 않음): Mamba fast path(causal_conv1d·selective_scan CUDA), 모델 계약
    (_seq·forward 가 PureETER_GRU 의 것, nn.GRU 없음, fp32 단계, 파라미터 수 637,150,928 / 668,210,949), 런처가 준
    EXPECT_SBAR·EXPECT_UNET_CHECKSUM(pod 데이터의 s̄ 325.2484691441059·U-Net 초기값 checksum 5.9212156172e+02 — ETER-net·U-Net
    단독 런과 같아야 함).
  - 재개 일관성 검사에 모델 클래스·하네스 라벨·fp32 단계·Mamba 차원·초기화·디버그 설정·모델 파일과 smoke_recipe_fix.py 의
    sha256 을 더했다(키가 없어도 거부).
  - 검증 epoch 마다 'MAMBA-DIAG' 줄(기록 전용 — 학습에 영향 없음): 단계·방향별 out_proj 노름 비(초기 대비), A_log·dt_proj.bias
    상대 변화, 시퀀스 하위 텐서별 Adam 2차 모멘트가 eps 아래인 비율(4000 step 시험에서 Δ 경로가 eps 에 묶였던 한계 감시).

아래는 원 트레이너(ampguard)의 설명 그대로다.

v8fix + AMP 보호 장치 트레이너 (2026-10-10) — main_train_pure_v8fix.py(RunPod 판, NaN-retry 포함)와 학습 내용은 같고
fp16 GradScaler 보호 장치 두 가지만 더했다(아래 AMP_GRAD_RETRY·AMP_MIN_SCALE 주석). 실행 중인 다른 런의 트레이너를 고치지 않으려고
새 파일로 분리했다. 런 폴더·wandb 이름 규칙은 원 트레이너와 같다.
10-10 추가: SEQ_MODEL=gru 이면 k-space 쪽 GRU(gru_h)를 fp32 로 계산한다(GRU_FP32_MODULES, 기본 gru_h) — 보호 장치만으로는
같은 지점(epoch 1 batch ≈3,535)에서 다시 무너졌고, 그 상태에서 gru_h 의 fp16 기울기가 NaN 이거나 최대 40 배 틀렸다(gru_precision.py 주석).

v8fix — 학습 붕괴 수정 레시피(DL) 트레이너 (2026-10-08).

배경: docs/v8_training_collapse_audit_2026-10-07.md — 기존 트레이너(main_train_pure_v8.py)의 Adam 결합형 L2(weight_decay 3e-5,
전 파라미터) + 전역 clip 1.0 + ×1e6 강도 단위 L1 조합으로 U-Net 깊은 층·시퀀스 모듈 가중치가 학습 초기(≈step 300)에 0 으로
소멸했다. 레시피 수정 스모크(v8_eter_pure/smoke_recipe_fix.py, results/smoke_recipe_fix/summary.md, 10-08 완료)에서
원 손실 균형을 유지한 조건 DL 만이 "시퀀스 출력 사용" 사전 기준을 통과했다(GRU, shuffle 하락 0.0148, 62/64).
기존 트레이너는 재현용으로 무수정 유지하고, 이 파일이 DL 레시피를 50 epoch 일정으로 적용한다.

레시피 DL (smoke_recipe_fix.CONDS['DL'] 과 같은 정의 — 아래 assert 로 묶음; 함수도 그 파일에서 import 해 단일 출처):
  - AdamW 분리형 weight decay 3e-5 (config LAMBDA_REGULAR_PER_PIXEL) + no-decay 그룹(bias·norm·SS2D A_log/D·_no_weight_decay),
    eps 1e-8, LR 2e-4 (config), cosine T_max = 50 epoch 전체 opt-step, eta_min 1e-6 (기존과 같음).
  - clip_grad_norm_ max_norm 1e3 — 폭주 방지 안전장치일 뿐(스모크 DL 에서 clip 계수 항상 1.0).
  - 샘플별 강도 정규화: s = zero-filled RSS(data_img) 의 99 백분위수. 영상·label /s, k-space ×(100/s)
    (로더가 data_img = 100·ifft2c(data) 이므로 선형 관계 유지). 검증·평가는 출력×s 로 원 단위에서 기존과 같은 공식.
  - λ_SSIM = config λ(1.0) / s̄ (s̄ = train 64 슬라이스 s 평균, smoke_recipe_fix.estimate_mean_scale — SEED 1 에서 354.35):
    원 스케일의 L1 : SSIM 항 균형(≈69:1)을 정규화 단위에서 재현. 처음 실행 때 계산해 last.pt·recipe.json 에 저장, 재개 때 재사용.
  - 그 밖(데이터·augment·모델·AMP·NaN-skip·BS)은 기존 트레이너와 같다. 초기화 직전 seed_all(SEED) 로 시퀀스 모듈은 스모크 DL
    job 과 같은 초기값에서 시작한다.

기존 트레이너와 다른 점 (레시피 외):
  - U-Net 초기값 통일(SAME_UNET_INIT=1 기본): seed_all(SEED) 직후 만든 U-Net 단독 모델의 U-Net 가중치를 GRU·SS2D 모델의 U-Net 에
    복사한다. 기존 런·스모크는 wrapper 가 시퀀스 모듈을 먼저 만들어 같은 SEED 에서도 U-Net 초기값이 모델마다 달랐다.
  - 데이터 흐름 = (SEED, epoch) 의 함수: 워커를 epoch 마다 새로 만들고 generator 를 SEED+1000×epoch 로 다시 시드한다.
    epoch 1 은 기존·스모크와 같은 스트림이고, 계획 정지나 장애로 재개해도 끊김 없는 런과 같은 데이터를 본다
    (기존 트레이너는 재개 직후 epoch 가 1 epoch 의 셔플·마스크 offset·flip 을 반복했고, RNG 상태 복원도 실패했다 — 둘 다 수정).
  - 재개 일관성 검사: T_max·BS·epoch 수·SEED·레시피 코드 판(smoke SCRIPT_VERSION·max_norm·eps·K_RATIO)이 정지 때와 다르면
    exit 2 로 거부(런처는 재시도하지 않음). 정지 중에는 smoke_recipe_fix.py 를 고치지 말 것.
  - SEQ_MODEL ∈ {unet, gru, ss2d} 만 (스모크로 검증한 모델). 런 폴더 logs/PureETER_{SEQ}_noDC_R4_brain384_v8fix[_s{SEED}]/.
  - best 체크포인트 = 검증 SSIM 최댓값 (composite 미사용). log.txt 에 composite 열 없음.
  - STOP_AFTER_EPOCH=N: N epoch 를 끝내고(검증·저장 포함) 'PAUSED_ep{N}' 표시 파일을 쓰고 정상 종료 — 50 epoch 일정 그대로
    나중에 last.pt 에서 이어 학습한다(LR 스케줄·데이터 흐름 연속). 런처로 재개할 때는 STOP_AFTER_EPOCH=50(또는 더 큰 중간값).
    N epoch 에는 VAL_EVERY 와 무관하게 검증한다. 완주 시 'DONE' 표시 파일.
  - 건강 감시: HEALTH_EVERY_STEPS(기본 500) step 마다 + epoch 끝에 그룹별 |w|<1e-6 비율(seq / unet_deep / unet_all —
    smoke_recipe_fix.build_groups) 을 log.txt·wandb 에 기록. seq 또는 unet_deep 이 0.5 이상이면 ALERT_COLLAPSE 파일과
    {prefix}_alert_state.pt 를 쓰고 exit 3 (런처는 3 이면 재시도하지 않고 큐를 멈춘다).
  - log.txt 와 probe_ep{N}.json 이 기록의 기준이다(장애 재개 뒤 다시 돈 step 은 wandb 가 단조성 때문에 버린다).
  - 검증 epoch 마다 기능 시험(smoke_recipe_fix.run_validation, val 에 고르게 퍼진 PROBE_VAL_SLICES=64 슬라이스):
    shuffle_cross / deep4 등의 SSIM 하락·부호 검정·사전 기준 판정을 log.txt 'PROBE' 줄과 probe_ep{N}.json 에 기록
    (체크포인트 저장 뒤에 실행, 실패해도 학습은 계속).

환경 변수: SEQ_MODEL, SEED, RUN_SUFFIX, SMOKE_BS, SANITY_NUM_EPOCHS(=일정 전체 epoch, 기본 config 50),
  SANITY_VAL_EVERY_N_EPOCHS, STOP_AFTER_EPOCH, HEALTH_EVERY_STEPS, PROBE_VAL_SLICES, WANDB_RUN_TAG, SAME_UNET_INIT,
  PREFETCH_FACTOR·NUM_WORKERS_VAL(데이터 로더 자원 — 결과 무관), WANDB_PROJECT(wandb 프로젝트, 기본 ViT-MRI-Recon — RunPod 런은 fastMRI-research (entity tkdwl05-hongik-university)),
  V8FIX_LOG_ROOT(기본 <repo>/logs — 디버그 런을 scratch 로 보내는 용도),
  VAL_MAX_RETRY(기본 3)·TRAIN_NAN_RETRY(기본 1) — 일시적 비유한 값 재계산 횟수(아래 run_val·학습 step 주석, 10-09),
  AMP_GRAD_RETRY(기본 1)·AMP_MIN_SCALE(기본 1024) — AMP 보호 장치(10-10, 이 파일에서만),
  GRU_FP32_MODULES(SEQ_MODEL=gru 일 때 기본 gru_h, 쉼표 구분, 빈 값 = 전부 fp16) — fp32 로 계산할 GRU 모듈(10-10),
  디버그 전용: DEBUG_MAX_STEPS(epoch 당 step 상한), DEBUG_VAL_SAMPLES(검증 앞쪽 N 슬라이스만).
Mamba-ETER 판(이 파일)에서: SEQ_MODEL 은 비우거나 mamba1d 만, GRU_FP32_MODULES 는 비워야 함, MAMBA_ETER_FP32(hv 만),
  EXPECT_SBAR·EXPECT_UNET_CHECKSUM(비우면 확인 생략 — 로컬 데이터 점검용; pod 런처는 반드시 지정).
"""

import os
import sys
import json
import time
import datetime
import random
import pytz

import torch
import numpy as np
import wandb
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as compare_ssim

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'tools'))

from myConfig_pure_eter_v8 import *           # noqa: F401,F403  (공유 하이퍼파라미터 — 기존 트레이너와 같은 config)
from u_choh_SSIM import SSIM
# TF32 끔(기본): 로컬 TITAN RTX(Turing)에는 TF32 가 없다 — Ampere 이상 GPU(RunPod 3090 등)에서도 같은 fp32 정밀도로 맞춘다(10-09).
if os.environ.get('ALLOW_TF32', '0') != '1':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
from dataloader_h5_v5 import FastMRI_H5_Dataloader
from torch.utils.data import DataLoader, Subset, default_collate
from check_recon_env import check_env_for_model
import smoke_recipe_fix as R                   # 레시피 함수의 단일 출처 (import 부작용: sys.path·PYTORCH_CUDA_ALLOC_CONF 기본값뿐)
import hashlib
import torch.nn as nn
from u_pure_eternet_gru import PureETER_GRU                               # 데이터 흐름 계약 확인용 (_seq·forward)
from u_pure_eternet_mamba1d import PureETER_Mamba1D, fp32_stages_from_str  # Mamba-ETER (mamba_ssm 2.2.2 import)
import mamba_ssm.modules.mamba_simple as _MS                              # fast path 함수 존재 확인용

# ── 레시피 DL 고정 (스모크 정의와 어긋나면 중단) ──
RECIPE = 'DL'
_SPEC = R.CONDS[RECIPE]
assert _SPEC['decoupled'] and _SPEC['normalize'] and _SPEC['lam'] == 'matched', _SPEC
MAX_NORM = float(_SPEC['max_norm'])            # 1e3
ADAM_EPS = float(_SPEC['adam_eps'])            # 1e-8

# ── 런 선택 (Mamba-ETER 전용) ──
# 시작 때 거부할 설정 오류는 모아 두었다가 main() 에서 exit 2 로 끝낸다 (assert 는 rc 1 → 런처가 200 번 재시도).
_STARTUP_BAD = []
SEQ_MODEL   = 'mamba1d'      # 런 이름·체크포인트 접두사·기록용
HARNESS_SEQ = 'gru'          # smoke_recipe_fix(R.*) 에 넘기는 라벨 — 모듈 docstring 참고 (이름과 분리)
HAS_SEQ     = True
_env_seq = os.environ.get('SEQ_MODEL', '').strip().lower()
if _env_seq not in ('', SEQ_MODEL):
    _STARTUP_BAD.append(f'SEQ_MODEL={_env_seq!r} — 이 트레이너는 mamba1d 전용')
if os.environ.get('GRU_FP32_MODULES', '').strip():
    _STARTUP_BAD.append('GRU_FP32_MODULES 가 설정됨 — Mamba-ETER 의 정밀도는 MAMBA_ETER_FP32 로만 정한다')
MAMBA_FP32_ENV = os.environ.get('MAMBA_ETER_FP32', 'hv')
try:
    MAMBA_FP32_STAGES = fp32_stages_from_str(MAMBA_FP32_ENV)
except ValueError as _e:
    MAMBA_FP32_STAGES = ()
    _STARTUP_BAD.append(str(_e))
if MAMBA_FP32_STAGES != ('h', 'v'):
    _STARTUP_BAD.append(f'MAMBA_ETER_FP32={MAMBA_FP32_ENV!r} → {MAMBA_FP32_STAGES} — 이 런은 hv(두 단계 fp32)만 (사용자 결정 10-12)')
MAMBA_INIT = 'mamba_ssm default (outmatch 없음)'
EXPECT_SEQ_PARAMS, EXPECT_TOTAL_PARAMS = 637_150_928, 668_210_949      # u_pure_eternet_mamba1d.py docstring


def _float_env(name):
    v = os.environ.get(name, '').strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        _STARTUP_BAD.append(f'{name}={v!r} 는 실수가 아님')
        return None


EXPECT_SBAR          = _float_env('EXPECT_SBAR')            # pod 데이터: 325.2484691441059 (ETER-net·SS2D·U-Net 단독 런 recipe.json)
EXPECT_UNET_CHECKSUM = _float_env('EXPECT_UNET_CHECKSUM')   # SEED 1: 5.9212156172e+02 (세 런의 SCRATCH START 줄)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _rel_close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-30)


def _code_hash(path):
    """동작을 정하는 코드의 지문 — docstring·주석·공백을 뺀 AST 의 sha256 (설명만 고친 파일로 재개가 거부되지 않게)."""
    import ast
    tree = ast.parse(open(path, encoding='utf-8').read())
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], 'value', None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()

# ── override ──
NUM_EPOCHS         = int(os.environ.get('SANITY_NUM_EPOCHS', NUM_EPOCHS))
VAL_EVERY_N_EPOCHS = int(os.environ.get('SANITY_VAL_EVERY_N_EPOCHS', VAL_EVERY_N_EPOCHS))
BATCH_SIZE         = int(os.environ.get('SMOKE_BS', BATCH_SIZE))
ACCUM_STEPS        = int(os.environ.get('ACCUM_STEPS', ACCUM_STEPS))
STOP_AFTER_EPOCH   = int(os.environ.get('STOP_AFTER_EPOCH', '0'))
HEALTH_EVERY_STEPS = int(os.environ.get('HEALTH_EVERY_STEPS', '500'))
PROBE_VAL_SLICES   = int(os.environ.get('PROBE_VAL_SLICES', '64'))
DEBUG_MAX_STEPS    = int(os.environ.get('DEBUG_MAX_STEPS', '0'))
DEBUG_VAL_SAMPLES  = int(os.environ.get('DEBUG_VAL_SAMPLES', '0'))
SAME_UNET_INIT     = os.environ.get('SAME_UNET_INIT', '1') == '1'
# 데이터 로더 자원 (결과 무관 — 워커가 맡는 샘플·난수는 prefetch·val 워커 수와 무관). RunPod(RAM 125 GB)에서 train prefetch 4 는
# 공유·고정 메모리가 한도에 닿아 OOM → 2 로 줄인다(10-09).
PREFETCH_FACTOR    = int(os.environ.get('PREFETCH_FACTOR', PREFETCH_FACTOR))
NUM_WORKERS_VAL    = int(os.environ.get('NUM_WORKERS_VAL', NUM_WORKERS_VAL))
# 일시적 비유한 값 재계산 (10-09): RunPod RTX 3090 Pod 에서 학습 배치 약 0.2% 의 loss 와 검증 배치 일부의 지표가 NaN/-inf 로 나왔다.
# 같은 슬라이스를 다시 읽어 다시 계산하면 매번 유한했고(검증 95 배치 × 5 회, GPU 결정적 반복 3000 회·전송 1 TB·VRAM 시험, 읽기 3.2 만 회 모두 무오류),
# 학습 로그에는 데이터 로더 워커의 overflow 경고(dataloader_h5_v5.py:234·235 의 ×1e4·×1e6, FFT 비유한)가 있었다 → 학습 중 Pod 의
# CPU 쪽에서 가끔 데이터가 깨진다(로컬 TITAN 은 16 epoch 동안 경고·NaN 0 건). epoch 2 검증 평균이 NaN 이 되어 best 체크포인트가
# 저장되지 않았다 → 비유한 검증 배치는 데이터셋에서 다시 읽어 다시 계산하고, 학습 배치는 한 번 다시 계산한 뒤 그래도 비유한이면 건너뛴다.
VAL_MAX_RETRY      = int(os.environ.get('VAL_MAX_RETRY', '3'))
TRAIN_NAN_RETRY    = int(os.environ.get('TRAIN_NAN_RETRY', '1'))
# AMP 보호 장치 (10-10): RunPod pod3 의 ETER-net(GRU) 런에서 epoch 1 batch ≈3,520~3,560 의 40 step 동안 기울기 비유한이 16 번 몰렸다.
# loss 는 유한했다(GRU 는 입력이 망가져도 게이트가 포화돼 출력이 유한하다) → 위 NaN-retry 가 작동하지 않았고, GradScaler 가 매번 배율을
# 절반으로 줄여 4096 → 0.0625 가 됐다. 배율은 2000 step 마다 2 배씩만 회복되므로 이후 fp16 기울기 대부분이 0 으로 사라져
# (GRU Adam 1 차 모멘트 ~1e-20) 학습이 무너졌다(학습 SSIM 0.73 → 0.1~0.3). 같은 때 같은 Pod 의 SS2D, pod2 U-Net, 로컬 점검 GRU 의
# 배율은 2048~131072 범위였다.
#   (1) AMP_GRAD_RETRY: 기울기에 비유한 값이 있으면 같은 배치를 다시 보내 다시 계산한다(GPU 쪽 일시 오류면 회복).
#   (2) AMP_MIN_SCALE: scaler.update() 뒤 배율이 이 값보다 작으면 이 값으로 되돌린다. 깨진 배치가 몰려도 배율이 무너지지 않고,
#       그 배치들은 GradScaler 가 평소처럼 건너뛴다. 1024 근거: L1 항의 화소당 기울기 ≈ 1/(뇌 마스크 화소 수) ≈ 3e-6 → ×1024 ≈ 3e-3 으로
#       fp16 정규 범위(≥ 6.1e-5) 안. 정상 런의 배율은 이보다 항상 커서 정상 학습에서는 작동하지 않는다.
#   기록: 사건마다 log.txt 'GRAD-retry'·'AMP-floor' 줄, epoch 줄에 amp_scale(epoch 끝)·amp_skip(GradScaler 가 건너뛴 step)·
#   grad_retry_total·grad_recovered_total·amp_floor_total.
#   시험 전용 AMP_TEST_INJECT_ONCE / AMP_TEST_INJECT_ALWAYS = "a-b"(epoch 안 배치 번호 범위): 첫 계산(ONCE) 또는 모든 계산(ALWAYS)의
#   기울기 한 원소를 NaN 으로 만들어 장치를 시험한다. 기본은 비어 있음 — 본 학습에서는 쓰지 않는다.
AMP_GRAD_RETRY     = int(os.environ.get('AMP_GRAD_RETRY', '1'))
AMP_MIN_SCALE      = float(os.environ.get('AMP_MIN_SCALE', '1024'))


def _batch_range(v):
    if not v:
        return set()
    a, _, b = v.partition('-')
    return set(range(int(a), int(b or a) + 1))


AMP_TEST_INJECT_ONCE   = _batch_range(os.environ.get('AMP_TEST_INJECT_ONCE', ''))
AMP_TEST_INJECT_ALWAYS = _batch_range(os.environ.get('AMP_TEST_INJECT_ALWAYS', ''))
assert ACCUM_STEPS == 1, 'v8fix 는 ACCUM_STEPS=1 만 (스모크와 같은 step 정의)'
NEAR_THR, ALERT_NEAR = R.NEAR_THR, R.TH_COLLAPSED_NEAR       # 1e-6, 0.5
EXIT_ALERT = 3
EXIT_REFUSE = 2

# ── 시드 (기존 트레이너 :59-66 과 같은 경로) ──
_SEED_ENV = os.environ.get('SEED')
SEED = int(_SEED_ENV) if _SEED_ENV is not None else None
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"[seed] SEED={SEED} — weights/shuffle/aug/mask-offset 고정 (fp16 atomics 로 비트단위 재현은 아님)")

_RUN_SUFFIX = os.environ.get('RUN_SUFFIX', f"_s{SEED}" if SEED is not None else '')
_RUN_NAME   = f"PureETER_{SEQ_MODEL.upper()}_noDC_R4_brain384_v8fix{_RUN_SUFFIX}"
_LOG_ROOT   = os.environ.get('V8FIX_LOG_ROOT', os.path.join(_PROJECT_ROOT, 'logs'))
PATH_FOLDER = os.path.join(_LOG_ROOT, _RUN_NAME)
os.makedirs(PATH_FOLDER, exist_ok=True)
PREFIX      = f"pure_{SEQ_MODEL}"
DATA_TRAIN  = os.path.join(_PROJECT_ROOT, 'fastMRI_data', 'multicoil_train')
DATA_VAL    = os.path.join(_PROJECT_ROOT, 'fastMRI_data', 'multicoil_val')
LOG_PATH    = os.path.join(PATH_FOLDER, 'log.txt')


def log_line(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def save_checkpoint_atomic(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json_atomic(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def skimage_ssim_batch_masked(pred, target, mask):
    """main_train_pure_v8.py:84-101 그대로 (배치 내 슬라이스 평균)."""
    p = pred.detach().float().cpu().numpy()
    t = target.detach().float().cpu().numpy()
    m = mask.detach().float().cpu().numpy()
    if p.ndim == 4:
        p, t, m = p[:, 0], t[:, 0], m[:, 0]
    m_bool = m > 0.5
    vals = []
    for i in range(p.shape[0]):
        if not m_bool[i].any():
            continue
        t_in = t[i][m_bool[i]]
        dr = float(t_in.max() - t_in.min())
        if dr <= 0:
            continue
        _, ssim_map = compare_ssim(t[i], p[i], data_range=dr, full=True)
        vals.append(float(ssim_map[m_bool[i]].mean()))
    return float(np.mean(vals)) if vals else 0.0


def _val_batch_metrics(model, sample, device, amp=True):
    """검증 한 배치의 (ssim, psnr, nmse, l1) — main_train_pure_v8.py:152-189 와 같은 지표 공식.
    매번 CPU 쪽 sample 에서 GPU 로 다시 보내 계산한다(재계산 때 GPU 쪽 사본을 재사용하지 않도록)."""
    data_in     = sample['data'].float().to(device)
    data_in_img = sample['data_img'].float().to(device)
    data_ref    = sample['label'].float().to(device)
    brain_mask  = sample['brain_mask'].float().to(device)
    mask        = sample['mask'].float().to(device)
    sens        = sample['sens'].float().to(device)
    s = R.per_sample_scale(data_in_img)
    x_ksp, x_img = R.normalize_inputs(data_in, data_in_img, s)

    with torch.amp.autocast('cuda', enabled=amp):
        out = model(x_img, x_ksp, mask, sens)

    out_f = out.float() * s
    ref_f = data_ref.float()
    m     = brain_mask
    m_sum = m.sum().clamp(min=1.0)
    diff_sq_sum = ((out_f - ref_f) ** 2 * m).sum()
    mse  = diff_sq_sum / m_sum
    ref_max_in_mask = (ref_f * m).max().clamp(min=1e-10)
    psnr = (20 * torch.log10(ref_max_in_mask / torch.sqrt(mse.clamp(min=1e-10)))).item()
    ref_sq_sum = (ref_f ** 2 * m).sum().clamp(min=1e-10)
    nmse = (diff_sq_sum / ref_sq_sum).item()
    ssim = skimage_ssim_batch_masked(out_f, ref_f, m)
    l1   = (((out_f - ref_f).abs() * m).sum() / m_sum).item()
    return ssim, psnr, nmse, l1


def run_val(model, val_loader, device):
    """main_train_pure_v8.py:152-189 와 같은 지표 공식 — 입력을 정규화해 forward 하고 출력×s 로 원 단위 label 과 비교.
    (composite 없음)
    지표 중 하나라도 비유한(NaN/±inf)인 배치는 그 배치를 데이터셋에서 다시 읽어(shuffle 없음 → batch bi = 인덱스
    [bi·bs, (bi+1)·bs)) 최대 VAL_MAX_RETRY 번 다시 계산하고, 그래도 비유한이면 fp32(autocast 끔)로 한 번 계산한다.
    그래도 비유한이면 그 배치를 평균에서 빼고 개수를 센다 (10-09, 위 VAL_MAX_RETRY 주석)."""
    model.eval()
    all_ssim, all_psnr, all_nmse, all_l1 = [], [], [], []
    n_retry = n_recovered = n_fp32 = n_excluded = 0
    val_bar = tqdm(val_loader, desc='  Val', leave=False, unit='batch')
    with torch.no_grad():
        for bi, sample in enumerate(val_bar):
            met = _val_batch_metrics(model, sample, device)
            if not np.all(np.isfinite(met)):
                bad0 = met
                amax0 = {k: float(sample[k].abs().max()) for k in ('data', 'data_img', 'label')}
                ds_v, bs_v = val_loader.dataset, val_loader.batch_size
                for _ in range(VAL_MAX_RETRY):
                    n_retry += 1
                    sample = default_collate([ds_v[j] for j in range(bi * bs_v, min((bi + 1) * bs_v, len(ds_v)))])
                    met = _val_batch_metrics(model, sample, device)
                    if np.all(np.isfinite(met)):
                        break
                how = 'retry'
                if not np.all(np.isfinite(met)):
                    n_fp32 += 1
                    met = _val_batch_metrics(model, sample, device, amp=False)
                    how = 'fp32'
                ok = bool(np.all(np.isfinite(met)))
                n_recovered += int(ok)
                log_line(f'VAL-nonfinite batch{bi} first={tuple(round(float(v), 4) for v in bad0)} input_absmax={amax0} '
                         f'{"recovered by " + how if ok else "EXCLUDED"} → {tuple(round(float(v), 4) for v in met)}')
                if not ok:
                    n_excluded += 1
                    continue
            ssim, psnr, nmse, l1 = met
            all_psnr.append(psnr); all_nmse.append(nmse); all_ssim.append(ssim); all_l1.append(l1)
            val_bar.set_postfix(SSIM=f'{ssim:.4f}', PSNR=f'{psnr:.2f}dB')
    model.train()
    mean = lambda v: float(np.mean(v)) if v else float('nan')   # noqa: E731
    return {'ssim': mean(all_ssim), 'psnr': mean(all_psnr), 'nmse': mean(all_nmse), 'l1': mean(all_l1),
            'n_batches': len(all_ssim), 'retry': n_retry, 'recovered': n_recovered, 'fp32': n_fp32,
            'excluded': n_excluded}


@torch.no_grad()
def near_dead_fracs(model, groups):
    """그룹별 |w| < NEAR_THR 비율 (큰 텐서는 32M 원소씩 — GPU 임시 메모리 상한)."""
    per = {}
    for n, p in model.named_parameters():
        cnt = 0
        for ch in p.detach().reshape(-1).split(R.STAT_CHUNK):
            cnt += int((ch.abs() < NEAR_THR).sum())
        per[n] = (cnt, p.numel())
    out = {}
    for g, names in groups.items():
        c = sum(per[n][0] for n in names)
        t = sum(per[n][1] for n in names)
        out[g] = c / max(t, 1)
    return out


def health_check(model, groups, epoch, global_step, where):
    nd = near_dead_fracs(model, groups)
    keys = [k for k in ('seq', 'unet_deep', 'unet_all') if k in nd]
    msg = (f'HEALTH ep{epoch + 1} step{global_step} ({where}) near_dead(|w|<{NEAR_THR:g}) '
           + ' '.join(f'{k}={100 * nd[k]:.2f}%' for k in keys))
    tqdm.write('  ' + msg)
    log_line(msg)
    try:
        wandb.log({f'health/near_dead_{k}': nd[k] for k in keys}, step=global_step)
    except Exception:
        pass
    bad = [k for k in ('seq', 'unet_deep') if k in nd and nd[k] >= ALERT_NEAR]
    if bad:
        alert = dict(time=datetime.datetime.now().isoformat(timespec='seconds'), epoch=epoch + 1,
                     global_step=global_step, where=where, near_dead=nd, threshold=ALERT_NEAR, groups=bad)
        write_json_atomic(alert, os.path.join(PATH_FOLDER, 'ALERT_COLLAPSE'))
        try:
            save_checkpoint_atomic(model.state_dict(), os.path.join(PATH_FOLDER, f'{PREFIX}_alert_state.pt'))
        except Exception as e:                                          # noqa: BLE001
            log_line(f'ALERT state 저장 실패: {e}')
        log_line(f'ALERT_COLLAPSE {bad} near_dead ≥ {ALERT_NEAR} → exit {EXIT_ALERT} (런처는 재시도하지 않음)')
        tqdm.write(f'  [ALERT] {bad} 소멸 감지 → exit {EXIT_ALERT}')
        try:
            wandb.finish(exit_code=EXIT_ALERT)
        except Exception:
            pass
        sys.exit(EXIT_ALERT)
    return nd


def run_probe(model, val_ds, probe_idx, device, epoch, global_step):
    """smoke_recipe_fix.run_validation (normalize=True) — 기능 시험. 결과 json + log 'PROBE' 줄."""
    t0 = time.time()
    res = R.run_validation(model, HARNESS_SEQ, True, val_ds, probe_idx, device, 2, True, True)
    tests = res['tests']
    _de = res.get('decomp_equiv_maxabs')
    if 'shuffle_cross' not in tests or _de is None or not (_de < 1e-3):
        # 하네스 라벨이 맞으면 shuffle 시험이 있고 forward 와 분해 경로의 출력이 같다 (4000 step 시험 실측 0.0)
        log_line(f'PROBE-WARN ep{epoch + 1} shuffle_cross={"있음" if "shuffle_cross" in tests else "없음"} '
                 f'decomp_equiv_maxabs={_de} — 기능 시험 경로 확인 필요')
    verdict = dict(deep_used=R.used_verdict(tests.get('deep4')))
    if HAS_SEQ:
        verdict['seq_used'] = R.used_verdict(tests.get('shuffle_cross'))
        verdict['seq_used_adj'] = R.used_verdict(tests.get('shuffle_adj'))
    out = dict(epoch=epoch + 1, global_step=global_step, n=res['n'], normal=res['normal'], tests=tests,
               verdict=verdict, criteria=f'mean>0 & mean>2SE & sign-test p<{R.TH_USED_P}',
               seconds=round(time.time() - t0, 1))
    for k in ('seq_out_spatial_std_mean', 'seq_out_rms_mean', 'seq_out_pair_rel_diff_mean', 'decomp_equiv_maxabs'):
        if k in res:
            out[k] = res[k]
    write_json_atomic(out, os.path.join(PATH_FOLDER, f'probe_ep{epoch + 1}.json'))

    def fmt(t):
        st = tests.get(t)
        if st is None:
            return f'{t}=-'
        se = st['se'] if st['se'] is not None else float('nan')
        p = f"{st['p_sign']:.2g}" if st['p_sign'] is not None else '-'      # 하락이 전부 0(동률)이면 p 없음
        return f"{t} dSSIM={st['mean']:+.5f}±{se:.5f} ({st['n_pos']}/{st['n']}, p={p})"

    names = (['shuffle_cross', 'shuffle_adj', 'zero'] if HAS_SEQ else []) + ['deep4']
    msg = (f'PROBE ep{epoch + 1} slices={res["n"]} ssim={res["normal"]["ssim"]:.4f} | '
           + ' | '.join(fmt(t) for t in names) + f' | verdict {verdict}')
    tqdm.write('  ' + msg)
    log_line(msg)
    try:
        wl = {'probe/ssim': res['normal']['ssim'], 'probe/deep4_dssim': tests['deep4']['mean']}
        if HAS_SEQ:
            wl.update({'probe/shuffle_dssim': tests['shuffle_cross']['mean'],
                       'probe/shuffle_npos': tests['shuffle_cross']['n_pos'],
                       'probe/zero_dssim': tests['zero']['mean']})
        wandb.log(wl, step=global_step)
    except Exception:
        pass
    return out


# ── Mamba 진단 (기록 전용 — 학습·RNG 에 영향 없음) ──
MAMBA_SUBS = ('in_proj.weight', 'conv1d.weight', 'conv1d.bias', 'x_proj.weight', 'dt_proj.weight', 'dt_proj.bias',
              'A_log', 'D', 'out_proj.weight')
MAMBA_BLOCKS = ('gru_h.fwd', 'gru_h.bwd', 'gru_v.fwd', 'gru_v.bwd')


@torch.no_grad()
def mamba_init_ref(model):
    """초기값 기준: 작은 텐서(A_log·dt_proj.bias·D)는 CPU 사본, 큰 가중치(out_proj·in_proj)는 노름만."""
    ref = {}
    for b in MAMBA_BLOCKS:
        blk = model.get_submodule(b)
        ref[b] = dict(A_log=blk.A_log.detach().float().cpu().clone(), dt_bias=blk.dt_proj.bias.detach().float().cpu().clone(),
                      D=blk.D.detach().float().cpu().clone(),
                      out_proj_norm=float(blk.out_proj.weight.detach().float().norm()),
                      in_proj_norm=float(blk.in_proj.weight.detach().float().norm()))
    return ref


@torch.no_grad()
def mamba_diag(model, optimizer, ref, epoch, global_step):
    """단계·방향별 out_proj·in_proj 노름 비(초기 대비), A_log·dt_bias·D 상대 변화, softplus(dt_bias) 평균, 그리고 시퀀스 하위
    텐서별 Adam √v̂ < eps 비율(eps 가 갱신을 지배하는 원소 비율 — 4000 step 시험의 eps_dom 과 같은 정의)."""
    eps = ADAM_EPS
    beta2 = optimizer.param_groups[0]['betas'][1]
    pmap = dict(model.named_parameters())
    out = dict(epoch=epoch + 1, global_step=global_step, blocks={}, eps_dom={})
    rel = lambda a, b: float((a - b).norm() / b.norm().clamp(min=1e-30))      # noqa: E731
    for b in MAMBA_BLOCKS:
        blk, r = model.get_submodule(b), ref[b]
        out['blocks'][b] = dict(
            out_proj_ratio=float(blk.out_proj.weight.float().norm()) / max(r['out_proj_norm'], 1e-30),
            in_proj_ratio=float(blk.in_proj.weight.float().norm()) / max(r['in_proj_norm'], 1e-30),
            A_log_rel=rel(blk.A_log.float().cpu(), r['A_log']), dt_bias_rel=rel(blk.dt_proj.bias.float().cpu(), r['dt_bias']),
            D_rel=rel(blk.D.float().cpu(), r['D']),
            softplus_dt_bias_mean=float(torch.nn.functional.softplus(blk.dt_proj.bias.float()).mean()))
    for stage in ('gru_h', 'gru_v'):
        for sub in MAMBA_SUBS:
            below = total = 0
            for d in ('fwd', 'bwd'):
                p = pmap[f'{stage}.{d}.{sub}']
                st = optimizer.state.get(p, {})
                if 'exp_avg_sq' not in st:
                    continue
                step = float(st['step'])
                thr = (eps ** 2) * (1.0 - beta2 ** step)            # √(v/(1-β2^t)) < eps  ⇔  v < eps²·(1-β2^t)
                for ch in st['exp_avg_sq'].reshape(-1).split(R.STAT_CHUNK):
                    below += int((ch < thr).sum())
                total += p.numel()
            if total:
                out['eps_dom'][f'{stage}.{sub}'] = below / total
    write_json_atomic(out, os.path.join(PATH_FOLDER, f'mamba_diag_ep{epoch + 1}.json'))
    bl = out['blocks']
    # json 은 git 동기화 대상이 아니므로(pod_git_sync.sh 는 log.txt·probe json 만) 모든 값을 log.txt 줄에도 남긴다
    msg = (f'MAMBA-DIAG ep{epoch + 1} out_proj_ratio ' + ' '.join(f'{b}={bl[b]["out_proj_ratio"]:.3f}' for b in MAMBA_BLOCKS)
           + ' | in_proj_ratio ' + ' '.join(f'{b}={bl[b]["in_proj_ratio"]:.3f}' for b in MAMBA_BLOCKS)
           + ' | A_log_rel ' + ' '.join(f'{b}={bl[b]["A_log_rel"]:.2e}' for b in MAMBA_BLOCKS)
           + ' | dt_bias_rel ' + ' '.join(f'{b}={bl[b]["dt_bias_rel"]:.2e}' for b in MAMBA_BLOCKS)
           + ' | D_rel ' + ' '.join(f'{b}={bl[b]["D_rel"]:.2e}' for b in MAMBA_BLOCKS)
           + ' | softplus_dt_bias ' + ' '.join(f'{b}={bl[b]["softplus_dt_bias_mean"]:.4g}' for b in MAMBA_BLOCKS)
           + ' | eps_dom ' + ' '.join(f'{k}={v:.3f}' for k, v in out['eps_dom'].items()))
    log_line(msg)
    try:
        wl = {f'mamba/{b}/out_proj_ratio': bl[b]['out_proj_ratio'] for b in MAMBA_BLOCKS}
        wl.update({f'mamba/eps_dom/{k}': v for k, v in out['eps_dom'].items()})
        wandb.log(wl, step=global_step)
    except Exception:
        pass
    return out


def main():
    print('====================================================')
    print(f' [Pure ETER-Net v8fix — recipe {RECIPE}]  SEQ_MODEL={SEQ_MODEL}  BS={BATCH_SIZE}  '
          f'EPOCHS={NUM_EPOCHS}  STOP_AFTER_EPOCH={STOP_AFTER_EPOCH or "-"}')
    print(f'   run={_RUN_NAME}   logs={PATH_FOLDER}')
    print('====================================================')

    def _refuse(where, bad):
        log_line(f'REFUSE {where}: ' + '; '.join(bad))
        print(f'[refuse] {where}: ' + '; '.join(bad))
        return EXIT_REFUSE

    if _STARTUP_BAD:
        return _refuse('start', _STARTUP_BAD)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필수.")
    device = torch.device("cuda")
    print(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
    if not check_env_for_model('ss2d', 'myConfig_pure_eter_v8', strict=True):   # 'ss2d' = mamba_ssm 의존성까지 점검
        return 1
    _fast = {k: getattr(_MS, k, None) is not None for k in ('causal_conv1d_fn', 'selective_scan_fn', 'mamba_inner_fn')}
    if not all(_fast.values()):
        return _refuse('start', [f'Mamba fast path 함수 없음 {_fast} — causal_conv1d·selective_scan CUDA 확장 확인'])

    for marker in ('DONE', 'ALERT_COLLAPSE'):
        if os.path.exists(os.path.join(PATH_FOLDER, marker)):
            print(f'[stop] {PATH_FOLDER}/{marker} 가 있음 — 실행하지 않음 (ALERT 는 조사 후 수동으로 지울 것)')
            return EXIT_ALERT if marker == 'ALERT_COLLAPSE' else 0

    if SEED is not None:
        R.seed_all(SEED)                       # 시퀀스 모듈 초기값 위치 = R.build_model 과 같음 (seed_all 직후 CPU 에서 생성)
    # R.build_model('gru') 와 같은 config 인자 + Mamba 차원(모델 기본값). CPU 에서 만든 뒤 GPU 로 (R.build_model 과 같음).
    model = PureETER_Mamba1D(
        dim=IMAGE_SIZE[0], n_coil=N_COIL, n_hidden_1=N_HIDDEN_LRNN_1, n_hidden_2=N_HIDDEN_LRNN_2,
        unet_depth=UNET_DEPTH, unet_wf=UNET_WF, use_dc=False, fp32_stages=MAMBA_FP32_STAGES,
    ).to(device)
    # 모델 계약 (sanity_mamba1d.py 와 같은 확인): 데이터 흐름은 PureETER_GRU 의 함수 그대로, nn.GRU 없음, fp32 단계, 파라미터 수
    _n_seq = sum(p.numel() for n, p in model.named_parameters() if n.startswith(('gru_h.', 'gru_v.')))
    _n_all = sum(p.numel() for p in model.parameters())
    _bad = []
    if type(model)._seq is not PureETER_GRU._seq or type(model).forward is not PureETER_GRU.forward:
        _bad.append('_seq/forward 가 PureETER_GRU 의 것이 아님')
    if any(isinstance(m, nn.RNNBase) for m in model.modules()):
        _bad.append('nn.GRU/RNN 모듈이 있음')
    if not (model.gru_h.force_fp32 and model.gru_v.force_fp32 and tuple(model.fp32_stages) == ('h', 'v')):
        _bad.append(f'fp32 단계 불일치 h={model.gru_h.force_fp32} v={model.gru_v.force_fp32} {model.fp32_stages}')
    if (_n_seq, _n_all) != (EXPECT_SEQ_PARAMS, EXPECT_TOTAL_PARAMS):
        _bad.append(f'파라미터 수 seq {_n_seq:,} / 전체 {_n_all:,} != {EXPECT_SEQ_PARAMS:,} / {EXPECT_TOTAL_PARAMS:,}')
    if _bad:
        return _refuse('model', _bad)
    MAMBA_CFG = dict(model.mamba_cfg)
    # 동작을 정하는 코드(모델·상속한 데이터 흐름·레시피 함수)의 AST 지문 — 재개 때 달라졌으면 거부
    CODE_SHA = {os.path.basename(f): _code_hash(f) for f in (sys.modules[PureETER_Mamba1D.__module__].__file__,
                                                            sys.modules[PureETER_GRU.__module__].__file__, R.__file__)}
    # 재개 일관성 검사·last.pt 의 run_config 에 더하는 항목 (JSON 으로 비교 가능한 기본형만). 학습 동작을 바꾸는 환경 변수 포함.
    RUN_CONFIG_MAMBA = dict(model_class='PureETER_Mamba1D', harness_seq=HARNESS_SEQ,
                            mamba_fp32_stages=list(MAMBA_FP32_STAGES), mamba_cfg=MAMBA_CFG, mamba_init=MAMBA_INIT,
                            code_ast_sha256=CODE_SHA, gru_fp32_modules=[],
                            train_env=dict(VAL_EVERY_N_EPOCHS=VAL_EVERY_N_EPOCHS, AMP_GRAD_RETRY=AMP_GRAD_RETRY,
                                           AMP_MIN_SCALE=AMP_MIN_SCALE, TRAIN_NAN_RETRY=TRAIN_NAN_RETRY,
                                           VAL_MAX_RETRY=VAL_MAX_RETRY, SAME_UNET_INIT=SAME_UNET_INIT),
                            debug=dict(DEBUG_MAX_STEPS=DEBUG_MAX_STEPS, DEBUG_VAL_SAMPLES=DEBUG_VAL_SAMPLES,
                                       PROBE_VAL_SLICES=PROBE_VAL_SLICES, HEALTH_EVERY_STEPS=HEALTH_EVERY_STEPS))
    print(f'[mamba1d] fp32_stages={MAMBA_FP32_STAGES} cfg={MAMBA_CFG} seq={_n_seq:,} total={_n_all:,} fast_path={_fast}')
    unet_init_note = 'own'
    if SEED is not None and SAME_UNET_INIT and SEQ_MODEL != 'unet':
        # U-Net 초기값을 모든 모델에서 같게: seed_all(SEED) 직후 만든 U-Net 단독 모델의 U-Net 가중치를 복사한다.
        # (wrapper 가 시퀀스 모듈을 먼저 만들어 같은 SEED 에서도 U-Net 초기값이 모델마다 달랐다 — 기존 런·스모크 공통.)
        # 전역 RNG(torch·numpy·python)는 복사 전 상태로 되돌린다.
        _np_state, _py_state = np.random.get_state(), random.getstate()
        with torch.random.fork_rng(devices=[]):
            R.seed_all(SEED)
            _ref = R.build_model('unet', torch.device('cpu'))
        np.random.set_state(_np_state); random.setstate(_py_state)
        _sd = _ref.unet.state_dict()
        assert list(_sd.keys()) == list(model.unet.state_dict().keys()), 'U-Net 구조 불일치 — 계약 위반'
        model.unet.load_state_dict(_sd)
        del _ref, _sd
        unet_init_note = 'copied from PureETER_UNET built right after seed_all(SEED)'
    groups = R.build_groups(model, HARNESS_SEQ)
    with torch.no_grad():                      # U-Net 초기값 확인용 (모델끼리 같아야 함 — SCRATCH START 줄에 기록)
        _unet_checksum = float(sum((p.double() * (i + 1)).sum() for i, p in enumerate(model.unet.parameters())))
    if EXPECT_UNET_CHECKSUM is not None and not _rel_close(_unet_checksum, EXPECT_UNET_CHECKSUM, 1e-9):
        return _refuse('start', [f'U-Net 초기값 checksum {_unet_checksum:.10e} != 기대 {EXPECT_UNET_CHECKSUM:.10e} '
                                 f'(ETER-net·U-Net 단독 런과 다름; unet_init={unet_init_note})'])
    _init_ref = mamba_init_ref(model)          # 진단 기준 (재개 때는 last.pt 의 것으로 바꿈)

    last_ckpt_path = os.path.join(PATH_FOLDER, f'{PREFIX}_last.pt')
    _full_state = None
    if os.path.exists(last_ckpt_path):
        _full_state = torch.load(last_ckpt_path, map_location=device)
        # 설정·코드 판 확인을 상태 복원·recipe.json·wandb 보다 먼저 (어긋나면 load_state_dict 가 예외 → rc 1 → 런처 재시도가
        # 되지 않도록 여기서 exit 2). 아래 원 트레이너의 검사(T_max 등)도 그대로 남겨 둔다.
        _rc0, _ri0, _bad0, _miss = _full_state.get('run_config', {}), _full_state.get('recipe', {}), [], object()
        for k, cur in list(RUN_CONFIG_MAMBA.items()) + [('batch_size', BATCH_SIZE), ('num_epochs', NUM_EPOCHS), ('seed', SEED)]:
            got = _rc0.get(k, _miss)
            if got is _miss or got != cur:
                _bad0.append(f'{k} ckpt {"(없음)" if got is _miss else got} != 현재 {cur}')
        for k, cur in (('smoke_script_version', R.SCRIPT_VERSION), ('max_norm', MAX_NORM), ('adam_eps', ADAM_EPS),
                       ('k_ratio', float(R.K_RATIO))):
            if _ri0.get(k, _miss) != cur:
                _bad0.append(f'recipe {k} ckpt {_ri0.get(k, "(없음)")} != 현재 {cur}')
        if _bad0:
            return _refuse('resume', _bad0)
        try:
            model.load_state_dict(_full_state['model'])
        except (RuntimeError, KeyError) as e:
            return _refuse('resume', [f'model state_dict 로드 실패: {type(e).__name__}: {str(e)[:300]}'])
        resume_mode = 'full'
        print(f"\n[Resume:full] {last_ckpt_path} → epoch {_full_state['epoch']} 완료, 전체 상태 복원 예정")
    else:
        resume_mode = 'scratch'
        print("\n[Scratch] 처음부터 학습")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {num_params / 1e6:.1f}M")

    criterion_ssim_loss = SSIM().to(device)
    optimizer, opt_info = R.build_optimizer(model, RECIPE)     # AdamW 분리형 + no-decay 그룹, eps 1e-8
    print(f"Optimizer: {opt_info['type']} lr={opt_info['lr']} wd={opt_info['weight_decay']} (decoupled) eps={opt_info['eps']} "
          f"| no-decay {opt_info['no_decay_numel']:,} 원소 {opt_info.get('no_decay_reasons')}")

    print("\nFastMRI 데이터 파이프라인 연결 중...")
    choh_data_train = FastMRI_H5_Dataloader(
        DATA_TRAIN, num_files=None, target_size=IMAGE_SIZE[0],
        augment=TRAIN_AUGMENT, augment_flip_p=TRAIN_AUGMENT_FLIP_P,
    )

    # λ_SSIM = λ/s̄ — 처음 실행 때 계산, 재개 때 저장값 사용 (train 64 슬라이스·고정 rng — 끝나면 dataset rng 복원)
    if _full_state is not None and 'recipe' in _full_state:
        recipe_info = _full_state['recipe']
    else:
        sbar = R.estimate_mean_scale(choh_data_train, SEED if SEED is not None else 1)
        recipe_info = dict(name=RECIPE, s_bar=sbar, lambda_base=float(LAMBDA_SSIM_PER_PIXEL),
                           lambda_ssim=float(LAMBDA_SSIM_PER_PIXEL) / sbar['mean'], max_norm=MAX_NORM,
                           adam_eps=ADAM_EPS, k_ratio=float(R.K_RATIO), optimizer=opt_info['type'],
                           weight_decay=float(LAMBDA_REGULAR_PER_PIXEL), lr=float(LEARNING_RATE_ADAM),
                           smoke_script_version=R.SCRIPT_VERSION)
    LAM = float(recipe_info['lambda_ssim'])
    if EXPECT_SBAR is not None and not _rel_close(float(recipe_info['s_bar']['mean']), EXPECT_SBAR, 1e-9):
        # 일시적 읽기 오류(이 Pod 계열에서 드물게 데이터가 깨짐)와 구별: 처음 실행이면 한 번 더 추정해 본다.
        # 다시 추정해 기대값과 맞으면 그 값으로 계속, 비유한이면 rc 1(런처 재시도), 두 추정이 같고 기대값과 다르면 거부(exit 2).
        _first = float(recipe_info['s_bar']['mean'])
        if _full_state is None:
            _sb2 = R.estimate_mean_scale(choh_data_train, SEED if SEED is not None else 1)
            log_line(f'SBAR-recheck first={_first!r} second={_sb2["mean"]!r} expect={EXPECT_SBAR!r}')
            if _rel_close(float(_sb2['mean']), EXPECT_SBAR, 1e-9):
                recipe_info.update(s_bar=_sb2, lambda_ssim=float(LAMBDA_SSIM_PER_PIXEL) / _sb2['mean'])
                LAM = float(recipe_info['lambda_ssim'])
            elif not (np.isfinite(_first) and np.isfinite(_sb2['mean'])):
                print('[s̄] 비유한 — 일시 오류로 보고 rc 1 (런처 재시도)')
                return 1
        if not _rel_close(float(recipe_info['s_bar']['mean']), EXPECT_SBAR, 1e-9):
            return _refuse('start', [f"s̄ {recipe_info['s_bar']['mean']!r} != 기대 {EXPECT_SBAR!r} — 학습 데이터가 "
                                     "ETER-net·U-Net 단독 런과 다름"])
    write_json_atomic(dict(recipe_info, run=_RUN_NAME, seq=SEQ_MODEL, seed=SEED, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                           harness_seq=HARNESS_SEQ, model_class='PureETER_Mamba1D', mamba_cfg=MAMBA_CFG,
                           mamba_fp32_stages=list(MAMBA_FP32_STAGES), mamba_init=MAMBA_INIT, code_ast_sha256=CODE_SHA,
                           precision_note='Mamba h·v fp32, U-Net fp16 autocast (기준 ETER-net: gru_h fp32, gru_v·U-Net fp16)'),
                      os.path.join(PATH_FOLDER, 'recipe.json'))
    print(f"레시피 {RECIPE}: s̄={recipe_info['s_bar']['mean']:.2f} → λ_SSIM={LAM:.6f}, max_norm={MAX_NORM:g}, "
          f"정규화(영상·label /s, k-space ×{R.K_RATIO:g}/s)")

    _train_loader_kwargs = dict(batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS_TRAIN, pin_memory=True)
    _g = None
    if SEED is not None:
        # 기존 트레이너 :237-252 의 SEED 경로(dataset rng·워커별 독립 스트림·셔플 순서 고정)를 epoch 단위로 결정적으로 만든다:
        # 워커를 epoch 마다 새로 만들고(persistent_workers=False), 각 epoch 시작 전에 generator 를 SEED+1000×epoch 로
        # 다시 시드한다 → 셔플 순서와 워커 base seed(→ 마스크 offset·flip) 가 (SEED, epoch) 만의 함수.
        # epoch 1 (epoch=0) 은 기존 트레이너·스모크와 같은 스트림이고, 재개(계획 정지·장애 재시작)해도 끊김 없는 런과 같은
        # epoch 데이터를 본다. 기존 트레이너는 재개 프로세스가 generator 를 SEED 로 새로 만들어 재개 직후 epoch 가 1 epoch 의
        # 셔플·마스크·flip 을 그대로 반복했다.
        choh_data_train.rng = np.random.default_rng(SEED + 1)   # 워커 0 개일 때만 쓰임 (워커는 worker_init_fn 이 재시드)
        _g = torch.Generator()
        _g.manual_seed(SEED)
        _train_loader_kwargs.update(worker_init_fn=R._worker_init_fn, generator=_g)
        if NUM_WORKERS_TRAIN > 0:
            _train_loader_kwargs.update(persistent_workers=False, prefetch_factor=PREFETCH_FACTOR)
    elif NUM_WORKERS_TRAIN > 0:
        _train_loader_kwargs.update(persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)
    trainloader = DataLoader(choh_data_train, **_train_loader_kwargs)
    print(f"Train Dataloader 준비 완료! ({len(choh_data_train)} 샘플)")
    choh_data_val = FastMRI_H5_Dataloader(
        DATA_VAL, num_files=None, target_size=IMAGE_SIZE[0],
        random_mask=False, augment=False,
    )
    val_full_n = len(choh_data_val)
    val_ds_for_loader = Subset(choh_data_val, list(range(min(DEBUG_VAL_SAMPLES, val_full_n)))) \
        if DEBUG_VAL_SAMPLES > 0 else choh_data_val
    val_loader = DataLoader(
        val_ds_for_loader, batch_size=max(1, BATCH_SIZE // 2), shuffle=False,
        num_workers=NUM_WORKERS_VAL, pin_memory=True,
    )
    probe_idx = R.even_indices(val_full_n, PROBE_VAL_SLICES) if PROBE_VAL_SLICES > 0 else []
    print(f"Val   Dataloader 준비 완료! ({len(val_ds_for_loader)} / {val_full_n} 샘플; 기능 시험 {len(probe_idx)} 슬라이스)")
    if DEBUG_MAX_STEPS or DEBUG_VAL_SAMPLES:
        print(f"  [DEBUG] DEBUG_MAX_STEPS={DEBUG_MAX_STEPS} DEBUG_VAL_SAMPLES={DEBUG_VAL_SAMPLES} — 측정·인용 불가 런")

    steps_per_epoch = len(trainloader)
    opt_steps_per_epoch = max(1, steps_per_epoch // ACCUM_STEPS)
    total_steps = opt_steps_per_epoch * NUM_EPOCHS       # 디버그 step 상한과 무관하게 정식 일정 기준
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    print(f"Scheduler: CosineAnnealingLR T_max={total_steps} opt-steps ({opt_steps_per_epoch}/epoch × {NUM_EPOCHS}), eta_min=1e-6")

    _wandb_tag = os.environ.get('WANDB_RUN_TAG', '')
    _wandb_id  = _RUN_NAME + (f'_{_wandb_tag}' if _wandb_tag else '')
    wandb.init(
        project=os.environ.get('WANDB_PROJECT', 'ViT-MRI-Recon'), name=f'{_wandb_id}_BS{BATCH_SIZE}',
        id=_wandb_id, resume='allow',
        config={
            'track': 'v8fix', 'recipe': RECIPE, 'seq_model': SEQ_MODEL, 'use_dc': False, 'seed': SEED,
            'image_size': IMAGE_SIZE, 'n_hidden_lrnn': N_HIDDEN_LRNN_2,
            'ss2d_d_inner': SS2D_D_INNER, 'ss2d_d_state': SS2D_D_STATE,
            'unet_depth': UNET_DEPTH, 'unet_wf': UNET_WF, 'batch_size': BATCH_SIZE, 'num_epochs': NUM_EPOCHS,
            'stop_after_epoch': STOP_AFTER_EPOCH, 'learning_rate': LEARNING_RATE_ADAM,
            'optimizer': 'AdamW(decoupled, no-decay group)', 'weight_decay': LAMBDA_REGULAR_PER_PIXEL,
            'adam_eps': ADAM_EPS, 'max_norm': MAX_NORM, 'normalize': 'per-sample p99 zero-filled RSS',
            'lambda_ssim': LAM, 's_bar': recipe_info['s_bar']['mean'], 'num_params': num_params,
            'train_samples': len(choh_data_train), 'val_samples': len(val_ds_for_loader),
            'best_metric': 'val_ssim', 'val_every_n_epochs': VAL_EVERY_N_EPOCHS, 'scheduler': 'CosineAnnealingLR',
            'harness_seq': HARNESS_SEQ, 'model_class': 'PureETER_Mamba1D', 'mamba_cfg': MAMBA_CFG,
            'mamba_fp32_stages': ''.join(MAMBA_FP32_STAGES), 'mamba_init': MAMBA_INIT,
            'precision_note': 'Mamba h,v fp32 / U-Net fp16 autocast (ref ETER-net: gru_h fp32, gru_v fp16)',
            'trainer': 'main_train_pure_v8fix_mamba1d.py (from ampguard)',
        },
    )

    scaler = torch.amp.GradScaler('cuda')
    model.train()
    best_val_ssim = -1.0
    best_val = {'ssim': None, 'psnr': None, 'nmse': None, 'l1': None, 'epoch': None}
    global_step = 0
    consec_skip = 0
    total_skip = 0
    total_nan_retry = total_nan_recovered = 0
    total_grad_retry = total_grad_recovered = total_amp_floor = 0
    start_epoch = 0
    tic = time.time()

    if resume_mode == 'full':
        try:
            optimizer.load_state_dict(_full_state['optimizer'])
            scheduler.load_state_dict(_full_state['scheduler'])
            scaler.load_state_dict(_full_state['scaler'])
        except (RuntimeError, ValueError, KeyError) as e:
            wandb.finish(exit_code=EXIT_REFUSE)
            return _refuse('resume', [f'optimizer/scheduler/scaler 상태 로드 실패: {type(e).__name__}: {str(e)[:300]}'])
        # 재개 일관성: 일정·BS·레시피 코드가 정지 때와 같아야 한다 (다르면 거부 — exit 2, 런처는 재시도하지 않음)
        _rc = _full_state.get('run_config', {})
        _ri = _full_state.get('recipe', {})
        _bad = []
        if scheduler.T_max != total_steps:
            _bad.append(f'T_max ckpt {scheduler.T_max} != 현재 {total_steps}')
        for k, cur in (('batch_size', BATCH_SIZE), ('num_epochs', NUM_EPOCHS), ('seed', SEED)):
            if k in _rc and _rc[k] != cur:
                _bad.append(f'{k} ckpt {_rc[k]} != 현재 {cur}')
        # Mamba-ETER: 정밀도·모델·코드 판이 정지 때와 같아야 한다 (키가 없어도 불일치로 본다)
        _missing = object()
        for k, cur in RUN_CONFIG_MAMBA.items():
            got = _rc.get(k, _missing)
            if got is _missing or got != cur:
                _bad.append(f'{k} ckpt {"(없음)" if got is _missing else got} != 현재 {cur}')
        for k, cur in (('smoke_script_version', R.SCRIPT_VERSION), ('max_norm', MAX_NORM), ('adam_eps', ADAM_EPS),
                       ('k_ratio', float(R.K_RATIO))):
            if k in _ri and _ri[k] != cur:
                _bad.append(f'recipe {k} ckpt {_ri[k]} != 현재 {cur}')
        if _bad:
            log_line('REFUSE resume: ' + '; '.join(_bad))
            print('[refuse] 재개 설정 불일치: ' + '; '.join(_bad))
            wandb.finish(exit_code=EXIT_REFUSE)
            return EXIT_REFUSE
        best_val_ssim = _full_state['best_val_ssim']
        best_val      = _full_state['best_val']
        global_step   = _full_state.get('global_step', 0)
        total_skip    = _full_state.get('total_skip', 0)
        total_nan_retry     = _full_state.get('total_nan_retry', 0)
        total_nan_recovered = _full_state.get('total_nan_recovered', 0)
        total_grad_retry     = _full_state.get('total_grad_retry', 0)
        total_grad_recovered = _full_state.get('total_grad_recovered', 0)
        total_amp_floor      = _full_state.get('total_amp_floor', 0)
        start_epoch   = _full_state['epoch']
        _ir = _full_state.get('mamba_init_ref')
        if _ir is not None:                    # 진단 기준 = 처음 실행 때의 초기값 (map_location=cuda 로 올라온 텐서는 CPU 로)
            _init_ref = {b: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()} for b, d in _ir.items()}
        rng = _full_state.get('rng', {})
        # torch.load(map_location=cuda) 가 RNG ByteTensor 를 GPU 로 옮기므로 CPU 로 되돌려 복원한다
        # (기존 트레이너는 이 때문에 재개 때마다 "RNG state must be a torch.ByteTensor" 로 복원에 실패했다).
        if rng.get('torch_cpu') is not None:
            torch.set_rng_state(rng['torch_cpu'].cpu())
        if rng.get('torch_cuda') is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in rng['torch_cuda']])
        if rng.get('numpy') is not None:
            np.random.set_state(rng['numpy'])
        if rng.get('python') is not None:
            random.setstate(rng['python'])
        print(f"[Resume:full] start_epoch={start_epoch}, LR={scheduler.get_last_lr()[0]:.3e}, "
              f"best_val_ssim={best_val_ssim:.4f}, global_step={global_step}")
        if not (STOP_AFTER_EPOCH and start_epoch >= STOP_AFTER_EPOCH and STOP_AFTER_EPOCH < NUM_EPOCHS):
            log_line(f'RESUME start_epoch={start_epoch} best_val_ssim={best_val_ssim:.4f} global_step={global_step} '
                     f'data_seed=SEED+1000*epoch')
        del _full_state
    else:
        log_line(f'SCRATCH START run={_RUN_NAME} recipe={RECIPE} BS={BATCH_SIZE} LR={LEARNING_RATE_ADAM} '
                 f'EPOCHS={NUM_EPOCHS} STOP_AFTER_EPOCH={STOP_AFTER_EPOCH or "-"} params={num_params/1e6:.1f}M '
                 f'lambda_ssim={LAM:.6f} s_bar={recipe_info["s_bar"]["mean"]:.2f} max_norm={MAX_NORM:g} '
                 f'unet_init={unet_init_note} unet_init_checksum={_unet_checksum:.10e} data_seed=SEED+1000*epoch '
                 f'trainer=mamba1d(from ampguard) amp_grad_retry={AMP_GRAD_RETRY} amp_min_scale={AMP_MIN_SCALE:g} '
                 f'seq_model={SEQ_MODEL} harness_seq={HARNESS_SEQ} mamba_fp32_stages={"".join(MAMBA_FP32_STAGES)} '
                 f'mamba_init="{MAMBA_INIT}" mamba_cfg={json.dumps(MAMBA_CFG, separators=(",", ":"))} '
                 f'seq_params={_n_seq} total_params={_n_all} '
                 f'expect_sbar={EXPECT_SBAR} expect_unet_checksum={EXPECT_UNET_CHECKSUM} '
                 f'debug={json.dumps(RUN_CONFIG_MAMBA["debug"], separators=(",", ":"))}')
        health_check(model, groups, -1, 0, 'init')

    if STOP_AFTER_EPOCH and start_epoch >= STOP_AFTER_EPOCH and STOP_AFTER_EPOCH < NUM_EPOCHS:
        print(f'[stop] 이미 epoch {start_epoch} 완료 ≥ STOP_AFTER_EPOCH {STOP_AFTER_EPOCH} — 학습 일시정지 상태 유지')
        wandb.finish()
        return 0

    print(f"\n학습 시작 (일정 {NUM_EPOCHS} epoch, epoch {start_epoch + 1} 부터)")
    epoch_bar = tqdm(range(start_epoch, NUM_EPOCHS), desc='전체 진행', unit='epoch',
                     initial=start_epoch, total=NUM_EPOCHS)
    def _grads_finite():
        gs = [p.grad for p in model.parameters() if p.grad is not None]
        return (not gs) or bool(torch.isfinite(torch.stack(torch._foreach_norm(gs))).all())

    def _test_inject(i, first):          # 시험 전용 (위 AMP_TEST_INJECT_* 주석)
        if (first and i in AMP_TEST_INJECT_ONCE) or i in AMP_TEST_INJECT_ALWAYS:
            p = next(p for p in model.parameters() if p.grad is not None)
            p.grad.view(-1)[0] = float('nan')

    for epoch in epoch_bar:
        ep_loss = ep_l1 = ep_ssimloss = 0.0
        ep_gn_max = 0.0
        ep_clipped = 0
        ep_amp_skip = 0
        n_done = 0
        if _g is not None:
            _ep_seed = SEED + 1000 * epoch
            _g.manual_seed(_ep_seed)
            if NUM_WORKERS_TRAIN == 0:
                choh_data_train.rng = np.random.default_rng(_ep_seed + 1)
        batch_bar = tqdm(trainloader, desc=f'Epoch {epoch+1:3d}/{NUM_EPOCHS}', leave=False, unit='batch')
        optimizer.zero_grad(set_to_none=True)

        def _train_forward(sample):
            data_in     = sample['data'].float().to(device)
            data_in_img = sample['data_img'].float().to(device)
            data_ref    = sample['label'].float().to(device)
            brain_mask  = sample['brain_mask'].float().to(device)
            mask        = sample['mask'].float().to(device)
            sens        = sample['sens'].float().to(device)
            # 레시피 DL: 영상·label /s, k-space ×(100/s)  (smoke_recipe_fix.run_job :1043-1046 과 같음)
            s = R.per_sample_scale(data_in_img)
            data_in, data_in_img = R.normalize_inputs(data_in, data_in_img, s)
            data_ref = data_ref / s

            with torch.amp.autocast('cuda'):
                out = model(data_in_img, data_in, mask, sens)

            out_fp    = out.float()
            m_sum     = brain_mask.sum().clamp(min=1.0)
            loss_l1   = ((out_fp - data_ref).abs() * brain_mask).sum() / m_sum
            loss_ssim = 1 - criterion_ssim_loss(out_fp, data_ref, mask=brain_mask)
            loss      = loss_l1 + LAM * loss_ssim
            return s, out, out_fp, loss_l1, loss_ssim, loss

        for i, sample in enumerate(batch_bar):
            if DEBUG_MAX_STEPS and i >= DEBUG_MAX_STEPS:
                break
            s, out, out_fp, loss_l1, loss_ssim, loss = _train_forward(sample)

            # 일시적 비유한 값(10-09, 위 VAL_MAX_RETRY 주석): 같은 배치를 CPU 쪽 sample 에서 다시 보내 다시 계산한다.
            # 다시 계산해도 비유한이면 아래 기존 NaN-skip. cpu_finite = CPU 쪽 입력이 유한한지(원인 진단용).
            for _r in range(TRAIN_NAN_RETRY):
                if torch.isfinite(loss):
                    break
                _first = loss.item()
                del out, out_fp, loss_l1, loss_ssim, loss
                _cpu_ok = all(bool(torch.isfinite(sample[k]).all()) for k in ('data', 'data_img', 'label', 'brain_mask', 'sens'))
                _amax = {k: f"{float(sample[k].abs().max()):.3g}" for k in ('data', 'data_img', 'label')}
                s, out, out_fp, loss_l1, loss_ssim, loss = _train_forward(sample)
                _ok = bool(torch.isfinite(loss))
                total_nan_retry += 1; total_nan_recovered += int(_ok)
                log_line(f'NaN-retry ep{epoch+1} batch{i} first={_first} cpu_finite={_cpu_ok} input_absmax={_amax} '
                         f'{"recovered" if _ok else "still non-finite"} retry_total={total_nan_retry} recovered_total={total_nan_recovered}')

            if not torch.isfinite(loss):           # 기존 트레이너 :360-377 NaN/Inf-skip 가드
                # 건너뛸 배치의 계산 그래프를 바로 놓는다 — 그대로 두면 다음 forward 동안 두 배치 분량이 GPU 에 남아
                # SS2D(평소 약 20 GB)가 24 GB GPU 에서 OOM 났다(RunPod RTX 3090, 10-09).
                del out, out_fp, loss_l1, loss_ssim
                optimizer.zero_grad(set_to_none=True)
                consec_skip += 1; total_skip += 1
                if consec_skip <= 3 or consec_skip % 50 == 0:
                    tqdm.write(f'  [NaN-skip] ep{epoch+1} batch{i} loss={loss.item()} consec={consec_skip} total={total_skip}')
                    log_line(f'NaN-skip ep{epoch+1} batch{i} consec={consec_skip} total={total_skip}')
                del loss
                if consec_skip >= MAX_CONSEC_SKIP:
                    _msg = f'FATAL: {consec_skip} consecutive non-finite loss (ep{epoch+1} batch{i}) → exit(1)'
                    tqdm.write('  ' + _msg)
                    log_line(_msg)
                    try:
                        wandb.finish(exit_code=1)
                    except Exception:
                        pass
                    sys.exit(1)
                continue
            consec_skip = 0

            scaler.scale(loss).backward()
            _test_inject(i, first=True)
            # AMP 보호 (1): 기울기에 비유한 값이 있으면 같은 배치를 다시 보내 다시 계산 (위 AMP_GRAD_RETRY 주석)
            for _r in range(AMP_GRAD_RETRY):
                if _grads_finite():
                    break
                _sc, _l_first = scaler.get_scale(), loss.item()
                del out, out_fp, loss_l1, loss_ssim, loss
                optimizer.zero_grad(set_to_none=True)
                _cpu_ok = all(bool(torch.isfinite(sample[k]).all()) for k in ('data', 'data_img', 'label', 'brain_mask', 'sens'))
                _amax = {k: f"{float(sample[k].abs().max()):.3g}" for k in ('data', 'data_img', 'label')}
                s, out, out_fp, loss_l1, loss_ssim, loss = _train_forward(sample)
                scaler.scale(loss).backward()
                _test_inject(i, first=False)
                _ok = _grads_finite()
                total_grad_retry += 1; total_grad_recovered += int(_ok)
                if total_grad_retry <= 50 or total_grad_retry % 50 == 0:
                    log_line(f'GRAD-retry ep{epoch+1} batch{i} loss={_l_first:.5g} amp_scale={_sc:g} cpu_finite={_cpu_ok} '
                             f'input_absmax={_amax} {"recovered" if _ok else "still non-finite"} '
                             f'retry_total={total_grad_retry} recovered_total={total_grad_recovered}')
            scaler.unscale_(optimizer)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM))
            scaler.step(optimizer)
            scaler.update()
            if not np.isfinite(gn):
                ep_amp_skip += 1                 # GradScaler 가 이 step 을 건너뛰고 배율을 절반으로 줄였다
            # AMP 보호 (2): 배율 하한 (위 AMP_MIN_SCALE 주석)
            if scaler.get_scale() < AMP_MIN_SCALE:
                _sc = scaler.get_scale()
                scaler.update(AMP_MIN_SCALE)
                total_amp_floor += 1
                if total_amp_floor <= 50 or total_amp_floor % 50 == 0:
                    log_line(f'AMP-floor ep{epoch+1} batch{i} amp_scale {_sc:g} -> {AMP_MIN_SCALE:g} floor_total={total_amp_floor}')
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            n_done += 1
            li, l1i, lsi = loss.item(), loss_l1.item(), loss_ssim.item()
            ep_loss += li; ep_l1 += l1i; ep_ssimloss += lsi
            if np.isfinite(gn):
                ep_gn_max = max(ep_gn_max, gn)
                ep_clipped += int(gn > MAX_NORM)
            if global_step % 20 == 0:
                wandb.log({'train/loss': li, 'train/loss_l1': l1i, 'train/loss_ssim': lsi,
                           'train/l1_over_ssim_term': l1i / max(LAM * lsi, 1e-12), 'train/grad_norm': gn,
                           'train/scale_s_mean': float(s.mean()), 'train/amp_scale': scaler.get_scale(),
                           'train/lr': scheduler.get_last_lr()[0]}, step=global_step)
            batch_bar.set_postfix(Loss=f'{li:.4f}', L1=f'{l1i:.4f}', SSIM_c=f'{1 - lsi:.4f}',
                                  gn=f'{gn:.3g}', LR=f'{scheduler.get_last_lr()[0]:.2e}')
            if HEALTH_EVERY_STEPS > 0 and global_step % HEALTH_EVERY_STEPS == 0:
                health_check(model, groups, epoch, global_step, 'step')

        n_b = max(n_done, 1)
        avg_loss, avg_l1, avg_sl = ep_loss / n_b, ep_l1 / n_b, ep_ssimloss / n_b
        wandb.log({'epoch': epoch + 1, 'epoch/train_loss': avg_loss, 'epoch/train_l1': avg_l1,
                   'epoch/train_ssim_loss': avg_sl, 'epoch/grad_norm_max': ep_gn_max,
                   'epoch/clipped_steps': ep_clipped, 'epoch/amp_skip': ep_amp_skip,
                   'epoch/amp_scale_end': scaler.get_scale()}, step=global_step)
        health_check(model, groups, epoch, global_step, 'epoch-end')

        is_stop = bool(STOP_AFTER_EPOCH) and (epoch + 1) == STOP_AFTER_EPOCH
        do_val = ((epoch + 1) % VAL_EVERY_N_EPOCHS == 0) or is_stop or (epoch + 1) == NUM_EPOCHS
        train_part = (f'train_loss={avg_loss:.5f}  train_l1={avg_l1:.5f}  train_ssim_loss={avg_sl:.5f}  '
                      f'l1_over_ssim_term={avg_l1 / max(LAM * avg_sl, 1e-12):.1f}  gn_max={ep_gn_max:.3g}  '
                      f'clipped={ep_clipped}  steps={n_done}  nan_retry_total={total_nan_retry}  '
                      f'nan_recovered_total={total_nan_recovered}  nan_skip_total={total_skip}  '
                      f'amp_scale={scaler.get_scale():g}  amp_skip={ep_amp_skip}  grad_retry_total={total_grad_retry}  '
                      f'grad_recovered_total={total_grad_recovered}  amp_floor_total={total_amp_floor}')
        if do_val:
            tqdm.write(f'  [Val ep{epoch+1}] running...')
            vm = run_val(model, val_loader, device)
            tqdm.write(f'  [Val] SSIM_m={vm["ssim"]:.4f}  PSNR={vm["psnr"]:.2f}dB  NMSE={vm["nmse"]:.4f}  L1={vm["l1"]:.4f}')
            wandb.log({'val/ssim_masked': vm['ssim'], 'val/psnr_masked': vm['psnr'],
                       'val/nmse_masked': vm['nmse'], 'val/l1_masked': vm['l1'],
                       'val/nonfinite_retry': vm['retry'], 'val/nonfinite_fp32': vm['fp32'],
                       'val/nonfinite_excluded': vm['excluded']}, step=global_step)
            log_line(f'Epoch {epoch+1}/{NUM_EPOCHS}  {train_part}  val_ssim_m={vm["ssim"]:.4f}  '
                     f'val_psnr={vm["psnr"]:.2f}  val_nmse={vm["nmse"]:.4f}  val_l1={vm["l1"]:.4f}  '
                     f'val_batches={vm["n_batches"]}  val_retry={vm["retry"]}  val_recovered={vm["recovered"]}  '
                     f'val_fp32={vm["fp32"]}  val_excluded={vm["excluded"]}')
            for _k in ('retry', 'recovered', 'fp32', 'excluded', 'n_batches'):
                vm.pop(_k)
            if np.isfinite(vm['ssim']) and vm['ssim'] > best_val_ssim:
                best_val_ssim = vm['ssim']
                best_val = dict(vm, epoch=epoch + 1)
                save_checkpoint_atomic(model.state_dict(), os.path.join(PATH_FOLDER, f'{PREFIX}_best.pt'))
                tqdm.write(f'  [Best] val SSIM {best_val_ssim:.4f} → {PREFIX}_best.pt')
        else:
            log_line(f'Epoch {epoch+1}/{NUM_EPOCHS}  {train_part}')

        if (epoch + 1) % 5 == 0 or is_stop:
            save_checkpoint_atomic(model.state_dict(), os.path.join(PATH_FOLDER, f'{PREFIX}_epoch_{epoch+1}.pt'))

        save_checkpoint_atomic({
            'epoch': epoch + 1, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'best_val_ssim': best_val_ssim, 'best_val': best_val,
            'global_step': global_step, 'total_skip': total_skip, 'recipe': recipe_info,
            'total_nan_retry': total_nan_retry, 'total_nan_recovered': total_nan_recovered,
            'total_grad_retry': total_grad_retry, 'total_grad_recovered': total_grad_recovered,
            'total_amp_floor': total_amp_floor,
            'rng': {'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(),
                    'numpy': np.random.get_state(), 'python': random.getstate()},
            'run_config': dict(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, total_steps=total_steps, seed=SEED,
                               unet_init=unet_init_note, **RUN_CONFIG_MAMBA),
            'mamba_init_ref': _init_ref,
        }, last_ckpt_path)

        # 기능 시험은 진단용 — 체크포인트 저장 뒤에 돌리고, 실패해도 학습을 멈추지 않는다
        if do_val and probe_idx:
            try:
                run_probe(model, choh_data_val, probe_idx, device, epoch, global_step)
            except Exception as e:                                      # noqa: BLE001
                import traceback
                log_line(f'PROBE ep{epoch + 1} FAILED: {type(e).__name__}: {e}')
                tqdm.write('  [PROBE 실패 — 계속]\n' + traceback.format_exc())
                model.train()
        if do_val:                             # Mamba 진단 (기록 전용, 실패해도 계속)
            try:
                mamba_diag(model, optimizer, _init_ref, epoch, global_step)
            except Exception as e:                                      # noqa: BLE001
                log_line(f'MAMBA-DIAG ep{epoch + 1} FAILED: {type(e).__name__}: {e}')

        if is_stop and (epoch + 1) < NUM_EPOCHS:
            msg = (f'PAUSED at epoch {epoch+1}/{NUM_EPOCHS} (STOP_AFTER_EPOCH) — 같은 일정으로 재개 가능: '
                   f'런처는 STOP_AFTER_EPOCH={NUM_EPOCHS} (또는 더 큰 중간값)으로 같은 명령, 트레이너 직접 실행은 '
                   f'같은 SMOKE_BS·SANITY_NUM_EPOCHS 에 STOP_AFTER_EPOCH 를 빼거나 키워서')
            log_line(msg)
            write_json_atomic(dict(epoch=epoch + 1, num_epochs=NUM_EPOCHS, global_step=global_step,
                                   time=datetime.datetime.now().isoformat(timespec='seconds')),
                              os.path.join(PATH_FOLDER, f'PAUSED_ep{epoch+1}'))
            print(f'\n학습 일시정지 (epoch {epoch+1}/{NUM_EPOCHS}) 소요: {time.time() - tic:.0f}초')
            wandb.finish()
            return 0

    toc = time.time()
    write_json_atomic(dict(epochs=NUM_EPOCHS, best_val=best_val, global_step=global_step,
                           time=datetime.datetime.now().isoformat(timespec='seconds')),
                      os.path.join(PATH_FOLDER, 'DONE'))
    print(f'\n학습 완료  소요: {toc - tic:.0f}초')
    if best_val['ssim'] is not None:
        print(f'  Best (ep {best_val["epoch"]}) → SSIM_m {best_val["ssim"]:.4f}  PSNR {best_val["psnr"]:.2f}dB  '
              f'NMSE {best_val["nmse"]:.4f}  L1 {best_val["l1"]:.4f}')
    wandb.finish()
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
