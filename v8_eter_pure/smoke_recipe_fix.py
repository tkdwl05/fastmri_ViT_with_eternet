"""
v8 학습 방법 수정 스모크 (2026-10-07, 리뷰 반영판 2026-10-07b) — 시퀀스 모듈·U-Net 가중치 소멸이 어떤 레시피 수정으로
막히는지 짧은 학습으로 측정한다.

배경: docs/v8_training_collapse_audit_2026-10-07.md — v8/v9 의 모든 런에서 시퀀스 모듈(bi-GRU 637M / SS2D)과 U-Net 파라미터의
약 97% 가 늦어도 epoch 5 에 |w|≈2e-40 으로 소멸했다. 원인 = main_train_pure_v8.py:225-226 의 Adam 결합형 L2(weight_decay 3e-5,
전 파라미터) + :384 의 전역 clip_grad_norm_(1.0) (×1e6 강도 단위 L1 때문에 clip 전 norm 44~76 → 계수 0.01~0.02).
데이터 파이프라인·모델 코드는 정상. 이 스크립트는 감사 §(5) 1순위 "레시피 원인 분리 스모크" 를 수행한다.

재사용 원칙 (트레이너는 import 하지 않는다 — 모듈 수준에서 SEED·폴더 생성·wandb 등 부작용이 있음):
  - 데이터: 같은 FastMRI_H5_Dataloader 인자·augment·랜덤 마스크 offset·BS·워커 수·SEED 경로
           (random/np/torch 시드, dataset rng = SEED+1, worker_init_fn, 셔플 generator) — 트레이너 줄 번호를 각 함수에 적었다.
  - 모델: 트레이너 build_model 과 같은 wrapper·같은 config 인자 (use_dc=False). seq ∈ {unet, gru, ss2d}.
  - 손실: masked L1 + λ_SSIM·(1 − masked SSIM) — u_choh_SSIM.SSIM 같은 호출, 같은 brain mask. λ_SSIM 기본 = config 1.0.
  - AMP fp16 autocast + GradScaler('cuda') (CUDA 에서만 켬 — 트레이너와 같음), NaN/Inf-skip 가드(연속 MAX_CONSEC_SKIP 이면
    job 중단), cosine LR (T_max = 이 job 의 step 수 — --tmax 로 변경 가능, eta_min 1e-6), LR 2e-4. ACCUM_STEPS = 1 (config 값).
  - 검증 지표: eval_paired_v8_nodc.slice_metrics (eval_unet_only_v8.py:45 와 같은 import) — 체크포인트 선택용 내부 점수는 버린다.
  - 수정·삭제한 저장소 파일 없음 (트레이너·config·로더·wrapper·모델·교수님 원본 무수정).

실험 조건 (--cond; 같은 seq 의 job 은 모두 같은 시드 초기값·같은 데이터 순서에서 시작 — job 마다 재시드 + DataLoader 재생성):
  A   현행 대조: Adam(lr, weight_decay=3e-5) 전 파라미터(결합형 L2) + clip 1.0 + 원 스케일(×1e6/1e4/1e6).
  B   AdamW(lr, weight_decay=3e-5, 분리형) + no-decay 그룹(bias 류, LayerNorm/BatchNorm/GroupNorm weight, SS2D A_log·D,
      `_no_weight_decay` 속성 파라미터 — ss2d.py:77,81) + clip 1.0 + 원 스케일.
      ※ 분리형 wd 3e-5 의 수축은 step 당 lr·wd = 6e-9 (4000 step 누계 2.4e-5) 라 B~E 는 사실상 wd≈0 이다.
        "보통의 AdamW decay(예 1e-2)가 50 epoch 에 안전한가" 는 이 스모크가 시험하지 않는다.
  C   B + 사실상 clip 없음 (clip_grad_norm_ max_norm=1e3 는 폭주 방지용 안전장치일 뿐; clip 전 norm 은 계속 기록).
  D   C + 샘플별 강도 정규화: s = zero-filled RSS 크기(data_img 의 16코일 복소 영상 RSS, 전체 FOV)의 99백분위수, s ≥ 1e-6.
      data_img·label 은 /s, k-space(data) 는 ×(100/s) (= /(s/100)). 로더가 data_img = 100·ifft2c(data) 로 만들므로
      (dataloader_h5_v5.py:131-132,234-235 의 ×1e6/×1e4) data_img/s = ifft2c(data·100/s) — 선형 관계가 그대로 유지된다.
      k-space 도 /s 로 나누면(리뷰 전 판) k RMS ≈5e-4 로 SS2D norm_in(LayerNorm eps 1e-5)에서 표본 화소의 99.3~99.8% 가 분산 < eps
      (원 스케일 0~20%), GRU gate 사전활성 std 0.13~0.33 → ≈5e-4 가 되어 초기값에서 이미 입력에 거의 무감했다
      (‖f(x)−f(0)‖/‖f(0)‖ GRU 2.71→0.014, SS2D 0.384→0.041). ×100/s 이면 k RMS ≈0.04~0.05, eps 미만 0~24% 로 원 스케일과 같은 영역.
      학습은 정규화 단위, 검증은 출력×s 로 원 단위에 복원해 원 label 과 같은 공식으로 계산(SSIM/PSNR/nMSE 는 스케일 불변).
      ※ D 는 단일 요인 비교가 아니다 — C→D 차이를 "정규화" 하나에 돌릴 수 없다:
        (1) 손실 균형: 원 스케일 L1≈128~156 vs SSIM 항≈1 (L1 이 손실의 ≈99%, 기울기는 unet.last 가 ≈99.7%) → 정규화 후
            L1≈0.4~0.6 vs SSIM 항≈1.0~1.1 (SSIM 이 ≈65%) — SSIM 쪽으로 ≈250배 이동.
        (2) u_choh_SSIM 은 동적 범위 L 을 출력 max 로 정한다(원 스케일 max>128 이면 255, 정규화 후 1~2) → 안정화 상수 C1·C2 의
            신호² 대비 크기가 3~7배 달라진다(C2/max² 원 스케일 3.6e-4~2.3e-3, 정규화 1.2e-4~3.4e-4).
        (3) U-Net 입력에서 zero-filled 채널과 시퀀스 채널의 크기 균형 (A–C: zf max 175~690 vs 시퀀스 출력 RMS 0.03 GRU / 0.35 SS2D).
        (4) SS2D norm_in LayerNorm 의 동작 영역.
  DL  D + λ_SSIM = λ/s̄ (s̄ = train 64 슬라이스(고르게 분산, 고정 rng)의 s 평균, 또는 --lambda-dl-scale) — C 의 L1:SSIM 균형을
      정규화 단위에서 재현. C→DL ≈ 스케일 요인(위 2~4)만, DL→D ≈ 손실 균형(위 1)만. (감사 §5 의 D 는 "λ_SSIM 재설정" 을 포함.)
  E   D + Adam eps 1e-12 — 깊은 U-Net 원소 기울기(≈2e-9)가 Adam eps 1e-8 근처라 step 이 eps 로 줄어드는지(동결) 시험하는 보험.
  ※ T_max: 기본(spec) = job step 수라 4000 step 안에 LR 2e-4 → 1e-6 로 줄고 마지막 ≈1000 step 은 LR < 3e-5 다. 따라서 이 스모크는
    50ep 런(T_max 406,450)이나 계획한 5ep 런(T_max 5×8129=40,645)의 앞부분이 아니며, 총 이동량이 실런 처음 4000 step 보다 ≈40%
    작다(잡음 표류 relΔ 0.21 vs 0.35, 기울기 RMS 3e-9). A 의 소멸 시점(|w|<1e-6 ≈250 step, <1e-30 ≈1300 step)은 T_max 와 무관.
    --tmax 40645 로 5ep 런의 앞부분과 같게 만들 수 있다.

판정 기준 (실행 전에 고정 — summary.json "verdict" 와 summary.md 에 자동 기록; 그룹 = seq / unet_deep / unet_all):
  - 소멸(collapsed):  near_dead_frac(|w|<1e-6) ≥ 0.5. (|w|<1e-30 dead_frac 은 보조 — fp32 기울기가 0 이 아니면 결합 L2 Adam 이
                       1e-8 근처에서 멈춰 dead 가 0% 로 보일 수 있다.)
  - 동결(frozen):     소멸 아님 & relΔ(‖w−w0‖/‖w0‖) < 0.01.
  - eps 제한:         위 둘 아님 & eps_dom_frac(√v̂ < Adam eps 인 원소 비율) ≥ 0.9 — 움직이지만 step 크기가 eps 로 줄어든 상태.
  - 움직임(moving):   그 외. ※ relΔ 만으로는 "학습" 과 "잡음 표류" 를 가를 수 없다(Adam 은 신호 없는 잡음 기울기에도
                       4000 step 에 relΔ 0.2~0.9). upd_ratio_mean = 평균 |m̂|/(√v̂+eps) (= |Δw|/lr; 순수 잡음 ≈0.18~0.23,
                       일관된 신호 → 1) 은 보조 지표. "쓰인다" 는 아래 기능 시험으로만 판정한다.
  - 사용(used):       기능 시험의 슬라이스별 SSIM 하락 d_i = SSIM(정상) − SSIM(시험) 에서 평균 > 0 & 평균 > 2·SE & 부호 검정
                       (동률 제외, 양측) p < 0.01. 시퀀스 모듈 = shuffle_cross, 깊은 U-Net = deep4 (down_path.4 출력 0 대체).
                       "used 아님(not shown)" 은 "쓰지 않는다" 의 증거가 아니다: 4000 step(≈0.49 epoch, LR 이 1e-6 까지 감소)의 null 은
                       결론 불가이며, 특히 A–C 는 위 (3) 의 크기 차이 때문에 짧은 학습 안에 U-Net 이 시퀀스 채널에 비중을 주기 어렵다.

출력 (--out-dir, 기본 results/smoke_recipe_fix/ — 상대 경로는 저장소 루트 기준, results/ 는 git 무시):
  <seq>_<cond>.jsonl         1행 = {"type":"meta"} 다음 {"type":"step"} 기록. 기록 시점 = step 0(초기값), 처음 --log-dense-until step
                              까지 --log-dense-every 마다(소멸 시작 시점 해상도), 이후 --log-every 마다, 마지막 step.
                              step 기록: lr(그 step 에 쓴 값), loss_total/loss_l1/loss_ssim/loss_ssim_term(구간 평균; D/DL/E 는 정규화 단위),
                              l1_over_ssim_term(구간), grad_norm_preclip(그 step)·grad_norm_interval(구간 mean/min/max),
                              clip_coef=min(1, max_norm/norm), scaler_scale, amp_skipped_steps(GradScaler overflow skip 누계),
                              nonfinite_loss_skips(누계), sec_per_step, seq_in_rms_train(시퀀스 모듈 입력 k-space RMS, 구간 평균),
                              groups{seq·seq 하위·unet.down_path.0..4·unet.up_path.0..3·unet.last·unet_deep·unet_all:
                                numel, dead_frac(|w|<1e-30), near_dead_frac(|w|<1e-6), rms, rel_change(‖w−w0‖/‖w0‖ — 전체 원소),
                                rel_change_interval(직전 기록 이후 ‖Δw‖/‖w‖ — 원소 1e6 초과 텐서는 고정 무작위 1e6 원소 표본으로 추정),
                                grad_norm(clip 전), eps_dom_frac, upd_ratio_mean (Adam 상태 — step 0 은 null)},
                              io_slices{unet_first_conv(down_path.0.block.0.weight)[:, :20]=시퀀스 채널·[:, 20:]=zero-filled 채널,
                                unet_last.weight 의 시퀀스/zero-filled/디코더 입력 slice: maxabs·dead_frac·rms},
                              probe{고정 val 2 슬라이스: seq_in_rms, zf_in_rms/max, seq_out_rms, seq_in_dep=‖f(x)−f(0)‖/‖f(0)‖,
                                seq_pair_rel=‖f(x1)−f(x2)‖/‖f(x1)‖, act_rms(U-Net 블록별 출력 RMS), out_rms,
                                deep4_out_rel(down_path.4 출력 0 대체 시 출력 상대 변화)}.
  <seq>_<cond>_summary.json  조건·optimizer 그룹·λ·T_max·처음/최종 파라미터 통계·소멸 시점·검증 결과(슬라이스별 SSIM 포함)·verdict·시간.
  summary.md                 out-dir 의 모든 *_summary.json 으로 매번 다시 만든다(중복 행 없음) — 표 2개(가중치 / 기능·검증).
  <seq>_<cond>_summary.prev.json  다시 실행한 job 의 이전 summary (job 시작 시 옮김 — jsonl 과 짝이 어긋나지 않게).
  <seq>_<cond>_end.pt        --save-ckpt end 일 때만 (model.state_dict()).
검증(job 끝): val set 전체에 고르게 퍼진 --val-slices 개 슬라이스(idx_k = floor((k+0.5)·N/n), 결정적)에서
  normal(평균 SSIM/PSNR/nMSE/L1), deep4 / deep34 (U-Net down_path.4 / down_path.3 출력을 0 으로 — 깊은 층 기능 시험, 모든 seq),
  gru/ss2d 는 추가로 zero(시퀀스 출력 0 대체), shuffle_cross(슬라이스 i 의 U-Net 에 j=(i+n/2) mod n 번째 슬라이스 — 다른 볼륨 — 의
  시퀀스 출력 + i 의 zero-filled 영상), shuffle_adj(같은 볼륨의 이웃 슬라이스 — 볼륨 수준 강도·코일 수 차이를 없앤 더 엄격한 짝).
  각 시험: 슬라이스별 SSIM, 하락의 평균 ± SE, 양수 개수, 부호 검정 p.

GPU 안전: --device cuda 는 CUDA_VISIBLE_DEVICES="0" 이 아니거나 GPU0 사용 메모리가 2 GB 를 넘으면 거부한다(--force 로 무시).
  GPU1 은 정책상 쓰지 않는다. --max-minutes 는 job 당 학습 시간 상한(초과 시 그 step 에서 멈추고 status=truncated_time 으로 검증).
  job 하나가 예외(OOM 등)로 실패하면 기록하고 메모리를 비운 뒤 다음 job 으로 간다. --skip-done 은 같은 설정(스크립트 판·steps·
  T_max·BS·seed·워커 수·λ·조건 정의)으로 끝난(status completed|truncated_time) job 만 건너뛴다(failed 는 다시 실행).
  종료 코드: 0 정상 / 2 거부(GPU0 사용 중·정책) / 3 일부 job 실패 / 4 CUDA 사용 불가(NVML EPERM 등 — 러너가 재시도).
CPU 모드(--device cpu, 건식 실행용 — CUDA_VISIBLE_DEVICES="" 필수, GPU0 드라이버를 건드리지 않기 위해): u_choh_SSIM 이 .cuda() 를
  하드코딩하므로 이 프로세스에서만 torch.Tensor.cuda 를 항등으로 바꾸고, SS2D 는 mamba_ssm CUDA 커널을 selective_scan_cpu(아래 —
  참조 구현 selective_scan_ref 와 같은 점화식, 역전파만 효율화)로 바꾼다(visualize_multimodel_compare.py:128 과 같은 이름 교체).
  CPU 모드는 코드 경로 점검용이지 측정용이 아니다. autocast·GradScaler 는 CPU 에서 끈다(fp32). 워커 수 기본값은 CPU 에서 2/1.

실행 예:
  # 전체 (GPU0 precheck 포함 러너, job 마다 새 프로세스, 로컬 전용 *.sh — 12 spec job 다음 추가 job unet:DL,unet:E,gru:DL,ss2d:DL):
  setsid nohup bash v8_eter_pure/runs/run_smoke_recipe_fix.sh > /dev/null 2>&1 < /dev/null & disown
  # 위 러너가 job 하나마다 실행하는 명령:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python v8_eter_pure/smoke_recipe_fix.py --device cuda \\
      --jobs "gru:C" --steps 4000 --log-every 100 --log-dense-every 20 --log-dense-until 500 --val-slices 64 --seed 1 \\
      --out-dir results/smoke_recipe_fix --save-ckpt none --max-minutes 180 --skip-done
  # 단일 job:
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/smoke_recipe_fix.py --seq gru --cond B --steps 4000
  # CPU 건식 실행 (GPU 를 건드리지 않음):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 19 python v8_eter_pure/smoke_recipe_fix.py --device cpu \\
      --jobs "unet:A,unet:D,ss2d:B" --steps 3 --log-every 1 --val-slices 2 --bs 1 --out-dir /tmp/smoke_dry
"""

import os
import sys
import gc
import json
import math
import time
import random
import argparse
import datetime
import subprocess
import traceback
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(_HERE)
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# 지표 공식·config 는 per-slice 평가 스크립트의 것을 그대로 쓴다 (eval_unet_only_v8.py:45 와 같은 import;
# import 부작용 = sys.path 추가·PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 설정뿐 — CUDA 초기화 전이라 유효).
from eval_paired_v8_nodc import C, slice_metrics
from dataloader_h5_v5 import FastMRI_H5_Dataloader
from u_choh_SSIM import SSIM          # main_train_pure_v8.py:40 (import 시 cudnn.deterministic=True — 트레이너와 같음)

SCRIPT_VERSION = '2026-10-07b'        # --skip-done 비교용 (이전 판의 결과는 다시 실행)
DATA_TRAIN = os.path.join(_PROJECT_ROOT, 'fastMRI_data', 'multicoil_train')   # main_train_pure_v8.py:74
DATA_VAL   = os.path.join(_PROJECT_ROOT, 'fastMRI_data', 'multicoil_val')     # main_train_pure_v8.py:75

SEQS = ('unet', 'gru', 'ss2d')
_ADAMW = 'AdamW(decoupled, no-decay group)'
CONDS = OrderedDict([
    ('A',  dict(optimizer='Adam(coupled L2, all params)', decoupled=False, max_norm=1.0, normalize=False,
                lam='base', adam_eps=1e-8)),
    ('B',  dict(optimizer=_ADAMW, decoupled=True, max_norm=1.0, normalize=False, lam='base', adam_eps=1e-8)),
    ('C',  dict(optimizer=_ADAMW, decoupled=True, max_norm=1e3, normalize=False, lam='base', adam_eps=1e-8)),
    ('D',  dict(optimizer=_ADAMW, decoupled=True, max_norm=1e3, normalize=True,  lam='base', adam_eps=1e-8)),
    ('DL', dict(optimizer=_ADAMW, decoupled=True, max_norm=1e3, normalize=True,  lam='matched', adam_eps=1e-8)),
    ('E',  dict(optimizer=_ADAMW, decoupled=True, max_norm=1e3, normalize=True,  lam='base', adam_eps=1e-12)),
])
SEQ_CH = 2 * C.N_HIDDEN_LRNN_2                 # 시퀀스 출력 채널 = 20 (U-Net 입력 앞쪽)
IN_CH = SEQ_CH + 2 * C.N_COIL                  # U-Net 원입력 = 52 (뒤쪽 32 = zero-filled 영상)
K_RATIO = float(C.DC_K_SCALE_RATIO)            # data_img / ifft2c(data) = 1e6/1e4 = 100 (main 에서 로더 속성과 대조)
DEAD_THR, NEAR_THR = 1e-30, 1e-6
STAT_CHUNK = 1 << 25                           # 큰 파라미터(GRU W_ih 141M)는 32M 원소씩 나눠 통계 — GPU 임시 메모리 상한
SUB_N = 1_000_000                              # 구간 변화량용 고정 표본 (원소 수가 이보다 큰 텐서만 표본, 나머지는 전체)
SUB_SEED = 20261007
DEEP_PREFIXES = ('unet.down_path.3.', 'unet.down_path.4.', 'unet.up_path.0.')   # 감사에서 ep5 에 100% 소멸한 깊은 층
LAMBDA_DL_N = 64                               # DL 의 s̄ 추정에 쓰는 train 슬라이스 수
ACCUM_STEPS = 1
assert C.ACCUM_STEPS == ACCUM_STEPS, 'config ACCUM_STEPS 가 1 이 아니면 이 스모크의 step 정의를 다시 맞춰야 한다'
# 판정 기준 (docstring "판정 기준" 과 같은 값 — 실행 전에 고정)
TH_COLLAPSED_NEAR = 0.5
TH_FROZEN_REL = 0.01
TH_EPS_DOM = 0.9
TH_USED_P = 0.01


# ------------------------------------------------------------------ 환경·안전
EXIT_REFUSED = 2          # 거부(GPU0 사용 중·정책 위반) — 러너는 중단
EXIT_JOB_FAILED = 3       # 일부 job 실패(summary 에 기록됨) — 러너는 새 프로세스로 1회 재시도
EXIT_NO_CUDA = 4          # CUDA 사용 불가(NVML EPERM 재발 등 일시 장애) — 러너가 대기 후 재시도


def refuse(msg, code=EXIT_REFUSED):
    print(f'[refuse] {msg}', flush=True)
    raise SystemExit(code)


def gpu0_precheck(force):
    """GPU0 단독 정책 + 2 GB 초과 사용 중이면 거부 (CUDA 초기화 전에 nvidia-smi 로 확인, 실패 시 mem_get_info)."""
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cvd is None or cvd.strip() != '0':
        msg = f'CUDA_VISIBLE_DEVICES={cvd!r} — GPU0 단독 정책상 "0" 만 허용'
        if not force:
            refuse(f'{msg} (무시하려면 --force)')
        print(f'[warn] {msg} (--force)')
    used = None
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', '0'],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            used = float(r.stdout.strip().splitlines()[0])
        else:
            print(f'[precheck] nvidia-smi 실패 (rc={r.returncode}): {r.stderr.strip()[:200]} → torch.cuda.mem_get_info 로 확인')
    except Exception as e:                                       # noqa: BLE001
        print(f'[precheck] nvidia-smi 실행 불가: {e} → torch.cuda.mem_get_info 로 확인')
    if used is None:
        if not torch.cuda.is_available():
            refuse('CUDA 사용 불가 (NVML/장치 접근 실패 가능 — 러너가 재시도)', EXIT_NO_CUDA)
        free, total = torch.cuda.mem_get_info(0)
        used = (total - free) / 2 ** 20
    print(f'[precheck] GPU0 사용 메모리 {used:.0f} MiB')
    if used > 2048:
        if not force:
            refuse(f'GPU0 에 이미 {used:.0f} MiB 사용 중 (> 2 GB) — 다른 런이 도는 중. 무시하려면 --force')
        print('[warn] GPU0 사용 중이지만 --force 로 진행')
    if not torch.cuda.is_available():
        refuse('CUDA 사용 불가 (NVML/장치 접근 실패 가능 — 러너가 재시도)', EXIT_NO_CUDA)


def _cuda_noop(self, *args, **kwargs):
    return self


def selective_scan_cpu(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """CPU 건식 실행 전용 selective scan — mamba_ssm selective_scan_ref 와 같은 점화식
    (x_t = exp(Δ_t A)·x_{t−1} + Δ_t B_t u_t,  y_t = C_t·x_t + D u_t), ss2d.py:101 의 호출 형태(실수 A, B·C = (B,N,L), z 없음)만 지원.
    selective_scan_ref 는 루프에서 deltaA[:, :, i] 로 큰 4D 텐서를 인덱싱해 역전파 때 step 마다 전체 크기 기울기를 만든다
    (384²·128·16 원소 × 384 step × 4 방향 → CPU 학습 1 step 에 20 분 이상). 여기서는 같은 텐서를 unbind 로 한 번에 나눠
    역전파가 한 번의 stack 이 되게 했을 뿐 수식은 같다(작은 입력에서 출력·기울기 일치를 확인함)."""
    assert z is None and delta_bias is None and not delta_softplus and not return_last_state
    assert not A.is_complex() and B.dim() == 3 and C.dim() == 3
    dtype_in = u.dtype
    u, delta, B, C = u.float(), delta.float(), B.float(), C.float()
    deltaA = torch.exp(torch.einsum('bdl,dn->lbdn', delta, A)).contiguous()        # (L, b, d, n)
    deltaB_u = torch.einsum('bdl,bnl,bdl->lbdn', delta, B, u).contiguous()
    x = A.new_zeros((u.shape[0], A.shape[0], A.shape[1]))
    ys = []
    for dA, dBu, c in zip(deltaA.unbind(0), deltaB_u.unbind(0), C.unbind(2)):
        x = dA * x + dBu
        ys.append(torch.einsum('bdn,bn->bd', x, c))
    y = torch.stack(ys, dim=2)                                                     # (b, d, L)
    out = y if D is None else y + u * D.float()[:, None]
    return out.to(dtype=dtype_in)


def install_cpu_shims(need_ss2d):
    """CPU 건식 실행 전용 (이 프로세스 안에서만). 원본 파일은 건드리지 않는다."""
    # u_choh_SSIM.py:19,43,45,46,119 가 .cuda() 를 하드코딩 → 같은 수식을 CPU 에서 돌리기 위해 Tensor.cuda 를 항등으로.
    torch.Tensor.cuda = _cuda_noop
    print('  [CPU] torch.Tensor.cuda → 항등 (u_choh_SSIM 의 하드코딩 .cuda() 우회, 수식 동일)')
    if need_ss2d:
        import ss2d as _ss2d
        _ss2d.selective_scan_fn = selective_scan_cpu           # visualize_multimodel_compare.py:128-138 과 같은 이름 교체 방식
        print('  [CPU] ss2d.selective_scan_fn → selective_scan_cpu (selective_scan_ref 와 같은 점화식, 역전파만 효율화)')


# ------------------------------------------------------------------ 시드·데이터
def seed_all(seed):
    """main_train_pure_v8.py:61-64."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init_fn(worker_id):
    """main_train_pure_v8.py:243-248 그대로 (워커별 독립 rng — torch 가 generator 로 정한 워커 base seed 사용)."""
    ws = torch.initial_seed() % 2 ** 32
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(ws)
    np.random.seed(ws)
    random.seed(ws)


def build_train_dataset():
    """main_train_pure_v8.py:229-232 (random_mask 기본 True = 랜덤 offset R4 마스크, flip augment)."""
    return FastMRI_H5_Dataloader(
        DATA_TRAIN, num_files=None, target_size=C.IMAGE_SIZE[0],
        augment=C.TRAIN_AUGMENT, augment_flip_p=C.TRAIN_AUGMENT_FLIP_P,
    )


def build_val_dataset():
    """main_train_pure_v8.py:255-258 (고정 마스크, augment 없음; NUM_VAL_FILES=None → 전체 val)."""
    return FastMRI_H5_Dataloader(
        DATA_VAL, num_files=None, target_size=C.IMAGE_SIZE[0],
        random_mask=False, augment=False,
    )


def make_train_loader(train_ds, bs, seed, num_workers, pin):
    """main_train_pure_v8.py:233-253 의 SEED 경로. job 마다 새로 만들어 같은 데이터 순서에서 시작한다."""
    train_ds.rng = np.random.default_rng(seed + 1)                 # :241
    kw = dict(batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin)   # :233-234
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=C.PREFETCH_FACTOR)         # :235-236
    g = torch.Generator()
    g.manual_seed(seed)                                             # :250-251
    kw.update(worker_init_fn=_worker_init_fn, generator=g)          # :252
    return DataLoader(train_ds, **kw)


def even_indices(n_total, n):
    """전체에 고르게 퍼진 결정적 슬라이스 번호: floor((k+0.5)·N/n)."""
    n = max(1, min(n, n_total))
    return sorted(set(int((k + 0.5) * n_total / n) for k in range(n)))


def iter_val(val_ds, indices, device, workers, pin):
    loader = DataLoader(Subset(val_ds, list(indices)), batch_size=1, shuffle=False,
                        num_workers=workers, pin_memory=pin)
    for idx, sample in zip(indices, loader):
        yield idx, {k: v.float().to(device) for k, v in sample.items()}


def adjacent_partner(val_ds, idx):
    """같은 볼륨(같은 .h5)의 이웃 슬라이스 (idx+1 우선, 없으면 idx−1, 둘 다 다른 볼륨이면 None)."""
    f = val_ds.samples[idx][0]
    for j in (idx + 1, idx - 1):
        if 0 <= j < len(val_ds) and val_ds.samples[j][0] == f:
            return j
    return None


def per_sample_scale(data_img):
    """조건 D/DL/E 의 s: data_img(32ch = 16코일 real/imag 교번) 의 RSS 크기(전체 FOV) 99백분위수, 샘플별, s ≥ 1e-6. (B,1,1,1)."""
    x = data_img.float()
    rss = torch.sqrt((x * x).sum(dim=1))                            # (B, H, W) = sqrt(Σ_c |z_c|²)
    s = torch.quantile(rss.reshape(rss.shape[0], -1), 0.99, dim=1)
    return s.clamp(min=1e-6).view(-1, 1, 1, 1)


def normalize_inputs(x_ksp, x_img, s):
    """k-space ×(K_RATIO/s), 영상 /s — data_img = K_RATIO·ifft2c(data) 이므로 정규화 후에도 x_img = ifft2c(x_ksp) 가 유지된다."""
    return x_ksp * (K_RATIO / s), x_img / s


def estimate_mean_scale(train_ds, seed, n=LAMBDA_DL_N):
    """DL 의 s̄: train 에서 고르게 퍼진 n 슬라이스의 s 평균 (마스크 offset·flip 은 고정 rng — 결정적). 끝나면 rng 를 되돌린다
    (어차피 make_train_loader 가 job 마다 rng = SEED+1 로 다시 설정)."""
    old = train_ds.rng
    train_ds.rng = np.random.default_rng(seed + 7919)
    ss = []
    for i in even_indices(len(train_ds), n):
        d = train_ds[i]
        ss.append(float(per_sample_scale(torch.from_numpy(np.asarray(d['data_img']))[None])))
    train_ds.rng = old
    return dict(mean=float(np.mean(ss)), median=float(np.median(ss)), min=float(np.min(ss)), max=float(np.max(ss)), n=len(ss))


# ------------------------------------------------------------------ 모델·optimizer
def build_model(seq, device):
    """main_train_pure_v8.py:104-149 의 gru / unet / ss2d 분기 (USE_DC=0 → use_dc=False, 같은 config 인자)."""
    if seq == 'gru':
        from u_pure_eternet_gru import PureETER_GRU
        model = PureETER_GRU(
            dim=C.IMAGE_SIZE[0], n_coil=C.N_COIL,
            n_hidden_1=C.N_HIDDEN_LRNN_1, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            use_dc=False, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        )
    elif seq == 'unet':
        from u_pure_eternet_unet import PureETER_UNET
        model = PureETER_UNET(
            n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            use_dc=False, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        )
    elif seq == 'ss2d':
        from u_pure_eternet_ss2d import PureETER_SS2D
        model = PureETER_SS2D(
            n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            ss2d_d_inner=C.SS2D_D_INNER, ss2d_d_state=C.SS2D_D_STATE,
            use_dc=False, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        )
    else:
        raise ValueError(seq)
    return model.to(device)


def seq_forward(model, seq, x_ksp):
    """시퀀스 모듈 출력 (B, 20, H, W) — wrapper forward 의 앞부분과 같은 호출 (u_pure_eternet_gru.py:108, _ss2d.py:75)."""
    if seq == 'gru':
        return model._seq(x_ksp)
    if seq == 'ss2d':
        return model.ss2d(x_ksp)
    raise ValueError(seq)


def unet_input(model, seq, x_ksp, x_img):
    """U-Net 원입력 cat(시퀀스 출력, zero-filled) — gru/ss2d wrapper forward 와 같은 순서; unet 은 0 채널
    (u_pure_eternet_unet.py:86-88 과 같은 dtype·순서). 반환: (unet_in, seq_out 또는 None)."""
    if seq in ('gru', 'ss2d'):
        so = seq_forward(model, seq, x_ksp)
    else:
        B, _, H, W = x_img.shape
        so = None
        z = torch.zeros((B, SEQ_CH, H, W), dtype=x_img.dtype, device=x_img.device)
        return torch.cat((z, x_img), dim=1), None
    return torch.cat((so, x_img), dim=1), so


def _zero_hook(mod, inp, out):
    return torch.zeros_like(out)


def unet_with_zeroed(model, block, unet_in):
    """block(U-Net 하위 모듈)의 출력을 0 으로 바꾼 U-Net forward (기능 시험)."""
    h = block.register_forward_hook(_zero_hook)
    try:
        return model.unet(unet_in)
    finally:
        h.remove()


_NORM_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)


def split_decay(model):
    """AdamW 의 decay / no-decay 분리. no-decay = `_no_weight_decay` 속성(ss2d.py:77,81) · norm 층 weight/bias ·
    bias 류(Conv/Linear 'bias', GRU 'bias_ih_l0'·'bias_hh_l0[_reverse]') · SS2D A_log·D (이름 일치)."""
    decay, no_decay, reasons, names_nd = [], [], {}, []
    seen = set()
    for mname, mod in model.named_modules():
        for pname, p in mod.named_parameters(recurse=False):
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            full = f'{mname}.{pname}' if mname else pname
            if getattr(p, '_no_weight_decay', False):
                why = '_no_weight_decay'
            elif isinstance(mod, _NORM_TYPES):
                why = 'norm'
            elif pname.startswith('bias'):
                why = 'bias'
            elif pname in ('A_log', 'D'):
                why = 'ssm_A_log_D'
            else:
                why = None
            if why is None:
                decay.append(p)
            else:
                no_decay.append(p)
                names_nd.append(full)
                r = reasons.setdefault(why, [0, 0])
                r[0] += 1
                r[1] += p.numel()
    n_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay) == n_all, 'decay 분리 누락'
    return decay, no_decay, reasons, names_nd


def build_optimizer(model, cond):
    lr, wd = C.LEARNING_RATE_ADAM, C.LAMBDA_REGULAR_PER_PIXEL
    eps = CONDS[cond]['adam_eps']
    if not CONDS[cond]['decoupled']:
        # A = 현행: main_train_pure_v8.py:225-226 그대로 (결합형 L2, 전 파라미터, `_no_weight_decay` 무시, eps 기본 1e-8)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        info = dict(type='Adam', lr=lr, weight_decay=wd, coupled_l2=True, eps=opt.defaults['eps'],
                    decay_numel=sum(p.numel() for p in model.parameters()), no_decay_numel=0)
        return opt, info
    decay, no_decay, reasons, names_nd = split_decay(model)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': wd},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=lr, eps=eps)
    info = dict(type='AdamW', lr=lr, weight_decay=wd, coupled_l2=False, eps=eps,
                decay_numel=sum(p.numel() for p in decay), no_decay_numel=sum(p.numel() for p in no_decay),
                no_decay_reasons={k: {'tensors': v[0], 'numel': v[1]} for k, v in reasons.items()},
                no_decay_names=names_nd)
    return opt, info


# ------------------------------------------------------------------ 파라미터 통계
def build_groups(model, seq):
    """통계 그룹 → 파라미터 이름 목록 (겹쳐도 됨 — 파라미터별 통계를 한 번 계산해 합산)."""
    names = [n for n, _ in model.named_parameters()]
    g = OrderedDict()

    def add(key, pred):
        lst = [n for n in names if pred(n)]
        if lst:
            g[key] = lst

    if seq == 'gru':
        add('seq', lambda n: n.startswith(('gru_h.', 'gru_v.')))
        add('seq.gru_h', lambda n: n.startswith('gru_h.'))
        add('seq.gru_v', lambda n: n.startswith('gru_v.'))
    elif seq == 'ss2d':
        add('seq', lambda n: n.startswith('ss2d.'))
        add('seq.ss2d_in', lambda n: n.startswith(('ss2d.norm_in.', 'ss2d.in_proj.', 'ss2d.dwconv.')))
        add('seq.ss2d_ssm', lambda n: n.startswith('ss2d.ssm_'))
        add('seq.ss2d_out', lambda n: n.startswith(('ss2d.merge_norm.', 'ss2d.merge.', 'ss2d.out_proj.')))
    for i in range(C.UNET_DEPTH):
        add(f'unet.down_path.{i}', lambda n, i=i: n.startswith(f'unet.down_path.{i}.'))
    for i in range(C.UNET_DEPTH - 1):
        add(f'unet.up_path.{i}', lambda n, i=i: n.startswith(f'unet.up_path.{i}.'))
    add('unet.last', lambda n: n.startswith('unet.last.'))
    add('unet_deep', lambda n: n.startswith(DEEP_PREFIXES))
    add('unet_all', lambda n: n.startswith('unet.'))
    covered = set(sum(g.values(), []))
    missing = [n for n in names if n not in covered]
    assert not missing, f'통계 그룹에 빠진 파라미터: {missing}'
    return g


def snapshot_init(model, device):
    """w0 (상대 변화량 기준, 전체 원소). CUDA 면 pinned CPU 메모리 — GRU 637M(2.5 GB)를 GPU 에 두지 않기 위해."""
    w0, w0sq = {}, {}
    for n, p in model.named_parameters():
        if device.type == 'cuda':
            t = torch.empty(p.shape, dtype=p.dtype, pin_memory=True)
            t.copy_(p.detach())
        else:
            t = p.detach().clone()
        w0[n] = t
        w0sq[n] = float(t.float().pow(2).sum())
    return w0, w0sq


class SubsetTracker:
    """직전 기록 이후의 변화량 ‖w_t − w_prev‖ 추정. 원소 수 > SUB_N 인 텐서는 고정 무작위 SUB_N 원소(복원 추출, 전용 generator —
    전역 RNG 무영향), 나머지는 전체 원소. 인덱스·이전 값은 CPU 에 둔다(GPU 상주 메모리 0). 합은 numel/표본수로 배율 보정."""

    def __init__(self, model):
        g = torch.Generator()
        g.manual_seed(SUB_SEED)
        self.idx, self.scale, self.prev = {}, {}, {}
        for n, p in model.named_parameters():
            N = p.numel()
            if N > SUB_N:
                self.idx[n] = torch.randint(0, N, (SUB_N,), generator=g)
                self.scale[n] = N / SUB_N
            else:
                self.idx[n] = None
                self.scale[n] = 1.0
            self.prev[n] = self._take(n, p)

    def _take(self, n, p):
        a = p.detach().reshape(-1)
        i = self.idx[n]
        if i is not None:
            a = a[i.to(a.device, non_blocking=True)]
        return a.float().cpu().clone()

    @torch.no_grad()
    def update(self, n, p):
        cur = self._take(n, p)
        prev = self.prev[n]
        psq = float(prev.pow(2).sum()) * self.scale[n]
        dsq = float((cur - prev).pow(2).sum()) * self.scale[n]
        self.prev[n] = cur
        return psq, dsq


@torch.no_grad()
def adam_param_stats(state, eps, betas):
    """Adam 상태 진단: eps 지배(√v̂ < eps) 원소 수, Σ |m̂|/(√v̂+eps) (= 원소별 |Δw|/lr). 상태 없으면 None."""
    if not state or 'exp_avg_sq' not in state:
        return None
    t = float(state['step'])
    if t <= 0:
        return None
    bc1 = 1.0 - betas[0] ** t
    bc2 = 1.0 - betas[1] ** t
    m = state['exp_avg'].reshape(-1)
    v = state['exp_avg_sq'].reshape(-1)
    epsdom, upd = 0, 0.0
    for m_c, v_c in zip(m.split(STAT_CHUNK), v.split(STAT_CHUNK)):
        den = (v_c.float() / bc2).sqrt_()
        epsdom += int((den < eps).sum())
        upd += float((m_c.float().abs().div_(bc1)).div_(den.add_(eps)).sum())
        del den
    return epsdom, upd


@torch.no_grad()
def per_param_stats(model, w0, optimizer, tracker):
    hp = {}
    for grp in optimizer.param_groups:
        for p in grp['params']:
            hp[id(p)] = (grp['eps'], grp['betas'])
    st = {}
    for n, p in model.named_parameters():
        a = p.detach().reshape(-1)
        b = w0[n].reshape(-1)
        dead = near = 0
        sq = dsq = 0.0
        for a_c, b_c in zip(a.split(STAT_CHUNK), b.split(STAT_CHUNK)):
            b_c = b_c.to(a_c.device, non_blocking=True).float()
            a_f = a_c.float()
            aa = a_f.abs()
            dead += int((aa < DEAD_THR).sum())
            near += int((aa < NEAR_THR).sum())
            sq += float(a_f.pow(2).sum())
            dsq += float((a_f - b_c).pow(2).sum())
            del aa, a_f, b_c
        isq, idsq = tracker.update(n, p)
        eps, betas = hp[id(p)]
        ad = adam_param_stats(optimizer.state.get(p), eps, betas)
        st[n] = dict(numel=a.numel(), dead=dead, near=near, sq=sq, dsq=dsq, gsq=None, isq=isq, idsq=idsq,
                     epsdom=None if ad is None else ad[0], upd=None if ad is None else ad[1])
    return st


def aggregate(st, groups, w0sq):
    out = OrderedDict()
    for key, names in groups.items():
        numel = sum(st[n]['numel'] for n in names)
        sq = sum(st[n]['sq'] for n in names)
        dsq = sum(st[n]['dsq'] for n in names)
        s0 = sum(w0sq[n] for n in names)
        isq = sum(st[n]['isq'] for n in names)
        idsq = sum(st[n]['idsq'] for n in names)
        gs = [st[n]['gsq'] for n in names if st[n]['gsq'] is not None]
        ad = [n for n in names if st[n]['epsdom'] is not None]
        ad_numel = sum(st[n]['numel'] for n in ad)
        out[key] = dict(
            numel=numel,
            dead_frac=sum(st[n]['dead'] for n in names) / numel,
            near_dead_frac=sum(st[n]['near'] for n in names) / numel,
            rms=math.sqrt(sq / numel),
            rel_change=(math.sqrt(dsq) / math.sqrt(s0)) if s0 > 0 else None,
            rel_change_interval=(math.sqrt(idsq) / math.sqrt(isq)) if isq > 0 else None,
            grad_norm=math.sqrt(sum(gs)) if gs else None,
            eps_dom_frac=(sum(st[n]['epsdom'] for n in ad) / ad_numel) if ad_numel else None,
            upd_ratio_mean=(sum(st[n]['upd'] for n in ad) / ad_numel) if ad_numel else None,
        )
    return out


@torch.no_grad()
def io_slice_stats(model):
    """U-Net 이 원입력(시퀀스 20ch + zero-filled 32ch)을 읽는 두 곳의 입력 채널 slice (u_pure_eternet_unet.py:17-21)."""
    def s(t):
        t = t.float()
        return dict(maxabs=float(t.abs().max()), rms=float(t.pow(2).mean().sqrt()),
                    dead_frac=float((t.abs() < DEAD_THR).float().mean()))
    W = model.unet.down_path[0].block[0].weight.detach()        # (64, 52, 3, 3)
    L = model.unet.last.weight.detach()                         # (1, 52+64, 1, 1) — cat([원입력, 디코더]) (myUNet_DF.py:193)
    return {
        'unet_first_conv': {'seq_in': s(W[:, :SEQ_CH]), 'zf_in': s(W[:, SEQ_CH:])},
        'unet_last': {'seq_in': s(L[:, :SEQ_CH]), 'zf_in': s(L[:, SEQ_CH:IN_CH]), 'dec_in': s(L[:, IN_CH:])},
    }


def _rms(t):
    t = t.detach().float()
    return float(torch.linalg.vector_norm(t) / math.sqrt(max(1, t.numel())))


def _norm(t):
    return float(torch.linalg.vector_norm(t.detach().float()))


def load_probe(val_ds, idxs):
    """고정 probe 슬라이스 (CPU 텐서, 배치 차원 1)."""
    out = []
    for i in idxs:
        d = val_ds[i]
        out.append({k: torch.from_numpy(np.ascontiguousarray(v)).float()[None] for k, v in d.items()})
    return out


@torch.no_grad()
def probe_stats(model, seq, probe, normalize, device, is_cuda):
    """고정 val probe 에서 시퀀스 모듈의 입력 의존성·표본 간 차이, U-Net 블록별 활성 RMS, 깊은 층 0 대체 시 출력 변화.
    학습과 같은 autocast·같은 정규화. eval 모드(드롭아웃 없음 — 전역 RNG 무소비) 후 train 모드로 되돌린다."""
    model.eval()
    has_seq = seq in ('gru', 'ss2d')
    blocks = [(f'down_path.{i}', m) for i, m in enumerate(model.unet.down_path)] + \
             [(f'up_path.{i}', m) for i, m in enumerate(model.unet.up_path)]
    acc = OrderedDict()

    def put(k, v):
        acc.setdefault(k, []).append(v)

    seq_outs = []
    for b in probe:
        x_ksp, x_img = b['data'].to(device), b['data_img'].to(device)
        if normalize:
            x_ksp, x_img = normalize_inputs(x_ksp, x_img, per_sample_scale(x_img))
        put('seq_in_rms', _rms(x_ksp))
        put('zf_in_rms', _rms(x_img))
        put('zf_in_max', float(x_img.abs().max()))
        with torch.amp.autocast('cuda', enabled=is_cuda):
            unet_in, so = unet_input(model, seq, x_ksp, x_img)
            if has_seq:
                f0 = seq_forward(model, seq, torch.zeros_like(x_ksp))
                put('seq_out_rms', _rms(so))
                put('seq_in_dep', _norm(so.float() - f0.float()) / max(_norm(f0), 1e-30))
                seq_outs.append(so.float().cpu())
            acts = {}
            hooks = [m.register_forward_hook(lambda mod, i, o, k=k: acts.__setitem__(k, _rms(o))) for k, m in blocks]
            try:
                out = model.unet(unet_in)
            finally:
                for h in hooks:
                    h.remove()
            out_d4 = unet_with_zeroed(model, model.unet.down_path[C.UNET_DEPTH - 1], unet_in)
        for k, v in acts.items():
            put(f'act_rms.{k}', v)
        put('out_rms', _rms(out))
        put('deep4_out_rel', _norm(out_d4.float() - out.float()) / max(_norm(out), 1e-30))
    res = OrderedDict()
    act = OrderedDict()
    for k, v in acc.items():
        if k.startswith('act_rms.'):
            act[k[len('act_rms.'):]] = float(np.mean(v))
        else:
            res[k] = float(np.mean(v))
    res['act_rms'] = act
    if len(seq_outs) >= 2:
        res['seq_pair_rel'] = _norm(seq_outs[0] - seq_outs[1]) / max(_norm(seq_outs[0]), 1e-30)
    model.train()
    return res


# ------------------------------------------------------------------ 검증
def _mean_metrics(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ('ssim', 'psnr', 'nmse', 'l1')} if rows else None


def sign_test_p(n_pos, n_neg):
    """양측 부호 검정 (동률 제외)."""
    n = n_pos + n_neg
    if n == 0:
        return None
    k = max(n_pos, n_neg)
    tail = sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def drop_stats(base, test):
    """슬라이스별 SSIM 하락 d = base − test (test 가 None 인 슬라이스 제외) 의 평균 ± SE·양수 개수·부호 검정."""
    d = np.array([b - t for b, t in zip(base, test) if t is not None], dtype=np.float64)
    if d.size == 0:
        return None
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else None
    return OrderedDict(mean=float(d.mean()), se=se, n=int(d.size), n_pos=n_pos, n_neg=n_neg,
                       p_sign=sign_test_p(n_pos, n_neg))


def used_verdict(ds):
    if ds is None:
        return None
    ok = (ds['mean'] > 0 and ds['se'] is not None and ds['mean'] > 2 * ds['se']
          and ds['p_sign'] is not None and ds['p_sign'] < TH_USED_P)
    return 'used' if ok else 'not shown'


@torch.no_grad()
def run_validation(model, seq, normalize, val_ds, indices, device, workers, pin, is_cuda):
    """고정 슬라이스에서 정상 / deep4 / deep34 (+ gru/ss2d: 0 대체 / shuffle_cross / shuffle_adj).
    지표 = eval_paired_v8_nodc.slice_metrics (원 단위; D/DL/E 는 출력×s)."""
    model.eval()
    has_seq = seq in ('gru', 'ss2d')
    indices = list(indices)
    n, N = len(indices), len(val_ds)
    partners = OrderedDict()
    if has_seq:
        if n >= 2:
            partners['shuffle_cross'] = {indices[i]: indices[(i + n // 2) % n] for i in range(n)}
        else:
            partners['shuffle_cross'] = {indices[0]: (indices[0] + N // 2) % N}
        partners['shuffle_adj'] = {i: adjacent_partner(val_ds, i) for i in indices}
    need = set(j for p in partners.values() for j in p.values() if j is not None)
    extra = sorted(need - set(indices))
    extra_set = set(extra)
    tests = ['normal', 'deep4', 'deep34'] + (['zero'] + list(partners) if has_seq else [])
    ssim = {t: {} for t in tests}
    rows_normal = []
    seq_store, out_store = {}, {}
    equiv = None
    autocast = lambda: torch.amp.autocast('cuda', enabled=is_cuda)   # noqa: E731 (main_train_pure_v8.py:165)
    for idx, b in iter_val(val_ds, indices + extra, device, workers, pin):
        x_ksp, x_img, ref, bm = b['data'], b['data_img'], b['label'], b['brain_mask']
        s = per_sample_scale(x_img) if normalize else None
        if normalize:
            x_ksp, x_img = normalize_inputs(x_ksp, x_img, s)
        back = (lambda o: o.float() * s) if normalize else (lambda o: o.float())
        with autocast():
            unet_in, so = unet_input(model, seq, x_ksp, x_img)
            if has_seq:
                seq_store[idx] = so.detach().to('cpu')
                if idx in extra_set:
                    continue
            out = model.unet(unet_in)
            if equiv is None:                                   # 분해 경로 == model.forward 확인 (1회)
                out_full = model(x_img, x_ksp, b['mask'], b['sens'])
                equiv = float((out_full.float() - out.float()).abs().max())
            out_d4 = unet_with_zeroed(model, model.unet.down_path[C.UNET_DEPTH - 1], unet_in)
            out_d34 = unet_with_zeroed(model, model.unet.down_path[C.UNET_DEPTH - 2], unet_in)
            out_z = model.unet(torch.cat((torch.zeros_like(so), x_img), dim=1)) if has_seq else None
        m = slice_metrics(back(out), ref, bm)
        rows_normal.append(m)
        ssim['normal'][idx] = m['ssim']
        ssim['deep4'][idx] = slice_metrics(back(out_d4), ref, bm)['ssim']
        ssim['deep34'][idx] = slice_metrics(back(out_d34), ref, bm)['ssim']
        if has_seq:
            ssim['zero'][idx] = slice_metrics(back(out_z), ref, bm)['ssim']
            out_store[idx] = out.float().cpu()
    res = OrderedDict(n=n, indices=indices, normal=_mean_metrics(rows_normal), decomp_equiv_maxabs=equiv)
    extras = OrderedDict()
    if has_seq:
        acc = {t: dict(rel_out=[], rel_seq=[]) for t in partners}
        sstd, srms = [], []
        for idx, b in iter_val(val_ds, indices, device, workers, pin):
            x_img, ref, bm = b['data_img'], b['label'], b['brain_mask']
            s = per_sample_scale(x_img) if normalize else None
            if normalize:
                x_img = x_img / s
            si = seq_store[idx].float()
            sstd.append(float(si.std(dim=(-2, -1)).mean()))
            srms.append(_rms(si))
            for t, pm in partners.items():
                j = pm.get(idx)
                if j is None:
                    ssim[t][idx] = None
                    continue
                with autocast():
                    out_sh = model.unet(torch.cat((seq_store[j].to(device), x_img), dim=1))
                o = out_sh.float() * s if normalize else out_sh.float()
                ssim[t][idx] = slice_metrics(o, ref, bm)['ssim']
                o_n = out_store[idx]
                acc[t]['rel_out'].append(float((out_sh.float().cpu() - o_n).abs().max() / o_n.abs().max().clamp(min=1e-12)))
                sj = seq_store[j].float()
                acc[t]['rel_seq'].append(float((si - sj).norm() / si.norm().clamp(min=1e-30)))
        for t in partners:
            extras[t] = OrderedDict(
                out_rel_maxdiff_max=float(np.max(acc[t]['rel_out'])) if acc[t]['rel_out'] else None,
                out_rel_maxdiff_mean=float(np.mean(acc[t]['rel_out'])) if acc[t]['rel_out'] else None,
                seq_out_pair_rel_diff_mean=float(np.mean(acc[t]['rel_seq'])) if acc[t]['rel_seq'] else None,  # 0 이면 입력 무관
            )
        res['seq_out_spatial_std_mean'] = float(np.mean(sstd))   # 채널별 공간 std 평균 — 0 이면 공간 상수
        res['seq_out_rms_mean'] = float(np.mean(srms))
        res['partners'] = {t: {str(k): v for k, v in pm.items()} for t, pm in partners.items()}
    base = [ssim['normal'][i] for i in indices]
    res['tests'] = OrderedDict()
    for t in tests:
        if t == 'normal':
            continue
        st = drop_stats(base, [ssim[t][i] for i in indices])
        if st is not None:
            st['ssim_mean'] = float(np.mean([v for v in (ssim[t][i] for i in indices) if v is not None]))
            if t in extras:
                st.update(extras[t])
        res['tests'][t] = st
    # 이전 판과 같은 이름의 요약 키 (shuffle = cross)
    if has_seq:
        res['dssim_shuffle'] = res['tests']['shuffle_cross']['mean']
        res['dssim_zero'] = res['tests']['zero']['mean']
        res['seq_out_pair_rel_diff_mean'] = extras['shuffle_cross']['seq_out_pair_rel_diff_mean']
    res['per_slice_ssim'] = {t: [ssim[t].get(i) for i in indices] for t in tests}
    model.train()
    return res


# ------------------------------------------------------------------ 판정
def group_verdict(g):
    if g is None:
        return None
    if g['near_dead_frac'] >= TH_COLLAPSED_NEAR:
        return 'collapsed'
    if g['rel_change'] is not None and g['rel_change'] < TH_FROZEN_REL:
        return 'frozen'
    if g.get('eps_dom_frac') is not None and g['eps_dom_frac'] >= TH_EPS_DOM:
        return 'eps-limited'
    return 'moving'


def make_verdict(final_groups, val):
    t = (val or {}).get('tests') or {}
    return OrderedDict(
        seq=group_verdict(final_groups.get('seq')),
        unet_deep=group_verdict(final_groups.get('unet_deep')),
        unet_all=group_verdict(final_groups.get('unet_all')),
        seq_used=used_verdict(t.get('shuffle_cross')),
        seq_used_adj=used_verdict(t.get('shuffle_adj')),
        deep_used=used_verdict(t.get('deep4')),
        criteria=dict(collapsed=f'near_dead_frac>={TH_COLLAPSED_NEAR}', frozen=f'rel_change<{TH_FROZEN_REL}',
                      eps_limited=f'eps_dom_frac>={TH_EPS_DOM}',
                      used=f'mean>0 & mean>2SE & sign-test p<{TH_USED_P}'),
    )


# ------------------------------------------------------------------ job
def _first_step(records, key, field, thr):
    for r in records:
        g = r['groups'].get(key)
        if g is not None and g[field] >= thr:
            return r['step']
    return None


def cond_lambda(cond, args):
    base = args.lambda_ssim
    if CONDS[cond]['lam'] == 'matched':
        return base / args.dl_scale['mean']
    return base


def run_job(seq, cond, args, train_ds, val_ds, val_indices, probe, device, out_dir):
    spec = CONDS[cond]
    tag = f'{seq}_{cond}'
    is_cuda = device.type == 'cuda'
    t_job = time.time()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    seed_all(args.seed)                                         # 같은 seq 의 모든 조건이 같은 초기값에서 시작
    model = build_model(seq, device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    w0, w0sq = snapshot_init(model, device)
    tracker = SubsetTracker(model)
    groups = build_groups(model, seq)
    criterion = SSIM().to(device)                               # main_train_pure_v8.py:224
    optimizer, opt_info = build_optimizer(model, cond)
    tmax = args.tmax_eff
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=1e-6)   # :268
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)      # :294 (CPU 에선 끔 — fp32)
    loader = make_train_loader(train_ds, args.bs, args.seed, args.num_workers, is_cuda)
    lam = cond_lambda(cond, args)
    max_norm = spec['max_norm']
    normalize = spec['normalize']

    meta = OrderedDict(
        type='meta', script_version=SCRIPT_VERSION, seq=seq, cond=cond, cond_spec=spec, steps=args.steps, tmax=tmax,
        bs=args.bs, seed=args.seed, lr=C.LEARNING_RATE_ADAM, eta_min=1e-6, lambda_ssim=lam, lambda_base=args.lambda_ssim,
        dl_scale=args.dl_scale if spec['lam'] == 'matched' else None, k_ratio=K_RATIO if normalize else None,
        weight_decay=C.LAMBDA_REGULAR_PER_PIXEL, optimizer=opt_info, n_params=n_params, device=str(device),
        torch=torch.__version__, num_workers=args.num_workers, train_samples=len(train_ds), val_samples=len(val_ds),
        loss_units='normalized (img,label /s; k-space x100/s)' if normalize else 'original (x1e6 label scale)',
        groups={k: len(v) for k, v in groups.items()}, dead_thr=DEAD_THR, near_thr=NEAR_THR, sub_n=SUB_N,
        deep_unet=list(DEEP_PREFIXES), probe_indices=args.probe_indices,
        log_every=args.log_every, log_dense_every=args.log_dense_every, log_dense_until=args.log_dense_until,
        started=datetime.datetime.now().isoformat(timespec='seconds'),
    )
    jsonl_path = os.path.join(out_dir, f'{tag}.jsonl')
    fj = open(jsonl_path, 'w')
    fj.write(json.dumps(meta, ensure_ascii=False) + '\n')
    fj.flush()
    print(f'\n========== job {tag}: {spec["optimizer"]}, max_norm={max_norm:g}, normalize={normalize}, '
          f'λ_SSIM={lam:.4g}, eps={opt_info["eps"]:g}, steps={args.steps}, T_max={tmax}, BS={args.bs}, '
          f'params={n_params / 1e6:.1f}M ==========')
    if opt_info['type'] == 'AdamW':
        print(f'  no-decay: {opt_info["no_decay_numel"]:,} 원소 {opt_info["no_decay_reasons"]}')

    records = []
    S = dict(step=0, epoch=0, consec_skip=0, nonfinite=0, amp_skips=0, status='completed',
             loss=0.0, l1=0.0, ssim=0.0, n=0, s=0.0, gnorms=[], kin=[], last_loss=None, last_gn=None, lr=None,
             l1_tot=0.0, ssimterm_tot=0.0,
             t_train0=time.time(), t_int=time.time(), steps_int=0)

    def _fmt(x, f):
        return format(x, f) if isinstance(x, (int, float)) else '-'

    def emit(grad_st):
        """현재 구간 누계로 step 기록 1행을 만들고 jsonl·stdout 에 쓴 뒤 구간 누계를 비운다. step 0 = 초기값."""
        t_now = time.time()
        step, k = S['step'], S['n']
        cc = lambda v: (min(1.0, max_norm / (v + 1e-6)) if np.isfinite(v) else 0.0)   # noqa: E731 (torch clip 식)
        g_all = torch.stack(S['gnorms']).cpu().numpy() if S['gnorms'] else np.zeros(0)
        fin = g_all[np.isfinite(g_all)]
        rec = OrderedDict(type='step', step=step, epoch=S['epoch'],
                          lr=S['lr'] if S['lr'] is not None else optimizer.param_groups[0]['lr'])
        rec.update(
            loss_total=S['loss'] / k if k else None, loss_l1=S['l1'] / k if k else None,
            loss_ssim=S['ssim'] / k if k else None, loss_ssim_term=lam * S['ssim'] / k if k else None,
            l1_over_ssim_term=(S['l1'] / (lam * S['ssim'])) if k and lam * S['ssim'] > 0 else None,
            loss_total_last=S['last_loss'], n_interval=k,
            grad_norm_preclip=S['last_gn'],
            grad_norm_interval=dict(mean=float(fin.mean()) if fin.size else None,
                                    min=float(fin.min()) if fin.size else None,
                                    max=float(fin.max()) if fin.size else None,
                                    n_nonfinite=int(g_all.size - fin.size)),
            clip_coef=cc(S['last_gn']) if S['last_gn'] is not None else None,
            clip_coef_interval_mean=float(np.mean([cc(v) for v in g_all])) if g_all.size else None,
            max_norm=max_norm, scaler_scale=scaler.get_scale() if is_cuda else None,
            amp_skipped_steps=S['amp_skips'], nonfinite_loss_skips=S['nonfinite'],
            sec_per_step=(t_now - S['t_int']) / S['steps_int'] if S['steps_int'] else None,
            elapsed_s=round(t_now - S['t_train0'], 1) if step else 0.0,
            seq_in_rms_train=float(torch.stack(S['kin']).mean()) if S['kin'] else None,
        )
        if normalize:
            rec['scale_s_mean'] = S['s'] / k if k else None
        st = per_param_stats(model, w0, optimizer, tracker)
        if grad_st is not None:
            for n_ in st:
                st[n_]['gsq'] = grad_st.get(n_)
        rec['groups'] = aggregate(st, groups, w0sq)
        if step == 0:                                           # 구간 변화량은 step 0 에 정의되지 않음
            for gv in rec['groups'].values():
                gv['rel_change_interval'] = None
        rec['io_slices'] = io_slice_stats(model)
        rec['probe'] = probe_stats(model, seq, probe, normalize, device, is_cuda)
        rec['stats_sec'] = round(time.time() - t_now, 3)
        if is_cuda:
            rec['gpu_mem_alloc_gb'] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
        fj.write(json.dumps(rec, ensure_ascii=False) + '\n')
        fj.flush()
        records.append(rec)
        g = rec['groups']
        pr = rec['probe']
        pct = lambda key, f='near_dead_frac': (f'{100 * g[key][f]:.1f}%' if key in g and g[key][f] is not None else '-')  # noqa: E731
        rel = lambda key: (_fmt(g[key]['rel_change'], '.3g') if key in g else '-')            # noqa: E731
        fc = rec['io_slices']['unet_first_conv']
        head = (f'[{tag}] step {step}/{args.steps}' + (' (init)' if step == 0 else '')
                + f' lr {_fmt(rec["lr"], ".2e")} loss {_fmt(rec["loss_total"], ".4f")} (l1 {_fmt(rec["loss_l1"], ".4f")}'
                f' ssim {_fmt(rec["loss_ssim"], ".4f")}) gnorm {_fmt(rec["grad_norm_preclip"], ".3g")}'
                f' clip {_fmt(rec["clip_coef"], ".3g")} scale {_fmt(rec["scaler_scale"], "g")}'
                f' skip amp/nan {S["amp_skips"]}/{S["nonfinite"]} {_fmt(rec["sec_per_step"], ".3f")}s/step')
        print(f'{head} | near-dead seq {pct("seq")} unet {pct("unet_all")} deep {pct("unet_deep")}'
              f' (dead seq {pct("seq", "dead_frac")}) | relΔ seq {rel("seq")} deep {rel("unet_deep")}'
              f' | eps-dom deep {pct("unet_deep", "eps_dom_frac")} | in-dep seq {_fmt(pr.get("seq_in_dep"), ".3g")}'
              f' deep4Δout {_fmt(pr.get("deep4_out_rel"), ".3g")} | 1st-conv max|w| seq {fc["seq_in"]["maxabs"]:.3g}'
              f' zf {fc["zf_in"]["maxabs"]:.3g} | stats {rec["stats_sec"]:.1f}s', flush=True)
        S.update(loss=0.0, l1=0.0, ssim=0.0, n=0, s=0.0, gnorms=[], kin=[], t_int=time.time(), steps_int=0)

    def is_log_step(step1):
        if step1 % args.log_every == 0 or step1 == args.steps:
            return True
        return args.log_dense_every > 0 and step1 <= args.log_dense_until and step1 % args.log_dense_every == 0

    emit(None)                                                  # step 0 = 초기값
    S['t_train0'] = S['t_int'] = time.time()
    it = iter(loader)
    while S['step'] < args.steps:
        try:
            sample = next(it)
        except StopIteration:                                   # 4000×8 < 1 epoch 이지만 일반화
            S['epoch'] += 1
            it = iter(loader)
            sample = next(it)
        # main_train_pure_v8.py:341-346
        data_in     = sample['data'].float().to(device)
        data_in_img = sample['data_img'].float().to(device)
        data_ref    = sample['label'].float().to(device)
        brain_mask  = sample['brain_mask'].float().to(device)
        mask        = sample['mask'].float().to(device)
        sens        = sample['sens'].float().to(device)
        if normalize:                                           # 조건 D/DL/E: 영상·label /s, k-space ×100/s (선형 관계 유지)
            s = per_sample_scale(data_in_img)
            data_in, data_in_img = normalize_inputs(data_in, data_in_img, s)
            data_ref = data_ref / s

        with torch.amp.autocast('cuda', enabled=is_cuda):       # :348-349
            out = model(data_in_img, data_in, mask, sens)
        out_fp    = out.float()                                 # :351-355
        m_sum     = brain_mask.sum().clamp(min=1.0)
        loss_l1   = ((out_fp - data_ref).abs() * brain_mask).sum() / m_sum
        loss_ssim = 1 - criterion(out_fp, data_ref, mask=brain_mask)
        loss      = loss_l1 + lam * loss_ssim

        if not torch.isfinite(loss):                            # :360-377 NaN/Inf-skip 가드
            optimizer.zero_grad(set_to_none=True)
            S['consec_skip'] += 1
            S['nonfinite'] += 1
            if S['consec_skip'] <= 3 or S['consec_skip'] % 50 == 0:
                print(f'  [NaN-skip] {tag} step{S["step"]} loss={loss.item()} consec={S["consec_skip"]} '
                      f'total={S["nonfinite"]}', flush=True)
            if S['consec_skip'] >= C.MAX_CONSEC_SKIP:
                S['status'] = 'aborted_nonfinite'
                print(f'  [FATAL] {tag}: 연속 {S["consec_skip"]} non-finite loss → job 중단', flush=True)
                break
            continue
        S['consec_skip'] = 0

        S['lr'] = optimizer.param_groups[0]['lr']               # 이 step 에 쓰는 LR
        scaler.scale(loss / ACCUM_STEPS).backward()             # :380
        scaler.unscale_(optimizer)                              # :383
        log_now = is_log_step(S['step'] + 1)
        grad_st = None
        if log_now:                                             # clip 전 파라미터별 grad 제곱합 (그룹 grad norm 용; 전체 크기 임시 없음)
            with torch.no_grad():
                names_g = [n_ for n_, p in model.named_parameters() if p.grad is not None]
                norms = [torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32)
                         for _, p in model.named_parameters() if p.grad is not None]
                grad_st = {n_: float(v) ** 2 for n_, v in zip(names_g, torch.stack(norms).tolist())} if norms else {}
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)   # :384 (A·B 1.0 / C~E 1e3)
        scale_before = scaler.get_scale() if is_cuda else None
        scaler.step(optimizer)                                  # :385-388
        scaler.update()
        if is_cuda and scaler.get_scale() < scale_before:       # GradScaler 가 inf/nan grad 로 step 을 건너뜀
            S['amp_skips'] += 1
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        S['step'] += 1
        S['steps_int'] += 1
        li, l1i, lsi = loss.item(), loss_l1.item(), loss_ssim.item()
        S['loss'] += li
        S['l1'] += l1i
        S['ssim'] += lsi
        S['l1_tot'] += l1i
        S['ssimterm_tot'] += lam * lsi
        S['n'] += 1
        if normalize:
            S['s'] += float(s.mean())
        S['gnorms'].append(gn.detach().float())
        S['kin'].append((torch.linalg.vector_norm(data_in.detach(), dtype=torch.float32)
                         / math.sqrt(data_in.numel())).detach())
        S['last_loss'], S['last_gn'] = li, float(gn)

        over_time = args.max_minutes > 0 and (time.time() - S['t_train0']) > args.max_minutes * 60
        if log_now or over_time:
            emit(grad_st)
        if over_time:
            S['status'] = 'truncated_time'
            print(f'  [max-minutes] {tag}: {args.max_minutes} 분 초과 → step {S["step"]} 에서 학습 중단', flush=True)
            break
    if records[-1]['step'] != S['step'] or S['n'] > 0:          # 중단(abort) 등으로 마지막 step 이 기록되지 않았으면
        emit(None)
    step, status = S['step'], S['status']
    amp_skips, nonfinite = S['amp_skips'], S['nonfinite']
    train_sec = time.time() - S['t_train0']
    del it, loader                                              # persistent 워커 종료

    if args.save_ckpt == 'end':
        torch.save(model.state_dict(), os.path.join(out_dir, f'{tag}_end.pt'))

    t_val = time.time()
    val = run_validation(model, seq, normalize, val_ds, val_indices, device, args.val_workers, is_cuda, is_cuda)
    val_sec = time.time() - t_val
    fj.close()

    final = records[-1]
    summ = OrderedDict(
        script_version=SCRIPT_VERSION, seq=seq, cond=cond, status=status, steps_requested=args.steps, steps_done=step,
        tmax=tmax, cond_spec=spec, optimizer=opt_info, bs=args.bs, seed=args.seed, num_workers=args.num_workers,
        lambda_ssim=lam, lambda_base=args.lambda_ssim, lambda_dl_scale_arg=args.lambda_dl_scale,
        dl_scale=args.dl_scale if spec['lam'] == 'matched' else None, n_params=n_params,
        amp_skipped_steps=amp_skips, nonfinite_loss_skips=nonfinite,
        loss_ratio_l1_over_ssim_term=(S['l1_tot'] / S['ssimterm_tot']) if S['ssimterm_tot'] > 0 else None,
        init_groups=records[0]['groups'], final_groups=final['groups'],
        init_io_slices=records[0]['io_slices'], final_io_slices=final['io_slices'],
        init_probe=records[0]['probe'], final_probe=final['probe'],
        final_train=dict((k, final.get(k)) for k in ('loss_total', 'loss_l1', 'loss_ssim', 'l1_over_ssim_term',
                                                      'grad_norm_preclip', 'clip_coef', 'scaler_scale', 'sec_per_step',
                                                      'seq_in_rms_train')),
        collapse_step={f'{g}_{fld}_ge_{thr}': _first_step(records, g, fld, thr)
                       for g in ('seq', 'unet_all', 'unet_deep')
                       for fld in ('dead_frac', 'near_dead_frac') for thr in (0.5, 0.99)},
        val=val, verdict=make_verdict(final['groups'], val),
        train_sec=round(train_sec, 1), val_sec=round(val_sec, 1),
        seconds=round(time.time() - t_job, 1),
        peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if is_cuda else None,
        finished=datetime.datetime.now().isoformat(timespec='seconds'),
    )
    v = val['normal']
    t = val['tests']
    print(f'[{tag}] 완료 status={status} steps={step} | val SSIM {v["ssim"]:.4f} PSNR {v["psnr"]:.2f} '
          f'nMSE {v["nmse"]:.5f} L1 {v["l1"]:.3f} | dSSIM deep4 {t["deep4"]["mean"]:+.5f}'
          + (f' shuffle {t["shuffle_cross"]["mean"]:+.5f} ({t["shuffle_cross"]["n_pos"]}/{t["shuffle_cross"]["n"]})'
             f' zero {t["zero"]["mean"]:+.5f}' if 'shuffle_cross' in t else '')
          + f' | verdict {dict((k, v_) for k, v_ in summ["verdict"].items() if k != "criteria")}'
          + f' | {summ["seconds"]:.0f}s', flush=True)
    del model, optimizer, scheduler, scaler, w0, tracker
    return summ


# ------------------------------------------------------------------ 요약 파일
MD_HEADER = (
    '# 학습 방법 수정 스모크 — job 요약\n\n'
    '이 파일은 out-dir 의 모든 `*_summary.json` 으로 매번 다시 만든다. 조건: A Adam 결합 L2 + clip 1 (현행) / '
    'B AdamW 분리형 + no-decay + clip 1 / C B + clip 1e3 / D C + 샘플별 정규화(영상·label /s, k-space ×100/s) / '
    'DL D + λ_SSIM = λ/s̄ (C 의 L1:SSIM 균형) / E D + Adam eps 1e-12.\n\n'
    '해석 주의 (실행 전 고정):\n'
    '- **D 는 단일 요인 비교가 아니다.** 정규화는 (1) 손실 균형(원 스케일 L1 ≈ 손실의 99% → 정규화 후 SSIM 항 ≈65%), '
    '(2) u_choh_SSIM 동적 범위 L(출력 max 로 결정)과 안정화 상수의 상대 크기, (3) U-Net 입력의 zero-filled/시퀀스 채널 크기 균형, '
    '(4) SS2D LayerNorm 동작 영역을 함께 바꾼다. C→DL ≈ 스케일 요인만, DL→D ≈ 손실 균형만.\n'
    '- B~E 의 분리형 wd 3e-5 는 step 당 6e-9 수축(4000 step 누계 2.4e-5)이라 사실상 wd≈0 이다. 큰 AdamW decay 의 안전성은 시험하지 않았다.\n'
    '- T_max = job step 수(기본)라 이 스모크는 50ep/5ep 런의 앞부분이 아니다(총 이동량 ≈40% 작음).\n'
    '- 판정: 소멸 = near-dead(|w|<1e-6) ≥ 50% · 동결 = relΔ < 0.01 · eps 제한 = √v̂<eps 원소 ≥ 90% · 그 외 움직임. '
    'relΔ 는 학습과 잡음 표류를 구분하지 못한다. **사용(used)** = 슬라이스별 SSIM 하락 평균 > 0, > 2·SE, 부호 검정 p < 0.01 '
    '(시퀀스 = shuffle_cross, 깊은 U-Net = deep4). "not shown" 은 사용하지 않는다는 증거가 아니다 — 4000 step(≈0.49 epoch)의 '
    'null 은 결론 불가이며, A~C 에서는 zero-filled 채널(max 175~690)과 시퀀스 출력(RMS 0.03~0.35)의 크기 차이가 크다.\n'
    '- 시퀀스 입력 의존성 in-dep = ‖f(x)−f(0)‖/‖f(0)‖ (고정 probe 2 슬라이스, 처음 → 끝).\n\n'
)
MD_T1 = (
    '## 표 1. 가중치 (끝 시점; 괄호 = 초기값)\n\n'
    '| seq | cond | steps | status | λ_SSIM | dead% seq | near-dead% seq | near-dead% U-Net | near-dead% deep '
    '| relΔ seq | relΔ deep | eps-dom% seq | eps-dom% deep | upd/lr deep | 1st-conv RMS seq / zf | 판정 seq | 판정 deep |\n'
    '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n'
)
MD_T2 = (
    '\n## 표 2. 기능·검증 (고정 val 슬라이스, 원 단위 — D/DL/E 는 출력×s; dSSIM = 정상 − 시험, ± SE, (양수/n))\n\n'
    '| seq | cond | val SSIM | val PSNR | val nMSE | L1 : SSIM 항 | shuffle dSSIM | adj-shuffle dSSIM | zero dSSIM '
    '| deep4 dSSIM | deep3-4 dSSIM | in-dep seq (init→end) | seq 사용 | deep 사용 | sec |\n'
    '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n'
)


def _num(x, fmt):
    return fmt.format(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else '—'


def _pct(g, key, field):
    return f'{100 * g[key][field]:.2f}' if key in g and g[key].get(field) is not None else '—'


def md_rows(s):
    seq, cond = s.get('seq', '?'), s.get('cond', '?')
    if s.get('status') == 'failed':
        err = (s.get('error') or '').replace('|', '/').replace('\n', ' ')[:120]
        r1 = f'| {seq} | {cond} | — | failed: {err} |' + ' — |' * 13 + '\n'
        r2 = f'| {seq} | {cond} |' + ' — |' * 13 + '\n'
        return r1, r2
    f = s.get('final_groups') or {}
    i = s.get('init_groups') or {}
    val = s.get('val') or {}
    v = val.get('normal') or {}
    t = val.get('tests') or {}
    vd = s.get('verdict') or {}

    def withinit(key, field):
        a = _pct(f, key, field)
        return a if key not in f else f'{a} ({_pct(i, key, field)})'

    def rel(key):
        return _num(f[key]['rel_change'], '{:.3g}') if key in f else '—'

    fc = (s.get('final_io_slices') or {}).get('unet_first_conv') or {}
    conv = (f'{_num(fc.get("seq_in", {}).get("rms"), "{:.3g}")} / {_num(fc.get("zf_in", {}).get("rms"), "{:.3g}")}'
            if fc else '—')
    r1 = (f'| {seq} | {cond} | {s.get("steps_done", "—")} | {s.get("status")} | {_num(s.get("lambda_ssim"), "{:.4g}")} '
          f'| {withinit("seq", "dead_frac")} | {withinit("seq", "near_dead_frac")} | {_pct(f, "unet_all", "near_dead_frac")} '
          f'| {_pct(f, "unet_deep", "near_dead_frac")} | {rel("seq")} | {rel("unet_deep")} '
          f'| {_pct(f, "seq", "eps_dom_frac")} | {_pct(f, "unet_deep", "eps_dom_frac")} '
          f'| {_num((f.get("unet_deep") or {}).get("upd_ratio_mean"), "{:.3g}")} | {conv} '
          f'| {vd.get("seq") or "—"} | {vd.get("unet_deep") or "—"} |\n')

    def ds(name):
        d = t.get(name)
        if not d:
            return '—'
        se = f' ± {d["se"]:.2g}' if d.get('se') is not None else ''
        return f'{d["mean"]:+.3g}{se} ({d["n_pos"]}/{d["n"]})'

    ip, fp = s.get('init_probe') or {}, s.get('final_probe') or {}
    dep = (f'{_num(ip.get("seq_in_dep"), "{:.3g}")} → {_num(fp.get("seq_in_dep"), "{:.3g}")}'
           if 'seq_in_dep' in fp else '—')
    r2 = (f'| {seq} | {cond} | {_num(v.get("ssim"), "{:.4f}")} | {_num(v.get("psnr"), "{:.2f}")} '
          f'| {_num(v.get("nmse"), "{:.5f}")} | {_num(s.get("loss_ratio_l1_over_ssim_term"), "{:.3g}")} '
          f'| {ds("shuffle_cross")} | {ds("shuffle_adj")} | {ds("zero")} | {ds("deep4")} | {ds("deep34")} | {dep} '
          f'| {vd.get("seq_used") or "—"} | {vd.get("deep_used") or "—"} | {_num(s.get("seconds"), "{:.0f}")} |\n')
    return r1, r2


def rebuild_summary_md(out_dir):
    summs = []
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith('_summary.json'):
            continue
        try:
            with open(os.path.join(out_dir, fn)) as f:
                summs.append(json.load(f))
        except Exception:                                       # noqa: BLE001
            continue
    conds = list(CONDS)
    summs.sort(key=lambda s: (SEQS.index(s['seq']) if s.get('seq') in SEQS else 99,
                              conds.index(s['cond']) if s.get('cond') in conds else 99))
    rows = [md_rows(s) for s in summs]
    md = os.path.join(out_dir, 'summary.md')
    tmp = md + '.tmp'
    with open(tmp, 'w') as f:
        f.write(MD_HEADER + MD_T1 + ''.join(r[0] for r in rows) + MD_T2 + ''.join(r[1] for r in rows))
    os.replace(tmp, md)


def write_summary(out_dir, summ):
    tag = f'{summ["seq"]}_{summ["cond"]}'
    p = os.path.join(out_dir, f'{tag}_summary.json')
    with open(p + '.tmp', 'w') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    os.replace(p + '.tmp', p)
    rebuild_summary_md(out_dir)


def done_matches(prev, seq, cond, args):
    """--skip-done: 같은 설정으로 끝난 job 인가 (설정이 다르면 다시 실행해 서로 비교 불가능한 결과가 섞이지 않게)."""
    if prev.get('status') not in ('completed', 'truncated_time'):
        return False, f'status={prev.get("status")}'
    want = dict(script_version=SCRIPT_VERSION, steps_requested=args.steps, tmax=args.tmax_eff, bs=args.bs,
                seed=args.seed, num_workers=args.num_workers, lambda_base=args.lambda_ssim,
                lambda_dl_scale_arg=args.lambda_dl_scale)
    for k, w in want.items():
        if prev.get(k) != w:
            return False, f'{k}: {prev.get(k)!r} != {w!r}'
    if json.dumps(prev.get('cond_spec'), sort_keys=True) != json.dumps(json.loads(json.dumps(CONDS[cond])), sort_keys=True):
        return False, 'cond_spec 다름'
    return True, 'same settings'


# ------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser(description='v8 학습 방법 수정 스모크 (소멸 방지 레시피 분리)',
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seq', choices=SEQS, help='단일 job 의 모델 (--jobs 가 있으면 무시)')
    p.add_argument('--cond', choices=list(CONDS), help='단일 job 의 조건 (--jobs 가 있으면 무시)')
    p.add_argument('--jobs', default=None, help='"unet:A,gru:B,..." — 한 프로세스에서 순차 실행')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--tmax', type=int, default=None,
                   help='cosine T_max (기본 = --steps, spec). 5ep 런의 앞부분과 같게 하려면 40645')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--log-dense-every', type=int, default=20, help='처음 --log-dense-until step 까지의 기록 간격 (0 = 끔)')
    p.add_argument('--log-dense-until', type=int, default=500)
    p.add_argument('--val-slices', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--lambda-ssim', type=float, default=C.LAMBDA_SSIM_PER_PIXEL,
                   help='λ_SSIM 기본값 (config 1.0 = spec). DL 은 이 값 / s̄')
    p.add_argument('--lambda-dl-scale', type=float, default=None,
                   help='DL 의 s̄ 를 직접 지정 (기본: train 64 슬라이스에서 추정)')
    p.add_argument('--out-dir', default='results/smoke_recipe_fix', help='상대 경로는 저장소 루트 기준')
    p.add_argument('--save-ckpt', choices=('none', 'end'), default='none')
    p.add_argument('--bs', type=int, default=8, help='기본 = 트레이너 BS 8 (runs/smoke_bs.txt)')
    p.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    p.add_argument('--force', action='store_true', help='GPU0 사용 중·CUDA_VISIBLE_DEVICES≠0 거부를 무시')
    p.add_argument('--max-minutes', type=float, default=0, help='job 당 학습 시간 상한(분), 0 = 없음')
    p.add_argument('--num-workers', type=int, default=None, help='학습 로더 워커 (기본: cuda=config 16, cpu=2)')
    p.add_argument('--val-workers', type=int, default=None, help='검증 로더 워커 (기본: cuda=config 4, cpu=1)')
    p.add_argument('--skip-done', action='store_true',
                   help='같은 설정으로 끝난(status completed|truncated_time) job 은 건너뜀 (재기동용; failed 는 다시 실행)')
    a = p.parse_args()
    if a.jobs:
        jobs = []
        for tok in a.jobs.split(','):
            tok = tok.strip()
            if not tok:
                continue
            if tok.count(':') != 1:
                p.error(f'잘못된 job: {tok}')
            sq, cd = tok.split(':')
            sq, cd = sq.strip().lower(), cd.strip().upper()
            if sq not in SEQS or cd not in CONDS:
                p.error(f'잘못된 job: {tok}')
            jobs.append((sq, cd))
    else:
        if not (a.seq and a.cond):
            p.error('--jobs 또는 --seq 와 --cond 를 지정')
        jobs = [(a.seq, a.cond)]
    a.job_list = jobs
    if a.num_workers is None:
        a.num_workers = C.NUM_WORKERS_TRAIN if a.device == 'cuda' else 2
    if a.val_workers is None:
        a.val_workers = C.NUM_WORKERS_VAL if a.device == 'cuda' else 1
    if a.steps < 1 or a.log_every < 1 or a.bs < 1 or a.val_slices < 1 or a.log_dense_every < 0:
        p.error('--steps/--log-every/--bs/--val-slices 는 1 이상, --log-dense-every 는 0 이상')
    a.tmax_eff = a.tmax if a.tmax is not None else a.steps
    if a.tmax_eff < 1:
        p.error('--tmax 는 1 이상')
    if not os.path.isabs(a.out_dir):
        a.out_dir = os.path.join(_PROJECT_ROOT, a.out_dir)
    return a


def main():
    args = parse_args()
    device = torch.device(args.device)
    print('=' * 72)
    print(f' v8 학습 방법 수정 스모크 ({SCRIPT_VERSION})  device={device}  '
          f'jobs={",".join(f"{s}:{c}" for s, c in args.job_list)}')
    print(f'   steps={args.steps} T_max={args.tmax_eff} log_every={args.log_every} '
          f'(dense {args.log_dense_every} until {args.log_dense_until}) bs={args.bs} seed={args.seed} '
          f'val_slices={args.val_slices} workers={args.num_workers}/{args.val_workers} λ={args.lambda_ssim} out={args.out_dir}')
    print(f'   {datetime.datetime.now().isoformat(timespec="seconds")}  torch {torch.__version__}')
    print('=' * 72)
    pending = []
    for seq, cond in args.job_list:                             # --skip-done: 같은 설정으로 끝난 job 은 데이터 준비 전에 제외
        tag = f'{seq}_{cond}'
        sp = os.path.join(args.out_dir, f'{tag}_summary.json')
        if args.skip_done and os.path.exists(sp):
            try:
                with open(sp) as f:
                    prev = json.load(f)
                ok, why = done_matches(prev, seq, cond, args)
                if ok:
                    print(f'[skip-done] {tag} 이미 끝남 (status={prev["status"]}, 같은 설정) — 건너뜀')
                    continue
                print(f'[skip-done] {tag} 기존 결과를 다시 실행 ({why})')
            except Exception as e:                              # noqa: BLE001
                print(f'[skip-done] {tag} 기존 summary 읽기 실패 ({e}) — 다시 실행')
        pending.append((seq, cond))
    if not pending:
        print('실행할 job 없음 (모두 끝남)')
        return 0

    if device.type == 'cuda':
        gpu0_precheck(args.force)
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            refuse('CPU 모드는 CUDA_VISIBLE_DEVICES="" 로만 실행한다 (GPU0 드라이버·NVML 을 건드리지 않기 위해)')
        install_cpu_shims(need_ss2d=any(s == 'ss2d' for s, _ in pending))
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = build_train_dataset()
    val_ds = build_val_dataset()
    for ds in (train_ds, val_ds):                               # k-space ×100/s 의 근거 (dataloader_h5_v5.py:131-132)
        assert abs(ds.val_amp_X_img / ds.val_amp_X_ksp - K_RATIO) < 1e-9, '로더 배율이 바뀌었다 — K_RATIO 재확인'
    val_indices = even_indices(len(val_ds), args.val_slices)
    print(f'  val 슬라이스 {len(val_indices)}개: {val_indices[:8]}{" ..." if len(val_indices) > 8 else ""}')
    args.probe_indices = [val_indices[0], val_indices[len(val_indices) // 2]] if len(val_indices) >= 2 else \
        [val_indices[0], (val_indices[0] + len(val_ds) // 2) % len(val_ds)]
    probe = load_probe(val_ds, args.probe_indices)
    print(f'  probe 슬라이스: {args.probe_indices}')
    args.dl_scale = None
    if any(CONDS[c]['lam'] == 'matched' for _, c in pending):
        if args.lambda_dl_scale is not None:
            args.dl_scale = dict(mean=float(args.lambda_dl_scale), source='--lambda-dl-scale')
        else:
            t0 = time.time()
            args.dl_scale = estimate_mean_scale(train_ds, args.seed)
            args.dl_scale['source'] = f'train {args.dl_scale["n"]} slices'
            print(f'  DL s̄ = {args.dl_scale["mean"]:.2f} (median {args.dl_scale["median"]:.2f}, '
                  f'{args.dl_scale["min"]:.1f}~{args.dl_scale["max"]:.1f}) → λ_DL = {args.lambda_ssim / args.dl_scale["mean"]:.4g}'
                  f' ({time.time() - t0:.0f}s)')

    n_fail = 0
    for seq, cond in pending:
        tag = f'{seq}_{cond}'
        sp = os.path.join(args.out_dir, f'{tag}_summary.json')
        if os.path.exists(sp):                                  # 다시 실행하는 job: 이전 summary 를 치워 jsonl 과 짝이 어긋나지 않게
            os.replace(sp, os.path.join(args.out_dir, f'{tag}_summary.prev.json'))
            rebuild_summary_md(args.out_dir)
        try:
            summ = run_job(seq, cond, args, train_ds, val_ds, val_indices, probe, device, args.out_dir)
        except KeyboardInterrupt:
            raise
        except Exception as e:                                  # noqa: BLE001 — OOM 등: 기록 후 다음 job
            n_fail += 1
            tb = traceback.format_exc()
            print(f'[FAIL] {tag}: {e}\n{tb}', flush=True)
            summ = OrderedDict(script_version=SCRIPT_VERSION, seq=seq, cond=cond, status='failed', error=str(e)[:2000],
                               traceback=tb[-4000:], finished=datetime.datetime.now().isoformat(timespec='seconds'))
        write_summary(args.out_dir, summ)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f'  [mem] job 후 GPU allocated {torch.cuda.memory_allocated() / 2 ** 30:.2f} GB '
                  f'reserved {torch.cuda.memory_reserved() / 2 ** 30:.2f} GB', flush=True)
    print(f'\n전체 완료 (실패 {n_fail}건) → {os.path.join(args.out_dir, "summary.md")}', flush=True)
    return EXIT_JOB_FAILED if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
