"""
Mamba-ETER 4000 step 스모크 (2026-10-11) — ETER-net(GRU)의 순환 셀만 양방향 Mamba 블록으로 바꾼 모델(PureETER_Mamba1D)을
smoke_recipe_fix 의 DL 레시피 그대로 학습해, U-Net 이 그 출력을 쓰는지(shuffle_cross 기능 시험) 본다.

배경: results/ss2d_diag_2026-10-11/(forensics.md·theory.md·verify.md) — 현행 SS2D 는 k-space → 영상 변환을 표현할 수 없고 U-Net 이
  그 출력을 쓰지 않는다. ETER-net 의 GRU 는 2단 순차 합성(h → v)과 step 블록 전체에 대한 dense 입력 투영으로 원리상 가능하다.
  Mamba-ETER(models/pure_eternet/u_pure_eternet_mamba1d.py)는 이 데이터 흐름을 글자 그대로 두고 셀만 바꾼 모델이다 —
  'SSM vs RNN' 의 가장 깨끗한 비교 (사용자 결정 10-11).

비교 기준 (같은 DL 레시피·4000 step·val 64 슬라이스, results/smoke_recipe_fix/*_DL_summary.json):
  gru:DL  (ETER-net)  val SSIM 0.7973 · shuffle_cross +0.0148 ± 0.0066 (62/64) '사용' · zero +0.0176
  ss2d:DL             val SSIM 0.7772 · shuffle_cross +0.0001 (40/64) 'not shown'
  unet:DL (U-Net 단독) val SSIM 0.7773

재사용 (smoke_recipe_fix.py = R, diag_ss2d_variants.py = D 를 모듈로 import — 두 파일·트레이너·wrapper 무수정):
  - R.run_job('gru', 'DL', …) 를 그대로 호출한다. 바꾸는 것은 R.build_model 이름 하나(런타임 교체 — D 의 import 가 원 함수를
    잡은 뒤에 교체)뿐이다. 하네스 라벨은 'gru' 여야 한다: R 의 has_seq·seq_forward(model._seq)·build_groups('gru'
    = 이름 접두사 gru_h./gru_v.) 가 정확히 맞고, R 에 그 밖의 gru 전용 경로는 없다. 따라서 데이터(같은 로더·augment·랜덤 마스크
    offset·셔플 generator·워커 시드), 손실(masked L1 + λ/s̄·(1−SSIM)), AdamW 분리형 + no-decay(Mamba A_log·D 는 `_no_weight_decay`,
    conv1d.bias·dt_proj.bias 는 'bias'), clip 1e3, cosine(T_max = steps, eta_min 1e-6), fp16 autocast + GradScaler, NaN-skip,
    기록(처음 500 step 20 마다·이후 100 마다), 검증(normal / deep4 / deep34 / zero / shuffle_cross / shuffle_adj,
    R.drop_stats·R.used_verdict)이 gru:DL 과 같다.
  - U-Net 초기값(--unet-init gru, 기본): 스모크 하네스에서는 wrapper 가 시퀀스 모듈을 먼저 만들어 모델마다 U-Net 초기값이 다르다
    (step 0 unet_all RMS: gru 0.009155682 / unet 0.009153541 / ss2d 0.009153275). 1차 기준 gru:DL 과 시퀀스 모듈만 다르게 하려고
    builder 가 호출 시점의 RNG 상태(run_job 의 seed_all(seed) 직후)에서 PureETER_GRU 를 CPU 에 만들어(fork_rng — 전역 RNG 보존)
    U-Net 만 복사한다 → gru:DL 의 U-Net 초기값과 비트 단위로 같다. 끝에 step 0 기록을 gru_DL.jsonl step 0 과 대조한다.
    --unet-init unet = v8fix 관례(PureETER_UNET 복사), own = Mamba-ETER 자체 생성 순서.
  - 학습 뒤 추가 분석은 D.post_analysis 를 프록시(.unet = model.unet, .ss2d = model._seq)로 그대로 재사용한다(에너지 분해·대체 시험·
    첫 conv 응답비·격자 지표 — SS2D 진단과 같은 정의). Mamba 전용(새 코드): 일반판 weight norm 비, softplus(dt_bias)·A·D 의 초기→끝,
    단계별(h·v)·방향별 출력 에너지.
  - 버전 고정: R.SCRIPT_VERSION == '2026-10-07b', D.DIAG_VERSION == '2026-10-11a' assert + 두 파일 sha256 을 meta 에 기록
    (R 의 gru 분기가 바뀌면 조용히 Mamba-ETER 에도 적용되는 것을 막는다).

실험 조건 (사전 등록 — 2026-10-11b 판에서 outmatch 를 필수 짝 조건으로 올림):
  default   mamba_ssm 기본 초기화. CPU fp32 실측 초기 출력 RMS ≈3e-5 로 ETER-net(0.019)의 약 1/600 이다. 그 결과 시퀀스 모듈의
            기울기가 ETER-net 보다 ≈450배 작고 전 원소가 Adam eps 1e-8 아래다(수정자 CPU 실측: in_proj RMS 4.7e-10 vs GRU W_ih 2.1e-7;
            x_proj·dt_proj·A_log 1e-14~5e-12). AdamW 갱신이 |g|/eps 로 줄어들므로 default 단독의 'not shown' 은 셀 구조가 아니라
            옵티마이저 수치의 결과일 수 있다 → 아래 '해석 규칙'.
  outmatch  구조·나머지 초기화는 같고 각 단계 out_proj.weight(정·역방향 같은 배율)만 k 배 — 단계 출력 RMS 를 같은 시드의 ETER-net 단계
            출력 RMS 에 맞춘다(k_h 먼저, 그 상태에서 k_v). 보정 자료 = train 8 슬라이스(D.load_fit_set, rng = seed+4242), fp32·eval·
            autocast 없음, 러너 시작 때 계산해 meta 에 기록. diag_ss2d_smallout(E-a)과 같은 '초기 크기만 다른' 조건. 수정자 CPU 실측
            (k≈25.5/23.6): in_proj 기울기 RMS 2.8e-7(h)·3.2e-7(v) 로 GRU W_ih 수준, out_proj 2e-8, D·conv1d.bias 는 eps 이상 —
            그러나 Δ 경로(dt_proj·x_proj·A_log)는 여전히 eps 의 1/30~1/1000 (아래 한계).
            기본 --init default,outmatch = 두 조건을 한 프로세스에서 차례로 실행(데이터 준비 공유). 하나만: --init default 등.
  정밀도    기본 --fp32 none (하네스 그대로 fp16 autocast; 기준 gru:DL 도 cuDNN GRU fp16 — '정확히 같은 DL 레시피'). fp16 조건이 끝난 뒤
            다음 중 하나라도 있으면 같은 init 을 --fp32 hv(두 단계 전체 fp32, 이름 접미사 _fp32hv)로 한 번 더 실행하고 두 결과를 모두
            보고한다(--no-fp32-fallback 으로 끔; fallback 은 모든 1차 조건 뒤에 실행):
              넘침: amp_skipped_steps > 20, nonfinite_loss_skips > 0, scaler_scale 최솟값 < 1024, status=aborted_nonfinite
              언더플로(사전): GPU preflight 의 실제 크기 fp16 autocast 기울기(손실 배율 32768/8 — 기준 gru:DL 의 정상 배율 ÷ BS)가
                in_proj/out_proj 에서 fp32 기울기 대비 상대오차 > 0.1 이거나, fp32 에서 0 이 아닌 원소 중 fp16 에서 정확히 0 인 비율 > 0.1
              학습 상태: 하네스 판정 verdict.seq ∈ {frozen, eps-limited}
            fallback 은 h 가 아니라 hv 다 — h 단 기울기도 fp16 인 v 단 역전파를 거쳐 들어오기 때문.
  비결정성  mamba_ssm selective_scan·causal_conv1d 역전파는 dB·dC·conv dweight 를 atomicAdd 로 누적한다 — u_choh_SSIM import 의
            cudnn.deterministic=True 는 cuDNN(GRU)에만 적용. 단일 실행이며 같은 시드에서도 비트 재현이 아니다(판정에는 영향 없음).

판정 (실행 전 고정, R 과 같은 기준): 시퀀스 출력 '사용' = shuffle_cross 슬라이스별 SSIM 하락 평균 > 0, > 2·SE, 부호 검정 p < 0.01.
  보조 = shuffle_adj, post 의 no_slice_specific_spatial. 'not shown' 은 4000 step 에서 결론 불가(R 문서와 같은 단서).
해석 규칙 (실행 전 고정):
  - 'not shown' 이면서 하네스 verdict.seq ∈ {frozen, eps-limited} (= seq 그룹 relΔ < 0.01 또는 √v̂<eps 원소 ≥ 90%) 인 행은
    '해석 불가(옵티마이저·정밀도 수치)' 다 — 셀 구조에 대한 증거로 쓰지 않는다. default 의 'not shown' 은 outmatch 행 없이 보고하지 않는다.
  - 'SSM vs RNN' 비교의 1차 근거는 outmatch 행(초기 이득을 ETER-net 에 맞춘 조건)이고, default 는 '초기 이득 교란 요인' 을 보여 주는 짝이다.
  - 한계: DL 레시피(eps 1e-8)와 이 손실 규모에서 Δ 경로(dt_proj·x_proj 의 Δ 행·A_log)는 기울기가 eps 보다 2~3 자릿수 작아 사실상
    고정될 수 있다. 하위 텐서별 그룹(seqsub.*)의 끝 eps_dom_frac ≥ 0.9 면 이 스모크가 시험한 것은 'Δ 선택성이 사실상 초기값에
    고정된 Mamba(B·C 의 입력 의존성은 학습)' 이며, 결과를 일반적인 'selective SSM vs RNN' 으로 해석하지 않는다.
  probe 의 seq_in_dep = ‖f(x)−f(0)‖/‖f(0)‖ 는 Mamba-ETER 에서 f(0)=0 이라 정의되지 않는다(≈1e28 로 기록됨) — 요약에서
  '정의 안 됨' 으로 표기하고 seq_out_rms(초기→끝)를 대신 본다.

출력 (--out-dir, 기본 results/mamba1d_smoke_2026-10-11/; 상대 경로는 저장소 루트 기준, results/ 는 git 무시):
  <vname>/<vname>.jsonl     R.run_job 의 step 기록 (원 이름 gru_DL.jsonl → 이름 변경, meta 줄에 model='Mamba-ETER',
                            harness_seq_label='gru' 등 추가). 콘솔 태그 '[gru_DL]' 은 표시만의 문제다.
  <vname>/<vname>_end.pt    끝 state_dict (≈2.7 GB)
  <vname>_final.json        meta(모델·차원·파라미터 표·하네스 라벨·seed·U-Net 초기값 모드와 checksum·fp32 단계·init 조건과 k·preflight·
                            버전·sha256) / run_job_summary / unet_init_check / precision_alarm / post{end, init, mamba_extra} / status / error
  summary.md                매번 다시 만든다 (기준 3행 + Mamba-ETER 행, post 표).
  vname = mamba1d[_outmatch][_unetinit-<mode>][_fp32h|_fp32v|_fp32hv]   (fallback = _fp32hv)

GPU: R.gpu0_precheck(CUDA_VISIBLE_DEVICES="0" 이고 GPU0 사용 ≤ 2 GB 일 때만 — 진단 체인이 도는 동안은 종료 코드 2 로 거부).
  종료 코드 R 과 같음: 0 정상 / 2 거부(판 불일치·GPU0 사용 중·num_workers ≠ 기준 16·U-Net 초기값이 gru:DL 과 다름) /
  3 job 또는 post 실패 / 4 CUDA 사용 불가.
  preflight(GPU): (1) 작은 MambaOut 블록에서 fast path(CUDA 커널) vs 참조 경로(torch conv + selective_scan_ref) — fp32 출력 상대오차
  < 1e-4, 입력 기울기 < 1e-3, 모든 파라미터 기울기(in_proj·conv1d·x_proj·dt_proj·out_proj·A_log·D) 기록(최대 > 5e-2 면 실패),
  fp16 autocast 출력 < 2e-2; (2) 실제 크기 h 단(probe 1 슬라이스, fp32, no grad) fast vs 참조 출력 상대오차 < 1e-3;
  (3) 실제 모델 probe autocast forward 유한·단계별 RMS; (4) 기울기 규모(grad_preflight — CPU 에서도 fp32 부분 실행): 실제 크기
  probe 1 슬라이스 DL 손실의 fp32 역전파로 시퀀스 하위 텐서별 |g| RMS·Adam eps(1e-8) 미만 비율을 같은 시드 ETER-net 과 나란히
  기록하고, GPU 에서는 학습과 같은 fp16 autocast 역전파(손실 ×4096)와 비교해 텐서별 상대오차·정확히 0 이 된 비율을 기록한다.
  U-Net 초기값(--unet-init gru)은 학습 전에 gru_DL.jsonl step 0(unet_all RMS·첫 conv zf slice RMS)·seed 1 checksum 과 대조하고
  다르면 거부한다(학습 뒤 step 0 대조도 유지).
CPU 건식 실행(--device cpu, CUDA_VISIBLE_DEVICES="" 필수): R.install_cpu_shims(False) + install_mamba_cpu_shims()
  (mamba_simple.causal_conv1d_fn = None → slow path nn.Conv1d+SiLU, selective_scan_fn → selective_scan_cpu_full). 코드 경로 점검용이지
  측정용이 아니다. fallback 판정은 GradScaler 가 없는 CPU 에서 하지 않는다.

실행 예 (진단 체인 pid 1189756·1192395 가 끝난 뒤 수동으로만):
  # 본 실행 (default → outmatch, 필요 시 각 _fp32hv fallback):
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u v8_eter_pure/smoke_mamba1d.py --skip-done
  # 30 step GPU 스모크 (결과는 scratch 로 — fallback 경로까지 실행, 끝 ckpt 2.7 GB × 최대 4 개는 확인 후 삭제):
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u v8_eter_pure/smoke_mamba1d.py --steps 30 \\
      --log-every 10 --log-dense-every 0 --val-slices 4 --post-slices 4 --out-dir /tmp/mamba1d_smoke30
  # CPU 건식 실행:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 19 python v8_eter_pure/smoke_mamba1d.py --device cpu --steps 2 --log-every 1 \\
      --log-dense-every 0 --val-slices 2 --post-slices 2 --bs 1 --num-workers 1 --val-workers 1 --out-dir <scratch>
"""

import os
import sys
import gc
import json
import math
import time
import random
import hashlib
import argparse
import datetime
import traceback
import contextlib
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import smoke_recipe_fix as R            # noqa: E402 — 레시피·데이터·검증 단일 출처 (sys.path 에 models/* 추가)
import diag_ss2d_variants as D          # noqa: E402 — post_analysis·prepare_data·load_fit_set 재사용 (import 시 원 R.build_model 을 잡음)
import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import torch.nn.functional as F         # noqa: E402
from mamba_ssm.modules import mamba_simple as MS                              # noqa: E402
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref         # noqa: E402
from u_pure_eternet_mamba1d import (PureETER_Mamba1D, MambaOut, fp32_stages_from_str,   # noqa: E402
                                    mamba_params_per_direction, gru_params_per_direction)

C = R.C
RUNNER_VERSION = '2026-10-11b'
REQ_SMOKE_VERSION = '2026-10-07b'
REQ_DIAG_VERSION = '2026-10-11a'

HARNESS_SEQ, HARNESS_COND = 'gru', 'DL'
SEQ_CH = R.SEQ_CH                                        # 20
MODEL_NAME = 'Mamba-ETER'
_ORIG_BUILD = R.build_model                              # 원 build_model (D 도 같은 것을 잡고 있음)
_ORIG_GROUPS = R.build_groups                            # 원 build_groups (run_job 안에서는 하위 텐서 그룹을 더한 판으로 교체)
REF_DIR = os.path.join(_PROJECT_ROOT, 'results', 'smoke_recipe_fix')
# gru:DL step 0 (results/smoke_recipe_fix/gru_DL.jsonl 실측 — 파일이 없을 때만 쓰는 값)
REF_GRU_STEP0 = dict(unet_all_rms=0.009155682296828884, unet_first_conv_zf_in_rms=0.02668183483183384)
# 하네스 seed_all(1) → build_model('gru') 의 U-Net checksum (v8fix 식; sanity_mamba1d.py [unetinit] 실측 2026-10-11)
REF_GRU_UNET_CHECKSUM_SEED1 = 780.8679536847122
UNET_INIT_RTOL = 1e-6                                    # 다른 U-Net 초기값끼리의 차이는 ≈2e-4 (gru vs unet)
REF_NUM_WORKERS = 16                                     # gru_DL_summary.json num_workers (워커 수가 augment rng 배분을 바꾼다)
FALLBACK_STAGES = ('h', 'v')                             # fallback = 두 단계 전체 fp32 (h 단 기울기도 v 단 fp16 역전파를 거침)
FALLBACK_RULES = dict(amp_skipped_steps_gt=20, nonfinite_loss_skips_gt=0, min_scaler_scale_lt=1024.0,
                      status_eq='aborted_nonfinite', seq_verdict_in=['frozen', 'eps-limited'],
                      preflight_fp16_underflow='in_proj/out_proj 기울기 fp16 vs fp32 상대오차 > 0.1 또는 0 으로 떨어진 비율 > 0.1',
                      fallback_stages='hv')
UNDERFLOW_REL_TH = 0.1
UNDERFLOW_ZERO_TH = 0.1
GRAD_PF_LOSS_SCALE = 32768.0 / 8                         # 기준 gru:DL 의 정상 GradScaler 배율 ÷ BS 8 (batch-mean 손실의 표본당 중간 기울기)
ADAM_EPS = R.CONDS[HARNESS_COND]['adam_eps']             # 1e-8
OUTMATCH_SLICES = 8
INITS = ('default', 'outmatch')
MAMBA_SUBS = ('in_proj', 'conv1d', 'x_proj', 'dt_proj', 'out_proj')
# 하위 텐서 (정·역방향 합산 그룹 seqsub.<stage>.<sub> — 하네스 per-parameter 통계를 그대로 합산)
MAMBA_PARAM_SUBS = ('in_proj.weight', 'conv1d.weight', 'conv1d.bias', 'x_proj.weight', 'dt_proj.weight', 'dt_proj.bias',
                    'A_log', 'D', 'out_proj.weight')
DELTA_SUBS = ('dt_proj.weight', 'dt_proj.bias', 'x_proj.weight', 'A_log')
NONDET_NOTE = ('mamba_ssm selective_scan·causal_conv1d 역전파는 dB·dC·conv dweight 를 atomicAdd 로 누적 — 비결정적(같은 시드에서도 비트 '
               '재현 아님, 단일 실행). cudnn.deterministic=True 는 cuDNN(기준 GRU)에만 적용.')


def check_versions():
    """R·D 의 판 고정 (R 의 gru 분기가 바뀌면 조용히 Mamba-ETER 에도 적용되는 것을 막는다). 다르면 거부(종료 코드 2)."""
    if R.SCRIPT_VERSION != REQ_SMOKE_VERSION:
        R.refuse(f'smoke_recipe_fix 판이 다르다: {R.SCRIPT_VERSION} ≠ {REQ_SMOKE_VERSION} (하네스 gru 분기 재확인 필요)')
    if D.DIAG_VERSION != REQ_DIAG_VERSION:
        R.refuse(f'diag_ss2d_variants 판이 다르다: {D.DIAG_VERSION} ≠ {REQ_DIAG_VERSION}')


# ------------------------------------------------------------------ CPU 경로 (sanity_mamba1d.py 도 import)
def selective_scan_cpu_full(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                            return_last_state=False):
    """CPU 전용 selective scan — mamba_ssm selective_scan_ref 와 같은 점화식·같은 인자(z 게이트·delta_bias·softplus 포함):
      Δ = softplus(Δ + bias),  x_t = exp(Δ_t A)·x_{t−1} + Δ_t B_t u_t,  y_t = C_t·x_t + D u_t,  out = y·SiLU(z).
    실수 A, B·C = (B, N, L) 만 지원(Mamba 블록의 호출 형태). selective_scan_ref 는 루프에서 deltaA[:, :, i] 로 큰 4D 텐서를 인덱싱해
    역전파 때 step 마다 전체 크기 기울기를 만든다(E=6192, L=384 에서 매우 느림). 여기서는 같은 텐서를 unbind 로 한 번에 나눠 역전파가
    한 번의 stack 이 되게 했을 뿐 수식은 같다(smoke_recipe_fix.selective_scan_cpu 와 같은 방식 + z·delta_bias·softplus).
    계산 dtype = max(입력, fp32) — float64 입력은 float64 로 계산(의존성 시험용), 반환은 입력 dtype (selective_scan_ref 와 같음)."""
    assert not A.is_complex() and B.dim() == 3 and C.dim() == 3, 'selective_scan_cpu_full: 실수 A, B·C=(B,N,L) 만 지원'
    dtype_in = u.dtype
    ct = torch.promote_types(dtype_in, torch.float32)
    u, delta = u.to(ct), delta.to(ct)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].to(ct)
    if delta_softplus:
        delta = F.softplus(delta)
    A, B, C = A.to(ct), B.to(ct), C.to(ct)
    deltaA = torch.exp(torch.einsum('bdl,dn->lbdn', delta, A)).contiguous()        # (L, b, d, n)
    deltaB_u = torch.einsum('bdl,bnl,bdl->lbdn', delta, B, u).contiguous()
    x = A.new_zeros((u.shape[0], A.shape[0], A.shape[1]))
    ys = []
    for dA, dBu, c in zip(deltaA.unbind(0), deltaB_u.unbind(0), C.unbind(2)):
        x = dA * x + dBu
        ys.append(torch.einsum('bdn,bn->bd', x, c))
    y = torch.stack(ys, dim=2)                                                      # (b, d, L)
    out = y if D is None else y + u * D.to(ct)[:, None]
    if z is not None:
        out = out * F.silu(z.to(ct))
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, x)


def install_mamba_cpu_shims():
    """CPU 전용 (이 프로세스 안에서만): Mamba 블록을 slow path + CPU 점화식으로. 원본 파일은 건드리지 않는다
    (visualize_multimodel_compare.py:128 의 SS2D 교체와 같은 런타임 이름 교체)."""
    MS.causal_conv1d_fn = None              # fast path 조건(causal_conv1d_fn is not None) 해제 → nn.Conv1d(+SiLU) slow path
    MS.selective_scan_fn = selective_scan_cpu_full
    print('  [CPU] mamba_simple.causal_conv1d_fn → None (slow path), selective_scan_fn → selective_scan_cpu_full')


@contextlib.contextmanager
def mamba_reference_path(scan=selective_scan_ref):
    """잠시 참조 경로(torch conv1d + 참조 scan)로 바꿨다가 되돌린다 (GPU preflight 의 fast path 대조용)."""
    old = (MS.causal_conv1d_fn, MS.selective_scan_fn)
    MS.causal_conv1d_fn, MS.selective_scan_fn = None, scan
    try:
        yield
    finally:
        MS.causal_conv1d_fn, MS.selective_scan_fn = old


# ------------------------------------------------------------------ 공통
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, path)


def _rms(t):
    t = t.detach().float()
    return float(t.pow(2).mean().sqrt())


def unet_checksum(model):
    """main_train_pure_v8fix.py 와 같은 U-Net 초기값 checksum (Σ_i (i+1)·Σ p_i, fp64)."""
    with torch.no_grad():
        return float(sum((p.double() * (i + 1)).sum() for i, p in enumerate(model.unet.parameters())))


def vname_of(args):
    v = 'mamba1d'
    if args.init == 'outmatch':
        v += '_outmatch'
    if args.unet_init != 'gru':
        v += f'_unetinit-{args.unet_init}'
    if args.fp32_stages:
        v += '_fp32' + ''.join(args.fp32_stages)
    return v


def model_kwargs(args):
    """build_model('gru') 와 같은 config 인자 + Mamba 차원(모델 기본값)."""
    return dict(dim=C.IMAGE_SIZE[0], n_coil=C.N_COIL, n_hidden_1=C.N_HIDDEN_LRNN_1, n_hidden_2=C.N_HIDDEN_LRNN_2,
                unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF, use_dc=False, fp32_stages=tuple(args.fp32_stages))


def seq_stages(model, x_ksp):
    """PureETER_GRU._seq 와 같은 식을 단계별로 돌려준다: (out_h_raw (B, 384, 2·H1·384), out_v (B, 2·H2, H, W))."""
    B = x_ksp.size(0)
    in_h = x_ksp.reshape([B, model.num_in_x, -1])
    out_h_raw, _ = model.gru_h(in_h)
    out_h = out_h_raw.reshape([B, model.num_in_x, model.num_out_y, -1]).permute(0, 2, 1, 3).reshape([B, model.num_out_y, -1])
    out_v, _ = model.gru_v(out_h)
    out_v = out_v.reshape([B, model.num_out_y, model.num_out_x, -1]).permute(0, 3, 2, 1)
    return out_h_raw, out_v


# ------------------------------------------------------------------ builder (R.build_model 자리)
def copy_unet_init(model, mode, rng_state):
    """mode 'gru': rng_state(= builder 호출 시점, run_job 의 seed_all 직후)에서 원 build_model('gru') 를 CPU 에 만들어 U-Net 만 복사 →
    스모크 gru:DL 의 U-Net 초기값과 비트 단위로 같다. 'unet': 같은 방식으로 PureETER_UNET (v8fix 관례). 'own': 그대로.
    전역 RNG(torch·numpy·python)는 복사 전 상태로 되돌린다."""
    if mode == 'own':
        return 'own (Mamba-ETER 생성 순서 그대로)'
    seq = {'gru': 'gru', 'unet': 'unet'}[mode]
    np_state, py_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng_state)
        ref = _ORIG_BUILD(seq, torch.device('cpu'))
    np.random.set_state(np_state)
    random.setstate(py_state)
    sd = ref.unet.state_dict()
    assert list(sd.keys()) == list(model.unet.state_dict().keys()), 'U-Net 구조 불일치 — 계약 위반'
    for k, v in model.unet.state_dict().items():
        assert v.shape == sd[k].shape, f'U-Net 모양 불일치 {k}'
    model.unet.load_state_dict(sd)
    del ref, sd
    return f'copied from smoke-harness build_model({seq!r}) at the builder-entry RNG state (= right after seed_all(seed))'


@torch.no_grad()
def apply_outmatch(model, k_h, k_v):
    for st, k in (('gru_h', k_h), ('gru_v', k_v)):
        mod = getattr(model, st)
        mod.fwd.out_proj.weight.mul_(k)
        mod.bwd.out_proj.weight.mul_(k)


def make_builder(args):
    """R.build_model 과 같은 시그니처 (seq, device) → Mamba-ETER. 시퀀스 모듈 초기값 = 호출 시점 RNG (GRU 와 같은 위치)."""
    def build(seq, device):
        assert seq == HARNESS_SEQ, f'Mamba-ETER builder 는 하네스 라벨 {HARNESS_SEQ!r} 로만 부른다 (받음 {seq!r})'
        state0 = torch.get_rng_state()
        model = PureETER_Mamba1D(**model_kwargs(args))
        note = copy_unet_init(model, args.unet_init, state0)
        if args.init == 'outmatch':
            assert args.k_h is not None and args.k_v is not None, 'outmatch 배율이 보정되지 않았다'
            apply_outmatch(model, args.k_h, args.k_v)
        model._unet_init_note = note
        return model.to(device)
    return build


def build_groups_mamba(model, seq):
    """R.build_groups 의 자리 (run_job 동안만 교체): 원 그룹(seq·seq.gru_h·seq.gru_v·U-Net — 판정·콘솔에 쓰는 키는 그대로)에
    Mamba 하위 텐서 그룹 seqsub.<stage>.<sub>(정·역방향 합산)을 더한다. 그룹은 겹쳐도 되고(R 문서) 통계는 파라미터별로 한 번 계산해
    합산하므로 학습·판정에는 영향이 없다 — 하위 텐서별 eps_dom_frac·upd_ratio·relΔ·grad_norm 이 jsonl·summary 에 남는다."""
    g = _ORIG_GROUPS(model, seq)
    if isinstance(model, PureETER_Mamba1D):
        names = {n for n, _ in model.named_parameters()}
        for st in ('gru_h', 'gru_v'):
            for sub in MAMBA_PARAM_SUBS:
                lst = [n for n in (f'{st}.fwd.{sub}', f'{st}.bwd.{sub}') if n in names]
                if lst:
                    g[f'seqsub.{st}.{sub}'] = lst
    return g


# ------------------------------------------------------------------ outmatch 보정
@torch.no_grad()
def _stage_rms_over(model, fitset, device):
    sh = sv = 0.0
    nh = nv = 0
    for xk, xi in fitset:
        xk, xi = xk.to(device), xi.to(device)
        xk, _ = R.normalize_inputs(xk, xi, R.per_sample_scale(xi))
        oh, ov = seq_stages(model, xk)
        sh += float(oh.float().pow(2).sum())
        nh += oh.numel()
        sv += float(ov.float().pow(2).sum())
        nv += ov.numel()
    return math.sqrt(sh / nh), math.sqrt(sv / nv)


@torch.no_grad()
def calibrate_outmatch(args, train_ds, device):
    """같은 시드의 ETER-net 단계 출력 RMS 에 Mamba-ETER 단계 출력 RMS 를 맞추는 out_proj 배율 (k_h 먼저, 그 상태에서 k_v).
    fp32·eval·autocast 없음, train 8 슬라이스(D.load_fit_set: 고르게 분산, rng = seed+4242)."""
    t0 = time.time()
    fitset = D.load_fit_set(train_ds, args.seed, OUTMATCH_SLICES)
    R.seed_all(args.seed)
    gru = _ORIG_BUILD('gru', device).eval()
    g_h, g_v = _stage_rms_over(gru, fitset, device)
    del gru
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    R.seed_all(args.seed)
    m = PureETER_Mamba1D(**model_kwargs(args), verbose=False).to(device).eval()
    m_h0, m_v0 = _stage_rms_over(m, fitset, device)
    k_h = g_h / max(m_h0, 1e-30)
    apply_outmatch(m, k_h, 1.0)
    m_h1, m_v1 = _stage_rms_over(m, fitset, device)
    k_v = g_v / max(m_v1, 1e-30)
    apply_outmatch(m, 1.0, k_v)
    m_h2, m_v2 = _stage_rms_over(m, fitset, device)
    del m
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    res = OrderedDict(k_h=k_h, k_v=k_v, n_slices=len(fitset), fit_rng_seed=args.seed + 4242,
                      gru_rms_h=g_h, gru_rms_out=g_v, mamba_rms_h_default=m_h0, mamba_rms_out_default=m_v0,
                      mamba_rms_out_after_kh=m_v1, mamba_rms_h_matched=m_h2, mamba_rms_out_matched=m_v2,
                      seconds=round(time.time() - t0, 1))
    print(f'  [outmatch] ETER-net RMS h {g_h:.4g} out {g_v:.4g} | Mamba-ETER default h {m_h0:.3g} out {m_v0:.3g} '
          f'→ k_h {k_h:.4g}, k_v {k_v:.4g} → h {m_h2:.4g} out {m_v2:.4g} ({res["seconds"]:.0f}s)', flush=True)
    return res


# ------------------------------------------------------------------ preflight
def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30))


def preflight(args, builder, probe, device):
    is_cuda = device.type == 'cuda'
    out = OrderedDict()
    if is_cuda:
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(12345)
            blk = MambaOut(256, 512, 96).to(device)
            x = torch.randn(2, 384, 256, device=device, requires_grad=True)
            pnames = [n for n, _ in blk.named_parameters()]
            params = [p for _, p in blk.named_parameters()]

            def fwd_bwd():
                y = blk(x)
                g = torch.autograd.grad(y.square().sum(), [x] + params)
                return y.detach(), g

            y_fast, g_fast = fwd_bwd()
            with mamba_reference_path():
                y_ref, g_ref = fwd_bwd()
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    y16 = blk(x)
            prel = OrderedDict((n, _rel(a, b)) for n, a, b in zip(pnames, g_fast[1:], g_ref[1:]))
            out['fastpath_vs_ref'] = dict(rel_err_fp32=_rel(y_fast, y_ref), rel_err_grad_fp32=_rel(g_fast[0], g_ref[0]),
                                          rel_err_param_grads_fp32=prel, rel_err_param_grads_max=max(prel.values()),
                                          rel_err_fp16_autocast=_rel(y16, y_ref),
                                          fast_path_active=MS.causal_conv1d_fn is not None)
            del blk, x, y_fast, y_ref, g_fast, g_ref, y16, params
        fp = out['fastpath_vs_ref']
        print(f'  [preflight] fast path vs 참조 (작은 블록): fp32 {fp["rel_err_fp32"]:.2e} (입력 grad {fp["rel_err_grad_fp32"]:.2e}, '
              f'파라미터 grad 최대 {fp["rel_err_param_grads_max"]:.2e} '
              f'{ {k: float(f"{v:.1e}") for k, v in fp["rel_err_param_grads_fp32"].items()} }), '
              f'fp16 autocast {fp["rel_err_fp16_autocast"]:.2e}, fast_path_active={fp["fast_path_active"]}', flush=True)
        assert fp['fast_path_active'], 'causal_conv1d 를 찾지 못해 fast path 가 꺼져 있다'
        assert fp['rel_err_fp32'] < 1e-4 and fp['rel_err_grad_fp32'] < 1e-3, 'fast path 와 참조 경로가 fp32 에서 다르다'
        assert fp['rel_err_param_grads_max'] < 5e-2, 'fast path 와 참조 경로의 파라미터 기울기가 fp32 에서 크게 다르다'
        assert fp['rel_err_fp16_autocast'] < 2e-2, 'fp16 autocast 오차가 크다'
    else:
        out['fastpath_vs_ref'] = 'skipped (cpu — fast path 없음)'
    # 실제 모델 probe 1 슬라이스 (학습과 같은 autocast) + 실제 크기 h 단 fast vs 참조 (fp32, no grad)
    with torch.random.fork_rng(devices=[device.index or 0] if is_cuda else []):
        R.seed_all(args.seed)
        m = builder(HARNESS_SEQ, device).eval()
        b = probe[0]
        xk, xi = b['data'].to(device), b['data_img'].to(device)
        xk, xi = R.normalize_inputs(xk, xi, R.per_sample_scale(xi))
        if is_cuda:
            with torch.no_grad():
                in_h = xk.reshape([xk.size(0), m.num_in_x, -1]).float()
                y_fast, _ = m.gru_h(in_h)
                with mamba_reference_path():
                    y_ref, _ = m.gru_h(in_h)
                out['real_size_h_fast_vs_ref'] = dict(rel_err_fp32=_rel(y_fast, y_ref), out_rms=_rms(y_ref))
                del in_h, y_fast, y_ref
            rh = out['real_size_h_fast_vs_ref']
            print(f'  [preflight] 실제 크기 h 단 fast vs 참조 (fp32): {rh["rel_err_fp32"]:.2e} (출력 RMS {rh["out_rms"]:.3g})',
                  flush=True)
            assert rh['rel_err_fp32'] < 1e-3, '실제 크기 h 단에서 fast path 와 참조 경로가 다르다'
        else:
            out['real_size_h_fast_vs_ref'] = 'skipped (cpu — fast path 없음)'
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=is_cuda):
            oh, ov = seq_stages(m, xk)
            out_full = m(xi, xk)
        out['probe_forward'] = dict(finite=bool(torch.isfinite(oh).all() and torch.isfinite(ov).all()
                                                and torch.isfinite(out_full).all()),
                                    h_out_rms=_rms(oh), seq_out_rms=_rms(ov), zf_in_rms=_rms(xi),
                                    seq_over_zf_rms=_rms(ov) / max(_rms(xi), 1e-30), out_rms=_rms(out_full),
                                    dtype_h=str(oh.dtype), dtype_seq=str(ov.dtype), seconds=round(time.time() - t0, 2),
                                    gpu_mem_alloc_gb=round(torch.cuda.memory_allocated() / 2 ** 30, 2) if is_cuda else None)
        del m, oh, ov, out_full
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()
    pf = out['probe_forward']
    print(f'  [preflight] probe forward: finite={pf["finite"]} h RMS {pf["h_out_rms"]:.3g} seq RMS {pf["seq_out_rms"]:.3g} '
          f'(zf {pf["zf_in_rms"]:.3g}, 비 {pf["seq_over_zf_rms"]:.3g}) dtype {pf["dtype_seq"]} {pf["seconds"]}s', flush=True)
    assert pf['finite'], 'probe forward 가 유한하지 않다'
    return out


# ------------------------------------------------------------------ 기울기 규모 preflight (Adam eps·fp16 언더플로)
def seq_param_keys(model):
    """시퀀스 하위 텐서 묶음 → 파라미터 이름 (정·역방향 합산). Mamba: <stage>.<sub>, GRU: <stage>.<weight_ih_l0 등>."""
    names = [n for n, _ in model.named_parameters() if n.startswith(('gru_h.', 'gru_v.'))]
    keys = OrderedDict()
    for n in names:
        st, rest = n.split('.', 1)
        if isinstance(model, PureETER_Mamba1D):
            rest = rest.split('.', 1)[1]                        # fwd./bwd. 제거
        else:
            rest = rest.replace('_reverse', '')
        keys.setdefault(f'{st}.{rest}', []).append(n)
    return keys


@torch.no_grad()
def _grad_stats(params, eps):
    numel = sum(p.grad.numel() for p in params)
    sq = sum(float(p.grad.double().pow(2).sum()) for p in params)
    lt = sum(int((p.grad.abs() < eps).sum()) for p in params)
    z = sum(int((p.grad == 0).sum()) for p in params)
    return dict(numel=numel, rms=math.sqrt(sq / numel), frac_lt_eps=lt / numel, frac_exact_zero=z / numel)


def _dl_loss(model, b, lam, crit, amp):
    xk, xi = b['xk'], b['xi']
    with torch.amp.autocast('cuda', enabled=amp):
        out = model(xi, xk)
    out = out.float()
    bm = b['bm']
    l1 = ((out - b['ref']).abs() * bm).sum() / bm.sum().clamp(min=1.0)
    return l1 + lam * (1 - crit(out, b['ref'], mask=bm))


def _grad_side(model, b, lam, crit, is_cuda, eps):
    """fp32 역전파 → 하위 텐서별 통계; GPU 면 학습과 같은 fp16 autocast 역전파(손실 × GRAD_PF_LOSS_SCALE)와 텐서별 비교."""
    keys = seq_param_keys(model)
    pmap = dict(model.named_parameters())
    model.train()
    model.zero_grad(set_to_none=True)
    loss32 = _dl_loss(model, b, lam, crit, False)
    loss32.backward()
    res = OrderedDict(loss_fp32=float(loss32))
    res['fp32'] = OrderedDict((k, _grad_stats([pmap[n] for n in ns], eps)) for k, ns in keys.items())
    seq_ps = [pmap[n] for ns in keys.values() for n in ns]
    res['fp32_seq_total'] = _grad_stats(seq_ps, eps)
    res['unet_grad_rms_fp32'] = _grad_stats([p for n, p in model.named_parameters() if n.startswith('unet.')], eps)['rms']
    if is_cuda:
        g32 = {n: p.grad.detach().to('cpu') for n, p in pmap.items() if n.startswith(('gru_h.', 'gru_v.'))}
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        loss16 = _dl_loss(model, b, lam, crit, True)
        (loss16 * GRAD_PF_LOSS_SCALE).backward()
        cmp_ = OrderedDict()
        with torch.no_grad():
            for k, ns in keys.items():
                dsq = rsq = 0.0
                nz32 = z16 = nonfin = 0
                for n in ns:
                    a32 = g32[n].to(device=pmap[n].grad.device)
                    a16 = pmap[n].grad.float() / GRAD_PF_LOSS_SCALE
                    nonfin += int((~torch.isfinite(a16)).sum())
                    dsq += float((a16 - a32).double().pow(2).sum())
                    rsq += float(a32.double().pow(2).sum())
                    m32 = a32 != 0
                    nz32 += int(m32.sum())
                    z16 += int(((a16 == 0) & m32).sum())
                    del a32, a16, m32
                cmp_[k] = dict(rel_err=math.sqrt(dsq / rsq) if rsq > 0 else None,
                               frac_zero_in_fp16_of_fp32_nonzero=(z16 / nz32) if nz32 else None, n_nonfinite=nonfin)
        res['loss_fp16'] = float(loss16)
        res['fp16_vs_fp32'] = cmp_
        del g32
    model.zero_grad(set_to_none=True)
    return res


def grad_preflight(args, builder, probe, device, lam):
    """실제 크기 probe 1 슬라이스(probe[0])·학습과 같은 DL 손실(정규화·masked L1 + λ/s̄·(1−SSIM))에서 시퀀스 하위 텐서별 기울기 규모를
    Adam eps 와 비교하고 같은 시드의 ETER-net 과 나란히 기록한다 (BS 1 — batch-mean 손실의 BS 8 와 같은 차수). GPU 에서는 fp16 autocast
    역전파(학습과 같은 fp32_stages)의 언더플로를 fp32 와 대조한다 — 손실 배율 = 기준 gru:DL 정상 배율 32768 ÷ 8. 기록용이며,
    언더플로 판정(in_proj/out_proj)만 fallback 규칙에 쓴다."""
    is_cuda = device.type == 'cuda'
    t0 = time.time()
    p0 = probe[0]
    xk, xi = p0['data'].to(device), p0['data_img'].to(device)
    s = R.per_sample_scale(xi)
    xk, xi = R.normalize_inputs(xk, xi, s)
    b = dict(xk=xk, xi=xi, ref=p0['label'].to(device) / s, bm=p0['brain_mask'].to(device))
    crit = R.SSIM().to(device)
    res = OrderedDict(probe_slice=0, loss_scale_fp16=GRAD_PF_LOSS_SCALE if is_cuda else None, adam_eps=ADAM_EPS, lam=lam,
                      note='한 step 의 |g| 와 eps 비교는 대리 지표다(Adam √v̂ 는 여러 step 의 RMS). 미표본 k-space 열이 0 이라 '
                           'in_proj/W_ih 의 많은 원소 기울기는 구조적으로 0 이다(두 모델 공통).')
    for kind in ('mamba', 'gru'):
        with torch.random.fork_rng(devices=[device.index or 0] if is_cuda else []):
            R.seed_all(args.seed)
            m = builder(HARNESS_SEQ, device) if kind == 'mamba' else _ORIG_BUILD('gru', device)
            res[kind] = _grad_side(m, b, lam, crit, is_cuda, ADAM_EPS)
            del m
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()
    mm, gg = res['mamba']['fp32'], res['gru']['fp32']
    res['in_proj_rms_over_gru_W_ih'] = OrderedDict(
        (st, mm[f'{st}.in_proj.weight']['rms'] / max(gg[f'{st}.weight_ih_l0']['rms'], 1e-300)) for st in ('gru_h', 'gru_v'))
    reasons = []
    if is_cuda and 'fp16_vs_fp32' in res['mamba']:
        for k, v in res['mamba']['fp16_vs_fp32'].items():
            if not k.endswith(('in_proj.weight', 'out_proj.weight')):
                continue
            if v['rel_err'] is not None and v['rel_err'] > UNDERFLOW_REL_TH:
                reasons.append(f'{k} fp16 rel_err {v["rel_err"]:.3g} > {UNDERFLOW_REL_TH}')
            zf = v['frac_zero_in_fp16_of_fp32_nonzero']
            if zf is not None and zf > UNDERFLOW_ZERO_TH:
                reasons.append(f'{k} fp16 0 비율 {zf:.3g} > {UNDERFLOW_ZERO_TH}')
            if v['n_nonfinite']:
                reasons.append(f'{k} fp16 non-finite {v["n_nonfinite"]}')
    res['fp16_underflow'] = OrderedDict(triggered=bool(reasons), reasons=reasons, evaluated=is_cuda,
                                        thresholds=dict(rel_err=UNDERFLOW_REL_TH, zero_frac=UNDERFLOW_ZERO_TH))
    res['seconds'] = round(time.time() - t0, 1)
    ms, gs = res['mamba']['fp32_seq_total'], res['gru']['fp32_seq_total']
    print(f'  [grad-preflight] seq |g| RMS Mamba-ETER {ms["rms"]:.2e} (<eps {100 * ms["frac_lt_eps"]:.1f}%) vs ETER-net '
          f'{gs["rms"]:.2e} (<eps {100 * gs["frac_lt_eps"]:.1f}%); in_proj/W_ih RMS 비 h {res["in_proj_rms_over_gru_W_ih"]["gru_h"]:.3g} '
          f'v {res["in_proj_rms_over_gru_W_ih"]["gru_v"]:.3g}; U-Net grad RMS {res["mamba"]["unet_grad_rms_fp32"]:.2e} / '
          f'{res["gru"]["unet_grad_rms_fp32"]:.2e}; fp16 언더플로 '
          f'{res["fp16_underflow"]["triggered"] if is_cuda else "평가 안 함(cpu)"} '
          f'{reasons[:4]} ({res["seconds"]:.0f}s)', flush=True)
    for k, v in mm.items():
        c = (res['mamba'].get('fp16_vs_fp32') or {}).get(k) or {}
        print(f'      {k:24s} rms {v["rms"]:.2e} <eps {100 * v["frac_lt_eps"]:5.1f}% 0 {100 * v["frac_exact_zero"]:5.1f}%'
              + (f' | fp16 rel_err {_f(c.get("rel_err"), ".3g")} 0화 {_f(c.get("frac_zero_in_fp16_of_fp32_nonzero"), ".3g")}'
                 if c else ''), flush=True)
    return res


# ------------------------------------------------------------------ 학습 뒤 분석 (Mamba 전용 — D.post_analysis 는 그대로 재사용)
class SeqProxy:
    """D.post_analysis 가 요구하는 인터페이스(.eval()/.train()/.unet/.ss2d(xk))를 Mamba-ETER 에 연결 (ss2d = model._seq)."""

    def __init__(self, model):
        self.model = model
        self.unet = model.unet

    def ss2d(self, xk):
        return self.model._seq(xk)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self


@torch.no_grad()
def weight_norm_ratios_mamba(model, model0):
    """일반판 weight norm 비: gru_{h,v}.{fwd,bwd}.{in_proj,conv1d,x_proj,dt_proj,out_proj}.weight 의 끝/초기 norm 비·cos(Δw, w0)
    + U-Net 첫 conv 의 시퀀스/zero-filled 입력 slice (D.weight_norm_ratios — ss2d 키는 없으므로 건너뛰고 U-Net 항목만 나온다)."""
    sd, sd0 = model.state_dict(), model0.state_dict()
    out = OrderedDict()
    for st in ('gru_h', 'gru_v'):
        for d in ('fwd', 'bwd'):
            for sub in MAMBA_SUBS:
                k = f'{st}.{d}.{sub}.weight'
                a, b = sd[k].float(), sd0[k].float()
                out[k] = dict(norm_end=float(a.norm()), norm_init=float(b.norm()),
                              ratio=float(a.norm() / b.norm().clamp(min=1e-30)),
                              cos_delta_w0=float(F.cosine_similarity((a - b).flatten(), b.flatten(), dim=0)),
                              rel_change=float((a - b).norm() / b.norm().clamp(min=1e-30)))
            # x_proj 의 Δ 행(앞 dt_rank 행)과 B·C 행을 나눠서 (Δ 경로 학습 여부 — 한계 항목)
            blk = getattr(getattr(model, st), d)
            R_ = blk.dt_rank
            a, b = sd[f'{st}.{d}.x_proj.weight'].float(), sd0[f'{st}.{d}.x_proj.weight'].float()
            for part, sl in (('dt_rows', slice(0, R_)), ('BC_rows', slice(R_, None))):
                out[f'{st}.{d}.x_proj.weight[{part}]'] = dict(
                    norm_end=float(a[sl].norm()), norm_init=float(b[sl].norm()),
                    ratio=float(a[sl].norm() / b[sl].norm().clamp(min=1e-30)),
                    rel_change=float((a[sl] - b[sl]).norm() / b[sl].norm().clamp(min=1e-30)))
            for k in (f'{st}.{d}.A_log', f'{st}.{d}.D', f'{st}.{d}.dt_proj.bias'):
                a, b = sd[k].float(), sd0[k].float()
                out[k] = dict(norm_end=float(a.norm()), norm_init=float(b.norm()),
                              rel_change=float((a - b).norm() / b.norm().clamp(min=1e-30)))
    out.update(D.weight_norm_ratios(model, model0))
    return out


@torch.no_grad()
def ssm_param_stats(model):
    """단계·방향별 Δ 기본값 softplus(dt_proj.bias), A = −exp(A_log), D 스킵의 요약."""
    out = OrderedDict()
    for st in ('gru_h', 'gru_v'):
        for d in ('fwd', 'bwd'):
            blk = getattr(getattr(model, st), d)
            dt = F.softplus(blk.dt_proj.bias.float())
            A = -torch.exp(blk.A_log.float())
            Dv = blk.D.float()
            out[f'{st}.{d}'] = dict(dt_mean=float(dt.mean()), dt_median=float(dt.median()), dt_min=float(dt.min()),
                                    dt_max=float(dt.max()), A_mean=float(A.mean()), A_min=float(A.min()),
                                    D_mean=float(Dv.mean()), D_std=float(Dv.std()),
                                    conv_bias_rms=_rms(blk.conv1d.bias))
    return out


@torch.no_grad()
def stage_probe(model, probe, device, is_cuda):
    """probe 슬라이스에서 단계별 출력 RMS(h, v), 방향별 에너지 비, f(0) 크기 (학습과 같은 정규화·autocast)."""
    model.eval()
    rows = []
    for b in probe:
        xk, xi = b['data'].to(device), b['data_img'].to(device)
        xk, xi = R.normalize_inputs(xk, xi, R.per_sample_scale(xi))
        with torch.amp.autocast('cuda', enabled=is_cuda):
            oh, ov = seq_stages(model, xk)
            f0 = model._seq(torch.zeros_like(xk))
        oh, ov = oh.float(), ov.float()
        Hh = oh.shape[-1] // 2
        eh = oh.pow(2)
        ev = ov.pow(2)
        Y = ov.shape[2]
        rows.append(dict(h_out_rms=_rms(oh), h_fwd_rms=_rms(oh[..., :Hh]), h_bwd_rms=_rms(oh[..., Hh:]),
                         h_fwd_energy_frac=float(eh[..., :Hh].sum() / eh.sum().clamp(min=1e-30)),
                         seq_out_rms=_rms(ov), zf_in_rms=_rms(xi), seq_over_zf_rms=_rms(ov) / max(_rms(xi), 1e-30),
                         v_fwd_rms=_rms(ov[:, :, :Y // 2]), v_bwd_rms=_rms(ov[:, :, Y // 2:]),
                         v_fwd_energy_frac=float(ev[:, :, :Y // 2].sum() / ev.sum().clamp(min=1e-30)),
                         f0_maxabs=float(f0.float().abs().max())))
    model.train()
    return OrderedDict((k, float(np.mean([r[k] for r in rows]))) for k in rows[0])


def run_post(args, builder, ckpt_path, val_ds, val_indices, probe, device):
    is_cuda = device.type == 'cuda'
    model = builder(HARNESS_SEQ, device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    del sd
    R.seed_all(args.seed)
    model0 = builder(HARNESS_SEQ, device)                 # 같은 시드로 다시 만든 초기 모델 (U-Net 초기값·outmatch 배율 포함)
    idx = val_indices[:args.post_slices]
    extra = OrderedDict(weight_norm_ratios=weight_norm_ratios_mamba(model, model0),
                        ssm_params=dict(end=ssm_param_stats(model), init=ssm_param_stats(model0)),
                        stage_probe=dict(end=stage_probe(model, probe, device, is_cuda),
                                         init=stage_probe(model0, probe, device, is_cuda)))
    extra['end'] = D.post_analysis(SeqProxy(model), val_ds, idx, device, args.val_workers)
    extra['init'] = D.post_analysis(SeqProxy(model0), val_ds, idx, device, args.val_workers)
    del model, model0
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()
    e, i0 = extra['end'], extra['init']
    wr = extra['weight_norm_ratios']
    print(f'[post] end: ssim {e["ssim_normal_mean"]:.4f} seq_out_rms {e["decomposition"]["seq_out_rms"]:.4g} '
          f'(init {i0["decomposition"]["seq_out_rms"]:.4g}) slice-specific rms {e["decomposition"]["slice_specific_spatial_rms"]:.4g} '
          f'| zero {e["substitution_tests"]["zero"]["mean"]:+.5f} slice_mean {e["substitution_tests"]["slice_mean_plane"]["mean"]:+.5f} '
          f'shuffle {e["substitution_tests"]["shuffle_cross"]["mean"]:+.5f} | grid R² zf {e["grid"]["r2_zf_rss"]:.3f} '
          f'logk {e["grid"]["r2_logk_sampled"]:.3f} | out_proj ratio h {wr["gru_h.fwd.out_proj.weight"]["ratio"]:.3f}/'
          f'{wr["gru_h.bwd.out_proj.weight"]["ratio"]:.3f} v {wr["gru_v.fwd.out_proj.weight"]["ratio"]:.3f}/'
          f'{wr["gru_v.bwd.out_proj.weight"]["ratio"]:.3f}', flush=True)
    return extra


# ------------------------------------------------------------------ jsonl 후처리·점검
def read_jsonl(path):
    meta, steps = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('type') == 'meta':
                meta = r
            elif r.get('type') == 'step':
                steps.append(r)
    return meta, steps


def rewrite_jsonl_meta(path, extra_meta):
    """R.run_job 이 쓴 jsonl 의 meta 줄에 Mamba-ETER 정보를 더해 다시 쓴다 (step 줄은 그대로)."""
    with open(path) as f:
        lines = f.readlines()
    meta = json.loads(lines[0])
    assert meta.get('type') == 'meta'
    meta.update(extra_meta)
    lines[0] = json.dumps(meta, ensure_ascii=False, default=float) + '\n'
    with open(path + '.tmp', 'w') as f:
        f.writelines(lines)
    os.replace(path + '.tmp', path)


def ref_gru_step0():
    p = os.path.join(REF_DIR, 'gru_DL.jsonl')
    try:
        _, steps = read_jsonl(p)
        s0 = steps[0]
        assert s0['step'] == 0
        return dict(unet_all_rms=s0['groups']['unet_all']['rms'],
                    unet_first_conv_zf_in_rms=s0['io_slices']['unet_first_conv']['zf_in']['rms'], source=p)
    except Exception as ex:                                             # noqa: BLE001
        return dict(REF_GRU_STEP0, source=f'constants (읽기 실패: {ex})')


def check_unet_init(steps, mode):
    s0 = steps[0]
    got = dict(unet_all_rms=s0['groups']['unet_all']['rms'],
               unet_first_conv_zf_in_rms=s0['io_slices']['unet_first_conv']['zf_in']['rms'])
    res = OrderedDict(mode=mode, got=got)
    if mode != 'gru':
        res['match'] = None
        res['note'] = 'gru:DL 과 대조하지 않음 (--unet-init gru 일 때만)'
        return res
    ref = ref_gru_step0()
    res['ref'] = ref
    rel = {k: abs(got[k] - ref[k]) / abs(ref[k]) for k in got}
    res['rel_diff'] = rel
    res['rtol'] = UNET_INIT_RTOL
    res['match'] = all(v <= UNET_INIT_RTOL for v in rel.values())
    return res


def precision_alarm(steps, status, summ=None, grad_pf=None):
    """사전 등록 fallback 규칙 (FALLBACK_RULES): 넘침(amp skip·non-finite·scaler 배율·abort) + 언더플로(GPU grad_preflight 의
    in_proj/out_proj fp16 대조) + 학습 상태(하네스 verdict.seq ∈ frozen/eps-limited). 하나라도 있으면 triggered."""
    amp = max((r.get('amp_skipped_steps') or 0) for r in steps) if steps else 0
    nonfin = max((r.get('nonfinite_loss_skips') or 0) for r in steps) if steps else 0
    scales = [r['scaler_scale'] for r in steps if r.get('scaler_scale') is not None]
    min_scale = min(scales) if scales else None
    reasons = []
    if amp > FALLBACK_RULES['amp_skipped_steps_gt']:
        reasons.append(f'amp_skipped_steps {amp} > {FALLBACK_RULES["amp_skipped_steps_gt"]}')
    if nonfin > FALLBACK_RULES['nonfinite_loss_skips_gt']:
        reasons.append(f'nonfinite_loss_skips {nonfin} > 0')
    if min_scale is not None and min_scale < FALLBACK_RULES['min_scaler_scale_lt']:
        reasons.append(f'min scaler_scale {min_scale:g} < {FALLBACK_RULES["min_scaler_scale_lt"]:g}')
    if status == FALLBACK_RULES['status_eq']:
        reasons.append(f'status {status}')
    uf = (grad_pf or {}).get('fp16_underflow') or {}
    if uf.get('triggered'):
        reasons.append('preflight fp16 언더플로: ' + '; '.join(uf.get('reasons') or []))
    vseq = ((summ or {}).get('verdict') or {}).get('seq')
    if vseq in FALLBACK_RULES['seq_verdict_in']:
        fg = ((summ or {}).get('final_groups') or {}).get('seq') or {}
        reasons.append(f'verdict.seq {vseq} (relΔ {_f(fg.get("rel_change"), ".3g")}, eps_dom {_f(fg.get("eps_dom_frac"), ".3f")})')
    return OrderedDict(triggered=bool(reasons), reasons=reasons, amp_skipped_steps=amp, nonfinite_loss_skips=nonfin,
                       min_scaler_scale=min_scale, seq_verdict=vseq, preflight_underflow=bool(uf.get('triggered')),
                       rules=FALLBACK_RULES)


def unet_init_precheck(model, args):
    """학습 전 U-Net 초기값 대조 (--unet-init gru): gru_DL.jsonl step 0 의 unet_all RMS·첫 conv zf slice RMS (rtol 1e-6) +
    seed 1 이면 하네스 gru checksum. 다르면 match False (호출자가 거부)."""
    res = OrderedDict(mode=args.unet_init)
    if args.unet_init != 'gru':
        res['match'] = None
        res['note'] = 'gru:DL 과 대조하지 않음 (--unet-init gru 일 때만)'
        return res
    with torch.no_grad():
        ps = [p.detach().double() for p in model.unet.parameters()]
        unet_rms = math.sqrt(sum(float(p.pow(2).sum()) for p in ps) / sum(p.numel() for p in ps))
    got = dict(unet_all_rms=unet_rms,
               unet_first_conv_zf_in_rms=R.io_slice_stats(model)['unet_first_conv']['zf_in']['rms'])
    ref = ref_gru_step0()
    rel = {k: abs(got[k] - ref[k]) / abs(ref[k]) for k in got}
    res.update(got=got, ref=ref, rel_diff=rel, rtol=UNET_INIT_RTOL)
    ok = all(v <= UNET_INIT_RTOL for v in rel.values())
    if args.seed == 1:
        cs = unet_checksum(model)
        res['checksum'] = cs
        res['checksum_ref_seed1'] = REF_GRU_UNET_CHECKSUM_SEED1
        ok = ok and abs(cs - REF_GRU_UNET_CHECKSUM_SEED1) <= 1e-9 * abs(REF_GRU_UNET_CHECKSUM_SEED1)
    res['match'] = ok
    return res


# ------------------------------------------------------------------ 한 조건 실행
class UnetInitRefused(Exception):
    pass


def param_table(model):
    tab = OrderedDict()
    for st in ('gru_h', 'gru_v'):
        for d in ('fwd', 'bwd'):
            blk = getattr(getattr(model, st), d)
            tab[f'{st}.{d}'] = OrderedDict((n, p.numel()) for n, p in blk.named_parameters())
            tab[f'{st}.{d}']['_total'] = sum(p.numel() for p in blk.parameters())
    n_seq = sum(p.numel() for n, p in model.named_parameters() if n.startswith(('gru_h.', 'gru_v.')))
    n_unet = sum(p.numel() for p in model.unet.parameters())
    gru_seq = (2 * gru_params_per_direction(C.IMAGE_SIZE[0] * C.N_COIL * 2, C.IMAGE_SIZE[0] * C.N_HIDDEN_LRNN_1)
               + 2 * gru_params_per_direction(2 * C.IMAGE_SIZE[0] * C.N_HIDDEN_LRNN_1, C.IMAGE_SIZE[0] * C.N_HIDDEN_LRNN_2))
    tab['seq_total'] = n_seq
    tab['unet_total'] = n_unet
    tab['model_total'] = n_seq + n_unet
    tab['gru_seq_total_analytic'] = gru_seq
    tab['seq_rel_diff_vs_gru'] = (n_seq - gru_seq) / gru_seq
    return tab


def run_condition(args, data, device):
    """한 조건(vname) 학습 + 검증 + post. 반환 (exit_code, final dict)."""
    train_ds, val_ds, val_indices, probe_indices, probe, dl_scale = data
    is_cuda = device.type == 'cuda'
    vname = vname_of(args)
    vdir = os.path.join(args.out_dir, vname)
    os.makedirs(vdir, exist_ok=True)
    final_path = os.path.join(args.out_dir, f'{vname}_final.json')
    jsonl_final = os.path.join(vdir, f'{vname}.jsonl')
    ckpt_final = os.path.join(vdir, f'{vname}_end.pt')
    print(f'\n########## {MODEL_NAME} 조건 {vname}: init={args.init} unet_init={args.unet_init} '
          f'fp32_stages={list(args.fp32_stages)} steps={args.steps} ##########', flush=True)

    meta = OrderedDict(
        runner_version=RUNNER_VERSION, model=MODEL_NAME, model_class='PureETER_Mamba1D',
        model_file='models/pure_eternet/u_pure_eternet_mamba1d.py', vname=vname,
        description=('ETER-net(PureETER_GRU) 의 데이터 흐름(_seq·forward 상속)을 그대로 두고 순환 셀 nn.GRU 만 양방향 Mamba 블록'
                     '(mamba_ssm 2.2.2, out_proj → 3840)으로 바꾼 모델. 레시피 = smoke_recipe_fix DL.'),
        harness_seq_label=HARNESS_SEQ, harness_cond=HARNESS_COND, seed=args.seed, steps=args.steps, bs=args.bs,
        init=args.init, unet_init=args.unet_init, fp32_stages=list(args.fp32_stages), started=now(), argv=sys.argv,
        device=str(device), val_slices=len(val_indices), val_slices_arg=args.val_slices, post_slices=args.post_slices, probe_indices=probe_indices,
        dl_scale=dl_scale, smoke_script_version=R.SCRIPT_VERSION, diag_version=D.DIAG_VERSION,
        sha256={'v8_eter_pure/smoke_recipe_fix.py': sha256_file(os.path.join(_HERE, 'smoke_recipe_fix.py')),
                'v8_eter_pure/diag_ss2d_variants.py': sha256_file(os.path.join(_HERE, 'diag_ss2d_variants.py')),
                'models/pure_eternet/u_pure_eternet_mamba1d.py':
                    sha256_file(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet', 'u_pure_eternet_mamba1d.py')),
                'v8_eter_pure/smoke_mamba1d.py': sha256_file(os.path.abspath(__file__))},
        versions=dict(torch=torch.__version__, mamba_ssm=_pkg_version('mamba_ssm'), causal_conv1d=_pkg_version('causal_conv1d'),
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if is_cuda else None),
        seq_in_dep_note='f(0)=0 이 정확히 성립해 seq_in_dep(‖f(x)−f(0)‖/‖f(0)‖)는 정의되지 않는다 — seq_out_rms 를 본다',
        console_tag_note="R.run_job 의 콘솔 태그 '[gru_DL]' 은 하네스 라벨일 뿐 모델은 Mamba-ETER",
        nondeterminism_note=NONDET_NOTE, num_workers=args.num_workers, val_workers=args.val_workers,
        max_minutes=args.max_minutes, fallback_rules=FALLBACK_RULES,
        interpretation_rules=('not shown 이면서 verdict.seq ∈ {frozen, eps-limited} 이면 해석 불가(옵티마이저·정밀도 수치); '
                              'SSM vs RNN 의 1차 근거는 outmatch 행; seqsub.*Δ 경로 eps_dom ≥ 0.9 면 Δ 선택성이 사실상 고정된 Mamba 를 시험한 것'),
    )
    status, summ, extra, err = 'failed', None, None, None
    unet_check, alarm, refused = None, None, None
    builder = make_builder(args)
    try:
        if args.init == 'outmatch':
            meta['outmatch'] = calibrate_outmatch(args, train_ds, device)
            args.k_h, args.k_v = meta['outmatch']['k_h'], meta['outmatch']['k_v']
        else:
            args.k_h = args.k_v = None
        meta['preflight'] = preflight(args, builder, probe, device)
        meta['grad_preflight'] = grad_preflight(args, builder, probe, device, C.LAMBDA_SSIM_PER_PIXEL / dl_scale['mean'])
        R.build_model = builder                                        # run_job 이 부르는 이름만 교체 (원 파일 무수정)
        R.build_groups = build_groups_mamba                            # 하위 텐서 통계 그룹 추가 (기록만 — 판정 키는 그대로)
        # 파라미터 표·U-Net 초기값 checksum: run_job 과 같은 절차(seed_all → builder)로 CPU 에 한 번 만들어 잰다 (run_job 이 다시 seed_all)
        R.seed_all(args.seed)
        with torch.random.fork_rng(devices=[]):
            _m = builder(HARNESS_SEQ, torch.device('cpu'))
            meta['param_table'] = param_table(_m)
            meta['mamba_cfg'] = _m.mamba_cfg
            meta['unet_init_checksum'] = unet_checksum(_m)
            meta['unet_init_note'] = _m._unet_init_note
            meta['unet_init_precheck'] = unet_init_precheck(_m, args)
            del _m
        gc.collect()
        upc = meta['unet_init_precheck']
        print(f'  U-Net 초기값 checksum {meta["unet_init_checksum"]:.10e} ({meta["unet_init_note"]}); 학습 전 gru:DL 대조 '
              f'{upc.get("match")} {upc.get("rel_diff", "")}', flush=True)
        if upc.get('match') is False:
            raise UnetInitRefused(f'U-Net 초기값이 gru:DL 과 다르다 (학습 전 대조): {upc}')
        ns = argparse.Namespace(
            seq=HARNESS_SEQ, cond=HARNESS_COND, jobs=None, steps=args.steps, tmax=None, tmax_eff=args.steps,
            log_every=args.log_every, log_dense_every=args.log_dense_every, log_dense_until=args.log_dense_until,
            val_slices=args.val_slices, seed=args.seed, lambda_ssim=C.LAMBDA_SSIM_PER_PIXEL, lambda_dl_scale=None,
            out_dir=vdir, save_ckpt='end', bs=args.bs, device=args.device, force=args.force, max_minutes=args.max_minutes,
            num_workers=args.num_workers, val_workers=args.val_workers, skip_done=False,
            job_list=[(HARNESS_SEQ, HARNESS_COND)], probe_indices=probe_indices, dl_scale=dl_scale)
        summ = R.run_job(HARNESS_SEQ, HARNESS_COND, ns, train_ds, val_ds, val_indices, probe, device, vdir)
        status = summ.get('status', 'completed')
        os.replace(os.path.join(vdir, f'{HARNESS_SEQ}_{HARNESS_COND}.jsonl'), jsonl_final)
        os.replace(os.path.join(vdir, f'{HARNESS_SEQ}_{HARNESS_COND}_end.pt'), ckpt_final)
        rewrite_jsonl_meta(jsonl_final, dict(model=MODEL_NAME, harness_seq_label=HARNESS_SEQ, runner_version=RUNNER_VERSION,
                                             vname=vname, init=args.init, unet_init=args.unet_init,
                                             fp32_stages=list(args.fp32_stages), mamba_cfg=meta['mamba_cfg'],
                                             outmatch=meta.get('outmatch'), seq_in_dep_note=meta['seq_in_dep_note']))
        _, steps = read_jsonl(jsonl_final)
        unet_check = check_unet_init(steps, args.unet_init)
        if unet_check['match'] is False:
            status += '+unet_init_mismatch'
            print(f'  [warn] U-Net 초기값이 gru:DL 과 다르다: {unet_check}', flush=True)
        else:
            print(f'  U-Net 초기값 대조: {unet_check.get("match")} {unet_check.get("rel_diff")}', flush=True)
        alarm = precision_alarm(steps, summ.get('status'), summ, meta.get('grad_preflight')) if is_cuda else OrderedDict(
            triggered=False, reasons=[], note='cpu — GradScaler 없음, 판정 안 함',
            would_trigger_on_cuda=precision_alarm(steps, summ.get('status'), summ, meta.get('grad_preflight'))['reasons'])
        if alarm['triggered']:
            print(f'  [precision] fp16 이상 징후: {alarm["reasons"]}', flush=True)
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()
        extra = run_post(args, builder, ckpt_final, val_ds, val_indices, probe, device)
    except KeyboardInterrupt:
        raise
    except UnetInitRefused as ex:
        refused = str(ex)
        status, err = 'refused_unet_init', refused
    except Exception as ex:                                             # noqa: BLE001
        err = f'{type(ex).__name__}: {ex}\n{traceback.format_exc()[-4000:]}'
        print(f'[FAIL] {vname}: {err}', flush=True)
        status = 'failed' if summ is None else f'{status}+post_failed'
    finally:
        R.build_model = _ORIG_BUILD
        R.build_groups = _ORIG_GROUPS
    meta['finished'] = now()
    final = OrderedDict(meta=meta, status=status, error=err, run_job_summary=summ, unet_init_check=unet_check,
                        precision_alarm=alarm, post=extra)
    write_json(final_path, final)
    rebuild_md(args.out_dir)
    if refused:
        R.refuse(refused)                                              # 종료 코드 2 (학습 전 — 결과 json 에 사유 기록)
    print(f'[{vname}] 끝 status={status} → {final_path}', flush=True)
    return (0 if summ is not None and err is None else R.EXIT_JOB_FAILED), final


def _pkg_version(name):
    try:
        import importlib.metadata as im
        return im.version(name.replace('_', '-'))
    except Exception:                                                   # noqa: BLE001
        try:
            return __import__(name).__version__
        except Exception:                                               # noqa: BLE001
            return None


# ------------------------------------------------------------------ 요약 md
def _f(x, fmt):
    return format(x, fmt) if isinstance(x, (int, float)) and not isinstance(x, bool) else '—'


def _ds(d):
    if not d:
        return '—'
    return f'{d["mean"]:+.4f} ± {_f(d.get("se"), ".4f")} ({d["n_pos"]}/{d["n"]}, p={_f(d.get("p_sign"), ".2g")})'


def _pctf(x):
    return f'{100 * x:.1f}%' if isinstance(x, (int, float)) and not isinstance(x, bool) else '—'


def interpret(s):
    """해석 규칙 (실행 전 고정 — 모듈 docstring)."""
    if not s:
        return '—'
    if s.get('status') != 'completed' or s.get('steps_done') != s.get('steps_requested'):
        return '비교 불가 (학습 미완)'
    vd = s.get('verdict') or {}
    used, vs = vd.get('seq_used'), vd.get('seq')
    if used is None:
        return '— (시퀀스 모듈 없음)'
    if used == 'used':
        return '사용'
    if vs in FALLBACK_RULES['seq_verdict_in']:
        return f'해석 불가 (seq {vs})'
    return 'not shown (결론 불가)'


def precision_label(fp32_stages):
    st = set(fp32_stages or [])
    if not st:
        return 'fp16 autocast'
    return ('h·v' if st == {'h', 'v'} else '·'.join(sorted(st))) + ' fp32 (U-Net fp16)'


def _row(name, s, precision='—', note=''):
    val = (s or {}).get('val') or {}
    v = val.get('normal') or {}
    t = val.get('tests') or {}
    vd = (s or {}).get('verdict') or {}
    fg = ((s or {}).get('final_groups') or {}).get('seq') or {}
    ip, fp = (s or {}).get('init_probe') or {}, (s or {}).get('final_probe') or {}
    steps = (s or {}).get('steps_done')
    if s and steps is not None and steps != s.get('steps_requested'):
        note += f' (중단: {steps}/{s.get("steps_requested")} step — 비교 불가)'
    sps = (s['train_sec'] / steps) if s and steps else None
    return (f'| {name} | {precision} | {(s or {}).get("status", "—")}{note} | {steps if steps is not None else "—"} '
            f'| {_f(v.get("ssim"), ".4f")} | {_f(v.get("psnr"), ".2f")} | {_f(v.get("nmse"), ".5f")} '
            f'| {_ds(t.get("shuffle_cross"))} | {_ds(t.get("shuffle_adj"))} | {_ds(t.get("zero"))} | {_ds(t.get("deep4"))} '
            f'| {vd.get("seq_used") or "—"} | {vd.get("seq") or "—"} | {_f(fg.get("rel_change"), ".3g")} '
            f'| {_pctf(fg.get("eps_dom_frac"))} | {_f(fg.get("upd_ratio_mean"), ".3g")} | **{interpret(s)}** '
            f'| {_f(ip.get("seq_out_rms"), ".3g")} → {_f(fp.get("seq_out_rms"), ".3g")} | {(s or {}).get("amp_skipped_steps", "—")} '
            f'| {_f(sps, ".3f")} | {_f((s or {}).get("peak_gpu_mem_gb"), ".1f")} |\n')


def rebuild_md(out_dir):
    lines = [f'# {MODEL_NAME} 스모크 — 요약 (자동 생성, smoke_mamba1d.py {RUNNER_VERSION})\n\n',
             'Mamba-ETER = ETER-net(PureETER_GRU)의 데이터 흐름을 그대로 두고 순환 셀 nn.GRU 만 양방향 Mamba 블록으로 바꾼 모델. '
             '모든 행은 같은 DL 레시피(smoke_recipe_fix)·SEED 1·데이터 순서(num_workers 16)·4000 step·검증 64 슬라이스(기준 행은 '
             'results/smoke_recipe_fix/*_DL_summary.json). dSSIM = 정상 − 시험 (양수 = 그 입력을 씀), ± SE, (양수 개수/n, 부호 검정 p). '
             '"사용" = mean>0 & mean>2SE & p<0.01 (shuffle_cross). "not shown" 은 4000 step 에서 결론 불가이지 "쓰지 않는다" 의 증거가 '
             '아니다. s/step = 학습 시간(기록 포함)/step.\n\n',
             '**해석 규칙 (실행 전 고정)**: (1) "not shown" 이면서 판정 seq 가 frozen(relΔ<0.01) 또는 eps-limited(√v̂<eps 원소 ≥ 90%) 인 '
             '행은 **해석 불가**(옵티마이저·정밀도 수치) — 셀 구조의 증거로 쓰지 않는다. (2) default 초기화는 시퀀스 출력이 ETER-net 의 '
             '≈1/600 이라 기울기 전 원소가 Adam eps 1e-8 아래에서 시작한다(표 4) — SSM vs RNN 의 1차 근거는 초기 이득을 맞춘 outmatch 행이고 '
             'default 는 초기 이득 교란 요인을 보여 주는 짝이다. (3) **한계**: DL 레시피(eps 1e-8)에서 Δ 경로(dt_proj·x_proj 의 Δ 행·A_log)의 '
             '기울기는 outmatch 에서도 eps 보다 2~3 자릿수 작다 — 표 3 의 seqsub 끝 eps-dom% 가 ≥ 90% 면 이 스모크가 시험한 것은 '
             '"Δ 선택성이 사실상 초기값에 고정된 Mamba(B·C 의 입력 의존성은 학습)" 이며, 일반적인 selective SSM vs RNN 으로 해석하지 않는다. '
             '(4) 정밀도 열: 기준 행은 모두 fp16(gru:DL = cuDNN GRU fp16). Mamba-ETER fp32 행(fallback)은 정밀도가 기준과 다르다.\n\n',
             f'비결정성: {NONDET_NOTE}\n\n',
             'Mamba-ETER 는 f(0)=0 이라 probe 의 seq_in_dep(‖f(x)−f(0)‖/‖f(0)‖)가 정의되지 않는다 — 시퀀스 출력 RMS(초기→끝)를 본다. '
             'U-Net 초기값: --unet-init gru 이면 gru:DL 과 비트 단위로 같다(학습 전·후 대조; ss2d:DL·unet:DL 과는 기준끼리도 다름).\n\n',
             '## 표 1. 기능·검증 (고정 val 슬라이스, 원 단위) + 시퀀스 모듈 학습 상태 (끝 시점, 하네스 그룹 seq)\n\n',
             '| 모델(조건) | 정밀도 | status | steps | val SSIM | PSNR | nMSE | shuffle_cross dSSIM | shuffle_adj dSSIM | zero dSSIM '
             '| deep4 dSSIM | seq 사용 | 판정 seq | relΔ seq | eps-dom% seq | upd/lr seq | 해석 | seq out RMS 초기→끝 | amp skip '
             '| s/step | peak GPU GB |\n',
             '|' + '---|' * 21 + '\n']
    for seq, label, prec in (('gru', 'ETER-net (gru:DL, 기준)', 'fp16 (cuDNN GRU)'), ('ss2d', 'SS2D (ss2d:DL)', 'fp16 autocast'),
                             ('unet', 'U-Net 단독 (unet:DL)', 'fp16 autocast')):
        p = os.path.join(REF_DIR, f'{seq}_DL_summary.json')
        try:
            with open(p) as f:
                lines.append(_row(label, json.load(f), prec))
        except Exception as ex:                                         # noqa: BLE001
            lines.append(f'| {label} | {prec} | (읽기 실패: {ex}) |' + ' — |' * 18 + '\n')
    finals = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith('mamba1d') and fn.endswith('_final.json'):
            try:
                with open(os.path.join(out_dir, fn)) as f:
                    finals.append(json.load(f))
            except Exception:                                           # noqa: BLE001
                pass
    for r in finals:
        m = r.get('meta') or {}
        s = r.get('run_job_summary')
        note = ''
        if r.get('precision_alarm') and r['precision_alarm'].get('triggered'):
            note = ' (fallback 규칙 발동)'
        if r.get('error'):
            note += ' (오류)'
        # seq out RMS = 하네스 probe(같은 probe 2 슬라이스·같은 autocast)의 init_probe/final_probe — 기준 행과 같은 정의
        lines.append(_row(f'**{MODEL_NAME}** ({m.get("vname")})', s, precision_label(m.get('fp32_stages')), note=note))
    if finals:
        lines.append('\n## 표 2. 학습 뒤 분석 (D.post_analysis 와 같은 정의, val 앞 post_slices 슬라이스; 끝 / 초기)\n\n')
        lines.append('| 조건 | 시점 | seq out RMS | 슬라이스 고유 공간 RMS | 슬라이스 고유 공간 비율 | zero | slice_mean_plane '
                     '| no_slice_specific_spatial | shuffle_cross | 첫 conv 응답 seq/img | R² zf RSS / log abs k |\n')
        lines.append('|' + '---|' * 11 + '\n')
        for r in finals:
            post = r.get('post') or {}
            for when in ('end', 'init'):
                e = post.get(when) or {}
                de = e.get('decomposition') or {}
                st = e.get('substitution_tests') or {}
                g = e.get('grid') or {}
                fc = e.get('first_conv_response') or {}
                lines.append(f'| {r["meta"].get("vname")} | {when} | {_f(de.get("seq_out_rms"), ".3g")} '
                             f'| {_f(de.get("slice_specific_spatial_rms"), ".3g")} | {_f(de.get("frac_slice_specific_spatial"), ".3f")} '
                             f'| {_ds(st.get("zero"))} | {_ds(st.get("slice_mean_plane"))} | {_ds(st.get("no_slice_specific_spatial"))} '
                             f'| {_ds(st.get("shuffle_cross"))} | {_f(fc.get("seq_over_img"), ".3g")} '
                             f'| {_f(g.get("r2_zf_rss"), ".3f")} / {_f(g.get("r2_logk_sampled"), ".3f")} |\n')
        lines.append('\n## 표 3. 하위 텐서별 학습 상태 (끝 시점, 하네스 per-parameter 통계를 정·역방향 합산 — seqsub.<단계>.<텐서>)\n\n'
                     'eps-dom% = √v̂ < eps(1e-8) 원소 비율, upd/lr = 평균 |m̂|/(√v̂+eps) (잡음 ≈0.2, 일관된 신호 → 1), relΔ = ‖w−w0‖/‖w0‖. '
                     'Δ 경로 = dt_proj.weight·dt_proj.bias·x_proj(Δ 행 포함)·A_log.\n\n')
        lines.append('| 조건 | 단계 | ' + ' | '.join(MAMBA_PARAM_SUBS) + ' |\n')
        lines.append('|' + '---|' * (2 + len(MAMBA_PARAM_SUBS)) + '\n')
        for r in finals:
            fg = ((r.get('run_job_summary') or {}).get('final_groups') or {})
            for st in ('gru_h', 'gru_v'):
                cells = []
                for sub in MAMBA_PARAM_SUBS:
                    g = fg.get(f'seqsub.{st}.{sub}') or {}
                    cells.append(f'{_pctf(g.get("eps_dom_frac"))} · {_f(g.get("upd_ratio_mean"), ".2g")} · {_f(g.get("rel_change"), ".2g")}')
                lines.append(f'| {r["meta"].get("vname")} | {st} | ' + ' | '.join(cells) + ' |\n')
        lines.append('\n(각 칸 = eps-dom% · upd/lr · relΔ)\n')
        lines.append('\n## 표 4. 학습 전 기울기 규모 (grad_preflight — 실제 크기 probe 1 슬라이스, DL 손실, fp32; 같은 시드 ETER-net 과 나란히)\n\n'
                     '기울기<eps = 한 step 기울기 원소가 Adam eps 1e-8 미만인 비율(대리 지표 — 미표본 k-space 열 때문에 in_proj/W_ih 원소의 상당수는 '
                     '두 모델 모두 구조적으로 0). fp16 대조 = 학습과 같은 autocast 역전파(손실 ×4096)의 fp32 대비 상대오차·0 으로 떨어진 비율.\n\n')
        lines.append('| 조건 | seq 기울기 RMS Mamba / GRU | seq 기울기<eps Mamba / GRU | in_proj ÷ W_ih RMS h · v | Δ 경로 기울기 RMS h (dt_proj.w · x_proj · A_log) '
                     '| fp16 rel_err in_proj h·v / out_proj h·v | fp16 0화 in_proj h·v | fp16 언더플로 판정 | U-Net 기울기 RMS Mamba / GRU |\n')
        lines.append('|' + '---|' * 9 + '\n')
        for r in finals:
            gp = (r.get('meta') or {}).get('grad_preflight') or {}
            mm, gg = gp.get('mamba') or {}, gp.get('gru') or {}
            ms, gs = mm.get('fp32_seq_total') or {}, gg.get('fp32_seq_total') or {}
            f32 = mm.get('fp32') or {}
            c16 = mm.get('fp16_vs_fp32') or {}
            rr = gp.get('in_proj_rms_over_gru_W_ih') or {}
            uf = gp.get('fp16_underflow') or {}

            def gr(k):
                return _f((f32.get(k) or {}).get('rms'), '.1e')

            def cr(k, fld='rel_err'):
                return _f((c16.get(k) or {}).get(fld), '.2g')
            lines.append(f'| {r["meta"].get("vname")} | {_f(ms.get("rms"), ".2e")} / {_f(gs.get("rms"), ".2e")} '
                         f'| {_pctf(ms.get("frac_lt_eps"))} / {_pctf(gs.get("frac_lt_eps"))} '
                         f'| {_f(rr.get("gru_h"), ".3g")} · {_f(rr.get("gru_v"), ".3g")} '
                         f'| {gr("gru_h.dt_proj.weight")} · {gr("gru_h.x_proj.weight")} · {gr("gru_h.A_log")} '
                         f'| {cr("gru_h.in_proj.weight")}·{cr("gru_v.in_proj.weight")} / {cr("gru_h.out_proj.weight")}·{cr("gru_v.out_proj.weight")} '
                         f'| {cr("gru_h.in_proj.weight", "frac_zero_in_fp16_of_fp32_nonzero")}·'
                         f'{cr("gru_v.in_proj.weight", "frac_zero_in_fp16_of_fp32_nonzero")} '
                         f'| {uf.get("triggered") if uf.get("evaluated") else "— (cpu)"} '
                         f'| {_f(mm.get("unet_grad_rms_fp32"), ".2e")} / {_f(gg.get("unet_grad_rms_fp32"), ".2e")} |\n')
        lines.append('\n## 표 5. Mamba 전용 (끝 / 초기)\n\n')
        lines.append('| 조건 | U-Net 초기값 = gru:DL (전 / 후) | fallback 규칙 | out_proj norm 비 h fwd/bwd · v fwd/bwd | in_proj norm 비 h · v (fwd) '
                     '| Δ 경로 relΔ h fwd: x_proj Δ행 · dt_proj.w · A_log | softplus(dt_bias) 평균 h·v (끝; 초기) | h 출력 RMS 끝 (초기) '
                     '| seq 출력 RMS 끝 (초기) | v 정방향 에너지 비 끝 (초기) | outmatch k_h / k_v |\n')
        lines.append('|' + '---|' * 11 + '\n')
        for r in finals:
            post = r.get('post') or {}
            wr = post.get('weight_norm_ratios') or {}
            sm = post.get('ssm_params') or {}
            sp = post.get('stage_probe') or {}
            se, si = sp.get('end') or {}, sp.get('init') or {}
            om = (r.get('meta') or {}).get('outmatch') or {}
            uc = r.get('unet_init_check') or {}
            upc = (r.get('meta') or {}).get('unet_init_precheck') or {}
            al = r.get('precision_alarm') or {}

            def wrr(k, fld='ratio'):
                return _f((wr.get(k) or {}).get(fld), '.3g')

            def dtm(when, st):
                return _f(((sm.get(when) or {}).get(f'{st}.fwd') or {}).get('dt_mean'), '.3g')
            lines.append(f'| {r["meta"].get("vname")} | {upc.get("match", "—")} / {uc.get("match", "—")} '
                         f'| {"; ".join(al.get("reasons") or []) or "없음"} '
                         f'| {wrr("gru_h.fwd.out_proj.weight")}/{wrr("gru_h.bwd.out_proj.weight")} · '
                         f'{wrr("gru_v.fwd.out_proj.weight")}/{wrr("gru_v.bwd.out_proj.weight")} '
                         f'| {wrr("gru_h.fwd.in_proj.weight")} · {wrr("gru_v.fwd.in_proj.weight")} '
                         f'| {wrr("gru_h.fwd.x_proj.weight[dt_rows]", "rel_change")} · {wrr("gru_h.fwd.dt_proj.weight", "rel_change")} · '
                         f'{wrr("gru_h.fwd.A_log", "rel_change")} '
                         f'| {dtm("end", "gru_h")}·{dtm("end", "gru_v")} ({dtm("init", "gru_h")}·{dtm("init", "gru_v")}) '
                         f'| {_f(se.get("h_out_rms"), ".3g")} ({_f(si.get("h_out_rms"), ".3g")}) '
                         f'| {_f(se.get("seq_out_rms"), ".3g")} ({_f(si.get("seq_out_rms"), ".3g")}) '
                         f'| {_f(se.get("v_fwd_energy_frac"), ".3f")} ({_f(si.get("v_fwd_energy_frac"), ".3f")}) '
                         f'| {_f(om.get("k_h"), ".4g")} / {_f(om.get("k_v"), ".4g")} |\n')
    p = os.path.join(out_dir, 'summary.md')
    with open(p + '.tmp', 'w') as f:
        f.write(''.join(lines))
    os.replace(p + '.tmp', p)


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Mamba-ETER 스모크 (smoke_recipe_fix DL 레시피, 4000 step)',
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--init', default='default,outmatch',
                   help='쉼표 목록 ⊆ {default, outmatch}, 적은 순서대로 실행 (기본 둘 다 — outmatch 는 필수 짝 조건)')
    p.add_argument('--unet-init', choices=('gru', 'unet', 'own'), default='gru',
                   help='gru = 스모크 gru:DL 과 같은 U-Net 초기값(기본), unet = v8fix 관례, own = 자체 생성 순서')
    p.add_argument('--fp32', default=os.environ.get('MAMBA_ETER_FP32', 'none'), help='none|h|v|hv (env MAMBA_ETER_FP32 와 같은 해석)')
    p.add_argument('--no-fp32-fallback', action='store_true', help='fallback 규칙 발동 시 --fp32 hv 자동 재실행을 끔')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--log-dense-every', type=int, default=20)
    p.add_argument('--log-dense-until', type=int, default=500)
    p.add_argument('--val-slices', type=int, default=64)
    p.add_argument('--post-slices', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--out-dir', default='results/mamba1d_smoke_2026-10-11', help='상대 경로는 저장소 루트 기준')
    p.add_argument('--max-minutes', type=float, default=0,
                   help='학습 시간 상한(분, 0 = 없음 — 기준 gru:DL 과 같음). 상한으로 중단된 조건은 비교 불가로 표기·skip-done 거부')
    p.add_argument('--num-workers', type=int, default=None, help=f'기본: cuda=config {C.NUM_WORKERS_TRAIN}, cpu=2 '
                                                                 f'(cuda 는 기준과 같은 {REF_NUM_WORKERS} 만 허용)')
    p.add_argument('--val-workers', type=int, default=None, help='기본: cuda=config 4, cpu=1')
    p.add_argument('--force', action='store_true', help='GPU0 사용 중·CUDA_VISIBLE_DEVICES≠0 거부를 무시')
    p.add_argument('--skip-done', action='store_true', help='같은 설정으로 끝난 조건은 건너뜀')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    a = p.parse_args(argv)
    a.inits = [x.strip() for x in a.init.split(',') if x.strip()]
    if not a.inits or any(x not in INITS for x in a.inits) or len(set(a.inits)) != len(a.inits):
        p.error(f'--init 은 {INITS} 의 중복 없는 쉼표 목록: {a.init!r}')
    a.init = a.inits[0]
    a.fp32_stages = fp32_stages_from_str(a.fp32)
    if a.num_workers is None:
        a.num_workers = C.NUM_WORKERS_TRAIN if a.device == 'cuda' else 2
    if a.val_workers is None:
        a.val_workers = C.NUM_WORKERS_VAL if a.device == 'cuda' else 1
    if a.steps < 1 or a.log_every < 1 or a.bs < 1 or a.val_slices < 1 or a.log_dense_every < 0 or a.max_minutes < 0:
        p.error('--steps/--log-every/--bs/--val-slices 는 1 이상, --log-dense-every·--max-minutes 는 0 이상')
    if not os.path.isabs(a.out_dir):
        a.out_dir = os.path.join(_PROJECT_ROOT, a.out_dir)
    a.post_slices = max(1, min(a.post_slices, a.val_slices))
    a.k_h = a.k_v = None
    return a


def done_ok(args):
    """같은 설정으로 끝까지(요청 step 전부) 정상 완료한 조건만 True. 시간 상한 중단·U-Net 초기값 불일치·post 실패·거부는 다시 실행."""
    fp = os.path.join(args.out_dir, f'{vname_of(args)}_final.json')
    if not os.path.exists(fp):
        return False, None
    try:
        with open(fp) as f:
            old = json.load(f)
    except Exception:                                                   # noqa: BLE001
        return False, None
    m = old.get('meta') or {}
    s = old.get('run_job_summary') or {}
    same = (m.get('runner_version') == RUNNER_VERSION and m.get('steps') == args.steps and m.get('seed') == args.seed
            and m.get('bs') == args.bs and m.get('init') == args.init and m.get('unet_init') == args.unet_init
            and m.get('fp32_stages') == list(args.fp32_stages) and m.get('val_slices_arg') == args.val_slices
            and m.get('post_slices') == args.post_slices and m.get('num_workers') == args.num_workers)
    ok = (same and old.get('error') is None and old.get('status') == 'completed'
          and s.get('steps_done') == args.steps and s.get('status') == 'completed')
    return ok, old


def cond_args(args, init, stages):
    a = argparse.Namespace(**vars(args))
    a.init, a.fp32_stages, a.k_h, a.k_v = init, tuple(stages), None, None
    return a


def wants_fallback(args, alarm, stages, device):
    return (bool(alarm and alarm.get('triggered')) and not args.no_fp32_fallback
            and set(stages) != set(FALLBACK_STAGES) and device.type == 'cuda')


def main(argv=None):
    args = parse_args(argv)
    check_versions()
    device = torch.device(args.device)
    print('=' * 72)
    print(f' {MODEL_NAME} 스모크 ({RUNNER_VERSION}; smoke {R.SCRIPT_VERSION}, diag {D.DIAG_VERSION}) device={device} '
          f'init={args.inits} unet_init={args.unet_init} fp32={list(args.fp32_stages)}')
    print(f'   steps={args.steps} bs={args.bs} seed={args.seed} val={args.val_slices} post={args.post_slices} '
          f'workers={args.num_workers}/{args.val_workers} max_minutes={args.max_minutes} out={args.out_dir}  {now()}  '
          f'torch {torch.__version__}')
    print('=' * 72, flush=True)
    if device.type == 'cuda' and args.num_workers != REF_NUM_WORKERS:
        R.refuse(f'--num-workers {args.num_workers} ≠ 기준 gru:DL 의 {REF_NUM_WORKERS} — 워커 수가 표본별 augment(마스크 offset·flip) '
                 f'rng 배분을 바꿔 학습 데이터가 달라진다')

    queue = [(init, tuple(args.fp32_stages)) for init in args.inits]   # 1차 조건 먼저, fallback 은 그 뒤
    fallbacks = []
    data, rc = None, 0
    while queue or fallbacks:
        init, stages = queue.pop(0) if queue else fallbacks.pop(0)
        a = cond_args(args, init, stages)
        if args.skip_done:
            ok, old = done_ok(a)
            if ok:
                print(f'[skip-done] {vname_of(a)} 이미 끝남 — 건너뜀', flush=True)
                if wants_fallback(args, old.get('precision_alarm'), stages, device) and \
                        (init, FALLBACK_STAGES) not in fallbacks:
                    print(f'   (fallback 규칙 발동 기록 {old["precision_alarm"].get("reasons")} → {init} _fp32hv 대기열)', flush=True)
                    fallbacks.append((init, FALLBACK_STAGES))
                continue
        if data is None:                                               # 실제로 돌릴 조건이 있을 때만 GPU 점검·데이터 준비
            if device.type == 'cuda':
                R.gpu0_precheck(args.force)
            else:
                if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
                    R.refuse('CPU 모드는 CUDA_VISIBLE_DEVICES="" 로만 실행한다 (GPU0 드라이버·NVML 을 건드리지 않기 위해)')
                R.install_cpu_shims(False)
                install_mamba_cpu_shims()
            os.makedirs(args.out_dir, exist_ok=True)
            train_ds, val_ds, val_indices, probe_indices, probe = D.prepare_data(args, device)
            print(f'  val 슬라이스 {len(val_indices)}개: {val_indices[:8]}{" ..." if len(val_indices) > 8 else ""} | probe {probe_indices}')
            t0 = time.time()
            dl_scale = R.estimate_mean_scale(train_ds, args.seed)
            dl_scale['source'] = f'train {dl_scale["n"]} slices'
            print(f'  DL s̄ = {dl_scale["mean"]:.4f} → λ_DL = {C.LAMBDA_SSIM_PER_PIXEL / dl_scale["mean"]:.6g} ({time.time() - t0:.0f}s)')
            data = (train_ds, val_ds, val_indices, probe_indices, probe, dl_scale)
        code, final = run_condition(a, data, device)
        rc = max(rc, code)
        al = final.get('precision_alarm') or {}
        if code == 0 and wants_fallback(args, al, stages, device) and (init, FALLBACK_STAGES) not in fallbacks:
            print(f'  [fallback] 규칙 발동 {al["reasons"]} → 같은 init({init})을 --fp32 hv 로 다시 실행 (사전 등록 규칙, 1차 조건 뒤)',
                  flush=True)
            fallbacks.append((init, FALLBACK_STAGES))
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if os.path.isdir(args.out_dir):
        rebuild_md(args.out_dir)
    print(f'\n전체 완료 (종료 코드 {rc}) → {os.path.join(args.out_dir, "summary.md")}', flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
