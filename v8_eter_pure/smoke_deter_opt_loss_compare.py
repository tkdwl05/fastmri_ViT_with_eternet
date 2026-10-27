"""손실 설정 요인 비교 (2026-10-15, smoke_deter_opt.py 설계 결정 16) — CPU 전용, torch 미사용. smoke_deter_opt_report.py 의
읽기·통계 함수(load_cond·loss_by_b·ratio_window·paired·e3_ok·T9_975)를 그대로 쓴다 (smoke_deter_opt_a2_compare.py 와 같은 방식).

질문: 손실을 교수님 형태로 바꾸면 ETER-net 대 U-Net 단독의 관계(같은 배치 학습 손실 비, GRU 포화, 시퀀스 출력 사용, 평가 SSIM)가 바뀌는가.
기준 P0 = 기존 ref·A·A2·A3: 뇌 마스크 L1 + 뇌 마스크 SSIM, λ = LAM (pod 값). 요인을 하나씩 바꾼 설정 X:
  F = 전체 영상 L1·SSIM, λ = LAM / L = 마스크, λ = 0.2 / P = 전체 영상, λ = 0.2 (교수님 ViT 손실 형태).
  조건 = refX (U-Net 단독, init seed 1) · A_kX (ETER-net, GRU 초기값 seed k = 1·2·3 → AX·A2X·A3X) · 선택 ref2X (U-Net init seed 2).

1차 판정 = S3 규칙 (docs/v8_fairness_followup_plan.md 의 S3 행 — 10-15 17:4x UTC, A3 실행 시작 전·손실 요인 조건 실행 전에 등록).
  설정 X 마다 (1단계는 X = P), 회차 k = GRU 초기값 seed 1·2·3:
    R0_k = R(A_k vs ref), RX_k = R(A_kX vs refX) — 각 조건 자기 목적함수(steps 의 loss)의 같은 배치 비, 배치 3001~4000.
    better : max(RX) < min(R0) 이고 median(R0) − median(RX) ≥ 5 %p   → X 에서 ETER-net 이 상대적으로 나음
    worse  : min(RX) > max(R0) 이고 median(RX) − median(R0) ≥ 5 %p   → 대칭
    unchanged : 두 범위 [min, max] 가 겹치고 |median(RX) − median(R0)| ≤ 5 %p
    unresolved: 그 밖.   '중앙값 차' = 중앙값끼리의 차 (회차 짝 없이 두 묶음 비교 — 등록 문구 '범위·중앙값' 그대로).
    타당한 회차가 양쪽 모두 ≥ 2 일 때만 판정 (등록 문구는 3 대 3 전제 — 3 대 3 완전 분리의 순열 p = 0.1, 2 대 2 면 1/3: 선별용).
    '변화'(better/worse) 일 때만 2단계(F·L 분리)를 제안한다.
2차 판정 = 아래 회차 짝 규칙 (구현 검토 중 10-15 ≈19:00 UTC 추가 — S3 규칙이 서로 다른 목적함수 안의 격차를 비교하고 회차 짝을
  쓰지 않는다는 비판에서 나옴). 추가 시점에 A3(P0, GRU seed 3)은 실행 중이었고 step ≈2250 까지의 궤적만 보였다:
  sat_v(2000) = 0.779, 배치 1501~2000 R ≈ +17 %. A3 의 배치 3001~4000 R 과 step 4000 sat_v 는 보지 않은 상태에서 모드 문턱을 정했다.
  1차와 2차가 다르면 둘 다 보고하고 1차를 앞에 둔다.
  0. 그 전 판(R_X 를 R_A 하나와 비교, d0 = |R_A2 − R_A| 를 잡음으로)은 폐기: d0 = 32.5 %p 는 잡음이 아니라 두 모드 사이 거리라서
     'changed' 문턱이 65 %p 가 되어 사실상 아무것도 가를 수 없었다(10-15 설계 비판).
  1. 단위 = GRU 회차 k (seed 1·2·3). 각 회차를 자기 P0 회차(A·A2·A3)와 비교한다 — 회차를 섞지 않는다.
     같은 배치 비 R_M(x, r) = Σ_b M_x(b) / Σ_b M_r(b) − 1, 배치 b = 3001~4000 (모든 조건의 배치 흐름이 같음 — E2).
     Δ_k(M) = R_M(A_kX, refX) − R_M(A_k, ref).  CI = 블록 짝 차: 100 배치 블록 10 개마다 d_j = R_M,X,j − R_M,0,j,
     Δ_k ± t_9,0.975 · sd(d)/√10 (두 비가 같은 배치 위치를 쓰므로 짝지어 뺀다 — R_0 의 자체 CI 를 무시하지 않음).
  2. 1차 지표 M (설정마다; 기준이 '지표만 바뀌어서는 움직이지 않게'):
       L : M = obj_L = l1_mask + 0.2·(1 − SSIM_mask). 기준 R_M(A_k, ref) 는 P0 회차의 기록 항(l1·ssim_loss, 마스크)을 L 목적함수로
           다시 잰 값(재학습 없음 — 10-15 점검값 A −0.33 %·A2 +25.4 %). L 회차 쪽도 같은 기록 항(aux.l1_mask·aux.ssim_loss_mask)으로 잰다.
       F·P : P0 회차에는 전체 영상 항이 없으므로(배경을 맞추지도 않음) 같은 척도인 마스크 항 비 두 개를 공동 1차로 쓴다:
           M1 = l1_mask, M2 = ssim_loss_mask (새 조건 aux.l1_mask·aux.ssim_loss_mask, P0 조건 l1·ssim_loss).
       각 조건 자신의 목적함수 비 R_X(own) 와 Δ(own) = R_X(own) − R_A_k(P0) 는 2차(기술).
  3. 회차 모드 (회차마다, 그 설정의 U-Net 대비): r = R_l1mask(A_kX, refX) (W6), s = probe sat_v(step 4000).
       bad  : r ≥ +10 % 또는 s ≥ 0.80
       good : r ≤ +3 % 그리고 s ≤ 0.60
       intermediate : 그 밖.   (P0 의 r 은 자기 목적함수 R 과 0.2 %p 이내: A −1.33 vs −1.30, A2 +31.4 vs +31.2 → A = good, A2 = bad)
     모드 판정에 마스크 L1 을 쓰는 이유: 설정과 무관한 같은 척도이고 P0 에서는 기존 R 과 사실상 같다.
  4. 회차 k 의 상태 (지표 M 마다):
       move  : (|Δ_k| ≥ 10 %p 그리고 짝 CI 가 0 을 포함하지 않음) 또는 모드가 good↔bad 로 뒤집힘.
               방향 = Δ_k 의 부호 (음수 = ETER-net 이 X 에서 상대적으로 나아짐), 모드 뒤집힘은 bad→good = 나아짐, good→bad = 나빠짐.
               두 근거의 방향이 반대면 'conflict'. (10 %p = 같은 seed 1 의 pod 대 로컬 차 ≥ 11 %p 에서 정함)
       stay  : |Δ_k| ≤ 3 %p 그리고 모드가 같음.
       neither : 그 밖.
  5. 설정 X 의 판정 (지표 M 마다, 타당한 회차만):
       타당한 회차 < 2 → 'descriptive_only' (단일 회차 결과는 기술만 — 판정하지 않음)
       'changed'   : 같은 방향으로 move 인 회차 ≥ 2 이고 반대 방향 move·conflict 회차 없음
       'unchanged' : 모든 타당한 회차가 stay
       'unresolved': 그 밖.
     F·P 는 두 공동 1차 중 하나라도 'changed' 면 'changed'(방향이 서로 다르면 'changed (mixed)'), 둘 다 'unchanged' 면 'unchanged',
     회차 < 2 면 'descriptive_only', 그 밖 'unresolved'. 두 지표를 보는 만큼 'changed' 가 쉬워진다 — 표에 지표별 판정을 같이 낸다.
  6. 타당성 (조건마다): status completed, I1 통과(U-Net 초기값 = 공통 값; ref2 계열은 unet_ck_ref2 대조), E2 vs_ref == 'match'
     (ref 자신 제외), E3 flag 없음(nan_skip ≤ 10, amp_floor ≤ 5), GRU 조건은 I6 통과 + 초기값(A 계열 I2, A2·A3 계열 seed 1 과 다름),
     손실 요인 조건은 preset = 기준 조건 preset + (lam_ssim, loss_region) = SETTINGS[X] 이고 W6 의 모든 배치에 aux 기록,
     GRU 조건은 step 4000 sat_v 기록. 회차 k 는 A_kX·refX·A_k·ref 넷 다 타당할 때만 판정에 들어간다.
기술 (판정에 쓰지 않음):
  - P 의 균형 표시(사전 고정, λ 재조정 없음): refX 의 배치 3001~4000 SSIM 몫 = Σ λ·ssim_loss / Σ (l1 + λ·ssim_loss) (자기 영역 항).
    45~65 % 밖이면 'P 는 교수님보다 SSIM 비중이 큼/작음' 으로 표시 (교수님 로그 52~58 %, L 설정의 예상 ≈51 %).
  - 같은 배치 성분 비·자기 목적함수 비(6 구간), GT 지지 영역 항(aux.l1_supp·ssim_loss_supp — 0 채움 테두리 제외)의 비,
    ref2X/refX(그 설정 아래 U-Net 회차 간 잡음; P0 의 ref2/ref = +0.29 % [−1.44, +2.03]).
  - GRU 포화 sat_v 궤적(probe), E256 시퀀스 출력 치환 시험(OURS, eval_summary.json tests; dualeval --tests 가 있으면 PROF_TEST 판도).
  - E256 step 4000 SSIM: OURS(eval_per_slice.csv)·PROF_TEST(smoke_deter_opt_dualeval.py 출력, config_hash·end_state 대조 통과분만)
    ·PROF_TEST_crop(같은 호출을 GT ≠ 0 외접 사각형에 — 0 채움 테두리를 뺀, 교수님 '전체 영상'에 가장 가까운 판; 읽는 규칙은 PROF_TEST 와 같음).
    잡음 주: 규약별 |ref2 − ref| (OURS +0.0081; PROF_TEST 는 ref2 의 dualeval — 10-15 CPU 점검값 −0.031)보다 작은 단일 시드 SSIM 차는
    해석하지 않는다. PROF_TEST 는 마스크로 학습한 모델(P0·L)에서 학습하지 않은 배경을 주로 잰다(zero-filled 가 ref 보다 높은 슬라이스
    75 %) — PROF_TEST Δ(AX − refX) 는 F·P 에서만, 그 설정의 잡음 바닥(ref2X − refX, PROF_TEST)이 있고 그보다 클 때만 읽는다.
    P0 의 A − ref 부호 반전(OURS +0.0022, PROF_TEST −0.100)을 ETER-net 의 성질로 보고하지 않는다.
해석 한계 (출력에 그대로 넣음): 주장은 '이 설계(BS 8·4000 step·U-Net 초기값 하나·같은 배치 흐름)의 선별 시험' 범위로 한정한다.
  'changed' 는 학습 손실 관계가 바뀌었다는 뜻이지 평가 성능의 우열이 아니다. 어떤 결과도 '시퀀스 모듈이 불필요하다/해롭다'로 쓰지 않는다.
출력: <out>/loss_compare.md · loss_compare.json  (--relax-preset: preset·aux 대조를 막지 않는 점검용 — 표 머리에 DRY RUN 표시)
사용: CUDA_VISIBLE_DEVICES="" python v8_eter_pure/smoke_deter_opt_loss_compare.py --out-dir results/smoke_deter_opt_2026-10-14
      [--dualeval <dir>]   (기본 <out>/dualeval)        규칙 단위 시험: ... --selftest
"""

import os
import sys
import csv
import json
import math
import argparse
import datetime
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_deter_opt_report as RP          # noqa: E402

# ── 사전 고정 상수 (위 머리말과 같아야 함) ──
MOVE, STAY = 0.10, 0.03                      # 회차 상태: |Δ_k| 문턱 (move 는 짝 CI 가 0 제외도 필요)
MODE_BAD_R, MODE_BAD_SAT = 0.10, 0.80        # bad : r ≥ +10 % 또는 sat_v(4000) ≥ 0.80
MODE_GOOD_R, MODE_GOOD_SAT = 0.03, 0.60      # good: r ≤ +3 % 그리고 sat_v(4000) ≤ 0.60
N_SEED_MIN = 2                               # 판정에 필요한 타당한 회차 수
S3_MED = 0.05                                # 1차(S3) 규칙: 중앙값 차 문턱 5 %p
S3_N_MIN = 2                                 # 1차(S3) 규칙: 양쪽 타당한 회차 최소 수 (등록 전제는 3)
SHARE_LO, SHARE_HI = 0.45, 0.65              # P 의 SSIM 몫 표시 범위 (교수님 로그 52~58 %)
SAT_STEP = 4000
LAM = 0.0030745725                           # smoke_deter_opt.LAM (pod2 λ)
LAM_L = 0.2                                  # smoke_deter_opt.LAM_PROF
SETTINGS = OrderedDict([('F', (LAM, 'full')), ('L', (LAM_L, 'mask')), ('P', (LAM_L, 'full'))])   # = smoke_deter_opt.LOSS_SETTINGS
SEEDS = OrderedDict([('A', 'A'), ('A2', 'A2'), ('A3', 'A3')])   # ETER-net 회차 (접두 → P0 조건)
PRIMARY = OrderedDict([('F', ('l1m', 'sm')), ('L', ('objL',)), ('P', ('l1m', 'sm'))])
METRIC_DESC = OrderedDict(l1m='masked L1 (aux.l1_mask | P0 l1)', sm='masked 1-SSIM (aux.ssim_loss_mask | P0 ssim_loss)',
                          objL='L objective l1_mask + 0.2*ssim_loss_mask (P0 runs rescored, no retraining)',
                          own='own objective (steps loss)', l1s='GT-support L1 (aux.l1_supp)', ss='GT-support 1-SSIM (aux.ssim_loss_supp)',
                          l1f='full-image L1 (aux.l1_full | F/P own l1)', sf='full-image 1-SSIM (aux.ssim_loss_full)')
W = RP.W6
WIN = f'{W[0]}-{W[1]}'
PROTOS = ('OURS', 'PROF_TEST', 'PROF_TEST_crop')       # crop = GT ≠ 0 외접 사각형 (0 채움 테두리 제외, 기술)
PROTO_COL = dict(PROF_TEST='ssim_prof', PROF_TEST_crop='ssim_prof_crop')
NOISE_REF2 = 0.0081                          # ref2 − ref (OURS, step 4000 E256) — a2_compare·report 와 같은 값
GUARD = ('주장은 이 설계(BS 8, 4000 step, U-Net 초기값 하나, 같은 배치 흐름)의 선별 시험 범위로 한정한다. '
         "'changed' 는 학습 손실 관계가 바뀌었다는 뜻이지 평가 성능의 우열이 아니며, 어떤 결과도 '시퀀스 모듈이 불필요하다/해롭다'로 "
         '쓰지 않는다. 단일 GRU 회차 결과는 기술일 뿐이다. PROF_TEST 는 F·P 에서만, 그 설정의 잡음 바닥(ref2X − refX)보다 클 때만 읽는다 — '
         '마스크로 학습한 모델의 PROF_TEST 차(예: P0 의 A − ref)는 학습하지 않은 배경의 차라서 ETER-net 의 성질로 보고하지 않는다.')


def pct(v, fmt='+.2f'):
    return '–' if v is None else f'{v * 100:{fmt}} %'


def ci_txt(ci):
    return f' [{pct(ci[0])}, {pct(ci[1])}]' if ci else ''


# ------------------------------------------------------------------ 계열 추출
def preset_of(d):
    """실행된 preset: summary.json → (없으면, 실행 중·중단) meta.json."""
    return ((d.get('summary') or {}).get('preset')) or ((d.get('meta') or {}).get('preset')) or {}


def is_lossfactor(d):
    """손실 요인 조건인가: preset 에 lam_ssim 이 있거나 steps 에 aux 기록이 있음 (summary 가 없어도 맞게)."""
    return 'lam_ssim' in preset_of(d) or any('aux' in s for s in d.get('steps', [])[:5])


def _aux(s, k):
    return (s.get('aux') or {}).get(k)


def metric_fn(d, M, relax=False):
    """배치별 값 함수 s → float. 손실 요인 조건은 aux, P0 조건은 기록 항 (relax 면 aux 가 없을 때 기록 항으로 대체 — 점검용)."""
    lf = is_lossfactor(d) and any('aux' in s for s in d.get('steps', [])[:5])
    reg = preset_of(d).get('loss_region', 'mask')
    if M == 'own':
        return lambda s: s.get('loss')
    if M in ('l1m', 'sm', 'objL'):
        if lf:
            g1, g2 = (lambda s: _aux(s, 'l1_mask')), (lambda s: _aux(s, 'ssim_loss_mask'))
        elif reg == 'mask' or relax:
            g1, g2 = (lambda s: s.get('l1')), (lambda s: s.get('ssim_loss'))
        else:
            return None
        if M == 'l1m':
            return g1
        if M == 'sm':
            return g2
        return lambda s: (None if g1(s) is None or g2(s) is None else g1(s) + LAM_L * g2(s))
    key = dict(l1s='l1_supp', ss='ssim_loss_supp', l1f='l1_full', sf='ssim_loss_full')[M]
    if lf:
        return lambda s: _aux(s, key)
    if M in ('l1f', 'sf') and reg == 'full':
        return (lambda s: s.get('l1')) if M == 'l1f' else (lambda s: s.get('ssim_loss'))
    return None


def series(d, M, relax=False):
    f = metric_fn(d, M, relax)
    if f is None:
        return None
    return RP.loss_by_b([dict(bpos=s['bpos'], skipped=s.get('skipped'), loss=(None if s.get('skipped') else f(s)))
                         for s in d['steps']])


def win_ratio(D, x, r, M, lo=W[0], hi=W[1], relax=False):
    if x not in D or r not in D:
        return None
    sx, sr = series(D[x], M, relax), series(D[r], M, relax)
    if not sx or not sr:
        return None
    return RP.ratio_window(sx, sr, lo, hi, blocks=(lo, hi) in (RP.W5, RP.W6))


def paired_delta(wx, w0):
    """Δ = R_X − R_0 와 블록 짝 차 t9 95 % CI (머리말 1)."""
    if not wx or not w0:
        return None
    bx, b0 = wx.get('blocks_r'), w0.get('blocks_r')
    out = OrderedDict(delta=wx['R'] - w0['R'], R_X=wx['R'], R_0=w0['R'])
    if bx and b0 and len(bx) == len(b0) >= 2:
        d = np.asarray(bx, float) - np.asarray(b0, float)
        se = float(d.std(ddof=1) / math.sqrt(d.size))
        out.update(block_diffs=d.tolist(), se=se, ci95_t9=[out['delta'] - RP.T9_975 * se, out['delta'] + RP.T9_975 * se])
    else:
        out.update(se=None, ci95_t9=None)
    return out


def sat_at(d, step=SAT_STEP):
    for p in d.get('probes', []):
        if int(p.get('step', -1)) == step and p.get('sat_v') is not None:
            return float(p['sat_v'])
    return None


def sat_traj(d):
    return OrderedDict((int(p['step']), float(p['sat_v'])) for p in d['probes'] if p.get('sat_v') is not None)


# ------------------------------------------------------------------ 규칙 (순수 함수 — selftest 대상)
def mode_of(r, s):
    """머리말 3. r = 마스크 L1 같은 배치 비(W6), s = sat_v(4000)."""
    if r is None or s is None:
        return None
    if r >= MODE_BAD_R or s >= MODE_BAD_SAT:
        return 'bad'
    if r <= MODE_GOOD_R and s <= MODE_GOOD_SAT:
        return 'good'
    return 'intermediate'


def seed_status(pdlt, mode0, modeX):
    """머리말 4. pdlt = paired_delta 결과. 반환 (status, direction): direction −1 = ETER-net 상대적으로 나아짐, +1 = 나빠짐."""
    if pdlt is None or mode0 is None or modeX is None:
        return 'missing', None
    dlt, ci = pdlt['delta'], pdlt.get('ci95_t9')
    mv = None
    if abs(dlt) >= MOVE and ci and not (ci[0] <= 0.0 <= ci[1]):
        mv = -1 if dlt < 0 else 1
    fl = None
    if {mode0, modeX} == {'good', 'bad'}:
        fl = -1 if mode0 == 'bad' else 1
    if mv is not None and fl is not None and mv != fl:
        return 'conflict', None
    if mv is not None or fl is not None:
        return 'move', (mv if mv is not None else fl)
    if abs(dlt) <= STAY and mode0 == modeX:
        return 'stay', None
    return 'neither', None


def metric_verdict(statuses):
    """머리말 5 (지표 하나). statuses = 타당한 회차의 (status, direction) 목록."""
    n = len(statuses)
    if n < N_SEED_MIN:
        return OrderedDict(verdict='descriptive_only', n_seeds=n, direction=None)
    pos = sum(1 for s, d in statuses if s == 'move' and d == 1)
    neg = sum(1 for s, d in statuses if s == 'move' and d == -1)
    conf = sum(1 for s, _ in statuses if s == 'conflict')
    if max(pos, neg) >= 2 and min(pos, neg) == 0 and conf == 0:
        return OrderedDict(verdict='changed', n_seeds=n,
                           direction='ETER-net relatively better under X' if neg else 'ETER-net relatively worse under X',
                           n_move=max(pos, neg))
    if all(s == 'stay' for s, _ in statuses):
        return OrderedDict(verdict='unchanged', n_seeds=n, direction=None)
    return OrderedDict(verdict='unresolved', n_seeds=n, direction=None, n_move_better=neg, n_move_worse=pos, n_conflict=conf)


def s3_verdict(r0, rx):
    """1차(S3) 규칙. r0 = 타당한 P0 회차의 R0_k 목록, rx = 타당한 X 회차의 RX_k 목록 (자기 목적함수, W6)."""
    out = OrderedDict(n0=len(r0), nX=len(rx), R0=list(r0), RX=list(rx))
    if len(r0) < S3_N_MIN or len(rx) < S3_N_MIN:
        out.update(verdict='descriptive_only', text=f'타당한 회차 부족 (P0 {len(r0)}, X {len(rx)}; 최소 {S3_N_MIN})')
        return out
    m0, mx = float(np.median(r0)), float(np.median(rx))
    dm = mx - m0
    overlap = max(rx) >= min(r0) and max(r0) >= min(rx)
    out.update(median_R0=m0, median_RX=mx, median_diff=dm, overlap=overlap,
               range_R0=[min(r0), max(r0)], range_RX=[min(rx), max(rx)])
    if max(rx) < min(r0) and -dm >= S3_MED:
        out.update(verdict='better', text='변화 — X 에서 ETER-net 이 U-Net 단독 대비 상대적으로 나음 (범위 완전 분리 + 중앙값 차 ≥ 5 %p)')
    elif min(rx) > max(r0) and dm >= S3_MED:
        out.update(verdict='worse', text='변화 — X 에서 ETER-net 이 U-Net 단독 대비 상대적으로 나쁨 (범위 완전 분리 + 중앙값 차 ≥ 5 %p)')
    elif overlap and abs(dm) <= S3_MED:
        out.update(verdict='unchanged', text='변화 없음 — 범위 겹침 + 중앙값 차 ≤ 5 %p')
    else:
        out.update(verdict='unresolved', text='보류')
    return out


def setting_verdict(per_metric):
    """머리말 5 (설정). per_metric = {M: metric_verdict 결과} — L 은 1 개, F·P 는 2 개(공동 1차)."""
    vs = [v['verdict'] for v in per_metric.values()]
    if any(v == 'descriptive_only' for v in vs):
        return 'descriptive_only'
    ch = [v for v in per_metric.values() if v['verdict'] == 'changed']
    if ch:
        dirs = {v['direction'] for v in ch}
        return 'changed' if len(dirs) == 1 else 'changed (mixed)'
    if all(v == 'unchanged' for v in vs):
        return 'unchanged'
    return 'unresolved'


# ------------------------------------------------------------------ 타당성
def validity(D, c, base, X, relax):
    """머리말 6. base = 기준 조건 이름(ref·ref2·A·A2·A3), X = None 이면 P0 조건 자체."""
    d = D.get(c)
    if d is None or not d['summary']:
        st = 'running/interrupted (meta only)' if (d and d.get('meta')) else '없음'
        return dict(ok=False, why=[st], notes=[], present=d is not None)
    s = d['summary']
    why, notes = [], []
    if not d['completed']:
        why.append(f"status {s.get('status')}")
    inv, end = s.get('invariants') or {}, s.get('end_checks') or {}
    if not (inv.get('I1') or {}).get('ok'):
        why.append('I1')
    if s.get('cond') != 'ref' and (end.get('E2') or {}).get('vs_ref') != 'match':      # ref 자신은 흐름의 기준
        why.append(f"E2 {(end.get('E2') or {}).get('vs_ref')}")
    if not RP.e3_ok(s):
        cnt = s.get('counters') or {}
        why.append(f"E3 nan_skip={cnt.get('nan_skip')} amp_floor={cnt.get('amp_floor')}")
    if base.startswith('A'):
        if base == 'A' and not (inv.get('I2') or {}).get('ok'):
            why.append('I2')
        if base != 'A' and not (inv.get('I2') or {}).get('differs_from_seed1'):
            why.append('GRU 초기값이 seed 1 과 같음')
        if not (inv.get('I6') or {}).get('ok'):
            why.append('I6')
        if sat_at(d) is None:
            why.append(f'sat_v(step {SAT_STEP}) 없음')
    if base == 'ref2' and not str((inv.get('I1') or {}).get('skipped', '')).startswith(('ref2', c)):
        notes.append('I1 이 ref2 형식(건너뜀 + unet_ck_ref2 대조)이 아님')
    if X is not None:
        P, Pb = s.get('preset') or {}, ((D.get(base) or {}).get('summary') or {}).get('preset')
        exp = dict(lam_ssim=SETTINGS[X][0], loss_region=SETTINGS[X][1])
        got = {k: P.get(k) for k in ('lam_ssim', 'loss_region')}
        rest = {k: v for k, v in P.items() if k not in ('lam_ssim', 'loss_region')}
        bad = []
        if got != exp:
            bad.append(f'손실 키 {got} != {exp}')
        if Pb is not None and rest != Pb:
            bad.append(f'나머지 preset != {base}')
        if s.get('cond') != c:
            bad.append(f"summary cond {s.get('cond')!r} != {c!r}")
        steps_w = [st for st in d['steps'] if not st.get('skipped') and W[0] <= st['bpos'] + 1 <= W[1]]
        if not steps_w or not all('aux' in st for st in steps_w):
            bad.append('W6 배치에 aux 기록 없음')
        if bad:
            (notes if relax else why).extend(bad)
        if (end.get('E7') or {}).get('ok') is False:
            notes.append('E7 flag (aux 기록 불일치 — 판정에는 영향 없음)')
    return dict(ok=not why, why=why, notes=notes, present=True)


# ------------------------------------------------------------------ dualeval (PROF_TEST)
def read_dualeval(de_dir, c, D):
    """dualeval/{c}_per_slice.csv — {c}_summary.json 의 config_hash·end_state 가 지금 조건 폴더와 같고 E256 전체일 때만 (낡은 결과 배제)."""
    sp, cp = os.path.join(de_dir, f'{c}_summary.json'), os.path.join(de_dir, f'{c}_per_slice.csv')
    if not (os.path.exists(sp) and os.path.exists(cp)):
        return None, None
    with open(sp) as f:
        s = json.load(f)
    why = []
    cur = ((D.get(c) or {}).get('summary') or {}).get('config_hash')
    if cur is None or s.get('config_hash') != cur:
        why.append(f"config_hash {s.get('config_hash')} != 조건 {cur}")
    if s.get('subset') is not None:
        why.append(f"subset {s.get('subset')}")
    es = s.get('end_state') or {}
    ep = os.path.join((D.get(c) or {}).get('dir', ''), 'end_state.pt')
    if os.path.exists(ep):
        st = os.stat(ep)
        mt = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')
        if es.get('bytes') != st.st_size or es.get('mtime') != mt:
            why.append('end_state.pt 크기·시각이 dualeval 때와 다름')
    else:
        why.append('end_state.pt 없음')
    if why:
        return None, why
    rows = {}
    with open(cp) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k == 'file':
                    row[k] = v
                elif v in ('', None):
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows[int(float(r['idx']))] = row
    return rows, None


def ssim_vec(D, DE, c, proto, col=None):
    """슬라이스 → SSIM: OURS 는 eval_per_slice.csv(하네스), PROF_TEST 는 dualeval CSV (col 기본 ssim_prof)."""
    if proto == 'OURS':
        e = (D.get(c) or {}).get('eval')
        return None if not e else {i: r['ssim'] for i, r in e.items()}
    e = DE.get(c)
    col = col or PROTO_COL[proto]
    return None if not e else {i: r[col] for i, r in e.items() if r.get(col) is not None}


def ssim_delta(D, DE, a, b, proto, col=None):
    ea, eb = ssim_vec(D, DE, a, proto, col), ssim_vec(D, DE, b, proto, col)
    if not ea or not eb:
        return None
    ids = [i for i in ea if i in eb]
    va, vb = np.array([ea[i] for i in ids], float), np.array([eb[i] for i in ids], float)
    p = RP.paired(va, vb)
    if p is not None:
        p['mean_a'], p['mean_b'] = float(va.mean()), float(vb.mean())
    return p


def gap_change(D, DE, aX, rX, a0, r0, proto):
    """슬라이스별 (aX − rX) − (a0 − r0) (기술)."""
    v = [ssim_vec(D, DE, c, proto) for c in (aX, rX, a0, r0)]
    if any(not x for x in v):
        return None
    ids = [i for i in v[0] if all(i in x for x in v[1:])]
    gx = np.array([v[0][i] - v[1][i] for i in ids], float)
    g0 = np.array([v[2][i] - v[3][i] for i in ids], float)
    return RP.paired(gx, g0)


def readable_flags(ssim, k):
    """머리말 '기술'의 잡음 바닥: 짝 k = 'x-r' 의 SSIM 차를 읽을 수 있는가 (규약별).
    OURS 바닥 = 같은 설정의 |ref2X − refX| (없으면 P0 의 |ref2 − ref| 차용 — 표시), PROF_TEST 는 F·P 이고 ref2X − refX PROF_TEST 가
    있을 때만 (P0·L 은 마스크로 학습해 배경을 맞추지 않으므로 읽지 않음)."""
    x, r = k.split('-')
    if x.startswith('ref2'):
        return OrderedDict((p, '(잡음 바닥)') for p in PROTOS)
    X = r[3:] if len(r) > 3 else '0'
    tag = 'ref2-ref' if X == '0' else f'ref2{X}-ref{X}'
    out = OrderedDict()
    o = ssim['OURS'].get(k)
    if o is not None:
        fo = ssim['OURS'].get(tag)
        note = ''
        if fo is None:
            fo = ssim['OURS'].get('ref2-ref')
            note = '' if X == '0' else ' (P0 바닥 차용)'
        thr = abs(fo['mean']) if fo else NOISE_REF2
        if not fo and X == '0':
            note = f' (고정값 {NOISE_REF2})'
        out['OURS'] = ('yes' if abs(o['mean']) > thr else 'no') + note
    for proto in PROTOS[1:]:
        p = ssim[proto].get(k)
        if p is not None:
            fp = ssim[proto].get(tag) if X in ('F', 'P') else None
            out[proto] = '–' if not fp else ('yes' if abs(p['mean']) > abs(fp['mean']) else 'no')
    return out


def prof_tests(DE, c):
    """dualeval --tests 출력이 있으면 PROF_TEST 판 치환 시험: 정상 − 치환 (평균, 양/음)."""
    e = DE.get(c)
    if not e:
        return None
    out = OrderedDict()
    for t in ('zero', 'chanmean', 'adj', 'adjprev', 'cross'):
        for col, nm in ((f'ssim_prof_{t}', 'ssim_prof'), (f'l1full_{t}', 'l1_full')):
            ds = [r['ssim_prof'] - r[col] if nm == 'ssim_prof' else r[col] - r['l1_full']
                  for r in e.values() if r.get(col) is not None and r.get(nm) is not None]
            if ds:
                out[f'{nm}:{t}'] = dict(mean=float(np.mean(ds)), n_pos=int(sum(x > 0 for x in ds)), n_neg=int(sum(x < 0 for x in ds)),
                                        n=len(ds))
    return out or None


# ------------------------------------------------------------------ 본체
def analyse(D, DE, relax):
    V = OrderedDict()
    V['ref'] = validity(D, 'ref', 'ref', None, relax)
    if 'ref2' in D:
        V['ref2'] = validity(D, 'ref2', 'ref2', None, relax)
    base = OrderedDict()
    for k, c in SEEDS.items():
        if c not in D:
            continue
        V[c] = validity(D, c, c, None, relax)
        ok = V[c]['ok'] and V['ref']['ok']
        r_l1 = win_ratio(D, c, 'ref', 'l1m', relax=relax)
        sv = sat_at(D[c])
        base[c] = OrderedDict(valid=ok, R_own=win_ratio(D, c, 'ref', 'own'), R_l1m=r_l1, sat_v4000=sv,
                              mode=mode_of(r_l1['R'] if r_l1 else None, sv) if ok else None,
                              R=OrderedDict((M, win_ratio(D, c, 'ref', M, relax=relax)) for M in ('l1m', 'sm', 'objL')))
    res_X = OrderedDict()
    for X in SETTINGS:
        rX = 'ref' + X
        if rX in D:
            V[rX] = validity(D, rX, 'ref', X, relax)
        if 'ref2' + X in D:
            V['ref2' + X] = validity(D, 'ref2' + X, 'ref2', X, relax)
        seeds = OrderedDict()
        for k, c0 in SEEDS.items():
            aX = k + X
            if aX not in D:
                continue
            V[aX] = validity(D, aX, c0, X, relax)
            b0 = base.get(c0)
            ok = bool(V[aX]['ok'] and V.get(rX, {}).get('ok') and b0 and b0['valid'])
            row = OrderedDict(base=c0, valid=ok)
            if V[aX]['ok'] and V.get(rX, {}).get('ok'):              # 무효 조건은 성분을 계산하지 않음 (P0 열 오표기 방지)
                r_l1 = win_ratio(D, aX, rX, 'l1m', relax=relax)
                sv = sat_at(D[aX])
                row.update(R_l1m=r_l1, sat_v4000=sv, mode=mode_of(r_l1['R'] if r_l1 else None, sv),
                           mode_P0=b0['mode'] if b0 else None)
                deltas, stats = OrderedDict(), OrderedDict()
                for M in ('l1m', 'sm', 'objL', 'own'):
                    wx = win_ratio(D, aX, rX, M, relax=relax)
                    w0 = (b0['R'].get(M) if M != 'own' else b0['R_own']) if b0 else None
                    deltas[M] = paired_delta(wx, w0)
                    if M in PRIMARY[X]:
                        stats[M] = seed_status(deltas[M], row['mode_P0'], row['mode']) if ok else ('invalid', None)
                row.update(delta=deltas, status=stats,
                           windows_own=OrderedDict((f'{lo}-{hi}', win_ratio(D, aX, rX, 'own', lo, hi)) for lo, hi in RP.WINDOWS),
                           windows_l1m=OrderedDict((f'{lo}-{hi}', win_ratio(D, aX, rX, 'l1m', lo, hi, relax)) for lo, hi in RP.WINDOWS),
                           supp=OrderedDict((M, win_ratio(D, aX, rX, M)) for M in ('l1s', 'ss', 'l1f', 'sf')))
            seeds[aX] = row
        per_metric = OrderedDict()
        for M in PRIMARY[X]:
            sts = [r['status'][M] for r in seeds.values()
                   if r['valid'] and M in r.get('status', {}) and r['status'][M][0] not in ('missing', 'invalid')]
            per_metric[M] = metric_verdict(sts)
        share = None
        if rX in D and (V.get(rX, {}).get('ok') or relax):
            lam = SETTINGS[X][0]
            st = [s for s in D[rX]['steps'] if not s.get('skipped') and W[0] <= s['bpos'] + 1 <= W[1]]
            if st:
                num = sum(lam * s['ssim_loss'] for s in st)
                den = sum(s['l1'] + lam * s['ssim_loss'] for s in st)
                sh = num / den if den else None
                share = OrderedDict(ssim_share=sh, n=len(st),
                                    label=(None if sh is None else 'professor-like balance (45-65 %)' if SHARE_LO <= sh <= SHARE_HI
                                           else ('SSIM-heavier than the professor' if sh > SHARE_HI else 'SSIM-lighter than the professor')))
        noise = None
        if 'ref2' + X in D and rX in D:
            noise = OrderedDict((M, win_ratio(D, 'ref2' + X, rX, M, relax=relax)) for M in ('own', 'l1m', 'sm', 'objL'))
        # 1차(S3): 양쪽 묶음을 따로 모은다 (회차 짝 없음) — P0 쪽은 base 의 타당한 회차, X 쪽은 A_kX·refX 가 타당한 회차
        r0 = [b['R_own']['R'] for c, b in base.items() if b['valid'] and b.get('R_own')]
        rxs = []
        if seeds:
            for aX, r in seeds.items():
                if V[aX]['ok'] and V.get(rX, {}).get('ok'):
                    w = win_ratio(D, aX, rX, 'own')
                    if w:
                        rxs.append(w['R'])
        s3 = s3_verdict(r0, rxs) if seeds else None
        res_X[X] = OrderedDict(setting=dict(lam_ssim=SETTINGS[X][0], loss_region=SETTINGS[X][1]), ref=rX, primary=PRIMARY[X],
                               s3=s3, seeds=seeds, metric_verdicts=per_metric,
                               verdict=setting_verdict(per_metric), ssim_share_refX=share, unet_noise_ref2X=noise)
    return V, base, res_X


def write_md(res, D, DE, V, base, res_X, ssim, gaps, tests, ptests, sat, out, relax):
    m = []
    w_ = m.append
    w_('# 손실 설정 요인 — ETER-net 대 U-Net 단독 관계 (d-ETER 스모크, 설계 결정 16)\n')
    if relax:
        w_('> **DRY RUN** (--relax-preset): preset·aux 대조를 타당성에 넣지 않았다 — 판정 문구는 점검용이며 인용하지 않는다.\n')
    w_(f'생성: {res["time"]} · 규칙은 첫 손실 요인 조건 실행 전에 고정 (`v8_eter_pure/smoke_deter_opt_loss_compare.py` 머리말). '
       'P0 = 마스크 L1 + 마스크 SSIM, λ = LAM (기존 ref·A·A2·A3). X = F(전체 영상, LAM) · L(마스크, 0.2) · P(전체 영상, 0.2). '
       f'회차 = GRU 초기값 seed (A = 1, A2 = 2, A3 = 3) — 회차마다 자기 P0 회차와 비교, 배치 {WIN}.\n')
    w_(f'모드: bad = r ≥ {MODE_BAD_R * 100:.0f} % 또는 sat_v({SAT_STEP}) ≥ {MODE_BAD_SAT}, good = r ≤ {MODE_GOOD_R * 100:.0f} % 그리고 '
       f'sat_v ≤ {MODE_GOOD_SAT} (r = 마스크 L1 같은 배치 비). move = |Δ| ≥ {MOVE * 100:.0f} %p 이고 짝 CI 가 0 제외, 또는 good↔bad. '
       f'stay = |Δ| ≤ {STAY * 100:.0f} %p 이고 같은 모드. 판정에는 타당한 회차 ≥ {N_SEED_MIN}.\n')
    w_('## P0 기준 회차\n')
    w_('| 회차 | R own (vs ref) | R 마스크 L1 | R 마스크 1−SSIM | R L 목적함수 (재학습 없음) | sat_v(4000) | 모드 | 타당성 |')
    w_('|---|---|---|---|---|---|---|---|')
    for c, b in base.items():
        g = lambda w: (pct(w['R']) + ci_txt(w.get('ci95_t9'))) if w else '–'                      # noqa: E731
        w_(f'| {c} | {g(b["R_own"])} | {g(b["R"]["l1m"])} | {g(b["R"]["sm"])} | {g(b["R"]["objL"])} | '
           f'{"–" if b["sat_v4000"] is None else format(b["sat_v4000"], ".3f")} | {b["mode"] or "–"} | '
           f'{"통과" if V[c]["ok"] else "실패 " + str(V[c]["why"])} |')
    w_('\n## 1차 판정 — S3 규칙 (계획 문서 S3 행, 실행 전 등록)\n')
    w_('R0_k = R(A_k vs ref), RX_k = R(A_kX vs refX), 자기 목적함수 같은 배치 비, 배치 3001–4000. '
       'better/worse = 범위 완전 분리 + 중앙값 차 ≥ 5 %p, unchanged = 범위 겹침 + 중앙값 차 ≤ 5 %p, 그 밖 unresolved '
       f'(양쪽 타당한 회차 ≥ {S3_N_MIN}; 3 대 3 완전 분리의 순열 p = 0.1 — 선별용).\n')
    w_('| 설정 | R0_k (P0) | RX_k | 중앙값 R0 → RX (차) | 범위 겹침 | **1차 판정** |')
    w_('|---|---|---|---|---|---|')
    for X, rx in res_X.items():
        s3 = rx.get('s3')
        if not s3:
            continue
        r0t = ', '.join(pct(v) for v in s3['R0']) or '–'
        rxt = ', '.join(pct(v) for v in s3['RX']) or '–'
        med = (f'{pct(s3["median_R0"])} → {pct(s3["median_RX"])} ({pct(s3["median_diff"])}p)'
               if 'median_diff' in s3 else '–')
        ov = '–' if 'overlap' not in s3 else ('예' if s3['overlap'] else '아니오')
        w_(f'| {X} | {r0t} | {rxt} | {med} | {ov} | **{s3["verdict"]}** — {s3["text"]} |')
    w_('\n## 2차 판정 — 회차 짝 규칙 (10-15 ≈19:00 UTC 추가, 머리말 1~6)\n')
    w_('| 설정 | λ · 영역 | 1차 지표 | 지표별 판정 | **설정 판정** | 타당한 회차 |')
    w_('|---|---|---|---|---|---|')
    for X, rx in res_X.items():
        lam, reg = SETTINGS[X]
        mv = '; '.join(f'{M}: {v["verdict"]}{(" — " + v["direction"]) if v.get("direction") else ""}'
                       for M, v in rx['metric_verdicts'].items())
        nv = sum(1 for r in rx['seeds'].values() if r['valid'])
        w_(f'| {X} | {lam:g} · {reg} | {", ".join(rx["primary"])} | {mv} | **{rx["verdict"]}** | {nv} / {len(rx["seeds"])} |')
    w_('\n## 회차별 Δ (1차 지표, 배치 3001–4000; Δ = R(A_kX vs refX) − R(A_k vs ref), 블록 짝 차 t9 95 % CI)\n')
    w_('| 회차 | 지표 | R(A_kX vs refX) | R(A_k vs ref) | Δ [CI] | 모드 P0 → X | 상태 | 타당성 |')
    w_('|---|---|---|---|---|---|---|---|')
    for X, rx in res_X.items():
        for aX, r in rx['seeds'].items():
            vr = V.get(rx['ref']) or dict(ok=False, why=[f'{rx["ref"]} 없음'], notes=[])
            vb = V.get(r['base']) or dict(ok=False, why=[f'{r["base"]} 없음'], notes=[])
            why = list(OrderedDict.fromkeys([f'{aX}: {w}' for w in V.get(aX, {}).get('why', [])]
                                            + [f'{rx["ref"]}: {w}' for w in vr['why']]
                                            + ([f'P0 {r["base"]}: {w}' for w in vb['why']] if not vb['ok'] else [])))
            notes = list(OrderedDict.fromkeys(V.get(aX, {}).get('notes', []) + vr.get('notes', [])))
            vt = ('통과' if r['valid'] else '실패 ' + str(why)) + ((' · ' + '; '.join(notes)) if notes else '')
            if 'delta' not in r:
                w_(f'| {aX} | – | – | – | – | – | – | {vt} |')
                continue
            for M in rx['primary'] + ('own',):
                pd_ = r['delta'].get(M)
                st = r['status'].get(M, ('2차 (기술)', None)) if M != 'own' else ('2차 (기술)', None)
                w_(f'| {aX} | {M} | {pct(pd_["R_X"]) if pd_ else "–"} | {pct(pd_["R_0"]) if pd_ else "–"} | '
                   f'{(pct(pd_["delta"]) + ci_txt(pd_.get("ci95_t9"))) if pd_ else "–"} | {r.get("mode_P0") or "–"} → {r.get("mode") or "–"} | '
                   f'{st[0]}{(" (" + ("better" if st[1] < 0 else "worse") + ")") if st[1] else ""} | {vt} |')
    w_('\n## 6 구간 같은 배치 비 (vs 같은 설정의 U-Net 단독; 자기 목적함수 / 마스크 L1)\n')
    heads = ['구간'] + [f'{c} (P0)' for c in base] + [aX for X in res_X for aX in res_X[X]['seeds']]
    w_('| ' + ' | '.join(heads) + ' |')
    w_('|' + '---|' * len(heads))
    for lo, hi in RP.WINDOWS:
        k = f'{lo}-{hi}'
        cells = [k]
        for c in base:
            a, b = win_ratio(D, c, 'ref', 'own', lo, hi), win_ratio(D, c, 'ref', 'l1m', lo, hi)
            cells.append(f'{pct((a or {}).get("R"))} / {pct((b or {}).get("R"))}')
        for X in res_X:
            for aX, r in res_X[X]['seeds'].items():
                a, b = (r.get('windows_own') or {}).get(k), (r.get('windows_l1m') or {}).get(k)
                cells.append(f'{pct((a or {}).get("R"))} / {pct((b or {}).get("R"))}')
        w_('| ' + ' | '.join(cells) + ' |')
    w_('\n## 기술: 균형·지지 영역·U-Net 잡음 (배치 3001–4000)\n')
    for X, rx in res_X.items():
        sh = rx['ssim_share_refX']
        if sh:
            w_(f'- {X}: ref{X} 의 SSIM 몫 {pct(sh["ssim_share"], ".1f")} → {sh["label"]} (사전 범위 {SHARE_LO * 100:.0f}~{SHARE_HI * 100:.0f} %, '
               '교수님 로그 52~58 %; λ 재조정 없음)')
        for aX, r in rx['seeds'].items():
            sp = r.get('supp')
            if sp:
                w_(f'- {aX} vs ref{X}: ' + ', '.join(f'{M} {pct((sp.get(M) or {}).get("R"))}' for M in ('l1s', 'ss', 'l1f', 'sf'))
                   + ' (l1s·ss = GT 지지 영역 = 0 채움 테두리 제외, l1f·sf = 384² 전체)')
        nz = rx['unet_noise_ref2X']
        if nz:
            w_(f'- U-Net 잡음 ref2{X} vs ref{X}: ' + ', '.join(f'{M} {pct((v or {}).get("R"))}{ci_txt((v or {}).get("ci95_t9"))}'
                                                         for M, v in nz.items()) + ' (P0: ref2/ref own +0.29 % [−1.44, +2.03])')
    if sat:
        w_('\n## GRU 포화 sat_v (probe, 기술)\n')
        steps = sorted({s for t in sat.values() for s in t})
        w_('| 조건 | ' + ' | '.join(str(s) for s in steps) + ' |')
        w_('|---|' + '---|' * len(steps))
        for c, t in sat.items():
            w_(f'| {c} | ' + ' | '.join(f'{t[s]:.3f}' if s in t else '–' for s in steps) + ' |')
    if tests:
        w_('\n## E256 시퀀스 출력 치환 시험 (정상 − 치환 SSIM, OURS = 마스크 안; 평균 (양/음), 기술)\n')
        tn = ('zero', 'adj', 'adjprev', 'cross', 'chanmean', 'adj_sym')
        w_('| 조건 | ' + ' | '.join(tn) + ' |')
        w_('|---|' + '---|' * len(tn))
        f_ = lambda s: '–' if not s else f'{s["mean"]:+.5f} ({s["n_pos"]}/{s["n_neg"]})'   # noqa: E731
        for c, t in tests.items():
            w_(f'| {c} | ' + ' | '.join(f_(t.get(k)) for k in tn) + ' |')
    if ptests:
        w_('\n치환 시험 PROF_TEST 판 (dualeval --tests; ssim_prof = 정상 − 치환, l1_full = 치환 − 정상 (384² 전체 L1, 원 단위); '
           '평균 (양/음), 기술 — F·P 모델에서 배경 쪽 사용을 보려는 것):\n')
        keys = sorted({k for t in ptests.values() for k in t})
        w_('| 조건 | ' + ' | '.join(keys) + ' |')
        w_('|---|' + '---|' * len(keys))
        for c, t in ptests.items():
            w_(f'| {c} | ' + ' | '.join(('–' if k not in t else f'{t[k]["mean"]:+.5g} ({t[k]["n_pos"]}/{t[k]["n_neg"]})') for k in keys) + ' |')
    noise, noise_prof = (ssim['OURS'].get('ref2-ref') or {}).get('mean'), (ssim['PROF_TEST'].get('ref2-ref') or {}).get('mean')
    w_('\n## step 4000 E256 SSIM 두 규약 (기술 — 판정에 쓰지 않음)\n')
    w_(f'OURS = 뇌 마스크 안 SSIM (현행 평가) · PROF_TEST = 교수님 테스트 호출(전체 384², data_range = max(GT, 예측), 마스크 없음; '
       f'dualeval 출력 {"사용: " + ", ".join(DE) if DE else "없음"}{("; 배제(낡음): " + ", ".join(res["dualeval_stale"])) if res["dualeval_stale"] else ""}). '
       f'잡음 주: ref2 − ref = OURS {NOISE_REF2:+.4f}{" (이번 계산 " + format(noise, "+.4f") + ")" if noise is not None else ""}'
       f' · PROF_TEST {format(noise_prof, "+.4f") if noise_prof is not None else "미계산 (ref2 dualeval 없음)"}. '
       '‘읽음’ 열 = 그 규약·설정의 잡음 바닥보다 큰가 (PROF_TEST 는 F·P 이고 ref2X − refX PROF_TEST 가 있을 때만; 그 밖 “–” = 해석하지 않음).\n')
    w_('| 짝 | OURS Δ ± SE (양/음) | OURS 평균 a / b | PROF_TEST Δ ± SE (양/음) | PROF_TEST 평균 a / b | PROF_TEST_crop Δ ± SE | '
       '읽음 (OURS / PROF / crop) |')
    w_('|---|---|---|---|---|---|---|')
    fo = lambda q: '–' if not q else f'{q["mean"]:+.4f} ± {q["se"]:.4f} ({q["n_pos"]}/{q["n_neg"]})'   # noqa: E731
    fm = lambda q: '–' if not q else f'{q["mean_a"]:.4f} / {q["mean_b"]:.4f}'                          # noqa: E731
    for k in ssim['OURS']:
        o, p, pc = ssim['OURS'].get(k), ssim['PROF_TEST'].get(k), ssim['PROF_TEST_crop'].get(k)
        rd = res['ssim_readable'].get(k, {})
        w_(f'| {k.replace("-", " − ")} | {fo(o)} | {fm(o)} | {fo(p)} | {fm(p)} | {fo(pc)} | '
           f'{rd.get("OURS", "–")} / {rd.get("PROF_TEST", "–")} / {rd.get("PROF_TEST_crop", "–")} |')
    if any(gaps[p] for p in gaps):
        w_('\n격차 변화 (슬라이스별 [(AX − refX) − (기준 회차 − ref)] 평균 ± SE (양/음), 기술 — 잡음 바닥보다 작으면 해석하지 않음):\n')
        w_('| 회차 | OURS | PROF_TEST | PROF_TEST_crop |')
        w_('|---|---|---|---|')
        for k in OrderedDict.fromkeys(list(gaps['OURS']) + list(gaps['PROF_TEST'])):
            w_(f'| {k} | {fo(gaps["OURS"].get(k))} | {fo(gaps["PROF_TEST"].get(k))} | {fo(gaps["PROF_TEST_crop"].get(k))} |')
    w_('\n## 해석 한계\n')
    w_(GUARD + '\n')
    with open(os.path.join(out, 'loss_compare.md'), 'w') as f:
        f.write('\n'.join(m) + '\n')


def main_report(a):
    out = os.path.abspath(a.out_dir)
    de_dir = os.path.abspath(a.dualeval or os.path.join(out, 'dualeval'))
    names = ['ref', 'ref2'] + list(SEEDS) + [p + x for x in SETTINGS for p in ['ref', 'ref2'] + list(SEEDS)]
    D = OrderedDict((c, RP.load_cond(out, c)) for c in names)
    D = OrderedDict((c, d) for c, d in D.items() if d is not None)
    if 'ref' not in D or 'A' not in D or not (D['ref']['completed'] and D['A']['completed']):
        print('[loss] ref·A 가 완료되지 않음 — 중단', flush=True)
        return 1
    V, base, res_X = analyse(D, None, a.relax_preset)

    sat = OrderedDict((c, sat_traj(D[c])) for c in D if not c.startswith('ref') and D[c]['probes'])
    tests = OrderedDict()
    for c in D:
        es = D[c]['eval_summary'] or {}
        if es.get('tests'):
            tests[c] = OrderedDict((t, es['tests'].get(t)) for t in ('zero', 'adj', 'adjprev', 'cross', 'chanmean', 'adj_sym'))
    DE, stale = OrderedDict(), OrderedDict()
    for c in D:
        rows, why = read_dualeval(de_dir, c, D)
        if rows is not None:
            DE[c] = rows
        elif why:
            stale[c] = why
    ptests = OrderedDict((c, t) for c in DE for t in [prof_tests(DE, c)] if t)
    pairs = [('A', 'ref')] + [(k, 'ref') for k in ('A2', 'A3')] + [(k + X, 'ref' + X) for X in SETTINGS for k in SEEDS]
    pairs += [('ref2', 'ref')] + [('ref2' + X, 'ref' + X) for X in SETTINGS]
    ssim = OrderedDict()
    for proto in PROTOS:
        ssim[proto] = OrderedDict()
        for x, r in pairs:
            if x in D and r in D:
                ssim[proto][f'{x}-{r}'] = ssim_delta(D, DE, x, r, proto)
    readable = OrderedDict((k, readable_flags(ssim, k)) for k in ssim['OURS'])
    gaps = OrderedDict()
    for proto in PROTOS:
        gaps[proto] = OrderedDict()
        for X in SETTINGS:
            for k, c0 in SEEDS.items():
                if k + X in D and 'ref' + X in D and c0 in D:
                    gaps[proto][k + X] = gap_change(D, DE, k + X, 'ref' + X, c0, 'ref', proto)

    res = OrderedDict(time=datetime.datetime.now().isoformat(timespec='seconds'), out=out, dry_run=a.relax_preset,
                      rule=OrderedDict(window=WIN, S3_MED=S3_MED, S3_N_MIN=S3_N_MIN, MOVE=MOVE, STAY=STAY, MODE_BAD_R=MODE_BAD_R, MODE_BAD_SAT=MODE_BAD_SAT,
                                       MODE_GOOD_R=MODE_GOOD_R, MODE_GOOD_SAT=MODE_GOOD_SAT, N_SEED_MIN=N_SEED_MIN,
                                       SHARE_LO=SHARE_LO, SHARE_HI=SHARE_HI, primary=PRIMARY, metrics=METRIC_DESC),
                      settings=SETTINGS, validity=V, baseline=base, settings_results=res_X, sat_v=sat, seq_tests=tests,
                      seq_tests_prof=ptests, ssim=ssim, ssim_readable=readable, ssim_gap_change=gaps,
                      noise=dict(ref2_minus_ref_crop=(ssim['PROF_TEST_crop'].get('ref2-ref') or {}).get('mean'),
                                 ref2_minus_ref_ours=(ssim['OURS'].get('ref2-ref') or {}).get('mean'),
                                 ref2_minus_ref_prof=(ssim['PROF_TEST'].get('ref2-ref') or {}).get('mean'), fixed_note_ours=NOISE_REF2),
                      dualeval_dir=de_dir, dualeval_conds=list(DE), dualeval_stale=stale, guard=GUARD)
    with open(os.path.join(out, 'loss_compare.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    write_md(res, D, DE, V, base, res_X, ssim, gaps, tests, ptests, sat, out, a.relax_preset)
    vs = {X: dict(primary_S3=(r['s3'] or {}).get('verdict'), secondary=r['verdict']) for X, r in res_X.items() if r['seeds']}
    print(f'[loss] {vs} modes P0 {dict((c, b["mode"]) for c, b in base.items())} → {out}/loss_compare.md', flush=True)
    return 0


# ------------------------------------------------------------------ 규칙 단위 시험 (합성 값)
def selftest():
    def pdl(delta, half):                                   # Δ 와 CI 반폭으로 paired_delta 모양 만들기
        return dict(delta=delta, ci95_t9=[delta - half, delta + half])
    assert mode_of(-0.013, 0.507) == 'good' and mode_of(0.314, 0.921) == 'bad'          # A·A2 실측
    assert mode_of(0.05, 0.5) == 'intermediate' and mode_of(0.02, 0.7) == 'intermediate' and mode_of(0.0, 0.85) == 'bad'
    assert mode_of(None, 0.5) is None
    S = seed_status
    cases = [
        # (Δ, CI 반폭, 모드 P0, 모드 X) → 상태
        ((-0.25, 0.05, 'bad', 'good'), ('move', -1)),       # 크게 나아짐 + 모드 뒤집힘
        ((-0.25, 0.30, 'bad', 'bad'), ('neither', None)),   # 크지만 CI 가 0 포함 → move 아님
        ((-0.12, 0.05, 'bad', 'bad'), ('move', -1)),        # 크기 기준만
        ((-0.05, 0.02, 'bad', 'good'), ('move', -1)),       # 모드 뒤집힘만
        ((0.15, 0.03, 'good', 'bad'), ('move', 1)),         # 나빠짐
        ((0.12, 0.03, 'bad', 'good'), ('conflict', None)),  # 크기(+) 와 뒤집힘(−) 이 반대
        ((0.01, 0.01, 'good', 'good'), ('stay', None)),
        ((0.01, 0.01, 'good', 'intermediate'), ('neither', None)),   # 작지만 모드가 바뀜
        ((0.05, 0.01, 'bad', 'bad'), ('neither', None)),
    ]
    for (d, h, m0, mx), exp in cases:
        got = S(pdl(d, h), m0, mx)
        assert got == exp, ((d, h, m0, mx), got, exp)
    MV = metric_verdict
    assert MV([('move', -1)])['verdict'] == 'descriptive_only'
    assert MV([('move', -1), ('move', -1)])['verdict'] == 'changed'
    assert MV([('move', -1), ('move', -1), ('stay', None)])['verdict'] == 'changed'
    assert MV([('move', -1), ('move', -1), ('move', 1)])['verdict'] == 'unresolved'
    assert MV([('move', -1), ('move', -1), ('conflict', None)])['verdict'] == 'unresolved'
    assert MV([('move', -1), ('stay', None)])['verdict'] == 'unresolved'
    assert MV([('stay', None), ('stay', None), ('stay', None)])['verdict'] == 'unchanged'
    assert MV([('stay', None), ('neither', None)])['verdict'] == 'unresolved'
    assert MV([('move', 1), ('move', 1)])['direction'] == 'ETER-net relatively worse under X'
    SV = setting_verdict
    ch_b = MV([('move', -1), ('move', -1)])
    ch_w = MV([('move', 1), ('move', 1)])
    un = MV([('stay', None), ('stay', None)])
    ur = MV([('stay', None), ('neither', None)])
    assert SV({'objL': ch_b}) == 'changed' and SV({'objL': un}) == 'unchanged' and SV({'objL': ur}) == 'unresolved'
    assert SV({'l1m': ch_b, 'sm': un}) == 'changed' and SV({'l1m': ch_b, 'sm': ch_w}) == 'changed (mixed)'
    assert SV({'l1m': un, 'sm': un}) == 'unchanged' and SV({'l1m': un, 'sm': ur}) == 'unresolved'
    assert SV({'l1m': MV([('move', -1)]), 'sm': MV([('stay', None)])}) == 'descriptive_only'
    # paired_delta: 블록 짝 차 CI 가 각 비의 CI 보다 좁을 수 있음 (같은 배치 위치의 공통 변동이 빠짐)
    wx = dict(R=0.20, blocks_r=[0.20 + 0.05 * math.sin(j) for j in range(10)])
    w0 = dict(R=0.10, blocks_r=[0.10 + 0.05 * math.sin(j) + 0.001 * j for j in range(10)])
    p = paired_delta(wx, w0)
    assert abs(p['delta'] - 0.10) < 1e-12 and p['ci95_t9'][1] - p['ci95_t9'][0] < 0.01, p
    # 1차(S3) 규칙
    r0 = [-0.013, 0.312, 0.17]                                   # A·A2 실측 + A3 가정값
    assert s3_verdict(r0, [-0.10, -0.08, -0.05])['verdict'] == 'better'
    assert s3_verdict(r0, [-0.03, -0.02, -0.05])['verdict'] == 'better'           # 분리 + 중앙값 차 = 19 %p
    assert s3_verdict(r0, [0.40, 0.35, 0.33])['verdict'] == 'worse'
    assert s3_verdict(r0, [0.00, 0.30, 0.15])['verdict'] == 'unchanged'           # 겹침, 차 −2 %p
    assert s3_verdict(r0, [0.00, 0.05, 0.08])['verdict'] == 'unresolved'          # 겹침, 차 −9 %p
    assert s3_verdict(r0, [-0.02, -0.015, -0.014])['verdict'] == 'better'         # 분리(max −1.4 < −1.3), 차 18.5 %p
    assert s3_verdict([0.0, 0.01, 0.02], [-0.01, -0.02, -0.005])['verdict'] == 'unresolved'   # 분리지만 차 < 5 %p
    assert s3_verdict(r0, [0.1])['verdict'] == 'descriptive_only'
    assert s3_verdict(r0[:2], [0.40, 0.35])['verdict'] == 'worse'
    print('[loss selftest] OK — s3_verdict/mode/seed_status/metric_verdict/setting_verdict/paired_delta', flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results',
                                                      'smoke_deter_opt_2026-10-14'))
    ap.add_argument('--dualeval', default=None, help='smoke_deter_opt_dualeval.py 출력 폴더 (기본 <out>/dualeval)')
    ap.add_argument('--relax-preset', action='store_true', help='점검용: preset·aux 대조 실패를 타당성에 넣지 않음 (DRY RUN 표시)')
    ap.add_argument('--selftest', action='store_true', help='규칙 함수 단위 시험 (합성 값)')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return main_report(a)


if __name__ == '__main__':
    sys.exit(main())
