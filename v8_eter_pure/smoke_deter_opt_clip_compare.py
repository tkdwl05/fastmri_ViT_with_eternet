"""기울기 급등 차단 확인 S4 판정 (2026-10-16) — CPU 전용, torch 미사용. smoke_deter_opt.py 설계 결정 17.

배경: 손실 요인 1단계(S3)의 교수님식 손실 P 세 회차(AP·A2P·A3P)는 모두 1~750 step 에서 GRU 가 포화되지 않은 채 U-Net 단독(refP)보다
마스크 L1 이 ≈8 % 낮았다(회차 간 차 ≤ 1.3 %p). 포화는 유한하지만 매우 큰 기울기(상한 1000 통과) 직후에 시작됐다:
A2P = step 828~861(gn 최대 1.5e3), A3P = step 848~861(gn 최대 1.5e6, 그 전 815~822 overflow 건너뜀 4 회),
AP = step 1591(gn 1758)·1652(gn 960) — AP 의 그 전 885~1522 overflow 건너뜀 14 회는 고립돼 있었고 해가 없었다(그 구간 마스크 L1 비
−8 ~ −11 %, sat_v 0). 그 뒤 열세로 남았다(docs/smoke_deter_opt_report_2026-10-15.md §8 — 결과를 본 뒤 고른 구간이라 가설로만).
S4 = 같은 조건에 기울기 크기 상한만 1000 → 20: APc·A2Pc·A3Pc(P, GRU seed 1·2·3) + A2c(P0 seed 2). 사용자 10-16 승인.
s* = 기준 조건(AP·A2P·A3P·A2)에서 처음 유한 gn > 20 인 step (AP 1591·A2P 828·A3P 848·A2 722). 새 조건은 s* 까지 기준 조건과 비트
단위로 같아야 하고(E_prefix), 상한의 효과는 s* 뒤에만 있다.

판정 규칙 (S4 실행 전에 고정, 10-16; 첫 판을 검토 워크플로 지적으로 고친 판 — 실행 전):
  회차 분류 (새 P 조건 A_kPc):
    구조 타당성: completed · I1 · E2(vs ref match) · I6 · GRU 초기값(seed 1 = I2 통과, seed 2·3 = seed 1 과 다르고 invariants.json 의
      gru_ck_A2·gru_ck_A3 와 같음) · preset = 기준 preset + max_norm 20 · end_checks.E3_clip.max_norm = 20 · nan_skip ≤ 10 ·
      E_prefix(bpos ≤ b* 의 loss·건너뜀이 기준 조건과 비트 같음, b* = s* 의 bpos) · s* 뒤에 기준 조건과 다른 loss 가 있음 ·
      counters.clipped ≥ 1 · P 조건은 3001~4000 배치에 aux 기록.
    안정 타당성 E3' (S3 의 E3 = amp_floor ≤ 5 를 대신함 — E3 는 배율이 하한에 있을 때의 고립 overflow 도 세어 결과에 좌우되므로):
      s* 뒤(bpos > b*) 건너뜀(amp_skip 또는 skipped) 비율 ≤ 1 % 그리고 가장 긴 연속 건너뜀 ≤ 20. s* 까지의 사건은 기준 조건에서
      물려받은 것(상한과 무관)이라 세지 않는다. S3 의 E3 값은 주석으로 같이 보고한다.
    분류: valid(둘 다 통과) / overflow(구조 통과, E3' 실패 — 결측이 아니라 '불안정 미제거(overflow 형)'라는 결과) / invalid(구조 실패).
  Q1 (1차, 가설) — 상한 20 에서 P 의 ETER-net 이 U-Net 단독보다 앞서는가.
    지표: R_k = R_l1m(A_kPc vs refP) = 같은 배치 마스크 L1 비 − 1, 배치 3001~4000, 블록 10 개 t_9 95 % CI
          (loss_compare.win_ratio 'l1m' = aux.l1_mask). refP 는 4000 step 내내 gn < 20 → 상한 20 에서도 같은 궤적(E_eq).
    valid 회차 ≥ 2 일 때:
      advantage(우세)   : 모든 회차 CI 상한 < 0  그리고 중앙값 ≤ −3 %   (중앙값 ≤ −7.5 % 면 '급등 전 크기 유지' 표시 — 기술)
      worse(열세)       : 모든 회차 CI 하한 > 0  그리고 중앙값 ≥ +3 %
      parity(동등)      : 모든 회차 |R_k| ≤ 3 % 그리고 CI ⊂ [−5 %, +5 %]
      mixed(보류)       : 그 밖
    valid < 2 이고 valid + overflow ≥ 2 이며 overflow ≥ 1 → not_blocked_overflow(불안정 미제거, overflow 형) / 그 밖 → descriptive_only.
    3 % = U-Net 회차 간 잡음(ref2P vs refP 마스크 L1 비 +1.38 %)의 약 2 배.
  Q1b (1차, 처치 효과) — 같은 seed 에서 상한 20 이 상한 1000 판보다 나은가 (s* 까지 비트 같고 같은 배치 → 가장 깨끗한 비교).
    Δ_k = R_l1m(A_kPc vs refP) − R_l1m(A_kP vs refP), 블록 짝 차 t_9 95 % CI (loss_compare.paired_delta). 상한 1000 판(AP 포함)은
    그 seed 의 반사실이므로 자신의 타당성과 관계없이 그대로 쓴다. valid 회차 ≥ 2 일 때:
      clip_helps : 모든 회차 CI 상한 < 0 그리고 모든 Δ ≤ −10 %p
      clip_hurts : 모든 회차 CI 하한 > 0 그리고 모든 Δ ≥ +10 %p
      no_effect  : 모든 |Δ| ≤ 3 %p
      unresolved : 그 밖 / valid < 2 → descriptive_only
  기전 (해석용, 사전 고정 — 판정을 바꾸지 않는다):
    회차별 포화 억제 = s* 뒤 probe sat_v 의 최댓값 ≤ 0.10 (S3 P 회차는 0.26~0.47, 급등 전 0); sat_v(4000) 도 같이 보고.
    잦은 clip 표시 = s* 뒤 gn > 20 인 step 비율 > 1 % (처치가 급등에만 걸리지 않음).
    · valid 회차 모두 억제 → Q1·Q1b 는 '상한 20(포화 억제) 상태에서 P 의 효과'로 읽는다.
    · 억제 안 된 회차가 있음 → '상한 20 이 그 회차의 포화를 막지 못함' — 이것은 급등 가설의 반증이 아니다(전역 상한 20 은 원소별로
      정규화되는 Adam 갱신을 비례로 줄이지 않는다 — 검토 지적).
  Q2 (2차) — 상한 20 에서 S3 규칙 재현, 짝 맞춤: RX = valid 인 A_kPc 의 자기 목적함수 비(vs refP), R0 = 같은 seed 의 P0 상한 20 판
    (seed 1 = A, 2 = A2c, 3 = A3; 자기 목적함수 vs ref) — loss_compare.s3_verdict 그대로(양쪽 ≥ 2).
  E_eq (재사용 조건 ref·refP·A·A3): 4000 step 의 유한 gn 최댓값 < 20 → 상한 20 에서 같은 궤적(clip_grad_norm_ 이 정확히 1.0 을
    곱함 — 하네스 selftest 가 이 torch 판에서 확인). 실패하면 그 조건을 쓰는 판정에 주석을 단다(Q2 에서는 뺀다).
  A2c(P0 seed 2): 구조 타당성 + E3' 로 Q2 의 R0 에 들어간다.
  같이 보고(기술): 7 구간 비(자기 목적함수 / 마스크 L1) — 새 조건과 상한 1000 판, A2c vs A2, 회차별 clip 수·s* 뒤 건너뜀·
    seq_rms·sat_v, step 4000 E256 SSIM(OURS; dualeval 이 있으면 PROF_TEST). SSIM 은 판정에 쓰지 않는다.
  해석 한계: 이 설계(BS 8, 4000 step, U-Net 초기값 하나, 같은 배치 흐름)의 선별 시험이다. 어떤 결과도 '시퀀스 모듈이
    불필요하다/해롭다'로 쓰지 않는다. 학습 손실 관계의 판정이지 평가 성능의 우열이 아니다.
출력: <out>/clip_compare.md · clip_compare.json
사용: CUDA_VISIBLE_DEVICES="" python v8_eter_pure/smoke_deter_opt_clip_compare.py --out-dir results/smoke_deter_opt_2026-10-14 [--selftest]
"""

import os
import sys
import json
import math
import argparse
import datetime
from collections import OrderedDict
from statistics import median

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_deter_opt_report as RP          # noqa: E402
import smoke_deter_opt_loss_compare as LC    # noqa: E402

# ── 사전 고정 상수 (위 머리말과 같아야 함) ──
CLIP = 20.0
Q1_MED = 0.03                                 # advantage/worse 의 중앙값 문턱, parity 의 |R| 문턱
PARITY_CI = 0.05                              # parity: CI ⊂ [−5 %, +5 %]
FULL_SIZE = -0.075                            # 기술: 급등 전 크기(마스크 L1 −7.5 ~ −8.8 %) 유지 표시
D_MOVE, D_STAY = 0.10, 0.03                   # Q1b: clip_helps/hurts 는 모든 |Δ| ≥ 10 %p, no_effect 는 모든 |Δ| ≤ 3 %p
N_MIN = 2
SAT_OK = 0.10                                 # 포화 억제: s* 뒤 probe sat_v 최댓값 ≤ 0.10
SKIP_SHARE, SKIP_RUN = 0.01, 20               # E3': s* 뒤 건너뜀 비율 ≤ 1 %, 연속 건너뜀 ≤ 20
CLIP_FREQ = 0.01                              # 기술: s* 뒤 gn > 20 비율 > 1 % → 잦은 clip
NAN_MAX = RP.E3_NAN_MAX
W = RP.W6
SEEDS = OrderedDict([(1, dict(P='AP', Pc='APc', P0='A', P0c=None)),
                     (2, dict(P='A2P', Pc='A2Pc', P0='A2', P0c='A2c')),
                     (3, dict(P='A3P', Pc='A3Pc', P0='A3', P0c=None))])
NEW = OrderedDict([('APc', 'AP'), ('A2Pc', 'A2P'), ('A3Pc', 'A3P'), ('A2c', 'A2')])   # 새 조건 → 기준 조건 (하네스 CLIP_DERIVED)
REUSED = ('ref', 'refP', 'A', 'A3')
WINDOWS = [(1, 100), (101, 500), (501, 750), (751, 1000), (1001, 2000), (2001, 3000), (3001, 4000)]
GUARD = ("주장은 이 설계(BS 8, 4000 step, U-Net 초기값 하나, 같은 배치 흐름)의 선별 시험 범위로 한정한다. 학습 손실 관계의 판정이지 "
         "평가 성능의 우열이 아니며, 어떤 결과도 '시퀀스 모듈이 불필요하다/해롭다'로 쓰지 않는다. 기전 표는 해석을 고를 뿐 판정을 바꾸지 않는다.")


def pct(v, fmt='+.2f'):
    return '–' if v is None else f'{v * 100:{fmt}} %'


def f3(v):
    return '–' if v is None else f'{v:.3f}'


def f4(v):
    return '–' if v is None else f'{v:.4f}'


def ci_txt(ci):
    return f' [{pct(ci[0])}, {pct(ci[1])}]' if ci else ''


def is_skip(st):
    return bool(st.get('amp_skip') or st.get('skipped'))


# ------------------------------------------------------------------ 규칙 (순수 함수 — selftest 대상)
def q1_verdict(rs, n_overflow=0):
    """rs = [(R, ci)] valid 회차만 (ci = (lo, hi) 또는 None). n_overflow = 구조 통과·E3' 실패 회차 수."""
    out = OrderedDict(n=len(rs), n_overflow=n_overflow, R=[r for r, _ in rs], ci=[c for _, c in rs])
    if len(rs) < N_MIN:
        if n_overflow >= 1 and len(rs) + n_overflow >= N_MIN:
            out.update(verdict='not_blocked_overflow', text=f'불안정 미제거(overflow 형) — overflow {n_overflow} 회차, valid {len(rs)}')
        else:
            out.update(verdict='descriptive_only', text=f'valid 회차 {len(rs)} 개 < {N_MIN} — 판정 불가(기술만)')
        return out
    med = median([r for r, _ in rs])
    out['median'] = med
    all_ci = all(c is not None for _, c in rs)
    if all_ci and all(c[1] < 0 for _, c in rs) and med <= -Q1_MED:
        v, t = 'advantage', '우세 — 모든 회차 CI 상한 < 0, 중앙값 ≤ −3 %'
        out['full_size'] = med <= FULL_SIZE
    elif all_ci and all(c[0] > 0 for _, c in rs) and med >= Q1_MED:
        v, t = 'worse', '열세 — 모든 회차 CI 하한 > 0, 중앙값 ≥ +3 %'
    elif all_ci and all(abs(r) <= Q1_MED for r, _ in rs) and all(-PARITY_CI <= c[0] and c[1] <= PARITY_CI for _, c in rs):
        v, t = 'parity', '동등 — 모든 회차 |R| ≤ 3 %, CI ⊂ [−5 %, +5 %]'
    else:
        v, t = 'mixed', '보류 — 위 셋에 들지 않음'
    if n_overflow:
        t += f' (그 밖 overflow 형 {n_overflow} 회차)'
    out.update(verdict=v, text=t)
    return out


def q1b_verdict(ds):
    """ds = [(Δ, ci)] valid 회차만."""
    out = OrderedDict(n=len(ds), delta=[d for d, _ in ds], ci=[c for _, c in ds])
    if len(ds) < N_MIN:
        out.update(verdict='descriptive_only', text=f'valid 회차 {len(ds)} 개 < {N_MIN} — 판정 불가(기술만)')
        return out
    all_ci = all(c is not None for _, c in ds)
    if all_ci and all(c[1] < 0 for _, c in ds) and all(d <= -D_MOVE for d, _ in ds):
        v, t = 'clip_helps', '상한 20 이 도움 — 모든 회차 CI 상한 < 0, 모든 Δ ≤ −10 %p'
    elif all_ci and all(c[0] > 0 for _, c in ds) and all(d >= D_MOVE for d, _ in ds):
        v, t = 'clip_hurts', '상한 20 이 해로움 — 모든 회차 CI 하한 > 0, 모든 Δ ≥ +10 %p'
    elif all(abs(d) <= D_STAY for d, _ in ds):
        v, t = 'no_effect', '효과 없음 — 모든 |Δ| ≤ 3 %p'
    else:
        v, t = 'unresolved', '보류 — 위 셋에 들지 않음'
    out.update(verdict=v, text=t)
    return out


def mech_reading(sat_maxes):
    """sat_maxes = valid 회차의 s* 뒤 probe sat_v 최댓값."""
    if not sat_maxes:
        return 'n/a', 'valid 회차 없음'
    if all(s is not None and s <= SAT_OK for s in sat_maxes):
        return 'suppressed', 'valid 회차 모두 포화 억제(s* 뒤 sat_v 최댓값 ≤ 0.10) — Q1·Q1b 는 상한 20(포화 억제) 상태에서 P 의 효과로 읽는다'
    return 'not_suppressed', ('포화가 억제되지 않은 회차가 있음 — 상한 20 이 그 회차의 포화를 막지 못했다. 급등 가설의 반증은 아니다'
                              '(전역 상한은 원소별 정규화되는 Adam 갱신을 비례로 줄이지 않음)')


def first_over(steps, thr=CLIP):
    """기준 조건에서 처음 유한 gn > thr 인 (step, bpos) (없으면 (None, None))."""
    for st in steps:
        g = st.get('gn')
        if not st.get('skipped') and g is not None and math.isfinite(g) and g > thr:
            return int(st['step']), int(st['bpos'])
    return None, None


def _same(a, b):
    la, lb = a.get('loss'), b.get('loss')
    same = (la == lb) or (isinstance(la, float) and isinstance(lb, float) and math.isnan(la) and math.isnan(lb))
    return same and bool(a.get('skipped')) == bool(b.get('skipped'))


def prefix_identity(sb, sc, b_star):
    """bpos ≤ b* (b* 없으면 공통 전 구간)의 loss·skipped 가 비트 같은가. → (ok, n_compared, first_mismatch_bpos)."""
    bb = {int(s['bpos']): s for s in sb}
    cc = {int(s['bpos']): s for s in sc}
    end = b_star if b_star is not None else min(max(bb, default=-1), max(cc, default=-1))
    n = 0
    for k in range(0, end + 1):
        a, b = bb.get(k), cc.get(k)
        if a is None or b is None or not _same(a, b):
            return False, n, k
        n += 1
    return True, n, None


def diverges_after(sb, sc, b_star):
    """bpos > b* 에서 기준 조건과 다른 줄이 하나라도 있는가."""
    bb = {int(s['bpos']): s for s in sb}
    return any(int(s['bpos']) > b_star and int(s['bpos']) in bb and not _same(bb[int(s['bpos'])], s) for s in sc)


def post_stability(sc, b_star):
    """bpos > b* (b* 없으면 step > 20) 의 건너뜀 비율·최장 연속 건너뜀·clip 비율."""
    post = sorted((s for s in sc if (int(s['bpos']) > b_star if b_star is not None else int(s['step']) > 20)),
                  key=lambda s: int(s['bpos']))
    n = len(post)
    nskip = sum(1 for s in post if is_skip(s))
    run = best = 0
    for s in post:
        run = run + 1 if is_skip(s) else 0
        best = max(best, run)
    nclip = sum(1 for s in post if not is_skip(s) and s.get('gn') is not None and math.isfinite(s['gn']) and s['gn'] > CLIP)
    share = nskip / n if n else None
    return OrderedDict(n_post=n, n_skip=nskip, skip_share=share, longest_skip_run=best, n_clip=nclip,
                       clip_share=nclip / n if n else None,
                       ok=bool(n) and share <= SKIP_SHARE and best <= SKIP_RUN)


def gn_max(steps):
    g = [s['gn'] for s in steps if s.get('gn') is not None and math.isfinite(s['gn'])]
    return max(g) if g else None


# ------------------------------------------------------------------ 타당성
def validity_new(D, inv, c, base):
    d, db = D.get(c), D.get(base)
    if d is None or not d['summary']:
        st = 'running/interrupted (meta only)' if (d and d.get('meta')) else '없음'
        return OrderedDict(cls='invalid', struct_ok=False, why=[st], notes=[], prefix=None, post=None)
    s = d['summary']
    why, notes = [], []
    if not d['completed']:
        why.append(f"status {s.get('status')}")
    ivs, end, cnt = s.get('invariants') or {}, s.get('end_checks') or {}, s.get('counters') or {}
    if not (ivs.get('I1') or {}).get('ok'):
        why.append('I1')
    if (end.get('E2') or {}).get('vs_ref') != 'match':
        why.append(f"E2 {(end.get('E2') or {}).get('vs_ref')}")
    if not (ivs.get('I6') or {}).get('ok'):
        why.append('I6')
    if cnt.get('nan_skip') is None or cnt['nan_skip'] > NAN_MAX:
        why.append(f"nan_skip {cnt.get('nan_skip')}")
    seed = 1 if base in ('AP', 'A') else (2 if base.startswith('A2') else 3)
    i2 = ivs.get('I2') or {}
    if seed == 1:
        if not i2.get('ok') or 'skipped' in i2:
            why.append('I2 (seed 1)')
    else:
        want = inv.get(f'gru_ck_A{seed}')
        if not i2.get('differs_from_seed1'):
            why.append('GRU 초기값이 seed 1 과 같음')
        if want is None or i2.get('gru_ck') != want:
            why.append(f"GRU 초기값 {i2.get('gru_ck')} != gru_ck_A{seed} {want}")
    P = s.get('preset') or {}
    Pb = ((db or {}).get('summary') or {}).get('preset')
    if P.get('max_norm') != CLIP:
        why.append(f"preset max_norm {P.get('max_norm')} != {CLIP}")
    if Pb is None:
        why.append(f'기준 조건 {base} summary 없음')
    elif {k: v for k, v in P.items() if k != 'max_norm'} != Pb:
        why.append(f'나머지 preset != {base}')
    if (end.get('E3_clip') or {}).get('max_norm') != CLIP:
        why.append(f"E3_clip.max_norm {(end.get('E3_clip') or {}).get('max_norm')} != {CLIP} (실제 적용 상한)")
    if 'lam_ssim' in P:
        steps_w = [st for st in d['steps'] if not st.get('skipped') and W[0] <= st['bpos'] + 1 <= W[1]]
        if not steps_w or not all('aux' in st for st in steps_w):
            why.append('W6 배치에 aux 기록 없음')
    pref, post = None, None
    if db is None or not db['steps']:
        why.append(f'기준 조건 {base} steps 없음')
    else:
        s_star, b_star = first_over(db['steps'])
        ok, n, mm = prefix_identity(db['steps'], d['steps'], b_star)
        pref = OrderedDict(s_star=s_star, b_star=b_star, ok=ok, n=n, first_mismatch_bpos=mm)
        if not ok:
            why.append(f'E_prefix: bpos {mm} 에서 {base} 와 다름 (s* = {s_star})')
        if b_star is not None:
            if not diverges_after(db['steps'], d['steps'], b_star):
                why.append(f's* 뒤에도 {base} 와 같음 (상한이 작동하지 않음)')
            if not cnt.get('clipped'):
                why.append('counters.clipped = 0')
        post = post_stability(d['steps'], b_star)
    e3_old = RP.e3_ok(s)
    notes.append(f"S3 의 E3(amp_floor ≤ 5): {'통과' if e3_old else '실패'} (amp_floor {cnt.get('amp_floor')}, 기준 조건에서 물려받은 s* 이전 포함)")
    struct_ok = not why
    if not struct_ok:
        cls = 'invalid'
    elif post is not None and post['ok']:
        cls = 'valid'
    else:
        cls = 'overflow'
    return OrderedDict(cls=cls, struct_ok=struct_ok, why=why, notes=notes, prefix=pref, post=post)


def eq_reused(D, c):
    d = D.get(c)
    if d is None or not d['completed']:
        return dict(ok=False, gn_max=None, why='없음/미완료')
    g = gn_max(d['steps'])
    return dict(ok=g is not None and g < CLIP, gn_max=g, why='' if (g is not None and g < CLIP) else f'gn_max {g} ≥ {CLIP}')


# ------------------------------------------------------------------ 기술 통계
def sat_after(d, s_star):
    tr = LC.sat_traj(d)
    vals = [v for k, v in tr.items() if s_star is None or k > s_star]
    return max(vals) if vals else None


def mech(d, v):
    st = d['steps']
    rms = [s['seq_rms'] for s in st if s.get('seq_rms') is not None]
    s_star = (v.get('prefix') or {}).get('s_star')
    post = v.get('post') or {}
    return OrderedDict(s_star=s_star, sat_max_after=sat_after(d, s_star), sat_v_4000=LC.sat_at(d),
                       n_clip_post=post.get('n_clip'), clip_share_post=post.get('clip_share'),
                       frequent_clip=(post.get('clip_share') or 0) > CLIP_FREQ,
                       skip_share_post=post.get('skip_share'), longest_skip_run=post.get('longest_skip_run'),
                       seq_rms_max=max(rms) if rms else None, seq_rms_4000=rms[-1] if rms else None)


def windows(D, x, r):
    out = OrderedDict()
    for lo, hi in WINDOWS:
        w_own, w_l1 = LC.win_ratio(D, x, r, 'own', lo, hi), LC.win_ratio(D, x, r, 'l1m', lo, hi)
        out[f'{lo}-{hi}'] = (None if not w_own else w_own['R'], None if not w_l1 else w_l1['R'])
    return out


def prof_mean(de_dir, c):
    p = os.path.join(de_dir, f'{c}_summary.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return (json.load(f).get('mean') or {}).get('ssim_prof')


# ------------------------------------------------------------------ 본체
def analyse(out):
    names = ['ref', 'refP', 'ref2P'] + [v for s in SEEDS.values() for v in (s['P'], s['Pc'], s['P0'], s['P0c']) if v]
    D = OrderedDict((c, RP.load_cond(out, c)) for c in dict.fromkeys(names))
    D = OrderedDict((c, d) for c, d in D.items() if d is not None)
    with open(os.path.join(out, 'invariants.json')) as f:
        inv = json.load(f)
    V = OrderedDict((c, validity_new(D, inv, c, b)) for c, b in NEW.items())
    EQ = OrderedDict((c, eq_reused(D, c)) for c in REUSED)
    notes = []
    if not EQ['refP']['ok']:
        notes.append(f"refP E_eq 실패 ({EQ['refP']['why']}) — Q1·Q1b 의 기준이 상한 20 판과 다를 수 있음")
    rows, rs, ds = OrderedDict(), [], []
    for k, sd in SEEDS.items():
        c, cb = sd['Pc'], sd['P']
        w = LC.win_ratio(D, c, 'refP', 'l1m') if c in D else None
        wb = LC.win_ratio(D, cb, 'refP', 'l1m') if cb in D else None
        pdlt = LC.paired_delta(w, wb) if (w and wb) else None
        w_own = LC.win_ratio(D, c, 'refP', 'own') if c in D else None
        w_sm = LC.win_ratio(D, c, 'refP', 'sm') if c in D else None
        cls = V[c]['cls']
        rows[c] = OrderedDict(seed=k, cls=cls, R_l1m=None if not w else w['R'], ci=None if not w else w.get('ci95_t9'),
                              R_l1m_base=None if not wb else wb['R'], delta=None if not pdlt else pdlt['delta'],
                              delta_ci=None if not pdlt else pdlt.get('ci95_t9'),
                              R_own=None if not w_own else w_own['R'], R_sm=None if not w_sm else w_sm['R'],
                              mech=mech(D[c], V[c]) if c in D else None)
        if cls == 'valid' and w is not None:
            rs.append((w['R'], tuple(w['ci95_t9']) if w.get('ci95_t9') else None))
            if pdlt is not None:
                ds.append((pdlt['delta'], tuple(pdlt['ci95_t9']) if pdlt.get('ci95_t9') else None))
    n_over = sum(1 for r in rows.values() if r['cls'] == 'overflow')
    q1 = q1_verdict(rs, n_over)
    q1b = q1b_verdict(ds)
    mr = mech_reading([r['mech']['sat_max_after'] for r in rows.values() if r['cls'] == 'valid' and r['mech']])
    # Q2 — 짝 맞춤
    r0, rx, q2src = [], [], OrderedDict()
    for k, sd in SEEDS.items():
        c0, cx = sd['P0c'] or sd['P0'], sd['Pc']
        ok0 = (V[c0]['cls'] == 'valid' if c0 in V else EQ.get(c0, {}).get('ok', False)) and EQ['ref']['ok']
        okx = V[cx]['cls'] == 'valid' and EQ['refP']['ok']
        w0 = LC.win_ratio(D, c0, 'ref', 'own') if c0 in D else None
        wx = LC.win_ratio(D, cx, 'refP', 'own') if cx in D else None
        used = bool(ok0 and okx and w0 and wx)
        if used:
            r0.append(w0['R'])
            rx.append(wx['R'])
        q2src[k] = OrderedDict(P0=c0, R0=None if not w0 else w0['R'], ok0=bool(ok0), PX=cx, RX=None if not wx else wx['R'],
                               okX=bool(okx), used=used)
    q2 = LC.s3_verdict(r0, rx)
    # 기술
    win = OrderedDict()
    for k, sd in SEEDS.items():
        for c in (sd['Pc'], sd['P']):
            if c in D:
                win[c] = windows(D, c, 'refP')
        for c in (sd['P0c'], sd['P0']):
            if c and c in D:
                win[c] = windows(D, c, 'ref')
    a2c = None
    if 'A2c' in D and 'A2' in D:
        a2c = OrderedDict(own=LC.win_ratio(D, 'A2c', 'A2', 'own'), l1m=LC.win_ratio(D, 'A2c', 'A2', 'l1m'),
                          vs_ref_own=LC.win_ratio(D, 'A2c', 'ref', 'own'))
    de_dir = os.path.join(out, 'dualeval')
    ssim = OrderedDict((c, OrderedDict(ours=(D[c].get('eval_summary') or {}).get('ssim'), prof=prof_mean(de_dir, c))) for c in D)
    noise = LC.win_ratio(D, 'ref2P', 'refP', 'l1m') if 'ref2P' in D else None
    return OrderedDict(V=V, EQ=EQ, rows=rows, q1=q1, q1b=q1b, mech_reading=mr, q2=q2, q2src=q2src, win=win, a2c=a2c,
                       ssim=ssim, noise_l1m=None if not noise else noise['R'], notes=notes)


def write_md(res, out):
    L = ['# 기울기 급등 차단 확인 S4 — P 의 기울기 상한 1000 → 20 (설계 결정 17)', '',
         f'생성: {datetime.datetime.now().isoformat(timespec="seconds")} · 규칙은 S4 실행 전 고정 '
         '(`v8_eter_pure/smoke_deter_opt_clip_compare.py` 머리말). 배치 3001–4000, 같은 배치 비.', '']
    for n in res['notes']:
        L.append(f'- 주: {n}')
    L += ['## Q1 (1차, 가설) — 기울기 상한 20 에서 P 의 ETER-net 대 U-Net 단독 (마스크 L1 비, vs refP)', '',
          '| 회차 | 조건 | 분류 | R 마스크 L1 [t9 CI] | 상한 1000 판 R | Δ (Q1b) [t9 CI] | R 자기 목적함수 | R 마스크 1−SSIM |',
          '|---|---|---|---|---|---|---|---|']
    for c, r in res['rows'].items():
        why = '' if r['cls'] == 'valid' else ' — ' + '; '.join(res['V'][c]['why'] or ['E3\' 실패'])
        L.append(f"| {r['seed']} | {c} | {r['cls']}{why} | {pct(r['R_l1m'])}{ci_txt(r['ci'])} | {pct(r['R_l1m_base'])} | "
                 f"{pct(r['delta'])}{ci_txt(r['delta_ci'])} | {pct(r['R_own'])} | {pct(r['R_sm'])} |")
    q1, q1b = res['q1'], res['q1b']
    L += ['', f"**Q1 판정 = {q1['verdict']}** — {q1['text']}" + (f" (중앙값 {pct(q1.get('median'))})" if 'median' in q1 else '')
          + (' · 급등 전 크기(≤ −7.5 %) 유지' if q1.get('full_size') else ''),
          f"**Q1b 판정 (상한 효과, 같은 seed) = {q1b['verdict']}** — {q1b['text']}"]
    if res['noise_l1m'] is not None:
        L.append(f"- U-Net 잡음 기준 ref2P vs refP 마스크 L1 비 {pct(res['noise_l1m'])} (문턱 3 % 의 근거)")
    L += ['', '## 기전 (해석용, 사전 고정)', '',
          '| 조건 | s* | s* 뒤 sat_v 최댓값 | sat_v(4000) | s* 뒤 clip step (비율) | 잦은 clip | s* 뒤 건너뜀 비율 | 최장 연속 건너뜀 | seq_rms 최대 / 4000 |',
          '|---|---|---|---|---|---|---|---|---|']
    for c, r in res['rows'].items():
        m = r['mech']
        if not m:
            L.append(f'| {c} | – | – | – | – | – | – | – | – |')
            continue
        L.append(f"| {c} | {m['s_star']} | {f3(m['sat_max_after'])} | {f3(m['sat_v_4000'])} | {m['n_clip_post']} ({pct(m['clip_share_post'])}) | "
                 f"{'예' if m['frequent_clip'] else '아니오'} | {pct(m['skip_share_post'])} | {m['longest_skip_run']} | "
                 f"{f3(m['seq_rms_max'])} / {f3(m['seq_rms_4000'])} |")
    L += ['', f"- 기전 해석: **{res['mech_reading'][0]}** — {res['mech_reading'][1]}", '']
    q2 = res['q2']
    L += ['## Q2 (2차) — 상한 20 에서 S3 규칙 재현, 짝 맞춤 (자기 목적함수)', '',
          '| 회차 | P0 조건 | R0 (vs ref) | P 조건 | RX (vs refP) | 판정에 씀 |', '|---|---|---|---|---|---|']
    for k, s in res['q2src'].items():
        L.append(f"| {k} | {s['P0']}{'' if s['ok0'] else ' (타당성 실패)'} | {pct(s['R0'])} | {s['PX']}{'' if s['okX'] else ' (valid 아님)'} | "
                 f"{pct(s['RX'])} | {'예' if s['used'] else '아니오'} |")
    L += ['', f"**Q2 판정 = {q2['verdict']}** — {q2.get('text', '')}"
          + (f" (중앙값 {pct(q2.get('median_R0'))} → {pct(q2.get('median_RX'))})" if 'median_R0' in q2 else ''), '',
          '## 타당성', '']
    for c, v in res['V'].items():
        p, po = v.get('prefix'), v.get('post')
        pre = ('–' if not p else f"s* = {p['s_star']} (bpos {p['b_star']}), {p['n']} 줄 비교 "
               + ('일치' if p['ok'] else f"불일치 @bpos {p['first_mismatch_bpos']}"))
        st = ('–' if not po else f"s* 뒤 {po['n_post']} 줄 중 건너뜀 {po['n_skip']} ({pct(po['skip_share'])}), 최장 연속 {po['longest_skip_run']} → "
              + ('통과' if po['ok'] else '실패'))
        L.append(f"- {c} (기준 {NEW[c]}): **{v['cls']}**" + (f" — {'; '.join(v['why'])}" if v['why'] else '')
                 + f" · E_prefix {pre} · E3' {st}" + (f" · {'; '.join(v['notes'])}" if v['notes'] else ''))
    for c, e in res['EQ'].items():
        L.append(f"- E_eq {c}: gn_max {e['gn_max']} → {'상한 20 과 같은 궤적' if e['ok'] else '실패 ' + e['why']}")
    cols = list(res['win'])
    L += ['', '## 구간별 같은 배치 비 (기술; 자기 목적함수 / 마스크 L1 — P 계열 vs refP, P0 계열 vs ref)', '',
          '| 구간 | ' + ' | '.join(cols) + ' |', '|---|' + '---|' * len(cols)]
    for wk in [f'{lo}-{hi}' for lo, hi in WINDOWS]:
        L.append(f'| {wk} | ' + ' | '.join(f"{pct(res['win'][c][wk][0], '+.1f')} / {pct(res['win'][c][wk][1], '+.1f')}" for c in cols) + ' |')
    L.append('')
    if res['a2c']:
        a = res['a2c']
        L += [f"- A2c vs A2 (상한 20 이 P0 seed 2 를 바꾼 크기, 기술): 자기 목적함수 {pct((a['own'] or {}).get('R'))}"
              f"{ci_txt((a['own'] or {}).get('ci95_t9'))}, 마스크 L1 {pct((a['l1m'] or {}).get('R'))}; A2c vs ref {pct((a['vs_ref_own'] or {}).get('R'))}", '']
    L += ['## step 4000 E256 SSIM (기술 — 판정에 쓰지 않음)', '', '| 조건 | OURS (마스크) | PROF_TEST (전체 영상, dualeval) |', '|---|---|---|']
    for c, s in res['ssim'].items():
        L.append(f"| {c} | {f4(s['ours'])} | {f4(s['prof'])} |")
    L += ['', '## 해석 한계', '', GUARD, '']
    with open(os.path.join(out, 'clip_compare.md'), 'w') as f:
        f.write('\n'.join(L))


def selftest():
    ci = lambda lo, hi: (lo, hi)                                    # noqa: E731
    assert q1_verdict([(-0.08, ci(-0.10, -0.06))])['verdict'] == 'descriptive_only'
    assert q1_verdict([(-0.08, ci(-0.10, -0.06))], n_overflow=1)['verdict'] == 'not_blocked_overflow'
    assert q1_verdict([], n_overflow=1)['verdict'] == 'descriptive_only'
    r = q1_verdict([(-0.08, ci(-0.10, -0.06)), (-0.09, ci(-0.11, -0.07)), (-0.06, ci(-0.08, -0.04))])
    assert r['verdict'] == 'advantage' and r['full_size']
    r = q1_verdict([(-0.08, ci(-0.10, -0.06)), (-0.02, ci(-0.04, -0.001))])                     # 중앙값 −5 %
    assert r['verdict'] == 'advantage' and not r['full_size']
    assert q1_verdict([(-0.02, ci(-0.03, -0.01)), (-0.025, ci(-0.04, -0.01))])['verdict'] == 'parity'
    assert q1_verdict([(-0.02, ci(-0.06, 0.02)), (0.01, ci(-0.01, 0.03))])['verdict'] == 'mixed'   # CI 가 −5 % 밖
    assert q1_verdict([(-0.08, ci(-0.10, -0.06)), (0.01, ci(-0.01, 0.03))])['verdict'] == 'mixed'
    assert q1_verdict([(0.20, ci(0.15, 0.25)), (0.10, ci(0.05, 0.15))])['verdict'] == 'worse'
    assert q1_verdict([(-0.08, None), (-0.06, None)])['verdict'] == 'mixed'
    assert q1b_verdict([(-0.30, ci(-0.35, -0.25)), (-0.12, ci(-0.18, -0.06))])['verdict'] == 'clip_helps'
    assert q1b_verdict([(-0.30, ci(-0.35, -0.25)), (-0.05, ci(-0.08, -0.02))])['verdict'] == 'unresolved'
    assert q1b_verdict([(0.01, ci(-0.01, 0.03)), (-0.02, ci(-0.05, 0.01))])['verdict'] == 'no_effect'
    assert q1b_verdict([(0.15, ci(0.10, 0.20)), (0.12, ci(0.05, 0.19))])['verdict'] == 'clip_hurts'
    assert mech_reading([0.0, 0.05])[0] == 'suppressed' and mech_reading([0.0, 0.3])[0] == 'not_suppressed'
    # E_prefix / 발산 / E3' (bpos 기준)
    sb = [dict(step=i + 1, bpos=i, loss=0.1 * i, skipped=False, gn=1.0 if i != 5 else 25.0, amp_skip=False) for i in range(60)]
    sc = [dict(st, loss=(st['loss'] if st['bpos'] <= 5 else 9.0)) for st in sb]
    assert first_over(sb) == (6, 5) and prefix_identity(sb, sc, 5) == (True, 6, None) and diverges_after(sb, sc, 5)
    assert not diverges_after(sb, [dict(s) for s in sb], 5)
    sc2 = [dict(s) for s in sc]
    sc2[2]['loss'] = 0.31
    assert prefix_identity(sb, sc2, 5) == (False, 2, 2)
    # 배율 하한의 고립 overflow 6 회는 통과 (S3 의 E3 라면 실패) — 800 줄 중 6 = 0.75 %
    post = [dict(step=i + 1, bpos=i, loss=0.1, skipped=False, gn=(float('inf') if i % 130 == 50 else 1.0),
                 amp_skip=(i % 130 == 50)) for i in range(806)]
    ps = post_stability(post, 5)
    assert ps['n_post'] == 800 and ps['n_skip'] == 6 and ps['longest_skip_run'] == 1 and ps['ok'], ps
    # 연속 50 건너뜀은 실패 → overflow 분류
    post2 = [dict(s, amp_skip=(100 <= s['bpos'] < 150), gn=(float('inf') if 100 <= s['bpos'] < 150 else 1.0)) for s in post]
    ps2 = post_stability(post2, 5)
    assert ps2['longest_skip_run'] == 50 and not ps2['ok'], ps2
    # NaN-skip 줄이 앞 step 번호를 다시 써도 bpos 로 비교
    sb3 = [dict(step=1, bpos=0, loss=0.1, skipped=False), dict(step=2, bpos=1, loss=0.2, skipped=False)]
    sc3 = [dict(step=1, bpos=0, loss=0.1, skipped=False), dict(step=1, bpos=1, loss=float('nan'), skipped=True)]
    assert prefix_identity(sb3, sc3, 1) == (False, 1, 1)
    print('[clip_compare] selftest ok', flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results',
                                                      'smoke_deter_opt_2026-10-14'))
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = os.path.abspath(a.out_dir)
    res = analyse(out)
    write_md(res, out)

    def _d(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return repr(o)
        return str(o)
    with open(os.path.join(out, 'clip_compare.json'), 'w') as f:
        json.dump(res, f, indent=1, default=_d)
    print(f"[clip_compare] Q1 {res['q1']['verdict']} · Q1b {res['q1b']['verdict']} · 기전 {res['mech_reading'][0]} · "
          f"Q2 {res['q2']['verdict']} → {out}/clip_compare.md", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
