"""d-ETER 최적화 교란 스모크 보고서 (2026-10-14) — CPU 전용, torch 미사용. 언제 실행해도 되고 빠진 조건은 건너뛴다.

입력: <out>/{ref,ref2,A,B,C,D,E,EC}/ 의 summary.json · steps.jsonl · log.jsonl · probe.jsonl · eval_per_slice.csv · eval{3500}_per_slice.csv
출력: <out>/summary.md (한국어) · decision.json · fig_*.png (영문 라벨)
판정 규칙은 실행 전에 고정한 것 — 아래 상수와 decide() 가 단일 출처. decision.json 'selected' 는 1차 규칙(spec §10)만으로 정한다.
1차 규칙 (spec §10, 구속):
  타당성  V1 모든 완료 조건의 I1~I9·E1·E2 / V2 ref·A 완료, E3(nan_skip ≤ 10, amp_floor ≤ 5) flag 없음
          / V3 R_A(W6) ≥ +0.020 — 실패하면 후보를 선정하지 않는다("4000 step 안에서 교란 재현 안 됨", S1·S2 는 기술로만)
          / V4 A 의 sat_v ≥ 0.50 (step ≥ 500 의 모든 probe) — 보고만, 선정을 막지 않음.
  S1 포화 낮음: step 500~4000 의 probe 8 개 모두 sat_v ≤ 0.10 (전체 probe 최댓값도 보고)
  S2 슬라이스 고유 정보: E256 에서 d_adj(이웃 = 같은 볼륨 i+1 우선) 평균 > 0, 양측 부호 검정 p < 0.05 (동률 제외)
  S3 최적화 불리 없음: W6 = 배치 3001~4000 의 R_X ≤ +0.005 (블록 t·MBB 95% CI 보고)
  S4 평가 비열등: NOT (mean ΔSSIM(X − ref) < 0 이고 부호 검정 p < 0.05), E256 step 4000
  후보 = S1 ∧ S2 ∧ S3 ∧ S4 (V1·V2·V3 충족 시). 순위 = ΔSSIM(X − ref) 내림차순. 1위와의 짝 부호 검정 p ≥ 0.05 인 후보는 동률
  → C > B > EC > E > D 순으로 고른다.
2차 규칙 (과학·공정성 비판 안, 비구속 — 'selected_amended' 로만 기록, 사용자 승인 전에는 결정에 쓰지 않음):
  V3·V4 실패 시 후보를 '잠정'으로 둠 / S1 + 배치 sat_v 100-step 이동평균 ≤ 0.10 / S2 양쪽 이웃 평균 d_adj_sym
  / S3 R_X ≤ τ = max(0.005, |R_ref2(W6)|) 그리고 R_X + 1.383·SE ≤ +0.02 (앞만 맞으면 'unresolved'; ref2 는 완료·흐름 일치일 때만)
  / S4 mean ΔSSIM(X − ref) ≥ −m, m = max(0.002, |ΔSSIM(ref2 − ref)|) (3500·4000 스냅숏 평균) / 동률 |ΔSSIM(X1 − X2)| < m.
서로 다른 두 망 사이의 슬라이스별 부호·Wilcoxon 검정은 S4·동률 규칙 외에는 기술 통계로만 보고한다.
사용: CUDA_VISIBLE_DEVICES="" python v8_eter_pure/smoke_deter_opt_report.py --out-dir results/smoke_deter_opt_2026-10-14
"""

import os
import sys
import csv
import glob
import json
import math
import argparse
import datetime
from collections import OrderedDict

import numpy as np

ORDER = ['ref', 'ref2', 'A', 'B', 'C', 'D', 'E', 'EC']
X_CONDS = ['B', 'C', 'D', 'E', 'EC']
GRU_CONDS = ['A', 'B', 'C', 'D', 'E', 'EC']
PREF = ['C', 'B', 'EC', 'E', 'D']                           # 동률일 때 (교수님 모델·공유 레시피에서 덜 벗어난 순)
WINDOWS = [(1, 100), (101, 500), (501, 1000), (1001, 2000), (2001, 3000), (3001, 4000)]
W5, W6 = (2001, 3000), (3001, 4000)
V3_MIN = 0.020
V4_MIN = 0.50
S1_MAX = 0.10
S2_P = 0.05
S3_TAU0 = 0.005                                             # 1차: R_X(W6) ≤ +0.005
S3_UPPER = 0.02                                             # 2차 전용
S4_P = 0.05                                                 # 1차 S4·동률의 부호 검정 수준
S4_M0 = 0.002                                               # 2차 전용
E3_NAN_MAX, E3_FLOOR_MAX = 10, 5
S1_PROBE_STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
T9_975 = 2.262                                              # t_{9}, 양측 95%
T9_90 = 1.383                                               # t_{9}, 단측 90%
BOOT_N, BOOT_SEED, BOOT_BLOCK = 10000, 20261014, 100
COLORS = {'ref': '#52514e', 'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a', 'E': '#eda100', 'D': '#e87ba4',
          'EC': '#008300', 'ref2': '#4a3aa7'}                # dataviz 기준 범주 색 순서 고정 (조건마다 고정 색, 순위와 무관)
STYLES = {'ref2': '--', 'EC': '-.', 'D': ':'}
LABELS = {'ref': 'ref (U-Net only)', 'ref2': 'ref2 (U-Net only, init seed 2)', 'A': 'A (current recipe)',
          'B': 'B (GRU LR x0.1)', 'C': 'C (GRU warmup 1000)', 'D': 'D (seq x0.06, eps comp.)', 'E': 'E (zero-init gate)',
          'EC': 'EC (gate + warmup)'}


# ------------------------------------------------------------------ 읽기
def _json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass                                     # 실행 중 마지막 줄이 잘렸을 수 있음
    return out


def _csv_rows(path):
    if not os.path.exists(path):
        return None
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if v == '':
                    d[k] = None
                elif k in ('file',):
                    d[k] = v
                else:
                    try:
                        d[k] = int(v) if k in ('idx', 'slice', 'adj_idx', 'adj2_idx', 'cross_idx') else float(v)
                    except ValueError:
                        d[k] = v
            rows[d['idx']] = d
    return rows


def load_cond(out, c):
    d = os.path.join(out, c)
    if not os.path.isdir(d):
        return None
    summ = _json(os.path.join(d, 'summary.json'))
    snaps = sorted(p for p in glob.glob(os.path.join(d, 'eval*_per_slice.csv')) if not p.endswith('/eval_per_slice.csv'))
    return dict(cond=c, dir=d, summary=summ, meta=_json(os.path.join(d, 'meta.json')),
                steps=_jsonl(os.path.join(d, 'steps.jsonl')), light=_jsonl(os.path.join(d, 'log.jsonl')),
                probes=_jsonl(os.path.join(d, 'probe.jsonl')), eval=_csv_rows(os.path.join(d, 'eval_per_slice.csv')),
                snap=_csv_rows(snaps[-1]) if snaps else None, snap_path=snaps[-1] if snaps else None,
                eval_summary=_json(os.path.join(d, 'eval_summary.json')),
                completed=bool(summ and summ.get('status') == 'completed'))


# ------------------------------------------------------------------ 통계
def sign_test_p(n_pos, n_neg):
    n = n_pos + n_neg
    if n == 0:
        return None
    k = max(n_pos, n_neg)
    return min(1.0, 2.0 * sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n)


def loss_by_b(steps):
    """1 부터 세는 배치 위치 b = bpos+1 → loss (건너뛴·비유한 배치 제외)."""
    out = {}
    for s in steps:
        if s.get('skipped'):
            continue
        v = s.get('loss')
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[s['bpos'] + 1] = float(v)
    return out


def ratio_window(lx, lr, lo, hi, blocks=False, boot=False):
    bs = [b for b in range(lo, hi + 1) if b in lx and b in lr]
    if not bs:
        return None
    x = np.array([lx[b] for b in bs])
    r = np.array([lr[b] for b in bs])
    R = float(x.sum() / r.sum() - 1)
    lrat = np.log(x / r)
    res = OrderedDict(n=len(bs), R=R, logratio_mean=float(lrat.mean()), logratio_median=float(np.median(lrat)),
                      frac_x_lt_ref=float((x < r).mean()))
    res['sign_flag'] = bool(np.sign(R) != np.sign(res['logratio_median']) and R != 0 and res['logratio_median'] != 0)
    if blocks:
        L = (hi - lo + 1) // 10
        rj = []
        for j in range(10):
            sel = [(lx[b], lr[b]) for b in range(lo + j * L, lo + (j + 1) * L) if b in lx and b in lr]
            if sel:
                a = np.array(sel)
                rj.append(float(a[:, 0].sum() / a[:, 1].sum() - 1))
        if len(rj) >= 2:
            se = float(np.std(rj, ddof=1) / math.sqrt(len(rj)))
            res.update(blocks_r=rj, block_se=se, ci95_t9=[R - T9_975 * se, R + T9_975 * se], upper90_t9=R + T9_90 * se,
                       blocks_x_lt_ref=int(sum(v < 0 for v in rj)), n_blocks=len(rj))
    if boot and len(bs) >= BOOT_BLOCK:
        rng = np.random.default_rng(BOOT_SEED)
        nb = max(1, len(bs) // BOOT_BLOCK)
        starts = rng.integers(0, len(bs) - BOOT_BLOCK + 1, size=(BOOT_N, nb))
        cx = np.concatenate([[0.0], np.cumsum(x)])
        cr = np.concatenate([[0.0], np.cumsum(r)])
        sx = (cx[starts + BOOT_BLOCK] - cx[starts]).sum(1)
        sr = (cr[starts + BOOT_BLOCK] - cr[starts]).sum(1)
        bt = sx / sr - 1
        res['boot_ci95_mbb'] = [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
    return res


def moving_ratio(lx, lr, width=100):
    bmax = max(set(lx) & set(lr)) if set(lx) & set(lr) else 0
    if bmax == 0:
        return None, None
    x = np.zeros(bmax + 1)
    r = np.zeros(bmax + 1)
    m = np.zeros(bmax + 1)
    for b in range(1, bmax + 1):
        if b in lx and b in lr:
            x[b], r[b], m[b] = lx[b], lr[b], 1
    cx, cr, cm = np.cumsum(x), np.cumsum(r), np.cumsum(m)
    bs = np.arange(width, bmax + 1)
    sx = cx[bs] - cx[bs - width]
    sr = cr[bs] - cr[bs - width]
    ok = (cm[bs] - cm[bs - width]) > width // 2
    return bs[ok], (sx[ok] / sr[ok] - 1)


def paired(a, b):
    """슬라이스별 a − b (기술 통계): 평균, SE, 95% t 구간, 양/음 개수, 부호 검정 p, Wilcoxon p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return None
    from scipy import stats
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    t = float(stats.t.ppf(0.975, d.size - 1))
    npos, nneg = int((d > 0).sum()), int((d < 0).sum())
    try:
        wp = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else None
    except ValueError:
        wp = None
    return OrderedDict(n=int(d.size), mean=float(d.mean()), se=se, ci95=[float(d.mean() - t * se), float(d.mean() + t * se)],
                       n_pos=npos, n_neg=nneg, p_sign=sign_test_p(npos, nneg), p_wilcoxon=wp)


def ssim_vec(D, c, ids, use_avg):
    e = D[c]['eval']
    if e is None or any(i not in e for i in ids):
        return None
    v = np.array([e[i]['ssim'] for i in ids])
    if use_avg:
        s = D[c]['snap']
        v = 0.5 * (v + np.array([s[i]['ssim'] for i in ids]))
    return v


def metric_vec(D, c, ids, key):
    e = D[c]['eval']
    if e is None or any(i not in e for i in ids):
        return None
    return np.array([e[i][key] for i in ids], float)


def probe_series(D, c, fn):
    xs, ys = [], []
    for p in D[c]['probes']:
        try:
            v = fn(p)
        except (KeyError, TypeError, IndexError):
            v = None
        if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
            xs.append(p['step'])
            ys.append(float(v))
    return xs, ys


def step_series(D, c, key):
    xs, ys = [], []
    for s in D[c]['steps']:
        if s.get('skipped') or s.get(key) is None:
            continue
        v = s[key]
        if isinstance(v, (int, float)) and math.isfinite(v):
            xs.append(s['step'])
            ys.append(float(v))
    return np.array(xs), np.array(ys)


def trailing_mean(y, w):
    if len(y) < w:
        return np.array([]), np.array([])
    c = np.concatenate([[0.0], np.cumsum(y)])
    return np.arange(w - 1, len(y)), (c[w:] - c[:-w]) / w


# ------------------------------------------------------------------ 동일성·타당성
def identity_checks(D):
    """공정성 비판 (a)~(e) — 조건이 의도한 요인만 다른지. 반환 (checks, nonequivalent{cond: [이유]})."""
    by_b = {c: {s['bpos']: s for s in D[c]['steps']} for c in D if D[c]['steps']}
    by_step = {c: {s['step']: s for s in D[c]['steps'] if not s.get('skipped')} for c in D if D[c]['steps']}
    chk, bad = OrderedDict(), {}

    def flag(c, why):
        bad.setdefault(c, []).append(why)

    def eqv(a, b):
        return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))

    abc = [c for c in ('A', 'B', 'C') if c in by_b]
    if len(abc) >= 2:
        r0 = abc[0]
        res = {}
        for c in abc[1:]:
            ok = all(bp in by_b[c] and bp in by_b[r0] and all(eqv(by_b[c][bp].get(k), by_b[r0][bp].get(k))
                                                                 for k in ('loss', 'gn', 'amp_skip')) for bp in (0, 1))
            res[c] = ok
            if not ok:
                flag(c, f'(a) bpos 0·1 의 loss·gn·amp_skip 이 {r0} 와 다름')
        chk['a_ABC_bpos01'] = dict(reference=r0, equal=res)
    if 'ref' in by_b:
        res = {}
        for c in ('E', 'EC'):
            if c in by_b:
                ok = all(bp in by_b[c] and bp in by_b['ref'] and eqv(by_b[c][bp].get('loss'), by_b['ref'][bp].get('loss'))
                         for bp in (0, 1))
                res[c] = ok
                if not ok:
                    flag(c, '(b) bpos 0·1 의 loss 가 ref 와 다름')
        if res:
            chk['b_gate_vs_ref_loss'] = res

    def ck(c, step, key):
        s = by_step.get(c, {}).get(step)
        return None if s is None else s.get(key)

    if len(abc) >= 2:
        v0 = ck(abc[0], 2, 'unet_ck')
        res = {c: ck(c, 2, 'unet_ck') == v0 and v0 is not None for c in abc[1:]}
        for c, ok in res.items():
            if not ok:
                flag(c, f'(c) step 2 unet_ck 가 {abc[0]} 와 다름')
        chk['c_unet_ck_step2_ABC'] = dict(reference=abc[0], value=v0, equal=res)
    if 'ref' in by_step:
        res = {}
        for c in ('E', 'EC'):
            if c in by_step:
                ok = ck(c, 2, 'unet_ck') is not None and ck(c, 2, 'unet_ck') == ck('ref', 2, 'unet_ck')
                res[c] = ok
                if not ok:
                    flag(c, '(c) step 2 unet_ck 가 ref 와 다름')
        if res:
            chk['c_unet_ck_step2_gate_vs_ref'] = res
    res = {}
    for c in GRU_CONDS:
        if c in by_step and D[c]['summary']:
            init = D[c]['summary'].get('gru_ck_init')
            ok = init is not None and ck(c, 1, 'gru_ck') == init
            res[c] = ok
            if not ok:
                flag(c, '(d) step 1 gru_ck 가 초기값과 다름 (step 1 은 GradScaler skip 이어야 함)')
    if res:
        chk['d_gru_ck_step1_eq_init'] = res
    if 'ref' in by_step:
        lr_ref = {k: s['lr_unet'] for k, s in by_step['ref'].items()}
        res = {}
        for c in by_step:
            if c == 'ref':
                continue
            common = [k for k in by_step[c] if k in lr_ref]
            ok = bool(common) and all(by_step[c][k]['lr_unet'] == lr_ref[k] for k in common)
            res[c] = ok
            if not ok:
                flag(c, '(e) lr_unet 열이 ref 와 비트 단위로 다름')
        chk['e_lr_unet_bitwise'] = res
    return chk, bad


def pod_equivalence(Dc):
    """ref·ref2 ↔ pod plain 트레이너: amp_floor 0 이고 grad_retry 가 모두 amp_skip step 에서 회복 0 이면 같은 step 의미."""
    s = Dc['summary'] or {}
    floor = (s.get('counters') or {}).get('amp_floor')
    bad = [x['bpos'] for x in Dc['steps'] if x.get('grad_retry') and not (x.get('amp_skip') and not x.get('grad_recovered'))]
    return dict(amp_floor=floor, retry_not_noop_bpos=bad[:20], equivalent=(floor == 0 and not bad))


def inv_ok(summ):
    """V1: I1~I9 와 E1·E2 (flag_only 항목 제외)."""
    if not summ:
        return False, ['summary 없음']
    why = []
    for k, v in (summ.get('invariants') or {}).items():
        if k.startswith('I') and k != 'I10' and not v.get('ok', False) and not v.get('flag_only'):
            why.append(k)
    need = ['I1', 'I3', 'I4', 'I5', 'I9'] + (['I2', 'I6'] if summ.get('cond') in GRU_CONDS else []) + \
           (['I7', 'I8'] if summ.get('cond') in ('E', 'EC') else [])
    why += [f'{k} 없음' for k in need if k not in (summ.get('invariants') or {})]
    for k in ('E1', 'E2'):
        if not ((summ.get('end_checks') or {}).get(k) or {}).get('ok', False):
            why.append(k)
    return not why, why


# ------------------------------------------------------------------ 판정
def e3_ok(summ):
    """spec E3: nan_skip ≤ 10 그리고 amp_floor ≤ 5 — counters 에서 다시 계산 (clip 은 판정에 쓰지 않음)."""
    cnt = (summ or {}).get('counters') or {}
    ns, fl = cnt.get('nan_skip'), cnt.get('amp_floor')
    return ns is not None and fl is not None and ns <= E3_NAN_MAX and fl <= E3_FLOOR_MAX


def pick(cands, score, tie_with_top):
    """순위 = score 내림차순. 1위와 동률인 후보들 중 PREF 순서가 앞선 것을 고른다. 반환 (선정, 동률 집합)."""
    if not cands:
        return None, []
    top = max(cands, key=score)
    tie = [c for c in cands if c == top or tie_with_top(c, top)]
    return (min(tie, key=PREF.index) if len(tie) > 1 else top), tie


def decide(D, use_avg):
    comp = [c for c in ORDER if c in D and D[c]['completed']]
    lb = {c: loss_by_b(D[c]['steps']) for c in D}
    dec = OrderedDict(time=datetime.datetime.now().isoformat(timespec='seconds'), completed=comp,
                      rule='primary = spec §10 (binding, sets selected); amended = critic proposal (non-binding)')
    # 동일성·흐름
    chk, noneq = identity_checks(D)
    dec['identity_checks'] = chk
    dec['nonequivalent_start'] = noneq
    stream = {}
    ref_md5 = (D.get('ref') or {}).get('summary', {}) or {}
    ref_md5 = ref_md5.get('stream_md5')
    for c in comp:
        mm = D[c]['summary'].get('stream_md5')
        stream[c] = 'reference' if c == 'ref' else ('unchecked' if ref_md5 is None else ('match' if mm == ref_md5 else 'mismatch'))
    dec['stream'] = stream
    dec['pod_equivalence'] = {c: pod_equivalence(D[c]) for c in ('ref', 'ref2') if c in D and D[c]['steps']}
    # 손실 비 (vs ref, vs A)
    ratios = OrderedDict()
    for c in D:
        if c == 'ref' or 'ref' not in D or not lb.get(c) or not lb.get('ref'):
            continue
        ratios[c] = OrderedDict()
        for lo, hi in WINDOWS:
            ratios[c][f'{lo}-{hi}'] = ratio_window(lb[c], lb['ref'], lo, hi, blocks=(hi - lo + 1) >= 1000,
                                                   boot=(lo, hi) == W6)
    dec['loss_ratio_vs_ref'] = ratios
    ratios_A = OrderedDict()
    if 'A' in D and lb.get('A'):
        for c in D:
            if c in ('A',) or not lb.get(c):
                continue
            ratios_A[c] = {f'{lo}-{hi}': ratio_window(lb[c], lb['A'], lo, hi, blocks=(hi - lo + 1) >= 1000)
                           for lo, hi in WINDOWS}
    dec['loss_ratio_vs_A'] = ratios_A
    key6 = f'{W6[0]}-{W6[1]}'

    def R6(c):
        w = (ratios.get(c) or {}).get(key6)
        return None if w is None else w['R']

    # 타당성 (1차)
    V = OrderedDict()
    v1 = {c: inv_ok(D[c]['summary']) for c in comp}
    V['V1'] = dict(ok=all(v[0] for v in v1.values()) and bool(comp), detail={c: v[1] for c, v in v1.items()})
    e3 = {c: (e3_ok(D[c]['summary']) if c in D else None) for c in ('ref', 'A')}
    V['V2'] = dict(ok=all(c in comp and e3[c] for c in ('ref', 'A')), completed={c: c in comp for c in ('ref', 'A')}, E3=e3)
    ra = R6('A') if 'A' in comp else None
    V['V3'] = dict(ok=ra is not None and ra >= V3_MIN, R_A_W6=ra, threshold=V3_MIN)
    sa = [v for s, v in zip(*probe_series(D, 'A', lambda p: p['sat_v'])) if s >= 500] if 'A' in D else []
    V['V4'] = dict(ok=bool(sa) and min(sa) >= V4_MIN, sat_v_min_from500=min(sa) if sa else None, threshold=V4_MIN,
                   blocking=False)
    dec['validity'] = V
    v12 = V['V1']['ok'] and V['V2']['ok']
    eligible = v12 and V['V3']['ok']
    # 평가 기준 슬라이스·잡음 기준(2차 전용: ref2 는 완료·흐름 일치일 때만)
    ids = sorted(D['ref']['eval']) if ('ref' in D and D['ref']['eval']) else None
    v_ref = ssim_vec(D, 'ref', ids, False) if ids else None
    v_ref_a = ssim_vec(D, 'ref', ids, use_avg) if ids else None
    ref2_ok = 'ref2' in comp and stream.get('ref2') == 'match'
    r2 = R6('ref2') if ref2_ok else None
    tau = max(S3_TAU0, abs(r2)) if r2 is not None else S3_TAU0
    v_ref2_a = ssim_vec(D, 'ref2', ids, use_avg) if (ids and ref2_ok) else None
    m = max(S4_M0, abs(float(np.mean(v_ref2_a - v_ref_a)))) if (v_ref_a is not None and v_ref2_a is not None) else S4_M0
    dec['noise_floor'] = dict(used_by='amended (non-binding) rule only', R_ref2_W6=r2, tau=tau, m=m,
                              ref2_available=v_ref2_a is not None,
                              ref2_present_not_usable=('ref2' in D and not ref2_ok),
                              dssim_ref2_minus_ref=None if v_ref2_a is None else float(np.mean(v_ref2_a - v_ref_a)),
                              ssim_basis='mean of step-3500 and step-4000 snapshots' if use_avg else 'step 4000 only')
    # 조건별 판정
    crit = OrderedDict()
    for c in X_CONDS:
        if c not in comp:
            continue
        r = OrderedDict()
        tgt = D[c]['summary'].get('steps_target') or 4000
        # S1
        ps, pv = probe_series(D, c, lambda p: p['sat_v'])
        at = dict(zip(ps, pv))
        want = [s for s in S1_PROBE_STEPS if s <= tgt]
        late = [(s, at[s]) for s in want if s in at]
        s1 = bool(want) and len(late) == len(want) and all(v <= S1_MAX for _, v in late)
        xs, ys = step_series(D, c, 'sat_v_batch')
        sel = xs >= 1
        tm_i, tm = trailing_mean(ys[sel], 100)
        tm_steps = xs[sel][tm_i] if len(tm_i) else np.array([])
        mv = tm[(tm_steps >= 500) & (tm_steps <= 4000)] if len(tm) else np.array([])
        s1_mov = bool(mv.size) and float(mv.max()) <= S1_MAX
        r['S1'] = dict(ok=s1, probes_from500=late, missing_probes=[s for s in want if s not in at],
                       max_probe_all=max(pv) if pv else None, max_moving100_batch=float(mv.max()) if mv.size else None,
                       amended_ok=bool(s1 and s1_mov))
        # S2
        es = D[c]['eval_summary'] or {}
        tests = es.get('tests') or {}
        st_adj, st_sym = tests.get('adj'), tests.get('adj_sym')

        def s2ok(st):
            return bool(st and st['mean'] > 0 and st.get('p_sign') is not None and st['p_sign'] < S2_P)
        vc = [p.get('var_change') for p in D[c]['probes'] if p.get('var_change') is not None]
        r['S2'] = dict(ok=s2ok(st_adj), adj=st_adj, adj_sym=st_sym, adjprev=tests.get('adjprev'),
                       var_change_final=vc[-1] if vc else None, amended_ok=s2ok(st_sym))
        r['S2']['random_feature_note'] = bool((r['S2']['ok'] or r['S2']['amended_ok']) and vc and vc[-1] < 0.1)
        # S3
        w6 = (ratios.get(c) or {}).get(key6)
        if w6 is None or 'block_se' not in w6:
            r['S3'] = dict(ok=False, result='missing', amended_result='missing')
        else:
            ok3 = w6['R'] <= S3_TAU0
            a_ok, b_ok = w6['R'] <= tau, w6['upper90_t9'] <= S3_UPPER
            r['S3'] = dict(ok=ok3, result='pass' if ok3 else 'fail', R=w6['R'], threshold=S3_TAU0, se=w6['block_se'],
                           ci95_t9=w6['ci95_t9'], boot_ci95=w6.get('boot_ci95_mbb'),
                           blocks_x_lt_ref=w6.get('blocks_x_lt_ref'), n_blocks=w6.get('n_blocks'), sign_flag=w6['sign_flag'],
                           W5=(ratios.get(c) or {}).get(f'{W5[0]}-{W5[1]}'),
                           amended_result='pass' if (a_ok and b_ok) else ('unresolved' if a_ok else 'fail'),
                           tau=tau, upper90=w6['upper90_t9'])
        # S4 (1차: step 4000 E256 짝 부호 검정 / 2차: 스냅숏 평균 여유 m)
        vx = ssim_vec(D, c, ids, False) if ids else None
        if vx is None or v_ref is None:
            r['S4'] = dict(ok=False, missing=True, amended_ok=False)
        else:
            pr = paired(vx, v_ref)
            harm = bool(pr and pr['mean'] < 0 and pr['p_sign'] is not None and pr['p_sign'] < S4_P)
            vxa = ssim_vec(D, c, ids, use_avg)
            dma = float(np.mean(vxa - v_ref_a)) if (vxa is not None and v_ref_a is not None) else None
            r['S4'] = dict(ok=bool(pr) and not harm, dssim_vs_ref=pr['mean'] if pr else None, se=pr['se'] if pr else None,
                           p_sign=pr['p_sign'] if pr else None, n_pos=pr['n_pos'] if pr else None,
                           n_neg=pr['n_neg'] if pr else None, harm=harm,
                           amended_ok=dma is not None and dma >= -m, dssim_vs_ref_amended=dma, m=m)
        valid = stream.get(c) != 'mismatch' and c not in noneq and inv_ok(D[c]['summary'])[0]
        r['valid'] = valid
        r['candidate'] = bool(valid and r['S1']['ok'] and r['S2']['ok'] and r['S3']['ok'] and r['S4']['ok'])
        r['candidate_amended'] = bool(valid and r['S1']['amended_ok'] and r['S2']['amended_ok']
                                      and r['S3'].get('amended_result') == 'pass' and r['S4']['amended_ok'])
        if V['V3']['ok'] and ra and w6:
            r['f_X'] = 1 - (w6['R'] / ra)
        crit[c] = r
    dec['criteria'] = crit

    # 1차 선정 (구속): V1·V2·V3 충족 시에만
    pool = [c for c, r in crit.items() if r['candidate']]

    def tie_spec(c, top):
        p = paired(ssim_vec(D, c, ids, False), ssim_vec(D, top, ids, False))
        return p is None or p['p_sign'] is None or p['p_sign'] >= S4_P
    sel, tie = pick(pool, lambda c: crit[c]['S4']['dssim_vs_ref'], tie_spec) if v12 else (None, [])
    dec['candidates'] = pool if eligible else []
    dec['tie_set'] = tie if eligible else []
    dec['selected'] = sel if eligible else None
    if not v12:
        dec['blocked'] = 'V1 또는 V2 불충족 — 후보를 선언하지 않음'
    elif not V['V3']['ok']:
        dec['blocked'] = ('V3 불충족 — 4000 step 안에서 교란(A 의 학습 손실 불리)이 재현되지 않음: S3 는 정보가 없어 '
                          '50 epoch 후보를 선정하지 않음 (S1·S2 는 기술로만)')
        dec['provisional_ranking'] = dict(note='기술용 — 선정이 아님', candidates=pool, would_select=sel, tie_set=tie)
    # 2차 (비구속)
    pool_a = [c for c, r in crit.items() if r['candidate_amended']]
    sel_a, tie_a = pick(pool_a, lambda c: crit[c]['S4']['dssim_vs_ref_amended'],
                        lambda c, top: abs(crit[c]['S4']['dssim_vs_ref_amended']
                                           - crit[top]['S4']['dssim_vs_ref_amended']) < m) if v12 else (None, [])
    dec['amended'] = dict(binding=False, candidates=pool_a, selected=sel_a, tie_set=tie_a,
                          provisional=bool(sel_a and not (V['V3']['ok'] and V['V4']['ok'])))
    dec['selected_amended'] = sel_a
    # 해석 부류 (1차 결과 기준)
    P = []
    s3 = {c: crit[c]['S3'].get('result') for c in crit}
    if 'A' not in comp:
        P.append(('P0', 'A 가 완료되지 않아 V3 를 판정할 수 없다 — 후보 선정 없음, P1~P4·f_X 생략.'))
    elif not V['V3']['ok']:
        P.append(('P0', f'4000 step 안에서 교란이 재현되지 않았다(V3: R_A(W6) = {_pct(ra)} < +2%). S3 는 정보가 없으므로 '
                        'P1~P4·f_X 를 생략하고 S1·S2 는 기술로만 적는다. 50 epoch 후보 선정 없음.'))
    else:
        if s3.get('D') == 'pass':
            P.append(('P1', 'S3(D) 통과: 시퀀스 기여의 초기 크기와 함수공간 step 을 16.7 배 줄이면 불리가 사라진다 '
                            '(포화 패턴의 크기만의 효과로 돌리지 않는다 — Adam 은 기울기 크기에 불변이라 c 가 U-Net 쪽 시퀀스 가중치의 '
                            '초기 크기·이동 속도를 함께 줄인다).'))
        if s3.get('E') == 'pass':
            P.append(('P2', 'S3(E) 통과: U-Net 단독과 정확히 같은 시작점 + 학습 가능한 게이트가 불리를 없앤다'
                            + (' — S1(E) 실패이므로 게이트가 닫히는 방식으로' if not crit['E']['S1']['ok'] else '') + '.'))
        if s3.get('B') == 'pass' or s3.get('C') == 'pass':
            if 'D' in crit and s3.get('D') != 'pass':
                P.append(('P3', 'S3(B) 또는 S3(C) 통과, S3(D) 실패: 불리는 크기만이 아니라 포화 동역학과 묶여 있다.'))
            elif 'D' not in crit:
                P.append(('P3?', 'S3(B) 또는 S3(C) 통과 — D 가 완료되지 않아 크기 효과와 포화 동역학을 분리하지 못한다.'))
        if crit and all(v != 'pass' for v in s3.values()):
            P.append(('P4', '어느 조건도 S3 를 통과하지 못했다: 4000 step 안에서 이 개입들로는 불리가 사라지지 않는다.'))
    if 'A' in comp and not V['V4']['ok']:
        P.append(('V4', f'A 의 포화 재현 실패(V4: step ≥ 500 의 sat_v 최솟값 {_f(V["V4"]["sat_v_min_from500"], ".3f")} < 0.50) '
                        '— 보고만, 선정을 막지 않음.'))
    s2p = [c for c in crit if crit[c]['S2']['ok']]
    if s2p:
        P.append(('P5', f'S2 통과 조건 {s2p}: 교란을 바꾸면 슬라이스 고유 정보 사용이 나타난다'
                        + (' (단 var_change < 0.1 인 조건은 학습된 특징이 아니라 거의 무작위 초기 GRU 특징을 쓴 것)'
                           if any(crit[c]['S2']['random_feature_note'] for c in s2p) else '') + '.'))
    elif crit:
        P.append(('P5-', '모든 조건에서 S2 실패: "이 설계에서는 이득이 없다" 쪽 근거가 강해진다.'))
    if 'C' in crit and not crit['C']['S1']['ok']:
        early = [v for s, v in crit['C']['S1']['probes_from500'] if s <= 1000]
        later = [v for s, v in crit['C']['S1']['probes_from500'] if s > 1000]
        if later and max(later) > S1_MAX and (not early or max(early) <= S1_MAX):
            P.append(('P6', 'S1(C) 실패, step 1000 이후 sat_v 상승: warmup 뒤의 full LR 도 안전하지 않다 → 영구적 GRU LR 하향(B 형)이 필요.'))
    dec['interpretation'] = [dict(code=a, text=b) for a, b in P]
    dec['s2_multiplicity'] = dict(n_tested=len(crit), expected_false_pass_upper=0.05 * len(crit))
    return dec


# ------------------------------------------------------------------ 그림
def make_figs(D, dec, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                         'axes.edgecolor': '#a3a29d', 'lines.linewidth': 1.6, 'legend.frameon': False})
    present = [c for c in ORDER if c in D and D[c]['steps']]
    lb = {c: loss_by_b(D[c]['steps']) for c in present}
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(out, name), dpi=150)
        plt.close(fig)
        made.append(name)

    # 1) 100 배치 이동 손실 비
    if 'ref' in lb:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for c in present:
            if c == 'ref':
                continue
            bs, rr = moving_ratio(lb[c], lb['ref'])
            if bs is not None and len(bs):
                ax.plot(bs, 100 * rr, color=COLORS[c], ls=STYLES.get(c, '-'), label=LABELS[c])
        for y, lab in ((0, None), (100 * S3_TAU0, 'S3 tol. +0.5%'), (100 * V3_MIN, 'V3 +2%')):
            ax.axhline(y, color='#a3a29d', lw=0.8, ls='-' if y == 0 else ':')
            if lab:
                ax.text(0.995, y, lab, fontsize=7, color='#52514e', va='bottom', ha='right',
                        transform=ax.get_yaxis_transform())
        ax.axvspan(W6[0], W6[1], color='#f0efec', zorder=0)
        ax.set_xscale('log')
        ax.set_xlabel('batch position b (log)')
        ax.set_ylabel('100-batch moving loss ratio vs ref (%)')
        ax.set_title('Paired training-loss ratio vs U-Net only (same batches); shaded = decision window W6')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=2)
        save(fig, 'fig_loss_ratio.png')
    # 2) seq RMS
    gru = [c for c in present if c in GRU_CONDS]
    if gru:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
        for c in gru:
            for ax, key in zip(axs, ('seq_rms', 'seqin_rms')):
                xs, ys = step_series(D, c, key)
                if len(ys) > 25:
                    i, mv = trailing_mean(ys, 25)
                    ax.plot(xs[i], mv, color=COLORS[c], ls=STYLES.get(c, '-'), label=LABELS[c])
        for ax, t in zip(axs, ('raw GRU output RMS (train batch, 25-step mean)', 'U-Net seq-input RMS (after transform)')):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('step (log)')
            ax.set_title(t, fontsize=9)
        axs[0].legend(fontsize=7)
        save(fig, 'fig_seq_rms.png')
        # 3) 포화·공통 패턴·게이트 미분
        panels = [('sat_v (probe, |out_v|>0.99)', lambda p: p['sat_v']), ('sat_h (probe)', lambda p: p['sat_h']),
                  ('common_frac', lambda p: p['seq']['common_frac']), ('dep (input dependence)', lambda p: p['seq']['dep']),
                  ('gru_v dz = mean z(1-z)', lambda p: 0.5 * (p['gate_v']['f']['dz'] + p['gate_v']['b']['dz'])),
                  ('gru_v dn = mean(1-n^2)', lambda p: 0.5 * (p['gate_v']['f']['dn'] + p['gate_v']['b']['dn'])),
                  ('var_change (slice-varying part vs init)', lambda p: p['var_change']),
                  ('feat_change (vs init)', lambda p: p['feat_change'])]
        fig, axs = plt.subplots(2, 4, figsize=(12, 5.4))
        for ax, (t, fn) in zip(axs.ravel(), panels):
            for c in gru:
                xs, ys = probe_series(D, c, fn)
                if xs:
                    ax.plot(xs, ys, color=COLORS[c], ls=STYLES.get(c, '-'), marker='o', ms=3, label=LABELS[c])
            ax.set_xscale('symlog', linthresh=1)
            ax.set_title(t, fontsize=8)
            ax.set_xlabel('step')
        axs[0, 0].axhline(S1_MAX, color='#a3a29d', ls=':', lw=0.8)
        axs[0, 0].legend(fontsize=6)
        save(fig, 'fig_sat_common.png')
        # 4) probe 기능 시험
        fig, axs = plt.subplots(1, 4, figsize=(12, 3.4))
        for ax, t in zip(axs, ('zero', 'adj', 'cross', 'chanmean')):
            for c in gru:
                xs, ys = probe_series(D, c, lambda p, t=t: p['tests'][t]['mean'])
                if xs:
                    ax.plot(xs, ys, color=COLORS[c], ls=STYLES.get(c, '-'), marker='o', ms=3, label=LABELS[c])
            ax.axhline(0, color='#a3a29d', lw=0.8)
            ax.set_xscale('symlog', linthresh=1)
            ax.set_title(f'SSIM drop when seq -> {t} (P32)', fontsize=8)
            ax.set_xlabel('step')
        axs[0].legend(fontsize=6)
        save(fig, 'fig_probe_tests.png')
    # 5) mid-eval SSIM
    if present:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for c in present:
            xs, ys = probe_series(D, c, lambda p: p['mid_eval']['ssim'])
            if xs:
                ax.plot(xs, ys, color=COLORS[c], ls=STYLES.get(c, '-'), marker='o', ms=3, label=LABELS[c])
        ax.set_xscale('symlog', linthresh=1)
        ax.set_xlabel('step')
        ax.set_ylabel('SSIM (P32 probe slices, brain mask)')
        ax.set_title('Mid-training validation SSIM on the 32 probe slices')
        ax.legend(fontsize=7)
        save(fig, 'fig_mid_eval.png')
    # 6) gate |α|
    gate = [c for c in present if c in ('E', 'EC')]
    if gate:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for c in gate:
            for key, ls in (('gate_absmean', '-'), ('gate_absmax', '--')):
                xs, ys = step_series(D, c, key)
                if len(xs):
                    ax.plot(xs, ys, color=COLORS[c], ls=ls, label=f'{LABELS[c]} {key.replace("gate_abs", "|alpha| ")}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('step (log)')
        ax.set_title('Gate magnitude |alpha| (per-channel, 20)')
        ax.legend(fontsize=7)
        save(fig, 'fig_gate.png')
    return made


# ------------------------------------------------------------------ 표·문서
def _pct(v, nd=2):
    return '–' if v is None else f'{100 * v:+.{nd}f}%'


def _f(v, fmt='.4f'):
    return '–' if v is None else format(v, fmt)


def write_md(D, dec, figs, out, use_avg):
    L = []
    a = L.append
    a('# d-ETER 최적화 교란 스모크 결과 (2026-10-14)')
    a('')
    a(f'생성: {dec["time"]} · 완료 조건: {", ".join(dec["completed"]) or "없음"} · 판정 규칙은 실행 전에 고정 '
      f'(`v8_eter_pure/smoke_deter_opt_report.py` 머리말). composite 지표는 쓰지 않는다.')
    a('')
    a('**범위**: 0.49 epoch(4000 step)·cosine 감쇠 전 LR 에서의 초기 학습 속도 선별 시험이다. pod 에서는 손실 격차가 32 epoch 동안 '
      '+12% → +2% 로 줄었으므로, 이른 시점의 이득·불리가 50 epoch 까지 남는다는 보장은 없다. "후보 없음"도 단일 시드 선별 결과다.')
    a('')
    # 결론
    a('## 판정 (1차 규칙 = spec §10, 구속)')
    a('')
    if dec.get('blocked'):
        a(f'- **선정 없음**: {dec["blocked"]}.')
        pr = dec.get('provisional_ranking')
        if pr:
            a(f'  - 기술용 순위(선정 아님 — 이 결과로 50 epoch 재학습을 시작하지 않는다): 조건 통과 {pr["candidates"] or "없음"}'
              + (f', 규칙대로라면 {pr["would_select"]}' if pr['would_select'] else '') + '.')
    elif dec['selected']:
        a(f'- **선정: {dec["selected"]}** ({LABELS[dec["selected"]]}). 후보 {dec["candidates"]}, 동률 집합 {dec["tie_set"] or "-"}. '
          '다음 단계(사용자 결정): 이 변경으로 ETER-net 50 epoch 재학습(pod ≈5일) + U-Net 단독 두 번째 시드(감사 §6-6).')
    else:
        inval = [c for c, r in dec['criteria'].items() if not r['valid']]
        a('- **후보 없음**: 이 최적화 수정들과 무관하게 이 설계에서는 이득이 없다는 쪽이다(단일 시드 — "해롭다"는 뜻이 아니다).'
          + (f' 단 무효로 제외된 조건 {inval} 은 판정에 들어가지 않았다 — 원인을 설명한 뒤 다시 판정할 것.' if inval else ''))
    for p in dec['interpretation']:
        a(f'- {p["code"]}: {p["text"]}')
    am = dec.get('amended') or {}
    a(f'- 2차 규칙(비판 안, 비구속): 후보 {am.get("candidates") or "없음"}, 선정 {am.get("selected") or "없음"}'
      + (' (잠정 — V3·V4 미충족)' if am.get('provisional') else '') + '. 사용자 승인 전에는 결정에 쓰지 않는다.')
    a('')
    # 타당성
    V = dec['validity']
    a('## 타당성')
    a('')
    a('| 항목 | 결과 | 내용 |')
    a('|---|---|---|')
    a(f'| V1 불변식(I1~I9·E1·E2) | {"통과" if V["V1"]["ok"] else "실패"} | '
      + '; '.join(f'{c}: {",".join(w) if w else "ok"}' for c, w in V['V1']['detail'].items()) + ' |')
    a(f'| V2 ref·A 완료, E3(nan_skip ≤ 10, amp_floor ≤ 5) flag 없음 | {"통과" if V["V2"]["ok"] else "실패"} | '
      f'완료 {V["V2"]["completed"]}, E3 {V["V2"]["E3"]} |')
    a(f'| V3 A 의 불리 재현 R_A(W6) ≥ +2% (실패 시 선정 없음) | {"통과" if V["V3"]["ok"] else "실패"} | R_A(W6) = {_pct(V["V3"]["R_A_W6"])} |')
    a(f'| V4 A 의 포화 재현 sat_v ≥ 0.50 (step ≥ 500; 보고만) | {"통과" if V["V4"]["ok"] else "실패"} | 최솟값 {_f(V["V4"]["sat_v_min_from500"], ".3f")} |')
    nf = dec['noise_floor']
    a('')
    a(f'잡음 기준(ref2, 2차 규칙 전용): R_ref2(W6) = {_pct(nf["R_ref2_W6"])} → τ = {_pct(nf["tau"])}; '
      f'ΔSSIM(ref2 − ref) = {_f(nf["dssim_ref2_minus_ref"], "+.5f")} → m = {nf["m"]:.5f} ({nf["ssim_basis"]}).'
      + ('' if nf['ref2_available'] else (' ref2 가 있으나 미완료·흐름 불일치라 쓰지 않음 — 기본값(τ 0.5%, m 0.002).'
                                          if nf.get('ref2_present_not_usable') else ' ref2 없음 — 기본값(τ 0.5%, m 0.002).')))
    a('')
    a(f'흐름 동일성(E2 stream_md5 vs ref): {dec["stream"]}. 시작 동일성 검사(a~e): '
      + ('모두 통과' if not dec['nonequivalent_start'] else f'**불일치 {dec["nonequivalent_start"]}** — 해당 조건은 설명 전까지 후보 제외')
      + '.')
    for c, pe in dec['pod_equivalence'].items():
        a(f'- {c} ↔ pod plain 트레이너 step 의미 동일: {"예" if pe["equivalent"] else "아니오"} (amp_floor {pe["amp_floor"]}, '
          f'회복성 retry bpos {pe["retry_not_noop_bpos"] or "없음"}). pod 와의 비교는 train 목록·순열이 달라 통계적으로만 가능.')
    a('- 분리형 weight decay 3e-5 는 이 LR 범위(≤ 2e-4)에서 수치상 작동하지 않는다: 한 step 상대 감쇠 lr·wd ≤ 6e-9 가 fp32 반 ulp '
      '(≥ 2^-24)보다 작아 가중치가 그대로 반올림된다(pod 런도 같음). 따라서 decay/no-decay 구분은 이 스모크의 어느 조건에도 영향이 없고, '
      'ref 의 E5 는 "0 입력 slice 가 갱신되지 않았다"(값 0.0)를 확인하는 것이다. v8fix 레시피·방법 서술의 "weight decay 3e-5" 는 확인 필요.')
    a('')
    # 손실 비 표
    a('## 같은 배치의 학습 손실 비 R_X = Σloss_X / Σloss_ref − 1')
    a('')
    hdr = '| 조건 | ' + ' | '.join(f'{lo}–{hi}' for lo, hi in WINDOWS) + ' | W6 SE | W6 t9 95% CI | W6 MBB 95% CI | W6 블록 X<ref | W6 log비 평균/중앙 | clip |'
    a(hdr)
    a('|' + '---|' * (len(WINDOWS) + 7))
    for c, ws in dec['loss_ratio_vs_ref'].items():
        w6 = ws.get(f'{W6[0]}-{W6[1]}') or {}
        cw = ((D[c]['summary'] or {}).get('clip_windows') or {})
        clip = sum(v.get('clipped', 0) for v in cw.values()) if cw else None
        a(f'| {c}{"" if D[c]["completed"] else " (미완료)"} | ' + ' | '.join(_pct((ws.get(f'{lo}-{hi}') or {}).get('R')) for lo, hi in WINDOWS)
          + f' | {_pct(w6.get("block_se"))} | '
          + (f'[{_pct(w6["ci95_t9"][0])}, {_pct(w6["ci95_t9"][1])}]' if w6.get('ci95_t9') else '–') + ' | '
          + (f'[{_pct(w6["boot_ci95_mbb"][0])}, {_pct(w6["boot_ci95_mbb"][1])}]' if w6.get('boot_ci95_mbb') else '–')
          + f' | {w6.get("blocks_x_lt_ref", "–")}/{w6.get("n_blocks", "–")} | '
          + (f'{_pct(w6.get("logratio_mean"))} / {_pct(w6.get("logratio_median"))}' + (' ⚠부호 반대' if w6.get('sign_flag') else '')
             if w6 else '–') + f' | {"–" if clip is None else clip} |')
    if dec['loss_ratio_vs_A']:
        a('')
        a('A 대비 (같은 배치): ' + '; '.join(f'{c}: ' + ', '.join(_pct((ws.get(f"{lo}-{hi}") or {}).get("R"), 1) for lo, hi in WINDOWS)
                                            for c, ws in dec['loss_ratio_vs_A'].items()))
    a('')
    # 조건별 판정 (1차)
    a('## 조건별 기준 — 1차 규칙 (X ∈ B, C, D, E, EC)')
    a('')
    a('S1 = step 500~4000 probe 8 개 모두 sat_v ≤ 0.10 · S2 = E256 d_adj(다음 슬라이스 우선) 평균 > 0, 부호 p < 0.05 · '
      'S3 = R_X(W6) ≤ +0.5% · S4 = NOT(ΔSSIM(X − ref) < 0 이고 부호 p < 0.05), step 4000.')
    a('')
    a('| 조건 | S1 sat_v (step≥500 최대 / 전체 최대) | S2 d_adj 평균 (n+/n−, p) | var_change(끝) | S3 R(W6) [t9 95% CI] | S4 ΔSSIM vs ref (n+/n−, p) | f_X | 후보 |')
    a('|---|---|---|---|---|---|---|---|')
    for c, r in dec['criteria'].items():
        s1, s2, s3, s4 = r['S1'], r['S2'], r['S3'], r['S4']
        st = s2.get('adj') or {}
        late = [v for _, v in s1['probes_from500']]
        ci = s3.get('ci95_t9')
        a(f'| {c} | {"✔" if s1["ok"] else "✘"} {_f(max(late) if late else None, ".3f")} / {_f(s1["max_probe_all"], ".3f")}'
          + (f' (probe 누락 {s1["missing_probes"]})' if s1.get('missing_probes') else '') + ' | '
          f'{"✔" if s2["ok"] else "✘"} {_f(st.get("mean"), "+.5f")} ({st.get("n_pos", "–")}/{st.get("n_neg", "–")}, '
          f'p={_f(st.get("p_sign"), ".2g")}) | {_f(s2["var_change_final"], ".3f")}' + (' (무작위 특징)' if s2['random_feature_note'] else '')
          + f' | {"✔" if s3["ok"] else "✘"} {_pct(s3.get("R"))} ' + (f'[{_pct(ci[0])}, {_pct(ci[1])}]' if ci else '')
          + (' ⚠' if s3.get('sign_flag') else '')
          + f' | {"✔" if s4["ok"] else "✘"} {_f(s4.get("dssim_vs_ref"), "+.5f")} ({s4.get("n_pos", "–")}/{s4.get("n_neg", "–")}, '
            f'p={_f(s4.get("p_sign"), ".2g")}) | '
          + (_f(r.get('f_X'), '.2f') if 'f_X' in r else '–') + f' | {"**예**" if r["candidate"] else "아니오"}'
          + ('' if r['valid'] else ' (무효)') + ' |')
    a('')
    a('### 2차 규칙 (비판 안, 비구속)')
    a('')
    a('| 조건 | S1 + 배치 이동평균 (최대) | S2 d_adj_sym 평균 (n+/n−, p) | S3 결과 R ≤ τ, 상한90 ≤ 2% | S4 ΔSSIM(스냅숏 평균) ≥ −m | 후보 |')
    a('|---|---|---|---|---|---|')
    for c, r in dec['criteria'].items():
        s1, s2, s3, s4 = r['S1'], r['S2'], r['S3'], r['S4']
        st = s2.get('adj_sym') or {}
        a(f'| {c} | {"✔" if s1["amended_ok"] else "✘"} {_f(s1["max_moving100_batch"], ".3f")} | '
          f'{"✔" if s2["amended_ok"] else "✘"} {_f(st.get("mean"), "+.5f")} ({st.get("n_pos", "–")}/{st.get("n_neg", "–")}, '
          f'p={_f(st.get("p_sign"), ".2g")}) | {s3.get("amended_result")} τ={_pct(s3.get("tau"))} [{_pct(s3.get("upper90"))}] | '
          f'{"✔" if s4["amended_ok"] else "✘"} {_f(s4.get("dssim_vs_ref_amended"), "+.5f")} (m {_f(s4.get("m"), ".4f")}) | '
          f'{"예" if r["candidate_amended"] else "아니오"} |')
    mult = dec['s2_multiplicity']
    a('')
    a(f'S2 는 {mult["n_tested"]} 조건에서 시험했다 — 전역 귀무가설에서 기대되는 거짓 통과 수 상한 ≈ {mult["expected_false_pass_upper"]:.2f} '
      '(S1·S3 와 AND 로 묶이므로 형식적 보정은 하지 않음). S2 의 보조 열(양쪽 평균·이전 슬라이스만)은 decision.json.')
    if 'B' in dec['criteria']:
        a('B 의 S1 결과는 궤적에 따라 달라질 수 있다(설계 실측에서 seq RMS 가 step 37 에 0.11 까지 올랐다가 기울기 반전으로 물러남). '
          'B 가 포화하면 ×0.05 가 대안이다.')
    a('')
    # 최종 평가 짝 비교 (기술 통계)
    ids = sorted(D['ref']['eval']) if ('ref' in D and D['ref']['eval']) else None
    if ids:
        a('## 최종 평가 E256 — 짝 차이 (X − ref 의 SSIM 부호 검정은 1차 S4 에 쓰임, 나머지는 기술 통계)')
        a('')
        a('| 비교 | ΔSSIM 평균 ± SE [95% t] (n+/n−) | 부호 p | Wilcoxon p | ΔPSNR (dB) | ΔnMSE |')
        a('|---|---|---|---|---|---|')
        pairs = [(c, 'ref') for c in ORDER if c not in ('ref',) and c in D and D[c]['completed']]
        pairs += [(c, 'A') for c in X_CONDS if c in D and D[c]['completed'] and 'A' in D and D['A']['completed']]
        dec['paired_eval'] = OrderedDict()
        for x, y in pairs:
            res = {k: paired(metric_vec(D, x, ids, k), metric_vec(D, y, ids, k))
                   if metric_vec(D, x, ids, k) is not None and metric_vec(D, y, ids, k) is not None else None
                   for k in ('ssim', 'psnr', 'nmse')}
            dec['paired_eval'][f'{x}-{y}'] = res
            s = res['ssim']
            if s is None:
                continue
            a(f'| {x} − {y} | {s["mean"]:+.5f} ± {s["se"]:.5f} [{s["ci95"][0]:+.5f}, {s["ci95"][1]:+.5f}] ({s["n_pos"]}/{s["n_neg"]}) | '
              f'{_f(s["p_sign"], ".2g")} | {_f(s["p_wilcoxon"], ".2g")} | {_f((res["psnr"] or {}).get("mean"), "+.3f")} | '
              f'{_f((res["nmse"] or {}).get("mean"), "+.5f")} |')
        a('')
        a('### 시퀀스 출력 치환 시험 (E256, SSIM 하락 = 정상 − 치환)')
        a('')
        a('| 조건 | zero | adj_sym | adj(다음) | adjprev | cross | chanmean | common_frac | dep | wconst_D |')
        a('|---|---|---|---|---|---|---|---|---|---|')
        for c in GRU_CONDS:
            if c not in D or not D[c]['eval_summary']:
                continue
            es = D[c]['eval_summary']
            t = es.get('tests') or {}
            fmt = lambda s: '–' if not s else f'{s["mean"]:+.5f} ({s["n_pos"]}/{s["n_neg"]})'   # noqa: E731
            q = es.get('seq256') or {}
            a(f'| {c} | ' + ' | '.join(fmt(t.get(k)) for k in ('zero', 'adj_sym', 'adj', 'adjprev', 'cross', 'chanmean'))
              + f' | {_f(q.get("common_frac"), ".4f")} | {_f(q.get("dep"), ".3f")} | {_f(q.get("wconst_D"), ".3f")} |')
        a('')
    # 경고·flag
    fl = {c: (D[c]['summary'] or {}).get('flags') for c in D if (D[c]['summary'] or {}).get('flags')}
    if fl:
        a('## flag')
        a('')
        for c, f in fl.items():
            a(f'- {c}: {f}')
        a('')
    a('## 그림')
    a('')
    for f in figs:
        a(f'- `{f}`')
    a('')
    a('D 를 돌린 경우의 해석 주의: D 는 A 에서 U-Net 이 시퀀스 채널을 읽는 가중치(첫 conv·마지막 1×1 의 [:, :20])의 초기 크기와 '
      '함수공간 step 을 0.06 배로 한 것과 같다(W·(c·so) = (cW)·so, Adam 은 기울기 크기에 불변). 포화는 줄지 않는다(설계 실측: 더 깊음, '
      'sat_v 0.94). E 에서 GRU 는 세 영역으로 갱신된다: α|g| < eps 인 원소는 감쇠, 그보다 큰 원소는 Adam 이 α 를 상쇄(α 가 커지는 동안 '
      'v̂ 지연으로 step 이 √3·lr 까지 커짐), gru_v 의 fp16 역전파는 α ≈ 1e-3 에서 기울기가 subnormal 하한에 닿는다 '
      '(log.jsonl 의 so_grad·gru_v_wgrad_zero_frac 로 구분).')
    with open(os.path.join(out, 'summary.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     'results', 'smoke_deter_opt_2026-10-14'))
    ap.add_argument('--no-figs', action='store_true')
    a = ap.parse_args()
    out = os.path.abspath(a.out_dir)
    D = OrderedDict()
    for c in ORDER:
        d = load_cond(out, c)
        if d is not None and (d['steps'] or d['summary']):
            D[c] = d
    if not D:
        print(f'[report] {out} 에 조건 결과가 없음')
        return 0
    comp = [c for c in D if D[c]['completed']]
    use_avg = bool(comp) and all(D[c]['snap'] is not None for c in comp)
    dec = decide(D, use_avg)
    figs = [] if a.no_figs else make_figs(D, dec, out)
    write_md(D, dec, figs, out, use_avg)
    tmp = os.path.join(out, 'decision.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(dec, f, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(tmp, os.path.join(out, 'decision.json'))
    print(f'[report] 조건 {list(D)} (완료 {comp}) → {out}/summary.md, decision.json, 그림 {len(figs)} 개; '
          f'선정 {dec["selected"]}, 후보 {dec["candidates"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
