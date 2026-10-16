"""A2(ETER-net 회차 간 변동) 비교 (2026-10-15) — CPU 전용, torch 미사용. smoke_deter_opt_report.py 의 통계 함수를 그대로 쓴다.

배경: d-ETER 스모크 8 조건(10-15 14:00 UTC 완료)에서 V3(A 의 학습 손실 열세 R_A(W6) ≥ +2 %)가 재현되지 않았다(−1.30 %).
같은 구간 pod ETER-net 은 U-Net 단독보다 +10.4 %(wandb, 같은 배치 흐름) — docs/smoke_deter_opt_report_2026-10-15.md §3.
A2 = A 에서 GRU 초기값만 seed 2 (smoke_deter_opt.py 설계 결정 15, 사용자 10-15 승인). U-Net 초기값·배치 흐름·레시피는 A 와 같다.

판정 규칙 (A2 실행 전에 고정, 10-15):
  지표: R_A2(W6) = Σloss_A2 / Σloss_ref − 1, 배치 3001~4000 (같은 배치), 블록 10 개 t_9 95 % CI (report 의 ratio_window).
  타당성: A2 completed, I1(U-Net 초기값 = A)·E2(stream_md5 = ref) 통과, E3 flag 없음(nan_skip ≤ 10, amp_floor ≤ 5).
  (i)   R_A2(W6) ≥ +2 % 이고 CI 하한 > 0      → '회차 의존': 로컬에서도 GRU 초기값에 따라 열세가 나타난다.
                                                  pod 불일치는 회차 간 변동으로 설명 가능 → 조건 비교는 다중 시드로 다시 설계.
  (ii)  R_A2(W6) ≤ +0.5 % 이고 CI 상한 < +2 % → '로컬 재현 없음': 두 회차 모두 열세 없음 → pod 쪽 원인(데이터·GPU)을 조사.
  (iii) 그 밖                                  → '보류'.
  같이 보고(기술): 구간별 R_A2·R_A, R_A2 − R_A(같은 배치 A2/A 비), sat_v 궤적, step 4000 E256 SSIM(A2 − ref, A2 − A).
  SSIM 은 판정에 쓰지 않는다 (ref2 − ref = +0.0081 이 조건 간 차보다 큼).
출력: <out>/a2_compare.md · a2_compare.json
사용: CUDA_VISIBLE_DEVICES="" python v8_eter_pure/smoke_deter_opt_a2_compare.py --out-dir results/smoke_deter_opt_2026-10-14
"""

import os
import sys
import json
import argparse
import datetime
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_deter_opt_report as RP          # noqa: E402

R_HI, R_LO, R_UP = 0.020, 0.005, 0.020
E3_NAN_MAX, E3_FLOOR_MAX = RP.E3_NAN_MAX, RP.E3_FLOOR_MAX
POD_JSON = 'pod_wandb_train_loss_steps1-16300.json'
POD_U = 'PureETER_UNET_noDC_R4_brain384_v8fix_s1_fullval_pod2'
POD_G = 'PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval_pod3f'


def pct(v, fmt='+.2f'):
    return '–' if v is None else f'{v * 100:{fmt}} %'


def sat_traj(d):
    return OrderedDict((int(p['step']), float(p['sat_v'])) for p in d['probes'] if p.get('sat_v') is not None)


def eval_vec(d, ids):
    return np.array([d['eval'][i]['ssim'] for i in ids], float)


def decide(r6, valid):
    if not valid['ok']:
        return 'invalid', '타당성 실패 — 판정하지 않음: ' + '; '.join(valid['why'])
    R, ci = r6['R'], r6.get('ci95_t9')
    if R >= R_HI and ci and ci[0] > 0:
        return 'run_dependent', '(i) 회차 의존 — 로컬에서도 GRU 초기값에 따라 손실 열세가 나타난다'
    if R <= R_LO and ci and ci[1] < R_UP:
        return 'not_reproduced', '(ii) 로컬 재현 없음 — 두 회차 모두 열세 없음, pod 쪽 원인(데이터·GPU) 조사'
    return 'unresolved', '(iii) 보류'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results',
                                                      'smoke_deter_opt_2026-10-14'))
    out = os.path.abspath(ap.parse_args().out_dir)
    D = {c: RP.load_cond(out, c) for c in ('ref', 'A', 'A2')}
    miss = [c for c, d in D.items() if d is None or not d['completed']]
    if miss:
        print(f'[a2] 미완료 조건 {miss} — 중단', flush=True)
        return 1
    L = {c: RP.loss_by_b(D[c]['steps']) for c in D}

    s2 = D['A2']['summary']
    inv = s2.get('invariants', {})
    end = s2.get('end_checks', {})
    cnt = s2.get('counters', {})
    why = []
    if not inv.get('I1', {}).get('ok'):
        why.append('I1')
    if end.get('E2', {}).get('vs_ref') != 'match':
        why.append(f"E2 stream {end.get('E2', {}).get('vs_ref')}")
    if cnt.get('nan_skip', 0) > E3_NAN_MAX or cnt.get('amp_floor', 0) > E3_FLOOR_MAX:
        why.append(f"E3 nan_skip={cnt.get('nan_skip')} amp_floor={cnt.get('amp_floor')}")
    if not inv.get('I2', {}).get('differs_from_seed1'):
        why.append('GRU 초기값이 seed 1 과 같음')
    valid = dict(ok=not why, why=why)

    rows = OrderedDict()
    for lo, hi in RP.WINDOWS:
        rows[f'{lo}-{hi}'] = OrderedDict(
            A2_vs_ref=RP.ratio_window(L['A2'], L['ref'], lo, hi, blocks=(lo, hi) in (RP.W5, RP.W6)),
            A_vs_ref=RP.ratio_window(L['A'], L['ref'], lo, hi, blocks=(lo, hi) in (RP.W5, RP.W6)),
            A2_vs_A=RP.ratio_window(L['A2'], L['A'], lo, hi, blocks=(lo, hi) in (RP.W5, RP.W6)))
    r6 = rows['3001-4000']['A2_vs_ref']
    verdict, text = decide(r6, valid)

    pod = None
    pj = os.path.join(out, POD_JSON)
    if os.path.exists(pj):
        pw = json.load(open(pj))
        pu, pg = dict((int(s), v) for s, v in pw[POD_U]), dict((int(s), v) for s, v in pw[POD_G])
        pod = OrderedDict()
        for lo, hi in RP.WINDOWS:
            ss = [s for s in range(lo, hi + 1) if s % 20 == 0 and s in pu and s in pg]
            loc = lambda X: (sum(L[X][s] for s in ss if s in L[X] and s in L['ref'])        # noqa: E731
                             / sum(L['ref'][s] for s in ss if s in L[X] and s in L['ref']) - 1)
            pod[f'{lo}-{hi}'] = dict(pod=sum(pg[s] for s in ss) / sum(pu[s] for s in ss) - 1, A20=loc('A'), A220=loc('A2'))

    ids = [i for i in D['ref']['eval'] if i in D['A']['eval'] and i in D['A2']['eval']]
    v = {c: eval_vec(D[c], ids) for c in D}
    ev = OrderedDict(n=len(ids), ssim={c: float(v[c].mean()) for c in D},
                     A2_minus_ref=RP.paired(v['A2'], v['ref']), A_minus_ref=RP.paired(v['A'], v['ref']),
                     A2_minus_A=RP.paired(v['A2'], v['A']))
    sat = {c: sat_traj(D[c]) for c in ('A', 'A2')}

    res = OrderedDict(time=datetime.datetime.now().isoformat(timespec='seconds'), verdict=verdict, verdict_text=text,
                      validity=valid, rule=dict(R_hi=R_HI, R_lo=R_LO, R_upper=R_UP, window='3001-4000'),
                      gru_ck_A2=inv.get('I2', {}).get('gru_ck'), counters_A2=cnt, windows=rows, pod_vs_local_every20=pod,
                      eval=ev, sat_v=sat)
    with open(os.path.join(out, 'a2_compare.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)

    m = []
    a = m.append
    a('# A2 — ETER-net 회차 간 변동 확인 (d-ETER 스모크 추가 조건)\n')
    a(f'생성: {res["time"]} · 규칙은 A2 실행 전에 고정 (`v8_eter_pure/smoke_deter_opt_a2_compare.py` 머리말). '
      'A2 = A 에서 GRU 초기값만 seed 2 (U-Net 초기값·배치 흐름·레시피 동일).\n')
    a(f'## 판정: **{text}**\n')
    a(f'- R_A2(3001–4000) = {pct(r6["R"])} [t9 95 % CI {pct(r6["ci95_t9"][0])}, {pct(r6["ci95_t9"][1])}] '
      f'(블록 A2<ref {r6["blocks_x_lt_ref"]}/10). 기준: (i) ≥ +2 % 이고 하한 > 0 / (ii) ≤ +0.5 % 이고 상한 < +2 %.')
    a(f'- 타당성: {"통과" if valid["ok"] else "실패 " + str(valid["why"])} · A2 카운터 {json.dumps(cnt)}')
    a(f'- GRU 초기 checksum: A2 {res["gru_ck_A2"]} (seed 1 = {RP.__dict__.get("EXPECT_GRU", "-2.7356114523e+03")})\n')
    a('## 같은 배치 학습 손실 비 (vs ref)\n')
    a('| 구간 | R_A2 | R_A | A2 / A − 1 |')
    a('|---|---|---|---|')
    for k, r in rows.items():
        ci = lambda x: (f' [{pct(x["ci95_t9"][0])}, {pct(x["ci95_t9"][1])}]' if x and 'ci95_t9' in x else '')   # noqa: E731
        a(f'| {k} | {pct(r["A2_vs_ref"]["R"])}{ci(r["A2_vs_ref"])} | {pct(r["A_vs_ref"]["R"])}{ci(r["A_vs_ref"])} '
          f'| {pct(r["A2_vs_A"]["R"])}{ci(r["A2_vs_A"])} |')
    if pod:
        a('\n## pod 와 비교 (20 step 마다 한 점, 같은 위치)\n')
        a('| 구간 | pod ETER-net / U-Net − 1 | 로컬 A | 로컬 A2 |')
        a('|---|---|---|---|')
        for k, r in pod.items():
            a(f'| {k} | {pct(r["pod"])} | {pct(r["A20"])} | {pct(r["A220"])} |')
    a('\n## GRU 포화 sat_v (probe)\n')
    steps = sorted(set(sat['A']) | set(sat['A2']))
    a('| step | ' + ' | '.join(str(s) for s in steps) + ' |')
    a('|---|' + '---|' * len(steps))
    for c in ('A', 'A2'):
        a(f'| {c} | ' + ' | '.join(f'{sat[c].get(s, float("nan")):.3f}' for s in steps) + ' |')
    a(f'\n## step 4000 E256 SSIM (기술 — 판정에 쓰지 않음; ref2 − ref = +0.0081)\n')
    a(f'ref {ev["ssim"]["ref"]:.4f} · A {ev["ssim"]["A"]:.4f} · A2 {ev["ssim"]["A2"]:.4f}\n')
    for k in ('A2_minus_ref', 'A_minus_ref', 'A2_minus_A'):
        p = ev[k]
        a(f'- {k.replace("_minus_", " − ")}: {p["mean"]:+.5f} ± {p["se"]:.5f} ({p["n_pos"]}/{p["n_neg"]})')
    with open(os.path.join(out, 'a2_compare.md'), 'w') as f:
        f.write('\n'.join(m) + '\n')
    print(f'[a2] {verdict}: {text} → {out}/a2_compare.md', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
