"""
eval_v8fix.py(Pod 런처가 학습 뒤 돌리는 자체 평가: per-slice CSV + 기능 시험)를 분석 계획 수정 3 의 fp32 재계산 규칙으로
실행하는 래퍼 (2026-10-22, 같은 날 기능 시험 규칙을 슬라이스 단위로 바꿈 — FV-8; 10-23 보강 — FVW-1 안 A: 다시 계산한
시퀀스 출력을 shuffle 입력으로 쓰는 짝 슬라이스도 표시, FVW-2: 대체된 판의 fp32 비유한도 결함, FVW-3: 슬라이스 확인 실패는 결함).

언제 쓰나: 계획 수정 3 §4 — 런처 결과(functional_<seq>.json)에 비유한 값이 있을 때만, 같은 512 슬라이스(--func-slices 512,
기본값)·같은 판정 기준으로 다시 계산한다. 비유한 값이 없으면 런처 결과를 그대로 쓰고 이 파일은 쓰지 않는다.
per-slice CSV 쪽(B3 교차 점검)은 이 파일이 아니라 공식 평가(eval_v8fix_official_fb.py)와 런처 CSV 의 대조로 다룬다(§3).
기능 시험 결과만 필요하면 --max-samples 1 로 per-slice 추론을 줄일 수 있다(기능 시험 색인은 len(ds) 와 --func-slices 로만
정해진다 — fb-verify 에서 --max-samples 1 과 6 의 functional JSON 이 seconds 외에 같음을 확인).

기록·종료 코드·시험 후크는 eval_v8fix_official_fb.py 와 같다(그 파일의 기계를 그대로 불러 씀). 다른 점:
  - per-slice CSV (eval_unet_only_v8.run_inference): 규칙의 단위는 공식 평가와 같이 슬라이스마다 모델 전체 forward 하나다.
  - 기능 시험 (smoke_recipe_fix.run_validation) — 슬라이스 단위 규칙:
      * 기능 시험은 모델 forward 를 거치지 않고 시퀀스 모듈과 U-Net 을 직접 부른다(unet_input -> PureETER_GRU._seq 또는
        SS2D.forward, model.unet(...), unet_with_zeroed -> model.unet(...)). 그래서 UNet_choh_skip.forward, SS2D.forward,
        PureETER_GRU._seq 도 감싸고, 이 호출들(시퀀스 모듈 호출, U-Net 호출)을 단위로 쓴다. 분해 점검에서 한 번 부르는 모델 전체
        forward 는 그릇으로만 다룬다(안쪽의 시퀀스 모듈·U-Net 호출이 단위) — 그래서 분해 경로와 모델 forward 가 같은 계산이 된다.
      * 각 단위는 run_validation 의 루프 변수 idx(그 순간 처리하는 슬라이스)에 속한다. 기능 시험 슬라이스(indices 의 원소)에
        속한 단위 하나라도 fp16 출력이 비유한이면 그 슬라이스를 '표시'한다.
      * 표시한 슬라이스의 U-Net 호출은 모두 fp16 으로 한 번 계산한 뒤 autocast 를 끄고 fp32 로 다시 계산해 그 출력을 돌려준다
        (같은 입력·같은 TF32 설정). 그래서 그 슬라이스의 정상 출력과 모든 대체 출력(deep4, deep34, zero, shuffle_cross,
        shuffle_adj)과 분해 점검이 같은 fp32 U-Net 계산에서 나온다. 표시하지 않은 슬라이스의 U-Net 호출은 그대로 둔다.
      * 시퀀스 모듈 호출은 호출 단위 규칙을 쓴다: 자기 fp16 출력이 비유한일 때만 fp32 로 다시 계산하고, 유한한 fp16 출력은 바꾸지
        않는다. 그 출력은 seq_store 에 저장되어 짝 슬라이스의 shuffle 입력으로도 쓰이므로, 유한한 출력을 fp32 로 바꾸면 입력이
        바뀌지 않아야 할 다른 슬라이스의 결과가 바뀐다.
      * 짝 슬라이스 표시 (FVW-1 안 A, 10-23 오케스트레이터 결정): seq_store 에 들어가는 시퀀스 출력(기능 시험 슬라이스의 것이든
        여분 짝 슬라이스의 것이든)을 위 규칙으로 fp32 로 다시 계산했으면, 그 출력을 shuffle_cross·shuffle_adj 입력으로 쓰는 기능 시험
        슬라이스(짝 슬라이스)도 표시한다. 런처에서는 그 슬라이스의 shuffle 출력이 비유한이었으므로(보충 메모 1 §1) 그 슬라이스의
        기능 시험 출력을 모두 fp32 U-Net 으로 다시 계산한다. 이 표시는 그 슬라이스의 shuffle U-Net 호출(run_validation 의 둘째
        루프)에서 일어나므로 늘 늦은 표시이고, 아래처럼 run_validation 을 다시 돌린다. 그래서 표시한 슬라이스의 시험 출력은 모두
        fp32 U-Net 에서 나오고(한 슬라이스 안에서 U-Net 정밀도가 섞이지 않는다), 그 입력인 시퀀스 출력(자기 것과 짝의 것)은 런처와
        같은 fp16 이거나, fp16 이 비유한이었던 것만 fp32 재계산이다. 표시하지 않은 슬라이스는 입력이 바뀌지 않으므로 비트까지 그대로다.
      * 한 슬라이스는 뒤의 호출(예: 둘째 루프의 shuffle 출력)에서 표시될 수 있다. 그때는 그 슬라이스의 앞선 U-Net 출력이 이미
        fp16 으로 쓰였으므로, 표시한 슬라이스 집합을 처음부터 알고 run_validation 을 처음부터 다시 돌린다(최대 MAX_PASSES 판).
        마지막 판의 결과를 원 스크립트에 돌려준다. 표시가 그 슬라이스의 첫 U-Net 호출이나 시퀀스 모듈 호출에서 일어나면 다시
        돌리지 않는다. 다시 돌린 판에서는 모든 단위의 fp16 출력이 첫 판과 비트까지 같은지 대조한다(repeat_check — 정수 체크섬).
      * 슬라이스 확인 (FVW-3, 10-23): 단위를 슬라이스·시험·짝에 붙이지 못하면 — 슬라이스 조회가 예외를 냄, idx 가 없음, 알 수
        없는 호출 줄, 기능 시험 슬라이스가 아닌 슬라이스의 U-Net 단위, 짝의 시퀀스 출력이 이번 판에 없는 shuffle 단위, 기능 시험
        중 run_validation 밖에서 일어난 감싼 호출 — 그 단위는 호출 단위 규칙으로 계산되지만 조용히 넘어가지 않는다:
        functional_slice_rule.lookup_failures 에 남고 평가는 결함(defect = true, defect_reasons, .DEFECT, 종료 코드 3)으로 끝난다.
      * 기록(JSON)의 records·recomputed_slices·counts 는 마지막 판(과 per-slice CSV)의 것이고, 앞 판의 기록은
        functional_slice_rule.superseded_records 에 둔다. counts.fp32_nonfinite 는 대체된 판의 fp32 비유한도 세고(FVW-2 — 하나라도
        있으면 결함, 종료 코드 3), counts.fp32_nonfinite_superseded 가 그 가운데 대체된 판의 수다. functional_slice_rule 에 판별
        요약, 표시한 슬라이스와 계기(reason = own_fp16_nonfinite 또는 partner_sequence_output_recomputed), 모든 기능 시험 출력이
        fp32 U-Net 에서 나온 기능 시험 슬라이스(recomputed_functional_slices), fp32 로 다시 계산한 저장 시퀀스 출력
        (recomputed_sequence_outputs — 여분 짝 슬라이스는 functional_test_slice = false)을 적는다. recomputed_slices 의 idx 에는
        여분 짝 슬라이스(기능 시험 슬라이스가 아님)의 시퀀스 재계산도 들어갈 수 있으므로, 다시 계산한 기능 시험 슬라이스의 수는
        recomputed_functional_slices 로 센다.
      * 호출 번호(call, 시험 후크 번호)는 한 번만 돌렸을 때의 번호다: per-slice CSV 의 모델 호출 0 … N−1(= idx) 다음에
        기능 시험의 단위가 run_validation 의 순서대로 이어지고, 다시 돌린 판도 같은 번호를 쓴다.
  - --seq gru 는 Pod 런처와 같게(eval_v8fix_grufp32.py) gru_precision.install_class_patch 로 gru_h 를 fp32 로 계산한 뒤 실행한다
    (GRU_FP32_MODULES, 기본 gru_h). gru_precision.py 는 Pod 브랜치(runpod/v8fix-pod3)의 v8_eter_pure/ 에 있다.
    --seq 는 줄여 쓰지 않는다(이 래퍼가 그 값을 보고 gru_h 패치를 정한다).
  - JSON 의 mode = launcher, by_phase 에 단계별 호출 수(eval_unet_only_v8.py:run_inference = per-slice CSV,
    smoke_recipe_fix.py:run_validation = 기능 시험의 마지막 판). 기능 시험 기록의 slice.site_src 는 run_validation 안의 호출 줄이고,
    test 는 그 호출이 만든 시험 출력(seq, normal, decomp_check, deep4, deep34, zero, shuffle_cross, shuffle_adj)이다.
  - 점검 실패(종료 코드 4)에 더하는 것: 기능 시험이 수렴하지 않음, repeat_check 불일치, run_validation 이 두 번 이상 불림,
    기능 시험에 U-Net(또는 시퀀스 모듈) 단위 호출이 없음, 결함이 아닌데 functional JSON 에 비유한 숫자가 남음.
    결함(종료 코드 3)에 더하는 것: 위 슬라이스 확인 실패(FVW-3), 대체된 판의 fp32 비유한(FVW-2).
  - --summary-only 는 모델 호출이 없어 기록을 쓰지 않는다. 그 실행은 summary_<seq>.md 를 다시 쓰므로, 그 뒤에는 앞 기록의
    outputs 에 적힌 summary md 의 sha256 이 파일과 맞지 않는다(CSV·functional JSON 의 sha256 은 그대로 맞는다).

실행 (인자는 eval_v8fix.py 와 같다. 출력 폴더는 런처 폴더가 아닌 새 폴더로):
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/eval_v8fix_fb.py --seq ss2d \\
      --ckpt logs/PureETER_SS2D_noDC_R4_brain384_v8fix_s3_fullval/pure_ss2d_best.pt \\
      --data-path ./fastMRI_data/multicoil_test_full --out-dir <새 폴더> --num-workers 4
"""

import os
import sys
import json
import argparse
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(_HERE, 'eval_v8fix.py')

_spec = importlib.util.spec_from_file_location('eval_v8fix_official_fb', os.path.join(_HERE, 'eval_v8fix_official_fb.py'))
FB = importlib.util.module_from_spec(_spec)
sys.modules['eval_v8fix_official_fb'] = FB
_spec.loader.exec_module(FB)

LAUNCHER_TARGETS = {
    'u_pure_eternet_unet': [('PureETER_UNET', 'forward')],
    'u_pure_eternet_gru': [('PureETER_GRU', 'forward'), ('PureETER_GRU', '_seq')],
    'u_pure_eternet_ss2d': [('PureETER_SS2D', 'forward')],
    'myUNet_DF': [('UNet_choh_skip', 'forward')],     # 교수님 원본 파일은 고치지 않음 — 실행 중 클래스 속성만 감쌈
    'ss2d': [('SS2D', 'forward')],                    # 프로젝트 공유 원본도 같음
}

FUNC_KEY = ('smoke_recipe_fix.py', 'run_validation')
FUNC_PHASE = 'smoke_recipe_fix.py:run_validation'
MODEL_LABELS = frozenset({'PureETER_UNET.forward', 'PureETER_GRU.forward', 'PureETER_SS2D.forward'})
SEQ_LABELS = frozenset({'PureETER_GRU._seq', 'SS2D.forward'})
UNET_LABELS = frozenset({'UNet_choh_skip.forward'})
SHUFFLE_TESTS = frozenset({'shuffle_cross', 'shuffle_adj'})
UNET_TESTS = frozenset({'normal', 'decomp_check', 'deep4', 'deep34', 'zero'}) | SHUFFLE_TESTS
SEQ_TESTS = frozenset({'seq', 'decomp_check'})          # 'seq' = unet_input 의 호출 = seq_store 에 저장되는 출력
MAX_PASSES = 4
MAX_LOOKUP_KEPT = 50                                    # 기록에 남기는 슬라이스 확인 실패 항목 수 (전체 수는 따로 셈)
FUNC_RULE = (
    'Functional test (smoke_recipe_fix.run_validation), slice-level rule. Units are the sequence-module calls '
    '(SS2D.forward, PureETER_GRU._seq) and the U-Net calls (UNet_choh_skip.forward); the whole-model forward of the '
    'decomposition check is a container whose inner calls are these units. Each unit belongs to the slice in '
    'run_validation\'s loop variable idx. A functional-test slice is flagged when any unit that belongs to it gives a '
    'non-finite fp16-autocast output. Every U-Net call of a flagged slice is computed in fp16 and then once more with '
    'autocast disabled (fp32, same inputs, TF32 flags untouched) and the fp32 output is returned, so its normal output and '
    'all substituted outputs (deep4, deep34, zero, shuffle_cross, shuffle_adj) and the decomposition check come from fp32 '
    'U-Net forwards; U-Net calls of other slices are returned unchanged. Sequence-module calls keep the per-call rule '
    '(recomputed only when their own fp16 output is non-finite; a finite fp16 sequence output is never replaced). '
    'Partner rule (FVW-1 option A): when a stored sequence output (the seq unit of a functional-test slice or of an extra '
    'partner slice, kept in seq_store) was recomputed in fp32, every functional-test slice whose shuffle_cross or '
    'shuffle_adj test uses that output is flagged too, at its shuffle U-Net call, so all of its functional-test outputs '
    'come from fp32 U-Net forwards as well; slices whose inputs did not change are returned unchanged. When a slice is '
    'flagged after some of its U-Net outputs were already used in fp16, run_validation is rerun from the start with the '
    'flagged set known (at most MAX_PASSES passes) and the last pass is returned; reruns check that every unit\'s fp16 '
    'output is bit-identical to the first pass. A unit that cannot be assigned to its slice, test or partner (FVW-3) is '
    'computed under the per-call rule, listed in lookup_failures and makes the run a defect (exit code 3); a non-finite '
    'fp32 recomputation in any pass, superseded passes included, is a defect too (FVW-2).')
CALL_NUMBERING = (
    'Logical call numbers as in a single pass: the per-slice CSV model calls 0..N-1 (= idx of per_slice_<seq>.csv), then '
    'the functional-test units in run_validation order; a rerun pass reuses the same numbers. records[].call, '
    'recomputed_slices[].call and the EVAL_FB_FORCE_NONFINITE(_FP32) test hooks use these numbers.')

FUNC = dict(invocations=0, active=False, indices=None, index_set=frozenset(), base_call=None, flagged={}, passes=[],
            cur=None, converged=None, ref=[], unexpected=[], lookup_failures=[], n_lookup_failures=0)

# run_validation 의 호출 줄 -> 시험 이름 (smoke_recipe_fix.py sha256 b1893836… 의 줄 내용; None = 지역 변수 t)
_TEST_BY_SRC = (('unet_input(', 'seq'), ('out_full = model(', 'decomp_check'), ('UNET_DEPTH - 1', 'deep4'),
                ('UNET_DEPTH - 2', 'deep34'), ('zeros_like(so)', 'zero'), ('seq_store[j]', None),
                ('model.unet(unet_in)', 'normal'))
_W = {}


def _seq_from_argv(argv):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--seq')
    ns, _ = p.parse_known_args(argv)
    if ns.seq is None and any(a.startswith('--se') and a != '--seq' and not a.startswith('--seq=') for a in argv):
        raise SystemExit('[eval_v8fix_fb] --seq 를 줄여 쓰지 말 것 (gru_h fp32 패치를 --seq 값으로 정함)')
    return ns.seq


def _gru_fp32():
    """eval_v8fix_grufp32.py 와 같은 순서: 경로 추가 -> gru_precision import -> install_class_patch."""
    sys.path.append(_HERE)
    sys.path.append(os.path.join(os.path.dirname(_HERE), 'models', 'pure_eternet'))
    import gru_precision                                  # noqa: E402 (Pod 브랜치 파일)
    names = tuple(n for n in os.environ.get('GRU_FP32_MODULES', 'gru_h').split(',') if n)
    gru_precision.install_class_patch(names)
    print(f'[eval_v8fix_fb] PureETER_GRU {names} -> fp32 (eval_v8fix_grufp32.py 와 같음)', flush=True)


# ------------------------------------------------------------------ 기능 시험: 슬라이스 단위 규칙
def _digest(x):
    """출력의 비트 체크섬 (GPU 정수 연산이라 합산 순서와 무관): (dtype, shape, Σ bits, Σ bits·w), w_i = i mod 251 + 1."""
    t = FB._first_tensor(x)
    if t is None:
        return None
    torch = FB.torch
    v = t.detach().contiguous().view(-1)
    if v.element_size() == 2:
        b = v.view(torch.int16).to(torch.int64) & 0xFFFF
    elif v.element_size() == 4:
        b = v.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    else:
        return (str(t.dtype), tuple(t.shape), float(v.double().sum()))
    k = (v.numel(), str(v.device))
    w = _W.get(k)
    if w is None:
        w = _W[k] = torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 251 + 1
    return (str(t.dtype).replace('torch.', ''), tuple(t.shape), int(b.sum()), int((b * w).sum()))


def _test_of(info, fr):
    """(시험 이름, 짝 슬라이스 idx) — run_validation 의 호출 줄과 지역 변수(t, j)로 정한다 (읽기만)."""
    src = info.get('site_src') or ''
    for frag, name in _TEST_BY_SRC:
        if frag in src:
            if name is None:
                L = fr.f_locals
                return L.get('t'), L.get('j')
            return name, None
    return None, None


def _repeat_check(P, ordinal, ident, dig):
    if P['pass'] == 1:
        FUNC['ref'].append((ident, dig))
        return
    rc = P['repeat']
    if ordinal >= len(FUNC['ref']):
        rc['extra_calls'] += 1
        return
    rid, rdig = FUNC['ref'][ordinal]
    rc['compared'] += 1
    if rid != ident or rdig != dig:
        rc['mismatched'] += 1
        if len(rc['examples']) < 10:
            rc['examples'].append(dict(call=FUNC['base_call'] + ordinal, pass1=dict(unit=rid, digest=rdig),
                                       this=dict(unit=ident, digest=dig)))


def _lookup_problems(info, idx, unit, test, partner, in_test, P):
    """FVW-3: 기능 시험 단위를 슬라이스·시험·짝에 붙이지 못한 이유들 (붙였으면 빈 목록)."""
    pr = []
    if info.get('error'):
        pr.append(f'slice lookup raised {info["error"]}')
    if type(idx) is not int:
        pr.append(f'no slice index (idx {idx!r})')
    if unit == 'unet':
        if test not in UNET_TESTS:
            pr.append(f'U-Net unit at an unknown call site (test {test!r}, line {info.get("site_src")!r})')
        if type(idx) is int and not in_test:
            pr.append(f'U-Net unit of slice {idx}, which is not a functional-test slice')
        if test in SHUFFLE_TESTS and (type(partner) is not int or partner not in P['seq_seen']):
            pr.append(f'{test} unit whose partner {partner!r} has no stored sequence output in this pass')
    elif test not in SEQ_TESTS:
        pr.append(f'sequence-module unit at an unknown call site (test {test!r}, line {info.get("site_src")!r})')
    return pr


def _note_lookup_failure(P, call, label, problems, info=None, test=None, partner=None):
    FUNC['n_lookup_failures'] += 1
    if P is not None:
        P['lookup_failures'] += 1
    item = {'call': call, 'pass': P['pass'] if P is not None else None, 'target': label, 'test': test,
            'partner_idx': partner, 'idx': (info or {}).get('idx'), 'site': (info or {}).get('site'),
            'site_src': (info or {}).get('site_src'), 'problems': problems}
    if len(FUNC['lookup_failures']) < MAX_LOOKUP_KEPT:
        FUNC['lookup_failures'].append(item)
        print(f'[fp32-fallback] 기능 시험 슬라이스 확인 실패 (결함, FVW-3): call {call} {label} — {"; ".join(problems)}'
              f'{" (이후 항목은 수만 셈)" if len(FUNC["lookup_failures"]) == MAX_LOOKUP_KEPT else ""}', flush=True)


def _flag(P, idx, info, reason, trigger):
    """기능 시험 슬라이스 idx 를 표시한다 (이미 표시했으면 그대로). 그 슬라이스의 U-Net 출력이 이미 fp16 으로 쓰였으면 늦은 표시."""
    if idx in FUNC['flagged']:
        return
    late = P['unet_fp16_used'].get(idx, 0) > 0
    FUNC['flagged'][idx] = dict(idx=idx, file=info.get('file'), slice_idx=info.get('slice_idx'), flagged_in_pass=P['pass'],
                                late=late, reason=reason, trigger=trigger)
    P['flagged_new'].append(idx)
    if late:
        P['late'].append(idx)


def _functional_unit(orig, self, args, kwargs, label, key, fr):
    P = FUNC['cur']
    ordinal = P['n']
    P['n'] += 1
    call = FUNC['base_call'] + ordinal                   # 논리 호출 번호 (다시 돌린 판도 같은 번호)
    FB.ST['calls'] += 1
    FB.retire_stale_once()
    info = FB._slice_info(key, fr) or {}
    idx = info.get('idx')
    test, partner = _test_of(info, fr)
    info['test'] = test
    if partner is not None:
        info['partner_idx'] = partner
    unit = 'seq' if label in SEQ_LABELS else 'unet'
    in_test = idx in FUNC['index_set']
    problems = _lookup_problems(info, idx, unit, test, partner, in_test, P)
    if problems:                                          # FVW-3: 조용히 호출 단위 규칙으로 바뀌지 않게 — 결함으로 기록
        _note_lookup_failure(P, call, label, problems, info, test, partner)
    stored_seq = unit == 'seq' and test == 'seq'          # seq_store[idx] 에 저장되어 짝 슬라이스의 shuffle 입력이 되는 출력
    st = P['phase']
    st['calls'] += 1
    P['calls_' + unit] += 1
    FB._TLS.depth = 1
    try:
        out = orig(self, *args, **kwargs)
        if stored_seq and type(idx) is int:
            P['seq_seen'][idx] = call                     # 이 판에서 seq_store[idx] 를 채운 호출 (짝 확인용)
        if not FB._autocast_on():
            return out
        FB.ST['calls_autocast'] += 1
        st['calls_autocast'] += 1
        P['calls_autocast'] += 1
        FB.note_first_call(self, label, out)
        _repeat_check(P, ordinal, (label, idx, info.get('site')), _digest(out))
        forced = call in FB.ST['force']
        ref16 = None
        if forced:
            ref16, out = out, FB._force_nonfinite(out)
        fin16 = FB._finite(out)
        if not fin16:
            P['fp16_nonfinite'] += 1
        if unit == 'unet' and in_test and test in SHUFFLE_TESTS and partner in P['seq_recomputed']:
            # FVW-1 안 A: 짝의 저장 시퀀스 출력을 fp32 로 다시 계산했다 — 런처에서는 이 shuffle 출력이 비유한이었다.
            # 이 출력도 비유한이면 두 계기가 같은 호출에서 겹친다; 기록에는 짝 쪽(근원)을 적는다.
            _flag(P, idx, info, 'partner_sequence_output_recomputed',
                  dict(call=call, target=label, test=test, site=info.get('site'), forced=forced, fp16=FB._stats(out),
                       partner_idx=partner, partner_seq_call=P['seq_recomputed'][partner]['call']))
        if not fin16 and in_test:
            _flag(P, idx, info, 'own_fp16_nonfinite',
                  dict(call=call, target=label, test=test, site=info.get('site'), forced=forced, fp16=FB._stats(out)))
        slice_rule = unit == 'unet' and in_test
        need32 = (not fin16) or (slice_rule and idx in FUNC['flagged'])
        if not need32:
            if slice_rule:
                P['unet_fp16_used'][idx] = P['unet_fp16_used'].get(idx, 0) + 1
            return out
        out32, rec = FB.recompute_fp32(orig, self, args, kwargs, call=call, label=label, phase=FUNC_PHASE, ph=st,
                                       slice_info=info, out16=out, forced=forced, ref16=ref16,
                                       extra={'pass': P['pass'], 'unit_rule': 'slice' if slice_rule else 'call',
                                              'test': test, 'fp16_finite': fin16})
        if stored_seq and type(idx) is int:
            P['seq_recomputed'][idx] = dict(call=call, functional_test_slice=in_test, forced=forced,
                                            fp32_finite=rec['fp32']['finite'])
        return out32
    finally:
        FB._TLS.depth = 0


def _unit_handler(orig, self, args, kwargs, label, key, fr):
    """eval_v8fix_official_fb 의 감싼 호출이 묻는다: 기능 시험 단계면 슬라이스 단위 규칙, 아니면 기본 규칙(FB.DEFAULT)."""
    if FUNC['active'] and key != FUNC_KEY:                # FVW-3: 기능 시험 중인데 run_validation 프레임을 찾지 못한 감싼 호출
        _note_lookup_failure(FUNC['cur'], FB.ST['calls'], label,
                             [f'wrapped call during the functional test outside run_validation (frame {key!r})'])
    if not FUNC['active'] or key != FUNC_KEY:
        return FB.DEFAULT                                 # per-slice CSV (run_inference) 등: 모델 전체 forward 단위
    if label in MODEL_LABELS:
        return orig(self, *args, **kwargs)                # 분해 점검의 모델 전체 forward: 그릇 — 안쪽 호출이 단위
    if label in SEQ_LABELS or label in UNET_LABELS:
        return _functional_unit(orig, self, args, kwargs, label, key, fr)
    return FB.DEFAULT


def _start_pass(p):
    if FUNC['passes']:                                    # 앞 판의 기능 시험 기록과 호출 수를 '대체됨'으로 옮긴다
        prev = FUNC['passes'][-1]
        old = [r for r in FB.ST['records'] if r.get('pass') == prev['pass']]
        FB.ST['records'][:] = [r for r in FB.ST['records'] if r.get('pass') != prev['pass']]
        FB.ST['superseded']['records'].extend(old)
        FB.ST['superseded']['calls'] += prev['n']
        FB.ST['superseded']['calls_autocast'] += prev['calls_autocast']
    P = {'pass': p, 'started_utc': FB._utc(), 'finished_utc': None, 'n': 0, 'calls_autocast': 0, 'calls_seq': 0,
         'calls_unet': 0, 'fp16_nonfinite': 0, 'flagged_at_start': sorted(FUNC['flagged']), 'flagged_new': [],
         'late': [], 'unet_fp16_used': {}, 'seq_seen': {}, 'seq_recomputed': {}, 'lookup_failures': 0,
         'repeat': dict(compared=0, mismatched=0, extra_calls=0, examples=[]) if p > 1 else None,
         'phase': dict(calls=0, calls_autocast=0, recomputed=0, fp32_nonfinite=0)}
    FUNC['passes'].append(P)
    FUNC['cur'] = P
    FB.ST['by_phase'][FUNC_PHASE] = P['phase']           # by_phase 의 기능 시험 칸 = 마지막 판
    return P


def _end_pass(P):
    P['finished_utc'] = FB._utc()
    if P['repeat'] is not None and P['n'] != len(FUNC['ref']):
        P['repeat']['call_count_differs'] = dict(pass1=len(FUNC['ref']), this=P['n'])
    rep = (f', 첫 판과 대조 {P["repeat"]["compared"]} 중 불일치 {P["repeat"]["mismatched"]}' if P['repeat'] else '')
    print(f'[fp32-fallback] 기능 시험 pass {P["pass"]}: 단위 호출 {P["n"]} (시퀀스 모듈 {P["calls_seq"]}, U-Net '
          f'{P["calls_unet"]}), fp16 비유한 {P["fp16_nonfinite"]}, 새로 표시한 슬라이스 {P["flagged_new"]} '
          f'(늦은 표시 {P["late"]}), fp32 로 다시 계산한 저장 시퀀스 출력 {sorted(P["seq_recomputed"])}, '
          f'재계산 {P["phase"]["recomputed"]}{rep}'
          f'{", 슬라이스 확인 실패 " + str(P["lookup_failures"]) if P["lookup_failures"] else ""}', flush=True)


def _drive(orig, args, kwargs):
    """run_validation 을 대신 부른다: 슬라이스 단위 규칙으로 한 번 돌리고, 늦게 표시한 슬라이스가 있으면 처음부터 다시."""
    import inspect                                        # 원 스크립트가 모든 import 를 마친 뒤
    FUNC['invocations'] += 1
    if FUNC['invocations'] > 1 or FUNC['active']:
        FUNC['unexpected'].append(f'run_validation call #{FUNC["invocations"]}: only one call is expected; '
                                  f'this call ran without the slice-level rule')
        return orig(*args, **kwargs)
    ba = inspect.signature(orig).bind(*args, **kwargs)
    indices = list(ba.arguments['indices'])
    ba.arguments['indices'] = indices
    FUNC.update(indices=indices, index_set=frozenset(indices), base_call=FB.ST['calls'], flagged={}, passes=[],
                ref=[], converged=False, lookup_failures=[], n_lookup_failures=0, active=True)
    try:
        res = None
        for p in range(1, MAX_PASSES + 1):
            P = _start_pass(p)
            res = orig(*ba.args, **ba.kwargs)
            _end_pass(P)
            if not P['late']:
                FUNC['converged'] = True
                break
            if p < MAX_PASSES:
                print(f'[fp32-fallback] 기능 시험: 슬라이스 {P["late"]} 는 앞선 U-Net 출력을 fp16 으로 쓴 뒤 표시됨 -> '
                      f'표시한 슬라이스 {sorted(FUNC["flagged"])} 를 처음부터 fp32 U-Net 으로, run_validation 을 다시 '
                      f'실행 (pass {p + 1})', flush=True)
        return res
    finally:
        FUNC['active'] = False


def _patch_run_validation(module):
    """smoke_recipe_fix.run_validation(모듈 함수)을 _drive 로 감싼다. eval_v8fix.main 은 R.run_validation 을 부른다."""
    orig = module.run_validation
    if getattr(orig, '_fp32_fallback_orig', None) is not None:
        return False

    def run_validation(*args, **kwargs):
        return _drive(orig, args, kwargs)

    for k in ('__module__', '__name__', '__qualname__', '__doc__'):
        try:
            setattr(run_validation, k, getattr(orig, k))
        except (AttributeError, TypeError):
            pass
    run_validation.__wrapped__ = orig
    run_validation._fp32_fallback_orig = orig
    module.run_validation = run_validation
    return True


def _seq_list(d):
    return [dict(idx=i, **v) for i, v in sorted(d.items())]


def _finalize(completed, ns, outputs, out_dir, seq, defect):
    """eval_v8fix_official_fb.finish 가 기록을 쓰기 전에 부른다: 기능 시험 요약·대조·점검 실패·결함(FVW-3).
    defect = 마지막 판(과 per-slice 단계)에 fp32 비유한이 있음 — functional JSON 에 남은 비유한 숫자를 설명하는지 볼 때만 쓴다."""
    passes = FUNC['passes']
    final = passes[-1] if passes else None
    flagged = FUNC['flagged']
    keys = dict(functional_slice_rule=dict(
        rule=FUNC_RULE, call_numbering=CALL_NUMBERING, max_passes=MAX_PASSES, invocations=FUNC['invocations'],
        n_functional_slices=len(FUNC['indices'] or []), n_passes=len(passes),
        final_pass=final['pass'] if final else None, converged=FUNC['converged'] if passes else None,
        flagged_slices=[flagged[i] for i in sorted(flagged)],
        recomputed_functional_slices=sorted(flagged),     # 기능 시험 출력이 모두 fp32 U-Net 에서 나온 기능 시험 슬라이스
        partner_flagged_slices=sorted(i for i, f in flagged.items() if f.get('reason') == 'partner_sequence_output_recomputed'),
        recomputed_sequence_outputs=_seq_list(final['seq_recomputed']) if final else [],
        lookup_failures=dict(n=FUNC['n_lookup_failures'], kept=FUNC['lookup_failures'], max_kept=MAX_LOOKUP_KEPT),
        passes=[{k: (_seq_list(v) if k == 'seq_recomputed' else v) for k, v in P.items()
                 if k not in ('unet_fp16_used', 'seq_seen')} for P in passes],
        superseded_records=FB.ST['superseded']['records'], unexpected=FUNC['unexpected']))
    defects = []
    if FUNC['n_lookup_failures']:                         # FVW-3: 슬라이스 확인 실패는 결함 (종료 코드 3)
        f0 = FUNC['lookup_failures'][0] if FUNC['lookup_failures'] else {}
        defects.append(f'functional-test slice lookup failed for {FUNC["n_lookup_failures"]} wrapped call(s), which '
                       f'therefore fell back to the per-call rule (FVW-3); first: call {f0.get("call")} {f0.get("target")}: '
                       f'{"; ".join(f0.get("problems") or [])}')
    failures, xc = [], {}
    if not completed:
        return dict(keys=keys, failures=failures, crosscheck=xc, defects=defects)
    expected = (getattr(ns, 'func_slices', 0) or 0) > 0
    name = f'functional_{seq}.json'
    o = outputs.get(name)
    n_json = idx_json = None
    if expected and o and o['written_this_run']:
        with open(os.path.join(out_dir, name)) as f:
            fj = json.load(f)
        n_json, idx_json = fj.get('n'), fj.get('indices')
    if expected:
        match = bool(FUNC['invocations'] == 1 and n_json is not None and n_json == len(FUNC['indices'] or [])
                     and idx_json == FUNC['indices'])
    else:
        match = FUNC['invocations'] == 0
    xc = dict(functional_expected=expected, functional_json_n=n_json, functional_slices_driven=len(FUNC['indices'] or []),
              run_validation_invocations=FUNC['invocations'], functional_match=match)
    if passes:
        if not FUNC['converged']:
            failures.append(f'functional slice rule did not converge in {MAX_PASSES} passes: late flags {final["late"]}')
        mism = sum(P['repeat']['mismatched'] for P in passes if P['repeat'])
        diffn = [P['pass'] for P in passes if P['repeat'] and (P['repeat'].get('call_count_differs')
                                                               or P['repeat']['extra_calls'])]
        if mism or diffn:
            failures.append(f'functional rerun did not reproduce the first pass: {mism} unit outputs differ, '
                            f'call sequence differs in passes {diffn}')
        if final['calls_unet'] == 0 or (seq in ('gru', 'ss2d') and final['calls_seq'] == 0):
            failures.append(f'functional test had no wrapped U-Net or sequence-module unit (U-Net {final["calls_unet"]}, '
                            f'sequence module {final["calls_seq"]}): the slice-level rule was not applied')
    if FUNC['unexpected']:
        failures.append('; '.join(FUNC['unexpected']))
    if expected and o and (o.get('nonfinite_values') or 0) > 0 and not defect:
        failures.append(f'{name} still has {o["nonfinite_values"]} non-finite numbers although no fp32 output was non-finite')
    return dict(keys=keys, failures=failures, crosscheck=xc, defects=defects)


def main():
    seq = _seq_from_argv(sys.argv[1:])
    FB.run(TARGET, LAUNCHER_TARGETS, mode='launcher', wrapper_files=[__file__, FB.__file__],
           pre=_gru_fp32 if seq == 'gru' else None, unit_handler=_unit_handler,
           post_import={'smoke_recipe_fix': [_patch_run_validation]}, finalize=_finalize)


if __name__ == '__main__':
    main()
