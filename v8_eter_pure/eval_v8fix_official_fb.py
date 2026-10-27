"""
eval_v8fix_official.py 를 분석 계획 수정 3 의 fp32 재계산 규칙으로 실행하는 얇은 래퍼 (2026-10-22, 같은 날 보강 FV-1~5,
10-23 보강 FVW-2·FVW-4 — 런처 래퍼의 FVW-1·FVW-3 이 쓰는 결함 기록 포함).

근거: paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/analysis_plan_seed23_amend3.md §2 (사용자 결정
'fp32 재계산'). SS2D(controlled) 시드 3 이 epoch 32 부터 fp16 순전파에서 넘쳤다(같은 검증 배치가 fp16 에서는 실패, fp32 에서는 성공).

규칙
  - 슬라이스마다 지금처럼 fp16 autocast 로 추론한다. 출력에 비유한 값(NaN, ±inf)이 하나라도 있으면 그 호출만 한 번 더
    계산한다: 같은 모듈 인스턴스(= 같은 체크포인트), 같은 입력 텐서, 같은 TF32 설정(건드리지 않음), autocast 끔, 부동소수
    입력은 float32 로 올림. 그 fp32 출력을 원 스크립트에 돌려주므로 brain mask 지표와 공식 정의 지표가 모두 그 출력으로 계산된다.
  - fp16 출력이 유한하면 아무것도 바꾸지 않는다(같은 텐서를 그대로 돌려준다).
  - fp32 출력도 비유한이면 기록하고 그대로 돌려준다. 원 스크립트는 끝까지 돈다. 래퍼는 기록을 쓴 뒤 --out-dir 에 결함 표시
    파일 fp32_fallback_<seq>.DEFECT 를 쓰고 종료 코드 3 으로 끝난다(계획: 평가 결함으로 보고, 처리는 사용자가 정함 — F6).
    원 스크립트 summary 는 볼륨 평균을 nanmean 으로 내므로 summary 만 보면 결함 볼륨이 평균에서 빠진 것이 드러나지 않는다.
    집계 단계는 summary 를 읽기 전에 종료 코드, .DEFECT 파일, 기록의 defect·counts.fp32_nonfinite 를 확인한다.

구현
  - 원 스크립트는 고치지 않는다. runpy 로 같은 명령행 인자를 넘겨 실행한다(eval_v8fix_grufp32.py 와 같은 방식).
  - 패치 위치 = 세 모델 클래스의 forward (PureETER_UNET, PureETER_GRU, PureETER_SS2D). 모델 모듈을 미리 import 하지 않는다.
    sys.meta_path 후크가 원 스크립트가 그 모듈을 import 하는 순간(클래스 정의 직후, 인스턴스를 만들기 전) 클래스 속성 forward 를
    감싼다. 그래서 원 스크립트의 import 순서와 sys.path 는 그대로다. state_dict 키도 그대로다.
  - --seq gru 는 원 스크립트가 build() 에서 gru_precision.install_class_patch 로 gru_h 를 fp32 로 계산한다. 이 래퍼는 그 위에
    얹힌다(fp32 재계산 때도 gru_h 패치가 그대로 작동한다).
  - 감싼 호출 안에서 다시 감싼 호출이 일어나면(안쪽 호출) 규칙을 따로 적용하지 않는다. 규칙의 단위는 가장 바깥 호출 하나다.
  - 호출 번호(call) = 가장 바깥 패치 호출을 0 부터 센 순번이다. 이 스크립트는 슬라이스마다 모델을 한 번 부르므로
    call = 슬라이스 번호(per_slice CSV 의 idx)다. 슬라이스 정보(idx, file, slice_idx)는 원 스크립트 루프의 지역 변수를 읽어
    얻는다(읽기만 한다).
  - 출력 파일 경로(--out-dir)와 --seq 는 원 스크립트의 argparse 결과를 그대로 받아 쓴다(파싱 결과는 바꾸지 않음).

기록 (끝날 때, --out-dir)
  fp32_fallback_<seq>.json: 규칙, 원 스크립트·래퍼·불러온 저장소 모듈의 sha256, torch/CUDA/cuDNN 판·GPU, TF32 플래그(첫 호출 때와
  끝날 때), 호출 수, 다시 계산한 호출 수와 목록(호출 번호, 볼륨·슬라이스, fp16 출력의 NaN/inf 개수, fp32 출력이 유한한지,
  두 출력이 모두 유한한 화소에서의 차), 원 스크립트 요약의 슬라이스 수와 호출 수의 대조(crosscheck), 그리고
    - outputs: 이 기록이 속한 원 스크립트 출력(per_volume_<seq>.csv, per_slice_<seq>.csv, summary_<seq>.json)의 sha256·크기·
      수정 시각·이번 실행에서 쓰였는지·비유한 칸 수. 기록과 출력 파일을 이 sha256 으로 짝짓는다.
    - checks: 래퍼 점검 결과(ok, failures)와 이 프로세스의 종료 코드(exit_code — 아래 '종료 코드'). retired_records: 시작 때
      이름을 바꾼 앞 실행의 기록.
    - defect, defect_reasons: 결함 여부와 그 이유. 결함 = fp32 재계산 출력도 비유한인 호출이 있음(대체된 판 포함, FVW-2), 또는
      런처 래퍼가 기능 시험 단위를 슬라이스에 붙이지 못함(FVW-3, eval_v8fix_fb.py).
    - counts.fp32_nonfinite (FVW-2): fp32 재계산 출력이 비유한이었던 재계산의 수다. 런처 래퍼에서 다시 돌려 대체된 기능 시험
      판(functional_slice_rule.superseded_records)의 것도 센다 — counts.fp32_nonfinite_superseded 가 그 가운데 대체된 판의 수다.
      records·recomputed_slices·counts.recomputed 는 지금처럼 마지막 판(과 per-slice 단계)의 것이다.
  - 단위 (FV-5): records[].fp16_vs_fp32_on_finite 의 maxabs·meanabs·fp32_absmax 는 감싼 호출 자체의 출력 단위다. 모델 전체
    forward 의 출력은 eval_v8fix.Normalized 가 ×s 로 원 단위에 되돌리기 전 값(정규화 단위)이며, 라벨 단위(RSS × 1e6)가 아니다.
    rel_maxabs = maxabs / fp32_absmax 는 단위가 없다. 기록의 notes 칸에도 적는다.
  - 시간 (FV-4): records[].seconds = fp32 재계산의 벽시계 시간(time.perf_counter). torch.cuda.synchronize() 뒤에 재기 시작하고,
    재계산이 끝나면 다시 synchronize 한 뒤 잰다. 그래서 GPU 커널 실행 시간이 들어간다.
  - 앞 실행의 기록 정리 (FV-3): 첫 모델 호출 때(= 원 스크립트가 '출력이 이미 있음' 같은 거부 점검을 통과하고 새 출력을 만들기
    시작할 때) 같은 --out-dir 의 fp32_fallback_<seq>.json · .incomplete.json · .DEFECT 를 '<이름>.stale-<UTC 시각>' 으로 바꾼다.
    --selftest, 거부, 모델 호출 전의 실패 때는 건드리지 않는다(앞 실행의 출력과 기록이 그대로 짝을 이룬다).
  - 원 스크립트가 비정상 종료하면 같은 내용을 fp32_fallback_<seq>.incomplete.json 에 쓴다(모델 호출이 한 번이라도 있었을 때).
  - 모델 호출이 한 번도 없을 때: 출력을 만들지 않는 실행(--selftest, --help, 런처의 --summary-only, 거부, 호출 전 실패)이면
    아무것도 쓰지 않는다. 원 스크립트가 정상으로 끝났는데 호출이 0 번이면(패치가 걸리지 않음 — 규칙 없이 출력이 만들어졌다)
    그 사실을 checks.failures 에 적은 기록을 쓰고 종료 코드 4 로 끝난다.
  - 끝에 요약 한 줄을 찍는다: [fp32-fallback] mode=... seq=... status=... calls=... recomputed=... fp32_nonfinite=... -> <json>
  - 원 스크립트의 출력 파일은 다시 계산한 호출이 없으면 원 스크립트만 돌렸을 때와 같다(summary 의 실행 시간 seconds 칸만
    실행마다 다름).

종료 코드 (원 스크립트가 정상으로 끝났을 때)
  0 = 기록을 썼고 결함·점검 실패가 없다 (또는 출력을 만들지 않는 실행이라 기록이 필요 없다)
  3 = 결함: fp32 재계산 출력도 비유한인 호출이 있다(대체된 판 포함), 또는 기능 시험 단위를 슬라이스에 붙이지 못했다
      (기록 + fp32_fallback_<seq>.DEFECT)
  4 = 점검 실패: 감싼 모델 호출이 0 번, 호출 수 대조 불일치, 원 스크립트 출력 파일이 없거나 이번 실행에서 쓰이지 않음 등
      (기록의 checks.failures 에 이유)
  5 = 기록 실패: 기록이나 결함 표시 파일을 쓰지 못함, --out-dir/--seq 를 받지 못함 (앞 실행 기록의 이름을 바꾸지 못하면 그
      모델 호출에서 예외로 멈춘다)
  여러 개가 겹치면 5, 3, 4 순으로 앞의 것을 쓴다. 원 스크립트가 실패하면(예외, 0 이 아닌 SystemExit) 그 종료 상태를 그대로 전한다:
  예외는 1, SystemExit(n) 은 n 의 하위 8비트(문자열 등은 1), KeyboardInterrupt 는 traceback 을 찍은 뒤 130 으로 끝난다.
  기록의 checks.exit_code 는 이 프로세스의 종료 코드와 늘 같다 (FVW-4, 10-23). 원 스크립트가 실패한 기록(.incomplete.json)도
  그 종료 상태를 적는다. 결함 표시 파일을 쓰지 못하면 기록을 exit_code 5 로 다시 쓰고 5 로 끝난다. 그 다시 쓰기도 실패하면
  기록을 지우고 5 로 끝나며, 지우지도 못하면 기록에 적힌 코드로 끝난다. 기록을 쓰기 전에 실패하면 기록은 남지 않는다.

시험 전용 후크 (기본 꺼짐)
  환경 변수 EVAL_FB_FORCE_NONFINITE="3,20" (쉼표로 구분한 호출 번호)를 주면, 그 호출들의 fp16 출력 사본의 첫 원소를 NaN,
  마지막 원소를 +inf 로 바꾼 뒤 규칙을 적용한다. EVAL_FB_FORCE_NONFINITE_FP32 는 같은 방식으로 fp32 재계산 출력 사본을 바꾼다
  (그 호출이 실제로 다시 계산될 때만 효과가 있다 — 결함 경로 시험용). 재계산 경로를 시험하는 용도다. 켜면 시작과 끝에 경고를
  찍고, JSON 의 test_hook.active 가 true 이며 그 기록은 forced / forced_fp32 = true 로 표시된다. 실제 평가에서는 두 변수를 두지
  않는다(빈 문자열 = 꺼짐).

실행 (인자는 eval_v8fix_official.py 와 같다)
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/eval_v8fix_official_fb.py --seq ss2d --tf32 default \\
      --ckpt logs/PureETER_SS2D_noDC_R4_brain384_v8fix_s1_fullval/pure_ss2d_best.pt \\
      --data-path ./fastMRI_data/multicoil_test_full --out-dir <폴더> --num-workers 4
  런처 평가(eval_v8fix.py)용 판은 eval_v8fix_fb.py 이고, 이 파일의 기계를 그대로 쓴다(기능 시험에는 슬라이스 단위 규칙을 더한다).
"""

import os
import sys
import csv
import json
import math
import time
import runpy
import hashlib
import argparse
import datetime
import platform
import linecache
import threading
import traceback
import importlib.abc


class _LazyTorch:
    """torch 는 원 스크립트가 import 한 뒤에야 쓴다 (첫 모델 호출 때). 래퍼가 먼저 import 하면 import 순서가 바뀌어
    (원 스크립트는 numpy -> torch) numpy(MKL) 의 다중 스레드 합산이 끝자리에서 달라질 수 있다 — 10-22 시험에서
    torch 를 먼저 import 하면 np.linalg.norm 이 마지막 자리에서 달라져 per_volume 의 nmse_off 가 1e-17 만큼 달랐다."""

    def __getattr__(self, name):
        import torch as _t
        return getattr(_t, name)


torch = _LazyTorch()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OFFICIAL_SCRIPT = os.path.join(_HERE, 'eval_v8fix_official.py')
FORCE_ENV = 'EVAL_FB_FORCE_NONFINITE'
FORCE32_ENV = 'EVAL_FB_FORCE_NONFINITE_FP32'
PLAN = 'paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/analysis_plan_seed23_amend3.md'
RULE = ('If an fp16-autocast output has any non-finite value, the same call is repeated once with autocast disabled and '
        'floating inputs cast to float32 (same module instance = same checkpoint, same input tensors, TF32 flags untouched) '
        'and that output is returned to the evaluation script. Finite fp16 outputs are returned unchanged. If the fp32 '
        'output is still non-finite it is recorded (evaluation defect) and the script continues; the wrapper then writes '
        'fp32_fallback_<seq>.DEFECT next to this record and exits with code 3.')
NOTES = dict(
    fp16_vs_fp32_on_finite=('Computed on the output of the wrapped call itself, over pixels where both outputs are finite. '
                            'For the whole-model forward (official evaluation, launcher per-slice CSV) this is the base model '
                            'output before the x s rescaling of eval_v8fix.Normalized (normalized units, not the RSS x 1e6 '
                            'label units); for functional-test units it is the U-Net output (normalized units) or the '
                            'sequence-module output. maxabs, meanabs and fp32_absmax are in these units; rel_maxabs = '
                            'maxabs / fp32_absmax is unitless.'),
    seconds=('Wall time of the fp32 recomputation (time.perf_counter), started after torch.cuda.synchronize() and '
             'stopped after another torch.cuda.synchronize(), so GPU kernel time is included.'),
    call='Outermost wrapped call number counted from 0; in the official evaluation call = idx of per_slice_<seq>.csv.',
    exit_codes=('0 ok; 3 defect (an fp32 recomputation is still non-finite, in any pass, or a functional-test unit could '
                'not be assigned to its slice; DEFECT marker written); 4 check failure (checks.failures); 5 record failure. '
                'A failing evaluation script keeps its own exit status (1 for an exception, the SystemExit code, 130 for '
                'KeyboardInterrupt). checks.exit_code is the exit status of this process: if the DEFECT marker cannot be '
                'written the record is rewritten with exit_code 5 (or removed if that rewrite fails).'),
    fp32_nonfinite=('counts.fp32_nonfinite counts every recomputation whose fp32 output was still non-finite, including '
                    'those of superseded functional-test passes (counts.fp32_nonfinite_superseded); records and '
                    'recomputed_slices list the final pass and the per-slice phase only.'),
)
EXIT_DEFECT, EXIT_CHECK, EXIT_RECORD = 3, 4, 5
EXIT_SIGINT = 130       # KeyboardInterrupt 로 원 스크립트가 멈추면 traceback 을 찍고 이 코드로 끝낸다 (기록과 같은 값)
DEFAULT = object()      # UNIT_HANDLER 가 돌려주면 기본 규칙(가장 바깥 호출 단위)을 쓴다
UNIT_HANDLER = None     # eval_v8fix_fb.py 가 기능 시험 단계의 슬라이스 단위 규칙을 넣는다 (공식 평가에서는 None)

# 모듈 이름 -> [(클래스, 메서드)] — 원 스크립트가 그 모듈을 import 할 때 감싼다
OFFICIAL_TARGETS = {
    'u_pure_eternet_unet': [('PureETER_UNET', 'forward')],
    'u_pure_eternet_gru': [('PureETER_GRU', 'forward')],
    'u_pure_eternet_ss2d': [('PureETER_SS2D', 'forward')],
}

# 기록이 짝을 이루는 원 스크립트 출력 (mode 별)
OUTPUTS = {
    'official': ('per_volume_{seq}.csv', 'per_slice_{seq}.csv', 'summary_{seq}.json'),
    'launcher': ('per_slice_{seq}.csv', 'summary_{seq}.md', 'functional_{seq}.json'),
}


# ------------------------------------------------------------------ 슬라이스 정보 (원 스크립트 루프의 지역 변수, 읽기만)
def _slice_official(L):
    # eval_v8fix_official.main: idx = 전역 슬라이스 번호, fp = 볼륨 경로, s = 볼륨 안 슬라이스 번호
    return dict(idx=int(L['idx']), file=os.path.basename(L['fp']), slice_idx=int(L['s']))


def _slice_from_dataset(ds_name):
    def get(L):
        i = int(L['idx'])
        fp, s, _ = L[ds_name].samples[i]
        return dict(idx=i, file=os.path.basename(fp), slice_idx=int(s))
    return get


# (파일 이름, 함수 이름) -> 지역 변수에서 슬라이스 정보를 꺼내는 함수. 호출 스택을 위로 올라가며 처음 맞는 프레임을 쓴다.
SLICE_FRAMES = {
    ('eval_v8fix_official.py', 'main'): _slice_official,
    ('eval_unet_only_v8.py', 'run_inference'): _slice_from_dataset('ds'),        # eval_v8fix.py 의 per-slice CSV
    ('smoke_recipe_fix.py', 'run_validation'): _slice_from_dataset('val_ds'),    # eval_v8fix.py 의 기능 시험
}

# ------------------------------------------------------------------ 상태
ST = dict(mode=None, target=None, wrapper_files=[], started_utc=None, target_started_utc=None, args=None,
          force=set(), force32=set(), calls=0, calls_autocast=0, by_phase={}, records=[], patched=[], patch_errors=[],
          first_call=None, gru=None, out_before={}, retired=[], retire_done=False,
          superseded=dict(calls=0, calls_autocast=0, records=[]), finalize=None)
_TLS = threading.local()


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _tf32():
    return dict(matmul=bool(torch.backends.cuda.matmul.allow_tf32), cudnn=bool(torch.backends.cudnn.allow_tf32))


def _autocast_on():
    try:
        return bool(torch.is_autocast_enabled('cuda'))        # torch >= 2.4
    except TypeError:
        return bool(torch.is_autocast_enabled())              # torch 2.3: 인자 없음 = CUDA autocast


def _autocast_dtype():
    try:
        return str(torch.get_autocast_dtype('cuda'))          # torch >= 2.4
    except (AttributeError, TypeError):
        return str(torch.get_autocast_gpu_dtype())


def _tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (tuple, list)):
        for v in x:
            yield from _tensors(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _tensors(v)


def _finite(x):
    for t in _tensors(x):
        if t.is_floating_point() and not bool(torch.isfinite(t).all()):
            return False
    return True


def _f32(x):
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x      # 이미 float32 면 같은 텐서 (복사 없음)
    if isinstance(x, tuple):
        return tuple(_f32(v) for v in x)
    if isinstance(x, list):
        return [_f32(v) for v in x]
    if isinstance(x, dict):
        return {k: _f32(v) for k, v in x.items()}
    return x


def _first_tensor(x):
    for t in _tensors(x):
        if t.is_floating_point():
            return t
    return None


def _sync(x):
    """x 의 첫 CUDA 텐서가 있는 장치를 동기화한다 (시간 측정용, 값은 바꾸지 않음)."""
    t = _first_tensor(x)
    if t is not None and t.is_cuda:
        torch.cuda.synchronize(t.device)


def _force_nonfinite(out):
    """시험 전용: 출력의 사본에서 첫 원소를 NaN, 마지막 원소를 +inf 로 바꾼다 (원래 텐서는 그대로)."""
    if not torch.is_tensor(out):
        raise RuntimeError(f'{FORCE_ENV}: 텐서 출력만 지원 (받은 형식 {type(out).__name__})')
    bad = out.clone(memory_format=torch.contiguous_format)
    flat = bad.view(-1)
    flat[0] = float('nan')
    if flat.numel() > 1:
        flat[-1] = float('inf')
    return bad


def _stats(x):
    t = _first_tensor(x)
    if t is None:
        return None
    return dict(dtype=str(t.dtype).replace('torch.', ''), shape=list(t.shape), numel=int(t.numel()),
                n_nan=int(torch.isnan(t).sum()), n_posinf=int(torch.isposinf(t).sum()),
                n_neginf=int(torch.isneginf(t).sum()))


def _diff(a, b):
    """두 출력이 모두 유한한 화소에서의 차 (fp16 출력은 float 로 올려 비교). 단위는 감싼 호출의 출력 단위 (NOTES)."""
    a, b = _first_tensor(a), _first_tensor(b)
    if a is None or b is None or a.shape != b.shape:
        return None
    a, b = a.float(), b.float()
    m = torch.isfinite(a) & torch.isfinite(b)
    if not bool(m.any()):
        return None
    d = (a - b).abs()[m]
    scale = float(b[torch.isfinite(b)].abs().max())
    mx = float(d.max())
    return dict(n_compared=int(m.sum()), maxabs=mx, meanabs=float(d.mean()), fp32_absmax=scale,
                rel_maxabs=(mx / scale) if scale > 0 else None)


# ------------------------------------------------------------------ 호출 스택에서 위치 찾기 (읽기만)
def _find_frame():
    f = sys._getframe(1)
    while f is not None:
        key = (os.path.basename(f.f_code.co_filename), f.f_code.co_name)
        if key in SLICE_FRAMES:
            return key, f
        f = f.f_back
    return None, None


def _slice_info(key, f):
    if key is None:
        return None
    try:
        info = SLICE_FRAMES[key](f.f_locals)
    except Exception as e:                                    # noqa: BLE001 — 기록 실패가 평가를 멈추지 않게
        info = dict(error=f'{type(e).__name__}: {e}')
    info['site'] = f'{key[0]}:{key[1]}:{f.f_lineno}'
    info['site_src'] = linecache.getline(f.f_code.co_filename, f.f_lineno).strip()[:160]
    return info


def _first_call_info(self, label, out):
    t = _first_tensor(out)
    return dict(utc=_utc(), target=label, cls=type(self).__name__, tf32=_tf32(), autocast_dtype=_autocast_dtype(),
                out_dtype=None if t is None else str(t.dtype).replace('torch.', ''),
                device=None if t is None else str(t.device),
                gpu=torch.cuda.get_device_name(t.device) if (t is not None and t.is_cuda) else None)


def _note_gru(self):
    """ETER-net: gru_h 의 fp32 인스턴스 패치(gru_precision)가 붙어 있는지 기록 (한 번)."""
    if ST['gru'] is not None or not hasattr(self, 'gru_h'):
        return
    ST['gru'] = dict(gru_h_instance_forward_patched='forward' in vars(self.gru_h),
                     gru_v_instance_forward_patched='forward' in vars(self.gru_v) if hasattr(self, 'gru_v') else None,
                     class_fp32_patched=getattr(type(self), '_fp32_patched', None))


def note_first_call(self, label, out):
    if ST['first_call'] is None:
        ST['first_call'] = _first_call_info(self, label, out)
    _note_gru(self)


def phase_stats(phase):
    return ST['by_phase'].setdefault(phase, dict(calls=0, calls_autocast=0, recomputed=0, fp32_nonfinite=0))


# ------------------------------------------------------------------ 앞 실행의 기록 정리 (FV-3)
def _record_names(seq):
    return (f'fp32_fallback_{seq}.json', f'fp32_fallback_{seq}.incomplete.json', f'fp32_fallback_{seq}.DEFECT')


def retire_stale_once():
    """같은 --out-dir 에 앞 실행의 기록·결함 표시가 있으면 '<이름>.stale-<UTC 시각>' 으로 바꾼다 (실행마다 한 번).
    첫 모델 호출 때 부른다(원 스크립트의 거부 점검을 통과한 뒤). 이름을 바꾸지 못하면 예외를 낸다."""
    if ST['retire_done']:
        return
    ns = ST['args']
    seq, out_dir = getattr(ns, 'seq', None), getattr(ns, 'out_dir', None)
    if not seq or not out_dir:
        return
    ST['retire_done'] = True
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    for name in _record_names(seq):
        p = os.path.join(out_dir, name)
        if not os.path.lexists(p):
            continue
        q = f'{p}.stale-{stamp}'
        sha = _sha256(p) if os.path.isfile(p) else None
        try:
            os.rename(p, q)
        except OSError as e:
            raise RuntimeError(f'[fp32-fallback] 앞 실행의 기록 {p} 의 이름을 바꾸지 못함: {e}') from e
        ST['retired'].append(dict(name=name, renamed_to=os.path.basename(q), sha256=sha, utc=_utc()))
        print(f'[fp32-fallback] 앞 실행의 기록 {name} -> {os.path.basename(q)}', flush=True)


# ------------------------------------------------------------------ fp32 재계산 한 번 (공식 평가와 기능 시험이 함께 씀)
def recompute_fp32(orig, self, args, kwargs, *, call, label, phase, ph, slice_info, out16, forced, ref16, extra=None):
    """한 호출을 autocast 끄고 fp32 로 다시 계산해 (출력, 기록) 을 돌려준다. 기록은 ST['records'] 에 더한다.
    재계산이 예외를 내면 기록을 남기고 그대로 올린다. out16 = 규칙을 건 fp16 출력(후크로 바꾼 사본일 수 있음)."""
    extra = extra or {}
    _sync(out16)
    t0 = time.perf_counter()
    try:
        with torch.autocast('cuda', enabled=False):
            out32 = orig(self, *_f32(args), **_f32(kwargs))
        _sync(out32)
    except BaseException as e:                                # fp32 재계산 자체가 실패 (예: 메모리) — 기록하고 그대로 올림
        ST['records'].append(dict(call=call, target=label, phase=phase, forced=forced, slice=slice_info, fp16=_stats(out16),
                                  fp32=dict(finite=False, exception=f'{type(e).__name__}: {e}'), utc=_utc(), **extra))
        ph['recomputed'] += 1
        ph['fp32_nonfinite'] += 1
        print(f'[fp32-fallback] call {call} {label}: fp32 재계산 실패 {type(e).__name__}: {e}', flush=True)
        raise
    dt = time.perf_counter() - t0
    forced32 = call in ST['force32']
    if forced32:
        out32 = _force_nonfinite(out32)
    ok32 = _finite(out32)
    rec = dict(call=call, target=label, phase=phase, forced=forced,
               slice=slice_info, fp16=_stats(out16),
               fp16_finite_before_forcing=_finite(ref16) if forced else None,
               fp32=dict(finite=ok32, **(_stats(out32) or {})),
               fp16_vs_fp32_on_finite=_diff(ref16 if forced else out16, out32),
               seconds=round(dt, 4), utc=_utc(), forced_fp32=forced32, **extra)
    ST['records'].append(rec)
    ph['recomputed'] += 1
    if not ok32:
        ph['fp32_nonfinite'] += 1
    si = rec['slice'] or {}
    rule = f' [{extra["unit_rule"]} rule{" " + str(extra["test"]) if extra.get("test") else ""}]' if extra.get('unit_rule') else ''
    print(f'[fp32-fallback] call {call} {label}{rule} {si.get("file", "?")} slice {si.get("slice_idx", "?")} '
          f'(idx {si.get("idx", "?")}): fp16 nan {rec["fp16"]["n_nan"]} +inf {rec["fp16"]["n_posinf"]} '
          f'-inf {rec["fp16"]["n_neginf"]} / {rec["fp16"]["numel"]} -> fp32 '
          f'{"finite" if ok32 else "NON-FINITE (evaluation defect)"}'
          f'{" [forced by test hook]" if forced else ""}{" [fp32 forced by test hook]" if forced32 else ""}', flush=True)
    return out32, rec


def _call_rule(orig, self, args, kwargs, label, key, fr):
    """기본 규칙: 가장 바깥 패치 호출 하나가 단위. fp16 출력이 비유한일 때만 그 호출을 fp32 로 다시 계산."""
    call = ST['calls']
    ST['calls'] += 1
    retire_stale_once()
    phase = f'{key[0]}:{key[1]}' if key else 'other'
    ph = phase_stats(phase)
    ph['calls'] += 1
    _TLS.depth = 1
    try:
        out = orig(self, *args, **kwargs)
        if not _autocast_on():
            return out
        ST['calls_autocast'] += 1
        ph['calls_autocast'] += 1
        note_first_call(self, label, out)
        forced = call in ST['force']
        ref16 = None
        if forced:
            ref16 = out
            out = _force_nonfinite(out)
        if _finite(out):
            return out
        out32, _ = recompute_fp32(orig, self, args, kwargs, call=call, label=label, phase=phase, ph=ph,
                                  slice_info=_slice_info(key, fr), out16=out, forced=forced, ref16=ref16)
        return out32
    finally:
        _TLS.depth = 0


# ------------------------------------------------------------------ 클래스 메서드 감싸기
def _wrap_method(cls, meth, label):
    orig = cls.__dict__.get(meth)
    if orig is None:
        raise AttributeError(f'{cls.__name__}.{meth} 없음')
    if getattr(orig, '_fp32_fallback_orig', None) is not None:
        return False                                          # 이미 감쌈

    def wrapper(self, *args, **kwargs):
        if getattr(_TLS, 'depth', 0) > 0:                     # 바깥 패치 호출 안쪽: 규칙은 바깥 호출 단위로만
            return orig(self, *args, **kwargs)
        key, fr = _find_frame()
        try:
            if UNIT_HANDLER is not None:                      # 런처 래퍼의 기능 시험 단계 (eval_v8fix_fb.py)
                out = UNIT_HANDLER(orig, self, args, kwargs, label, key, fr)
                if out is not DEFAULT:
                    return out
            return _call_rule(orig, self, args, kwargs, label, key, fr)
        finally:
            del fr

    for k in ('__module__', '__name__', '__qualname__', '__doc__'):
        try:
            setattr(wrapper, k, getattr(orig, k))
        except (AttributeError, TypeError):
            pass
    wrapper.__wrapped__ = orig
    wrapper._fp32_fallback_orig = orig
    setattr(cls, meth, wrapper)
    return True


def _patch_module(module, modname, items, how, post=()):
    for cls_name, meth in items:
        tgt = f'{modname}.{cls_name}.{meth}'
        try:
            cls = getattr(module, cls_name)
            done = _wrap_method(cls, meth, f'{cls_name}.{meth}')
            ST['patched'].append(dict(target=tgt, how=how, utc=_utc(), wrapped=done,
                                      file=os.path.relpath(getattr(module, '__file__', '') or '', _ROOT)))
        except Exception as e:                                # noqa: BLE001
            ST['patch_errors'].append(dict(target=tgt, error=f'{type(e).__name__}: {e}'))
            print(f'[fp32-fallback] 패치 실패 {tgt}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            raise
    for fn in post:                                           # 모듈 함수 감싸기 등 (eval_v8fix_fb.py 의 run_validation)
        tgt = f'{modname}:{getattr(fn, "__name__", fn)}'
        try:
            done = fn(module)
            ST['patched'].append(dict(target=tgt, how=how, utc=_utc(), wrapped=done,
                                      file=os.path.relpath(getattr(module, '__file__', '') or '', _ROOT)))
        except Exception as e:                                # noqa: BLE001
            ST['patch_errors'].append(dict(target=tgt, error=f'{type(e).__name__}: {e}'))
            print(f'[fp32-fallback] 패치 실패 {tgt}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            raise


class _PatchingLoader(importlib.abc.Loader):
    """원 로더로 모듈을 실행한 직후 클래스 메서드를 감싼다. 나머지(get_source 등)는 원 로더에 맡긴다."""

    def __init__(self, inner, modname, items, post=()):
        self._inner, self._modname, self._items, self._post = inner, modname, items, post

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch_module(module, self._modname, self._items, how='on import', post=self._post)

    def __getattr__(self, name):
        inner = self.__dict__.get('_inner')                   # 복사 중 등 _inner 가 없을 때 재귀 방지
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """대상 모듈 이름만 가로챈다. 모듈을 찾는 일은 sys.meta_path 의 다른 finder 에 같은 순서로 맡긴다(평소 import 와 같은 파일)."""

    def __init__(self, targets, post=None):
        self.targets = targets
        self.post = post or {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.targets and fullname not in self.post:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        if spec is None:
            return None
        if spec.loader is not None and hasattr(spec.loader, 'exec_module'):
            spec.loader = _PatchingLoader(spec.loader, fullname, self.targets.get(fullname, []),
                                          self.post.get(fullname, ()))
        return spec


def install(targets, post_import=None):
    """import 후크를 맨 앞에 넣는다. 이미 import 된 대상 모듈은 바로 감싼다."""
    post_import = post_import or {}
    for modname in list(targets) + [m for m in post_import if m not in targets]:
        if modname in sys.modules:
            _patch_module(sys.modules[modname], modname, targets.get(modname, []), how='already imported',
                          post=post_import.get(modname, ()))
    if not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport(targets, post_import))


# ------------------------------------------------------------------ argparse 결과 받기 (값은 그대로 돌려줌)
_ORIG_PARSE_ARGS = argparse.ArgumentParser.parse_args


def _output_paths(ns):
    seq, out_dir = getattr(ns, 'seq', None), getattr(ns, 'out_dir', None)
    if not seq or not out_dir or ST['mode'] not in OUTPUTS:
        return {}
    return {t.format(seq=seq): os.path.join(out_dir, t.format(seq=seq)) for t in OUTPUTS[ST['mode']]}


def _stat_key(p):
    try:
        st = os.stat(p)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_ino, st.st_size)


def _capture_args():
    def parse_args(self, *a, **k):
        ns = _ORIG_PARSE_ARGS(self, *a, **k)
        if ST['args'] is None and hasattr(ns, 'seq') and hasattr(ns, 'out_dir'):
            ST['args'] = ns
            argparse.ArgumentParser.parse_args = _ORIG_PARSE_ARGS     # 한 번 받으면 되돌림
            ST['out_before'] = {n: _stat_key(p) for n, p in _output_paths(ns).items()}   # 출력이 이번 실행에서 쓰였는지 판정용
        return ns
    argparse.ArgumentParser.parse_args = parse_args


def _parse_force(env):
    raw = os.environ.get(env, '').strip()
    if not raw:
        return set()
    try:
        idx = {int(v) for v in raw.replace(' ', '').split(',') if v != ''}
    except ValueError:
        raise SystemExit(f'[fp32-fallback] {env}={raw!r}: 쉼표로 구분한 0 이상의 정수여야 함')
    if any(i < 0 for i in idx):
        raise SystemExit(f'[fp32-fallback] {env}={raw!r}: 음수 호출 번호')
    return idx


# ------------------------------------------------------------------ 끝날 때 기록
def _repo_modules():
    out = {}
    for name, m in list(sys.modules.items()):
        fp = getattr(m, '__file__', None)
        if not fp or not fp.endswith('.py'):
            continue
        ap = os.path.abspath(fp)
        if ap.startswith(_ROOT + os.sep) and os.path.exists(ap):
            try:
                out[os.path.relpath(ap, _ROOT)] = _sha256(ap)
            except OSError:
                pass
    return dict(sorted(out.items()))


def _nonfinite_json(x):
    if isinstance(x, float):
        return 0 if math.isfinite(x) else 1
    if isinstance(x, dict):
        return sum(_nonfinite_json(v) for v in x.values())
    if isinstance(x, list):
        return sum(_nonfinite_json(v) for v in x)
    return 0


def _count_nonfinite(p):
    """CSV 칸·JSON 숫자 가운데 비유한 값의 개수 (기술용; md 는 None)."""
    try:
        if p.endswith('.csv'):
            n = 0
            with open(p, newline='') as f:
                for row in csv.reader(f):
                    for v in row:
                        try:
                            n += 0 if math.isfinite(float(v)) else 1
                        except ValueError:
                            pass
            return n
        if p.endswith('.json'):
            with open(p) as f:
                return _nonfinite_json(json.load(f))
    except Exception:                                         # noqa: BLE001 — 기술용 칸
        return None
    return None


def _outputs_record(ns):
    res = {}
    for n, p in _output_paths(ns).items():
        now = _stat_key(p)
        if now is None or not os.path.isfile(p):
            res[n] = None
            continue
        res[n] = dict(sha256=_sha256(p), bytes=now[2],
                      mtime_utc=datetime.datetime.fromtimestamp(now[0] / 1e9, datetime.timezone.utc)
                      .strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                      written_this_run=(ST['out_before'].get(n) != now), nonfinite_values=_count_nonfinite(p))
    return res


def _required_outputs(ns):
    seq = ns.seq
    if ST['mode'] == 'official':
        return [f'per_volume_{seq}.csv', f'per_slice_{seq}.csv', f'summary_{seq}.json']
    if ST['mode'] == 'launcher':
        return [f'per_slice_{seq}.csv', f'summary_{seq}.md'] + \
            ([f'functional_{seq}.json'] if (getattr(ns, 'func_slices', 0) or 0) > 0 else [])
    return []


def _crosscheck(out_dir, seq, completed, outputs):
    """원 스크립트의 출력과 호출 수 대조 (official: summary 의 n_slice, launcher: per_slice CSV 행 수)."""
    if not completed:
        return dict(match=None, reason='run not completed')
    calls_total = ST['calls'] - ST['superseded']['calls']
    if ST['mode'] == 'official':
        name = f'summary_{seq}.json'
        calls = ST['by_phase'].get('eval_v8fix_official.py:main', {}).get('calls', 0)
        n = None
        if outputs.get(name) and outputs[name]['written_this_run']:
            with open(os.path.join(out_dir, name)) as f:
                n = json.load(f).get('n_slice')
        return dict(summary_n_slice=n, calls_in_slice_loop=calls, calls_total=calls_total,
                    match=bool(n is not None and n == calls == calls_total))
    if ST['mode'] == 'launcher':
        name = f'per_slice_{seq}.csv'
        calls = ST['by_phase'].get('eval_unet_only_v8.py:run_inference', {}).get('calls', 0)
        n = None
        if outputs.get(name) and outputs[name]['written_this_run']:
            with open(os.path.join(out_dir, name), newline='') as f:
                n = sum(1 for _ in csv.DictReader(f))
        return dict(per_slice_csv_rows=n, calls_in_run_inference=calls, calls_total=calls_total,
                    per_slice_match=bool(n is not None and n == calls), match=bool(n is not None and n == calls))
    return dict(match=None, reason=f'unknown mode {ST["mode"]!r}')


def _noop_reason(ns):
    """모델 호출 없이 끝나도 되는 실행 (출력을 만들지 않음)."""
    if ns is None:
        return '--help' if any(a in ('-h', '--help') for a in sys.argv[1:]) else None
    if getattr(ns, 'selftest', False):
        return '--selftest'
    if getattr(ns, 'summary_only', False):
        return '--summary-only'
    return None


def _write_atomic(path, text):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _say(*a, **k):
    """print — 기록을 쓴 뒤의 출력 실패(닫힌 파이프 등)가 종료 코드를 바꾸지 않게 (FVW-4)."""
    try:
        print(*a, **k)
    except Exception:                                         # noqa: BLE001
        pass


def exit_status(exc):
    """원 스크립트가 exc 로 끝났을 때 이 프로세스의 종료 상태 (FVW-4: 기록의 checks.exit_code 에 같은 값을 적는다).
    SystemExit(None) 0, SystemExit(정수) 그 하위 8비트(POSIX exit), 그 밖의 SystemExit 1 (파이썬이 값을 stderr 에 찍고 1),
    KeyboardInterrupt EXIT_SIGINT (run() 이 traceback 을 찍고 이 코드로 끝낸다), 그 밖의 예외 1 (traceback)."""
    if isinstance(exc, SystemExit):
        c = exc.code
        if c is None:
            return 0
        if isinstance(c, int):
            return c & 0xFF
        return 1
    if isinstance(exc, KeyboardInterrupt):
        return EXIT_SIGINT
    return 1


def finish(status, target_code=None):
    """기록을 쓰고 이 프로세스의 종료 코드를 돌려준다. 원 스크립트가 끝까지 돌았으면 래퍼의 코드(0, 3 결함, 4 점검 실패,
    5 기록 실패), 실패했으면 target_code(원 스크립트의 종료 상태 — run() 이 그대로 전한다). 예외를 내지 않는다.
    기록이 디스크에 남으면 그 checks.exit_code 는 돌려주는 값과 같다 (FVW-4)."""
    try:
        return _finish(status, target_code)
    except Exception as e:                                    # noqa: BLE001 — 기록을 쓰기 전의 실패: 기록은 남지 않는다
        code = EXIT_RECORD if status == 'completed' else target_code
        _say(f'[fp32-fallback] 기록 실패 (종료 코드 {code}): {type(e).__name__}: {e}', file=sys.stderr, flush=True)
        return code


def _defect_header(n_bad32, other_reasons):
    parts = (['an fp32 recomputation is still non-finite'] if n_bad32 else []) + \
        (['the slice-level rule could not be applied to some functional-test units'] if other_reasons else [])
    return (f'EVALUATION DEFECT: {"; ".join(parts)} (analysis plan amendment 3 §2; the user decides the handling, F6). '
            'Check the record before reading any summary.')


def _finish(status, target_code=None):
    ns = ST['args']
    seq = getattr(ns, 'seq', None)
    out_dir = getattr(ns, 'out_dir', None)
    hook, hook32 = sorted(ST['force']), sorted(ST['force32'])
    warn = '[TEST HOOK ACTIVE - not a valid evaluation] ' if (hook or hook32) else ''
    completed = (status == 'completed')
    if ST['calls'] == 0:
        if not completed:
            print(f'[fp32-fallback] {warn}모델 호출 없음 (status: {status}) — 기록을 쓰지 않음', flush=True)
            return 0
        why = _noop_reason(ns)
        if why:
            print(f'[fp32-fallback] {warn}모델 호출 없음 ({why}) — 기록을 쓰지 않음', flush=True)
            return 0
        if ns is None:
            print(f'[fp32-fallback] 원 스크립트가 끝났는데 감싼 모델 호출이 0 번이고 --seq/--out-dir 도 받지 못함 — '
                  f'규칙이 걸리지 않았다 (종료 코드 {EXIT_CHECK})', file=sys.stderr, flush=True)
            return EXIT_CHECK
    if not out_dir or not seq:
        print(f'[fp32-fallback] {warn}--out-dir/--seq 를 받지 못함 — 기록을 쓸 수 없음. 호출 {ST["calls"]}, '
              f'재계산 {len(ST["records"])} (종료 코드 {EXIT_RECORD})', file=sys.stderr, flush=True)
        return EXIT_RECORD
    retire_stale_once()                                       # 호출이 0 번인 실패 기록을 쓸 때 (보통은 첫 호출 때 이미 함)
    recs = ST['records']
    sup = ST['superseded']
    n_bad32_final = sum(1 for r in recs if not r['fp32']['finite'])
    n_bad32_sup = sum(1 for r in sup['records'] if not r['fp32']['finite'])     # FVW-2: 대체된 기능 시험 판도 센다
    n_bad32 = n_bad32_final + n_bad32_sup
    n_forced = sum(1 for r in recs if r['forced'])
    outputs = _outputs_record(ns)
    crosscheck = _crosscheck(out_dir, seq, completed, outputs)
    extra = ST['finalize'](completed=completed, ns=ns, outputs=outputs, out_dir=out_dir, seq=seq,
                           defect=bool(n_bad32_final)) if ST['finalize'] else {}
    if extra.get('crosscheck'):
        crosscheck.update(extra['crosscheck'])
        if completed:
            crosscheck['match'] = bool(crosscheck.get('match') and crosscheck.get('functional_match', True))
    other_defects = list(extra.get('defects', []))           # FVW-3 (런처 래퍼: 기능 시험 단위를 슬라이스에 붙이지 못함)
    defect_reasons = []
    if n_bad32:
        defect_reasons.append(f'{n_bad32} fp32 recomputation(s) still non-finite' + (
            f' ({n_bad32_final} in the per-slice phase or the final functional-test pass, {n_bad32_sup} in superseded '
            'functional-test passes)' if ST['mode'] == 'launcher' else ''))
    defect_reasons += other_defects
    defect = bool(defect_reasons)
    failures = []
    if completed:
        if ST['calls'] - sup['calls'] == 0:
            failures.append('zero wrapped model calls in a completed run: the fp32 rule was never applied')
        if crosscheck.get('match') is not True:
            failures.append(f'crosscheck mismatch: {json.dumps(crosscheck, ensure_ascii=False, default=str)}')
        for n in _required_outputs(ns):
            o = outputs.get(n)
            if o is None:
                failures.append(f'output missing: {n}')
            elif not o['written_this_run']:
                failures.append(f'output not written in this run: {n}')
        failures += extra.get('failures', [])
    code = EXIT_DEFECT if defect else (EXIT_CHECK if failures else 0)
    exit_code = code if completed else target_code          # FVW-4: 이 프로세스의 종료 코드
    wrappers = {os.path.relpath(p, _ROOT): _sha256(p) for p in ST['wrapper_files']}
    gp = getattr(sys.modules.get('gru_precision'), '__file__', None)          # ETER-net gru_h fp32 패치 파일 (저장소 밖이어도)
    gru_precision = dict(path=os.path.abspath(gp), sha256=_sha256(gp)) if gp else None
    data = dict(
        rule=RULE, plan=PLAN + ' §2 (official) / §4 (launcher functional test)',
        mode=ST['mode'], status=status, completed=completed,
        seq=seq, out_dir=out_dir, argv=sys.argv[1:], args=vars(ns),
        target_script=dict(path=os.path.relpath(ST['target'], _ROOT), sha256=_sha256(ST['target'])),
        wrapper=wrappers,
        started_utc=ST['started_utc'], target_started_utc=ST['target_started_utc'], finished_utc=_utc(),
        test_hook=dict(env=FORCE_ENV, active=bool(hook or hook32), force_nonfinite_calls=hook,
                       env_fp32=FORCE32_ENV, force_nonfinite_fp32_calls=hook32),
        counts=dict(calls=ST['calls'] - sup['calls'], calls_autocast=ST['calls_autocast'] - sup['calls_autocast'],
                    recomputed=len(recs), recomputed_not_forced=len(recs) - n_forced, forced=n_forced,
                    fp32_nonfinite=n_bad32, calls_superseded=sup['calls'], fp32_nonfinite_superseded=n_bad32_sup),
        defect=defect, defect_reasons=defect_reasons, by_phase=ST['by_phase'],
        recomputed_slices=[dict(call=r['call'], phase=r['phase'], target=r['target'], forced=r['forced'],
                                fp32_finite=r['fp32']['finite'],
                                **{k: (r['slice'] or {}).get(k) for k in ('idx', 'file', 'slice_idx')},
                                **{k: r[k] for k in ('pass', 'unit_rule', 'test') if k in r})
                           for r in recs],
        records=recs,
        crosscheck=crosscheck,
        outputs=outputs,
        checks=dict(ok=bool(completed and not defect and not failures), exit_code=exit_code, failures=failures),
        retired_records=ST['retired'],
        patched=ST['patched'], patch_errors=ST['patch_errors'], first_call=ST['first_call'], eter_gru=ST['gru'],
        gru_precision=gru_precision,
        env=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                 cudnn=torch.backends.cudnn.version(), tf32_at_exit=_tf32(),
                 cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                 cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                 float32_matmul_precision=torch.get_float32_matmul_precision(),
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
        repo_modules_sha256=_repo_modules(),
        notes=NOTES,
        **extra.get('keys', {}),
    )
    name = f'fp32_fallback_{seq}.json' if completed else f'fp32_fallback_{seq}.incomplete.json'
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    _write_atomic(path, json.dumps(data, ensure_ascii=False, indent=1, default=str))
    # ---- 여기부터는 기록이 디스크에 있다: 예외를 내지 않고, 돌려주는 코드 = 디스크 기록의 checks.exit_code (FVW-4)
    _say(f'[fp32-fallback] {warn}mode={ST["mode"]} seq={seq} status={status} calls={data["counts"]["calls"]} '
         f'(autocast {data["counts"]["calls_autocast"]}) recomputed={len(recs)} (forced {n_forced}) fp32_nonfinite={n_bad32}'
         f'{" DEFECT" if defect else ""} -> {path}', flush=True)
    if defect:                                                # FV-1: 기록을 쓴 뒤 결함 표시 파일
        mpath = os.path.join(out_dir, f'fp32_fallback_{seq}.DEFECT')
        try:
            bad = [r for r in recs if not r['fp32']['finite']] + \
                [dict(r, superseded=True) for r in sup['records'] if not r['fp32']['finite']]
            lines = [_defect_header(n_bad32, other_defects),
                     f'mode={ST["mode"]} seq={seq} status={status} utc={_utc()}',
                     f'record={name} sha256={_sha256(path)}',
                     f'fp32_nonfinite={n_bad32} test_hook_active={bool(hook or hook32)}']
            lines += [f'reason={m}' for m in other_defects]
            lines += [f'call={r["call"]} phase={r["phase"]} target={r["target"]} idx={(r["slice"] or {}).get("idx")} '
                      f'file={(r["slice"] or {}).get("file")} slice_idx={(r["slice"] or {}).get("slice_idx")}'
                      f'{" pass=" + str(r["pass"]) if "pass" in r else ""}{" superseded" if r.get("superseded") else ""}'
                      for r in bad[:50]]
            _write_atomic(mpath, '\n'.join(lines) + '\n')
            _say(f'[fp32-fallback] DEFECT 표시 -> {mpath}', flush=True)
        except Exception as e:                                # noqa: BLE001 — FVW-4: 기록을 프로세스 종료 코드에 맞춘다
            new_code = EXIT_RECORD if completed else exit_code   # 원 스크립트가 실패했으면 그 종료 상태가 그대로 남는다
            msg = f'DEFECT marker {os.path.basename(mpath)} could not be written: {type(e).__name__}: {e}'
            data['checks'].update(ok=False, exit_code=new_code)
            data['checks']['failures'] = list(failures) + [msg]
            _say(f'[fp32-fallback] {msg} — 기록을 exit_code {new_code} 로 다시 씀', file=sys.stderr, flush=True)
            try:
                _write_atomic(path, json.dumps(data, ensure_ascii=False, indent=1, default=str))
            except Exception as e2:                           # noqa: BLE001
                _say(f'[fp32-fallback] 기록을 다시 쓰지 못함: {type(e2).__name__}: {e2}', file=sys.stderr, flush=True)
                if new_code != exit_code:
                    try:
                        os.remove(path)                       # 다른 종료 코드를 적은 기록이 남지 않게
                        _say(f'[fp32-fallback] 기록 {path} 를 지움 (종료 코드 {new_code} 와 맞출 수 없음)', file=sys.stderr,
                             flush=True)
                    except Exception as e3:                   # noqa: BLE001 — 지우지도 못하면 기록에 적힌 코드로 끝낸다
                        _say(f'[fp32-fallback] 기록을 지우지 못함: {type(e3).__name__}: {e3} — 기록의 종료 코드 '
                             f'{exit_code} 로 끝냄', file=sys.stderr, flush=True)
                        return exit_code
            exit_code = new_code
    for m in defect_reasons:
        _say(f'[fp32-fallback] 결함: {m}', file=sys.stderr, flush=True)
    for msg in failures:
        _say(f'[fp32-fallback] 점검 실패: {msg}', file=sys.stderr, flush=True)
    return exit_code


# ------------------------------------------------------------------ 실행
def run(target_script, targets, mode, wrapper_files, pre=None, unit_handler=None, post_import=None, finalize=None):
    """후크 설치 -> (pre) -> runpy 로 원 스크립트 실행 -> 끝날 때 기록. 원 스크립트가 실패하면 그 종료 상태를 그대로 전하고
    (KeyboardInterrupt 는 traceback 을 찍고 EXIT_SIGINT), 정상으로 끝났으면 래퍼의 종료 코드(0, 3, 4, 5)로 끝난다.
    기록의 checks.exit_code 는 이 종료 코드와 같다 (FVW-4)."""
    global UNIT_HANDLER
    ST.update(mode=mode, target=os.path.abspath(target_script), started_utc=_utc(),
              wrapper_files=[os.path.abspath(p) for p in wrapper_files], finalize=finalize)
    ST['force'] = _parse_force(FORCE_ENV)
    ST['force32'] = _parse_force(FORCE32_ENV)
    UNIT_HANDLER = unit_handler
    _capture_args()
    install(targets, post_import)
    names = ', '.join(f'{c}.{m}' for items in targets.values() for c, m in items)
    print(f'[{os.path.basename(ST["wrapper_files"][0])}] 수정 3 fp32 재계산 규칙: {names} 를 import 시점에 감쌈 '
          f'-> runpy {os.path.basename(target_script)}', flush=True)
    if ST['force'] or ST['force32']:
        print(f'[fp32-fallback] WARNING: TEST HOOK ACTIVE {FORCE_ENV}={sorted(ST["force"])} '
              f'{FORCE32_ENV}={sorted(ST["force32"])} — 시험 전용, 실제 평가 아님', flush=True)
    if pre is not None:
        pre()
    sys.argv[0] = ST['target']
    status, exc = 'completed', None
    ST['target_started_utc'] = _utc()
    try:
        runpy.run_path(ST['target'], run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            status, exc = f'SystemExit({e.code!r})', e
    except BaseException as e:                                # noqa: BLE001 — 기록한 뒤 그대로 다시 올린다
        status, exc = f'{type(e).__name__}: {e}', e
    target_code = None if exc is None else exit_status(exc)
    code = finish(status, target_code)
    if exc is not None:
        if isinstance(exc, KeyboardInterrupt):                # 신호로 죽지 않고 기록과 같은 코드로 끝낸다 (FVW-4)
            try:
                traceback.print_exception(type(exc), exc, exc.__traceback__)
            except Exception:                                 # noqa: BLE001
                pass
            raise SystemExit(target_code)
        raise exc
    if code:
        why = {EXIT_DEFECT: '결함', EXIT_CHECK: '점검 실패', EXIT_RECORD: '기록 실패'}.get(code, '?')
        _say(f'[fp32-fallback] 종료 코드 {code} ({why})', file=sys.stderr, flush=True)
        raise SystemExit(code)


if __name__ == '__main__':
    run(OFFICIAL_SCRIPT, OFFICIAL_TARGETS, mode='official', wrapper_files=[__file__])
