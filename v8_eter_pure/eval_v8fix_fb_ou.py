"""
원래 U-Net(OU) 세 모델을 런처 평가(eval_v8fix.py: per-slice CSV + 기능 시험)로, 수정 3 의 fp32 재계산 규칙 래퍼(eval_v8fix_fb.py)를
그대로 써서 평가하는 진입점 (2026-10-26). infra/runpod/pod_run_ou.sh 가 학습 뒤에 부른다.
계획 paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan.md §3 (기능 시험: 시드 1 과 같은 512 슬라이스).

기존 파일(eval_v8fix.py · eval_v8fix_fb.py · eval_v8fix_official_fb.py · smoke_recipe_fix.py 등)은 고치지 않는다. 이 파일은 같은 프로세스 안에서만:
  1. eval_v8fix_fb.py 를 모듈로 불러(그 파일이 eval_v8fix_official_fb.py 를 불러 쓰는 것과 같은 방식) 그 기계(_unit_handler ·
     _patch_run_validation · _finalize · _gru_fp32)를 그대로 쓴다. 기록·종료 코드·기능 시험의 슬라이스 단위 규칙은 그 파일 그대로다.
  2. 감쌀 대상(LAUNCHER_TARGETS)에 OU 모델 클래스의 forward 세 개(모델 전체 forward — 기능 시험에서는 그릇)와 OriginalUNet.forward
     (기능 시험의 U-Net 단위)를 더하고, 그 래퍼의 라벨 집합(MODEL_LABELS · UNET_LABELS)에 같은 이름을 더한다. 시퀀스 모듈 단위는
     그대로 PureETER_GRU._seq 다(gru_ou · gruo_ou 는 그것을 상속한다).
  3. eval_v8fix.py 의 --seq 선택지와 SPECS 에 OU 이름·제목을 더하고(eval_v8fix_official_fb_ou.install_seq_choices), 원 스크립트가
     smoke_recipe_fix 를 import 하는 순간 recipe_ou.register 로 R.build_model · R.run_validation 이 OU 이름을 받게 한다 — 래퍼의
     run_validation 감싸기(_patch_run_validation)보다 먼저. 그래서 기능 시험은 래퍼의 슬라이스 단위 규칙 안에서 R.run_validation 원
     함수로 돈다(o-ETER (OU) 는 recipe_ou 의 대리 객체 — U-Net 이 시퀀스 출력 20 채널만 받는다).
  4. gru_ou · gruo_ou 는 --seq gru 와 같이 실행 전에 래퍼의 _gru_fp32(gru_precision.install_class_patch, 기본 gru_h)를 부른다.
  5. 점검 하나를 더한다: gru_ou · gruo_ou 의 기능 시험에 시퀀스 모듈 단위 호출이 없으면 점검 실패(래퍼가 gru · ss2d 에 하는 점검과 같음).
기존 이름(unet · gru · ss2d)으로 부르면 이 진입점이 더한 것은 아무 일도 하지 않는다(원 래퍼와 같은 실행).

실행 (인자는 eval_v8fix.py 와 같다. --seq 는 줄여 쓰지 않는다. 출력 폴더는 새 폴더로):
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/eval_v8fix_fb_ou.py --seq gruo_ou \\
      --ckpt logs/PureETER_GRUOOU_noDC_R4_brain384_v8fix_s1_fullval/pure_gruo_ou_best.pt \\
      --data-path ./fastMRI_data/multicoil_test_full --out-dir results/eval/v8fix_ou_s1_fullval_test_full --skip-precheck --num-workers 8
"""

import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import eval_v8fix_official_fb_ou as OE    # noqa: E402  등록 도우미 (import 부작용 없음 — 래퍼·torch 를 불러오지 않는다)


def _load_launcher_fb():
    spec = importlib.util.spec_from_file_location('eval_v8fix_fb', os.path.join(_HERE, 'eval_v8fix_fb.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['eval_v8fix_fb'] = m
    spec.loader.exec_module(m)            # 그 안에서 eval_v8fix_official_fb 를 불러온다 (FBL.FB)
    return m


FBL = _load_launcher_fb()
FB = FBL.FB
LAUNCHER_TARGETS_OU = dict(FBL.LAUNCHER_TARGETS)
LAUNCHER_TARGETS_OU.update(OE.OU_MODEL_TARGETS)
LAUNCHER_TARGETS_OU.update(OE.OU_UNET_TARGETS)
OU_MODEL_LABELS = frozenset(f'{c}.{m}' for items in OE.OU_MODEL_TARGETS.values() for c, m in items)
OU_UNET_LABELS = frozenset(f'{c}.{m}' for items in OE.OU_UNET_TARGETS.values() for c, m in items)
FBL.MODEL_LABELS = FBL.MODEL_LABELS | OU_MODEL_LABELS     # _unit_handler 가 부를 때마다 모듈 전역으로 읽는다
FBL.UNET_LABELS = FBL.UNET_LABELS | OU_UNET_LABELS


def _finalize(completed, ns, outputs, out_dir, seq, defect):
    """래퍼의 _finalize 그대로 + OU bi-GRU 모델의 시퀀스 모듈 단위 점검 (래퍼는 seq ∈ {gru, ss2d} 에만 한다)."""
    res = FBL._finalize(completed=completed, ns=ns, outputs=outputs, out_dir=out_dir, seq=seq, defect=defect)
    passes = FBL.FUNC['passes']
    if completed and passes and seq in OE.OU_GRU_NAMES and passes[-1]['calls_seq'] == 0:
        res['failures'].append(f'functional test of {seq} had no wrapped sequence-module unit (PureETER_GRU._seq): '
                               'the slice-level rule was not applied to the bi-GRU')
    return res


def main():
    seq = FBL._seq_from_argv(sys.argv[1:])

    def pre():
        OE.install_seq_choices()
        if seq == 'gru' or seq in OE.OU_GRU_NAMES:
            FBL._gru_fp32()               # eval_v8fix_grufp32.py 와 같은 순서 (경로 추가 → gru_precision → install_class_patch)

    FB.run(FBL.TARGET, LAUNCHER_TARGETS_OU, mode='launcher', wrapper_files=[__file__, FBL.__file__, FB.__file__, OE.__file__],
           pre=pre, unit_handler=FBL._unit_handler,
           post_import={'smoke_recipe_fix': [OE.register_recipe, FBL._patch_run_validation]}, finalize=_finalize)


if __name__ == '__main__':
    main()
