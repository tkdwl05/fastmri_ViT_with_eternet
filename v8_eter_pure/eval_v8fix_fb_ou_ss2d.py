"""
원래 U-Net(OU) 판 SS2D 모델을 런처 평가(eval_v8fix.py: per-slice CSV + 기능 시험)로, 수정 3 의 fp32 재계산 규칙 래퍼(eval_v8fix_fb.py)를
그대로 써서 평가하는 진입점 (2026-10-26). infra/runpod/pod_run_ou_ss2d.sh 가 학습 뒤에 부른다. 계획 보충
paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan_amend1_ss2d.md (기능 시험: 시드 1 과 같은 512 슬라이스).
eval_v8fix_fb_ou.py(OU 세 모델 진입점)와 같은 방식이고 기존 파일은 고치지 않는다. 같은 프로세스 안에서만:
  1. eval_v8fix_fb.py 를 모듈로 불러 그 기계(_unit_handler · _patch_run_validation · _finalize)를 그대로 쓴다.
  2. 감쌀 대상에 OU 세 모델·SS2D OU 네 모델의 forward(모델 전체 — 기능 시험에서는 그릇)와 OriginalUNet.forward(기능 시험의 U-Net 단위)를
     더하고, 래퍼의 라벨 집합(MODEL_LABELS · UNET_LABELS)에 같은 이름을 더한다. 시퀀스 모듈 단위는 래퍼 그대로 SS2D.forward 다 —
     d-SS2D (OU)·o-SS2D (OU) 의 SS2D, o-SS2D-F (OU) 의 FourierSS2D 본체(SS2D 를 상속; 헤드의 선형 경로·변환은 fp32 로 그 뒤에 계산).
     호출의 시험 이름은 run_validation 의 현재 줄로 정하므로(eval_v8fix_fb._test_of) 헤드가 한 겹 감싸도 'seq' 로 잡힌다.
     o-LIN-F (OU) 는 SS2D 본체가 없어 시퀀스 모듈 단위가 없다(선형 경로·변환만, fp32).
  3. eval_v8fix.py 의 --seq 선택지와 SPECS 에 SS2D 이름을 더하고(eval_v8fix_official_fb_ou_ss2d.extend_seq_choices), 원 스크립트가
     smoke_recipe_fix 를 import 하는 순간 recipe_ou_ss2d·recipe_ou.register 로 R.build_model · R.run_validation 이 SS2D OU 이름을 받게
     한다 — 래퍼의 run_validation 감싸기(_patch_run_validation)보다 먼저.
  4. 점검 하나를 더한다: SS2D 본체가 있는 모델(ss2d_ou · ss2do_ou · ss2dfo_ou)의 기능 시험에 시퀀스 모듈 단위 호출이 없으면 점검 실패
     (래퍼가 gru · ss2d 에 하는 점검과 같음).
실행 (인자는 eval_v8fix.py 와 같다. --seq 는 줄여 쓰지 않는다. 출력 폴더는 새 폴더로):
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/eval_v8fix_fb_ou_ss2d.py --seq ss2d_ou \\
      --ckpt logs/PureETER_SS2DOU_noDC_R4_brain384_v8fix_s1_fullval/pure_ss2d_ou_best.pt \\
      --data-path ./fastMRI_data/multicoil_test_full --out-dir results/eval/v8fix_ou_s1_fullval_test_full --skip-precheck --num-workers 8
"""

import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import eval_v8fix_official_fb_ou as OE          # noqa: E402  등록 도우미 (import 부작용 없음)
import eval_v8fix_official_fb_ou_ss2d as OES    # noqa: E402  SS2D 이름·대상 (import 부작용 없음)


def _load_launcher_fb():
    spec = importlib.util.spec_from_file_location('eval_v8fix_fb', os.path.join(_HERE, 'eval_v8fix_fb.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['eval_v8fix_fb'] = m
    spec.loader.exec_module(m)            # 그 안에서 eval_v8fix_official_fb 를 불러온다 (FBL.FB)
    return m


FBL = _load_launcher_fb()
FB = FBL.FB

LAUNCHER_TARGETS_OU_SS2D = dict(FBL.LAUNCHER_TARGETS)
LAUNCHER_TARGETS_OU_SS2D.update(OE.OU_MODEL_TARGETS)
LAUNCHER_TARGETS_OU_SS2D.update(OE.OU_UNET_TARGETS)
LAUNCHER_TARGETS_OU_SS2D.update(OES.SS2D_OU_MODEL_TARGETS)
_MODEL_LABELS_ADD = frozenset(f'{c}.{m}' for T in (OE.OU_MODEL_TARGETS, OES.SS2D_OU_MODEL_TARGETS)
                              for items in T.values() for c, m in items)
_UNET_LABELS_ADD = frozenset(f'{c}.{m}' for items in OE.OU_UNET_TARGETS.values() for c, m in items)
FBL.MODEL_LABELS = FBL.MODEL_LABELS | _MODEL_LABELS_ADD     # _unit_handler 가 부를 때마다 모듈 전역으로 읽는다
FBL.UNET_LABELS = FBL.UNET_LABELS | _UNET_LABELS_ADD


def _finalize(completed, ns, outputs, out_dir, seq, defect):
    """래퍼의 _finalize 그대로 + SS2D 본체가 있는 OU 모델의 시퀀스 모듈 단위 점검 (래퍼는 seq ∈ {gru, ss2d} 에만 한다)."""
    res = FBL._finalize(completed=completed, ns=ns, outputs=outputs, out_dir=out_dir, seq=seq, defect=defect)
    passes = FBL.FUNC['passes']
    if completed and passes and seq in OES.SS2D_CORE_NAMES and passes[-1]['calls_seq'] == 0:
        res['failures'].append(f'functional test of {seq} had no wrapped sequence-module unit (SS2D.forward): '
                               'the slice-level rule was not applied to the SS2D')
    return res


def main():
    FB.run(FBL.TARGET, LAUNCHER_TARGETS_OU_SS2D, mode='launcher',
           wrapper_files=[__file__, FBL.__file__, FB.__file__, OE.__file__, OES.__file__],
           pre=OES.extend_seq_choices, unit_handler=FBL._unit_handler,
           post_import={'smoke_recipe_fix': [OES.register_recipe, FBL._patch_run_validation]}, finalize=_finalize)


if __name__ == '__main__':
    main()
