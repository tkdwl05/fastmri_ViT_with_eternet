"""
원래 U-Net(OU) 판 SS2D 모델(ss2d_ou · ss2do_ou · ss2dfo_ou · linfo_ou)을 수정 3 의 fp32 재계산 규칙 공식 평가 래퍼
(eval_v8fix_official_fb.py)로 평가하는 진입점 (2026-10-26) — 계획 보충
paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan_amend1_ss2d.md.
eval_v8fix_official_fb_ou.py(OU 세 모델 진입점)와 같은 방식이고, 그 파일·recipe_ou.py 를 포함한 기존 파일은 고치지 않는다. 같은 프로세스
안에서만:
  1. eval_v8fix_official_fb.py 를 모듈로 불러(eval_v8fix_official_fb_ou.load_official_fb) 그 run() 으로 원 스크립트 eval_v8fix_official.py
     를 runpy 실행한다 — fp32 재계산 규칙·기록·종료 코드·시험 후크는 그 파일 그대로다.
  2. 감쌀 대상에 OU 세 모델 클래스(eval_v8fix_official_fb_ou.OU_MODEL_TARGETS)와 SS2D OU 네 클래스(u_pure_eternet_ou_ss2d:
     PureETER_SS2DOU · PureETER_SS2DOOU · PureETER_SS2DFOOU · PureETER_LINFOOU)의 forward 를 더한다(슬라이스마다 모델 전체 forward 하나).
  3. 원 스크립트의 --seq 선택지와 SPECS 에 SS2D 이름을 더한다 — eval_v8fix_official_fb_ou.install_seq_choices 가 읽는 그 모듈의
     OU_SPECS·OU_NAMES 에 이 프로세스 안에서만 SS2D 이름을 더한 뒤 그것을 부른다.
  4. 원 스크립트가 smoke_recipe_fix 를 import 하는 순간 recipe_ou_ss2d(SS2D 이름을 recipe_ou 등록부에 더함)를 import 하고
     recipe_ou.register 로 R.build_model · R.run_validation 이 OU·SS2D OU 이름을 받게 한다. SS2D 는 gru_h 패치가 필요 없다.
--tf32 는 SS2D(controlled) 와 같이 default (eval_v8fix_official.py 주석: U-Net only·SS2D default, ETER-net off).
실행 (인자는 eval_v8fix_official.py 와 같다):
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/eval_v8fix_official_fb_ou_ss2d.py --seq ss2d_ou --tf32 default \\
      --ckpt logs/PureETER_SS2DOU_noDC_R4_brain384_v8fix_s1_fullval/pure_ss2d_ou_best.pt \\
      --data-path ./fastMRI_data/multicoil_test_full --out-dir <새 폴더> --num-workers 4
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import eval_v8fix_official_fb_ou as OE    # noqa: E402  등록 도우미 (import 부작용 없음)

SS2D_OU_SPECS = {
    'ss2d_ou': dict(name='d-SS2D (OU)', title='d-SS2D (OU) — SS2D 출력 + zero-filled 영상 → 원래 U-Net — 레시피 DL'),
    'ss2do_ou': dict(name='o-SS2D (OU)', title='o-SS2D (OU) — SS2D 출력만 → 원래 U-Net (짧은 비교 시험) — 레시피 DL'),
    'ss2dfo_ou': dict(name='o-SS2D-F (OU)', title='o-SS2D-F (OU) — SS2D + 푸리에 출력 헤드 → 원래 U-Net — 레시피 DL'),
    'linfo_ou': dict(name='o-LIN-F (OU)', title='o-LIN-F (OU) — 선형 경로 + 역푸리에 변환 → 원래 U-Net (대조) — 레시피 DL'),
}
SS2D_OU_NAMES = tuple(SS2D_OU_SPECS)
SS2D_CORE_NAMES = ('ss2d_ou', 'ss2do_ou', 'ss2dfo_ou')        # SS2D 본체(SS2D.forward)가 있는 모델 — 기능 시험 단위 점검용
SS2D_OU_MODEL_TARGETS = {'u_pure_eternet_ou_ss2d': [('PureETER_SS2DOU', 'forward'), ('PureETER_SS2DOOU', 'forward'),
                                                    ('PureETER_SS2DFOOU', 'forward'), ('PureETER_LINFOOU', 'forward')]}


def extend_seq_choices():
    """OE.install_seq_choices 가 읽는 OE.OU_SPECS · OE.OU_NAMES 에 SS2D 이름을 더하고(이 프로세스 안에서만) 그것을 부른다."""
    for n, s in SS2D_OU_SPECS.items():
        OE.OU_SPECS.setdefault(n, dict(s))
    OE.OU_NAMES = tuple(OE.OU_SPECS)
    return OE.install_seq_choices()


def register_recipe(module):
    """래퍼의 post_import 후크: recipe_ou_ss2d 를 import(등록)한 뒤 recipe_ou.register(module, gru_fp32=True)."""
    import recipe_ou
    import recipe_ou_ss2d
    if set(recipe_ou_ss2d.SS2D_OU_NAMES) != set(SS2D_OU_NAMES):
        raise RuntimeError(f'SS2D OU 이름 불일치: recipe_ou_ss2d {recipe_ou_ss2d.SS2D_OU_NAMES} vs 진입점 {SS2D_OU_NAMES}')
    for n in SS2D_OU_NAMES:
        if n not in recipe_ou.MODELS:
            raise RuntimeError(f'recipe_ou.MODELS 에 {n} 이 없음 (recipe_ou_ss2d 등록 실패)')
    return recipe_ou.register(module, gru_fp32=True)


def main():
    fb = OE.load_official_fb()
    targets = dict(fb.OFFICIAL_TARGETS)
    targets.update(OE.OU_MODEL_TARGETS)
    targets.update(SS2D_OU_MODEL_TARGETS)
    fb.run(fb.OFFICIAL_SCRIPT, targets, mode='official', wrapper_files=[__file__, OE.__file__, fb.__file__],
           pre=extend_seq_choices, post_import={'smoke_recipe_fix': [register_recipe]})


if __name__ == '__main__':
    main()
