"""
원래 U-Net(OU) 세 모델(unet_ou · gru_ou · gruo_ou)을 수정 3 의 fp32 재계산 규칙 공식 평가 래퍼(eval_v8fix_official_fb.py)로 평가하는
진입점 (2026-10-26) — 계획 paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan.md §3.

기존 파일(eval_v8fix_official.py · eval_v8fix_official_fb.py · smoke_recipe_fix.py 등)은 고치지 않는다. 이 파일은 같은 프로세스 안에서만:
  1. eval_v8fix_official_fb.py 를 모듈로 불러(eval_v8fix_fb.py 가 하는 것과 같은 방식) 그 run() 으로 원 스크립트
     eval_v8fix_official.py 를 runpy 실행한다 — fp32 재계산 규칙·기록(fp32_fallback_<seq>.json)·종료 코드·시험 후크는 그 파일 그대로다.
  2. 감쌀 대상(OFFICIAL_TARGETS)에 OU 모델 클래스의 forward 세 개를 더한다 (u_pure_eternet_ou: PureETER_UNETOU · PureETER_GRUOU ·
     PureETER_GRUOOU). 규칙의 단위는 기존 세 모델과 같이 슬라이스마다 모델 전체 forward 하나다.
  3. 원 스크립트의 --seq 선택지에 OU 이름을 더한다 (install_seq_choices — argparse add_argument 에서, 선택지에 unet·gru·ss2d 가 모두
     있는 --seq 인자 한 곳만 바꾼다. 그 스크립트에 전역 SPECS(eval_v8fix.py 의 이름·요약 제목 표)가 있으면 OU 이름·제목도 더한다).
  4. 원 스크립트가 smoke_recipe_fix 를 import 하는 순간(래퍼의 post_import 후크) recipe_ou.register 로 R.build_model · R.run_validation 이
     OU 이름도 받게 한다. gru_ou · gruo_ou 는 모델을 만들기 직전에 gru_precision.install_class_patch(GRU_FP32_MODULES, 기본 gru_h) —
     원 스크립트 build() 가 --seq gru 에 하는 것과 같은 함수·같은 자리(R.build_model 직전)다.
기존 이름(unet · gru · ss2d)으로 부르면 이 진입점이 더한 것은 아무 일도 하지 않는다(원 래퍼와 같은 실행; 기록의 wrapper 칸에 이 파일이
더 적힐 뿐이다). 이 모듈을 import 만 하면 아무것도 하지 않는다(eval_v8fix_fb_ou.py 가 아래 도우미를 쓴다).

실행 (인자는 eval_v8fix_official.py 와 같다):
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/eval_v8fix_official_fb_ou.py --seq gru_ou --tf32 off \\
      --ckpt logs/PureETER_GRUOU_noDC_R4_brain384_v8fix_s1_fullval/pure_gru_ou_best.pt \\
      --data-path ./fastMRI_data/multicoil_test_full --out-dir <새 폴더> --num-workers 4
  --tf32 는 기존 런과 같은 규칙으로 고른다(eval_v8fix_official.py 주석: U-Net only·SS2D default, ETER-net off).
"""

import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# eval_v8fix.py 의 SPECS 꼴 (요약의 모델 이름·제목). 이름은 recipe_ou.MODELS 와 같아야 한다(register_recipe 에서 확인).
OU_SPECS = {
    'unet_ou': dict(name='U-Net only (OU)', title='U-Net only (OU) — 원래 U-Net 정제망, 시퀀스 모듈 없음 — 레시피 DL'),
    'gru_ou': dict(name='d-ETER (OU)', title='d-ETER (OU) — bi-GRU 출력 + zero-filled 영상 → 원래 U-Net — 레시피 DL'),
    'gruo_ou': dict(name='o-ETER (OU)', title='o-ETER (OU) — bi-GRU 출력만 → 원래 U-Net (2020 ETER-net 배선) — 레시피 DL'),
}
OU_NAMES = tuple(OU_SPECS)
OU_GRU_NAMES = ('gru_ou', 'gruo_ou')
BASE_SEQ_CHOICES = frozenset({'unet', 'gru', 'ss2d'})
# 모듈 이름 → [(클래스, 메서드)] (eval_v8fix_official_fb 의 targets 꼴)
OU_MODEL_TARGETS = {'u_pure_eternet_ou': [('PureETER_UNETOU', 'forward'), ('PureETER_GRUOU', 'forward'),
                                          ('PureETER_GRUOOU', 'forward')]}
OU_UNET_TARGETS = {'unet_original': [('OriginalUNet', 'forward')]}      # 런처 평가 기능 시험의 U-Net 단위 (eval_v8fix_fb_ou.py)


def install_seq_choices():
    """argparse 의 add_argument 를 감싸 --seq 선택지(unet·gru·ss2d 를 모두 포함한 것)에 OU 이름을 더한다. 부른 곳의 전역에
    SPECS(unet·gru·ss2d 키를 가진 dict — eval_v8fix.py)가 있으면 OU 이름·제목을 더한다(setdefault — 기존 항목은 그대로).
    다른 인자·다른 파서는 그대로다. 이미 감쌌으면 False."""
    import argparse
    orig = argparse._ActionsContainer.add_argument
    if getattr(orig, '_ou_seq_choices', False):
        return False

    def add_argument(self, *args, **kwargs):
        ch = kwargs.get('choices')
        if '--seq' in args and ch is not None and BASE_SEQ_CHOICES <= set(ch):
            specs = sys._getframe(1).f_globals.get('SPECS')
            if isinstance(specs, dict) and BASE_SEQ_CHOICES <= set(specs):
                for n, s in OU_SPECS.items():
                    specs.setdefault(n, dict(s))
            kwargs['choices'] = list(ch) + [n for n in OU_NAMES if n not in ch]
        return orig(self, *args, **kwargs)

    add_argument._ou_seq_choices = True
    add_argument._ou_orig = orig
    argparse._ActionsContainer.add_argument = add_argument
    return True


def register_recipe(module):
    """래퍼의 post_import 후크 (원 스크립트가 smoke_recipe_fix 를 import 한 직후): recipe_ou.register(module, gru_fp32=True)."""
    if _HERE not in sys.path:
        sys.path.append(_HERE)
    import recipe_ou
    if set(recipe_ou.OU_NAMES) != set(OU_NAMES) or set(recipe_ou.OU_GRU_NAMES) != set(OU_GRU_NAMES):
        raise RuntimeError(f'OU 이름 불일치: recipe_ou {recipe_ou.OU_NAMES}/{recipe_ou.OU_GRU_NAMES} vs 진입점 {OU_NAMES}/{OU_GRU_NAMES}')
    return recipe_ou.register(module, gru_fp32=True)


def load_official_fb():
    """eval_v8fix_official_fb.py 를 모듈로 불러온다 (eval_v8fix_fb.py 와 같은 방식, 같은 sys.modules 이름)."""
    if 'eval_v8fix_official_fb' in sys.modules:
        return sys.modules['eval_v8fix_official_fb']
    spec = importlib.util.spec_from_file_location('eval_v8fix_official_fb', os.path.join(_HERE, 'eval_v8fix_official_fb.py'))
    fb = importlib.util.module_from_spec(spec)
    sys.modules['eval_v8fix_official_fb'] = fb
    spec.loader.exec_module(fb)
    return fb


def main():
    fb = load_official_fb()
    targets = dict(fb.OFFICIAL_TARGETS)
    targets.update(OU_MODEL_TARGETS)
    fb.run(fb.OFFICIAL_SCRIPT, targets, mode='official', wrapper_files=[__file__, fb.__file__],
           pre=install_seq_choices, post_import={'smoke_recipe_fix': [register_recipe]})


if __name__ == '__main__':
    main()
