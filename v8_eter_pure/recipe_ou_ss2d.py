"""
원래 U-Net(OU) 판 SS2D 모델을 recipe_ou(OU 이름 등록부)에 더한다 (2026-10-26) — 계획
paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan_amend1_ss2d.md (학습 전에 고정).

recipe_ou.py 는 고치지 않는다(실행 중인 OU 런이 재시작하면 그 파일을 다시 읽는다). 이 모듈을 import 하면 같은 프로세스 안에서만:
  - recipe_ou.MODELS(같은 OrderedDict 객체)에 SS2D 이름 넷을 더한다. recipe_ou 의 함수(base_seq·has_seq·run_name·unet_ref_name·
    partial_unet_copy·build_model·build_groups·copy_unet_init·unet_init_note·run_validation·register)는 MODELS 를 부를 때마다 읽으므로
    새 이름도 그대로 받는다. import 할 때 만든 튜플(recipe_ou.OU_NAMES·OU_GRU_NAMES)은 바꾸지 않는다(기존 진입점의 이름 대조가 그것을 쓴다).
      ss2d_ou   = d-SS2D (OU)        → logs/PureETER_SS2DOU_noDC_R4_brain384_v8fix{RUN_SUFFIX}/pure_ss2d_ou_*.pt
      ss2do_ou  = o-SS2D (OU)        → logs/PureETER_SS2DOOU_…   (짧은 비교 시험 전용)
      ss2dfo_ou = o-SS2D-F (OU)      → logs/PureETER_SS2DFOOU_…
      linfo_ou  = o-LIN-F (OU) (대조) → logs/PureETER_LINFOOU_…   (짧은 비교 시험 전용)
  - recipe_ou._build_ou 를 감싸 새 이름을 config 인자(R.C: N_COIL·N_HIDDEN_LRNN_2·UNET_DEPTH·UNET_WF·SS2D_D_INNER·SS2D_D_STATE —
    R.build_model 의 ss2d 분기와 같은 인자)로 만든다. 기존 이름은 원 함수 그대로.
  - recipe_ou.functional_target 를 감싸 o- SS2D 이름(seq_only_input)에 SS2D 용 대리 객체를 넘긴다 — R.run_validation 은 seq 'ss2d' 의
    시퀀스 출력을 model.ss2d(x_ksp) 로 얻고 정제망 입력을 cat(시퀀스 출력 20, zero-filled 32) 로 만든다. 대리 객체의 U-Net 은 앞
    20 채널만 실제 U-Net 에 넘긴다(recipe_ou._SeqOnlyUNetView 그대로). 그래서 정상 출력 = model.forward(분해 점검), zero = 0 입력 U-Net,
    shuffle = 짝 슬라이스의 시퀀스 출력만으로 만든 출력이다. 해석 주의(recipe_ou 의 o-ETER 와 같음): o- 모델의 하락은 '출력이 그
    슬라이스의 k-space 에 의존하는가' 를 잰다.
  - build_groups: 바탕 이름 'ss2d' 로 R.build_groups 를 부른다(recipe_ou.build_groups 그대로). FourierSS2D 는 SS2D 를 상속해 하위 모듈
    이름이 같으므로(ss2d.norm_in·…·ss2d.out_proj) 그룹이 그대로 맞고, 헤드의 선형 경로(ss2d.res)는 'seq' 그룹에 든다.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import recipe_ou as RO          # noqa: E402  (smoke_recipe_fix 를 import 한다 — 레시피의 단일 출처)

C = RO.C

SS2D_MODELS = (
    ('ss2d_ou', dict(tag='SS2DOU', base='ss2d', cls='PureETER_SS2DOU', seq_only_input=False, label='d-SS2D (OU)')),
    ('ss2do_ou', dict(tag='SS2DOOU', base='ss2d', cls='PureETER_SS2DOOU', seq_only_input=True, label='o-SS2D (OU)')),
    ('ss2dfo_ou', dict(tag='SS2DFOOU', base='ss2d', cls='PureETER_SS2DFOOU', seq_only_input=True, label='o-SS2D-F (OU)')),
    ('linfo_ou', dict(tag='LINFOOU', base='ss2d', cls='PureETER_LINFOOU', seq_only_input=True, label='o-LIN-F (OU)')),
)
SS2D_OU_NAMES = tuple(n for n, _ in SS2D_MODELS)
TRAIN_NAMES = ('ss2d_ou', 'ss2dfo_ou')          # 50 epoch 학습 대상 (나머지 둘은 짧은 비교 시험 전용)

for _n, _m in SS2D_MODELS:
    assert _m['tag'] == _n.replace('_', '').upper(), (_n, _m['tag'])     # 런 폴더 표시 규칙 = pod_run_ou_ss2d.sh 와 같음
    assert _n not in RO.DFU_NAMES
    if _n in RO.MODELS:
        assert RO.MODELS[_n] == _m, f'recipe_ou.MODELS[{_n!r}] 이 이미 다른 값으로 있음'
    else:
        RO.MODELS[_n] = dict(_m)
del _n, _m

_orig_build_ou = getattr(RO._build_ou, '_ss2d_orig', RO._build_ou)
_orig_functional_target = getattr(RO.functional_target, '_ss2d_orig', RO.functional_target)


def _build_ou(name, device):
    if name not in SS2D_OU_NAMES:
        return _orig_build_ou(name, device)
    import u_pure_eternet_ou_ss2d as M      # models/pure_eternet (R 이 sys.path 에 넣음)
    common = dict(n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2, unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF)
    cls = getattr(M, RO.MODELS[name]['cls'])
    if name == 'linfo_ou':
        model = cls(**common)
    else:
        model = cls(**common, ss2d_d_inner=C.SS2D_D_INNER, ss2d_d_state=C.SS2D_D_STATE)
    assert type(model).__name__ == RO.MODELS[name]['cls'], (name, type(model).__name__)
    return model.to(device)


class SS2DSeqOnlyInputView(RO.SeqOnlyInputView):
    """R.run_validation 에 seq='ss2d' 로 넘기는 o- SS2D 모델의 대리 객체 — recipe_ou.SeqOnlyInputView 에 .ss2d 를 더한 것."""

    def __init__(self, model):
        super().__init__(model)
        self.ss2d = model.ss2d


def functional_target(model, name):
    if name in SS2D_OU_NAMES:
        if RO.MODELS[name]['seq_only_input']:
            return SS2DSeqOnlyInputView(model), 'ss2d'
        return model, 'ss2d'
    return _orig_functional_target(model, name)


_build_ou._ss2d_orig = _orig_build_ou
functional_target._ss2d_orig = _orig_functional_target
RO._build_ou = _build_ou                    # recipe_ou.build_model·register 의 build_model 이 부를 때 모듈 전역으로 읽는다
RO.functional_target = functional_target    # recipe_ou.run_validation·register 의 run_validation 이 같은 방식으로 읽는다
