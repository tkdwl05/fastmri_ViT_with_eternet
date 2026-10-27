"""
FSSM (OU) 세 모델을 recipe_ou(OU 이름 등록부)에 더한다 (2026-10-27) — 설계 docs/fssm_design_2026-10-27.md (결과 전 고정).

recipe_ou.py·recipe_ou_ss2d.py 는 고치지 않는다(실행 중인 OU 런이 재시작하면 그 파일을 다시 읽는다). 이 모듈을 import 하면 같은
프로세스 안에서만:
  - recipe_ou_ss2d 를 import 한다(SS2D 이름 등록 유지 — 그 위에 감싼다).
  - recipe_ou.MODELS 에 FSSM 이름 셋을 더한다(바탕 이름 'ss2d', seq_only_input=True):
      fssm_ou  = FSSM (OU)            → logs/PureETER_FSSMOU_noDC_R4_brain384_v8fix{RUN_SUFFIX}/pure_fssm_ou_*.pt
      fssmr_ou = FSSM-real (OU) (절제) → logs/PureETER_FSSMROU_…
      fssmt_ou = FSSM-T (OU) (절제)    → logs/PureETER_FSSMTOU_…
    시퀀스 모듈을 model.ss2d 에 두므로 R.build_groups(바탕 'ss2d')의 seq 그룹·기능 시험(model.ss2d(x_ksp))·평가 래퍼가 그대로 맞는다.
  - recipe_ou._build_ou 를 감싸 새 이름을 config 인자(R.C: N_COIL·UNET_DEPTH·UNET_WF·IMAGE_SIZE)로 만든다.
  - recipe_ou.functional_target 을 감싸 FSSM 이름에 recipe_ou_ss2d.SS2DSeqOnlyInputView 를 넘긴다(대리 객체의 U-Net 은 앞
    seq_out_ch = 32 채널만 받음). 해석 주의: FSSM 의 시퀀스 출력에는 정확한 IFT 경로(zero-filled 영상)가 들어 있어 shuffle·zero 시험은
    '입력 영상 교체'가 된다 — 시퀀스 모듈 기여는 보정 끄기 시험(out_proj = 0, branch_off_fssm.py)으로 잰다.
  - recipe_ou.copy_unet_init 을 감싼다(트레이너 SAME_UNET_INIT): FSSM 모델이면 모양이 같은 텐서를 복사하고, 첫 합성곱 weight
    (64, 32, 3, 3) 는 기준 U-Net only (OU) 의 (64, 52, 3, 3) 에서 [:, 20:52](zero-filled 영상 자리)를 복사한다. 기준의 앞 20 채널은
    항상 0 을 받으므로 초기 모델 = U-Net only (OU) 초기 모델(FSSM 초기 출력 = x_img). recipe_ou.unet_init_note 도 감싸 그 조각 복사를 적는다.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import recipe_ou_ss2d as ROS    # noqa: E402  (SS2D 이름 등록 — recipe_ou 를 import 한다)

RO = ROS.RO
C = RO.C

FSSM_MODELS = (
    ('fssm_ou', dict(tag='FSSMOU', base='ss2d', cls='PureETER_FSSMOU', seq_only_input=True, label='FSSM (OU)')),
    ('fssmr_ou', dict(tag='FSSMROU', base='ss2d', cls='PureETER_FSSMROU', seq_only_input=True, label='FSSM-real (OU)')),
    ('fssmt_ou', dict(tag='FSSMTOU', base='ss2d', cls='PureETER_FSSMTOU', seq_only_input=True, label='FSSM-T (OU)')),
)
FSSM_NAMES = tuple(n for n, _ in FSSM_MODELS)
FSSM_CLASSES = tuple(m['cls'] for _, m in FSSM_MODELS)
SEQ_CH_REF = int(C.N_HIDDEN_LRNN_2) * 2          # 기준 U-Net only (OU) 입력의 앞 0 채널 수 (= 20)

for _n, _m in FSSM_MODELS:
    assert _m['tag'] == _n.replace('_', '').upper(), (_n, _m['tag'])     # 런 폴더 표시 규칙 = pod_run_ou.sh 와 같음
    assert _n not in RO.DFU_NAMES
    if _n in RO.MODELS:
        assert RO.MODELS[_n] == _m, f'recipe_ou.MODELS[{_n!r}] 이 이미 다른 값으로 있음'
    else:
        RO.MODELS[_n] = dict(_m)
del _n, _m

_orig_build_ou = getattr(RO._build_ou, '_fssm_orig', RO._build_ou)
_orig_functional_target = getattr(RO.functional_target, '_fssm_orig', RO.functional_target)
_orig_copy_unet_init = getattr(RO.copy_unet_init, '_fssm_orig', RO.copy_unet_init)
_orig_unet_init_note = getattr(RO.unet_init_note, '_fssm_orig', RO.unet_init_note)


def _build_ou(name, device):
    if name not in FSSM_NAMES:
        return _orig_build_ou(name, device)
    import u_pure_eternet_fssm as M        # models/pure_eternet (R 이 sys.path 에 넣음)
    cls = getattr(M, RO.MODELS[name]['cls'])
    model = cls(n_coil=C.N_COIL, unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF, pe_len=int(C.IMAGE_SIZE[-1]))
    assert type(model).__name__ == RO.MODELS[name]['cls'], (name, type(model).__name__)
    return model.to(device)


def functional_target(model, name):
    if name in FSSM_NAMES:
        return ROS.SS2DSeqOnlyInputView(model), 'ss2d'
    return _orig_functional_target(model, name)


def copy_unet_init(model, ref, partial=False):
    """recipe_ou.copy_unet_init 와 같고, FSSM 모델이면 첫 합성곱 weight 를 기준의 영상 채널 자리 [:, 20:52] 에서 복사한다.
    반환 rec 에 sliced=[{name, shape, ref_shape, ref_slice}] 를 더하고 own_init 은 비운다(조각 복사는 '복사'로 센다)."""
    if type(model).__name__ not in FSSM_CLASSES:
        return _orig_copy_unet_init(model, ref, partial=partial)
    import torch
    rec = _orig_copy_unet_init(model, ref, partial=True)
    src = ref.unet.state_dict()
    params = dict(model.unet.named_parameters())
    c0 = int(getattr(ref, 'seq_out_ch', SEQ_CH_REF))        # 기준의 0 채널 수 (U-Net only (OU) = 20)
    c_img = int(model.seq_out_ch)                           # FSSM 출력 = 16코일 영상 32 채널
    sliced = []
    for o in rec['own_init']:
        k = o['name'][len('unet.'):]
        shp, rshp = tuple(o['shape']), tuple(o['ref_shape'])
        ok = (len(shp) == 4 and len(rshp) == 4 and shp[0] == rshp[0] and shp[2:] == rshp[2:]
              and shp[1] == c_img and rshp[1] == c0 + c_img)
        if not ok:
            raise AssertionError(f'FSSM U-Net 초기값: 조각 복사로 맞출 수 없는 텐서 {o} (기대: 첫 합성곱 (·, {c_img}, ·, ·) vs '
                                 f'기준 (·, {c0 + c_img}, ·, ·))')
        with torch.no_grad():
            params[k].copy_(src[k][:, c0:c0 + c_img])
        sliced.append(dict(name=o['name'], shape=list(shp), ref_shape=list(rshp), ref_slice=f'[:, {c0}:{c0 + c_img}]'))
    if len(sliced) != 1:
        raise AssertionError(f'FSSM U-Net 초기값: 조각 복사 대상이 첫 합성곱 weight 하나여야 함 (받은 것 {sliced})')
    return dict(rec, n_copied=rec['n_copied'] + len(sliced), own_init=[], sliced=sliced)


def unet_init_note(name, rec):
    s = _orig_unet_init_note(name, rec)
    if name in FSSM_NAMES and rec.get('sliced'):
        extra = '; '.join(f'{o["name"]} {tuple(o["shape"])} = ref{o["ref_slice"]} of {tuple(o["ref_shape"])} '
                          f'(zero-filled image channels — the ref channels before them always receive 0)'
                          for o in rec['sliced'])
        s = (s[:-1] + '; ' + extra + ')') if s.endswith(')') else (s + ' — ' + extra)
    return s


_build_ou._fssm_orig = _orig_build_ou
functional_target._fssm_orig = _orig_functional_target
copy_unet_init._fssm_orig = _orig_copy_unet_init
unet_init_note._fssm_orig = _orig_unet_init_note
RO._build_ou = _build_ou                    # recipe_ou.build_model·register 가 모듈 전역으로 읽는다
RO.functional_target = functional_target    # recipe_ou.run_validation·register 가 같은 방식으로 읽는다
RO.copy_unet_init = copy_unet_init          # 트레이너가 RO.copy_unet_init 으로 부른다 (호출 때 모듈 속성으로 읽음)
RO.unet_init_note = unet_init_note          # 트레이너가 RO.unet_init_note 로 부른다
