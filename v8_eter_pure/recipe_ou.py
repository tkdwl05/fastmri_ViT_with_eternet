"""
원래 U-Net(OU) 판 세 모델을 v8fix 레시피에 더한다 (2026-10-26) — 계획
paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan.md (학습 전에 고정).

레시피의 상수·함수는 smoke_recipe_fix.py(v8fix 의 단일 출처)에서 import 만 한다 — 여기에 옮겨 적은 레시피 상수는 없다
(config 는 R.C, 정규화·옵티마이저·건강 감시 그룹·기능 시험은 R.* 그대로). 이 파일이 더하는 것:
  - 이름 (SEQ_MODEL / 런 폴더 표시 / 체크포인트):
      unet_ou = U-Net only (OU) → logs/PureETER_UNETOU_noDC_R4_brain384_v8fix{RUN_SUFFIX}/pure_unet_ou_*.pt
      gru_ou  = d-ETER (OU)     → logs/PureETER_GRUOU_noDC_R4_brain384_v8fix{RUN_SUFFIX}/pure_gru_ou_*.pt
      gruo_ou = o-ETER (OU)     → logs/PureETER_GRUOOU_noDC_R4_brain384_v8fix{RUN_SUFFIX}/pure_gruo_ou_*.pt
    런 폴더 표시 = 이름에서 '_' 를 빼고 대문자 (infra/runpod/pod_run_ou.sh 도 같은 규칙 — 아래 assert). 기존 런 폴더
    (PureETER_UNET·GRU·SS2D·…)와 겹치지 않는다.
  - build_model(name, device): OU 이름은 config 인자(R.C)로 만든다 — R.build_model 의 gru / unet 분기와 같은 인자(use_dc 없음).
    기존 이름(unet | gru | ss2d)은 R.build_model 그대로 (트레이너 동등성 시험용).
  - build_groups(model, name): R.build_groups 를 그 모델의 바탕 이름으로 (unet_ou → unet, gru_ou·gruo_ou → gru).
    원래 U-Net 은 DFU 와 하위 모듈 이름이 같아(down_path / up_path / last) 그룹 정의(DEEP_PREFIXES 포함)가 그대로 맞는다.
  - copy_unet_init: 트레이너의 SAME_UNET_INIT (seed_all(SEED) 직후 만든 기준 모델의 U-Net 가중치 복사). OU 모델의 기준 =
    U-Net only (OU). d-ETER (OU) 는 전부 복사(이름·모양이 하나라도 다르면 오류 — 기존 트레이너와 같은 전체 복사), o-ETER (OU) 는
    모양이 같은 텐서만 복사하고 다른 것(첫 합성곱 weight (64, 20, 3, 3) vs 기준 (64, 52, 3, 3))은 자기 초기값을 두고 그 목록을
    돌려준다(트레이너가 SCRATCH START 줄과 last.pt run_config.unet_init 에 적는다).
  - run_validation(model, name, ...): R.run_validation(기능 시험)을 그 모델에 맞게 부른다. o-ETER (OU) 는 대리 객체를 넘긴다 —
    R.run_validation 은 ETER-net 과 같은 U-Net 입력 cat(시퀀스 출력 20, zero-filled 32) 을 만들므로, 대리 객체의 U-Net 은 앞
    20 채널(시퀀스 출력)만 실제 U-Net 에 넘긴다. 그래서 정상 출력은 model.forward 와 같고(분해 점검 decomp_equiv_maxabs),
    zero = 0 입력 U-Net, shuffle_cross·shuffle_adj = 짝 슬라이스의 시퀀스 출력만으로 만든 출력, deep4·deep34 는 같은 정의다.
    해석 주의: o-ETER 는 시퀀스 출력이 U-Net 의 유일한 입력이라, 그 하락은 'zero-filled 영상과 함께 시퀀스 출력을 쓰는가' 가
    아니라 '출력이 그 슬라이스의 k-space 에 의존하는가' 를 잰다.
  - register(module): 평가 진입점(eval_v8fix_official_fb_ou.py, eval_v8fix_fb_ou.py)이 원 평가 스크립트가 smoke_recipe_fix 를
    import 하는 순간 부른다 — 그 모듈의 build_model·run_validation 이 OU 이름도 받게 감싼다(기존 이름은 원 함수 그대로).
    트레이너는 이것을 부르지 않는다(이 모듈의 함수를 직접 부름).
"""

import os
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)

import smoke_recipe_fix as R      # noqa: E402  레시피의 단일 출처 (import 부작용: sys.path·PYTORCH_CUDA_ALLOC_CONF 기본값뿐)

C = R.C                           # 공유 config (myConfig_pure_eter_v8 — R 과 같은 객체)

MODELS = OrderedDict([
    ('unet_ou', dict(tag='UNETOU', base='unet', cls='PureETER_UNETOU', seq_only_input=False, label='U-Net only (OU)')),
    ('gru_ou', dict(tag='GRUOU', base='gru', cls='PureETER_GRUOU', seq_only_input=False, label='d-ETER (OU)')),
    ('gruo_ou', dict(tag='GRUOOU', base='gru', cls='PureETER_GRUOOU', seq_only_input=True, label='o-ETER (OU)')),
])
OU_NAMES = tuple(MODELS)
OU_GRU_NAMES = tuple(n for n, m in MODELS.items() if m['base'] == 'gru')
DFU_NAMES = tuple(R.SEQS)                 # ('unet', 'gru', 'ss2d') — R.build_model 이 만드는 기존 모델
for _n, _m in MODELS.items():             # 런 폴더 표시 규칙 = pod_run_ou.sh 의 `tr -d '_' | tr a-z A-Z`
    assert _m['tag'] == _n.replace('_', '').upper(), (_n, _m['tag'])
    assert _n not in DFU_NAMES
del _n, _m


def is_ou(name):
    return name in MODELS


def base_seq(name):
    """smoke_recipe_fix 의 seq 이름 (unet | gru | ss2d) — build_groups·기능 시험 분기용."""
    if name in MODELS:
        return MODELS[name]['base']
    if name in DFU_NAMES:
        return name
    raise ValueError(f'모르는 모델 이름 {name!r} (OU {OU_NAMES} / 기존 {DFU_NAMES})')


def has_seq(name):
    return base_seq(name) in ('gru', 'ss2d')


def uses_gru(name):
    return base_seq(name) == 'gru'


def unet_ref_name(name):
    """SAME_UNET_INIT 의 기준 모델 (그 이름 자신이면 복사하지 않음): OU → unet_ou, 기존 이름 → unet (기존 트레이너와 같음)."""
    base_seq(name)
    return 'unet_ou' if name in MODELS else 'unet'


def partial_unet_copy(name):
    """o-ETER (OU) 만 True — 첫 합성곱의 입력 채널 수(20)가 기준(52)과 달라 모양이 같은 텐서만 복사한다."""
    return name in MODELS and MODELS[name]['seq_only_input']


def run_name(name, run_suffix, check=False):
    """런 폴더·wandb 이름. OU = PureETER_{TAG}_noDC_R4_brain384_v8fix{suffix}.
    check=True (트레이너 동등성 시험 전용, 기존 이름만) = PureETER_{SEQ}_noDC_R4_brain384_v8fix_OUtrainerCheck{suffix} —
    기존 런 폴더와 겹치지 않는다(트레이너는 이때 V8FIX_LOG_ROOT 를 저장소 logs/ 밖으로 요구한다)."""
    if name in MODELS:
        if check:
            raise ValueError('check 는 기존 이름(동등성 시험) 전용')
        return f"PureETER_{MODELS[name]['tag']}_noDC_R4_brain384_v8fix{run_suffix}"
    if name in DFU_NAMES and check:
        return f"PureETER_{name.upper()}_noDC_R4_brain384_v8fix_OUtrainerCheck{run_suffix}"
    raise ValueError(f'run_name: {name!r} (check={check}) — OU 이름이거나, 기존 이름이면 check=True 여야 함')


def _build_ou(name, device):
    import u_pure_eternet_ou as M         # models/pure_eternet (R 이 sys.path 에 넣음)
    if name == 'unet_ou':
        model = M.PureETER_UNETOU(n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
                                  unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF)
    elif name in ('gru_ou', 'gruo_ou'):
        cls = M.PureETER_GRUOU if name == 'gru_ou' else M.PureETER_GRUOOU
        model = cls(dim=C.IMAGE_SIZE[0], n_coil=C.N_COIL,
                    n_hidden_1=C.N_HIDDEN_LRNN_1, n_hidden_2=C.N_HIDDEN_LRNN_2,
                    unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF)
    else:
        raise ValueError(name)
    assert type(model).__name__ == MODELS[name]['cls'], (name, type(model).__name__)
    return model.to(device)


def build_model(name, device):
    """OU 이름은 config 인자(R.C)로 만든다 — R.build_model 의 gru / unet 분기와 같은 인자. 기존 이름은 R.build_model 그대로."""
    if name in MODELS:
        return _build_ou(name, device)
    if name in DFU_NAMES:
        return R.build_model(name, device)
    raise ValueError(f'모르는 모델 이름 {name!r}')


def build_groups(model, name):
    return R.build_groups(model, base_seq(name))


def copy_unet_init(model, ref, partial=False):
    """ref.unet 의 가중치를 model.unet 에 복사 (트레이너 SAME_UNET_INIT). state_dict 키(이름·순서)는 같아야 한다.
    partial=False: 모든 텐서의 모양이 같아야 한다 (기존 트레이너와 같은 전체 복사 — load_state_dict(ref.unet.state_dict())).
    partial=True:  모양이 같은 텐서만 복사하고, 다른 텐서는 자기 초기값을 둔다 (o-ETER (OU)).
    반환: dict(ref, n_copied, n_total, own_init=[{name, shape, ref_shape}])."""
    src = ref.unet.state_dict()
    dst = model.unet.state_dict()
    if list(src.keys()) != list(dst.keys()):
        raise AssertionError('U-Net 구조 불일치 — 계약 위반 (state_dict 키가 다름)')
    new, own = OrderedDict(), []
    for k, v in dst.items():
        if tuple(src[k].shape) == tuple(v.shape):
            new[k] = src[k]
        else:
            new[k] = v
            own.append(dict(name=f'unet.{k}', shape=list(v.shape), ref_shape=list(src[k].shape)))
    if own and not partial:
        raise AssertionError(f'U-Net 텐서 모양 불일치 — 전체 복사 계약 위반: {own}')
    model.unet.load_state_dict(new)
    return dict(ref=type(ref).__name__, n_copied=len(dst) - len(own), n_total=len(dst), own_init=own)


def unet_init_note(name, rec):
    """SCRATCH START 줄·last.pt run_config 의 unet_init 글자. 기존 이름은 기존 트레이너와 같은 글자."""
    if name not in MODELS:
        return 'copied from PureETER_UNET built right after seed_all(SEED)'
    s = (f'copied from {rec["ref"]} built right after seed_all(SEED) '
         f'({rec["n_copied"]}/{rec["n_total"]} U-Net tensors')
    if rec['own_init']:
        s += '; own init (shape differs): ' + ', '.join(
            f'{o["name"]} {tuple(o["shape"])} vs ref {tuple(o["ref_shape"])}' for o in rec['own_init'])
    return s + ')'


# ------------------------------------------------------------------ 기능 시험 (R.run_validation)
class _SeqOnlyUNetView:
    """o-ETER (OU) 의 U-Net 을 R.run_validation 이 부를 때의 대리자: 받은 cat(시퀀스 출력, zero-filled) 의 앞 n 채널만 넘긴다.
    down_path·up_path·last 는 실제 U-Net 의 모듈 그대로다(R.unet_with_zeroed 의 forward hook 이 실제 모듈에 걸린다)."""

    def __init__(self, unet, n):
        self._unet, self._n = unet, int(n)
        self.down_path, self.up_path, self.last = unet.down_path, unet.up_path, unet.last

    def __call__(self, x):
        return self._unet(x[:, :self._n])


class SeqOnlyInputView:
    """R.run_validation 에 seq='gru' 로 넘기는 o-ETER (OU) 의 대리 객체 (_seq · unet · forward · eval · train 만 쓴다)."""

    def __init__(self, model):
        self._m = model
        self.unet = _SeqOnlyUNetView(model.unet, model.seq_out_ch)

    def _seq(self, x_ksp):
        return self._m._seq(x_ksp)

    def __call__(self, *args, **kwargs):
        return self._m(*args, **kwargs)

    def eval(self):
        self._m.eval()
        return self

    def train(self, mode=True):
        self._m.train(mode)
        return self


def functional_target(model, name):
    """R.run_validation 에 넘길 (모델 또는 대리 객체, smoke_recipe_fix 의 seq 이름)."""
    if name in MODELS and MODELS[name]['seq_only_input']:
        return SeqOnlyInputView(model), MODELS[name]['base']
    return model, base_seq(name)


def run_validation(model, name, normalize, val_ds, indices, device, workers, pin, is_cuda):
    """R.run_validation(model, seq, normalize, val_ds, indices, device, workers, pin, is_cuda) 의 이름판 (트레이너 run_probe 용)."""
    obj, seq = functional_target(model, name)
    return R.run_validation(obj, seq, normalize, val_ds, indices, device, workers, pin, is_cuda)


# ------------------------------------------------------------------ 평가 진입점용 등록
def install_gru_fp32():
    """eval_v8fix_official.build() 가 --seq gru 에 하는 것과 같다 (gru_precision.install_class_patch, GRU_FP32_MODULES 기본 gru_h).
    gru_ou·gruo_ou 는 PureETER_GRU 하위 클래스라 그 클래스 패치가 그대로 걸린다. 이미 걸려 있으면 아무것도 하지 않는다."""
    import gru_precision                  # v8_eter_pure/ (Pod 브랜치 파일)
    names = tuple(n for n in os.environ.get('GRU_FP32_MODULES', 'gru_h').split(',') if n)
    gru_precision.install_class_patch(names)
    print(f'[recipe_ou] PureETER_GRU {names} → fp32 (eval_v8fix_official.build 의 --seq gru 와 같음)', flush=True)


def register(module=None, gru_fp32=True):
    """smoke_recipe_fix 모듈(module, 기본 R)의 build_model·run_validation 을 OU 이름도 받게 감싼다 (이 프로세스 안에서만).
    기존 이름은 원 함수를 그대로 부른다. gru_fp32=True 면 gru_ou·gruo_ou 를 만들기 직전에 install_gru_fp32().
    감싼 run_validation 은 원 함수와 인자 이름이 같다(평가 래퍼 eval_v8fix_fb._drive 가 inspect.signature 로 indices 를 읽음).
    이미 감쌌으면 False."""
    M = module if module is not None else R
    if getattr(M.build_model, '_ou_registered', False):
        return False
    orig_build, orig_val = M.build_model, M.run_validation

    def build_model(seq, device):
        if seq in MODELS:
            if gru_fp32 and MODELS[seq]['base'] == 'gru':
                install_gru_fp32()
            return _build_ou(seq, device)
        return orig_build(seq, device)

    def run_validation(model, seq, normalize, val_ds, indices, device, workers, pin, is_cuda):
        if seq in MODELS:
            obj, base = functional_target(model, seq)
            return orig_val(obj, base, normalize, val_ds, indices, device, workers, pin, is_cuda)
        return orig_val(model, seq, normalize, val_ds, indices, device, workers, pin, is_cuda)

    for f, o in ((build_model, orig_build), (run_validation, orig_val)):
        f._ou_registered = True
        f._ou_orig = o
    M.build_model = build_model
    M.run_validation = run_validation
    print(f'[recipe_ou] {M.__name__}.build_model·run_validation 에 OU 이름 {OU_NAMES} 등록', flush=True)
    return True
