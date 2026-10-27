"""eval_v8fix.py 로 Mamba-ETER(PureETER_Mamba1D)를 평가하는 얇은 래퍼 (2026-10-12). 인자는 eval_v8fix.py 와 같고 --seq mamba1d.

학습(main_train_pure_v8fix_mamba1d.py)과 같은 모델·정밀도로 평가한다:
  - --seq mamba1d 를 SPECS 에 더하고, smoke_recipe_fix.build_model('mamba1d') 이 PureETER_Mamba1D(학습과 같은 config 인자,
    fp32_stages = MAMBA_ETER_FP32 — 기본 hv)를 만들게 한다. 다른 --seq 는 원래 함수로 넘긴다.
  - smoke_recipe_fix.run_validation(기능 시험)에는 라벨 'gru' 를 넘긴다 — 'mamba1d' 를 넘기면 시퀀스 출력 대신 0 을 넣고 시퀀스
    시험을 조용히 뺀다(트레이너 docstring 과 같은 이유).
  - 정밀도는 생성 인자라 state_dict(strict 로드)로는 확인되지 않으므로, --ckpt 옆 recipe.json 의 mamba_fp32_stages 와
    MAMBA_ETER_FP32 가 다르면 거부하고 출력에도 찍는다.
  - TF32 는 학습과 같게 끈다(ALLOW_TF32=1 이면 켬) — torch 기본은 cuDNN TF32 켬이다.
eval_v8fix_grufp32.py 처럼 runpy 로 다시 실행하면 SPECS 가 새로 정의되므로, 여기서는 import 한 뒤 main() 을 부른다.

  예) MAMBA_ETER_FP32=hv python v8_eter_pure/eval_v8fix_mamba1d.py --seq mamba1d \\
        --ckpt logs/PureETER_MAMBA1D_noDC_R4_brain384_v8fix_s1_fullval/pure_mamba1d_best.pt --out-dir ... --skip-precheck
"""
import os
import sys
import json

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.join(os.path.dirname(_HERE), 'models', 'pure_eternet'))

import eval_v8fix as V                                                       # noqa: E402 (main 은 __main__ 일 때만 실행)
from u_pure_eternet_mamba1d import PureETER_Mamba1D, fp32_stages_from_str    # noqa: E402

R = V.R
C = R.C
STAGES = fp32_stages_from_str(os.environ.get('MAMBA_ETER_FP32', 'hv'))
HARNESS_SEQ = 'gru'
if os.environ.get('ALLOW_TF32', '0') != '1':           # 트레이너와 같음 (main_train_pure_v8fix_mamba1d.py 의 TF32 주석)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _check_recipe_precision(argv):
    """--ckpt 옆 recipe.json 이 있으면 학습 때 fp32 단계와 같은지 확인 (다르면 종료)."""
    ck = None
    for i, a in enumerate(argv):
        if a == '--ckpt' and i + 1 < len(argv):
            ck = argv[i + 1]
        elif a.startswith('--ckpt='):
            ck = a.split('=', 1)[1]
    if not ck:
        return
    rj = os.path.join(os.path.dirname(os.path.abspath(ck)), 'recipe.json')
    if not os.path.exists(rj):
        print(f'[eval_v8fix_mamba1d] recipe.json 없음 ({rj}) — 정밀도 대조 생략')
        return
    with open(rj) as f:
        got = json.load(f).get('mamba_fp32_stages')
    if got is not None and list(got) != list(STAGES):
        raise SystemExit(f'[eval_v8fix_mamba1d] 정밀도 불일치: 학습 {got} vs MAMBA_ETER_FP32 → {list(STAGES)} — 중단')
    print(f'[eval_v8fix_mamba1d] recipe.json mamba_fp32_stages={got} = 평가 {list(STAGES)}')

V.SPECS['mamba1d'] = dict(name='Mamba-ETER', title='Mamba-ETER 비교 모델 (ETER-net 의 bi-GRU 셀 → 양방향 Mamba) — 레시피 DL')

_orig_build_model = R.build_model
_orig_run_validation = R.run_validation


def build_model(seq, device):
    if seq != 'mamba1d':
        return _orig_build_model(seq, device)
    model = PureETER_Mamba1D(dim=C.IMAGE_SIZE[0], n_coil=C.N_COIL, n_hidden_1=C.N_HIDDEN_LRNN_1,
                             n_hidden_2=C.N_HIDDEN_LRNN_2, unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
                             use_dc=False, fp32_stages=STAGES)
    print(f'[eval_v8fix_mamba1d] PureETER_Mamba1D fp32_stages={STAGES} '
          f'(h={model.gru_h.force_fp32}, v={model.gru_v.force_fp32}) cfg={model.mamba_cfg}')
    return model.to(device)


def run_validation(model, seq, *a, **k):
    return _orig_run_validation(model, HARNESS_SEQ if seq == 'mamba1d' else seq, *a, **k)


R.build_model = build_model
R.run_validation = run_validation

if __name__ == '__main__':
    print(f'[eval_v8fix_mamba1d] MAMBA_ETER_FP32 → {STAGES}; 기능 시험 라벨 mamba1d → {HARNESS_SEQ!r}; '
          f'TF32 matmul={torch.backends.cuda.matmul.allow_tf32} cudnn={torch.backends.cudnn.allow_tf32}')
    _check_recipe_precision(sys.argv[1:])
    V.main()
