"""
SS2D 인과 진단 추가 조건 E-a (2026-10-11) — SS2D out_proj 를 작게 초기화해 '초기 크기 고착'과 '구조'를 가른다.

배경 (results/ss2d_diag_2026-10-11/forensics.md·theory.md·verify.md):
  - 학습이 SS2D 출력을 out_proj 크기 축소로 거의 0 까지 줄였다(RMS 0.351 → 0.0043). 초기 SS2D 출력은 zero-filled 영상 채널의
    6.6배(U-Net 첫 conv 응답 6.1배)였고, 같은 레시피의 ETER-net(GRU) 초기 출력은 0.21배였다가 학습 중 46배 커졌다.
  - 두 설명이 남는다. H4 초기 크기 고착: 초기 출력이 너무 커서 먼저 억제됐고 그 뒤 다시 쓰이지 못했다.
    H1 구조: 십자형 수용 영역·실수 감쇠 스캔·norm_in·k-space 좌표 때문에 SS2D 특징이 U-Net 에 쓸모없어 억제됐다.
  - 이 조건은 out_proj weight·bias 를 k 배(기본 0.03 → 초기 출력 RMS ≈ 0.011, 영상 채널의 ≈0.2배 = ETER-net 초기 수준)로
    줄인 것만 V0 ctrl 과 다르다. 나머지(데이터·레시피·SEED·U-Net 초기값·검증·기능 시험·추가 분석)는 같다. 스케일은 RNG 를
    쓰지 않으므로 U-Net 초기값과 데이터 순서가 V0 과 같다. AdamW 는 기울기 크기에 불변이라 SS2D 내부 갱신 폭은 k 와 무관하다.
  - 판정: 4000 step 끝에 shuffle_cross·슬라이스 고유 공간 성분 시험에서 '사용'이면 H4 쪽, V0 처럼 억제되면 H1 쪽.

구현: diag_ss2d_variants.py 를 모듈로 import 해 variant 'smallout' 를 인스턴스 수준으로 더한다(그 파일·ss2d.py·smoke_recipe_fix.py
무수정 — 진단 체인이 같은 파일을 쓰는 중이라 새 파일로 분리). 출력 이름 v3_smallout_k<k>, 출력 폴더·요약 md 는 같은 곳.

실행 예:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python -u v8_eter_pure/diag_ss2d_smallout.py --out-scale 0.03 --out-dir results/ss2d_diag_2026-10-11 --skip-done
  # 스모크: ... --steps 30 --log-every 10 --log-dense-every 0 --val-slices 4 --post-slices 4 --out-dir <scratch>
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import diag_ss2d_variants as D                                         # noqa: E402


def _pop_out_scale(argv):
    """--out-scale 값을 argv 에서 빼고 돌려준다 (D.parse_args 는 이 인자를 모른다)."""
    k, out, i = 0.03, [], 0
    while i < len(argv):
        a = argv[i]
        if a == '--out-scale':
            k = float(argv[i + 1])
            i += 2
            continue
        if a.startswith('--out-scale='):
            k = float(a.split('=', 1)[1])
            i += 1
            continue
        out.append(a)
        i += 1
    return k, out


OUT_SCALE, _ARGV = _pop_out_scale(sys.argv[1:])
assert 0 < OUT_SCALE <= 1, OUT_SCALE

D.VNAME['smallout'] = 'v3_smallout'
D.VDESC['smallout'] = (f'V3 원 SS2D (k-space 입력), out_proj weight·bias 를 초기에 ×{OUT_SCALE:g} — 초기 출력 크기 가설(H4) 시험')

_orig_apply = D.apply_variant
_orig_vname = D.vname_of


def apply_variant(model, variant, gamma):
    if variant != 'smallout':
        return _orig_apply(model, variant, gamma)
    with torch.no_grad():
        model.ss2d.out_proj.weight.mul_(OUT_SCALE)
        if model.ss2d.out_proj.bias is not None:
            model.ss2d.out_proj.bias.mul_(OUT_SCALE)
    model._diag_variant = variant
    model._diag_out_scale = OUT_SCALE
    return model


def vname_of(args):
    if args.variant == 'smallout':
        return f'v3_smallout_k{OUT_SCALE:g}'
    return _orig_vname(args)


D.apply_variant = apply_variant                                        # make_builder 가 호출 시점에 모듈 전역에서 찾는다
D.vname_of = vname_of

if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--variant', 'smallout'] + [a for a in _ARGV if a not in ('--variant', 'smallout')]
    sys.exit(D.main())
