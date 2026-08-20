"""
v9 언리시드 SS2D sanity — 차원 계약 + 게이팅/스택 구조 + 원본 무수정 + no-WD 그룹 검증.

구조/계약/git/no-WD 검사는 GPU 무접촉(CPU 가능). forward-finite 는 mamba_ssm CUDA 커널이
필요하므로 CUDA 있을 때만 수행(없으면 SKIP). 실제 forward+backward 검증은 스모크에서.

실행:
  CUDA_VISIBLE_DEVICES="" python v9_mamba_unleashed/sanity_ss2d_v9.py   # 구조/계약/git/no-WD (GPU-free)
  python v9_mamba_unleashed/sanity_ss2d_v9.py                            # + forward-finite (GPU)
"""

import os
import sys
import subprocess

import torch

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))

import myConfig_ss2d_v9 as C
from u_pure_eternet_ss2d_v9 import PureETER_SS2D_V9

H = W = C.IMAGE_SIZE[0]


def build():
    return PureETER_SS2D_V9(
        n_coil=C.N_COIL, out_ch=C.SS2D_OUT_CH,
        unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
        ss2d_d_inner=C.SS2D_D_INNER, ss2d_d_state=C.SS2D_D_STATE,
        ss2d_n_blocks=C.SS2D_N_BLOCKS, ss2d_dropout=C.SS2D_DROPOUT,
        ss2d_use_checkpoint=C.SS2D_USE_CHECKPOINT,
    )


def unet_in_ch(model):
    return model.unet.down_path[0].block[0].in_channels


def main():
    print(f"H=W={H}  out_ch={C.SS2D_OUT_CH}  d_inner={C.SS2D_D_INNER}  d_state={C.SS2D_D_STATE}  "
          f"n_blocks={C.SS2D_N_BLOCKS}  unet_depth={C.UNET_DEPTH}")
    m = build()

    # 1) U-Net 계약: in_channels == out_ch + 2*coil, n_hidden == in/2
    expected_in = C.SS2D_OUT_CH + C.N_COIL * 2
    in_ch = unet_in_ch(m)
    assert in_ch == expected_in, f"unet_in_ch({in_ch}) != out_ch+2coil({expected_in})"
    print(f"  [contract] unet_in_ch == out_ch+2*coil == {in_ch}  (n_hidden={in_ch//2})  OK")

    # 2) 게이팅 존재: 블록 in_proj 가 d_inner → 2*d_inner (x_ssm + z)
    blk0 = m.ss2d.blocks[0]
    assert blk0.in_proj.out_features == 2 * C.SS2D_D_INNER, \
        f"게이팅 분기 없음: in_proj out={blk0.in_proj.out_features} != 2*d_inner={2*C.SS2D_D_INNER}"
    assert len(m.ss2d.blocks) == C.SS2D_N_BLOCKS, "블록 수 불일치"
    print(f"  [gate] block.in_proj: {C.SS2D_D_INNER} → {blk0.in_proj.out_features} (=2·d_inner, x+z 게이팅)  OK")
    print(f"  [stack] n_blocks == {len(m.ss2d.blocks)}  OK")

    # 3) no-WD 그룹: Mamba A_log/D 가 _no_weight_decay 플래그를 가짐
    n_nodecay = sum(1 for p in m.parameters() if getattr(p, '_no_weight_decay', False))
    expected_nodecay = 2 * 4 * C.SS2D_N_BLOCKS   # (A_log,D) × 4방향 × n_blocks
    assert n_nodecay == expected_nodecay, \
        f"no-WD 파라미터 수 {n_nodecay} != 예상 {expected_nodecay} (A_log/D × 4dir × n_blocks)"
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  [no-WD] _no_weight_decay 플래그 파라미터 {n_nodecay}개 (=2×4×{C.SS2D_N_BLOCKS})  OK")
    print(f"  [params] 총 {n_params:.1f}M")

    # 4) 원본 무수정 (git): 교수님 DFU + 원본 ss2d.py (SelectiveScan1D)
    print("\n  [git] 원본 파일 무수정 확인:")
    targets = ['models/hybrid_eternet/', 'models/mamba_eternet/ss2d.py']
    dirty = subprocess.run(
        ['git', '-C', _PROJECT_ROOT, 'status', '--porcelain'] + targets,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        print("    [WARN] 원본 파일 변경 감지:\n" + dirty)
        raise SystemExit("원본 파일 수정됨 — 확인 요망.")
    print("    OK — 원본 무수정 (ss2d_v9.py 는 신규 파일)")

    # 5) forward-finite (CUDA 있을 때만; mamba_ssm 커널 필요)
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        m.eval().to(dev)
        x_img = torch.randn(1, C.INPUT_CHANNELS, H, W, device=dev)
        x_ksp = torch.randn(1, C.INPUT_CHANNELS, H, W, device=dev)
        with torch.no_grad():
            out = m(x_img, x_ksp)
        assert tuple(out.shape) == (1, 1, H, W), f"출력 shape 오류: {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "forward 출력에 non-finite"
        print(f"\n  [forward] out_shape={tuple(out.shape)}  finite=OK (GPU)")
    else:
        print("\n  [forward] SKIP (CUDA 없음 — mamba_ssm 커널 필요, 스모크에서 검증)")

    print("\n✅ SANITY PASS — 차원 계약·게이팅·스택·no-WD·원본무수정 확인.")


if __name__ == '__main__':
    main()
