"""
v9 radapt sanity — 마스크 조건화 + DC + 차원 계약 + no-WD + 원본 무수정 + multi-AR dataloader import.

구조/계약/git/no-WD 는 GPU 무접촉. forward-finite(DC 포함)는 CUDA 있을 때만.

실행:
  CUDA_VISIBLE_DEVICES="" python v9_mamba_radapt/sanity_ss2d_v9_radapt.py   # 구조 (GPU-free)
  python v9_mamba_radapt/sanity_ss2d_v9_radapt.py                            # + forward-finite (GPU)
"""

import os
import sys
import subprocess

import torch

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))

import myConfig_ss2d_v9_radapt as C
from u_pure_eternet_ss2d_v9_radapt import PureETER_SS2D_V9_Radapt

H = W = C.IMAGE_SIZE[0]


def build():
    return PureETER_SS2D_V9_Radapt(
        n_coil=C.N_COIL, out_ch=C.SS2D_OUT_CH,
        unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
        ss2d_d_inner=C.SS2D_D_INNER, ss2d_d_state=C.SS2D_D_STATE,
        ss2d_n_blocks=C.SS2D_N_BLOCKS, ss2d_dropout=C.SS2D_DROPOUT,
        ss2d_use_checkpoint=C.SS2D_USE_CHECKPOINT,
        mask_condition=C.MASK_CONDITION,
        dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
    )


def unet_in_ch(model):
    return model.unet.down_path[0].block[0].in_channels


def main():
    print(f"H=W={H}  out_ch={C.SS2D_OUT_CH}  mask_cond={C.MASK_CONDITION}  AR={C.AR_CHOICES}")
    m = build()

    # 1) U-Net 계약: complex head n_classes=2, in_channels == out_ch + 2*coil
    expected_in = C.SS2D_OUT_CH + C.N_COIL * 2
    in_ch = unet_in_ch(m)
    assert in_ch == expected_in, f"unet_in_ch({in_ch}) != out_ch+2coil({expected_in})"
    # UNet_choh_skip.last = 마지막 conv → n_classes
    print(f"  [contract] unet_in_ch == {in_ch} (n_hidden={in_ch//2})  OK")

    # 2) 마스크 조건화: SS2D 입력 채널 = 2*coil + 1 (mask)
    ss2d_in = m.ss2d.in_proj.in_features
    expected_seq_in = C.N_COIL * 2 + (1 if C.MASK_CONDITION else 0)
    assert ss2d_in == expected_seq_in, f"SS2D c_in({ss2d_in}) != 2coil+mask({expected_seq_in})"
    print(f"  [mask-cond] SS2D c_in == 2*coil + mask == {ss2d_in}  OK")

    # 3) DC block 존재 + α 파라미터
    assert hasattr(m, 'dc') and hasattr(m.dc, 'alpha'), "DCBlock 없음"
    print(f"  [DC] DCBlock 존재, α₀={m.dc.alpha.item():.3f}, k_scale={float(m.dc.k_scale_ratio):.0f}  OK")

    # 4) 게이팅 + no-WD
    blk0 = m.ss2d.blocks[0]
    assert blk0.in_proj.out_features == 2 * C.SS2D_D_INNER, "게이팅 분기 없음"
    n_nodecay = sum(1 for p in m.parameters() if getattr(p, '_no_weight_decay', False))
    expected_nodecay = 2 * 4 * C.SS2D_N_BLOCKS
    assert n_nodecay == expected_nodecay, f"no-WD {n_nodecay} != {expected_nodecay}"
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  [gate] block.in_proj 256→{blk0.in_proj.out_features} (게이팅)  OK")
    print(f"  [no-WD] {n_nodecay}개 (=2×4×{C.SS2D_N_BLOCKS})  |  [params] {n_params:.1f}M")

    # 5) multi-AR dataloader import + 서브클래스 계약 (데이터 스캔은 생략 — 무거움)
    from dataloader_h5_v9_multiAR import FastMRI_H5_MultiAR
    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    assert issubclass(FastMRI_H5_MultiAR, FastMRI_H5_Dataloader), "MultiAR 서브클래스 아님"
    assert '__getitem__' in FastMRI_H5_MultiAR.__dict__, "__getitem__ 오버라이드 없음"
    print(f"  [multiAR] FastMRI_H5_MultiAR(서브클래스) import OK, __getitem__ override OK")

    # 6) 원본 무수정 (git)
    print("\n  [git] 원본 파일 무수정 확인:")
    targets = ['models/hybrid_eternet/', 'models/mamba_eternet/ss2d.py',
               'models/mamba_eternet/u_choh_model_SS2D_ViT_v4.py', 'dataloaders/dataloader_h5_v5.py']
    dirty = subprocess.run(
        ['git', '-C', _PROJECT_ROOT, 'status', '--porcelain'] + targets,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        print("    [WARN] 원본 파일 변경 감지:\n" + dirty)
        raise SystemExit("원본 파일 수정됨 — 확인 요망.")
    print("    OK — 원본 무수정")

    # 7) forward-finite (CUDA 있을 때만; mamba 커널 + DC FFT)
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        m.eval().to(dev)
        x_img = torch.randn(1, C.INPUT_CHANNELS, H, W, device=dev)
        x_ksp = torch.randn(1, C.INPUT_CHANNELS, H, W, device=dev)
        mask  = torch.zeros(1, 1, H, W, device=dev); mask[..., ::4] = 1.0
        sens  = torch.randn(1, C.INPUT_CHANNELS, H, W, device=dev)
        with torch.no_grad():
            out = m(x_img, x_ksp, mask, sens)
        assert tuple(out.shape) == (1, 1, H, W), f"출력 shape: {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "forward non-finite"
        print(f"\n  [forward] out_shape={tuple(out.shape)}  finite=OK (DC 포함, GPU)")
    else:
        print("\n  [forward] SKIP (CUDA 없음 — 스모크에서 검증)")

    print("\n✅ RADAPT SANITY PASS — 마스크조건화·DC·게이팅·no-WD·multiAR·원본무수정 확인.")


if __name__ == '__main__':
    main()
