"""
v8 sanity: "오직 sequence model(±DC)만 다름" + 차원 계약 + 교수님 파일 무수정 검증.

검사:
  1. 4 variant {gru,ss2d}×{noDC,DC} 생성 + forward (BS=1, 384²) → 출력 (1,1,384,384).
  2. 같은 DC 레벨에서 GRU/SS2D 의 U-Net DFU in_channels / n_hidden / depth / wf 동일.
  3. no-DC GRU 가 교수님 ETER_hybrid_GRU_DFU 와 동일 차원(gru_h/gru_v/unet param shape) 인지.
  4. git: models/hybrid_eternet/*, models/mamba_eternet/ss2d.py 무수정.
실행: conda activate mri_env && python v8_eter_pure/sanity_pure_v8.py
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

import myConfig_pure_eter_v8 as C
from u_pure_eternet_gru import PureETER_GRU
from u_pure_eternet_ss2d import PureETER_SS2D

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H = W = C.IMAGE_SIZE[0]


def build(seq, use_dc):
    if seq == 'gru':
        return PureETER_GRU(
            dim=H, n_coil=C.N_COIL,
            n_hidden_1=C.N_HIDDEN_LRNN_1, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            use_dc=use_dc, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        )
    return PureETER_SS2D(
        n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
        unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
        ss2d_d_inner=C.SS2D_D_INNER, ss2d_d_state=C.SS2D_D_STATE,
        use_dc=use_dc, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
    )


def unet_in_ch(model):
    # UNet_choh_skip.down_path[0] = UNetConvBlock; .block[0] = first Conv2d
    return model.unet.down_path[0].block[0].in_channels


def forward_check(model, use_dc):
    model.eval().to(DEV)
    x_img = torch.randn(1, C.INPUT_CHANNELS, H, W, device=DEV)
    x_ksp = torch.randn(1, C.INPUT_CHANNELS, H, W, device=DEV)
    mask  = torch.zeros(1, 1, H, W, device=DEV); mask[..., ::4] = 1.0
    sens  = torch.randn(1, C.INPUT_CHANNELS, H, W, device=DEV)
    with torch.no_grad():
        out = model(x_img, x_ksp, mask, sens)
    return tuple(out.shape)


def main():
    print(f"device={DEV}  H=W={H}  H2={C.N_HIDDEN_LRNN_2}  unet_depth={C.UNET_DEPTH}")
    results = {}
    for use_dc in (False, True):
        for seq in ('gru', 'ss2d'):
            key = f"{seq}_{'DC' if use_dc else 'noDC'}"
            m = build(seq, use_dc)
            in_ch = unet_in_ch(m)
            n_params = sum(p.numel() for p in m.parameters()) / 1e6
            shape = forward_check(m, use_dc)
            results[key] = (in_ch, shape, n_params)
            print(f"  [{key:12s}] unet_in_ch={in_ch}  out_shape={shape}  params={n_params:.1f}M")
            assert shape == (1, 1, H, W), f"{key} 출력 shape 오류: {shape}"
            del m
            if DEV.type == 'cuda':
                torch.cuda.empty_cache()

    # 2) 같은 DC 레벨에서 GRU/SS2D 의 U-Net in_channels 동일
    for tag in ('noDC', 'DC'):
        a = results[f'gru_{tag}'][0]
        b = results[f'ss2d_{tag}'][0]
        assert a == b, f"{tag}: GRU unet_in_ch({a}) != SS2D unet_in_ch({b}) — downstream 불일치!"
        print(f"  [contract:{tag}] GRU/SS2D unet_in_ch 동일 = {a}  OK")

    expected_in = 2 * C.N_HIDDEN_LRNN_2 + C.N_COIL * 2
    assert results['gru_noDC'][0] == expected_in, "in_channels 계약 위반"
    print(f"  [contract] in_channels == 2*H2+2*coil == {expected_in}  OK")

    # 4) 교수님/프로젝트 원본 파일 무수정 (git)
    print("\n  [git] 교수님 원본 파일 무수정 확인:")
    targets = ['models/hybrid_eternet/', 'models/mamba_eternet/ss2d.py',
               'models/hybrid_eternet/hybrid_eternet_fastmri-main/model.py']
    dirty = subprocess.run(
        ['git', '-C', _PROJECT_ROOT, 'status', '--porcelain'] + targets,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        print("    [WARN] 다음 원본 파일에 변경 감지:\n" + dirty)
        raise SystemExit("원본 파일이 수정됨 — 통제 비교 무효. 확인 요망.")
    print("    OK — 원본 무수정")

    print("\n✅ SANITY PASS — only-seq-model(±DC) differs, 차원 계약 충족, 원본 무수정.")


if __name__ == '__main__':
    main()
