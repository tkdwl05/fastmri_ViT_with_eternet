"""
Pure ETER-Net (Transformer) — v8 통제비교의 3번째 팔 (2026-09-02).
구현 = axial attention(행/열 MHSA).

교수님 ETER-net 의 양방향 GRU 를 Transformer(axial attention, 행/열 MHSA)로 치환한 arm.
구조: x_ksp → AxialAttentionStack(out_ch=2*H2) → cat(aliased image) → UNet_choh_skip(DFU) → 출력
  - GRU/SS2D arm 과 downstream(cat 순서, U-Net DFU, 입출력) **완전 동일**.
    유일한 차이는 시퀀스 모델 [gru_h+gru_v | SS2D] → [axial MHSA] 뿐.
  - out_ch = 2*n_hidden_2 강제 → cat 후 U-Net in_channels 가 두 기존 arm 과 동일.
설계·근거: docs/axial_transformer_arm_design.md. 원본 파일 무수정 (UNet_choh_skip 재사용).
forward 시그니처: forward(x_img, x_ksp, mask=None, sens=None)  — 기존 arm 과 동일.
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))
sys.path.append(os.path.join(_HERE, '..', 'attn_eternet'))

from myUNet_DF import UNet_choh_skip     # 교수님 DFU (무수정)
from transformer_v10 import TransformerStack


class PureETER_TRANSFORMER(nn.Module):
    def __init__(
        self,
        *,
        n_coil: int = 16,
        n_hidden_2: int = 10,
        unet_depth: int = 5,
        unet_wf: int = 6,
        axial_d_model: int = 64,
        axial_n_pairs: int = 2,
        axial_n_heads: int = 4,
        use_dc: bool = False,
        dc_k_scale_ratio: float = 100.0,
        dc_init_alpha: float = 1.0,
    ):
        super().__init__()
        self.use_dc = use_dc

        c_in   = n_coil * 2                       # 32 (k-space real/imag)
        out_ch = 2 * n_hidden_2                   # GRU out_v 채널과 동일하게 강제
        num_feat_ch = out_ch + c_in               # DFU in_channels (== GRU/SS2D arm)
        n_hidden    = n_hidden_2 + n_coil
        assert n_hidden * 2 == num_feat_ch, \
            f"UNet_choh_skip 계약 위반: n_hidden*2({n_hidden*2}) != in_channels({num_feat_ch})"

        # 시퀀스 모델 = Transformer/axial attention (GRU/SS2D 대체)
        self.axial = TransformerStack(
            c_in=c_in, out_ch=out_ch,
            d_model=axial_d_model, n_pairs=axial_n_pairs, n_heads=axial_n_heads)

        # 최종 합성 = 교수님 DFU (기존 arm 과 동일)
        n_classes = 2 if use_dc else 1
        self.unet = UNet_choh_skip(
            in_channels=num_feat_ch, n_classes=n_classes,
            depth=unet_depth, wf=unet_wf, padding=True,
            batch_norm=False, up_mode='upconv', n_hidden=n_hidden,
        )

        if use_dc:
            sys.path.append(os.path.join(_HERE, '..', 'mamba_eternet'))
            from u_choh_model_SS2D_ViT_v4 import DCBlock
            self.dc = DCBlock(k_scale_ratio=dc_k_scale_ratio, init_alpha=dc_init_alpha)

        print(f"   'PureETER_TRANSFORMER' (H2={n_hidden_2}, out_ch={out_ch}, "
              f"d_model={axial_d_model}, n_pairs={axial_n_pairs}, n_heads={axial_n_heads}, "
              f"in_ch={num_feat_ch}, n_hidden={n_hidden}, unet_depth={unet_depth}, "
              f"use_dc={use_dc})")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        out    = self.axial(x_ksp)                 # (B, 2*H2, H, W)
        in_cnn = torch.cat((out, x_img), dim=1)
        out    = self.unet(in_cnn)                 # (B, 1 or 2, H, W)
        if self.use_dc:
            x_ri = self.dc(out, x_ksp, mask, sens)
            out  = torch.sqrt(x_ri[:, 0:1] ** 2 + x_ri[:, 1:2] ** 2 + 1e-12)
        return out
