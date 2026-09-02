"""
Pure ETER-Net (Pixel-GRU) — v8 통제비교의 4번째 팔: 가중치 공유 재귀 (2026-09-02).

원본 ETER-GRU(flatten-reshape, 668M)와 달리 픽셀 단위 가중치 공유 bi-GRU 를 같은 슬롯에
넣어 "재귀 메커니즘 vs 파라미터화" confound 를 분리한다 (docs/v8_fairness_followup_plan.md ③).
구조: x_ksp → PixelGRUStack(out_ch=2*H2) → cat(aliased image) → UNet_choh_skip(DFU) → 출력
  - GRU/SS2D/axial arm 과 downstream **완전 동일**. 유일한 차이는 시퀀스 모듈 뿐.
forward 시그니처: forward(x_img, x_ksp, mask=None, sens=None) — 기존 arm 과 동일.
원본 파일 무수정 (UNet_choh_skip 재사용).
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))
sys.path.append(os.path.join(_HERE, '..', 'rnn_eternet'))

from myUNet_DF import UNet_choh_skip     # 교수님 DFU (무수정)
from pixelgru_v10 import PixelGRUStack


class PureETER_PIXELGRU(nn.Module):
    def __init__(
        self,
        *,
        n_coil: int = 16,
        n_hidden_2: int = 10,
        unet_depth: int = 5,
        unet_wf: int = 6,
        pixelgru_hidden: int = 64,
        use_dc: bool = False,
        dc_k_scale_ratio: float = 100.0,
        dc_init_alpha: float = 1.0,
    ):
        super().__init__()
        self.use_dc = use_dc

        c_in   = n_coil * 2                       # 32 (k-space real/imag)
        out_ch = 2 * n_hidden_2                   # 기존 arm 과 동일 강제 (=20)
        num_feat_ch = out_ch + c_in
        n_hidden    = n_hidden_2 + n_coil
        assert n_hidden * 2 == num_feat_ch, \
            f"UNet_choh_skip 계약 위반: n_hidden*2({n_hidden*2}) != in_channels({num_feat_ch})"

        # 시퀀스 모델 = 가중치 공유 pixel-scan bi-GRU
        self.pixelgru = PixelGRUStack(c_in=c_in, out_ch=out_ch, hidden=pixelgru_hidden)

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

        print(f"   'PureETER_PIXELGRU' (H2={n_hidden_2}, out_ch={out_ch}, "
              f"hidden={pixelgru_hidden}, in_ch={num_feat_ch}, n_hidden={n_hidden}, "
              f"unet_depth={unet_depth}, use_dc={use_dc})")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        out    = self.pixelgru(x_ksp)              # (B, 2*H2, H, W)
        in_cnn = torch.cat((out, x_img), dim=1)
        out    = self.unet(in_cnn)                 # (B, 1 or 2, H, W)
        if self.use_dc:
            x_ri = self.dc(out, x_ksp, mask, sens)
            out  = torch.sqrt(x_ri[:, 0:1] ** 2 + x_ri[:, 1:2] ** 2 + 1e-12)
        return out
