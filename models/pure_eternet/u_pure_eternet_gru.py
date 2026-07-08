"""
Pure ETER-Net (GRU) — 교수님 ETER_hybrid_GRU_DFU 의 통제비교용 재현 (v8 2×2 ablation).

교수님 원본 (models/hybrid_eternet/hybrid_eternet_fastmri-main/model.py::ETER_hybrid_GRU_DFU):
    x_ksp → gru_h(양방향) → reshape → gru_v(양방향) → cat(aliased image) → UNet_choh_skip(DFU) → 출력
forward reshape 로직을 그대로 복제하고, 동일 building block(nn.GRU, UNet_choh_skip[, DCBlock])만
재사용한다. **교수님/프로젝트 원본 파일은 일절 수정하지 않는다.**

v8 2×2 ablation 토글:
  - use_dc=False : 원본과 동일 (UNet n_classes=1, magnitude 출력). ★ no-DC arm = 교수님 ETER-net 충실.
  - use_dc=True  : UNet n_classes=2(complex) → DCBlock → magnitude (DC-CNN/VarNet 계열 증강).

forward 시그니처는 SS2D arm 과 통일: forward(x_img, x_ksp, mask=None, sens=None).
no-DC 는 mask/sens 를 무시한다.
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))
sys.path.append(os.path.join(_HERE, '..', 'mamba_eternet'))

from myUNet_DF import UNet_choh_skip   # 교수님 DFU (무수정 재사용)


class PureETER_GRU(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 384,
        n_coil: int = 16,
        n_hidden_1: int = 10,
        n_hidden_2: int = 10,
        unet_depth: int = 5,
        unet_wf: int = 6,
        use_dc: bool = False,
        dc_k_scale_ratio: float = 100.0,
        dc_init_alpha: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_coil = n_coil
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.num_layers = 1
        self.use_dc = use_dc

        # 원본 ETER_hybrid_GRU_DFU 의 차원 산정 그대로
        self.num_in_x = dim
        self.num_in_y = dim
        self.num_out_x = dim
        self.num_out_y = dim
        input_size = self.num_in_y * n_coil * 2
        self.num_out1 = self.num_out_y * n_hidden_1
        num_in2       = self.num_in_x * n_hidden_1
        self.num_out2 = self.num_out_x * n_hidden_2

        # 시퀀스 모델 = 양방향 GRU ×2 (교수님 원본과 동일 building block)
        self.gru_h = nn.GRU(input_size, self.num_out1, self.num_layers,
                            batch_first=True, bidirectional=True)
        self.gru_v = nn.GRU(num_in2 * 2, self.num_out2, self.num_layers,
                            batch_first=True, bidirectional=True)

        # 최종 합성 = 교수님 DFU (cat(seq_out, x_img) → U-Net)
        num_feat_ch = 2 * n_hidden_2 + n_coil * 2      # out_v(2*H2) + x_img(2*coil=32)
        n_hidden    = n_hidden_2 + n_coil
        assert n_hidden * 2 == num_feat_ch, \
            f"UNet_choh_skip 계약 위반: n_hidden*2({n_hidden*2}) != in_channels({num_feat_ch})"
        n_classes = 2 if use_dc else 1
        self.unet = UNet_choh_skip(
            in_channels=num_feat_ch, n_classes=n_classes,
            depth=unet_depth, wf=unet_wf, padding=True,
            batch_norm=False, up_mode='upconv', n_hidden=n_hidden,
        )

        if use_dc:
            # DC arm 일 때만 import (no-DC GRU 는 mamba_ssm 의존성 회피)
            from u_choh_model_SS2D_ViT_v4 import DCBlock
            self.dc = DCBlock(k_scale_ratio=dc_k_scale_ratio, init_alpha=dc_init_alpha)

        print(f"   'PureETER_GRU' (H1={n_hidden_1}, H2={n_hidden_2}, "
              f"in_ch={num_feat_ch}, n_hidden={n_hidden}, unet_depth={unet_depth}, "
              f"use_dc={use_dc})")

    def _seq(self, x_ksp):
        """교수님 ETER_hybrid_GRU_DFU.forward 의 gru_h→gru_v reshape 그대로 복제.
        반환: out_v (B, 2*n_hidden_2, H, W).

        원본은 h0 를 명시적 zeros 로 넘기지만, 여기선 h0=None(기본값 zeros)을 써서
        autocast 시 입력(fp16)과 h0(fp32) 의 dtype 불일치를 회피한다. 기능 동일."""
        B = x_ksp.size(0)
        in_h = x_ksp.reshape([B, self.num_in_x, -1])
        out_h, _ = self.gru_h(in_h)
        out_h = out_h.reshape([B, self.num_in_x, self.num_out_y, -1])
        out_h = out_h.permute(0, 2, 1, 3)
        out_h = out_h.reshape([B, self.num_out_y, -1])

        out_v, _ = self.gru_v(out_h)
        out_v = out_v.reshape([B, self.num_out_y, self.num_out_x, -1])
        out_v = out_v.permute(0, 3, 2, 1)              # (B, 2*H2, H, W)
        return out_v

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        out_v  = self._seq(x_ksp)
        in_cnn = torch.cat((out_v, x_img), dim=1)
        out    = self.unet(in_cnn)                     # (B, 1 or 2, H, W)
        if self.use_dc:
            x_ri = self.dc(out, x_ksp, mask, sens)     # (B, 2, H, W)
            out  = torch.sqrt(x_ri[:, 0:1] ** 2 + x_ri[:, 1:2] ** 2 + 1e-12)
        return out
