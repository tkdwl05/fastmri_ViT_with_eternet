"""
Pure ETER-Net (SS2D v9, 언리시드 Mamba) — v8 `u_pure_eternet_ss2d.py` 의 강화판.

v8 SS2D arm 은 GRU 와의 공정 비교를 위해 out_ch 를 2*H2(=20)로 강제하고 단일 블록·게이팅
없음으로 조여져 있었다. v9 는 그 비교와 분리된 신규 트랙으로 **병목을 해제**한다:

  x_ksp → SS2DStackV9(out_ch 자유, 게이팅+residual 스택) → cat(aliased image) → UNet_choh_skip(DFU) → magnitude

- downstream(cat 순서, U-Net DFU, magnitude, forward 시그니처)은 v8 SS2D arm 과 **동일 계약**.
- 유일한 차이: 시퀀스 모델이 [v8 SS2D(20ch, 1블록)] → [SS2DStackV9(out_ch 자유, N블록, 게이팅)].
- **DC 미사용** (v9 는 순수 품질 최대화 트랙; forward 의 mask/sens 는 무시).
- U-Net 계약: n_hidden*2 == in_channels, in_channels = out_ch + c_in(32) → n_hidden=(out_ch+32)//2.
  따라서 out_ch 는 짝수여야 한다.

교수님/프로젝트 원본 파일 무수정 (UNet_choh_skip, SS2DStackV9 는 import 재사용).
forward 시그니처: forward(x_img, x_ksp, mask=None, sens=None) — v8 과 통일.
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))
sys.path.append(os.path.join(_HERE, '..', 'mamba_eternet'))

from myUNet_DF import UNet_choh_skip   # 교수님 DFU (무수정)
from ss2d_v9 import SS2DStackV9        # v9 언리시드 SS2D 스택


class PureETER_SS2D_V9(nn.Module):
    def __init__(
        self,
        *,
        n_coil: int = 16,
        out_ch: int = 64,
        unet_depth: int = 5,
        unet_wf: int = 6,
        ss2d_d_inner: int = 256,
        ss2d_d_state: int = 32,
        ss2d_n_blocks: int = 3,
        ss2d_dropout: float = 0.05,
        ss2d_use_checkpoint: bool = False,
        ss2d_downsample: int = 1,
    ):
        super().__init__()

        c_in = n_coil * 2                          # 32 (k-space real/imag)
        assert out_ch % 2 == 0, f"out_ch 는 짝수여야 함 (U-Net n_hidden 계약): {out_ch}"
        num_feat_ch = out_ch + c_in                # DFU in_channels
        n_hidden    = num_feat_ch // 2
        assert n_hidden * 2 == num_feat_ch, \
            f"UNet_choh_skip 계약 위반: n_hidden*2({n_hidden*2}) != in_channels({num_feat_ch})"

        # 시퀀스 모델 = 언리시드 SS2D 스택 (게이팅 + residual, 병목 해제, 옵션 다운샘플)
        self.ss2d = SS2DStackV9(
            c_in=c_in, d_inner=ss2d_d_inner, d_state=ss2d_d_state,
            out_ch=out_ch, n_blocks=ss2d_n_blocks, dropout=ss2d_dropout,
            use_checkpoint=ss2d_use_checkpoint, downsample=ss2d_downsample,
        )

        # 최종 합성 = 교수님 DFU (v8 arm 과 동일 building block, 채널만 out_ch 반영)
        self.unet = UNet_choh_skip(
            in_channels=num_feat_ch, n_classes=1,
            depth=unet_depth, wf=unet_wf, padding=True,
            batch_norm=False, up_mode='upconv', n_hidden=n_hidden,
        )

        print(f"   'PureETER_SS2D_V9' (out_ch={out_ch}, d_inner={ss2d_d_inner}, "
              f"d_state={ss2d_d_state}, n_blocks={ss2d_n_blocks}, dropout={ss2d_dropout}, "
              f"in_ch={num_feat_ch}, n_hidden={n_hidden}, unet_depth={unet_depth}, "
              f"grad_ckpt={ss2d_use_checkpoint})")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        out    = self.ss2d(x_ksp)                  # (B, out_ch, H, W)
        in_cnn = torch.cat((out, x_img), dim=1)
        out    = self.unet(in_cnn)                 # (B, 1, H, W)
        return out
