"""
Pure ETER-Net (SS2D v9 radapt) — R 대응(operator-근사) 변형.

v9 언리시드 백본(`SS2DStackV9`)에 **오퍼레이터식 R-일반화 요소 2가지**를 얹는다
(model-based DL 근사 — 완전한 neural-operator 는 아니지만 우리 파이프라인 내 가능 범위):

  1. **마스크(측정연산자) 명시 조건화**: sampling mask 를 seq 입력에 채널 concat →
     "지금 어떤 k-line 이 측정됐는지(=어떤 R 인지)" 를 명시 신호로 제공(§4.2-②).
     no-DC/단일-R 은 x_ksp 의 0-패턴으로 암묵 인지만 함 → 여기선 explicit.
  2. **Data Consistency(측정 앵커)**: 끝에 1-iter soft DCBlock → 측정된 k-line 을 R 무관하게
     강제(§4.2-①). U-Net n_classes=2(complex) 출력 → DC → magnitude.

  x_ksp ─┬─(mask concat)→ SS2DStackV9(c_in=33) → cat(aliased image) → U-Net(complex) → DCBlock → |·|
   mask ─┘

downstream 계약은 언리시드와 동일(n_hidden=(out_ch+2coil)/2). DCBlock 은 v8 `u_choh_model_SS2D_ViT_v4`
재사용(무수정). forward 시그니처: forward(x_img, x_ksp, mask, sens) — mask/sens 필수(DC·조건화).
교수님/프로젝트 원본 파일 무수정.
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))
sys.path.append(os.path.join(_HERE, '..', 'mamba_eternet'))

from myUNet_DF import UNet_choh_skip   # 교수님 DFU (무수정)
from ss2d_v9 import SS2DStackV9        # v9 언리시드 백본


class PureETER_SS2D_V9_Radapt(nn.Module):
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
        mask_condition: bool = True,
        dc_k_scale_ratio: float = 100.0,
        dc_init_alpha: float = 1.0,
    ):
        super().__init__()
        self.mask_condition = mask_condition

        c_img = n_coil * 2                                   # 32 (aliased image ch)
        c_seq_in = c_img + (1 if mask_condition else 0)      # 33: k-space + mask 채널
        assert out_ch % 2 == 0, f"out_ch 는 짝수여야 함: {out_ch}"
        num_feat_ch = out_ch + c_img                         # U-Net in_channels
        n_hidden    = num_feat_ch // 2
        assert n_hidden * 2 == num_feat_ch, \
            f"UNet_choh_skip 계약 위반: n_hidden*2({n_hidden*2}) != in_channels({num_feat_ch})"

        # 백본: 언리시드 SS2D 스택 (마스크 조건화로 c_in=33, 옵션 다운샘플)
        self.ss2d = SS2DStackV9(
            c_in=c_seq_in, d_inner=ss2d_d_inner, d_state=ss2d_d_state,
            out_ch=out_ch, n_blocks=ss2d_n_blocks, dropout=ss2d_dropout,
            use_checkpoint=ss2d_use_checkpoint, downsample=ss2d_downsample,
        )

        # 최종 합성 = 교수님 DFU, complex head (n_classes=2 → DC)
        self.unet = UNet_choh_skip(
            in_channels=num_feat_ch, n_classes=2,
            depth=unet_depth, wf=unet_wf, padding=True,
            batch_norm=False, up_mode='upconv', n_hidden=n_hidden,
        )

        # Data Consistency (측정 앵커) — v8 DCBlock 재사용
        from u_choh_model_SS2D_ViT_v4 import DCBlock
        self.dc = DCBlock(k_scale_ratio=dc_k_scale_ratio, init_alpha=dc_init_alpha)

        print(f"   'PureETER_SS2D_V9_Radapt' (out_ch={out_ch}, d_inner={ss2d_d_inner}, "
              f"d_state={ss2d_d_state}, n_blocks={ss2d_n_blocks}, mask_cond={mask_condition}, "
              f"seq_in={c_seq_in}, in_ch={num_feat_ch}, n_hidden={n_hidden}, DC(α₀={dc_init_alpha}))")

    def forward(self, x_img, x_ksp, mask, sens):
        if self.mask_condition:
            seq_in = torch.cat((x_ksp, mask), dim=1)   # (B, 33, H, W) — 측정연산자 명시
        else:
            seq_in = x_ksp
        out    = self.ss2d(seq_in)                     # (B, out_ch, H, W)
        in_cnn = torch.cat((out, x_img), dim=1)
        out    = self.unet(in_cnn)                     # (B, 2, H, W) complex
        x_ri   = self.dc(out, x_ksp, mask, sens)       # (B, 2, H, W) DC 적용
        out    = torch.sqrt(x_ri[:, 0:1] ** 2 + x_ri[:, 1:2] ** 2 + 1e-12)
        return out
