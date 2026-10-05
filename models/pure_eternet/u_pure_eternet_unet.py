"""
Pure ETER-Net (U-Net only) — v8 통제비교의 대조 모델: 시퀀스 모듈 제거 ablation (2026-10-01).

교수님 ETER-net 파이프라인에서 시퀀스 모듈 f_θ 를 **제거**하고 나머지는 전부 동일하게 둔다
(데이터·마스크·loss·optimizer·U-Net DFU·에폭·시드). 시퀀스 모듈이 "있다는 것 자체"의 기여를
기준 모델(GRU)·비교 모델(SS2D/Transformer/pixel-GRU)과 같은 U-Net 위에서 분리해 측정하는 실험 조건.

구조: g_φ( cat( 0, F⁻¹y ) )  — 즉 f_θ ≡ 0 (항등적으로 0 을 출력하는 시퀀스 모듈)
  x_ksp ─(사용 안 함)
  zeros(B, 2*H2, H, W) ─┐
                         ├─ cat → UNet_choh_skip(DFU, in_channels=52) → 출력
  x_img(aliased, 32ch) ──┘

왜 0 채널로 채우는가 (U-Net 을 다른 비교 모델과 **완전히 같게** 유지):
  - U-Net in_channels = 2*H2 + 2*coil = 52, n_hidden = H2 + coil = 26 그대로 → 계약(n_hidden*2 == in_ch)과
    U-Net 의 모든 파라미터 shape·state_dict key 가 GRU/SS2D/Transformer/pixel-GRU 와 동일.
  - UNet_choh_skip 은 52ch 원입력을 정확히 두 곳에서만 읽는다:
      (1) 첫 conv  down_path[0].block[0]  (3×3, 52→64)
      (2) 마지막 1×1 conv  last  — forward 끝에서 cat([blocks[0](=원입력), x]) 후 적용
          (myUNet_DF.py:166 `self.last = Conv2d(prev_channels + n_hidden*2, ...)`, :193 `torch.cat([blocks[0], x], 1)`)
    두 conv 모두 입력 채널 0..2*H2-1 이 항상 0 이므로 해당 가중치 slice 의 기여 = 0, gradient 도 정확히 0
    (weight-grad = Σ 입력 × 출력-grad = 0). 따라서 이 모델은 "같은 U-Net 이 시퀀스 출력 없이 aliased image 만
    보는 것"과 함수적으로 동일하며, 52ch → 32ch 로 U-Net 을 축소하는 방식과 달리 U-Net 구조·초기화 분포·
    파라미터 그룹이 다른 실험 조건과 정확히 일치한다 (0 채널 가중치 slice 는 L2 weight decay 로 줄어들 뿐 출력과 무관).
  - DFU 의 residual 경로(UNetUpBlock residual=blocks[-i])는 blocks[1:] (conv 출력)만 쓰므로 원입력과 무관.

forward 시그니처: forward(x_img, x_ksp, mask=None, sens=None) — 다른 비교 모델과 동일.
  x_ksp 는 시퀀스 모듈이 없으므로 no-DC 에서는 읽지 않는다 (use_dc=True 일 때만 DC block 이 사용).
원본 파일 무수정 (UNet_choh_skip 재사용).
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))

from myUNet_DF import UNet_choh_skip     # 교수님 DFU (무수정)


class PureETER_UNET(nn.Module):
    def __init__(
        self,
        *,
        n_coil: int = 16,
        n_hidden_2: int = 10,
        unet_depth: int = 5,
        unet_wf: int = 6,
        use_dc: bool = False,
        dc_k_scale_ratio: float = 100.0,
        dc_init_alpha: float = 1.0,
    ):
        super().__init__()
        self.use_dc = use_dc

        c_in   = n_coil * 2                       # 32 (aliased image real/imag)
        out_ch = 2 * n_hidden_2                   # 시퀀스 모듈 출력 자리 (=20) — 0 으로 채움
        num_feat_ch = out_ch + c_in               # DFU in_channels (=52, 다른 비교 모델과 동일)
        n_hidden    = n_hidden_2 + n_coil
        assert n_hidden * 2 == num_feat_ch, \
            f"UNet_choh_skip 계약 위반: n_hidden*2({n_hidden*2}) != in_channels({num_feat_ch})"
        self.seq_out_ch = out_ch

        # 시퀀스 모듈 없음 (f_θ ≡ 0). 최종 합성 = 교수님 DFU (다른 비교 모델과 동일)
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

        print(f"   'PureETER_UNET' (H2={n_hidden_2}, zero_ch={out_ch} [시퀀스 모듈 없음, f=0], "
              f"in_ch={num_feat_ch}, n_hidden={n_hidden}, unet_depth={unet_depth}, "
              f"use_dc={use_dc})")

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        B, _, H, W = x_img.shape
        zeros  = torch.zeros((B, self.seq_out_ch, H, W),
                             dtype=x_img.dtype, device=x_img.device)   # f_θ ≡ 0 자리
        in_cnn = torch.cat((zeros, x_img), dim=1)  # 채널 순서 = 다른 비교 모델과 동일 (seq 출력, aliased)
        out    = self.unet(in_cnn)                 # (B, 1 or 2, H, W)
        if self.use_dc:
            x_ri = self.dc(out, x_ksp, mask, sens)
            out  = torch.sqrt(x_ri[:, 0:1] ** 2 + x_ri[:, 1:2] ** 2 + 1e-12)
        return out
