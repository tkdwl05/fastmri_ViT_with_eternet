"""
원래 U-Net (Ronneberger et al. 2015 형태) — 원래 U-Net 실험의 정제망 (2026-10-26).

계획: paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan.md §1 (학습 전에 고정).
  - 단계(level)마다 3×3 합성곱 + ReLU 두 번 (패딩 1, 배치 정규화 없음 — DFU 와 같음).
  - 내려갈 때 2×2 max pooling 4 번 (5 단계), 채널 64·128·256·512·1024 (wf = 6, depth = 5).
  - 올라갈 때 2×2 전치 합성곱(stride 2)으로 올리고, 같은 단계의 인코더 출력(skip)을 채널 방향으로 이어 붙인 뒤 3×3 합성곱 두 번.
  - 마지막 1×1 합성곱으로 n_classes(= 1) 채널 출력.
  - dual-frame 잔차 빼기 없음, 입력 재연결 없음. 입력 채널 수(in_channels)는 인자다(U-Net only (OU)·d-ETER (OU) 52, o-ETER (OU) 20).

교수님 DFU(models/hybrid_eternet/myUNet_DF.py 의 UNet_choh_skip)와 정제망 구조에서 다른 점은 다음 세 가지뿐이다:
  (1) 내려갈 때 2×2 max pooling — DFU 는 F.avg_pool2d (myUNet_DF.py:176).
  (2) 올라갈 때 dual-frame 잔차 빼기 없음 — UNetUpBlock 을 residual 없이 부른다. DFU 는 두 번째 올림부터
      up(x, blocks[-i-1], blocks[-i]) 로 불러 x − blocks[-i] 를 올린다 (myUNet_DF.py:183-186, :125-128).
  (3) 마지막 1×1 합성곱이 디코더 출력(64 채널)만 읽는다 — DFU 는 cat([원입력, 디코더 출력]) 을 읽는다
      (in_channels + 64 → 1, myUNet_DF.py:166, :193).
같은 것: 합성곱 블록·올림 블록은 myUNet_DF.py 의 UNetConvBlock·UNetUpBlock 을 그대로 import 해 쓴다(교수님 원본 무수정,
  padding=True, batch_norm=False, up_mode='upconv'). 초기화도 같다(PyTorch 기본값, 따로 초기화하지 않음). 하위 모듈 이름
  (down_path / up_path / last)과 만드는 순서도 DFU 와 같다. 그래서
    - v8fix 의 건강 감시 그룹(smoke_recipe_fix.build_groups: unet.down_path.i · unet.up_path.i · unet.last, 깊은 층
      DEEP_PREFIXES = down_path.3 · down_path.4 · up_path.0)과 기능 시험(deep4 = down_path.4 출력 0 대체, deep34 = down_path.3)이
      DFU 와 같은 단계를 가리킨다.
    - 같은 난수 상태에서 만들면 last 를 뺀 모든 합성곱·전치 합성곱 텐서의 초기값이 같은 입력 채널 수의 DFU 와 같다
      (모양과 만드는 순서가 같음; last 만 모양이 다르다 — DFU (1, in_channels + 64, 1, 1), 여기 (1, 64, 1, 1)).

입력 크기: 높이·너비가 2^(depth−1)(= 16)의 배수여야 한다(384 = 16 × 24). 그러면 UNetUpBlock.center_crop 은 아무것도 자르지 않는다.
"""

import os
import sys

import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))

from myUNet_DF import UNetConvBlock, UNetUpBlock   # noqa: E402  교수님 DFU 의 building block (무수정 재사용)


class OriginalUNet(nn.Module):
    """원래 U-Net 정제망. forward(x): (B, in_channels, H, W) → (B, n_classes, H, W)."""

    def __init__(self, in_channels: int, n_classes: int = 1, depth: int = 5, wf: int = 6):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_classes = int(n_classes)
        self.depth = int(depth)
        self.wf = int(wf)
        prev = self.in_channels
        self.down_path = nn.ModuleList()
        for i in range(self.depth):
            self.down_path.append(UNetConvBlock(prev, 2 ** (self.wf + i), True, False))   # padding=True, batch_norm=False
            prev = 2 ** (self.wf + i)
        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            self.up_path.append(UNetUpBlock(prev, 2 ** (self.wf + i), 'upconv', True, False))
            prev = 2 ** (self.wf + i)
        self.last = nn.Conv2d(prev, self.n_classes, kernel_size=1)    # 디코더 출력(64 채널)만 — 입력 재연결 없음

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f'OriginalUNet: 입력 모양 {tuple(x.shape)} — (B, {self.in_channels}, H, W) 이어야 함')
        k = 2 ** (self.depth - 1)
        if x.shape[-2] % k or x.shape[-1] % k:
            raise ValueError(f'OriginalUNet: 높이·너비 {tuple(x.shape[-2:])} 가 {k} 의 배수가 아님')
        skips = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)                 # 2×2 max pooling (DFU 는 avg_pool2d)
        for i, up in enumerate(self.up_path):
            x = up(x, skips[-i - 1])                   # residual 인자 없음 = dual-frame 잔차 빼기 없음
        return self.last(x)

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, n_classes={self.n_classes}, depth={self.depth}, wf={self.wf}, '
                f'pool=max2x2, up=convtranspose2x2, residual_subtraction=False, input_reconcat=False')
