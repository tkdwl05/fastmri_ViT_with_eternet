"""
Pixel-scan bi-GRU 스택 — 도메인 변환 슬롯의 4번째 팔(가중치 공유 재귀) 모듈 (2026-09-02).

목적(공정성 — docs/v8_fairness_followup_plan.md ③): v8 원본 ETER-GRU 는
(i) 재귀 메커니즘과 (ii) flatten-reshape 파라미터화(한 줄 12,288차원 입력·hidden 384 양자화·
하한 63M)가 얽혀 있어 "재귀 vs SSM" 을 격리하지 못한다. 이 모듈은 재귀에게 SS2D 와 같은
**공간 가중치 공유** 몸을 준다: 픽셀을 시퀀스 원소로, 행(W=384-step)→열(H=384-step)
양방향 GRU 스캔 — 모든 위치가 같은 가중치.

  x (B,32,H,W) → 행 bi-GRU(입력 c_in, hidden h) → (B,2h,H,W)
              → 열 bi-GRU(입력 2h, hidden h)   → (B,2h,H,W) → head 1×1 conv → out_ch

- 기본 h=64: 스택 ~0.115M — SS2D 스택(~0.1M)·axial 스택(0.104M)과 동급 예산
  (flatten-GRU 에선 불가능했던 예산 매칭이 가중치 공유로는 가능하다는 것 자체가 논점).
- stem 없음: 원본 gru_h 가 k-space 를 날것으로 받는 것과 대칭('순정 재귀').
- cuDNN GRU(fp16 autocast OK). 해석: pixel-GRU ≈ SS2D → v8 간극은 파라미터화 탓(재귀 무죄) /
  pixel-GRU < SS2D → 선택적 상태전이라는 메커니즘 차이 실재.
원본 파일 무수정 — 신규 파일만 추가 (프로젝트 관례).
"""

import torch
import torch.nn as nn


class PixelGRUStack(nn.Module):
    def __init__(self, *, c_in: int = 32, out_ch: int = 20, hidden: int = 64):
        super().__init__()
        self.gru_row = nn.GRU(c_in, hidden, batch_first=True, bidirectional=True)
        self.gru_col = nn.GRU(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.head    = nn.Conv2d(2 * hidden, out_ch, kernel_size=1)

    def forward(self, x):                        # x: (B, C, H, W)
        B, C, H, W = x.shape
        self.gru_row.flatten_parameters()
        self.gru_col.flatten_parameters()
        # 행 스캔: 각 행을 W-step 시퀀스로 (양방향 = 좌→우 + 우→좌)
        t = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        t, _ = self.gru_row(t)                   # (B*H, W, 2h)
        t = t.reshape(B, H, W, -1)
        # 열 스캔: 각 열을 H-step 시퀀스로 (상→하 + 하→상)
        u = t.permute(0, 2, 1, 3).reshape(B * W, H, t.shape[-1])
        u, _ = self.gru_col(u)                   # (B*W, H, 2h)
        u = u.reshape(B, W, H, -1).permute(0, 3, 2, 1)   # (B, 2h, H, W)
        return self.head(u)
