"""
SS2D: 2D Selective State Space module (mamba_ssm CUDA 커널 버전)

ETER-Net의 Bidirectional GRU를 VMamba 스타일의 SS2D로 대체.

스캔 방향 (4방향):
  1. Horizontal left  → right  (각 행을 길이 W=384의 시퀀스로)
  2. Horizontal right → left
  3. Vertical   top   → bottom (각 열을 길이 H=384의 시퀀스로)
  4. Vertical   bottom→ top

References:
  - VMamba: Visual State Space Model (Liu et al., 2024)
  - Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


# ──────────────────────────────────────────────
#  SelectiveScan1D: mamba_ssm CUDA 커널 래퍼
# ──────────────────────────────────────────────

class SelectiveScan1D(nn.Module):
    """
    1D SSM: mamba_ssm의 CUDA 커널(selective_scan_fn)을 사용하는 래퍼.

    입력 x: (B, L, d_inner)
    출력 y: (B, L, d_inner)

    핵심 점화식: x_t = dA_t * x_{t-1} + dB_t * u_t
                 y_t = C_t * x_t + D * u_t

    selective_scan_fn 입력 규격:
      u:     (B, D, L)
      delta: (B, D, L)   > 0
      A:     (D, N)      < 0
      B:     (B, N, L)
      C:     (B, N, L)
      D:     (D,)
    """

    def __init__(self, d_inner: int, d_state: int = 8, dt_rank: int = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(1, d_inner // 8)

        # x → (dt, B_ssm, C_ssm) 투영
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt 투영 (rank → d_inner)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # dt bias 초기화 (Mamba 논문 방식)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A: 상태전이 행렬 (log 형태, 음수 보장)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n', d=d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D: 스킵 커넥션
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_inner)  →  y: (B, L, d_inner)"""
        B_size, L, D = x.shape

        # selective_scan_fn은 float32 필요
        x_f32 = x.float()

        x_dbl = self.x_proj(x_f32).float()  # autocast 시 fp16 방지
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt.float())).float()  # (B, L, D)
        A  = -torch.exp(self.A_log.float())        # (D, N), < 0

        # selective_scan_fn 입력 형식: (B, D, L)
        u     = x_f32.transpose(1, 2)   # (B, D, L)
        delta = dt.transpose(1, 2)       # (B, D, L)
        B_in  = B_ssm.transpose(1, 2)   # (B, N, L)
        C_in  = C_ssm.transpose(1, 2)   # (B, N, L)

        y = selective_scan_fn(u, delta, A, B_in, C_in, self.D.float())  # (B, D, L)

        y = y.transpose(1, 2)  # (B, L, D)
        return y.to(dtype=x.dtype)


# ──────────────────────────────────────────────
#  SS2D: 4-direction 2D selective scan
# ──────────────────────────────────────────────

class SS2D(nn.Module):
    """
    VMamba 스타일 2D 선택적 스캔 모듈.

    입력:  (B, C_in, H, W)  ← MRI k-space 또는 이미지 특징
    출력:  (B, out_ch, H, W)

    구조:
      1. 입력 정규화 + 차원 투영  (C_in → d_inner)
      2. Depthwise Conv2d         (지역 문맥 혼합)
      3. 4방향 SSM 스캔           (GRU 대체)
         - H방향: 각 행을 시퀀스로 (L=W)
         - V방향: 각 열을 시퀀스로 (L=H)
         - 정방향 + 역방향 = 4개
      4. 4방향 출력 병합          (4*d_inner → d_inner)
      5. 출력 투영                (d_inner → out_ch)
    """

    def __init__(
        self,
        c_in: int,           # 입력 채널 수 (k-space: 32)
        d_inner: int = 32,   # SSM 내부 차원
        d_state: int = 8,    # SSM 상태 차원 N
        out_ch: int = 20,    # 출력 채널 수
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_in = c_in
        self.d_inner = d_inner
        self.d_state = d_state
        self.out_ch = out_ch
        dt_rank = max(1, d_inner // 8)

        # 입력 정규화 + 투영
        self.norm_in = nn.LayerNorm(c_in)
        self.in_proj = nn.Linear(c_in, d_inner, bias=False)
        self.act = nn.SiLU()

        # Depthwise Conv: 지역 문맥 혼합
        self.dwconv = nn.Conv2d(
            d_inner, d_inner,
            kernel_size=3, padding=1, groups=d_inner, bias=True
        )

        # 4방향 SSM (각 방향마다 독립 파라미터)
        self.ssm_h_fwd = SelectiveScan1D(d_inner, d_state, dt_rank)  # 좌→우
        self.ssm_h_bwd = SelectiveScan1D(d_inner, d_state, dt_rank)  # 우→좌
        self.ssm_v_fwd = SelectiveScan1D(d_inner, d_state, dt_rank)  # 위→아래
        self.ssm_v_bwd = SelectiveScan1D(d_inner, d_state, dt_rank)  # 아래→위

        # 4방향 결과 병합
        self.merge_norm = nn.LayerNorm(d_inner * 4)
        self.merge = nn.Linear(d_inner * 4, d_inner, bias=False)

        # 출력 투영
        self.out_proj = nn.Conv2d(d_inner, out_ch, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        returns: (B, out_ch, H, W)
        """
        B, C, H, W = x.shape

        # 채널 마지막 형식으로 변환 → 정규화 + 투영
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm_in(x)
        x = self.act(self.in_proj(x))              # (B, H, W, d_inner)

        # Depthwise Conv (지역 문맥)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.act(self.dwconv(x))
        x = rearrange(x, 'b c h w -> b h w c')    # (B, H, W, d_inner)

        # ── 방향 1, 2: 수평 스캔 (각 행 = 길이 W의 시퀀스) ──
        x_h = rearrange(x, 'b h w c -> (b h) w c')   # (B*H, W, d_inner)

        y_h_fwd = self.ssm_h_fwd(x_h)                 # (B*H, W, d_inner)
        y_h_bwd = self.ssm_h_bwd(x_h.flip(1)).flip(1)

        y_h_fwd = rearrange(y_h_fwd, '(b h) w c -> b h w c', b=B)
        y_h_bwd = rearrange(y_h_bwd, '(b h) w c -> b h w c', b=B)

        # ── 방향 3, 4: 수직 스캔 (각 열 = 길이 H의 시퀀스) ──
        x_v = rearrange(x, 'b h w c -> (b w) h c')   # (B*W, H, d_inner)

        y_v_fwd = self.ssm_v_fwd(x_v)                 # (B*W, H, d_inner)
        y_v_bwd = self.ssm_v_bwd(x_v.flip(1)).flip(1)

        y_v_fwd = rearrange(y_v_fwd, '(b w) h c -> b h w c', b=B)
        y_v_bwd = rearrange(y_v_bwd, '(b w) h c -> b h w c', b=B)

        # ── 4방향 병합 ──
        y = torch.cat([y_h_fwd, y_h_bwd, y_v_fwd, y_v_bwd], dim=-1)  # (B, H, W, 4*d_inner)
        y = self.merge_norm(y)
        y = self.merge(y)                             # (B, H, W, d_inner)
        y = self.dropout(y)

        # 출력 투영
        y = rearrange(y, 'b h w c -> b c h w')
        y = self.out_proj(y)                          # (B, out_ch, H, W)
        return y
