"""
SS2D v9 — 언리시드 Mamba: 게이팅(z) + residual 스택 SS2D.

v8 의 `ss2d.py::SS2D` 는 (1) 단일 블록, (2) 4방향 scan 후 곧장 out_proj,
(3) 공식 Mamba 블록의 `y = y·SiLU(z)` 게이팅 분기가 **없다**. v9 는 이를 정식 Mamba
블록에 맞추고(게이팅 복원) residual 스택 가능하게 만든다.

  SS2DBlockV9 : d_inner 공간에서 채널수 불변 residual + 게이팅 (스택 단위)
  SS2DStackV9 : stem(c_in→d_inner) → N×블록 → head(d_inner→out_ch, 병목 자유)

원본 `ss2d.py::SelectiveScan1D`(dt_proj/A_log/D 초기화 포함, mamba_ssm CUDA 커널 래퍼)를
**그대로 import 재사용** — 원본 파일 무수정.

References:
  - Mamba: Gu & Dao, 2023 (게이팅 SiLU(z) 분기가 selective SSM 블록의 핵심)
  - VMamba: Liu et al., 2024 (4방향 2D 스캔)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class SelectiveScan1DV9(nn.Module):
    """원본 `ss2d.py::SelectiveScan1D` 와 **동일 수식·초기화**이나 스캔을 **fp16/bf16 허용**한다.

    원본은 `x.float()` 로 스캔을 강제 fp32 화한다 → 메모리 2×·속도 저하 → v9 3블록에서 checkpointing
    강제 → 2× 재계산. mamba_ssm 의 selective_scan_fn 은 u/delta/B/C 를 fp16/bf16 로 받고 **내부
    recurrence 는 fp32 누적**(안정)이므로, autocast dtype 그대로 흘려도 수치적으로 안전하다
    (벤치: fp16 이 fp32 대비 메모리 정확히 절반, ~25% 빠름, 출력 finite). A_log/D 는 fp32 유지.

    원본 무수정 원칙 준수: `ss2d.py` 는 손대지 않고 v9 전용으로 새로 정의(초기화 로직 동일 복제).
    """

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(1, d_inner // 8)

        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_inner) → y: (B, L, d_inner). 스캔은 x.dtype(fp16 가능) 로 수행."""
        x_dbl = self.x_proj(x)                                   # autocast dtype 유지
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                        # (B, L, D)
        A  = -torch.exp(self.A_log.float())                      # (D, N) fp32 (커널 요구)

        # selective_scan_fn 은 u/delta/B/C 가 **동일 dtype** 이어야 함. autocast 가 softplus 를 fp32 로
        # 승격시켜 delta 만 fp32 가 되므로 u.dtype 으로 통일 (A/D 는 fp32 유지 = 커널 계약).
        sdt   = x.dtype
        u     = x.transpose(1, 2)                    # (B, D, L)
        delta = dt.transpose(1, 2).to(sdt)           # (B, D, L)
        B_in  = B_ssm.transpose(1, 2).to(sdt)        # (B, N, L)
        C_in  = C_ssm.transpose(1, 2).to(sdt)        # (B, N, L)

        y = selective_scan_fn(u, delta, A, B_in, C_in, self.D.float())   # 내부 fp32 누적
        return y.transpose(1, 2)         # (B, L, D)


class SS2DBlockV9(nn.Module):
    """정식 Mamba 스타일 게이팅 + residual 블록 (채널수 d_inner 불변, 스택 가능).

    forward(x): x, out 모두 (B, H, W, d_inner)
        residual = x
        x = LayerNorm(x)
        x_ssm, z = in_proj(x).chunk(2)          # 게이팅 분기 분리
        x_ssm = SiLU(dwconv(x_ssm))             # 지역 문맥
        y = 4방향 selective-scan(x_ssm) → merge # (B,H,W,d_inner)
        y = y * SiLU(z)                          # ★ 게이팅 (v8 누락분)
        y = out_proj(y)
        return residual + Dropout(y)
    """

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = None,
                 dropout: float = 0.0):
        super().__init__()
        self.d_inner = d_inner

        self.norm = nn.LayerNorm(d_inner)
        # in_proj: d_inner → 2*d_inner  (SSM 입력 x_ssm + 게이트 z)
        self.in_proj = nn.Linear(d_inner, 2 * d_inner, bias=False)
        self.act = nn.SiLU()

        # Depthwise Conv (지역 문맥 혼합) — SSM 분기에만 적용
        self.dwconv = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1,
                                groups=d_inner, bias=True)

        # 4방향 SSM (각 방향 독립 파라미터) — v9 fp16 허용 스캔
        self.ssm_h_fwd = SelectiveScan1DV9(d_inner, d_state, dt_rank)  # 좌→우
        self.ssm_h_bwd = SelectiveScan1DV9(d_inner, d_state, dt_rank)  # 우→좌
        self.ssm_v_fwd = SelectiveScan1DV9(d_inner, d_state, dt_rank)  # 위→아래
        self.ssm_v_bwd = SelectiveScan1DV9(d_inner, d_state, dt_rank)  # 아래→위

        # 4방향 병합
        self.merge_norm = nn.LayerNorm(d_inner * 4)
        self.merge = nn.Linear(d_inner * 4, d_inner, bias=False)

        # 게이팅 후 출력 투영
        self.out_proj = nn.Linear(d_inner, d_inner, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _scan2d(self, x):
        """x: (B, H, W, d_inner) → y: (B, H, W, d_inner)  (4방향 스캔 + 병합)."""
        B = x.shape[0]

        # ── 수평 스캔 (각 행 = 길이 W 시퀀스) ──
        x_h = rearrange(x, 'b h w c -> (b h) w c')
        y_h_fwd = self.ssm_h_fwd(x_h)
        y_h_bwd = self.ssm_h_bwd(x_h.flip(1)).flip(1)
        y_h_fwd = rearrange(y_h_fwd, '(b h) w c -> b h w c', b=B)
        y_h_bwd = rearrange(y_h_bwd, '(b h) w c -> b h w c', b=B)

        # ── 수직 스캔 (각 열 = 길이 H 시퀀스) ──
        x_v = rearrange(x, 'b h w c -> (b w) h c')
        y_v_fwd = self.ssm_v_fwd(x_v)
        y_v_bwd = self.ssm_v_bwd(x_v.flip(1)).flip(1)
        y_v_fwd = rearrange(y_v_fwd, '(b w) h c -> b h w c', b=B)
        y_v_bwd = rearrange(y_v_bwd, '(b w) h c -> b h w c', b=B)

        y = torch.cat([y_h_fwd, y_h_bwd, y_v_fwd, y_v_bwd], dim=-1)  # (B,H,W,4d)
        y = self.merge_norm(y)
        y = self.merge(y)                                            # (B,H,W,d)
        return y

    def forward(self, x):
        """x: (B, H, W, d_inner) → (B, H, W, d_inner)."""
        residual = x
        x = self.norm(x)

        x_ssm, z = self.in_proj(x).chunk(2, dim=-1)   # 각 (B,H,W,d_inner)

        # Depthwise conv (SSM 분기)
        x_ssm = rearrange(x_ssm, 'b h w c -> b c h w')
        x_ssm = self.act(self.dwconv(x_ssm))
        x_ssm = rearrange(x_ssm, 'b c h w -> b h w c')

        y = self._scan2d(x_ssm)                        # (B,H,W,d_inner)
        y = y * self.act(z)                            # ★ 게이팅 곱 (공식 Mamba)
        y = self.out_proj(y)
        y = self.dropout(y)
        return residual + y


class SS2DStackV9(nn.Module):
    """언리시드 SS2D 스택: stem → N×게이팅 residual 블록 → head(병목 자유).

    입력:  (B, c_in, H, W)     ← MRI k-space
    출력:  (B, out_ch, H, W)   ← out_ch 는 GRU-정합 제약 없음 (자유)
    """

    def __init__(self, c_in: int, d_inner: int = 256, d_state: int = 32,
                 out_ch: int = 64, n_blocks: int = 3, dt_rank: int = None,
                 dropout: float = 0.0, use_checkpoint: bool = False,
                 downsample: int = 1):
        super().__init__()
        self.c_in = c_in
        self.d_inner = d_inner
        self.d_state = d_state
        self.out_ch = out_ch
        self.n_blocks = n_blocks
        self.use_checkpoint = use_checkpoint
        self.downsample = downsample

        # stem: c_in → d_inner (full-res)
        self.norm_in = nn.LayerNorm(c_in)
        self.in_proj = nn.Linear(c_in, d_inner, bias=False)
        self.act = nn.SiLU()

        # (옵션) 다운샘플: 스캔을 coarse grid 에서 → 스캔 cost ∝ 면적이라 ds²× 절감.
        # SSM=전역 문맥(coarse), U-Net=풀해상도 디테일 분업(VMamba/vision-mamba 정석).
        if downsample > 1:
            self.down = nn.Conv2d(d_inner, d_inner, kernel_size=downsample, stride=downsample)
            self.up_proj = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1)  # interp 후 정련
        else:
            self.down = None
            self.up_proj = None

        # body: N × residual 게이팅 블록 (coarse grid)
        self.blocks = nn.ModuleList([
            SS2DBlockV9(d_inner, d_state, dt_rank, dropout)
            for _ in range(n_blocks)
        ])

        # head: d_inner → out_ch (병목 해제, full-res)
        self.norm_out = nn.LayerNorm(d_inner)
        self.out_proj = nn.Conv2d(d_inner, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        """x: (B, c_in, H, W) → (B, out_ch, H, W)."""
        H, W = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm_in(x)
        x = self.act(self.in_proj(x))                  # (B,H,W,d_inner)

        if self.down is not None:                      # → coarse grid
            x = rearrange(x, 'b h w c -> b c h w')
            x = self.act(self.down(x))
            x = rearrange(x, 'b c h w -> b h w c')

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm_out(x)
        x = rearrange(x, 'b h w c -> b c h w')

        if self.up_proj is not None:                   # → full-res 복원
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = self.act(self.up_proj(x))

        x = self.out_proj(x)                           # (B, out_ch, H, W)
        return x
