"""
Transformer 스택 — 도메인 변환 슬롯의 3번째 팔 모듈 (2026-09-02).

구현 방식 = axial attention(행/열 축별 MHSA) — 팔 명칭은 'Transformer'(사용자 지시 09-02),
axial 은 구현 세부를 가리키는 말로만 사용.

설계 근거: docs/axial_transformer_arm_design.md.
GRU(수평/수직 양방향)·SS2D(4방향 스캔)와 **같은 축(행/열)** 을 attention 으로 훑는
가장 공정한 대응물. full 2D attention(384²=147k 토큰)은 제곱 비용으로 불가.

  stem 1×1 conv(c_in→d_model) → +2D sin-cos PE(고정) →
  [행 MHSA → 열 MHSA → FFN] × n_pairs (prenorm + residual) →
  head 1×1 conv(d_model→out_ch)

- dropout 기본 0.0 (v8 SS2D 스택의 검약과 매칭 — 통제판)
- PE 는 고정 sin-cos (학습 파라미터 없음 — 통제 변수 최소화). (H,W)별 lazy 캐시.
- TITAN RTX(Turing, sm_75)는 FlashAttention 불가 → SDPA mem-efficient/math fallback 전제.
- 기본 d_model=64, n_pairs=2 → 스택 ~0.1M (v8 SS2D 스택과 동급 예산 = 통제판).
원본 파일 무수정 — 신규 파일만 추가 (프로젝트 관례).
"""

import math

import torch
import torch.nn as nn


def _sincos_1d(n_pos: int, dim: int, device, dtype) -> torch.Tensor:
    """(n_pos, dim) 표준 sin-cos 인코딩. dim 은 짝수."""
    pos = torch.arange(n_pos, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_pos, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class _AxialPair(nn.Module):
    """행 MHSA → 열 MHSA → FFN (각각 prenorm LayerNorm + residual)."""

    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ln_r  = nn.LayerNorm(d_model)
        self.att_r = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_c  = nn.LayerNorm(d_model)
        self.att_c = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_f  = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model), nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model),
        )

    def forward(self, x):                       # x: (B, H, W, D)
        B, H, W, D = x.shape
        # 행 방향: 각 행 안에서 W 축 self-attention
        t = x.reshape(B * H, W, D)
        h = self.ln_r(t)
        t = t + self.att_r(h, h, h, need_weights=False)[0]
        x = t.reshape(B, H, W, D)
        # 열 방향: 각 열 안에서 H 축 self-attention
        t = x.permute(0, 2, 1, 3).reshape(B * W, H, D)
        h = self.ln_c(t)
        t = t + self.att_c(h, h, h, need_weights=False)[0]
        x = t.reshape(B, W, H, D).permute(0, 2, 1, 3)
        # FFN
        x = x + self.ffn(self.ln_f(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, *, c_in: int = 32, out_ch: int = 20,
                 d_model: int = 64, n_pairs: int = 2, n_heads: int = 4,
                 ffn_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        assert d_model % 2 == 0 and d_model % n_heads == 0
        self.d_model = d_model
        self.stem  = nn.Conv2d(c_in, d_model, kernel_size=1)
        self.pairs = nn.ModuleList(
            _AxialPair(d_model, n_heads, ffn_ratio, dropout) for _ in range(n_pairs))
        self.head  = nn.Conv2d(d_model, out_ch, kernel_size=1)
        self._pe_cache = {}                     # (H, W, device) → (1, D, H, W) fp32

    def _pe(self, H, W, device, dtype):
        key = (H, W, str(device))
        if key not in self._pe_cache:
            half = self.d_model // 2
            pe_h = _sincos_1d(H, half, device, torch.float32)      # (H, D/2)
            pe_w = _sincos_1d(W, self.d_model - half, device, torch.float32)
            pe = torch.cat([
                pe_h[:, None, :].expand(H, W, half),
                pe_w[None, :, :].expand(H, W, self.d_model - half),
            ], dim=-1)                                             # (H, W, D)
            self._pe_cache[key] = pe.permute(2, 0, 1).unsqueeze(0) # (1, D, H, W)
        return self._pe_cache[key].to(dtype)

    def forward(self, x):                       # x: (B, c_in, H, W)
        z = self.stem(x)                        # (B, D, H, W)
        z = z + self._pe(z.shape[-2], z.shape[-1], z.device, z.dtype)
        z = z.permute(0, 2, 3, 1)               # (B, H, W, D)
        for blk in self.pairs:
            z = blk(z)
        z = z.permute(0, 3, 1, 2)               # (B, D, H, W)
        return self.head(z)
