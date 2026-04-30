"""
SS2D-ViT 모델 v4: dropout + complex 출력 + Data Consistency block

v3 대비 변경점:
  - Transformer decoder에 dropout 전달 (과적합 억제)
  - RefinementBlock 출력 1ch(magnitude) → 2ch(real/imag)
  - 1-iteration Soft DC block 추가 (ACS-based sens, 학습 가능한 α)
  - forward()에 mask, sens 파라미터 추가. 반환은 magnitude (1, H, W)로 loss 호환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
import math
import sys
import os

# SS2D 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ss2d import SS2D

# 원본 ViT 인코더 임포트
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hybrid_eternet'))
from u_choh_model_ETER_ViT import choh_ViT


# ──────────────────────────────────────────────
#  공용 Transformer 블록 (원본과 동일, checkpoint 적용)
# ──────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = checkpoint.checkpoint(attn, x, use_reentrant=False) + x
            x = checkpoint.checkpoint(ff,   x, use_reentrant=False) + x
        return self.norm(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * 4, 3, padding=1)
    def forward(self, x):
        return F.pixel_shuffle(self.conv(x), 2)


class ResBlock(nn.Module):
    """잔차 합성곱 블록: Conv→LeakyReLU→Conv + skip connection."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.act(self.conv1(x))))


class RefinementBlock(nn.Module):
    """Conv 1개를 대체하는 잔차 합성 블록 (3×ResBlock)."""
    def __init__(self, in_ch, mid_ch=64, out_ch=1, num_blocks=3):
        super().__init__()
        self.head = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(mid_ch) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(mid_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act(self.head(x))
        x = self.body(x)
        return self.tail(x)


# ──────────────────────────────────────────────
#  Data Consistency block (1-iteration, soft DC)
# ──────────────────────────────────────────────

class DCBlock(nn.Module):
    """
    Single-iteration soft Data Consistency.

    입력:
      x_ri         (B, 2, H, W)   — 모델이 추정한 coil-combined complex image (real, imag)
      k_meas_ri    (B, 32, H, W)  — dataloader의 masked k-space, val_amp_X_ksp 스케일, real/imag 교번
      mask         (B, 1, H, W)   — 샘플링 마스크 (1=sampled)
      sens_ri      (B, 32, H, W)  — ACS 기반 sensitivity map (Σ|s|²=1), real/imag 교번
    출력:
      (B, 2, H, W) — DC 적용된 coil-combined complex image

    내부 처리:
      1) real/imag-packed 텐서를 complex로 재구성
      2) 측정 k-space를 image 스케일로 재조정 (k_scale_ratio)
      3) multicoil = sens · x_c → FFT → k_pred
      4) k_dc = k_pred + mask·α·(k_meas_scaled − k_pred)
      5) iFFT → Σ multicoil_dc · sens* → coil-combined
    """
    def __init__(self, k_scale_ratio=100.0, init_alpha=1.0):
        super().__init__()
        self.register_buffer('k_scale_ratio', torch.tensor(float(k_scale_ratio), dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    @staticmethod
    def _fft2c(x_complex):
        x = torch.fft.ifftshift(x_complex, dim=(-2, -1))
        x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        return torch.fft.fftshift(x, dim=(-2, -1))

    @staticmethod
    def _ifft2c(x_complex):
        x = torch.fft.ifftshift(x_complex, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
        return torch.fft.fftshift(x, dim=(-2, -1))

    def forward(self, x_ri, k_meas_ri, mask, sens_ri):
        # FFT는 fp32/complex64에서만 수행 (amp fp16 autocast 보호)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x_ri      = x_ri.float()
            k_meas_ri = k_meas_ri.float()
            mask      = mask.float()
            sens_ri   = sens_ri.float()

            # real/imag pack → complex
            x_c      = torch.complex(x_ri[:, 0], x_ri[:, 1]).unsqueeze(1)         # (B, 1, H, W)
            k_meas_c = torch.complex(k_meas_ri[:, 0::2], k_meas_ri[:, 1::2])       # (B, 16, H, W)
            sens_c   = torch.complex(sens_ri[:, 0::2],  sens_ri[:, 1::2])          # (B, 16, H, W)

            # 측정 k-space를 image 스케일로 재조정
            k_meas_scaled = k_meas_c * self.k_scale_ratio

            # 예측 multicoil + FFT
            multicoil = sens_c * x_c                 # (B, 16, H, W)
            k_pred    = self._fft2c(multicoil)

            # Soft DC
            alpha = self.alpha
            k_dc  = k_pred + mask * alpha * (k_meas_scaled - k_pred)

            # 다시 image 도메인 → coil combine
            multicoil_dc = self._ifft2c(k_dc)
            x_comb       = torch.sum(multicoil_dc * sens_c.conj(), dim=1)  # (B, H, W)

            out_ri = torch.stack([x_comb.real, x_comb.imag], dim=1)        # (B, 2, H, W)
        return out_ri


# ──────────────────────────────────────────────
#  SS2D-ViT 디코더
# ──────────────────────────────────────────────

class choh_Decoder_SS2D_ViT(nn.Module):
    """
    ViT 인코더 + SS2D 디코더.

    GRU 대체 구조:
      원본: gru_h (수평 양방향 GRU) + gru_v (수직 양방향 GRU)
      신규: SS2D (수평 ×2 + 수직 ×2 = 4방향 SSM)

    출력 채널 수 변경:
      원본: 2 * eter_n_vert_hidden  (= 20, GRU 양방향 출력)
      신규: ss2d_out_ch             (기본값 20, 동일하게 맞춤)
    """

    def __init__(
        self,
        *,
        encoder,
        # SS2D 파라미터
        ss2d_d_inner: int = 32,
        ss2d_d_state: int = 8,
        ss2d_out_ch:  int = 20,
        # 디코더 파라미터
        decoder_dim,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
        decoder_dim_mlp_hidden=3072,
        decoder_out_ch_up_tail=4,
        decoder_out_feat_size_final_linear=32,
        # v4 추가
        dropout: float = 0.1,
        dc_k_scale_ratio: float = 100.0,
        dc_init_alpha: float = 1.0,
    ):
        super().__init__()
        print("   'choh_Decoder_SS2D_ViT_v4  @u_choh_model_SS2D_ViT_v4'   ")

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        # ── ViT 디코더 (v4: dropout 적용) ──
        self.decoder_dim = decoder_dim
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim else nn.Identity()
        )
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim, depth=decoder_depth, heads=decoder_heads,
            dim_head=decoder_dim_head, mlp_dim=decoder_dim_mlp_hidden,
            dropout=dropout,
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)

        # ── 업샘플링 테일 (원본과 동일) ──
        self.decoder_out_ch_up_tail = decoder_out_ch_up_tail
        self.decoder_out_feat_size_final_linear = decoder_out_feat_size_final_linear
        dim_for_final_linear = (
            decoder_out_ch_up_tail
            * decoder_out_feat_size_final_linear
            * decoder_out_feat_size_final_linear
        )
        self.final_linear = nn.Linear(decoder_dim, dim_for_final_linear)

        up_steps = int(
            math.log(encoder.patch_size[0], 2)
            - math.log(decoder_out_feat_size_final_linear, 2)
        )
        self.up_tail = nn.Sequential(*[Upsample(decoder_out_ch_up_tail) for _ in range(up_steps)])

        # ── SS2D (GRU 대체) ──
        self.ss2d = SS2D(
            c_in=32,                  # k-space: 16코일 × 2(실수/허수) = 32채널
            d_inner=ss2d_d_inner,
            d_state=ss2d_d_state,
            out_ch=ss2d_out_ch,
        )
        self.ss2d_out_ch = ss2d_out_ch

        # ── 최종 합성 (v4: 2ch complex 출력 + DC block) ──
        num_ch_last = decoder_out_ch_up_tail + 32 + ss2d_out_ch
        self.last = RefinementBlock(in_ch=num_ch_last, mid_ch=64, out_ch=2, num_blocks=3)
        self.dc = DCBlock(k_scale_ratio=dc_k_scale_ratio, init_alpha=dc_init_alpha)

    def forward(self, in_imgs, in_ksp, mask, sens):
        """
        in_imgs: 앨리어싱 이미지        (B, 32, H, W)
        in_ksp:  언더샘플링 k-space     (B, 32, H, W)   — val_amp_X_ksp 스케일
        mask:    샘플링 마스크          (B, 1, H, W)    — 1=sampled
        sens:    sens map (real/imag)   (B, 32, H, W)   — ACS 기반, Σ|s|²=1
        returns: 재구성 magnitude        (B, 1, H, W)
        """
        device = in_imgs.device

        # ── ViT 인코더 (원본과 동일) ──
        patches = rearrange(
            in_imgs, 'bb cc (h p1) (w p2) -> bb (h w) (p1 p2 cc)',
            p1=self.encoder.patch_size[0], p2=self.encoder.patch_size[1]
        )
        batch, num_patches, *_ = patches.shape

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)

        indices = repeat(torch.arange(num_patches, device=device), 'pp -> bb pp', bb=batch)
        tokens = tokens[torch.arange(batch, device=device)[:, None], indices]
        encoded_tokens = self.encoder.transformer(tokens)

        # ── ViT 디코더 + 업샘플 (원본과 동일) ──
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(indices)
        decoded_tokens = self.decoder(decoder_tokens)

        pred_latent = self.final_linear(decoded_tokens)
        x = rearrange(
            pred_latent,
            'bb (nh nw) (cc p1 p2) -> bb cc (nh p1) (nw p2)',
            nh=self.encoder.num_patch_h, nw=self.encoder.num_patch_w,
            p1=self.decoder_out_feat_size_final_linear,
            p2=self.decoder_out_feat_size_final_linear
        )
        x = self.up_tail(x)  # (B, decoder_out_ch_up_tail, H, W)

        # ── SS2D: k-space 아티팩트 보정 (GRU 대체) ──
        # v4 OOM 수정 (2026-04-27): merge_norm 단계 (B,H,W,4*d_inner=256) 임시버퍼가
        # 8GB GPU에서 OOM 유발. forward를 checkpoint로 감싸 backward 시 재계산.
        if self.training:
            out_ss2d = checkpoint.checkpoint(self.ss2d, in_ksp, use_reentrant=False)
        else:
            out_ss2d = self.ss2d(in_ksp)  # (B, ss2d_out_ch, H, W)

        # ── 최종 합성: 2ch complex 출력 ──
        x = torch.cat([x, in_imgs, out_ss2d], dim=1)
        x_ri = self.last(x)                         # (B, 2, H, W) — real/imag

        # ── Data Consistency (soft, 1-iter) ──
        x_ri = self.dc(x_ri, in_ksp, mask, sens)    # (B, 2, H, W)

        # ── Magnitude 반환 (기존 loss 호환) ──
        mag = torch.sqrt(x_ri[:, 0:1] ** 2 + x_ri[:, 1:2] ** 2 + 1e-12)
        return mag
