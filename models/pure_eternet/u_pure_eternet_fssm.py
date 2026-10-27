"""
FSSM (OU) — 푸리에 변환을 표현할 수 있는 복소 SSM 시퀀스 모듈 (2026-10-27) — 설계 docs/fssm_design_2026-10-27.md (결과 전 고정).
(사용자 결정 10-27: 다음 논문 줄기 — ① 푸리에 변환을 표현할 수 있는 SSM 으로 변환 모듈을 다시 설계, ② R=8·무작위 평가.)

| 모델                 | 클래스            | 시퀀스 모듈(self.ss2d)                         | 정제망           |
|----------------------|-------------------|------------------------------------------------|------------------|
| FSSM (OU)            | PureETER_FSSMOU   | FSSM: 복소 선택적 SSM 보정 + 고정 IFT          | OriginalUNet(32) |
| FSSM-real (OU) (절제)| PureETER_FSSMROU  | 같은 구조, A 실수(Mamba 기본 초기화)            | OriginalUNet(32) |
| FSSM-T (OU) (절제)   | PureETER_FSSMTOU  | 복소 SSM 보정 + 학습형 Vandermonde 변환         | OriginalUNet(32) |

FSSM 의 계산 (x_ksp = 정규화된 k-space (B, 32, H, W), 16코일 real/imag 교번; 판독 축 = H(−2, 완전 샘플), 위상 부호화(마스크)
축 = W(−1, 로더의 R4 마스크가 고르는 줄)):
  1. z = ifft1c_H(x_ksp)                       하이브리드 공간(복소 (B, 16, H, W)) — 판독 방향 고정 1D IFT (DOTA 의 착안)
  2. m_w = 1[max_{c,h}|x_ksp| > 0]              측정 줄 (B, W) — forward 에 마스크 인자가 없으므로 입력에서 읽는다
  3. e = 측정 줄만의 코일 RMS |z| 를 W 축 폭 15 창으로 평균(+ 하한 1e-6·mean) — SS2D 의 위치별 LayerNorm 대신(세기를 지우지 않고
     마지막에 다시 곱함). R ≤ 8 이면 창 안에 측정 줄이 있다.
  4. 행 (b, h) 마다 W 축 수열, 위치 특징 [Re z/e (16), Im z/e (16), m_w] = 33 → in_proj(33 → 2D) → (u, gate) → depthwise
     Conv1d(k=3, 가운데) + SiLU → 정방향·역방향 선택적 SSM(방향마다 x_proj·dt_proj·A·D 따로) → (y_f + y_b)·SiLU(gate)
     → out_proj(D → 32, weight·bias 0 초기화) = r̂ → r = e · complex(r̂)
  5. y = ifft1c_W(z + r)  (FSSM-T 는 학습형 변환 T) → 16코일 영상 (B, 32, H, W) real/imag 교번, float32
  초기값에서 r = 0 이므로 y = ifft2c(x_ksp) = x_img (정규화 뒤에도 성립 — smoke_recipe_fix.normalize_inputs). 정제망 첫 합성곱을
  U-Net only (OU) 의 영상 채널 자리 [:, 20:52] 에서 복사하면(recipe_ou_fssm.copy_unet_init) 초기 모델 = U-Net only (OU) 초기 모델.

복소 A: A = −exp(A_re_log) + i·A_im, S4D-Lin 초기화 A_n = −1/2 + iπn (n = 0..N−1, N = 16). 줄당 위상 πnΔ (Δ 는 Mamba 기본
[1e-3, 1e-1] 로그 균일)이 채널에 걸쳐 [0, ~1.5π] 를 덮어 앨리어싱 주기 성분 π/2(R4)·π/4(R8)를 포함한다.
실수판(절제): A = −exp(A_log), Mamba 기본 A_n = −(n+1), 상태 수 2N = 32 (복소 16 상태 = 실수 32 개와 같은 크기) — x_proj 출력 수
(dt_rank + 64)와 A 파라미터 수(64·32)가 복소판과 같다.

mamba_ssm 2.2.2 의 selective_scan 규약 (ops/selective_scan_interface.py 를 읽고 확인, 10-27):
  - u: 실수 (B, D, L), delta: 실수 (B, D, L) — 여기서는 softplus 를 밖에서 하고 넘긴다(ss2d.py SelectiveScan1D 와 같음).
  - A 가 복소 (D, N) 이면 입력 의존 B·C 는 실수 텐서 (B, N, 2L) 로 넘긴다 — 마지막 축이 위치마다 (re, im) 교번
    ("... (L two) -> ... L two" 후 view_as_complex). 이산화 dA = exp(Δ·A), dB·u = Δ·B·u (Mamba 와 같음).
  - 출력 y_t = 2·Re(Σ_n C_{t,n} x_{t,n}) + D·u_t (복소면 실수부 ×2 — 켤레쌍 상태를 하나로 친 것), 실수 (B, D, L).
  - CUDA 커널(selective_scan_cuda)에 c10::complex<float> 가중치판이 있다(nm 확인). 같은 배치를 받는다 — sanity_fssm.py --gpu 가
    커널 = ref 를 확인한다. CPU 에서는 selective_scan_ref(같은 식의 순차 계산)를 쓴다 — _scan 이 장치로 고른다.
  - 스캔은 SS2D(controlled) 처럼 fp32 입력으로 돈다(projection 은 autocast 그대로).

학습형 변환(FSSM-T, design §3.6): T[n,t] = c_n · λ_n^{N−1−t} · b_t, λ_n = exp(−δ_n − iφ_n), N = W.
  초기값 = 중심 IDFT (fftshift∘ifft∘ifftshift, ortho; N/2 짝수에서 M[n,t] = N^{−1/2}(−1)^{t+n}e^{2πitn/N}):
  φ_n = 2πn/N + π, δ_n ≈ 0, b_t = 1, c_n = N^{−1/2}(−1)^n e^{iφ_n(N−1)}.
  학습 변수: φ = φ0 + η/N (η 초기 0), δ = softplus(ρ)/N (ρ 초기 −14 → δ ≈ 2e-9, 초기 오차 ≈ 8e-7) — 설계 문서의 relu(ρ) 는 ρ = 0 에서
  기울기가 0 이라 감쇠를 배울 수 없어 softplus 로 바꿨다(구현 중 변경, 결과 전). φ0·(N−1−t) 의 2π 나머지는 정수 산술로 미리 계산해
  fp32 위상 오차를 막는다.

건강 감시: 0 초기화 텐서(out_proj, A_im 의 n = 0 열, η, b 의 허수부)는 seq 파라미터의 소수 — sanity_fssm.py 가 비율을 잰다.
교수님·프로젝트 원본 파일과 기존 OU 파일(recipe_ou·recipe_ou_ss2d·트레이너·모델)은 고치지 않는다(새 파일).
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))

from unet_original import OriginalUNet                                              # noqa: E402
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref  # noqa: E402

ENVELOPE_WIDTH = 15          # 측정 줄 평균 창 (R ≤ 8 을 덮음)
ENVELOPE_FLOOR = 1e-6        # 하한 = 1e-6 · 샘플별 평균
RHO_INIT = -14.0             # 학습형 변환의 감쇠 초기값 (softplus(−14)/N ≈ 2e-9)


def _scan(u, delta, A, B, C, D):
    """selective scan — CUDA 면 커널(selective_scan_fn), 아니면 selective_scan_ref (같은 식)."""
    if u.is_cuda:
        return selective_scan_fn(u, delta, A, B, C, D)
    return selective_scan_ref(u, delta, A, B, C, D)


def ri_to_complex(x):
    """(B, 2n, H, W) real/imag 교번 → (B, n, H, W) complex64."""
    x = x.float()
    return torch.complex(x[:, 0::2], x[:, 1::2])


def complex_to_ri(z):
    """(B, n, H, W) complex → (B, 2n, H, W) real/imag 교번 float32."""
    return torch.stack((z.real, z.imag), dim=2).reshape(z.shape[0], 2 * z.shape[1], *z.shape[2:]).float()


def ifft1c(z, dim):
    """중심 1D IFFT — 로더 규약(ifftshift → ifft(norm='ortho') → fftshift)을 한 축에."""
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(z, dim=dim), dim=dim, norm='ortho'), dim=dim)


def measured_lines(x_ksp):
    """측정 줄 표시 (B, W) float32: 코일·판독 축 어디든 0 이 아니면 1 (마스크 축 = 마지막 축)."""
    return (x_ksp.detach().abs().amax(dim=(1, 2)) > 0).float()


def line_envelope(z, m, width=ENVELOPE_WIDTH):
    """z 하이브리드 공간 (B, n, H, W) complex, m (B, W) → e (B, 1, H, W) float32:
    코일 RMS |z| 를 W 축 폭 width 창에서 측정 줄만 평균, 하한 = ENVELOPE_FLOOR · 샘플별 평균."""
    a = torch.sqrt((z.real * z.real + z.imag * z.imag).mean(dim=1, keepdim=True))       # (B, 1, H, W)
    mw = m[:, None, None, :].to(a.dtype)                                                  # (B, 1, 1, W)
    pool = lambda t: F.avg_pool2d(t, kernel_size=(1, width), stride=1, padding=(0, width // 2),   # noqa: E731
                                  count_include_pad=True)
    num = pool(a * mw)                                                                    # (B, 1, H, W)
    den = pool(mw)                                                                        # (B, 1, 1, W)
    e = torch.where(den > 0, num / den.clamp(min=1e-12), torch.zeros_like(num))
    floor = ENVELOPE_FLOOR * e.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-30)
    return torch.maximum(e, floor)


def _mamba_dt_init(dt_proj, dt_rank, d_inner):
    """Mamba 기본 Δ 초기화 (ss2d.py SelectiveScan1D 와 같음): weight U(±dt_rank^−½), bias = softplus⁻¹(dt), dt ~ 로그균일[1e-3, 1e-1]."""
    std = dt_rank ** -0.5
    nn.init.uniform_(dt_proj.weight, -std, std)
    dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
    with torch.no_grad():
        dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))


def _no_wd(p):
    p._no_weight_decay = True
    return p


class DirectionalSSM(nn.Module):
    """한 방향의 선택적 SSM. u (B', L, D) float32 → y (B', L, D) float32.
    complex_ssm=True: A (D, N) 복소(S4D-Lin), B·C 입력 의존 복소 — x_proj 출력 = dt_rank + 4N (B re·im, C re·im).
    complex_ssm=False: A (D, N) 실수(Mamba 기본 −(n+1)), B·C 입력 의존 실수 — x_proj 출력 = dt_rank + 2N."""

    def __init__(self, d_inner, d_state, dt_rank, complex_ssm):
        super().__init__()
        self.d_inner, self.d_state, self.dt_rank, self.complex_ssm = d_inner, d_state, dt_rank, bool(complex_ssm)
        self.n_bc = (4 if self.complex_ssm else 2) * d_state
        self.x_proj = nn.Linear(d_inner, dt_rank + self.n_bc, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        _mamba_dt_init(self.dt_proj, dt_rank, d_inner)
        n = torch.arange(d_state, dtype=torch.float32)
        if self.complex_ssm:
            self.A_re_log = _no_wd(nn.Parameter(torch.full((d_inner, d_state), math.log(0.5))))     # Re A = −1/2
            self.A_im = _no_wd(nn.Parameter((math.pi * n).repeat(d_inner, 1)))                      # Im A = πn
        else:
            self.A_log = _no_wd(nn.Parameter(torch.log(n + 1.0).repeat(d_inner, 1)))                # A = −(n+1)
        self.D = _no_wd(nn.Parameter(torch.ones(d_inner)))

    def A(self):
        if self.complex_ssm:
            return torch.complex(-torch.exp(self.A_re_log.float()), self.A_im.float())
        return -torch.exp(self.A_log.float())

    def forward(self, u):
        Bp, L, D = u.shape
        N = self.d_state
        x_dbl = self.x_proj(u).float()                                     # autocast 시 fp16 → fp32 (ss2d.py 와 같음)
        dt, bc = x_dbl.split([self.dt_rank, self.n_bc], dim=-1)
        delta = F.softplus(self.dt_proj(dt).float()).float()                # (B', L, D)
        if self.complex_ssm:
            b_re, b_im, c_re, c_im = bc.split([N, N, N, N], dim=-1)         # 각 (B', L, N)
            Bm = torch.stack((b_re, b_im), dim=-1).permute(0, 2, 1, 3).reshape(Bp, N, 2 * L)   # 위치마다 (re, im) 교번
            Cm = torch.stack((c_re, c_im), dim=-1).permute(0, 2, 1, 3).reshape(Bp, N, 2 * L)
        else:
            b, c = bc.split([N, N], dim=-1)
            Bm, Cm = b.transpose(1, 2), c.transpose(1, 2)                   # (B', N, L)
        y = _scan(u.float().transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous(), self.A(),
                  Bm.contiguous(), Cm.contiguous(), self.D.float())         # (B', D, L)
        return y.transpose(1, 2)


class VandermondeIDFT(nn.Module):
    """학습형 위상 부호화 방향 변환 (마지막 축): out[..., n] = Σ_t T[n, t] k[..., t],
    T[n, t] = c_n · exp(−(δ_n + iφ_n)(N−1−t)) · b_t — 초기값 = 중심 IDFT (모듈 설명 참고)."""

    def __init__(self, n):
        super().__init__()
        if n % 4:
            raise ValueError(f'VandermondeIDFT: N={n} 은 4 의 배수여야 함 (N/2 짝수 — 중심 IDFT 식의 부호)')
        self.n = int(n)
        idx = torch.arange(n, dtype=torch.int64)
        p = (n - 1) - idx                                                   # 지수 N−1−t
        # θ0[n, t] = φ0_n·(N−1−t) mod 2π = 2π·((n·p) mod N)/N + π·(p mod 2)  (정수 산술)
        theta0 = (2 * math.pi / n) * ((idx[:, None] * p[None, :]) % n).double() + math.pi * (p % 2)[None, :].double()
        self.register_buffer('theta0', theta0.float(), persistent=False)
        self.register_buffer('pexp', p.float(), persistent=False)
        # c_n = N^{−1/2}(−1)^n e^{iφ0_n(N−1)}, φ0_n(N−1) mod 2π = 2π·((n(N−1)) mod N)/N + π·((N−1) mod 2)
        ang = (2 * math.pi / n) * ((idx * (n - 1)) % n).double() + math.pi * ((n - 1) % 2)
        sign = 1.0 - 2.0 * (idx % 2).double()
        c0 = sign * torch.polar(torch.ones_like(ang), ang) / math.sqrt(n)
        self.c = _no_wd(nn.Parameter(torch.stack((c0.real, c0.imag), dim=-1).float()))            # (N, 2)
        self.b = _no_wd(nn.Parameter(torch.stack((torch.ones(n), torch.zeros(n)), dim=-1)))       # (N, 2)
        self.eta = _no_wd(nn.Parameter(torch.zeros(n)))                                           # φ = φ0 + η/N
        self.rho = _no_wd(nn.Parameter(torch.full((n,), RHO_INIT)))                               # δ = softplus(ρ)/N

    def matrix(self):
        n = self.n
        delta = F.softplus(self.rho.float()) / n                                                  # (N,)
        phase = self.theta0 + (self.eta.float() / n)[:, None] * self.pexp[None, :]                # (N, N)
        mag = torch.exp(-delta[:, None] * self.pexp[None, :])                                     # (N, N)
        E = torch.polar(mag, -phase)
        c = torch.complex(self.c[:, 0].float(), self.c[:, 1].float())
        b = torch.complex(self.b[:, 0].float(), self.b[:, 1].float())
        return c[:, None] * E * b[None, :]                                                        # T[n, t]

    def forward(self, k):
        if k.shape[-1] != self.n:
            raise ValueError(f'VandermondeIDFT: 마지막 축 {k.shape[-1]} ≠ N={self.n}')
        return torch.matmul(k, self.matrix().transpose(0, 1))


class FSSM(nn.Module):
    """FSSM 시퀀스 모듈 (모델의 self.ss2d). forward(x_ksp (B, 2·n_coil, H, W)) → (B, 2·n_coil, H, W) 영상 좌표 real/imag 교번.
    하위 모듈 이름은 smoke_recipe_fix.build_groups 의 SS2D 부분 그룹과 맞춘다(in_proj·dwconv = ss2d_in, ssm_* = ss2d_ssm,
    out_proj = ss2d_out); transform(FSSM-T)은 seq 그룹에만 든다."""

    def __init__(self, *, n_coil=16, d_inner=64, d_state=16, complex_ssm=True, learn_transform=False, pe_len=384,
                 env_width=ENVELOPE_WIDTH):
        super().__init__()
        self.n_coil, self.d_inner, self.complex_ssm = int(n_coil), int(d_inner), bool(complex_ssm)
        self.env_width = int(env_width)
        c_ri = 2 * self.n_coil
        self.c_feat = c_ri + 1
        self.dt_rank = math.ceil(d_inner / 16)
        n_state = d_state if self.complex_ssm else 2 * d_state           # 실수판: 상태 수 2N (복소 N 과 같은 크기)
        self.in_proj = nn.Linear(self.c_feat, 2 * d_inner, bias=False)
        self.dwconv = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=True)
        self.ssm_f = DirectionalSSM(d_inner, n_state, self.dt_rank, self.complex_ssm)
        self.ssm_b = DirectionalSSM(d_inner, n_state, self.dt_rank, self.complex_ssm)
        self.out_proj = nn.Linear(d_inner, c_ri, bias=True)
        with torch.no_grad():
            self.out_proj.weight.zero_()
            self.out_proj.bias.zero_()
        self.transform = VandermondeIDFT(pe_len) if learn_transform else None

    def _ssm(self, feat):
        """feat (B', L, c_feat) → r̂ (B', L, 2·n_coil) — autocast 그대로 (projection), 스캔은 fp32."""
        u, gate = self.in_proj(feat).chunk(2, dim=-1)
        u = F.silu(self.dwconv(u.transpose(1, 2)).transpose(1, 2)).float()
        y = self.ssm_f(u) + self.ssm_b(u.flip(1)).flip(1)
        return self.out_proj(y * F.silu(gate.float()))

    def forward_parts(self, x_ksp):
        """진단용: dict(z 하이브리드, m 측정 줄, e 포락선, r 보정(하이브리드), y 출력(영상, real/imag 교번))."""
        dev = x_ksp.device.type
        with torch.autocast(device_type=dev, enabled=False):
            z = ifft1c(ri_to_complex(x_ksp), dim=-2)                         # (B, n, H, W) — 판독 방향 IFT
            m = measured_lines(x_ksp)                                        # (B, W)
            e = line_envelope(z, m, self.env_width)                          # (B, 1, H, W)
            B_, n, H, W = z.shape
            feat = torch.cat((complex_to_ri(z / e), m[:, None, None, :].expand(B_, 1, H, W)), dim=1)
            feat = feat.permute(0, 2, 3, 1).reshape(B_ * H, W, self.c_feat)  # 행 (b, h) 마다 W 축 수열
        rhat = self._ssm(feat)
        with torch.autocast(device_type=dev, enabled=False):
            rhat = rhat.float().reshape(B_, H, W, 2 * n).permute(0, 3, 1, 2)
            r = e * ri_to_complex(rhat)
            k = z + r
            y = self.transform(k) if self.transform is not None else ifft1c(k, dim=-1)
            out = complex_to_ri(y)
        return dict(z=z, m=m, e=e, r=r, y=out)

    def forward(self, x_ksp):
        return self.forward_parts(x_ksp)['y']


def _summary(name, model):
    seq = sum(p.numel() for p in model.ss2d.parameters())
    unet = sum(p.numel() for p in model.unet.parameters())
    print(f"   '{name}' (FSSM: complex={model.ss2d.complex_ssm}, learn_T={model.ss2d.transform is not None}, "
          f"d_inner={model.ss2d.d_inner}; seq {seq:,} + OriginalUNet(in={model.seq_out_ch}) {unet:,})")


class PureETER_FSSMOU(nn.Module):
    """FSSM (OU): OriginalUNet(FSSM(x_ksp)) — 복소 SSM 보정 + 고정 IFT. x_img 는 읽지 않는다(초기에는 FSSM(x_ksp) = x_img)."""

    def __init__(self, *, n_coil=16, unet_depth=5, unet_wf=6, d_inner=64, d_state=16, pe_len=384):
        super().__init__()
        self.ss2d = FSSM(n_coil=n_coil, d_inner=d_inner, d_state=d_state, complex_ssm=True, learn_transform=False,
                         pe_len=pe_len)
        self.seq_out_ch = 2 * n_coil
        self.unet = OriginalUNet(in_channels=self.seq_out_ch, n_classes=1, depth=unet_depth, wf=unet_wf)
        _summary('PureETER_FSSMOU', self)

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        return self.unet(self.ss2d(x_ksp))


class PureETER_FSSMROU(nn.Module):
    """FSSM-real (OU) (절제): A 실수(Mamba 기본), 상태 수 2N — 그 밖은 FSSM (OU) 와 같다."""

    def __init__(self, *, n_coil=16, unet_depth=5, unet_wf=6, d_inner=64, d_state=16, pe_len=384):
        super().__init__()
        self.ss2d = FSSM(n_coil=n_coil, d_inner=d_inner, d_state=d_state, complex_ssm=False, learn_transform=False,
                         pe_len=pe_len)
        self.seq_out_ch = 2 * n_coil
        self.unet = OriginalUNet(in_channels=self.seq_out_ch, n_classes=1, depth=unet_depth, wf=unet_wf)
        _summary('PureETER_FSSMROU', self)

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        return self.unet(self.ss2d(x_ksp))


class PureETER_FSSMTOU(nn.Module):
    """FSSM-T (OU) (절제): 복소 SSM 보정 + 학습형 Vandermonde 변환(초기값 = 중심 IDFT) — 그 밖은 FSSM (OU) 와 같다."""

    def __init__(self, *, n_coil=16, unet_depth=5, unet_wf=6, d_inner=64, d_state=16, pe_len=384):
        super().__init__()
        self.ss2d = FSSM(n_coil=n_coil, d_inner=d_inner, d_state=d_state, complex_ssm=True, learn_transform=True,
                         pe_len=pe_len)
        self.seq_out_ch = 2 * n_coil
        self.unet = OriginalUNet(in_channels=self.seq_out_ch, n_classes=1, depth=unet_depth, wf=unet_wf)
        _summary('PureETER_FSSMTOU', self)

    def forward(self, x_img, x_ksp, mask=None, sens=None):
        return self.unet(self.ss2d(x_ksp))
