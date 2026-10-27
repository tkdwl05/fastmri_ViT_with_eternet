"""
Mamba-ETER (2026-10-11) — 교수님 ETER-net(PureETER_GRU)에서 **순환 셀 nn.GRU 만** 양방향 선택적 SSM(mamba_ssm 2.2.2 의 Mamba
블록)으로 바꾼 비교 모델. 'SSM vs RNN' 을 가장 깨끗하게 비교하기 위한 모델이다 (사용자 결정 10-11).

배경 (results/ss2d_diag_2026-10-11/forensics.md·theory.md·verify.md):
  - 현행 SS2D 모듈(화소 토큰, 1 층, 행·열 병렬 스캔, 0.1M)은 k-space → 영상 변환을 표현할 수 없다(십자형 수용 영역, 이동 등변,
    위치별 가중치 없음). ETER-net 의 GRU 는 원리상 가능하다 — (1) gru_v 가 gru_h 출력을 받는 2단 순차 합성, (2) gru_h 의 한 step 이
    한 블록(한 채널의 연속 32 행 × 384 열 = 12,288 차원 벡터)을 dense W_ih 로 받는다(한 step 안의 위치마다 다른 가중치).
  - 그래서 SS2D 와의 비교는 '셀' 이 아니라 '데이터 흐름' 차이를 함께 잰다. Mamba-ETER 는 데이터 흐름을 ETER-net 과 글자 그대로
    같게 두고 셀만 바꾼다.

PureETER_GRU 와 같은 것 (상속으로 보장 — _seq·forward 를 재정의하지 않는다; sanity_mamba1d.py 가 함수 객체 동일성을 확인):
  - reshape/permute 데이터 흐름: x_ksp (B,32,384,384) → (B,384,12288): gru_h step t = 채널 ⌊t/12⌋, 행 32·(t mod 12)…+31, 열 384 전부.
    gru_h 출력 (B,384,7680) → reshape (B,384,384,20)·permute(0,2,1,3)·reshape (B,384,7680): gru_v step a = 모든 gru_h step 의
    unit [20a, 20a+19] (a<192 = 정방향 unit, a≥192 = 역방향 unit). gru_v 출력 → reshape·permute(0,3,2,1) → (B,20,384,384):
    출력 (c,y,x) = gru_v step x 의 unit 20y+c (y<192 = v 정방향, y≥192 = v 역방향).
  - 2단 순차 합성(h 출력이 v 입력), step 벡터 전체에 대한 dense 입력 투영(in_proj 12288→2·6192, 7680→2·6568 — GRU 의
    W_ih 12288→3·3840, 7680→3·3840 과 같은 역할: step 안의 행·열 위치마다 다른 가중치, step 간 공유), 양방향 출력 배치
    [정방향 H | 역방향 H] (H = 3840), 역방향 블록 출력을 다시 뒤집어 원래 step 색인에 맞추는 것(nn.GRU 와 같음).
  - cat(시퀀스 출력 20ch, zero-filled 영상 32ch) → UNet_choh_skip(in 52, depth 5, wf 6, n_hidden 26) — 같은 인자.
  - 슬롯 이름 gru_h / gru_v: 상속한 _seq 와 smoke_recipe_fix.build_groups('gru') 의 이름 접두사가 이 이름을 쓰므로 유지하는
    **코드 식별자**다. 내용물은 nn.GRU 가 아니라 BiMambaSeq 이다.

바뀌는 것 = 순환 셀뿐:
  GRU 셀(리셋·갱신 게이트 + tanh 후보, 방향당 상태 3840)
  → Mamba 블록: in_proj(바이어스 없음) → x·z 분할 → 깊이별 인과 conv1d(k=4) → SiLU → 입력 의존 Δ·B·C 의 선택적 SSM
    (실수 대각 A, 방향당 상태 d_inner×16) + D 스킵 → SiLU(z) 게이트 → out_proj(d_inner → 3840, 바이어스 없음).
  셀 고유의 차이(한계 항목): 상태 크기(3,840 vs ≈10만), 셀 안의 인과 conv1d(인접 step = 인접 행 블록을 섞고 12 step 마다 채널
  경계를 넘음), SiLU 게이트, 바이어스 유무(Mamba-ETER 는 f(0)=0 이 정확히 성립, GRU 는 f(0)≠0).

차원 (파라미터 정합 — 단계·방향별로 GRU 에 맞춘 정수해, 8 의 배수):
  h 단 d_inner 6192 (입력 12,288 → 3,840), v 단 d_inner 6568 (입력 7,680 → 3,840), d_state 16, d_conv 4, dt_rank 'auto'
  (= ⌈d_model/16⌉ = 768 / 480). 방향당 파라미터 = E·(2D + H + d_conv + 3 + 2R + 3N).
  시퀀스 모듈 637,150,928 vs GRU 637,102,080 (+0.0077%), 모델 전체 668,210,949 vs 668,162,101, U-Net 31,060,021 동일.
  정규화 층·잔차 없음(GRU 에도 없다; step 단위 RMSNorm 은 행 블록 사이의 에너지 비를 지워 SS2D norm_in 과 같은 계열의 위험).
  초기화 = mamba_ssm 기본값(GRU 도 PyTorch 기본값). MixerModel 의 _init_weights 는 Mamba 단독 사용 시 적용되지 않는다.

MambaOut: Mamba 를 그대로 만든 뒤 out_proj 만 Linear(d_inner, H, bias=False) 로 교체. fast path(mamba_inner_fn: F.linear(out_z,
  out_proj_weight))와 slow path(self.out_proj(y)) 모두 출력 차원을 out_proj.weight 모양에서만 정하므로 교체가 안전하다.
  교체한 out_proj 의 기본 초기화 분포 U(±1/√d_inner) 는 원래 out_proj 와 같다(fan_in 같음). 원 out_proj 생성이 소비하는 RNG·일시
  메모리는 결정적이므로 그대로 둔다. expand = (d_inner + 0.5)/d_model — int() 절삭의 부동소수 오차로 d_inner 가 1 작아지는 것을 막음.

정밀도: fp32_stages ⊆ {'h','v'} 에 든 단계는 autocast 가 켜져 있어도 그 단계 전체(in_proj matmul·conv·scan·out_proj)를 fp32 로
  돈다(입력 float() 승격, 출력 fp32). 스모크 기본은 () — 하네스 그대로 fp16 autocast. pod3 의 GRU 붕괴(12,288 차원 입력의 W_ih
  기울기 fp16 넘침)와 같은 모양의 위험이 in_proj 에 있으므로 50 epoch 에서는 GRU(pod3f, gru_h fp32)와 같게 ('h',) 로 맞춘다.

CPU: mamba_ssm 의 fast path 는 CUDA 전용이다. CPU 에서는 v8_eter_pure/smoke_mamba1d.py 의 install_mamba_cpu_shims() 가
  mamba_simple.causal_conv1d_fn = None(slow path: nn.Conv1d + SiLU)·mamba_simple.selective_scan_fn = 참조 점화식으로 바꾼다
  (visualize_multimodel_compare.py 의 SS2D CPU 교체와 같은 런타임 이름 교체). 이 파일은 아무것도 바꾸지 않는다.

교수님/프로젝트 원본 파일(u_pure_eternet_gru.py·myUNet_DF.py·mamba_ssm)은 일절 수정하지 않는다 — 새 파일·상속만.
"""

import os
import sys
import math

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)
sys.path.append(os.path.join(_HERE, '..', 'hybrid_eternet'))

from u_pure_eternet_gru import PureETER_GRU          # 데이터 흐름(_seq·forward) 단일 출처 — 무수정 상속
from myUNet_DF import UNet_choh_skip                 # 교수님 DFU (무수정 재사용)
from mamba_ssm.modules.mamba_simple import Mamba     # mamba_ssm 2.2.2 표준 블록 (무수정)

# 기본 차원 (파라미터 정합 — 모듈 docstring)
D_INNER_H = 6192
D_INNER_V = 6568
D_STATE = 16
D_CONV = 4


def fp32_stages_from_str(s):
    """'none'|''|'h'|'v'|'hv' → fp32_stages 튜플 (러너 --fp32 와 이후 트레이너 env MAMBA_ETER_FP32 공통 해석)."""
    s = (s or 'none').strip().lower()
    if s in ('none', '0', ''):
        return ()
    if not set(s) <= {'h', 'v'}:
        raise ValueError(f'fp32 단계 지정은 none|h|v|hv: {s!r}')
    return tuple(sorted(set(s)))


def mamba_params_per_direction(d_model, d_inner, out_dim, d_state=D_STATE, d_conv=D_CONV, dt_rank='auto'):
    """MambaOut 1 개(한 방향)의 파라미터 수 해석식 = E·(2D + H + d_conv + 3 + 2R + 3N).
    in_proj 2ED + conv1d E·(K+1) + x_proj E·(R+2N) + dt_proj E·R + E + A_log E·N + D E + out_proj E·H."""
    R = math.ceil(d_model / 16) if dt_rank == 'auto' else int(dt_rank)
    E, D, H, N, K = d_inner, d_model, out_dim, d_state, d_conv
    return E * (2 * D + H + K + 3 + 2 * R + 3 * N)


def gru_params_per_direction(d_in, hidden):
    """nn.GRU 1 층 한 방향의 파라미터 수 = 3H·(D+H) + 6H."""
    return 3 * hidden * (d_in + hidden) + 6 * hidden


class MambaOut(Mamba):
    """mamba_ssm Mamba 블록, out_proj 만 Linear(d_inner, out_dim, bias=False) 로 교체 (출력 차원 ≠ d_model).
    나머지(in_proj·conv1d·x_proj·dt_proj·A_log·D, 초기화·fast/slow path)는 Mamba 그대로."""

    def __init__(self, d_model, d_inner, out_dim, d_state=D_STATE, d_conv=D_CONV, dt_rank='auto'):
        super().__init__(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=(d_inner + 0.5) / d_model,
                         dt_rank=dt_rank)
        assert self.d_inner == d_inner, f'd_inner {self.d_inner} != 요청 {d_inner} (expand 절삭)'
        self.out_dim = out_dim
        self.out_proj = nn.Linear(d_inner, out_dim, bias=False)


class BiMambaSeq(nn.Module):
    """nn.GRU(input_size, hidden, 1, batch_first=True, bidirectional=True) 의 자리 대체 모듈.

    forward(x: (B, L, D_in)) → (cat([fwd(x), bwd(x.flip(1)).flip(1)], -1), None) — (B, L, 2H).
    출력 배치 [정방향 H | 역방향 H] 와 역방향 출력의 step 색인(다시 뒤집어 원래 시간 순서)은 nn.GRU 양방향 출력과 같다.
    두 번째 반환값(hidden)은 nn.GRU 와 시그니처를 맞추기 위한 None — PureETER_GRU._seq 는 이것을 버린다(`out_h, _ = ...`).
    fwd·bwd 는 서로 독립 파라미터(nn.GRU 의 *_l0 / *_l0_reverse 와 같은 관계).
    force_fp32: True 이고 autocast 가 켜져 있으면 이 단계 전체를 fp32 로(입력 float() 승격, 출력 fp32)."""

    def __init__(self, input_size, hidden_size, d_inner, d_state=D_STATE, d_conv=D_CONV, dt_rank='auto',
                 force_fp32=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.force_fp32 = bool(force_fp32)
        self.fwd = MambaOut(input_size, d_inner, hidden_size, d_state, d_conv, dt_rank)
        self.bwd = MambaOut(input_size, d_inner, hidden_size, d_state, d_conv, dt_rank)

    def _run(self, x):
        yf = self.fwd(x)
        yb = self.bwd(x.flip(1)).flip(1)
        return torch.cat((yf, yb), dim=-1)

    def forward(self, x):
        if self.force_fp32 and torch.is_autocast_enabled():
            with torch.autocast('cuda', enabled=False):
                return self._run(x.float()), None
        return self._run(x), None


class PureETER_Mamba1D(PureETER_GRU):
    """Mamba-ETER — PureETER_GRU 의 데이터 흐름(_seq·forward)을 코드 그대로 상속하고 gru_h/gru_v 의 내용물만 BiMambaSeq 로 바꾼 모델.

    PureETER_GRU.__init__ 은 부르지 않는다(nn.Module.__init__ 직접 호출) — 637M nn.GRU 를 만들었다 버리지 않기 위해
    (2.5 GB 낭비·RNG 소비 변화 없음). 대신 PureETER_GRU 와 같은 속성(dim, n_coil, n_hidden_1/2, num_layers, use_dc,
    num_in_x/num_in_y/num_out_x/num_out_y, num_out1, num_out2)을 같은 식으로 설정하고, 같은 순서(gru_h → gru_v → unet)로 만든다.
    no-DC 전용(use_dc=True 는 NotImplementedError)."""

    seq_kind = 'mamba1d'

    def __init__(
        self,
        *,
        dim: int = 384,
        n_coil: int = 16,
        n_hidden_1: int = 10,
        n_hidden_2: int = 10,
        unet_depth: int = 5,
        unet_wf: int = 6,
        use_dc: bool = False,
        d_inner_h: int = D_INNER_H,
        d_inner_v: int = D_INNER_V,
        d_state: int = D_STATE,
        d_conv: int = D_CONV,
        dt_rank='auto',
        fp32_stages=(),
        verbose: bool = True,
    ):
        if use_dc:
            raise NotImplementedError('Mamba-ETER 는 no-DC 전용 (DC 요인 폐기 — CLAUDE.md)')
        fp32_stages = tuple(sorted(set(fp32_stages)))
        assert set(fp32_stages) <= {'h', 'v'}, f'fp32_stages 는 h/v 의 부분집합: {fp32_stages}'
        nn.Module.__init__(self)
        # ---- PureETER_GRU.__init__ 과 같은 속성·같은 식 (u_pure_eternet_gru.py)
        self.dim = dim
        self.n_coil = n_coil
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.num_layers = 1
        self.use_dc = use_dc
        self.num_in_x = dim
        self.num_in_y = dim
        self.num_out_x = dim
        self.num_out_y = dim
        input_size = self.num_in_y * n_coil * 2
        self.num_out1 = self.num_out_y * n_hidden_1
        num_in2 = self.num_in_x * n_hidden_1
        self.num_out2 = self.num_out_x * n_hidden_2

        # ---- 시퀀스 모듈 = 양방향 Mamba ×2 (슬롯 이름 gru_h/gru_v 는 코드 식별자 — 상속한 _seq·build_groups('gru') 가 쓴다)
        self.gru_h = BiMambaSeq(input_size, self.num_out1, d_inner_h, d_state, d_conv, dt_rank,
                                force_fp32='h' in fp32_stages)
        self.gru_v = BiMambaSeq(num_in2 * 2, self.num_out2, d_inner_v, d_state, d_conv, dt_rank,
                                force_fp32='v' in fp32_stages)

        # ---- 최종 합성 = 교수님 DFU (PureETER_GRU 와 같은 인자)
        num_feat_ch = 2 * n_hidden_2 + n_coil * 2
        n_hidden = n_hidden_2 + n_coil
        assert n_hidden * 2 == num_feat_ch, \
            f"UNet_choh_skip 계약 위반: n_hidden*2({n_hidden*2}) != in_channels({num_feat_ch})"
        self.unet = UNet_choh_skip(
            in_channels=num_feat_ch, n_classes=1,
            depth=unet_depth, wf=unet_wf, padding=True,
            batch_norm=False, up_mode='upconv', n_hidden=n_hidden,
        )

        self.fp32_stages = fp32_stages
        self.mamba_cfg = dict(d_inner_h=d_inner_h, d_inner_v=d_inner_v, d_state=d_state, d_conv=d_conv,
                              dt_rank_h=self.gru_h.fwd.dt_rank, dt_rank_v=self.gru_v.fwd.dt_rank,
                              fp32_stages=list(fp32_stages))
        if verbose:
            n_seq = sum(p.numel() for p in self.gru_h.parameters()) + sum(p.numel() for p in self.gru_v.parameters())
            n_all = sum(p.numel() for p in self.parameters())
            print(f"   'PureETER_Mamba1D' (Mamba-ETER; H1={n_hidden_1}, H2={n_hidden_2}, d_inner h/v={d_inner_h}/{d_inner_v}, "
                  f"d_state={d_state}, dt_rank h/v={self.gru_h.fwd.dt_rank}/{self.gru_v.fwd.dt_rank}, "
                  f"seq={n_seq / 1e6:.2f}M total={n_all / 1e6:.2f}M, fp32_stages={list(fp32_stages)}, "
                  f"in_ch={num_feat_ch}, n_hidden={n_hidden}, unet_depth={unet_depth}, use_dc={use_dc})")

    # _seq, forward: PureETER_GRU 의 것을 그대로 상속 (재정의 금지 — sanity_mamba1d.py 가 확인)
