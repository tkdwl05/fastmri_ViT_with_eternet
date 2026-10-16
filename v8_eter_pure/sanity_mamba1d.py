"""
Mamba-ETER(PureETER_Mamba1D) CPU 점검 (2026-10-11) — GPU 를 쓰지 않는다 (CUDA_VISIBLE_DEVICES="" 가 아니면 거부).

목적: 학습 전에 "ETER-net 과 순환 셀만 다르다" 는 계약과 하네스 호환을 확인한다. GPU 전용 검사(fast path CUDA 커널 vs 참조 경로)는
smoke_mamba1d.py 의 preflight 로 미룬다. CPU forward 는 mamba_ssm 의 slow path(nn.Conv1d+SiLU) + selective_scan_cpu_full
(smoke_mamba1d.py — selective_scan_ref 와 같은 점화식, 아래 [scan] 에서 출력·기울기 일치 확인)로 돈다.

검사 (괄호 = 작업 지시의 항목):
  [scan]     selective_scan_cpu_full 이 selective_scan_ref 와 출력·기울기 일치 (z·delta_bias·softplus·D 포함).
  [params]   (d) 실제 크기 모델을 CPU 에 만들어 단계·방향·하위 텐서별 파라미터 수, 해석식 일치, GRU 대비 차이(|Δ| < 0.01%).
  [identity] _seq·forward 가 PureETER_GRU 의 함수 객체와 같음, 비모듈 속성(dim … num_out2)이 PureETER_GRU 인스턴스와 같은 값.
  [unet]     U-Net state_dict 키·모양이 PureETER_GRU·PureETER_UNET 과 같음, in_channels 52·depth 5·wf 6.
  [unetinit] (e) 러너 builder(--unet-init gru) 의 U-Net 이 seed_all(1) 직후 PureETER_GRU 의 U-Net 과 비트 단위로 같고(= 스모크
             gru:DL), --unet-init unet 은 PureETER_UNET 과 같음. 하네스 gru/ss2d/unet 모델의 U-Net checksum(v8fix 식) 비교 —
             하네스에서는 모델마다 U-Net 초기값이 다르다(알려진 사실). 시퀀스 모듈 초기값은 같은 시드에서 동일, builder 뒤 전역 RNG 는
             Mamba-ETER 만 만든 경우와 같음.
  [bidir]    (b) 작은 BiMambaSeq(D=24, H=6, E=12, L=10): 정방향 절반은 과거(≤t)에만, 역방향 절반은 미래(≥t)에만 의존 — autograd 의존
             마스크가 nn.GRU(bidirectional) 와 같음. 역방향 출력 = 역방향 블록을 뒤집은 시퀀스에 직접 돌려 다시 뒤집은 것.
  [dep]      (c) 작은 Mamba-ETER(dim 16, 코일 2, H 2)에서 시퀀스 출력의 모든 화소가 모든 입력(k-space 원소)에 의존 (전체 Jacobian,
             float64) — 같은 크기 ETER-net(GRU)과 함께.
  [routing]  (a) gru_h/gru_v 를 같은 결정적 stub 으로 바꾼 PureETER_GRU·PureETER_Mamba1D 의 시퀀스 출력·wrapper 출력이 비트 단위로
             같음 (작은 크기) + 실제 크기에서 색인 태그 stub 으로 색인 규칙 추적: h step t 입력 = x_ksp[c=⌊t/12⌋, 32(t mod 12):+32, :]
             평탄화, v step a 의 (t,k) 위치 = h 출력[t, 20a+k], 최종 (c,y,x) = v step x 의 unit 20y+c — 두 wrapper 모두.
  [harness]  smoke_recipe_fix 와의 호환: build_groups(model,'gru') 누락 없음, split_decay 의 no-decay = A_log·D(_no_weight_decay) +
             conv1d.bias·dt_proj.bias(bias), decay = in_proj·conv1d.weight·x_proj·dt_proj.weight·out_proj; seq_forward == _seq.
  [forward]  (f) 실제 크기 모델(seed 1)의 CPU forward: 정규화된 val probe 2 슬라이스(smoke 와 같은 vi[0], vi[32])에서 유한, 단계별 RMS,
             시퀀스 출력 RMS / zero-filled RMS 비 (같은 시드 ETER-net 과 비교), f(0)=0, 슬라이스 간 출력 차이.
  [train]    (g) 작은 모델 3 step CPU 학습(R.build_optimizer 'DL' = AdamW 분리형 + no-decay, clip 1e3, masked L1 + λ·(1−SSIM),
             R.normalize_inputs): 손실·기울기 유한, 모든 Mamba 파라미터에 0 이 아닌 기울기, 두 param group 모두 갱신.
  [groups]   (2026-10-11b) run_job 동안 R.build_groups 자리에 쓰는 build_groups_mamba: 원 그룹 키·목록 그대로 + 하위 텐서 그룹
             seqsub.<stage>.<sub> 이 시퀀스 파라미터를 정확히 한 번씩 덮음; ETER-net 모델에는 아무것도 더하지 않음.
  [alarm]    (2026-10-11b) fallback 규칙(precision_alarm: 넘침·preflight 언더플로·verdict.seq frozen/eps-limited), fallback 단계 = hv,
             done_ok(시간 상한 중단·U-Net 초기값 불일치·num_workers 불일치 거부), 해석 규칙 interpret() — 합성 입력 단위 시험.
  [unetpre]  (2026-10-11b) 학습 전 U-Net 초기값 대조: builder(--unet-init gru) 는 gru_DL.jsonl step 0·seed 1 checksum 과 일치,
             자체 생성(own) U-Net 은 불일치로 잡힘.
  [gradscale] (2026-10-11b) grad_preflight 의 CPU fp32 부분을 실제 크기로: default·outmatch(러너의 calibrate_outmatch 로 CPU 보정)
             Mamba-ETER 와 같은 시드 ETER-net 의 시퀀스 하위 텐서별 기울기 RMS·Adam eps 미만 비율. 통과 = 전부 유한 + outmatch 의
             in_proj 기울기 RMS 가 ETER-net W_ih 의 0.1~10 배(초기 이득 보정이 기울기를 eps 위로 올리는지 — 검토 지적 확인).
             default 의 eps 미만 비율·Δ 경로 기울기는 기록(발견 사항).
  [orig]     원본 무수정: git status --porcelain 이 보호 경로(교수님 7d4e4e0 원본 포함)에서 비어 있음 + 미추적 진단 스크립트 sha256 기록.

출력: stdout + --json-out (기본 results/mamba1d_smoke_2026-10-11/sanity_cpu.json). 실패가 하나라도 있으면 종료 코드 1.

실행:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 19 python v8_eter_pure/sanity_mamba1d.py
  # 실제 크기 검사를 빼고 빠르게:  ... --skip-real
"""

import os
import sys

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    print('[refuse] sanity_mamba1d.py 는 CPU 전용 — CUDA_VISIBLE_DEVICES="" 로 실행한다 (GPU0 은 다른 런이 사용 중일 수 있음)')
    sys.exit(2)

import json            # noqa: E402
import time            # noqa: E402
import argparse        # noqa: E402
import hashlib         # noqa: E402
import subprocess      # noqa: E402
import traceback       # noqa: E402
from collections import OrderedDict   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import smoke_mamba1d as SM              # noqa: E402 — R·D·mamba 경로 import, CPU shim 단일 출처 (import 부작용 없음)
import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import torch.nn as nn                   # noqa: E402
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref   # noqa: E402
from u_pure_eternet_mamba1d import (PureETER_Mamba1D, BiMambaSeq, mamba_params_per_direction,   # noqa: E402
                                    gru_params_per_direction)
from u_pure_eternet_gru import PureETER_GRU     # noqa: E402
from u_pure_eternet_unet import PureETER_UNET   # noqa: E402

R, D, C = SM.R, SM.D, SM.C
RESULTS = OrderedDict()


def record(name, passed, detail, data=None):
    RESULTS[name] = OrderedDict(passed=bool(passed), detail=detail, data=data)
    print(f'[{"PASS" if passed else "FAIL"}] {name}: {detail}', flush=True)


def run_test(name, fn, *a, **kw):
    t0 = time.time()
    try:
        fn(*a, **kw)
    except Exception as ex:                                             # noqa: BLE001
        record(name, False, f'예외 {type(ex).__name__}: {ex}', dict(traceback=traceback.format_exc()[-3000:]))
    print(f'    ({name}: {time.time() - t0:.1f}s)', flush=True)


def numel(mod):
    return sum(p.numel() for p in mod.parameters())


def small_kwargs(**over):
    kw = dict(dim=16, n_coil=2, n_hidden_1=2, n_hidden_2=2, unet_depth=2, unet_wf=2)
    kw.update(over)
    return kw


# ------------------------------------------------------------------ [scan]
def t_scan():
    g = torch.Generator().manual_seed(0)
    b, d, n, L = 2, 6, 4, 9
    mk = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).requires_grad_(True)   # noqa: E731
    u, delta, z = mk(b, d, L), mk(b, d, L, scale=0.5), mk(b, d, L)
    A = (-torch.rand(d, n, generator=g) * 3 - 0.1).requires_grad_(True)
    B, Cm, Dv, db = mk(b, n, L), mk(b, n, L), mk(d), mk(d, scale=0.3)
    ins = (u, delta, A, B, Cm, Dv, z, db)
    outs = []
    for fn in (selective_scan_ref, SM.selective_scan_cpu_full):
        y = fn(u, delta, A, B, Cm, Dv, z=z, delta_bias=db, delta_softplus=True)
        w = torch.linspace(-1, 1, y.numel()).reshape(y.shape)
        grads = torch.autograd.grad((y * w).sum(), ins)
        outs.append((y.detach(), grads))
    rel = lambda a, b_: float((a - b_).norm() / b_.norm().clamp(min=1e-30))   # noqa: E731
    e_out = rel(outs[1][0], outs[0][0])
    e_grad = max(rel(a, b_) for a, b_ in zip(outs[1][1], outs[0][1]))
    record('scan', e_out < 1e-5 and e_grad < 1e-5,
           f'selective_scan_cpu_full vs selective_scan_ref: 출력 상대오차 {e_out:.2e}, 기울기(8 입력) 최대 {e_grad:.2e}',
           dict(rel_err_out=e_out, rel_err_grad_max=e_grad))


# ------------------------------------------------------------------ [bidir]
def dep_mask(mod, L, Din, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, L, Din, generator=g, dtype=dtype).requires_grad_(True)
    y, _ = mod(x)
    H = y.shape[-1] // 2
    w = torch.randn(y.shape, generator=g, dtype=dtype)
    M = torch.zeros(2, L, L, dtype=torch.bool)
    for s in range(L):
        for h in range(2):
            gx, = torch.autograd.grad((y[0, s, h * H:(h + 1) * H] * w[0, s, h * H:(h + 1) * H]).sum(), x, retain_graph=True)
            M[h, s] = gx[0].abs().sum(-1) > 0
    return M


def t_bidir():
    torch.manual_seed(0)
    L, Din, H, E = 10, 24, 6, 12
    bm = BiMambaSeq(Din, H, E).double()
    gr = nn.GRU(Din, H, 1, batch_first=True, bidirectional=True).double()
    Mb, Mg = dep_mask(bm, L, Din), dep_mask(gr, L, Din)
    tri_lo = torch.tril(torch.ones(L, L, dtype=torch.bool))            # [s, t]: t ≤ s
    exp_fwd, exp_bwd = tri_lo, tri_lo.t()
    ok_mask = torch.equal(Mb, Mg) and torch.equal(Mb[0], exp_fwd) and torch.equal(Mb[1], exp_bwd)
    # 역방향 출력 정의
    with torch.no_grad():
        x = torch.randn(2, L, Din, dtype=torch.float64)
        y, _ = bm(x)
        yb_direct = bm.bwd(x.flip(1)).flip(1)
        yf_direct = bm.fwd(x)
        ok_def = torch.equal(y[..., H:], yb_direct) and torch.equal(y[..., :H], yf_direct)
        f0 = bm(torch.zeros(1, L, Din, dtype=torch.float64))[0].abs().max().item()
    record('bidir', ok_mask and ok_def,
           f'의존 마스크 Mamba == nn.GRU(bidirectional): {torch.equal(Mb, Mg)}, 정방향 = t≤s: {torch.equal(Mb[0], exp_fwd)}, '
           f'역방향 = t≥s: {torch.equal(Mb[1], exp_bwd)}; 출력 배치 [fwd | flip(bwd(flip x))]: {ok_def}; f(0) max {f0:.1e}',
           dict(mask_fwd_step4=Mb[0, 4].int().tolist(), mask_bwd_step4=Mb[1, 4].int().tolist(), f0_maxabs=f0))


# ------------------------------------------------------------------ [dep]
def full_jacobian_stats(model, x):
    J = torch.autograd.functional.jacobian(lambda inp: model._seq(inp), x)      # (1,Co,H,W, 1,Ci,H,W)
    n_out = int(np.prod(J.shape[:4]))
    J = J.reshape(n_out, -1).abs()
    nz = J > 0
    per_out_all = bool(nz.all(dim=1).all())
    return dict(n_out=n_out, n_in=J.shape[1], frac_nonzero=float(nz.float().mean()), every_output_depends_on_all_inputs=per_out_all,
                min_abs=float(J.min()), max_abs=float(J.max()),
                min_over_max=float(J.min() / J.max().clamp(min=1e-300)))


def t_dep():
    out = OrderedDict()
    for name in ('mamba', 'gru'):
        torch.manual_seed(1)
        if name == 'mamba':
            m = PureETER_Mamba1D(**small_kwargs(), d_inner_h=16, d_inner_v=16, verbose=False)
        else:
            m = PureETER_GRU(**small_kwargs())
        m = m.double().eval()
        g = torch.Generator().manual_seed(3)
        x = torch.randn(1, 4, 16, 16, generator=g, dtype=torch.float64)
        out[name] = full_jacobian_stats(m, x)
    ok = out['mamba']['every_output_depends_on_all_inputs']
    record('dep', ok,
           f"작은 Mamba-ETER(dim16·코일2·H2): 출력 {out['mamba']['n_out']} 화소 × 입력 {out['mamba']['n_in']} 원소 Jacobian 비영 비율 "
           f"{out['mamba']['frac_nonzero']:.4f} (min|J|/max|J| {out['mamba']['min_over_max']:.1e}); "
           f"같은 크기 ETER-net(GRU) {out['gru']['frac_nonzero']:.4f} (min/max {out['gru']['min_over_max']:.1e})", out)


# ------------------------------------------------------------------ [routing] 작은 크기
class LinStub(nn.Module):
    """결정적 stub: y = x·W, 정방향 절반은 시간 누적합, 역방향 절반은 역누적합 — nn.GRU 와 같은 (out, None) 시그니처."""

    def __init__(self, din, dout2, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer('W', torch.randn(din, dout2, generator=g, dtype=torch.float64) / din ** 0.5)

    def forward(self, x):
        y = x.to(self.W.dtype) @ self.W
        H = y.shape[-1] // 2
        yf = torch.cumsum(y[..., :H], dim=1)
        yb = torch.flip(torch.cumsum(torch.flip(y[..., H:], (1,)), dim=1), (1,))
        return torch.cat((yf, yb), dim=-1), None


def t_routing_small():
    kw = small_kwargs()
    torch.manual_seed(1)
    mg = PureETER_GRU(**kw).double()
    torch.manual_seed(2)
    mm = PureETER_Mamba1D(**kw, d_inner_h=8, d_inner_v=8, verbose=False).double()
    mm.unet.load_state_dict(mg.unet.state_dict())
    Dh, Dv = 16 * 2 * 2, 2 * 16 * 2
    sh, sv = LinStub(Dh, 2 * 16 * 2, 11), LinStub(Dv, 2 * 16 * 2, 12)
    for m in (mg, mm):
        m.gru_h, m.gru_v = sh, sv
    g = torch.Generator().manual_seed(5)
    xk = torch.randn(2, 4, 16, 16, generator=g, dtype=torch.float64)
    xi = torch.randn(2, 4, 16, 16, generator=g, dtype=torch.float64)
    with torch.no_grad():
        sg, sm = mg._seq(xk), mm._seq(xk)
        og, om = mg(xi, xk), mm(xi, xk)
    ok = torch.equal(sg, sm) and torch.equal(og, om)
    record('routing_small', ok, f'같은 stub 을 끼운 두 wrapper: 시퀀스 출력 비트 동일 {torch.equal(sg, sm)}, '
                                f'wrapper 출력(같은 U-Net) 비트 동일 {torch.equal(og, om)} (출력 {tuple(og.shape)})')


# ------------------------------------------------------------------ [routing] 실제 크기, 색인 태그
class TagStub(nn.Module):
    """입력을 기록하고 출력 (B, L, out) 을 step·unit 태그 (t·1e5 + u, float64 정확) 로 채운다."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.last_in = None

    def forward(self, x):
        self.last_in = x.detach().clone()
        B, L, _ = x.shape
        tag = torch.arange(L, dtype=torch.float64)[:, None] * 1e5 + torch.arange(self.out_dim, dtype=torch.float64)[None]
        return tag[None].expand(B, L, self.out_dim).clone(), None


def trace_routing(model):
    dim, H1, H2, nc = model.dim, model.n_hidden_1, model.n_hidden_2, model.n_coil
    model.gru_h = TagStub(2 * dim * H1)
    model.gru_v = TagStub(2 * dim * H2)
    xk = torch.arange(2 * nc * dim * dim, dtype=torch.float64).reshape(1, 2 * nc, dim, dim)
    with torch.no_grad():
        out = model._seq(xk)                                            # (1, 2·H2, dim, dim)
    in_h, in_v = model.gru_h.last_in[0], model.gru_v.last_in[0]
    # h step t 입력 = x_ksp[c=⌊t/12⌋, 32(t mod 12):32(t mod 12)+32, :] 평탄화 (dim=384, 32 채널 → step 당 12288 = 32 행 × 384 열)
    steps_per_ch = dim // (2 * nc)                                      # 384/32 = 12
    rows = dim // steps_per_ch                                          # 32
    ok_h = all(torch.equal(in_h[t], xk[0, t // steps_per_ch, rows * (t % steps_per_ch):rows * (t % steps_per_ch) + rows, :].reshape(-1))
               for t in range(dim))
    # v step a 의 (t, k) 위치 = h 출력[t, G·a + k], G = 2·H1 = 20 unit / v-step (k < G)
    G = (2 * dim * H1) // dim                                           # 20 unit / v-step
    a = torch.arange(dim, dtype=torch.float64)[:, None, None]
    t = torch.arange(dim, dtype=torch.float64)[None, :, None]
    k = torch.arange(G, dtype=torch.float64)[None, None, :]
    exp_v = t * 1e5 + G * a + k                                         # h 태그 = t·1e5 + unit, unit = 20a + k
    ok_v = torch.equal(in_v.reshape(dim, dim, G), exp_v)
    # 최종 (c, y, x) = v step x 의 unit 20y + c
    Gv = (2 * dim * H2) // dim
    c = torch.arange(Gv, dtype=torch.float64)[:, None, None]
    y = torch.arange(dim, dtype=torch.float64)[None, :, None]
    x = torch.arange(dim, dtype=torch.float64)[None, None, :]
    exp_out = x * 1e5 + Gv * y + c
    ok_o = torch.equal(out[0], exp_out)
    return out, dict(h_input_rule=ok_h, v_input_rule=ok_v, output_rule=ok_o, steps_per_channel=steps_per_ch, rows_per_step=rows,
                     units_per_v_step=G)


def t_routing_real(m_mamba, m_gru):
    om, rm = trace_routing(m_mamba)
    og, rg = trace_routing(m_gru)
    same = torch.equal(om, og)
    ok = all(rm[k] for k in ('h_input_rule', 'v_input_rule', 'output_rule')) and \
        all(rg[k] for k in ('h_input_rule', 'v_input_rule', 'output_rule')) and same
    record('routing_real', ok,
           f'실제 크기 색인 추적 — Mamba-ETER {rm}, ETER-net {rg}; 두 wrapper 의 태그 출력 비트 동일 {same}',
           dict(mamba=rm, gru=rg, identical=same))


# ------------------------------------------------------------------ [params]·[identity]·[unet]·[unetinit]·[harness]
def stage_table(model, kind):
    tab = OrderedDict()
    for st in ('gru_h', 'gru_v'):
        mod = getattr(model, st)
        if kind == 'mamba':
            for d in ('fwd', 'bwd'):
                blk = getattr(mod, d)
                sub = OrderedDict((n, p.numel()) for n, p in blk.named_parameters())
                tab[f'{st}.{d}'] = dict(total=numel(blk), parts=sub)
        else:
            for d, suf in (('fwd', '_l0'), ('bwd', '_l0_reverse')):
                parts = OrderedDict((n, p.numel()) for n, p in mod.named_parameters() if n.endswith(suf))
                tab[f'{st}.{d}'] = dict(total=sum(parts.values()), parts=parts)
    return tab


def t_params(mm, mg):
    tm, tg = stage_table(mm, 'mamba'), stage_table(mg, 'gru')
    rows = OrderedDict()
    ok = True
    for k in tm:
        st = 'h' if k.startswith('gru_h') else 'v'
        Din = mm.gru_h.input_size if st == 'h' else mm.gru_v.input_size
        E = mm.mamba_cfg[f'd_inner_{st}']
        ana_m = mamba_params_per_direction(Din, E, 3840)
        ana_g = gru_params_per_direction(Din, 3840)
        ok &= (ana_m == tm[k]['total']) and (ana_g == tg[k]['total'])
        rows[k] = dict(gru=tg[k]['total'], mamba=tm[k]['total'], diff=tm[k]['total'] - tg[k]['total'],
                       rel=(tm[k]['total'] - tg[k]['total']) / tg[k]['total'], mamba_parts=tm[k]['parts'],
                       gru_parts=tg[k]['parts'], analytic_match=(ana_m == tm[k]['total'], ana_g == tg[k]['total']))
    seq_m = numel(mm.gru_h) + numel(mm.gru_v)
    seq_g = numel(mg.gru_h) + numel(mg.gru_v)
    un_m, un_g = numel(mm.unet), numel(mg.unet)
    rel = (seq_m - seq_g) / seq_g
    ok &= abs(rel) < 1e-4 and un_m == un_g
    data = dict(per_stage_direction=rows, seq=dict(gru=seq_g, mamba=seq_m, diff=seq_m - seq_g, rel=rel),
                unet=dict(gru=un_g, mamba=un_m), total=dict(gru=numel(mg), mamba=numel(mm),
                                                            rel=(numel(mm) - numel(mg)) / numel(mg)),
                state_size_per_direction=dict(gru=3840, mamba_h=mm.mamba_cfg['d_inner_h'] * mm.mamba_cfg['d_state'],
                                              mamba_v=mm.mamba_cfg['d_inner_v'] * mm.mamba_cfg['d_state']))
    lines = '; '.join(f'{k} GRU {v["gru"]:,} / Mamba {v["mamba"]:,} ({100 * v["rel"]:+.4f}%)' for k, v in rows.items())
    record('params', ok, f'{lines}; 시퀀스 모듈 GRU {seq_g:,} vs Mamba {seq_m:,} ({100 * rel:+.4f}%), U-Net {un_g:,} = {un_m:,}, '
                         f'전체 {numel(mg):,} vs {numel(mm):,}; 해석식 일치', data)


def public_attrs(m):
    return {k: v for k, v in vars(m).items() if not k.startswith('_') and k != 'training'}


def t_identity(mm, mg):
    same_fn = PureETER_Mamba1D._seq is PureETER_GRU._seq and PureETER_Mamba1D.forward is PureETER_GRU.forward
    am, ag = public_attrs(mm), public_attrs(mg)
    missing = [k for k in ag if k not in am]
    diff = [k for k in ag if k in am and am[k] != ag[k]]
    extra = sorted(k for k in am if k not in ag)
    ok = same_fn and not missing and not diff and set(extra) <= {'fp32_stages', 'mamba_cfg'}
    record('identity', ok, f'_seq/forward 함수 객체 동일 {same_fn}; GRU 비모듈 속성 {sorted(ag)} 값 일치 '
                           f'(누락 {missing}, 다름 {diff}); Mamba 추가 속성 {extra}; 슬롯 gru_h/gru_v 의 내용물 '
                           f'{type(mm.gru_h).__name__}/{type(mm.gru_v).__name__}',
           dict(gru_attrs={k: str(v) for k, v in ag.items()}, extra=extra))


def t_unet_contract(mm, mg):
    torch.manual_seed(0)
    mu = PureETER_UNET(n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2, unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF)
    def ks(m):
        return [(k, tuple(v.shape)) for k, v in m.unet.state_dict().items()]
    w0 = mm.unet.down_path[0].block[0].weight
    ok = ks(mm) == ks(mg) == ks(mu) and w0.shape[1] == 52 and len(mm.unet.down_path) == 5 and w0.shape[0] == 2 ** 6
    record('unet', ok, f'U-Net state_dict 키·모양 Mamba == GRU == UNET: {ks(mm) == ks(mg) == ks(mu)} ({len(ks(mm))} 텐서); '
                       f'in_channels {w0.shape[1]}, depth {len(mm.unet.down_path)}, wf {int(np.log2(w0.shape[0]))}')


def sd_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def builder_args(unet_init, init='default', k_h=None, k_v=None):
    return argparse.Namespace(unet_init=unet_init, init=init, fp32_stages=(), k_h=k_h, k_v=k_v, seed=1)


def t_unetinit(mm, mg, rng_after_mamba):
    cpu = torch.device('cpu')
    res = OrderedDict()
    R.seed_all(1)
    mb = SM.make_builder(builder_args('gru'))('gru', cpu)
    rng_after_builder = torch.get_rng_state()
    eq_unet_gru = sd_equal(mb.unet, mg.unet)
    eq_seq = sd_equal(mb.gru_h, mm.gru_h) and sd_equal(mb.gru_v, mm.gru_v)
    eq_rng = torch.equal(rng_after_builder, rng_after_mamba)
    res['checksum_builder_gru'] = SM.unet_checksum(mb)
    del mb
    R.seed_all(1)
    mbu = SM.make_builder(builder_args('unet'))('gru', cpu)
    res['checksum_builder_unet'] = SM.unet_checksum(mbu)
    unet_part = mbu.unet
    del mbu
    R.seed_all(1)
    hu = R.build_model('unet', cpu)
    eq_unet_unet = sd_equal(unet_part, hu.unet)
    res['checksum_harness_unet'] = SM.unet_checksum(hu)
    del hu, unet_part
    R.seed_all(1)
    hs = R.build_model('ss2d', cpu)
    res['checksum_harness_ss2d'] = SM.unet_checksum(hs)
    del hs
    res['checksum_harness_gru'] = SM.unet_checksum(mg)
    res['checksum_mamba_own'] = SM.unet_checksum(mm)
    ok = eq_unet_gru and eq_unet_unet and eq_seq and eq_rng and res['checksum_builder_gru'] == res['checksum_harness_gru']
    record('unetinit', ok,
           f'builder(--unet-init gru) U-Net == 하네스 gru U-Net(seed_all(1)) 비트 동일 {eq_unet_gru}; '
           f'builder(--unet-init unet) == 하네스 unet {eq_unet_unet}; 시퀀스 모듈 초기값 = 같은 시드 단독 생성 {eq_seq}; '
           f'builder 뒤 전역 RNG = Mamba 단독 생성 뒤 {eq_rng}; checksum(v8fix 식) builder-gru {res["checksum_builder_gru"]:.10e} '
           f'= harness-gru {res["checksum_harness_gru"]:.10e}, harness-unet {res["checksum_harness_unet"]:.10e}, '
           f'harness-ss2d {res["checksum_harness_ss2d"]:.10e}, Mamba 자체(own) {res["checksum_mamba_own"]:.10e}', res)


def t_harness(mm):
    g = R.build_groups(mm, 'gru')
    keys_ok = all(k in g for k in ('seq', 'seq.gru_h', 'seq.gru_v', 'unet_all'))
    decay, nd, reasons, names_nd = R.split_decay(mm)
    pid_nd = {id(p) for p in nd}
    seq_nd = sorted({n.split('.', 2)[2] for n, p in mm.named_parameters() if n.startswith(('gru_h.', 'gru_v.')) and id(p) in pid_nd})
    seq_d = sorted({n.split('.', 2)[2] for n, p in mm.named_parameters() if n.startswith(('gru_h.', 'gru_v.')) and id(p) not in pid_nd})
    exp_nd = ['A_log', 'D', 'conv1d.bias', 'dt_proj.bias']
    exp_d = ['conv1d.weight', 'dt_proj.weight', 'in_proj.weight', 'out_proj.weight', 'x_proj.weight']
    n_nd_seq = sum(p.numel() for n, p in mm.named_parameters() if n.startswith(('gru_h.', 'gru_v.')) and id(p) in pid_nd)
    # seq_forward == _seq (작은 모델)
    torch.manual_seed(0)
    ms = PureETER_Mamba1D(**small_kwargs(), d_inner_h=8, d_inner_v=8, verbose=False)
    x = torch.randn(1, 4, 16, 16)
    with torch.no_grad():
        eq = torch.equal(R.seq_forward(ms, 'gru', x), ms._seq(x))
        ui, so = R.unet_input(ms, 'gru', x, torch.randn(1, 4, 16, 16))
    ok = keys_ok and seq_nd == exp_nd and seq_d == exp_d and eq and so is not None
    record('harness', ok, f'build_groups(gru) 그룹 {[k for k in g if k.startswith("seq")]} (누락 없음 assert 통과); '
                          f'시퀀스 no-decay {seq_nd} ({n_nd_seq:,} 원소), decay {seq_d}; no-decay 사유 '
                          f'{ {k: v for k, v in reasons.items()} }; R.seq_forward(model,"gru") == _seq {eq}; '
                          f'R.unet_input 이 시퀀스 출력 사용 {so is not None}',
           dict(no_decay_reasons=reasons, seq_no_decay_numel=n_nd_seq))


# ------------------------------------------------------------------ [forward]
def t_forward(mm, mg):
    val = R.build_val_dataset()
    vi = R.even_indices(len(val), 64)
    pidx = [vi[0], vi[32]]
    probe = R.load_probe(val, pidx)
    res = OrderedDict(probe_indices=pidx)
    for name, m in (('mamba', mm), ('gru', mg)):
        m.eval()
        rows, outs = [], []
        for b in probe:
            xk, xi = R.normalize_inputs(b['data'], b['data_img'], R.per_sample_scale(b['data_img']))
            t0 = time.time()
            with torch.no_grad():
                oh, ov = SM.seq_stages(m, xk)
                full = m._seq(xk)
                f0 = m._seq(torch.zeros_like(xk))
            rows.append(dict(sec=round(time.time() - t0, 1), finite=bool(torch.isfinite(ov).all() and torch.isfinite(oh).all()),
                             same_as_seq=bool(torch.equal(full, ov)), xk_rms=SM._rms(xk), zf_rms=SM._rms(xi),
                             h_out_rms=SM._rms(oh), seq_out_rms=SM._rms(ov), seq_over_zf=SM._rms(ov) / SM._rms(xi),
                             f0_rms=SM._rms(f0), f0_maxabs=float(f0.abs().max()),
                             v_fwd_energy_frac=float(ov[:, :, :192].pow(2).sum() / ov.pow(2).sum())))
            outs.append(ov)
        pair = float((outs[0] - outs[1]).norm() / outs[0].norm())
        res[name] = dict(slices=rows, pair_rel_diff=pair,
                         seq_over_zf_mean=float(np.mean([r['seq_over_zf'] for r in rows])),
                         seq_out_rms_mean=float(np.mean([r['seq_out_rms'] for r in rows])),
                         h_out_rms_mean=float(np.mean([r['h_out_rms'] for r in rows])))
    rm, rg = res['mamba'], res['gru']
    ok = all(r['finite'] and r['same_as_seq'] and r['f0_maxabs'] == 0.0 for r in rm['slices']) and rm['pair_rel_diff'] > 0
    record('forward', ok,
           f"실제 크기 CPU forward (val {pidx}, seed 1 초기값): Mamba-ETER 시퀀스 출력 RMS {rm['seq_out_rms_mean']:.3g} "
           f"(h 단 {rm['h_out_rms_mean']:.3g}), zero-filled 대비 {rm['seq_over_zf_mean']:.3g}배; ETER-net(GRU) "
           f"{rg['seq_out_rms_mean']:.3g} (h {rg['h_out_rms_mean']:.3g}), {rg['seq_over_zf_mean']:.3g}배 → Mamba/GRU 출력 비 "
           f"{rm['seq_out_rms_mean'] / rg['seq_out_rms_mean']:.2e}; f(0) max Mamba {max(r['f0_maxabs'] for r in rm['slices']):.1e} "
           f"vs GRU {max(r['f0_maxabs'] for r in rg['slices']):.3g}; 슬라이스 간 상대 차이 Mamba {rm['pair_rel_diff']:.3f} "
           f"GRU {rg['pair_rel_diff']:.3f}; 유한 {all(r['finite'] for r in rm['slices'])}", res)


# ------------------------------------------------------------------ [train]
def t_train():
    torch.manual_seed(7)
    dim, nc = 32, 2
    m = PureETER_Mamba1D(dim=dim, n_coil=nc, n_hidden_1=2, n_hidden_2=2, unet_depth=3, unet_wf=3,
                         d_inner_h=16, d_inner_v=16, verbose=False)
    m.train()
    opt, info = R.build_optimizer(m, 'DL')
    max_norm = R.CONDS['DL']['max_norm']
    crit = R.SSIM()
    g = torch.Generator().manual_seed(9)
    names = [n for n, _ in m.named_parameters()]
    w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    nd_ids = {id(p) for p in opt.param_groups[1]['params']}
    losses, grad_ok, mamba_nz, gnorms = [], True, True, []
    zero_grad_mamba = set()
    for step in range(3):
        xk = torch.randn(2, 2 * nc, dim, dim, generator=g)
        xk[..., torch.rand(dim, generator=g) < 0.69] = 0.0                                 # 비표본 열 ≈69%
        xi = D.ifft2c_packed(xk) * R.K_RATIO
        ref = xi.pow(2).sum(1, keepdim=True).sqrt() * 1.1 + 0.01
        bm = torch.ones_like(ref)
        s = R.per_sample_scale(xi)
        xk_n, xi_n = R.normalize_inputs(xk, xi, s)
        ref_n = ref / s
        lam = C.LAMBDA_SSIM_PER_PIXEL / float(s.mean())
        out = m(xi_n, xk_n).float()
        l1 = ((out - ref_n).abs() * bm).sum() / bm.sum()
        lss = 1 - crit(out, ref_n, mask=bm)
        loss = l1 + lam * lss
        loss.backward()
        for n, p in m.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                grad_ok = False
            elif n.startswith(('gru_h.', 'gru_v.')) and float(p.grad.abs().max()) == 0.0:
                mamba_nz = False
                zero_grad_mamba.add(n)
        gnorms.append(float(torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=max_norm)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
    moved = {n: float((p.detach() - w0[n]).abs().max()) for n, p in m.named_parameters()}
    unmoved = [n for n in names if moved[n] == 0.0]
    groups_moved = [any(moved[n] > 0 for n, p in m.named_parameters() if (id(p) in nd_ids) == (gi == 1)) for gi in (0, 1)]
    nd_names = sorted({n.split('.', 2)[2] for n, p in m.named_parameters() if id(p) in nd_ids and n.startswith('gru_')})
    ok = (all(np.isfinite(losses)) and grad_ok and mamba_nz and all(groups_moved)
          and nd_names == ['A_log', 'D', 'conv1d.bias', 'dt_proj.bias'] and info['type'] == 'AdamW')
    record('train', ok, f'작은 모델 3 step (AdamW 분리형 wd {info["weight_decay"]:g} + no-decay, clip {max_norm:g}): 손실 '
                        f'{[round(x, 4) for x in losses]}, clip 전 grad norm {[round(x, 3) for x in gnorms]}; 모든 기울기 유한 {grad_ok}; '
                        f'모든 Mamba 파라미터 기울기 ≠ 0 {mamba_nz} {sorted(zero_grad_mamba) or ""}; param group decay/no-decay 모두 갱신 '
                        f'{groups_moved}; 시퀀스 no-decay {nd_names}; 갱신 안 된 텐서 {unmoved or "없음"}',
           dict(losses=losses, grad_norms=gnorms, no_decay_numel=info['no_decay_numel'], decay_numel=info['decay_numel'],
                unmoved=unmoved))



# ------------------------------------------------------------------ [groups]
def t_groups(mm, mg):
    g0 = R.build_groups(mm, 'gru')
    g1 = SM.build_groups_mamba(mm, 'gru')
    same_orig = all(k in g1 and g1[k] == v for k, v in g0.items())
    sub = {k: v for k, v in g1.items() if k.startswith('seqsub.')}
    seq_names = [n for n, _ in mm.named_parameters() if n.startswith(('gru_h.', 'gru_v.'))]
    flat = [n for v in sub.values() for n in v]
    exact_cover = sorted(flat) == sorted(seq_names) and len(flat) == len(set(flat))
    only_added = set(g1) - set(g0) == set(sub)
    gg = SM.build_groups_mamba(mg, 'gru')
    gru_unchanged = list(gg) == list(R.build_groups(mg, 'gru'))
    ok = same_orig and exact_cover and only_added and gru_unchanged and len(sub) == 2 * len(SM.MAMBA_PARAM_SUBS)
    record('groups', ok, f'원 그룹 {len(g0)} 개 그대로 {same_orig}; seqsub 그룹 {len(sub)} 개가 시퀀스 파라미터 {len(seq_names)} 개를 '
                         f'정확히 한 번씩 덮음 {exact_cover}; 추가 키는 seqsub 뿐 {only_added}; ETER-net 에는 추가 없음 {gru_unchanged}',
           dict(seqsub_keys=list(sub)))


# ------------------------------------------------------------------ [alarm]
def t_alarm():
    import tempfile
    step = lambda **kw: dict(dict(type='step', step=1, amp_skipped_steps=0, nonfinite_loss_skips=0, scaler_scale=32768.0), **kw)  # noqa: E731
    summ_mv = dict(status='completed', steps_done=10, steps_requested=10, verdict=dict(seq='moving', seq_used='not shown'),
                   final_groups=dict(seq=dict(rel_change=0.3, eps_dom_frac=0.1)))
    summ_eps = dict(summ_mv, verdict=dict(seq='eps-limited', seq_used='not shown'),
                    final_groups=dict(seq=dict(rel_change=0.05, eps_dom_frac=0.97)))
    summ_fz = dict(summ_mv, verdict=dict(seq='frozen', seq_used='not shown'))
    summ_used = dict(summ_mv, verdict=dict(seq='eps-limited', seq_used='used'))
    gp_uf = dict(fp16_underflow=dict(triggered=True, reasons=['gru_h.in_proj.weight fp16 0 비율 0.7 > 0.1']))
    gp_ok = dict(fp16_underflow=dict(triggered=False, reasons=[]))
    cases = OrderedDict(
        clean=(SM.precision_alarm([step()], 'completed', summ_mv, gp_ok)['triggered'], False),
        eps_limited=(SM.precision_alarm([step()], 'completed', summ_eps, gp_ok)['triggered'], True),
        frozen=(SM.precision_alarm([step()], 'completed', summ_fz, gp_ok)['triggered'], True),
        underflow=(SM.precision_alarm([step()], 'completed', summ_mv, gp_uf)['triggered'], True),
        amp21=(SM.precision_alarm([step(amp_skipped_steps=21)], 'completed', summ_mv, gp_ok)['triggered'], True),
        scale_low=(SM.precision_alarm([step(scaler_scale=512.0)], 'completed', summ_mv, gp_ok)['triggered'], True),
        aborted=(SM.precision_alarm([step()], 'aborted_nonfinite', summ_mv, gp_ok)['triggered'], True),
        legacy_sig=(SM.precision_alarm([step()], 'completed')['triggered'], False),
    )
    base = SM.parse_args(['--device', 'cpu', '--steps', '10', '--num-workers', '1'])
    cuda, cpu = torch.device('cuda'), torch.device('cpu')
    al = dict(triggered=True)
    fb = OrderedDict(
        fp16_cuda=(SM.wants_fallback(base, al, (), cuda), True),
        fp32h_cuda=(SM.wants_fallback(base, al, ('h',), cuda), True),
        fp32hv_cuda=(SM.wants_fallback(base, al, ('h', 'v'), cuda), False),
        cpu=(SM.wants_fallback(base, al, (), cpu), False),
        not_triggered=(SM.wants_fallback(base, dict(triggered=False), (), cuda), False),
        stages_hv=(SM.FALLBACK_STAGES == ('h', 'v') and SM.vname_of(SM.cond_args(base, 'outmatch', SM.FALLBACK_STAGES))
                   == 'mamba1d_outmatch_fp32hv', True),
        inits_default=(SM.parse_args(['--device', 'cpu']).inits == ['default', 'outmatch'], True),
        max_minutes_default=(SM.parse_args(['--device', 'cpu']).max_minutes == 0, True),
    )
    interp = OrderedDict(
        used=(SM.interpret(summ_used), '사용'),
        eps=(SM.interpret(summ_eps), '해석 불가 (seq eps-limited)'),
        frozen=(SM.interpret(summ_fz), '해석 불가 (seq frozen)'),
        moving=(SM.interpret(summ_mv), 'not shown (결론 불가)'),
        truncated=(SM.interpret(dict(summ_mv, status='truncated_time', steps_done=5)), '비교 불가 (학습 미완)'),
    )
    with tempfile.TemporaryDirectory() as td:
        a = SM.cond_args(base, 'default', ())
        a.out_dir = td

        def put(status='completed', steps_done=10, nw=1, rv=SM.RUNNER_VERSION, err=None):
            meta = dict(runner_version=rv, steps=10, seed=1, bs=a.bs, init='default', unet_init='gru', fp32_stages=[],
                        val_slices_arg=a.val_slices, post_slices=a.post_slices, num_workers=nw)
            SM.write_json(os.path.join(td, f'{SM.vname_of(a)}_final.json'),
                          dict(meta=meta, status=status, error=err,
                               run_job_summary=dict(status=status.split('+')[0], steps_done=steps_done)))
            return SM.done_ok(a)[0]
        dk = OrderedDict(
            completed=(put(), True),
            truncated=(put(status='truncated_time', steps_done=5), False),
            unet_mismatch=(put(status='completed+unet_init_mismatch'), False),
            post_failed=(put(status='completed+post_failed', err='x'), False),
            workers=(put(nw=16), False),
            old_version=(put(rv='2026-10-11a'), False),
            refused=(put(status='refused_unet_init', err='x'), False),
        )
    allc = OrderedDict(list(('alarm.' + k, v) for k, v in cases.items()) + list(('fallback.' + k, v) for k, v in fb.items())
                       + list(('interpret.' + k, v) for k, v in interp.items()) + list(('done_ok.' + k, v) for k, v in dk.items()))
    bad = [k for k, (got, exp) in allc.items() if got != exp]
    record('alarm', not bad, f'{len(allc)} 사례 중 기대와 다른 것 {bad or "없음"} (fallback 단계 hv, 해석 규칙, done_ok 거부 사유)',
           {k: dict(got=str(g), expected=str(e)) for k, (g, e) in allc.items()})


# ------------------------------------------------------------------ [unetpre]
def t_unetpre(mm):
    cpu = torch.device('cpu')
    R.seed_all(1)
    mb = SM.make_builder(builder_args('gru'))('gru', cpu)
    good = SM.unet_init_precheck(mb, builder_args('gru'))
    del mb
    bad = SM.unet_init_precheck(mm, builder_args('gru'))               # mm = Mamba-ETER 자체 생성 U-Net (own)
    skip = SM.unet_init_precheck(mm, builder_args('own'))
    ok = good['match'] is True and bad['match'] is False and skip['match'] is None
    record('unetpre', ok, f'builder(gru) 학습 전 대조 {good["match"]} (rel {good["rel_diff"]}, checksum {good.get("checksum")}); '
                          f'own U-Net → {bad["match"]} (rel {bad["rel_diff"]}); --unet-init own → 대조 안 함 {skip["match"]}',
           dict(good=good, bad=bad))


# ------------------------------------------------------------------ [gradscale]
def t_gradscale():
    cpu = torch.device('cpu')
    val = R.build_val_dataset()
    vi = R.even_indices(len(val), 64)
    probe = R.load_probe(val, [vi[0], vi[32]])
    with open(os.path.join(SM.REF_DIR, 'gru_DL_summary.json')) as f:
        dl_mean = json.load(f)['dl_scale']['mean']
    lam = C.LAMBDA_SSIM_PER_PIXEL / dl_mean
    res = OrderedDict(lam=lam)
    a = builder_args('gru')
    res['default'] = SM.grad_preflight(a, SM.make_builder(a), probe, cpu, lam)
    train = R.build_train_dataset()
    cal = SM.calibrate_outmatch(builder_args('gru', 'outmatch'), train, cpu)
    a2 = builder_args('gru', 'outmatch', cal['k_h'], cal['k_v'])
    res['outmatch_calibration'] = cal
    res['outmatch'] = SM.grad_preflight(a2, SM.make_builder(a2), probe, cpu, lam)

    def finite(d):
        if isinstance(d, dict):
            return all(finite(v) for v in d.values())
        if isinstance(d, float):
            return np.isfinite(d)
        return True
    fin = finite(res)
    r_def, r_om = res['default']['in_proj_rms_over_gru_W_ih'], res['outmatch']['in_proj_rms_over_gru_W_ih']
    lifted = all(0.1 <= r_om[st] <= 10 for st in ('gru_h', 'gru_v'))
    d32, o32 = res['default']['mamba']['fp32'], res['outmatch']['mamba']['fp32']
    delta_keys = [f'{st}.{k}' for st in ('gru_h', 'gru_v') for k in ('dt_proj.weight', 'x_proj.weight', 'A_log')]
    ok = fin and lifted
    dd = {k: float('%.1e' % d32[k]['rms']) for k in delta_keys}
    do = {k: float('%.1e' % o32[k]['rms']) for k in delta_keys}
    record('gradscale', ok,
           f"실제 크기 CPU fp32 (probe vi[0], λ {lam:.4g}): default 시퀀스 기울기 RMS {res['default']['mamba']['fp32_seq_total']['rms']:.2e} "
           f"(<eps {100 * res['default']['mamba']['fp32_seq_total']['frac_lt_eps']:.1f}%) vs ETER-net "
           f"{res['default']['gru']['fp32_seq_total']['rms']:.2e} (<eps {100 * res['default']['gru']['fp32_seq_total']['frac_lt_eps']:.1f}%); "
           f"in_proj/W_ih RMS 비 default h {r_def['gru_h']:.3g} v {r_def['gru_v']:.3g} → outmatch(k {cal['k_h']:.3g}/{cal['k_v']:.3g}) "
           f"h {r_om['gru_h']:.3g} v {r_om['gru_v']:.3g} (0.1~10 이면 통과: {lifted}); outmatch 시퀀스 <eps "
           f"{100 * res['outmatch']['mamba']['fp32_seq_total']['frac_lt_eps']:.1f}%; Δ 경로 RMS default "
           f"{dd} / outmatch {do} (eps 1e-8); 유한 {fin}", res)

# ------------------------------------------------------------------ [orig]
PROTECTED = ['models/hybrid_eternet', 'models/mamba_eternet', 'models/pure_eternet/u_pure_eternet_gru.py',
             'models/pure_eternet/u_pure_eternet_unet.py', 'models/pure_eternet/u_pure_eternet_ss2d.py',
             'v8_eter_pure/smoke_recipe_fix.py', 'v8_eter_pure/main_train_pure_v8fix.py', 'v8_eter_pure/configs',
             'configs', 'dataloaders', 'infra/runpod']
UNTRACKED_DIAG = ['v8_eter_pure/diag_ss2d_variants.py', 'v8_eter_pure/diag_ss2d_smallout.py']


def t_orig():
    def git(*a):
        return subprocess.run(['git', '-C', _PROJECT_ROOT, *a], capture_output=True, text=True, timeout=120)
    orig = [p for p in git('ls-tree', '-r', '--name-only', '7d4e4e0').stdout.splitlines() if p]
    st = git('status', '--porcelain', '--', *PROTECTED).stdout.strip()
    st_orig = ''
    for i in range(0, len(orig), 200):
        st_orig += git('status', '--porcelain', '--', *orig[i:i + 200]).stdout
    st_orig = st_orig.strip()
    # infra/runpod 의 미추적 파일(작업 전부터 있던 것)은 '??' 로 보이므로 수정(M/D) 만 센다
    mod = [ln for ln in st.splitlines() if not ln.startswith('??')]
    mod_orig = [ln for ln in st_orig.splitlines() if not ln.startswith('??')]
    sha = {}
    for p in UNTRACKED_DIAG:
        h = hashlib.sha256()
        with open(os.path.join(_PROJECT_ROOT, p), 'rb') as f:
            h.update(f.read())
        sha[p] = h.hexdigest()
    ok = not mod and not mod_orig
    record('orig', ok, f'보호 경로 수정 {mod or "없음"}; 교수님 원본(7d4e4e0, {len(orig)} 파일) 수정 {mod_orig or "없음"}; '
                       f'미추적 진단 스크립트 sha256 {sha}',
           dict(protected_status=st, original_status=st_orig, untracked_sha256=sha))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description='Mamba-ETER CPU 점검')
    ap.add_argument('--skip-real', action='store_true', help='실제 크기 모델 검사(params·identity·unet·unetinit·harness·forward·'
                                                            'routing_real)를 건너뜀')
    ap.add_argument('--json-out', default='results/mamba1d_smoke_2026-10-11/sanity_cpu.json')
    a = ap.parse_args()
    if not os.path.isabs(a.json_out):
        a.json_out = os.path.join(_PROJECT_ROOT, a.json_out)
    print('=' * 72)
    print(f' Mamba-ETER CPU 점검 (러너 {SM.RUNNER_VERSION})  torch {torch.__version__}  threads {torch.get_num_threads()}  {SM.now()}')
    print('=' * 72, flush=True)
    R.install_cpu_shims(False)
    SM.install_mamba_cpu_shims()
    t_all = time.time()

    run_test('scan', t_scan)
    run_test('bidir', t_bidir)
    run_test('dep', t_dep)
    run_test('routing_small', t_routing_small)
    run_test('train', t_train)
    run_test('alarm', t_alarm)
    if not a.skip_real:
        cpu = torch.device('cpu')
        t0 = time.time()
        R.seed_all(1)
        mm = PureETER_Mamba1D(dim=C.IMAGE_SIZE[0], n_coil=C.N_COIL, n_hidden_1=C.N_HIDDEN_LRNN_1,
                              n_hidden_2=C.N_HIDDEN_LRNN_2, unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF)
        rng_after_mamba = torch.get_rng_state()
        R.seed_all(1)
        mg = R.build_model('gru', cpu)                       # 스모크 하네스와 같은 생성 (seed_all → build_model)
        print(f'  실제 크기 모델 생성 {time.time() - t0:.0f}s', flush=True)
        run_test('params', t_params, mm, mg)
        run_test('identity', t_identity, mm, mg)
        run_test('unet', t_unet_contract, mm, mg)
        run_test('unetinit', t_unetinit, mm, mg, rng_after_mamba)
        run_test('harness', t_harness, mm)
        run_test('groups', t_groups, mm, mg)
        run_test('unetpre', t_unetpre, mm)
        run_test('forward', t_forward, mm, mg)
        run_test('routing_real', t_routing_real, mm, mg)      # 마지막 — gru_h/gru_v 를 stub 으로 바꾼다
        del mm, mg
        run_test('gradscale', t_gradscale)
    run_test('orig', t_orig)

    n_fail = sum(1 for r in RESULTS.values() if not r['passed'])
    out = OrderedDict(script='v8_eter_pure/sanity_mamba1d.py', runner_version=SM.RUNNER_VERSION, finished=SM.now(), seconds=round(time.time() - t_all, 1),
                      torch=torch.__version__, threads=torch.get_num_threads(), skip_real=a.skip_real,
                      n_tests=len(RESULTS), n_fail=n_fail, results=RESULTS)
    os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
    SM.write_json(a.json_out, out)
    print('=' * 72)
    print(f' 결과: {len(RESULTS) - n_fail}/{len(RESULTS)} 통과 ({out["seconds"]:.0f}s) → {a.json_out}')
    for k, r in RESULTS.items():
        print(f'   {"PASS" if r["passed"] else "FAIL"}  {k}')
    print('=' * 72, flush=True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
