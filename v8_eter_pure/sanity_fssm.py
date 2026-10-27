"""
FSSM (OU) 계약 점검 (2026-10-27) — 설계 docs/fssm_design_2026-10-27.md §4.
CPU (기본 — 로컬 GPU 가 학습 중일 때도 돌 수 있음, SSM 은 selective_scan_ref):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 19 python -B v8_eter_pure/sanity_fssm.py [--out x.json]
GPU 점검까지 (GPU 가 비었을 때만):
  CUDA_VISIBLE_DEVICES=0 python -B v8_eter_pure/sanity_fssm.py --gpu [--out x.json]
도우미(check·nparams·emulate_same_unet_init)는 sanity_ou.py 의 것을 그대로 쓴다.

점검 (하나라도 실패하면 종료 코드 1):
  1a. 하이브리드 → 영상: ifft1c_W(ifft1c_H(x)) = ifft2c_ri(x) (u_pure_eternet_ou_ss2d 의 로더 규약 함수), 실제 val 슬라이스.
  1b. 학습형 변환(FSSM-T) 초기값 = 중심 IDFT: 행렬 최대 오차, 난수 입력에서 ifft1c_W 와의 상대 오차.
  2a. 초기 FSSM 출력 = x_img (정규화한 실제 val 슬라이스, 세 모델).
  2b. 트레이너 SAME_UNET_INIT 절차 뒤 초기 모델 출력 = 같은 시드의 U-Net only (OU) 출력 (세 모델), U-Net 텐서는 첫 합성곱 weight 를
      뺀 전부 비트 일치이고 첫 합성곱 weight = 기준 [:, 20:52] (비트 일치), rec 의 own_init 비어 있음.
  3.  selective_scan_ref 규약(복소 A: B·C 실수 (B, N, 2L) 위치마다 (re, im) 교번, 출력 2·Re + D·u) 과 DirectionalSSM 의 B·C 배치를
      순차 계산과 대조 (복소판·실수판).
  4.  건강 감시: seq 그룹 near_dead(|w|<1e-6) 비율 < 0.5 (초기값), 그룹에 ss2d.* 전부, 파라미터 수(실수판 = 복소판), no-decay 이름.
  5.  기울기: (a) 초기값(out_proj = 0)에서 out_proj·U-Net 기울기 ≠ 0, 보정 경로 앞쪽은 정확히 0, 학습형 변환(FSSM-T)은 ≠ 0.
      (b) out_proj 를 작은 값으로 바꾼 뒤 ss2d.*
      모든 텐서의 기울기 ≠ 0, 유한 (세 모델, 작은 입력 (1, 32, 32, 384)).
  6.  CPU forward 시간 (1, 32, 384, 384) 한 슬라이스 — FSSM·U-Net 따로 (기록만).
  --gpu:
  g1. CUDA selective_scan_fn = selective_scan_ref (복소 A·실수 A, fp32, fwd·bwd — 작은 크기).
  g2. BS 8 학습 30 step 스모크 (fssm_ou, autocast + GradScaler, AdamW lr 2e-4, 실제 val 슬라이스 8 개 반복): NaN 없음, s/step,
      최대 VRAM < 22 GB.
"""

import os
import sys
import json
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.append(_HERE)

import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402

import sanity_ou as SO             # noqa: E402  (recipe_ou·smoke_recipe_fix 를 import)
import recipe_ou_fssm as ROF       # noqa: E402  (FSSM 이름을 recipe_ou 등록부에 더함)

RO, R, C = SO.RO, SO.R, SO.C
check = SO.check
NAMES = ROF.FSSM_NAMES
NEAR_THR = 1e-6


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-30))


def _val_slices(n):
    ds = R.build_val_dataset()
    idx = R.even_indices(len(ds), 64)
    pick = [idx[(k * len(idx)) // n + len(idx) // (2 * n)] for k in range(n)]
    out = []
    for i in pick:
        b = ds[i]
        x_ksp = torch.as_tensor(b['data']).float().unsqueeze(0)
        x_img = torch.as_tensor(b['data_img']).float().unsqueeze(0)
        s = R.per_sample_scale(x_img)
        x_ksp, x_img = R.normalize_inputs(x_ksp, x_img, s)
        lab = torch.as_tensor(b['label']).float().unsqueeze(0) if 'label' in b else None
        out.append(dict(idx=i, x_ksp=x_ksp, x_img=x_img, s=s, label=lab))
    return out


def near_dead(model, names):
    c = t = 0
    params = dict(model.named_parameters())
    for n in names:
        p = params[n].detach()
        c += int((p.abs() < NEAR_THR).sum())
        t += p.numel()
    return c / max(t, 1), c, t


def _synthetic_ksp(h, w, gen):
    """작은 기울기 시험 입력: 난수 복소 k-space (1, 32, h, w), 등간격 R4(시작 3) + 중앙 31 줄만 남김."""
    x = torch.randn(1, 32, h, w, generator=gen)
    m = torch.zeros(w)
    m[3::4] = 1
    c = w // 2
    m[c - 15:c + 16] = 1
    return x * m[None, None, None, :]


def cpu_checks(slices):
    import u_pure_eternet_fssm as M
    import u_pure_eternet_ou_ss2d as MS
    dev = torch.device('cpu')
    sl = slices[0]
    x_ksp, x_img = sl['x_ksp'], sl['x_img']

    # 1a
    xc = M.ri_to_complex(x_ksp)
    via = M.complex_to_ri(M.ifft1c(M.ifft1c(xc, dim=-2), dim=-1))
    ref2 = MS.ifft2c_ri(x_ksp)
    check('1a ifft1c_W∘ifft1c_H = ifft2c_ri (val 슬라이스)', _rel(via, ref2) < 1e-6, rel=_rel(via, ref2),
          rel_vs_x_img=_rel(ref2, x_img))

    # 1b
    T = M.VandermondeIDFT(384)
    with torch.no_grad():
        Mt = T.matrix()
        eye = torch.eye(384, dtype=torch.complex64)
        Mref = M.ifft1c(eye, dim=0)                     # 열 t = e_t 의 중심 IDFT → Mref[n, t]
        g = torch.Generator().manual_seed(3)
        k = torch.complex(torch.randn(2, 16, 8, 384, generator=g), torch.randn(2, 16, 8, 384, generator=g))
        rel_v = _rel(T(k), M.ifft1c(k, dim=-1))
    err_m = float((Mt - Mref).abs().max())
    check('1b 학습형 변환 초기값 = 중심 IDFT', err_m < 1e-5 and rel_v < 1e-5, matrix_maxabs=err_m, rel_on_random=rel_v)

    # 2a / 2b
    for name in NAMES:
        model, rec = SO.emulate_same_unet_init(name, 1)
        R.seed_all(1)
        ref = RO.build_model('unet_ou', dev)
        model.eval()
        ref.eval()
        with torch.no_grad():
            y0 = model.ss2d(x_ksp)
            out_m = model(x_img, x_ksp)
            out_r = ref(x_img, x_ksp)
        check(f'2a {name}: 초기 FSSM 출력 = x_img', y0.dtype == torch.float32 and _rel(y0, x_img) < 1e-5,
              rel=_rel(y0, x_img), dtype=str(y0.dtype), shape=list(y0.shape))
        sd_m, sd_r = model.unet.state_dict(), ref.unet.state_dict()
        first = [k for k in sd_m if tuple(sd_m[k].shape) != tuple(sd_r[k].shape)]
        same = all(torch.equal(sd_m[k], sd_r[k]) for k in sd_m if k not in first)
        sliced_ok = (len(first) == 1 and torch.equal(sd_m[first[0]], sd_r[first[0]][:, 20:52]))
        note = RO.unet_init_note(name, rec)
        check(f'2b {name}: 초기 모델 출력 = U-Net only (OU) (SAME_UNET_INIT)',
              _rel(out_m, out_r) < 1e-4 and same and sliced_ok and rec['own_init'] == [] and len(rec['sliced']) == 1,
              rel=_rel(out_m, out_r), other_unet_tensors_equal=same, first_conv=first, first_conv_slice_equal=sliced_ok,
              n_copied=rec['n_copied'], n_total=rec['n_total'], note=note)
        del model, ref

    # 3
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    g = torch.Generator().manual_seed(0)
    Bp, D, N, L = 2, 3, 4, 7
    u = torch.randn(Bp, D, L, generator=g)
    dl = torch.rand(Bp, D, L, generator=g) * 0.5 + 0.05
    A = torch.complex(-torch.rand(D, N, generator=g) - 0.1, torch.randn(D, N, generator=g) * 3)
    Bm, Cm, Dv = torch.randn(Bp, N, 2 * L, generator=g), torch.randn(Bp, N, 2 * L, generator=g), torch.randn(D, generator=g)
    y = selective_scan_ref(u, dl, A, Bm, Cm, Dv)
    Bc, Cc = torch.complex(Bm[..., 0::2], Bm[..., 1::2]), torch.complex(Cm[..., 0::2], Cm[..., 1::2])
    x = torch.zeros(Bp, D, N, dtype=torch.complex64)
    ys = []
    for t in range(L):
        x = torch.exp(dl[:, :, t, None] * A[None]) * x + (dl[:, :, t, None] * u[:, :, t, None]) * Bc[:, None, :, t]
        ys.append(2 * (x * Cc[:, None, :, t]).sum(-1).real + Dv[None] * u[:, :, t])
    e_ref = float((y - torch.stack(ys, -1)).abs().max())
    errs = {}
    for cplx in (True, False):
        torch.manual_seed(5)
        ssm = M.DirectionalSSM(8, 4, 1, cplx)
        uu = torch.randn(2, 9, 8, generator=g)
        with torch.no_grad():
            yy = ssm(uu)
            xd = ssm.x_proj(uu)
            dt, bc = xd.split([1, ssm.n_bc], dim=-1)
            delta = F.softplus(ssm.dt_proj(dt))                       # (2, 9, 8)
            n4 = ssm.d_state
            if cplx:
                b = torch.complex(bc[..., :n4], bc[..., n4:2 * n4])
                c = torch.complex(bc[..., 2 * n4:3 * n4], bc[..., 3 * n4:])
            else:
                b, c = bc[..., :n4], bc[..., n4:]
            Am = ssm.A()
            st = torch.zeros(2, 8, n4, dtype=torch.complex64 if cplx else torch.float32)
            outs = []
            for t in range(9):
                st = torch.exp(delta[:, t, :, None] * Am[None]) * st + (delta[:, t, :, None] * uu[:, t, :, None]) * b[:, t, None, :]
                yt = (st * c[:, t, None, :]).sum(-1)
                yt = 2 * yt.real if cplx else yt
                outs.append(yt + ssm.D[None] * uu[:, t])
            errs['complex' if cplx else 'real'] = float((yy - torch.stack(outs, 1)).abs().max())
    check('3 selective_scan_ref 규약·DirectionalSSM B·C 배치 = 순차 계산', e_ref < 1e-5 and max(errs.values()) < 1e-5,
          ref_vs_loop=e_ref, directional_vs_loop=errs)

    # 4
    counts = {}
    for name in NAMES:
        R.seed_all(1)
        model = RO.build_model(name, dev)
        groups = RO.build_groups(model, name)
        seq_names = groups.get('seq', [])
        all_ss2d = sorted(n for n, _ in model.named_parameters() if n.startswith('ss2d.'))
        frac, nz, tot = near_dead(model, seq_names)
        deep = near_dead(model, groups['unet_deep'])[0]
        _, nd, _, names_nd = R.split_decay(model)
        counts[name] = dict(seq=int(sum(p.numel() for p in model.ss2d.parameters())),
                            unet=int(sum(p.numel() for p in model.unet.parameters())), total=SO.nparams(model))
        check(f'4 {name}: 건강 감시 seq near_dead < 0.5 · 그룹', frac < 0.5 and deep < 0.5 and sorted(seq_names) == all_ss2d,
              seq_near_dead=round(frac, 4), zeros=nz, seq_numel=tot, unet_deep_near_dead=round(deep, 6),
              subgroups={k: len(v) for k, v in groups.items() if k.startswith('seq')},
              no_decay_seq=[n for n in names_nd if n.startswith('ss2d.')], **counts[name])
    check('4 파라미터 수: FSSM-real = FSSM, FSSM-T = FSSM + 6·384 (변환 c·b 복소 + η·ρ)',
          counts['fssmr_ou']['seq'] == counts['fssm_ou']['seq']
          and counts['fssmt_ou']['seq'] == counts['fssm_ou']['seq'] + 6 * 384, counts=counts)

    # 5
    g = torch.Generator().manual_seed(11)
    xk = _synthetic_ksp(32, 384, g)
    xi = MS.ifft2c_ri(xk)
    for name in NAMES:
        R.seed_all(1)
        model = RO.build_model(name, dev)
        model.train()
        out = model(xi, xk)
        (out.float() ** 2).mean().backward()
        gp = {n: (None if p.grad is None else float(p.grad.norm())) for n, p in model.named_parameters()}
        # 보정 경로의 앞쪽(out_proj 앞) = 0 이어야 함. 학습형 변환(transform.*)은 z 에 바로 걸려 초기값에서도 기울기가 있어야 함.
        front = [n for n in gp if n.startswith('ss2d.') and not n.startswith(('ss2d.out_proj.', 'ss2d.transform.'))]
        tr = [n for n in gp if n.startswith('ss2d.transform.')]
        a_ok = (gp['ss2d.out_proj.weight'] or 0) > 0 and (gp['unet.last.weight'] or 0) > 0 and \
            all((gp[n] or 0.0) == 0.0 for n in front) and all((gp[n] or 0.0) > 0 for n in tr)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.ss2d.out_proj.weight.normal_(0, 1e-2, generator=g)
            model.ss2d.out_proj.bias.normal_(0, 1e-2, generator=g)
        out = model(xi, xk)
        (out.float() ** 2).mean().backward()
        zero = [n for n, p in model.named_parameters() if n.startswith('ss2d.') and (p.grad is None or float(p.grad.norm()) == 0.0)]
        nonfinite = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        check(f'5 {name}: 기울기 (a 초기값: out_proj·U-Net ≠ 0, 앞쪽 = 0 / b out_proj 작은 값: ss2d.* 전부 ≠ 0)',
              a_ok and not zero and not nonfinite and torch.isfinite(out).all().item(),
              a_out_proj=gp['ss2d.out_proj.weight'], a_front_max=max((gp[n] or 0.0) for n in front),
              a_transform={n[len('ss2d.transform.'):]: gp[n] for n in tr},
              b_zero_grad=zero, nonfinite=nonfinite)

    # 6
    R.seed_all(1)
    model = RO.build_model('fssm_ou', dev).eval()
    with torch.no_grad():
        t0 = time.time()
        y = model.ss2d(x_ksp)
        t1 = time.time()
        _ = model.unet(y)
        t2 = time.time()
    check('6 CPU forward 시간 (1, 32, 384, 384) — 기록', True, fssm_s=round(t1 - t0, 2), unet_s=round(t2 - t1, 2),
          threads=torch.get_num_threads())


def gpu_checks(slices):
    import u_pure_eternet_fssm as M
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    dev = torch.device('cuda')
    g = torch.Generator(device='cpu').manual_seed(21)
    res = {}
    for cplx in (True, False):
        Bp, D, N, L = 4, 16, 8, 64
        u = torch.randn(Bp, D, L, generator=g)
        dl = torch.rand(Bp, D, L, generator=g) * 0.1 + 0.001
        if cplx:
            A = torch.complex(-torch.rand(D, N, generator=g) * 0.5 - 0.1, torch.randn(D, N, generator=g) * 3)
            Bm, Cm = torch.randn(Bp, N, 2 * L, generator=g), torch.randn(Bp, N, 2 * L, generator=g)
        else:
            A = -torch.rand(D, N, generator=g) - 0.1
            Bm, Cm = torch.randn(Bp, N, L, generator=g), torch.randn(Bp, N, L, generator=g)
        Dv = torch.randn(D, generator=g)
        outs, grads = [], []
        for fn in (selective_scan_fn, selective_scan_ref):
            ts = [t.clone().to(dev).requires_grad_(True) for t in (u, dl, A, Bm, Cm, Dv)]
            y = fn(*ts)
            y.backward(torch.ones_like(y))
            outs.append(y.detach().float().cpu())
            grads.append([t.grad.detach().cpu() for t in ts])
        gerr = [_rel(a, b) for a, b in zip(grads[0], grads[1])]
        res['complex' if cplx else 'real'] = dict(fwd=_rel(outs[0], outs[1]), bwd=dict(zip(('u', 'delta', 'A', 'B', 'C', 'D'), gerr)))
    ok = all(v['fwd'] < 1e-4 and max(v['bwd'].values()) < 1e-3 for v in res.values())
    check('g1 CUDA selective_scan_fn = selective_scan_ref (복소·실수 A, fp32, fwd·bwd)', ok, **res)

    # g2
    R.seed_all(1)
    model = RO.build_model('fssm_ou', dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=3e-5)
    scaler = torch.cuda.amp.GradScaler()
    xs = torch.cat([s['x_ksp'] for s in slices]).to(dev)
    xi = torch.cat([s['x_img'] for s in slices]).to(dev)
    # 출력 × s 가 label 단위(R.run_validation 의 back) → 목표 = label / s
    lab = torch.cat([sl['label'].reshape(1, 1, *sl['label'].shape[-2:]) / sl['s'].reshape(1, 1, 1, 1) for sl in slices]).to(dev) \
        if all(sl['label'] is not None for sl in slices) else torch.zeros(xs.shape[0], 1, *xs.shape[-2:], device=dev)
    torch.cuda.reset_peak_memory_stats()
    times, losses, bad = [], [], 0
    for step in range(30):
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(xi, xs)
            loss = F.l1_loss(out.float(), lab)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(float(loss))
        bad += int(not torch.isfinite(loss).item())
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    sp = sum(times[3:]) / max(1, len(times) - 3)
    check('g2 BS 8 학습 30 step 스모크 (fssm_ou, autocast)', bad == 0 and peak < 22.0, s_per_step=round(sp, 3),
          peak_vram_gib=round(peak, 2), loss_first=losses[0], loss_last=losses[-1], nonfinite_steps=bad,
          out_proj_norm=float(model.ss2d.out_proj.weight.norm()), amp_scale=float(scaler.get_scale()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true', help='CUDA 커널 대조 + BS 8 스모크 (GPU 가 비었을 때만)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.gpu:
        assert torch.cuda.is_available(), '--gpu 인데 CUDA 가 없음'
    slices = _val_slices(8 if args.gpu else 1)
    cpu_checks(slices[:1])
    if args.gpu:
        gpu_checks(slices)
    print(f'\n{len(SO.RESULTS) - len(SO.FAILS)}/{len(SO.RESULTS)} 점검 통과' + (f' — 실패 {SO.FAILS}' if SO.FAILS else ''))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(dict(ok=not SO.FAILS, fails=SO.FAILS, results=SO.RESULTS), f, ensure_ascii=False, indent=1, default=str)
    return 1 if SO.FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
