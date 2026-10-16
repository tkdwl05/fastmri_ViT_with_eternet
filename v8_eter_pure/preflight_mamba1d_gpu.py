"""Mamba-ETER GPU 사전 점검 (2026-10-12) — 새 GPU(RunPod RTX 3090 등)에서 50 epoch 학습 전에 한 번 돌린다. 학습과 무관(읽기 전용).

smoke_mamba1d.py 의 preflight 와 같은 기준을 의존성 없이 다시 쓴 것 (4000 step 시험은 로컬 TITAN RTX 에서만 돌았다):
  (1) 작은 MambaOut 블록: fast path(causal_conv1d + selective_scan CUDA 커널) vs 참조 경로(torch conv1d + selective_scan_ref) —
      fp32 출력 상대오차 < 1e-4, 입력 기울기 < 1e-3, 파라미터 기울기 최대 < 1e-4 (smoke 기준 5e-2 보다 엄격 — 실측 ≈1e-6).
  (2) 실제 크기 h 단(val 1 슬라이스, 학습과 같은 정규화) fast vs 참조 출력 (fp32, no grad) < 1e-3.
  (3) 실제 모델(fp32_stages hv) val 1 슬라이스의 autocast forward + backward: 출력·기울기 유한, 시퀀스 모듈 기울기 노름 > 0, 시퀀스 출력 fp32.
실패하면 exit 1. 결과는 --out json.
  예) CUDA_VISIBLE_DEVICES=0 /opt/mri310/bin/python v8_eter_pure/preflight_mamba1d_gpu.py --out /root/_setup/preflight_mamba1d.json
"""
import os
import sys
import json
import time
import argparse
import contextlib

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, os.path.join(_ROOT, 'models', 'pure_eternet'), os.path.join(_ROOT, 'models', 'hybrid_eternet')]

import smoke_recipe_fix as R                                                   # noqa: E402
import mamba_ssm.modules.mamba_simple as MS                                    # noqa: E402
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref          # noqa: E402
from u_pure_eternet_mamba1d import PureETER_Mamba1D, MambaOut                  # noqa: E402

C = R.C


@contextlib.contextmanager
def reference_path():
    old = (MS.causal_conv1d_fn, MS.selective_scan_fn)
    MS.causal_conv1d_fn, MS.selective_scan_fn = None, selective_scan_ref
    try:
        yield
    finally:
        MS.causal_conv1d_fn, MS.selective_scan_fn = old


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', default=os.path.join(_ROOT, 'fastMRI_data', 'multicoil_val'))
    p.add_argument('--out', required=True)
    a = p.parse_args()
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False          # 트레이너와 같음 (ALLOW_TF32 기본 0)
    torch.backends.cudnn.allow_tf32 = False
    res = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               fast_fns={k: getattr(MS, k) is not None for k in ('causal_conv1d_fn', 'selective_scan_fn', 'mamba_inner_fn')})
    ok = all(res['fast_fns'].values())

    # (1) 작은 블록
    torch.manual_seed(12345)
    blk = MambaOut(256, 512, 96).to(dev)
    x = torch.randn(2, 384, 256, device=dev, requires_grad=True)
    names = [n for n, _ in blk.named_parameters()]
    params = [q for _, q in blk.named_parameters()]

    def fwd_bwd():
        y = blk(x)
        return y.detach(), torch.autograd.grad(y.square().sum(), [x] + params)

    y_f, g_f = fwd_bwd()
    with reference_path():
        y_r, g_r = fwd_bwd()
    pr = {n: rel(u, v) for n, u, v in zip(names, g_f[1:], g_r[1:])}
    res['small_block'] = dict(out=rel(y_f, y_r), grad_in=rel(g_f[0], g_r[0]), grad_params=pr, grad_params_max=max(pr.values()))
    sb = res['small_block']
    ok &= sb['out'] < 1e-4 and sb['grad_in'] < 1e-3 and sb['grad_params_max'] < 1e-4   # 실측 ≈1e-6 (TITAN·3090)
    del blk, x, y_f, y_r, g_f, g_r, params

    # (2)(3) 실제 모델 · val 1 슬라이스
    R.seed_all(1)
    m = PureETER_Mamba1D(dim=C.IMAGE_SIZE[0], n_coil=C.N_COIL, n_hidden_1=C.N_HIDDEN_LRNN_1, n_hidden_2=C.N_HIDDEN_LRNN_2,
                         unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF, use_dc=False, fp32_stages=('h', 'v')).to(dev)
    from dataloader_h5_v5 import FastMRI_H5_Dataloader
    ds = FastMRI_H5_Dataloader(a.data_path, num_files=None, target_size=C.IMAGE_SIZE[0], random_mask=False, augment=False)
    b = ds[len(ds) // 2]
    xk = torch.as_tensor(b['data'])[None].float().to(dev)
    xi = torch.as_tensor(b['data_img'])[None].float().to(dev)
    ref = torch.as_tensor(b['label'])[None].float().to(dev)
    bm = torch.as_tensor(b['brain_mask'])[None].float().to(dev)
    s = R.per_sample_scale(xi)
    xk, xi = R.normalize_inputs(xk, xi, s)
    ref = ref / s
    m.eval()
    with torch.no_grad():
        in_h = xk.reshape([xk.size(0), m.num_in_x, -1]).float()
        yh_f, _ = m.gru_h(in_h)
        with reference_path():
            yh_r, _ = m.gru_h(in_h)
    res['real_h'] = dict(out=rel(yh_f, yh_r), out_rms=float(yh_r.float().pow(2).mean().sqrt()))
    ok &= res['real_h']['out'] < 1e-3
    del in_h, yh_f, yh_r
    m.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.amp.autocast('cuda'):
        so = m._seq(xk)
        out = m(xi, xk)
    loss = ((out.float() - ref).abs() * bm).sum() / bm.sum().clamp(min=1)
    loss.backward()
    torch.cuda.synchronize()
    gs = {n: q.grad for n, q in m.named_parameters() if q.grad is not None}
    finite = bool(torch.isfinite(out).all() and torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in gs.values())
    seq_g = [g.float().norm() for n, g in gs.items() if n.startswith(('gru_h.', 'gru_v.'))]
    res['real_fwd_bwd'] = dict(finite=finite, seq_out_dtype=str(so.dtype), seq_out_rms=float(so.float().pow(2).mean().sqrt()),
                               loss=float(loss), seq_grad_norm=float(torch.stack(seq_g).norm()) if seq_g else None,
                               seconds=round(time.time() - t0, 2), peak_gb_bs1=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    sgn = res['real_fwd_bwd']['seq_grad_norm']
    ok &= finite and so.dtype == torch.float32 and sgn is not None and sgn > 0   # 시퀀스 모듈까지 기울기가 흘러야 함
    res['ok'] = bool(ok)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print('PREFLIGHT', 'OK' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
