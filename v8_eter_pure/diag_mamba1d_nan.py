"""Mamba-ETER 50 epoch 런(pod2)의 비유한 손실 진단 (2026-10-12, 읽기 전용 — 학습과 무관).

배경: pod2 런이 epoch 1 batch ≈2,000 부터 손실 NaN(NaN-retry 해도 같음 → 결정적)·손실 급등(3~22)·AMP 배율 1024 를 보였다.
ETER-net·SS2D·U-Net 단독 런은 NaN-skip 0. 어디서 비유한이 생기는지 가른다:
  (a) Mamba 단계(fp32) 안 — h/v 단 출력 자체가 비유한
  (b) Mamba 출력이 커서 fp16 U-Net 에서 넘침 — 시퀀스 출력 |max| > 65504 또는 U-Net 을 fp32 로 돌리면 유한
  (c) U-Net 자체(fp16) — 시퀀스 출력은 정상인데 fp16 U-Net 출력만 비유한
입력: --ckpt (last.pt 의 'model' 또는 state_dict), log.txt 의 NaN-skip·GRAD-retry 배치 번호(epoch 1).
epoch 1 의 배치 구성은 트레이너와 같은 방법으로 다시 만든다: generator(SEED+1000×epoch) 에서 DataLoader 의 base_seed 한 번
→ RandomSampler randperm. (마스크 offset·flip 은 워커 rng 라 정확히 같지는 않다 — 진단에는 충분.)
같은 크기 묶음의 정상 배치도 대조로 본다. 출력 json.
"""
import os
import re
import sys
import json
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [_HERE, os.path.join(_ROOT, 'models', 'pure_eternet'), os.path.join(_ROOT, 'models', 'hybrid_eternet'),
                os.path.join(_ROOT, 'dataloaders')]
import smoke_recipe_fix as R                                                    # noqa: E402
from u_pure_eternet_mamba1d import PureETER_Mamba1D                             # noqa: E402
from dataloader_h5_v5 import FastMRI_H5_Dataloader                              # noqa: E402

C = R.C
FP16_MAX = 65504.0


def stat(t):
    t = t.detach().float()
    fin = torch.isfinite(t)
    tf = t[fin]
    return dict(finite=bool(fin.all()), n_nonfinite=int((~fin).sum()),
                absmax=float(tf.abs().max()) if tf.numel() else None, rms=float(tf.pow(2).mean().sqrt()) if tf.numel() else None)


@torch.no_grad()
def probe_sample(m, b, dev, lam):
    xk = torch.as_tensor(b['data'])[None].float().to(dev)
    xi = torch.as_tensor(b['data_img'])[None].float().to(dev)
    ref = torch.as_tensor(b['label'])[None].float().to(dev)
    bm = torch.as_tensor(b['brain_mask'])[None].float().to(dev)
    s = R.per_sample_scale(xi)
    xk, xi = R.normalize_inputs(xk, xi, s)
    ref = ref / s
    out = dict(scale_s=float(s.flatten()[0]), in_ksp=stat(xk), in_img=stat(xi))
    in_h = xk.reshape([xk.size(0), m.num_in_x, -1]).float()
    with torch.amp.autocast('cuda'):
        oh, _ = m.gru_h(in_h)
        so = m._seq(xk)
    out['h_out'] = stat(oh)
    out['seq_out'] = stat(so)
    out['seq_out_over_fp16'] = bool(out['seq_out']['absmax'] is not None and out['seq_out']['absmax'] > FP16_MAX)
    cat = torch.cat((so.float(), xi), dim=1)
    for name, amp in (('unet_fp16', True), ('unet_fp32', False)):
        with torch.amp.autocast('cuda', enabled=amp):
            y = m.unet(cat)
        yf = y.float()
        l1 = ((yf - ref).abs() * bm).sum() / bm.sum().clamp(min=1)
        out[name] = dict(out=stat(yf), l1=float(l1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--log', required=True, help='런 log.txt (epoch 1 의 NaN-skip·GRAD-retry 배치 번호)')
    p.add_argument('--data-train', default=os.path.join(_ROOT, 'fastMRI_data', 'multicoil_train'))
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--max-bad', type=int, default=12, help='진단할 비유한 배치 수 (앞에서부터)')
    p.add_argument('--n-ctrl', type=int, default=6, help='대조 정상 배치 수')
    p.add_argument('--mem-frac', type=float, default=0.0, help='GPU 메모리 상한 비율 (0 = 없음)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    dev = torch.device('cuda')
    if a.mem_frac > 0:                     # 학습 중인 GPU 를 같이 쓸 때 이 진단만 OOM 나도록 상한
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    txt = open(a.log).read()
    bad = sorted({int(x) for x in re.findall(r'NaN-skip ep1 batch(\d+)', txt)})
    gr = sorted({int(x) for x in re.findall(r'GRAD-retry ep1 batch(\d+)', txt)} - {0, 16})
    print(f'NaN-skip 배치 {len(bad)} 개 (첫 {bad[:5]}), GRAD-retry(초기 배율 탐색 제외) {gr}')
    targets = [('nan', i) for i in bad[-a.max_bad:]] + [('gradretry', i) for i in gr]
    rng = np.random.default_rng(0)
    pool = sorted(set(range(max(bad + [0]) + 1)) - set(bad) - set(gr))
    ctrl = sorted(rng.choice([i for i in pool if i > 1500], size=a.n_ctrl, replace=False).tolist())
    targets += [('ctrl', i) for i in ctrl]

    ds = FastMRI_H5_Dataloader(a.data_train, num_files=None, target_size=C.IMAGE_SIZE[0],
                               augment=C.TRAIN_AUGMENT, augment_flip_p=C.TRAIN_AUGMENT_FLIP_P)
    n = len(ds)
    g = torch.Generator()
    g.manual_seed(a.seed + 1000 * 0)                       # epoch 1 (트레이너 _g.manual_seed(SEED + 1000*epoch))
    torch.empty((), dtype=torch.int64).random_(generator=g)  # DataLoader iterator 의 base_seed 한 번
    perm = torch.randperm(n, generator=g).tolist()

    m = PureETER_Mamba1D(dim=C.IMAGE_SIZE[0], n_coil=C.N_COIL, n_hidden_1=C.N_HIDDEN_LRNN_1, n_hidden_2=C.N_HIDDEN_LRNN_2,
                         unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF, use_dc=False, fp32_stages=('h', 'v'), verbose=False)
    sd = torch.load(a.ckpt, map_location='cpu', mmap=True)
    if isinstance(sd, dict) and 'model' in sd:
        print(f"ckpt epoch {sd.get('epoch')} global_step {sd.get('global_step')}")
        sd = sd['model']
    m.load_state_dict(sd)
    del sd
    m = m.to(dev).eval()

    res = dict(ckpt=a.ckpt, n_train=n, bad_batches=bad, gradretry_batches=gr, ctrl_batches=ctrl, samples=[])
    for kind, bi in targets:
        for j, idx in enumerate(perm[bi * a.bs:(bi + 1) * a.bs]):
            r = probe_sample(m, ds[idx], dev, None)
            r.update(kind=kind, batch=bi, pos=j, idx=idx)
            res['samples'].append(r)
            print(f"{kind:9s} b{bi:5d} idx{idx:6d} s={r['scale_s']:.3g} in_ksp|max|={r['in_ksp']['absmax']:.3g} "
                  f"h|max|={r['h_out']['absmax']} seq|max|={r['seq_out']['absmax']} fin={r['seq_out']['finite']} "
                  f"unet16 fin={r['unet_fp16']['out']['finite']} l1={r['unet_fp16']['l1']:.4g} | "
                  f"unet32 fin={r['unet_fp32']['out']['finite']} l1={r['unet_fp32']['l1']:.4g}", flush=True)
    by = {}
    for r in res['samples']:
        k = r['kind']
        d = by.setdefault(k, dict(n=0, seq_nonfinite=0, seq_over_fp16=0, unet16_nonfinite=0, unet32_nonfinite=0,
                                  seq_absmax=[], l1_16=[], l1_32=[]))
        d['n'] += 1
        d['seq_nonfinite'] += int(not r['seq_out']['finite'])
        d['seq_over_fp16'] += int(r['seq_out_over_fp16'])
        d['unet16_nonfinite'] += int(not r['unet_fp16']['out']['finite'])
        d['unet32_nonfinite'] += int(not r['unet_fp32']['out']['finite'])
        if r['seq_out']['absmax'] is not None:
            d['seq_absmax'].append(r['seq_out']['absmax'])
        d['l1_16'].append(r['unet_fp16']['l1']); d['l1_32'].append(r['unet_fp32']['l1'])
    for k, d in by.items():
        sa = np.array(d.pop('seq_absmax') or [np.nan]); l16 = np.array(d.pop('l1_16')); l32 = np.array(d.pop('l1_32'))
        d.update(seq_absmax_median=float(np.nanmedian(sa)), seq_absmax_max=float(np.nanmax(sa)),
                 l1_16_median=float(np.nanmedian(l16)), l1_32_median=float(np.nanmedian(l32)))
    res['summary'] = by
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(by, indent=1))


if __name__ == '__main__':
    main()
