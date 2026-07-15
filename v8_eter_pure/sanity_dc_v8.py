"""v8 DC-arm 사전 검증 (GPU 무접촉, CPU-only) — 2026-07-08 16/16 PASS.

학습 중인 GPU 를 방해하지 않고 DC 경로의 수학/정합성을 검증한다.
SS2D+DC 자동 시작 전(GRU+DC 완주 무렵) 재실행 권장:
    CUDA_VISIBLE_DEVICES="" python v8_eter_pure/sanity_dc_v8.py

검증 항목:
  [A] 4변형(GRU/SS2D × DC/noDC) 생성자 — import/shape 계약 (SS2D forward 는 CUDA 전용이라 제외)
  [B] 실 데이터 1샘플 스케일/shape 정합 (ksp×100 ↔ img, Σ|sens|², mask/ACS 정렬)
  [C] DCBlock 물리 항등식 (CPU, 실 데이터):
      C1. alpha=0        → out == x   (정확 항등식)
      C2. mask=0         → out == x   (정확 항등식)
      C3. mask=1, α=1    → out == Σ ifft(k_meas·100)·sens*  (해석해)
      C4. α=1 zero-filled 시작 → sampled-line k-space residual 감소
  [D] GRU+DC 풀 forward (CPU, 실 데이터 B=1): 출력 shape/finite/양수 + loss 계산 가능
"""
import os, sys
import numpy as np
import torch

torch.set_num_threads(4)                      # 학습 worker 방해 금지

ROOT = '/home/snorlax/shared/fastmri_ViT_with_eternet'
for p in ['v8_eter_pure/configs', 'dataloaders', 'models/pure_eternet',
          'models/hybrid_eternet', 'models/mamba_eternet', 'tools']:
    sys.path.append(os.path.join(ROOT, p))

results = []
def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

# ── [A] 4변형 생성자 ──────────────────────────────────────
print("\n[A] 4변형 생성자 (CPU)")
from u_pure_eternet_gru import PureETER_GRU
from u_pure_eternet_ss2d import PureETER_SS2D
kw = dict(n_coil=16, n_hidden_2=10, unet_depth=5, unet_wf=6)
models = {}
for use_dc in (False, True):
    m = PureETER_GRU(dim=384, n_hidden_1=10, use_dc=use_dc,
                     dc_k_scale_ratio=100.0, dc_init_alpha=1.0, **kw)
    models[('gru', use_dc)] = m
    n = sum(p.numel() for p in m.parameters()) / 1e6
    check(f"GRU  use_dc={use_dc} ctor", True, f"{n:.1f}M params")
for use_dc in (False, True):
    m = PureETER_SS2D(ss2d_d_inner=128, ss2d_d_state=16, use_dc=use_dc,
                      dc_k_scale_ratio=100.0, dc_init_alpha=1.0, **kw)
    models[('ss2d', use_dc)] = m
    n = sum(p.numel() for p in m.parameters()) / 1e6
    check(f"SS2D use_dc={use_dc} ctor", True, f"{n:.1f}M params")

# DC/noDC 쌍의 시퀀스모델+UNet backbone 파라미터 수 검증: DC 는 UNet 마지막 conv 출력채널만 1→2
p_gru_nodc = sum(p.numel() for p in models[('gru', False)].parameters())
p_gru_dc   = sum(p.numel() for p in models[('gru', True)].parameters())
# UNet_choh_skip.last = Conv2d(prev(2**wf=64) + n_hidden*2(52) = 116, n_classes, 1)
# → 추가 출력채널 1개 = 116w+1b = 117, + DC alpha 1개 = 118
diff = p_gru_dc - p_gru_nodc
check("GRU DC−noDC param 차 = last-conv 1ch(117) + alpha(1)", diff == 116 + 1 + 1, f"diff={diff}")

# ── [B] 실 데이터 1샘플 ──────────────────────────────────
print("\n[B] 실 데이터 스케일/정합 (multicoil_val 1파일)")
from dataloader_h5_v5 import FastMRI_H5_Dataloader, ifft2c, fft2c, build_r4_mask
ds = FastMRI_H5_Dataloader(os.path.join(ROOT, 'fastMRI_data/multicoil_val'),
                           num_files=1, target_size=384, random_mask=False, augment=False)
s = ds[len(ds) // 2]                          # 중앙 슬라이스
data, dimg, label = s['data'], s['data_img'], s['label']
mask, sens, bmask = s['mask'], s['sens'], s['brain_mask']
check("shapes", data.shape == (32, 384, 384) and dimg.shape == (32, 384, 384)
      and label.shape == (1, 384, 384) and mask.shape == (1, 384, 384)
      and sens.shape == (32, 384, 384),
      f"data{data.shape} img{dimg.shape} label{label.shape} mask{mask.shape} sens{sens.shape}")

# ksp×100 을 iFFT 하면 data_img 와 일치해야 (같은 마스킹, 스케일비 100)
ksp_c = (data[0::2] + 1j * data[1::2]).astype(np.complex64)
img_c = (dimg[0::2] + 1j * dimg[1::2]).astype(np.complex64)
img_from_ksp = ifft2c(ksp_c * 100.0)
rel = np.linalg.norm(img_from_ksp - img_c) / max(np.linalg.norm(img_c), 1e-12)
check("iFFT(ksp×100) == data_img (스케일비=DC_K_SCALE_RATIO)", rel < 1e-4, f"rel_err={rel:.2e}")

# Σ|sens|² ∈ [0,1], brain 영역에선 ≈1
sens_c = (sens[0::2] + 1j * sens[1::2]).astype(np.complex64)
ssq = np.sum(np.abs(sens_c) ** 2, axis=0)
in_brain = ssq[bmask[0] > 0.5]
check("Σ|sens|² ≈ 1 (brain 내)", np.allclose(in_brain, 1.0, atol=1e-3),
      f"min={in_brain.min():.4f} max={in_brain.max():.4f}")

# mask 의 ACS(중앙 연속밴드)가 sens 추출 윈도우와 동일한지
nlf = int(round(384 * 0.08)); pad = (384 - nlf + 1) // 2
m1d = mask[0, 0]                              # (W,)
check("mask ACS 밴드 = sens 추출 윈도우", bool(m1d[pad:pad + nlf].all()),
      f"cols[{pad}:{pad+nlf}] 전부 sampled={bool(m1d[pad:pad+nlf].all())}")
check("mask R≈4 (배경 undersampling)", 0.2 < m1d.mean() < 0.4, f"duty={m1d.mean():.3f}")

# ── [C] DCBlock 항등식 (CPU) ─────────────────────────────
print("\n[C] DCBlock 물리 항등식")
from u_choh_model_SS2D_ViT_v4 import DCBlock
dc = DCBlock(k_scale_ratio=100.0, init_alpha=1.0)

t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0)
k_t, s_t, m_t = t(data), t(sens), t(mask)

# 시작점 x0 = sens-combine(zero-filled recon) — (B,2,H,W)
x0_c = np.sum(img_c * np.conj(sens_c), axis=0)
x0_t = torch.stack([torch.from_numpy(x0_c.real), torch.from_numpy(x0_c.imag)], dim=0).unsqueeze(0).float()

# C1: alpha=0 → 항등
with torch.no_grad():
    dc.alpha.fill_(0.0)
    out = dc(x0_t, k_t, m_t, s_t)
rel = (out - x0_t).norm() / x0_t.norm()
check("C1 α=0 → out==x (항등)", rel < 1e-4, f"rel={rel:.2e}")

# C2: mask=0 → 항등 (α=1)
with torch.no_grad():
    dc.alpha.fill_(1.0)
    out = dc(x0_t, k_t, torch.zeros_like(m_t), s_t)
rel = (out - x0_t).norm() / x0_t.norm()
check("C2 mask=0 → out==x (항등)", rel < 1e-4, f"rel={rel:.2e}")

# C3: mask=1, α=1 → out == Σ ifft(k_meas·100)·sens* (해석해; 입력 x 무관)
with torch.no_grad():
    x_rand = torch.randn_like(x0_t) * x0_t.abs().max()
    out = dc(x_rand, k_t, torch.ones_like(m_t), s_t)
expect_c = np.sum(ifft2c(ksp_c * 100.0) * np.conj(sens_c), axis=0)
expect = torch.stack([torch.from_numpy(expect_c.real), torch.from_numpy(expect_c.imag)], 0).unsqueeze(0).float()
rel = (out - expect).norm() / expect.norm()
check("C3 mask=1,α=1 → 해석해 일치", rel < 1e-3, f"rel={rel:.2e}")

# C4: α=1, zero-filled 시작 → sampled-line k-residual 감소
def k_residual(x_t):
    xc = torch.complex(x_t[:, 0], x_t[:, 1]).unsqueeze(1)
    sc = torch.complex(s_t[:, 0::2], s_t[:, 1::2])
    kp = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(sc * xc, dim=(-2, -1)),
                                           norm='ortho'), dim=(-2, -1))
    km = torch.complex(k_t[:, 0::2], k_t[:, 1::2]) * 100.0
    return ((kp - km) * m_t).norm().item()
with torch.no_grad():
    dc.alpha.fill_(1.0)
    out = dc(x0_t, k_t, m_t, s_t)
r0, r1 = k_residual(x0_t), k_residual(out)
check("C4 DC 후 sampled-line residual 감소", r1 < r0, f"{r0:.1f} → {r1:.1f} ({100*(1-r1/r0):.1f}%↓)")

# ── [D] GRU+DC 풀 forward (CPU) ──────────────────────────
print("\n[D] GRU+DC 풀 forward (CPU, 668M — 수십 초)")
m = models[('gru', True)].eval()
with torch.no_grad():
    out = m(t(dimg), k_t, m_t, s_t)
lbl_t = t(label); bm_t = t(bmask)
finite = torch.isfinite(out).all().item()
l1 = ((out - lbl_t).abs() * bm_t).sum() / bm_t.sum()
check("D1 출력 (1,1,384,384)/finite/양수", out.shape == (1, 1, 384, 384) and finite
      and (out.min() >= 0).item(), f"range=[{out.min():.3f},{out.max():.3f}]")
check("D2 masked L1 계산 가능/finite", torch.isfinite(l1).item(), f"L1={l1.item():.2f} (미학습 모델)")

# ── 요약 ──
print("\n" + "=" * 60)
fails = [n for n, ok, _ in results if not ok]
print(f"총 {len(results)}건 중 PASS {len(results)-len(fails)} / FAIL {len(fails)}")
for n in fails:
    print(f"  ✗ {n}")
print("VERIFY", "OK" if not fails else "FAILED")
