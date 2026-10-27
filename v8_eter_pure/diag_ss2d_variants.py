"""
SS2D 인과 진단 (2026-10-11) — U-Net 이 SS2D 출력을 쓰지 않는 이유를 입력 영역·norm_in·표현 용량으로 나눠 시험한다.

배경:
  - v8fix(DL 레시피) 50 epoch 런(pod3)에서 SS2D 출력의 0 대체 하락이 ep6 이후 0(또는 음수), shuffle_cross 하락은 모든 epoch 에 ≈0.
    같은 레시피의 ETER-net(GRU)은 shuffle_cross 하락 +0.0020(45/64, p=0.0016)으로 '사용' 판정.
  - 체크포인트 사후 분석(results/ss2d_diag_2026-10-11/forensics.md): U-Net 이 시퀀스 채널을 끈 것이 아니라 SS2D 가 자기 출력을
    줄였다(출력 RMS 0.351 → 0.0043, 주로 out_proj.weight 크기 축소, 방향 유지). 손실 기울기가 SS2D 출력을 '쓸모없는 입력'으로
    보고 줄인 것과 맞는 결과다.
  - 10-08 가설(docs/v8_training_collapse_audit_2026-10-07.md §(8)): (a) 행+열 십자형 수용 영역은 2D IFFT 를 표현할 수 없어 출력이
    k-space 격자에 머문다, (b) norm_in(화소별 32채널 LayerNorm)이 k-space 크기 정보를 지우고 비표본 화소(≈69%, 정확히 0)를
    상수로 만든다, (c) U-Net 은 SS2D 출력을 bias 평면으로만 쓴다.

실험 조건 (--variant; 각각 별도 프로세스, 같은 DL 레시피·SEED·데이터 순서·검증 64 슬라이스·기능 시험):
  ctrl    V0 원 SS2D (k-space 입력) — smoke_recipe_fix 'ss2d:DL' job 의 재현(그 job 의 val SSIM 0.7772, shuffle +0.0001 (40/64)).
  img     V1 같은 SS2D 모듈, 입력만 코일별 centered 2D IFFT(ifft2c) 한 영상 영역 복소 코일 영상(같은 32채널 real/imag 교번 배치).
          로더 관례(dataloader_h5_v5.py:30-34 ifftshift→ifft2(ortho)→fftshift, _pack_complex_to_real 의 [0::2]=real·[1::2]=imag)와
          같고, 정규화 뒤 k-space(×100/s)에 적용하면 정규화된 zero-filled 영상(data_img/s)과 일치한다(상대 오차 ≈1.5e-7 실측,
          실행 때마다 probe 슬라이스에서 다시 확인해 기록). 입력 영역(수용 영역) 가설 (a) 시험.
  nonorm  V2 k-space 입력 그대로, ss2d.norm_in 을 인스턴스 수준에서 nn.Identity 로 교체(--nonorm-gamma ≠ 1 이면 고정 스칼라 곱).
          정규화 뒤 k-space 크기(train 32 슬라이스 실측): RMS 중앙값 0.050(0.035~0.066), 화소별 32채널 벡터 norm 은 표본 화소 중앙값
          0.07·99백분위수 1.3·최대 18~60(k-space 중심), 화소의 69% 는 정확히 0. ETER-net 의 GRU 가 받는 입력과 같은 크기 영역이라
          기본은 스칼라 없이(γ=1, 순수 Identity) 둔다. 초기값에서 probe 8 슬라이스로 잰 SSM 출력 최댓값(fp16 로 되돌리는 값):
          γ=1 7.2 · γ=5 543 · γ=10 2.5e4 · γ=20 inf(오버플로) — RMS≈1 이 되는 γ=20 은 쓸 수 없다. 대신 입력 의존성
          ‖f(x)−f(0)‖/‖f(0)‖ 가 γ=1 0.028 · γ=5 0.086 으로 원 SS2D(0.42)보다 작아(중심 화소가 지배) γ=5 를 민감도 확인용 추가 조건
          (출력 이름 v2_nonorm_g5)으로 둔다. norm_in 가설 (b) 시험.
  cap     U-Net 없는 표현 용량 시험(짧음, 먼저 실행): 모듈 + 1×1 선형 readout 을 정규화 단위 MSE 로 학습해 정규화된 zero-filled
          영상(target 'rss' = 코일 결합 크기 RSS 1채널, 'coil' = 32채널 real/imag 코일 영상)을 x_ksp 에서 회귀한다.
          모듈: ss2d_ksp(원 SS2D, k-space 입력) / ss2d_img(같은 SS2D, ifft2c 입력 — 영역 변환이 필요 없는 항등에 가까운 과제) /
          ss2d_ksp_nonorm·ss2d_img_nonorm(norm_in 제거, γ=1) / sep(분리형 dense 기준선: 코일 공유 복소 행렬 A_r(H×H)·A_c(W×W),
          y_c = A_r z_c A_cᵀ — ifft2c 를 정확히 표현할 수 있는 전역 수용 영역 선형 모듈. target 'rss' 에서는 고정 비선형 RSS 를
          거친 뒤 1×1 readout). 고정 예산(--cap-steps, 기본 1500 step) 뒤 검증 64 슬라이스에서 R²·NMSE(전체 FOV·brain mask).
          학습 설정: 레시피 DL 과 같은 AdamW(분리형 wd 3e-5 + no-decay)·clip 1e3·fp16 autocast·cosine(T_max = cap-steps)·BS 8·
          정규화 단위 MSE. readout 은 시작 때 초기 features 의 최소제곱 해로 맞추고(1×1 readout 의 느린 학습이 결과를 가리지 않게,
          모든 모듈 공통), 끝에 고정 train 16 슬라이스로 다시 맞춘 readout(refit)으로도 평가 — refit R² = 학습된 features 의 선형
          해독 가능성. sep 만 LR 1e-3(300~400 step 보정: 2e-4 는 예산 안에 행렬이 거의 움직이지 않음, 1e-3 은 400 step 에 coil
          R² 0.95, 3e-3 은 0.998). SS2D 끼리는 같은 레시피 LR 2e-4 로 비교한다.
          'coil' target 은 SS2D 출력이 20채널이라 1×1 readout 의 상한이 rank-20 → 같은 검증 자료의 채널 PCA 상한도 함께 기록.

재사용 (smoke_recipe_fix.py 를 모듈로 import — 그 파일·ss2d.py·트레이너는 무수정):
  - V0/V1/V2 는 R.run_job('ss2d', 'DL', …) 를 그대로 호출한다. 바꾸는 것은 R.build_model 이름 하나(원 build_model 로 만든 뒤
    인스턴스 수준 변형만 적용: V1 = model.ss2d 의 forward_pre_hook 으로 입력 변환, V2 = model.ss2d.norm_in 교체). 따라서 데이터
    (같은 로더·augment·마스크 offset·SEED 경로), 손실(masked L1 + λ/s̄·(1−SSIM)), AdamW 분리형 + no-decay, clip 1e3, fp16 autocast +
    GradScaler, NaN-skip, cosine(T_max = steps), 기록(처음 500 step 20 마다·이후 100 마다: loss·L1·SSIM·그룹별 grad norm·relΔ·probe),
    검증(R.run_validation: normal / deep4 / deep34 / zero / shuffle_cross / shuffle_adj, R.drop_stats·R.used_verdict)이 스모크와 같다.
    V1 의 state_dict 는 원 모델과 같다(hook 은 파라미터가 없다). V2 는 norm_in.weight/bias 두 텐서만 빠진다.
  - 학습 뒤 추가 분석(초기값과 끝 상태 모두, 같은 64 슬라이스): 시퀀스 출력 에너지 분해(전역 상수 / 고정 패턴 / 슬라이스 상수 /
    슬라이스 고유 공간 성분 — forensics 와 같은 정의), 대체 시험(zero / slice_mean_plane / fixed_pattern / no_slice_specific_spatial /
    shuffle_cross — R.drop_stats·R.used_verdict), U-Net 첫 conv 응답비, 출력이 영상 격자인지 k-space 격자인지(정규화 zero-filled
    RSS 와 log|k| 에 대한 20채널 선형 회귀 R², 채널별 최대 |상관|), out_proj·merge·in_proj·U-Net 첫 conv 시퀀스 입력 가중치의
    norm 비(끝/초기).

출력 (--out-dir, 기본 results/ss2d_diag_2026-10-11/; 상대 경로는 저장소 루트 기준):
  <vname>/<vname>.jsonl       R.run_job 의 step 기록(meta 1행 + step 행) — vname ∈ v0_ctrl, v1_img, v2_nonorm
  <vname>/<vname>_end.pt      끝 state_dict
  <vname>_final.json          진단 meta + R.run_job summary + 추가 분석 + 상태
  cap/<model>__<target>.jsonl CAP job 별 step 기록, cap_final.json = CAP 전체 결과
  diag_summary.md             위 final json 들로 매번 다시 만든다.

GPU: R.gpu0_precheck(CUDA_VISIBLE_DEVICES="0" 이고 GPU0 사용 메모리 ≤ 2 GB 일 때만 실행). 종료 코드는 스모크와 같다
  (0 정상 / 2 거부 / 3 실패 / 4 CUDA 사용 불가 — 러너가 재시도).

실행 예:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python v8_eter_pure/diag_ss2d_variants.py --variant img
  # 스모크(≤30 step, 결과는 scratch 로):
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/diag_ss2d_variants.py --variant nonorm --steps 30 --log-every 10 \\
      --log-dense-every 0 --val-slices 4 --post-slices 4 --out-dir /tmp/diag_smoke
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/diag_ss2d_variants.py --variant cap --cap-steps 30 --val-slices 4 --out-dir /tmp/diag_smoke
"""

import os
import sys
import gc
import json
import math
import time
import argparse
import datetime
import traceback
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import smoke_recipe_fix as R            # noqa: E402 — 레시피·데이터·검증 단일 출처 (sys.path 에 models/* 추가)
import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import torch.nn as nn                   # noqa: E402
import torch.nn.functional as F         # noqa: E402

C = R.C
slice_metrics = R.slice_metrics

DIAG_VERSION = '2026-10-11a'
SEQ_CH, IN_CH = R.SEQ_CH, R.IN_CH       # 20, 52
VNAME = OrderedDict([('ctrl', 'v0_ctrl'), ('img', 'v1_img'), ('nonorm', 'v2_nonorm'), ('cap', 'cap')])
VDESC = {
    'ctrl': 'V0 원 SS2D (k-space 입력) — smoke ss2d:DL 재현',
    'img': 'V1 같은 SS2D, 입력 = 코일별 ifft2c(x_ksp) (영상 영역 복소 코일 영상, 32ch real/imag 교번)',
    'nonorm': 'V2 k-space 입력, ss2d.norm_in = Identity (γ≠1 이면 고정 스칼라 곱)',
    'cap': 'U-Net 없는 표현 용량 시험 (모듈 + 1×1 readout, 정규화 zero-filled 영상 회귀)',
}
# sep 은 LR 1e-3: 레시피 LR 2e-4 에서는 384×384 복소 행렬(원소 크기 ≈0.05)이 예산 안에 거의 움직이지 않는다
# (보정 시험: readout 최소제곱 초기화 뒤 400 step 에 coil R² 1e-3 → 0.95, 3e-3 → 0.998). SS2D 끼리는 같은 레시피 LR 로 비교.
# SS2D job 은 ≈1.07 s/step 이라 1500 step 당 ≈28 분 — 우선순위 순서(k-space vs 영상 영역 → norm_in → 선형 target).
CAP_DEFAULT_JOBS = ('sep:rss@1e-3,sep:coil@1e-3,ss2d_ksp:rss,ss2d_img:rss,ss2d_img_nonorm:rss,ss2d_ksp_nonorm:rss,'
                    'ss2d_ksp:coil')
CAP_KINDS = ('sep', 'ss2d_ksp', 'ss2d_img', 'ss2d_ksp_nonorm', 'ss2d_img_nonorm')
CAP_TARGETS = ('rss', 'coil')


# ------------------------------------------------------------------ 변환·변형
def ifft2c_packed(x):
    """(B, 32, H, W) real/imag 교번 k-space → 같은 배치의 코일별 centered 2D IFFT (dataloader_h5_v5.py:30-34 와 같은 순서, ortho).
    fp32 로 계산(384 는 2 의 거듭제곱이 아니라 cuFFT half 미지원)."""
    xf = x.float()
    z = torch.complex(xf[:, 0::2], xf[:, 1::2])
    z = torch.fft.ifftshift(z, dim=(-2, -1))
    z = torch.fft.ifft2(z, dim=(-2, -1), norm='ortho')
    z = torch.fft.fftshift(z, dim=(-2, -1))
    B, Cc, H, W = z.shape
    return torch.stack((z.real, z.imag), dim=2).reshape(B, 2 * Cc, H, W)


def _ifft_pre_hook(mod, args):
    return (ifft2c_packed(args[0]),) + tuple(args[1:])


class ScaleOnly(nn.Module):
    """norm_in 대체용 고정 스칼라 곱 (학습 파라미터 없음, buffer)."""

    def __init__(self, gamma):
        super().__init__()
        self.register_buffer('gamma', torch.tensor(float(gamma)), persistent=False)

    def forward(self, x):
        return x * self.gamma


def make_norm_replacement(gamma):
    return nn.Identity() if float(gamma) == 1.0 else ScaleOnly(gamma)


def apply_variant(model, variant, gamma):
    """인스턴스 수준 변형 (ss2d.py·wrapper 무수정)."""
    if variant == 'img':
        model._diag_hook = model.ss2d.register_forward_pre_hook(_ifft_pre_hook)
    elif variant == 'nonorm':
        model.ss2d.norm_in = make_norm_replacement(gamma)
    elif variant != 'ctrl':
        raise ValueError(variant)
    model._diag_variant = variant
    return model


_ORIG_BUILD = R.build_model


def make_builder(variant, gamma):
    def build(seq, device):
        m = _ORIG_BUILD(seq, device)
        if seq == 'ss2d':
            apply_variant(m, variant, gamma)
        return m
    return build


# ------------------------------------------------------------------ 공통
def vname_of(args):
    """출력 이름. nonorm 은 γ≠1 이면 접미사 (예: v2_nonorm_g5)."""
    v = VNAME[args.variant]
    if args.variant == 'nonorm' and float(args.nonorm_gamma) != 1.0:
        v += f'_g{args.nonorm_gamma:g}'
    return v


def _rms(t):
    t = t.detach().float()
    return float(t.pow(2).mean().sqrt())


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def prepare_data(args, device):
    """R.main 의 준비 순서와 같다 (val 64 슬라이스·probe 2 슬라이스·DL s̄)."""
    train_ds = R.build_train_dataset()
    val_ds = R.build_val_dataset()
    for ds in (train_ds, val_ds):
        assert abs(ds.val_amp_X_img / ds.val_amp_X_ksp - R.K_RATIO) < 1e-9, '로더 배율이 바뀌었다 — K_RATIO 재확인'
    val_indices = R.even_indices(len(val_ds), args.val_slices)
    probe_indices = [val_indices[0], val_indices[len(val_indices) // 2]] if len(val_indices) >= 2 else \
        [val_indices[0], (val_indices[0] + len(val_ds) // 2) % len(val_ds)]
    probe = R.load_probe(val_ds, probe_indices)
    return train_ds, val_ds, val_indices, probe_indices, probe


@torch.no_grad()
def check_ifft_convention(probe, device):
    """정규화 후 ifft2c_packed(x_ksp) 와 정규화 zero-filled 영상(data_img/s)의 상대 오차 (probe 슬라이스별)."""
    out = []
    for b in probe:
        xk, xi = b['data'].to(device), b['data_img'].to(device)
        xk, xi = R.normalize_inputs(xk, xi, R.per_sample_scale(xi))
        rec = ifft2c_packed(xk)
        out.append(float((rec - xi).norm() / xi.norm().clamp(min=1e-30)))
    return out


@torch.no_grad()
def input_scale_stats(probe, device, gamma):
    """V2 기록용: 정규화 후 k-space 크기 (γ 곱 전/후)."""
    rows = []
    for b in probe:
        xk, xi = b['data'].to(device), b['data_img'].to(device)
        xk, _ = R.normalize_inputs(xk, xi, R.per_sample_scale(xi))
        pn = xk.float().pow(2).sum(1).sqrt().flatten()
        nz = pn[pn > 0]
        rows.append(dict(rms=_rms(xk), pix_norm_median_nonzero=float(nz.median()),
                         pix_norm_p99_nonzero=float(torch.quantile(nz[::3], 0.99)), pix_norm_max=float(pn.max()),
                         frac_zero_pix=float((pn == 0).float().mean())))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg['gamma'] = float(gamma)
    agg['rms_after_gamma'] = agg['rms'] * gamma
    agg['pix_norm_max_after_gamma'] = agg['pix_norm_max'] * gamma
    return agg


# ------------------------------------------------------------------ 학습 뒤 추가 분석 (V0/V1/V2)
def _lstsq_r2(Y, X):
    """Y (P,) 를 X (P, k) + 절편으로 선형 회귀한 R² (fp64)."""
    X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)), dim=1)
    sol = torch.linalg.lstsq(X, Y[:, None]).solution
    res = Y - (X @ sol)[:, 0]
    sst = (Y - Y.mean()).pow(2).sum()
    return float(1.0 - res.pow(2).sum() / sst.clamp(min=1e-30))


def _max_abs_corr(Y, X):
    Yc = Y - Y.mean()
    Xc = X - X.mean(0, keepdim=True)
    num = (Xc * Yc[:, None]).sum(0)
    den = Xc.pow(2).sum(0).sqrt() * Yc.pow(2).sum().sqrt()
    c = num / den.clamp(min=1e-30)
    return float(c.abs().max())


@torch.no_grad()
def post_analysis(model, val_ds, indices, device, workers, sub=4):
    """시퀀스 출력 분해·대체 시험·첫 conv 응답비·영상/k-space 격자 지표. 정규화·autocast·지표는 run_validation 과 같다."""
    model.eval()
    t0 = time.time()
    n = len(indices)
    W1 = model.unet.down_path[0].block[0].weight.detach().float()          # (64, 52, 3, 3)
    so_store, base_ssim, grid = [], [], []
    for idx, b in R.iter_val(val_ds, indices, device, workers, True):
        x_ksp, x_img, ref, bm = b['data'], b['data_img'], b['label'], b['brain_mask']
        s = R.per_sample_scale(x_img)
        xk, xi = R.normalize_inputs(x_ksp, x_img, s)
        with torch.amp.autocast('cuda'):
            so = model.ss2d(xk)
            out = model.unet(torch.cat((so, xi), dim=1))
        base_ssim.append(slice_metrics(out.float() * s, ref, bm)['ssim'])
        sof = so.float()
        so_store.append(sof.cpu())
        # 격자 지표: 출력(20ch, 4×4 부분표본)이 zero-filled RSS(영상 격자) / log|k|(k-space 격자)를 얼마나 설명하나
        # log|k| 는 표본 화소(|k|>0)에서만 — 비표본 열(≈69%, 정확히 0)을 넣으면 고정 마스크 열 패턴만으로 R² 가 커진다.
        Xs = sof[0, :, ::sub, ::sub].reshape(SEQ_CH, -1).t().double()
        zf = xi.float().pow(2).sum(1).sqrt()[0, ::sub, ::sub].reshape(-1).double()
        kn = xk.float().pow(2).sum(1).sqrt()[0, ::sub, ::sub].reshape(-1).double()
        sm = kn > 0
        lk = torch.log(kn[sm])
        grid.append(dict(r2_zf_rss=_lstsq_r2(zf, Xs), r2_logk_sampled=_lstsq_r2(lk, Xs[sm]),
                         maxcorr_zf_rss=_max_abs_corr(zf, Xs), maxcorr_logk_sampled=_max_abs_corr(lk, Xs[sm])))
    S = torch.cat(so_store, 0).double()                                      # (n, 20, H, W)
    m_c = S.mean(dim=(0, 2, 3))
    xbar = S.mean(dim=0)
    mu_s = S.mean(dim=(2, 3))
    P = S.shape[2] * S.shape[3]
    E = float(S.pow(2).sum())
    G = float(n * P * (m_c ** 2).sum())
    A_ = float(n * ((xbar - m_c[:, None, None]) ** 2).sum())
    Bsl = float(P * ((mu_s - m_c[None]) ** 2).sum())
    Rsp = E - G - A_ - Bsl
    resid_rms = math.sqrt(max(Rsp, 0.0) / (n * SEQ_CH * P))
    decomp = OrderedDict(seq_out_rms=math.sqrt(E / (n * SEQ_CH * P)), slice_specific_spatial_rms=resid_rms,
                         frac_global_const=G / E, frac_fixed_spatial_pattern=A_ / E,
                         frac_slice_specific_bias=Bsl / E, frac_slice_specific_spatial=Rsp / E)
    del S
    m_c_d, xbar_d = m_c.float().to(device), xbar.float().to(device)
    names = ('zero', 'slice_mean_plane', 'fixed_pattern', 'no_slice_specific_spatial', 'shuffle_cross')
    tests = OrderedDict((k, []) for k in names)
    resp = dict(seq_over_img=[], slice_specific_over_img=[], img_rms=[], seq_rms=[])
    for i, (idx, b) in enumerate(R.iter_val(val_ds, indices, device, workers, True)):
        x_ksp, x_img, ref, bm = b['data'], b['data_img'], b['label'], b['brain_mask']
        s = R.per_sample_scale(x_img)
        _, xi = R.normalize_inputs(x_ksp, x_img, s)
        so = so_store[i].to(device)
        mus = so.mean(dim=(2, 3), keepdim=True)
        j = (i + n // 2) % n
        alts = OrderedDict(zero=torch.zeros_like(so), slice_mean_plane=mus.expand_as(so), fixed_pattern=xbar_d[None],
                           no_slice_specific_spatial=xbar_d[None] + mus - m_c_d[None, :, None, None],
                           shuffle_cross=so_store[j].to(device))
        for k, a in alts.items():
            with torch.amp.autocast('cuda'):
                o = model.unet(torch.cat((a.to(xi.dtype), xi), dim=1))
            tests[k].append(slice_metrics(o.float() * s, ref, bm)['ssim'])
        r_seq = F.conv2d(so, W1[:, :SEQ_CH], padding=1)
        r_img = F.conv2d(xi.float(), W1[:, SEQ_CH:], padding=1)
        resid = so - (xbar_d[None] + mus - m_c_d[None, :, None, None])
        r_res = F.conv2d(resid, W1[:, :SEQ_CH], padding=1)
        ri = _rms(r_img)
        resp['seq_over_img'].append(_rms(r_seq) / max(ri, 1e-30))
        resp['slice_specific_over_img'].append(_rms(r_res) / max(ri, 1e-30))
        resp['img_rms'].append(ri)
        resp['seq_rms'].append(_rms(r_seq))
    abl = OrderedDict()
    for k in names:
        st = R.drop_stats(base_ssim, tests[k])
        st['ssim_mean'] = float(np.mean(tests[k]))
        st['verdict_used'] = R.used_verdict(st)
        abl[k] = st
    model.train()
    return OrderedDict(n_slices=n, ssim_normal_mean=float(np.mean(base_ssim)), decomposition=decomp,
                       substitution_tests=abl,
                       first_conv_response={k: float(np.mean(v)) for k, v in resp.items()},
                       grid={k: float(np.mean([g[k] for g in grid])) for k in grid[0]},
                       grid_subsample=sub, seconds=round(time.time() - t0, 1))


@torch.no_grad()
def weight_norm_ratios(model, model0):
    sd, sd0 = model.state_dict(), model0.state_dict()
    out = OrderedDict()
    for k in ('ss2d.out_proj.weight', 'ss2d.out_proj.bias', 'ss2d.merge.weight', 'ss2d.in_proj.weight',
              'ss2d.dwconv.weight', 'ss2d.norm_in.weight'):
        if k in sd and k in sd0:
            a, b = sd[k].float(), sd0[k].float()
            out[k] = dict(norm_end=float(a.norm()), norm_init=float(b.norm()), ratio=float(a.norm() / b.norm().clamp(min=1e-30)),
                          cos_delta_w0=float(F.cosine_similarity((a - b).flatten(), b.flatten(), dim=0)))
    k = 'unet.down_path.0.block.0.weight'
    a, b = sd[k].float()[:, :SEQ_CH], sd0[k].float()[:, :SEQ_CH]
    out['unet_first_conv_seq_in'] = dict(norm_end=float(a.norm()), norm_init=float(b.norm()),
                                         ratio=float(a.norm() / b.norm().clamp(min=1e-30)))
    a, b = sd[k].float()[:, SEQ_CH:], sd0[k].float()[:, SEQ_CH:]
    out['unet_first_conv_zf_in'] = dict(norm_end=float(a.norm()), norm_init=float(b.norm()),
                                        ratio=float(a.norm() / b.norm().clamp(min=1e-30)))
    return out


# ------------------------------------------------------------------ V0/V1/V2
def run_variant(args, device):
    variant = args.variant
    vname = vname_of(args)
    vdir = os.path.join(args.out_dir, vname)
    os.makedirs(vdir, exist_ok=True)
    final_path = os.path.join(args.out_dir, f'{vname}_final.json')
    jsonl_final = os.path.join(vdir, f'{vname}.jsonl')
    ckpt_final = os.path.join(vdir, f'{vname}_end.pt')

    train_ds, val_ds, val_indices, probe_indices, probe = prepare_data(args, device)
    print(f'  val 슬라이스 {len(val_indices)}개: {val_indices[:8]}{" ..." if len(val_indices) > 8 else ""} | probe {probe_indices}')
    t0 = time.time()
    dl_scale = R.estimate_mean_scale(train_ds, args.seed)
    dl_scale['source'] = f'train {dl_scale["n"]} slices'
    print(f'  DL s̄ = {dl_scale["mean"]:.4f} → λ_DL = {C.LAMBDA_SSIM_PER_PIXEL / dl_scale["mean"]:.6g} ({time.time() - t0:.0f}s)')

    diag = OrderedDict(diag_version=DIAG_VERSION, smoke_script_version=R.SCRIPT_VERSION, variant=variant, vname=vname,
                       description=VDESC[variant], started=now(), argv=sys.argv, seed=args.seed, steps=args.steps,
                       nonorm_gamma=args.nonorm_gamma if variant == 'nonorm' else None,
                       reference_smoke_ss2d_DL=dict(val_ssim=0.7771596375387162, shuffle_cross_mean=0.00010094023309648037,
                                                    shuffle_cross_npos=40, n=64, dl_scale_mean=354.3457614183426))
    if variant == 'img':
        diag['ifft_check_rel_err'] = check_ifft_convention(probe, device)
        print(f'  ifft2c 관례 확인: ifft2c(x_ksp_n) vs x_img_n 상대 오차 {diag["ifft_check_rel_err"]}')
        assert max(diag['ifft_check_rel_err']) < 1e-4, 'ifft2c 관례가 로더와 다르다'
    if variant == 'nonorm':
        diag['input_scale'] = input_scale_stats(probe, device, args.nonorm_gamma)
        print(f'  V2 입력 크기 (probe): {diag["input_scale"]}')

    R.build_model = make_builder(variant, args.nonorm_gamma)          # run_job 이 부르는 이름만 교체 (원 파일 무수정)
    ns = argparse.Namespace(
        seq='ss2d', cond='DL', jobs=None, steps=args.steps, tmax=None, log_every=args.log_every,
        log_dense_every=args.log_dense_every, log_dense_until=args.log_dense_until, val_slices=args.val_slices,
        seed=args.seed, lambda_ssim=C.LAMBDA_SSIM_PER_PIXEL, lambda_dl_scale=None, out_dir=vdir, save_ckpt='end',
        bs=args.bs, device='cuda', force=args.force, max_minutes=args.max_minutes, num_workers=args.num_workers,
        val_workers=args.val_workers, skip_done=False, job_list=[('ss2d', 'DL')], tmax_eff=args.steps,
        probe_indices=probe_indices, dl_scale=dl_scale)
    status = 'failed'
    summ, extra, err = None, None, None
    try:
        summ = R.run_job('ss2d', 'DL', ns, train_ds, val_ds, val_indices, probe, device, vdir)
        status = summ.get('status', 'completed')
        os.replace(os.path.join(vdir, 'ss2d_DL.jsonl'), jsonl_final)
        os.replace(os.path.join(vdir, 'ss2d_DL_end.pt'), ckpt_final)
        gc.collect()
        torch.cuda.empty_cache()
        # 추가 분석: 끝 상태 / 초기값 (같은 시드로 다시 만든 초기 모델)
        build = R.build_model
        model = build('ss2d', device)
        sd = torch.load(ckpt_final, map_location=device)
        model.load_state_dict(sd, strict=True)
        R.seed_all(args.seed)
        model0 = build('ss2d', device)
        extra = OrderedDict(weight_norm_ratios=weight_norm_ratios(model, model0))
        extra['end'] = post_analysis(model, val_ds, val_indices[:args.post_slices], device, args.val_workers)
        extra['init'] = post_analysis(model0, val_ds, val_indices[:args.post_slices], device, args.val_workers)
        del model, model0, sd
        e, i0 = extra['end'], extra['init']
        print(f'[post] end: ssim {e["ssim_normal_mean"]:.4f} seq_out_rms {e["decomposition"]["seq_out_rms"]:.4g} '
              f'(init {i0["decomposition"]["seq_out_rms"]:.4g}) slice-specific rms {e["decomposition"]["slice_specific_spatial_rms"]:.4g} '
              f'| zero {e["substitution_tests"]["zero"]["mean"]:+.5f} slice_mean {e["substitution_tests"]["slice_mean_plane"]["mean"]:+.5f} '
              f'shuffle {e["substitution_tests"]["shuffle_cross"]["mean"]:+.5f} | grid R² zf {e["grid"]["r2_zf_rss"]:.3f} '
              f'logk {e["grid"]["r2_logk_sampled"]:.3f} (init {i0["grid"]["r2_zf_rss"]:.3f}/{i0["grid"]["r2_logk_sampled"]:.3f}) '
              f'| out_proj ratio {extra["weight_norm_ratios"]["ss2d.out_proj.weight"]["ratio"]:.3f}', flush=True)
    except KeyboardInterrupt:
        raise
    except Exception as ex:                                             # noqa: BLE001
        err = f'{type(ex).__name__}: {ex}\n{traceback.format_exc()[-4000:]}'
        print(f'[FAIL] {vname}: {err}', flush=True)
        status = 'failed' if summ is None else f'{status}+post_failed'
    diag['finished'] = now()
    out = OrderedDict(diag=diag, status=status, error=err, run_job_summary=summ, post=extra)
    write_json(final_path, out)
    rebuild_md(args.out_dir)
    print(f'[{vname}] 끝 status={status} → {final_path}', flush=True)
    return 0 if summ is not None and err is None else R.EXIT_JOB_FAILED


# ------------------------------------------------------------------ CAP
class SepDense(nn.Module):
    """분리형 dense 기준선: 코일 공유 복소 행렬, y_c = A_r · z_c · A_cᵀ (fp32, autocast 끔). ifft2c 를 정확히 표현할 수 있다."""

    def __init__(self, n):
        super().__init__()
        std = 1.0 / math.sqrt(2 * n)
        self.ar_re = nn.Parameter(torch.randn(n, n) * std)
        self.ar_im = nn.Parameter(torch.randn(n, n) * std)
        self.ac_re = nn.Parameter(torch.randn(n, n) * std)
        self.ac_im = nn.Parameter(torch.randn(n, n) * std)

    def forward(self, x):
        with torch.autocast('cuda', enabled=False):
            xf = x.float()
            z = torch.complex(xf[:, 0::2], xf[:, 1::2])                      # (B, 16, H, W)
            Ar = torch.complex(self.ar_re, self.ar_im)
            Ac = torch.complex(self.ac_re, self.ac_im)
            y = torch.matmul(torch.matmul(Ar, z), Ac.transpose(0, 1))
            B, Cc, H, W = y.shape
            return torch.stack((y.real, y.imag), dim=2).reshape(B, 2 * Cc, H, W)


class CapModel(nn.Module):
    def __init__(self, kind, target, gamma):
        super().__init__()
        from ss2d import SS2D
        self.kind, self.target = kind, target
        self.domain = 'img' if '_img' in kind else 'ksp'
        T = 1 if target == 'rss' else 2 * C.N_COIL
        if kind.startswith('ss2d'):
            self.body = SS2D(c_in=2 * C.N_COIL, d_inner=C.SS2D_D_INNER, d_state=C.SS2D_D_STATE, out_ch=SEQ_CH)
            if kind.endswith('_nonorm'):
                self.body.norm_in = make_norm_replacement(gamma)
            feat = SEQ_CH
        elif kind == 'sep':
            self.body = SepDense(C.IMAGE_SIZE[0])
            feat = 1 if target == 'rss' else 2 * C.N_COIL
        else:
            raise ValueError(kind)
        self.readout = nn.Conv2d(feat, T, kernel_size=1, bias=True)

        self.n_feat, self.n_out = feat, T

    def features(self, xk):
        inp = ifft2c_packed(xk) if self.domain == 'img' else xk
        f = self.body(inp)
        if self.kind == 'sep' and self.target == 'rss':                      # 고정 비선형 RSS (기준선 전용)
            f = torch.sqrt(f.float().pow(2).sum(1, keepdim=True) + 1e-12)
        return f

    def forward(self, xk):
        return self.readout(self.features(xk))


def cap_target(xi, target):
    return xi.float().pow(2).sum(1, keepdim=True).sqrt() if target == 'rss' else xi.float()


def load_fit_set(train_ds, seed, n):
    """readout 최소제곱 적합용 고정 train 슬라이스 (마스크 offset·flip 은 고정 rng — 결정적, 끝나면 rng 복원)."""
    old = train_ds.rng
    train_ds.rng = np.random.default_rng(seed + 4242)
    out = []
    for i in R.even_indices(len(train_ds), n):
        d = train_ds[i]
        out.append((torch.from_numpy(np.ascontiguousarray(d['data'])).float()[None],
                    torch.from_numpy(np.ascontiguousarray(d['data_img'])).float()[None]))
    train_ds.rng = old
    return out


@torch.no_grad()
def lsq_readout(model, fitset, target, device, sub=4, ridge=1e-6):
    """고정 적합 슬라이스에서 features → target 의 최소제곱 1×1 readout (W: (T,k), b: (T,)). 4×4 부분표본 화소, 상대 ridge."""
    was_training = model.training
    model.eval()
    k, T = model.n_feat, model.n_out
    XtX = torch.zeros(k + 1, k + 1, dtype=torch.float64, device=device)
    XtY = torch.zeros(k + 1, T, dtype=torch.float64, device=device)
    for xk, xi in fitset:
        xk, xi = R.normalize_inputs(xk.to(device), xi.to(device), R.per_sample_scale(xi.to(device)))
        with torch.amp.autocast('cuda'):
            f = model.features(xk)
        X = f.float()[0, :, ::sub, ::sub].reshape(k, -1).t().double()
        X = torch.cat((X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=device)), dim=1)
        Y = cap_target(xi, target)[0, :, ::sub, ::sub].reshape(T, -1).t().double()
        XtX += X.t() @ X
        XtY += X.t() @ Y
    reg = torch.zeros_like(XtX)
    d = torch.diagonal(XtX)[:k]
    reg[range(k), range(k)] = ridge * float(d.mean().clamp(min=1e-30))
    sol = torch.linalg.solve(XtX + reg, XtY)                                 # (k+1, T)
    model.train(was_training)
    return sol[:k].t().contiguous(), sol[k].contiguous()


@torch.no_grad()
def set_readout(model, W, b):
    model.readout.weight.copy_(W.view(model.n_out, model.n_feat, 1, 1).to(model.readout.weight.dtype))
    model.readout.bias.copy_(b.to(model.readout.bias.dtype))


def _batch_r2(p, y):
    yc = y - y.mean(dim=(2, 3), keepdim=True)
    return float(1.0 - (p - y).pow(2).sum() / yc.pow(2).sum().clamp(min=1e-30))


@torch.no_grad()
def cap_eval(model, target, val_ds, indices, device, workers):
    model.eval()
    sse_f = sst_f = sse_b = sst_b = 0.0
    nm_f, nm_b = [], []
    for idx, b in R.iter_val(val_ds, indices, device, workers, True):
        xk, xi = R.normalize_inputs(b['data'], b['data_img'], R.per_sample_scale(b['data_img']))
        y = cap_target(xi, target)
        with torch.amp.autocast('cuda'):
            p = model(xk)
        p = p.float()
        d = p - y
        sse_f += float(d.pow(2).sum())
        sst_f += float((y - y.mean(dim=(2, 3), keepdim=True)).pow(2).sum())
        nm_f.append(float(d.pow(2).sum() / y.pow(2).sum().clamp(min=1e-30)))
        m = b['brain_mask'].expand_as(y)
        msum = m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        ym = (y * m).sum(dim=(2, 3), keepdim=True) / msum
        sse_b += float((d * m).pow(2).sum())
        sst_b += float(((y - ym) * m).pow(2).sum())
        nm_b.append(float((d * m).pow(2).sum() / (y * m).pow(2).sum().clamp(min=1e-30)))
    model.train()
    return OrderedDict(n=len(indices), r2_fov=1.0 - sse_f / max(sst_f, 1e-30), r2_brain=1.0 - sse_b / max(sst_b, 1e-30),
                       nmse_fov_mean=float(np.mean(nm_f)), nmse_brain_mean=float(np.mean(nm_b)))


@torch.no_grad()
def coil_rank_ceiling(val_ds, indices, device, workers, rank=SEQ_CH):
    """'coil' target 에서 20채널 표현 + 1×1 readout 의 상한: 채널 공분산(전역 평균 중심) 의 상위 rank 고유값 밖 에너지 =
    최소 SSE. R² 는 cap_eval 과 같은 SST(슬라이스·채널별 평균 중심)로 나눈다 (전체 FOV)."""
    Cn = 2 * C.N_COIL
    s1 = torch.zeros(Cn, dtype=torch.float64, device=device)
    s2 = torch.zeros(Cn, Cn, dtype=torch.float64, device=device)
    N = 0
    sst = 0.0
    for idx, b in R.iter_val(val_ds, indices, device, workers, True):
        _, xi = R.normalize_inputs(b['data'], b['data_img'], R.per_sample_scale(b['data_img']))
        y = xi.double()[0].reshape(Cn, -1)
        s1 += y.sum(1)
        s2 += y @ y.t()
        N += y.shape[1]
        sst += float((y - y.mean(1, keepdim=True)).pow(2).sum())
    mu = s1 / N
    cov = s2 / N - torch.outer(mu, mu)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0).flip(0)
    sse_min = float(ev[rank:].sum()) * N
    return OrderedDict(rank=rank, r2_ceiling_fov=1.0 - sse_min / max(sst, 1e-30),
                       energy_frac_top_rank=float(ev[:rank].sum() / ev.sum().clamp(min=1e-30)))


def run_cap_job(kind, target, lr_override, args, train_ds, val_ds, val_indices, fitset, device, cap_dir):
    lr = lr_override if lr_override is not None else (args.cap_lr if args.cap_lr is not None else C.LEARNING_RATE_ADAM)
    tag = f'{kind}__{target}' + (f'__lr{lr_override:g}' if lr_override is not None else '')
    t_job = time.time()
    torch.cuda.reset_peak_memory_stats()
    R.seed_all(args.seed)
    model = CapModel(kind, target, args.nonorm_gamma).to(device)
    model.train()
    # readout 초기값 = 초기 features 에 대한 최소제곱 해 (readout 초기화가 1×1 readout 의 느린 학습으로 결과를 가리지 않게 — 전 모듈 공통)
    W0r, b0r = lsq_readout(model, fitset, target, device)
    set_readout(model, W0r, b0r)
    ev_init = cap_eval(model, target, val_ds, val_indices, device, args.val_workers)
    print(f'  [CAP {tag}] 초기 features + 최소제곱 readout: R² fov {ev_init["r2_fov"]:.4f} brain {ev_init["r2_brain"]:.4f}', flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    groups = OrderedDict(body=[n for n, _ in model.named_parameters() if n.startswith('body.')],
                         readout=[n for n, _ in model.named_parameters() if n.startswith('readout.')])
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    decay, no_decay, reasons, _ = R.split_decay(model)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': C.LAMBDA_REGULAR_PER_PIXEL},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=lr, eps=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.cap_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    loader = R.make_train_loader(train_ds, args.bs, args.seed, args.num_workers, True)
    max_norm = R.CONDS['DL']['max_norm']
    jl = open(os.path.join(cap_dir, f'{tag}.jsonl'), 'w')
    meta = OrderedDict(type='meta', diag_version=DIAG_VERSION, tag=tag, kind=kind, target=target, domain=model.domain,
                       steps=args.cap_steps, bs=args.bs, lr=lr, eta_min=1e-6, weight_decay=C.LAMBDA_REGULAR_PER_PIXEL,
                       max_norm=max_norm, seed=args.seed, n_params=n_params, no_decay_reasons=reasons,
                       nonorm_gamma=args.nonorm_gamma if kind.endswith('_nonorm') else None,
                       loss='MSE (normalized units: x_img/s; target rss = sqrt(sum_c x_img^2))',
                       readout_init='least squares on initial features (fit set)', fit_slices=len(fitset),
                       eval_init_lsq=ev_init, started=now())
    jl.write(json.dumps(meta, ensure_ascii=False, default=float) + '\n')
    jl.flush()
    print(f'\n========== CAP {tag}: params {n_params:,} lr {lr:g} steps {args.cap_steps} ==========', flush=True)

    pdict = dict(model.named_parameters())

    def rel(names):
        num = sum(float((pdict[n].detach() - w0[n]).pow(2).sum()) for n in names)
        den = sum(float(w0[n].pow(2).sum()) for n in names)
        return math.sqrt(num) / math.sqrt(den) if den > 0 else None

    S = dict(step=0, loss=0.0, r2=0.0, k=0, t=time.time(), nsk=0, skips=0, consec=0)

    def emit(gn):
        k = S['k']
        rec = OrderedDict(type='step', step=S['step'], lr=opt.param_groups[0]['lr'],
                          loss=S['loss'] / k if k else None, batch_r2=S['r2'] / k if k else None,
                          grad_norm=gn, rel_change={g: rel(v) for g, v in groups.items()},
                          scaler_scale=scaler.get_scale(), amp_skips=S['skips'],
                          sec_per_step=(time.time() - S['t']) / S['nsk'] if S['nsk'] else None)
        jl.write(json.dumps(rec, ensure_ascii=False, default=float) + '\n')
        jl.flush()
        gtxt = ' '.join(f'{g}={v:.3g}' for g, v in (gn or {}).items())
        print(f'[CAP {tag}] step {S["step"]} loss {rec["loss"] if rec["loss"] is None else format(rec["loss"], ".4g")} '
              f'R²(batch) {rec["batch_r2"] if rec["batch_r2"] is None else format(rec["batch_r2"], ".4f")} grad {gtxt} '
              f'relΔ body {rec["rel_change"]["body"] or 0:.3g} | {rec["sec_per_step"] or 0:.3f}s/step', flush=True)
        S.update(loss=0.0, r2=0.0, k=0, t=time.time(), nsk=0)

    emit(None)
    it = iter(loader)
    status = 'completed'
    while S['step'] < args.cap_steps:
        try:
            sample = next(it)
        except StopIteration:
            it = iter(loader)
            sample = next(it)
        xk = sample['data'].float().to(device, non_blocking=True)
        xi = sample['data_img'].float().to(device, non_blocking=True)
        xk, xi = R.normalize_inputs(xk, xi, R.per_sample_scale(xi))
        y = cap_target(xi, target)
        with torch.amp.autocast('cuda'):
            p = model(xk)
        loss = F.mse_loss(p.float(), y)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            S['consec'] += 1
            if S['consec'] >= C.MAX_CONSEC_SKIP:
                status = 'aborted_nonfinite'
                break
            continue
        S['consec'] = 0
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        log_now = (S['step'] + 1) % args.cap_log_every == 0 or S['step'] + 1 == args.cap_steps
        gn = None
        if log_now:
            gn = {}
            for g, names in groups.items():
                sq = sum(float(pdict[n].grad.detach().float().pow(2).sum()) for n in names if pdict[n].grad is not None)
                gn[g] = math.sqrt(sq)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        sc0 = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < sc0:
            S['skips'] += 1
        sched.step()
        opt.zero_grad(set_to_none=True)
        S['step'] += 1
        S['nsk'] += 1
        S['loss'] += float(loss)
        S['r2'] += _batch_r2(p.detach().float(), y)
        S['k'] += 1
        if log_now:
            emit(gn)
    train_sec = time.time() - t_job
    del it, loader
    ev = cap_eval(model, target, val_ds, val_indices, device, args.val_workers)
    # 학습된 features 의 선형 해독 가능성: readout 을 적합 슬라이스 최소제곱 해로 다시 맞춘 뒤 같은 val 평가
    Wt, bt = model.readout.weight.detach().clone(), model.readout.bias.detach().clone()
    W1r, b1r = lsq_readout(model, fitset, target, device)
    set_readout(model, W1r, b1r)
    ev_refit = cap_eval(model, target, val_ds, val_indices, device, args.val_workers)
    with torch.no_grad():
        model.readout.weight.copy_(Wt)
        model.readout.bias.copy_(bt)
    res = OrderedDict(tag=tag, kind=kind, target=target, domain=model.domain, status=status, steps_done=S['step'], lr=lr,
                      n_params=n_params, amp_skips=S['skips'], eval=ev, eval_refit=ev_refit, eval_init_lsq=ev_init,
                      final_rel_change={g: rel(v) for g, v in groups.items()},
                      train_sec=round(train_sec, 1), seconds=round(time.time() - t_job, 1),
                      sec_per_step=round(train_sec / max(1, S['step']), 4),
                      peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), finished=now())
    jl.write(json.dumps(dict(type='final', **res), ensure_ascii=False, default=float) + '\n')
    jl.close()
    print(f'[CAP {tag}] 끝: R² fov {ev["r2_fov"]:.4f} brain {ev["r2_brain"]:.4f} NMSE fov {ev["nmse_fov_mean"]:.4f} '
          f'brain {ev["nmse_brain_mean"]:.4f} | refit R² fov {ev_refit["r2_fov"]:.4f} brain {ev_refit["r2_brain"]:.4f} '
          f'(초기 {ev_init["r2_fov"]:.4f}/{ev_init["r2_brain"]:.4f}) | {res["sec_per_step"]:.3f}s/step {res["seconds"]:.0f}s', flush=True)
    del model, opt, sched, scaler
    return res


def run_cap(args, device):
    cap_dir = os.path.join(args.out_dir, 'cap')
    os.makedirs(cap_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, 'cap_final.json')
    jobs = []
    for tok in args.cap_jobs.split(','):
        tok = tok.strip()
        if not tok:
            continue
        lr_o = None
        if '@' in tok:
            tok, lr_s = tok.split('@')
            lr_o = float(lr_s)
        k, t = tok.split(':')
        assert k in CAP_KINDS and t in CAP_TARGETS, tok
        jobs.append((k, t, lr_o))
    train_ds, val_ds, val_indices, _, probe = prepare_data(args, device)
    fitset = load_fit_set(train_ds, args.seed, args.cap_fit_slices)
    prev = {}
    if args.skip_done and os.path.exists(final_path):
        with open(final_path) as f:
            old = json.load(f)
        if old.get('diag', {}).get('cap_steps') == args.cap_steps and old.get('diag', {}).get('diag_version') == DIAG_VERSION:
            prev = {r['tag']: r for r in old.get('jobs', []) if r.get('status') == 'completed' and 'tag' in r}
    diag = OrderedDict(diag_version=DIAG_VERSION, smoke_script_version=R.SCRIPT_VERSION, variant='cap', description=VDESC['cap'],
                       started=now(), argv=sys.argv, seed=args.seed, cap_steps=args.cap_steps, bs=args.bs,
                       cap_lr=args.cap_lr if args.cap_lr is not None else C.LEARNING_RATE_ADAM,
                       jobs=[f'{k}:{t}' + (f'@{lo:g}' if lo is not None else '') for k, t, lo in jobs],
                       fit_slices=len(fitset), readout='init = least squares on initial features; eval also with refit readout',
                       nonorm_gamma=args.nonorm_gamma, ifft_check_rel_err=check_ifft_convention(probe, device))
    out = OrderedDict(diag=diag, jobs=[], coil_rank20_ceiling=None, status='running')
    if any(t == 'coil' for _, t, _ in jobs):
        out['coil_rank20_ceiling'] = coil_rank_ceiling(val_ds, val_indices, device, args.val_workers)
        print(f'  coil target rank-20 상한: {out["coil_rank20_ceiling"]}', flush=True)
    n_fail = 0
    for k, t, lo in jobs:
        tag = f'{k}__{t}' + (f'__lr{lo:g}' if lo is not None else '')
        if tag in prev:
            print(f'[skip-done] CAP {tag}')
            out['jobs'].append(prev[tag])
            continue
        try:
            res = run_cap_job(k, t, lo, args, train_ds, val_ds, val_indices, fitset, device, cap_dir)
        except KeyboardInterrupt:
            raise
        except Exception as ex:                                         # noqa: BLE001
            n_fail += 1
            tb = traceback.format_exc()
            print(f'[FAIL] CAP {tag}: {ex}\n{tb}', flush=True)
            res = OrderedDict(tag=tag, kind=k, target=t, status='failed', error=f'{type(ex).__name__}: {ex}', traceback=tb[-3000:])
        out['jobs'].append(res)
        write_json(final_path, out)
        rebuild_md(args.out_dir)
        gc.collect()
        torch.cuda.empty_cache()
    out['status'] = 'completed' if n_fail == 0 else f'{n_fail} failed'
    out['diag']['finished'] = now()
    write_json(final_path, out)
    rebuild_md(args.out_dir)
    return 0 if n_fail == 0 else R.EXIT_JOB_FAILED


# ------------------------------------------------------------------ 요약 md
def _f(x, fmt):
    return format(x, fmt) if isinstance(x, (int, float)) and not isinstance(x, bool) else '—'


def _ds(d):
    if not d:
        return '—'
    p = d.get('p_sign')
    return f'{d["mean"]:+.2e} ± {_f(d.get("se"), ".1e")} ({d["n_pos"]}/{d["n"]}, p={_f(p, ".2g")})'


def rebuild_md(out_dir):
    lines = ['# SS2D 인과 진단 — 요약 (자동 생성, diag_ss2d_variants.py)\n',
             '각 조건은 같은 DL 레시피·SEED 1·데이터 순서·검증 64 슬라이스(smoke_recipe_fix 와 같음). dSSIM = 정상 − 시험 '
             '(양수 = 그 입력을 씀), ± SE, (양수 개수/n, 부호 검정 p). "사용" = mean>0 & mean>2SE & p<0.01.\n']
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if not (fn.startswith('v') and fn.endswith('_final.json')):
            continue
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    rows.append(json.load(f))
            except Exception:                                           # noqa: BLE001
                pass
    if rows:
        lines.append('\n## V0/V1/V2 (U-Net 포함 4000 step 학습)\n')
        lines.append('| 조건 | status | steps | val SSIM | PSNR | nMSE | shuffle_cross dSSIM | zero dSSIM | 사용 판정 | '
                     'seq out RMS 초기→끝 | 슬라이스 고유 공간 RMS | slice_mean_plane dSSIM | no_slice_specific dSSIM | '
                     'R² zf RSS / log abs k (표본 화소) (끝; 초기) | out_proj norm 비 |\n')
        lines.append('|' + '---|' * 15 + '\n')
        for r in rows:
            s = r.get('run_job_summary') or {}
            val = s.get('val') or {}
            v = val.get('normal') or {}
            t = val.get('tests') or {}
            post = r.get('post') or {}
            e, i0 = post.get('end') or {}, post.get('init') or {}
            de, di = e.get('decomposition') or {}, i0.get('decomposition') or {}
            st = e.get('substitution_tests') or {}
            ge, gi = e.get('grid') or {}, i0.get('grid') or {}
            wr = (post.get('weight_norm_ratios') or {}).get('ss2d.out_proj.weight') or {}
            lines.append(
                f'| {r["diag"]["vname"]} | {r.get("status")} | {s.get("steps_done", "—")} | {_f(v.get("ssim"), ".4f")} '
                f'| {_f(v.get("psnr"), ".2f")} | {_f(v.get("nmse"), ".5f")} | {_ds(t.get("shuffle_cross"))} | {_ds(t.get("zero"))} '
                f'| {(s.get("verdict") or {}).get("seq_used", "—")} '
                f'| {_f(di.get("seq_out_rms"), ".3g")} → {_f(de.get("seq_out_rms"), ".3g")} '
                f'| {_f(de.get("slice_specific_spatial_rms"), ".3g")} | {_ds(st.get("slice_mean_plane"))} '
                f'| {_ds(st.get("no_slice_specific_spatial"))} '
                f'| {_f(ge.get("r2_zf_rss"), ".3f")} / {_f(ge.get("r2_logk_sampled"), ".3f")} ({_f(gi.get("r2_zf_rss"), ".3f")} / '
                f'{_f(gi.get("r2_logk_sampled"), ".3f")}) | {_f(wr.get("ratio"), ".3f")} |\n')
        lines.append('\n참고 — smoke_recipe_fix ss2d:DL: val SSIM 0.7772, shuffle_cross +1.0e-04 (40/64), zero +2.6e-02 (59/64).\n')
    cp = os.path.join(out_dir, 'cap_final.json')
    if os.path.exists(cp):
        try:
            with open(cp) as f:
                cap = json.load(f)
            lines.append(f'\n## CAP (U-Net 없음, {cap["diag"]["cap_steps"]} step, 기본 LR {cap["diag"]["cap_lr"]:g}, 정규화 단위 MSE)\n')
            ce = cap.get('coil_rank20_ceiling')
            if ce:
                lines.append(f'\ncoil target 의 rank-20 readout 상한(R² 전체 FOV) = {ce["r2_ceiling_fov"]:.4f} '
                             f'(상위 20 채널 PCA 에너지 비 {ce["energy_frac_top_rank"]:.4f}) — SS2D(20ch 출력)에만 해당, sep 은 32ch.\n')
            lines.append('\nreadout 은 시작 때 초기 features 의 최소제곱 해로 맞추고(모든 모듈 공통) 끝에 다시 맞춘 값(refit)으로도 평가한다. '
                         'R² = 1 − SSE/SST (SST = 슬라이스·채널별 평균 중심), val 64 슬라이스.\n')
            lines.append('\n| 모듈 | target | 입력 영역 | LR | status | R² FOV (학습 readout) | R² brain | R² FOV (refit) | R² brain (refit) '
                         '| R² FOV (초기 features) | NMSE FOV | s/step | params |\n')
            lines.append('|' + '---|' * 13 + '\n')
            for j in cap.get('jobs', []):
                ev, er, ei = j.get('eval') or {}, j.get('eval_refit') or {}, j.get('eval_init_lsq') or {}
                lines.append(f'| {j["kind"]} | {j["target"]} | {j.get("domain", "—")} | {_f(j.get("lr"), "g")} | {j.get("status")} '
                             f'| {_f(ev.get("r2_fov"), ".4f")} | {_f(ev.get("r2_brain"), ".4f")} | {_f(er.get("r2_fov"), ".4f")} '
                             f'| {_f(er.get("r2_brain"), ".4f")} | {_f(ei.get("r2_fov"), ".4f")} | {_f(ev.get("nmse_fov_mean"), ".4f")} '
                             f'| {_f(j.get("sec_per_step"), ".3f")} | {j.get("n_params", "—")} |\n')
        except Exception as ex:                                         # noqa: BLE001
            lines.append(f'\n(cap_final.json 읽기 실패: {ex})\n')
    p = os.path.join(out_dir, 'diag_summary.md')
    with open(p + '.tmp', 'w') as f:
        f.write(''.join(lines))
    os.replace(p + '.tmp', p)


# ------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser(description='SS2D 인과 진단 (V0 ctrl / V1 img / V2 nonorm / CAP)',
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', required=True, choices=list(VNAME))
    p.add_argument('--steps', type=int, default=4000, help='V0/V1/V2 학습 step (스모크와 같은 4000)')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--log-dense-every', type=int, default=20)
    p.add_argument('--log-dense-until', type=int, default=500)
    p.add_argument('--val-slices', type=int, default=64)
    p.add_argument('--post-slices', type=int, default=64, help='추가 분석 슬라이스 수 (val 슬라이스 앞쪽부터)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--nonorm-gamma', type=float, default=1.0, help='V2·CAP nonorm 의 norm_in 대체 스칼라 (1 = Identity)')
    p.add_argument('--cap-steps', type=int, default=1500)
    p.add_argument('--cap-log-every', type=int, default=100)
    p.add_argument('--cap-lr', type=float, default=None, help='CAP LR (기본 = 레시피 LR 2e-4)')
    p.add_argument('--cap-jobs', default=CAP_DEFAULT_JOBS, help='"모듈:target[@lr]" 쉼표 목록')
    p.add_argument('--cap-fit-slices', type=int, default=16, help='readout 최소제곱 적합용 train 슬라이스 수')
    p.add_argument('--out-dir', default='results/ss2d_diag_2026-10-11')
    p.add_argument('--max-minutes', type=float, default=240)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--val-workers', type=int, default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--skip-done', action='store_true', help='같은 설정으로 끝난 조건(또는 CAP job)은 건너뜀')
    a = p.parse_args()
    if a.num_workers is None:
        a.num_workers = C.NUM_WORKERS_TRAIN
    if a.val_workers is None:
        a.val_workers = C.NUM_WORKERS_VAL
    if not os.path.isabs(a.out_dir):
        a.out_dir = os.path.join(_PROJECT_ROOT, a.out_dir)
    a.post_slices = min(a.post_slices, a.val_slices)
    return a


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print('=' * 72)
    print(f' SS2D 인과 진단 ({DIAG_VERSION}, smoke {R.SCRIPT_VERSION}) variant={args.variant} — {VDESC[args.variant]}')
    print(f'   steps={args.steps} cap_steps={args.cap_steps} bs={args.bs} seed={args.seed} val={args.val_slices} '
          f'γ={args.nonorm_gamma} out={args.out_dir}  {now()}  torch {torch.__version__}')
    print('=' * 72, flush=True)
    if args.variant != 'cap' and args.skip_done:
        fp = os.path.join(args.out_dir, f'{vname_of(args)}_final.json')
        if os.path.exists(fp):
            try:
                with open(fp) as f:
                    old = json.load(f)
                d = old.get('diag', {})
                if (old.get('status') in ('completed', 'truncated_time') and old.get('error') is None
                        and d.get('diag_version') == DIAG_VERSION and d.get('steps') == args.steps):
                    print(f'[skip-done] {vname_of(args)} 이미 끝남 — 건너뜀')
                    return 0
            except Exception as ex:                                     # noqa: BLE001
                print(f'[skip-done] 이전 결과 읽기 실패 ({ex}) — 다시 실행')
    R.gpu0_precheck(args.force)
    device = torch.device('cuda')
    if args.variant == 'cap':
        return run_cap(args, device)
    return run_variant(args, device)


if __name__ == '__main__':
    sys.exit(main())
