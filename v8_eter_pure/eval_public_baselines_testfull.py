"""
공개 가중치 기준선(fastMRI U-Net · E2E-VarNet · PromptMR+)을 test_full 에서 우리 모델과 같은 지표로 평가한다 (2026-10-23).

계획(결과 전, 구속): paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/public_baselines_testfull_plan.md
                    + 같은 폴더 public_baselines_testfull_plan_note1.md (샘플 줄 대응 규칙). 공개 모델 결과는 '외부 기준선'이다
                    (학습 데이터·레시피가 다름 — U-Net·E2E-VarNet 은 train+val, PromptMR+ 는 train). 시퀀스 모듈 효과를 보는
                    통제 비교가 아니다. 통제 비교는 그대로 기준 모델 ETER-net 대 비교 모델이다. 볼륨별 짝 차이·부트스트랩 CI 는
                    이 파일이 아니라 분석 단계가 per_volume CSV 로 계산한다.

두 방식 (--mode)
  native (고유 전처리, 주 보고) — 각 모델 원 저장소의 데이터 코드를 그대로 부른다. 코일 수·해상도는 원래대로, 마스크만 공유.
    - fastmri_unet : fastmri 0.3.0 SliceDataset → UnetDataTransform('multicoil', mask_func=공유 마스크) (zero-filled RSS 를
                     정답 크기로 자르고 instance norm·±6 clamp) → Unet → output·std + mean (UnetModule.test_step 규칙).
    - e2e_varnet   : SliceDataset → VarNetDataTransform(mask_func=공유 마스크) (전체 코일·원 k-space, 헤더 취득 구간 밖 마스크 0,
                     num_low_frequencies = 마스크 함수 값) → VarNet(masked_kspace, mask, num_low) → VarNetModule.test_step 의
                     자르기 규칙(recon_size, FLAIR 203 이면 정사각).
    - promptmr_plus: PromptMR-plus FastmriSliceDataset(num_adj_slices 5, 경계 복제) → FastmriDataTransform 의 fm-brain 추론
                     설정(configs/inference/pmr-plus/fm-brain.yaml: mask_func None, uniform_resolution [384,384], mask_type
                     cartesian, test_num_low_frequencies −1). 시험 분할 파일처럼 샘플하지 않은 줄을 0 으로 만든 k-space 와 공유
                     마스크를 넣고 target=None 으로 부른다(그들의 시험 경로: 판독 축만 384 로 자르고 위상 부호화 폭은 원래대로,
                     ACS 는 마스크에서 검출). → PromptMR(…, compute_sens_per_coil=--pmr-sens-per-coil) → predict_step 자르기 규칙.
                     (target 이 있을 때의 그들 학습·검증 경로는 두 축을 384 로 반사 패딩한 뒤 그 격자에 마스크를 씌운다 —
                     해상도를 바꾸므로 '해상도 원래대로'인 이 방식에서는 쓰지 않는다.)
    공유 샘플 줄 (보충 메모 1): 우리 로더(dataloader_h5_v5.py:192-212)는 원 k-space 를 영상으로 바꾼 뒤 위상 부호화 축을 384 로
      가운데 자르거나 0 으로 채우고 다시 FFT 한다. 그래서 384 틀의 줄 간격 = 1/384 (화소당 주기), 원 격자의 줄 간격 = 1/W 이고,
      두 격자의 줄이 하나씩 정확히 대응하는 것은 원 폭 W = 384 일 때뿐이다('exact': 같은 줄 번호 집합 = 우리 마스크 그대로).
      그 밖의 볼륨('rule')에는 같은 규칙을 원 격자에 적용한다: 등간격 R = 4, 중앙 줄 수 = round(W × 31/384)(반올림, 정수 산술),
      시작 위치 3 (build_r4_mask(W, n/W, 4, rng=None) — 우리 로더의 함수). 볼륨마다 방식·샘플 줄 수·실효 가속 계수를 기록한다.
  ours (우리 전처리, 보조) — 예전 val 평가(eval_baselines_full.py)와 같은 입력: 우리 평가 데이터셋
    (eval_unet_only_v8.build_val_dataset = eval_v8fix_official.py 와 같음) · 16코일 절단 · 384 틀 · 우리 평가 마스크
    (R 4, 중앙 31줄, 시작 3, 119줄 — 슬라이스마다 줄 단위로 확인) · visualize_multimodel_compare 의 prep_/forward_ 함수.
    VarNet·PromptMR+ 는 예전처럼 k-space 를 최댓값 1 로 나눠 넣고, 출력에 그 배율을 되돌려 물리 단위로 만든다(두 모델은
    양의 배율에 동차 — mean/std 정규화·선형 DC·RSS; 상수 항 없음. 시험 결과는 보고서에).

지표 (우리 모델과 같은 함수 — eval_v8fix_official.py 와 같은 절차, 최소제곱 강도 보정 없음)
  - brain mask 지표: eval_paired_v8_nodc.slice_metrics(출력, label, brain mask) — 384 틀, label = reconstruction_rss × 1e6,
    brain mask = 우리 로더(Otsu×0.4 + 최대 연결 성분). ours 방식은 우리 모델처럼 384 출력 전체를, native 방식은 정답 크기로
    자른 출력을 같은 오프셋으로 384 틀에 0 으로 채워 넣어 쓴다.
  - 공식 정의: eval_v8fix_official.official_volume(정답 RSS, 정답 크기로 자른 출력, brain mask) — SSIM(슬라이스 평균)·볼륨
    PSNR·볼륨 NMSE·안/밖 분해. 민감도 ssim_mask_volrange 도 그 파일 함수 그대로.
  - 보조 ls_* 열: 예전 val 프로토콜(brain mask 안 슬라이스별 최소제곱 강도 배율 보정, VM.ls_scale + VM.slice_metrics_np) —
    results/eval/baselines_384_full 재현 확인과 연속성 용도이며 주 지표가 아니다.
  - 정밀도: fp32(autocast 없음 — 각 원 구현의 기본). --tf32 off(기본) 이면 matmul·cuDNN TF32 를 끈다. 기록한다.
  - 비유한 출력(계획 5절, 분석 계획 수정 3 §2·보충 메모 1 #2): 공개 모델은 이미 fp32 라 다시 계산하지 않고 '결함'으로 보고한다.
    그 슬라이스 지표는 NaN 으로 두고(볼륨 값도 NaN), 끝에 <stem>.DEFECT 를 쓰고 종료 코드 3 으로 끝난다. 처리는 사용자가 정한다.

출력 (--out-dir, stem = <model>_<mode>; 이미 있으면 덮어쓰지 않고 거부 — 종료 코드 2)
  per_volume_<stem>.csv : eval_v8fix_official.py 와 같은 열(file … label_vs_rss_relmax) 뒤에 마스크·실효 가속·기하 열.
  per_slice_<stem>.csv  : 같은 열(idx, file, slice_idx, ssim, psnr, nmse, l1, ssim_official) 뒤에 finite·ls_*·배율·시간 열.
  summary_<stem>.json   : 지표(볼륨 평균·SD) · 실효 가속 계수 평균 · 마스크 대응 수 · 점검 · 결함 · 가중치/코드/패키지 sha256·판.
  partial_<stem>.jsonl  : 볼륨이 끝날 때마다 한 줄씩 덧붙이는 진행 기록(--resume 이 이어 쓴다; 끝나도 지우지 않는다).
  <stem>.DEFECT         : 비유한 출력이 있을 때만.
  dump_<stem>/          : --dump-volumes K 일 때 앞 K 볼륨의 배열(npz) — 독립 계산 대조용.

종료 코드: 0 정상 · 2 거부(출력 있음·설정 불일치·파일 목록 불일치) · 3 결함(비유한 출력) · 그 밖의 예외 1.

실행 (저장소 루트, GPU 하나):
  for m in fastmri_unet e2e_varnet promptmr_plus; do for d in native ours; do
    CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/eval_public_baselines_testfull.py --model $m --mode $d \\
      --data-path ./fastMRI_data/multicoil_test_full --out-dir results/eval/public_baselines_test_full \\
      --expect-files-from results/eval/v8fix_official_test_full/per_volume_unet.csv --num-workers 4 || break 2
  done; done
  자가 점검(데이터·가중치 불필요): python v8_eter_pure/eval_public_baselines_testfull.py --selftest
  + 데이터 마스크 점검: ... --selftest --data-path <폴더> [--max-files N]   (볼륨별 대응 표를 찍는다)
  배관 점검: --max-files N (앞 N 볼륨). 중단 뒤 이어서: 같은 명령 + --resume.
"""

import os
import sys
import csv
import json
import time
import types
import hashlib
import argparse
import datetime
import platform
import fractions
import subprocess

import numpy as np

_THIS = os.path.abspath(__file__)          # import 시점(= cwd 가 바뀌기 전)에 절대 경로로
_HERE = os.path.dirname(_THIS)
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, 'dataloaders')):
    if _p not in sys.path:
        sys.path.append(_p)

# ------------------------------------------------------------------ 상수 (우리 평가와 같은 값)
GRID = 384                 # 우리 틀 (myConfig_pure_eter_v8.IMAGE_SIZE)
ACCEL = 4                  # R
CENTER_FRACTION = 0.08     # build_val_dataset 의 center_fraction → 중앙 round(384 × 0.08) = 31줄
OURS_CENTER = 31
OFFSET = ACCEL - 1         # build_r4_mask(rng=None) 의 시작 위치 3
OURS_LINES = 119           # build_r4_mask(384, 0.08, 4, None).sum()
VAL_AMP_Y = 1e6            # label = reconstruction_rss × 1e6
AMP_X_KSP = 1e4            # data = k-space × 1e4
PMR_ROOT = os.path.join(_ROOT, 'external', 'PromptMR-plus')
PMR_INFER_CFG = dict(mask_func=None, uniform_resolution=[384, 384], use_seed=True, mask_type='cartesian',
                     test_num_low_frequencies=-1)                  # configs/inference/pmr-plus/fm-brain.yaml
PMR_NUM_ADJ = 5                                                   # 같은 설정 num_adj_slices (ckpt hyper_parameters 와 대조)

MODELS = {
    'fastmri_unet': dict(
        display='U-Net (fastMRI leaderboard weights)', weights='models/pretrained/brain_leaderboard_state_dict.pt',
        trained_on='fastMRI brain train + val splits (leaderboard model; fastMRI README)', vm='unet'),
    'e2e_varnet': dict(
        display='E2E-VarNet (fastMRI leaderboard weights)', weights='models/pretrained/varnet_brain_leaderboard_state_dict.pt',
        trained_on='fastMRI brain train + val splits (leaderboard model; fastMRI README)', vm='varnet'),
    'promptmr_plus': dict(
        display='PromptMR+ (fm-brain weights)', weights='external/weights/promptmr-plus-fm-brain-ep44.ckpt',
        upstream_name='promptmr-plus-epoch=44-step=1591830.ckpt', ckpt_expect=dict(epoch=44, global_step=1591830),
        trained_on='fastMRI brain train split (PromptMR-plus README table)', vm='promptmr'),
}
MODES = ('native', 'ours')
PLAN_FILES = ('paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/public_baselines_testfull_plan.md',
              'paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/public_baselines_testfull_plan_note1.md')
EXTERNAL_WORDING = ('external baselines with public weights, trained on different data (train and validation splits for '
                    'U-Net and E2E-VarNet; the training split for PromptMR+) and with different recipes; not a controlled '
                    'comparison')

# 열 순서: 앞부분은 eval_v8fix_official.py 와 같다 (per_slice :265, per_volume :247-254)
SLICE_COLS = ['idx', 'file', 'slice_idx', 'ssim', 'psnr', 'nmse', 'l1', 'ssim_official']
SLICE_EXTRA = ['finite', 'n_nonfinite', 'ls_alpha', 'ls_ssim', 'ls_psnr', 'ls_nmse', 'ls_l1', 'out_to_label',
               'coils_used', 'sec']
VOL_COLS = ['file', 'n', 'H', 'W', 'ssim_mask', 'psnr_mask', 'nmse_mask', 'ssim_mask_volrange', 'ssim_off', 'psnr_off',
            'nmse_off', 'maxval', 'ssim_off_in', 'ssim_off_out', 'frac_in', 'decomp_maxabs', 'label_vs_rss_relmax']
VOL_EXTRA = ['acquisition', 'coils_file', 'coils_used', 'kspace_H', 'kspace_W', 'recon_size', 'pad_left', 'pad_right',
             'grid_W', 'mapping', 'center_lines', 'acs_lines_model', 'mask_lines_grid', 'mask_lines_model', 'acq_width',
             'mask_lines_acq', 'R_eff', 'R_eff_acq', 'ours_grid_W', 'ours_lines', 'ours_R_eff', 'ours_acq_width',
             'ours_lines_acq', 'ours_R_eff_acq', 'ours_lines_on_native_grid', 'shared_physical_lines', 'mask_hex',
             'crop_note', 'n_nonfinite_slices', 'ls_ssim_mask', 'ls_psnr_mask', 'ls_nmse_mask', 'seconds']
SUMMARY_METRICS = ('ssim_mask', 'psnr_mask', 'nmse_mask', 'ssim_mask_volrange', 'ssim_off', 'psnr_off', 'nmse_off',
                   'ssim_off_in', 'ssim_off_out', 'frac_in')            # eval_v8fix_official.py:274-275
SUMMARY_LS = ('ls_ssim_mask', 'ls_psnr_mask', 'ls_nmse_mask')
COL_ZERO_RTOL = 1e-4       # '샘플하지 않은 줄은 0' 점검: 열 에너지 ≤ 이 비율 × 최대 열 에너지면 0 으로 본다(FFT 왕복 반올림 허용)


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def _identity(x):
    return x


def _worker_init(_):
    import torch
    torch.set_num_threads(1)


# ------------------------------------------------------------------ 공유 샘플 줄 (보충 메모 1)
def ours_mask_1d():
    """우리 평가 마스크 — 우리 로더의 함수 그대로 (FastMRI_H5_Dataloader(random_mask=False) 와 같은 호출)."""
    from dataloader_h5_v5 import build_r4_mask
    return build_r4_mask(GRID, CENTER_FRACTION, ACCEL, rng=None)


def native_center_lines(W):
    """중앙 줄 수 = round(W × 31/384), 반올림(0.5 는 올림), 정수 산술 (부동소수 오차 없음)."""
    return int(fractions.Fraction(W * OURS_CENTER, GRID) + fractions.Fraction(1, 2))


def native_mask_1d(W):
    """원 위상 부호화 폭 W 의 공유 마스크 → (mask (W,) float32, mapping, 중앙 줄 수).
    W == 384: 우리 로더가 그 축을 건드리지 않으므로 줄이 하나씩 대응 — 우리 마스크 그대로 ('exact').
    그 밖: 같은 규칙을 원 격자에 ('rule') — build_r4_mask(W, n/W, 4, rng=None): 중앙 n 줄(pad = (W−n+1)//2) + 3::4."""
    from dataloader_h5_v5 import build_r4_mask
    W = int(W)
    if W == GRID:
        return ours_mask_1d(), 'exact', OURS_CENTER
    n = native_center_lines(W)
    assert int(round(W * (n / W))) == n, (W, n)          # build_r4_mask 안의 int(round(W·cf)) 가 n 을 되돌리는지
    return build_r4_mask(W, n / W, ACCEL, rng=None), 'rule', n


def mask_hex(m):
    return np.packbits(np.asarray(m) > 0.5).tobytes().hex()


def physical_overlap(m384, m_nat, W):
    """우리 384 줄 i(주파수 (i−192)/384)와 원 줄 j(주파수 (j−W//2)/W)가 같은 물리적 줄인 경우를 정수 산술로 센다.
    → (우리 샘플 줄 가운데 원 격자 줄과 겹치는 수, 그 가운데 원 마스크에서도 샘플된 수)."""
    i = np.flatnonzero(np.asarray(m384) > 0.5).astype(np.int64)
    num = (i - GRID // 2) * int(W)
    on = (num % GRID) == 0
    j = num[on] // GRID + int(W) // 2
    ok = (j >= 0) & (j < W)
    j = j[ok]
    return int(ok.sum()), int((np.asarray(m_nat)[j] > 0.5).sum())


def window_on_384(W, pad_l, pad_r):
    """원 취득 구간 [pad_l, pad_r) 을 384 틀 줄로 옮긴 bool (384,): 주파수 (i−192)/384 가
    [(pad_l − W//2 − ½)/W, (pad_r − 1 − W//2 + ½)/W] 안이면 True (원 줄 칸의 경계까지)."""
    W = int(W)
    a, b = max(0, int(pad_l)), min(W, int(pad_r))
    i = np.arange(GRID, dtype=np.int64) - GRID // 2
    lo = (2 * (a - W // 2) - 1) * GRID
    hi = (2 * (b - 1 - W // 2) + 1) * GRID
    return (2 * W * i >= lo) & (2 * W * i <= hi)


def line_record(mode, W, m_grid, m_model, pad_l, pad_r, center_lines, mapping):
    """볼륨 한 개의 샘플 줄·실효 가속 계수 기록 (공개 모델 + 우리 모델 같은 볼륨)."""
    m384 = ours_mask_1d()
    gw = GRID if mode == 'ours' else int(W)
    if mode == 'ours':
        win = window_on_384(W, pad_l, pad_r)
    else:
        win = np.zeros(gw, bool)
        win[max(0, int(pad_l)):min(gw, int(pad_r))] = True
    n_grid = int((np.asarray(m_grid) > 0.5).sum())
    n_model = int((np.asarray(m_model) > 0.5).sum())
    n_acq = int(((np.asarray(m_grid) > 0.5) & win).sum())
    win384 = window_on_384(W, pad_l, pad_r)
    o_acq = int(((m384 > 0.5) & win384).sum())
    on, shared = physical_overlap(m384, m_grid if mode == 'native' else native_mask_1d(W)[0], W)
    return dict(grid_W=gw, mapping=mapping, center_lines=int(center_lines), mask_lines_grid=n_grid,
                mask_lines_model=n_model, acq_width=int(win.sum()), mask_lines_acq=n_acq,
                R_eff=gw / n_grid, R_eff_acq=(int(win.sum()) / n_acq) if n_acq else float('nan'),
                ours_grid_W=GRID, ours_lines=int((m384 > 0.5).sum()), ours_R_eff=GRID / float((m384 > 0.5).sum()),
                ours_acq_width=int(win384.sum()), ours_lines_acq=o_acq,
                ours_R_eff_acq=(int(win384.sum()) / o_acq) if o_acq else float('nan'),
                ours_lines_on_native_grid=on, shared_physical_lines=shared, mask_hex=mask_hex(m_grid))


def zero_cols_ok(col_energy, mask_bool):
    """샘플하지 않은 줄의 열 에너지가 0(상대 허용 COL_ZERO_RTOL)인지 → (ok, 샘플 줄 가운데 에너지 0 인 줄 수)."""
    e = np.asarray(col_energy, dtype=np.float64)
    top = float(e.max()) if e.size else 0.0
    nz = e > COL_ZERO_RTOL * top if top > 0 else np.zeros_like(e, bool)
    return bool(not (nz & ~mask_bool).any()), int((~nz & mask_bool).sum())


# ------------------------------------------------------------------ 출력 자르기 (각 모델의 test/predict 규칙)
def flair203_crop(out_w, crop_size):
    """fastmri 0.3.0 pl_modules/varnet_module.py:131-137 = PromptMR-plus pl_modules/promptmr_module.py:229-234:
    출력 폭이 crop_size[1] 보다 좁으면 (폭, 폭) — 'FLAIR 203' 규칙. 이 세 줄만 옮겨 적는다(Lightning 미설치)."""
    crop = (int(crop_size[0]), int(crop_size[1]))
    if out_w < crop[1]:
        crop = (int(out_w), int(out_w))
    return crop


def to_rss_shape(out2d, rss_hw):
    """모델 자체 자르기 뒤 남은 모양 차이를 정답(RSS) 크기로 맞춘다 → (배열, 기록). 보통은 이미 같다('own')."""
    from dataloader_h5_v5 import crop_or_pad_to
    h, w = out2d.shape[-2:]
    H, W = rss_hw
    if (h, w) == (H, W):
        return out2d, 'own'
    note = 'own+' + ('crop' if (h >= H and w >= W) else 'pad')
    return crop_or_pad_to(out2d, (H, W)), note


# ------------------------------------------------------------------ 헤더 (fastmri 의 메타데이터 함수 그대로)
def header_meta(fp):
    import h5py
    from pathlib import Path
    from fastmri.data.mri_data import SliceDataset
    meta, n = SliceDataset._retrieve_metadata(None, Path(fp))       # self 를 쓰지 않는 함수 (fastmri 0.3.0 mri_data.py:343)
    with h5py.File(fp, 'r') as f:
        ks = f['kspace'].shape
    acq = meta.get('acquisition', '')
    acq = acq.decode() if isinstance(acq, bytes) else str(acq)
    return dict(n=int(n), coils=int(ks[1]), kH=int(ks[2]), kW=int(ks[3]), pad_l=int(meta['padding_left']),
                pad_r=int(meta['padding_right']), recon_size=(int(meta['recon_size'][0]), int(meta['recon_size'][1])),
                enc_size=tuple(int(v) for v in meta['encoding_size']), acquisition=acq)


# ------------------------------------------------------------------ 모델
def _vm():
    """visualize_multimodel_compare (예전 val 평가의 로더·prep·forward). import 할 때 cwd 를 저장소 루트로 바꾸므로 경로를
    절대 경로로 바꾼 뒤에 부른다."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    import visualize_multimodel_compare as VM
    return VM


def load_model(key, weights, device):
    VM = _vm()
    fn = {'fastmri_unet': VM.load_unet, 'e2e_varnet': VM.load_varnet, 'promptmr_plus': VM.load_promptmr}[key]
    model = fn(weights, device)
    model.eval()
    return model


def pmr_modules():
    """PromptMR-plus 의 data 패키지 (VM.load_promptmr 가 sys.path[0] 에 PMR_ROOT 를 넣고 models 를 바꿔 끼운 뒤)."""
    import importlib
    if PMR_ROOT not in sys.path:
        sys.path.insert(0, PMR_ROOT)
    PMT = importlib.import_module('data.transforms')
    PMD = importlib.import_module('data.mri_data')
    PMU = importlib.import_module('models.utils')
    for m in (PMT, PMD, PMU):
        assert os.path.abspath(m.__file__).startswith(os.path.abspath(PMR_ROOT)), (m.__name__, m.__file__)
    return PMT, PMD, PMU


# ------------------------------------------------------------------ native 방식: 각 모델의 데이터 변환
class FixedMaskFunc:
    """fastmri MaskFunc 인터페이스 mask_func(shape, offset, seed) → (mask (1,…,W,1) float32, num_low_frequencies).
    매번 새 텐서를 돌려준다 (apply_mask 가 padding 을 제자리에서 0 으로 바꾸므로)."""

    def __init__(self, mask_1d, num_low):
        self.mask = np.asarray(mask_1d, dtype=np.float32)
        self.num_low = int(num_low)

    def __call__(self, shape, offset=None, seed=None):
        import torch
        assert int(shape[-2]) == self.mask.shape[0], (shape, self.mask.shape)
        ms = [1] * len(shape)
        ms[-2] = self.mask.shape[0]
        return torch.from_numpy(self.mask.reshape(*ms).copy()), self.num_low


class NativeTransform:
    """SliceDataset / FastmriSliceDataset 의 transform 자리에 들어가 각 모델의 원 변환을 부른다."""

    def __init__(self, key):
        self.key = key
        self.pmr_tf = None

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        W = int(kspace.shape[-1])
        m, mapping, n_low = native_mask_1d(W)
        pad_l, pad_r = int(attrs['padding_left']), int(attrs['padding_right'])
        out = dict(fname=str(fname), slice_num=int(slice_num), mapping=mapping, center_lines=int(n_low),
                   mask_grid=m, kW=W, kH=int(kspace.shape[-2]), coils_used=int(kspace.shape[0]))
        if self.key in ('fastmri_unet', 'e2e_varnet'):
            from fastmri.data import transforms as T
            mf = FixedMaskFunc(m, n_low)
            if self.key == 'fastmri_unet':
                smp = T.UnetDataTransform('multicoil', mask_func=mf, use_seed=True)(kspace, mask, target, attrs, fname,
                                                                                     slice_num)
                # 점검: UnetDataTransform 안의 apply_mask(padding 없음)를 같은 마스크 함수로 다시 불러 씌운 마스크·열을 본다
                mk_t, mm_t, _ = T.apply_mask(T.to_tensor(kspace), mf)
                mm = mm_t.reshape(-1).numpy() > 0.5
                cols_ok, zero_sampled = zero_cols_ok(mk_t.pow(2).sum(dim=-1).sum(dim=(0, 1)).numpy(), mm)
                out.update(image=smp.image, mean=float(smp.mean), std=float(smp.std), mask_model=mm.astype(np.float32),
                           acs_lines_model=-1, mask_ok=bool(np.array_equal(mm, m > 0.5)), cols_ok=cols_ok,
                           sampled_zero_cols=zero_sampled)
                return out
            smp = T.VarNetDataTransform(mask_func=mf, use_seed=True)(kspace, mask, target, attrs, fname, slice_num)
            num_low = int(smp.num_low_frequencies)
            crop_size = (int(smp.crop_size[0]), int(smp.crop_size[1]))
        else:
            if self.pmr_tf is None:
                PMT, _, _ = pmr_modules()
                self.pmr_tf = PMT.FastmriDataTransform(**PMR_INFER_CFG)
            ksp_m = (kspace * m[np.newaxis, np.newaxis, :]).astype(np.complex64)   # 시험 분할 파일처럼 샘플 안 한 줄 = 0
            smp = self.pmr_tf(ksp_m, m, None, attrs, fname, slice_num)
            num_low = int(smp.num_low_frequencies)
            crop_size = (int(smp.crop_size[0]), int(smp.crop_size[1]))
            out['mask_type'] = str(smp.mask_type)
            out['kH_model'] = int(smp.masked_kspace.shape[-3])
        mm = smp.mask.reshape(-1).numpy().astype(bool)
        exp = m.copy()                     # 두 변환 모두 같은 슬라이싱으로 취득 구간 밖을 0 으로 (fastmri transforms.py:79-80,
        exp[:pad_l] = 0                    #   VarNetDataTransform 의 apply_mask(padding) / PromptMR-plus transforms.py:526-527)
        exp[pad_r:] = 0
        col = smp.masked_kspace.pow(2).sum(dim=-1).sum(dim=(0, 1))   # (W,) 열 에너지
        cols_ok, zero_sampled = zero_cols_ok(col.numpy(), mm)
        out.update(masked_kspace=smp.masked_kspace, mask=smp.mask, num_low=num_low, crop_size=crop_size,
                   mask_model=mm.astype(np.float32), mask_ok=bool(np.array_equal(mm, exp > 0.5)), cols_ok=cols_ok,
                   sampled_zero_cols=zero_sampled, acs_lines_model=num_low)
        return out


class FileFilter:
    def __init__(self, names):
        self.names = set(names)

    def __call__(self, raw_sample):
        return raw_sample[0].name in self.names


class NativeItems:
    """(모델 데이터셋 항목, 우리 로더 항목) 짝 — 우리 로더에서는 384 틀 label·brain mask 만 쓴다."""

    def __init__(self, model_ds, ours_ds, pairs):
        self.model_ds, self.ours_ds, self.pairs = model_ds, ours_ds, list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        mi, oi = self.pairs[k]
        item = self.model_ds[mi]
        fp, si, _ = self.ours_ds.samples[oi]
        assert item['fname'] == os.path.basename(fp) and item['slice_num'] == int(si), (item['fname'], item['slice_num'], fp, si)
        s = self.ours_ds[oi]
        item.update(idx=int(oi), label=s['label'][0].astype(np.float32), brain=s['brain_mask'][0].astype(np.float32))
        return item


# ------------------------------------------------------------------ ours 방식: 예전 val 평가 입력
class OursItems:
    def __init__(self, key, ours_ds, indices, num_adj):
        self.key, self.ds, self.indices, self.num_adj = key, ours_ds, list(indices), int(num_adj)
        self.m384 = ours_mask_1d() > 0.5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        VM = _vm()
        idx = self.indices[k]
        s = self.ds[idx]
        m_s = VM._mask_1d(s) > 0.5
        mask_ok = bool(np.array_equal(m_s, self.m384))
        ksp_all = VM.unpack_complex(s['data'])
        cols_ok, zero_sampled = zero_cols_ok((np.abs(ksp_all) ** 2).sum(axis=(0, 1)), m_s)
        scale_relerr = 0.0
        if self.key == 'fastmri_unet':
            inp = VM.prep_unet(s)
            coils = int(VM._kspace_keep_coils(s['data'])[1].sum())
            out_to_label = VAL_AMP_Y                              # 출력 = RSS 단위 (zero-filled RSS 의 mean/std 로 되돌림)
            acs = -1
        else:
            if self.key == 'e2e_varnet':
                inp = VM.prep_varnet(s)
                raw, keep = VM._kspace_keep_coils(s['data'])
                mask_ok = mask_ok and bool(np.array_equal(np.asarray(inp['mask1d']).astype(bool), self.m384))
            else:
                nb_idx = VM.neighbor_indices(self.ds, idx, self.num_adj)
                nbs = [s if i == idx else self.ds[i] for i in nb_idx]
                inp = VM.prep_promptmr(s, nbs)
                _, keep = VM._kspace_keep_coils(s['data'])
                raw = np.concatenate([VM._kspace_keep_coils(nb['data'], keep)[0] for nb in nbs], axis=0)
                mask_ok = mask_ok and bool(np.array_equal(np.asarray(inp['mask1d']).astype(bool), self.m384)) \
                    and all(np.array_equal(VM._mask_1d(nb) > 0.5, self.m384) for nb in nbs)
                inp['neighbors'] = [int(i) for i in nb_idx]
            sc = float(np.abs(raw).max()) + 1e-12                 # VM.prep_varnet / prep_promptmr 와 같은 식 (나눈 값)
            t = inp['ksp'][0].numpy()
            back = (t[..., 0] + 1j * t[..., 1]) * sc
            scale_relerr = float(np.abs(back - raw).max() / max(float(np.abs(raw).max()), 1e-30))
            out_to_label = sc * VAL_AMP_Y / AMP_X_KSP            # 출력 × (sc / 1e4) = RSS 단위, × 1e6 = label 단위
            coils = int(inp['coils_kept'])
            acs = int(round(GRID * CENTER_FRACTION))              # VM.forward_varnet / forward_promptmr 가 넘기는 값
        inp.update(idx=int(idx), label=s['label'][0].astype(np.float32), brain=s['brain_mask'][0].astype(np.float32),
                   out_to_label=float(out_to_label), coils_used=coils, mask_ok=mask_ok, cols_ok=cols_ok,
                   sampled_zero_cols=zero_sampled, scale_relerr=scale_relerr, acs_lines_model=acs,
                   mask_model=self.m384.astype(np.float32))
        return inp


# ------------------------------------------------------------------ 순전파
def forward_native(key, model, item, device, pmr_sens_per_coil):
    import torch
    if key == 'fastmri_unet':
        x = torch.as_tensor(item['image']).float()[None, None].to(device)
        out = model(x)[0, 0]                                       # UnetModule.forward: unet(image.unsqueeze(1)).squeeze(1)
        out = out * item['std'] + item['mean']                     # UnetModule.test_step (fastmri pl_modules/unet_module.py:107-116)
        return out.float().cpu().numpy()
    mk = item['masked_kspace'].float()[None].to(device)
    mask = item['mask'][None].to(device)
    if key == 'e2e_varnet':
        from fastmri.data import transforms as T
        out = model(mk, mask, item['num_low'])                    # VarNetModule.test_step (pl_modules/varnet_module.py:128-137)
        out = T.center_crop(out, flair203_crop(out.shape[-1], item['crop_size']))
        return out[0].float().cpu().numpy()
    PMT, _, _ = pmr_modules()
    nlf = torch.tensor([item['num_low']], device=device)
    out = model(mk, mask, nlf, mask_type=(item['mask_type'],), compute_sens_per_coil=pmr_sens_per_coil)['img_pred']
    out = PMT.center_crop(out, flair203_crop(out.shape[-1], item['crop_size']))   # PromptMrModule.predict_step
    return out[0].float().cpu().numpy()


def forward_ours(key, model, item, device):
    VM = _vm()
    if key == 'fastmri_unet':
        return VM.forward_unet(model, item, device)
    if key == 'e2e_varnet':
        return VM.forward_varnet(model, item, device)
    return VM.forward_promptmr(model, item, device)


def pmr_detected_acs(mask_bool_1d):
    """PromptMR+ 가 num_low_frequencies=−1 일 때 마스크에서 검출하는 ACS 줄 수 (그들의 KspaceACSExtractor 함수 그대로)."""
    import torch
    _, _, PMU = pmr_modules()
    m = torch.from_numpy(np.asarray(mask_bool_1d) > 0.5).view(1, 1, 1, -1, 1)
    _, n = PMU.KspaceACSExtractor(True).get_pad_and_num_low_freqs(m, torch.tensor([-1]))
    return int(n[0])


# ------------------------------------------------------------------ 지표 (우리 모델과 같은 함수)
def ls_block(rec, gt, bm):
    """예전 val 평가(eval_baselines_full.py:187-194)와 같은 보조 지표 — 최소제곱 강도 배율 보정 뒤 brain mask 지표."""
    VM = _vm()
    rec = np.nan_to_num(np.asarray(rec, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    m = bm > 0.5
    r, g = rec[m], gt[m]
    denom = float((r * r).sum())
    alpha = float((r * g).sum()) / denom if (m.any() and denom >= 1e-12) else 1.0
    rec = VM.ls_scale(rec, gt, bm)
    met = VM.slice_metrics_np(rec, gt, bm)
    met['l1'] = float((np.abs(rec - gt) * bm).sum() / max(float(bm.sum()), 1.0))
    return alpha, met


def volume_metrics(gt, preds_rss, labels384, bms384, pred384_full, O, slice_metrics, device):
    """eval_v8fix_official.main 의 볼륨 절차 그대로 → (볼륨 행, 슬라이스 지표 목록).
    gt (S,H,W) RSS · preds_rss (S,H,W) RSS 단위 · labels384/bms384 (S,384,384) · pred384_full: 마스크 지표에 쓸 384 출력
    (label 단위; ours = 모델 384 출력 전체, native = 정답 크기 출력을 0 으로 채워 넣은 것)."""
    import torch
    from dataloader_h5_v5 import crop_or_pad_to
    S, H, W = gt.shape
    sl = []
    for s in range(S):
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None].to(device)  # noqa: E731
        m = slice_metrics(to_t(pred384_full[s]), to_t(labels384[s]), to_t(bms384[s]))
        sl.append({k: float(m[k]) for k in ('ssim', 'psnr', 'nmse', 'l1')})
    pred = np.stack(preds_rss).astype(np.float32)
    bm384 = np.stack(bms384) > 0.5
    lab = np.stack(labels384)
    bm_hw = O.center_crop_np(bm384, (H, W))
    lab_check = float(np.max(np.abs(O.center_crop_np(lab, (H, W)) / VAL_AMP_Y - gt)) / max(gt.max(), 1e-30))
    off = O.official_volume(gt, pred, bm_hw)
    pred384z = crop_or_pad_to(pred * VAL_AMP_Y, (GRID, GRID))        # = eval_v8fix_official.py:240-242 (같은 오프셋)
    vr = O.masked_ssim_volrange(lab, pred384z, bm384)
    for r, v in zip(sl, off['ssim_slices']):
        r['ssim_official'] = float(v)
    row = dict(n=S, H=H, W=W,
               ssim_mask=float(np.mean([r['ssim'] for r in sl])), psnr_mask=float(np.mean([r['psnr'] for r in sl])),
               nmse_mask=float(np.mean([r['nmse'] for r in sl])), ssim_mask_volrange=vr,
               ssim_off=off['ssim'], psnr_off=off['psnr'], nmse_off=off['nmse'], maxval=off['maxval'],
               ssim_off_in=off['ssim_in'], ssim_off_out=off['ssim_out'], frac_in=off['frac_in'],
               decomp_maxabs=off['decomp_maxabs'], label_vs_rss_relmax=lab_check)
    return row, sl


# ------------------------------------------------------------------ 기록
def provenance(weights_path, weights_sha):
    import torch
    import importlib.metadata as md
    mods = {}
    try:
        import fastmri
        fm_dir = os.path.dirname(os.path.abspath(fastmri.__file__))
    except Exception:
        fm_dir = None
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if not f or not f.endswith('.py'):
            continue
        f = os.path.abspath(f)
        if f.startswith(_ROOT + os.sep):
            key = os.path.relpath(f, _ROOT)
        elif fm_dir and f.startswith(fm_dir + os.sep):
            key = 'site-packages/fastmri/' + os.path.relpath(f, fm_dir)
        else:
            continue
        if key not in mods:
            mods[key] = _sha256(f)

    def ver(p):
        try:
            return md.version(p)
        except Exception:
            return None

    def git_head(d):
        try:
            return subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                  timeout=20).stdout.strip() or None
        except Exception:
            return None
    plan = {}
    for p in PLAN_FILES:
        fp = os.path.join(_ROOT, p)
        plan[p] = _sha256(fp) if os.path.exists(fp) else None
    return dict(
        script=dict(path=os.path.relpath(_THIS, _ROOT), sha256=_sha256(_THIS)),
        weights=dict(path=weights_path, sha256=weights_sha, bytes=os.path.getsize(weights_path)),
        code_sha256=mods, repo_git_head=git_head(_ROOT), promptmr_plus_git_head=git_head(PMR_ROOT),
        packages=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(), numpy=np.__version__, fastmri=ver('fastmri'),
                      scikit_image=ver('scikit-image'), scipy=ver('scipy'), h5py=ver('h5py'), einops=ver('einops')),
        plan_sha256=plan)


def _fmt_row(r, cols):
    return {c: r.get(c, '') for c in cols}


def write_csv(path, rows, cols):
    with open(path, 'x', newline='') as f:                      # 'x' = 있으면 실패 (덮어쓰기 금지)
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(_fmt_row(r, cols))


def write_json(path, obj):
    with open(path, 'x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


def out_paths(out_dir, stem):
    return dict(per_volume=os.path.join(out_dir, f'per_volume_{stem}.csv'),
                per_slice=os.path.join(out_dir, f'per_slice_{stem}.csv'),
                summary=os.path.join(out_dir, f'summary_{stem}.json'),
                defect=os.path.join(out_dir, f'{stem}.DEFECT'),
                partial=os.path.join(out_dir, f'partial_{stem}.jsonl'),
                dump=os.path.join(out_dir, f'dump_{stem}'))


def read_partial(path):
    """진행 기록 읽기 → (기록 목록, 읽지 못한 줄 수, 끝이 줄바꿈인지). 중단으로 잘린 마지막 줄은 건너뛴다(볼륨을 다시 계산)."""
    with open(path) as f:
        raw = f.read()
    recs, bad = [], 0
    for x in raw.split('\n'):
        if not x.strip():
            continue
        try:
            recs.append(json.loads(x))
        except json.JSONDecodeError:
            bad += 1
    return recs, bad, (raw == '' or raw.endswith('\n'))


def refuse_existing(paths, resume, dump):
    """덮어쓰기 거부. 끝난 출력이 하나라도 있으면 거부. 진행 기록(partial)은 --resume 일 때만 이어 쓴다."""
    final = [paths[k] for k in ('per_volume', 'per_slice', 'summary', 'defect') if os.path.exists(paths[k])]
    if final:
        return f'출력이 이미 있음 (덮어쓰지 않음): {final}'
    if os.path.exists(paths['partial']) and not resume:
        return f'진행 기록이 이미 있음: {paths["partial"]} — 이어서 하려면 --resume, 새로 하려면 다른 --out-dir'
    if dump and os.path.exists(paths['dump']) and not resume:
        return f'덤프 폴더가 이미 있음: {paths["dump"]}'
    return None


# ------------------------------------------------------------------ 본체
def run(a):
    import torch
    t_start = time.time()
    started = _utc()
    key, mode = a.model, a.mode
    spec = MODELS[key]
    stem = f'{key}_{mode}'
    a.data_path = os.path.abspath(a.data_path)            # 모두 절대 경로로 (VM import 가 cwd 를 바꾸기 전)
    a.out_dir = os.path.abspath(a.out_dir)
    if a.expect_files_from:
        a.expect_files_from = os.path.abspath(a.expect_files_from)
    weights = os.path.abspath(a.weights or os.path.join(_ROOT, spec['weights']))
    paths = out_paths(a.out_dir, stem)
    msg = refuse_existing(paths, a.resume, a.dump_volumes > 0)
    if msg:
        print('[거부] ' + msg, flush=True)
        return 2
    hook =sorted({int(x) for x in a.test_inject_nonfinite.split(',') if x.strip()}) if a.test_inject_nonfinite else []
    if hook:
        print(f'[경고] 시험 후크 켜짐: idx {hook} 의 출력 첫 화소를 NaN 으로 바꾼다 (실제 평가에서는 쓰지 않는다)', flush=True)

    # 정밀도 — fp32, TF32 끔(기본)
    if a.tf32 == 'off':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tf32 = dict(option=a.tf32, matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn=bool(torch.backends.cudnn.allow_tf32), float32_matmul_precision=torch.get_float32_matmul_precision())
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('[거부] CUDA 를 쓸 수 없음 (--device cpu 로 지정)', flush=True)
        return 2

    # 데이터 (우리 평가 데이터셋 = 순서·label·brain mask 의 단일 출처) — 파일 목록 점검이 실패하면 모델을 올리기 전에 거부
    import eval_unet_only_v8 as E
    import eval_v8fix_official as O
    from eval_paired_v8_nodc import slice_metrics
    VM = _vm()
    assert (VM.ACCEL, VM.CENTER_FRACTION, E.C.IMAGE_SIZE[0]) == (ACCEL, CENTER_FRACTION, GRID)
    ds = E.build_val_dataset(a.data_path)
    files = []
    for i, (fp, s, _) in enumerate(ds.samples):
        if not files or files[-1][0] != fp:
            files.append([fp, 0, i])
        files[-1][1] += 1
    n_all = len(files)
    if a.max_files > 0:
        files = files[:a.max_files]
    names = [os.path.basename(fp) for fp, _, _ in files]
    names_sha = hashlib.sha256('\n'.join(names).encode()).hexdigest()
    expect = None
    if a.expect_files_from:
        with open(a.expect_files_from, newline='') as f:
            want = sorted({r['file'] for r in csv.DictReader(f)})
        have_all = sorted(os.path.basename(fp) for fp in ds.files)
        expect = dict(csv=os.path.abspath(a.expect_files_from), n_csv=len(want), n_data=len(have_all),
                      equal=(want == have_all), missing_in_data=sorted(set(want) - set(have_all))[:20],
                      extra_in_data=sorted(set(have_all) - set(want))[:20])
        if not expect['equal']:
            print(f'[거부] 데이터 파일 목록이 {a.expect_files_from} 와 다름: {expect}', flush=True)
            return 2
    print(f'[{stem}] 볼륨 {len(files)} / {n_all}, 슬라이스 {sum(n for _, n, _ in files)}', flush=True)

    # 가중치 · 모델
    print(f'[{stem}] 가중치 sha256 계산: {weights}', flush=True)
    w_sha = _sha256(weights)
    ckpt_meta = {}
    if key == 'promptmr_plus':
        obj = torch.load(weights, map_location='cpu', weights_only=False)
        ckpt_meta = dict(epoch=obj.get('epoch'), global_step=obj.get('global_step'),
                         num_adj_slices=obj['hyper_parameters'].get('num_adj_slices'),
                         pytorch_lightning_version=obj.get('pytorch-lightning_version'))
        del obj
        exp = spec['ckpt_expect']
        if (ckpt_meta['epoch'], ckpt_meta['global_step']) != (exp['epoch'], exp['global_step']):
            print(f'[거부] PromptMR+ ckpt 가 계획의 {spec["upstream_name"]} 와 다름: {ckpt_meta}', flush=True)
            return 2
        if ckpt_meta['num_adj_slices'] != PMR_NUM_ADJ:
            print(f'[거부] num_adj_slices {ckpt_meta["num_adj_slices"]} ≠ 추론 설정 {PMR_NUM_ADJ}', flush=True)
            return 2
    t0 = time.time()
    model = load_model(key, weights, device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'[{stem}] {spec["display"]}  params {n_par / 1e6:.1f}M  로드 {time.time() - t0:.0f}s  device {device}  TF32 {tf32}',
          flush=True)

    # 진행 기록 (header + 볼륨 줄)
    header = dict(type='header', version=1, model=key, mode=mode, weights_sha256=w_sha, script_sha256=_sha256(_THIS),
                  data_path=a.data_path, files=names, tf32=a.tf32, device_type=device.type,
                  pmr_sens_per_coil=bool(a.pmr_sens_per_coil), test_hook=hook)
    done = {}
    if os.path.exists(paths['partial']):
        lines, n_bad_lines, ends_nl = read_partial(paths['partial'])
        old = lines[0] if lines and lines[0].get('type') == 'header' else {}
        diff = [k for k in header if k != 'type' and old.get(k) != header[k]]
        if diff:
            print(f'[거부] --resume: 진행 기록의 설정이 지금과 다름 ({diff})', flush=True)
            return 2
        for rec in lines[1:]:
            if rec.get('type') == 'volume':
                done[rec['row']['file']] = rec
        if not ends_nl:                                          # 잘린 줄 뒤에 새 기록이 붙지 않도록 줄을 바꿔 둔다(덧붙이기만)
            with open(paths['partial'], 'a') as f:
                f.write('\n')
        print(f'[{stem}] --resume: 끝난 볼륨 {len(done)} 개를 건너뛴다 (읽지 못한 줄 {n_bad_lines})', flush=True)
    else:
        os.makedirs(a.out_dir, exist_ok=True)
        with open(paths['partial'], 'x') as f:
            f.write(json.dumps(header, ensure_ascii=False) + '\n')
    todo = [(fp, n, i0) for fp, n, i0 in files if os.path.basename(fp) not in done]

    # 항목 데이터셋
    num_adj = PMR_NUM_ADJ
    if mode == 'native':
        tf = NativeTransform(key)
        sel = FileFilter([os.path.basename(fp) for fp, _, _ in todo])
        if key == 'promptmr_plus':
            _, PMD, _ = pmr_modules()
            mds = PMD.FastmriSliceDataset(a.data_path, 'multicoil', transform=tf, use_dataset_cache=False,
                                          raw_sample_filter=sel, num_adj_slices=num_adj)
        else:
            from fastmri.data.mri_data import SliceDataset
            mds = SliceDataset(a.data_path, 'multicoil', transform=tf, use_dataset_cache=False, raw_sample_filter=sel)
        want = [(os.path.basename(fp), s) for fp, n, i0 in todo for s in range(n)]
        got = [(rs[0].name, int(rs[1])) for rs in mds.raw_samples]
        if want != got:
            raise RuntimeError(f'모델 데이터셋 순서가 우리 데이터셋과 다름 (앞 5개: {got[:5]} vs {want[:5]})')
        pairs = list(zip(range(len(got)), [i0 + s for fp, n, i0 in todo for s in range(n)]))
        items = NativeItems(mds, ds, pairs)
    else:
        items = OursItems(key, ds, [i0 + s for fp, n, i0 in todo for s in range(n)], num_adj)
    from torch.utils.data import DataLoader
    loader = DataLoader(items, batch_size=None, shuffle=False, num_workers=a.num_workers, collate_fn=_identity,
                        worker_init_fn=_worker_init, prefetch_factor=(2 if a.num_workers > 0 else None))

    # 볼륨 루프
    import h5py
    it = iter(loader)
    for vi, (fp, n, i0) in enumerate(todo):
        tv = time.time()
        hm = header_meta(fp)
        with h5py.File(fp, 'r') as f:
            gt = f['reconstruction_rss'][()].astype(np.float32)
        S, H, W = gt.shape
        assert S == n == hm['n'], (fp, S, n, hm['n'])
        if H > GRID or W > GRID:
            raise RuntimeError(f'정답 크기 {H}×{W} > {GRID}: 384 틀 지표를 계산할 수 없음 ({fp})')
        preds_rss, labels, bms, p384, srows = [], [], [], [], []
        crop_notes = set()
        for s in range(n):
            item = next(it)
            idx = int(item['idx'])
            assert idx == i0 + s, (idx, i0, s)
            if not item['mask_ok']:
                raise RuntimeError(f'마스크 점검 실패 (idx {idx}, {mode}) — 공유 마스크와 다름')
            if not item['cols_ok']:
                raise RuntimeError(f'열 점검 실패 (idx {idx}, {mode}) — 샘플하지 않은 줄에 k-space 값이 있음')
            t1 = time.time()
            with torch.no_grad():
                if mode == 'native':
                    out = forward_native(key, model, item, device, a.pmr_sens_per_coil)
                else:
                    out = forward_ours(key, model, item, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            sec = time.time() - t1
            out = np.array(out, dtype=np.float32, copy=True)
            if idx in hook:
                out.reshape(-1)[0] = np.nan
            n_bad = int((~np.isfinite(out)).sum())
            label = np.asarray(item['label'], dtype=np.float32)
            brain = np.asarray(item['brain'], dtype=np.float32)
            if mode == 'native':
                pr, note = to_rss_shape(out, (H, W))                     # RSS 단위
                crop_notes.add(note)
                from dataloader_h5_v5 import crop_or_pad_to
                full384 = crop_or_pad_to(pr * VAL_AMP_Y, (GRID, GRID))
                o2l = VAL_AMP_Y
                ls_alpha, ls = ls_block(full384, label, brain)
            else:
                assert out.shape == (GRID, GRID), out.shape
                o2l = float(item['out_to_label'])
                full384 = out * o2l                                      # label 단위 (우리 모델 출력과 같은 단위)
                pr = O.center_crop_np(full384, (H, W)) / VAL_AMP_Y      # = eval_v8fix_official.py:229
                crop_notes.add('ours384')
                ls_alpha, ls = ls_block(out, label, brain)              # 예전 평가와 같은 입력(배율 되돌리기 전 출력)
            preds_rss.append(pr)
            labels.append(label)
            bms.append(brain)
            p384.append(full384)
            srows.append(dict(idx=idx, file=os.path.basename(fp), slice_idx=s, finite=int(n_bad == 0), n_nonfinite=n_bad,
                              ls_alpha=ls_alpha, ls_ssim=ls['ssim'], ls_psnr=ls['psnr'], ls_nmse=ls['nmse'],
                              ls_l1=ls['l1'], out_to_label=o2l, coils_used=int(item['coils_used']), sec=round(sec, 3),
                              _mask_model=np.asarray(item['mask_model']), _acs=int(item['acs_lines_model']),
                              _zero_sampled=int(item['sampled_zero_cols']), _scale_relerr=float(item.get('scale_relerr', 0.0)),
                              _mapping=item.get('mapping', 'ours384'), _center=int(item.get('center_lines', OURS_CENTER)),
                              _mask_grid=np.asarray(item.get('mask_grid', item['mask_model']))))
        row, sl = volume_metrics(gt, preds_rss, labels, bms, p384, O, slice_metrics, device)
        for r, m in zip(srows, sl):
            r.update(m)
        # 볼륨 안에서 마스크가 같아야 한다 (같은 폭)
        mg = srows[0]['_mask_grid']
        mm = srows[0]['_mask_model']
        assert all(np.array_equal(r['_mask_grid'], mg) and np.array_equal(r['_mask_model'], mm) for r in srows)
        if mode == 'native':
            assert hm['kW'] == mg.shape[0], (hm['kW'], mg.shape)
            lr = line_record('native', hm['kW'], mg, mm, hm['pad_l'], hm['pad_r'], srows[0]['_center'], srows[0]['_mapping'])
        else:
            lr = line_record('ours', hm['kW'], mg, mm, hm['pad_l'], hm['pad_r'], OURS_CENTER, 'ours384')
        acs = srows[0]['_acs']
        if key == 'promptmr_plus' and mode == 'native':
            acs = pmr_detected_acs(mm)                                    # num_low −1 → 그들 함수로 검출한 값
        row.update(file=os.path.basename(fp), acquisition=hm['acquisition'], coils_file=hm['coils'],
                   coils_used=max(r['coils_used'] for r in srows), kspace_H=hm['kH'], kspace_W=hm['kW'],
                   recon_size=f'{hm["recon_size"][0]}x{hm["recon_size"][1]}', pad_left=hm['pad_l'], pad_right=hm['pad_r'],
                   acs_lines_model=acs, crop_note='+'.join(sorted(crop_notes)),
                   n_nonfinite_slices=sum(1 for r in srows if not r['finite']),
                   ls_ssim_mask=float(np.mean([r['ls_ssim'] for r in srows])),
                   ls_psnr_mask=float(np.mean([r['ls_psnr'] for r in srows])),
                   ls_nmse_mask=float(np.mean([r['ls_nmse'] for r in srows])),
                   seconds=round(time.time() - tv, 1), **lr)
        checks = dict(sampled_zero_cols_max=max(r['_zero_sampled'] for r in srows),
                      scale_relerr_max=max(r['_scale_relerr'] for r in srows))
        slim = [{k: v for k, v in r.items() if not k.startswith('_')} for r in srows]
        with open(paths['partial'], 'a') as f:
            f.write(json.dumps(dict(type='volume', row=row, slices=slim, checks=checks), ensure_ascii=False) + '\n')
        if a.dump_volumes > 0 and vi < a.dump_volumes:
            os.makedirs(paths['dump'], exist_ok=True)
            np.savez_compressed(os.path.join(paths['dump'], os.path.basename(fp).replace('.h5', '.npz')),
                                gt_rss=gt, pred_rss=np.stack(preds_rss), pred384=np.stack(p384), label384=np.stack(labels),
                                brain384=np.stack(bms), mask_grid=mg, mask_model=mm)
        el = time.time() - t_start
        print(f'  [{stem}] {vi + 1}/{len(todo)} {os.path.basename(fp)} W{hm["kW"]} {row["mapping"]} lines '
              f'{row["mask_lines_grid"]}/{row["grid_W"]} | SSIM mask {row["ssim_mask"]:.4f} off {row["ssim_off"]:.4f} '
              f'PSNR off {row["psnr_off"]:.2f} | non-finite {row["n_nonfinite_slices"]} | {el / 60:.1f} 분', flush=True)

    # 마무리: 진행 기록 전체(이어 쓴 것 포함)에서 출력 만들기
    recs, n_bad_lines, _ = read_partial(paths['partial'])
    by_file = {r['row']['file']: r for r in recs if r.get('type') == 'volume'}
    missing = [nm for nm in names if nm not in by_file]
    if missing:
        raise RuntimeError(f'끝나지 않은 볼륨: {missing[:5]} …')
    vol_rows = [by_file[nm]['row'] for nm in names]
    sl_rows = [r for nm in names for r in by_file[nm]['slices']]
    vcols = VOL_COLS + VOL_EXTRA
    scols = SLICE_COLS + SLICE_EXTRA

    def ms(k):
        v = np.array([float(r[k]) for r in vol_rows], dtype=float)
        return dict(mean=float(np.nanmean(v)) if np.isfinite(v).any() else None,
                    sd=float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else None,
                    n_finite=int(np.isfinite(v).sum()))
    bad = [dict(idx=r['idx'], file=r['file'], slice_idx=r['slice_idx'], n_nonfinite=r['n_nonfinite'])
           for r in sl_rows if not int(r['finite'])]
    defect = bool(bad)
    reasons = ([f'{len(bad)} slice(s) with non-finite fp32 output (public model already fp32: not recomputed, '
                f'reported as an evaluation defect; plan section 5 / analysis plan amendment 3 section 2)'] if bad else [])
    exit_code = 3 if defect else 0
    rv = np.array([r['R_eff'] for r in vol_rows], float)
    rva = np.array([r['R_eff_acq'] for r in vol_rows], float)
    orva = np.array([r['ours_R_eff_acq'] for r in vol_rows], float)
    mapping_counts = {}
    for r in vol_rows:
        mapping_counts[r['mapping']] = mapping_counts.get(r['mapping'], 0) + 1
    summ = dict(
        model=key, model_display=spec['display'], mode=mode, status='completed', completed=True,
        defect=defect, defect_reasons=reasons, exit_code=exit_code, nonfinite_slices=bad,
        test_hook=dict(active=bool(hook), inject_nonfinite_idx=hook),
        external_baseline=dict(trained_on=spec['trained_on'], wording=EXTERNAL_WORDING),
        weights_file=os.path.relpath(weights, _ROOT) if weights.startswith(_ROOT) else weights,
        weights_upstream_name=spec.get('upstream_name'), ckpt_meta=ckpt_meta,
        data_path=a.data_path, n_files_in_data=n_all, max_files=a.max_files, n_vol=len(vol_rows), n_slice=len(sl_rows),
        file_names_sha256=names_sha, expect_files=expect,
        device=str(device), gpu=(torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
        precision='fp32 (no autocast)', tf32=tf32, pmr_sens_per_coil=(bool(a.pmr_sens_per_coil) if key == 'promptmr_plus' else None),
        seconds=round(time.time() - t_start, 1), resumed_volumes=len(done), started_utc=started, finished_utc=_utc(),
        argv=sys.argv[1:],
        metrics={k: ms(k) for k in SUMMARY_METRICS}, metrics_ls_aux={k: ms(k) for k in SUMMARY_LS},
        effective_acceleration=dict(
            public_grid=('native phase-encoding width' if mode == 'native' else f'{GRID} frame (our preprocessing)'),
            public_mean_R_eff=float(np.mean(rv)), public_mean_R_eff_acq=float(np.nanmean(rva)),
            public_lines_mean=float(np.mean([r['mask_lines_grid'] for r in vol_rows])),
            public_ratio_of_sums=float(sum(r['grid_W'] for r in vol_rows) / sum(r['mask_lines_grid'] for r in vol_rows)),
            ours_R_eff=GRID / OURS_LINES, ours_mean_R_eff_acq=float(np.nanmean(orva)),
            note=('R_eff = grid width / sampled lines on that grid; R_eff_acq = acquired lines (ismrmrd '
                  'kspace_encoding_step_1 window [padding_left, padding_right)) / sampled lines inside that window; for our '
                  'models the window is mapped to the 384 frame by physical frequency')),
        mask=dict(ours_384=dict(lines=OURS_LINES, center=OURS_CENTER, offset=OFFSET, R=ACCEL, hex=mask_hex(ours_mask_1d())),
                  native_rule='equispaced R=4, centre lines round(W*31/384) (half up), start 3, build_r4_mask(W, n/W, 4, rng=None)',
                  exact_condition='native phase-encoding width W == 384 (loader crops/pads the PE axis in the image domain)',
                  mapping_counts=mapping_counts),
        checks=dict(decomp_maxabs=max(r['decomp_maxabs'] for r in vol_rows),
                    label_vs_rss_relmax=max(r['label_vs_rss_relmax'] for r in vol_rows),
                    mask_and_columns='every slice: mask passed to the model = shared mask (native: with the model padding '
                                     'rule; ours: our evaluation mask line for line), unsampled lines carry no k-space',
                    sampled_zero_cols_max=max(by_file[nm]['checks']['sampled_zero_cols_max'] for nm in names),
                    scale_relerr_max=max(by_file[nm]['checks']['scale_relerr_max'] for nm in names),
                    crop_notes=sorted({r['crop_note'] for r in vol_rows}), partial_unreadable_lines=n_bad_lines),
        outputs=dict(per_volume=os.path.basename(paths['per_volume']), per_slice=os.path.basename(paths['per_slice']),
                     partial=os.path.basename(paths['partial'])),
        provenance=provenance(weights, w_sha),
    )
    write_csv(paths['per_volume'], vol_rows, vcols)
    write_csv(paths['per_slice'], sl_rows, scols)
    write_json(paths['summary'], summ)
    if defect:
        with open(paths['defect'], 'x') as f:
            f.write(json.dumps(dict(model=key, mode=mode, reasons=reasons, nonfinite_slices=bad[:50]), ensure_ascii=False) + '\n')
    print(json.dumps({k: summ[k] for k in ('model', 'mode', 'n_vol', 'n_slice', 'defect', 'seconds')}, ensure_ascii=False))
    print(json.dumps(summ['metrics'], ensure_ascii=False))
    print(f'[{stem}] {"DEFECT " if defect else ""}출력: {paths["summary"]}  (종료 코드 {exit_code})', flush=True)
    return exit_code


# ------------------------------------------------------------------ 자가 점검
def selftest(a):
    import torch
    from dataloader_h5_v5 import build_r4_mask, crop_or_pad_to, ifft2c, fft2c
    import eval_v8fix_official as O
    from eval_paired_v8_nodc import slice_metrics
    ok = []

    # 1. 우리 마스크 = 근거 E-167 의 정의 (R 4 등간격 시작 3 + 중앙 31줄)
    m384 = ours_mask_1d()
    ref = np.zeros(GRID, np.float32)
    ref[OFFSET::ACCEL] = 1
    pad = (GRID - OURS_CENTER + 1) // 2
    ref[pad:pad + OURS_CENTER] = 1
    assert np.array_equal(m384, ref) and int(m384.sum()) == OURS_LINES, (m384.sum(), pad)
    ok.append(f'우리 마스크: 중앙 {pad}..{pad + OURS_CENTER - 1}, 3::4, {OURS_LINES}줄, R_eff {GRID / OURS_LINES:.4f}')

    # 2. 원 격자 규칙
    for W in (192, 213, 260, 264, 272, 276, 320, 322, 324, 368, 383, 384, 385, 392, 396, 576, 640):
        m, mapping, n = native_mask_1d(W)
        assert mapping == ('exact' if W == GRID else 'rule'), (W, mapping)
        exact_n = fractions.Fraction(W * OURS_CENTER, GRID)
        assert n == int(exact_n + fractions.Fraction(1, 2)) and abs(n - exact_n) <= fractions.Fraction(1, 2), (W, n)
        p = (W - n + 1) // 2
        exp = np.zeros(W, np.float32)
        exp[OFFSET::ACCEL] = 1
        exp[p:p + n] = 1
        assert np.array_equal(m, exp), W
        if W == GRID:
            assert np.array_equal(m, m384)
    assert native_center_lines(192) == 16 and native_center_lines(576) == 47     # 정확히 .5 인 경우 올림 (15.5, 46.5)
    ok.append('원 격자 규칙: 17개 폭에서 중앙 줄 수·위치·3::4 확인, W=384 만 exact')

    # 3. 줄 대응의 물리: 우리 로더 연산(ifft2c → crop_or_pad_to → fft2c)과 마스크의 순서 교환
    rng = np.random.default_rng(0)
    for W in (384, 320, 396, 213):
        k = (rng.normal(size=(2, 24, W)) + 1j * rng.normal(size=(2, 24, W))).astype(np.complex64)
        m_nat, mapping, _ = native_mask_1d(W)
        loader_then_mask = fft2c(crop_or_pad_to(ifft2c(k), (24, GRID))) * m384[None, None, :]
        mask_then_loader = fft2c(crop_or_pad_to(ifft2c(k * m_nat[None, None, :]), (24, GRID)))
        rel = float(np.abs(loader_then_mask - mask_then_loader).max() / np.abs(loader_then_mask).max())
        on, shared = physical_overlap(m384, m_nat, W)
        if W == GRID:
            assert rel < 1e-5 and on == OURS_LINES and shared == OURS_LINES, (rel, on, shared)
        else:
            assert rel > 1e-2, (W, rel)       # 원 격자 마스크를 씌운 뒤 384 로 옮긴 것 ≠ 384 격자 마스크
        ok.append(f'  W={W} ({mapping}): 마스크 순서 교환 상대차 {rel:.2e}, 우리 119줄 중 원 격자 줄과 겹침 {on}, '
                  f'원 마스크에서도 샘플 {shared}')
    w = window_on_384(320, 48, 272)
    assert w.sum() == 269 and w[57] and not w[56] and w[325] and not w[326], (w.sum(), np.flatnonzero(w)[[0, -1]])
    assert window_on_384(396, 0, 396).all() and window_on_384(384, 0, 384).all()
    lr = line_record('native', 320, native_mask_1d(320)[0], native_mask_1d(320)[0], 48, 272, 26, 'rule')
    assert lr['mask_lines_grid'] == int(native_mask_1d(320)[0].sum()) and lr['acq_width'] == 224, lr
    ok.append(f'취득 구간: W320 [48,272) → 384 틀 {int(w.sum())}줄; 원 격자 R_eff {lr["R_eff"]:.3f}, 구간 안 {lr["R_eff_acq"]:.3f}')

    # 4. 지표 배관 = 같은 함수 직접 호출 = fastmri.evaluate
    sys.modules.setdefault('runstats', types.SimpleNamespace(Statistics=object))
    from fastmri import evaluate as FE
    for (H, W) in ((320, 320), (320, 260), (213, 213), (384, 312)):
        gt = (rng.random((3, H, W)) * 3e-4).astype(np.float32)
        pr = (gt + rng.normal(0, 2e-5, gt.shape)).astype(np.float32)
        lab = np.stack([crop_or_pad_to(g, (GRID, GRID)) * VAL_AMP_Y for g in gt]).astype(np.float32)
        bm = np.zeros_like(lab)
        bm[:, 150:250, 150:240] = 1
        p384 = np.stack([crop_or_pad_to(p * VAL_AMP_Y, (GRID, GRID)) for p in pr])
        row, sl = volume_metrics(gt, list(pr), list(lab), list(bm), list(p384), O, slice_metrics, torch.device('cpu'))
        for s in range(3):
            d = slice_metrics(torch.from_numpy(p384[s])[None, None], torch.from_numpy(lab[s])[None, None],
                              torch.from_numpy(bm[s])[None, None])
            assert all(abs(sl[s][k] - d[k]) == 0 for k in ('ssim', 'psnr', 'nmse', 'l1')), (sl[s], d)
        off = O.official_volume(gt, pr, O.center_crop_np(bm > 0.5, (H, W)))
        assert row['ssim_off'] == off['ssim'] and row['psnr_off'] == off['psnr'] and row['nmse_off'] == off['nmse']
        tg = O.center_crop_np(gt, (W, W)) if H >= W else gt
        rc = O.center_crop_np(pr, (W, W)) if H >= W else pr
        for k, f in (('ssim', FE.ssim), ('psnr', FE.psnr), ('nmse', FE.nmse)):
            v = float(np.asarray(f(tg, rc)).reshape(-1)[0])        # fastmri.evaluate.ssim 은 (1,) 배열을 돌려준다
            assert abs(row[f'{k}_off'] - v) <= 1e-6 * max(1.0, abs(v)), (k, row[f'{k}_off'], v)
    ok.append('지표: volume_metrics = slice_metrics·official_volume 직접 호출(차 0) = fastmri.evaluate (1e-6)')

    # 5. 출력 자르기 규칙
    assert flair203_crop(213, (512, 408)) == (213, 213) and flair203_crop(276, (512, 512)) == (276, 276)
    assert flair203_crop(320, (320, 320)) == (320, 320) and flair203_crop(396, (384, 384)) == (384, 384)
    x = rng.random((384, 396)).astype(np.float32)
    y, note = to_rss_shape(x, (384, 384))
    assert note == 'own+crop' and np.array_equal(y, x[:, 6:390])
    y, note = to_rss_shape(x[:, :384], (384, 384))
    assert note == 'own'
    ok.append('자르기: FLAIR 203 규칙 (213→213², 276→276²), 잔여 차이 기록')

    # 6. 덮어쓰기 거부
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = out_paths(td, 'x_native')
        assert refuse_existing(p, False, False) is None
        open(p['partial'], 'w').close()
        assert refuse_existing(p, False, False) and refuse_existing(p, True, False) is None
        open(p['summary'], 'w').close()
        assert refuse_existing(p, True, False)
        try:
            write_json(p['summary'], {})
            raise AssertionError('write_json overwrote')
        except FileExistsError:
            pass
    ok.append('덮어쓰기 거부: 출력·진행 기록·덤프 · open(x) 모드')

    # 7. (선택) 실제 데이터: 우리 평가 데이터셋 마스크 = 우리 마스크 (줄 단위), 볼륨별 대응 표
    if a.data_path:
        import eval_unet_only_v8 as E
        ds = E.build_val_dataset(os.path.abspath(a.data_path))
        nf = a.max_files if a.max_files > 0 else 2
        seen, n_sl = [], 0
        for i, (fp, s, _) in enumerate(ds.samples):
            if fp not in seen:
                if len(seen) == nf:
                    break
                seen.append(fp)
            mk = ds[i]['mask']
            assert np.array_equal(mk.reshape(-1, mk.shape[-1])[0] > 0.5, m384 > 0.5) and (mk == mk[:, :1, :]).all(), i
            n_sl += 1
        ok.append(f'데이터: 앞 {len(seen)} 볼륨 {n_sl} 슬라이스의 평가 마스크 = 우리 마스크 (줄 단위)')
        table = {}
        for fp in ds.files:
            hm = header_meta(fp)
            m, mapping, n = native_mask_1d(hm['kW'])
            lr = line_record('native', hm['kW'], m, m, hm['pad_l'], hm['pad_r'], n, mapping)
            kk = (hm['kW'], hm['pad_l'], hm['pad_r'])
            if kk not in table:
                table[kk] = dict(n_vol=0, mapping=mapping, center=n, lines=lr['mask_lines_grid'], R_eff=lr['R_eff'],
                                 acq=lr['acq_width'], lines_acq=lr['mask_lines_acq'], R_eff_acq=lr['R_eff_acq'],
                                 ours_on_native=lr['ours_lines_on_native_grid'], shared=lr['shared_physical_lines'],
                                 example=os.path.basename(fp))
            table[kk]['n_vol'] += 1
        ok.append(f'데이터 {len(ds.files)} 볼륨의 원 격자 대응 (W, pad_l, pad_r): ' + json.dumps(
            {f'{k[0]},{k[1]},{k[2]}': v for k, v in sorted(table.items())}, ensure_ascii=False))
    for line in ok:
        print('  ' + line)
    print('selftest PASS')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description='공개 가중치 기준선(U-Net·E2E-VarNet·PromptMR+) test_full 평가 — 두 전처리 방식')
    p.add_argument('--model', choices=sorted(MODELS))
    p.add_argument('--mode', choices=MODES)
    p.add_argument('--data-path')
    p.add_argument('--out-dir')
    p.add_argument('--weights', default=None, help='기본 = 계획의 가중치 파일 (MODELS 표)')
    p.add_argument('--max-files', type=int, default=-1, help='-1 = 전체; N = 앞 N 볼륨 (볼륨 단위)')
    p.add_argument('--expect-files-from', default=None,
                   help='per_volume CSV(file 열) — 데이터 파일 목록이 같아야 한다 (예: 우리 모델 test_full 평가 CSV)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--tf32', choices=['off', 'default'], default='off', help='off(기본) = fp32 행렬곱·합성곱에서 TF32 끔')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=0, help='>0 이면 torch CPU 스레드 수')
    p.add_argument('--pmr-sens-per-coil', action='store_true',
                   help='PromptMR+ native: 감도 지도를 코일마다 계산 (메모리 절약; 그들 기본은 끔). ours 방식은 예전처럼 늘 켬')
    p.add_argument('--resume', action='store_true', help='같은 설정의 진행 기록(partial_*.jsonl)에 이어서')
    p.add_argument('--dump-volumes', type=int, default=0, help='앞 K 볼륨의 배열을 dump_<stem>/ 에 npz 로 (독립 대조용)')
    p.add_argument('--test-inject-nonfinite', default='', help='시험 전용: 쉼표로 준 idx 의 출력 첫 화소를 NaN 으로')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args(argv)
    if a.selftest:
        return selftest(a)
    for k in ('model', 'mode', 'data_path', 'out_dir'):
        if getattr(a, k) is None:
            p.error(f'--{k.replace("_", "-")} 필요')
    return run(a)


if __name__ == '__main__':
    sys.exit(main())
