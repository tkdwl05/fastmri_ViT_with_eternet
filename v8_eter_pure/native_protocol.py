"""
네이티브 프로토콜 기준선 추론 — fastMRI 공식 `run_pretrained_varnet_inference.py` 규약 재현.

우리 파이프라인(16코일 절단 · 이미지도메인 384² crop/pad · 재-FFT)은 사전학습 leaderboard
가중치의 학습 조건과 다르다. 그 domain shift 를 걷어낸 **VarNet/U-Net 자기 프로토콜** 행을
만들기 위해, 원본 h5 의 native k-space 를 그대로 쓰고 공식 crop 규약으로 잘라낸다.

공식과 맞춘 지점:
  - 전체 코일 사용(절단 없음), native 해상도 k-space 그대로.
  - 마스크는 ismrmrd 헤더의 acquired 구간 [padding_left, padding_right) 밖을 0 으로
    (`VarNetDataTransform` 의 `mask_torch[:, :, :acq_start] = 0` 재현).
  - `num_low_frequencies=None` → SensitivityModel 이 마스크에서 ACS 폭을 자동 검출
    (공식 추론 스크립트는 `model(masked_kspace, mask)` 로 인자를 넘기지 않는다).
  - 출력 crop = reconSpace matrixSize, 단 출력 폭이 더 좁으면 정사각으로 축소
    (공식의 FLAIR 203 예외 처리 `if output.shape[-1] < crop_size[1]`).

우리 쪽과 맞춘 지점(비교 가능성):
  - 언더샘플 마스크는 우리 val 규약과 동일한 생성기(`build_r4_mask`, center_fraction 0.08,
    R=4, offset=accel-1 고정)를 native 폭에 적용.
  - GT 는 파일의 `reconstruction_rss` × 1e6 (dataloader 의 `val_amp_Y` 와 동일 스케일 —
    L1 은 스케일 의존 지표라 맞춰야 비교된다).
  - brain mask 는 dataloader_h5_v5 와 같은 처방(Otsu×0.4 + largest CC).
"""

import sys
import os
import xml.etree.ElementTree as etree

import h5py
import numpy as np
import torch
from skimage.filters import threshold_otsu
from scipy.ndimage import label as ndi_label

from fastmri.data.mri_data import et_query

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(_HERE), 'dataloaders'))
from dataloader_h5_v5 import build_r4_mask, ifft2c   # noqa: E402  (동일 마스크·FFT 규약 재사용)

AMP_Y = 1e6   # dataloader val_amp_Y


def read_header_geometry(f):
    """ismrmrd 헤더 → (recon_size(x,y), padding_left, padding_right). fastmri _retrieve_metadata 동일."""
    et = etree.fromstring(f['ismrmrd_header'][()])
    rec = ['encoding', 'reconSpace', 'matrixSize']
    enc = ['encoding', 'encodedSpace', 'matrixSize']
    lim = ['encoding', 'encodingLimits', 'kspace_encoding_step_1']
    recon_size = (int(et_query(et, rec + ['x'])), int(et_query(et, rec + ['y'])))
    enc_y = int(et_query(et, enc + ['y']))
    center = int(et_query(et, lim + ['center']))
    n_steps = int(et_query(et, lim + ['maximum'])) + 1
    pad_l = enc_y // 2 - center
    pad_r = pad_l + n_steps
    return recon_size, pad_l, pad_r


def brain_mask_from_gt(gt):
    """dataloader_h5_v5.__getitem__ 8-b 와 동일한 brain mask (Otsu×0.4 + largest CC)."""
    gt = gt.astype(np.float32)
    gt_max = float(gt.max())
    if gt_max <= 0:
        return np.zeros_like(gt, dtype=np.float32)
    non_zero = gt[gt > 0]
    if non_zero.size > 100:
        try:
            thr = float(threshold_otsu(non_zero)) * 0.4
        except Exception:
            thr = 0.05 * gt_max
    else:
        thr = 0.05 * gt_max
    raw = gt > thr
    lbl, n_cc = ndi_label(raw)
    if n_cc > 0:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        raw = (lbl == int(sizes.argmax()))
    return raw.astype(np.float32)


def center_crop_np(x, shape):
    """마지막 두 축 중앙 crop (fastmri.data.transforms.center_crop 의 numpy 판)."""
    h, w = x.shape[-2:]
    th, tw = shape
    th, tw = min(th, h), min(tw, w)
    t = (h - th) // 2
    l = (w - tw) // 2
    return x[..., t:t + th, l:l + tw]


def load_native_slice(file_path, slice_idx, center_fraction=0.08, acceleration=4):
    """원본 h5 슬라이스 → (masked_kspace(C,H,W) complex, mask_1d(W,), gt(H',W'), crop_size)."""
    with h5py.File(file_path, 'r') as f:
        ksp = f['kspace'][slice_idx].astype(np.complex64)      # (C,H,W) 전체 코일
        gt = f['reconstruction_rss'][slice_idx].astype(np.float32)
        recon_size, pad_l, pad_r = read_header_geometry(f)

    W = ksp.shape[-1]
    mask_1d = build_r4_mask(W, center_fraction, acceleration, rng=None)  # offset=accel-1 고정
    # 공식 VarNetDataTransform: 취득 구간 밖은 마스크 0
    mask_1d[:max(pad_l, 0)] = 0.0
    if pad_r < W:
        mask_1d[pad_r:] = 0.0
    ksp_masked = ksp * mask_1d[np.newaxis, np.newaxis, :]
    return ksp_masked, mask_1d, gt, recon_size


def _official_crop(out, recon_size, gt_shape):
    """공식 추론 스크립트의 crop 규약 + GT 형상 정합."""
    crop = (recon_size[0], recon_size[1])
    if out.shape[-1] < crop[1]:            # FLAIR 203 등: 출력이 더 좁으면 정사각으로
        crop = (out.shape[-1], out.shape[-1])
    out = center_crop_np(out, crop)
    if out.shape != gt_shape:              # 잔여 불일치는 공통 최소 영역으로
        common = (min(out.shape[-2], gt_shape[-2]), min(out.shape[-1], gt_shape[-1]))
        out = center_crop_np(out, common)
    return out


def run_varnet_native(model, file_path, slice_idx, device,
                      center_fraction=0.08, acceleration=4):
    """공식 규약 VarNet 추론 → (recon, gt, brain_mask) — 셋 다 native crop 프레임."""
    ksp_masked, mask_1d, gt, recon_size = load_native_slice(
        file_path, slice_idx, center_fraction, acceleration)
    W = ksp_masked.shape[-1]
    mk = torch.stack([torch.from_numpy(np.ascontiguousarray(ksp_masked.real)),
                      torch.from_numpy(np.ascontiguousarray(ksp_masked.imag))], dim=-1)
    mk = mk.unsqueeze(0).float().to(device)                       # (1,C,H,W,2)
    mask_t = torch.from_numpy(mask_1d > 0.5).view(1, 1, 1, W, 1).to(device)
    with torch.no_grad():
        out = model(mk, mask_t)                                   # num_low_frequencies=None (공식 동일)
    out = out.squeeze(0).float().cpu().numpy()
    out = _official_crop(out, recon_size, gt.shape)
    gt_c = center_crop_np(gt, out.shape) * AMP_Y
    return out, gt_c, brain_mask_from_gt(gt_c)


def run_unet_native(model, file_path, slice_idx, device,
                    center_fraction=0.08, acceleration=4):
    """공식 규약 U-Net 추론(zero-filled RSS → crop → instance norm) → (recon, gt, brain_mask)."""
    ksp_masked, mask_1d, gt, recon_size = load_native_slice(
        file_path, slice_idx, center_fraction, acceleration)
    zf = ifft2c(ksp_masked)
    zf_rss = np.sqrt((np.abs(zf) ** 2).sum(0)).astype(np.float32)
    zf_rss = _official_crop(zf_rss, recon_size, gt.shape)         # 공식: 정규화 전에 crop
    t = torch.from_numpy(zf_rss).to(device)
    mean, std = t.mean(), t.std().clamp(min=1e-8)
    x = ((t - mean) / std).clamp(-6.0, 6.0)[None, None]
    with torch.no_grad():
        out = model(x).squeeze()
    out = (out * std + mean).float().cpu().numpy()
    gt_c = center_crop_np(gt, out.shape) * AMP_Y
    return out, gt_c, brain_mask_from_gt(gt_c)
