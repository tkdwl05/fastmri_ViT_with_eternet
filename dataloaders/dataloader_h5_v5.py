"""
FastMRI brain multicoil H5 데이터로더 (v5: 분포 폭 확장 + augmentation)

v4 대비 변경점:
  1) 사이즈 필터 완화
     - v4: rss shape == (320, 320) 인 파일만 사용 → 360개 스킵
     - v5: reconstruction_rss 만 있으면 모두 사용. 이미지 도메인에서 320×320 으로
           center-crop 또는 zero-pad. k-space 도 image-domain 으로 변환 후 동일 처리.
  2) Augmentation
     - 랜덤 H/V flip (image_crop, gt_rss 동기 적용). mask 는 매 샘플마다 새로 만들어지므로
       flip 영향 없음. flip 은 image 도메인에서 적용 후 FFT 로 k-space 재계산.
  3) 모델/loss 호환
     - 반환 dict 키 / 채널 수 / 스케일 팩터 모두 v4 와 동일

참고: facebookresearch/fastMRI — fftc.py, transforms.py
"""

import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────
#  Centered iFFT / FFT (fastMRI 표준 순서)
# ──────────────────────────────────────────────

def ifft2c(kspace: np.ndarray) -> np.ndarray:
    x = np.fft.ifftshift(kspace, axes=(-2, -1))
    x = np.fft.ifft2(x, axes=(-2, -1), norm='ortho')
    x = np.fft.fftshift(x, axes=(-2, -1))
    return x


def fft2c(image: np.ndarray) -> np.ndarray:
    x = np.fft.ifftshift(image, axes=(-2, -1))
    x = np.fft.fft2(x, axes=(-2, -1), norm='ortho')
    x = np.fft.fftshift(x, axes=(-2, -1))
    return x


def crop_or_pad_to(x: np.ndarray, target_hw: tuple) -> np.ndarray:
    """마지막 두 축에 대해 center-crop(큰 경우) / zero-pad(작은 경우)로 target_hw 맞춤.
    complex64 / float32 어느 쪽이든 np.pad 가 real/imag 별도로 잘 처리한다.
    """
    th, tw = target_hw

    # H 축 처리
    H = x.shape[-2]
    if H > th:
        sh = (H - th) // 2
        x = x[..., sh:sh + th, :]
    elif H < th:
        ph = (th - H) // 2
        ph2 = th - H - ph
        pad = [(0, 0)] * (x.ndim - 2) + [(ph, ph2), (0, 0)]
        x = np.pad(x, pad)

    # W 축 처리
    W = x.shape[-1]
    if W > tw:
        sw = (W - tw) // 2
        x = x[..., sw:sw + tw]
    elif W < tw:
        pw = (tw - W) // 2
        pw2 = tw - W - pw
        pad = [(0, 0)] * (x.ndim - 1) + [(pw, pw2)]
        x = np.pad(x, pad)

    return x


# ──────────────────────────────────────────────
#  R=4 equispaced mask (phase-encoding = 마지막 축)
# ──────────────────────────────────────────────

def build_r4_mask(width: int, center_fraction: float = 0.08, acceleration: int = 4,
                  rng: np.random.Generator = None) -> np.ndarray:
    num_low_freqs = int(round(width * center_fraction))
    mask = np.zeros(width, dtype=np.float32)

    pad = (width - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = 1.0

    if rng is not None:
        offset = int(rng.integers(0, acceleration))
    else:
        offset = acceleration - 1
    mask[offset::acceleration] = 1.0

    return mask


# ──────────────────────────────────────────────
#  FastMRI H5 Dataloader (v5)
# ──────────────────────────────────────────────

class FastMRI_H5_Dataloader(Dataset):
    """
    FastMRI brain multicoil .h5 dataloader (v5).

    v4 와 동일한 반환 dict:
      data:     (2*N_COIL_CH, N_OUTPUT, N_OUTPUT)  — aliased k-space (real/imag 교번)
      data_img: (2*N_COIL_CH, N_OUTPUT, N_OUTPUT)  — aliased image
      label:    (1,           N_OUTPUT, N_OUTPUT)  — GT RSS magnitude
      mask:     (1,           N_OUTPUT, N_OUTPUT)
      sens:     (2*N_COIL_CH, N_OUTPUT, N_OUTPUT)  — ACS-based sens (Σ|s|²=1)
    """

    def __init__(self, data_folder, num_files=None,
                 target_size: int = 320, num_coil_ch: int = 16,
                 acceleration: int = 4, center_fraction: float = 0.08,
                 random_mask: bool = True,
                 augment: bool = False, augment_flip_p: float = 0.5):
        print('\n  @ dataloader_h5_v5.py (size-relaxed + flip augmentation)')
        print('  Dataset : FastMRI brain multicoil .h5')

        self.data_folder = data_folder
        all_files = sorted([
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder) if f.endswith('.h5')
        ])

        self.N_OUTPUT  = target_size
        self.N_COIL_CH = num_coil_ch
        self.acceleration   = acceleration
        self.center_fraction = center_fraction

        self.val_amp_X_img = 1e6
        self.val_amp_X_ksp = 1e4
        self.val_amp_Y     = 1e6

        # 사이즈 필터 완화: reconstruction_rss 가 존재하기만 하면 사용
        kept_files = []
        skipped = 0
        for fp in all_files:
            try:
                with h5py.File(fp, 'r') as f:
                    if 'reconstruction_rss' in f and 'kspace' in f:
                        kept_files.append(fp)
                    else:
                        skipped += 1
            except Exception:
                skipped += 1

        if num_files is not None:
            kept_files = kept_files[:num_files]
        self.files = kept_files

        self.samples = []
        for fp in self.files:
            with h5py.File(fp, 'r') as f:
                n_slices = f['kspace'].shape[0]
            for s in range(n_slices):
                self.samples.append((fp, s, True))

        self.random_mask = random_mask
        self.augment = augment
        self.augment_flip_p = float(augment_flip_p)
        self.rng = np.random.default_rng() if (random_mask or augment) else None

        print(f"    Loaded {len(self.files)} .h5 files  →  {len(self.samples)} slices total")
        print(f"    (skipped {skipped} files: missing reconstruction_rss/kspace)")
        print(f"    target_size={self.N_OUTPUT}, coils={self.N_COIL_CH}, "
              f"R={self.acceleration}, center_fraction={self.center_fraction}, "
              f"augment={self.augment}")

    def __len__(self):
        return len(self.samples)

    def _pack_complex_to_real(self, arr_complex: np.ndarray) -> np.ndarray:
        C, H, W = arr_complex.shape
        out = np.empty((2 * C, H, W), dtype=np.float32)
        out[0::2] = arr_complex.real.astype(np.float32)
        out[1::2] = arr_complex.imag.astype(np.float32)
        return out

    def __getitem__(self, idx):
        file_path, slice_idx, has_rss = self.samples[idx]

        with h5py.File(file_path, 'r') as f:
            kspace_slice = f['kspace'][slice_idx]
            if has_rss:
                gt_rss = f['reconstruction_rss'][slice_idx].astype(np.float32)
            else:
                gt_rss = None

        coils = kspace_slice.shape[0]

        # 1) Full k-space → iFFT → complex image (raw 사이즈)
        img_full = ifft2c(kspace_slice.astype(np.complex64))

        # 2) Image 도메인에서 320×320 으로 crop or pad (v5: 사이즈 필터 완화)
        img_crop = crop_or_pad_to(img_full, (self.N_OUTPUT, self.N_OUTPUT))
        if gt_rss is not None:
            gt_rss = crop_or_pad_to(gt_rss, (self.N_OUTPUT, self.N_OUTPUT))

        # 3) Augmentation: random H/V flip (image 도메인에서 동기 적용)
        if self.augment and self.rng is not None:
            if self.rng.random() < self.augment_flip_p:
                img_crop = np.ascontiguousarray(img_crop[..., ::-1, :])
                if gt_rss is not None:
                    gt_rss = np.ascontiguousarray(gt_rss[::-1, :])
            if self.rng.random() < self.augment_flip_p:
                img_crop = np.ascontiguousarray(img_crop[..., :, ::-1])
                if gt_rss is not None:
                    gt_rss = np.ascontiguousarray(gt_rss[:, ::-1])

        # 4) Cropped image → FFT → k-space at target resolution
        ksp_crop = fft2c(img_crop)

        # 5) R=4 mask
        mask_1d = build_r4_mask(
            width=self.N_OUTPUT,
            center_fraction=self.center_fraction,
            acceleration=self.acceleration,
            rng=self.rng if self.random_mask else None,
        )
        mask = mask_1d[np.newaxis, np.newaxis, :]
        ksp_masked = ksp_crop * mask

        # 6) Masked k-space → iFFT → aliased image
        img_alias = ifft2c(ksp_masked)

        # 7) 코일 16 채널로 맞춤
        take = min(coils, self.N_COIL_CH)
        ksp_out = np.zeros((self.N_COIL_CH, self.N_OUTPUT, self.N_OUTPUT), dtype=np.complex64)
        img_out = np.zeros((self.N_COIL_CH, self.N_OUTPUT, self.N_OUTPUT), dtype=np.complex64)
        ksp_out[:take] = ksp_masked[:take]
        img_out[:take] = img_alias[:take]

        data     = self._pack_complex_to_real(ksp_out) * self.val_amp_X_ksp
        data_img = self._pack_complex_to_real(img_out) * self.val_amp_X_img

        # 8) GT
        if gt_rss is None:
            img_full_take = img_crop[:take]
            gt_rss = np.sqrt(np.sum(np.abs(img_full_take) ** 2, axis=0)).astype(np.float32)
        label = gt_rss[np.newaxis] * self.val_amp_Y

        # 9) DC block용 mask / sens
        mask_out = np.broadcast_to(
            mask_1d[np.newaxis, :], (self.N_OUTPUT, self.N_OUTPUT)
        )[np.newaxis].astype(np.float32)

        num_low_freqs = int(round(self.N_OUTPUT * self.center_fraction))
        pad_acs = (self.N_OUTPUT - num_low_freqs + 1) // 2
        acs_ksp = np.zeros_like(ksp_out)
        acs_ksp[..., pad_acs:pad_acs + num_low_freqs] = ksp_out[..., pad_acs:pad_acs + num_low_freqs]
        acs_img = ifft2c(acs_ksp)
        rss_acs = np.sqrt(np.sum(np.abs(acs_img) ** 2, axis=0, keepdims=True))
        rss_acs = np.maximum(rss_acs, 1e-9)
        sens_c  = acs_img / rss_acs
        sens_packed = self._pack_complex_to_real(sens_c)

        return {
            'data':     data.astype(np.float32),
            'data_img': data_img.astype(np.float32),
            'label':    label.astype(np.float32),
            'mask':     mask_out,
            'sens':     sens_packed.astype(np.float32),
        }


if __name__ == '__main__':
    ds = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_val', num_files=2, augment=True
    )
    loader = DataLoader(ds, batch_size=1)
    for batch in loader:
        for k, v in batch.items():
            print(f"  {k:>8}: {tuple(v.shape)}  [{v.min().item():.3f}, {v.max().item():.3f}]")
        break
