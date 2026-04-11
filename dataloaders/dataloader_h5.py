"""
FastMRI brain multicoil H5 데이터로더 (fastMRI 공식 표준 전처리)

참고: facebookresearch/fastMRI — fastmri/data/transforms.py (UnetDataTransform)
      facebookresearch/fastMRI — fastmri/fftc.py (ifft2c_new)

핵심 원칙 (공식 코드와 일치):
  1. Ground Truth는 h5의 `reconstruction_rss`를 직접 사용 (320×320)
  2. iFFT는 ifftshift → ifft2 → fftshift 순서
  3. Crop은 iFFT **후** image 도메인에서 수행
  4. R=4 언더샘플링 마스크는 phase-encoding 축(=W, 마지막 축)에 한 번만 적용
"""

import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────
#  Centered iFFT / FFT (fastMRI 표준 순서)
# ──────────────────────────────────────────────

def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centered 2D iFFT along last two axes.  fastMRI ifft2c_new와 동일 순서."""
    x = np.fft.ifftshift(kspace, axes=(-2, -1))
    x = np.fft.ifft2(x, axes=(-2, -1), norm='ortho')
    x = np.fft.fftshift(x, axes=(-2, -1))
    return x


def fft2c(image: np.ndarray) -> np.ndarray:
    """Centered 2D FFT along last two axes."""
    x = np.fft.ifftshift(image, axes=(-2, -1))
    x = np.fft.fft2(x, axes=(-2, -1), norm='ortho')
    x = np.fft.fftshift(x, axes=(-2, -1))
    return x


def complex_center_crop(x: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Image 도메인 center crop. 마지막 두 축(-2, -1)에 대해 수행."""
    H, W = x.shape[-2], x.shape[-1]
    th, tw = target_hw
    sh = (H - th) // 2
    sw = (W - tw) // 2
    return x[..., sh:sh + th, sw:sw + tw]


def center_crop_real(x: np.ndarray, target_hw: tuple) -> np.ndarray:
    """실수 배열용 center crop."""
    return complex_center_crop(x, target_hw)


# ──────────────────────────────────────────────
#  R=4 equispaced mask (phase-encoding = 마지막 축)
# ──────────────────────────────────────────────

def build_r4_mask(width: int, center_fraction: float = 0.08, acceleration: int = 4) -> np.ndarray:
    """
    1D equispaced mask along width (phase-encoding) axis.
    - center_fraction * width 만큼의 중앙 라인은 모두 샘플링 (ACS)
    - 나머지 영역은 `acceleration` 간격으로 샘플링
    """
    num_low_freqs = int(round(width * center_fraction))
    mask = np.zeros(width, dtype=np.float32)

    # ACS (low frequency) center lines
    pad = (width - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = 1.0

    # Equispaced outer lines: offset 고정 (데이터셋 전체에서 동일한 패턴)
    offset = acceleration - 1  # 원본 코드와 동일하게 3부터 시작
    mask[offset::acceleration] = 1.0

    return mask  # shape (W,)


# ──────────────────────────────────────────────
#  FastMRI H5 Dataloader
# ──────────────────────────────────────────────

class FastMRI_H5_Dataloader(Dataset):
    """
    FastMRI brain multicoil .h5 dataloader.

    반환 형태 (슬라이스별):
      data:     (2*N_COIL_CH, N_OUTPUT, N_OUTPUT)  — aliased k-space (실수/허수 교번)
      data_img: (2*N_COIL_CH, N_OUTPUT, N_OUTPUT)  — aliased image   (실수/허수 교번)
      label:    (1,           N_OUTPUT, N_OUTPUT)  — GT RSS magnitude (h5 reconstruction_rss)

    전처리 흐름 (fastMRI 공식 표준):
      full_ksp  →  ifft2c  →  complex_center_crop((N_OUTPUT, N_OUTPUT))  →  aliased mask 적용 후 fft2c로 ksp input 생성
      GT        =  h5['reconstruction_rss'][slice]   (이미 320×320 magnitude)
    """

    def __init__(self, data_folder, num_files=None,
                 target_size: int = 320, num_coil_ch: int = 16,
                 acceleration: int = 4, center_fraction: float = 0.08):
        print('\n  @ dataloader_h5.py (fastMRI 공식 표준 전처리)')
        print('  Dataset : FastMRI brain multicoil .h5')

        self.data_folder = data_folder
        all_files = sorted([
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder) if f.endswith('.h5')
        ])

        self.N_OUTPUT  = target_size      # 320 (fastMRI brain 표준)
        self.N_COIL_CH = num_coil_ch      # 16
        self.acceleration   = acceleration
        self.center_fraction = center_fraction

        # 스케일 팩터
        # - GT/aliased image: reconstruction_rss는 ~1e-4 → ×1e6 후 [1, 300]
        # - aliased k-space : ortho fft 결과 image보다 ~30배 크므로 ×1e4로 낮춰 image와 균형
        self.val_amp_X_img = 1e6
        self.val_amp_X_ksp = 1e4
        self.val_amp_Y     = 1e6

        # 파일 필터:
        #   - reconstruction_rss 의 spatial shape 가 정확히 (N_OUTPUT, N_OUTPUT) 인 파일만 사용
        #   - kspace H, W 가 N_OUTPUT 이상이어야 image-domain crop 이 가능
        #   - test 셋(reconstruction_rss 없음)은 이 dataloader 평가에 사용하지 않으므로 제외
        kept_files = []
        skipped = 0
        for fp in all_files:
            try:
                with h5py.File(fp, 'r') as f:
                    if 'reconstruction_rss' not in f:
                        skipped += 1
                        continue
                    rss_shape = f['reconstruction_rss'].shape  # (n_slices, h, w)
                    ksp_shape = f['kspace'].shape              # (n_slices, n_coils, H, W)
                if (rss_shape[-2] == self.N_OUTPUT and rss_shape[-1] == self.N_OUTPUT
                        and ksp_shape[-2] >= self.N_OUTPUT and ksp_shape[-1] >= self.N_OUTPUT):
                    kept_files.append(fp)
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        if num_files is not None:
            kept_files = kept_files[:num_files]
        self.files = kept_files

        # 파일별 슬라이스 수 선계산
        self.samples = []
        for fp in self.files:
            with h5py.File(fp, 'r') as f:
                n_slices = f['kspace'].shape[0]
            for s in range(n_slices):
                self.samples.append((fp, s, True))

        # 1D 마스크 선계산 (모든 샘플 공통)
        self.mask_1d = build_r4_mask(
            width=self.N_OUTPUT,
            center_fraction=self.center_fraction,
            acceleration=self.acceleration,
        )  # (N_OUTPUT,)

        print(f"    Loaded {len(self.files)} .h5 files  →  {len(self.samples)} slices total")
        print(f"    (skipped {skipped} files: rss shape != ({self.N_OUTPUT},{self.N_OUTPUT}) or kspace too small)")
        print(f"    target_size={self.N_OUTPUT}, coils={self.N_COIL_CH}, "
              f"R={self.acceleration}, center_fraction={self.center_fraction}")

    def __len__(self):
        return len(self.samples)

    def _pack_complex_to_real(self, arr_complex: np.ndarray) -> np.ndarray:
        """(C, H, W) complex → (2C, H, W) float32 (real/imag 교번 배치)."""
        C, H, W = arr_complex.shape
        out = np.empty((2 * C, H, W), dtype=np.float32)
        out[0::2] = arr_complex.real.astype(np.float32)
        out[1::2] = arr_complex.imag.astype(np.float32)
        return out

    def __getitem__(self, idx):
        file_path, slice_idx, has_rss = self.samples[idx]

        with h5py.File(file_path, 'r') as f:
            kspace_slice = f['kspace'][slice_idx]  # (coils, H_raw, W_raw) complex64
            if has_rss:
                gt_rss = f['reconstruction_rss'][slice_idx].astype(np.float32)  # (h, w)
            else:
                gt_rss = None

        coils = kspace_slice.shape[0]

        # 1) Full k-space → iFFT → complex image
        img_full = ifft2c(kspace_slice.astype(np.complex64))  # (coils, H_raw, W_raw)

        # 2) Image 도메인 center crop → (coils, N_OUTPUT, N_OUTPUT)
        img_crop = complex_center_crop(img_full, (self.N_OUTPUT, self.N_OUTPUT))

        # 3) Cropped image → FFT → k-space at target resolution
        ksp_crop = fft2c(img_crop)  # (coils, N_OUTPUT, N_OUTPUT)

        # 4) R=4 mask (phase-encoding = 마지막 축 W)
        mask = self.mask_1d[np.newaxis, np.newaxis, :]  # (1, 1, W)
        ksp_masked = ksp_crop * mask                     # (coils, N, N)

        # 5) Masked k-space → iFFT → aliased image
        img_alias = ifft2c(ksp_masked)                   # (coils, N, N) complex

        # 6) 코일 16개로 맞춤 (brain AXFLAIR는 이미 16이지만 안전장치)
        take = min(coils, self.N_COIL_CH)
        ksp_out   = np.zeros((self.N_COIL_CH, self.N_OUTPUT, self.N_OUTPUT), dtype=np.complex64)
        img_out   = np.zeros((self.N_COIL_CH, self.N_OUTPUT, self.N_OUTPUT), dtype=np.complex64)
        ksp_out[:take] = ksp_masked[:take]
        img_out[:take] = img_alias[:take]

        data     = self._pack_complex_to_real(ksp_out) * self.val_amp_X_ksp  # (32, N, N)
        data_img = self._pack_complex_to_real(img_out) * self.val_amp_X_img  # (32, N, N)

        # 7) Ground Truth
        if gt_rss is not None:
            # reconstruction_rss는 이미 magnitude (N_OUTPUT × N_OUTPUT 예상)
            if gt_rss.shape != (self.N_OUTPUT, self.N_OUTPUT):
                gt_rss = center_crop_real(gt_rss, (self.N_OUTPUT, self.N_OUTPUT))
        else:
            # reconstruction_rss가 없으면 cropped full image로부터 RSS 계산 (test 셋 fallback)
            img_full_take = img_crop[:take]  # (take, N, N) complex
            gt_rss = np.sqrt(np.sum(np.abs(img_full_take) ** 2, axis=0)).astype(np.float32)

        label = gt_rss[np.newaxis] * self.val_amp_Y  # (1, N, N)

        return {
            'data':     data.astype(np.float32),
            'data_img': data_img.astype(np.float32),
            'label':    label.astype(np.float32),
        }


if __name__ == '__main__':
    dataset = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_val', num_files=2
    )
    loader = DataLoader(dataset, batch_size=1)
    for batch in loader:
        print("data     (aliased ksp):", batch['data'].shape,
              batch['data'].min().item(), batch['data'].max().item())
        print("data_img (aliased img):", batch['data_img'].shape,
              batch['data_img'].min().item(), batch['data_img'].max().item())
        print("label    (GT RSS):    ", batch['label'].shape,
              batch['label'].min().item(), batch['label'].max().item())
        break
