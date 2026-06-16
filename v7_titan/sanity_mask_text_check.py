"""
brain_mask 의 수치 + ASCII 시각화 (PNG 못 보는 환경용).

검증 지표:
  - ratio: brain 픽셀 비율 (30~45% 가 정상)
  - centroid: 무게중심 (이미지 중심 ±30 이내 정상)
  - bbox: 가로/세로 (180~280 정도 정상)
  - n_components: connected components (1~2 — 좌우 반구 분리 시 2)
  - solidity: 면적 / convex_hull 면적 (0.85+ 정상, 낮으면 거칠거나 구멍)

ASCII 시각화: 384x384 → 48x24 (16배 다운샘플), '#' = brain, '.' = bg.
"""

import os
import sys
import numpy as np
from scipy.ndimage import label as ndi_label, find_objects, center_of_mass, binary_fill_holes
try:
    from skimage.morphology import convex_hull_image
    HAS_SKIMAGE_MORPH = True
except ImportError:
    HAS_SKIMAGE_MORPH = False

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'dataloaders'))

from myConfig_choh_SS2D_model_v7_titan import IMAGE_SIZE
from dataloader_h5_v5 import FastMRI_H5_Dataloader


def ascii_visualize(mask: np.ndarray, target: np.ndarray, scale: int = 16):
    """384x384 → 384/16 = 24 행 (행은 절반 줄임 ratio 보존) ASCII"""
    h_step = scale
    w_step = scale
    H, W = mask.shape
    rows_m, rows_t = [], []
    for r in range(0, H, h_step):
        row_m, row_t = [], []
        for c in range(0, W, w_step):
            patch_m = mask[r:r+h_step, c:c+w_step]
            patch_t = target[r:r+h_step, c:c+w_step]
            # mask: 평균값 → 임계로 character
            mv = float(patch_m.mean())
            row_m.append('#' if mv > 0.5 else ('+' if mv > 0.15 else '.'))
            # target: max-normalized intensity → character
            tv = float(patch_t.max() / max(target.max(), 1e-9))
            if tv > 0.5:
                row_t.append('#')
            elif tv > 0.2:
                row_t.append('+')
            elif tv > 0.05:
                row_t.append('.')
            else:
                row_t.append(' ')
        rows_m.append(''.join(row_m))
        rows_t.append(''.join(row_t))
    return rows_t, rows_m


def stats_for_mask(mask: np.ndarray):
    H, W = mask.shape
    total = H * W
    area = int(mask.sum())
    ratio = area / total

    # centroid
    if area > 0:
        cy, cx = center_of_mass(mask)
    else:
        cy = cx = float('nan')

    # connected components
    lbl, n_comp = ndi_label(mask)

    # bbox of full mask
    if area > 0:
        ys, xs = np.where(mask > 0)
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
        bbox_h = bbox[2] - bbox[0]
        bbox_w = bbox[3] - bbox[1]
    else:
        bbox = (0, 0, 0, 0)
        bbox_h = bbox_w = 0

    # solidity (area / convex_hull area). skimage 가능 시.
    if HAS_SKIMAGE_MORPH and area > 0:
        hull = convex_hull_image(mask.astype(bool))
        hull_area = int(hull.sum())
        solidity = area / max(hull_area, 1)
    else:
        solidity = float('nan')

    # holes inside mask
    filled = binary_fill_holes(mask.astype(bool))
    holes = int(filled.sum()) - area

    return {
        'area':         area,
        'ratio':        ratio,
        'cy':           cy,
        'cx':           cx,
        'n_components': n_comp,
        'bbox':         bbox,
        'bbox_hw':      (bbox_h, bbox_w),
        'solidity':     solidity,
        'holes':        holes,
    }


def main(n_samples=6):
    print('=' * 60)
    print(' brain_mask 수치 + ASCII 검증 (PNG 없이)')
    print('=' * 60)
    ds = FastMRI_H5_Dataloader(
        './fastMRI_data/multicoil_val', num_files=3,
        target_size=IMAGE_SIZE[0], augment=False, random_mask=False,
    )
    n = min(n_samples, len(ds))
    print(f'samples = {n}, IMAGE_SIZE = {IMAGE_SIZE}')

    all_ratios = []
    for i in range(n):
        idx = i * (len(ds) // n)
        s = ds[idx]
        mask   = s['brain_mask'][0]
        target = s['label'][0]
        st = stats_for_mask(mask)
        all_ratios.append(st['ratio'])

        print(f'\n────── sample {i + 1} / {n}  (idx={idx}) ──────')
        print(f'  area={st["area"]:6d}  ratio={st["ratio"]*100:5.1f}%'
              f'  centroid=({st["cy"]:5.1f}, {st["cx"]:5.1f})  '
              f'(이미지 중심 ~{IMAGE_SIZE[0] / 2:.0f})')
        print(f'  bbox=(y0={st["bbox"][0]:3d}, x0={st["bbox"][1]:3d},'
              f' y1={st["bbox"][2]:3d}, x1={st["bbox"][3]:3d})'
              f'   bbox h×w = {st["bbox_hw"][0]}×{st["bbox_hw"][1]}')
        print(f'  n_components={st["n_components"]}  '
              f'solidity={st["solidity"]:.3f}  holes={st["holes"]}')

        # ASCII (target vs mask 가로 나란히)
        if i < 3:    # 처음 3개만 ASCII 출력 (스팸 방지)
            rows_t, rows_m = ascii_visualize(mask, target, scale=16)
            print(f'  ASCII (24×24, 16× downsample):  TARGET (left) | MASK (right)')
            for rt, rm in zip(rows_t, rows_m):
                print(f'    {rt}  |  {rm}')

    print('\n────── summary ──────')
    print(f'  ratio range:  min={min(all_ratios)*100:.1f}%  '
          f'max={max(all_ratios)*100:.1f}%  '
          f'mean={float(np.mean(all_ratios))*100:.1f}%')
    print('\n  정상 범위 가이드:')
    print('    ratio:       25 ~ 45%   (너무 작으면 brain 가장자리 잘림, 너무 크면 배경 포함)')
    print('    centroid:    180 ~ 210  (이미지 중심 192 근처)')
    print('    n_components: 1 ~ 3     (1: 한 덩어리, 2~3: 좌우반구 분리 또는 작은 외부 노이즈)')
    print('    bbox h×w:    180~280    (정상 brain crop)')
    print('    solidity:    >0.85      (낮으면 mask 가 거칠거나 구멍 많음)')
    print('    holes:        0 ~ 100   (eye/ventricle 등 작은 구멍 허용)')


if __name__ == '__main__':
    main()
