# 에러맵 v2 — 정규화 + Brain mask + optional scale matching

[visualize_compare_versions.py](../visualize_compare_versions.py) 의 에러맵 시각화 정책을 raw amplitude 비교 (v1) → 정규화 + masked (v2) 로 교체. 2026-06-01.

## 동기

v1 (raw amplitude) 의 3가지 문제 — [presentation_overview.md §4](presentation_overview.md), [visual_metric_gap_v6.md](visual_metric_gap_v6.md) 와 정합:

1. **Background bias (가장 큰 문제)** — fastMRI brain AXFLAIR 슬라이스의 ~50% 이상이 0 amplitude background. 에러맵 전체 면적 절반이 항상 검정 → 뇌 내부의 미세한 모델 간 차이가 시각적으로 묻힘. raw SSIM 도 동일 이유로 부풀려짐 (visual_metric_gap_v6.md 에서 확인).
2. **UNet 스케일 매칭의 임의성** — UNet 은 fastMRI `UnetDataTransform` 의 z-score 정규화로 학습되어 출력 amplitude ~`1e-4`, SS2D/ETER 는 H5 RSS magnitude (~`1e2`). v1 은 `unet_scale = gt.max() / unet_gt.max()` 으로 단일 픽셀 max 기준 1:1 매칭 — outlier 한 점에 좌우.
3. **SS2D/ETER magnitude bias** — L1 + SSIM loss 가 작은 곱 상수에 둔감해 모델 출력이 GT 대비 균일한 0.95× 스케일 오차를 가질 수 있음. 에러맵 전체에 약한 균일 색조 → 시각적 비교 불공정.

## 변경 내용

[visualize_compare_versions.py:311-409](../visualize_compare_versions.py#L311-L409) 의 플롯 블록 교체.

### 1. Per-slice [0, 1] 정규화

```python
gt_max = max(float(gt.max()), 1e-8)
unet_gt_max = max(float(unet_gt.max()), 1e-8)
gt_n           = gt           / gt_max
recons_n[k]    = recon[k]     / gt_max          # SS2D/ETER (H5 스케일)
unet_gt_n      = unet_gt      / unet_gt_max     # UNet 은 자기 도메인 max
unet_recon_n   = unet_recon   / unet_gt_max
```

모든 디스플레이는 `vmin=0, vmax=1` 의 dimensionless 단위. 모델별/도메인별 절대 magnitude 차이 제거.

### 2. Brain mask

```python
brain      = gt_n      > args.err_mask_thresh   # 기본 0.05 (5%)
unet_brain = unet_gt_n > args.err_mask_thresh   # UNet 은 자기 GT 기준 (도메인 일관)
masked_err = np.where(brain, |recon_n - gt_n|, np.nan)
hot_bad    = plt.get_cmap('hot').copy(); hot_bad.set_bad('white')
```

Background → 흰색. 뇌 내부 픽셀만 색조로 표시. matplotlib 의 `cmap.set_bad('white')` 가 NaN 을 흰색으로 처리.

Threshold 선택 (`err_mask_thresh=0.05`) 근거:
- 0.0 → 거의 모든 픽셀 통과 (background bias 그대로)
- 0.05 (5% of max) → fastMRI brain 슬라이스 기준 skull/CSF 경계 이상만 통과, air/배경 제외. 보수적.
- 0.1 → 일부 hypointense 조직 (CSF 등) 도 제외 — 너무 공격적
- Otsu 자동 threshold 도 가능하나 implementation 단순화 위해 고정값 사용.

### 3. (옵션) Per-slice least-squares scale matching

`--match-scale` flag (기본 off) 활성화 시:

```python
def maybe_match_scale(recon_n, gt_n, mask):
    r, g = recon_n[mask], gt_n[mask]
    α = ⟨r, g⟩ / ⟨r, r⟩       # arg min_α ||α·recon - gt||²
    return α · recon_n
```

목적: 모델의 **균일 magnitude bias 만 제거** → 구조적 오차 (sharpness, hallucination) 만 시각화. 단점: 모델의 실제 calibration 결함을 가림 → 정량 평가용 아닌 시각 비교 보조용.

권장 사용:
- **기본 (off)** — v6 vs v6_3 같은 fine-tune 비교에서는 magnitude bias 가 미미 → 보정 불필요
- **on** — v4 같이 명백한 scale 결함이 의심될 때, 구조 vs scale 기여 분리 디버깅

### 4. 에러맵 colormap 단위

| 항목 | v1 (raw) | v2 (masked) |
|---|---|---|
| 단위 | raw amplitude | normalized [0, 1] |
| vmax 기본 | `0.1 × gt.max()` (= 10% of GT max) | `0.05` (5% of normalized max) |
| 의미 | 절대 amplitude error | "GT 최대값의 5% 이상 오차" |
| 직관성 | 슬라이스마다 max 다름 → vmax 값도 다름 | 모든 슬라이스/모델 공통 단위 |

v2 의 vmax=0.05 는 v1 vmax=0.1×max 보다 2배 민감 (5% vs 10%). 뇌 내부 픽셀만 보여주므로 더 조밀한 스케일 적합.

## 그대로 둔 것 (v2 가 안 건드림)

- **정량 metric (PSNR/SSIM)** — 여전히 raw 픽셀 기준 (`calc_psnr`, `skimage.compare_ssim` on full image). 마스킹된 메트릭은 별도 작업 (eval_full_compare 에 추가 검토). suptitle 에 `metric=raw SSIM/PSNR` 명시.
- **모델 forward / 캐싱 / 출력 디렉토리 구조** — outer-loop 모델 로딩, 슬라이스 input 캐싱 그대로.
- **출력 파일명** — `compare_NNNN.png`. v1 결과는 `results/vis/aligned/vis_compare_versions/`, v2 는 `results/vis/aligned/vis_compare_versions_masked/` 로 분리 보존 (비교 가능).

## CLI 옵션 (신규/변경)

| Flag | 기본 | 의미 |
|---|---|---|
| `--err-vmax-frac` | 0.05 (변경: 0.1 → 0.05) | 정규화 단위 vmax |
| `--err-mask-thresh` | 0.05 (신규) | Brain mask threshold (gt.max() 비율) |
| `--match-scale` | False (신규) | Per-slice LS scale matching |
| `--out-dir` | `results/vis/aligned/vis_compare_versions_masked` (변경) | v1 결과 보존을 위해 신규 dir |

## 사용 예

```bash
# 기본 — v2 masked + normalize, scale match off
python visualize_compare_versions.py

# Scale matching 켜고 비교 (구조 오차만 강조)
python visualize_compare_versions.py --match-scale --out-dir results/vis/aligned/vis_compare_versions_masked_scaled

# Threshold 강화 (CSF 도 제외)
python visualize_compare_versions.py --err-mask-thresh 0.1

# v1 raw amplitude 재현 (혹시 비교 필요시) — 구버전 git 체크아웃 필요
git show HEAD:visualize_compare_versions.py > /tmp/v1.py
```

## 다음 단계 (선택)

- masked PSNR/SSIM 계산 추가 → suptitle 에 raw + masked 둘 다 표시.
- Otsu / multi-Otsu threshold 옵션.
- Error map 의 mean(over brain) 을 PNG 파일명에 prefix 로 추가 (`compare_NNNN_meanerr0.012.png`) → 시각 정렬용.
