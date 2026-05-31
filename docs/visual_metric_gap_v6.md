# Visual-Metric Gap 분석 및 개선 계획 (v6)

## 배경

전체 val 셋 7270 슬라이스 평가 결과 ([results/eval_full_v6_summary.txt](../results/eval_full_v6_summary.txt)):

| 모델 | PSNR (dB) | SSIM | NMSE |
|---|---:|---:|---:|
| SS2D-ViT v6 | 34.81 ± 2.78 | 0.8913 ± 0.0974 | 0.00906 |
| ETER-ViT v6 | 33.59 ± 2.74 | 0.8862 ± 0.0938 | 0.01143 |
| U-Net (PT) | 34.66 ± 2.95 | 0.8858 ± 0.1222 | 0.00737 |

수치상 SS2D 가 U-Net 을 SSIM 으로 능가하지만, [results/vis_compare_v6/compare_3230.png](../results/vis_compare_v6/compare_3230.png) 등 시각화 결과는 GT 대비 sulci/혈관 같은 fine detail 이 평균화된 흐릿한 출력으로 보이고, 에러맵은 brain 전체에 산재돼 있어 "정량 0.89 SSIM 인데 시각적으로는 그만큼 좋아 보이지 않는다" 는 인상을 줌.

이 문서는 메트릭과 시각 인상의 괴리 원인 4가지와 진단 도구 개선 계획을 정리한다.

---

## 원인

### 1. raw amplitude 스케일의 SSIM 부풀림

fastMRI brain `.h5` 의 raw 값은 1e-5 ~ 1e-4 범위. 슬라이스의 절반 이상이 배경 (~0). 현재 SSIM 계산:

```python
ssim = compare_ssim(target, pred, data_range=target.max() - target.min())
```

- `data_range` 가 한두 픽셀의 outlier 에 좌우 → 분모 과대 → 정규화 에러 과소
- 배경 픽셀은 자동으로 "잘 맞은" 상태로 SSIM 평균을 끌어올림
- fastMRI 공식 leaderboard 는 brain mask + per-volume 정규화 후 평가 → 우리 측정과는 다른 정의

**영향**: 보고된 SSIM 0.89 가 brain mask 기반으로 환산되면 0.83 ~ 0.87 수준일 가능성.

### 2. 시각화 슬라이스 (3230) 가 평균 이하

[compare_3230.png](../results/vis_compare_v6/compare_3230.png) 의 per-slice 메트릭:
- SS2D 0.8506 / ETER 0.8392 / UNet 0.8485

데이터셋 평균 (0.8913 / 0.8862 / 0.8858) 대비 모두 ~0.04 낮음. val SSIM 표준편차 0.0974 (eval CSV 기준) 이므로 슬라이스 간 편차가 큼. 3230 은 brain 외곽 (sulci/외곽 노이즈 비중↑) 으로 추정되는 어려운 케이스.

**영향**: 보여진 이미지가 best-case 도 average-case 도 아닌 below-average 라 모델 능력이 시각적으로 과소 평가됨.

### 3. L1 + SSIM loss 의 mean-prediction 편향

- L1 loss → 픽셀별 중앙값 예측 → 디테일 평균화
- SSIM loss → local mean/variance 유사도만 평가 → 부드러운 출력을 처벌하지 않음
- 결과: 모델이 fine vasculature / 작은 sulci 같이 "확신 없는" 텍스처를 평균화 → 흐릿한 brain
- SSIM 은 텍스처 유사도가 비슷하면 높은 점수 → 흐릿함이 SSIM 점수에 잘 안 잡힘

**영향**: 모델이 "흐릿하지만 SSIM 적당" 영역에 수렴. 시각 품질과 SSIM 의 디커플.

### 4. 에러맵 colormap 의 조기 saturation

[visualize_compare.py:157](../visualize_compare.py#L157):
```python
error_max = max(ss2d_error.max(), eter_error.max(), unet_error.max(), 1e-8) * 0.5
```

- `* 0.5` 라 가장 큰 에러의 절반에서 colormap 이 white 까지 도달
- `'hot'` colormap 은 black → red → yellow → white. 중간 강도가 강렬한 red
- 결과: 절대 에러가 작아도(컬러바 5e-5 수준) brain 전반이 "빨갛게" 보임

**영향**: 정량 NMSE 0.009 (작은 에러) 가 시각적으로는 큰 에러처럼 보임.

---

## 개선 계획

진단용 새 스크립트 `visualize_diagnostic_v6.py` 에 4가지 개선을 한 번에 적용:

### Step 1: best/worst/median 슬라이스 자동 선정 (원인 2 대응)

- 입력: `results/eval_full_v6.csv`
- 기준: SS2D SSIM 컬럼
- 선정: top-3 (best) / bottom-3 (worst) / median 부근 3개 = **총 9 슬라이스**
- 이유: 분포의 양 끝과 중앙을 모두 보여 단일 슬라이스 편향 제거 + 모델의 실패 케이스도 정직하게 노출

### Step 2: brain-mask SSIM 추가 측정 (원인 1 대응)

- mask: `gt > 0.05 * gt.max()` (배경 제외, 약한 brain tissue 까지 포함)
- mask 내부 픽셀만으로 SSIM / PSNR / NMSE 재측정 (`data_range = gt[mask].max() - gt[mask].min()`)
- 출력: raw SSIM 과 masked SSIM 을 타이틀에 둘 다 표기 → 메트릭이 배경 덕에 얼마나 부풀려지는지 정량화

### Step 3: zoom-in crop 추가 (원인 3 대응)

- 각 비교 이미지에 brain 중앙 160×160 zoom-in 행을 추가
- 레이아웃: **3 행 × 4 열** (full / zoom / error_zoom)
- fine detail (sulci, vasculature) 비교가 시각적으로 가능해지면, L1 loss 의 흐릿함 편향이 어디서 드러나는지 명확

### Step 4: percentile normalization (원인 4 대응)

- 표시 windowing: `vmin = np.percentile(gt, 1)`, `vmax = np.percentile(gt, 99)` → outlier 1% 잘라 contrast 정상화
- 에러맵: `error_vmax = np.percentile(error, 99)` → 가장 밝은 1 % 에러로 인한 colormap 포화 방지
- 결과: 흐릿함과 에러 패턴 둘 다 정량 metric 에 가깝게 시각화

---

## 작업 순서

1. **본 문서 작성** (= 이 파일)
2. **`visualize_diagnostic_v6.py` 구현**
   - eval CSV 로부터 9 슬라이스 인덱스 선정
   - load_ss2d / load_eter / load_unet 은 `visualize_compare.py` 와 동일 (v6 호환 v5 클래스)
   - brain mask + masked SSIM 함수
   - 3×4 figure 빌더 (full / zoom / error_zoom)
   - 결과: `results/vis_diagnostic_v6/` 에 9 PNG + 종합 summary
3. **실행 → 결과 검토**
4. **[CLAUDE.md](../CLAUDE.md) 핵심 문서 섹션에 본 문서 등록**

## 기대 결과

- raw SSIM vs masked SSIM 차이 ≥ 0.03 → 원인 1 확정 (배경 효과 정량화)
- best 슬라이스 시각화가 GT 와 거의 구별 안 됨 → 원인 2 확정 (3230 이 편향된 케이스)
- zoom-in 에서 SS2D/ETER 둘 다 fine vessel 손실, U-Net 은 더 심한 흐림 관찰 예상 → 원인 3 확정
- 새 에러맵에서 brain 외곽선/sulci 경계에 에러 집중, brain 내부는 거의 비어 보임 예상 → 원인 4 확정
