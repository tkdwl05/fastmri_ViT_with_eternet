# 공개 모델 기준선 — 전체 검증셋 (우리 프로토콜 행: 384² 재-FFT · 16코일 절단 · R4 · brain-masked · per-slice LS 정합)

- 생성: `v8_eter_pure/eval_baselines_full.py --summary` (2026-09-06 21:33)
- 우리 3모델: `results/eval/v9_unleashed/per_slice_paired_v9.csv` (GPU fp16 평가, LS 정합 없음 α≈1) · zero-filled: `results/eval/zero_filled/per_slice_zero_filled.csv` (raw)
- 공개 모델은 CPU fp32 추론(정본 12 슬라이스에서 GPU 와 SSIM ≤0.003 · PSNR ≤0.3 dB 차이 확인) · non-finite 출력은 0 으로 치환 후 집계(개수 별도 표기)
- † = fastMRI leaderboard 공개 가중치(train+val 합본 학습 → 본 검증셋이 학습 데이터에 포함, 누수 참고선). PromptMR+ = train 구획만 학습(누수 없음)·인접 5슬라이스 입력(정보량 다름)
- 처리 순서가 interleaved 라 미완 상태의 완료 집합도 전 볼륨에 고른 층화 표본이다(중간 집계는 n 을 반드시 함께 인용).

## 진행 상황

| 방법 | 완료 슬라이스 | non-finite | 평균 s/slice |
|---|---|---|---|
| U-Net† | 11 / 7334 | 0 | 26.6 |
| E2E-VarNet† | 7334 / 7334 | 0 | 8.3 |
| PromptMR+ | 1486 / 7334 | 0 | 41.2 |

## A. 방법별 비교 (각 공개 모델의 완료 집합; 우리 모델도 같은 슬라이스로 재평균)

### U-Net† — 슬라이스 11 / 볼륨 8

volume 단위 mean±SD (n=8, SD ddof=1)

| Method | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ | L1 (×10⁻⁶) ↓ |
|---|---|---|---|---|
| Zero-filled (raw) | 0.7056±0.0443 | 24.99±1.96 | 1.990±0.456 | 12.17±3.00 |
| U-Net† | 0.8712±0.0281 | 30.75±1.87 | 0.525±0.097 | 6.44±1.31 |
| bi-GRU (original) | 0.8926±0.0324 | 32.84±2.40 | 0.326±0.066 | 5.20±0.66 |
| SS2D (controlled) | 0.8946±0.0333 | 32.97±2.42 | 0.315±0.057 | 5.12±0.61 |
| Enhanced SS2D | 0.8950±0.0332 | 33.00±2.46 | 0.315±0.068 | 5.10±0.66 |

slice 단위 mean±SD (n=11, SD ddof=1)

| Method | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ | L1 (×10⁻⁶) ↓ |
|---|---|---|---|---|
| Zero-filled (raw) | 0.7111±0.0440 | 25.24±2.11 | 1.881±0.439 | 12.21±2.52 |
| U-Net† | 0.8780±0.0282 | 31.03±2.04 | 0.494±0.102 | 6.42±1.11 |
| bi-GRU (original) | 0.8981±0.0302 | 33.21±2.42 | 0.301±0.072 | 5.18±0.57 |
| SS2D (controlled) | 0.8998±0.0310 | 33.30±2.44 | 0.294±0.063 | 5.14±0.53 |
| Enhanced SS2D | 0.9011±0.0311 | 33.40±2.49 | 0.290±0.074 | 5.07±0.57 |

U-Net† 우위 비율 (슬라이스 / 볼륨) · Δ 볼륨평균(공개−우리) · Wilcoxon(볼륨 paired)

| 대비 | 지표 | 우위 슬라이스 % | 우위 볼륨 % | Δ 볼륨 평균 | p (볼륨) |
|---|---|---|---|---|---|
| vs bi-GRU (original) | ssim | 0.0 | 0.0 | -0.0214 | 7.81e-03 |
| vs bi-GRU (original) | psnr | 0.0 | 0.0 | -2.0939 | 7.81e-03 |
| vs bi-GRU (original) | nmse | 0.0 | 0.0 | +0.1995 | 7.81e-03 |
| vs bi-GRU (original) | l1 | 0.0 | 0.0 | +1.2467 | 7.81e-03 |
| vs SS2D (controlled) | ssim | 0.0 | 0.0 | -0.0234 | 7.81e-03 |
| vs SS2D (controlled) | psnr | 0.0 | 0.0 | -2.2269 | 7.81e-03 |
| vs SS2D (controlled) | nmse | 0.0 | 0.0 | +0.2106 | 7.81e-03 |
| vs SS2D (controlled) | l1 | 0.0 | 0.0 | +1.3211 | 7.81e-03 |
| vs Enhanced SS2D | ssim | 0.0 | 0.0 | -0.0238 | 7.81e-03 |
| vs Enhanced SS2D | psnr | 0.0 | 0.0 | -2.2476 | 7.81e-03 |
| vs Enhanced SS2D | nmse | 0.0 | 0.0 | +0.2101 | 7.81e-03 |
| vs Enhanced SS2D | l1 | 0.0 | 0.0 | +1.3431 | 7.81e-03 |

### E2E-VarNet† — 슬라이스 7334 / 볼륨 464

volume 단위 mean±SD (n=464, SD ddof=1)

| Method | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ | L1 (×10⁻⁶) ↓ |
|---|---|---|---|---|
| Zero-filled (raw) | 0.7523±0.0410 | 24.76±2.11 | 3.935±2.166 | 24.85±8.84 |
| E2E-VarNet† | 0.9181±0.0386 | 32.78±3.21 | 1.133±1.291 | 10.51±5.41 |
| bi-GRU (original) | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 | 9.01±2.57 |
| SS2D (controlled) | 0.9141±0.0365 | 33.91±1.90 | 0.438±0.283 | 8.89±2.59 |
| Enhanced SS2D | 0.9146±0.0361 | 33.92±1.90 | 0.439±0.304 | 8.89±2.67 |

slice 단위 mean±SD (n=7334, SD ddof=1)

| Method | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ | L1 (×10⁻⁶) ↓ |
|---|---|---|---|---|
| Zero-filled (raw) | 0.7521±0.0767 | 24.76±2.96 | 3.938±3.806 | 24.80±11.48 |
| E2E-VarNet† | 0.9180±0.0986 | 32.78±4.35 | 1.135±3.534 | 10.48±6.59 |
| bi-GRU (original) | 0.9126±0.0894 | 33.78±2.59 | 0.449±0.525 | 9.00±3.05 |
| SS2D (controlled) | 0.9140±0.0890 | 33.90±2.63 | 0.438±0.526 | 8.88±3.05 |
| Enhanced SS2D | 0.9145±0.0875 | 33.92±2.63 | 0.439±0.536 | 8.88±3.13 |

E2E-VarNet† 우위 비율 (슬라이스 / 볼륨) · Δ 볼륨평균(공개−우리) · Wilcoxon(볼륨 paired)

| 대비 | 지표 | 우위 슬라이스 % | 우위 볼륨 % | Δ 볼륨 평균 | p (볼륨) |
|---|---|---|---|---|---|
| vs bi-GRU (original) | ssim | 72.7 | 70.0 | +0.0054 | 2.19e-18 |
| vs bi-GRU (original) | psnr | 47.9 | 39.2 | -1.0003 | 1.31e-11 |
| vs bi-GRU (original) | nmse | 47.9 | 28.9 | +0.6843 | 1.44e-36 |
| vs bi-GRU (original) | l1 | 53.6 | 46.3 | +1.5000 | 1.74e-04 |
| vs SS2D (controlled) | ssim | 70.8 | 66.8 | +0.0040 | 1.74e-12 |
| vs SS2D (controlled) | psnr | 45.5 | 35.3 | -1.1259 | 2.31e-16 |
| vs SS2D (controlled) | nmse | 45.5 | 27.8 | +0.6950 | 2.04e-40 |
| vs SS2D (controlled) | l1 | 51.1 | 43.5 | +1.6195 | 1.12e-07 |
| vs Enhanced SS2D | ssim | 70.3 | 65.7 | +0.0035 | 2.53e-10 |
| vs Enhanced SS2D | psnr | 45.1 | 34.9 | -1.1427 | 7.30e-17 |
| vs Enhanced SS2D | nmse | 45.1 | 27.6 | +0.6941 | 2.08e-40 |
| vs Enhanced SS2D | l1 | 50.5 | 42.5 | +1.6232 | 7.02e-08 |

### PromptMR+ — 슬라이스 1486 / 볼륨 464

volume 단위 mean±SD (n=464, SD ddof=1)

| Method | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ | L1 (×10⁻⁶) ↓ |
|---|---|---|---|---|
| Zero-filled (raw) | 0.7561±0.0416 | 24.87±2.13 | 3.768±2.218 | 25.26±9.41 |
| PromptMR+ | 0.9497±0.0353 | 36.38±4.13 | 0.497±0.884 | 8.12±5.89 |
| bi-GRU (original) | 0.9204±0.0366 | 33.95±1.97 | 0.421±0.303 | 9.13±2.77 |
| SS2D (controlled) | 0.9218±0.0365 | 34.09±1.99 | 0.409±0.303 | 9.00±2.76 |
| Enhanced SS2D | 0.9222±0.0361 | 34.10±2.02 | 0.414±0.352 | 9.00±2.90 |

slice 단위 mean±SD (n=1486, SD ddof=1)

| Method | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ | L1 (×10⁻⁶) ↓ |
|---|---|---|---|---|
| Zero-filled (raw) | 0.7561±0.0591 | 24.87±2.75 | 3.768±3.139 | 25.22±11.70 |
| PromptMR+ | 0.9498±0.0557 | 36.39±5.00 | 0.497±1.162 | 8.10±7.03 |
| bi-GRU (original) | 0.9204±0.0553 | 33.95±2.42 | 0.424±0.414 | 9.13±3.17 |
| SS2D (controlled) | 0.9218±0.0552 | 34.08±2.45 | 0.412±0.410 | 9.00±3.15 |
| Enhanced SS2D | 0.9221±0.0544 | 34.09±2.46 | 0.416±0.444 | 9.00±3.25 |

PromptMR+ 우위 비율 (슬라이스 / 볼륨) · Δ 볼륨평균(공개−우리) · Wilcoxon(볼륨 paired)

| 대비 | 지표 | 우위 슬라이스 % | 우위 볼륨 % | Δ 볼륨 평균 | p (볼륨) |
|---|---|---|---|---|---|
| vs bi-GRU (original) | ssim | 97.4 | 99.1 | +0.0293 | 1.36e-77 |
| vs bi-GRU (original) | psnr | 85.7 | 84.1 | +2.4247 | 1.69e-34 |
| vs bi-GRU (original) | nmse | 85.7 | 81.7 | +0.0761 | 9.69e-17 |
| vs bi-GRU (original) | l1 | 85.5 | 82.8 | -1.0101 | 1.44e-20 |
| vs SS2D (controlled) | ssim | 97.3 | 99.1 | +0.0279 | 1.58e-77 |
| vs SS2D (controlled) | psnr | 85.5 | 83.0 | +2.2916 | 7.41e-33 |
| vs SS2D (controlled) | nmse | 85.5 | 80.8 | +0.0887 | 5.68e-16 |
| vs SS2D (controlled) | l1 | 85.5 | 82.3 | -0.8750 | 6.41e-20 |
| vs Enhanced SS2D | ssim | 96.7 | 98.9 | +0.0276 | 1.75e-77 |
| vs Enhanced SS2D | psnr | 85.5 | 83.4 | +2.2784 | 4.09e-32 |
| vs Enhanced SS2D | nmse | 85.5 | 80.8 | +0.0839 | 4.32e-16 |
| vs Enhanced SS2D | l1 | 85.4 | 82.5 | -0.8817 | 8.49e-20 |

## B. 전 방법 공통 집합 — 아직 8 슬라이스(<50) 라 생략

(우위 비율 = proportion favoring the public model; nMSE·L1 은 낮을수록 우위. 이 표는 참고선이며 순위 판정에 쓰지 않는다 — †는 누수, PromptMR+ 는 다중 슬라이스 입력·물리 모델 계열.)
