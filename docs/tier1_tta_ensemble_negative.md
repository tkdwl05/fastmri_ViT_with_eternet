# Tier 1 — TTA / Ensemble 무효 결과 (2026-05-25)

추가 학습 없이 SSIM/PSNR 끌어올리는 후보로 TTA(4-way flip 평균)와 v4/v6/cross-arch
앙상블을 시도. 500-sample (seed 42) 부분 평가에서 모두 임계 (+0.003 SSIM 또는
+0.15 dB PSNR) 미달.

## 실험 설정

- 평가 스크립트: `eval_tta_ensemble.py`
- 데이터: fastMRI brain multicoil val, 500 슬라이스 random subset (seed 42)
- 메트릭: skimage SSIM, fastMRI PSNR/NMSE, L1
- 모델: SS2D v4/v6, ETER v4/v6, UNet (PT)
- TTA: 4-way (identity / H / W / HW) — image domain flip → FFT 로 k-space 재계산 →
  mask(W) 도 flip → 모델 forward → 출력 un-flip → 평균

## 결과

| Method | PSNR (dB) | SSIM | vs SS2D v6 base |
|---|---:|---:|---|
| SS2D-ViT v6 (base) | 34.87 | 0.8968 | — |
| SS2D-ViT v6 (TTA)  | 34.52 | 0.8941 | **−0.35 dB / −0.0027** |
| ETER-ViT v6 (base) | 33.57 | 0.8903 | — |
| ETER-ViT v6 (TTA)  | 33.73 | 0.8923 | +0.16 / +0.0020 |
| Ens SS2D v4+v6     | 34.68 | 0.8942 | −0.19 / −0.0026 |
| Ens ETER v4+v6     | 32.59 | 0.8781 | (ETER 비교) −0.98 / −0.012 |
| Ens SS2D+ETER v6   | 34.67 | 0.8991 | −0.20 / **+0.0023** |
| Ens SS2D+ETER v6 TTA | 34.51 | 0.8975 | −0.36 / +0.0007 |

## 해석

- **TTA가 SS2D 에서 회귀**: v5/v6 의 train aug 가 random H/V flip 을 적용했으므로
  모델은 이미 flip-equivariant 에 가깝다. 4-way 평균은 약간의 blurring 만 추가.
- **TTA 가 ETER 에서만 미약 상승**: ETER 가 SS2D 보다 flip robustness 가 약함을 시사
  (DC block 없는 점이 한 가설). 하지만 +0.0020 SSIM 은 임계 미달.
- **v4 앙상블이 양쪽에서 회귀**: v4 가 모든 메트릭에서 v6 대비 1–3 dB 약함
  (앞서 풀평가 확정). 약한 모델 출력 평균은 강한 모델 출력에 노이즈만 추가.
- **Cross-arch (SS2D+ETER v6)**: SSIM +0.0023 / PSNR −0.20. SSIM 미세 상승이
  PSNR 의 0.20 dB 손실을 정당화하지 못함.

## 결론

TTA / 앙상블은 v6 baseline 을 의미있게 끌어올리지 못한다. Tier 2 (재학습) 로 이동.
이 doc 는 향후 “왜 TTA/앙상블 안 했나” 질문 대비.

## 결과 파일

- `results/eval_tta_500_summary.txt` — 11개 method × 4 metric
- `results/eval_tta_500.csv` — 슬라이스별 raw
- `runs/eval/run_eval_tta_500.log` — stdout
