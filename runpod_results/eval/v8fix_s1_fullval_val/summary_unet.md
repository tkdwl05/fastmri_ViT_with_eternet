# v8fix (레시피 DL) — U-Net 단독 대조 모델 (시퀀스 모듈 제거) — 레시피 DL per-slice 평가

- 체크포인트: `logs/PureETER_UNET_noDC_R4_brain384_v8fix_s1_fullval/pure_unet_best.pt` — ep 48/50 (log.txt 검증 SSIM 최댓값으로 추정)
- 평가 슬라이스: 21,842 / 21,842 (볼륨 1378)
- 프로토콜: `eval_paired_v8_nodc.py` 와 동일하되 입력을 학습과 같이 샘플별 정규화하고 출력×s 로 원 단위 복원 (같은 val 순서, R=4 고정 마스크, brain-masked, batch 1, AMP, 지표 = `slice_metrics` import)
- sanity: 슬라이스 단위 SSIM 평균 = **0.9236** — log.txt ep 48 의 val_ssim_m = 0.9236 와 같아야 정상

## 1. 볼륨 단위 평균±표준편차 (n = 1378 볼륨, SD ddof=1 — `paper/make_tables.py` 표 1 과 같은 계산)

| Method | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---:|---:|---:|
| U-Net only | 0.9236±0.0362 | 34.95±1.77 | 0.347±0.197 |

### (참고) 슬라이스 단위 평균±표준편차 (n = 21,842 슬라이스, SD ddof=1 — 표 1 의 슬라이스 단위 변형)

| Method | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---:|---:|---:|
| U-Net only | 0.9236±0.0849 | 34.95±2.64 | 0.347±0.476 |

(nMSE (%) = 슬라이스 nMSE ×100. 볼륨 단위 = 볼륨별 슬라이스 평균을 낸 뒤 볼륨 사이 평균±SD.)

## 2. 대응(paired) 비교

(--ref 미지정 — 생략)


## 3. 기능 시험 (val 에 고르게 퍼진 512 슬라이스 — `smoke_recipe_fix.run_validation`, normalize=True)

- SSIM 하락 = 정상 출력 SSIM − 시험 출력 SSIM (슬라이스별, 원 단위). 판정 기준(스모크와 같음, 실행 전 고정): mean>0 & mean>2SE & sign-test p<0.01.
- 이 슬라이스들의 정상 SSIM 평균 0.9162 (전체 val 표와 표본이 달라 값이 다를 수 있음).

| 시험 | 내용 | SSIM 하락 평균 ± SE | 하락 슬라이스 | 부호 검정 p | 판정 |
|---|---|---:|---:|---:|---|
| deep4 | U-Net down_path.4 출력 0 대체 | +0.01086 ± 0.00025 | 510/512 | 2e-149 | used |
| deep34 | U-Net down_path.3 출력 0 대체 | +0.05985 ± 0.00105 | 510/512 | 2e-149 | used |
