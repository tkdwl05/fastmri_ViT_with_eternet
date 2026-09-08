%% IEIE .src.md 붙여넣기용 — 공개 모델 참고선(볼륨 단위, 순위 표시 없음). 열 폭 합 9400 twips(page). 미완주 방법은 [TBD] 셀.
@table: page | 1900,1000,800,1400,1300,1300,1700
@cap_ko: 공개 모델 참고선 — 전체 검증 집합(464 볼륨/7,334 슬라이스)을 본 논문과 동일한 프로토콜(384² 재-FFT·16코일·R=4·brain-masked)로 추론한 볼륨 단위 평균±표준편차. 참고선이므로 순위 표시(굵게·밑줄)는 두지 않는다. 마지막 열은 통제 SS2D보다 나은 슬라이스의 비율(SSIM / PSNR, %)
@cap_en: Public-model reference lines — full validation set (464 volumes/7,334 slices) inferred under the protocol of this paper (384² re-FFT, 16 coils, R = 4, brain-masked); mean±SD over volumes. Reference only, hence no ranking marks. Last column: fraction of slices on which the method beats the controlled SS2D (SSIM / PSNR, %)
| Method | Training split | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ | Favoring vs. SS2D (%) SSIM / PSNR |
|---|---|---|---|---|---|---|
| U-Net† | train+val | 496 | 0.8971±0.0366 | 30.95±2.29 | 0.973±0.796 | 7.7 / 3.9 |
| E2E-VarNet† | train+val | 30 | 0.9181±0.0386 | 32.78±3.21 | 1.133±1.291 | 70.8 / 45.5 |
| PromptMR+ | train | 93 | 0.9417±0.0349 | 36.12±4.02 | 0.526±0.841 | 96.9 / 85.4 |
| bi-GRU (original) | train | 668 | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 | 21.8 / 26.2 |
| SS2D (controlled) | train | 31 | 0.9141±0.0365 | 33.91±1.90 | 0.438±0.283 | – |
| Enhanced SS2D | train | 34 | 0.9146±0.0361 | 33.92±1.90 | 0.439±0.304 | 55.8 / 54.2 |
@note: †: public fastMRI leaderboard weights trained on the train+val split, so this validation set is part of their training data. PromptMR+: public weights trained on the train split only, but a 12-cascade unrolled model that takes five adjacent slices as input. All public weights were trained with their native coil configuration and are applied here to the 384² re-FFT/16-coil protocol (domain shift). Public-model metrics are computed after per-slice least-squares intensity alignment inside the brain mask (their output scales differ; the alignment can only favor them); the three rows of this paper are the unaligned values of Table 2. CPU fp32 inference.
@end
