**Table 4. Public-model results — validation subset (464 volumes/7,334 slices) evaluated under our protocol (384×384 preprocessing, 16 coils, R = 4, brain-masked); mean±SD over volumes. Reported for reference only, hence no ranking marks. Last column: fraction of slices (SSIM / PSNR, %) on which the method outperforms SS2D (controlled).** †: public fastMRI leaderboard weights trained on the train+val split, so this validation set is part of their training data. PromptMR+: public weights trained on the train split only, but a 12-cascade unrolled model that takes five adjacent slices as input. All public weights were trained with their native coil configuration and are applied here to the 384×384-preprocessing/16-coil protocol (domain shift). Public-model metrics are computed after per-slice least-squares intensity scaling inside the brain mask (their output scales differ; the scaling can only favor them); the three rows for our models are the unscaled values of Table 2. CPU fp32 inference.

| Method | Training split | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ | Slices better than SS2D (%) SSIM / PSNR |
|---|---|---:|---:|---:|---:|---:|
| U-Net† | train+val | 496 | 0.8971±0.0366 | 30.95±2.29 | 0.973±0.796 | 7.7 / 3.9 |
| E2E-VarNet† | train+val | 30 | 0.9181±0.0386 | 32.78±3.21 | 1.133±1.291 | 70.8 / 45.5 |
| PromptMR+ | train | 93 | 0.9417±0.0349 | 36.12±4.02 | 0.526±0.841 | 96.9 / 85.4 |
| bi-GRU (original) | train | 668 | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 | 21.8 / 26.2 |
| SS2D (controlled) | train | 31 | 0.9141±0.0365 | 33.91±1.90 | 0.438±0.283 | – |
| SS2D (enhanced) | train | 34 | 0.9146±0.0361 | 33.92±1.90 | 0.439±0.304 | 55.8 / 54.2 |
