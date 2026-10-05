**Table 1 (slice-level). Quantitative comparison on the fastMRI brain multi-coil validation subset (7,334 slices, R = 4, brain-masked); mean±SD over slices, best in bold, second best underlined.**

| Method | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---:|---:|---:|---:|
| Zero-filled | – | 0.7521±0.0767 | 24.76±2.96 | 3.938±3.806 |
| bi-GRU (original) | 668 | 0.9126±0.0894 | 33.78±2.59 | 0.449±0.525 |
| SS2D (controlled) | 31 | __0.9140±0.0890__ | __33.90±2.63__ | **0.438±0.526** |
| SS2D (enhanced) | 34 | **0.9145±0.0875** | **33.92±2.63** | __0.439±0.536__ |

(Zero-filled = RSS of the inverse FFT of the undersampled k-space, no intensity rescaling; SD = sample standard deviation (ddof = 1). Public leaderboard U-Net/E2E-VarNet checkpoints were trained on train+val and are excluded from the ranking; see text.)
