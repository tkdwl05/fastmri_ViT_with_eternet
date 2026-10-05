**Table 1 (volume-level). Quantitative comparison on the fastMRI brain multi-coil validation subset (464 volumes, R = 4, brain-masked); mean±SD over volumes, best in bold, second best underlined.**

| Method | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---:|---:|---:|---:|
| Zero-filled | – | 0.7523±0.0410 | 24.76±2.11 | 3.935±2.166 |
| bi-GRU (original) | 668 | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 |
| SS2D (controlled) | 31 | __0.9141±0.0365__ | __33.91±1.90__ | **0.438±0.283** |
| SS2D (enhanced) | 34 | **0.9146±0.0361** | **33.92±1.90** | __0.439±0.304__ |

(Zero-filled = RSS of the inverse FFT of the undersampled k-space, no intensity rescaling; SD = sample standard deviation (ddof = 1). Public leaderboard U-Net/E2E-VarNet checkpoints were trained on train+val and are excluded from the ranking; see text.)
