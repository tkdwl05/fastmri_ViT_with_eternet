%% IEIE .src.md 붙여넣기용 표 4 — 볼륨 단위 SSIM + 우위 슬라이스 비율(SSIM 기준, 괄호 = SSIM·PSNR·nMSE 범위)
@table: page | 1100,800,1100,1100,1100,2150,2150
@cap_ko: Contrast 하위 집단별 SSIM(볼륨 단위 평균, 가장 좋은 값 굵게)과 우위 슬라이스 비율. 우위 비율 칸은 SSIM 기준이며 괄호는 세 지표(SSIM·PSNR·nMSE)에 걸친 범위
@cap_en: Per-contrast SSIM (volume-level mean, best in bold) and fraction of slices on which the first model of each pair is better. The fraction cells are SSIM-based; parentheses give the range over the three metrics (SSIM, PSNR, nMSE)
| Contrast | n (volumes) | SSIM bi-GRU | SSIM SS2D (controlled) | SSIM SS2D (enhanced) | SS2D vs. bi-GRU (%) | SS2D (enhanced) vs. SS2D (controlled) (%) |
|---|---|---|---|---|---|---|
| AXFLAIR | 33 | 0.8716 | 0.8731 | **0.8745** | 76.8 (71.0–76.8) | 67.8 (64.3–67.8) |
| AXT1 | 32 | 0.9072 | 0.9086 | **0.9088** | 78.5 (68.7–78.5) | 48.2 (48.2–49.2) |
| AXT1POST | 99 | 0.9266 | 0.9283 | **0.9287** | 83.3 (77.6–83.3) | 54.3 (54.3–55.5) |
| AXT1PRE | 29 | 0.9031 | 0.9052 | **0.9054** | 84.7 (80.3–84.7) | 52.8 (47.4–52.8) |
| AXT2 | 271 | 0.9143 | 0.9156 | **0.9160** | 75.8 (72.6–75.8) | 56.0 (53.8–56.0) |
@end
