%% IEIE .src.md 붙여넣기용 — 볼륨 단위 (권장, fastMRI 관례)
@table: col | 1120,560,1040,880,960
@cap_ko: fastMRI brain multi-coil 검증 집합(464 볼륨, R=4, brain-masked)의 정량 비교(볼륨 단위 평균±표준편차, 가장 좋은 값 굵게·두 번째로 좋은 값 밑줄)
@cap_en: Quantitative comparison on the fastMRI brain multi-coil validation subset (464 volumes, R = 4, brain-masked); mean±SD over volumes, best in bold, second best underlined
| Method | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---|---|---|---|
| Zero-filled | – | 0.7523±0.0410 | 24.76±2.11 | 3.935±2.166 |
| bi-GRU (original) | 668 | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 |
| SS2D (controlled) | 31 | __0.9141±0.0365__ | __33.91±1.90__ | **0.438±0.283** |
| SS2D (enhanced) | 34 | **0.9146±0.0361** | **33.92±1.90** | __0.439±0.304__ |
@end

%% 슬라이스 단위 변형 (현행 문서·초안의 대표 수치 0.9126/0.9140/0.9145 와 일치)
@table: col | 1120,560,1040,880,960
@cap_ko: fastMRI brain multi-coil 검증 집합(464 볼륨, R=4, brain-masked)의 정량 비교(슬라이스 단위 평균±표준편차, 가장 좋은 값 굵게·두 번째로 좋은 값 밑줄)
@cap_en: Quantitative comparison on the fastMRI brain multi-coil validation subset (7,334 slices, R = 4, brain-masked); mean±SD over slices, best in bold, second best underlined
| Method | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---|---|---|---|
| Zero-filled | – | 0.7521±0.0767 | 24.76±2.96 | 3.938±3.806 |
| bi-GRU (original) | 668 | 0.9126±0.0894 | 33.78±2.59 | 0.449±0.525 |
| SS2D (controlled) | 31 | __0.9140±0.0890__ | __33.90±2.63__ | **0.438±0.526** |
| SS2D (enhanced) | 34 | **0.9145±0.0875** | **33.92±2.63** | __0.439±0.536__ |
@end
