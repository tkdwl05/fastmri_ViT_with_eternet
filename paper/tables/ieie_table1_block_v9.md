%% IEIE .src.md 붙여넣기용 — v9 표 1 (v8fix 50 epochs, test_full, 볼륨 단위, brain mask 내부, 순위 표시 없음, Zero-filled 행 없음 — 사용자 결정 D2). paper/make_tables_v8fix.py 가 생성 — 손편집 금지
@table: col | 1025,1340,1035,1135
@cap_en: Volume-level results (mean ± standard deviation) on the fastMRI brain test_full split (558 volumes, R=4, brain-masked)
| Model | SSIM | PSNR (dB) | nMSE (%) |
|---|---|---|---|
| U-Net only | 0.9213±0.0466 | 34.91±1.85 | 0.371±0.409 |
| ETER-net | 0.9191±0.0474 | 34.69±1.84 | 0.388±0.434 |
| SS2D<br>(controlled) | 0.9199±0.0469 | 34.78±1.86 | 0.381±0.414 |
@note: Note. 본문의 차이는 짝지은 차이의 평균(반올림 전)이므로, 반올림한 평균끼리 뺀 값과 마지막 자리가 다를 수 있다.
@end
