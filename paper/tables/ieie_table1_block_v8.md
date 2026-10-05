%% IEIE .src.md 붙여넣기용 — v8 표 1 (볼륨 단위, 순위 표시 없음; run 1 = 시드 미고정, run 2 = 시드 1, U-Net only = 시드 1; 공개 모델 행 제외)
@table: col | 1180,540,1015,870,930
@cap_en: Volume-level results (mean ± standard deviation) on the fastMRI brain multicoil validation subset (464 volumes/7,334 slices, R=4, brain-masked)
| Method | Params (M) | SSIM | PSNR (dB) | nMSE (%) |
|---|---|---|---|---|
| Zero-filled | – | 0.7523±0.0410 | 24.76±2.11 | 3.935±2.166 |
| U-Net only | 31.1 | 0.9127±0.0366 | 33.76±1.88 | 0.452±0.285 |
| ETER-net, run 1 | 668.2 | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 |
| ETER-net, run 2 | 668.2 | 0.9136±0.0364 | 33.86±1.86 | 0.442±0.286 |
| SS2D(controlled), run 1 | 31.2 | 0.9141±0.0365 | 33.91±1.90 | 0.438±0.283 |
| SS2D(controlled), run 2 | 31.2 | 0.9133±0.0364 | 33.82±1.87 | 0.444±0.277 |
| SS2D(enhanced) | 34.2 | 0.9146±0.0361 | 33.92±1.90 | 0.439±0.304 |
@note: Note. Zero-filled는 언더샘플링된 k-space에 코일별 역 푸리에 변환을 적용한 코일 영상(처음 16개 코일)의 RSS 영상이며, 강도 배율 보정은 적용하지 않았다. U-Net only는 시퀀스 모듈의 출력 20채널을 0으로 대체하고, 다른 모델과 같은 구조의 U-Net을 처음부터 학습한 모델이다. run 1(1회차)은 난수 시드를 고정하지 않은 학습, run 2(2회차)와 U-Net only는 난수 시드를 1로 고정한 학습이며(모두 50 epochs), SS2D(enhanced)는 난수 시드를 고정하지 않고 80 epochs 동안 한 번 학습하였다. 본문의 SSIM 차이는 같은 볼륨끼리 짝지은 차이의 평균(반올림 전 값)이므로, 표의 반올림한 평균끼리 뺀 값과 마지막 자리에서 0.0001 다를 수 있다.
@end
