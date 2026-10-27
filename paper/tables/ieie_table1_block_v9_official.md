%% IEIE .src.md 붙여넣기용 — v9 표 1 대안 (brain mask 행 묶음 + fastMRI 공식 정의 행 묶음, 같은 세 모델·같은 5열). paper/make_tables_v8fix.py 가 생성 — 손편집 금지
@table: col | 1180,540,1015,870,930
@cap_en: Volume-level results (mean ± standard deviation) computed inside the brain mask and with the fastMRI official definition on the fastMRI brain multicoil test_full set (558 volumes, R=4); each model was trained once (seed 1), and its checkpoint was selected on the official validation split (1,378 volumes)
| Model | Params (M) | SSIM | PSNR (dB) | nMSE (%) |
|---|---|---|---|---|
| **Brain mask** | | | | |
| U-Net only | 31.1 | 0.9213±0.0466 | 34.91±1.85 | 0.371±0.409 |
| ETER-net | 668.2 | 0.9191±0.0474 | 34.69±1.84 | 0.388±0.434 |
| SS2D(controlled) | 31.2 | 0.9199±0.0469 | 34.78±1.86 | 0.381±0.414 |
| **fastMRI official** | | | | |
| U-Net only | 31.1 | 0.8965±0.0400 | 34.81±2.38 | 1.504±0.641 |
| ETER-net | 668.2 | 0.8652±0.0399 | 32.88±2.13 | 2.303±0.811 |
| SS2D(controlled) | 31.2 | 0.8389±0.0575 | 32.04±2.33 | 2.815±0.926 |
@note: Note. Brain mask 행의 지표는 brain mask 내부에서 계산하였다. fastMRI official 행의 지표는 fastMRI 공식 평가 정의에 따라 마스크를 쓰지 않고, 참조 영상과 복원 영상을 참조 영상의 가로 크기를 한 변으로 하는 정사각형으로 가운데를 잘라 계산하였으며, 동적 범위(data range)는 참조 볼륨의 최댓값으로 두었다. 이 정의에서 SSIM은 슬라이스별 값을 볼륨마다 평균한 값이고, PSNR과 nMSE는 볼륨 전체에서 한 번에 계산한 값이다. 공식 정의는 머리 밖 배경을 포함하는데, 학습 손실은 brain mask 내부에서만 계산하였으므로 어느 모델도 이 배경을 학습하지 않았다. U-Net only는 시퀀스 모듈의 출력 20채널을 0으로 대체하고, 다른 모델과 같은 구조의 U-Net을 처음부터 학습한 모델이다. 각 모델은 난수 시드를 1로 고정하여 50 epochs 동안 한 번만 학습하였으므로, 표의 수치는 모델별로 한 번의 학습 결과를 나타낸다. 본문에 인용한 차이는 같은 볼륨끼리 짝지은 차이의 평균(반올림 전 값)이므로, 반올림한 평균끼리 뺀 값과 마지막 자리에서 다를 수 있다.
@end
