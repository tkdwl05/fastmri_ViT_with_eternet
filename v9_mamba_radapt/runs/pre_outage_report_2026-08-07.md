# radapt 정전 전 상태 보고 (자동 스냅샷)

- 생성: 2026-08-07 13:27 UTC = KST 08-07 22:27
- 사유: **2026-08-08 학교 정전 예정** — 8/7 시점 진행상황 보존 (요청 2026-08-05)
- run: PureETER_SS2D_V9_radapt_multiAR_brain384 (scratch 재기동 2026-08-05 02:42 UTC, BS8, multi-AR R∈{2,3,4,5,6,8}, 34.2M params)

## 진행 요약

- 완료 epoch: **20/80** · best val_composite(현재까지): **0.9290** (val 은 R4 고정, 매 2ep)
- 잔여: 60 epochs × ~2.75 h/ep ≈ 165h — 정전 복구·재개 시점 기준으로 ETA 재계산
- 참고 목표: unleashed R4 매치가 아니라 **R-sweep 완만함**(단일-R 대비)이 성공 기준. R4 val 은 unleashed(0.9203)보다 낮게 나오는 것이 정상.

## 학습 로그 전문 (logs/PureETER_SS2D_V9_radapt_multiAR_brain384/log.txt)
```
SCRATCH START run=PureETER_SS2D_V9_radapt_multiAR_brain384 BS=8 ACCUM=1 LR=0.0002 EPOCHS=80 params=34.2M multiAR=(2, 3, 4, 5, 6, 8)
Epoch 1/80  train_loss=13.1179  dc_alpha=0.8899
Epoch 2/80  train_loss=10.3987  val_composite=0.9094  val_ssim_m=0.9070  val_psnr=34.26  val_nmse=0.0054  val_l1=9.7820  dc_alpha=0.8647
Epoch 3/80  train_loss=9.9234  dc_alpha=0.9031
Epoch 4/80  train_loss=9.5377  val_composite=0.9175  val_ssim_m=0.9116  val_psnr=35.01  val_nmse=0.0044  val_l1=9.0510  dc_alpha=0.9046
Epoch 5/80  train_loss=9.3664  dc_alpha=0.9002
Epoch 6/80  train_loss=9.2149  val_composite=0.9224  val_ssim_m=0.9161  val_psnr=35.35  val_nmse=0.0040  val_l1=8.7225  dc_alpha=0.9037
Epoch 7/80  train_loss=9.1073  dc_alpha=0.9019
Epoch 8/80  train_loss=9.0283  val_composite=0.9232  val_ssim_m=0.9164  val_psnr=35.44  val_nmse=0.0039  val_l1=8.6387  dc_alpha=0.8994
Epoch 9/80  train_loss=8.9265  dc_alpha=0.8989
Epoch 10/80  train_loss=8.9363  val_composite=0.9249  val_ssim_m=0.9171  val_psnr=35.61  val_nmse=0.0037  val_l1=8.4729  dc_alpha=0.8986
Epoch 11/80  train_loss=8.7984  dc_alpha=0.8990
Epoch 12/80  train_loss=8.7341  val_composite=0.9252  val_ssim_m=0.9186  val_psnr=35.56  val_nmse=0.0038  val_l1=8.5453  dc_alpha=0.8980
Epoch 13/80  train_loss=8.7028  dc_alpha=0.8990
Epoch 14/80  train_loss=8.6391  val_composite=0.9243  val_ssim_m=0.9173  val_psnr=35.52  val_nmse=0.0039  val_l1=8.6649  dc_alpha=0.8958
Epoch 15/80  train_loss=8.6720  dc_alpha=0.8984
Epoch 16/80  train_loss=8.6460  val_composite=0.9276  val_ssim_m=0.9191  val_psnr=35.83  val_nmse=0.0035  val_l1=8.3009  dc_alpha=0.8958
Epoch 17/80  train_loss=8.6174  dc_alpha=0.8952
Epoch 18/80  train_loss=8.5690  val_composite=0.9281  val_ssim_m=0.9191  val_psnr=35.90  val_nmse=0.0035  val_l1=8.2177  dc_alpha=0.8972
Epoch 19/80  train_loss=8.5414  dc_alpha=0.8954
Epoch 20/80  train_loss=8.5948  val_composite=0.9290  val_ssim_m=0.9202  val_psnr=35.94  val_nmse=0.0035  val_l1=8.2184  dc_alpha=0.8953
```

## 프로세스 / GPU (스냅샷 시점)
```
(학습 프로세스 없음 — 8/7 21:00 KST 이후 깨끗한 정지 뒤라면 정상. 그 외 시점이면 RESUME_AFTER_OUTAGE.md 절차로 재기동)
1 %, 352 MiB
```

## ckpt 상태 (logs/PureETER_SS2D_V9_radapt_multiAR_brain384/ — 정전 시 이 파일들이 재개 지점)
```
total 1068648
drwxr-xr-x  2 root root      4096 Aug  7 13:26 .
drwxr-xr-x 13 root root      4096 Jul 30 22:52 ..
-rw-r--r--  1 root root      1975 Aug  7 13:26 log.txt
-rw-r--r--  1 root root 136777880 Aug  7 13:26 ss2d_v9_radapt_best.pt
-rw-r--r--  1 root root 136778480 Aug  6 08:08 ss2d_v9_radapt_epoch_10.pt
-rw-r--r--  1 root root 136778480 Aug  6 22:40 ss2d_v9_radapt_epoch_15.pt
-rw-r--r--  1 root root 136778480 Aug  7 13:26 ss2d_v9_radapt_epoch_20.pt
-rw-r--r--  1 root root 136778330 Aug  5 17:13 ss2d_v9_radapt_epoch_5.pt
-rw-r--r--  1 root root 410365321 Aug  7 13:26 ss2d_v9_radapt_last.pt
```

## 최근 학습 로그 tail
```
Epoch  21/80:   0%|          | 11/8129 [00:53<3:31:27,  1.56s/batch, LR=1.71e-04, Loss=9.9897, PSNR=37.79dB, a=0.895] [A
Epoch  21/80:   0%|          | 12/8129 [00:53<3:17:17,  1.46s/batch, LR=1.71e-04, Loss=9.9897, PSNR=37.79dB, a=0.895][A
Epoch  21/80:   0%|          | 12/8129 [00:55<3:17:17,  1.46s/batch, LR=1.71e-04, Loss=8.6707, PSNR=36.78dB, a=0.895][A
Epoch  21/80:   0%|          | 13/8129 [00:55<3:07:15,  1.38s/batch, LR=1.71e-04, Loss=8.6707, PSNR=36.78dB, a=0.895][A
```

## 정전 안전성 & 재개 (상세: RESUME_AFTER_OUTAGE.md)

- 정전으로 프로세스가 급사해도 안전: `*_last.pt` 는 매 epoch **atomic save**(tmp→rename) full-state → **true-resume 손실 ≤1 epoch**.
- 복구 후 재개(요약):
  ```bash
  nvidia-smi && python -c "import torch; print(torch.cuda.is_available())"   # NVML 확인
  cd /home/snorlax/shared/fastmri_ViT_with_eternet
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online MAX_RETRY=200 \
    setsid nohup bash v9_mamba_radapt/runs/run_ss2d_v9_radapt_autoresume.sh \
    > v9_mamba_radapt/runs/run_ss2d_v9_radapt.log 2>&1 < /dev/null & disown
  ```
