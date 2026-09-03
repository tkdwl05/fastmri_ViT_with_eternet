# radapt ep57 경계 clean-stop 상태 보고 (자동 스냅샷, 2026-09-02)

- 생성: 2026-09-02 04:49 UTC = KST 09-02 13:49
- 사유: **공정성 스위트(E1 멀티시드 등) 우선 실행** 을 위한 epoch 경계 무손실 정지 (사용자 결정 2026-09-02, `docs/v8_fairness_followup_plan.md`). 재개 = `post_reboot_rearm.sh` (잔여 23ep, 공정성 큐 6단계)
- run: PureETER_SS2D_V9_radapt_multiAR_brain384 (scratch 재기동 2026-08-05 02:42 UTC, BS8, multi-AR R∈{2,3,4,5,6,8}, 34.2M params)

## 진행 요약

- 완료 epoch: **57/80** · best val_composite(현재까지): **0.9353** (val 은 R4 고정, 매 2ep)
- 잔여: 23 epochs × ~2.75 h/ep ≈ 63h — 정전 복구·재개 시점 기준으로 ETA 재계산
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
RESUME start_epoch=20 best_composite=0.9290
Epoch 21/80  train_loss=8.5679  dc_alpha=0.8946
Epoch 22/80  train_loss=8.5054  val_composite=0.9287  val_ssim_m=0.9192  val_psnr=35.98  val_nmse=0.0034  val_l1=8.1591  dc_alpha=0.8956
Epoch 23/80  train_loss=8.4627  dc_alpha=0.8952
Epoch 24/80  train_loss=8.4434  val_composite=0.9300  val_ssim_m=0.9202  val_psnr=36.08  val_nmse=0.0033  val_l1=8.0643  dc_alpha=0.8945
Epoch 25/80  train_loss=8.4651  dc_alpha=0.8940
Epoch 26/80  train_loss=8.4319  val_composite=0.9288  val_ssim_m=0.9198  val_psnr=35.94  val_nmse=0.0035  val_l1=8.2077  dc_alpha=0.8940
Epoch 27/80  train_loss=8.4031  dc_alpha=0.8921
Epoch 28/80  train_loss=8.4084  val_composite=0.9314  val_ssim_m=0.9223  val_psnr=36.12  val_nmse=0.0033  val_l1=8.0437  dc_alpha=0.8927
Epoch 29/80  train_loss=8.3372  dc_alpha=0.8913
Epoch 30/80  train_loss=8.2528  val_composite=0.9308  val_ssim_m=0.9215  val_psnr=36.09  val_nmse=0.0035  val_l1=8.0614  dc_alpha=0.8929
Epoch 31/80  train_loss=8.3386  dc_alpha=0.8921
Epoch 32/80  train_loss=8.3220  val_composite=0.9316  val_ssim_m=0.9222  val_psnr=36.16  val_nmse=0.0033  val_l1=8.0132  dc_alpha=0.8922
Epoch 33/80  train_loss=8.2610  dc_alpha=0.8926
Epoch 34/80  train_loss=8.3017  val_composite=0.9309  val_ssim_m=0.9217  val_psnr=36.10  val_nmse=0.0033  val_l1=8.0571  dc_alpha=0.8908
Epoch 35/80  train_loss=8.2381  dc_alpha=0.8910
Epoch 36/80  train_loss=8.2414  val_composite=0.9320  val_ssim_m=0.9222  val_psnr=36.21  val_nmse=0.0033  val_l1=7.9691  dc_alpha=0.8904
Epoch 37/80  train_loss=8.2068  dc_alpha=0.8899
Epoch 38/80  train_loss=8.2124  val_composite=0.9328  val_ssim_m=0.9228  val_psnr=36.27  val_nmse=0.0032  val_l1=7.9230  dc_alpha=0.8908
Epoch 39/80  train_loss=8.2377  dc_alpha=0.8888
Epoch 40/80  train_loss=8.2570  val_composite=0.9331  val_ssim_m=0.9228  val_psnr=36.32  val_nmse=0.0032  val_l1=7.8836  dc_alpha=0.8885
Epoch 41/80  train_loss=8.1187  dc_alpha=0.8879
Epoch 42/80  train_loss=8.2071  val_composite=0.9339  val_ssim_m=0.9234  val_psnr=36.37  val_nmse=0.0032  val_l1=7.8273  dc_alpha=0.8878
Epoch 43/80  train_loss=8.1684  dc_alpha=0.8875
RESUME start_epoch=43 best_composite=0.9339
Epoch 44/80  train_loss=8.1231  val_composite=0.9335  val_ssim_m=0.9230  val_psnr=36.35  val_nmse=0.0033  val_l1=7.8865  dc_alpha=0.8874
Epoch 45/80  train_loss=8.1398  dc_alpha=0.8874
Epoch 46/80  train_loss=8.1315  val_composite=0.9343  val_ssim_m=0.9234  val_psnr=36.44  val_nmse=0.0031  val_l1=7.7640  dc_alpha=0.8871
Epoch 47/80  train_loss=8.0808  dc_alpha=0.8866
Epoch 48/80  train_loss=8.0907  val_composite=0.9333  val_ssim_m=0.9224  val_psnr=36.36  val_nmse=0.0031  val_l1=7.8419  dc_alpha=0.8863
Epoch 49/80  train_loss=8.1007  dc_alpha=0.8859
Epoch 50/80  train_loss=8.0807  val_composite=0.9345  val_ssim_m=0.9234  val_psnr=36.45  val_nmse=0.0031  val_l1=7.7604  dc_alpha=0.8849
Epoch 51/80  train_loss=8.0505  dc_alpha=0.8852
Epoch 52/80  train_loss=8.0736  val_composite=0.9347  val_ssim_m=0.9239  val_psnr=36.46  val_nmse=0.0031  val_l1=7.7541  dc_alpha=0.8847
Epoch 53/80  train_loss=8.0005  dc_alpha=0.8848
Epoch 54/80  train_loss=7.9619  val_composite=0.9352  val_ssim_m=0.9243  val_psnr=36.49  val_nmse=0.0030  val_l1=7.7255  dc_alpha=0.8839
Epoch 55/80  train_loss=7.9808  dc_alpha=0.8838
Epoch 56/80  train_loss=8.0097  val_composite=0.9353  val_ssim_m=0.9241  val_psnr=36.51  val_nmse=0.0030  val_l1=7.7105  dc_alpha=0.8831
Epoch 57/80  train_loss=8.0327  dc_alpha=0.8829
```

## 프로세스 / GPU (스냅샷 시점)
```
(학습 프로세스 없음 — 8/7 21:00 KST 이후 깨끗한 정지 뒤라면 정상. 그 외 시점이면 RESUME_AFTER_OUTAGE.md 절차로 재기동)
1 %, 200 MiB
```

## ckpt 상태 (logs/PureETER_SS2D_V9_radapt_multiAR_brain384/ — 정전 시 이 파일들이 재개 지점)
```
total 2003700
drwxr-xr-x  2 root root      4096 Sep  2 04:48 .
drwxr-xr-x 13 root root      4096 Jul 30 22:52 ..
-rw-r--r--  1 root root      5441 Sep  2 04:48 log.txt
-rw-r--r--  1 root root 136777880 Sep  2 02:03 ss2d_v9_radapt_best.pt
-rw-r--r--  1 root root 136778480 Aug  6 08:08 ss2d_v9_radapt_epoch_10.pt
-rw-r--r--  1 root root 136778480 Aug  6 22:40 ss2d_v9_radapt_epoch_15.pt
-rw-r--r--  1 root root 136778480 Aug  7 13:26 ss2d_v9_radapt_epoch_20.pt
-rw-r--r--  1 root root 136778480 Aug 18 21:55 ss2d_v9_radapt_epoch_25.pt
-rw-r--r--  1 root root 136778480 Aug 19 12:50 ss2d_v9_radapt_epoch_30.pt
-rw-r--r--  1 root root 136778480 Aug 20 03:26 ss2d_v9_radapt_epoch_35.pt
-rw-r--r--  1 root root 136778480 Aug 20 18:23 ss2d_v9_radapt_epoch_40.pt
-rw-r--r--  1 root root 136778480 Aug 31 18:16 ss2d_v9_radapt_epoch_45.pt
-rw-r--r--  1 root root 136778330 Aug  5 17:13 ss2d_v9_radapt_epoch_5.pt
-rw-r--r--  1 root root 136778480 Sep  1 08:48 ss2d_v9_radapt_epoch_50.pt
-rw-r--r--  1 root root 136778480 Sep  1 23:03 ss2d_v9_radapt_epoch_55.pt
-rw-r--r--  1 root root 410366921 Sep  2 04:48 ss2d_v9_radapt_last.pt
```

## 최근 학습 로그 tail
```
Epoch  58/80:   0%|          | 17/8129 [00:57<2:48:10,  1.24s/batch, LR=3.89e-05, Loss=8.6608, PSNR=36.24dB, a=0.883][A
Epoch  58/80:   0%|          | 18/8129 [00:57<2:46:41,  1.23s/batch, LR=3.89e-05, Loss=8.6608, PSNR=36.24dB, a=0.883][A
Epoch  58/80:   0%|          | 18/8129 [00:58<2:46:41,  1.23s/batch, LR=3.89e-05, Loss=7.5357, PSNR=38.50dB, a=0.883][A
Epoch  58/80:   0%|          | 19/8129 [00:58<2:45:43,  1.23s/batch, LR=3.89e-05, Loss=7.5357, PSNR=38.50dB, a=0.883][A
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
