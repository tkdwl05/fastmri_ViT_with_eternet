# SS2D-ViT v7_titan 변경 / 재학습 정상화 (2026-05-31)

## 0. 배경

목표는 **SS2D-ViT vs ETER-ViT 의 공정한 head-to-head 비교** (CLAUDE.md: "유일한 변수 = sequence model 종류").
v7_titan 본 학습(384×384, ViT-Base, [eval_metric_redesign.md](eval_metric_redesign.md) 적용, 단일 TITAN RTX 24GB)에서
**ETER 만 50ep 완주하고 SS2D 는 중단**되어 비교가 성립하지 않았다. 이 문서는 SS2D 재학습을 정상화한 내역과
현재 가동 중인 설정을 기록한다. (승인 plan: `calm-tickling-newell` → 실행 plan `dapper-tickling-graham`.)

---

## 1. SS2D v7_titan 모델 사양 (vs v6 / v7)

| 항목 | v6 (8GB, 320) | v7 (320) | **v7_titan (384)** |
|---|---|---|---|
| IMAGE_SIZE | 320×320 | 320×320 | **384×384** |
| PATCH_SIZE | 16×16 | 16×16 | 16×16 (num_patches 576) |
| ViT 인코더 | ViT-Small (dim384/L6/H6/MLP1536) | ViT-Small | **ViT-Base (dim768/L12/H12/MLP3072)** |
| SS2D | d_state16/d_inner64/out20 | 동일 | **d_state32 / d_inner128 / out32** |
| Decoder | dim512, depth6 | 동일 | dim512, depth6 |
| DC block | 1-iter soft (k_scale100, α=1.0) | 동일 | 동일 |
| 파라미터 수 | — | — | **114.8M** |

인코더/디코더 클래스는 무변경 import: `choh_ViT` (`u_choh_model_ETER_ViT`), `choh_Decoder_SS2D_ViT` (`u_choh_model_SS2D_ViT_v4`).

---

## 2. 현재 가동 학습 조건 (2026-05-31 재기동)

| 항목 | 값 | 비고 |
|---|---|---|
| 시작 방식 | **scratch** (`RESUME_CKPT=None`) | ETER(scratch) 와 공정 비교 — §3 참고 |
| GPU | **단일 GPU** (GPU0, ~19.6GB/88%) | ETER 와 동일. DDP 아님 (effective-batch confounder 회피) |
| BATCH_SIZE | **6** | §3 smoke 근거. BS8 은 19.72GB(=84%)로 실전 OOM |
| NUM_EPOCHS | **50** | T_max=541900 step (10838/ep × 50). ETER 와 동일 LR곡선 |
| Scheduler | CosineAnnealingLR, eta_min=1e-6 | T_max=NUM_EPOCHS (cosine 단조 감소) |
| LR / weight_decay / dropout | 2e-4 / 3e-5 / 0.2 | v5~ 레시피 |
| augment | H/V flip p=0.5 | v5~ 레시피 |
| **VAL_EVERY_N_EPOCHS** | **2** | 2026-05-31 5→2. epoch10 중간점검용 (ep 2/4/6/8/10 = 5점) |
| **EARLYSTOP_PATIENCE** | **50** | 2026-05-31 10→50. VAL_EVERY=2 면 50ep 중 val-check 25회뿐 → **early-stop 50ep 내 사실상 비활성** (ETER 와 동일, ETER 도 완주). epoch10 점검 후 계속/중단은 **수동 결정** |
| best / EarlyStop 기준 | **composite** (0.5·SSIM + 0.3·PSNR/40 + 0.2·(1−NMSE)) | mask 안에서 측정 |
| brain mask | Otsu × 0.4 + largest CC (no erode/fill) | `dataloader_h5_v5.py:243` 하드코딩 ([eval_metric_redesign.md §10-C](eval_metric_redesign.md)) |
| val 측정 | skimage SSIM (mask 안 `max−min` data_range) | |

데이터: train 65028 slice / val 7334 slice (NVMe).

---

## 3. 검출된 결함 + 수정 (E1–E8)

진단: ETER 는 완주(best **composite 0.9127 / ssim_m 0.9084 / PSNR 34.59**)했으나 SS2D 는 DDP 로 epoch 10 까지만
(이후 단일 BS8 OOM, BS6 재시도는 당시 GPU 다운으로 크래시). 주요 수정:

- **E3 — scratch + true-resume**: 과거 "epoch_10 resume" 은 사실 weights-only warm-start 였고 DDP 10ep head-start +
  LR 불연속(2e-4 리셋) confounder. → DDP 산출물을 `logs/SS2D_ViT_R4_brain384_v7_titan/_ddp_archive/` 로 격리,
  `RESUME_CKPT=None` 으로 scratch 재시작. 중단 내성은 **true-resume**(§4)이 담당.
- **E4 — BS6 smoke 신뢰성**: `sanity_smoke_test_v7_titan.py` 의 loss 를 실제 학습과 동일한 **masked L1 + (1−SSIM)** +
  unscale + clip_grad 로 교체(이전엔 단순 l1). 측정: **BS6 peak_alloc 14.91GB / reserved 18.66GB (8-step 안정) = 안전**.
  BS8 은 19.72GB(=84%)로 cudnn workspace + display + 단편화 여유 부족 → 실전 OOM 원인 확인.
- **E2 — 오케스트레이션**: 고아 marker(`v7_titan/runs/ss2d/.bs6_started`) 제거. 옛 watcher/chain 미경유, 직접 기동(§5).
- **E6/E7 위생**: config 의 미사용 `BRAIN_MASK_THRESHOLD/ERODE_ITER` 제거(실제 mask 는 dataloader 하드코딩),
  earlystop 라벨 `val_ssim`→`val_composite`(실제 기준은 composite) 정정, [eval_metric_redesign.md](eval_metric_redesign.md) Table2/§D1 를 §10-C 로 정정.
- **미변경(의도적)**: E5(50ep 유지), E8(`u_choh_SSIM.py` 가 `val_range` 미전달 → SSIM-loss auto-detect) — 완료 ETER 와
  loss 일치 보존 위해 그대로. E8 은 향후 클린 재학습 항목으로만 기록.

---

## 4. true checkpoint resume (신규, 핵심)

`main_train_ss2d_v7_titan.py` (및 대칭성 위해 `main_train_eter_v7_titan.py`):

- **저장**: 매 epoch rolling `ss2d_vit_last.pt` 에 full-state 원자적 저장(`save_checkpoint_atomic`: tmp→`os.replace`).
  내용 = `{epoch, model, optimizer, scheduler, scaler, best_val_composite, best_val, no_improve_val_count, global_step, RNG(torch/cuda/numpy/python)}`.
- **재개 우선순위**: ① `*_last.pt`(full-state) 존재 → 전 상태 복원 + `start_epoch=ckpt['epoch']`,
  학습 루프 `range(start_epoch, NUM_EPOCHS)`, scheduler/scaler state 복원으로 **LR 연속** → ② `RESUME_CKPT`(레거시 weights-only warm) → ③ scratch.
- **검증**: 더미 모델 unit-test 로 "중단 없이 20 step" vs "10 step 저장→fresh 복원→10 step" 의 LR 궤적이 **byte-identical** 임을 확인 (PASS).
- 기존 weights-only `epoch_N.pt` / `best.pt` 저장은 eval/visualize 호환 위해 유지.

효과: GPU 가 epoch 30 에서 죽어도 epoch 30 의 정확한 LR 부터 이어감 → 멀티데이 학습 중단 내성.

---

## 5. auto-restart supervisor

`v7_titan/runs/run_ss2d_v7_titan_autoresume.sh` (setsid+nohup detached, 세션 종료에도 생존):

- `main_train_ss2d_v7_titan.py` 직접 실행 (watcher/chain 미경유).
- transient 비정상 종료 시 60s 후 true-resume 로 자동 재기동(`*_last.pt` 부터), `max_retry=50`.
- "학습 완료" stdout sentinel 감지 시 종료.
- 로그: stdout `v7_titan/runs/ss2d/run_ss2d_v7_titan_scratch.log`, supervisor `runner_supervisor.log`, 메트릭 `logs/SS2D_ViT_R4_brain384_v7_titan/log.txt`.

---

## 6. epoch 10 중간점검 + 비교 baseline

- **중간점검(~2.3일 후, epoch 10)**: `log.txt` 의 ep 2/4/6/8/10 val 궤적을 ETER 와 대조 → 계속/중단 수동 결정.
  좋으면 그대로 ep50 까지 (true-resume 라 재시작 불필요).
- **비교 baseline (ETER v7_titan, 완주)**: **composite 0.9127 / ssim_m 0.9084 / PSNR 34.59** (`logs/ETER_ViT_R4_brain384_v7_titan/eter_vit_best.pt`). ETER 중간 ep10 무렵 ssim_m ≈ 0.8966.
- **공정성**: 양측 scratch · 단일GPU · 50ep · 384 · masked L1+SSIM loss · composite metric **동일**. 차이 = sequence model(Mamba+DC block vs GRU) + BATCH_SIZE(6 vs 8).
- (참고, unmasked·직접비교 주의: U-Net 0.8865, SS2D v6 0.8903.)
- **ETA**: ~1.9s/batch × 10838 → epoch당 ~5.6h, 50ep ≈ **11–12일** (SS2D mamba 가 ETER 보다 느림 + BS6 step 수 많음). true-resume 가 중단 내성 핵심.

---

## 7. 변경 파일 인벤토리

- `v7_titan/configs/myConfig_choh_SS2D_model_v7_titan.py` — `RESUME_CKPT=None`, `VAL_EVERY_N_EPOCHS=2`, `EARLYSTOP_PATIENCE=50`, stale mask params 제거
- `v7_titan/main_train_ss2d_v7_titan.py` — true-resume 저장/복원, earlystop 라벨 정정
- `v7_titan/main_train_eter_v7_titan.py` — true-resume 미러(ETER 는 이미 완주, 향후용)
- `v7_titan/sanity_smoke_test_v7_titan.py` — 풀 train-step(실 loss) smoke 로 교체
- `v7_titan/runs/run_ss2d_v7_titan_autoresume.sh` — auto-restart supervisor (신규)
- `logs/SS2D_ViT_R4_brain384_v7_titan/_ddp_archive/` — 폐기된 DDP 산출물 격리
- `docs/eval_metric_redesign.md` — mask 공식 §10-C 정정
