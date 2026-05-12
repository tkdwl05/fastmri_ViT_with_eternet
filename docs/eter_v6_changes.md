# ETER-ViT v6: Metric 정합화 + EarlyStop 기준 수정 + Resume

날짜: 2026-05-04 (실행 2026-05-08 / 재실행 2026-05-11)

[ss2d_v6_changes.md](ss2d_v6_changes.md) 와 동일 처방을 ETER 에 적용한 버전. v5 와 동일 데이터 / regularization / aug 를 쓰되 **val 평가 metric 과 EarlyStop 판정 로직만** 수정한다. 아키텍처 비교 (SS2D vs ETER) 의 변수를 모델로 한정.

## 1. 동기: v5 ETER 에서 발견된 두 결함

| # | 결함 | 증거 | v6 의 처리 |
|---|---|---|---|
| 1 | **EarlyStop 기준이 잘못됨** | v5 의 SSIM 은 ep1→ep9 단조 증가 (0.6865 → 0.7421) 했지만 composite 는 ep 4 (SSIM 0.7217) 에 피크 후 정체 → patience=5 발동, **ep 9 에 종료**. ckpt 도 composite-best 인 ep 4 만 저장 ([eter_v5 의 ckpt 디렉토리](../logs/ETER_ViT_R4_brain320_v5/) 에 `epoch_5.pt` 만 존재) | **EarlyStop / best 기준을 val SSIM 단일로 변경** |
| 2 | **val SSIM 측정이 부정확** | v5 는 `u_choh_SSIM` 그대로 val. `val_range=None` → L=1 고정 → fastMRI raw RSS 의 슬라이스별 dynamic range 변동을 제대로 반영 못함 | **val SSIM 을 skimage `structural_similarity` (data_range = target.max() − target.min()) 로 교체** |

→ v5 best val SSIM 0.7421 (custom) ≒ 0.8469 (skimage). SSIM 자체는 계속 오를 추세였는데 composite 가 제동을 걸어 학습이 50ep 도 못 가서 끝난 상황.

## 2. 새로 만든 파일 (v5 보존)

`_v6` 접미사로 신규. 기존 v5 파일은 건드리지 않는다.

| 신규 | 원본 | 변경점 |
|---|---|---|
| [configs/myConfig_choh_ETER_model_v6.py](../configs/myConfig_choh_ETER_model_v6.py) | `..._v5.py` | `PATH_FOLDER=logs/ETER_ViT_R4_brain320_v6/`, 신규 `EARLYSTOP_PATIENCE=10`, `VAL_EVERY_N_EPOCHS=5`, `RESUME_CKPT='./logs/ETER_ViT_R4_brain320_v5/eter_vit_epoch_5.pt'`. **(2026-05-11 추가)** `BATCH_SIZE 8 → 4` — 재부팅 후에도 BS=8 OOM 재발, 첫 forward 의 conv2d 가 100 MiB 부족으로 사망. v5 와 동일 BS 였으나 v6 의 baseline 측정 단계가 cudnn workspace 를 더 점유하는 것으로 보임 |
| [main_train_eter_v6.py](../main_train_eter_v6.py) | `main_train_eter_v5.py` | (1) `from skimage.metrics import structural_similarity as compare_ssim` (2) `skimage_ssim_batch()` 헬퍼 — `data_range = t[i].max() − t[i].min()` 슬라이스별 산출 (3) `run_val()` 에서 SSIM 만 skimage 사용, PSNR/NMSE/L1 은 기존대로 (4) `RESUME_CKPT` 존재 시 weight load + baseline 측정 후 best.pt 로 저장 (5) train_best 트리거 폐지, `(epoch+1) % VAL_EVERY_N_EPOCHS == 0` 시점에만 val (6) best/EarlyStop 기준을 `val_ssim` 단일로 (7) wandb run name `ETER_v6_resume_BS{BS}_LR{LR}_EP{EP}`, `'val_metric': 'skimage_ssim'` 기록 |
| [run_chain_v6.sh](../run_chain_v6.sh) | (공유) | SS2D v6 → ETER v6 순차 실행 |

**Dataloader / 모델 파일은 신규 생성하지 않는다.** v6 는 v5 의 [dataloader_h5_v5.py](../dataloaders/dataloader_h5_v5.py) 와 [u_choh_model_ETER_ViT_v5.py](../models/hybrid_eternet/u_choh_model_ETER_ViT_v5.py) (dropout=0.2 thin wrapper) 를 그대로 import.

## 3. SS2D v6 와의 정합성

| 항목 | SS2D v6 | ETER v6 | 차이의 의미 |
|---|---|---|---|
| Val metric | skimage SSIM | skimage SSIM | 동일 |
| Best/EarlyStop 기준 | val_ssim 단일 | val_ssim 단일 | 동일 |
| Patience | 10 val check | 10 val check | 동일 |
| Val 빈도 | 매 5 epoch | 매 5 epoch | 동일 |
| Resume 출처 | v5 epoch_10.pt | v5 epoch_5.pt | 각 v5 의 최고 발전 weight |
| Dataloader | dataloader_h5_v5 | dataloader_h5_v5 | 공유 |
| Dropout / WD / Aug | v5 와 동일 (0.2 / 3e-5 / flip) | v5 와 동일 | 동일 |
| BATCH_SIZE | 4 (DC block 메모리) | 4 (v6 OOM 후 강하) | v5 8 → v6 4 |
| 모델 차이 | SS2D + DC block (1-iter) | GRU h/v (DC 없음) | **유일한 비교 변수** |

평가 metric, 학습 제어, 데이터 모두 동일. SS2D v6 vs ETER v6 의 best val SSIM 차이는 (1) sequence model 종류 (Mamba vs GRU) + (2) DC block 유무 의 합산 효과.

## 4. EarlyStopping 정의 (v6)

- 트리거: **val SSIM (skimage)** 가 새 best 를 갱신하지 못한 **연속 val check 회수** ≥ `EARLYSTOP_PATIENCE=10`
- val 빈도: `(epoch+1) % 5 == 0` (즉 매 5 epoch)
- patience=10 × 5 epoch = **~50 epoch 무개선 시 정지**

v5 에서 ep 9 에 잘못 멈춘 패턴을 방지하기 위해 patience 를 두 배로 잡고, 기준을 단일 SSIM 으로 단순화. composite 의 false-stop 위험 제거.

## 5. Resume 의 근거

v5 의 ckpt 후보:
- `eter_vit_best.pt` — composite-best 인 ep 4 시점, val SSIM 0.7217 (custom)
- `eter_vit_epoch_5.pt` — ep 5 시점, val SSIM 0.7249 (custom) ≒ 가장 높은 단조 증가 지점

v6 는 `epoch_5.pt` 부터 resume. v5 가 ep 9 까지 단조 증가했음을 감안하면 ep 5 ckpt 가 사용 가능한 weight 중 가장 발전된 시점.

## 6. BATCH_SIZE 강하 (2026-05-11)

v6 첫 실행 (2026-05-08 14:16) 에서 OOM 으로 즉시 사망 (14분 만에 exit 1):
```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 100.00 MiB.
GPU 0 has a total capacity of 7.52 GiB of which 102.88 MiB is free.
```

위치: 첫 forward 의 conv2d layer ([u_choh_model_ETER_ViT.py:22](../models/hybrid_eternet/u_choh_model_ETER_ViT.py#L22)). v5 는 BS=8 로 통과했었음에도 v6 에서 재발한 이유:
- v6 의 **resume + baseline 측정 단계** 가 cudnn workspace 와 activation 캐시를 추가로 점유
- 7.10 GiB 가 이미 사용 중인 상태에서 첫 train step 이 100 MiB 추가 할당을 못함

재부팅 후 재시도 (2026-05-11) 에서도 동일 OOM → **`BATCH_SIZE = 8 → 4` 강하**. 학습 속도는 절반으로 떨어지지만 안정성 확보. 실측 ~21.8분/epoch, EarlyStop 안 걸리면 200ep ≈ 72시간.

## 7. 검증 포인트 (학습 시작 후)

| 확인 | 방법 |
|---|---|
| skimage SSIM 정상 동작 | epoch 5 val 출력의 SSIM 이 baseline 0.86~0.87 근방인지 |
| resume 효과 | `Resumed weights from ...` 직후 출력의 baseline val SSIM 이 v5 ep 5 의 0.7249(custom) ≒ 0.84~ 0.85(skimage) 인지 |
| EarlyStop 오작동 재발 여부 | wandb `val/no_improve_count` 가 천천히 누적되는지 / SSIM 실제 정체 시점에만 patience 도달 |
| SS2D v6 와의 갭 | 같은 데이터/평가에서 두 모델의 best val SSIM 차이 — 아키텍처 효과의 정량 측정 |
| OOM 재발 여부 | 첫 epoch 의 batch 1~10 trace |

## 8. 결정하지 않은 것 (v6 에서 제외)

- **Loss 함수 변경** — `1 - u_choh_SSIM` 그대로. val metric 만 바꾸고 loss 까지 건드리면 변수 분리 불가
- **DC block 추가** — ETER 는 처음부터 DC 없음 설계. SS2D vs ETER 비교에서 DC 유무가 변수
- **GRU layer 증설** — capacity 변수 추가. 별도 실험으로 분리
- **WarmRestarts 복귀** — v4 plan 에서 보류한 항목, v7 후보

## 9. 실행

```bash
# SS2D v6 종료 후 자동 시작 (chain)
./run_chain_v6.sh &

# 또는 SS2D v6 가 이미 종료된 상태에서 ETER v6 만 단독 실행
nohup python main_train_eter_v6.py > run_eter_v6.log 2>&1 &
```

로그:
- chain start/end: `run_chain_v6.log`
- 학습 stdout/stderr: `run_eter_v6.log`
- 체크포인트 / 에폭 로그: `logs/ETER_ViT_R4_brain320_v6/`

## 10. 진행 상황

- 2026-05-08 14:16 — 첫 실행 즉시 OOM (BS=8)
- 2026-05-11 12:03 — 재부팅 후 재실행, BS=8 동일 OOM
- 2026-05-11 14:00 — `BATCH_SIZE=4` 로 수정 후 재시작, baseline 측정 통과
- 2026-05-11 23:00 — epoch 4/200 진행, val SSIM 0.8620 (skimage), 21.8분/epoch 평속

## 11. 참고 문서

- [eter_v5_changes.md](eter_v5_changes.md) — v5 의 dropout/wd/aug/dataloader 설정
- [ss2d_v6_changes.md](ss2d_v6_changes.md) — 동일 레시피의 SS2D 적용판
- [architecture_ETER_vs_SS2D.md](architecture_ETER_vs_SS2D.md) — 두 아키텍처 비교
- [presentation_overview.md](presentation_overview.md) — v1~v6 전체 변천사
