# SS2D-ViT v6: Metric 정합화 + EarlyStop 기준 수정 + Resume

날짜: 2026-05-04

[ss2d_v5_changes.md](ss2d_v5_changes.md) 의 실제 학습 결과에서 드러난 두 가지 결함을 한꺼번에 고치는 버전. v5 의 데이터 / regularization / aug 설정은 그대로 두고, **val 평가 metric 과 EarlyStop 판정 로직만 수정**한다. v5 산출물(`logs/SS2D_ViT_R4_brain320_v5/`)은 보존.

## 1. 동기: v5 에서 발견된 두 결함

| # | 결함 | 증거 | v6 의 처리 |
|---|---|---|---|
| 1 | **EarlyStop 기준이 잘못됨** | v5 는 composite (= SSIM ratio + NMSE inv-ratio + PSNR ratio + L1 inv-ratio 평균) 기준. SSIM 은 ep1→ep12 단조 증가 추세였는데 composite 는 ep ~4 에 피크 후 정체 → patience=5 발동해 ep 12 에서 종료. composite-best 시점인 ep 4 ckpt 만 best 로 저장. | **EarlyStop / best 갱신 기준을 val SSIM 단일로 변경** |
| 2 | **val SSIM 측정이 부정확** | v5 는 학습용 `u_choh_SSIM` 을 그대로 val 에도 씀. `val_range=None` 시 L=1 로 고정 → fastMRI raw RSS (dynamic range 가 슬라이스별 다름) 를 일관되게 평가 못함. U-Net 평가용 [eval_unet_pretrained.py](../eval_unet_pretrained.py) 는 skimage `structural_similarity` + `data_range = target.max() − target.min()` 사용. | **val SSIM 을 skimage 표준으로 교체** (loss 는 v5 그대로 유지) |

→ v5 best val SSIM **0.7227 (custom)** = 사실상 ep 4 의 값. skimage 표준으로 다시 재면 **0.8584** 로 환산. 단순한 평가 metric 일관성 문제가 EarlyStop 까지 오작동시킨 구조.

## 2. 새로 만든 파일 (v5 보존)

`_v6` 접미사로 신규. 기존 v5 파일은 건드리지 않는다.

| 신규 | 원본 | 변경점 |
|---|---|---|
| [configs/myConfig_choh_SS2D_model_v6.py](../configs/myConfig_choh_SS2D_model_v6.py) | `..._v5.py` | `PATH_FOLDER=logs/SS2D_ViT_R4_brain320_v6/`, 신규 `EARLYSTOP_PATIENCE=10`, `VAL_EVERY_N_EPOCHS=5`, `RESUME_CKPT='./logs/SS2D_ViT_R4_brain320_v5/ss2d_vit_epoch_10.pt'` |
| [main_train_ss2d_v6.py](../main_train_ss2d_v6.py) | `main_train_ss2d_v5.py` | (1) `from skimage.metrics import structural_similarity as compare_ssim` (2) `skimage_ssim_batch()` 헬퍼 — `data_range = t[i].max() − t[i].min()` 슬라이스별 산출 (3) `run_val()` 에서 SSIM 만 skimage 사용, PSNR/NMSE/L1 은 기존대로 (4) `RESUME_CKPT` 존재 시 weight load (5) train_best 갱신 시 val 트리거 폐지, `(epoch+1) % VAL_EVERY_N_EPOCHS == 0` 시점에만 val (6) best/EarlyStop 기준을 `val_ssim` 단일로 (7) wandb run name `SS2D_v6_resume_...`, `'val_metric': 'skimage_ssim'` 기록 |
| [runs/chain/run_chain_v6.sh](../runs/chain/run_chain_v6.sh) | `runs/chain/run_chain_ss2d_v5.sh` | SS2D v6 → ETER v6 순차 실행, 각 단계 exit code 합산 |

**Dataloader / 모델 파일은 신규 생성하지 않는다.** v6 는 v5 의 [dataloader_h5_v5.py](../dataloaders/dataloader_h5_v5.py) 와 [u_choh_model_SS2D_ViT_v4.py](../models/mamba_eternet/u_choh_model_SS2D_ViT_v4.py) 를 그대로 import. 변수를 데이터/모델이 아닌 평가 metric 과 학습 제어 로직에 한정.

## 3. skimage SSIM 의 차이 (실측)

| ckpt | custom u_choh_SSIM | skimage SSIM | 비고 |
|---|---|---|---|
| v5 best (ep 4) | 0.7227 | 0.8584 | data_range 미정 → custom 이 과소평가 |
| U-Net pretrained | (미측정) | 0.8865 | [eval_unet_pretrained_summary.txt](../results/eval_unet_pretrained_summary.txt) |

같은 모델인데 평가 방식만 다른 두 수치 가 0.13 이상 차이. U-Net 과 비교하려면 같은 metric 이어야 의미가 있음.

## 4. EarlyStopping 정의 (v6)

- 트리거: **val SSIM (skimage)** 가 새 best 를 갱신하지 못한 **연속 val check 회수** ≥ `EARLYSTOP_PATIENCE=10`
- val 빈도: `(epoch+1) % VAL_EVERY_N_EPOCHS == 0` (즉 매 5 epoch). train_best 트리거는 폐지
- patience=10 × 5 epoch = **~50 epoch 무개선 시 정지** (v5 의 5 → 10 으로 보수적)

## 5. Resume 의 근거

v5 의 ckpt 후보:
- `ss2d_vit_best.pt` — composite-best 인 ep ~4 시점 (val SSIM 낮음)
- `ss2d_vit_epoch_10.pt` — v5 종료 직전 ckpt, val SSIM 0.7257 로 가장 높음

v6 는 `epoch_10.pt` 부터 resume. v5 의 학습 신호가 SSIM 단조 증가였으므로, 가장 발전된 weight 에서 출발하는 게 합리적. random init 으로 되돌리면 v5 50 epoch 분량을 버리는 셈.

## 6. 메모리 환경변수

[main_train_ss2d_v6.py:17](../main_train_ss2d_v6.py#L17):
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```
v4 의 SS2D forward 에 gradient checkpointing 이 들어가있어 BS=4 가 8 GB 에 빠듯하게 들어감. fragmentation 최소화 목적.

## 7. 검증 포인트 (학습 시작 후)

| 확인 | 방법 |
|---|---|
| skimage SSIM 정상 동작 | epoch 1 val 출력의 SSIM 이 v5 best (0.8584) 근방인지 |
| resume 효과 | baseline 측정값 (`Resumed weights from ...` 직후 출력) 이 v5 ep 10 의 0.7257(custom) ≒ 0.86~ 0.87(skimage) 인지 |
| EarlyStop 오작동 재발 여부 | wandb `val/no_improve_count` 가 천천히 누적되고 SSIM 이 실제로 정체할 때만 patience 한계 도달 |
| U-Net 추격 가능성 | best val SSIM 이 0.8865 (U-Net) 에 얼마나 근접하는지가 1차 success metric |

## 8. 결정하지 않은 것 (v6 에서 제외)

- **Loss 함수 변경** — `1 - u_choh_SSIM` 그대로 사용. val metric 만 바꾸고 loss 까지 건드리면 변수 두 개가 동시에 변해 효과 분리 불가
- **학습형 sensitivity map (E2E-VarNet 스타일)** — v5 plan §7 에서 보류한 항목, v7 후보
- **DC iteration 수 증가** — 동일 이유로 v7 후보

## 9. 실행

```bash
# 즉시 시작
./runs/chain/run_chain_v6.sh &
```

로그:
- chain start/end: `runs/chain/run_chain_v6.log`
- SS2D v6 stdout/stderr: `runs/ss2d/run_ss2d_v6.log`
- ETER v6 stdout/stderr: `runs/eter/run_eter_v6.log` (SS2D v6 종료 후 자동 시작)
- 체크포인트 / 에폭 로그: `logs/SS2D_ViT_R4_brain320_v6/`

## 10. 결과 (2026-05-08 SS2D v6 종료)

| epoch | val SSIM (skimage) | PSNR | NMSE | L1 |
|---|---|---|---|---|
| 60 | 0.8830 | 35.52 dB | 0.0091 | 7.8119 |
| 100 | 0.8868 | 35.72 dB | 0.0086 | 7.6138 |
| 145 | 0.8901 | 35.91 dB | 0.0082 | 7.4363 |
| 155 | 0.8903 | 35.89 dB | 0.0082 | 7.4354 |
| 200 (종료) | (best ≈ 0.8903) | — | — | — |

EarlyStop 발동 없이 200ep 풀 학습 완료. v5 best 0.8584 (skimage) → v6 best **0.8903 (+0.032)**. U-Net 0.8865 와 동등 또는 약간 상회.

## 11. 참고 문서

- [ss2d_v5_changes.md](ss2d_v5_changes.md) — v5 의 데이터/regularization/aug 설정
- [eter_v6_changes.md](eter_v6_changes.md) — 같은 레시피의 ETER 적용판
- [presentation_overview.md](presentation_overview.md) — v1~v6 전체 변천사
