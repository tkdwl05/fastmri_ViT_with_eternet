# ETER-ViT v5: 분포 폭 확장 + 일반화 강화 + EarlyStopping

날짜: 2026-04-30

[eter_v4_analysis.md](eter_v4_analysis.md) §4 의 v5 plan 을 [ss2d_v5_changes.md](ss2d_v5_changes.md) 와 동일 레시피로 적용. ETER v4(0.7320, v3 대비 회귀)와 SS2D v4(0.7340, U-Net 0.8865 대비 큰 SSIM 갭) 둘 다 같은 처방을 적용해 **아키텍처 비교의 변수를 데이터/regularization 으로 한정**한다.

## 1. 동기

| 가설 | 근거 | v5 의 처리 |
|---|---|---|
| 학습 분포가 좁음 | [dataloader_h5.py](../dataloaders/dataloader_h5.py) 의 strict 320×320 필터로 360 file 스킵 | **A** — `dataloader_h5_v5` 공유 (image-domain crop/pad) |
| 디코더 dropout 미적용 | [u_choh_model_ETER_ViT.py:86](../models/hybrid_eternet/u_choh_model_ETER_ViT.py#L86) 의 `Transformer(...)` 호출이 dropout 인자를 안 넘김 → 사실상 0 | **B** — v5 wrapper 가 dropout=0.2 적용 Transformer 로 교체 |
| weight_decay 사실상 무효 | v4 까지 `LAMBDA_REGULAR_PER_PIXEL=1e-7` (≈0) | **B** — `1e-7 → 3e-5` (SS2D v5 와 동일) |
| best 이후 epoch 낭비 | v4 ep 30~40 피크 후 단조 감소, 200ep 풀 학습 | **C** — EarlyStopping (patience=5 val check) |
| 데이터 다양성 부족 | augmentation 없음 | **A** — H/V flip aug (train only) |

## 2. 새로 만든 파일 (v4 보존)

| 신규 | 변경점 |
|---|---|
| [configs/myConfig_choh_ETER_model_v5.py](../configs/myConfig_choh_ETER_model_v5.py) | `PATH_FOLDER=logs/ETER_ViT_R4_brain320_v5/`, `DROPOUT=0.2` (신규), `LAMBDA_REGULAR_PER_PIXEL 1e-7→3e-5`, `EARLYSTOP_PATIENCE=5`, `TRAIN_AUGMENT=True`, `TRAIN_AUGMENT_FLIP_P=0.5` |
| [models/hybrid_eternet/u_choh_model_ETER_ViT_v5.py](../models/hybrid_eternet/u_choh_model_ETER_ViT_v5.py) | `choh_Decoder3_ETER_v5` — v4 클래스 상속 후 `self.decoder = Transformer(..., dropout=dropout)` 로 교체. 파라미터 수/입출력 shape 모두 v4 와 동일. |
| [main_train_eter_v5.py](../main_train_eter_v5.py) | v5 config / v5 dataloader / v5 model import, train aug flag 전달, EarlyStopping break, wandb run name `ETER_v5_...` |
| [run_chain_eter_v5.sh](../run_chain_eter_v5.sh) | 기본적으로 SS2D v5 PID(21604) 종료까지 폴링 후 시작. `WAIT_PID` env var 로 override 가능 |

**v4 산출물 보존**: `logs/ETER_ViT_R4_brain320_v4/` 및 [main_train_eter.py](../main_train_eter.py)는 손대지 않는다.

## 3. SS2D v5 와의 정합성

| 항목 | SS2D v5 | ETER v5 | 차이의 의미 |
|---|---|---|---|
| dataloader | dataloader_h5_v5 (공유) | 동일 | 분포 변수 통제 |
| dropout (decoder) | 0.2 | 0.2 | 동일 |
| weight_decay | 3e-5 | 3e-5 | 동일 |
| H/V flip aug | p=0.5 | p=0.5 | 동일 |
| EarlyStop patience | 5 val check | 5 val check | 동일 |
| BATCH_SIZE | 4 (DC block 메모리) | 8 (DC 없음) | 메모리 제약 차이 |
| 모델 차이 | SS2D + DC block (1-iter) | GRU h/v (DC 없음) | **유일한 비교 변수** |

`BATCH_SIZE` 만 모델별 메모리 제약으로 다르게 두고 나머지는 모두 일치. SS2D v5 vs ETER v5 의 SSIM 차이는 (1) sequence model 종류 (Mamba vs GRU) + (2) DC block 유무 두 축의 합산 효과.

## 4. EarlyStopping 정의

[ss2d_v5_changes.md §5](ss2d_v5_changes.md) 와 동일. val composite (SSIM ratio + NMSE inv-ratio + PSNR ratio + L1 inv-ratio 평균) 기준 5회 연속 미개선 시 정지. v4 가 ep 30~40 피크였던 패턴 기준 충분.

## 5. 검증 포인트

| 확인 | 방법 |
|---|---|
| 모델 파라미터 수 v4 와 동일 | `run_eter_v5.log` 의 "모델 파라미터 수" 라인. v4 와 일치해야 함 (v5 wrapper 는 architecture 동일) |
| dropout 적용 확인 | 첫 줄 print: `'choh_Decoder3_ETER_v5  ...  (decoder dropout=0.2)'` |
| 데이터 회수 확인 | "Loaded 901 .h5 files → 14262 slices" (train) / "460 → 7270" (val) — SS2D v5 와 동일한 풀 |
| flip aug 가 발산 유발하지 않는지 | epoch 1~5 train SSIM 추세 |
| v4 회귀 해소 | v4 best 0.7320 ↑ 회복 + v3 0.7475 추월하는지 |
| SS2D v5 와의 갭 | 같은 데이터/regularization 에서 두 모델의 best val SSIM 차이 — 아키텍처 효과의 정량 측정 |

## 6. 실행

```bash
# SS2D v5 (PID 21604) 끝나면 자동 시작
./run_chain_eter_v5.sh &

# 다른 PID 뒤에 붙이려면
WAIT_PID=<other_pid> ./run_chain_eter_v5.sh &

# 즉시 시작 (단독)
WAIT_PID= ./run_chain_eter_v5.sh &
```

로그:
- chain start/end: `run_chain_eter_v5.log`
- 학습 stdout/stderr: `run_eter_v5.log`
- 체크포인트 / 에폭 로그: `logs/ETER_ViT_R4_brain320_v5/`

## 7. 결정하지 않은 것

- **WarmRestarts 복귀** — [eter_v4_analysis.md §4](eter_v4_analysis.md) 의 선택 항목. 추가 변수가 늘어 v5 효과 분리가 어려워져 v5 에서는 제외. v6 후보.
- **GRU dropout** — `nn.GRU(... dropout=...)` 는 num_layers=1 일 때 무효. layer 수를 늘리는 건 capacity 증설이라 별도 실험으로 분리.

## 8. 참고 문서

- [eter_v4_analysis.md](eter_v4_analysis.md) — v4 회귀 분석과 v5 plan 출처
- [ss2d_v5_changes.md](ss2d_v5_changes.md) — 같은 레시피의 SS2D 적용판
- [architecture_ETER_vs_SS2D.md](architecture_ETER_vs_SS2D.md) — 두 아키텍처의 구조 비교
