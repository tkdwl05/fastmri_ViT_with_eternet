# SS2D-ViT v5: 분포 폭 확장 + 일반화 강화 + EarlyStopping

날짜: 2026-04-30

[ss2d_v4_changes.md](ss2d_v4_changes.md) 결과(`val SSIM 0.7340` vs 동일 val 셋의 fastMRI pretrained U-Net `0.8865`)에서 드러난 문제를 해결하기 위한 버전. v4 모델/스케줄러는 그대로 두고 **데이터 파이프라인과 학습 루프만 손본다.** v4 산출물(`logs/SS2D_ViT_R4_brain320_v4/`)은 보존.

## 1. 동기

[ss2d_v4_changes.md](ss2d_v4_changes.md) §1과 [eter_v4_analysis.md](eter_v4_analysis.md)에서 진단된 두 가지 회귀 원인을 함께 공격한다.

| 가설 | 근거 | v5 의 처리 |
|---|---|---|
| 학습 분포가 좁음 | [dataloader_h5_v4.py:140-144](../dataloaders/dataloader_h5_v4.py#L140-L144)에서 `rss shape != (320,320)` 인 파일 360 개 스킵 → 실제 학습 슬라이스가 fastMRI brain 코퍼스의 ~60% 수준 | **A** — image-domain center-crop / zero-pad 로 모든 파일 흡수 |
| 일반화 부족 / 과적합 | [eter_v4_analysis.md §3](eter_v4_analysis.md) "capacity ceiling 조기 도달", v4 train-val gap 0.055 | **B** — Transformer dropout 0.1→0.2, weight_decay 1e-5→3e-5, H/V flip aug |
| best 이후 epoch 낭비 | v4 는 val 피크 후 단조 감소했지만 200ep 풀 학습 | **C** — EarlyStopping (val composite 기준, patience=5 val check) |

## 2. 새로 만든 파일 (v4 보존)

`_v5` 접미사로 신규. 기존 v4 파일은 건드리지 않는다.

| 신규 | 원본 | 변경점 |
|---|---|---|
| [configs/myConfig_choh_SS2D_model_v5.py](../configs/myConfig_choh_SS2D_model_v5.py) | `..._v4.py` | `PATH_FOLDER=logs/SS2D_ViT_R4_brain320_v5/`, `DROPOUT 0.1→0.2`, `LAMBDA_REGULAR_PER_PIXEL 1e-5→3e-5`, 신규 상수 `EARLYSTOP_PATIENCE=5`, `TRAIN_AUGMENT=True`, `TRAIN_AUGMENT_FLIP_P=0.5` |
| [dataloaders/dataloader_h5_v5.py](../dataloaders/dataloader_h5_v5.py) | `dataloader_h5_v4.py` | 사이즈 필터 완화 (rss 존재만 확인) + `crop_or_pad_to` 헬퍼, `__init__` 에 `augment / augment_flip_p` 인자, image-domain H/V flip aug (img_crop, gt_rss 동기 적용 후 FFT 재계산) |
| [main_train_ss2d_v5.py](../main_train_ss2d_v5.py) | `main_train_ss2d_v4.py` | v5 config / v5 dataloader import (모델은 **v4 그대로**), train aug flag 전달, `no_improve_val_count` 카운터 + patience 도달 시 break, wandb run name `SS2D_v5_...` |
| [run_chain_ss2d_v5.sh](../run_chain_ss2d_v5.sh) | `run_chain_ss2d_v4.sh` | `WAIT_PID` 환경변수가 있으면 폴링, 없으면 즉시 시작 |

**모델 파일은 신규 생성하지 않는다.** v4 [u_choh_model_SS2D_ViT_v4.py](../models/mamba_eternet/u_choh_model_SS2D_ViT_v4.py)의 `dropout` 파라미터가 이미 config 주입형이라 코드 변경 없이 0.2 가 들어간다. 모델 비교의 변수를 데이터/regularization 으로 한정하기 위함.

## 3. 데이터 회수 효과 (smoke test)

| 셋 | v4 (strict 320×320) | v5 (relaxed crop/pad) | 회수 |
|---|---|---|---|
| train | 541 file / 8,548 slice | 901 file / 14,262 slice | **+67%** |
| val | 285 file / 4,492 slice | 460 file / 7,270 slice | **+62%** |

v5 val 셋이 [eval_unet_pretrained.py](../eval_unet_pretrained.py) 기준 7,270 과 일치 → U-Net leaderboard 와 같은 슬라이스 모집단에서 직접 비교 가능.

## 4. Augmentation 정합성 노트

flip 은 **iFFT 후 image 도메인에서 적용** 한 뒤, 이어지는 `fft2c → mask → ifft2c` 파이프라인을 통과한다.

- mask 는 매 샘플마다 `build_r4_mask` 가 새로 만들고 phase-encoding(W) 축에 1D 패턴이라 image flip 과 독립적
- `gt_rss` 는 `reconstruction_rss` 에서 별도로 읽지만 같은 H/V flip 을 동기 적용해 학습 정합성 유지
- ACS-only sens 추정도 flipped image 의 k-space 에서 다시 계산되므로 일관

## 5. EarlyStopping 정의

- 트리거: val composite (= SSIM ratio + NMSE inv-ratio + PSNR ratio + L1 inv-ratio 평균) 가 새 best 를 갱신하지 못한 **연속 val check 회수** ≥ `EARLYSTOP_PATIENCE`
- val 빈도: train SSIM best 갱신 시 + 매 `VAL_EVERY_N_EPOCHS=10` epoch 마다
- patience=5 → 약 50 epoch 무개선 시 정지. v4 가 ep ~30~40 에 피크였던 패턴 기준 충분한 마진.

## 6. 검증 포인트 (학습 시작 후)

| 확인 | 방법 |
|---|---|
| 첫 batch OOM 여부 | `run_ss2d_v5.log` epoch1 batch1 trace. v4 와 동일 capacity / BS=4 라 동일 거동 예상 |
| flip aug 가 손실 발산을 일으키지 않는지 | epoch 1~5 의 train SSIM 추세 — v4 첫 epoch 대비 ±5% 내면 정상 |
| val composite 단조 개선 | wandb `val/no_improve_count` 가 0 ↔ 작은 값 사이를 오가야 함. 빨리 patience 한계에 도달하면 일반화가 v4 대비 개선되지 않은 것 |
| v4 대비 SSIM 갭 축소 | best ckpt 시점의 val SSIM 이 0.7340(v4) → 0.78+ 로 가는지가 1차 success metric. UNet 0.8865 와의 갭이 30% 이상 축소되는지가 2차 metric |

## 7. v5 에서 결정하지 않은 것

- **학습형 sensitivity map** (E2E-VarNet 스타일) — DC block 의 sens 품질이 천장이 되는지 확인 후 결정
- **DC iteration 수 증가** (1→2~4) — capacity ceiling 이 dropout/aug 만으로 풀리지 않을 때 v6 후보
- **ETER v5** — [eter_v4_analysis.md §4](eter_v4_analysis.md) 의 동일 레시피로 별도 진행 가능

## 8. 실행

```bash
# GPU 비어 있을 때 즉시 시작
./run_chain_ss2d_v5.sh &
# 또는 다른 학습이 끝나길 기다리며
WAIT_PID=<현재학습PID> ./run_chain_ss2d_v5.sh &
```

로그:
- 단계 start/end: `run_chain_v5.log`
- 학습 stdout/stderr: `run_ss2d_v5.log`
- 체크포인트 / 에폭 로그: `logs/SS2D_ViT_R4_brain320_v5/`

## 9. 참고 문서

- [SS2D_v1_analysis.md](SS2D_v1_analysis.md) — 원본 문제 진단 (v1 ~ v3 미해결 항목 추적)
- [ss2d_v4_changes.md](ss2d_v4_changes.md) — A+B+C (capacity / regularization / DC) 도입
- [eter_v4_analysis.md](eter_v4_analysis.md) — capacity ceiling 가설과 v5 plan 출처
- [scheduler_change.md](scheduler_change.md) — CosineAnnealingLR 적용 (v5 도 그대로)
