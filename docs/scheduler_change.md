# LR 스케줄러 변경: WarmRestarts → CosineAnnealingLR

날짜: 2026-04-20

## 1. 동기

기존 SS2D v2 (200 epoch) wandb 로그에서 **LR 톱니 패턴**이 관찰됨. 원인은 스케줄러가 `CosineAnnealingWarmRestarts(T_0=1 epoch, T_mult=2, eta_min=1e-6)` 였기 때문. Adam 옵티마이저와는 별개로 LR 정책이 주기적으로 **리스타트**하며 LR을 최댓값으로 되돌려 튕겨올린다.

리스타트 시점: epoch **1, 3, 7, 15, 31, 63, 127** (T_mult=2로 사이클 길이가 2배씩 증가).

### Warm restarts의 의도와 리스크

- **의도**: 주기적 LR 상승으로 local minima 탈출 (Loshchilov & Hutter, SGDR, ICLR 2017).
- **리스크**: 수렴이 막 일어나려는 시점에 LR이 튀어 val loss가 뒤집힘. **픽셀 단위 정밀 수렴**이 중요한 MRI 재구성에서는 득보다 실이 큼.

## 2. 변경 내용

### 스케줄러

`torch.optim.lr_scheduler.CosineAnnealingLR(T_max=total_steps, eta_min=1e-6)` 로 교체. 리스타트 없이 200 epoch 동안 LR을 2e-4 → 1e-6으로 **한 번만** 부드럽게 감소.

`scheduler.step()`이 배치 단위로 호출되므로 `T_max = steps_per_epoch × NUM_EPOCHS` 로 설정해 호출 주기를 기존 코드와 맞춤.

- [main_train_ss2d.py:153-158](../main_train_ss2d.py#L153-L158)
- [main_train_eter.py:139-144](../main_train_eter.py#L139-L144)
- wandb config의 `scheduler` 필드도 `CosineAnnealingLR`로 갱신 (비교 가능성 유지).

### 버전 디렉토리

기존 결과 보존을 위해 `PATH_FOLDER`를 한 단계씩 올림.

| 모델 | 이전 | 변경 후 |
|---|---|---|
| SS2D | `logs/SS2D_ViT_R4_brain320_v2/` | `logs/SS2D_ViT_R4_brain320_v3/` |
| ETER | `logs/ETER_ViT_R4_brain320_v3/` | `logs/ETER_ViT_R4_brain320_v4/` |

- [configs/myConfig_choh_SS2D_model.py:9](../configs/myConfig_choh_SS2D_model.py#L9)
- [configs/myConfig_choh_ETER_model.py:9](../configs/myConfig_choh_ETER_model.py#L9)

### 체인 실행 스크립트

[run_chain_ss2d_then_eter.sh](../run_chain_ss2d_then_eter.sh) 추가.

- `WAIT_PID=5792` (현재 ETER v3 학습) 종료 대기 → SS2D v3 학습 → exit 0이면 ETER v4 학습
- 각 단계 start/end 시각은 `run_chain.log`, stdout/stderr은 `run_ss2d_v3.log`, `run_eter_v4.log` 로 분리.

## 3. 결정하지 않은 것

- **ReduceLROnPlateau**: val loss 정체 시 LR 감소 방식. 현 변경이 불충분하면 다음 후보.
- **Constant LR (스케줄러 없음)**: 가장 단순하나 막판 과적합/진동 가능.

## 4. 공정 비교 유지

SS2D와 ETER 모두 동일한 스케줄러(`CosineAnnealingLR`, 동일 하이퍼파라미터)를 쓰도록 맞춤. 이전 v2/v3 세대의 `WarmRestarts` 조건과 비교하려면 반드시 **같은 세대 내에서**(SS2D v3 vs ETER v4) 비교할 것.
