# Tier 2 — Sharpness recovery 실험 (2026-05-25 launch)

Tier 1 (TTA / 앙상블) 이 v6 baseline 대비 임계 미달 → 재학습 시도.
세 가지 직교 가설로 v6_2 / v6_3 / v6_4 fine-tune 동시 준비.

## 가설 매트릭스

모두 "v6 의 mean-prediction blurring" 을 다른 각도로 처벌:

| 버전 | 가설 | 변경점 | loss |
|---|---|---|---|
| v6_1 | 직접 edge 처벌 (강함) | λ_grad=10 | L1 + (1-SSIM) + 10·grad_L1 |
| v6_2 | edge 처벌 완화 (v6_1 회귀 대응) | λ_grad=3 | L1 + (1-SSIM) + 3·grad_L1 |
| **v6_3** | sharp 허용 (regularization 완화) | dropout 0.2→0.1, WD 3e-5→1e-5 | L1 + (1-SSIM) |
| **v6_4** | feature-space 처벌 | +λ_perc=0.1 VGG | L1 + (1-SSIM) + 0.1·VGG_perc |

공통:
- v6 best ckpt 부터 50ep fine-tune
- LR 5e-5 → 5e-7 cosine
- EarlyStop patience 5
- VAL_EVERY_N_EPOCHS = 5
- BATCH_SIZE 4, dataloader v5

## 현재 상태 (2026-05-25 15:30)

| 버전 | 상태 | PID | ETA |
|---|---|---|---|
| v6_2 SS2D | running | 44459 (bash), 44467 (py) | ~3-6h |
| v6_2 ETER | queued in v6_2 chain | — | start after v6_2 SS2D |
| v6_3 SS2D | queued (WAIT_PID 44459) | 44621 (bash) | start after v6_2 chain |
| v6_3 ETER | queued in v6_3 chain | — | start after v6_3 SS2D |
| v6_4 SS2D | **NOT linked** — manual review after v6_2/v6_3 | — | — |

## v6_4 (perceptual loss) — 수동 launch 절차

v6_2 / v6_3 모두 회귀하면 v6_4 시도:

```bash
# train script 미작성 — main_train_ss2d_v6_2.py 클론 후
#   from myConfig_choh_SS2D_model_v6_4 import *
#   from perceptual_loss import VGGPerceptualLoss
#   perceptual = VGGPerceptualLoss(device, layers=('relu1_2','relu2_2','relu3_3'))
#   loss = L1 + λ_ssim·(1-SSIM) + LAMBDA_PERCEPTUAL · perceptual(pred, gt)
# 동일 패턴으로 v6_4 train script 2개 + chain 작성
```

준비물:
- `configs/myConfig_choh_{SS2D,ETER}_model_v6_4.py` (작성 완료)
- `tools/perceptual_loss.py` (VGG16 helper, 작성 완료)
- `main_train_*_v6_4.py` (미작성)
- `runs/chain/run_chain_v6_4.sh` (미작성)

## 결정 트리

v6_2 끝난 후 (eval_full_compare 으로 평가):
- **PSNR ≥ v6 동등 & SSIM > v6** → v6_2 채택. v6_3/v6_4 보류.
- **PSNR < v6 회귀** → v6_2 폐기. v6_3 결과 대기.
  - v6_3 도 회귀하면 → v6_4 launch.
  - v6_3 성공하면 → v6_3 채택, v6_4 보류.
- **두 버전 모두 마진 (PSNR ±0.1)** → SSIM 으로 결정, 시각 비교까지 가서 채택.

## Tier 1 결과 (보존)

부분 평가 (500 슬라이스) 에서 TTA / 앙상블 임계 미달. 풀평가 생략. 자세한 결과는
[docs/tier1_tta_ensemble_negative.md](tier1_tta_ensemble_negative.md).
