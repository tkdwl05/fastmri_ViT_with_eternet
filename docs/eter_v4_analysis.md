# ETER ViT v4 학습 분석 및 v5 계획

날짜: 2026-04-27

## 1. 결과 요약

| 항목 | v3 (WarmRestarts) | v4 (CosineAnnealingLR) |
|---|---|---|
| Best val SSIM | **0.7475** (ep ~179) | **0.7320** (ep ~30~40) |
| 200ep 종료 val SSIM | (별도 기록) | 0.7235 |
| 200ep 종료 train SSIM | — | 0.7790 |
| Train-Val gap (200ep) | — | 0.055 |
| Best PSNR | 34.97 dB | 32.99 dB |

v4가 v3 대비 ~2.1% 회귀. val SSIM은 ep 30~40에 피크 후 단조 감소.

## 2. v3 → v4 변경점

[scheduler_change.md](scheduler_change.md) 참고. **단일 변경: scheduler.**
- v3: `CosineAnnealingWarmRestarts(T_0=1, T_mult=2)` — ep 1, 3, 7, 15, 31, 63, 127에서 LR 재가열
- v4: `CosineAnnealingLR(T_max=total_steps, eta_min=1e-6)` — 단일 부드러운 cosine decay
- 모델/데이터/regularization은 동일

## 3. 회귀 원인 가설

1. **WarmRestarts 부재의 부작용**: 주기적 재가열이 saddle escape 역할을 했을 가능성. 단일 cosine decay는 평탄한 minimum에서 머무름. 이는 [scheduler_change.md](scheduler_change.md)가 가정한 "톱니 LR이 pixel precision에 해롭다"와 반대 결과.
2. **Capacity ceiling 조기 도달**: v4는 ep 30~40에 val 피크. 이후는 train만 fit되고 val noise/overfit. [eter_8gb축소.md](eter_8gb축소.md)에서 GRU hidden 축소(10→2/4)로 표현력 한계가 이미 지적됨.
3. **EarlyStopping 부재**: best 이후 165 epoch을 낭비, train-val gap 누적(0.055).

## 4. v5 계획 (config/docs 단계, 실행 별도)

- **EarlyStopping**: patience=20 (val SSIM 기준), best ckpt 보존
- **weight_decay**: 1e-7 → 1e-5 (SS2D v4와 동일 수준)
- **Transformer decoder dropout**: 0.1 → 0.2
- (선택) **data augmentation**: H/V flip — [ss2d_v4_changes.md](ss2d_v4_changes.md) §7 후보를 ETER에도 동일 적용
- (선택) **scheduler 복귀**: WarmRestarts T_0=20, T_mult=2 + EarlyStop 결합

## 5. 참고 문서

- [scheduler_change.md](scheduler_change.md)
- [eter_8gb축소.md](eter_8gb축소.md)
- [SS2D_v1_analysis.md](SS2D_v1_analysis.md)
- [ss2d_v4_changes.md](ss2d_v4_changes.md)
