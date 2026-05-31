# SS2D / ETER v6_1 — Gradient(edge) Loss Fine-tune

## 배경

v6 200ep 풀학습 결과 ([results/eval_full_v6_summary.txt](../results/eval_full_v6_summary.txt)):

| 모델 | PSNR (dB) | SSIM | NMSE |
|---|---:|---:|---:|
| SS2D-ViT v6 | 34.81 ± 2.78 | 0.8913 ± 0.0974 | 0.00906 |
| ETER-ViT v6 | 33.59 ± 2.74 | 0.8862 ± 0.0938 | 0.01143 |
| U-Net (PT) | 34.66 ± 2.95 | 0.8858 ± 0.1222 | 0.00737 |

정량적으론 SS2D > U-Net > ETER, 모두 SSIM 0.89 수준. 그러나 [docs/visual_metric_gap_v6.md](visual_metric_gap_v6.md) 진단 결과:

- **fine detail 흐림** — best 슬라이스 PSNR 42~44dB 인데 median/worst 케이스에선 sulci/혈관 detail 평균화
- L1+SSIM loss 가 **mean-prediction 편향**을 처벌하지 않음 (SSIM은 local mean/variance 기반)

→ v6_1 은 이 mean-prediction 편향을 직접 처벌하는 **gradient (edge) loss** 를 추가하여 v6 best 로부터 fine-tune.

---

## 변경점

### 1. Gradient (edge) loss 추가 — 원인 3 직접 대응

기존 loss (v6):
```python
loss = L1(pred, gt) + λ_ssim * (1 - SSIM(pred, gt))
# λ_ssim = 1.0
```

v6_1:
```python
def gradient_loss(pred, gt):
    # finite-difference, axis=H, W
    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dx_gt   = gt[...,   :, 1:] - gt[...,   :, :-1]
    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dy_gt   = gt[...,   1:, :] - gt[...,   :-1, :]
    return F.l1_loss(dx_pred, dx_gt) + F.l1_loss(dy_pred, dy_gt)

loss = L1(pred, gt) + λ_ssim * (1 - SSIM(pred, gt)) + λ_grad * grad_loss(pred, gt)
# λ_ssim = 1.0, λ_grad = 10.0
```

**λ_grad 선택 근거**:
- L1 loss 값 ~ 1e-5 (fastMRI raw amplitude scale)
- gradient L1 ~ 1e-6 (인접 픽셀 차이)
- λ_grad = 10.0 으로 gradient loss 기여를 L1 과 비슷한 자릿수로 끌어올림
- 첫 epoch 후 loss 비율 확인하여 5.0 ~ 20.0 범위로 미세조정 가능

### 2. v6 best ckpt 로부터 fine-tune (from-scratch 재학습 X)

- SS2D v6_1: `logs/SS2D_ViT_R4_brain320_v6/ss2d_vit_best.pt` → epoch 0 부터 fine-tune
- ETER v6_1: `logs/ETER_ViT_R4_brain320_v6/eter_vit_best.pt` → epoch 0 부터 fine-tune
- `model_state` 만 load, **optimizer / scheduler 는 새로 시작** (loss 분포가 바뀌어 momentum/lr 그대로 쓰면 발산 위험)

### 3. Fine-tune hyperparameter (50 epoch 짧게)

| 항목 | v6 | v6_1 |
|---|---|---|
| NUM_EPOCHS | 200 | **50** |
| LR (CosineAnnealing) | 1e-4 → 1e-6 (200ep) | **5e-5 → 5e-7 (50ep)** |
| EARLYSTOP_PATIENCE | 10 val checks | **5 val checks** (≈25ep) |
| VAL_EVERY_N_EPOCHS | 5 | 5 |
| BATCH_SIZE | SS2D 4 / ETER 4 | 동일 |
| LAMBDA_SSIM_PER_PIXEL | 1.0 | 1.0 |
| **LAMBDA_GRAD_PER_PIXEL** | — | **10.0 (신규)** |
| weight_decay | 3e-5 | 3e-5 |
| dropout | 0.2 | 0.2 |

**LR을 1e-4 보다 절반(5e-5)으로 낮춘 이유**: 이미 수렴한 weight 에 새 loss term 을 도입하므로 큰 LR 로 시작하면 기존 SSIM 학습 성과를 깨뜨릴 수 있음. 5e-5 면 v6 cosine 최종부 LR 보다는 1.5~2 자리수 높아 fine-tune 충분.

### 4. Dataloader / model / regularization 은 v6 그대로

- `dataloader_h5_v5.py` 그대로 사용 (size-relaxed + flip aug, val 7270 슬라이스)
- 모델 클래스 (`choh_Decoder_SS2D_ViT_v4`, `choh_Decoder3_ETER_v5`) 동일 import
- DC block 동일

### 5. SSIM val 평가는 v6 그대로 (skimage `structural_similarity`)

`val_loop_ssim()` 의 SSIM 계산은 v6 와 동일. EarlyStop / best 기준도 `val_ssim` 단일.

---

## 신규 파일

| 파일 | 역할 |
|---|---|
| `configs/myConfig_choh_SS2D_model_v6_1.py` | SS2D v6_1 config (v6 base + λ_grad, fine-tune epochs/LR) |
| `configs/myConfig_choh_ETER_model_v6_1.py` | ETER v6_1 config (동일) |
| `main_train_ss2d_v6_1.py` | SS2D v6_1 train script (v6 base + gradient_loss term) |
| `main_train_eter_v6_1.py` | ETER v6_1 train script (동일 구조) |
| `runs/chain/run_chain_v6_1.sh` | SS2D v6_1 → ETER v6_1 체인 실행 |
| `logs/SS2D_ViT_R4_brain320_v6_1/` | ckpt + train_log.txt (실행 시 자동 생성) |
| `logs/ETER_ViT_R4_brain320_v6_1/` | ckpt + train_log.txt (실행 시 자동 생성) |

---

## 기대 결과

- val SSIM: v6 0.8913 (SS2D) / 0.8862 (ETER) 에서 **±0.005 범위 유지 또는 소폭 상승**
- PSNR: gradient loss 가 edge 영역의 에러를 강하게 처벌 → **PSNR +0.3~0.8 dB** 상승 예상
- visualize_diagnostic_v6 결과 재실행 시 zoom-in 행에서 sulci/혈관 detail **시각적으로 sharp** 해지는지 확인
- error map 의 brain 외곽선/sulci 경계 강도 감소

만약 gradient loss 가 over-sharpening (noise 증폭) 을 유발하면 λ_grad 5.0 으로 줄여 재학습.

---

## 작업 순서

1. ✅ 본 문서 작성
2. config v6_1 두 개 작성 (SS2D, ETER)
3. train script v6_1 두 개 작성 (v6 클론 + gradient_loss term 삽입 + RESUME 처리)
4. chain script 작성
5. CLAUDE.md 핵심 문서 섹션 등록
6. 실행 → SS2D 50ep → ETER 50ep → eval/visualize 재실행 → 결과 비교
