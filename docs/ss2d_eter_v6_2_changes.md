# SS2D / ETER v6_2 — λ_grad 완화 (v6_1 over-sharpening 회귀 대응)

## 배경: v6_1 결과 분석

v6_1 (gradient L1 loss `λ_grad=10.0` 50ep fine-tune) 풀평가 결과:

| 모델 | PSNR (dB) | SSIM | NMSE | L1 |
|---|---:|---:|---:|---:|
| SS2D-ViT v6   | 34.81 ± 2.78 | 0.8913 ± 0.0974 | 0.00906 | 7.37 |
| SS2D-ViT v6_1 | **34.18 ± 2.75** | **0.8953 ± 0.0928** | 0.01018 | 7.76 |
| Δ (v6 → v6_1) | **−0.63** | **+0.0040** | +0.00112 | +0.39 |
| ETER-ViT v6   | 33.59 ± 2.74 | 0.8862 ± 0.0938 | 0.01143 | 8.21 |
| ETER-ViT v6_1 | **33.36 ± 2.72** | **0.8865 ± 0.0940** | 0.01207 | 8.46 |
| Δ (v6 → v6_1) | **−0.23** | +0.0003 (≈ 0) | +0.00064 | +0.25 |

**기대 vs 실제**: v6_1 doc 은 "PSNR +0.3~0.8 dB" 를 기대했으나 양 모델 모두 PSNR 회귀.
SSIM 미세 상승(SS2D)~정체(ETER) 대비 PSNR/NMSE/L1 모두 후퇴. 동 doc 의 fallback 시나리오
("gradient loss 가 over-sharpening 을 유발하면 λ_grad 5.0 으로 줄여 재학습") 가 그대로
발생한 상태.

### 중간 슬라이스 시각 비교 (8슬라이스 평균, [results/vis_compare_versions/](../results/vis_compare_versions/))

| 모델 | PSNR | SSIM |
|---|---:|---:|
| SS2D v6 | 34.55 | 0.8939 |
| SS2D v6_1 | 33.58 (−0.97) | 0.8971 (+0.003) |
| ETER v6 | 33.26 | 0.8891 |
| ETER v6_1 | 32.84 (−0.42) | 0.8828 (−0.006) |

중간 슬라이스에선 ETER 가 PSNR/SSIM 모두 후퇴 — SS2D 보다 더 명확한 over-edge 회귀.

### 원인 가설

- `λ_grad=10` 으로 gradient L1 기여가 L1 본항과 같은 자릿수까지 끌어올려졌고,
  이미 수렴한 v6 weight 에서 edge 강도를 강하게 처벌 → 픽셀 평균값을 GT 에서
  멀리 끌어당겨 PSNR/L1/NMSE 후퇴.
- SSIM 은 local mean/variance 기반이라 edge sharpening 으로 약간 이득을 봤으나
  pixel-wise fidelity 손실은 정량지표 3개에서 일관 확인됨.

---

## v6_2 변경점 (단일변수)

| 항목 | v6_1 | v6_2 |
|---|---:|---:|
| **LAMBDA_GRAD_PER_PIXEL** | **10.0** | **3.0** (−70%) |
| RESUME_CKPT | v6 best | **v6 best (동일, v6_1 X)** |
| NUM_EPOCHS | 50 | 50 |
| LEARNING_RATE_ADAM | 5e-5 | 5e-5 |
| LR cosine min | 5e-7 | 5e-7 |
| LAMBDA_SSIM_PER_PIXEL | 1.0 | 1.0 |
| EARLYSTOP_PATIENCE | 5 | 5 |
| VAL_EVERY_N_EPOCHS | 5 | 5 |
| BATCH_SIZE | 4 | 4 |
| dropout / weight_decay / flip aug | v6 동일 | v6 동일 |
| Dataloader / Model class / DC block | v6 동일 | v6 동일 |

### 왜 v6 에서 재시작 (v6_1 best 가 아닌)

- v6_1 의 over-edged state 를 누적시키지 않고 깨끗한 시작점에서 λ_grad 단일변수 효과 검증.
- v6_1 weight 에서 그냥 λ_grad 만 줄여 이어 학습하면 over-edge 가 부분적으로 잔존,
  v6_2 가 v6_1 의 후처리인지 본격 비교인지 모호.

### λ_grad = 3.0 선택 근거

- v6_1 doc 의 fallback 권고치 "5.0" 보다 더 보수적으로 3.0 선택.
- gradient L1 vs L1 본항의 raw magnitude 비율을 고려할 때 λ_grad=3 이면 gradient 기여가
  L1 본항의 ~30% 수준에 머무름 (over-shadow 방지, edge sharpening 잔존).
- v6_2 결과가 여전히 PSNR 후퇴면 λ_grad=1.0 또는 gradient loss 자체 제거 (v6 그대로 유지)
  로 추가 시도.

---

## 신규 파일

| 파일 | 역할 |
|---|---|
| `configs/myConfig_choh_SS2D_model_v6_2.py` | SS2D v6_2 config (λ_grad=3.0) |
| `configs/myConfig_choh_ETER_model_v6_2.py` | ETER v6_2 config (동일) |
| `main_train_ss2d_v6_2.py` | v6_1 train script clone (config import 만 v6_2) |
| `main_train_eter_v6_2.py` | 위와 동일 |
| `runs/chain/run_chain_v6_2.sh` | SS2D v6_2 → ETER v6_2 체인 |
| `logs/SS2D_ViT_R4_brain320_v6_2/`, `logs/ETER_ViT_R4_brain320_v6_2/` | 실행 시 자동 생성 |

---

## 기대 결과 (가설)

- **PSNR**: v6_1 의 −0.63 (SS2D) / −0.23 (ETER) 회귀가 0 근처로 복귀.
  v6 와 동등 또는 +0.1~0.3 dB 소폭 상승.
- **SSIM**: v6_1 의 미세 상승 (+0.004) 대부분 유지, 또는 ±0.002 동등.
- **L1 / NMSE**: v6 수준으로 복귀 (v6_1 의 5~10% 악화 회복).

만약 v6_2 도 여전히 PSNR/L1 후퇴라면:
1. λ_grad → 1.0 한 단계 더 완화 시도, 또는
2. gradient L1 → Charbonnier smooth penalty 로 outlier 완화, 또는
3. gradient loss 자체 폐기, v6 를 최종으로 확정.

---

## 작업 순서

1. ✅ 본 문서 작성
2. ✅ configs/v6_2 두 개 작성
3. ✅ train script v6_2 두 개 작성 (v6_1 clone + sed)
4. ✅ chain script `runs/chain/run_chain_v6_2.sh`
5. CLAUDE.md 핵심 문서 섹션 등록
6. 실행 → SS2D 50ep → ETER 50ep → eval/visualize 재실행 → 결과 비교
