# ViT 기반 MRI 재구성 — 연구 진행 발표 정리

작성일: 2026-05-06 (v6 진행 중)
최종 갱신: **2026-05-28** (v6 완료 + Tier 1/2 실험 결과 반영)
대상: fastMRI brain AXFLAIR multicoil R=4 재구성

---

## 0. 한 줄 요약

ViT 인코더와 시퀀스 모델 디코더(GRU = ETER, Mamba = SS2D)를 결합한 MRI 재구성 모델을
**v1 → v6 → v6_3** 까지 단계적으로 개선해, fastMRI pretrained U-Net 베이스라인(SSIM 0.8858)을
**SSIM·PSNR 모두에서 능가** (SS2D-ViT v6_3: SSIM **0.8924**, PSNR **36.05 dB**)했다.
연구 과정에서 두 차례의 핵심 발견 — **(1) 학습 평가 metric 의 버그**, **(2) loss 함수가 만드는
mean-prediction blurring** — 이 결과의 절반을 만들었다.

---

## 1. 프로젝트 개요

| 항목 | 값 |
|---|---|
| 데이터 | fastMRI brain AXFLAIR multicoil (16 코일, 320×320) |
| 가속률 | R = 4 equispaced (center fraction 0.08) |
| 입력 | 언더샘플링된 k-space + aliased 이미지 |
| 출력 | reconstruction_rss 기반 단일 채널 magnitude (320×320) |
| 비교 베이스라인 | fastMRI 공식 pretrained U-Net (chans=256, 약 496M params) |
| GPU | RTX 5060 Ti **8 GB** (메모리 제약이 모든 의사결정의 핵심) |
| 환경 | conda `mri_env`, PyTorch + mamba_ssm CUDA 커널 |

핵심 엔트리포인트:
- `main_train_eter.py`, `main_train_ss2d.py` (v3 까지) → `main_train_*_v4.py`,
  `main_train_*_v5.py`, `main_train_*_v6.py` (각 세대별)
- 평가: `eval.py`, `eval_full_compare.py`
- 시각화: `visualize.py`, `visualize_compare.py`

---

## 2. 사용 데이터

### 2.1 데이터 출처
- **fastMRI brain multicoil** 공개 데이터셋
- 컨트라스트: AXFLAIR
- 코일 수 16개, real/imag 분리해 32 채널로 입력 사용

### 2.2 H5 파일 구조와 키
- `kspace`: complex multicoil k-space `(slice, coil, H, W)`
- `reconstruction_rss`: 공식 GT magnitude `(slice, H, W)`
- 파일별 rss shape 이 320×320, 320×264, 768×396 등으로 **이기종**

### 2.3 마스크 / 전처리 (`dataloaders/dataloader_h5.py`)
1. `ifft2c` (`ifftshift → ifft2 → fftshift`) 로 image domain 변환
2. **image domain 에서 320×320 center crop**
3. `fft2c` 로 k-space 재구성
4. **R=4 equispaced 1D mask** (phase encoding 방향, center 8%)
5. masked k-space 를 다시 `ifft2c` → aliased 이미지 (`data_img`)
6. 출력 스케일링: `ksp × 1e4`, `img × 1e6`, `gt × 1e6`

### 2.4 버전별 데이터 풀 (이게 모델 성능에 직접 영향을 줌)

| 세대 | dataloader | 필터링 정책 | train slice | val slice |
|---|---|---|---|---|
| v1–v4 | `dataloader_h5.py` / `_v4.py` | rss shape 이 정확히 320×320 인 파일만 채택 | 8,548 | **4,492** |
| v5–v6 | `dataloader_h5_v5.py` | rss 존재만 확인, image-domain crop/pad 로 모든 파일 흡수 | 14,262 (+67%) | **7,270 (+62%)** |

> v5 의 7,270 slice 가 **fastMRI U-Net leaderboard 평가 슬라이스 수와 동일**해,
> 비로소 같은 모집단에서 직접 비교가 가능해졌다.

### 2.5 Augmentation
- v1 ~ v4: 없음
- **v5 ~ v6**: image-domain H/V flip (p=0.5)
  - flip 후 `fft2c → mask → ifft2c` 파이프라인 그대로 통과시켜 정합성 유지
  - `gt_rss` 도 동기화 flip

---

## 3. 공통 학습 파이프라인

### 3.1 모델 데이터 흐름

```
입력 1: aliased image (B, 32, 320, 320)
입력 2: k-space        (B, 32, 320, 320)
       │                                          │
   ViT 인코더 (공통)                          시퀀스 모델
   patch 32×32 → 100 토큰                ┌────────┴────────┐
   384 dim, 6 layer, 6 head              │                  │
       │                                ETER             SS2D
   ViT 디코더 (공통)                  Bi-GRU h/v      4-방향 SSM
   512 dim, 6 layer, 8 head           (수평→수직)    (Mamba 커널)
       │                                  │
   final_linear (512→65,536)              │
   PixelShuffle 업샘플 (×2)               │
       │                                  │
       └─────── concat ────────┬──────────┘
                               │
                       최종 합성 (Conv 또는 RefinementBlock)
                               │
                       출력 (B, 1, 320, 320)
```

### 3.2 학습 설정 (전 버전 공통, 변동분은 §4 표시)

| 항목 | 기본값 | 변경 이력 |
|---|---|---|
| Loss | `L1 + λ × (1 − SSIM)` | λ: v1 = 0.2 → v2~ = 1.0 |
| Loss precision | fp32 (autocast 밖) | label ≈ 959 → SSIM 제곱이 fp16 max(65504) 초과 방지 |
| Optimizer | Adam, lr = 2e-4 | weight_decay v1=1e-7 → v4=1e-5 → v5+=3e-5 |
| Scheduler | `CosineAnnealingWarmRestarts` (T₀=1, T_mult=2) | v3+ `CosineAnnealingLR` (T_max=total_steps, η_min=1e-6) |
| AMP | GradScaler + autocast | forward fp16, loss fp32 |
| Batch size | 8 | SS2D v4+ DC block 메모리 → 4 |
| Epoch | 200 | v5+ EarlyStopping 조기 종료 |
| Validation | train SSIM best 갱신 시 | v5+ 매 N epoch 강제 + EarlyStop |
| Logging | wandb (project: ViT-MRI-Recon) | gradient histogram, batch/epoch 단위 |

### 3.3 평가 지표
- PSNR, NMSE, SSIM, L1 (per-slice 평균)
- **SSIM 정의 변천 (★ 발표 핵심 발견):**
  - v1 ~ v5 학습 로그: `u_choh_SSIM` (커스텀, fastMRI 표준 아님 — 발견 ① 참조)
  - 최종 비교: `skimage.metrics.structural_similarity`,
    `data_range = target.max() − target.min()` (fastMRI 표준)

---

## 4. 버전별 변천사

각 버전은 **배경 → 변경 → 근거 → 결과** 4단으로 정리한다.

---

### v1 — Baseline ViT + (GRU 또는 SS2D) (2026-04-09)

**배경**
- 8 GB GPU 환경에 맞춰 원본 ETER-Net 을 절반 축소 + 320×320 입력으로 시작.

**구성**
- patch 32×32 (100 토큰), ViT-Small 인코더, GRU hidden=2 / SS2D `d_inner=32, d_state=8`
- 최종 합성: `Conv2d(308 → 1, 3×3)` 단일 레이어 (파라미터 2,773개)
- Loss: `L1 + 0.2 × (1 − SSIM)`, scheduler: `CosineAnnealingWarmRestarts`

**근거**
- 원본 ETER-Net (RTX 서버, 384×384, GRU hidden=10, U-Net 후처리) 을 8 GB 에 맞추기
  위해 인코더(Base→Small), GRU(10→2), 후처리(U-Net→Conv 1개) 를 **모두 축소**
- 자세한 축소 의사결정 표는 `docs/eter_8gb축소.md` 참조

**결과 (200 epoch, val 4,492)**
| 지표 | 값 |
|---|---|
| Val SSIM | **0.599** |
| Val PSNR | 26.45 dB |
| Train-Val gap | 0.066 (과적합) |

**문제 진단** (`docs/SS2D_v1_analysis.md`)
1. patch 32×32 → 공간 정보 85:1 압축, 고주파 손실
2. 정보 병목: 입력 3.27M 값 → 인코더 38K 값 (1.2%)
3. 최종 합성 Conv 2,773 params 로 308채널 결합 = 사실상 가중평균 → blurry
4. SS2D `d_inner=32` 로 k-space 상관관계 포착 불가
5. SSIM weight 0.2 → 실제 loss 기여도 0.7%
6. 과적합 gap 0.066
7. Data Consistency 부재

---

### v2 — 핵심 4축 동시 개선 (2026-04 중)

**배경**
- v1 의 7가지 문제 중 4개를 동시에 수술.

**변경**
| 항목 | v1 | v2 |
|---|---|---|
| Patch size | 32×32 (100 토큰) | **16×16 (400 토큰)** |
| 최종 합성 | Conv 1개 (2.7K params) | **RefinementBlock** (3× ResBlock, ~120K params) |
| SSIM weight | 0.2 (기여 0.7%) | **1.0** (기여 ~3.5%) |
| Decoder OUT_CH × FEAT | 256 × 16 | 64 × 8 |
| (ETER) GRU hidden | 2 | **4** |

**근거**
- 압축비 85:1 → 21:1 로 완화해 고주파 보존
- RefinementBlock 으로 단일 conv 가중평균 한계 극복
- SSIM loss 가 실제로 학습 신호에 기여하도록 weight 상향

**결과 (SS2D 132 epoch)**
| 지표 | v1 | v2 | 개선 |
|---|---|---|---|
| Val SSIM | 0.599 | **0.759** | +0.160 |
| Val PSNR | 26.45 | **36.38** | +9.93 dB |
| Val NMSE | 0.1278 | **0.0086** | 14.8× |
| Train-Val gap | 0.066 | **0.018** | 과적합 대폭 감소 |

> v2 단계에서 시각적으로도 blurry 가 사라지고 구조 디테일이 회복됨.

---

### v3 — Scheduler 교체 (CosineAnnealingLR) (2026-04-20)

**배경**
- v2 wandb 로그에서 LR 톱니 패턴 관찰.
- `WarmRestarts` 가 epoch 1, 3, 7, 15, 31, 63, 127 에서 **LR 재가열**해
  픽셀 정밀 수렴 직전에 튕겨나가는 현상.

**변경**
- `CosineAnnealingWarmRestarts(T₀=1, T_mult=2)` → `CosineAnnealingLR(T_max=total_steps, η_min=1e-6)`
- 200 epoch 동안 **단일 부드러운 cosine decay** (재가열 없음)
- SS2D v2 → **v3**, ETER v3 → **v4** 로 디렉토리 분리해 결과 보존

**근거**
- MRI 재구성은 픽셀 단위 정밀 수렴이 결정적 → 톱니 LR 의 saddle escape 효과보다 손실이 큼
- SS2D 와 ETER 모두 동일 스케줄러로 통일해 공정 비교 유지

**결과** (SS2D v3 vs ETER v4 단계 별로 진행)
- SS2D v3: val SSIM ≈ 0.74 부근까지 안정 수렴
- **ETER v4: 0.7320 (← v3 의 0.7475 대비 -0.0155 회귀)** 발생

---

### v3.5 / v4 — Capacity + Regularization + Data Consistency (2026-04-22 ~ 04-27)

**배경**
- ETER v4 회귀 (`docs/eter_v4_analysis.md`) → "단일 cosine decay 로는 capacity ceiling 에서
  early peak 이후 단조 감소" 라는 가설.
- SS2D 측은 `d_inner=32, d_state=8` 그대로라 표현력 한계가 의심됨.
- v1 분석 체크리스트의 **#4(SS2D 용량), #5(과적합), #6(Data Consistency)** 가 미해결.

**변경 (SS2D v4 — A + B + C 동시 적용)**

| 축 | v3 | v4 |
|---|---|---|
| **A. SS2D capacity** | `d_inner=32, d_state=8` | **`d_inner=64, d_state=16`** |
| **B. Regularization** | weight_decay=1e-7, dropout=0 | **weight_decay=1e-5, Transformer dropout=0.1** |
| **C. Data Consistency** | 없음 | **1-iter soft DC block** (ACS 기반 sens 추정 + 학습 α) |
| Batch size | 8 | **4** (SS2D forward gradient checkpointing 까지 적용해도 merge tensor 800 MiB → BS=4 로 fit) |

**DC block 동작**
1. 모델 출력 (real, imag) → coil-combined complex `x_c`
2. `multicoil = sens_c × x_c`, `k_pred = fft2c(multicoil)`
3. `k_dc = k_pred + mask × α × (k_meas_scaled − k_pred)` (soft DC)
4. `ifft2c(k_dc)` → coil-combine → magnitude 출력
- AMP 안정성 위해 DC 내부는 `autocast(enabled=False)` + fp32 cast

**근거**
- 표현력(A), 일반화(B), 물리 사전 지식(C) 의 **세 축을 분리하지 않고 동시에 시도**해
  v3 대비 의미 있는 개선 폭이 나오는지 먼저 확인 → 추후 ablation 으로 분해
- DC 는 sens 를 ACS-only 로 추정해 추가 모듈 부담 없이 시작

**결과 (SS2D v4 best ckpt, val 4,492)**
| 지표 | v4 best | U-Net 베이스라인 |
|---|---|---|
| Val SSIM | 0.7340 | 0.8865 |
| Val PSNR | 34.97 dB | 35.03 dB |
| Val NMSE | 0.0093 | 0.0075 |

> PSNR 은 U-Net 과 거의 동등하지만 **SSIM 갭이 0.15** 로 여전히 큼.
> 이 갭이 **발견 ① (평가 metric 의 결함)** 으로 이어진다.

**ETER v4 결과 (200ep 풀 학습, scheduler 만 변경)**
| 항목 | v3 | v4 |
|---|---|---|
| Best val SSIM | 0.7475 (ep ~179) | **0.7320 (ep ~30~40)** |
| 200ep 종료 val SSIM | — | 0.7235 |
| 회귀 폭 | — | -2.1% |

---

### v5 — 데이터 풀 확장 + 일반화 강화 + EarlyStopping (2026-04-30)

**배경 (SS2D 와 ETER 양쪽 동일 레시피로 진행)**
- v4 의 두 가지 회귀 원인:
  1. 학습 분포가 좁음 — strict 320×320 필터로 fastMRI corpus 의 ~60% 만 사용
  2. 일반화 부족 — capacity ceiling 도달 후 best 이후 epoch 낭비

**변경 (v4 모델은 그대로, 데이터 + 학습 루프만 수정)**

| 축 | v4 | v5 |
|---|---|---|
| **A. 데이터 풀** | strict 320×320 (train 8,548 / val 4,492) | image-domain crop/pad (train 14,262 / val **7,270**) |
| **A. Augmentation** | 없음 | H/V flip p=0.5 |
| **B. Dropout** | 0.1 (SS2D 만), ETER 사실상 0 | **0.2** (양쪽 모두) |
| **B. weight_decay** | SS2D 1e-5, ETER 1e-7 | **3e-5** (양쪽 통일) |
| **C. EarlyStopping** | 없음 | val composite 기준 patience=5 |

**근거**
- v4 의 strict 필터 (`rss_shape != (320,320)` 인 파일 360개 스킵) 가 1차 원인 — **분포 자체가 좁음**
- ETER v4 dropout 미적용 (`Transformer(...)` 호출에 dropout 인자 누락) 발견 → wrapper 로 명시적 주입
- 모델은 v4 그대로 둬서 **데이터/regularization 효과만 분리**해 측정
- val 셋 7,270 = U-Net leaderboard 와 **같은 모집단**에서 비교 가능해짐

**SS2D v5 vs ETER v5 의 변수 통제**

| 항목 | SS2D v5 | ETER v5 |
|---|---|---|
| dataloader | dataloader_h5_v5 (공유) | 동일 |
| dropout / wd / aug / EarlyStop | 모두 동일 | 동일 |
| BATCH_SIZE | 4 (DC block 메모리) | 8 |
| **유일한 차이** | SS2D + DC block | GRU h/v (DC 없음) |

**결과 (학습 시 커스텀 SSIM 기준)**
| 모델 | Best val SSIM (custom) | 종료 시점 |
|---|---|---|
| SS2D v5 | 0.7227 | EarlyStop ep 12 |
| ETER v5 | 0.7421 | EarlyStop ep 9 |

> 학습 로그 상으로는 ETER v5 > SS2D v5 처럼 보였으나, **이 결론은 발견 ① 에서 뒤집힌다.**

---

### ★ 발견 ① — 평가 metric 의 결함 (2026-05-04)

**문제**
- `models/hybrid_eternet/u_choh_SSIM.py` (line 24-37) 의 `val_range=None` 분기:
```python
if val_range is None:
    if torch.max(img1) > 128:
        max_val = 255
    else:
        max_val = 1   # ← BUG
    L = max_val - min_val
```
- fastMRI raw RSS 값은 일반적으로 1 미만 → `L=1` 분기로 빠짐 → SSIM 상수 `C1 = (0.01·L)²`,
  `C2 = (0.03·L)²` 가 매우 작아져 픽셀 노이즈에 과민, 절대값이 인위적으로 낮게 나옴

**검증 — fastMRI 표준 skimage SSIM 으로 7,270 슬라이스 full eval**

| 모델 | custom SSIM (학습 로그) | **skimage SSIM (full eval)** |
|---|---|---|
| SS2D v5 | 0.7227 | **0.8584 ± 0.1119** |
| ETER v5 | 0.7421 | **0.8469 ± 0.1048** |
| U-Net (PT) | — | **0.8858 ± 0.1222** |

> **결론**:
> 1. v1 ~ v5 의 모든 SSIM 절대값과 ranking 이 **잘못된 metric 위에서 비교**되고 있었음
> 2. skimage 표준으로 다시 보면 **SS2D v5 > ETER v5** 로 ranking 이 뒤집힘
> 3. U-Net 과의 갭도 0.15 가 아니라 **0.027 (SS2D)** 로 사실상 거의 따라잡은 상태

---

### v6 — Metric 정합화 + Resume (2026-05-04 ~ 05-11 완료)

**배경**
- 발견 ① 에 따라 **val/EarlyStop 기준만 fastMRI 표준 skimage SSIM 으로 교체**
- 학습 loss 는 변수 추가를 피해 그대로 둠 (custom SSIM 기반 backward 신호 유지)
- v5 best ckpt 에서 **resume** 해 무의미한 재학습을 피함

**변경**
| 항목 | v5 | v6 |
|---|---|---|
| Val/EarlyStop SSIM | custom `u_choh_SSIM` | **skimage `structural_similarity`, `data_range = target.max() − target.min()`** |
| EarlyStop patience | 5 | **10** (여유 확보) |
| Val 빈도 | train SSIM best 갱신 시 | **매 5 epoch 강제 + train best 갱신 시** |
| 시작점 | scratch | **v5 best ckpt 에서 resume** (SS2D: epoch_10, ETER: epoch_5) |
| Loss | `L1 + 1·(1 − custom SSIM)` | **변경 없음** |
| 기타 (모델/데이터/aug/optimizer) | — | v5 그대로 |

**근거**
- metric 만 표준으로 바꾸면 EarlyStop 트리거가 정확해져 진짜 best 를 잡을 수 있음
- loss 까지 바꾸면 backward 동역학이 변해 v5 와 비교 불가능 → metric 만 교체
- v5 best 가 일반화 능력이 가장 잘 학습된 시점이므로 resume 으로 학습 시간 절약

**결과 (200ep 풀 학습 완료, 2026-05-10 ~ 05-11)**
| 모델 | Best val SSIM (skimage) | Best val PSNR | 비고 |
|---|---|---|---|
| **SS2D-ViT v6** | **0.8913 ± 0.0974** | **35.96 dB** | U-Net 0.8858 **추월** |
| **ETER-ViT v6** | **0.8862 ± 0.0938** | 34.63 dB | U-Net SSIM 동등 |
| (참고) U-Net (PT) | 0.8858 ± 0.1222 | 34.66 dB | 베이스라인 |

> v5 → v6 단계에서 한 일은 **metric 만 표준으로 교체** + EarlyStop 기준 단순화. 모델·데이터·loss
> 모두 그대로. 그럼에도 SSIM 절대값이 0.8584 → 0.8913 (+0.033) 으로 도약 — v5 의 숫자가 잘못된
> metric 위의 underestimate 였음이 확정됨.

---

### ★ 발견 ② — Visual-Metric Gap (2026-05-20)

**현상**
- v6 의 정량 SSIM 0.89 / PSNR 35.96 dB 는 U-Net 동등 수준이지만, 시각적으로는
  sulci/혈관 같은 fine detail 이 **평균화된 흐릿한 출력**으로 보임.
- 즉 "숫자는 좋은데 그림은 흐릿한" 괴리.

**원인 분석** ([docs/visual_metric_gap_v6.md](visual_metric_gap_v6.md))

| 원인 | 종류 | 영향 |
|---|---|---|
| (1) raw amplitude SSIM 의 배경 부풀림 | **측정 결함** | 슬라이스의 절반 이상이 배경 (~0) 이라 SSIM 평균을 끌어올림. brain-mask 기준이면 0.83–0.87 |
| (2) `linspace` 등간격 시각화 슬라이스가 below-average | **표시 결함** | slice 3230 이 SS2D 0.85, 평균보다 0.04 낮은 어려운 케이스라 모델이 시각적으로 과소평가 |
| (3) **L1 + SSIM loss 의 mean-prediction 편향** | **본질적 모델 결함** | L1 → 픽셀 중앙값, SSIM → local mean/variance. 둘 다 흐릿함을 처벌 못 함 |
| (4) error map colormap 조기 saturation | **표시 결함** | `*0.5` 곱으로 NMSE 0.009 가 시각적으로 큰 에러처럼 보임 |

**진단 도구** — `visualize_diagnostic_v6.py` 로 SS2D SSIM 분포에서 best/median/worst 각 3개 슬라이스
선정 + brain mask SSIM 동시 측정 + zoom-in / percentile windowing 적용.

**확정 사실**
- 모든 모델에서 raw vs masked SSIM 격차 음수 (특히 U-Net **−0.04 ~ −0.44**) → 원인 (1) 정량 확정
- SS2D 가 ETER 보다 masked SSIM **+0.01**, PSNR **+1.3 dB** 로 일관 우위 → SS2D > ETER 결론 견고
- 원인 (3) 은 측정/표시로 환원 불가능 → **loss 함수 자체를 바꿔야 함** ⇒ §5 (Sharpness Recovery)

---

## 5. Sharpness Recovery — v6 이후 실험 (2026-05-20 ~ 05-28)

발견 ② 의 **원인 (3)** (mean-prediction blurring) 을 직접 공격하기 위해 두 단계로 진행.

### 5.1 Tier 1 — 학습 없이 시도 (TTA + 앙상블)

**가설**: 추가 학습 없이 추론 시 정보를 더 쓰면 sharpness 일부 회복 가능.

**구현** (`eval_tta_ensemble.py`)
- **TTA**: 4-way k-space coherent flip 평균 (image flip → FFT → flipped mask + flipped sens)
- **앙상블**: v4 + v6 / SS2D + ETER (cross-arch) 4가지 조합

**결과 (500-슬라이스 부분 평가, 임계 +0.003 SSIM / +0.15 dB)**

| 방법 | ΔSSIM | ΔPSNR | 결론 |
|---|---:|---:|---|
| SS2D v6 + TTA | **−0.0027** | **−0.35 dB** | v6 가 이미 flip aug 로 학습됨 → TTA 가 blur 만 추가 |
| ETER v6 + TTA | +0.0020 | +0.16 dB | 임계 marginal, 풀평가 가치 부족 |
| Ens (v4+v6) | 양음 혼재 | — | 약한 모델이 노이즈 추가 |
| Ens (SS2D+ETER v6) | +0.0023 | −0.20 dB | SSIM↑/PSNR↓ trade-off |

**판정**: 모두 임계 미달. Tier 1 폐기, Tier 2 (재학습) 로 이동. ([docs/tier1_tta_ensemble_negative.md](tier1_tta_ensemble_negative.md))

---

### 5.2 Tier 2 — 직교 가설 3종 fine-tune

**프레임워크**: v6 best ckpt 부터 50ep fine-tune, LR 5e-5 → 5e-7 cosine, EarlyStop patience=5,
3가지 가설을 같은 base 에서 단일 변수만 바꿔 비교 ([docs/tier2_sharpness_plan.md](tier2_sharpness_plan.md)).

| 버전 | 가설 | 변경 | Loss |
|---|---|---|---|
| **v6_1** | edge 직접 처벌 (강) | λ_grad = 10 | L1 + (1−SSIM) + 10·grad_L1 |
| **v6_2** | edge 처벌 완화 | λ_grad = 3 | L1 + (1−SSIM) + 3·grad_L1 |
| **v6_3** | regularization 완화 | dropout 0.2→0.1, WD 3e-5→1e-5 | L1 + (1−SSIM) (gradient loss 없음) |
| **v6_4** | feature-space 처벌 | +VGG perceptual | L1 + (1−SSIM) + 0.1·VGG (대기 중) |

**결과**

| 버전 | 모델 | ΔSSIM | ΔPSNR | ΔL1 | 결론 |
|---|---|---:|---:|---:|---|
| v6_1 | SS2D | (소량 ↑) | **−0.63 dB** | ↑ | over-sharpening 회귀 → 폐기 |
| v6_1 | ETER | (양음 혼재) | **−0.23 dB** | ↑ | 폐기 |
| v6_2 | SS2D | +0.0044 | **−0.47 dB** | ↑ | 동일 trade-off, 폐기 |
| v6_2 | ETER | **−0.0017** | **−0.40 dB** | ↑ | 양쪽 회귀, 명확 폐기 |
| **v6_3** | **SS2D** | **+0.0011** | **+0.09 dB** | **−0.07** | **모든 지표 개선 — 첫 non-degradation** |
| v6_3 | ETER | 진행 중 (ep5: baseline 동등) | — | — | 28일 밤 완료 예상 |

**해석**
1. **gradient L1 직접 처벌은 over-sharpening 부작용이 PSNR/L1 회귀로 항상 나타남** — λ를 1/3 로
   줄여도(v6_2) 부작용이 그대로. edge L1 은 우리 setup 에 부적합한 loss term 임을 확정.
2. **regularization 완화(v6_3) 가 작지만 정직한 개선**. dropout 0.2 와 WD 3e-5 가 v6 에서 살짝
   under-fit 영역에 머물게 했다는 가설을 지지.
3. v6_3 의 마진(+0.0011 SSIM, +0.09 dB) 은 통계적으로는 PSNR 만 유의 (~2.7σ), SSIM 은 marginal
   (~0.9σ). **임상적 sharp 복원과는 거리가 멀지만**, v6 의 mean-prediction blurring 한계 안에서
   안전하게 짜낸 미세 개선.

**남은 카드** — v6_4 (VGG perceptual loss). feature-space 에서 처벌하면 pixel-level over-sharpening
부작용 없이 detail 회복 가능성 있음. ETER v6_3 결과 확인 후 결정 (config 와 VGG helper 는 준비됨,
train script 만 작성하면 launch 가능).

---

---

## 6. 종합 비교 결과

### 6.1 fastMRI 표준 skimage SSIM full evaluation (7,270 slices, R=4 equispaced)

| 모델 | PSNR (dB) | NMSE | **SSIM** | L1 |
|---|---:|---:|---:|---:|
| **U-Net (pretrained, 496M params)** | 34.66 ± 2.95 | 0.00737 | 0.8858 ± 0.1222 | 6×10⁻⁶ |
| SS2D-ViT v5 | 33.39 ± 2.97 | 0.01333 | 0.8584 ± 0.1119 | 8.87 |
| ETER-ViT v5 | 30.85 ± 2.83 | 0.02194 | 0.8469 ± 0.1048 | 11.03 |
| **SS2D-ViT v6** | **35.96** | **0.00810** | **0.8913 ± 0.0974** | 7.37 |
| ETER-ViT v6 | 34.63 | 0.01080 | 0.8862 ± 0.0938 | 8.20 |
| SS2D-ViT v6_1 (λ_grad=10) | 35.33 | — | 0.892 (소상승) | ↑ |
| SS2D-ViT v6_2 (λ_grad=3) | 35.49 | 0.00900 | **0.8957** | 7.59 |
| **SS2D-ViT v6_3** (★ 채택 후보) | **36.05** | **0.00800** | **0.8924 ± —** | **7.30** |

> v6_2 는 SSIM 최고치 (+0.0044) 지만 PSNR/L1/NMSE 가 일관되게 회귀 → over-sharpening 부작용.
> v6_3 는 절대값 차이는 작아도 **모든 지표가 v6 보다 개선된 유일한 fine-tune** 으로 채택.

### 6.2 SSIM 추이 (전 버전, val set)

| 버전 | Val SSIM | 비고 |
|---|---|---|
| v1 (SS2D, 200ep) | 0.599 (custom) | 최초 baseline |
| v2 (SS2D, 132ep) | 0.759 (custom) | patch 16, RefinementBlock, SSIM w=1.0 |
| v3 (SS2D) | ~0.74 (custom) | scheduler 교체 |
| v4 (SS2D, 200ep) | 0.7340 (custom) | A(SS2D 증설)+B(reg)+C(DC) |
| v5 (SS2D) | 0.7227 (custom) / 0.8584 (skimage) | 데이터 확장 + EarlyStop |
| v5 (ETER) | 0.7421 (custom) / 0.8469 (skimage) | SS2D 와 동일 레시피 |
| v6 (SS2D) | **0.8913 (skimage)** | metric 정합화 + resume, U-Net **추월** |
| v6 (ETER) | 0.8862 (skimage) | U-Net SSIM 동등 |
| v6_1/v6_2 (SS2D) | 0.892 ~ 0.8957 / PSNR ↓ | gradient loss 회귀 |
| **v6_3 (SS2D, ★ 최종)** | **0.8924, PSNR 36.05** | regularization 완화, 모든 지표 ↑ |

### 6.3 핵심 인사이트 (발표 5분 요약용)

1. **데이터 풀 확장(v5)이 정성적으로 가장 큰 한 방** — strict 320×320 필터를 푸는 단순한 변경이
   train +67% / val +62% 회수로 이어졌고, 7,270 슬라이스 모집단을 U-Net 평가셋과 동일하게 맞춰
   **공정 비교 자체가 가능**해졌다.
2. **두 번의 평가 발견** —
   - (i) v5 까지의 custom SSIM 버그 (data_range=1 분기): skimage 로 환산 시 SS2D > ETER 로 ranking
     반전 + U-Net 갭이 5× 축소.
   - (ii) v6 의 raw amplitude SSIM 부풀림 (배경 ~50%): brain-mask SSIM 으로는 0.83–0.87 수준. 모델
     성능을 과대 추정한 것은 아니지만, 시각 인상과의 괴리 원인 4가지 중 하나로 확인.
3. **Mamba(SS2D) > Bi-GRU(ETER)** 가 동일 레시피·동일 dataloader 에서 일관되게 관찰됨 — 4-방향
   SSM 의 장거리 의존성 포착 능력이 k-space aliasing 에 효과적.
4. **8 GB GPU 에서 U-Net 추월 달성** — v6_3 가 U-Net 0.8858 대비 SSIM **+0.0066**, PSNR
   **+1.39 dB** 까지 도달. U-Net 이 496M params 인 반면 ViT+SS2D 는 그보다 작은 capacity.
5. **L1 + SSIM supervised loss 의 본질적 한계 확인** — Tier 2 의 v6_1/v6_2 가 모두 over-sharpening
   회귀로 끝남. mean-prediction blurring 은 loss 함수 paradigm 을 바꿔야 (perceptual/adversarial/
   diffusion) 만 깰 수 있다는 가설을 v6_4 (perceptual) 로 검증 예정.

---

## 7. 현재 상태 & 향후 과제

### 7.1 진행 상태 (2026-05-28)

| 항목 | 상태 |
|---|---|
| SS2D-ViT v6_3 | **완료** (best val SSIM 0.8924, PSNR 36.05 dB) — 채택 후보 |
| ETER-ViT v6_3 | 진행 중 (ep5: 0.8862, baseline 동등). 28일 밤 완료 예상 |
| v6_4 (VGG perceptual) | config + VGG helper 준비. train script 미작성, ETER v6_3 결과 보고 launch 결정 |
| 풀평가 (eval_full_compare) | v6_3 best ckpt 로 재실행 예약 |

### 7.2 향후 과제 (우선순위 순)

**단기 (Tier 2 마무리)**
1. ETER v6_3 완료 후 풀평가 → v6_3 채택 확정
2. v6_4 (VGG perceptual) 시도 — feature-space 처벌이 v6_3 와 합산되면 추가 상승 가능성

**중기 (Tier 3, 본 프로젝트 범위 내)**
3. **학습형 sensitivity map** (E2E-VarNet 스타일 CNN) — 현재 ACS-only sens 가 DC block 의 천장
4. **DC iteration 수 증가** (1 → 2~4 cascade) — 8GB 메모리 한계와의 trade-off
5. **ETER+DC 통합** — 현재 DC 는 SS2D 에만 적용. 같은 처방으로 ETER 도 측정
6. **ViT 인코더 self-supervised pretraining** (MAE 등) — random init 한계 추가 완화

**장기 (패러다임 변경, 본 프로젝트 범위 밖)**
7. **Adversarial loss / Diffusion prior** — supervised L1+SSIM 의 mean-prediction 한계를 깨는
   유일한 길. 8GB GPU 와 supervised dataset 만으로는 어려움.

---

## 8. 부록: 문서 인덱스

이 발표 자료는 다음 문서들의 통합본이다.

| 문서 | 다루는 내용 |
|---|---|
| `docs/architecture_ETER_vs_SS2D.md` | 두 아키텍처의 구조 비교 |
| `docs/ss2d_v4_changes.md` | v4 의 A+B+C 동시 적용 |
| `docs/eter_v4_analysis.md` | ETER v4 회귀 분석과 v5 plan |
| `docs/ss2d_v5_changes.md` | v5 데이터 확장 + EarlyStop |
| `docs/eter_v5_changes.md` | ETER v5 동일 레시피 적용 |
| `docs/ss2d_v6_changes.md` | SS2D v6 metric 정합화 + resume |
| `docs/eter_v6_changes.md` | ETER v6 SS2D 레시피 적용 + BS=4 |
| `docs/visual_metric_gap_v6.md` | v6 의 정량/시각 괴리 4원인 + 진단 |
| `docs/ss2d_eter_v6_1_changes.md` | v6_1 gradient loss (λ_grad=10) → over-sharp 회귀 |
| `docs/ss2d_eter_v6_2_changes.md` | v6_2 gradient loss 완화 (λ_grad=3) → 동일 회귀 |
| `docs/tier1_tta_ensemble_negative.md` | TTA/앙상블 모두 임계 미달 |
| `docs/tier2_sharpness_plan.md` | v6_2/v6_3/v6_4 가설 매트릭스 + 결정 트리 |
| `docs/script_version_history.md` | 삭제된 .py 버전 진화 기록 |
| `docs/cleanup_log.md` | 삭제 파일 대장 |
| `PROJECT_SUMMARY.md` | 코드베이스 전체 기술 요약 |
