# 버전 변천사 — V4 → V6 → V7 (SS2D-ViT / ETER-ViT)

> 흩어진 버전별 `*_changes.md` 를 **한 곳에서 비교**하기 위한 통합 문서. 작성 2026-06-16.
> 모든 수치는 실제 **config**(`configs/`, `v7/configs/`, `v7_titan/configs/`) · **모델 코드** · **학습 로그**에서 확인했다.
> 각 절 끝에 원본 상세 문서를 링크한다(중복 서술 대신 포인터).

---

## 0. 한눈에 보는 흐름

3단계로 읽으면 된다 — **V4(기반 구축) → V6(평가 정합 + U-Net 추월) → V7(스케일업 + 공정 비교)**.

- **V4** : SS2D capacity 증설 + **DC block(데이터 컨시스턴시) 도입** + 정규화 강화. 하지만 custom SSIM metric 버그로 점수가 과소평가되고, ETER 는 v3 대비 회귀.
- **V5** : 데이터 +67% 복원 · dropout/weight_decay 상향 · flip aug · EarlyStop 도입. **레시피는 V6 로 계승**되나, composite EarlyStop 오발동으로 조기종료된 **비정상 학습 → 결과는 baseline/비교에서 제외**([[feedback_v5_ignore]]).
- **V6** : **custom SSIM `val_range` 버그를 skimage 로 교체**(핵심 발견 ①). 같은 모델인데 점수가 ~0.72 → ~0.89 로 드러나며 **SS2D 가 U-Net 을 처음 추월**. 이후 `v6_1~v6_4` 로 blurring(핵심 발견 ②) 완화 실험.
- **V7** : 두 갈래. `v7`(320, ViT-S, BS↑, GRU 복원 — **환경 이주용 중간 baseline**)와 `v7_titan`(**384 · ViT-Base**, SS2D capacity↑, **ETER tail = U-Net head**, **brain-mask composite 평가**, true-resume 인프라). v7_titan 에서 SS2D·ETER 가 **dead-heat** 로 수렴.

> ⚠ **v6(0.89)와 v7_titan(0.91)은 직접 비교 불가** — 해상도(320 vs 384)·평가지표(raw vs brain-masked composite)가 다르다. §5 참조.

---

## 1. 공통 아키텍처 (불변 골격)

```
입력: aliased image (B,32,H,W) + masked k-space (B,32,H,W)
        │
   ViT 인코더 ──→ ViT 디코더 ┐
        │                    ├─ cat → tail → 출력 (B,1,H,W)
   SS2D | GRU ───────────────┘
```

- **ViT 인코더/디코더**: 두 모델 공통. v4~v6 는 **ViT-Small**(dim 384·layer 6·head 6·mlp 1536), v7_titan 만 **ViT-Base**(768·12·12·3072). 디코더 dim 512·depth 6·head 8·mlp 2048 은 전 버전 불변.
- **SS2D 계열**: sequence model = Mamba 기반 SS2D. tail = `RefinementBlock`(→2ch real/imag) **+ `DCBlock`**(v4 도입, `models/mamba_eternet/u_choh_model_SS2D_ViT_v4.py:131/287`). DC 는 1-iter soft data consistency: `k_dc = k_pred + mask·α·(k_meas−k_pred)`.
- **ETER 계열**: sequence model = 양방향(H/V) GRU. tail = `RefinementBlock`(→1ch) — **DC block 없음**. v7_titan 에서 tail 을 `UNet_choh_skip(depth=3, wf=6)` 로 교체(`u_choh_model_ETER_ViT_v7_titan.py:51`, 원본 ETER-Net 의 U-Net DFU 복원).
- 따라서 **SS2D vs ETER 의 차이 = (Mamba SS2D + DC block) vs (양방향 GRU) + tail 종류**.

상세: [[architecture_ETER_vs_SS2D]]

---

## 2. 하이퍼파라미터 비교표 (config ground-truth)

범례: **굵게** = 직전 버전 대비 변경. `–` = 해당 없음. WD = `LAMBDA_REGULAR_PER_PIXEL`(optimizer weight decay). v3 = base config(`myConfig_choh_*_model.py`), 참고용.

### 2.1 SS2D-ViT

| 항목 | v3(base) | **v4** | v5 | **v6** | v6_1 | v6_2 | v6_3 | v6_4 | **v7**(320) | **v7_titan**(384) |
|---|---|---|---|---|---|---|---|---|---|---|
| 해상도 | 320 | 320 | 320 | 320 | 320 | 320 | 320 | 320 | 320 | **384** |
| ViT 인코더 | S | S | S | S | S | S | S | S | S | **Base** |
| SS2D d_inner | 32 | **64** | 64 | 64 | 64 | 64 | 64 | 64 | 64 | **128** |
| SS2D d_state | 8 | **16** | 16 | 16 | 16 | 16 | 16 | 16 | 16 | **32** |
| SS2D out_ch | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | **32** |
| DC block | – | **도입** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BATCH_SIZE | 8 | **4** | 4 | 4 | 4 | 4 | 4 | 4 | **16** | **6** |
| epochs | 200 | 200 | 200 | 200 | **50** | 50 | 50 | 50 | 200 | 50 |
| LR(Adam) | 2e-4 | 2e-4 | 2e-4 | 2e-4 | **5e-5** | 5e-5 | 5e-5 | 5e-5 | 2e-4 | 2e-4 |
| WD | 1e-7 | **1e-5** | **3e-5** | 3e-5 | 3e-5 | 3e-5 | **1e-5** | 3e-5 | 3e-5 | 3e-5 |
| dropout | 0 | **0.1** | **0.2** | 0.2 | 0.2 | 0.2 | **0.1** | 0.2 | 0.2 | 0.2 |
| λ_grad | – | – | – | – | **10.0** | **3.0** | **0.0** | – | – | – |
| λ_perc | – | – | – | – | – | – | – | **0.1** | – | – |
| EarlyStop patience | – | – | 5 | **10** | 5 | 5 | 5 | 5 | 10 | **50** |
| VAL_EVERY | 매 ep | 매 ep | 매 ep | **5** | 5 | 5 | 5 | 5 | 5 | **2** |
| 시작점 | scratch | scratch | scratch | **v5 ep10** | **v6 best** | v6 best | v6 best | v6 best | **scratch** | scratch |

### 2.2 ETER-ViT

| 항목 | **v4**(base) | v5 | **v6** | v6_1 | v6_2 | v6_3 | v6_4 | **v7**(320) | **v7_titan**(384) |
|---|---|---|---|---|---|---|---|---|---|
| 해상도 | 320 | 320 | 320 | 320 | 320 | 320 | 320 | 320 | **384** |
| ViT 인코더 | S | S | S | S | S | S | S | S | **Base** |
| GRU hidden(H/V) | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | **10/10** | 10/10 |
| tail | Refine | Refine | Refine | Refine | Refine | Refine | Refine | Refine | **U-Net(d3,wf6)** |
| BATCH_SIZE | 8 | 8 | **4** | 4 | 4 | 4 | 4 | **16** | **8** |
| epochs | 200 | 200 | 200 | **50** | 50 | 50 | 50 | 200 | 50 |
| LR(Adam) | 2e-4 | 2e-4 | 2e-4 | **5e-5** | 5e-5 | 5e-5 | 5e-5 | 2e-4 | 2e-4 |
| WD | 1e-7 | **3e-5** | 3e-5 | 3e-5 | 3e-5 | **1e-5** | 3e-5 | 3e-5 | 3e-5 |
| dropout | 0 | **0.2** | 0.2 | 0.2 | 0.2 | **0.1** | 0.2 | 0.2 | 0.2 |
| λ_grad | – | – | – | **10.0** | **3.0** | **0.0** | – | – | – |
| λ_perc | – | – | – | – | – | – | **0.1** | – | – |
| EarlyStop patience | – | 5 | **10** | 5 | 5 | 5 | 5 | 10 | 10 |
| VAL_EVERY | 매 ep | 매 ep | **5** | 5 | 5 | 5 | 5 | 5 | 5 |
| 시작점 | scratch | scratch | **v5 ep5** | **v6 best** | v6 best | v6 best | v6 best | **scratch** | scratch |

> 두 모델의 동일-axis 차이가 좁혀지는 지점: v7/v7_titan 에서 ETER GRU hidden 이 6→**10** 으로 복원되고(원본 ETER-Net spec), v7_titan 에서 ViT-Base + tail U-Net 으로 capacity 가 SS2D 와 대등해진다.

---

## 2.5 파라미터 수 & 학습 환경

### 파라미터 수 (체크포인트 state_dict `sum(numel)` 실측)

| 모델 | tier | 파라미터 |
|---|---|---|
| **SS2D** v4/v5/v6/v6_x/v7 | 320 · ViT-Small · SS2D 64/16/20 | **39.4M** |
| **SS2D** v7_titan | 384 · ViT-Base · SS2D 128/32/32 | **121.1M** |
| **ETER** v4/v5/v6/v6_x | 320 · ViT-Small · GRU 6 · RefinementBlock | **~482M** |
| **ETER** v7 | 320 · ViT-Small · GRU 10 · RefinementBlock | **481.8M** |
| **ETER** v7_titan | 384 · ViT-Base · GRU 10 · U-Net head(d3,wf6) | **759.7M** |

- 측정 출처: `logs/{SS2D,ETER}_ViT_R4_brain{320_v7,384_v7_titan}/*_best.pt`. 320·ViT-S 행은 v7(320) ckpt 기준 — v4~v6 는 동일 architecture tier(ETER GRU hidden 6↔10 차이는 <1M 로 무시 가능, SS2D 는 완전 동일).
- **핵심 (효율)**: ETER 가 SS2D 보다 **6~12× 큰 파라미터**다(인코더/디코더는 ~40M 로 동급이나, ETER 의 dense up-projection decoder tail 이 ~440M+ 를 차지해 지배적). 그럼에도 성능은 v6 에서 **SS2D 우위**, v7_titan 에서 **near-tie** → **SS2D(Mamba) 가 훨씬 파라미터-효율적**임을 시사.

### 학습 환경

| tier | 하드웨어 | 제약 / 특이사항 |
|---|---|---|
| v4~v6_x (320) | **RTX 5060Ti 8GB** 단일 · conda `mri_env` | 8GB 제약 → BATCH_SIZE 작게(SS2D 4 / ETER 8→4), SS2D forward **gradient checkpointing**, GRU hidden 축소 |
| v7 (320) | **TITAN RTX 24GB ×2** (128GB DRAM · 16TB HDD) 로 이주 | 8GB 제약 해소 → BS 16, GRU 10 복원. 단일 24GB 또는 DDP×2 실험 |
| v7_titan (384) | **TITAN RTX 24GB 단일**(GPU0; machine 은 ×2 이나 공정 비교 위해 GPU1 idle) | scratch · true-resume(full-state `*_last.pt`) · auto-restart supervisor · BS SS2D 6 / ETER 8 · 50ep · VRAM peak ≤ ~22GB · SS2D ~5.6h/epoch(mamba_ssm CUDA 커널) · wandb |

- 데이터: NVMe overlay(`fastMRI_data → fastmri_data_nvme`), v7_titan 기준 train 65,028 / val 7,334 slice (384×384, R4).
- 공통 학습: PyTorch + AMP(autocast/GradScaler), Adam optimizer, CosineAnnealingLR scheduler.

---

## 3. 전환별 상세 (무엇 · 왜 · 결과)

### 3.1 (참고) v3 → v4 — capacity + DC block + 정규화 도입
- **무엇**: SS2D `d_inner 32→64`·`d_state 8→16`, **DCBlock 신규**, WD `1e-7→1e-5`, dropout `0→0.1`, BATCH_SIZE `8→4`(DC backward 메모리 확보).
- **왜**: 표현력·장거리 의존성 확대 + 측정 일관성을 위한 soft DC.
- **결과**: SS2D best ~**0.734**, ETER ~**0.732**(custom SSIM, raw) — ETER 는 v3 대비 회귀(피크 후 단조 감소). 상세 [[ss2d_v4_changes]], [[eter_v4_analysis]].

### 3.2 v4 → v5 — 데이터 복원 + 정규화 강화 + EarlyStop  ⚠ 결과 제외
- **무엇**: dataloader `v4→v5` (320 strict 필터 → image-domain crop/pad ⇒ **train +67%, val 7270 = U-Net 평가셋**), dropout `→0.2`, WD `→3e-5`(ETER 는 1e-7→3e-5), H/V flip aug, **EarlyStop patience=5**.
- **왜**: v4 의 데이터 협소·과적합·brake 부재 해소.
- **결과**: **비정상 조기종료**(composite EarlyStop 오발동) → baseline 제외. 단 레시피는 V6 로 계승. 상세 [[ss2d_v5_changes]], [[eter_v5_changes]].

### 3.3 v5 → v6 — 평가지표 버그 수정 (★ 핵심 발견 ①)
- **무엇**: val SSIM 을 custom `u_choh_SSIM`(val_range 버그) → **skimage `structural_similarity(data_range=max−min)`**. EarlyStop/best 기준을 composite → **val_ssim 단일**, patience `5→10`, VAL_EVERY `매 ep→5`. v5 best ckpt 에서 resume(SS2D ep10 / ETER ep5). 모델·데이터·정규화는 v5 그대로.
- **왜**: composite 오발동 + custom SSIM 과소평가(아래 §4①).
- **결과(raw, skimage, best-val)**: SS2D **~0.890** / ETER **~0.886** / U-Net **~0.886** → **SS2D 가 U-Net 첫 추월**. (ETER v6 는 첫 forward OOM 으로 BS 8→4 강하.) 상세 [[ss2d_v6_changes]], [[eter_v6_changes]].

### 3.4 v6 → v6_1 / v6_2 / v6_3 / v6_4 — sharpness 회복 실험
v6 best 에서 50ep fine-tune (LR 5e-5). blurring(§4②) 을 세 직교 처방으로 공격:
- **v6_1** (λ_grad=10): finite-diff gradient L1 추가 → **over-sharpening**, PSNR ↓(SS2D −0.63dB), L1/NMSE 후퇴.
- **v6_2** (λ_grad=3, v6 에서 재시작): −70% 완화에도 trade-off 동일 → **폐기**.
- **v6_3** (loss 원복, **dropout 0.1 + WD 1e-5**): 정규화만 완화. **SS2D 전 지표 v6 대비 개선**(SSIM 0.8913→**0.8924**, PSNR 35.96→36.05, L1 7.37→7.30) — Tier 2 의 유일한 non-degradation.
- **v6_4** (λ_perc=0.1, VGG perceptual): config 만 준비, 수동 launch 대기.
- 상세 [[visual_metric_gap_v6]], [[ss2d_eter_v6_1_changes]], [[ss2d_eter_v6_2_changes]], [[tier2_sharpness_plan]].

### 3.5 v6 → v7 (320) — 환경 이주용 중간 baseline
- **무엇**: 8GB → TITAN 24GB 이주. 아키텍처는 v6 동일(320·ViT-S·SS2D 64/16/20)이되 **BATCH_SIZE 4→16**, **ETER GRU 6→10 복원**, **scratch**(resume 없음).
- **왜**: 새 하드웨어에서 BS·GRU capacity 를 키운 공정 baseline 확보.
- **결과**: 중간 트랙(자세한 결과는 v7_titan 으로 흡수). `v7/` 폴더에 train/eval/viz 도구 존재.

### 3.6 v7 → v7_titan (384) — 스케일업 + 평가 재설계 (★ 핵심 발견 ②) + 인프라
- **아키텍처**: 해상도 **320→384**, ViT **Small→Base**(768·12·12·3072), SS2D **64/16/20 → 128/32/32**, **ETER tail RefinementBlock → U-Net(depth3,wf6)**.
- **평가 재설계**: raw RSS → **brain-mask**(Otsu×0.4 + largest CC, `dataloaders/dataloader_h5_v5.py:243`), loss = **masked L1 + masked (1−SSIM)**, best/EarlyStop = **weighted composite** `0.5·SSIM_m + 0.3·min(PSNR_m,40)/40 + 0.2·max(0,1−NMSE_m)`.
- **인프라**: scratch(DDP 폐기·격리), **true full-state resume**(`*_last.pt`: epoch/optimizer/scheduler/scaler/RNG, LR 연속 unit-test PASS), 자동 재기동 supervisor. SS2D 만 VAL_EVERY 5→2·patience 10→50.
- **결과(masked, 로그 검증, ep50 완주)**: SS2D **composite 0.9127 / SSIM_m 0.9083 / PSNR 34.60 / NMSE 0.0048 / L1 9.298**, ETER **0.9127 / 0.9084 / 34.59 / 0.0044 / 9.518** → **dead-heat**(composite 동일, SSIM_m −0.0001, **L1 은 SS2D 우위**, NMSE 는 ETER 우위). 상세 [[ss2d_v7_titan_changes]], [[eval_metric_redesign]], [[summary_2026-06-11]].

---

## 4. 두 번의 핵심 발견

### ① custom SSIM `val_range` 버그 (v5→v6 에서 발견)
custom `u_choh_SSIM` 가 `val_range=None` 일 때 `L=1` 고정으로 SSIM 을 계산 → fastMRI raw 값이 1 미만이라 **점수가 구조적으로 과소평가**(같은 모델이 custom ~0.72 vs skimage ~0.86). skimage `data_range=per-slice(max−min)` 로 교체하며 V6 의 "성능 점프" 가 사실은 **측정 정합** 임이 드러남. → [[ss2d_v6_changes]].

### ② visual-metric gap (v6 에서 발견 → v7_titan 재설계 동기)
SSIM 0.89 인데 시각적으로 흐릿한 괴리. 원인 4가지: (a) **raw-amplitude SSIM 이 배경(≈50%)으로 부풀림**, (b) 비교 슬라이스 below-average 편향, (c) **L1+SSIM 의 mean-prediction blurring**, (d) 에러맵 colormap 조기 saturation. → v7_titan 의 **brain-mask + masked loss + composite metric** 으로 직접 대응. → [[visual_metric_gap_v6]], [[eval_metric_redesign]].

---

## 5. 버전별 헤드라인 결과 + 직접비교 주의

| 버전 | 해상도 | 지표 종류 | SS2D | ETER | 비고 |
|---|---|---|---|---|---|
| v4 | 320 | custom SSIM (raw) | 0.734 | 0.732 | DC 도입; ETER 회귀 |
| v5 | 320 | custom (raw) | *제외* | *제외* | 조기종료 비정상 |
| **v6** | 320 | skimage SSIM (raw) | **~0.890** | ~0.886 | SS2D 첫 U-Net(~0.886) 추월 |
| v6_1 | 320 | raw | ~0.895 (PSNR↓) | ~0.887 (PSNR↓) | over-sharpening → 폐기 |
| v6_3 | 320 | raw | **0.8924** | (별도) | 전 지표 개선 (유일 성공) |
| **v7_titan** | **384** | **masked composite** | **0.9127 / SSIM_m 0.9083** | **0.9127 / 0.9084** | **dead-heat** (로그 검증) |

> ⚠ **raw(v6, 320) ↔ masked(v7_titan, 384) 직접 비교 금지**. 두 트랙은 평행하며, 의미 있는 비교는 **같은 트랙 안에서 SS2D vs ETER** 다. v6 트랙: SS2D > ETER(raw). v7_titan 트랙: **near-tie**(masked) — v6 의 "SS2D clean win" 이 384·masked 에서는 재현되지 않음.
> (v6 계열 정확 수치는 best-val 기준 — 측정 기반(val-subset vs full-eval)에 따라 소수 셋째 자리가 다를 수 있다. full-eval 은 `results/eval/eval_full_*`.)

---

## 6. v7 vs v7_titan — 혼동 주의

| | **v7** (`v7/`) | **v7_titan** (`v7_titan/`) |
|---|---|---|
| 해상도 | 320 | **384** |
| ViT 인코더 | Small | **Base** |
| SS2D capacity | 64/16/20 | **128/32/32** |
| ETER tail | RefinementBlock | **U-Net(d3,wf6)** |
| 평가지표 | 표준 (raw) | **brain-mask composite** |
| 성격 | 환경 이주용 중간 baseline | **본 트랙 (현 최신)** |

헤드라인 숫자(0.9127 등)는 전부 **v7_titan**. "V7" 이라고만 하면 두 갈래가 섞이니 문서·발표에서 구분할 것.

---

## 7. 평가/비교 정책 — 비교 기준 해상도 = 384 (v7_titan 트랙)  [2026-06-17 결정]

**320(v4~v6, v7) ↔ 384(v7_titan) 는 직접 비교 불가.** 320 은 8GB GPU 제약으로 줄인 타협이며,
다음 때문에 트랙 간 "통일 평가"가 본질적으로 안 된다:
- **해상도 고정**: ViT patch/positional-embedding 이 해상도에 묶여 320 모델로 384 입력 불가 (각자 native 해상도로만 추론).
- **GT 자체가 다름**: image-domain crop 이 320 vs 384 → **FOV(시야)가 다른 영상**이라 "같은 정답"이 아님.
- **검증셋 변경**: v4 4,492 → v5+ 7,270 슬라이스(size-filter 완화) → 옛 절대값끼리도 모집단이 다름.

**방침**:
- **canonical 비교 = 384 (`v7_titan` 이후)** — 교수님 원본 ETER-Net 해상도와 일치. 신규 모델은 384 로 만들어
  `v7_titan` 과 **동일 dataloader·GT·brain-mask·지표**로 한 표 비교.
- **320 (`v4`~`v7`) = "복원(restoration) 단계" 히스토리 참고용** — 절대 SSIM/PSNR 을 384 와 직접 비교하지 않음.
- 현재 통일 baseline = `visualize_v7_titan_compare.py` (SS2D/ETER/U-Net 4-way, 동일 384 파이프라인).
  평가영역(배경 포함/제외) 민감도 = `visualize_eval_modes_compare.py` (full/narrow/wide).
- ⚠ **"단일 U-Net SSIM" 은 없다** — 평가 프로토콜(지표·해상도·검증셋·입력경로)마다 다름:
  v4 **0.8865**(val 4,492) vs v6 **0.8858**(val 7,270) 은 검증셋 차이, v7_titan **masked ~0.92** 는 metric·해상도·입력경로가 모두 달라 별개.

---

## 8. 원본 문서 인덱스 (교차링크)

- 아키텍처: [[architecture_ETER_vs_SS2D]]
- v4: [[ss2d_v4_changes]] · [[eter_v4_analysis]]
- v5: [[ss2d_v5_changes]] · [[eter_v5_changes]] · [[feedback_v5_ignore]]
- v6: [[ss2d_v6_changes]] · [[eter_v6_changes]]
- v6 sharpness: [[visual_metric_gap_v6]] · [[ss2d_eter_v6_1_changes]] · [[ss2d_eter_v6_2_changes]] · [[tier2_sharpness_plan]]
- v7_titan: [[ss2d_v7_titan_changes]] · [[eval_metric_redesign]]
- 마스터 요약: [[summary_2026-06-11]] · 전체 인덱스 [[INDEX]]
- 스크립트 진화/삭제 이력: [[script_version_history]] · [[cleanup_log]]
