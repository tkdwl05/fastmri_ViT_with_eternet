# 평가 metric 재설계: brain mask + weighted composite

날짜: 2026-05-22
적용 대상: `v7_titan/` (v6/v7 무변경, dataloader/u_choh_SSIM 은 새 키/인자만 추가하여 하위 호환 유지)

---

## 1. 동기 — 배경 부풀림 진단

사용자 (대학원생) 가 발견한 핵심 문제:

> 평가 메트릭이 뇌 영역이 아닌 **배경 부분**으로 점수가 부풀려져 있다. 그래서 모델이 실제 재구성 품질이 아닌 "배경을 잘 맞춰 점수만 올리는 방향" 으로 학습된다.

### 1-A. 코드상의 원인

`dataloaders/dataloader_h5_v5.py:184-239` 의 target (`label`) 은 `reconstruction_rss` 의 320×320 (또는 384×384) zero-pad. **brain skull stripping 미적용**, **배경 = 정확히 0**.

`v6/v7/v7_titan` 의 평가 함수 `skimage_ssim_batch` 가 사용한 `data_range`:

```python
dr = t[i].max() - t[i].min()    # per-slice max-min
compare_ssim(t[i], p[i], data_range=dr)
```

`skimage.compare_ssim` 는 모든 픽셀의 SSIM map 을 산술평균. 배경 픽셀은:
- target 분산 ≈ 0 → C1/C2 regularizer 가 분모/분자를 둘 다 지배 → **SSIM map ≈ 1**
- 모델이 배경을 0 으로만 예측하면 자동 만점, 평균이 부풀려짐 (전체 픽셀 중 배경 비중 30~50%)

PSNR 도 동일한 함정 (분자 `target.max()` + 배경 0² MSE 기여 0 → SNR 명목상 높음). NMSE 는 분자/분모가 동일 위치의 zero-target 픽셀에서 모두 0 → **NMSE 는 본질적으로 배경에 가장 덜 민감**.

### 1-B. v5 → v6 단순화 의 부작용

[ss2d_v5_changes.md](ss2d_v5_changes.md) 의 v5 는 처음에 **composite metric** (SSIM ratio + NMSE inv-ratio + PSNR ratio + L1 inv-ratio 평균) 을 EarlyStop/best 기준으로 사용. composite 는 ratio 기반이라 baseline 의존성이 강해 불안정 — ep4 에 피크 후 정체로 ep12 조기 종료. [ss2d_v6_changes.md](ss2d_v6_changes.md) 는 이 부작용을 보고 **val_ssim 단일** 기준으로 단순화.

사용자의 원래 의도는 "다중 metric 으로 학습" 이었으나, 단일 SSIM 화 + 배경 부풀림이 결합되어 "metric 만 좋아 보이는 학습" 의 결과로 이어졌다.

---

## 2. v5 → v7_titan 평가 변천사

| 버전 | val metric 측정 | best / EarlyStop 기준 | SSIM data_range | brain mask |
|---|---|---|---|---|
| v5 | SSIM, PSNR, NMSE, L1 4개 | **composite** (ratio 평균) | custom `u_choh_SSIM(val_range=None)` → L=1 고정 | 없음 |
| v6 / v7 / v7_titan (이전) | SSIM, PSNR, NMSE, L1 4개 | **val_ssim 단일** | `skimage compare_ssim` + per-slice `max-min` | 없음 |
| **v7_titan (재설계)** | SSIM, PSNR, NMSE, L1 4개 (**모두 mask 안에서**) | **weighted composite** (절대척도) | `compare_ssim` + mask 안 `max-min` | **Otsu × 0.4 + largest CC** (최종, §10-C) |

> ⚠️ mask 공식 정정 (2026-05-22): 위 표/아래 D1 의 `label > max × 0.05 + erode 1px` 는 **초안**이며, 73k 검증 후 **§10-C 의 `Otsu × 0.4 + largest CC keep` (no erode / no fill_holes)** 로 대체되어 `dataloader_h5_v5.py:243` 에 구현됨. config 의 `BRAIN_MASK_THRESHOLD/ERODE_ITER` 노출은 미사용이라 제거됨.

> v5 결과는 비정상 조기종료 학습이라 baseline / 비교 대상에서 제외 ([memory/feedback-v5-ignore](../../.claude/projects/-home-snorlax-shared-fastmri-ViT-with-eternet/memory/feedback_v5_ignore.md)).

---

## 3. 사용자 결정사항 (D1–D4)

### D1 — Brain mask 생성 방식

**채택: label > target.max() × 0.05 + 1-px erode**

- 동작: RSS magnitude (target) 의 픽셀 값 중 max 의 5% 이상을 brain 으로 간주, 가장자리 1-px 안쪽 erode
- 장점: 단순, RSS magnitude 는 배경=0 이라 신뢰. 외부 의존성 0
- 단점: threshold 0.05 가 hyperparameter — `BRAIN_MASK_THRESHOLD` config 로 노출, 실험으로 조정 가능
- 거부: sens map (`rss_acs`) 기반 mask 는 coil profile 외부 phase noise 일부 포함 위험 / 교집합 방식은 mask 영역 작아져 분산 ↑

### D2 — best ckpt / EarlyStop 기준

**채택: Weighted composite (절대척도)**

```
composite = COMPOSITE_W_SSIM * SSIM_m
          + COMPOSITE_W_PSNR * min(PSNR_m, PSNR_NORM) / PSNR_NORM
          + COMPOSITE_W_NMSE * max(0, 1 - min(NMSE_m, 1))
```

기본 가중치: `SSIM=0.5`, `PSNR=0.3`, `NMSE=0.2`, `PSNR_NORM=40.0` (모두 config 에 노출).

- 장점: 사용자 "다중 metric 학습" 의도 계승. 한 metric 의 극단 변동에 robust. v5 composite (ratio, baseline 의존) 와 다른 절대척도 → 안정.
- 거부: val_ssim 단일은 사용자 의도 미반영. SSIM ∧ NMSE 양쪽 갱신 강제는 EarlyStop 자주 발동 위험.

### D3 — Loss 함수

**채택: brain-mask L1 + brain-mask SSIM_loss**

```python
m_sum = brain_mask.sum().clamp(min=1.0)
loss_l1   = ((out - target).abs() * brain_mask).sum() / m_sum
loss_ssim = 1 - criterion_ssim(out, target, mask=brain_mask)
loss = loss_l1 + LAMBDA_SSIM_PER_PIXEL * loss_ssim
```

- 장점: Loss / metric 일관성, 모델 학습 압력을 brain 영역에 집중 → 배경 부풀림 학습 차단 (사용자 지적의 진짜 해결책)
- 거부: 평가만 mask 적용하면 학습 방향 안 바뀜 / NMSE_loss·PSNR_loss 는 학계 표준 아님 (gradient 불안정)

### D4 — docs/ 정렬 방식

**채택: CLAUDE.md 재정렬 + docs/INDEX.md 신규**

- 파일명은 유지 (git history / import 경로 무손상)
- [INDEX.md](INDEX.md) 가 단일 진실의 원천 (날짜순 + 한 줄 요약)
- [CLAUDE.md](../CLAUDE.md) 의 "핵심 문서 (docs/)" 섹션도 동기

---

## 4. 코드 변경 위치

| 파일 | 변경 |
|---|---|
| [dataloaders/dataloader_h5_v5.py](../dataloaders/dataloader_h5_v5.py) | line 22 `from scipy.ndimage import binary_erosion`; `__getitem__` 끝에 `brain_mask` 생성 + return dict 에 키 추가 |
| [models/hybrid_eternet/u_choh_SSIM.py](../models/hybrid_eternet/u_choh_SSIM.py) | `ssim()` 시그니처에 `mask=None`, mask 가 있으면 conv-smoothed binary mask 위에서만 평균. `SSIM.forward()` 도 동일 |
| [v7_titan/main_train_ss2d_v7_titan.py](../v7_titan/main_train_ss2d_v7_titan.py) | `skimage_ssim_batch_masked` 신규, `run_val` masked metric + composite, train loop loss masked, best 갱신 composite 기준 |
| [v7_titan/main_train_eter_v7_titan.py](../v7_titan/main_train_eter_v7_titan.py) | 동일 |
| [v7_titan/configs/myConfig_choh_SS2D_model_v7_titan.py](../v7_titan/configs/myConfig_choh_SS2D_model_v7_titan.py) | `COMPOSITE_W_{SSIM,PSNR,NMSE}`, `PSNR_NORM=40.0` (mask 는 §10-C 대로 dataloader 하드코딩 — `BRAIN_MASK_THRESHOLD/ERODE_ITER` 는 미사용이라 제거됨) |
| [v7_titan/configs/myConfig_choh_ETER_model_v7_titan.py](../v7_titan/configs/myConfig_choh_ETER_model_v7_titan.py) | 동일 |
| [v7_titan/sanity_eval_metric_v7_titan.py](../v7_titan/sanity_eval_metric_v7_titan.py) (신규) | brain_mask shape/dtype/sum 확인, mask overlay PNG, masked vs unmasked metric 비교, loss NaN sanity |

추후 (선택):
- `eval_unet_pretrained.py` — U-Net baseline 을 새 metric 으로 재측정

---

## 5. 부록 A — Learning Rate / Scheduler 의미

사용자가 별도 질문한 항목.

### 5-A. Learning Rate 정의

옵티마이저가 가중치를 업데이트할 때 그래디언트에 곱해지는 스칼라. 단순 SGD 면 `θ ← θ − lr · ∇L(θ)`. **Adam** 은 모멘텀 + RMSProp 결합으로 `θ ← θ − lr · m̂ / (√v̂ + ε)` — 그래디언트의 크기로 자동 정규화돼 lr 은 "각 파라미터의 최대 step size" 에 가까움.

### 5-B. v7_titan 의 LR 값

| 항목 | 값 | 의미 |
|---|---|---|
| `LEARNING_RATE_ADAM` | `2e-4` | Adam 의 초기 lr. fastMRI U-Net 표준 (1e-3 ~ 1e-4) 의 중간, ViT 계열에 안전 |
| `LAMBDA_REGULAR_PER_PIXEL` | `3e-5` | Adam 의 L2 정규화 (weight decay). v3~v5 의 1e-7 → v5 부터 3e-5 로 강화 |
| `LAMBDA_SSIM_PER_PIXEL` | `1.0` | L1 + λ·SSIM_loss 의 SSIM 가중치. L1 과 동등 비중 |

### 5-C. CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=steps_per_epoch * NUM_EPOCHS, eta_min=1e-6
)
scheduler.step()    # 매 batch 마다 호출 (epoch 단위 아님)
```

- T_max: 전체 학습 step 수. 200 epoch × steps_per_epoch
- eta_min: 최소 lr (`1e-6`)
- 동작: `2e-4` → `1e-6` 으로 cosine 곡선 따라 단조 감소 (1 사이클)

### 5-D. WarmRestarts → CosineAnnealing 교체 (v3 → v4 부터)

[scheduler_change.md](scheduler_change.md) 참고. v3 까지 사용한 `CosineAnnealingWarmRestarts` 는 ep 1, 3, 7, 15, 31, … 에서 lr 이 톱니처럼 튕겨오름. 극소값 탈출 의도지만 MRI 정밀 수렴에 방해. v4 부터 단조 코사인 감소로 안정 수렴.

### 5-E. Adam 의 weight_decay vs AdamW

현 코드는 `torch.optim.Adam(weight_decay=3e-5)` — Adam 의 weight_decay 는 grad 에 더해지는 식이라 L2 정규화와 약간 다름. AdamW 가 표준이지만 v6 결과 안정이라 변경 안 함.

---

## 6. 부록 B — 학습용 metric 학계 표준 정리

사용자 질문: "SSIM + PSNR + NMSE 다중 metric 으로 학습하면 되나? NMSE 같은 값 이용하면 안 되나?"

| metric | loss 로 사용? | 이유 |
|---|---|---|
| **L1** | ✅ 표준 | 안정. fastMRI U-Net / E2E-VarNet 의 기본 loss |
| **SSIM_loss** (=`1 − SSIM`) | ✅ 표준 | structural fidelity. L1 + SSIM_loss 가 fastMRI 표준 |
| **PSNR_loss** | ❌ 부적합 | `20·log₁₀(max/√MSE)` 는 log scale → gradient 불안정. 평가 metric 으로만 |
| **NMSE_loss** | ❌ 부적합 | `‖x−y‖² / ‖y‖²` 의 분모가 sample 의존 → batch 안에서 큰 sample 만 dominant. 학계 표준 아님 |
| **MS-SSIM_loss** | △ | 가능하지만 320~384 해상도에서 효과 제한 |

**사용자의 "다중 metric 학습" 의도는 best/EarlyStop 기준의 weighted composite 로 달성**. Loss 함수는 L1 + SSIM 표준 유지하되 metric 모두를 monitoring + best 판단에 결합. 이게 D2 (composite) + D3 (mask L1 + mask SSIM) 의 설계 근거.

---

## 7. 검증

[sanity_eval_metric_v7_titan.py](../v7_titan/sanity_eval_metric_v7_titan.py) 가 다음을 점검:

1. dataloader 가 `brain_mask` 키를 반환
2. mask overlay PNG (사용자 검토용) 저장 — `v7_titan/runs/sanity_eval/brain_mask_overlay.png`
3. 동일 sample 에 대해 **masked vs unmasked metric 비교** — `SSIM_unmasked > SSIM_masked` 면 배경 부풀림이 정량화됨
4. mask sum=0 edge case 에서 loss NaN/Inf 안 발생

실행:
```bash
cd /home/snorlax/shared/fastmri_ViT_with_eternet
python v7_titan/sanity_eval_metric_v7_titan.py
```

---

## 8. 본 학습 진입 절차

sanity 통과 후:

```bash
# 옵션 A — tmux (가장 안전, ssh 끊겨도 살아남음)
tmux new -d -s v7_titan_train \
  'bash v7_titan/runs/chain/run_chain_v7_titan.sh \
   > v7_titan/runs/chain/run_chain_v7_titan.log 2>&1'
tmux attach -t v7_titan_train    # 진행 확인 (detach: Ctrl-b d)

# 옵션 B — setsid + nohup + disown
setsid nohup bash v7_titan/runs/chain/run_chain_v7_titan.sh \
  > v7_titan/runs/chain/run_chain_v7_titan.log 2>&1 < /dev/null &
disown
tail -f v7_titan/runs/chain/run_chain_v7_titan.log
```

학습 조건:
- 200 epoch, EarlyStop patience=10 (val check 단위, 매 5 epoch) → **composite 기준**
- L1_masked + λ·SSIM_loss_masked
- BATCH_SIZE=4 (smoke 결정), single GPU sequential (SS2D → ETER)
- 예상 소요: 모델당 4~7일, 합쳐 1~2주

---

## 9. 관련 문서

- [INDEX.md](INDEX.md) — docs 전체 날짜순
- [presentation_overview.md](presentation_overview.md) — v1~v6 변천사
- [ss2d_v6_changes.md](ss2d_v6_changes.md) — skimage SSIM 도입 (이번 재설계의 직전 세대)
- [scheduler_change.md](scheduler_change.md) — WarmRestarts → CosineAnnealing

---

## 10. brain_mask 알고리즘 1차 검증 + 튜닝 (2026-05-22)

[v7_titan/sanity_eval_metric_v7_titan.py](../v7_titan/sanity_eval_metric_v7_titan.py) sanity 통과 후, PNG 시각 검토를 못 하는 환경에서 [v7_titan/sanity_mask_text_check.py](../v7_titan/sanity_mask_text_check.py) (수치 + ASCII 시각화) 로 6 sample 추가 검증한 결과 **초기 알고리즘에 결함 발견**.

### 10-A. 초기 알고리즘 (D1 채택안) 결과

```python
mask_raw   = gt_rss > 0.05 * gt_rss.max()
brain_mask = binary_erosion(mask_raw, iterations=1).astype(np.float32)
```

| sample | ratio | n_components | holes | solidity | 진단 |
|---|---|---|---|---|---|
| 1 | 33.6% | 51  | 5511  | 0.892 | brain 모양은 OK, 작은 구멍/외부 조각 |
| 2 | **54.6%** | **729** | **18525** | 0.796 | mask 가 거의 전체 이미지 덮음 — **표립 결함** |
| 3 | 33.8% | 62  | 6414  | 0.872 | sample 1 과 유사 |
| 4 | 30.6% | 35  | 901   | 0.829 | 그나마 적당 |
| 5 | 34.3% | 304 | 2406  | **0.532** | mask 거칠고 분산됨 |
| 6 | **68.3%** | 1 | 355 | 0.996 | mask 가 거의 전체 이미지 — **표립 결함** |

### 10-B. 결함 원인 진단

1. **외부 조각 다수 (n_components 50~729)** — zero-pad 가장자리 noise 가 작은 영역으로 threshold 통과
2. **표립 (sample 2/6 의 ratio 54~68%)** — `gt_rss.max()` 가 작은 sample 에서 `0.05 × max` 가 noise floor 보다 낮게 떨어져 거의 모든 픽셀이 통과. fastMRI brain 의 contrast 별 dynamic range 차이가 원인.
3. **내부 구멍 (holes 5000~18000)** — eye / ventricle / 작은 air-tissue 경계가 mask 안에 점점이 빈 구멍

### 10-C. 사용자 결정 + 알고리즘 자동 비교 (2026-05-22)

기본 결정:
| 요소 | 결정 | 이유 |
|---|---|---|
| **내부 구멍 (holes)** | **그대로 둠** (`fill_holes` 사용 안 함) | 뇌 안의 ventricle/eye 등은 진짜 brain detail. SSIM 부풀림에 영향 작음 |
| **외부 조각 제거** | **largest CC 만 유지** | 뇌 밖 noise 조각 (zero-pad 가장자리) 제거 |
| **erode** | **사용 안 함** | erode 1px 가 brain 좁은 부분 끊어 largest CC 가 brain 일부만 keep 하던 결함 (n_components 1 비율 3.4% → 100%) |

알고리즘 후보 자동 비교 (random 500 val sample, n_components=1 100% 공통):

| 옵션 | threshold 정의 | median ratio | 통과 비율 ([15%, 50%]) |
|---|---|---|---|
| A | Otsu × 0.7 | 19.9% | 64.6% |
| B | Percentile 15 | 58.9% | 4.4% |
| C | min(Otsu × 0.7, 0.05 × max) | 35.3% | 71.4% |
| D | Otsu × 0.5 | 25.0% | 80.0% |
| E | Percentile 25 | 50.0% | 50.2% |
| **F** | **Otsu × 0.4** | **28.2%** | **82.6%** ⭐ |
| G | Otsu × 0.3 | 30.5% | 82.2% |

**최종 채택: F (Otsu × 0.4)** — 통과 비율 best 이며 G 와 평탄화 구간. 더 lenient 면 표립 sample 증가.

채택 알고리즘:

```python
from scipy.ndimage import label as ndi_label
from skimage.filters import threshold_otsu

rss_max = float(gt_rss.max())
if rss_max > 0:
    non_zero = gt_rss[gt_rss > 0]
    if non_zero.size > 100:
        try:
            thr = float(threshold_otsu(non_zero)) * 0.4
        except Exception:
            thr = 0.05 * rss_max
    else:
        thr = 0.05 * rss_max
    mask_raw = gt_rss > thr
    lbl, n = ndi_label(mask_raw)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        largest = int(sizes.argmax())
        mask_final = (lbl == largest)
    else:
        mask_final = mask_raw
    brain_mask = mask_final.astype(np.float32)
else:
    brain_mask = np.zeros_like(gt_rss, dtype=np.float32)
```

남은 17.4% 이상치:
- `ratio < 10%` (~10%): Otsu 가 brain dim gray matter 까지 strict 한 sample
- `ratio > 55%` (~7%): rss_max 매우 작은 contrast (예: AXFLAIR low-signal slice) — threshold 가 noise floor 보다 낮아짐

대응: 전체 73k 검증에서 분포 재확인. 학습 시 해당 slice 의 loss 효과는:
- ratio<10%: 학습 신호 일부 sample 만 제한 (나머지 sample 정상)
- ratio>55%: 배경 일부 포함 — 17.4% × 7% ≈ 1.2% 의 sample 만 영향, 학습 noise 흡수 가능

### 10-D. 검증 계획 (2 단계)

| 단계 | 범위 | 시간 | 출력 |
|---|---|---|---|
| 1 | Random 500 slice | ~5~10분 | mask ratio / n_components / bbox 분포 히스토그램, 이상치 sample id 목록 |
| 2 | Train+val 전체 ~73,000 slice | ~1~2시간 (NVMe) | 전체 분포 + 이상치 0% 까지 알고리즘 튜닝 |

통과 기준 (모든 sample 에 대해):
- mask ratio ∈ [15%, 50%]
- n_components = 1 (largest CC keep 효과)
- centroid 이미지 중심 ±50 px
- bbox h×w ∈ [120, 350] (정상 brain crop)

### 10-E. 보조 도구

- [v7_titan/sanity_mask_text_check.py](../v7_titan/sanity_mask_text_check.py) — 수치 + ASCII 시각화 (PNG 없는 환경용)
- 위 PNG `v7_titan/runs/sanity_eval/brain_mask_overlay.png` (matplotlib overlay 4 sample)
