# 스크립트 버전 이력 (삭제된 .py 파일 요약)

2026-05-20 정리에서 삭제된 `.py` 파일들의 출처 / 역할 / 버전 진화 방향 기록.
세부 알고리즘 변경은 각 버전 `docs/*_changes.md` 참조.

---

## 1) 루트 학습 entry script — `main_train_*.py`

엔코더 (ViT) + 디코더 (GRU=ETER / SS2D=Mamba) + RefinementBlock 의 통합 학습 루프.
모든 버전이 같은 골격을 공유하고, 손실 / 정규화 / 데이터로더 / 모델 클래스만 버전별로 교체.

| 파일 | 날짜 | 버전 | 핵심 차이 | 참조 changelog |
|---|---|---|---|---|
| `main_train.py` | 4/8 | 통합 prototype | argparse 로 모델 종류 선택. 이후 entry 가 ETER/SS2D 로 분리되며 사용 중단 | — |
| `main_train_eter.py` | 4/20 | v3 (ETER) | `choh_Decoder3_ETER_skip_up_tail` 사용. CosineAnnealingLR 정착 시점 | `scheduler_change.md` (삭제됨) |
| `main_train_ss2d.py` | 4/20 | v3 (SS2D) | `choh_Decoder_SS2D_ViT` 사용. SS2D capacity = 작음 | — |
| `main_train_ss2d_v4.py` | 4/22 | v4 (SS2D) | DC block (1-iter soft) + mask/sens 입력 + AMP. dataloader v4 사용 | `ss2d_v4_changes.md` |
| `main_train_eter_v5.py` | 4/30 | v5 (ETER) | dropout 0→0.2, weight_decay 1e-7→3e-5, EarlyStopping(patience=5), flip aug. dataloader v5 사용 | `eter_v5_changes.md` |
| `main_train_ss2d_v5.py` | 4/30 | v5 (SS2D) | 위와 동일 레시피 (공유 dataloader_h5_v5) | `ss2d_v5_changes.md` |
| `main_train_eter_v6.py` | 5/4 | v6 (ETER) | skimage SSIM 평가, val_ssim 단일 EarlyStop, patience 5→10, val 매 5 epoch, BATCH 8→4, v5 epoch_5 resume | `eter_v6_changes.md` |
| `main_train_ss2d_v6.py` | 5/4 | v6 (SS2D) | 위와 동일 평가 / EarlyStop 변경, v5 epoch_10 resume | `ss2d_v6_changes.md` |
| `main_train_eter_v6_resume.py` | 5/15 | v6 (ETER) | v6 학습 중 OOM 복구용 resume entry. v6 동일 hyperparam, ckpt 에서 재개 | `eter_v6_changes.md` §재개 |

**보존 (현재 활성):** `main_train_ss2d_v6_1.py`, `main_train_eter_v6_1.py` — gradient L1 loss 추가, v6 best 에서 50ep fine-tune.

### 진화 방향 요약
1. v3 (4/20) → v4 (4/22, SS2D): **DC block 도입** — under-sampled k-space 와의 일치성 강제.
2. v4 → v5 (4/30): **regularization 강화** — dropout/weight_decay 증설, flip aug, EarlyStop. dataloader 사이즈 필터 완화로 train +67%.
3. v5 → v6 (5/4): **평가 정합성 수정** — custom u_choh_SSIM (val_range 버그) → skimage `structural_similarity`. EarlyStop 기준 composite → val_ssim 단일.
4. v6 → v6_1 (5/20): **edge sharpness** — gradient L1 loss (`λ=10`) 추가, v6 best 에서 fine-tune. mean-prediction blurring 직접 처벌.

---

## 2) 루트 평가 / 시각화 script — `eval*.py`, `visualize*.py`

| 파일 | 날짜 | 버전 | 역할 | 후속 |
|---|---|---|---|---|
| `eval.py` | 4/15 | v3 시그니처 (`model(img, ksp)`) | val 셋 SSIM/PSNR/NMSE/L1 측정 | v4 부터 시그니처 변경으로 `eval_v4.py` 분리 |
| `eval_v4.py` | 4/30 | v4 시그니처 (`model(img, ksp, mask, sens)`) | DC block 입력 지원 | v5/v6 도 같은 시그니처 사용했으나 `eval_full_compare.py` 가 통합 흡수 |
| `eval_unet_pretrained.py` | 4/30 | U-Net 베이스라인 | fastMRI 공식 pretrained U-Net (chans=256) 을 로컬 val 평가. 결과는 `results/eval_unet_pretrained.csv` 에 저장 | — |
| `visualize.py` | 4/15 | v3 | val 전체 순회 + 4지표 + 결과 PNG 저장 | `visualize_compare.py` 가 모델 다중 비교로 확장 |
| `visualize_v4.py` | 4/30 | v4 | DC block 입력 시그니처 지원 | v5/v6 도 호환되었으나 후속 도구로 대체 |
| `vis_v6_preview.py` | 5/7 | v6 학습 진행 중 미리보기 | SS2D v6 vs ETER v5 vs GT 비교 (UNet 제외, GPU 절약) | v6 학습 완료 후 효용 종료 |

**보존 (현재 활성):**
- `eval_full_compare.py` — U-Net / SS2D / ETER 통합 평가, CSV 출력
- `visualize_compare.py` — 모델 4종 비교 PNG
- `visualize_diagnostic_v6.py` — v6 metric/visual 괴리 진단용 (raw vs masked SSIM, percentile windowing 등)

---

## 3) 옛 config — `configs/myConfig_choh_*` (2024년) — **복원됨**

ViT v1 시절 (320×320 정착 전, 384×384 / 통합 prototype) 의 config. 처음 정리에서 삭제했으나 `scripts_legacy/choh_train_*` 가 이들을 import 하는 것을 사후 확인하고 복원. scripts_legacy 보존과 일관성을 맞추는 차원.

| 파일 | 시점 | 메모 |
|---|---|---|
| `myConfig_choh_model3.py` | 2024-04 | ViT-Base (768d, 12L, 12H) + 1280d decoder. main_train.py 와 짝 |
| `myConfig_choh_ViT_ETER_R4regular.py` | 2024-10 | R=4 regular ETER 학습용 (384×384 시절) |
| `myConfig_choh_ViT_ETER_R4regular_v2.py` | 2025-10 | 위 변형 v2 (legacy) |
| `myConfig_choh_ViT_autoencoder_R4regular.py` | 2024-10 | ViT autoencoder pretraining 시도 (사용 안 함) |
| `myConfig_choh_ViT_recon_R4regular.py` | 2024-08 | ViT recon 단독 prototype |
| `myConfig_temp.py` | 2024-08 | 이름 그대로 임시 설정 |

**보존 (현재 활성):**
- `myConfig_choh_SS2D_model.py`, `..._v4.py`, `..._v5.py`, `..._v6.py`, `..._v6_1.py`
- `myConfig_choh_ETER_model.py`, `..._v5.py`, `..._v6.py`, `..._v6_1.py`

---

## 4) 옛 dataloader — `dataloaders/` (대부분 복원됨)

처음 정리에서 11개 삭제했으나 10개 사후 복원 (9개는 tools/smoke_test_320.py + scripts_legacy 의존, 1개는 `myDataloader_temp.py` — 초기 commit `7d4e4e0` 교수님 원본 보존). 최종 삭제 유지는 1개 (`dataloader_h5_v4.py`, 사용자의 v4 작업).

| 파일 | 시점 | 메모 |
|---|---|---|
| `dataloader_h5.py` | 4/13 | v3 dataloader. fastMRI 공식 `UnetDataTransform` 기반, R=4 언더샘플 + RSS GT |
| `dataloader_h5_v4.py` | 4/22 | v4 dataloader. v3 + `mask` 와 `sens` (ACS 저주파 추정) 반환 추가 — DC block 입력용 |
| `myDataloader_fastmri_brain_230425.py` | 2023-10 | 원본 ETER-Net dataloader (R=4 고정, 384×384). 참조 용도, 사용 안 함 |
| `myDataloader_fastmri_brain_240817.py` | 2024-12 | 통합 시도 prototype |
| `myDataloader_fastmri_brain_R4_251012.py` | 2025-10 | R=4 regular 변형 |
| `myDataloader_fastmri_brain_R8_250903.py` | 2025-09 | R=8 시도 |
| `myDataloader_fastmri_brain_R8_251012.py` | 2025-10 | R=8 변형 |
| `myDataloader_fastmri_brain_randomR8_251007.py` | 2025-10 | random R8 mask |
| `myDataloader_fastmri_brain_random_231027.py` | 2023-10 | random mask (원본 ETER) |
| `myDataloader_fastmri_brain_random_250905.py` | 2025-09 | random mask 변형 |
| `myDataloader_temp.py` | 2024-08 | 임시 |

**보존 (현재 활성):**
- `dataloader_h5_v5.py` — v5/v6/v6_1 가 공유. v4 대비 사이즈 필터 완화 (320×320 외에 crop/pad 로 통합), train +67%, val 7270 (U-Net 평가셋 동등)
- `list_brain_train_320.txt`, `list_brain_unseen_*.txt` — 분할 파일 목록
- `R4_idx_part1.npy`, `R4_idx_part2.npy` — R=4 마스크 인덱스

### 진화 방향
- v3 (`dataloader_h5.py`) — fastMRI 표준 전처리, RSS GT, mask 반환 없음
- v4 (`dataloader_h5_v4.py`) — DC block 용 `mask` / `sens` 반환 추가
- v5 (`dataloader_h5_v5.py`) — 사이즈 필터 완화로 데이터셋 +67% (U-Net 평가셋과 정합), 나머지 v4 동일

---

## 5) Legacy 임시 — `scripts_legacy/temp*.py` — **복원됨 (2026-05-20)**

| 파일 | 시점 | 메모 |
|---|---|---|
| `temp.py` | 2024-09 | 이름 그대로 폐기성 작업 결과 |
| `temp2.py` | 2024-08 | 위와 동일 |
| `temp3_train_chohViT.py` | 2024-08 | 학습 prototype 폐기본 |

처음에는 "어디서도 import 안 됨" 으로 삭제했으나, 세 파일 모두 초기 워크스페이스 commit `7d4e4e0` 의 교수님 원본임을 사후 확인하고 복원. scripts_legacy/ 의 나머지 (`choh_train_*`, `myTEST_*`) 와 함께 원본 ETER-Net / 옛 학습 코드 참고용으로 보존.

**루트 `choh_train_ViT_ETER_R4regular_240916py` (확장자 누락, 9K)** 도 같은 이유 (commit `7d4e4e0`) 로 복원. `scripts_legacy/choh_train_ViT_ETER_R4regular_240916.py` 와 내용은 중복이지만 교수님 원본 보존 원칙 적용.
