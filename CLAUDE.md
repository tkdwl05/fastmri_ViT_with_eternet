# ViT-based MRI Reconstruction

## 프로젝트 개요

fastMRI brain AXFLAIR multicoil 데이터에 대한 MRI 재구성 모델 연구.
ViT 인코더 + 시퀀스 모델 디코더(GRU 또는 SS2D) 구조를 사용한다.

## 핵심 문서 (docs/)

작업 히스토리와 설계 판단의 근거를 기록한 문서들:

- **[docs/presentation_overview.md](docs/presentation_overview.md)** — **발표용 통합본**. v1 → v6 → v6_3 까지의 전체 흐름, 두 차례 핵심 발견 (① custom SSIM metric 버그, ② visual-metric gap), Tier 1 (TTA/앙상블) 와 Tier 2 (v6_1/v6_2/v6_3 fine-tune) 결과 정리. **최종 갱신 2026-05-28** — v6_3 SS2D best val SSIM 0.8924 / PSNR 36.05 dB (U-Net 추월) 까지 반영, ETER v6_3 진행 중. 발표 5분 요약은 §6.3, 진행 상태와 향후 과제는 §7.
- **[docs/architecture_ETER_vs_SS2D.md](docs/architecture_ETER_vs_SS2D.md)** — ETER-ViT(GRU)와 SS2D-ViT(Mamba) 아키텍처 상세 비교. 공통 파이프라인, 인코더/디코더 구조, 설정값, 학습 조건을 정리.
- **[docs/ss2d_v4_changes.md](docs/ss2d_v4_changes.md)** — SS2D v4에서 A(SS2D capacity 증설) + B(weight_decay/dropout) + C(1-iter soft Data Consistency block) 세 축을 동시 적용한 내역. `_v4` 접미사 신규 파일 5개(config/dataloader/model/train/chain), DC block 파이프라인, FFT AMP 처리, 체인 예약. §8: 첫 batch OOM 사후 수정(SS2D forward gradient checkpointing).
- **[docs/eter_v4_analysis.md](docs/eter_v4_analysis.md)** — ETER v4 200ep 결과 분석. v3(0.7475) 대비 v4 best val SSIM 0.7320 회귀, ep 30~40에 피크 후 단조 감소. 회귀 원인 가설(WarmRestarts 부재 / capacity ceiling / EarlyStopping 부재) 및 v5 계획(EarlyStop, weight_decay↑, dropout↑).
- **[docs/ss2d_v5_changes.md](docs/ss2d_v5_changes.md)** — SS2D v5: dataloader 사이즈 필터 완화(crop/pad → train +67%, val 7270 = U-Net 평가셋과 동일), Transformer dropout 0.1→0.2, weight_decay 1e-5→3e-5, H/V flip aug, EarlyStopping(patience=5 val check). 모델 코드는 v4 그대로 import.
- **[docs/eter_v5_changes.md](docs/eter_v5_changes.md)** — ETER v5: SS2D v5 와 동일 레시피(공유 dataloader_h5_v5, dropout 0→0.2, weight_decay 1e-7→3e-5, flip aug, EarlyStop). 모델은 v4 클래스 상속한 thin wrapper로 decoder Transformer 에 dropout 주입. SS2D v5 chain 후 자동 시작.
- **[docs/ss2d_v6_changes.md](docs/ss2d_v6_changes.md)** — SS2D v6: v5 의 두 결함 수정 — (1) val SSIM 을 custom `u_choh_SSIM`(val_range 버그) → skimage `structural_similarity`(data_range=target.max−min) 로 교체, (2) EarlyStop/best 기준을 composite(SSIM+NMSE+PSNR+L1 평균) → val_ssim 단일로 단순화. patience 5→10, val 빈도 매 5 epoch, v5 epoch_10.pt 부터 resume. 데이터/모델/regularization 은 v5 그대로. 결과: 200ep 풀 학습 완료, best val SSIM 0.8903 (U-Net 0.8865 동등).
- **[docs/eter_v6_changes.md](docs/eter_v6_changes.md)** — ETER v6: SS2D v6 와 동일 처방(skimage SSIM, SSIM 단일 EarlyStop, patience 10, val 매 5 epoch, v5 epoch_5.pt 부터 resume). 추가로 BATCH_SIZE 8→4 강하(2026-05-11) — v5 는 BS=8 통과했으나 v6 의 baseline 측정 추가로 cudnn workspace 가 커져 첫 forward OOM 발생, 재부팅 후에도 재발하여 4 로 내림. SS2D v6 vs ETER v6 비교의 유일한 변수는 sequence model 종류(Mamba vs GRU) + DC block 유무.
- **[docs/visual_metric_gap_v6.md](docs/visual_metric_gap_v6.md)** — v6 모델의 정량 metric (SSIM 0.89) 과 시각 인상의 괴리 원인 4가지 분석 + 진단 도구 (`visualize_diagnostic_v6.py`) 설계. 원인: (1) raw amplitude SSIM 부풀림 (배경 50%+), (2) 비교 슬라이스 3230 below-average 편향, (3) L1+SSIM mean-prediction 흐림, (4) 에러맵 colormap 조기 saturation. 진단 결과 (`results/vis/vis_diagnostic_v6/`): raw vs masked SSIM gap 모든 모델 음수, U-Net 의 가장 큰 격차 (-0.04 ~ -0.44), SS2D 가 ETER 보다 masked SSIM +0.01, PSNR +1.3dB 일관 우위. → v6_1 개선 방향: edge-aware loss / gradient loss / perceptual loss 도입.
- **[docs/ss2d_eter_v6_1_changes.md](docs/ss2d_eter_v6_1_changes.md)** — SS2D / ETER v6_1: 원인 3 (mean-prediction blurring) 직접 처벌을 위해 finite-difference gradient L1 loss 를 v6 loss 에 추가 (`loss = L1 + 1.0·(1-SSIM) + 10.0·grad_L1`). v6 best ckpt 로부터 50ep fine-tune, LR 5e-5→5e-7 cosine, EARLYSTOP_PATIENCE=5. 신규 파일 5개 (config/train 각 2개 + chain). 기대: PSNR +0.3~0.8dB, zoom-in 에서 sulci/혈관 detail sharp. **실제 결과: SS2D PSNR −0.63dB, ETER PSNR −0.23dB, L1/NMSE 모두 후퇴 (over-sharpening 회귀) — v6_2 로 λ_grad 완화.**
- **[docs/ss2d_eter_v6_2_changes.md](docs/ss2d_eter_v6_2_changes.md)** — SS2D / ETER v6_2: v6_1 의 over-sharpening 회귀 대응. v6_1 doc fallback 적용으로 `λ_grad` 10.0 → **3.0** (−70%) 단일변수 변경. **v6 best 에서 재시작** (v6_1 over-edged state 누적 회피, 단일변수 비교 명확화). 그 외 epochs/LR/dropout/dataloader/DC 모두 v6_1 동일. 기대: PSNR 회복 + SSIM 미세 상승 유지. v6_2 도 후퇴 시 λ_grad=1.0 또는 gradient loss 폐기로 추가 시도.
- **[docs/tier1_tta_ensemble_negative.md](docs/tier1_tta_ensemble_negative.md)** — Tier 1: TTA(4-way flip 평균) / 앙상블(v4+v6, SS2D+ETER cross-arch) 모두 500-sample 부분 평가에서 임계 (+0.003 SSIM, +0.15 dB PSNR) 미달. v6 가 이미 flip aug 로 학습되어 TTA 가 SS2D 에서는 회귀, ETER 에서는 미약 상승. v4 출력 평균은 약한 모델이 노이즈 추가. 풀평가 생략하고 Tier 2 재학습으로 이동.
- **[docs/tier2_sharpness_plan.md](docs/tier2_sharpness_plan.md)** — Tier 2: v6_2 (λ_grad=3), v6_3 (sharp ablation: dropout 0.1, WD 1e-5), v6_4 (VGG perceptual, 수동 launch) 가설 매트릭스. 세 직교 처방으로 v6 의 mean-prediction blurring 동시 공격. **결과 (2026-05-28)**: v6_1/v6_2 양쪽 모델 모두 PSNR 회귀 (over-sharpening 부작용) → 폐기. **v6_3 SS2D 는 모든 지표 v6 보다 개선** (SSIM 0.8913→0.8924, PSNR 35.96→36.05dB, L1 7.37→7.30) — Tier 2 의 첫 non-degradation. ETER v6_3 진행 중. v6_4 train script 는 ETER v6_3 결과 후 결정.
- **[docs/error_map_v2_masked.md](docs/error_map_v2_masked.md)** — 2026-06-01 시각화 정책 개정. `visualize_compare_versions.py` 의 에러맵을 raw amplitude → per-slice [0,1] 정규화 + brain mask (gt_n > 0.05) 로 교체, optional `--match-scale` LS 보정 flag 추가. 출력 dir `vis_compare_versions_masked/` 로 분리해 v1 결과 보존. 정량 metric 은 raw 유지, suptitle 에 명시.
- **[docs/script_version_history.md](docs/script_version_history.md)** — 2026-05-20 정리에서 삭제한 `.py` 파일 (main_train_v3~v6, eval/visualize 옛 버전, 2024년 config, 옛 dataloader 등) 의 출처/역할/버전 진화 방향 기록. v3→v4 DC block, v4→v5 regularization, v5→v6 평가 정합성, v6→v6_1 gradient loss 4단계 정리.
- **[docs/cleanup_log.md](docs/cleanup_log.md)** — 프로젝트에서 삭제된 파일들의 대장. 무엇이 있었고 왜 지웠는지 날짜별 기록.

## 모델 구조

```
입력: aliased image (B, 32, 320, 320) + k-space (B, 32, 320, 320)
       │                                    │
   ViT Encoder → ViT Decoder           GRU 또는 SS2D
       │                                    │
       └──── cat + RefinementBlock ─────────┘
                      │
              출력: (B, 1, 320, 320)
```

## 주요 설정 파일

현재 활성 (v6_2):
- `configs/myConfig_choh_SS2D_model_v6_2.py`
- `configs/myConfig_choh_ETER_model_v6_2.py`

버전 reference 로 보존: `..._v4.py`, `..._v5.py`, `..._v6.py`, `..._v6_1.py` (각 모델).

## 실행 로그 / 체인 스크립트 위치 (2026-05-12 정리)

루트에 흩어져 있던 `run_*.sh` / `run_*.log` 를 `runs/` 폴더로 통합. 학습 entry script (`main_train_*.py`) 와 ckpt 디렉토리 (`logs/`) 는 원위치 유지.

```
runs/
├── ss2d/         # SS2D 학습 stdout/stderr (run_ss2d_v3.log ~ v6.log)
├── eter/         # ETER 학습 stdout/stderr (run_eter_v4.log ~ v6.log)
├── chain/        # chain 실행 스크립트 + chain 로그
│   ├── run_chain_v6.sh                  # 현재 활성 (SS2D v6 → ETER v6)
│   ├── run_chain_ss2d_v4.sh / v5.sh     # 과거 버전
│   ├── run_chain_eter_v5.sh             # 과거 버전
│   └── run_chain_*.log                  # 각 chain 의 단계별 start/end 기록
├── eval/         # 평가 스크립트 stdout (run_eval_*.log)
└── visualize/    # 시각화 스크립트 stdout (run_vis_*.log)
```

- chain 스크립트 내부 경로 (`CHAIN_LOG` / `SS2D_LOG` / `ETER_LOG`) 도 새 폴더 구조로 업데이트됨.
- 현재 실행 중인 학습 프로세스는 `mv` (inode-only rename) 로 옮겨도 안전하게 새 경로에 계속 기록.

## 결과 폴더 구조 (results/) — 2026-06-01 정리

루트에 흩어져 있던 `eval_*.csv` / `vis_*/` 들을 카테고리별로 그룹화. 파일명은 그대로 유지(스크립트 호환).

```
results/
├── eval/                                  # 평가 CSV + summary
│   ├── eval_full_v4/v5/v6/v6_1/...        # 현재 active (full 풀평가)
│   ├── eval_tta_500 / eval_tta_smoke      # TTA 실험
│   └── legacy/                            # 옛 sanity / 초기 baseline
│       ├── eval_ss2d_v4_ss2d_vit_best.*   # 최초 SS2D v4 평가
│       ├── eval_sanity_v5.*               # v5 sanity check
│       └── eval_unet_pretrained.*         # U-Net baseline
├── vis/                                   # 시각화 PNG
│   ├── aligned/      ★ 최신 정합본 (use this)
│   │   ├── vis_compare_v4_aligned/        # 단일 버전 모델 비교 (정합)
│   │   └── vis_compare_versions_aligned/  # 버전 cross-comparison (정합)
│   ├── per_version/                       # 단일 버전 비교 (옛 미정합)
│   │   └── vis_compare_v4/v6/v6_1/
│   ├── cross_versions/                    # 버전 cross (옛 미정합)
│   │   └── vis_compare_versions/
│   ├── vis_diagnostic_v6/                 # raw vs masked SSIM 진단
│   └── legacy/                            # superseded / 옛 / partial run
│       ├── vis_ss2d_v4_ss2d_vit_best/
│       ├── vis_compare_v6_partial/
│       └── vis_compare_versions_bak_middle50/
└── smoke_test_320/                        # 그대로 유지 (smoke_test_320.py 가 직접 참조)
```

- `vis/aligned/` 는 2026-06-01 정합 작업 결과 — slice indices `[1817, 2220, ..., 5452]` 동일, err_vmax = `gt.max() * 0.1` 절대 기준, H5 스케일 통일.
- 스크립트 default 경로도 새 구조로 업데이트됨 (`visualize_compare.py`, `visualize_compare_versions.py`, `visualize_diagnostic_v6.py`).

## 실행

```bash
# 학습 (현재 활성: v6_1)
python main_train_ss2d_v6_1.py    # SS2D-ViT v6_1 (gradient loss fine-tune)
python main_train_eter_v6_1.py    # ETER-ViT v6_1

# 평가 (모델 다중 비교 통합)
python eval_full_compare.py

# 시각화
python visualize_compare.py            # U-Net / SS2D / ETER / GT 비교 PNG
python visualize_diagnostic_v6.py      # raw vs masked SSIM 진단
```

체인 실행: `runs/chain/run_chain_v6_1.sh` (SS2D v6_1 → ETER v6_1 순차).

## 환경

- conda 환경: `mri_env`
- GPU: 8GB (BATCH_SIZE, GRU hidden 등 메모리 제약 있음)
- 주요 의존성: PyTorch, mamba_ssm (SS2D용 CUDA 커널), einops, wandb
