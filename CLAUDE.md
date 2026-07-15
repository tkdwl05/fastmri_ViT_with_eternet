# ViT-based MRI Reconstruction

## 프로젝트 개요

fastMRI brain multicoil 데이터에 대한 MRI 재구성 모델 연구
(컨트라스트는 "AXFLAIR" 단일이 아니라 AXT1/AXT1POST/AXT1PRE/AXT2/AXFLAIR 등 **혼합** — "AXFLAIR" 로만 표기하면 부정확).

ViT 인코더 + 시퀀스 모델 디코더(GRU=ETER 또는 SS2D=Mamba) 구조가 기본 축이다. `v8_eter_pure` 갈래는
ViT 를 아예 빼고 교수님 원본 순수 ETER-Net 위에서 시퀀스 모델(GRU vs SS2D)만 비교하는 통제실험이다.

### 세 갈래 (평행 진행)

| 트랙 | 해상도 | GPU / conda 환경 | 상태 |
|---|---|---|---|
| v1~v6_x (루트) | 320×320, ViT-Small | RTX 5060Ti 8GB, `mri_env` — **옛 머신 전용, 이 저장소엔 ckpt 없음** | "복원" 단계, v6_3 채택후보(ETER v6_3 완료 여부 미문서화) |
| v7 → v7_titan | 384×384, ViT-Base | TITAN RTX 24GB×2(단일 GPU 사용), `base` | "향상" 단계, **현재 운영** — ETER/SS2D 완주, dead-heat |
| v8_eter_pure | 384×384, ViT 없음 | TITAN RTX 24GB(단일), `base` | GRU vs SS2D 통제비교 — no-DC 쌍 완주(SS2D 완승) = **최종**; DC 축 폐기(비표준 확장) |

트랙별 상세 비교표는 `docs/summary_2026-06-11.md` §2, 버전별 하이퍼파라미터는 `docs/version_evolution.md` §2 참고.

## 핵심 문서 (docs/)

작업 히스토리와 설계 판단의 근거를 기록한 문서들:

- **[docs/presentation_overview.md](docs/presentation_overview.md)** — **발표용 통합본**. v1 → v6 → v6_3 까지의 전체 흐름, 두 차례 핵심 발견 (① custom SSIM metric 버그, ② visual-metric gap), Tier 1 (TTA/앙상블) 와 Tier 2 (v6_1/v6_2/v6_3 fine-tune) 결과 정리. **본문(§1~§8)은 2026-05-28 시점 v6_3 까지로 고정**(v6_3 SS2D best val SSIM 0.8924 / PSNR 36.05 dB, ETER v6_3 진행 중 — 완료 여부 미문서화). 발표 5분 요약은 §6.3. **갱신 2026-07-08**: §9 에 v7_titan/v8_eter_pure 확장 트랙 요약 + 문서 포인터 추가.
- **[docs/version_evolution.md](docs/version_evolution.md)** — **버전 변천 통합본 (V4→V6→V7)**. SS2D/ETER 하이퍼파라미터 비교표(config ground-truth) + 전환별 무엇/왜/결과 + 두 핵심발견(① custom SSIM 버그, ② visual-metric gap) + raw(v6)↔masked(v7_titan) 비교 주의 + v7 vs v7_titan 구분. 흩어진 `*_changes.md` 의 단일 비교 진입점. **2026-06-16 작성** (SS2D v7_titan ep50 완주 0.9127/0.9083 = ETER 와 dead-heat 반영).
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
- **[docs/logs_archive.md](docs/logs_archive.md)** — 루트에 흩어져 있던 `run_*.sh`/`run_*.log` 를 `runs/` 폴더로 통합한 기록 (2026-05-15). 해당 `runs/` 자체는 이 머신에 없음(옛 8GB 머신) — 역사적 참고용.

### v7 갈래 (TITAN RTX×2 24GB 마이그레이션, 320×320 유지) — v6→v7_titan 중간 단계
- **[v7/README_v7.md](v7/README_v7.md)** (2026-05-16) — 8GB→TITAN RTX×2 24GB 마이그레이션. BATCH_SIZE 4→16, GRU hidden 6→10, DDP 옵션 추가. v6 코드/ckpt 는 그대로 두고 새 `v7/` 폴더에서 진행(capacity 복원이 목적, 해상도는 아직 320 유지) — 384 승격 + ViT-Base 전환 + ETER U-Net 후처리 복원은 v7_titan 에서.

### v7_titan 갈래 (384×384 · ViT-Base · TITAN RTX 24GB x2) — v6 (320) 와 별개 평행 트랙
- **[docs/eval_metric_redesign.md](docs/eval_metric_redesign.md)** (2026-05-22) — **brain mask + weighted composite metric 재설계**. 배경 부풀림 진단, brain mask = Otsu×0.4 + largest CC (`dataloader_h5_v5.py:243`), composite = 0.5·SSIM + 0.3·(PSNR/40) + 0.2·(1−NMSE), masked L1+SSIM loss. v7_titan 본 학습 직전 적용.
- **[docs/ss2d_v7_titan_changes.md](docs/ss2d_v7_titan_changes.md)** (2026-05-31) — **SS2D v7_titan 재학습 정상화**. DDP 폐기→scratch, true checkpoint resume(full-state `ss2d_vit_last.pt`, LR연속, unit-test PASS), auto-restart supervisor, BS6 풀-step smoke. VAL_EVERY 5→2, patience 10→50, NUM_EPOCHS=50(ETER 비교). 비교 baseline: ETER v7_titan masked composite 0.9127 / SSIM 0.9084.
- **[docs/summary_2026-06-11.md](docs/summary_2026-06-11.md)** (2026-06-11, 갱신 2026-06-16) — **최신 마스터 요약**. v6(320)+v7_titan(384) 통합. SS2D v7_titan **ep50 완주**(composite 0.9127 / SSIM_m 0.9083), ETER 완주(0.9127/0.9084)와 **dead-heat(near-tie 확정)**. 동일-epoch ep10~30 SS2D 우위 → **ep40 ETER 재추월(교차점, §4.5)** — 2026-06-02 "조기 우위" 서사 정정. L1 SS2D 우위(9.298<9.518), NMSE ETER 우위. (이전 스냅샷: [summary_2026-06-02.md](docs/summary_2026-06-02.md))
### v8_eter_pure 갈래 (순수 ETER-Net · GRU vs SS2D 통제비교) — no-DC 쌍 완주 = 최종 (DC 축 폐기)
- **[docs/eternet_paper_data_consistency.md](docs/eternet_paper_data_consistency.md)** (2026-05-31) — 교수님 원본 ETER-net(논문+코드) 확인 결과 명시적 Data-Consistency 블록 없음. 프로젝트의 DC block(v4~)은 SS2D-arm 전용 증강이라 v7_titan ETER-vs-SS2D 비교의 confound였음 — v8_eter_pure 의 no-DC 우선 설계 근거.
- **[docs/v8_eter_pure_rnn_vs_ss2d.md](docs/v8_eter_pure_rnn_vs_ss2d.md)** (2026-07-05, 갱신 2026-07-15) — **교수님 순수 ETER-Net(no ViT, no DC)에서 sequence model 만 GRU↔SS2D 교체하는 통제비교**. v7_titan dead-heat 의 confound(Mamba+DC vs GRU) 제거. 결과: **SS2D 완승** — best composite **0.9200**(ep48) vs GRU 0.9182(ep50), 5지표 전부·params 21×↓(31M vs 668M), matched-epoch 전구간 wire-to-wire 우위 → "DC 목발" 가설 반박. 로그기반 분석 `v8_eter_pure/analyze_v8_nodc.py` → `results/eval/v8_nodc/`. **per-slice paired 검증**(전체 7334 슬라이스, `eval_paired_v8_nodc.py`): SS2D 가 5지표 전부 74~78% 슬라이스 승률(Wilcoxon p≈0). **4-way viz**(`visualize_v8_pure_compare.py`): GRU 는 두개골 바깥 배경에 ringing 아티팩트, SS2D 는 깨끗함(§6). **갱신 2026-07-15**: DC 축 폐기 확정(§7) — 문헌상 비표준(교수님 ETER-net·문헌 RNN+DC 모두 다른 구조)·GRU+DC ep4 NaN·SS2D+DC 도 GRU+DC 와 수치 등가로 판명 → no-DC 가 최종 비교. 문헌 §10.

- (전체 날짜순 인덱스: **[docs/INDEX.md](docs/INDEX.md)**)

## 모델 구조

### 루트 트랙 (v1~v6_x, 320×320) — 역사적 기록, 이 머신에 ckpt 없음

```
입력 1: aliased image (B, 32, 320, 320)
입력 2: k-space        (B, 32, 320, 320)
       │                                    │
   ViT Encoder → ViT Decoder           GRU(ETER) 또는 SS2D(Mamba)
       │                                    │
       └── cat(ViT출력, aliased image, seq출력) ──┘   ← 3-way concat (2-way 아님)
                      │
         최종 합성: RefinementBlock(3×ResBlock)
                      │  (SS2D 만: 이 뒤에 1-iter soft DC block 추가. ETER 는 DC 없음)
              출력: (B, 1, 320, 320)
```

### v7_titan (384×384, ViT-Base) — 현재 운영 트랙

SS2D 는 루트와 동일 클래스(`u_choh_model_SS2D_ViT_v4.py`)를 그대로 쓰고 config 값만 키운다
(ViT-Base, SS2D d_inner 64→128 / d_state 16→32). **ETER 는 최종 합성을 교체**한다:
`RefinementBlock(3×ResBlock)` → `UNet_choh_skip(depth=3, wf=6)` — 교수님 원본 ETER-Net(GRU→U-Net
후처리) 복원이 목적(`choh_Decoder3_ETER_v7_titan`, `models/hybrid_eternet/u_choh_model_ETER_ViT_v7_titan.py`).
즉 v7_titan 에서는 **ETER 와 SS2D 의 최종 합성 구조 자체가 다르다**(ETER=U-Net, SS2D=RefinementBlock+DC) —
이 비대칭이 `docs/eternet_paper_data_consistency.md` 가 지적하는 confound 중 하나.

### v8_eter_pure (384×384, ViT 없음) — GRU vs SS2D 순수 통제비교

```
입력 1: aliased image (B, 32, 384, 384)
입력 2: k-space        (B, 32, 384, 384)
       │                                    │
  (ViT 없음)                          GRU(양방향 h+v) 또는 SS2D
       │                                    │
       └──────── cat(seq출력, aliased image) ────┘   ← 2-way concat
                      │
         UNet_choh_skip (DFU, depth=5, wf=6)
                      │  (use_dc=True 인 DC arm 만: 이 뒤에 DC block 추가 — DC 축 폐기, no-DC 만 최종)
              출력: (B, 1, 384, 384)
```
GRU/SS2D 를 제외한 모든 것이 100% 동일(`models/pure_eternet/u_pure_eternet_{gru,ss2d}.py`) — 단일 변수 통제.

## 주요 설정 파일

### 루트 트랙 (320×320, 이 머신엔 ckpt 없음)
현재 채택후보 v6_3 (SS2D 전지표 개선 확인, **ETER v6_3 완료 여부는 미문서화** —
`docs/summary_2026-06-11.md` §6-⑤):
- `configs/myConfig_choh_SS2D_model_v6_3.py`
- `configs/myConfig_choh_ETER_model_v6_3.py`

버전 reference 로 보존: `..._v4.py` ~ `..._v6_2.py` (각 모델). `..._v6_4.py` 는 config 만 있고
`main_train_*_v6_4.py` 는 미작성(v6_3 성공으로 보류, `docs/tier2_sharpness_plan.md`).

### v7_titan (384×384, 현재 운영)
- `v7_titan/configs/myConfig_choh_SS2D_model_v7_titan.py`
- `v7_titan/configs/myConfig_choh_ETER_model_v7_titan.py`

### v8_eter_pure (384×384, ViT 없음)
- `v8_eter_pure/configs/myConfig_pure_eter_v8.py` — GRU/SS2D × no-DC/DC 4런이 공유하는 단일 config(env var 로 분기)

## 실행 로그 / 체인 스크립트 위치

### 루트 트랙 — `runs/` 폴더는 이 머신에 없음
루트 트랙의 학습 ckpt/로그는 옛 8GB 머신에만 있어(§프로젝트 개요) 이 저장소에는 `runs/` 폴더
자체가 존재하지 않는다. 2026-05-12 정리 당시의 `runs/{ss2d,eter,chain,eval,visualize}/` 구조는
`docs/logs_archive.md` 에 역사적 기록으로만 남아있다.

### v7_titan/runs/
```
v7_titan/runs/
├── ss2d/ eter/ chain/ sanity_eval/     # 학습/체인 로그
├── run_ss2d_v7_titan_autoresume.sh     # true-resume auto-restart supervisor
└── watcher_eter_to_ss2d.sh             # ETER 종료 감지 → SS2D 자동 launch(BS=6)
```
평가/시각화는 독립 `eval/`/`visualize/` 서브폴더 없이 저장소 공통 `results/eval/`,
`results/vis/v7_titan_compare/` 를 그대로 사용한다.

### v8_eter_pure/runs/
```
v8_eter_pure/runs/
├── chain/ gru/ ss2d/            # 모델축(GRU/SS2D) 기준 로그 — root 의 ss2d/eter 명명과 다름
├── run_pure_v8_autoresume.sh    # SEQ_MODEL(gru|ss2d) × USE_DC(0|1) env var 로 4런 분기
└── smoke_bs.txt
```
평가 결과는 `results/eval/v8_nodc/`, 시각화는 `results/vis/v8_pure_eternet_compare/` (v7_titan 과
동일하게 저장소 공통 `results/` 사용).

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

위 트리는 위 2026-06-01 정리 시점(루트 트랙 전용) 스냅샷이다. 이후 트랙들의 결과는 같은
`results/` 아래 별도 서브폴더에 쌓인다 — `results/eval/v7_titan_compare` 계열,
`results/vis/v7_titan_compare/`, `results/eval/v8_nodc/`(per-slice CSV·win-rate 포함),
`results/vis/v8_pure_eternet_compare/`. `.gitignore` 상 `results/` 는 기본 무시되고
`results/vis/v7_titan_compare/`·`results/vis/v7_titan_eval_modes/` 의 PNG/txt 만 예외적으로
GitHub 에 공유되도록 화이트리스트돼 있다(v8 쪽은 아직 화이트리스트 미추가 — 로컬에만 존재).

## 실행

### 루트 트랙 (참고용 — ckpt 부재로 이 머신에서 재현 불가)
```bash
python main_train_ss2d_v6_3.py    # SS2D-ViT v6_3 (채택후보, sharpness 완화)
python main_train_eter_v6_3.py    # ETER-ViT v6_3
python eval_full_compare.py
python visualize_compare.py            # U-Net / SS2D / ETER / GT 비교 PNG
python visualize_diagnostic_v6.py      # raw vs masked SSIM 진단
```

### v7_titan (384, 현재 운영)
```bash
# GPU0 단독, auto-restart supervisor (true-resume, "학습 완료" sentinel 감지 시 종료)
bash v7_titan/runs/run_ss2d_v7_titan_autoresume.sh
python v7_titan/main_train_eter_v7_titan.py

# 4-way 비교 시각화 (GT / U-Net / ETER / SS2D)
python visualize_v7_titan_compare.py
```

### v8_eter_pure (384, ViT 없음, GRU vs SS2D)
```bash
# SEQ_MODEL=gru|ss2d, USE_DC=0|1 로 런 선택 (no-DC 2런이 최종; DC 축 폐기 — docs/v8_eter_pure_rnn_vs_ss2d.md §7)
SEQ_MODEL=ss2d USE_DC=0 CUDA_VISIBLE_DEVICES=0 bash v8_eter_pure/runs/run_pure_v8_autoresume.sh

# per-slice paired 평가 (전체 val, ~2h) + 4-way 시각화 (GT/U-Net/GRU/SS2D)
python v8_eter_pure/eval_paired_v8_nodc.py
python visualize_v8_pure_compare.py
```

## 환경

**이 저장소가 있는 현재 머신** (v7 / v7_titan / v8_eter_pure 가 실제로 도는 곳):
- conda 환경: **`base`** (`/opt/conda`) — `mri_env` 라는 이름의 conda env 는 이 머신에 **존재하지 않는다**. 학습은 그냥 `python ...` (activate 불필요).
- GPU: **TITAN RTX 24GB × 2** — v7_titan/v8_eter_pure 는 정책상 **GPU0 단독** 사용, GPU1 은 교수님 작업 회피용으로 항상 비워둠.
- 주요 의존성: PyTorch 2.3.1, mamba_ssm 2.2.2(SS2D용 CUDA 커널), einops, wandb

**역사적 환경** (v1~v6_x 가 실제로 학습된 옛 머신 — 이 저장소엔 해당 ckpt/로그 없음):
- conda 환경: `mri_env`
- GPU: RTX 5060Ti 8GB — BATCH_SIZE, GRU hidden 축소 등 v1~v6 의 모든 용량 관련 의사결정이 이 제약에서 나옴
