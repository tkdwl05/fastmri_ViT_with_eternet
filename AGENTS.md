# ViT-based MRI Reconstruction

## 프로젝트 개요

fastMRI brain multicoil 데이터에 대한 MRI 재구성 모델 연구
(컨트라스트는 "AXFLAIR" 단일이 아니라 AXT1/AXT1POST/AXT1PRE/AXT2/AXFLAIR 등 **혼합** — "AXFLAIR" 로만 표기하면 부정확).

ViT 인코더 + 시퀀스 모델 디코더(GRU=ETER 또는 SS2D=Mamba) 구조가 기본 축이다. `v8_eter_pure` 갈래는
ViT 를 아예 빼고 교수님 원본 순수 ETER-Net 위에서 시퀀스 모델(GRU vs SS2D)만 비교하는 통제실험이다.
`v9_mamba` 갈래(unleashed/radapt)는 그 통제실험의 승자(SS2D)를 강화(게이팅·다중 잔차블록·병목 해제)해
R4 품질을 더 밀어붙이고(unleashed), 가속률 R 일반화까지 확장(radapt)하는 **현재 운영** 트랙이다.

### 네 갈래 (평행 진행)

| 트랙 | 해상도 | GPU / conda 환경 | 상태 |
|---|---|---|---|
| v1~v6_x (루트) | 320×320, ViT-Small | RTX 5060Ti 8GB, `mri_env` — **옛 머신 전용, 이 저장소엔 ckpt 없음** | "복원" 단계, v6_3 채택후보(ETER v6_3 완료 여부 미문서화) |
| v7 → v7_titan | 384×384, ViT-Base | TITAN RTX 24GB×2(단일 GPU 사용), `base` | "향상" 단계 — ETER/SS2D ep50 완주, dead-heat (완료·역사) |
| v8_eter_pure | 384×384, ViT 없음 | TITAN RTX 24GB(단일), `base` | GRU vs SS2D 통제비교 — no-DC 쌍 완주(SS2D 완승 0.9200) = **GRU↔SS2D 비교의 최종 결론**; DC 축 폐기(비표준 확장) |
| v9_mamba (unleashed/radapt) | 384×384, ViT 없음 | TITAN RTX 24GB(GPU0 단독), `base` | **현재 운영** — v8 승자(SS2D) 강화(게이팅·3블록·병목해제). **unleashed 80ep 완주(07-30): best 0.9203 > v8 SS2D 0.9200 목표 달성**(근소·유의, per-slice 5지표 win-rate 54~56%). radapt 는 07-30 NVML 좌초 → 08-05 재기동 → 08-07 정전대비 clean-stop(ep20, 손실 0) → **08-18 true-resume 재개**(ETA ~08-25) |

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
- **[docs/eval_metric_redesign.md](docs/eval_metric_redesign.md)** (2026-05-22) — **brain mask + weighted composite metric 재설계**. 배경 부풀림 진단, brain mask = Otsu×0.4 + largest CC (`dataloader_h5_v5.py:243`), composite = 0.5·SSIM + 0.3·(PSNR/40) + 0.2·(1−NMSE), masked L1+SSIM loss. v7_titan 본 학습 직전 적용. **⚠ 2026-08-07 사용자 결정: composite(자체 설계·문헌 비표준)는 이후 모든 보고·평가·비교에서 사용하지 않음 — 표준 지표만(SSIM 주지표 + PSNR/NMSE/L1; v8 best = SSIM 0.9140/0.9126, v9 = 0.9145 로 인용). 이 문서와 기존 로그·트랙표의 composite 수치는 역사 기록이며, 실행 중인 트레이너의 best-ckpt 선택 기준은 학습 무결성을 위해 소급 변경하지 않는다.**
- **[docs/ss2d_v7_titan_changes.md](docs/ss2d_v7_titan_changes.md)** (2026-05-31) — **SS2D v7_titan 재학습 정상화**. DDP 폐기→scratch, true checkpoint resume(full-state `ss2d_vit_last.pt`, LR연속, unit-test PASS), auto-restart supervisor, BS6 풀-step smoke. VAL_EVERY 5→2, patience 10→50, NUM_EPOCHS=50(ETER 비교). 비교 baseline: ETER v7_titan masked composite 0.9127 / SSIM 0.9084.
- **[docs/summary_2026-06-11.md](docs/summary_2026-06-11.md)** (2026-06-11, 갱신 2026-07-08: §4.6 v8 트랙 추가) — **최신 마스터 요약**. v6(320)+v7_titan(384) 통합. SS2D v7_titan **ep50 완주**(composite 0.9127 / SSIM_m 0.9083), ETER 완주(0.9127/0.9084)와 **dead-heat(near-tie 확정)**. 동일-epoch ep10~30 SS2D 우위 → **ep40 ETER 재추월(교차점, §4.5)** — 2026-06-02 "조기 우위" 서사 정정. L1 SS2D 우위(9.298<9.518), NMSE ETER 우위. (이전 스냅샷: [summary_2026-06-02.md](docs/summary_2026-06-02.md))
### v8_eter_pure 갈래 (순수 ETER-Net · GRU vs SS2D 통제비교) — no-DC 쌍 완주 = GRU↔SS2D 비교의 최종 (DC 축 폐기)
- **[docs/eternet_paper_data_consistency.md](docs/eternet_paper_data_consistency.md)** (2026-05-31) — 교수님 원본 ETER-net(논문+코드) 확인 결과 명시적 Data-Consistency 블록 없음. 프로젝트의 DC block(v4~)은 SS2D-arm 전용 증강이라 v7_titan ETER-vs-SS2D 비교의 confound였음 — v8_eter_pure 의 no-DC 우선 설계 근거.
- **[docs/v8_eter_pure_rnn_vs_ss2d.md](docs/v8_eter_pure_rnn_vs_ss2d.md)** (2026-07-05, 갱신 2026-07-15) — **교수님 순수 ETER-Net(no ViT, no DC)에서 sequence model 만 GRU↔SS2D 교체하는 통제비교**. v7_titan dead-heat 의 confound(Mamba+DC vs GRU) 제거. 결과: **SS2D 완승** — best composite **0.9200**(ep48) vs GRU 0.9182(ep50), 5지표 전부·params 21×↓(31M vs 668M), matched-epoch 전구간 wire-to-wire 우위 → "DC 목발" 가설 반박. 로그기반 분석 `v8_eter_pure/analyze_v8_nodc.py` → `results/eval/v8_nodc/`. **per-slice paired 검증**(전체 7334 슬라이스, `eval_paired_v8_nodc.py`): SS2D 가 5지표 전부 74~78% 슬라이스 승률(Wilcoxon p≈0). **4-way viz**(`visualize_v8_pure_compare.py`): GRU 는 두개골 바깥 배경에 ringing 아티팩트, SS2D 는 깨끗함(§6). **갱신 2026-07-15**: DC 축 폐기 확정(§7) — 문헌상 비표준(교수님 ETER-net·문헌 RNN+DC 모두 다른 구조)·GRU+DC ep4 NaN·SS2D+DC 도 GRU+DC 와 수치 등가로 판명 → no-DC 가 최종 비교. 문헌 §10.
- **[docs/v8_ss2d_kspace_domain_review.md](docs/v8_ss2d_kspace_domain_review.md)** (2026-07-08) — 외부 리뷰(ETER-net+SS2D) 판정 + SS2D 가 domain-transform 자리(k-space 입력, GRU 와 동일)임을 코드로 확정. 리뷰어의 "지역성↔전역성 미스매치" 우려는 no-DC 완승으로 실증 기각. 후속 로드맵: 가속률(R) 일반화 성립 조건(DC·mask-aware·multi-AR·전역 RF) — **v9_mamba radapt 의 설계 근거**.

### v9_mamba 갈래 (강화 SS2D · R4 품질 극대화 + R 일반화) — 현재 운영
- **[docs/v9_mamba_unleashed_and_radapt.md](docs/v9_mamba_unleashed_and_radapt.md)** (2026-07-23) — **v9 신규 트랙 두 변형**. v8 no-DC 파이프라인은 그대로 두고 **시퀀스 모델만 강화 SS2D**(`models/mamba_eternet/ss2d_v9.py`: 게이팅 복원 `y=y·SiLU(z)` + 3-블록 잔차 스택 + 병목 해제 out_ch 20→64, d_inner 128→256/d_state 16→32)로 교체. **fp16 selective-scan + ds=3 다운샘플**(128² coarse scan)로 풀용량 유지하며 v8(2.78 h/ep)보다 빠름(2.51 h/ep) → epochs 50→80. **unleashed** = 고정 R4 품질 극대화(mask/DC 없음), **radapt** = 같은 백본 + R 일반화 3요소(mask-channel conditioning·v8 DC block 재사용·multi-AR 학습 R∈{2,3,4,5,6,8}). 원본(`ss2d.py`·`myUNet_DF.py`·`dataloader_h5_v5.py`) **무수정**, 신규 파일만 추가. **결과(08-05 갱신)**: unleashed 80ep 완주(07-30, early_stop 없음) — best ep78 **composite 0.9203 / ssim_m 0.9145 / psnr 35.18** 로 목표(v8 SS2D 0.9200/0.9140) **근소 돌파**. 0.9200 도달 최초 ep70 — matched-ep50 시점엔 0.9171 로 미달, 우위는 80ep 연장 구간에서(§11 정직 주석: best 도달 wall-clock 은 v9 가 더 소요). **per-slice paired 검증**(`v9_mamba_unleashed/eval_paired_v9.py`, 전체 7334슬라이스, v8 CSV 조인): vs v8-SS2D **5지표 전부 승, win-rate 54~56%**(Wilcoxon p≤4e-13 — 유의하나 근소) / vs GRU 78~82% 완승. 로그기반 3-way 곡선·표 `analyze_v9_unleashed.py` → `results/eval/v9_unleashed/`. radapt 는 07-30 자동 launch 가 NVML 다운으로 좌초(supervisor 50회 소진, ckpt 0개) → **08-05 재기동**(scratch, MAX_RETRY=200) → 08-07 정전대비 clean-stop(ep20 경계, 손실 0) → **08-18 true-resume 재개**(ETA ~08-25).

### paper/ (논문 트랙, 2026-08~) — 커밋 prefix "paper:"
- **★ `paper/paper.md` — 논문 초안 수정용 작업 기준서(진입점). `draft_ko_v2.md` 를 고치거나 영문 전환하기 전에 반드시 먼저 읽는다.** 수치 정본(권위 순서: draft 본문 표 > make_tables.py > results/eval > docs)·**폐기 수치 목록**(h/ep 2.78/2.51, best 도달 181h/133h, "학습 1회 9~12일")·금지 서술(composite, "같은 학습량에서 더 좋다", 기준선 우열 판정)·필수 caveat(단일 시드, matched-ep50 미달, NMSE 통제판 우세, contrast 불균일)·한국어 문체 표준(격식체, 주관적 어휘 `확인했다/확정했다/발견했다` 금지, 용어 `가속화정도`·`aliasing artifact` 통일)·미해결 ✎ 10건.
- `paper/draft_ko_v2.md`/`.docx` — 한국어 투고 초안 v2.1 (MDPI 공학형, 스코프 v8+v9 unleashed, 외부검토 08-18 반영 P0 8건·P1 8건). `references.bib` 73항목 서지 전건 확정(⚠ 0건). 프로젝트 여정 서사는 `project_story_v1_to_v9.md`.
- **`paper/make_tables.py` — Table 1·2·2b·3·S1 을 md+tex 양쪽으로 자동 생성(`paper/tables/`). 수치가 바뀌면 표를 손편집하지 말고 이 스크립트를 재실행.** 그림은 `make_fig1_architecture.py`/`make_fig4_per_slice.py` → `paper/figs/`.
- 논문·보고 지표는 표준 지표만(SSIM 주지표 + PSNR/NMSE/L1) — composite 사용 금지 (08-07 전면 결정, `docs/eval_metric_redesign.md` ⚠ 참조).

- (전체 날짜순 인덱스: **[docs/INDEX.md](docs/INDEX.md)**)

## 모델 구조

### 코드 레이아웃 (models/ · dataloaders/)
- `models/` — `pure_eternet/`(v8·v9 순수 ETER: `u_pure_eternet_{gru,ss2d}.py`, `u_pure_eternet_ss2d_v9{,_radapt}.py`), `mamba_eternet/`(SS2D 클래스: `ss2d.py`·`ss2d_v9.py`·`u_choh_model_SS2D_ViT_v4.py`(DC block 정의)), `hybrid_eternet/`(ViT+seq: v7_titan ETER `u_choh_model_ETER_ViT_v7_titan.py`), `mae/`·`vit_pytorch/`(ViT 백본), `pretrained/`(U-Net/VarNet leaderboard ckpt — baseline).
- `dataloaders/` — 공유 로더 `dataloader_h5_v5.py`(brain mask `:243`, v5~v9 전부 이 클래스 상속), v9 `dataloader_h5_v9_multiAR.py`(per-sample R∈{2,3,4,5,6,8} 랜덤), + 옛 `myDataloader_fastmri_brain_*.py`.

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

### v7_titan (384×384, ViT-Base) — 완료·역사 트랙 (현재 운영은 v9)

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

### v9_mamba (384×384, ViT 없음) — v8 SS2D 강화 + R 일반화

v8 no-DC SS2D 파이프라인을 그대로 물려받되(`cat(ss2d출력, aliased image)` → `UNet_choh_skip` DFU depth=5/wf=6,
384·R4·brain-mask·masked loss·composite 전부 동일) **시퀀스 모델만 강화 SS2D 로 교체**한다
(`models/mamba_eternet/ss2d_v9.py` = `SelectiveScan1DV9`/`SS2DBlockV9`/`SS2DStackV9`):
- **게이팅 복원**: `in_proj → 2·d_inner` chunk 후 `y = y·SiLU(z)` (v8 이 누락한 Mamba 게이트)
- **잔차 스택**: 채널불변 블록 3개(residual skip) — v8 은 단일 블록
- **병목 해제**: out_ch 20(GRU 매칭 고정)→64(자유). d_inner 128→256, d_state 16→32, dropout 0→0.05
- **fp16 스캔 + ds=3 다운샘플**: stem(풀해상도) → ds=3(128² coarse scan) → 3블록 → 업샘플 → head.
  풀용량 유지하며 v8(2.78 h/ep)보다 빠름(2.51 h/ep) → epochs 50→80. 총 ~33-34M params(SS2D stack 자체는 ~2M, U-Net DFU 가 지배).

**unleashed** = 고정 R4 품질 극대화(mask/DC 없음). **radapt** = 같은 백본 + R 일반화 3요소:
(1) mask-channel conditioning(`cat(x_ksp, mask)`, SS2D 입력 c_in 32→33), (2) v8 DC block 재사용(U-Net
n_classes=2 복소 → 1-iter soft DC → magnitude), (3) multi-AR 학습(R∈{2,3,4,5,6,8} per-sample 랜덤, val 은
R4 고정). R-embedding/FiLM/adaptive-norm 은 쓰지 않음 — 위 3요소로 간접 R-조건화. DC fp16 안정화(v8 DC ep4
NaN 재발 방지): α clamp[0,1] + GradScaler init_scale 8192 + NaN-skip. 교수님/프로젝트 원본(`ss2d.py`·
`myUNet_DF.py`·`u_choh_model_SS2D_ViT_v4.py`·`dataloader_h5_v5.py`) **무수정** — v9 는 신규 파일만 추가.

## 주요 설정 파일

### 루트 트랙 (320×320, 이 머신엔 ckpt 없음)
현재 채택후보 v6_3 (SS2D 전지표 개선 확인, **ETER v6_3 완료 여부는 미문서화** —
`docs/summary_2026-06-11.md` §6-⑤):
- `configs/myConfig_choh_SS2D_model_v6_3.py`
- `configs/myConfig_choh_ETER_model_v6_3.py`

버전 reference 로 보존: `..._v4.py` ~ `..._v6_2.py` (각 모델). `..._v6_4.py` 는 config 만 있고
`main_train_*_v6_4.py` 는 미작성(v6_3 성공으로 보류, `docs/tier2_sharpness_plan.md`).

### v7_titan (384×384) — 완료·역사
- `v7_titan/configs/myConfig_choh_SS2D_model_v7_titan.py`
- `v7_titan/configs/myConfig_choh_ETER_model_v7_titan.py`

### v8_eter_pure (384×384, ViT 없음)
- `v8_eter_pure/configs/myConfig_pure_eter_v8.py` — GRU/SS2D × no-DC/DC 4런이 공유하는 단일 config(env var 로 분기)

### v9_mamba (384×384, ViT 없음) — 현재 운영
v8 처럼 env var 로 분기하지 않고 **변형별 config 파일이 따로** 있다(공유 백본 하이퍼파라미터는 동일):
- `v9_mamba_unleashed/configs/myConfig_ss2d_v9.py` — R4 품질(d_inner 256·d_state 32·n_blocks 3·out_ch 64·ds 3·dropout 0.05 / BS 8·80ep·LR 2e-4·WD 3e-5·patience 40·val 매 2ep)
- `v9_mamba_radapt/configs/myConfig_ss2d_v9_radapt.py` — 동일 백본 + R 일반화(`MASK_CONDITION=True`·`AR_CHOICES=(2,3,4,5,6,8)`·`VAL_ACCELERATION=4`) + DC 안정화(`DC_ALPHA_MIN/MAX=0/1`·`GRADSCALER_INIT_SCALE=8192`)

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

### v9_mamba_unleashed/runs/ + v9_mamba_radapt/runs/
```
v9_mamba_unleashed/runs/
├── run_v9_chain_gpu0.sh           # ★ 진입점: unleashed→(DONE 확인)→radapt 순차 체인 드라이버
├── run_ss2d_v9_autoresume.sh      # unleashed true-resume supervisor (MAX_RETRY=50, 60s 간격)
├── chain_v9.log chain_v9_outer.log ss2d/*.log   # 체인·학습 로그
└── smoke_bs.txt (=8)
v9_mamba_radapt/runs/
├── run_ss2d_v9_radapt_autoresume.sh   # radapt supervisor
├── post_reboot_rearm.sh               # ★ 재부팅/컨테이너 재시작 후 원-커맨드 재개 (절차: RESUME_AFTER_OUTAGE.md)
├── clean_stop_pre_outage.sh · snapshot_pre_outage.sh   # 정전대비 clean-stop/스냅샷 (08-07 사용, pre_outage_report 참고)
├── ss2d/                              # 학습 로그
└── smoke_bs.txt
```
- ckpt/state 는 v7/v8 의 `runs/` 서브폴더와 달리 **`./logs/<RUN_NAME>/`** 아래에 쌓인다: `ss2d_v9_last.pt`(매 ep, full-state resume), `ss2d_v9_epoch_N.pt`(매 5ep), `ss2d_v9_best.pt`(best composite). RUN_NAME = `PureETER_SS2D_V9_unleashed_R4_brain384` / `..._radapt_multiAR_brain384`.
- 완료 sentinel = **`logs/<RUN_NAME>/DONE` 파일**(v8 의 stale-log grep 버그 대체). 체인 드라이버가 unleashed 의 DONE 을 확인한 뒤에야 radapt 를 시작한다.
- 평가 결과: `results/eval/v9_unleashed/`(per-slice paired CSV·win-rate·3-way 곡선, 2026-08-05 — 로컬 전용). 시각화·radapt R-sweep 은 radapt 완주 후 post-hoc(v8 `v8_eter_pure/eval_r_generalization_v8.py` 재사용/변형).

## 결과 폴더 구조 (results/)

**현재 디스크 실제 상태** (루트 트랙 결과는 이 저장소에서 빠져 없음 — 맨 아래 "역사적 스냅샷" 참고):
```
results/
├── eval/
│   ├── v8_nodc/                  # v8 per-slice CSV·win-rate·matched-epoch·composite/SSIM/PSNR 곡선
│   ├── v8_r_sweep/               # v8 R 일반화 cross-eval (raw)
│   ├── v8_r_sweep_norm/          #   〃 (normalized)
│   ├── baselines_384_sample300/  # U-Net/VarNet 384 기준선 (train+val 누수·zero-coil NaN 캐비엇 — docs 참고)
│   └── v9_unleashed/             # v9 per-slice paired CSV·win-rate·3-way 곡선 (2026-08-05, 학습머신 로컬 전용)
└── vis/
    ├── v7_titan_compare/         # v7_titan 4-way GT/U-Net/ETER/SS2D
    ├── v7_titan_eval_modes/      # v7_titan eval-mode 비교
    ├── v8_pure_eternet_compare/  # v8 4-way GT/U-Net/GRU/SS2D
    ├── v9_unleashed_compare/     # v9 4-way GT/U-Net/v8-SS2D/v9 (visualize_v9_compare.py)
    ├── root_track/               # 옛 result/ 병합분(2026-08-27): 루트 트랙 v4~v6 시각화
    │   ├── compare_versions_aligned/  #   (구 vis/aligned/vis_compare_versions_aligned)
    │   ├── compare_v4/ compare_versions/
    ├── _slides_assets_v6/        # v6 발표덱 자산 (구 result/vis/_slides_assets — build_v6_deck.py 가 참조)
    └── _slides_assets_v8/        # v8 발표덱 자산
```
(radapt R-sweep 은 radapt 완주 후. **2026-08-27**: 루트의 옛 `result/`(단수) 폴더를 `results/` 로 병합·삭제 —
`result/vis/{compare_versions_aligned,compare_v4,compare_versions}` → `results/vis/root_track/`,
`result/vis/_slides_assets` → `results/vis/_slides_assets_v6/`. 참조 갱신: `tools/build_v6_deck.py`·`docs/figma_deck_handoff.md`.
전역 `*.png` ignore 로 git 미추적(로컬 전용) — 루트 트랙 결과 비공유 정책 유지.)

**`.gitignore` 화이트리스트** — `results/` 는 기본 무시, 아래만 예외적으로 GitHub 공유:
`results/vis/{v7_titan_compare,v7_titan_eval_modes,v8_pure_eternet_compare,v9_unleashed_compare}/` 의 PNG/txt +
`results/eval/v8_nodc/`(matched_epoch_table.md·win_rate_summary.md·곡선 PNG) +
`results/eval/v8_r_sweep{,_norm}/`(r_generalization_table.md·PNG) +
`results/eval/baselines_384{,_sample300}/`(baseline_summary.md). 대용량 per-slice CSV/log 은 계속 무시.
`root_track/`·`_slides_assets_v6/`·`_slides_assets_v8/` 는 화이트리스트 미추가 — 로컬 전용.

---

**역사적 스냅샷 (2026-06-01, 루트 트랙 전용 — 아래 트리의 폴더들은 현재 디스크에 없음)**: 당시 루트 트랙
`eval_*.csv`/`vis_*/` 를 카테고리별로 그룹화한 구조. 루트 트랙 결과가 저장소에서 빠지며 함께 제거됐다.
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

- (참고) `vis/aligned/` 정합 기준: slice `[1817, 2220, …, 5452]` 동일, err_vmax = `gt.max()*0.1` 절대 기준, H5 스케일 통일 (`visualize_compare*.py`·`visualize_diagnostic_v6.py` default 경로가 이 구조를 가리켰음).

## 실행

### 루트 트랙 (참고용 — ckpt 부재로 이 머신에서 재현 불가)
```bash
python main_train_ss2d_v6_3.py    # SS2D-ViT v6_3 (채택후보, sharpness 완화)
python main_train_eter_v6_3.py    # ETER-ViT v6_3
python eval_full_compare.py
python visualize_compare.py            # U-Net / SS2D / ETER / GT 비교 PNG
python visualize_diagnostic_v6.py      # raw vs masked SSIM 진단
```

### v7_titan (384) — 완료·역사
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

### v9_mamba (384, ViT 없음, 강화 SS2D — 현재 운영)
```bash
# ★ 진입점: unleashed→radapt 전체 체인 (GPU0, true-resume, 각 단계 DONE sentinel 확인 후 다음 단계)
setsid nohup bash v9_mamba_unleashed/runs/run_v9_chain_gpu0.sh \
  > v9_mamba_unleashed/runs/chain_v9_outer.log 2>&1 < /dev/null & disown
# (체인 스크립트가 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True · CUDA_VISIBLE_DEVICES=0 · WANDB_MODE=online export)

# 개별 단계만 돌릴 때 — 각 supervisor 직접 실행 (radapt 는 unleashed DONE 후)
bash v9_mamba_unleashed/runs/run_ss2d_v9_autoresume.sh          # unleashed
bash v9_mamba_radapt/runs/run_ss2d_v9_radapt_autoresume.sh      # radapt
```

## 환경

**이 저장소가 있는 현재 머신** (v7 / v7_titan / v8_eter_pure / v9_mamba 가 실제로 도는 곳):
- conda 환경: **`base`** (`/opt/conda`) — `mri_env` 라는 이름의 conda env 는 이 머신에 **존재하지 않는다**. 학습은 그냥 `python ...` (activate 불필요).
- GPU: **TITAN RTX 24GB × 2** — v7_titan/v8_eter_pure/v9_mamba 는 정책상 **GPU0 단독** 사용, GPU1 은 교수님 작업 회피용으로 항상 비워둠.
- 주요 의존성: PyTorch 2.3.1, mamba_ssm 2.2.2(SS2D용 CUDA 커널), einops, wandb
- git 인증: **SSH** (`git@github.com` remote, `~/.ssh/id_ed25519`) — 2026-08-20 PAT 만료로 전환. push 실패 시 `ssh -T git@github.com` 부터 확인, remote URL 에 credential 임베드 금지.

**역사적 환경** (v1~v6_x 가 실제로 학습된 옛 머신 — 이 저장소엔 해당 ckpt/로그 없음):
- conda 환경: `mri_env`
- GPU: RTX 5060Ti 8GB — BATCH_SIZE, GRU hidden 축소 등 v1~v6 의 모든 용량 관련 의사결정이 이 제약에서 나옴
