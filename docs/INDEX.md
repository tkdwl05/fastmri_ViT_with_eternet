# docs/ 인덱스 (날짜순)

| 날짜 | 파일 | 한 줄 요약 |
|---|---|---|
| 2026-04-11 | [architecture_ETER_vs_SS2D.md](architecture_ETER_vs_SS2D.md) | ETER-ViT(GRU) vs SS2D-ViT(Mamba) 아키텍처/설정/학습 조건 상세 비교 |
| 2026-04-15 | [SS2D_v1_analysis.md](SS2D_v1_analysis.md) | SS2D v1 의 blurry 복원 원인 7가지 진단 (patch 32x32, Conv2d 1개, SSIM weight 부족 등) |
| 2026-04-16 | [eter_8gb축소.md](eter_8gb축소.md) | 원본 ETER-Net(RTX, 384x384) → 8GB GPU(320x320) 축소 과정 (GRU h 10→2, U-Net→Conv2d 1개) |
| 2026-04-20 | [scheduler_change.md](scheduler_change.md) | `CosineAnnealingWarmRestarts` (톱니) → `CosineAnnealingLR` (단조 감소) 교체 근거 |
| 2026-04-22 | [ss2d_v4_changes.md](ss2d_v4_changes.md) | SS2D v4: A capacity 증설 + B weight_decay/dropout + C 1-iter soft DC block |
| 2026-04-27 | [eter_v4_analysis.md](eter_v4_analysis.md) | ETER v4 회귀 분석 (v3 0.7475 → v4 0.7320, EarlyStop 부재 등) |
| 2026-04-30 | [ss2d_v5_changes.md](ss2d_v5_changes.md) | SS2D v5: dataloader 사이즈 필터 완화 (+67%), dropout 0.2, weight_decay 3e-5, H/V flip aug, EarlyStop patience=5 |
| 2026-04-30 | [eter_v5_changes.md](eter_v5_changes.md) | ETER v5: SS2D v5 와 동일 레시피 (공유 dataloader_h5_v5, dropout/wd, flip aug, EarlyStop) |
| 2026-05-04 | [ss2d_v6_changes.md](ss2d_v6_changes.md) | SS2D v6: val SSIM custom→skimage 교체, EarlyStop val_ssim 단일, patience 5→10. 200ep 완주 best 0.8903 |
| 2026-05-04 | [eter_v6_changes.md](eter_v6_changes.md) | ETER v6: SS2D v6 와 동일 처방. BS 8→4 강하 (OOM 회피) |
| 2026-05-06 | [presentation_overview.md](presentation_overview.md) | v1~v6 전체 변천사, 발표용 정리 |
| 2026-05-15 | [logs_archive.md](logs_archive.md) | 루트의 `run_*.sh`/`run_*.log` 를 `runs/` 폴더로 통합 정리 |
| 2026-05-22 | [eval_metric_redesign.md](eval_metric_redesign.md) | brain mask + weighted composite metric 재설계 (D1-D4) + LR 의미 부록 + 학계 metric 표준 |
| **2026-05-31** | **[ss2d_v7_titan_changes.md](ss2d_v7_titan_changes.md)** | **SS2D v7_titan 재학습 정상화: scratch + true-resume(full-state last.pt, LR연속) + auto-restart supervisor + BS6 풀-step smoke. VAL_EVERY 5→2, patience 10→50, NUM_EPOCHS=50 유지(ETER 비교). 비교 baseline ETER 0.9127/0.9084** |
| **2026-06-02** | **[summary_2026-06-02.md](summary_2026-06-02.md)** | **전체 md 통합 마스터 정리 — v6(320) + v7_titan(384) 트랙 + 라이브 상태(SS2D ep7/50, ep6에서 ETER ep10 추월) 한 곳에. §6 에 문제점 12종 정리** |
| **2026-06-11** | **[summary_2026-06-11.md](summary_2026-06-11.md)** | **마스터 요약(최신, 갱신 2026-06-16) — SS2D v7_titan ep50 완주 0.9127/0.9083, ETER 0.9127/0.9084 와 near-tie 확정(SSIM 동률, L1 SS2D 우위). 동일-epoch ep10~30 SS2D 우위 → ep40 ETER 재추월(교차점, 06-02 "조기 우위" 서사 정정). §4.5 head-to-head** |
| **2026-06-16** | **[version_evolution.md](version_evolution.md)** | **V4→V6→V7 버전 변천 통합본 — SS2D/ETER 하이퍼파라미터 비교표(config 검증) + 전환별 무엇/왜/결과 + 두 핵심발견(SSIM 버그·visual-metric gap) + v7 vs v7_titan 구분. SS2D v7_titan ep50 완주 0.9127/0.9083 = ETER dead-heat 반영** |
| **2026-07-05** | **[v8_eter_pure_rnn_vs_ss2d.md](v8_eter_pure_rnn_vs_ss2d.md)** | **v8: 교수님 순수 ETER-Net(no ViT, no DC)에서 GRU↔SS2D 만 교체하는 통제비교 — v7_titan dead-heat 의 confound(DC) 제거. SS2D 완승(composite 0.9200 vs 0.9182, 21×↓ params), wire-to-wire, "DC 목발" 가설 반박. 갱신 2026-07-07: per-slice paired win-rate(74~78%, p≈0) + 4-way viz(GRU 배경 ringing) 로 검증 완료** |

> **v5 결과는 비정상 조기종료 학습** (EarlyStop 잘못된 composite 기준, ep4 에서 종료) 이라 baseline / 비교 대상에서 제외 합니다. 자세히는 [ss2d_v6_changes.md §1](ss2d_v6_changes.md) 참고.

> 본 학습 / 평가 코드의 변천 흐름은 v4 → (v5: 미완료) → v6 → v7 → v7_titan 로 이어집니다. v7_titan 이 현재 운영 버전 (TITAN RTX x2, 384×384, ViT-Base, ETER U-Net 후처리 복원, brain mask 평가).
