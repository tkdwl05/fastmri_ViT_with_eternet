# docs/ 인덱스 (날짜순)

| 날짜 | 파일 | 한 줄 요약 |
|---|---|---|
| 2026-04-11 | [architecture_ETER_vs_SS2D.md](architecture_ETER_vs_SS2D.md) | ETER-ViT(GRU) vs SS2D-ViT(Mamba) 아키텍처/설정/학습 조건 상세 비교 |
| 2026-04-11 | [project_summary_2026-04-11.md](project_summary_2026-04-11.md) | (역사) 옛 8GB 머신·루트 320 트랙 기준 코드베이스 기술 요약 — 구 루트 `PROJECT_SUMMARY.md`, 2026-09-03 docs/ 로 이동 |
| 2026-04-22 | [ss2d_v4_changes.md](ss2d_v4_changes.md) | SS2D v4: A capacity 증설 + B weight_decay/dropout + C 1-iter soft DC block |
| 2026-04-27 | [eter_v4_analysis.md](eter_v4_analysis.md) | ETER v4 회귀 분석 (v3 0.7475 → v4 0.7320, EarlyStop 부재 등) |
| 2026-04-30 | [ss2d_v5_changes.md](ss2d_v5_changes.md) | SS2D v5: dataloader 사이즈 필터 완화 (+67%), dropout 0.2, weight_decay 3e-5, H/V flip aug, EarlyStop patience=5 |
| 2026-04-30 | [eter_v5_changes.md](eter_v5_changes.md) | ETER v5: SS2D v5 와 동일 레시피 (공유 dataloader_h5_v5, dropout/wd, flip aug, EarlyStop) |
| 2026-05-04 | [ss2d_v6_changes.md](ss2d_v6_changes.md) | SS2D v6: val SSIM custom→skimage 교체, EarlyStop val_ssim 단일, patience 5→10. 200ep 완주 best 0.8903 |
| 2026-05-04 | [eter_v6_changes.md](eter_v6_changes.md) | ETER v6: SS2D v6 와 동일 처방. BS 8→4 강하 (OOM 회피) |
| 2026-05-06 | [presentation_overview.md](presentation_overview.md) | v1~v6 전체 변천사, 발표용 정리 (갱신 2026-07-08: §9 에 v7_titan/v8 확장 트랙 포인터 추가) |
| 2026-05-15 | [logs_archive.md](logs_archive.md) | 루트의 `run_*.sh`/`run_*.log` 를 `runs/` 폴더로 통합 정리 |
| 2026-05-16 | [../v7/README_v7.md](../v7/README_v7.md) | v7: TITAN RTX×2 24GB 마이그레이션 + capacity 복원(BATCH_SIZE 4→16, GRU hidden 6→10) + DDP 옵션. v6 코드/ckpt 는 그대로 두고 새 폴더에서 진행 — v7_titan 이전 단계 |
| 2026-05-20 | [cleanup_log.md](cleanup_log.md) | 삭제 파일 대장 — 무엇을 왜 지웠는지 날짜별 기록 (`SS2D_v1_analysis.md`/`eter_8gb축소.md`/`scheduler_change.md` 삭제 근거 포함) |
| 2026-05-20 | [script_version_history.md](script_version_history.md) | 삭제된 `.py` 버전(main_train_v3~v6, 옛 eval/visualize, 2024 config, 옛 dataloader) 의 출처/역할/진화 기록 |
| 2026-05-20 | [ss2d_eter_v6_1_changes.md](ss2d_eter_v6_1_changes.md) | SS2D/ETER v6_1: gradient L1 loss(λ=10) 추가 — over-sharpening 회귀 (PSNR 후퇴) |
| 2026-05-22 | [eval_metric_redesign.md](eval_metric_redesign.md) | brain mask + weighted composite metric 재설계 (D1-D4) + LR 의미 부록 + 학계 metric 표준 |
| 2026-05-25 | [ss2d_eter_v6_2_changes.md](ss2d_eter_v6_2_changes.md) | SS2D/ETER v6_2: v6_1 회귀 대응 λ_grad 10→3, v6 best 에서 재시작 — 여전히 회귀, 폐기 |
| 2026-05-25 | [tier1_tta_ensemble_negative.md](tier1_tta_ensemble_negative.md) | Tier 1: TTA(4-way flip)/앙상블 500-sample 부분평가 모두 임계 미달 — 전부 negative |
| 2026-05-25 | [tier2_sharpness_plan.md](tier2_sharpness_plan.md) | Tier 2: v6_2/v6_3/v6_4 sharpness 가설 매트릭스. v6_3 만 전지표 개선(첫 non-degradation) |
| 2026-05-28 | [visual_metric_gap_v6.md](visual_metric_gap_v6.md) | v6 정량 SSIM(0.89) ↔ 시각 인상 괴리 4원인 분석 + 진단 도구. SS2D masked SSIM/PSNR 이 ETER 보다 일관 우위 |
| 2026-05-31 | [eternet_paper_data_consistency.md](eternet_paper_data_consistency.md) | 교수님 원본 ETER-net(논문+코드) 에 명시적 Data-Consistency 블록이 없음을 확인 — 프로젝트의 DC block(v4~) 은 SS2D-arm 전용 증강이라 v7_titan ETER-vs-SS2D 비교의 confound. v8_eter_pure 의 no-DC 설계 근거 |
| **2026-05-31** | **[ss2d_v7_titan_changes.md](ss2d_v7_titan_changes.md)** | **SS2D v7_titan 재학습 정상화: scratch + true-resume(full-state last.pt, LR연속) + auto-restart supervisor + BS6 풀-step smoke. VAL_EVERY 5→2, patience 10→50, NUM_EPOCHS=50 유지(ETER 비교). 비교 baseline ETER 0.9127/0.9084** |
| **2026-06-02** | **[summary_2026-06-02.md](summary_2026-06-02.md)** | **전체 md 통합 마스터 정리 — v6(320) + v7_titan(384) 트랙 + 라이브 상태(SS2D ep7/50, ep6에서 ETER ep10 추월) 한 곳에. §6 에 문제점 12종 정리** |
| **2026-06-11** | **[summary_2026-06-11.md](summary_2026-06-11.md)** | **마스터 요약(갱신 2026-07-08: §4.6 에 v8_eter_pure 트랙 추가) — SS2D v7_titan ep50 완주 0.9127/0.9083, ETER 0.9127/0.9084 와 near-tie 확정(SSIM 동률, L1 SS2D 우위). 동일-epoch ep10~30 SS2D 우위 → ep40 ETER 재추월(교차점, 06-02 "조기 우위" 서사 정정). §4.5 head-to-head, §4.6 v8** |
| **2026-06-16** | **[version_evolution.md](version_evolution.md)** | **V4→V6→V7 버전 변천 통합본 — SS2D/ETER 하이퍼파라미터 비교표(config 검증) + 전환별 무엇/왜/결과 + 두 핵심발견(SSIM 버그·visual-metric gap) + v7 vs v7_titan 구분. SS2D v7_titan ep50 완주 0.9127/0.9083 = ETER dead-heat 반영** |
| **2026-07-05** | **[v8_eter_pure_rnn_vs_ss2d.md](v8_eter_pure_rnn_vs_ss2d.md)** | **v8: 교수님 순수 ETER-Net(no ViT, no DC)에서 GRU↔SS2D 만 교체하는 통제비교 — v7_titan dead-heat 의 confound(DC) 제거. SS2D 완승(composite 0.9200 vs 0.9182, 21×↓ params), wire-to-wire, "DC 목발" 가설 반박. 갱신 2026-07-07: per-slice paired win-rate(74~78%, p≈0) + 4-way viz(GRU 배경 ringing) 검증. **갱신 2026-07-15**: DC 축 폐기(§7 — 문헌상 비표준·GRU+DC NaN·SS2D+DC 가 GRU+DC 와 수치 등가), no-DC 가 최종. §10 관련 문헌** |
| 2026-07-08 | [v8_ss2d_kspace_domain_review.md](v8_ss2d_kspace_domain_review.md) | 외부 리뷰(ETER-net+SS2D) 판정 + SS2D 가 domain-transform 자리(k-space 입력, GRU 와 동일)임을 코드로 확정 — 리뷰어의 "지역성↔전역성 미스매치" 우려는 no-DC 완승으로 실증 기각. 항목별 판정표. 후속: 가속률(R) 일반화 성립 조건(DC·mask-aware·R-불변 정규화·전역 RF) + 단계적 방향(R cross-eval→multi-AR→operator, LMO CVPR2025 참조) |
| **2026-07-23** | **[v9_mamba_unleashed_and_radapt.md](v9_mamba_unleashed_and_radapt.md)** | **v9: v8 승자(SS2D) 강화 신규 트랙 — 게이팅+3블록+병목해제(d_inner 256/d_state 32/out_ch 64), fp16 selective-scan+ds=3 다운샘플로 v8(2.78 h/ep)보다 빠름(2.51 h/ep). unleashed(고정 R4 품질)→radapt(mask-cond+multi-AR R∈{2,3,4,5,6,8}+v8 DC 재사용, R 일반화) 80ep 순차 체인. 원본 무수정·신규 파일만 추가. launch 2026-07-21 → **unleashed 80ep 완주(07-30): best ep78 comp 0.9203/ssim_m 0.9145 로 v8 SS2D 0.9200 근소 돌파**(per-slice 7334 검증: 5지표 win-rate 54~56% 유의, vs GRU 78~82%). radapt 는 NVML 좌초 후 08-05 재기동(ETA ~08-14)** |
| **2026-08-07** | **[../paper/archive/draft_ko_v1.md](../paper/archive/draft_ko_v1.md)** | **논문 한국어 초안 v1 (교수님 상의용)** — 스코프 v8 통제비교+v9 unleashed, ETER-Net 계열(Oh2020→**Oh2025 ViT-BiRNN, J.Imaging — 문헌검색으로 확인된 교수님 신작**) 직접 후속 포지셔닝. 참고문헌 18건, 투고처 5곳 평가(1위 Bioengineering SI, 2위 Sensors), 우위 슬라이스 비율(probabilistic index)·p<0.001 표기, **결과 수치는 표준 지표만(composite 제거, 08-07 결정)**. 기준선 U-Net/E2E-VarNet 평가 도구 `v8_eter_pure/eval_paired_baselines.py` 준비(GPU 풀런은 radapt 완주 후) |
| **2026-08-20** | **[../paper/draft_ko_v2.md](../paper/draft_ko_v2.md)** | **논문 초안 v2.x (현행)** — MDPI 공학형, 스코프 v8+v9 unleashed(+radapt 3막·Transformer 팔 편입 결정 09-02), 외부검토 08-18 반영, 표는 `make_tables.py` 자동생성, `references.bib` 74건 검증 |
| 2026-08-21 | [worklog_2026-06_07.md](worklog_2026-06_07.md) | 2026-06~08 작업 일지(v8/v9 launch·좌초·재개, 논문 초안, 기준선 평가, 4-way viz 정본 슬라이스) |
| 2026-09-01 | [frontier_baselines_plan.md](frontier_baselines_plan.md) | 최전선 공개 모델 기준선 계획 — PromptMR+ fm-brain(train-only, 누수 없음)·DDS(knee prior 캐비엇)·CM-RED, 실행 순서 |
| **2026-09-01** | **[v8_fairness_followup_plan.md](v8_fairness_followup_plan.md)** | **v8 공정성 후속 실험(현행 계획)** — E1 멀티시드(seeds 0,1,2×{GRU,SS2D}×25ep)·E2 GRU LR·E3 용량(param-matched GRU 구조적 불가: H=1 하한 62.9M). **비교 기준점 = 교수님 원본 GRU** 명문화, radapt ep57 정지 → 공정성 스위트 우선 GPU 큐(~40일) |
| 2026-09-02 | [axial_transformer_arm_design.md](axial_transformer_arm_design.md) | 3번째 팔 **Transformer**(구현=axial attention, `models/attn_eternet/transformer_v10.py`, 스택 0.104M) 설계 + 4번째 팔 pixel-GRU(가중치 공유 재귀, 0.115M) — 메커니즘/파라미터화 confound 분리. diffusion 은 블록 치환 불가 근거(§6) |

> **v5 결과는 비정상 조기종료 학습** (EarlyStop 잘못된 composite 기준, ep4 에서 종료) 이라 baseline / 비교 대상에서 제외 합니다. 자세히는 [ss2d_v6_changes.md §1](ss2d_v6_changes.md) 참고.

> 본 학습 / 평가 코드의 변천 흐름은 v4 → (v5: 미완료) → v6 → v7 → v7_titan → v8_eter_pure → **v9_mamba** 로 이어집니다. v7_titan(384, ViT-Base, ETER U-Net 후처리 복원, brain mask 평가) 과 v8_eter_pure(ViT 없는 순수 ETER-Net, GRU↔SS2D 통제비교 — no-DC SS2D 완승 = 그 비교의 최종; **DC 축 폐기** — 비표준 확장, `v8_eter_pure_rnn_vs_ss2d.md` §7) 는 완주. **현재 운영(2026-09-03) 은 v8 공정성 스위트(E1 멀티시드 → Transformer·pixel-GRU 팔 → E3·E2) 이고, v9_mamba radapt 는 ep57 에서 무손실 정지 후 큐 6단계에서 재개** — 직전 운영 버전 v9_mamba — v8 승자 SS2D 를 강화(게이팅·3블록·병목해제, fp16 스캔+ds=3)해 unleashed(R4 품질)→radapt(R 일반화) 순차 학습(`v9_mamba_unleashed_and_radapt.md`).

> **2026-07-08 정리**: `SS2D_v1_analysis.md`/`eter_8gb축소.md`/`scheduler_change.md` 3개는 2026-05-20 cleanup 에서 삭제되어 이 표에서 제거함(`CLAUDE.md` 인덱스에서는 그때 이미 제거됐었음 — 이 파일만 갱신 누락). 삭제 근거는 [cleanup_log.md](cleanup_log.md) 참고.
