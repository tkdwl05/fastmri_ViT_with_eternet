# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

fastMRI brain multicoil 데이터에 대한 MRI 재구성 모델 연구
(대조도(contrast)는 "AXFLAIR" 단일이 아니라 AXT1/AXT1POST/AXT1PRE/AXT2/AXFLAIR 등 **혼합** — "AXFLAIR" 로만 표기하면 부정확).

교수님 원본 ETER-Net(k-space bi-GRU → U-Net) 의 **시퀀스 모듈만 치환**해 비교하는 연구다. 초기(v1~v7_titan)는
ViT 인코더 + 시퀀스 모듈(GRU=ETER 또는 SS2D=Mamba) 하이브리드였고(역사), `v8_eter_pure` 갈래부터 ViT 를 빼고
순수 ETER-Net 위에서 시퀀스 모듈만 바꾸는 통제 실험이 본류다 — GRU vs SS2D(완료) → 2026-09 부터 Transformer·pixel-GRU
모델을 더한 **네 모델 통제 비교 보강 실험(현재 운영)**. `v9_mamba` 갈래(unleashed/radapt)는 v8 최고 성능 모델(SS2D)을 강화해 R4 품질
(unleashed, 학습 완료)과 다른 가속화 계수에 대한 일반화(radapt, ep57 에서 epoch 경계 계획 정지 — 통제 비교 보강 실험 후 재개)를 미는 트랙이다.

### 트랙

| 트랙 | 해상도 / 구성 | 상태 |
|---|---|---|
| v1~v6_x (`legacy_320/`, 루트 `configs/`) | 320, ViT-Small, 이전 RTX 5060Ti 8GB·`mri_env` | 역사 — 이 저장소엔 ckpt 없음, 재현 불가 |
| v7 → v7_titan | 384, ViT-Base + GRU/SS2D 하이브리드 | 완료 — ETER/SS2D ep50 사실상 동등 |
| **v8_eter_pure** | 384, ViT 없음, 시퀀스 모듈 4종 | **현재 운영** — GRU vs SS2D no-DC 주 실험 완료(SSIM_m 0.9126 vs 0.9140, 시드 미고정 단일 학습; DC 요인 폐기). **1b(seed 1 50ep) 재현 안 됨: 볼륨 SSIM GRU 0.9136 vs SS2D 0.9133 → 두 모델 차이가 학습 회차 간 변동과 구분되지 않음**(10-02). **U-Net 단독(시퀀스 모듈 제거, seed 1) 0.9127 — 시퀀스 모듈 기여 ≤0.0015 로 런 간 변동 수준**(10-04). 통제 비교 보강 실험 큐 진행 중 |
| v9_mamba unleashed / radapt | 384, ViT 없음, 강화 SS2D | unleashed 80ep 완료(SSIM_m 0.9145) · radapt ep57/80 정지(재개 = 큐 6단계) |

**진행 상태·큐·ETA·E1 결과표의 단일 출처 = `docs/v8_fairness_followup_plan.md`** (여기엔 복사하지 않는다 — 금방 낡음).
큐 요지: E1 다중 시드(25ep×3시드, 09-18 완료 — 시드 간 차이의 부호가 일관되지 않음, 두 모델 동등) → 1b 50ep seed 1 쌍 GRU→SS2D(10-02 완료 — Δ 재현 안 됨, 학습 회차 간 변동과 구분되지 않음,
런 폴더 `_s1_50ep`) → U-Net 단독 모델(시퀀스 모듈 제거, SEED=1, 10-04 완료 — 볼륨 SSIM 0.9127, IEIE 표 1 행) → **▶ pixel-GRU(SEED=1, 10-05 10:00 UTC 실행 시작)** → Transformer → E3 최소-GRU → E2 LR 스윕 → radapt 재개 → 추론-only 일괄.
각 단계 완료 후 다음 단계는 **수동 실행 시작**(자동 체인 없음 — E1 후 2.5일 유휴 재발 방지).

## 작업 규칙 (사용자 결정 — 이후 모든 작업에 적용)

- **교수님 원본 파일 무수정**: 초기 커밋 `7d4e4e0` 에서 들어온 파일(루트 `choh_train_ViT_ETER_R4regular_240916py` — 확장자 없음이 의도적 복원 상태, `scripts_legacy/*`, `dataloaders/myDataloader_*`, `configs/myConfig_choh_ViT_*`·`myConfig_temp.py`, `models/hybrid_eternet/myUNet_DF.py`·`u_choh_*`, vendored `models/mae`·`models/vit_pytorch`)은 삭제·이동·수정하지 않는다(`docs/cleanup_log.md` §6, 2026-05-20). 확장은 **새 파일 추가**로만(v8/v9 가 그렇게 했음). 출처 확인: `git log --follow --diff-filter=A --format=%h -- <path> | tail -1` 이 `7d4e4e0` 이면 원본. 프로젝트 공유 원본(`ss2d.py`·`u_choh_model_SS2D_ViT_v4.py`·`dataloader_h5_v5.py`)도 v9 이후 무수정 관례.
- **지표는 표준 지표만**(SSIM 주 평가 지표 + PSNR/nMSE/L1). 자체 설계 composite 은 보고·비교·문서에서 인용 금지(2026-08-07) — 인용 수치는 v8 SS2D 0.9140 / GRU 0.9126, v9 unleashed 0.9145(슬라이스 단위; **IEIE 초안은 2026-09-04 부터 fastMRI 관례대로 볼륨 단위 0.9141 / 0.9127 / 0.9146 을 대표 수치로 사용** — 순위·유의성은 두 단위에서 동일, `paper/tables/ieie_table1_block.md` 첫 블록). 단, 실행 중·정지 중 트레이너의 best-ckpt 선택 기준(composite)은 학습 무결성을 위해 소급 변경하지 않는다(`log.txt` 의 `val_composite` 열은 그래서 남아 있음 — 읽되 인용하지 말 것).
- **비교 기준 모델 = 교수님 원본 ETER-Net(GRU)**. SS2D 는 "현재 최고 성능 비교 모델"이지 기준 모델이 아니다 — 1차(주) 비교는 "각 모델 vs 원본 GRU", 모델 간 pairwise 는 2차(2026-09-02).
- **비교 모델 이름**: 3번째 비교 모델은 문서·표·env 에서 **"Transformer"** 로 표기(구현 세부 "axial attention" 은 설계문서 안에서만). 4번째 비교 모델은 "pixel-GRU". **'팔(arm)'이라는 용어는 쓰지 않는다**(2026-09-28 사용자 결정) — 네트워크 전체는 '(비교) 모델', 교수님 원본은 '기준 모델', 바꿔 끼우는 부분은 '시퀀스 모듈'(구성·후보), 실험 한 칸은 '실험 조건'. 코드 식별자(`ARMS` env, `--arm`, 파일명 `axial_transformer_arm_design.md`)는 인터페이스라 유지.
- **용어**: 원고·문서·메모리는 `docs/terminology_glossary.md`(2026-10-01 확정) 규칙을 따른다. 교수님 문구와 겹치는 보류 9건은 현행 유지.
- **v1~v6(이전 8GB 머신, 320)** 결과는 384 서버 트랙의 근거·비교 대상으로 쓰지 않는다(역사 기록 전용). v5 는 비정상 조기종료라 baseline 에서도 제외.
- **GPU0 단독 순차 큐** — GPU1 은 전 큐에서 사용하지 않는다(교수님 작업용). 큐 순서를 바꾸면 `docs/v8_fairness_followup_plan.md` 의 큐 표부터 갱신.
- **실행 중 런의 코드 편집 금지**: GPU0 런이 도는 동안 그 런의 트레이너·공유 config·런처 `*.sh` 를 고치지 않는다 — 트레이너/config 는 supervisor 재시작 시 새 코드로 반영되고, bash 는 실행 중 스크립트를 이어서 읽는다.
- **활성 런 보호**: 정리·이동·삭제 시 진행 중 런 폴더(`logs/PureETER_*_v8_s*/`, `v8_eter_pure/runs/multiseed/`, 활성 `wandb/run-*`)와 radapt 정지 ckpt(`logs/PureETER_SS2D_V9_radapt_multiAR_brain384/`)는 건드리지 않는다. 파일 삭제는 `docs/cleanup_log.md` 에 날짜별로 기록.
- **git**: `*.sh` 는 기본 무시(예외 = `.gitignore` 화이트리스트: `infra/docker/*.sh`·`v9_mamba_radapt/runs/{post_reboot_rearm,clean_stop_pre_outage,snapshot_pre_outage}.sh`) — 런처 스크립트는 로컬 전용이므로 재현 절차는 문서/CLAUDE.md 에 적는다. 커밋 prefix 관례 `paper:`/`docs:`/`v8:`/`infra:`/`viz:`/`feat:`. 작업 브랜치 `docs/summary-2026-06-02`(main 보다 앞섬, 주기적으로 main 에 merge). 인증은 SSH(§환경).

## 문서 지도 (docs/)

전체 날짜순 목록·한 줄 요약은 **[docs/INDEX.md](docs/INDEX.md)** — 새 문서를 추가하면 INDEX 에 행을 추가한다. 현행 작업에 필요한 진입점만:

- **현행 계획·큐**: `docs/v8_fairness_followup_plan.md` (E1/E2/E3·기준 모델 명문화·GPU 큐 표). 3·4번째 비교 모델 설계 `docs/axial_transformer_arm_design.md`. 공개 모델 기준선 `docs/frontier_baselines_plan.md`(리더보드(leaderboard) U-Net/VarNet 은 학습·검증 통합 데이터(train+val)로 학습했으므로 순위 비교 없이 참고 결과로만).
- **v8 결과·근거**: `docs/v8_eter_pure_rnn_vs_ss2d.md`(GRU↔SS2D 통제 비교, per-slice paired 검증, DC 요인 폐기 §7, 문헌 §10), `docs/eternet_paper_data_consistency.md`(원본 ETER-net 에 DC 없음 → no-DC 설계 근거), `docs/v8_ss2d_kspace_domain_review.md`(외부 리뷰 판정, radapt 설계 근거).
- **v9**: `docs/v9_mamba_unleashed_and_radapt.md`. radapt 정지 상태 보고 `v9_mamba_radapt/runs/clean_stop_report_2026-09-02_ep57.md`, 재개 절차 `v9_mamba_radapt/runs/RESUME_AFTER_OUTAGE.md`.
- **지표**: `docs/eval_metric_redesign.md`(brain mask = Otsu×0.4 + largest CC, `dataloader_h5_v5.py:243`; composite 부분은 ⚠ 역사).
- **용어 규칙**: `docs/terminology_glossary.md`(2026-10-01 확정 — 표기 규범·원고 표현·은어 교체·사실 오류 정정, 교수님 문구와 겹치는 보류 9건·공통 예외·바꾸지 않는 표현).
- **역사 요약**: `docs/summary_2026-06-11.md`(v6+v7_titan 마스터), `docs/version_evolution.md`(V4→V7 하이퍼파라미터·두 핵심발견: custom SSIM 버그·정량 지표와 시각 품질의 괴리), `docs/presentation_overview.md`(v1~v6_3 발표용), `docs/worklog_2026-06_07.md`(6~8월 일지).
- **정리 대장**: `docs/cleanup_log.md`(삭제 기록), `docs/script_version_history.md`(삭제된 .py 출처).

## paper/ (논문 트랙) — 커밋 prefix "paper:"

- `paper/draft_ko_v2.md`/`.docx` — 한국어 투고 초안 v2.1(MDPI 공학형, 스코프 v8+v9 unleashed). `references.bib` 서지 전건 확정. 주장 범위(09-02 잠금): "SSM>RNN" 일반화가 아니라 **"SS2D 치환 > 원 bi-GRU 설계"로 한정** + 메커니즘/파라미터화 교란 요인 한계 항목. IEIE v7 초록의 "세 지표 모두 우위" 는 1b 결과 후 유지/완화 결정(09-21). 역사: `paper/archive/draft_ko_v1.*`, 개발 이력 문서 `paper/project_story_v1_to_v9.md`.
- **`paper/make_tables.py` — 표를 md+tex 로 자동 생성(`paper/tables/`). 수치가 바뀌면 표를 손편집하지 말고 재실행.** 그림 `make_fig{1_architecture,2_curves,3_qualitative,4_per_slice}.py`·`make_figs_conf_arch.py` → `paper/figs/`.
- **그림 조판 규칙**: 빌더가 그림을 page 폭 6.69 in(학술대회판 단 폭 3.15 in)에 맞춰 삽입하므로 figsize 폭을 그 값으로 **직접 조판**(더 넓게 그리면 글자가 비례 축소), 최소 글자 6.5 pt, Liberation Sans(`paper/fonts/`), 600 dpi. 고전적 논문 블록 다이어그램 문법(한 줄 라벨 균일 블록·화살표 위 크기 표기·실데이터 썸네일)이고 양식 도우미의 단일 출처는 `make_figs_conf_arch.py`. `check_docx_structure.py` 는 dpi 만 보고 글자 크기는 못 보므로 렌더 PNG 를 눈으로 확인.
- **`paper/ieie/` — IEIE 투고 초안. ★v8(10-05) = 최신 통합본** `draft_ieie_v8.src.md`: 교수님 10-01 검토본(추적 변경 수락)+본문 수정 목록 전체+1b·U-Net 단독 결과, 표 1 = `paper/tables/ieie_table1_block_v8.md`(make_tables.py 생성 — run 1·run 2·U-Net only, 공개 모델 행 없음). 학술대회판 ≈5.2~5.5쪽(5쪽 초과 가능 → 감축 라운드). 아래 v7 설명은 이전 판 기준: 단일 소스 `draft_ieie_v7.src.md` → `build_ieie_docx.py`(학술지판, 서면 심사용 blind 4쪽 — `check_blind()`) / `build_ieie_conf_docx.py`(학술대회판, 학술대회 논문집 1~5쪽, 저자 `***` 자리표시자 유지). 두 판은 내용 동일·저자만 차이. 저장소 루트에서 `CUDA_VISIBLE_DEVICES="" python paper/ieie/build_ieie_*.py [--src …] [--out …]`. 2026 추계학술대회 마감 **2026-10-19**, MDPI SI 마감 11-30.
  - **소스 규칙**: 인라인 수식 `$…$`(빌더가 OMML 변환 — 평문 `y_c`·`f_θ` 첨자 금지, `\mathbb` 미지원), 디스플레이 `$$…$$` 한 줄, 캡션 **`@cap_en` 필수(두 판 영문만, 09-14)**·`@cap_ko` 선택, 표 수치는 `paper/tables/ieie_table1_block.md`·`ieie_table_ref_block.md` 그대로(볼륨 단위).
  - **문체 규칙(교수님 09-14 검토)**: 구어체 금지, 문장 성분(주어·목적어·서술어·조사) 완성, 주어–서술어 일치, 수식의 모든 변수를 말로 설명, 약어는 첫 등장에서 풀어 씀(RSS·ACS·SSIM·PSNR·nMSE·MSE·fp16/32), 구성요소=모듈·네트워크 전체=모델.
  - **`paper/ieie/reviews/` 는 git-ignore — 교수님 실명·소속·e-mail 포함. push 금지, 추출 텍스트를 추적 파일·커밋 메시지에 붙이지 말 것.**
  - **2026-09-28 현재 작업 방식**: 제1저자가 교수님 코멘트 v6 docx 를 Word 에서 직접 수정 중(수정 가이드 `paper/ieie/reviews/2026-09-28/v6_edit_guide.md`) — 그 docx 가 작업 원본이고 `draft_ieie_v7.src.md` 는 수정 문장의 출처(참조)로만 쓴다. 학술지판(같은 src 에서 빌드) 동기화 방식은 미정 — docx 확정 뒤 src 역반영 여부를 사용자에게 확인.
  - 쪽수 초과(학술지판 Word 실측 7쪽 > 4쪽) 감축은 별도 라운드. 빌드·그림·점검기 기댓값·쪽수·Word 체크리스트 상세 = `docs/paper_table_conventions.md` §4~§7.

## 모델 구조

### 코드 레이아웃
- `models/pure_eternet/` — v8·v9 순수 ETER wrapper(`u_pure_eternet_{gru,ss2d,transformer,pixelgru}.py`, 시퀀스 모듈을 뺀 대조 모델 `u_pure_eternet_unet.py`(시퀀스 출력 자리 20채널을 0 으로 채워 U-Net 은 동일), v9 `u_pure_eternet_ss2d_v9{,_radapt}.py`). 시퀀스 모듈 본체는 `mamba_eternet/`(`ss2d.py`·`ss2d_v9.py`, DC block 정의는 `u_choh_model_SS2D_ViT_v4.py`), `attn_eternet/transformer_v10.py`(Transformer 모델), `rnn_eternet/pixelgru_v10.py`(pixel-GRU 모델), `hybrid_eternet/`(교수님 원본 `myUNet_DF.py` U-Net DFU + v7_titan ETER-ViT).
- `dataloaders/dataloader_h5_v5.py` — 공유 로더(brain mask `:243`, v5~v9 전부 상속). v9 radapt 는 `dataloader_h5_v9_multiAR.py`(per-sample R∈{2,3,4,5,6,8}).
- 루트 `visualize_*_compare.py` — 트랙별 4-way 시각화(`visualize_slices_canonical.json` 고정 대표 슬라이스). `visualize_multimodel_compare.py` 는 8-way 정성 비교(논문 그림 3) — **CPU 전용 설계**(SS2D 는 `selective_scan_ref` 런타임 몽키패치, PromptMR+ 는 `external/PromptMR-plus` 어댑터): `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 19 python visualize_multimodel_compare.py`.
- `tools/`(환경 점검·스모크), `infra/docker/`(이미지 재구성 00~50 + `RUNBOOK.md`), `external/`(외부 레포 clone, git-ignore), `legacy_320/`(320 트랙 스크립트, 실행 불가·역사 보존).

### v8_eter_pure (384×384, ViT 없음) — 시퀀스 모듈 4종 순수 통제 비교

```
입력 1: zero-filled 영상(aliased) (B, 32, 384, 384)
입력 2: k-space        (B, 32, 384, 384)
       │                                    │
  (ViT 없음)               GRU(양방향 h+v) | SS2D | Transformer | pixel-GRU
       │                                    │
       └──────── cat(seq출력, zero-filled 영상) ────┘   ← 2-way concat
                      │
         UNet_choh_skip (DFU, depth=5, wf=6)
                      │  (use_dc=True 인 DC 조건만: 이 뒤에 DC block 추가 — DC 요인 폐기, no-DC 만 최종)
              출력: (B, 1, 384, 384)
```
시퀀스 모듈을 제외한 모든 것이 100% 동일(출력 채널 out_ch=2×H2=20 원본 상수) — 통제 비교.
치환 모델(SS2D·Transformer·pixel-GRU)의 시퀀스 모듈 파라미터 규모는 ~0.1M 로 서로 매칭(가중치 공유 파라미터화의 최소 공통 규모, total ≈31.2M);
원본 GRU 모듈은 flatten-reshape 구조 때문에 H=1 에서도 31.9M(total 62.9M)이 하한이라 하향 매칭이 불가능하다(주 실험 H=10 은 668M).

### v9_mamba (384×384, ViT 없음) — v8 SS2D 강화 + 다른 가속화 계수에 대한 일반화

v8 no-DC SS2D 파이프라인(concat → U-Net DFU, 384·R4·brain-mask·masked loss)을 그대로 물려받되 **시퀀스 모듈만 강화 SS2D 로 교체**
(`models/mamba_eternet/ss2d_v9.py`): 게이팅 추가 `y = y·SiLU(z)`(v8 이 누락한 Mamba 게이트), 채널 수를 유지하는 잔차 블록 3개,
채널 폭 확대 out_ch 20→64·d_inner 128→256·d_state 16→32·dropout 0.05, fp16 selective-scan + ds=3 다운샘플(128² 격자에서의 스캔) → epochs 50→80.
**총 34.2M params**(U-Net 31.1M + SS2D 모듈 3.1M; 논문 "~34M"). h/ep 확정 실측값(08-18 실측, 검증 포함) = **GRU 2.41 / v8 SS2D 3.07 / v9 2.84** —
문서의 2.51/2.78 은 스모크 추정치라 논문에 쓰지 않는다.

- **unleashed** = 고정 R4 품질 극대화(mask/DC 없음).
- **radapt** = 같은 백본 + 다른 가속화 계수에 대한 일반화 3요소: (1) mask-channel conditioning(`cat(x_ksp, mask)`, c_in 32→33), (2) v8 DC block 재사용(U-Net n_classes=2 복소 → 1-iter soft DC → magnitude), (3) 다중 가속화 계수 학습(R∈{2,3,4,5,6,8}, val 은 R4 고정). R-embedding/FiLM 없음. DC fp16 안정화: α clamp[0,1] + GradScaler init_scale 8192 + NaN-skip(v8 DC ep4 NaN 재발 방지).

### v7_titan / 루트 320 (역사)
v7_titan 은 ViT-Base 하이브리드로, **ETER 와 SS2D 의 최종 합성 구조 자체가 달랐다**(ETER = `UNet_choh_skip(depth=3, wf=6)` 원본 복원,
SS2D = RefinementBlock + DC) — 이 비대칭이 `docs/eternet_paper_data_consistency.md` 의 교란 요인이고 v8 이 이를 제거했다.
루트 320 트랙은 ViT 출력·zero-filled 영상·seq 출력의 3-way concat → RefinementBlock(SS2D 만 DC 추가). 상세는 `docs/architecture_ETER_vs_SS2D.md`·`docs/version_evolution.md`.

## 설정

- **v8**: `v8_eter_pure/configs/myConfig_pure_eter_v8.py` — 모든 모델·런이 공유하는 단일 config. 분기는 `v8_eter_pure/main_train_pure_v8.py` 의 env var: `SEQ_MODEL=gru|ss2d|transformer|pixelgru|unet`(unet = 시퀀스 모듈 제거 대조 모델, 10-01), `USE_DC=0|1`(DC 폐기 → 0), `SEED=<int>`(설정 시 가중치 초기화·셔플·마스크 offset·flip·워커별 rng 고정 + 런 폴더 `_s{SEED}` 접미; 미설정 시 기존 런과 동일 코드 경로), `RUN_SUFFIX`, `SANITY_NUM_EPOCHS`·`SANITY_VAL_EVERY_N_EPOCHS`, `SMOKE_BS`, `ACCUM_STEPS`, 모델별 `TRANSFORMER_{D_MODEL,N_HEADS,N_PAIRS}`·`PIXELGRU_HIDDEN`, `WANDB_RUN_TAG`. 런 폴더 = `logs/PureETER_{ARM}_{noDC|DC}_R4_brain384_v8[_s{SEED}][RUN_SUFFIX]/`.
- **v9**: env var 가 아니라 **변형별 config 파일** — `v9_mamba_unleashed/configs/myConfig_ss2d_v9.py`(BS 8·80ep·LR 2e-4·WD 3e-5·patience 40·val 매 2ep) / `v9_mamba_radapt/configs/myConfig_ss2d_v9_radapt.py`(+ `MASK_CONDITION`·`AR_CHOICES`·`VAL_ACCELERATION=4`·DC 안정화).
- 역사: `v7_titan/configs/`, 루트 `configs/myConfig_choh_{SS2D,ETER}_model_v4~v6_4.py`(v6_3 = 320 트랙 채택후보).

## 로그 · ckpt · 결과 위치

- **ckpt·per-epoch 요약** = `logs/<RUN_NAME>/`: `log.txt`(한 줄/epoch — 분석 스크립트는 이것을 읽지 거대한 tqdm `runs/*.log` 를 읽지 않는다), v8 `pure_<arm>_{last,best,epoch_N}.pt`, v9 `ss2d_v9_{last,best,epoch_N}.pt`(last = 전체 상태 재개(true-resume)). v9 RUN_NAME = `PureETER_SS2D_V9_unleashed_R4_brain384` / `..._radapt_multiAR_brain384`, 완료 표시 파일 = `logs/<RUN_NAME>/DONE`.
- **tqdm·supervisor 로그** = `<track>/runs/`: v8 은 모델축 기준 `runs/{gru,ss2d,chain}/`·`runs/multiseed/`(E1·1b 런 로그), v9 는 각 `runs/ss2d/`. 완료된 런의 tqdm 로그는 gzip(`zcat` 으로 열람). 런처 `*.sh` 는 git-ignore(로컬 전용).
- **결과** = 저장소 공통 `results/`: `eval/v8_nodc/`(per-slice paired·우위 슬라이스 비율·matched-epoch), `eval/v8_nodc_s1_50ep/`(1b seed 1 짝 평가), `eval/v8_unet_only/`(U-Net 단독, `eval_unet_only_v8.py` — `--seq pixelgru|transformer` 로 단일 비교 모델도 같은 프로토콜 평가 → `eval/v8_{pixelgru,transformer}/`), `eval/v8_r_sweep{,_norm}/`(stride-4 서브샘플 — 논문 표 불가, 큐 7단계에서 전체 val 재실행), `eval/v9_unleashed/`, `eval/baselines_384_full/`(공개 모델 전체 val 본 연구 프로토콜 행: U-Net† 0.8971 / E2E-VarNet† 0.9181 / PromptMR+ 0.9417, 볼륨 SSIM), `eval/zero_filled/`, `vis/{v7_titan_compare,v8_pure_eternet_compare,v9_unleashed_compare,multimodel_compare}/`.
- **`.gitignore`**: `results/` 는 기본 무시, 요약 md·PNG 만 화이트리스트(per-slice CSV·log·ckpt·npz 는 계속 무시). 새 결과 폴더를 공유하려면 디렉터리 + 파일 패턴 두 줄(`!results/eval/<dir>/` 와 `!results/eval/<dir>/<file>`)을 함께 추가.

## 실행

모든 명령은 저장소 루트에서. 학습은 `python ...` 그대로(conda `base`, activate 불필요).

### v8_eter_pure (현재 운영)
```bash
# 단일 런 (전체 상태 재개 supervisor): SEQ_MODEL=gru|ss2d|transformer|pixelgru|unet, USE_DC=0
SEQ_MODEL=pixelgru USE_DC=0 CUDA_VISIBLE_DEVICES=0 bash v8_eter_pure/runs/run_pure_v8_autoresume.sh

# U-Net 단독 50ep SEED=1 (10-04 03:31 UTC 완료 — 평가 results/eval/v8_unet_only/; 1b 런처를 ARMS=unet 으로 재사용, 완료 런 skip·미완 런 전체 상태 재개)
SEED=1 ARMS=unet SMOKE_BS=8 CUDA_VISIBLE_DEVICES=0 setsid nohup bash v8_eter_pure/runs/run_v8_seed1_50ep.sh \
  > v8_eter_pure/runs/multiseed_outer_unet_s1_50ep.log 2>&1 < /dev/null & disown
# 진행 확인: tail -2 logs/PureETER_UNET_noDC_R4_brain384_v8_s1_50ep/log.txt ; nvidia-smi
# ★ 현재 단계: pixel-GRU 50ep SEED=1 (10-05 10:00 UTC 실행 시작; 같은 런처 ARMS=pixelgru — 재기동도 같은 명령)
SEED=1 ARMS=pixelgru SMOKE_BS=8 CUDA_VISIBLE_DEVICES=0 setsid nohup bash v8_eter_pure/runs/run_v8_seed1_50ep.sh \
  > v8_eter_pure/runs/multiseed_outer_pixelgru_s1_50ep.log 2>&1 < /dev/null & disown
# 진행 확인: tail -2 logs/PureETER_PIXELGRU_noDC_R4_brain384_v8_s1_50ep/log.txt ; nvidia-smi   (완료 후 다음 단계 Transformer 는 수동 실행 시작)
#   ⚠ 런처는 끝나도 init 이 회수하지 않아 좀비로 남는다 — 종료 감시는 kill -0 이 아니라 `ps -o stat=` 의 Z 를 확인
# 1b(완료, 10-02): 같은 런처 기본값 ARMS="gru ss2d" → logs/PureETER_{GRU,SS2D}_noDC_R4_brain384_v8_s1_50ep/, 짝 평가 results/eval/v8_nodc_s1_50ep/
# E1 다중 시드 런처(완료): v8_eter_pure/runs/run_v8_multiseed.sh — SEEDS × ARMS × EPOCHS env, 완료 런 skip

# per-slice paired 평가 (전체 val, ~2h) + 4-way 시각화
python v8_eter_pure/eval_paired_v8_nodc.py
python visualize_v8_pure_compare.py

# 공개 모델 기준선 전체 val (CPU·재개 가능 — 완료 슬라이스 skip; 처리 순서 interleaved = 완료 prefix 가 항상 전 볼륨에 고르게 분산된 부분 집합(계통 추출))
CUDA_VISIBLE_DEVICES="" nice -n 19 python v8_eter_pure/eval_baselines_full.py --methods promptmr --threads 5 --num-workers 1
CUDA_VISIBLE_DEVICES="" python v8_eter_pure/eval_baselines_full.py --summary   # → baseline_summary_full.md
```

### v9_mamba
```bash
# ★ radapt 재개 (큐 6단계 — GPU0 에 다른 런이 없을 때만): NVML 확인 + supervisor 전체 상태 재개
bash v9_mamba_radapt/runs/post_reboot_rearm.sh
# epoch 경계 계획 정지 (last.pt 갱신 감시 → SIGTERM → 상태 보고 저장)
DEADLINE_OVERRIDE=<epoch초> bash v9_mamba_radapt/runs/clean_stop_pre_outage.sh
# 개별 supervisor: v9_mamba_unleashed/runs/run_ss2d_v9_autoresume.sh / v9_mamba_radapt/runs/run_ss2d_v9_radapt_autoresume.sh
# (역사 진입점: v9_mamba_unleashed/runs/run_v9_chain_gpu0.sh — unleashed→DONE→radapt 체인)
```

### 역사 트랙
v7_titan: `bash v7_titan/runs/run_ss2d_v7_titan_autoresume.sh`, `python v7_titan/main_train_eter_v7_titan.py`, `python visualize_v7_titan_compare.py`.
320 트랙: `legacy_320/README.md`(ckpt 부재로 이 머신에서 재현 불가).

### 검증 · 스모크 (pytest 스위트 없음 — 스크립트 단위)
```bash
python tools/check_recon_env.py                          # torch/mamba_ssm/CUDA/데이터 의존성 점검 (Docker 재구성 후엔 infra/docker/50_verify_env.sh 도)
python v8_eter_pure/sanity_pure_v8.py                    # v8: 모델 간 "시퀀스 모듈만 다름" 계약 + forward 형상 + 교수님 파일 무수정 검증
python v8_eter_pure/sanity_smoke_test_pure_v8.py         # v8: full train-step VRAM 스모크 → runs/smoke_bs.txt (supervisor 가 SMOKE_BS 로 주입)
CUDA_VISIBLE_DEVICES="" python v8_eter_pure/sanity_unet_only_v8.py   # U-Net 단독: U-Net 이 다른 모델과 동일 + 0 채널 기울기 0 (CPU; --gpu --bs 8 = VRAM·속도 스모크, smoke_bs.txt 미기록)
CUDA_VISIBLE_DEVICES="" python v9_mamba_unleashed/sanity_ss2d_v9.py   # v9: 구조·게이팅·no-WD·원본 무수정 (CPU 가능; forward 는 CUDA 시만)
python v9_mamba_radapt/sanity_ss2d_v9_radapt.py          # radapt: 마스크 조건화 + DC + 다중 가속화 계수 로더
python v9_mamba_unleashed/smoke_v9.py                    # v9 두 변형 VRAM/속도 스모크 → 각 runs/smoke_bs.txt
python paper/make_tables.py                              # 논문 표 재생성 (수치 변경 시 필수, 손편집 금지)
python paper/ieie/check_docx_structure.py                # IEIE docx OOXML 구조 점검(기본 v7; 기대 FAIL 0 — 렌더 확인은 Word)
```
GPU 를 쓰는 스크립트는 GPU0 에서 학습이 도는 중이면 실행하지 않는다(CPU 가능한 것은 `CUDA_VISIBLE_DEVICES=""`).
새 비교 모델을 추가할 때는 `sanity_pure_v8.py` 의 계약 검사(U-Net in_channels/depth/wf 동일, 시퀀스 모듈 param 수)를 통과시킨 뒤 스모크로 배치 크기(BS)를 확정한다.

## 환경

- conda 환경 **`base`**(`/opt/conda`) — `mri_env` 는 이전 머신 이름이며 이 머신에 없다.
- GPU: **TITAN RTX 24GB × 2** — 정책상 **GPU0 단독**, GPU1 은 항상 비워둔다.
- 컨테이너: Docker 이미지 **`mri:v1`**(`infra/docker/Dockerfile`, 2026-08-31 재구성 — 절차 `infra/docker/RUNBOOK.md`, 스크립트 00_backup→10_preflight→20_build→30_run→40_restore→50_verify). torch 2.3.1 / numpy 1.26.4 는 mamba_ssm 2.2.2 wheel ABI 때문에 동결.
- **컨테이너 재시작 후 재개 = 큐 표의 *현재 단계* 런처 재실행**(현재 pixel-GRU: `SEED=1 ARMS=pixelgru SMOKE_BS=8 … run_v8_seed1_50ep.sh`, 완료 런 skip·전체 상태 재개 — 위 §실행). `post_reboot_rearm.sh` 는 GPU0 점유를 확인하지 않고 radapt 를 띄우는 radapt 전용 스크립트라 큐 6단계 전에는 쓰지 않는다 — `infra/docker/RUNBOOK.md` [7] 은 아직 radapt 기준으로 적혀 있다.
- **NVML/`/dev/nvidia0` EPERM 반복 재발**: 원인은 cgroup device allowlist 탈락(드라이버·이미지 문제 아님) → `30_host_run.sh` 가 `--device` 를 명시. 실행 중 CUDA 컨텍스트는 살아남지만 **새 프로세스 실행 시작이 실패**하므로 장기 런은 `MAX_RETRY=200` supervisor 로 띄운다.
- wandb `WANDB_MODE=online`(미동기 `offline-run-*` 는 sync 안 된 것).
- git 인증 **SSH**(`git@github.com` remote, `~/.ssh/id_ed25519`). push 실패 시 `ssh -T git@github.com` 부터 확인, remote URL 에 credential 임베드 금지.
