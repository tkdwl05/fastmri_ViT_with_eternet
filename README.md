# fastMRI ETER-Net — 시퀀스 모듈 치환 연구 (GRU · SS2D/Mamba · Transformer)

fastMRI brain multicoil(혼합 대조도(contrast), R=4 기본) 재구성. 교수님 원본 **ETER-Net**
(k-space → 양방향 GRU 도메인 변환 → zero-filled 영상(에일리어싱 포함)과 concat → U-Net DFU) 의 **시퀀스 모듈만
교체**해 무엇이 달라지는지 통제 비교하는 것이 현재 연구 축이다.

| 트랙 | 내용 | 상태 (2026-09-28) |
|---|---|---|
| `v8_eter_pure/` | 원본 ETER-Net(ViT 없음, no-DC) 에서 **GRU ↔ SS2D** 통제 비교 + 통제 비교 보강 실험(다중 시드·3/4번째 비교 모델 Transformer·pixel-GRU·용량/LR) | 주 실험 완료(SS2D 우위, 시드 미고정 단일 학습) · E1 다중 시드 25ep 완료(시드 간 차이의 부호가 일관되지 않음 — 두 모델 동등) · **1b 50ep seed 1 재현 검증 진행 중(GPU0)** — 진행 상태는 `docs/v8_fairness_followup_plan.md` |
| `v9_mamba_unleashed/` | v8 SS2D 강화(게이팅·3블록·채널 폭 확대) R4 품질 | 80ep 학습 완료 · 검증 완료 |
| `v9_mamba_radapt/` | 같은 백본 + 다른 가속화 계수에 대한 일반화(mask-cond·DC·다중 가속화 계수 학습) | ep57/80 에서 epoch 경계 계획 정지 — 통제 비교 보강 실험 후 재개 |
| `v7_titan/`, `v7/` | ViT-Base + ETER/SS2D 하이브리드 (384 / 320) | 완료·역사 (v7_titan 사실상 동등) |
| `legacy_320/`, 루트 `configs/` | ViT-Small 320 트랙 (이전 8GB 머신) | 역사 — 이 머신에 ckpt 없음 |

- **작업 안내(Claude Code 용 단일 기준 문서)**: [`CLAUDE.md`](CLAUDE.md) — 구조·실행·규칙·결정 사항.
- **문서 인덱스(날짜순)**: [`docs/INDEX.md`](docs/INDEX.md). 최신 계획: `docs/v8_fairness_followup_plan.md`.
- **논문 트랙**: `paper/draft_ko_v2.md` (MDPI 초안, `make_tables.py` 로 표 자동 생성, `references.bib`) · IEIE 초안 작업 기준본 `paper/ieie/draft_ieie_v7.src.md`(빌더 2종 → 학술지판/학술대회판 docx).
- **환경**: Docker `mri:v1` (`infra/docker/RUNBOOK.md`), conda `base`, PyTorch 2.3.1 + mamba_ssm 2.2.2, TITAN RTX 24GB — **GPU0 단독**.
- 교수님 원본 파일(`scripts_legacy/`, `dataloaders/myDataloader_*`, `models/hybrid_eternet/u_choh_*`, 루트
  `choh_train_ViT_ETER_R4regular_240916py`)은 **무수정·무삭제 원칙** — 새 기능은 신규 파일로만 추가한다.

보고·평가 지표는 표준 지표(brain-masked SSIM 주 평가 지표 + PSNR/nMSE/L1)만 사용한다(composite 금지).
