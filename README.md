# fastMRI ETER-Net — 시퀀스 모델 치환 연구 (GRU · SS2D/Mamba · Transformer)

fastMRI brain multicoil(혼합 contrast, R=4 기본) 재구성. 교수님 원본 **ETER-Net**
(k-space → 양방향 GRU 도메인 변환 → aliased image 와 concat → U-Net DFU) 의 **시퀀스 모델 슬롯만
교체**해 무엇이 달라지는지 통제비교하는 것이 현재 연구 축이다.

| 트랙 | 내용 | 상태 (2026-09-03) |
|---|---|---|
| `v8_eter_pure/` | 원본 ETER-Net(ViT 없음, no-DC) 에서 **GRU ↔ SS2D** 단일변수 통제비교 + 공정성 스위트(멀티시드·3/4번째 팔 Transformer·pixel-GRU·용량/LR) | 본편 완주(SS2D 우위) · **E1 멀티시드 진행 중(GPU0)** |
| `v9_mamba_unleashed/` | v8 SS2D 강화(게이팅·3블록·병목해제) R4 품질 | 80ep 완주 · 검증 완료 |
| `v9_mamba_radapt/` | 같은 백본 + R 일반화(mask-cond·DC·multi-AR) | ep57/80 에서 무손실 정지 — 공정성 스위트 후 재개 |
| `v7_titan/`, `v7/` | ViT-Base + ETER/SS2D 하이브리드 (384 / 320) | 완료·역사 (dead-heat) |
| 루트 `main_train_*_v6_x.py`, `configs/` | ViT-Small 320 트랙 (옛 8GB 머신) | 역사 — 이 머신에 ckpt 없음 |

- **작업 안내(Claude Code 용 정본)**: [`CLAUDE.md`](CLAUDE.md) — 구조·실행·규칙·결정 사항.
- **문서 인덱스(날짜순)**: [`docs/INDEX.md`](docs/INDEX.md). 최신 계획: `docs/v8_fairness_followup_plan.md`.
- **논문 트랙**: `paper/draft_ko_v2.md` (+ `make_tables.py` 로 표 자동 생성, `references.bib`).
- **환경**: Docker `mri:v1` (`infra/docker/RUNBOOK.md`), conda `base`, PyTorch 2.3.1 + mamba_ssm 2.2.2, TITAN RTX 24GB — **GPU0 단독**.
- 교수님 원본 파일(`scripts_legacy/`, `dataloaders/myDataloader_*`, `models/hybrid_eternet/u_choh_*`, 루트
  `choh_train_ViT_ETER_R4regular_240916py`)은 **무수정·무삭제 원칙** — 새 기능은 신규 파일로만 추가한다.

보고·평가 지표는 표준 지표(brain-masked SSIM 주지표 + PSNR/NMSE/L1)만 사용한다(composite 금지).
