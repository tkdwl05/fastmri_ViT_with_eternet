# 정리 / 삭제 대장

프로젝트 디렉토리에서 삭제한 항목의 기록. 무엇이 있었고 왜 지웠는지 추후 참고용.

---

## 2026-05-20

### 1) 문서 정리 (md 파일 4개)

| 파일 | 크기 | 내용 요약 | 삭제 사유 |
|---|---|---|---|
| `docs/SS2D_v1_analysis.md` | 10.7K | SS2D-ViT v1 의 blurry 복원 7가지 원인 분석 (patch_size 32 정보손실, Conv2d 1개 합성 한계, SSIM weight 부족 등) + v2 해결 체크리스트 | v2~v5 에서 모든 문제 해결됨. 결과가 `ss2d_v4_changes.md` / `ss2d_v5_changes.md` 에 흡수 |
| `docs/eter_8gb축소.md` | 4.7K | 원본 ETER-Net (RTX 서버 384×384) 을 8GB GPU (320×320) 에 맞추는 과정 — GRU hidden 10→2, U-Net→Conv2d 1개로 축소한 historical note | 8GB 제약 자체는 현재 코드/설정에 반영됨. README 등에 GPU 제약 명시 |
| `docs/scheduler_change.md` | 2.8K | LR 스케줄러를 CosineAnnealingWarmRestarts(톱니) → CosineAnnealingLR (단일 decay) 로 교체한 근거, SS2D v2→v3 / ETER v3→v4 분리 | v4 이후 모든 버전이 이 스케줄러 사용 중. 1회성 결정 문서 |
| `presentation_script.md` | 47K | 21슬라이드 발표 대본 (v6 까지 반영, 2026-05-12) | 읽기용 `docs/presentation_overview.md` 와 중복. 대본은 발표 후 효용 종료 |

`CLAUDE.md` 의 docs/ 인덱스에서도 위 3개 bullet 제거. `presentation_script.md` 는 인덱스 없었음.

### 2) 학습 ckpt 정리 — ETER v6 중간 epoch

`logs/ETER_ViT_R4_brain320_v6/` 디렉토리에서 epoch ckpt 21개 삭제 (각 ~926MB, 총 ~19GB).

**삭제 파일 21개:**
- `eter_vit_epoch_90.pt`, `eter_vit_epoch_95.pt`
- `eter_vit_epoch_100.pt`, `..._105.pt`, `..._110.pt`, `..._115.pt`, `..._120.pt`, `..._125.pt`, `..._130.pt`, `..._135.pt`, `..._140.pt`, `..._145.pt`, `..._150.pt`, `..._155.pt`, `..._160.pt`, `..._165.pt`, `..._170.pt`, `..._175.pt`, `..._180.pt`, `..._185.pt`, `..._190.pt`

**유지:**
- `eter_vit_best.pt` (970MB) — v6_1 의 `RESUME_CKPT` 베이스. 절대 필요.
- `log.txt` (15K) — epoch별 학습 metric 텍스트 로그.

**삭제 사유:** ETER v6 학습은 200ep 도중 EarlyStop 으로 종료 (best 가 epoch 145 부근). 그 이후로는 v6_1 fine-tune 시작 — fine-tune 은 `best.pt` 에서 시작하므로 중간 epoch ckpt 는 재사용 가치 없음. epoch별 metric 추적은 `log.txt` 와 `docs/logs_archive.md` 에 보존.

**비교:** SS2D v6 는 동일 200ep 학습 후 `best.pt` 만 보존 (139MB). ETER v6 만 epoch ckpt 가 누적되어 있던 것은 train script 의 save 정책 차이.

---

### 3) 옛 학습/평가/시각화 .py 정리 (루트 + configs + dataloaders)

v6_1 가 import 하지 않는 옛 버전들. 버전 진화 방향은 [docs/script_version_history.md](script_version_history.md) 에 통합 기록.

**루트 (15개):**
- 학습: `main_train.py`, `main_train_eter.py`, `main_train_ss2d.py`, `main_train_ss2d_v4.py`, `main_train_eter_v5.py`, `main_train_ss2d_v5.py`, `main_train_eter_v6.py`, `main_train_ss2d_v6.py`, `main_train_eter_v6_resume.py`
- 평가: `eval.py`, `eval_v4.py`, `eval_unet_pretrained.py`
- 시각화: `visualize.py`, `visualize_v4.py`, `vis_v6_preview.py`

**configs 2024년 (6개) — 추후 복원됨 (2026-05-20):**
- `myConfig_choh_model3.py`, `myConfig_choh_ViT_ETER_R4regular.py`, `myConfig_choh_ViT_ETER_R4regular_v2.py`, `myConfig_choh_ViT_autoencoder_R4regular.py`, `myConfig_choh_ViT_recon_R4regular.py`, `myConfig_temp.py`
- 처음에는 "현재 v6_1 에서 import 안 됨" 으로 판단해 삭제했으나, `scripts_legacy/choh_train_*` 학습 스크립트들이 이 config 들을 import 한다는 것을 사후 확인. scripts_legacy 보존 결정과 일관성을 맞추기 위해 `git checkout HEAD --` 로 6개 모두 복원.

**dataloaders 옛 (11개 → 10개 복원, 1개만 삭제 유지):**
- 복원됨: `dataloader_h5.py` (tools/smoke_test_320.py 가 import), `myDataloader_fastmri_brain_*.py` 8개 (scripts_legacy 의 학습/테스트 스크립트들이 import), `myDataloader_temp.py` (초기 commit `7d4e4e0` — 교수님 원본 보존)
- 삭제 유지: `dataloader_h5_v4.py` — 사용자가 v4 SS2D 작업 중 작성 (`decbcb8`), 현재 어디서도 import 안 됨

처음에는 "v6_1 에서 import 안 됨" 으로만 판단했으나, scripts_legacy 와 tools/ 의 보존된 코드들이 이들을 import 한다는 것을 사후 확인하고 복원. 추가로 교수님이 초기 워크스페이스 `7d4e4e0` 에 포함한 파일은 모두 보존 결정.

**scripts_legacy/ temp (3개) — 복원됨 (2026-05-20):**
- `temp.py`, `temp2.py`, `temp3_train_chohViT.py` — 모두 초기 commit `7d4e4e0` 의 교수님 워크스페이스 일부. 사용자 작성이 아니므로 복원 유지.

**보존 (현재 활성):**
- 학습: `main_train_ss2d_v6_1.py`, `main_train_eter_v6_1.py`
- 평가/시각화: `eval_full_compare.py`, `visualize_compare.py`, `visualize_diagnostic_v6.py`
- 환경: `download_repos.py`
- configs: `_v4`/`_v5`/`_v6`/`_v6_1` (각 모델, snapshot reference)
- dataloaders: `dataloader_h5_v5.py` (v5/v6/v6_1 공유)
- scripts_legacy: temp 제외 나머지 (원본 ETER 참조용)

**삭제 사유:** v6_1 entry 인 `main_train_*_v6_1.py` 에서 import 되는 것은 `dataloader_h5_v5`, `myConfig_choh_*_v6_1`, `u_choh_model_*_v4/v5` (`models/`), `u_choh_SSIM` 뿐. 위 35개 .py 는 어디서도 import 되지 않음. 모델/dataloader/config 의 버전별 의도는 `docs/script_version_history.md` 와 기존 `*_changes.md` 가 흡수.

---

### 4) 큰 로그 / 중복 파일 / 옛 결과

**runs/eter/ 큰 로그 (3개, 약 340MB):**
- `run_eter_v4.log` (106MB), `run_eter_v6.log` (111MB), `run_eter_v6_resume.log` (122MB)
- tqdm carriage-return 누적으로 비대. epoch별 metric 은 `docs/logs_archive.md` 와 `logs/*/log.txt` 에 보존.

**기타 (3개):**
- `runs/chain/run_chain_v6_1.nohup` (185B) — `run_chain_v6_1.log` 와 내용 동일
- `presentation_script.txt` (15K) — 5/20 에 삭제한 `.md` 의 `.txt` 형제 (사용자 작성, `c09e9d4`)
- ~~`choh_train_ViT_ETER_R4regular_240916py` (확장자 누락, 9K)~~ — **복원됨 (2026-05-20)**: 초기 commit `7d4e4e0` 의 교수님 원본 파일. `scripts_legacy/` 의 `.py` 와 내용은 중복이지만 교수님 원본 보존 원칙에 따라 유지.

**results/ 옛 결과 (v4 유지, 나머지 삭제):**
- 삭제 CSV: `eval_eter_eter_vit_epoch_{40,65,70,185,190,200}.pt.csv`, `eval_ss2d_ss2d_vit_best.pt.csv`, `eval_ss2d_ss2d_vit_epoch_200.pt.csv`
- 삭제 dir: `vis_compare/`, `vis_eter_eter_vit_epoch_200/`, `vis_ss2d_ss2d_vit_best/`, `vis_ss2d_ss2d_vit_epoch_200/`
- 유지: `eval_ss2d_v4_*`, `vis_compare_v4/`, `vis_ss2d_v4_ss2d_vit_best/`, `eval_unet_pretrained*`, `eval_full_v5/v6*`, `vis_compare_v6/`, `vis_diagnostic_v6/`, `smoke_test_320/`

**기타 디렉토리:**
- `.repos_research/` — 빈 디렉토리 제거
- `__pycache__/` 6개 — Python 자동 캐시 제거 (필요시 재생성)

---

### 5) wandb 옛 run 일괄 정리 (A2)

`wandb/` 의 22 run 중 활성 1개만 남기고 21개 삭제 (약 18GB).

**유지:**
- `wandb/run-20260520_160939-co58fom3/` — SS2D v6_1 활성 학습 (현재 학습 프로세스가 실시간 기록 중)
- `wandb/latest-run`, `wandb/debug.log`, `wandb/debug-internal.log` — 위 run 가리키는 symlink

**삭제 21개:**

| 분류 | run | 학습 entry | 크기 | runtime |
|---|---|---|---|---|
| 실패/취소 (25-26s) | `0ugtbwi9` | main_train_eter.py | 164K | 25s |
| 실패/취소 (25-26s) | `p3hmpszz` | main_train_ss2d_v4.py | 68K | 26s |
| 실패/취소 (25-26s) | `wx8n6k0v` | main_train_ss2d_v4.py | 72K | 26s |
| v3 SS2D | `d2mhwqox` (4/10), `hrxgpgbq` (4/12), `hlvtltoj` (4/13), `2klw6057` (4/15), `5bhjidho` (4/21) | main_train_ss2d.py | 합 6.7GB | 20~40h |
| v3 ETER | `b0yh34j0` (4/12), `7irgep63` (4/19), `7ukvzrpe` (4/19), `t2yxtqsv` (4/22) | main_train_eter.py | 합 ~3.1GB | 0.1~43h |
| v4 SS2D | `dnsg78jb` (4/27) | main_train_ss2d_v4.py | 2.3GB | 67h |
| v5 SS2D | `2f9kjg8q` (4/30) | main_train_ss2d_v5.py | 208MB | 7h |
| v5 ETER | `0u905a19` (5/1) | main_train_eter_v5.py | 93MB | 4h |
| v6 SS2D | `wl9cne8r` (5/4) | main_train_ss2d_v6.py | 2.9GB | 90h |
| v6 ETER (시도) | `vj6fpnw6` (5/8), `tl251y4i` (5/11) | main_train_eter_v6.py | 합 1.3MB | 12~13min |
| v6 ETER (본학습) | `w9qm02dr` (5/11), `mrpcy3a1` (5/16) | main_train_eter_v6.py + resume | 합 2.7GB | 38~41h |
| v3 ETER (early) | `b4gw2f8s` (4/19) | main_train_eter.py | 6.6M | (short) |

**삭제 사유:** 모든 옛 run 은 wandb.ai 클라우드에 동기화 완료. 로컬 `run-*.wandb` 바이너리는 클라우드 데이터의 백업 사본일 뿐, 일반적 분석은 wandb.ai 대시보드에서 가능. 핵심 학습 metric 은 별도로 `docs/logs_archive.md` 와 `logs/*/log.txt` 에 보존.

**위험:** 클라우드 계정 손실 시 raw gradient histogram 복구 불가. (현재 wandb 정상 sync 작동 중이라 무시 가능)

---

### 6) 교수님 원본 파일 복원 (2026-05-20 최종 정정)

사용자 요청: "내가 생성한 파일이 아닌 교수님이 생성하셨던 파일들은 그대로 두고 싶어".

**판별 기준:** `git log --diff-filter=A --follow` 로 파일이 최초로 추가된 commit 확인. 초기 워크스페이스 commit `7d4e4e0 Initialize restructured ViT-ETER_net Workspace` 에서 추가된 파일 = 교수님 원본.

**복원 (5개, `git checkout HEAD --`):**

| 파일 | 추가 commit | 크기 |
|---|---|---|
| `choh_train_ViT_ETER_R4regular_240916py` (루트, 확장자 누락) | `7d4e4e0` | 8.9K |
| `scripts_legacy/temp.py` | `7d4e4e0` | 17K |
| `scripts_legacy/temp2.py` | `7d4e4e0` | 13K |
| `scripts_legacy/temp3_train_chohViT.py` | `7d4e4e0` | 14K |
| `dataloaders/myDataloader_temp.py` | `7d4e4e0` | 23K |

**삭제 유지 (사용자 본인 작성):**

| 파일 | 추가 commit | 사유 |
|---|---|---|
| `dataloaders/dataloader_h5_v4.py` | `decbcb8` (SS2D v4) | 사용자의 v4 작업 산물 |
| `docs/SS2D_v1_analysis.md` | `e0dc4c9` | 사용자가 작성한 분석 노트 |
| `docs/eter_8gb축소.md` | `dfc4fe5` | 사용자가 작성한 docs |
| `docs/scheduler_change.md` | `6506085` | 사용자의 refactor 노트 |
| `presentation_script.md` / `.txt` | `c09e9d4` | 사용자의 발표 자료 |

이전 §3 / §4 의 표기를 위 표 기준으로 정정했음. 향후 정리에서는 "현재 코드가 import 하느냐" 외에 "초기 commit `7d4e4e0` 에서 온 파일인가" 도 함께 확인할 것.


---

## 2026-09-03 — 프로젝트 구성 점검 (`/init`) 에 따른 정리

**원칙 재확인**: 교수님 원본 파일(초기 commit `7d4e4e0` 유래 — `scripts_legacy/`·`dataloaders/myDataloader_*`·
`configs/myConfig_choh_ViT_*`/`myConfig_temp.py`·`models/hybrid_eternet/u_choh_*`·루트
`choh_train_ViT_ETER_R4regular_240916py`(확장자 누락) 포함)은 §6(05-20) 결정대로 **손대지 않음**.
`dataloaders/myDataloader_*` 8종·옛 `myConfig_choh_ViT_*` 는 현행 코드가 import 하지 않고 `scripts_legacy/` 만
참조하지만 같은 이유로 유지.

### 1) 삭제 (tracked → `git rm`)

| 파일 | 사유 |
|---|---|
| `models/vit_pytorch/vit-pytorch-main/tests/.DS_Store` | 벤더 저장소 zip 에 딸려온 macOS 잔재(교수님 저작물 아님) |
| `.claude/settings.json` | 옛 머신 경로(`/home/snorlax-dw/…/mri_env`) 허용 규칙 2건뿐 — 이 머신에서 무의미. 로컬 설정은 `.claude/settings.local.json`(비추적) |

### 2) 삭제 (비추적 잔재)

- `v7/runs/eter/sanity.pid`, `v7/runs/extract.pid`, `v7/runs/ss2d/sanity.pid` — 2026-05 죽은 프로세스 PID
- `__pycache__/` 16개 (재생성물)

### 3) 이동·정정 (tracked)

| 변경 | 사유 |
|---|---|
| `PROJECT_SUMMARY.md` → `docs/project_summary_2026-04-11.md` | 옛 머신·루트 320 트랙 기준 04-11 스냅샷이 루트에 있어 현행 구조로 오독 유발. 제목에 역사 표기 + INDEX 등재, 참조 3곳(`presentation_overview`·`summary_2026-06-02/11`) 경로 갱신 |
| `README.md` 전면 재작성 | `main_train.py`·`download_repos.py`·`myConfig_choh_model3.py` 등 존재하지 않는 진입점 서술(04-03 작성) → 현행 트랙표 + 정본 문서 포인터 |
| `v9_mamba_radapt/runs/pre_outage_report_2026-08-07.md` 원복 + `clean_stop_report_2026-09-02_ep57.md` 신설 | 09-02 ep57 clean-stop 때 `snapshot_pre_outage.sh` 가 08-07 보고를 덮어씀(파일명 고정). 08-07 원본은 git 에서 복원, 09-02 내용은 새 파일로 분리 |
| `snapshot_pre_outage.sh` OUT 경로를 실행일 스탬프(`OUT` env 재지정 가능)로 변경 | 위 덮어쓰기 재발 방지 |
| `docs/INDEX.md` 08-07 이후 5행 추가(draft v2·worklog·frontier·fairness·transformer arm) + 현재 운영 문구 | 09-01~02 신규 문서 미등재 |
| `CLAUDE.md` 갱신(`/init`) | 표준 prefix 추가, 정체된 상태 서술 정정(radapt "08-18 재개 ETA 08-25" → 09-02 ep57 정지, "v9 화이트리스트 미추가" → 추가됨), §작업 규칙(교수님 파일 무수정·표준 지표·기준점=원본 GRU·"Transformer" 표기·GPU0 단독·활성 런 보호·git 관례) 신설, v8 4팔 env var(`SEQ_MODEL`·`SEED`…)·E1 런처·검증/스모크 명령·Docker `mri:v1`/NVML cgroup 항목 추가, 신규 docs(fairness·transformer arm·frontier·worklog) 등재 |
| `CLAUDE.md` 의 2026-06-01 루트 `results/` 역사 트리 → `docs/logs_archive.md` 부록으로 이관 | 현재 디스크에 없는 폴더 트리 28행이 운영 지침을 희석 — 기록은 옛 머신 아카이브 문서가 제자리 |

### 4) 사용자 승인 후 실행 (비가역·대용량) — 09-03 (승인: 3개 질문 전 항목)

`df` 기준 여유 **696 G → 811 G (+115 G)**. 활성 런(E1 seed0 `logs/PureETER_SS2D_noDC_R4_brain384_v8_s0/`·
`wandb/run-20260902_045022-*`·`v8_eter_pure/runs/multiseed/`)과 radapt 정지 ckpt/로그는 전·후 생존 확인, 무접촉.
자동모드 분류기가 여러 대상을 합친 단일 `rm`/`find -delete` 를 거부해 대상별 명시 `rm` 으로 분할 실행.

| 실행 | 내용 | 회수 |
|---|---|---|
| DC 축 ckpt 삭제 | `logs/PureETER_GRU_DC_R4_brain384_v8/{best,last}.pt`, `..._NAN_bak/` `.pt` 12개(best·last·epoch_5~50), `logs/PureETER_SS2D_DC_R4_brain384_v8/*.pt`. 세 폴더 `log.txt` 는 보존(8~12 KB) | ~45 GB |
| v7(320) ckpt 삭제 | `logs/{ETER,SS2D}_ViT_R4_brain320_v7/` 의 `*_best.pt`·`*.pt.hdd_baseline`·`log.txt.hdd_baseline` (`log.txt` 보존) | ~3.9 GB |
| 완주 런 중간 epoch ckpt 삭제 | `logs/PureETER_GRU_noDC_R4_brain384_v8/pure_gru_epoch_*.pt`, `logs/ETER_ViT_R4_brain384_v7_titan/eter_vit_epoch_*.pt`, `logs/SS2D_ViT_R4_brain384_v7_titan/ss2d_vit_epoch_*.pt` 30개. best/last 보존(ETER v7_titan 은 원래 last 없음). v8 SS2D noDC·v9 unleashed 의 epoch ckpt(각 125 MB급)는 **유지** | ~54 GB |
| 완주 런 tqdm 로그 gzip | `v9_mamba_unleashed/runs/ss2d/run_…unleashed….log`, `v7_titan/runs/{ss2d/run_ss2d_v7_titan_scratch,eter/run_eter_v7_titan}.log`, `v8_eter_pure/runs/{ss2d/run_…SS2D_noDC…,gru/run_…GRU_noDC…,gru/run_…GRU_DC…}.log` → `.log.gz` 6개(합 ~70 MB; SS2D DC 로그는 87 KB 라 그대로). 분석 스크립트는 `logs/<RUN>/log.txt` 만 읽음(grep 확인). **`.gitignore` 에 `*.log.gz` 추가**(gzip 이 untracked 로 노출되던 것 차단) | ~1 GB |
| 크래시 로그 백업 삭제 | `v7/runs/`·`v7_titan/runs/{ss2d,eter,chain}/` 의 `*.bak` 13개, `v7/runs/ss2d/run_ss2d_v7_nvme.log.partial_20260520`, `v9_mamba_radapt/runs/ss2d/*.log.failed-20260730`(사건은 `docs/v9_mamba_unleashed_and_radapt.md`·`host_nvml_issue` memory 에 기록, 문서 포인터 갱신) | 29 MB |
| wandb 옛 run 삭제 | `wandb/` 40개 dir — `offline-run-*` 4개(05-18 2개, **06-25 2개는 v8 noDC GRU ep31-32/SS2D ep41-42 미동기화 세그먼트 — 내용 확인 후 `logs/*/log.txt` 와 중복으로 판단**) + online `run-*` 전부(클라우드 보존). 잔존: 활성 E1 `run-20260902_045022-…_v8_s0` + `latest-run`·debug 로그 (173 MB) | ~8 GB |
| `paper/draft_ko_v1.{md,docx}` → `paper/archive/` | `git mv`. 참조 갱신: `docs/INDEX.md`·`paper/project_story_v1_to_v9.md`(3곳)·`paper/draft_ko_v2.md` 기반 표기·`docs/worklog_2026-06_07.md`·`CLAUDE.md` | — |
| 루트 320 스크립트 11개 → `legacy_320/` | `git mv`: `main_train_{ss2d,eter}_v6_{1,2,3}.py`·`eval_full_compare.py`·`eval_tta_ensemble.py`·`visualize_compare.py`·`visualize_compare_versions.py`·`visualize_diagnostic_v6.py`(전부 사용자 작성, `7d4e4e0` 아님). 각 파일 `current_dir` 1행을 상위 폴더(저장소 루트)로 수정 → `python legacy_320/<script>.py` 로 import 경로 성립(ast 파싱 + 6개 `myConfig_*` import 스모크 PASS; ckpt 부재로 실행은 불가). `legacy_320/README.md` 신설, `CLAUDE.md` 실행/레이아웃·`docs/script_version_history.md` 추기. Python 측 참조는 주석/docstring 뿐(`v7/eval_v7_compare.py:4`, `configs/myConfig_choh_*_v6_1.py:4`). 루트 잔류: 현행 `visualize_v7_titan_compare.py`·`visualize_v8_pure_compare.py`·`visualize_v9_compare.py`·`visualize_eval_modes_compare.py`·`visualize_slices_canonical.json`, 교수님 `choh_train_ViT_ETER_R4regular_240916py` | — |

### 5) 잔여 후보 (미승인 — 다음 정리 때 판단)

| 후보 | 크기 | 비고 |
|---|---|---|
| `logs/SS2D_ViT_R4_brain384_v7_titan/_ddp_archive/` | 1.3 GB | 05-31 폐기된 DDP 시도(`log.txt`·`log_ddp.txt`·best·epoch_5/10). 이번 승인 목록에 없어 보존 |
