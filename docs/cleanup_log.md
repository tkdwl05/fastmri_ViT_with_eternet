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

