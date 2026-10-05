# 저장소 점검 보고 (2026-09-28)

읽기 전용 감사 5개 영역(docs·memory·학습 코드·논문 파이프라인·git/infra) → 영역별 반박 검증 → 종합. 제안만 기록했고 수정은 하지 않았다(CLAUDE.md·README·INDEX·계획서 1b 행·fig3 명령은 같은 날 별도 반영).

## 높음

**1. 재시작 후 복구 절차가 큐 6단계인 radapt 를 먼저 띄운다. GPU0 가 비었는지는 확인하지 않는다** (git-infra·docs·train-code)
- 근거
  - `v9_mamba_radapt/runs/post_reboot_rearm.sh:15-22` 는 radapt 가 도는지만 pgrep 으로 본다.
  - `infra/docker/RUNBOOK.md:227-231`, `infra/docker/50_verify_env.sh:94`, `v9_mamba_radapt/runs/RESUME_AFTER_OUTAGE.md:11` 이 모두 이 스크립트를 "학습 재개" 명령으로 안내한다.
  - 지금 진행 중인 런은 1b SS2D 다(ep18/50, `docs/v8_fairness_followup_plan.md:128`).
  - 컨테이너 CMD 가 `sleep infinity` 라서 재시작 뒤 자동으로 재개되는 런은 없다.
  - 그 밖에: 08-07 타이머(:24-33)는 이미 지나 매번 건너뛰는 죽은 코드다. 런북의 컨테이너 이름 `snorlax_WORK0` 는 hostname 이고 실제 이름은 `mri_gpu0` 이다.
- 제안
  - rearm 에 GPU0 점유 가드를 넣는다(`nvidia-smi --query-compute-apps` 또는 `pgrep -f 'main_train_pure_v8|main_train_ss2d_v9'`). radapt 는 `FORCE_RADAPT=1` 을 줬을 때만 띄운다. 08-07 타이머 블록은 지운다.
  - RUNBOOK [7], 50_verify, 런북의 안내를 "큐 표의 현재 단계 런처 재실행(지금은 `run_v8_seed1_50ep.sh`)"으로 바꾼다.
  - 컨테이너 이름을 `mri_gpu0` 으로 고치고, ETA 는 "잔여 23ep × 2.84 h" 로 적는다.

**2. IEIE v7 정본과 빌더·점검기·그림 수정분이 커밋되지 않았다. 검토본을 막는 ignore 규칙도 커밋되지 않았다** (paper·git-infra)
- 근거
  - `?? paper/ieie/draft_ieie_v7.*` 5개(09-15). 수정 파일 22개(+364/−210). HEAD 238eb3a(09-09)가 origin 과 같으므로, 그 뒤 작업은 모두 로컬 디스크에만 있다.
  - 커밋된 빌더는 기본 SRC 가 v2(`build_ieie_docx.py:40`)이고, 작업트리 빌더는 v7 을 가리킨다.
  - reviews/ 를 무시하는 규칙(`.gitignore:114`, 실명이 담긴 파일 11개)이 HEAD 에 없다. 로컬 권한은 `git add/commit/push *` 를 모두 허용한다.
- 제안
  - ① `.gitignore` 의 reviews 3줄을 먼저 단독으로 커밋한다(`.git/info/exclude` 에 같이 넣는 것도 검토).
  - ② v7 src/md/docx 두 판, 빌더 2개, 점검기, 다시 만든 그림을 파일 이름을 하나씩 지정해 한 커밋으로 올린다. `git add paper/ieie` 는 쓰지 않는다. 쓰면 v3 docx 와 v2 수정분이 섞여 들어간다.
  - ③ v2(작업트리 수정분을 버릴지 커밋할지 결정)와 v3 는 archive/ 로 옮기고, `archive/README.md` 에 "정본 = draft_ieie_v7.src.md" 를 적는다.
  - ④ 커밋 메시지에 검토본 내용을 인용하지 않는다.

**3. 매 세션 자동 로드되는 MEMORY.md 인덱스가 너무 크고, 한 줄 안에서 서로 모순된다** (memory)
- 근거: `memory/MEMORY.md:10` 한 줄이 2,193자로 파일의 약 40% 다. 같은 괄호 안에 "push 완료"와 "push 보류, 11 커밋 앞섬"이 함께 있는데, 실측으로는 ahead 0 이다. :3·:12·:17·:26 도 긴 줄이다.
- 제안: 항목마다 150자 안팎의 한 줄로 줄인다. 커밋 해시, push 상태, 쪽수, ETA 처럼 바뀌는 정보는 본문 파일에만 둔다.

## 중간

### 다음 큐 단계 launch 전에 고칠 것 (1b 완주 ~10-02 뒤, 2단계 pixel-GRU 전)

**4. 환경점검이 실패해도 프로세스가 rc 0 으로 끝나 런처가 "완주"로 오판한다** (train-code)
- 근거
  - `v8_eter_pure/main_train_pure_v8.py:196-197` 은 점검 실패 시 예외 없이 `return` 한다.
  - `run_v8_seed1_50ep.sh:31-33` 과 `run_v8_multiseed.sh:32-34` 는 rc 0 을 완주로 본다.
  - 해당 경우: 데이터 마운트 누락, mamba_ssm 손상, config import 실패. NVML 불능은 :194 에서 raise 하므로 해당하지 않는다.
- 제안: 점검 실패 시 `sys.exit(2)` 로 끝낸다. 완주할 때 `logs/<RUN>/DONE` sentinel 을 쓰고, 런처는 DONE 이나 log.txt 의 `Epoch N/N` 으로 완주를 판정한다.

**5. CLAUDE.md 가 새 비교 모델 예시로 드는 `run_pure_v8_autoresume.sh` 가 E1·1b 런처와 관례가 다르다** (git-infra·train-code)
- 근거
  - :26 MAX_RETRY 기본값이 50 이다(관례는 200).
  - :14 RUN 이름에 SEED/RUN_SUFFIX 가 반영되지 않는다.
  - :16 transformer·pixelgru 로그가 `runs/gru/` 에 섞인다.
  - :29-30 alloc conf 와 WANDB_MODE 설정이 없다.
  - :32 누적 로그를 "학습 완료"로 grep 하므로, 같은 모델을 다시 돌리면 완주로 오탐한다.
- 제안: 당장은 이미 파라미터화된 1b 런처를 `SEED=… ARMS=pixelgru` 로 재사용한다. 이렇게 하면 MAX_RETRY 200 과 `Epoch 50/50` 판정이 그대로 적용된다. 이후 supervisor 를 DONE 판정, `SUBDIR=$SEQ_MODEL`, RUN_SUFFIX 반영, 기본 200 으로 정비한다.

**6. 계획서가 본편을 "50ep seed0"이라고 잘못 부르고 있어, 2·3단계를 어떤 SEED 경로로 띄울지 정해지지 않았다** (docs·train-code)
- 근거
  - 본편은 SEED 인프라(9421b86, 09-02)가 들어오기 전의 무시드 런이다(log.txt 첫 줄 SCRATCH, 폴더에 `_s` 접미사 없음).
  - 그런데 계획서 `docs/v8_fairness_followup_plan.md:47,109,128-130` 은 이를 "seed0" 으로 적는다. IEIE v7 은 "시드 미고정"으로 정확하게 쓰고 있다.
  - 무시드 경로에서는 persistent 워커 16개가 같은 rng 를 복제해 같은 flip·마스크 스트림을 낸다(`dataloaders/dataloader_h5_v5.py:162`, worker_init_fn 없음, v9 트레이너 포함). 효과는 작다.
- 제안
  - 본편 표기를 "무시드 단일 런"으로 고친다.
  - 2·3단계는 SEED 를 지정해 띄우는 것을 권장한다. launch 명령에 SEED 와 런 폴더명을 적고, 1b 해석에 "시드 경로 vs 무시드 경로" 차이를 한 줄 남긴다.
  - v8 무시드 분기와 v9 트레이너 두 개에 worker_init_fn 을 추가한다. radapt 재개 전에 넣어도 무결성에는 영향이 없다.

**7. SEED 경로로 재개하면 DataLoader generator 가 SEED 로 다시 초기화되어, 재개 후 에폭이 ep1 의 셔플·증강 스트림부터 다시 재생된다** (train-code·git-infra)
- 근거
  - `main_train_pure_v8.py:242-244` 가 프로세스 시작마다 generator 를 재시드한다. last.pt 에는 전역 rng 만 저장된다(:450-457). CPU 로 재현했다.
  - 지금까지 시드 런에는 RESUME 이 없어 E1 과 1b GRU 는 영향을 받지 않았다. 실행 중인 1b SS2D 가 재시작되면 해당된다.
  - 부수 문제: NaN fail-fast(FATAL, rc 1) 뒤에도 런처는 같은 last.pt 에서 최대 200번 재시도한다(`run_v8_seed1_50ep.sh:29-36`).
- 제안: 재개 시 epoch 에 따라 재시드한다(예: `_g.manual_seed(SEED + 1009*start_epoch)`). 다음 런처부터는 `tail -1 log.txt | grep ^FATAL` 이 연속으로 나오면 중단하게 한다.

**8. 큐 4·5단계(E3 최소-GRU, E2 LR 스윕)는 "확정"으로 적혀 있지만 실행할 경로가 없다** (docs·train-code·git-infra)
- 근거
  - 트레이너에 LR 과 GRU hidden env override 가 없다(`main_train_pure_v8.py:109,217`, config `:22-23,:36` 은 공유 상수).
  - RUN_SUFFIX 없이 띄우면 기존 완주 폴더의 last.pt 로 full-resume 한다. 그러면 0 스텝으로 "학습 완료"(rc 0)가 되고, 기준 log.txt 와 wandb run 이 오염된다(:203-206, :266-268, :319). SEED=0 으로 띄우면 `_s0` 폴더(E1 25ep last.pt)와 충돌한다.
  - H=1 이면 U-Net 입력이 20ch 에서 2ch 로 바뀐다(`u_pure_eternet_gru.py:69`). 이는 계획 :79 의 "원본 상수" 서술과 충돌한다.
  - 계획 :110 의 `ARMS="… axial"` 은 트레이너 assert(:46)에 걸린다.
- 제안
  - 트레이너에 `LR`, `GRU_H1`/`GRU_H2` env 훅을 새로 넣는다. 값이 기본과 다르면 RUN_SUFFIX 를 자동으로 붙이고, SCRATCH 줄과 wandb config 에 기록한다.
  - 재개 시 last.pt 의 설정이 현재와 다르면 거부한다.
  - E3 는 "H1 만 축소(U-Net 입력 불변)"와 "H1=H2 축소" 중 무엇으로 할지 문서에서 정한다.
  - 계획서 상태를 "결정됨 — 미구현"으로 바꾸고 :110 의 axial 을 transformer 로 고친다. 공유 config 는 직접 고치지 않는다.

**9. 새 비교 모델(Transformer·pixel-GRU)이 sanity·스모크 검증 대상에 없다. 스모크를 문서대로 돌리면 git 이 추적하는 smoke_bs.txt(8)가 4 로 덮어써진다** (train-code)
- 근거
  - `sanity_smoke_test_pure_v8.py:127` 기본 BS 후보가 (4,2,1)이다. :130-132 는 gru/ss2d × DC 만 돌리고, :143-146 이 가능한 BS 중 최소값을 기록한다. 런처 3종이 이 파일을 읽는다.
  - `sanity_pure_v8.py:68-69` 에는 스택 파라미터 수 검사가 없고, :9 에는 `mri_env` 가 남아 있다.
- 제안: 모델 목록을 env(ARMS)로 받고 DC 는 기본에서 뺀다. 파일 기록은 `--write` 옵션일 때만 하고, 후보는 (8,6,4,2,1)로 한다. sanity 에 스택 파라미터(~0.1M) assert 를 넣는다. 새 비교 모델의 BS8 VRAM 확인은 1b 뒤 GPU 로 한다.

**10. 7단계 R-sweep 은 "준비 완료"로 적혀 있지만, 평가 파이프라인은 GRU/SS2D 만 로드하고 composite 를 1순위로 출력한다** (train-code)
- 근거
  - `eval_paired_v8_nodc.py:78-93` build_model 은 gru 가 아니면 전부 SS2D 로 만든다. `eval_r_generalization_v8.py:117-118` 도 두 모델만 로드한다. 모델 빌더는 8개 파일에 복제돼 있다.
  - `eval_r_generalization_v8.py:149` 가 "## composite (핵심)" 섹션을 맨 앞에 만든다.
  - 계획서 :134·:137 은 "준비 완료", "세 모델 + Zero-filled" 로 적는다.
- 제안: 신규 파일 `models/pure_eternet/build_v8.py` 에 팩토리를 두고, eval_paired 를 `--arm name=ckpt` 로 여러 모델을 받는 구조로 바꾼다(기준 모델 GRU). 다시 돌리기 전에 composite 섹션을 빼고 SSIM 을 1순위로 둔다. 계획서 상태는 "스크립트 확장 필요"로 바꾼다.

### 문서·초안 정합

**11. E1 결과(25ep 에서 부호 불안정)와 1b 진행 상황이 주 결과 문서와 초안에 반영되지 않았다** (docs·paper)
- 근거
  - 다음 문서들이 "완승·wire-to-wire"로 단정하면서, 공정성 계획으로 가는 링크는 0건이다: `docs/v8_eter_pure_rnn_vs_ss2d.md:3`, `docs/summary_2026-06-11.md:13,29,180,188`, `docs/presentation_overview.md:622`, v9 문서 §11.
  - `paper/draft_ko_v2.md:7,620-621,677,692` 는 여전히 "09-02 launch·진행 중"이다.
  - `paper/ieie/draft_ieie_v7.src.md:21,24,93` 의 "세 지표 모두 우수"에는 다시 검토하라는 표시가 없다.
  - 1b GRU seed1 50ep 은 SSIM_m 0.9135 로, 본편 GRU 0.9126 보다 +0.0009 높다. 판정이 뒤집힐 수 있는 상태다.
- 제안
  - 주 문서 상단과 v9 §11 에 주의 상자를 넣는다: "무시드 단일 런이다. E1 에서 부호가 불안정했다 → fairness plan 참조. 1b 판정은 ~10-02." summary 와 presentation 에는 포인터 한 줄만 넣는다.
  - draft_ko_v2 의 상태 문구를 갱신한다.
  - v7 src 21·24·93행에 `%% 1b 결과 후 유지/완화 결정` 주석을 달아 10-19 마감 전에 다시 보도록 한다.

**12. 운영 현황 문서들이 09-03 이전 상태에 머물러 있다** (docs)
- 근거
  - `docs/v9_mamba_unleashed_and_radapt.md:3` 이 "학습 미시작"이다. §11.4 의 ETA 는 08-13/14 이고 ep57 정지가 빠져 있다.
  - `docs/INDEX.md:43` 에 E1 완주와 1b 가 없다. :32 는 h/ep 스모크 추정치(2.51/2.78)와 radapt ETA 08-14, :37 은 "~40일", :34 는 bib 74건(실제 75건)이다.
  - `docs/v8_fairness_followup_plan.md:139` 는 "10월 중순 종료"다. 실측으로 다시 계산하면 1b SS2D 2.93 h/ep 로 ~10-02 에 끝나고, 2~8단계 23~25일을 더해 ~10-25~27 에 끝난다. :4-5 와 :112 에는 폐기된 "radapt ~09-04 완주 후" 전제가 남아 있다.
- 제안
  - radapt 연표를 적는다: 08-05 → 08-07 ep20 → 08-18 → 08-21 ep43 → 09-02 ep57. 정지 보고서 링크를 함께 넣는다.
  - h/ep 는 정본 2.84/3.07, 큐 종료는 ~10-27, bib 는 75로 고친다. SI 11-30 집필 여유가 줄어든다는 점을 명시한다. IEIE 10-19 에는 영향이 없다.

**13. 학습 로그의 PSNR/nMSE/L1 은 검증 배치 단위(val BS 4)로 모은 값인데, v8 평가 docstring 은 "로그와 근접해야 정상"이라고 쓴다** (train-code)
- 근거: `v8_eter_pure/eval_paired_v8_nodc.py:5-7` 과 `v9_mamba_unleashed/eval_paired_v9.py:10-15` 가 서로 반대로 설명한다. SS2D PSNR 은 로그 35.16, per-slice 33.90 이다. SSIM 만 양쪽 0.9140 으로 같다.
- 제안: v8 docstring 을 고친다. radapt 36.49 같은 로그 PSNR 에는 "batch-pooled" 라고 명시하고, 논문 표 값과 섞지 않는다.

### 논문 산출물

**14. 쪽수 점검이 규정 한도와 비교하지 않아, 학술지판 6.7쪽(규정 4쪽)도 PASS 로 나온다** (paper)
- 근거: `paper/ieie/check_docx_structure.py:473` 은 `rep.ok` 만 호출한다. `journal` 인자는 쓰이지 않는다(:337, :497-498). 쪽수 추정은 학술대회 빌더(:303, :376)에 별도로 또 있다.
- 제안: journal 4쪽, conf 5쪽을 넘으면 WARN 을 낸다. 추정 구현은 하나로 합치고 v6 Word 실측값(7쪽/5쪽)으로 계수를 보정한다.

**15. v7 에 들어가는 그림 1·2 의 글자가 4.7~4.9 pt 로, 문서화된 최소 6.5 pt 보다 작다. 1:1 로 삽입되므로 그대로 인쇄된다** (paper)
- 근거
  - `paper/make_figs_conf_arch.py:77-78` 은 FS=6.5 로 정의한다. 그런데 fig1 은 :244-296 에서 4.8~4.9 pt, fig2 는 :368-412 에서 4.7~4.8 pt 를 쓴다. :107 자동 축소 하한은 4.6 pt 다.
  - `make_fig3_qualitative.py:37` 도 5.5~6.0 pt 다.
  - 기준 문서는 `docs/paper_table_conventions.md:61-62` 다.
- 제안: 규칙을 하나로 정한다(예: 라벨 ≥6 pt, 주석 ≥5 pt). 그 값에 맞춰 fs 와 축소 하한을 올리고, 넘치는 라벨은 문구를 줄인다.

**16. fig3_qualitative_col 재현 명령에 `--slices` 가 빠져 있어, 그대로 실행하면 7행 그림이 v7 그림을 덮어쓴다** (paper)
- 근거: `docs/paper_table_conventions.md:98`. `make_fig3_qualitative.py:63-64,79` 의 기본값은 3슬라이스라 7행이 된다. 현재 PNG 는 3행이다.
- 제안: 명령에 `--slices 3368 --gain-slice 3368` 을 넣는다. CLAUDE.md 에 있는 같은 명령도 고친다.

**17. git 이 추적하는 `paper/draft_ko_v2.docx`(09-02 판)에 정정 전 수치 "~33M" 이 남아 있다** (paper)
- 근거: docx 안에 2건이 있다. md 는 이미 34.2M 으로 고쳐졌다(`draft_ko_v2.md:223,394`). 이 환경에는 pandoc 과 빌드 스크립트가 없다.
- 제안: pandoc 이 있는 환경에서 docx 를 다시 만든다. 어렵다면 md 상단에 "docx 는 09-02 판 — 인용 금지"를 표시하거나 docx 를 추적에서 뺀다.

### 메모리 본문

**18. 일지 형태의 메모리가 현재 상태와 모순되는 옛 지시를 담고 있다** (memory)
- 근거
  - v7 계열 7개 파일(31KB): `v7_titan_train_resume.md:28-31` 에 tmux 체인 실행 지시, `v7_nvme_migration.md:22-35` 에 옛 재개 명령이 있다. `v7_titan_brain_mask.md:3` 은 "진행 중"인데 :26 은 "완료"다.
  - `dataset_storage_layout.md:25-36` 은 학습 링크를 HDD 로 적는다. 실제로는 NVMe(ext4)의 `fastmri_data_nvme` 를 가리킨다. How to apply 는 폐기된 "HDD 에 두고 링크" 방식을 권한다.
  - `host_nvml_issue.md:3,52-67` 은 5월의 docker restart 처방을 그대로 두고 있어, 같은 파일 :11-22 의 결론(cgroup 문제, `--device` 명시)과 모순된다.
  - `v9_mamba_status.md:59,73` 에 철회된 h/ep 2.51/2.78 이 남아 있다.
  - `v8_eter_pure_status.md:48-60` 에는 옛 "현재 방식" 재시작 명령과, 크기 관계가 반대인 옛 속도 수치가 있다. 1b GRU 결과 0.9135 는 빠져 있다.
  - `paper_draft_status.md:94` 는 archive 로 옮겨진 `paper/draft_ko_v1.md` 를 가리킨다. "draft_ko_v2.md:231 미수정"은 이미 수정됐다.
- 제안: v7 계열 7개는 `v7_history.md` 하나로 합친다. 나머지는 20줄 안팎의 현황 메모와 docs 단일 출처 포인터로 다시 쓰고, 재개 명령과 "다음 세션 할 일"은 지운다. 이렇게 하면 메모리 전체가 약 161KB 에서 30KB 안팎으로 줄 것으로 예상된다.

## 낮음

- **radapt 보고 생성기가 composite 를 대표 수치로 쓴다.** `snapshot_pre_outage.sh:15,28,30` 에 composite 와 unleashed 0.9203 이 하드코딩돼 있다. 그 결과 `clean_stop_report_2026-09-02_ep57.md:9,11` 에 틀린 기대 문장("unleashed 보다 낮은 것이 정상")이 들어갔다. best.pt 는 ep56 저장분이고, SSIM_m 이 가장 높은 ep54 의 가중치는 남아 있지 않다. → 보고를 best SSIM_m(epoch)과 PSNR 기준으로 바꾸고 "best.pt = composite 선택(ep56)"을 병기한다. R-sweep 에 쓸 ckpt 를 명시한다. (docs·git-infra)
- **`clean_stop_pre_outage.sh:17` 의 기본 DEADLINE(08-07)이 이미 지났다.** override 없이 실행하면 epoch 경계를 기다리지 않고 바로 kill 한다. 로그 파일명도 하드코딩돼 있다(:16). → override 를 필수로 하거나 기본값을 "지금 + 6h" 로 바꾼다. radapt 재개 전에 고친다. (git-infra)
- **`infra/docker/50_verify_env.sh:61` 의 ckpt 로드 검사가 unleashed `ss2d_v9_last.pt` 만 잡는다.** `RUNBOOK.md:63` 의 radapt 파일명과 epoch 도 틀렸다(실제는 `ss2d_v9_radapt_last.pt`, ep57). → glob 을 `*_last.pt` 로 넓히고 가장 최근 파일을 검사한다. (git-infra)
- **git 이 추적하는 rearm 스크립트가 추적되지 않는 `run_ss2d_v9_radapt_autoresume.sh` 를 호출한다**(`.gitignore:105-108`, rearm :19). → 화이트리스트에 추가한다. (git-infra)
- **`infra/docker/old_container_inspect.json` 은 우리 옛 컨테이너가 아니라, 08-31 오추출 때 잡힌 교수님 GPU01 설정이다**(비밀정보 없음). → `git rm --cached` 로 추적에서 빼고 ignore 에 추가한다. (git-infra)
- **`.gitignore:2` 의 `fastMRI_data/` 가 심볼릭 링크에 매칭되지 않는다**(check-ignore rc=1). → `/fastMRI_data` 와 `/fastMRI_data_hdd` 로 바꾼다. `v7_titan/runs/` 의 txt 3개도 추적하거나 ignore 한다. (git-infra)
- **v9 sanity 두 개가 ds 인자를 넘기지 않아 ds=1 모델을 검증한다**(`v9_mamba_unleashed/sanity_ss2d_v9.py:31-38`, radapt :32-41). → `ss2d_downsample=C.SS2D_DOWNSAMPLE` 를 넘기고 34.2M assert 를 넣는다. (train-code)
- **코드와 주석이 맞지 않는 곳들(코드는 그대로 두고 주석만 고친다).** (train-code)
  - dt_proj.bias 는 실제로 weight decay 를 받는데, 주석은 제외된다고 쓴다(`ss2d_v9.py:61,63` vs `main_train_ss2d_v9.py:6,98`, config 주석, v9 문서 :116).
  - `eval_zero_filled_v8.py:12` 는 논문 표에 ls 를 쓴다고 하지만 실제는 raw 다.
  - `myConfig_pure_eter_v8.py:2-5` 와 트레이너 docstring 이 "2×2 ablation" 시절 그대로다. `path_folder`/`ckpt_prefix` 는 호출하는 곳이 없다.
  - `eval_paired_v8_nodc.py:4` 의 슬라이스 수 7270 은 7334 로 고쳐야 한다. :134-136 의 0워커 고정 사유는 07-08 에 해소됐다.
- **논문 표·빌더 문구가 낡았다.** (paper·docs)
  - `paper/make_tables.py:534` REF_NOTE 의 "Table 2" 를 "Table 1" 로 고치고 :467·:470 주석도 갱신한 뒤 재실행한다(수치는 바뀌지 않음).
  - `build_ieie_conf_docx.py:3,22,323` 과 `build_ieie_docx.py:1037` 의 "학술대회 2쪽" 을 "1~5쪽" 으로 고치고 재빌드한다.
  - `docs/paper_table_conventions.md:48` 의 PromptMR+ `[TBD]` 를 09-08 확정으로 바꾼다.
- **v7 표 1(7행, 단 폭, "SS2D (enhanced)" 라벨)을 만드는 생성기가 없다**(`draft_ieie_v7.src.md:70-82`). 지금 수치는 모두 일치한다. 판 번호 기본값은 세 파일에 따로 있고, 빌더 간 DrawingML 코드도 복제돼 있다. → 1b 로 표가 바뀌기 전에 make_tables.py 에 v7 블록 출력을 추가하거나, 최소한 라벨을 통일한다. 공용 상수와 도우미를 둔다. (paper)
- **PDF 를 다시 렌더할 때마다 CreationDate 만 바뀐 diff 가 생긴다**(`make_figs_conf_arch.py:172` 외 make_fig*.py). → 저장 시 `metadata={'CreationDate': None}` 을 준다. 쓰지 않는 conf_fig3_enhanced 는 생성 목록에서 빼거나 "미사용"으로 표시한다. 현재 수정 상태(M)인 PDF 2개를 되돌릴지 정한다. (paper·git-infra)
- **`docs/v8_fairness_followup_plan.md:39` 의 E1 SS2D PSNR 이 로그와 다르다.** 34.80±0.08 을 34.82±0.05 로 고친다. (docs)
- **`v9_mamba_unleashed/analyze_v9_unleashed.py:28` 이 폐기된 h/ep 2.51/2.78 을 쓰고, 동률일 때 첫 epoch(ep72)을 고른다**(best ckpt 는 ep78). `results/eval/v9_unleashed/` 는 composite 중심에 ep72 기준이라, 화이트리스트에 넣기 전에 표준 지표로 다시 만들어야 한다. `.gitignore:76-78` 의 없는 폴더 baselines_384 항목은 지운다. (paper·git-infra)
- **죽은 링크.** legacy_320 이동으로 깨진 3곳(`docs/error_map_v2_masked.md:3,15`, `docs/visual_metric_gap_v6.md:55`)은 `../legacy_320/…` 로 고친다. 없는 세션 계획 파일 링크(`v9_mamba_unleashed_and_radapt.md:7,261`, `v8_ss2d_kspace_domain_review.md:88,122`)는 "(세션 계획 파일, 보존 안 됨)"으로 표기한다. (docs)
- **메모리 잔여 정리.** (memory)
  - 죽은 위키링크 4종(`v9_mamba_status.md:54` 는 `[[no-composite-metric]]` 로 고치는 등)을 정리한다.
  - 없는 plan 파일 인용 13종 중, 현재 근거로 쓰이는 `v9_mamba_status.md:66` 만 docs 포인터로 바꾼다.
  - `container_recreate_bigmem.md` 는 삭제하고 shm 관련 사실만 docker_env_rebuild 로 옮긴다. `docker_env_rebuild.md:17` 의 "미커밋"은 "커밋됨"으로 고친다.
  - `feedback_no_composite.md:27-29` 의 "radapt 진행 중"과 미결 문구를 갱신한다. `dataset_brain_contrast.md:10` 의 틀린 전제를 지운다. VarNet NaN 팁은 `baseline_leaderboard_leakage.md` 본문으로 옮긴다.
- **디스크 정리 후보.** 완주한 E1 GRU 3런(`logs/PureETER_GRU_noDC_R4_brain384_v8_s{0,1,2}/`)의 epoch_*.pt 와 last.pt 가 약 64GB 다(best.pt 와 log.txt 는 유지). 1b 판정 뒤에는 `_s1_50ep` GRU 의 약 35GB 도 후보다. 사용자 확인 후 지우고 cleanup_log 에 기록한다. 여유 공간이 695G 라 급하지 않다. (git-infra·docs)
- **`/root/.claude` 는 컨테이너 overlay 에 있는데 최신 백업이 08-31 이다.** → 재구성 전에 `00_backup_state.sh` 를 다시 실행하는 게이트를 RUNBOOK(:261)에 넣고, 백업이 7일을 넘으면 경고하게 한다. (git-infra)
- **작업 브랜치가 main 보다 52커밋 앞서 있다**(main 쪽 내용 변경은 없음). → v7 을 커밋한 뒤 PR 로 main 에 병합하고 `git gc` 를 한 번 돌린다. (git-infra)
- **`.claude/settings.local.json:7,10,44-46` 이 bash·python·git push 를 전면 허용한다.** 없는 스크립트를 가리키는 항목도 있다(:23-25, :32). 유지할지는 사용자가 판단할 일이고, 에이전트가 편집할 대상이 아니다. (git-infra)

## 확인 불가 / 범위 밖

- CLAUDE.md 는 동시에 재작성 중이라 감사에서 뺐다. 다만 "실험 종료 ~10월 중순", "1b ~10-03", fig3 재현 명령(`--slices` 누락)은 같은 이유로 갱신이 필요하다.
- 옛 컨테이너(snorlax_GPU0 등)가 정리됐는지는 컨테이너 안에서 docker 에 접근할 수 없어 확인하지 못했다.
- 새 비교 모델의 BS8 VRAM 은 GPU 가 필요해 확인하지 못했다. 1b 뒤에 확인한다.
- 본편 no-DC 런의 실제 DataLoader 워커 수(0 으로 추정)는 확인하지 못했다.
- 그림 PNG 의 시각적 렌더와 docx 의 실제 쪽수는 렌더러가 없어 확인하지 못했다. Word 로 실측해야 한다.
- `paper/ieie/reviews/` 는 실명이 들어 있어 파일 개수만 셌고 내용은 열지 않았다.
- 제외한 것: 교수님 원본 파일, 320 트랙 역사 문서의 죽은 링크(~20곳), 08-07 이전 문서의 composite 수치(역사 기록), 트레이너의 composite best-ckpt 선택(의도된 설계).
- 점검 중 다른 작업으로 이미 해소되어 뺀 것: README 상태, INDEX 누락 행, 1b 진행 미기록.
- 정상으로 확인된 것
  - v7 초안의 수치는 표·평가 요약과 모두 일치한다.
  - v7 산출물은 소스에서 다시 빌드한 결과와 바이트 단위로 같다.
  - blind 요건(docx 메타데이터, 머리글·바닥글)에 문제가 없다.
  - 대상 .py 40개가 모두 파싱되고, import 하는 모듈은 모두 git 에 추적돼 있다.

**주의:** 1b SS2D 가 ep18/50 에서 실행 중이다(~10-02 완료). `main_train_pure_v8.py`, 공유 config, `run_v8_seed1_50ep.sh` 는 1b 가 끝나기 전에 편집하면 안 된다. 트레이너와 config 는 런이 재시작될 때 새로 반영되고, bash 는 실행 중인 스크립트를 이어서 읽기 때문이다.