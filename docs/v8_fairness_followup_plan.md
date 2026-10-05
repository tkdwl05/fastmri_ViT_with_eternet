# v8 통제 비교 보강 실험 계획 (한계 ①시드 ②학습 설정 ③용량)

작성 2026-09-01. 배경: v8 GRU↔SS2D 비교의 "잘 학습됐는가" 4단계 점검에서 점검 단계 1~3(무결성·포화·
공정성)은 로그로 입증됐고, 남은 점검 단계 4 한계 3건을 실험으로 닫는 계획. 실행은 radapt(가속화 계수 적응형 강화 SS2D) 학습 완료(~09-04)
후 GPU 큐. **오늘 완료된 준비(CPU)**: 시드 인프라 + 검증 + 런처 + 용량 하한 실측.

## 준비 완료 (2026-09-01, GPU 불필요분)

- **시드 패치**: `v8_eter_pure/main_train_pure_v8.py` 에 `SEED` 환경 변수 분기 추가 —
  가중치 초기화·셔플(generator)·dataset rng(마스크 offset·flip)·워커별 독립 스트림(worker_init_fn).
  **SEED 미설정 시 기존 런과 코드 경로 동일**(하위 호환). 런 폴더는 `..._v8_s{SEED}` 로 격리(RUN_SUFFIX
  env 로 별도 지정 가능). fp16 atomics 로 비트단위 재현은 아님(문서화용 독립 시드가 목적).
- **CPU 재현성 검증 PASS**: 동일 seed → 동일 샘플(마스크 offset·flip 포함), 다른 seed → 상이.
- **런처**: `v8_eter_pure/runs/run_v8_multiseed.sh` (시드별 쌍 우선 루프 → 적응형 중간 분석,
  전체 상태 재개(true-resume) + MAX_RETRY 200).
- **부수 수리**: 기존 시드 미고정 경로의 "persistent_workers fork 시 16워커 동일 rng 복제" 문제가
  시드 경로에선 worker_init_fn 재시드로 해소됨 (기존 완료된 런은 소급하지 않음 — 두 모델 동일 조건
  이었으므로 비교 공정성 무영향).

## E1 — 다중 시드 단축 학습 (한계 ①, 최우선)

- **설계**: SEED∈{0,1,2} × {GRU, SS2D} × **25ep**(cosine-to-25 **단축 학습** — 50ep 스케줄 절단이
  아님. 각 시드가 "학습이 완료된 런"의 분산을 근사 추정; 판정은 ep24 val, VAL_EVERY=2 유지).
- **판정**: (i) 부호 안정성 — 3시드 전부 SS2D>GRU 인가, (ii) paired 차이 d₀,d₁,d₂ 의 평균·범위
  보고. 주 실험 Δ(ssim 0.0014)와 v9 Δ(0.0005)를 이 분산 위에서 해석(후자는 묻힐 수 있음 — 그 결과도
  "v9 동급, 기여는 radapt" 논지와 일치하므로 그대로 보고).
- **비용**: 쌍/시드 ≈ 25×(2.41+3.07)h ≈ **5.7일** → 3시드 **~17일**. 적응형: seed0 → seed1 이
  부호 일치하면 seed2 로 확정, 불일치하면 즉시 중단·재설계.

### E1 결과 (09-18 학습 완료, 6/6 런 — 판정 09-21)

| seed | SS2D SSIM_m | GRU SSIM_m | Δ(SS2D−GRU) |
|---|---|---|---|
| 0 | 0.9101 | 0.9096 | +0.0005 |
| 1 | 0.9104 | 0.9109 | −0.0005 |
| 2 | 0.9107 | 0.9112 | −0.0005 |
| 평균±SD | 0.9104±0.0003 | 0.9106±0.0009 | −0.0002 |

(ep24 val = best, 슬라이스 단위 masked SSIM; PSNR SS2D 34.80±0.08 / GRU 34.82±0.10 dB, nMSE 0.0042 동일.
런 폴더 `logs/PureETER_{SS2D,GRU}_noDC_R4_brain384_v8_s{0,1,2}/log.txt`.)

- **판정 (i) 부호 안정성: 불성립** — seed 1·2 에서 GRU 우세로 반전. paired Δ 범위 ±0.0005, 시드 SD 0.0003~0.0009.
  → **25ep 예산에서 두 모델은 통계적으로 구분되지 않는다.** 런처는 seed0/seed1 불일치 시 자동 중단 없이 seed2 까지
  진행했으나(중간 분석은 수동), 3시드 데이터가 판정에는 오히려 유리.
- 주 실험 50ep 단일 런 Δ(+0.0014; 0.9140 vs 0.9126)는 E1 시드 분산의 1.5~3배이고, 주 실험 런도 ep24 시점 Δ 는 +0.0002 로
  격차가 ep26~50 에서 벌어졌다(`results/eval/v8_nodc/matched_epoch_table.md`). "50ep 에서만 나타나는 효과"인지
  "주 실험(시드 미고정)의 우연 변동"인지 E1 만으로는 판별 불가 → **재설계(09-21, 사용자 결정)**: 50ep 본 학습 설정 seed 1 쌍(GRU→SS2D)
  로 주 실험 Δ 재현 검증(큐 1b). 견고한 주장 = "1/21 파라미터로 원본 GRU 와 동등 품질"; "세 지표 모두 우위"(IEIE v7 초록)는
  1b 결과 후 유지/완화 결정(사용자 결정 09-21).
- **1b 결과(10-02)**: seed 1 50ep 에서 GRU 0.9136 / SS2D 0.9133(볼륨 SSIM, ΔSSIM −0.0003, 짝 검정은 GRU 쪽으로 유의) — 주 실험 Δ(+0.0014) **재현 안 됨**.
  같은 모델의 두 50ep 런 차이(GRU 0.9127→0.9136, SS2D 0.9141→0.9133)가 모델 간 차이보다 크다 → 50ep 에서도 두 모델은 구분되지 않음(E1 과 같은 결론).
  짝 Wilcoxon 의 p<0.001 은 한 쌍의 학습된 모델에 대한 평가 데이터 변동만 반영한다는 점을 원고에 명시해야 함. 수치: `results/eval/v8_nodc_s1_50ep/volume_paired_summary.md`.
- **U-Net 단독 결과(10-04)**: 시퀀스 모듈을 뺀 같은 U-Net(seed 1) 볼륨 SSIM 0.9127 — ETER-net·SS2D(controlled) 네 런(0.9127~0.9141)과 같은 범위. 시퀀스 모듈의 추가 이득 0.0001~0.0015 는
  런 간 변동과 같은 수준(1회차 ETER-net 은 U-Net 단독과 p 0.33). 교수님 코드(`u_choh_model.py:80`)도 U-Net 입력에 zero-filled 영상을 함께 넣는 구조라, R=4·16코일·ACS 8% 에서는
  U-Net 이 재구성의 대부분을 담당한다는 해석. 원고 반영 방식은 교수님 상의 권장(사용자 결정 대기). 수치: `results/eval/v8_unet_only/summary_unet_only.md`.
- v9 Δ(0.0005)는 예상대로 이 분산에 묻힘 → "v9 unleashed 는 v8 SS2D 동급, 기여는 radapt" 논지 유지.

## E2 — GRU LR 미니스윕 (한계 ②, 선택 — 심사 대응)

- GRU, LR∈{1e-4, 4e-4}×10ep×seed0 (기존 2e-4 와 비교). 목적: "학습 설정이 GRU 를 불리하게 했다"
  반론 차단. 기본 심사 대응은 이미 있음(학습 설정이 교수님 ETER/GRU 계보 v5/v6 튜닝 출신). ≈ **2일**.

## E3 — 용량 매칭 (한계 ③): ★구조적 불가능 실측 → 논거로 전환

2026-09-01 CPU 실측 (`PureETER_GRU(n_hidden_1=H, n_hidden_2=H, use_dc=False)` 파라미터 수):

| H (=H1=H2) | total | GRU 모듈 | 비고 |
|---|---|---|---|
| **1 (최소)** | **62.9M** | **31.9M** | GRU 모듈 최소치가 이미 SS2D 모델 **전체**(31.2M) 초과 |
| 2 | 101.8M | 70.8M | |
| 3 | 147.9M | 116.8M | |
| 5 | 261.1M | 230.1M | |
| 10 (주 실험) | 668.2M | 637.1M | U-Net 등 공통분 ~31.1M |

- **결론: param-matched GRU 는 이 아키텍처에서 정의 불가능.** flatten-reshape 가 H당 ~64M 를
  강제(하한 63M total)하는 반면 SS2D 모듈은 ~0.1M — 파라미터 비대는 튜닝 선택이 아니라 **구조의
  하한**이다. 이 표 자체가 §5-(2) "파라미터 효율" 논거의 정량 보강이며, "매칭 실험이 없다"는
  비판에 대한 1차 답변.
- **선택 실험**: 최소 GRU(H=1, 62.9M) 50ep 1런 (≈**4일**, h/ep 은 68 정도로 짧아질 것 — 실측 후 갱신).
  해석 매트릭스: H1 ≈ H10 → "668M 은 과잉용량" 실증 / H1 ≪ H10 → "GRU 는 그 용량이 실제로 필요"
  — 어느 쪽이든 §5 에 정보.

## 비교 기준 모델 명문화 (2026-09-02, 사용자 지적 반영)

**기준 모델은 교수님 원본 ETER-Net(GRU, 668M)이다 — SS2D 가 아니다.**
- 모든 모델(SS2D·Transformer·pixel-GRU)은 교수님 골격(out_ch=2×H2=20 원본 상수, U-Net DFU,
  dual-input concat [51], 행→열 스캔 축, GRU-계보 학습 설정)에 연결되는 **치환 후보**다.
- **1차(주) 비교 = 각 모델 vs 원본 GRU** ("원본 대비 치환 이득표"). 모델 간 pairwise(예: pixel-GRU
  vs SS2D 의 메커니즘 해석)는 2차 비교 — 분석 스크립트·표·논지 모두 이 순서를 따른다.
- 신규 모델의 파라미터 규모를 SS2D 모듈(~0.1M)에 맞춘 것은 SS2D 가 표준이어서가 아니라 **"가중치 공유
  파라미터화의 최소 공통 규모"** 규약이기 때문(원본은 하한 63M 이라 하향 매칭 불가 — 상향
  매칭은 구조적 비효율의 복제라 무의미). 문구에 주의: SS2D 는 "현재 최고 성능 비교 모델"이지
  "기준 모델"이 아니다.

## 추가 결정 (2026-09-02 오후, 사용자) — ★통제 비교 보강 실험 최우선 재편

- **radapt 를 epoch 경계(ckpt 저장 직후)에서 계획 정지**하고(ep57 예정, `clean_stop` 재사용,
  손실 ~0) 통제 비교 보강 실험부터 실행. radapt 잔여 ~23ep(≈2.6일)는 통제 비교 보강 실험 후 전체 상태 재개
  (`post_reboot_rearm.sh` — 검증된 재개 경로).
- **E1 다중 시드 즉시 실행 시작** (seeds 0,1,2 × {ss2d,gru} × 25ep) — 정지→실행 시작 자동 체인 무장.
- **④번째 비교 모델 pixel-GRU 구현 완료**: 위치 간 가중치 공유 순환(행→열 pixel-scan bi-GRU, stem 없음) —
  메커니즘 vs 파라미터화 교란 요인 분리. `models/rnn_eternet/pixelgru_v10.py` +
  `u_pure_eternet_pixelgru.py` + `SEQ_MODEL=pixelgru`. CPU 스모크 PASS:
  **total 31.17M / 모듈 0.115M** (SS2D 0.1M·Transformer 0.104M 동급 — flatten-GRU 에선 불가능했던
  파라미터 규모 매칭이 가중치 공유로 성립함 자체가 §5 논점). 해석: ≈SS2D → 파라미터화 탓(순환 무죄) /
  <SS2D → 선택적 상태전이 메커니즘 이득 실재.
- 초안 주장 범위 잠금(⓪): §5-(4)·§6 의 "SSM>RNN" 일반화 표현을 "SS2D 치환 > 원 bi-GRU
  설계"로 한정 + §5-(6)에 메커니즘/파라미터화 교란 요인 한계 항목 신설.

## 결정 (2026-09-02 오전, 사용자)

- **E2·E3 실행 확정** (보류 아님 — 투고 전 수행).
- **GPU1 사용 안 함** — 전 큐 GPU0 단독 순차.
- **radapt 세 번째 결과 절 편입 진행** (가속화 계수별 평가 결과로 최종 확정) · **Transformer(3번째 비교 모델) 이번 논문 포함**
  (구현 완료 — `axial_transformer_arm_design.md` 참조. 시드 대칭성: 본표는 50ep seed0,
  E1 에 Transformer 포함(ARMS="ss2d gru transformer", +~8.5일) 여부는 아래 큐 확정 시 선택).

## GPU 큐 통합 (radapt ~09-04 학습 완료 후)

**09-02 오후 재편 — 통제 비교 보강 실험 최우선 (radapt ep57 정지 후):**

**09-21 갱신**: E1 학습 완료(09-18 18:37 UTC) 후 GPU0 가 2.5일 유휴였음(E1 런처는 체인 없이 종료) → 1b 삽입. 1b 런처(로컬 `*.sh`, git-ignore):
```
CUDA_VISIBLE_DEVICES=0 setsid nohup bash v8_eter_pure/runs/run_v8_seed1_50ep.sh \
  > v8_eter_pure/runs/multiseed_outer_s1_50ep.log 2>&1 < /dev/null & disown
```
= `SEED=1 RUN_SUFFIX=_s1_50ep SEQ_MODEL={gru,ss2d} USE_DC=0 SANITY_NUM_EPOCHS=50 SMOKE_BS=8` 순차(전체 상태 재개, MAX_RETRY 200),
런 폴더 `logs/PureETER_{GRU,SS2D}_noDC_R4_brain384_v8_s1_50ep/`(E1 의 `_s1` 25ep 폴더와 분리), 학습 설정은 주 실험 50ep 와 동일.
완료 판정 = `log.txt` 의 `Epoch 50/50`. 이후 2단계부터는 각 단계 완료 시 수동 실행 시작(자동 체인 없음 — 유휴 재발 주의).

| 순서 | 작업 | 비용 | 상태 |
|---|---|---|---|
| 1 | **E1 다중 시드** seeds 0,1,2 × {ss2d,gru} × 25ep | ~17일 | ✅ 09-18 완료 — 시드 간 차이의 부호가 일관되지 않음(위 E1 결과) |
| 1b | **50ep seed 1 쌍** GRU→SS2D (주 실험 Δ 재현 검증, E1 재설계) | ~11.4일 (GRU 5.0 + SS2D 6.4) | ✅ **10-02 01:04 UTC 학습 완료**(09-21 07:26 실행 시작, 실측 GRU 4.65일 + SS2D 6.09일, 두 런 모두 재시작 0회). **best = 두 모델 모두 ep50: GRU SSIM_m 0.9135 / SS2D 0.9132**(학습 로그·슬라이스 단위; 주 실험(시드 미고정) 0.9126 / 0.9140). 재현 기준 SS2D ≥0.9149 **미달** — seed 1 에서 Δ = −0.0003. 짝 평가 ✅(10-02 03:07, `results/eval/v8_nodc_s1_50ep/`, 요약 `volume_paired_summary.md`): **볼륨 SSIM GRU 0.9136±0.0364 / SS2D 0.9133±0.0364, PSNR 33.86 / 33.82, nMSE 0.442 / 0.444 %; SS2D 우위 슬라이스 42.6 %·볼륨 35.8 %, ΔSSIM −0.0003 [−0.0004, −0.0002], 볼륨 Wilcoxon p 2.3e-11(GRU 쪽)**. 주 실험(+0.0014, p 2.4e-70, SS2D 쪽)과 방향이 반대 → 짝 검정은 평가 데이터 변동만 반영하고 학습 시드 변동은 반영하지 않음. 모델 차이 < 학습 런 간 변동(같은 모델 두 런 차 GRU 0.0009·SS2D 0.0008) → **판정: 두 모델 동등**(E1 과 일치). 원고 주장은 '1/21 파라미터로 동등'으로 완화 필요(사용자 확인 대기) |
| 2 | **U-Net 단독 모델 50ep SEED=1** (시퀀스 모듈 제거 = $f_\theta \equiv 0$, U-Net 입력 52채널 유지·시퀀스 모듈 자리 0 채움) — 시퀀스 모듈의 기여 확인(치환 이득의 분모), **IEIE 표 1 행 목표** | 실측 **2.0일**(10-02 03:08 → 10-04 03:31 UTC, 0.87 h/ep 학습 — 데이터 로딩 CPU 병목) | ✅ **10-04 03:31 UTC 학습 완료**(재시작 0회, best ep50) · 평가 ✅ 10-04 12:00 UTC `results/eval/v8_unet_only/summary_unet_only.md`: **볼륨 SSIM 0.9127±0.0366 / PSNR 33.76±1.88 / nMSE 0.452±0.285 %**. 시퀀스 모듈 모델 − U-Net only ΔSSIM: ETER-net run 2 +0.0010(볼륨 83.6 %, p<0.001)·SS2D run 2 +0.0006(73.3 %)·ETER-net run 1 +0.0001(p 0.33, 유의한 차이 없음)·SS2D run 1 +0.0015(94.2 %) → **시퀀스 모듈 기여 ≤0.0015 = 학습 런 간 변동(~0.0009)과 같은 수준**. 이력: **10-01 사용자 결정으로 8단계에서 앞당김.** 래퍼 `models/pure_eternet/u_pure_eternet_unet.py`·점검 `v8_eter_pure/sanity_unet_only_v8.py`·평가 `v8_eter_pure/eval_unet_only_v8.py` 준비. ▶ **10-02 03:08 UTC 실행 시작** — 트레이너 `SEQ_MODEL=unet` 분기 적용, GPU 스모크(BS 8 합성 입력 0.16 s/iter, peak 3.8 GB) 후 `SEED=1 ARMS=unet SMOKE_BS=8 CUDA_VISIBLE_DEVICES=0 setsid nohup bash v8_eter_pure/runs/run_v8_seed1_50ep.sh > v8_eter_pure/runs/multiseed_outer_unet_s1_50ep.log 2>&1 < /dev/null & disown`. 런 폴더 `logs/PureETER_UNET_noDC_R4_brain384_v8_s1_50ep/`(params 31.1M). 완료 후 `eval_unet_only_v8.py` → 표 1 행(1b seed 1 CSV 를 `--ref` 로 같은 데이터 스트림 짝 비교) |
| 2b | **pixel-GRU 50ep SEED=1** (④번째 모델 — 메커니즘 분리; 1b·U-Net 단독과 같은 데이터 스트림) | ~2일 추정(ep1 초반 2.4 batch/s ≈ 0.95 h/ep; ep1 후 갱신) | ▶ **10-05 10:00 UTC 실행 시작**(사용자 10-05 '다 적용' — SEED=1 확정). GPU 스모크(BS 8 합성 입력 0.32 s/iter, peak 9.7 GB/reserved 12.7 GB) 후 `SEED=1 ARMS=pixelgru SMOKE_BS=8 CUDA_VISIBLE_DEVICES=0 setsid nohup bash v8_eter_pure/runs/run_v8_seed1_50ep.sh > v8_eter_pure/runs/multiseed_outer_pixelgru_s1_50ep.log 2>&1 < /dev/null & disown`. 런 폴더 `logs/PureETER_PIXELGRU_noDC_R4_brain384_v8_s1_50ep/`(params 31.2M, 모듈 0.115M). 완료 후 평가 = `CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/eval_unet_only_v8.py --seq pixelgru --ref …`(10-05 `--seq` 추가, unet 출력 바이트 동일·pixel-GRU CPU 배관 점검 통과) → `results/eval/v8_pixelgru/` |
| 3 | Transformer 50ep seed0 (③번째 모델, 구현 세부는 `axial_transformer_arm_design.md`) | ~5.6일 | 구현 완료 |
| 4 | E3 최소-GRU 50ep | ~4일 | 확정 |
| 5 | E2 LR 스윕 | ~2일 | 확정 |
| 6 | **radapt 재개** (잔여 ~23ep) | ~2.6일 | 전체 상태 재개 대기 |
| 7 | 추론-only 일괄 (Table 4·PromptMR+/DDS·ms/VRAM·가속화 계수별 평가·ringing) | 1~2일 | 준비 완료 |
| 8 | ~~U-Net-only 학습~~ | — | **2단계로 이동(10-01)** |

**7단계 세부(09-03 추가, 논문 표 관례 정비 `docs/paper_table_conventions.md` §3)**: (a) 가속화 계수별 평가는 전체 val 7,334 슬라이스로 재실행(현 `results/eval/v8_r_sweep/` 은 stride-4 서브샘플 n=1,834 → 논문 표 불가), 세 모델 + Zero-filled, R∈{2,4,6,8}; (b) 모델별 추론 ms/slice·peak VRAM(배치 크기(BS) 1, AMP, 동일 GPU) → 표 5 [TBD] 칸; (c) 공개 U-Net/E2E-VarNet 전체 val(n=464 볼륨, 실측 코일만 전달) — 누수 참고 결과·각주용 — **본 연구 프로토콜 행은 PromptMR+ 까지 포함해 09-06 CPU 로 선행 실행 시작**(`v8_eter_pure/eval_baselines_full.py` → `results/eval/baselines_384_full/`, GPU0 불사용, `docs/frontier_baselines_plan.md` §3; 7단계엔 native 행·ms/VRAM 만 남음); (d) fastMRI 표준 프로토콜(320 center-crop·무마스크·볼륨 단위) 1회 산출 — 외부 수치와의 비교 가능성 확보. Zero-filled 기준선은 CPU 로 09-03 완료(`results/eval/zero_filled/`).

합계(모두 GPU0 순차) ≈ **40일 → 10월 중순 종료**(통제 비교 보강 실험 ~33일 + radapt/평가 ~7일).
SI 마감 11-30 대비 집필 병행 필요. (구 상의 항목 (a)(b)는 09-02 결정으로 종결 — 위 '결정' 절.)
