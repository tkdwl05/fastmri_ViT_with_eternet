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

## 결정 (2026-10-12, 사용자) — 교수님 원본 학습 설정 대조 후속

- 대조 결과(`docs/professor_settings_compare_2026-10-12.md`, 45 항목): 현재 v8fix 는 교수님 순수 ETER-net 과 **네트워크 구조만 같고 데이터·학습 방식 대부분이 다르다**. H 는 10(공개 model.py 12) — 교수님 확인 대기. 교수님 순수 ETER-net 학습 설정 파일(`myConfig_ETER_fastmri_brain_acs32R4_5`)은 저장소·git·서버 어디에도 없다.
- **IEIE(10-19)**: 현재 v8fix 설정 그대로 마무리하고, 방법 절에 원본과 다른 점(데이터·세기 정규화·마스크·손실·최적화·정밀도·H)을 명시 — 문안(안) 대조 문서 §5.3.
- **MDPI(11-30)**: 교수님께 설정(또는 그 런의 `log.txt`)·H·데이터를 여쭌 뒤(대조 문서 §5.4) 재현 학습 — 큐 P 행.

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
| **S** | **레시피 수정 스모크** (`v8_eter_pure/smoke_recipe_fix.py`, 10-07 학습 붕괴 감사 후속 — `docs/v8_training_collapse_audit_2026-10-07.md`) | ~18 h (핵심 9 job ~8.5 h) | ✅ **10-08 05:53 UTC 완료**(18 job, 실패 0; 10-07 12:26 실행 시작). 붕괴 원인 = 결합형 L2(B 만으로 U-Net 생존). 시퀀스 모듈은 clip 해제 + 세기 정규화가 필요합니다. **원 손실 균형을 되돌린 DL 에서만 GRU 출력 사용이 사전 기준을 통과했습니다**(shuffle 하락 0.0148, 62/64; GRU DL 0.7973 vs U-Net 단독 DL 0.7773). SS2D 모듈은 전 조건에서 거의 학습되지 않았습니다(relΔ ≤0.026). 다음 단계(DL 레시피 5 epoch 등)는 사용자 결정 대기 — 상세 감사 문서 §(7). job 순서: unet:A,D,E → gru:A,D,E → ss2d:A,D,E → unet/gru/ss2d:B,C → DL. 각 4000 step(BS 8, SEED 1). A=현행(Adam 결합 L2 3e-5+clip 1.0) / B=AdamW(no-decay 그룹) / C=B+clip 해제 / D=C+입력·label 슬라이스별 정규화 / DL=D+λ 정규화 / E=D+Adam eps 1e-12. 기록: 층별 near-dead·relΔ·eps 지배 비율, 검증 64장 SSIM + shuffle/zero/deep 절제. 결과 `results/smoke_recipe_fix/summary.md` |
| **F5** | **v8fix 5 epoch 점검** — 수정 레시피 DL 트레이너 `v8_eter_pure/main_train_pure_v8fix.py`(기존 트레이너 무수정; 레시피 함수는 `smoke_recipe_fix.py` 단일 출처), U-Net 단독 → ETER-net(GRU) → SS2D, SEED=1, 50 epoch 일정으로 시작해 epoch 5 에서 정지(PAUSED_ep5) → 같은 런 폴더에서 50 epoch 까지 이어 학습 가능. U-Net 초기값 세 모델 동일(SAME_UNET_INIT), 데이터 흐름 = (SEED, epoch) 함수(재개해도 같음), 500 step 마다 가중치 소멸 감시(ALERT exit 3), 검증 epoch 마다 기능 시험 64 슬라이스. 평가 `eval_v8fix.py`(전체 val + 기능 시험 512 슬라이스) → `results/eval/v8fix_ep5/` | 약 1.3~1.5일 (0.97+2.23+2.92 h/ep × 5 + 평가) | ▶ **10-08 13:20 UTC 실행 시작** (사용자 10-08 "그렇게 진행하자"). 코드 검토 워크플로(8 에이전트) 지적 반영 + GPU 디버그 2회 통과(정지·재개·거부 경로, 재개 epoch 손실 = 끊김 없는 런과 동일, U-Net 초기값 checksum 세 모델 동일). 명령: `CUDA_VISIBLE_DEVICES=0 setsid nohup bash v8_eter_pure/runs/run_v8fix_gate.sh > v8_eter_pure/runs/v8fix_gate_outer.log 2>&1 < /dev/null & disown`. 50 epoch 이어 학습 = 같은 명령에 `STOP_AFTER_EPOCH=50` (사용자 결정 후) |
| **R1** | **RunPod RTX 3090 — v8fix 50 epoch (val 0~2 전체·test_full)** — U-Net 단독 → ETER-net(GRU) → SS2D, SEED=1, 런 이름 `_s1_fullval`. 데이터 = 공식 링크로 Pod 가 직접 받음: train 4,110 파일/65,060 슬라이스(로컬과 같은 목록, 로컬에서 잘렸던 2 파일은 온전), val 0~2 전체 1,378/21,842(학습 중 검증·베스트 선택, 10-09 사용자 결정), test_full 558/8,852(최종 평가). 공식 split 이라 train–val 중복 없음 (로컬 val 0번에는 train 볼륨 4 개가 섞여 있었음 — file_brain_AXT2_200_2000{306,308,310,311}). s̄=325.25 → λ_SSIM 0.003075(로컬 0.002822). 환경 = 로컬과 같은 버전(python 3.10/torch 2.3.1+cu121/cudnn 8902/mamba_ssm 2.2.2), TF32 끔, prefetch 2·val 워커 12(Pod RAM 125 GB — prefetch 4 는 OOM). wandb = tkdwl05-hongik-university/fastMRI-research. 결과 동기화 = git `runpod/v8fix` 브랜치 `runpod_results/`(30 분), 체크포인트 = 로컬 `logs/runpod_mirror/`(6 시간). 스크립트 `infra/runpod/` (README). 3090 속도: U-Net 1.9 batch/s(데이터 공급 병목), GRU 1.37, SS2D 1.22 → 합계 약 10.7 일 | 약 10.7 일 · 약 $235 | ▶ **10-09 10:06 UTC 실행 시작**(사용자 10-09 "50epoch 로 충분할 것 같으면 그대로 진행"). NaN-skip 약 0.3% (데이터 무관·GPU 반복 시험 이상 없음 — 비결정 수치 넘침 추정, 기록하며 감시) · **10-11** U-Net 단독 학습 완료(pod2) · **10-12 12:43 UTC ETER-net pod3→pod2 이전**: pod3 는 호스트 메모리 경합으로 데이터 로딩이 밀려 GPU 사용률 약 50 %·epoch 3.6 h 까지 늘어남 → ep15 끝 last.pt 에서 pod2 로 재개(같은 트레이너 md5, wandb id `_pod3f` 유지, 1.08 s/batch ≈2.4 h/epoch, 종료 ≈10-16 03 UTC). SS2D 는 pod3 GPU1 단독(GPU0 비운 뒤 0.82 s/batch, 종료 ≈10-14 저녁 UTC). epoch_5/10/15 스냅숏은 pod3 에만 있음 — pod3 종료 전 회수 · **10-15 SS2D 학습·평가 완료(pod3, 00:50 UTC) · pod3 파일 로컬 회수(md5 일치) · 03:5x UTC pod2 가 종료되어(사용자가 pod3 대신 종료) ETER-net ep37 진행분 손실 → 로컬 사본 ep36 끝 last.pt 를 pod3 로 올려 08:38 UTC pod3 GPU0 에서 재개(RESUME start_epoch=36·best 0.9204·같은 wandb run, 1.09 s/batch) — 학습 종료 ≈10-16 23 UTC + test_full·val 평가. pod3 는 ETER-net 끝날 때까지 유지** · **✅ 10-16 23:49 UTC ETER-net 50 epoch 학습 완료(pod3, DONE)**: best ep46 val SSIM 0.92138, test_full 볼륨 SSIM **0.9191±0.0474 / PSNR 34.69±1.84 / nMSE 0.388±0.434 %**(런처 평가 00:39 UTC; U-Net 단독 0.9213·SS2D 0.9199 — 세 모델 중 가장 낮음, 기능 시험 shuffle_cross +0.0019 used·shuffle_adj not shown). 공식 정의 세 모델 평가 = `docs/ssim_protocol_survey_2026-10-16.md` (f)(뇌 안 순위 U-Net only > SS2D > ETER-net 이 정의와 무관, 평가 TF32 영향 없음). val 0~2 전체 평가 10-17 02:32 UTC 완료: 볼륨 SSIM 0.9214±0.0368 / PSNR 34.72±1.77 / nMSE 0.364±0.207 % (슬라이스 평균 0.9214 = log.txt ep46 val_ssim_m 0.92138 과 일치 — 정상), 기능 시험 shuffle_cross +0.00160 used·shuffle_adj not shown. **✅ 10-17 02:37 UTC pod3 회수 완료 → 종료 가능**: 체크포인트 8 개(best·epoch_5/10/15/40/45/50·last) 로컬 `logs/runpod_mirror_pod3/logs/PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval/` md5 일치(pod 재확인 포함), 작은 파일 155 개(결과·로그·wandb·official_eval·probe json) `logs/runpod_mirror/pod3_final_extra_20261017/` md5 일치, pod 쪽 git 브랜치 runpod/v8fix-pod3 = origin(8e2ddae). 로컬 pull 루프 정지. `official_eval/ckpt/pure_unet_best.pt` 는 로컬에서 올린 사본이라 회수 안 함 · **10-17 사용자가 pod3 종료 — R1 완료, 현재 RunPod Pod 없음** |
| **R2** | **RunPod pod2 (RTX 3090 1 장) — Mamba-ETER v8fix 50 epoch** (사용자 10-12 "Mamba-ETER 로 진행"). ETER-net 의 bi-GRU 셀만 양방향 Mamba 로 바꾼 `PureETER_Mamba1D`(`models/pure_eternet/u_pure_eternet_mamba1d.py`, 시퀀스 모듈 637.15M — GRU 637.10M 과 정합), mamba_ssm 기본 초기화(outmatch 없음), Mamba h·v 두 단계 fp32(U-Net fp16 — 기준 ETER-net 은 gru_h 만 fp32: 정밀도 비대칭은 방법 절에 명시), SEED=1, 런 이름 `PureETER_MAMBA1D_noDC_R4_brain384_v8fix_s1_fullval`. 레시피·데이터 순서·U-Net 초기값(checksum 5.9212156172e+02)·s̄ 325.25·BS 8·검증(val 0~2, 2 epoch 마다)·AMP 보호 장치는 pod3 ETER-net 런과 같고 어긋나면 시작 거부. 트레이너 `v8_eter_pure/main_train_pure_v8fix_mamba1d.py`(pod3 ampguard 트레이너 복사 + 모델 교체), 평가 `eval_v8fix_mamba1d.py`, 런처 `infra/runpod/pod_run_mamba1d.sh`, GPU 사전 점검 `preflight_mamba1d_gpu.py`(로컬 TITAN·pod2 3090 통과, fast path vs 참조 ≈3e-7). 검증 epoch 마다 MAMBA-DIAG(eps 지배 비율·out_proj 노름 비 — 4000 step 시험의 Δ 경로 eps 한계 감시). 해석 규칙(4000 step 시험에서 이어받음): 끝에서 시퀀스 eps 지배 ≥ 0.9 이거나 out_proj 상대 변화 < 0.01 이면 "셀 비교로 해석 불가"로 보고 | 3090 실측 1.36 batch/s(0.74 s/step, 피크 14.6 GB) → 학습 약 1.7 h/epoch + 검증 → 약 4.5 일 (≈10-16~17 종료, 10-19 학술대회 원고 불가·MDPI 용) | ▶ **10-12 07:56 UTC 실행 시작**(`WANDB_RUN_TAG=pod2 setsid -f bash infra/runpod/pod_run_mamba1d.sh`). 시작 점검 통과: s̄ 325.2484691441059·λ_SSIM 0.0030745725…·U-Net checksum 5.9212156172e+02 (ETER-net 런과 같음). 사전 점검: 독립 검토(3 관점) 반영·로컬 재개 시험·pod2 300 step 점검. 코드 파일은 pod2 에 git 미추적으로 복사(동기화 대상 아님) · **⏸ 10-12 12:2x UTC 정지**(`PAUSED_ep2_nan`): ep1 NaN-skip 221 · ep2 15 · ep3 batch 2,770 까지 약 505(250 배치 구간별 23~53 %) — ep2 val SSIM 0.8704(같은 epoch ETER-net 0.8657·SS2D 0.8842·U-Net 단독 0.8875). last.pt = ep2 끝(사본 `pure_mamba1d_last_ep2_backup.pt`). NaN 위치(Mamba fp32 vs U-Net fp16) 진단 전에는 재개하지 않음 |
| **S2** | **d-ETER 최적화 교란 스모크 (로컬 GPU0, 10-14 사용자 "학습 비교 돌려보자")** — 근거 `docs/v8fix_setup_audit_2026-10-14.md` §6-4. 조건당 4000 step·BS 8·SEED 1, 같은 데이터 흐름(stream md5 동일)·U-Net 초기값(checksum 5.9212156172e+02)·GRU 초기값·λ_SSIM 0.0030745725(pod 값)·cosine T_max 406,650(pod 런 첫 4000 step 의 LR 과 같음). 조건: ref = U-Net 단독 / A = pod2 레시피 그대로(gru_h fp32) / B = GRU LR ×0.1 / C = GRU 만 LR warmup 1000 step / D = 시퀀스 출력 ×0.06(영상 채널 RMS 에 맞춤, GRU Adam eps 보정) / E = zero-init 채널별 게이트 α(20) / 추가(사용자 승인 표 밖, 맨 뒤): ref2 = U-Net 초기값 seed 2(잡음 기준) · EC = E + C. 하네스 `v8_eter_pure/smoke_deter_opt.py`, 보고 `smoke_deter_opt_report.py`, 런처 `v8_eter_pure/runs/run_smoke_deter_opt.sh`(로컬), 출력 `results/smoke_deter_opt_2026-10-14/`. 판정 규칙(사전 등록) = 보고 스크립트 머리말: V3(A 의 학습 손실 열세 재현 R_A ≥ +2 %) 전제, S1 포화 sat_v ≤ 0.10 · S2 shuffle_adj > 0 (p<0.05) · S3 R_X(3001~4000) ≤ +0.5 % · S4 SSIM 비열등 | 약 12.8 h (ref 0.45 + GRU 조건 1.95 h × 6 + ref2 0.45) | ✅ **10-15 14:00 UTC 완료 — 1차 규칙 선정 없음(V3 불충족: R_A(3001~4000) −1.30 %, 기준 ≥ +2 %)**. 실행 뒤 확인: 같은 구간 pod ETER-net 은 U-Net 단독 대비 +10.4 %(wandb 같은 배치 흐름), 로컬 ref ≈ pod U-Net 수준 → 불일치는 ETER-net 쪽, 원인 미확인(회차 간 변동·데이터·GPU 후보). C 는 warmup 끝(step 1050) 발산·NaN, EC 는 넘김; B 만 포화 없앰(이득 없음); ref2−ref SSIM +0.0081(초기값만 다름) → 4000 step SSIM 순위 해석 불가. 보고 `docs/smoke_deter_opt_report_2026-10-15.md`. 다음(승인 필요): A2(GRU 초기값 seed 변경, ≈2 h) |
| **S2b** | **A2 — ETER-net 회차 간 변동 확인 (로컬 GPU0, 10-15 사용자 "승인할게 돌려보자")** — S2 의 A 와 같고 GRU 초기값만 seed 2(`smoke_deter_opt.py` 설계 결정 15; U-Net 초기값·배치 흐름·레시피 동일, 8 조건 실행 판은 `harness_as_run_ref-EC_md5_47aa0081.py` 로 보존). 판정(실행 전 고정, `v8_eter_pure/smoke_deter_opt_a2_compare.py`): R_A2(3001~4000) ≥ +2 % 이고 CI 하한 > 0 → 회차 의존(다중 시드 재설계) / ≤ +0.5 % 이고 상한 < +2 % → 로컬 재현 없음(pod 쪽 원인 조사) / 그 밖 보류. SSIM 은 판정에 쓰지 않음. 런처 `CONDS="A2" run_smoke_deter_opt.sh`, 출력 `results/smoke_deter_opt_2026-10-14/A2/`·`a2_compare.md` | 약 2 h | ✅ **10-15 17:31 UTC 완료 — 판정 (i) 회차 의존**: R_A2(3001~4000) **+31.2 %** [+22.3, +40.1] (A 는 −1.3 %), sat_v 0.44(step 500) → 0.92(step 4000)로 계속 깊어짐, E256 SSIM 0.7817(A 0.8142, ref 0.8120). GRU 초기값만 바꿔도 ETER-net 학습 궤적이 크게 갈림 → 단일 시드 조건 비교는 해석 불가, 다중 시드 재설계 필요. 결과 `results/smoke_deter_opt_2026-10-14/a2_compare.md`, 실행 판 `harness_as_run_A2_md5_7da42e3f.py`. (실행 전 기록: 사전 점검 통과(수정 판으로 A 30 step = 설계 실측 로그 비트 일치, A2 두 번 비트 동일, A2 GRU checksum −8.2039511092e+02 ≠ seed 1, U-Net·배치 흐름 = A)) |
| **S3** | **손실 요인 분리 1단계 — 교수님식 손실(P) vs 우리 손실(P0), GRU seed 3 개씩 (로컬 GPU0, 10-15 사용자 "실행해보자")** — 근거: 교수님 실제 학습 로그(ViT model3, `/mnt/sda/choh/shared/W_python/myViT/logs/251124_model3_…/log.txt`)의 손실 중 SSIM 항 52~57 % vs 우리 v8fix ≈1.5 %, 그리고 A2 의 회차 의존 판정. P = 전체 영상 L1 + 0.2·(1 − 전체 영상 SSIM)(교수님 ViT 형태, 우리 정규화 단위에서 비율 ≈ 1:1), P0 = 현행(마스크 L1 + λ 0.00307·(1 − 마스크 SSIM)). 조건: A3(P0, GRU seed 3) + refP·AP(seed 1)·A2P(seed 2)·A3P(seed 3). 판정(실행 전 고정): seed k 별 R0_k = R(A_k vs ref), RP_k = R(AP_k vs refP), 3001~4000 구간; max(RP) < min(R0) 이고 중앙값 차 ≥ 5 %p → P 에서 ETER-net 상대적으로 나음 (반대도 대칭) / 범위 겹치고 중앙값 차 ≤ 5 %p → 변화 없음 / 그 밖 보류 (3 대 3 완전 분리의 순열 p = 0.1 — 선별용). 기술: seed 별 sat_v·출력 치환 시험·평가 SSIM 두 방식(마스크 / 교수님 테스트식 전체 영상). '변화' 일 때만 2단계(F·L 분리, seed 3 개씩 ≈13 h). 구현: 하네스 설계 결정 16(사본에서 작업 후 교체), `smoke_deter_opt_loss_compare.py`(1차 = 위 규칙; 2차 = 회차 짝 규칙 — 구현 검토 중 10-15 ≈19:00 UTC 추가, A3 앞 절반을 본 뒤라 2차로만 보고), `smoke_deter_opt_dualeval.py`. 추가(맨 뒤, 판정 밖): ref2P = U-Net 초기값 seed 2 의 P — P 에서의 U-Net 회차 간 잡음 기준(교수님 테스트식 SSIM 차를 읽을 바닥) | 약 8.8 h (A3 1.95 + refP 0.45 + AP×3 5.85 + ref2P 0.45) | ✅ A3 10-15 19:36 UTC 완료(하네스 실행본 `harness_as_run_A3_md5_3f184ec0.py` 보존). 하네스 교체(md5 fc7e5950) 뒤 GPU 사전 점검 통과: A·Aeq 30 step 설계 실측과 비트 일치, refP·AP 30 step 불변식·E7 통과. ▶ **10-15 19:46 UTC `CONDS="refP AP A2P A3P ref2P"` 실행 시작**(≈6.75 h, 끝 ≈10-16 02:40 UTC). 진행: refP 20:16 완료(타당) · **AP 22:38 완료 — 타당성 실패(E3 amp_floor 1067 > 5)**: step ≈1350 부터 fp16 기울기 overflow 로 GradScaler 가 하한 1024 에 닿고, ≈1650~2730 거의 매 step 갱신 건너뜀(손실 0.07 → 0.37 후 일부 회복), 최종 SSIM 0.7782. 발산 전(1~1500) 기술: R(AP vs refP) −8~−23 %, GRU 포화 sat_v 0(P0 는 10~20 step 에 포화) — 판정에는 쓰지 않음. · **A2P 10-16 00:34 완료(타당 — E3 카운터 0·amp_floor 0)**: R(A2P vs refP, 3001~4000) **+59.4 %** [t9 +55.4, +63.4] (같은 seed 의 P0 A2 +31.2 %), E256 SSIM 0.7805·PSNR 26.04 (refP 0.8443·29.19). 경과(기술): 1~750 R −20 %·sat_v 0 → step 830 fp16 overflow 1회·831 gn 1523(clip 1000) → 시퀀스 출력 RMS 0.06 → 0.76(≈12 배)·sat_v 0 → 0.30·손실 ≈3 배, 이후 회복하지만 +50~60 % 에 머묾. AP(≈1350)와 같은 꼴의 사건이 더 이르게 한 번 — P 의 두 회차 모두 "포화 전 우세 → 기울기 급등 사건 → 포화·열세". · **A3P 02:31 완료(타당 — amp_floor 0·nan_skip 0)**: R(A3P vs refP) **+43.2 %** [t9 +38.9, +47.5] (P0 A3 +26.5 %), E256 SSIM 0.8187·PSNR 28.16. 경과(기술): 1~750 R −20 %·sat_v 0 → 815~822 fp16 overflow 4회(배율 1024 도달, 하한 밑으로는 안 감)·848 gn 1.5e6(clip) → 시퀀스 출력 RMS 0.06 → 0.78·sat_v 0.34 → 0.46 로 깊어지며 R +30 → +40~48 %. **세 P 회차 모두 같은 꼴(포화 전 −20 % 우세 → 830~1350 step 기울기 급등 → 포화·열세)**. **✅ 10-16 03:02 UTC 전 조건 완료(ref2P 타당), 03:29 판정**: **1차(S3) = 나빠짐**(R0 {−1.3, +31.2, +26.5} vs RX {+59.4, +43.2} 범위 완전 분리, 중앙값 차 +24.8 %p; P 쪽 타당 회차 2 개 — 등록 전제 3, AP 무효이나 넣어도 +81.5 % 같은 방향) · **2차(회차 짝) = 보류**(마스크 L1 Δ: A2P +23.2 %p move, A3P −1.8 %p stay). U-Net 잡음 ref2P − refP 손실 비 +0.18 %, SSIM +0.0047(OURS)/+0.0012(PROF_TEST). 사후 기술: 급등 전(1~750) P 세 회차 마스크 L1 비 −7.6~−8.1 %(P0 +16~+27 %), 815~885 step 기울기 급등 → 포화·열세; 포화 뒤 P 모델의 GRU 는 배경·전역 오프셋만 공급(0 치환 시 마스크 SSIM ≈ 그대로). 보고 `docs/smoke_deter_opt_report_2026-10-15.md` §8, 자동 보고 `results/smoke_deter_opt_2026-10-14/loss_compare.md`. **다음 = 사용자 결정 대기**(권고: 급등 차단 S4 — P 의 기울기 상한 1000 → 20, seed 3 + A2 재실행 ≈8 h; 2단계 F·L 은 같은 교란을 안아 비권고). GPU0 비어 있음 |
| **S4** | **기울기 급등 차단 확인 — P 에 기울기 크기 상한만 1000 → 20 (로컬 GPU0, 10-16 사용자 "그래 이걸로 진행해")** — 근거: S3 의 P 세 회차 모두 급등 전 GRU 미포화·마스크 L1 −8 % 우세 → 유한한 큰 기울기(상한 1000 통과; A2P 828·A3P 848·AP 1591 step, gn 1.5e3~1.5e6) 직후 포화·열세(`docs/smoke_deter_opt_report_2026-10-15.md` §8, 사후 관찰 = 가설). 하네스 설계 결정 17(preset 키 max_norm 하나; 기존 조건 키 없음 → 동작·config_hash 그대로, 완료 15 조건 hash 재현 확인; 이전 실행본 `harness_as_run_P_md5_fc7e5950.py` 보존). 조건: A2Pc·A3Pc·APc(P, GRU seed 2·3·1) + A2c(P0 seed 2 — A2 만 gn > 20 이 2 step). 재사용: ref·refP·A·A3 은 4000 step 내내 gn < 20 → 상한 20 에서 같은 궤적(clip_grad_norm_ 이 정확히 1.0 을 곱함, selftest 확인). 판정(실행 전 고정, `v8_eter_pure/smoke_deter_opt_clip_compare.py`; 첫 판을 검토 워크플로(3 관점 + 반박 검증) 지적으로 고친 판): s* = 기준 조건의 첫 gn > 20 step(AP 1591·A2P 828·A3P 848·A2 722), E_prefix(s* 까지 비트 같음)·s* 뒤 발산·clip ≥ 1 확인; 안정 타당성 E3' = s* 뒤 건너뜀 비율 ≤ 1 %·연속 ≤ 20(S3 의 E3 는 배율 하한의 고립 overflow 도 세어 결과에 좌우되므로 대체 — 실패는 결측이 아니라 "불안정 미제거(overflow 형)" 결과). Q1(1차 가설) = R 마스크 L1(A_kPc vs refP, 3001~4000): 모든 CI 상한 < 0·중앙값 ≤ −3 % → 우세 / 모든 CI 하한 > 0·중앙값 ≥ +3 % → 열세 / 모든 |R| ≤ 3 %·CI ⊂ ±5 % → 동등 / 그 밖 보류. Q1b(1차 처치 효과) = 같은 seed Δ = R(A_kPc) − R(A_kP), 블록 짝 t9: 모든 CI 상한 < 0·모든 Δ ≤ −10 %p → 상한이 도움 / 모든 |Δ| ≤ 3 %p → 효과 없음 / 그 밖 보류. 기전(해석용) = s* 뒤 probe sat_v 최댓값 ≤ 0.10 이면 포화 억제(억제 실패는 급등 가설의 반증 아님). Q2(2차) = 짝 맞춘 S3 규칙. 사전 점검(GPU): Ac(meas.log)·A2Pc(A2P)·A3Pc(A3P)·A2c(A2) 30 step 비트 일치·GRU 초기값 일치 | 약 7.8 h (4 × 1.95) | ▶ **10-16 07:23 UTC `CONDS="A2Pc A3Pc APc A2c"` 실행 시작**(하네스 md5 90cbc106, 판정 스크립트 md5 d6983cc7; 최종판 사전 점검 APc·A2Pc 30 step 비트 일치), 끝 ≈15:10 UTC → overflow 로 늦어져 ≈17~19 UTC. · **A2Pc 10:11 완료 — E3' 실패(overflow 형)**: 828~831 기울기 급등(상한 20 으로 잘려도 GRU 출력 RMS 0.07→0.50) → 832 부터 거의 매 step fp16 overflow, ≈870~3070 step 갱신 전부 건너뜀(amp_skip 2239·amp_floor 2233) → ≈3070 에 포화(sat_v 0.62)로 빠져나온 뒤 학습 재개, 3001~4000 자기 목적함수 비 +117 %, E256 SSIM 0.6542(zero-filled 0.7529 보다 낮음). 상한 1000 판(A2P)은 큰 갱신으로 바로 포화돼 overflow 구간을 피했음 — 포화가 overflow 에서 빠져나오는 길로 보임(기술). · **A3Pc 12:08 완료 — 타당(valid)**: 848 step gn 1.5e6 을 20 으로 잘랐으나(상한 적용 7 회, overflow 건너뜀 6·배율 하한 0) GRU 는 같은 구간에 포화(s* 뒤 sat_v 최댓값 0.40 > 0.10 → 포화 억제 안 됨), R 마스크 L1 +25.9 % [t9 CI +22.1, +29.8] (상한 1000 판 A3P +24.9 %), Q1b Δ +1.0 %p [−3.9, +5.9], E256 SSIM 0.8091(A3P 0.8187·refP 0.8443). · **APc 14:50 완료 — E3' 실패(overflow 형)**: 1591 step gn 1758 을 20 으로 자른 뒤에는 포화 없음(AP 와 달리; sat_v ≤ 0.05) → 2210·2226 step gn 1.2e9·2.8e18(상한 적용) → ≈2240 부터 4000 까지 매 step fp16 overflow 로 갱신 건너뜀(amp_skip 1805·amp_floor 68, s* 뒤 건너뜀 74 %·연속 1774), GRU 출력 RMS 0.45 에서 멈춘 채 포화 없음(sat_v 0.01) → E256 SSIM 0.281·PSNR 11.6 dB(AP 0.7782). **Q1 확정 = 불안정 미제거(overflow 형)**(valid 1 = A3Pc, overflow 2 = A2Pc·APc; 사전 규칙 N_MIN 2 미달), Q1b = 기술만(valid 1, Δ +1.0 %p). 기술: 상한 1000 판은 급등 직후 포화로 overflow 를 피했고, 상한 20 판은 포화 없이 GRU 출력이 커진 회차(A2Pc·APc)에서 overflow 가 이어짐(배율 하한 1024 라 배율로 회복 불가; overflow 는 상한 적용 전 단계라 상한 값과 무관) — 원인 미확인. A2c(Q2 용) 14:50 실행 시작 · **✅ 10-16 16:46 UTC 전체 완료 — 판정(`results/smoke_deter_opt_2026-10-14/clip_compare.md`): Q1 불안정 미제거(overflow 형) · Q1b 기술만(A3Pc Δ +1.0 %p) · 기전 포화 억제 안 됨(A3Pc sat_v 0.40) · Q2 기술만(짝 맞는 타당 회차 1)**. A2c 타당(A2 대비 자기 목적함수 −2.5 %, 상한 영향 거의 없음). 결론: 상한 20 은 P 의 GRU 불안정을 없애지 못하고 끝(포화/overflow)만 바꿈 → 사전 등록 가설 미지지. 보고 `docs/smoke_deter_opt_report_2026-10-15.md` §9. 손실 요인 줄기 종료 권고 |
| **P** | **교수님 설정 재현 학습 (MDPI 용, 10-12 사용자 결정)** — 교수님 순수 ETER-net 학습 설정을 받은 뒤 ETER-net(기준 모델) → U-Net 단독 → SS2D. 붕괴 감시를 켜고 기준 모델이 7만 step 을 넘겨도 정상이면 나머지 시작. 대조표·질문 목록·데이터 준비 = `docs/professor_settings_compare_2026-10-12.md` §5 | 런당 약 1~2 일 추정(ViT 레시피 길이 기준, 미측정) | ⏳ 교수님 설정 대기 |
| 3 | Transformer 50ep seed0 (③번째 모델, 구현 세부는 `axial_transformer_arm_design.md`) | ~5.6일 | 구현 완료 |
| 4 | E3 최소-GRU 50ep | ~4일 | 확정 |
| 5 | E2 LR 스윕 | ~2일 | 확정 |
| 6 | **radapt 재개** (잔여 ~23ep) | ~2.6일 | 전체 상태 재개 대기 |
| 7 | 추론-only 일괄 (Table 4·PromptMR+/DDS·ms/VRAM·가속화 계수별 평가·ringing) | 1~2일 | 준비 완료 |
| 8 | ~~U-Net-only 학습~~ | — | **2단계로 이동(10-01)** |

**⚠ 10-07 학습 붕괴 발견**: v8·v9 전 런에서 시퀀스 모듈·U-Net 약 97% 가중치가 epoch 5 이전 0 으로 붕괴(`docs/v8_training_collapse_audit_2026-10-07.md`). 아래 3~7 단계(Transformer·E3·E2·radapt 재개·추론 일괄)는 **레시피 수정 스모크(S) 결과가 나올 때까지 보류**. 기존 표의 GRU/SS2D/v9/U-Net 단독 수치는 시퀀스 모듈 효과가 아님.

**7단계 세부(09-03 추가, 논문 표 관례 정비 `docs/paper_table_conventions.md` §3)**: (a) 가속화 계수별 평가는 전체 val 7,334 슬라이스로 재실행(현 `results/eval/v8_r_sweep/` 은 stride-4 서브샘플 n=1,834 → 논문 표 불가), 세 모델 + Zero-filled, R∈{2,4,6,8}; (b) 모델별 추론 ms/slice·peak VRAM(배치 크기(BS) 1, AMP, 동일 GPU) → 표 5 [TBD] 칸; (c) 공개 U-Net/E2E-VarNet 전체 val(n=464 볼륨, 실측 코일만 전달) — 누수 참고 결과·각주용 — **본 연구 프로토콜 행은 PromptMR+ 까지 포함해 09-06 CPU 로 선행 실행 시작**(`v8_eter_pure/eval_baselines_full.py` → `results/eval/baselines_384_full/`, GPU0 불사용, `docs/frontier_baselines_plan.md` §3; 7단계엔 native 행·ms/VRAM 만 남음); (d) fastMRI 표준 프로토콜(320 center-crop·무마스크·볼륨 단위) 1회 산출 — 외부 수치와의 비교 가능성 확보. Zero-filled 기준선은 CPU 로 09-03 완료(`results/eval/zero_filled/`).

합계(모두 GPU0 순차) ≈ **40일 → 10월 중순 종료**(통제 비교 보강 실험 ~33일 + radapt/평가 ~7일).
SI 마감 11-30 대비 집필 병행 필요. (구 상의 항목 (a)(b)는 09-02 결정으로 종결 — 위 '결정' 절.)
