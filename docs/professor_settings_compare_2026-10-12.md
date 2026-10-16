# 교수님 원본 ETER-Net 학습 설정 대조 (2026-10-12)

사용자 질문(10-12): "교수님 실험의 세팅은 어떻게 되어 있었나? 저번(v8)과 같은가? 지금 학습되는 세팅이 같았으면 좋겠다."
교수님 원본 코드, 이전 v8, 현재 v8fix를 45개 항목으로 대조했다. 방법은 코드와 문서 읽기 4갈래, 종합 1회, 항목별 검증 3갈래다. 검증 단계에서 15개 항목이 정정됐다. 핵심 수치는 원본 파일에서 다시 확인했다(§1).
코드는 아무것도 고치지 않았다.

## 요약

- **결론:** 지금 학습(v8fix)은 교수님 원본과 같지 않다.
  - **같은 것:** 네트워크 구조. bi-GRU(flatten-reshape 스캔) → zero-filled 영상과 연결 → U-Net DFU(depth 5, wf 6)이고, DC는 없다. 384×384, 16코일, 32채널 입력도 같다.
  - **다른 것:** 데이터와 학습 방식 대부분.
  - **H:** 공개 model.py는 12, v8/v8fix는 10이다. 교수님 확인 대기.
- **비교할 수 없는 부분:** 교수님 순수 ETER-net 학습 설정 파일(`myConfig_ETER_fastmri_brain_acs32R4_5`)이 저장소, git 기록, 이 서버 디스크 어디에도 없다. 그래서 LR, BS, epoch, weight decay, 손실 종류, λ_SSIM을 알 수 없다. 이 서버의 교수님 공유 폴더(`/mnt/sda/choh/shared/`)에도 ViT 계열 model3 로그 하나뿐이다.
- **학습 설정이 전부 남은 교수님 원본은 ViT+ETER 하이브리드(240916) 하나다.** 이 원본은 네트워크가 다르다(ViT-B + GRU + 3×3 conv, U-Net 없음). 그래서 아래 표의 학습 항목에서 교수님 값은 대부분 이 스크립트 값이다.
- **10-07 붕괴를 일으킨 조합은 교수님 설정이 아니다.** 결합형 L2 3e-5, clip 1.0, ×1e6 단위 L1의 조합은 v5~v8에서 바뀐 부분이다. weight decay 1e-7→3e-5는 v5에서, clip과 스케일은 v8에서 바뀌었다.
- **결정(10-12, 사용자 "둘 다 진행"):** IEIE(10-19)는 지금 설정으로 마무리하고, 방법 절에 원본과 다른 점을 명시한다. MDPI(11-30)용으로는 교수님 설정을 확보한 뒤 재현 학습을 한다(§5).

## 1. 교수님 원본 출처와 읽은 방법

| 갈래 | 파일 | 남아 있는 것 | 비고 |
|---|---|---|---|
| (a) ViT+ETER 하이브리드 2024 | 루트 `choh_train_ViT_ETER_R4regular_240916py` + `configs/myConfig_choh_ViT_ETER_R4regular.py`(v1, 실효 PATH_FOLDER = run 29) + `dataloaders/myDataloader_fastmri_brain_240817.py`(`_v4_1`) | 학습 설정 전체 | 네트워크가 다름(ViT-B 인코더·디코더 + bi-GRU + 3×3 conv 1개, U-Net 없음) |
| (b) 순수 ETER-net 공개 코드 | `models/hybrid_eternet/hybrid_eternet_fastmri-main/`(model.py, 테스트 스크립트, 로더) | 모델 상수(H=12, U-Net depth 5)와 추론 경로 | 학습 스크립트 없음, 가중치 `tensors_R4.pt`는 저장소에 없음 |
| (c) 순수 ETER-net 학습 스크립트 | `scripts_legacy/choh_train_ETER_hybrid_fastmri_brain_acs32R4_231030.py`(d-ETER), `choh_train_oETER_*`(o-ETER) | 학습 루프 구조 | config 모듈이 없다. `myOptimizer(eternet, OPTIMIZER, LR)`를 인자 3개로 부르지만, 저장소의 `myUtils.myOptimizer`는 인자 2개만 받고 lr 1e-7로 고정돼 있다. 그 런에 쓰인 myUtils가 저장소의 것과 다르다는 뜻이다 |
| (d) model3 2025 | `scripts_legacy/choh_train_model3_*.py` + `configs/myConfig_choh_model3.py` | 학습 설정 | ViT 계열. LR 1e-4, 200 epoch, 무작위 마스크. 작업본 config는 0cdc914에서 축소됨 |

- **원본 값을 읽는 방법:** 교수님 값은 `git show 7d4e4e0:<path>`로 읽었다. 일부 원본 파일이 7d4e4e0 이후에 수정됐기 때문이다(§6).
- **"현재"의 범위:** "현재"는 ETER-net 기준 런 `PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval`을 말한다. 이 런은 ampguard 트레이너로 돌고, Pod 브랜치 `runpod/v8fix-pod3`에 있다. SS2D와 U-Net 단독 런은 정밀도만 다르다(T11).
- **원본 파일에서 직접 확인한 값(10-12):**
  - `list_brain_train_320.txt`는 379줄이고 전부 AXT2다.
  - 240817 로더는 `np.fft.ifft2`를 기본 backward 정규화로 쓴다. ortho 줄은 주석 처리돼 있다.
  - 240817 로더의 세기 배율은 1e6/1e4/1e6이고, 마스크는 `3:384:4` + `192-16:192+16`이다.
  - 230425 로더의 세기 배율은 1e5/1e3/1e5다.
  - model.py는 `N_HIDDEN_LRNN_1/2 = 12`다.
  - `myCriterion`의 기본값은 MSE다.
  - 240916은 `scheduler.step()`을 배치마다 부르고, clip·autocast·GradScaler가 없다.

## 2. 한눈에 보기

- **같음:** D6 해상도, D7 k-space 격자(배율만 다름), D9 입력 배치, M1 모델 계열, M3 GRU 구조·스캔, M6 출력 head, M7 DC 없음, T14 초기화 분포, T17 조기 종료 없음.
  - M4·M5(채널 수, U-Net 폭)는 H 때문에만 다르다.
- **다름 — 데이터:**
  - 학습 데이터와 대조도
  - 슬라이스 구성, 학습 중 검증
  - FFT 정규화, 세기 스케일
  - 마스크, 증강
  - 뇌 마스크 손실, 데이터 순서
- **다름 — 학습:**
  - λ_SSIM, SSIM 구현
  - optimizer, weight decay, LR, 스케줄
  - BS, 길이, clip, 정밀도, 시드
  - 검증·best 기준, 최종 평가
- **알 수 없음:**
  - 교수님 순수 ETER-net의 학습 하이퍼파라미터 전부: LR, BS, epoch, optimizer/WD, 손실(CRITERION), λ_SSIM.
  - 학습에 쓴 H.
  - 하드웨어.

## 3. 전체 대조표 (45개 항목)

범례:
- **판정**은 "교수님 원본 대비 현재 v8fix" 기준이다.
- **[ViT]**는 ViT 하이브리드 240916 값, **[순수]**는 순수 ETER-net 코드 값이다.
- 근거의 `:줄`은 저장소 기준 경로다. `git show 7d4e4e0:`로 읽은 것은 그렇게 표시했다.
- 검증 단계에서 정정된 항목은 ID에 *를 붙였다.

### 3.1 데이터 (D1~D16)

| ID | 항목 | 교수님 원본 | 이전 v8 | 현재 v8fix | 판정 | 근거 |
|---|---|---|---|---|---|---|
| D1* | 학습 데이터·크기 | 교수님 .mat k-space(`mat_brain_multi_train_16ch_396ky_ksp`, 이 서버에 AXT2 620개·334 GB), 목록 0~299번 = 300볼륨. 100볼륨씩 3묶음 × 16칸 = epoch당 4,800 슬라이스, 그중 22칸은 채움(D3). [순수] 설정 파일은 없지만 런 이름('t300', 'train/val/test 300/20/59')으로 보아 같은 300볼륨 | 로컬 h5 4,108/4,110 파일(잘린 2개 제외), 65,028 슬라이스 | Pod 4,110 파일(2개 온전), 65,060 슬라이스(BS 8에서 8,133 step/epoch) | 다름 | `configs/myConfig_choh_ViT_ETER_R4regular.py:35-36`, `choh_train_…240916py:216,219-224`, `myDataloader_fastmri_brain_240817.py:76-102`, `configs/myConfig_temp.py:117-166`, pod log `steps=8133` |
| D2* | 대조도 | AXT2만(목록 379개 전부 AXT2; 스캐너 태그 200 ×275, 201 ×78, 204 ×26) | 혼합: AXT2 2,461 / AXT1POST 874 / AXFLAIR 314 / AXT1PRE 235 / AXT1 224 | 혼합(4,110: AXFLAIR 315, AXT1PRE 236, 나머지 같음) | 다름 | `dataloaders/list_brain_train_320.txt`, `hybrid_eternet_fastmri-main/list_brain.txt`, `infra/runpod/dataset_manifest_nvme.json` |
| D3* | 볼륨당 슬라이스 | 항상 16칸. 원래 슬라이스가 16개보다 적은 9볼륨은 .mat에서 6~7번 슬라이스를 14~15칸에 반복하고, 1볼륨(10장)은 4칸이 0이다. 4,800칸 중 22칸이 채움 | h5 전 슬라이스(개수 가변) | v8과 같음 | 다름 | `myDataloader_fastmri_brain_240817.py:70-71,100-102`, `dataloader_h5_v5.py:136-157` |
| D4 | 학습 중 검증 데이터 | [ViT] 없음(검증 블록 주석, 0 저장). [순수] epoch마다 1볼륨(d-ETER 목록 300번, o-ETER 310번), 마지막 배치 손실만 출력 | 로컬 val 0번 464볼륨/7,334 슬라이스 | 공식 val 0~2 전체 1,378볼륨/21,842 슬라이스(10-09 사용자 결정) | 다름 | `choh_train_…240916py:279-288,300-301`, `choh_train_ETER_hybrid_…231030.py:268-277`, plan R1 |
| D5* | train–val/test 중복 | 없음(train 0~299, test 320~378). 아래 4볼륨은 교수님 목록 64~67번째 줄, 즉 train 범위 안이다 | 로컬 val 0번(464)에 train 볼륨 4개 섞임(`file_brain_AXT2_200_2000{306,308,310,311}`, 바이트 동일) | **없음** — Pod val 0~2(1,378)에 이 4개가 없고, train에만 있음(10-12 Pod 직접 확인; `verify.log` "multicoil_val … 없음 4") | 같음 | `list_brain_train_320.txt:64-67`, `infra/runpod/pod_verify_data.py:72-84`, pod2 `/root/_setup/verify.log:4` |
| D6 | 영상 크기 | 384×384(N_OUTPUT=384), 로더에서 crop·pad 없음 | 384×384 | 384×384 | 같음 | `myDataloader_fastmri_brain_240817.py:63-66`, `myConfig_pure_eter_v8.py:16`, `docs/version_evolution.md:205` |
| D7* | 384×384 k-space 만드는 법 | 오프라인 .mat: 768×396 원 k-space를 backward ifft → 영상에서 가운데 384×384 crop → backward FFT. .mat 4개 실측 = v8 384×384 k-space ÷ √(768·396)/384 ≈1.436(상대오차 ≤5.5e-16) | ortho ifft2c → 코일 영상 축별 crop/zero-pad → ortho fft2c | v8과 같음 | 같음(격자 동일, 전역 배율만 다름 — D10·D13) | `myDataloader_fastmri_brain_240817.py:100`, `dataloader_h5_v5.py:30-72,193-212` |
| D8* | 코일 처리 | 모든 볼륨이 원래 16코일(AXT2 목록, 로컬 h5 348/348 확인). 선택·채움·압축 없음, 모델은 16 고정 | N_COIL=16: 파일 순서 앞 16개. 적으면 0 채움, 많으면 자름(원 코일 수 2~28, train 슬라이스 26.9%가 16 초과) | v8과 같음 | 규칙 같음·데이터 다름 | `git show 7d4e4e0:models/hybrid_eternet/u_choh_model_ETER_ViT.py:37`, `dataloader_h5_v5.py:228-232`, 감사 :46 |
| D9 | 입력 표현 | k-space (B,32,384,384) 16코일 × (re,im) 교대 배치, 미획득 줄 0. 둘째 입력 = 코일별 zero-filled 영상 32채널(RSS 아님) | 같은 배치. data_img = 100·ifft2c(data) | v8과 같음. 트레이너가 k-space ×100/s, 영상 ×1/s(D13) | 같음 | `myDataloader_fastmri_brain_240817.py:105-106,119-122`, `dataloader_h5_v5.py:173-178,234-235` |
| D10 | 영상·정답의 FFT 정규화 | numpy `ifft2` 기본 backward(ortho 줄은 주석), fftshift/ifftshift | ortho | ortho(이후 슬라이스별 정규화) | 다름 | `myDataloader_fastmri_brain_240817.py:38-44`, `dataloader_h5_v5.py:30-41`, 감사 :20,:54 |
| D11 | 언더샘플링 마스크 | 고정 regular: `3:384:4`(96줄) + ACS 32(`192-16:192+16`), train·test 모든 슬라이스 같은 마스크 = 120/384줄, R_eff 3.2(이름은 R4). 변형: o-ETER random(볼륨마다 96줄, 시드 없음), hybrid-main fixed(`R4_idx_part*.npy`), model3 R4random/R8 | 등간격 R4: center_fraction 0.08 → ACS 31(177~207열) + `mask[offset::4]`. train은 샘플마다 offset 무작위 [0,4), val은 offset 3. 약 119/384줄 | v8과 같음(train 무작위, val·test offset 3) | 다름(val 마스크는 offset 같고 ACS 31 vs 32) | `myDataloader_fastmri_brain_240817.py:114-115`, `dataloader_h5_v5.py:79-93,112-116,215-222` |
| D12* | 정답 영상 | 384×384 .mat k-space의 backward ifft로 16코일 RSS(로더에서 계산). 수치상 fastMRI `reconstruction_rss` ÷ 551.48(상관 1.0) | h5 `reconstruction_rss`(전 코일 RSS), 384로 crop/pad, ×1e6 | v8과 같음. 학습 때 ÷s, 검증은 출력×s와 ×1e6 정답 비교 | 정의 같음·배율 다름. 혼합 데이터에서만 생기는 불일치 있음: 16코일 초과 슬라이스(정답에 입력에 없는 코일 포함, RSS16/GT 평균 0.896), 320×320 저장 RSS의 384 zero-pad(train 슬라이스 58%) | `myDataloader_fastmri_brain_240817.py:108-110`, `dataloader_h5_v5.py:186,198,241`, 감사 :114 |
| D13* | 세기 스케일·정규화 | 고정 상수, 슬라이스별 정규화 없음. [ViT] `_v4_1`: 영상 ×1e6, k-space ×1e4, 정답 ×1e6. [순수] 로더(230425·random_231027·hybrid-main): ×1e5/×1e3/×1e5 | 고정 ×1e6/×1e4/×1e6 + ortho. 정답 최대 약 170~1,300, L1 약 9~18 | 로더 ×1e6/×1e4/×1e6 후 트레이너가 샘플마다 s = zero-filled RSS 99 백분위수(≥1e-6)로 영상·정답 ÷s, k-space ×100/s. 검증은 출력×s | 다름 | `myDataloader_fastmri_brain_240817.py:56-69`, `myDataloader_fastmri_brain_230425.py:83-85`, `smoke_recipe_fix.py:324-334`, `results/smoke_recipe_fix/summary.md` |
| D14 | 증강 | 없음 | train만 상하 flip p 0.5 + 좌우 flip p 0.5(영상 영역, 정답도 같이) | v8과 같음 | 다름 | `myConfig_pure_eter_v8.py:40-41`, `dataloader_h5_v5.py:201-209`, `docs/eter_v5_changes.md:15` |
| D15 | 손실의 뇌 마스크 | 없음(384×384 전체에서 L1·SSIM) | Otsu(정답 ≠0)×0.4 + 최대 연결 성분. 마스크 L1, 마스크 SSIM 손실, 검증 지표 전부 | v8과 같음 | 다름 | `choh_train_…240916py:247-249`, `dataloader_h5_v5.py:243-265`, `docs/version_evolution.md:213-222` |
| D16 | DataLoader·데이터 순서 | `shuffle=True`, num_workers 0. epoch 안에서 100볼륨 묶음을 다시 읽어, 섞임은 묶음 안에서만이고 묶음 순서는 0,1,2 고정 | 전 슬라이스 섞기, train 16 워커·prefetch 4. 시드 런은 시드 generator | 전 슬라이스 섞기, generator를 epoch마다 SEED+1000·epoch로 재시드(재개해도 같은 순서), 워커 rng 재시드, prefetch 2, val BS 4·12 워커 | 다름 | `choh_train_…240916py:219-228`, `main_train_pure_v8.py:233-252`, `main_train_pure_v8fix.py:24-26` |

### 3.2 모델 (M1~M9)

| ID | 항목 | 교수님 원본 | 이전 v8 | 현재 v8fix | 판정 | 근거 |
|---|---|---|---|---|---|---|
| M1 | 모델 계열 | (a) [ViT] ViT-B + bi-GRU + 3×3 conv, U-Net 없음. (b) [순수] bi-GRU → cat(aliased 영상) → UNet_choh_skip | 순수 ETER-net(b) | 순수 ETER-net(b) | 같음(순수 기준) | `hybrid_eternet_fastmri-main/model.py:18-65`, `u_pure_eternet_gru.py:63-78,107-114` |
| M2 | GRU hidden 배수 H | [순수] 공개 model.py 12/12(방향당 4,608). [ViT]·model3 원 config 10/10 | 10/10(방향당 3,840) | 10/10 | 다름(교수님 확인 대기) | `model.py:6-15`, `myConfig_choh_ViT_ETER_R4regular.py:64-65`, `paper/draft_ko_v2.md:628-633` |
| M3 | bi-GRU 구조·스캔 | 1층 양방향 GRU 2개: gru_h 입력 12,288 → 384·H, gru_v 입력 384·H·2. h0 = 0. flatten-reshape `x.reshape([B,384,-1])` → 한 스텝 = 한 코일 채널의 연속 32행. ([ViT]는 행 단위: 한 행 × 32채널) | model.py 그대로(출력 max\|diff\| = 0) | v8과 같음(ampguard는 gru_h fp32 — T11) | 같음 | `model.py:42-59`, `u_pure_eternet_gru.py:49,57-66,96-105`, 감사 :25,:44 |
| M4* | 시퀀스 출력 채널·병합 | [순수] out_v 2·H = 24채널 + aliased 32 = U-Net 입력 56. [ViT] cat(ViT tail 16, aliased 32, GRU 20) = 68 | 20 + 32 = 52 | 52 | 병합 구조 같음, 채널 수만 H 차이 | `model.py:31,62`, `u_pure_eternet_gru.py:69,104,109` |
| M5* | U-Net(UNet_choh_skip, DFU) | depth 5, wf 6(64~1024), BN 없음, upconv, avg_pool, DFU 잔차(x − residual), 마지막 1×1 conv(입력과 마지막 특징 cat), n_hidden = H+16 | 같은 클래스, in 52, n_hidden 26, 약 31.1M | v8과 같음 | 같음(H 의존 폭만 다름: in 52 vs 56, 마지막 conv 116→1 vs 120→1, 31.060M vs 31.062M) | `models/hybrid_eternet/myUNet_DF.py:79-195`, `myConfig_pure_eter_v8.py:29-30` |
| M6 | 출력 head | [순수] U-Net 마지막 1×1 conv, 활성화 없음. [ViT] Conv2d(68→1, 3×3). 작업본의 RefinementBlock(e0dc4c9)은 교수님 코드 아님 | 1×1 conv(116→1) | 같음(정규화 단위 출력) | 같음 | `myUNet_DF.py:166,193`, `git show 7d4e4e0:…/u_choh_model_ETER_ViT.py:75-76,140` |
| M7 | DC 블록 | 없음 | 없음(USE_DC=0) | 없음 | 같음 | `model.py:47-65`, `docs/eternet_paper_data_consistency.md` |
| M8 | 전체 파라미터 | [순수] H=12: 880,519,481(H=10이면 668,162,101). [ViT] v1: 845,985,613 | 668.2M | 668.2M(gru_h 371.6M + gru_v 265.5M + U-Net 31.1M) | 다름(H에서 옴) | model.py meta-device 계수, pod log `params=668.2M` |
| M9 | ViT 설정 | [ViT]만: 384, patch 32, dim 768, depth 12, heads 12, mlp 3072, dropout 0.1 | 없음 | 없음 | 해당 없음 | `choh_train_…240916py:134-160` |

### 3.3 학습 (T1~T20)

| ID | 항목 | 교수님 원본 | 이전 v8 | 현재 v8fix | 판정 | 근거 |
|---|---|---|---|---|---|---|
| T1 | 손실 형태 | [ViT] `nn.L1Loss`(전체 영상 평균) + 0.2·(1−SSIM), 마스크 없음. [순수] `myCriterion(CRITERION)` + λ·(1−SSIM) — 기본값 MSE, 실제 CRITERION은 config가 없어 모름 | 마스크 L1 + 1.0·(1 − 마스크 SSIM), ×1e6 단위(L1이 손실의 약 98.5%) | 마스크 L1 + λ·(1 − 마스크 SSIM), 정규화 단위. L1 : λ·SSIM 항 ≈ 64~69 : 1 | 다름 | `choh_train_…240916py:195-196,247-249`, `choh_train_ETER_hybrid_…231030.py:190,235-236`, `scripts_legacy/myUtils.py:111-120` |
| T2 | λ_SSIM | [ViT] 0.2(model3도 0.2). [순수] 모름 | 1.0(×1e6 단위) | 1/s̄ = 0.0030745725(s̄ = 325.25, train 64 슬라이스 평균; 로컬 점검 런은 0.002822) | 다름 | `myConfig_choh_ViT_ETER_R4regular.py:43`, `smoke_recipe_fix.py:166,337-347`, `paper/project_story_v1_to_v9.md:255` |
| T3* | SSIM 손실 구현 | 원 `u_choh_SSIM`(7d4e4e0): 가우시안 창 11·σ 1.5, valid conv, 전체 SSIM 지도 평균. L = 255(pred max > 128) 아니면 1; import 시 `cudnn.deterministic=True` | 같은 모듈 + `mask=` 인자(afa89a0). ×1e6 단위라 초기 L 2, pred max > 128 이후 255/256 | 같은 마스크 모듈. 정규화 단위라 pred max < 128 → L 1(또는 2) — 추론, 로그 없음 | 다름 | `git show 7d4e4e0:models/hybrid_eternet/u_choh_SSIM.py`, 현 `u_choh_SSIM.py:24-37,54-64`, 감사 :48 |
| T4 | Optimizer | [ViT] `torch.optim.Adam` 단일 그룹, 기본 betas, eps 1e-8. o-ETER 같은 형태. d-ETER 231030은 복원 불가(myOptimizer 인자 불일치) | Adam 단일 그룹 | AdamW 2그룹(가중치 감쇠 / 바이어스 전부 wd 0), eps 1e-8 | 다름 | `choh_train_…240916py:197`, `myUtils.py:21,84-88`, `smoke_recipe_fix.py:420-470` |
| T5 | Weight decay | [ViT]·model3: 1e-7 결합형 L2(Adam 안, 전 파라미터). [순수] 모름 | 3e-5 결합형 L2(전 파라미터) | 3e-5 분리형(AdamW, 가중치만). 실효 수축 lr·wd = 6e-9/step ≈ 0 | 다름 | `myConfig_choh_ViT_ETER_R4regular.py:42`, `myConfig_pure_eter_v8.py:37`, `docs/eter_v5_changes.md:13`, 감사 :55 |
| T6 | 학습률 | [ViT] v1 1e-3(run 29). v2·legacy 사본·model3 1e-4. [순수] 모름 | 2e-4 | 2e-4 | 다름 | `myConfig_choh_ViT_ETER_R4regular.py:41`, `…_v2.py:23-24`, `myConfig_pure_eter_v8.py:36` |
| T7 | LR 스케줄 | [ViT] `CosineAnnealingWarmRestarts(T_0=40, T_mult=2, eta_min=0)`을 **배치마다** step → 40, 120, 280, 600 … 에서 재시작(24만 step 동안 약 12회), warmup 없음. [순수]·model3 'WOanneal'·legacy 사본: 스케줄러 없음(고정 LR) | CosineAnnealingLR T_max 406,450, eta_min 1e-6, step마다 | CosineAnnealingLR T_max 406,650, eta_min 1e-6, 재시작 없음 | 다름 | `choh_train_…240916py:199-203,254`, `main_train_pure_v8.py:265-268,387`, `docs/eter_v4_analysis.md:19-21` |
| T8 | 배치 크기 | [ViT] 2(model3도 2). 교수님 d-ETER 테스트 스크립트는 4. [순수] 학습 BS 모름 | 8(SMOKE_BS가 config 2를 덮어씀), ACCUM 1 | 8, ACCUM 1, val BS 4 | 다름 | `myConfig_choh_ViT_ETER_R4regular.py:39`, `v8_eter_pure/runs/smoke_bs.txt` |
| T9 | 학습 길이 | [ViT] 100 epoch × 2,400 step = 24만 step(슬라이스 약 48만 회). model3 200 epoch. [순수] 모름 | 50 epoch, 406,450 step | 50 epoch, 406,650 step, 슬라이스 약 325만 회 | 다름 | `myConfig_choh_ViT_ETER_R4regular.py:40`, `infra/runpod/pod_run_arm_ampguard.sh` |
| T10* | Gradient clip | 없음(240916, scripts_legacy 순수 ETER 모두) | 전역 clip 1.0. 학습 초기 기울기 노름 44~98 → 계수 0.01~0.02 | max_norm 1e3(안전장치). 매 epoch clipped=0(ep1 gn_max 166, 이후 3~8) → 사실상 clip 없음 | 다름(실효상 교수님과 같음) | `choh_train_…240916py:250-253`, `main_train_pure_v8.py:383-384`, pod log `clipped=0` |
| T11* | 정밀도 | 전부 fp32(autocast·GradScaler 없음) | AMP: 순전파 전체 fp16, 손실 fp32, TITAN RTX(TF32 없음) | ETER-net: gru_h만 fp32, gru_v·U-Net fp16, TF32 끔. SS2D: 전체 fp16(selective-scan 입력만 모듈 안에서 fp32). U-Net 단독: fp16. Mamba-ETER: h·v fp32, U-Net fp16 | 다름(모델 간 비대칭 — 방법 절 명시 필요) | `choh_train_…240916py:236-238`, `gru_precision.py:3-31`(pod), `models/mamba_eternet/ss2d.py:87-101` |
| T12* | 비유한·AMP 보호 장치 | 없음 | 손실 비유한 배치 건너뜀, 연속 300회면 종료 | 손실 비유한이면 1회 재계산 후 건너뜀. 기울기 비유한이면 재계산(AMP_GRAD_RETRY). GradScaler 배율 하한 1024. val 재시도 3회 → fp32 → 제외. ep16까지 nan_skip 0, floor 0, val 제외 0 | 다름(최적화에는 영향 없음) | `main_train_pure_v8.py:360-377`, ampguard 트레이너(pod 브랜치), pod log |
| T13* | 시드·결정성 | 시드 없음. SSIM import로 `cudnn.deterministic=True`만 | run 1 시드 없음. 1b·U-Net 단독·pixel-GRU SEED=1, E1 0/1/2 | SEED=1, 데이터 순서 SEED+1000·epoch(재개해도 같음). 비트 단위 재현은 아님(fp16 atomics) | 다름 | `git show 7d4e4e0:…/u_choh_SSIM.py:6`, `main_train_pure_v8.py:59-66`, `main_train_pure_v8fix.py:24-26` |
| T14 | 가중치 초기화 | PyTorch 기본(GRU 균등 ±1/√hidden, Conv kaiming-uniform), 시드 없음 | PyTorch 기본, 모델마다 U-Net 초기값 다름 | PyTorch 기본 + U-Net 초기값을 모든 모델이 공유(SAME_UNET_INIT, checksum 5.9212156172e+02) | 같음(분포) | `u_pure_eternet_gru.py:63-78`, `main_train_pure_v8fix.py:21-23` |
| T15 | 검증 주기·지표 | [ViT] 검증 없음. [순수] epoch마다 1볼륨, 마지막 배치 손실만 | 2 epoch마다 val 0번, 뇌 마스크 안 skimage SSIM(7×7)·PSNR·nMSE·L1 | 2 epoch마다 + 마지막 epoch, val 0~2, 같은 지표(원 단위), + 64 슬라이스 기능 시험(PROBE) | 다름 | `choh_train_…240916py:279-288`, `main_train_pure_v8.py:84-101,152-189` |
| T16 | best 체크포인트 기준 | 없음 — 매 epoch 전체 모델 저장(`tensors_{epoch}.pt`) | 자체 composite 지표(보고 인용 금지) | val 마스크 SSIM 최대. last.pt 매 epoch, epoch_N 5마다 | 다름 | `choh_train_…240916py:274-277,292,297`, `main_train_pure_v8.py:186-188` |
| T17 | 조기 종료 | 없음 | patience 50회(실제로는 발동 안 함) | 없음(50 epoch 전부) | 같음 | `myConfig_pure_eter_v8.py:44`, `main_train_pure_v8fix.py:31-37` |
| T18 | 붕괴 감시 | 없음 | 없음 | 500 step마다 거의 0인 가중치 비율(\|w\| < 1e-6). seq·U-Net 깊은 층 ≥ 50%면 ALERT_COLLAPSE·exit 3. 현재 약 0.01% | 해당 없음(최적화 무관) | ampguard 트레이너 :313-340(pod 브랜치), pod log HEALTH 줄 |
| T19* | 최종 평가 | 대화형 뷰어만(`myTEST_ViT_ETER_regular_240918.py`): eval·fp32·BS 4, R4 offset 3 + ACS 32, `--set testset` = 목록 320~378(AXT2 59볼륨·944 슬라이스). 입력한 슬라이스 하나의 skimage SSIM(전체 영상)·nMSE만 계산, 마스크·평균 없음 | 로컬 val 0번 슬라이스 짝 평가, 마스크 지표, 볼륨 SSIM 보고 | best ckpt를 test_full 0~2(558볼륨/8,852 슬라이스)와 val에서 평가: BS 1, AMP, offset 3, 슬라이스별 정규화, 마스크 지표. GRU 평가는 cuDNN TF32 기본(켜짐) — 학습과 다름 | 다름 | `scripts_legacy/myTEST_ViT_ETER_regular_240918.py:196-198,373-388`, `v8_eter_pure/eval_v8fix.py:4-7,46-55` |
| T20 | 하드웨어 | 명시 없음(런 폴더 이름 끝 'rtx2') | 로컬 TITAN RTX 24 GB, docker mri:v1, torch 2.3.1 | RunPod RTX 3090 24 GB. 10-10 pod3 시작, 10-12 ep15 끝에서 pod2로 이전(같은 트레이너 md5). torch 2.3.1+cu121 | 알 수 없음 | `myConfig_choh_ViT_ETER_R4regular.py:15-20`, `infra/runpod/README.md` |

## 4. 이전 v8 → 현재 v8fix 에서 바뀐 것

- **바뀌지 않은 것:**
  - 데이터 파이프라인과 모델(H=10, U-Net depth 5/wf 6, DC 없음)
  - LR 2e-4와 cosine(→1e-6) 스케줄, BS 8, 50 epoch
  - 2 epoch마다 검증, 마스크 L1+SSIM 형태
- **바뀐 것 — 대부분 10-07 붕괴 수정이다(`docs/v8_training_collapse_audit_2026-10-07.md`, `results/smoke_recipe_fix/summary.md`):**
  1. **Optimizer:** Adam 결합형 L2(전 파라미터) → AdamW 분리형 + 바이어스 no-decay 그룹. 붕괴의 필요조건을 없앤 변경이다.
  2. **Gradient clip:** 1.0 → 1e3(사실상 해제). clip 1.0은 붕괴를 증폭했다.
  3. **세기와 λ:** ×1e6 단위 + λ 1.0 → 슬라이스별 99 백분위수 정규화 + λ = 1/s̄. v8의 L1:SSIM 균형을 유지한 것이고, 교수님의 균형이 아니다.
  4. **best 기준:** composite → val SSIM. composite는 보고 금지 결정에 따라 뺐다.
  5. **검증 데이터:** 로컬 val 0번(464) → 공식 val 0~2 전체(1,378). 10-09 사용자 결정이다.
  6. **학습 데이터:** 65,028 → 65,060 슬라이스. Pod에서는 잘렸던 2파일이 온전하다.
  7. **재현성:** U-Net 초기값을 모든 모델이 공유하고, epoch마다 결정적인 데이터 순서를 쓴다. 통제 비교와 재개 일관성을 위한 것이다.
  8. **정밀도:** ETER-net gru_h만 fp32로 바꾸고 AMP 보호 장치를 넣었다. pod3에서 fp16 gru_h 기울기 오류로 ep1 batch 약 3,535에서 붕괴가 재현됐기 때문이다(10-10).
  9. **하드웨어:** TITAN RTX → RTX 3090.

## 5. 결정(2026-10-12)과 후속 계획

### 5.1 IEIE (10-19): 지금 설정 유지

- **이유 1 — 공통 설정:** 지금 런들(ETER-net, SS2D, U-Net 단독)은 모든 비교 모델이 같은 데이터, 레시피, 데이터 순서, U-Net 초기값을 쓴다. 정밀도만 다르다(T11). 통제 비교의 조건은 갖춘 셈이다.
- **이유 2 — 재학습 불가:** 설정을 바꾸면 모든 비교 모델을 다시 학습해야 한다. 지금 레시피로 50 epoch 한 런이 3090에서 약 4~6일이라 마감 전에는 불가능하다.
- **해야 할 일:** 방법 절에 원본과 다른 점을 적는다. 문안은 §5.3이다.

### 5.2 MDPI (11-30): 교수님 설정 재현 학습

1. **설정 확보 (선행):** §5.4 질문으로 교수님 순수 ETER-net 설정을 받는다. config 또는 그 런의 `log.txt`가 필요하다. 교수님 루트 스크립트는 시작할 때 설정 변수를 전부 출력한다(`choh_train_…240916py:77-82`). 받지 못하면 학습 설정이 남은 [ViT] 레시피를 쓰고 그렇게 명시한다.
2. **데이터 준비 — 다음 둘 중 하나로 한다:**
   - **(a) .mat 그대로 쓰기:** 교수님 .mat를 쓴다. 300볼륨은 약 160 GB이고, Pod에서 학습하려면 이를 전송해야 한다.
   - **(b) h5에서 다시 만들기:** 같은 300볼륨을 h5에서 같은 방식으로 다시 만든다. D7에서 격자가 같고 배율 1.436만 다르다는 것을 확인했다. 단 다음은 확인이 남았다.
     - 목록 379볼륨 중 로컬 h5에 있는 것은 348볼륨이다. 첫 300볼륨이 모두 있는지는 아직 모른다.
     - 16칸 채움 방식(D3)을 따라야 한다.
3. **붕괴 점검:** 교수님 레시피는 결합형 L2(1e-7), clip 없음, fp32다. 감사 결과, 결합형 L2는 1e-7이어도 기울기가 0인 가중치를 3.2만~6.5만 step 안에 0으로 만든다(감사 :55).
   - **점검 기준 — step 수:** [ViT] 레시피에서 1 epoch는 2,400 step이라 5 epoch는 1.2만 step뿐이다. 그래서 epoch 수가 아니라 step 수로 본다.
   - **진행 방식:** 기준 모델 ETER-net부터 지금 트레이너의 붕괴 감시(T18, 500 step마다, ALERT 시 중단)를 켜고 학습한다. 7만 step을 넘겨도 정상이면 나머지 모델을 시작한다.
4. **학습 순서:** ETER-net(기준 모델) → U-Net 단독 → SS2D. Mamba-ETER는 ep3 NaN 원인을 진단한 뒤에 넣는다.
5. **비용 추정(미측정):** [ViT] 레시피 길이(BS 2 × 24만 step = 슬라이스 약 48만 회)는 지금(약 325만 회)의 약 1/7이다. fp32를 감안해도 3090에서 런당 약 1~2일로 추정한다. 교수님 [순수] 설정이 다르면 다시 계산한다.
6. **비교:** 같은 ETER-net의 "교수님 설정"과 "지금 설정" 결과를 나란히 두면, 결론이 학습 설정에 얼마나 민감한지도 보고할 수 있다.

### 5.3 IEIE 방법 절 추가 문안 (안)

제1저자가 Word에서 다듬는 것을 전제로 한 초안이다. [ ] 안은 교수님 확인 후 확정한다. 교수님 공개 논문의 학습 설정은 저장소에 기록이 없어서, "원 구현"은 교수님 연구실 코드를 뜻한다.

> 본 연구의 모든 비교 모델은 동일한 학습 설정으로 학습하였으며, 이 학습 설정은 원 ETER-Net 구현의 학습 설정과 다음 항목에서 다르다. 첫째, 학습 데이터로 fastMRI 뇌 다중 코일 학습 데이터 중 4,110개 볼륨(65,060개 슬라이스)을 사용하였고, 이 데이터에는 AXT2 외에 AXT1, AXT1POST, AXT1PRE, AXFLAIR 대조도가 함께 포함된다. 원 구현은 16개 코일로 획득한 AXT2 볼륨 300개를 사용하였다. 둘째, 입력과 정답 영상의 세기는 슬라이스마다 영 채움(zero-filled) 영상의 RSS(root-sum-of-squares) 99 백분위수로 나누어 정규화하였으며, 원 구현은 고정된 상수 배율을 사용하였다. 셋째, 학습 시에는 등간격 언더샘플링 마스크의 시작 위치를 슬라이스마다 무작위로 정하고 중앙 31개 줄을 ACS(auto-calibration signal) 영역으로 포함하였으며, 검증과 시험에는 시작 위치를 고정하였다. 원 구현은 모든 슬라이스에 시작 위치와 32개 줄의 ACS 영역이 고정된 마스크를 사용하였다. 넷째, 손실 함수는 뇌 영역 마스크 안에서 계산한 L1 손실과 SSIM(structural similarity index measure) 손실의 가중합이며, 학습 영상에는 상하·좌우 뒤집기 증강을 적용하였다. 다섯째, 최적화에는 AdamW를 사용하였고, 학습률 2×10⁻⁴에서 코사인 일정으로 감소시키면서 배치 크기 8로 50 epoch 동안 학습하였다. 가중치 감쇠(weight decay)는 실효적으로 적용되지 않았다(설정값 3×10⁻⁵ 는 이 학습률에서 한 step 감쇠량이 단정밀도 가중치의 표현 간격보다 작아 가중치를 바꾸지 못한다). [10-16 추가 — `docs/v8fix_setup_audit_2026-10-14.md` 의 WD no-op 확인. 지금 IEIE v8 초안 68 줄의 "Adam(weight decay 3×10⁻⁵)·×10⁶ 배율·gradient norm 상한 1.0" 은 붕괴했던 옛 레시피 서술이라, 새 결과로 갱신할 때 이 문단으로 바꾼다.] 여섯째, 계산은 반정밀도(fp16, 16-bit floating point) 혼합 정밀도로 수행하였으나, 기준 모델의 가로 방향 GRU(gated recurrent unit)는 반정밀도에서 기울기 오차로 학습이 붕괴하여 단정밀도(fp32, 32-bit floating point)로 계산하였다. GRU 은닉 크기 배수 H는 [10]으로 설정하였다[공개 구현은 12 — 교수님 확인 후 확정].

### 5.4 교수님께 여쭐 것

1. **순수 ETER-net 학습 설정:** 순수 ETER-net(`choh_train_ETER_hybrid_fastmri_brain_acs32R4_231030.py`) 학습에 쓰신 설정 파일 `myConfig_ETER_fastmri_brain_acs32R4_5`나 그 런의 `log.txt`를 받을 수 있을지 여쭙는다. 필요한 값은 학습률, 배치 크기, epoch, optimizer와 weight decay, 손실 종류(CRITERION), λ_SSIM, 스케줄러 사용 여부다.
2. **H:** 보고하신 R4 모델(공개 `tensors_R4.pt`)의 GRU hidden 배수 H가 10인지 12인지 여쭙는다.
3. **학습 데이터:** 재현할 때 AXT2 300볼륨(.mat, 16칸)을 그대로 써야 하는지, 아니면 공식 학습 데이터의 혼합 대조도를 써도 되는지 여쭙는다. .mat를 만든 변환 코드(396 ky → 384)가 있는지도 여쭙는다.
4. **마스크:** 학습에서 마스크 시작 위치를 3으로 고정하신 것(ACS 32, 실효 R 3.2)이 의도된 설정인지 여쭙는다.
5. **평가:** 논문 수치의 평가 데이터(목록 320~378번 59볼륨인지)와 평가 방식(전체 영상 또는 뇌 영역, 슬라이스 평균 또는 볼륨 평균)을 여쭙는다.
6. **[ViT] 스케줄:** 2024 ViT 하이브리드의 `CosineAnnealingWarmRestarts(T_0=40)`가 배치마다 step되는데, 이것이 의도된 설정인지 여쭙는다. [ViT] 레시피를 대신 쓰게 될 때에만 필요하다.

## 6. 대조 중 함께 발견한 것

1. **train–val 중복 (10-12 Pod 직접 확인):** RunPod 데이터에는 train–val 중복이 없다.
   - 그 4볼륨(`file_brain_AXT2_200_2000{306,308,310,311}`)은 공식 train 볼륨이다. 교수님 train 목록 64~67번째 줄에도 있다.
   - 로컬 val 0번에만 섞여 있었고, Pod val 0~2(1,378)에는 없다.
   - 대조 워크플로는 "Pod에도 중복이 남아 있다"고 판정했고, 같은 날 사용자 보고에도 그렇게 전했다. 이것은 틀렸다. 계획 문서 R1 행의 "중복 없음"이 맞다.
2. **교수님 원본 파일의 수정 기록:** 다음 3개 파일이 7d4e4e0 이후 수정됐다.
   - `u_choh_SSIM.py`(afa89a0, mask 인자)
   - `u_choh_model_ETER_ViT.py`(0cdc914 gradient checkpointing·h0, e0dc4c9 RefinementBlock head)
   - `configs/myConfig_choh_model3.py`(0cdc914, ViT-S·H=2로 축소)

   CLAUDE.md의 "무수정" 규칙과 어긋난다. `sanity_pure_v8.py`는 이를 "unmodified OK"로 잘못 보고한다(감사 :48). 수정 여부는 사용자가 결정한다.
3. **정밀도 비대칭:** 모델마다 정밀도가 다르다(T11). 방법 절에 적어야 한다.
4. **GRU 평가의 TF32:** GRU 최종 평가(`eval_v8fix.py` via `eval_v8fix_grufp32.py`)는 cuDNN TF32를 끄지 않는다. 학습 때는 꺼져 있었다. TF32를 끄고 다시 평가할지는 사용자가 결정한다.
5. **로컬 트레이너가 옛판:** 로컬 `v8_eter_pure/main_train_pure_v8fix.py`는 Pod에서 도는 판보다 옛판이다. TF32 끄기, NaN 재시도, val 재시도, env 덮어쓰기가 없다.
6. **README의 낡은 수치:** `infra/runpod/README.md:17-18`의 "4,108 파일 / 65,028 슬라이스"는 낡았다. 실제 Pod은 4,110 / 65,060이다.
7. **IEIE 초안의 GRU 스캔 설명:** 초안은 행 단위("384개 행", "12,288차원")로 설명한다. 실제 코드는 flatten-reshape이고, 한 스텝이 한 채널의 연속 32행이다. 그림 2 대안 B와 관련된다.
8. **"원 손실 균형"이라는 표현:** v8fix 문서와 docstring의 "원 손실 균형"은 v8의 균형(λ 1.0, ortho ×1e6, L1:SSIM ≈69:1)을 말한다. 교수님 균형(λ 0.2)이 아니므로, 원고에서는 혼동되지 않게 쓴다.
