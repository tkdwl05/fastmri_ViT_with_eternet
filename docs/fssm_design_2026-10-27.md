# FSSM — 푸리에 변환을 표현할 수 있는 복소 SSM 시퀀스 모듈 설계 (2026-10-27, 결과 전)

- 상태: **설계·시험 계획 (결과를 보기 전에 고정)**. 결과를 본 뒤 바꾸는 것은 날짜·이유와 함께 '사후 변경'으로 이 문서 끝에 적는다.
- 사용자 결정(2026-10-27): "①을 새 구조 후보로, ②를 평가 조건으로 묶어 다음 논문 주제로" — ① 푸리에 변환을 표현할 수 있는 SSM 으로
  변환 모듈을 다시 설계, ② 평가 조건에 R=8·무작위 마스크 추가. 지금의 통제 비교(MDPI, 11-30)와는 별개의 다음 논문 줄기다.
- 이 문서의 모델 이름은 작업명이다(FSSM = Fourier-capable SSM). 원고 이름은 나중에 정한다.

## 0. 요약 (설명용)

1. 지금 SS2D(Mamba)는 상태 행렬이 실수이고 감쇠만 해서, 푸리에 변환에 필요한 진동하는 핵을 만들 수 없다(10-11 진단 + 아래 §1).
   상태 행렬을 복소수로 두면(S4D 계열) 진동을 직접 표현할 수 있고, 복소 대각 SSM 이 실수 SSM 보다 훨씬 적은 차원·작은 값으로
   같은 사상을 표현한다는 이론 결과도 있다(Ran-Milo et al., 2024).
2. 물리적 근거: R 배 등간격 샘플링의 영상 겹침(앨리어싱)은 k-space 에서 **주기 R 줄의 위상 변화**와 같다(영상에서 N/R 화소 이동
   ↔ k-space 에서 줄마다 2π/R 씩 도는 위상). 복소 SSM 은 이 위상 변화를 그대로 표현할 수 있고, 실수 감쇠 SSM 은 표현하지 못한다.
3. 구조: 판독 방향은 정확한 1D 역푸리에 변환(고정)으로 먼저 풀고(DOTA 의 착안), 줄을 빼는 위상 부호화 방향의 각 줄을 복소 SSM 이
   양방향으로 읽어 k-space 보정을 낸 뒤, 정확한 역푸리에 변환으로 16코일 영상(32채널)을 만들어 원래 U-Net 에 넣는다.
4. **학습 시작 시점에 이 모델은 U-Net only (OU) 와 수치적으로 같다**(보정 출력 0 초기화 + U-Net 첫 합성곱을 U-Net only 의 영상
   채널 부분에서 복사). 그래서 학습 후의 차이는 모두 시퀀스 모듈이 만든 것이다 — "U-Net 보다 나은가"를 바로 잰다.
5. 먼저 로컬 4,000 step 시험으로 U-Net only (OU)·o-SS2D (OU)·o-SS2D-F (OU)·o-ETER (OU) 의 같은 시험 결과와 비교하고, 복소 대 실수
   절제를 함께 본다. 그 뒤 R=8 시험, 50 epoch 는 사용자 결정.

## 1. 근거

### 1.1 지금 SS2D 가 영상 정보를 못 담는 이유 (원래 U-Net 계획 보충 1 §1, 10-11 진단)

- 수용 영역: 출력 화소가 k-space 의 같은 행·열(폭 3 십자)만 본다(가로·세로 스캔을 차례로 합성하지 않고 따로 더함).
- 좌표: 출력이 k-space 좌표에 남는다.
- 세기: 입력 단계 norm_in(위치마다 LayerNorm)이 위치별 k-space 세기를 지운다.
- 스캔 핵: Mamba 의 상태 행렬 A 는 실수·음수 대각(`A = −exp(A_log)`)이라 핵이 감쇠 지수뿐 — 진동 성분이 없다.

FSSM 은 넷을 모두 바꾼다: 위상 부호화 방향 한 축만 스캔(각 줄 전체를 양방향으로 봄), 출력은 정확한 변환으로 영상 좌표,
위치별 정규화 대신 국소 크기(포락선)로 나누고 다시 곱함, A 는 복소수.

### 1.2 앨리어싱과 위상 변화

등간격 R 배 샘플링(줄 간격 R)에서 zero-filled 영상은 원 영상과 그것을 위상 부호화 방향으로 N/R, 2N/R, … 만큼 옮긴 복사본의 합이다.
영상에서 Δ 화소 이동은 k-space 에서 `e^{2πi t Δ / N}` 의 위상 변화(줄 t 마다)이므로, Δ = N/R 이동은 줄마다 `2π/R` 씩 도는 위상
(R=4 → π/2, R=8 → π/4)이다. GRAPPA 처럼 빠진 줄을 이웃 측정 줄과 코일 정보로 메우는 연산은 이 주기 성분을 다뤄야 한다.
복소 대각 SSM 의 한 상태는 `λ = e^{Δ·A}`, A = −a + iω 로 줄마다 `ωΔ` 씩 도는 위상을 그대로 가진다.

### 1.3 문헌 위치 (2026-10-27 검색 — Consensus 4회 + references.bib)

| 갈래 | 대표 | FSSM 과의 관계 |
|---|---|---|
| 변환 자체를 학습 | AUTOMAP(zhu2018automap), DOTA(eo2020dota — 판독 방향 고정 1D IFT 뒤 위상 부호화 방향만 완전 연결층), ETER-net, AFTNet(복소수 망으로 변환 학습, 2023) | DOTA 의 하이브리드 공간 착안을 따름. 변환은 정확한 IFT 로 두고(절제에서 학습형 변환도 봄) SSM 이 하이브리드 공간 보정을 맡음 |
| 교차 영역 | KIKI-net(eo2018kiki) | k-space 망 → IFT → 영상 망 배선이 같은 계열 |
| k-space 보간 | RAKI(2019), k-space deep learning(Han et al., 2020) | 빠진 줄을 메우는 목적이 같음. FSSM 은 데이터 전체로 학습한 복소 SSM |
| k-space 에 Mamba | DH-Mamba(dhmamba2025), PAS-Mamba(kui2027pasmamba) — 원형 스캔, 보통의 실수 Mamba | 실수 동역학. FSSM 은 복소 동역학 + 위상 부호화 방향 스캔 |
| 영상 영역 Mamba | MambaRecon, MambaRoll, MMR-Mamba, HiFi-Mamba, SO-Mamba | 영상 영역 정규화기 — 다른 질문 |
| 앨리어싱 1D 분리 | AliasNet(2023), Yuan et al.(ISMRM 2024) | 위상 부호화 방향 1D 구조를 쓴다는 착안이 같음 |
| SSM 이론 | S4·S4D(gu2022s4), Ran-Milo et al.(2024, 복소 대 실수 대각 SSM 의 표현력 차) | 복소 상태의 근거 |
| 같은 계열의 높은 가속 결과 | oh2025vitbirnn(Journal of Imaging 2025) — 초록: BiRNN 결합 모델이 특히 높은 가속·무작위 샘플링에서 U-Net 대비 우위 | ② 평가 조건의 근거(R4 등간격 차이는 원문 확인 필요) |

- 잠정 판단: "복소(진동) SSM 동역학을 앨리어싱 위상 구조에 근거해 위상 부호화 방향 하이브리드 공간에 쓰고, U-Net only 를 초기값으로
  포함하는 모델"은 이번 검색에서 찾지 못했다. **투고 전 L-04 처럼 다시 확인**(특히 복소 Mamba·S4 를 MRI 에 쓴 사례, AFTNet 초기화).

## 2. 가설 (사전 고정)

- H1: FSSM (OU) 의 SSM 보정 경로는 쓰인다(보정 끄기 시험의 SSIM 하락이 'used').
- H2: FSSM (OU) 는 같은 시험 틀의 U-Net only (OU) 보다 검증 SSIM 이 높다.
- H3: 복소 동역학이 실수 동역학(같은 구조, A 실수)보다 낫다.
- H4(부): 학습형 변환(정확한 IFT 로 초기화한 Vandermonde 변환)이 고정 IFT 보다 더하는 것이 있다.
- 시험(4,000 step)은 판정 근거가 아니라 다음 단계 결정용이다. 원고에는 예비 시험으로만, 사용자가 정하면 쓴다.

## 3. 모델

입력: 정규화된 k-space `x_ksp` (B, 32, H, W) — 16코일 real/imag 교번(로더와 같은 배치), 판독 축 = H(−2, 완전 샘플),
위상 부호화(마스크) 축 = W(−1, 로더의 R4 마스크가 고르는 줄). 정규화 뒤에도 `x_img = ifft2c(x_ksp)`(sanity_ou_ss2d 계약 점검 5).

1. **하이브리드 공간 (고정)**: `z = ifft1c_H(x_ksp)` (복소 (B, 16, H, W); ifftshift → ifft(norm='ortho') → fftshift, 로더 규약).
2. **측정 줄 표시**: `m_w = 1[ max_{c,h} |x_ksp| > 0 ]` (B, W) — forward 인자에 마스크가 없으므로 입력에서 읽는다.
3. **포락선**: `e(b,h,w)` = 코일 RMS `|z|` 를 W 축 폭 15 창에서 **측정 줄만** 평균(R ≤ 8 이면 창 안에 측정 줄이 있음) + 하한
   `1e-6·mean(e)`. `ẑ = z / e`. (SS2D 의 위치별 LayerNorm 대신 — 세기를 지우지 않고 마지막에 다시 곱한다.)
4. **복소 선택적 SSM (양방향, W 축)**: 행 (b, h) 마다 길이 W 의 수열, 위치 특징 = [Re ẑ (16), Im ẑ (16), m_w (1)] = 33.
   - in_proj: Linear(33 → 2D) → (u, gate), D = 64. depthwise Conv1d(k=3, 가운데 정렬) + SiLU (u).
   - 방향마다(정방향·역방향 — 역방향은 수열을 뒤집어 같은 연산) x_proj: Linear(D → dt_rank + 4N) (Δ, B 의 re·im, C 의 re·im),
     dt_proj: Linear(dt_rank → D) (Mamba 기본 Δ 초기화: [1e-3, 1e-1] 로그 균일), N = 16.
   - **A (D, N) 복소**: `A = −exp(A_re_log) + i·A_im`, S4D-Lin 초기화 `A_n = −1/2 + iπn` (n = 0..N−1). 줄당 위상 `πnΔ` 는
     Δ 분포에 걸쳐 [0, ~1.5π] 를 덮어 π/2(R4)·π/4(R8)를 포함한다.
   - 스캔: `mamba_ssm.selective_scan_fn`(설치본 2.2.2 의 CUDA 커널에 `complex<float>` 가중치판이 있음 — 10-27 `nm` 확인; CPU 에서는
     `selective_scan_ref`), u·Δ 실수, B·C 는 입력 의존 복소(실수 텐서의 2L 교번), 출력 실수(켤레쌍 → 실수부 ×2 — mamba_ssm 규약).
   - `y = (y_fwd + flip(y_bwd)) · SiLU(gate)`, **out_proj: Linear(D → 32) weight·bias 0 초기화** → r̂ → `r = e · complex(r̂)`.
5. **보정**: `k_h = z + r` (모든 위치 — 측정 줄 유지 제약은 v0 에 없음, o-SS2D-F 와 같음. DC 판은 사후 선택지).
6. **변환**: `y = ifft1c_W(k_h)` → 16코일 영상 (B, 32, H, W) real/imag 교번 float32.
   - 학습형 변환판(H4): `ifft1c_W` 대신 `T[n,t] = c_n · λ_n^{N−1−t} · b_t` (N = W = 384), `λ_n = exp(−δ_n − iφ_n)`.
     중심 IDFT 와 정확히 같게 하는 초기값: `φ_n = 2πn/N + π`, `δ_n = 0`, `b_t = 1`, `c_n = N^{−1/2}·(−1)^n·e^{iφ_n(N−1)}`
     (N/2 짝수일 때 `fftshift∘ifft∘ifftshift` 행렬 = `N^{−1/2}(−1)^{t+n} e^{2πitn/N}` 에서 유도). 학습 변수 φ 는 `φ0 + η/N`,
     δ 는 `relu(ρ)/N` 로 두어 기울기 크기를 맞춘다. 복소 행렬 곱(fp32)이라 계산이 가볍다(학습형 Vandermonde 변환 — 파라미터 3N 복소;
     DOTA 의 완전 연결층은 N²).
7. **정제망**: OriginalUNet(in = 32, n_classes = 1, depth 5, wf 6). **SAME_UNET_INIT**: 모양이 같은 텐서는 U-Net only (OU) 에서
   복사, 첫 합성곱 weight (64, 32, 3, 3) 는 기준 (64, 52, 3, 3) 의 `[:, 20:52]`(zero-filled 영상 자리)를 복사 — 기준의 앞 20 채널은
   항상 0 을 받으므로 **초기 모델 = U-Net only (OU) 초기 모델**(수치 오차 수준). 데이터 순서도 같은 시드라 같다.
8. **정밀도**: FFT·포락선·변환은 autocast 끄고 fp32(384 는 2 의 거듭제곱이 아님). SSM 본체는 autocast(SS2D(controlled)와 같음).
9. **건강 감시**: seq 그룹의 |w|<1e-6 비율이 0.5 이상이면 트레이너가 exit 3 — 0 초기화 텐서(out_proj, δ·η, b 의 허수부)는
   seq 파라미터의 약 10 % 라 통과해야 한다(10-27 로컬 비교 시험의 o-LIN-F (OU) 는 선택 행렬 초기값 때문에 96.9 % 로 28 초 만에 정지).

| 이름 (SEQ_MODEL) | 클래스 | 차이 | 용도 |
|---|---|---|---|
| FSSM (OU) (`fssm_ou`) | PureETER_FSSMOU | 위 1–8, 복소 A, 고정 IFT | 주 모델 |
| FSSM-real (OU) (`fssmr_ou`) | PureETER_FSSMROU | A 실수(Mamba 기본 `A_n = −(n+1)`, 허수부 없음), B·C 실수 | H3 절제 |
| FSSM-T (OU) (`fssmt_ou`) | PureETER_FSSMTOU | 복소 A + 학습형 변환(6) | H4 절제 |

- 등록부: `recipe_ou_fssm.py` 가 recipe_ou(·recipe_ou_ss2d)의 MODELS 에 세 이름을 더한다(바탕 이름 'ss2d', seq_only_input=True —
  시퀀스 모듈을 `model.ss2d` 에 두어 건강 감시 그룹·기능 시험·평가 래퍼가 그대로 맞음, `seq_out_ch = 32`).
  `copy_unet_init` 은 위 7 의 조각 복사로 감싼다. 기존 파일(recipe_ou·recipe_ou_ss2d·트레이너·모델)은 고치지 않는다 — 실행 중인 OU 런.
- 트레이너: `main_train_pure_v8fix_ou_ss2d.py` 복사본 `main_train_pure_v8fix_ou_fssm.py` — 다른 점은 import(recipe_ou_fssm)·받는 이름·
  SEQ_MODEL 기본값뿐.
- 기능 시험 주의: 시퀀스 출력에 정확한 IFT 경로(zero-filled 영상)가 들어 있어 shuffle·zero 시험은 '입력 영상 교체'가 되어 크게
  떨어진다(의미 없음). 시퀀스 모듈 기여는 **보정 끄기 시험**(out_proj = 0 → 출력 = x_img)으로 잰다.

## 4. 점검 (구현 뒤, 시험 전)

CPU(`CUDA_VISIBLE_DEVICES=""`, SSM 은 selective_scan_ref):
1. `ifft1c_W ∘ ifft1c_H = ifft2c_ri` (상대 오차 < 1e-6), 학습형 변환 초기값 = 중심 IDFT 행렬(최대 오차 < 1e-5).
2. 초기 출력 = `x_img`(실제 val 슬라이스로 — 상대 오차 < 1e-5), 초기 모델 출력 = 같은 시드 U-Net only (OU) 출력(SAME_UNET_INIT 뒤,
   상대 오차 < 1e-4).
3. 복소 selective_scan_ref 결과 = 같은 식의 순차 계산(작은 크기) — B·C 교번 규약 확인.
4. 건강 감시 near_dead 비율(seq < 0.5, 초기값 기준) · 파라미터 수 · 그룹(seq 에 ss2d.* 전부).
5. out_proj 를 작은 값으로 바꾼 뒤 한 걸음 역전파 — seq 의 모든 텐서에 기울기가 0 이 아님(실수판·학습형 변환판 포함).
GPU(로컬 비교 시험이 끝난 뒤): CUDA 커널(복소 A) = ref(작은 크기, fwd·bwd), BS 8 학습 30 step 스모크(VRAM·s/batch, NaN 없음).

## 5. 시험 (로컬 GPU0)

- 틀: 원래 U-Net 계획 보충 1 §4 와 같음 — 시드 1, 레시피 DL, 50 epoch 일정의 epoch 1 앞 4,000 step(DEBUG_MAX_STEPS=4000,
  STOP_AFTER_EPOCH=1, SANITY_NUM_EPOCHS=50), BS 8, 로컬 val 전체, PROBE 64 슬라이스, PREFETCH 2·NUM_WORKERS_VAL 12,
  V8FIX_LOG_ROOT 스크래치, wandb 끔.
- 시작: 다른 세션의 비교 시험 사슬(status.txt 의 PILOT_DONE)이 끝나고 GPU 가 빈 뒤(≈10-27 06 UTC 예상).
- 순서: fssm_ou → fssmr_ou → fssmt_ou, 각 런 뒤 보정 끄기 시험(PROBE 64 슬라이스, out_proj = 0).
- 비교 기준(같은 틀, 이미 있음): U-Net only (OU) val SSIM 0.8184 / PSNR 29.78(4,000 step), o-SS2D (OU) 0.2850, o-SS2D-F (OU)·
  o-ETER (OU)(10-27 진행 중).
- 예상 시간: 런마다 약 1.5 h(학습 4,000 step + 검증) → 세 런 ≈ 5 h.

### 읽는 규칙 (사전 고정 — 판정이 아니라 다음 단계 결정용, 결정은 사용자)

- '보정 경로 사용' = 보정 끄기 하락이 평균 > 0, 평균 > 2SE, 부호 검정 p < 0.01 (기존 기능 시험 기준과 같음).
- 다음 단계(R=8 시험, 50 epoch) 권고 조건: fssm_ou 가 (a) 보정 경로 사용이고 (b) 검증 SSIM 이 U-Net only (OU) 시험값보다 높음.
  (a) 만 맞으면 R=8 시험을 먼저 권고(가속이 높을수록 시퀀스 모듈 몫이 커진다는 가설 ②). (a) 가 아니면 설계를 다시 본다(사후 변경 기록).
- H3·H4 는 fssmr_ou·fssmt_ou 의 검증 SSIM·보정 끄기 하락을 fssm_ou 와 나란히 적는다(기술).

## 6. ② 평가 조건 (R=8·무작위)

- R=8 등간격: 로더 `dataloader_h5_v5.build_r4_mask(width, center_fraction, acceleration)` 가 acceleration 인자를 받는다 — 트레이너·config
  에서 가속 계수·중앙 비율(fastMRI 관례 8× → 0.04)을 바꾸는 경로를 구현 때 확인. 포락선 창(15)은 R ≤ 8 을 덮는다.
- 무작위(변밀도) 줄 샘플링: v5 로더의 `random_mask` 는 등간격 시작 위치만 무작위 — 무작위 마스크는 새 로더 파일이 필요(나중).
- R=8 시험은 fssm_ou 와 U-Net only (OU) 두 런(같은 4,000 step 틀).

## 7. 위험

- R=4 등간격은 U-Net 이 이미 잘하는 조건이라, 보정 경로가 쓰여도 U-Net only 와 차이가 작을 수 있다(그래서 ②).
- 복소 가중치 CUDA 커널은 이 프로젝트에서 처음 쓴다 — fp16 입력에서의 수치 안정성, 속도는 스모크로 먼저 확인.
- 측정 줄 유지(DC) 제약이 없어 측정 줄도 바뀐다 — 필요하면 사후 변경으로 DC 판을 더한다.

## 8. 사후 변경 기록

- 2026-10-27 구현 중 (결과 전, 학습 전):
  - §3.6 학습형 변환의 감쇠를 `δ = relu(ρ)/N`(ρ = 0)에서 `δ = softplus(ρ)/N`(ρ 초기 −14)로 바꿨다 — relu 는 ρ = 0 에서 기울기가
    0 이라 감쇠를 배울 수 없다. 초기 변환과 중심 IDFT 의 오차는 ≈8e-7(점검 1b 5.5e-8).
  - FSSM-real (OU) 의 상태 수를 32(= 복소 16 상태와 같은 실수 크기)로 두어 x_proj·A 의 파라미터 수를 복소판과 같게 했다.
  - 구현·점검 결과(CPU 17개 통과): 초기 모델 출력 = U-Net only (OU) (상대 오차 4.5e-7, U-Net 텐서 46/46 복사 — 첫 합성곱은
    기준 [:, 20:52]), seq 그룹 near_dead 0.110(fssm·fssmr)·0.133(fssmt), seq 파라미터 20,128 / 20,128 / 22,432,
    전체 31,068,577 / 31,068,577 / 31,070,881. 파일: `models/pure_eternet/u_pure_eternet_fssm.py`,
    `v8_eter_pure/{recipe_ou_fssm,main_train_pure_v8fix_ou_fssm,sanity_fssm}.py`.
