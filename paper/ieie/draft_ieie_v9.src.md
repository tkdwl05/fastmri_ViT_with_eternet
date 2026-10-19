%% IEIE 2026 추계학술대회 투고용 초안 v9 (2026-10-17) — ARS academic-pipeline Stage 2(WRITE) 산출물.
%% 1차 수정(2026-10-17): Stage 2 점검 5종(30건) 반영 — 처리 기록 = paper/ieie/reviews/ars_pipeline_20261017/stage2/check_responses_v9.md (git 무시).
%% 2차 수정(2026-10-17): 내부 모의 심사 A·B(Major 5건·Minor 23건) 처리 — 처리 기록 = paper/ieie/reviews/ars_pipeline_20261017/stage2/internal_review_responses_v9.md (git 무시).
%% 출처: draft_ieie_v8.src.md 를 복사한 뒤 승인된 개요(paper/ieie/reviews/ars_pipeline_20261017/stage2/outline_v9.md C판, §8 critique 응답 포함, 2026-10-17 승인)와
%%   논증 설계(같은 폴더 argument_blueprint_v9.md)를 절별로 적용(KEEP / KEEP+EDIT / REPLACE / DELETE). 검토 폴더(paper/ieie/reviews/)는 git 무시 — 그 안의 개인 정보는 이 소스에 옮기지 않는다.
%% 결과 출처: v8fix 세 모델(U-Net only, ETER-net, SS2D(controlled); 난수 시드 1, 50 epochs, 모델마다 한 번 학습) — 공식 검증 분할(1,378 볼륨)로 체크포인트 선택, test_full(558 볼륨)에서 평가.
%%   등록 실험 ID(material_passport.yaml): v8fix-unet-s1, v8fix-eter-s1, v8fix-ss2d-s1, v8fix-paired-testfull, v8fix-ss2d-functional, v8fix-fig-inference.
%%   붕괴 런(이전 v8/v9) 수치는 쓰지 않는다. 결과 수치는 paper/tables/ieie_v9_tokens.md(paper/make_tables_v8fix.py 생성)의 토큰에서만 가져오며, 수치가 든 문단 뒤 "%% tokens:" 줄에 사용 토큰을 적는다.
%%   표 1 = paper/tables/ieie_table1_block_v9.md 를 그대로 붙여 넣은 블록(손편집 금지 — 수치가 바뀌면 생성기를 다시 실행하고 블록을 다시 붙여 넣는다).
%% 결정(passport D1–D9): D1 수정판(2026-10-17 — SS2D(controlled)−ETER-net 차이를 상태공간 모델링의 효과로 해석하지 않음, 근거 = 2차 비교 SS2D(controlled)−U-Net only;
%%   SS2D 모듈 출력 크기·교체 시험은 중립 서술), D2(Zero-filled 행 없음), D3(TF32 — 원고 문구 없음), D4(그림 3 슬라이스·세기 보정 없음), D5(brain mask SSIM 주 지표 + 공식 정의 값 본문 보고),
%%   D7(원 구현과 다른 설정·정밀도·SS2D 설계 한계·학습 손실 격차·단일 학습), D8 기본안(제목 'ETER-Net' 교수님 문구 유지·본문 'ETER-net', 그림 2 A판, "(controlled)" 유지,
%%   SS2D(enhanced)·공개 모델 행 삭제, H=10 명시), D9(학술대회판만 투고), fp32 = 사실만(이유 구절 없음).
%% 10-18 수정(사용자 승인, run ledger seq 36–45): s 정의식(식 (3))과 손실 식 (4)의 x*/s, 학술대회판 Word 5쪽 감축(캡션·표 1 Params 열 삭제·2.3·3.1·3.2·결론·영문 초록 축약),
%%   2.2절 원 SS2D 서술 정정(P4, 참고문헌 사전 확인 C-054), 참고문헌 저자 3인 이상 et al.(빌더), oh2025vitbirnn 연도 2026(bib). 근거·미리보기 = 검토 폴더 stage2/ (git 무시).
%% 10-19 수정(사용자 결정, run ledger seq 60–65): 2.5단계 무결성 경미 수정 IL-MINOR-1~4(절제 실험 구절 삭제, RMS 중립 서술, x̂ = 원래 세기의 복원 영상·식 (4)는 x̂/s, 공식 평가 정의 [10]); IL-MINOR-5([7] 권·쪽)는 보류.
%%   서면 심사용 4쪽판은 별도 소스 draft_ieie_v9_review.src.md(내용이 이 원고의 축약판)로 빌드한다.
%% 빌드(저장소 루트, CPU; --out 은 소스 폴더 기준 stem 이므로 생략하면 draft_ieie_v9 / draft_ieie_v9_conf):
%%   CUDA_VISIBLE_DEVICES="" python paper/ieie/build_ieie_conf_docx.py --src paper/ieie/draft_ieie_v9.src.md   → draft_ieie_v9_conf.{md,docx}  (학술대회 논문집 게재용 — 영문 초록만 인쇄, 분량 추정 출력)
%%   CUDA_VISIBLE_DEVICES="" python paper/ieie/build_ieie_docx.py      --src paper/ieie/draft_ieie_v9.src.md   → draft_ieie_v9.{md,docx}       (서면 심사용, blind 점검 내장 — 지금은 투고하지 않으나 같은 소스로 빌드되어야 함)
%%   python paper/ieie/check_docx_structure.py paper/ieie/draft_ieie_v9_conf.docx ; python paper/ieie/check_docx_structure.py paper/ieie/draft_ieie_v9.docx
%% 규칙: 표준 지표만(SSIM 주 평가 지표 + PSNR/nMSE %; composite 금지) · 기준 모델 = ETER-net · 모델명 U-Net only / ETER-net / SS2D(controlled), 모듈명 bi-GRU 모듈 / SS2D 모듈 · 모든 차이는 한 번의 학습에서 관측한 값.
%% 표기: 인라인 수식은 달러 기호 한 쌍(\mathbb·\leq·\geq·\dots·\sim·\varphi 사용 불가), 표시식은 $$ 로 시작하는 한 줄 · 그림·표는 col 폭, 캡션은 @cap_en 만 · 표는 @note 한 줄 뒤 @end ·
%%   인용 [@bibkey](초록에는 넣지 않음) · 본문에 blind 점검어를 쓰지 않음 · 저자·소속·e-mail 은 학술대회판에만 들어가며 "***" 자리표시자는 교수님 확인 후 기입.
@author_ko: ***, ***
@affil_ko: *** 소속
@email: e-mail : ***
@author_en: *** and ***
@affil_en: *** University

@title_ko: 가속 MRI 재구성 네트워크 ETER-Net에서 순환신경망을 선택적 상태공간모델로 치환한 통제 비교 연구
@title_en: A Controlled Study of Replacing the RNN with a Selective SSM in ETER-Net for Accelerated MRI Reconstruction
@keywords: MRI reconstruction, accelerated MRI, ETER-net, recurrent neural network, selective state-space model, SS2D, controlled comparison

@abstract_ko:
ETER-net은 언더샘플링된 k-space를 양방향 순환신경망을 활용하여 영상 도메인으로 직접 변환하는 MRI 재구성 방법이다. 본 연구는 ETER-net의 도메인 변환 모듈(이하 시퀀스 모듈)인 양방향 GRU(gated recurrent unit, bi-GRU) 모듈을 2차원 선택적 상태공간모델(SS2D)로 교체한 모델(이하 SS2D(controlled))과 ETER-net의 재구성 성능을 비교한다. 본 연구의 구성에서는 U-Net이 다중 코일 zero-filled 영상도 함께 입력받는다. 또한 본 연구는 시퀀스 모듈의 출력을 0으로 대체한 U-Net only 모델을 대조 모델로 함께 학습하였다. fastMRI brain multicoil 데이터셋(384×384)의 k-space를 가속화 계수 4(R=4)로 언더샘플링한 조건에서, 모든 모델은 시퀀스 모듈과, ETER-net에만 적용한 GRU 한 개의 32비트 계산 및 기울기 보호 장치를 제외하면 데이터, 언더샘플링 마스크, U-Net 구조와 초깃값, 손실 함수, 최적화 설정이 같으며, 명시적 데이터 일관성 블록을 포함하지 않는다. SS2D(controlled)와 U-Net only 모델의 파라미터 수는 ETER-net의 약 1/21이다. 각 모델을 같은 난수 시드로 50 epochs 동안 한 번씩 학습하여 test_full 분할(558개 볼륨)에서 평가한 결과, brain mask 내부에서 계산한 볼륨 평균 structural similarity index(SSIM) 값은 기준 모델인 ETER-net보다 U-Net only가 0.0022, SS2D(controlled)가 0.0008 높았다(볼륨 단위 대응 Wilcoxon signed-rank 검정, $p<0.001$). SS2D(controlled)의 SSIM은 U-Net only보다 0.0014 낮았으므로, 본 연구는 SS2D(controlled)와 ETER-net의 차이를 상태공간 모델링의 효과로 해석하지 않는다. 학습 손실에 포함되지 않은 배경까지 평가하는 fastMRI 공식 평가 정의에서는 SS2D(controlled)의 SSIM이 ETER-net보다 낮았다(0.8389 대 0.8652). 각 모델을 한 번씩만 학습하였으므로, 이 차이들로 시퀀스 모듈이 재구성에 도움이 되는지 또는 해가 되는지는 판단할 수 없다.
%% tokens: PARAMS_RATIO_SS2D, PARAMS_RATIO_UNET (1/21; derivation in the token descriptions), N_TEST_VOL, D_UNET_ETER_SSIM_ABS, D_SS2D_ETER_SSIM_ABS, P_UNET_ETER_SSIM, P_SS2D_ETER_SSIM, D_SS2D_UNET_SSIM_ABS, OFFSSIM_SS2D, OFFSSIM_ETER (+ CFG: 50 epochs, seed 1, R=4)

@abstract_en:
ETER-net is an MRI reconstruction method that transforms undersampled k-space directly into the image domain with a bidirectional recurrent neural network. This study replaces its bidirectional gated recurrent unit (GRU) sequence module with a two-dimensional selective state-space model (SS2D), giving SS2D(controlled), and trains a U-Net only model whose sequence-module output is zeroed. All U-Nets also receive the zero-filled coil images, no model has a data-consistency block, and the models differ only in the sequence module (and, in ETER-net, 32-bit computation of one GRU and non-finite-gradient guards); SS2D(controlled) and the U-Net only model have about 1/21 of ETER-net's parameters. Each model was trained once (50 epochs, same seed) on the fastMRI brain multicoil dataset (384×384, R=4) and evaluated on the test_full split (558 volumes). Relative to the reference ETER-net, the volume-averaged structural similarity index (SSIM) within a brain mask was 0.0022 higher for the U-Net only model and 0.0008 higher for SS2D(controlled) (Wilcoxon signed-rank test, $p<0.001$). Because SS2D(controlled) scored 0.0014 below the U-Net only model, the SS2D(controlled)–ETER-net difference is not attributed to state-space modeling. Under the official fastMRI SSIM definition, which includes the background excluded from the loss, SS2D(controlled) scored below ETER-net (0.8389 vs. 0.8652). As each model was trained once, these differences do not show whether the sequence modules help or hurt reconstruction.
%% tokens: PARAMS_RATIO_SS2D, PARAMS_RATIO_UNET (1/21; derivation in the token descriptions), N_TEST_VOL, D_UNET_ETER_SSIM_ABS, D_SS2D_ETER_SSIM_ABS, P_UNET_ETER_SSIM, P_SS2D_ETER_SSIM, D_SS2D_UNET_SSIM_ABS, OFFSSIM_SS2D, OFFSSIM_ETER (+ CFG: 50 epochs, seed 1, R=4)

@body:

# 서론

자기공명영상(MRI)은 k-space를 순차적으로 수집하므로 촬영 시간이 길다. 촬영 시간을 단축하기 위해 언더샘플링된 k-space로부터 영상을 복원하는 딥러닝 기법이 연구되어 왔다. 대표적인 접근으로는 물리 모델 기반 반복 최적화 과정을 신경망 층으로 펼치는 방식(unrolling)[@sriram2020endtoend]과 k-space를 영상으로 직접 변환하는 도메인 변환 방식[@zhu2018automap]이 있다. ETER-net[@oh2021eternet]은 후자에 속한다. 양방향 GRU(gated recurrent unit, bi-GRU)가 k-space를 행·열 방향으로 읽어 영상 도메인 특징으로 변환하고 U-Net이 에일리어싱(aliasing) 아티팩트를 줄이며, ETER-net은 명시적 데이터 일관성 블록을 사용하지 않는다[@oh2021eternet; @oh2022radial]. 이후 non-Cartesian 궤적[@oh2022radial]과 Vision Transformer 인코더 결합[@oh2025vitbirnn]으로 확장되었으나 도메인 변환의 핵심인 양방향 순환신경망(bi-RNN)은 유지되었다. 그러나 ETER-net은 행·열 방향을 순차 처리하는 구조여서 병렬화가 어렵고, flattened 2D 고차원 입력을 GRU로 처리하므로 대규모 가중치 행렬이 필요하며, 전체 파라미터 수가 수억 개에 이른다. 선택적 상태공간모델(SSM)인 Mamba[@gu2023mamba]는 입력 의존 순환을 병렬 스캔으로 선형 시간에 계산하고, 이를 2차원으로 확장한 SS2D[@liu2024vmamba]는 선택적 스캔을 여러 방향으로 적용하여 2차원 공간 정보를 처리한다. 기존의 Mamba 기반 MRI 재구성 연구[@korkmaz2025mambarecon; @huang2024mambamir]는 새로운 구조 전체를 제안하였고, 기존 골격에서 순환신경망만 SSM으로 바꾼 효과를 분리하여 보고하지 않았다. 본 논문은 ETER-net에서 도메인 변환을 담당하는 시퀀스 모듈만 bi-GRU에서 SS2D로 치환하고, 나머지 구조는 모두 고정한 상태에서 통제 비교 분석을 수행한다. 또한 시퀀스 모듈의 기여를 가늠하기 위해 그 출력을 0으로 대체한 U-Net only 모델을 함께 학습하여 비교한다.

# 방법

## 비교 실험의 공통 재구성 파이프라인

그림 1의 파이프라인에서는 완전 샘플링된(fully-sampled) k-space $y_c$에 가속화 계수가 4인(R=4) 등간격(equispaced) 마스크 $M$을 적용하여, 언더샘플링된 k-space $\tilde{y}_c$($=M \odot y_c$, $\odot$는 원소별 곱)를 생성한다. 이를 실수·허수로 분리한 32채널 텐서를 모델의 입력으로 사용한다. 마스크 $M$은 k-space 중앙 8%의 auto-calibration signal(ACS) 라인을 모두 포함한다. 한편, 참조 영상(Ground Truth) $x^{*}$는 데이터셋이 제공하는 전체 코일 RSS(Root-Sum-of-Squares) 영상을 384×384 크기로 크롭 및 패딩하여 생성한다. 순차적 입력을 처리하는 모듈 $f_\theta$(이하 시퀀스 모듈)는 입력 k-space를 영상 도메인 특징(20채널)으로 변환하도록 구성된다. 동일한 $\tilde{y}_c$에 코일별 역 푸리에 변환을 적용한 zero-filled 영상(32채널)과 시퀀스 모듈의 출력을 채널 축으로 결합한다. 후처리를 담당하는 U-Net $g_\phi$가 결합 텐서를 입력으로 하여 최종 결과인 크기(magnitude) 영상 $\hat{x}$을 출력한다(식 (1)). 실험에 사용한 U-Net은 dual-frame skip을 포함하며, 깊이가 5, 첫 단계의 채널 수가 64이고, 파라미터의 수는 31.1M이다.
%% tokens: PARAMS_UNETBODY (+ architecture constants: ACS 8%, 32/20 channels, depth 5, 64)

$$ \hat{x} = g_{\phi}\left( \mathrm{concat}\left( f_{\theta}(\{\tilde{y}_c\}), \mathcal{F}^{-1}\{\tilde{y}_c\} \right) \right) $$

식 (1)에서 $\{\tilde{y}_c\}$는 모든 코일($c$는 코일 인덱스)의 언더샘플링된 k-space, $\mathcal{F}^{-1}$는 코일별 2차원 역 푸리에 변환, $\mathrm{concat}$은 두 텐서를 채널 축으로 결합하는 연산이며, $\theta$와 $\phi$는 각각 $f_\theta$와 $g_\phi$의 학습 파라미터이다. 이하 비교하는 세 모델의 구조는 시퀀스 모듈의 구성(bi-GRU, SS2D, 없음)에서만 다르며, U-Net only 모델은 시퀀스 모듈의 출력 20채널을 0으로 대체한 모델(전체 파라미터 수 31.1M)이다. bi-RNN의 출력만 U-Net에 입력하는 [@oh2022radial]의 구조와 달리, 본 연구의 ETER-net은 공개된 구현 코드와 같이 U-Net이 시퀀스 모듈의 출력과 함께 32채널 zero-filled 영상을 입력받는다. zero-filled 영상은 언더샘플링된 k-space와 같은 정보를 담고 있으므로, U-Net only 모델도 다른 모델과 같은 입력 정보를 받는다.
%% tokens: PARAMS_UNET

@figure: paper/figs/conf_fig1_pipeline_v9.png | col | 1.0
@cap_en: Common ETER-net pipeline; only the sequence module $f_\theta$ (dashed box) differs (output zeroed in the U-Net only model)

## 시퀀스 모듈의 두 구성

ETER-net의 bi-GRU 모듈은 그림 2(a)에 나타나듯이, 입력 k-space를 384개 행의 시퀀스(스텝당 12,288차원)로 펼쳐 양방향 GRU를 통과시킨 뒤, 전치하여 열 방향으로 한 번 더 통과시키는 구조이다. 다만 공개 구현과 같이 32×384×384 입력을 순서를 바꾸지 않고 384×12,288로 재배열하므로, 한 스텝은 한 채널의 연속된 32개 행에 해당한다. bi-GRU 모듈의 파라미터의 수는 637.1M, 모델의 전체 파라미터의 수는 668.2M이다. 그림 2(b)는 SS2D 모듈을 나타낸다. 이 모듈은 각 k-space 위치의 32채널 벡터를 층 정규화(Layer Normalization, LN)와 선형 층을 거쳐 128채널로 바꾸고 3×3 depthwise convolution으로 이웃 위치의 정보를 혼합한 뒤, 각 행에 대해서는 좌우 두 방향, 각 열에 대해서는 상하 두 방향으로 S6를 적용하여 선택적 스캔을 수행한다. SS2D 모듈은 원 SS2D[@liu2024vmamba]와 같이 방향마다 독립된 가중치를 적용하지만, 특징 지도 전체를 네 가지 순회 경로마다 한 시퀀스로 펼치는 원 SS2D와 달리 384 길이의 각 행과 열을 독립 시퀀스로 처리한다. 마지막으로 네 방향의 출력을 채널 축으로 결합한 뒤 bi-GRU 모듈의 출력과 같은 20채널로 변환한다. S6는 Mamba[@gu2023mamba]에서 정의한 구조(S4 + selection + scan)로, 구조화 상태 공간 시퀀스 모델(S4)에 입력에 따라 계수가 달라지는 선택(selection) 메커니즘을 더한 것이며 스캔(scan) 알고리즘으로 계산된다. S6는 한 행 또는 한 열의 시퀀스를 식 (2)의 이산화된 상태 공간 시스템으로 처리한다.
%% tokens: PARAMS_GRUMOD, PARAMS_ETER

$$ h_t = \exp(\Delta_t A)\, h_{t-1} + \Delta_t B_t x_t, \quad y_t = C_t h_t + D x_t $$

식 (2)에서 아래첨자 $t$는 스캔 방향을 따라 부여한 시퀀스 내 위치($t = 1, \ldots, L$, 시퀀스 길이 $L=384$)를 의미한다. $x_t$와 $y_t$는 위치 $t$의 입력과 출력이고, $h_t$와 $h_{t-1}$은 각각 위치 $t$와 $t-1$까지의 입력을 요약한 $N$차원($N=16$) 은닉 상태 벡터이다. $A$는 학습 가능한 $N \times N$ 대각 상태 행렬이고, $\exp(\Delta_t A)$는 $A$를 간격 $\Delta_t$로 이산화한 상태 전이 행렬이다. $B_t$와 $C_t$는 각각 $N \times 1$ 입력 행렬과 $1 \times N$ 출력 행렬로, 입력을 은닉 상태에 반영하는 계수와 은닉 상태로부터 출력을 계산하는 계수이다. $D$는 입력 $x_t$에 곱해져 출력에 직접 더해지는 skip 계수이다. 식 (2)는 128개 내부 채널 각각에 독립적으로 적용되므로, $x_t$와 $y_t$는 한 채널의 스칼라 값이고 $A$와 $D$는 채널마다 따로 학습된다. softplus 함수를 거친 채널별 양수 $\Delta_t$와 모든 채널에 공통인 $B_t$, $C_t$는 위치 $t$의 128채널 입력 벡터를 선형 층에 통과시켜 얻은 값이다.

bi-GRU 모듈도 모든 스텝에 같은 가중치를 사용하지만, 한 스텝의 입력이 12,288(=32×384)차원 벡터이고 은닉 상태가 3,840(=10×384)차원이어서 가중치 행렬의 크기가 영상 한 변의 크기(384)의 제곱에 비례한다. 반면 SS2D 모듈은 한 k-space 위치를 한 스텝으로 처리하여 한 스텝의 입력이 128채널 벡터이고 은닉 상태가 채널마다 $N$(=16)차원이므로, 가중치 행렬의 크기가 영상 크기와 무관하고 파라미터 수가 0.12M이다. 또한 SS2D 모듈의 각 출력은 같은 행과 열 주변(폭 3의 십자 영역)의 입력에만 의존하므로, 이 모듈만으로는 2차원 역 푸리에 변환을 표현할 수 없다. SS2D(controlled)는 SS2D 모듈 하나만을 시퀀스 모듈로 구성한 모델(전체 파라미터 수 31.2M)이다. 본 연구에서는 모델별 하이퍼파라미터 조정을 수행하지 않았다.
%% tokens: PARAMS_SS2DMOD, PARAMS_SS2D (+ architecture constants 12,288, 3,840, 384, N=16)

@figure: paper/figs/conf_fig2_modules_v9.png | col | 1.0
@cap_en: Sequence modules $f_\theta$: (a) bi-GRU module of ETER-net, (b) SS2D module

## 학습·평가 프로토콜

학습에는 fastMRI brain multicoil 데이터셋[@zbontar2018fastmri]의 공식 학습 분할에서 확보한 4,110개 볼륨을 사용하였다. 공식 검증 분할 전체(1,378개 볼륨)는 학습 중 검증과 최적 체크포인트 선택에 사용하였다. 최종 평가는 참조 영상이 공개된 공식 시험 분할(test_full, 558개 볼륨, 슬라이스 8,852개)에서 수행하였으며, 세 분할 사이에 겹치는 볼륨은 없다. 데이터셋에는 축상면(axial) T1, 조영제 투여 전후 T1, T2, 액체감쇠역전회복(FLAIR) 영상이 혼합되어 있다(평가 볼륨의 약 58%가 축상면 T2 영상(AXT2)). $y_c$는 완전 샘플링된 k-space를 코일별 영상으로 변환하여 384×384 크기로 크롭 및 패딩한 뒤 다시 k-space로 변환한 것이다. 코일은 처음 16개만 사용하고 부족한 채널은 0으로 채워 16개로 통일하였다. 모델에는 식 (1)의 $\tilde{y}_c$ 대신 식 (3)의 슬라이스별 정규화 계수 $s$로 나눈 $\tilde{y}_c/s$를 입력하고, 출력에 $s$를 곱한 값을 복원 영상 $\hat{x}$으로 하여 원래 세기에서 평가하였다. 등간격 샘플링 라인의 시작 위치(mask offset)는 학습 시에만 샘플마다 무작위로 정하였다. 학습 시에는 영상을 상하·좌우로 무작위 반전하여 데이터를 증강하였다. 배경 영역의 영향을 줄이기 위해 슬라이스마다 참조 영상 $x^{*}$의 0이 아닌 화소에서 Otsu 임곗값 $T$[@otsu1979threshold]를 구하고, $x^{*} > 0.4T$인 화소들의 연결 성분 중 가장 큰 것을 brain mask $m$으로 정의하였다. $T$는 화소를 두 집단으로 나누었을 때 집단 간 분산을 최대로 하는 임곗값이다. 학습 시 역전파에 사용한 손실 함수(식 (4))와 세 평가 지표인 structural similarity index(SSIM), peak signal-to-noise ratio(PSNR), normalized mean squared error(nMSE)는 모두 이 마스크 내부에서만 계산하였다. 평가 SSIM은 scikit-image로 구한 SSIM 지도의 마스크 내부 평균이고, 동적 범위(data range)는 마스크 내부 $x^{*}$의 최댓값과 최솟값의 차이이며, PSNR의 최대 신호 값(peak)과 nMSE의 분모도 마스크 내부의 $x^{*}$로 구하였다.
%% tokens: N_TRAIN_VOL, N_VAL_VOL, N_TEST_VOL, N_TEST_SLICE, TEST_AXT2_PCT0 (+ CFG: 16 coils, 0.4T)

$$ s = \mathrm{P}_{99}\left( \sqrt{\sum_{c} \left| \mathcal{F}^{-1}\{\tilde{y}_c\} \right|^{2}} \right) $$

$$ \mathcal{L} = \frac{\sum_{i} m_i \left| \hat{x}_i/s - x^{*}_i/s \right|}{\sum_{i} m_i} + \lambda \left( 1 - \mathrm{SSIM}_m(\hat{x}/s, x^{*}/s) \right) $$

식 (3)의 $s$는 zero-filled RSS 영상의 384×384 화소 값에 대한 99 백분위수($\mathrm{P}_{99}$)이다. 식 (4)에서 $\mathcal{L}$은 손실 함수, $i$는 화소 인덱스, $\sum_{i}$는 모든 화소에 대한 합이다. $m_i$는 화소 $i$가 brain mask $m$ 내부이면 1, 외부이면 0이고, $x^{*}_i$와 $\hat{x}_i$는 각각 화소 $i$에서의 참조 영상 $x^{*}$와 복원 영상 $\hat{x}$의 값이다. $\mathrm{SSIM}_m(\hat{x}/s, x^{*}/s)$는 $\hat{x}/s$와 $x^{*}/s$의 SSIM 지도를 마스크 $m$ 내부에서 평균한 값이다. SSIM 항의 가중치 $\lambda$는 원래 세기에서의 가중치 $10^{-6}$을 학습 슬라이스 64개의 $s$ 평균 $\bar{s}$로 나눈 값(약 $3.1\times10^{-3}$)이며, 학습 종료 시 첫째 항은 둘째 항의 약 66배였다. 최적화에는 분리형(decoupled) weight decay를 적용한 AdamW를 사용하였으나, 설정값 $3\times10^{-5}$의 감쇠는 단정밀도 가중치를 바꾸지 못할 만큼 작아 실효적으로 적용되지 않았다. 학습률은 코사인 어닐링 방식으로 $2\times10^{-4}$에서 $1\times10^{-6}$까지 감소하였고, 배치 크기는 8이다. 계산은 16비트 부동소수점(fp16) 자동 혼합 정밀도로 수행하고 기울기가 유한하지 않은 step은 건너뛰었으나, ETER-net의 가로 방향(1단계) GRU는 32비트 부동소수점(fp32)으로 계산하였고, ETER-net에는 그러한 step에 대한 보호 장치(재계산과 손실 배율 하한)를 더하였다. 세 모델은 모두 난수 시드를 1로 고정하여 같은 순서의 학습 데이터와 같은 U-Net 초깃값으로 50 epochs 동안 한 번씩 처음부터 학습하였고, 2 epoch마다 검증하여 검증 집합의 SSIM이 가장 높은 체크포인트를 최종 평가에 사용하였다. 이 공통 학습 설정은 원 ETER-net 구현과 학습 데이터(혼합 대조도 4,110개 볼륨 대 AXT2 300개 볼륨), 세기 정규화(슬라이스별 정규화 대 고정 배율), 마스크(무작위 시작 위치와 ACS 31줄 대 고정 시작 위치와 32줄), 증강, 손실 함수, 최적화 방법, 계산 정밀도, GRU 은닉 크기 배수 $H$(본 연구 10, 공개 구현 12)에서 다르다. 각 지표는 슬라이스별로 계산하여 볼륨마다 평균한 뒤, 평가 볼륨 558개에 대한 평균과 표준편차로 나타내었다. 또한 같은 볼륨에 대한 두 모델의 지표를 짝지어 대응(paired) 비교하고, 상대 모델보다 우수한 볼륨의 비율(우위 비율)을 지표별로 구하였다. 유의성은 볼륨 단위 양측 Wilcoxon signed-rank 검정으로 평가하였다.
%% tokens: LAMBDA_SSIM_SCI, SBAR_N, L1_OVER_SSIM_APPROX, N_TRAIN_VOL, N_TEST_VOL (+ CFG: 99th percentile, 10^-6, 3e-5, 2e-4, 1e-6, batch 8, seed 1, 50 epochs, every 2 epochs, H 10/12, AXT2 300, ACS 31/32 — D8 defaults)

# 실험 결과

## 정량 평가

표 1은 각 모델의 최적 체크포인트를 test_full 분할에 적용한 결과이다. U-Net only와 SS2D(controlled)의 SSIM과 PSNR은 기준 모델인 ETER-net보다 각각 0.0022와 0.23 dB, 0.0008과 0.10 dB 높았고(SSIM 우위 비율 99.6%, 81.2%), nMSE는 두 모델 모두 ETER-net보다 낮았다(모든 비교에서 $p<0.001$). 그러나 마스크 없이 참조 볼륨의 최댓값을 동적 범위로 쓰는 fastMRI 공식 평가 정의[@zbontar2018fastmri]의 SSIM은 U-Net only 0.8965, ETER-net 0.8652, SS2D(controlled) 0.8389로, SS2D(controlled)가 ETER-net보다 0.0262 낮았다(우위 비율 14.3%, $p<0.001$). 같은 정의의 SSIM 지도를 평균하면 brain mask 내부에서는 SS2D(controlled)가 0.0005 높았고, 학습 손실에 포함되지 않은 마스크 외부에서는 0.0398 낮았다.
%% tokens: D_UNET_ETER_SSIM_ABS, D_UNET_ETER_PSNR_ABS, D_UNET_ETER_NMSE_ABS (sign only), WIN_UNET_ETER_SSIM_VOLPCT, P_UNET_ETER_SSIM, P_UNET_ETER_PSNR, P_UNET_ETER_NMSE, D_SS2D_ETER_SSIM_ABS, D_SS2D_ETER_PSNR_ABS, D_SS2D_ETER_NMSE_ABS (sign only), WIN_SS2D_ETER_SSIM_VOLPCT, P_SS2D_ETER_SSIM, P_SS2D_ETER_PSNR, P_SS2D_ETER_NMSE, OFFSSIM_UNET, OFFSSIM_ETER, OFFSSIM_SS2D, D_SS2D_ETER_OFFSSIM_ABS, WIN_SS2D_ETER_OFFSSIM_VOLPCT, P_SS2D_ETER_OFFSSIM, D_SS2D_ETER_OFFSSIMIN_ABS, D_SS2D_ETER_OFFSSIMOUT_ABS

SS2D(controlled)의 표 1 SSIM은 시퀀스 모듈이 없는 U-Net only보다 0.0014 낮았고(SS2D(controlled)가 우수한 볼륨 558개 중 12개, $p<0.001$), SS2D 모듈 출력의 제곱평균제곱근은 0.0057이었으며, 시험 슬라이스 512개에서 이 출력을 다른 볼륨의 출력으로 바꾸었을 때 SSIM 하락은 평균 0.00047이었다. 따라서 본 연구는 SS2D(controlled)와 ETER-net의 차이를 상태공간 모델링의 효과로 해석하지 않는다. 또한 마지막 epoch의 학습 손실은 ETER-net과 SS2D(controlled)가 U-Net only보다 각각 2.0%, 1.6% 높아, 시퀀스 모듈을 포함한 모델의 결과에는 최적화 차이(ETER-net은 계산 정밀도와 보호 장치의 차이 포함)가 반영되었을 수 있다. 각 모델을 한 번씩만 학습하였으므로 위 검정은 평가 볼륨 간 변동만 반영하고 학습 회차 간 변동은 반영하지 않는다.
%% tokens: D_SS2D_UNET_SSIM_ABS, N_TEST_VOL, WIN_SS2D_UNET_SSIM_VOL, P_SS2D_UNET_SSIM, FUNC_RMS, FUNC_N, FUNC_SHUFFLE_DROP, TRAINLOSS_GAP_ETER, TRAINLOSS_GAP_SS2D

%% TABLE1_V9_BLOCK (paper/make_tables_v8fix.py — do not edit)
%% IEIE .src.md 붙여넣기용 — v9 표 1 (v8fix 50 epochs, test_full, 볼륨 단위, brain mask 내부, 순위 표시 없음, Zero-filled 행 없음 — 사용자 결정 D2). paper/make_tables_v8fix.py 가 생성 — 손편집 금지
@table: col | 1025,1340,1035,1135
@cap_en: Volume-level results (mean ± standard deviation) on the fastMRI brain test_full split (558 volumes, R=4, brain-masked)
| Model | SSIM | PSNR (dB) | nMSE (%) |
|---|---|---|---|
| U-Net only | 0.9213±0.0466 | 34.91±1.85 | 0.371±0.409 |
| ETER-net | 0.9191±0.0474 | 34.69±1.84 | 0.388±0.434 |
| SS2D<br>(controlled) | 0.9199±0.0469 | 34.78±1.86 | 0.381±0.414 |
@note: Note. 본문의 차이는 짝지은 차이의 평균(반올림 전)이므로, 반올림한 평균끼리 뺀 값과 마지막 자리가 다를 수 있다.
@end

## 정성 비교

그림 3은 사전에 지정한 검증 AXT2 슬라이스 1개에 대한 세 모델의 출력이다. 표시 강도를 4배로 증폭한 하단 행에서 ETER-net과 SS2D(controlled)의 머리 밖 배경에는 U-Net only보다 뚜렷한 구조적 패턴이 나타났으며, 평가 볼륨 전체의 brain mask 외부 공식 정의 SSIM 지도 평균도 U-Net only 0.8592, ETER-net 0.8123, SS2D(controlled) 0.7725였다.
%% tokens: OFFSSIMOUT_UNET, OFFSSIMOUT_ETER, OFFSSIMOUT_SS2D (figure = v8fix-fig-inference, file_brain_AXT2_200_2000556 slice 6; panel values stay in the figure)

@figure: paper/figs/fig3_qualitative_col_v8fix.png | col | 1.0
@cap_en: AXT2 validation slice without intensity rescaling: reconstructions with brain-masked PSNR (dB)/SSIM (top), brain-masked error maps (middle), and ×4 gain images (bottom)

# 결론

본 연구에서 기준 모델인 ETER-net과, 그 시퀀스 모듈만 교체하거나 제거한 두 모델을 fastMRI brain 데이터셋의 R=4 조건에서 한 번씩 학습하여 test_full 분할에서 평가한 결과, brain mask 내부의 볼륨 평균 SSIM은 ETER-net보다 U-Net only가 0.0022, SS2D(controlled)가 0.0008 높았다. 다만 SS2D(controlled)는 U-Net only보다 SSIM이 0.0014 낮았으므로, 본 연구는 SS2D(controlled)와 ETER-net의 차이를 상태공간 모델링의 효과로 해석하지 않으며 SSM과 순환신경망 전반의 우열로 일반화할 수도 없다. 두 모듈은 순환 방식과 파라미터 구성이 다르고 SS2D 모듈은 2차원 역 푸리에 변환을 표현할 수 없어, 본 연구에서는 각 요인의 기여를 분리하지 못하였다. 또한 각 모델을 한 번씩만 학습하였고 시퀀스 모듈을 포함한 모델의 학습 손실이 더 높았으므로, 시퀀스 모듈이 재구성에 미치는 효과의 방향은 판단할 수 없다. 공식 평가 정의에서 마스크 외부의 차이로 SS2D(controlled)와 ETER-net의 순위가 바뀐 점, 학습 설정이 원 구현과 다른 점, AXT2 비중이 높은 단일 데이터셋과 단일 가속화 계수에서 평가한 점도 본 연구의 한계이다. 향후 연구에서는 여러 난수 시드와 원 구현의 학습 설정, 더 높은 가속화 계수에서 시퀀스 모듈의 기여를 평가할 계획이다.
%% tokens: D_UNET_ETER_SSIM_ABS, D_SS2D_ETER_SSIM_ABS, D_SS2D_UNET_SSIM_ABS (+ CFG: R=4)

# REFERENCES
