# 가속 MRI 재구성 네트워크 ETER-Net에서 순환신경망을 선택적 상태공간모델로 치환한 통제 비교 연구

**A Controlled Study of Replacing the RNN with a Selective SSM in ETER-Net for Accelerated MRI Reconstruction**

*(대한전자공학회 투고용 논문 양식 2021 — 저자·소속은 투고 시스템/게재용 양식에서 기입)*

## 요 약

ETER-Net은 언더샘플링된 k-space를 양방향 순환신경망(bi-RNN)을 활용해 영상 도메인으로 직접 변환하는 MRI 재구성 방법이다. 본 연구는 ETER-Net의 도메인 변환 모듈을 제외한 구조와 학습 설정을 동일하게 유지하고, 양방향 GRU(bi-GRU)를 2차원 선택적 상태공간모델(SS2D)로 교체했을 때의 재구성 성능을 비교한다. fastMRI brain multicoil 데이터셋(384×384, R=4)에서 두 모델은 도메인 변환 모듈을 제외한 모든 구성요소(데이터·마스크·U-Net 구조·손실·최적화 설정)가 동일하며, 가중치 공유 없이 각각 독립적으로 학습되었고, 기존 ETER-Net 설계와 동일하게 명시적 데이터 일관성(DC) 블록은 두지 않았다. 검증 집합(464 볼륨/7,334 슬라이스) 평가 결과, SS2D 모델은 bi-GRU 모델의 약 1/21에 해당하는 파라미터(31M vs. 668M)로 볼륨 단위 평균 brain-masked SSIM 0.9141(bi-GRU 0.9127), PSNR 33.91 dB(33.78 dB)를 기록하여 세 평가 지표(SSIM·PSNR·nMSE) 모두에서 기존 모델보다 우수한 성능을 보였다. SS2D가 우위인 비율은 슬라이스 단위로 SSIM 78.2%·PSNR 73.8%·nMSE 73.8%, 볼륨 단위로 각각 94.8%·89.9%·90.1%였다(볼륨 단위 Wilcoxon signed-rank 검정, p<0.001). 제시한 정성 예시에서는 두개골 바깥 배경의 주기적 ringing 아티팩트가 bi-GRU 재구성보다 SS2D 재구성에서 덜 두드러졌다. fastMRI brain 데이터의 R=4 조건에서 모델별로 1회씩 학습한 결과는, bi-GRU 모듈을 SS2D로 교체하여 명시적 DC 블록 없이도 더 적은 파라미터로 재구성 품질을 개선할 수 있음을 시사한다.

## Abstract

ETER-Net is an MRI reconstruction method that directly transforms undersampled k-space into the image domain using a bidirectional recurrent neural network (bi-RNN). This study evaluates the effect of replacing the bi-GRU module in ETER-Net with a two-dimensional selective state-space model (SS2D), while keeping the remaining architecture and training settings unchanged. Experiments are conducted on the fastMRI brain multicoil dataset at an image size of 384×384 and an acceleration factor of R=4. Both models use the same data, undersampling masks, U-Net architecture, loss function, and optimization settings. They are trained independently without weight sharing, and neither includes an explicit data-consistency (DC) block, consistent with the original ETER-Net design. On the validation set (464 volumes/7,334 slices), the SS2D model, with approximately one twenty-first as many parameters as the bi-GRU model (31M vs. 668M), achieves a volume-averaged brain-masked SSIM (computed within an intensity-based foreground mask covering the head region) of 0.9141 (vs. 0.9127) and PSNR of 33.91 dB (vs. 33.78 dB), outperforming the original bi-GRU design on all three standard metrics (SSIM, PSNR, nMSE). SS2D is superior on 78.2%/73.8%/73.8% of slices and on 94.8%/89.9%/90.1% of volumes for SSIM/PSNR/nMSE, respectively (volume-level Wilcoxon signed-rank test, p<0.001). In the illustrated example, periodic ringing artifacts outside the skull are less pronounced in the SS2D reconstruction than in the bi-GRU reconstruction. These findings suggest that replacing the bi-GRU module with SS2D can improve reconstruction quality with substantially fewer parameters and without an explicit DC block, under the evaluated setting of one training run per model on fastMRI brain data at R=4.

**Keywords :** MRI reconstruction, accelerated MRI, ETER-Net, recurrent neural network, selective state-space model, SS2D, controlled comparison

---

## Ⅰ. 서론

MRI는 k-space를 순차적으로 수집하므로 촬영 시간이 길다. 촬영 시간을 단축하기 위해 언더샘플링된 k-space로부터 영상을 복원하는 딥러닝 기법이 연구되어 왔다. 대표적인 접근으로는 물리 모델 기반 반복 최적화 과정을 신경망 층으로 펼치는 방식(unrolling)[1]과 k-space를 영상으로 직접 변환하는 도메인 변환 방식[2]이 있다. ETER-Net[3]은 후자에 속하며, 양방향 GRU(bi-GRU)가 k-space를 행·열 방향으로 읽어 영상 도메인 특징으로 변환하고 U-Net이 aliasing 아티팩트를 줄이며, 명시적 데이터 일관성(DC) 블록은 없다. 이후 non-Cartesian 궤적[4]과 ViT 인코더 결합[5]으로 확장되었으나 도메인 변환의 핵심인 bi-RNN은 유지되었다. 그러나 bi-GRU는 순차 처리 구조여서 병렬화가 어렵다. 또한 행·열 방향으로 펼친 고차원 입력을 GRU로 처리하므로 대규모 가중치 행렬이 필요하며, 전체 파라미터 수가 수억 개에 이른다. 선택적 상태공간모델(SSM)인 Mamba[6]는 입력 의존 순환을 병렬 스캔으로 선형 시간에 계산하고, 이를 2차원으로 확장한 SS2D[7]는 선택적 스캔을 여러 방향으로 적용하여 2차원 공간 정보를 처리한다. 기존의 Mamba 기반 MRI 재구성[8, 9]은 새로운 구조 전체를 제안하므로, 기존 골격에서 순환신경망만 SSM으로 바꾼 효과를 분리하여 평가할 수 없다. 본 논문은 ETER-Net에서 도메인 변환 시퀀스 모델만 bi-GRU에서 SS2D로 치환하고 나머지를 모두 고정한 통제 비교를 보고한다.

## Ⅱ. 방법

### 1. 비교 실험의 공통 재구성 파이프라인

그림 1의 파이프라인에서는 완전 샘플링된(fully-sampled) k-space y_c(처음 16개 코일 채널)에 R=4 equispaced 마스크 M(ACS 8%)을 곱해 언더샘플링 k-space ỹ_c=M⊙y_c를 얻는다. 이를 실수·허수 채널로 분리한 (32, 384, 384) 텐서가 입력이며, 참조 영상 x*는 데이터셋이 제공하는 전체 코일 RSS 영상을 384×384로 crop/pad한 것이다. 시퀀스 모델 f_θ가 k-space를 영상 도메인 특징(20채널)으로 직접 변환하고, 이를 동일한 언더샘플링 k-space ỹ_c에 코일별 역 FFT를 적용하여 얻은 zero-filled 코일 영상(32채널)과 채널 결합한다. 후처리 U-Net g_φ(기존 ETER-Net 구현과 동일한 구조: dual-frame skip, depth 5, 기본 채널 수 64, 31.1M 파라미터)는 이 결합 텐서로부터 magnitude 영상 x̂를 출력한다(식 (1)).

$$ \hat{x} = g_{\phi}\left( \mathrm{concat}\left( f_{\theta}(\{\tilde{y}_c\}), \mathcal{F}^{-1}\{\tilde{y}_c\} \right) \right) \qquad (1) $$

이하 두 비교 모델은 f_θ 외의 데이터·마스크·손실·최적화·U-Net 구조가 동일하고, 가중치 공유 없이 동일한 학습 설정으로 각각 처음부터 독립 학습하며, 원 논문[3]에 따라 DC 블록은 두지 않는다.

![그림 1](../figs/conf_fig1_pipeline.png)

그림 1. 두 모델에 공통인 ETER-Net 통제 파이프라인 구성(점선 상자의 f_θ만 교체 대상)

### 2. 시퀀스 모델의 두 구성

그림 2(a)의 bi-GRU 모델은 원 ETER-Net의 구조를 그대로 따라 k-space를 384개 행의 시퀀스(스텝당 12,288차원)로 펼쳐 양방향 GRU를 통과시킨 뒤, 전치하여 열 방향으로 한 번 더 통과시킨다. 두 GRU의 입력–은닉 행렬이 파라미터의 대부분을 차지하여 bi-GRU 스택만 637.1M, 모델 전체 668.2M에 이르며(방향별 은닉 차원 3,840 = 384×10 기준; 원 코드의 또 다른 설정 384×12에서는 880.5M), 순환 연산은 시간 스텝 순서대로 순차 계산되어야 한다. 그림 2(b)의 SS2D 모델은 SS2D[7]의 구성을 따라, 픽셀별 LN·Linear(32→128)·SiLU와 depthwise conv 뒤에 선택적 스캔(S6)을 네 방향(각 행 →/←, 각 열 ↓/↑, L=384)으로 적용하고 네 출력을 채널 결합한 뒤 LN·Linear와 1×1 conv로 bi-GRU와 같은 20채널로 투영한다. S6[6]는 식 (2)의 이산화된 상태공간 시스템에서 (Δ_t, B_t, C_t)를 입력 x_t로부터 생성하는 선택적 순환이다(h_t: 은닉 상태, y_t: 출력, A: 대각 상태 행렬, D: skip 계수). 채널(d_inner=128)마다 독립적으로 적용되며 상태 차원은 d_state=16이다. 본 구현은 각 행과 열을 길이 L=384의 독립된 시퀀스로 처리하며, 각 위치에서 행 방향과 열 방향의 스캔 출력을 결합한다.

$$ h_t = \exp(\Delta_t A)\, h_{t-1} + \Delta_t B_t x_t, \quad y_t = C_t h_t + D x_t \qquad (2) $$

각 방향에서 사용하는 하나의 가중치 집합을 그 방향의 모든 행(열)이 공유하므로 SSM 스택은 0.12M(모델 전체 31.2M)에 그친다. 선택적 스캔은 병렬화가 가능하며, 연산량은 시퀀스 길이 L에 선형적으로 비례한다. 표 1의 SS2D (controlled), 즉 통제 SS2D 모델은 게이팅을 제외한 단일 블록으로 구성하였으며, bi-GRU 모델보다 적은 파라미터를 사용한다. 통제를 완화한 변형으로, stride-3 다운샘플링과 fp16 스캔을 적용한 뒤 Mamba 게이팅 y·SiLU(z)를 복원한 잔차 SS2D 블록 3개(d_inner 256, d_state 32, 출력 64채널)를 쌓은 강화 SS2D 모델(표 1의 SS2D (enhanced), 34.2M)도 학습하였다. 강화 SS2D 모델은 통제 모델과 달리 80 epoch 동안 학습하였으며, 구조, 학습 기간 및 파라미터화가 함께 변경되었다. 각 변경 요소에 대한 절제 실험(ablation study)은 수행하지 않았으므로, 개별 요소의 기여는 구분할 수 없다.

![그림 2](../figs/conf_fig2_arms.png)

그림 2. 시퀀스 모델 f_θ의 두 구성: (a) 원 bi-GRU, (b) SS2D

### 3. 학습·평가 프로토콜

fastMRI brain multicoil 데이터셋[10]의 공식 학습·검증 분할을 유지하였다. 파일 손상으로 읽을 수 없는 학습 볼륨 2개를 제외하고, 나머지 데이터를 모두 사용하였다. train 4,108 볼륨/65,028 슬라이스, val 464 볼륨/7,334 슬라이스이고, contrast는 AXT1·AXT1POST·AXT1PRE·AXT2·AXFLAIR 혼합이다(val의 58%가 AXT2). 전처리는 full k-space → 역 FFT → 384×384 crop/pad → 재-FFT의 retrospective 프로토콜이다. 강도 정규화는 슬라이스별로 하지 않고 고정 배율(영상·참조 영상 ×10⁶, k-space ×10⁴)을 곱하였으며, 코일은 처음 16개 채널을 사용하고(16개 미만인 볼륨은 0으로 채움), 학습 시 마스크 오프셋은 각 샘플에 대해 무작위로 설정하고 검증 시에는 고정하였다. 데이터 증강은 영상 flip 후 k-space를 재계산하는 방식을 사용하였다. 배경 영역이 평가 결과에 미치는 영향을 줄이기 위해 참조 영상 x*의 강도를 기준으로 슬라이스마다 전경 마스크 m(Otsu 임계×0.4 이상 화소의 최대 연결성분; 이하 brain mask)을 만들어 손실(식 (3))과 지표를 그 내부에서만 계산하며, 같은 마스크를 두 모델과 표 1의 공개 모델 평가에 동일하게 적용한다. SSIM_m은 SSIM 지도의 마스크 내부 위치에 해당하는 값을 평균한 값이다. 손실에서는 원 ETER-Net 코드의 구현(11×11 가우시안 창)을, 평가에서는 scikit-image 구현(data_range = 마스크 내 x*의 최대−최소)을 사용하여 SSIM 지도를 구하였고, PSNR의 peak와 nMSE의 분모도 마스크 내부에서 계산하였다. 따라서 표 1의 수치는 전체 영상 기준의 fastMRI 공식 수치와 직접 비교되지 않는다.

$$ \mathcal{L} = \frac{\sum_{i} m_i \left| \hat{x}_i - x^{*}_i \right|}{\sum_{i} m_i} + \left( 1 - \mathrm{SSIM}_m(\hat{x}, x^{*}) \right) \qquad (3) $$

최적화에는 Adam을 사용하였으며, 학습률은 2×10⁻⁴, weight decay는 3×10⁻⁵로 설정하였다. Cosine 학습률 스케줄의 최소 학습률은 1×10⁻⁶로 설정하였고, gradient clipping 1.0, AMP 및 배치 크기 8을 적용하였다. 각 모델은 TITAN RTX GPU 1대에서 독립적으로 1회 학습하였으며, 난수 시드는 고정하지 않았다. bi-GRU와 통제 SS2D 모델은 50 epoch, 강화 SS2D 모델은 80 epoch 동안 학습하였고, 검증은 2 epoch마다 수행하였다. 최적 체크포인트는 검증 집합에서 계산한 복합 선택 점수 C = 0.5·SSIM + 0.3·min(PSNR, 40 dB)/40 + 0.2·(1 − min(nMSE, 1))이 최대인 시점으로 선정하였다. 세 지표는 모두 brain mask 내부에서 계산한 값이고, PSNR은 40 dB를 상한으로 클리핑하여 [0, 1]로 정규화하였으며, 낮을수록 좋은 nMSE(무차원 비율)는 1로 클리핑한 뒤 1에서 빼 값이 클수록 좋은 방향으로 변환하였다. 이 점수는 체크포인트 선택에만 사용하였고, 보고하는 성능은 모두 SSIM·PSNR·nMSE 원값이다. 평가 지표로는 SSIM(주지표), PSNR 및 nMSE(%)를 사용하였다. 각 지표는 슬라이스별로 계산한 뒤 볼륨 내에서 평균하였고, 최종 결과는 각 볼륨의 평균값에 대한 전체 검증 집합의 평균±표준편차로 보고한다. paired 설계이므로 우위 비율(슬라이스·볼륨)을 함께 보고하고, 볼륨 단위 양측 Wilcoxon signed-rank 검정과 볼륨 클러스터 부트스트랩(2,000회) 95% 신뢰구간(CI)으로 유의성을 평가하였다.

## Ⅲ. 실험 결과

### 1. 정량 비교

표 1은 학습·평가 프로토콜에 기술한 기준으로 선정한 최적 체크포인트(bi-GRU 50 epoch, 통제 SS2D 48 epoch, 강화 SS2D 78 epoch)를 검증 집합 전체에 적용한 결과이다. 체크포인트 선택과 최종 평가에는 동일한 검증 집합을 사용하였으며, 별도의 테스트 집합은 구성하지 않았다. 통제 SS2D 모델은 bi-GRU 모델의 약 1/21에 해당하는 파라미터를 사용하면서도 더 높은 SSIM과 PSNR, 더 낮은 nMSE를 기록하였다. 슬라이스 단위 paired 비교에서 SS2D 우위 비율은 SSIM 78.2%(95% CI 76.8~79.7), PSNR 73.8%, nMSE 73.8%이고 볼륨 단위로는 94.8%·89.9%·90.1%였다(볼륨 단위 양측 Wilcoxon signed-rank 검정, 모든 지표 p<0.001; 슬라이스 단위 ΔSSIM 평균 +0.0014, 95% CI +0.0013~+0.0015). PSNR과 nMSE는 슬라이스 단위에서 MSE의 단조 함수이므로 슬라이스 우위 비율이 서로 같다. 학습 로그의 25회 검증 시점 전부에서 SS2D의 검증 SSIM이 bi-GRU 이상이었고(동률 1회) PSNR은 25회 모두 상회하였으며, contrast 서브그룹과 평가 지표의 모든 조합에서 SS2D가 우수한 슬라이스의 비율은 68.7% 이상이었다. 다만 epoch당 학습 시간은 cuDNN으로 최적화된 bi-GRU(2.41 h)가 SS2D(3.07 h)보다 짧았다. 이는 사용한 선택적 스캔 커널이 cuDNN의 GRU 구현만큼 최적화되어 있지 않기 때문으로 추정되며, 본 연구에서 SS2D의 효율 이점은 연산 시간이 아니라 파라미터 수에 국한된다. 강화 SS2D가 통제 모델보다 우수한 값을 보인 슬라이스의 비율은 지표별로 54~56%였다. 별도로 수행한 볼륨 단위 양측 Wilcoxon signed-rank 검정에서는 세 지표 모두 두 모델 간 차이가 유의하였다(p<0.001). 반면 볼륨 단위 평균 nMSE는 강화 모델 0.439%, 통제 모델 0.438%로 유사했으며, 강화 모델이 소폭 높았다. 50 epoch 시점의 검증 SSIM은 통제 모델 0.9138, 강화 모델 0.9130으로 통제 모델이 더 높았다. 학습 중 검증과 최종 평가는 동일한 SSIM 구현을 사용하며, 표 1의 값은 최적 체크포인트를 볼륨 단위로 집계한 것이므로 이 시점의 값과는 다르다. 따라서 최종 성능 차이에 학습 기간 연장이 기여했을 가능성이 있으나, 구조도 함께 변경되었으므로 각 요인의 영향을 분리할 수 없다. 표 1 하단의 U-Net†[10]·E2E-VarNet†[1]·PromptMR+[11]는 표 1 주석의 조건으로 추론한 참고 결과이며, 우열 비교 대상이 아니다.

표 1. fastMRI brain 검증 집합 전체(464 볼륨/7,334 슬라이스, R=4, brain-masked)의 볼륨 단위 결과(평균±표준편차)

| Method | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---|---|---|---|
| Zero-filled | – | 0.7523±0.0410 | 24.76±2.11 | 3.935±2.166 |
| bi-GRU (original) | 668 | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 |
| SS2D (controlled) | 31 | <u>0.9141±0.0365</u> | <u>33.91±1.90</u> | **0.438±0.283** |
| SS2D (enhanced) | 34 | **0.9146±0.0361** | **33.92±1.90** | <u>0.439±0.304</u> |
| U-Net† | 496 | 0.8971±0.0366 | 30.95±2.29 | 0.973±0.796 |
| E2E-VarNet† | 30 | 0.9181±0.0386 | 32.78±3.21 | 1.133±1.291 |
| PromptMR+ | 93 | 0.9417±0.0349 | 36.12±4.02 | 0.526±0.841 |

상단 4행은 본 연구 프로토콜의 결과이며, 굵게·밑줄은 학습 모델 3개 중 최고·차선값이다. Zero-filled는 마스크된 k-space를 역 FFT한 코일 영상의 RSS이다. 하단 3행은 공개 가중치를 본 연구 프로토콜(384² 재-FFT·16코일·R=4)로 CPU fp32에서 추론한 참고 결과이다. 원 학습 조건과 달라 도메인 시프트가 있고, brain mask 내 슬라이스별 최소제곱 강도 정합을 공개 모델에만 적용하였으므로 직접적인 성능 비교 대상에서는 제외하였다. †: fastMRI leaderboard 가중치(본 검증 집합을 포함한 train+val로 학습). PromptMR+: train 분할로만 학습, 12-cascade unrolled, 인접 5슬라이스 입력.

### 2. 정성 비교

그림 3은 검증 집합에서 사전에 지정한 검사용 슬라이스 12개 중 1개(AXT2)의 결과이다. 위 행은 재구성 영상(참조 영상·zero-filled·bi-GRU·SS2D 순, 패널 안 수치는 해당 슬라이스의 PSNR/SSIM), 가운데 행은 brain-masked 오차 지도, 아래 행은 표시 강도를 4배 증폭하여 배경의 저강도 아티팩트를 드러낸 ×4 gain 영상이다. 오차 지도와 패널 수치에서 SS2D 재구성 영상의 오차가 다소 작았으며, gain 영상에서는 두개골 바깥 배경의 주기적 ringing 아티팩트가 bi-GRU 재구성에서 두드러지는 반면 SS2D 재구성에서는 약하게 나타난다. brain mask 밖의 이러한 아티팩트는 표 1의 지표에 반영되지 않으며 별도로 정량화하지는 않았다.

![그림 3](../figs/fig3_qualitative_col.png)

그림 3. 검증 슬라이스(AXT2) 정성 비교: 재구성(위), brain-masked 오차(가운데), ×4 gain(아래)

## Ⅳ. 결론

본 연구는 ETER-Net의 도메인 변환 모듈을 bi-GRU에서 SS2D로 교체하고, 나머지 구조와 학습 설정을 동일하게 유지하여 재구성 성능을 비교하였다. fastMRI brain 검증 집합의 R=4 조건에서 통제 SS2D 모델은 명시적 DC 블록 없이 bi-GRU 모델의 약 1/21에 해당하는 파라미터를 사용하면서 SSIM, PSNR 및 nMSE에서 개선된 결과를 보였다. 제시한 정성 예시에서도 두개골 바깥 배경의 링잉 아티팩트가 약하게 관찰되었다. 다만 이러한 결과는 본 연구의 실험 조건에 한정되며, SSM과 RNN 전반의 우열로 일반화할 수는 없다. 두 모델은 순환 메커니즘과 파라미터화가 함께 달라 각 요인의 기여를 분리하지 못하였다. 또한 도메인 변환 모듈을 제외한 기준 모델과의 비교를 수행하지 않아 해당 모듈 자체의 기여는 평가하지 못하였다. 난수 시드를 고정하지 않은 모델별 단일 학습, AXT2 비중이 높은 단일 데이터셋, 단일 가속률에서의 평가도 본 연구의 한계이다. 현재 여러 난수 시드를 사용한 반복 실험을 진행하고 있으며, Transformer 및 가중치를 공유하는 pixel-GRU 모델과의 비교와 여러 가속률에서의 일반화 성능 평가를 수행할 계획이다.

## REFERENCES

[1] A. Sriram et al., "End-to-End Variational Networks for Accelerated MRI Reconstruction," in *Proc. MICCAI*, LNCS, vol. 12262, pp. 64–73, 2020.  
[2] B. Zhu, J. Z. Liu, S. F. Cauley, B. R. Rosen, and M. S. Rosen, "Image reconstruction by domain-transform manifold learning," *Nature*, vol. 555, pp. 487–492, 2018.  
[3] C. Oh, D. Kim, J.-Y. Chung, Y. Han, and H. Park, "A k-space-to-image reconstruction network for MRI using recurrent neural network," *Med. Phys.*, vol. 48, no. 1, pp. 193–203, 2021.  
[4] C. Oh, J.-Y. Chung, and Y. Han, "An End-to-End Recurrent Neural Network for Radial MR Image Reconstruction," *Sensors*, vol. 22, no. 19, Art. no. 7277, 2022.  
[5] C. Oh, "A Hybrid Vision Transformer-BiRNN Architecture for Direct k-Space to Image Reconstruction in Accelerated MRI," *J. Imaging*, vol. 12, no. 1, Art. no. 11, 2025.  
[6] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv preprint arXiv:2312.00752, 2023.  
[7] Y. Liu et al., "VMamba: Visual State Space Model," in *Proc. NeurIPS*, 2024.  
[8] Y. Korkmaz and V. M. Patel, "MambaRecon: MRI Reconstruction with Structured State Space Models," in *Proc. IEEE/CVF WACV*, 2025.  
[9] J. Huang et al., "Enhancing global sensitivity and uncertainty quantification in medical image reconstruction with Monte Carlo arbitrary-masked Mamba," *Med. Image Anal.*, vol. 99, Art. no. 103334, 2025.  
[10] J. Zbontar, F. Knoll, A. Sriram et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI," arXiv preprint arXiv:1811.08839, 2018.  
[11] B. Xin, M. Ye, L. Axel, and D. N. Metaxas, "Rethinking Deep Unrolled Model for Accelerated MRI Reconstruction," in *Proc. ECCV*, LNCS, vol. 15133, pp. 164–181, 2024.  
