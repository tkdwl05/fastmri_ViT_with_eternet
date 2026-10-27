# 가속 MRI 재구성 네트워크 ETER-Net에서 순환신경망을 선택적 상태공간모델로 치환한 통제 비교 연구

***, *** (*** 소속, e-mail : ***)

**A Controlled Study of Replacing the RNN with a Selective SSM in ETER-Net for Accelerated MRI Reconstruction**

*** and *** (*** University)

*(대한전자공학회 학술대회 2쪽 양식 — 작성예시 example_conference_2page.docx 기준; 저자·소속은 "***" 자리표시자)*

## Abstract

ETER-Net is an MRI reconstruction method that directly transforms undersampled k-space into the image domain using a bidirectional recurrent neural network (bi-RNN). This study evaluates the effect of replacing the bidirectional gated recurrent unit (bi-GRU) module in ETER-Net with a two-dimensional selective state-space model (SS2D), while keeping the remaining architecture and training settings unchanged. Experiments are conducted on the fastMRI brain multicoil dataset at an image size of 384×384 and an acceleration factor of R=4. Both models use the same data, undersampling masks, U-Net architecture, loss function, and optimization settings. They are trained independently without weight sharing, and neither includes an explicit data-consistency (DC) block, consistent with the original ETER-Net design. The SS2D model uses approximately 1/21 of the parameters of the bi-GRU model (31M vs. 668M). Despite its significantly smaller size, the SS2D model achieved superior performance on the validation set, yielding a volume-averaged SSIM of 0.9141 (vs. 0.9127), PSNR of 33.91 dB (vs. 33.78 dB), and nMSE of 0.438% (vs. 0.448%), all computed within a brain mask. SS2D is superior on 78.2%/73.8%/73.8% of slices and on 94.8%/89.9%/90.1% of volumes for SSIM/PSNR/nMSE, respectively (volume-level Wilcoxon signed-rank test, p<0.001). In the illustrated example, periodic ringing artifacts outside the skull are less pronounced in the SS2D reconstruction than in the bi-GRU reconstruction. These findings suggest that replacing the bi-GRU module with SS2D can improve reconstruction quality with substantially fewer parameters and without an explicit DC block, under the evaluated setting of one training run per model on fastMRI brain data at R=4.

---

## Ⅰ. 서론

MRI는 k-space를 순차적으로 수집하므로 촬영 시간이 길다. 촬영 시간을 단축하기 위해 언더샘플링된 k-space로부터 영상을 복원하는 딥러닝 기법이 연구되어 왔다. 대표적인 접근으로는 물리 모델 기반 반복 최적화 과정을 신경망 층으로 펼치는 방식(unrolling)[1]과 k-space를 영상으로 직접 변환하는 도메인 변환 방식[2]이 있다. ETER-Net[3]은 후자에 속하며, 양방향 GRU(gated recurrent unit, bi-GRU)가 k-space를 행·열 방향으로 읽어 영상 도메인 특징으로 변환하며, U-Net을 통해 aliasing 아티팩트를 줄이며, 명시적 데이터 일관성(DC) 블록은 사용하지 않았다. 이후 non-Cartesian 궤적[4]과 Vision Transformer(ViT) 인코더 결합[5]으로 확장되었으나 도메인 변환의 핵심인 bi-RNN은 유지되었다. 그러나 ETER-Net은 행·열 방향을 순차 처리하는 구조여서 병렬화가 어렵고, flattened 2D 고차원 입력을 GRU로 처리하므로 대규모 가중치 행렬이 필요하며, 전체 파라미터 수가 수억 개에 이른다. 선택적 상태공간모델(SSM)인 Mamba[6]는 입력 의존 순환을 병렬 스캔으로 선형 시간에 계산하고, 이를 2차원으로 확장한 SS2D[7]는 선택적 스캔을 여러 방향으로 적용하여 2차원 공간 정보를 처리한다. 기존의 Mamba 기반 MRI 재구성[8, 9]은 새로운 구조 전체를 제안하므로, 기존 골격에서 순환신경망만 SSM으로 바꾼 효과를 분리하여 평가할 수 없다. 본 논문은 ETER-Net에서 도메인 변환 시퀀스 모듈만 bi-GRU에서 SS2D로 치환하고, 나머지는 모두 고정한 상태에서 통제 비교 분석을 수행한다.

## Ⅱ. 방법

### 2.1 비교 실험의 공통 재구성 파이프라인

그림 1의 파이프라인에서는 완전 샘플링된(fully-sampled) k-space $y_c$($c$는 코일 인덱스)에 가속화 계수가 4인(R=4) 등간격(equispaced) 마스크 $M$(auto-calibration signal(ACS) 8%)을 적용하여, 언더샘플링된 k-space $\tilde{y}_c$(=$M \odot y_c$, $\odot$는 원소별 곱)를 생성한다. 이를 실수·허수 채널로 분리한 (32, 384, 384) 크기의 텐서가 네트워크의 입력이다. 한편, 참조 영상 $x^{*}$는 데이터셋이 제공하는 전체 코일 root-sum-of-squares(RSS) 영상을 384×384 크기로 크롭 및 패딩하여 생성한다. 순차적 입력을 처리하는 모듈 $f_\theta$는 입력 k-space를 영상 도메인 특징(20채널)으로 직접 변환한다. 동일한 $\tilde{y}_c$에 코일별 역 푸리에 변환을 적용한 zero-filled 영상(32채널)과 앞서 생성된 영상 도메인 특징을 채널 축으로 결합한다. 후처리를 담당하는 U-Net $g_\phi$가 결합 텐서를 입력으로 하여 최종 결과인 크기(magnitude) 영상 $\hat{x}$를 출력한다(식 (1)). 실험에 사용한 U-Net은 dual-frame skip이 포함된 구조이며, 파라미터로 depth=5, 기본 채널 수=64를 사용하였으며, 전체 파라미터의 수는 31.1M이다.

$$ \hat{x} = g_{\phi}\left( \mathrm{concat}\left( f_{\theta}(\{\tilde{y}_c\}), \mathcal{F}^{-1}\{\tilde{y}_c\} \right) \right) \qquad (1) $$

식 (1)에서 $\{\tilde{y}_c\}$는 언더샘플링된 k-space $\tilde{y}_c$를 모든 코일 인덱스 $c$에 대해 모은 집합이고, $f_\theta(\cdot)$는 학습 파라미터 $\theta$를 갖는 시퀀스 모듈이며, $\mathcal{F}^{-1}\{\cdot\}$는 코일별 2차원 역 푸리에 변환이다. $\mathrm{concat}(\cdot, \cdot)$은 두 입력 텐서를 채널 축으로 결합하는 연산이고, $g_\phi(\cdot)$는 학습 파라미터 $\phi$를 갖는 U-Net이며, $\hat{x}$는 재구성된 크기 영상이다. 이하에서 비교하는 두 모델은 $f_\theta$를 제외한 데이터, 마스크, 손실 함수, 최적화 설정, U-Net 구조가 동일하며, 가중치 공유 없이 동일한 학습 설정으로 각각 처음부터 독립적으로 학습되었고, 원 논문[3]과 동일하게 명시적 DC 블록을 포함하지 않는다.

![Fig. 1](../figs/conf_fig1_pipeline.png)

Fig. 1. Common ETER-Net pipeline of the two compared models (no weight sharing); only the sequence module $f_\theta$ (dashed box) is replaced

### 2.2 시퀀스 모듈의 두 구성

ETER-Net은 그림 2(a)에 나타나듯이, 입력 k-space를 384개 행의 시퀀스(스텝당 12,288차원)로 펼쳐 양방향 GRU를 통과시킨 뒤, 전치하여 열 방향으로 한 번 더 통과시키는 구조이다. 두 GRU의 입력–은닉 행렬이 파라미터의 대부분을 차지하여 bi-GRU의 파라미터의 수가 637.1M, 모델의 전체 파라미터의 수가 668.2M이다. 그림 2(b)는 SS2D[7] 모듈을 나타낸다. 이 모듈은 먼저 각 k-space 위치의 32채널 입력 벡터에 층 정규화(LN), 32채널을 128채널로 바꾸는 선형 투영, SiLU(sigmoid linear unit) 활성화를 차례로 적용한 뒤, 3×3 depthwise convolution과 SiLU 활성화로 이웃 위치의 정보를 혼합한다. 이어서 선택적 스캔(S6)을 네 방향, 즉 각 행의 좌우 두 방향과 각 열의 상하 두 방향으로 적용한다. 이때 각 행과 열은 길이 $L$=384의 독립된 시퀀스로 처리되며, 네 방향은 서로 다른 S6 파라미터를 사용한다. 마지막으로 이 모듈은 네 방향의 출력을 채널 축으로 결합(512채널)한 뒤 LN, 512채널을 128채널로 바꾸는 선형 투영, 128채널을 20채널로 바꾸는 1×1 convolution을 차례로 거쳐 bi-GRU와 같은 20채널의 영상 도메인 특징으로 투영한다. S6[6]는 구조화 상태공간 시퀀스 모델(structured state space sequence model, S4)에 선택 메커니즘(selection mechanism)을 더하고 스캔(scan) 알고리즘으로 계산하는 모델을 가리키는 약칭으로서, S4의 명칭을 이루는 네 개의 S에 selective와 scan의 S 두 개를 더하여 붙인 이름이며, 식 (2)의 이산화된 상태공간 시스템에서 계수 $(\Delta_t, B_t, C_t)$를 위치 $t$의 입력 벡터로부터 생성하는 선택적 순환 연산이다.

$$ h_t = \exp(\Delta_t A)\, h_{t-1} + \Delta_t B_t x_t, \quad y_t = C_t h_t + D x_t \qquad (2) $$

식 (2)에서 아래첨자 $t$는 스캔 방향을 따라 부여한 시퀀스 내 위치($t = 1, \ldots, L$, $L$=384)를 의미하고, $x_t$는 위치 $t$의 입력, $h_t$는 위치 $t$까지의 정보를 요약한 $N$차원 은닉 상태 벡터, $h_{t-1}$은 직전 위치 $t-1$의 은닉 상태 벡터, $y_t$는 위치 $t$의 출력이다. $A$는 학습 가능한 $N \times N$ 대각 상태 행렬이고, $\exp(\Delta_t A)$는 $A$를 간격 $\Delta_t$로 이산화한 상태 전이 행렬이다. $\Delta_t$는 이산화 간격의 역할을 하는 채널별 양의 스칼라, $B_t$는 입력을 은닉 상태에 반영하는 입력 행렬($N$차원 벡터), $C_t$는 은닉 상태로부터 출력을 계산하는 출력 행렬($N$차원 벡터)이며, $D$는 입력을 출력에 직접 더하는 학습 가능한 스칼라 skip 계수이다. 식 (2)는 32채널을 128채널로 바꾸는 선형 투영 후의 내부 채널 $d_{\mathrm{inner}}$=128개 각각에 독립적으로 적용되므로, $x_t$와 $y_t$는 해당 채널의 스칼라 값이고 $A$와 $D$는 채널마다 별도로 학습된다. $\Delta_t$, $B_t$, $C_t$는 위치 $t$의 128차원 입력 벡터를 선형 투영하여 생성한 값이다. 이때 $\Delta_t$는 투영 결과에 softplus 함수를 적용하여 채널마다 얻는 값이고, $B_t$와 $C_t$는 모든 채널이 공유하는 값이다. 은닉 상태의 차원 $N$은 $d_{\mathrm{state}}$=16으로 설정하였다. $\Delta_t$, $B_t$, $C_t$가 입력에 따라 달라지는 점이 고정 계수를 사용하는 기존 상태공간모델과의 차이이며, 이러한 성질을 선택적(selective)이라 부른다. bi-GRU의 순환 연산이 시간 스텝 순서대로 순차 계산되어야 하는 것과 달리, 선택적 스캔은 병렬 스캔 알고리즘으로 계산할 수 있으며, 연산량은 시퀀스 길이 $L$에 선형적으로 비례한다. 각 방향의 S6는 하나의 가중치 집합을 그 방향의 모든 행 또는 열에 공통으로 적용하므로, SS2D 모듈의 파라미터 수는 0.12M이고 모델 전체의 파라미터 수는 31.2M이다. 표 1의 SS2D (controlled), 즉 통제 SS2D 모델은 게이팅이 없는 단일 SS2D 블록을 시퀀스 모듈로 사용하며, bi-GRU 모델보다 적은 파라미터를 갖는다. 통제를 완화한 변형으로, stride-3 다운샘플링으로 축소한 격자에서 16비트 부동소수점(fp16) 정밀도로 스캔을 수행하고 Mamba의 게이팅을 포함한 잔차 SS2D 블록 3개($d_{\mathrm{inner}}$=256, $d_{\mathrm{state}}$=32)를 쌓고 64채널로 출력하는 강화 SS2D 모델(표 1의 SS2D (enhanced), 파라미터 34.2M)도 학습하였다. 여기서 게이팅은 블록의 입력 투영을 두 갈래로 나누어 한 갈래는 네 방향 스캔에 입력하고 다른 갈래 $z$는 게이트로 사용하여, 네 방향 스캔을 결합한 출력 $s$에 $\mathrm{SiLU}(z)$를 원소별로 곱하는 연산 $s \odot \mathrm{SiLU}(z)$를 의미한다. 강화 SS2D 모델은 통제 모델과 달리 80 epoch 동안 학습되었으며, 구조, 학습 기간 및 파라미터화가 함께 변경되었다. 각 변경 요소에 대한 절제 실험(ablation study)은 수행하지 않았으므로, 개별 요소의 기여는 구분할 수 없다.

![Fig. 2](../figs/conf_fig2_modules.png)

Fig. 2. Two configurations of the sequence module $f_\theta$: (a) original bi-GRU, (b) SS2D

### 2.3 학습·평가 프로토콜

학습에는 fastMRI brain multicoil 데이터셋[10]의 공식 학습 분할에서 확보한 볼륨 중 손상된 2개를 제외한 4,108개 볼륨(슬라이스 65,028개)을, 검증에는 공식 검증 분할의 첫 번째 배포 묶음(batch 0)에 포함된 464개 볼륨(슬라이스 7,334개)을 사용하였다. 데이터에는 축상면 T1 강조 영상(AXT1), 조영 후 T1 강조 영상(AXT1POST), 조영 전 T1 강조 영상(AXT1PRE), T2 강조 영상(AXT2), FLAIR 영상(AXFLAIR)이 혼합되어 있으며, 검증 집합 볼륨의 58%가 AXT2이다. 전처리는 완전 샘플링된 k-space에 역 푸리에 변환을 적용하고 384×384 크기로 크롭 및 패딩한 뒤 다시 푸리에 변환하는 절차를 따른다. 슬라이스별 강도 정규화 대신 zero-filled 영상과 참조 영상에는 10⁶, k-space에는 10⁴의 고정 배율을 곱하였다. 각 볼륨에서 처음 16개의 코일을 사용하였고, 코일 수가 16개 미만인 볼륨에서는 나머지 코일 채널을 0으로 채웠다. 학습 시에는 마스크의 오프셋을 각 샘플에 대해 무작위로 설정하였고, 검증 시에는 고정하였다. 데이터 증강으로는 영상을 상하 방향과 좌우 방향으로 각각 무작위로 반전한 뒤 k-space를 다시 계산하는 방식을 사용하였다. 배경 영역이 손실과 평가 결과에 미치는 영향을 줄이기 위해, 참조 영상 $x^{*}$의 강도를 기준으로 슬라이스마다 전경 마스크 $m$을 생성하였다. 이 마스크는 Otsu 임곗값의 0.4배를 초과하는 화소 집합에서 가장 큰 연결 성분을 취한 것이며, 이하 이를 brain mask라 한다. 손실(식 (3))과 평가 지표는 이 마스크 내부에서만 계산하였고, 같은 마스크를 두 모델과 표 1의 공개 모델 평가에 동일하게 적용하였다. 손실 계산에서는 원 ETER-Net 코드의 구현(11×11 가우시안 창)으로, 평가에서는 scikit-image 구현으로 structural similarity index(SSIM) 지도를 구하였으며, 평가 시 SSIM 계산의 동적 범위(data range)는 마스크 내부 $x^{*}$의 최댓값과 최솟값의 차이로 설정하였다. peak signal-to-noise ratio(PSNR)의 최대 신호 값(peak)은 마스크 내부 $x^{*}$의 최댓값으로, normalized mean squared error(nMSE)는 마스크 내부에서 구한 오차 제곱합과 $x^{*}$ 제곱합의 비로 계산하였다. 따라서 표 1의 수치는 전체 영상을 기준으로 계산한 fastMRI 공식 수치와 직접 비교할 수 없다.

$$ \mathcal{L} = \frac{\sum_{i} m_i \left| \hat{x}_i - x^{*}_i \right|}{\sum_{i} m_i} + \left( 1 - \mathrm{SSIM}_m(\hat{x}, x^{*}) \right) \qquad (3) $$

식 (3)에서 $\mathcal{L}$은 학습에 사용한 손실 함수이고, $i$는 영상의 화소 인덱스이며, $\sum_{i}$는 모든 화소에 대한 합을 나타낸다. $m_i \in \{0, 1\}$은 화소 $i$가 brain mask 내부이면 1, 외부이면 0인 마스크 값이고, $\hat{x}_i$와 $x^{*}_i$는 각각 재구성 영상 $\hat{x}$와 참조 영상 $x^{*}$의 화소 $i$에서의 값이다. 첫째 항은 마스크 내부 화소에 대한 평균 절대 오차(L1 손실)이고, 둘째 항의 $\mathrm{SSIM}_m(\hat{x}, x^{*})$는 두 영상 사이의 SSIM 지도를 brain mask $m$ 내부에서 평균한 값이며(아래첨자 $m$은 이 마스크를 뜻한다), 둘째 항은 이를 1에서 뺀 값이다. 최적화에는 Adam을 사용하였으며, 학습률은 2×10⁻⁴, weight decay는 3×10⁻⁵으로 설정하였다. 학습률은 cosine 스케줄로 감소시켰으며 최소 학습률은 1×10⁻⁶으로 설정하였다. gradient norm은 최대 1.0으로 클리핑하였고, 자동 혼합 정밀도(AMP)를 적용하였으며, 배치 크기는 8로 설정하였다. 각 모델은 TITAN RTX GPU 1대에서 독립적으로 1회씩 학습되었으며, 난수 시드는 고정하지 않았다. bi-GRU 모델과 통제 SS2D 모델은 50 epoch 동안, 강화 SS2D 모델은 80 epoch 동안 학습되었고, 검증은 2 epoch마다 수행하였다. 최적 체크포인트는 검증 집합에서 계산한 복합 선택 점수 $C = 0.5\,\mathrm{SSIM} + 0.3\,\min(\mathrm{PSNR}, 40)/40 + 0.2\,(1 - \min(\mathrm{nMSE}, 1))$이 최대인 시점으로 선정하였다. 이 점수에서 SSIM, PSNR(dB), nMSE는 모두 brain mask 내부에서 계산하여 검증 집합 전체에 대해 평균한 값이고, 0.5, 0.3, 0.2는 세 항의 가중치이다. PSNR은 40 dB를 상한으로 클리핑한 뒤 40으로 나누어 [0, 1] 범위로 정규화하였으며, 값이 낮을수록 좋은 nMSE(무차원 비율)는 1을 상한으로 클리핑한 뒤 1에서 빼어 값이 클수록 좋은 방향으로 변환하였다. 이 점수는 체크포인트 선택에만 사용하였고, 보고하는 성능은 모두 선택 점수로 합치지 않은 SSIM, PSNR, nMSE 각각의 값이다. 평가 지표로는 SSIM(주 평가 지표), PSNR 및 nMSE(%)를 사용하였다. 각 지표는 슬라이스별로 계산한 뒤 볼륨 내에서 평균하였고, 최종 결과는 볼륨별 평균값을 전체 검증 집합에 대해 평균한 값과 표준편차(평균±표준편차)로 보고한다. 본 실험은 두 모델을 같은 슬라이스에 대해 비교하는 대응(paired) 설계이므로 슬라이스 단위와 볼륨 단위의 우위 비율을 함께 보고하고, 유의성은 볼륨 단위 양측 Wilcoxon signed-rank 검정과 볼륨 단위 군집(cluster) 부트스트랩(2,000회 재표집) 95% 신뢰구간(CI)으로 평가하였다.

## Ⅲ. 실험 결과

### 3.1 정량 비교

표 1은 학습·평가 프로토콜에 기술한 기준으로 선정한 최적 체크포인트(bi-GRU 50번째 epoch, 통제 SS2D 48번째 epoch, 강화 SS2D 78번째 epoch)를 검증 집합 전체에 적용한 결과이다. 체크포인트 선택과 최종 평가에는 동일한 검증 집합을 사용하였으며, 별도의 테스트 집합은 구성하지 않았다. 통제 SS2D 모델은 bi-GRU 모델의 약 1/21에 해당하는 파라미터를 사용하면서도 더 높은 SSIM과 PSNR, 더 낮은 nMSE를 기록하였다. 슬라이스 단위 대응 비교에서 SS2D 모델이 우위인 비율은 SSIM 78.2%(95% CI 76.8~79.7), PSNR 73.8%, nMSE 73.8%이고, 볼륨 단위로는 각각 94.8%, 89.9%, 90.1%였다. 볼륨 단위 양측 Wilcoxon signed-rank 검정에서는 세 지표 모두에서 두 모델 간 차이가 유의확률 $p<0.001$로 유의하였고, 슬라이스별 SSIM 차이 $\Delta \mathrm{SSIM}$(SS2D 모델의 SSIM에서 bi-GRU 모델의 SSIM을 뺀 값)의 평균은 +0.0014(95% CI +0.0013~+0.0015)였다. PSNR과 nMSE는 슬라이스 단위에서 평균 제곱 오차(MSE)의 단조 함수이므로 슬라이스 우위 비율이 서로 같다. 학습 로그의 25회 검증 시점 전부에서 SS2D 모델의 검증 SSIM은 bi-GRU 모델의 값 이상이었고(동률 1회), 검증 PSNR은 25회 모두에서 bi-GRU 모델의 값을 상회하였다. 또한 대조도(contrast)별 하위 집단과 평가 지표의 모든 조합에서 SS2D 모델이 우수한 슬라이스의 비율은 68.7% 이상이었다. 다만 epoch당 학습 시간은 cuDNN으로 최적화된 bi-GRU 모델(2.41 h)이 SS2D 모델(3.07 h)보다 짧았다. 이는 사용한 선택적 스캔 커널이 cuDNN의 GRU 구현만큼 최적화되어 있지 않기 때문으로 추정되며, 본 연구에서 SS2D 모델의 효율 이점은 연산 시간이 아니라 파라미터 수에 국한된다. 강화 SS2D 모델이 통제 모델보다 우수한 값을 보인 슬라이스의 비율은 지표별로 54~56%였다. 별도로 수행한 볼륨 단위 양측 Wilcoxon signed-rank 검정에서는 세 지표 모두에서 두 모델 간 차이가 유의하였다($p<0.001$). 반면 볼륨 단위 평균 nMSE는 강화 모델 0.439%, 통제 모델 0.438%로 유사하였으며, 강화 모델의 값이 소폭 높았다. 50 epoch 시점의 검증 SSIM은 통제 모델 0.9138, 강화 모델 0.9130으로 통제 모델의 값이 더 높았다. 학습 중 검증과 최종 평가에는 동일한 SSIM 구현을 사용하였지만, 표 1의 값은 최적 체크포인트의 결과를 볼륨 단위로 집계한 것이므로 이 시점의 값과는 다르다. 따라서 최종 성능 차이에는 학습 기간 연장이 기여하였을 가능성이 있으나, 구조도 함께 변경되었으므로 각 요인의 영향을 분리할 수 없다. 표 1 하단의 U-Net†[10], E2E-VarNet†[1], PromptMR+[11]는 표 1 주석의 조건으로 추론한 참고 결과이며, 우열 비교 대상이 아니다.

Table 1. Volume-level results (mean ± SD) on the fastMRI brain multi-coil validation subset (464 volumes/7,334 slices, R=4, brain-masked)

| Method | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---|---|---|---|
| Zero-filled | – | 0.7523±0.0410 | 24.76±2.11 | 3.935±2.166 |
| bi-GRU (original) | 668 | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 |
| SS2D (controlled) | 31 | <u>0.9141±0.0365</u> | <u>33.91±1.90</u> | **0.438±0.283** |
| SS2D (enhanced) | 34 | **0.9146±0.0361** | **33.92±1.90** | <u>0.439±0.304</u> |
| U-Net† | 496 | 0.8971±0.0366 | 30.95±2.29 | 0.973±0.796 |
| E2E-VarNet† | 30 | 0.9181±0.0386 | 32.78±3.21 | 1.133±1.291 |
| PromptMR+ | 93 | 0.9417±0.0349 | 36.12±4.02 | 0.526±0.841 |

Note. 상단 4행은 본 연구 프로토콜의 결과이며, 굵은 글씨와 밑줄은 학습 모델 3개 중 가장 좋은 값과 두 번째로 좋은 값을 각각 나타낸다. Zero-filled는 언더샘플링된 k-space에 코일별 역 푸리에 변환을 적용한 코일 영상의 root-sum-of-squares(RSS) 영상이다. 하단 3행은 공개 가중치를 본 연구 프로토콜(384×384 전처리, 16코일, R=4)에 적용하여 CPU에서 32비트 부동소수점(fp32) 정밀도로 추론한 참고 결과이다. 공개 가중치는 원래의 코일 구성으로 학습되어 본 프로토콜과의 도메인 차이(domain shift)가 존재하고, brain mask 내부의 슬라이스별 최소제곱 강도 배율 보정을 공개 모델에만 적용하였으므로, 하단 3행은 직접적인 성능 비교 대상에서 제외하였다. †는 본 검증 집합을 포함한 학습·검증 통합 분할(train+val)로 학습된 fastMRI leaderboard 공개 가중치를 나타낸다. PromptMR+의 공개 가중치는 학습 분할로만 학습되었으나, 이 모델은 12단 캐스케이드 펼침(unrolled) 구조로서 인접한 5개 슬라이스를 입력으로 사용한다.

### 3.2 정성 비교

그림 3은 시각 비교용으로 사전에 지정한 검증 슬라이스 12개 중 1개(AXT2)의 결과이다. 상단 행은 참조 영상, zero-filled 영상, bi-GRU 재구성 영상, SS2D 재구성 영상을 순서대로 나타내며, 참조 영상을 제외한 각 패널 안의 수치는 해당 슬라이스의 PSNR과 SSIM이다. 중간 행은 brain mask 내부에서 참조 영상의 최댓값으로 정규화한 절대 오차 지도이고, 하단 행은 표시 강도를 4배로 증폭하여 배경의 저강도 아티팩트를 드러낸 영상(×4 gain)이다. 오차 지도와 패널 수치에서 SS2D 재구성 영상의 오차가 소폭 작았으며, gain 영상에서는 두개골 바깥 배경의 주기적 ringing 아티팩트가 bi-GRU 재구성 영상에서 두드러지는 반면 SS2D 재구성 영상에서는 약하게 나타났다. brain mask 밖의 이러한 아티팩트는 표 1의 지표에 반영되지 않으며, 본 연구에서는 이를 별도로 정량화하지 않았다.

![Fig. 3](../figs/fig3_qualitative_col.png)

Fig. 3. Qualitative comparison on a validation slice (AXT2): reconstructions (top), brain-masked error maps (middle), ×4 gain images (bottom)

## Ⅳ. 결론

본 연구는 ETER-Net의 도메인 변환 모듈을 bi-GRU에서 SS2D로 교체하고, 나머지 구조와 학습 설정을 동일하게 유지하여 재구성 성능을 비교하였다. fastMRI brain 검증 집합의 R=4 조건에서 통제 SS2D 모델은 명시적 DC 블록 없이 bi-GRU 모델의 약 1/21에 해당하는 파라미터를 사용하면서 SSIM, PSNR 및 nMSE에서 개선된 결과를 보였다. 제시한 정성 예시에서도 두개골 바깥 배경의 ringing 아티팩트가 SS2D 재구성 영상에서 더 약하게 관찰되었다. 다만 이러한 결과는 본 연구의 실험 조건에 한정되며, SSM과 순환신경망 전반의 우열로 일반화할 수는 없다. 두 모델은 순환 메커니즘과 파라미터화가 함께 다르므로, 본 연구에서는 각 요인의 기여를 분리하지 못하였다. 또한 본 연구에서는 도메인 변환 모듈을 제외한 U-Net 단독 모델과의 비교를 수행하지 않았으므로 해당 모듈 자체의 기여는 평가하지 못하였다. 난수 시드를 고정하지 않은 모델별 단일 학습, AXT2 비중이 높은 단일 데이터셋의 사용, 단일 가속화 계수에서의 평가도 본 연구의 한계이다. 현재 여러 난수 시드를 사용한 반복 실험이 진행 중이며, 향후 연구에서는 도메인 변환 모듈을 Transformer 및 모든 행·열 위치에 같은 가중치를 적용하는(위치 간 가중치 공유) pixel-GRU로 치환한 모델과의 비교와 여러 가속화 계수에서의 일반화 성능 평가를 수행할 계획이다.

## 참고문헌

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
