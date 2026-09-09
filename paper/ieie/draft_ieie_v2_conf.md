# 가속 MRI 재구성 네트워크 ETER-Net에서 순환신경망을 선택적 상태공간모델로 치환한 통제 연구

***, *** (*** 소속, e-mail : ***)

**A Controlled Study of Replacing RNN with a Selective SSM in ETER-Net for Accelerated Direct MRI**

*** and *** (*** University)

*(대한전자공학회 학술대회 2쪽 양식 — 작성예시 example_conference_2page.docx 기준; 저자·소속은 "***" 자리표시자)*

## Abstract

ETER-Net is an MR reconstruction method that directly transforms undersampled k-space into the image domain using a bidirectional recurrent neural network (bi-RNN). This paper evaluates the effect of replacing the original bi-GRU with a 2D selective state-space model (SS2D). On the fastMRI brain multicoil dataset (384×384, R=4), both models are identical in every component except the sequence model (same data, mask, U-Net architecture, loss, and optimization; trained independently without weight sharing) and are trained without an explicit data-consistency (DC) block, adhering strictly to the original design. Evaluation on validation slices shows that the SS2D replacement model, with 21× fewer parameters (31M vs. 668M), achieves a volume-averaged brain-masked SSIM of 0.9141 (vs. 0.9127) and PSNR of 33.91 dB (vs. 33.78 dB), outperforming the original bi-GRU design across all standard metrics (SSIM, PSNR, nMSE) and winning on 78.2% (SSIM), 73.8% (PSNR), and 73.8% (nMSE) of slices and on 94.8%, 89.9%, and 90.1% of volumes, respectively (volume-level Wilcoxon signed-rank test, p<0.001). Qualitatively, periodic ringing artifacts outside the skull in the bi-GRU reconstructions are significantly suppressed in the SS2D reconstructions. These results indicate that substituting bi-GRU with SS2D in direct domain-transformation architectures achieves consistent quality gains without DC blocks, even with substantially fewer parameters.

---

## Ⅰ. 서론

MRI는 k-space를 순차 수집하므로 촬영이 느리며, 언더샘플링 후의 딥러닝 재구성은 물리 모델의 반복 최적화를 펼치는 unrolled 계열[1, 2]과 신경망이 k-space를 영상으로 직접 변환하는 도메인 변환 계열[3]로 나뉜다. ETER-Net[4]은 후자로, 양방향 GRU(bi-GRU)가 k-space를 행·열 방향으로 읽어 영상 도메인 특징으로 변환하고 U-Net이 aliasing을 제거하며, 명시적 데이터 일관성(DC) 블록은 없다. 이후 non-Cartesian 궤적[5]과 ViT 인코더 결합[6]으로 확장되었으나 도메인 변환의 핵심인 bi-RNN은 유지되었다. 그러나 bi-GRU는 순차 처리라 병렬화가 어렵고 flatten-reshape 구조 탓에 파라미터가 수억 개에 이른다. 선택적 상태공간모델 Mamba[7]는 입력 의존 순환을 병렬 스캔으로 선형 시간에 계산하고, 2차원 확장 SS2D[8]는 4방향 스캔으로 전역 수용영역을 얻는다. Mamba 기반 MRI 재구성[9, 10]은 새 구조 전체를 제안하므로, 기존 골격에서 순환신경망만 SSM으로 바꾼 효과를 분리한 비교는 없었다. 본 논문은 ETER-Net에서 도메인 변환 시퀀스 모델만 bi-GRU에서 SS2D로 치환하고 나머지를 모두 고정한 통제 비교를 보고한다.

## Ⅱ. 방법

### 2.1 통제 파이프라인

그림 1은 두 팔이 공유하는 파이프라인이다. 완전 샘플링 16코일 k-space y_c에서 정답 x*(RSS 영상, 384×384 crop/pad)를 만들고, R=4 equispaced 마스크 M(ACS 8%)을 곱한 ỹ_c=M⊙y_c를 실수·허수로 분리한 (32, 384, 384) 텐서가 입력이다. 시퀀스 모델 f_θ가 k-space를 영상 도메인 특징(20채널)으로 직접 변환하고, 같은 ỹ_c를 코일별 역 FFT한 zero-filled 코일 영상(32채널)과 채널 결합해 후처리 U-Net g_φ(dual-frame skip, depth 5, 31.1M)가 magnitude 영상 x̂를 출력한다.

$$ \hat{x} = g_{\phi}\left( \mathrm{concat}\left( f_{\theta}(\{\tilde{y}_c\}), \mathcal{F}^{-1}\{\tilde{y}_c\} \right) \right) \qquad (1) $$

점선 상자의 f_θ만이 두 팔 사이의 유일한 변수이며 데이터·마스크·손실·최적화·U-Net 구조는 동일하다. 두 팔은 가중치를 공유하지 않고 같은 레시피로 각각 처음부터 독립 학습하며, 원 논문에 따라 DC 블록은 두지 않는다.

![그림 1](../figs/conf_fig1_pipeline.png)

그림 1. 두 팔이 공유하는 ETER-Net 통제 파이프라인(점선 상자의 f_θ만 변수)

### 2.2 시퀀스 모델의 두 팔

그림 2(a)의 bi-GRU 팔은 원본 ETER-Net 그대로 k-space를 384개 행의 시퀀스(스텝당 12,288차원)로 펼쳐 양방향 GRU를 통과시킨 뒤, 전치해 열 방향으로 한 번 더 통과시킨다. 두 GRU의 입력–은닉 행렬이 파라미터의 대부분이라 GRU 스택만 637.1M, 팔 전체 668.2M이며(원 코드의 hidden 배수 10 기준; canonical 12에서는 880.5M), 재귀는 스텝 순서대로만 계산된다. 그림 2(b)의 SS2D 팔은 픽셀별 LN·Linear(32→128)·SiLU와 depthwise conv 뒤에 선택적 스캔(S6)을 네 방향(각 행 →/←, 각 열 ↓/↑, L=384)으로 적용하고 네 출력을 채널 결합해 LN·Linear와 1×1 conv로 GRU와 같은 20채널에 정합한다. S6는 이산화된 상태공간 시스템

$$ h_t = \exp(\Delta_t A)\, h_{t-1} + \Delta_t B_t x_t, \quad y_t = C_t h_t + D x_t \qquad (2) $$

의 (Δ_t, B_t, C_t)를 입력 x_t에서 생성하는 선택적 순환으로(d_inner 128, d_state 16), 방향별 한 조의 가중치를 그 방향의 모든 행(열)이 공유하므로 SSM 스택은 0.12M(팔 전체 31.2M)에 그치고 스캔은 병렬 O(L)로 계산된다. 통제판은 게이팅 없는 단일 블록으로 용량을 GRU 이하로 억제한 최소 구성이다. 통제를 해제한 상한으로, Mamba 게이팅 y·SiLU(z)를 복원한 잔차 SS2D 블록 3개(d_inner 256, d_state 32, 출력 64채널)를 stride-3 다운샘플·fp16 스캔 위에 쌓은 강화 SS2D(34.2M)도 80 epoch 학습하였다.

![그림 2](../figs/conf_fig2_arms.png)

그림 2. 시퀀스 모델 f_θ의 두 팔: (a) 원 bi-GRU, (b) SS2D

### 2.3 학습·평가 프로토콜

fastMRI brain multicoil[11] 확보 서브셋(혼합 contrast: AXT1·AXT1POST·AXT1PRE·AXT2·AXFLAIR)을 공식 구획대로 사용하였다(train 4,108 파일/65,028 슬라이스, val 464 볼륨/7,334 슬라이스). 전처리는 full k-space → 역 FFT → 384×384 crop/pad → 재-FFT의 retrospective 프로토콜이고, 코일은 앞 16개를 사용하며, train 마스크 offset은 매 샘플 랜덤(val 고정), 증강은 flip 후 FFT 재계산이다. 배경이 값을 부풀리지 않도록 손실과 지표는 brain mask m(Otsu×0.4 + 최대 연결성분) 내부에서 계산하며, 손실은 다음과 같다.

$$ \mathcal{L} = \frac{\sum_{i} m_i \left| \hat{x}_i - x^{*}_i \right|}{\sum_{i} m_i} + \left( 1 - \mathrm{SSIM}_m(\hat{x}, x^{*}) \right) \qquad (3) $$

Adam(2×10⁻⁴, weight decay 3×10⁻⁵)·cosine 스케줄·AMP·batch 8·50 epoch(검증 매 2 epoch)으로 TITAN RTX 1장에서 학습하였다. 지표는 SSIM(주지표)·PSNR·nMSE(%)를 슬라이스 단위로 계산해 fastMRI 관례대로 볼륨 단위 평균±표준편차로 보고하고, paired 설계이므로 우위 비율(슬라이스·볼륨)과 볼륨 단위 Wilcoxon signed-rank 검정, 볼륨 클러스터 부트스트랩(2,000회) 95% 신뢰구간(CI)으로 유의성을 평가하였다.

## Ⅲ. 실험 결과

### 3.1 정량 비교

표 1은 best checkpoint 결과다. SS2D 치환 모델은 21배 적은 파라미터로 세 지표 전부에서 원 bi-GRU 설계를 상회했다. 슬라이스 단위 paired 비교에서 SS2D 우위 비율은 SSIM 78.2%(95% CI 76.8~79.7), PSNR 73.8%, nMSE 73.8%이고 볼륨 단위로는 94.8%·89.9%·90.1%였다(모든 지표 p<0.001; ΔSSIM 평균 +0.0014, 95% CI +0.0013~+0.0015). 우위는 25회 검증 시점 전부(동률 1회)와 5개 contrast 서브그룹 전부(우위 슬라이스 ≥68.7%)에서 유지되었다. 다만 epoch당 학습 시간은 cuDNN GRU가 짧아(2.41 h 대 3.07 h) 효율 이점은 파라미터 수에 있다. 강화 SS2D는 통제판 대비 세 지표 모두 근소하게 개선되었으나(우위 슬라이스 54~56%), 이 이득은 동일 50 epoch 시점(SSIM 0.9130)이 아닌 80 epoch 연장 구간의 것이다.

표 1. 검증 집합 전체(464 볼륨/7,334 슬라이스, R=4, brain-masked)의 볼륨 단위 결과(평균±표준편차)

| Method | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---|---|---|---|
| Zero-filled | – | 0.7523±0.0410 | 24.76±2.11 | 3.935±2.166 |
| bi-GRU (original) | 668 | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 |
| SS2D (controlled) | 31 | <u>0.9141±0.0365</u> | <u>33.91±1.90</u> | **0.438±0.283** |
| Enhanced SS2D | 34 | **0.9146±0.0361** | **33.92±1.90** | <u>0.439±0.304</u> |
| U-Net† | 496 | 0.8971±0.0366 | 30.95±2.29 | 0.973±0.796 |
| E2E-VarNet† | 30 | 0.9181±0.0386 | 32.78±3.21 | 1.133±1.291 |
| PromptMR+ | 93 | 0.9417±0.0349 | 36.12±4.02 | 0.526±0.841 |

위 네 행 = 본 연구(최고값 굵게·차선 밑줄); 아래 세 행 = 같은 프로토콜로 추론한 공개 모델 참고선. †: leaderboard 가중치(train+val 학습). PromptMR+: train 학습, 12-cascade unrolled, 인접 5슬라이스 입력.

### 3.2 정성 비교와 참고선

그림 3은 정본 검증 슬라이스의 재구성, brain-masked 오차 지도, ×4 gain 영상이다. 오차 지도에서 SS2D의 잔차가 전반적으로 작고, gain 영상에서는 bi-GRU 재구성에만 두개골 바깥 배경에 주기적 ringing 아티팩트가 남는다. 이 배경 아티팩트는 brain-masked 지표에 벌점을 주지 않으므로 표 1의 우위는 bi-GRU에 유리한 보수적 하한이다. 표 1 하단의 공개 모델은 참고선이다. U-Net†·E2E-VarNet†[2]는 train+val 합본으로 학습된 leaderboard 가중치라 우열 비교 대상이 아니고, PromptMR+[12]는 train 구획만 학습했으나 12-cascade unrolled 구조에 인접 5슬라이스를 입력받아 계열이 다르다.

![그림 3](../figs/fig3_qualitative_col.png)

그림 3. 검증 슬라이스(AXT2) 정성 비교: 재구성(위), brain-masked 오차(가운데), ×4 gain(아래)

## Ⅳ. 결론

ETER-Net의 도메인 변환 자리에서 bi-GRU를 SS2D로 치환하는 것만으로 DC 블록 없이, 21배 적은 파라미터로 표준 지표 전부와 대다수 슬라이스·볼륨에서 일관된 개선을 얻었고 배경 ringing 아티팩트도 억제되었다. 이 결론은 "SSM이 RNN보다 우월하다"는 일반론이 아니라 "SS2D 치환이 원 bi-GRU 설계보다 낫다"로 한정된다. 두 팔은 순환 메커니즘과 파라미터화가 함께 달라 각 요인의 기여를 분리하지 못했고, 단일 시드·단일 가속률(R=4)이라는 한계가 있다. 멀티시드 재현, Transformer·pixel-GRU 팔 추가, 가속률 일반화 학습을 진행 중이다.

## 참고문헌

[1] K. Hammernik et al., "Learning a variational network for reconstruction of accelerated MRI data," *Magnetic Resonance in Medicine*, Vol. 79, no. 6, pp. 3055-3071, 2018.  
[2] A. Sriram et al., "End-to-End Variational Networks for Accelerated MRI Reconstruction," in *Medical Image Computing and Computer Assisted Intervention (MICCAI 2020)*, LNCS, Vol. 12262, pp. 64-73, 2020.  
[3] B. Zhu, J. Z. Liu, S. F. Cauley, B. R. Rosen, and M. S. Rosen, "Image reconstruction by domain-transform manifold learning," *Nature*, Vol. 555, pp. 487-492, 2018.  
[4] C. Oh, D. Kim, J.-Y. Chung, Y. Han, and H. Park, "A k-space-to-image reconstruction network for MRI using recurrent neural network," *Medical Physics*, Vol. 48, no. 1, pp. 193-203, 2021.  
[5] C. Oh, J.-Y. Chung, and Y. Han, "An End-to-End Recurrent Neural Network for Radial MR Image Reconstruction," *Sensors*, Vol. 22, no. 19, Art. no. 7277, 2022.  
[6] C. Oh, "A Hybrid Vision Transformer-BiRNN Architecture for Direct k-Space to Image Reconstruction in Accelerated MRI," *Journal of Imaging*, Vol. 12, no. 1, Art. no. 11, 2025.  
[7] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv preprint arXiv:2312.00752, 2023.  
[8] Y. Liu et al., "VMamba: Visual State Space Model," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.  
[9] Y. Korkmaz and V. M. Patel, "MambaRecon: MRI Reconstruction with Structured State Space Models," in *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2025.  
[10] J. Huang et al., "Enhancing global sensitivity and uncertainty quantification in medical image reconstruction with Monte Carlo arbitrary-masked Mamba," *Medical Image Analysis*, Vol. 99, Art. no. 103334, 2025.  
[11] J. Zbontar, F. Knoll, A. Sriram et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI," arXiv preprint arXiv:1811.08839, 2018.  
[12] B. Xin, M. Ye, L. Axel, and D. N. Metaxas, "Rethinking Deep Unrolled Model for Accelerated MRI Reconstruction," in *Computer Vision - ECCV 2024*, LNCS, Vol. 15133, pp. 164-181, 2024.  
