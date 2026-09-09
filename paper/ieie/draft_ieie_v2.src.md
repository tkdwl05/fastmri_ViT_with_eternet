%% IEIE 투고용 통합 초안 v2 (2026-09-09) — 학술지 양식(서면 심사용, 4쪽·blind)과 학술대회 양식(프로시딩 게재용, 1~5쪽·저자 포함) 이 **같은 소스**를 쓴다.
%% 교수님 지시(09-09): 두 파일은 내용이 동일해야 하고 서면 심사용은 저자·소속만 삭제 / 캡션은 국문만·간결하게(세부는 본문) / 초록·제목은 교수님 수정본 / 참고문헌 축소.
%% 빌드(저장소 루트, CPU):
%%   CUDA_VISIBLE_DEVICES="" python paper/ieie/build_ieie_docx.py      --src paper/ieie/draft_ieie_v2.src.md   → draft_ieie_v2.{md,docx}       (서면 심사용, 저자 없음 — blind 점검 내장)
%%   CUDA_VISIBLE_DEVICES="" python paper/ieie/build_ieie_conf_docx.py --src paper/ieie/draft_ieie_v2.src.md   → draft_ieie_v2_conf.{md,docx}  (프로시딩 게재용, @author_* 사용)
%% 규칙: 표준 지표만(SSIM 주지표 + PSNR/nMSE %; composite 인용 금지) · 표 수치 = paper/tables/ieie_table1_block.md·ieie_table_ref_block.md(볼륨 단위, make_tables.py) · 기준점 = 교수님 원본 bi-GRU · 인용 [@bibkey] · 미확정 [TBD]
%% 저자·소속·e-mail 은 프로시딩 게재용에만 들어간다(학술지 빌더는 무시). "***" 자리표시자는 교수님 확인 후 기입.
@author_ko: ***, ***
@affil_ko: *** 소속
@email: e-mail : ***
@author_en: *** and ***
@affil_en: *** University

@title_ko: 가속 MRI 재구성 네트워크 ETER-Net에서 순환신경망을 선택적 상태공간모델로 치환한 통제 연구
@title_en: A Controlled Study of Replacing RNN with a Selective SSM in ETER-Net for Accelerated Direct MRI
@keywords: MRI reconstruction, ETER-Net, recurrent neural network, selective state-space model, controlled comparison

@abstract_ko:
ETER-Net은 언더샘플링된 k-space를 양방향 순환신경망(bi-RNN)을 활용해 영상 도메인으로 직접 변환하는 MRI 재구성 방법이다. 본 논문은 이 골격에서 도메인 변환 모듈을 단일 변수로 통제하여, 기존 설계의 bi-GRU를 2차원 선택적 상태공간모델(SS2D)로 치환했을 때의 효과를 검증한다. fastMRI brain multicoil 데이터셋(384×384, R=4)에서 두 모델은 시퀀스 변환 모듈을 제외한 모든 구성요소(데이터·마스크·U-Net 구조·손실·최적화 설정)가 동일하며(가중치 공유 없이 각각 독립 학습), 원 논문에 충실하게 명시적 데이터 일관성(DC) 블록 없이 학습을 수행하였다. 검증 슬라이스 평가 결과, SS2D 치환 모델은 기존 대비 21배 적은 파라미터(31M vs. 668M)로 볼륨 단위 평균 brain-masked SSIM 0.9141(GRU 0.9127), PSNR 33.91 dB(33.78 dB)를 기록하여 주요 평가 지표(SSIM·PSNR·nMSE) 전체에서 기존 설계를 상회했으며, 슬라이스 단위로는 SSIM 78.2%·PSNR 73.8%·nMSE 73.8%, 볼륨 단위로는 94.8%·89.9%·90.1%에서 우위였다(볼륨 단위 Wilcoxon signed-rank, p<0.001). 정성적 평가에서는 bi-GRU 재구성 영상의 두개골 외부 배경 영역에서 관찰되는 주기적 ringing 아티팩트가 SS2D 적용 모델에서는 억제됨을 확인하였다. 본 연구는 직접 도메인 변환형 재구성 구조에서 SS2D로의 치환이 DC 블록 없이도 경량화된 파라미터로 일관된 품질 향상을 달성함을 입증한다.

@abstract_en:
ETER-Net is an MR reconstruction method that directly transforms undersampled k-space into the image domain using a bidirectional recurrent neural network (bi-RNN). This paper evaluates the effect of replacing the original bi-GRU with a 2D selective state-space model (SS2D). On the fastMRI brain multicoil dataset (384×384, R=4), both models are identical in every component except the sequence model (same data, mask, U-Net architecture, loss, and optimization; trained independently without weight sharing) and are trained without an explicit data-consistency (DC) block, adhering strictly to the original design. Evaluation on validation slices shows that the SS2D replacement model, with 21× fewer parameters (31M vs. 668M), achieves a volume-averaged brain-masked SSIM of 0.9141 (vs. 0.9127) and PSNR of 33.91 dB (vs. 33.78 dB), outperforming the original bi-GRU design across all standard metrics (SSIM, PSNR, nMSE) and winning on 78.2% (SSIM), 73.8% (PSNR), and 73.8% (nMSE) of slices and on 94.8%, 89.9%, and 90.1% of volumes, respectively (volume-level Wilcoxon signed-rank test, p<0.001). Qualitatively, periodic ringing artifacts outside the skull in the bi-GRU reconstructions are significantly suppressed in the SS2D reconstructions. These results indicate that substituting bi-GRU with SS2D in direct domain-transformation architectures achieves consistent quality gains without DC blocks, even with substantially fewer parameters.

@body:

# 서론

MRI는 k-space를 순차 수집하므로 촬영이 느리며, 언더샘플링 후의 딥러닝 재구성은 물리 모델의 반복 최적화를 펼치는 unrolled 계열[@hammernik2018learning; @sriram2020endtoend]과 신경망이 k-space를 영상으로 직접 변환하는 도메인 변환 계열[@zhu2018automap]로 나뉜다. ETER-Net[@oh2021eternet]은 후자로, 양방향 GRU(bi-GRU)가 k-space를 행·열 방향으로 읽어 영상 도메인 특징으로 변환하고 U-Net이 aliasing을 제거하며, 명시적 데이터 일관성(DC) 블록은 없다. 이후 non-Cartesian 궤적[@oh2022radial]과 ViT 인코더 결합[@oh2025vitbirnn]으로 확장되었으나 도메인 변환의 핵심인 bi-RNN은 유지되었다. 그러나 bi-GRU는 순차 처리라 병렬화가 어렵고 flatten-reshape 구조 탓에 파라미터가 수억 개에 이른다. 선택적 상태공간모델 Mamba[@gu2023mamba]는 입력 의존 순환을 병렬 스캔으로 선형 시간에 계산하고, 2차원 확장 SS2D[@liu2024vmamba]는 4방향 스캔으로 전역 수용영역을 얻는다. Mamba 기반 MRI 재구성[@korkmaz2025mambarecon; @huang2024mambamir]은 새 구조 전체를 제안하므로, 기존 골격에서 순환신경망만 SSM으로 바꾼 효과를 분리한 비교는 없었다. 본 논문은 ETER-Net에서 도메인 변환 시퀀스 모델만 bi-GRU에서 SS2D로 치환하고 나머지를 모두 고정한 통제 비교를 보고한다.

# 방법

## 통제 파이프라인

그림 1은 두 팔이 공유하는 파이프라인이다. 완전 샘플링 16코일 k-space y_c에서 정답 x*(RSS 영상, 384×384 crop/pad)를 만들고, R=4 equispaced 마스크 M(ACS 8%)을 곱한 ỹ_c=M⊙y_c를 실수·허수로 분리한 (32, 384, 384) 텐서가 입력이다. 시퀀스 모델 f_θ가 k-space를 영상 도메인 특징(20채널)으로 직접 변환하고, 같은 ỹ_c를 코일별 역 FFT한 zero-filled 코일 영상(32채널)과 채널 결합해 후처리 U-Net g_φ(dual-frame skip, depth 5, 31.1M)가 magnitude 영상 x̂를 출력한다.

$$ \hat{x} = g_{\phi}\left( \mathrm{concat}\left( f_{\theta}(\{\tilde{y}_c\}), \mathcal{F}^{-1}\{\tilde{y}_c\} \right) \right) $$

점선 상자의 f_θ만이 두 팔 사이의 유일한 변수이며 데이터·마스크·손실·최적화·U-Net 구조는 동일하다. 두 팔은 가중치를 공유하지 않고 같은 레시피로 각각 처음부터 독립 학습하며, 원 논문에 따라 DC 블록은 두지 않는다.

@figure: paper/figs/conf_fig1_pipeline.png | col | 1.0
@cap_ko: 두 팔이 공유하는 ETER-Net 통제 파이프라인(점선 상자의 f_θ만 변수)

## 시퀀스 모델의 두 팔

그림 2(a)의 bi-GRU 팔은 원본 ETER-Net 그대로 k-space를 384개 행의 시퀀스(스텝당 12,288차원)로 펼쳐 양방향 GRU를 통과시킨 뒤, 전치해 열 방향으로 한 번 더 통과시킨다. 두 GRU의 입력–은닉 행렬이 파라미터의 대부분이라 GRU 스택만 637.1M, 팔 전체 668.2M이며(원 코드의 hidden 배수 10 기준; canonical 12에서는 880.5M), 재귀는 스텝 순서대로만 계산된다. 그림 2(b)의 SS2D 팔은 픽셀별 LN·Linear(32→128)·SiLU와 depthwise conv 뒤에 선택적 스캔(S6)을 네 방향(각 행 →/←, 각 열 ↓/↑, L=384)으로 적용하고 네 출력을 채널 결합해 LN·Linear와 1×1 conv로 GRU와 같은 20채널에 정합한다. S6는 이산화된 상태공간 시스템

$$ h_t = \exp(\Delta_t A)\, h_{t-1} + \Delta_t B_t x_t, \quad y_t = C_t h_t + D x_t $$

의 (Δ_t, B_t, C_t)를 입력 x_t에서 생성하는 선택적 순환으로(d_inner 128, d_state 16), 방향별 한 조의 가중치를 그 방향의 모든 행(열)이 공유하므로 SSM 스택은 0.12M(팔 전체 31.2M)에 그치고 스캔은 병렬 O(L)로 계산된다. 통제판은 게이팅 없는 단일 블록으로 용량을 GRU 이하로 억제한 최소 구성이다. 통제를 해제한 상한으로, Mamba 게이팅 y·SiLU(z)를 복원한 잔차 SS2D 블록 3개(d_inner 256, d_state 32, 출력 64채널)를 stride-3 다운샘플·fp16 스캔 위에 쌓은 강화 SS2D(34.2M)도 80 epoch 학습하였다.

@figure: paper/figs/conf_fig2_arms.png | col | 1.0
@cap_ko: 시퀀스 모델 f_θ의 두 팔: (a) 원 bi-GRU, (b) SS2D

## 학습·평가 프로토콜

fastMRI brain multicoil[@zbontar2018fastmri] 확보 서브셋(혼합 contrast: AXT1·AXT1POST·AXT1PRE·AXT2·AXFLAIR)을 공식 구획대로 사용하였다(train 4,108 파일/65,028 슬라이스, val 464 볼륨/7,334 슬라이스). 전처리는 full k-space → 역 FFT → 384×384 crop/pad → 재-FFT의 retrospective 프로토콜이고, 코일은 앞 16개를 사용하며, train 마스크 offset은 매 샘플 랜덤(val 고정), 증강은 flip 후 FFT 재계산이다. 배경이 값을 부풀리지 않도록 손실과 지표는 brain mask m(Otsu×0.4 + 최대 연결성분) 내부에서 계산하며, 손실은 다음과 같다.

$$ \mathcal{L} = \frac{\sum_{i} m_i \left| \hat{x}_i - x^{*}_i \right|}{\sum_{i} m_i} + \left( 1 - \mathrm{SSIM}_m(\hat{x}, x^{*}) \right) $$

Adam(2×10⁻⁴, weight decay 3×10⁻⁵)·cosine 스케줄·AMP·batch 8·50 epoch(검증 매 2 epoch)으로 TITAN RTX 1장에서 학습하였다. 지표는 SSIM(주지표)·PSNR·nMSE(%)를 슬라이스 단위로 계산해 fastMRI 관례대로 볼륨 단위 평균±표준편차로 보고하고, paired 설계이므로 우위 비율(슬라이스·볼륨)과 볼륨 단위 Wilcoxon signed-rank 검정, 볼륨 클러스터 부트스트랩(2,000회) 95% 신뢰구간(CI)으로 유의성을 평가하였다.

# 실험 결과

## 정량 비교

표 1은 best checkpoint 결과다. SS2D 치환 모델은 21배 적은 파라미터로 세 지표 전부에서 원 bi-GRU 설계를 상회했다. 슬라이스 단위 paired 비교에서 SS2D 우위 비율은 SSIM 78.2%(95% CI 76.8~79.7), PSNR 73.8%, nMSE 73.8%이고 볼륨 단위로는 94.8%·89.9%·90.1%였다(모든 지표 p<0.001; ΔSSIM 평균 +0.0014, 95% CI +0.0013~+0.0015). 우위는 25회 검증 시점 전부(동률 1회)와 5개 contrast 서브그룹 전부(우위 슬라이스 ≥68.7%)에서 유지되었다. 다만 epoch당 학습 시간은 cuDNN GRU가 짧아(2.41 h 대 3.07 h) 효율 이점은 파라미터 수에 있다. 강화 SS2D는 통제판 대비 세 지표 모두 근소하게 개선되었으나(우위 슬라이스 54~56%), 이 이득은 동일 50 epoch 시점(SSIM 0.9130)이 아닌 80 epoch 연장 구간의 것이다.

@table: col | 1060,520,1070,900,985
@cap_ko: 검증 집합 전체(464 볼륨/7,334 슬라이스, R=4, brain-masked)의 볼륨 단위 결과(평균±표준편차)
| Method | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---|---|---|---|
| Zero-filled | – | 0.7523±0.0410 | 24.76±2.11 | 3.935±2.166 |
| bi-GRU (original) | 668 | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 |
| SS2D (controlled) | 31 | __0.9141±0.0365__ | __33.91±1.90__ | **0.438±0.283** |
| Enhanced SS2D | 34 | **0.9146±0.0361** | **33.92±1.90** | __0.439±0.304__ |
| U-Net† | 496 | 0.8971±0.0366 | 30.95±2.29 | 0.973±0.796 |
| E2E-VarNet† | 30 | 0.9181±0.0386 | 32.78±3.21 | 1.133±1.291 |
| PromptMR+ | 93 | 0.9417±0.0349 | 36.12±4.02 | 0.526±0.841 |
@note: 위 네 행 = 본 연구(최고값 굵게·차선 밑줄); 아래 세 행 = 같은 프로토콜로 추론한 공개 모델 참고선. †: leaderboard 가중치(train+val 학습). PromptMR+: train 학습, 12-cascade unrolled, 인접 5슬라이스 입력.
@end

## 정성 비교와 참고선

그림 3은 정본 검증 슬라이스의 재구성, brain-masked 오차 지도, ×4 gain 영상이다. 오차 지도에서 SS2D의 잔차가 전반적으로 작고, gain 영상에서는 bi-GRU 재구성에만 두개골 바깥 배경에 주기적 ringing 아티팩트가 남는다. 이 배경 아티팩트는 brain-masked 지표에 벌점을 주지 않으므로 표 1의 우위는 bi-GRU에 유리한 보수적 하한이다. 표 1 하단의 공개 모델은 참고선이다. U-Net†·E2E-VarNet†[@sriram2020endtoend]는 train+val 합본으로 학습된 leaderboard 가중치라 우열 비교 대상이 아니고, PromptMR+[@xin2024rethinking]는 train 구획만 학습했으나 12-cascade unrolled 구조에 인접 5슬라이스를 입력받아 계열이 다르다.

@figure: paper/figs/fig3_qualitative_col.png | col | 1.0
@cap_ko: 검증 슬라이스(AXT2) 정성 비교: 재구성(위), brain-masked 오차(가운데), ×4 gain(아래)

# 결론

ETER-Net의 도메인 변환 자리에서 bi-GRU를 SS2D로 치환하는 것만으로 DC 블록 없이, 21배 적은 파라미터로 표준 지표 전부와 대다수 슬라이스·볼륨에서 일관된 개선을 얻었고 배경 ringing 아티팩트도 억제되었다. 이 결론은 "SSM이 RNN보다 우월하다"는 일반론이 아니라 "SS2D 치환이 원 bi-GRU 설계보다 낫다"로 한정된다. 두 팔은 순환 메커니즘과 파라미터화가 함께 달라 각 요인의 기여를 분리하지 못했고, 단일 시드·단일 가속률(R=4)이라는 한계가 있다. 멀티시드 재현, Transformer·pixel-GRU 팔 추가, 가속률 일반화 학습을 진행 중이다.

# REFERENCES
