%% 대한전자공학회 학술대회 2쪽 논문 초안 v1 — 작성예시(example_conference_2page.docx) 양식용 소스
%% 빌드: CUDA_VISIBLE_DEVICES="" python paper/ieie/build_ieie_conf_docx.py  →  draft_ieie_conf_ko_v1.{md,docx}
%% 규칙: 표준 지표만(SSIM 주지표 + PSNR/nMSE %; L1 은 손실 항으로만 — 표 양식은 관례형, paper/tables/ieie_table1_block.md 첫 블록 = 볼륨 단위(09-04 결정); composite 인용 금지) · 기준점 = 교수님 원본 bi-GRU · 인용 [@bibkey] · 미확정 [TBD]
%% 저자·소속·e-mail 은 작성예시의 "***" 자리표시자 그대로(교수님 상의 후 기입)
@author_ko: ***, ***
@affil_ko: *** 소속
@email: e-mail : ***
@author_en: *** and ***
@affil_en: *** University

@title_ko: ETER-Net MRI 재구성에서 순환신경망을 선택적 상태공간모델로 치환한 단일 변수 통제 비교
@title_en: A Controlled Single-Variable Comparison of Replacing the Recurrent Neural Network with a Selective State-Space Model in ETER-Net MRI Reconstruction

@abstract_en:
ETER-Net reconstructs MR images directly from undersampled k-space with a bidirectional GRU that performs the k-space-to-image domain transform. We replace only this bi-GRU with a two-dimensional selective state-space model (SS2D), keeping the data, mask, loss, optimizer, schedule, and post-processing U-Net identical, on fastMRI brain multicoil data at 4-fold acceleration. SS2D improves all three standard metrics (volume-averaged SSIM 0.9141 vs. 0.9127) with 21× fewer parameters (31M vs. 668M) and wins on 74–78% of 7,334 validation slices (Wilcoxon p<0.001); an enhanced SS2D variant adds a small but significant gain (SSIM 0.9146).

@body:

# 서론

MRI는 k-space를 순차 수집하므로 촬영이 느리며, 언더샘플링 후 딥러닝 복원은 물리 모델 반복을 펼치는 unrolled 계열[@hammernik2018learning]과 신경망이 k-space를 영상으로 직접 변환하는 도메인 변환 계열[@zhu2018automap]로 나뉜다. ETER-Net[@oh2021eternet]은 후자로, 양방향 GRU(bi-GRU)가 k-space를 영상 도메인 특징으로 변환한 뒤 U-Net이 aliasing을 제거한다. 그러나 bi-GRU는 순차 처리라 병렬화가 어렵고 flatten-reshape 구조 탓에 파라미터가 수억 개에 이른다. 상태공간모델 Mamba[@gu2023mamba]는 입력 의존 순환을 병렬 스캔으로 선형 시간에 계산하며, 2차원 확장 SS2D[@liu2024vmamba]는 네 방향 스캔으로 전역 수용영역을 얻는다. 기존 Mamba 기반 MRI 재구성[@korkmaz2025mambarecon]은 새 구조 전체를 제안해, 순환신경망만 SSM으로 바꾼 효과를 분리한 비교는 없다. 본 논문은 ETER-Net에서 도메인 변환 시퀀스 모델만 bi-GRU→SS2D로 치환하고 나머지를 모두 고정한 단일 변수 통제 비교를 보고한다.

# 방법

그림 1은 두 팔이 공유하는 통제 파이프라인이다. 완전 샘플링 16코일 k-space y_c에서 GT(RSS 영상, 384×384 crop/pad)를 만들고, R=4 equispaced 마스크 M(ACS 8%)을 곱한 언더샘플링 k-space ỹ=M⊙y_c를 실수·허수로 분리한 (32, 384, 384) 텐서가 입력이다. 이 입력은 두 갈래로 흐른다. 시퀀스 모델 f_θ가 k-space를 영상 도메인 특징(20채널)으로 직접 변환하고, 같은 ỹ를 코일별 역 FFT한 zero-filled 코일 영상(32채널)과 채널 결합(52채널)해 후처리 U-Net g_φ(dual-frame skip, depth 5, 31.1M)가 magnitude 영상 x̂를 출력한다. 손실은 brain mask 내부의 L1+(1−SSIM)이고 data-consistency 블록과 ViT 인코더[@oh2025vitbirnn]는 두지 않는다. 점선 상자의 f_θ만이 두 팔 사이의 유일한 변수이며 데이터·마스크·손실·최적화·U-Net은 동일하다.

@figure: paper/figs/conf_fig1_pipeline.png | col | 1.0
@cap_ko: 두 팔이 공유하는 ETER-Net 통제 파이프라인. 점선 상자의 시퀀스 모델 f_θ만이 유일한 변수이고 마스크·zero-filled 분기·U-Net·손실은 두 팔에서 동일하다. 화살표의 숫자는 채널 수(공간 384²), 영상은 검증 슬라이스 예시

그림 2는 f_θ 자리에 들어가는 두 팔의 내부다. (a) bi-GRU 팔은 원본 ETER-Net 그대로 k-space를 384개 행의 시퀀스(스텝당 12,288차원)로 펼쳐 양방향 GRU를 통과시킨 뒤, 전치해 열 방향으로 한 번 더 통과시킨다. 두 GRU의 입력–은닉 행렬(방향별 12,288×11,520과 7,680×11,520)이 파라미터의 대부분이라 GRU 스택만 637.1M, 팔 전체 668.2M이며, 재귀는 스텝 순서대로만 계산된다. (b) SS2D 팔은 픽셀별 LN·Linear(32→128)·SiLU와 depthwise conv 뒤에 selective scan(S6)을 네 방향(각 행 →/←, 각 열 ↓/↑, L=384)으로 적용하고 네 출력을 채널 결합해 LN·Linear와 1×1 conv로 GRU와 같은 20채널에 정합한다. 상태 갱신 h_t=Ā_t h_{t−1}+B̄_t x_t의 (Δ_t, B_t, C_t)는 입력에서 생성되고(d_inner 128, d_state 16), 방향별 한 조의 S6 가중치를 그 방향의 모든 행(열)이 공유하므로 SSM 스택은 0.12M(팔 전체 31.2M)에 그치며 스캔은 병렬 O(L)로 계산된다.

@figure: paper/figs/conf_fig2_arms.png | col | 1.0
@cap_ko: 시퀀스 모델 f_θ의 두 팔. (a) 원본 bi-GRU: k-space 행을 시퀀스로 펼친 양방향 GRU를 행·열 방향으로 2단 적용(flatten-reshape, 668.2M). (b) SS2D: 4방향 cross-scan을 방향별 S6로 병렬 스캔한 뒤 채널 결합(SSM 스택 0.12M, 팔 전체 31.2M)

강화 SS2D(그림 3)는 통제를 해제한 변형이다. stem(LN·Linear 32→256·SiLU) 뒤 stride-3 conv로 128² 격자로 내린 다음, Mamba 게이팅을 복원한 잔차 SS2D 블록 3개(LN·Linear 256→512를 x_ssm|z로 분할 → DWConv·SiLU → 4방향 스캔(d_inner 256, d_state 32, fp16) → y·SiLU(z) → Linear·dropout 0.05 → 잔차 합)를 쌓고, LN·bilinear 업샘플·3×3 conv(SiLU)·1×1 conv로 64채널 특징을 낸다. 이후의 결합·U-Net·손실은 그림 1과 같다. coarse scan 덕분에 epoch당 시간은 통제판 수준(2.84 h 대 3.07 h)이고 파라미터는 34.2M(SSM 스택 3.1M)이다.

@figure: paper/figs/conf_fig3_enhanced.png | col | 1.0
@cap_ko: 강화 SS2D 변형(f_θ 자리 교체, 34.2M). 위: stem → stride-3 다운샘플(128²) → 게이팅 잔차 SS2D 블록 3개 → 업샘플·head(64채널). 아래: 블록 내부 — SSM 분기 x_ssm과 게이트 분기 z의 곱 y·SiLU(z)에 잔차를 더한다

# 실험 및 결과

fastMRI brain multicoil[@zbontar2018fastmri] 확보 서브셋(혼합 contrast)을 공식 구획대로 사용하였다(train 65,028 슬라이스, val 464 볼륨/7,334 슬라이스). GT는 RSS 영상의 384×384 crop/pad, 언더샘플링은 R=4 equispaced Cartesian 마스크(ACS 8%)다. Adam(2×10⁻⁴)·cosine 스케줄·AMP·batch 8·50 epoch(강화판 80)으로 TITAN RTX 1장에서 학습하였다. 지표는 brain mask 내부 SSIM·PSNR·nMSE(%)를 슬라이스 단위로 계산해 fastMRI 관례대로 볼륨 단위 평균±표준편차로 보고하고(표 1, zero-filled 기준선 포함), paired 설계이므로 우위 슬라이스 비율과 볼륨 단위 Wilcoxon 검정으로 유의성을 평가하였다.

표 1에서 SS2D 치환은 21배 적은 파라미터로 세 지표 전부에서 원 설계를 앞섰고, paired 비교에서 슬라이스 74~78%(SSIM 78.2%)·볼륨 90~95%에서 우위였다(모든 지표 p<0.001). 우위는 25회 검증 시점 전부와 5개 contrast 서브그룹 전부(≥68.7%)에서 유지되었다. 정성적으로 bi-GRU만 두개골 바깥에 주기적 ringing을 남겼다. 다만 epoch당 학습시간은 cuDNN GRU가 짧아(2.41 h 대 3.07 h) 효율 이점은 파라미터 수에 있다. 같은 프로토콜로 전체 검증셋을 추론한 공개 모델은 볼륨 SSIM 기준 U-Net† 0.8971, E2E-VarNet† 0.9181, PromptMR+ 0.9417이었다(†: train+val 학습 leaderboard 가중치, 참고선; PromptMR+는 train 구획만 학습한 12-cascade unrolled 모델로 인접 5슬라이스를 입력받아 계열이 다르다). 강화 SS2D는 통제판 대비 세 지표 모두 근소하게 개선되었으나(우위 슬라이스 54~56%), 이 이득은 동일 50 epoch 시점(SSIM 0.9130)이 아닌 80 epoch 연장 구간의 것이다.

@table: col | 1060,520,1070,900,985
@cap_ko: 검증 집합 전체(464 볼륨/7,334 슬라이스, R=4, brain-masked)에 대한 best checkpoint 결과(볼륨 단위 평균±표준편차, 최고값 굵게·차선 밑줄)
@cap_en: Best-checkpoint results on the full validation set (464 volumes/7,334 slices, R = 4, brain-masked); mean±SD over volumes, best in bold, second best underlined
| Method | Params (M) | SSIM ↑ | PSNR (dB) ↑ | nMSE (%) ↓ |
|---|---|---|---|---|
| Zero-filled | – | 0.7523±0.0410 | 24.76±2.11 | 3.935±2.166 |
| bi-GRU (original) | 668 | 0.9127±0.0366 | 33.78±1.86 | 0.448±0.274 |
| SS2D (controlled) | 31 | __0.9141±0.0365__ | __33.91±1.90__ | **0.438±0.283** |
| Enhanced SS2D | 34 | **0.9146±0.0361** | **33.92±1.90** | __0.439±0.304__ |
@end

# 결론

ETER-Net의 도메인 변환 자리에서 bi-GRU를 SS2D로 치환하는 것만으로 DC 없이, 21배 적은 파라미터로 표준 지표(SSIM·PSNR·nMSE) 전부와 대다수 슬라이스·볼륨에서 일관된 개선을 얻었다. 이 결론은 "SSM이 RNN보다 우월하다"는 일반론이 아니라 "SS2D 치환이 원 bi-GRU 설계보다 낫다"로 한정되며, 두 팔은 메커니즘과 파라미터화가 함께 달라 그 기여를 분리하지 못한 한계가 있다. 멀티시드 재현, Transformer·pixel-GRU 팔 추가, 가속률 일반화 학습을 진행 중이다.

# REFERENCES
