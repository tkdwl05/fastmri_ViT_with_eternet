# [초안 v2] ETER-Net 골격에서 순환신경망의 선택적 상태공간모델 치환: 직접 k-space-이미지 MRI 재구성의 단일 변수 통제 비교

> **상태**: 투고 구조 한국어 초안 v2 (2026-08-18 작성).
> **기반**: `draft_ko_v1.md`(2026-08-07, 내용 원본) + 『MRI Reconstruction 논문 투고 준비 자료집』(2026-08-18, IMRaD 설계·문헌·저널 분석) + `references.bib`(병합본, 73항목).
> **구조**: MDPI 공학형(Sensors / Bioengineering SI 겨냥 — 자료집 §5·§6, 부록 B-2) — Related Works 독립 절, 아키텍처·ablation 중심.
> **스코프**: v8 통제비교 + v9 unleashed. radapt(R 일반화)는 학습 재개 완료(08-18, ETA ~08-25) — 완주 후 포함 여부 결정(부록 B-3).
> **미완 요소**: §4.4 기준선 결과(스크립트 준비, GPU 대기), Fig.1(미작성), 참고문헌 중 ⚠ 표시 서지(투고 전 확정), §3.4 데이터 서브셋 확보 경위(✎), 저자·소속·펀딩. (Fig.4 는 08-18 생성 완료)
> 영어 전환은 내용(교수님 검토) 확정 후.

**제목 (영문 가제)**: *Replacing the RNN with a Selective State-Space Model in ETER-Net:
A Controlled Single-Variable Comparison for Direct k-Space-to-Image MRI Reconstruction*

**저자**: (미정 — 교수님 상의, 부록 B-1) · **소속**: (미정)

---

## 초록 (Abstract)

ETER-Net [1]은 언더샘플된 k-space 를 양방향 순환신경망(bidirectional RNN)으로 이미지 도메인에
직접 변환(direct domain transformation)하는 MRI 재구성 계열이다. 본 연구는 이 골격에서 **시퀀스
모델 하나만을 단일 변수로 통제**하여, 도메인 변환 자리의 bi-GRU 를 선택적 상태공간모델(selective
state-space model; Mamba [30]의 2차원 확장인 SS2D [31])로 교체했을 때의 효과를 정량 평가한다.
fastMRI brain multicoil 데이터 [19] (384×384, R=4)에서 두 모델은 시퀀스 모델을 제외한 모든
구성요소(데이터로더·손실·최적화·후처리 U-Net)를 공유하며, 원 논문에 충실하게 명시적
data-consistency(DC) 블록 없이 학습된다. 그 결과 SS2D 치환 모델은 **21× 적은 파라미터(31M vs
668M)로 표준 4지표(SSIM·PSNR·NMSE·L1) 전부에서 GRU 를 상회**했으며(best brain-masked SSIM
0.9140 vs 0.9126, PSNR 35.16 vs 35.03 dB), 전체 검증 7,334 슬라이스의 paired 비교에서 지표별로
슬라이스의 74~78%에서 우위를 보였다(Wilcoxon signed-rank, p<0.001). 정성적으로 GRU 재구성에서
관찰되는 두개골 외부 배경의 ringing 아티팩트가 SS2D 에서는 나타나지 않았다. 나아가 통제를
해제하고 SS2D 를 강화(게이팅 복원·잔차 3블록·병목 해제·coarse-scan 다운샘플)한 변형은 epoch 당
학습시간을 단축(2.78→2.51 h/ep)하면서 최종 품질을 추가 개선했다(SSIM 0.9145, 슬라이스의
54~56%에서 우위, p<0.001). 본 결과는 직접 도메인 변환형 재구성에서 RNN→SSM 치환이 DC 의 도움
없이도 일관된 이득을 줌을 통제된 조건에서 보인 것으로, ETER-Net 계열의 자연스러운 차세대 확장
방향을 제시한다.

**Keywords**: MRI reconstruction; accelerated MRI; k-space; domain transformation; recurrent
neural network; state-space model; Mamba; fastMRI

---

## 1. 서론 (Introduction)

**[문단 ① — 연구 동기: MRI 는 왜 느린가, 그리고 느림의 비용]**
MRI 는 전리방사선 없이 뛰어난 연조직 대조도를 제공하는 핵심 진단 기법이지만, 근본적으로 "느린"
영상법이다. MRI 는 영상을 직접 찍지 않는다 — 스캐너가 실제로 수집하는 것은 영상의 2차원 푸리에
계수인 **k-space** 이며, k-space 는 통상 한 번의 반복시간(TR)에 한 줄(phase-encoding line)씩
순차적으로 채워진다. 완전한 영상을 얻으려면 이 줄들을 나이퀴스트 조건에 맞춰 빠짐없이 수집해야
하므로 한 시퀀스의 촬영이 수 분, 다중 contrast 검사 전체로는 수십 분에 이른다. 이 수집 속도는
경사자계 전환에 따른 말초신경 자극이나 RF 에너지 축적(SAR) 같은 물리·생리적 안전 한계로
제약되므로, 하드웨어 성능만으로는 근본적으로 단축할 수 없다 [19]. 긴 촬영은 곧 비용이다: 환자는
그 시간 동안 정지해 있어야 하고(폐쇄공포 환자·소아·중증 환자에게 특히 가혹하다), 움직임은 모션
아티팩트와 재촬영으로 이어지며, 검사 처리량 저하는 대기 시간과 의료 비용 증가로, 심장 등 동적
영상에서는 시간해상도 제한으로 직결된다 [19,22,24]. 따라서 k-space 의 일부만 수집하고
(언더샘플링) 부족한 정보를 복원으로 메우는 **가속 MRI** 는 MRI 물리·신호처리의 오랜 중심
과제였다. 특히 딥러닝 재구성은 최근 전향적 임상 검증 — 표준 촬영과의 진단 호환성 입증 [43],
촬영 시간을 절반으로 줄인 임상 운용 [44] — 을 통과해 상용 장비에 탑재되는 단계에 이르렀고, 가속
상한을 탐구하는 연구로 이어지고 있다 [45]. 즉 문제의 임상적 가치는 확립되어 있으며, 남은 관건은
어떤 모델 구조가 이 복원 문제를 더 정확하고 효율적으로 푸는가이다.

**[문단 ② — 복원 문제의 성격과 고전 해법: 병렬영상과 압축센싱]**
언더샘플링의 대가는 명확하다. 나이퀴스트 조건을 어긴 k-space 를 그대로 역푸리에 변환하면 해부
구조가 겹쳐 접히는 aliasing 아티팩트가 생기고, 원본 복원은 미지수가 방정식보다 많은 ill-posed
역문제가 되어 유일해가 존재하지 않는다. 답을 하나로 고정하려면 사전지식(prior)이 필요하며, 고전적
해법 두 가지가 각각 다른 사전지식을 사용했다. 다중 수신 코일의 공간 감도를 이용하는 병렬영상 —
SENSE [4], GRAPPA [5] — 은 임상 표준이 되었으나 가속률이 커지면 노이즈 증폭(g-factor)이
급격해진다. 압축센싱(CS-MRI) [6]은 무작위 샘플링과 변환 희소성으로 더 높은 가속의 이론적 근거를
제시했으나, 반복 최적화의 긴 재구성 시간과 정규화 파라미터 민감성이 임상 보급의 병목이었다 [24].

**[문단 ③ — 딥러닝 재구성의 두 계열과 벤치마크]**
2016년 이후 딥러닝이 이 자리를 대체하기 시작했다 [7]. 방법론은 크게 두 계열로 나뉜다 [22,23].
첫째, **unrolled 최적화 계열**은 반복 최적화 알고리즘을 신경망 층으로 펼치고 물리 모델
(데이터 일관성, DC)을 매 반복에 끼워 넣는다 — Variational Network [8], Deep Cascade CNN [9],
MoDL [10]이 원형이고, 코일 감도까지 종단 학습하는 E2E-VarNet [11]이 fastMRI 챌린지를
거치며 사실상의 표준 baseline 으로 자리잡았다 [20,21]. 둘째, **직접 도메인 변환(direct mapping) 계열**은 센서(k-space) 도메인에서 이미지
도메인으로의 변환 자체를 신경망이 학습한다 — AUTOMAP [17]이 완전연결층으로 이를 처음 보였고,
ETER-Net [1]은 그 자리를 양방향 RNN 으로 대체해 파라미터를 크게 줄였으며, k-space CNN 을
경유하는 교차 도메인 KIKI-net [18]도 같은 문제의식을 공유한다. 이 밖에 score 기반 생성모델
(diffusion) 계열 [25,26]이 최근 세 번째 축으로 부상했다. 공정한 비교의 기반으로는 대규모 raw
k-space 공개 데이터셋 fastMRI [19]와 그 챌린지 [20,21]가 표준 벤치마크로 자리잡았다.

**[문단 ④ — 남은 문제: 도메인 변환 자리의 시퀀스 모델]**
본 연구는 두 번째 계열, 그중에서도 ETER-Net 계열 [1,2,3]의 심장부인 **도메인 변환 시퀀스 모델**
에 주목한다. ETER-Net 의 bi-RNN(GRU)은 k-space 행/열을 순차로 읽어 이미지 특징으로 변환하는데,
flatten-reshape 구조 탓에 파라미터가 비대해지고(본 세팅 668M), 순차 의존으로 병렬화가 제한된다.
한편 선택적 상태공간모델 Mamba [30]는 입력 의존 상태 전이로 장거리 의존을 **선형 복잡도**로
모델링하며, 4방향 selective scan 으로 2차원에 확장한 SS2D(VMamba) [31] 이후 비전 과제에서
RNN/Transformer 의 대안으로 빠르게 자리잡았다. MRI 재구성에도 Mamba 적용 연구가 이미 다수
존재하지만 [32–40], 이들은 모두 **이미지 도메인 prior/정규화기 또는 unrolled 백본** 자리에 SSM
을 넣는 새 아키텍처 제안이다. 즉 **도메인 변환(k-space→image) 자리의 RNN 을 SSM 으로 1:1
치환하면 무엇이 달라지는가에 대한 통제 비교는 문헌에 없다**. 우리의 선행 내부 실험(ViT 인코더
하이브리드 [3] 유사 구조)에서 GRU(+U-Net 후처리) 대 SS2D(+DC) 비교는 brain-masked SSIM 0.9084
vs 0.9083 의 사실상 동률(dead-heat)로 끝났으나, 이 비교는 DC 유무와 후처리 구조가 시퀀스 모델
종류와 얽힌 confound 를 안고 있었다. 본 연구는 confound 를 제거한 순수 골격에서 질문을 격리한다:
**"직접 도메인 변환 자리에서, Mamba 는 GRU 보다 나은가?"**

**[문단 ⑤ — 기여와 논문 구성]**
본 논문의 기여는 세 가지다.

1. **단일 변수 통제 비교**: ETER-Net 골격(ViT 없음, DC 없음)에서 시퀀스 모델만 GRU↔SS2D 로
   교체한 통제 실험으로, SS2D 가 표준 4지표 전부·matched-epoch 전 구간(wire-to-wire)·검증
   슬라이스의 74~78%에서 일관 우위임을 보인다 — **파라미터는 21× 적다**(31M vs 668M).
2. **DC 무관성 실증**: DC 가 전혀 없는 세팅에서의 완승으로, 선행 dead-heat 에 대한 "SS2D 는 DC
   덕"(DC-crutch) 가설을 반박한다. 아울러 도메인 변환형 골격에 종단 single soft-DC 를 붙이는
   것이 unrolled 문헌의 DC 관행 [8–11,13–16]과 구조적으로 다름을 정리하고 no-DC 설계를 정당화한다.
3. **통제 해제 시의 상한 탐색**: 게이팅·잔차 스택·병목 해제·coarse-scan 다운샘플을 더한 강화
   SS2D 로 epoch 당 학습 속도와 최종 품질을 동시 개선한다. matched-epoch 기준의 정직한 해석
   (§5.3)을 함께 제시한다.

이하 §2 는 관련 연구, §3 은 골격·변형·학습/평가 프로토콜, §4 는 결과, §5 는 고찰, §6 은 결론이다.

## 2. 관련 연구 (Related Works)

### 2.1 직접 도메인 변환 (direct k-space-to-image) 계열

AUTOMAP [17]은 센서→이미지 매핑을 완전연결층으로 통째로 학습할 수 있음을 보였으나 해상도 제곱에
비례하는 파라미터가 실용의 벽이었다. ETER-Net [1]은 이 변환을 수평/수직 양방향 RNN 두 개로
분해해 파라미터를 낮추고 CNN(U-Net)에 de-aliasing 을 맡기는 구조로, R=4 에서 SSIM 0.931 을
보고했으며 **명시적 DC 블록이 없다**. 이후 radial 등 non-Cartesian 궤적으로 확장되었고 [2],
최근에는 ViT 인코더와의 하이브리드 [3]가 제안되어 BiRNN 의 k-space 순차 처리가 고가속·랜덤
샘플링 강건성의 핵심임이 보고되었다. k-space 도메인 CNN 을 포함하는 교차 도메인 계열로는
KIKI-net [18]이 있다. **본 연구의 위치**: [1,3]의 골격을 유지한 채 도메인 변환 모듈만 교체하는
직접 후속(ablation) 연구다.

### 2.2 Unrolled 최적화 + DC 계열

CRNN-MRI [13], Variational Network [8], Deep Cascade [9], MoDL [10], E2E-VarNet [11],
RecurrentVarNet [14], CIRIM [15] 등은 공통적으로 **작은 recurrent/conv unit 을 최적화 반복으로
unroll 하고 DC 를 매 반복에 interleave** 한다. 체계적 비교 연구 [16]가 정리하듯 DC 는 이 계열의
"cornerstone"이다. 병렬영상 연산을 신경망과 명시적으로 결합한 GrappaNet [12]도 넓게는 이 물리
주도 계열에 속한다. 반면 본 골격(거대 bi-RNN/SSM 도메인 변환 + 종단 U-Net)은 unroll 반복이
없으며 원 논문 [1]에도 DC 가 없다. 따라서 본 비교는 no-DC 를 기본 축으로 설계했다(근거와 내부
실험은 §5-3).

### 2.3 Transformer / SSM(Mamba) 기반 MRI 재구성

attention 의 전역 수용영역을 활용한 Transformer 계열 — SwinMR [27], HUMUS-Net [28], recurrent
구조와 결합한 ReconFormer [29] — 이 CNN 의 지역성 한계를 공략해 왔으나, 시퀀스 길이 제곱의
연산이 고해상도에서 부담이다. Mamba [30]와 그 2차원 확장 SS2D/VMamba [31]는 같은 전역 문맥을
선형 복잡도로 제공한다. MRI 재구성 적용은 이미 활발하다: 이미지/이중 도메인 prior 로서의
MambaRecon [32], DM/DH-Mamba [33](k-space 직접 스캔의 스펙트럼 파괴 문제를 지적), CAM [35],
HiFi-Mamba [36], 불확실성 추정을 겸한 MambaMIR [39], 다중 모달 융합 MMR-Mamba [40], unrolled
백본으로서의 MambaRoll [34], SO-Mamba [38], 연산자 학습 관점의 LMO [37] 등이다. **차별점**:
이들 연구는 새 아키텍처 제안과 SOTA 경쟁이 목적이고, SSM 의 자리는 이미지 도메인(또는 이중
도메인) prior/정규화기다. 본 연구는 **기존 골격에서 RNN↔SSM 치환 효과를 격리하는 통제 실험**이
목적이며, 도메인 변환(k→image) 자리의 SSM 은 [32–40] 어디에도 없다.

### 2.4 본 연구의 위치 (비교 표)

| 계열 | 대표 문헌 | 시퀀스/전역 모듈의 자리 | DC | 본 연구와의 관계 |
|---|---|---|---|---|
| 직접 도메인 변환 | AUTOMAP [17], **ETER-Net [1–3]**, KIKI-net [18] | **k-space→image 변환 그 자체** | 없음(원 논문 기준) | 본 연구의 골격 — 변환 모듈의 세대 교체를 검증 |
| Unrolled + DC | [8–11,13–16] | 반복 내부의 작은 정규화 unit | 매 반복 interleave | 구조가 달라 직접 비교 대상 아님 (§5-3) |
| Mamba-MRI | [32–40] | 이미지/이중 도메인 prior, unrolled 백본 | 대부분 있음 | SSM 을 쓰지만 **자리가 다름** |
| **본 연구** | — | **도메인 변환 자리의 RNN↔SSM 1:1 치환** | 없음 (통제) | 문헌에 없는 통제 비교를 채움 |

## 3. 방법 (Methods)

### 3.1 문제 정식화

C 개 코일의 fully-sampled k-space 를 y = {y_c}, 언더샘플링 마스크를 M 이라 하면 관측은

    ỹ_c = M ⊙ y_c ,   c = 1, …, C

이고, 목표는 관측 {ỹ_c}로부터 기준 영상 x*(root-sum-of-squares, RSS magnitude)를 복원하는
것이다. Unrolled 계열이 x̂ = argmin_x ‖M F x − ỹ‖² + R(x) 의 반복을 펼치는 것과 달리, 직접
도메인 변환 계열은 매핑

    x̂ = g_φ( concat( f_θ({ỹ_c}),  F⁻¹{ỹ_c} ) )

을 종단 학습한다. 여기서 f_θ 는 k-space 입력을 이미지 도메인 특징으로 변환하는 **시퀀스 모델**
(본 연구의 유일 변수: bi-GRU 또는 SS2D), F⁻¹{ỹ_c} 는 zero-filled(aliased) 코일 영상, g_φ 는
de-aliasing 후처리 U-Net 이다. f_θ 를 제외한 모든 것을 고정한다.

### 3.2 공통 골격 (순수 ETER-Net; ViT·DC 없음) — Fig. 1

```
입력 1: aliased 코일영상  (B, 32, 384, 384)   # 16코일 × (실수부/허수부)
입력 2: k-space           (B, 32, 384, 384)
                     │
        시퀀스 모델 f_θ = bi-GRU (수평+수직)  또는  SS2D    ← 유일한 변수
                     │
        concat(시퀀스 출력, aliased 코일영상)                ← 2-way concat
                     │
        후처리 U-Net g_φ (skip-connection, depth=5, wf=6)   # ~30M, 양 arm 공유
                     │
        출력: magnitude 영상 (B, 1, 384, 384)
```

- **GRU arm**: ETER-Net 원본 [1]의 양방향(수평+수직) GRU. 행/열 flatten-reshape 구조로 총
  668M 파라미터.
- **SS2D arm (통제판)**: 4방향 selective scan(SS2D [31]) 단일 블록. 출력 채널을 GRU arm 과
  동일한 20으로 강제 정합해 용량 상한을 GRU 이하로 억제 — 총 31M(이 중 공유 U-Net 이 ~30M 로
  지배적이고, SSM 자체는 ~1M).
- 이 외 **모든 것이 동일**하다: 데이터로더·언더샘플링 마스크·손실·옵티마이저·스케줄·에폭·후처리
  U-Net·랜덤 시드.

### 3.3 강화 SS2D (통제 해제 변형) — Fig. 1(b)

통제비교의 SS2D 는 공정성을 위해 의도적으로 최소 구성이다. 통제를 해제한 강화 변형은 세 가지를
복원/확장한다:

1. **게이팅 복원** — 공식 Mamba [30]의 `y = y · SiLU(z)` 게이트 분기(통제판에는 없음).
2. **잔차 스택** — 채널 불변 SS2D 블록 × 3 (residual skip).
3. **병목 해제** — 출력 채널 20→64, d_inner 128→256, d_state 16→32, dropout 0.05.

384² 풀해상도 4방향 스캔의 연산 병목은 **fp16 selective scan** 과 **다운샘플 front-end(ds=3)**
로 해결했다: stem 이 풀해상도 k-space 를 먼저 처리한 뒤 특징을 128² 로 낮춰 coarse scan 하고
bilinear 업샘플해 U-Net 에 전달한다(전역 문맥은 SSM, 풀해상도 디테일은 U-Net 이 분업). 그 결과
풀용량을 유지하면서 통제판보다 빠른 2.51 h/epoch(vs 2.78)를 달성해, 동일 학습 예산에서 epoch
50→80 연장이 가능했다. 총 파라미터 ~33M(SSM 스택 ~2M). 학습 위생으로 Mamba 상태 파라미터
(`A_log`, `D`)는 weight-decay 에서 제외했다.

### 3.4 데이터셋과 언더샘플링 프로토콜

| 항목 | 값 |
|---|---|
| 데이터 | fastMRI brain multicoil [19], **혼합 contrast** (AXT1/AXT1POST/AXT1PRE/AXT2/AXFLAIR) |
| 규모 | train 4,108 파일 / 65,028 슬라이스 · val 464 파일 / 7,334 슬라이스 — fastMRI 제공 train/val 구획의 확보 서브셋(`reconstruction_rss` 보유 파일 전부 사용; 서브셋 확보 경위 서술 ✎투고 전 확정) |
| 정답(GT) | 데이터셋 제공 RSS 재구성(`reconstruction_rss`)을 384×384 center-crop/zero-pad |
| 전처리 | full k-space → iFFT(ortho) → 이미지 도메인 384×384 crop/pad → 재-FFT 로 384² k-space 유도 (retrospective) |
| 코일 | 앞 16개 코일 사용(초과분 절단, 부족분 zero-fill) → 실/허수 분리 32채널 |
| 언더샘플링 | R=4 equispaced Cartesian 1D 마스크, 중앙 ACS 8% (train 은 매 샘플 offset 랜덤, val 은 고정) |
| 증강 | 수평/수직 flip p=0.5 — flip 후 FFT 재계산으로 k-space 물리 정합 유지 |

### 3.5 손실 함수와 평가지표 (brain-masked)

- **Brain mask**: Otsu 임계 × 0.4 + 최대 연결성분(largest CC). 배경(영상의 절반 이상)이 지표를
  부풀리는 것을 차단한다.
- **손실**: masked L1 + (1 − SSIM).
- **평가**: brain-masked SSIM / PSNR / NMSE / L1 — **주 지표는 SSIM**(fastMRI 챌린지 표준
  [20,21]), 나머지는 병기한다. 단 본 지표는 뇌 영역 한정이므로, 배경을 포함하는 fastMRI 공식
  프로토콜(320 crop·raw)의 리더보드/문헌 수치와는 좌표계가 달라 직접 대조하지 않는다.
- **모델 선택**: best checkpoint·조기 종료 기준으로는 SSIM·PSNR·NMSE 를 가중 합성한 내부
  스칼라를 사용했다. 이 스칼라는 문헌 표준이 아니므로 방법 절의 이 한 줄로만 서술하며, **본
  논문의 모든 결과 표·수치는 표준 지표로만 제시**한다. 본 논문의 결론은 표준 지표 전부에서
  성립한다(§4.1).

### 3.6 학습 세부

Adam(LR 2e-4), CosineAnnealingLR, AMP(fp16), gradient clipping 1.0, batch size 8. 통제비교
50 epoch / 강화판 80 epoch, 검증 매 2 epoch. 하드웨어는 NVIDIA TITAN RTX 24GB 단일 GPU. 구현은
PyTorch 2.3 + mamba_ssm 2.2(CUDA selective-scan 커널). 코드는 게재 시 GitHub 에 공개할 계획이다
(재현성, 자료집 §5-Methods 권고).

### 3.7 기준선 (baselines)

fastMRI **brain leaderboard 사전학습 U-Net·E2E-VarNet [11]** 을 동일 검증 파이프라인(같은
슬라이스·R4 마스크·GT·brain mask)에서 추론 평가해 참고 기준선으로 제시한다. 두 모델의 출력
스케일은 자체 정규화 기준이므로 지표 계산 전 per-slice 최소제곱 스케일 정합을 적용한다(우리
모델은 α≈1). 단 leaderboard 가중치의 원 학습분포(전체 코일·native 해상도)와 본 전처리
(16코일·384² 재-FFT) 사이에 domain shift 가 있으므로 절대 우열 판정이 아닌 **참고선**으로
표기한다. (VarNet 의 감도 추정이 k-space ortho 스케일에서 발산하는 문제는 unit-max 정규화로
완화했으며, 잔여 non-finite 슬라이스는 제외하지 않고 개수를 보고한다.)

### 3.8 통계 분석

두 모델이 동일 슬라이스를 재구성하는 **paired 설계**다. 지표별 paired 차이에 Wilcoxon
signed-rank 검정을 적용하고, 효과크기로 **우위 슬라이스 비율**(proportion of slices favoring;
통계학의 probabilistic index [46]에 해당, 임상시험의 win-ratio 계열 [47]과 동족)을 함께
보고한다. 영어 원고 표기 예: *"SS2D achieved higher SSIM in 78.2% of slices (5,737/7,334;
Wilcoxon signed-rank test, p < 0.001)."* p 값은 저널 관례에 따라 p<0.001 로 표기한다(원값은
저장소 산출물에 보존).

## 4. 결과 (Results)

### 4.1 통제 비교 — SS2D 의 일관 우위 (Table 1, 2; Fig. 2)

**Table 1. Best checkpoint 기준 (val 전체, brain-masked).**

| | epoch | SSIM | PSNR (dB) | NMSE | L1 | params |
|---|---:|---:|---:|---:|---:|---:|
| **SS2D (통제판)** | 48 | **0.9140** | **35.16** | **0.0039** | **8.931** | **31M** |
| GRU | 50 | 0.9126 | 35.03 | 0.0040 | 9.054 | 668M |

- **Matched-epoch 전 구간 우위(wire-to-wire)**: 25개 검증 지점(ep2~50)에서 SS2D 가 SSIM 기준 전
  지점 ≥ GRU(Δ +0.0000~+0.0055; 동률은 ep14 1지점뿐), PSNR 기준 전 지점 우위였다. ViT
  하이브리드 선행 실험에서 관찰됐던 후반 역전(crossover)은 없었다. (Fig. 2)
- **Per-slice paired 검증** (val 7,334 슬라이스 전수, Wilcoxon signed-rank; 차이 분포는 Fig. 4a):

**Table 2. Per-slice paired 비교.**

| 지표 | GRU mean±std | SS2D mean±std | SS2D 우위 슬라이스 비율 | p |
|---|---:|---:|---:|---:|
| SSIM | 0.9126±0.0894 | 0.9140±0.0890 | **78.2%** | <0.001 |
| PSNR | 33.78±2.59 | 33.90±2.63 | **73.8%** | <0.001 |
| NMSE | 0.0045±0.0053 | 0.0044±0.0053 | **73.8%** | <0.001 |
| L1 | 9.00±3.05 | 8.88±3.05 | **76.1%** | <0.001 |

즉 aggregate 우위가 소수 슬라이스에 의한 것이 아니라 **대다수 슬라이스에서 일관**된다.
(주: 본 표의 PSNR/NMSE/L1 절대값은 Table 1 과 다르다 — 학습 시 검증이 배치 풀링으로 계산된 반면
본 표는 슬라이스 단위 계산이다. SSIM 은 양쪽 모두 슬라이스 단위라 정확히 일치하며, paired 비교
자체는 3개 모델 동일 프로토콜로 유효하다.)

### 4.2 정성 비교 — GRU 의 배경 ringing (Fig. 3)

4-way 시각화(GT / U-Net / GRU / SS2D)에서 **GRU 는 두개골 바깥 배경에 반복적 ringing/줄무늬
아티팩트**를 보이는 반면 SS2D 의 해당 영역은 깨끗했다(검토한 모든 슬라이스에서 일관).
brain-mask 밖이라 정량 지표에는 반영되지 않는 순수 정성적 차이로, GRU 재구성이 관심영역 밖에서
덜 안정적임을 시사한다.

### 4.3 강화 SS2D — 상한 탐색 (Table 3; Fig. 2, 4)

**Table 3. 강화 SS2D vs 통제판 (val 전체, brain-masked, best checkpoint).**

| | SSIM | PSNR (dB) | NMSE | L1 |
|---|---:|---:|---:|---:|
| **강화 SS2D (best ep78/80)** | **0.9145** | **35.18** | 0.0039 | 8.931 |
| SS2D 통제판 (best ep48/50) | 0.9140 | 35.16 | 0.0039 | 8.931 |
| GRU (best ep50/50) | 0.9126 | 35.03 | 0.0040 | 9.054 |

- Per-slice 우위 슬라이스 비율: vs 통제판 SS2D **54~56%**(표준 4지표 전부, Wilcoxon p<0.001 —
  유의하나 근소), vs GRU **78~82%**(완승). 차이 분포는 Fig. 4b.
- **정직한 해석**: matched-ep50 시점 강화판 SSIM 은 0.9130 으로 통제판 best(0.9140)에 미달하며,
  통제판 best 에 도달한 것은 연장 구간의 ep64(동률)~ep66(상회)이다. 즉 "같은 학습량에서 더
  좋다"가 아니라 **"epoch 당 속도 이득(2.51 vs 2.78 h/ep)으로 더 긴 스케줄(80ep)을 소화해 최종
  품질을 넘었다"**가 정확한 서사다(best 도달 wall-clock 은 강화판 ≈181h > 통제판 ≈133h).
  아키텍처 강화 자체의 순수 이득은 근소하다.
- Coarse-scan(ds=3) 다운샘플은 품질을 해치지 않았다: ep40 시점 열위였다가 후반 cosine anneal
  구간에서 역전, 최종 상회했다.

### 4.4 기준선 비교 — U-Net / E2E-VarNet **[결과 삽입 예정]**

§3.7 프로토콜로 전체 val 7,334 슬라이스에서 사전학습 U-Net·E2E-VarNet [11]을 추론 평가한 결과
(표준 4지표 mean±std + 본 모델들과의 우위 슬라이스 비율)를 **Table 4** 로 여기에 삽입한다.
**[스크립트 준비 완료: `v8_eter_pure/eval_paired_baselines.py` — radapt 학습(GPU 점유, ~08-25
완주 예상) 종료 후 실행.]**

## 5. 고찰 (Discussion)

**(1) DC-crutch 가설의 반박.** ViT 하이브리드 선행 비교의 dead-heat 는 "SS2D 의 성능은 DC
덕분"이라는 해석을 허용했다. 그러나 DC 를 완전히 제거한 본 통제 세팅에서 SS2D 는 오히려 더
확실하게 이겼다. Mamba 의 이득은 DC 와 무관한 시퀀스 모델링 능력 자체에서 나온다.

**(2) 파라미터 효율의 해석.** 668M 의 GRU 를 31M 모델이 이긴다(21×↓). 도메인 변환 자리 RNN 의
flatten-reshape 구조가 비효율의 근원이며, SSM 은 같은 자리를 선형 복잡도·저용량으로 대체한다.
이는 AUTOMAP [17]→ETER-Net [1]이 밟았던 "같은 기능, 더 효율적인 모듈로" 궤적의 다음 단계로 읽을
수 있다.

**(3) DC 축을 주 비교에서 제외한 근거.** (a) 원 논문 [1]에 DC 가 없고, (b) 문헌의 DC 는 unrolled
반복에 interleave 되는 구조 [8–11,13–16]로, 본 골격의 종단 single soft-DC 와 다르며, (c) 내부
실험에서 종단 soft-DC 는 fp16 학습 불안정(gradient overflow)을 유발했고 GRU/SS2D 를 유의미하게
구분하지 못했다. DC 도입은 문헌식 재설계가 필요한 별개 과제로 분리한다(진행 중인 R-일반화
변형에서 다룸 — 아래 (6)).

**(4) 선행 Mamba-MRI 와의 관계.** 본 결과는 [32–40]의 아키텍처 기여와 경쟁하지 않는다. 그들이
"어떤 새 Mamba 구조가 SOTA 인가"를 묻는다면, 본 연구는 "기존 도메인 변환 골격에서 RNN→SSM
치환만으로 무엇이 달라지는가"를 격리해 답한다. 특히 DM/DH-Mamba [33]가 지적한 k-space 직접
스캔의 스펙트럼 파괴 우려에 대해, 본 결과는 ETER-Net 식 도메인 변환 자리(k-space 입력)에서도
SSM 이 RNN 을 일관되게 상회함을 실증한다 — 이는 해당 우려가 도메인 변환형 골격에는 그대로
적용되지 않음을 시사한다.

**(5) 평가지표의 신뢰성.** SSIM/PSNR 이 높아도 병변 소실이나 구조 hallucination 을 잡지 못한다는
것은 fastMRI 챌린지 보고 [21] 이후 정설이며, 딥러닝 재구성의 불안정성 [41]과 정확도–안정성
트레이드오프 [42]도 이론적으로 정리되어 있다. 본 연구는 (i) 배경 부풀림을 차단하는 brain-masked
지표, (ii) aggregate 평균이 아닌 per-slice 우위 비율 + 비모수 검정(§3.8), (iii) 정성 비교
(§4.2)로 평가의 성실성을 보강했으나, **영상의학과 의사 reader study 는 수행하지 않았다** — 이는
본 연구가 임상 성능이 아닌 아키텍처 통제 비교를 주장하는 이유이자 한계다.

**(6) 한계와 향후 연구.**
- 단일 데이터셋(fastMRI brain)·단일 가속률(R=4)·단일 시드 — 가속률 일반화는 mask 조건화 + DC +
  multi-AR(R∈{2,3,4,5,6,8}) 학습을 결합한 **R-적응 변형(radapt)** 으로 진행 중이다(완주 후 본
  논문 포함 여부 결정 — 부록 B-3).
- 통제비교의 SS2D 는 의도적 최소 구성이며, 강화판이 상한을 일부 보완하나 이득은 근소하다(§4.3).
  게이팅/깊이/병목의 개별 기여 분리(ablation)는 수행하지 않았다.
- Retrospective 시뮬레이션(384² 재-FFT 프로토콜)·앞 16코일 절단은 재현성을 위한 선택이나 원
  수집 조건과의 차이다. Knee 등 타 해부부위, prospective 언더샘플링, non-Cartesian 궤적 [2]은
  미검증이다.

**(7) Novelty 포지셔닝 (솔직한 자기 평가).** Mamba-MRI 아키텍처 자체는 이미 성숙 분야다
[32–40]. 본 논문의 가치는 새 아키텍처가 아니라 (a) 도메인 변환 자리에서의 1:1 통제 치환 실험,
(b) no-DC 조건의 DC 무관성 실증, (c) ETER-Net 계열 [1–3]의 직접 후속이라는 점이다. 투고 시 이
프레임을 유지해야 리뷰 방어가 가능하다.

## 6. 결론 (Conclusions)

ETER-Net 골격의 도메인 변환 자리에서 bi-GRU 를 SS2D 로 치환하는 것만으로 — DC 없이, 21× 적은
파라미터로 — 표준 4지표 전부·matched-epoch 전 구간·검증 슬라이스 대다수에서 일관된 품질 향상을
얻었다. 게이팅·깊이·병목 해제를 더한 강화 SS2D 는 epoch 당 학습 속도를 앞당기며 최종 품질을
추가로 근소 개선했다. 직접 도메인 변환형 재구성의 차세대 시퀀스 모델로서 SSM 은 RNN 의 자연스러운
대체재이며, 가속률 일반화(radapt)와 타 궤적·부위 확장이 후속 과제다.

---

## 후반부 항목 (MDPI 양식)

- **Author Contributions**: (CRediT 분류로 작성 예정 — 저자 구성 확정 후, 부록 B-1)
- **Funding**: (미정 — 확인 필요)
- **Institutional Review Board Statement**: 본 연구는 공개 비식별 데이터셋 fastMRI [19]의 2차
  이용만을 포함한다. (NYU fastMRI 데이터 사용 약정 준수; 별도 IRB 심의 면제 해당 여부 문구는
  투고 저널 양식에 맞춰 확정)
- **Data Availability Statement**: fastMRI 데이터셋은 https://fastmri.med.nyu.edu 에서 신청
  가능. 학습·평가 코드는 게재 시 GitHub 공개 예정.
- **Conflicts of Interest**: 없음(예정 — 확인 필요).
- **셀프 체크**: 투고 전 CLAIM 체크리스트 [48] 자체 점검(자료집 §5 권고).

---

## 참고문헌 (번호 ↔ `paper/references.bib` 키 매핑)

*⚠ 표시 = 서지 미확정(bib note 참조), 투고 전 Crossref/출판사 페이지로 확정할 것.*

**직접 계보 (ETER-Net 계열)**
1. Oh et al. (2020). A k-space-to-image reconstruction network for MRI using recurrent neural network. *Medical Physics*. `oh2020eternet` ⚠
2. Oh et al. (2022). An end-to-end recurrent neural network for radial MR image reconstruction. *Sensors*. `oh2022radial` ⚠
3. Oh (2025). A hybrid Vision Transformer-BiRNN architecture for direct k-space to image reconstruction in accelerated MRI. *Journal of Imaging*. `oh2025vitbirnn` ⚠ ★직접 선행

**고전 기반**
4. Pruessmann et al. (1999). SENSE: sensitivity encoding for fast MRI. *MRM* 42(5):952–962. `pruessmann1999sense`
5. Griswold et al. (2002). GRAPPA. *MRM* 47(6):1202–1210. `griswold2002grappa`
6. Lustig et al. (2007). Sparse MRI: the application of compressed sensing for rapid MR imaging. *MRM* 58(6):1182–1195. `lustig2007sparse`

**딥러닝 기둥 (unrolled·직접변환·교차도메인)**
7. Wang et al. (2016). Accelerating magnetic resonance imaging via deep learning. *IEEE ISBI*. `wang2016accelerating`
8. Hammernik et al. (2018). Learning a variational network for reconstruction of accelerated MRI data. *MRM* 79(6):3055–3071. `hammernik2018learning`
9. Schlemper et al. (2018). A deep cascade of CNNs for dynamic MR image reconstruction. *IEEE TMI* 37(2):491–503. `schlemper2018deep`
10. Aggarwal et al. (2019). MoDL: model-based deep learning architecture for inverse problems. *IEEE TMI* 38(2):394–405. `aggarwal2019modl`
11. Sriram et al. (2020). End-to-end variational networks for accelerated MRI reconstruction. *MICCAI*. `sriram2020endtoend`
12. Sriram et al. (2020). GrappaNet. *CVPR*. `sriram2020grappanet`
13. Qin et al. (2019). Convolutional recurrent neural networks for dynamic MR image reconstruction. *IEEE TMI* 38(1):280–290. `qin2019crnn`
14. Yiasemis et al. (2022). Recurrent Variational Network. *CVPR*. `yiasemis2022recurrentvarnet`
15. Karkalousos et al. (2022). Assessment of data consistency through cascades of independently recurrent inference machines (CIRIM). *Phys. Med. Biol.* `karkalousos2022cirim` ⚠
16. Hammernik et al. (2021). Systematic evaluation of iterative deep neural networks for fast parallel MRI reconstruction. *MRM* 86. `hammernik2021systematic` ⚠
17. Zhu et al. (2018). Image reconstruction by domain-transform manifold learning (AUTOMAP). *Nature* 555:487–492. `zhu2018automap`
18. Eo et al. (2018). KIKI-net. *MRM* 80(5):2188–2201. `eo2018kiki`

**fastMRI 벤치마크**
19. Zbontar et al. (2018). fastMRI: an open dataset and benchmarks for accelerated MRI. arXiv:1811.08839. `zbontar2018fastmri`
20. Knoll et al. (2020). Overview of the 2019 fastMRI challenge. *MRM* 84(6):3054–3070. `knoll2020advancing`
21. Muckley et al. (2021). Results of the 2020 fastMRI challenge. *IEEE TMI* 40(9):2306–2317. `muckley2021results`

**리뷰**
22. Heckel et al. (2024). Deep learning for accelerated and robust MRI reconstruction. *MAGMA* 37:335–368. `heckel2024deep`
23. Hammernik et al. (2023). Physics-driven deep learning for computational MRI. *IEEE SPM* 40(1):98–114. `hammernik2023physics`
24. Knoll et al. (2020). Deep-learning methods for parallel MR image reconstruction. *IEEE SPM* 37(1):128–140. `knoll2020deeplearning`

**생성모델**
25. Chung & Ye (2022). Score-based diffusion models for accelerated MRI. *Med. Image Anal.* 80:102479. `chung2022score`
26. Jalal et al. (2021). Robust compressed sensing MRI with deep generative priors. *NeurIPS*. `jalal2021robust`

**Transformer**
27. Huang et al. (2022). Swin transformer for fast MRI. *Neurocomputing* 493:281–304. `huang2022swinmr`
28. Fabian & Soltanolkotabi (2022). HUMUS-Net. *NeurIPS*. `fabian2022humus`
29. Guo et al. (2024). ReconFormer. *IEEE TMI* 43(1). `guo2024reconformer`

**SSM/Mamba 원류·MRI 적용**
30. Gu & Dao (2023). Mamba: linear-time sequence modeling with selective state spaces. arXiv:2312.00752. `gu2023mamba`
31. Liu et al. (2024). VMamba: visual state space model. *NeurIPS*. `liu2024vmamba`
32. Korkmaz et al. (2025). MambaRecon. *WACV*. `korkmaz2025mambarecon` ⚠
33. Meng et al. (2025). DM/DH-Mamba: dual-domain (hierarchical/multi-scale) Mamba for MRI reconstruction. `dmmamba2025`+`meng2025dhmamba` ⚠(동일 논문 여부 확인 후 통합)
34. Kabas et al. (2024). Physics-driven autoregressive state space models (MambaRoll). `kabas2024mambaroll` ⚠
35. Meng et al. (2026). Image content aware state space model (CAM). `meng2026cam` ⚠
36. Chen et al. (2025). HiFi-Mamba. `chen2025hifimamba` ⚠
37. Li et al. (2025). LMO: linear Mamba operator for MRI reconstruction. *CVPR*. `li2025lmo`
38. Fang et al. (2026). SO-Mamba: state-ownership Mamba for unrolled MRI reconstruction. arXiv:2605.22031. `somamba2026`
39. Huang et al. (2024). MambaMIR. arXiv:2402.18451. `huang2024mambamir` ⚠
40. Zou et al. (2024). MMR-Mamba. arXiv:2406.18950. `zou2024mmrmamba` ⚠

**신뢰성·임상 검증**
41. Antun et al. (2020). On instabilities of deep learning in image reconstruction. *PNAS* 117(48):30088–30095. `antun2020instabilities`
42. Gottschling et al. (2025). The troublesome kernel. *SIAM Review* 67(1). `gottschling2025troublesome`
43. Recht et al. (2020). Using deep learning to accelerate knee MRI at 3T: interchangeability study. *AJR* 215(6):1421–1429. `recht2020interchangeability`
44. Johnson et al. (2023). Deep learning reconstruction enables prospectively accelerated clinical knee MRI. *Radiology* 307(2):e220425. `johnson2023prospective`
45. Radmanesh et al. (2022). Exploring the acceleration limits of deep learning VarNet-based 2D brain MRI. *Radiology: AI* 4(6):e210313. `radmanesh2022limits`

**통계·보고 지침**
46. Acion et al. (2006). Probabilistic index. *Statistics in Medicine* 25(4):591–602. `acion2006probabilistic`
47. Wang & Pocock (2016). A win ratio approach to comparing continuous non-normal outcomes. *Pharmaceutical Statistics*. `wang2016winratio` ⚠
48. Mongan et al. (2020). CLAIM checklist. *Radiology: AI* 2(2):e200029. `mongan2020claim`

---

## 부록 A. 그림·표 ↔ 저장소 산출물 매핑

| 논문 요소 | 소스 (저장소 경로) | 상태 |
|---|---|---|
| Fig.1 아키텍처 다이어그램 (a: 공통 골격+양 arm, b: 강화 SS2D) | 신규 작성 필요 (§3.2/3.3 기반) | ❌ 미작성 |
| Fig.2 학습 곡선 (GRU vs SS2D vs 강화판, SSIM/PSNR) | `results/eval/v9_unleashed/curves_v9_vs_v8.png` | ✅ (논문용 재도색 권장) |
| Fig.3 4-way 정성 비교 + 배경 ringing | `results/vis/v8_pure_eternet_compare/compare_*.png` | ✅ (슬라이스 선별 필요) |
| Fig.4 per-slice 우위 비율/차이 분포 | `paper/figs/fig4_per_slice_distribution.{png,pdf}` — 생성 스크립트 `paper/make_fig4_per_slice.py` (입력 `results/eval/v9_unleashed/per_slice_paired_v9.csv`) | ✅ 2026-08-18 (2행×4지표 paired-Δ 히스토그램 + win-rate, 양수=치환/강화 우위 규약) |
| Tab.1 best 지표 비교 | `docs/v8_eter_pure_rnn_vs_ss2d.md` §3 | ✅ |
| Tab.2 per-slice win-rate | `results/eval/*/win_rate_summary*.md` | ✅ |
| Tab.3 강화판 비교 | `docs/v9_mamba_unleashed_and_radapt.md` §11.1 | ✅ |
| Tab.4 기준선(U-Net/E2E-VarNet) | `v8_eter_pure/eval_paired_baselines.py` → `results/eval/baselines_384/` | ⏳ 스크립트 준비, GPU 대기(radapt 완주 ~08-25 후) |
| (참고) matched-epoch 표 | `results/eval/*/matched_epoch_table*.md` | ✅ (본문 서술로만 사용) |

## 부록 B. 교수님 상의 포인트 (초안 논외, 회의용)

1. **[3](교수님 2025 ViT-BiRNN, J. Imaging)과의 관계 설정** — 본 논문을 [1,3]의 직접 후속(도메인
   변환 모듈의 세대 교체)으로 프레이밍하는 안. 저자 구성·기여 서술에 영향.
2. **투고처** — 자료집 §6 + v1 부록 B 평가 종합 (2026-08-18 갱신):

   | 순위(안) | 저널 | 근거 | 비고 |
   |---|---|---|---|
   | **1** | **Bioengineering SI** (MDPI, 마감 11-30) | 뇌+DL+**컴퓨팅 가속**(fp16 스캔·ds=3·선형복잡도) 서사 정합. IF 4.4 로 후보 중 최고 | "탐지" 중심 SI 라 **재구성 스코프 게스트 에디터 사전 문의 필수**. 11-30 마감이라 radapt 포함 가능 |
   | **2** | **Sensors** (MDPI) | **교수님 ETER-Net 확장판(2022)이 Sensors 게재** [2] — 계보 연속성. MRI recon 게재 전례 다수 [자료집 2.1절 [6]] | 마감 없음. SI 문의가 부정적이면 1순위 승격 |
   | 3 | QIMS (AME) | 정량 영상 전문지 | 임상 독자 성향 — reader study 부재가 약점 |
   | 4 | Diagnostics (MDPI) | IF 3.8 | 진단 중심 스코프 스트레치. (v1 시점 SI 마감 8-31 은 경과) |
   | 5 | CMC | 심사 빠름 | 의학영상 독자 부재, ETER-Net 계보와 단절 — 권장하지 않음 |

   **권장 액션**: Bioengineering SI 게스트 에디터에게 가제+초록 사전 문의(비용 0) → 긍정이면 SI,
   부정이면 Sensors. 전 후보 OA·APC — 금액·할인 투고 전 확인. ([3] *J. Imaging* 도 MDPI —
   교수님 선호 확인 가치.)
3. **radapt(R 일반화) 포함 여부** — 08-18 학습 재개, **~08-25 완주 예상**. 포함 시
   "치환+강화+일반화" 3막 구성으로 기여가 두터워지나 투고 지연. 미포함 시 본 초안 그대로 + 후속
   논문 분리. (Bioengineering SI 마감 11-30 이면 포함 여유 있음.)
4. **DC 서술 수위** — §5-(3)의 "문헌 비표준 + 학습 불안정" 근거를 본문에 둘지 부록으로 뺄지.
5. **단일 시드 한계** — 리뷰 방어용 멀티시드 재학습(비용 큼) 필요성 판단.
6. **composite 노출 수위 — 해결(2026-08-07 사용자 결정)**: 결과 표·본문 수치에서 전면 제거,
   §3.5 에 모델 선택 기준으로만 한 줄 서술. 모든 서사(wire-to-wire·연장구간 돌파)는 표준 지표로
   재검증 완료.
7. **(v2 신규) reader study** — 자료집 §3.2 가 권고하는 소규모 영상의학과 의사 평가를 넣을지.
   현재는 §5-(5)에서 한계로 정면 서술하는 전략. 의학 계열 저널로 갈 경우 재론.

---

*변경 이력: v1(2026-08-07, 교수님 상의용) → v2(2026-08-18, 투고 준비 자료집 기반 재구조화 —
서론 5문단 체계·Related Works 확장·문제 정식화·평가 신뢰성 고찰·MDPI 후반부 항목·참고문헌
48편 병합[`paper/references.bib`]).*
