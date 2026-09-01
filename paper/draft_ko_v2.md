# [초안 v2] ETER-Net 골격에서 순환신경망의 선택적 상태공간모델 치환: 직접 k-space-이미지 MRI 재구성의 단일 변수 통제 비교

> **상태**: 투고 구조 한국어 초안 v2 (2026-08-18 작성).
> **기반**: `draft_ko_v1.md`(2026-08-07, 내용 원본) + 『MRI Reconstruction 논문 투고 준비 자료집』(2026-08-18, IMRaD 설계·문헌·저널 분석) + `references.bib`(병합본, 73항목).
> **구조**: MDPI 공학형(Sensors / Bioengineering SI 겨냥 — 자료집 §5·§6, 부록 B-2) — Related Works 독립 절, 아키텍처·ablation 중심.
> **스코프**: v8 통제비교 + v9 unleashed. radapt(R 일반화)는 08-31 재개, ep49+/80 진행 중(ETA ~09-04) — 완주 후 포함 여부 결정(부록 B-3).
> **미완 요소**: §4.4 기준선 결과(GPU 대기), 저자·소속·펀딩, 외부 검토(08-18) 항목 중 GPU 필요분(U-Net-only 기준·추론 속도/VRAM 측정·표준 프로토콜 보충표·ringing 정량화 — radapt 완주 후 일괄) + 멀티시드(교수님 결정, 부록 B-5). 서지는 08-18 전건 확정 + 09-01 계보 2건 추가([50][51] Crossref 확정)(잔여 ✎: [34] TMI 권호 배정 대기만 — [33] 은 09-01 확정). **외부 검토 반영 현황은 부록 D.**

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
668M)로 표준 4지표(SSIM·PSNR·NMSE·L1) 전부에서 GRU 와 동등 이상 — 실제로는 전 지표 우위 — 의
품질**을 보였다(brain-masked 슬라이스 평균 SSIM 0.9140 vs 0.9126, PSNR 33.90 vs 33.78 dB).
전체 검증 7,334 슬라이스(464 볼륨)의 paired 비교에서 지표별로 슬라이스의 74~78%, 볼륨의
90~95%에서 우위였다(볼륨 단위 Wilcoxon signed-rank, p<0.001). 정성적으로 GRU 재구성에서
관찰되는 두개골 외부 배경의 ringing 아티팩트가 SS2D 에서는 관찰되지 않았다. 나아가 통제를
해제한 강화 SS2D 변형(게이팅·잔차 스택·병목 해제·coarse-scan)은 더 긴 학습 스케줄(80 epoch)을
소화해 최종 SSIM 을 0.9145 로 추가 개선했다(슬라이스의 54~56%에서 우위 — 유의하나 근소). 본
결과는 직접 도메인 변환형 재구성에서 RNN→SSM 치환이 DC 의 도움 없이, 대폭 적은 파라미터로,
일관된 품질 이득을 줌을 통제된 조건에서 보인 것으로, ETER-Net 계열의 자연스러운 차세대 확장
방향을 제시한다.

**Keywords**: MRI reconstruction; accelerated MRI; k-space; domain transformation; recurrent
neural network; state-space model; Mamba; parameter efficiency; fastMRI

---

## 1. 서론 (Introduction)

**[문단 ① — 연구 동기: MRI 는 왜 느린가, 그리고 느림의 비용]**
MRI 는 전리방사선 없이 뛰어난 연조직 대조도를 제공하는 핵심 진단 기법이지만, 근본적으로 "느린"
영상법이다. MRI 는 영상을 직접 찍지 않는다 — 스캐너가 실제로 수집하는 것은 영상의 2차원 푸리에
계수인 **k-space** 이며, k-space 는 통상 한 번의 반복시간(TR)에 한 줄(phase-encoding line)씩
순차적으로 채워진다. 완전한 영상을 얻으려면 이 줄들을 나이퀴스트 조건에 맞춰 빠짐없이 수집해야
하므로 한 시퀀스의 촬영이 수 분, 다중 contrast 검사 전체로는 수십 분에 이른다. 이 수집 속도는
경사자계 전환에 따른 말초신경 자극이나 RF 에너지 축적(SAR) 같은 물리·생리적 안전 한계로
제약되므로, 하드웨어 성능만으로는 근본적으로 단축할 수 없다 [19,22]. 긴 촬영은 곧 비용이다: 환자는
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
도메인으로의 변환 자체를 신경망이 학습한다 — AUTOMAP [17]이 완전연결층으로 이를 처음 보였고, DOTA-MRI [50]는 x-방향 1D IFT 를 해석적으로
선행해 그 파라미터 벽을 낮췄으며, ETER-Net [1]은 그 자리를 양방향 RNN 으로 대체해 파라미터를 크게 줄였으며, k-space CNN 을
경유하는 교차 도메인 KIKI-net [18]도 같은 문제의식을 공유한다. 이 밖에 score 기반 생성모델
(diffusion) 계열 [25,26]이 최근 세 번째 축으로 부상했다. 공정한 비교의 기반으로는 대규모 raw
k-space 공개 데이터셋 fastMRI [19]와 그 챌린지 [20,21]가 표준 벤치마크로 자리잡았다.

**[문단 ④ — 남은 문제: 도메인 변환 자리의 시퀀스 모델]**
본 연구는 두 번째 계열, 그중에서도 ETER-Net 계열 [1,2,51,3]의 심장부인 **도메인 변환 시퀀스 모델**
에 주목한다. ETER-Net 의 bi-RNN(GRU)은 k-space 행/열을 순차로 읽어 이미지 특징으로 변환하는데,
flatten-reshape 구조 탓에 파라미터가 비대해지고(본 세팅 668M), 순차 의존으로 병렬화가 제한된다.
한편 선택적 상태공간모델 Mamba [30]는 입력 의존 상태 전이로 장거리 의존을 **선형 복잡도**로
모델링하며, 4방향 selective scan 으로 2차원에 확장한 SS2D(VMamba) [31] 이후 비전 과제에서
RNN/Transformer 의 대안으로 빠르게 자리잡았다. MRI 재구성에도 Mamba 적용 연구가 이미 다수
존재하지만 [32–40], 이들은 모두 **이미지 도메인 prior/정규화기 또는 unrolled 백본** 자리에 SSM
을 넣는 새 아키텍처 제안이다. 본 논문에서 **"도메인 변환 자리"란 언더샘플된 k-space 를 입력받아
이미지 도메인 특징을 직접 출력함으로써 역푸리에 변환의 역할 자체를 학습으로 대체하는 모듈**
(입력=k-space, 출력=이미지 도메인)을 가리킨다 — 이 자리의 RNN 을 SSM 으로 1:1 치환하면 무엇이
달라지는가에 대한 통제 비교는, **우리가 아는 한(to our knowledge), 문헌에 없다**. 우리의 선행 내부 실험(ViT 인코더
하이브리드 [3] 유사 구조)에서 GRU(+U-Net 후처리) 대 SS2D(+DC) 비교는 brain-masked SSIM 0.9084
vs 0.9083 의 사실상 동률(dead-heat)로 끝났으나, 이 비교는 DC 유무와 후처리 구조가 시퀀스 모델
종류와 얽힌 confound 를 안고 있었다. 본 연구는 confound 를 제거한 순수 골격에서 질문을 격리한다:
**"직접 도메인 변환 자리에서, Mamba 는 GRU 보다 나은가?"**

**[문단 ⑤ — 기여와 논문 구성]**
본 논문의 기여는 세 가지다.

1. **단일 변수 통제 비교**: ETER-Net 골격(ViT 없음, DC 없음)에서 시퀀스 모델만 GRU↔SS2D 로
   교체한 통제 실험으로, **21× 적은 파라미터**(31M vs 668M)의 SS2D 가 표준 4지표 전부·
   matched-epoch 전 구간(wire-to-wire)·검증 슬라이스의 74~78%(볼륨의 90~95%)에서 동등 이상
   — 실제로는 일관 우위 — 임을 보인다.
2. **DC 무관성 실증**: DC 가 전혀 없는 세팅에서의 완승으로, 선행 dead-heat 에 대한 "SS2D 는 DC
   덕"(DC-crutch) 가설을 반박한다. 아울러 도메인 변환형 골격에 종단 single soft-DC 를 붙이는
   것이 unrolled 문헌의 DC 관행 [8–11,13–16]과 구조적으로 다름을 정리하고 no-DC 설계를 정당화한다.
3. **통제 해제 시의 상한 탐색**: 게이팅·잔차 스택·병목 해제·coarse-scan 다운샘플을 더한 강화
   SS2D 로 더 긴 학습 스케줄(80 epoch)을 소화해 최종 품질을 추가로 근소 개선한다.
   matched-epoch 기준의 정직한 해석(§4.3)을 함께 제시한다.

이하 §2 는 관련 연구, §3 은 골격·변형·학습/평가 프로토콜, §4 는 결과, §5 는 고찰, §6 은 결론이다.

## 2. 관련 연구 (Related Works)

### 2.1 직접 도메인 변환 (direct k-space-to-image) 계열

AUTOMAP [17]은 센서→이미지 매핑을 완전연결층으로 통째로 학습할 수 있음을 보였으나 해상도 제곱에
비례하는 파라미터가 실용의 벽이었다. DOTA-MRI [50]는 주파수-인코딩 방향 1D IFT 를 해석적으로
선행하고 phase-encoding 방향 1D 전역 변환만 학습해 이 벽을 O(N²)→O(N)으로 낮췄다. ETER-Net [1]은 이 변환을 수평/수직 양방향 RNN 두 개로
분해해 파라미터를 낮추고 CNN(U-Net)에 de-aliasing 을 맡기는 구조로, R=4 에서 SSIM 0.931 을
보고했으며 **명시적 DC 블록이 없다**. 이후 radial 등 non-Cartesian 궤적으로 확장되었고 [2], folded image 를 보조 입력으로 더해
랜덤·불규칙 궤적과 R∈{4,8}에서 안정성을 높인 dual-input ETER-net [51]으로 이어졌으며,
최근에는 ViT 인코더와의 하이브리드 [3]가 제안되어 BiRNN 의 k-space 순차 처리가 고가속·랜덤
샘플링 강건성의 핵심임이 보고되었다. k-space 도메인 CNN 을 포함하는 교차 도메인 계열로는
KIKI-net [18]이 있다. **본 연구의 위치**: [1,51,3]의 골격을 유지한 채 도메인 변환 모듈만 교체하는
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
MambaRecon [32], DH-Mamba [33], CAM [35], HiFi-Mamba [36], 불확실성 추정을 겸한 MambaMIR
[39], 다중 모달 융합 MMR-Mamba [40], unrolled 백본으로서의 MambaRoll [34], SO-Mamba [38],
연산자 학습 관점의 LMO [37] 등이다. 특히 DH-Mamba [33]은 이중 도메인 구조의 k-space 브랜치에서
SSM 스캔을 수행하며 k-space 직접 스캔의 스펙트럼 파괴 위험을 지적했다 — 그러나 그 구조에서
k-space↔이미지 도메인 사이의 이동은 여전히 명시적 (i)FFT 가 담당하고, SSM 은 각 도메인 안의
보정(prior) 역할이다. **차별점**: 이들 연구는 새 아키텍처 제안과 SOTA 경쟁이 목적이고, SSM 의
자리는 도메인 내부의 prior/정규화기다. 본 연구는 **기존 골격에서 RNN↔SSM 치환 효과를 격리하는
통제 실험**이 목적이며, 도메인 변환(k→image) 자리 — 즉 (i)FFT 의 역할 자체를 학습하는 자리
(§1 의 조작적 정의) — 의 SSM 은 우리가 아는 한 [32–40] 어디에도 없다.

### 2.4 본 연구의 위치 (비교 표)

| 계열 | 대표 문헌 | 시퀀스/전역 모듈의 자리 | DC | 본 연구와의 관계 |
|---|---|---|---|---|
| 직접 도메인 변환 | AUTOMAP [17], DOTA-MRI [50], **ETER-Net 계열 [1,2,51,3]**, KIKI-net [18] | **k-space→image 변환 그 자체** | 없음(원 논문 기준) | 본 연구의 골격 — 변환 모듈의 세대 교체를 검증 |
| Unrolled + DC | [8–11,13–16] | 반복 내부의 작은 정규화 unit | 매 반복 interleave | 구조가 달라 직접 비교 대상 아님 (§5-3) |
| Mamba-MRI | [32–40] | 이미지/이중 도메인 prior, unrolled 백본 | 대부분 있음 | SSM 을 쓰지만 **자리가 다름** |
| **본 연구** | — | **도메인 변환 자리의 RNN↔SSM 1:1 치환** | 없음 (통제) | 우리가 아는 한 문헌에 없는 통제 비교를 채움 |

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
풀용량을 유지한 채 epoch 당 학습시간을 통제판과 비슷한 수준으로 눌러(2.84 vs 3.07 h/ep — 측정
정의·실행환경 주의는 §4.5 Table 5), 실험 기간 내에 epoch 50→80 연장이 가능했다. 총 파라미터
~33M(SSM 스택 ~2M). 학습 위생으로 Mamba 상태 파라미터
(`A_log`, `D`)는 weight-decay 에서 제외했다.

### 3.4 데이터셋과 언더샘플링 프로토콜

| 항목 | 값 |
|---|---|
| 데이터 | fastMRI brain multicoil [19], **혼합 contrast** (AXT1/AXT1POST/AXT1PRE/AXT2/AXFLAIR) |
| 규모 | NYU fastMRI 공식 brain multicoil 배포본의 확보 서브셋 — train 구획 4,108 파일 / 65,028 슬라이스(확보 4,110개 중 `reconstruction_rss` 부재 2개 제외) · val 구획 464 파일 / 7,334 슬라이스(확보분 전부 사용, 제외 0). 공식 train/val 구획 분리를 그대로 따름(구획 간 교차 없음) |
| 정답(GT) | 데이터셋 제공 RSS 재구성(`reconstruction_rss`)을 384×384 center-crop/zero-pad |
| 전처리 | full k-space → iFFT(ortho) → 이미지 도메인 384×384 crop/pad → 재-FFT 로 384² k-space 유도 (retrospective) |
| 코일 | 앞 16개 코일 사용(초과분 절단, 부족분 zero-fill) → 실/허수 분리 32채널 — 코일 압축(SCC/GCC) 대신 재현 단순성을 위한 선택(§5-(6)) |
| 언더샘플링 | R=4 equispaced Cartesian 1D 마스크, 중앙 ACS 8% (train 은 매 샘플 offset 랜덤, val 은 고정) |
| 증강 | 수평/수직 flip p=0.5 — flip 후 FFT 재계산으로 k-space 물리 정합 유지 |

### 3.5 손실 함수와 평가지표 (brain-masked)

- **Brain mask**: Otsu 임계 × 0.4 + 최대 연결성분(largest CC). 배경(영상의 절반 이상)이 지표를
  부풀리는 것을 차단한다.
- **손실**: masked L1 + (1 − SSIM).
- **평가 지표 정의** (전부 슬라이스 단위, 정답 y·재구성 x̂·마스크 m):
  - **SSIM**: skimage `structural_similarity` 를 전체 영상에서 계산(기본 윈도, data_range =
    마스크 내부 y 의 max−min)한 뒤 **SSIM map 을 마스크 픽셀에서만 평균**.
  - **PSNR** = 20·log₁₀( max_m(y) / √MSE_m ), MSE_m = Σ((x̂−y)²·m)/Σm — peak 는 슬라이스별
    마스크 내 최댓값. **NMSE** = Σ((x̂−y)²·m)/Σ(y²·m). **L1** = Σ(|x̂−y|·m)/Σm.
  - 모든 영상은 ×10⁶ 스케일된 RSS magnitude 기준 — 즉 표의 L1 8.88 은 원 신호 단위로
    8.88×10⁻⁶ 에 해당한다.
  - **본문·표의 모든 수치는 슬라이스 단위 통계로 통일**한다(학습 로그의 배치 풀링 수치는 학습
    곡선(Fig. 2) 재현용으로만 사용).
- **주 지표는 SSIM**(fastMRI 챌린지 표준 [20,21]), 나머지는 병기한다. 본 지표는 뇌 영역
  한정이므로, 배경을 포함하는 fastMRI 공식 프로토콜(320 crop·raw)의 리더보드/문헌 수치와는
  좌표계가 달라 직접 대조하지 않는다. 한편 마스킹은 GRU 의 두개골 외부 아티팩트(§4.2)에 벌점을
  주지 않으므로, 본 비교 맥락에서는 **GRU 에 유리한 보수적 선택**이다.
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

fastMRI **brain leaderboard 사전학습 U-Net·E2E-VarNet [11]** 을 두 가지 프로토콜로 추론 평가해
참고 기준선으로 제시한다. (i) **동일 파이프라인** — 우리 모델과 완전히 같은 측정값(같은 슬라이스·
R4 마스크·16코일·384² 재-FFT·GT·brain mask)을 입력한다. (ii) **네이티브 프로토콜** — fastMRI 공식
추론 규약[19]을 그대로 따라 전체 코일·native 해상도 k-space 를 쓰고 ismrmrd 헤더의 reconSpace
크기로 중앙 crop 한다(취득 구간 밖 마스크 0, 감도맵 ACS 폭은 마스크에서 자동 검출). (i)은 입력
동일성을, (ii)는 leaderboard 가중치의 학습 조건 근접성을 각각 확보하며, 둘의 차이가 곧 전처리
domain shift(16코일 절단·해상도/FOV)의 크기다. 두 모델의 출력 스케일은 자체 정규화 기준이므로
지표 계산 전 per-slice 최소제곱 스케일 정합을 적용한다(우리 모델은 α≈1).

**두 가지 캐비엇을 명시한다.** 첫째, **본 검증셋 전체가 두 기준선의 학습 데이터에 포함돼 있다** —
공식 저장소는 leaderboard 모델에 대해 "The leaderboard model was trained where the `train` split
included both the `train` and `val` splits from the public data" 라고 명시한다[19]. 따라서 기준선
수치는 낙관적으로 편향되며, `train` 만으로 학습한 본 모델들과의 우열 판정은 성립하지 않는다.
둘째, 사전학습 가중치는 R4·R8 혼합 학습본이고 원 학습분포가 본 전처리와 다르다. 이 두 이유로
기준선은 **절대 우열이 아닌 참고선**으로만 읽어야 한다.

(구현 주의: 데이터로더가 코일을 16채널로 zero-pad 하므로 16코일 미만 볼륨—본 검증셋 464개 중
197개, 슬라이스 3,122/7,334—을 VarNet 에 그대로 넣으면 감도추정 U-Net 의 채널별 정규화에서
std=0 → NaN 이 발생해 슬라이스 전체가 무효가 된다. 실측 코일만 전달해 회피했다. VarNet 은
정규화·선형 DC·RSS 로 구성돼 양의 스칼라에 대해 positively homogeneous 이므로 입력 스케일
선택은 지표에 영향을 주지 않는다.)

### 3.8 통계 분석

두 모델이 동일 슬라이스를 재구성하는 **paired 설계**다. 지표별 paired 차이에 Wilcoxon
signed-rank 검정을 적용하고, 효과크기로 **우위 슬라이스 비율**(proportion of slices favoring;
통계학의 probabilistic index [46]에 해당, 임상시험의 win-ratio 계열 [47]과 동족)을 함께
보고한다. 영어 원고 표기 예: *"SS2D achieved higher SSIM in 78.2% of slices (5,737/7,334;
Wilcoxon signed-rank test, p < 0.001)."* p 값은 저널 관례에 따라 p<0.001 로 표기한다(원값은
저장소 산출물에 보존). 검정은 양측, 유의수준 0.05 다. **같은 볼륨의 슬라이스들은 독립이 아니므로
(클러스터 구조), 슬라이스 단위 분석과 볼륨 단위 분석을 병행한다**: 볼륨별 평균 paired 차이에
대한 Wilcoxon signed-rank(n=464)와 볼륨 단위 우위 비율을 함께 보고하고, 슬라이스 단위 우위
비율과 평균 차이의 95% CI 는 볼륨 클러스터 부트스트랩(2,000회 재표집)으로 구한다. 4개 지표는
상관이 높아 다중비교 보정(Bonferroni ×4)을 적용해도 본문의 모든 유의성 결론은 불변이다.

## 4. 결과 (Results)

### 4.1 통제 비교 — SS2D 의 일관 우위 (Table 1, 2; Fig. 2)

**Table 1. Best checkpoint 기준 (val 전체 7,334 슬라이스, brain-masked, 슬라이스 단위 평균).**

| | best epoch | SSIM | PSNR (dB) | NMSE | L1 (×10⁻⁶) | params |
|---|---:|---:|---:|---:|---:|---:|
| **SS2D (통제판)** | 48/50 | **0.9140** | **33.90** | **0.00438** | **8.883** | **31M** |
| GRU | 50/50 | 0.9126 | 33.78 | 0.00449 | 9.002 | 668M |

- **Matched-epoch 전 구간 우위(wire-to-wire)**: 25개 검증 지점(ep2~50)에서 SS2D 가 SSIM 기준 전
  지점 ≥ GRU(Δ +0.0000~+0.0055; 동률은 ep14 1지점뿐), PSNR 기준 전 지점 우위였다. ViT
  하이브리드 선행 실험에서 관찰됐던 후반 역전(crossover)은 없었다. (Fig. 2 — 단일 시드의 단일
  궤적임을 캡션에 명시; §5-(6))
- **Per-slice paired 검증** (val 7,334 슬라이스 전수, Wilcoxon signed-rank; 차이 분포는 Fig. 4a):

**Table 2. Per-slice paired 비교** (Δ 는 항상 양수 = SS2D 우위 방향, NMSE/L1 부호 반전 —
Fig. 4 규약과 동일. CI 는 볼륨 클러스터 부트스트랩 2,000회, §3.8).

| 지표 | GRU mean±std | SS2D mean±std | Δ 중앙값 (IQR) | 우위 슬라이스 [95% CI] | 우위 볼륨 | p(볼륨, n=464) |
|---|---:|---:|---:|---:|---:|---:|
| SSIM | 0.9126±0.0894 | 0.9140±0.0890 | +0.0013 (+0.0002, +0.0025) | **78.2%** [76.8, 79.7] | **94.8%** | <0.001 |
| PSNR (dB) | 33.78±2.59 | 33.90±2.63 | +0.12 (−0.01, +0.26) | **73.8%** [72.1, 75.5] | **89.9%** | <0.001 |
| NMSE — Δ×10⁻⁵ | 0.00449±0.00525 | 0.00438±0.00526 | +8.9 (−0.6, +20.6) | **73.8%** [72.0, 75.5] | **90.1%** | <0.001 |
| L1 (×10⁻⁶) | 9.00±3.05 | 8.88±3.05 | +0.115 (+0.005, +0.237) | **76.1%** [74.4, 77.8] | **90.5%** | <0.001 |

즉 aggregate 우위가 소수 슬라이스·소수 볼륨에 의한 것이 아니라 **대다수 슬라이스(74~78%)와
압도적 다수 볼륨(90~95%)에서 일관**된다. 슬라이스 평균 Δ 의 클러스터 부트스트랩 95% CI 도 4지표
전부 0 을 배제한다(예: ΔSSIM +0.0014 [+0.0013, +0.0015]). contrast 서브그룹별 일관성은 부록 C
Table S1(5개 contrast 전부에서 모든 지표 우위 비율 ≥68.7%).

### 4.2 정성 비교 — GRU 의 배경 ringing (Fig. 3)

4-way 시각화(GT / 사전학습 U-Net 참고 기준선(§3.7) / GRU / SS2D)에서 **GRU 는 두개골 바깥
배경에 반복적 ringing/줄무늬 아티팩트**를 보이는 반면 SS2D 의 해당 영역은 깨끗했다(검토한
시각화 슬라이스에서 일관되게 관찰 — 검토 슬라이스 수 명시와 마스크 외부 잔차의 정량화는 보완
예정 ✎부록 D P1-5). brain-mask 밖이라 정량 지표에는 반영되지 않는 순수 정성적 차이로(§3.5 의
보수적 마스킹 논점 참조), GRU 재구성이 관심영역 밖에서 덜 안정적임을 시사한다.

### 4.3 강화 SS2D — 상한 탐색 (Table 3; Fig. 2, 4)

**Table 3. 강화 SS2D vs 통제판 (val 전체 7,334 슬라이스, brain-masked, 슬라이스 단위 평균).**

| | SSIM | PSNR (dB) | NMSE | L1 (×10⁻⁶) |
|---|---:|---:|---:|---:|
| **강화 SS2D (best ep78/80)** | **0.9145** | **33.92** | 0.00439 | **8.879** |
| SS2D 통제판 (best ep48/50) | 0.9140 | 33.90 | **0.00438** | 8.883 |
| GRU (best ep50/50) | 0.9126 | 33.78 | 0.00449 | 9.002 |

- Per-slice 우위 슬라이스 비율: vs 통제판 SS2D **54~56%**(4지표, 클러스터 부트스트랩 95% CI
  하한 52.5%), vs GRU **78~82%**(완승). 볼륨 단위로도 4지표 전부 유의하다(우위 볼륨 55.0~66.4%,
  Wilcoxon n=464, 최대 p=0.002). 차이 분포는 Fig. 4b.
- **이득의 크기는 작고, 지표·서브그룹에 따라 균일하지 않다**: 평균 차이의 95% CI 가 0 을
  배제하는 것은 SSIM 뿐이고(ΔSSIM +0.0005 [+0.0003, +0.0006]), PSNR·L1 의 평균 차이는 CI 가
  0 을 포함하며, NMSE 는 평균 기준 사실상 동률이다(Δ −1.0×10⁻⁵ — Table 3 에서 통제판이 근소
  우세) — 순위 기반(중앙값·우위 비율·Wilcoxon)으로는 4지표 전부 강화판 우위. contrast
  서브그룹에서도 SSIM 우위 비율이 AXFLAIR 67.8% ~ AXT1 48.2%(역전)로 불균일하다(부록 C
  Table S1).
- **정직한 해석**: matched-ep50 시점 강화판 SSIM 은 0.9130 으로 통제판 best(0.9140)에 미달하며,
  통제판 best 에 도달한 것은 연장 구간의 ep64(동률)~ep66(상회)이다. 즉 "같은 학습량에서 더
  좋다"가 아니라 **"더 긴 스케줄(80ep)을 소화해 최종 품질을 근소하게 넘었다"**가 정확한 서사다
  (best 도달 wall-clock 강화판 ≈187 h(ep66×2.84) > 통제판 ≈147 h(ep48×3.07); h/ep 측정 정의는
  §4.5). 아키텍처 강화 자체의 순수 이득은 근소하다.
- Coarse-scan(ds=3) 다운샘플은 품질을 해치지 않았다: ep40 시점 열위였다가 후반 cosine anneal
  구간에서 역전, 최종 상회했다.

### 4.4 기준선 비교 — U-Net / E2E-VarNet **[결과 삽입 예정]**

§3.7 의 두 프로토콜(동일 파이프라인 / 네이티브)로 사전학습 U-Net·E2E-VarNet [11]을 추론 평가한
결과(표준 4지표 mean±std + 본 모델들과의 우위 슬라이스 비율)를 **Table 4** 로 여기에 삽입한다.
해석은 §3.7 의 두 캐비엇 — **검증셋이 기준선의 학습 데이터에 포함**, 전처리 domain shift — 아래에서
"참고선"으로 한정한다.
**[스크립트 준비 완료: `v8_eter_pure/eval_paired_baselines.py` + `native_protocol.py`. 2026-08-20
층화 표본(299 슬라이스, contrast × 코일수) CPU 예비 실행 중 — 전체 7,334 슬라이스 GPU 풀런은
radapt 완주(~08-25) 후.]**

### 4.5 효율 비교 (Table 5)

**Table 5. 파라미터·시간 효율.** 학습 h/ep 는 5-epoch 체크포인트 저장 간격의 중앙값(검증 포함
wall-clock, 재시작·정전으로 인한 outlier 구간 제외)으로 산출했다.

| | params | 학습 h/ep (검증 포함)† | 추론 ms/slice | peak VRAM |
|---|---:|---:|---:|---:|
| GRU | 668M | 2.41 | ✎측정 예정 | ✎ |
| SS2D 통제판 | 31M | 3.07 | ✎ | ✎ |
| 강화 SS2D | ~33M | 2.84‡ | ✎ | ✎ |

† v8 두 arm(GRU/SS2D 통제판)은 동일 실행환경에서 학습되어 상호 비교 가능하다 — **epoch 당
학습시간은 GRU 가 더 빠르다**(순차 RNN 이지만 cuDNN 최적화 이점). 본 논문의 효율 주장은
학습 속도가 아니라 **파라미터(21×)와 동등 이상 품질**에 있다.
‡ 강화판은 v8 완주 후 컨테이너/데이터로더 환경 개선을 거쳐 학습되어, v8 과의 wall-clock 직접
비교에는 환경 차이가 섞여 있다(명목 2.84 < 3.07 이나 이 차이의 해석에는 주의). 추론
ms/slice·VRAM 은 radapt 완주 후 측정해 채운다(✎부록 D P1-3).

## 5. 고찰 (Discussion)

**(1) DC-crutch 가설의 반박.** ViT 하이브리드 선행 비교의 dead-heat 는 "SS2D 의 성능은 DC
덕분"이라는 해석을 허용했다. 그러나 DC 를 완전히 제거한 본 통제 세팅에서 SS2D 는 오히려 더
확실하게 이겼다. Mamba 의 이득은 DC 와 무관한 시퀀스 모델링 능력 자체에서 나온다.

**(2) 파라미터 효율의 해석.** 668M 의 GRU 를 31M 모델이 이긴다(21×↓). 도메인 변환 자리 RNN 의
flatten-reshape 구조가 비효율의 근원이며, SSM 은 같은 자리를 선형 복잡도·저용량으로 대체한다.
이는 AUTOMAP [17]→DOTA-MRI [50]→ETER-Net [1]이 밟았던 "같은 기능, 더 효율적인 모듈로" 궤적의 다음 단계로 읽을
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
- best checkpoint 선택과 최종 보고가 같은 val 세트를 공유한다(fastMRI 관행이나 명시해 둔다) —
  별도 내부 test 분할은 두지 않았다.
- 코일 압축(SCC/GCC) 대신 앞 16코일 절단을 채택했다(재현 단순성) — 코일 압축과의 결합은 향후
  과제다.
- 시퀀스 모듈을 제거한 **U-Net-only 기준**(치환 이득 해석의 분모)은 아직 없다 — 학습·평가해
  보완할 예정이다(✎부록 D P0-2).

**(7) Novelty 포지셔닝 (솔직한 자기 평가).** Mamba-MRI 아키텍처 자체는 이미 성숙 분야다
[32–40]. 본 논문의 가치는 새 아키텍처가 아니라 (a) 도메인 변환 자리에서의 1:1 통제 치환 실험,
(b) no-DC 조건의 DC 무관성 실증, (c) ETER-Net 계열 [1–3,51]의 직접 후속이라는 점이다. 투고 시 이
프레임을 유지해야 리뷰 방어가 가능하다.

## 6. 결론 (Conclusions)

ETER-Net 골격의 도메인 변환 자리에서 bi-GRU 를 SS2D 로 치환하는 것만으로 — DC 없이, 21× 적은
파라미터로 — 표준 4지표 전부·matched-epoch 전 구간·검증 슬라이스 대다수에서 일관된 품질 향상을
얻었다. 게이팅·깊이·병목 해제를 더한 강화 SS2D 는 epoch 당 시간을 통제판과 비슷한 수준으로
유지한 채 더 긴 스케줄을 소화해 최종 품질을 근소하게 추가 개선했다. 직접 도메인 변환형 재구성의
차세대 시퀀스 모델로서 SSM 은 RNN 의 자연스러운
대체재이며, 가속률 일반화(radapt)와 타 궤적·부위 확장이 후속 과제다.

---

## 후반부 항목 (MDPI 양식)

- **Author Contributions**: (CRediT 분류로 작성 예정 — 저자 구성 확정 후, 부록 B-1)
- **Funding**: (미정 — 확인 필요)
- **Institutional Review Board Statement**: 본 연구는 공개 비식별 데이터셋 fastMRI [19,49]의
  2차 이용만을 포함한다. (NYU fastMRI 데이터 사용 약정 준수; 별도 IRB 심의 면제 해당 여부
  문구는 투고 저널 양식에 맞춰 확정)
- **Informed Consent Statement**: Not applicable (공개 비식별 데이터셋의 2차 이용).
- **Data Availability Statement**: fastMRI 데이터셋은 https://fastmri.med.nyu.edu 에서 신청
  가능. 학습·평가 코드는 게재 시 GitHub 공개 예정.
- **Conflicts of Interest**: 없음(예정 — 확인 필요).
- **Acknowledgments**: Data used in the preparation of this article were obtained from the NYU
  fastMRI Initiative database (fastmri.med.nyu.edu) [19,49]. NYU fastMRI investigators provided
  the data but did not participate in the analysis or writing of this article. (✎ fastMRI DUA 의
  공식 acknowledgment 문구 원문과 대조 확인)
- **셀프 체크**: 투고 전 CLAIM 체크리스트 [48] 자체 점검(자료집 §5 권고).

---

## 참고문헌 (번호 ↔ `paper/references.bib` 키 매핑)

*전 항목 Crossref/arXiv 조회로 서지 확정(2026-08-18). 잔여 확인(✎) 2건: [33] IEEE TCSVT 권호,
[34] 저널판 권호(게재 확정 시) — 아직 arXiv 인용으로 유효.*

**직접 계보 (ETER-Net 계열)**
1. Oh, Kim, Chung, Han & Park (2021). A k-space-to-image reconstruction network for MRI using recurrent neural network. *Medical Physics* 48(1):193–203. doi:10.1002/mp.14566. `oh2021eternet`
2. Oh, Chung & Han (2022). An end-to-end recurrent neural network for radial MR image reconstruction. *Sensors* 22(19):7277. doi:10.3390/s22197277. `oh2022radial`
3. Oh (2025). A hybrid Vision Transformer-BiRNN architecture for direct k-space to image reconstruction in accelerated MRI. *Journal of Imaging* 12(1):11. doi:10.3390/jimaging12010011. `oh2025vitbirnn` ★직접 선행 — 단독 저자

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
15. Karkalousos et al. (2022). Assessment of data consistency through cascades of independently recurrent inference machines (CIRIM). *Phys. Med. Biol.* 67(12):124001. doi:10.1088/1361-6560/ac6cc2. `karkalousos2022cirim`
16. Hammernik et al. (2021). Systematic evaluation of iterative deep neural networks for fast parallel MRI reconstruction. *MRM* 86(4):1859–1872. doi:10.1002/mrm.28827. `hammernik2021systematic`
17. Zhu et al. (2018). Image reconstruction by domain-transform manifold learning (AUTOMAP). *Nature* 555:487–492. `zhu2018automap`
18. Eo et al. (2018). KIKI-net. *MRM* 80(5):2188–2201. `eo2018kiki`

**fastMRI 벤치마크**
19. Zbontar et al. (2018). fastMRI: an open dataset and benchmarks for accelerated MRI. arXiv:1811.08839. `zbontar2018fastmri`
20. Knoll et al. (2020). Overview of the 2019 fastMRI challenge. *MRM* 84(6):3054–3070. `knoll2020advancing`
21. Muckley et al. (2021). Results of the 2020 fastMRI challenge. *IEEE TMI* 40(9):2306–2317. `muckley2021results`
49. Knoll et al. (2020). fastMRI: a publicly available raw k-space and DICOM dataset of knee images for accelerated MR image reconstruction using machine learning. *Radiology: AI* 2(1):e190007. `knoll2020fastmri` — 데이터 사용 요건상 [19]와 병행 인용 (Acknowledgments 참조)

**계보 보강 (2026-09-01 추가 — Crossref 확정)**
50. Eo, Shin, Jun, Kim & Hwang (2020). Accelerating Cartesian MRI by domain-transform manifold learning in phase-encoding direction (DOTA-MRI). *Med. Image Anal.* 63:101689. doi:10.1016/j.media.2020.101689. `eo2020dota`
51. Oh, Chung & Han (2024). Domain transformation learning for MR image reconstruction from dual domain input (dual-input ETER-net). *Comput. Biol. Med.* 170:108098. doi:10.1016/j.compbiomed.2024.108098. `oh2024dualinput` — ★ v8 백본 dual-input(cat(seq출력, aliased)) 의 직접 전신 + 랜덤/불규칙 궤적·R∈{4,8} (radapt 위치설정: 부록 B-3)

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
32. Korkmaz & Patel (2025). MambaRecon: MRI reconstruction with structured state space models. *WACV*. arXiv:2409.12401. `korkmaz2025mambarecon`
33. Meng et al. (2026). DH-Mamba: exploring dual-domain hierarchical state space models for MRI reconstruction. *IEEE TCSVT* 36(3):3290–3305. doi:10.1109/TCSVT.2025.3614828 (arXiv:2501.08163, v1 제목 DM-Mamba). `dhmamba2025`
34. Kabas et al. (2026). Physics-driven autoregressive state space models for medical image reconstruction (MambaRoll). *IEEE TMI* (early access). doi:10.1109/TMI.2026.3716153 (arXiv:2412.09331). `kabas2024mambaroll` (✎ 권호/페이지 배정 시 갱신)
35. Meng, Yang, Fu, Song & Shi (2026). Image content matters: an image content aware state space model for accelerated MRI reconstruction (CAM). *Proc. AAAI* 40(10):8025–8033. doi:10.1609/aaai.v40i10.37748. `meng2026cam`
36. Chen et al. (2025). HiFi-Mamba: dual-stream W-Laplacian enhanced Mamba for high-fidelity MRI reconstruction. arXiv:2508.09179. `chen2025hifimamba`
37. Li et al. (2025). LMO: linear Mamba operator for MRI reconstruction. *CVPR*. `li2025lmo`
38. Fang et al. (2026). SO-Mamba: state-ownership Mamba for unrolled MRI reconstruction. arXiv:2605.22031. `somamba2026`
39. Huang et al. (2025). Enhancing global sensitivity and uncertainty quantification in medical image reconstruction with Monte Carlo arbitrary-masked Mamba (= MambaMIR 저널판; preprint arXiv:2402.18451). *Med. Image Anal.* 99:103334. doi:10.1016/j.media.2024.103334. `huang2024mambamir`
40. Zou et al. (2025). MMR-Mamba: multi-modal MRI reconstruction with Mamba and spatial-frequency information fusion. *Med. Image Anal.* 102:103549. doi:10.1016/j.media.2025.103549 (preprint arXiv:2406.18950). `zou2024mmrmamba`

**신뢰성·임상 검증**
41. Antun et al. (2020). On instabilities of deep learning in image reconstruction. *PNAS* 117(48):30088–30095. `antun2020instabilities`
42. Gottschling et al. (2025). The troublesome kernel. *SIAM Review* 67(1). `gottschling2025troublesome`
43. Recht et al. (2020). Using deep learning to accelerate knee MRI at 3T: interchangeability study. *AJR* 215(6):1421–1429. `recht2020interchangeability`
44. Johnson et al. (2023). Deep learning reconstruction enables prospectively accelerated clinical knee MRI. *Radiology* 307(2):e220425. `johnson2023prospective`
45. Radmanesh et al. (2022). Exploring the acceleration limits of deep learning VarNet-based 2D brain MRI. *Radiology: AI* 4(6):e210313. `radmanesh2022limits`

**통계·보고 지침**
46. Acion et al. (2006). Probabilistic index. *Statistics in Medicine* 25(4):591–602. `acion2006probabilistic`
47. Wang & Pocock (2016). A win ratio approach to comparing continuous non-normal outcomes. *Pharmaceutical Statistics* 15(3):238–245. doi:10.1002/pst.1743. `wang2016winratio`
48. Mongan et al. (2020). CLAIM checklist. *Radiology: AI* 2(2):e200029. `mongan2020claim`

---

## 부록 A. 그림·표 ↔ 저장소 산출물 매핑

| 논문 요소 | 소스 (저장소 경로) | 상태 |
|---|---|---|
| Fig.1 아키텍처 다이어그램 (a: 공통 골격+양 arm, b: 강화 SS2D) | `paper/figs/fig1_architecture.{png,pdf}` — 생성 스크립트 `paper/make_fig1_architecture.py` | ✅ 2026-08-18 (색 규약 Fig.4 와 통일: blue=SS2D, red=GRU, 회색=공유) |
| Fig.2 학습 곡선 (GRU vs SS2D vs 강화판, SSIM/PSNR) | `results/eval/v9_unleashed/curves_v9_vs_v8.png` | ✅ (논문용 재도색 권장) |
| Fig.3 4-way 정성 비교 + 배경 ringing | `results/vis/v8_pure_eternet_compare/compare_*.png` | ✅ (슬라이스 선별 필요) |
| Fig.4 per-slice 우위 비율/차이 분포 | `paper/figs/fig4_per_slice_distribution.{png,pdf}` — 생성 스크립트 `paper/make_fig4_per_slice.py` (입력 `results/eval/v9_unleashed/per_slice_paired_v9.csv`) | ✅ 2026-08-18 (2행×4지표 paired-Δ 히스토그램 + win-rate, 양수=치환/강화 우위 규약) |
| Tab.1·2·2b·3·S1 (결과 표 일체) | **자동 생성**: `paper/make_tables.py` → `paper/tables/*.{md,tex}` — per-slice CSV 단일 원천, median(IQR)·클러스터 부트스트랩 CI·볼륨 Wilcoxon 내장, seed 고정으로 초안 수치 자가검증(ALL PASS). `.tex` 는 LaTeX 투고용(MDPI 는 Word 도 허용 — 선택 사용) | ✅ 2026-08-20 |
| Tab.4 기준선(U-Net/E2E-VarNet) | `v8_eter_pure/eval_paired_baselines.py` + `v8_eter_pure/native_protocol.py` → `results/eval/baselines_384{,_sample300}/` | ⏳ 08-20 CPU 층화 표본(299) 실행 중 · 전체 GPU 풀런은 radapt 완주 후. 08-20 수정: zero-coil→NaN 회피(검증셋 42.6% 무효화 방지) + 네이티브 프로토콜 행 추가 |
| (참고) matched-epoch 표 | `results/eval/*/matched_epoch_table*.md` | ✅ (본문 서술로만 사용) |

## 부록 B. 교수님 상의 포인트 (초안 논외, 회의용)

1. **[3] (교수님 ViT-BiRNN, *J. Imaging* 2025)과의 관계 설정** — 본 논문을 [1,51,3]의 직접 후속
   (도메인 변환 모듈의 세대 교체)으로 프레이밍하는 안. 저자 구성·기여 서술에 영향.
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
   교수님 선호 확인 가치.) **문의 골자(외부 검토 제안, 08-18)**: *"We study brain MRI
   reconstruction with a parameter-efficient state-space model that cuts parameters 21× — would
   a controlled architecture-comparison study on accelerated brain MRI reconstruction fit the
   scope of this Special Issue?"* — 컴퓨팅 가속 요소(fp16 selective scan·coarse scan·단일 GPU
   학습)를 전면 배치.
3. **radapt(R 일반화) 포함 여부** — 08-18 학습 재개, **~08-25 완주 예상**. 포함 시
   "치환+강화+일반화" 3막 구성으로 기여가 두터워지나 투고 지연. 미포함 시 본 초안 그대로 + 후속
   논문 분리. (Bioengineering SI 마감 11-30 이면 포함 여유 있음.) **포함 시 위치설정(09-01 추가)**: 교수님
   dual-input ETER-net [51]이 folded-image 보조 입력으로 랜덤/불규칙 궤적·R∈{4,8}을 다룬 같은
   질문의 RNN-시대 해법이므로, radapt(mask-conditioning+multi-AR+DC)를 그 SSM-시대 직접 후속으로 서술. **외부 검토 의견(08-18)**:
   결과가 깨끗하면 결과 절 한 절로 압축 포함 권장 — "단일 가속률" 한계를 스스로 해소하는 이득이
   지연 비용보다 큼; 애매하면 미련 없이 후속 논문 분리.
4. **DC 서술 수위** — §5-(3)의 "문헌 비표준 + 학습 불안정" 근거를 본문에 둘지 부록으로 뺄지.
   **외부 검토 의견(08-18): 본문 유지 확정** — "왜 DC 가 없나"는 리뷰어의 확정 질문.
5. **단일 시드 한계** — 리뷰 방어용 멀티시드 재학습(비용 큼) 필요성 판단. **외부 검토 의견
   (08-18): 필요** — ΔSSIM 0.0014 가 시드 노이즈보다 큰지가 성립 조건. 전체 재학습이 부담이면
   축약 프로토콜(예: 25ep×3시드로 GRU/SS2D 순위 안정성만 확인) 절충 제안.
6. **composite 노출 수위 — 해결(2026-08-07 사용자 결정)**: 결과 표·본문 수치에서 전면 제거,
   §3.5 에 모델 선택 기준으로만 한 줄 서술. 모든 서사(wire-to-wire·연장구간 돌파)는 표준 지표로
   재검증 완료.
7. **(v2 신규) reader study** — 자료집 §3.2 가 권고하는 소규모 영상의학과 의사 평가를 넣을지.
   현재는 §5-(5)에서 한계로 정면 서술하는 전략. 의학 계열 저널로 갈 경우 재론. **외부 검토 의견
   (08-18)**: 공학 저널(Sensors/Bioengineering) 전제면 현행 전략 유지.

## 부록 C. 보충표 (Supplementary)

**Table S1. Contrast 서브그룹별 우위 슬라이스 비율** (per-slice CSV 재집계, 2026-08-18.
각 칸 = SSIM 기준 우위 비율 (4지표 범위)).

| Contrast | n (슬라이스) | SS2D vs GRU | 강화판 vs 통제판 |
|---|---:|---:|---:|
| AXFLAIR | 518 | 76.8% (71.0~76.8) | 67.8% (64.3~67.8) |
| AXT1 | 492 | 78.5% (68.7~78.5) | **48.2% (48.2~49.2)** |
| AXT1POST | 1,560 | 83.3% (77.6~83.3) | 54.3% (53.4~55.5) |
| AXT1PRE | 458 | 84.7% (80.3~84.7) | 52.8% (47.4~52.8) |
| AXT2 | 4,306 | 75.8% (72.6~75.8) | 56.0% (53.8~56.0) |

해석: **통제 비교(SS2D vs GRU)의 우위는 5개 contrast 전부에서 일관**된다(모든 지표 ≥68.7%,
SSIM 기준 전부 ≥75.8%) — 혼합 contrast 학습의 서브그룹 어디에서도 결론이 뒤집히지 않는다.
반면 **강화판의 근소 우위는 contrast 간 불균일**하다 — AXFLAIR 에서 가장 크고(67.8%),
AXT1 에서는 역전된다(48.2%) — §4.3 의 "유의하나 근소·불균일" 서술의 근거.

## 부록 D. 외부 검토 보고서(2026-08-18) 반영 현황

**P0 (투고 전 필수)**

| # | 항목 | 상태 |
|---|---|---|
| 1 | Table 4 기준선 | ⏳ radapt 완주(~08-25) 후 즉시 실행 |
| 2 | U-Net-only 정량 | 부분 — Fig.3 의 U-Net = leaderboard 사전학습본임을 §4.2 에 명시. **동일 파이프라인 U-Net-only 학습은 미실시** → §5-(6) 한계 명기 + radapt 후 학습·평가 예정 |
| 3 | 슬라이스 상관(클러스터) | ✅ 볼륨 단위 Wilcoxon(n=464)·우위 볼륨 비율·클러스터 부트스트랩 CI 병행(§3.8, Table 2, §4.3). 4지표 전부 볼륨 수준 유의 확인 |
| 4 | 표 수치 체계 통일 | ✅ 본문·표 전부 슬라이스 단위로 재계산·통일(§3.5 규정). 배치 풀링 수치는 Fig.2 재현용으로만 |
| 5 | Table 3 동일값 의심 | ✅ 해소 — 슬라이스 단위 재계산으로 구분됨(L1 8.879 vs 8.883; NMSE 는 실제로 강화판이 평균 근소 열위 — §4.3 에 정직 서술) |
| 6 | wall-clock 산식 | ✅ 재산출 — ckpt 간격 중앙값 기준 GRU 2.41 / SS2D 3.07 / 강화판 2.84 h/ep(Table 5). **기존 문서값(2.78/2.51)은 측정 기준 불일치로 폐기**. 부수 발견: v8↔강화판 사이 실행환경 개선(컨테이너/데이터로더)이 있어 wall-clock 직접 비교에 confound — Table 5 각주 명시 |
| 7 | 지표 정의 | ✅ §3.5 수식 수준 정의 + ×10⁶ 스케일 명시(코드 `eval_paired_v9.py` 대조) |
| 8 | 깨진 링크 2곳 | ✅ 수정 |
| 9 | ⚠ 서지 | ✅ **전건 확정** — 검토 5건 + Crossref/arXiv 직접 조회 8건+권호 보완([1] 48(1):193–203, [2] Sensors 22(19):7277, [3] 2025 확정·단독저자, [15] PMB 67(12), [16] MRM 86(4), [35] **AAAI 2026** 40(10):8025–8033, [36] arXiv:2508.09179, [39][40] **Med.Image.Anal. 저널판**(99:103334 / 102:103549), [47] 15(3):238–245). 잔여 ✎ 해소(09-01): [33] TCSVT 36(3):3290–3305 확정·[34] TMI early-access DOI 확정(권호 배정 대기) |
| 10 | fastMRI 인용/문구 | ✅ [49] 병행 인용 + Acknowledgments 신설(✎공식 문구 원문 대조 1건 잔여) |
| 11 | Informed Consent | ✅ 추가 |

**P1 (방어력 강화)**

| # | 항목 | 상태 |
|---|---|---|
| 1 | 멀티시드 | ⏸ 교수님 결정(부록 B-5 — 검토 의견·축약 프로토콜 반영) |
| 2 | 효율 프레임·CI | ✅ 초록·기여 재프레임("21×, 동등 이상") + Table 2 에 median(IQR)·클러스터 CI·볼륨 통계 |
| 3 | 효율 표 | 부분 — Table 5 신설(params·h/ep 실측). 추론 ms/slice·VRAM 은 GPU 확보 후(✎) |
| 4 | 표준 프로토콜 보충표 | 부분 — 보수적 마스킹 논점 §3.5 명시 ✅. 320-crop 표준 지표 보충표는 GPU 확보 후(✎) |
| 5 | ringing 정량화 | 부분 — 문구 완화·U-Net 정체 명시 ✅. 마스크 외부 잔차 정량·오류맵·N 명시는 GPU 확보 후(✎) |
| 6 | contrast 서브그룹 | ✅ Table S1(부록 C) + §4.1/§4.3 본문 연결 |
| 7 | "to our knowledge"·정의 명문화·[33] 구분 | ✅ §1④ 조작적 정의, §2.3 DH-Mamba 정면 구분, 절대 부정 명제 전부 완화 |
| 8 | val 이중 사용 | ✅ §5-(6) 명시 |
| 9 | 코일 압축 | ✅ §3.4·§5-(6) 명시 |
| 10 | 초록 압축 | ✅ 강화판 세부 한 구절로 압축 + keywords 에 parameter efficiency |

**GPU 큐 (권장 순서 — radapt 완주 ~08-25 후, 2차 검산 08-18 합의)**:
① **추론-only 일괄**(합쳐 하루 안쪽): Table 4 기준선(08-20 CPU 층화 표본 299로 선행 확인 → GPU 전량 재실행) → 추론 ms/slice·peak VRAM(Table 5 완성) →
320-crop 표준 프로토콜 보충표 → ringing 마스크-외부 잔차 정량 + Fig.3 오류맵/N 명시 →
② **U-Net-only 학습**(~50ep, 며칠 — 과학적으로 가장 중요한 잔여 항목: 치환 이득의 분모) →
③ 교수님 결정 후 **멀티시드 축약판**(25ep×3시드 순위 안정성).
교수님 상의 3건(저자·radapt 포함·멀티시드)과 ✎ 3건([33] TCSVT 권호·[34] 저널판·fastMRI DUA
공식 문구 대조)은 별도 트랙.

---

*변경 이력: v1(2026-08-07, 교수님 상의용) → v2(2026-08-18, 투고 준비 자료집 기반 재구조화 —
서론 5문단 체계·Related Works 확장·문제 정식화·평가 신뢰성 고찰·MDPI 후반부 항목·참고문헌
48편 병합[`paper/references.bib`]) → v2.1(2026-08-18, 외부 검토 보고서 반영 — 부록 D: 볼륨 단위
통계·클러스터 CI·표 슬라이스 단위 통일·h/ep 재측정(환경 confound 발견)·효율 프레임 전환·지표
정의 정밀화·contrast 보충표·서지 5건 확정) → v2.2(2026-09-01, 계보 서지 보강 — DOTA-MRI [50]·dual-input
ETER-net [51] 추가·본문 반영(§1③·§2.1·§2.4·§5-(2)·부록 B), Crossref 확정; 같은 날 서지 전건 재검증 74항목 — IEEE SPM 2편 부제 복원·JKSR 2편 저자 보강·SENSE/ReconFormer/WACV/MICCAI/CVPR DOI 보강·[33][34] 권호 확정·lin2024robustness 저자 오표기 키 교정).*
