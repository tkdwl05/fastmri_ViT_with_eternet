# [초안 v1] ETER-Net 골격에서 순환신경망(GRU)의 선택적 상태공간모델(SS2D) 치환: 통제 비교 연구

> **⛔ 대체됨 (SUPERSEDED, 2026-08-20 표기)**: 이 문서는 `paper/draft_ko_v2.md` (v2.1, 외부검토
> 08-18 반영)로 대체됐다 — 아카이브 보존용. 본문의 radapt "~08-14 완주 예정" 등 일정 서술은
> 작성 당시(08-07) 기준이며 실제로는 08-18 재개·ETA ~08-25.

> **상태**: 교수님 상의용 한국어 초안 (2026-08-07 작성, 결과 범위 = v8 통제비교 + v9 unleashed).
> radapt(R 일반화, ~08-14 완주 예정)는 §6.4 및 부록 B 에 "확장 예정" 으로만 표기.
> 영어 번역·투고 포맷 전환은 내용 확정 후. 그림은 부록 A 의 기존 산출물 매핑 참조.

**가제 (영문)**: *Replacing the RNN with a Selective State Space Model in ETER-Net:
a Controlled Single-Variable Comparison for Direct k-Space-to-Image MRI Reconstruction*

저자: (미정 — 사용자/교수님 상의)

---

## 초록 (안)

ETER-Net[1]은 언더샘플된 k-space 를 양방향 순환신경망(bi-RNN)으로 이미지 도메인에 직접
변환(domain transformation)하는 MRI 재구성 계열이다. 본 연구는 이 골격에서 **시퀀스 모델
하나만을 단일 변수로 통제**하여, 도메인 변환 자리의 bi-GRU 를 선택적 상태공간모델
(selective state space model; Mamba 의 2차원 확장인 SS2D)로 교체했을 때의 효과를 정량
평가한다. fastMRI brain multicoil 데이터(384×384, R=4)에서 두 모델은 시퀀스 모델을 제외한
모든 구성요소(데이터로더·손실·최적화·후처리 U-Net)를 공유하며, 원 논문에 충실하게 명시적
data-consistency(DC) 블록 없이 학습된다. 그 결과 SS2D 치환 모델은 **21× 적은 파라미터
(31M vs 668M)로 표준 4지표(SSIM·PSNR·NMSE·L1) 전부에서 GRU 를 상회**했으며
(best masked SSIM 0.9140 vs 0.9126, PSNR 35.16 vs 35.03 dB),
전체 검증 7,334 슬라이스의 paired 비교에서
지표별로 슬라이스의 74~78%에서 우위를 보였다(Wilcoxon signed-rank, p<0.001). 정성적으로 GRU 재구성에서
관찰되는 두개골 외부 배경의 ringing 아티팩트가 SS2D 에서는 나타나지 않았다. 나아가 통제를
해제하고 SS2D 를 강화(게이팅 복원·잔차 3블록·병목 해제·coarse-scan 다운샘플)한 변형은
epoch 당 학습시간을 단축(2.78→2.51 h/ep)하면서 최종 품질을 추가 개선했다(SSIM 0.9145,
슬라이스의 54~56%에서 우위, p<0.001). 본 결과는 도메인 변환형 재구성에서 RNN→SSM 치환이 DC 의
도움 없이도 일관된 이득을 줌을 통제된 조건에서 보인 것으로, ETER-Net 계열의 자연스러운
차세대 확장 방향을 제시한다.

**키워드**: MRI reconstruction, k-space, domain transformation, state space model, Mamba, ETER-Net

---

## 1. 서론

- **배경**: MRI 가속화(언더샘플링)는 ill-posed 역문제. 딥러닝 재구성의 두 계열 —
  (a) **unrolled 최적화 + data consistency(DC)** 계열[5-10], (b) **직접 도메인 변환(direct
  mapping)** 계열[1,2,4]. ETER-Net[1]은 (b)의 대표로, bi-RNN 이 k-space→image 변환 자체를
  학습하고 CNN(U-Net)이 de-aliasing 을 담당한다.
- **문제의식**: ETER-Net 의 bi-RNN(GRU)은 flatten reshape 구조 탓에 파라미터가 비대
  (본 세팅 668M)하고, 시퀀스 길이에 대한 순차 의존으로 병렬화가 제한된다. 최근 선택적
  상태공간모델(Mamba)은 선형 복잡도로 장거리 의존을 모델링하며 비전 과제에서 RNN/Transformer 의
  대안으로 부상했고, MRI 재구성에도 다수 적용됐다[11-15]. 그러나 기존 Mamba-MRI 연구는
  대부분 **이미지 도메인 prior/unrolled 정규화기** 자리에 SSM 을 넣는 새 아키텍처 제안이며,
  **도메인 변환 자리의 RNN 을 SSM 으로 1:1 치환하는 통제 비교는 부재**하다.
- **선행 내부 실험(동기)**: ViT 인코더 하이브리드[3] 위에서 GRU(+U-Net 후처리)와
  SS2D(+DC)를 비교했을 때 dead-heat(masked SSIM 0.9084 vs 0.9083 — 사실상 동률)였으나, 이 비교는 DC 유무와
  후처리 구조가 얽힌 confound 를 가진다. 본 연구는 confound 를 제거한 순수 골격에서 질문을
  격리한다: **"도메인 변환 자리에서 Mamba 는 GRU 보다 나은가?"**
- **기여**:
  1. ETER-Net 골격에서 시퀀스 모델만 교체한 **단일 변수 통제 비교** — SS2D 가 표준 4지표 전부,
     matched-epoch 전 구간(wire-to-wire), 슬라이스의 74~78%에서 우위 (파라미터 21×↓).
  2. **DC 무관성 실증** — DC 없는 순수 세팅에서의 완승으로, 선행 dead-heat 의 "SS2D 는 DC
     덕"(DC-crutch) 가설을 반박. 도메인 변환형에서 single soft-DC 부착이 문헌 표준(unrolled
     interleave[5-10])과 다름을 정리하고 no-DC 설계를 정당화.
  3. 통제 해제 시의 상한 탐색 — **강화 SS2D**(게이팅·잔차 3블록·병목 해제·coarse-scan)로
     학습속도와 최종 품질 동시 개선. matched-epoch 기준의 정직한 해석 포함(§5.3).

## 2. 관련 연구

### 2.1 직접 도메인 변환 (direct k-space-to-image) 계열
- **ETER-Net**[1]: 두 bi-RNN 이 수평/수직 방향 도메인 변환 수행 + CNN de-aliasing. R=4
  fastMRI 에서 SSIM 0.931 보고. 명시적 DC 블록 없음.
- radial 등 non-Cartesian 궤적으로 확장[2], ViT 인코더와의 하이브리드[3] (교수님 2025 —
  BiRNN 의 k-space 순차 처리가 고가속·랜덤 샘플링 강건성에 핵심임을 보고).
- k-space 도메인 CNN 을 포함한 교차 도메인 계열: KIKI-net[4].
- **본 연구의 위치**: [1,3]의 골격을 유지한 채 도메인 변환 모듈만 교체하는 후속 ablation.

### 2.2 Unrolled 최적화 + DC 계열
- CRNN-MRI[5], Variational Network[6], E2E-VarNet[7], RecurrentVarNet[8], CIRIM[9] —
  공통적으로 **작은 recurrent/conv unit 을 최적화 반복으로 unroll 하고 DC 를 매 반복
  interleave**. DC 는 이 계열의 "cornerstone"[10].
- 본 골격(거대 bi-RNN 도메인 변환 + 종단 single soft-DC)은 이 구조와 다르며, 원 논문[1]에도
  DC 가 없음 → 본 비교는 no-DC 를 기본 축으로 설계 (DC 축 폐기 근거는 §6.3).

### 2.3 SSM/Mamba 기반 MRI 재구성
- 이미지/k-space 도메인 prior 로서의 Mamba: MambaRecon[11], DH-Mamba[12](k-space 스캔의
  스펙트럼 파괴 문제 지적), CAM[14], HiFi-Mamba[15]. unrolled+DC 결합: MambaRoll[13].
- Transformer 의 recurrent 결합: ReconFormer[16].
- **차별점**: 기존 연구는 새 아키텍처 제안·SOTA 경쟁이 목적. 본 연구는 **기존 골격에서
  RNN↔SSM 치환의 효과를 격리하는 통제 실험**이 목적이며, 도메인 변환(k→image) 자리에서의
  SSM 은 [11-15] 어디에도 없다.

## 3. 방법

### 3.1 공통 골격 (순수 ETER-Net, ViT·DC 없음)

```
입력 1: aliased 이미지  (B, 32, 384, 384)   # 16코일 × (실수부/허수부)
입력 2: k-space         (B, 32, 384, 384)
                     │
        시퀀스 모델 = GRU (양방향 h+v)  또는  SS2D   ← 유일한 변수
                     │
        cat(시퀀스 출력, aliased 이미지)              ← 2-way concat
                     │
        UNet_choh_skip (DFU, depth=5, wf=6)          # de-aliasing 후처리 (~30M, 공유)
                     │
        출력: magnitude 이미지 (B, 1, 384, 384)
```

- **GRU arm**: ETER-Net 원본의 양방향(수평+수직) GRU. flatten reshape 구조로 668M 파라미터.
- **SS2D arm**: 4방향 selective scan(SS2D) 단일 블록. GRU 출력 채널(20)에 강제 정합해
  용량 상한을 GRU 이하로 억제(31M — 이 중 U-Net 이 ~30M 로 지배, SSM 자체는 ~1M).
- 이 외 **모든 것 동일**: 데이터로더·마스크·손실·옵티마이저·스케줄·에폭·후처리 U-Net·시드.

### 3.2 강화 SS2D (통제 해제 변형, "v9")

통제비교의 SS2D 는 공정성을 위해 의도적으로 최소화된 구성이다. 강화 변형은 세 가지를 복원/확장:
1. **게이팅 복원** — 공식 Mamba 의 `y = y·SiLU(z)` 분기 (통제판에는 없음).
2. **잔차 스택** — 채널 불변 SS2D 블록 × 3 (residual).
3. **병목 해제** — out_ch 20→64, d_inner 128→256, d_state 16→32, dropout 0.05.

연산 병목(384² 풀해상도 스캔) 해결: **fp16 selective scan** + **다운샘플 front-end(ds=3)** —
stem 이 풀해상도 k-space 를 먼저 처리한 뒤 feature 를 128² 로 낮춰 coarse scan, bilinear
업샘플 후 U-Net 에 전달(전역 문맥=SSM, 풀해상도 디테일=U-Net 분업). 결과적으로 풀용량을
유지하면서 GRU 통제판보다 빠른 2.51 h/ep(vs v8 2.78)를 달성, 동일 예산에서 epochs 50→80 확장.
총 파라미터 ~33M(SSM 스택 ~2M).

### 3.3 손실·지표 (brain-masked)

- brain mask: Otsu 임계 × 0.4 + 최대 연결성분 (배경이 지표를 부풀리는 것을 차단).
- 손실: masked L1 + (1 − SSIM).
- 평가: masked SSIM / PSNR / NMSE / L1 — **주 지표는 SSIM**(fastMRI 표준), 나머지 병기.
- 모델 선택(best ckpt)·조기 종료 기준: SSIM·PSNR·NMSE 를 가중 합성한 내부 스칼라
  (composite, 자체 설계 — `docs/eval_metric_redesign.md`)를 사용했다. **이 스칼라는 문헌
  표준이 아니므로 방법 절의 이 한 줄로만 서술하고, 본 논문의 모든 결과 표·수치는 표준
  지표로만 제시한다** (2026-08-07 결정, 부록 B-6). 결론은 표준 지표 전부에서 성립한다(§5.1).
- **통계 분석**: 두 모델이 동일 슬라이스를 재구성하는 paired 설계 — 지표별 paired 차이에
  Wilcoxon signed-rank 검정을 적용하고, 효과크기로 **우위 슬라이스 비율**(proportion of slices
  favoring; 통계학의 probabilistic index[17]에 해당, 임상시험의 win ratio 계열[18]과 동족)을
  함께 보고한다. 영어 원고 표기 예: *"SS2D achieved higher SSIM in 78.2% of slices (5,737/7,334;
  Wilcoxon signed-rank test, p < 0.001)"*. p 값은 저널 관례에 따라 p<0.001 로 표기
  (원값은 `results/eval/` 산출물에 보존).

## 4. 실험 설정

| 항목 | 값 |
|---|---|
| 데이터 | fastMRI brain multicoil, **혼합 contrast**(AXT1/AXT1POST/AXT1PRE/AXT2/AXFLAIR) |
| 규모 | train 4,108 파일 / 65,028 슬라이스, val 464 파일 / 7,334 슬라이스 |
| 전처리 | 384×384 center-crop/zero-pad, 16코일×실/허수 32채널 |
| 언더샘플링 | R=4 equispaced Cartesian, 중앙 ACS 8% (매 샘플 offset 랜덤) |
| 증강 | H/V flip p=0.5 (flip 후 FFT 재계산으로 k-space 물리 정합 유지) |
| 최적화 | Adam LR 2e-4, CosineAnnealingLR, AMP(fp16), grad-clip 1.0, BS 8 |
| 학습량 | 통제비교 50 ep / 강화판 80 ep, val 매 2 ep |
| 하드웨어 | TITAN RTX 24GB 단일 GPU |

- 강화판 추가 위생: Mamba `A_log`/`D` weight-decay 제외 그룹.
- (radapt 변형에만 해당하는 mask 조건화·DC·multi-AR 은 본 초안 범위 외 — §6.4.)

**기준선(baselines)**: fastMRI **brain leaderboard 사전학습 U-Net·E2E-VarNet[7]** 을 동일 val
파이프라인(같은 슬라이스·R4 마스크·GT·brain mask)에서 추론 평가해 참고 기준선으로 제시한다.
두 모델은 출력 스케일이 자체 정규화 기준이므로 지표 계산 전 per-slice 최소제곱 스케일 정합을
적용한다(우리 모델은 α≈1). 단 leaderboard 가중치의 원 학습분포(전체 코일·native 해상도)와
우리 전처리(16코일·384) 사이에 domain shift 가 있어 절대 우열 판정이 아닌 **참고선**으로
표기한다. 리더보드/문헌의 공식 프로토콜 수치(320 crop·배경 포함 raw SSIM)와 본 masked 지표는
좌표계가 달라 직접 대조하지 않는다.

## 5. 결과

### 5.1 통제 비교 — SS2D 완승 (v8)

**best checkpoint 기준 (val 전체):**

| | epoch | SSIM | PSNR(dB) | NMSE | L1 | params |
|---|---:|---:|---:|---:|---:|---:|
| **SS2D** | 48 | **0.9140** | **35.16** | **0.0039** | **8.931** | **31M** |
| GRU | 50 | 0.9126 | 35.03 | 0.0040 | 9.054 | 668M |

- **matched-epoch 전 구간 우위(wire-to-wire)**: 25개 val 지점(ep2~50)에서 SS2D 가 SSIM 기준
  전 지점 ≥ GRU(Δ +0.0000~+0.0055; 동률은 ep14 1지점뿐), PSNR 기준 전 지점 우위.
  ViT 하이브리드 트랙에서 관찰됐던 후반 역전(crossover) 없음.
- **per-slice paired 검증** (val 7,334 슬라이스 전수, Wilcoxon signed-rank):

| 지표 | GRU mean±std | SS2D mean±std | SS2D 우위 슬라이스 비율 | p |
|---|---:|---:|---:|---:|
| SSIM | 0.9126±0.0894 | 0.9140±0.0890 | **78.2%** | <0.001 |
| PSNR | 33.78±2.59 | 33.90±2.63 | **73.8%** | <0.001 |
| NMSE | 0.0045±0.0053 | 0.0044±0.0053 | **73.8%** | <0.001 |
| L1 | 9.00±3.05 | 8.88±3.05 | **76.1%** | <0.001 |

  → aggregate 우위가 소수 슬라이스에 의한 것이 아니라 대다수 슬라이스에서 일관됨.
  (주: 이 표의 PSNR/NMSE/L1 절대값은 best 표와 다름 — 학습 시 val 이 배치(BS=4) 풀링으로
  계산된 반면 본 표는 슬라이스 단위. SSIM 은 양쪽 모두 슬라이스 단위라 정확히 일치하며,
  paired 비교는 3모델 동일 프로토콜로 유효.)

### 5.2 정성 비교 — GRU 배경 ringing

4-way 시각화(GT/U-Net/GRU/SS2D)에서 **GRU 는 두개골 바깥 배경에 반복적 ringing/줄무늬
아티팩트**를 보이는 반면 SS2D 는 해당 영역이 깨끗함(확인한 모든 슬라이스에서 일관).
brain-mask 밖이라 정량 지표에 반영되지 않는 순수 정성적 차이로, GRU 재구성이 ROI 밖에서
덜 안정적임을 시사. (그림: 부록 A)

### 5.3 강화 SS2D — 상한 탐색 (v9 unleashed, 80 ep)

| | SSIM | PSNR | NMSE | L1 |
|---|---:|---:|---:|---:|
| **강화 SS2D (best ep78)** | **0.9145** | **35.18** | 0.0039 | 8.931 |
| 통제판 SS2D (best ep48) | 0.9140 | 35.16 | 0.0039 | 8.931 |
| GRU (best ep50) | 0.9126 | 35.03 | 0.0040 | 9.054 |

- per-slice 우위 슬라이스 비율: vs 통제판 SS2D **54~56% (표준 4지표 전부, Wilcoxon p<0.001 —
  유의하나 근소)**, vs GRU **78~82% (완승)**.
- **정직한 해석 (반드시 유지)**: matched-ep50 시점 강화판 SSIM = 0.9130 으로 통제판
  best(0.9140)에 미달. 통제판 best SSIM 에 도달한 것은 연장 구간의 ep64(동률)~ep66(상회).
  즉 "같은 학습량에서 더 좋다"가 아니라 **"ep 당 속도 이득(2.51 vs 2.78 h/ep)으로 더 긴
  스케줄(80ep)을 소화해 최종 품질을 넘었다"**가 정확한 서사(best 도달 wall-clock 은 강화판
  ≈181h > 통제판 ≈133h). 아키텍처 강화 자체의 순수 이득은 근소.
- coarse-scan(ds=3) 다운샘플이 품질을 해치지 않음: ep40 시점 열위 → 후반 cosine anneal
  구간에서 역전, 최종 상회.

### 5.4 기준선 비교 — U-Net / E2E-VarNet (결과 삽입 예정)

§4 기준선 프로토콜로 전체 val 7,334 슬라이스에서 U-Net·E2E-VarNet[7]을 추론 평가한 결과
(표준 4지표 mean±std + 본 모델들과의 우위 슬라이스 비율)를 여기에 삽입한다.
**[스크립트 준비 완료: `v8_eter_pure/eval_paired_baselines.py` — radapt 완주(~08-15) 후 GPU 실행]**
(과거 시도에서 VarNet 의 sensitivity 추정이 k-space ortho 스케일(~1e-4)에서 발산하는 문제가
확인되어 unit-max 정규화로 완화되어 있음 — 잔여 non-finite 슬라이스는 제외하지 않고 개수를
보고한다.)

## 6. 고찰

1. **DC-crutch 가설 반박**: ViT 하이브리드 비교의 dead-heat 가 "SS2D 는 DC 덕"이라는
   해석을 허용했으나, DC 를 완전히 제거한 통제 세팅에서 SS2D 가 더 확실히 이긴다.
   Mamba 의 이득은 DC 와 무관한 시퀀스 모델링 자체에서 나온다.
2. **파라미터 효율**: 668M GRU 를 31M 이 이긴다(21×↓). 도메인 변환 자리의 RNN flatten
   구조가 비효율의 근원이며, SSM 은 같은 자리를 선형 복잡도·저용량으로 대체.
3. **DC 축을 주 비교에서 제외한 근거**: (a) 원 논문[1]에 DC 없음, (b) 문헌의 DC 는 unrolled
   interleave 구조[5-10]로 본 골격의 종단 single soft-DC 와 상이, (c) 내부 실험에서 종단
   soft-DC 는 fp16 학습 불안정(gradient overflow)을 유발했고 GRU/SS2D 를 유의미하게
   구분하지 못함. DC 도입은 문헌식 재설계(향후 과제)로 분리.
4. **한계**:
   - 단일 데이터셋(fastMRI brain)·단일 가속률(R=4)·단일 시드 — R 일반화는 진행 중인 radapt
     변형(mask 조건화 + DC + multi-AR 학습)에서 다룸 (완주 후 본 논문 포함 여부 결정).
   - 통제비교의 SS2D 는 의도적으로 최소 구성 — 강화판이 상한을 일부 보완하나 근소.
   - knee 등 타 해부부위·prospective 언더샘플링 미검증.
5. **Novelty 포지셔닝(솔직)**: Mamba-MRI 아키텍처 자체는 이미 성숙 분야[11-15]. 본 논문의
   가치는 새 아키텍처가 아니라 **(a) 도메인 변환 자리에서의 1:1 통제 치환 실험, (b) no-DC
   조건의 DC 무관성 실증, (c) ETER-Net 계열[1-3]의 직접 후속**이라는 점. 투고 시 이 프레임을
   유지해야 리뷰 방어가 가능.

## 7. 결론

ETER-Net 골격의 도메인 변환 자리에서 bi-GRU 를 SS2D 로 치환하는 것만으로, DC 없이,
21× 적은 파라미터로, 전 지표·전 구간·슬라이스 대다수에서 일관된 품질 향상을 얻었다.
게이팅·깊이·병목 해제를 더한 강화 SS2D 는 학습 속도를 앞당기며 최종 품질을 추가로 근소
개선했다. 도메인 변환형 재구성의 차세대 시퀀스 모델로서 SSM 은 RNN 의 자연스러운 대체재이며,
가속률 일반화(radapt)·타 궤적/부위 확장이 후속 과제다.

---

## 참고문헌 (Consensus 검색, 2026-08-07 — URL 은 서지 확인용)

1. [A k-space-to-image reconstruction network for MRI using recurrent neural network](https://consensus.app/papers/details/e086f6c01053552982194e3a7516114d/?utm_source=claude_desktop) — Oh et al., *Medical Physics*, 2020.
2. [An End-to-End Recurrent Neural Network for Radial MR Image Reconstruction](https://consensus.app/papers/details/d02a7cfc56ed52bc8de6406e34f15963/?utm_source=claude_desktop) — Oh et al., *Sensors*, 2022.
3. [A Hybrid Vision Transformer-BiRNN Architecture for Direct k-Space to Image Reconstruction in Accelerated MRI](https://consensus.app/papers/details/2709b67d75f1538ea5a6705baf16a30e/?utm_source=claude_desktop) — Oh, *Journal of Imaging*, 2025. **★ 교수님 최신작 — 포지셔닝 필수 참조**
4. [KIKI-net: cross-domain convolutional neural networks for reconstructing undersampled magnetic resonance images](https://consensus.app/papers/details/62152bc263095f5a9c1d6c08d642f4b6/?utm_source=claude_desktop) — Eo et al., *MRM*, 2018.
5. [Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction](https://consensus.app/papers/details/80309beecaa85cce904684c268cd0fbb/?utm_source=claude_desktop) — Qin et al., *IEEE TMI*, 2017.
6. [Learning a Variational Network for Reconstruction of Accelerated MRI Data](https://consensus.app/papers/details/bf2b78c450a75fc7b57446197ae6bd61/?utm_source=claude_desktop) — Hammernik et al., *MRM*, 2017.
7. [End-to-End Variational Networks for Accelerated MRI Reconstruction](https://consensus.app/papers/details/78ee422de9a5542ea20e60f018b25cd9/?utm_source=claude_desktop) — Sriram et al., 2020.
8. [Recurrent Variational Network: A Deep Learning Inverse Problem Solver applied to the task of Accelerated MRI Reconstruction](https://consensus.app/papers/details/ce327daf6a8a5c748558439c3e19953a/?utm_source=claude_desktop) — Yiasemis et al., *CVPR*, 2022.
9. [Assessment of data consistency through cascades of independently recurrent inference machines for fast and robust accelerated MRI reconstruction](https://consensus.app/papers/details/39f625ae1ab651af8d85b9dbf8188f09/?utm_source=claude_desktop) — Karkalousos et al., *PMB*, 2021.
10. [Systematic evaluation of iterative deep neural networks for fast parallel MRI reconstruction with sensitivity-weighted coil combination](https://consensus.app/papers/details/a41e1c97b3335bbfb4e18277791c0865/?utm_source=claude_desktop) — Hammernik et al., *MRM*, 2021.
11. [MambaRecon: MRI Reconstruction with Structured State Space Models](https://consensus.app/papers/details/8f16a903af71561b966fc3983a2c6353/?utm_source=claude_desktop) — Korkmaz et al., *WACV*, 2025.
12. [DH-Mamba: Exploring Dual-Domain Hierarchical State Space Models for MRI Reconstruction](https://consensus.app/papers/details/094e29c419ea5efe88580993aad27475/?utm_source=claude_desktop) — Meng et al., *IEEE TCSVT*, 2025.
13. [Physics-Driven Autoregressive State Space Models for Medical Image Reconstruction (MambaRoll)](https://consensus.app/papers/details/539ddda516335fd7b32bf9dadc2fe58f/?utm_source=claude_desktop) — Kabas et al., *IEEE TMI*, 2024.
14. [Image Content Matters: An Image Content Aware State Space Model for Accelerated MRI Reconstruction (CAM)](https://consensus.app/papers/details/b9a915408dfe5945ba965e71ae65bae8/?utm_source=claude_desktop) — Meng et al., 2026.
15. [HiFi-Mamba: Dual-Stream W-Laplacian Enhanced Mamba for High-Fidelity MRI Reconstruction](https://consensus.app/papers/details/bb84be7eb6065d308a478dbc64957511/?utm_source=claude_desktop) — Chen et al., 2025.
16. [ReconFormer: Accelerated MRI Reconstruction Using Recurrent Transformer](https://consensus.app/papers/details/515feed12cc358b5ae97cb124bdba8f6/?utm_source=claude_desktop) — Guo et al., *IEEE TMI*, 2022.
17. [Probabilistic index: an intuitive non-parametric approach to measuring the size of treatment effects](https://onlinelibrary.wiley.com/ai/10.1002/sim.2256) — Acion et al., *Statistics in Medicine*, 2005. (우위 슬라이스 비율의 통계 명칭 근거)
18. [A win ratio approach to comparing continuous non-normal outcomes in clinical trials](https://onlinelibrary.wiley.com/ai/10.1002/pst.1743) — Wang & Pocock, *Pharmaceutical Statistics*, 2016.

**추가 확보 필요 (표준 인용 — 서지 미검증, 투고 전 확인)**: Gu & Dao 2023 (Mamba 원 논문),
Liu et al. 2024 (VMamba/SS2D), Zbontar et al. 2018 (fastMRI 데이터셋), Huang et al. MambaMIR
(*Med. Image Anal.* 2024), Zou et al. MMR-Mamba (*Med. Image Anal.* 2024).

---

## 부록 A. 그림·표 ↔ 저장소 산출물 매핑

| 논문 요소 | 소스 (저장소 경로) | 상태 |
|---|---|---|
| Fig.1 아키텍처 다이어그램 | 신규 작성 필요 (§3.1/3.2 기반) | ❌ 미작성 |
| Fig.2 학습 곡선 (GRU vs SS2D vs 강화판) | `results/eval/v9_unleashed/curves_v9_vs_v8.png` | ✅ (논문용 재도색 권장) |
| Fig.3 4-way 정성 비교 + ringing | `results/vis/v8_pure_eternet_compare/compare_*.png` | ✅ (슬라이스 선별 필요) |
| Fig.4 per-slice 우위 비율/분포 | `results/eval/{v8_nodc,v9_unleashed}/per_slice_paired*.csv` 에서 신규 플롯 | ❌ 미작성 |
| Tab.1 best 지표 비교 | `docs/v8_eter_pure_rnn_vs_ss2d.md` §3, `docs/v9_mamba_unleashed_and_radapt.md` §11.1 | ✅ |
| Tab.2 win-rate | `results/eval/*/win_rate_summary*.md` | ✅ |
| Tab.3 matched-epoch | `results/eval/*/matched_epoch_table*.md` | ✅ |
| Tab.4 기준선(U-Net/E2E-VarNet) | `v8_eter_pure/eval_paired_baselines.py` → `results/eval/baselines_384/` | ⏳ 스크립트 준비, GPU 대기(radapt 완주 후) |

## 부록 B. 교수님 상의 포인트 (초안 논외, 회의용)

1. **[3](교수님 2025 ViT-BiRNN, J. Imaging)과의 관계 설정** — 본 논문을 [1,3]의 직접 후속
   (도메인 변환 모듈의 세대 교체)으로 프레이밍하는 안. 저자 구성·기여 서술에 영향.
2. **투고처** — 사용자 제시 후보 5곳 평가 (2026-08-07):

   | 순위(안) | 저널 | 스코프 적합 | 일정 | 비고 |
   |---|---|---|---|---|
   | **1** | **Bioengineering SI** (MDPI, "Next-Gen Neurodiagnostics: DL·hyperspectral·computing acceleration", 마감 11-30) | △→○ 뇌+DL+**컴퓨팅 가속**(fp16 스캔·ds=3·선형복잡도 SSM 서사) 정합. 단 "탐지(detection)" 중심이라 **재구성 포함 여부 게스트 에디터 사전 문의 필수** | ◎ 11-30 마감 → **radapt(R 일반화) 포함 가능** = 치환+강화+일반화 3막 완성 | SI 는 게스트 에디터 핸들링으로 심사 흐름이 부드러운 편 |
   | **2** | **Sensors** (MDPI) | ○ **교수님 ETER-net 확장판(2022)이 이미 Sensors 게재**[2] — 저널 연속성까지 "직접 후속" 서사 완성. MRI recon 게재 전례 다수 | ○ 마감 없음 | 가장 안전한 스코프. SI 문의 결과가 부정적이면 1순위로 승격 |
   | 3 | QIMS (AME) | ○ 정량 영상 저널, MRI 기술 논문 게재. 다만 임상 독자 성향 | △ 심사 1~2개월 | 의학영상 커뮤니티 노출 원할 때 |
   | 4 | Diagnostics (MDPI, IF 3.8, 마감 8-31) | △ 진단 중심 — 재구성은 스코프 스트레치 | ✗ **8-31 은 비현실적**(영어 전환+교수님 검토 3주, radapt 배제 강제). 연장 가능하다 해도 이점 없음 | IF 는 5곳 중 최고 |
   | 5 | CMC (Tech Science) | △ CS 응용 저널 — 기술적으론 가능하나 **의학영상 독자 부재**, ETER-Net 계보(Med.Phys→Sensors→J.Imaging)와 단절 | ○ 승인까지 평균 71일 | 권장하지 않음 |

   **권장 액션**: Bioengineering SI 게스트 에디터에게 가제+초록 사전 문의(재구성 스코프 확인,
   비용 0) → 긍정이면 SI 로(라 radapt 포함), 부정이면 Sensors 로. 전 후보 OA·APC 발생 —
   금액·할인(SI 초청 여부)은 투고 전 확인. (참고: [3] *J. Imaging* 도 MDPI — 교수님 선호
   확인 가치.)
3. **radapt(R 일반화) 포함 여부** — ~08-14 완주 예정. 포함 시 "치환+강화+일반화" 3막 구성으로
   기여가 두터워지나 투고가 ~2주+ 지연. 미포함 시 본 초안 그대로 + 후속 논문 분리.
4. **DC 서술 수위** — §6.3 의 "문헌 비표준 + 학습 불안정" 근거를 본문에 둘지 부록으로 뺄지.
5. **단일 시드 한계** — 리뷰 방어용 멀티시드 재학습(비용 큼) 필요성 판단.
6. **composite 노출 수위 — 해결(2026-08-07 사용자 결정)**: 결과 표·본문 수치에서 composite
   전면 제거, 방법 절(§3.3)에 모델 선택 기준으로만 한 줄 서술. 과거 실험 서술(v7_titan
   dead-heat 등)도 SSIM 기준으로 통일. wire-to-wire·연장구간 돌파 서사는 SSIM/PSNR 기준으로
   재검증 완료(동일하게 성립).
