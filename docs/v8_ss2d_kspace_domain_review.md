# v8 SS2D 의 위치(k-space domain transform) + 외부 리뷰 판정 + 가속률(R) 일반화 방향

작성 2026-07-08. 외부 리뷰(ETER-net 에 SS2D 를 얹는 발상에 대한 아키텍처 조언)를 우리 실제 코드/결과와 대조해 판정하고, 후속으로 제안된 **가속률(AR/R) 일반화** 방향의 성립 조건을 정리한다. 근거 실험: `docs/v8_eter_pure_rnn_vs_ss2d.md`(no-DC 쌍 SS2D 완승), 근거 코드: `models/pure_eternet/`.

---

## 1. SS2D 가 들어간 자리 = domain transform (refinement 아님)

리뷰의 핵심 질문 = "SS2D 를 **k-space→이미지 변환 자리**에 넣었나, **refinement(이미지) 자리**에 넣었나". 코드로 확정: **domain transform 자리(k-space 입력)**.

```
# GRU arm  (u_pure_eternet_gru.py:107-110, _seq:89-105)
out_v  = self._seq(x_ksp)            # in_h = x_ksp.reshape(...) → gru_h(양방향) → gru_v(양방향)
in_cnn = cat(out_v, x_img)           # ← aliased image 는 여기서 skip 으로 합류
out    = self.unet(in_cnn)           # UNet_choh_skip (DFU) = refinement

# SS2D arm (u_pure_eternet_ss2d.py:74-77)
out    = self.ss2d(x_ksp)            # ← GRU 와 정확히 같은 입력(k-space), 같은 자리
in_cnn = cat(out, x_img)             # ← downstream 완전 동일
out    = self.unet(in_cnn)
```

- 원본 교수님 ETER-net(`models/hybrid_eternet/hybrid_eternet_fastmri-main/model.py:18` `ETER_hybrid_GRU_DFU`, `forward(x, x_img)`)도 `x`=k-space 를 GRU 로 훑는다 → 리뷰어의 "RNN 이 k-space 를 훑는다"는 우리 코드에서 사실.
- 즉 **seq model(GRU/SS2D)이 k-space→image-feature 변환(domain transform)을 담당**하고, U-Net(DFU)이 그 뒤 refinement 를 한다. SS2D 는 GRU 와 **완전히 같은 자리·같은 입출력 계약**(cat 순서, U-Net, 크기 모두 동일)에 놓였고, 유일 변수는 seq model 종류다.
- 함의: 리뷰어가 "전자(=domain transform 자리)라면 지역성-전역성 미스매치를 신경 쓰라"고 한 **그 케이스에 정확히 해당**한다. 그리고 이것이 리뷰어가 지목한 차별점("k-space direct transform 에 SS2D 적용")과 우리 구조가 일치함을 뜻한다 — 대부분의 Mamba-MRI 는 image/dual-domain denoiser 로 Mamba 를 쓰지, k-space→image 변환 자체를 SSM 으로 학습시키지 않는다.

### 반전: 그 미스매치 우려는 우리 실험이 이미 실증 반박

리뷰어의 우려 = "SS2D 는 이미지 지역성 가정 블록이라, k-space 전역 변환에서 RNN 만큼 full receptive field 를 확보 못할 수 있다". 그런데 **그 검증이 바로 v8 no-DC 통제**였고, 결과는 SS2D 완승이다:

- SS2D no-DC best composite **0.9200** vs GRU 0.9182, **5개 지표 전부·25 matched-epoch wire-to-wire·params 21×↓**(31M vs 668M). per-slice 74~78% 승률(Wilcoxon p≈0).
- → SS2D 의 selective scan 이 이 k-space→image 변환에서 GRU(full-receptive RNN)의 전역성을 **따라잡을 뿐 아니라 더 잘한다**는 경험적 증거. 리뷰어의 이론적 우려는 우리 데이터에서 기각된다. (메커니즘 수준 검증 = effective receptive field 측정은 미실행 — 유효한 후속 분석.)

---

## 2. 외부 리뷰 항목별 판정

| 리뷰 항목 | 리뷰어 주장 | 우리 실제(코드/결과) | 판정 |
|---|---|---|---|
| ① domain transform 자리 | k-space 에 넣었으면 지역성↔전역성 미스매치 우려 | SS2D = k-space 입력(GRU 와 동일 자리) — `u_pure_eternet_ss2d.py:75` | **자리 확정 + 우려는 실증 기각**(SS2D 완승) |
| ② 복소수 처리 | 채널분리 vs 복소 SSM 결정이 성능 좌우 | `c_in=32`=16코일×2(real/imag) 채널분리, GRU `input_size`도 동일 방식 | 통제 정합. 복소 SSM 은 미탐색(부차 ablation) |
| ③ 입출력 크기(radial) | Kx×Ky→Nx×Ny 크기변환 설계 필요, radial 대응 | **Cartesian 384 고정, 입출력 동일**(`build_r4_mask`, target 384) | 우리 실험 범위 밖 — **전제 어긋남** |
| baseline GRU 통제 | 순수 효과 격리에 필수 | v8 2×2 전체가 그 통제. no-DC SS2D 완승 | **이미 완료** |
| 논문 수치(radial R4 L1: nMSE 1.98%/SSIM 0.922) | "최소 이 근처는 나와야" | 우리는 Cartesian·masked·brain(SSIM_m 0.914, nMSE 0.39%) | **직접 비교 부적절**(도메인·마스킹·데이터 상이) |
| 파라미터·속도 | 성능 비슷해도 효율 이득이면 기여 | 31M vs 668M(21×↓)인데 **성능도 이김** | **이미 충족(초과)** |
| 수렴 안정성(A 초기화·lr) | 발산·정체 시 초기화/lr 문제 | mamba 기본 초기화·lr 2e-4 로 안정 수렴·완승 | 우려 미실현(단 Mamba 전용 lr 튜닝은 안 함) |
| 가속률 일반화(R=2,6,8) | 재학습 없이 대응이 핵심 스토리 | **미실행**(R4 고정 학습·평가) | **유효한 미실행 제안** → §4 |

---

## 3. 종합 판단

- **방향·차별점 포지셔닝은 정확**하다. "k-space direct transform 에 SS2D" 라는 차별점을 우리 구조가 실제로 구현하고 있고, no-DC 통제가 DC 없이 순수 seq-model 축만으로 그 각도의 clean 한 증거를 냈다.
- 다만 리뷰어는 **우리 코드·결과를 못 본 상태**라 ③(radial/크기변환)은 Cartesian 세팅에 무효, 논문 수치 직접비교는 부적절, "확인하라"의 대부분(baseline 통제·효율·수렴)은 **이미 지나온 지점**이며, 핵심 우려(①)도 실증으로 답이 났다.
- **실질적으로 값진 유일한 미실행 제안 = 가속률(R) 일반화**(§4). 우리는 Cartesian mask 라 R4 학습 모델을 다른 R 에서 재학습 없이 평가할 수 있어 비용이 거의 없다.

---

## 4. 가속률(R) 일반화 — 성립 조건과 우리 방향

리뷰어가 인용한 **LMO(Linear Mamba Operator, Li et al., CVPR 2025)**의 핵심 주장 = "가속률(AR)이 바뀌어도 **재학습 없이** 재학습 수준의 성능"이며, 이는 deep unfolding 이 OOD(가속률 변화)에서 파국적으로 무너지는 문제를 겨냥한 것이다. 우리 파이프라인에서 R 을 바꾸면 무엇이 달라지는지부터 짚고(코드 사실), 일반화가 성립할 조건과 우리가 잡을 방향을 정리한다.

### 4.1 우리 파이프라인에서 R 변경 시 실제로 바뀌는 것 (코드 사실)

`dataloaders/dataloader_h5_v5.py`:
- `build_r4_mask(width, center_fraction=0.08, acceleration)` → `mask[offset::acceleration]=1` + ACS 중앙밴드(`round(width·0.08)` 라인). **R↑ → 측정 라인 수↓**, ACS 는 유지.
- `x_ksp`(masked k-space)·`x_img`(=iFFT(masked ksp), aliased image) 모두 R 에 따라 변한다 — R↑ 일수록 aliasing 심화·에너지↓.
- **`val_amp` 는 고정**(`X_img=1e6, X_ksp=1e4, Y=1e6`) → R 이 바뀌어도 스케일 배수는 그대로라, **입력 magnitude 가 R 에 따라 표류**(R↑ → 측정 에너지↓인데 amp 동일). = §4.2-③의 약점.
- `acceleration` 은 dataloader 생성 인자 → **R∈{2,4,6,8} 평가는 인자 하나로 가능**(단 현재 `eval_paired_*`/`visualize_*` 는 `acceleration=4` 하드코딩 → 인자화 필요).

### 4.2 R 일반화가 성립하려면 (조건)

| # | 조건 | 왜 | 우리 상태 |
|---|---|---|---|
| ① | **Data Consistency(측정값 앵커)** | 측정된 k-space 라인을 R 무관하게 강제 → 측정 집합이 달라져도 원리적으로 대응 | **DC arm 이 보유**(1-iter soft DC). no-DC 는 없음 → DC 가 더 R-robust 가설 |
| ② | **Mask-aware 입력** | 모델이 "지금 어떤 라인이 측정됐는지" 알아야 R 에 적응 | no-DC 는 mask 를 암묵(x_ksp 의 0 패턴)으로만, **DC 는 mask 명시 사용** → DC 유리 |
| ③ | **R-불변 입력 정규화** | R 에 따라 입력 에너지가 표류하면 OOD 분포 shift | **현재 없음**(val_amp 고정) — R 일반화의 구조적 약점, 차기 보완 후보 |
| ④ | **전역 receptive field** | R 변화는 전역 aliasing 구조를 바꿈 → 전역 처리 필요 | RNN·SS2D 둘 다 전역 충족. SS2D 의 input-dependent scan 이 R 을 암묵 조건화하면 GRU 보다 유리할 여지 |
| ⑤ | **단일-R 과적합 회피** (multi-AR 노출 또는 operator 구조) | 단일 R 학습이 그 R 의 aliasing 을 외우면 OOD 파국 | **현재 R4 고정 학습** → 과적합 노출. LMO 는 operator learning 으로, 흔한 대안은 학습 시 R 랜덤화 |

핵심: 위 ①②④는 **DC arm 이 no-DC 보다 여러 개를 이미 충족**한다. 반면 리뷰어/LMO 가 경고한 "deep unfolding 의 OOD 취약성"은 **여러 iter 를 쌓을 때** 문제이고, 우리 DC 는 1-iter soft 라 얕아 그 과적합에서 상대적으로 자유롭다 — "DC 가 R 일반화에 도움" 가설과 "unfolding 이 OOD 에 취약" 이 모순되지 않는 지점.

### 4.3 우리가 잡을 방향 (단계적)

1. **[무비용·즉시, 완주 후] R cross-eval** — R4 학습 4 ckpt(GRU/SS2D × noDC/DC)를 R∈{2,4,6,8}에서 **재학습 없이** 평가 → 일반화 곡선. 예상 관전 포인트:
   - DC arm 의 하락이 no-DC 보다 완만한가(①②) → "DC 가 R-robust" 검증.
   - SS2D 가 GRU 보다 완만한가(④) → "Mamba 전역성이 일반화에도 유리" 스토리.
   - 구현: `eval_paired_v8_*.py`·`visualize_*` 의 `acceleration=4` 를 `--accel` 인자화(dataloader 는 이미 지원). §계획(`abundant-wishing-pebble.md`) Phase 2 확장 축.
2. **[중간, 차기 재학습] R-불변화 보강 + multi-AR** — ③(R 밀도 보정 정규화) 도입, dataloader `acceleration` 을 배치별 랜덤{2,4,6,8}로 노출(⑤). "multi-AR 학습이 SS2D 에서 더 효과적인가".
3. **[연구 확장, v9] operator/unrolled + DC 강화** — LMO 식 함수공간 operator 로 discretization 불변성 확보. 우리 순수 ETER-net 은 직접 매핑이라 여기까진 아키텍처 변경. **정직한 한계**: 우리 세팅에서 볼 수 있는 것은 축 간 **상대 robustness**(DC vs no-DC, SS2D vs GRU)이지, LMO 급 절대 일반화(재학습 없이 재학습 수준)는 아키텍처가 달라 별개다.

### 4.4 연속/소수 가속률 (실제 스캐너 정합)

실제 스캐너는 R 을 소수로 설정/달성한다 — ACS 포함 **net(effective) acceleration** 은 거의 항상 소수이고(우리 "R4"도 `center_fraction=0.08`→ACS ~31 라인 포함 시 net R ≈ 3.2), variable-density CS·2D 가속(Ry×Rz)에서 유효 R 은 연속이다. 두 층으로 나눠 본다:

- **마스크(데이터) 쪽 — 지금은 정수만, 손봐야 함**: `build_r4_mask` 의 `mask[offset::acceleration]`(`dataloader_h5_v5.py:91`)은 **파이썬 슬라이싱 step 이라 정수 R 만** 표현(시그니처도 `acceleration: int`). 소수 nominal R 은 자동 반영 안 됨. → 밀도 기반으로 일반화하면 해결: 목표 취득 라인 수 `= round(width / R)` 를 (equispaced 근사 또는 variable-density 로) 선택하는 `build_mask(width, R: float)`. 작은 변경이며 ACS 처리는 그대로.
- **모델 쪽 — 자연스럽게 처리됨**: seq model 은 R 스칼라를 입력받지 않고 `x_ksp` 의 0 패턴(= 측정 라인)으로 샘플링을 인지한다. 따라서 소수 R 이든 정수 R 이든 **그냥 다른 샘플링 패턴일 뿐** → 별도 조건/재학습 없이 처리(§4.2-②④ mask-aware·전역 RF 가 뒷받침). 단 학습 밀도에서 먼 R 은 OOD → §4.2-⑤(multi-AR 학습)로 완화.
- **오히려 이득**: 이산 {2,4,6,8} 대신 **R∈[2,8] 연속 sweep** 으로 평가하면 일반화 곡선이 연속이 되어 LMO 의 "연속 AR 대응" 화두와 정합 + 임상 정합↑. → §4.3-1단계의 R cross-eval 을 연속 sweep 으로 확장 권장(마스크만 밀도 기반으로 바꾸면 평가 코드는 동일).

---

관련: `docs/v8_eter_pure_rnn_vs_ss2d.md`(no-DC 완승·2×2), `docs/eternet_paper_data_consistency.md`(ETER-net 엔 DC 없음, DC 는 증강 probe), 계획 `/root/.claude/plans/abundant-wishing-pebble.md`(DC 완주→2×2, R cross-eval 확장 후보).
