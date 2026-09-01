# v8 3번째 팔 설계 초안 — 도메인 변환 슬롯의 Axial-Attention (교수님 상의용)

작성 2026-09-01. 목적: v8 통제비교(GRU vs SS2D)를 **RNN vs SSM vs Attention 삼자 통제비교**로
확장할지 결정하기 위한 설계·비용 자료. 실행 여부·시점은 미정 (멀티시드 P1-1 과 GPU 큐 경합 —
부록 B 상의 항목).

## 1. 근거 — 이 슬롯은 Transformer 쪽도 비어 있다 (2026-09-01 문헌 재확인)

직접 k→image 변환 자리의 attention 은 문헌에 없음. 가장 가까운 것들:
- **k-space 보간 계열** (출력이 k-space, 변환은 여전히 IFFT): K-Space Transformer (Zhao, BMVC
  2022, arXiv:2206.06947), k-GIN (Pan 2023), radial spoke 예측 (Gao 2022)
- **이미지 도메인 백본**: SwinMR·HUMUS-Net·ReconFormer·KTMR — 초안 §2.3 에 이미 정리
- **k-space 직접 ViT**: kViT (Rempe 2026, arXiv) — **분류** 과제, radial patching 특수처리 필요
- 교수님 2025 [3]: ViT-only(모델1·2)가 BiRNN k-space 처리 대비 열세 — attention 단독이 이 슬롯에서
  약하다는 선행 신호. 우리 v7_titan ViT 하이브리드 dead-heat 도 같은 방향.

→ 결과가 어느 방향이든 정보 가치: SS2D 승 = "attention 대비도 SSM 우위" / attention 승 =
"선형 복잡도 없이도 가능" — 삼자 비교 자체가 논문 기여를 "sequence-model family 통제비교"로 격상.

## 2. 설계 — v8 통제 원칙 유지 (시퀀스 모듈만 교체, 나머지 100% 동일)

```
입력 k-space (B,32,384,384)
  → stem 1×1 conv 32→d_model (+LayerNorm)
  → [행방향 MHSA(L=384) → 열방향 MHSA(L=384) → FFN] × N층   ← bi-GRU 수평/수직 스캔의 attention 대응물
  → head 1×1 conv d_model→20                                  ← v8 out_ch=20 매칭 (통제)
  → cat(·, aliased image) → UNet_choh_skip DFU (v8 그대로)
```
- **axial 선택 이유**: full 2D attention 은 384²=147k 토큰 제곱이라 불가. 축별 attention 은
  GRU/SS2D 와 같은 축(행/열)을 같은 방향성(양방향=전역)으로 훑는 가장 공정한 대응물.
- positional encoding: 2D sin-cos 고정형 (학습형은 파라미터 변수 추가라 통제 흐림).
- k-space 다이내믹레인지: v8 입력 스케일 그대로 (특수처리 없는 '순정 attention' 이 실험 목적 —
  kViT/DH-Mamba 는 특수처리가 필요했다는 것이 우리 서사의 일부).
- 신규 파일만: `models/attn_eternet/axial_v10.py` + `models/pure_eternet/u_pure_eternet_axial.py`
  (원본 무수정 관례). config 는 v8 config 재사용 + `SEQ_MODEL=axial` 분기.

## 3. 파라미터 매칭 2안

| 안 | d_model | N층(행+열 쌍) | 스택 params | 비고 |
|---|---|---|---|---|
| **(a) 통제판** | 96 | 2 | ~0.5M (v8 SS2D 스택급) | 주 비교용 — v8 쌍과 동일 예산 |
| (b) 용량판 | 192 | 3 | ~3M (v9 스택급) | (a) 열세 시 용량 탓인지 분리용 (선택) |

## 4. 연산·비용

- TITAN RTX = Turing(sm_75) → FlashAttention 불가, PyTorch SDPA mem-efficient fallback 사용
  (L=384 축별이라 어차피 attention 행렬 미미: B·heads·384² fp16 ≈ 수십 MB).
- fp16 autocast (v8 과 동일). 예상 epoch 시간 GRU(2.41)~SS2D(3.07) 사이 ≈ **~2.7h/ep**.
- **50ep 1런 ≈ 5.6일** + per-slice 평가 ~2h (`eval_paired_v8_nodc.py` 3-way 조인 확장).
- 시드: 아래 멀티시드 프로토콜 확정 후 같은 시드 체계로 (1런이면 seed 0 고정 명시).

## 5. 판정 기준

- 1차: per-slice paired 5지표 win-rate + 볼륨 Wilcoxon (v8 체계 그대로, 3-way).
- 예상 시나리오: SS2D ≥ axial ≳ GRU (근거 §1). axial 이 GRU 도 못 이기면 "순차 재귀 귀납편향이
  k-space 라인 구조에 필수" 라는 더 강한 결론.
- 서사 배치: 본 논문 §4 확장 또는 후속 논문("the sequence-model zoo in the domain-transform
  slot") 분리 — 마감(Bioengineering SI 11-30)과 멀티시드 우선순위에 따라 교수님 결정.

## 6. Diffusion 은 왜 슬롯 치환이 아닌가 (상의용 한 단락)

diffusion 은 feedforward 모듈이 아니라 반복 샘플링 생성 프레임워크 — "블록 교체" 통제비교가
정의되지 않음. k-space 조건부 diffusion head 는 학습·평가(7,334 × NFE)·확률적 출력(시드별 상이,
hallucination) 모두에서 TITAN 1장 비현실적이고, perception–distortion tradeoff 상 SSIM/PSNR 은
회귀 모델에 통상 열세라 우리 지표 체계에서 정보가 없음. → 공개 가중치 diffusion(DDS/CM-RED)을
**기준선 행**으로 흡수하고(`docs/frontier_baselines_plan.md`), 슬롯 치환은 1~4-step consistency
모델 성숙 후의 후속 과제로 §5 에 한 줄.
