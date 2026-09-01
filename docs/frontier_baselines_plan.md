# 최전선 공개 모델 기준선 계획 (PromptMR+ / DDS) — Table 4 확장

작성 2026-09-01. 근거 조사(문헌·리포 검증)는 세션 기록 참조. 실행은 radapt 완주(~09-04) 후
"추론-only 일괄" GPU 큐에 편입. 클론은 `external/`(gitignore, 로컬 전용).

## 1. 왜 이 두 개인가

| 모델 | 계열 | 가중치 | 우리 val 누수 | 역할 |
|---|---|---|---|---|
| **PromptMR+** (`external/PromptMR-plus`) | unrolled 최전선 (CMRxRecon2024 양 트랙 1위) | HF `hellopipu/PromptMR` — **fm-brain 372.6MB** (`promptmr-plus-epoch=44-step=1591830.ckpt`, `external/weights/`) | **✅ 없음 — train 스플릿만 학습** (README FastMRI-Brain 표: PromptMR+=train; PromptMR=train+val 이라 **plus 만 사용**) | Table 4 의 유일한 **누수-프리 최전선 참조행**. 기존 U-Net/VarNet leaderboard(train+val 누수, `baseline_leaderboard_leakage`) 캐비엇을 안 받는다 |
| **DDS** (`external/DDS`) | diffusion 샘플러 (Chung, 2024) | dropbox — ⚠ **README 의 "brain" wget 이 실제로는 `fastmri_knee_320_complex_1m.pt` 를 가리킴**. eval 스크립트도 knee config 사용 → 공개 prior 는 사실상 knee | 무관 (prior 는 GT 분포 학습, 우리 val 아님 — 단 knee prior 를 brain 에 쓰면 anatomy shift) | "생성 계열 + 추론 지연시간 대비" 행. ms/slice 수백 배 차이가 직접 변환 계열의 저지연 서사를 정량화 |

- DDS 대안: **CM-RED** (arXiv:2608.20561, 2026 — consistency 모델, knee+brain 가중치 공개 주장) — DDS 의
  knee-prior 문제가 걸리면 교체 검토. score-MRI 도 knee 전용이라 동일 문제.

## 2. PromptMR+ 어댑터 — 프로토콜 매핑

확인된 사실 (`configs/inference/pmr-plus/fm-brain.yaml`, `configs/train/pmr-plus/fm-brain.yaml` 대조):
- **해상도 `uniform_resolution: [384,384]`** — 우리 트랙과 동일 384. 큰 정합.
- **학습 가속률 4x/8x** — 우리 R4 평가는 in-distribution. 마스크는 공식 equispaced 규약
  (우리 119/384·ACS31 ≈ 공식 120/384·ACS≈0.08N — `eval_paired_baselines.py` 검증 로직 재사용).
- **`num_adj_slices: 5`** — 입력이 인접 5슬라이스 스택. 어댑터가 볼륨에서 z±2 를 함께 잘라 공급해야
  함 (경계 슬라이스는 그들 transform 의 복제 패딩 규약 확인).
- 코일: 그들은 native 전 코일 + 자체 sens 추정(`compute_sens_per_coil` 옵션으로 VRAM 절약).
  **VarNet 때의 교훈 그대로: zero-pad 코일 금지, 실측 코일만 전달.**
- lightning CLI 기반(`main.py predict`) — 우리 h5 를 그들 `FastmriSliceDataset` 포맷으로 먹이는 게
  기본 경로. 두 행 산출:
  1. **native 프로토콜 행**: 원본 k-space + 공식 마스크 규약 (`v8_eter_pure/native_protocol.py` 프레임
     재사용) — 그들 학습 분포와 일치, "최전선의 정점" 수치.
  2. **우리 프로토콜 행**: 384 re-FFT·16코일 절단·우리 마스크 — domain shift 로 그들이 불리해질 수
     있음을 §3.7 캐비엇에 명시 (leaderboard 기준선과 대칭 서술).
- 추론 VRAM: 12-cascade + sens — 24GB 에서 `compute_sens_per_coil=true` 로 안전. BS=1.

## 3. 실행 순서 (radapt 완주 후, 예상 합계 ~1일)

1. CPU 스모크: ckpt 로드 + 1 슬라이스 forward (모델 생성자·키 매칭 확인) — GPU 전 검증.
2. 층화 표본 299 (기존 `eval_paired_baselines.py` 표본과 동일 seed 0) → 방향 확인.
3. 전체 7,334 GPU 풀런 → per-slice CSV → `make_tables.py` Table 4 재생성.
4. ms/slice·peak VRAM 측정을 같은 런에서 채집 (Table 5).
5. DDS(또는 CM-RED): 표본 299 만이라도 — NFE=50 기준 ms/slice 대비가 목적. 풀 7,334 는
   diffusion 에선 비현실적(수일)이므로 표본 + 명시가 정직.

## 4. 논문 서술 포인트

- §3.7 에 PromptMR+ 행 추가: "**train-only 학습 공개 가중치** — 본 검증셋에 대한 누수 없음" —
  U-Net/VarNet 누수 캐비엇과 대구.
- 예상 결과: PromptMR+ SSIM 이 우리·VarNet 보다 높을 것(공식 test 4x 0.9615). 프레임은 기존대로
  "SOTA 경쟁 아님 — 품질/지연/파라미터 트레이드오프 좌표 제시". PromptMR+ 는 12-cascade unrolled
  (반복 sens+DC)라 지연시간에서 직접 변환 계열이 유리할 것 — Table 5 에서 확인.
