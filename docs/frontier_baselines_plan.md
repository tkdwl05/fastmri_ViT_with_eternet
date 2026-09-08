# 최전선 공개 모델 기준선 계획 (PromptMR+ / DDS) — Table 4 확장

작성 2026-09-01(갱신 09-06). 근거 조사(문헌·리포 검증)는 세션 기록 참조. 우리 프로토콜 행의 전체 val 추론은
**09-06 CPU 로 선행 실행 중**(§3), native 행·ms/VRAM 은 "추론-only 일괄" GPU 큐(7단계) 유지. 클론은 `external/`(gitignore, 로컬 전용).

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
   **✅ 2026-09-04 완료** — `visualize_multimodel_compare.py`(정성 그림용 CPU 추론)에서 PromptMR+ 첫 forward
   실행. `hyper_parameters` 로 `PromptMR` 직접 생성 + `promptmr.` 접두사 제거 로드(strict, `loss.w` 만 제외),
   인접 5슬라이스(z±2, 경계 복제) 스택을 우리 측정값에서 유도. CPU fp32 16 s/슬라이스(8 스레드).
   ⚠ **업스트림 버그**: `models/promptmr_v2.py:249` `PromptMRBlock.forward` 가 정의되지 않은 `self.n_buffer` 를
   참조(의도 = `self.model.n_buffer`) → AttributeError. 외부 clone 은 무수정, 로더가 각 cascade 에
   `blk.n_buffer = blk.model.n_buffer` 를 런타임 부여해 우회(풀런 어댑터도 같은 처리 필요).
   정본 12 슬라이스 결과: `results/vis/multimodel_compare/metrics_summary.txt`(PromptMR+ 가 예상대로 최상위 —
   단 12장 정성 표본이지 Table 4 수치가 아님; 다중 슬라이스 입력(5장)이라 단일 슬라이스 모델과 입력 정보량이 다름을 캡션에 명시).
2. 층화 표본 299 (기존 `eval_paired_baselines.py` 표본과 동일 seed 0) → 방향 확인.
   → **생략**: 정본 12 슬라이스(1단계)로 방향은 확인됐고, 3단계 풀런을 CPU 로 바로 시작(아래).
3. 전체 7,334 풀런 → per-slice CSV → `make_tables.py` Table 4 재생성.
   **▶ 2026-09-06 CPU 로 launch — 우리 프로토콜 행**(384² 재-FFT·16코일 절단·R4·brain-masked·LS 정합):
   `v8_eter_pure/eval_baselines_full.py`(러너·지표를 `visualize_multimodel_compare.py` 에서 import — 정본 슬라이스
   3368/7333 에서 저장 수치와 ≤1e-6 일치 확인, `n_buffer` 런타임 패치 포함). 방법별 CSV
   `results/eval/baselines_384_full/per_slice_{varnet,unet,promptmr}.csv` 에 슬라이스마다 append(재개 가능),
   PromptMR+(5 스레드) 와 E2E-VarNet†→U-Net†(순차, 3 스레드) **2 프로세스**(워커 1씩·nice 19, GPU0 의 E1 학습은 그대로;
   처리 순서 interleaved = 완료 prefix 가 항상 전 볼륨 층화 표본 → 중간 `--summary` 가 편향 없음).
   09-06 실측(호스트 외부 부하 ~10코어·유휴 ≈8코어): 4 스레드 단독 s/slice VarNet ≈5 · U-Net ≈11 · PromptMR+ ≈40, 12 스레드가
   4 스레드보다 느린 oversubscription 확인 → 총 CPU 일량 ≈450 코어시간 = 유휴 8코어 기준 **≈2.5일**(3 프로세스 12 스레드
   첫 시도는 서로 경합해 3배 느려져 재시작). PromptMR+ 가 long pole(외부 부하가 빠지면 단축).
   요약 `--summary` → `baseline_summary_full.md`(방법별 완료집합 블록 A + 전방법 공통집합 B: 슬라이스·볼륨 단위 mean±SD,
   GRU/SS2D/v9 대비 우위 비율·Wilcoxon, contrast 별).
   **09-06 21:30 — E2E-VarNet† 전체 완료**(7,334 슬라이스, non-finite 0, 8.3 s/slice): 볼륨 단위 SSIM 0.9181±0.0386 /
   PSNR 32.78±3.21 dB / nMSE 1.133±1.291 % — SS2D(0.9141/33.91/0.438) 대비 SSIM 은 +0.004(우위 볼륨 66.8%) 이나 PSNR −1.13 dB
   (우위 35.3%)·nMSE 2.6배 — train+val 누수에도 우리 프로토콜(16코일 절단·재-FFT)에서 domain shift 가 큼(정본 12장 관찰과 일치).
   PromptMR+ 1,486/7,334 시점 중간(464 볼륨 전부 포함) SSIM 0.9497 / PSNR 36.38 / nMSE 0.497 % — 확정은 완주 후.
   **09-08 09:22 — U-Net† 전체 완료**(7,334, non-finite 0): 볼륨 SSIM 0.8971±0.0366 / PSNR 30.95±2.29 / nMSE 0.973±0.796 % — 세 지표 전부
   원 bi-GRU 에도 미달(SS2D 대비 우위 슬라이스 SSIM 7.7 % / PSNR 3.9 %). PromptMR+ 6,801/7,334 시점 중간 볼륨 SSIM 0.9411 / PSNR 36.15 / nMSE 0.527 %.
   **▶ 논문 반영(09-08, 사용자 지시 "표 자리를 남겨두고 반영")**: `make_tables.py` 에 Table C4 참고선 생성기 추가(`paper/tables/tableC4_reference.{md,tex}` +
   IEIE 블록 `ieie_table_ref_block.md`; 미완주 방법은 자동 `[TBD]` 셀, 완주 시 재실행만으로 확정) → 학술지판 신설 Ⅳ장 6절 "공개 모델 참고선"(표 4, 이후 표·절 번호
   +1)·표 2 note·보강실험 (4)·고찰 "공개 모델 참고선의 해석" 문단·서지 `xin2024rethinking`(ECCV 2024) 추가, 학술대회판 한 문장(1.99 쪽 유지). PromptMR+ 행·간격 수치는 완주 후 `[TBD]` 교체.
   **native 프로토콜 행**(원본 코일·해상도·공식 마스크)은 계속 GPU 큐 7단계 몫(`eval_paired_baselines.py` 의 `native_protocol`).
4. ms/slice·peak VRAM 측정은 GPU 큐 7단계에서 별도 채집 (Table 5) — CPU 풀런의 시간은 지연시간 지표로 쓰지 않는다.
5. DDS(또는 CM-RED): 표본 299 만이라도 — NFE=50 기준 ms/slice 대비가 목적. 풀 7,334 는
   diffusion 에선 비현실적(수일)이므로 표본 + 명시가 정직.

## 4. 논문 서술 포인트

- §3.7 에 PromptMR+ 행 추가: "**train-only 학습 공개 가중치** — 본 검증셋에 대한 누수 없음" —
  U-Net/VarNet 누수 캐비엇과 대구.
- 예상 결과: PromptMR+ SSIM 이 우리·VarNet 보다 높을 것(공식 test 4x 0.9615). 프레임은 기존대로
  "SOTA 경쟁 아님 — 품질/지연/파라미터 트레이드오프 좌표 제시". PromptMR+ 는 12-cascade unrolled
  (반복 sens+DC)라 지연시간에서 직접 변환 계열이 유리할 것 — Table 5 에서 확인.
