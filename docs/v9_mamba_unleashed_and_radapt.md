# v9 트랙 — 언리시드 Mamba(품질) + R-대응 Mamba(일반화)

작성 2026-07-21. **상태: 코드·구조 검증 완료(sanity+스모크 PASS), 학습 미시작(GPU idle).**
**성능 엔지니어링 확정(§10)**: 초기 config(384² 풀해상도 스캔)는 compute-bound(19 h/ep)였으나, ①스캔
**fp16 화**(라이브러리 최적화) + ②**다운샘플 front-end(ds=3, 128² 스캔)** 로 **풀 용량 유지하며 v8보다
빠름**(2.51 h/ep). 최종 백본 = d_inner256·d_state32·3블록·게이팅·**ds=3**·BS8·80ep.
계획: `/root/.claude/plans/shimmying-doodling-dusk.md`. 관련 근거: `docs/v8_eter_pure_rnn_vs_ss2d.md`(no-DC
SS2D 완승), `docs/v8_ss2d_kspace_domain_review.md`(R 일반화 §4), `docs/eternet_paper_data_consistency.md`.

---

## 1. 왜 (동기)

v8_eter_pure 의 SS2D 는 GRU 와의 **공정 통제비교**를 위해 일부러 최소로 조여져 있었다 —
`out_ch=20`(GRU out_v 채널에 강제 정합), `d_state=16`(v7_titan 은 32), **단일 블록**, 공식 Mamba
블록의 **게이팅(z) 분기 생략**. 그 결과 31M 최소 SSM 이고, "Mamba 를 제대로 못 쓴다"는 인상은 정확했다.

v8 통제비교(no-DC SS2D 완승, comp 0.9200 vs GRU 0.9182, 21×↓ params)는 **완주·최종 deliverable 이라
건드리지 않는다.** v9 는 그와 **분리된 신규 트랙**으로, 사용자 결정에 따라 **직교하는 두 축**을 각각
별도 변형으로 구축한다:

| 변형 | 목표 | 질문 |
|---|---|---|
| **A. unleashed** | R4 고정 **품질 최대화** | "Mamba 를 제대로 쓰면 이 파이프라인에서 재구성 품질이 어디까지?" |
| **B. radapt** | 여러 R **재학습-없이 대응** | "측정연산자 명시 + DC + multi-AR 로 R-일반화가 되는가?" (LMO/오퍼레이터 논의의 실용 근사) |

**중요 — 두 축은 직교**: unleashed 는 R4 를 *더 잘* 하지만 R-일반화는 안 준다(오히려 용량↑ = R4 과적합
심화 가능, §v8_ss2d_kspace_domain_review §4.5 가 단일-R 과적합 실측). radapt 는 R 전역 robustness 를
노리되 R4 절대품질은 일부 양보할 수 있다.

---

## 2. 공통 백본 — 강화 SS2D (`models/mamba_eternet/ss2d_v9.py`)

원본 `ss2d.py`(`SelectiveScan1D`, mamba_ssm CUDA 커널 래퍼)는 **무수정 import 재사용**. v8 SS2D 대비
3가지 복원/확장:

1. **게이팅(z) 분기 복원** — 공식 Mamba 의 `y = y·SiLU(z)` 곱. v8 엔 없던 selective 블록의 핵심.
2. **residual 스택** — 채널수 불변(d_inner) residual 블록을 N개 쌓아 깊이 확보(Mamba 진짜 강점).
3. **병목 해제** — head 의 `out_ch` 가 더 이상 20 강제 아님(자유).

```
SS2DBlockV9(d_inner):           # 채널 불변, residual, 게이팅
  residual = x
  x = LayerNorm(x)
  x_ssm, z = in_proj(x).chunk(2)        # ← 게이팅 분기
  x_ssm = SiLU(dwconv(x_ssm))
  y = 4방향 selective-scan(x_ssm) → merge(4d→d)
  y = y * SiLU(z)                        # ← 게이팅 곱 (v8 누락분)
  return residual + Dropout(out_proj(y))

SS2DStackV9:
  stem:  LayerNorm(c_in) → Linear(c_in→d_inner) → SiLU
  body:  N × SS2DBlockV9
  head:  LayerNorm → Conv1x1(d_inner→out_ch)   # out_ch 자유
```

**관찰(중요)**: v9 총 params ≈ **33.0M** 인데 SS2D 스택은 ~2M 뿐이고 **U-Net DFU(~30M)가 지배적**(v8 도
동일 구조). GRU 가 668M 였던 건 flatten reshape 탓. 즉 이 ETER-net 파이프라인에서 "언리시드"의 실체는
param 수가 아니라 **k→image 변환의 질**(게이팅·깊이·d_state). 더 키우려면 d_inner 512 / n_blocks 6+
로 밀 수 있으나 U-Net 이 여전히 지배 → 후속 ablation 축.

---

## 3. 변형 A — unleashed (`v9_mamba_unleashed/`)

```
x_ksp → SS2DStackV9(out_ch=64, 게이팅+3블록) → cat(aliased image) → UNet_choh_skip(DFU) → magnitude
```
DC 없음, 단일 R4 학습. downstream 계약: `n_hidden*2 == out_ch+2coil` → `out_ch=64 → in=96 → n_hidden=48`.

| 파일 | 내용 |
|---|---|
| `models/pure_eternet/u_pure_eternet_ss2d_v9.py` | `PureETER_SS2D_V9` 래퍼 |
| `configs/myConfig_ss2d_v9.py` | d_inner 256, d_state 32, n_blocks 3, out_ch 64, dropout 0.05 |
| `main_train_ss2d_v9.py` | v8 trainer 클론 + **optimizer no-WD 그룹** + **DONE sentinel** |
| `sanity_ss2d_v9.py` | 계약·게이팅·no-WD·원본무수정 (✅ PASS, 33.0M) |
| `runs/run_ss2d_v9_autoresume.sh` | supervisor(DONE 파일 기반) |

## 4. 변형 B — radapt (`v9_mamba_radapt/`)

언리시드 백본 + **오퍼레이터식 R-일반화 요소 3가지**(model-based DL 근사):

```
x_ksp ─┬─(mask concat)→ SS2DStackV9(c_in=33) → cat(aliased) → UNet(complex, n_classes=2) → DCBlock → |·|
 mask ─┘
```
1. **마스크 명시 조건화** — sampling mask 를 seq 입력에 채널 concat(c_in 32→33) → "어떤 R 인지" 명시(§4.2-②).
2. **Data Consistency** — 끝에 1-iter soft DCBlock(측정 앵커, §4.2-①). v8 `u_choh_model_SS2D_ViT_v4.DCBlock` 재사용.
3. **multi-AR 학습** — 배치별 R∈{2,3,4,5,6,8} 랜덤(§4.2-⑤). val 은 R4 고정(best-ckpt 비교 기준).

| 파일 | 내용 |
|---|---|
| `dataloaders/dataloader_h5_v9_multiAR.py` | `FastMRI_H5_MultiAR` — v5 서브클래스, `__getitem__` 마다 R 랜덤(원본 무수정) |
| `models/pure_eternet/u_pure_eternet_ss2d_v9_radapt.py` | `PureETER_SS2D_V9_Radapt`(마스크조건화+DC) |
| `configs/myConfig_ss2d_v9_radapt.py` | 백본 동일 + AR_CHOICES, MASK_CONDITION, DC_*, GRADSCALER_INIT_SCALE |
| `main_train_ss2d_v9_radapt.py` | radapt trainer(multi-AR loader + **α clamp[0,1] + GradScaler init_scale 8192** + no-WD) |
| `sanity_ss2d_v9_radapt.py` | 마스크조건화(c_in=33)·DC·게이팅·no-WD·multiAR (✅ PASS, 33.0M) |
| `runs/run_ss2d_v9_radapt_autoresume.sh` | supervisor |

**DC 안정화(v8 NaN 진단 반영)**: v8 DC 쌍이 ep4 NaN 좌초한 원인 = (1) α overshoot(1.0→1.35)→fp16
forward non-finite, (2) DC-증폭 gradient 의 U-Net fp16 backward overflow(GradScaler 기본 scale 65536
과도, ≤8192 유한). radapt 는 **α clamp[0,1]**(overshoot 물리 차단) + **GradScaler init_scale=8192** +
**NaN-skip self-heal** 3중으로 방어. DCBlock forward 자체는 fp32(autocast off)라 안전.

---

## 5. 공통 개선 (두 변형 공유)

- **Optimizer no-WD 그룹**: Mamba `A_log`/`D`(및 `dt_proj.bias`)의 `_no_weight_decay` 플래그를 존중해
  weight_decay 제외(v8 은 `Adam(model.parameters(), wd=3e-5)` 로 무시했음). 정석 Mamba 위생.
  *효과는 wd=3e-5 로 작지만 올바름.* sanity 로 no-WD 파라미터 24개(=A_log·D × 4방향 × 3블록) 확인.
- **DONE sentinel 파일**: supervisor 완료 판정을 append-log `grep '학습 완료'`(v8 BUG2 = 옛 런의 stale
  sentinel 오탐) → `logs/<run>/DONE` 파일 존재로 교체. 재시작 안전.
- **원본 무수정 불변식**: `ss2d.py`·`myUNet_DF.py`·`u_choh_model_SS2D_ViT_v4.py`·`dataloader_h5_v5.py`
  는 import/서브클래스만. sanity git 체크로 강제.

---

## 6. 데이터/지표 정합 (v8 과 비교 가능)

두 변형 모두 384·brain-mask·masked(L1 + 1−SSIM) loss·composite(0.5·SSIM+0.3·PSNR/40+0.2·(1−NMSE))
= v8 no-DC 와 동일. 비교 레퍼런스(동일 val): **v8 SS2D 0.9200/ssim_m 0.9140, v8 GRU 0.9182/0.9126**.
LR 2e-4 / CosineAnnealingLR / 50ep / grad-clip 1.0 / AMP.

---

## 7. 실행 (학습 전 스모크·확인 필요)

```bash
# 1) CPU sanity (구조/계약/게이팅/no-WD/원본무수정) — 이미 PASS
CUDA_VISIBLE_DEVICES="" python v9_mamba_unleashed/sanity_ss2d_v9.py
CUDA_VISIBLE_DEVICES="" python v9_mamba_radapt/sanity_ss2d_v9_radapt.py

# 2) GPU 스모크 (BS 탐색·forward+backward finite) — 미실행
SANITY_NUM_EPOCHS=1 SMOKE_BS=2 python v9_mamba_unleashed/main_train_ss2d_v9.py   # (일부 배치만)

# 3) 학습 launch (GPU0 단독, 스모크 통과 후) — 미실행
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online \
  setsid nohup bash v9_mamba_unleashed/runs/run_ss2d_v9_autoresume.sh \
  > v9_mamba_unleashed/runs/run_ss2d_v9.log 2>&1 < /dev/null & disown
# radapt 는 run_ss2d_v9_radapt_autoresume.sh (단일 GPU → unleashed 완주 후 순차 또는 별도 결정)
```

## 8. 평가 계획

- **unleashed**: 완주 후 `v8_eter_pure/eval_paired_v8_nodc.py` 재사용 → v8 SS2D/GRU 대비 per-slice
  win-rate + 4-way viz. **성공 기준 = composite/ssim_m 이 v8 SS2D(0.9200/0.9140) 상회.**
- **radapt**: `v8_eter_pure/eval_r_generalization_v8.py` 패턴으로 **R-sweep(R∈{2,4,6,8})**.
  **성공 기준 = radapt 의 R2/6/8 하락폭이 단일-R(unleashed/v8)보다 작아 R-sweep 곡선이 평탄**(matched-R4
  는 다소 못 미쳐도 무방 — 용량 일부를 일반화에 씀).

## 9. 리스크 / 한계

- **메모리(최대 리스크)**: d_inner 256×3블록×4방향 fp32 scan → 스모크로 BS 확정. OOM 시 config
  `SS2D_USE_CHECKPOINT=True`(블록별 gradient checkpointing) 또는 BS↓+ACCUM↑. GPU1 절대 미사용.
- **radapt DC 재발 가능성**: 3중 방어(α clamp·init_scale·NaN-skip)에도 재발 시 fp32 DC-forward fallback.
- **정직한 한계**: radapt 는 완전한 neural-operator(FNO/LMO 식 이산화 불변)가 아닌 model-based 근사 —
  절대 일반화가 아니라 **단일-R 대비 상대 robustness** 가 목표. 진짜 오퍼레이터(spectral operator layer)는
  v10 연구로 분리.
- **비교 해석**: v9 는 통제비교가 아님 — "SS2D>GRU" 근거로 쓰지 말 것(그건 v8). v9 질문은 "강화/대응
  Mamba 의 상한".

---

## 10. 성능 엔지니어링 (2026-07-21) — compute-bound 해결

초기 config(384² 풀해상도 SS2D 스캔)는 학습 불가(19 h/ep, 50ep≈39일)였다. 두 축으로 해결:

### 10.1 스캔 fp16 화 (라이브러리 최적화)
- 원본 `ss2d.py::SelectiveScan1D` 는 스캔을 **강제 fp32**(주석 "fp32 필요"). 그러나 mamba_ssm
  `selective_scan_fn` 은 u/delta/B/C 를 **fp16/bf16 로 받고 내부 recurrence 는 fp32 누적**(안정).
- 벤치(BS4 대표 스캔): fp32 296ms·**7.88GB** → fp16 224ms·**3.94GB**(메모리 정확히 절반·~25%↑·finite).
- v9 전용 `SelectiveScan1DV9`(`ss2d_v9.py`) 신설 — **원본 무수정**, 동일 수식·초기화, fp32 강제만 제거.
  함정: autocast 가 softplus 를 fp32 승격 → u/delta dtype 불일치(커널 에러) → delta/B/C 를 u.dtype 통일.

### 10.2 다운샘플 front-end (아키텍처)
- 근본 병목 = **384² 풀해상도 스캔**(스캔이 시간 ~65%, cost ∝ 면적). fp16 만으론 여전히 18.8 h/ep.
- stem(384²)에서 k-space 전체 → feature 추출 후 **feature 를 ds×ds 다운샘플 → coarse grid 스캔 →
  bilinear 업샘플 → U-Net**. SSM=전역문맥(coarse 적합), U-Net=풀해상도 디테일 분업(VMamba 정석).
  **입력 k-space 정보 손실 없음**(stem 이 먼저 full-res 처리).
- 스모크(풀 용량 256/32/3블록, fp16, no-ckpt):

  | ds | 스캔 res | BS | peak | h/ep |
  |---:|---|---:|---:|---:|
  | 1 | 384² | 2 | 21.7 | 18.8 ❌ |
  | 2 | 192² | 4 | 14.1 | 4.99 |
  | **3** | **128²** | **8** | **16.7** | **2.51** ✅ |
  | 4 | 96² | 8 | 12.9 | 1.80 |

- **채택 ds=3**(사용자 결정): v8(2.78 h/ep)보다 빠르면서 다운샘플 최소. 속도 여유로 **epoch 50→80** 상향.

### 10.3 최종 config (두 변형 공통 백본)
d_inner256 · d_state32 · n_blocks3 · 게이팅 · out_ch64 · **ds=3** · fp16 스캔 · BS8 · 80ep · ~34M params.
최종 스모크: unleashed 2.51 h/ep·16.7GB, radapt 2.61 h/ep·17.2GB (둘 다 v8 보다 빠름).
**품질 미검증**: 다운샘플이 정량지표에 주는 영향은 학습 후 실측(ds=2/4 는 ablation 후보). d_state 32→16 이
스캔 cost 의 최대 레버였음(§ 스윕). → **§11 에서 해소**(ds=3 로 v8 돌파, ds=2 ablation 불필요 판정).

## 11. 결과 — unleashed 80ep 완주 (2026-08-05 최종 갱신; 07-23 라이브 로그를 최종본으로 대체)

**완주**: launch 2026-07-21 13:02 UTC → **DONE 2026-07-30 22:51** (80ep, early_stop 없음, ~9.4일 — 실측 ~2.51 h/ep 유지).

### 11.1 최종 성적 — 목표(v8 SS2D 0.9200 돌파) 달성 (근소)

| | composite | SSIM_m | PSNR | NMSE | L1 |
|---|---:|---:|---:|---:|---:|
| **v9 best ckpt (ep78)** | **0.9203** | 0.9145 | 35.18 | 0.0039 | 8.9314 |
| v8 SS2D best (ep48) | 0.9200 | 0.9140 | 35.16 | 0.0039 | 8.9311 |
| v8 GRU best (ep50) | 0.9182 | 0.9126 | 35.03 | 0.0040 | 9.0542 |

- 0.9203 최초 도달 ep72(**ssim_m 최고 0.9147**), best ckpt 저장분은 ep78(동률 갱신), 최종 ep80 = 0.9201.
- 로그 기준 L1 은 v8-SS2D 와 사실상 동률(8.9314 vs 8.9311) — 단 per-slice 평균 l1 은 v9 우위(8.879 < 8.883, win-rate 54.4%).
- **ds=3 다운샘플 우려 해소**: ep40 시점 비관(0.9146, v8 대비 열위) → 후반 cosine anneal 로 역전.
  **ds=2 ablation 불필요** 판정.

### 11.2 per-slice paired 검증 (2026-08-05, `v9_mamba_unleashed/eval_paired_v9.py`)

전체 val **7334 슬라이스**에서 v9 best 를 추론, v8 no-DC per-slice CSV(`results/eval/v8_nodc/`)와
`(file, slice_idx)` 조인(**7334/7334 전건 매칭**). 산출 → `results/eval/v9_unleashed/`.

| 비교 | 5지표 win-rate | Wilcoxon |
|---|---|---|
| **v9 vs v8-SS2D** | **54.2~56.0% — 5지표 전부 v9 승** | p ≤ 4e-13 (유의하나 근소) |
| v9 vs v8-GRU | 78.4~82.3% 완승 | p ≈ 0 |

- 정합 sanity: per-slice ssim 평균 **0.9145 = 학습 로그 ep78 val_ssim_m 정확 일치**(v8-SS2D 0.9140,
  GRU 0.9126 도 각자 로그와 일치). composite 절대값(0.9107)이 로그(0.9203)보다 낮은 것은 per-slice vs
  배치풀링(학습 val BS=4, 배치 공유 ref-max) 정의 차이 — v8 도 동일 오프셋(0.9104 vs 0.9200), paired 비교는 유효.
- 해석: **v9 는 v8-SS2D 를 전 지표에서 통계적으로 유의하게 이기지만 격차는 근소(≈56%)** — "동급+α".
  GRU 대비는 v8-SS2D 때(74~78%)보다 더 벌어진 **78~82%**.

### 11.3 정직 주석 — 우위는 80ep 연장 구간에서

- **matched-ep50 시점 v9 = 0.9171 < v8-SS2D@ep48~50(0.9200)**. v9 가 v8 best 에 도달한 최초 epoch = **ep70**.
- ep당 시간은 v9 가 빠르지만(2.51 vs 2.78 h/ep) **best 도달 wall-clock 은 v9 ≈181h > v8-SS2D ≈133h**.
- 즉 "같은 학습량에서 더 좋다"가 아니라 "**ep당 속도 이득으로 더 긴 스케줄(80ep)을 소화해 최종 품질을
  넘었다**"가 정확한 서사. 강화 SS2D(게이팅·3블록·병목해제)+ds=3 의 순수 아키텍처 이득은 근소.
- 로그기반 3-way 곡선·matched-epoch 표: `v9_mamba_unleashed/analyze_v9_unleashed.py` →
  `results/eval/v9_unleashed/{curves_v9_vs_v8.png, matched_epoch_table_v9.md, win_rate_summary_v9.md}`.

### 11.4 인프라 사건 — radapt 자동 체인 좌초 → 08-05 재기동

- 07-23 nvidia-smi `NVML Unknown Error` 재발([[host-nvml-issue]]) — **학습 프로세스는 CUDA 컨텍스트
  유지로 생존**, 완주까지 무사.
- **07-30 22:52 체인이 radapt 자동 launch → 새 프로세스가 CUDA 획득 실패**(`RuntimeError: CUDA GPU 필수`,
  wandb.init 이전이라 쓰레기 run 없음). supervisor 50회(~52분) 소진 → 23:44 체인 중단(radapt ckpt 0개).
  07-29 경고했던 리스크가 정확히 현실화.
- **08-05 host docker restart 로 NVML 복구** 확인 후 radapt supervisor **단독 재기동**(scratch,
  `MAX_RETRY=200` env 오버라이드로 flake 자가복구 창 ~52분→~3.5h 확대). ETA 80ep×2.61 h/ep ≈ 8.7일 →
  **~08-13/14 완주 예상**. 이전 실패 로그 `v9_mamba_radapt/runs/ss2d/*.log.failed-20260730` 는 2026-09-03 정리에서 삭제(NVML 좌초 사건 자체는 §이 절·`docs/cleanup_log.md`·memory 에 기록됨).

---

관련: 계획 `shimmying-doodling-dusk.md`, `docs/v8_eter_pure_rnn_vs_ss2d.md`, `docs/v8_ss2d_kspace_domain_review.md`.
CLAUDE.md 트랙표·`docs/INDEX.md` 최종값 반영 완료(2026-08-05).
