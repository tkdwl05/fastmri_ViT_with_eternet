# v8 Pure ETER-Net — RNN(GRU) vs SS2D 통제비교 (no-DC)

**한 줄 결론**: 교수님 순수 ETER-Net에서 **sequence model만 GRU→SS2D 로 교체**하면 (다른 모든 것 고정, DC 없음) **SS2D 가 모든 지표에서 GRU 를 완승**한다 — best masked composite **0.9200 vs 0.9182**, 게다가 **파라미터 21× 적음(31M vs 668M)**. 25개 matched-epoch 전 구간에서 SS2D 가 한 번도 지지 않는 **wire-to-wire 우위**.

작성 2026-07-05 (SS2D no-DC ep50 완주 직후, 로그 기반). **갱신 2026-07-07**: per-slice paired 검증(전체 7334 슬라이스) + 4-way 시각화 완료 — §6. **갱신 2026-07-15**: DC 축 폐기 결정(문헌상 비표준 + SS2D+DC 이점 없음) — §7, 문헌 §10.

---

## 1. 목적 — v7_titan dead-heat 의 confound 제거

v7_titan(384) 에서 **SS2D-ViT ≈ ETER-ViT** dead-heat(composite 둘 다 0.9127)였다. 그러나 그 비교는 두 축이 얽혀 있었다:
- SS2D-ViT = **Mamba(SS2D) + 1-iter soft DC block**
- ETER-ViT = **GRU + no DC**

즉 "Mamba가 GRU보다 나은가?" 를 순수하게 답하지 못한다 (DC 유무가 confound). v8 은 **교수님 원본 순수 ETER-Net(ViT 없음)** 위에서 **sequence model 하나만** 바꿔 이 질문을 격리한다.

## 2. 설정 — 단일 변수 통제

| 고정 | GRU arm | SS2D arm |
|---|---|---|
| 데이터 | fastMRI brain, 384×384, R4, masked composite | (동일) |
| 백본 | 교수님 `ETER_hybrid_*_DFU` (ViT 없음, **DC 없음**) | (동일) |
| loss / opt / LR / epoch | masked L1 + (1−SSIM), 50ep, 동일 스케줄 | (동일) |
| dataloader / brain-mask / 지표 | 동일 | (동일) |
| **sequence model** | **GRU (668M)** | **SS2D/Mamba (31M)** ← 유일 변수 |

- 코드: 교수님 원본 무수정, wrapper 만 (`models/pure_eternet/u_pure_eternet_{gru,ss2d}.py`, `use_dc=False`).
- 2×2 설계 = {GRU,SS2D}×{no-DC, DC} 중 **no-DC 쌍**(교수님 ETER-Net 충실). DC 축은 §7 참조.
- best 기준 = masked **composite** = 0.5·SSIM + 0.3·(PSNR/40) + 0.2·(1−NMSE).

## 3. 결과 — 최종 best (완승)

| | epoch | composite | SSIM_m | PSNR(dB) | NMSE | L1 | params |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SS2D no-DC** | 48 | **0.9200** | **0.9140** | **35.16** | **0.0039** | **8.931** | **31M** |
| **GRU no-DC** | 50 | 0.9182 | 0.9126 | 35.03 | 0.0040 | 9.054 | 668M |
| **Δ (SS2D−GRU)** | | **+0.0018** | +0.0014 | +0.13 | −0.0001 | −0.123 | **21× ↓** |
| v7_titan(384) 참고 | | 0.9127 | ~0.9084 | ~34.6 | — | — | (ViT 트랙) |

- SS2D 가 **5개 지표 전부** 우위. ckpt: `logs/PureETER_SS2D_noDC_R4_brain384_v8/pure_ss2d_best.pt`(=ep48), `.../PureETER_GRU_noDC_.../pure_gru_best.pt`(=ep50).
- 두 arm 모두 v7_titan(0.9127) 을 크게 상회 → 순수 ETER-Net(no ViT) 가 ViT 트랙보다 이 세팅에서 강함.
- **per-slice paired 검증(§6)에서도 동일 방향 확인**: 전체 7334 슬라이스 중 SS2D 가 5개 지표 전부에서 74~78% 를 이김(Wilcoxon p≈0) — aggregate 우위가 소수 슬라이스에 좌우된 것이 아님.

## 4. matched-epoch 궤적 — wire-to-wire 우위

25개 val 지점(ep2,4,…,50) 전 구간에서 **SS2D ≥ GRU** (Δcomposite +0.0009 ~ +0.0058, 전부 양수). 전체 표: `results/eval/v8_nodc/matched_epoch_table.md`, 곡선: `results/eval/v8_nodc/curves_composite_ssim_psnr.png`.

대표 지점:

| epoch | GRU comp | SS2D comp | Δ comp |
|---:|---:|---:|---:|
| 2 | 0.8767 | 0.8798 | +0.0031 |
| 10 | 0.9023 | 0.9041 | +0.0018 |
| 20 | 0.9100 | 0.9118 | +0.0018 |
| 30 | 0.9146 | 0.9162 | +0.0016 |
| 40 | 0.9175 | 0.9188 | +0.0013 |
| 48 | 0.9180 | **0.9200** | +0.0020 |
| 50 | 0.9182 | 0.9198 | +0.0016 |

**v7_titan 과의 대조**: v7_titan(ViT 트랙)은 SS2D 가 ep10~30 앞서다 **ep40 에서 ETER 가 재추월(crossover)** → ep50 dead-heat 였다. v8(순수 ETER-Net)에는 **crossover 가 없다** — SS2D 가 처음부터 끝까지 앞선다.

## 5. 해석

1. **"DC 목발" 가설 반박** — v7_titan 동률이 "SS2D 가 DC 덕을 봤다" 때문이라는 가설은 틀렸다. **DC 를 완전히 뺀** 순수 세팅에서 SS2D 는 GRU 를 오히려 **더 확실히** 이긴다. Mamba 의 우위는 DC 와 무관한 sequence modeling 자체의 이득.
2. **파라미터 효율** — SS2D 31M 이 GRU 668M 을 이긴다(21× ↓). 순수 ETER-Net 의 GRU 는 668M 으로 과대(hidden 큼)한데도 성능은 뒤진다.
3. **v7_titan 서사 재정립** — ViT 트랙 dead-heat 는 두 축(Mamba vs GRU, DC 유무)이 상쇄된 결과였을 가능성. 순수 트랙에서 sequence model 축을 격리하니 SS2D 우위가 드러남.
4. **GRU 의 배경 ringing** — 4-way 시각화(§6)에서 GRU 는 두개골 바깥 배경에 반복적 ringing/줄무늬 아티팩트를 보이는 반면 SS2D 는 그 영역이 깨끗함. brain-mask 밖이라 정량 지표엔 안 잡히지만, GRU 재구성이 ROI 밖에서 덜 안정적이라는 추가 정성적 근거.

## 6. per-slice paired 검증 + 4-way 시각화 (완료, 2026-07-07)

완주 직후(2026-07-05 ~14:48) NVML 재다운으로 보류됐던 GPU 단계를 호스트 도커 재시작 후 실행 완료.

**per-slice win-rate** (전체 val set 7334 슬라이스, `v8_eter_pure/eval_paired_v8_nodc.py` → `results/eval/v8_nodc/{per_slice_paired.csv, win_rate_summary.md}`):

| 지표 | GRU mean±std | SS2D mean±std | SS2D win-rate | Wilcoxon p |
|---|---:|---:|---:|---:|
| SSIM_m | 0.9126±0.0894 | 0.9140±0.0890 | **78.2%** (5737/7334) | ≈0 |
| PSNR(dB) | 33.78±2.59 | 33.90±2.63 | **73.8%** (5412/7334) | ≈0 |
| NMSE | 0.0045±0.0053 | 0.0044±0.0053 | **73.8%** (5412/7334) | ≈0 |
| L1 | 9.00±3.05 | 8.88±3.05 | **76.1%** (5578/7334) | ≈0 |
| composite | 0.9087±0.0596 | 0.9104±0.0596 | **76.6%** (5621/7334) | ≈0 |

SS2D 가 **5개 지표 전부에서 슬라이스의 74~78%** 를 이긴다(Wilcoxon signed-rank, n=7334, p≈0 — 5개 전부 부동소수점 표시 한계 이하). §3 의 aggregate 우위가 소수 슬라이스에 좌우된 결과가 아니라 슬라이스 대다수에서 일관된 우위임을 확인.

⚠ **표의 절대값에 대한 주의**: 이 표의 PSNR/NMSE/L1(GRU 33.78dB 등)은 §3 학습-시점 best_val 값(GRU 35.03dB 등)과 다르다 — **버그가 아니라 평가 batch size 차이**. 학습 시 실제 `BATCH_SIZE=8`(smoke_bs.txt, config 기본값 2 아님) → val_loader 는 `max(1,BS//2)=4` 로 슬라이스 4개를 묶어 `run_val()` 이 PSNR/NMSE/L1 을 계산했다(`ref_max_in_mask`/`diff_sq_sum` 이 배치 4개에 걸쳐 pooled — 배치 내 최댓값 공유 등으로 PSNR 이 다소 부풀려짐). 반면 SSIM 은 `skimage_ssim_batch_masked` 내부에서 애초에 슬라이스별 loop 이라 batch size 와 무관 — 그래서 본 per-slice(batch=1) 재계산과 **SSIM 만 정확히 일치**(0.9126=0.9126, GRU 기준) 하고, 이는 이 clone 이 정확함을 뒷받침하는 근거다. **방향성(SS2D>GRU)은 두 계산 방식 모두 동일**하므로 §3/§5 의 결론에는 영향 없음.

**4-way 시각화** (`visualize_v8_pure_compare.py`[repo root] → `results/vis/v8_pure_eternet_compare/`, 12 슬라이스 `[0, 666, ..., 7333]`): sulci/혈관 디테일 자체는 두 모델이 육안으로 비슷한 수준이나(슬라이스별 PSNR 차 +0.1~+0.75dB, §5 의 "근소하지만 일관된 우위" 와 정합), **GRU 는 두개골 바깥 배경 영역에 반복적인 ringing/줄무늬 아티팩트**가 나타나는 반면 SS2D 는 그 영역이 뚜렷이 깨끗하다 — 확인한 4개 슬라이스(#1999, #3333, #4666, #5999) 전부에서 일관 관찰. brain-mask 밖이라 정량 지표엔 반영되지 않는 순수 정성적 차이 (§5-4 참조). 재현 명령은 §8.

## 7. DC 축 — 폐기 (2026-07-15)

**결정**: 2×2 의 DC 쌍(GRU+DC, SS2D+DC)을 **폐기**한다. no-DC 비교(§3~6)가 교수님 ETER-Net 에 충실한 **최종 비교**다. (2026-07-08 DC 쌍 학습을 개시했으나 아래 이유로 중단.) 근거 넷:

1. **문헌상 비표준.** 교수님 원본 ETER-net(Oh 2020, Med.Phys)엔 DC 가 없다(`docs/eternet_paper_data_consistency.md`). 문헌의 RNN+DC(CRNN-MRI Qin 2017; RIM Lønning 2019; RecurrentVarNet Yiasemis 2021; CIRIM Karkalousos 2021)와 Mamba+DC(MambaRoll Kabas 2024; SO-Mamba 2026)는 모두 **recurrence 가 최적화 반복을 unroll + DC 를 매 iteration interleave + 작은 unit** 인 구조다. 우리처럼 **거대 bi-GRU(도메인 변환) 뒤에 single soft-DC 를 한 번** 붙이는 건 문헌 표준이 아니다(§10).

2. **GRU+DC 학습 불안정(NaN).** 개시한 GRU+DC 가 ep4 에서 발산(train_loss=NaN). GPU 진단: 원인 = **DC 가 증폭한 UNet backward gradient 의 fp16 overflow** — forward 는 fp16=fp32 로 정상이고, GradScaler 기본 scale(65536)이 과도한 게 핵심(scale≤8192 면 유한). 별개로 supervisor 가 옛 런의 잔존 `학습 완료` sentinel 을 매치해 실패(rc=1)를 완주로 오인하는 버그도 발견.

3. **SS2D+DC 가 GRU+DC 보다 안정적이지 않음.** "SS2D 는 DC 를 잘 처리한다"를 통제 검증(fresh-init, 동일 배치, 150 step)한 결과 **두 arm 거의 등가** — GRU+DC overflow 2/150(scale 정착 16384) vs SS2D+DC overflow 3/150(scale 정착 8192; 오히려 미세하게 더 overflow-prone). 둘 다 GradScaler 자동 backoff 로 nan 없이 안정. SS2D+DC 통합 자체는 수학적으로 올바르나(sanity 16/16, §9.1) 특별한 이점이 없어, DC 축이 두 모델을 유의미하게 구분하지 못한다.

4. **novelty.** SSM/Mamba 기반 MRI 재구성은 이미 성숙·경쟁적 분야(MambaRecon, MambaMIR, DH-Mamba, MMR-Mamba 등 2024~2026, §10). 본 트랙의 기여는 새 아키텍처가 아니라 **교수님 ETER-Net 골격에서 RNN→SSM 치환의 통제된 ablation(no-DC)** 이다.

**향후 DC 재개 시**: 문헌식(interleaved DC cascade + 작은 recurrent unit)으로 재설계, fp16 학습이면 GradScaler scale 상한(≤8192) 또는 fp32. 기존 DC 코드·sanity(§9.1)·`sanity_dc_v8.py` 는 보존.

## 8. 재현

```
# 학습 (교수님 원본 무수정, SEQ_MODEL×USE_DC env 토글)
configs/... : v8_eter_pure/configs/myConfig_pure_eter_v8.py
train       : v8_eter_pure/main_train_pure_v8.py   (true-resume full-state)
models      : models/pure_eternet/u_pure_eternet_{gru,ss2d}.py  (use_dc 토글)
supervisor  : v8_eter_pure/runs/run_pure_v8_autoresume.sh
재개(1cmd)  : PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SEQ_MODEL=ss2d USE_DC=0 \
              CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online bash v8_eter_pure/runs/run_pure_v8_autoresume.sh

# ckpt (best)
logs/PureETER_SS2D_noDC_R4_brain384_v8/pure_ss2d_best.pt   (ep48, comp 0.9200)
logs/PureETER_GRU_noDC_R4_brain384_v8/pure_gru_best.pt     (ep50, comp 0.9182)

# 정량 분석 (로그 파싱, GPU 불필요)
python v8_eter_pure/analyze_v8_nodc.py
  → results/eval/v8_nodc/matched_epoch_table.md
  → results/eval/v8_nodc/curves_composite_ssim_psnr.png

# per-slice paired win-rate (GPU, 전체 val, ~2h) — §6
python v8_eter_pure/eval_paired_v8_nodc.py
  → results/eval/v8_nodc/per_slice_paired.csv
  → results/eval/v8_nodc/win_rate_summary.md

# 4-way 시각화 (GPU, 12 슬라이스, ~1분, repo root) — §6
python visualize_v8_pure_compare.py
  → results/vis/v8_pure_eternet_compare/compare_*.png
  → results/vis/v8_pure_eternet_compare/metrics_summary.txt
```

## 9. 코드 검증 (2026-07-08) 및 개선점

### 9.1 DC 경로 사전검증 — 16/16 PASS

DC 쌍 학습 개시와 동시에, 실학습 이력이 없던 DC 경로를 CPU-only(학습 중 GPU 무접촉)로 사전검증했다. 스크립트 `v8_eter_pure/sanity_dc_v8.py` — **SS2D+DC 자동 시작 전(GRU+DC 완주 무렵) 재실행 권장**.

- **DCBlock 물리 항등식** (실 val 데이터, rel err ~1e-7): ① α=0 → out==x (항등) ② mask=0 → out==x (항등) ③ mask=1·α=1 → 해석해 `Σ ifft(k_meas·100)·sens*` 일치 ④ α=1, zero-filled 시작 → sampled-line k-residual 7.5%↓
- **스케일/컨벤션 정합**: `iFFT(ksp×100) == data_img` (val_amp 1e4/1e6 비율 = `DC_K_SCALE_RATIO=100`, rel err 5e-8) · FFT centering(ifftshift→ortho→fftshift) dataloader↔DCBlock 동일 · sens 의 ACS 추출 윈도우 = mask 중앙 샘플링 밴드(cols 177:208) 정확 정렬 · Σ|sens|²=1 (brain 내)
- **flip aug 와 DC 물리**: dataloader 가 flip 을 image 도메인에서 적용한 **뒤** FFT 로 k-space 재계산(`dataloader_h5_v5.py` step 3→4 순서)이므로 augmented 샘플에서도 DC 의 FFT 정합이 유지됨 — 이미지/k-space 를 따로 뒤집는 구현이었다면 DC arm 만 조용히 망가졌을 지점.
- **통제 불변식 수치 확인**: DC−noDC param 차 = 118 = `UNet_choh_skip.last`(in 116ch) 출력 1ch 추가(117) + DC α(1) — DC 축이 출력 head 와 DCBlock 외 아무것도 바꾸지 않음.
- **GRU+DC 풀 forward** (CPU, 실 데이터): 출력 (1,1,384,384) finite/양수, masked L1 계산 정상. supervisor '학습 완료' sentinel 오염 없음(GRU+DC 로그 0건, SS2D+DC 로그 미존재).

잔여 리스크는 코드가 아니라 **SS2D+DC 의 VRAM** 뿐: noDC 가 23.0GB 로 빠듯했고 DC 의 fp32 FFT 버퍼가 추가됨 — `expandable_segments` 로 완화, 실확인은 SS2D+DC 기동 시.

### 9.2 개선점

**적용됨 (2026-07-08 — 수치 무영향, I/O 견고성)**:
- `best.pt` / `epoch_N.pt` 저장 **atomic 화** (`save_checkpoint_atomic`: tmp 파일 → `os.replace`). 저장 도중 crash 시 ckpt 파손 창 제거 — `last.pt` 는 원래부터 atomic 이었고 best/epoch 스냅샷만 빠져 있었음. 실행 중인 GRU+DC 프로세스는 메모리에 올린 구 코드로 계속 돌고, supervisor 재시작 또는 SS2D+DC 시작부터 적용된다(학습 수치 영향 0 → 통제비교 무손상).

**차기 재학습부터 적용 (지금 적용 금지 — noDC 쌍과 "everything else identical" 유지가 우선)**:

전제: 2026-07-08 실측 GPU util 99% 포화 — 아래는 wall-time 개선일 뿐 결과 수치와 무관하며, 상한도 크지 않다.

1. **H2D 전송 `non_blocking=True`** (상한 ~4%): 배치당 ~470MB(BS=8, 6텐서 fp32)의 동기 복사 ~40ms 를 GPU 연산과 중첩. `pin_memory=True` 가 이미 켜져 있어 `.to(device, non_blocking=True)` 로 바꾸기만 하면 됨(train/val 각 6줄).
2. **`torch.backends.cudnn.benchmark = True`** (UNet conv 한정 5~15%, **조건부**): 입력 shape 고정(384², BS 고정)이라 이득 조건은 충족. 단 ① 알고리즘 탐색/선택에 따른 cudnn workspace VRAM 스파이크 — ETER v6 에서 workspace 증가로 첫 forward OOM → BS 8→4 강하했던 전례(`docs/eter_v6_changes.md`), ② 현재의 "step1 max 도달 후 평평 = OOM 예측 가능" 성질 상실, ③ run 간 bit-level 재현성 상실. → **23GB+ 빠듯한 SS2D 계열 런에는 금지**, VRAM 여유 런에만 선별 적용. GRU 의 cuDNN RNN 커널과 SS2D 의 mamba custom 커널은 이 플래그와 무관(효과는 DFU U-Net conv 에만).
3. **val 의 per-slice 화**: `run_val` 의 PSNR/NMSE/L1 이 val batch(=BS//2=4)에 pooled 되는 §6 주의사항의 근본 해소. per-slice 로 통일하면 학습-시점 val 과 사후 paired 평가가 같은 좌표계가 되어 §6 식의 절대값 불일치 자체가 사라짐. skimage SSIM 의 CPU 실행(val ~25분/회의 일부)도 이때 함께 재검토.
4. (미미) wandb per-batch 로깅 → every-N throttle, tqdm 로그 파일 비대화 억제.

## 10. 관련 문헌 (RNN/SSM + Data Consistency, 2026-07-15 조사)

DC 축 폐기 근거(§7)의 문헌 배경. Consensus 검색 기반.

- **교수님 ETER-net**: Oh et al., "A k-space-to-image reconstruction network for MRI using recurrent neural network", *Med. Phys.* 2020 — bi-RNN 도메인 변환, **DC 없음**.
- **RNN + DC (unrolled·interleaved)**: Qin et al. *CRNN-MRI* (IEEE TMI 2017, 566 cite); Lønning et al. *RIM* (Med. Image Anal. 2019); Yiasemis et al. *RecurrentVarNet* (CVPR 2022); Karkalousos et al. *CIRIM* (PMB 2021). 공통점: 작은 recurrent unit × DC 매 iteration.
- **Mamba/SSM MRI 재구성**: Korkmaz et al. *MambaRecon* (WACV 2025); Huang et al. *MambaMIR* (Med. Image Anal. 2024); Meng et al. *DH-Mamba* (IEEE TCSVT 2025); Zou et al. *MMR-Mamba* (Med. Image Anal. 2024). — 이미 성숙·경쟁적 분야.
- **Mamba + DC (unrolled)**: Kabas et al. *MambaRoll* (2024, physics-driven unrolled SSM); Fang et al. *SO-Mamba* (2026, DC-coupled unrolled).
- **SS2D 비전 백본**: Ruan et al. *VM-UNet* (ACM TOMM 2024).

---

관련: `docs/eternet_paper_data_consistency.md`(교수님 ETER-Net 엔 DC 없음), `docs/summary_2026-06-11.md`(v7_titan dead-heat), `docs/eval_metric_redesign.md`(masked composite 정의).
