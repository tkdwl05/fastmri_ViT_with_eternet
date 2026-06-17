# v7 실험 — TITAN x2 환경 마이그레이션 + Capacity 복원 + DDP

## 배경

v6 까지는 RTX 5060Ti 8GB VRAM 환경에서 학습되어 다음 제약이 있었다:
- `BATCH_SIZE = 4` (메모리 한계)
- ETER GRU hidden = 6 (원본 ETER-Net 의 10 보다 축소)
- `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'` 환경변수 필요
- Gradient checkpointing 강제 사용

새 머신은 **2x NVIDIA TITAN RTX 24GB + 128GB DRAM + 16TB HDD**. 8GB 제약이 사라졌고 multi-GPU 도 사용 가능.

기존 v6 코드와 ckpt 는 그대로 두고, 새 `v7/` 폴더에 신규 파일을 두어 (a) 환경 마이그레이션, (b) capacity 복원, (c) DDP 도입 을 순차 적용한다.

## v6 → v7 변경 매트릭스

| 항목 | v6 (8GB) | v7 (24GB single) | v7 DDP (24GB x2) |
|---|---|---|---|
| BATCH_SIZE | 4 | 16 | 16 per rank (eff. 32) |
| ETER GRU hidden | 6 | 10 | 10 |
| num_workers (train) | 4 | 16 | 16 |
| prefetch_factor | 2 | 4 | 4 |
| RESUME_CKPT | v5 ckpt | None (scratch) | None (scratch) |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments | (제거) | (제거) |
| 나머지 (ViT, DC, dropout, EarlyStop) | - | v6 동일 | v6 동일 |

ViT 인코더/디코더 크기, SS2D dim, DC block, dropout, augmentation, EarlyStop patience 등은 v6 그대로 유지.

## 파일 구조

```
v7/
├── configs/
│   ├── myConfig_choh_SS2D_model_v7.py
│   └── myConfig_choh_ETER_model_v7.py
├── main_train_ss2d_v7.py           # single GPU
├── main_train_eter_v7.py
├── main_train_ss2d_v7_ddp.py       # DDP (torchrun)
├── main_train_eter_v7_ddp.py
├── eval_v7_compare.py              # SS2D-v5 / ETER-v5 / UNet baseline + v7 컬럼 추가
├── visualize_v7_compare.py         # 2x5 grid (기존 4-모델 + v7 슬롯 1)
├── vis_v7_preview.py               # SS2D-v7 vs ETER-v7 (2x3 grid)
├── runs/
│   ├── extract_tars.sh             # tar.xz 11개 → files/ 해제
│   ├── extract.log
│   ├── ss2d/  eter/                # 학습 stdout/stderr
│   ├── chain/
│   │   ├── run_chain_v7.sh         # SS2D → ETER (single GPU)
│   │   └── run_chain_v7_ddp.sh     # SS2D → ETER (DDP)
│   └── eval/
└── README_v7.md
```

기존 `models/`, `dataloaders/dataloader_h5_v5.py`, `tools/` 는 import 로 재사용 (수정 없음).

## 사전 조건 (Step 1)

### 데이터 압축 해제 (HDD 제자리)

```bash
bash v7/runs/extract_tars.sh
```

- `/mnt/sda/choh/shared/data/FastMRI_brain/h5/*.tar.xz` 11개를 같은 폴더의 `files/multicoil_{train,val}/` 로 해제
- `xz -T0 -dc | tar -xf - -C files/ --skip-old-files` (이미 풀려있는 파일은 skip)
- 예상 시간: 1.5~2.5 시간 (CPU 24 core)
- HDD 사용량: 1.5TB 추가 (압축본은 보존)

### 심볼릭 링크

```bash
ln -s /mnt/sda/choh/shared/data/FastMRI_brain/h5/files \
      /home/snorlax/shared/fastmri_ViT_with_eternet/fastMRI_data
```

기존 코드의 상대경로 `./fastMRI_data/multicoil_{train,val}` 가 자동 해결.

## Step 2 — Capacity 복원 (single GPU)

### 1-epoch sanity test

```bash
cd /home/snorlax/shared/fastmri_ViT_with_eternet
SANITY_NUM_EPOCHS=1 SANITY_VAL_EVERY_N_EPOCHS=1 python v7/main_train_ss2d_v7.py
```

확인 사항:
- 학습이 OOM 없이 끝나는가
- VRAM peak ≤ 22GB (24GB 안전 마진)
- val SSIM 이 계산되어 출력되는가
- ckpt 가 `logs/SS2D_ViT_R4_brain320_v7/` 에 저장되는가

ETER 도 동일:
```bash
SANITY_NUM_EPOCHS=1 SANITY_VAL_EVERY_N_EPOCHS=1 python v7/main_train_eter_v7.py
```

### 풀 학습

```bash
bash v7/runs/chain/run_chain_v7.sh
```

또는 개별:
```bash
python v7/main_train_ss2d_v7.py     # 200 epoch, EarlyStop patience=10 val
python v7/main_train_eter_v7.py
```

### 평가 / 시각화

```bash
# 메트릭 (SS2D-v5/ETER-v5/UNet baseline + 옵셔널 v7)
python v7/eval_v7_compare.py \
    --ss2d-v7-ckpt logs/SS2D_ViT_R4_brain320_v7/ss2d_vit_best.pt \
    --eter-v7-ckpt logs/ETER_ViT_R4_brain320_v7/eter_vit_best.pt

# 4-모델 비교 + v7 슬롯 1 (2x5 grid)
python v7/visualize_v7_compare.py \
    --v7-model ss2d \
    --v7-ckpt logs/SS2D_ViT_R4_brain320_v7/ss2d_vit_best.pt

# SS2D-v7 vs ETER-v7 미리보기 (2x3 grid)
python v7/vis_v7_preview.py
```

## Step 3 — Multi-GPU DDP

### Sanity test (1 epoch on 2 GPU)

```bash
SANITY_NUM_EPOCHS=1 SANITY_VAL_EVERY_N_EPOCHS=1 \
torchrun --nproc_per_node=2 v7/main_train_ss2d_v7_ddp.py
```

확인:
- `nvidia-smi` 로 GPU 0, 1 모두 활성 (util > 80%)
- effective_BS = 32 출력 확인
- best ckpt 가 rank 0 에서만 한 번 저장

### 풀 DDP 학습

```bash
bash v7/runs/chain/run_chain_v7_ddp.sh   # 2 GPU 기본, NPROC=4 등으로 override 가능
```

## DDP ckpt 호환성

DDP 학습 시 `model.module.state_dict()` 로 저장 → `module.` prefix 제거된 형태. Single-GPU 평가 코드 (visualize / eval) 는 prefix 가 있어도 자동 strip 처리:

```python
state = torch.load(ckpt_path, map_location=device)
if any(k.startswith('module.') for k in state.keys()):
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
```

`v7/visualize_v7_compare.py`, `v7/vis_v7_preview.py`, `v7/eval_v7_compare.py` 모두 위 처리 포함.

## 주의사항

- **AXFLAIR 표기**: CLAUDE.md 가 "AXFLAIR multicoil" 이라고 적혀있으나 실제 전체 풀셋은 AXT1/AXT1POST/AXT1PRE/AXT2/AXFLAIR 혼합. eval 결과 해석 시 contrast 별 분리 측정 권장.
- **v6 ckpt 없음**: 새 머신에 v6 best ckpt 가 없어 v7 은 scratch 부터 시작. v6 (0.8903 val SSIM) 과 동등 수준 도달 여부가 v7 성공 지표.
- **HDD I/O**: 데이터는 HDD 에 있음. 첫 epoch 은 cold cache 로 GPU util 떨어질 수 있음. `iostat -x 2`, `nvidia-smi dmon -s u` 로 모니터링. GPU util 평균 < 60% 이고 HDD %util > 90% 이면 SSD 로 데이터 이전 검토.
- **LR scaling**: BS 4 → 16 (4x) 또는 DDP 시 effective BS 32 (8x) 로 증가하지만 LR 은 v6 의 2e-4 유지. 첫 run 결과 보고 4e-4, 8e-4 등으로 단계적 증가 검토 (linear scaling rule).
