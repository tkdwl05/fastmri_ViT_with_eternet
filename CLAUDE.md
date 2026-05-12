# ViT-based MRI Reconstruction

## 프로젝트 개요

fastMRI brain AXFLAIR multicoil 데이터에 대한 MRI 재구성 모델 연구.
ViT 인코더 + 시퀀스 모델 디코더(GRU 또는 SS2D) 구조를 사용한다.

## 핵심 문서 (docs/)

작업 히스토리와 설계 판단의 근거를 기록한 문서들:

- **[docs/SS2D_v1_analysis.md](docs/SS2D_v1_analysis.md)** — SS2D-ViT v1의 blurry 복원 결과 원인 분석. patch_size 32x32의 정보 손실, Conv2d 1개 합성의 한계, SSIM weight 부족 등 7가지 문제를 진단하고 v2 해결 상태를 체크리스트로 정리.
- **[docs/eter_8gb축소.md](docs/eter_8gb축소.md)** — 원본 ETER-Net(RTX 서버, 384x384)을 8GB GPU(320x320)에 맞추며 GRU hidden 10→2, U-Net→Conv2d 1개로 축소한 과정과 그로 인한 성능 저하 원인 분석. 원본 설계 자체는 문제없고 축소 과정의 문제임을 확인.
- **[docs/architecture_ETER_vs_SS2D.md](docs/architecture_ETER_vs_SS2D.md)** — ETER-ViT(GRU)와 SS2D-ViT(Mamba) 아키텍처 상세 비교. 공통 파이프라인, 인코더/디코더 구조, 설정값, 학습 조건을 정리.
- **[docs/scheduler_change.md](docs/scheduler_change.md)** — LR 스케줄러를 `CosineAnnealingWarmRestarts`(톱니 LR)에서 `CosineAnnealingLR`(단일 부드러운 decay)로 교체한 근거와 적용 범위. SS2D v2→v3, ETER v3→v4 버전 분리, 체인 실행 스크립트 포함.
- **[docs/ss2d_v4_changes.md](docs/ss2d_v4_changes.md)** — SS2D v4에서 A(SS2D capacity 증설) + B(weight_decay/dropout) + C(1-iter soft Data Consistency block) 세 축을 동시 적용한 내역. `_v4` 접미사 신규 파일 5개(config/dataloader/model/train/chain), DC block 파이프라인, FFT AMP 처리, 체인 예약. §8: 첫 batch OOM 사후 수정(SS2D forward gradient checkpointing).
- **[docs/eter_v4_analysis.md](docs/eter_v4_analysis.md)** — ETER v4 200ep 결과 분석. v3(0.7475) 대비 v4 best val SSIM 0.7320 회귀, ep 30~40에 피크 후 단조 감소. 회귀 원인 가설(WarmRestarts 부재 / capacity ceiling / EarlyStopping 부재) 및 v5 계획(EarlyStop, weight_decay↑, dropout↑).
- **[docs/ss2d_v5_changes.md](docs/ss2d_v5_changes.md)** — SS2D v5: dataloader 사이즈 필터 완화(crop/pad → train +67%, val 7270 = U-Net 평가셋과 동일), Transformer dropout 0.1→0.2, weight_decay 1e-5→3e-5, H/V flip aug, EarlyStopping(patience=5 val check). 모델 코드는 v4 그대로 import.
- **[docs/eter_v5_changes.md](docs/eter_v5_changes.md)** — ETER v5: SS2D v5 와 동일 레시피(공유 dataloader_h5_v5, dropout 0→0.2, weight_decay 1e-7→3e-5, flip aug, EarlyStop). 모델은 v4 클래스 상속한 thin wrapper로 decoder Transformer 에 dropout 주입. SS2D v5 chain 후 자동 시작.
- **[docs/ss2d_v6_changes.md](docs/ss2d_v6_changes.md)** — SS2D v6: v5 의 두 결함 수정 — (1) val SSIM 을 custom `u_choh_SSIM`(val_range 버그) → skimage `structural_similarity`(data_range=target.max−min) 로 교체, (2) EarlyStop/best 기준을 composite(SSIM+NMSE+PSNR+L1 평균) → val_ssim 단일로 단순화. patience 5→10, val 빈도 매 5 epoch, v5 epoch_10.pt 부터 resume. 데이터/모델/regularization 은 v5 그대로. 결과: 200ep 풀 학습 완료, best val SSIM 0.8903 (U-Net 0.8865 동등).
- **[docs/eter_v6_changes.md](docs/eter_v6_changes.md)** — ETER v6: SS2D v6 와 동일 처방(skimage SSIM, SSIM 단일 EarlyStop, patience 10, val 매 5 epoch, v5 epoch_5.pt 부터 resume). 추가로 BATCH_SIZE 8→4 강하(2026-05-11) — v5 는 BS=8 통과했으나 v6 의 baseline 측정 추가로 cudnn workspace 가 커져 첫 forward OOM 발생, 재부팅 후에도 재발하여 4 로 내림. SS2D v6 vs ETER v6 비교의 유일한 변수는 sequence model 종류(Mamba vs GRU) + DC block 유무.

## 모델 구조

```
입력: aliased image (B, 32, 320, 320) + k-space (B, 32, 320, 320)
       │                                    │
   ViT Encoder → ViT Decoder           GRU 또는 SS2D
       │                                    │
       └──── cat + RefinementBlock ─────────┘
                      │
              출력: (B, 1, 320, 320)
```

## 주요 설정 파일

- `configs/myConfig_choh_SS2D_model.py` — SS2D-ViT 설정
- `configs/myConfig_choh_ETER_model.py` — ETER-ViT 설정

## 실행

```bash
# 학습
python main_train_ss2d.py    # SS2D-ViT
python main_train_eter.py    # ETER-ViT

# 평가
python eval.py --model ss2d --ckpt logs/SS2D_ViT_R4_brain320_v2/ss2d_vit_best.pt
python eval.py --model eter --ckpt logs/ETER_ViT_R4_brain320_v2/eter_vit_best.pt

# 시각화
python visualize.py --model ss2d --ckpt logs/SS2D_ViT_R4_brain320_v2/ss2d_vit_best.pt
```

## 환경

- conda 환경: `mri_env`
- GPU: 8GB (BATCH_SIZE, GRU hidden 등 메모리 제약 있음)
- 주요 의존성: PyTorch, mamba_ssm (SS2D용 CUDA 커널), einops, wandb
