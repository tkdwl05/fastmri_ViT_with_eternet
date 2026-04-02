# ViT-ETER_net: Vision Transformer + ETER-Net for MRI Reconstruction

Vision Transformer 기반 인코더와 ETER-Net(Bidirectional GRU) 디코더를 결합한 FastMRI 뇌 영상 재구성 프레임워크.

---

## 개요

언더샘플링된 k-space 데이터(가속률 R=4)로부터 고품질 MRI 영상을 재구성하는 하이브리드 딥러닝 모델.

- **인코더**: ViT-Base (768-dim, 12 layers, 12 heads) — 고수준 특징 학습
- **디코더**: 커스텀 트랜스포머 (1280-dim, 8 heads, 12 layers) + ETER-Net GRU — 아티팩트 보정 및 영상 재구성
- **손실 함수**: L1 픽셀 손실 + SSIM 지각 손실 (λ=0.2)
- **데이터**: FastMRI 멀티코일 뇌 데이터셋, 16채널, 384×384

---

## 디렉토리 구조

```
ViT_based_MRIrecon/
├── main_train.py                          # 메인 학습 진입점 (통합 버전)
├── download_repos.py                      # 외부 저장소 자동 다운로드
├── configs/                               # 실험별 설정 파일
│   ├── myConfig_choh_model3.py            # 주 설정 (ViT-Base + ETER-ViT)
│   ├── myConfig_choh_ViT_ETER_R4regular.py
│   ├── myConfig_choh_ViT_ETER_R4regular_v2.py
│   ├── myConfig_choh_ViT_recon_R4regular.py
│   └── myConfig_choh_ViT_autoencoder_R4regular.py
├── dataloaders/                           # 데이터 로딩 및 전처리
│   ├── dataloader_h5.py                   # 메인 H5 데이터로더 (R=4 언더샘플링)
│   ├── myDataloader_fastmri_brain_R4_251012.py
│   ├── myDataloader_fastmri_brain_R8_251012.py
│   ├── myDataloader_fastmri_brain_randomR8_251007.py
│   ├── list_brain_train_320.txt           # 학습 파일 목록
│   ├── list_brain_unseen_*.txt            # 검증/테스트 분할 목록
│   ├── R4_idx_part1.npy                   # R=4 언더샘플링 인덱스
│   └── R4_idx_part2.npy
├── models/
│   ├── hybrid_eternet/                    # 메인 모델 구현
│   │   ├── u_choh_model_ETER_ViT.py       # 핵심 모델 정의 (인코더 + 디코더)
│   │   ├── u_choh_SSIM.py                 # SSIM / MS-SSIM 손실
│   │   └── hybrid_eternet_fastmri-main/   # 외부 ETER-Net 저장소
│   ├── vit_pytorch/
│   │   ├── u_choh_vit.py                  # 커스텀 ViT 래퍼
│   │   └── vit-pytorch-main/              # 외부 vit-pytorch 저장소
│   └── mae/
│       └── mae-main/                      # 외부 MAE 저장소
├── fastMRI_data/                          # FastMRI 데이터셋 (직접 준비 필요)
│   ├── multicoil_train/
│   ├── multicoil_val/
│   └── multicoil_test/
├── logs/                                  # 학습 로그 및 체크포인트
└── scripts_legacy/                        # 실험 이력 스크립트 (30+ 버전)
```

---

## 모델 아키텍처

### 전체 데이터 흐름

```
입력: 언더샘플링 k-space (R=4, 32채널, 384×384)
 + 앨리어싱 영상 (32채널, 384×384)
          │
          ▼
  ┌───────────────────────────────────────┐
  │         choh_ViT (인코더)             │
  │   32×32 패치 → 144 토큰              │
  │   12 레이어, 768-dim, 12 헤드        │
  └───────────────────────────────────────┘
          │
          ▼ 토큰 (dim=768)
  ┌───────────────────────────────────────┐
  │  choh_Decoder3_ETER_skip_up_tail      │
  │  ┌─────────────────────────────────┐  │
  │  │ ViT 디코더                      │  │
  │  │ 768→1280 프로젝션               │  │
  │  │ 12 레이어, 8 헤드               │  │
  │  │ → 256×16×16 → 업샘플 → 256ch   │  │
  │  └─────────────────────────────────┘  │
  │  ┌─────────────────────────────────┐  │
  │  │ ETER-Net (Bidirectional GRU)    │  │
  │  │ 수평 GRU (hidden=10)            │  │
  │  │ 수직 GRU (hidden=10)            │  │
  │  │ → 20ch 아티팩트 보정 특징       │  │
  │  └─────────────────────────────────┘  │
  └───────────────────────────────────────┘
          │
          ▼ Concat [256 + 32 + 20 = 308채널]
  Conv2d (3×3, 308→1)
          │
          ▼
  재구성 영상 (384×384, 1채널)
```

### 데이터 전처리 흐름 (dataloader_h5.py)

```
FastMRI .h5 파일 (멀티코일 뇌)
          │
          ▼
복소수 k-space 로드 (슬라이스, 16코일, H, W)
          │
          ▼
중앙 크롭 → 384×384
실수/허수 분리 → 32채널 (16코일 × 2)
          │
     ┌────┼────────────────┐
     ▼    ▼                ▼
풀 k-space  언더샘플링 k-space   언더샘플링 k-space
  IFFT2↓    (3::4 라인 +      IFFT2↓
Ground truth  중앙 ±16 라인)   앨리어싱 영상
(label)      (data)           (data_img)
```

---

## 핵심 클래스

| 클래스 | 파일 | 역할 |
|--------|------|------|
| `FastMRI_H5_Dataloader` | `dataloaders/dataloader_h5.py` | FastMRI H5 데이터셋 로더 |
| `choh_ViT` | `models/hybrid_eternet/u_choh_model_ETER_ViT.py` | ViT-Base 인코더 |
| `choh_Decoder3_ETER_skip_up_tail` | `models/hybrid_eternet/u_choh_model_ETER_ViT.py` | 메인 디코더 (ViT + ETER-Net) |
| `Transformer` | `models/hybrid_eternet/u_choh_model_ETER_ViT.py` | 트랜스포머 블록 |
| `Attention` | `models/hybrid_eternet/u_choh_model_ETER_ViT.py` | 멀티헤드 셀프 어텐션 |
| `Upsample` | `models/hybrid_eternet/u_choh_model_ETER_ViT.py` | 2x 업샘플링 (Conv2d + PixelShuffle) |
| `SSIM` | `models/hybrid_eternet/u_choh_SSIM.py` | SSIM 손실 |
| `MSSSIM` | `models/hybrid_eternet/u_choh_SSIM.py` | 멀티스케일 SSIM 손실 |

---

## 주요 하이퍼파라미터 (myConfig_choh_model3.py)

```python
# 인코더 (ViT-Base)
NUM_VIT_ENCODER_HIDDEN       = 768
NUM_VIT_ENCODER_LAYER        = 12
NUM_VIT_ENCODER_MLP_SIZE     = 3072
NUM_VIT_ENCODER_HEAD         = 12

# ETER-Net (Bidirectional GRU)
NUM_ETER_HORI_HIDDEN         = 10
NUM_ETER_VERT_HIDDEN         = 10

# 디코더
NUM_VIT_DECODER_DIM          = 1280
NUM_VIT_DECODER_DEPTH        = 12
NUM_VIT_DECODER_HEAD         = 8
NUM_VIT_DECODER_DIM_MLP_HIDDEN        = 5120
NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH   = 256
NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT = 16

# 학습
BATCH_SIZE                   = 2
NUM_EPOCHS                   = 200
LEARNING_RATE_ADAM           = 1e-4
LAMBDA_SSIM_PER_PIXEL        = 0.2
LAMBDA_REGULAR_PER_PIXEL     = 1e-7
```

---

## 시작하기

### 요구 사항

```bash
pip install torch h5py numpy einops
```

### 데이터 준비

FastMRI 멀티코일 뇌 데이터셋을 다음 경로에 배치:

```
fastMRI_data/
├── multicoil_train/   # 학습용 .h5 파일
├── multicoil_val/     # 검증용 .h5 파일
└── multicoil_test/    # 테스트용 .h5 파일
```

### 외부 저장소 다운로드

```bash
python download_repos.py
```

### 학습 실행

```bash
python main_train.py
```

학습 로그 및 체크포인트는 `logs/` 폴더에 저장됩니다.

---

## 출력 구조

```
logs/
└── {실험명}/
    ├── log.txt                            # 에폭별 손실 로그
    ├── choh_vit_eternet_epoch_5.pt
    ├── choh_vit_eternet_epoch_10.pt
    └── ...                                # 5 에폭마다 체크포인트 저장
```

---

## 손실 함수

```
Total Loss = L1(pred, target) + 0.2 × (1 - SSIM(pred, target))
```

옵티마이저: Adam (lr=1e-4)  
스케줄러: CosineAnnealingWarmRestarts (T_0=40, T_mult=2)

---

## 개발 이력

`scripts_legacy/` 폴더에 30개 이상의 실험 스크립트가 보존되어 있으며, 아래 흐름으로 발전:

1. ETER-Net 기반 FastMRI 뇌 재구성 (초기)
2. 랜덤 언더샘플링 패턴 도입
3. ViT 기반 인코더 통합 (ViT-Base, Autoencoder 변형)
4. ViT + ETER-Net 하이브리드 통합 (스킵 연결, 업샘플링 테일)
5. 학습률 튜닝, 어닐링 전략, 어블레이션 스터디
6. 최종 통합 버전 (`main_train.py`, exp3_18)
