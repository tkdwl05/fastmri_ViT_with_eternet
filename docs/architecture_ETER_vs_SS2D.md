# ETER-ViT vs SS2D-ViT 아키텍처 비교

fastMRI brain AXFLAIR multicoil MRI 재구성 모델로, 두 가지 변형(ETER vs SS2D)을 동일 조건에서 비교 실험하는 구조이다.

---

## 1. 공통 파이프라인

```
언더샘플링 k-space (B,32,320,320) ──┐
                                    ├──→ [최종 합성 Conv] → 재구성 이미지 (B,1,320,320)
앨리어싱 이미지 (B,32,320,320) ─────┤
   │                                │
   └→ [ViT 인코더] → [ViT 디코더] → [업샘플] ──┘
```

입력은 2가지이다:
- **`in_imgs`** (B, 32, H, W): 16코일 x 2(실수/허수) = 32채널 앨리어싱 이미지
- **`in_ksp`** (B, 32, H, W): 언더샘플링 k-space 데이터

---

## 2. 공통 인코더: `choh_ViT`

> 파일: `models/hybrid_eternet/u_choh_model_ETER_ViT.py` (class choh_ViT, line 652)

ViT-Small 구조이다:
- 320x320 이미지를 **32x32 패치**로 분할 → **10x10 = 100 패치**
- 각 패치를 `(32x32x32 = 32768)` 차원에서 `dim=384`로 Linear 투영
- **6층 Transformer** (heads=6, mlp=1536)로 인코딩
- 두 모델 모두 이 인코더를 **동일하게** 사용

### 인코더 설정값

| 파라미터 | 값 |
|---------|-----|
| IMAGE_SIZE | (320, 320) |
| PATCH_SIZE | (32, 32) |
| INPUT_CHANNELS | 32 |
| NUM_VIT_ENCODER_HIDDEN | 384 |
| NUM_VIT_ENCODER_LAYER | 6 |
| NUM_VIT_ENCODER_MLP_SIZE | 1536 |
| NUM_VIT_ENCODER_HEAD | 6 |

---

## 3. ETER 디코더: Bidirectional GRU 방식

> 파일: `models/hybrid_eternet/u_choh_model_ETER_ViT.py` (class choh_Decoder3_ETER_skip_up_tail, line 14)
> 학습 스크립트: `main_train_eter.py`
> 설정 파일: `configs/myConfig_choh_ETER_model.py`

원본 ETER-Net 방식으로, **Bidirectional GRU**를 사용한다.

### k-space 처리 흐름

```
in_ksp (B, 32, 320, 320)
  │
  ├─ rearrange → (B, 320, 320x32)     # 각 행을 시퀀스로
  │
  ├─ gru_h: 수평 양방향 GRU (좌→우 + 우→좌)
  │    input_size  = 320 x 16 x 2 = 10240
  │    hidden_size = 320 x 2 = 640
  │    출력: (B, 320, 320x2x2) → rearrange
  │
  ├─ gru_v: 수직 양방향 GRU (위→아래 + 아래→위)
  │    input_size  = 320 x 2 x 2 = 1280
  │    hidden_size = 320 x 2 = 640
  │    출력: (B, 2x2, 320, 320)
  │
  └─ out_v: (B, 4, H, W)   # 2 * eter_n_vert_hidden = 4
```

- **gru_h**: k-space의 각 행을 시퀀스로 → 수평 방향 양방향 GRU
- **gru_v**: gru_h 출력을 재배열 후 각 열을 시퀀스로 → 수직 방향 양방향 GRU
- `NUM_ETER_HORI_HIDDEN=2`, `NUM_ETER_VERT_HIDDEN=2` (메모리 제약 때문에 작게 설정)
- GRU hidden이 `image_size x hidden_size`에 비례하여 메모리 소모가 큼

### 최종 합성

```
cat([ViT 업샘플 출력(4ch), 앨리어싱 이미지(32ch), GRU 출력(4ch)])  # 총 40ch
  → Conv2d(40, 1, kernel_size=3)
  → 재구성 이미지 (B, 1, H, W)
```

---

## 4. SS2D 디코더: VMamba 스타일 SSM 방식

> 모델 파일: `models/mamba_eternet/u_choh_model_SS2D_ViT.py` (class choh_Decoder_SS2D_ViT, line 102)
> SS2D 모듈: `models/mamba_eternet/ss2d.py` (class SS2D, line 111)
> 학습 스크립트: `main_train_ss2d.py`
> 설정 파일: `configs/myConfig_choh_SS2D_model.py`

ETER의 GRU를 **VMamba 스타일 SS2D(Selective State Space)**로 대체한 것이다.

### SS2D 모듈 내부 구조

```
in_ksp (B, 32, H, W)
  │
  ├─ rearrange → (B, H, W, 32)
  ├─ LayerNorm → Linear(32 → d_inner=32) → SiLU
  ├─ rearrange → DepthwiseConv2d(3x3) → SiLU     # 지역 문맥 혼합
  │
  ├─ 4방향 SSM 스캔 (독립 병렬 처리):
  │    1. ssm_h_fwd: 수평 좌→우  (각 행을 길이 W의 시퀀스로)
  │    2. ssm_h_bwd: 수평 우→좌  (flip + SSM + flip)
  │    3. ssm_v_fwd: 수직 위→아래 (각 열을 길이 H의 시퀀스로)
  │    4. ssm_v_bwd: 수직 아래→위 (flip + SSM + flip)
  │
  ├─ cat 4방향 → (B, H, W, 4x32=128)
  ├─ LayerNorm → Linear(128 → 32)
  ├─ rearrange → Conv1x1(32 → 20)
  │
  └─ out: (B, 20, H, W)
```

### SelectiveScan1D (SSM 핵심)

> 파일: `models/mamba_eternet/ss2d.py` (class SelectiveScan1D, line 30)

Mamba의 CUDA 커널(`selective_scan_fn`)을 사용하는 1D SSM이다:

- **점화식**:
  - `x_t = dA_t * x_{t-1} + dB_t * u_t`
  - `y_t = C_t * x_t + D * u_t`
- 입력 x로부터 `dt`, `B`, `C`를 **선택적으로(selective)** 생성 → 입력에 따라 상태 전이가 달라짐
- `d_inner=32` (SSM 내부 차원), `d_state=8` (SSM 상태 차원 N)
- `dt_rank = max(1, d_inner // 8) = 4`

### SS2D 설정값

| 파라미터 | 값 |
|---------|-----|
| NUM_SS2D_D_INNER | 32 |
| NUM_SS2D_D_STATE | 8 |
| NUM_SS2D_OUT_CH | 20 |

### 최종 합성

```
cat([ViT 업샘플 출력(4ch), 앨리어싱 이미지(32ch), SS2D 출력(20ch)])  # 총 56ch
  → Conv2d(56, 1, kernel_size=3)
  → 재구성 이미지 (B, 1, H, W)
```

---

## 5. ETER vs SS2D 핵심 차이

| 항목 | ETER (GRU) | SS2D (Mamba) |
|------|-----------|-------------|
| 시퀀스 모델링 | Bidirectional GRU (2단계) | 4방향 SSM (1단계, 병렬) |
| 스캔 방향 | 수평→수직 (직렬 2단계) | 수평x2 + 수직x2 (독립 4방향) |
| 복잡도 | O(L^2) (GRU hidden이 image_size에 비례) | O(L) (SSM 선형 시간) |
| 메모리 | hidden=2로 제한 (320x2=640) | d_inner=32, d_state=8 (이미지 크기 무관) |
| CUDA 커널 | 없음 (PyTorch GRU) | mamba_ssm CUDA 최적화 |
| 출력 채널 | 2 x vert_hidden = 4 | ss2d_out_ch = 20 |
| 최종 합성 입력 채널 | 40 (4+32+4) | 56 (4+32+20) |

---

## 6. 공통 ViT 디코더 + 업샘플링

인코더 출력 이후의 디코더/업샘플 구조는 두 모델 모두 동일하다:

```
encoded_tokens (B, 100, 384)
  │
  ├─ enc_to_dec: Linear(384 → 512)
  ├─ + decoder_pos_emb
  ├─ Transformer 디코더 (6층, heads=8, dim_head=64, mlp=2048)
  │
  ├─ final_linear: Linear(512 → 256x16x16 = 65536)
  ├─ rearrange → (B, 256, 160, 160)    # 10x16=160
  ├─ Upsample (PixelShuffle x2) → (B, 256, 320, 320)
  │    ※ log2(32/16) = 1 → 업샘플 1회
  │
  └─ 출력: (B, 4, 320, 320)    # decoder_out_ch_up_tail = 256 → shuffle 후 4ch 아닌가?
```

> 실제로는 `decoder_out_ch_up_tail=256`이 아니라 설정상 기본값에 따라 결정됨.
> 설정: `NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH=256`, `NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT=16`

### 디코더 설정값 (공통)

| 파라미터 | 값 |
|---------|-----|
| NUM_VIT_DECODER_DIM | 512 |
| NUM_VIT_DECODER_DEPTH | 6 |
| NUM_VIT_DECODER_HEAD | 8 |
| NUM_VIT_DECODER_DIM_HEAD | 64 |
| NUM_VIT_DECODER_DIM_MLP_HIDDEN | 2048 |
| NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH | 256 |
| NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT | 16 |

---

## 7. 학습 설정 (두 모델 공통)

| 항목 | 값 |
|------|-----|
| Loss | L1 + 0.2 x (1 - SSIM) |
| Loss 계산 정밀도 | fp32 (SSIM 내부 제곱이 fp16에서 overflow 방지) |
| Optimizer | Adam (lr=2e-4, weight_decay=1e-7) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=1epoch, T_mult=2, eta_min=1e-6) |
| AMP | GradScaler + autocast (fp16 forward, fp32 loss) |
| Batch Size | 8 |
| Epochs | 200 |
| Data | fastMRI brain multicoil 전체 train/val |
| Checkpoint | 5 epoch마다 정기 저장 |
| Best Checkpoint | SSIM/PSNR/NMSE/L1 4개 지표의 composite score로 판단 |
| Validation 전략 | train SSIM이 best를 갱신할 때만 val 실행 |
| Logging | wandb (프로젝트: ViT-MRI-Recon) |

### Composite Score 계산

```
composite = (
    val_ssim / best_ssim +      # 높을수록 좋음
    best_nmse / val_nmse +      # 낮을수록 좋음
    val_psnr / best_psnr +      # 높을수록 좋음
    best_l1 / val_l1            # 낮을수록 좋음
) / 4.0
```

4개 지표를 best 대비 비율로 정규화하여, 모두 1.0 이상이면 전체적으로 개선된 것으로 판단한다.
