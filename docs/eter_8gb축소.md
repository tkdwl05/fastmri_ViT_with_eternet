# ETER-Net 8GB GPU 축소 과정에서 발생한 성능 저하 원인

## 배경

원본 ETER-Net은 RTX 서버(VRAM 충분)에서 384x384 이미지로 설계되었다.
이를 8GB GPU (320x320)에 맞추며 여러 설정을 축소했는데,
그 과정에서 핵심 구성 요소가 과도하게 줄어들어 성능이 크게 떨어졌다.

**결론: 원본 ETER-Net 설계 자체는 문제없다. GPU 제약에 맞추는 적응 과정에서 문제가 생겼다.**

---

## 원본 vs 8GB 축소판 (v1) 비교

| 설정 | 원본 (RTX 서버) | v1 (8GB GPU) | 변화 |
|------|----------------|-------------|------|
| 이미지 크기 | 384x384 | 320x320 | 합리적 축소 |
| Patch size | 32x32 | 32x32 | 동일 |
| 인코더 | ViT-Base (768d, 12L, head=12) | ViT-Small (384d, 6L, head=6) | 절반 축소 |
| **GRU hidden** | **10** | **2** | **5배 축소 (치명적)** |
| **최종 합성** | **GRU → U-Net 정제** | **GRU → Conv2d 1개** | **핵심 구조 제거** |
| SSIM weight | 0.2 | 0.2 | 동일 |
| Decoder OUT_CH | 16 | 256 | 변경 |
| Decoder OUT_FEAT | 32 | 16 | 변경 |

### 근거: 원본 설정 (git 초기 커밋)

```python
# configs/myConfig_choh_ViT_ETER_R4regular.py (commit 7d4e4e0)
NUM_VIT_ENCODER_HIDDEN = 768    # ViT-Base
NUM_VIT_ENCODER_LAYER = 12
NUM_ETER_HORI_HIDDEN = 10       # hidden=10
NUM_ETER_VERT_HIDDEN = 10       # hidden=10
```

```python
# choh_train_ViT_ETER_R4regular_240916py (commit 7d4e4e0)
image_size = (384, 384)
patch_size = (32, 32)
```

### 근거: 원본 모델 클래스에 U-Net 후처리가 존재

```python
# u_choh_model_ETER_ViT.py — class ETER_hybrid_GRU_DFU (line 191)
# 원본 ETER-Net은 GRU 출력을 U-Net으로 정제
self.unet = UNet_choh_skip(
    in_channels=num_feat_ch, n_classes=1,
    depth=n_unet_depth, wf=6, batch_norm=False,
    up_mode='upconv', n_hidden=n_hidden
)
```

---

## 3가지 핵심 문제

### 1. GRU hidden 10 → 2 (5배 축소)

GRU의 hidden size는 `image_size x hidden_num`에 비례하여 메모리를 소모한다:

```
원본: gru_h output = 384 x 10 x 2(bidirectional) = 7,680 차원/행
v1:   gru_h output = 320 x 2  x 2(bidirectional) = 1,280 차원/행
```

- 원본 대비 **6배 적은 표현력**
- bidirectional 출력 채널: 원본 2x10=20ch → v1 2x2=**4ch**
- k-space의 복잡한 코일간/주파수간 상관관계를 4채널로는 포착 불가

**축소 이유**: GRU hidden이 image_size에 곱해지므로 메모리가 `O(image_size^2 x hidden)`으로 급증.
8GB GPU에서 hidden=10이면 OOM 발생.

### 2. U-Net 후처리 → Conv2d 1개로 대체

원본 ETER-Net의 데이터 흐름:

```
원본: GRU출력 + aliased image → U-Net (depth=3~5, wf=6) → 재구성 이미지
v1:   GRU출력 + aliased image + ViT업샘플 → Conv2d(40, 1, 3x3) → 재구성 이미지
```

- 원본 U-Net: 수백만 파라미터, 다단계 비선형 정제
- v1 Conv2d: **파라미터 361개** (40x3x3x1 + 1), 사실상 가중 평균

**대체 이유**: ViT 디코더(6층 Transformer)가 U-Net의 역할을 할 것으로 기대.
그러나 ViT는 패치 단위로 동작하므로 픽셀 단위 정제에는 부적합했다.

### 3. 인코더 ViT-Base → ViT-Small (절반)

```
원본: dim=768, layer=12, head=12, mlp=3072  (~85M params)
v1:   dim=384, layer=6,  head=6,  mlp=1536  (~22M params)
```

이 자체만으로는 합리적인 축소이지만, 1+2와 결합되면서 전체 모델의 표현력이
임계점 아래로 떨어졌다.

---

## v2에서의 대응

| 문제 | v2 대응 |
|------|--------|
| GRU hidden=2 | **hidden=4**로 증가 (bidirectional 출력 4→8ch) |
| Conv2d 1개 | **RefinementBlock** (3xResBlock, ~120K params) |
| Patch 32x32 | **16x16**으로 축소 (패치 수 100→400, 공간 정보 보존) |
| SSIM weight 0.2 | **1.0**으로 증가 |

### v2 결과 (SS2D 모델 기준, 132 epoch)

| 지표 | v1 (200ep) | v2 (132ep) | 개선 |
|------|-----------|-----------|------|
| Val SSIM | 0.599 | **0.759** | +0.160 |
| Val PSNR | 26.45 dB | **36.38 dB** | +9.93 dB |
| Val NMSE | 0.1278 | **0.0086** | 14.8배 |
| Train-Val gap | 0.066 | **0.018** | 과적합 대폭 감소 |

---

## 남은 과제

- **GRU hidden을 더 올릴 수 있는지**: hidden=4도 원본(10) 대비 여전히 작음.
  메모리 허용 범위에서 6~8 테스트 가치 있음
- **Data Consistency layer**: 현재 구조에서는 single-channel magnitude 출력이라
  multi-coil DC 적용 불가. coil sensitivity estimation 추가 시 가능
- **원본 U-Net 후처리 복원 가능성**: RefinementBlock(3xResBlock)이 U-Net보다
  가볍지만, 현재 결과가 충분하다면 유지. 부족하면 경량 U-Net 도입 검토
