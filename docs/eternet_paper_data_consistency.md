# 교수님 ETER-net과 Data Consistency (DC) — 논문·코드 분석

작성 2026-06-17. 질문: "DC는 MRI 재구성 표준 기법인데, 교수님 ETER-net은 DC를 어떻게 적용하나?" → **명시적 DC를 쓰지 않는다.**

## 1. 결론 (TL;DR)
- 교수님 ETER-net = **bi-RNN ×2 + CNN(U-Net DFU)**, **명시적 DC 블록 없음**. k-space→image를 RNN이 직접 학습(end-to-end).
- 코드(원본 repo)·논문(2018·2021) 양쪽에서 DC/data-fidelity 부재 확인.
- 설계 의도: RNN이 FFT를 대체 → **non-Cartesian 포함 임의 trajectory** 지원. 표준 DC(FFT+Cartesian+mask 전제)와 양립 불가.
- 우리 프로젝트의 DC = v4에서 **SS2D arm에만** 추가된 DC-CNN/VarNet 계열 증강. 교수님 원본엔 없음 → v7_titan 비교의 confound.

## 2. 교수님 ETER-net 아키텍처 (논문)
- 출처: Oh et al., *ETER-net: End To End MR Image Reconstruction Using Recurrent Neural Network*, MLMIR(MICCAI workshop) 2018; 확장판 *A k-space-to-image reconstruction network for MRI using recurrent neural network*, Medical Physics 2021; 특허 US11348291.
- 구성:
  - **bi-RNN ×2 (gru_h 수평 + gru_v 수직)**: k-space 양방향 스캔으로 **도메인 변환**(k-space→image). FFT의 학습형 대체.
  - **CNN/U-Net (DFU=Dual-Frame U-net)**: 이미지 도메인 de-aliasing.
  - 입출력: undersampled k-space → reconstructed image, "direct mapping".
- 강점: trajectory 무관(Cartesian+non-Cartesian), 파라미터 효율, 고해상도 확장.
- 성능(참고, 2021): FastMRI R=4 nMSE 1.05% / SSIM 0.931; R=8 3.12% / 0.884; in-house R=4 1.09% / 0.938.

## 3. DC가 ETER-net에 적용되나? → 아니오 (근거)
### 3.1 논문 근거
2021 Medical Physics 전문 확인: 구성요소는 bi-RNN + CNN뿐, **"data consistency"/"data fidelity" 언급 0**. "direct mapping of input k-space data and reconstructed images" — physics 제약 없이 망 자체로 재구성.
### 3.2 코드 근거 (`models/hybrid_eternet/hybrid_eternet_fastmri-main/`)
- `model.py::ETER_hybrid_GRU_DFU`: gru_h → gru_v → cat(aliased image) → UNet_choh_skip. **k-space 재투영/FFT 없음**.
- 전수 grep: `DCBlock`/`k_dc`/soft-dc/data-consistency **0건**. FFT(`np.fft.ifft2`)는 **dataloader**(aliased image 생성)에만.
### 3.3 왜 안 쓰나 (설계상 필연)
- thesis = "RNN이 도메인 변환 학습 → 임의 trajectory(non-Cartesian)".
- 표준 DC = 측정 k-space를 sampled 위치에 재투영, **FFT+Cartesian 격자+sampling mask(+sens)** 전제. non-Cartesian에선 균일 FFT 불성립 → DC layer를 끼울 수 없음.
- ETER-net의 일관성은 **암묵적·학습형**: 입력 k-space를 RNN이 직접 처리 + aliased image를 U-Net 입력 + L1/SSIM loss가 측정 충실도를 간접 유도.

## 4. MRI 재구성 일반에서의 DC (배경)
- **DC-CNN (Schlemper 2017)**: CNN 블록 ↔ **DC layer** 교대 unroll. DC: sampled 위치에서 `k=(k_cnn+λ·k_meas)/(1+λ)`.
- **VarNet / E2E-VarNet (Hammernik 2018 / Sriram 2020)**: variational unrolling, cascade마다 DC + (E2E) 학습형 sensitivity map.
- 공통 전제: Cartesian k-space, sampling mask, multicoil시 sens map, FFT. 효과: 측정 주파수 fidelity 보장 → 보통 PSNR/NMSE↑, 단 Cartesian 가정에 묶임.

## 5. 우리 프로젝트의 DC (대조)
- 도입: v4(2026-04), **SS2D arm에만**. `models/mamba_eternet/u_choh_model_SS2D_ViT_v4.py::DCBlock`.
- 형태: 1-iter **soft DC**, ACS 기반 sens, learnable α: `k_dc = k_pred + mask·α·(k_meas_scaled − k_pred)` → coil-combine.
- ETER(GRU) arm엔 없음 → **v7_titan dead-heat = "Mamba+DC vs GRU"** confound.
- 교수님 ETER-net 관점: DC는 **외부 증강**(DC-CNN/VarNet 계열)이지 ETER-net 구성요소 아님.

## 6. v8 함의 (2×2 ablation)
- **{GRU,SS2D} × {no DC, DC} = 4런**.
- **no-DC 2종** = 교수님 ETER-net 충실(원본 + RNN→SS2D 치환), FFT-free·non-Cartesian thesis 보존.
- **DC 2종** = DC-CNN/VarNet식 증강(Cartesian 가정 도입, 성능 상한 탐색).
- 분리되는 질문: ① sequence model 효과(RNN vs SS2D) ② DC 효과(no-DC vs DC) ③ v7_titan 동률이 DC 목발 덕인가(= no-DC SS2D vs no-DC GRU).
- 주의: DC를 켜려면 U-Net 출력 head를 1ch(magnitude)→2ch(complex)로 바꿔야 함 → "DC 효과"에 head 변경이 불가피하게 약간 결합(표준).

## 7. 출처
- ETER-net 2018 (MLMIR): https://link.springer.com/chapter/10.1007/978-3-030-00129-2_2
- Medical Physics 2021: https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.14566
- 특허 US11348291
- DC-CNN: Schlemper et al. 2017; (E2E-)VarNet: Hammernik 2018 / Sriram 2020
- 원본 코드: `models/hybrid_eternet/hybrid_eternet_fastmri-main/model.py` / 대조 `models/mamba_eternet/u_choh_model_SS2D_ViT_v4.py`
