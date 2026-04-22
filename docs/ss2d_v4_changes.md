# SS2D-ViT v4: capacity 증설 + 정규화 + Data Consistency

날짜: 2026-04-22

[docs/SS2D_v1_analysis.md](SS2D_v1_analysis.md) 체크리스트에서 v2/v3가 미해결로 남긴 항목을 동시에 공격한 버전. **기존 파일은 건드리지 않고** 모든 수정은 `_v4` 접미사 신규 파일에 적용해 진행 중인 v3 학습/예약된 ETER v4에 영향을 주지 않는다.

## 1. 근거 (docs 체크리스트 대응)

| 번호 | v1 분석 문제 | v3까지 | v4에서의 처리 |
|---|---|---|---|
| 4 | SS2D 용량 부족 (`d_inner=32`, `d_state=8`) | 그대로 | **A** — `d_inner 32→64`, `d_state 8→16` |
| 5 | Train-Val 과적합 gap 0.066 (부분 해결) | 명시적 regularization 없음 | **B** — `weight_decay 1e-7→1e-5`, Transformer decoder `dropout=0.1` |
| 6 | Data Consistency layer 부재 | 미해결 | **C** — 1-iteration soft DC block 추가 (ACS 기반 sens 추정 + 학습 α) |

## 2. 새로 만든 파일

원본 파일은 그대로 유지하고 `_v4` 접미사로 복제 후 편집. ETER v4가 공유하는 [dataloader_h5.py](../dataloaders/dataloader_h5.py) 등은 절대 수정하지 않음.

| 신규 | 원본 | 주된 변경 |
|---|---|---|
| [configs/myConfig_choh_SS2D_model_v4.py](../configs/myConfig_choh_SS2D_model_v4.py) | `myConfig_choh_SS2D_model.py` | `PATH_FOLDER` = `logs/SS2D_ViT_R4_brain320_v4/`, A/B 하이퍼파라미터, DC 상수(`DROPOUT`, `DC_K_SCALE_RATIO`, `DC_INIT_ALPHA`) |
| [dataloaders/dataloader_h5_v4.py](../dataloaders/dataloader_h5_v4.py) | `dataloader_h5.py` | 반환 dict에 `mask` (1,H,W)와 `sens` (32,H,W) 추가. sens는 ACS 저주파만 iFFT 후 RSS로 정규화 |
| [models/mamba_eternet/u_choh_model_SS2D_ViT_v4.py](../models/mamba_eternet/u_choh_model_SS2D_ViT_v4.py) | `u_choh_model_SS2D_ViT.py` | Transformer `dropout=0.1`, `RefinementBlock` 출력 `1ch→2ch`, `DCBlock` 클래스 추가, `forward(..., mask, sens)` |
| [main_train_ss2d_v4.py](../main_train_ss2d_v4.py) | `main_train_ss2d.py` | v4 모듈 import, 모델에 `mask/sens` 전달, wandb run name `SS2D_v4_...` | | ** 체인 대기** → `main_train_ss2d_v4.py` 실행 |

## 3. Data Consistency block 구체

1-iteration soft DC. 파이프라인은 다음과 같다:

```
x_ri (B, 2, H, W)        — 모델이 추정한 coil-combined complex image (real, imag)
sens_ri (B, 32, H, W)    — ACS에서 추정한 sens, Σ|s|²=1
k_meas_ri (B, 32, H, W)  — 측정된 masked k-space (val_amp_X_ksp 스케일)
mask (B, 1, H, W)        — 1=sampled

1) real/imag 팩 → complex 재구성
2) k_meas_scaled = k_meas_c * K_SCALE_RATIO    ← 이미지-스케일로 보정
3) multicoil    = sens_c * x_c                 ← 추정 multicoil 이미지
4) k_pred       = fft2c(multicoil)
5) k_dc         = k_pred + mask * α * (k_meas_scaled - k_pred)   ← soft DC
6) multicoil_dc = ifft2c(k_dc)
7) x_comb       = Σ multicoil_dc * sens_c.conj()                 ← coil-combine
8) 반환         = (real, imag) → magnitude = √(r²+i²)
```

- **K_SCALE_RATIO = 100.0**: dataloader의 `val_amp_X_img / val_amp_X_ksp = 1e6/1e4`에서 유도. 고정 버퍼.
- **α (학습 파라미터)**: 초기값 1.0 → hard DC. 학습 중 조정 가능. sens 추정 품질이 나쁘면 α를 낮춰 soft하게.
- **AMP 처리**: `torch.fft`가 fp16에서 불안정할 수 있어 `DCBlock.forward` 내부 전체를 `autocast(enabled=False)` + `.float()` cast로 감쌈.

## 4. 학습 체인 예약

`run_chain_ss2d_v4.sh`가 **PID 50369**(기존 체인: SS2D v3 → ETER v4)의 종료를 60초 간격으로 폴링한다. 종료되는 즉시 `main_train_ss2d_v4.py`를 `run_ss2d_v4.log`로 리다이렉트해 실행. 실행 중인 스크립트 PID는 `ps -ef | grep run_chain_ss2d_v4` 로 확인.

로그:
- 단계 start/end: `run_chain_v4.log`
- SS2D v4 학습 stdout/stderr: `run_ss2d_v4.log`

## 5. 리스크와 확인 포인트

| 리스크 | 확인 방법 |
|---|---|
| FFT 스케일 불일치로 NaN/diverge | `run_ss2d_v4.log`의 첫 5~10 epoch 내 loss 값 확인. fp16 → fp32 전환이 제대로 들어갔는지 |
| ACS-only sens map 품질 부족 | val SSIM이 v3 대비 하락하면 sens 품질이 원인일 가능성. 학습형 sens 모듈(E2E-VarNet 스타일)은 v5 후보 |
| 복소 출력이 학습 수렴 저해 | train loss가 epoch 5까지 감소하는지 모니터. 감소 안 하면 α 초기값 0.1로 낮춰 soft start |
| DC 추가로 속도 저하 | 약 20~30% forward 시간 증가 예상. 200 epoch 완주 가능성 있음 |

## 6. 공정 비교

| 비교 대상 | 의미 |
|---|---|
| v3 vs v4 | A+B+C 종합 효과 측정. 단, 세 변수가 동시에 바뀌어 **어느 요인이 주효한지는 분리 불가** |
| v2 vs v3 | 스케줄러만 다름 (WarmRestarts vs CosineAnnealingLR) — 기록은 [scheduler_change.md](scheduler_change.md) |

추후 A/B/C를 분리해 원인 기여도를 보고 싶다면 v5에서 하나씩 ablation 진행.

## 7. 결정하지 않은 것

- **학습형 sensitivity map estimation** (E2E-VarNet 스타일 CNN)
- **DC iteration 수 증가** (cascade 2~4회) — 현재 1-iter로 고정
- **data augmentation** (flip/rotation) — 과적합이 여전히 심하면 v5에서 추가
