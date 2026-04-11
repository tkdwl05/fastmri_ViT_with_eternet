# 프로젝트 세부 기술 요약 (TECHNICAL)

최종 정리일: 2026-04-11 (2026-04-08 Codex 초안 → 2026-04-11 Claude Code 갱신)  
대상: `/home/snorlax-dw/바탕화면/ViT_based_MRIrecon`

## 0) 내가 정리한 목적

이 프로젝트는 `fastMRI brain multicoil` 데이터로부터 언더샘플링된 k-space을 입력받아 MRI 영상을 재구성하는 파이프라인이다.  
핵심은 ViT 인코더에 기반한 디코더(ETER 또는 SS2D) 구조를 학습/검증/시각화하는 것이다.

동작 엔트리포인트는 다음 3개이다.
- `main_train_eter.py`: ETER(양방향 GRU) 전용 학습
- `main_train_ss2d.py`: SS2D 전용 학습
- `main_train.py`: 과거/통합형(작동은 하나 현재 최신 실험 기준에서는 정밀 비교군이 아님)

## 1) 데이터 계층(입·출력 스펙과 전처리)

### 파일: `dataloaders/dataloader_h5.py`

- 클래스: `FastMRI_H5_Dataloader`
- 데이터셋 타입: `torch.utils.data.Dataset`

### 출력 딕셔너리 키
- `data`: `float32`, shape `(2*C, H, W)` (기본 `2*16=32, H=W=320`)  
  의미: 언더샘플링된 k-space의 real/imag 교차 배치
- `data_img`: `float32`, shape `(2*C, H, W)`  
  의미: 언더샘플링 후 역변환한 앨리어싱 영상(동일 채널 수)
- `label`: `float32`, shape `(1, H, W)`  
  의미: `reconstruction_rss` 기반 GT magnitude (FastMRI 제공)

### 전처리 정합성(코드 주석 기준 fastMRI 공식 정리와 동일)
- `ifft2c`는 `ifftshift -> ifft2 -> fftshift` 순서로 수행
- k-space를 먼저 iFFT하고, 이미지 도메인에서 center crop(`H=W=320`) 수행
- crop된 이미지에 `fft2c`를 다시 적용하여 k-space를 재구성
- R=4 마스크를 width축(phase-encoding)에서 고정 equispaced 패턴으로 적용
  - `center_fraction=0.08`
  - offset은 `3`에서 시작해 `3,7,11,...`로 열 샘플링
- masked k-space를 다시 iFFT하여 `data_img` 생성
- `reconstruction_rss`가 있는 파일만 우선 사용하며(평가 스크립트가 요구), 없는 경우에는 fallback로 RSS 계산
- 출력 스케일링
  - k-space 입력: `val_amp_X_ksp = 1e4`
  - 앨리어싱 이미지: `val_amp_X_img = 1e6`
  - GT: `val_amp_Y = 1e6`

### 파일/샘플 처리 (320×320 필터링)
- 생성자에서 파일 전체를 스캔 후 shape 검사로 **이기종 파일 필터링**:
  - `reconstruction_rss` 존재 필수
  - `rss_shape[-2:] == (N_OUTPUT, N_OUTPUT)` 및 `kspace H,W >= N_OUTPUT`
  - fastMRI brain은 파일마다 rss shape이 다름 (320×320, 320×264, 768×396 등)
  - 필터 결과: train 541/901 파일 (8548 슬라이스), val 285/460 파일 (4492 슬라이스)
- 실제 사용 파일 수는 `num_files`로 절단 가능(디버깅/짧은 실험용)
- `__getitem__`은 `kspace`와 `reconstruction_rss`를 매 샘플 읽어 tensor dict를 반환

## 2) 모델 계층 상세

### 공통 인코더: `models/hybrid_eternet/u_choh_model_ETER_ViT.py`의 `choh_ViT`

- 패치 임베딩: `patch_size=(32,32)` 기준 입력 영상을 patch token으로 변환
- 토큰 구조
  - `patch_embedding` = `LayerNorm -> Linear -> LayerNorm`
  - CLS token + positional embedding 사용
  - `pool='mean'` 또는 `pool='cls'`
- 트랜스포머 블록
  - 클래스: `Transformer`
  - 내부: `Attention` + `FeedForward`를 레이어 반복
  - gradient checkpoint 사용(`checkpoint.checkpoint`)으로 메모리 절감
- 출력은 `out = mlp_head(dim, num_classes)`이지만, 디코더 쪽에서는 feature token만 사용(기본 클래스 재사용)

### ETER 디코더: `choh_Decoder3_ETER_skip_up_tail`

- 입력: `in_imgs (B,32,H,W)`, `in_ksp (B,32,H,W)`
- ViT 디코더 경로
  1. `in_imgs` patch embedding 및 인코더 transformer 적용
  2. `enc_to_dec` 선형 투영
  3. decoder transformer
  4. `final_linear`로 patch latent 확장
  5. `up_tail`(연속 `PixelShuffle` 2x 업샘플)로 채널 수/해상도 복원
- ETER (양방향 GRU) 경로
  - 수평 GRU(`gru_h`)
  - 수직 GRU(`gru_v`)
  - GRU 출력 채널은 `2 * NUM_ETER_VERT_HIDDEN`로 늘어남
- 최종 합성
  - 채널 concat: `x(업샘플 결과) + in_imgs + out_v`
  - `Conv2d(in_channels=decoder_out_ch_up_tail + 32 + 2*eter_n_vert_hidden, out=1, k=3, p=1)`

### SS2D 디코더: `models/mamba_eternet/u_choh_model_SS2D_ViT.py`

- 구조는 ETER 디코더의 ViT 트랜스포머/업샘플 파트는 유지
- `in_ksp`에 대해 GRU 대신 SS2D 모듈(`self.ss2d`) 적용
- SS2D 기본 동작
  - 입력 정규화 + 1x1 입력 투영으로 `d_inner` 생성
  - depthwise conv로 지역 문맥 보강
  - 4방향 선택 스캔(`SS2D`):
    - 가로 정방향/역방향
    - 세로 정방향/역방향
  - `SelectiveScan1D`에서 `mamba_ssm` CUDA 커널 사용
    - 출력 4방향 concat -> `LayerNorm` -> linear merge -> 2D projection
- 최종 concat 채널
  - `x(업샘플) + in_imgs + out_ss2d`
  - 출력 채널은 `ss2d_out_ch`(기본 20)
- 의존성
  - SS2D는 `mamba_ssm` 패키지/커널이 있어야만 import/실행

### SSIM 지표 모듈: `models/hybrid_eternet/u_choh_SSIM.py`

- `SSIM` 클래스: 채널별 SSIM 계산 기반의 손실/평가 지표
- `MSSSIM` 클래스: 멀티스케일 SSIM 보조
- 학습에서 손실로 `1 - SSIM` 사용

## 3) 학습 파이프라인

### ETER 학습: `main_train_eter.py` / SS2D 학습: `main_train_ss2d.py`

두 스크립트는 디코더만 다르고 나머지 구조는 동일함 (ETER: `choh_Decoder3_ETER_skip_up_tail`, SS2D: `choh_Decoder_SS2D_ViT`).

1. CUDA 체크 후 `device = torch.device("cuda")`
2. `check_env_for_model(...)` 실행 — 환경/패키지/CUDA/데이터셋/config 점검
3. Encoder/Decoder 인스턴스 생성
   - Encoder: `choh_ViT` (양쪽 동일)
   - Decoder: ETER는 Bi-GRU, SS2D는 VMamba 기반
4. Optimizer/Scheduler
   - Adam(`lr=2e-4`, `weight_decay=1e-7`)
   - `CosineAnnealingWarmRestarts(T_0=steps_per_epoch, T_mult=2, eta_min=1e-6)`
   - **scheduler는 dataloader 생성 후 초기화** (T_0이 정확한 배치 수를 반영하도록)
5. DataLoader (**RAM OOM 방지를 위해 worker 수 제한**)
   - Train: `batch_size=8, num_workers=4, persistent_workers=True, prefetch_factor=2`
   - Val: `batch_size=4, num_workers=2`
6. **wandb 로깅 통합** (프로젝트: `ViT-MRI-Recon`)
   - `wandb.init()` — 모델/하이퍼파라미터/데이터셋 크기 등 config 기록
   - `wandb.watch(model, log='gradients', log_freq=100)` — gradient histogram
7. 학습 루프
   - **forward는 `autocast('cuda')` 안에서 fp16**, **loss는 autocast 밖에서 fp32로 계산**
     - 이유: SSIM 내부 제곱 연산이 fp16에서 overflow (label 값 ~959, 959²≈920k > fp16 max 65504)
   - `loss = L1(out_fp, label) + 0.2*(1-SSIM(out_fp, label))`
   - GradScaler + mixed precision 역전파
   - step 후 scheduler 갱신
   - **매 batch마다 wandb 로깅**: `train/loss`, `train/loss_l1`, `train/loss_ssim`, `train/ssim`, `train/psnr`, `train/nmse`, `train/lr`
   - **매 epoch 평균 wandb 로깅**: `epoch/train_loss`, `epoch/train_ssim`, `epoch/train_psnr`, `epoch/train_nmse`, `epoch/train_l1`
8. 검증 트리거
   - `avg_train_ssim > best_train_ssim`일 때만 `run_val` 실행 (**tqdm 진행률 bar 있음**)
   - val 결과 **wandb 로깅**: `val/ssim`, `val/psnr`, `val/nmse`, `val/l1`
   - composite score 계산: 현재/best 비율 기반 (SSIM·PSNR은 현재/best, NMSE·L1은 best/현재)
   - composite 최고 갱신 시 `*_best.pt` 저장
   - **주의**: composite는 비율 기반이라 초반 큰 점프 이후 갱신이 잘 안 됨 → 마지막 epoch ckpt가 실제 최선일 수 있음
9. 체크포인트
   - 5 epoch마다 `*_epoch_{n}.pt`
   - log는 `PATH_FOLDER/log.txt`
   - 학습 종료 시 `wandb.finish()`

### 통합/레거시 학습: `main_train.py`

- 매우 단순화된 예전 버전
- `myConfig_choh_model3` 사용
- 기본적으로 10 epoch 학습, 5 epoch마다 `choh_vit_eternet_epoch_*.pt`만 저장
- 5-튜플 로깅은 있으나 실험 추적/검증 로직은 `main_train_eter.py`/`main_train_ss2d.py` 대비 덜 정교함

## 4) 평가·시각화 파이프라인

### `eval.py`

- 인자: `--model eter|ss2d`, `--ckpt CHECKPOINT_PATH`
- config별 모델 로딩 로직을 런타임 import로 분기
- 평가 지표
  - PSNR, NMSE, SSIM, L1
- 출력
  - `results/eval_{model}_{ckptname}.csv`
  - **tqdm 진행률 bar** + 마지막에 평균 ± 표준편차 출력
- forward는 `autocast('cuda')` → 지표는 fp32로 계산
- 입력 데이터: `multicoil_val` 전체 사용 (`reconstruction_rss` 존재 필수)

### `visualize.py`

- 인자: `--model eter|ss2d`, `--ckpt`, `--top_k`
- Pass 1: 전체 val 샘플에서 지표 계산
- Pass 2: composite score 기준 best/worst 샘플만 재추론해서 PNG 저장
- 출력 경로: `results/vis_{model}_{ckpt_name}/`
- 저장 항목
  - GT, Aliased, Reconstructed, Error map 4컷 시각화
  - Composite/SSIM/PSNR/NMSE/L1 라벨 포함

## 5) 환경 점검 유틸

### `tools/check_recon_env.py`

- 목적: 학습 시작 전 환경 fail-fast
- 점검 항목
  - Python 버전
  - `torch`, `h5py`, `numpy`, `einops` import
  - CUDA 사용 가능성/장치 메모리
  - SS2D 모델 시 `mamba_ssm`(선택/필수)
  - `./fastMRI_data/multicoil_train`, `./fastMRI_data/multicoil_val` 존재 및 파일 수
  - config import가 가능한지 확인
- `strict=True`면 누락 조건에서 학습 즉시 종료

## 6) 설정(config) 파일 매핑

### 활성 실험 기준(320 기준, SS2D/ETER 비교군)
- `configs/myConfig_choh_ETER_model.py`
  - `IMAGE_SIZE=(320,320)`, `PATCH_SIZE=(32,32)`, `INPUT_CHANNELS=32`
  - `NUM_VIT_ENCODER_HIDDEN=384`, `NUM_VIT_ENCODER_LAYER=6`, `NUM_VIT_ENCODER_MLP_SIZE=1536`, `NUM_VIT_ENCODER_HEAD=6`
  - `NUM_ETER_HORI_HIDDEN=2`, `NUM_ETER_VERT_HIDDEN=2`
  - `NUM_VIT_DECODER_DIM=512`, `NUM_VIT_DECODER_DEPTH=6`, `NUM_VIT_DECODER_HEAD=8`, `NUM_VIT_DECODER_DIM_MLP_HIDDEN=2048`
  - `NUM_EPOCHS=200`, `BATCH_SIZE=8`, `LR=2e-4`
  - `PATH_FOLDER='./logs/ETER_ViT_R4_brain320/'`
- `configs/myConfig_choh_SS2D_model.py`
  - `IMAGE_SIZE=(320,320)`, `PATCH_SIZE=(32,32)`, `INPUT_CHANNELS=32`
  - ViT: `NUM_VIT_ENCODER_*`를 ETER와 동일하게 둬 비교 편의성 확보
  - SS2D: `NUM_SS2D_D_INNER=32`, `NUM_SS2D_D_STATE=8`, `NUM_SS2D_OUT_CH=20`
  - Decoder: `NUM_VIT_DECODER_*`는 ETER와 동일
  - `PATH_FOLDER='./logs/SS2D_ViT_R4_brain320/'`
- `configs/myConfig_choh_model3.py`
  - 과거 실험 로그와 경로가 매우 많고 값이 누적 덮어쓰기 구조가 있음
  - 마지막에 남은 값이 유효하므로 설정 변경 추적이 까다로움
  - `PATH_FOLDER`가 다단계 문자열 재할당됨

### 레거시/사용하지 않는 경향 높은 설정
- `myConfig_choh_ViT_ETER_R4regular.py`
- `myConfig_choh_ViT_ETER_R4regular_v2.py`
- `myConfig_choh_ViT_recon_R4regular.py`
- `myConfig_choh_ViT_autoencoder_R4regular.py`
- `myConfig_temp.py`
- 공통 특징: 과거 파라미터 실험 로그가 남아 있으며 최근 320/SS2D/ETER 비교용 기준값과 직접 다를 수 있음

## 7) 의존성/실행 정합성(실무 체크)

- 외부 코드 자동 다운로드: `download_repos.py`
  - `hybrid_eternet`, `vit_pytorch`, `mae` GitHub ZIP을 `models/` 하위에 저장
- `main_train_eter.py`/`main_train_ss2d.py`는 상대 경로 기반 import 경로 설정으로 실행 디렉토리 의존성이 있음
- SS2D 실험은 `mamba_ssm`가 없으면 환경 점검에서 즉시 실패
- dataloader는 `reconstruction_rss` 기반 label을 사용해 학습/평가를 기대, 테스트셋 일부(`reconstruction_rss` 없을 수 있음)는 평가 대상으로 부적합
- 로그는 `PATH_FOLDER/log.txt`에 epoch 단위 문자열 append
- 체크포인트 경로는 config별 절대/상대 조합이 다르며, 잘못 매칭 시 `eval.py`/`visualize.py`에서 로드 실패

## 8) 코드 특성상 주의해야 할 포인트

- `main_train.py`는 과거 단순형으로, 최신 ETER/SS2D 비교 실험 기준에는 `main_train_eter.py` 또는 `main_train_ss2d.py`를 권장
- `myConfig_choh_model3.py`, 일부 레거시 config는 `PATH_FOLDER`/파라미터가 여러 번 재할당되어 마지막 값만 유효
- ETER/SS2D는 모두 `data_img`와 `data`를 함께 받는 `model(data_img, data)` 시그니처를 따르므로 가중치 로더/평가 스크립트도 동일 시그니처 필요
- **fp16 SSIM overflow**: label 값 ~959일 때 제곱이 fp16 max(65504) 초과 → loss를 autocast 밖에서 `out.float()`로 fp32 계산 필수
- **시스템 RAM OOM**: DataLoader worker 수가 많으면 systemd-oomd가 프로세스를 kill함 (25.4 GB peak 관측). `num_workers=4, prefetch_factor=2` (train) / `num_workers=2` (val)로 제한해 해결
- **composite scoring 한계**: best 대비 비율 기반이라 초반 큰 점프 후 갱신이 잘 안 됨 → `*_best.pt`가 epoch 10 것일 수 있음. 실제 최선은 마지막 epoch ckpt인 경우가 많음
- **tmux 사용 권장**: GUI 터미널(vte-spawn)에서 실행 시 OOM으로 터미널+학습 모두 사라짐. `tmux new -s train`으로 격리 필요

## 8.1) SS2D 50 epoch 학습 결과 (2026-04-09 완료)

| 시점 | val SSIM | val PSNR | val NMSE | val L1 |
|---|---|---|---|---|
| Epoch 1 | 0.4449 | 20.18 | 0.3608 | 38.60 |
| Epoch 10 | 0.5261 | 24.31 | 0.1603 | 23.75 |
| Epoch 30 | 0.5693 | 25.67 | 0.1431 | 20.31 |
| Epoch 50 | 0.5846 | 26.03 | 0.1399 | 19.51 |

- 소요 시간: 32147초 (~8.9시간), RTX 5060 Ti 8 GB
- 50 epoch까지 수렴하지 않음 (모든 지표 개선 중) → 200 epoch으로 연장
- 과적합 없음 (train SSIM 0.62 vs val 0.58, 차이 작음)
- fastMRI U-Net baseline (SSIM ~0.90+) 대비 부족 — ViT random init from scratch의 한계

## 9) 한 줄 요약

이 저장소는 FastMRI 기준의 320 또는 384 입력을 대상으로,  
ViT 인코더 + (ETER GRU 또는 SS2D 디코더) 구조를 학습해 k-space aliasing 복원 성능을 측정하고,  
`eval`·`visualize`로 정량/정성 평가까지 포함하는 end-to-end MRI 재구성 실험 코드베이스다.

## 10) 함수/클래스 호출 그래프 (텍스트)

### 10.1 `main_train_eter.py`

```text
main()
├─ check_env_for_model("eter", "myConfig_choh_ETER_model", strict=True)
│  └─ 환경 패키지/CUDA/데이터셋/config 점검
├─ choh_ViT(...)                 # models/hybrid_eternet/u_choh_model_ETER_ViT.py
├─ choh_Decoder3_ETER_skip_up_tail(...)
│  ├─ encoder.to_patch_embedding(...)
│  ├─ encoder.transformer(...)     # ViT 인코더
│  ├─ self.enc_to_dec(...)
│  ├─ Transformer.decoder(...)
│  ├─ final_linear(...)
│  ├─ up_tail(Upsample × n)
│  ├─ gru_h(...)                   # in_ksp 기반 수평 양방향 GRU
│  ├─ gru_v(...)                   # 수직 양방향 GRU
│  └─ self.last(...)               # 최종 1채널 재구성
├─ for epoch in range(NUM_EPOCHS)
│  ├─ for sample in trainloader
│  │  ├─ with autocast: model(data_in_img, data_in)
│  │  ├─ loss = L1 + λ(1-SSIM) [fp32, autocast 밖]
│  │  ├─ GradScaler -> backward -> optimizer -> scheduler
│  │  ├─ wandb.log(train/loss, ssim, psnr, nmse, lr)
│  │  └─ batch 메트릭 출력 (tqdm)
│  ├─ wandb.log(epoch/train_loss, ssim, psnr, nmse, l1)
│  ├─ if avg_train_ssim 갱신 시
│  │  └─ run_val(...) [tqdm bar 있음]
│  │     ├─ model.eval()
│  │     ├─ val_loader 순회 (BS=4)
│  │     ├─ model(data_in_img, data_in)
│  │     ├─ PSNR / NMSE / SSIM / L1 집계
│  │     ├─ wandb.log(val/ssim, psnr, nmse, l1)
│  │     └─ model.train()
│  ├─ log.txt append
│  └─ ckpt 저장 (epoch % 5)
└─ best composite 계산 후 eter_vit_best.pt 저장, wandb.finish()
```

### 10.2 `main_train_ss2d.py`

```text
main()
├─ check_env_for_model("ss2d", "myConfig_choh_SS2D_model", strict=True)
├─ choh_ViT(...)
├─ choh_Decoder_SS2D_ViT(...)
│  ├─ encoder.to_patch_embedding(...)
│  ├─ encoder.transformer(...)
│  ├─ self.decoder(Transformer)
│  ├─ final_linear()
│  ├─ up_tail(Upsample × n)
│  ├─ ss2d(in_ksp)  # SS2D.forward
│  │  ├─ norm_in -> in_proj -> dwconv
│  │  ├─ SelectiveScan1D(수평 fwd)
│  │  ├─ SelectiveScan1D(수평 bwd)
│  │  ├─ SelectiveScan1D(수직 fwd)
│  │  ├─ SelectiveScan1D(수직 bwd)
│  │  ├─ merge_norm/merge
│  │  └─ out_proj
│  └─ self.last(...)
├─ for epoch/ batch:
│  ├─ model(data_in_img, data_in)
│  ├─ loss = L1 + λ(1-SSIM) [fp32, autocast 밖]
│  ├─ wandb.log(train/*, epoch/*)
│  └─ run_val 분기/ckpt 저장/wandb val 로깅은 ETER과 동일
└─ best composite 계산 후 ss2d_vit_best.pt 저장, wandb.finish()
```

### 10.3 `eval.py`

```text
main()
├─ model_type + ckpt 파라미터 파싱
├─ load_model(model_type, ckpt_path, device)
│  ├─ model_type=="eter"
│  │  ├─ myConfig_choh_ETER_model import
│  │  ├─ choh_ViT(...)
│  │  └─ choh_Decoder3_ETER_skip_up_tail(...)
│  ├─ model_type=="ss2d"
│  │  ├─ myConfig_choh_SS2D_model import
│  │  ├─ choh_ViT(...)
│  │  └─ choh_Decoder_SS2D_ViT(...)
│  └─ state_dict load, model.eval()
├─ FastMRI_H5_Dataloader("./fastMRI_data/multicoil_val", num_files=None)
├─ val_loader 순회
│  ├─ model(data_in_img, data_in)
│  ├─ psnr / nmse / ssim / l1 계산
│  └─ 결과 CSV append
└─ 평균/표준편차 출력
```

### 10.4 `visualize.py`

```text
main()
├─ load_model(...)               # eval.py와 동일한 모델 로딩 패턴
├─ Pass 1: 전체 val_loader 순회
│  ├─ model(data_in_img, data_in)
│  ├─ SSIM/PSNR/NMSE/L1 계산
│  └─ all_metrics 수집
├─ compute_composite(all_metrics)
│  ├─ SSIM, PSNR: 높을수록 유리 정규화
│  └─ NMSE, L1: 낮을수록 유리 정규화 후 뒤집기
├─ best/top_k, worst/top_k 추출
└─ Pass 2: selected 샘플만 재추론
   ├─ GT / aliased / recon numpy 변환
   ├─ save_figure(...) PNG 저장
   └─ 콘솔에 best/worst 요약 출력
```

### 10.5 `dataloaders/dataloader_h5.py` 핵심 호출

```text
FastMRI_H5_Dataloader(...)
├─ __init__:
│  ├─ H5 파일 스캔 및 reconstruction_rss 존재/크기 필터
│  ├─ 마스크 build_r4_mask 생성(폭축)
│  └─ samples[(file,slice,has_rss)] 생성
└─ __getitem__(idx)
   ├─ h5에서 kspace, reconstruction_rss 읽기
   ├─ ifft2c(kspace) -> complex_center_crop(target_size)
   ├─ fft2c(cropped image) -> ksp_crop
   ├─ R4 mask 적용
   ├─ ifft2c(masked kspace) -> data_img
   ├─ _pack_complex_to_real(data, data_img)
   └─ label = reconstruction_rss or fallback RSS
```

## 11) config 변경 이력(요약)

### 11.1 현재 실험 기준에서 직접 사용되는 설정

1) `myConfig_choh_ETER_model.py`
- 목적: ETER 비교군 고정 설정
- 최종 유효값
  - `IMAGE_SIZE=(320,320)`, `PATCH_SIZE=(32,32)`, `INPUT_CHANNELS=32`
  - `BATCH_SIZE=8`, `NUM_EPOCHS=200`, `LEARNING_RATE_ADAM=2e-4`
  - ViT: `NUM_VIT_ENCODER_HIDDEN=384`, `NUM_VIT_ENCODER_LAYER=6`, `NUM_VIT_ENCODER_MLP_SIZE=1536`, `NUM_VIT_ENCODER_HEAD=6`
  - ETER: `NUM_ETER_HORI_HIDDEN=2`, `NUM_ETER_VERT_HIDDEN=2`
  - Decoder: `NUM_VIT_DECODER_DIM=512`, `NUM_VIT_DECODER_DEPTH=6`, `NUM_VIT_DECODER_HEAD=8`, `NUM_VIT_DECODER_DIM_MLP_HIDDEN=2048`, `NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH=256`, `NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT=16`
  - `PATH_FOLDER='./logs/ETER_ViT_R4_brain320/'`

2) `myConfig_choh_SS2D_model.py`
- 목적: SS2D 비교군 고정 설정
- 최종 유효값
  - `IMAGE_SIZE=(320,320)`, `PATCH_SIZE=(32,32)`, `INPUT_CHANNELS=32`
  - `BATCH_SIZE=8`, `NUM_EPOCHS=200`, `LEARNING_RATE_ADAM=2e-4`
  - ViT: `NUM_VIT_ENCODER_HIDDEN=384`, `NUM_VIT_ENCODER_LAYER=6`, `NUM_VIT_ENCODER_MLP_SIZE=1536`, `NUM_VIT_ENCODER_HEAD=6`
  - SS2D: `NUM_SS2D_D_INNER=32`, `NUM_SS2D_D_STATE=8`, `NUM_SS2D_OUT_CH=20`
  - Decoder: `NUM_VIT_DECODER_DIM=512`, `NUM_VIT_DECODER_DEPTH=6`, `NUM_VIT_DECODER_HEAD=8`, `NUM_VIT_DECODER_DIM_MLP_HIDDEN=2048`, `NUM_VIT_DECODER_FINAL_LINEAR_OUT_CH=256`, `NUM_VIT_DECODER_FINAL_LINEAR_OUT_FEAT=16`
  - `PATH_FOLDER='./logs/SS2D_ViT_R4_brain320/'`

3) `myConfig_choh_model3.py`
- 목적: 레거시지만 간헐적으로 `main_train.py`에서 참조 가능한 스냅샷
- 특징
  - `PATH_FOLDER`가 다단계로 마지막 값이 최종 적용(현재 `logs/251124_model3_R4random_enc_B_dec_ismrm_adam041_exp3_18/`, 조건부 `./` 추가)
  - ViT/ETER/Decoder 파라미터도 마지막 블록의 값이 유효
  - 실제 하이퍼파라미터는 320 실험군이 아니라 과거 실험 로그 영향이 커서 직접 사용 시 버전관리 오해 가능

### 11.2 레거시 설정(비활성/실험 후보)

- `myConfig_choh_ViT_ETER_R4regular.py`
- `myConfig_choh_ViT_ETER_R4regular_v2.py`
- `myConfig_choh_ViT_recon_R4regular.py`
- `myConfig_choh_ViT_autoencoder_R4regular.py`
- `myConfig_temp.py`

공통 패턴
- 과거 실험 로그가 대량 누적되어 `PATH_FOLDER`가 여러 번 재정의됨
- 변수 값도 대체 실험 템플릿이 섞여 있어, 파일만 바꾸면 결과가 어떻게 달라지는지 예측이 어렵다는 특성
- `NUM_VIT_*` 계열은 대체로 Base/Large 계열 값이 혼재 상태(일부 파일 마지막 라인만 실행 시 유효)

요약 규칙
- 각 설정 파일은 “마지막 비주석 할당 라인”이 import 시 최종값이므로,
  실수로 과거 라인을 살펴보고 잘못된 값으로 오인해 버리는 일이 잦다.
- 비교 실험을 안정적으로 하려면:
  - ETER: `myConfig_choh_ETER_model.py`
  - SS2D: `myConfig_choh_SS2D_model.py`
  - 이 두 개로 통일하고, 실행 로그에 `PATH_FOLDER`, `NUM_EPOCHS`, `BATCH_SIZE`, `NUM_ETER_H* / NUM_SS2D_*`, `NUM_VIT_DECODER_*`를 항상 출력해 추적한다.

## 12) `main_train.py` 레거시 실행 경로 정밀 정리

### 12.1 전체 실행 의존성
- 진입점: `if __name__ == '__main__': main()`
- 실행 방식:
  - `sys.path`에 `configs`, `dataloaders`, `models/*`를 강제로 추가
  - `from myConfig_choh_model3 import *` 기반의 전역 하이퍼파라미터 주입
  - `FastMRI_H5_Dataloader` + `choh_ViT` + `choh_Decoder3_ETER_skip_up_tail` + `SSIM` 조합으로 고정된 레거시 파이프라인 구성
- 환경 강제치:
  - `os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'`
- 데이터/디바이스:
  - `torch.cuda.is_available()` 분기 후 `cuda` 사용 가능 시 GPU 우선
  - 실패 시 CPU 폴백(다만 내부 `torch.cuda.FloatTensor` 캐스팅은 CUDA 전제성이 강해 CPU 사용성이 사실상 제한)

### 12.2 호출 그래프 (텍스트)

```text
main_train.py main()
├─ 환경 설정
│  ├─ os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
│  └─ sys.path.append(configs|dataloaders|models/hybrid_eternet|models/mae|models/vit_pytorch)
├─ 모델 생성
│  ├─ choh_ViT(image_size=(384,384), patch_size=(32,32), num_classes=1000, ...)
│  │  └─ 하이퍼파라미터 주입: NUM_VIT_ENCODER_HIDDEN / NUM_VIT_ENCODER_LAYER / NUM_VIT_ENCODER_HEAD / NUM_VIT_ENCODER_MLP_SIZE
│  ├─ choh_Decoder3_ETER_skip_up_tail(encoder=vit_choh, eter_h=NUM_ETER_HORI_HIDDEN, eter_v=NUM_ETER_VERT_HIDDEN, decoder_*)
│  │  ├─ encoder.to_patch_embedding
│  │  ├─ encoder.transformer (forward 내부)
│  │  ├─ 인코더-디코더 latent projection
│  │  ├─ decoder transformer
│  │  ├─ up_tail(PixelShuffle chain)
│  │  ├─ gru_h / gru_v
│  │  └─ 마지막 1채널 합성 Conv
│  └─ criterion
│     ├─ nn.L1Loss
│     └─ SSIM().to(device)
├─ 옵티마이저/스케줄러
│  ├─ torch.optim.Adam(choh_decoder.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=LAMBDA_REGULAR_PER_PIXEL)
│  └─ CosineAnnealingWarmRestarts(T_0=40, T_mult=2, eta_min=0)
├─ 데이터 로더
│  └─ FastMRI_H5_Dataloader('./fastMRI_data/multicoil_train', num_files=None)
│     ├─ DataLoader(batch_size=1, shuffle=True)
│     └─ 예외 시 즉시 return
└─ 학습 반복
   ├─ for epoch in range(10)
   │  ├─ for batch in trainloader
   │  │  ├─ sample['data'] → data_in (FloatTensor, cuda cast)
   │  │  ├─ sample['data_img'] → data_in_img (FloatTensor, cuda cast)
   │  │  ├─ sample['label'] → data_ref (FloatTensor, cuda cast)
   │  │  ├─ with autocast('cuda'):
   │  │  │  ├─ out = choh_decoder(data_in_img, data_in)
   │  │  │  ├─ loss_pixel = L1(out, data_ref)
   │  │  │  ├─ loss_ssim = 1 - SSIM(out, data_ref)
   │  │  │  └─ loss = loss_pixel + LAMBDA_SSIM_PER_PIXEL * loss_ssim
   │  │  ├─ GradScaler.scale(loss).backward()
   │  │  ├─ scaler.step(optimizer)
   │  │  ├─ scaler.update()
   │  │  ├─ optimizer.zero_grad()
   │  │  └─ scheduler.step()
   │  └─ 배치 단위 손실/SSIM/LR 출력
   └─ if (epoch+1) % 5 == 0:
      └─ torch.save(choh_decoder.state_dict(), './models/choh_vit_eternet_epoch_{epoch+1}.pt')
```

### 12.3 왜 레거시로 분류되어야 하는가 (기능 관점)
- 하이퍼파라미터 고정성이 약함
  - `myConfig_choh_model3.py`의 다중 재정의 패턴이 반영되어, 파일 내 최종 값 의존성이 높음
- 실험 추적 미흡
  - `PATH_FOLDER`, `best model`/`composite` 로직이 없음
  - validation 단계 부재(정규화된 모델 비교용 metric 수집 불가)
- 운영 안정성 이슈
  - 학습 루프가 `batch_size=1` 고정이라 최신 스크립트 대비 성능 비교가 비효율적
  - `torch.cuda.FloatTensor` 직접 캐스팅으로 CPU fallback 환경에서 예외 가능성 존재
- 모델 비교 정책 충돌
  - 최신 실험표준(`main_train_eter.py`, `main_train_ss2d.py`)은 `STRICT` 환경 체크 + 로그/체크포인트 정책 + `NUM_VAL_FILES` 기반 검증 분기 사용

### 12.4 main_train.py를 운영/분석 시 권장 보완 목록
1. `myConfig_choh_model3` 의존 제거
   - `main_train_eter.py`/`main_train_ss2d.py`로 통합하고 타입별 config를 명시적 import
2. 검증 단계 복원
   - epoch 단위 또는 성능 임계치 기반 `run_val` 추가
3. 장치 처리 정합성
   - `torch.cuda.FloatTensor` 하드코딩 제거, `to(device)` 일괄화
4. 체크포인트 정책 통일
   - config별 `PATH_FOLDER` 하위로 epoch/best 분기 저장
5. 학습 추적 정합화
   - 로그 템플릿(손실/지표/학습시간)을 최신 스크립트 스키마에 맞춰 통일
6. `main_train.py`를 엔트리포인트 통합기로 정비
   - `--mode legacy|eter|ss2d` 라우팅으로 실행 형태를 명시화하고
   - 기본 실행 동작은 기존(`legacy`)과 호환, 최신 실험은 `eter`/`ss2d` 라우팅 사용

## 13) “할 수 있는 것은 다” 반영 체크리스트 (현재 기준 완료 항목)
- [x] 프로젝트 핵심 파일을 섹션 단위로 분해해 읽기/파싱 압력 감소
- [x] ETER/SS2D 핵심 엔트리포인트 전체 호출 그래프 정리
- [x] `main_train.py` 레거시 실행 경로 호출 그래프 정리
- [x] 설정 변경/최종값 추적 규칙(마지막 비주석 할당) 명시
- [x] 레거시 대비 최신 경로 마이그레이션 가이드 제시
