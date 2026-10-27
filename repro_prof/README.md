# repro_prof — 교수님 순수 ETER-net 학습 재현 도구

교수님 fastMRI 순수 ETER-net 학습(2023-10, d-ETER `scripts_legacy/choh_train_ETER_hybrid_fastmri_brain_acs32R4_231030.py`,
o-ETER `scripts_legacy/choh_train_oETER_fastmri_brain_fixed_acs32R4_231022.py`)을 같은 데이터(.mat)·같은 로더 출력·같은
학습 루프로 다시 돌리기 위한 새 코드다. 교수님 원본 파일(초기 커밋 7d4e4e0)과 `/mnt/sda/choh/shared/` 는 읽기·import 만 한다.

## 파일

| 파일 | 내용 |
|---|---|
| `config_repro.py` | 모든 설정 값과 출처(코드 줄·잔재 줄·런 이름·논문·설계·우리 추가). 프리셋 `dETER`·`oETER`·`unet_only`. 시작할 때 전부 출력 |
| `train_repro.py` | 학습 루프(231030:164-305 그대로: 묶음 루프, 묶음 안 셔플, fp32, 스케줄·clip 없음, 매 epoch 저장, 검증 마지막 배치) + 우리 추가(시드, 재개, 감시, 캐시·미리 읽기) |
| `models_repro.py` | o-ETER·d-ETER 사본(원문과 키·초기값·출력 동일 확인), 정제망 `dfu`·`dfu_plain`·`unet2015`·`fastmri`, U-Net 단독 기준선(`coil32`·`rss1`) |
| `unet2015.py` | 원래 U-Net(Ronneberger 2015): max pooling, 수준마다 3×3 conv+ReLU 2번, up-conv, skip 이어 붙이기, DFU 잔차·원입력 연결 없음 |
| `myoptimizer_shim.py` | 없는 인자 3개 `myOptimizer` 대체(가정은 파일 머리말) |
| `monitor.py` | 가중치 소멸 감시(|w|<1e-6 비율, v8fix 와 같은 문턱·그룹, 기본 기록만) |
| `data_cache.py` | 교수님 로더 출력을 비트 단위로 재현하는 압축 캐시(빌드·검증·읽기 CLI) |
| `prof_paths.py` | 원본 경로·무수정 import 도우미(로더 경로 접두사는 메모리에서만 바꿈) |
| `selftest_cpu.py` | CPU 점검: 파라미터 수(교수님 식·S22·JI26·2020 공개 코드), 원문 클래스 대 사본, 작은 순전파·역전파, 설정 출처 |
| `smoke_gpu.py` | GPU0 짧은 스모크: 원문 클래스 대 사본 출력 비트 비교, 모델별 메모리·step 시간·손실 감소, 시간 추정 |

## 실행 (저장소 루트)

```bash
# 0) CPU 점검
python -B repro_prof/selftest_cpu.py
python -B repro_prof/train_repro.py --preset dETER --dry-run          # 설정 출력·파라미터 수·디스크·데이터 확인

# 1) 캐시(한 번, CPU, 약 39.3 GB, NVMe). 만든 뒤 몇 볼륨을 원 로더와 비트 대조
python -B repro_prof/data_cache.py build --out /home/snorlax/shared/cache_repro_prof/prof379_f32_v1 --start 0 --end 379 --workers 4
python -B repro_prof/data_cache.py verify --out /home/snorlax/shared/cache_repro_prof/prof379_f32_v1 --idx 0 43 62 150 299 300 320 378

# 2) 학습 (GPU0 이 비어 있을 때)
CUDA_VISIBLE_DEVICES=0 python -B repro_prof/train_repro.py --preset dETER
CUDA_VISIBLE_DEVICES=0 python -B repro_prof/train_repro.py --preset unet_only --set N_UNET_DEPTH=5
CUDA_VISIBLE_DEVICES=0 python -B repro_prof/train_repro.py --preset dETER --resume     # 같은 RUN_NAME 으로 epoch 단위 재개
#   값 바꾸기: --set NAME=VALUE (예: BATCH_SIZE=1, N_UNET_DEPTH=5, REFINE='unet2015', BASELINE_INPUT='rss1', DATA_SOURCE='prof')

# 3) 스모크(몇십 step)
CUDA_VISIBLE_DEVICES=0 python -B repro_prof/smoke_gpu.py --cache <캐시> --out <스크래치> --steps 40
```

- 기록: `logs/repro_prof/<RUN_NAME>/`(log.txt, settings.json, train_steps.csv, training_loss.npy, validation_loss.npy)
- 체크포인트: `/mnt/sda/snorlax/repro_prof/ckpt/<RUN_NAME>/`(tensors_{epoch}.pt, tensors_entire.pt, resume.pt). 시작할 때 여유 공간을 확인한다.
- `DATA_SOURCE='prof'` 는 교수님 로더 클래스를 그대로 돌린다(볼륨당 약 10 초, GPU 는 그동안 쉰다 — 원 코드와 같음).

## 원본과 다른 점 (모두 config 에 'ours' 로 표시)

- h0 를 `.cuda()` 대신 입력 장치에 만든다(값 같음). 모듈 상수는 생성자 인자.
- 체크포인트 기본 형식은 state_dict(`CKPT_FORMAT='full_model'` 이면 교수님처럼 모델 전체).
- 시드와 묶음별 전용 셔플 생성기(교수님 코드에는 시드가 없다), epoch 단위 재개, 감시 기록, 검증 볼륨 전체 평균 기록.
- 캐시와 다음 묶음 미리 읽기(입력은 원 로더와 비트 단위로 같다).
- Ampere GPU 에서 TF32 끔.
