# RunPod 실행 환경 (2026-10-09)

GPU0 한 장으로는 v8fix 50 epoch 비교가 오래 걸려(세 모델 약 13일) RunPod RTX 3090 Pod 를 추가로 쓴다.
비교 공정성을 위해 한 비교에 들어가는 모델들은 같은 GPU 종류에서 돌린다.

## Pod 구성
- RTX 3090 24 GB, vCPU 31, RAM 125 GB, 템플릿 Runpod Pytorch 2.8.0(SSH 22 노출), **컨테이너 디스크 3 TB(로컬 NVMe)**.
- 네트워크 볼륨은 쓰지 않는다: S3 기반 FUSE(geesefs) 볼륨에서 압축 해제 쓰기와 h5 무작위 읽기가 멈췄다(10-09 실측).
  컨테이너 디스크는 Pod 를 멈추거나 지우면 사라지므로, 결과는 아래 git 동기화와 로컬 서버 복사로 보존한다.

## 데이터 (`pod_dl_extract.sh`)
- 공식 fastMRI 링크(개인 서명 URL — 저장소에 두지 않음, Pod 의 `/root/_setup/urls.txt`)에서 Pod 가 직접 받는다
  (Pod 쪽 다운로드 약 30 MB/s × 묶음 수, 압축 해제가 병목).
- train 0~9 묶음 = 로컬 서버와 같은 파일만(`dataset_manifest_nvme.json` 목록) → 로컬과 같은 학습 데이터.
- val 0~2 묶음 전체(1,378 볼륨, 10-09 사용자 결정 — 학습 중 검증부터 사용), test_full 0~2 묶음 전체(정답 RSS 포함, 최종 평가용).
  `test`(대회용, 정답 없음)는 받지 않는다.
- 검증: `pod_verify_data.py` — 대조표와 파일·크기·부분 해시 비교, 로컬에서 잘려 로더가 건너뛰는 train 2 파일 삭제,
  로더 슬라이스 수 확인(train 4108 파일 / 65,028 슬라이스).

## 환경 (`pod_setup_env.sh`)
로컬 Docker `mri:v1`(infra/docker/Dockerfile)과 같은 핵심 버전: python 3.10 venv(`/opt/mri310`), torch 2.3.1+cu121,
numpy 1.26.4, mamba_ssm 2.2.2·causal_conv1d 1.4.0 prebuilt wheel, setuptools 69.5.1(triton import 에 필요).

## git 연동 (`pod_git_sync.sh`)
- Pod 에는 이 저장소 전용 **Deploy key**(쓰기 허용)만 둔다(`/root/.ssh/id_ed25519_github`). 개인 계정 키는 두지 않는다.
- Pod 의 저장소는 `runpod/v8fix` 브랜치. 동기화 스크립트가 30 분마다 작은 텍스트 결과만 `runpod_results/` 에 모아 push 한다.
- 코드 갱신: 로컬에서 작업 브랜치에 push → 런 사이에 Pod 에서 `git fetch origin && git merge origin/<작업 브랜치>`.
  실행 중인 런의 코드는 바꾸지 않는다(자동 pull 없음).
