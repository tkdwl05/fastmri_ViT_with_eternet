# Docker 환경 재구성 런북 (2026-08-31)

기존 운영 컨테이너(`snorlax_WORK0`)를 버리고, **Dockerfile 로 정의된 새 이미지**에서
환경을 다시 세운다. 이 문서는 ① 기존 환경이 어떻게 만들어졌는지 ② 지금 무엇이
고장났는지 ③ 어떤 순서로 갈아탈지를 담는다.

---

## 1. 기존 환경은 어떻게 구성되어 있었나 (역방향 분석)

Dockerfile 이 없었기 때문에 컨테이너 내부 증거(`apt history`, `pip freeze`, 마운트,
`/proc`)로 역산했다.

### 1.1 계보

```
pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel     ← 공식 베이스 (여기까지는 재현 가능)
  │   Ubuntu 22.04.3 · /opt/conda · Python 3.10.14
  │   CUDA 12.1.1 · cuDNN 8.9.0.131 · NCCL 2.17.1 · torch 2.3.1
  │
  ├─ 2025-11-26  apt: sudo, openssh-server, nano, git, x11-apps
  ├─ 2026-05-16  apt: nodejs(20), curl, apt-transport-https, gnupg   ← Claude Code 용
  ├─ 2026-05-20  apt: rsync
  ├─ 2026-05-21  apt: tmux
  ├─ (시점 불명) pip: mamba_ssm 2.2.2, causal_conv1d 1.4.0 (prebuilt wheel),
  │              einops, h5py, scipy, pandas, scikit-image, matplotlib,
  │              wandb 0.27.0, transformers 4.35.2, fastmri 0.3.0(--no-deps),
  │              pypandoc_binary, ninja
  │
  ├─ 2026-07-06  docker commit  → snorlax_GPU01 에서 GPU0 전용 컨테이너 분리
  └─ 2026-07-08  docker commit  → snorlax_gpu0:v2_bigmem_snapshot (현재 이미지)
```

**핵심 문제는 마지막 두 줄이다.** 현재 이미지는 빌드 산출물이 아니라 *돌던 컨테이너를
통째로 찍어낸 스냅샷을 다시 찍어낸 것*이다. 그래서:

- 무엇이 언제 왜 설치됐는지 이미지만 봐서는 알 수 없다 (감사 불가)
- 학습 중 생긴 임시파일·캐시·손상된 상태가 그대로 이미지에 굳어 있다
- 같은 환경을 다시 만들 방법이 없다 (재현 불가)
- 레이어가 계속 누적된다

`docker commit` 은 원래 응급 백업 수단이지 환경 정의 수단이 아니다. 이번 재구성은
이 지점을 끊는 것이 목적이다.

### 1.2 컨테이너 레이어에만 있는 자산 (버리면 사라지는 것)

| 경로 | 내용 | 비고 |
|---|---|---|
| `/root/.ssh/id_ed25519` | GitHub 배포 키 | 08-20 PAT 만료 후 **유일한 인증 수단** |
| `/root/.netrc` | wandb API 토큰 | |
| `/root/.claude` (36M) | Claude Code 프로젝트 메모리·세션 기록 | |
| `/root/.local` (855M) | Claude Code CLI 본체 v2.1.251 | |
| `/opt/conda` (8.5G) | Python 환경 | Dockerfile 로 재생성됨 |

### 1.3 바인드 마운트 — **안전** (재구성과 무관하게 그대로 유지)

| 컨테이너 경로 | 실체 | 내용 |
|---|---|---|
| `/home/snorlax/shared` | NVMe `/dev/nvme0n1p2` | repo + `logs/` **128G 체크포인트 전량** |
| `/mnt/sda` | HDD `/dev/sda1` | fastMRI 원본 tar |

> v7/v8/v9 의 모든 학습 결과는 바인드 마운트에 있다. 컨테이너를 버려도 **학습 결과는
> 하나도 잃지 않는다.** radapt 의 `ss2d_v9_last.pt`(ep43) 도 그대로다.

### 1.4 자원 설정 (현재 값 — 유지할 것)

`shm 128G` · `memory.max 256GiB` · `--restart unless-stopped` — 07-08 에 설정된
값이며 컨테이너 안에서 실측 확인했다(`/sys/fs/cgroup/memory.max` = 274877906944).
DataLoader `num_workers=16` 이 bus error 없이 도는 근거이므로 새 컨테이너도 동일하게 준다.

---

## 2. 지금 고장난 것의 정확한 원인

증상은 `nvidia-smi` → `Failed to initialize NVML: Unknown Error`,
`torch.cuda.is_available()` → `False`. 2026-05-16 이후 최소 6회 재발했다.

컨테이너 안에서 층별로 확인한 결과:

| 계층 | 상태 |
|---|---|
| 커널 모듈 | ✅ 정상 — NVRM 550.163.01 |
| GPU 인식 | ✅ 정상 — `/proc/driver/nvidia/gpus/0000:51:00.0` = TITAN RTX |
| 디바이스 노드 | ✅ 존재 — `/dev/nvidia0`, `nvidiactl`, `nvidia-uvm` (모드 `crw-rw-rw-`) |
| 라이브러리 | ✅ 존재 — `libnvidia-ml.so.550.163.01`, `libcuda.so.550.163.01` |
| **디바이스 open()** | ❌ **root 인데도 `EPERM (Operation not permitted)`** |

파일 권한이 `rw-rw-rw-` 인데 root 의 `open()` 이 EPERM 으로 막히는 경우는 하나뿐이다 —
**cgroup device allowlist 에서 nvidia 노드가 빠진 것**이다.

즉 **이미지 문제가 아니다.** nvidia-container-toolkit 이 컨테이너 생성 시 out-of-band 로
넣어준 device cgroup 규칙을, 호스트에서 systemd 가 cgroup 을 재적용할 때
(`systemctl daemon-reload` 등) 도커가 아는 디바이스 목록만으로 덮어써서 날려버리는
알려진 이슈다 (NVIDIA/nvidia-docker#1730 유형).

### 이것이 재구성 계획에 주는 함의

> **이미지를 새로 만드는 것만으로는 이 증상이 다시 안 나타난다는 보장이 없다.**
> 같은 방식으로 컨테이너를 띄우면 같은 조건에서 또 날아간다.

그래서 이번 재구성은 두 가지를 같이 한다:

1. **깨끗한 이미지** — commit 스냅샷 체인을 끊고 Dockerfile 로 정의 (재현성·감사성)
2. **`--device` 를 명시한 컨테이너 기동** — nvidia 디바이스를 도커 자신의
   `HostConfig.Devices` 에 등록시킨다. 그러면 systemd 가 cgroup 을 재적용해도
   도커가 아는 목록에 nvidia 노드가 포함되어 있으므로 규칙이 **다시 복원된다.**
   (`30_host_run.sh` 가 이걸 한다. 호스트 전역 설정은 건드리지 않으므로
   교수님 GPU1 컨테이너에 아무 영향이 없다.)

### 그래도 재발하면 — 호스트 레벨 선택지 (관리자 조율 필요)

교수님 컨테이너에도 영향이 가므로 단독으로 실행하지 말 것:

- `nvidia-container-toolkit` 업그레이드 (≥1.16 에서 관련 처리 개선)
- `/etc/docker/daemon.json` → `"exec-opts": ["native.cgroupdriver=cgroupfs"]`
  (systemd 가 cgroup 을 재적용하지 않게 됨, 도커 재시작 필요)
- `/etc/nvidia-container-runtime/config.toml` → `no-cgroups = true`
  (툴킷이 cgroup 을 아예 안 건드리고 `--device` 에 전적으로 의존)
- 호스트에서도 `nvidia-smi` 가 실패한다면 그건 드라이버 레벨 문제이며 위 대책과 무관 —
  모듈 재적재 또는 호스트 재부팅이 필요하다.

---

## 3. 새 환경 설계

| 항목 | 값 | 이유 |
|---|---|---|
| 베이스 | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` | 현재와 **동일** 조합. 공식 이미지라 재현 가능 |
| torch | 2.3.1 **고정** | mamba_ssm/causal_conv1d wheel 이 `cu122torch2.3cxx11abiFALSE-cp310` 전용. 올리면 SS2D 커널이 깨지고 v7/v8/v9 완주 런과의 수치 비교가 무효 |
| numpy | 1.26.4 고정 | 2.x 는 torch 2.3.1·mamba 바이너리와 ABI 충돌 |
| SS2D 커널 | prebuilt wheel URL | 소스 빌드 30분+ & ABI 실패 위험 회피 |
| fastmri | `--no-deps` | 현행과 동일. 의존성을 끌면 torch 재설치 위험 |
| 비밀정보 | 이미지에 굽지 않음 | 런타임 복원 (`40_restore_state.sh`) |
| 추가분 | `fonts-nanum` | 기존 환경에 CJK 폰트가 없어 그림 제목 한글이 `□□` 였음 (commit `3bcc88b`) |
| GPU | 우리 TITAN RTX **한 장만** (UUID 고정) | 교수님 GPU 는 디바이스 노드 자체를 안 넣음 → 컨테이너에서 보이지도 않음 |
| 이미지명 | `mri:v1` | |
| 컨테이너명 | `mri_gpu0` (hostname 은 `snorlax_WORK0` 유지) | 기존 로그·스크립트 호환 |

### 3.1 GPU 격리 — 어떻게 GPU0 만 쓰게 되는가

3중으로 막는다. 하나가 뚫려도 나머지가 막는다.

| 계층 | 설정 | 효과 |
|---|---|---|
| ① 디바이스 노드 | `--device /dev/nvidia<우리minor>` + 공통 제어 노드만 | 다른 카드의 노드는 컨테이너에 **존재조차 하지 않음**. `CUDA_VISIBLE_DEVICES` 를 실수로 `1` 로 줘도 잡을 수 없다 |
| ② 툴킷 노출 | `NVIDIA_VISIBLE_DEVICES=GPU-93881c1e-…` (**UUID**) | 인덱스가 아니라 UUID 로 지정 → 드라이버 재적재·카드 추가로 번호가 밀려도 다른 카드를 잡지 않음 |
| ③ 검증 | `30_host_run.sh` 가 기동 직후 `nvidia-smi` 출력 | TITAN RTX 가 **정확히 1개**여야 정상. 2개 보이면 즉시 중단 |

`10_host_preflight.sh` 가 호스트의 `/proc/driver/nvidia/gpus/*/information` 에서 UUID
`GPU-93881c1e-8244-c1c6-cbae-26c672d13400` (= `0000:51:00.0`, 현재 우리 컨테이너가 쓰던
바로 그 카드)를 찾아 대응하는 `/dev/nvidiaN` 을 역산한다. 못 찾으면 인덱스 0 으로
폴백하면서 경고를 띄우므로, 그때는 호스트 `nvidia-smi` 목록에서 UUID 를 확인해
`OUR_GPU_UUID=` 로 다시 실행하면 된다.

기존 설정은 인덱스(`--gpus '"device=0"'`)에만 의존했는데, 이번에 UUID 기준으로 바꿨다.

---

## 4. 진행 순서

각 스크립트는 앞 단계 산출물을 확인하고, 되돌릴 수 없는 지점 전에 멈춰 확인을 받는다.

### [1] 상태 백업 — **현재(옛) 컨테이너 안에서**

```bash
bash infra/docker/00_backup_state.sh
```
`/home/snorlax/shared/container_state_backup/<날짜>/` 에 ssh 키·wandb 토큰·Claude
메모리·CLI 를 tar 로 저장한다. 바인드 마운트라 컨테이너를 버려도 남는다.
**이 단계를 건너뛰면 GitHub 인증과 Claude 메모리를 잃는다.**

### [2] 호스트 진단 + 설정 추출 — **호스트에서**

```bash
bash 10_host_preflight.sh
```
읽기 전용. 호스트 GPU 상태, 툴킷 버전, cgroup 드라이버를 점검하고, 기존 컨테이너에서
마운트·포트를 추출해 `container.env` 를 만든다. 값이 마음에 안 들면 그 파일을 직접 고친다.

> ⚠ **08-31 실사고**: 자동탐색 grep 의 `gpu0` 패턴이 우리 운영 컨테이너가 아니라
> **GPU01**(choh 이미지, `/home/choh` 바인드, shm 64MB — 교수님 쪽)을 잡아,
> `/home/snorlax/shared` 가 빠진 MOUNTS 와 GPU01 의 ssh 포트(22203)가
> `container.env` 로 들어갔다 → 그걸로 만든 새 컨테이너에서는 **repo 가 통째로
> 안 보인다** (도커는 `-v` 로 명시한 호스트 경로만 보여준다). 이후 preflight 에
> 검증 게이트(추출 MOUNTS 에 `/home/snorlax/shared` 필수)와 30_host_run 의
> 동일 게이트를 넣어 재발을 막았고, `container.env` 는 §1.3 실측값으로 수기
> 정정했다. 마운트는 기존 컨테이너에 추가할 수 없으므로, 잘못 만든 컨테이너는
> `docker rm` 하고 다시 `bash 30_host_run.sh` 로 만들어야 한다.

### [3] 이미지 빌드 — **호스트에서**

```bash
bash 20_host_build.sh          # 15~40분
```
완전히 안전하다. 돌고 있는 컨테이너에 영향 없음. 실패하면 그냥 다시 하면 된다.
끝에 import 스모크가 자동으로 돈다.

### [4] 컨테이너 교체 — **호스트에서**

```bash
bash 30_host_run.sh            # YES 입력 게이트 있음
```
> ⚠ **이 시점에 현재 Claude Code 세션이 끊긴다** (세션이 옛 컨테이너 안에서 돌고 있음).
> ⚠ 옛 컨테이너는 **삭제하지 않고 stop 만** 한다 → 언제든 롤백 가능.

### [5] 상태 복원 — **새 컨테이너 안에서**

```bash
docker exec -it mri_gpu0 bash
bash infra/docker/40_restore_state.sh
```
08-31 부터 Claude Code CLI 는 이미지에 미리 설치돼 있다(npm 전역, `/usr/bin/claude`
— 첫 mri:v1 컨테이너에서 수동 설치해야 했던 것 반영). 40_restore 가 `/root/.local`
(구 컨테이너 CLI 본체)을 복원하면 `PATH` 상 그쪽이 우선한다. 로그인·프로젝트
메모리는 `claude_state.tgz` 복원으로 돌아온다 — restore 없이 claude 를 새로 켜면
빈 메모리로 시작하니 반드시 [5]를 먼저 돌릴 것.

### [6] 검증 — **새 컨테이너 안에서**

```bash
bash infra/docker/50_verify_env.sh
```
import 확인이 아니라 실제 동작을 본다: 디바이스 `open()`, CUDA matmul/FFT,
**SS2D selective-scan 커널 실행**, ckpt 로드, h5 읽기, shm 크기, wandb/git 인증.
전부 통과해야 학습 재개.

### [7] 학습 재개

```bash
bash v9_mamba_radapt/runs/post_reboot_rearm.sh     # radapt ep43 부터 true-resume
```

---

## 5. 롤백

새 환경에 문제가 있으면 즉시 되돌린다 (같은 호스트 포트를 쓰므로 새것을 먼저 정지):

```bash
docker stop mri_gpu0
docker start <옛 컨테이너 이름>        # 10_host_preflight.sh 가 출력해 준다
```
옛 컨테이너와 옛 이미지는 재구성이 안정화될 때까지 지우지 말 것.

## 6. 정리 (안정화 확인 후)

새 환경에서 학습 1 epoch 이상 정상 완료를 확인한 뒤에만:

```bash
docker rm <옛 컨테이너>
docker rmi snorlax_gpu0:v2_bigmem_snapshot     # 누적 commit 이미지들
docker image prune
```

---

## 7. 앞으로의 원칙

1. **`docker commit` 으로 환경을 저장하지 않는다.** 패키지가 필요하면 `Dockerfile` 을
   고치고 태그를 올려 다시 빌드한다 (`mri:v2` …). 이 저장소에 커밋해 이력을 남긴다.
2. **컨테이너 레이어에 상태를 두지 않는다.** 학습 산출물은 이미 바인드 마운트에 있다.
   인증·설정도 마운트로 옮기는 편이 낫다(향후 개선 항목).
3. **torch/CUDA 조합은 논문 트랙이 끝날 때까지 동결.** 올려야 할 이유가 생기면 v7/v8/v9
   재평가 비용을 먼저 계산할 것.
4. GPU 가 또 죽으면 진단 순서: 호스트 `nvidia-smi` → 컨테이너 안 `/dev/nvidia0` `open()`
   → EPERM 이면 cgroup 문제(§2), 호스트도 실패하면 드라이버 문제.
