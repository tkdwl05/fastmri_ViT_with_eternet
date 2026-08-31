#!/bin/bash
# ============================================================================
#  [2/6] 호스트에서 실행 — 현재 상태 진단 + 새 컨테이너 설정값 확정
#
#  이 스크립트는 아무것도 바꾸지 않는다(읽기 전용). 하는 일:
#    · GPU / 드라이버 / nvidia-container-toolkit 상태 확인
#    · 기존 컨테이너에서 마운트·포트·GPU 지정을 추출해 container.env 로 저장
#    · 디스크 여유 확인 (새 이미지 ~14GB 필요)
#
#  실행:  bash 10_host_preflight.sh            (기본 컨테이너명 자동탐색)
#         OLD_CTN=snorlax_GPU0_v2 bash 10_host_preflight.sh
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HERE}/container.env"

echo "════════ [A] 호스트 GPU 상태 ════════"
if nvidia-smi -L 2>/dev/null; then
  echo "  ✓ 호스트 nvidia-smi 정상"
  HOST_GPU_OK=1
else
  echo "  ✗ 호스트에서도 nvidia-smi 실패 → 드라이버/커널 모듈 레벨 문제."
  echo "    이 경우 컨테이너를 새로 만들어도 해결되지 않는다. 먼저 호스트를 고칠 것:"
  echo "      sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia"
  echo "      (실패하면 호스트 재부팅)"
  HOST_GPU_OK=0
fi
echo "  드라이버: $(cat /proc/driver/nvidia/version 2>/dev/null | head -1)"
echo "  디바이스 노드:"; ls -la /dev/nvidia* 2>/dev/null | sed 's/^/    /'

echo
echo "════════ [B] nvidia-container-toolkit ════════"
for c in nvidia-container-cli nvidia-container-runtime nvidia-ctk; do
  printf "  %-26s %s\n" "$c" "$(command -v $c >/dev/null && $c --version 2>/dev/null | head -1 || echo '(없음)')"
done
echo "  runtime 등록 여부:"
docker info 2>/dev/null | grep -i -A2 "Runtimes" | sed 's/^/    /'
echo "  cgroup driver: $(docker info 2>/dev/null | grep -i 'Cgroup Driver' | sed 's/^ *//')"
echo "  cgroup version: $(docker info 2>/dev/null | grep -i 'Cgroup Version' | sed 's/^ *//')"
echo
echo "  [참고] 컨테이너 안에서 /dev/nvidia0 open() 이 EPERM 이면 이미지 문제가 아니라"
echo "         device cgroup allowlist 에서 nvidia 노드가 빠진 것이다(NVIDIA/nvidia-docker#1730)."
echo "         30_host_run.sh 는 --device 를 명시해 도커가 규칙을 직접 소유하도록 한다."

echo
echo "════════ [C] 기존 컨테이너 설정 추출 ════════"
OLD_CTN="${OLD_CTN:-}"
if [ -z "$OLD_CTN" ]; then
  OLD_CTN=$(docker ps -a --format '{{.Names}}' 2>/dev/null | grep -iE 'snorlax|work0|gpu0' | head -1)
fi
if [ -z "$OLD_CTN" ] || ! docker inspect "$OLD_CTN" >/dev/null 2>&1; then
  echo "  ⚠ 기존 컨테이너를 자동으로 못 찾음. 아래에서 골라 OLD_CTN= 로 다시 실행:"
  docker ps -a --format '    {{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null
  exit 1
fi
echo "  대상: $OLD_CTN"
docker inspect "$OLD_CTN" > "${HERE}/old_container_inspect.json"

OLD_MOUNTS=$(docker inspect -f '{{range .Mounts}}{{if eq .Type "bind"}}-v {{.Source}}:{{.Destination}} {{end}}{{end}}' "$OLD_CTN")
OLD_PORTS=$(docker inspect -f '{{range $p, $c := .HostConfig.PortBindings}}{{range $c}}-p {{.HostPort}}:{{$p}} {{end}}{{end}}' "$OLD_CTN")
OLD_WORKDIR=$(docker inspect -f '{{.Config.WorkingDir}}' "$OLD_CTN")
OLD_SHM=$(docker inspect -f '{{.HostConfig.ShmSize}}' "$OLD_CTN")
OLD_MEM=$(docker inspect -f '{{.HostConfig.Memory}}' "$OLD_CTN")
OLD_IMAGE=$(docker inspect -f '{{.Config.Image}}' "$OLD_CTN")
OLD_NETMODE=$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$OLD_CTN")

echo "  이미지 : $OLD_IMAGE"
echo "  마운트 : ${OLD_MOUNTS:-(없음)}"
echo "  포트   : ${OLD_PORTS:-(없음)}"
echo "  workdir: ${OLD_WORKDIR:-(없음)}"
echo "  shm    : ${OLD_SHM} bytes / memory: ${OLD_MEM} bytes (0=무제한)"
echo "  network: ${OLD_NETMODE}"

echo
echo "════════ [D] 디스크 여유 ════════"
docker system df 2>/dev/null | sed 's/^/  /'
echo "  /var/lib/docker 여유:"; df -h /var/lib/docker 2>/dev/null | tail -1 | sed 's/^/    /'
echo "  ※ 새 이미지 빌드에 ~14GB 필요."

# --- GPU 지정: 우리 TITAN RTX 를 UUID 로 특정 -------------------------------
# 인덱스(0)는 드라이버 재적재·카드 추가 시 밀릴 수 있다. 교수님 GPU 를 실수로
# 잡는 사고를 원천 차단하려면 UUID 로 못 박는 것이 안전하다.
OUR_GPU_UUID="${OUR_GPU_UUID:-GPU-93881c1e-8244-c1c6-cbae-26c672d13400}"   # 0000:51:00.0 TITAN RTX
echo
echo "════════ [E] GPU 지정 (UUID 고정) ════════"
echo "  호스트 GPU 목록:"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id --format=csv 2>/dev/null | sed 's/^/    /'

# UUID → /dev/nvidiaN 매핑을 /proc 에서 역산 (nvidia-smi 없이도 동작)
OUR_MINOR=""
for info in /proc/driver/nvidia/gpus/*/information; do
  if grep -q "$OUR_GPU_UUID" "$info" 2>/dev/null; then
    OUR_MINOR=$(grep -i "Device Minor" "$info" 2>/dev/null | awk '{print $NF}')
    echo "  ✓ UUID 일치: $(dirname "$info" | xargs basename)  →  /dev/nvidia${OUR_MINOR}"
  fi
done
if [ -z "$OUR_MINOR" ]; then
  echo "  ⚠ UUID '${OUR_GPU_UUID}' 를 호스트에서 못 찾음 → 인덱스 0 으로 폴백."
  echo "    카드가 바뀐 게 아니라면 위 목록에서 우리 TITAN RTX 의 UUID 를 확인해"
  echo "    OUR_GPU_UUID= 로 다시 실행할 것."
  OUR_MINOR=0
  VISIBLE="0"
else
  VISIBLE="$OUR_GPU_UUID"
fi

# 공통 제어 노드 + 우리 카드의 노드만. 다른 /dev/nvidiaN 은 절대 넣지 않는다.
NV_DEVICES=""
for d in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
  [ -e "$d" ] && NV_DEVICES="${NV_DEVICES} --device ${d}"
done
[ -e "/dev/nvidia${OUR_MINOR}" ] && NV_DEVICES="${NV_DEVICES} --device /dev/nvidia${OUR_MINOR}"
echo "  최종 디바이스: ${NV_DEVICES}"
echo "  NVIDIA_VISIBLE_DEVICES=${VISIBLE}"

# --- container.env 생성 ------------------------------------------------------

cat > "$OUT" <<ENVEOF
# 10_host_preflight.sh 가 $(date -Is) 에 생성. 30_host_run.sh 가 읽는다.
# 값이 마음에 안 들면 여기서 직접 고치면 된다.

OLD_CTN="${OLD_CTN}"
NEW_CTN="\${NEW_CTN:-mri_gpu0}"
NEW_HOSTNAME="\${NEW_HOSTNAME:-snorlax_WORK0}"
IMAGE="\${IMAGE:-mri:v1}"

# 기존 컨테이너에서 추출 (필요시 수정)
MOUNTS="${OLD_MOUNTS}"
PORTS="${OLD_PORTS}"
WORKDIR="${OLD_WORKDIR:-/home/snorlax/shared/fastmri_ViT_with_eternet}"

# 네트워크 모드 승계. host 모드면 --hostname/-p 를 쓸 수 없으므로 30_host_run.sh 가
# 자동으로 그 둘을 뺀다.
NETMODE="${OLD_NETMODE:-default}"

# 자원 — 07-08 설정이 실제로 반영돼 있었음(shm 128G, memory.max 256GiB) → 유지
SHM_SIZE="128g"
MEM_LIMIT="256g"

# GPU 노출: 우리 TITAN RTX 한 장만. 교수님 GPU 는 노드 자체를 넣지 않으므로
# 컨테이너 안에서 존재조차 보이지 않는다(설정 실수로도 접근 불가).
#   · NVIDIA_VISIBLE_DEVICES  = UUID 고정 → 인덱스가 밀려도 다른 카드를 안 잡음
#   · --device                = 우리 카드 노드 + 공통 제어 노드만
#     이 --device 명시가 이번 재구성의 핵심이기도 하다 — 도커가 device cgroup
#     규칙을 직접 소유하게 되어 systemd 가 cgroup 을 재적용해도 규칙이 복원된다.
NVIDIA_VISIBLE="${VISIBLE}"
NV_DEVICES="${NV_DEVICES}"
ENVEOF

echo
echo "══ 설정 저장: $OUT"
cat "$OUT" | sed 's/^/  /'
echo
echo "  다음 단계: bash 20_host_build.sh"
