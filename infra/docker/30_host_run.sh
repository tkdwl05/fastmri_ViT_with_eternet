#!/bin/bash
# ============================================================================
#  [4/6] 호스트에서 실행 — 기존 컨테이너 정지(보존) + 새 컨테이너 기동
#
#  ⚠ 이 단계에서 현재 Claude Code 세션은 끊어진다 (세션이 옛 컨테이너 안에서
#    돌고 있으므로). 00_backup_state.sh 를 먼저 돌렸는지 반드시 확인할 것.
#
#  ⚠ 기존 컨테이너는 **삭제하지 않는다**. stop 만 한다 → 언제든 롤백 가능.
#
#  실행:  bash 30_host_run.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "${HERE}/container.env" ] || { echo "✗ container.env 없음 — 10_host_preflight.sh 를 먼저 실행"; exit 1; }
# shellcheck disable=SC1090
source "${HERE}/container.env"

echo "════════ 실행 계획 ════════"
echo "  정지(보존) : ${OLD_CTN}"
echo "  신규 생성  : ${NEW_CTN}  (hostname ${NEW_HOSTNAME})"
echo "  이미지     : ${IMAGE}"
echo "  마운트     : ${MOUNTS}"
echo "  포트       : ${PORTS:-(없음)}"
echo "  네트워크   : ${NETMODE:-default}"
echo "  GPU 노출    : ${NVIDIA_VISIBLE:-0}"
echo "  GPU 디바이스: ${NV_DEVICES}"
echo "  shm/mem    : ${SHM_SIZE} / ${MEM_LIMIT}"
echo

docker image inspect "${IMAGE}" >/dev/null 2>&1 || { echo "✗ 이미지 ${IMAGE} 없음 — 20_host_build.sh 먼저"; exit 1; }
if docker inspect "${NEW_CTN}" >/dev/null 2>&1; then
  echo "✗ '${NEW_CTN}' 이름이 이미 존재. docker rm ${NEW_CTN} 후 재실행하거나 NEW_CTN= 을 바꿀 것."; exit 1
fi

# 백업 확인 게이트 -----------------------------------------------------------
BK_ROOT=/home/snorlax/shared/container_state_backup
if [ -d "$BK_ROOT" ] && [ -n "$(ls -A "$BK_ROOT" 2>/dev/null)" ]; then
  echo "  ✓ 상태 백업 발견: $(ls -1t "$BK_ROOT" | head -1)"
else
  echo "  ⚠ ${BK_ROOT} 에 백업이 없다!"
  echo "    옛 컨테이너 안에서 먼저:  bash infra/docker/00_backup_state.sh"
  echo "    (ssh 키·wandb 토큰·Claude 메모리가 옛 컨테이너 레이어에만 있음)"
fi

# 학습 프로세스 확인 ---------------------------------------------------------
RUNNING_PY=$(docker top "${OLD_CTN}" 2>/dev/null | grep -cE '[p]ython' || true)
if [ "${RUNNING_PY:-0}" -gt 0 ]; then
  echo "  ⚠ 옛 컨테이너에서 python 프로세스 ${RUNNING_PY}개 실행 중 — stop 시 함께 종료됨."
  echo "    (v9 트레이너는 true-resume 이라 손실은 마지막 epoch 저장 이후 구간뿐)"
fi

echo
read -r -p "  진행하려면 YES 입력: " ans
[ "$ans" = "YES" ] || { echo "  취소됨. 아무것도 바뀌지 않았다."; exit 0; }

echo "── 기존 컨테이너 정지 (삭제 아님)"
docker stop "${OLD_CTN}" || true

echo "── 새 컨테이너 기동"
# 네트워크 모드에 따라 인자 구성 — host 모드에서는 --hostname/-p 를 쓸 수 없다.
NET_ARGS=""; HOSTNAME_ARG=""; PORT_ARGS="${PORTS}"
case "${NETMODE:-default}" in
  host)     NET_ARGS="--network host"; PORT_ARGS=""
            echo "  ⓘ host 네트워크 모드 → --hostname/-p 생략 (도커 제약)" ;;
  default|bridge|"")  HOSTNAME_ARG="--hostname ${NEW_HOSTNAME}" ;;
  container:*)  echo "✗ NetworkMode=${NETMODE} (다른 컨테이너 공유) — 수동 확인 필요"; exit 1 ;;
  *)        NET_ARGS="--network ${NETMODE}"; HOSTNAME_ARG="--hostname ${NEW_HOSTNAME}" ;;
esac

# shellcheck disable=SC2086
docker run -d \
  --name "${NEW_CTN}" \
  ${HOSTNAME_ARG} ${NET_ARGS} \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE:-0}" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  ${NV_DEVICES} \
  --shm-size="${SHM_SIZE}" \
  --memory="${MEM_LIMIT}" --memory-swap="${MEM_LIMIT}" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart unless-stopped \
  ${PORT_ARGS} ${MOUNTS} \
  -w "${WORKDIR}" \
  "${IMAGE}" sleep infinity

echo
echo "── 즉시 검증"
sleep 3
echo "  [GPU — TITAN RTX 정확히 1개만 보이면 정상. 2개면 즉시 중단할 것]"
docker exec "${NEW_CTN}" nvidia-smi --query-gpu=index,name,uuid --format=csv || echo "  ✗ nvidia-smi 실패 — docker logs ${NEW_CTN} 확인"
echo "  [torch]"
docker exec "${NEW_CTN}" python -c "import torch;print('  cuda:',torch.cuda.is_available(),'count:',torch.cuda.device_count())"
echo "  [shm — ${SHM_SIZE}]"; docker exec "${NEW_CTN}" df -h /dev/shm | tail -1 | sed 's/^/  /'
echo "  [memory.max]"; docker exec "${NEW_CTN}" cat /sys/fs/cgroup/memory.max 2>/dev/null | sed 's/^/  /'
echo "  [hostname]"; docker exec "${NEW_CTN}" hostname | sed 's/^/  /'
echo "  [마운트 확인]"; docker exec "${NEW_CTN}" ls /home/snorlax/shared/fastmri_ViT_with_eternet >/dev/null && echo "  ✓ repo 보임"

echo
echo "══ 완료. 접속:  docker exec -it ${NEW_CTN} bash"
echo "  다음 단계 (새 컨테이너 안에서):"
echo "    bash infra/docker/40_restore_state.sh"
echo "    bash infra/docker/50_verify_env.sh"
echo
echo "  롤백이 필요하면:"
echo "    docker stop ${NEW_CTN} && docker start ${OLD_CTN}"
