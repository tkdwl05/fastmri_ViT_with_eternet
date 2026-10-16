#!/bin/bash
# 로컬 서버: RunPod Pod 의 학습 결과·체크포인트를 주기적으로 받아 온다 (2026-10-09).
#   Pod 의 컨테이너 디스크는 Pod 가 멈추거나 지워지면 사라진다(잔액 소진 포함). 그때 새 Pod 에서 전체 상태 재개를 하려면
#   last.pt 가 필요하므로 6 시간마다 끌어온다. 작은 텍스트 결과는 pod_git_sync.sh 가 git 으로 따로 올린다.
#   대상: logs/*_fullval/ 의 log.txt·*.json·표시 파일·pure_*_{best,last,epoch_N}.pt, results/eval/v8fix_*fullval*/
#   저장: logs/runpod_mirror/ (git 무시). 전송은 rsync --partial(끊기면 이어 받기).
# 실행: POD_HOST=<pod IP> POD_PORT=<ssh 포트> setsid nohup bash infra/runpod/local_pull_from_pod.sh > logs/runpod_mirror/pull.log 2>&1 < /dev/null & disown
#   Pod 를 새로 만들면 IP·포트가 바뀐다 → 이 프로세스를 끄고 새 값으로 다시 띄울 것.
set -u
ROOT=/home/snorlax/shared/fastmri_ViT_with_eternet
H="${POD_HOST:?POD_HOST}"; P="${POD_PORT:?POD_PORT}"
INTERVAL="${PULL_INTERVAL:-21600}"
DST="$ROOT/logs/runpod_mirror"; mkdir -p "$DST/logs" "$DST/results"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -i /root/.ssh/id_ed25519_runpod -p $P"
while :; do
  echo "$(date -u +%F' '%T) pull 시작"
  nice -n 19 rsync -a --partial --timeout=600 -e "$SSH" \
    --include='*_fullval/' --include='*_fullval/log.txt' --include='*_fullval/*.json' --include='*_fullval/PAUSED_ep*' \
    --include='*_fullval/DONE' --include='*_fullval/ALERT_COLLAPSE' --include='*_fullval/pure_*.pt' --exclude='*' \
    "root@$H:/root/fastmri_ViT_with_eternet/logs/" "$DST/logs/" && echo "$(date -u +%T) logs ok" || echo "$(date -u +%T) logs 실패"
  nice -n 19 rsync -a --partial --timeout=600 -e "$SSH" --include='v8fix_*fullval*/' --include='v8fix_*fullval*/**' --exclude='*' \
    "root@$H:/root/fastmri_ViT_with_eternet/results/eval/" "$DST/results/" 2>/dev/null && echo "$(date -u +%T) results ok" || true
  du -sh "$DST"
  sleep "$INTERVAL"
done
