#!/bin/bash
# RunPod Pod → GitHub 결과 동기화 (2026-10-09). 작은 텍스트 결과만 runpod_results/ 에 복사해 runpod/* 브랜치로 push 한다.
#   - 대상: logs/<RUN>/{log.txt, recipe.json, probe_ep*.json, PAUSED_ep*, DONE, ALERT_COLLAPSE},
#           results/eval/v8fix*/{summary_*.md, functional_*.json}, 학습 tqdm 로그의 마지막 진행 줄(progress.txt).
#   - 체크포인트(*.pt)·데이터·대용량 로그는 올리지 않는다.
#   - 코드는 pull 하지 않는다: 실행 중인 런의 코드를 바꾸지 않기 위해(CLAUDE.md 작업 규칙). 코드 갱신은 런 사이에 수동으로
#     `git fetch origin && git merge origin/<작업 브랜치>`.
# 실행 (Pod): nohup setsid bash infra/runpod/pod_git_sync.sh > /root/_setup/git_sync.log 2>&1 < /dev/null &
#   env: SYNC_BRANCH(기본 runpod/v8fix) SYNC_INTERVAL(초, 기본 1800) ONCE=1(한 번만)
set -u
REPO=${REPO:-/root/fastmri_ViT_with_eternet}
BR=${SYNC_BRANCH:-runpod/v8fix}
INTERVAL=${SYNC_INTERVAL:-1800}
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
cd "$REPO" || exit 1
while :; do
  D=$REPO/runpod_results
  mkdir -p "$D/runs" "$D/eval"
  for r in "$REPO"/logs/*/; do
    [ -d "$r" ] || continue
    n=$(basename "$r"); mkdir -p "$D/runs/$n"
    for f in log.txt recipe.json probe_ep*.json PAUSED_ep* DONE ALERT_COLLAPSE; do
      for g in "$r"/$f; do [ -e "$g" ] && cp -p "$g" "$D/runs/$n/"; done
    done
  done
  for e in "$REPO"/results/eval/v8fix*/; do
    [ -d "$e" ] || continue
    n=$(basename "$e"); mkdir -p "$D/eval/$n"
    cp -p "$e"/summary_*.md "$e"/functional_*.json "$D/eval/$n/" 2>/dev/null
  done
  for t in "$REPO"/v8_eter_pure/runs/runpod/*.log; do
    [ -e "$t" ] || continue
    echo "$(basename "$t"): $(tail -c 2000 "$t" | tr '\r' '\n' | grep -a 'Epoch\|Val' | tail -1)"
  done > "$D/progress.txt" 2>/dev/null
  git add runpod_results
  if ! git diff --cached --quiet; then
    git commit -q -m "runpod: 결과 동기화 $(date -u '+%F %H:%M') UTC

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HmxKEwWc2RtRpbAfKQyNKC"
    git push -q origin "HEAD:$BR" && echo "$(date -u +%T) pushed" || echo "$(date -u +%T) push 실패"
  fi
  [ "${ONCE:-0}" = 1 ] && break
  sleep "$INTERVAL"
done
