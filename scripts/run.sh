#!/usr/bin/env bash
# 멀티에이전트 헤드리스 비교 러너.
# 같은 base 커밋에서 에이전트별 worktree 를 만들어 동일 프롬프트를 병렬 실행한다.
# 실행 설정은 전부 이 스크립트의 플래그에 명시한다 — 스크립트 자체가 실험 기록이다.
# (~/.codex/config.toml, .claude/settings.json 은 건드리지 않는다.)
#
# 사용법:
#   scripts/run.sh <task-id> [agent ...]
#     agent  : claude | codex | oc  (생략하면 설치된 에이전트 전부 1회씩)
#     agent:N: 같은 에이전트 N회 반복 — 샘플링 분산 실험 (예: claude:3)
#
# 전제:
#   ~/runs/<task-id>/PROMPT.md 가 있어야 한다. 저장소 밖에 두는 이유는
#   에이전트가 프롬프트 파일 자체를 수정하지 못하게 하기 위함이다.
#
# 산출물:
#   worktree  : ../wt-<task-id>-<agent>   브랜치: try/<task-id>-<agent>
#   로그      : ~/runs/<task-id>/logs/<agent>.log (+ exits.txt, meta.txt)
#
# 비교:
#   git range-diff try/<task>-claude...try/<task>-codex
#   diff <(git diff BASE try/<task>-claude) <(git diff BASE try/<task>-codex)
#
# 정리:
#   git worktree remove ../wt-<task>-<agent> && git branch -D try/<task>-<agent>
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TASK="${1:?usage: $0 <task-id> [claude|codex|oc][:N] ...}"
shift || true
RUN="$HOME/runs/$TASK"
PROMPT_FILE="$RUN/PROMPT.md"

[ -f "$PROMPT_FILE" ] || { echo "오류: $PROMPT_FILE 이 없다. 프롬프트를 먼저 작성할 것." >&2; exit 1; }
PROMPT="$(cat "$PROMPT_FILE")"
BASE="$(git -C "$REPO" rev-parse HEAD)"

mkdir -p "$RUN/logs"
: > "$RUN/logs/exits.txt"
{
  echo "task=$TASK"
  echo "base=$BASE"
  echo "date=$(date -u +%FT%TZ)"
  echo "claude=$(claude --version 2>/dev/null || echo absent)"
  echo "codex=$(codex --version 2>/dev/null || echo absent)"
  echo "opencode=$(opencode --version 2>/dev/null || echo absent)"
} > "$RUN/logs/meta.txt"

# 셸 상속 변수를 끊고 화이트리스트만 전달 — 에이전트 간 환경 차이를 변수에서 제거.
# TERM=dumb / NO_COLOR=1 은 로그 diff 시 ANSI 노이즈 제거용.
run_isolated() {
  env -i \
    HOME="$HOME" PATH="$PATH" TERM=dumb \
    LANG=C.UTF-8 NO_COLOR=1 CI=1 \
    "$@"
}

launch() {  # $1=이름  $2...=명령
  local n="$1"; shift
  local wt="$REPO/../wt-$TASK-$n"
  local br="try/$TASK-$n"
  git -C "$REPO" worktree add "$wt" -b "$br" "$BASE"
  "$REPO/scripts/prepare-worktree.sh" "$wt"
  (
    cd "$wt"
    rc=0
    run_isolated "$@" >"$RUN/logs/$n.log" 2>&1 || rc=$?
    echo "$n exit=$rc" >> "$RUN/logs/exits.txt"
  ) &
}

launch_agent() {  # $1=에이전트종류  $2=런이름(라벨)
  local kind="$1" id="$2"
  case "$kind" in
    claude)
      # --output-format json: 파싱 가능한 최종 결과. 승인은 acceptEdits (파일 편집 자동승인).
      launch "$id" claude -p "$PROMPT" \
        --permission-mode acceptEdits --output-format json
      ;;
    codex)
      # exec 는 비대화형이라 승인 프롬프트가 없다 — sandbox 가 유일한 안전장치.
      # --json: JSONL 이벤트 스트림(로그로), -o: 최종 메시지만 별도 파일.
      launch "$id" codex exec \
        --sandbox workspace-write --color never --json \
        -o "$RUN/logs/$id.final.md" \
        "$PROMPT"
      ;;
    oc)
      # 권한은 .opencode/agent/ frontmatter 또는 opencode.json 의 permissions 를 따른다.
      launch "$id" opencode run "$PROMPT"
      ;;
    *)
      echo "오류: 알 수 없는 에이전트 '$kind' (claude|codex|oc)" >&2; exit 1
      ;;
  esac
}

# 에이전트 목록 결정: 인자 없으면 설치된 것 전부 1회씩
SPECS=("$@")
if [ ${#SPECS[@]} -eq 0 ]; then
  SPECS=()
  command -v claude   >/dev/null && SPECS+=(claude)
  command -v codex    >/dev/null && SPECS+=(codex)
  command -v opencode >/dev/null && SPECS+=(oc)
  [ ${#SPECS[@]} -gt 0 ] || { echo "오류: 설치된 에이전트 CLI 가 없다." >&2; exit 1; }
fi

for spec in "${SPECS[@]}"; do
  kind="${spec%%:*}"
  count="${spec#*:}"; [ "$count" = "$spec" ] && count=1
  case "$kind" in
    claude) command -v claude   >/dev/null || { echo "skip: claude 미설치" | tee -a "$RUN/logs/exits.txt"; continue; } ;;
    codex)  command -v codex    >/dev/null || { echo "skip: codex 미설치"  | tee -a "$RUN/logs/exits.txt"; continue; } ;;
    oc)     command -v opencode >/dev/null || { echo "skip: opencode 미설치" | tee -a "$RUN/logs/exits.txt"; continue; } ;;
  esac
  if [ "$count" -eq 1 ]; then
    launch_agent "$kind" "$kind"
  else
    for i in $(seq 1 "$count"); do
      launch_agent "$kind" "$kind-$i"
    done
  fi
done

wait
echo "=== 완료 ==="
cat "$RUN/logs/exits.txt"
echo
echo "비교: git range-diff try/$TASK-claude...try/$TASK-codex"
