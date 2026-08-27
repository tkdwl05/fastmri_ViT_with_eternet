#!/usr/bin/env bash
# 새 worktree 에 "git 이 안 따라가는" 필수 파일을 채워넣는 준비 훅.
# run.sh 가 worktree 생성 직후 자동 호출한다. 단독 실행도 가능:
#   scripts/prepare-worktree.sh <worktree-path>
#
# 이 저장소는 Python 연구 저장소라 현재 복사할 untracked 필수 파일이 없다.
# (.env 류가 생기면 아래 목록이 자동으로 처리한다. node_modules 같은 무거운
# 의존성 폴더가 생기면 맨 아래 심링크 블록의 주석을 풀어서 쓴다.)
set -euo pipefail

wt="${1:?usage: prepare-worktree.sh <worktree-path>}"
src="$(cd "$(dirname "$0")/.." && pwd)"

# gitignore 된 로컬 설정 파일 복사
for f in .env .env.local; do
  if [ -f "$src/$f" ]; then
    cp "$src/$f" "$wt/"
    echo "prepare-worktree: copied $f"
  fi
done

# 무거운 의존성 폴더는 복사 대신 심링크 (필요해지면 주석 해제)
# ln -sfn "$src/node_modules" "$wt/node_modules"
