#!/bin/bash
# 컨테이너 진입점: sshd 기동(선택) 후 전달된 커맨드 실행.
set -e

if [ "${START_SSHD:-1}" = "1" ]; then
    mkdir -p /run/sshd
    # 호스트키가 없으면 최초 1회 생성 (컨테이너 레이어에 생성됨)
    ssh-keygen -A >/dev/null 2>&1 || true
    /usr/sbin/sshd 2>/dev/null || echo "[entrypoint] sshd 기동 실패 — 무시하고 계속" >&2
fi

exec "$@"
