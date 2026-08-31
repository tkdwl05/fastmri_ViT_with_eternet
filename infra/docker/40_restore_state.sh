#!/bin/bash
# ============================================================================
#  [5/6] 새 컨테이너 안에서 실행 — 인증·Claude 상태 복원
#
#  실행:  docker exec -it <NEW_CTN> bash
#         bash infra/docker/40_restore_state.sh
# ============================================================================
set -euo pipefail

BK_ROOT="${BK_ROOT:-/home/snorlax/shared/container_state_backup}"
SRC="${SRC:-$(ls -1dt ${BK_ROOT}/*/ 2>/dev/null | head -1)}"
[ -n "${SRC:-}" ] && [ -d "$SRC" ] || { echo "✗ 백업을 찾을 수 없음: ${BK_ROOT}"; exit 1; }
echo "── 복원 원본: ${SRC}"

restore() {   # $1 = tarball 이름
  local t="${SRC}/$1"
  if [ -f "$t" ]; then echo "   · $1"; tar -C /root -xzf "$t"; else echo "   · $1 (없음, 건너뜀)"; fi
}

echo "── [1/3] 인증·설정"
restore root_secrets.tgz
chmod 700 /root/.ssh 2>/dev/null || true
chmod 600 /root/.ssh/id_ed25519 /root/.netrc 2>/dev/null || true

echo "── [2/3] Claude Code 상태"
restore claude_state.tgz

echo "── [3/3] Claude Code CLI"
if [ -x /root/.local/bin/claude ]; then
  echo "   · 이미 설치돼 있음 — 건너뜀"
else
  restore root_local.tgz
fi
grep -q '.local/bin' /root/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc
export PATH="/root/.local/bin:$PATH"

echo
echo "══ 복원 확인"
printf "  ssh key      : %s\n" "$([ -f /root/.ssh/id_ed25519 ] && echo '있음' || echo '없음')"
printf "  wandb .netrc : %s\n" "$([ -f /root/.netrc ] && echo '있음' || echo '없음')"
printf "  claude memory: %s\n" "$([ -d /root/.claude ] && du -sh /root/.claude | cut -f1 || echo '없음')"
printf "  claude CLI   : %s\n" "$(command -v claude || echo '없음 — 재설치 필요')"
echo
echo "  GitHub SSH 확인:"; ssh -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | sed 's/^/    /' || true
echo
echo "  claude CLI 가 없으면 둘 중 하나로 재설치:"
echo "    curl -fsSL https://claude.ai/install.sh | bash"
echo "    npm install -g @anthropic-ai/claude-code"
echo
echo "  다음 단계: bash infra/docker/50_verify_env.sh"
