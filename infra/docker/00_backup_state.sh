#!/bin/bash
# ============================================================================
#  [1/6] 현재 컨테이너 안에서 실행 — 컨테이너 레이어에만 있는 상태를 백업
#
#  왜 필요한가: 아래 항목들은 바인드 마운트가 아니라 컨테이너 쓰기 레이어에
#  있어, 컨테이너를 버리면 함께 사라진다.
#    /root/.ssh        GitHub 배포 키 (SSH 인증 — PAT 만료 후 유일 수단)
#    /root/.netrc      wandb API 키
#    /root/.claude     Claude Code 프로젝트 메모리·세션 기록 (36M)
#    /root/.local      Claude Code CLI 본체 (855M, 오프라인 복원용)
#  반대로 아래는 안전하다(백업 불필요):
#    /home/snorlax/shared/**  = NVMe 바인드 마운트 (repo + logs/ 128G ckpt)
#    /mnt/sda/**              = HDD 바인드 마운트
#
#  실행:  bash infra/docker/00_backup_state.sh
# ============================================================================
set -euo pipefail

DEST_ROOT="${DEST_ROOT:-/home/snorlax/shared/container_state_backup}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="${DEST_ROOT}/${STAMP}"

echo "── 백업 대상 : ${DEST}"
mkdir -p "$DEST"
chmod 700 "$DEST_ROOT" "$DEST"      # 비밀정보 포함 — 소유자만

# --- 1. 비밀정보 / 설정 (작음) ---------------------------------------------
echo "── [1/4] 인증·설정"
tar -C /root -czf "${DEST}/root_secrets.tgz" \
    $( for f in .ssh .netrc .gitconfig .git-credentials .bashrc .bash_history .config; do
         [ -e "/root/$f" ] && echo "$f"; done ) 2>/dev/null
chmod 600 "${DEST}/root_secrets.tgz"
echo "   → root_secrets.tgz ($(du -h "${DEST}/root_secrets.tgz" | cut -f1))"

# --- 2. Claude Code 상태 (메모리·기록) -------------------------------------
echo "── [2/4] Claude Code 상태 (.claude, .claude.json)"
tar -C /root -czf "${DEST}/claude_state.tgz" .claude .claude.json 2>/dev/null || true
echo "   → claude_state.tgz ($(du -h "${DEST}/claude_state.tgz" | cut -f1))"

# --- 3. Claude Code CLI 본체 (네트워크 없이 복원하기 위해) ------------------
echo "── [3/4] Claude Code CLI (.local — 수 분 소요)"
tar -C /root -czf "${DEST}/root_local.tgz" .local 2>/dev/null || true
echo "   → root_local.tgz ($(du -h "${DEST}/root_local.tgz" | cut -f1))"

# --- 4. 환경 증거물 (새 이미지 대조용) --------------------------------------
echo "── [4/4] 환경 스냅샷 기록"
{
  echo "# 백업 시각: $(date -Is)"
  echo "# hostname : $(hostname)"
  echo "# kernel   : $(uname -r)"
  echo "# os       : $(grep PRETTY /etc/os-release)"
} > "${DEST}/ENVIRONMENT.txt"
pip freeze                       > "${DEST}/pip_freeze.txt"      2>/dev/null || true
dpkg -l                          > "${DEST}/dpkg_list.txt"       2>/dev/null || true
env | sort                       > "${DEST}/env.txt"             2>/dev/null || true
cp /var/log/apt/history.log        "${DEST}/apt_history.log"     2>/dev/null || true

echo
echo "══ 백업 완료: ${DEST}"
du -sh "$DEST"
echo
echo "  ⚠ 이 디렉토리에는 SSH 개인키와 wandb 토큰이 들어 있다."
echo "    git 저장소 밖(${DEST_ROOT})에 두었으니 커밋되지 않는다. 그대로 둘 것."
echo
echo "  다음 단계: 호스트에서  bash infra/docker/10_host_preflight.sh"
