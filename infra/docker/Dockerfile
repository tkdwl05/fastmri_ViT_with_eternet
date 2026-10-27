# ============================================================================
#  fastmri_ViT_with_eternet — 재현 가능한 학습 환경 (2026-08-31 신규 작성)
#
#  설계 원칙
#   1) 공식 베이스 이미지에서 출발한다. 기존 운영 이미지는 `docker commit` 스냅샷
#      체인(snorlax_gpu0:v2_bigmem_snapshot ...)이라 Dockerfile 이 없고, 무엇이
#      언제 왜 들어갔는지 감사할 수 없었다. 이 파일이 그 대체물이다.
#   2) torch 2.3.1 / CUDA 12.1 / cuDNN 8 / python 3.10 조합은 **고정**한다.
#      mamba_ssm·causal_conv1d 는 prebuilt wheel(cu122torch2.3cxx11abiFALSE-cp310)
#      을 쓰므로 torch 를 올리면 SS2D CUDA 커널이 깨지고, 완주한 v7/v8/v9 런과의
#      수치 비교가 무효가 된다. 연구 연속성상 업그레이드 금지.
#   3) 비밀정보(ssh 키·wandb 토큰)는 이미지에 굽지 않는다. 40_restore_state.sh 로
#      런타임에 복원한다.
#   4) 데이터·체크포인트(logs/ 128G)는 이미지가 아니라 바인드 마운트에 있다.
# ============================================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1

# ---------------------------------------------------------------------------
# OS 패키지 — 기존 컨테이너의 apt history 에서 확인된 것만 그대로 복원
#   2025-11-26: sudo openssh-server nano git x11-apps
#   2026-05-20: rsync        2026-05-21: tmux
#   + fonts-nanum: 기존 환경에 CJK 폰트가 없어 matplotlib 제목 한글이 □□ 로
#     렌더링됐다(commit 3bcc88b 참조). 폰트만 깔아두면 rcParams 기본값은
#     그대로이므로 기존 그림 스타일에 영향 없다.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo openssh-server nano vim git rsync tmux less htop procps \
        curl wget ca-certificates gnupg apt-transport-https \
        x11-apps fonts-nanum fontconfig \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Node.js 20 (Claude Code CLI 실행에 필요) — NodeSource
# ---------------------------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node -v && npm -v

# ---------------------------------------------------------------------------
# Python 패키지 — 현재 운영 환경 pip freeze 에서 역산한 "베이스 대비 델타"
#   constraints.txt 가 numpy 를 1.26.4 에 묶는다(numpy 2.x 로 올라가면 torch
#   2.3.1 / mamba_ssm 바이너리와 ABI 충돌).
# ---------------------------------------------------------------------------
COPY requirements.txt constraints.txt /tmp/build/
RUN pip install --upgrade pip==24.0 \
    && pip install -r /tmp/build/requirements.txt -c /tmp/build/constraints.txt \
    && pip install --no-deps fastmri==0.3.0 \
    && rm -rf /tmp/build

# fastmri 는 --no-deps 로 넣는다: 상위 의존성(pytorch-lightning·torchmetrics·
# runstats)은 기존 환경에도 설치돼 있지 않고, 이 프로젝트는 fastmri 의 순수
# 함수(fft/coil combine/mask)만 쓴다. 의존성을 끌어오면 torch 를 재설치할 위험.

# ---------------------------------------------------------------------------
# Claude Code CLI — 새 컨테이너에서 별도 설치 없이 바로 쓰도록 이미지에 굽는다
#   (08-31: 첫 mri:v1 컨테이너에서 claude 가 없어 수동 설치해야 했던 것 반영).
#   · npm 전역 설치 → /usr/bin/claude 로 잡혀 PATH 문제 없음
#   · PATH 맨 앞의 /root/.local/bin: 40_restore_state.sh 가 구 컨테이너의 CLI 본체
#     (/root/.local, 855M)를 복원하면 그 버전이 npm 판보다 우선하도록. 네이티브
#     인스톨러(claude.ai/install.sh)로 재설치할 때도 같은 경로라 바로 잡힌다.
#   · 로그인·프로젝트 메모리는 이미지가 아니라 claude_state.tgz(.claude,
#     .claude.json) 복원으로 승계된다 (비밀정보는 이미지에 굽지 않는 원칙 유지).
# ---------------------------------------------------------------------------
ENV PATH="/root/.local/bin:${PATH}"
RUN npm install -g @anthropic-ai/claude-code \
    && claude --version

# ---------------------------------------------------------------------------
# sshd 준비 (기존 컨테이너에서 sshd 가 떠 있었으므로 동등하게 유지)
#   ※ root 로그인 키/비밀번호는 이미지에 넣지 않는다. 필요하면 컨테이너 안에서
#     /root/.ssh/authorized_keys 를 채우거나 passwd 로 설정할 것.
# ---------------------------------------------------------------------------
RUN mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /home/snorlax/shared/fastmri_ViT_with_eternet
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
