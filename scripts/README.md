# scripts/ — 멀티에이전트 헤드리스 비교 러너 (2026-08-27)

같은 base 커밋에서 에이전트별 git worktree 를 만들어 동일 프롬프트를 병렬 실행하고,
결과 브랜치를 diff 로 비교하기 위한 도구. `.gitignore` 의 `*.sh` 전역 무시에
`!scripts/*.sh` 예외를 걸어 이 폴더의 스크립트만 git 추적한다 (worktree 전파 목적).

## 구성

| 파일 | 역할 |
|---|---|
| `run.sh` | 러너 본체. worktree 생성 → prepare 훅 → 격리 환경(`env -i` 화이트리스트)에서 에이전트 병렬 실행 → 로그/종료코드 수집 |
| `prepare-worktree.sh` | worktree 생성 직후 훅 — git 이 안 따라가는 untracked 필수 파일(.env 류)을 채움. 현재 이 저장소엔 해당 파일 없음(no-op) |

## 지시 파일 정책

- **`AGENTS.md` 가 정본** (Codex·OpenCode 가 읽음), `CLAUDE.md` 는 그 심링크 (Claude Code 가 읽음).
- 둘 다 커밋되어 있으므로 새 worktree 에 자동으로 따라 들어간다. **지시 수정은 반드시 AGENTS.md 에서.**

## 실행 설정 정책

- 실행 설정(권한·샌드박스·출력형식)은 **전부 `run.sh` 안의 CLI 플래그로 명시** — 스크립트가 곧 실험 기록.
- `~/.codex/config.toml`, `.claude/settings.json` 은 실험용으로 건드리지 않는다 (조건이 조용히 달라지는 것 방지).
- 현재 플래그 (2026-08-27, claude 2.1.247 / codex 0.150.1 기준):
  - Claude Code: `claude -p "$PROMPT" --permission-mode acceptEdits --output-format json`
  - Codex: `codex exec --sandbox workspace-write --color never --json -o <final.md> "$PROMPT"`
    (주의: `codex exec` 에는 `--full-auto` 플래그가 **없다** — 대화형 `codex` 전용. exec 는 원래 비대화형이라 승인 프롬프트 자체가 없고 sandbox 모드가 유일한 안전장치)
  - OpenCode: `opencode run "$PROMPT"` — 권한은 `.opencode/agent/*.md` frontmatter 또는 `opencode.json`
    (이 머신엔 opencode **미설치** — 러너가 자동 skip. 설치: `brew install sst/tap/opencode` 후 `opencode auth login`)

## 사용법

```bash
# 1) 프롬프트 작성 — 반드시 저장소 밖 (에이전트가 수정 못 하게)
vim ~/runs/task-002/PROMPT.md        # 템플릿: ~/runs/task-001/PROMPT.md

# 2) 실행
scripts/run.sh task-002                    # 설치된 에이전트 전부 1회씩 (도구 비교)
scripts/run.sh task-002 claude codex       # 명시 선택
scripts/run.sh task-002 claude:3           # 같은 에이전트 3회 (샘플링 분산 실험)

# 3) 비교
git range-diff try/task-002-claude...try/task-002-codex
diff <(git -C ../wt-task-002-claude diff) <(git -C ../wt-task-002-codex diff)

# 4) 정리
git worktree remove ../wt-task-002-claude --force
git branch -D try/task-002-claude
```

산출물: worktree `../wt-<task>-<agent>` / 브랜치 `try/<task>-<agent>` /
로그 `~/runs/<task>/logs/` (`<agent>.log`, `exits.txt`, `meta.txt` — base 커밋·CLI 버전 기록).

## 실험 설계 메모

- **도구 비교** (claude vs codex vs oc)와 **분산 측정** (`claude:3`)은 다른 실험이다.
  같은 모델 n회가 서로 다른 답을 내면 태스크 명세(PROMPT.md)가 모호하다는 신호.
- 이 저장소의 Mac clone 은 학습 실행 불가 (GPU/데이터/ckpt 없음 — AGENTS.md §환경) →
  비교 태스크는 문서·논문·표 스크립트·시각화 코드 수정 같은 **비학습 작업**으로 한정할 것.
