#!/bin/bash
# 8/8 정전 대비 — 8/7 밤 radapt 깨끗한 정지 (사용자 요청 2026-08-05).
# 분리 타이머가 8/7 21:00 KST(12:00 UTC)에 호출. 수동 실행 시 즉시 경계-대기 시작.
#
# 동작:
#   1) last.pt 가 새로 저장되는 epoch 경계를 감시 (저장 순서상 last.pt 가 epoch 말 마지막 저장물
#      → mtime 변경 = 그 epoch 의 모든 ckpt 저장 완료. 이후 정지하면 진행 손실 ~0)
#   2) 하드 데드라인 23:55 KST — 경계 미감지여도 정지 (부분 epoch 만 resume 시 재학습, ≤1ep)
#   3) supervisor SIGTERM → trainer SIGTERM → (20s) 잔여 SIGKILL → sync
#   4) 사전보고 스냅샷 최종 갱신 (정지 후 상태 반영)
#
# DRY_RUN=1 bash clean_stop_pre_outage.sh  → kill/sync/스냅샷 생략하고 감시 로직만 검증
set -u
ROOT=/home/snorlax/shared/fastmri_ViT_with_eternet
CKPT="${CKPT_OVERRIDE:-$ROOT/logs/PureETER_SS2D_V9_radapt_multiAR_brain384/ss2d_v9_radapt_last.pt}"
STOPLOG=$ROOT/v9_mamba_radapt/runs/clean_stop_2026-08-07.log
DEADLINE="${DEADLINE_OVERRIDE:-1786114500}"   # 2026-08-07 14:55 UTC = 23:55 KST
DRY_RUN="${DRY_RUN:-0}"

log(){ echo "[clean-stop] $(date -u '+%m-%d %H:%M:%S UTC') $*" | tee -a "$STOPLOG"; }

log "epoch 경계 감시 시작 (deadline $(date -u -d @$DEADLINE '+%H:%M UTC') = KST $(date -u -d @$((DEADLINE+32400)) '+%H:%M'), DRY_RUN=$DRY_RUN)"
M0=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
BOUNDARY=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  M1=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
  if [ "$M1" != "$M0" ] && [ "$M1" != 0 ]; then
    sleep 30   # 저장(atomic rename) 직후 안전 마진
    log "epoch 경계 감지 (last.pt 갱신) → 정지 진행"
    BOUNDARY=1
    break
  fi
  sleep 60
done
[ "$BOUNDARY" = 0 ] && log "deadline 도달 — 경계 미감지 상태로 정지 (진행 중이던 epoch 은 resume 시 재학습)"

if [ "$DRY_RUN" = 1 ]; then
  log "DRY_RUN — kill/sync/스냅샷 생략, 종료"
  exit 0
fi

pkill -TERM -f run_ss2d_v9_radapt_autoresume.sh 2>/dev/null && log "supervisor SIGTERM" || log "supervisor 없음"
sleep 3
pkill -TERM -f main_train_ss2d_v9_radapt.py 2>/dev/null && log "trainer SIGTERM" || log "trainer 없음"
sleep 20
pkill -KILL -f main_train_ss2d_v9_radapt.py 2>/dev/null && log "잔여 trainer SIGKILL"
sleep 2
sync
LEFT=$(pgrep -cf 'main_train_ss2d_v9_radapt' 2>/dev/null || echo 0)
log "sync 완료. 잔존 trainer 프로세스: ${LEFT}개 (0 이어야 정상)"
bash "$ROOT/v9_mamba_radapt/runs/snapshot_pre_outage.sh" >> "$STOPLOG" 2>&1
log "✅ 깨끗한 정지 완료 + 최종 보고 갱신(pre_outage_report_2026-08-07.md). 재개 = RESUME_AFTER_OUTAGE.md"
