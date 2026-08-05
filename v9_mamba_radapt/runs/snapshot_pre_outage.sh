#!/bin/bash
# 2026-08-08 학교 정전 대비 — radapt 진행상황 사전 보고 스냅샷.
# 8/7 저녁(20:54 KST) 분리 타이머가 자동 실행하며, 수동 실행도 안전(멱등, 재실행 시 보고 갱신):
#   bash v9_mamba_radapt/runs/snapshot_pre_outage.sh
# 출력: v9_mamba_radapt/runs/pre_outage_report_2026-08-07.md
set -u
ROOT=/home/snorlax/shared/fastmri_ViT_with_eternet
RUN=PureETER_SS2D_V9_radapt_multiAR_brain384
LOGDIR="$ROOT/logs/$RUN"
RUNLOG="$ROOT/v9_mamba_radapt/runs/ss2d/run_${RUN}.log"
OUT="$ROOT/v9_mamba_radapt/runs/pre_outage_report_2026-08-07.md"

LAST_EP=$(grep -c '^Epoch' "$LOGDIR/log.txt" 2>/dev/null || echo 0)
BEST=$(grep -o 'val_composite=[0-9.]*' "$LOGDIR/log.txt" 2>/dev/null | cut -d= -f2 | sort -g | tail -1)
REMAIN=$((80 - LAST_EP))
REMAIN_H=$(( REMAIN * 11 / 4 ))   # ~2.75 h/ep

{
echo "# radapt 정전 전 상태 보고 (자동 스냅샷)"
echo
echo "- 생성: $(date -u '+%Y-%m-%d %H:%M UTC') = KST $(date -u -d '+9 hours' '+%m-%d %H:%M')"
echo "- 사유: **2026-08-08 학교 정전 예정** — 8/7 시점 진행상황 보존 (요청 2026-08-05)"
echo "- run: $RUN (scratch 재기동 2026-08-05 02:42 UTC, BS8, multi-AR R∈{2,3,4,5,6,8}, 34.2M params)"
echo
echo "## 진행 요약"
echo
echo "- 완료 epoch: **${LAST_EP}/80** · best val_composite(현재까지): **${BEST:-—}** (val 은 R4 고정, 매 2ep)"
echo "- 잔여: ${REMAIN} epochs × ~2.75 h/ep ≈ ${REMAIN_H}h — 정전 복구·재개 시점 기준으로 ETA 재계산"
echo "- 참고 목표: unleashed R4 매치가 아니라 **R-sweep 완만함**(단일-R 대비)이 성공 기준. R4 val 은 unleashed(0.9203)보다 낮게 나오는 것이 정상."
echo
echo "## 학습 로그 전문 (logs/$RUN/log.txt)"
echo '```'
cat "$LOGDIR/log.txt" 2>/dev/null || echo "(log.txt 없음)"
echo '```'
echo
echo "## 프로세스 / GPU (스냅샷 시점)"
echo '```'
ps aux | grep -E 'main_train_ss2d_v9_radapt|run_ss2d_v9_radapt_autoresume' | grep -v grep || echo "(학습 프로세스 없음 — 8/7 21:00 KST 이후 깨끗한 정지 뒤라면 정상. 그 외 시점이면 RESUME_AFTER_OUTAGE.md 절차로 재기동)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1
echo '```'
echo
echo "## ckpt 상태 (logs/$RUN/ — 정전 시 이 파일들이 재개 지점)"
echo '```'
ls -la "$LOGDIR" 2>/dev/null
echo '```'
echo
echo "## 최근 학습 로그 tail"
echo '```'
tail -c 1500 "$RUNLOG" 2>/dev/null | tr '\r' '\n' | grep -v '^$' | tail -4
echo '```'
echo
echo "## 정전 안전성 & 재개 (상세: RESUME_AFTER_OUTAGE.md)"
echo
echo "- 정전으로 프로세스가 급사해도 안전: \`*_last.pt\` 는 매 epoch **atomic save**(tmp→rename) full-state → **true-resume 손실 ≤1 epoch**."
echo "- 복구 후 재개(요약):"
echo '  ```bash'
echo '  nvidia-smi && python -c "import torch; print(torch.cuda.is_available())"   # NVML 확인'
echo '  cd /home/snorlax/shared/fastmri_ViT_with_eternet'
echo '  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online MAX_RETRY=200 \'
echo '    setsid nohup bash v9_mamba_radapt/runs/run_ss2d_v9_radapt_autoresume.sh \'
echo '    > v9_mamba_radapt/runs/run_ss2d_v9_radapt.log 2>&1 < /dev/null & disown'
echo '  ```'
} > "$OUT"
sync
echo "[snapshot] 보고 저장 완료: $OUT (epoch ${LAST_EP}/80, best ${BEST:-—})"
