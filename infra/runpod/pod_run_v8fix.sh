#!/bin/bash
# RunPod Pod: v8fix(수정 레시피 DL) 50 epoch 학습 + test_full·val 전체 평가 (2026-10-09).
#   데이터 프로토콜(10-09 사용자 결정): train = 공식 train 묶음 중 로컬 대조표에 있는 파일, val = 공식 val 0~2 전체(학습 중 검증·베스트 선택),
#   test_full = 공식 test_full 0~2 전체(최종 평가, 학습 중에는 안 봄). 로컬 서버(val 0번만)와는 검증 조건이 다르므로 런 이름에 _fullval 을 붙인다.
#   순서 unet → gru → ss2d, SEED=1, 50 epoch 일정. 재기동도 같은 명령(DONE/PAUSED_ep{STOP} 런 skip, 미완 런 전체 상태 재개).
#   CUDA_VISIBLE_DEVICES=0 nohup setsid bash infra/runpod/pod_run_v8fix.sh > /root/_setup/run_outer.log 2>&1 < /dev/null &
#   트레이너 exit 3 = ALERT_COLLAPSE, exit 2 = 재개 설정 불일치 → 재시도하지 않고 중단.
set -u
ROOT=/root/fastmri_ViT_with_eternet
cd "$ROOT"
PY=/opt/mri310/bin/python
SEED="${SEED:-1}"
ARMS="${ARMS:-unet gru ss2d}"
EPOCHS="${EPOCHS:-50}"
STOP="${STOP_AFTER_EPOCH:-$EPOCHS}"
SUFFIX="${RUN_SUFFIX:-_s${SEED}_fullval}"
MAX_RETRY="${MAX_RETRY:-200}"
SMOKE_BS="${SMOKE_BS:-8}"
LOGDIR="$ROOT/v8_eter_pure/runs/runpod"; mkdir -p "$LOGDIR"
unset V8FIX_LOG_ROOT DEBUG_MAX_STEPS DEBUG_VAL_SAMPLES PROBE_VAL_SLICES HEALTH_EVERY_STEPS SAME_UNET_INIT

for ARM in $ARMS; do
  ARM_UP=$(echo "$ARM" | tr '[:lower:]' '[:upper:]')
  RUN="PureETER_${ARM_UP}_noDC_R4_brain384_v8fix${SUFFIX}"
  LOG="$LOGDIR/run_${RUN}.log"
  if [ -e "logs/$RUN/DONE" ] || [ -e "logs/$RUN/PAUSED_ep${STOP}" ]; then echo "[pod] $RUN epoch ${STOP} 완료 — skip"; continue; fi
  if [ -e "logs/$RUN/ALERT_COLLAPSE" ]; then echo "[pod] $RUN ALERT_COLLAPSE — 중단"; exit 3; fi
  ok=0
  for i in $(seq 1 "$MAX_RETRY"); do
    echo "[pod] $RUN attempt $i BS=$SMOKE_BS EPOCHS=$EPOCHS STOP=$STOP $(date -u)" | tee -a "$LOG"
    SEED=$SEED RUN_SUFFIX=$SUFFIX SEQ_MODEL=$ARM SANITY_NUM_EPOCHS=$EPOCHS STOP_AFTER_EPOCH=$STOP SMOKE_BS=$SMOKE_BS \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online \
      $PY v8_eter_pure/main_train_pure_v8fix.py >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { ok=1; break; }
    [ $rc -eq 3 ] && { echo "[pod] $RUN exit 3 ALERT_COLLAPSE — 중단" | tee -a "$LOG"; exit 3; }
    [ $rc -eq 2 ] && { echo "[pod] $RUN exit 2 거부 — 중단" | tee -a "$LOG"; exit 2; }
    echo "[pod] rc=$rc → 60s 후 재시도" | tee -a "$LOG"; sleep 60
  done
  [ $ok = 1 ] || { echo "[pod] $RUN MAX_RETRY 소진"; exit 1; }
  echo "[pod] === $RUN epoch ${STOP} 도달 $(date -u) ==="
done

# 평가: 각 모델의 best(val 전체 기준) 체크포인트를 test_full(최종)과 val 전체(참고)에서. U-Net 단독은 두 모델 CSV 를 기준으로 대응 비교.
ELOG="$LOGDIR/eval_${SUFFIX#_}.log"
ckpt() { echo "logs/PureETER_$(echo "$1" | tr '[:lower:]' '[:upper:]')_noDC_R4_brain384_v8fix${SUFFIX}/pure_$1_best.pt"; }
for split in test_full val; do
  OUT="results/eval/v8fix${SUFFIX}_${split}"
  DP="./fastMRI_data/multicoil_${split}"
  for seq in gru ss2d unet; do
    [ -e "$OUT/summary_${seq}.md" ] && continue
    refs=()
    [ "$seq" = ss2d ] && refs=(--ref "ETER-net (bi-GRU)=$OUT/per_slice_gru.csv:")
    [ "$seq" = unet ] && refs=(--ref "ETER-net (bi-GRU)=$OUT/per_slice_gru.csv:" --ref "SS2D=$OUT/per_slice_ss2d.csv:")
    echo "[pod] eval $split $seq $(date -u)" | tee -a "$ELOG"
    $PY v8_eter_pure/eval_v8fix.py --seq "$seq" --ckpt "$(ckpt "$seq")" --data-path "$DP" --out-dir "$OUT" \
      --skip-precheck --force --num-workers 8 "${refs[@]}" >> "$ELOG" 2>&1 || echo "[pod] eval $split $seq 실패" | tee -a "$ELOG"
  done
  $PY v8_eter_pure/eval_v8fix.py --seq gru --ckpt "$(ckpt gru)" --data-path "$DP" --out-dir "$OUT" --summary-only \
    --ref "SS2D=$OUT/per_slice_ss2d.csv:" >> "$ELOG" 2>&1 || true
done
echo "[pod] 전체 완료 $(date -u)" | tee -a "$ELOG"
