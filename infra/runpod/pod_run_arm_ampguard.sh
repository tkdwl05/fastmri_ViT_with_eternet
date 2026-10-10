#!/bin/bash
# RunPod Pod: v8fix 한 모델만 50 epoch 학습 + 그 모델의 test_full·val 평가 — AMP 보호 장치 트레이너판 (2026-10-10).
#   pod_run_arm.sh 와 같고 트레이너만 main_train_pure_v8fix_ampguard.py(fp16 GradScaler 배율 하한·기울기 비유한 재계산 — 그 파일 주석).
#   pod3 ETER-net(GRU) 런이 배율 붕괴로 무너진 뒤 GRU 재학습에 쓴다. 실행 중인 다른 런의 런처를 고치지 않으려고 새 파일로 분리.
#   모델을 여러 Pod·GPU 에 나눠 동시에 돌릴 때 쓴다(pod_run_v8fix.sh 는 한 Pod 에서 unet → gru → ss2d 차례로).
#   데이터·학습 설정·런 이름은 pod_run_v8fix.sh 와 같다(_s1_fullval, val 0~2 로 best 선택, test_full 최종 평가).
#   ARM=gru|ss2d|unet  GPU=<장치 번호>  WANDB_RUN_TAG=<Pod 이름 — 같은 런 이름이 다른 Pod 의 wandb run 과 겹치지 않게>
#   WAIT_PID=<pid>: 그 프로세스(이미 돌고 있는 같은 모델의 트레이너)가 끝날 때까지 기다린 뒤 시작 — 런처를 바꿔 끼울 때.
#   재기동도 같은 명령(DONE 이면 학습 skip, 미완이면 전체 상태 재개). exit 3 = ALERT_COLLAPSE, exit 2 = 재개 설정 불일치 → 중단.
#   대응(paired) 비교 요약은 세 모델의 per_slice CSV 를 한곳에 모은 뒤 eval_v8fix.py --summary-only 로 따로 만든다.
#   예) ARM=gru GPU=0 WANDB_RUN_TAG=pod3r setsid -f bash infra/runpod/pod_run_arm_ampguard.sh > /root/_setup/run_gru.log 2>&1 < /dev/null
set -u
ROOT=/root/fastmri_ViT_with_eternet
cd "$ROOT"
PY=/opt/mri310/bin/python
ARM="${ARM:?ARM=gru|ss2d|unet}"
GPU="${GPU:-0}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-50}"
SUFFIX="${RUN_SUFFIX:-_s${SEED}_fullval}"
MAX_RETRY="${MAX_RETRY:-200}"
ARM_UP=$(echo "$ARM" | tr '[:lower:]' '[:upper:]')
RUN="PureETER_${ARM_UP}_noDC_R4_brain384_v8fix${SUFFIX}"
LOGDIR="$ROOT/v8_eter_pure/runs/runpod"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/run_${RUN}.log"
export CUDA_VISIBLE_DEVICES="$GPU"
unset V8FIX_LOG_ROOT DEBUG_MAX_STEPS DEBUG_VAL_SAMPLES PROBE_VAL_SLICES HEALTH_EVERY_STEPS SAME_UNET_INIT

if [ -n "${WAIT_PID:-}" ]; then
  echo "[arm] $RUN: pid $WAIT_PID 종료 대기 $(date -u)" | tee -a "$LOG"
  # 부모(옛 런처)를 끈 트레이너는 끝나도 컨테이너 init 이 회수하지 않아 좀비로 남을 수 있다 → 상태 Z 도 종료로 본다.
  while kill -0 "$WAIT_PID" 2>/dev/null && [ "$(ps -o stat= -p "$WAIT_PID" 2>/dev/null | cut -c1)" != Z ]; do sleep 60; done
fi
[ -e "logs/$RUN/ALERT_COLLAPSE" ] && { echo "[arm] $RUN ALERT_COLLAPSE — 중단"; exit 3; }

if [ ! -e "logs/$RUN/DONE" ]; then
  ok=0
  for i in $(seq 1 "$MAX_RETRY"); do
    echo "[arm] $RUN attempt $i GPU=$GPU EPOCHS=$EPOCHS $(date -u)" | tee -a "$LOG"
    SEED=$SEED RUN_SUFFIX=$SUFFIX SEQ_MODEL=$ARM SANITY_NUM_EPOCHS=$EPOCHS STOP_AFTER_EPOCH=$EPOCHS SMOKE_BS=8 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PREFETCH_FACTOR=2 NUM_WORKERS_VAL=12 WANDB_MODE=online \
      WANDB_ENTITY=tkdwl05-hongik-university WANDB_PROJECT=fastMRI-research \
      $PY v8_eter_pure/main_train_pure_v8fix_ampguard.py >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { ok=1; break; }
    [ $rc -eq 3 ] && { echo "[arm] $RUN exit 3 ALERT_COLLAPSE — 중단" | tee -a "$LOG"; exit 3; }
    [ $rc -eq 2 ] && { echo "[arm] $RUN exit 2 거부 — 중단" | tee -a "$LOG"; exit 2; }
    echo "[arm] rc=$rc → 60s 후 재시도" | tee -a "$LOG"; sleep 60
  done
  [ $ok = 1 ] || { echo "[arm] $RUN MAX_RETRY 소진"; exit 1; }
fi

# 평가: best(val 0~2 기준) 체크포인트를 test_full(최종)과 val 전체(참고)에서. 대응 비교 요약은 따로.
ELOG="$LOGDIR/eval_${ARM}${SUFFIX}.log"
for split in test_full val; do
  OUT="results/eval/v8fix${SUFFIX}_${split}"
  [ -e "$OUT/summary_${ARM}.md" ] && continue
  echo "[arm] eval $split $ARM $(date -u)" | tee -a "$ELOG"
  $PY v8_eter_pure/eval_v8fix.py --seq "$ARM" --ckpt "logs/$RUN/pure_${ARM}_best.pt" --data-path "./fastMRI_data/multicoil_${split}" \
    --out-dir "$OUT" --skip-precheck --force --num-workers 8 >> "$ELOG" 2>&1 || echo "[arm] eval $split $ARM 실패" | tee -a "$ELOG"
done
echo "[arm] $RUN 학습·평가 완료 $(date -u)" | tee -a "$ELOG"
