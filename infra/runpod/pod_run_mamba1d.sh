#!/bin/bash
# RunPod pod2: Mamba-ETER(PureETER_Mamba1D) 50 epoch 학습 + test_full·val 평가 (2026-10-12, 사용자 결정 — SEED 1, mamba_ssm 기본 초기화,
#   Mamba h·v fp32). pod_run_arm_ampguard.sh(pod3 ETER-net 런처)와 같은 구조·환경 변수이고 다른 점만:
#   - 트레이너 main_train_pure_v8fix_mamba1d.py(ampguard 트레이너 복사본 — 모델만 Mamba-ETER), 평가 eval_v8fix_mamba1d.py.
#   - MAMBA_ETER_FP32=hv 를 트레이너·평가에 같이 넘긴다(정밀도는 생성 인자라 state_dict 로 확인되지 않음).
#   - EXPECT_SBAR·EXPECT_UNET_CHECKSUM: ETER-net·SS2D·U-Net 단독 런과 같은 학습 데이터(s̄)·U-Net 초기값이어야 시작(아니면 exit 2).
#   - 재시도 루프 전에 한 번만 도는 사전 점검(import·파일) — 실패하면 재시도 없이 exit 2.
#   재기동도 같은 명령(DONE 이면 학습 skip, 미완이면 전체 상태 재개). exit 3 = ALERT_COLLAPSE, exit 2 = 거부 → 중단.
#   예) WANDB_RUN_TAG=pod2 setsid -f bash infra/runpod/pod_run_mamba1d.sh > /root/_setup/run_mamba1d.log 2>&1 < /dev/null
set -u
ROOT=/root/fastmri_ViT_with_eternet
cd "$ROOT"
PY=/opt/mri310/bin/python
ARM=mamba1d
GPU="${GPU:-0}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-50}"
SUFFIX="${RUN_SUFFIX:-_s${SEED}_fullval}"
MAX_RETRY="${MAX_RETRY:-200}"
RUN="PureETER_MAMBA1D_noDC_R4_brain384_v8fix${SUFFIX}"
LOGDIR="$ROOT/v8_eter_pure/runs/runpod"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/run_${RUN}.log"
export CUDA_VISIBLE_DEVICES="$GPU"
export MAMBA_ETER_FP32=hv
# 학습 동작을 바꾸는 환경 변수는 지워서 트레이너 기본값(= ETER-net 런과 같은 값)을 쓰게 한다
unset V8FIX_LOG_ROOT DEBUG_MAX_STEPS DEBUG_VAL_SAMPLES PROBE_VAL_SLICES HEALTH_EVERY_STEPS SAME_UNET_INIT GRU_FP32_MODULES \
      AMP_TEST_INJECT_ONCE AMP_TEST_INJECT_ALWAYS ALLOW_TF32 SANITY_VAL_EVERY_N_EPOCHS AMP_MIN_SCALE AMP_GRAD_RETRY \
      TRAIN_NAN_RETRY VAL_MAX_RETRY ACCUM_STEPS
# 같은 학습 데이터·U-Net 초기값 (ETER-net·SS2D·U-Net 단독 _s1_fullval 런의 recipe.json·SCRATCH START 줄; SEED 1 에서만 해당)
if [ "$SEED" = 1 ]; then
  export EXPECT_SBAR="${EXPECT_SBAR:-325.2484691441059}" EXPECT_UNET_CHECKSUM="${EXPECT_UNET_CHECKSUM:-5.9212156172e+02}"
fi

if [ -n "${WAIT_PID:-}" ]; then
  echo "[mamba1d] $RUN: pid $WAIT_PID 종료 대기 $(date -u)" | tee -a "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null && [ "$(ps -o stat= -p "$WAIT_PID" 2>/dev/null | cut -c1)" != Z ]; do sleep 60; done
fi
[ -e "logs/$RUN/ALERT_COLLAPSE" ] && { echo "[mamba1d] $RUN ALERT_COLLAPSE — 중단"; exit 3; }

# 사전 점검 (한 번만, 재시도 없음): 파일·import·CUDA 확장(결정적 오류만). torch 를 먼저 import 해야 CUDA 확장이 libc10 을 찾는다.
# GPU 사용 가능 여부는 일시 오류일 수 있어 여기서 보지 않는다(트레이너가 rc 1 → 재시도).
for f in v8_eter_pure/main_train_pure_v8fix_mamba1d.py v8_eter_pure/eval_v8fix_mamba1d.py \
         models/pure_eternet/u_pure_eternet_mamba1d.py v8_eter_pure/smoke_recipe_fix.py; do
  [ -e "$f" ] || { echo "[mamba1d] 파일 없음 $f — 중단" | tee -a "$LOG"; exit 2; }
done
$PY - >> "$LOG" 2>&1 <<'EOF' || { echo "[mamba1d] 사전 점검 실패 (로그 참고) — 중단" | tee -a "$LOG"; exit 2; }
import sys
import torch, causal_conv1d_cuda, selective_scan_cuda  # noqa: F401
sys.path[:0] = ['models/pure_eternet', 'models/hybrid_eternet', 'v8_eter_pure']
from u_pure_eternet_mamba1d import PureETER_Mamba1D  # noqa: F401
import mamba_ssm.modules.mamba_simple as MS
assert all(getattr(MS, k) is not None for k in ('causal_conv1d_fn', 'selective_scan_fn', 'mamba_inner_fn'))
print('[mamba1d preflight] ok', torch.__version__)
EOF

if [ ! -e "logs/$RUN/DONE" ]; then
  ok=0
  for i in $(seq 1 "$MAX_RETRY"); do
    echo "[mamba1d] $RUN attempt $i GPU=$GPU EPOCHS=$EPOCHS $(date -u)" | tee -a "$LOG"
    SEED=$SEED RUN_SUFFIX=$SUFFIX SEQ_MODEL=$ARM SANITY_NUM_EPOCHS=$EPOCHS STOP_AFTER_EPOCH=$EPOCHS SMOKE_BS=8 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PREFETCH_FACTOR=2 NUM_WORKERS_VAL=12 WANDB_MODE=online \
      WANDB_ENTITY=tkdwl05-hongik-university WANDB_PROJECT=fastMRI-research \
      $PY v8_eter_pure/main_train_pure_v8fix_mamba1d.py >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { ok=1; break; }
    [ $rc -eq 3 ] && { echo "[mamba1d] $RUN exit 3 ALERT_COLLAPSE — 중단" | tee -a "$LOG"; exit 3; }
    [ $rc -eq 2 ] && { echo "[mamba1d] $RUN exit 2 거부 — 중단" | tee -a "$LOG"; exit 2; }
    echo "[mamba1d] rc=$rc → 60s 후 재시도" | tee -a "$LOG"; sleep 60
  done
  [ $ok = 1 ] || { echo "[mamba1d] $RUN MAX_RETRY 소진"; exit 1; }
fi

# 평가: best(val 0~2 기준) 체크포인트를 test_full(최종)과 val 전체(참고)에서 — ETER-net 런처와 같은 규칙. 대응 비교 요약은 따로.
ELOG="$LOGDIR/eval_${ARM}${SUFFIX}.log"
ef=0
for split in test_full val; do
  OUT="results/eval/v8fix${SUFFIX}_${split}"
  [ -e "$OUT/summary_${ARM}.md" ] && continue
  echo "[mamba1d] eval $split $(date -u)" | tee -a "$ELOG"
  $PY v8_eter_pure/eval_v8fix_mamba1d.py --seq "$ARM" --ckpt "logs/$RUN/pure_${ARM}_best.pt" \
    --data-path "./fastMRI_data/multicoil_${split}" --out-dir "$OUT" --skip-precheck --force --num-workers 8 >> "$ELOG" 2>&1 \
    || { echo "[mamba1d] eval $split 실패" | tee -a "$ELOG"; ef=1; }
done
if [ $ef = 1 ]; then
  echo "[mamba1d] $RUN 학습 완료, 평가 실패 있음 — 같은 명령으로 다시 실행하면 빠진 평가만 돈다 $(date -u)" | tee -a "$ELOG"; exit 4
fi
echo "[mamba1d] $RUN 학습·평가 완료 $(date -u)" | tee -a "$ELOG"
