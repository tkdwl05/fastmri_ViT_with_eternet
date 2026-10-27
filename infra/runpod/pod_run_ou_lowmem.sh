#!/bin/bash
# (10-26 RAM 57 GiB Pod 용 사본) pod_run_ou.sh 와 같고 학습 prefetch 만 2 → 1. 트레이너 주석대로 prefetch·val 워커 수는 결과와 무관
#   (워커가 맡는 샘플·난수 불변). 한도 116 GiB Pod 에서 같은 트레이너가 공유 메모리 약 54 GB 를 써서, 57 GiB Pod 에서는 OOM(rc=137).
# RunPod Pod: 원래 U-Net(OU) 실험의 한 모델을 50 epoch 학습 + 런처 평가(test_full·val) (2026-10-26).
#   계획: paper/ieie/reviews/ars_pipeline_mdpi_20261020/stage2/v1_prep/origunet_s1_plan.md (학습 전에 고정).
#   pod_run_arm.sh · pod_run_arm_ampguard.sh 와 같은 구조, 같은 학습 환경 변수(SEED·RUN_SUFFIX·SANITY_NUM_EPOCHS·STOP_AFTER_EPOCH·
#   SMOKE_BS=8·expandable_segments·PREFETCH_FACTOR=2·NUM_WORKERS_VAL=12·wandb), 같은 재시도 규칙. 다른 점:
#     - MODEL=unet_ou | gru_ou | gruo_ou  (U-Net only (OU) / d-ETER (OU) / o-ETER (OU)).
#       트레이너: unet_ou → v8_eter_pure/main_train_pure_v8fix_ou.py (U-Net only 를 학습한 기본 트레이너의 OU 판),
#                 gru_ou·gruo_ou → v8_eter_pure/main_train_pure_v8fix_ampguard_ou.py (ETER-net 을 학습한 AMP 보호 장치판, gru_h fp32).
#     - 런 폴더 logs/PureETER_{UNETOU|GRUOU|GRUOOU}_noDC_R4_brain384_v8fix${SUFFIX} — 이름에서 '_' 를 빼고 대문자(recipe_ou.run_name 과
#       같은 규칙, sanity_ou.py 점검 9). 기존 런 폴더(PureETER_UNET·GRU·SS2D·…)와 겹치지 않고, OU 꼴이 아니면 시작하지 않는다.
#     - 트레이너 시험용 환경 변수(OU_TRAINER_DFU_CHECK·AMP_TEST_INJECT_*·디버그 상한 등)와 GRU_FP32_MODULES·ALLOW_TF32 를 지운다
#       (트레이너 기본값 = 시드 1 ETER-net 과 같은 gru_h fp32·TF32 끔).
#     - 런처 평가: v8_eter_pure/eval_v8fix_fb_ou.py — eval_v8fix.py(per-slice CSV + 기능 시험 512 슬라이스)를 수정 3 의 fp32 재계산
#       규칙 래퍼(eval_v8fix_fb.py)로 돌리는 OU 진입점. fp16 출력이 모두 유한하면 eval_v8fix.py 를 그대로 돌린 것과 출력이 같다.
#       출력 results/eval/v8fix_ou${SUFFIX}_{test_full,val}/ (기존 결과 폴더와 분리; summary_${MODEL}.md 가 이미 있으면 건너뜀).
#       공식 정의 평가(eval_v8fix_official_fb_ou.py)는 따로 돌린다.
#   MODEL=…  GPU=<장치 번호>  WANDB_RUN_TAG=<Pod 이름 — 같은 런 이름이 다른 Pod 의 wandb run 과 겹치지 않게>
#   WAIT_PID=<pid>: 그 프로세스가 끝날 때까지 기다린 뒤 시작 (pod_run_arm.sh 와 같음).
#   재기동도 같은 명령(DONE 이면 학습 skip, 미완이면 전체 상태 재개). exit 3 = ALERT_COLLAPSE, exit 2 = 재개 설정 불일치·이름 거부 → 중단.
#   예) MODEL=gru_ou GPU=0 WANDB_RUN_TAG=pod4 setsid -f bash infra/runpod/pod_run_ou.sh > /root/_setup/run_gru_ou.log 2>&1 < /dev/null
set -u
ROOT=/root/fastmri_ViT_with_eternet
cd "$ROOT"
PY=/opt/mri310/bin/python
MODEL="${MODEL:?MODEL=unet_ou|gru_ou|gruo_ou}"
case "$MODEL" in
  unet_ou)        TRAINER=v8_eter_pure/main_train_pure_v8fix_ou.py ;;
  gru_ou|gruo_ou) TRAINER=v8_eter_pure/main_train_pure_v8fix_ampguard_ou.py ;;
  *) echo "[ou] MODEL=$MODEL — unet_ou|gru_ou|gruo_ou 만"; exit 2 ;;
esac
GPU="${GPU:-0}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-50}"
SUFFIX="${RUN_SUFFIX:-_s${SEED}_fullval}"
MAX_RETRY="${MAX_RETRY:-200}"
TAG=$(echo "$MODEL" | tr -d '_' | tr '[:lower:]' '[:upper:]')
RUN="PureETER_${TAG}_noDC_R4_brain384_v8fix${SUFFIX}"
case "$TAG" in UNETOU|GRUOU|GRUOOU) ;; *) echo "[ou] 런 이름 $RUN 이 OU 꼴이 아님 — 중단"; exit 2 ;; esac
for f in "$TRAINER" v8_eter_pure/recipe_ou.py v8_eter_pure/eval_v8fix_fb_ou.py v8_eter_pure/eval_v8fix_fb.py \
         v8_eter_pure/eval_v8fix_official_fb.py v8_eter_pure/eval_v8fix_official_fb_ou.py v8_eter_pure/gru_precision.py \
         models/pure_eternet/u_pure_eternet_ou.py models/pure_eternet/unet_original.py; do
  [ -e "$f" ] || { echo "[ou] 파일 없음: $f — 중단"; exit 2; }
done
LOGDIR="$ROOT/v8_eter_pure/runs/runpod"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/run_${RUN}.log"
export CUDA_VISIBLE_DEVICES="$GPU"
unset V8FIX_LOG_ROOT DEBUG_MAX_STEPS DEBUG_VAL_SAMPLES PROBE_VAL_SLICES HEALTH_EVERY_STEPS SAME_UNET_INIT SEQ_MODEL \
      OU_TRAINER_DFU_CHECK GRU_FP32_MODULES ALLOW_TF32 AMP_TEST_INJECT_ONCE AMP_TEST_INJECT_ALWAYS \
      EVAL_FB_FORCE_NONFINITE EVAL_FB_FORCE_NONFINITE_FP32

if [ -n "${WAIT_PID:-}" ]; then
  echo "[ou] $RUN: pid $WAIT_PID 종료 대기 $(date -u)" | tee -a "$LOG"
  # 부모(옛 런처)를 끈 트레이너는 끝나도 컨테이너 init 이 회수하지 않아 좀비로 남을 수 있다 → 상태 Z 도 종료로 본다.
  while kill -0 "$WAIT_PID" 2>/dev/null && [ "$(ps -o stat= -p "$WAIT_PID" 2>/dev/null | cut -c1)" != Z ]; do sleep 60; done
fi
[ -e "logs/$RUN/ALERT_COLLAPSE" ] && { echo "[ou] $RUN ALERT_COLLAPSE — 중단"; exit 3; }

if [ ! -e "logs/$RUN/DONE" ]; then
  ok=0
  for i in $(seq 1 "$MAX_RETRY"); do
    echo "[ou] $RUN attempt $i GPU=$GPU EPOCHS=$EPOCHS trainer=$TRAINER $(date -u)" | tee -a "$LOG"
    SEED=$SEED RUN_SUFFIX=$SUFFIX SEQ_MODEL=$MODEL SANITY_NUM_EPOCHS=$EPOCHS STOP_AFTER_EPOCH=$EPOCHS SMOKE_BS=8 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PREFETCH_FACTOR=1 NUM_WORKERS_VAL=12 WANDB_MODE=online \
      WANDB_ENTITY=tkdwl05-hongik-university WANDB_PROJECT=fastMRI-research \
      $PY "$TRAINER" >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { ok=1; break; }
    [ $rc -eq 3 ] && { echo "[ou] $RUN exit 3 ALERT_COLLAPSE — 중단" | tee -a "$LOG"; exit 3; }
    [ $rc -eq 2 ] && { echo "[ou] $RUN exit 2 거부 — 중단" | tee -a "$LOG"; exit 2; }
    echo "[ou] rc=$rc → 60s 후 재시도" | tee -a "$LOG"; sleep 60
  done
  [ $ok = 1 ] || { echo "[ou] $RUN MAX_RETRY 소진"; exit 1; }
fi

# 런처 평가: best(val 0~2 기준) 체크포인트를 test_full(최종)과 val 전체(참고)에서. 대응 비교·공식 정의 평가는 따로.
ELOG="$LOGDIR/eval_${MODEL}${SUFFIX}.log"
for split in test_full val; do
  OUT="results/eval/v8fix_ou${SUFFIX}_${split}"
  [ -e "$OUT/summary_${MODEL}.md" ] && continue
  echo "[ou] eval $split $MODEL $(date -u)" | tee -a "$ELOG"
  $PY v8_eter_pure/eval_v8fix_fb_ou.py --seq "$MODEL" --ckpt "logs/$RUN/pure_${MODEL}_best.pt" \
    --data-path "./fastMRI_data/multicoil_${split}" --out-dir "$OUT" --skip-precheck --force --num-workers 8 >> "$ELOG" 2>&1 \
    || echo "[ou] eval $split $MODEL 실패 (rc=$?)" | tee -a "$ELOG"
done
echo "[ou] $RUN 학습·평가 완료 $(date -u)" | tee -a "$ELOG"
