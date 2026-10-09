#!/bin/bash
# RunPod Pod: 본 학습 전 점검 (2026-10-09) — 모델별로 짧게 학습해 동작·속도·GPU 메모리·wandb 기록을 확인한다.
#   DEBUG_MAX_STEPS 만큼 1 epoch 를 돌리고 val 앞쪽 DEBUG_VAL_SAMPLES 장으로 검증한다(측정·인용 불가 런). 로그는 /root/check_logs (본 런 폴더와 분리),
#   wandb 는 WANDB_RUN_TAG=podcheck 로 본 런과 다른 run 에 기록한다.
#   bash infra/runpod/pod_check_train.sh  → 끝에 모델별 batch/s · 최대 GPU 메모리 요약
set -u
ROOT=/root/fastmri_ViT_with_eternet; cd "$ROOT"
PY=/opt/mri310/bin/python
OUT=/root/check_logs; mkdir -p $OUT
STEPS="${STEPS:-400}"
for ARM in ${ARMS:-unet gru ss2d}; do
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > $OUT/${ARM}_gpumem.txt 2>/dev/null &
  MON=$!
  SEED=1 SEQ_MODEL=$ARM SANITY_NUM_EPOCHS=50 STOP_AFTER_EPOCH=1 SMOKE_BS=8 V8FIX_LOG_ROOT=$OUT/logs \
    DEBUG_MAX_STEPS=$STEPS DEBUG_VAL_SAMPLES=256 PROBE_VAL_SLICES=8 HEALTH_EVERY_STEPS=200 \
    WANDB_MODE=online WANDB_PROJECT=fastMRI-v8fix-runpod WANDB_RUN_TAG=podcheck PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY v8_eter_pure/main_train_pure_v8fix.py > $OUT/${ARM}.log 2>&1
  rc=$?
  kill $MON 2>/dev/null
  rate=$(tr '\r' '\n' < $OUT/${ARM}.log | grep -a "Epoch   1/50" | grep -o "[0-9.]*batch/s\|[0-9.]*s/batch" | tail -1)
  mem=$(sort -n $OUT/${ARM}_gpumem.txt | tail -1)
  echo "[check] $ARM rc=$rc rate=$rate peak_gpu_mem=${mem}MiB | $(grep -a '^Epoch 1/50\|ALERT\|FATAL' $OUT/logs/*/log.txt 2>/dev/null | grep -i "$ARM" | tail -1 | cut -c1-200)"
  grep -a "wandb: .*View run\|wandb: 🚀" $OUT/${ARM}.log | tail -1
done
