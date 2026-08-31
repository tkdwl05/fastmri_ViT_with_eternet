#!/bin/bash
# 재부팅/컨테이너 재시작/정전 복구 후 원-커맨드 재장전 (작성 2026-08-06).
#   bash v9_mamba_radapt/runs/post_reboot_rearm.sh
# 수행: ①NVML 확인 ②radapt supervisor 재기동(true-resume) ③8/7 정전대비 타이머 2개 재장전
#       (이미 지난 시각의 타이머는 skip — 8/8 이후 실행하면 자연히 재개만 수행).
set -u
ROOT=/home/snorlax/shared/fastmri_ViT_with_eternet
cd "$ROOT"

echo "== 1) GPU/NVML =="
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || { echo "✗ NVML 실패 — host 에서 sudo systemctl restart docker 후 재시도"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda False'; print('torch.cuda OK')" || exit 1

echo "== 2) radapt supervisor 재기동 (true-resume from last.pt) =="
if pgrep -f main_train_ss2d_v9_radapt.py >/dev/null; then
  echo "이미 실행 중 — skip"
else
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online MAX_RETRY=200 \
    setsid nohup bash v9_mamba_radapt/runs/run_ss2d_v9_radapt_autoresume.sh \
    > v9_mamba_radapt/runs/run_ss2d_v9_radapt.log 2>&1 < /dev/null & disown
  echo "launch 완료"
fi

echo "== 3) 8/7 정전대비 타이머 재장전 =="
arm_timer(){ # $1=epoch초 $2=실행스크립트 $3=타이머로그 $4=라벨
  local T=$1
  if [ "$(date +%s)" -ge "$T" ]; then echo "$4: 시각 경과 — skip (필요시 수동: bash $2)"; return; fi
  if pgrep -f "T=$T" >/dev/null; then echo "$4: 이미 armed — skip"; return; fi
  setsid nohup bash -c "T=$T; N=\$(date +%s); [ \"\$T\" -gt \"\$N\" ] && sleep \$((T-N)); bash $2" > "$3" 2>&1 < /dev/null & disown
  echo "$4: armed (KST $(date -u -d @$((T+32400)) '+%m-%d %H:%M') 실행 예정)"
}
arm_timer 1786103640 "$ROOT/v9_mamba_radapt/runs/snapshot_pre_outage.sh"   "$ROOT/v9_mamba_radapt/runs/snapshot_timer.log"   "보고 타이머(8/7 20:54 KST)"
arm_timer 1786104000 "$ROOT/v9_mamba_radapt/runs/clean_stop_pre_outage.sh" "$ROOT/v9_mamba_radapt/runs/clean_stop_timer.log" "정지 타이머(8/7 21:00 KST)"

echo "== 4) 상태 =="
sleep 5
pgrep -af 'main_train_ss2d_v9_radapt|T=17861' | sed 's/^/  /' || true
echo "완료. ~5분 후 학습 전진 확인:"
echo "  tail -c 2000 v9_mamba_radapt/runs/ss2d/run_PureETER_SS2D_V9_radapt_multiAR_brain384.log | tr '\\r' '\\n' | tail -3"
