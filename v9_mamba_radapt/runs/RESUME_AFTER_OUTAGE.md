# 정전(2026-08-08) 후 radapt 재개 런북

작성 2026-08-05. **8/7 밤 자동 깨끗한 정지 예약됨**: 21:00 KST 부터 epoch 경계(last.pt 새 저장)를
감시하다 저장 직후 supervisor→trainer 순으로 정지(진행 손실 ~0), 늦어도 23:55 KST 하드 데드라인에 정지
(`clean_stop_pre_outage.sh`, 로그 `clean_stop_2026-08-07.log`). 따라서 8/8 정전 시점엔 이미 아무것도
돌고 있지 않다. 만에 하나 정지가 실행되지 않았더라도 안전: `ss2d_v9_radapt_last.pt` 가 매 epoch
atomic save(tmp→rename) full-state 라 **true-resume 손실 ≤1 epoch**.

## 재개 절차 (전원 복구 후)

> **⚡ 원-커맨드(권장)**: `bash v9_mamba_radapt/runs/post_reboot_rearm.sh` — 아래 2~3단계 + 8/7
> 정전대비 타이머 재장전(지난 시각이면 자동 skip)을 한 번에 수행. **8/8 이전의 어떤 재부팅/컨테이너
> 재시작 후에도 이 스크립트를 실행해야 타이머가 되살아난다** (분리 타이머는 재부팅을 못 넘김).

1. **host 부팅 + 컨테이너 기동 확인** — 컨테이너(`snorlax_WORK0`)가 자동 시작 안 되면 host 에서
   `sudo docker start <container>` (또는 도커 데몬부터).
2. **GPU/NVML 확인** (컨테이너 안):
   ```bash
   nvidia-smi                                                    # TITAN RTX GPU0 보여야 함
   python -c "import torch; print(torch.cuda.is_available())"    # True
   ```
   False 면 host 에서 `sudo systemctl restart docker` 후 재확인 ([[host-nvml-issue]] 패턴 —
   부팅 직후엔 보통 정상).
3. **supervisor 재기동** (true-resume, scratch 아님 — last.pt 자동 감지):
   ```bash
   cd /home/snorlax/shared/fastmri_ViT_with_eternet
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online MAX_RETRY=200 \
     setsid nohup bash v9_mamba_radapt/runs/run_ss2d_v9_radapt_autoresume.sh \
     > v9_mamba_radapt/runs/run_ss2d_v9_radapt.log 2>&1 < /dev/null & disown
   ```
4. **재개 검증** (~5분):
   ```bash
   ps aux | grep main_train_ss2d_v9_radapt | grep -v grep        # 트레이너 살아있음
   tail -c 2000 v9_mamba_radapt/runs/ss2d/run_PureETER_SS2D_V9_radapt_multiAR_brain384.log | tr '\r' '\n' | tail -3
   #   → "Epoch N/80" 이 정전 직전 epoch+1 부터 이어져야 정상 (resume 로그 라인 확인)
   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # ~99%, ~18.6GB
   ```

## 참고

- 정전 직전 상태 보고: `v9_mamba_radapt/runs/pre_outage_report_2026-08-07.md`
  (8/7 20:54 KST 자동 저장; 수동 갱신 = `bash v9_mamba_radapt/runs/snapshot_pre_outage.sh`)
- 완주 sentinel: `logs/PureETER_SS2D_V9_radapt_multiAR_brain384/DONE`. 완주 후 작업(R-sweep·4-way
  viz·화이트리스트)은 메모리 `v9-mamba-status` 의 "잔여" 항목 참고.
- 정지 시간만큼 ETA 밀림: 재개 시점 기준 잔여 epoch × ~2.75 h/ep (예: 8/8 저녁 재개 시 완주 ~08-15/16).
