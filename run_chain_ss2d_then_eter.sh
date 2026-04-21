#!/usr/bin/env bash
# Wait for the currently running ETER training (PID 5792) to finish,
# then run SS2D (v3) training, then the modified ETER (v4) training.

set -u

PY=/home/snorlax-dw/miniforge3/envs/mri_env/bin/python3.10
CWD=/home/snorlax-dw/바탕화면/ViT_based_MRIrecon
WAIT_PID=5792

CHAIN_LOG="$CWD/run_chain.log"
SS2D_LOG="$CWD/run_ss2d_v3.log"
ETER_LOG="$CWD/run_eter_v4.log"

cd "$CWD" || { echo "chdir failed" >&2; exit 1; }

log()  { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN_LOG"; }

log "chain script started; waiting for PID $WAIT_PID to finish..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done
log "PID $WAIT_PID has exited."

log "starting SS2D (v3) training → $SS2D_LOG"
"$PY" main_train_ss2d.py > "$SS2D_LOG" 2>&1
SS2D_RC=$?
log "SS2D finished with exit code $SS2D_RC"
if [ "$SS2D_RC" -ne 0 ]; then
    log "SS2D failed; aborting chain. See $SS2D_LOG"
    exit "$SS2D_RC"
fi

log "starting ETER (v4) training → $ETER_LOG"
"$PY" main_train_eter.py > "$ETER_LOG" 2>&1
ETER_RC=$?
log "ETER finished with exit code $ETER_RC"

log "chain script done."
exit "$ETER_RC"
