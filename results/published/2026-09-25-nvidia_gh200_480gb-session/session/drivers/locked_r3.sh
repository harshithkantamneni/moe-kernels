#!/bin/bash
# On the GH200: let r3-g4-s0 finish, stop the duty-cycle chain, then re-run R3's
# ratio at every G with the SM clock locked at $LOCK MHz, reset it, mark done.
set -u
LOCK=${LOCK:-1710}
L=~/moe/session/locked-r3-$LOCK; mkdir -p "$L"
log() { echo "$(date -u +%FT%TZ) $*" >> "$L/status"; }
echo "locked-r3" > ~/autopilot.state
. ~/moe/env.sh; cd "$REPO" || { log "no repo"; exit 9; }
export MOE_RESULTS_DIR=$RESULTS_ROOT/gaps-$MOE_CARD
TAG=alpha_g-nvidia_gh200_480gb-20260925T071107Z-lock$LOCK
reset() { sudo nvidia-smi -rgc > "$L/rgc.txt" 2>&1; nvidia-smi -q -d CLOCK > "$L/clocks-after-reset.txt" 2>&1; }
trap reset EXIT
sudo nvidia-smi -lgc $LOCK,$LOCK > "$L/lgc.txt" 2>&1 || { log "lock refused"; exit 8; }
nvidia-smi -q -d CLOCK,PERFORMANCE,POWER > "$L/clocks-locked.txt" 2>&1
log "locked at $LOCK MHz"
for G in 1 2 4 16 64; do
  log "G=$G start"
  timeout --signal=INT --kill-after=60 1800 "$PY_VLLM" scripts/private_weight_reference.py \
    --model mixtral-8x7b --block-m 32 --treads 6 --repeats 9 --group-m "$G" --duty 0.25 \
    --seed 0 --session-tag "$TAG" > "$L/r3-g$G.log" 2>&1
  log "G=$G rc=$?"
done
reset; trap - EXIT
log "clock reset; DONE"
echo done > ~/autopilot.state; touch ~/AUTOPILOT_DONE
