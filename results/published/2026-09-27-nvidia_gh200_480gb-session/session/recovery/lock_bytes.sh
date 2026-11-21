#!/usr/bin/env bash
# Recovery: the 1710-lock byte pages the driver's loop never took. It stopped on V10 at G=1,
# which the G=1 page itself shows is a gate artefact: time = 22.7 us + cycles / 1705 MHz over
# 54 GEMMs (the clock held; ncu's time window carries a fixed offset). V10 is re-scored on the
# laptop later; every page keeps its .ncu-rep. Runs only after DRIVER-DONE.
set -uo pipefail
. ~/moe/env.sh && cd "$REPO" || exit 2
S=$SESSION_ROOT; R=$(cat "$S/counters-dir.txt"); C=$S/census.json; F=1710; L=$R/lock$F
ST=$S/gh200-driver/status
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU busy"; exit 2; }
echo "$(date -u +%FT%TZ) lock-1710 byte recovery start (V10 is a timing-window artefact on this card)" >> "$ST"
( set -o pipefail
  trap 'trap "" INT TERM HUP; moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null' EXIT
  trap 'exit 130' INT TERM HUP
  moe_counter ncu --clock-control reset
  sudo -n nvidia-smi -lgc "$F,$F" || exit 2
  for G in 2 4 16 8 32 3 64; do
    LOG=$S/logs/r3c-g$G-lock$F.log
    moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --group-m "$G" \
      --tiles 1,2,3,4,5,6,7,8,9 --census "$C" --page-clock none --page-lock-mhz "$F" \
      --out "$L/r3c-g$G.json" 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    failed=$(grep -E '^RESULT: VALIDITY [^ ]+ FAIL' "$LOG" | awk '{print $3}' | sort -u | tr '\n' ' ')
    echo "$(date -u +%FT%TZ) recovery lock bytes G=$G exit $rc failed: ${failed:-none}" >> "$ST"
    bash "$REPO/scripts/vm_results_push.sh" push --message "recovery lock bytes G=$G: exit $rc" >/dev/null 2>&1
    case $rc in 0|1) ;; 3) [ -f "$L/r3c-g$G.json" ] || exit 3 ;; *) exit "$rc" ;; esac
  done )
echo "$(date -u +%FT%TZ) recovery lock block exit $?" >> "$ST"
sleep 5
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv >> "$S/clocks-after-recovery.txt"
P=$REPO/results/published/2026-09-25-nvidia_gh200_480gb-session/results/gaps-nvidia_gh200_480gb/private_weight_reference
TIMED=$(ls "$P"/*d9f1f37c/report.json "$P"/*df37ea07/report.json "$P"/*01c08abd/report.json "$P"/*1b285de2/report.json "$P"/*6ff34777/report.json | tr '\n' ' ')
python3 scripts/dram_counter_route.py --analyse "$L"/r3c-g*.json --timed-reference $TIMED --out "$L/summary.json" 2>&1 | tee "$S/logs/analyse-lock$F-recovery.log"
echo "$(date -u +%FT%TZ) recovery analyse exit ${PIPESTATUS[0]}; recovery END" >> "$ST"
bash "$REPO/scripts/vm_results_push.sh" push --message "lock-1710 byte recovery: END"
