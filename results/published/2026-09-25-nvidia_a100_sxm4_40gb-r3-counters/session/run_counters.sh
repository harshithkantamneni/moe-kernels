#!/bin/bash
set -u
. ~/moe/env.sh && cd "$REPO"
S=$SESSION_ROOT
R=$RESULTS_ROOT/$(date -u +%F)-$MOE_CARD-r3-counters && mkdir -p "$R" "$S/logs"
echo "R=$R" > ~/run_counters.status
for G in 64 1 4 2 16; do
  echo "G=$G start $(date -u +%H:%M:%S)" >> ~/run_counters.status
  moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms \
    --group-m $G --census "$S/census.json" --out "$R/r3c-g$G.json" > "$S/logs/r3c-g$G.log" 2>&1
  echo "G=$G rc=$? end $(date -u +%H:%M:%S)" >> ~/run_counters.status
done
python3 scripts/dram_counter_route.py --analyse "$R"/r3c-g{1,2,4,16,64}.json --out "$R/summary.json" > "$S/logs/analyse.log" 2>&1
echo "analyse rc=$? $(date -u +%H:%M:%S)" >> ~/run_counters.status
echo DONE >> ~/run_counters.status
