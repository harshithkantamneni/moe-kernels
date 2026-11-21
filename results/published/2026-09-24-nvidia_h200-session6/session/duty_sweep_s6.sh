#!/bin/bash
# Session 6 same-card duty sweep (owner-approved plan, 2026-09-24): R3 at G=1 and G=16, duty 0.5 and 1.0, seed 0.
# Duty 1.0 pages are controls (V7 expected to fail at full duty); they are not claims.
set -u
cd /workspace/moe-kernels
export MOE_RESULTS_DIR=/workspace/results/gaps-nvidia_h200
for d in 0.5 1.0; do for g in 1 16; do
  log=/workspace/session/alpha_g-nvidia_h200-20260924T042526Z/chain-logs/duty-sweep-g${g}-d${d}.log
  echo "== g=$g duty=$d start $(date -u +%H:%M:%S)" >> /workspace/session/alpha_g-nvidia_h200-20260924T042526Z/DUTY-SWEEP.tsv
  timeout --signal=INT --kill-after=60 1800 /workspace/venvs/vllm/bin/python scripts/private_weight_reference.py --model mixtral-8x7b --block-m 32 --treads 6 --repeats 9 --group-m $g --duty $d --seed 0 --session-tag alpha_g-nvidia_h200-20260924T042526Z > $log 2>&1
  echo "g=$g duty=$d rc=$? end $(date -u +%H:%M:%S) log=$log" >> /workspace/session/alpha_g-nvidia_h200-20260924T042526Z/DUTY-SWEEP.tsv
done; done
echo DONE >> /workspace/session/alpha_g-nvidia_h200-20260924T042526Z/DUTY-SWEEP.tsv
