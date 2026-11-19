#!/usr/bin/env bash
# AFTER THE MAIN RENTAL, ON THE LAPTOP (review 1, MAJOR 7): the CPU work the rental does not pay for. Frontier's
# trained estimators predict every call (one batch per model: frontier_op.py _predict) and score_prod.py scores
# v8 (at the board's sag par), the roofline and Frontier against the in-serving and replay truths.
#
#   git fetch origin <run branch> && git worktree add ../run <run branch>      # the VM pushed vm/results/...
#   FR_PY_OFFLINE=<a python with Frontier db5a85b on its path, pandas and the trainer's scikit-learn> \
#   FRONTIER_SRC=<Frontier db5a85b checkout> \
#   bash scripts/prod/offline.sh --results ../run/vm/results/prod-main-<utc> [--workers 4] [--dry-run]
#
# Memory-bounded for an 18 GB laptop: --workers defaults to 4 (each worker clears v8's per-call caches every
# r3_timing_model_v8.CACHE_EVERY prefill calls). The predict step refuses a scikit-learn other than the trainer's
# (versions.json beside each model). Writes <results>/frontier/predictions.jsonl and <results>/score/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/kitlib.sh"

RES=""; WORKERS="${PROD_WORKERS:-4}"; MODEL=ernie-4.5-21b-a3b
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RES="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dry-run) KIT_DRY=1; shift ;;
    *) echo "FAIL offline: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RES" ]] || { echo "FAIL offline: --results DIR (the main rental's results directory)" >&2; exit 2; }
R="$RES"
SUMMARY_DIR="$R/offline-logs"
SC="$REPO/scripts/prod"
PY="${PY_SCORE:-python3}"
FPY="${FR_PY_OFFLINE:-python3}"
FSRC="${FRONTIER_SRC:-}"
export PYTHONPATH="$REPO:$REPO/scripts:$SC${PYTHONPATH:+:$PYTHONPATH}"
if [[ $KIT_DRY -eq 0 ]]; then mkdir -p "$SUMMARY_DIR"; fi
# the step runner's logs go beside the rental's, under offline-logs/
R_SAVE="$R"; kit_init "$SUMMARY_DIR"; R="$R_SAVE"
cd "$REPO"
CALLS=(--calls A1710="$R/A1710" --calls A1410="$R/A1410" --calls Adefault="$R/Adefault" --calls Aon="$R/Aon" --calls Beager="$R/Beager")
CSVNAME="compute/gh200/ernie45-21b-a3b-moe/moe_kernel_only.csv"
predict() {   # predict LABEL MODELS_DIR ARM_LABEL
  local lab="$1" models="$2" arm="$3" d="$R/frontier/$1"
  [[ $KIT_DRY -eq 1 || -d "$models" ]] || { echo "SKIP $arm: no $models (the rental dropped it)"; return 0; }
  local have=()
  for c in "${CALLS[@]}"; do [[ "$c" == --calls ]] && continue; [[ $KIT_DRY -eq 1 || -d "${c#*=}" ]] && have+=(--calls "$c"); done
  step "frontier-predict-$arm" --soft -- env PYTHONPATH="$FSRC:$PYTHONPATH" "$FPY" "$SC/frontier_op.py" _predict \
    --model "$MODEL" --arm-label "$arm" --models "$models" --csv "$d/profile/$CSVNAME" ${have[@]+"${have[@]}"} \
    --out "$d/predictions-$arm.jsonl"
}
for lab in primary-uniform variant-all4; do
  predict "$lab" "$R/frontier/$lab/models" "$lab"
  predict "$lab" "$R/frontier/$lab/models-simpath" "$lab-simpath"
done
step frontier-collect -- bash -c "cat '$R'/frontier/*/predictions-*.jsonl > '$R/frontier/predictions.jsonl' 2>/dev/null || : > '$R/frontier/predictions.jsonl'"
RUNS=()
for lab in A1710 A1410 Adefault Aon Beager; do
  [[ $KIT_DRY -eq 1 || -s "$R/$lab/calls.jsonl" ]] && RUNS+=(--run "$lab=$R/$lab")
done
step score -- "$PY" "$SC/score_prod.py" --model "$MODEL" --sag "$R/sag/par.json" ${RUNS[@]+"${RUNS[@]}"} \
  --replay decode="$R/replay-decode/private_weight_reference" --replay prefill="$R/replay-prefill/private_weight_reference" \
  --frontier "$R/frontier/predictions.jsonl" --workers "$WORKERS" --out "$R/score"
kit_summary
