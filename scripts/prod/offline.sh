#!/usr/bin/env bash
# AFTER THE MAIN RENTAL, ON THE LAPTOP (review 1, MAJOR 7): the CPU work the rental does not pay for. Frontier's
# trained estimators predict every call (one batch per model: frontier_op.py _predict) and score_prod.py scores
# v8 (at the board's sag par), the roofline and Frontier against the in-serving and replay truths.
#
#   git fetch origin <run branch> && git worktree add ../run <run branch>      # the VM pushed vm/results/...
#   FR_PY_OFFLINE=<a python with Frontier db5a85b on its path, pandas and the trainer's scikit-learn> \
#   FRONTIER_SRC=<Frontier db5a85b checkout> \
#   bash scripts/prod/offline.sh --results ../run/vm/results/prod-main-<utc> [--from STEP] [--workers 4] [--dry-run]
#
# RESUME: `--from STEP` re-enters at STEP; the steps before it are SKIPPED-RESUME (a hard one's declared output, such
# as expected-outputs' runs.txt, must be present; kitlib.sh).
# Memory-bounded for an 18 GB laptop: --workers defaults to 4 (each worker clears v8's per-call caches every
# r3_timing_model_v8.CACHE_EVERY prefill calls). The predict step refuses a scikit-learn other than the trainer's
# (versions.json beside each model). Writes <results>/frontier/predictions.jsonl and <results>/score/; its own
# SUMMARY.txt and step logs go to <results>/offline-logs/ (created; a fetched tree needs no logs/ of its own).
#
# THE FETCHED TREE IS CHECKED FIRST (review 2, MAJOR A): every <f>.xz the push compressed is written out as <f>
# (decompress), and expected-outputs FAILS (hard) unless the tree holds everything the rental measured, by the
# rental's own SUMMARY.txt: A1710 and A1410, every secondary run whose extract PASSED, the board par, both replays and
# every Frontier model set whose train / simpath PASSED. A run expected but missing stops here, never scored around.
# Calls are read across their shards (prodlib.read_calls / load_counts). Every step has its wall cap (MAJOR C).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/kitlib.sh"

RES=""; FROM=""; WORKERS="${PROD_WORKERS:-4}"; MODEL=ernie-4.5-21b-a3b
ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RES="$2"; KIT_RESULTS_GIVEN=1; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dry-run) KIT_DRY=1; shift ;;
    *) echo "FAIL offline: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RES" ]] || { echo "FAIL offline: --results DIR (the main rental's results directory)" >&2; exit 2; }
R="$RES"
SC="$REPO/scripts/prod"
PY="${PY_SCORE:-python3}"
FPY="${FR_PY_OFFLINE:-python3}"
FSRC="${FRONTIER_SRC:-}"
export PYTHONPATH="$REPO:$REPO/scripts:$SC${PYTHONPATH:+:$PYTHONPATH}"
# the step runner's summary and logs go to <results>/offline-logs/{SUMMARY.txt,logs/} (kit_init creates them)
kit_init "$R/offline-logs"
kit_resume_init "$FROM" ${ARGS[@]+"${ARGS[@]}"}
cd "$REPO"
step decompress --cap 3600 -- "$PY" "$SC/prodlib.py" decompress --root "$R"
step expected-outputs --cap 600 --out "$R/offline-logs/runs.txt" -- "$PY" "$SC/prodlib.py" expected-outputs --results "$R" --runs-out "$R/offline-logs/runs.txt"
LABELS=(A1710 A1410 Adefault Aon Beager)
if [[ $KIT_DRY -eq 0 ]]; then
  LABELS=()
  while IFS= read -r lab; do [[ -n "$lab" ]] && LABELS+=("$lab"); done < "$R/offline-logs/runs.txt"
fi
CALLS=()
RUNS=()
for lab in ${LABELS[@]+"${LABELS[@]}"}; do
  CALLS+=(--calls "$lab=$R/$lab")
  RUNS+=(--run "$lab=$R/$lab")
done
CSVNAME="compute/gh200/ernie45-21b-a3b-moe/moe_kernel_only.csv"
predict() {   # predict LABEL MODELS_DIR ARM_LABEL (a model set the rental dropped is absent: expected-outputs vouched)
  local lab="$1" models="$2" arm="$3" d="$R/frontier/$1"
  [[ $KIT_DRY -eq 1 || -d "$models" ]] || { echo "SKIP $arm: no $models (the rental dropped it)"; return 0; }
  step "frontier-predict-$arm" --soft --cap 7200 --out "$d/predictions-$arm.jsonl" -- env PYTHONPATH="$FSRC:$PYTHONPATH" "$FPY" "$SC/frontier_op.py" _predict \
    --model "$MODEL" --arm-label "$arm" --models "$models" --csv "$d/profile/$CSVNAME" ${CALLS[@]+"${CALLS[@]}"} \
    --out "$d/predictions-$arm.jsonl"
}
for lab in primary-uniform variant-all4; do
  predict "$lab" "$R/frontier/$lab/models" "$lab"
  predict "$lab" "$R/frontier/$lab/models-simpath" "$lab-simpath"
done
step frontier-collect --cap 600 --out "$R/frontier/predictions.jsonl" -- bash -c "cat '$R'/frontier/*/predictions-*.jsonl > '$R/frontier/predictions.jsonl' 2>/dev/null || : > '$R/frontier/predictions.jsonl'"
step score --cap 21600 -- "$PY" "$SC/score_prod.py" --model "$MODEL" --sag "$R/sag/par.json" ${RUNS[@]+"${RUNS[@]}"} \
  --replay decode="$R/replay-decode/private_weight_reference" --replay prefill="$R/replay-prefill/private_weight_reference" \
  --frontier "$R/frontier/predictions.jsonl" --workers "$WORKERS" --out "$R/score"
kit_summary
