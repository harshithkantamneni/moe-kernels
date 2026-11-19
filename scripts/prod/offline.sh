#!/usr/bin/env bash
# AFTER THE MAIN RENTAL, ON THE LAPTOP (review 1, MAJOR 7): the score. Frontier's trained estimators already predicted
# every call on the VM (owner decision 2026-11-19: main.sh frontier-predict, one batch per model, written to
# <results>/frontier/predictions/: index.json + npy); score_prod.py scores v8 (at the board's sag par), the roofline and Frontier
# against the in-serving and replay truths, with the replay-vs-serving deltas with and without the API path's fills
# (main.sh replay-offset-check-<branch>: replay_check.json).
#
#   git fetch origin <run branch> && git worktree add ../run <run branch>      # the VM pushed vm/results/...
#   bash scripts/prod/offline.sh --results ../run/vm/results/prod-main-<utc> [--from STEP] [--workers 4] [--dry-run]
#
# RESUME: `--from STEP` re-enters at STEP; the steps before it are SKIPPED-RESUME (a hard one's declared output, such
# as expected-outputs' runs.txt, must be present; kitlib.sh).
# Memory-bounded for an 18 GB laptop: --workers defaults to 4 (each worker clears v8's per-call caches every
# r3_timing_model_v8.CACHE_EVERY prefill calls). Writes <results>/score/; its own SUMMARY.txt and step logs go to
# <results>/offline-logs/ (created; a fetched tree needs no logs/ of its own).
#
# THE FETCHED TREE IS CHECKED FIRST (review 2, MAJOR A): every <f>.xz the push compressed is written out as <f>
# (decompress), and expected-outputs FAILS (hard) unless the tree holds everything the rental measured, by the
# rental's own SUMMARY.txt: A1710 and A1410, every secondary run whose extract PASSED, the board par, both replays,
# Frontier's predictions when frontier-predict PASSED, and each replay_check.json whose check PASSED. A run expected
# but missing stops here, never scored around. Calls are read across their shards (prodlib.read_calls / load_counts).
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
RUNS=()
for lab in ${LABELS[@]+"${LABELS[@]}"}; do
  RUNS+=(--run "$lab=$R/$lab")
done
FRONT=()
if [[ $KIT_DRY -eq 1 || -s "$R/frontier/predictions/index.json" ]]; then FRONT=(--frontier "$R/frontier/predictions"); fi
RCHK=()
for b in decode prefill; do
  if [[ $KIT_DRY -eq 1 || -f "$R/replay-offset-$b/replay_check.json" ]]; then
    RCHK+=(--replay-check "$b=$R/replay-offset-$b/replay_check.json")
  fi
done
step score --cap 21600 -- "$PY" "$SC/score_prod.py" --model "$MODEL" --sag "$R/sag/par.json" ${RUNS[@]+"${RUNS[@]}"} \
  --replay decode="$R/replay-decode/private_weight_reference" --replay prefill="$R/replay-prefill/private_weight_reference" \
  ${RCHK[@]+"${RCHK[@]}"} ${FRONT[@]+"${FRONT[@]}"} --workers "$WORKERS" --out "$R/score"
kit_summary
