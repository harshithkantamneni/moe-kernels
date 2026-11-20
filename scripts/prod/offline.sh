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
#
# THE EXTRAS (scripts/prod/extras.sh, the same rental after main.sh): when <results>/extras/SUMMARY.txt exists (or
# with --extras, the dry run's way to show them), extras-expected-outputs (soft) holds every extras step that PASSED to
# its output, and the same report gets, each labelled and DESCRIPTIVE: X1 the noise floor (--noise-floor
# A1710=extras/A1710r), X2 the counted DRAM read bytes against v8's predicted reads (--ncu-bytes, --ncu-sample), X4 the
# long-context run (--run LC1410); then X3: score-qwen3 (this scorer with --model qwen3-30b-a3b on extras/qwen3: its
# runs, its replay, its Frontier predictions, the board's par from main's sag1410: the same board) and score-merge,
# which appends that score to <results>/score/prod.score.{json,txt} under "extras qwen3".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/kitlib.sh"

RES=""; FROM=""; WORKERS="${PROD_WORKERS:-4}"; MODEL=ernie-4.5-21b-a3b; EXTRAS=0
ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RES="$2"; KIT_RESULTS_GIVEN=1; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --extras) EXTRAS=1; shift ;;
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
EX="$R/extras"
XARGS=()
if [[ $EXTRAS -eq 1 || -f "$EX/SUMMARY.txt" ]]; then
  EXTRAS=1
  step extras-expected-outputs --soft --cap 600 -- "$PY" "$SC/extras_lib.py" expected-outputs --results "$R" \
    --out "$R/offline-logs/extras.json"
  if [[ $KIT_DRY -eq 1 || -e "$EX/A1710r/calls.jsonl" || -e "$EX/A1710r/calls.shards.json" ]]; then
    XARGS+=(--noise-floor "A1710=$EX/A1710r")
  fi
  NCU=0
  for b in decode prefill; do
    if [[ $KIT_DRY -eq 1 || -f "$EX/ncu-$b/ncu_bytes.json" ]]; then XARGS+=(--ncu-bytes "$EX/ncu-$b/ncu_bytes.json"); NCU=1; fi
  done
  if [[ $NCU -eq 1 ]]; then XARGS+=(--ncu-sample "$EX/ncu-sample/sample.json"); fi
  if [[ $KIT_DRY -eq 1 || -e "$EX/LC1410/calls.jsonl" || -e "$EX/LC1410/calls.shards.json" ]]; then
    RUNS+=(--run "LC1410=$EX/LC1410")
  fi
fi
step score --cap 21600 -- "$PY" "$SC/score_prod.py" --model "$MODEL" --sag "$R/sag/par.json" ${RUNS[@]+"${RUNS[@]}"} \
  --replay decode="$R/replay-decode/private_weight_reference" --replay prefill="$R/replay-prefill/private_weight_reference" \
  ${RCHK[@]+"${RCHK[@]}"} ${FRONT[@]+"${FRONT[@]}"} ${XARGS[@]+"${XARGS[@]}"} --workers "$WORKERS" --out "$R/score"
if [[ $EXTRAS -eq 1 ]]; then   # X3: Qwen3-30B-A3B, every predictor, then appended to the same report
  Q="$EX/qwen3"
  QRUNS=(); QREP=(); QFR=()
  for lab in A1710 A1410; do
    if [[ $KIT_DRY -eq 1 || -e "$Q/$lab/calls.jsonl" || -e "$Q/$lab/calls.shards.json" ]]; then QRUNS+=(--run "$lab=$Q/$lab"); fi
  done
  for b in decode prefill; do
    if [[ $KIT_DRY -eq 1 ]] || find "$Q/replay-$b" -name 'cells.csv*' 2>/dev/null | grep -q .; then
      QREP+=(--replay "$b=$Q/replay-$b/private_weight_reference")
    fi
  done
  if [[ $KIT_DRY -eq 1 || -s "$Q/frontier/predictions/index.json" ]]; then QFR=(--frontier "$Q/frontier/predictions"); fi
  if [[ $KIT_DRY -eq 1 || ${#QRUNS[@]} -gt 0 ]]; then
    step score-qwen3 --soft --cap 21600 -- "$PY" "$SC/score_prod.py" --model qwen3-30b-a3b --sag "$R/sag/par.json" \
      ${QRUNS[@]+"${QRUNS[@]}"} ${QREP[@]+"${QREP[@]}"} ${QFR[@]+"${QFR[@]}"} --workers "$WORKERS" --out "$R/score/qwen3"
    step score-merge --soft --cap 600 --needs score-qwen3 -- "$PY" "$SC/score_prod.py" merge --into "$R/score" \
      --extra "qwen3=$R/score/qwen3"
  fi
fi
kit_summary
