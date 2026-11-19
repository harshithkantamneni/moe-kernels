#!/usr/bin/env bash
# THE SMOKE RENTAL (CRITIQUE-DISPOSITION RENTALS (1)): the whole kit end to end on OLMoE-1B-7B (E 64 like ERNIE,
# already studied, keeps ERNIE blind), ~45 min on one GH200, BEFORE the kit freezes. Run on the VM after
# setup_vm.sh, from the checkout:
#
#   . ~/moe/env.sh && cd "$REPO" && bash scripts/prod/smoke.sh [--results DIR [--from STEP]] [--dry-run]
#
# RESUME: `--results DIR --from STEP` re-enters the run in DIR at STEP; the steps before it are SKIPPED-RESUME (their
# declared outputs must be in DIR), the --always steps (preflight, env, power, unlocks) run again (kitlib.sh).
# Fail-fast; each step prints PASS/FAIL with the reason; everything lands under one results dir
# (default $RESULTS_ROOT/prod-smoke-<utc>). What each step proves is in scripts/prod/README.md "smoke verifies".
# Every step carries its registered wall cap (--cap seconds; kitlib.sh `step`, review 2 MAJOR C).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/kitlib.sh"

RES=""; FROM=""
ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RES="$2"; KIT_RESULTS_GIVEN=1; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --dry-run) KIT_DRY=1; shift ;;
    *) echo "FAIL smoke: unknown option $1" >&2; exit 2 ;;
  esac
done
MODEL=olmoe-1b-7b
TOK=allenai/OLMoE-1B-7B-0924
PY="${PY_VLLM:-python3}"
PRICER="${PROD_PRICER:-r3_timing_model_v8}"
R="${RES:-${RESULTS_ROOT:-$HOME/moe/results}/prod-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
kit_init "$R"
kit_locate_nsys
kit_env "$PY"
kit_resume_init "$FROM" ${ARGS[@]+"${ARGS[@]}"}
kit_trap_unlock
FR="${MOE_HOME:-$HOME/moe}/frontier"
SC="$REPO/scripts/prod"
export PYTHONPATH="$REPO:$REPO/scripts:$SC${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

echo "smoke: model $MODEL, results $R, nsys $NSYS, pricer $PRICER"
# the environment, all of it, before anything long (smoke rental 2026-11-19: SMOKE-FIXES.md 1-5)
step env-preflight --always --cap 300 -- "$PY" "$SC/env_preflight.py" --repo "$REPO" --model $MODEL --nsys "$NSYS" \
  --power-w "${PROD_POWER_W:-700}" --locks 1710,1410 --tmpdir "$KIT_TMP" --frontier-src "$FR/src" \
  --disk "${MOE_HOME:-$HOME/moe}" --min-free-gb "${PROD_MIN_FREE_GB:-200}" --out "$R/env-preflight.json"
step env --always --cap 300 -- kit_env_check
step power-readback --always --soft --cap 60 -- kit_power_check "${PROD_POWER_W:-700}"
step unlock-start --always --cap 60 -- kit_unlock
# the kit's contract with v8: one real decode and one real prefill call priced (review 1, BLOCKER 1)
step price-check --cap 900 -- "$PY" "$SC/prodlib.py" price-check --model $MODEL --pricer "$PRICER"

# prompts: the dataset's sha256 is printed here and pinned in prompts.py before the kit freezes. The model and its
# tokenizer load at prodlib.HF_REVISIONS (resolved on the laptop by scripts/prod/hf_revisions.py); serve_capture's
# G-REV fails a pinned run whose loaded snapshot differs or cannot be resolved
step prompts-fetch --cap 1800 --out "$R/data/ShareGPT_V3_unfiltered_cleaned_split.json" -- "$PY" "$SC/prompts.py" fetch --source sharegpt --dest "$R/data" --allow-unpinned
step prompts-build --cap 1800 --out "$R/prompts.jsonl" -- "$PY" "$SC/prompts.py" build --source sharegpt --data "$R/data/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --tokenizer "$TOK" --n 1024 --seed 0 --max-input 2048 --out-dist sharegpt --allow-unpinned --out "$R/prompts.jsonl"
step serve-plan --cap 300 -- "$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency 1,8,48,112,256 \
  --requests-per-level 16 --mode graphs --min-steps-per-tile 0 --out "$R/A" --dry-run

# A: graphs on, nsys CUDA-graph NODE tracing, ~60 s of serving (hook, graph safety, node tracing on aarch64)
step capture-graphs-nsys --cap 2400 --out "$R/A/serve.nsys-rep,$R/A/manifest.json" -- "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite true -o "$R/A/serve" \
  "$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency 1,8,48,112,256 \
  --requests-per-level 16 --mode graphs --shared-stream off --min-steps-per-tile 0 --max-seconds 60 --out "$R/A"
step nsys-export --cap 1800 --out "$R/A/serve.sqlite" -- "$NSYS" export --type sqlite --force-overwrite true -o "$R/A/serve.sqlite" "$R/A/serve.nsys-rep"
step extract-gate-A --cap 1800 --out "$R/A/calls.jsonl|$R/A/calls.shards.json" -- "$PY" "$SC/nsys_extract.py" --run "$R/A" --sqlite "$R/A/serve.sqlite" --model $MODEL
# the per-path kernel sequences (the naive path's one fill kernel, two GEMMs, one act, one moe_sum): a gate here
step sequences-A --cap 600 -- "$PY" "$SC/nsys_extract.py" check-sequences --run "$R/A" --model $MODEL

# the node-tracing overhead: same load, --cuda-graph-trace=graph (step intervals compared in the summary)
step capture-graphmode-overhead --soft --cap 2400 --out "$R/A_graphmode/serve.nsys-rep" -- "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=graph --force-overwrite true \
  -o "$R/A_graphmode/serve" "$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" \
  --concurrency 1,8,48,112,256 --requests-per-level 16 --mode graphs --shared-stream off --min-steps-per-tile 0 \
  --max-seconds 60 --out "$R/A_graphmode"

# B: eager, vLLM's torch profiler around 20 steps of L=8 (independent tooling)
step capture-eager-torchprof --cap 2400 --out "$R/B/torch_trace" -- "$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency 8,48 \
  --requests-per-level 16 --mode eager --torch-profiler 8:20 --min-steps-per-tile 0 --max-seconds 90 --out "$R/B"
step extract-gate-B --cap 1800 --out "$R/B/calls.jsonl|$R/B/calls.shards.json" -- "$PY" "$SC/nsys_extract.py" --run "$R/B" --torch-trace "$R/B/torch_trace" --model $MODEL

# the board's 128x128 sag par at 1410: the v7 prefill blind's units and finaliser, as main.sh runs them (BLOCKER 2)
step sag1410 --cap 5400 --out "$R/sag/par.json" -- bash "$SC/sag1410.sh" --out "$R/sag"
step unlock-after-sag --always --cap 60 -- kit_unlock

# C: replay with the study's instrument, real ids, at each branch's lock (locked_r3 owns the lock). BOTH branches are
# hard, and the prefill sample holds a T >= 8256 cell (the run's own, else a synthetic T 16384; review 1, MAJOR 6)
step replay-sample-decode --cap 600 --out "$R/sample-decode/sample.json" -- "$PY" "$SC/replay.py" sample --run "$R/A" --model $MODEL --branch decode --n 12 --out "$R/sample-decode"
step replay-sample-prefill --cap 600 --out "$R/sample-prefill/sample.json" -- "$PY" "$SC/replay.py" sample --run "$R/A" --model $MODEL --branch prefill --n 8 \
  --min-big-T 8256 --out "$R/sample-prefill"
step replay-plan --cap 600 -- kit_dry_plan "$PY" "$SC/replay.py" --model $MODEL --histogram "$R/sample-decode/sample.json" --branch decode --dry-run
for b in decode prefill; do
  lock=1710; [[ $b == prefill ]] && lock=1410
  step "replay-$b" --cap 3600 --out "$R/replay-$b" -- env MOE_RESULTS_DIR="$R/replay-$b" python3 scripts/locked_r3.py --session-tag "smoke-replay-$b" \
    --locks $lock --groups 1 --slip-policy cell --min-clean-repeats 0 --out-dir "$R/locked-replay-$b" \
    --r3-script "$SC/replay.py" -- --model $MODEL --histogram "$R/sample-$b/sample.json" --branch $b --inst GRR \
    --repeats 3 --duty 0.25 --seed 0
done
step unlock-after-replay --always --cap 60 -- kit_unlock
# the wall time per cell per tile (main.sh's replay budget: copy $R/replay_wall.json to scripts/prod/ before freeze)
step replay-wall --cap 120 --out "$R/replay_wall.json" -- "$PY" "$SC/replay.py" wall-merge --from "$R/replay-decode" --from "$R/replay-prefill" \
  --model $MODEL --out "$R/replay_wall.json"
# the replay's kernels against serving's, per tile (the CORE sequence align..moe_sum and the grids; the API path's
# fill kernels counted, not compared: SMOKE-FIXES 6), under nsys AT THE BRANCH CLOCK under locked_r3 like replay C
# (SMOKE-FIXES 7: the instrument offset at replay C's clock), cell marks on (each cell's own calls)
for b in decode prefill; do
  lock=1710; [[ $b == prefill ]] && lock=1410
  step "replay-nsys-$b" --cap 3600 --out "$R/replay-nsys-$b/replay.nsys-rep" -- "$NSYS" profile -t cuda --cuda-graph-trace=node --force-overwrite true -o "$R/replay-nsys-$b/replay" \
    env MOE_RESULTS_DIR="$R/replay-nsys-$b" python3 scripts/locked_r3.py --session-tag "smoke-replay-nsys-$b" --locks $lock --groups 1 \
    --slip-policy cell --min-clean-repeats 0 --out-dir "$R/locked-replay-nsys-$b" --r3-script "$SC/replay.py" -- --model $MODEL \
    --histogram "$R/sample-$b/sample.json" --branch $b --inst GRR --repeats 1 --duty 0.25 --seed 0 --cell-marks
  step "replay-nsys-export-$b" --cap 1800 --out "$R/replay-nsys-$b/replay.sqlite" -- "$NSYS" export --type sqlite --force-overwrite true -o "$R/replay-nsys-$b/replay.sqlite" \
    "$R/replay-nsys-$b/replay.nsys-rep"
  step "replay-check-$b" --cap 1800 --out "$R/replay-nsys-$b/replay_check.json" -- "$PY" "$SC/nsys_extract.py" replay-check --serving "$R/A" --sample "$R/sample-$b/sample.json" \
    --replay-sqlite "$R/replay-nsys-$b/replay.sqlite" --model $MODEL --branch $b --out "$R/replay-nsys-$b/replay_check.json"
done

# Frontier, its predictor on our profiles (frontier_op.py; owner decision 2026-11-19), small: its grid to 256 tokens,
# the timer's two branch passes (every point), its trainer, hold-out and batch predictions on the VM
step frontier-setup --soft --cap 3600 --out "$FR/frontier.env" -- bash "$SC/frontier_setup.sh" --dest "$FR" --arms primary
if [[ $KIT_DRY -eq 0 && -f "$FR/frontier.env" ]]; then . "$FR/frontier.env"; fi
FSRC="${FRONTIER_SRC:-$FR/src}"
FD="$R/frontier"
CSV="$FD/profile/compute/gh200/olmoe-1b-7b-moe/moe_kernel_only.csv"
step frontier-grid-check --soft --cap 300 --needs frontier-setup -- "$PY" "$SC/frontier_op.py" grid-check --model $MODEL --src "$FSRC" --max-tokens 256
export R MODEL SC PY NSYS FD
fr_timer() {   # fr_timer BRANCH LOCK: every grid point of the branch (to 256), one process, eager, under nsys
  local d="$FD/timer-$1"
  mkdir -p "$d"
  "$NSYS" profile -t cuda --force-overwrite true -o "$d/timer" "$PY" "$SC/frontier_op.py" timer --model "$MODEL" \
    --branch "$1" --lock "$2" --max-tokens 256 --out "$d"
}
for b in decode prefill; do
  lock=1710; [[ $b == prefill ]] && lock=1410
  d="$FD/timer-$b"
  step "frontier-timer-lock-$b" --soft --always --cap 120 -- kit_lock "$lock"
  step "frontier-timer-$b" --soft --cap 1800 --needs "frontier-timer-lock-$b" --out "$d/timer.json,$d/timer.nsys-rep" -- fr_timer "$b" "$lock"
  step "frontier-timer-export-$b" --soft --cap 900 --needs "frontier-timer-$b" --out "$d/timer.sqlite" -- "$NSYS" export --type sqlite --force-overwrite true \
    -o "$d/timer.sqlite" "$d/timer.nsys-rep"
done
step frontier-timer-unlock --always --cap 60 -- kit_unlock
step frontier-points --soft --cap 300 --out "$FD/points.json" -- "$PY" "$SC/frontier_op.py" fpoints --model $MODEL --timer "$FD/timer-decode" \
  --timer "$FD/timer-prefill" --max-tokens 256 --out "$FD/points.json"
step frontier-csv --soft --cap 300 --needs frontier-points --out "$CSV" -- "$PY" "$SC/frontier_op.py" csv --model $MODEL --points "$FD/points.json" --src "$FSRC" \
  --out "$FD/profile"
step frontier-train --soft --cap 1800 --needs frontier-csv --out "$FD/models" -- "$PY" "$SC/frontier_op.py" train --model $MODEL --csv "$CSV" --src "$FSRC" \
  --out "$FD/models"
step frontier-holdout --soft --cap 1800 --needs frontier-csv --out "$FD/holdout" -- "$PY" "$SC/frontier_op.py" holdout --model $MODEL --csv "$CSV" --src "$FSRC" \
  --out "$FD/holdout"
step frontier-simpath --soft --cap 1800 --needs frontier-csv --out "$FD/models-simpath" -- "$PY" "$SC/frontier_op.py" simpath --model $MODEL --arm-label simpath \
  --csv "$CSV" --src "$FSRC" --out "$FD/models-simpath"
step frontier-predict --soft --cap 1800 --needs frontier-train --out "$FD/predictions/index.json" -- "$PY" "$SC/frontier_op.py" predict --model $MODEL \
  --models cli="$FD/models" --models simpath="$FD/models-simpath" --csv "$CSV" --calls Adefault="$R/A" --sample "$R/sample-decode/sample.json" \
  --src "$FSRC" --out "$FD/predictions"

# score end to end (the smoke's run A is at default clocks: labelled Adefault, a production delta, never model error);
# small, so it runs here; main.sh's score runs on the laptop (offline.sh)
step score --cap 3600 --out "$R/score" -- "$PY" "$SC/score_prod.py" --model $MODEL --pricer "$PRICER" --sag "$R/sag/par.json" \
  --run Adefault="$R/A" --run Beager="$R/B" --sample-run decode=Adefault --sample-run prefill=Adefault \
  --replay decode="$R/replay-decode" --replay prefill="$R/replay-prefill" \
  --replay-check decode="$R/replay-nsys-decode/replay_check.json" --replay-check prefill="$R/replay-nsys-prefill/replay_check.json" \
  --frontier "$R/frontier/predictions" --workers 4 --out "$R/score"
kit_summary
