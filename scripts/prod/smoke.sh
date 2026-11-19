#!/usr/bin/env bash
# THE SMOKE RENTAL (CRITIQUE-DISPOSITION RENTALS (1)): the whole kit end to end on OLMoE-1B-7B (E 64 like ERNIE,
# already studied, keeps ERNIE blind), ~45 min on one GH200, BEFORE the kit freezes. Run on the VM after
# setup_vm.sh, from the checkout:
#
#   . ~/moe/env.sh && cd "$REPO" && bash scripts/prod/smoke.sh [--results DIR] [--dry-run]
#
# Fail-fast; each step prints PASS/FAIL with the reason; everything lands under one results dir
# (default $RESULTS_ROOT/prod-smoke-<utc>). What each step proves is in scripts/prod/README.md "smoke verifies".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/kitlib.sh"

RES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RES="$2"; shift 2 ;;
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
kit_trap_unlock
FR="${MOE_HOME:-$HOME/moe}/frontier"
SC="$REPO/scripts/prod"
export PYTHONPATH="$REPO:$REPO/scripts:$SC${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

echo "smoke: model $MODEL, results $R, nsys $NSYS, pricer $PRICER"
step env -- kit_env_check
step power-readback --soft -- kit_power_check "${PROD_POWER_W:-700}"
step unlock-start -- kit_unlock
# the kit's contract with v8: one real decode and one real prefill call priced (review 1, BLOCKER 1)
step price-check -- "$PY" "$SC/prodlib.py" price-check --model $MODEL --pricer "$PRICER"

# prompts: the dataset's sha256 is printed here and pinned in prompts.py before the kit freezes; the model and
# tokenizer run unpinned here and serve_capture's G-REV prints the resolved HF commit to pin in prodlib.HF_REVISIONS
step prompts-fetch -- "$PY" "$SC/prompts.py" fetch --source sharegpt --dest "$R/data" --allow-unpinned
step prompts-build -- "$PY" "$SC/prompts.py" build --source sharegpt --data "$R/data/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --tokenizer "$TOK" --n 1024 --seed 0 --max-input 2048 --out-dist sharegpt --allow-unpinned --out "$R/prompts.jsonl"
step serve-plan -- "$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency 1,8,48,112,256 \
  --requests-per-level 16 --mode graphs --min-steps-per-tile 0 --out "$R/A" --dry-run

# A: graphs on, nsys CUDA-graph NODE tracing, ~60 s of serving (hook, graph safety, node tracing on aarch64)
step capture-graphs-nsys -- "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite true -o "$R/A/serve" \
  "$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency 1,8,48,112,256 \
  --requests-per-level 16 --mode graphs --shared-stream off --min-steps-per-tile 0 --max-seconds 60 --out "$R/A"
step nsys-export -- "$NSYS" export --type sqlite --force-overwrite true -o "$R/A/serve.sqlite" "$R/A/serve.nsys-rep"
step extract-gate-A -- "$PY" "$SC/nsys_extract.py" --run "$R/A" --sqlite "$R/A/serve.sqlite" --model $MODEL
# the per-path kernel sequences (the naive path's one fill kernel, two GEMMs, one act, one moe_sum): a gate here
step sequences-A -- "$PY" "$SC/nsys_extract.py" check-sequences --run "$R/A" --model $MODEL

# the node-tracing overhead: same load, --cuda-graph-trace=graph (step intervals compared in the summary)
step capture-graphmode-overhead --soft -- "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=graph --force-overwrite true \
  -o "$R/A_graphmode/serve" "$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" \
  --concurrency 1,8,48,112,256 --requests-per-level 16 --mode graphs --shared-stream off --min-steps-per-tile 0 \
  --max-seconds 60 --out "$R/A_graphmode"

# B: eager, vLLM's torch profiler around 20 steps of L=8 (independent tooling)
step capture-eager-torchprof -- "$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency 8,48 \
  --requests-per-level 16 --mode eager --torch-profiler 8:20 --min-steps-per-tile 0 --max-seconds 90 --out "$R/B"
step extract-gate-B -- "$PY" "$SC/nsys_extract.py" --run "$R/B" --torch-trace "$R/B/torch_trace" --model $MODEL

# the board's 128x128 sag par at 1410: the v7 prefill blind's units and finaliser, as main.sh runs them (BLOCKER 2)
step sag1410 -- bash "$SC/sag1410.sh" --out "$R/sag"
step unlock-after-sag -- kit_unlock

# C: replay with the study's instrument, real ids, at each branch's lock (locked_r3 owns the lock). BOTH branches are
# hard, and the prefill sample holds a T >= 8256 cell (the run's own, else a synthetic T 16384; review 1, MAJOR 6)
step replay-sample-decode -- "$PY" "$SC/replay.py" sample --run "$R/A" --model $MODEL --branch decode --n 12 --out "$R/sample-decode"
step replay-sample-prefill -- "$PY" "$SC/replay.py" sample --run "$R/A" --model $MODEL --branch prefill --n 8 \
  --min-big-T 8256 --out "$R/sample-prefill"
step replay-plan -- kit_dry_plan "$PY" "$SC/replay.py" --model $MODEL --histogram "$R/sample-decode/sample.json" --branch decode --dry-run
for b in decode prefill; do
  lock=1710; [[ $b == prefill ]] && lock=1410
  step "replay-$b" -- env MOE_RESULTS_DIR="$R/replay-$b" python3 scripts/locked_r3.py --session-tag "smoke-replay-$b" \
    --locks $lock --groups 1 --slip-policy cell --min-clean-repeats 0 --out-dir "$R/locked-replay-$b" \
    --r3-script "$SC/replay.py" -- --model $MODEL --histogram "$R/sample-$b/sample.json" --branch $b --inst GRR \
    --repeats 3 --duty 0.25 --seed 0
done
step unlock-after-replay -- kit_unlock
# the wall time per cell per tile (main.sh's replay budget: copy $R/replay_wall.json to scripts/prod/ before freeze)
step replay-wall -- "$PY" "$SC/replay.py" wall-merge --from "$R/replay-decode" --from "$R/replay-prefill" --out "$R/replay_wall.json"
# the replay's kernels against serving's, per tile (names and grids), under nsys (review 1, MINOR)
for b in decode prefill; do
  step "replay-nsys-$b" -- "$NSYS" profile -t cuda --cuda-graph-trace=node --force-overwrite true -o "$R/replay-nsys-$b/replay" \
    env MOE_RESULTS_DIR="$R/replay-nsys-$b" "$PY" "$SC/replay.py" --model $MODEL --histogram "$R/sample-$b/sample.json" \
    --branch $b --inst GRR --repeats 1 --duty 0.25 --seed 0 --session-tag "smoke-replay-nsys-$b"
  step "replay-nsys-export-$b" -- "$NSYS" export --type sqlite --force-overwrite true -o "$R/replay-nsys-$b/replay.sqlite" \
    "$R/replay-nsys-$b/replay.nsys-rep"
  step "replay-check-$b" -- "$PY" "$SC/nsys_extract.py" replay-check --serving "$R/A" --sample "$R/sample-$b/sample.json" \
    --replay-sqlite "$R/replay-nsys-$b/replay.sqlite" --model $MODEL
done

# Frontier: both stacks install, a tiny profile / train / predict / holdout on each
step frontier-setup --soft -- bash "$SC/frontier_setup.sh" --dest "$FR" --arms primary,fidelity
if [[ $KIT_DRY -eq 0 && -f "$FR/frontier.env" ]]; then . "$FR/frontier.env"; fi
for arm in primary fidelity; do
  step "frontier-config-$arm" --soft -- "$PY" "$SC/frontier_op.py" write-config --model $MODEL --src "${FRONTIER_SRC:-$FR/src}"
  step "frontier-profile-$arm" --soft -- "$PY" "$SC/frontier_op.py" profile --arm $arm --model $MODEL --dists uniform \
    --max-tokens 256 --src "${FRONTIER_SRC:-$FR/src}" --out "$R/frontier/$arm/profile"
  CSV="$R/frontier/$arm/profile/compute/gh200/olmoe-1b-7b-moe/moe_kernel_only.csv"
  step "frontier-train-$arm" --soft -- "$PY" "$SC/frontier_op.py" train --arm $arm --model $MODEL --csv "$CSV" \
    --src "${FRONTIER_SRC:-$FR/src}" --out "$R/frontier/$arm/models"
  step "frontier-holdout-$arm" --soft -- "$PY" "$SC/frontier_op.py" holdout --arm $arm --model $MODEL --csv "$CSV" \
    --src "${FRONTIER_SRC:-$FR/src}" --out "$R/frontier/$arm/holdout"
done
step frontier-predict-primary --soft -- "$PY" "$SC/frontier_op.py" predict --arm primary --model $MODEL \
  --models "$R/frontier/primary/models" --csv "$R/frontier/primary/profile/compute/gh200/olmoe-1b-7b-moe/moe_kernel_only.csv" \
  --calls Adefault="$R/A" --sample "$R/sample-decode/sample.json" --src "${FRONTIER_SRC:-$FR/src}" --out "$R/frontier/predictions.jsonl"

# score end to end (the smoke's run A is at default clocks: labelled Adefault, a production delta, never model error);
# small, so it runs here; main.sh's score runs on the laptop (offline.sh)
step score -- "$PY" "$SC/score_prod.py" --model $MODEL --pricer "$PRICER" --sag "$R/sag/par.json" \
  --run Adefault="$R/A" --run Beager="$R/B" --sample-run decode=Adefault --sample-run prefill=Adefault \
  --replay decode="$R/replay-decode" --replay prefill="$R/replay-prefill" \
  --frontier "$R/frontier/predictions.jsonl" --workers 4 --out "$R/score"
kit_summary
