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
FR="${MOE_HOME:-$HOME/moe}/frontier"
SC="$REPO/scripts/prod"
export PYTHONPATH="$REPO:$REPO/scripts:$SC${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

echo "smoke: model $MODEL, results $R, nsys $NSYS, pricer $PRICER"
step env -- kit_env_check
step power-readback --soft -- kit_power_check "${PROD_POWER_W:-700}"
step unlock-start -- kit_unlock

# prompts: the dataset's sha256 is printed here and pinned in prompts.py before the kit freezes
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

# the node-tracing overhead: same load, --cuda-graph-trace=graph (step intervals compared in the summary)
step capture-graphmode-overhead --soft -- "$NSYS" profile -t cuda,nvtx --cuda-graph-trace=graph --force-overwrite true \
  -o "$R/A_graphmode/serve" "$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" \
  --concurrency 1,8,48,112,256 --requests-per-level 16 --mode graphs --shared-stream off --min-steps-per-tile 0 \
  --max-seconds 60 --out "$R/A_graphmode"

# B: eager, vLLM's torch profiler around 20 steps of L=8 (independent tooling)
step capture-eager-torchprof -- "$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency 8,48 \
  --requests-per-level 16 --mode eager --torch-profiler 8:20 --min-steps-per-tile 0 --max-seconds 90 --out "$R/B"
step extract-gate-B -- "$PY" "$SC/nsys_extract.py" --run "$R/B" --torch-trace "$R/B/torch_trace" --model $MODEL

# C: replay of 20 calls with the study's instrument, real ids, at each branch's lock (locked_r3 owns the lock)
step replay-sample -- "$PY" "$SC/replay.py" sample --run "$R/A" --model $MODEL --n 20 --out "$R/sample"
step replay-plan -- kit_dry_plan "$PY" "$SC/replay.py" --model $MODEL --histogram "$R/sample/sample.json" --branch decode --dry-run
step replay-decode -- env MOE_RESULTS_DIR="$R/replay" python3 scripts/locked_r3.py --session-tag smoke-replay-decode \
  --locks 1710 --groups 1 --slip-policy cell --min-clean-repeats 0 --out-dir "$R/locked-replay-decode" \
  --r3-script "$SC/replay.py" -- --model $MODEL --histogram "$R/sample/sample.json" --branch decode --inst GRR \
  --repeats 3 --duty 0.25 --seed 0
step replay-prefill --soft -- env MOE_RESULTS_DIR="$R/replay" python3 scripts/locked_r3.py --session-tag smoke-replay-prefill \
  --locks 1410 --groups 1 --slip-policy cell --min-clean-repeats 0 --out-dir "$R/locked-replay-prefill" \
  --r3-script "$SC/replay.py" -- --model $MODEL --histogram "$R/sample/sample.json" --branch prefill --inst GRR \
  --repeats 3 --duty 0.25 --seed 0
step unlock-after-replay -- kit_unlock

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
  --calls Adefault="$R/A" --sample "$R/sample/sample.json" --src "${FRONTIER_SRC:-$FR/src}" --out "$R/frontier/predictions.jsonl"

# score end to end (the smoke's run A is at default clocks: labelled Adefault, a production delta, never model error)
step score -- "$PY" "$SC/score_prod.py" --model $MODEL --pricer "$PRICER" --run Adefault="$R/A" --run Beager="$R/B" \
  --sample-run decode=Adefault \
  --replay decode="$R/replay" \
  --frontier "$R/frontier/predictions.jsonl" --workers 4 --out "$R/score"
kit_summary
