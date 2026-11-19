#!/usr/bin/env bash
# THE MAIN RENTAL (CRITIQUE-DISPOSITION RENTALS (2)): ERNIE-4.5-21B-A3B on one GH200, staged fail-fast, ~3.5-4 h.
# Run on the VM after setup_vm.sh, from the pushed, registered checkout:
#
#   . ~/moe/env.sh && cd "$REPO" && bash scripts/prod/main.sh [--results DIR] [--skip-frontier-variant] [--dry-run]
#
# ORDER (each step PASS/FAIL with its reason; a hard FAIL stops the rental's work):
#   M0  env (vLLM 0.27.1, GH200, nsys), power limit read back, default clocks
#   M1  prompts: the PINNED ShareGPT file (no --allow-unpinned: an unpinned sha256 stops here), ERNIE's tokenizer
#   M2  THE 60 s ERNIE GATE, before any long run: graphs + nsys node tracing at default clocks, then the gates
#       G-REC (ring records == capture kernels), G-GEMM (fused_moe kernels == 2 x records), G-STEP, G-IDS,
#       G-GRID (nsys gridX == cdiv(EM,BM) npn at the logged padded T), G-CLASS, G-BACKEND (TRITON), G-CFG
#   M3  A1710: run A (graphs, nsys node tracing, shared-experts stream OFF: VLLM_DISABLE_SHARED_EXPERTS_STREAM=1,
#       CRITIQUE M6) under the 1710 MHz lock -> scores the decode branch (T <= 128)
#   M4  A1410: the same under the 1410 MHz lock -> scores the prefill branch (T > 128)       (CRITIQUE B1)
#   M5  Adefault: the same at default clocks (production delta only)
#   M6  Aon: shared-experts stream ON at the 1710 lock (overlap delta)
#   M7  Beager: eager, vLLM's torch profiler around 40 steps of L=112, default clocks (independent tooling)
#   M8  replay C: the registered stratified subsample (decode tiles from A1710, prefill tiles from A1410), each
#       branch timed by the study's GRR instrument under locked_r3 at its branch clock; then one short replay under
#       nsys node tracing (the instrument offset, CRITIQUE M7)
#   M9  Frontier: both stacks; PRIMARY on vLLM 0.27.1 (uniform, to 16384 tokens; train; predict every call and
#       the sample; its own hold-out), the VARIANT (all four load distributions), FIDELITY on its pinned stack
#       (its own hold-out only). A Frontier stack that cannot install or run is reported and dropped (soft).
#   M10 score (DESCRIPTIVE)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/kitlib.sh"

RES=""
VARIANT=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RES="$2"; shift 2 ;;
    --skip-frontier-variant) VARIANT=0; shift ;;
    --dry-run) KIT_DRY=1; shift ;;
    *) echo "FAIL main: unknown option $1" >&2; exit 2 ;;
  esac
done
MODEL=ernie-4.5-21b-a3b
TOK=baidu/ERNIE-4.5-21B-A3B-PT
LEVELS="${PROD_LEVELS:-1,8,24,48,80,112,256,1024x3}"
NREQ="${PROD_REQUESTS_PER_LEVEL:-64}"
PY="${PY_VLLM:-python3}"
PRICER="${PROD_PRICER:-r3_timing_model_v8}"
R="${RES:-${RESULTS_ROOT:-$HOME/moe/results}/prod-main-$(date -u +%Y%m%dT%H%M%SZ)}"
kit_init "$R"
kit_locate_nsys
FR="${MOE_HOME:-$HOME/moe}/frontier"
SC="$REPO/scripts/prod"
export PYTHONPATH="$REPO:$REPO/scripts:$SC${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
SERVE=("$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency "$LEVELS"
       --requests-per-level "$NREQ")
NSYS_A=("$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite true)

echo "main: model $MODEL, levels $LEVELS, N/level $NREQ, results $R, pricer $PRICER"
# M0
step env -- kit_env_check
step power-readback -- kit_power_check "${PROD_POWER_W:-700}"
step unlock-start -- kit_unlock
step pricer-import -- "$PY" -c "import sys; sys.path[:0]=['$REPO/scripts']; import $PRICER; print('PASS pricer', '$PRICER')"
# M1
step prompts-fetch -- "$PY" "$SC/prompts.py" fetch --source sharegpt --dest "$R/data"
step prompts-build -- "$PY" "$SC/prompts.py" build --source sharegpt --data "$R/data/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --tokenizer "$TOK" --n 4096 --seed 0 --max-input 2048 --out-dist sharegpt --out "$R/prompts.jsonl"
step serve-plan -- "${SERVE[@]}" --mode graphs --out "$R/A1710" --dry-run

# M2: the 60 s gate
step gate60-capture -- "${NSYS_A[@]}" -o "$R/gate60/serve" "${SERVE[@]}" --mode graphs --shared-stream off \
  --min-steps-per-tile 0 --max-seconds 60 --out "$R/gate60"
step gate60-export -- "$NSYS" export --type sqlite --force-overwrite true -o "$R/gate60/serve.sqlite" "$R/gate60/serve.nsys-rep"
step gate60-extract -- "$PY" "$SC/nsys_extract.py" --run "$R/gate60" --sqlite "$R/gate60/serve.sqlite" --model $MODEL

serve_A() {   # serve_A LABEL LOCK(0=default) STREAM(on|off)
  local lab="$1" lock="$2" stream="$3"
  if [[ "$lock" != 0 ]]; then step "$lab-lock" -- kit_lock "$lock"; else step "$lab-unlock" -- kit_unlock; fi
  step "$lab-capture" -- "${NSYS_A[@]}" -o "$R/$lab/serve" "${SERVE[@]}" --mode graphs --shared-stream "$stream" --out "$R/$lab"
  step "$lab-unlock-after" -- kit_unlock
  step "$lab-export" -- "$NSYS" export --type sqlite --force-overwrite true -o "$R/$lab/serve.sqlite" "$R/$lab/serve.nsys-rep"
  step "$lab-extract" -- "$PY" "$SC/nsys_extract.py" --run "$R/$lab" --sqlite "$R/$lab/serve.sqlite" --model $MODEL
}
# M3..M6
serve_A A1710 1710 off
serve_A A1410 1410 off
serve_A Adefault 0 off
serve_A Aon 1710 on
# M7
step Beager-capture -- "${SERVE[@]}" --mode eager --torch-profiler 112:40 --shared-stream off --out "$R/Beager"
step Beager-extract -- "$PY" "$SC/nsys_extract.py" --run "$R/Beager" --torch-trace "$R/Beager/torch_trace" --model $MODEL

# M8: replay C (1500 calls registered: decode tiles 4/7, prefill tiles 3/7 of it)
step replay-sample-decode -- "$PY" "$SC/replay.py" sample --run "$R/A1710" --model $MODEL --branch decode --n 857 --out "$R/sample-decode"
step replay-sample-prefill -- "$PY" "$SC/replay.py" sample --run "$R/A1410" --model $MODEL --branch prefill --n 643 --out "$R/sample-prefill"
for b in decode prefill; do
  lock=1710; [[ $b == prefill ]] && lock=1410
  step "replay-$b" -- env MOE_RESULTS_DIR="$R/replay-$b" python3 scripts/locked_r3.py --session-tag "prod-replay-$b" \
    --locks $lock --groups 1 --slip-policy cell --min-clean-repeats 0 --out-dir "$R/locked-replay-$b" \
    --r3-script "$SC/replay.py" -- --model $MODEL --histogram "$R/sample-$b/sample.json" --branch $b --inst GRR \
    --repeats 9 --duty 0.25 --seed 0
done
step replay-offset-lock -- kit_lock 1710
step replay-offset-nsys --soft -- "${NSYS_A[@]}" -o "$R/replay-offset/replay" env MOE_RESULTS_DIR="$R/replay-offset" \
  "$PY" "$SC/replay.py" --model $MODEL --histogram "$R/sample-decode/sample.json" --branch decode --inst GRR --repeats 1 \
  --duty 0.25 --seed 0 --session-tag prod-replay-offset
step replay-offset-unlock -- kit_unlock

# M9: Frontier
step frontier-setup --soft -- bash "$SC/frontier_setup.sh" --dest "$FR" --arms primary,fidelity
if [[ $KIT_DRY -eq 0 && -f "$FR/frontier.env" ]]; then . "$FR/frontier.env"; fi
FSRC="${FRONTIER_SRC:-$FR/src}"
CSVNAME="compute/gh200/ernie45-21b-a3b-moe/moe_kernel_only.csv"
CALLS=(--calls A1710="$R/A1710" --calls A1410="$R/A1410" --calls Adefault="$R/Adefault" --calls Aon="$R/Aon" --calls Beager="$R/Beager")
step frontier-config --soft -- "$PY" "$SC/frontier_op.py" write-config --model $MODEL --src "$FSRC"
fr_arm() {   # fr_arm ARM LABEL DISTS... : profile, train, hold-out, (primary stack) predict
  local arm="$1" lab="$2"; shift 2
  local d="$R/frontier/$lab"
  step "frontier-profile-$lab" --soft -- "$PY" "$SC/frontier_op.py" profile --arm "$arm" --model $MODEL --dists "$@" \
    --src "$FSRC" --out "$d/profile"
  step "frontier-train-$lab" --soft -- "$PY" "$SC/frontier_op.py" train --arm "$arm" --model $MODEL --csv "$d/profile/$CSVNAME" \
    --src "$FSRC" --out "$d/models"
  step "frontier-holdout-$lab" --soft -- "$PY" "$SC/frontier_op.py" holdout --arm "$arm" --model $MODEL --csv "$d/profile/$CSVNAME" \
    --arm-label "$lab" --src "$FSRC" --out "$d/holdout"
  step "frontier-simpath-$lab" --soft -- "$PY" "$SC/frontier_op.py" simpath --arm "$arm" --model $MODEL --csv "$d/profile/$CSVNAME" \
    --src "$FSRC" --out "$d/models-simpath"
  if [[ "$arm" == primary ]]; then
    step "frontier-predict-$lab" --soft -- "$PY" "$SC/frontier_op.py" predict --arm "$arm" --arm-label "$lab" --model $MODEL \
      --models "$d/models" --csv "$d/profile/$CSVNAME" "${CALLS[@]}" --src "$FSRC" --out "$d/predictions.jsonl"
    step "frontier-predict-$lab-simpath" --soft -- "$PY" "$SC/frontier_op.py" predict --arm "$arm" --arm-label "$lab-simpath" \
      --model $MODEL --models "$d/models-simpath" --csv "$d/profile/$CSVNAME" "${CALLS[@]}" --src "$FSRC" \
      --out "$d/predictions-simpath.jsonl"
  fi
}
fr_arm primary primary-uniform uniform
if [[ $VARIANT -eq 1 ]]; then fr_arm primary variant-all4 uniform balanced skewed extremely_skewed; fi
fr_arm fidelity fidelity-uniform uniform
step frontier-collect --soft -- bash -c "cat '$R'/frontier/*/predictions*.jsonl > '$R/frontier/predictions.jsonl'"

# M10
step score -- "$PY" "$SC/score_prod.py" --model $MODEL --pricer "$PRICER" \
  --run A1710="$R/A1710" --run A1410="$R/A1410" --run Adefault="$R/Adefault" --run Aon="$R/Aon" --run Beager="$R/Beager" \
  --replay decode="$R/replay-decode/private_weight_reference" --replay prefill="$R/replay-prefill/private_weight_reference" \
  --frontier "$R/frontier/predictions.jsonl" --workers 32 --out "$R/score"
kit_summary
