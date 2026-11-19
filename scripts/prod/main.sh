#!/usr/bin/env bash
# THE MAIN RENTAL (CRITIQUE-DISPOSITION RENTALS (2)): ERNIE-4.5-21B-A3B on one GH200, staged fail-fast. MEASUREMENTS
# ONLY: Frontier's predictions and every score run on the laptop afterwards (scripts/prod/offline.sh; review 1,
# MAJOR 7). Run on the VM after setup_vm.sh, from the pushed, registered checkout:
#
#   . ~/moe/env.sh && cd "$REPO" && bash scripts/prod/main.sh [--results DIR [--from STEP]] [--frontier-fidelity]
#       [--frontier-all-dists] [--graphmode-overhead] [--dry-run]
#
# RESUME: `--results DIR --from STEP` re-enters the run in DIR at STEP: the steps before it are SKIPPED-RESUME and
# never re-run (a hard one whose declared output is missing REFUSES the resume); the --always steps (env-preflight,
# env, power, locks, unlocks) run again, so a capture is never resumed at the wrong clock (kitlib.sh).
# OPT-IN (smoke rental 2026-11-19, efficiency; none is a registered primary measurement): --frontier-fidelity
# (Frontier's own pinned stack, hold-out only), --frontier-all-dists (the variant: all four load distributions;
# default uniform only), --graphmode-overhead (a 60 s capture under --cuda-graph-trace=graph: the node-tracing
# overhead, which the smoke measures). Every step, its cap and its expected time: scripts/prod/README.md.
#
# ORDER (review 1, MAJOR 5: the primary measurements first, the secondary runs last and SOFT; each step PASS/FAIL with
# its reason; a hard FAIL stops the rental's work; the card is unlocked on any exit, trap):
#   M0  env-preflight (scripts/prod/env_preflight.py: clean checkout at the expected commit, GPU idle, 700 W
#       enforced, 1710/1410 settable and restorable, TMPDIR, ninja/nsys/ncu, nsys temp dir, the HF pin resolvable,
#       Frontier at its pin, disk, setup_vm READY); env (vLLM 0.27.1, GH200, nsys), power limit read back, default clocks; the model's HF revision pinned
#       (prodlib.HF_REVISIONS); the smoke's replay wall times present (scripts/prod/replay_wall.json, the budget);
#       price-check: the kit prices one real decode and one real prefill call through v8 (review 1, BLOCKER 1)
#   M1  prompts: the PINNED ShareGPT file and ERNIE's tokenizer at its pinned revision (no --allow-unpinned)
#   M2  THE 60 s ERNIE GATE, before any long run: graphs + nsys node tracing at default clocks, then the gates
#       G-REC, G-GEMM, G-PAIR (marker gridY == layer + 1), G-STEP, G-IDS, G-GRID, G-CLASS, G-BACKEND, G-CFG, G-REV,
#       and check-sequences (the per-path kernel sequences, the naive path's one fill)
#   M3  sag1410: the board's 128x128 sag par at 1410 MHz by the v7 prefill blind's own units and finaliser
#       (scripts/prod/sag1410.sh -> $R/sag/par.json; review 1, BLOCKER 2): a FAIL stops (the prefill branch has no
#       par without it)
#   M4  A1710: run A (graphs, nsys node tracing, shared-experts stream OFF: VLLM_DISABLE_SHARED_EXPERTS_STREAM=1,
#       CRITIQUE M6) under the 1710 MHz lock -> the decode branch (T <= 128)
#   M5  A1410: the same under the 1410 MHz lock -> the prefill branch (T > 128)                 (CRITIQUE B1)
#   M6  replay C: the registered stratified subsample, scaled to REPLAY_BUDGET_S: per branch a probe priced by the
#       smoke's walls x ERNIE's expert-bytes ratio, then the rest priced by the probe's own ERNIE walls (replay.py
#       budget rule), decode tiles from A1710 at 1710, prefill tiles from A1410 at 1410, under locked_r3
#   M7  Frontier: PRIMARY on vLLM 0.27.1 (uniform, to 16384 tokens) profile + train + hold-out (+ the simpath
#       model); opt-in: the VARIANT (all four load distributions, --frontier-all-dists) and FIDELITY on its pinned
#       stack (hold-out only, --frontier-fidelity).
#       Soft: a stack that cannot install or run is reported and dropped. NO predict here (offline.sh)
#   M8  SECONDARY, every one SOFT: Adefault (default clocks: production delta), Aon (shared-experts stream on at
#       1710: overlap delta), Beager (eager + torch profiler: independent tooling), the replay under nsys (the
#       instrument offset)
# A push to the run's branch (soft) follows M2, M5, M6, M7 and M8; the last hard step (held-back-check) FAILS when the
# push held back a file the laptop needs, and the EXIT trap pushes once more with the exit code (--final), whatever
# stopped the sequence (review 2, MAJOR A).
# CAPS (review 2, MAJOR C): every step carries its registered wall cap (--cap seconds, kitlib.sh `step`): a step past
# its cap is killed and FAILS (soft: dropped), so no hung step holds the rental. A secondary step whose lock failed
# is SKIPPED, not run at the wrong clock (--needs, MINOR 6).
# REPLAY RE-MEASURE (review 2, MAJOR B): per branch a probe of replay.PROBE_PER_TILE cells per tile (priced by the
# smoke's OLMoE walls scaled by ERNIE's expert bytes) is replayed first; its own ERNIE walls size the rest, inside the
# same fixed REPLAY_BUDGET_S (2400 s) per branch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/kitlib.sh"

RES=""; FROM=""
VARIANT=0; FIDELITY=0; GRAPHMODE=0
ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RES="$2"; KIT_RESULTS_GIVEN=1; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --frontier-all-dists) VARIANT=1; shift ;;
    --frontier-fidelity) FIDELITY=1; shift ;;
    --graphmode-overhead) GRAPHMODE=1; shift ;;
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
WALL="${PROD_REPLAY_WALL:-$HERE/replay_wall.json}"
R="${RES:-${RESULTS_ROOT:-$HOME/moe/results}/prod-main-$(date -u +%Y%m%dT%H%M%SZ)}"
kit_init "$R"
kit_locate_nsys
kit_env "$PY"
kit_resume_init "$FROM" ${ARGS[@]+"${ARGS[@]}"}
kit_trap_unlock --push
FR="${MOE_HOME:-$HOME/moe}/frontier"
SC="$REPO/scripts/prod"
export PYTHONPATH="$REPO:$REPO/scripts:$SC${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
SERVE=("$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency "$LEVELS"
       --requests-per-level "$NREQ" --require-pinned)
NSYS_A=("$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite true)

echo "main: model $MODEL, levels $LEVELS, N/level $NREQ, results $R, pricer $PRICER"
# M0: the whole environment first (smoke rental 2026-11-19: SMOKE-FIXES.md 1-5), fail-fast with its reasons
step env-preflight --always --cap 300 -- "$PY" "$SC/env_preflight.py" --repo "$REPO" --model $MODEL --nsys "$NSYS" \
  --power-w "${PROD_POWER_W:-700}" --locks 1710,1410 --tmpdir "$KIT_TMP" --frontier-src "$FR/src" \
  --disk "${MOE_HOME:-$HOME/moe}" --min-free-gb "${PROD_MIN_FREE_GB:-200}" --out "$R/env-preflight.json"
step env --always --cap 300 -- kit_env_check
step power-readback --always --cap 60 -- kit_power_check "${PROD_POWER_W:-700}"
step unlock-start --always --cap 60 -- kit_unlock
step revision-pinned --always --cap 120 -- "$PY" -c "import sys; sys.path[:0]=['$SC']; import prodlib as P; r=P.hf_revision('$MODEL'); print(('PASS revision ' + r) if r else 'FAIL revision: $MODEL is not pinned in prodlib.HF_REVISIONS'); sys.exit(0 if r else 3)"
step replay-wall-present --always --cap 30 -- test -s "$WALL"
step price-check --cap 900 -- "$PY" "$SC/prodlib.py" price-check --model $MODEL --pricer "$PRICER"
# M1
step prompts-fetch --cap 1800 --out "$R/data/ShareGPT_V3_unfiltered_cleaned_split.json" -- "$PY" "$SC/prompts.py" fetch --source sharegpt --dest "$R/data"
step prompts-build --cap 1800 --out "$R/prompts.jsonl" -- "$PY" "$SC/prompts.py" build --source sharegpt --data "$R/data/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --tokenizer "$TOK" --n 4096 --seed 0 --max-input 2048 --out-dist sharegpt --out "$R/prompts.jsonl"
step serve-plan --cap 300 -- "${SERVE[@]}" --mode graphs --out "$R/A1710" --dry-run

# M2: the 60 s gate (the first ERNIE load: the weights download here)
step gate60-capture --cap 3600 --out "$R/gate60/serve.nsys-rep,$R/gate60/manifest.json" -- "${NSYS_A[@]}" -o "$R/gate60/serve" "${SERVE[@]}" --mode graphs --shared-stream off \
  --min-steps-per-tile 0 --max-seconds 60 --out "$R/gate60"
step gate60-export --cap 1800 --out "$R/gate60/serve.sqlite" -- "$NSYS" export --type sqlite --force-overwrite true -o "$R/gate60/serve.sqlite" "$R/gate60/serve.nsys-rep"
step gate60-extract --cap 1800 --out "$R/gate60/calls.jsonl|$R/gate60/calls.shards.json" -- "$PY" "$SC/nsys_extract.py" --run "$R/gate60" --sqlite "$R/gate60/serve.sqlite" --model $MODEL
step gate60-sequences --cap 600 -- "$PY" "$SC/nsys_extract.py" check-sequences --run "$R/gate60" --model $MODEL
step push-gate60 --soft --cap 3600 -- kit_push "prod main: the 60 s ERNIE gate"

# M3: the board's sag par at 1410 (the v7 prefill blind's units and finaliser; the driver's own deadline is 75 min)
step sag1410 --cap 5400 --out "$R/sag/par.json" -- bash "$SC/sag1410.sh" --out "$R/sag"
step unlock-after-sag --always --cap 60 -- kit_unlock

serve_A() {   # serve_A LABEL LOCK(0=default) STREAM(on|off) [--soft]
  local lab="$1" lock="$2" stream="$3" soft=("${@:4}") pre
  if [[ "$lock" != 0 ]]; then
    pre="$lab-lock"; step "$pre" ${soft[@]+"${soft[@]}"} --always --cap 120 -- kit_lock "$lock"
  else
    pre="$lab-unlock"; step "$pre" ${soft[@]+"${soft[@]}"} --always --cap 60 -- kit_unlock
  fi
  step "$lab-capture" ${soft[@]+"${soft[@]}"} --cap 10800 --needs "$pre" --out "$R/$lab/serve.nsys-rep,$R/$lab/manifest.json" -- "${NSYS_A[@]}" -o "$R/$lab/serve" "${SERVE[@]}" \
    --mode graphs --shared-stream "$stream" --out "$R/$lab"
  step "$lab-unlock-after" --always --cap 60 -- kit_unlock
  step "$lab-export" ${soft[@]+"${soft[@]}"} --cap 5400 --needs "$lab-capture" --out "$R/$lab/serve.sqlite" -- "$NSYS" export --type sqlite \
    --force-overwrite true -o "$R/$lab/serve.sqlite" "$R/$lab/serve.nsys-rep"
  step "$lab-extract" ${soft[@]+"${soft[@]}"} --cap 3600 --needs "$lab-export" \
    --out "$R/$lab/calls.jsonl|$R/$lab/calls.shards.json" -- "$PY" "$SC/nsys_extract.py" --run "$R/$lab" \
    --sqlite "$R/$lab/serve.sqlite" --model $MODEL
}
# M4, M5: the primary runs (hard)
serve_A A1710 1710 off
serve_A A1410 1410 off
step push-primary --soft --cap 3600 -- kit_push "prod main: A1710 and A1410"

# M6: replay C (1500 calls registered: decode tiles 4/7, prefill tiles 3/7), per branch a probe, then the rest sized by
# the probe's own ERNIE walls inside the branch's fixed budget (replay.py module doc, MAJOR B)
for b in decode prefill; do
  lock=1710; src=A1710; n=857
  if [[ $b == prefill ]]; then lock=1410; src=A1410; n=643; fi
  replay_cell() {   # replay_cell STEP OUTDIR SAMPLE TAG CAP
    step "$1" --cap "$5" --out "$2" -- env MOE_RESULTS_DIR="$2" python3 scripts/locked_r3.py --session-tag "$4" \
      --locks $lock --groups 1 --slip-policy cell --min-clean-repeats 0 --out-dir "$R/locked-$(basename "$2")" \
      --r3-script "$SC/replay.py" -- --model $MODEL --histogram "$3" --branch $b --inst GRR \
      --repeats 9 --duty 0.25 --seed 0
  }
  step "replay-probe-sample-$b" --cap 600 --out "$R/sample-probe-$b/sample.json" -- "$PY" "$SC/replay.py" sample --run "$R/$src" --model $MODEL --branch $b \
    --n $n --per-tile "${PROD_PROBE_PER_TILE:-2}" --wall "$WALL" --repeats 9 --out "$R/sample-probe-$b"
  replay_cell "replay-probe-$b" "$R/replay-probe-$b" "$R/sample-probe-$b/sample.json" "prod-replay-probe-$b" 1800
  step "replay-probe-wall-$b" --cap 120 --out "$R/replay-probe-$b/replay_wall.json" -- "$PY" "$SC/replay.py" wall-merge --from "$R/replay-probe-$b" --model $MODEL \
    --out "$R/replay-probe-$b/replay_wall.json"
  step "replay-sample-$b" --cap 600 --out "$R/sample-$b/sample.json" -- "$PY" "$SC/replay.py" sample --run "$R/$src" --model $MODEL --branch $b --n $n \
    --probe-wall "$R/replay-probe-$b/replay_wall.json" --repeats 9 --out "$R/sample-$b"
  replay_cell "replay-$b" "$R/replay-$b" "$R/sample-$b/sample.json" "prod-replay-$b" 5400
done
step unlock-after-replay --always --cap 60 -- kit_unlock
step push-replay --soft --cap 3600 -- kit_push "prod main: replay C"

# M7: Frontier (profile, train, hold-out on the rental; predict on the laptop)
FARMS=primary; [[ $FIDELITY -eq 1 ]] && FARMS=primary,fidelity
step frontier-setup --soft --cap 3600 --out "$FR/frontier.env" -- bash "$SC/frontier_setup.sh" --dest "$FR" --arms "$FARMS"
if [[ $KIT_DRY -eq 0 && -f "$FR/frontier.env" ]]; then . "$FR/frontier.env"; fi
FSRC="${FRONTIER_SRC:-$FR/src}"
CSVNAME="compute/gh200/ernie45-21b-a3b-moe/moe_kernel_only.csv"
step frontier-config --soft --cap 300 --needs frontier-setup -- "$PY" "$SC/frontier_op.py" write-config --model $MODEL --src "$FSRC"
fr_arm() {   # fr_arm ARM LABEL DISTS... : profile, train, hold-out, simpath (no predict: offline.sh)
  local arm="$1" lab="$2"; shift 2
  local d="$R/frontier/$lab"
  step "frontier-profile-$lab" --soft --cap 10800 --needs frontier-config --out "$d/profile/$CSVNAME" -- "$PY" "$SC/frontier_op.py" profile --arm "$arm" \
    --model $MODEL --dists "$@" --src "$FSRC" --out "$d/profile"
  step "frontier-train-$lab" --soft --cap 3600 --needs "frontier-profile-$lab" --out "$d/models" -- "$PY" "$SC/frontier_op.py" train --arm "$arm" \
    --model $MODEL --csv "$d/profile/$CSVNAME" --src "$FSRC" --out "$d/models"
  step "frontier-holdout-$lab" --soft --cap 1800 --needs "frontier-profile-$lab" --out "$d/holdout" -- "$PY" "$SC/frontier_op.py" holdout \
    --arm "$arm" --model $MODEL --csv "$d/profile/$CSVNAME" --arm-label "$lab" --src "$FSRC" --out "$d/holdout"
  if [[ "$arm" == primary ]]; then
    step "frontier-simpath-$lab" --soft --cap 3600 --needs "frontier-profile-$lab" --out "$d/models-simpath" -- "$PY" "$SC/frontier_op.py" simpath \
      --arm "$arm" --model $MODEL --csv "$d/profile/$CSVNAME" --src "$FSRC" --out "$d/models-simpath"
  fi
}
fr_arm primary primary-uniform uniform
if [[ $VARIANT -eq 1 ]]; then fr_arm primary variant-all4 uniform balanced skewed extremely_skewed; fi
if [[ $FIDELITY -eq 1 ]]; then fr_arm fidelity fidelity-uniform uniform; fi
step push-frontier --soft --cap 3600 -- kit_push "prod main: Frontier profiles and models"

# M8: the secondary runs (soft: a failure is reported and dropped, the primary data stands)
serve_A Adefault 0 off --soft
serve_A Aon 1710 on --soft
step Beager-capture --soft --cap 5400 --out "$R/Beager/torch_trace" -- "${SERVE[@]}" --mode eager --torch-profiler 112:40 --shared-stream off --out "$R/Beager"
step Beager-extract --soft --cap 3600 --needs Beager-capture --out "$R/Beager/calls.jsonl|$R/Beager/calls.shards.json" -- "$PY" "$SC/nsys_extract.py" --run "$R/Beager" \
  --torch-trace "$R/Beager/torch_trace" --model $MODEL
step replay-offset-lock --soft --always --cap 120 -- kit_lock 1710
step replay-offset-nsys --soft --cap 3600 --needs replay-offset-lock --out "$R/replay-offset/replay.nsys-rep" -- "${NSYS_A[@]}" -o "$R/replay-offset/replay" \
  env MOE_RESULTS_DIR="$R/replay-offset" "$PY" "$SC/replay.py" --model $MODEL --histogram "$R/sample-decode/sample.json" \
  --branch decode --inst GRR --repeats 1 --duty 0.25 --seed 0 --session-tag prod-replay-offset
if [[ $GRAPHMODE -eq 1 ]]; then   # opt-in: the node-tracing overhead (the smoke measures it), default clocks
  step graphmode-unlock --soft --always --cap 60 -- kit_unlock
  step graphmode-capture --soft --cap 3600 --needs graphmode-unlock --out "$R/graphmode/serve.nsys-rep" -- "$NSYS" profile \
    -t cuda,nvtx --cuda-graph-trace=graph --force-overwrite true -o "$R/graphmode/serve" "${SERVE[@]}" --mode graphs \
    --shared-stream off --min-steps-per-tile 0 --max-seconds 60 --out "$R/graphmode"
fi
step replay-offset-unlock --always --cap 60 -- kit_unlock
step push-final --soft --cap 3600 -- kit_push "prod main: measurements done (score on the laptop: scripts/prod/offline.sh)"
# the push left every needed file of this run on the branch (else: copy the listed files off the VM before the
# terminate); a FAIL here is the rental's last word, and the EXIT trap still pushes with the exit code
step held-back-check --cap 120 -- "$PY" "$SC/prodlib.py" held-back-check \
  --held "${MOE_PUSH_HOME:-$HOME/moe}/push/HELD-BACK.txt" --results "$R"
kit_summary
