#!/usr/bin/env bash
# THE MAIN RENTAL (CRITIQUE-DISPOSITION RENTALS (2)): ERNIE-4.5-21B-A3B on one GH200, staged fail-fast. Measurements,
# Frontier's training and its batch predictions run here; the score runs on the laptop afterwards
# (scripts/prod/offline.sh). Run on the VM after setup_vm.sh, from the pushed, registered checkout:
#
#   . ~/moe/env.sh && cd "$REPO" && bash scripts/prod/main.sh [--results DIR [--from STEP]] [--graphmode-overhead]
#       [--dry-run]
#
# RESUME: `--results DIR --from STEP` re-enters the run in DIR at STEP: the steps before it are SKIPPED-RESUME and
# never re-run (a hard one whose declared output is missing REFUSES the resume); the --always steps (env-preflight,
# env, power, locks, unlocks) run again, so a capture is never resumed at the wrong clock (kitlib.sh).
# OPT-IN: --graphmode-overhead (a 60 s capture under --cuda-graph-trace=graph: the node-tracing overhead, which the
# smoke measured). The Frontier fidelity and all-distributions arms are gone (owner decision 2026-11-19: Frontier's
# profiler cannot run on the GH200; its predictor is trained on our profiles, scripts/prod/frontier_op.py).
# DURATION: the dry run prints every step's expected seconds and the total (scripts/prod/plan_time.py: the smoke
# rental's measured step seconds, scaled to ERNIE); the default plan is <= 4.0 h wall. README "main.sh: every step".
#
# ORDER (review 1, MAJOR 5: the primary measurements first, the secondary runs last and SOFT; each step PASS/FAIL with
# its reason; a hard FAIL stops the rental's work; the card is unlocked on any exit, trap):
#   M0  env-preflight (scripts/prod/env_preflight.py: clean checkout at the expected commit, GPU idle, 700 W
#       enforced, 1710/1410 settable and restorable, TMPDIR, ninja/nsys/ncu, nsys temp dir, the HF pin resolvable,
#       Frontier at its pin, disk, setup_vm READY); env (vLLM 0.27.1, GH200, nsys), power limit read back, default
#       clocks; the model's HF revision pinned (prodlib.HF_REVISIONS); the smoke's replay wall times present
#       (scripts/prod/replay_wall.json, the budget); price-check: one real decode and one real prefill call through v8
#   M1  prompts: the PINNED ShareGPT file and ERNIE's tokenizer at its pinned revision (no --allow-unpinned)
#   M2  THE 60 s ERNIE GATE, before any long run: graphs + nsys node tracing at default clocks, then the gates
#       G-REC, G-GEMM, G-PAIR (marker gridY == layer + 1), G-STEP, G-IDS, G-GRID, G-CLASS, G-BACKEND, G-CFG, G-REV,
#       and check-sequences (the per-path kernel sequences, the naive path's one fill)
#   M3  sag1410: the board's 128x128 sag par at 1410 MHz by the v7 prefill blind's own units and finaliser
#       (scripts/prod/sag1410.sh -> $R/sag/par.json; review 1, BLOCKER 2): a FAIL stops
#   M4  A1710: run A (graphs, nsys node tracing, shared-experts stream OFF) under the 1710 MHz lock -> decode branch
#   M5  A1410: the same under the 1410 MHz lock -> the prefill branch (T > 128)                 (CRITIQUE B1)
#       THE SERVING WORKLOAD (smoke 2026-11-19, the 4.0 h plan): levels 1,8,24,48,80,112,256,1024x3 at N 8 per level
#       (quota max(N, 2L)); the estimator gives every tile >= 30 steps (G16: the three 1024 bursts)
#   M6  replay C: the registered stratified subsample, scaled to REPLAY_BUDGET_S: per branch a probe priced by the
#       smoke's walls x ERNIE's expert-bytes ratio, then the rest priced by the probe's own ERNIE walls, under
#       locked_r3 at the branch clock
#   M7  Frontier, ITS PREDICTOR ON OUR PROFILES (owner decision 2026-11-19; frontier_op.py module doc): Frontier's
#       venv (primary arm only); its token grid (to 16384) with its round-robin routing timed by the replay harness
#       under nsys at the branch clock in pages (decode: one page; prefill: 16 interleaved pages in bit-reversed
#       order, run while the FRONTIER BUDGET lasts), the per-cell kernel-sum in Frontier's CSV schema, then
#       Frontier's own trainer (its CLI) and hold-out, and the simulator-path model. All soft.
#   M8  SECONDARY, every one SOFT: Adefault (default clocks: production delta), Aon (shared-experts stream on at
#       1710: overlap delta), Beager (eager + torch profiler, one level: independent tooling), the replay under
#       nsys per branch at its branch clock under locked_r3 (SMOKE-FIXES 7: the instrument offset and the API
#       path's fill kernels, nsys_extract replay-check), then Frontier's batch predictions of every run's calls
# A push to the run's branch (soft) follows M2, M5, M6, M7 and M8; the last hard step (held-back-check) FAILS when the
# push held back a file the laptop needs, and the EXIT trap pushes once more with the exit code (--final).
# CAPS (review 2, MAJOR C): every step carries its registered wall cap (--cap seconds, kitlib.sh `step`), each at
# least 2x its expected time (plan_time.py; a test holds it): a step past its cap is killed and FAILS (soft:
# dropped). A secondary step whose lock failed is SKIPPED, not run at the wrong clock (--needs, MINOR 6).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/kitlib.sh"

RES=""; FROM=""
GRAPHMODE=0
ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RES="$2"; KIT_RESULTS_GIVEN=1; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --graphmode-overhead) GRAPHMODE=1; shift ;;
    --dry-run) KIT_DRY=1; shift ;;
    *) echo "FAIL main: unknown option $1" >&2; exit 2 ;;
  esac
done
MODEL=ernie-4.5-21b-a3b
TOK=baidu/ERNIE-4.5-21B-A3B-PT
LEVELS="${PROD_LEVELS:-1,8,24,48,80,112,256,1024x3}"
NREQ="${PROD_REQUESTS_PER_LEVEL:-8}"
#: Frontier's profile pages run while elapsed + the dearest page so far <= this (seconds; plan_time.py prices it)
FR_BUDGET="${PROD_FRONTIER_BUDGET_S:-2700}"
FR_PAGES=16
FR_ROUTING="${PROD_FRONTIER_ROUTING:-balanced}"
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
# Beager: the torch-profiler window is 40 steps of L 112; one level of eager serving holds it (the earlier levels were
# eager warm-up only), stopped feeding after 60 s
BEAGER=("$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency 112
        --requests-per-level "$NREQ" --require-pinned)

echo "main: model $MODEL, levels $LEVELS, N/level $NREQ, results $R, pricer $PRICER, Frontier budget ${FR_BUDGET}s"
if [[ $KIT_DRY -eq 1 ]]; then   # the expected wall time of this plan (README "main.sh: every step")
  "$PY" "$SC/plan_time.py" --levels "$LEVELS" --n "$NREQ" --frontier-budget "$FR_BUDGET" \
    || echo "PLAN (plan_time.py could not run under $PY: needs numpy)"
fi
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

# M7: Frontier, its predictor on our profiles (frontier_op.py module doc). Pages: the replay harness under nsys at
# the branch clock (locked_r3), cell marks on, then the per-cell kernel-sum (nsys_extract replay-cells)
step frontier-setup --soft --cap 3600 --out "$FR/frontier.env" -- bash "$SC/frontier_setup.sh" --dest "$FR" --arms primary
if [[ $KIT_DRY -eq 0 && -f "$FR/frontier.env" ]]; then . "$FR/frontier.env"; fi
FSRC="${FRONTIER_SRC:-$FR/src}"
FD="$R/frontier"
CSV="$FD/profile/$("$PY" -c "import sys; sys.path[:0]=['$SC']; import frontier_op as F; print(F.csv_relpath('$MODEL'))" 2>/dev/null \
      || echo compute/gh200/ernie45-21b-a3b-moe/moe_kernel_only.csv)"
step frontier-grid-check --soft --cap 300 --needs frontier-setup -- "$PY" "$SC/frontier_op.py" grid-check --model $MODEL --src "$FSRC" \
  --max-tokens 16384
export R MODEL SC PY NSYS FR_PAGES FR_ROUTING
fr_page() {   # fr_page BRANCH PAGE: one page of Frontier's points, timed at the branch clock under nsys
  local b="$1" pg="$2" lock=1710 reps=3 d
  if [[ $b == prefill ]]; then lock=1410; reps=1; fi
  d="$R/frontier/points/$b-$(printf %02d "$pg")"
  mkdir -p "$d"
  "$PY" "$SC/frontier_op.py" fsample --model "$MODEL" --branch "$b" --page "$pg" --pages "$FR_PAGES" \
    --routing "$FR_ROUTING" --max-tokens 16384 --out "$d" || return 3
  "$NSYS" profile -t cuda --cuda-graph-trace=node --force-overwrite true -o "$d/page" \
    env MOE_RESULTS_DIR="$d" python3 scripts/locked_r3.py --session-tag "prod-frontier-$b-$pg" --locks "$lock" \
    --groups 1 --slip-policy cell --min-clean-repeats 0 --out-dir "$d/locked" --r3-script "$SC/replay.py" -- \
    --model "$MODEL" --histogram "$d/sample.json" --branch "$b" --inst GRR --repeats "$reps" --duty 0.25 --seed 0 \
    --cell-marks || return 3
  "$NSYS" export --type sqlite --force-overwrite true -o "$d/page.sqlite" "$d/page.nsys-rep" || return 3
  "$PY" "$SC/nsys_extract.py" replay-cells --sample "$d/sample.json" --replay-sqlite "$d/page.sqlite" \
    --model "$MODEL" --branch "$b" --out "$d/cells.json"
}
FR_T0=""; FR_MAX=0
fr_page_step() {   # fr_page_step BRANCH PAGE CAP: the page while the Frontier budget lasts (elapsed + dearest page)
  local b="$1" pg="$2" cap="$3" name t dt
  name="frontier-page-$b-$(printf %02d "$pg")"
  if [[ $KIT_DRY -eq 0 && $KIT_RESUMING -eq 0 && -n "$FR_T0" ]] \
     && (( $(date +%s) - FR_T0 + FR_MAX > FR_BUDGET )); then
    KIT_BAD+="$name "
    kit_record "SKIPPED $name: the Frontier profile budget (${FR_BUDGET}s): $(( $(date +%s) - FR_T0 ))s spent, dearest page ${FR_MAX}s"
    return 0
  fi
  t=$(date +%s); [[ -n "$FR_T0" ]] || FR_T0=$t
  step "$name" --soft --cap "$cap" --needs frontier-setup --out "$R/frontier/points/$b-$(printf %02d "$pg")/cells.json" -- fr_page "$b" "$pg"
  dt=$(( $(date +%s) - t )); if (( dt > FR_MAX )); then FR_MAX=$dt; fi
}
fr_page_step decode 0 1200
for pg in $("$PY" -c "import sys; sys.path[:0]=['$SC']; import frontier_op as F; print(*F.page_order($FR_PAGES))" 2>/dev/null \
            || echo 0 8 4 12 2 10 6 14 1 9 5 13 3 11 7 15); do
  fr_page_step prefill "$pg" 1800
done
step frontier-points --soft --cap 600 --out "$FD/points.json" -- "$PY" "$SC/frontier_op.py" fpoints --model $MODEL --pages-root "$FD/points" \
  --max-tokens 16384 --out "$FD/points.json"
step frontier-csv --soft --cap 600 --needs frontier-points --out "$CSV" -- "$PY" "$SC/frontier_op.py" csv --model $MODEL --points "$FD/points.json" \
  --src "$FSRC" --out "$FD/profile"
step frontier-train --soft --cap 1800 --needs frontier-csv --out "$FD/models" -- "$PY" "$SC/frontier_op.py" train --model $MODEL --csv "$CSV" \
  --src "$FSRC" --out "$FD/models"
step frontier-holdout --soft --cap 1800 --needs frontier-csv --out "$FD/holdout" -- "$PY" "$SC/frontier_op.py" holdout --model $MODEL --csv "$CSV" \
  --src "$FSRC" --out "$FD/holdout"
step frontier-simpath --soft --cap 1800 --needs frontier-csv --out "$FD/models-simpath" -- "$PY" "$SC/frontier_op.py" simpath --model $MODEL \
  --arm-label simpath --csv "$CSV" --src "$FSRC" --out "$FD/models-simpath"
step push-frontier --soft --cap 3600 -- kit_push "prod main: Frontier profiles and models"

# M8: the secondary runs (soft: a failure is reported and dropped, the primary data stands)
serve_A Adefault 0 off --soft
serve_A Aon 1710 on --soft
step Beager-capture --soft --cap 1800 --out "$R/Beager/torch_trace" -- "${BEAGER[@]}" --mode eager --torch-profiler 112:40 --shared-stream off \
  --max-seconds 60 --out "$R/Beager"
step Beager-extract --soft --cap 1800 --needs Beager-capture --out "$R/Beager/calls.jsonl|$R/Beager/calls.shards.json" -- "$PY" "$SC/nsys_extract.py" --run "$R/Beager" \
  --torch-trace "$R/Beager/torch_trace" --model $MODEL
# the replay under nsys per branch, at its branch clock under locked_r3 like replay C (SMOKE-FIXES 7), cell marks on:
# the instrument offset, and replay-check: the core sequence and grids against serving's, the API path's fills
for b in decode prefill; do
  lock=1710; src=A1710
  if [[ $b == prefill ]]; then lock=1410; src=A1410; fi
  step "replay-offset-nsys-$b" --soft --cap 2400 --out "$R/replay-offset-$b/replay.nsys-rep" -- "$NSYS" profile -t cuda --cuda-graph-trace=node \
    --force-overwrite true -o "$R/replay-offset-$b/replay" env MOE_RESULTS_DIR="$R/replay-offset-$b" python3 scripts/locked_r3.py \
    --session-tag "prod-replay-offset-$b" --locks $lock --groups 1 --slip-policy cell --min-clean-repeats 0 \
    --out-dir "$R/locked-replay-offset-$b" --r3-script "$SC/replay.py" -- --model $MODEL --histogram "$R/sample-$b/sample.json" \
    --branch $b --inst GRR --repeats 1 --duty 0.25 --seed 0 --cell-marks
  step "replay-offset-export-$b" --soft --cap 1800 --needs "replay-offset-nsys-$b" --out "$R/replay-offset-$b/replay.sqlite" -- "$NSYS" export --type sqlite \
    --force-overwrite true -o "$R/replay-offset-$b/replay.sqlite" "$R/replay-offset-$b/replay.nsys-rep"
  step "replay-offset-check-$b" --soft --cap 1800 --needs "replay-offset-export-$b" --out "$R/replay-offset-$b/replay_check.json" -- "$PY" "$SC/nsys_extract.py" \
    replay-check --serving "$R/$src" --sample "$R/sample-$b/sample.json" --replay-sqlite "$R/replay-offset-$b/replay.sqlite" \
    --model $MODEL --branch $b --out "$R/replay-offset-$b/replay_check.json"
done
if [[ $GRAPHMODE -eq 1 ]]; then   # opt-in: the node-tracing overhead (the smoke measured it), default clocks
  step graphmode-unlock --soft --always --cap 60 -- kit_unlock
  step graphmode-capture --soft --cap 3600 --needs graphmode-unlock --out "$R/graphmode/serve.nsys-rep" -- "$NSYS" profile \
    -t cuda,nvtx --cuda-graph-trace=graph --force-overwrite true -o "$R/graphmode/serve" "${SERVE[@]}" --mode graphs \
    --shared-stream off --min-steps-per-tile 0 --max-seconds 60 --out "$R/graphmode"
fi
step replay-offset-unlock --always --cap 60 -- kit_unlock
# Frontier's batch predictions on the VM (owner decision 2026-11-19): every run whose calls were extracted, ONE
# DataFrame per model (the CLI model and the simulator-path model), written compact to $R/frontier/predictions/ (index.json + npy)
FCALLS=()
for lab in A1710 A1410 Adefault Aon Beager; do
  if [[ $KIT_DRY -eq 1 || -e "$R/$lab/calls.jsonl" || -e "$R/$lab/calls.shards.json" ]]; then FCALLS+=(--calls "$lab=$R/$lab"); fi
done
FMODELS=()
for m in cli=models simpath=models-simpath; do
  # a model set counts once its trainer PASSED (frontier_op.py writes versions.json only then)
  if [[ $KIT_DRY -eq 1 || -f "$FD/${m#*=}/versions.json" ]]; then FMODELS+=(--models "${m%%=*}=$FD/${m#*=}"); fi
done
step frontier-predict --soft --cap 3600 --needs frontier-train --out "$FD/predictions/index.json" -- "$PY" "$SC/frontier_op.py" predict --model $MODEL \
  ${FMODELS[@]+"${FMODELS[@]}"} --csv "$CSV" ${FCALLS[@]+"${FCALLS[@]}"} --src "$FSRC" --out "$FD/predictions"
step push-final --soft --cap 3600 -- kit_push "prod main: measurements and Frontier predictions done (score on the laptop: scripts/prod/offline.sh)"
# the push left every needed file of this run on the branch (else: copy the listed files off the VM before the
# terminate); a FAIL here is the rental's last word, and the EXIT trap still pushes with the exit code
step held-back-check --cap 120 -- "$PY" "$SC/prodlib.py" held-back-check \
  --held "${MOE_PUSH_HOME:-$HOME/moe}/push/HELD-BACK.txt" --results "$R"
kit_summary
