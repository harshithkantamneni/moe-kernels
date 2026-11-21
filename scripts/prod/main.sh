#!/usr/bin/env bash
# THE MAIN RENTAL (CRITIQUE-DISPOSITION RENTALS (2)): ERNIE-4.5-21B-A3B on one GH200, staged fail-fast. Measurements,
# Frontier's training and its batch predictions run here; the score runs on the laptop afterwards
# (scripts/prod/offline.sh). Run on the VM after setup_vm.sh, from the pushed, registered checkout:
#
#   . ~/moe/env.sh && cd "$REPO" && bash scripts/prod/main.sh [--results DIR [--from STEP]] [--graphmode-overhead]
#       [--frontier-balanced-arm] [--model KEY] [--profile full|blind] [--utc STAMP] [--dry-run]
#
# MODEL (2026-11-20): --model KEY (default ernie-4.5-21b-a3b) is any prodlib.MODELS key, the blind registry
# (prodlib.BLIND_SPECS) included; TOK, the serving levels, the Frontier probe points and its CSV path come from
# `prodlib.py model-info`. PROFILE: full (default, the sequence below) or blind (scripts/prod/blind.sh: ONE new model
# per 1x GH200, lean, ~1.2 h, scored on the VM by v8, v9 and Frontier). --utc names the instance: blind results go
# to prod-<model>-<utc> and the launcher's run branch is run-gh200-blind-<model>-<utc> (`prodlib.py run-names`).
#
# RESUME: `--results DIR --from STEP` re-enters the run in DIR at STEP: the steps before it are SKIPPED-RESUME and
# never re-run (a hard one whose declared output is missing REFUSES the resume); the --always steps (env-preflight,
# env, power, locks, unlocks) run again, so a capture is never resumed at the wrong clock (kitlib.sh).
# OPT-IN: --graphmode-overhead (a 60 s capture under --cuda-graph-trace=graph: the node-tracing overhead, which the
# smoke measured); --frontier-balanced-arm (Frontier's round-robin "balanced" routing as a labelled second arm,
# cli-balanced: two more timer passes, its CSV and its trainer). The Frontier fidelity and all-distributions arms are gone (owner decision 2026-11-19: Frontier's
# profiler cannot run on the GH200; its predictor is trained on our profiles, scripts/prod/frontier_op.py).
# DURATION: the dry run prints every step's expected seconds and the total (scripts/prod/plan_time.py: the smoke
# rental's measured step seconds, scaled to ERNIE); the default plan is <= 4.0 h wall. README "main.sh: every step".
#
# ORDER (review 1, MAJOR 5: the primary measurements first, the secondary runs last and SOFT; each step PASS/FAIL with
# its reason; a hard FAIL stops the rental's work; the card is unlocked on any exit, trap):
#   M0  env-preflight (cap 600 s; scripts/prod/env_preflight.py: clean checkout at the expected commit, GPU idle,
#       700 W enforced, 1710/1410 settable and restorable, TMPDIR, ninja/nsys/ncu, nsys temp dir, the HF pin
#       resolvable, the HF download rate (>= 50 MB/s, measured: ERNIE's weights priced), Frontier at its pin, disk,
#       setup_vm READY); env (vLLM 0.27.1, GH200, nsys), power limit read back, default
#       clocks; the model's HF revision pinned (prodlib.HF_REVISIONS); the smoke's replay wall times present
#       (scripts/prod/replay_wall.json, the budget); price-check: one real decode and one real prefill call through v8
#   M1  prompts: the PINNED ShareGPT file and ERNIE's tokenizer at its pinned revision (no --allow-unpinned)
#   M2  THE 60 s ERNIE GATE, before any long run: graphs + nsys node tracing at default clocks, then the gates
#       G-REC, G-GEMM, G-PAIR (marker gridY == layer + 1), G-STEP, G-IDS, G-GRID, G-CLASS, G-BACKEND, G-CFG, G-REV,
#       and check-sequences (the per-path kernel sequences, the naive path's one fill); then THE EARLY FRONTIER
#       PROBE (hard, ~5 min): Frontier's venv and its grid-check, then `frontier-timer-probe`: the new timer on 3 ERNIE
#       grid points (one decode T, two prefill T, frontier_op.PROBE_POINTS) at their branch clocks, then fpoints, the
#       CSV, Frontier's own trainer (a narrow grid search) and its predictions on those points: a failure stops the
#       run before the expensive blocks
#   M3  sag1410: the board's 128x128 sag par at 1410 MHz by the v7 prefill blind's own units and finaliser
#       (scripts/prod/sag1410.sh -> $R/sag/par.json; review 1, BLOCKER 2): a FAIL stops
#   M4  A1710: run A (graphs, nsys node tracing, shared-experts stream OFF) under the 1710 MHz lock -> decode branch
#   M5  A1410: the same under the 1410 MHz lock -> the prefill branch (T > 128)                 (CRITIQUE B1)
#       THE SERVING WORKLOAD (smoke 2026-11-19, the 4.0 h plan): levels 1,8,24,48,80,112,256,1024x3 at N 8 per level
#       (quota max(N, 2L)); the estimator gives every tile >= 30 steps (G16: the three 1024 bursts)
#   M6  replay C: the registered stratified subsample, scaled to REPLAY_BUDGET_S: per branch a probe priced by the
#       smoke's walls x ERNIE's expert-bytes ratio, then the rest priced by the probe's own ERNIE walls, under
#       locked_r3 at the branch clock; every locked_r3 call names its run cap (--run-cap-s: replay-<b> 5000 under its
#       5400 s step cap, the probes 1700; locked_r3's 1800 s default would stop a long branch)
#   M7  Frontier, ITS PREDICTOR ON OUR PROFILES (owner decision 2026-11-19; frontier_op.py module doc; its venv,
#       primary arm only, was built before the probe in M2): its FULL token grid (387 points to 16384, owner decision
#       2026-11-19: no budgeted subset) with Frontier's default routing ("uniform", 3 seeded samples per point;
#       PROD_FRONTIER_ROUTING), timed by a lightweight timer that mirrors Frontier's profiler (one process per branch
#       pass, weights built once, eager, 2 warm-up + 20 timed calls per sample) under nsys at the
#       branch clock (decode 1710, prefill 1410: kit_lock, read back per point), the per-point kernel-sum in
#       Frontier's CSV schema, then Frontier's own trainer (its CLI) and hold-out, and the simulator-path model; a
#       second CSV and model under Frontier's NATIVE truth (with the functional API's fills, arm cli-fills); when grid
#       points are missing, a separately labelled PARTIAL arm (cli-partial; the complete-grid arm stays primary).
#       All soft.
#   M8  SECONDARY, every one SOFT: Adefault (default clocks: production delta), Aon (shared-experts stream on at
#       1710: overlap delta), Beager (eager + torch profiler, one level: independent tooling), the replay under
#       nsys per branch at its branch clock under locked_r3 (SMOKE-FIXES 7: the instrument offset and the API
#       path's fill kernels, nsys_extract replay-check), then Frontier's batch predictions of every run's calls
# TRACES (review m6): every raw *.nsys-rep and *.sqlite is written under $MOE_HOME/traces/<run>/, OUTSIDE the
# pushed results tree, so vm_results_push.sh (pinned) never xz-compresses multi-GB traces; each push first records
# their sha256 in $R/traces.sha256.json (kit_push); scripts/prod/pull_traces.sh <ip> <run> copies them to the
# laptop before the terminate.
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
FR_BALANCED=0
MODEL=ernie-4.5-21b-a3b
PROFILE=full
UTC=""
ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RES="$2"; KIT_RESULTS_GIVEN=1; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --utc) UTC="$2"; shift 2 ;;
    --graphmode-overhead) GRAPHMODE=1; shift ;;
    --frontier-balanced-arm) FR_BALANCED=1; shift ;;
    --dry-run) KIT_DRY=1; shift ;;
    *) echo "FAIL main: unknown option $1" >&2; exit 2 ;;
  esac
done
case "$PROFILE" in
  full) ;;
  blind) if [[ $GRAPHMODE -eq 1 || $FR_BALANCED -eq 1 ]]; then
           echo "FAIL main: --profile blind runs no opt-in arm (--graphmode-overhead, --frontier-balanced-arm)" >&2; exit 2
         fi ;;
  *) echo "FAIL main: --profile full|blind, not '$PROFILE'" >&2; exit 2 ;;
esac
PY="${PY_VLLM:-python3}"
# the model's names from prodlib.MODELS (the blind registry included): TOK (its own repo: the tokenizer is pinned by
# the model's revision), its serving levels and Frontier probe points when its spec names them, Frontier's CSV path
minfo() { "$PY" "$HERE/prodlib.py" "$@" 2>/dev/null || python3 "$HERE/prodlib.py" "$@"; }
INFO="$(minfo model-info --model "$MODEL")" || { echo "FAIL main: --model $MODEL: $INFO" >&2; exit 2; }
eval "$INFO"     # TOK SPEC_LEVELS SPEC_PROBE_POINTS CSVREL (shell-quoted by prodlib.shell_model_info)
UTC="${UTC:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="prod-main-$UTC"
if [[ $PROFILE == blind ]]; then   # per instance: prod-<model>-<utc>, branch run-gh200-blind-<model>-<utc>
  NAMES="$(minfo run-names --model "$MODEL" --utc "$UTC")" || { echo "FAIL main: $NAMES" >&2; exit 2; }
  eval "$NAMES"  # RESULTS_NAME RUN_BRANCH
  RUN_NAME="$RESULTS_NAME"
fi
LEVELS="${PROD_LEVELS:-${SPEC_LEVELS:-1,8,24,48,80,112,256,1024x3}}"
NREQ="${PROD_REQUESTS_PER_LEVEL:-8}"
FR_ROUTING="${PROD_FRONTIER_ROUTING:-uniform}"
PRICER="${PROD_PRICER:-r3_timing_model_v8}"
WALL="${PROD_REPLAY_WALL:-$HERE/replay_wall.json}"
R="${RES:-${RESULTS_ROOT:-$HOME/moe/results}/$RUN_NAME}"
kit_init "$R"
if [[ $KIT_DRY -eq 0 && ! -f "$R/kit-profile.json" ]]; then   # offline.sh's expected-outputs reads it (blind: no
  # secondary run, no replay); written once, so a resume keeps the run's own names
  printf '{"profile": "%s", "model": "%s", "utc": "%s", "run_branch": "%s"}\n' "$PROFILE" "$MODEL" "$UTC" \
    "${RUN_BRANCH:-}" > "$R/kit-profile.json"
fi
kit_locate_nsys
kit_env "$PY"
kit_resume_init "$FROM" ${ARGS[@]+"${ARGS[@]}"}
kit_trap_unlock --push
FR="${MOE_HOME:-$HOME/moe}/frontier"
SC="$REPO/scripts/prod"
# the raw traces, outside the pushed tree (review m6); kit_push records their sha256 in $R/traces.sha256.json
TR="${MOE_HOME:-$HOME/moe}/traces/$(basename "$R")"
export KIT_TRACES="$TR" KIT_RESULTS="$R"
if [[ $KIT_DRY -eq 0 ]]; then
  for d in gate60 A1710 A1410 Adefault Aon graphmode replay-offset-decode replay-offset-prefill frontier \
           frontier-probe; do mkdir -p "$TR/$d"; done
fi
export PYTHONPATH="$REPO:$REPO/scripts:$SC${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
SERVE=("$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency "$LEVELS"
       --requests-per-level "$NREQ" --require-pinned)
NSYS_A=("$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite true)
# Beager: the torch-profiler window is 40 steps of L 112; one level of eager serving holds it (the earlier levels were
# eager warm-up only), stopped feeding after 60 s
BEAGER=("$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency 112
        --requests-per-level "$NREQ" --require-pinned)

echo "main: model $MODEL, profile $PROFILE, levels $LEVELS, N/level $NREQ, results $R, pricer $PRICER" \
  "${RUN_BRANCH:+, run branch $RUN_BRANCH}"
if [[ $KIT_DRY -eq 1 && $PROFILE == full ]]; then   # the expected wall time of this plan (README "main.sh: every step")
  "$PY" "$SC/plan_time.py" --levels "$LEVELS" --n "$NREQ" \
    || echo "PLAN (plan_time.py could not run under $PY: needs numpy)"
fi
# Frontier's paths and step helpers (defined once, used by the full sequence and by blind.sh)
FD="$R/frontier"
CSV="$FD/profile/$CSVREL"
PROBE_POINTS="${PROD_FRONTIER_PROBE_POINTS:-$SPEC_PROBE_POINTS}"
fr_probe() {   # the probe: per branch lock, timer (under nsys), export; then fpoints, csv, train (narrow), predict
  local P="$R/frontier-probe" PT="$TR/frontier-probe" b lock
  for b in decode prefill; do
    lock=1710; [[ $b == prefill ]] && lock=1410
    mkdir -p "$P/timer-$b" "$PT/timer-$b"
    kit_lock "$lock" || return 3
    "$NSYS" profile -t cuda --force-overwrite true -o "$PT/timer-$b/timer" "$PY" "$SC/frontier_op.py" timer \
      --model "$MODEL" --branch "$b" --lock "$lock" --routing "$FR_ROUTING" --max-tokens 16384 \
      --grid-points "$PROBE_POINTS" --out "$P/timer-$b" || { kit_unlock; return 3; }
    kit_unlock
    "$NSYS" export --type sqlite --force-overwrite true -o "$PT/timer-$b/timer.sqlite" "$PT/timer-$b/timer.nsys-rep" \
      || return 3
  done
  "$PY" "$SC/frontier_op.py" fpoints --model "$MODEL" --timer "$P/timer-decode" --timer-sqlite \
    "$PT/timer-decode/timer.sqlite" --timer "$P/timer-prefill" --timer-sqlite "$PT/timer-prefill/timer.sqlite" \
    --grid-points "$PROBE_POINTS" --max-tokens 16384 --out "$P/points.json" || return 3
  "$PY" "$SC/frontier_op.py" csv --model "$MODEL" --points "$P/points.json" --src "$FSRC" --out "$P/profile" || return 3
  "$PY" "$SC/frontier_op.py" train --narrow --model "$MODEL" --csv "$P/profile/$CSVREL" --src "$FSRC" \
    --out "$P/models" || return 3
  "$PY" "$SC/frontier_op.py" predict --model "$MODEL" --models "cli=$P/models" --csv "$P/profile/$CSVREL" \
    --points "$P/points.json" --src "$FSRC" --out "$P/predictions" || return 3
  echo "PASS frontier-timer-probe: points $PROBE_POINTS ($FR_ROUTING), timer + Frontier's csv/train/predict"
}
# M7's helpers: the timer (every grid point and sample, one process per branch pass at the branch clock, a marker
# kernel per point), the two branch passes of one routing, then the per-point kernel-sum (fpoints); the traces under
# $TR (outside the pushed tree)
fr_timer() {   # fr_timer BRANCH LOCK ROUTING DIR: every grid point of the branch, one process, weights once, eager, nsys
  local b="$1" lock="$2" routing="$3" d="$R/frontier/$4" t="$TR/frontier/$4"
  mkdir -p "$d" "$t"
  "$NSYS" profile -t cuda --force-overwrite true -o "$t/timer" "$PY" "$SC/frontier_op.py" timer --model "$MODEL" \
    --branch "$b" --lock "$lock" --routing "$routing" --max-tokens 16384 --out "$d"
}
fr_passes() {   # fr_passes PREFIX ROUTING [--soft]: the two branch passes (lock, timer, export) of one routing
  local pre="$1" routing="$2" b lock
  for b in decode prefill; do
    lock=1710; [[ $b == prefill ]] && lock=1410
    step "$pre-lock-$b" --soft --always --cap 120 -- kit_lock "$lock"
    step "$pre-$b" --soft --cap 3600 --needs "$pre-lock-$b" --out "$FD/${pre#frontier-}-$b/timer.json,$TR/frontier/${pre#frontier-}-$b/timer.nsys-rep" \
      -- fr_timer "$b" "$lock" "$routing" "${pre#frontier-}-$b"
    step "$pre-export-$b" --soft --cap 1800 --needs "$pre-$b" --out "$TR/frontier/${pre#frontier-}-$b/timer.sqlite" -- "$NSYS" export --type sqlite \
      --force-overwrite true -o "$TR/frontier/${pre#frontier-}-$b/timer.sqlite" "$TR/frontier/${pre#frontier-}-$b/timer.nsys-rep"
  done
}
fr_fpoints() {   # fr_fpoints STEP PREFIX OUT: fpoints over the two passes, their traces named
  local x="${2#frontier-}"
  step "$1" --soft --cap 1800 --out "$3" -- "$PY" "$SC/frontier_op.py" fpoints --model $MODEL --timer "$FD/$x-decode" \
    --timer-sqlite "$TR/frontier/$x-decode/timer.sqlite" --timer "$FD/$x-prefill" --timer-sqlite "$TR/frontier/$x-prefill/timer.sqlite" \
    --max-tokens 16384 --out "$3"
}
# M0: the whole environment first (smoke rental 2026-11-19: SMOKE-FIXES.md 1-5), fail-fast with its reasons
step env-preflight --always --cap 600 -- "$PY" "$SC/env_preflight.py" --repo "$REPO" --model $MODEL --nsys "$NSYS" \
  --power-w "${PROD_POWER_W:-700}" --locks 1710,1410 --tmpdir "$KIT_TMP" --frontier-src "$FR/src" \
  --disk "${MOE_HOME:-$HOME/moe}" --min-free-gb "${PROD_MIN_FREE_GB:-200}" --out "$R/env-preflight.json"
step env --always --cap 300 -- kit_env_check
step power-readback --always --cap 60 -- kit_power_check "${PROD_POWER_W:-700}"
step unlock-start --always --cap 60 -- kit_unlock
step revision-pinned --always --cap 120 -- "$PY" -c "import sys; sys.path[:0]=['$SC']; import prodlib as P; r=P.hf_revision('$MODEL'); print(('PASS revision ' + r) if r else 'FAIL revision: $MODEL is not pinned in prodlib.HF_REVISIONS'); sys.exit(0 if r else 3)"
if [[ $PROFILE == blind ]]; then   # ONE blind model per GH200, lean (scripts/prod/blind.sh), then the summary
  . "$HERE/blind.sh"
  kit_summary
  exit 0
fi
step replay-wall-present --always --cap 30 -- test -s "$WALL"
step price-check --cap 900 -- "$PY" "$SC/prodlib.py" price-check --model $MODEL --pricer "$PRICER"
# M1
step prompts-fetch --cap 1800 --out "$R/data/ShareGPT_V3_unfiltered_cleaned_split.json" -- "$PY" "$SC/prompts.py" fetch --source sharegpt --dest "$R/data"
step prompts-build --cap 1800 --out "$R/prompts.jsonl" -- "$PY" "$SC/prompts.py" build --source sharegpt --data "$R/data/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --tokenizer "$TOK" --n 4096 --seed 0 --max-input 2048 --out-dist sharegpt --out "$R/prompts.jsonl"
step serve-plan --cap 300 -- "${SERVE[@]}" --mode graphs --out "$R/A1710" --dry-run

# M2: the 60 s gate (the model's first load: the weights download here)
step gate60-capture --cap 3600 --out "$TR/gate60/serve.nsys-rep,$R/gate60/manifest.json" -- "${NSYS_A[@]}" -o "$TR/gate60/serve" "${SERVE[@]}" --mode graphs --shared-stream off \
  --min-steps-per-tile 0 --max-seconds 60 --out "$R/gate60"
step gate60-export --cap 1800 --out "$TR/gate60/serve.sqlite" -- "$NSYS" export --type sqlite --force-overwrite true -o "$TR/gate60/serve.sqlite" "$TR/gate60/serve.nsys-rep"
step gate60-extract --cap 1800 --out "$R/gate60/calls.jsonl|$R/gate60/calls.shards.json" -- "$PY" "$SC/nsys_extract.py" --run "$R/gate60" --sqlite "$TR/gate60/serve.sqlite" --model $MODEL
step gate60-sequences --cap 600 -- "$PY" "$SC/nsys_extract.py" check-sequences --run "$R/gate60" --model $MODEL

# THE EARLY FRONTIER PROBE (hard; review 2026-11-19): Frontier's venv, its grid-check, then the new timer on 3 ERNIE
# grid points at their branch clocks and Frontier's own csv / trainer (narrow) / predict on them, ~5 min. A failure
# here stops the run before the expensive blocks (the Frontier block M7 would otherwise fail hours later)
step frontier-setup --cap 3600 --out "$FR/frontier.env" -- bash "$SC/frontier_setup.sh" --dest "$FR" --arms primary
if [[ $KIT_DRY -eq 0 && -f "$FR/frontier.env" ]]; then . "$FR/frontier.env"; fi
FSRC="${FRONTIER_SRC:-$FR/src}"
export R TR MODEL SC PY NSYS FR_ROUTING FSRC CSVREL PROBE_POINTS FR_PY_PRIMARY 2>/dev/null || true
step frontier-grid-check --cap 300 --needs frontier-setup -- "$PY" "$SC/frontier_op.py" grid-check --model $MODEL --src "$FSRC" \
  --max-tokens 16384
step frontier-timer-probe --cap 900 --needs frontier-grid-check --out "$R/frontier-probe/predictions/index.json" -- fr_probe
step push-gate60 --soft --cap 3600 -- kit_push "prod main: the 60 s $MODEL gate and the Frontier probe"

# M3: the board's sag par at 1410 (the v7 prefill blind's units and finaliser; the driver's own deadline is 75 min)
step sag1410 --cap 5400 --out "$R/sag/par.json" -- bash "$SC/sag1410.sh" --out "$R/sag"
step unlock-after-sag --always --cap 60 -- kit_unlock

serve_A() {   # serve_A LABEL LOCK(0=default) STREAM(on|off) [--soft]; G-COV gates the branch of the lock only
  local lab="$1" lock="$2" stream="$3" soft=("${@:4}") pre cov=all
  if [[ "$lock" != 0 ]]; then
    cov=decode; [[ "$lock" == 1410 ]] && cov=prefill
    pre="$lab-lock"; step "$pre" ${soft[@]+"${soft[@]}"} --always --cap 120 -- kit_lock "$lock"
  else
    pre="$lab-unlock"; step "$pre" ${soft[@]+"${soft[@]}"} --always --cap 60 -- kit_unlock
  fi
  step "$lab-capture" ${soft[@]+"${soft[@]}"} --cap 10800 --needs "$pre" --out "$TR/$lab/serve.nsys-rep,$R/$lab/manifest.json" -- "${NSYS_A[@]}" -o "$TR/$lab/serve" "${SERVE[@]}" \
    --mode graphs --shared-stream "$stream" --cov-branch "$cov" --out "$R/$lab"
  step "$lab-unlock-after" --always --cap 60 -- kit_unlock
  step "$lab-export" ${soft[@]+"${soft[@]}"} --cap 5400 --needs "$lab-capture" --out "$TR/$lab/serve.sqlite" -- "$NSYS" export --type sqlite \
    --force-overwrite true -o "$TR/$lab/serve.sqlite" "$TR/$lab/serve.nsys-rep"
  step "$lab-extract" ${soft[@]+"${soft[@]}"} --cap 3600 --needs "$lab-export" \
    --out "$R/$lab/calls.jsonl|$R/$lab/calls.shards.json" -- "$PY" "$SC/nsys_extract.py" --run "$R/$lab" \
    --sqlite "$TR/$lab/serve.sqlite" --model $MODEL
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
  replay_cell() {   # replay_cell STEP OUTDIR SAMPLE TAG CAP RUNCAP (locked_r3's own run cap, below the step's)
    step "$1" --cap "$5" --out "$2" -- env MOE_RESULTS_DIR="$2" python3 scripts/locked_r3.py --session-tag "$4" \
      --locks $lock --groups 1 --slip-policy cell --min-clean-repeats 0 --run-cap-s "$6" --out-dir "$R/locked-$(basename "$2")" \
      --r3-script "$SC/replay.py" -- --model $MODEL --histogram "$3" --branch $b --inst GRR \
      --repeats 9 --duty 0.25 --seed 0
  }
  step "replay-probe-sample-$b" --cap 600 --out "$R/sample-probe-$b/sample.json" -- "$PY" "$SC/replay.py" sample --run "$R/$src" --model $MODEL --branch $b \
    --n $n --per-tile "${PROD_PROBE_PER_TILE:-2}" --wall "$WALL" --repeats 9 --out "$R/sample-probe-$b"
  replay_cell "replay-probe-$b" "$R/replay-probe-$b" "$R/sample-probe-$b/sample.json" "prod-replay-probe-$b" 1800 1700
  step "replay-probe-wall-$b" --cap 120 --out "$R/replay-probe-$b/replay_wall.json" -- "$PY" "$SC/replay.py" wall-merge --from "$R/replay-probe-$b" --model $MODEL \
    --out "$R/replay-probe-$b/replay_wall.json"
  step "replay-sample-$b" --cap 600 --out "$R/sample-$b/sample.json" -- "$PY" "$SC/replay.py" sample --run "$R/$src" --model $MODEL --branch $b --n $n \
    --probe-wall "$R/replay-probe-$b/replay_wall.json" --repeats 9 --out "$R/sample-$b"
  replay_cell "replay-$b" "$R/replay-$b" "$R/sample-$b/sample.json" "prod-replay-$b" 5400 5000
done
step unlock-after-replay --always --cap 60 -- kit_unlock
step push-replay --soft --cap 3600 -- kit_push "prod main: replay C"

# M7: Frontier, its predictor on our profiles (frontier_op.py module doc "PROFILER"; its venv and grid-check ran
# before the probe). The timer: every grid point and sample, one process per branch pass at the branch clock, a marker
# kernel per point, then the per-point kernel-sum (fpoints); the traces under $TR (outside the pushed tree)
fr_passes frontier-timer "$FR_ROUTING"
step frontier-timer-unlock --always --cap 60 -- kit_unlock
fr_fpoints frontier-points frontier-timer "$FD/points.json"
# the primary arm: the complete grid, kernel-sum without the API fills (the scorer's serving truth)
step frontier-csv --soft --cap 600 --needs frontier-points --out "$CSV" -- "$PY" "$SC/frontier_op.py" csv --model $MODEL --points "$FD/points.json" \
  --src "$FSRC" --truth kernel_sum --out "$FD/profile"
step frontier-train --soft --cap 1800 --needs frontier-csv --out "$FD/models" -- "$PY" "$SC/frontier_op.py" train --model $MODEL --csv "$CSV" \
  --src "$FSRC" --out "$FD/models"
step frontier-holdout --soft --cap 1800 --needs frontier-csv --out "$FD/holdout" -- "$PY" "$SC/frontier_op.py" holdout --model $MODEL --csv "$CSV" \
  --src "$FSRC" --out "$FD/holdout"
step frontier-simpath --soft --cap 1800 --needs frontier-csv --out "$FD/models-simpath" -- "$PY" "$SC/frontier_op.py" simpath --model $MODEL \
  --arm-label simpath --csv "$CSV" --src "$FSRC" --out "$FD/models-simpath"
# Frontier's NATIVE truth (record_function: every kernel of the functional call, the fills included): arm cli-fills
step frontier-csv-fills --soft --cap 600 --needs frontier-points --out "$FD/profile-fills/$CSVREL" -- "$PY" "$SC/frontier_op.py" csv --model $MODEL \
  --points "$FD/points.json" --src "$FSRC" --truth with_fills --out "$FD/profile-fills"
step frontier-train-fills --soft --cap 1800 --needs frontier-csv-fills --out "$FD/models-fills" -- "$PY" "$SC/frontier_op.py" train --model $MODEL \
  --arm-label cli-fills --csv "$FD/profile-fills/$CSVREL" --src "$FSRC" --out "$FD/models-fills"
# grid points missing (frontier-points FAILED): the separately labelled PARTIAL arm, its missing points in
# profile-partial/csv_meta.json; on a complete grid both steps PASS and write no model
step frontier-csv-partial --soft --cap 600 --out "$FD/profile-partial/csv_meta.json" -- "$PY" "$SC/frontier_op.py" csv --model $MODEL \
  --points "$FD/points.json" --src "$FSRC" --truth kernel_sum --partial --out "$FD/profile-partial"
step frontier-train-partial --soft --cap 1800 --needs frontier-csv-partial -- "$PY" "$SC/frontier_op.py" train --model $MODEL \
  --arm-label cli-partial --csv "$FD/profile-partial/$CSVREL" --src "$FSRC" --only-if-partial "$FD/profile-partial/csv_meta.json" \
  --out "$FD/models-partial"
if [[ $FR_BALANCED -eq 1 ]]; then   # opt-in: the round robin ("balanced") as a labelled second arm, cli-balanced
  fr_passes frontier-balanced balanced
  step frontier-balanced-unlock --always --cap 60 -- kit_unlock
  fr_fpoints frontier-balanced-points frontier-balanced "$FD/points-balanced.json"
  step frontier-balanced-csv --soft --cap 600 --needs frontier-balanced-points --out "$FD/profile-balanced/$CSVREL" -- "$PY" "$SC/frontier_op.py" csv \
    --model $MODEL --points "$FD/points-balanced.json" --src "$FSRC" --truth kernel_sum --out "$FD/profile-balanced"
  step frontier-balanced-train --soft --cap 1800 --needs frontier-balanced-csv --out "$FD/models-balanced" -- "$PY" "$SC/frontier_op.py" train \
    --model $MODEL --arm-label cli-balanced --csv "$FD/profile-balanced/$CSVREL" --src "$FSRC" --out "$FD/models-balanced"
fi
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
  step "replay-offset-nsys-$b" --soft --cap 2400 --out "$TR/replay-offset-$b/replay.nsys-rep" -- "$NSYS" profile -t cuda --cuda-graph-trace=node \
    --force-overwrite true -o "$TR/replay-offset-$b/replay" env MOE_RESULTS_DIR="$R/replay-offset-$b" python3 scripts/locked_r3.py \
    --session-tag "prod-replay-offset-$b" --locks $lock --groups 1 --slip-policy cell --min-clean-repeats 0 --run-cap-s 1700 \
    --out-dir "$R/locked-replay-offset-$b" --r3-script "$SC/replay.py" -- --model $MODEL --histogram "$R/sample-$b/sample.json" \
    --branch $b --inst GRR --repeats 1 --duty 0.25 --seed 0 --cell-marks
  step "replay-offset-export-$b" --soft --cap 1800 --needs "replay-offset-nsys-$b" --out "$TR/replay-offset-$b/replay.sqlite" -- "$NSYS" export --type sqlite \
    --force-overwrite true -o "$TR/replay-offset-$b/replay.sqlite" "$TR/replay-offset-$b/replay.nsys-rep"
  step "replay-offset-check-$b" --soft --cap 1800 --needs "replay-offset-export-$b" --out "$R/replay-offset-$b/replay_check.json" -- "$PY" "$SC/nsys_extract.py" \
    replay-check --serving "$R/$src" --sample "$R/sample-$b/sample.json" --replay-sqlite "$TR/replay-offset-$b/replay.sqlite" \
    --model $MODEL --branch $b --out "$R/replay-offset-$b/replay_check.json"
done
if [[ $GRAPHMODE -eq 1 ]]; then   # opt-in: the node-tracing overhead (the smoke measured it), default clocks
  step graphmode-unlock --soft --always --cap 60 -- kit_unlock
  step graphmode-capture --soft --cap 3600 --needs graphmode-unlock --out "$TR/graphmode/serve.nsys-rep" -- "$NSYS" profile \
    -t cuda,nvtx --cuda-graph-trace=graph --force-overwrite true -o "$TR/graphmode/serve" "${SERVE[@]}" --mode graphs \
    --shared-stream off --min-steps-per-tile 0 --max-seconds 60 --out "$R/graphmode"
fi
step replay-offset-unlock --always --cap 60 -- kit_unlock
# Frontier's batch predictions on the VM (owner decision 2026-11-19): every run whose calls were extracted, ONE
# DataFrame per model (cli, simpath, cli-fills, and cli-partial / cli-balanced when trained), written compact to
# $R/frontier/predictions/ (index.json + npy). No --needs: a model set counts once ITS trainer passed (versions.json)
FCALLS=()
for lab in A1710 A1410 Adefault Aon Beager; do
  if [[ $KIT_DRY -eq 1 || -e "$R/$lab/calls.jsonl" || -e "$R/$lab/calls.shards.json" ]]; then FCALLS+=(--calls "$lab=$R/$lab"); fi
done
FMODELS=()
FARMS=(cli=models simpath=models-simpath cli-fills=models-fills cli-partial=models-partial)
[[ $FR_BALANCED -eq 1 ]] && FARMS+=(cli-balanced=models-balanced)
for m in "${FARMS[@]}"; do
  # a model set counts once its trainer PASSED (frontier_op.py writes versions.json only then)
  if [[ $KIT_DRY -eq 1 || -f "$FD/${m#*=}/versions.json" ]]; then FMODELS+=(--models "${m%%=*}=$FD/${m#*=}"); fi
done
step frontier-predict --soft --cap 3600 --out "$FD/predictions/index.json" -- "$PY" "$SC/frontier_op.py" predict --model $MODEL \
  ${FMODELS[@]+"${FMODELS[@]}"} --csv "$CSV" ${FCALLS[@]+"${FCALLS[@]}"} --src "$FSRC" --out "$FD/predictions"
step push-final --soft --cap 3600 -- kit_push "prod main: measurements and Frontier predictions done (score on the laptop: scripts/prod/offline.sh)"
# the push left every needed file of this run on the branch (else: copy the listed files off the VM before the
# terminate); a FAIL here is the rental's last word, and the EXIT trap still pushes with the exit code
step held-back-check --cap 120 -- "$PY" "$SC/prodlib.py" held-back-check \
  --held "${MOE_PUSH_HOME:-$HOME/moe}/push/HELD-BACK.txt" --results "$R"
kit_summary
