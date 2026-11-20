#!/usr/bin/env bash
# THE EXTRAS (owner 2026-11-19: "this is the final run; get as much data and information as possible"): the GPU work
# right after scripts/prod/main.sh, in the SAME rental, on the SAME results root. main.sh's outputs are read, never
# written: everything goes under <results>/extras/ (its own SUMMARY.txt and logs/), the raw traces under
# $MOE_HOME/traces/<run>/extras/ (outside the pushed tree; their sha256 in <results>/extras/traces.sha256.json; pull
# them with `bash scripts/prod/pull_traces.sh <ip> <run>/extras`).
#
# THE COMMIT SWITCH (a documented operator step, on the VM, after main.sh's SUMMARY.txt is complete): extras.sh is at a
# commit NEWER than main's, so (the checkout cloned with --repo has an origin; one cloned from a bundle has none: fetch
# the new bundle into origin's refs)
#
#   cd "$REPO" && git fetch origin && git checkout --detach <extras sha>                               # --repo clone
#   cd "$REPO" && git fetch <bundle> '+refs/heads/*:refs/remotes/origin/*' && git checkout --detach <extras sha>
#   . ~/moe/env.sh && PROD_EXTRAS_COMMIT=<extras sha> bash scripts/prod/extras.sh --results <main's results dir> \
#       [--budget-s 9000] [--no-long-context] [--after-main-fail] [--from STEP] [--dry-run]
#
# and the first steps hold it (hard): extras-main-finished (main's SUMMARY.txt records PASS held-back-check, unless
# the operator passes --after-main-fail, which is logged in the extras' SUMMARY.txt; no main.sh, serve_capture.py,
# locked_r3.py or nsys running), extras-commit-switch (a clean tree, env-preflight P1's rule, at the expected extras
# commit, main's own commit, from main's env-preflight.json, an ancestor of it), then env-preflight with
# --expect-commit <extras sha> (P1) on ERNIE (cached by main), P9 (Frontier's remote), P11 (setup_vm's verdict) and P12
# (a timed download) skipped: none is needed and a network blip must not stop the extras; P10's disk need is
# main's ACTUAL trace sizes (extras_lib.py disk-need). Qwen3's pin, download rate and disk (P8, P12, P10) are X3's own
# first check (X3-hf-preflight, soft): a Hub or disk fault drops X3 only. price-check-qwen3 is soft, and X3's budget
# step needs it: when v8 cannot price Qwen3, X3 is skipped whole.
#
# DONE (review of the extras M2): main.sh's last push wrote DRIVER-DONE on the run's branch; the extras never touch it.
# Their EXIT trap writes <results>/extras/EXTRAS-DONE ("exit RC at UTC") and pushes WITHOUT --final. TERMINATE ONLY
# AFTER THE EXTRAS: the operator watches the branch for vm/results/<run>/extras/EXTRAS-DONE and 'PASS
# extras-held-back-check' in vm/results/<run>/extras/SUMMARY.txt (DRIVER-DONE alone means main ended, not the rental).
#
# BLOCKS, in order, each SOFT (a failure drops that block's later steps, never another block's), each step with its
# registered cap (>= 2x its expected time, plan_time.extras_plan; a test holds it), a push after each block:
#   X1 NOISE FLOOR   main's A1710 again, identical (same prompts, levels, N, seed, lock, mode, stream off) -> A1710r;
#                    X1-identical holds its manifest to A1710's (levels, N, seed, mode, prompt body sha256); the scorer
#                    pairs call k with call k when the (step, layer, T_pad) sequences are identical (run-to-run
#                    kernel-sum variation per tile), else compares per-tile medians of measured / v8 and / roofline
#   X2 NCU BYTES     a stratified subsample (30 + 30 cells) of main's replay C sample, both branches, through the
#                    study's counter route (scripts/dram_counter_route.py, imported; ncu_bytes.py): counted w1 / w2
#                    DRAM read bytes per call at the branch clock, real ids; the scorer joins v8's predicted reads
#   X3 QWEN3         Qwen/Qwen3-30B-A3B (E 128, k 8, H 2048, I 768, 48 MoE layers; prodlib.QWEN3, pinned): its prompt
#                    set, the 60 s gate (G-CFG, G-BACKEND, G-REV, every extraction gate and the sequences), A1710 and
#                    A1410 at levels 1..256 ($PROD_QWEN3_LEVELS: no 1024x3, its KV cache; the estimator still gives
#                    every tile >= 30 steps), each capture pushed on its own after its push-size check (GitHub's
#                    limits), replay C on a smaller budget (1200 s per branch): a 2-per-tile probe on Qwen3 priced by
#                    main's ERNIE probe walls x the ACTIVE-bytes ratio, then the rest by the probe's own Qwen3 walls;
#                    Frontier's grid-check and timer on Qwen3's shapes (the full grid, uniform x 3 seeds, branch
#                    clocks), its trainer under both truths and its predictions. The board's sag par is main's
#   X4 LONG CONTEXT  (unless --no-long-context) ERNIE served with long prompts (unrelated ShareGPT turns concatenated to
#                    4096..16000 tokens, 16 output tokens; levels 1,2,4,8, N 24) at the 1410 lock: many calls on
#                    128x128 and G16 (T >= 8256), scored as their own run
# THE BUDGET: one total for the extras (default 9000 s, --budget-s / $PROD_EXTRAS_BUDGET_S), counted from this script's
# start. Before each block its budget step passes only if the time spent plus the block's expected seconds (at this
# run's own $PROD_LEVELS, $PROD_REQUESTS_PER_LEVEL, $PROD_QWEN3_LEVELS) fits (extras_lib.py budget); else the block is
# SKIPPED, step by step. THE DEADLINE (M4): EXTRAS_DEADLINE = start + budget + grace ($PROD_EXTRAS_GRACE_S, 900 s),
# exported; kitlib's step skips every step that would start after it and caps each step's timeout at the time left;
# the final push and the held-back check (--no-deadline) and the EXIT trap (unlock, push) always run. --dry-run prints
# every step and the expected seconds per block and in total.
# RESUME: --from STEP as main.sh (kitlib.sh); a resume counts the budget from its own start.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
. "$HERE/kitlib.sh"

RES=""; FROM=""; EXPECT="${PROD_EXTRAS_COMMIT:-}"; BUDGET="${PROD_EXTRAS_BUDGET_S:-9000}"; LONGCTX=1; AFTER_FAIL=0
GRACE="${PROD_EXTRAS_GRACE_S:-900}"
ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RES="$2"; KIT_RESULTS_GIVEN=1; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --expect-commit) EXPECT="$2"; shift 2 ;;
    --budget-s) BUDGET="$2"; shift 2 ;;
    --no-long-context) LONGCTX=0; shift ;;
    --after-main-fail) AFTER_FAIL=1; shift ;;
    --dry-run) KIT_DRY=1; shift ;;
    *) echo "FAIL extras: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RES" ]] || { echo "FAIL extras: --results DIR (main.sh's results directory) is required" >&2; exit 2; }
[[ "$BUDGET" =~ ^[0-9]+$ ]] || { echo "FAIL extras: --budget-s $BUDGET is not a whole number of seconds" >&2; exit 2; }
[[ "$GRACE" =~ ^[0-9]+$ ]] || { echo "FAIL extras: \$PROD_EXTRAS_GRACE_S $GRACE is not a whole number of seconds" >&2; exit 2; }
MODEL=ernie-4.5-21b-a3b
TOK=baidu/ERNIE-4.5-21B-A3B-PT
QM=qwen3-30b-a3b
QTOK=Qwen/Qwen3-30B-A3B
LEVELS="${PROD_LEVELS:-1,8,24,48,80,112,256,1024x3}"
# Qwen3's KV cache (M7): no 1024x3 bursts (every manifest records num_gpu_blocks, preemptions, realized concurrency)
QLEVELS="${PROD_QWEN3_LEVELS:-1,8,24,48,80,112,256}"
NREQ="${PROD_REQUESTS_PER_LEVEL:-8}"
FR_ROUTING="${PROD_FRONTIER_ROUTING:-uniform}"
PY="${PY_VLLM:-python3}"
PRICER="${PROD_PRICER:-r3_timing_model_v8}"
R="$RES"
EX="$R/extras"
Q="$EX/qwen3"
kit_init "$EX"
kit_locate_nsys
kit_env "$PY"
kit_resume_init "$FROM" ${ARGS[@]+"${ARGS[@]}"}
# M2: the EXIT trap writes EXTRAS-DONE and pushes without --final (main's DRIVER-DONE keeps its meaning)
if [[ $KIT_DRY -eq 0 ]]; then KIT_DONE_FILE="$EX/EXTRAS-DONE"; fi
kit_trap_unlock --push
FR="${MOE_HOME:-$HOME/moe}/frontier"
SC="$REPO/scripts/prod"
TR="${MOE_HOME:-$HOME/moe}/traces/$(basename "$R")/extras"
export KIT_TRACES="$TR" KIT_RESULTS="$EX"
EXTRAS_T0="$(date +%s)"
EXTRAS_DEADLINE=$(( EXTRAS_T0 + BUDGET + GRACE ))
export EXTRAS_T0 EXTRAS_BUDGET_S="$BUDGET" EXTRAS_DEADLINE
# M4: kitlib skips a step that would start past the deadline and caps each step's timeout at the time left
if [[ $KIT_DRY -eq 0 ]]; then KIT_DEADLINE="$EXTRAS_DEADLINE"; fi
if [[ $KIT_DRY -eq 0 ]]; then
  for d in A1710r ncu-decode ncu-prefill qwen3/gate60 qwen3/A1710 qwen3/A1410 qwen3/frontier LC1410; do
    mkdir -p "$TR/$d"
  done
fi
export PYTHONPATH="$REPO:$REPO/scripts:$SC${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
DATA="$R/data"
SHAREGPT="$DATA/ShareGPT_V3_unfiltered_cleaned_split.json"
NSYS_A=("$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite true)
# X1: main.sh's SERVE, verbatim (the same prompt file, levels, N, seed 0 by default, pin)
SERVE=("$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$R/prompts.jsonl" --concurrency "$LEVELS"
       --requests-per-level "$NREQ" --require-pinned)
QSERVE=("$PY" "$SC/serve_capture.py" --model $QM --prompts "$Q/prompts.jsonl" --concurrency "$QLEVELS"
        --requests-per-level "$NREQ" --require-pinned)
LSERVE=("$PY" "$SC/serve_capture.py" --model $MODEL --prompts "$EX/long/prompts.jsonl" --concurrency 1,2,4,8
        --requests-per-level 24 --max-model-len 17408 --min-steps-per-tile 0 --require-pinned)

echo "extras: results $R -> $EX, budget ${BUDGET}s (deadline +${GRACE}s grace), long context" \
     "$([[ $LONGCTX -eq 1 ]] && echo on || echo off), expected commit ${EXPECT:-(none: extras-commit-switch fails)}"
NOLC=(); [[ $LONGCTX -eq 0 ]] && NOLC=(--no-long-context)
# P10 (MINOR): the disk the extras need, from main's actual trace sizes (+ Qwen3's weights), else 200 GB
XNEED="$("$PY" "$SC/extras_lib.py" disk-need --results "$R" ${NOLC[@]+"${NOLC[@]}"} 2>/dev/null || echo 200)"
MINFREE="${PROD_EXTRAS_MIN_FREE_GB:-$XNEED}"
if [[ $KIT_DRY -eq 1 ]]; then   # the expected seconds per step, per block and in total (plan_time.extras_plan)
  "$PY" "$SC/plan_time.py" --extras --levels "$LEVELS" --n "$NREQ" --qwen3-levels "$QLEVELS" --budget-s "$BUDGET" \
    ${NOLC[@]+"${NOLC[@]}"} \
    || echo "PLAN (plan_time.py could not run under $PY: needs numpy)"
fi

# X0: main is done, the tree is the extras commit, the environment (hard)
AFTER=(); [[ $AFTER_FAIL -eq 1 ]] && AFTER=(--after-main-fail)
step extras-main-finished --always --cap 60 -- "$PY" "$SC/extras_lib.py" main-finished --results "$R" ${AFTER[@]+"${AFTER[@]}"}
if [[ $AFTER_FAIL -eq 1 && $KIT_DRY -eq 0 ]]; then
  kit_record "OVERRIDE extras-main-finished: --after-main-fail (the operator runs the extras after a main.sh that did not pass held-back-check)"
fi
step extras-commit-switch --always --cap 120 -- "$PY" "$SC/extras_lib.py" commit-switch --repo "$REPO" --results "$R" \
  --expect "$EXPECT"
step env-preflight --always --cap 600 -- "$PY" "$SC/env_preflight.py" --repo "$REPO" --model $MODEL --nsys "$NSYS" \
  --expect-commit "${EXPECT:-unset}" --power-w "${PROD_POWER_W:-700}" --locks 1710,1410 --tmpdir "$KIT_TMP" \
  --frontier-src "$FR/src" --disk "${MOE_HOME:-$HOME/moe}" --min-free-gb "$MINFREE" --skip P9,P11,P12 \
  --out "$EX/env-preflight.json"
step env --always --cap 300 -- kit_env_check
step power-readback --always --cap 60 -- kit_power_check "${PROD_POWER_W:-700}"
step unlock-start --always --cap 60 -- kit_unlock
step revision-pinned-qwen3 --always --cap 120 -- "$PY" -c "import sys; sys.path[:0]=['$SC']; import prodlib as P; r=P.hf_revision('$QM'); print(('PASS revision ' + r) if r else 'FAIL revision: $QM is not pinned'); sys.exit(0 if r else 3)"
step price-check-qwen3 --soft --cap 900 -- "$PY" "$SC/prodlib.py" price-check --model $QM --pricer "$PRICER"

budget() {   # budget BLOCK [NEEDS]: the block's budget and input checks (soft; every later step of the block --needs them)
  step "$1-budget" --soft --cap 60 ${2:+--needs "$2"} -- "$PY" "$SC/extras_lib.py" budget --block "$1" --levels "$LEVELS" \
    --n "$NREQ" --qwen3-levels "$QLEVELS"
  step "$1-inputs" --soft --cap 60 --needs "$1-budget" -- "$PY" "$SC/extras_lib.py" inputs --block "$1" --results "$R"
}
capture() {   # capture PREFIX LOCK OUTDIR TRACEDIR NEEDS SERVE-ARGS...: lock, nsys capture, unlock, export, extract (soft)
  local pre="$1" lock="$2" out="$3" t="$4" needs="$5"; shift 5
  step "$pre-lock" --soft --always --cap 120 --needs "$needs" -- kit_lock "$lock"
  step "$pre-capture" --soft --cap 10800 --needs "$pre-lock" --out "$t/serve.nsys-rep,$out/manifest.json" -- \
    "${NSYS_A[@]}" -o "$t/serve" "$@" --mode graphs --shared-stream off --out "$out"
  step "$pre-unlock-after" --always --cap 60 -- kit_unlock
  step "$pre-export" --soft --cap 5400 --needs "$pre-capture" --out "$t/serve.sqlite" -- "$NSYS" export --type sqlite \
    --force-overwrite true -o "$t/serve.sqlite" "$t/serve.nsys-rep"
}
extract() {   # extract STEP RUN TRACEDIR MODEL NEEDS
  step "$1" --soft --cap 3600 --needs "$5" --out "$2/calls.jsonl|$2/calls.shards.json" -- "$PY" "$SC/nsys_extract.py" \
    --run "$2" --sqlite "$3/serve.sqlite" --model "$4"
}

# X1 NOISE FLOOR: main's A1710, identical
budget X1
capture X1 1710 "$EX/A1710r" "$TR/A1710r" X1-inputs "${SERVE[@]}"
step X1-identical --soft --cap 60 --needs X1-capture -- "$PY" "$SC/extras_lib.py" x1-identical --base "$R/A1710" \
  --repeat "$EX/A1710r"
extract X1-extract "$EX/A1710r" "$TR/A1710r" $MODEL X1-export
step X1-push --soft --cap 3600 -- kit_push "prod extras: X1 noise floor (A1710r)"

# X2 NCU BYTES: the counter route on a stratified subsample of main's replay C sample, each branch at its clock
budget X2
step X2-subsample --soft --cap 600 --needs X2-inputs --out "$EX/ncu-sample/sample.json" -- "$PY" "$SC/ncu_bytes.py" \
  subsample --model $MODEL --sample decode="$R/sample-decode/sample.json" --sample prefill="$R/sample-prefill/sample.json" \
  --n-per-branch "${PROD_NCU_PER_BRANCH:-30}" --out "$EX/ncu-sample"
for b in decode prefill; do
  lock=1710; [[ $b == prefill ]] && lock=1410
  step "X2-lock-$b" --soft --always --cap 120 --needs X2-subsample -- kit_lock $lock
  step "X2-ncu-$b" --soft --cap 2400 --needs "X2-lock-$b" --out "$EX/ncu-$b/ncu_bytes.json" -- "$PY" "$SC/ncu_bytes.py" run \
    --model $MODEL --sample "$EX/ncu-sample/sample.json" --branch $b --lock-mhz $lock --profile-dir "$TR/ncu-$b" \
    --timeout-s 2200 --out "$EX/ncu-$b/ncu_bytes.json"
  step "X2-unlock-$b" --always --cap 60 -- kit_unlock
done
step X2-push --soft --cap 3600 -- kit_push "prod extras: X2 ncu bytes"

# X3 QWEN3-30B-A3B: the second blind model
budget X3 price-check-qwen3
# Qwen3's pin (P8: config.json and the weight index at the pin, the fixed resolver), its download rate (P12) and the
# disk for its ~61 GB (P10): env_preflight.py's own checks, the rest skipped
step X3-hf-preflight --soft --cap 300 --needs X3-inputs -- "$PY" "$SC/env_preflight.py" --repo "$REPO" --model $QM \
  --skip P1,P2,P3,P4,P5,P6,P7,P9,P11 --disk "${MOE_HOME:-$HOME/moe}" --min-free-gb "${PROD_QWEN3_MIN_FREE_GB:-80}" \
  --out "$Q/hf-preflight.json"
step X3-prompts-fetch --soft --cap 1800 --needs X3-hf-preflight --out "$SHAREGPT" -- "$PY" "$SC/prompts.py" fetch --source sharegpt \
  --dest "$DATA"
step X3-prompts-build --soft --cap 1800 --needs X3-prompts-fetch --out "$Q/prompts.jsonl" -- "$PY" "$SC/prompts.py" build \
  --source sharegpt --data "$SHAREGPT" --tokenizer "$QTOK" --n 4096 --seed 0 --max-input 2048 --out-dist sharegpt \
  --out "$Q/prompts.jsonl"
step X3-unlock-gate --soft --always --cap 60 --needs X3-prompts-build -- kit_unlock
step X3-gate60-capture --soft --cap 3600 --needs X3-unlock-gate --out "$TR/qwen3/gate60/serve.nsys-rep,$Q/gate60/manifest.json" -- \
  "${NSYS_A[@]}" -o "$TR/qwen3/gate60/serve" "${QSERVE[@]}" --mode graphs --shared-stream off --min-steps-per-tile 0 \
  --max-seconds 60 --out "$Q/gate60"
step X3-gate60-export --soft --cap 1800 --needs X3-gate60-capture --out "$TR/qwen3/gate60/serve.sqlite" -- "$NSYS" export \
  --type sqlite --force-overwrite true -o "$TR/qwen3/gate60/serve.sqlite" "$TR/qwen3/gate60/serve.nsys-rep"
step X3-gate60-extract --soft --cap 1800 --needs X3-gate60-export --out "$Q/gate60/calls.jsonl|$Q/gate60/calls.shards.json" -- \
  "$PY" "$SC/nsys_extract.py" --run "$Q/gate60" --sqlite "$TR/qwen3/gate60/serve.sqlite" --model $QM
step X3-gate60-sequences --soft --cap 600 --needs X3-gate60-extract -- "$PY" "$SC/nsys_extract.py" check-sequences \
  --run "$Q/gate60" --model $QM
step X3-push-gate60 --soft --cap 3600 -- kit_push "prod extras: X3 Qwen3 gate"
# each capture pushed on its own, its size checked against GitHub's limits first (traces stay off the push)
for lab in A1710 A1410; do
  capture "X3-$lab" "${lab#A}" "$Q/$lab" "$TR/qwen3/$lab" X3-gate60-sequences "${QSERVE[@]}"
  extract "X3-$lab-extract" "$Q/$lab" "$TR/qwen3/$lab" $QM "X3-$lab-export"
  step "X3-$lab-push-size" --soft --cap 120 --needs "X3-$lab-extract" -- "$PY" "$SC/extras_lib.py" push-size --dir "$Q/$lab"
  step "X3-push-$lab" --soft --cap 3600 -- kit_push "prod extras: X3 Qwen3 $lab"
done
# replay C (M6): a 2-per-tile probe ON QWEN3 priced by main's ERNIE probe walls x the active-bytes ratio, then the
# rest priced by the probe's own Qwen3 walls, probe + rest inside the branch's budget
QB="${PROD_QWEN3_REPLAY_BUDGET_S:-1200}"
qreplay() {   # qreplay STEP OUTDIR SAMPLE TAG CAP RUNCAP
  step "$1" --soft --cap "$5" --needs "$7" --out "$2" -- env MOE_RESULTS_DIR="$2" python3 scripts/locked_r3.py \
    --session-tag "$4" --locks $lock --groups 1 --slip-policy cell --min-clean-repeats 0 --run-cap-s "$6" \
    --out-dir "$Q/locked-$(basename "$2")" --r3-script "$SC/replay.py" -- --model $QM --histogram "$3" --branch $b \
    --inst GRR --repeats 9 --duty 0.25 --seed 0
}
for b in decode prefill; do
  lock=1710; src=A1710; n=857
  if [[ $b == prefill ]]; then lock=1410; src=A1410; n=643; fi
  step "X3-replay-probe-sample-$b" --soft --cap 600 --needs "X3-$src-extract" --out "$Q/sample-probe-$b/sample.json" -- \
    "$PY" "$SC/replay.py" sample --run "$Q/$src" --model $QM --branch $b --n $n --per-tile "${PROD_PROBE_PER_TILE:-2}" \
    --wall "$R/replay-probe-$b/replay_wall.json" --scale active-bytes --repeats 9 --budget-s "$QB" \
    --out "$Q/sample-probe-$b"
  qreplay "X3-replay-probe-$b" "$Q/replay-probe-$b" "$Q/sample-probe-$b/sample.json" "prod-extras-qwen3-probe-$b" 1200 \
    1000 "X3-replay-probe-sample-$b"
  step "X3-replay-probe-wall-$b" --soft --cap 120 --needs "X3-replay-probe-$b" --out "$Q/replay-probe-$b/replay_wall.json" -- \
    "$PY" "$SC/replay.py" wall-merge --from "$Q/replay-probe-$b" --model $QM --out "$Q/replay-probe-$b/replay_wall.json"
  step "X3-replay-sample-$b" --soft --cap 600 --needs "X3-replay-probe-wall-$b" --out "$Q/sample-$b/sample.json" -- "$PY" \
    "$SC/replay.py" sample --run "$Q/$src" --model $QM --branch $b --n $n \
    --probe-wall "$Q/replay-probe-$b/replay_wall.json" --repeats 9 --budget-s "$QB" --out "$Q/sample-$b"
  qreplay "X3-replay-$b" "$Q/replay-$b" "$Q/sample-$b/sample.json" "prod-extras-qwen3-replay-$b" 2700 2500 \
    "X3-replay-sample-$b"
done
step X3-unlock-after-replay --always --cap 60 -- kit_unlock
step X3-push-replay --soft --cap 3600 -- kit_push "prod extras: X3 Qwen3 replay C"
# Frontier on Qwen3's shapes: main's venv (frontier_setup.sh ran in main's M2), its full grid, uniform x 3 seeds
if [[ $KIT_DRY -eq 0 && -f "$FR/frontier.env" ]]; then . "$FR/frontier.env"; fi
FSRC="${FRONTIER_SRC:-$FR/src}"
QF="$Q/frontier"
QCSVREL="$("$PY" -c "import sys; sys.path[:0]=['$SC']; import frontier_op as F; print(F.csv_relpath('$QM'))" 2>/dev/null \
        || echo compute/gh200/qwen3-30b-a3b-moe/moe_kernel_only.csv)"
QCSV="$QF/profile/$QCSVREL"
export Q TR QM SC PY NSYS FR_ROUTING FSRC FR_PY_PRIMARY 2>/dev/null || true
qfr_timer() {   # qfr_timer BRANCH LOCK: every grid point of the branch on Qwen3's shapes, one process, eager, under nsys
  local b="$1" lock="$2" d="$Q/frontier/timer-$1" t="$TR/qwen3/frontier/timer-$1"
  mkdir -p "$d" "$t"
  "$NSYS" profile -t cuda --force-overwrite true -o "$t/timer" "$PY" "$SC/frontier_op.py" timer --model "$QM" \
    --branch "$b" --lock "$lock" --routing "$FR_ROUTING" --max-tokens 16384 --out "$d"
}
step X3-frontier-env --soft --cap 60 --needs X3-inputs -- test -s "$FR/frontier.env"
step X3-frontier-grid-check --soft --cap 300 --needs X3-frontier-env -- "$PY" "$SC/frontier_op.py" grid-check --model $QM \
  --src "$FSRC" --max-tokens 16384
for b in decode prefill; do
  lock=1710; [[ $b == prefill ]] && lock=1410
  step "X3-frontier-lock-$b" --soft --always --cap 120 --needs X3-frontier-grid-check -- kit_lock $lock
  step "X3-frontier-timer-$b" --soft --cap 3600 --needs "X3-frontier-lock-$b" \
    --out "$QF/timer-$b/timer.json,$TR/qwen3/frontier/timer-$b/timer.nsys-rep" -- qfr_timer $b $lock
  step "X3-frontier-export-$b" --soft --cap 1800 --needs "X3-frontier-timer-$b" --out "$TR/qwen3/frontier/timer-$b/timer.sqlite" -- \
    "$NSYS" export --type sqlite --force-overwrite true -o "$TR/qwen3/frontier/timer-$b/timer.sqlite" \
    "$TR/qwen3/frontier/timer-$b/timer.nsys-rep"
done
step X3-frontier-unlock --always --cap 60 -- kit_unlock
step X3-frontier-points --soft --cap 1800 --needs X3-frontier-export-decode,X3-frontier-export-prefill --out "$QF/points.json" -- \
  "$PY" "$SC/frontier_op.py" fpoints --model $QM --timer "$QF/timer-decode" --timer-sqlite \
  "$TR/qwen3/frontier/timer-decode/timer.sqlite" --timer "$QF/timer-prefill" --timer-sqlite \
  "$TR/qwen3/frontier/timer-prefill/timer.sqlite" --max-tokens 16384 --out "$QF/points.json"
step X3-frontier-csv --soft --cap 600 --needs X3-frontier-points --out "$QCSV" -- "$PY" "$SC/frontier_op.py" csv --model $QM \
  --points "$QF/points.json" --src "$FSRC" --truth kernel_sum --out "$QF/profile"
step X3-frontier-train --soft --cap 1800 --needs X3-frontier-csv --out "$QF/models" -- "$PY" "$SC/frontier_op.py" train \
  --model $QM --csv "$QCSV" --src "$FSRC" --out "$QF/models"
step X3-frontier-csv-fills --soft --cap 600 --needs X3-frontier-points --out "$QF/profile-fills/$QCSVREL" -- "$PY" \
  "$SC/frontier_op.py" csv --model $QM --points "$QF/points.json" --src "$FSRC" --truth with_fills --out "$QF/profile-fills"
step X3-frontier-train-fills --soft --cap 1800 --needs X3-frontier-csv-fills --out "$QF/models-fills" -- "$PY" \
  "$SC/frontier_op.py" train --model $QM --arm-label cli-fills --csv "$QF/profile-fills/$QCSVREL" --src "$FSRC" \
  --out "$QF/models-fills"
QCALLS=()
for lab in A1710 A1410; do
  if [[ $KIT_DRY -eq 1 || -e "$Q/$lab/calls.jsonl" || -e "$Q/$lab/calls.shards.json" ]]; then QCALLS+=(--calls "$lab=$Q/$lab"); fi
done
QMODELS=()
for m in cli=models cli-fills=models-fills; do
  if [[ $KIT_DRY -eq 1 || -f "$QF/${m#*=}/versions.json" ]]; then QMODELS+=(--models "${m%%=*}=$QF/${m#*=}"); fi
done
step X3-frontier-predict --soft --cap 3600 --needs X3-frontier-env --out "$QF/predictions/index.json" -- "$PY" \
  "$SC/frontier_op.py" predict --model $QM ${QMODELS[@]+"${QMODELS[@]}"} --csv "$QCSV" ${QCALLS[@]+"${QCALLS[@]}"} \
  --src "$FSRC" --out "$QF/predictions"
step X3-push-frontier --soft --cap 3600 -- kit_push "prod extras: X3 Qwen3 Frontier"

# X4 LONG-CONTEXT PREFILL (ERNIE at the 1410 lock): only when asked and only inside the budget
if [[ $LONGCTX -eq 1 ]]; then
  budget X4
  step X4-prompts-fetch --soft --cap 1800 --needs X4-inputs --out "$SHAREGPT" -- "$PY" "$SC/prompts.py" fetch --source sharegpt \
    --dest "$DATA"
  step X4-prompts-build --soft --cap 1800 --needs X4-prompts-fetch --out "$EX/long/prompts.jsonl" -- "$PY" "$SC/prompts.py" \
    build-long --data "$SHAREGPT" --tokenizer "$TOK" --n 96 --seed 0 --min-input 4096 --max-input 16000 --out-dist fixed:16 \
    --out "$EX/long/prompts.jsonl"
  capture X4 1410 "$EX/LC1410" "$TR/LC1410" X4-prompts-build "${LSERVE[@]}"
  extract X4-extract "$EX/LC1410" "$TR/LC1410" $MODEL X4-export
  step X4-push --soft --cap 3600 -- kit_push "prod extras: X4 long-context prefill"
fi

step extras-push-final --soft --no-deadline --cap 3600 -- kit_push "prod extras: done (score on the laptop: scripts/prod/offline.sh)"
step extras-held-back-check --no-deadline --cap 120 -- "$PY" "$SC/prodlib.py" held-back-check \
  --held "${MOE_PUSH_HOME:-$HOME/moe}/push/HELD-BACK.txt" --results "$R"
kit_summary
