#!/usr/bin/env bash
# Every open experiment, as ONE unattended H200 run, in the order their results
# are READ.
#
#   bash scripts/h200_gaps_session.sh --dry-run     # laptop, free: every arm's
#                                                   # own plan, predictions, cost
#   bash scripts/h200_gaps_session.sh               # the pod run, ~150 minutes
#   bash scripts/h200_gaps_session.sh --list        # the arms and what each closes
#   bash scripts/h200_gaps_session.sh --only roofline,noise_floor
#
# WHAT CHANGED AND WHY THIS FILE WAS REWRITTEN ON 2026-09-02. Two parallel
# workflows each wrote a driver; the second one's arms -- bm128_depth,
# noise_floor, anchor, counter -- were the highest-priority items from an
# adversarial evaluation, and the first one's driver overwrote the file that
# scheduled them. Ten scripts sat on disk with four of them unreachable, and the
# surviving schedule led with a BLOCK_M=16 cap test. Since then three findings
# reordered everything:
#
#   * cap = 2*BM/(alpha_fitted*b) is EXACTLY the three-term cap; the published
#     cap numbers stand. What was wrong was the NAME on alpha_fitted.
#   * BLOCK_M=128 is the only tile vLLM runs multi-tile (59 of 87 cells, up to
#     32 tiles per expert). At 16/32/64 it never does. The re-read term, and
#     therefore the ceiling, only fires at 128. Everything else is formula work.
#   * TEMPO states the tile term is inactive in decode. That holds below ~256
#     tokens and fails above. Contesting that sentence with a measurement is
#     what this study has to offer, and it lives entirely at BLOCK_M=128.
#
# THE ORDER IS THE ARGUMENT, so it is stated before the code. The rule behind it:
# anything whose result changes how a later arm is READ runs before that arm.
#
#   0 calibrate      THIS pod's own ceilings. Not optional: five of the arms
#                    below REFUSE without a calibration for the attached device,
#                    and the H200's dense bf16 moved 7.1% between two sessions
#                    while its bandwidth held to 0.014%, so the ridge is not a
#                    constant you can carry over.
#   0 pin_probe      Does MOE_FORCE_TILE reach the kernel on this build. Every
#                    arm below forces a tile and is worthless if the pin is not
#                    honoured. That is the 2026-09-01 S6a failure. Cheap.
#
#   1 roofline       THE CLAIM. At BLOCK_M=128, forced, sweep tokens through the
#                    multi-tile range and measure achieved throughput against
#                    this card's own roof. No fit, no alpha, no anchor, no
#                    estimator: it is immune to every criticism the evaluation
#                    landed. Two outcomes and BOTH are publishable, which is why
#                    it runs WITHOUT --fail-on-gate. If throughput reaches the
#                    roof, arms 2 through 6 are measuring a ceiling that never
#                    binds, and you want to know that at minute ten.
#   2 bm128_depth    The regime every other arm is read in. The whole 128 row
#                    currently rests on two fits across two cards, one of them
#                    on a non-monotone ladder that should have been discarded.
#   3 noise_floor    Nothing above it can be scored without it. The study has
#                    NO true replicates; its closest proxy confounds num_stages.
#
#   4 bn_g16, bn_g1  The only clean separation of alpha_a from alpha_b -- BN
#                    appears in one term of the blend and BM in two -- and the
#                    test of whether the model is COMPLETE: three terms means a
#                    straight line in BM/BN, and structure in the residual names
#                    the missing one. The last attempt was discarded by a check
#                    that tested proportionality and never level; this one
#                    checks level. G=16 primary; G=1 is the production swizzle.
#   5 anchor         alpha_fitted's LEVEL, which the cap divides by. In 12 of 12
#                    fits the measured n=1 tread sits above the fitted branch;
#                    three defensible anchors give 0.45/0.65/0.71 for one cell.
#                    Survives the cap-equivalence finding because a wrong level
#                    is a wrong cap, and at 128 the cap is knife-edge.
#   6 occupancy      Does the standard predictor transfer. Reuse distance says
#                    G=64 should cut the weight re-read to 0.016; measured 0.67.
#                    If alpha tracks RESIDENCY rather than program order, the
#                    swizzle is a dead lever, the cross-card null is explained
#                    (both caches saturated), and TileSight's method does not
#                    apply in this regime -- a correction, not a re-derivation.
#                    P2 is EXPECTED to fail and that FAIL is the finding.
#
#   7 mma_switch     STUDY item 3's loose end, cheap: is the instruction chosen
#                    by the tile alone, at fixed tokens.
#   8 ruler          Prices a ridge change without making one. Last of the
#                    ridge-related arms because it changes how none above is
#                    read: --write-calibration is off.
#   9 cap_test       BLOCK_M=16, DEMOTED. vLLM never runs 16 multi-tile (0 of 24
#                    cells), so this tests the FORMULA, not production. Worth
#                    having; not a headline.
#  10 dtype          Is the 1.15 fp8/bf16 crossing the FORMAT or the CONFIG.
#  11 span           Is the 0.563 the span EXTENT or the KERNEL. Two runs.
#  12 counter_plan   Free. Probes whether a DRAM counter route is open on this
#                    box and prints the manual ncu command if so. A counter is
#                    the ONLY thing that turns alpha_b into a number rather than
#                    an interval, and it is blocked on rented pods.
#
# EVERY ARM IS INDEPENDENT AND NONE IS FATAL. One failing arm records its status
# and the rest continue; a long unattended run that aborts at minute six on a
# shape that does not compile is worse than no automation.
#
# THREE STATES, NOT TWO. DONE means the arm measured and reported (whatever its
# claim gates said). REFUSED means it exited 2 BEFORE measuring -- no
# calibration, a tile that cannot run as pinned, an import that drifted -- and
# it is recorded as such rather than as a gate failure. RETRY is anything else.
# Until 2026-09-02 the refusal paths exited 1 and were indistinguishable from
# "ran and a claim gate failed"; that is fixed in the scripts and honoured here.
#
# WHAT IT NEVER DOES. It never terminates a pod, never commits and never pushes.
# It prints what to commit and stops.
#
# THREE SHELL HABITS THIS FILE AVOIDS ON PURPOSE, each of which has already cost
# this project a run:
#   * NO `set -e`. See above.
#   * NO PIPELINE around a measured command. `cmd > log 2>&1` then `cat log`,
#     never `cmd | tee log`, because a pipeline reports the exit status of its
#     LAST element and the ledger would record the success of `tee`.
#   * NO BACKGROUND JOB, therefore no `$!` and no PID variable read in a shell
#     that started nothing.
set -uo pipefail          # NOT -e: an arm may fail and the run must continue

# --------------------------------------------------------------------------
# Where things are. REPO is derived from this file rather than hardcoded to
# /workspace/repo, so --dry-run works from a checkout on a laptop.
# --------------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY_BASE="${PY_BASE:-/workspace/venvs/base/bin/python}"
PY_VLLM="${PY_VLLM:-/workspace/venvs/vllm/bin/python}"
[[ -x "$PY_BASE" ]] || PY_BASE="$REPO/.venv/bin/python"
[[ -x "$PY_BASE" ]] || PY_BASE="$(command -v python3 || true)"
[[ -x "$PY_VLLM" ]] || PY_VLLM="$PY_BASE"

DRY=0
ONLY=""
LIST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --list)    LIST=1; shift ;;
    --only)    ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,95p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\n==== %s ====\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# --------------------------------------------------------------------------
# Does git keep this path. ASKED, never asserted from a reading of .gitignore:
# the answer differs for results/, results/published/ and a path outside the
# work tree. rc 128 is a path outside the tree -- the pod default /workspace --
# and calling that "tracked" is how output gets written where nobody collects it.
# --------------------------------------------------------------------------
git_note() {
  local p="$1" rc=0
  git -C "$REPO" check-ignore -q -- "$p" >/dev/null 2>&1 || rc=$?
  case "$rc" in
    0) echo "IGNORED by git -- nothing written here enters the repo (the intended deal for raw output; publish with scripts/publish_results.sh)" ;;
    1) echo "git WILL KEEP this path -- anything written here is committable" ;;
    *) echo "UNVERIFIED: git check-ignore exited $rc. Usually the path is outside this work tree, the normal case for /workspace on a pod. It is NOT 'tracked'." ;;
  esac
}

# --------------------------------------------------------------------------
# The arms, declared once, in the order their results are read. The cost table,
# --list and the closing summary all read this ONE list.
# --------------------------------------------------------------------------
ARM_NAMES=(calibrate pin_probe
           roofline bm128_depth noise_floor
           bn_g16 bn_g1 anchor_measure anchor_rescore occupancy
           mma_switch ruler cap_test dtype span span_dense counter_plan)

arm_minutes()  { case "$1" in
  calibrate) echo 3 ;;  pin_probe) echo 4 ;;
  roofline) echo 4 ;;   bm128_depth) echo 12 ;;  noise_floor) echo 25 ;;
  bn_g16) echo 11 ;;    bn_g1) echo 11 ;;        anchor_measure) echo 8 ;;
  anchor_rescore) echo 0 ;;                       occupancy) echo 20 ;;
  mma_switch) echo 7 ;; ruler) echo 4 ;;         cap_test) echo 9 ;;
  dtype) echo 25 ;;     span) echo 30 ;;         span_dense) echo 20 ;;
  counter_plan) echo 1 ;;
esac; }

# EXIT CODES THAT MEAN THE ARM IS FINISHED. Wider than "rc 0" on purpose: a
# CLAIM gate that FAILED is a RESULT in every one of these scripts, and
# re-running the arm would spend the same minutes to obtain the same refutation.
# Exit 2 is never here: it is REFUSED and handled separately in arm().
arm_done_codes() { case "$1" in
  calibrate)  echo 0 ;;
  pin_probe)  echo 0 ;;
  # roofline, bn, occupancy, anchor: 0 = measured, whatever the claims said.
  # None of these is run with --fail-on-gate, so 1 cannot occur by contract.
  roofline)   echo 0 ;;
  bm128_depth) echo 0 ;;
  noise_floor) echo 0 ;;
  bn_g16|bn_g1) echo 0 ;;
  # memory_branch_anchor returns 1 when a CLAIM gate fails, with no
  # --fail-on-gate involved. --rescore already does on the committed corpus:
  # C1 FAILS because every published alpha implies a bandwidth the card does
  # not have. That is the evaluation's weakest-link finding as a scored gate,
  # and it is FINISHED, not RETRY.
  anchor_measure|anchor_rescore) echo 0,1 ;;
  occupancy)  echo 0 ;;
  # 1 = a gate failed, and G4 failing is a result about Triton's predicate.
  mma_switch) echo 0,1 ;;
  ruler)      echo 0 ;;
  cap_test)   echo 0 ;;
  dtype)      echo 0 ;;
  span|span_dense) echo 0 ;;
  # --probe exits 0 only when a route is OPEN and 3 otherwise; BLOCKED is an
  # answer, so 3 is finished too. The plan step afterwards always exits 0.
  counter_plan) echo 0,3 ;;
esac; }

arm_gate_regex() { case "$1" in
  # bm128_roofline renders "[PASS] VALIDITY V0 ..." / "[FAIL] CLAIM    C1 ..."
  # and one verdict line under "## Verdict".
  roofline)   echo '^\[(PASS|FAIL|UNKNOWN)\][[:space:]]+(VALIDITY|CLAIM)|^  (CEILING|STILL RISING|PLATEAU IS NOT|NOT SETTLED)' ;;
  bm128_depth) echo '^\[(PASS|FAIL|UNKNOWN|UNDECIDED)\]|^(P|C|V)[0-9]+[[:space:]]+(PASS|FAIL|UNKNOWN|UNDECIDED)' ;;
  noise_floor) echo '^[[:space:]]*V[0-9][[:space:]]|floor|sigma' ;;
  bn_g16|bn_g1) echo '^\[(PASS|FAIL|UNKNOWN)\]|^(V|C|S)[0-9]+[[:space:]]+(PASS|FAIL|UNKNOWN)' ;;
  anchor_measure|anchor_rescore) echo '^\[(PASS|FAIL|UNKNOWN|UNDECIDED)\]|^(C|V|P)[0-9]+[[:space:]]' ;;
  occupancy)  echo '^\[(PASS|FAIL|UNKNOWN)\]|^(V|P|A)[0-9]+[[:space:]]+(PASS|FAIL|UNKNOWN)' ;;
  cap_test|mma_switch) echo '^[GVC][0-9]+[[:space:]]+(VALIDITY|CLAIM)[[:space:]]' ;;
  dtype)      echo '^\[(PASS|FAIL|UNKNOWN)[[:space:]]*\][[:space:]]+(VALIDITY|CLAIM)' ;;
  span|span_dense) echo '^\[(PASS|FAIL|UNKNOWN)\][[:space:]]+[VC][0-9]' ;;
  ruler)      echo '^(VALIDITY|CLAIM)[[:space:]]+[0-9]+[[:space:]]+(PASS|FAIL|UNDECIDED)' ;;
  pin_probe)  echo '^\[force-tile\] GATE ' ;;
  counter_plan) echo 'OPEN|BLOCKED|REFUSE|THE COMMAND' ;;
  calibrate)  echo '^(ridge|bandwidth|settle|clocks|peak)' ;;
esac; }

arm_closes() { case "$1" in
  calibrate)  echo "This pod's own ridge and both dtype peaks. Five arms below REFUSE without it." ;;
  pin_probe)  echo "The S6a gate ('observed tile_block_m = none'). CLOSES whether MOE_FORCE_TILE reaches the kernel. Every forced-tile arm below depends on it." ;;
  roofline)   echo "THE HEADLINE. Whether BLOCK_M=128 -- the only tile production runs multi-tile -- reaches this card's compute roof. Contests TEMPO's 'the tile term is inactive in decode'. No fit involved; immune to the anchor and estimator critiques." ;;
  bm128_depth) echo "The evaluation's #2: five clean memory-bound treads at 128, monotone. The whole 128 row is currently n=2 across two cards, one on a ladder where time falls as rows rise." ;;
  noise_floor) echo "The evaluation's #3: a real between-replicate sd. The study has none; every effect so far is scored against nothing. Also publishes the num_stages control that would have caught the cross-card null." ;;
  bn_g16)     echo "alpha_a as a fitted slope rather than a two-point guess, and the residual that says whether the three-term model is COMPLETE. The only clean lever on the decomposition." ;;
  bn_g1)      echo "The same at the production swizzle, for comparability with every published arm. C1 reads UNKNOWN by design there." ;;
  anchor_measure) echo "The evaluation's weakest link: the memory-branch level, measured at matched reuse rather than extrapolated. Decides whether any numeric alpha is publishable." ;;
  anchor_rescore) echo "Free: every committed report re-scored under the anchor arm 0 just calibrated, so the size of the correction to every published alpha is known." ;;
  occupancy)  echo "Whether alpha tracks residency or program order. If residency, the swizzle is a dead lever, the cross-card null is explained, and reuse-distance prediction does not transfer to this regime." ;;
  mma_switch) echo "STUDY item 3's loose end. CLOSES whether the tile alone selects the instruction at fixed tokens." ;;
  ruler)      echo "STUDY item 2's follow-up. Prices the read-vs-triad and clocks-first changes on the committed corpus without adopting them." ;;
  cap_test)   echo "FINDINGS' fourth readout, DEMOTED: BLOCK_M=16 never runs multi-tile in production, so this tests the formula, not the claim." ;;
  dtype)      echo "STUDY C2's confound: how much of the 1.15 is the config vLLM resolved differently per dtype." ;;
  span)       echo "The 0.563 EXTENT-versus-KERNEL split, three of four corners." ;;
  span_dense) echo "The same on the grid where C3's mechanism is observable at all." ;;
  counter_plan) echo "Whether a DRAM counter is reachable here. A counter is the only route to alpha_b as a number rather than an interval; on rented pods it is blocked and this records which way." ;;
esac; }

arm_offgpu_gates() { case "$1" in
  roofline)   echo "scripts/bm128_roofline.py --self-test --fail-on-gate  (three planted worlds, exit 0 required)" ;;
  bm128_depth) echo "scripts/bm128_depth.py --self-test  (three worlds from the law)" ;;
  noise_floor) echo "its plan prints the power table; the floor itself needs replicates" ;;
  bn_g16|bn_g1) echo "scripts/bn_decomposition.py --self-test --capability 9.0 --group-m 16 --reps 17 --plant-noise 0.008  (four worlds, four distinct verdicts)" ;;
  anchor_measure|anchor_rescore) echo "scripts/memory_branch_anchor.py --rescore  (free, scores every committed report)" ;;
  occupancy)  echo "scripts/occupancy_vs_swizzle.py --self-test and --audit  (A1 FAILs on the corpus by design)" ;;
  cap_test)   echo "scripts/tile_cap_test.py --self-test 0.558 and --self-test 0.10" ;;
  span|span_dense) echo "scripts/span_extent_separation.py --self-test kernel|extent|neither --densify" ;;
  dtype)      echo "C1 and C2 by --dry-run; C3 by --self-test 2.033|2.400|1.000 --self-test-alpha 0.2" ;;
  ruler)      echo "scripts/ruler_rebaseline.py --corpus-only  (hermetic)" ;;
  mma_switch) echo "its four gates need two real compiles; --dry-run registers thresholds only" ;;
  pin_probe)  echo "F1 and F2 need a vLLM span; on a laptop the plan REFUSES, which is the refusal working" ;;
  counter_plan) echo "scripts/dram_counter_route.py --dry-run and --bracket  (the plan and the counter-free bound)" ;;
  calibrate)  echo "none: it is a measurement and nothing else" ;;
esac; }

if (( LIST )); then
  say "ARMS, in the order their results are read"
  for n in "${ARM_NAMES[@]}"; do
    printf '\n  %-15s ~%s min   finished on exit %s\n' \
      "$n" "$(arm_minutes "$n")" "$(arm_done_codes "$n")"
    printf '    %s\n' "$(arm_closes "$n")"
  done
  exit 0
fi

wanted() {
  [[ -z "$ONLY" ]] && return 0
  case ",$ONLY," in (*",$1,"*) return 0 ;; esac
  return 1
}

# --------------------------------------------------------------------------
# Preflight. Costs nothing; every refusal here saves a pod hour.
# --------------------------------------------------------------------------
say "PREFLIGHT"
[[ -d "$REPO/scripts" ]] || { echo "REFUSED: no scripts/ under REPO=$REPO"; exit 3; }
[[ -x "$PY_BASE" ]] || { echo "REFUSED: no usable base interpreter (set PY_BASE=)"; exit 3; }
note "repo      $REPO"
note "base py   $PY_BASE"
note "vllm py   $PY_VLLM"

# Every script this driver calls must exist and parse. A missing script would
# otherwise surface as an arm RETRY forty minutes in, with the reason buried.
missing=0
for s in calibrate_hardware bm128_roofline bm128_depth replicate_noise_floor \
         bn_decomposition memory_branch_anchor occupancy_vs_swizzle \
         tile_cap_test dtype_tile_confound span_extent_separation \
         ruler_rebaseline dram_counter_route; do
  if [[ ! -f "$REPO/scripts/$s.py" ]]; then note "MISSING scripts/$s.py"; missing=1
  elif ! "$PY_BASE" -m py_compile "$REPO/scripts/$s.py" 2>/dev/null; then
    note "DOES NOT PARSE scripts/$s.py"; missing=1
  fi
done
[[ -f "$REPO/scripts/check_mma_path.sh" ]] || { note "MISSING scripts/check_mma_path.sh"; missing=1; }
if (( missing )); then echo "REFUSED: a script this driver schedules is missing or broken."; exit 3; fi
note "scripts   all 13 present and parse"

CARD="nocard"; CAPABILITY=""; CARD_OK=0
read -r CARD CAPABILITY < <("$PY_BASE" - <<'PY' 2>/dev/null || echo "nocard "
import re
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        slug = re.sub(r"[^a-z0-9]+", "_", p.name.lower()).strip("_")
        print(slug, f"{p.major}.{p.minor}")
    else:
        print("nocard", "")
except Exception:
    print("nocard", "")
PY
)
CAPABILITY="${CAPABILITY:-}"
[[ "$CARD" != "nocard" ]] && CARD_OK=1
note "card      $CARD${CAPABILITY:+  compute capability $CAPABILITY}"

if (( DRY == 0 && CARD_OK == 0 )); then
  echo
  echo "REFUSED: no CUDA device, and this driver measures. Nothing was run."
  echo "  For the review step, off GPU and free:  bash scripts/h200_gaps_session.sh --dry-run"
  exit 3
fi

# NVML clock visibility. bm128_roofline's V3 reads UNKNOWN when nvidia-smi cannot
# report clocks, and its verdict() forces NOT SETTLED on any VALIDITY non-PASS --
# so a restricted container turns the headline arm into a non-result. Check it
# BEFORE spending the arm, not after.
if (( CARD_OK )); then
  if nvidia-smi -q -d CLOCK 2>/dev/null | grep -qE "Graphics[[:space:]]*:[[:space:]]*[0-9]+ MHz"; then
    note "clocks    nvidia-smi reports them; roofline V3 can score"
  else
    note "clocks    nvidia-smi does NOT report clocks on this container."
    note "          roofline's V3 will read UNKNOWN and its verdict NOT SETTLED."
    note "          That is a container restriction, not a measurement. Recorded."
  fi
fi

# The session and results root, with the card IN BOTH NAMES: /workspace/results
# outlives the pod, so without the card a second card resumes the first's
# directories and reports the first's timings under the second's heading.
if [[ -d /workspace ]]; then
  SESSION="${SESSION:-/workspace/session/gaps-$CARD-$(date -u +%Y%m%dT%H%M%SZ)}"
  RESULTS="${MOE_RESULTS_DIR:-/workspace/results}"
else
  SESSION="${SESSION:-$REPO/results/h200_gaps/session-$CARD-$(date -u +%Y%m%dT%H%M%SZ)}"
  RESULTS="${MOE_RESULTS_DIR:-$REPO/results}"
fi
export MOE_RESULTS_DIR="$RESULTS"
LOGS="$SESSION/logs"
if (( DRY )); then LEDGER="$SESSION/ARMS-dryrun.tsv"; else LEDGER="$SESSION/ARMS.tsv"; fi
mkdir -p "$LOGS" || { echo "REFUSED: cannot create $LOGS"; exit 3; }
[[ -f "$LEDGER" ]] || printf 'arm\tstate\trc\tseconds\tlog\n' > "$LEDGER"

CALIB_YAML="$REPO/moe/bench/hardware/measured_$CARD.yaml"
note "session   $SESSION"
note "          $(git_note "$SESSION")"
note "results   $RESULTS   (exported as MOE_RESULTS_DIR to every arm)"
note "          $(git_note "$RESULTS/bm128_roofline")"
note "ledger    $LEDGER"
if (( CARD_OK )); then
  note "calib     $CALIB_YAML"
  note "          $(git_note "$CALIB_YAML")   <- THIS ONE is meant to be committed"
fi

say "WHAT THIS COMMITS YOU TO"
total=0
for n in "${ARM_NAMES[@]}"; do
  wanted "$n" || continue
  m="$(arm_minutes "$n")"; total=$((total + m))
  printf '  %-15s ~%3s min   %s\n' "$n" "$m" "$(arm_closes "$n" | cut -c1-92)"
done
if (( DRY )); then
  note ""
  note "--dry-run: each arm runs its OWN --dry-run. Free, off GPU, seconds."
  note "The minutes above are what the POD run would cost, not this one."
else
  note ""
  note "TOTAL ~$total minutes. The first three arms after calibration -- roofline,"
  note "bm128_depth, noise_floor -- are ~40 min and contain the claim. Arms"
  note "already DONE in $LEDGER are skipped, so a re-run costs only what is left."
fi

started=$(date -u +%s)

# --------------------------------------------------------------------------
# One arm. Three states: DONE, REFUSED (exit 2, measured nothing), RETRY.
# --------------------------------------------------------------------------
arm() {
  local name="$1"; shift
  if ! wanted "$name"; then
    note "SKIP $name (--only $ONLY)"; return 0
  fi
  if (( DRY == 0 )) && \
     awk -F'\t' -v n="$name" '$1 == n && $2 == "DONE" { f = 1 } END { exit !f }' \
       "$LEDGER" 2>/dev/null; then
    note "SKIP $name (already DONE in $LEDGER; delete its row to force a re-run)"
    return 0
  fi
  local log="$LOGS/$name.log" t0 t1 rc
  note "-> $name   log $log"
  t0=$(date -u +%s)
  "$@" > "$log" 2>&1
  rc=$?
  t1=$(date -u +%s)
  local state=RETRY
  if (( DRY )); then
    state=PLANNED
  elif (( rc == 2 )); then
    # REFUSED before measuring. Not DONE (nothing to resume into), not a gate
    # failure (nothing was scored). The first stderr line says why.
    state=REFUSED
  else
    case ",$(arm_done_codes "$name")," in (*",$rc,"*) state=DONE ;; esac
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$state" "$rc" "$((t1 - t0))" "$log" >> "$LEDGER"
  note "   $state (exit $rc) in $((t1 - t0))s"
  if [[ "$state" == "REFUSED" ]]; then
    note "   $(grep -m1 'REFUSED' "$log" || tail -1 "$log")"
  fi
  return 0
}

say "SESSION  card=$CARD  $( ((DRY)) && echo '(DRY RUN: plans only)' || echo '(MEASURING)')"

# --------------------------------------------------------------------------
# 0. THIS POD'S OWN CEILINGS, and whether the pin reaches the kernel.
# --------------------------------------------------------------------------
say "0. calibrate THIS card"
if (( DRY )); then
  note "calibrate_hardware.py is a measurement and has no --dry-run; skipped in a plan."
else
  arm calibrate "$PY_BASE" "$REPO/scripts/calibrate_hardware.py"
fi

say "0. does MOE_FORCE_TILE reach the kernel on this build"
PIN='{"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":64,"GROUP_SIZE_M":1,"num_warps":8,"num_stages":4}'
# --env vllm --impl vllm_fused_experts is what makes candidate_impls plan a vLLM
# span at all; without it only torch's CUTLASS spans are planned and the gate
# reads "observed = none" for the wrong reason. Set on this command, never exported.
if (( DRY )); then
  arm pin_probe env MOE_FORCE_TILE="$PIN" "$PY_VLLM" -m moe.bench.cli \
      --profile profile-cell --groups baselines --env vllm --impl vllm_fused_experts \
      --dry-run --out-dir "$SESSION/pin-probe"
else
  arm pin_probe env MOE_FORCE_TILE="$PIN" "$PY_VLLM" -m moe.bench.cli \
      --profile profile-cell --groups baselines --env vllm --impl vllm_fused_experts \
      --out-dir "$SESSION/pin-probe"
fi

# --------------------------------------------------------------------------
# 1. THE CLAIM. Runs WITHOUT --fail-on-gate: a C1 FAIL means BLOCK_M=128 reached
#    the roof, which is one of the two registered outcomes and a result.
# --------------------------------------------------------------------------
say "1. does BLOCK_M=128, the tile production runs multi-tile, reach the roof"
if (( DRY )); then
  arm roofline "$PY_BASE" "$REPO/scripts/bm128_roofline.py" --dry-run
else
  arm roofline "$PY_VLLM" "$REPO/scripts/bm128_roofline.py" \
      --model mixtral-8x7b --dtype bf16 --control 256 \
      --r-min 32 --r-max 4096 --reps 3 --plateau-doublings 2
fi

say "2. depth at BLOCK_M=128: five clean memory-bound treads, monotone"
if (( DRY )); then
  arm bm128_depth "$PY_BASE" "$REPO/scripts/bm128_depth.py" --dry-run --model mixtral-8x7b
else
  arm bm128_depth "$PY_VLLM" "$REPO/scripts/bm128_depth.py" --model mixtral-8x7b --r-max 2048
fi

say "3. the noise floor, without which nothing above can be scored"
if (( DRY )); then
  arm noise_floor "$PY_BASE" "$REPO/scripts/replicate_noise_floor.py" --dry-run
else
  arm noise_floor "$PY_VLLM" "$REPO/scripts/replicate_noise_floor.py"
fi

# --------------------------------------------------------------------------
# 4-6. WHAT alpha IS MADE OF.
# --------------------------------------------------------------------------
say "4. alpha_a and alpha_b separated, and whether the three-term model is complete"
if (( DRY )); then
  arm bn_g16 "$PY_BASE" "$REPO/scripts/bn_decomposition.py" --dry-run --capability "${CAPABILITY:-9.0}" --group-m 16
  arm bn_g1  "$PY_BASE" "$REPO/scripts/bn_decomposition.py" --dry-run --capability "${CAPABILITY:-9.0}" --group-m 1
else
  # On sm_80 both need --num-stages 3: BM=256 x BN=128 at 4 stages asks 192 KiB
  # against 164. The script refuses and names the fix; we pass it up front.
  STAGES=(); [[ "${CAPABILITY%%.*}" == "8" ]] && STAGES=(--num-stages 3)
  arm bn_g16 "$PY_VLLM" "$REPO/scripts/bn_decomposition.py" --group-m 16 --reps 17 "${STAGES[@]}"
  arm bn_g1  "$PY_VLLM" "$REPO/scripts/bn_decomposition.py" --group-m 1  --reps 17 "${STAGES[@]}"
fi

say "5. the memory-branch anchor, measured rather than extrapolated"
if (( DRY )); then
  arm anchor_measure "$PY_BASE" "$REPO/scripts/memory_branch_anchor.py" --dry-run
else
  arm anchor_measure "$PY_VLLM" "$REPO/scripts/memory_branch_anchor.py" --measure --model mixtral-8x7b
fi
# Free either way: re-scores every committed report under this card's calibration.
arm anchor_rescore "$PY_BASE" "$REPO/scripts/memory_branch_anchor.py" --rescore

say "6. residency or program order: does the standard predictor transfer"
if (( DRY )); then
  arm occupancy "$PY_BASE" "$REPO/scripts/occupancy_vs_swizzle.py" --dry-run
else
  # No stage override: --stages is the residency LADDER, and the script prunes
  # rungs the attached card cannot hold (s=5 on sm_80) from its own capability.
  arm occupancy "$PY_VLLM" "$REPO/scripts/occupancy_vs_swizzle.py" --run
fi

# --------------------------------------------------------------------------
# 7-11. THE DOCS' OWN BACKLOG, in its previous order, after the claim.
# --------------------------------------------------------------------------
say "7. the ISA switch: is the instruction selected by the tile alone"
if [[ -n "$CAPABILITY" && "${CAPABILITY%%.*}" -lt 9 ]]; then
  note "SKIP mma_switch: compute capability $CAPABILITY reaches no warpgroup MMA at any tile."
  printf 'mma_switch\tSKIPPED\t-\t0\tsm<9.0\n' >> "$LEDGER"
elif (( DRY )); then
  arm mma_switch bash "$REPO/scripts/check_mma_path.sh" --dry-run --model deepseek-v3 --tokens 256 \
      --block-m 16,64 --block-n 64 --block-k 64 --group-m 1 --warps 4 --stages 3
else
  # ONE call, two tiles: --block-m takes a list, and everything else is held
  # fixed, so the only thing that moves between the two PTX dumps is BLOCK_M.
  arm mma_switch bash "$REPO/scripts/check_mma_path.sh" --model deepseek-v3 --tokens 256 \
      --block-m 16,64 --block-n 64 --block-k 64 --group-m 1 --warps 4 --stages 3 \
      --out "$SESSION/ptx/mma-switch"
fi

say "8. the ruler: price a ridge change without making one"
if (( DRY )); then
  arm ruler "$PY_BASE" "$REPO/scripts/ruler_rebaseline.py" --dry-run
else
  arm ruler "$PY_BASE" "$REPO/scripts/ruler_rebaseline.py"
fi

say "9. BLOCK_M=16 cap test -- the FORMULA, not the production claim"
if (( DRY )); then
  arm cap_test "$PY_BASE" "$REPO/scripts/tile_cap_test.py" --dry-run --capability "${CAPABILITY:-9.0}"
else
  arm cap_test "$PY_VLLM" "$REPO/scripts/tile_cap_test.py"
fi

say "10. is the 1.15 fp8/bf16 crossing the FORMAT or the CONFIG"
if (( DRY )); then
  arm dtype "$PY_BASE" "$REPO/scripts/dtype_tile_confound.py" --dry-run
else
  arm dtype "$PY_VLLM" "$REPO/scripts/dtype_tile_confound.py"
fi

say "11. is the 0.563 separation the span EXTENT or the KERNEL"
if (( DRY )); then
  arm span       "$PY_BASE" "$REPO/scripts/span_extent_separation.py" --dry-run
  arm span_dense "$PY_BASE" "$REPO/scripts/span_extent_separation.py" --dry-run --densify
else
  arm span       "$PY_VLLM" "$REPO/scripts/span_extent_separation.py" --max-minutes 45
  arm span_dense "$PY_VLLM" "$REPO/scripts/span_extent_separation.py" --densify --max-minutes 35
fi

say "12. is a DRAM counter route open on this box"
if (( DRY )); then
  arm counter_plan "$PY_BASE" "$REPO/scripts/dram_counter_route.py" --dry-run
else
  arm counter_plan "$PY_BASE" "$REPO/scripts/dram_counter_route.py" --probe --out "$SESSION/counter_route.json"
  # The plan is printed regardless: if the probe said OPEN it is the command to
  # type next; if BLOCKED it is what to run on the box where it is not.
  "$PY_BASE" "$REPO/scripts/dram_counter_route.py" --dry-run >> "$LOGS/counter_plan.log" 2>&1 || true
fi

# --------------------------------------------------------------------------
# Every arm's verdict, together, against the item it closes.
# --------------------------------------------------------------------------
if (( DRY )); then
  say "THE PLANS, ARM BY ARM. NOTHING BELOW IS A MEASUREMENT."
else
  say "THE GATES, ARM BY ARM"
fi
for n in "${ARM_NAMES[@]}"; do
  wanted "$n" || continue
  log="$LOGS/$n.log"
  printf '\n--- %s ---\n' "$n"
  printf '  closes: %s\n' "$(arm_closes "$n")"
  state="$(awk -F'\t' -v a="$n" '$1==a{s=$2} END{print s}' "$LEDGER")"
  [[ -n "$state" ]] && printf '  state:  %s\n' "$state"
  if [[ ! -f "$log" ]]; then
    printf '  NO LOG at %s: this arm did not run in this session.\n' "$log"
    continue
  fi
  if [[ "$state" == "REFUSED" ]]; then
    printf '  REFUSED BEFORE MEASURING. Nothing below is a gate:\n'
    grep -m2 'REFUSED' "$log" | sed 's/^/    /'
    continue
  fi
  found=$(grep -Ec "$(arm_gate_regex "$n")" "$log")
  if [[ "$found" == "0" ]]; then
    if (( DRY )); then
      printf '  PLAN ONLY: --dry-run scores no gate for this arm.\n'
      printf '  off GPU its gates are exercised by: %s\n' "$(arm_offgpu_gates "$n")"
    else
      printf '  NO GATE LINE MATCHED /%s/ in %s.\n' "$(arm_gate_regex "$n")" "$log"
      printf '  This arm was NOT scored. Read the log before quoting anything from it.\n'
    fi
    printf '  last 5 lines:\n'
    tail -5 "$log" | sed 's/^/    /'
  else
    grep -E "$(arm_gate_regex "$n")" "$log" | sed 's/^/  /'
  fi
done

say "ARMS"
cat "$LEDGER"
printf '\ntotal %s min of wall clock\n' "$(( ($(date -u +%s) - started) / 60 ))"

say "READ THESE THREE FIRST"
cat <<EOF
  roofline     the ## Verdict line. CEILING BINDING AT THE PRODUCTION TILE is the
               study's claim confirmed; CEILING NOT BINDING means it is refuted
               and the paper becomes a model note. Either is publishable.
  noise_floor  the between-replicate sd. Every effect anywhere in this study is
               to be read against it from now on. If it is above ~0.05, the
               swizzle and footprint effects at G>1 are inside the noise.
  bn_g16       the residual line. Noise-sized residual = the three-term model is
               complete. Structure in it names what is missing.
EOF

say "WHAT TO COMMIT, AND WHAT NOT TO"
cat <<EOF
  NOTHING HERE IS COMMITTED AND NOTHING IS PUSHED. This script runs no git write
  command and terminates nothing. Read the gates above first.

  THE ONE FILE THAT BELONGS IN THE REPO is this card's calibration:
      git -C $REPO diff --stat moe/bench/hardware/
      git -C $REPO add moe/bench/hardware/measured_<this card>.yaml
  Do not commit it if arm 8 (ruler) P1 FAILED: the compute peak was then sampled
  in the wrong clock state and the ridge derived from it is not this card's.

  EVERYTHING ELSE LANDS UNDER $RESULTS, which git ignores by design. To publish
  an arm, copy its run directory under results/published/<date>-<gpu>-<arm>/ --
  the report.json files are kilobytes -- then:
      git -C $REPO check-ignore -v <the path you chose>     # must print NOTHING
  Three published directories were silently dropped by results/* this week
  before that check was made routine.

  THE SESSION DIRECTORY IS $SESSION.
  Copy it off before releasing the pod:
      tar czf /workspace/exfil-gaps-$CARD.tar.gz -C "$(dirname "$SESSION")" "$(basename "$SESSION")"
EOF
