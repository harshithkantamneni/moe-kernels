#!/usr/bin/env bash
# Every open experiment, as ONE unattended pod run, in the order their results
# are READ.
#
#   bash scripts/h200_gaps_session.sh --dry-run     # laptop, free: every arm's
#                                                   # own plan, its cost, the MDE
#   bash scripts/h200_gaps_session.sh               # the pod run, ~3.4 hours
#   bash scripts/h200_gaps_session.sh --list        # the arms and what each closes
#   bash scripts/h200_gaps_session.sh --only calibrate,roofline-n256-g16,noise_floor
#                                                   # a subset. calibrate belongs
#                                                   # in EVERY subset: the ruler
#                                                   # gate below is not scoped to
#                                                   # --only, because nothing that
#                                                   # measures is either.
#
# WHAT THIS FILE IS. A schedule and a ledger. It measures nothing itself: every
# number comes from the script an arm runs, and every verdict comes from that
# script's own gates. What this file owns is the ORDER, the exit-code contract,
# and the closing summary -- and each of those three has already been wrong in a
# way that cost a rented pod.
#
# WHAT CHANGED ON 2026-09-02, second rewrite, after the standards audit. Seven
# defects, each of which made a broken run look like a clean one:
#
#   * THE EXIT-CODE CONTRACT WAS INVERTED AND PER-ARM. This file read 2 as
#     REFUSED and kept a hand-maintained list of "done" codes per arm, while
#     memory_branch_anchor.py documented the opposite (2 = a VALIDITY gate
#     failed AFTER an eight-minute measurement, 3 = nothing measured) and three
#     other scripts refused with 3. A measured-and-invalid anchor run was
#     therefore logged REFUSED and printed "REFUSED BEFORE MEASURING", and
#     genuine refusals were queued as retries. Both lists are gone. There is one
#     table, in moe/bench/exit_codes.py, and `ledger_state` below is its shell
#     mirror; tests/test_h200_gaps_session.py asserts the two agree code for
#     code, so they cannot drift.
#   * THE SUMMARY GREPPED PROSE. `floor|sigma` matched a REFUSED noise-floor log
#     eighteen times and printed `floor: 0.0905` -- an IMPORTED prior, not a
#     measurement -- and `C1 ... [PASS]` -- a pre-registered expectation, not a
#     result -- under the heading "THE GATES". The summary now greps `^RESULT: `
#     and nothing else, the one line moe/bench/exit_codes.result_line renders and
#     parse_result_lines reads back. An arm that printed no RESULT line was not
#     scored, and the summary says exactly that instead of finding words.
#   * A TRACEBACKING --dry-run LOOKED LIKE A CLEAN PLAN. Every dry-run exit code
#     was mapped to PLANNED. A plan now records its rc and gets one of three
#     words, and the session exits non-zero if any plan is BROKEN.
#   * THE CARD WAS MISSING FROM THE RESULTS ROOT. The comment said the card was
#     "IN BOTH NAMES" while the export was a bare /workspace/results, which
#     outlives the pod: a second card resumed the first card's directories and
#     published the first card's timings under the second card's heading. That
#     is a committed collision, not a hypothetical (provenance.py, collision 2).
#   * THE HEADLINE ARM COULD NOT CONFIRM ITS OWN CLAIM. It ran BLOCK_N=64,
#     GROUP_SIZE_M=1 -- the study's swept configuration -- under a heading about
#     what production runs. vLLM 0.27.1's tuned entry for this shape
#     (moe/bench/hardware/vllm_configs/E=8,N=14336,device_name=NVIDIA_H200.json)
#     is BLOCK_N=256 with GROUP_SIZE_M=16, and 32 at 2048 tokens. Three roofline
#     arms now run: the swept-configuration control, which can REFUTE and not
#     confirm, and the two production configurations, which can confirm.
#   * THE PIN WAS PROBED AT A CONFIGURATION NO ARM RUNS. The probe pinned
#     BLOCK_N=128 while every arm pins 64 (or, now, 256). A pin that reaches the
#     kernel at one BLOCK_N is evidence about that BLOCK_N.
#   * THE ONE TABLE WAS ADOPTED HERE AND NOWHERE ELSE, SILENTLY. Adopting
#     moe/bench/exit_codes fixed this file's reading and fixed none of the
#     twenty scripts it reads. That has since been closed for the measuring
#     arms: every script this driver runs to time a cell imports the module,
#     which is why `adopts_exit_codes` is a live check per row rather than a
#     list maintained here. What remains unadopted is the analysis and probe
#     tail, and one of them is an arm: dram_counter_route.py returns 3 for
#     every verdict that is not
#     OPEN -- so a BLOCKED counter route, which is that arm's registered ANSWER
#     on a rented pod, lands in the ledger as INVALID and is described to the
#     operator as measured-and-unquotable. The dry-run said this, for free,
#     where it costs nothing; the pod run said nothing, where it costs an arm.
#     The states are NOT patched per arm -- that is the list R1 deleted. Instead
#     `adopts_exit_codes` ASKS each arm's file whether it imports the module,
#     and `contract_caveat` / `contract_disclosure` print, next to every REFUSED
#     or INVALID row that came from a file which does not, what the state may
#     actually mean and in which direction. When a sibling slice lands the fix,
#     the caveat stops printing on its own; nothing here has to be remembered.
#
# WHAT CHANGED ON 2026-09-02, third pass, after the whole-repo verification.
# Five more, and the first two were introduced by the rewrite above:
#
#   * THE SESSION NEVER PUBLISHED THE CALIBRATION IT MEASURED. Arm 0 ran
#     calibrate_hardware.py bare. That script's default output had moved to an
#     untracked session path and the copy into the tracked tree had become a
#     separate --publish decision, so arm 0 spent three minutes measuring THIS
#     card's ridge into a directory nothing reads while arms 1-9 resolved theirs
#     from whatever measured_<card>.yaml the LAST rental left in the checkout --
#     and labelled it "measured on this machine". That is the audit's own "a
#     constant from another machine presented as a measurement", recreated by
#     the fix for a different one, and the closing `git diff --stat
#     moe/bench/hardware/` would have shown zero changes and read as agreement.
#     The flag is passed now, and the gate that follows arm 0 REFUSES the rest
#     of the session unless BOTH halves answer: the tracked yaml for this card
#     carries a provenance.utc at or after this session's own start, AND the
#     calibrate row in this session's ledger reads DONE. The second half is not
#     redundant -- calibrate_hardware.py publishes the yaml BEFORE it scores a
#     gate, so a run that lands INVALID on clock_established leaves a
#     fresh-stamped ruler nothing verified behind it, and the first version of
#     this gate read that as "measured in THIS session". The committed H200 yaml
#     carries no provenance block at all, so it reads UNDATED and does not pass.
#   * THE TWO SPAN ARMS DERIVED ONE RUN ID. --densify became the default while
#     the driver still booked `span_dense` with it and `span` bare, so both
#     densified, both hashed to the same plan, and the second restored the
#     first's rows, measured nothing and landed DONE with 30 minutes booked.
#     The sparse arm names --no-densify now, which is in the key.
#   * THE DISCLOSURE COVERED TWO STATES AND THE NON-ADOPTERS SPEND NEITHER.
#     contract_caveat and contract_disclosure fired only on REFUSED and INVALID
#     while check_mma_path.sh spent 1 on a failed gate and moe/bench/cli.py
#     spent 4 on one, so a ledger holding only `mma_switch CLAIM_FAIL 1`
#     printed the ALL-CLEAR over the one row whose word was wrong. Which files
#     have adopted is still ASKED of the files, so a sibling slice landing a
#     fix turns the disclosure off by itself.
#   * AN EXIT 1 THIS FILE CANNOT READ WAS LATCHED AS A RESULT. Python spends 1
#     on any exception that escapes main, and at this pass three of the
#     thirteen Python files this driver runs install the ERROR(4) top-level
#     handler that would say so (calibrate_hardware, bm128_roofline,
#     memory_branch_anchor), so a crash in any of the other ten was filed as
#     CLAIM_FAIL, "a result, never re-run", and the arm was skipped forever. An
#     exit 1 from a file that has not adopted the table is recorded UNKNOWN and
#     not latched.
#   * DRY MODE COULD NO LONGER TELL A PLAN FROM A REFUSAL. Once every script
#     adopted the table its --dry-run began exiting REFUSED, correctly, and
#     eleven of the sixteen planned arms collapsed onto PLAN_REFUSED. `dry_state` reads
#     the log now: a plan that printed and then refused is PLANNED, a refusal
#     with nothing printed before it is PLAN_REFUSED.
#
# THE THREE FINDINGS THAT SET THE ORDER, restated because two of them were
# retracted since this file last said them:
#
#   * THE CAP IDENTITY IS RETRACTED. "cap = 2*BM/(alpha_fitted*b) is EXACTLY the
#     three-term cap, so the published cap numbers stand" was a tautology: the
#     test that checked it defined alpha_fitted by the formula it was testing.
#     The estimator actually in use is LadderFit.alpha = B/(A+B), which is
#     (alpha_b + phi)/(1 + phi + delta) -- the (EXA) form in
#     moe/bench/ai_model.py -- not the (LIN) blend the identity assumed. A cap
#     built from a fitted alpha is high by (1 + phi + delta): 31% at
#     BLOCK_M=128, which is larger than the cap-to-ridge gap it was being used
#     to decide. No published cap number is quotable until it is re-derived
#     through (EXA) on ladders re-timed with moe/bench/timing.time_kernel.
#   * THE REGIME, counted on max rows -- what moe_align_block_size actually pads
#     to -- and on UNIFORM routing: BLOCK_M=128 is the only tile vLLM runs
#     multi-tile in more than one cell, 65 of 87 cells, up to 33-34 tiles per
#     expert. BLOCK_M=16 (1 of 24) and BLOCK_M=64 (5 of 112) fire in isolated
#     cells. SKEWED ROUTINGS WERE NEVER COUNTED and production routing is
#     skewed, so this is a statement about uniform routing and about nothing
#     else. The earlier "59 of 87, up to 32, and 16/32/64 never" was counted on
#     seed rows rather than padded rows.
#   * TEMPO states the tile term is inactive in decode. Contesting that sentence
#     with a measurement is what this study has to offer, and it lives at
#     BLOCK_M=128 in the configuration production ships, which is why the two
#     BLOCK_N=256 arms exist and why the BLOCK_N=64 arm is labelled a control.
#
# THE ORDER IS THE ARGUMENT, so it is stated before the code. The rule behind it:
# anything whose result changes how a later arm is READ runs before that arm.
#
#   0 calibrate      THIS pod's own ceilings. Not optional: five of the arms
#                    below REFUSE without a calibration for the attached device,
#                    and the H200's dense bf16 moved 7.1% between two sessions
#                    while its bandwidth held to 0.014%, so the ridge is not a
#                    constant you can carry over.
#   0 pin_probe-*    Does MOE_FORCE_TILE reach the kernel on this build, AT THE
#                    CONFIGURATION THE ARMS RUN -- once at BLOCK_N=64 for the
#                    control arm and every alpha arm, once at BLOCK_N=256 for
#                    the production arms. That is the 2026-09-01 S6a failure,
#                    and a probe at a third BLOCK_N would have been evidence
#                    about a configuration nothing runs. Cheap.
#
#   1 roofline-n64-g1     THE CONTROL, not the claim. BLOCK_M=128 forced at the
#                    study's swept configuration (BLOCK_N=64, GROUP_SIZE_M=1).
#                    Production never ships this shape, so a plateau here
#                    REFUTES nothing about production and CONFIRMS nothing about
#                    it either: if this reaches the roof, every richer
#                    configuration does too, and arms 2-6 are measuring a
#                    ceiling that never binds. That asymmetry is the whole
#                    reason it runs first, and its predicted outcome from the
#                    published G=1 ladders is already known (gap 0.02-0.06
#                    against a 0.10 separation threshold: NOT TILE-ATTRIBUTABLE).
#   1 roofline-n256-g16   THE CLAIM. The configuration vLLM actually ships for
#                    mixtral at BLOCK_M=128 on the H200, at every token count
#                    from 512 up. This is the only arm in the session that can
#                    CONFIRM the ceiling, and it has never been run. It REFUSES
#                    today, free and before the pod, because no BLOCK_M=256
#                    control fits at BLOCK_N=256; see the note above the arm.
#   1 roofline-n256-g32   The same at the swizzle vLLM ships at 2048 tokens,
#                    which is inside the multi-tile range the claim is about.
#                    Four minutes; without it the claim rests on one swizzle.
#   2 bm128_depth    The regime every other arm is read in. The whole 128 row
#                    currently rests on two fits across two cards, one of them
#                    on a non-monotone ladder that should have been discarded.
#   3 noise_floor    Nothing above it can be scored without it. The study has
#                    NO true replicates; its closest proxy confounds num_stages
#                    and is the sigma the MDE line below is forced to assume.
#
#   4 bn_g16, bn_g1  The only clean separation of alpha_a from alpha_b -- BN
#                    appears in one term of the blend and BM in two -- and the
#                    test of whether the model is COMPLETE: three terms means a
#                    straight line in BM/BN, and structure in the residual names
#                    the missing one. G=16 is the primary arm. G=1 is NOT a
#                    second reading of C2: at GROUP_SIZE_M=1 the planted MISSING
#                    world passes C2 by construction, so C2 there is UNKNOWN
#                    however the data fall, and the arm is scheduled for
#                    alpha_b and C3 at the swizzle every published arm swept.
#   5 anchor         alpha_fitted's LEVEL, which the cap divides by. In 12 of 12
#                    fits the measured n=1 tread sits above the fitted branch;
#                    three defensible anchors give 0.45/0.65/0.71 for one cell.
#                    A wrong level is a wrong cap, and at 128 the cap is
#                    knife-edge.
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
#   9 cap_test       BLOCK_M=16, DEMOTED. vLLM runs 16 multi-tile in 1 of 24
#                    cells, so this tests the FORMULA, not production.
#  10 dtype          Is the 1.15 fp8/bf16 crossing the FORMAT or the CONFIG.
#  11 span_dense     Is the 0.563 the span EXTENT or the KERNEL. The DENSE grid
#                    runs first: it is the only grid on which C3's mechanism is
#                    observable at all, and the sparse grid's own kernel world
#                    predicts C2 FAIL. A CLAIM gate failing there is a RESULT
#                    (exit 1, CLAIM_FAIL), never a retry.
#  11 span           The sparse grid, second, for the extent comparison.
#  12 counter_plan   Free. Probes whether a DRAM counter route is open on this
#                    box and prints the manual ncu command if so. A counter is
#                    the ONLY thing that turns alpha_b into a number rather than
#                    an interval, and it is blocked on rented pods.
#
# EVERY ARM IS INDEPENDENT AND NONE IS FATAL. One failing arm records its status
# and the rest continue; a long unattended run that aborts at minute six on a
# shape that does not compile is worse than no automation.
#
# THE STATES. Measuring, straight from moe/bench/exit_codes.py:
#
#   DONE        0  measured, every VALIDITY and CLAIM gate PASSED
#   CLAIM_FAIL  1  measured, VALIDITY passed, a CLAIM gate did not. A RESULT.
#                  The arm is FINISHED and is never re-run: repeating it spends
#                  the same minutes to obtain the same refutation.
#   REFUSED     2  nothing measured, a precondition was not met. Free. A re-run
#                  of this driver DOES re-attempt it, because a refusal costs no
#                  pod minutes and the usual reason to re-run is that the
#                  precondition was fixed.
#   INVALID     3  measured, then a VALIDITY gate FAILED. Nothing on the page
#                  may be quoted, and the arm is NOT auto-retried: the cause is
#                  the instrument, and repeating it repeats the failure unless
#                  the log says in words that it was transient.
#   RETRY     4+  a code nobody chose. Read the log.
#
# Planning (--dry-run), which is a different question and gets different words:
#
#   PLANNED       the arm printed its plan (rc 0)
#   PLAN_REFUSED  the arm refused to plan (rc 2), which off a GPU box is the
#                 refusal working and costs nothing
#   BROKEN        anything else, a traceback included. The plan on the page is
#                 not a plan, and the session exits non-zero.
#   NOT_PLANNED   the arm has no plan mode and was not run. It is named with the
#                 reason instead of being silently absent.
#
# WHAT IT NEVER DOES. It never terminates a pod, never commits and never pushes.
# It prints what to commit and stops. It also never writes into the work tree:
# the anchor re-score is given an --out-dir under the session directory, because
# its default is results/published/ and it used to rewrite two TRACKED files on
# every --dry-run.
#
# THREE SHELL HABITS THIS FILE AVOIDS ON PURPOSE, each of which has already cost
# this project a run:
#   * NO `set -e`. See above.
#   * NO PIPELINE around a measured command. `cmd > log 2>&1` then read the log,
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

# The driver's own exit codes speak the same table as the arms'. A stop before
# anything is measured is REFUSED (2); a plan that did not survive its own
# --dry-run is INVALID (3), because what is on the page is not a plan.
RC_REFUSED=2
RC_INVALID=3
RC_RETRY=4

DRY=0
ONLY=""
LIST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --list)    LIST=1; shift ;;
    --only)    ONLY="$2"; shift 2 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
                 "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "REFUSED: unknown argument: $1" >&2; exit "$RC_REFUSED" ;;
  esac
done

# >>> LIFTABLE: function definitions only, no top-level statements.
# tests/test_h200_gaps_session.py evaluates everything between these two markers
# in its own shell and calls the functions with planted inputs, because the
# states below are exactly what cannot be checked by reading a pod log an hour
# later. Nothing in here may run at source time or the test would run the
# session.

say()  { printf '\n==== %s ====\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# THE SHELL MIRROR OF moe/bench/exit_codes.ledger_state. Kept here rather than
# shelling out to python once per arm, and pinned by a test that compares this
# function with that function for every code from 0 to 6 plus 127 and 130. If
# the table ever changes, that test fails before a pod does.
ledger_state() { case "$1" in
  0) echo DONE ;;
  1) echo CLAIM_FAIL ;;
  2) echo REFUSED ;;
  3) echo INVALID ;;
  *) echo RETRY ;;
esac; }

# WHICH FILE EACH ARM ACTUALLY RUNS, repo-relative, so this driver can ASK that
# file whether it speaks the table above instead of assuming it does. It decides
# NO arm's state: R1 deleted the per-arm done-code lists and this restores none.
# `ledger_state` is still the only thing that turns an exit code into a word.
arm_script() { case "$1" in
  calibrate)                     echo scripts/calibrate_hardware.py ;;
  pin_probe-*)                   echo moe/bench/cli.py ;;
  roofline-*)                    echo scripts/bm128_roofline.py ;;
  bm128_depth)                   echo scripts/bm128_depth.py ;;
  noise_floor)                   echo scripts/replicate_noise_floor.py ;;
  bn_g16|bn_g1)                  echo scripts/bn_decomposition.py ;;
  anchor_measure|anchor_rescore) echo scripts/memory_branch_anchor.py ;;
  occupancy)                     echo scripts/occupancy_vs_swizzle.py ;;
  mma_switch)                    echo scripts/check_mma_path.sh ;;
  ruler)                         echo scripts/ruler_rebaseline.py ;;
  cap_test)                      echo scripts/tile_cap_test.py ;;
  dtype)                         echo scripts/dtype_tile_confound.py ;;
  span|span_dense)               echo scripts/span_extent_separation.py ;;
  counter_plan)                  echo scripts/dram_counter_route.py ;;
  *)                             echo "" ;;
esac; }

# Does the file an arm runs speak moe/bench/exit_codes' table. ASKED of the file
# on every run rather than carried in a list here, because a list of who has
# adopted goes stale the day someone adopts, which is the failure mode R1
# removed. rc 0 adopted, rc 1 not, rc 2 UNKNOWN -- a file this cannot find, which
# is not the same answer as adopted and is treated here as not.
adopts_exit_codes() {
  local rel="$1"
  [[ -n "$rel" && -f "${REPO:-}/$rel" ]] || return 2
  grep -q 'moe\.bench\.exit_codes\|moe\.bench import exit_codes' -- "${REPO:-}/$rel"
}

# WHAT A ROW MAY ACTUALLY MEAN when the file that produced it has not adopted
# the table this session reads it by. Prints nothing -- rc 1 -- for a file that
# has adopted, so it is silent on every row whose word is trustworthy. It
# changes no state and no exit code; the whole content is disclosure, which is
# what the measuring path did not have.
#
# IT USED TO COVER TWO STATES AND THE TWO NON-ADOPTERS SPENT NEITHER. REFUSED
# and INVALID were the only branches, while scripts/check_mma_path.sh spent 1
# on a failed gate and moe/bench/cli.py spent 4 on one, so a ledger holding
# nothing but `mma_switch CLAIM_FAIL 1` walked every row, matched none, and
# printed the ALL-CLEAR over the single row whose word was wrong. Every state a
# non-adopting file can produce is covered here now, and the all-clear says
# "in any state" rather than naming two.
contract_caveat() {
  local name="$1" state="$2" rel why rc=0
  rel="$(arm_script "$name")"
  adopts_exit_codes "$rel" || rc=$?
  case "$rc" in
    0) return 1 ;;
    2) why="which this driver could not find, so whether it speaks that\n  module is UNKNOWN -- and UNKNOWN is not the same answer as adopted" ;;
    *) why="which does not import moe/bench/exit_codes" ;;
  esac
  case "$state" in
    REFUSED)
      printf '  CAVEAT: this row came from %s,\n' "${rel:-a command outside scripts/}"
      printf '  %b.\n' "$why"
      printf '  It therefore reads REFUSED for one reason and one only: the command\n'
      printf '  exited 2. A file that has not adopted the table may spend 2 on\n'
      printf '  something else. scripts/memory_branch_anchor.py used to document 2\n'
      printf '  as a VALIDITY gate that failed AFTER an eight-minute measurement,\n'
      printf '  the opposite reading, under which such a run is unquotable rather\n'
      printf '  than free; it has since adopted the table, so it no longer reaches\n'
      printf '  this caveat, and that is the shape of the risk for whatever has not.\n'
      printf '  Read the log before believing that nothing was measured, and\n'
      printf '  before re-running it.\n' ;;
    INVALID)
      printf '  CAVEAT: this row came from %s,\n' "${rel:-a command outside scripts/}"
      printf '  %b.\n' "$why"
      printf '  It therefore reads INVALID for one reason and one only: the command\n'
      printf '  exited 3. A file that has not adopted the table may spend 3 on a\n'
      printf '  REFUSAL, which measured nothing and costs nothing to re-run, or on\n'
      printf '  a registered ANSWER: scripts/dram_counter_route.py returns 3 for\n'
      printf '  every verdict that is not OPEN, and BLOCKED on a rented pod is what\n'
      printf '  that arm exists to find out, not a broken instrument. INVALID rows\n'
      printf '  are latched and skipped on every later run; delete this row from\n'
      printf '  the ledger to run the arm again.\n' ;;
    CLAIM_FAIL|UNKNOWN)
      printf '  CAVEAT: this row came from %s,\n' "${rel:-a command outside scripts/}"
      printf '  %b.\n' "$why"
      printf '  The command exited 1. Under the table that is CLAIM_FAIL: measured,\n'
      printf '  VALIDITY passed, a pre-registered CLAIM did not -- a RESULT, and the\n'
      printf '  one state this ledger LATCHES as finished so the arm is never spent\n'
      printf '  again. From a file that has not adopted the table, 1 is three things\n'
      printf '  at once. scripts/check_mma_path.sh documents "1 a gate failed" and\n'
      printf '  spends it on the instruction-follows-the-tile reading, which is a\n'
      printf '  VALIDITY gate, AND on "no interpreter at $PY" and "no .ptx under the\n'
      printf '  dump dir", which measured nothing and are refusals. And 1 is what\n'
      printf '  Python returns for any exception that escapes main, which is ERROR.\n'
      printf '  So this driver records UNKNOWN and does NOT latch the row: a state\n'
      printf '  it cannot tell apart is not a result it may file. Read the log --\n'
      printf '  the three cases do not resemble each other in it -- and delete or\n'
      printf '  keep the row deliberately.\n' ;;
    DONE)
      printf '  CAVEAT: this row came from %s,\n' "${rel:-a command outside scripts/}"
      printf '  %b.\n' "$why"
      printf '  It therefore reads DONE for one reason and one only: the command\n'
      printf '  exited 0. Under the table 0 means every VALIDITY and every CLAIM\n'
      printf '  gate PASSED. A file that has not adopted it may spend 0 on a run\n'
      printf '  that scored no gate at all: scripts/check_mma_path.sh exits 0 from\n'
      printf '  its own --dry-run, and again from the ladder path whose closing\n'
      printf '  lines say the cell cannot attribute the instruction to a tile.\n'
      printf '  DONE is latched and skipped on every later run, so read the RESULT\n'
      printf '  lines above before taking this row for gates that passed. A check\n'
      printf '  that examined nothing reports no failures.\n' ;;
    RETRY)
      printf '  CAVEAT: this row came from %s,\n' "${rel:-a command outside scripts/}"
      printf '  %b.\n' "$why"
      printf '  It therefore reads RETRY only because the command exited a code the\n'
      printf '  table does not name, and this session exits 4 over it and the next\n'
      printf '  one attempts the arm again. A file that has not adopted the table\n'
      printf '  may spend such a code on a REGISTERED ANSWER rather than a crash:\n'
      printf '  moe/bench/cli.py returns 4 when the implementations ran under the\n'
      printf '  pin and no row showed that tile, which is the VALIDITY failure that\n'
      printf '  probe exists to detect, not an unplanned exception. Re-running it\n'
      printf '  spends the minutes again to reach the same number. Read the log\n'
      printf '  before the next session does.\n' ;;
    *) return 1 ;;
  esac
  return 0
}

# THE MEASURING PATH'S COUNTERPART TO THE DRY-RUN BANNER, which said only under
# --dry-run, where it costs nothing, that a refusal exiting 3 is a refusal
# wearing an INVALID's number. On the pod the same collision costs an arm and
# nothing said it. Reads the ledger rather than the arm list so it can be tested
# with a planted ledger, and so an arm added later is covered without being
# named twice. Both branches print: the all-clear is a sentence, not silence,
# because an empty section reads as nothing to report.
contract_disclosure() {
  local ledger="$1" n state rest rel any=0
  while IFS=$'\t' read -r n state rest; do
    # EVERY STATE A NON-ADOPTING FILE CAN PRODUCE, not the two this used to
    # name. The header row and NOT_PLANNED fall out here because neither is a
    # state a command exited with.
    case "$state" in
      DONE|CLAIM_FAIL|REFUSED|INVALID|RETRY|UNKNOWN) ;;
      *) continue ;;
    esac
    rel="$(arm_script "$n")"
    [[ -n "$rel" ]] && adopts_exit_codes "$rel" && continue
    if (( any == 0 )); then
      say "THE ROWS WHOSE STATE MAY BE THE WRONG WORD"
      printf '  This session reads every exit code through moe/bench/exit_codes.\n'
      printf '  The files below have not adopted that module, so their codes were\n'
      printf '  chosen against some other table and the state in the ledger is a\n'
      printf '  translation nobody agreed to. Nothing here is patched per arm; the\n'
      printf '  fix belongs in those files, through exit_codes.classify.\n\n'
      any=1
    fi
    printf '  %-19s %s   from %s\n' "$n" "$state" "${rel:-a command outside scripts/}"
    contract_caveat "$n" "$state"
    printf '\n'
  done < "$ledger"
  if (( any == 0 )); then
    say "THE EXIT-CODE CONTRACT"
    printf '  No row in this session, in ANY state and not only REFUSED and\n'
    printf '  INVALID, came from a file that has not adopted moe/bench/exit_codes,\n'
    printf '  so every state word above is the one that table defines.\n'
  fi
  return 0
}

# A PLAN IS NOT A MEASUREMENT, so a plan does not get measuring words. Three
# outcomes, and only one of them is silent-failure-shaped: a --dry-run that
# tracebacks exits 1, which lands in BROKEN and stops the session, which is the
# defect this mapping exists to catch.
#
# rc 2 IS NOT A WORD ON ITS OWN, and that is the 2026-09-02 regression. When
# every script adopted the table, every one of their --dry-runs began exiting
# REFUSED -- correctly: a plan measured nothing and scored no gate -- so eleven
# of the sixteen planned arms landed on PLAN_REFUSED and DRY mode lost the
# only distinction it exists to draw. The log still carries it, so rc 2 is
# decided by `printed_a_plan` and not by the number: a plan that printed and
# then refused is PLANNED, a refusal with nothing printed before it is
# PLAN_REFUSED. Called with no log, rc 2 is PLAN_REFUSED: nothing printed.
dry_state() { case "$1" in
  0) echo PLANNED ;;
  2) if printed_a_plan "${2:-}"; then echo PLANNED; else echo PLAN_REFUSED; fi ;;
  *) echo BROKEN ;;
esac; }

# The P1 check, re-asked after every arm rather than once at preflight. Two arms
# used to write into the tracked tree while the session ran -- calibrate_hardware
# into moe/bench/hardware/measured_<card>.yaml and the anchor --rescore into
# results/published/ANCHOR_RESCORE.{txt,json} -- so "0 dirty files" at preflight
# was true and meaningless, and 44,872 published rows carry git_dirty=True.
# --untracked-files=all counts FILES, matching moe/bench/provenance.py. Prints
# "-" when git cannot answer, which is not the same number as 0.
dirty_count() {
  local out rc=0
  out="$(git -C "$REPO" status --porcelain --untracked-files=all 2>/dev/null)" || rc=$?
  if (( rc != 0 )); then echo "-"; return 0; fi
  if [[ -z "$out" ]]; then echo 0; else printf '%s\n' "$out" | wc -l | tr -d ' '; fi
}

# AN ISO-8601 UTC STAMP AS THE INTEGER YYYYMMDDhhmmss, or nothing at all when
# the string does not carry one. Compared as an integer rather than through
# `date -d`, which is GNU-only and absent on the laptop half of this project,
# and rather than with `[[ a > b ]]`, which collates by locale: which machine
# reads the clock must not decide whether a session runs. Returns 1 on a string
# it cannot read, so no caller can mistake "no timestamp" for "an old one".
utc_stamp() {
  local raw="${1:-}" digits
  raw="${raw%%+*}"
  digits="$(printf '%s' "$raw" | tr -cd '0-9')"
  (( ${#digits} >= 14 )) || return 1
  printf '%s\n' "${digits:0:14}"
}

# WHEN THE TRACKED CALIBRATION SAYS IT WAS MEASURED, as that integer.
# calibrate_hardware.py writes moe/bench/provenance.provenance_block under a
# top-level `provenance:` key, and `utc` is the one field in it that answers
# "was this THIS rental". Read with awk over that block alone: `detail:` carries
# a hundred nested keys and a grep for `utc:` over the whole file would find
# whichever of them came first. rc 2 no such file, rc 3 a file with no stamp.
calibration_stamp() {
  local yaml="$1" raw
  [[ -f "$yaml" ]] || return 2
  raw="$(awk '/^provenance:/ {p = 1; next}
              p && /^[^[:space:]]/ {p = 0}
              p && $1 == "utc:" {print $2; exit}' "$yaml" | tr -d "\"'")"
  [[ -n "$raw" ]] || return 3
  utc_stamp "$raw"
}

# WHOSE RULER THE ARMS BELOW WILL BE SCORED AGAINST. One word, so both branches
# are plantable with a yaml and a stamp and neither is reachable only on a pod:
#   PUBLISHED   the tracked file for this card exists and carries a stamp at or
#               after this session's own start. Arm 0 measured THIS card and
#               published it, which is the only state the rest may run in.
#   MISSING     no tracked calibration for this card at all: every arm that
#               needs a ridge would refuse, one at a time, for 3.4 hours.
#   UNDATED     a tracked file with no provenance.utc. It cannot say which
#               rental measured it, and "cannot say" is not "this one". The
#               committed measured_nvidia_h200.yaml is exactly this shape.
#   STALE       a tracked file measured BEFORE this session began -- the
#               previous rental's, still in the checkout. This is the state the
#               A6 fix created: arm 0 measures this card into an untracked
#               session path and every later arm quotes the last pod's ridge
#               while `ridge_source` names the attached device. The audit's own
#               "a constant from another machine presented as a measurement".
#   NO_BASELINE this driver could not stamp its own start, so it cannot say
#               which side of it the file is on. Refuse rather than guess.
calibration_state() {
  local yaml="$1" since="${2:-}" stamp rc=0
  [[ -n "$since" ]] || { echo NO_BASELINE; return 0; }
  stamp="$(calibration_stamp "$yaml")" || rc=$?
  case "$rc" in
    0) ;;
    2) echo MISSING;  return 0 ;;
    *) echo UNDATED;  return 0 ;;
  esac
  if (( 10#$stamp >= 10#$since )); then echo PUBLISHED; else echo STALE; fi
}

# THE CALIBRATION GATE'S DECISION, from the two words that answer its two
# questions, in ONE place so that the session and its test read the same rule.
# Prints the half that failed and the word that failed it -- "ARM INVALID",
# "YAML STALE" -- or OK, and returns non-zero on anything but OK. A missing
# calibrate row prints ARM NO_ROW rather than ARM, because an empty word in a
# refusal reads as a bug in the refusal.
#
# WHY THE ARM HALF COMES FIRST. When --only leaves arm 0 out, the yaml half
# reports on a file this session never touched, and its answer (UNDATED, on the
# committed H200 ruler) names the wrong problem. The arm half names --only.
calibration_verdict() {
  local row="$1" state="$2"
  [[ "$row" == DONE ]]      || { echo "ARM ${row:-NO_ROW}"; return 1; }
  [[ "$state" == PUBLISHED ]] || { echo "YAML $state";      return 1; }
  echo OK
}

# THE REFUSAL ITSELF, for whichever half of the gate failed, as a function so
# that the WORDS can be planted and read in a test. A refusal that names the
# wrong cause costs the operator the same hour as no refusal at all: the first
# version of this gate had only the yaml half, so a session run with --only
# would have been told its committed ruler was UNDATED when what actually
# happened is that arm 0 was never scheduled.
calibration_refusal() {
  local verdict="$1" card="$2" yaml="$3" state="$4" word="${1#* }"
  case "${verdict%% *}" in
   ARM)
    echo "REFUSED: arm 0 did not stand behind a ruler for $card."
    echo "  calibrate is '$word' in $LEDGER; $yaml is $state."
    case "$word" in
      NO_ROW)     if [[ -n "$ONLY" ]]; then
                    echo "  No calibrate row at all: --only $ONLY left arm 0 out, and this"
                    echo "  gate is deliberately NOT scoped to --only. Name calibrate in it,"
                    echo "  first:"
                    echo "      bash scripts/h200_gaps_session.sh --only calibrate,$ONLY"
                  else
                    echo "  No calibrate row at all, and no --only to explain it: arm 0 did"
                    echo "  not run in this session directory. Read $LEDGER."
                  fi ;;
      INVALID)    echo "  It MEASURED and then failed a VALIDITY gate, and it published the"
                  echo "  yaml before scoring one. A clock that could not be established"
                  echo "  makes every number normalised by it unquotable, including the"
                  echo "  ridge every arm below would divide by." ;;
      CLAIM_FAIL) echo "  It measured and a CLAIM gate failed: no access pattern reached the"
                  echo "  pin rate, which is this instrument saying its own byte accounting"
                  echo "  is wrong. The yaml it published carries that accounting." ;;
      REFUSED)    echo "  It refused before measuring, so the yaml on disk is some earlier"
                  echo "  run's however fresh its stamp reads." ;;
      UNKNOWN)    echo "  It exited 1 from a file this driver could not confirm speaks"
                  echo "  moe/bench/exit_codes, so the driver will not read DONE, CLAIM_FAIL"
                  echo "  or a crash out of it. Read the log and decide by hand." ;;
      *)          echo "  That is not DONE, and DONE is the only state in which this"
                  echo "  instrument stands behind what it wrote." ;;
    esac
    ruler_stakes
    echo "  Read $LOGS/calibrate.log where there is one, fix what it names, and"
    echo "  re-run arm 0:"
    echo "      $PY_BASE $REPO/scripts/calibrate_hardware.py --publish"
    echo "  A CLAIM_FAIL or INVALID row is LATCHED and will not be re-attempted:"
    echo "  delete its row from $LEDGER to force one." ;;
   YAML)
    echo "REFUSED: this session has no calibration of its own for $card."
    echo "  calibrate is DONE in $LEDGER but $yaml is $state."
    case "$word" in
      MISSING)     echo "  Nothing is there. A DONE arm 0 that wrote no tracked file was run"
                   echo "  without --publish: its ruler is in the session path nothing reads." ;;
      UNDATED)     echo "  It is there and carries no provenance.utc, so it cannot say which"
                   echo "  rental measured it. 'Cannot say' is not 'this one'." ;;
      STALE)       echo "  It was measured before this session began, so it is a PREVIOUS"
                   echo "  rental's ridge sitting in this checkout." ;;
      NO_BASELINE) echo "  This driver could not stamp its own start, so it cannot say which"
                   echo "  side of it that file is on. Unset SESSION and take the default." ;;
      *)           echo "  That is not PUBLISHED, and PUBLISHED is the only state in which"
                   echo "  that file is this session's own." ;;
    esac
    ruler_stakes
    echo "  Run arm 0 and let it publish:"
    echo "      $PY_BASE $REPO/scripts/calibrate_hardware.py --publish"
    echo "  then re-run this session; finished arms in $LEDGER are skipped." ;;
   *)
    # A word neither half issues. There is no safe default here: the two states
    # this gate decides between are "this card's ruler" and "another machine's",
    # and guessing either is the defect it exists to prevent.
    echo "REFUSED: calibration_verdict said '$verdict', which this refusal does"
    echo "  not know how to read. Row and yaml state are in $LEDGER and $yaml." ;;
  esac
}

# WHAT A WRONG RULER COSTS, in the words both halves of the calibration gate
# need. Factored so the two refusals cannot drift into saying different things
# about the same consequence.
ruler_stakes() {
  echo "  Every arm below resolves its ridge through roofline.load_measured(),"
  echo "  labels it 'measured on this machine', and would score this card's"
  echo "  roof fractions, LEVEL flags and alphas against another machine's"
  echo "  ceilings. The H200's dense bf16 moved 7.1% between two sessions."
}

# DID THIS --dry-run PRINT A PLAN BEFORE IT REFUSED. rc 0 it did, rc 1 it did
# not, and there is no third answer because the question is about the log.
#
# WHY THE QUESTION EXISTS. Every script this driver runs now exits REFUSED from
# its own --dry-run -- a plan measured nothing and scored no gate, which is what
# the table calls 2 -- so `dry_state` mapping 2 to PLAN_REFUSED gave eleven of
# the sixteen planned arms the same word and DRY mode stopped being able to
# say "this plan is sound" at all. The two states are still different things and
# the log still tells them apart: bm128_depth prints 90 lines of plan and THEN
# says it measured nothing, while bm128_roofline at BLOCK_N=256 prints the
# refusal on line 1 and never plans. So the driver counts the non-blank lines
# printed before the first refusal marker. A script that printed a banner and
# then refused would read as a plan; the fix for that is in the banner, and
# nothing here is a threshold that can be tuned to hide it.
printed_a_plan() {
  local log="${1:-}"
  [[ -n "$log" && -f "$log" ]] || return 1
  awk '/REFUSED|NOT A RESULT/ {exit}
       /[^[:space:]]/ {n++}
       END {exit (n > 0 ? 0 : 1)}' "$log"
}

wanted() {
  [[ -z "$ONLY" ]] && return 0
  case ",$ONLY," in (*",$1,"*) return 0 ;; esac
  return 1
}

# WHAT THIS SESSION'S LEDGER HOLDS FOR ONE ARM, last row wins, empty when the
# arm has no row at all. It exists because a FILE landing on disk is not the
# same event as the ARM that wrote it standing behind what it wrote:
# calibrate_hardware.py copies the yaml into the tracked tree BEFORE it scores
# a single gate, and says so in its own --help ("A calibration whose clock
# could not be established is INVALID rather than DONE ... the yaml is still
# written"). So a calibrate that fails VALIDITY clock_established, or the pin
# rate gate, still leaves a fresh-stamped ruler behind it, and a gate that asks
# only WHEN the file was written reads that as this session's own. Asking the
# ledger asks the second question: did the instrument that wrote it pass its
# own gates. Nothing here decides an arm's state; `ledger_state` still does.
ledger_arm_state() {
  awk -F'\t' -v a="$1" '$1 == a { s = $2 } END { print s }' "$LEDGER" 2>/dev/null
}

# THE ONE LINE THE SUMMARY MAY GREP. Anchored at column zero on the prefix
# moe/bench/exit_codes.result_line renders and parse_result_lines reads back.
# A line that merely CONTAINS "PASS", "floor" or "sigma" is prose; the previous
# summary matched eighteen such lines in a REFUSED log and printed an imported
# prior and a pre-registered expectation as if they were this session's output.
result_lines() { grep -E '^RESULT: ' -- "$1" 2>/dev/null; }

# What one arm contributes to the closing summary. Returns 0 when it printed at
# least one RESULT line, 1 when the arm was not scored, so the caller can say
# what the absence means in the mode it is in.
summarize_arm() {
  local name="$1" log="$2" state="$3" hits
  if [[ "$state" == "NOT_PLANNED" ]]; then
    printf '  NOT PLANNED, and not a result. Nothing was run for this arm.\n'
    return 1
  fi
  if [[ ! -f "$log" ]]; then
    printf '  NO LOG at %s: this arm did not run in this session.\n' "$log"
    return 1
  fi
  if [[ "$state" == "REFUSED" || "$state" == "PLAN_REFUSED" ]]; then
    printf '  REFUSED BEFORE MEASURING. Nothing below is a gate:\n'
    grep -m2 'REFUSED' -- "$log" | sed 's/^/    /'
    [[ "$state" == "REFUSED" ]] && contract_caveat "$name" REFUSED
    return 1
  fi
  if [[ "$state" == "UNKNOWN" ]]; then
    printf '  STATE UNKNOWN. This arm exited 1 and the file it ran has not adopted\n'
    printf '  moe/bench/exit_codes, where 1 is CLAIM_FAIL and a CLAIM_FAIL is a\n'
    printf '  finding that is never re-run. It may equally be a refusal or an\n'
    printf '  exception Python exited 1 for. The row is NOT latched; the next\n'
    printf '  session will attempt this arm again unless you decide otherwise:\n'
    contract_caveat "$name" UNKNOWN
  fi
  if [[ "$state" == "INVALID" ]]; then
    printf '  MEASURED, THEN A VALIDITY GATE FAILED. Nothing from this arm may be\n'
    printf '  quoted, its cells must not be scored, and it is NOT auto-retried:\n'
    printf '  the instrument broke while in use. Re-run it only after the log\n'
    printf '  says in words that the cause was transient.\n'
    contract_caveat "$name" INVALID
  fi
  hits="$(result_lines "$log")"
  if [[ -z "$hits" ]]; then
    return 1
  fi
  printf '%s\n' "$hits"
  return 0
}

# One arm. Runs it, times it, asks git what it did to the tree, and writes one
# ledger row. Never returns non-zero: an arm's verdict belongs in the ledger,
# not in this function's exit status, or `set -o pipefail` and a future `set -e`
# would end the session on the first refusal.
arm() {
  local name="$1"; shift
  if ! wanted "$name"; then
    note "SKIP $name (--only $ONLY)"; return 0
  fi
  local prior
  prior="$(awk -F'\t' -v n="$name" \
    '$1 == n && ($2 == "DONE" || $2 == "CLAIM_FAIL" || $2 == "INVALID") { s = $2 } END { print s }' \
    "$LEDGER" 2>/dev/null)"
  if (( DRY == 0 )) && [[ -n "$prior" ]]; then
    note "SKIP $name (already $prior in $LEDGER; delete its row to force a re-run)"
    return 0
  fi
  local log="$LOGS/$name.log" t0 t1 rc state before after
  note "-> $name   log $log"
  before="$(dirty_count)"
  t0=$(date -u +%s)
  "$@" > "$log" 2>&1
  rc=$?
  t1=$(date -u +%s)
  after="$(dirty_count)"
  if (( DRY )); then state="$(dry_state "$rc" "$log")"; else state="$(ledger_state "$rc")"; fi
  # AN EXIT 1 FROM A FILE THAT DOES NOT SPEAK THE TABLE IS NOT A RESULT, and
  # CLAIM_FAIL is the one word this ledger LATCHES as one: the resume check
  # above skips it forever. Python spends 1 on any exception that escapes main,
  # and three of the thirteen Python files this driver runs install the ERROR(4)
  # handler that would say so, so a crash in the other ten would be filed as
  # "the world disagreed with the claim, do not re-run" -- the most expensive
  # single mislabel available here, because the arm is never attempted again and
  # the summary reports its silence as a finding. UNKNOWN is not latched, is not
  # RETRY either (the session does not exit 4 over it), and is disclosed by name
  # below. `ledger_state` is untouched: this is the driver declining to read a
  # word out of a table the file never agreed to, not a second table.
  if [[ "$state" == "CLAIM_FAIL" ]] && ! adopts_exit_codes "$(arm_script "$name")"; then
    state=UNKNOWN
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$state" "$rc" "$((t1 - t0))" "$after" "$log" "" >> "$LEDGER"
  note "   $state (exit $rc) in $((t1 - t0))s; work tree $after dirty file(s)"
  case "$state" in
    BROKEN)  BROKEN_ARMS=$((BROKEN_ARMS + 1))
             note "   BROKEN: this is a PLAN, and it did not survive its own --dry-run."
             note "   $(tail -1 "$log")" ;;
    RETRY)   RETRY_ARMS=$((RETRY_ARMS + 1))
             note "   exit $rc is not in the table. Read the log before re-running." ;;
    UNKNOWN) note "   exit 1 from a file that has not adopted moe/bench/exit_codes."
             note "   In that table 1 is CLAIM_FAIL, a RESULT that is never re-run."
             note "   From this file it may equally be a crash Python exited 1 for,"
             note "   or a VALIDITY gate. NOT latched: read the log and decide."
             contract_caveat "$name" UNKNOWN ;;
    REFUSED|PLAN_REFUSED)
             note "   $(grep -m1 'REFUSED' -- "$log" || tail -1 "$log")"
             [[ "$state" == "REFUSED" ]] && contract_caveat "$name" REFUSED ;;
    INVALID) note "   Do NOT re-run and do NOT quote it: a VALIDITY gate failed after measuring."
             contract_caveat "$name" INVALID ;;
  esac
  if [[ "$before" != "-" && "$after" != "-" && "$after" -gt "$before" ]]; then
    note "   WARNING: this arm dirtied the work tree ($before -> $after files)."
    note "   Every row it published carries git_dirty=True. Find what it wrote."
  fi
  return 0
}

# An arm that is deliberately not run, named with the reason. Absence in a
# ledger reads as "nothing to report", which is the one thing it never means.
skip_arm() {
  local name="$1" reason="$2"
  wanted "$name" || return 0
  note "NOT PLANNED $name: $reason"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "NOT_PLANNED" "-" "0" "$(dirty_count)" "-" "$reason" >> "$LEDGER"
}

# THE DETECTION LIMIT OF THE WHOLE SESSION, printed before anything is spent.
# Every difference this study has published is one run per condition, so the
# only honest MDE is the known-variance form with sigma imported from the
# replicate floor: scripts/replicate_noise_floor.mde_external_sigma, which is
# 3.96*sigma at n=1. The floor is NOT MEASURED yet -- part (a) has never run on
# a card -- so the sigma is the declared prior in results/published/
# NOISE_FLOOR.json and the line says so in the word ASSUMED. Arm 3 is what
# replaces the assumption with a number.
mde_line() {
  local out rc=0
  out="$("$PY_BASE" - "$REPO" <<'PY' 2>&1
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "scripts"))
import replicate_noise_floor as RNF                                # noqa: E402

path = repo / "results" / "published" / "NOISE_FLOOR.json"
try:
    floor = RNF.noise_floor(path)
    sigma, basis, source = floor.sd, "MEASURED", floor.provenance
except Exception as exc:                                          # noqa: BLE001
    payload = json.loads(path.read_text())
    sigma = float(payload["prior_sd"])
    basis = "ASSUMED"
    source = f'{payload["prior_sd_source"]} [{exc.__class__.__name__}]'
mde = RNF.mde_external_sigma(sigma, 1)
print(f"MDE {mde:.4f} in fitted alpha at the design this study runs "
      f"(one run per condition, known-variance z test, "
      f"level {RNF.TEST_LEVEL}, power {RNF.TEST_POWER}: 3.96*sigma)")
print(f"    sigma {sigma:.4f}, {basis}: {source}")
PY
)" || rc=$?
  if (( rc != 0 )); then
    printf 'MDE UNAVAILABLE: the noise floor could not be read, so this session\n'
    printf '    states no detection limit and every effect below is unscored.\n'
    printf '%s\n' "$out" | tail -3 | sed 's/^/    /'
    return 1
  fi
  printf '%s\n' "$out"
  return 0
}

# Is the attached card pre-Hopper. SM_MAJOR is set once, from a capability that
# matched a number; an unparsed capability leaves it EMPTY and this returns
# false, so an arm is never skipped on a string nobody could read. Written as a
# function because `[[ "${CAPABILITY%%.*}" -lt 9 ]]` evaluates its operands as
# ARITHMETIC: on 2026-09-02 a capability field that had been filled with the
# words "no CUDA" by a delimiter bug made bash look for a variable named CUDA
# and, under `set -u`, end the session there.
pre_hopper() {
  [[ -n "$SM_MAJOR" ]] || return 1
  (( SM_MAJOR < 9 ))
}

# Does git keep this path. ASKED, never asserted from a reading of .gitignore:
# the answer differs for results/, results/published/ and a path outside the
# work tree. rc 128 is a path outside the tree -- the pod default /workspace --
# and calling that "tracked" is how output gets written where nobody collects it.
git_note() {
  local p="$1" rc=0
  git -C "$REPO" check-ignore -q -- "$p" >/dev/null 2>&1 || rc=$?
  case "$rc" in
    0) echo "IGNORED by git -- nothing written here enters the repo (the intended deal for raw output; publish with scripts/publish_results.sh)" ;;
    1) echo "git WILL KEEP this path -- anything written here is committable" ;;
    *) echo "UNVERIFIED: git check-ignore exited $rc. Usually the path is outside this work tree, the normal case for /workspace on a pod. It is NOT 'tracked'." ;;
  esac
}
# <<< LIFTABLE

BROKEN_ARMS=0
RETRY_ARMS=0

# --------------------------------------------------------------------------
# The arms, declared once, in the order their results are read. The cost table,
# --list and the closing summary all read this ONE list. The configuration is
# IN THE NAME wherever two arms differ only by configuration, because a ledger
# row that says "roofline" and a report that says BLOCK_N=64 are the same
# defect as a run id without its card.
# --------------------------------------------------------------------------
ARM_NAMES=(calibrate pin_probe-n64-g1 pin_probe-n256-g16
           roofline-n64-g1 roofline-n256-g16 roofline-n256-g32
           bm128_depth noise_floor
           bn_g16 bn_g1 anchor_measure anchor_rescore occupancy
           mma_switch ruler cap_test dtype span_dense span counter_plan)

arm_minutes()  { case "$1" in
  calibrate) echo 3 ;;
  pin_probe-n64-g1) echo 2 ;;   pin_probe-n256-g16) echo 2 ;;
  roofline-n64-g1) echo 4 ;;    roofline-n256-g16) echo 4 ;;
  roofline-n256-g32) echo 4 ;;
  bm128_depth) echo 12 ;;       noise_floor) echo 25 ;;
  bn_g16) echo 11 ;;            bn_g1) echo 11 ;;        anchor_measure) echo 8 ;;
  anchor_rescore) echo 0 ;;     occupancy) echo 20 ;;
  mma_switch) echo 7 ;;         ruler) echo 4 ;;         cap_test) echo 9 ;;
  dtype) echo 25 ;;             span_dense) echo 20 ;;   span) echo 30 ;;
  counter_plan) echo 1 ;;
esac; }

arm_closes() { case "$1" in
  calibrate)  echo "This pod's own ridge and both dtype peaks. Five arms below REFUSE without it, and the H200's dense bf16 moved 7.1% between two sessions, so it is not a constant anything can carry over. It also WRITES a tracked yaml, which is one of the two reasons the dirty-file count is re-asked after every arm." ;;
  pin_probe-n64-g1) echo "The S6a gate ('observed tile_block_m = none') at BLOCK_N=64, GROUP_SIZE_M=1 -- the configuration the control roofline, both bn arms, the anchor and the cap test all pin. Every one of them is worthless if the pin is not honoured." ;;
  pin_probe-n256-g16) echo "The same at BLOCK_N=256, GROUP_SIZE_M=16, the shape vLLM 0.27.1 ships for mixtral at BLOCK_M=128. A pin that reaches the kernel at BLOCK_N=64 is evidence about BLOCK_N=64." ;;
  roofline-n64-g1) echo "THE CONTROL. BLOCK_M=128 at the SWEPT configuration, which production does not ship. It can REFUTE the ceiling (if 128 reaches the roof here, it reaches it everywhere richer) and it CANNOT confirm one for production. Its likely outcome is already predictable from the published G=1 ladders." ;;
  roofline-n256-g16) echo "THE CLAIM, and the only arm that can confirm it. BLOCK_M=128 at vLLM's own tuned entry for this shape (BLOCK_N=256, GROUP_SIZE_M=16, num_stages 4), which no arm in this study has ever measured. Contests TEMPO's 'the tile term is inactive in decode' in the configuration TEMPO's readers run. No fit, no alpha, no anchor. TODAY IT REFUSES: no BLOCK_M=256 control fits at BLOCK_N=256 (256 registers per thread against 255 at 8 warps; 256 KiB of shared memory against 227 at 16), and this driver will not run the subject without the control that cancels the fused layer." ;;
  roofline-n256-g32) echo "The same at GROUP_SIZE_M=32, vLLM's entry at 2048 tokens. Without it the production claim rests on a single swizzle, and the swizzle is the lever this study has already shown moves alpha by 0.39. Refuses for the same missing control as the G=16 arm, and one fix unblocks both." ;;
  bm128_depth) echo "The evaluation's #2: five clean memory-bound treads at 128, monotone. The whole 128 row is currently n=2 across two cards, one on a ladder where time falls as rows rise." ;;
  noise_floor) echo "The evaluation's #3: a real between-replicate sd. The study has none; every effect so far is scored against an IMPORTED prior, including the MDE this session prints. Also publishes the num_stages control that would have caught the cross-card null." ;;
  bn_g16)     echo "alpha_a as a fitted slope rather than a two-point guess, and the residual that says whether the three-term model is COMPLETE. The only clean lever on the decomposition." ;;
  bn_g1)      echo "alpha_b and C3 at the production swizzle of every published arm, for comparability. NOT a second reading of C2: at GROUP_SIZE_M=1 the planted MISSING world passes C2 by construction, so C2 is UNKNOWN there however the data fall." ;;
  anchor_measure) echo "The evaluation's weakest link: the memory-branch level, measured at matched reuse rather than extrapolated. Decides whether any numeric alpha is publishable." ;;
  anchor_rescore) echo "Free: every committed report re-scored under the anchor arm 0 just calibrated, so the size of the correction to every published alpha is known. Written under the session directory, never into results/published." ;;
  occupancy)  echo "Whether alpha tracks residency or program order. If residency, the swizzle is a dead lever, the cross-card null is explained, and reuse-distance prediction does not transfer to this regime." ;;
  mma_switch) echo "STUDY item 3's loose end. CLOSES whether the tile alone selects the instruction at fixed tokens." ;;
  ruler)      echo "STUDY item 2's follow-up. Prices the read-vs-triad and clocks-first changes on the committed corpus without adopting them." ;;
  cap_test)   echo "FINDINGS' fourth readout, DEMOTED: BLOCK_M=16 runs multi-tile in 1 of 24 cells on uniform routing, so this tests the formula, not the claim." ;;
  dtype)      echo "STUDY C2's confound: how much of the 1.15 is the config vLLM resolved differently per dtype." ;;
  span_dense) echo "The 0.563 EXTENT-versus-KERNEL split on the DENSE grid, the only grid where C3's mechanism is observable. Runs before the sparse arm because the sparse grid's own kernel world predicts C2 FAIL, and a CLAIM gate failing is a result, not a retry." ;;
  span)       echo "The same on the PUBLISHED grid, booked --no-densify, which is what puts it in a different run id from span_dense: with --densify the default, a bare arm derived the dense arm's id, restored its rows, measured nothing and still landed DONE. EXPECT IT TO REFUSE: on a powers-of-two grid the padding factor is exactly 1.00 everywhere, so c2_grid_power stops it before it spends a minute. The refusal is the extent comparison's honest answer on that grid, and it is free." ;;
  counter_plan) echo "Whether a DRAM counter is reachable here. A counter is the only route to alpha_b as a number rather than an interval; on rented pods it is blocked and this records which way. READ ITS VERDICT LINE, NOT ITS LEDGER STATE: scripts/dram_counter_route.py returns 0 only for OPEN and 3 for everything else, and 3 is INVALID in the table this session adopted, so the BLOCKED answer this arm exists to obtain is filed as a validity failure. BLOCKED is the ANSWER, not a broken instrument; the fix belongs in that script, which audit A4 does not schedule." ;;
esac; }

arm_offgpu_gates() { case "$1" in
  roofline-n64-g1|roofline-n256-g16|roofline-n256-g32)
              echo "scripts/bm128_roofline.py --self-test --fail-on-gate  (three planted worlds, exit 0 required)" ;;
  bm128_depth) echo "scripts/bm128_depth.py --self-test  (three worlds from the law)" ;;
  noise_floor) echo "its plan prints the power table; the floor itself needs replicates" ;;
  bn_g16|bn_g1) echo "scripts/bn_decomposition.py --self-test --capability 9.0 --group-m 16 --reps 17 --plant-noise 0.008  (four worlds, four distinct verdicts)" ;;
  anchor_measure|anchor_rescore) echo "scripts/memory_branch_anchor.py --rescore --out-dir <a path outside the tree>  (free, scores every committed report)" ;;
  occupancy)  echo "scripts/occupancy_vs_swizzle.py --self-test and --audit  (A1 FAILs on the corpus by design)" ;;
  cap_test)   echo "scripts/tile_cap_test.py --self-test 0.558 and --self-test 0.10" ;;
  span|span_dense) echo "scripts/span_extent_separation.py --self-test kernel|extent|neither --densify" ;;
  dtype)      echo "C1 and C2 by --dry-run; C3 by --self-test 2.033|2.400|1.000 --self-test-alpha 0.2" ;;
  ruler)      echo "scripts/ruler_rebaseline.py --corpus-only  (hermetic)" ;;
  mma_switch) echo "its four gates need two real compiles; --dry-run registers thresholds only" ;;
  pin_probe-n64-g1|pin_probe-n256-g16) echo "F1 and F2 need a vLLM span, which registers only on the GPU box" ;;
  counter_plan) echo "scripts/dram_counter_route.py --dry-run and --bracket  (the plan and the counter-free bound)" ;;
  calibrate)  echo "none: it is a measurement and nothing else" ;;
esac; }

if (( LIST )); then
  say "ARMS, in the order their results are read"
  echo "  Exit codes are moe/bench/exit_codes.py's table for every arm alike:"
  echo "    0 DONE   1 CLAIM_FAIL (a result, never re-run)   2 REFUSED (free)"
  echo "    3 INVALID (measured, unquotable, not auto-retried)   4+ RETRY"
  for n in "${ARM_NAMES[@]}"; do
    printf '\n  %-19s ~%s min\n' "$n" "$(arm_minutes "$n")"
    printf '    %s\n' "$(arm_closes "$n")"
  done
  exit 0
fi

# --------------------------------------------------------------------------
# Preflight. Costs nothing; every refusal here saves a pod hour.
# --------------------------------------------------------------------------
say "PREFLIGHT"
[[ -d "$REPO/scripts" ]] || { echo "REFUSED: no scripts/ under REPO=$REPO"; exit "$RC_REFUSED"; }
[[ -x "$PY_BASE" ]] || { echo "REFUSED: no usable base interpreter (set PY_BASE=)"; exit "$RC_REFUSED"; }
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
if (( missing )); then echo "REFUSED: a script this driver schedules is missing or broken."; exit "$RC_REFUSED"; fi
note "scripts   all 13 present and parse"

# THE CARD, through moe/bench/provenance so that this file and every report
# agree on the slug character for character. A card this cannot name is the
# literal string "nocard" WITH the reason printed: an unnamed card in a path is
# how one pod's directories were resumed by another pod's session.
CARD="nocard"; CAPABILITY=""; CARD_REASON="not probed"; CARD_OK=0
IFS='|' read -r CARD CAPABILITY CARD_REASON < <("$PY_BASE" - "$REPO" <<'PY' 2>/dev/null
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
try:
    from moe.bench.provenance import card_slug, provenance_block
except Exception as exc:                                          # noqa: BLE001
    print(f"nocard||moe.bench.provenance is not importable from {repo}: "
          f"{exc.__class__.__name__}")
    raise SystemExit(0)

prov = provenance_block(repo_root=repo)
if not prov.gpu_name:
    print(f"nocard||{prov.missing.get('gpu_name', 'no reason recorded')}")
    raise SystemExit(0)

capability = ""
try:
    import torch
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    capability = f"{props.major}.{props.minor}"
except Exception as exc:                                          # noqa: BLE001
    capability = ""
    print(f"{card_slug(prov.gpu_name)}||compute capability unreadable: "
          f"{exc.__class__.__name__}")
    raise SystemExit(0)
print(f"{card_slug(prov.gpu_name)}|{capability}|")
PY
)
CARD="${CARD:-nocard}"; CAPABILITY="${CAPABILITY:-}"; CARD_REASON="${CARD_REASON:-}"
[[ "$CARD" != "nocard" ]] && CARD_OK=1
SM_MAJOR=""
[[ "$CAPABILITY" =~ ^([0-9]+)\. ]] && SM_MAJOR="${BASH_REMATCH[1]}"
if (( CARD_OK )); then
  note "card      $CARD${CAPABILITY:+  compute capability $CAPABILITY}"
  [[ -z "$CAPABILITY" ]] && note "          $CARD_REASON"
else
  note "card      nocard -- $CARD_REASON"
  note "          Every path below carries the literal 'nocard', so nothing"
  note "          measured on a real card can resume into this session."
fi

if (( DRY == 0 && CARD_OK == 0 )); then
  echo
  echo "REFUSED: no CUDA device, and this driver measures. Nothing was run."
  echo "  For the review step, off GPU and free:  bash scripts/h200_gaps_session.sh --dry-run"
  exit "$RC_REFUSED"
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

# The session and results root, with the card IN BOTH NAMES. /workspace/results
# outlives the pod: without the card a second card resumes the first's
# directories and reports the first's timings under the second's heading, which
# is a collision this repo has already published (two arms, two sm_counts, one
# file name). An operator-supplied MOE_RESULTS_DIR is REFUSED rather than
# silently rewritten if it does not name the card.
if [[ -d /workspace ]]; then
  SESSION="${SESSION:-/workspace/session/gaps-$CARD-$(date -u +%Y%m%dT%H%M%SZ)}"
  RESULTS_ROOT=/workspace/results
else
  SESSION="${SESSION:-$REPO/results/h200_gaps/session-$CARD-$(date -u +%Y%m%dT%H%M%SZ)}"
  RESULTS_ROOT="$REPO/results"
fi
if [[ -n "${MOE_RESULTS_DIR:-}" ]]; then
  RESULTS="$MOE_RESULTS_DIR"
  case "$RESULTS" in
    *"$CARD"*) ;;
    *) echo "REFUSED: MOE_RESULTS_DIR=$RESULTS does not contain the card '$CARD'."
       echo "  Two pods share a network volume and one card's run resumes the"
       echo "  other's directories when the card is not in the path. Use"
       echo "  MOE_RESULTS_DIR=$RESULTS/gaps-$CARD or unset it and take the default."
       exit "$RC_REFUSED" ;;
  esac
else
  RESULTS="$RESULTS_ROOT/gaps-$CARD"
fi
export MOE_RESULTS_DIR="$RESULTS"
LOGS="$SESSION/logs"
if (( DRY )); then LEDGER="$SESSION/ARMS-dryrun.tsv"; else LEDGER="$SESSION/ARMS.tsv"; fi
mkdir -p "$LOGS" || { echo "REFUSED: cannot create $LOGS"; exit "$RC_REFUSED"; }
[[ -f "$LEDGER" ]] || printf 'arm\tstate\trc\tseconds\tdirty\tlog\tnote\n' > "$LEDGER"

CALIB_YAML="$REPO/moe/bench/hardware/measured_$CARD.yaml"
# THE INSTANT THIS SESSION BEGAN, as the integer `calibration_state` compares a
# calibration's own timestamp against. Taken from the session directory's name,
# which already carries a UTC stamp, so a session RESUMED into that directory
# still counts the calibration its first pass published instead of refusing an
# hour of finished arms. An operator-supplied SESSION that carries no stamp
# falls back to now, and a resume under such a name refuses until calibrate runs
# again: that is the direction that costs three minutes rather than the session.
SESSION_SINCE="$(utc_stamp "${SESSION##*-}")" || SESSION_SINCE="$(date -u +%Y%m%d%H%M%S)"
DIRTY_AT_START="$(dirty_count)"
note "session   $SESSION"
note "          $(git_note "$SESSION")"
note "results   $RESULTS   (exported as MOE_RESULTS_DIR to every arm; the card is in the name)"
note "          $(git_note "$RESULTS/bm128_roofline")"
note "ledger    $LEDGER"
note "tree      $DIRTY_AT_START dirty file(s) before anything ran, by git status"
note "          --porcelain --untracked-files=all. Re-asked after every arm: two"
note "          arms used to write into the tracked tree while the session ran."
if (( CARD_OK )); then
  note "calib     $CALIB_YAML"
  note "          $(git_note "$CALIB_YAML")   <- THIS ONE is meant to be committed"
fi

say "WHAT THIS COMMITS YOU TO"
total=0
for n in "${ARM_NAMES[@]}"; do
  wanted "$n" || continue
  m="$(arm_minutes "$n")"; total=$((total + m))
  printf '  %-19s ~%3s min   %s\n' "$n" "$m" "$(arm_closes "$n" | cut -c1-92)"
done
note ""
mde_line | sed 's/^/  /'
note ""
note "  Read every effect this session reports against that limit. The three"
note "  effects already registered in NOISE_FLOOR.json are the swizzle swing"
note "  0.3855 and the footprint spread 0.411, both far above it, and the"
note "  cross-card 0.0117, which is BELOW it and therefore unresolved."
if (( DRY )); then
  note ""
  note "--dry-run: each arm runs its OWN --dry-run. Free, off GPU, seconds."
  note "The minutes above are what the POD run would cost, not this one."
else
  note ""
  note "TOTAL ~$total minutes. The four arms after the pin probes -- three"
  note "rooflines and bm128_depth -- are ~24 min and contain the claim. Arms"
  note "already finished in $LEDGER are skipped, so a re-run costs only what is"
  note "left; a REFUSED arm is re-attempted, because refusing costs nothing."
fi

started=$(date -u +%s)

say "SESSION  card=$CARD  $( ((DRY)) && echo '(DRY RUN: plans only)' || echo '(MEASURING)')"

# --------------------------------------------------------------------------
# 0. THIS POD'S OWN CEILINGS, and whether the pin reaches the kernel.
# --------------------------------------------------------------------------
say "0. calibrate THIS card"
if (( DRY )); then
  skip_arm calibrate "calibrate_hardware.py is a measurement and has no --dry-run."
else
  # --publish IS THE ARM. Without it this measures this card's ridge into an
  # untracked session path that nothing reads, and every arm below resolves its
  # ridge from whatever measured_*.yaml the LAST rental left in the checkout --
  # 162.81 quoted from another pod while `ridge_source` says "measured on this
  # machine". calibrate_hardware.py's own header says so: "the copy is what
  # makes the ruler visible to roofline.load_measured() and therefore to the
  # sweep". The flag dirties a TRACKED file, which is the cost the A6 fix was
  # avoiding, and that cost is disclosed below rather than paid silently.
  arm calibrate "$PY_BASE" "$REPO/scripts/calibrate_hardware.py" --publish
  # AND THE GATE, in TWO questions, because passing the flag is not the same as
  # the file landing and the file landing is not the same as the file being a
  # ruler.
  #
  # WHEN was it written. `calibration_state` answers MISSING (calibrate refused
  # before it wrote anything), UNDATED (the committed H200 yaml, which carries
  # no provenance block at all) or STALE (the previous rental's). Each of the
  # three means the same thing to every arm below: the ruler is not this card's,
  # so nothing measured against it is this card's either.
  #
  # DID THE ARM THAT WROTE IT PASS ITS OWN GATES. `calibration_state` cannot
  # answer that and a fresh stamp is not the answer: calibrate_hardware.py
  # copies the yaml into the tracked tree BEFORE it scores anything, so a run
  # that lands INVALID on VALIDITY clock_established -- "the samples disagree
  # with the settle plateau; nothing normalised by the clock may be quoted" --
  # or CLAIM_FAIL on the pin rate has still published a fresh-stamped ruler.
  # Asking WHEN alone, this gate printed "measured in THIS session" over a file
  # arm 0 itself refused to stand behind, and 3.4 hours of arms scored every
  # roof fraction, LEVEL flag and alpha against it while `ridge_source` named
  # the attached device. So the ledger row is the second question, and both
  # must answer: DONE, and PUBLISHED.
  #
  # THE GATE IS NOT SCOPED TO --only. `arm calibrate` above returns early when
  # --only names other arms, and this does not, because every arm resolves its
  # ridge through roofline.load_measured() no matter which subset was asked for.
  # A resume into the SAME session directory still passes without re-measuring:
  # the ledger is that directory's, so the first pass's DONE row is still there.
  # The session stops here, where three minutes were spent, instead of at the
  # end, where 3.4 hours were.
  CALIB_STATE="$(calibration_state "$CALIB_YAML" "$SESSION_SINCE")"
  CALIB_ROW="$(ledger_arm_state calibrate)"
  CALIB_VERDICT="$(calibration_verdict "$CALIB_ROW" "$CALIB_STATE")"
  if [[ "$CALIB_VERDICT" == "OK" ]]; then
    note "   ruler     $CALIB_YAML, measured in THIS session by an arm that"
    note "             passed its own gates (calibrate DONE in $LEDGER)."
    note "   It is TRACKED and now dirty, so every row measured below carries"
    note "   git_dirty=True until it is committed. Commit it from another shell"
    note "   before quoting any number that names a commit."
  else
    echo
    calibration_refusal "$CALIB_VERDICT" "$CARD" "$CALIB_YAML" "$CALIB_STATE"
    exit "$RC_REFUSED"
  fi
fi

say "0. does MOE_FORCE_TILE reach the kernel AT THE CONFIGURATIONS THE ARMS RUN"
# BLOCK_N=64/GROUP_SIZE_M=1 is block_m_crossing_sweep.FIXED, what the control
# roofline, both bn arms, the anchor and the cap test pin. BLOCK_N=256/
# GROUP_SIZE_M=16 num_stages=4 is vLLM 0.27.1's tuned entry for mixtral at
# BLOCK_M=128 (moe/bench/hardware/vllm_configs/E=8,N=14336,device_name=
# NVIDIA_H200.json, 512 through 4096 tokens), what the production arms pin. The
# probe used to pin BLOCK_N=128, which is neither.
PIN_SWEPT='{"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":64,"GROUP_SIZE_M":1,"num_warps":8,"num_stages":4}'
PIN_PROD='{"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":64,"GROUP_SIZE_M":16,"num_warps":8,"num_stages":4}'
# --env vllm --impl vllm_fused_experts is what makes candidate_impls plan a vLLM
# span at all; without it only torch's CUTLASS spans are planned and the gate
# reads "observed = none" for the wrong reason. Set on this command, never exported.
if (( DRY == 0 )); then
  arm pin_probe-n64-g1 env MOE_FORCE_TILE="$PIN_SWEPT" "$PY_VLLM" -m moe.bench.cli \
      --profile profile-cell --groups baselines --env vllm --impl vllm_fused_experts \
      --out-dir "$SESSION/pin-probe-n64-g1"
elif (( CARD_OK )); then
  arm pin_probe-n64-g1 env MOE_FORCE_TILE="$PIN_SWEPT" "$PY_VLLM" -m moe.bench.cli \
      --profile profile-cell --groups baselines --env vllm --impl vllm_fused_experts \
      --dry-run --out-dir "$SESSION/pin-probe-n64-g1"
else
  skip_arm pin_probe-n64-g1 \
    "no CUDA device: no framework span registers off the GPU box, so the plan would refuse for a reason that is about this laptop and not about the pin."
fi

if pre_hopper; then
  skip_arm pin_probe-n256-g16 \
    "compute capability $CAPABILITY: BLOCK_M=128 x BLOCK_N=256 at 4 stages asks 192 KiB of shared memory against this card's 164. The production configuration is the H200's tuned entry and is not runnable here."
elif (( DRY == 0 )); then
  arm pin_probe-n256-g16 env MOE_FORCE_TILE="$PIN_PROD" "$PY_VLLM" -m moe.bench.cli \
      --profile profile-cell --groups baselines --env vllm --impl vllm_fused_experts \
      --out-dir "$SESSION/pin-probe-n256-g16"
elif (( CARD_OK )); then
  arm pin_probe-n256-g16 env MOE_FORCE_TILE="$PIN_PROD" "$PY_VLLM" -m moe.bench.cli \
      --profile profile-cell --groups baselines --env vllm --impl vllm_fused_experts \
      --dry-run --out-dir "$SESSION/pin-probe-n256-g16"
else
  skip_arm pin_probe-n256-g16 \
    "no CUDA device: no framework span registers off the GPU box, so the plan would refuse for a reason that is about this laptop and not about the pin."
fi

# --------------------------------------------------------------------------
# 1. THE CLAIM, in three configurations. All three run WITHOUT --fail-on-gate:
#    a C1 FAIL means BLOCK_M=128 reached the roof, which is one of the two
#    registered outcomes and a result.
# --------------------------------------------------------------------------
say "1a. CONTROL: BLOCK_M=128 at the SWEPT configuration (BN=64, G=1)"
note "This arm can refute a ceiling and cannot confirm one for production."
if (( DRY )); then
  arm roofline-n64-g1 "$PY_BASE" "$REPO/scripts/bm128_roofline.py" --dry-run \
      --block-n 64 --group-m 1 --control 256
else
  arm roofline-n64-g1 "$PY_VLLM" "$REPO/scripts/bm128_roofline.py" \
      --model mixtral-8x7b --dtype bf16 --control 256 \
      --block-n 64 --group-m 1 \
      --r-min 32 --r-max 4096 --reps 3 --plateau-doublings 2
fi

say "1b. THE CLAIM: BLOCK_M=128 at vLLM's own tuned entry (BN=256, G=16)"
# THE CONTROL DOES NOT FIT AT THIS BLOCK_N, AND THAT IS THE ARM'S OWN FINDING.
# bm128_roofline needs a positive control -- the same sweep at a tile with no
# ceiling of its own, which cancels the fused layer's fixed cost, the clocks and
# the occupancy -- and it pins ONE BLOCK_N for the subject and the control alike.
# At BLOCK_N=256 no BLOCK_M=256 control can run on this hardware, by the script's
# own resource model: at num_warps=8 the 256x256 fp32 accumulator asks 256
# registers per thread against 255, and at num_warps=16 it fits the registers and
# then asks 256 KiB of shared memory at num_stages=4 against sm_90's 227. Both
# escapes (num_stages 3, or num_warps 16) change the SUBJECT away from the
# configuration vLLM ships, which is the one thing this arm exists to measure.
# So the arm is scheduled with the control it needs and REFUSES, free, before any
# pod time, printing the bill. That refusal is the finding: the production
# configuration has no BLOCK_M=256 control at its own BLOCK_N, and until
# bm128_roofline can pin the control at a BLOCK_N of its own, the production
# claim has a subject and nothing to cancel against. Running the subject alone
# would produce another uninterpretable "0.5 of the roof", which is the number
# this whole session exists to stop quoting.
if pre_hopper; then
  skip_arm roofline-n256-g16 \
    "compute capability $CAPABILITY: BLOCK_N=256 at 4 stages needs 192 KiB of shared memory against 164. The claim is about the H200 entry and this card cannot run it."
elif (( DRY )); then
  arm roofline-n256-g16 "$PY_BASE" "$REPO/scripts/bm128_roofline.py" --dry-run \
      --block-n 256 --group-m 16 --control 256
else
  arm roofline-n256-g16 "$PY_VLLM" "$REPO/scripts/bm128_roofline.py" \
      --model mixtral-8x7b --dtype bf16 --control 256 \
      --block-n 256 --group-m 16 \
      --r-min 32 --r-max 4096 --reps 3 --plateau-doublings 2
fi

say "1c. THE CLAIM at the swizzle vLLM ships at 2048 tokens (BN=256, G=32)"
if pre_hopper; then
  skip_arm roofline-n256-g32 \
    "compute capability $CAPABILITY: same 192 KiB shared-memory refusal as the G=16 arm."
elif (( DRY )); then
  arm roofline-n256-g32 "$PY_BASE" "$REPO/scripts/bm128_roofline.py" --dry-run \
      --block-n 256 --group-m 32 --control 256
else
  arm roofline-n256-g32 "$PY_VLLM" "$REPO/scripts/bm128_roofline.py" \
      --model mixtral-8x7b --dtype bf16 --control 256 \
      --block-n 256 --group-m 32 \
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
  STAGES=(); [[ "$SM_MAJOR" == "8" ]] && STAGES=(--num-stages 3)
  arm bn_g16 "$PY_VLLM" "$REPO/scripts/bn_decomposition.py" --group-m 16 --reps 17 "${STAGES[@]}"
  arm bn_g1  "$PY_VLLM" "$REPO/scripts/bn_decomposition.py" --group-m 1  --reps 17 "${STAGES[@]}"
fi

say "5. the memory-branch anchor, measured rather than extrapolated"
if (( DRY )); then
  arm anchor_measure "$PY_BASE" "$REPO/scripts/memory_branch_anchor.py" --dry-run
else
  arm anchor_measure "$PY_VLLM" "$REPO/scripts/memory_branch_anchor.py" --measure --model mixtral-8x7b
fi
# Free, and it SCORES: --rescore reads the committed corpus and returns a gate
# verdict, so it is a measurement of the corpus and not a plan. It is therefore
# not run in a --dry-run session, where a CLAIM_FAIL would be recorded as a
# broken plan. --out-dir keeps it out of results/published, whose two tracked
# files it rewrote on every previous invocation, --dry-run included.
if (( DRY )); then
  skip_arm anchor_rescore \
    "--rescore scores the committed corpus and returns a gate verdict, which is a result and not a plan. Run it directly: scripts/memory_branch_anchor.py --rescore --out-dir <path outside the tree>."
else
  arm anchor_rescore "$PY_BASE" "$REPO/scripts/memory_branch_anchor.py" --rescore \
      --out-dir "$SESSION/anchor-rescore"
fi

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
if pre_hopper; then
  skip_arm mma_switch "compute capability $CAPABILITY reaches no warpgroup MMA at any tile."
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
note "Dense grid first: it is the only grid on which C3's mechanism is observable."
# THE SPARSE ARM IS BOOKED --no-densify, AND THAT IS THE WHOLE FIX. --densify
# became the default on 2026-09-02 (argparse BooleanOptionalAction), so the bare
# arm densified too: both arms derived the SAME run id, the second restored
# every row the first had written, timed nothing, and landed DONE in the ledger
# with 30 minutes booked against it. The id is a hash of the plan and `densify`
# is one of its knobs, so naming the flag is what separates them -- verified off
# GPU, `--densify` derives ...-densifytrue-...-e95805af and `--no-densify`
# ...-densifyfalse-...-5f329a66. --max-minutes could not have separated them:
# it prices a run rather than defining one and is deliberately not in the key.
# The sparse arm now REFUSES before measuring, by that script's own c2_grid_power
# gate ("grid too sparse for C2"), which is free and is the honest answer for
# the grid V5's argument prefers.
if (( DRY )); then
  arm span_dense "$PY_BASE" "$REPO/scripts/span_extent_separation.py" --dry-run --densify
  arm span       "$PY_BASE" "$REPO/scripts/span_extent_separation.py" --dry-run --no-densify
else
  arm span_dense "$PY_VLLM" "$REPO/scripts/span_extent_separation.py" --densify --max-minutes 35
  arm span       "$PY_VLLM" "$REPO/scripts/span_extent_separation.py" --no-densify --max-minutes 45
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
# Every arm's verdict, together, against the item it closes. THE ONLY THING
# GREPPED IS `^RESULT: `, printed verbatim: the arms' own scored-gate lines and
# nothing this driver worded itself.
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
  reason="$(awk -F'\t' -v a="$n" '$1==a{s=$7} END{print s}' "$LEDGER")"
  [[ -n "$state" ]] && printf '  state:  %s\n' "$state"
  [[ -n "$reason" ]] && printf '  reason: %s\n' "$reason"
  if summarize_arm "$n" "$log" "$state"; then
    continue
  fi
  if [[ "$state" == "REFUSED" || "$state" == "PLAN_REFUSED" || "$state" == "NOT_PLANNED" ]]; then
    continue
  fi
  [[ -f "$log" ]] || continue
  if [[ "$state" == "BROKEN" ]]; then
    printf '  THE PLAN IS BROKEN: this arm exited %s under --dry-run, which is not\n' \
      "$(awk -F'\t' -v a="$n" '$1==a{r=$3} END{print r}' "$LEDGER")"
    printf '  a plan and not a refusal. Nothing above is a schedule for this arm.\n'
  elif (( DRY )); then
    printf '  PLAN ONLY: a --dry-run scores no gate, so it prints no RESULT line.\n'
    printf '  off GPU its gates are exercised by: %s\n' "$(arm_offgpu_gates "$n")"
  else
    printf '  NO `RESULT: ` LINE IN %s. This arm was NOT scored: a gate that\n' "$log"
    printf '  examined nothing reports no failures. Read the log before quoting\n'
    printf '  anything from it, and do not read prose in it as a verdict.\n'
  fi
  printf '  last 5 lines:\n'
  tail -5 "$log" | sed 's/^/    /'
done

say "ARMS"
cat "$LEDGER"
(( DRY )) || contract_disclosure "$LEDGER"
printf '\ntotal %s min of wall clock\n' "$(( ($(date -u +%s) - started) / 60 ))"
printf 'work tree %s dirty file(s) at start, %s now\n' "$DIRTY_AT_START" "$(dirty_count)"

say "READ THESE THREE FIRST"
cat <<EOF
  roofline-n256-g16  the ## Verdict line, and the only arm here that can CONFIRM
               the study's claim: BLOCK_M=128 at the configuration vLLM ships.
               CEILING BINDING AT THE PRODUCTION TILE is the claim confirmed;
               CEILING NOT BINDING refutes it and the paper becomes a model
               note. Either is publishable. Read roofline-n64-g1 beside it as
               the control it is: it can only refute. And if this arm REFUSED,
               read the control bill it printed: the claim has no positive
               control at BLOCK_N=256 yet, and nothing else in this session can
               supply one.
  noise_floor  the between-replicate sd. Every effect anywhere in this study is
               to be read against it from now on, and until it exists the MDE
               printed at the top of this run is an ASSUMPTION carried from a
               same-session, same-hour, mixtral-only proxy that confounds
               num_stages.
  bn_g16       the residual line. Noise-sized residual = the three-term model is
               complete. Structure in it names what is missing. bn_g1 is not a
               second reading of this: C2 is UNKNOWN at GROUP_SIZE_M=1.
EOF

say "WHAT TO COMMIT, AND WHAT NOT TO"
cat <<EOF
  NOTHING HERE IS COMMITTED AND NOTHING IS PUSHED. This script runs no git write
  command and terminates nothing. Read the gates above first.

  THE ONE FILE THAT BELONGS IN THE REPO is this card's calibration:
      git -C $REPO diff --stat moe/bench/hardware/
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

# The session's own exit code, by the same rule it applies to its arms: a plan
# that did not survive its own --dry-run is not a plan, and an arm that exited a
# code nobody chose has not reached a state. Neither is a claim, so neither is 1.
if (( DRY )) && (( BROKEN_ARMS > 0 )); then
  say "$BROKEN_ARMS PLAN(S) BROKEN. The plan above is not a plan; fix them before the pod."
  awk -F'\t' '$2 == "BROKEN" { printf "  %-19s exited %s under --dry-run   %s\n", $1, $3, $6 }' "$LEDGER"
  cat <<'EOF'
  A plan may exit 0 (PLANNED) or 2 (PLAN_REFUSED, the refusal working off a GPU
  box). Any other code is a plan that did not print. An arm here exiting 3 is
  almost always the exit-code contract this session adopted and its script has
  not: 3 is INVALID, "measured and then a VALIDITY gate failed", and a
  --dry-run measures nothing, so a refusal that exits 3 is a refusal wearing an
  INVALID's number. Fix it in the script, through moe/bench/exit_codes.classify
  and REFUSED, not here.
EOF
  exit "$RC_INVALID"
fi
if (( DRY == 0 )) && (( RETRY_ARMS > 0 )); then
  say "$RETRY_ARMS ARM(S) EXITED A CODE OUTSIDE THE TABLE. Read their logs."
  exit "$RC_RETRY"
fi
exit 0
