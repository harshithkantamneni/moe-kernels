#!/usr/bin/env bash
# Every open experiment, as ONE unattended pod run, in the order their results
# are READ.
#
#   bash scripts/h200_gaps_session.sh --dry-run     # laptop, free: every arm's
#                                                   # own plan, its cost, the MDE
#   bash scripts/h200_gaps_session.sh               # the pod run, ~3.4 hours
#   bash scripts/h200_gaps_session.sh --list        # the arms and what each closes
#   bash scripts/h200_gaps_session.sh --only roofline-n256-g16,noise_floor
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

# WHAT A REFUSED OR INVALID ROW MAY ACTUALLY MEAN when the file that produced it
# has not adopted the table this session reads it by. Prints nothing -- rc 1 --
# for a file that has adopted, and nothing for any other state, so it is silent
# on every row whose word is trustworthy. It changes no state and no exit code;
# the whole content is disclosure, which is what the measuring path did not have.
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
    case "$state" in REFUSED|INVALID) ;; *) continue ;; esac
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
    printf '  No REFUSED or INVALID row in this session came from a file that has\n'
    printf '  not adopted moe/bench/exit_codes, so every state word above is the\n'
    printf '  one that table defines.\n'
  fi
  return 0
}

# A PLAN IS NOT A MEASUREMENT, so a plan does not get measuring words. Three
# outcomes, and only one of them is silent-failure-shaped: a --dry-run that
# tracebacks exits 1, which lands in BROKEN and stops the session, which is the
# defect this mapping exists to catch. rc 2 is a refusal, and off a GPU box a
# refusal is what several of these plans are SUPPOSED to do.
dry_state() { case "$1" in
  0) echo PLANNED ;;
  2) echo PLAN_REFUSED ;;
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

wanted() {
  [[ -z "$ONLY" ]] && return 0
  case ",$ONLY," in (*",$1,"*) return 0 ;; esac
  return 1
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
  if (( DRY )); then state="$(dry_state "$rc")"; else state="$(ledger_state "$rc")"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$state" "$rc" "$((t1 - t0))" "$after" "$log" "" >> "$LEDGER"
  note "   $state (exit $rc) in $((t1 - t0))s; work tree $after dirty file(s)"
  case "$state" in
    BROKEN)  BROKEN_ARMS=$((BROKEN_ARMS + 1))
             note "   BROKEN: this is a PLAN, and it did not survive its own --dry-run."
             note "   $(tail -1 "$log")" ;;
    RETRY)   RETRY_ARMS=$((RETRY_ARMS + 1))
             note "   exit $rc is not in the table. Read the log before re-running." ;;
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
  span)       echo "The same on the sparse grid, for the extent comparison. CLAIM_FAIL here is the registered outcome of the kernel world and is recorded as finished." ;;
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
  arm calibrate "$PY_BASE" "$REPO/scripts/calibrate_hardware.py"
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
if (( DRY )); then
  arm span_dense "$PY_BASE" "$REPO/scripts/span_extent_separation.py" --dry-run --densify
  arm span       "$PY_BASE" "$REPO/scripts/span_extent_separation.py" --dry-run
else
  arm span_dense "$PY_VLLM" "$REPO/scripts/span_extent_separation.py" --densify --max-minutes 35
  arm span       "$PY_VLLM" "$REPO/scripts/span_extent_separation.py" --max-minutes 45
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
