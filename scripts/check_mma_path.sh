#!/usr/bin/env bash
# C3: does vLLM's decode path use Hopper's warpgroup tensor core, or not?
#
#   bash scripts/check_mma_path.sh
#   bash scripts/check_mma_path.sh --tokens 16 --model deepseek-v3
#   bash scripts/check_mma_path.sh --tokens 256 --block-m 16,64   # the switch, tile-forced
#   bash scripts/check_mma_path.sh --block-m 16,64 --dry-run      # the plan, no GPU needed
#
# THE QUESTION. Hopper's `wgmma.mma_async.m64nNk16` has M fixed at 64 by the
# instruction set, and Triton selects it only when BLOCK_M % 64 == 0 AND
# num_warps % 4 == 0 (supportMMA, the version==3 branch, triton release/3.7.x
# lib/Analysis/Utility.cpp). Not "BLOCK_M >= 64": 80 or 96 fail the modulo and
# compile to `mma.sync` anyway. And below compute capability 9.0
# getMMAVersionSafe returns {2} alone, so on the A100 arm no tile reaches the
# warpgroup instruction at any size and this script has nothing to ask there.
#
# WHICH TILE ACTUALLY RUNS. An earlier version of this header asserted that
# vLLM's tuned H200 config sets BLOCK_SIZE_M to 16 for every batch size from 1
# to 256. That ladder is real but belongs to E=128,N=512, which is no model in
# this study, and the error went unchallenged for days because the published
# CSVs carry no column recording the tile the kernel actually ran. The two study
# shapes that DO ship a tuned bf16 H200 file (vLLM v0.27.1, under
# vllm/model_executor/layers/fused_moe/configs/) both climb well before 256:
#
#   E=8,N=14336,device_name=NVIDIA_H200.json    mixtral-8x7b
#     1:16 2:32 4:16 8:16 16:16 24:16 32:16 48:32 64:32 96:32
#     128:64 256:128 512:128 1024:128 1536:128 2048:128 3072:128 4096:128
#   E=64,N=2560,device_name=NVIDIA_H200.json    qwen2-57b-a14b
#     1:16 2:16 4:16 8:16 16:16 24:16 32:16 48:16 64:16 96:32
#     128:32 256:64 512:128 1024:128 1536:128 2048:128 3072:128 4096:128
#
# Those are 2 of the 8 (model x card) cells in this study. NOTHING ships for
# NVIDIA_A100-SXM4-80GB at E=8,N=14336 / E=64,N=2560 / E=64,N=1408 /
# E=256,N=2048, and nothing ships for H200 at the last two either, so the other
# six cells take the hardcoded bf16 ladder in `get_default_config`:
# M<=32 -> 16, M<=96 -> 32, M<=512 -> 64, else 128. deepseek-v3 (E=256,N=2048),
# the default --model below, is one of the six: the 16 it runs at --tokens 16 is
# that fallback and not a grid-search optimum, and vLLM says so on the run log
# with "Using default MoE config. Performance might be sub-optimal!".
#
# Two more traps in that lookup. The key is chosen by NEAREST, not floor
# (`configs[min(configs.keys(), key=lambda x: abs(x - M))]`), so M=200 reads the
# 256 entry; and M is the rows entering the layer, not tokens x top_k.
# Separately, the fp8_w8a8 files for those same two H200 shapes sit at
# BLOCK_SIZE_M 64 with num_warps 4 from M=1 upward, which passes the predicate,
# so "vLLM declines the warpgroup instruction at decode" is a claim about the
# bf16 path only.
#
# Before this script the study recorded the instruction as an INFERENCE, not a
# measurement. This settles it: Triton writes its generated PTX to disk when
# asked, CUTLASS and PTX both carry the instruction name, and the answer is a
# grep.
#
# There is nothing to interpret. The instruction is in the file or it is not.
#
# --------------------------------------------------------------------------
# --block-m: THE LOOSE END, which is a different question from the one above.
#
# `docs/STUDY.md` item 3 is marked DONE with a tail: "confirm the instruction
# actually switched by re-running check_mma_path.sh under the override". The
# 2026-08-27 census DID see wgmma=0 at BLOCK_M=16 and wgmma=8 at 64 and 128 --
# but through vLLM's CONFIG LADDER, by asking for different token counts and
# letting the ladder pick a different tile at each. Two things move along that
# ladder at once. `get_default_config` sets num_warps to 4 at M<=128 and 8 above
# it, and num_warps is the OTHER half of Triton's warpgroup predicate, so a
# census taken across token counts cannot separate "the tile selects the
# instruction" from "the warp count does" -- or from "the batch size does",
# which is what a reader who does not know the ladder will assume.
#
# So --block-m holds the token count FIXED and forces each tile in turn through
# MOE_FORCE_TILE, with BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps and
# num_stages pinned identically across every arm. num_warps=8 in particular
# satisfies `num_warps % 4 == 0` at every tile this can run, which leaves
# `BLOCK_M % 64 == 0` as the only term in the predicate that moves. That is the
# whole difference between "the instruction correlates with batch size" and
# "the instruction is selected by the tile".
#
# THE PIN IS VERIFIED, NEVER ASSUMED. "The variable was set" and "the kernel ran
# that tile" are different facts, and the gap between them cost the 2026-09-01
# session its pinned crossing (gate S6a, FAIL, "observed tile_block_m = none").
# Each arm reads all six OBSERVED tile columns back out of its own fresh run CSV
# -- written by `_framework_config.recording_tile_config`, which reads the config
# out of vLLM during a real call -- and this script REFUSES to census an arm
# whose observed tile is not exactly the tile it asked for. A census attributed
# to a tile that did not run is worse than no census.
#
# EXIT CODES ARE moe.bench.exit_codes's, MIRRORED. This is bash and cannot
# import the module, so the five integers are written out below and
# tests/test_check_mma_path.py compares them with the module's own constants and
# fails the build on a drift. The dotted spelling is load-bearing as well as
# accurate: the driver's `adopts_exit_codes` greps this file for it and prints a
# caveat beside every REFUSED or INVALID row from an arm that does not name it.
#
# The table this file declared until 2026-09-02 was its own -- "0 every gate
# passed. 1 a gate failed. 2 refused before measuring" -- and it inverted two of
# the three states it names. A VALIDITY gate that
# fails AFTER the arms have compiled is INVALID (3): the pod minutes are spent
# and the census must not be quoted, which is a different instruction to the
# driver than CLAIM_FAIL (1), "measured, valid, and the world disagreed with the
# prediction". Exiting 1 for a validity failure told the session this arm had
# produced a RESULT, and the session LATCHES a result: `arm()` skips a name
# whose ledger row already says DONE, CLAIM_FAIL or INVALID. Two more paths
# exited 1 with no gate in sight, and they are not the same state as each other:
# no interpreter at $PY fires before anything compiles and is REFUSED (2), free
# and re-runnable; the ladder cell that dumped no PTX fires after the cell has
# run, so it is now gate L1 and INVALID (3). Folding those two together is what
# costs a second rental to tell apart.
#
# And the driver reads gates through one greppable line, `RESULT: <KIND> <NAME>
# <VERDICT> <detail>` at column zero. This script scored four gates and printed
# none of them, so the session summary reported four scored gates as "This arm
# was NOT scored". Every gate here now prints exactly one, in the format
# `exit_codes.result_line` renders, and the test pins the two against each other
# character for character rather than trusting this comment.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

#: The table, mirrored from moe/bench/exit_codes.py.
#:   DONE       0  measured; every VALIDITY and CLAIM gate PASSED
#:   CLAIM_FAIL 1  measured; VALIDITY passed; a CLAIM gate did not (a RESULT)
#:   REFUSED    2  nothing measured; a precondition was not met (free)
#:   INVALID    3  measured; a VALIDITY gate failed after measuring; unquotable
#:   ERROR      4  crashed; an unplanned failure
EXIT_DONE=0
EXIT_CLAIM_FAIL=1
EXIT_REFUSED=2
EXIT_INVALID=3
EXIT_ERROR=4

# An unplanned failure is ERROR, and it has to SAY it was unplanned. Without
# this trap `set -e` exits with whatever status the failing command happened to
# return -- a `grep` that matched nothing exits 1, which is CLAIM_FAIL, and the
# session would ledger a dead script as a refuted claim and never re-run it.
# `set -E` above is what makes the trap fire inside functions and command
# substitutions; the run of the sweep is exempt because it is captured and
# translated explicitly (see `sweep_failed`).
on_unhandled_error() {
  local rc=$?
  printf '[mma] UNHANDLED FAILURE (status %s) at line %s: %s\n' \
    "$rc" "${BASH_LINENO[0]}" "$BASH_COMMAND" >&2
  printf '[mma] Nothing above was scored and nothing may be read as a verdict.\n' >&2
  exit "$EXIT_ERROR"
}
trap on_unhandled_error ERR

# >>> LIFTABLE: function definitions only, no top-level statements. Lifted by
# tests/test_check_mma_path.py, which evaluates this block in a fresh bash and
# asks the SHIPPED functions rather than a python copy of their case statements.

#: `RESULT: <KIND> <NAME> <VERDICT> <detail>`, the ONE line the driver greps.
#: Mirrors `moe/bench/exit_codes.py:result_line`, including its refusals: a name
#: with whitespace in it, or a kind or verdict outside the table, renders a line
#: `parse_result_lines` cannot read back, and a gate the driver cannot parse is a
#: gate that silently disappears from the summary. The detail is squeezed onto
#: one line for the same reason.
result_line() {   # <kind> <name> <verdict> [detail ...]
  local kind="$1" name="$2" verdict="$3"; shift 3
  local detail
  case "$kind" in
    VALIDITY|CLAIM) : ;;
    *) echo "[mma] result_line: kind '$kind' is not VALIDITY or CLAIM" >&2
       exit "$EXIT_ERROR" ;;
  esac
  case "$verdict" in
    PASS|FAIL|UNKNOWN) : ;;
    *) echo "[mma] result_line: verdict '$verdict' is not PASS, FAIL or UNKNOWN" >&2
       exit "$EXIT_ERROR" ;;
  esac
  case "$name" in
    ""|*[[:space:]]*)
       echo "[mma] result_line: name '$name' must be one non-empty token" >&2
       exit "$EXIT_ERROR" ;;
  esac
  detail="$(printf '%s' "$*" | tr '\n\t' '  ' | tr -s ' ' | sed 's/^ *//;s/ *$//')"
  if [[ -n "$detail" ]]; then
    printf 'RESULT: %s %s %s %s\n' "$kind" "$name" "$verdict" "$detail"
  else
    printf 'RESULT: %s %s %s\n' "$kind" "$name" "$verdict"
  fi
}

#: PASS or FAIL from an arithmetic truth value, so a caller writes the condition
#: once and the word is derived from it rather than typed beside it.
pass_fail() {   # <0|1>
  if (( $1 )); then echo PASS; else echo FAIL; fi
}

#: One scored gate: its RESULT line, then the three human lines this script has
#: always printed. Both are kept because they have different readers, and only
#: the first is the machine contract. Recorded as well as printed, so `classify`
#: scores exactly the gates that reached the page.
GATE_ROWS=""
GATE_N=0
gate() {   # <tag> <kind> <verdict> <claim> <measured> <threshold> <consequence>
  local tag="$1" kind="$2" verdict="$3" claim="$4" measured="$5" \
        threshold="$6" cons="$7"
  result_line "$kind" "$tag" "$verdict" \
    "[$kind] $claim | measured $measured | gate $threshold"
  printf '%-3s %-8s %-4s %s\n' "$tag" "$kind" "$verdict" "$claim"
  printf '             measured %s   gate %s\n' "$measured" "$threshold"
  printf '             if this FAILS: %s\n' "$cons"
  GATE_ROWS="${GATE_ROWS}${kind}"$'\t'"${verdict}"$'\n'
  GATE_N=$((GATE_N + 1))
}

#: The code the recorded gates imply, by `exit_codes.classify`'s rule: any
#: VALIDITY gate not PASS is INVALID, else any CLAIM gate not PASS is
#: CLAIM_FAIL, else DONE. UNKNOWN counts against a gate exactly as FAIL does --
#: "a check that examined nothing reports zero failures" is this project's
#: documented failure shape. No gate at all returns non-zero rather than DONE,
#: for the same reason `classify([])` raises.
classify() {
  local kind verdict invalid=0 claim_failed=0
  if (( GATE_N == 0 )); then
    echo "[mma] classify: no gate was scored, so there is no verdict to exit with" >&2
    return 1
  fi
  while IFS=$'\t' read -r kind verdict; do
    if [[ -z "$kind" ]]; then continue; fi
    if [[ "$verdict" != PASS ]]; then
      if [[ "$kind" == VALIDITY ]]; then invalid=1; else claim_failed=1; fi
    fi
  done <<< "$GATE_ROWS"
  if (( invalid )); then echo "$EXIT_INVALID"
  elif (( claim_failed )); then echo "$EXIT_CLAIM_FAIL"
  else echo "$EXIT_DONE"; fi
}
# <<< LIFTABLE

WORKSPACE="${WORKSPACE:-/workspace}"
DUMP_DIR="${MOE_PTX_DIR:-$WORKSPACE/ptx}"
MODEL="deepseek-v3"
TOKENS="16"
ENV_NAME="vllm"
BLOCK_M_LIST=""
DRY_RUN=0
SELF_TEST=""

# The five knobs held IDENTICAL across every forced arm, so BLOCK_SIZE_M is the
# only thing that can move. Same values scripts/tile_sweep.py and
# scripts/block_m_crossing_sweep.py pin, so a PTX census and a timing sweep
# describe the same kernel. num_warps=8 is load bearing: it clears the
# `num_warps % 4 == 0` half of Triton's warpgroup predicate at every tile, which
# is what leaves BLOCK_M as the only term that varies.
BLOCK_N="64"
BLOCK_K="64"
GROUP_M="1"
NUM_WARPS="8"
NUM_STAGES="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokens)   TOKENS="$2"; shift 2 ;;
    --model)    MODEL="$2"; shift 2 ;;
    --env)      ENV_NAME="$2"; shift 2 ;;
    --out)      DUMP_DIR="$2"; shift 2 ;;
    --block-m)  BLOCK_M_LIST="$2"; shift 2 ;;
    --block-n)  BLOCK_N="$2"; shift 2 ;;
    --block-k)  BLOCK_K="$2"; shift 2 ;;
    --group-m)  GROUP_M="$2"; shift 2 ;;
    --warps)    NUM_WARPS="$2"; shift 2 ;;
    --stages)   NUM_STAGES="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --self-test) SELF_TEST="$2"; shift 2 ;;
    # REFUSED: an argument this script does not know is a session asking for
    # something it will not do, before anything is measured.
    *) echo "[mma] REFUSED: unknown argument: $1" >&2; exit "$EXIT_REFUSED" ;;
  esac
done

log() { printf '[mma] %s\n' "$*"; }
#: REFUSED (2): nothing was measured, no pod minute was spent, and the arm is
#: re-runnable the moment the precondition holds. Never used after an arm has
#: compiled -- that is INVALID, and the difference is a whole rental.
refuse() { echo "[mma] REFUSED: $*" >&2; exit "$EXIT_REFUSED"; }

# --------------------------------------------------------------------------
# THE GATES, and the one exit. Both scoring paths live here rather than beside
# their call sites so `--self-test` can drive the SHIPPED scorer over a planted
# world: a gate that has only ever been seen passing is the shape of check this
# apparatus is being repaired for.
# --------------------------------------------------------------------------

#: The unforced cell, on vLLM's own ladder. One VALIDITY gate and one CLAIM
#: gate: the census is only evidence if a kernel with a tensor-core instruction
#: was actually dumped, and C3 is the claim that cell was run to test.
score_ladder_gates() {   # reads ARM_PTX ARM_WGMMA ARM_MMA
  local c3
  if   (( ARM_WGMMA > 0 ));                     then c3=FAIL
  elif (( ARM_MMA > 0 ));                       then c3=PASS
  else                                               c3=UNKNOWN
  fi
  gate L1 VALIDITY "$(pass_fail $(( ARM_PTX > 0 && ARM_WGMMA + ARM_MMA > 0 )))" \
    "the cell dumped a PTX carrying a tensor-core instruction" \
    "$ARM_PTX PTX file(s); $ARM_WGMMA with wgmma, $ARM_MMA with mma.sync" \
    ">= 1 of each" \
    "an absent instruction and an absent kernel look identical in the census
             above, so nothing here may be read as a statement about the ISA.
             Check the run log: the span may never have run, vLLM may be absent
             from this env, or the cell may have fallen back off Triton."
  gate L2 CLAIM "$c3" \
    "C3: the decode path's kernel carries no warpgroup MMA" \
    "wgmma in $ARM_WGMMA kernel(s), mma.sync in $ARM_MMA" \
    "wgmma == 0 with mma.sync > 0" \
    "warpgroup MMA IS present, so Triton reached M=64 some other way and the
             inference from BLOCK_SIZE_M=16 was wrong. UNKNOWN when neither
             instruction was found, which decides nothing and counts against the
             gate for exactly that reason."
}

#: The forced arms. G1-G3 are VALIDITY -- they say whether the census belongs to
#: the tile it is attributed to -- and G4 is the CLAIM the arms were run to
#: test. A G4 FAIL is a finding, and the table gives it its own code (1) so the
#: session records it as finished rather than retrying until it passes.
score_forced_gates() {   # reads narms g1_fail g2_fail distinct g4_fail
  gate G1 VALIDITY "$(pass_fail $(( g1_fail == 0 )))" \
    "every arm ran the tile it was given" \
    "$(( narms - g1_fail ))/$narms arms observed exactly what they forced" \
    "all $narms" \
    "MOE_FORCE_TILE did not reach the kernel in some arm, so its census belongs to
             a tile nobody chose. Nothing on this page may be attributed to a tile.
             Check that moe/bench/force_tile.py is present in this checkout and that
             the impl under test exposes the force_tile_config hook."
  gate G2 VALIDITY "$(pass_fail $(( g2_fail == 0 )))" \
    "every arm dumped a PTX carrying a tensor-core instruction" \
    "$(( narms - g2_fail ))/$narms arms" "all $narms" \
    "an arm compiled nothing, or nothing with a tensor core in it. An absent
             instruction and an absent kernel look identical in the table above."
  gate G3 VALIDITY "$(pass_fail $(( distinct == narms )))" \
    "the arms compiled DIFFERENT kernels" \
    "$distinct distinct PTX checksum(s)" "== $narms" \
    "two arms produced byte-identical PTX, so the tile did not reach the compile
             and the census is one kernel compared with itself."
  gate G4 CLAIM "$(pass_fail $(( g4_fail == 0 )))" \
    "wgmma appears in exactly the arms BLOCK_M % 64 == 0 names" \
    "$(( narms - g4_fail ))/$narms arms match the prediction" "all $narms" \
    "the instruction is NOT selected by the tile height alone at fixed num_warps,
             which is a result: Triton's predicate as read from supportMMA does not
             describe what this build emits."
}

#: Score, then leave. The code is `classify` over the gates just printed, so
#: `exit_codes.classify_text` over this log recomputes the integer the process
#: returned and a disagreement between the two is itself a defect.
finish() {
  local rc
  rc="$(classify)" || exit "$EXIT_ERROR"
  echo
  case "$rc" in
    "$EXIT_DONE")
      echo "[mma] exit $rc DONE: measured, and every VALIDITY and CLAIM gate PASSED." ;;
    "$EXIT_CLAIM_FAIL")
      echo "[mma] exit $rc CLAIM_FAIL: measured, every validity gate passed, and a"
      echo "[mma] pre-registered claim did not hold. That is a RESULT and the arm is"
      echo "[mma] FINISHED: quote the table, not the prediction, and do not re-run it"
      echo "[mma] hoping for the other answer." ;;
    "$EXIT_INVALID")
      echo "[mma] exit $rc INVALID: a VALIDITY gate failed AFTER the arms compiled."
      echo "[mma] Nothing on this page may be quoted, the census must not be scored,"
      echo "[mma] and this arm is NOT auto-retried: re-run it only once the log says"
      echo "[mma] in words what changed." ;;
  esac
  exit "$rc"
}

#: The sweep under one arm did not finish. Its code is `moe/bench/exit_codes.py`'s
#: already, so it is adopted rather than re-invented: a sweep that refused before
#: measuring makes this arm a refusal too, and a sweep whose own validity gate
#: failed after measuring makes it INVALID. Anything outside the table is ERROR,
#: because a code nobody chose carries no information about what happened.
sweep_failed() {   # <label> <rc> <log-file>
  local label="$1" rc="$2" log_file="$3"
  echo >&2
  case "$rc" in
    "$EXIT_REFUSED")
      refuse "the [$label] sweep refused before measuring (exit $rc). Its first
  REFUSED line above says what to fix; nothing was compiled, so no census exists
  and none was scored." ;;
    "$EXIT_INVALID"|"$EXIT_CLAIM_FAIL")
      echo "[mma] the [$label] sweep exited $rc: it ran and then failed a gate of" >&2
      echo "[mma] its own -- see its RESULT lines above. The pin is the thing this" >&2
      echo "[mma] script attributes its census to, so a run that cannot show the" >&2
      echo "[mma] tile leaves nothing here quotable." >&2
      exit "$EXIT_INVALID" ;;
    *)
      echo "[mma] the [$label] sweep exited $rc, which is not in the table. Read" >&2
      echo "[mma] $log_file before re-running." >&2
      exit "$EXIT_ERROR" ;;
  esac
}

#: The gates, off GPU, over a planted world. Every case names the verdicts it
#: expects the scorer to produce, and three of the six plant a FAILING branch,
#: because a scorer only ever exercised on its passing path is how G4 could have
#: been printing the wrong verdict for four arms without anyone noticing. The
#: proof that this can itself fail is `unhandled`, which plants a command that
#: does not exist and asserts the ERR trap turns it into ERROR rather than into
#: whatever integer the shell happened to return.
self_test() {   # <case>
  echo "=== --self-test $1: a PLANTED world. Nothing was compiled and no GPU"
  echo "    was touched, so no number below is a measurement. The gates, the"
  echo "    RESULT lines and the exit code are the shipped ones."
  echo
  case "$1" in
    forced-pass)     narms=2 g1_fail=0 g2_fail=0 distinct=2 g4_fail=0
                     echo "planted: two arms, both honoured the pin, both compiled"
                     echo "         their own kernel, both matched the prediction"
                     echo "         -> DONE ($EXIT_DONE)"; echo
                     score_forced_gates ;;
    forced-invalid)  narms=2 g1_fail=1 g2_fail=0 distinct=2 g4_fail=0
                     echo "planted: one arm observed a tile it did not force"
                     echo "         -> INVALID ($EXIT_INVALID), G1 FAIL"; echo
                     score_forced_gates ;;
    forced-claim)    narms=2 g1_fail=0 g2_fail=0 distinct=2 g4_fail=1
                     echo "planted: every validity gate passed and one arm's wgmma"
                     echo "         census contradicted the predicate"
                     echo "         -> CLAIM_FAIL ($EXIT_CLAIM_FAIL), G4 FAIL"; echo
                     score_forced_gates ;;
    ladder-pass)     ARM_PTX=3 ARM_WGMMA=0 ARM_MMA=2
                     echo "planted: a ladder cell on mma.sync alone -> DONE ($EXIT_DONE)"; echo
                     score_ladder_gates ;;
    ladder-refuted)  ARM_PTX=3 ARM_WGMMA=1 ARM_MMA=1
                     echo "planted: warpgroup MMA present on the ladder cell"
                     echo "         -> CLAIM_FAIL ($EXIT_CLAIM_FAIL), L2 FAIL"; echo
                     score_ladder_gates ;;
    ladder-invalid)  ARM_PTX=0 ARM_WGMMA=0 ARM_MMA=0
                     echo "planted: nothing compiled -> INVALID ($EXIT_INVALID), L1 FAIL"
                     echo "         and L2 UNKNOWN, which counts against it"; echo
                     score_ladder_gates ;;
    unhandled)       echo "planted: a command that does not exist -> ERROR ($EXIT_ERROR)"; echo
                     a_command_this_script_never_planned_for ;;
    *) refuse "--self-test '$1' is not one of forced-pass, forced-invalid,
  forced-claim, ladder-pass, ladder-refuted, ladder-invalid, unhandled." ;;
  esac
  finish
}

if [[ -n "$SELF_TEST" ]]; then
  self_test "$SELF_TEST"
fi

# --------------------------------------------------------------------------
# The plan and the prediction, printed before anything runs and before the
# interpreter is even looked for, so --dry-run works on a laptop.
# --------------------------------------------------------------------------
predicts_wgmma() {   # <block_m> <num_warps> -> echoes yes|no
  if (( $1 % 64 == 0 && $2 % 4 == 0 )); then echo yes; else echo no; fi
}

ARMS=()
if [[ -n "$BLOCK_M_LIST" ]]; then
  case "$TOKENS" in
    *,*) refuse "--block-m forces the tile at a FIXED token count, and --tokens
  '$TOKENS' names several. Splitting the batch across arms puts back exactly the
  confound this mode removes. Run one token count per invocation." ;;
  esac
  OLD_IFS="$IFS"; IFS=','; for bm in $BLOCK_M_LIST; do ARMS+=("$bm"); done
  IFS="$OLD_IFS"
  (( ${#ARMS[@]} >= 2 )) || refuse "--block-m needs at least two tiles: this mode
  is a COMPARISON at one token count, and one arm compares nothing. Try
  --block-m 16,64."

  yes_n=0; no_n=0
  for bm in "${ARMS[@]}"; do
    [[ "$bm" =~ ^[0-9]+$ ]] || refuse "--block-m '$bm' is not a number"
    if [[ "$(predicts_wgmma "$bm" "$NUM_WARPS")" == yes ]]; then
      yes_n=$((yes_n + 1)); else no_n=$((no_n + 1)); fi
  done
  # NON-VACUITY, at plan time. Every arm on the same side of `BLOCK_M % 64 == 0`
  # produces a table where the instruction never changes, which reads as a
  # finding and is a tautology.
  (( yes_n > 0 && no_n > 0 )) || refuse "every tile in --block-m '$BLOCK_M_LIST'
  falls on the SAME side of the predicate BLOCK_M % 64 == 0 at num_warps=$NUM_WARPS
  ($yes_n predicted to emit wgmma, $no_n not). The census would then be constant
  by construction and could not show a switch. Include one tile from each side,
  e.g. --block-m 16,64."

  echo "=== plan: the instruction at a FIXED token count, only BLOCK_SIZE_M moving ==="
  echo "  model $MODEL   tokens $TOKENS   env $ENV_NAME"
  echo "  pinned across every arm: BLOCK_SIZE_N=$BLOCK_N BLOCK_SIZE_K=$BLOCK_K"
  echo "                           GROUP_SIZE_M=$GROUP_M num_warps=$NUM_WARPS num_stages=$NUM_STAGES"
  echo "  forced through MOE_FORCE_TILE, and each arm's OBSERVED tile columns are"
  echo "  read back out of its own run CSV before its PTX is censused."
  echo
  echo "  PREDICTIONS, registered before the run. Triton emits wgmma only when"
  echo "  BLOCK_M % 64 == 0 AND num_warps % 4 == 0; num_warps is pinned at"
  echo "  $NUM_WARPS in every arm, so only the first term can move:"
  for bm in "${ARMS[@]}"; do
    if [[ "$(predicts_wgmma "$bm" "$NUM_WARPS")" == yes ]]; then
      printf '    BLOCK_SIZE_M=%-4s wgmma  > 0   (%s %% 64 == 0)\n' "$bm" "$bm"
    else
      printf '    BLOCK_SIZE_M=%-4s wgmma == 0   (%s %% 64 != 0, so mma.sync)\n' "$bm" "$bm"
    fi
  done
  echo
  echo "  GATES"
  echo "    G1 VALIDITY  every arm's OBSERVED six tile columns equal the forced ones"
  echo "    G2 VALIDITY  every arm dumped at least one PTX carrying a tensor-core instruction"
  echo "    G3 VALIDITY  the arms compiled DIFFERENT kernels (distinct PTX checksums)"
  echo "    G4 CLAIM     wgmma is present in exactly the arms the predicate names"
  echo "  A G1-G3 FAIL means no census on the page may be attributed to a tile,"
  echo "  and the run exits 3 INVALID: measured, unquotable, and NOT re-run."
  echo "  A G4 FAIL is a result: the instruction is not selected by the tile alone,"
  echo "  and the run exits 1 CLAIM_FAIL, which the session records as finished."
  echo "  Each of the four prints one RESULT: line, which is all the driver reads."
  echo "  dumps to $DUMP_DIR/bm<N>/"
else
  echo "=== plan: one cell on vLLM's own config ladder ==="
  echo "  model $MODEL   tokens $TOKENS   env $ENV_NAME   dump $DUMP_DIR"
  echo "  no tile is forced, so the tile is whatever the ladder resolves and the"
  echo "  instruction cannot be attributed to it alone. Use --block-m 16,64 for"
  echo "  the attribution."
fi
echo

if (( DRY_RUN )); then
  # REFUSED (2), which is what a plan is: it scores no gate, so it prints no
  # RESULT line, and `classify_text` over a log with none raises NoGatesScored
  # -- the module documents that as exactly what a REFUSED log looks like from
  # there. DONE would read "measured; every VALIDITY and CLAIM gate PASSED",
  # which is false of every plan, one line under "nothing was executed". The
  # driver re-queues neither state: `dry_state` maps 0 to PLANNED and 2 to
  # PLAN_REFUSED, and `arm` retries neither.
  echo "[mma] REFUSED: --dry-run scored no gate, because nothing was executed"
  echo "[mma] and nothing was written. The thresholds above are REGISTERED,"
  echo "[mma] which is not the same statement as met."
  exit "$EXIT_REFUSED"
fi

PY="${MOE_PYTHON:-$WORKSPACE/venvs/$ENV_NAME/bin/python}"
# REFUSED (2), not 1. Nothing has been compiled at this point, so this costs
# nothing and the arm runs the moment the venv exists. Exiting 1 said CLAIM_FAIL
# -- a pre-registered claim tested and refuted -- and the session ledgers a
# result as finished and never asks for it again.
[[ -x "$PY" ]] || refuse "no interpreter at $PY; run setup_runpod.sh"

#: Dropped inside a dump directory so a later run can recognise its own output.
DUMP_MARKER=".moe-ptx-dump"

# DUMP_DIR arrives from MOE_PTX_DIR or --out and is handed straight to `rm -rf`,
# so it is checked before it is deleted. MOE_PTX_DIR="" collapses the default,
# `--out` with a missing value swallows the next flag, and MOE_PTX_DIR=$HOME on a
# rented pod is one keystroke from a path whose recursive removal cannot be
# undone. A blacklist alone is not enough, so an EXISTING directory must also be
# empty or carry this script's own marker: that is what "clearly a dump
# directory" means here, and nothing else is deleted.
case "$DUMP_DIR" in
  "") refuse "refusing to rm -rf an empty path" ;;
  /*) : ;;
  *)  refuse "refusing to rm -rf '$DUMP_DIR': not an absolute path" ;;
esac
case "${DUMP_DIR%/}" in
  ""|"/"|"/root"|"/home"|"/tmp"|"/usr"|"/etc"|"/var"|"/workspace"|"$HOME"|"$WORKSPACE"|"$REPO_ROOT")
    refuse "refusing to rm -rf '$DUMP_DIR': that is a root, home, workspace or repo directory" ;;
esac
if [[ -e "$DUMP_DIR" ]]; then
  [[ -d "$DUMP_DIR" ]] || refuse "refusing to rm -rf '$DUMP_DIR': it exists and is not a directory"
  if [[ ! -e "$DUMP_DIR/$DUMP_MARKER" && -n "$(ls -A "$DUMP_DIR")" ]]; then
    refuse "refusing to rm -rf '$DUMP_DIR': it is not empty and carries no $DUMP_MARKER, so this script did not write it"
  fi
fi

DUMP_DIR="${DUMP_DIR%/}"   # so the -prune path below matches find's output exactly

rm -rf "$DUMP_DIR"
mkdir -p "$DUMP_DIR"
: > "$DUMP_DIR/$DUMP_MARKER"
log "dumping Triton IR and PTX to $DUMP_DIR"

# --------------------------------------------------------------------------
# One arm: compile, then census. Set as globals rather than returned, because
# bash returns an exit status and these are four numbers.
#
# ARM_WGMMA / ARM_MMA  kernels containing each instruction
# ARM_PTX              PTX files found
# ARM_SUM              checksum over the PTX, for the "did the kernel change" assay
# --------------------------------------------------------------------------
run_arm() {   # <dump-subdir> <label> [force-json]
  local dir="$1" label="$2" force="${3:-}"
  local cache="$dir/_cache" results="$dir/results" log_file="$dir/run.log"
  mkdir -p "$dir" "$results"

  # A CACHE HIT DUMPS NO PTX, and that is the failure this line exists to
  # prevent. TRITON_KERNEL_DUMP asks for the dump, but Triton does not recompile
  # a kernel it has already built, and setup_runpod.sh exports a SHARED
  # TRITON_CACHE_DIR=$WORKSPACE/triton-cache that every earlier sweep on the pod
  # has already populated with this exact fused_moe specialisation. Inheriting it
  # means nothing compiles, nothing is written, and the script exits 1 reporting
  # that the kernel never compiled -- which is very likely why the A100 was never
  # successfully dumped. A per-ARM cache forces every specialisation to be built,
  # exactly as scripts/tile_sweep.py arm_ptx_dump does.
  #
  # The run CSV goes under the arm too, and that is not tidiness: the observed
  # tile is read back out of it, and a shared results directory accumulating
  # run_*.csv from previous arms and previous days would hand this arm another
  # arm's tile.
  #
  # TEE'd rather than piped to `tail -5`, and with no `|| true`. The discarded
  # lines were the evidence: vLLM logs "Using configuration from FILE for MoE
  # layer." on a tuned hit and "Using default MoE config. Performance might be
  # sub-optimal!" on a miss, and that single line is what turns every tile
  # statement in FINDINGS C3 and C5 from DERIVED into OBSERVED. `|| true` also
  # meant a sweep that died -- OOM, missing vLLM, a bad --model -- still reached
  # the PTX scan and reported on whatever stale files were lying around.
  #
  # THE SWEEP'S OWN EXIT CODE IS READ, AND IT IS THE SAME TABLE. `moe.bench.cli`
  # is itself a gated arm under MOE_FORCE_TILE (gates F1 and F2), so its integer
  # already says which of REFUSED / INVALID / ERROR happened, and translating it
  # here is the difference between "the pin never reached the kernel" and "this
  # script crashed". `set -e` AND the ERR trap are lifted around the pipeline
  # for exactly as long as it takes to read PIPESTATUS[0]. Both are needed: the
  # ERR trap fires on a failing command whether or not errexit is on, so without
  # `trap - ERR` every one of those states is reported as ERROR (4) -- which is
  # what this script did on the first run of the repaired path, turning the
  # sweep's own REFUSED into a crash. `tee`'s status is not the sweep's, which
  # is why the code comes from PIPESTATUS[0] and not from `$?`.
  local rc=0
  set +e
  trap - ERR
  if [[ -n "$force" ]]; then
    MOE_FORCE_TILE="$force" TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$dir" \
      TRITON_CACHE_DIR="$cache" \
      "$PY" -m moe.bench.cli --env "$ENV_NAME" --profile smoke \
            --groups baselines --models "$MODEL" --tokens "$TOKENS" \
            --out-dir "$results" 2>&1 | tee "$log_file"
  else
    TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$dir" TRITON_CACHE_DIR="$cache" \
      "$PY" -m moe.bench.cli --env "$ENV_NAME" --profile smoke \
            --groups baselines --models "$MODEL" --tokens "$TOKENS" \
            --out-dir "$results" 2>&1 | tee "$log_file"
  fi
  rc=${PIPESTATUS[0]}
  trap on_unhandled_error ERR
  set -e
  if (( rc != 0 )); then
    sweep_failed "$label" "$rc" "$log_file"
  fi

  echo
  echo "=== [$label] which config vLLM resolved (quote this, do not derive it) ==="
  grep -E 'Using configuration from|Using default MoE config' "$log_file" \
    || echo "  NEITHER LINE IN $log_file. vLLM logs one of them once per
  (E,N,dtype,device) via logger.info_once/warning_once, so a second cell in the
  same process is silent; check that this run was the first fused_experts call,
  and that the log level lets info through."

  # `find` rather than `shopt -s globstar`, which is bash 4+ and therefore cannot
  # be exercised on a macOS bash 3.2 laptop before the pod is rented. The -prune
  # is load-bearing: the per-arm Triton cache lives UNDER the arm directory and a
  # cache entry stores its own .ptx beside the cubin, so without it every kernel
  # is counted twice, once under a hash-named path.
  ARM_FILES=()
  local f
  while IFS= read -r -d '' f; do
    ARM_FILES+=("$f")
  done < <(find "$dir" -path "$cache" -prune -o -name '*.ptx' -print0)
  ARM_PTX=${#ARM_FILES[@]}
  ARM_WGMMA=0
  ARM_MMA=0
  ARM_SUM="none"
  if (( ARM_PTX == 0 )); then
    return 0
  fi

  echo
  echo "=== [$label] tensor core instructions, by kernel ==="
  local w m l
  for f in "${ARM_FILES[@]}"; do
    w=$(grep -c 'wgmma' "$f" || true)
    m=$(grep -c 'mma\.sync' "$f" || true)
    l=$(grep -c 'ld\.global' "$f" || true)
    : "${w:=0}" "${m:=0}" "${l:=0}"
    # `if` rather than the `(( ... )) && continue` this line used to be. The
    # census now lives in a function, and a && list whose left side is false on
    # the last iteration leaves the loop -- and so the function -- returning
    # non-zero, which under `set -e` aborts the run at the call site with no
    # message at all.
    if (( w != 0 || m != 0 )); then
      printf '  %-52s wgmma=%-6s mma.sync=%-6s ld.global=%s\n' \
        "$(basename "$f")" "$w" "$m" "$l"
    fi
  done

  echo
  echo "=== [$label] exact instruction shapes seen ==="
  grep -ohE 'wgmma\.[a-z0-9_.]*|mma\.sync\.[a-z0-9_.]*' "${ARM_FILES[@]}" \
    | sort | uniq -c | sort -rn | head -20

  # `|| true` is load-bearing: grep exits 1 when it matches nothing, and under
  # `set -o pipefail` that fails the pipeline, fails the assignment, and aborts
  # the script before the verdict prints. Which is exactly what happened on the
  # first real run, where wgmma=0 was the whole answer.
  ARM_WGMMA=$( { grep -l 'wgmma' "${ARM_FILES[@]}" 2>/dev/null || true; } | wc -l | tr -d ' ')
  ARM_MMA=$( { grep -l 'mma\.sync' "${ARM_FILES[@]}" 2>/dev/null || true; } | wc -l | tr -d ' ')
  # cksum rather than sha1sum/shasum: POSIX, present on both the pod and a
  # macOS laptop, and this is an "are these two files identical" question with
  # no adversary in it. Sorted so file discovery order cannot change the digest.
  ARM_SUM=$( { cat "${ARM_FILES[@]}" 2>/dev/null || true; } | sort | cksum | awk '{print $1}')
}

# The six OBSERVED tile columns, read back out of one arm's own run CSV. Prints
# `bm=.. bn=.. bk=.. gm=.. warps=.. stages=.. src=.. rows=N`, with each field
# the comma-joined set of DISTINCT values seen. Distinct, not a mean or a first:
# two rows disagreeing about the tile is a fact this must show rather than
# average away.
observed_tile() {   # <results-dir>
  "$PY" - "$1" "$REPO_ROOT" <<'PYEOF'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])
from moe.bench.schema import UNRECORDED, read_csv

COLS = [("bm", "tile_block_m"), ("bn", "tile_block_n"), ("bk", "tile_block_k"),
        ("gm", "tile_group_m"), ("warps", "tile_num_warps"),
        ("stages", "tile_num_stages"), ("src", "tile_config_source")]
seen = {short: set() for short, _ in COLS}
rows = 0
for path in sorted(Path(sys.argv[1]).glob("run_*.csv")):
    for row in read_csv(path):
        rows += 1
        for short, column in COLS:
            value = row.get(column)
            if value not in (None, "", UNRECORDED, "0", 0):
                seen[short].add(str(value))
parts = [f"{short}={','.join(sorted(seen[short])) or 'none'}" for short, _ in COLS]
print(" ".join(parts) + f" rows={rows}")
PYEOF
}

# --------------------------------------------------------------------------
# UNFORCED: one cell on vLLM's own ladder. The original behaviour, unchanged,
# and still the right question when what is wanted is "what does this cell
# actually run".
# --------------------------------------------------------------------------
if [[ -z "$BLOCK_M_LIST" ]]; then
  run_arm "$DUMP_DIR" "ladder"
  if (( ARM_PTX == 0 )); then
    echo "[mma] no .ptx under $DUMP_DIR (excluding its cache)." >&2
    echo "[mma] A stale cache is no longer a possible cause: TRITON_CACHE_DIR is" >&2
    echo "[mma] per-run and was empty a moment ago, so every kernel recompiled." >&2
    echo "[mma] What is left: the span never ran (check the log), vLLM is absent" >&2
    echo "[mma] from this env, or the cell fell back to a non-Triton path." >&2
  else
    log "found $ARM_PTX PTX file(s)"
  fi
  echo
  echo "=== verdict ==="
  echo "  kernels containing wgmma   : $ARM_WGMMA"
  echo "  kernels containing mma.sync: $ARM_MMA"
  if (( ARM_WGMMA == 0 && ARM_MMA > 0 )); then
    echo "  -> C3 CONFIRMED: no warpgroup MMA. The decode path is on the"
    echo "     Ampere-era instruction, on Hopper silicon."
  elif (( ARM_WGMMA > 0 )); then
    echo "  -> C3 REFUTED: warpgroup MMA is present. Triton is reaching M=64"
    echo "     some other way, and the inference from BLOCK_SIZE_M=16 was wrong."
  else
    echo "  -> INCONCLUSIVE: neither instruction found. Check the fused_moe kernel"
    echo "     actually compiled, and that the right env was used."
  fi
  echo
  echo "  THE TILE HERE IS THE LADDER'S, NOT AN OBSERVATION UNDER CONTROL. Along"
  echo "  the ladder num_warps moves with BLOCK_SIZE_M (4 at M<=128, 8 above), so"
  echo "  this cell cannot say which of the two selected the instruction. That is"
  echo "  what --block-m 16,64 is for."
  echo
  echo "  PTX kept at $DUMP_DIR, run log at $DUMP_DIR/run.log."
  echo "  Both are transient pod output until they are committed; FINDINGS lists"
  echo "  that as the reason C1 and C3 cannot be checked without a GPU."
  echo
  echo "=== gates ==="
  # THE LADDER CELL IS SCORED TOO. It used to exit 0 having scored nothing,
  # which is DONE -- "measured; every gate PASSED" -- out of a run where no gate
  # existed to pass, and 1 when no PTX was dumped, which is CLAIM_FAIL. Both are
  # states the table already names: an absent kernel is a VALIDITY failure after
  # the cell has run (INVALID), and warpgroup MMA turning up on the ladder is
  # C3 refuted (CLAIM_FAIL), which is a result rather than a fault.
  score_ladder_gates
  finish
fi

# --------------------------------------------------------------------------
# FORCED: one arm per tile, at one token count, with only BLOCK_SIZE_M moving.
# --------------------------------------------------------------------------
SUMMARY="$DUMP_DIR/summary.tsv"
: > "$SUMMARY"
g1_fail=0; g2_fail=0

for bm in "${ARMS[@]}"; do
  dir="$DUMP_DIR/bm$bm"
  force="{\"BLOCK_SIZE_M\":$bm,\"BLOCK_SIZE_N\":$BLOCK_N,\"BLOCK_SIZE_K\":$BLOCK_K,\"GROUP_SIZE_M\":$GROUP_M,\"num_warps\":$NUM_WARPS,\"num_stages\":$NUM_STAGES}"
  echo
  echo "############ arm BLOCK_SIZE_M=$bm  (T=$TOKENS, everything else pinned) ############"
  echo "MOE_FORCE_TILE=$force"
  run_arm "$dir" "BM=$bm" "$force"

  # G1's evidence, read before the census is believed. `|| true` so a failure to
  # read the CSV becomes the string "none" and a G1 FAIL, rather than aborting
  # under `set -e` and losing the arms already measured.
  obs="$(observed_tile "$dir/results" 2>/dev/null || true)"
  [[ -n "$obs" ]] || obs="bm=none bn=none bk=none gm=none warps=none stages=none src=none rows=0"
  want="bm=$bm bn=$BLOCK_N bk=$BLOCK_K gm=$GROUP_M warps=$NUM_WARPS stages=$NUM_STAGES src=vllm_override"
  got="${obs% rows=*}"
  [[ "$got" == "$want" ]] || g1_fail=$((g1_fail + 1))
  (( ARM_WGMMA + ARM_MMA > 0 )) || g2_fail=$((g2_fail + 1))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$bm" "$(predicts_wgmma "$bm" "$NUM_WARPS")" "$ARM_WGMMA" "$ARM_MMA" \
    "$ARM_PTX" "$ARM_SUM" "$obs" >> "$SUMMARY"
  echo
  echo "  observed tile: $obs"
  echo "  asked for    : $want"
done

echo
echo "=== the switch at T=$TOKENS, only BLOCK_SIZE_M moving ==="
printf '  %-9s %-11s %-9s %-11s %-6s %s\n' \
  BLOCK_M "predicted" "kernels" "kernels" "PTX" "observed tile"
printf '  %-9s %-11s %-9s %-11s %-6s %s\n' \
  "" "wgmma" "w/ wgmma" "w/ mma.sync" "files" ""
while IFS=$'\t' read -r bm pred w m n sum obs; do
  printf '  %-9s %-11s %-9s %-11s %-6s %s\n' \
    "$bm" "$( [[ "$pred" == yes ]] && echo '> 0' || echo '== 0' )" "$w" "$m" "$n" "$obs"
done < "$SUMMARY"
echo "  PTX checksums: $(cut -f6 "$SUMMARY" | tr '\n' ' ')"

# G3: did the arms compile DIFFERENT kernels. Identical PTX across two tile
# settings means the override did not reach the compile, which is the same
# failure the crossing sweep's gate 0 counts Triton artefacts for, and it looks
# from the outside exactly like a tidy null result.
distinct=$(cut -f6 "$SUMMARY" | sort -u | wc -l | tr -d ' ')
narms=${#ARMS[@]}
# G4: wgmma present in exactly the arms the predicate names.
g4_fail=0
while IFS=$'\t' read -r bm pred w m n sum obs; do
  if [[ "$pred" == yes ]]; then (( w > 0 )) || g4_fail=$((g4_fail + 1))
  else (( w == 0 )) || g4_fail=$((g4_fail + 1)); fi
done < "$SUMMARY"

echo
echo "=== gates ==="
score_forced_gates

echo
if (( g1_fail || g2_fail || distinct != narms )); then
  echo "READING IT. A validity gate failed. The census above may not be quoted,"
  echo "and STUDY.md item 3's loose end stays open."
elif (( g4_fail == 0 )); then
  echo "READING IT. At a FIXED token count of $TOKENS, with num_warps pinned at"
  echo "$NUM_WARPS and every other tile knob equal, the warpgroup instruction"
  echo "appears exactly where BLOCK_M % 64 == 0 and nowhere else. THE INSTRUCTION"
  echo "IS SELECTED BY THE TILE, not by the batch size and not by the warp count."
  echo "That closes the loose end on STUDY.md item 3: the 2026-08-27 census saw"
  echo "the same switch through the config ladder, where the batch, the tile and"
  echo "num_warps all moved together."
else
  echo "READING IT. The instruction did not follow the tile in $g4_fail arm(s)"
  echo "while every validity gate passed, so this is a measurement and not a"
  echo "broken run. Triton's warpgroup predicate as this study reads it does not"
  echo "describe this build; quote the table, not the predicate."
fi

echo
echo "  PTX kept under $DUMP_DIR/bm<N>/, run logs beside it, summary at $SUMMARY."
echo "  *.ptx is gitignored at any depth on purpose, so the raw dumps leave a pod"
echo "  as a tarball and the table above leaves as text."

# The `status` variable this used to set by hand is gone. It read 1 on a
# VALIDITY failure -- CLAIM_FAIL, "measured, valid, and the world disagreed" --
# beside its own text saying the census may not be quoted, and the session
# latched the row on it. `finish` derives the code from the gates that printed.
finish
