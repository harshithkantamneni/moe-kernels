#!/bin/bash
# The alpha(G) matrix session: the chain that produces the re-read-fraction
# table the analytical model needs, with both ratio arms held OFF the power
# cap, every geometry's clock elasticity measured beside it, and every G run
# as a seed triple scored jointly.
#
#   bash scripts/alpha_g_chain.sh --dry-run          # plan and price, nothing measured
#   bash scripts/alpha_g_chain.sh                    # a new chain session on this card
#   bash scripts/alpha_g_chain.sh --new              # a new one beside an existing one, on purpose
#   bash scripts/alpha_g_chain.sh --resume           # continue the newest one holding CHAIN.tsv
#   SESSION=<dir> bash scripts/alpha_g_chain.sh      # continue a named one
#   bash scripts/alpha_g_chain.sh --resume --past-gpu-tests   # go on past a red test_gpu.py, recorded
#   bash scripts/alpha_g_chain.sh --resume --past-v8          # go on past the pilot's V8, recorded
#   END_SUITE=skip bash scripts/alpha_g_chain.sh --resume     # the end suite as a SKIPPED row
#
# ON THE POD, FIRST. The volume's clone carries the last session's ruler yaml
# modified, and git will not switch branches over it, even when it is
# byte-identical to the committed one:
#   git -C /workspace/moe-kernels checkout -- moe/bench/hardware/measured_nvidia_h200.yaml
#   git -C /workspace/moe-kernels fetch origin
#   git -C /workspace/moe-kernels checkout -B r3-align origin/r3-align
#   git -C /workspace/moe-kernels log -1 --format=%h   # must print the head that was pushed
# Then launch it DETACHED, so a dropped ssh session does not take a run of
# four hours and more with it, and watch the log and the ledger:
#   cd /workspace/moe-kernels && nohup setsid bash scripts/alpha_g_chain.sh \
#       > /workspace/alpha_g_chain.out 2>&1 < /dev/null &
#   tail -f /workspace/alpha_g_chain.out       # and $SESSION/CHAIN.tsv, one row per step
# The chain's calibrate re-dirties that yaml: --publish writes this card's
# ruler into the tracked file, every row after it carries git_dirty, and the
# yaml is committed WITH the results. The exfil line printed at the end
# carries it and calibrate's own run directory.
#
# WHICH SESSION. A bare run REFUSES when a chain session for this card already
# holds CHAIN.tsv: every R1 and R3 run id carries the session's tag, so a
# fresh tag re-measures every arm that session finished. --resume takes the
# newest directory holding CHAIN.tsv, never a dry run's (CHAIN-dryrun.tsv
# only), and --new opens a fresh one on purpose. --dry-run always plans into a
# fresh directory of its own, and with --resume or SESSION= it is REFUSED: it
# would overwrite that session's step logs, and the driver's thermal.log, with
# plan pages. A measuring run is REFUSED on a box whose card cannot be named
# (the probe's reason is printed); it holds a lock on the session for the
# whole run (a second chain on it is refused; a held flock, the pod's, is never
# taken over, since its holder may be an arm a killed chain left running, and
# only the mkdir fallback's --resume takes over a lock whose pid is dead or
# whose host differs); and it records the card's UUID in
# $SESSION/DEVICE, refusing a resume on another card: R1's run id carries no
# UUID, so its resume would pool two cards' cells into one elasticity.
#
# WHAT IT RUNS, IN ORDER, and why that order:
#   preflight      the two arms' --self-test on the box's interpreter: the
#                  scorers are proven before a cent is spent. Each must be
#                  DONE; an INVALID self-test is a scorer that failed its own
#                  planted world, and it STOPS the chain.
#   preconditions  scripts/h200_gaps_session.sh --only thermal,calibrate,pin_probe-n64-g1
#                  in THIS session directory: card healthy, ruler measured on
#                  this card. The chain STOPS when the driver exits 2 (its
#                  thermal, calibration or reference-grade gate refused), and
#                  unless thermal and calibrate are DONE in its ARMS.tsv. The
#                  step latches only on the driver's exit 0, so a row the
#                  driver still owes is re-attempted on --resume.
#                  pin_probe-n64-g1 RUNS FOR THE RECORD AND IS NOT GATED: it
#                  asks whether MOE_FORCE_TILE reaches the kernel, and neither
#                  arm below reads MOE_FORCE_TILE. Both pin through vLLM's
#                  override_config.
#   gpu-tests      tests/test_gpu.py on this card, from PY_BASE (the venv
#                  WITHOUT vLLM): the timing, clock and graph primitives both
#                  arms stand on, on the card that will time them, in minutes.
#                  Not green STOPS the chain, and an exit 0 in which no test
#                  passed is not green (off a card every one of them skips);
#                  --past-gpu-tests goes on and writes that decision to the
#                  ledger.
#   r3-g<G>-s0     scripts/private_weight_reference.py at --duty 0.25, every G
#                  at seed 0 first, so the ratio arm's never-run paths (the V8
#                  alignment probe under the graph, the duty timer) meet the
#                  card minutes in, not an hour in. At 0.25 session 4's clock
#                  arm sat flat at the ceiling; at 0.5 it still tracked board
#                  power. The chain STOPS after r3-g1-s0 when its V8 is not
#                  PASS, and when it wrote no report.json: V8 describes the
#                  instrument, not G, so every later page would repeat it;
#                  --past-v8 goes on and writes that decision to the ledger.
#   r1-g<G>        scripts/clock_elasticity.py at each G, three cap-binding
#                  duty states (1.0 0.7 0.5), the per-M-tile elasticity gated:
#                  the REGIME word for that G, below.
#   r3-g<G>-s1,s2  seeds 1 then 2 at every G, each later seed scored WITH the
#                  earlier ones of its G through --replicate-of (DESIGN
#                  DECISION 14). The R1 block sits between seed 0 and seed 1,
#                  so a G's seeds are an hour or more apart and the envelope
#                  sees the drift it exists to capture. A G whose seed-0 page
#                  read V7 not PASS gets a SKIPPED row (not latched) for seeds
#                  1 and 2, which are not run: a clock split between the two
#                  ratio arms is the power state, and two more seeds would
#                  buy the same split.
#   suite          the whole suite, uncapped, from PY_BASE, AFTER every arm:
#                  a record of this box that gates nothing, its row in the
#                  ledger. END_SUITE=skip writes a SKIPPED row instead.
#                  Both pytest steps run with this chain's own knobs (SESSION,
#                  END_SUITE, G_LADDER and the rest of CHAIN_KNOBS) removed
#                  from their environment: the suite's tests spawn this chain.
#
# THE REGIME WORD, per G, off R1's interval through the arm's own band_of:
#   RAW-STANDS        wholly below 0.25: the per-tile cost is a traffic
#                     quantity, and the ratio beside it is a re-read fraction.
#   UNREGISTERED-GAP  wholly inside [0.25, 0.40]: NEITHER registered
#                     consequence is licensed; the ratio is quoted with the
#                     interval and no word.
#   CLOCK-CARRIES     wholly above 0.40: the ratio is a time ratio, a blend of
#                     traffic and clock, and the table says so.
#   STRADDLES         the interval crosses an edge: no word.
#   withheld:<EXIT>   R1's page exited INVALID, REFUSED, ERROR or unscored: no
#                     word is read off a page its own gates did not stand behind.
#   unmeasured        no R1 report for that G on disk yet.
# THE WORD IS A SECANT, between the capped clock at duty 1.0 and the clock at
# 0.7 and 0.5; R3 runs at 0.25, at the ceiling. For a per-tile cost A + B/f
# the local elasticity falls as f rises, so RAW-STANDS carries over to R3's
# operating point and CLOCK-CARRIES is only an upper bound there.
#
# WHAT IT LEAVES: $SESSION/CHAIN.tsv (one row per step: state, rc, seconds,
# log, note, with the driver's second opinion taken off the RESULT lines);
# $SESSION/PAIRS.tsv, one row per ratio run, REBUILT from the reports on disk
# at the end of every pass and before every STOP, so a resume never
# duplicates a row and an R1 finished on a later pass is joined: G, seed,
# ratio, interval, exit word, duty, run id; the joint reading over the seeds
# so far, off that run's replicates block (n, spread, sd, envelope, verdict;
# `none` for a run scored alone); each arm's median clock over the ladder and
# the count of LEVEL LOW (arm, tread) cells; R1's eta, interval, word and exit.
# $SESSION/PAIRS-fixed.tsv holds the coordinates every row shares (model,
# tile, pinned config, treads, repeats, duty) and where each was read.
# $SESSION/DEVICE, logs under $SESSION/chain-logs/, and the driver's own
# ARMS.tsv beside them. --resume skips every step whose newest row is latched
# (DONE, CLAIM_FAIL, INVALID) and re-runs REFUSED, ERROR, UNKNOWN and SKIPPED
# ones.
#
# THE THREE HABITS THIS REPOSITORY HAS BEEN BURNED BY, and how this file
# avoids them: no `set -e` (a failed arm is a ledger row, not the end of a
# rented session); every measured command's exit code is captured with
# `|| rc=$?` and never through a pipeline; nothing here starts a background
# job, so there is no PID variable.
set -uo pipefail

# >>> LIFTABLE
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY_BASE="${PY_BASE:-/workspace/venvs/base/bin/python}"
PY_VLLM="${PY_VLLM:-/workspace/venvs/vllm/bin/python}"
[[ -x "$PY_BASE" ]] || PY_BASE="$REPO/.venv/bin/python"
[[ -x "$PY_BASE" ]] || PY_BASE="$(command -v python3 || true)"
[[ -x "$PY_VLLM" ]] || PY_VLLM="$PY_BASE"
HELPERS="$REPO/scripts/alpha_g_chain_helpers.py"

#: The ladder. 1, 16 and 64 are the swizzles vLLM's tuned config for this model
#: on this card actually uses; 4 is the mid-doubling. G=32 is not booked: with
#: at most 8 x 6 real M-tiles per launch it is the same ordering as 64 at every
#: tread but two. Three seeds so a spread is a spread and not a range.
G_LADDER="${G_LADDER:-1 4 16 64}"
SEEDS="${SEEDS:-0 1 2}"
#: The ratio arms off the cap (DESIGN DECISION 15), at 0.25: on session 4's
#: clock arm, duty 0.5 still tracked board power (-1.09 MHz/W over 1882-1965
#: MHz, DRIFT failing in one cell in six), and duty 0.25 sat flat at 1965 MHz
#: with no drift. R3's own --duty default stays 1.0: it is a design key, and a
#: default that moved would refuse --replicate-of against session 4's runs.
R3_DUTY="${R3_DUTY:-0.25}"
R3_TREADS=6
R3_REPEATS=9
#: The elasticity arm's states: cap-binding ones. Session 4 showed duty 0.5,
#: 0.25 and 0.10 were one clock cluster, so two of its four states bought
#: nothing and cost 14/17 of the wall; three states is MIN_STATES.
R1_DUTY="${R1_DUTY:-1.0 0.7 0.5}"
R1_TREADS=8
R1_REPEATS=13
RATE_USD_H="${RATE_USD_H:-4.59}"
#: The price per collected test ON A POD, not on a laptop: session 4's capped
#: run (results/published/2026-09-21-nvidia_h200-session4/session/
#: chain-logs/session4-pytest.log) did 2242 tests in 1465 s, 2.5x the laptop's
#: rate, off the network volume. A dry run multiplies it by each step's count.
SUITE_S_PER_TEST="${SUITE_S_PER_TEST:-0.66}"
#: A hung suite (the volume's MooseFS has hung before) must not eat the
#: booking: about 1.7x the priced run, then pytest is interrupted and its
#: tally of what ran is still printed. test_gpu.py gets its own, smaller cap.
SUITE_TIMEOUT_S="${SUITE_TIMEOUT_S:-5400}"
GPU_TESTS_TIMEOUT_S="${GPU_TESTS_TIMEOUT_S:-900}"
#: `skip` records the end suite as a SKIPPED row and does not run it: a
#: resume that owes one arm need not buy the whole suite again.
END_SUITE="${END_SUITE:-run}"
#: The seed a G starts at, and the step after which V8 is read.
FIRST_SEED="${SEEDS%% *}"
FIRST_G="${G_LADDER%% *}"
PILOT="r3-g$FIRST_G-s$FIRST_SEED"

#: exit_codes.py's table, mirrored for the ledger word; compared with the
#: module by the driver's tests and by this file's.
state_word() {
  case "$1" in
    0) echo DONE ;; 1) echo CLAIM_FAIL ;; 2) echo REFUSED ;; 3) echo INVALID ;;
    4) echo ERROR ;; *) echo UNKNOWN ;;
  esac
}

#: The SECOND OPINION, the driver's rule: a page's RESULT lines must support
#: the integer the process returned, or the row is UNKNOWN and not latched.
#: $1 rc, $2 what the log implies (a code, NONE, or UNREADABLE). Prints
#: STATE<TAB>note.
state_for() {
  local rc="$1" implied="$2" word
  word="$(state_word "$rc")"
  case "$implied" in
    UNREADABLE*) printf 'UNKNOWN\tSECOND OPINION UNAVAILABLE: exit %s, the log could not be classified\n' "$rc"; return 0 ;;
    NONE)
      case "$rc" in
        0) printf 'UNKNOWN\tUNEARNED DONE: exit 0 with no RESULT line; not latched\n' ;;
        2) printf 'REFUSED\t\n' ;;
        3) printf 'UNKNOWN\tUNEARNED INVALID: exit 3 with no RESULT line; not latched\n' ;;
        4) printf 'ERROR\tcrash before any gate; re-runs on --resume\n' ;;
        *) printf 'UNKNOWN\texit %s with no RESULT line; not latched\n' "$rc" ;;
      esac
      return 0 ;;
  esac
  if [[ "$implied" == "$rc" ]]; then
    printf '%s\tlog agrees: RESULT lines imply %s\n' "$word" "$implied"
  else
    printf 'UNKNOWN\tDEFECT: page implies %s (%s), exit %s; not latched\n' \
      "$implied" "$(state_word "$implied")" "$rc"
  fi
}

#: A step's newest ledger word, or nothing when it has no row. $1 the step
#: (or the driver's arm), $2 the ledger (CHAIN.tsv or the driver's ARMS.tsv).
newest_state() {
  local name="$1" ledger="$2"
  [[ -f "$ledger" ]] || return 0
  awk -F'\t' -v n="$name" '$1==n {s=$2} END {print s}' "$ledger"
}

#: Is a step's newest ledger row a latched state: a RESULT, which --resume
#: does not buy again. NOT the same question as "is it DONE": an INVALID or a
#: CLAIM_FAIL row is latched too, so no gate asks this one.
latched() {
  case "$(newest_state "$1" "$2")" in DONE|CLAIM_FAIL|INVALID) return 0 ;; *) return 1 ;; esac
}

#: The driver's dirty count, verbatim: rc handled apart from the pipeline.
dirty_count() {
  local out
  out="$(git -C "$REPO" status --porcelain --untracked-files=all 2>/dev/null)" || { echo "-"; return 0; }
  printf '%s\n' "$out" | grep -c . || true
}

#: Run a command from the repository root: the suite's rootdir and testpaths.
in_repo() { ( cd "$REPO" && "$@" ); }

#: pytest's closing tally off a log ("4772 passed, 57 skipped in ..."), or a
#: word saying there was none. One line, no tabs: it goes in a ledger note.
suite_tally() {
  local line
  line="$(grep -E '[0-9]+ (passed|failed|errors?|skipped|tests? collected)' "$1" 2>/dev/null | tail -1)"
  line="$(printf '%s' "$line" | sed -E 's/^=+ //; s/ =+$//' | tr '\t' ' ')"
  printf '%s\n' "${line:-no tally in the log}"
}

#: How many tests a tally says passed (or were collected, $2 = collected).
tally_count() {
  local n
  n="$(printf '%s\n' "$1" | grep -oE "[0-9]+ ${2:-passed}" | grep -oE '^[0-9]+' | head -1)"
  printf '%s\n' "${n:-0}"
}

#: A row for a step this pass decided not to run, and why. SKIPPED is not a
#: latched word: the next pass asks again.
skip_row() {
  printf '%s\tSKIPPED\t-\t0\t%s\t-\t%s\n' "$1" "$(dirty_count)" "$2" >> "$LEDGER"
  printf '%-16s %-10s %s\n' "$1" SKIPPED "$2"
}

#: Run one step and write its row. $1 the opinion, $2 name, $3 log, $4.. the
#: command. The command's exit code is captured directly: never `cmd | tee`,
#: which reports tee's. Returns the command's rc. The opinion is an ARGUMENT,
#: never an environment variable: `OPINION=suite f` on a function exports it
#: to every child of f, and the end suite's own tests then scored their rows
#: with it. The state is decided by the opinion:
#:   page    (run_step) the arm's RESULT lines must support its exit code.
#:   driver  the preconditions step, the driver sequencing arms, which prints
#:           no RESULT line of its own: exit 0 is DONE, exit 2 is REFUSED (one
#:           of its gates refused the card or the ruler), and ANY OTHER CODE
#:           is UNKNOWN, not latched, because the driver exits 3 when a row is
#:           still owed and 4 when an arm crashed, and --resume re-attempts
#:           both only if this step runs again. Its per-arm second opinion is
#:           its own ARMS.tsv, which the chain reads back.
#:   suite   pytest, which prints no RESULT line either: exit 0 with at least
#:           one test passed is DONE; exit 0 with none passed is UNKNOWN (off
#:           a card every GPU test skips, and that is not green); any other
#:           code is ERROR. Never a latched word unless DONE, so a red run
#:           re-runs on --resume. The note is pytest's tally.
#:   collect a dry run's `pytest --collect-only`: DONE when it collected.
run_step_as() {
  local opinion="$1" name="$2" log="$3"; shift 3
  local rc=0 t0 secs implied state note tally
  t0="$(date +%s)"
  "$@" > "$log" 2>&1 || rc=$?
  secs="$(( $(date +%s) - t0 ))"
  case "$opinion" in
    driver)
      case "$rc" in
        0) state=DONE; note="the driver's own ARMS.tsv carries each arm's second opinion" ;;
        2) state=REFUSED; note="the driver refused: its thermal, calibration or reference-grade gate spoke" ;;
        *) state=UNKNOWN
           note="driver exit $rc ($(state_word "$rc")): a row in ARMS.tsv is owed or crashed; not latched, --resume re-runs the driver" ;;
      esac ;;
    suite)
      tally="$(suite_tally "$log")"
      if (( rc == 0 )) && (( $(tally_count "$tally") > 0 )); then
        state=DONE; note="pytest exit 0: $tally"
      elif (( rc == 0 )); then
        state=UNKNOWN; note="UNEARNED DONE: pytest exit 0 but no test passed ($tally); not latched"
      elif (( rc == 124 )); then
        state=ERROR; note="pytest TIMED OUT (exit 124): $tally"
      else
        state=ERROR; note="pytest exit $rc: $tally"
      fi ;;
    collect)
      tally="$(suite_tally "$log")"
      if (( rc == 0 )) && (( $(tally_count "$tally" "tests? collected") > 0 )); then
        state=DONE
      else
        state=ERROR
      fi
      note="pytest exit $rc: $tally" ;;
    *)
      implied="$("$PY_BASE" "$HELPERS" verdict "$log" 2>/dev/null)" || implied="UNREADABLE"
      IFS=$'\t' read -r state note < <(state_for "$rc" "$implied") ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$state" "$rc" "$secs" \
    "$(dirty_count)" "$log" "$note" >> "$LEDGER"
  printf '%-16s %-10s rc=%s %5ss  %s\n' "$name" "$state" "$rc" "$secs" "$note"
  return "$rc"
}

#: An arm's step, its page's RESULT lines the second opinion. $1 name, $2 log,
#: $3.. the command.
run_step() { run_step_as page "$@"; }

#: Every variable this chain reads from its environment. A pytest step runs
#: WITHOUT them: the suite's tests spawn this chain and the driver with
#: os.environ merged in, so an operator's SESSION=<dir> or END_SUITE=skip
#: reached those children and steered them into the real session.
#: MOE_RESULTS_DIR stays (tests/conftest.py sandboxes it), and so do PY_BASE
#: and PY_VLLM (tests/_hermetic.py replaces them).
CHAIN_KNOBS="REPO SESSION SESSION_ROOT RESULTS_ROOT WORKSPACE G_LADDER SEEDS R3_DUTY R1_DUTY RATE_USD_H SUITE_S_PER_TEST SUITE_TIMEOUT_S GPU_TESTS_TIMEOUT_S END_SUITE LOCK_TOOL CAPABILITY"
without_knobs() {
  local -a unset_args=()
  local k
  for k in $CHAIN_KNOBS; do unset_args+=(-u "$k"); done
  env ${unset_args[@]+"${unset_args[@]}"} "$@"
}

#: The tests' interpreter must NOT import vLLM: the tests plant every refusal
#: door, and from a venv with vLLM an unplanted bare --run would MEASURE
#: (pod_session.sh P11c). Returns 0 when the interpreter is safe.
suite_interpreter_ok() { ! "$1" -c "import vllm" >/dev/null 2>&1; }

#: A pytest step: the interpreter refusal, the timeout wrapper, the run.
#: $1 the step's name, $2 the timeout in seconds, $3.. pytest's arguments.
pytest_step() {
  local name="$1" secs="$2"; shift 2
  local -a tmo=()
  if ! suite_interpreter_ok "$PY_BASE"; then
    printf '%s\tREFUSED\t2\t0\t%s\t-\t%s\n' "$name" "$(dirty_count)" \
      "the interpreter $PY_BASE imports vllm; an unplanted --run would MEASURE" >> "$LEDGER"
    printf '%-16s %-10s %s\n' "$name" REFUSED "$PY_BASE imports vllm; set PY_BASE to the venv without it"
    return 0
  fi
  command -v timeout >/dev/null 2>&1 && tmo=(timeout --signal=INT --kill-after=60 "$secs")
  run_step_as suite "$name" "$LOGS/$name.log" in_repo without_knobs ${tmo[@]+"${tmo[@]}"} \
    "$PY_BASE" -m pytest "$@" || true
}

#: Go on past a gate on its flag, and write that decision to the ledger as its
#: own OVERRIDDEN row, so it travels with the results. $1 the row's name, $2
#: the flag, $3 what was gone past. A flag counts for the pass it is given on.
override_row() {
  printf '%s\tOVERRIDDEN\t-\t0\t%s\t-\t%s\n' "$1" "$(dirty_count)" \
    "the operator went on with $2; $3" >> "$LEDGER"
  echo "  $2: going on past $3; the ledger says so"
}

#: The preflight gate: both self-tests DONE, which is not the same as latched.
#: Returns 0 to go on, 3 to stop.
preflight_gate() {
  local step st
  for step in preflight-r1 preflight-r3; do
    st="$(newest_state "$step" "$LEDGER")"
    if [[ "$st" != DONE ]]; then
      echo "STOP: $step is ${st:-absent}, not DONE; the scorer is not proven on this interpreter."
      echo "  Read $LOGS/$step.log: an INVALID or CLAIM_FAIL self-test is a scorer that"
      echo "  failed its own planted world, and every page it scores would carry that."
      return 3
    fi
  done
  return 0
}

#: The preconditions gate: the driver did not refuse, and thermal and calibrate
#: are DONE in its ARMS.tsv. $1 the driver's ARMS.tsv. pin_probe-n64-g1 is not
#: read: neither arm reads MOE_FORCE_TILE. Returns 0 to go on, 3 to stop.
preconditions_gate() {
  local arms="$1" need st why
  if [[ "$(newest_state preconditions "$LEDGER")" == REFUSED ]]; then
    why="$(grep -m1 '^REFUSED' "$LOGS/preconditions.log" 2>/dev/null)"
    echo "STOP: the driver REFUSED this card or its ruler (exit 2): ${why:-read the log}"
    echo "  Read $LOGS/preconditions.log. No R1 or R3 cell is scored against a ruler"
    echo "  arm 0 did not stand behind."
    return 3
  fi
  for need in thermal calibrate; do
    st="$(newest_state "$need" "$arms")"
    if [[ "$st" != DONE ]]; then
      echo "STOP: the driver's $need row is ${st:-absent}, not DONE, in $arms;"
      echo "  nothing below can be scored. Read $LOGS/preconditions.log."
      return 3
    fi
  done
  return 0
}

#: The gpu-tests gate: tests/test_gpu.py's newest row DONE, or --past-gpu-tests.
#: $1 1 when the flag was given. Returns 0 to go on, 3 to stop.
gpu_tests_gate() {
  local past="$1" newest
  newest="$(newest_state gpu-tests "$LEDGER")"
  [[ "$newest" == DONE ]] && return 0
  if (( past )); then
    override_row gpu-tests-override --past-gpu-tests "tests/test_gpu.py's newest row: ${newest:-never ran}"
    return 0
  fi
  echo "STOP: tests/test_gpu.py is ${newest:-never run}, not green, on this card. Read"
  echo "  $LOGS/gpu-tests.log (-rfE lists every failure and error). If none of it bears"
  echo "  on the arms, go on with --resume --past-gpu-tests, which the ledger records."
  return 3
}

#: The run's report.json, off the run id its plan page printed, or nothing.
#: $1 the step's log, $2 the experiment (the ratio arm's by default).
report_of() {
  local exp="${2:-private_weight_reference}" rid
  rid="$("$PY_BASE" "$HELPERS" run-id "$1" "$exp" 2>/dev/null)" || rid=""
  [[ -n "$rid" && -f "$RESULTS/$exp/$rid/report.json" ]] && echo "$RESULTS/$exp/$rid/report.json"
  return 0
}

#: The pilot gate: the first ratio run's V8, off its report.json. $1 1 when
#: --past-v8 was given. Returns 0 to go on, 3 to stop.
v8_gate() {
  local past="$1" rep v8
  rep="$(report_of "$LOGS/$PILOT.log")"
  if [[ -z "$rep" ]]; then
    v8="unread: no report.json"
  else
    v8="$("$PY_BASE" "$HELPERS" gate "$rep" V8 2>/dev/null)" || v8="unreadable"
  fi
  [[ "$v8" == PASS ]] && return 0
  if (( past )); then
    override_row v8-override --past-v8 "$PILOT's V8 $v8"
    return 0
  fi
  echo "STOP: $PILOT's V8 is $v8, not PASS. V8 is the alignment probe under the"
  echo "  graph: it describes the instrument, not G, so every later ratio page would"
  echo "  repeat it. Read $LOGS/$PILOT.log. Going on anyway is --resume --past-v8,"
  echo "  which the ledger records."
  return 3
}

#: Seed 0's V7 verdict for a G, or nothing when seed 0 left no report.
seed0_v7() {
  local rep
  rep="$(report_of "$LOGS/r3-g$1-s$FIRST_SEED.log")"
  [[ -n "$rep" ]] || return 0
  "$PY_BASE" "$HELPERS" gate "$rep" V7 2>/dev/null || echo unreadable
}

#: REWRITE PAIRS.tsv and PAIRS-fixed.tsv from the reports on disk. Idempotent.
rebuild_pairs() {
  local out rc=0
  out="$("$PY_BASE" "$HELPERS" pairs-table "$SESSION" "$RESULTS" "$G_LADDER" "$SEEDS" \
    "model=--model mixtral-8x7b" "block_m=--block-m 32" \
    "r3_treads=--treads $R3_TREADS" "r3_repeats=--repeats $R3_REPEATS" "r3_duty=--duty $R3_DUTY" \
    "r1_treads=--treads $R1_TREADS" "r1_repeats=--repeats $R1_REPEATS" "r1_duty=--duty $R1_DUTY" \
    2>&1)" || rc=$?
  if (( rc == 0 )); then
    echo "pairs     $SESSION/PAIRS.tsv ($out, rebuilt from the reports), fixed coordinates in PAIRS-fixed.tsv"
  else
    echo "pairs     NOT rebuilt (exit $rc): $out"
  fi
}

#: THE DIRECTORY TO RESUME: the newest chain session for this card holding a
#: measuring ledger, or rc 1. By name is by time (the names carry a UTC
#: stamp). A dry run's directory holds CHAIN-dryrun.tsv only and latches
#: nothing: the driver's latest_session, lifted, which the chain lacked.
latest_chain_session() {
  local root="$1" prefix="$2" d found=""
  for d in "$root/$prefix"*/; do
    [[ -f "$d/CHAIN.tsv" ]] && found="${d%/}"
  done
  [[ -n "$found" ]] || return 1
  printf '%s\n' "$found"
}

#: WHICH DIRECTORY THIS PASS RUNS IN, the driver's session_choice with the
#: chain's two extra rules. $1 DRY, $2 --resume, $3 --new, $4 SESSION= as
#: given, $5 the root, $6 the per-card prefix. Prints HOW<TAB>WHAT; rc 1 on:
#:   REFUSED        contradictory flags; a dry run pointed at a real session
#:                  (it would overwrite its logs with plan pages); nothing to
#:                  resume (the directories found are listed)
#:   LATEST_EXISTS  a measuring run without --new found a chain session
#: and rc 0 on NAMED, RESUMED and NEW.
session_choice() {
  local dry="$1" resume="$2" new="$3" explicit="$4" root="$5" prefix="$6" latest="" d others=""
  if (( resume )) && (( new )); then
    printf 'REFUSED\t--resume and --new contradict each other. Give one.\n'; return 1
  fi
  if (( dry )) && { (( resume )) || [[ -n "$explicit" ]]; }; then
    printf 'REFUSED\t--dry-run plans into a fresh directory of its own. With --resume or SESSION= it would overwrite that session'"'"'s chain-logs, and the driver'"'"'s logs/thermal.log, with plan pages. Run the dry run alone.\n'
    return 1
  fi
  if [[ -n "$explicit" ]]; then
    if (( resume )) || (( new )); then
      printf 'REFUSED\tSESSION=%s names the directory outright, and --resume / --new choose one under %s. Give one or the other.\n' "$explicit" "$root"
      return 1
    fi
    printf 'NAMED\t%s\n' "$explicit"; return 0
  fi
  latest="$(latest_chain_session "$root" "$prefix")" || latest=""
  if (( resume )); then
    if [[ -z "$latest" ]]; then
      for d in "$root/$prefix"*/; do [[ -d "$d" ]] && others+=" ${d%/}"; done
      printf 'REFUSED\t--resume found no session holding CHAIN.tsv under %s/%s*; the directories there (dry runs, or never measured):%s\n' \
        "$root" "$prefix" "${others:- none}"
      return 1
    fi
    printf 'RESUMED\t%s\n' "$latest"; return 0
  fi
  if (( dry == 0 )) && (( new == 0 )) && [[ -n "$latest" ]]; then
    printf 'LATEST_EXISTS\t%s\n' "$latest"; return 1
  fi
  printf 'NEW\t%s/%s%s\n' "$root" "$prefix" "$(date -u +%Y%m%dT%H%M%SZ)"
}

#: THE CARD THIS SESSION WAS MEASURED ON. $1 the session, $2 this card's
#: identity. Writes DEVICE on first use; returns 2, saying why, when the
#: session was measured on another card, when its ledger has rows but no
#: DEVICE (the card those rows came from is unknown), or when this card's
#: UUID cannot be read. R3 guards its own directories on the UUID; R1 keys
#: its resume on the card NAME, which every H200 shares, so without this a
#: resume on a replacement pod would pool two governors into one elasticity.
device_check() {
  local file="$1/DEVICE" ident="$2" recorded rows
  if [[ -z "$ident" ]]; then
    echo "REFUSED: this card's UUID could not be read (torch and nvidia-smi gave none), so"
    echo "  a resume here could not be shown to be on the card that began it."
    return 2
  fi
  if [[ -f "$file" ]]; then
    recorded="$(tr -d '[:space:]' < "$file")"
    [[ "$recorded" == "$ident" ]] && return 0
    echo "REFUSED: $1 was measured on the card $recorded, and this card is $ident."
    echo "  Every R1 run id resumes by card NAME, so its ladder would take this card's cells"
    echo "  into the other card's fit. A fresh session on this card, on purpose:"
    echo "      bash scripts/alpha_g_chain.sh --new"
    return 2
  fi
  rows="$(awk 'NR > 1' "$1/CHAIN.tsv" 2>/dev/null | grep -c . || true)"
  if (( ${rows:-0} > 0 )); then
    echo "REFUSED: $1 holds $rows ledger row(s) and no DEVICE file, so the card they were"
    echo "  measured on is unknown. A fresh session: bash scripts/alpha_g_chain.sh --new"
    return 2
  fi
  printf '%s\n' "$ident" > "$file"
}

#: Is a recorded mkdir-lock owner ("pid host") gone: its host differs from
#: this one (the pod it ran on is not this pod), or its pid is dead here. $1
#: the owner, $2 this host. Returns 0 when stale. The flock branch never asks:
#: a dead recorded pid says nothing about the arm that pid left running.
lock_is_stale() {
  local pid="${1%% *}" host="${1#* }"
  [[ -n "$1" && "$1" == *" "* ]] || return 0
  [[ "$host" != "$2" ]] && return 0
  kill -0 "$pid" 2>/dev/null && return 1
  return 0
}

#: THE LOCK, held for the whole measuring run. flock where the box has it,
#: else an atomic mkdir; either way the holder is recorded as "pid host". $1
#: the session, $2 1 when this pass continues an existing session. Returns 2,
#: saying why, when refused; sets LOCK_DIR to what release_lock removes
#: (nothing for flock, which the kernel releases when this shell and every
#: child holding fd 9 are gone).
#:   flock  A held lock is NEVER taken over, whatever pid it records: the
#:          kernel's answer is that a live process holds the file, and when
#:          the recorded chain is dead that process is the arm it left running
#:          (`pkill -f alpha_g_chain.sh` kills the shell, not its python),
#:          which inherited fd 9 and is still timing the card. A resume that
#:          took it over would run the same step, same run id, on the same
#:          card beside it.
#:   mkdir  The fallback where there is no flock (a laptop). It cannot see an
#:          orphaned arm, so a resuming pass ($2 1) takes over a lock whose
#:          pid is dead or whose host differs.
LOCK_TOOL="${LOCK_TOOL:-}"
[[ -n "$LOCK_TOOL" ]] || { command -v flock >/dev/null 2>&1 && LOCK_TOOL=flock || LOCK_TOOL=mkdir; }
LOCK_DIR=""
LOCK_OWNER=""
chain_lock() {
  local dir="$1" resuming="$2" host owner
  host="$(hostname 2>/dev/null || uname -n 2>/dev/null || echo unknown-host)"
  LOCK_OWNER="$$ $host"
  if [[ "$LOCK_TOOL" == flock ]]; then
    local file="$dir/chain.lock"
    exec 9>>"$file" || { echo "REFUSED: cannot open $file"; return 2; }
    if ! flock -n 9; then
      owner="$(cat "$file" 2>/dev/null)"
      exec 9>&-
      echo "REFUSED: another chain holds $file (it recorded ${owner:-no holder}); two chains"
      echo "  on one session would time the card at once and resume the same run ids."
      echo "  The kernel says a live process holds it, whatever pid it records: a chain, or"
      echo "  an arm a killed chain left running, still timing the card. A held flock is"
      echo "  never taken over. Find the holder, stop it, then --resume:"
      echo "      fuser -v $file     (or: lsof $file)"
      echo "  If neither names a process here, it is on another pod sharing this volume."
      return 2
    fi
    printf '%s\n' "$LOCK_OWNER" > "$file"
    return 0
  fi
  local ldir="$dir/chain.lock.d"
  if ! mkdir "$ldir" 2>/dev/null; then
    owner="$(cat "$ldir/owner" 2>/dev/null)"
    if (( resuming )) && lock_is_stale "$owner" "$host"; then
      rm -rf "$ldir"
      mkdir "$ldir" 2>/dev/null || { echo "REFUSED: lost the race for $ldir"; return 2; }
      echo "  took over a stale lock (held by ${owner:-nobody recorded})"
    else
      echo "REFUSED: another chain holds $ldir (${owner:-no holder recorded}); two chains"
      echo "  on one session would time the card at once and resume the same run ids."
      echo "  If no chain is running, a --resume takes a lock whose pid is dead over"
      echo "  (this box has no flock, so an arm a killed chain left running is not seen)."
      return 2
    fi
  fi
  printf '%s\n' "$LOCK_OWNER" > "$ldir/owner"
  LOCK_DIR="$ldir"
  return 0
}

#: The EXIT trap's half of the mkdir lock: removed only while it still names
#: this shell. A pass that took the lock over from a holder it judged stale
#: must not lose it when that holder, alive after all on another host, exits.
release_lock() {
  [[ -n "$LOCK_DIR" ]] || return 0
  [[ "$(cat "$LOCK_DIR/owner" 2>/dev/null)" == "$LOCK_OWNER" ]] && rm -rf "$LOCK_DIR"
  return 0
}
# <<< LIFTABLE

usage() { sed -n '2,/^set -uo pipefail$/p' "$0" | sed '$d'; }

DRY=0; RESUME=0; NEW=0; PAST_GPU_TESTS=0; PAST_V8=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --resume)  RESUME=1; shift ;;
    --new)     NEW=1; shift ;;
    --past-gpu-tests) PAST_GPU_TESTS=1; shift ;;
    --past-v8) PAST_V8=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "REFUSED: unknown argument $1"; usage; exit 2 ;;
  esac
done

[[ -x "$PY_BASE" ]] || { echo "REFUSED: no usable base interpreter (set PY_BASE=)"; exit 2; }
[[ -f "$HELPERS" ]] || { echo "REFUSED: $HELPERS is missing"; exit 2; }

smi_line() {
  local out
  if command -v nvidia-smi >/dev/null 2>&1; then
    out="$(nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader 2>&1)" \
      || out="nvidia-smi failed: $out"
    printf '%s\n' "$out" | sed 's/^/  nvidia-smi  /'
  else
    echo "  nvidia-smi  not on PATH (name, driver version and power limit unread)"
  fi
}

# THE CARD, as the driver resolves it: through torch on PY_BASE, as a slug,
# with the probe's own reason when it cannot, and its stderr NOT discarded.
CARD=""; CARD_REASON=""
IFS=$'\t' read -r CARD CARD_REASON < <("$PY_BASE" "$HELPERS" card)
CARD="${CARD:-nocard}"
[[ "$CARD" == nocard ]] && CARD_REASON="${CARD_REASON:-the card probe printed nothing}"
if [[ "$CARD" == nocard ]] && ! (( DRY )); then
  echo "REFUSED: no CUDA device this chain can name, and a measuring run measures."
  echo "  Nothing was run and no session directory was opened."
  echo "  the probe ($PY_BASE, through moe.bench.provenance) said: $CARD_REASON"
  smi_line
  echo "  A torch wheel built for a newer CUDA than the host's driver reads as no device:"
  echo "  compare the driver version above with the wheel's. Off GPU and free:"
  echo "      bash scripts/alpha_g_chain.sh --dry-run"
  exit 2
fi

if [[ -d /workspace ]]; then
  SESSION_ROOT="${SESSION_ROOT:-/workspace/session}"
  RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"
  WORKSPACE="${WORKSPACE:-/workspace}"
else
  SESSION_ROOT="${SESSION_ROOT:-$REPO/results/h200_gaps}"
  RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results}"
  WORKSPACE="${WORKSPACE:-$RESULTS_ROOT}"
fi
IFS=$'\t' read -r SESSION_HOW SESSION_WHAT \
  <<< "$(session_choice "$DRY" "$RESUME" "$NEW" "${SESSION:-}" "$SESSION_ROOT" "alpha_g-$CARD-")"
case "$SESSION_HOW" in
  NAMED|RESUMED|NEW) SESSION="$SESSION_WHAT" ;;
  LATEST_EXISTS)
    echo "REFUSED: a chain session for $CARD already exists, with a measuring ledger:"
    echo "  $SESSION_WHAT"
    echo "  A fresh session opens an EMPTY ledger under a new tag, and every R1 and R3"
    echo "  run id carries the tag, so every arm that session finished would be measured"
    echo "  again. Nothing was run. Choose, in words:"
    echo "      bash scripts/alpha_g_chain.sh --resume     # continue it"
    echo "      SESSION=$SESSION_WHAT bash scripts/alpha_g_chain.sh"
    echo "      bash scripts/alpha_g_chain.sh --new        # a fresh session, on purpose"
    exit 2 ;;
  *)
    echo "REFUSED: ${SESSION_WHAT:-session_choice printed nothing this chain can read}"
    exit 2 ;;
esac
CONTINUING=0
[[ "$SESSION_HOW" == NEW ]] || CONTINUING=1
# The driver's rule, applied here BEFORE the driver is reached and before any
# directory is made: an operator-supplied MOE_RESULTS_DIR must carry the card
# in its name. Two pods share a network volume, and one card's run resumes the
# other's directories when the card is not in the path.
if [[ -n "${MOE_RESULTS_DIR:-}" ]]; then
  RESULTS="$MOE_RESULTS_DIR"
  case "$RESULTS" in
    *"$CARD"*) ;;
    *) echo "REFUSED: MOE_RESULTS_DIR=$RESULTS does not contain the card '$CARD'."
       echo "  Two pods share a network volume and one card's run resumes the"
       echo "  other's directories when the card is not in the path. Use"
       echo "  MOE_RESULTS_DIR=$RESULTS/gaps-$CARD or unset it and take the default."
       exit 2 ;;
  esac
else
  RESULTS="$RESULTS_ROOT/gaps-$CARD"
fi
export MOE_RESULTS_DIR="$RESULTS"
TAG="$(basename "$SESSION")"
LOGS="$SESSION/chain-logs"
if (( DRY )); then LEDGER="$SESSION/CHAIN-dryrun.tsv"; else LEDGER="$SESSION/CHAIN.tsv"; fi
mkdir -p "$LOGS" || { echo "REFUSED: cannot create $LOGS"; exit 2; }

if ! (( DRY )); then
  chain_lock "$SESSION" "$CONTINUING" || exit 2
  trap release_lock EXIT
  DEVICE_ID="$("$PY_BASE" "$HELPERS" device)" || DEVICE_ID=""
  device_check "$SESSION" "$DEVICE_ID" || exit 2
fi
[[ -f "$LEDGER" ]] || printf 'step\tstate\trc\tseconds\tdirty\tlog\tnote\n' > "$LEDGER"

echo "alpha(G) chain  $TAG   ($SESSION_HOW)"
echo "  card        $CARD${CARD_REASON:+  -- $CARD_REASON}"
smi_line
(( DRY )) || echo "  device      ${DEVICE_ID:-}   (recorded in $SESSION/DEVICE)"
echo "  session     $SESSION"
echo "  results     $RESULTS   (exported as MOE_RESULTS_DIR to every arm)"
echo "  ledger      $LEDGER"
echo "  ladder      G in {$G_LADDER}, seeds {$SEEDS}, ratio arms at duty $R3_DUTY, elasticity states $R1_DUTY"
echo "  interpreters base $PY_BASE / vllm $PY_VLLM"
(( DRY )) && echo "  DRY RUN: every arm's own --dry-run is run and priced; nothing is measured"
if ! (( DRY )) && [[ -t 1 ]]; then
  echo "  NOTE: attached to a terminal. A dropped ssh session kills this chain and the arm"
  echo "  in flight; the launch line in this file's header detaches it."
fi

#: Every STOP after the session exists leaves the table as the disk has it.
stop_chain() { rebuild_pairs; exit 3; }

# --------------------------------------------------------------------------
# the command lines, in one place each
# --------------------------------------------------------------------------
r1_cmd() {   # $1 G, $2 dry
  local g="$1" dry="$2"
  if (( dry )); then
    echo "$PY_BASE" "$REPO/scripts/clock_elasticity.py" --dry-run
  else
    echo "$PY_VLLM" "$REPO/scripts/clock_elasticity.py"
  fi
  # shellcheck disable=SC2086
  echo --model mixtral-8x7b --dtype bf16 --group-m "$g" --treads "$R1_TREADS" --duty $R1_DUTY \
       --repeats "$R1_REPEATS" --burst-ms 40 --target-ms 200 --trials 3 --warm-ms 200 \
       --settle-seconds 10 --session-tag "$TAG"
}
r3_cmd() {   # $1 G, $2 seed, $3 dry, $4.. replicate reports
  local g="$1" seed="$2" dry="$3"; shift 3
  if (( dry )); then
    echo "$PY_BASE" "$REPO/scripts/private_weight_reference.py" --dry-run \
         --capability "${CAPABILITY:-9.0}" --device-memory-gb 140
  else
    echo "$PY_VLLM" "$REPO/scripts/private_weight_reference.py"
  fi
  echo --model mixtral-8x7b --block-m 32 --treads "$R3_TREADS" --repeats "$R3_REPEATS" \
       --group-m "$g" --duty "$R3_DUTY" --seed "$seed" --session-tag "$TAG"
  if (( $# )); then echo --replicate-of "$@"; fi
}

TOTAL_S=0
GPU_TESTS_S=0; GPU_TESTS_N=""; SUITE_S=0; SUITE_N=""
price() {   # $1 log: add the arm's own dry-run estimate to the total
  local est
  est="$("$PY_BASE" "$HELPERS" estimate "$1" 2>/dev/null)" || est=""
  if [[ -n "$est" ]]; then TOTAL_S=$(( TOTAL_S + est )); echo "    priced ${est} s off its own plan"; else echo "    (no estimate on its plan page)"; fi
}
price_tests() {   # $1 log of a --collect-only; prints "<count> <seconds>"
  local n
  n="$(tally_count "$(suite_tally "$1")" "tests? collected")"
  printf '%s %s\n' "$n" "$(awk -v n="$n" -v r="$SUITE_S_PER_TEST" 'BEGIN {printf "%d", n * r + 0.5}')"
}

#: One ratio run: pairing with the earlier seeds of its G that FORMED a ratio
#: (the rule load_replicates applies), the step, and its console line.
r3_step() {   # $1 G, $2 seed
  local g="$1" seed="$2" step s2 elog rep line
  local -a earlier=() paired=()
  step="r3-g$g-s$seed"
  if latched "$step" "$LEDGER" && ! (( DRY )); then echo "  $step latched, skipped"; return 0; fi
  for s2 in $SEEDS; do
    [[ "$s2" == "$seed" ]] && break
    elog="$LOGS/r3-g$g-s$s2.log"
    [[ -f "$elog" ]] || continue
    rep="$(report_of "$elog")"
    [[ -n "$rep" ]] && earlier+=("$rep")
  done
  if (( ${#earlier[@]} )); then
    while IFS= read -r line; do [[ -n "$line" ]] && paired+=("$line"); done \
      < <("$PY_BASE" "$HELPERS" pairs ${earlier[@]+"${earlier[@]}"} 2>/dev/null)
  fi
  # shellcheck disable=SC2046
  run_step "$step" "$LOGS/$step.log" $(r3_cmd "$g" "$seed" "$DRY" ${paired[@]+"${paired[@]}"}) || true
  if (( DRY )); then
    price "$LOGS/$step.log"
    (( ${#paired[@]} )) && echo "    scored WITH ${#paired[@]} earlier seed(s)"
    return 0
  fi
  rep="$(report_of "$LOGS/$step.log")"
  if [[ -n "$rep" ]]; then
    local rg rseed ratio lo hi word duty rid rest e elo ehi band eexit
    IFS=$'\t' read -r rg rseed ratio lo hi word duty rid rest < <("$PY_BASE" "$HELPERS" reading "$rep")
    IFS=$'\t' read -r e elo ehi band eexit < <("$PY_BASE" "$HELPERS" eta-for "$SESSION" "$RESULTS" "$g")
    echo "    G=$rg seed $rseed  ratio $ratio [$lo, $hi]  $word  duty $duty  eta $e [$elo, $ehi] $band  (R1 page: $eexit)"
  else
    echo "    no report.json for $step (see $LOGS/$step.log)"
  fi
}

# --------------------------------------------------------------------------
# 1. preflight: the scorers, proven on this interpreter
# --------------------------------------------------------------------------
echo; echo "== preflight"
if ! latched preflight-r1 "$LEDGER" || (( DRY )); then
  run_step preflight-r1 "$LOGS/preflight-r1.log" "$PY_BASE" "$REPO/scripts/clock_elasticity.py" --self-test --draws 50 || true
fi
if ! latched preflight-r3 "$LEDGER" || (( DRY )); then
  run_step preflight-r3 "$LOGS/preflight-r3.log" "$PY_BASE" "$REPO/scripts/private_weight_reference.py" --self-test refit || true
fi
(( DRY )) || preflight_gate || stop_chain

# --------------------------------------------------------------------------
# 2. preconditions, through the driver, in this session directory
# --------------------------------------------------------------------------
echo; echo "== preconditions (the driver's thermal, calibrate; pin_probe-n64-g1 for the record)"
if ! latched preconditions "$LEDGER" || (( DRY )); then
  pre_flags=(--only thermal,calibrate,pin_probe-n64-g1)
  (( DRY )) && pre_flags+=(--dry-run)
  run_step_as driver preconditions "$LOGS/preconditions.log" \
    env SESSION="$SESSION" bash "$REPO/scripts/h200_gaps_session.sh" "${pre_flags[@]}" || true
fi
(( DRY )) || preconditions_gate "$SESSION/ARMS.tsv" || stop_chain

# --------------------------------------------------------------------------
# 3. tests/test_gpu.py on this card, before any arm is booked on it
# --------------------------------------------------------------------------
echo; echo "== tests/test_gpu.py on this card, from the base venv (gated)"
if (( DRY )); then
  # a dry run COLLECTS and prices; off a card every one of them would skip
  run_step_as collect gpu-tests "$LOGS/gpu-tests.log" in_repo without_knobs "$PY_BASE" -m pytest \
    tests/test_gpu.py --collect-only -q -p no:cacheprovider || true
  read -r GPU_TESTS_N GPU_TESTS_S < <(price_tests "$LOGS/gpu-tests.log")
  echo "    priced ${GPU_TESTS_S} s: $GPU_TESTS_N tests at $SUITE_S_PER_TEST s each, session 4's pod rate"
elif ! latched gpu-tests "$LEDGER" && ! (( PAST_GPU_TESTS )); then
  # --past-gpu-tests is a decision taken AFTER reading a red page: it does not
  # run the same file again
  pytest_step gpu-tests "$GPU_TESTS_TIMEOUT_S" tests/test_gpu.py -q -rfE -p no:cacheprovider
fi
(( DRY )) || gpu_tests_gate "$PAST_GPU_TESTS" || stop_chain

# --------------------------------------------------------------------------
# 4. the ratio at seed 0 at every G; the pilot's V8 read first
# --------------------------------------------------------------------------
echo; echo "== the re-read fraction, off the cap, seed $FIRST_SEED at every G (the pilot: $PILOT)"
for g in $G_LADDER; do
  r3_step "$g" "$FIRST_SEED"
  if [[ "$g" == "$FIRST_G" ]] && ! (( DRY )); then
    v8_gate "$PAST_V8" || stop_chain
  fi
done

# --------------------------------------------------------------------------
# 5. the elasticity at every G of the ladder
# --------------------------------------------------------------------------
echo; echo "== clock elasticity per geometry (the regime word)"
for g in $G_LADDER; do
  step="r1-g$g"
  if latched "$step" "$LEDGER" && ! (( DRY )); then echo "  $step latched, skipped"; continue; fi
  # shellcheck disable=SC2046
  run_step "$step" "$LOGS/$step.log" $(r1_cmd "$g" "$DRY") || true
  (( DRY )) && price "$LOGS/$step.log"
done

# --------------------------------------------------------------------------
# 6. the later seeds, seed-major, each scored with the earlier ones of its G
# --------------------------------------------------------------------------
echo; echo "== the re-read fraction, the later seeds, scored with the earlier ones"
for seed in $SEEDS; do
  [[ "$seed" == "$FIRST_SEED" ]] && continue
  for g in $G_LADDER; do
    step="r3-g$g-s$seed"
    if latched "$step" "$LEDGER" && ! (( DRY )); then echo "  $step latched, skipped"; continue; fi
    v7="$(seed0_v7 "$g")"
    if [[ -n "$v7" && "$v7" != PASS ]]; then
      skip_row "$step" "seed $FIRST_SEED's page (r3-g$g-s$FIRST_SEED) read V7 $v7: the two ratio arms ran at different clocks, which is the power state and not the routing draw, so another seed would buy the same split"
      continue
    fi
    r3_step "$g" "$seed"
  done
done

# --------------------------------------------------------------------------
# 7. the whole suite, after every arm: a record of this box, gating nothing
# --------------------------------------------------------------------------
echo; echo "== the whole suite, uncapped, from the base venv (a record; gates nothing)"
if [[ "$END_SUITE" == skip ]]; then
  if (( DRY )); then echo "    END_SUITE=skip: not collected, not priced"; else skip_row suite "END_SUITE=skip: the operator did not run the end suite this pass"; fi
elif (( DRY )); then
  # a dry run COLLECTS and prices; running it here would take the laptop 20
  # minutes and run this chain's own dry-run tests inside itself
  run_step_as collect suite "$LOGS/suite.log" in_repo without_knobs "$PY_BASE" -m pytest tests/ \
    --collect-only -q -p no:cacheprovider || true
  read -r SUITE_N SUITE_S < <(price_tests "$LOGS/suite.log")
  echo "    priced ${SUITE_S} s: $SUITE_N tests at $SUITE_S_PER_TEST s each, session 4's pod rate"
elif latched suite "$LEDGER"; then
  echo "  suite latched, skipped"
else
  pytest_step suite "$SUITE_TIMEOUT_S" tests/ -q -rfE --durations=25 -p no:cacheprovider
fi

# --------------------------------------------------------------------------
# 8. the table, the price, and what to copy off
# --------------------------------------------------------------------------
echo
if (( DRY )); then
  n_r1=0; for g in $G_LADDER; do n_r1=$(( n_r1 + 1 )); done
  n_r3=0; for s in $SEEDS; do for g in $G_LADDER; do n_r3=$(( n_r3 + 1 )); done; done
  overhead=$(( 4 * 60 + n_r3 * 60 + 5 * 60 ))
  wall=$(( TOTAL_S + GPU_TESTS_S + SUITE_S + overhead ))
  echo "PRICE, off the arms' own plans: $n_r1 elasticity runs + $n_r3 ratio runs = $TOTAL_S s of arms,"
  echo "  tests/test_gpu.py ~$GPU_TESTS_S s (${GPU_TESTS_N:-?} tests) before them and the end suite ~$SUITE_S s"
  echo "  (${SUITE_N:-0} tests) after them, at $SUITE_S_PER_TEST s a test, session 4's pod rate,"
  echo "  plus ~$overhead s (preconditions, per-run compiles and weight copies, exfil) = $wall s"
  echo "  = $(( (wall + 59) / 60 )) min; at \$$RATE_USD_H/h about \$$(awk -v w="$wall" -v r="$RATE_USD_H" 'BEGIN {printf "%.2f", w / 3600 * r}'). Book $(( (wall + 3599) / 3600 + 1 )) h."
  echo "  NOT in that figure: pod boot and checkout, any arm that REFUSES and is re-run,"
  echo "  and the seeds a V7 failure at seed 0 skips (less, not more)."
  echo "on the pod, first:"
  echo "    git -C /workspace/moe-kernels checkout -- moe/bench/hardware/measured_nvidia_h200.yaml"
  echo "    git -C /workspace/moe-kernels fetch origin && git -C /workspace/moe-kernels checkout -B r3-align origin/r3-align"
  echo "    git -C /workspace/moe-kernels log -1 --format=%h    # must print the head that was pushed"
  echo "then detached:  cd /workspace/moe-kernels && nohup setsid bash scripts/alpha_g_chain.sh > /workspace/alpha_g_chain.out 2>&1 < /dev/null &"
  echo "and watch:      tail -f /workspace/alpha_g_chain.out   (and \$SESSION/CHAIN.tsv)"
else
  rebuild_pairs
fi
echo "ledger    $LEDGER"
echo "read a pair on the laptop:  .venv/bin/python scripts/private_weight_reference.py --read RUN1/report.json --replicate-of RUN0/report.json"
YAML_REL="moe/bench/hardware/measured_$CARD.yaml"
CALIB_DIR="$("$PY_BASE" "$HELPERS" calibration-dir "$SESSION/logs/calibrate.log" 2>/dev/null)" || CALIB_DIR=""
CALIB_REL="${CALIB_DIR#"$REPO"/}"
if (( DRY )); then
  CALIB_ARG="-C \"$REPO\" results/calibration/<the run dir calibrate prints>"
elif [[ -n "$CALIB_DIR" && "$CALIB_REL" != "$CALIB_DIR" ]]; then
  CALIB_ARG="-C \"$REPO\" \"$CALIB_REL\""
elif [[ -n "$CALIB_DIR" ]]; then
  CALIB_ARG="-C \"$(dirname "$CALIB_DIR")\" \"$(basename "$CALIB_DIR")\""
else
  CALIB_ARG=""
  echo "  (no '[calibrate] wrote' line in $SESSION/logs/calibrate.log: copy calibrate's run dir"
  echo "   under $REPO/results/calibration off by hand)"
fi
echo "the ruler this session scored against is $REPO/$YAML_REL, TRACKED and dirty after"
echo "  calibrate: commit it with the results. copy off before releasing the pod:"
echo "    tar czf $WORKSPACE/exfil-alpha_g-$CARD.tar.gz -C \"$(dirname "$SESSION")\" \"$(basename "$SESSION")\" -C \"$(dirname "$RESULTS")\" \"$(basename "$RESULTS")\" -C \"$REPO\" \"$YAML_REL\"${CALIB_ARG:+ $CALIB_ARG}"
exit 0
