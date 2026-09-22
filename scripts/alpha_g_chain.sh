#!/bin/bash
# The alpha(G) matrix session: the chain that produces the re-read-fraction
# table the analytical model needs, with both ratio arms held OFF the power
# cap, every geometry's clock elasticity measured beside it, and every G run
# as a seed triple scored jointly.
#
#   bash scripts/alpha_g_chain.sh --dry-run          # plan and price, nothing measured
#   bash scripts/alpha_g_chain.sh                    # a new chain session on this card
#   bash scripts/alpha_g_chain.sh --resume           # continue the latest chain session
#   SESSION=<dir> bash scripts/alpha_g_chain.sh      # continue a named one
#
# WHAT IT RUNS, IN ORDER, and why that order:
#   preflight      the two arms' --self-test on the box's interpreter: the
#                  scorers are proven before a cent is spent.
#   preconditions  scripts/h200_gaps_session.sh --only thermal,calibrate,pin_probe-n64-g1
#                  in THIS session directory: card healthy, ruler measured on
#                  this card, tile pin honoured. Its own ledger (ARMS.tsv) is
#                  read back and thermal/calibrate must be DONE.
#   r1-g<G>        scripts/clock_elasticity.py at each G of the ladder, three
#                  cap-binding duty states (1.0 0.7 0.5), the per-M-tile
#                  elasticity gated. This is the REGIME test for that G: below
#                  the RAW-STANDS edge the per-tile cost is a traffic quantity
#                  and the ratio beside it is a re-read fraction; above it the
#                  ratio is a time ratio and the table says so.
#   r3-g<G>-s<seed> scripts/private_weight_reference.py at --duty 0.5, seed
#                  major (every G at seed 0, then seed 1, then seed 2), each
#                  later seed scored WITH the earlier ones of its G through
#                  --replicate-of (DESIGN DECISION 14). Seed-major so a pair
#                  is 20+ minutes apart and the envelope sees the drift it
#                  exists to capture.
#
# WHAT IT LEAVES: $SESSION/CHAIN.tsv (one row per step: state, rc, seconds,
# log, note, with the driver's second opinion taken off the RESULT lines),
# $SESSION/PAIRS.tsv (one row per ratio run: G, seed, ratio, interval, exit
# word, duty, run id, and the G's elasticity with its band), logs under
# $SESSION/chain-logs/, and the driver's own ARMS.tsv beside them. --resume
# skips every step whose newest row is latched (DONE, CLAIM_FAIL, INVALID)
# and re-runs REFUSED, ERROR and UNKNOWN ones.
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
#: The ratio arms off the cap (DESIGN DECISION 15): session 4's clock arm sat at
#: the boost ceiling at every tread at or below 50% duty.
R3_DUTY="${R3_DUTY:-0.5}"
#: The elasticity arm's states: cap-binding ones. Session 4 showed duty 0.5,
#: 0.25 and 0.10 were one clock cluster, so two of its four states bought
#: nothing and cost 14/17 of the wall; three states is MIN_STATES.
R1_DUTY="${R1_DUTY:-1.0 0.7 0.5}"
RATE_USD_H="${RATE_USD_H:-4.59}"

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

#: Is a step's newest ledger row a latched state.
latched() {
  local name="$1" ledger="$2" state
  [[ -f "$ledger" ]] || return 1
  state="$(awk -F'\t' -v n="$name" '$1==n {s=$2} END {print s}' "$ledger")"
  case "$state" in DONE|CLAIM_FAIL|INVALID) return 0 ;; *) return 1 ;; esac
}

#: The driver's dirty count, verbatim: rc handled apart from the pipeline.
dirty_count() {
  local out
  out="$(git -C "$REPO" status --porcelain --untracked-files=all 2>/dev/null)" || { echo "-"; return 0; }
  printf '%s\n' "$out" | grep -c . || true
}

#: Run one step and write its row. $1 name, $2 log, $3.. the command. The
#: command's exit code is captured directly: never `cmd | tee`, which reports
#: tee's. Returns the command's rc. OPINION=driver (the preconditions step,
#: which is the driver sequencing arms and prints no RESULT line of its own)
#: takes the exit code as the state and leaves the per-arm second opinion to
#: the driver's ARMS.tsv, which the chain reads back.
run_step() {
  local name="$1" log="$2"; shift 2
  local rc=0 t0 secs implied state note
  t0="$(date +%s)"
  "$@" > "$log" 2>&1 || rc=$?
  secs="$(( $(date +%s) - t0 ))"
  if [[ "${OPINION:-page}" == driver ]]; then
    state="$(state_word "$rc")"
    note="the driver's own ARMS.tsv carries each arm's second opinion"
  else
    implied="$("$PY_BASE" "$HELPERS" verdict "$log" 2>/dev/null)" || implied="UNREADABLE"
    IFS=$'\t' read -r state note < <(state_for "$rc" "$implied")
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$state" "$rc" "$secs" \
    "$(dirty_count)" "$log" "$note" >> "$LEDGER"
  printf '%-16s %-10s rc=%s %5ss  %s\n' "$name" "$state" "$rc" "$secs" "$note"
  return "$rc"
}
# <<< LIFTABLE

usage() { sed -n '2,45p' "$0"; }

DRY=0; RESUME=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --resume)  RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "REFUSED: unknown argument $1"; usage; exit 2 ;;
  esac
done

[[ -x "$PY_BASE" ]] || { echo "REFUSED: no usable base interpreter (set PY_BASE=)"; exit 2; }
[[ -f "$HELPERS" ]] || { echo "REFUSED: $HELPERS is missing"; exit 2; }

# THE CARD, as the driver resolves it: through torch on PY_BASE, as a slug.
CARD="nocard"
CARD="$("$PY_BASE" - "$REPO" <<'PY' 2>/dev/null
import sys
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
try:
    from moe.bench.provenance import card_slug, provenance_block
    prov = provenance_block(repo_root=repo)
    print(card_slug(prov.gpu_name) if prov.gpu_name else "nocard")
except Exception:                                                  # noqa: BLE001
    print("nocard")
PY
)" || CARD="nocard"
CARD="${CARD:-nocard}"

if [[ -d /workspace ]]; then
  SESSION_ROOT="${SESSION_ROOT:-/workspace/session}"
  RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"
  WORKSPACE="${WORKSPACE:-/workspace}"
else
  SESSION_ROOT="${SESSION_ROOT:-$REPO/results/h200_gaps}"
  RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results}"
  WORKSPACE="${WORKSPACE:-$RESULTS_ROOT}"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "${SESSION:-}" ]]; then
  :
elif (( RESUME )); then
  SESSION="$(ls -d "$SESSION_ROOT"/alpha_g-"$CARD"-* 2>/dev/null | sort | tail -1)"
  [[ -n "$SESSION" ]] || { echo "REFUSED: --resume found no alpha_g-$CARD-* session under $SESSION_ROOT"; exit 2; }
else
  SESSION="$SESSION_ROOT/alpha_g-$CARD-$STAMP"
fi
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
PAIRS="$SESSION/PAIRS.tsv"
mkdir -p "$LOGS" || { echo "REFUSED: cannot create $LOGS"; exit 2; }
[[ -f "$LEDGER" ]] || printf 'step\tstate\trc\tseconds\tdirty\tlog\tnote\n' > "$LEDGER"
[[ -f "$PAIRS" ]] || printf 'G\tseed\tratio\tlo\thi\texit\tduty\trun_id\teta\teta_lo\teta_hi\tband\teta_exit\n' > "$PAIRS"

echo "alpha(G) chain  $TAG"
echo "  card        $CARD"
echo "  session     $SESSION"
echo "  results     $RESULTS   (exported as MOE_RESULTS_DIR to every arm)"
echo "  ledger      $LEDGER"
echo "  ladder      G in {$G_LADDER}, seeds {$SEEDS}, ratio arms at duty $R3_DUTY, elasticity states $R1_DUTY"
echo "  interpreters base $PY_BASE / vllm $PY_VLLM"
(( DRY )) && echo "  DRY RUN: every arm's own --dry-run is run and priced; nothing is measured"

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
  echo --model mixtral-8x7b --dtype bf16 --group-m "$g" --treads 8 --duty $R1_DUTY \
       --repeats 13 --burst-ms 40 --target-ms 200 --trials 3 --warm-ms 200 \
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
  echo --model mixtral-8x7b --block-m 32 --treads 6 --repeats 9 \
       --group-m "$g" --duty "$R3_DUTY" --seed "$seed" --session-tag "$TAG"
  if (( $# )); then echo --replicate-of "$@"; fi
}

TOTAL_S=0
price() {   # $1 log: add the arm's own dry-run estimate to the total
  local est
  est="$("$PY_BASE" "$HELPERS" estimate "$1" 2>/dev/null)" || est=""
  if [[ -n "$est" ]]; then TOTAL_S=$(( TOTAL_S + est )); echo "    priced ${est} s off its own plan"; else echo "    (no estimate on its plan page)"; fi
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
for step in preflight-r1 preflight-r3; do
  if ! latched "$step" "$LEDGER"; then
    (( DRY )) || { echo "STOP: $step is not DONE; the scorer is not proven here"; exit 3; }
  fi
done

# --------------------------------------------------------------------------
# 2. preconditions, through the driver, in this session directory
# --------------------------------------------------------------------------
echo; echo "== preconditions (the driver's thermal, calibrate, pin_probe-n64-g1)"
if ! latched preconditions "$LEDGER" || (( DRY )); then
  pre_flags=(--only thermal,calibrate,pin_probe-n64-g1)
  (( DRY )) && pre_flags+=(--dry-run)
  SESSION="$SESSION" OPINION=driver run_step preconditions "$LOGS/preconditions.log" bash "$REPO/scripts/h200_gaps_session.sh" "${pre_flags[@]}" || true
  if ! (( DRY )); then
    arms="$SESSION/ARMS.tsv"
    for need in thermal calibrate; do
      if ! latched "$need" "$arms"; then
        echo "STOP: the driver's $need row is not DONE in $arms; nothing below can be scored"; exit 3
      fi
    done
  fi
fi

# --------------------------------------------------------------------------
# 3. the elasticity at every G of the ladder
# --------------------------------------------------------------------------
echo; echo "== clock elasticity per geometry (the regime test)"
for g in $G_LADDER; do
  step="r1-g$g"
  if latched "$step" "$LEDGER" && ! (( DRY )); then echo "  $step latched, skipped"; continue; fi
  # shellcheck disable=SC2046
  run_step "$step" "$LOGS/$step.log" $(r1_cmd "$g" "$DRY") || true
  (( DRY )) && price "$LOGS/$step.log"
done

# --------------------------------------------------------------------------
# 4. the ratio at every G, seed-major, later seeds scored with the earlier ones
# --------------------------------------------------------------------------
echo; echo "== the re-read fraction, off the cap, G x seed"
report_of() {   # $1 log: the run's report.json, off the run id its plan page printed
  local rid
  rid="$("$PY_BASE" "$HELPERS" run-id "$1" private_weight_reference 2>/dev/null)" || rid=""
  [[ -n "$rid" ]] && echo "$RESULTS/private_weight_reference/$rid/report.json"
}
for seed in $SEEDS; do
  for g in $G_LADDER; do
    step="r3-g$g-s$seed"
    if latched "$step" "$LEDGER" && ! (( DRY )); then echo "  $step latched, skipped"; continue; fi
    # the earlier seeds of this G that FORMED a ratio (the rule load_replicates applies)
    earlier=()
    for s2 in $SEEDS; do
      [[ "$s2" == "$seed" ]] && break
      elog="$LOGS/r3-g$g-s$s2.log"
      [[ -f "$elog" ]] || continue
      rep="$(report_of "$elog")"
      [[ -n "$rep" && -f "$rep" ]] && earlier+=("$rep")
    done
    paired=()
    if (( ${#earlier[@]} )); then
      while IFS= read -r line; do [[ -n "$line" ]] && paired+=("$line"); done \
        < <("$PY_BASE" "$HELPERS" pairs ${earlier[@]+"${earlier[@]}"} 2>/dev/null)
    fi
    # shellcheck disable=SC2046
    run_step "$step" "$LOGS/$step.log" $(r3_cmd "$g" "$seed" "$DRY" ${paired[@]+"${paired[@]}"}) || true
    if (( DRY )); then
      price "$LOGS/$step.log"
      (( ${#paired[@]} )) && echo "    scored WITH ${#paired[@]} earlier seed(s)"
      continue
    fi
    rep="$(report_of "$LOGS/$step.log")"
    if [[ -n "$rep" && -f "$rep" ]]; then
      IFS=$'\t' read -r rg rseed ratio lo hi word duty rid < <("$PY_BASE" "$HELPERS" reading "$rep")
      r1log="$LOGS/r1-g$g.log"
      eta_row="none	none	none	unmeasured	unscored"
      if [[ -f "$r1log" ]]; then
        r1id="$("$PY_BASE" "$HELPERS" run-id "$r1log" clock_elasticity 2>/dev/null)" || r1id=""
        [[ -n "$r1id" && -f "$RESULTS/clock_elasticity/$r1id/report.json" ]] \
          && eta_row="$("$PY_BASE" "$HELPERS" eta "$RESULTS/clock_elasticity/$r1id/report.json")"
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rg" "$rseed" "$ratio" "$lo" "$hi" "$word" "$duty" "$rid" "$eta_row" >> "$PAIRS"
      echo "    G=$rg seed $rseed  ratio $ratio [$lo, $hi]  $word  duty $duty  eta/band: $(printf '%s' "$eta_row" | cut -f1,4)"
    else
      echo "    no report.json for $step (see $LOGS/$step.log)"
    fi
  done
done

# --------------------------------------------------------------------------
# 5. the price, and what to copy off
# --------------------------------------------------------------------------
echo
if (( DRY )); then
  n_r1=0; for g in $G_LADDER; do n_r1=$(( n_r1 + 1 )); done
  n_r3=0; for s in $SEEDS; do for g in $G_LADDER; do n_r3=$(( n_r3 + 1 )); done; done
  overhead=$(( 4 * 60 + n_r3 * 60 + 5 * 60 ))
  wall=$(( TOTAL_S + overhead ))
  echo "PRICE, off the arms' own plans: $n_r1 elasticity runs + $n_r3 ratio runs = $TOTAL_S s of arms,"
  echo "  plus ~$overhead s (preconditions, per-run compiles and weight copies, exfil) = $wall s"
  echo "  = $(( (wall + 59) / 60 )) min; at \$$RATE_USD_H/h about \$$(awk -v w="$wall" -v r="$RATE_USD_H" 'BEGIN {printf "%.2f", w / 3600 * r}'). Book $(( (wall + 3599) / 3600 + 1 )) h."
  echo "  NOT in that figure: pod boot and checkout, and any arm that REFUSES and is re-run."
fi
echo "ledger    $LEDGER"
echo "pairs     $PAIRS"
echo "read a pair on the laptop:  .venv/bin/python scripts/private_weight_reference.py --read RUN1/report.json --replicate-of RUN0/report.json"
echo "copy off before releasing the pod:"
echo "    tar czf $WORKSPACE/exfil-alpha_g-$CARD.tar.gz -C \"$(dirname "$SESSION")\" \"$(basename "$SESSION")\" -C \"$(dirname "$RESULTS")\" \"$(basename "$RESULTS")\""
exit 0
