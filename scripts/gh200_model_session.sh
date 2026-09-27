#!/usr/bin/env bash
# THE GH200 MODEL-TEST SESSION, RUN UNATTENDED ON THE VM (docs/LAMBDA.md section 3c).
#
#   bash gh200_model_session.sh --dry-run                        # the plan; runs nothing (exit 2)
#   bash gh200_model_session.sh --setup --commit <sha> --repo <url> --deadline <epoch s>
#   bash gh200_model_session.sh --deadline <epoch s>             # after setup_vm.sh printed READY
#   bash gh200_model_session.sh --from eta --deadline <epoch s>  # resume at a step
#
# WHAT IT TESTS. The two NOT-FINAL models of docs/COUNTERS.md 6.7 on a GH200
# (`gpu_1x_gh200`): the wave-split byte model (scripts/wave_split_bytes.py) and
# the timing model LOWM-1/2 (scripts/r3_timing_model.py, predictions P1 to P8).
# The duty chain cannot hold this card's clock still (section 3b, the record of
# 2026-09-25: V7 and V0 failed under duty), so every timed page runs under an
# nvidia-smi lock through scripts/locked_r3.py.
#
# THE OWNER'S DECISIONS, 2026-09-26: byte pages count every tread from 1 to 9;
# calibrate runs on the VM after the byte pages; the GPU power limit is set to
# 700 W at the start (-sc 0), the limit the 2026-09-25 lock pages ran at; a VM
# that ships r570 is upgraded to 580.105.08 before setup_vm.sh
# (scripts/vm_run.sh does that, not this script).
#
# THE STEPS, in the owner's priority order, so a run cut short has run the
# earlier ones. Minutes are estimates from the 2026-09-25 logs and the tools'
# own plans; each step also has a cap, past which it is stopped:
#
#   prelude    3c.1  supported clocks, persistence, 700 W        the locks the later steps may use
#   bytes      3c.2  byte pages at the 1710 lock, G = 1 2 4 16 8 32 3 64, treads 1-9,
#                    and one base-clock control                  the byte predictions; counted bytes for every timed G
#   calibrate  3c.3  this card's ruler, with no lock in force   the ruler R3 needs
#   timed      3c.3  timed R3 at 1710: G=8, 32 (treads 6), G=3 (treads 8)   P2, P5
#   eta        3c.4  timed R3 at held locks 1410 (G=4, 2), 1500 and 1605 (G=4)   P1
#   floor      3c.5  floor counters: G=64, G=2 at ncu's base clock, G=64 unlocked
#                    (a record), G=64 under the 1710 lock        P6
#   deep       3c.6  timed R3 at 1710 to tread 9, G=4 then G=2  P3, P4
#   r1lock     3c.7  R1 in lock mode at 1710 1500 1410, G=4 then G=1   the floor's clock exponent
#
# WHY UNATTENDED, AND HOW THE DRAFT'S "STOPS" BECAME RULES. The draft of
# section 3c (2026-09-26) was blocks a person pastes, each with a "Stops"
# paragraph read before the next block. On an unattended VM every one of those
# readings is a rule here, and a stop that needs a person stops only its own
# step: the driver resets both kinds of lock, pushes what the step wrote, and
# goes on to the next step, because every step checks the card for itself
# before it measures (idle, no lock left, its own dry run). Three things stop
# the whole session instead, because nothing after them could be trusted: the
# prelude refusing (a dirty checkout, a busy card, no 1710 lock, persistence or
# 700 W not taken), a reset that fails (`nvidia-smi -rgc` exiting non-zero:
# the card may still be locked), and a card still busy five minutes after a
# step ended. The rules, step by step, are in docs/LAMBDA.md section 3c.
#
# RESULTS GO TO GITHUB AS THEY LAND. After every step, and only between steps
# (never while a page is measured), scripts/vm_results_push.sh pushes the VM's
# results and session to the run's own branch. The laptop reads them with `git
# fetch`, and terminates the instance once DRIVER-DONE is on the branch.
#
# TIME. --deadline is the epoch second by which the driver must have ended,
# final push included (the laptop sets it: launch + 400 minutes, for a 7-hour
# stop). Before each step the driver adds the estimates of every step still to
# run; while that overruns the deadline less RESERVE_MIN it drops, in the
# draft's order, R1 at G=1, then R1 at G=4, then the 1605 lock, then the G=64
# byte page and the base-clock control. A step whose own estimate no longer
# fits is skipped. A step's cap never runs past the deadline.
#
# EXIT CODES, the repo's table (moe/bench/exit_codes.py): 0 every step ran and
# ended 0 or 1; 2 refused before measuring (the prelude, or bad arguments; and
# --dry-run); 3 a step stopped short, was skipped or ran out of time; 4 a lock
# reset failed, the card stayed busy, or a step crashed.
#
# PRIVACY. locked_r3.py's logs and ledger, and every page, carry the card's
# UUID. This driver's own ledger ($SESSION_ROOT/gh200-driver/status) copies
# only the ledger lines that name no card (`G=<n> <state> at lock <F>: R3 exit`)
# and never a `card:` line, so it can be read whole.
#
# TEST SEAMS: MOE_HOME (default $HOME/moe: env.sh, session, results),
# MOE_DRIVER_RESERVE_MIN, MOE_DRIVER_BUSY_WAIT_S, MOE_DRIVER_NOW (a fixed
# clock, epoch seconds, for the budget tests), MOE_DRIVER_REPRICE_S (the G=1
# page's re-price threshold, 420 s).
set -uo pipefail

EXIT_DONE=0
EXIT_CLAIM_FAIL=1
EXIT_REFUSED=2
EXIT_INVALID=3
EXIT_ERROR=4

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
MOE_HOME="${MOE_HOME:-$HOME/moe}"
ENV_SH="$MOE_HOME/env.sh"

# ---- the design: the owner's decisions and the registered predictions ------
#: The lock every timed page and every byte page runs at: P2 to P6 are
#: registered at it, and the 2026-09-25 pages ran at it.
LOCK_TIMED=1710
#: The owner's GPU power limit (2026-09-26), GPU scope only (-sc 0). On a GH200
#: the module's limit (-sc 1) is shared with the Grace CPU; both are recorded.
POWER_LIMIT_W=700
#: Byte pages count every tread from 1 to 9: private_weight_reference.py's
#: COUNTER_MAX_TREADS (a test holds this to it).
TREADS=1,2,3,4,5,6,7,8,9
#: The owner's six, then 3 (the timed G=3 page's bytes), then 64 (dropped first).
BYTE_GS=(1 2 4 16 8 32 3 64)
#: The first byte page re-prices the rest: past this many seconds, the G=64
#: page and the base-clock control are dropped (the draft's 7 minutes).
BYTE_G1_REPRICE_S="${MOE_DRIVER_REPRICE_S:-420}"
#: P1's registered held locks, scripts/r3_timing_model.py P1_CLOCKS (a test
#: holds these to it). 1605 falls back to 1590, whose P1 numbers were
#: registered with 1605's; any other substitute is labelled as computed after
#: registration.
P1_LOCKS=(1410 1500 1605)
P1_FALLBACK_1605=1590
#: The 2026-09-25 GH200 ruler's numbers: R3 takes them when calibrate fails.
RULER_FALLBACK=(--ridge 177.93 --bandwidth-gbps 3725.1)
#: R3's design for every timed page (the chain's, as on 2026-09-25).
#: The model R3, R1 and every counter page run (--model; the steps read it
#: from MOE_DRIVER_MODEL, which the orchestrator exports). Mixtral 8x7B is the
#: study's; another model is a cross-model test of the two per-card models
#: (docs/registered/README.md), with its own census and no 8x7B references.
DEFAULT_MODEL=mixtral-8x7b
MODEL="${MOE_DRIVER_MODEL:-$DEFAULT_MODEL}"
R3_BASE=(--model "$MODEL" --block-m 32 --repeats 9 --duty 0.25 --seed 0)
#: R1 in lock mode (section 3b); each state a held SM clock.
R1_BASE=(--model "$MODEL" --dtype bf16 --treads 8 --repeats 13 --burst-ms 40
         --target-ms 200 --trials 3 --warm-ms 200 --settle-seconds 10 --lock-duty 0.25)
#: R1's claim spans down to this clock: its lowest lock must be at or under it.
R1_SPAN_MAX_LOW=1459
#: The GH200's 2026-09-25 R3 pages timed at 1710 (session tag ending
#: -lock1710), which C5 scores the byte pages against.
TIMED_REF_DIR="results/published/2026-09-25-nvidia_gh200_480gb-session/results/gaps-nvidia_gh200_480gb/private_weight_reference"
TIMED_REF_IDS=(d9f1f37c df37ea07 01c08abd 1b285de2 6ff34777)
#: The floor's G=2 capture is dropped when the G=64 capture's ncu log shows
#: more replay passes than this per launch (the draft's rule).
FLOOR_MAX_PASSES=50

# ---- the steps: name, estimate and cap in minutes, what it answers ----------
STEPS=(prelude bytes calibrate timed eta floor deep r1lock)
#: Another model's pages run longer by about its weight bytes over 8x7B's
#: (8x22B: 4.83 GB against 2.82, x1.7); the estimates and caps below are
#: 8x7B's, scaled by this percentage for the steps that measure.
model_scale_pct() { case "$MODEL" in "$DEFAULT_MODEL") echo 100 ;; mixtral-8x22b) echo 170 ;; *) echo 200 ;; esac; }
_scaled() { case "$1" in prelude|calibrate) echo "$2" ;; *) echo $(( $2 * $(model_scale_pct) / 100 )) ;; esac; }
step_est() {
  local m
  case "$1" in
    prelude) m=3 ;; bytes) m=55 ;; calibrate) m=8 ;; timed) m=37 ;;
    eta) m=50 ;; floor) m=15 ;; deep) m=33 ;; r1lock) m=62 ;;
  esac
  _scaled "$1" "$m"
}
step_cap() {
  local m
  case "$1" in
    prelude) m=15 ;; bytes) m=120 ;; calibrate) m=30 ;; timed) m=90 ;;
    eta) m=115 ;; floor) m=45 ;; deep) m=75 ;; r1lock) m=130 ;;
  esac
  _scaled "$1" "$m"
}
#: locked_r3.py's cap on one R3 run (its default, 1800 s, fits 8x7B's pages);
#: another model's deep page runs about 27 min, so it takes an hour.
lr3_cap() { [[ "$MODEL" == "$DEFAULT_MODEL" ]] || printf '%s\n' --run-cap-s 3600; }
#: The census the byte and floor pages read: the preflight's (PF6, 8x7B's) for
#: the study's model, else one taken for this model at ncu's base clock.
census_for() {
  if [[ "$MODEL" == "$DEFAULT_MODEL" ]]; then printf '%s\n' "$S/census.json"; return 0; fi
  local C="$S/census-$MODEL.json"
  if [[ ! -s "$C" ]]; then
    moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --census-only \
      --model "$MODEL" --out "$C" > "$S/logs/census-$MODEL.log" 2>&1 \
      || { ledger "census for $MODEL failed (logs/census-$MODEL.log)" >&2; return 1; }
    moe_counter ncu --clock-control reset >/dev/null 2>&1 || true
    ledger "census for $MODEL written ($C)" >&2
  fi
  printf '%s\n' "$C"
}
step_what() {
  case "$1" in
    prelude)   echo "3c.1 supported clocks, persistence, ${POWER_LIMIT_W} W" ;;
    bytes)     echo "3c.2 byte pages at the ${LOCK_TIMED} lock, G = ${BYTE_GS[*]}, treads $TREADS; base-clock control" ;;
    calibrate) echo "3c.3 calibrate (the ruler), no lock in force" ;;
    timed)     echo "3c.3 timed R3 at ${LOCK_TIMED}: G=8, 32 (treads 6), G=3 (treads 8): P2, P5" ;;
    eta)       echo "3c.4 timed R3 at held locks 1410 (G=4, 2), 1500 and 1605 (G=4): P1" ;;
    floor)     echo "3c.5 floor counters, G=64 and G=2 at base, G=64 unlocked (record), G=64 at ${LOCK_TIMED}: P6" ;;
    deep)      echo "3c.6 timed R3 at ${LOCK_TIMED} to tread 9, G=4 then G=2: P3, P4" ;;
    r1lock)    echo "3c.7 R1 in lock mode, G=4 then G=1: the floor's clock exponent" ;;
  esac
}
#: What the deadline drops, in the draft's order, with the minutes each saves.
DROPS=(r1_g1 r1_g4 eta_1605 bytes_tail)
drop_step() { case "$1" in r1_g1|r1_g4) echo r1lock ;; eta_1605) echo eta ;; bytes_tail) echo bytes ;; esac; }
drop_min()  { case "$1" in r1_g1|r1_g4) echo 31 ;; eta_1605) echo 12 ;; bytes_tail) echo 14 ;; esac; }
drop_what() {
  case "$1" in
    r1_g1) echo "R1 at G=1" ;; r1_g4) echo "R1 at G=4" ;; eta_1605) echo "the 1605 lock" ;;
    bytes_tail) echo "the G=64 byte page and the base-clock control" ;;
  esac
}
RESERVE_MIN="${MOE_DRIVER_RESERVE_MIN:-8}"
BUSY_WAIT_S="${MOE_DRIVER_BUSY_WAIT_S:-300}"

# ---- small tools ---------------------------------------------------------
utc()    { date -u +%Y-%m-%dT%H:%M:%SZ; }
now_s()  { if [[ -n "${MOE_DRIVER_NOW:-}" ]]; then echo "$MOE_DRIVER_NOW"; else date +%s; fi; }
utc_of() { date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ; }
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
#: One UTC line in the driver's ledger, and on the console.
ledger() {
  local line; line="$(utc) $*"
  printf '%s\n' "$line"
  [[ -n "${D:-}" ]] && { mkdir -p "$D"; printf '%s\n' "$line" >> "$D/status"; }
  return 0
}
dropped() { [[ -f "$D/drops" ]] && grep -qx -- "$1" "$D/drops"; }
#: A session tag no earlier run of this driver used: BASE, else BASEb, BASEc...
#: locked_r3.py refuses a tag whose ledger or pages exist, so a rerun of a step
#: takes the next letter rather than being refused.
fresh_tag() {
  local base="$1" s t
  for s in "" b c d e f g h i j; do
    t="$base$s"
    if ! grep -qx -- "$t" "$D/tags-used" 2>/dev/null && [[ ! -e "$SESSION_ROOT/locked-r3/$t" ]]; then
      printf '%s\n' "$t" >> "$D/tags-used"
      printf '%s\n' "$t"
      return 0
    fi
  done
  return 1
}
gpu_busy() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1; }
#: Every ledger line of a locked_r3.py run that names no card.
ledger_lines() { grep -hE 'G=[0-9]+ [a-z_-]+ at lock [0-9]+: R3 exit' "$@" 2>/dev/null || true; }

#: R3's own dry run may go on to a lock: no REFUSED line, the 9-copy
#: declaration, a retracted tread of at least the treads planned, and a ruler
#: (not the H200 HYPOTHESIS R3 falls back to without one, under which it
#: refuses the timed run).
check_dry() {
  local log="$1" planned="$2" n
  if grep -q '^REFUSED:' "$log"; then
    echo "REFUSED: $(grep -m1 '^REFUSED:' "$log" | cut -c10-160)"; return 1
  fi
  grep -qE 'n_decl = 9 against n_max' "$log" || { echo "no 'n_decl = 9 against n_max' line"; return 1; }
  n="$(grep -oE 'retracted +tread +[0-9]+ +against the [0-9]+ planned' "$log" | head -1 | awk '{print $3}')"
  [[ -n "$n" && "$n" -ge "$planned" ]] || { echo "retracted tread ${n:-(none printed)} is under the $planned planned"; return 1; }
  if grep -qE '^ *ridge .*HYPOTHESIS' "$log"; then
    echo "no ruler: the ridge is R3's H200 HYPOTHESIS"; return 1
  fi
  return 0
}

#: The ruler arguments every timed step hands R3: none when calibrate's ruler
#: stands, the 2026-09-25 numbers when it failed or never ran.
load_ruler() {
  RULER_ARGS=()
  RULER_KIND=""
  if [[ -f "$D/ruler.env" ]]; then
    # shellcheck disable=SC1090
    . "$D/ruler.env"
  elif [[ -f "$REPO/moe/bench/hardware/measured_$MOE_CARD.yaml" ]]; then
    RULER_KIND=measured
  else
    RULER_KIND=fallback
    RULER_ARGS=("${RULER_FALLBACK[@]}")
    ledger "no ruler decision on record (calibrate did not run): R3 takes the 2026-09-25 ruler's numbers, ${RULER_FALLBACK[*]}"
  fi
}

#: The locks the prelude resolved from this card's supported list.
load_locks() {
  LOCK_ETA_A=""; LOCK_ETA_B=""; LOCK_ETA_C=""
  # shellcheck disable=SC1090
  [[ -f "$D/locks.env" ]] && . "$D/locks.env"
  : "${LOCK_ETA_A:=${P1_LOCKS[0]}}" "${LOCK_ETA_B:=${P1_LOCKS[1]}}" "${LOCK_ETA_C:=${P1_LOCKS[2]}}"
}

#: The step-mode prologue: the VM's environment and the checkout.
enter() {
  [[ -f "$ENV_SH" ]] || { echo "no $ENV_SH: run setup_vm.sh first (docs/LAMBDA.md section 2)"; exit "$EXIT_REFUSED"; }
  # shellcheck disable=SC1090
  . "$ENV_SH"
  cd "$REPO" || exit "$EXIT_REFUSED"
  S="$SESSION_ROOT"
  D="$S/gh200-driver"
  mkdir -p "$S/logs" "$D"
  B="$(cat "$S/base-tag.txt" 2>/dev/null || true)"
  # Every later step needs what the prelude set: the base tag, the lock plan,
  # persistence and the 700 W limit.
  if [[ "$1" != prelude && ( -z "$B" || ! -f "$D/locks.env" ) ]]; then
    echo "step $1 REFUSED: the prelude has not passed on this VM (no base-tag.txt or locks.env)"
    exit "$EXIT_REFUSED"
  fi
}

# ==========================================================================
# 3c.1 prelude
# ==========================================================================
step_prelude() {
  [[ -s "$S/base-tag.txt" ]] || echo "gh200-$(date -u +%Y%m%dT%H%M%SZ)" > "$S/base-tag.txt"
  B="$(cat "$S/base-tag.txt")"
  echo "base tag $B (every locked run adds its own suffix)"
  git rev-parse HEAD | tee "$S/commit.txt"
  git status --porcelain | tee "$S/git-at-start.txt"
  if [[ -s "$S/git-at-start.txt" ]]; then
    ledger "prelude REFUSED: the checkout is not clean (git-at-start.txt)"; return "$EXIT_REFUSED"
  fi
  if ! sudo -n true 2>/dev/null; then
    ledger "prelude REFUSED: no passwordless sudo, which every lock needs"; return "$EXIT_REFUSED"
  fi
  local busy; busy="$(gpu_busy)"
  if [[ -n "$busy" ]]; then
    ledger "prelude REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"
  fi
  if ! nvidia-smi -q -d SUPPORTED_CLOCKS > "$S/supported-clocks.txt"; then
    ledger "prelude REFUSED: nvidia-smi could not list the supported clocks"; return "$EXIT_REFUSED"
  fi
  # Which of the registered locks this card supports, and the locks the later
  # steps take: 1710 or nothing; 1605, else its registered 1590; 1410 and 1500,
  # else the nearest supported clock one 15 MHz step away, labelled.
  if ! python3 - "$S/supported-clocks.txt" "$LOCK_TIMED" "${P1_LOCKS[@]}" "$P1_FALLBACK_1605" \
      > "$S/lock-plan.txt" <<'PY'
import re, sys
g = sorted({int(x) for x in re.findall(r"Graphics\s*:\s*(\d+)\s*MHz", open(sys.argv[1]).read())})
if not g:
    sys.exit("no graphics clock read: look at supported-clocks.txt")
timed, a, b, c, c_fb = map(int, sys.argv[2:7])
print(f"# {len(g)} graphics clocks, {g[0]} to {g[-1]} MHz")
for f in (timed, c, c_fb, b, a):
    print(f"# {f} " + ("supported" if f in g else
                       f"NOT supported; nearest {[s for s in g if abs(s - f) <= 15]}"))
def near(f):
    c = [s for s in g if abs(s - f) <= 15]
    return min(c, key=lambda s: (abs(s - f), -s)) if c else None
out = {"LOCK_TIMED_OK": 1 if timed in g else 0}
for name, f in (("LOCK_ETA_A", a), ("LOCK_ETA_B", b)):
    if f in g:
        out[name], note = f, "registered"
    elif near(f):
        out[name], note = near(f), f"SUBSTITUTED for {f}: P1 at it is computed after registration"
    else:
        out[name], note = "none", f"{f} and every clock within 15 MHz unsupported: dropped"
    print(f"# {name} {out[name]} ({note})")
if c in g:
    out["LOCK_ETA_C"], note = c, "registered"
elif c_fb in g:
    out["LOCK_ETA_C"], note = c_fb, f"the registered fallback for {c}"
elif near(c):
    out["LOCK_ETA_C"], note = near(c), f"SUBSTITUTED for {c}: P1 at it is computed after registration"
else:
    out["LOCK_ETA_C"], note = "none", f"{c}, {c_fb} and every clock within 15 MHz unsupported: dropped"
print(f"# LOCK_ETA_C {out['LOCK_ETA_C']} ({note})")
for k, v in out.items():
    print(f"{k}={v}")
PY
  then
    ledger "prelude REFUSED: the supported-clock list could not be read (lock-plan.txt)"; return "$EXIT_REFUSED"
  fi
  grep '^#' "$S/lock-plan.txt" | sed 's/^# //' | tee "$S/supported-locks.txt"
  grep -v '^#' "$S/lock-plan.txt" > "$D/locks.env"
  local line; while IFS= read -r line; do ledger "lock plan: $line"; done < <(grep '^# LOCK_' "$S/lock-plan.txt" | sed 's/^# //')
  if ! grep -qx 'LOCK_TIMED_OK=1' "$D/locks.env"; then
    ledger "prelude REFUSED: $LOCK_TIMED MHz is not a supported graphics clock: every step locks it"; return "$EXIT_REFUSED"
  fi
  nvidia-smi -q -d POWER > "$S/power-at-start.txt"   # both scopes, before any change
  sudo -n nvidia-smi -pm 1 || { ledger "prelude REFUSED: persistence mode not set"; return "$EXIT_REFUSED"; }
  sudo -n nvidia-smi -pl "$POWER_LIMIT_W" -sc 0 \
    || { ledger "prelude REFUSED: the GPU power limit was not set to $POWER_LIMIT_W W"; return "$EXIT_REFUSED"; }
  nvidia-smi -q -d POWER > "$S/power-set.txt"
  nvidia-smi --query-gpu=persistence_mode,clocks.sm,clocks.max.sm,power.limit,power.default_limit,clocks_event_reasons.active \
    --format=csv | tee "$S/clocks-at-start.txt"
  local pm pl
  IFS=', ' read -r pm pl < <(nvidia-smi --query-gpu=persistence_mode,power.limit --format=csv,noheader,nounits | head -1)
  if [[ "$pm" != Enabled ]]; then
    ledger "prelude REFUSED: persistence_mode reads '$pm', not Enabled"; return "$EXIT_REFUSED"
  fi
  if ! awk -v p="$pl" -v w="$POWER_LIMIT_W" 'BEGIN { exit !(p + 0 > w - 1 && p + 0 < w + 1) }'; then
    ledger "prelude REFUSED: power.limit reads '$pl' W, not $POWER_LIMIT_W"; return "$EXIT_REFUSED"
  fi
  ledger "prelude: persistence Enabled, power.limit $pl W, base tag $B"
  return "$EXIT_DONE"
}

# ==========================================================================
# 3c.2 byte pages at the 1710 lock
# ==========================================================================
counters_dir() {
  local R
  if [[ -s "$S/counters-dir.txt" ]]; then R="$(cat "$S/counters-dir.txt")"
  else R="$RESULTS_ROOT/$(date -u +%F)-$MOE_CARD-r3-counters"; echo "$R" > "$S/counters-dir.txt"; fi
  mkdir -p "$R/base"
  printf '%s\n' "$R"
}

step_bytes() {
  local R F C L G tail_dropped=0
  R="$(counters_dir)"; F="$LOCK_TIMED"; L="$R/lock$F"
  C="$(census_for)" || return "$EXIT_REFUSED"
  mkdir -p "$L"
  [[ -s "$C" ]] || { ledger "bytes REFUSED: no census at $C (the preflight's PF6 writes it)"; return "$EXIT_REFUSED"; }
  dropped bytes_tail && tail_dropped=1
  local gs=()
  for G in "${BYTE_GS[@]}"; do
    [[ "$G" == 64 && "$tail_dropped" == 1 ]] || gs+=("$G")
  done
  (( tail_dropped )) && ledger "bytes: the G=64 page and the base-clock control are dropped (deadline)"
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "the GPU is in use: stop"; exit 2; }
    moe_counter ncu --clock-control reset
    MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
    [ "$F" -le "$MAX" ] || { echo "F=$F MHz is above this card's maximum, $MAX MHz"; exit 2; }
    sudo -n nvidia-smi -lgc "$F,$F" || exit 2
    for G in "${gs[@]}"; do
      if [ "$G" = 64 ] && [ -f "$D/bytes-tail-repriced" ]; then continue; fi
      LOG=$S/logs/r3c-g$G-lock$F.log
      t0=$(date +%s)
      moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --group-m "$G" \
        --model "$MODEL" --tiles "$TREADS" --census "$C" --page-clock none --page-lock-mhz "$F" \
        --out "$L/r3c-g$G.json" 2>&1 | tee "$LOG"
      rc=${PIPESTATUS[0]}
      secs=$(( $(date +%s) - t0 ))
      echo "G=$G exit $rc $(date -u +%T) ${secs} s" | tee -a "$S/logs/lock$F-pages.status"
      failed=$(grep -E '^RESULT: VALIDITY [^ ]+ FAIL' "$LOG" | awk '{print $3}' | sort -u | tr '\n' ' ')
      case $rc in
        0|1) ;;
        3) if [ -f "$L/r3c-g$G.json" ] && [ "$G" -ge 32 ] && [ "$failed" = "V7 " ]; then
             echo "G=$G INVALID on V7 alone, as expected at G >= 32: the page is written, go on"
           else
             echo "G=$G INVALID on ${failed:-no gate line}: the byte pages stop here; read $LOG"; exit 3
           fi ;;
        *) exit "$rc" ;;
      esac
      if [ "$G" = 1 ] && [ "$secs" -gt "$BYTE_G1_REPRICE_S" ]; then
        touch "$D/bytes-tail-repriced"
        echo "G=1 took $secs s, over $BYTE_G1_REPRICE_S: the G=64 page and the base-clock control are dropped"
      fi
    done )
  local rc=$?
  ledger "bytes: lock block exit $rc ($(tr '\n' ';' < "$S/logs/lock$F-pages.status" 2>/dev/null))"
  [[ -f "$D/bytes-tail-repriced" ]] && tail_dropped=1 \
    && ledger "bytes: G=1 took over $BYTE_G1_REPRICE_S s: the G=64 page and the base-clock control dropped"
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks_event_reasons.active,persistence_mode \
    --format=csv | tee "$S/clocks-after-lock-pages.txt"
  if (( ! tail_dropped )); then
    # the base-clock control: same board, commit and treads, at ncu's base clock
    moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --group-m 2 \
      --model "$MODEL" --tiles "$TREADS" \
      --census "$C" --out "$R/base/r3c-g2.json" 2>&1 | tee "$S/logs/r3c-g2-base.log"
    ledger "bytes: base-clock control G=2 exit ${PIPESTATUS[0]}"
    moe_counter ncu --clock-control reset
  fi
  local timed=() id p
  [[ "$MODEL" == "$DEFAULT_MODEL" ]] || ledger "bytes: no C5 references (the 2026-09-25 timed pages are $DEFAULT_MODEL's)"
  for id in "${TIMED_REF_IDS[@]}"; do
    [[ "$MODEL" == "$DEFAULT_MODEL" ]] || break
    for p in "$REPO/$TIMED_REF_DIR"/*"$id"/report.json; do
      if [[ -f "$p" ]]; then timed+=("$p"); else ledger "bytes: timed reference *$id is missing from the checkout"; fi
    done
  done
  local pages=()
  for p in "$L"/r3c-g*.json; do [[ -f "$p" ]] && pages+=("$p"); done
  if (( ${#pages[@]} )); then
    python3 scripts/dram_counter_route.py --analyse "${pages[@]}" \
      ${timed[@]+--timed-reference "${timed[@]}"} --out "$L/summary.json" 2>&1 | tee "$S/logs/analyse-lock$F.log"
    ledger "bytes: analyse of ${#pages[@]} pages exit ${PIPESTATUS[0]} ($L/summary.json)"
  else
    ledger "bytes: no page written, nothing to analyse"
  fi
  return "$rc"
}

# ==========================================================================
# 3c.3 calibrate, with no lock in force
# ==========================================================================
step_calibrate() {
  export MOE_RESULTS_DIR="$RESULTS_ROOT/gaps-$MOE_CARD"
  local busy; busy="$(gpu_busy)"
  [[ -z "$busy" ]] || { ledger "calibrate REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }
  moe_counter ncu --clock-control reset; sudo -n nvidia-smi -rgc
  nvidia-smi -q -d POWER > "$S/power-before-calibrate.txt"
  "$PY_BASE" scripts/calibrate_hardware.py --publish --results-root "$RESULTS_ROOT" 2>&1 | tee "$S/logs/calibrate.log"
  local rc=${PIPESTATUS[0]} Y="moe/bench/hardware/measured_$MOE_CARD.yaml" kind why
  grep '^RESULT:' "$S/logs/calibrate.log" | cut -c1-160 | while IFS= read -r l; do ledger "calibrate: $l"; done
  if (( rc == EXIT_DONE )) && [[ -f "$Y" ]]; then
    kind=measured; why="calibrate exit 0"
  elif (( rc == EXIT_CLAIM_FAIL )) && grep -qE '^RESULT: CLAIM not_throttled FAIL' "$S/logs/calibrate.log"; then
    kind=fallback; why="calibrate exit 1 with not_throttled FAILED"
  elif (( rc == EXIT_CLAIM_FAIL )) && [[ -f "$Y" ]]; then
    kind=measured
    why="calibrate exit 1, not_throttled did not fail (failed: $(grep -E '^RESULT: [A-Z]+ [^ ]+ FAIL' "$S/logs/calibrate.log" | awk '{print $3}' | tr '\n' ' '))"
  else
    kind=fallback; why="calibrate exit $rc$([[ -f "$Y" ]] || echo ', no ruler written')"
  fi
  if [[ "$kind" == measured ]]; then
    cp "$Y" "$S/"
    printf 'RULER_KIND=measured\nRULER_ARGS=()\n' > "$D/ruler.env"
    ledger "calibrate: the ruler stands ($why): $Y"
  else
    rm -f "$Y"
    { printf 'RULER_KIND=fallback\nRULER_ARGS=('; printf '%q ' "${RULER_FALLBACK[@]}"; printf ')\n'; } > "$D/ruler.env"
    ledger "calibrate: FALLBACK ($why): the ruler is removed and R3 takes ${RULER_FALLBACK[*]}, the 2026-09-25 ruler's"
  fi
  git status --porcelain | tee "$S/git-after-calibrate.txt"   # the ruler, untracked, when it stands
  return "$rc"
}

# ==========================================================================
# 3c.3 timed R3 at 1710 (P2, P5)
# ==========================================================================
#: R3's dry run of one design, logged, then checked: prints the reason when the
#: design may not run.
r3_dry() {   # LOG PLANNED TAG G ARGS...
  local log="$1" planned="$2" tag="$3" g="$4"; shift 4
  "$PY_VLLM" scripts/private_weight_reference.py "$@" --group-m "$g" --session-tag "$tag" --dry-run \
    > "$log" 2>&1
  grep -E '^REFUSED:|^ *ridge |n_decl = [0-9]+ against|retracted +tread|predicted peak' "$log" | cut -c1-160
  local why
  if why="$(check_dry "$log" "$planned")"; then return 0; fi
  ledger "dry run G=$g treads $planned may not lock: $why ($log)"
  return 1
}

step_timed() {
  export MOE_RESULTS_DIR="$RESULTS_ROOT/gaps-$MOE_CARD"
  load_ruler
  local R6=("${R3_BASE[@]}" --treads 6 ${RULER_ARGS[@]+"${RULER_ARGS[@]}"})
  local R8=("${R3_BASE[@]}" --treads 8 ${RULER_ARGS[@]+"${RULER_ARGS[@]}"})
  local T2 T5; T2="$(fresh_tag "$B-p2")"; T5="$(fresh_tag "$B-p5")"
  local busy; busy="$(gpu_busy)"
  [[ -z "$busy" ]] || { ledger "timed REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }
  moe_counter ncu --clock-control reset; sudo -n nvidia-smi -rgc
  local p2=() G
  for G in 8 32; do
    r3_dry "$S/logs/r3-g$G-t6-dry.log" 6 "$T2-lock$LOCK_TIMED" "$G" "${R6[@]}" && p2+=("$G")
  done
  local p5=0
  r3_dry "$S/logs/r3-g3-t8-dry.log" 8 "$T5-lock$LOCK_TIMED" 3 "${R8[@]}" && p5=1
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    rc2=0; rc5=0
    if (( ${#p2[@]} )); then
      python3 scripts/locked_r3.py --session-tag "$T2" --groups "${p2[@]}" --locks "$LOCK_TIMED" $(lr3_cap) -- "${R6[@]}" 2>&1 \
        | tee -i "$S/logs/locked_r3-$T2.log"
      rc2=${PIPESTATUS[0]}
      echo "P2 locked_r3 exit $rc2" >> "$D/timed-results"
      (( rc2 >= 4 )) && exit "$rc2"   # a crash, or the card left locked
    else
      echo "P2 skipped: no design passed its dry run" >> "$D/timed-results"; rc2=3
    fi
    if (( p5 )); then
      python3 scripts/locked_r3.py --session-tag "$T5" --groups 3 --locks "$LOCK_TIMED" $(lr3_cap) -- "${R8[@]}" 2>&1 \
        | tee -i "$S/logs/locked_r3-$T5.log"
      rc5=${PIPESTATUS[0]}
      echo "P5 locked_r3 exit $rc5" >> "$D/timed-results"
    else
      echo "P5 skipped: its design did not pass its dry run" >> "$D/timed-results"; rc5=3
    fi
    exit $(( rc2 > rc5 ? rc2 : rc5 )) )
  local rc=$?
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active,power.limit --format=csv \
    | tee "$S/clocks-after-timed.txt"
  local l; while IFS= read -r l; do ledger "timed: $l"; done < <(
    cat "$D/timed-results" 2>/dev/null; ledger_lines "$S/locked-r3/$T2/status" "$S/locked-r3/$T5/status")
  : > "$D/timed-results"
  return "$rc"
}

# ==========================================================================
# 3c.4 held locks below 1710 (P1)
# ==========================================================================
step_eta() {
  export MOE_RESULTS_DIR="$RESULTS_ROOT/gaps-$MOE_CARD"
  load_ruler; load_locks
  local R6=("${R3_BASE[@]}" --treads 6 ${RULER_ARGS[@]+"${RULER_ARGS[@]}"})
  local ok4=0 ok2=0
  r3_dry "$S/logs/r3-eta-g4-dry.log" 6 "$B-eta-check" 4 "${R6[@]}" && ok4=1
  r3_dry "$S/logs/r3-eta-g2-dry.log" 6 "$B-eta-check" 2 "${R6[@]}" && ok2=1
  # 1410 first: it alone answers the falsifier. Each lock is its own run, since
  # locked_r3.py stops at the first lock that holds.
  local specs=() gs
  if [[ "$LOCK_ETA_A" != none ]]; then
    gs=""; (( ok4 )) && gs="4"; (( ok2 )) && gs="$gs 2"
    [[ -n "${gs# }" ]] && specs+=("$LOCK_ETA_A:${gs# }")
  fi
  (( ok4 )) && [[ "$LOCK_ETA_B" != none ]] && specs+=("$LOCK_ETA_B:4")
  if (( ok4 )) && [[ "$LOCK_ETA_C" != none ]]; then
    if dropped eta_1605; then ledger "eta: the $LOCK_ETA_C lock is dropped (deadline)"
    else specs+=("$LOCK_ETA_C:4"); fi
  fi
  (( ${#specs[@]} )) || { ledger "eta: no lock to run (dry runs or supported clocks)"; return "$EXIT_INVALID"; }
  local busy; busy="$(gpu_busy)"
  [[ -z "$busy" ]] || { ledger "eta REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }
  : > "$D/eta-results.new"
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    moe_counter ncu --clock-control reset; sudo -n nvidia-smi -rgc
    worst=0
    for spec in "${specs[@]}"; do
      F=${spec%%:*}; GS=${spec#*:}
      for try in 1 2; do
        tag=$(fresh_tag "$B-eta$F") || { echo "no fresh tag for $B-eta$F"; exit 4; }
        # shellcheck disable=SC2086
        python3 scripts/locked_r3.py --session-tag "$tag" --locks "$F" --groups $GS $(lr3_cap) -- "${R6[@]}" 2>&1 \
          | tee -i "$S/logs/eta-lock$F-$tag.log"
        rc=${PIPESTATUS[0]}
        echo "$F $rc $tag" >> "$D/eta-results.new"
        # Exit 1 this far under 1710 more likely means a lock left over than the
        # power cap: reset, record the clocks, and try once more under a new tag.
        if (( rc == 1 && try == 1 )); then
          sudo -n nvidia-smi -rgc
          nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv \
            > "$S/clocks-after-eta-$tag.txt"
          continue
        fi
        break
      done
      (( rc > worst )) && worst=$rc
      (( rc >= 4 )) && exit "$rc"
    done
    exit "$worst" )
  local rc=$?
  cat "$D/eta-results.new" >> "$D/eta-results"
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv | tee "$S/clocks-after-eta.txt"
  local l tag; while IFS= read -r l; do
    tag="${l##* }"; ledger "eta: lock ${l%% *} locked_r3 exit $(echo "$l" | awk '{print $2}') ($tag)"
    ledger_lines "$S/locked-r3/$tag/status" | while IFS= read -r x; do ledger "eta: $x"; done
  done < "$D/eta-results.new"
  rm -f "$D/eta-results.new"
  return "$rc"
}

# ==========================================================================
# 3c.5 floor counters (P6)
# ==========================================================================
step_floor() {
  local R C F=$LOCK_TIMED; R="$(counters_dir)"
  C="$(census_for)" || return "$EXIT_REFUSED"
  [[ -s "$C" ]] || { ledger "floor REFUSED: no census at $C"; return "$EXIT_REFUSED"; }
  : > "$D/floor-results"
  ( set -o pipefail
    SMI=
    trap 'trap "" INT TERM HUP; [ -n "${SMI:-}" ] && kill "$SMI" 2>/dev/null; moe_counter ncu --clock-control reset >/dev/null 2>&1; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "the GPU is in use: stop"; exit 2; }
    moe_counter ncu --clock-control reset
    sudo -n nvidia-smi -rgc >/dev/null || exit 2
    worst=0
    # 2 is refused (census, commit or door) and would refuse every capture;
    # 3 is INVALID with the file written, and the next capture still answers.
    capture() {   # NAME, then dram_counter_route.py's arguments
      local name=$1; shift
      moe_counter "$PY_VLLM" scripts/dram_counter_route.py --run --family r3-arms --model "$MODEL" "$@" 2>&1 \
        | tee "$S/logs/$name.log"
      local rc=${PIPESTATUS[0]}
      echo "$name exit $rc" >> "$D/floor-results"
      (( rc > worst )) && worst=$rc
      case $rc in 0|3) return 0 ;; *) exit "$rc" ;; esac
    }
    capture r3f-g64 --group-m 64 --census "$C" --floor --out "$R/r3f-g64.json"
    passes=$(grep -oE -- '- [0-9]+ pass(es)?' "$S/logs/r3f-g64.log" | awk '{print $2}' | sort -n | tail -1)
    if [ "${passes:-0}" -gt "$FLOOR_MAX_PASSES" ]; then
      echo "r3f-g2 dropped: the G=64 capture took $passes replay passes a launch, over $FLOOR_MAX_PASSES" >> "$D/floor-results"
    else
      capture r3f-g2 --group-m 2 --census "$C" --floor --out "$R/r3f-g2.json"
    fi
    moe_counter ncu --clock-control reset
    # a RECORD, scoring nothing: P7 is registered for the duty-0.25 timed regime
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
      --format=csv,noheader,nounits -lms 100 > "$S/logs/r3f-g64-unlocked.smi.csv" 2>&1 & SMI=$!
    capture r3f-g64-unlocked --group-m 64 --census "$C" --floor --floor-clock none --out "$R/r3f-g64-unlocked.json"
    kill "$SMI" 2>/dev/null; SMI=
    moe_counter ncu --clock-control reset
    MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
    [ "$F" -le "$MAX" ] || { echo "F=$F MHz is above this card's maximum, $MAX MHz"; exit 2; }
    sudo -n nvidia-smi -lgc "$F,$F" || exit 2
    capture "r3f-g64-lock$F" --group-m 64 --census "$C" --floor --floor-clock none --floor-lock-mhz "$F" \
      --out "$R/r3f-g64-lock$F.json"
    exit "$worst" )
  local rc=$?
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks_event_reasons.active --format=csv \
    | tee "$S/clocks-after-floor.txt"
  local l; while IFS= read -r l; do ledger "floor: $l"; done < "$D/floor-results"
  return "$rc"
}

# ==========================================================================
# 3c.6 timed R3 to tread 9 (P3, P4)
# ==========================================================================
step_deep() {
  export MOE_RESULTS_DIR="$RESULTS_ROOT/gaps-$MOE_CARD"
  load_ruler
  local R9=("${R3_BASE[@]}" --treads 9 ${RULER_ARGS[@]+"${RULER_ARGS[@]}"})
  local T; T="$(fresh_tag "$B-deep")"
  moe_counter ncu --clock-control reset; sudo -n nvidia-smi -rgc
  local busy; busy="$(gpu_busy)"
  [[ -z "$busy" ]] || { ledger "deep REFUSED: the GPU is in use: $busy"; return "$EXIT_REFUSED"; }
  local gs=() G
  for G in 4 2; do   # G=4 first: a run cut short still answers P4
    r3_dry "$S/logs/r3-deep-g$G-dry.log" 9 "$T-lock$LOCK_TIMED" "$G" "${R9[@]}" && gs+=("$G")
  done
  (( ${#gs[@]} )) || { ledger "deep: no design passed its dry run"; return "$EXIT_INVALID"; }
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    python3 scripts/locked_r3.py --session-tag "$T" --locks "$LOCK_TIMED" --groups "${gs[@]}" $(lr3_cap) -- "${R9[@]}" 2>&1 \
      | tee -i "$S/logs/locked_r3-$T.log" )
  local rc=$?
  ledger "deep: locked_r3 exit $rc ($T, G = ${gs[*]})"
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv | tee "$S/clocks-after-deep.txt"
  local l; while IFS= read -r l; do ledger "deep: $l"; done < <(ledger_lines "$S/locked-r3/$T/status")
  # P3 and P4 off the two pages: read only when both show every VALIDITY gate
  # passing and the 9-copy declaration.
  "$PY_VLLM" - "$MOE_RESULTS_DIR/private_weight_reference" "$T-lock$LOCK_TIMED" > "$S/logs/deep-check-$T.txt" 2>&1 <<'PY'
import glob, json, statistics, sys
root, tag = sys.argv[1:3]; ms = {}
for p in sorted(glob.glob(f"{root}/*/report.json")):
    j = json.load(open(p))
    if j.get("session_tag") != tag:
        continue
    G = j["pinned"]["GROUP_SIZE_M"]
    bad = [f"{g.get('tag')} {g.get('verdict')}" for g in j["gates"]
           if g.get("kind") == "VALIDITY" and g.get("verdict") != "PASS"]
    print(f"G={G} treads 1..{j['treads'][-1]} declared {j['copies_declared']} "
          f"VALIDITY not PASS: {bad or 'none'}")
    ms.update({(G, r["tiles"]): r["ms_p50"] for r in j["treads_table"] if r["arm"] == "shared"})
for G in (2, 4):
    t = [ms[(G, n)] for n in range(1, 10) if (G, n) in ms]
    print(f"G={G} SHARED ms", " ".join(f"{v:.4f}" for v in t),
          "| increments", " ".join(f"{b - a:.3f}" for a, b in zip(t, t[1:])))
if all((g, n) in ms for g in (2, 4) for n in (3, 5, 7, 9)):
    print("P3 G=2 minus G=4 at n = 3 5 7 9:",
          " ".join(f"{ms[(2, n)] - ms[(4, n)]:.3f}" for n in (3, 5, 7, 9)), "ms")
if all((4, n) in ms for n in range(2, 10)):
    inc = [ms[(4, n + 1)] - ms[(4, n)] for n in range(2, 9)]
    print(f"P4 G=4 n=8 to 9: {inc[-1]:.3f} ms against the median of the others, "
          f"{statistics.median(inc[:-1]):.3f}")
PY
  while IFS= read -r l; do ledger "deep: $l"; done < "$S/logs/deep-check-$T.txt"
  return "$rc"
}

# ==========================================================================
# 3c.7 R1 in lock mode, last
# ==========================================================================
step_r1lock() {
  export MOE_RESULTS_DIR="$RESULTS_ROOT/gaps-$MOE_CARD"
  load_locks
  # Every lock R1 takes held in 3c.4; a lock that did not is replaced by the
  # ones that did. Before 3c.4 has run, its planned locks.
  local locks=("$LOCK_TIMED") f
  if [[ -s "$D/eta-results" ]]; then
    for f in "$LOCK_ETA_B" "$LOCK_ETA_A"; do
      [[ "$f" != none ]] && awk -v f="$f" '$1 == f && $2 == 0 { found = 1 } END { exit !found }' "$D/eta-results" \
        && locks+=("$f")
    done
  else
    for f in "$LOCK_ETA_B" "$LOCK_ETA_A"; do [[ "$f" != none ]] && locks+=("$f"); done
  fi
  if (( ${#locks[@]} < 2 )) || (( locks[${#locks[@]} - 1] > R1_SPAN_MAX_LOW )); then
    ledger "r1lock: no held lock at or under $R1_SPAN_MAX_LOW MHz (locks ${locks[*]}): R1 cannot span its claim; not run"
    return "$EXIT_INVALID"
  fi
  local gs=() G
  dropped r1_g4 && ledger "r1lock: R1 at G=4 is dropped (deadline)" || gs+=(4)
  dropped r1_g1 && ledger "r1lock: R1 at G=1 is dropped (deadline)" || gs+=(1)
  (( ${#gs[@]} )) || return "$EXIT_INVALID"
  local tag; tag="$(fresh_tag "$B-r1lock")"
  local R1=("${R1_BASE[@]}" --session-tag "$tag")
  for G in "${gs[@]}"; do
    "$PY_VLLM" scripts/clock_elasticity.py --dry-run --group-m "$G" --lock-clocks "${locks[@]}" "${R1[@]}" \
      > "$S/logs/r1lock-g$G.plan.log" 2>&1
    echo "plan G=$G exit $? (2 expected)"
  done
  nvidia-smi -q -d POWER > "$S/power-before-r1lock.txt"
  : > "$D/r1-results"
  ( set -o pipefail
    trap 'trap "" INT TERM HUP; sudo -n nvidia-smi -rgc >/dev/null' EXIT
    trap 'exit 130' INT TERM HUP
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "the GPU is in use: stop"; exit 2; }
    moe_counter ncu --clock-control reset
    sudo -n nvidia-smi -rgc
    for G in "${gs[@]}"; do
      "$PY_VLLM" scripts/clock_elasticity.py --group-m "$G" --lock-clocks "${locks[@]}" "${R1[@]}" \
        2>&1 | tee "$S/logs/r1lock-g$G.log"
      rc=${PIPESTATUS[0]}; echo "r1lock G=$G exit $rc (locks ${locks[*]})" >> "$D/r1-results"
      [ "$rc" -le 1 ] || exit "$rc"   # 0 DONE, 1 CLAIM_FAIL: go on; 2, 3, 4: stop
    done )
  local rc=$?
  sleep 5
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active --format=csv | tee "$S/clocks-after-r1lock.txt"
  local l; while IFS= read -r l; do ledger "r1lock: $l"; done < "$D/r1-results"
  return "$rc"
}

# ==========================================================================
# the orchestrator
# ==========================================================================
push_results() {   # MESSAGE [FINAL_EXIT]
  (( NO_PUSH )) && return 0
  local p="$MOE_HOME/repo/scripts/vm_results_push.sh"
  [[ -f "$p" ]] || p="$(dirname "$SELF")/vm_results_push.sh"
  local rc=0
  timeout -k 30 900 bash "$p" push --message "$1" ${2:+--final "$2"} || rc=$?
  (( rc == 0 )) || ledger "push of '$1' exited $rc (results stay on the VM; the next push carries them)"
  return 0
}

#: Between steps: clear both kinds of lock, wait for an idle card, record.
hygiene() {
  moe_counter ncu --clock-control reset >/dev/null 2>&1 || true
  if ! sudo -n nvidia-smi -rgc >/dev/null 2>&1; then
    ledger "HYGIENE: nvidia-smi -rgc failed: the card may still be locked"; return 1
  fi
  local waited=0 busy
  while busy="$(gpu_busy)"; [[ -n "$busy" ]]; do
    if (( waited >= BUSY_WAIT_S )); then
      ledger "HYGIENE: the GPU is still in use ${BUSY_WAIT_S} s after the step: $busy"; return 2
    fi
    sleep 10; waited=$(( waited + 10 ))
  done
  ledger "clocks: $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.limit,clocks_event_reasons.active \
    --format=csv,noheader 2>&1 | head -1)"
  return 0
}

left_min() { echo $(( (DEADLINE - $(now_s)) / 60 - RESERVE_MIN )); }

est_now() {   # the step's estimate after its drops
  local s="$1" m d; m="$(step_est "$s")"
  for d in "${DROPS[@]}"; do
    [[ "$(drop_step "$d")" == "$s" ]] && dropped "$d" && m=$(( m - $(drop_min "$d") ))
  done
  echo "$m"
}

#: Before step i: drop in the draft's order while the rest overruns the time left.
budget() {   # INDEX
  local i="$1" j need left d
  while :; do
    need=0
    for (( j = i; j < ${#PLAN[@]}; j++ )); do need=$(( need + $(est_now "${PLAN[$j]}") )); done
    left="$(left_min)"
    (( need <= left )) && return 0
    local applied=0
    for d in "${DROPS[@]}"; do
      dropped "$d" && continue
      local ds; ds="$(drop_step "$d")"
      for (( j = i; j < ${#PLAN[@]}; j++ )); do
        if [[ "${PLAN[$j]}" == "$ds" ]]; then
          echo "$d" >> "$D/drops"
          ledger "DROPPED $(drop_what "$d"): ${need} min of steps left, ${left} min to the deadline"
          applied=1; break
        fi
      done
      (( applied )) && break
    done
    (( applied )) || return 0
  done
}

STEP_PID=""
on_signal() {
  local sig="$1"
  trap '' INT TERM HUP
  ledger "STOPPED by SIG$sig during ${CURRENT:-no step}: stopping the step, then the reset"
  [[ -n "$STEP_PID" ]] && { kill -TERM "$STEP_PID" 2>/dev/null; wait "$STEP_PID" 2>/dev/null; }
  hygiene || true
  push_results "stopped by SIG$sig during ${CURRENT:-no step}" 143
  exit 143
}

run_step() {   # NAME CAP_S
  local name="$1" cap="$2" rc
  local log="$S/logs/driver-$name.log"
  timeout -k 240 "$cap" bash "$SELF" --step "$name" >> "$log" 2>&1 < /dev/null &
  STEP_PID=$!
  wait "$STEP_PID"; rc=$?
  STEP_PID=""
  return "$rc"
}

print_plan() {
  local s total=0
  echo "THE GH200 MODEL-TEST SESSION (docs/LAMBDA.md section 3c), unattended"
  [[ "$MODEL" == "$DEFAULT_MODEL" ]] || echo "  MODEL $MODEL: a cross-model test (docs/registered/README.md); its own census, estimates x$(model_scale_pct)%, no eta or R1 unless --steps asks"
  echo "  owner, 2026-09-26: byte treads $TREADS; calibrate after the byte pages; GPU power limit"
  echo "  ${POWER_LIMIT_W} W (-sc 0); r570 upgraded to 580.105.08 before setup (scripts/vm_run.sh)"
  printf '  %-10s %4s %4s  %s\n' step est cap what
  for s in "${PLAN[@]}"; do
    printf '  %-10s %4s %4s  %s\n' "$s" "$(step_est "$s")" "$(step_cap "$s")" "$(step_what "$s")"
    total=$(( total + $(step_est "$s") ))
  done
  echo "  $total min of steps, estimated, plus setup_vm.sh (~30 min) when --setup; about $(( (total + 30) / 60 )) h"
  echo "  deadline: $([[ -n "$DEADLINE" ]] && utc_of "$DEADLINE" || echo 'none given (the run itself needs --deadline)')"
  echo "  drop order when time runs short: R1 at G=1, R1 at G=4, the 1605 lock, the G=64 byte page and"
  echo "  the base-clock control; a step whose own estimate no longer fits is skipped"
  echo "  pushes: after every step, to the run's branch (scripts/vm_results_push.sh)$( (( NO_PUSH )) && echo ': OFF (--no-push)')"
  echo "COMMANDS (\$S, \$R and \$B as on the VM; R3 takes ${RULER_FALLBACK[*]} only if calibrate fails):"
  echo "  bytes   moe_counter \$PY_VLLM scripts/dram_counter_route.py --run --family r3-arms --group-m G --tiles $TREADS"
  echo "            --census \$S/census.json --page-clock none --page-lock-mhz $LOCK_TIMED --out \$R/lock$LOCK_TIMED/r3c-gG.json"
  echo "            for G in ${BYTE_GS[*]}, under nvidia-smi -lgc $LOCK_TIMED,$LOCK_TIMED; then G=2 at ncu's base clock"
  echo "  cal     \$PY_BASE scripts/calibrate_hardware.py --publish --results-root \$RESULTS_ROOT"
  echo "  timed   python3 scripts/locked_r3.py --session-tag \$B-p2 --groups 8 32 --locks $LOCK_TIMED -- ${R3_BASE[*]} --treads 6"
  echo "          python3 scripts/locked_r3.py --session-tag \$B-p5 --groups 3 --locks $LOCK_TIMED -- ${R3_BASE[*]} --treads 8"
  echo "  eta     python3 scripts/locked_r3.py --session-tag \$B-eta<F> --locks <F> --groups <G> -- ${R3_BASE[*]} --treads 6"
  echo "            for 1410:4 2, 1500:4, 1605:4 (1590 if 1605 is not supported)"
  echo "  floor   moe_counter \$PY_VLLM scripts/dram_counter_route.py --run --family r3-arms --floor --census \$S/census.json"
  echo "            G=64 and G=2 at base; G=64 --floor-clock none (unlocked record); G=64 --floor-clock none --floor-lock-mhz $LOCK_TIMED"
  echo "  deep    python3 scripts/locked_r3.py --session-tag \$B-deep --locks $LOCK_TIMED --groups 4 2 -- ${R3_BASE[*]} --treads 9"
  echo "  r1lock  \$PY_VLLM scripts/clock_elasticity.py --group-m G --lock-clocks 1710 1500 1410 ${R1_BASE[*]} --session-tag \$B-r1lock"
  echo "            for G in 4 1"
  echo "Every timed design runs R3's own --dry-run first and locks only when it prints no REFUSED line,"
  echo "'n_decl = 9 against n_max', a retracted tread of at least the treads planned, and a ruler."
}

usage() { sed -n '2,7p' "$SELF" | sed 's/^# \{0,1\}//'; }

# ---- arguments -----------------------------------------------------------
DRY=0; NO_PUSH=0; SETUP=0; COMMIT=""; REPO_URL=""; DEADLINE=""; FROM=""; ONLY=""; ONE_STEP=""
ARGS=("$@")
while (( $# )); do
  case "$1" in
    --dry-run)  DRY=1; shift ;;
    --no-push)  NO_PUSH=1; shift ;;
    --setup)    SETUP=1; shift ;;
    --commit)   COMMIT="${2:-}"; shift 2 ;;
    --repo)     REPO_URL="${2:-}"; shift 2 ;;
    --deadline) DEADLINE="${2:-}"; shift 2 ;;
    --from)     FROM="${2:-}"; shift 2 ;;
    --steps)    ONLY="${2:-}"; shift 2 ;;
    --model)    MODEL="${2:-}"; shift 2 ;;
    --step)     ONE_STEP="${2:-}"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit "$EXIT_REFUSED" ;;
  esac
done

is_step() { local s; for s in "${STEPS[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }

[[ "$MODEL" =~ ^[a-z0-9][a-z0-9.-]*$ ]] || { echo "--model $MODEL: not a model name" >&2; exit "$EXIT_REFUSED"; }
export MOE_DRIVER_MODEL="$MODEL"
R3_BASE[1]="$MODEL"; R1_BASE[1]="$MODEL"

# One step, in its own process: the orchestrator runs this under a cap.
if [[ -n "$ONE_STEP" ]]; then
  is_step "$ONE_STEP" || { echo "no step $ONE_STEP (${STEPS[*]})"; exit "$EXIT_REFUSED"; }
  enter "$ONE_STEP"
  CURRENT="$ONE_STEP"
  echo "== $(utc) step $ONE_STEP: $(step_what "$ONE_STEP")"
  "step_$ONE_STEP"
  exit $?
fi

PLAN=()
if [[ -n "$ONLY" ]]; then
  for s in ${ONLY//,/ }; do is_step "$s" || { echo "no step $s (${STEPS[*]})" >&2; exit "$EXIT_REFUSED"; }; done
  for s in "${STEPS[@]}"; do [[ ",$ONLY," == *",$s,"* ]] && PLAN+=("$s"); done
elif [[ -n "$FROM" ]]; then
  is_step "$FROM" || { echo "no step $FROM (${STEPS[*]})" >&2; exit "$EXIT_REFUSED"; }
  on=0; for s in "${STEPS[@]}"; do [[ "$s" == "$FROM" ]] && on=1; (( on )) && PLAN+=("$s"); done
elif [[ "$MODEL" != "$DEFAULT_MODEL" ]]; then
  # P1's locks and R1 answered this card's clock question on 8x7B; another
  # model's session measures its bytes, times and floor (docs/registered)
  for s in "${STEPS[@]}"; do [[ "$s" == eta || "$s" == r1lock ]] || PLAN+=("$s"); done
else
  PLAN=("${STEPS[@]}")
fi

if (( DRY )); then
  print_plan
  echo "DRY RUN: nothing ran, nothing was written (exit 2, as every dry run in this repo)"
  exit "$EXIT_REFUSED"
fi

[[ "$DEADLINE" =~ ^[0-9]+$ ]] || { echo "--deadline <epoch seconds> is required (the laptop sets launch + 400 min)" >&2; exit "$EXIT_REFUSED"; }
command -v timeout >/dev/null || { echo "no timeout(1) on PATH: each step runs under a cap" >&2; exit "$EXIT_REFUSED"; }

# ---- setup: setup_vm.sh, then the checkout's own copy of this driver -------
if (( SETUP )); then
  [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "--setup needs --commit <full sha>" >&2; exit "$EXIT_REFUSED"; }
  [[ -n "$REPO_URL" ]] || { echo "--setup needs --repo <url>" >&2; exit "$EXIT_REFUSED"; }
  D="$MOE_HOME/session/gh200-driver"; mkdir -p "$D"
  SV="$(dirname "$SELF")/setup_vm.sh"
  [[ -f "$SV" ]] || { ledger "setup REFUSED: no $SV (scripts/vm_run.sh copies it beside this driver)"; exit "$EXIT_REFUSED"; }
  ledger "setup_vm.sh start: commit $COMMIT from $REPO_URL, deadline $(utc_of "$DEADLINE")"
  rc=0
  timeout -k 60 3600 bash "$SV" --repo "$REPO_URL" --commit "$COMMIT" > "$HOME/setup_vm.log" 2>&1 < /dev/null || rc=$?
  ledger "setup_vm.sh exit $rc ($(grep -cE '^RESULT: [A-Z]+ PF[0-9] PASS' "$HOME/setup_vm.log" 2>/dev/null || echo 0) preflight checks passed)"
  if (( rc != 0 )); then
    push_results "setup_vm.sh exit $rc: STOPPED before any measurement" "$rc"
    exit "$rc"
  fi
  push_results "setup_vm.sh: READY"
  NEXT="$MOE_HOME/repo/scripts/gh200_model_session.sh"
  [[ -f "$NEXT" ]] || { ledger "setup: no $NEXT after setup"; exit "$EXIT_ERROR"; }
  [[ "$(sha256_of "$NEXT")" == "$(sha256_of "$SELF")" ]] \
    || ledger "setup: the checkout's driver differs from the copy that ran setup; the checkout's runs"
  REST=()
  i=0
  while (( i < ${#ARGS[@]} )); do
    case "${ARGS[$i]}" in
      --setup) i=$(( i + 1 )) ;;
      --commit|--repo) i=$(( i + 2 )) ;;
      *) REST+=("${ARGS[$i]}"); i=$(( i + 1 )) ;;
    esac
  done
  exec bash "$NEXT" ${REST[@]+"${REST[@]}"}
fi

# ---- the run ---------------------------------------------------------------
[[ -f "$ENV_SH" ]] || { echo "no $ENV_SH: run setup_vm.sh first (or --setup)" >&2; exit "$EXIT_REFUSED"; }
# shellcheck disable=SC1090
. "$ENV_SH"
S="$SESSION_ROOT"; D="$S/gh200-driver"
mkdir -p "$S/logs" "$D"
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP
ledger "driver start: commit $(git -C "$REPO" rev-parse HEAD 2>/dev/null), steps ${PLAN[*]}, deadline $(utc_of "$DEADLINE") ($(left_min) min of work left)"
FINAL=$EXIT_DONE
: > "$D/summary.new"
for (( i = 0; i < ${#PLAN[@]}; i++ )); do
  CURRENT="${PLAN[$i]}"
  budget "$i"
  est="$(est_now "$CURRENT")"; left="$(left_min)"
  if (( est <= 0 || est > left )); then
    if (( est <= 0 )); then why="every part of it is dropped"
    else why="its estimate, $est min, is over the $left min left"; fi
    ledger "SKIPPED $CURRENT: $why"
    printf '%-10s SKIPPED (%s)\n' "$CURRENT" "$why" >> "$D/summary.new"
    (( FINAL < EXIT_INVALID )) && FINAL=$EXIT_INVALID
    if [[ "$CURRENT" == prelude ]]; then
      ledger "STOPPED: the prelude was skipped, and every step after it needs it"
      cp "$D/summary.new" "$D/summary.txt"
      push_results "prelude: SKIPPED, the session STOPPED" "$FINAL"
      exit "$FINAL"
    fi
    push_results "$CURRENT: SKIPPED ($why)"
    continue
  fi
  cap=$(( $(step_cap "$CURRENT") * 60 ))
  (( cap > left * 60 )) && cap=$(( left * 60 ))
  ledger "$CURRENT START: $(step_what "$CURRENT") (estimate $est min, cap $(( cap / 60 )) min, $left min left)"
  t0=$(now_s)
  run_step "$CURRENT" "$cap"; rc=$?
  mins=$(( ( $(now_s) - t0 + 30 ) / 60 ))
  case $rc in
    124|137) what="STOPPED at its cap" ;;
    0|1) what="exit $rc" ;;
    *) what="exit $rc" ;;
  esac
  ledger "$CURRENT END: $what after $mins min"
  printf '%-10s %s after %s min\n' "$CURRENT" "$what" "$mins" >> "$D/summary.new"
  if [[ "$CURRENT" == prelude && $rc -ne 0 ]]; then
    ledger "STOPPED: the prelude did not pass, and every step after it needs it"
    FINAL=$EXIT_REFUSED
    hygiene || true
    push_results "prelude: exit $rc, the session STOPPED" "$FINAL"
    cp "$D/summary.new" "$D/summary.txt"
    exit "$FINAL"
  fi
  case $rc in
    0|1) ;;
    2|3|124) (( FINAL < EXIT_INVALID )) && FINAL=$EXIT_INVALID ;;
    *) FINAL=$EXIT_ERROR ;;
  esac
  if ! hygiene; then
    ledger "STOPPED after $CURRENT: the card could not be made safe for the next step"
    FINAL=$EXIT_ERROR
    cp "$D/summary.new" "$D/summary.txt"
    push_results "$CURRENT: $what; hygiene FAILED, the session STOPPED" "$FINAL"
    exit "$FINAL"
  fi
  cp "$D/summary.new" "$D/summary.txt"
  push_results "$CURRENT: $what ($mins min)"
done
CURRENT=""
ledger "driver end: exit $FINAL"
cp "$D/summary.new" "$D/summary.txt"
push_results "the session ended: driver exit $FINAL" "$FINAL"
exit "$FINAL"
