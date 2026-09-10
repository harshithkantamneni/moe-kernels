#!/usr/bin/env bash
# One-shot benchmark session. On the expensive box this should be the only
# command you type.
#
#   bash scripts/run_all.sh --dry-run                 # free, works on a laptop
#   bash scripts/run_all.sh --profile smoke           # ~2 min shakedown
#   bash scripts/run_all.sh --profile standard --max-minutes 45
#   bash scripts/run_all.sh --profile standard --run-id abc123   # resume
#   bash scripts/run_all.sh --envs base               # deliberately torch only
#
# Order is deliberate: pull, setup, TESTS, smoke, then the sweep. If the test
# suite fails, the sweep never starts and you have spent seconds rather than an
# hour producing numbers from broken code.
#
# THE DOCUMENTED ONE-COMMAND SESSION USED TO SWEEP ONLY TORCH.
# `ENVS` defaulted to "base", and no runbook line ever passed `--envs
# base,vllm,sglang`, so `bash scripts/run_all.sh --profile standard
# --max-minutes 45` -- the command `docs/RUNPOD.md` gives for "every session
# after that" -- produced an arm with no vLLM and no SGLang rows: none of the
# kernels this study's claims are about. It exited 0 and printed a summary, so
# nothing said otherwise, and `--dry-run` did not even look at `$ENVS`. Every
# published three-way arm required the undocumented flag.
#
# The default is now AUTO: the venvs that exist and import are the envs that
# run, which is the same detection `scripts/pod_session.sh` already does before
# calling this script. Setup installs the union of that and what the profile is
# priced for, so a FRESH pod bootstraps the three-way arm from this one command
# instead of detecting "base", installing "base", and refusing. And when the
# profile's own spans still need a framework that will not run -- an install
# that failed -- this REFUSES rather than sweeping a third of the experiment
# under the profile's name. `--envs` still names them explicitly, for the case
# where torch alone is the point, and is never widened.
set -euo pipefail

PROFILE="standard"
MAX_MINUTES=""
RUN_ID=""
DRY_RUN=""
#: --summary-only DIR: print the end-of-run summary over rows that already
#: exist and exit. It is how the summary's wording is tested without a sweep.
SUMMARY_ONLY=""
#: Empty means AUTO-DETECT. "base" is no longer a default, it is an answer.
ENVS=""
SKIP_SETUP=""
SKIP_TESTS=""
ALLOW_MISSING_ENVS=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)      PROFILE="$2"; shift 2 ;;
    --max-minutes)  MAX_MINUTES="$2"; shift 2 ;;
    --run-id)       RUN_ID="$2"; shift 2 ;;
    --envs)         ENVS="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --summary-only) SUMMARY_ONLY="$2"; shift 2 ;;
    --skip-setup)   SKIP_SETUP=1; shift ;;
    --skip-tests)   SKIP_TESTS=1; shift ;;
    --allow-missing-envs) ALLOW_MISSING_ENVS=1; shift ;;
    *)              EXTRA+=("$1"); shift ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKSPACE="${WORKSPACE:-/workspace}"
VENVS="${MOE_VENV_ROOT:-$WORKSPACE/venvs}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKSPACE/torchinductor-cache}"

# The sweep generates random weights and must never reach a model hub. If some
# future code path tries, this makes it fail in seconds rather than quietly
# pulling tens of GB mid-session. scripts/capture_traces.py is run separately
# and deliberately, so it is unaffected.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RESULTS_DIR="${MOE_RESULTS_DIR:-$WORKSPACE/results}"
[[ -d "$WORKSPACE" ]] || RESULTS_DIR="$REPO_ROOT/results"

log() { printf '\n[run_all] %s\n' "$*"; }

PY="${MOE_PYTHON:-$VENVS/base/bin/python}"
[[ -x "$PY" ]] || PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

#: The end-of-run summary over a results directory. A function because it has
#: two callers, the end of a sweep and --summary-only, and the wording of the
#: throttled line was wrong at one call site for as long as it had only one.
print_summary() {
  "$PY" - "$1" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from moe.bench.schema import passed, read_csv

results = Path(sys.argv[1])
rows = []
# NOT *.csv: merged.csv contains the same rows as the run_*.csv files it was
# built from, so globbing both counts every row twice.
for p in sorted(results.glob("run_*.csv")):
    try:
        rows.extend(read_csv(p))
    except Exception as e:
        print(f"  {p.name}: unreadable ({e})")
print(f"  rows            {len(rows)}")
ok = [r for r in rows if passed(r)]
print(f"  correctness ok  {len(ok)}")
failed = [r for r in rows if not passed(r)]
if failed:
    print(f"  CORRECTNESS FAILURES {len(failed)}:")
    for r in failed[:10]:
        print(f"    {r['impl']:<28} {r['model']}/T{r['num_tokens']} "
              f"abs_err={float(r['max_abs_err']):.3e}")


def emit(line):
    print(f"  CLOCK FLAG      {line}")


# `throttled` IS ONE COLUMN WITH TWO MEANINGS, split by the instrument that
# wrote the row, and until 2026-09-03 this line described neither: "clocks
# dropped >5% mid-cell" named a detector that never existed. On a row written
# before schema v5 the flag is the retired two-sample drift check: the SM clock
# read at an idle instant before the cell and again after it, set on a >5%
# drop. That detected whether the FIRST read had caught the idle boost clock,
# not throttling under load (moe/bench/timing.py, CLOCKS ARE READ UNDER LOAD:
# on the alpha-0558 arm it flagged 91% of vLLM rows above T=4096 while flagged
# and unflagged replicates timed at ratio 0.998). On a v5 row the driver sets
# it when the DRIFT verdict taken WHILE the trials ran failed, and on nothing
# else (moe/bench/driver.py).
#
# DRIFT IS THE WHOLE EXCLUSION RULE, SINCE 2026-09-09, AND THE LEVEL SIDE IS A
# RECORD. Between 03df2d4 (2026-09-03) and that day the driver also wrote a
# LEVEL failure on the LOW side into `throttled`, and this line reported it as
# a throttle. The 750-cell H200 census that closed the first gaps session says
# it is not one: under a 700 W cap the SM clock under load is set PER TILE by
# that kernel's own power draw. BLOCK_M=128 holds a median 1395 MHz
# over 196 cells, BLOCK_M=256 1650 over 311, BLOCK_M=32 1474 at GROUP_SIZE_M=1
# over 18 and 1740 from GROUP_SIZE_M=8 up over 50, memory-shaped
# cells 1950-1980, and the calibration GEMM itself 1485 at 691 W, which is near
# the LOW end of what dense tensor work does on this card rather than the
# middle of anything. So a +/-5% band around the GEMM's clock is a rule against
# a TILE: it excluded every multi-tile BLOCK_M=128 cell of the roofline arm and
# 110 of 168 depth treads, each of them the steady state of one tile family in
# every rep, while excluding nothing at all in the arms whose tiles happen to
# sit near the GEMM. LEVEL is still SCORED and its side is still WRITTEN, on
# every row, because "this cell ran at 1395 and the roof was measured at 1485"
# is a fact the reader needs; it excludes nothing on either side. What is wrong
# on a cell whose clock differs from the reference is the FIXED-roof fraction
# `pct_of_achieved_tflops`, and `pct_of_roof_at_cell_clock` is the correction,
# on the LOW side exactly as on the HIGH side.
#
# So this line reads DRIFT for the flag and reports the LEVEL sides beside it
# as the record they are. The two flag origins are counted apart and named by
# what each detected; a row whose instrument cannot be read is reported as
# exactly that rather than filed under either.
from collections import Counter

from moe.bench.schema import (UNRECORDED, VERDICT_FAILED, TimingInstrumentUnrecorded,
                              has_kernel_timing, row_bool, timing_verdict)
try:
    from moe.bench.timing import DRIFT_FRACTION, LEVEL_FRACTION, LEVEL_HIGH_FRACTION
    level_word = (f"outside the band {LEVEL_FRACTION:.0%} to "
                  f"{LEVEL_HIGH_FRACTION:.0%} of")
    drift_word = f"more than {DRIFT_FRACTION:.0%} apart"
except Exception:  # torch absent: name the constants rather than guess their values
    level_word = "outside the band timing.LEVEL_FRACTION to timing.LEVEL_HIGH_FRACTION of"
    drift_word = "more than timing.DRIFT_FRACTION apart"


def side_of(r):
    # The row's own word, never a constant copied here: the word is written by
    # moe/bench/driver.py from timing.LEVEL_LOW / LEVEL_HIGH and read back as
    # is, so a side this file has never heard of is printed, not misfiled.
    # A v5 row (00f3324 to 03df2d4) carries no side because that instrument
    # had no high edge, so its LEVEL failure is low by the instrument's own
    # definition, not by this file's default; a v6 row with no side is a row
    # the driver did not write, and is printed as unrecorded.
    side = str(r.get("clock_level_side") or "").strip()
    if side and side != UNRECORDED:
        return side
    try:
        version = int(float(r.get("schema_version") or 0))
    except ValueError:
        version = 0
    return "low" if 0 < version < 6 else "unrecorded"


def sides_text(counter):
    return ", ".join(f"{side} {n}" for side, n in sorted(counter.items()))


flagged = [r for r in rows if row_bool(r, "throttled")]
# EVERY LEVEL FAILURE IN THE FILE, SPLIT ON THE ONE RULE THAT EXCLUDES. A LEVEL
# failure is a record and not an exclusion, so it is counted whether the row is
# flagged or not: counting it over the un-flagged rows alone is what this file
# did while LOW was a throttle. But a row that failed LEVEL and ALSO failed
# DRIFT is excluded, on DRIFT, and until now it was counted here and printed
# under "NONE of them is excluded for it". It is counted apart and reported on
# its own line instead. It is not a hypothetical row: memory_branch_anchor's
# 2026-09-09 cells carry 18 LOW and 12 DRIFT with 4 rows in both.
level_sides = Counter()
drifted_sides = Counter()
for r in rows:
    try:
        if not has_kernel_timing(r):
            continue
        if timing_verdict(r, "clock_level_ok") != VERDICT_FAILED:
            continue
        drifted = timing_verdict(r, "clock_drift_ok") == VERDICT_FAILED
    except TimingInstrumentUnrecorded:
        continue
    (drifted_sides if drifted else level_sides)[side_of(r)] += 1
if flagged:
    under_load, legacy, unreadable = [], [], []
    drift = no_drift = 0
    for r in flagged:
        try:
            if not has_kernel_timing(r):
                legacy.append(r)
                continue
            dr = timing_verdict(r, "clock_drift_ok")
        except TimingInstrumentUnrecorded:
            unreadable.append(r)
            continue
        under_load.append(r)
        if dr == VERDICT_FAILED:
            drift += 1
        else:
            no_drift += 1
    if under_load:
        emit(f"{len(under_load)} rows carry throttled=True from the under-load clock "
             f"check: DRIFT failed on {drift} (first and last under-load samples "
             f"{drift_word}, either direction), which is the whole rule since "
             f"2026-09-09")
    if no_drift:
        emit(f"{no_drift} of those rows carry throttled=True with DRIFT not failed, "
             f"which moe/bench/driver.py never writes: since 2026-09-09 it sets the "
             f"column from the DRIFT verdict alone. This file was not written by that "
             f"driver, or it predates the rule and carries a LEVEL-low exclusion; do "
             f"not pool its flagged rows with rows scored under the current rule")
    if legacy:
        emit(f"{len(legacy)} rows carry throttled=True from the retired pre-v5 drift "
             f"flag: two idle-instant SM-clock reads either side of the cell, >5% "
             f"apart. It detected an idle-boost catch, not throttling under load "
             f"(moe/bench/timing.py)")
    if unreadable:
        emit(f"{len(unreadable)} rows carry throttled=True with an unreadable "
             f"instrument column, so which detector set it cannot be said")
if level_sides:
    emit(f"{sum(level_sides.values())} rows failed LEVEL (SM clock under load "
         f"{level_word} the clock the calibration GEMM ran at) and NONE of them is "
         f"excluded for it: side {sides_text(level_sides)}, recorded. Under a power "
         f"cap the clock is set per tile by the kernel's own draw, so a LOW cell is a "
         f"hungry tile at its steady state and a HIGH one a memory-shaped cell "
         f"boosting; what is wrong for both is the fixed-roof fraction "
         f"(pct_of_achieved_tflops), and pct_of_roof_at_cell_clock is the column to "
         f"read beside it")
if drifted_sides:
    emit(f"{sum(drifted_sides.values())} further rows failed LEVEL AND drifted "
         f"(side {sides_text(drifted_sides)}): they are EXCLUDED, on DRIFT, and are "
         f"not in the count above. A row whose warmup never reached one operating "
         f"point has no steady clock for a side to describe")
# WHICH ENVS ACTUALLY LANDED ROWS. A sweep that meant to run three and ran one
# is invisible in a total, and that is exactly what the ENVS default did for
# every documented session.
envs = sorted({r.get("env_name", "") for r in rows})
print(f"  envs with rows  {', '.join(e for e in envs if e) or 'NONE'}")
print(f"  results in      {results}")
PYEOF
}

if [[ -n "$SUMMARY_ONLY" ]]; then
  log "summary only: $SUMMARY_ONLY"
  print_summary "$SUMMARY_ONLY"
  exit 0
fi

# --------------------------------------------------------------------------
# which environments will actually run
# --------------------------------------------------------------------------
#: base always (it is this repo's own venv), plus each framework whose venv
#: exists AND imports. Importing is the test rather than the directory existing,
#: because a half-built venv is the normal state after an install that ran out
#: of disk, and `scripts/pod_session.sh` probes it exactly this way.
detect_envs() {
  local found="base" py
  for name in vllm sglang; do
    py="$VENVS/$name/bin/python"
    [[ -x "$py" ]] || continue
    "$py" -c "import $name" >/dev/null 2>&1 && found="$found,$name"
  done
  printf '%s\n' "$found"
}

#: Which framework environments this profile is priced to sweep. Asked of the
#: harness rather than pattern-matched on a profile name, so a profile added
#: later cannot quietly fall outside the check: `estimated_hours` returns one
#: entry per environment that measures spans, from counts taken off the
#: published arms, and an environment it prices at zero is one this profile
#: does not need.
profile_needs() {
  "$PY" - "$1" <<'NEEDS'
import sys
sys.path.insert(0, ".")
try:
    from moe.bench import profiles as PR
    hours = PR.estimated_hours(PR.get(sys.argv[1]))
except Exception:
    # An unknown profile is the sweep's own problem to report; this check does
    # not get to decide that, and must not refuse a session over it.
    print("")
    raise SystemExit(0)
print(",".join(sorted(env for env, h in hours.items()
                      if env not in ("base", "total") and h > 0)))
NEEDS
}

resolve_envs() {
  if [[ -n "$ENVS" ]]; then
    printf '%s\n' "$ENVS"
  else
    detect_envs
  fi
}

#: Is there a card here at all? The same probe scripts/pod_session.sh uses.
#: Nothing on a laptop is metered, so a shortfall there costs nothing to
#: discover and a laptop can never have the framework venvs anyway.
have_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  [[ -n "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" ]]
}

#: Does what will run cover what the profile is for? Prints the shortfall and
#: returns 1 when it does not.
#:
#: A SHORTFALL THE OPERATOR NAMED IS NOT THE SAME AS ONE THE BOX PRODUCED. An
#: explicit `--envs base` is a decision and gets a loud warning; an
#: auto-detected shortfall is the documented one-command session quietly
#: sweeping a third of the experiment, and that stops.
check_envs_cover_profile() {
  local envs="$1" profile="$2" needed missing="" name
  needed="$(profile_needs "$profile")"
  [[ -n "$needed" ]] || return 0
  for name in ${needed//,/ }; do
    case ",$envs," in
      *",$name,"*) ;;
      *) missing="${missing:+$missing,}$name" ;;
    esac
  done
  [[ -n "$missing" ]] || return 0
  if [[ -n "$ENVS" ]]; then
    {
      echo "[run_all] WARNING: --envs named '$envs' and profile '$profile' is priced"
      echo "[run_all]   for $needed. This arm will carry no $missing rows, so it is"
      echo "[run_all]   not comparable with the published three-way arms."
    } >&2
    return 0
  fi
  {
    echo "[run_all] REFUSE: profile '$profile' spans $needed, and this box will run"
    echo "[run_all]   only: $envs"
    echo "[run_all]   Missing: $missing. Sweeping anyway produces an arm with no"
    echo "[run_all]   $missing rows -- none of the kernels the study's claims are"
    echo "[run_all]   about -- and it would exit 0 and print a summary, which is"
    echo "[run_all]   what every documented session did."
    echo "[run_all]   Fix the environment:  bash scripts/setup_runpod.sh ${missing//,/ }"
    echo "[run_all]   or say you meant it:  --envs $envs"
  } >&2
  return 1
}

# A SWEEP IS ONE EXPERIMENT, SO IT GETS ONE ID. run_env hands identical cli_args
# to every venv, so a --run-id given here reaches all of them and the arms stay
# findable together afterwards. Left empty, each venv falls through to
# driver.RunConfig's own uuid4 default and one sweep lands as three unrelated
# run ids: that is why publishing the 2026-08-26 three-way took three commits.
# Same 12-hex shape as the driver default, so nothing downstream can tell which
# side generated it.
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')"
fi

RESOLVED_ENVS="$(resolve_envs)"

# The laptop path: validate the whole matrix without a GPU and stop. It writes
# nothing, tracked or otherwise.
if [[ -n "$DRY_RUN" ]]; then
  log "dry run (no GPU, nothing spent)"
  printf '[run_all] sweep run id  %s  (every venv writes run_%s_<env>.csv)\n' \
    "$RUN_ID" "$RUN_ID"
  # THE LINE THE DRY RUN USED TO OMIT. $ENVS was never printed here, so the
  # laptop pre-check could not expose the defect it exists to expose.
  if [[ -n "$ENVS" ]]; then why="named with --envs"; else why="auto-detected from $VENVS"; fi
  printf '[run_all] envs: %s   (%s)\n' "$RESOLVED_ENVS" "$why"
  if ! check_envs_cover_profile "$RESOLVED_ENVS" "$PROFILE"; then
    # THE REFUSAL IS ABOUT SPENDING, AND A LAPTOP SPENDS NOTHING. A dry run
    # with no card is the rehearsal: it has to SHOW the shortfall, which it
    # now does, and it must not exit red every time, because a check that can
    # only fail on the machine you rehearse on is one you learn to skip. On a
    # pod the same shortfall stops the session before a minute is metered.
    if have_gpu && [[ -z "$ALLOW_MISSING_ENVS" ]]; then
      echo "[run_all] a card is present, so this is a rehearsal on the pod itself:" >&2
      echo "[run_all] stopping before anything is spent." >&2
      exit 1
    fi
    if [[ -n "$ALLOW_MISSING_ENVS" ]]; then
      printf '[run_all] --allow-missing-envs: continuing with %s\n' "$RESOLVED_ENVS"
    else
      echo "[run_all] no card here, so nothing is at risk and the plan still" >&2
      echo "[run_all] prints. On a pod this shortfall stops the session." >&2
    fi
  fi
  # B14: the smallest difference this sweep could resolve, from a noise level
  # measured on the published rows rather than assumed. Printed here because
  # the plan is where a comparison is still cheap to redesign.
  "$PY" - <<'PYEOF' || true
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from pathlib import Path

from calibrate_hardware import mde_line, published_sigma_pct

sigma, _n, why = published_sigma_pct(Path("results/published"))
print(f"[run_all] {mde_line(sigma, 3, why)}")
PYEOF
  exec "$PY" -m moe.bench.cli --profile "$PROFILE" --dry-run "${EXTRA[@]+"${EXTRA[@]}"}"
fi

log "git"
git rev-parse --short HEAD 2>/dev/null || true
if [[ -z "${MOE_NO_PULL:-}" ]]; then
  # THE EXIT CODE USED TO BE THE TAIL'S. `git pull ... | tail -2 || echo skipped`
  # reports whether `tail` succeeded, which it always does, so a failed pull was
  # indistinguishable from a clean one and the session swept an unknown tree.
  pull_out=""
  if pull_out="$(git pull --ff-only 2>&1)"; then
    printf '%s\n' "$pull_out" | tail -2
  else
    printf '[run_all] git pull FAILED, sweeping the tree as it stands:\n'
    printf '%s\n' "$pull_out" | sed 's/^/[run_all]   /' | tail -5
  fi
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[run_all] WARNING: working tree is dirty; every row will be marked git_dirty"
  git status --porcelain | sed 's/^/[run_all]   /' | head -5
fi

# WHAT SETUP INSTALLS IS WHAT THE PROFILE NEEDS, NOT ONLY WHAT IS ALREADY HERE.
# Driving this from the detection alone made a FRESH pod unbootstrappable by the
# documented command: nothing is installed, so "base" is detected, "base" is
# installed, "base" is re-detected, and the refusal below fires on the very
# `bash scripts/run_all.sh --profile standard` that docs/RUNPOD.md gives. The
# operator's only way out was `--envs base,vllm,sglang`, the undocumented flag
# this requirement exists to remove. So the auto path installs the UNION of what
# is here and what the profile is priced for, and the refusal below keeps its
# job: it now fires when an install actually failed, which is the case it is
# for.
#
# `--envs` is left alone. That is a decision an operator typed, and `--envs
# base` must not quietly grow a vLLM install because the profile would like one.
setup_envs() {
  local envs="$1" needed="" name
  if [[ -z "$ENVS" ]]; then
    needed="$(profile_needs "$PROFILE")"
  fi
  for name in ${needed//,/ }; do
    case ",$envs," in
      *",$name,"*) ;;
      *) envs="$envs,$name" ;;
    esac
  done
  printf '%s\n' "$envs"
  return 0
}

if [[ -z "$SKIP_SETUP" ]]; then
  log "environment"
  SETUP_ENVS="$(setup_envs "$RESOLVED_ENVS")"
  [[ "$SETUP_ENVS" == "$RESOLVED_ENVS" ]] || \
    log "installing $SETUP_ENVS: profile '$PROFILE' is priced for more than this box has"
  bash scripts/setup_runpod.sh ${SETUP_ENVS//,/ }
fi

# Re-detect AFTER setup: an install that just succeeded adds an env, and one
# that failed removes one, and either way the sweep must run what is there
# rather than what was there a minute ago.
if [[ -z "$ENVS" ]]; then
  RESOLVED_ENVS="$(detect_envs)"
fi
log "envs: $RESOLVED_ENVS"
if ! check_envs_cover_profile "$RESOLVED_ENVS" "$PROFILE"; then
  if [[ -z "$ALLOW_MISSING_ENVS" ]]; then
    echo "[run_all] stopping before anything is spent." >&2
    exit 1
  fi
  log "--allow-missing-envs: sweeping $RESOLVED_ENVS anyway"
fi

PY="$VENVS/base/bin/python"
[[ -x "$PY" ]] || PY="python3"

if [[ -z "$SKIP_TESTS" ]]; then
  log "test suite (a failure here stops the session before it costs anything)"
  # tests/test_gpu.py auto-skips off a device, so on the box this is the first
  # and only verification the CUDA timing paths get.
  "$PY" -m pytest tests/ -q -x
fi

# Nsight Compute cannot run on a rented pod (ERR_NVGPUCTRPERM), so the roofline
# would otherwise rest on a datasheet peak. Measure the real ceilings once.
#
# Ask the harness, rather than globbing. A glob for measured_*.yaml matches a
# calibration committed for a DIFFERENT device: on an H100 the repo's
# measured_nvidia_h200.yaml satisfied it, the gate was skipped, and the sweep
# then ran with no ceilings at all and silently empty efficiency columns.
# load_measured() is the same resolution the sweep itself uses, so the gate and
# the run cannot disagree about whether this machine is calibrated.
if ! "$PY" -c "
import sys
from moe.bench.roofline import HardwareMismatch, load_measured
try:
    sys.exit(0 if load_measured() is not None else 1)
except HardwareMismatch:
    sys.exit(1)
" 2>/dev/null; then
  log "calibrating achievable bandwidth and BF16 (once per pod type)"
  # --publish IS REQUIRED HERE AND NOWHERE ELSE. The calibration's default
  # output is an untracked session path, so running it never dirties the tree
  # by accident; but roofline.load_measured() reads moe/bench/hardware/, and
  # the sweep two steps below quotes every efficiency column against whatever
  # is there. So this one call asks for the tracked copy on purpose, and says
  # so, rather than the copy happening as a side effect nobody chose.
  "$PY" scripts/calibrate_hardware.py --publish || \
    echo "[run_all] calibration failed; efficiency columns will stay empty"
fi

log "smoke: correctness and plumbing"
# baselines included: until a kernel exists they are the only implementations
# there are, and a smoke step that benchmarks nothing proves nothing.
"$PY" -m moe.bench.cli --profile smoke --out-dir "$RESULTS_DIR" \
  --groups reference,kernels,baselines "${EXTRA[@]+"${EXTRA[@]}"}"

log "sweep: profile=$PROFILE envs=$RESOLVED_ENVS"
ARGS=(--profile "$PROFILE" --out-dir "$RESULTS_DIR" --groups reference,kernels,baselines)
[[ -n "$MAX_MINUTES" ]] && ARGS+=(--max-minutes "$MAX_MINUTES")
ARGS+=(--run-id "$RUN_ID")   # never empty: see the sweep-id note above
ARGS+=("${EXTRA[@]+"${EXTRA[@]}"}")

"$PY" - "$RESOLVED_ENVS" "$RESULTS_DIR" "${ARGS[@]}" <<'PYEOF'
import sys
from pathlib import Path
from moe.runner.subproc import run_envs

envs = [e for e in sys.argv[1].split(",") if e]
out_dir = Path(sys.argv[2])
args = sys.argv[3:]
merged = out_dir / "merged.csv"
run_envs(envs, args, merged, cwd=Path.cwd())
PYEOF

# FIGURES ARE DRAWN FROM THE ARM, NOT FROM THE VOLUME. This used to plot the
# whole of $RESULTS_DIR into a repo-root plots/ directory, and
# publish_results.sh then copied every PNG in it into whichever arm was being
# published: the ridge-resolution arm shipped a toy plot its CSVs do not
# contain, and the bf16-only alpha-0558 arm shipped six fp8 figures byte
# identical to another arm's. `results/` outlives a session, so "the whole
# volume" is never one experiment. publish_results.sh now regenerates from the
# published arm's own CSVs; this one is for looking at during the session.
log "plots (this session's results dir, for looking at; publish redraws per arm)"
"$PY" scripts/plot.py --results "$RESULTS_DIR" --out "$RESULTS_DIR/plots" || \
  echo "[run_all] plotting skipped"

log "summary"
print_summary "$RESULTS_DIR"

log "done. Stop or terminate the pod now; the volume keeps everything."
