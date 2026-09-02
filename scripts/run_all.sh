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
# calling this script. And when the profile's own spans need a framework that
# will not run, this REFUSES rather than sweeping a third of the experiment
# under the profile's name. `--envs` still names them explicitly, for the case
# where torch alone is the point.
set -euo pipefail

PROFILE="standard"
MAX_MINUTES=""
RUN_ID=""
DRY_RUN=""
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

if [[ -z "$SKIP_SETUP" ]]; then
  log "environment"
  bash scripts/setup_runpod.sh ${RESOLVED_ENVS//,/ }
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
"$PY" - "$RESULTS_DIR" <<'PYEOF'
import sys
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
from moe.bench.schema import row_bool
throttled = [r for r in rows if row_bool(r, "throttled")]
if throttled:
    print(f"  THROTTLED ROWS  {len(throttled)} (clocks dropped >5% mid-cell)")
# WHICH ENVS ACTUALLY LANDED ROWS. A sweep that meant to run three and ran one
# is invisible in a total, and that is exactly what the ENVS default did for
# every documented session.
envs = sorted({r.get("env_name", "") for r in rows})
print(f"  envs with rows  {', '.join(e for e in envs if e) or 'NONE'}")
print(f"  results in      {results}")
PYEOF

log "done. Stop or terminate the pod now; the volume keeps everything."
