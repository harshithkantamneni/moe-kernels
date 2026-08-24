#!/usr/bin/env bash
# One-shot benchmark session. On the expensive box this should be the only
# command you type.
#
#   bash scripts/run_all.sh --dry-run                 # free, works on a laptop
#   bash scripts/run_all.sh --profile smoke           # ~2 min shakedown
#   bash scripts/run_all.sh --profile standard --max-minutes 45
#   bash scripts/run_all.sh --profile standard --run-id abc123   # resume
#
# Order is deliberate: pull, setup, TESTS, smoke, then the sweep. If the test
# suite fails, the sweep never starts and you have spent seconds rather than an
# hour producing numbers from broken code.
set -euo pipefail

PROFILE="standard"
MAX_MINUTES=""
RUN_ID=""
DRY_RUN=""
ENVS="base"
SKIP_SETUP=""
SKIP_TESTS=""
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

# The laptop path: validate the whole matrix without a GPU and stop.
if [[ -n "$DRY_RUN" ]]; then
  PY="${MOE_PYTHON:-$REPO_ROOT/.venv/bin/python}"
  [[ -x "$PY" ]] || PY="python3"
  log "dry run (no GPU, nothing spent)"
  exec "$PY" -m moe.bench.cli --profile "$PROFILE" --dry-run "${EXTRA[@]+"${EXTRA[@]}"}"
fi

log "git"
git rev-parse --short HEAD 2>/dev/null || true
if [[ -z "${MOE_NO_PULL:-}" ]]; then
  git pull --ff-only 2>&1 | tail -2 || echo "[run_all] pull skipped"
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[run_all] WARNING: working tree is dirty; every row will be marked git_dirty"
fi

if [[ -z "$SKIP_SETUP" ]]; then
  log "environment"
  bash scripts/setup_runpod.sh ${ENVS//,/ }
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
if ! compgen -G "moe/bench/hardware/measured_*.yaml" >/dev/null; then
  log "calibrating achievable bandwidth and BF16 (once per pod type)"
  "$PY" scripts/calibrate_hardware.py || \
    echo "[run_all] calibration failed; efficiency columns will stay empty"
fi

log "smoke: correctness and plumbing"
# baselines included: until a kernel exists they are the only implementations
# there are, and a smoke step that benchmarks nothing proves nothing.
"$PY" -m moe.bench.cli --profile smoke --out-dir "$RESULTS_DIR" \
  --groups reference,kernels,baselines "${EXTRA[@]+"${EXTRA[@]}"}"

log "sweep: profile=$PROFILE envs=$ENVS"
ARGS=(--profile "$PROFILE" --out-dir "$RESULTS_DIR" --groups reference,kernels,baselines)
[[ -n "$MAX_MINUTES" ]] && ARGS+=(--max-minutes "$MAX_MINUTES")
[[ -n "$RUN_ID" ]] && ARGS+=(--run-id "$RUN_ID")
ARGS+=("${EXTRA[@]+"${EXTRA[@]}"}")

"$PY" - "$ENVS" "$RESULTS_DIR" "${ARGS[@]}" <<'PYEOF'
import sys
from pathlib import Path
from moe.runner.subproc import run_envs

envs = [e for e in sys.argv[1].split(",") if e]
out_dir = Path(sys.argv[2])
args = sys.argv[3:]
merged = out_dir / "merged.csv"
run_envs(envs, args, merged, cwd=Path.cwd())
PYEOF

log "plots"
"$PY" scripts/plot.py --results "$RESULTS_DIR" --out plots || \
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
print(f"  results in      {results}")
PYEOF

log "done. Stop or terminate the pod now; the volume keeps everything."
