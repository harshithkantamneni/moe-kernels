#!/usr/bin/env bash
# Idempotent environment bootstrap for a RunPod H200 pod backed by a network
# volume. First run installs; every later run detects an unchanged requirements
# file by content hash and skips in about a second.
#
#   bash scripts/setup_runpod.sh                      # all environments
#   bash scripts/setup_runpod.sh base                 # just one
#   bash scripts/setup_runpod.sh --dry-run            # the plan, free, writes nothing
#   bash scripts/setup_runpod.sh --check              # is resolved-*.txt still true
#   bash scripts/setup_runpod.sh --base-python 3.12   # pin base's interpreter
#   bash scripts/setup_runpod.sh --isolated base      # base without the image's torch
#   MOE_FORCE=1 bash scripts/setup_runpod.sh          # rebuild regardless of hashes
#
# Everything expensive lives on the volume, so a terminated pod costs nothing
# but the pod.
#
# THE PIN USED TO BE A SOURCE EDIT, AND THE EDIT TAINTED EVERY ROW.
# `docs/RUNPOD.md` told the operator to change this file's `setup_env base
# --system-site-packages` into `setup_env base --python 3.12` to reach the torch
# 2.13.0 the published rows were measured on. That edit modifies a TRACKED file,
# so `pod_session.sh` P1 (clean working tree) fails and `driver.py` stamps
# `git_dirty=True` on every row of the session: the 2026-09-01 alpha-0558 arm is
# 100% dirty, as are the 08-26 three-way and the 08-28 v2lite arms. A pin that
# can only be applied by dirtying the tree is not a pin, it is a fork. It is a
# FLAG now (`--base-python` / `--isolated` / `--system-site-packages`, or
# `MOE_BASE_PYTHON` / `MOE_BASE_ISOLATION`), and the session records which was
# used instead of the operator remembering.
#
# THE RESOLVED SETS WERE WRITTEN AND NEVER READ.
# `uv pip freeze > requirements/resolved-<env>.txt` has run at the end of every
# install since the beginning, and `docs/RUNPOD.md` promised that "every later
# session then installs the exact resolved set rather than re-resolving, and
# anyone reproducing your numbers gets the same environment". No code path ever
# installed from one. A fresh volume re-resolved `base.txt`, whose top-level
# pins are unversioned (numpy, pandas, matplotlib, ...), so a stranger's
# environment was whatever the index held that day. This script now installs
# FROM the resolved set when one is present and still true, and REFUSES when it
# is not, naming what has drifted. `--check` asks the same question and spends
# nothing; `--fresh` re-resolves on purpose.
#
# WHY --check RATHER THAN REGENERATING HERE. Regenerating a resolved set means
# resolving and downloading the real wheels, which needs the pod: the CUDA
# wheels are gigabytes and the index that has them is not reachable from a
# laptop in any useful sense. So the laptop's job is to SAY that the committed
# set is stale (`resolved-base.txt` records transformers 5.15.1 against the
# `<4.54` cap `base.txt` grew on 2026-09-01, and prints `torch==2.13.0` with no
# `+cu130` local tag, so the CUDA wheel index is not encoded in it at all), and
# the pod's job is to fix it with `--fresh`.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENVS="${MOE_VENV_ROOT:-$WORKSPACE/venvs}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# One seam, so a test can point the checks at fixture requirement files without
# touching the repo's own. Everything else resolves from REPO_ROOT.
REQ_DIR="${MOE_REQUIREMENTS_DIR:-$REPO_ROOT/requirements}"

# Caches on the volume. TRITON_CACHE_DIR is the one that matters most: without
# it every session recompiles every autotuned kernel variant from scratch.
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKSPACE/torchinductor-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/uv-cache}"

DRY_RUN=0
CHECK_ONLY=0
FRESH=0
BASE_PYTHON="${MOE_BASE_PYTHON:-}"
#: system | isolated. `system` is the historical default: the base venv inherits
#: the RunPod image's CUDA-matched torch rather than guessing a wheel tag.
BASE_ISOLATION="${MOE_BASE_ISOLATION:-system}"
ISOLATION_SET=0
targets=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)               DRY_RUN=1; shift ;;
    --check)                 CHECK_ONLY=1; shift ;;
    --fresh)                 FRESH=1; shift ;;
    --base-python)           BASE_PYTHON="${2:?--base-python needs a version, e.g. 3.12}"; shift 2 ;;
    --isolated)              BASE_ISOLATION="isolated"; ISOLATION_SET=1; shift ;;
    --system-site-packages)  BASE_ISOLATION="system"; ISOLATION_SET=1; shift ;;
    -h|--help)               sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '/^set /d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)                      echo "[setup] unknown flag: $1" >&2; exit 2 ;;
    *)                       targets+=("$1"); shift ;;
  esac
done

# A CONTRADICTION IS REFUSED RATHER THAN RESOLVED BY PRECEDENCE. A venv built on
# a DIFFERENT interpreter cannot meaningfully inherit the image's site-packages:
# the image's torch is built for the image's python, and importing it from
# another one is the ABI trap `capture_traces.py:_disable_torch_extensions`
# exists to work around. Asking for both is a mistake worth stopping for.
if [[ -n "$BASE_PYTHON" && "$BASE_ISOLATION" == "system" ]]; then
  if [[ "$ISOLATION_SET" == "1" ]]; then
    echo "[setup] REFUSING: --base-python $BASE_PYTHON with --system-site-packages." >&2
    echo "[setup]   The image's torch is built for the image's interpreter; a venv on" >&2
    echo "[setup]   another one that inherits it hits the torchvision/torchaudio ABI" >&2
    echo "[setup]   trap. Pick one: --base-python for a pinned torch, or" >&2
    echo "[setup]   --system-site-packages to inherit the image's." >&2
    exit 2
  fi
  # Implied, and said out loud rather than assumed.
  BASE_ISOLATION="isolated"
  echo "[setup] --base-python $BASE_PYTHON implies an isolated base venv"
fi

if [[ ${#targets[@]} -eq 0 ]]; then targets=(base vllm sglang); fi

PY_CHECK="${MOE_PYTHON:-$VENVS/base/bin/python}"
[[ -x "$PY_CHECK" ]] || PY_CHECK="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY_CHECK" ]] || PY_CHECK="$(command -v python3 || true)"

log() { printf '[setup] %s\n' "$*"; }

# --------------------------------------------------------------------------
# is requirements/resolved-<env>.txt still a true description of <env>.txt
# --------------------------------------------------------------------------
#: Prints one line per drift and returns 1 when there is any. Pure text
#: comparison: it never installs, never resolves, and runs on a laptop.
#: `--check` is this and nothing else.
check_resolved() {
  local env="$1"
  "$PY_CHECK" - "$REQ_DIR/${env}.txt" "$REQ_DIR/resolved-${env}.txt" "$env" <<'PYEOF'
import re
import sys
from pathlib import Path

want_path, got_path, env = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if not want_path.is_file():
    print(f"  {env}: no requirements/{env}.txt")
    raise SystemExit(0)
if not got_path.is_file():
    print(f"  {env}: no resolved set yet; the next install writes one")
    raise SystemExit(0)


def entries(text):
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            yield line


NAME = re.compile(r"^([A-Za-z0-9._-]+)\s*(.*)$")


def key(name):
    # PEP 503 normalisation: uv writes nvidia-ml-py, a file may say nvidia_ml_py.
    return re.sub(r"[-_.]+", "-", name).lower()


pins, editable = {}, []
for line in entries(got_path.read_text()):
    if line.startswith("-e "):
        editable.append(line)
        continue
    if "==" in line:
        name, version = line.split("==", 1)
        pins[key(name)] = version.strip()

# TWO SEVERITIES, because one of them can never be fixed from a laptop and a
# check that can only fail is a check nobody reads.
#   blocking: the resolved set CONTRADICTS its input, so installing from it
#             builds an environment the input file forbids.
#   notes   : true, worth saying, and not a reason to refuse. The editable
#             path line is dropped by sanitised_resolved before any install,
#             and the missing +cu130 tag can only be repaired by re-resolving
#             on a pod with the CUDA index reachable.
blocking, notes = [], []
for line in entries(want_path.read_text()):
    m = NAME.match(line)
    if not m:
        continue
    name, spec = m.group(1), m.group(2).strip()
    have = pins.get(key(name))
    if have is None:
        blocking.append(f"{name} is required by {env}.txt and absent from the "
                        f"resolved set, so the resolved set predates it")
        continue
    if not spec:
        continue
    # Only the operators this project's files actually use. An unrecognised one
    # is REPORTED as unchecked rather than silently treated as satisfied.
    for clause in spec.split(","):
        clause = clause.strip()
        cm = re.match(r"^(==|!=|>=|<=|<|>|~=)\s*([0-9][^,\s]*)$", clause)
        if not cm:
            notes.append(f"{name}{spec}: clause {clause!r} is not one this "
                         "check understands, so it was NOT checked")
            continue
        op, bound = cm.group(1), cm.group(2)

        def parts(v):
            head = v.split("+", 1)[0]
            out = []
            for piece in head.split("."):
                num = re.match(r"^\d+", piece)
                out.append(int(num.group()) if num else 0)
            return tuple(out)

        a, b = parts(have), parts(bound)
        width = max(len(a), len(b))
        a = a + (0,) * (width - len(a))
        b = b + (0,) * (width - len(b))
        ok = {"==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b,
              "<": a < b, ">": a > b, "~=": a >= b}[op]
        if not ok:
            blocking.append(f"{name}: resolved says {have}, {env}.txt says "
                            f"{name}{spec}")

torch = pins.get("torch")
if torch and "+" not in torch:
    notes.append(f"torch=={torch} carries no local tag (+cu130), so the CUDA "
                 "wheel index is not encoded and a fresh install can take a "
                 "wheel built against another CUDA. Only a pod can fix this: "
                 "re-resolve there with --fresh")
for line in editable:
    if line.startswith("-e file://"):
        notes.append(f"{line!r} is a path on the machine that wrote this file; "
                     "sanitised_resolved drops it before any install, and the "
                     "repo is installed editable separately")

for line in blocking:
    print(f"  {env}: STALE {line}")
for line in notes:
    print(f"  {env}: note  {line}")
raise SystemExit(1 if blocking else 0)
PYEOF
}

# --------------------------------------------------------------------------
# the resolved set, made installable
# --------------------------------------------------------------------------
#: `uv pip freeze` writes the whole closure INCLUDING `-e file:///workspace/repo`,
#: which names a directory that exists on one machine. Installing the file as it
#: stands therefore fails everywhere else, which is part of why it was never
#: installed from. This writes a copy with the editable lines dropped (the repo
#: is installed separately, editable, a few lines below) and echoes the path.
sanitised_resolved() {
  local env="$1" src="$REQ_DIR/resolved-${env}.txt" dst="$VENVS/.resolved-${env}.txt"
  grep -v '^-e ' "$src" > "$dst"
  printf '%s\n' "$dst"
}

hash_of() { sha256sum "$1" | cut -d' ' -f1; }

free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# Running out of disk halfway through a vLLM install is a slow, expensive
# failure: the wheels are large, the pod is metered, and the error surfaces
# only after several minutes of downloading. Check before starting.
require_space() {
  local path="$1" need="$2" label="$3"
  local avail; avail="$(free_gb "$path")"
  if [[ -z "$avail" ]]; then
    log "could not read free space on $path; continuing"
    return 0
  fi
  # On a RunPod network volume this reports the shared cluster, not your quota,
  # so a huge number here is not evidence of anything. It still catches the case
  # that actually bites: a local/container filesystem filling up.
  log "$label: ${avail}G free on $path (want ~${need}G)"
  if (( avail < need )); then
    echo "[setup] ABORT: only ${avail}G free on $path, need about ${need}G." >&2
    echo "[setup]   Network volumes can be grown in the RunPod console." >&2
    echo "[setup]   Or install fewer environments: bash $0 base" >&2
    return 1
  fi
  return 0
}

# Rough per-environment footprint, measured against the pinned requirement sets.
space_for() {
  case "$1" in
    base)   echo 8 ;;    # more if the base torch is pinned to its own CUDA wheel
    vllm)   echo 20 ;;   # vLLM plus its own torch and CUDA libraries
    sglang) echo 20 ;;   # likewise
    cutile) echo 6 ;;
    *)      echo 5 ;;
  esac
}

#: The uv venv flags for one environment, from the isolation choice. Printed by
#: --dry-run, so the pin is visible before anything is built.
venv_args() {
  local env="$1"
  if [[ "$env" != "base" ]]; then
    # The framework envs are isolated because they each pin a torch of their own.
    printf ''
    return 0
  fi
  if [[ -n "$BASE_PYTHON" ]]; then
    printf -- '--python %s' "$BASE_PYTHON"
  elif [[ "$BASE_ISOLATION" == "isolated" ]]; then
    printf ''
  else
    printf -- '--system-site-packages'
  fi
}

#: Which requirements file an install would read, and why. Echoes
#: `<path>|<reason>`; returns 1 when a resolved set is present and stale.
#:
#: A STALE RESOLVED SET IS A STALE CACHE, NOT A VETO. `<env>.txt` is the file
#: somebody wrote and the source of truth; `resolved-<env>.txt` is a freeze of
#: one machine's answer to it. When the two contradict each other, obeying the
#: freeze would build an environment the input file forbids -- transformers
#: 5.15.1 against a `<4.54` cap, which is the committed state -- so the freeze
#: is BYPASSED and rebuilt from the input, loudly, and the install writes a new
#: freeze at the end. Refusing the session outright would make one stale file
#: cost a pod rental; missing the cache costs one resolve.
requirements_for() {
  local env="$1" plain="$REQ_DIR/${env}.txt" resolved="$REQ_DIR/resolved-${env}.txt"
  if [[ "$FRESH" == "1" ]]; then
    printf '%s|%s\n' "$plain" "--fresh: re-resolving from ${env}.txt"
    return 0
  fi
  if [[ ! -f "$resolved" ]]; then
    printf '%s|%s\n' "$plain" "no resolved-${env}.txt yet; this install writes one"
    return 0
  fi
  if check_resolved "$env" >/dev/null 2>&1; then
    printf '%s|%s\n' "$resolved" "resolved-${env}.txt is still true for ${env}.txt"
    return 0
  fi
  printf '%s|%s\n' "$plain" \
    "resolved-${env}.txt has DRIFTED from ${env}.txt: bypassed, and rebuilt at the end"
  return 1
}

setup_env() {
  local env="$1"
  local req="$REQ_DIR/${env}.txt"
  [[ -f "$req" ]] || { log "no requirements/${env}.txt, skipping"; return 0; }

  local pick reason install_from
  if pick="$(requirements_for "$env")"; then
    install_from="${pick%%|*}"; reason="${pick#*|}"
  else
    install_from="${pick%%|*}"; reason="${pick#*|}"
    log "$env: the committed resolved set is STALE and is being bypassed:"
    check_resolved "$env" || true
    log "$env:   installing from it would build an environment ${env}.txt forbids."
    log "$env:   Resolving from ${env}.txt instead; commit the new"
    log "$env:   requirements/resolved-${env}.txt this run writes."
  fi

  local stamp="$VENVS/.stamp-${env}"
  local want; want="$(hash_of "$install_from")"

  if [[ -z "${MOE_FORCE:-}" && -f "$stamp" && "$(cat "$stamp")" == "$want" \
        && -x "$VENVS/$env/bin/python" ]]; then
    log "$env: unchanged, skipping"
    return 0
  fi

  require_space "$VENVS" "$(space_for "$env")" "$env" || return 1

  log "$env: building from $(basename "$install_from") ($reason)"
  local vargs; vargs="$(venv_args "$env")"
  if [[ ! -x "$VENVS/$env/bin/python" ]]; then
    # shellcheck disable=SC2086  # vargs is a deliberate word list, possibly empty
    uv venv $vargs "$VENVS/$env"
  fi

  # THE OLDER PIN, kept because docs/RUNPOD.md names it and a pod may still be
  # driven that way: MOE_BASE_TORCH overrides the image's torch inside an
  # otherwise inherited venv. --base-python is the newer and cleaner route
  # (a whole interpreter of its own), and the two do not conflict: this only
  # fires when the variable is set.
  if [[ "$env" == "base" && -n "${MOE_BASE_TORCH:-}" ]]; then
    log "base: pinning ${MOE_BASE_TORCH} (overrides the image's torch)"
    if [[ -n "${MOE_TORCH_INDEX:-}" ]]; then
      uv pip install --python "$VENVS/$env/bin/python" \
        --index-url "$MOE_TORCH_INDEX" "$MOE_BASE_TORCH"
    else
      uv pip install --python "$VENVS/$env/bin/python" "$MOE_BASE_TORCH"
    fi
  fi

  # SGLang 0.5.18 pins cuda-tile==1.6.0rc5, a prerelease, and uv refuses
  # prereleases by default. `if-necessary-or-explicit` permits one only when a
  # dependency explicitly asks for it or nothing else satisfies the graph, so
  # this does not quietly upgrade anything else to a release candidate. Our own
  # top-level versions stay exactly pinned either way. An overrides file, when
  # present, additionally repoints a pin that upstream got wrong; see
  # requirements/overrides-sglang.txt.
  local override_args=()
  local overrides="$REQ_DIR/overrides-${env}.txt"
  if [[ -f "$overrides" ]]; then
    log "$env: applying dependency overrides from $(basename "$overrides")"
    override_args=(--override "$overrides")
  fi

  local from="$install_from"
  if [[ "$install_from" == *"/resolved-${env}.txt" ]]; then
    from="$(sanitised_resolved "$env")"
    log "$env: installing the exact resolved closure (editable path lines dropped)"
  fi

  uv pip install --python "$VENVS/$env/bin/python" \
    --prerelease=allow "${override_args[@]}" -r "$from"
  # Editable install so `moe` is importable in every environment and edits to
  # your kernels take effect without reinstalling.
  uv pip install --python "$VENVS/$env/bin/python" -e "$REPO_ROOT" --no-deps

  # ONLY WHEN THE INPUT WAS THE PLAIN FILE. Freezing after installing FROM the
  # resolved set would rewrite it with whatever this machine happened to have,
  # which is how a resolved set silently becomes a record of the last pod rather
  # than of a decision.
  if [[ "$install_from" == "$req" ]]; then
    uv pip freeze --python "$VENVS/$env/bin/python" \
      > "$REQ_DIR/resolved-${env}.txt"
    log "$env: done, resolved set written to requirements/resolved-${env}.txt"
  else
    log "$env: done, installed from the committed resolved set (not re-frozen)"
  fi
}

# --------------------------------------------------------------------------
# --check and --dry-run: neither installs, neither writes a tracked file
# --------------------------------------------------------------------------
if (( CHECK_ONLY )); then
  log "--check: is each resolved set still a true description of its input"
  stale=0
  for env in "${targets[@]}"; do
    if check_resolved "$env"; then
      log "  $env: resolved-${env}.txt agrees with ${env}.txt"
    else
      stale=$((stale + 1))
    fi
  done
  if (( stale )); then
    log "$stale environment(s) have drifted. Re-resolve on the pod: bash $0 --fresh"
    exit 1
  fi
  log "every resolved set is still true"
  exit 0
fi

if (( DRY_RUN )); then
  log "dry run: nothing is installed and nothing is written"
  log "workspace           $WORKSPACE"
  log "venvs               $VENVS"
  log "base interpreter    ${BASE_PYTHON:-the image interpreter}"
  log "base isolation      $BASE_ISOLATION"
  for env in "${targets[@]}"; do
    [[ -f "$REQ_DIR/${env}.txt" ]] || { log "$env: no requirements/${env}.txt"; continue; }
    vargs="$(venv_args "$env")"
    log "$env: uv venv ${vargs:-<no flags>} $VENVS/$env"
    if pick="$(requirements_for "$env")"; then
      log "$env:   install -r $(basename "${pick%%|*}")   (${pick#*|})"
    else
      log "$env:   install -r $(basename "${pick%%|*}")   (${pick#*|})"
      check_resolved "$env" || true
    fi
  done
  log "re-run without --dry-run to build; --check reports staleness alone"
  exit 0
fi

mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$UV_CACHE_DIR" \
         "$VENVS" "$WORKSPACE/results" "$WORKSPACE/traces/raw" 2>/dev/null || true

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "[setup] uv is not on PATH" >&2; exit 1; }

# A framework that fails to install must not abort the run: base is what your
# kernels need, and the baselines are independent of each other. Failures are
# collected and reported at the end instead.
failed=()
for env in "${targets[@]}"; do
  setup_env "$env" || failed+=("$env")
done

log "--- environment ---"
if mountpoint -q "$WORKSPACE" 2>/dev/null; then
  log "workspace           $WORKSPACE (mounted volume, survives pod termination)"
else
  log "workspace           $WORKSPACE  *** NOT A MOUNTED VOLUME ***"
  log "                    Everything here is lost when the pod is terminated,"
  log "                    including the venvs you just paid to build."
fi
log "free on volume      $(free_gb "$WORKSPACE")G"
log "venvs               $VENVS"
log "base interpreter    ${BASE_PYTHON:-the image interpreter}"
log "base isolation      $BASE_ISOLATION"
log "HF_HOME             $HF_HOME"
log "TRITON_CACHE_DIR    $TRITON_CACHE_DIR"
if [[ -x "$VENVS/base/bin/python" ]]; then
  "$VENVS/base/bin/python" - <<'REPORT' || true
import torch
print(f"[setup] torch               {torch.__version__}")
print(f"[setup] cuda                {torch.version.cuda}")
try:
    import triton
    print(f"[setup] triton              {triton.__version__}")
except ImportError:
    print("[setup] triton              NOT INSTALLED")

# THE CLOCK SAMPLER'S SOURCE, reported here because it is invisible everywhere
# else until a whole session has been measured without a clock. timing.py reads
# torch.cuda.clock_rate, which is torch's NVML binding; without nvidia-ml-py
# every KernelTiming carries clock_source "none" and the LEVEL flag is None.
try:
    import pynvml  # noqa: F401
    print("[setup] nvidia-ml-py         installed (SM clock under load is readable)")
except ImportError:
    print("[setup] nvidia-ml-py         MISSING -- every timing row will record "
          "clock_source 'none'")

if not torch.cuda.is_available():
    print("[setup] device              NONE")
else:
    props = torch.cuda.get_device_properties(0)
    hopper = (props.major, props.minor) == (9, 0)
    print(f"[setup] device              {props.name}")
    print(f"[setup] capability          sm_{props.major}{props.minor}"
          + ("  (build CUDA with sm_90a, not sm_90)" if hopper else ""))
    print(f"[setup] memory              {props.total_memory / 1e9:.0f} GB")
    print(f"[setup] SMs / L2            {props.multi_processor_count} / "
          f"{getattr(props, 'L2_cache_size', 0) / 2**20:.0f} MiB")
    try:
        from moe.bench.roofline import (ambiguous_for_device,
                                         available_profiles, for_device)
        from moe.bench.roofline import power_limit_w
        tdp = power_limit_w()
        if tdp:
            print(f"[setup] power limit         {tdp:.0f} W"
                  + ("  (SXM)" if tdp > 650 else "  (NVL)"))
        # Report the MEASURED profile first when one exists. `for_device`
        # answers "which datasheet part is this" and deliberately skips measured
        # ones, so on a card with no datasheet entry -- the A100 has none in this
        # repo -- it said NONE MATCHES and advised running the calibration that
        # had in fact already produced the profile sitting right there.
        measured = None
        try:
            from moe.bench.roofline import load_measured
            measured = load_measured()
        except Exception:
            measured = None
        if measured is not None:
            print(f"[setup] roofline profile    measured, for {props.name}")
        else:
            profile = for_device(props.name, tdp_w=tdp)
            if profile:
                print(f"[setup] roofline profile    {profile}  (datasheet)")
            else:
                tied = ambiguous_for_device(props.name)
                why = (f"AMBIGUOUS between {tied}" if tied
                       else f"NO MEASURED PROFILE, and no datasheet entry either "
                            f"(have {available_profiles()})")
                print(f"[setup] roofline profile    {why}")
                print("[setup]                     -> run scripts/calibrate_hardware.py"
                      " --publish;")
                print("[setup]                        measured ceilings beat any datasheet")
    except ImportError:
        pass

# Decides whether the torch grouped_mm baseline exists at all.
ok = hasattr(torch.nn.functional, "grouped_mm")
print(f"[setup] grouped_mm baseline {'available' if ok else 'MISSING (torch too old)'}")
REPORT
  printf '[setup] driver              %s\n' \
    "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
fi
if (( ${#failed[@]} )); then
  log "--- ${#failed[@]} environment(s) FAILED: ${failed[*]} ---"
  log "the rest are usable; re-run just the failed one after fixing, e.g."
  log "  bash scripts/setup_runpod.sh ${failed[0]}"
else
  log "all environments ready"
fi
log "commit requirements/resolved-*.txt so later sessions install the exact set"
