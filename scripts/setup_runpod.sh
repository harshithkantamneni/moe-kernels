#!/usr/bin/env bash
# Idempotent environment bootstrap for a RunPod H200 pod backed by a network
# volume. First run installs; every later run detects an unchanged requirements
# file by content hash and skips in about a second.
#
#   bash scripts/setup_runpod.sh              # all environments
#   bash scripts/setup_runpod.sh base         # just one
#   MOE_FORCE=1 bash scripts/setup_runpod.sh  # rebuild regardless of hashes
#
# Everything expensive lives on the volume, so a terminated pod costs nothing
# but the pod.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENVS="${MOE_VENV_ROOT:-$WORKSPACE/venvs}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Caches on the volume. TRITON_CACHE_DIR is the one that matters most: without
# it every session recompiles every autotuned kernel variant from scratch.
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKSPACE/torchinductor-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/uv-cache}"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$UV_CACHE_DIR" \
         "$VENVS" "$WORKSPACE/results" "$WORKSPACE/traces/raw"

log() { printf '[setup] %s\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "[setup] uv is not on PATH" >&2; exit 1; }

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
    base)   echo 8 ;;    # more if MOE_BASE_TORCH pins its own CUDA torch
    vllm)   echo 20 ;;   # vLLM plus its own torch and CUDA libraries
    sglang) echo 20 ;;   # likewise
    cutile) echo 6 ;;
    *)      echo 5 ;;
  esac
}

setup_env() {
  local env="$1"; shift
  local req="$REPO_ROOT/requirements/${env}.txt"
  [[ -f "$req" ]] || { log "no requirements/${env}.txt, skipping"; return 0; }

  local stamp="$VENVS/.stamp-${env}"
  local want; want="$(hash_of "$req")"

  if [[ -z "${MOE_FORCE:-}" && -f "$stamp" && "$(cat "$stamp")" == "$want" \
        && -x "$VENVS/$env/bin/python" ]]; then
    log "$env: unchanged, skipping"
    return 0
  fi

  require_space "$VENVS" "$(space_for "$env")" "$env" || return 1

  log "$env: building (this is the part you only pay for once)"
  if [[ ! -x "$VENVS/$env/bin/python" ]]; then
    # base inherits the image's CUDA-matched torch; the framework envs are
    # isolated because they each pin a torch of their own.
    uv venv "$@" "$VENVS/$env"
  fi

  # Optional: pin base's torch so it matches the torch the baselines pin,
  # rather than inheriting whatever the image ships. See requirements/base.txt.
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
  # top-level versions stay exactly pinned either way.
  # SGLang 0.5.18 pins a cuda-tile prerelease, so prereleases must be permitted
  # at all. An overrides file, when present, additionally repoints a pin that
  # upstream got wrong; see requirements/overrides-sglang.txt.
  local override_args=()
  local overrides="$REPO_ROOT/requirements/overrides-${env}.txt"
  if [[ -f "$overrides" ]]; then
    log "$env: applying dependency overrides from $(basename "$overrides")"
    override_args=(--override "$overrides")
  fi

  uv pip install --python "$VENVS/$env/bin/python" \
    --prerelease=allow "${override_args[@]}" -r "$req"
  # Editable install so `moe` is importable in every environment and edits to
  # your kernels take effect without reinstalling.
  uv pip install --python "$VENVS/$env/bin/python" -e "$REPO_ROOT" --no-deps

  uv pip freeze --python "$VENVS/$env/bin/python" \
    > "$REPO_ROOT/requirements/resolved-${env}.txt"
  echo "$want" > "$stamp"
  log "$env: done, resolved set written to requirements/resolved-${env}.txt"
}

targets=("$@")
# cutile is deliberately NOT in the default set: its cuda-toolkit pin conflicts
# with torch's, so it is opt-in via `bash scripts/setup_runpod.sh cutile`.
if [[ ${#targets[@]} -eq 0 ]]; then targets=(base vllm sglang); fi

# A framework that fails to install must not abort the run: base is what your
# kernels need, and the baselines are independent of each other. Failures are
# collected and reported at the end instead.
failed=()
for env in "${targets[@]}"; do
  case "$env" in
    base) setup_env base --system-site-packages || failed+=("$env") ;;
    *)    setup_env "$env" || failed+=("$env") ;;
  esac
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
        from moe.bench.roofline import available_profiles, for_device
        profile = for_device(props.name)
        extra = "" if profile else f"  (have {available_profiles()}; run calibrate_hardware.py)"
        print(f"[setup] roofline profile    {profile or 'NONE MATCHES'}{extra}")
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
