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

  log "$env: building (this is the part you only pay for once)"
  if [[ ! -x "$VENVS/$env/bin/python" ]]; then
    # base inherits the image's CUDA-matched torch; the framework envs are
    # isolated because they each pin a torch of their own.
    uv venv "$@" "$VENVS/$env"
  fi

  uv pip install --python "$VENVS/$env/bin/python" -r "$req"
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

for env in "${targets[@]}"; do
  case "$env" in
    base) setup_env base --system-site-packages ;;
    *)    setup_env "$env" ;;
  esac
done

log "--- environment ---"
log "workspace           $WORKSPACE"
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
log "commit requirements/resolved-*.txt so later sessions install the exact set"
