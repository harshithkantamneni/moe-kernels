#!/usr/bin/env bash
# Frontier (github.com/NetX-lab/Frontier, pinned db5a85b) in its OWN environments on the VM (KIT-SPEC 6).
#
#   bash scripts/prod/frontier_setup.sh --dest $MOE_HOME/frontier [--arms primary,fidelity] [--dry-run]
#
# PRIMARY  venv-primary: Python of the study's vLLM venv, torch at the study venv's EXACT build (version and CUDA
#          tag read off $PY_VLLM), then requirements/resolved-vllm.txt (vLLM 0.27.1 and its closure, as the study
#          runs it), then Frontier's runtime dependencies CONSTRAINED by resolved-vllm.txt (nothing the study pins
#          moves), then Frontier itself `-e --no-deps`. Frontier's `import vllm` therefore resolves to 0.27.1
#          (CRITIQUE B2): checked after the build, with the API branch its kernel module selects.
# FIDELITY venv-fidelity: Frontier's pinned stack (environment_profiling.yml): Python 3.10, torch 2.8.0 (cu128
#          aarch64 wheel), vLLM 0.10.2, flashinfer-python 0.3.x, triton 3.x, then Frontier `-e --no-deps`.
#          If it cannot install on aarch64 the arm is DROPPED and said so (no substitute): exit 4 for that arm.
# SOURCE   cloned at the pinned commit; `git diff --quiet` before and after (Frontier's code is never edited);
#          the only addition is frontier_op.py write-config's untracked data/config/models/<name>.json.
# WRITES   $DEST/frontier.env (FRONTIER_SRC, FR_PY_PRIMARY, FR_PY_FIDELITY), $DEST/versions.json.
# EXIT     0 every requested arm PASS; 4 an arm failed (its FAIL line says which and why); 2 bad arguments.
set -euo pipefail

DEST="${MOE_HOME:-$HOME/moe}/frontier"
ARMS="primary,fidelity"
DRY=0
COMMIT="db5a85b"
URL="https://github.com/NetX-lab/Frontier"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "FAIL frontier_setup: unknown option $1" >&2; exit 2 ;;
  esac
done

run() { echo "  \$ $*"; if [[ $DRY -eq 0 ]]; then "$@"; fi; }
say() { echo "[frontier_setup] $*"; }

SRC="$DEST/src"
say "dest $DEST, arms $ARMS, commit $COMMIT"
if [[ $DRY -eq 0 ]]; then mkdir -p "$DEST"; fi
if [[ ! -d "$SRC/.git" ]]; then run git clone --quiet "$URL" "$SRC"; fi
run git -C "$SRC" fetch --quiet origin
run git -C "$SRC" checkout --quiet --detach "$COMMIT"
if [[ $DRY -eq 0 ]]; then
  head="$(git -C "$SRC" rev-parse HEAD)"
  [[ "$head" == "$COMMIT"* ]] || { echo "FAIL frontier_setup: HEAD $head is not $COMMIT"; exit 4; }
  git -C "$SRC" diff --quiet || { echo "FAIL frontier_setup: Frontier's tracked files are modified"; exit 4; }
  echo "PASS frontier source at $head, tracked files unmodified"
fi

rc=0
FR_PY_PRIMARY=""
FR_PY_FIDELITY=""
deps=(ddsketch'>=3.0,<4' fasteners'>=0.19,<1' pandas'>=1.5' plotly'>=5.0' scikit-learn'>=1.1' scipy'>=1.9' tqdm'>=4.64' pyyaml'>=6.0')

if [[ ",$ARMS," == *",primary,"* ]]; then
  V="$DEST/venv-primary"
  : "${PY_VLLM:?PY_VLLM unset: source ~/moe/env.sh (the study venv decides the primary stack)}"
  if [[ $DRY -eq 0 ]]; then
    pyver="$("$PY_VLLM" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    tver="$("$PY_VLLM" -c 'import torch; print(torch.__version__)')"
  else
    pyver="3.12"; tver="2.13.0+cu130"
  fi
  tag="${tver#*+}"; tbase="${tver%%+*}"
  # the freeze carries `-e file:///workspace/moe-kernels` (setup_runpod.sh sanitised_resolved drops it the same way)
  RES="$DEST/.resolved-vllm.txt"
  if [[ $DRY -eq 0 ]]; then grep -v '^-e ' "$REPO/requirements/resolved-vllm.txt" > "$RES"; fi
  set +e
  (
    set -e
    run uv venv --python "$pyver" "$V"
    run uv pip install --python "$V/bin/python" --index-url "https://download.pytorch.org/whl/$tag" "torch==$tbase"
    run uv pip install --python "$V/bin/python" -r "$RES"
    run uv pip install --python "$V/bin/python" -c "$RES" "${deps[@]}"
    run uv pip install --python "$V/bin/python" --no-deps -e "$SRC"
  )
  prc=$?
  set -e
  if [[ $prc -ne 0 ]]; then
    echo "FAIL frontier primary venv: install exited $prc"; rc=4
  elif [[ $DRY -eq 0 ]]; then
    got="$("$V/bin/python" -c 'import vllm, torch; print(vllm.__version__, torch.__version__)')"
    if [[ "$got" != "0.27.1 $tver" ]]; then
      echo "FAIL frontier primary venv: vLLM/torch are '$got', want '0.27.1 $tver'"; rc=4
    else
      api="$(cd "$SRC" && PYTHONPATH="$SRC" "$V/bin/python" -c 'import frontier.profiling.moe.moe_vllm_kernel as K; print(K.VLLM_API_VERSION)' 2>/dev/null | tail -1)"
      echo "PASS frontier primary venv: vLLM 0.27.1, torch $tver, Frontier's vLLM API branch: $api"
      FR_PY_PRIMARY="$V/bin/python"
    fi
  else
    FR_PY_PRIMARY="$V/bin/python"
  fi
fi

if [[ ",$ARMS," == *",fidelity,"* ]]; then
  V="$DEST/venv-fidelity"
  set +e
  (
    set -e
    run uv venv --python 3.10 "$V"
    run uv pip install --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0"
    run uv pip install --python "$V/bin/python" "vllm==0.10.2" "flashinfer-python>=0.3,<0.4" "triton>=3,<4" "torch==2.8.0" "${deps[@]}"
    run uv pip install --python "$V/bin/python" --no-deps -e "$SRC"
  )
  frc=$?
  set -e
  if [[ $frc -ne 0 ]]; then
    echo "FAIL frontier fidelity venv (DROPPED, no substitute): install exited $frc on $(uname -m)"; rc=4
  elif [[ $DRY -eq 0 ]]; then
    got="$("$V/bin/python" -c 'import vllm, torch; print(vllm.__version__, torch.__version__)' 2>&1 | tail -1)"
    if [[ "$got" != 0.10.2\ 2.8.0* ]]; then
      echo "FAIL frontier fidelity venv (DROPPED): imports '$got'"; rc=4
    else
      echo "PASS frontier fidelity venv: $got"
      FR_PY_FIDELITY="$V/bin/python"
    fi
  else
    FR_PY_FIDELITY="$V/bin/python"
  fi
fi

if [[ $DRY -eq 0 ]]; then
  git -C "$SRC" diff --quiet || { echo "FAIL frontier_setup: Frontier's tracked files changed during setup"; exit 4; }
  cat > "$DEST/frontier.env" <<EOF
export FRONTIER_SRC="$SRC"
export FR_PY_PRIMARY="$FR_PY_PRIMARY"
export FR_PY_FIDELITY="$FR_PY_FIDELITY"
EOF
  vers() { [[ -n "$1" ]] && "$1" -c 'import vllm, torch; print(vllm.__version__, torch.__version__)' 2>/dev/null | tail -1; }
  python3 - "$DEST/versions.json" "$(git -C "$SRC" rev-parse HEAD)" "$(vers "$FR_PY_PRIMARY")" "$(vers "$FR_PY_FIDELITY")" <<'PYEOF'
import json, sys
json.dump({"commit": sys.argv[2], "primary": sys.argv[3], "fidelity": sys.argv[4]}, open(sys.argv[1], "w"), indent=2)
PYEOF
  say "wrote $DEST/frontier.env and versions.json"
fi
exit $rc
