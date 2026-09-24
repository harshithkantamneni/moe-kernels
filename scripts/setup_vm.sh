#!/usr/bin/env bash
# A LAMBDA VM, SET UP FOR THE R3 COUNTER RUN, IN ONE COMMAND.
#
# A Lambda instance is a plain Ubuntu VM with root, Lambda Stack, a driver
# nobody knows until it boots, and NO network volume: its disk survives a
# reboot and is lost at termination. docs/LAMBDA.md is the runbook around this
# script (launch, run, exfiltrate, terminate, cost).
#
#   bash setup_vm.sh --commit <full sha> --bundle moe.bundle   # build, then preflight
#   bash setup_vm.sh --commit <full sha> --repo <git url>      # clone over the network instead
#   bash setup_vm.sh --commit <sha> --bundle moe.bundle --dry-run  # the plan; installs and writes nothing
#   bash setup_vm.sh --commit <sha> --check           # verify clone, venvs and ncu, then preflight
#   bash setup_vm.sh --commit <sha> --preflight-only  # detect, then preflight
#
# Options: --home DIR (default $HOME/moe; nothing is ever placed under
# /workspace), --python 3.12, --torch-index auto|cu130|cu128,
# --counter-door auto|open|sudo|module, --envs base,vllm, --no-ncu-install,
# --fresh (re-resolve the requirement sets, which rewrites
# requirements/resolved-*.txt in the checkout).
#
# WHY A SCRIPT OF ITS OWN AND NOT A --vm MODE OF setup_runpod.sh. A VM has jobs a
# pod never has: clone at a pinned commit, choose the torch wheel index from the
# driver, apt, install ncu without touching the driver, pick a counter door, and
# prove a counter readable. Folding them into setup_runpod.sh would turn a
# volume-centric installer into two scripts behind one flag. The one job both
# share, building venvs from the resolved sets under content-hash stamps, stays
# in setup_runpod.sh and is CALLED, never copied.
#
# TWO STAGES, SO THE COMMIT BEING MEASURED BUILDS ITS OWN ENVIRONMENT. Stage 1
# runs from wherever the operator copied this file: it detects the box, refuses
# what cannot work, installs git/curl/gcc if missing, and clones or checks the
# repo at exactly --commit (HEAD == sha, empty `git status --porcelain`). It
# then execs THAT checkout's scripts/setup_vm.sh with --stage2, which builds the
# venvs, ncu, the counter door and $HOME/moe/env.sh, and runs the preflight.
#
#   S0 DETECT     read-only, always: OS, arch, root/sudo, nvidia-smi (name,
#                 count, memory, driver, the header's CUDA version), Lambda
#                 Stack (recorded, never used), RAM, disk, gcc, the module's
#                 RestrictProfilingToAdminUsers and this process's CapEff.
#                 REFUSED unless nvidia-smi exits 0 and lists a card with a
#                 driver version: "Failed to initialize NVML: Driver/library
#                 version mismatch" is an unreachable driver, not a card.
#   S1 REPO       clone (--bundle or --repo) and check out --detach <sha>;
#                 skipped when HEAD == sha and clean; REFUSED when dirty, at
#                 another sha, or the sha is not a full, known commit.
#   S2 DECIDE     the torch wheel index and the ncu package, from the driver:
#                 r580+ cu130 (a cu130 wheel needs r580+, pod_session.sh P2c),
#                 570-579 cu128, below 570 REFUSED (this script never moves a
#                 driver). 570-579 is REFUSED under --torch-index auto too: the
#                 vllm venv's torch is resolved-vllm.txt's cu13 build, which
#                 needs r580+, so PF2 would fail it after the whole build; an
#                 explicit --torch-index cu128 builds that untested path by
#                 request. Python 3.12, the pod rows' interpreter.
#   S3 SYSTEM     git, curl, and build-essential only if gcc is missing (Triton
#                 builds its launcher with it). Runs in stage 1, because S1
#                 needs git.
#   S4 VENVS      setup_runpod.sh --base-python 3.12 base vllm, with the torch
#                 pin read off requirements/resolved-base.txt, the index from
#                 S2, the framework interpreter pinned and MOE_HOST_KIND=vm.
#                 Idempotent through setup_runpod.sh's own stamps.
#   S5 NCU        located the way the alpha(G) chain locates it
#                 (alpha_g_chain_helpers.py ncu-locate); if absent, the CUDA
#                 apt repo's nsight-compute package, installed only after
#                 `apt-get -s` shows the transaction touches no driver package.
#   S6 DOOR       how the counter steps run: `open` (the module already allows
#                 it), `sudo` (the default with root or passwordless sudo), or
#                 `module` (opt-in: rewrite the module option and reload the
#                 driver, refused while any process holds the GPU).
#   S7 ENV        $HOME/moe/env.sh: interpreters, roots, caches, ncu first on
#                 PATH, the counter launcher, the card. And the exfil line.
#   S8 PREFLIGHT  scripts/vm_preflight.py, never cached: PASS only when a
#                 counter was actually READ (dram_counter_route.py --probe
#                 --family r3-arms says OPEN) and the census fits. Writes
#                 $HOME/moe/session/PREFLIGHT.txt and PREFLIGHT.json.
#
# EXIT CODES, the repo's table (moe/bench/exit_codes.py; a test holds these to
# it): 0 ready, 1 the preflight did not pass (the box cannot take the
# measurement yet; PREFLIGHT.txt says which check), 2 REFUSED before anything
# was spent (no GPU, a driver nvidia-smi cannot reach, driver below 570, or
# below 580 under --torch-index auto, a dirty or other checkout, an apt
# transaction that would move the driver, root needed and absent, not enough
# disk), 4 a step crashed.
#
# TEST SEAMS, read-only paths a test points at fixtures: MOE_NVIDIA_PARAMS
# (/proc/driver/nvidia/params), MOE_PROC_STATUS (/proc/self/status),
# MOE_OS_RELEASE (/etc/os-release), MOE_MEMINFO (/proc/meminfo).
set -euo pipefail

EXIT_DONE=0
EXIT_CLAIM_FAIL=1
EXIT_REFUSED=2
EXIT_ERROR=4

#: THE WHEEL-INDEX RULE, pod_session.sh P2c's (a cu130 torch wheel needs driver
#: r580+; on an older driver use the cu128 index; the index matches the DRIVER,
#: not the image). A test holds CU130_MIN_DRIVER to P2c's threshold. Below
#: CU128_MIN_DRIVER no index this repo has measured on runs, and the remedy is a
#: driver upgrade this script never performs.
CU130_MIN_DRIVER=580
CU128_MIN_DRIVER=570
TORCH_INDEX_ROOT="https://download.pytorch.org/whl"
#: The interpreter the pod rows were measured under (profiles/q2_kernel_names.txt:
#: /usr/bin/python3.12), and so the one the resolved sets were frozen under.
DEFAULT_PYTHON=3.12
#: Where ncu is looked for after PATH: the alpha(G) chain's own NCU_SEARCH
#: default (scripts/alpha_g_chain.sh; a test holds this to a superset of it),
#: plus the runfile layout that keeps ncu under the toolkit's own directory.
NCU_SEARCH_DEFAULT="/usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda*/nsight-compute*/ncu"
#: An apt transaction that installs, removes or purges any package named here
#: moves the driver or the userspace that must match it, which is the one thing
#: installing ncu must never do. The rule is by family, not a list of known
#: names: nvidia-utils-580, libnvidia-cfg1-580, nvidia-open-580 or
#: nvidia-fabricmanager-580 moving under a loaded module is exactly how a VM
#: ends at "Driver/library version mismatch". Only the ncu packages themselves
#: (NCU_PACKAGES_RE) are exempt, and they match none of these prefixes today.
DRIVER_PACKAGES_RE='^(nvidia-|libnvidia-|libcuda|cuda-drivers|cuda-compat|linux-modules-nvidia|linux-objects-nvidia|linux-signatures-nvidia|xserver-xorg-video-nvidia)'
NCU_PACKAGES_RE='^(nsight-compute|cuda-nsight-compute)'
CUDA_REPO_ROOT="https://developer.download.nvidia.com/compute/cuda/repos"
CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
MODPROBE_CONF="/etc/modprobe.d/moe-ncu-profiling.conf"

NVIDIA_PARAMS="${MOE_NVIDIA_PARAMS:-/proc/driver/nvidia/params}"
PROC_STATUS="${MOE_PROC_STATUS:-/proc/self/status}"
OS_RELEASE="${MOE_OS_RELEASE:-/etc/os-release}"
MEMINFO="${MOE_MEMINFO:-/proc/meminfo}"
NCU_SEARCH="${NCU_SEARCH:-$NCU_SEARCH_DEFAULT}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG_ARGS=("$@")

COMMIT=""
SRC_REPO=""
SRC_BUNDLE=""
MOE_HOME=""
PYVER="$DEFAULT_PYTHON"
TORCH_INDEX_ARG="auto"
DOOR_ARG="auto"
ENVS_ARG="base,vllm"
NCU_INSTALL=1
DRY_RUN=0
CHECK=0
PREFLIGHT_ONLY=0
FRESH=0
STAGE2=0

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '/^set /d' | sed 's/^# \{0,1\}//'; }

need_value() { [[ -n "${2:-}" && "${2:0:2}" != "--" ]] || { echo "[vm] $1 needs a value" >&2; exit "$EXIT_REFUSED"; }; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)          need_value "$1" "${2:-}"; COMMIT="$2"; shift 2 ;;
    --repo)            need_value "$1" "${2:-}"; SRC_REPO="$2"; shift 2 ;;
    --bundle)          need_value "$1" "${2:-}"; SRC_BUNDLE="$2"; shift 2 ;;
    --home)            need_value "$1" "${2:-}"; MOE_HOME="$2"; shift 2 ;;
    --python)          need_value "$1" "${2:-}"; PYVER="$2"; shift 2 ;;
    --torch-index)     need_value "$1" "${2:-}"; TORCH_INDEX_ARG="$2"; shift 2 ;;
    --counter-door)    need_value "$1" "${2:-}"; DOOR_ARG="$2"; shift 2 ;;
    --envs)            need_value "$1" "${2:-}"; ENVS_ARG="$2"; shift 2 ;;
    --no-ncu-install)  NCU_INSTALL=0; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    --check)           CHECK=1; shift ;;
    --preflight-only)  PREFLIGHT_ONLY=1; shift ;;
    --fresh)           FRESH=1; shift ;;
    --stage2)          STAGE2=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)                 echo "[vm] unknown argument: $1 (see --help)" >&2; exit "$EXIT_REFUSED" ;;
  esac
done

MOE_HOME="${MOE_HOME:-$HOME/moe}"
REPO="$MOE_HOME/repo"
VENVS="$MOE_HOME/venvs"
SESSION="$MOE_HOME/session"
RESULTS="$MOE_HOME/results"
ENV_SH="$MOE_HOME/env.sh"
PY_BASE="$VENVS/base/bin/python"
PY_VLLM="$VENVS/vllm/bin/python"
# The caches setup_runpod.sh would put under $WORKSPACE, named here once and
# handed to it explicitly, so env.sh and the venv build cannot disagree.
HF_CACHE="$MOE_HOME/hf-cache"
TRITON_CACHE="$MOE_HOME/triton-cache"
INDUCTOR_CACHE="$MOE_HOME/torchinductor-cache"
UV_CACHE="$MOE_HOME/uv-cache"
IFS=',' read -r -a ENVS <<< "$ENVS_ARG"

say()   { printf '[vm] %s\n' "$*"; }
trim()  { local v="$1"; v="${v#"${v%%[![:space:]]*}"}"; v="${v%"${v##*[![:space:]]}"}"; printf '%s' "$v"; }
stage() { printf '\n== %s\n' "$*"; }

REFUSALS=()
#: A refusal: exit 2 now, or under --dry-run collect it, finish the plan, and
#: exit 2 at the end, so a laptop rehearsal shows every refusal at once.
refuse() {
  printf '[vm] REFUSED: %s\n' "$*" >&2
  if (( DRY_RUN )); then REFUSALS+=("$*"); return 0; fi
  exit "$EXIT_REFUSED"
}
fail()  { printf '[vm] ERROR: %s\n' "$*" >&2; exit "$EXIT_ERROR"; }

#: Print a command the plan would run, or run it. Every side effect goes
#: through here or through as_root, so --dry-run writes nothing.
run() {
  if (( DRY_RUN )); then printf '[vm]   would run: %s\n' "$*"; return 0; fi
  printf '[vm]   $ %s\n' "$*"
  "$@"
}

# --------------------------------------------------------------------------
# the arguments that decide everything else
# --------------------------------------------------------------------------
if (( CHECK + PREFLIGHT_ONLY + DRY_RUN > 1 )); then
  refuse "--dry-run, --check and --preflight-only are separate modes; pick one"
  exit "$EXIT_REFUSED"
fi
if [[ -z "$COMMIT" ]]; then
  echo "[vm] REFUSED: --commit <full sha> is required: the environment is built by the script AT the commit being measured" >&2
  exit "$EXIT_REFUSED"
fi
if ! [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[vm] REFUSED: --commit $COMMIT is not a full 40-hex sha; a short sha can name two commits, so it is never resolved here" >&2
  exit "$EXIT_REFUSED"
fi
if [[ -n "$SRC_REPO" && -n "$SRC_BUNDLE" ]]; then
  echo "[vm] REFUSED: --repo and --bundle both given; name one source" >&2
  exit "$EXIT_REFUSED"
fi
case "$TORCH_INDEX_ARG" in auto|cu130|cu128) ;; *)
  echo "[vm] REFUSED: --torch-index $TORCH_INDEX_ARG is not auto, cu130 or cu128" >&2; exit "$EXIT_REFUSED" ;; esac
case "$DOOR_ARG" in auto|open|sudo|module) ;; *)
  echo "[vm] REFUSED: --counter-door $DOOR_ARG is not auto, open, sudo or module" >&2; exit "$EXIT_REFUSED" ;; esac
for env in "${ENVS[@]}"; do
  case "$env" in base|vllm|sglang) ;; *)
    echo "[vm] REFUSED: --envs names $env; the environments are base, vllm and sglang" >&2; exit "$EXIT_REFUSED" ;; esac
done
if [[ -n "$SRC_BUNDLE" && "$SRC_BUNDLE" != /* ]]; then SRC_BUNDLE="$PWD/$SRC_BUNDLE"; fi
# A dry run writes nothing, and that includes the bytecode a helper's import
# would leave beside the repo's modules.
(( DRY_RUN )) && export PYTHONDONTWRITEBYTECODE=1

# --------------------------------------------------------------------------
# S0 DETECT: read-only, always
# --------------------------------------------------------------------------
OS_ID=""; OS_VERSION=""; ARCH=""; UID_NOW=""; SUDO_STATE=""
GPU_COUNT=0; GPU_NAME=""; GPU_MEM_MIB=""; GPU_UUID=""; DRIVER=""; DRIVER_MAJOR=""
SMI_RC=""; SMI_FIRST=""
CUDA_DRIVER=""; LAMBDA_STACK=""; SYS_TORCH=""; RAM_GB=""; DISK_GB=""; HAVE_GCC=0
RESTRICT=""; CAP_EFF=""; CAP_PERFMON=""; CAP_SYS_ADMIN=""

#: The nearest existing ancestor of a path, for `df` on a directory not made yet.
existing_ancestor() { local p="$1"; while [[ ! -e "$p" ]]; do p="$(dirname "$p")"; done; printf '%s\n' "$p"; }

detect() {
  stage "S0 DETECT (read-only)"
  if [[ -r "$OS_RELEASE" ]]; then
    # shellcheck source=/dev/null
    OS_ID="$(. "$OS_RELEASE" && printf '%s' "${ID:-}")"
    # shellcheck source=/dev/null
    OS_VERSION="$(. "$OS_RELEASE" && printf '%s' "${VERSION_ID:-}")"
  fi
  ARCH="$(uname -m)"
  UID_NOW="$(id -u)"
  if [[ "$UID_NOW" == 0 ]]; then
    SUDO_STATE="root"
  elif ! command -v sudo >/dev/null 2>&1; then
    SUDO_STATE="none"
  elif (( DRY_RUN )); then
    # A dry run invokes no sudo at all, not even `sudo -n true`.
    SUDO_STATE="unverified"
  elif sudo -n true >/dev/null 2>&1; then
    SUDO_STATE="passwordless"
  else
    SUDO_STATE="needs-password"
  fi
  say "os                  ${OS_ID:-unknown} ${OS_VERSION:-} ($ARCH)"
  say "user                uid $UID_NOW, sudo: $SUDO_STATE"

  if command -v nvidia-smi >/dev/null 2>&1; then
    local q row name mem drv uuid extra
    SMI_RC=0
    q="$(nvidia-smi --query-gpu=name,memory.total,driver_version,uuid \
           --format=csv,noheader,nounits 2>&1)" || SMI_RC=$?
    SMI_FIRST="$(grep -m1 . <<< "$q" || true)"
    # A card is a row of exactly four fields whose driver is a version number.
    # Anything else nvidia-smi prints ("Failed to initialize NVML: ...", a
    # warning) is not a card, and counting it as one named an NVML error as
    # the GPU and read no driver.
    while IFS= read -r row; do
      IFS=',' read -r name mem drv uuid extra <<< "$row"
      drv="$(trim "${drv:-}")"
      [[ -z "${extra:-}" && -n "$(trim "${uuid:-}")" && "$drv" =~ ^[0-9]+\. ]] || continue
      GPU_COUNT=$((GPU_COUNT + 1))
      if (( GPU_COUNT == 1 )); then
        GPU_NAME="$(trim "$name")"; GPU_MEM_MIB="$(trim "$mem")"
        DRIVER="$drv"; GPU_UUID="$(trim "$uuid")"
        DRIVER_MAJOR="${DRIVER%%.*}"
      fi
    done <<< "$q"
    CUDA_DRIVER="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: *[0-9.]*' | awk '{print $NF}' | head -1 || true)"
  fi
  say "gpu                 ${GPU_COUNT}x ${GPU_NAME:-none} ${GPU_MEM_MIB:+(${GPU_MEM_MIB} MiB)} ${GPU_UUID}"
  say "driver              ${DRIVER:-none} (CUDA ${CUDA_DRIVER:-unknown} per nvidia-smi)"

  if command -v dpkg-query >/dev/null 2>&1; then
    LAMBDA_STACK="$(dpkg-query -W -f='${Version}' lambda-stack-cuda 2>/dev/null || true)"
  fi
  SYS_TORCH="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
  say "lambda stack        ${LAMBDA_STACK:-not installed} (system torch ${SYS_TORCH:-none}; recorded, never used)"

  if [[ -r "$MEMINFO" ]]; then
    RAM_GB="$(awk '/^MemTotal:/{printf "%d", $2/1048576}' "$MEMINFO")"
  fi
  DISK_GB="$(df -Pk "$(existing_ancestor "$MOE_HOME")" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')"
  command -v gcc >/dev/null 2>&1 && HAVE_GCC=1
  say "host                ${RAM_GB:-?} GB RAM, ${DISK_GB:-?} GB free at $(existing_ancestor "$MOE_HOME"), gcc $([[ $HAVE_GCC == 1 ]] && echo present || echo MISSING)"

  if [[ -r "$NVIDIA_PARAMS" ]]; then
    RESTRICT="$(grep -m1 RestrictProfilingToAdminUsers "$NVIDIA_PARAMS" | grep -o '[0-9][0-9]*' | tail -1 || true)"
  fi
  if [[ -r "$PROC_STATUS" ]]; then
    CAP_EFF="$(awk '/^CapEff:/{print $2}' "$PROC_STATUS")"
    if [[ -n "$CAP_EFF" ]]; then
      CAP_SYS_ADMIN=$(( (16#$CAP_EFF >> 21) & 1 ))
      CAP_PERFMON=$(( (16#$CAP_EFF >> 38) & 1 ))
    fi
  fi
  say "counter gate        RestrictProfilingToAdminUsers=${RESTRICT:-unread} ($NVIDIA_PARAMS); CapEff ${CAP_EFF:-unread} (CAP_SYS_ADMIN ${CAP_SYS_ADMIN:-?}, CAP_PERFMON ${CAP_PERFMON:-?})"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    refuse "no GPU: nvidia-smi is not on PATH; there is nothing to measure on this box"
  elif (( SMI_RC != 0 )) || [[ ! "$DRIVER" =~ ^[0-9]+\. ]]; then
    local remedy=""
    if grep -q 'version mismatch' <<< "$q"; then
      remedy=". A Driver/library version mismatch is NVIDIA userspace upgraded under the loaded kernel module (an unattended upgrade does this): a reboot loads the matching module, or rent another instance"
    fi
    refuse "no GPU reachable: nvidia-smi exited $SMI_RC and listed no card with a driver version (${SMI_FIRST:-it printed nothing}), so no wheel index, ncu package or card can be chosen$remedy"
  elif (( GPU_COUNT > 1 )); then
    say "note                $GPU_COUNT GPUs; the counter run uses GPU 0 only"
  fi
}

# --------------------------------------------------------------------------
# S2 DECIDE: pure functions of the driver
# --------------------------------------------------------------------------
#: The torch wheel index a driver major can run: cu130 at r580+, cu128 at
#: 570-579; returns 1 below that.
torch_index_for_driver() {
  local major="$1"
  [[ "$major" =~ ^[0-9]+$ ]] || return 1
  if (( major >= CU130_MIN_DRIVER )); then echo cu130
  elif (( major >= CU128_MIN_DRIVER )); then echo cu128
  else return 1
  fi
}

#: The CUDA apt repo's Nsight Compute package matching the same rule.
ncu_package_for_driver() {
  case "$(torch_index_for_driver "$1")" in
    cu130) echo cuda-nsight-compute-13-0 ;;
    cu128) echo cuda-nsight-compute-12-8 ;;
    *)     return 1 ;;
  esac
}

TORCH_INDEX=""; NCU_PACKAGE=""
decide() {
  stage "S2 DECIDE (from the driver)"
  if [[ -z "$DRIVER_MAJOR" ]]; then
    say "no driver read, so no wheel index and no ncu package can be chosen"
    return 0
  fi
  local rule
  if ! rule="$(torch_index_for_driver "$DRIVER_MAJOR")"; then
    refuse "driver $DRIVER is below r$CU128_MIN_DRIVER: no torch wheel index this repo has measured on (cu130 needs r$CU130_MIN_DRIVER+, cu128 r$CU128_MIN_DRIVER+) runs on it. Remedy: upgrade the driver yourself (this script never moves a driver), or rent another instance"
    return 0
  fi
  # A driver the base venv's index can serve is not yet a box the run can use:
  # the vllm venv's torch comes from requirements/resolved-vllm.txt, a cu13
  # build, whatever index the base venv takes.
  local vllm_dead=0
  (( DRIVER_MAJOR < CU130_MIN_DRIVER )) && vllm_dead=1
  case "$TORCH_INDEX_ARG" in
    auto)  if (( vllm_dead )); then
             refuse "driver $DRIVER is r$CU128_MIN_DRIVER-r$((CU130_MIN_DRIVER - 1)): the base venv could take the $rule index, but the vllm venv's torch comes from requirements/resolved-vllm.txt, a cu13 build that needs r$CU130_MIN_DRIVER+, so the preflight's PF2 would fail for it after the whole venv build. Remedy: rent an instance whose driver is r$CU130_MIN_DRIVER+ (this script never moves a driver), or pass --torch-index cu128 to build on the untested cu128 path anyway"
           fi
           TORCH_INDEX="$rule" ;;
    cu130) (( DRIVER_MAJOR >= CU130_MIN_DRIVER )) || refuse "--torch-index cu130 on driver $DRIVER: a cu130 wheel needs r$CU130_MIN_DRIVER+"
           TORCH_INDEX=cu130 ;;
    cu128) TORCH_INDEX=cu128 ;;
  esac
  NCU_PACKAGE="$(ncu_package_for_driver "$DRIVER_MAJOR")"
  say "torch index         $TORCH_INDEX_ROOT/$TORCH_INDEX  (driver $DRIVER, rule: $rule)"
  say "ncu package         $NCU_PACKAGE (if no ncu is found)"
  say "python              $PYVER"
  if (( vllm_dead )) && [[ "$TORCH_INDEX_ARG" == cu128 ]]; then
    say "OPT-IN: --torch-index cu128 on driver $DRIVER builds the untested cu128 path. The"
    say "        vllm venv's torch comes from requirements/resolved-vllm.txt, a cu13 build that"
    say "        needs r$CU130_MIN_DRIVER+, so the preflight's PF2 fails for the vllm venv. A cu128"
    say "        vLLM path is untested in this repo."
  fi
}

# --------------------------------------------------------------------------
# root, when a step needs it
# --------------------------------------------------------------------------
have_root() { [[ "$SUDO_STATE" == root || "$SUDO_STATE" == passwordless || "$SUDO_STATE" == unverified ]]; }

as_root() {
  if (( DRY_RUN )); then printf '[vm]   would run as root: %s\n' "$*"; return 0; fi
  case "$SUDO_STATE" in
    root)          printf '[vm]   # %s\n' "$*"; "$@" ;;
    passwordless)  printf '[vm]   $ sudo %s\n' "$*"; sudo -n "$@" ;;
    *)             refuse "root needed for '$*' and this user has neither root nor passwordless sudo ($SUDO_STATE)" ;;
  esac
}

# --------------------------------------------------------------------------
# S3 SYSTEM
# --------------------------------------------------------------------------
system_packages() {
  stage "S3 SYSTEM"
  local need=()
  command -v git  >/dev/null 2>&1 || need+=(git)
  command -v curl >/dev/null 2>&1 || need+=(curl)
  [[ "$HAVE_GCC" == 1 ]] || need+=(build-essential)
  if (( ${#need[@]} == 0 )); then
    say "S3 skip: git, curl and gcc are present"
    return 0
  fi
  have_root || refuse "S3 needs root to install ${need[*]}, and this user has neither root nor passwordless sudo"
  as_root apt-get update
  as_root apt-get install -y --no-install-recommends "${need[@]}"
}

# --------------------------------------------------------------------------
# S1 REPO
# --------------------------------------------------------------------------
#: Whether SOURCE (a bundle or a URL) holds COMMIT, without writing under
#: --home: a bundle's heads first, then a throwaway fetch into $TMPDIR. A URL is
#: not contacted by a dry run; its sha is checked when stage 1 fetches it.
source_has_commit() {
  local src="$1" tmp rc=1
  if [[ -n "$SRC_BUNDLE" ]]; then
    [[ -f "$src" ]] || { say "bundle $src does not exist"; return 1; }
    if git bundle list-heads "$src" 2>/dev/null | awk '{print $1}' | grep -qx "$COMMIT"; then
      say "bundle $src: $COMMIT is one of its heads"
      return 0
    fi
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/setup-vm-bundle.XXXXXX")"
    if git init -q "$tmp" 2>/dev/null \
       && git -C "$tmp" fetch -q "$src" '+refs/*:refs/remotes/bundle/*' 2>/dev/null \
       && git -C "$tmp" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
      say "bundle $src: contains $COMMIT"
      rc=0
    else
      say "bundle $src: does NOT contain $COMMIT"
    fi
    rm -rf "$tmp"
    return "$rc"
  fi
  say "repo $src: not contacted by a dry run; $COMMIT is checked when it is fetched"
  return 0
}

repo_state() {
  # clean | dirty | other:<sha> | absent
  if [[ ! -e "$REPO/.git" ]]; then echo absent; return; fi
  local head; head="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$head" != "$COMMIT" ]]; then echo "other:${head:-none}"; return; fi
  if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo dirty; return; fi
  echo clean
}

repo_stage() {
  stage "S1 REPO ($REPO at $COMMIT)"
  local state src; state="$(repo_state)"
  src="${SRC_BUNDLE:-$SRC_REPO}"
  case "$state" in
    clean)
      say "S1 skip: $REPO is at $COMMIT and clean"
      return 0 ;;
    dirty)
      refuse "$REPO is at $COMMIT with local changes (git status --porcelain is not empty); every page measured from it would be stamped dirty. Commit them on the laptop, or remove the checkout"
      return 0 ;;
    other:*)
      refuse "$REPO is at ${state#other:}, not $COMMIT. This script does not move a checkout: run git -C $REPO fetch <source> '+refs/heads/*:refs/remotes/origin/*' && git -C $REPO checkout --detach $COMMIT yourself, or pass --home to a fresh directory"
      return 0 ;;
  esac
  if [[ -z "$src" ]]; then
    refuse "no checkout at $REPO and neither --bundle nor --repo to clone one from"
    return 0
  fi
  if (( DRY_RUN )); then
    source_has_commit "$src" || refuse "$src does not contain $COMMIT; bundle the branch that holds it (git bundle create moe.bundle <branch>)"
    run git init -q "$REPO.partial"
    run git -C "$REPO.partial" fetch "$src" "+refs/heads/*:refs/remotes/origin/*"
    run git -C "$REPO.partial" checkout -q --detach "$COMMIT"
    run mv "$REPO.partial" "$REPO"
    return 0
  fi
  # Into a partial directory, moved into place only once HEAD == sha and the
  # tree is clean, so an interrupted or refused clone never leaves a checkout
  # that the next run would take for a finished one.
  local part="$REPO.partial"
  rm -rf "$part"
  mkdir -p "$MOE_HOME"
  run git init -q "$part"
  run git -C "$part" fetch "$src" '+refs/heads/*:refs/remotes/origin/*' || fail "git could not fetch $src"
  if [[ -n "$SRC_REPO" ]]; then git -C "$part" remote add origin "$SRC_REPO"; fi
  if ! git -C "$part" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
    rm -rf "$part"
    refuse "$src does not contain $COMMIT"
  fi
  run git -C "$part" checkout -q --detach "$COMMIT" || fail "git could not check out $COMMIT"
  [[ "$(git -C "$part" rev-parse HEAD)" == "$COMMIT" ]] || fail "HEAD is not $COMMIT after the checkout"
  [[ -z "$(git -C "$part" status --porcelain)" ]] || fail "a fresh checkout of $COMMIT is not clean"
  mv "$part" "$REPO"
  say "S1 done: $REPO at $COMMIT, clean"
}

# --------------------------------------------------------------------------
# S4 VENVS, through setup_runpod.sh
# --------------------------------------------------------------------------
#: The torch the base venv pins, off requirements/resolved-base.txt: the pin
#: the pod rows were measured on, read and never typed.
torch_pin() {
  local f="$1/requirements/resolved-base.txt"
  [[ -f "$f" ]] || return 1
  awk -F'==' '/^torch==/{sub(/\+.*/, "", $2); print $2; exit}' "$f"
}

runpod_env() {
  local index="$TORCH_INDEX_ROOT/$TORCH_INDEX"
  [[ -n "$TORCH_INDEX" ]] || index="<the index S2 picks from the driver>"
  printf '%s\n' "WORKSPACE=$MOE_HOME" "MOE_VENV_ROOT=$VENVS" "HF_HOME=$HF_CACHE" \
    "TRITON_CACHE_DIR=$TRITON_CACHE" "TORCHINDUCTOR_CACHE_DIR=$INDUCTOR_CACHE" \
    "UV_CACHE_DIR=$UV_CACHE" "MOE_BASE_TORCH=torch==$1" \
    "MOE_TORCH_INDEX=$index" "MOE_FRAMEWORK_PYTHON=$PYVER" "MOE_HOST_KIND=vm"
}

venv_stage() {
  local repo="$1" pin
  stage "S4 VENVS (${ENVS[*]}, via setup_runpod.sh)"
  # runpod_env's placeholder is the plan's wording for the index; a real build
  # handed it would pass it to uv as an --index-url. S0 refuses a box with no
  # driver read, so this is the second lock on the same door.
  if [[ -z "$TORCH_INDEX" ]] && (( ! DRY_RUN )); then
    refuse "S2 chose no torch wheel index (no driver was read), so no venv is built: uv would be handed the plan's placeholder as an index"
  fi
  if ! pin="$(torch_pin "$repo")" || [[ -z "$pin" ]]; then
    if (( DRY_RUN )); then
      pin="<the torch== line of $REPO/requirements/resolved-base.txt>"
    else
      refuse "$repo/requirements/resolved-base.txt names no torch==; the base venv's pin is read from it"
    fi
  fi
  local -a envv args
  mapfile_compat envv < <(runpod_env "$pin")
  args=(--base-python "$PYVER")
  (( FRESH )) && args+=(--fresh)
  args+=("${ENVS[@]}")
  if (( FRESH )); then
    say "WARNING: --fresh re-resolves, and setup_runpod.sh then writes requirements/resolved-*.txt"
    say "         INTO THE CHECKOUT: every page measured afterwards is stamped dirty until the"
    say "         new sets are copied back to the laptop and committed."
  fi
  if (( DRY_RUN )); then
    say "  would run: env ${envv[*]} PATH=\$HOME/.local/bin:\$PATH bash $REPO/scripts/setup_runpod.sh ${args[*]}"
    return 0
  fi
  # uv, where setup_runpod.sh's own installer puts it, found on the next run too.
  envv+=("PATH=$HOME/.local/bin:$PATH")
  mkdir -p "$SESSION/logs"
  local log="$SESSION/logs/setup_runpod.log" rc=0
  say "  $ env ... bash $repo/scripts/setup_runpod.sh ${args[*]}   (log: $log)"
  env "${envv[@]}" bash "$repo/scripts/setup_runpod.sh" "${args[@]}" 2>&1 | tee "$log" || rc=$?
  local bad=() env
  for env in "${ENVS[@]}"; do
    [[ -x "$VENVS/$env/bin/python" && -f "$VENVS/.stamp-$env" ]] || bad+=("$env")
  done
  if (( ${#bad[@]} )); then
    if grep -q 'ABORT: only' "$log"; then
      refuse "not enough disk for ${bad[*]} (setup_runpod.sh's space_for; see $log)"
    fi
    fail "setup_runpod.sh (exit $rc) left ${bad[*]} unbuilt; read $log"
  fi
  say "S4 done: ${ENVS[*]} built and stamped (a later run skips them on the stamp)"
}

#: `mapfile` without bash 4 (a laptop's /bin/bash is 3.2).
mapfile_compat() {
  local __name="$1" __line; eval "$__name=()"
  while IFS= read -r __line; do eval "$__name+=(\"\$__line\")"; done
}

# --------------------------------------------------------------------------
# S5 NCU
# --------------------------------------------------------------------------
NCU_BIN="none"; NCU_WHERE="none"; NCU_CANDS="none"

#: The python that runs the repo's helpers: the base venv once it exists, the
#: system python3 before (the helpers import no torch).
helper_python() { if [[ -x "$PY_BASE" ]]; then echo "$PY_BASE"; else command -v python3; fi; }

#: alpha_g_chain_helpers.py ncu-locate, the chain's own search, from REPO.
locate_ncu() {
  local repo="$1" out=""
  if [[ -f "$repo/scripts/alpha_g_chain_helpers.py" ]]; then
    out="$("$(helper_python)" "$repo/scripts/alpha_g_chain_helpers.py" ncu-locate "$NCU_SEARCH" 2>/dev/null || true)"
  fi
  [[ -n "$out" ]] || out=$'none\tnone\tnone'
  IFS=$'\t' read -r NCU_BIN NCU_WHERE NCU_CANDS <<< "$out"
}

cuda_repo_dir() {
  [[ "$OS_ID" == ubuntu && -n "$OS_VERSION" ]] || return 1
  local arch
  case "$ARCH" in x86_64) arch=x86_64 ;; aarch64|arm64) arch=sbsa ;; *) return 1 ;; esac
  printf '%s/ubuntu%s/%s\n' "$CUDA_REPO_ROOT" "${OS_VERSION//./}" "$arch"
}

#: The driver packages an `apt-get -s` transcript would install, remove or
#: purge, one per line; empty when the transaction leaves the driver alone.
driver_packages_touched() {
  awk '$1=="Inst" || $1=="Remv" || $1=="Purg" {print $2}' \
    | grep -E "$DRIVER_PACKAGES_RE" | grep -Ev "$NCU_PACKAGES_RE" || true
}

ncu_stage() {
  local repo="$1"
  stage "S5 NCU"
  locate_ncu "$repo"
  if [[ "$NCU_BIN" != none ]]; then
    say "S5 skip install: ncu at $NCU_BIN (found by $NCU_WHERE)"
  elif [[ ! -f "$repo/scripts/alpha_g_chain_helpers.py" ]]; then
    say "ncu is located at stage 2 by the checkout's alpha_g_chain_helpers.py ncu-locate (PATH, then $NCU_SEARCH)"
    say "  if none is found: $NCU_PACKAGE from the CUDA apt repo, after apt-get -s shows no driver package moves"
    return 0
  elif (( ! NCU_INSTALL )); then
    say "no ncu (PATH, then $NCU_SEARCH) and --no-ncu-install: the preflight will fail its ncu checks"
    return 0
  else
    install_ncu
    (( DRY_RUN )) && return 0
    locate_ncu "$repo"
    if [[ "$NCU_BIN" == none ]] && command -v dpkg-query >/dev/null 2>&1; then
      # Wherever the package put it, by the package's own file list.
      local pkgs cand
      pkgs="$(dpkg-query -W -f='${Package}\n' 'nsight-compute*' 'cuda-nsight-compute*' 2>/dev/null || true)"
      # shellcheck disable=SC2086  # a word list of package names
      cand="$( [[ -n "$pkgs" ]] && dpkg -L $pkgs 2>/dev/null | grep -E '/ncu$' | while read -r f; do [[ -x "$f" && -f "$f" ]] && echo "$f"; done | head -1 || true)"
      if [[ -n "$cand" ]]; then NCU_BIN="$cand"; NCU_WHERE="dpkg -L $NCU_PACKAGE"; NCU_CANDS="$cand"; fi
    fi
    [[ "$NCU_BIN" != none ]] || fail "$NCU_PACKAGE installed and no ncu found on PATH, at $NCU_SEARCH, or in its file list"
    say "ncu installed: $NCU_BIN (found by $NCU_WHERE)"
  fi
  (( DRY_RUN )) && return 0
  [[ "$NCU_BIN" != none ]] || return 0
  mkdir -p "$SESSION"
  {
    echo "binary     $NCU_BIN"
    echo "found by   $NCU_WHERE"
    echo "candidates $NCU_CANDS"
    echo "--- ncu --version"
    "$NCU_BIN" --version 2>&1 || true
    echo "--- ncu --list-chips"
    "$NCU_BIN" --list-chips 2>&1 || true
  } > "$SESSION/ncu.txt"
  say "recorded $SESSION/ncu.txt: $(grep -m1 -E '^Version ' "$SESSION/ncu.txt" || echo 'ncu printed no Version line')"
}

install_ncu() {
  local dir sim touched
  [[ -n "$NCU_PACKAGE" ]] || { refuse "no ncu, and no driver was read to choose an ncu package by"; return 0; }
  if ! dir="$(cuda_repo_dir)"; then
    refuse "no ncu, and this is not an Ubuntu x86_64/aarch64 box (${OS_ID:-unknown} ${OS_VERSION:-} $ARCH), so there is no CUDA apt repo to install $NCU_PACKAGE from; install ncu yourself and re-run"
    return 0
  fi
  have_root || { refuse "installing $NCU_PACKAGE needs root, and this user has neither root nor passwordless sudo"; return 0; }
  say "no ncu found (PATH, then $NCU_SEARCH): installing $NCU_PACKAGE from $dir"
  if (( DRY_RUN )) || ! dpkg-query -W -f='${Status}' cuda-keyring 2>/dev/null | grep -q 'install ok installed'; then
    run curl -fsSL -o "${TMPDIR:-/tmp}/$CUDA_KEYRING_DEB" "$dir/$CUDA_KEYRING_DEB" || fail "could not download $dir/$CUDA_KEYRING_DEB"
    as_root dpkg -i "${TMPDIR:-/tmp}/$CUDA_KEYRING_DEB" || fail "could not install the CUDA apt keyring"
    as_root apt-get update || fail "apt-get update failed after adding the CUDA repo"
  else
    say "the CUDA apt keyring is already installed"
  fi
  if (( DRY_RUN )); then
    say "  would run: apt-get -s install --no-install-recommends $NCU_PACKAGE, and REFUSE if it names a driver package ($DRIVER_PACKAGES_RE)"
    as_root apt-get install -y --no-install-recommends "$NCU_PACKAGE"
    return 0
  fi
  # THE SIMULATION FIRST. Installing ncu must never move the driver: a new
  # driver under a running VM is a reboot at best and a dead instance at worst.
  sim="$(apt-get -s install --no-install-recommends "$NCU_PACKAGE" 2>&1)" \
    || { refuse "apt cannot resolve $NCU_PACKAGE: $(tail -1 <<< "$sim")"; return 0; }
  touched="$(driver_packages_touched <<< "$sim")"
  if [[ -n "$touched" ]]; then
    refuse "installing $NCU_PACKAGE would touch the driver ($(tr '\n' ' ' <<< "$touched")): not installed. Install an ncu that matches this driver by hand, then re-run"
  fi
  say "apt-get -s: $NCU_PACKAGE touches no driver package"
  as_root apt-get install -y --no-install-recommends "$NCU_PACKAGE" || fail "apt-get install $NCU_PACKAGE failed"
}

# --------------------------------------------------------------------------
# S6 COUNTER DOOR
# --------------------------------------------------------------------------
DOOR=""; LAUNCHER=""

door_stage() {
  stage "S6 COUNTER DOOR (--counter-door $DOOR_ARG)"
  local door="$DOOR_ARG"
  if [[ "$door" == auto ]]; then
    if [[ "$RESTRICT" == 0 ]]; then
      door=open
    elif have_root; then
      door=sudo
    else
      refuse "the module restricts counters to admin users (RestrictProfilingToAdminUsers=${RESTRICT:-unread}) and this user has neither root nor passwordless sudo"
      return 0
    fi
  fi
  case "$door" in
    open)
      DOOR=open; LAUNCHER=""
      if [[ "$RESTRICT" == 0 ]]; then
        say "door open: RestrictProfilingToAdminUsers=0, counters need no admin"
      else
        say "door open BY REQUEST with RestrictProfilingToAdminUsers=${RESTRICT:-unread}: the preflight's probe decides"
      fi ;;
    sudo)
      have_root || { refuse "--counter-door sudo needs root or passwordless sudo ($SUDO_STATE)"; return 0; }
      DOOR=sudo
      if [[ "$SUDO_STATE" == root ]]; then
        LAUNCHER=""
        say "door sudo: already root, the counter steps need no launcher"
      else
        LAUNCHER='sudo -E env PATH=$PATH HOME=$HOME'
        say "door sudo: counter steps run under: $LAUNCHER (results chowned back after)"
        if [[ "$SUDO_STATE" == unverified ]]; then
          say "  passwordless sudo is verified (sudo -n true) when the plan runs, not by a dry run"
        fi
      fi ;;
    module)
      module_door ;;
  esac
}

#: The opt-in door: set the module option and reload the driver, so counters
#: need no admin at all. Refused while any process holds the GPU, because a
#: reload under a live context kills it.
module_door() {
  have_root || { refuse "--counter-door module needs root or passwordless sudo ($SUDO_STATE)"; return 0; }
  if [[ "$RESTRICT" == 0 ]]; then
    DOOR=module; LAUNCHER=""
    say "door module: RestrictProfilingToAdminUsers is already 0; nothing to reload"
    return 0
  fi
  local apps
  apps="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$apps" ]]; then
    refuse "--counter-door module reloads the driver and a process holds the GPU ($(tr '\n' ';' <<< "$apps")); stop it first"
    return 0
  fi
  as_root sh -c "echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > $MODPROBE_CONF"
  as_root sh -c 'systemctl stop nvidia-persistenced 2>/dev/null || true'
  as_root modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia || fail "the nvidia modules would not unload; a reboot applies $MODPROBE_CONF instead"
  as_root modprobe nvidia || fail "modprobe nvidia failed after the reload"
  as_root modprobe nvidia_uvm || true
  as_root sh -c 'systemctl start nvidia-persistenced 2>/dev/null || true'
  if (( ! DRY_RUN )); then
    RESTRICT="$(grep -m1 RestrictProfilingToAdminUsers "$NVIDIA_PARAMS" | grep -o '[0-9][0-9]*' | tail -1 || true)"
    [[ "$RESTRICT" == 0 ]] || fail "after the reload RestrictProfilingToAdminUsers=${RESTRICT:-unread}, not 0"
  fi
  DOOR=module; LAUNCHER=""
  say "door module: $MODPROBE_CONF written and the driver reloaded; counters need no admin"
}

# --------------------------------------------------------------------------
# S7 ENV
# --------------------------------------------------------------------------
card_slug() {
  local repo="$1"
  [[ -n "$GPU_NAME" && -f "$repo/moe/bench/provenance.py" ]] || { echo unknown; return; }
  PYTHONPATH="$repo" "$(helper_python)" -c 'import sys; from moe.bench.provenance import card_slug; print(card_slug(sys.argv[1]))' "$GPU_NAME" 2>/dev/null || echo unknown
}

env_text() {
  local repo="$1" ncu_dir=""
  [[ "$NCU_BIN" != none ]] && ncu_dir="$(dirname "$NCU_BIN"):"
  cat <<EOF
# Written by scripts/setup_vm.sh at $COMMIT. Source it: . $ENV_SH
export MOE_HOME="$MOE_HOME"
export REPO="$REPO"
export WORKSPACE="$MOE_HOME"
export PY_BASE="$PY_BASE"
export PY_VLLM="$PY_VLLM"
export RESULTS_ROOT="$RESULTS"
export MOE_RESULTS_DIR="$RESULTS"
export SESSION_ROOT="$SESSION"
export HF_HOME="$HF_CACHE"
export TRITON_CACHE_DIR="$TRITON_CACHE"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_CACHE"
export UV_CACHE_DIR="$UV_CACHE"
export NCU_SEARCH="$NCU_SEARCH"
export PATH="${ncu_dir}\$HOME/.local/bin:\$PATH"
export MOE_COUNTER_DOOR="$DOOR"
export MOE_COUNTER_LAUNCHER="$LAUNCHER"
export MOE_CARD="$(card_slug "$repo")"
# A counter step: \$MOE_COUNTER_LAUNCHER, then the results handed back to you
# (the sudo door leaves root-owned files and caches behind).
moe_counter() {
  local rc=0
  \$MOE_COUNTER_LAUNCHER "\$@" || rc=\$?
  if [[ -n "\$MOE_COUNTER_LAUNCHER" ]]; then
    sudo -n chown -R "\$(id -u):\$(id -g)" "\$MOE_HOME/session" "\$MOE_HOME/results" \\
      "\$TRITON_CACHE_DIR" "\$TORCHINDUCTOR_CACHE_DIR" "\$HF_HOME" 2>/dev/null || true
    for d in "\$HOME/.cache" "\$HOME/.triton" "\$HOME/.nv"; do
      [[ -e "\$d" ]] && sudo -n chown -R "\$(id -u):\$(id -g)" "\$d" 2>/dev/null || true
    done
  fi
  return \$rc
}
EOF
}

env_stage() {
  local repo="$1"
  stage "S7 ENV ($ENV_SH)"
  if (( DRY_RUN )); then
    say "would write $ENV_SH:"
    env_text "$repo" | sed 's/^/    /'
  else
    mkdir -p "$SESSION" "$RESULTS"
    env_text "$repo" > "$ENV_SH.tmp"
    mv "$ENV_SH.tmp" "$ENV_SH"
    say "wrote $ENV_SH (source it before any measurement)"
  fi
  say "EXFIL before terminating (the disk dies with the instance), from the laptop:"
  say "  rsync -avz ubuntu@<instance ip>:$MOE_HOME/results ubuntu@<instance ip>:$MOE_HOME/session ./lambda-exfil/"
}

# --------------------------------------------------------------------------
# S8 PREFLIGHT
# --------------------------------------------------------------------------
chown_back() {
  [[ "$DOOR" == sudo && -n "$LAUNCHER" ]] || return 0
  local d
  for d in "$SESSION" "$RESULTS" "$TRITON_CACHE" "$INDUCTOR_CACHE" "$HF_CACHE" \
           "$HOME/.cache" "$HOME/.triton" "$HOME/.nv"; do
    [[ -e "$d" ]] && sudo -n chown -R "$(id -u):$(id -g)" "$d" 2>/dev/null || true
  done
}

preflight_stage() {
  local repo="$1" rc=0
  stage "S8 PREFLIGHT (never cached)"
  if (( DRY_RUN )); then
    say "would run: . $ENV_SH && python3 $REPO/scripts/vm_preflight.py"
    say "  PF1 card, PF2 wheel vs driver, PF3 stack, PF4 ncu metrics, PF5 the r3-arms probe"
    say "  (must read OPEN), PF6 the census, PF7 host RAM; exit 0 only when all pass"
    return 0
  fi
  [[ -f "$ENV_SH" ]] || fail "no $ENV_SH: run setup_vm.sh without --preflight-only first"
  # shellcheck disable=SC1090
  . "$ENV_SH"
  "$(helper_python)" "$repo/scripts/vm_preflight.py" || rc=$?
  chown_back
  return "$rc"
}

# --------------------------------------------------------------------------
# --check: verify, install nothing
# --------------------------------------------------------------------------
check_stage() {
  local repo="$1" short=0 env
  stage "CHECK (installs nothing)"
  for env in "${ENVS[@]}"; do
    if [[ -x "$VENVS/$env/bin/python" && -f "$VENVS/.stamp-$env" ]]; then
      say "venv $env: built and stamped"
    else
      say "venv $env: MISSING or unstamped"; short=1
    fi
  done
  locate_ncu "$repo"
  if [[ "$NCU_BIN" != none ]]; then
    say "ncu: $NCU_BIN (found by $NCU_WHERE)"
  else
    say "ncu: MISSING"; short=1
  fi
  if [[ -f "$ENV_SH" ]]; then say "env.sh: $ENV_SH"; else say "env.sh: MISSING"; short=1; fi
  return "$short"
}

# --------------------------------------------------------------------------
# the stages, in order
# --------------------------------------------------------------------------
say "setup_vm.sh $( ((STAGE2)) && echo 'stage 2' || echo 'stage 1'), $SELF_DIR/setup_vm.sh, commit $COMMIT, home $MOE_HOME$( ((DRY_RUN)) && echo ', DRY RUN: nothing is installed or written')"

detect
if (( ! PREFLIGHT_ONLY )); then decide; fi

if (( DRY_RUN )); then
  system_packages
  repo_stage
  # The rest of the plan, as THIS copy would run it; stage 2 runs the commit's.
  PLAN_REPO="$(cd "$SELF_DIR/.." && pwd)"
  venv_stage "$PLAN_REPO"
  ncu_stage "$PLAN_REPO"
  door_stage
  env_stage "$PLAN_REPO"
  preflight_stage "$PLAN_REPO"
  if (( ${#REFUSALS[@]} )); then
    echo
    say "PLAN REFUSED (${#REFUSALS[@]}):"
    for r in "${REFUSALS[@]}"; do say "  - $r"; done
    exit "$EXIT_REFUSED"
  fi
  echo
  say "plan complete; re-run without --dry-run to build"
  exit "$EXIT_DONE"
fi

if (( ! STAGE2 )); then
  if (( CHECK || PREFLIGHT_ONLY )); then
    state="$(repo_state)"
    if [[ "$state" != clean ]]; then
      echo "[vm] CHECK FAILED: $REPO is $state, not a clean checkout at $COMMIT" >&2
      exit "$EXIT_CLAIM_FAIL"
    fi
  else
    system_packages
    repo_stage
  fi
  [[ -f "$REPO/scripts/setup_vm.sh" ]] \
    || refuse "$COMMIT has no scripts/setup_vm.sh: it predates this setup, and a commit is measured with the environment its own script builds"
  say "stage 1 done: exec $REPO/scripts/setup_vm.sh --stage2"
  exec bash "$REPO/scripts/setup_vm.sh" --stage2 "${ORIG_ARGS[@]}"
fi

# stage 2: this file is the checkout's own copy
SELF_REPO="$(cd "$SELF_DIR/.." && pwd -P)"
[[ "$SELF_REPO" == "$(cd "$REPO" 2>/dev/null && pwd -P)" ]] \
  || fail "--stage2 runs only from $REPO/scripts/setup_vm.sh, not from $SELF_REPO"
[[ "$(repo_state)" == clean ]] || refuse "$REPO is $(repo_state), not a clean checkout at $COMMIT"

if (( PREFLIGHT_ONLY )); then
  pf_rc=0
  preflight_stage "$REPO" || pf_rc=$?
  exit "$pf_rc"
fi
if (( CHECK )); then
  check_rc=0
  check_stage "$REPO" || check_rc=$?
  pf_rc=0
  preflight_stage "$REPO" || pf_rc=$?
  (( check_rc == 0 && pf_rc == 0 )) && exit "$EXIT_DONE"
  exit "$EXIT_CLAIM_FAIL"
fi

system_packages
venv_stage "$REPO"
ncu_stage "$REPO"
door_stage
env_stage "$REPO"
pf_rc=0
preflight_stage "$REPO" || pf_rc=$?
echo
if (( pf_rc == 0 )); then
  say "READY: a counter was read on this box. . $ENV_SH, then docs/LAMBDA.md's run section"
  exit "$EXIT_DONE"
fi
say "NOT READY: the preflight exited $pf_rc; $SESSION/PREFLIGHT.txt names each check"
(( pf_rc == EXIT_CLAIM_FAIL )) && exit "$EXIT_CLAIM_FAIL"
exit "$EXIT_ERROR"
