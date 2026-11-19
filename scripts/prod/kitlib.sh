# Sourced by scripts/prod/smoke.sh, main.sh and offline.sh: the step runner, the clock lock and nsys location.
# Every step prints exactly one "PASS <step>", "FAIL <step>: <reason>" or "SKIPPED <step>: <reason>" line, logs to
# $KIT_LOG/<step>.log ($KIT_LOG = <kit_init dir>/logs, created by kit_init), and a hard FAIL stops the sequence
# (fail-fast) with the step's exit code. A soft step (Frontier, the secondary runs: reported and dropped, no
# substitute) prints "FAIL (soft, dropped)" and goes on. --dry-run prints every step's command and runs nothing.
#
#   step NAME [--soft] --cap SECONDS [--needs STEP[,STEP]] -- CMD...
#
# CAP (review 2, MAJOR C): every step runs under `timeout --kill-after=60 CAP` (GNU coreutils; the whole process
# group: nsys and the python under it), CAP the step's REGISTERED wall cap in seconds, written at the step in the
# script (tests/test_prod_scripts.py holds every step to having one). A step without --cap is refused (exit 2, dry
# run included). A step that runs past its cap is a TIMEOUT: FAIL on a hard step, FAIL (soft, dropped) on a soft one,
# so a hung soft step cannot hold the steps after it. A shell function (kit_lock, kit_push, ...) runs in a child bash
# that is handed the kit's functions.
# NEEDS (review 2, MINOR 6): a step whose named prerequisite FAILED or was SKIPPED is not run: it is recorded
# "SKIPPED NAME: needs PRE (failed)" (a capture after a failed lock never runs at the wrong clock).

KIT_DRY=${KIT_DRY:-0}
KIT_STEPS=()
KIT_BAD=" "
KIT_TIMEOUT="${KIT_TIMEOUT:-timeout}"
KIT_KILL_AFTER="${KIT_KILL_AFTER:-60}"

kit_init() {   # kit_init DIR: the summary is DIR/SUMMARY.txt, the step logs DIR/logs/ (created); $R is the caller's
  KIT_DIR="$1"
  KIT_LOG="$1/logs"
  if [[ $KIT_DRY -eq 0 ]]; then mkdir -p "$KIT_LOG"; fi
  SUMMARY="$1/SUMMARY.txt"
}

kit_record() {
  KIT_STEPS+=("$1")
  echo "$1"
  if [[ $KIT_DRY -eq 0 ]]; then echo "$1" >> "$SUMMARY"; fi
}

step() {
  local name="$1"; shift
  local soft=0 cap="" needs="" pre
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --soft) soft=1; shift ;;
      --cap) cap="${2:-}"; shift 2 ;;
      --needs) needs="${2:-}"; shift 2 ;;
      --) shift; break ;;
      *) break ;;
    esac
  done
  if ! [[ "$cap" =~ ^[0-9]+$ ]] || [[ "$cap" -le 0 ]]; then
    echo "FAIL $name: no registered --cap SECONDS (every step runs under a timeout)" >&2
    exit 2
  fi
  if [[ $KIT_DRY -eq 1 ]]; then
    printf 'STEP %-26s %scap=%s %s%s\n' "$name" "$([[ $soft -eq 1 ]] && echo '[soft] ')" "$cap" \
      "$([[ -n "$needs" ]] && echo "needs=$needs ")" "$(printf '%q ' "$@")"
    return 0
  fi
  for pre in ${needs//,/ }; do
    if [[ "$KIT_BAD" == *" $pre "* ]]; then
      KIT_BAD+="$name "
      kit_record "SKIPPED $name: needs $pre (failed or skipped)"
      return 0
    fi
  done
  local log="$KIT_LOG/$name.log" t0 rc runner
  t0=$(date +%s)
  if [[ "$(type -t "$1")" == function ]]; then
    export NSYS REPO PY_VLLM KIT_PUSH_SCRIPT 2>/dev/null || true
    runner=(bash -c "$(declare -f); \"\$@\"" "kit-step-$name" "$@")
  else
    runner=("$@")
  fi
  set +e
  "$KIT_TIMEOUT" --kill-after="$KIT_KILL_AFTER" "$cap" "${runner[@]}" > "$log" 2>&1
  rc=$?
  set -e
  local dt=$(( $(date +%s) - t0 ))
  if [[ $rc -eq 0 ]]; then
    kit_record "PASS $name (${dt}s)"
    return 0
  fi
  KIT_BAD+="$name "
  local why
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    why="TIMEOUT: over its registered cap of ${cap}s (killed)"
  else
    # (|| true: a log with no such line must not end the sequence under set -e / pipefail)
    why="$(grep -E '^(FAIL|REFUSED|ERROR|Traceback)|Error:' "$log" | tail -3 | tr '\n' ' ' | cut -c1-400 || true)"
    [[ -z "$why" ]] && why="$(tail -2 "$log" | tr '\n' ' ' | cut -c1-300 || true)"
  fi
  if [[ $soft -eq 1 ]]; then
    kit_record "FAIL (soft, dropped) $name: exit $rc: $why"
    return 0
  fi
  kit_record "FAIL $name: exit $rc: $why"
  echo "fail-fast: stopping after $name; logs in $KIT_LOG" >&2
  exit "$rc"
}

kit_locate_nsys() {
  NSYS="$(command -v nsys 2>/dev/null || true)"
  if [[ -z "$NSYS" ]]; then
    for c in /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda*/nsight-systems*/bin/nsys; do
      [[ -x "$c" ]] && { NSYS="$c"; break; }
    done
  fi
  [[ -n "$NSYS" ]] || NSYS=nsys
}

# SM clock lock / unlock (root via sudo -n, as locked_r3.py does); the read-back is the gate
kit_lock() {
  local f="$1"
  sudo -n nvidia-smi -pm 1 >/dev/null
  sudo -n nvidia-smi -lgc "$f,$f" | tee /dev/stderr | grep -q "GPU clocks set to" || { echo "FAIL lock: -lgc $f did not take"; return 3; }
  echo "PASS lock $f MHz"
}
kit_unlock() { sudo -n nvidia-smi -rgc >/dev/null && echo "PASS unlock (default clocks)"; }

kit_power_check() {   # kit_power_check WATTS: the enforced power limit must read back WATTS (CRITIQUE MINOR)
  local want="$1" got
  got="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1 | tr -d ' ')"
  if [[ "$got" != "$want" ]]; then echo "FAIL power: enforced limit $got W, want $want W"; return 3; fi
  echo "PASS power limit $got W"
}

kit_env_check() {   # the VM: vLLM 0.27.1 in $PY_VLLM, a GH200, nsys
  : "${PY_VLLM:?FAIL env: PY_VLLM unset (source ~/moe/env.sh)}"
  local v
  v="$("$PY_VLLM" -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)"
  [[ "$v" == "0.27.1" ]] || { echo "FAIL env: vLLM in \$PY_VLLM is '$v', not 0.27.1"; return 3; }
  "$PY_VLLM" -c 'import triton, torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "triton", triton.__version__, torch.cuda.get_device_name(0))'
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -q GH200 || { echo "FAIL env: not a GH200"; return 3; }
  "$NSYS" --version || { echo "FAIL env: no nsys"; return 3; }
  echo "PASS env: vLLM 0.27.1, GH200, $("$NSYS" --version | head -1)"
}

kit_dry_plan() {   # a tool's own --dry-run exits 2 (REFUSED: "nothing measured") by the study's convention
  set +e; "$@"; local rc=$?; set -e
  [[ $rc -eq 0 || $rc -eq 2 ]] || { echo "FAIL plan: exit $rc"; return $rc; }
  echo "PASS plan (the tool's dry run, exit $rc)"
}

# kit_push MESSAGE [--final RC]: the VM pushes its results to the run's branch (scripts/vm_results_push.sh,
# registered and pinned, initialised by the launcher; KIT_PUSH_SCRIPT overrides it in the tests). main.sh calls it
# as a SOFT step after each measurement block, so a lost rental keeps what it measured
kit_push() {
  local p="${KIT_PUSH_SCRIPT:-$REPO/scripts/vm_results_push.sh}"
  [[ -f "$p" ]] || { echo "FAIL push: no $p"; return 3; }
  bash "$p" push --message "$@"
}

# whatever stops the sequence, the card is left unlocked (review 1, MAJOR 5) and, with --push (main.sh), the
# results are pushed one last time with the sequence's exit code (review 2, MAJOR A: a hard FAIL used to exit with
# the unlock only, leaving what the earlier blocks measured unpushed). Soft: its failure never changes the exit code.
kit_trap_unlock() {
  [[ $KIT_DRY -eq 1 ]] && return 0
  KIT_TRAP_PUSH=0
  [[ "${1:-}" == "--push" ]] && KIT_TRAP_PUSH=1
  trap 'kit_on_exit $?' EXIT
}
kit_on_exit() {
  local rc="$1"
  set +e
  sudo -n nvidia-smi -rgc >/dev/null 2>&1
  if [[ "${KIT_TRAP_PUSH:-0}" -eq 1 ]]; then
    local log="/dev/null"
    [[ -n "${KIT_LOG:-}" && -d "${KIT_LOG:-}" ]] && log="$KIT_LOG/final-push.log"
    echo "exit $rc: the final push (soft) -> $log" >&2
    export REPO KIT_PUSH_SCRIPT
    "$KIT_TIMEOUT" --kill-after=60 "${KIT_PUSH_CAP:-3600}" bash -c "$(declare -f); kit_push \"\$@\"" kit-final-push \
      "$(basename "$0") exit $rc: final push" --final "$rc" >> "$log" 2>&1 \
      || echo "the final push failed (soft): $log" >&2
  fi
  exit "$rc"
}

kit_summary() {
  echo "==== $(basename "$0") summary ($KIT_DIR)"
  printf '%s\n' "${KIT_STEPS[@]+"${KIT_STEPS[@]}"}"
}
