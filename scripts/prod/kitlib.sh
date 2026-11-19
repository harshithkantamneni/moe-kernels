# Sourced by scripts/prod/smoke.sh and main.sh: the step runner, the clock lock and nsys location.
# Every step prints exactly one "PASS <step>" or "FAIL <step>: <reason>" line, logs to $R/logs/<step>.log, and a
# hard FAIL stops the sequence (fail-fast) with the step's exit code. A soft step (Frontier's fidelity arm, which
# may not install on aarch64: reported and dropped, no substitute) prints "FAIL (soft, dropped)" and goes on.
# --dry-run prints every step's command and runs nothing.

KIT_DRY=${KIT_DRY:-0}
KIT_STEPS=()

kit_init() {   # kit_init RESULTS_DIR
  R="$1"
  if [[ $KIT_DRY -eq 0 ]]; then mkdir -p "$R/logs"; fi
  SUMMARY="$R/SUMMARY.txt"
}

kit_record() {
  KIT_STEPS+=("$1")
  echo "$1"
  if [[ $KIT_DRY -eq 0 ]]; then echo "$1" >> "$SUMMARY"; fi
}

# step NAME [--soft] -- CMD...   (the command's stdout+stderr go to the log; its last FAIL/REFUSED line is the reason)
step() {
  local name="$1"; shift
  local soft=0
  if [[ "${1:-}" == "--soft" ]]; then soft=1; shift; fi
  [[ "${1:-}" == "--" ]] && shift
  if [[ $KIT_DRY -eq 1 ]]; then
    printf 'STEP %-26s %s\n' "$name" "$(printf '%q ' "$@")"
    return 0
  fi
  local log="$R/logs/$name.log" t0 rc
  t0=$(date +%s)
  set +e
  "$@" > "$log" 2>&1
  rc=$?
  set -e
  local dt=$(( $(date +%s) - t0 ))
  if [[ $rc -eq 0 ]]; then
    kit_record "PASS $name (${dt}s)"
    return 0
  fi
  local why
  why="$(grep -E '^(FAIL|REFUSED|ERROR|Traceback)|Error:' "$log" | tail -3 | tr '\n' ' ' | cut -c1-400)"
  [[ -z "$why" ]] && why="$(tail -2 "$log" | tr '\n' ' ' | cut -c1-300)"
  if [[ $soft -eq 1 ]]; then
    kit_record "FAIL (soft, dropped) $name: exit $rc: $why"
    return 0
  fi
  kit_record "FAIL $name: exit $rc: $why"
  echo "fail-fast: stopping after $name; logs in $R/logs" >&2
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

kit_summary() {
  echo "==== $(basename "$0") summary ($R)"
  printf '%s\n' "${KIT_STEPS[@]+"${KIT_STEPS[@]}"}"
}
