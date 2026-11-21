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
# OUT (smoke rental 2026-11-19, RESUME): `--out PATH[,PATH]` declares what the step writes that a later step reads
# (`a|b`: either one). ALWAYS: `--always` marks a cheap state-setting or state-checking step (preflight, env, power,
# lock, unlock) that runs again on a resume.
# RESUME: `--from STEP --results DIR` (kit_resume_init) re-enters a sequence at STEP in an existing results dir:
# every step before STEP that is not --always is recorded "SKIPPED-RESUME NAME" and not run. A skipped HARD step
# whose declared output is missing REFUSES the resume (exit 2, before anything runs past it); a skipped SOFT step
# whose output is missing stays dropped (its --needs dependents are SKIPPED). A --from naming no step is refused.
# BACKGROUND (main.sh --profile blind, 2026-11-20): `step NAME --bg ... -- CMD` starts the step under its cap and
# returns at once ("STARTED NAME"); `step_join NAME` waits and records it exactly as a foreground step (PASS, FAIL
# (soft, dropped), or FAIL: fail-fast), so KIT_BAD and --needs keep working; a step that --needs a running
# background step joins it first. kit_summary joins what is still running; the EXIT trap kills it. Only CPU or
# network work goes in the background: a GPU-timed step never overlaps another GPU-timed step.
# DEADLINE (extras.sh only, review of the extras M4): with KIT_DEADLINE set (epoch seconds; the caller sets it AFTER
# sourcing this file, so main.sh and smoke.sh never see one) a step that would start at or past it is recorded
# "SKIPPED NAME: past the deadline" and not run, and every other step's timeout is min(its cap, deadline - now). A step
# marked --no-deadline (the final push and the held-back check) keeps its own cap. The EXIT trap (unlock, push) is not
# a step: it always runs.
# DONE FILE (extras.sh only, M2): with KIT_DONE_FILE set, the EXIT trap writes "exit RC at UTC" there and pushes
# WITHOUT --final, so the push branch's DRIVER-DONE keeps meaning "main.sh ended" (vm_results_push.sh writes it).
# ENV (smoke rental 2026-11-19): kit_env PY exports TMPDIR=$MOE_HOME/tmp (created; nsys cannot use the VM's
# root-owned /tmp/nvidia) and puts PY's bin dir first on PATH (FlashInfer's JIT shells out to `ninja`), for every
# step the sequence runs; --dry-run prints both as ENV lines.

KIT_DRY=${KIT_DRY:-0}
KIT_STEPS=()
KIT_BAD=" "
KIT_TIMEOUT="${KIT_TIMEOUT:-timeout}"
KIT_KILL_AFTER="${KIT_KILL_AFTER:-60}"
KIT_FROM=""
KIT_RESUMING=0
KIT_DEADLINE=""
KIT_DONE_FILE=""
KIT_BG_OPEN=" "

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

# kit_missing_outputs "a,b|c": prints the declared outputs that are absent (an `x|y` item is present if either is)
kit_missing_outputs() {
  local item alt ok miss=() alts=() items=()
  IFS=',' read -r -a items <<< "$1"
  for item in ${items[@]+"${items[@]}"}; do
    [[ -z "$item" ]] && continue
    ok=0
    IFS='|' read -r -a alts <<< "$item"
    for alt in "${alts[@]}"; do [[ -e "$alt" ]] && { ok=1; break; }; done
    [[ $ok -eq 1 ]] || miss+=("$item")
  done
  echo "${miss[*]+"${miss[*]}"}"
}

step() {
  local name="$1"; shift
  local soft=0 cap="" needs="" outs="" always=0 nodl=0 bg=0 pre
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --soft) soft=1; shift ;;
      --bg) bg=1; shift ;;
      --no-deadline) nodl=1; shift ;;
      --always) always=1; shift ;;
      --cap) cap="${2:-}"; shift 2 ;;
      --needs) needs="${2:-}"; shift 2 ;;
      --out) outs="${2:-}"; shift 2 ;;
      --) shift; break ;;
      *) break ;;
    esac
  done
  if ! [[ "$cap" =~ ^[0-9]+$ ]] || [[ "$cap" -le 0 ]]; then
    echo "FAIL $name: no registered --cap SECONDS (every step runs under a timeout)" >&2
    exit 2
  fi
  if [[ $KIT_RESUMING -eq 1 ]]; then
    if [[ "$name" == "$KIT_FROM" ]]; then
      KIT_RESUMING=0
    elif [[ $always -eq 0 ]]; then
      local missing
      missing="$(kit_missing_outputs "$outs")"
      if [[ -n "$missing" ]]; then
        if [[ $soft -eq 1 ]]; then
          KIT_BAD+="$name "
          kit_record "SKIPPED-RESUME $name: output absent ($missing), stays dropped"
          return 0
        fi
        echo "RESUME REFUSED: --from $KIT_FROM skips $name, but its output is missing: $missing" >&2
        exit 2
      fi
      kit_record "SKIPPED-RESUME $name"
      return 0
    fi
  fi
  if [[ $KIT_DRY -eq 1 ]]; then
    printf 'STEP %-26s %scap=%s %s%s%s%s\n' "$name" "$([[ $soft -eq 1 ]] && echo '[soft] ')" "$cap" \
      "$([[ $always -eq 1 ]] && echo 'always ')$([[ $nodl -eq 1 ]] && echo 'nodeadline ')$([[ $bg -eq 1 ]] && echo 'bg ')" \
      "$([[ -n "$needs" ]] && echo "needs=$needs ")" \
      "$([[ -n "$outs" ]] && echo "out=$outs ")" "$(printf '%q ' "$@")"
    return 0
  fi
  for pre in ${needs//,/ }; do
    # a prerequisite still running in the background is joined first (its PASS/FAIL decides this step)
    if [[ "$KIT_BG_OPEN" == *" $pre "* ]]; then step_join "$pre"; fi
    if [[ "$KIT_BAD" == *" $pre "* ]]; then
      KIT_BAD+="$name "
      kit_record "SKIPPED $name: needs $pre (failed or skipped)"
      return 0
    fi
  done
  local log="$KIT_LOG/$name.log" t0 rc runner
  t0=$(date +%s)
  if [[ -n "$KIT_DEADLINE" && $nodl -eq 0 ]]; then
    local left=$(( KIT_DEADLINE - t0 ))
    if [[ $left -le 0 ]]; then
      KIT_BAD+="$name "
      kit_record "SKIPPED $name: past the deadline ($(( -left ))s past $KIT_DEADLINE)"
      return 0
    fi
    [[ $left -lt $cap ]] && cap=$left
  fi
  if [[ "$(type -t "$1")" == function ]]; then
    export NSYS REPO PY_VLLM KIT_PUSH_SCRIPT 2>/dev/null || true
    runner=(bash -c "$(declare -f); \"\$@\"" "kit-step-$name" "$@")
  else
    runner=("$@")
  fi
  if [[ $bg -eq 1 ]]; then
    if [[ "$KIT_BG_OPEN" == *" $name "* ]]; then echo "FAIL $name: already running in the background" >&2; exit 2; fi
    mkdir -p "$KIT_LOG/.bg"
    "$KIT_TIMEOUT" --kill-after="$KIT_KILL_AFTER" "$cap" "${runner[@]}" > "$log" 2>&1 < /dev/null &
    printf '%s %s %s %s\n' "$!" "$t0" "$soft" "$cap" > "$KIT_LOG/.bg/$name"
    KIT_BG_OPEN+="$name "
    echo "STARTED $name (background, pid $!)"
    return 0
  fi
  set +e
  "$KIT_TIMEOUT" --kill-after="$KIT_KILL_AFTER" "$cap" "${runner[@]}" > "$log" 2>&1
  rc=$?
  set -e
  kit_finish "$name" "$rc" "$(( $(date +%s) - t0 ))" "$soft" "$cap" "$log"
}

# kit_finish NAME RC SECONDS SOFT CAP LOG: one step's result line, KIT_BAD, and fail-fast on a hard FAIL
kit_finish() {
  local name="$1" rc="$2" dt="$3" soft="$4" cap="$5" log="$6"
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

# step_join NAME: wait for a background step (`step NAME --bg ...`) and record it exactly as a foreground step:
# PASS / FAIL (soft, dropped) / FAIL into SUMMARY and KIT_BAD, a hard FAIL stops the sequence. A name that is not
# running (never started: resumed past, skipped by --needs, or already joined) is a no-op. Dry run: a JOIN line.
step_join() {
  local name="$1" pid t0 soft cap rc
  if [[ $KIT_DRY -eq 1 ]]; then echo "JOIN $name"; return 0; fi
  [[ "$KIT_BG_OPEN" == *" $name "* ]] || return 0
  read -r pid t0 soft cap < "$KIT_LOG/.bg/$name"
  set +e
  wait "$pid"
  rc=$?
  set -e
  KIT_BG_OPEN="${KIT_BG_OPEN/ $name / }"
  rm -f "$KIT_LOG/.bg/$name"
  kit_finish "$name" "$rc" "$(( $(date +%s) - t0 ))" "$soft" "$cap" "$KIT_LOG/$name.log"
}

# every background step still running is joined (kit_summary), or killed (the EXIT trap, a hard FAIL elsewhere)
kit_join_all() {
  local n
  for n in $KIT_BG_OPEN; do step_join "$n"; done
}
kit_kill_bg() {
  local n pid _r
  for n in $KIT_BG_OPEN; do
    [[ -f "$KIT_LOG/.bg/$n" ]] || continue
    read -r pid _r < "$KIT_LOG/.bg/$n"
    kill "$pid" 2>/dev/null && echo "killed background step $n (pid $pid): the sequence stopped" >&2
  done
}

# kit_resume_init FROM SCRIPT_ARGS...: enter the sequence at step FROM (empty: a fresh run). The step must exist in
# the script's own plan for these arguments (its --dry-run without --from), and --results must name the run's dir
kit_resume_init() {
  local from="$1"; shift
  [[ -n "$from" ]] || return 0
  [[ -n "${KIT_RESULTS_GIVEN:-}" ]] || { echo "RESUME REFUSED: --from needs --results DIR (the run to resume)" >&2; exit 2; }
  local args=() plan
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --from) shift 2 ;;
      --dry-run) shift ;;
      *) args+=("$1"); shift ;;
    esac
  done
  plan="$(bash "$0" --dry-run ${args[@]+"${args[@]}"} 2>/dev/null | awk '$1 == "STEP" {print $2}')"
  if ! grep -qxF -- "$from" <<< "$plan"; then
    echo "RESUME REFUSED: --from $from: no such step in $(basename "$0")'s plan" >&2
    exit 2
  fi
  KIT_FROM="$from"
  KIT_RESUMING=1
  local line
  line="RESUME from $from at $(date -u +%Y-%m-%dT%H:%M:%SZ) in $KIT_DIR"
  echo "$line"
  if [[ $KIT_DRY -eq 0 ]]; then echo "$line" >> "$SUMMARY"; fi
}

# kit_env PY: every step runs with TMPDIR=$KIT_TMP (created) and PY's bin dir first on PATH (smoke 2026-11-19)
kit_env() {
  local py="$1" bin=""
  KIT_TMP="${KIT_TMPDIR:-${MOE_HOME:-$HOME/moe}/tmp}"
  if [[ $KIT_DRY -eq 0 ]]; then mkdir -p "$KIT_TMP"; fi
  export TMPDIR="$KIT_TMP"
  if [[ "$py" == */* ]]; then
    bin="$(dirname "$py")"
    export PATH="$bin:$PATH"
  fi
  echo "ENV TMPDIR=$TMPDIR"
  echo "ENV PATH first=${bin:-(PY has no dir: PATH unchanged)}"
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

# kit_power_check WATTS: the ENFORCED power limit must read back WATTS (CRITIQUE MINOR). Not power.limit: the GH200
# reports power.limit 900 (the default setting) while it enforces 700 (smoke rental 2026-11-19)
kit_power_check() {
  local want="$1" got
  got="$(nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1 | tr -d ' ')"
  if [[ "$got" != "$want" ]]; then echo "FAIL power: enforced limit $got W, want $want W"; return 3; fi
  echo "PASS power limit $got W"
}

kit_env_check() {   # the VM: vLLM 0.27.1 in $PY_VLLM, a GH200, nsys
  : "${PY_VLLM:?FAIL env: PY_VLLM unset (source ~/moe/env.sh)}"
  local v
  v="$("$PY_VLLM" -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)"
  [[ "$v" == "0.27.1" ]] || { echo "FAIL env: vLLM in \$PY_VLLM is '$v', not 0.27.1"; return 3; }
  "$PY_VLLM" -c 'import triton, torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "triton", triton.__version__, torch.cuda.get_device_name(0))'
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,enforced.power.limit --format=csv
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
# TRACES (review m6): with KIT_TRACES (the run's raw traces, outside the pushed tree) and KIT_RESULTS set, each push
# first records the traces' sha256 in $KIT_RESULTS/traces.sha256.json (prodlib.py trace-manifest; soft: a failed
# manifest is reported and the push goes on)
# CHUNKS (v10 blind run 2026-11-21): when the pinned script leaves its commit unpushed because GitHub refused a pack
# over 2 GiB, the same tree is pushed in commits under the cap (scripts/prod/push_chunks.sh; KIT_CHUNK_SCRIPT
# overrides it in the tests), and kit_push returns that script's exit
kit_push() {
  local p="${KIT_PUSH_SCRIPT:-$REPO/scripts/vm_results_push.sh}"
  [[ -f "$p" ]] || { echo "FAIL push: no $p"; return 3; }
  if [[ -n "${KIT_TRACES:-}" && -n "${KIT_RESULTS:-}" ]]; then
    "${PY_VLLM:-python3}" "$REPO/scripts/prod/prodlib.py" trace-manifest --traces "$KIT_TRACES" \
      --out "$KIT_RESULTS/traces.sha256.json" || echo "WARN push: the trace manifest failed (the push goes on)"
  fi
  local rc=0
  bash "$p" push --message "$@" || rc=$?
  local err="${MOE_PUSH_HOME:-$HOME/moe}/push/.git/push-error.txt"
  if [[ $rc -eq 1 && -f "$err" ]] && grep -q "pack exceeds maximum allowed size" "$err"; then
    echo "push: GitHub refused a pack over 2 GiB: pushing the same tree in parts (push_chunks.sh)"
    rc=0
    bash "${KIT_CHUNK_SCRIPT:-$REPO/scripts/prod/push_chunks.sh}" || rc=$?
  fi
  return "$rc"
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
  kit_kill_bg
  sudo -n nvidia-smi -rgc >/dev/null 2>&1
  if [[ "${KIT_TRAP_PUSH:-0}" -eq 1 ]]; then
    local log="/dev/null"
    [[ -n "${KIT_LOG:-}" && -d "${KIT_LOG:-}" ]] && log="$KIT_LOG/final-push.log"
    echo "exit $rc: the final push (soft) -> $log" >&2
    export REPO KIT_PUSH_SCRIPT
    local fin=(--final "$rc")
    if [[ -n "${KIT_DONE_FILE:-}" ]]; then
      printf 'exit %s at %s: %s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(basename "$0")" > "$KIT_DONE_FILE" \
        || echo "could not write $KIT_DONE_FILE" >&2
      fin=()
    fi
    "$KIT_TIMEOUT" --kill-after=60 "${KIT_PUSH_CAP:-3600}" bash -c "$(declare -f); kit_push \"\$@\"" kit-final-push \
      "$(basename "$0") exit $rc: final push" ${fin[@]+"${fin[@]}"} >> "$log" 2>&1 \
      || echo "the final push failed (soft): $log" >&2
  fi
  exit "$rc"
}

kit_summary() {
  if [[ $KIT_RESUMING -eq 1 ]]; then echo "RESUME REFUSED: --from $KIT_FROM was never reached" >&2; exit 2; fi
  kit_join_all
  echo "==== $(basename "$0") summary ($KIT_DIR)"
  printf '%s\n' "${KIT_STEPS[@]+"${KIT_STEPS[@]}"}"
}
