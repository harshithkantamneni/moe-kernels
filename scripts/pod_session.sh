#!/usr/bin/env bash
# The whole pod session, as one gated script. Type this and nothing else.
#
#   bash scripts/pod_session.sh --dry-run          # laptop, free, no GPU, no vLLM
#   bash scripts/pod_session.sh --preflight-only   # pod, ~3 min, spends almost nothing
#   bash scripts/pod_session.sh --label alpha-0558 # the real session
#   bash scripts/pod_session.sh --from 5           # resume after a crash
#   bash scripts/pod_session.sh --only 6 --force   # redo one step
#   bash scripts/pod_session.sh --only 2b          # just the alias ablation
#
# WHY THIS EXISTS RATHER THAN A LIST OF COMMANDS IN A DOC. Three separate
# failures on this project were not wrong hypotheses but a session that produced
# plausible numbers from a broken instrument, and every one of them was invisible
# at the terminal: a calibration written 28 minutes AFTER the sweep it was
# published beside (docs/INSTRUMENTATION.md defect 7, cost claim C5 its target
# for three days), a shared TRITON_CACHE_DIR that made a PTX dump emit nothing
# while reporting success, and an unanchored `plots/` line in .gitignore that
# silently dropped every published figure on `git add`. None of those is caught
# by reading a table. Each is caught by a number compared against a threshold,
# which is what this script does and why every step prints PASS or FAIL rather
# than output for a human to eyeball at 02:00 with the meter running.
#
# THE SCIENCE THIS SESSION TESTS. Arithmetic intensity of an MoE expert GEMM is
#
#     AI(r) = (2r/b) / Q(r),   Q(r) = 1 + alpha (ceil(r/BM) - 1)
#
# because each extra M-tile re-reads that expert's weights, discounted by L2 by
# alpha. Two consequences: AI is BOUNDED at 2 BM / (alpha b), and the crossing
# solves R = ridge b Q(R)/2, a step function on both sides.
#
# alpha was refit on 2026-09-01 from 0.10 to 0.558 (90% band 0.529-0.588, 10,813
# rows, placebo -0.002); the 0.10 was an estimator artefact. At 0.558 the ceiling
# puts BLOCK_M of 16, 32 AND 64 below the measured ridge band 160.3-176.2, so
# only 128 and 256 can ever cross. The two alphas differ QUALITATIVELY, not by a
# few percent, which is what makes this session worth a pod:
#
#     BLOCK_M=32   AI cap  57   NO CROSSING AT ALL
#     BLOCK_M=64   AI cap 115   NO CROSSING AT ALL
#     BLOCK_M=128  AI cap 229   R_cross 250   (mixtral 999 tok, qwen2 1998, ds-v3 7992)
#     BLOCK_M=256  AI cap 459   R_cross 160   (mixtral 641 tok, qwen2 1282, ds-v3 5130)
#
# 128 and 256 must differ by 1.56x. At the retracted 0.10 all four crossed and the
# spread was 1.10x. So the sweep either separates them by about 1.56x or the
# refit is wrong, and there is no reading of the data where both hold.
#
# AND THAT IS ONE REGRESSION AGAINST A BYTE MODEL WITH NO TILE TERM IN IT.
# alpha_refit fits alpha from implied_traffic_ratio = time x bandwidth /
# COMPULSORY BYTES, and C4 is a confirmed finding that the compulsory-byte ruler
# was itself wrong by 1.85% until it was fixed. A number a paper's headline rests
# on should not rest on one estimator over one derived column, so this session
# measures alpha a SECOND time by a route that touches neither: step 2b aliases
# the weight pointer so every re-read hits L2, and reads alpha out of the
# difference of two timings in the same session on the same clock. The two
# estimates are printed TOGETHER at the end, with their intervals, and the
# session says plainly whether they agree. Two independent measurements of the
# study's central parameter disagreeing is the most important thing this session
# could produce and it is the last thing on the screen, not a line in a log.
#
# WHY nsys IS PROBED IN PRE-FLIGHT AND NOT LATER. Every byte figure in this study
# is modelled, never counted, because ncu is walled off on a rented pod by
# ERR_NVGPUCTRPERM. nsys samples DRAM through a different mechanism and MIGHT
# work here. If it does, steps 2, 3 and 4 and the dtype headline stop being
# inferences from a byte model and become measurements, so it is a force
# multiplier on everything after it and worth two minutes BEFORE anything is
# spent. It is never fatal: on a pod where the sampler is blocked the session
# runs exactly as it did before and every downstream traffic figure is labelled
# INFERRED instead of MEASURED, in the output and in an exfiled file, not just in
# a log line.
#
# TIMELINE. Offsets are from the moment the pod is up, and the mixtral download
# is started at 0:00 in the background precisely so 93 GB is never on the
# critical path. Budget re-derived 2026-09-01 when 2b and the nsys probe landed.
#
#   0:00  mixtral weights start downloading, backgrounded          step 0
#   0:00  PRE-FLIGHT, now including the nsys DRAM probe    5 min
#   0:05  fp8 SAME-SESSION calibration                     3 min   step 1
#   0:08  BLOCK_M sweep, multi-tile                        4-6     step 2
#   0:14  nsys traffic measurement of the profiled cell    2 min   step 2 (if nsys)
#   0:16  ALIAS ABLATION, the independent alpha            6-10    step 2b
#   0:26  GROUP_SIZE_M sweep                               12-15   step 3
#   0:41  tuned vs forced-fallback                         <= 35   step 4
#   1:16  config/ISA provenance and PTX, both cards        15-30   step 5
#   1:46  dense uniform grid, tile pinned                  54 min  step 6
#   2:40  trace capture                                    25-40   step 7
#   3:20  exfil, and NOTHING is torn down before it passes 10 min  step 8
#   3:30  done, worst case; 2:51 best case
#
# THE BUDGET STILL FITS, with more slack than before: the old plan quoted 4:20 to
# reach exfil and this one reaches it at 3:20 even on the pessimistic arm of every
# range. The 93 GB download needs about 90 minutes and step 7 does not start until
# 2:40, so it is still nowhere near the critical path. Step 2b is dominated by
# allocation and Triton compilation rather than kernel time, and its deepseek-v3
# rung needs about 16 GiB free on the card, which is why it runs before the
# long-lived sweeps rather than beside them.
#
# STEP 2b RUNS IMMEDIATELY AFTER STEP 2 ON PURPOSE. Both estimate alpha. Putting
# them in one session on one thermal state means a disagreement between them
# cannot be blamed on the card being a different card, which is the first thing a
# reviewer would reach for.
#
# EVERY STEP IS RE-RUNNABLE. A step that has already passed is skipped on a
# second invocation unless --force, the ledger at $SESSION/LEDGER.tsv is the
# record, and the sweeps resume through the harness's own --run-id manifest, so
# aborting costs at most one cell. Ctrl-C is safe at any point except mid-publish.
#
# EXIT CODES. 0 every gate passed or was deliberately skipped; 1 a soft gate
# failed and the session continued with a named consequence; 2 a fatal gate
# failed and the session stopped; 3 the script itself was used wrongly.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 3

# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------
DRY_RUN=0
PREFLIGHT_ONLY=0
FROM_STEP=0
ONLY_STEP=""
FORCE=0
ALLOW_DIRTY=0
NO_DOWNLOAD=0
NO_NSYS=0
LABEL=""
EXPECT_GPU="${MOE_EXPECT_GPU:-NVIDIA H200}"
SESSION_DIR=""
SKIP_TESTS=0

#: The header comment IS the help text, so it is extracted rather than restated.
#: Bounded by the first non-comment line rather than a hardcoded line number: the
#: previous version stopped at line 70 and silently truncated the moment the
#: header grew, which is how --help came to omit the timeline it exists to show.
usage() {
  sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '/^[^#]/d' | sed 's/^# \{0,1\}//'
  echo
  echo "flags: --dry-run --preflight-only --from N --only N --force --allow-dirty"
  echo "       --no-download --no-nsys --skip-tests --label NAME --expect-gpu NAME"
  echo "       --session-dir DIR"
  echo "steps: 0 1 2 2b 3 4 5 6 7 8   (--only takes any one of these; --from a number)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)        DRY_RUN=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --from)           FROM_STEP="${2:?--from needs a step number}"; shift 2 ;;
    --only)           ONLY_STEP="${2:?--only needs a step number}"; shift 2 ;;
    --force)          FORCE=1; shift ;;
    --allow-dirty)    ALLOW_DIRTY=1; shift ;;
    --no-download)    NO_DOWNLOAD=1; shift ;;
    --no-nsys)        NO_NSYS=1; shift ;;
    --skip-tests)     SKIP_TESTS=1; shift ;;
    --label)          LABEL="${2:?--label needs a name}"; shift 2 ;;
    --expect-gpu)     EXPECT_GPU="${2:?--expect-gpu needs a name}"; shift 2 ;;
    --session-dir)    SESSION_DIR="${2:?--session-dir needs a path}"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 3 ;;
  esac
done

# THE STEP LIST IS NO LONGER ALL DIGITS, so it is validated here rather than
# discovered by arithmetic. `wanted` compares with `-ge`, which EVALUATES its
# operands: a --from of `2b` or of anything else non-numeric is a bash arithmetic
# error at the first step, hundreds of lines from the typo that caused it, and
# under `set -uo pipefail` without -e it does not even stop. An unknown step is a
# usage error, which is exit 3, and it costs nothing to say so here.
STEP_IDS="0 1 2 2b 3 4 5 6 7 8"
if [[ -n "$ONLY_STEP" ]]; then
  case " $STEP_IDS " in
    *" $ONLY_STEP "*) ;;
    *) echo "--only: no step '$ONLY_STEP'. Steps are: $STEP_IDS" >&2; exit 3 ;;
  esac
fi
# --from is a NUMBER because it means "everything from here on", and 2b is not a
# position in that order; --from 2 includes it, which is what resuming after a
# crash in step 2 should mean.
if [[ ! "$FROM_STEP" =~ ^[0-9]+$ ]]; then
  echo "--from takes a number (2b is included by --from 2). Steps: $STEP_IDS" >&2
  exit 3
fi

# --------------------------------------------------------------------------
# paths, and the rule that everything lands somewhere that outlives the pod
# --------------------------------------------------------------------------
#
# A RunPod Network Volume at /workspace survives termination; the container
# filesystem does not. Writing a session's only copy of anything to the
# container is a guaranteed total loss at teardown, and preflight P14 refuses to
# start if $SESSION is not on a different mount from `/`. On a laptop there is
# no volume and the check degrades to a warning, because nothing is at risk.
WORKSPACE="${WORKSPACE:-/workspace}"
[[ -d "$WORKSPACE" ]] || WORKSPACE="$REPO_ROOT"
VENVS="${MOE_VENV_ROOT:-$WORKSPACE/venvs}"
RESULTS_DIR="${MOE_RESULTS_DIR:-$WORKSPACE/results}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
# Resume into the newest existing session rather than opening a new one, so
# --from 5 lands in the directory steps 1-4 wrote to and reads their ledger. An
# explicit --session-dir always wins, because a resume that silently retargets
# the directory the human named is worse than no resume at all.
if [[ -z "$SESSION_DIR" ]]; then
  SESSION_DIR="$WORKSPACE/session/$STAMP"
  if [[ "$FROM_STEP" -gt 0 || -n "$ONLY_STEP" ]]; then
    newest="$(ls -1d "$WORKSPACE"/session/*/ 2>/dev/null | tail -1)"
    [[ -n "$newest" ]] && SESSION_DIR="${newest%/}"
  fi
fi
SESSION="$SESSION_DIR"
mkdir -p "$SESSION/logs" "$SESSION/ptx" "$SESSION/calibration" "$SESSION/exfil" || exit 3

LEDGER="$SESSION/LEDGER.tsv"
[[ -f "$LEDGER" ]] || printf 'step\tname\tstatus\tobserved\tgate\tconsequence\n' > "$LEDGER"

[[ -n "$LABEL" ]] || LABEL="session-$STAMP"

PY_BASE="${MOE_PYTHON:-$VENVS/base/bin/python}"
[[ -x "$PY_BASE" ]] || PY_BASE="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY_BASE" ]] || PY_BASE="$(command -v python3 || true)"
PY_VLLM="$VENVS/vllm/bin/python"
[[ -x "$PY_VLLM" ]] || PY_VLLM="$PY_BASE"
PY_SGL="$VENVS/sglang/bin/python"
[[ -x "$PY_SGL" ]] || PY_SGL="$PY_BASE"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKSPACE/torchinductor-cache}"
# The SHARED Triton cache is correct for sweeps (it saves minutes of metered
# recompilation) and CATASTROPHIC for a PTX dump, because Triton does not
# recompile a kernel it has already built and a cache hit dumps no PTX at all.
# Every dump below overrides this with a per-run cache; it is exported here so
# the sweeps get the benefit and so the override is visibly an override.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE/triton-cache}"

# --------------------------------------------------------------------------
# output helpers
# --------------------------------------------------------------------------
BOLD=""; DIM=""; OFF=""
if [[ -t 1 ]]; then BOLD=$'\033[1m'; DIM=$'\033[2m'; OFF=$'\033[0m'; fi

N_PASS=0; N_FAIL_SOFT=0; N_FAIL_FATAL=0; N_SKIP=0
STOPPED=""
#: Set the moment the first metered step starts. Until then a stop costs
#: nothing and the summary must say so rather than telling a tired human not
#: to tear down a pod that has produced nothing.
SPENT=0

say()   { printf '%s\n' "$*"; }
head1() { printf '\n%s==== %s ====%s\n' "$BOLD" "$*" "$OFF"; }
head2() { printf '\n%s-- %s%s\n' "$BOLD" "$*" "$OFF"; }
note()  { printf '%s    %s%s\n' "$DIM" "$*" "$OFF"; }

#: One row of the ledger. Written on every verdict so a killed session still
#: says exactly which gates had passed, which is what --from reads.
ledger() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "${6:-}" >> "$LEDGER"
}

#: PASS/FAIL against a NUMBER, which is the whole contract of this script.
#   verdict <id> <name> <ok:0|1> <observed> <gate> <policy:fatal|soft> <consequence>
# `ok` is an exit-code convention: 0 is true, so `[[ ... ]]; verdict ... $? ...`
# reads correctly.
verdict() {
  local id="$1" name="$2" ok="$3" observed="$4" gate="$5" policy="$6" consequence="${7:-}"
  if [[ "$ok" == "0" ]]; then
    printf '  %sPASS%s  %-34s %s  (gate: %s)\n' "$BOLD" "$OFF" "$name" "$observed" "$gate"
    ledger "$id" "$name" PASS "$observed" "$gate" ""
    N_PASS=$((N_PASS + 1))
    return 0
  fi
  printf '  %sFAIL%s  %-34s %s  (gate: %s)\n' "$BOLD" "$OFF" "$name" "$observed" "$gate"
  [[ -n "$consequence" ]] && printf '        %s\n' "$consequence"
  ledger "$id" "$name" FAIL "$observed" "$gate" "$consequence"
  if [[ "$policy" == "fatal" ]]; then
    printf '        %sSTOP.%s Later steps read this, so continuing produces numbers that cannot be published.\n' "$BOLD" "$OFF"
    N_FAIL_FATAL=$((N_FAIL_FATAL + 1))
    # The FIRST fatal is the one to fix, and the later ones are often its
    # consequence. Preflight deliberately runs to the end anyway, so one
    # three-minute pass shows every problem rather than one per re-rent.
    [[ -n "$STOPPED" ]] || STOPPED="$name"
    return 2
  fi
  printf '        CONTINUE. This invalidates the named result only; the rest of the session stands.\n'
  N_FAIL_SOFT=$((N_FAIL_SOFT + 1))
  return 1
}

skipped() {
  local id="$1" name="$2" why="$3"
  printf '  %sSKIP%s  %-34s %s\n' "$DIM" "$OFF" "$name" "$why"
  ledger "$id" "$name" SKIP "$why" "" ""
  N_SKIP=$((N_SKIP + 1))
}

#: Did any ledger row whose id starts with this prefix pass? Used by the exfil
#: manifest, which must expect an artefact only from a step that claimed to make
#: one. A prefix rather than an exact id because step 5 writes one row per cell.
#: The continuation must be a NON-DIGIT or absent, so `S2` still matches `S2a` and
#: `S2b` as intended while `P1` no longer swallows `P11a`. Found by shellcheck
#: SC1087 pointing at the unbraced expansion, not by the collision itself.
ledger_passed() {
  local tab
  tab="$(printf '\t')"
  grep -qE "^$1([^0-9$tab][^$tab]*)?$tab[^$tab]*${tab}PASS$tab" "$LEDGER" 2>/dev/null
}

#: Did this session measure anything at all? Guards the publish, so a rehearsal
#: on a laptop can run every step for real without committing to the repo.
measured_anything() {
  ledger_passed S1a || ledger_passed S5tar || ledger_passed S6b || ledger_passed "S7-"
}

#: Has this STEP already finished cleanly in this session directory?
#: Idempotence lives here: a re-run repeats nothing that already has at least one
#: PASS and no FAIL under its id prefix, so --from is a convenience and not the
#: safety mechanism. A step that failed anything re-runs, which is what resuming
#: after a fix should mean; --force re-runs everything.
already_passed() {
  [[ "$FORCE" == "1" ]] && return 1
  local tab p f
  tab="$(printf '\t')"
  p="$(grep -cE "^$1([^0-9$tab][^$tab]*)?$tab[^$tab]*${tab}PASS$tab" "$LEDGER" 2>/dev/null)"
  f="$(grep -cE "^$1([^0-9$tab][^$tab]*)?$tab[^$tab]*${tab}FAIL$tab" "$LEDGER" 2>/dev/null)"
  [[ "${p:-0}" -ge 1 && "${f:-0}" == "0" ]]
}

#: Where a step sits in the running order, which is not always its name. 2b is
#: the alias ablation and it runs immediately after step 2, in the same session
#: and the same thermal state, because both measure alpha and a difference
#: between them must not be attributable to the card having cooled. Its ORDER is
#: therefore 2: --from 2 resumes into it, --only 2b runs it alone.
step_order() {
  case "$1" in
    2b) echo 2 ;;
    *)  echo "$1" ;;
  esac
}

#: Should this step run at all, given --from / --only? --only matches the step ID
#: exactly, so `--only 2` does NOT drag 2b along with it; --from matches the
#: ORDER, so `--from 2` does.
wanted() {
  local n="$1"
  if [[ -n "$ONLY_STEP" ]]; then [[ "$n" == "$ONLY_STEP" ]]; return $?; fi
  [[ "$(step_order "$n")" -ge "$FROM_STEP" ]]
}

#: A step that a fatal preflight failure has taken out.
halted() { [[ -n "$STOPPED" ]]; }

# --------------------------------------------------------------------------
# capability probes, so the same script runs on a laptop and on the pod
# --------------------------------------------------------------------------
HAVE_GPU=0
HAVE_VLLM=0
HAVE_SGLANG=0
GPU_NAME=""

probe_capabilities() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    [[ -n "$GPU_NAME" ]] && HAVE_GPU=1
  fi
  "$PY_VLLM" -c "import vllm" >/dev/null 2>&1 && HAVE_VLLM=1
  "$PY_SGL"  -c "import sglang" >/dev/null 2>&1 && HAVE_SGLANG=1
}

#: The message a degraded step prints. One phrasing everywhere, so "this is a
#: laptop" never gets confused with "the pod is broken".
absent_reason() {
  local need="$1"
  case "$need" in
    gpu)    echo "no CUDA device visible (nvidia-smi absent or empty)" ;;
    vllm)   echo "vLLM does not import in $PY_VLLM" ;;
    sglang) echo "SGLang does not import in $PY_SGL" ;;
  esac
}

# --------------------------------------------------------------------------
# traffic provenance: is a byte figure COUNTED or MODELLED
# --------------------------------------------------------------------------
#
# EVERY BYTE FIGURE IN THIS STUDY IS COMPULSORY-TRAFFIC ARITHMETIC. Nothing has
# ever counted a DRAM transaction, because ncu is walled off on a rented pod by
# ERR_NVGPUCTRPERM, and docs/FINDINGS.md says so under "what this does not
# settle". Pre-flight P-nsys asks whether nsys can sample DRAM here instead. If it
# can, the traffic figures behind steps 1 through 4 stop being inferences.
#
# THE DISTINCTION HAS TO REACH THE READER, not a log. A number that was modelled
# and a number that was counted are different kinds of evidence and they look
# identical in a table, which is exactly the shape of failure this whole script
# exists to prevent. So the word is printed at the head of every step that
# produces a traffic figure, written to an exfiled file that the manifest checks,
# and repeated in the summary.
NSYS_BIN=""
NSYS_SAMPLES_DRAM=0
NSYS_ROUTE=""
NSYS_WHY="not probed"
TRAFFIC_PROVENANCE="INFERRED"
PROVENANCE_FILE=""

#: One sentence, the same one everywhere, so a step can never imply more than the
#: probe established.
provenance_line() {
  if [[ "$NSYS_SAMPLES_DRAM" == "1" ]]; then
    echo "TRAFFIC: MEASURED. nsys sampled DRAM here via $NSYS_ROUTE, and the cell measured under step 2 scores the compulsory byte model against counted bytes. Device-wide and edge-quantised: quote it with its resolution line."
  else
    echo "TRAFFIC: INFERRED. Every byte figure below is compulsory-traffic arithmetic, never counted. Reason: $NSYS_WHY"
  fi
}

#: Printed by each step that produces a traffic figure.
provenance_banner() {
  printf '  %s%s%s\n' "$BOLD" "$(provenance_line)" "$OFF"
}

#: The file the manifest checks, written the moment the probe finishes so it
#: exists even if every later step is skipped.
write_provenance() {
  PROVENANCE_FILE="$SESSION/exfil/TRAFFIC_PROVENANCE.txt"
  {
    echo "# is a byte figure in this session counted or modelled"
    echo "# written by scripts/pod_session.sh pre-flight P-nsys on $(date -u +%FT%TZ)"
    echo "# card: ${GPU_NAME:-none}"
    echo
    echo "$TRAFFIC_PROVENANCE"
    echo
    provenance_line
    echo
    echo "route      : ${NSYS_ROUTE:-none}"
    echo "reason     : $NSYS_WHY"
    echo "probe      : $SESSION/nsys"
    echo
    echo "Steps 1, 2, 2b, 3 and 4 all carry this label. Step 2b is the ONE"
    echo "measurement in this session that is independent of the byte model"
    echo "whatever this file says, because it never divides by bytes at all."
  } > "$PROVENANCE_FILE"
}

# --------------------------------------------------------------------------
#: A hard wall-clock cap, so no pre-flight check can hang the session.
#:
#: The nsys probe walks a six-rung ladder and each rung can sit at its own
#: --timeout, so its worst case is a multiple of a number this script passes it.
#: That is bounded but not SHORT, and pre-flight has a two-minute contract. On a
#: box without coreutils timeout (a mac laptop) the command runs uncapped, which
#: is correct there: nothing is metered and nothing is at risk.
with_deadline() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout -k 5 "$secs" "$@"
  else
    "$@"
  fi
}

# --------------------------------------------------------------------------
#: P-nsys. CAN DRAM TRAFFIC BE COUNTED ON THIS POD RATHER THAN MODELLED?
#:
#: WHY IT IS IN PRE-FLIGHT AND NOT A LATE STEP. It is a FORCE MULTIPLIER, not a
#: result. If nsys samples DRAM here then steps 2, 3 and 4 and the dtype headline
#: all become measurements instead of inferences from a byte model, and knowing
#: that BEFORE the sweeps run is what lets the sweeps say so. Knowing it at 3:00
#: would be worth nothing. Two minutes, before anything is spent.
#:
#: IT IS NEVER FATAL AND NEVER EVEN A SOFT FAIL FOR BEING ABSENT. nsys having no
#: DRAM sampler on a rented pod is the EXPECTED case: ncu is already blocked by
#: ERR_NVGPUCTRPERM and this route has never been tested. A gate that cries FAIL
#: on the expected case is a gate nobody reads, and this one has to be read. So
#: absence is an INFO row and the consequence is carried as a label rather than a
#: verdict: every downstream traffic figure says INFERRED.
#:
#: THERE IS EXACTLY ONE FAILURE HERE WORTH A VERDICT, and it is the opposite one.
#: The probe runs a workload whose DRAM traffic is KNOWN without any model and
#: checks the sampler against it. A sampler that answers, and answers WRONG on a
#: case with a known answer, is an instrument that would have silently corrupted
#: every step after it. That is a soft FAIL with a named consequence, and it is
#: the reason --calibrate is passed here rather than left for later.
preflight_nsys() {
  head2 "P-nsys  can DRAM traffic be COUNTED here, or does it stay modelled"
  local plog="$SESSION/logs/nsys_probe.log" pdir="$SESSION/nsys"
  mkdir -p "$pdir"

  if [[ "$NO_NSYS" == "1" ]]; then
    NSYS_WHY="--no-nsys was passed, so the sampler was never asked"
    ledger Pnsys "nsys DRAM sampling" INFO "$NSYS_WHY" "informational" ""
    note "$NSYS_WHY"
    return 0
  fi
  if [[ ! -f scripts/nsys_dram_probe.py ]]; then
    NSYS_WHY="scripts/nsys_dram_probe.py is not on this branch"
    ledger Pnsys "nsys DRAM sampling" INFO "$NSYS_WHY" "informational" ""
    note "$NSYS_WHY"
    return 0
  fi
  if [[ "$HAVE_GPU" == "0" ]]; then
    NSYS_WHY="$(absent_reason gpu)"
    ledger Pnsys "nsys DRAM sampling" INFO "$NSYS_WHY" "informational" ""
    note "$NSYS_WHY"
    return 0
  fi
  if [[ -z "$NSYS_BIN" ]]; then
    NSYS_WHY="nsys is not installed on this pod (P10 says so above)"
    ledger Pnsys "nsys DRAM sampling" INFO "$NSYS_WHY" "informational" ""
    note "$NSYS_WHY"
    return 0
  fi

  # --timeout is PER ATTEMPT and the ladder has six of them, so 12 s caps the
  # ladder near 72 s; --calibrate adds one run at twice that. with_deadline is
  # the outer belt, because nsys export has its own timeout this script does not
  # set.
  say "  \$ $PY_BASE scripts/nsys_dram_probe.py --calibrate --timeout 12 --out $pdir"
  with_deadline 150 "$PY_BASE" scripts/nsys_dram_probe.py --calibrate \
    --timeout 12 --seconds 1.5 --out "$pdir" > "$plog" 2>&1
  local rc=$?

  # THE EXIT CODE IS NOT THE ANSWER, the report is: the probe exits 3 both when
  # no invocation sampled anything and when one did and then failed calibration,
  # and those are opposite findings. probe.json distinguishes them by name.
  local parsed
  parsed="$("$PY_BASE" - "$pdir/probe.json" <<'PYEOF' 2>/dev/null
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print("none|no probe.json was written|")
    raise SystemExit(0)
try:
    doc = json.loads(p.read_text())
except Exception as exc:
    print(f"none|probe.json is unreadable: {type(exc).__name__}|")
    raise SystemExit(0)
winner = ""
for att in doc.get("attempts") or []:
    if att.get("metrics_rows") and att.get("dram_metrics"):
        winner = att.get("label") or "an unnamed invocation"
cal = (doc.get("calibration") or {}).get("calibration_verdict") or ""
if winner:
    state = "lying" if cal.startswith("FAIL") else "ok"
else:
    state = "none"
why = (doc.get("verdict") or "the probe wrote no verdict").replace("\n", " ")
# FREE TEXT LAST. bash `read` with three names puts every surplus field into the
# THIRD one, so a pipe character anywhere in the verdict prose would silently
# shift the route into it if the order were the other way round.
print("|".join([state, winner, why[:400]]))
PYEOF
)"
  local state why route
  IFS='|' read -r state route why <<< "$parsed"

  case "${state:-none}" in
    ok)
      NSYS_SAMPLES_DRAM=1
      NSYS_ROUTE="$route"
      NSYS_WHY="nsys samples DRAM here"
      TRAFFIC_PROVENANCE="MEASURED"
      ledger Pnsys "nsys DRAM sampling" INFO "MEASURED via $route" "informational" ""
      note "MEASURED. $route"
      note "steps 2, 3 and 4 and the dtype headline are now measurements rather than inferences"
      ;;
    lying)
      # The sampler answered and the known-traffic case says its answer is wrong.
      # This is the one outcome here that is worth a verdict, because it is the
      # one that would otherwise have corrupted everything downstream silently.
      NSYS_WHY="nsys sampled DRAM and then FAILED a workload with a known answer, so its numbers are not usable"
      false; verdict PnsysCal "nsys sampler tells the truth" $? \
        "sampler present, calibration FAILED" "a known-traffic case within tolerance" soft \
        "The sampler works and disagrees with a case whose answer needs no model. Nothing measured through it may be quoted; the session continues with traffic INFERRED exactly as before. See $plog and $pdir/probe.json."
      ;;
    *)
      NSYS_WHY="${why:-no invocation of this nsys sampled a DRAM metric}"
      ledger Pnsys "nsys DRAM sampling" INFO "INFERRED: $NSYS_WHY" "informational" ""
      note "INFERRED. $NSYS_WHY"
      note "this is the EXPECTED outcome on a rented pod and costs the session nothing; the control rung in $plog says whether nsys itself works here at all"
      ;;
  esac
  [[ "$rc" == "0" || "$rc" == "3" ]] || note "probe exited $rc, which is neither 0 nor its no-sampler 3; see $plog"
  note "probe log $plog"
}

# ==========================================================================
# PRE-FLIGHT
# ==========================================================================
#
# Every check below costs under two minutes and each one kills a whole CLASS of
# failure that this project has actually suffered. They run before anything is
# spent. Ordered cheapest-first so a fatal one stops the session in seconds, and
# P-nsys goes LAST because it is the only one that costs minutes rather than
# seconds and there is no sense sampling DRAM on a pod whose tests are red.
preflight() {
  head1 "PRE-FLIGHT  (about 5 minutes, kills a class of failure per check)"
  say "  session dir : $SESSION"
  say "  repo        : $REPO_ROOT"
  say "  base python : $PY_BASE"
  say "  gpu         : ${GPU_NAME:-none}   vllm=$HAVE_VLLM sglang=$HAVE_SGLANG"

  head2 "P1  repo identity and a clean working tree"
  local sha dirty
  sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  dirty="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  note "HEAD $sha on $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  # 76.7% of the canonical published pool carries git_dirty=True and two whole
  # arms are 100% dirty, so those rows are not reproducible from the commit they
  # name. This is the one gate a human may legitimately override, which is why
  # --allow-dirty exists and why using it is recorded in the ledger.
  if [[ "$ALLOW_DIRTY" == "1" && "$dirty" != "0" ]]; then
    skipped P1 "clean working tree" "$dirty dirty file(s), waived by --allow-dirty"
  else
    [[ "$dirty" == "0" ]]; verdict P1 "clean working tree" $? \
      "$dirty dirty file(s)" "== 0" soft \
      "Every row this session writes carries git_dirty=True and is not reproducible from $sha. Commit or stash, or re-run with --allow-dirty to accept it."
  fi

  head2 "P2  is this the GPU the calibration and the ridge belong to"
  # moe/bench/hardware/measured_<device>.yaml matches by device NAME, so a
  # SECOND H200 pod silently inherits the FIRST one's ceilings, and an A100 pod
  # would be scored against an H200 roof. Name, count, memory and driver are all
  # cheap and all decisive.
  if [[ "$HAVE_GPU" == "0" ]]; then
    skipped P2 "expected GPU" "$(absent_reason gpu)"
  else
    local count mem driver major
    count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
    mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
    major="${driver%%.*}"
    note "$count x $GPU_NAME, ${mem} MiB, driver $driver"
    [[ "$GPU_NAME" == *"$EXPECT_GPU"* ]]; verdict P2a "GPU name" $? \
      "$GPU_NAME" "contains '$EXPECT_GPU'" fatal \
      "measured_<device>.yaml resolves by NAME. On the wrong card every efficiency column is quoted against another part's roof and nothing downstream can tell. Set --expect-gpu if this card is deliberate."
    [[ "${mem:-0}" -ge 80000 ]]; verdict P2b "device memory" $? \
      "${mem} MiB" ">= 80000 MiB" fatal \
      "deepseek-v3 at E=256,N=2048 needs tens of GB for one layer; a smaller card changes which cells are even runnable."
    [[ "${major:-0}" -ge 580 ]]; verdict P2c "driver major" $? \
      "$driver" ">= 580" soft \
      "A cu130 torch wheel needs r580+. On an older driver install the cu128 index instead; the index must match the driver, not the image name."
  fi

  head2 "P3  torch and Triton are the pair the published rows were measured on"
  # A different torch ships a different CUTLASS, so the grouped GEMM profiled
  # would not be the one the published rows describe; a different Triton emits
  # different PTX, so step 5's ISA answer would be about another compiler.
  # scripts/profile_open_questions.sh already refuses on this mismatch.
  local versions
  versions="$("$PY_BASE" - <<'PYEOF' 2>/dev/null
try:
    import torch
    t = torch.__version__
except Exception:
    t = "absent"
try:
    import triton
    r = triton.__version__
except Exception:
    r = "absent"
print(f"{t} {r}")
PYEOF
)"
  local tv rv
  tv="${versions%% *}"; rv="${versions##* }"
  note "torch $tv, triton $rv (published rows: torch 2.13.0, triton 3.7.1)"
  [[ "$tv" == 2.13.0* ]]; verdict P3a "torch is 2.13.0" $? \
    "$tv" "starts with 2.13.0" soft \
    "Rows measured here are not directly comparable with the ten published arms, and step 5's PTX describes a different CUTLASS. Pin the base venv per docs/RUNPOD.md before publishing anything as a continuation of those arms."
  [[ "$rv" == 3.7.1* ]]; verdict P3b "triton is 3.7.1" $? \
    "$rv" "starts with 3.7.1" soft \
    "Step 5's wgmma-versus-mma.sync verdict would be about a different compiler's instruction selection, so it cannot be quoted beside the existing C3 result."

  head2 "P4  override_config binds for a shape vLLM has never seen"
  # Steps 2, 3 and 4 are all override_config experiments. If the hook does not
  # bind, they sweep NOTHING while printing a full table, which is precisely the
  # silent-plausible-number failure this project keeps hitting. deepseek-v3 at
  # E=256,N=2048 is the sharpest test available: vLLM v0.27.1 ships no tuned
  # file for it on any card or dtype, so the override is the ONLY thing that can
  # choose the tile and a bind failure cannot hide behind a file that agrees.
  if [[ "$HAVE_VLLM" == "0" ]]; then
    skipped P4 "override_config binds" "$(absent_reason vllm)"
  else
    local bind
    bind="$("$PY_VLLM" - <<'PYEOF' 2>&1
import importlib, sys
sys.path.insert(0, ".")
CANDIDATES = ("vllm.model_executor.layers.fused_moe",
              "vllm.model_executor.layers.fused_moe.fused_moe",
              "vllm.model_executor.layers.fused_moe.config")
override = getcfg = None
for name in CANDIDATES:
    try:
        mod = importlib.import_module(name)
    except ImportError:
        continue
    override = override or getattr(mod, "override_config", None)
    getcfg = getcfg or getattr(mod, "get_config", None)
if override is None or getcfg is None:
    print("hook-missing"); raise SystemExit(0)
want = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4}
with override(want):
    got = getcfg()
after = getcfg()
from moe.bench.tile_resolve import config_file_name, ships
from moe.spec import MODEL_CONFIGS
cfg = MODEL_CONFIGS["deepseek-v3"]
fname = config_file_name(cfg.num_experts, cfg.intermediate_size, "bf16", "NVIDIA H200")
print("bound" if got == want else f"bound-wrong:{got}",
      "released" if not after else f"leaked:{after}",
      "unshipped" if not ships(fname) else f"shipped:{fname}")
PYEOF
)"
    note "$bind"
    [[ "$bind" == bound\ released\ unshipped* ]]; verdict P4 "override_config binds and releases" $? \
      "$bind" "== 'bound released unshipped'" fatal \
      "Steps 2, 3 and 4 force a tile through this hook. If it does not bind they measure vLLM's own choice while labelling it a forced tile, and the BLOCK_M separation those steps exist to find would be manufactured. Check the installed vLLM version; try_get_optimal_moe_config reads the hook via get_config(), so it exists under some name."
  fi

  head2 "P5  an isolated TRITON_CACHE_DIR really does produce PTX"
  # THE FAILURE THIS PREVENTS, verbatim from the repo's history: with the shared
  # $WORKSPACE/triton-cache inherited, every fused_moe specialisation is already
  # built, nothing recompiles, no .ptx is written, and check_mma_path.sh exits
  # reporting that the kernel never compiled. That is very likely why the A100
  # was never successfully dumped. Proving the isolation works costs one trivial
  # kernel compile.
  if [[ "$HAVE_GPU" == "0" || "$HAVE_VLLM" == "0" && "$HAVE_SGLANG" == "0" ]]; then
    if [[ "$HAVE_GPU" == "0" ]]; then
      skipped P5 "PTX dump isolation" "$(absent_reason gpu)"
    fi
  fi
  if [[ "$HAVE_GPU" == "1" ]]; then
    local probe="$SESSION/ptx/_preflight"
    rm -rf "$probe"; mkdir -p "$probe"
    local nptx perr
    perr="$probe/_probe.stderr"
    # THE PROBE MUST BE A REAL FILE. Triton's @jit calls inspect.getsourcelines
    # on the decorated function, which raises "could not get source code" when
    # the script arrived on stdin -- so a heredoc piped to `python -` can NEVER
    # compile a kernel, and this gate reported "dumped 0 .ptx" on every machine
    # it had ever run on. With stderr suppressed, that ValueError was
    # indistinguishable from a dump environment that genuinely produced nothing,
    # which is the exact shape of failure this script exists to catch. stderr is
    # now kept and a raise is reported as -2 rather than collapsing into 0.
    cat > "$probe/_probe.py" <<'PYEOF'
import sys
from pathlib import Path
out = Path(sys.argv[1])
try:
    import torch, triton, triton.language as tl
except Exception:
    print(-1); raise SystemExit(0)


@triton.jit
def _k(p, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(p + i, tl.load(p + i, mask=i < n) * 2.0, mask=i < n)


try:
    x = torch.ones(1024, device="cuda")
    _k[(1,)](x, 1024, BLOCK=1024)
    torch.cuda.synchronize()
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    print(-2); raise SystemExit(0)
cache = out / "_cache"
print(sum(1 for q in out.rglob("*.ptx") if cache not in q.parents))
PYEOF
    nptx="$(TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$probe" TRITON_CACHE_DIR="$probe/_cache" \
      "$PY_BASE" "$probe/_probe.py" "$probe" 2>"$perr")"
    if [[ "${nptx:-0}" == "-2" ]]; then
      note "the probe kernel RAISED: $(head -1 "$perr")"
    fi
    note "dumped ${nptx:-0} .ptx file(s) outside the per-run cache into $probe"
    [[ "${nptx:-0}" -ge 1 ]]; verdict P5 "PTX dump isolation" $? \
      "${nptx:-0} .ptx" ">= 1" fatal \
      "Step 5 cannot answer the ISA question without a dump. If this is 0 the dump env is not reaching the compiler; -1 means triton did not import; -2 means the probe kernel itself raised and the reason is on the line above. Note that TRITON_CACHE_DIR is exported to the SHARED cache for the sweeps, and every dump must override it with a per-run directory."
    rm -rf "$probe"
  fi

  head2 "P6  disk, sized against the 93 GB that starts downloading at 0:00"
  local vol_free ctr_free
  vol_free="$(df -Pk "$WORKSPACE" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}')"
  ctr_free="$(df -Pk / 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}')"
  note "volume ${vol_free:-?} GiB free at $WORKSPACE, container ${ctr_free:-?} GiB free at /"
  # mixtral-8x7b is 93.4 GB; 110 leaves room for the shards plus the HF blob
  # cache holding a second copy mid-verification.
  [[ "${vol_free:-0}" -ge 110 ]]; verdict P6a "volume free space" $? \
    "${vol_free:-0} GiB" ">= 110 GiB" soft \
    "Step 7 (traces) cannot pull mixtral. Grow the Network Volume, or run this session with --no-download and capture deepseek-v2-lite (31 GB) instead. Everything else fits in 100 GB."
  [[ "${ctr_free:-0}" -ge 10 ]]; verdict P6b "container disk free" $? \
    "${ctr_free:-0} GiB" ">= 10 GiB" soft \
    "Wheel extraction for vLLM and SGLang needs several GB of temp space; running out mid-install is a slow failure on a metered box."

  head2 "P7  the calibration provenance machinery still refuses what it should"
  # entitled_ridge is the guard that stops an arm being quoted against a ruler
  # measured in another session (defect 7, which cost claim C5 its target for
  # three days). Two of the ten published arms must be refused. A change that
  # silently stops refusing is invisible in any table, so it is checked here
  # against a count.
  local refusals
  refusals="$("$PY_BASE" - <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, ".")
from pathlib import Path
from moe.bench.published import entitled_ridge
arms = sorted(p for p in Path("results/published").iterdir() if p.is_dir())
print(sum(1 for a in arms if entitled_ridge(a)[0] is None), len(arms))
PYEOF
)"
  note "entitled_ridge refuses ${refusals:-?} (expected: 2 of 10, the fp8-three-kernel and whole-layer arms)"
  [[ "$refusals" == "2 10" ]]; verdict P7 "provenance gate refuses 2 of 10" $? \
    "${refusals:-none}" "== '2 10'" soft \
    "The guard that would catch a borrowed calibration has changed behaviour. Publishing from this session is still safe, but re-read moe/bench/published.py before quoting any absolute efficiency number."

  head2 "P8  the weights step 7 pulls are actually reachable"
  # THIS USED TO CHECK FOR A TOKEN, WHICH IS NOT THE REQUIREMENT. It ran
  # HfApi().whoami() and failed without credentials, advising "step 7 will fail
  # on the gated mixtral repo". Mistral UNGATED Mixtral-8x7B-Instruct-v0.1: it
  # is apache-2.0, model_info reports gated=False, and config.json downloads
  # anonymously. So the gate demanded a credential nothing needed and justified
  # it with a fact that had stopped being true, which is worse than not checking
  # -- a reader trusts the stated reason. It also never checked the thing that
  # DOES fail: a token present but a licence unaccepted passes a whoami and
  # fails at 2:40, after every other step has been paid for.
  #
  # So it asks the question step 7 asks: are THESE repos reachable, by whatever
  # credentials this box has? The repo ids come from moe/spec.py rather than
  # being written here twice, and a token is reported as an advisory because it
  # buys download RATE on 93.4 GB, not access.
  if [[ "$NO_DOWNLOAD" == "1" ]]; then
    skipped P8 "weights reachable" "--no-download, step 7 will not pull weights"
  else
    local reach
    reach="$("$PY_BASE" - <<'PYEOF' 2>/dev/null
import sys
try:
    from huggingface_hub import HfApi
    from moe.spec import MODEL_CONFIGS
except Exception as exc:
    print(f"unknown|import failed: {type(exc).__name__}"); raise SystemExit(0)

api = HfApi()
try:
    who = api.whoami().get("name", "?")
except Exception:
    who = ""

bad = []
for key in ("mixtral-8x7b", "deepseek-v2-lite"):
    cfg = MODEL_CONFIGS.get(key)
    repo = getattr(cfg, "hf_repo", None) if cfg else None
    if not repo:
        bad.append(f"{key}: no hf_repo in spec")
        continue
    try:
        info = api.model_info(repo)
        if info.gated:
            bad.append(f"{repo}: gated={info.gated!r}")
    except Exception as exc:
        bad.append(f"{repo}: {type(exc).__name__}")
print(("ok" if not bad else "blocked") + "|"
      + (f"as {who}" if who else "anonymous")
      + ("" if not bad else "; " + "; ".join(bad)))
PYEOF
)"
    local rstate="${reach%%|*}" rdetail="${reach#*|}"
    note "reachability: ${rstate:-unknown} (${rdetail:-no detail})"
    if [[ "$rstate" == "ok" && "$rdetail" == anonymous* ]]; then
      note "no token: access is fine, but HF rate-limits anonymous transfers and step 0 pulls 93.4 GB"
    fi
    [[ "$rstate" == "ok" ]]; verdict P8 "weights reachable" $? \
      "${rstate:-unknown}" "every repo step 7 pulls resolves" soft \
      "Step 7 cannot fetch what it needs. If a repo reports gated, accept its licence while signed in and log in on this box; if it reports an HTTP error, the hub or the network is the problem, not the credentials. --no-download skips step 7 entirely."
  fi

  head2 "P9  .gitignore anchoring, tested by probing the exact exfil paths"
  # THE FAILURE THIS PREVENTS: an unanchored `plots/` line matched at ANY depth
  # and silently swallowed results/published/<arm>/plots/*.png on every publish.
  # publish_results.sh logged "included N figure(s)" and `git add` dropped every
  # one; zero .png files are tracked under results/published/ across all ten
  # arms. The rule is now `/plots/`, but `*.ptx`, `*.so`, `*.nsys-rep` and
  # `*.qdrep` still match at any depth ON PURPOSE, which is why step 5's dumps
  # leave as a tarball. This probes the real paths rather than trusting either.
  local probe_dir="results/published/_preflight_probe" ignored=0 ig_report=""
  mkdir -p "$probe_dir/plots"
  : > "$probe_dir/plots/probe.png"
  : > "$probe_dir/ISA_CENSUS.txt"
  : > "$probe_dir/ptx-h200.tar.gz"
  : > "$probe_dir/session.log"
  : > "$probe_dir/merged.csv"
  mkdir -p traces && : > traces/_probe.npz
  local f
  for f in "$probe_dir/plots/probe.png" "$probe_dir/ISA_CENSUS.txt" \
           "$probe_dir/ptx-h200.tar.gz" "$probe_dir/session.log" \
           "$probe_dir/merged.csv" traces/_probe.npz; do
    if git check-ignore -q "$f" 2>/dev/null; then
      ignored=$((ignored + 1))
      ig_report="$ig_report $(git check-ignore -v "$f" 2>/dev/null | tr '\t' ' ')"
    fi
  done
  rm -rf "$probe_dir" traces/_probe.npz
  [[ "$ignored" == "0" ]]; verdict P9 "exfil paths are committable" $? \
    "$ignored of 6 ignored${ig_report:+ --$ig_report}" "== 0" fatal \
    "Anything ignored here is lost at teardown even after a successful publish, and the loss is silent: git add reports nothing. Fix the rule or change the exfil filename before spending an hour producing the artefact."
  note "reminder: *.ptx *.so *.nsys-rep *.qdrep are ignored at any depth by design, so raw dumps exfil as .tar.gz"

  head2 "P10  which profiler is actually available"
  # ncu fails on a rented pod with ERR_NVGPUCTRPERM because GPU performance
  # counters need a host module flag a container tenant cannot set. nsys uses
  # CUDA tracing instead and often works. Recorded, not gated: the session is
  # designed around having neither.
  local have_nsys=no have_ncu=no
  command -v nsys >/dev/null 2>&1 && have_nsys=yes
  command -v ncu  >/dev/null 2>&1 && have_ncu=yes
  note "nsys=$have_nsys ncu=$have_ncu (ncu is expected to be present but to fail with ERR_NVGPUCTRPERM)"
  ledger P10 "profiler availability" INFO "nsys=$have_nsys ncu=$have_ncu" "informational" ""
  [[ "$have_nsys" == "yes" ]] && NSYS_BIN="$(command -v nsys)"
  note "P-nsys below asks the harder question: whether nsys can SAMPLE DRAM here"

  head2 "P11  the step scripts exist and parse"
  # Three other agents are writing these concurrently. A missing or unparseable
  # step should cost a line here, not forty minutes and a stack trace at 1:10.
  local missing=0 broken=0 s
  for s in scripts/block_m_crossing_sweep.py scripts/group_m_alpha_sweep.py \
           scripts/alias_ablation.py scripts/nsys_dram_probe.py \
           scripts/alpha_refit.py \
           scripts/tuned_vs_fallback.py scripts/check_mma_path.sh \
           scripts/capture_traces.py scripts/calibrate_hardware.py \
           scripts/publish_results.sh; do
    if [[ ! -f "$s" ]]; then
      missing=$((missing + 1)); note "MISSING  $s"; continue
    fi
    case "$s" in
      *.py) "$PY_BASE" -m py_compile "$s" >/dev/null 2>&1 || { broken=$((broken + 1)); note "SYNTAX   $s"; } ;;
      *.sh) bash -n "$s" >/dev/null 2>&1 || { broken=$((broken + 1)); note "SYNTAX   $s"; } ;;
    esac
  done
  [[ "$missing" == "0" && "$broken" == "0" ]]; verdict P11a "step scripts present and parse" $? \
    "$missing missing, $broken unparseable" "0 and 0" soft \
    "The affected step is skipped and its claim goes unmeasured this session; every other step still runs. Pull the branch that carries the missing script before spending the pod on the rest."

  # THE FLAGS THIS SESSION PASSES MUST EXIST. Three of these scripts are being
  # written concurrently by other people, and a renamed flag is an argparse error
  # forty minutes into a metered session rather than a line here. Checked against
  # the SOURCE rather than by running --help, because --help on a script that
  # imports vLLM cannot run on a laptop, and this check has to work off-GPU.
  local badflag=0 pair name flag
  for pair in "scripts/block_m_crossing_sweep.py:--out" \
              "scripts/group_m_alpha_sweep.py:--out" \
              "scripts/group_m_alpha_sweep.py:--run" \
              "scripts/alias_ablation.py:--out" \
              "scripts/alias_ablation.py:--run" \
              "scripts/alias_ablation.py:--replay" \
              "scripts/nsys_dram_probe.py:--out" \
              "scripts/nsys_dram_probe.py:--calibrate" \
              "scripts/nsys_dram_probe.py:--measure" \
              "scripts/nsys_dram_probe.py:--timeout" \
              "scripts/tuned_vs_fallback.py:--out-dir" \
              "scripts/check_mma_path.sh:--model" \
              "scripts/check_mma_path.sh:--tokens" \
              "scripts/check_mma_path.sh:--out" \
              "scripts/capture_traces.py:--model" \
              "scripts/capture_traces.py:--phase" \
              "scripts/capture_traces.py:--corpus" \
              "scripts/capture_traces.py:--out"; do
    name="${pair%%:*}"; flag="${pair##*:}"
    [[ -f "$name" ]] || continue
    # The flag must be DELIMITED, so --out does not match --out-dir: argparse
    # writes it as "--out" and a bash case arm writes it as --out), and both end
    # in a character that cannot be part of a longer flag.
    grep -qE -- "[\"' ]${flag}[\"')=]" "$name" \
      || { badflag=$((badflag + 1)); note "NO FLAG  $name $flag"; }
  done
  [[ "$badflag" == "0" ]]; verdict P11b "step scripts accept the flags used here" $? \
    "$badflag flag(s) not found in the scripts" "== 0" soft \
    "The step that uses the missing flag will die on an argparse error partway through the session. Fix the invocation in scripts/pod_session.sh, or the flag name in the step script, before starting."

  head2 "P12  the test suite, which costs seconds and saves the hour"
  if [[ "$SKIP_TESTS" == "1" ]]; then
    skipped P12 "test suite" "--skip-tests"
  else
    local tlog="$SESSION/logs/pytest.log"
    "$PY_BASE" -m pytest tests/ -q > "$tlog" 2>&1
    local trc=$? tline
    tline="$(tail -3 "$tlog" | grep -E '[0-9]+ (passed|failed)' | tail -1)"
    [[ "$trc" == "0" ]]; verdict P12 "test suite" $? \
      "${tline:-see $tlog}" "exit 0" fatal \
      "A failure here costs seconds. The same failure discovered after an hour of benchmarking costs an hour, and every row produced in between is suspect. Full log at $tlog."
  fi

  head2 "P13  thermal and clock state before anything is timed"
  # On rented hardware thermal state is the largest source of run-to-run
  # disagreement, and the harness records the symptom rather than controlling it.
  if [[ "$HAVE_GPU" == "0" ]]; then
    skipped P13 "clocks and throttle" "$(absent_reason gpu)"
  else
    local thr temp
    thr="$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    temp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    note "throttle reasons $thr, ${temp}C"
    [[ "$thr" == "0x0000000000000000" ]]; verdict P13a "no active throttle" $? \
      "$thr" "== 0x0000000000000000" soft \
      "Timings will carry a thermal component the ridge band already blames for a 9.9% spread on one card. Wait for the card to settle, or accept a wider band and say so in the writeup."
    [[ "${temp:-100}" -le 60 ]]; verdict P13b "cold start" $? \
      "${temp:-?} C" "<= 60 C" soft \
      "The calibration in step 1 is measured on a hot card and every efficiency figure this session is quoted against it."
  fi

  head2 "P14  results survive teardown"
  # A Network Volume at /workspace survives pod termination. The container
  # filesystem does not. This compares mount points rather than paths, because
  # /workspace on a pod WITHOUT a volume attached is just a container directory
  # and looks identical.
  local m_sess m_root
  m_sess="$(df -P "$SESSION" 2>/dev/null | awk 'NR==2 {print $1}')"
  m_root="$(df -P / 2>/dev/null | awk 'NR==2 {print $1}')"
  note "session on '$m_sess', container root on '$m_root'"
  if [[ "$HAVE_GPU" == "0" ]]; then
    skipped P14 "session dir outlives the pod" "not on a pod; nothing is at risk"
  else
    [[ -n "$m_sess" && "$m_sess" != "$m_root" ]]; verdict P14 "session dir outlives the pod" $? \
      "${m_sess:-unknown} vs ${m_root:-unknown}" "different mounts" fatal \
      "$SESSION is on the container filesystem and dies with the pod. Attach the Network Volume at /workspace, or pass --session-dir pointing into it. Step 8 can still push to git, but a session that crashes before step 8 would leave nothing at all."
  fi

  # LAST IN PRE-FLIGHT, AND STILL BEFORE ANYTHING IS SPENT. It is the only check
  # here that costs minutes rather than seconds, so it goes after the cheap fatal
  # ones: there is no sense sampling DRAM for two minutes on a pod whose test
  # suite is red or whose GPU is the wrong part. It is skipped entirely once a
  # fatal has tripped, because that session is not going to run.
  if halted; then
    NSYS_WHY="a fatal pre-flight gate tripped first ($STOPPED), so the probe was not run"
    head2 "P-nsys  skipped"
    note "$NSYS_WHY"
    ledger Pnsys "nsys DRAM sampling" SKIP "$NSYS_WHY" "" ""
  else
    preflight_nsys
  fi
  write_provenance
  note "provenance recorded at $PROVENANCE_FILE; every step that quotes a byte figure reads it"
}

# ==========================================================================
# STEPS
# ==========================================================================

#: Run a command, tee to a per-step log, return the COMMAND's exit code.
#:
#: A pipe into tee rather than `> >(tee ...)` on purpose. Process substitution
#: leaves tee running asynchronously, so a gate that greps the log immediately
#: afterwards can read it before the last line has been flushed and report a
#: missing verdict that was in fact printed. Step 5 greps its log for the wgmma
#: count on the very next line, which is exactly that race. PIPESTATUS[0] keeps
#: the command's status rather than tee's.
run_logged() {
  local logfile="$1"; shift
  say "  \$ $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    note "dry run: not executed"
    return 0
  fi
  "$@" 2>&1 | tee -a "$logfile"
  return "${PIPESTATUS[0]}"
}

#: The convention every step script in this session honours: it prints at least
#: one line carrying the word PASS and no line carrying the word FAIL. Counted
#: here so a sibling script's own scientific gate is enforced by the session
#: rather than read by a human. A script that prints NEITHER is a failure of a
#: different kind, and is reported as such rather than silently passing.
#:
#: CORRECTED 2026-09-01. The first version of this matched NONE of the three
#: scripts it exists to read, and was validated against synthetic logs rather than
#: real ones. Measured against the actual output:
#:      group_m         "  [PASS] regime: every cell is memory bound"
#:      block_m         "GATE 0  PASS      override_config changed ..."
#:      tuned_vs_fallb  "[PASS] G0  every arm same layer"
#: the old '(^|:)[[:space:]]*PASS' read 0 PASS and 0 FAIL from all three, so S2b,
#: S3b and S4b FAILED on a perfect run and a genuine refutation was equally
#: invisible. The entire scientific payload of this session had no verdict channel.
#:
#: TWO ACCEPTED SHAPES, matched exactly rather than loosely:
#:      [PASS] anything          the bracketed tag, at the start of the line
#:      GATE 7  PASS  anything   block_m's form
#: Deliberately NOT a bare whitespace boundary: block_m prints the prose line
#: "a FAIL here is the interesting answer", and render_gates prints a
#: "2 PASS, 1 FAIL" summary. Both would false-positive under a looser rule, and a
#: false PASS is the one direction this must never fail in.
gate_from_log() {
  local logfile="$1" rx p f
  rx='^[[:space:]]*\[(%s)\]|^GATE [0-9]+[[:space:]]+(%s)([[:space:]]|$)'
  # shellcheck disable=SC2059
  p="$(grep -cE "$(printf "$rx" PASS PASS)" "$logfile" 2>/dev/null || true)"
  f="$(grep -cE "$(printf "$rx" FAIL FAIL)" "$logfile" 2>/dev/null || true)"
  echo "${p:-0} ${f:-0}"
}

# --------------------------------------------------------------------------
step0_download() {
  head1 "STEP 0  (0:00)  mixtral weights, backgrounded off the critical path"
  say "  PREDICTION: 93.4 GB lands in $HF_HOME within about 90 minutes, so it is"
  say "  ready when step 7 needs it at 3:40 and never blocks a timed step."
  if [[ "$NO_DOWNLOAD" == "1" ]]; then
    skipped S0 "mixtral download" "--no-download"; return 0
  fi
  if already_passed S0; then skipped S0 "mixtral download" "already started this session"; return 0; fi
  local dlog="$SESSION/logs/download.log"
  # A DRY RUN MUST STILL EXERCISE THE BACKGROUNDING. This used to print the
  # command and return, so the only untested line in the script was the one that
  # launches a 93 GB transfer -- and it was broken from the day it was written.
  # `$!` was read in a shell that had started no background job, `set -u` aborted
  # the session, and a laptop dry run could never have seen it. So the dry run
  # now runs the SAME shape against a harmless command: same subshell, same
  # backgrounding, same `$!` read, same pid file. It costs milliseconds and it
  # covers the structure, which is the part that was wrong.
  if [[ "$DRY_RUN" == "1" ]]; then
    say "  \$ (background) huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1"
    ( : ) >> "$dlog" 2>&1 &
    # set +u around the read, because `${!:-}` does NOT rescue an unset `$!`:
    # bash treats the `!` as indirect expansion and still aborts under set -u,
    # so the obvious-looking default is dead code. Verified both ways.
    set +u; local dry_pid="${!:-}"; set -u
    if [[ -z "$dry_pid" ]]; then
      verdict S0 "mixtral download" 1 "no pid from a backgrounded subshell" \
        "\$! is set after backgrounding" fatal \
        "The real step 0 reads \$! the same way and would abort the session under set -u. This is the structure test, not the download."
      return 0
    fi
    wait "$dry_pid" 2>/dev/null || true
    skipped S0 "mixtral download" "dry run; backgrounding structure exercised, pid $dry_pid"
    return 0
  fi
  # HF_HUB_OFFLINE is exported =1 by run_all.sh so a sweep can never reach a hub;
  # this is the one deliberate download and it must not inherit that.
  # THE `&` GOES OUTSIDE THE SUBSHELL. It used to sit inside, which backgrounded
  # the job in a child shell whose PID the parent never sees: the parent had
  # started no background job at all, so `$!` was UNSET, and `set -u` aborted the
  # whole session with "line 1083: $!: unbound variable" -- after nohup had
  # already launched the 93 GB download, so the run died while the transfer it
  # had just started kept going invisibly. Backgrounding the subshell itself
  # makes `$!` the subshell's PID, which is what download.pid is for.
  ( HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 nohup "$PY_BASE" - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("mistralai/Mixtral-8x7B-Instruct-v0.1",
                      allow_patterns=["*.json", "*.safetensors", "*.model"],
                      max_workers=8)
print(f"DOWNLOAD-COMPLETE {p}")
PYEOF
  ) >> "$dlog" 2>&1 &
  set +u; local pid="${!:-}"; set -u
  if [[ -z "$pid" ]]; then
    # Belt and braces: never let a missing PID abort a session over a step that
    # is explicitly OFF the critical path.
    verdict S0 "mixtral download started" 1 "no pid" "a backgrounded pid" soft \
      "The download may still be running. Check with: pgrep -fa snapshot_download"
    return 0
  fi
  echo "$pid" > "$SESSION/download.pid"
  verdict S0 "mixtral download started" 0 "pid $pid, log $dlog" "backgrounded" soft ""
  note "progress: tail -f $dlog   ;   du -sh $HF_HOME"
}

# --------------------------------------------------------------------------
step1_calibrate() {
  head1 "STEP 1  (0:05)  fp8 SAME-SESSION calibration -- one line, two results"
  cat <<'TXT'
  WHAT IT GATES. Two published results currently rest on a calibration that did
  not belong to the session that measured them:
    1. the dtype headline (bf16 1.162 against fp8 1.361 pooled, 1.475 matched)
       comes from an arm entitled_ridge REFUSES, because that arm's calibration
       measured no fp8 ceiling;
    2. every fp8 row in it carries achieved_peak_tflops = 0.0, so its
       implied_traffic_ratio column is empty and alpha cannot be fit across
       dtypes at all.
  Measuring an fp8 ceiling in THIS session fixes both going forward. It does NOT
  rescue the published fp8 arm: restamping those rows against today's ruler would
  be exactly the borrowed calibration of instrument defect 7, and
  scripts/recompute_ceilings.py is the wrong tool here.

  PREDICTION, from the three existing H200 calibrations:
    bandwidth (triad)  4374-4377 GB/s   reproduces to 0.06% across sessions
    dense bf16         701-771 TFLOP/s  the term that does NOT reproduce, 9.9% spread
    bf16 ridge         160.3-176.2 FLOP/byte
    fp8_e4m3 peak      about 1409 TFLOP/s, 1.83x the bf16 figure
TXT
  # The dtype headline is a RATIO OF EFFICIENCIES, and an efficiency is a time
  # divided by a modelled byte count. Whether those bytes were counted or
  # computed is part of the claim, so it is printed with it.
  provenance_banner
  if already_passed S1; then skipped S1 "calibration" "already passed"; return 0; fi
  if [[ "$HAVE_GPU" == "0" ]]; then
    skipped S1 "calibration" "$(absent_reason gpu)"; return 0
  fi
  local clog="$SESSION/logs/calibrate.log"
  run_logged "$clog" "$PY_BASE" scripts/calibrate_hardware.py
  local rc=$?
  [[ "$rc" == "0" ]]; verdict S1a "calibrate_hardware exit" $? "exit $rc" "== 0" fatal \
    "Without a calibration for this machine the sweep runs with EMPTY efficiency columns, which is how an H100 pod once silently satisfied the repo's H200 yaml. Log at $clog."
  halted && return 2

  # Read the numbers back out of the file the sweep will actually resolve, not
  # out of the log, so the gate tests what the rows will be stamped from.
  local vals
  vals="$("$PY_BASE" - <<'PYEOF' 2>/dev/null
import datetime, sys
sys.path.insert(0, ".")
import yaml
from moe.bench.calibrate import read_stamp
from moe.bench.roofline import HARDWARE_DIR, current_gpu_name, measured_slug
# measured_slug(current_gpu_name()) is the SAME resolution load_measured and
# therefore the sweep itself uses. Globbing measured_*.yaml instead is how an
# H100 pod once satisfied a committed H200 file: the gate must not be able to
# disagree with the run about which ruler applies.
# NO APOSTROPHES ANYWHERE BELOW. bash 3.2 tracks quotes through a heredoc that
# sits inside a command substitution, so one stray quote here is a syntax error
# in the whole script, reported at a line hundreds of lines away.
name = measured_slug(current_gpu_name() or "")
path = HARDWARE_DIR / (name + ".yaml")
if not path.exists():
    print("|||||")
    raise SystemExit(0)
doc = yaml.safe_load(path.read_text()) or {}
peaks = doc.get("compute_dense_tflops") or {}
bw = ((doc.get("memory") or {}).get("bandwidth_tb_s") or 0.0) * 1000.0
stamp = read_stamp(path)
ridge = (stamp.ridge("bf16") if stamp else 0.0) or 0.0
fields = [str(path), str(doc.get("checked_on")), "%.1f" % bw,
          "%.1f" % (peaks.get("bf16", 0) or 0), "%.1f" % (peaks.get("fp8_e4m3", 0) or 0),
          "%.1f" % ridge, datetime.date.today().isoformat()]
print("|".join(fields))
PYEOF
)"
  local capath cdate cbw cbf16 cfp8 cridge today
  IFS='|' read -r capath cdate cbw cbf16 cfp8 cridge today <<< "$vals"
  note "$capath  checked_on=$cdate  bw=${cbw} GB/s  bf16=${cbf16} TFLOP/s  fp8_e4m3=${cfp8} TFLOP/s  ridge=${cridge}"

  [[ -n "$today" && "$cdate" == "$today" ]]; verdict S1b "calibration is same-session" $? \
    "checked_on ${cdate:-none}, today ${today:-unreadable}" "equal, and both readable" fatal \
    "The yaml was not rewritten, so this session would publish against another session's ruler. That is instrument defect 7 exactly, and it cost claim C5 its target for three days."

  "$PY_BASE" -c "import sys; sys.exit(0 if float('${cfp8:-0}') > 0 else 1)"
  verdict S1c "fp8 ceiling measured" $? \
    "fp8_e4m3 ${cfp8:-0} TFLOP/s" "> 0" soft \
    "Any fp8 row measured today will again carry achieved_peak_tflops = 0.0 and be unquotable, so the dtype headline stays resting on a refused arm. On a card without fp8 tensor cores (the A100) this SHOULD be 0 and the failure is correct."

  "$PY_BASE" -c "
import sys
bw = float('${cbw:-0}')
sys.exit(0 if 3900.0 <= bw <= 4800.0 else 1)"
  verdict S1d "bandwidth in the reproducible band" $? \
    "${cbw:-0} GB/s" "3900-4800" soft \
    "Bandwidth reproduces to 0.06% across three sessions on this part. A figure outside this band means the buffer fit in cache (too high) or the card is contended (too low), and every implied_traffic_ratio this session is divided by it."

  "$PY_BASE" -c "
import sys
r = float('${cridge:-0}')
sys.exit(0 if 150.0 <= r <= 185.0 else 1)"
  verdict S1e "bf16 ridge in the measured band" $? \
    "${cridge:-0} FLOP/byte" "150-185, band is 160.3-176.2" soft \
    "Every AI-cap prediction in this session is stated against 160.3. A ridge outside the band moves which BLOCK_M can cross, so re-derive the predictions before reading step 2."

  # Snapshot the ruler and its checksum NOW. publish_results.sh copies whatever
  # is in measured_<device>.yaml AT PUBLISH TIME, and on 2026-08-28 a
  # recalibration landed between a sweep and its publish; the two rulers disagree
  # by 9.9% on the compute ceiling and nothing downstream could tell. Step 8
  # re-checks this sum.
  if [[ -f "$capath" ]]; then
    cp "$capath" "$SESSION/calibration/"
    ( cd "$(dirname "$capath")" && shasum -a 256 "$(basename "$capath")" 2>/dev/null \
      || sha256sum "$(basename "$capath")" ) > "$SESSION/calibration/measured.sha256"
    note "ruler snapshotted to $SESSION/calibration/ and checksummed; step 8 verifies it never moved"
  fi

  # END TO END, which the yaml alone does not prove: run a handful of real fp8
  # cells and confirm the ROWS carry a non-zero achieved_peak_tflops. The yaml
  # having an fp8 key and the driver stamping it onto a row are two different
  # facts, and it is the second one the dtype headline needs.
  local fdir="$RESULTS_DIR/fp8-stamp-check" flog="$SESSION/logs/fp8_stamp.log"
  run_logged "$flog" "$PY_BASE" -m moe.bench.cli --profile fp8 \
    --models deepseek-v2-lite --tokens 64 --routings uniform \
    --groups reference,kernels,baselines --out-dir "$fdir" --max-minutes 3
  local stamped
  stamped="$("$PY_BASE" - "$fdir" <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, ".")
from pathlib import Path
from moe.bench.schema import read_csv
rows = []
for p in sorted(Path(sys.argv[1]).glob("run_*.csv")):
    rows.extend(read_csv(p))
timed = [r for r in rows if float(r.get("ms_p50") or 0) > 0]
good = [r for r in timed if float(r.get("achieved_peak_tflops") or 0) > 0]
print(f"{len(good)} {len(timed)}")
PYEOF
)"
  local ngood ntimed
  ngood="${stamped%% *}"; ntimed="${stamped##* }"
  [[ "${ntimed:-0}" -gt 0 && "${ngood:-0}" == "${ntimed:-0}" ]]
  verdict S1f "fp8 rows carry a peak" $? \
    "${ngood:-0} of ${ntimed:-0} timed fp8 rows stamped" "all of them, and at least one" soft \
    "The yaml has an fp8 ceiling but the driver is not stamping it onto rows, so an fp8 arm swept today would repeat the exact defect this step exists to close. Check moe/bench/driver.py around achieved_peak_tflops."
}

# --------------------------------------------------------------------------
step2_block_m() {
  head1 "STEP 2  (0:08)  BLOCK_M sweep, multi-tile -- the session's central result"
  cat <<'TXT'
  WHAT IT TESTS. Whether arithmetic intensity is BOUNDED by the tile height, as
  AI -> 2 BM / (alpha b) says it is, or whether the uncorrected 2R/b holds and
  the crossing does not move with BLOCK_M at all.

  PREDICTION at alpha = 0.558, ridge 160.3, bf16:
    BLOCK_M=32    AI cap  57    NO CROSSING AT ANY BATCH
    BLOCK_M=64    AI cap 115    NO CROSSING AT ANY BATCH
    BLOCK_M=128   AI cap 229    R_cross 250  (mixtral 999 tok, qwen2 1998, ds-v3 7992)
    BLOCK_M=256   AI cap 459    R_cross 160  (mixtral 641 tok, qwen2 1282, ds-v3 5130)
  128 and 256 must separate by 1.56x. The retracted alpha = 0.10 predicts all four
  crossing with a 1.10x spread, so the two alphas are QUALITATIVELY different here
  and one sweep decides between them.

  RUN IT IN THE MULTI-TILE REGIME. C3 measured the tile at T=16, where every
  expert is one tile at every BLOCK_M and there are no re-reads to save, so only
  occupancy could move. That regime is not this one and its null result does not
  transfer. Wave count must exceed about 10 on BOTH sides of a step.

  A PASS means the tile-corrected roofline survives a test that could have
  killed it. A FAIL where 32 or 64 DOES cross means alpha < 0.0998 after all and
  the refit is wrong; a FAIL where 128 and 256 separate by ~1.10x instead of
  1.56x means the uncorrected 2R/b is right and this whole section retracts.

  STEP 2b RUNS STRAIGHT AFTER THIS ONE and measures the same alpha by ablation,
  touching no byte model at all. Read the two together in the summary; this step
  on its own cannot tell a wrong alpha from a wrong byte model.
TXT
  provenance_banner
  if already_passed S2; then skipped S2 "BLOCK_M sweep" "already passed"; return 0; fi
  [[ -f scripts/block_m_crossing_sweep.py ]] || { skipped S2 "BLOCK_M sweep" "scripts/block_m_crossing_sweep.py not present"; return 0; }
  if [[ "$HAVE_VLLM" == "0" ]]; then skipped S2 "BLOCK_M sweep" "$(absent_reason vllm)"; return 0; fi
  local slog="$SESSION/logs/block_m.log"
  # --fail-on-gate is REQUIRED: without it block_m exits 0 with three gates
  # failed, so S2a passes on a refutation and the session's central result has
  # no way to report one. group_m and tuned_vs_fallback already exit 1.
  run_logged "$slog" "$PY_VLLM" scripts/block_m_crossing_sweep.py --fail-on-gate --out "$SESSION/block_m"
  local rc=$?
  [[ "$rc" == "0" ]]; verdict S2a "sweep exit" $? "exit $rc" "== 0" soft \
    "The alpha refit stays unvalidated on hardware and docs/FINDINGS.md keeps its NOT YET VALIDATED banner. Steps 3 and 4 still stand: they measure how alpha VARIES, which is a separate claim from its level."
  local counts p f
  counts="$(gate_from_log "$slog")"; p="${counts%% *}"; f="${counts##* }"
  [[ "${f:-1}" == "0" && "${p:-0}" -ge 1 ]]; verdict S2b "sweep's own gates" $? \
    "${p:-0} PASS, ${f:-0} FAIL lines" "at least one PASS, no FAIL" soft \
    "Read $slog and decide per prediction which of the four BLOCK_M rows failed. A FAIL at 32 or 64 is the interesting outcome, not a broken run."
}

# --------------------------------------------------------------------------
#: The one traffic figure this session can actually COUNT, if P-nsys said yes.
#:
#: WHY ONE MEASUREMENT AND NOT A TRACE OVER THE SWEEP. nsys samples the DRAM
#: counters at a fixed rate, so a single launch of the cell this study cares
#: about is not measurable at ANY rate nsys offers: the probe prints the
#: arithmetic and one 54 us launch buys half a sample at the nsys default and ten
#: at the 200 kHz ceiling, and the quantisation is charged per WINDOW rather than
#: per sample. What IS measurable is a long contiguous run of back-to-back
#: launches merged into one window. A 45-minute sweep is the opposite of that: it
#: is thousands of separate launches with compilation and allocation between them,
#: so wrapping the sweep in nsys would buy samples and no accuracy at all. So the
#: cell is profiled ON ITS OWN, in a loop, next to the sweep that ran it.
#:
#: WHAT IT SETTLES. deepseek-v3 up-stage at T=4096, where the two candidate
#: traffic models are 45% apart: 256 active experts x 58.72 MB = 15.03 GB if
#: traffic is active x W, against 370 M-tiles x 58.72 MB = 21.73 GB if every
#: M-tile re-reads. That gap IS the tile term, so this is a direct read on the
#: same physics as alpha, by counting rather than by fitting.
step2_nsys_cell() {
  if [[ "$NSYS_SAMPLES_DRAM" != "1" ]]; then
    skipped SN "nsys traffic measurement" "traffic is INFERRED: $NSYS_WHY"
    return 0
  fi
  if already_passed SN; then skipped SN "nsys traffic measurement" "already passed"; return 0; fi
  head2 "step 2, under nsys: the compulsory byte model against counted bytes"
  local mlog="$SESSION/logs/nsys_cell.log" mdir="$SESSION/nsys/cell"
  mkdir -p "$mdir"
  run_logged "$mlog" with_deadline 300 "$PY_BASE" scripts/nsys_dram_probe.py \
    --measure --timeout 30 --seconds 3.0 --out "$mdir"
  local rc=$?
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  local ratio
  ratio="$("$PY_BASE" - "$mdir/probe.json" <<'PYEOF' 2>/dev/null
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print("no probe.json")
    raise SystemExit(0)
cell = (json.loads(p.read_text()).get("cell") or {})
line = (cell.get("comparison") or cell.get("error") or "the probe measured no cell")
print(line.replace("\n", " ")[:200])
PYEOF
)"
  note "$ratio"
  if [[ "$rc" != "0" ]]; then
    # THE LABEL FOLLOWS THE MEASUREMENT, NOT THE CAPABILITY. A sampler that works
    # and produced no cell figure leaves every downstream traffic number exactly
    # as modelled as it was before, so the word downstream must go back to
    # INFERRED. Claiming MEASURED because the instrument exists is the same shape
    # of error as quoting an arm against a calibration from another session.
    NSYS_SAMPLES_DRAM=0
    NSYS_WHY="nsys samples DRAM here but the cell measurement failed (exit $rc), so nothing was actually counted"
    TRAFFIC_PROVENANCE="INFERRED"
    write_provenance
  fi
  [[ "$rc" == "0" ]]; verdict SN "nsys traffic measurement" $? \
    "exit $rc: ${ratio:-nothing parsed}" "== 0 and a measured-over-modelled ratio" soft \
    "The sampler worked in pre-flight and did not produce a cell measurement here. Traffic figures go back to INFERRED for this session and every later step says so; nothing else is affected. Log at $mlog."
}

# --------------------------------------------------------------------------
step2b_alias_ablation() {
  head1 "STEP 2b (0:16)  ALIAS ABLATION -- alpha again, without the byte model"
  cat <<'TXT'
  WHY THIS EXISTS AND WHY IT RUNS HERE. alpha = 0.558 is the number the whole
  tile-corrected roofline rests on, and it comes from ONE regression against a
  byte model that has no tile term in it: alpha_refit fits it out of
  implied_traffic_ratio = time x bandwidth / COMPULSORY BYTES. C4 is a confirmed
  finding that the compulsory-byte ruler was itself wrong by 1.85% until it was
  fixed. One estimator over one derived column is not enough support for a
  headline, so this measures the same physical quantity by a route that touches
  no compulsory bytes, no calibrated bandwidth, no ridge and no fitted intercept.

  THE METHOD. Run the same grouped-GEMM access pattern twice at n M-tiles per
  expert, n = 1, 2, 4, 8. NORMAL reads each expert weight block once per M-tile.
  ALIASED points every weight load at ONE resident tile, so the loads still
  execute and still occupy the same instructions but they HIT L2 instead of
  missing to HBM. Everything else is identical and cancels:

    D(n) = T_normal(n) - T_aliased(n) = W (1 + alpha (n - 1)),   D(1) = W

  so alpha = (D(n)/D(1) - 1)/(n - 1) with W measured in the same units by the
  same clock, never predicted. Rows per expert is an exact multiple of BLOCK_M at
  every rung, so the padding term is EXACTLY zero and the tile count is the only
  thing that moves.

  IT REPORTS A BRACKET, NOT A POINT, and that is deliberate. The aliased variant
  issues the same loads, so both variants push n W bytes through L2 and that
  common cost cancels only if L2 and HBM service ADD. A streaming kernel runs
  closer to max(L2, HBM), where the naive difference estimator is biased toward
  exactly the retracted 0.10. So a second estimator, biased the other way, is run
  beside it and the answer is the interval between them.

  PREDICTION: the interval contains 0.558 and excludes 0.10 and TEMPO's 0.33.

  EXIT CODES ARE MEANINGS. 0 every gate passed. 1 a gate FAILED, which is a
  refutation and a result, not a crash. 3 it cannot run here. 4 the design did
  not identify alpha and the honest answer is NOT TESTABLE rather than a number.

  THE THERMAL POINT. This runs immediately after step 2 so both alpha estimates
  share one session and one thermal state. A disagreement between them then
  cannot be explained away by the card.
TXT
  provenance_banner
  note "this step is independent of the line above: it never divides by a byte count at all"
  if already_passed SB; then skipped SB "alias ablation" "already passed"; return 0; fi
  [[ -f scripts/alias_ablation.py ]] || { skipped SB "alias ablation" "scripts/alias_ablation.py not present"; return 0; }
  if [[ "$HAVE_GPU" == "0" ]]; then skipped SB "alias ablation" "$(absent_reason gpu)"; return 0; fi

  # PY_VLLM, not PY_BASE: the kernel reproduces vLLM 0.27.1 B-pointer arithmetic
  # and tile constants, and its ISA gate reads PTX from the compiler step 5
  # characterises, which is the vLLM venv one. PY_VLLM falls back to PY_BASE when
  # that venv is absent, so a base-only pod still runs it.
  local slog="$SESSION/logs/alias_ablation.log"
  run_logged "$slog" "$PY_VLLM" scripts/alias_ablation.py --run --out "$SESSION/alias_ablation"
  local rc=$?
  [[ "$rc" == "0" ]]; verdict SBa "ablation exit" $? "exit $rc" "== 0" soft \
    "exit 1 is a REFUTATION and the most valuable outcome this session can produce: read the P1 gate line, which names which candidate alpha the interval does contain. exit 4 means the interval was too wide to separate 0.10 from 0.558 and no candidate may be quoted from it. exit 3 means it never ran; the deepseek-v3 rung needs about 16 GiB free on the card. Either way the regression alpha from step 3 stands alone and the paper keeps its single-estimator caveat."
  local counts p f
  counts="$(gate_from_log "$slog")"; p="${counts%% *}"; f="${counts##* }"
  [[ "${f:-1}" == "0" && "${p:-0}" -ge 1 ]]; verdict SBb "ablation's own gates" $? \
    "${p:-0} PASS, ${f:-0} FAIL lines" "at least one PASS, no FAIL" soft \
    "A FAIL on ISA means the two variants did not compile to the same instruction stream, and nothing from this run may be quoted whatever P1 says. A FAIL on placebo, signal, form, control or resolution voids the design rather than the prediction. A FAIL on P1 alone is the refutation. $slog names which."
  note "the reconciliation against step 3's regression alpha is printed at the very end of this session"
}

# --------------------------------------------------------------------------
step3_group_m() {
  head1 "STEP 3  (0:26)  GROUP_SIZE_M sweep -- is alpha a scalar or a swizzle knob"
  cat <<'TXT'
  WHAT IT TESTS. The refit found alpha FALLING with GROUP_SIZE_M: 0.570 at 1,
  0.488 at 16. That is exactly what a swizzle-for-L2-reuse mechanism predicts,
  which turns alpha from a fudge factor into something with a named cause. But
  GROUP_SIZE_M 32 and 64 have ZERO discriminating rows in the published pool, so
  the direction is UNTESTED beyond 16 and cannot be tested from existing data at
  any effort. Only override_config varying it settles it, and NOT at fixed batch:
  # group_m_alpha_sweep.py deliberately refuses that instruction, because one batch
  # cannot identify alpha under this estimator -- the token count IS the intercept,
  # so a single x-level is absorbed exactly and only curvature is left. On the
  # design's own x values at 0.5% noise the top rung alone gives a 90% band of
  # 0.373-0.756 against the seven-rung ladder's 0.552-0.580, 14x narrower against an
  # effect size of 0.082. The ladder is identical across every GROUP_SIZE_M, so the
  # cross-setting comparison is still at fixed design.

  PREDICTION: alpha continues to fall monotonically at 32 and 64, and the fall
  flattens as the swizzle stops buying reuse. A rise at any point refutes the
  mechanism and returns alpha to being an unexplained constant.

  This is also the step that tests whether alpha may be reported as a scalar at
  all. It already drifts with BLOCK_M (0.466 at 64, 0.625 at 128), so a single
  number carries a range whatever this step finds.

  THIS IS THE SESSION'S REGRESSION ESTIMATE OF ALPHA. Its GROUP_SIZE_M = 1 row is
  what step 2b's ablation is reconciled against at the end, and unlike the
  published refit it is measured on this card in this session.
TXT
  provenance_banner
  if already_passed S3; then skipped S3 "GROUP_SIZE_M sweep" "already passed"; return 0; fi
  [[ -f scripts/group_m_alpha_sweep.py ]] || { skipped S3 "GROUP_SIZE_M sweep" "scripts/group_m_alpha_sweep.py not present"; return 0; }
  if [[ "$HAVE_VLLM" == "0" ]]; then skipped S3 "GROUP_SIZE_M sweep" "$(absent_reason vllm)"; return 0; fi
  local slog="$SESSION/logs/group_m.log"
  run_logged "$slog" "$PY_VLLM" scripts/group_m_alpha_sweep.py --run --out "$SESSION/group_m"
  local rc=$?
  [[ "$rc" == "0" ]]; verdict S3a "sweep exit" $? "exit $rc" "== 0" soft \
    "The swizzle mechanism for alpha stays a two-point trend from the published pool, and 'alpha varies with the swizzle' must keep its UNTESTED label."
  local counts p f
  counts="$(gate_from_log "$slog")"; p="${counts%% *}"; f="${counts##* }"
  [[ "${f:-1}" == "0" && "${p:-0}" -ge 1 ]]; verdict S3b "sweep's own gates" $? \
    "${p:-0} PASS, ${f:-0} FAIL lines" "at least one PASS, no FAIL" soft \
    "A non-monotonic alpha here is a real result and should be reported, not retried."
}

# --------------------------------------------------------------------------
step4_tuned_vs_fallback() {
  head1 "STEP 4  (0:41)  tuned config against a forced fallback"
  cat <<'TXT'
  WHAT IT TESTS. Only 2 of the 8 (model x card) cells in this study have a tuned
  vLLM config at all, both on the H200: E=8,N=14336 (mixtral) and E=64,N=2560
  (qwen2). The other six take the hardcoded bf16 ladder, M<=32 -> 16, M<=96 -> 32,
  M<=512 -> 64, else 128, and vLLM says so on the log with "Using default MoE
  config. Performance might be sub-optimal!". Nothing in this study has ever
  measured what that warning is worth.

  PREDICTION. In the memory-bound regime the difference is small, because the
  tile is not on the critical path there; in the multi-tile regime the tuned file
  climbs to BLOCK_M=128 at M=256 while the fallback sits at 64, and the tile
  ceiling says only 128 can ever cross. So the gap should OPEN with batch rather
  than being a constant offset.

  This is also the step that prices "BLOCK_M is not a knob", a claim this repo
  published and had to retract: vLLM and SGLang both ship tuned BLOCK_SIZE_M, so
  the honest question is what the tuning buys, not whether it exists.
TXT
  provenance_banner
  if already_passed S4; then skipped S4 "tuned vs fallback" "already passed"; return 0; fi
  [[ -f scripts/tuned_vs_fallback.py ]] || { skipped S4 "tuned vs fallback" "scripts/tuned_vs_fallback.py not present"; return 0; }
  if [[ "$HAVE_VLLM" == "0" ]]; then skipped S4 "tuned vs fallback" "$(absent_reason vllm)"; return 0; fi
  local slog="$SESSION/logs/tuned_vs_fallback.log"
  # --max-minutes caps the only step that can exceed its slot: 76 distinct
  # (model, config) specialisations means ~152 Triton compiles at 8-25 s each,
  # which is 21-64 min of pure compile. The script writes a PARTIAL RUN report
  # on the cap and resumes on re-entry.
  run_logged "$slog" "$PY_VLLM" scripts/tuned_vs_fallback.py --max-minutes 35 --out-dir "$SESSION/tuned_vs_fallback"
  local rc=$?
  [[ "$rc" == "0" ]]; verdict S4a "comparison exit" $? "exit $rc" "== 0" soft \
    "The cost of running on the fallback ladder stays unmeasured, and six of the study's eight cells are on it."
  local counts p f
  counts="$(gate_from_log "$slog")"; p="${counts%% *}"; f="${counts##* }"
  [[ "${f:-1}" == "0" && "${p:-0}" -ge 1 ]]; verdict S4b "comparison's own gates" $? \
    "${p:-0} PASS, ${f:-0} FAIL lines" "at least one PASS, no FAIL" soft \
    "Check $slog for which shape disagreed with the prediction before treating it as a bug."
}

# --------------------------------------------------------------------------
step5_isa() {
  head1 "STEP 5  (1:16)  config and ISA provenance, both cards"
  cat <<'TXT'
  WHAT IT TESTS, AND WHY IT MUST LEAVE THE POD AS A FILE. C1 and C3 are the only
  claims in docs/FINDINGS.md that rest on transient pod output: the PTX dumps,
  the CUTLASS kernel names and the "Using default MoE config" warning were all
  quoted from run logs that were never committed. Every other claim can be
  recomputed from results/published/ on a laptop; these two cannot be checked
  without a GPU, which is a hole a reviewer opens first.

  THREE H200 CELLS, EACH WITH A DIFFERENT PREDICTION, from tile_resolve against
  the vLLM v0.27.1 snapshot:
    deepseek-v3 T=16    BM=16  warps=4   -> wgmma = 0, mma.sync > 0, "Using default"
    deepseek-v3 T=256   BM=64  warps=8   -> wgmma > 0, "Using default"
    mixtral     T=256   BM=128 warps=8   -> wgmma > 0, "Using configuration from"
  The third is the tuned specialisation (BM=128, BN=256) that FINDINGS names as
  never having been compiled to disk. Triton takes the warpgroup path only when
  BLOCK_M % 64 == 0 AND num_warps % 4 == 0, so the first cell declining it and
  the other two reaching it is one prediction, not three.

  ON THE A100 the prediction is different and simpler: getMMAVersionSafe returns
  {2} alone below compute capability 9.0, so NO tile reaches wgmma at any size.
  The A100 has never been dumped successfully, very likely because of the shared
  Triton cache that preflight P5 now tests. Both cards means two pods; run this
  step again on the A100 session and exfil both censuses.

  A cache hit dumps no PTX, so each cell gets its OWN dump directory:
  check_mma_path.sh begins with rm -rf on its --out, and three cells sharing one
  directory would leave only the last.
TXT
  if already_passed S5; then skipped S5 "ISA provenance" "already passed"; return 0; fi
  if [[ "$HAVE_GPU" == "0" || "$HAVE_VLLM" == "0" ]]; then
    skipped S5 "ISA provenance" "needs a GPU and vLLM"; return 0
  fi
  local census="$SESSION/exfil/ISA_CENSUS.txt"
  : > "$census"
  {
    echo "# what instruction each cell compiled to, and which config vLLM resolved"
    echo "# card: $GPU_NAME"
    echo "# generated by scripts/pod_session.sh step 5 on $(date -u +%FT%TZ)"
    echo
  } >> "$census"

  local ncell=0 nfail=0
  local cell
  for cell in "deepseek-v3 16 0 default" "deepseek-v3 256 1 default" "mixtral-8x7b 256 1 tuned"; do
    set -- $cell
    local model="$1" tok="$2" want_wgmma="$3" want_cfg="$4"
    local out="$SESSION/ptx/${model}-T${tok}"
    local clog="$SESSION/logs/mma-${model}-T${tok}.log"
    ncell=$((ncell + 1))
    head2 "cell: $model T=$tok  (expect wgmma $( [[ $want_wgmma == 1 ]] && echo '> 0' || echo '== 0' ), '$want_cfg' config)"
    run_logged "$clog" bash scripts/check_mma_path.sh --model "$model" --tokens "$tok" --out "$out"
    local rc=$?
    if [[ "$DRY_RUN" == "1" ]]; then continue; fi
    local got_w got_cfg
    got_w="$(grep -E 'kernels containing wgmma' "$clog" | tail -1 | awk '{print $NF}')"
    if grep -q 'Using configuration from' "$clog" 2>/dev/null; then got_cfg=tuned
    elif grep -q 'Using default MoE config' "$clog" 2>/dev/null; then got_cfg=default
    else got_cfg=unlogged; fi
    {
      echo "## $model T=$tok on $GPU_NAME"
      echo "resolved config : $got_cfg"
      echo "kernels with wgmma   : ${got_w:-?}"
      grep -A20 '=== exact instruction shapes seen ===' "$clog" 2>/dev/null | head -12
      echo
    } >> "$census"
    local ok=1
    if [[ "$want_wgmma" == "1" && "${got_w:-0}" -gt 0 ]]; then ok=0; fi
    if [[ "$want_wgmma" == "0" && "${got_w:-1}" == "0" ]]; then ok=0; fi
    verdict "S5-$model-$tok" "wgmma $model T=$tok" "$ok" \
      "${got_w:-?} kernel(s) with wgmma, config=$got_cfg" \
      "$( [[ $want_wgmma == 1 ]] && echo '> 0' || echo '== 0' ), config=$want_cfg" soft \
      "This is the measurement C3 rests on. A wgmma at T=16 refutes C3 and says the inference from BLOCK_SIZE_M=16 was wrong; no wgmma at BM=128 says Triton declines the warpgroup instruction even when the tile allows it, which is a finding in its own right. Log at $clog." \
      || nfail=$((nfail + 1))
    [[ "$got_cfg" == "$want_cfg" ]]; verdict "S5cfg-$model-$tok" "config source $model T=$tok" $? \
      "$got_cfg" "== $want_cfg" soft \
      "'unlogged' usually means this was not the first fused_experts call in the process: vLLM emits the line once per (E,N,dtype,device) via info_once. Every tile statement about the ten published v3 arms is DERIVED from vLLM's source, and this line is what would make it OBSERVED."
    [[ "$rc" == "0" ]] || note "check_mma_path.sh exited $rc for this cell"
  done

  # THE EXFIL SHAPE. *.ptx is gitignored at any depth on purpose, so the raw
  # dumps leave as a tarball and the human-readable census leaves as text. This
  # is the step whose output the repo has already lost once.
  if [[ "$DRY_RUN" != "1" ]]; then
    local devslug tarball
    devslug="$(echo "$GPU_NAME" | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-')"
    tarball="$SESSION/exfil/ptx-${devslug}.tar.gz"
    tar -czf "$tarball" -C "$SESSION" ptx 2>/dev/null
    local tarsize
    tarsize="$(wc -c < "$tarball" 2>/dev/null | tr -d ' ')"
    [[ "${tarsize:-0}" -gt 1000 ]]; verdict S5tar "PTX packaged for exfil" $? \
      "${tarsize:-0} bytes at $tarball" "> 1000 bytes" soft \
      "The PTX exists only on the pod. This repo has already lost every published figure to an unanchored gitignore rule and the A100 PTX to a shared Triton cache; an empty tarball here means C1 and C3 stay uncheckable without a GPU for another cycle."
  fi
  note "census: $census"
}

# --------------------------------------------------------------------------
step6_dense_grid() {
  head1 "STEP 6  (1:46)  dense uniform grid, tile pinned"
  cat <<'TXT'
  WHAT IT TESTS. The crossing itself, on the profile built for it. crossing-uniform
  is uniform-only (2R/b is a uniform-routing statement and pooling the seven
  regimes is INVALID, not merely noisy), 7 seeds, a 2^(1/4) grid from 1 to 16384
  over the ridge band, L2-warm eager because the cold basis loses 5 of 8
  one-stage crossings to throttle exclusion.

  WHY THE TILE MUST BE PINNED. Along an unpinned grid the tile CHANGES with the
  token count -- mixtral climbs 16, 32, 64, 128 across the sweep -- so the
  staircase the detector reads is partly the config ladder stepping and partly
  the roofline, and crossing_from_points returns the FIRST crossing, which is
  usually a tile step. Pinning removes the first mechanism, so what remains is
  attributable.

  PREDICTION at a pinned BLOCK_M=128, alpha 0.558, ridge 160.3: mixtral crosses
  near 999 tokens, qwen2 near 1998, deepseek-v3 near 7992. At BLOCK_M=64 the cap
  is 115 and no crossing exists at any batch on the grid.

  READ IT WITH octave_ladders. Fed whole to crossing_from_points this grid is
  biased 4-18% LOW and twice as wide as the powers-of-two grid it extends.
TXT
  if already_passed S6; then skipped S6 "dense uniform grid" "already passed"; return 0; fi
  if [[ "$HAVE_GPU" == "0" ]]; then skipped S6 "dense uniform grid" "$(absent_reason gpu)"; return 0; fi

  # Does a pinning hook exist at all? Asked of the OBSERVED v4 tile column rather
  # than assumed, because "the env var was set" and "the kernel ran that tile"
  # are different facts and only the second one matters. If nothing honours it
  # the grid still runs, unpinned, and says so.
  local pin_dir="$RESULTS_DIR/pin-probe" pinned=0
  if [[ "$DRY_RUN" != "1" && "$HAVE_VLLM" == "1" ]]; then
    MOE_FORCE_TILE='{"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":64,"GROUP_SIZE_M":1,"num_warps":8,"num_stages":4}' \
      "$PY_VLLM" -m moe.bench.cli --profile profile-cell --groups baselines \
      --out-dir "$pin_dir" > "$SESSION/logs/pin_probe.log" 2>&1
    local observed
    observed="$("$PY_BASE" - "$pin_dir" <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, ".")
from pathlib import Path
from moe.bench.schema import UNRECORDED, read_csv
vals = set()
for p in sorted(Path(sys.argv[1]).glob("run_*.csv")):
    for r in read_csv(p):
        v = r.get("tile_block_m")
        if v not in (None, "", UNRECORDED, "0"):
            vals.add(str(v))
print(",".join(sorted(vals)) or "none")
PYEOF
)"
    [[ "$observed" == "128" ]] && pinned=1
    [[ "$pinned" == "1" ]]; verdict S6a "tile pinning is honoured" $? \
      "observed tile_block_m = ${observed:-none}" "== 128, forced via MOE_FORCE_TILE" soft \
      "No pinning hook is wired into the sweep path, so the dense grid below runs on vLLM's own ladder. The crossing it produces is then a MIX of tile steps and the roofline transition, which is instrument defect 3, and it may not be quoted as a tile-pinned crossing. Run scripts/block_m_crossing_sweep.py (step 2) for the pinned answer instead."
    rm -rf "$pin_dir"
  fi

  local glog="$SESSION/logs/crossing_uniform.log"
  local run_id
  run_id="$(grep -m1 '^S6run' "$LEDGER" 2>/dev/null | cut -f4)"
  if [[ -z "$run_id" ]]; then
    run_id="$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')"
    ledger S6run "sweep run id" INFO "$run_id" "reused on resume" ""
  fi
  note "run id $run_id -- pass it to --run-id to resume; the manifest flushes per cell so an abort costs at most one cell"

  # The expected row count comes from the harness's own planner, so the gate
  # cannot drift from the profile. A sweep that produces 40% of its plan has
  # usually hit --max-minutes or an unsupported implementation, and both are
  # worth knowing before the plots are read.
  local planned
  planned="$("$PY_BASE" - <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, ".")
import moe
moe.bootstrap("reference", "kernels", "baselines")
from moe.bench import profiles as PR
print(PR.plan(PR.get("crossing-uniform"), env="base").timing_rows)
PYEOF
)"
  note "base env plans ${planned:-?} timing rows; vLLM and SGLang add their own"

  local envs="base"
  [[ "$HAVE_VLLM" == "1" ]] && envs="$envs,vllm"
  [[ "$HAVE_SGLANG" == "1" ]] && envs="$envs,sglang"
  run_logged "$glog" bash scripts/run_all.sh --profile crossing-uniform \
    --envs "$envs" --run-id "$run_id" --skip-setup --skip-tests --max-minutes 100
  local rc=$?
  [[ "$rc" == "0" ]]; verdict S6b "sweep exit" $? "exit $rc" "== 0" soft \
    "Resume with --from 6 and the same run id ($run_id); completed cells are skipped before the expensive fp32 oracle runs."

  if [[ "$DRY_RUN" != "1" ]]; then
    local sweepstat
    sweepstat="$("$PY_BASE" - "$RESULTS_DIR" "$run_id" <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, ".")
from pathlib import Path
from moe.bench.schema import passed, read_csv, row_bool
rows = []
for p in sorted(Path(sys.argv[1]).glob(f"run_{sys.argv[2]}*.csv")):
    rows.extend(read_csv(p))
timed = [r for r in rows if float(r.get("ms_p50") or 0) > 0]
bad = [r for r in rows if not passed(r)]
thr = [r for r in timed if row_bool(r, "throttled")]
pct = (100.0 * len(thr) / len(timed)) if timed else 100.0
print(f"{len(rows)} {len(timed)} {len(bad)} {pct:.1f}")
PYEOF
)"
    local nrows ntimed nbad pthr
    read -r nrows ntimed nbad pthr <<< "$sweepstat"
    note "$nrows rows, $ntimed timed, $nbad correctness failures, ${pthr}% throttled"
    [[ "${nbad:-1}" == "0" ]]; verdict S6c "correctness" $? \
      "${nbad:-?} failing rows" "== 0" fatal \
      "A correctness failure means the kernel computed the wrong layer, so every timing in this arm is a timing of the wrong thing. Do not publish it."
    "$PY_BASE" -c "import sys; sys.exit(0 if float('${pthr:-100}') < 5.0 else 1)"
    verdict S6d "thermal stability" $? \
      "${pthr:-?}% of timed rows throttled" "< 5%" soft \
      "Throttled rows are excluded from crossing detection, and the cold-L2 basis already loses 5 of 8 one-stage crossings that way. A high rate here narrows the grid the detector can actually use."
    [[ "${ntimed:-0}" -ge "${planned:-1}" ]]; verdict S6e "coverage against the plan" $? \
      "${ntimed:-0} timed rows against a base-env plan of ${planned:-?}" ">= the base plan" soft \
      "The sweep stopped short, almost certainly on --max-minutes. Resume with --from 6 and run id $run_id rather than reading a crossing off a truncated grid."
  fi
}

# --------------------------------------------------------------------------
step7_traces() {
  head1 "STEP 7  (2:40)  trace capture -- the axis this repo calls its differentiator"
  cat <<'TXT'
  WHAT IT FIXES. traces/ holds a single .gitkeep. Every routing distribution in
  this study is parametric, so every claim about realistic skew rests on zipf,
  hot and dirichlet standing in for measurements never taken, and
  scripts/capture_traces.py has never been run. Captures are kilobytes and they
  are committed on purpose; model weights never enter the repo.

  PREDICTION: mixtral-8x7b (93.4 GB), qwen2-57b-a14b (114.8 GB) and
  deepseek-v2-lite (31.4 GB) all fit on one 141 GB H200. deepseek-v3 does NOT
  (1369 GB, five cards), so its routing has never been captured and must keep
  being described as parametric geometry wherever it appears. Claiming a captured
  V3 trace would be false and is the kind of thing a reviewer checks first.

  This runs LAST because it is the only step that needs the download, and the
  download was started at 0:00 precisely so it is never on the critical path.
TXT
  if already_passed S7; then skipped S7 "trace capture" "already passed"; return 0; fi
  if [[ "$NO_DOWNLOAD" == "1" ]]; then skipped S7 "trace capture" "--no-download"; return 0; fi
  if [[ "$HAVE_GPU" == "0" ]]; then skipped S7 "trace capture" "$(absent_reason gpu)"; return 0; fi

  local dlog="$SESSION/logs/download.log"
  if [[ "$DRY_RUN" != "1" ]]; then
    grep -q 'DOWNLOAD-COMPLETE' "$dlog" 2>/dev/null
    verdict S7a "weights arrived before they were needed" $? \
      "$(tail -1 "$dlog" 2>/dev/null | cut -c1-70)" "log contains DOWNLOAD-COMPLETE" soft \
      "The 93 GB is still landing. Wait, or capture deepseek-v2-lite (31.4 GB, ungated) alone. Do not kill the pod: the volume keeps the partial download and it resumes."
  fi

  local model
  for model in mixtral-8x7b deepseek-v2-lite; do
    local tlog="$SESSION/logs/trace-$model.log"
    run_logged "$tlog" "$PY_BASE" scripts/capture_traces.py --model "$model" \
      --phase decode --corpus chat --out traces
    local rc=$?
    if [[ "$DRY_RUN" == "1" ]]; then continue; fi
    local n
    n="$(ls -1 traces/${model}-chat-decode*.npz 2>/dev/null | wc -l | tr -d ' ')"
    [[ "${n:-0}" -ge 1 && "$rc" == "0" ]]; verdict "S7-$model" "trace $model" $? \
      "${n:-0} .npz written, exit $rc" ">= 1 and exit 0" soft \
      "Skew claims for $model stay parametric. traces/ is committed on purpose, so a missing capture is a visible gap rather than a silent one."
  done
}

# --------------------------------------------------------------------------
#: THE TWO ALPHAS, SIDE BY SIDE, WITH THEIR INTERVALS.
#:
#: This session measures the study's central parameter TWICE by routes that share
#: almost nothing. Step 3 fits it by regression on implied_traffic_ratio, which is
#: time x bandwidth / compulsory bytes: it depends on the byte model, on the
#: calibrated bandwidth, and on the estimator. Step 2b measures it by ablation
#: from a difference of two timings: it depends on none of those.
#:
#: TWO INDEPENDENT MEASUREMENTS OF THAT NUMBER DISAGREEING IS THE MOST IMPORTANT
#: THING THIS SESSION COULD PRODUCE, so it is written to its own exfiled file, it
#: is a ledger row, and it is the LAST thing on the screen rather than a line
#: four hundred lines up a log. A session that buried it would be a session that
#: answered the question and did not tell anyone.
#:
#: THEY ARE NOT THE SAME KIND OF INTERVAL and the file says so: the regression
#: band is a cluster bootstrap over sampling noise, the ablation interval is a
#: BRACKET between two estimators biased in opposite directions by how L2 and HBM
#: service compose. Overlap is therefore weak evidence of agreement and disjoint
#: is strong evidence of disagreement, which is the asymmetry to read it with.
alpha_reconciliation() {
  head1 "ALPHA, MEASURED TWICE"
  local out="$SESSION/exfil/ALPHA_RECONCILIATION.txt"
  local block
  block="$("$PY_BASE" - "$SESSION/logs/group_m.log" "$SESSION/logs/alias_ablation.log" \
                      "$SESSION/alias_ablation/report.md" <<'PYEOF' 2>&1
import re
import sys
from pathlib import Path

# NO APOSTROPHES ANYWHERE IN THIS BLOCK. bash 3.2 tracks quotes through a heredoc
# inside a command substitution and one stray quote is a syntax error reported
# hundreds of lines away.
group_log, alias_log, alias_report = (Path(a) for a in sys.argv[1:4])

#: The refit as published, used only when this session did not produce its own
#: regression. Labelled as such wherever it is printed, because a number from
#: another session is exactly what the calibration guards exist to catch.
PUBLISHED = (0.558, 0.529, 0.588)


def text(path):
    try:
        return path.read_text()
    except OSError:
        return ""


def regression_alpha(blob):
    """alpha at GROUP_SIZE_M = 1 out of the sweep table, with its 90% band.

    Read by COLUMN NAME rather than by position: the table has grown a column
    twice already, and the second table in that report is keyed on GROUP_SIZE_M
    as well, so a positional read would silently return a timing ratio.
    """
    lines = blob.splitlines()
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split("|")]
        if "GROUP_SIZE_M" not in cells or "alpha" not in cells:
            continue
        try:
            a_col = cells.index("alpha")
            b_col = next(j for j, c in enumerate(cells) if c.startswith("90%"))
        except (ValueError, StopIteration):
            continue
        for row in lines[i + 1:]:
            if not row.strip().startswith("|"):
                break
            got = [c.strip() for c in row.split("|")]
            if len(got) <= max(a_col, b_col) or got[1] != "1":
                continue
            band = re.findall(r"[0-9]*\.?[0-9]+", got[b_col])
            try:
                alpha = float(got[a_col])
            except ValueError:
                return None
            if len(band) != 2:
                return None
            return alpha, float(band[0]), float(band[1])
    return None


def ablation_alpha(blob):
    """The pooled bracket and the 90% interval around it."""
    m = re.search(r"POOLED[^\n]*?alpha is in ([0-9.]+) to ([0-9.]+), "
                  r"90% interval ([0-9.]+) to ([0-9.]+)", blob)
    if not m:
        return None
    lo, hi, ilo, ihi = (float(g) for g in m.groups())
    return lo, hi, ilo, ihi


def verdict_line(blob, marker):
    for line in blob.splitlines():
        if line.startswith(marker):
            return line.strip()
    return ""


gblob = text(group_log)
ablob = text(alias_log) or text(alias_report)
reg = regression_alpha(gblob)
abl = ablation_alpha(ablob)

out = []
out.append("# the two estimates of alpha, the cost of an extra M-tile")
out.append("#")
out.append("# REGRESSION  fits alpha out of implied_traffic_ratio, which is")
out.append("#             time x bandwidth / COMPULSORY BYTES. It depends on the")
out.append("#             byte model, on the calibrated bandwidth and on the")
out.append("#             estimator. C4 is a confirmed finding that the")
out.append("#             compulsory-byte ruler was wrong by 1.85% until it was")
out.append("#             fixed, which is why one route is not enough.")
out.append("# ABLATION    reads alpha out of the difference between two timings")
out.append("#             of the same access pattern, one of which hits L2 where")
out.append("#             the other misses. No bytes, no bandwidth, no ridge.")
out.append("")

if reg is None:
    ra, rlo, rhi = PUBLISHED
    rsrc = ("the PUBLISHED refit over 10,813 rows -- step 3 produced no readable "
            "table in this session, so this number is NOT from this card")
else:
    ra, rlo, rhi = reg
    rsrc = "step 3, GROUP_SIZE_M = 1, measured in THIS session on THIS card"

out.append(f"REGRESSION  alpha = {ra:.3f}   90% band {rlo:.3f} .. {rhi:.3f}")
out.append(f"            source: {rsrc}")
out.append("")

state = "INCOMPLETE"
if abl is None:
    out.append("ABLATION    NOT AVAILABLE. Step 2b produced no pooled interval.")
    out.append("            The paper keeps its single-estimator caveat: alpha")
    out.append("            rests on one regression against a byte model with no")
    out.append("            tile term in it, and that is the thing step 2b exists")
    out.append("            to fix. Re-run with --only 2b before publishing.")
    out.append("")
    out.append("VERDICT     one estimate only. Nothing is reconciled.")
else:
    lo, hi, ilo, ihi = abl
    out.append(f"ABLATION    alpha in {lo:.3f} .. {hi:.3f}   "
               f"90% interval {ilo:.3f} .. {ihi:.3f}")
    out.append("            source: step 2b, same session, same thermal state")
    out.append("            a BRACKET, not a band: the two estimators inside it")
    out.append("            are biased in opposite directions by whether L2 and")
    out.append("            HBM service add or compose as a max.")
    out.append("")
    overlap = min(rhi, ihi) - max(rlo, ilo)
    if overlap >= 0:
        state = "AGREE"
        out.append(f"VERDICT     THEY AGREE. The intervals overlap over "
                   f"{overlap:.3f}.")
        out.append("            Read the asymmetry honestly: these are different")
        out.append("            kinds of interval, so overlap is weak evidence of")
        out.append("            agreement while disjoint would have been strong")
        out.append("            evidence against. What it buys is that alpha no")
        out.append("            longer rests on the byte model alone.")
    else:
        state = "DISAGREE"
        gap = -overlap
        out.append(f"VERDICT     THEY DISAGREE. The intervals are DISJOINT by "
                   f"{gap:.3f}.")
        out.append("            This is the most important thing in the session.")
        out.append("            The two routes share almost nothing, so at least")
        out.append("            one of them is measuring something other than the")
        out.append("            extra-tile cost. The tile-corrected roofline may")
        out.append("            NOT be published on either number until that is")
        out.append("            resolved. Check first: did the ablation ISA gate")
        out.append("            pass, and was the regression run in the")
        out.append("            multi-tile regime at all.")
    out.append("")
    for candidate, name in ((0.100, "this repo, retracted"),
                            (0.330, "TEMPO arXiv:2608.13057"),
                            (0.558, "the 2026-09-01 refit")):
        in_reg = "IN " if rlo <= candidate <= rhi else "out"
        in_abl = "IN " if ilo <= candidate <= ihi else "out"
        out.append(f"  alpha = {candidate:.3f} ({name}): "
                   f"regression says {in_reg}, ablation says {in_abl}")

for marker, blob in (("VERDICT:", ablob),):
    line = verdict_line(blob, marker)
    if line:
        out.append("")
        out.append("ablation verdict as the script itself stated it:")
        out.append("  " + line)

out.append("")
out.append(f"RECON {state}")
print("\n".join(out))
PYEOF
)"
  # RECON is the machine-readable marker the case below reads; it is stripped
  # from anything a human sees so the file does not end in a token that looks
  # like a result and is not one.
  printf '%s\n' "$block" | grep -v '^RECON ' | sed 's/^/  /'
  mkdir -p "$SESSION/exfil"
  printf '%s\n' "$block" > "$out"
  local state
  state="$(printf '%s\n' "$block" | awk '/^RECON /{print $2}' | tail -1)"
  case "${state:-INCOMPLETE}" in
    AGREE)
      verdict SRa "the two alphas agree" 0 "regression and ablation intervals overlap" \
        "overlapping intervals" soft "" ;;
    DISAGREE)
      false; verdict SRa "the two alphas agree" $? \
        "regression and ablation intervals are DISJOINT" "overlapping intervals" soft \
        "STOP AND READ $out BEFORE PUBLISHING ANY alpha. Two routes that share no byte model, no bandwidth and no estimator disagree about the study's central parameter. Neither number may carry the tile-corrected roofline until this is resolved." ;;
    *)
      skipped SRa "the two alphas agree" "only one estimate exists; nothing to reconcile" ;;
  esac
  note "written to $out, and repeated in the summary below"
}

# --------------------------------------------------------------------------
step8_exfil() {
  head1 "STEP 8  (3:20)  EXFIL -- nothing is torn down before this passes"
  cat <<'TXT'
  THIS REPO HAS LOST WORK TWICE AT EXACTLY THIS POINT. Every published figure
  was dropped by an unanchored `plots/` rule that git add applied silently, and
  the A100 PTX was never produced because a shared Triton cache meant nothing
  recompiled. Both losses were invisible at the terminal. So this step does not
  copy files and hope: it lists what must exist, checks each one, checks that git
  will actually accept it, and refuses to say the session is done otherwise.

  WHAT MUST LEAVE THE POD:
    1. the sweep CSVs and manifests, via publish_results.sh, with the calibration
       they were measured against copied in beside them
    2. that calibration's checksum, unchanged since step 1 wrote it
    3. ISA_CENSUS.txt and ptx-<card>.tar.gz from step 5 (raw .ptx is gitignored
       at any depth on purpose, so it leaves as a tarball)
    4. traces/*.npz from step 7
    5. every step log, so a number can be traced to the run that made it
    6. LEDGER.tsv, which is the machine-readable record of every gate above
    7. the alias ablation report and its per-cell measurements, which are the
       ONLY estimate of alpha in this session that does not go through the byte
       model, and therefore the only thing that can corroborate or refute it
    8. TRAFFIC_PROVENANCE.txt and the nsys probe report, which say whether any
       byte figure here was counted or modelled
    9. ALPHA_RECONCILIATION.txt, the two estimates side by side
TXT
  alpha_reconciliation
  local missing=0 present=0 item
  head2 "the manifest"
  # AN ARTEFACT IS EXPECTED ONLY IF ITS PRODUCING STEP CLAIMED SUCCESS. Listing
  # everything unconditionally would report a skipped step as a lost file, and a
  # gate that cries wolf on a laptop is a gate nobody reads on the pod. So GONE
  # here means exactly one thing: the step said it worked and the file is not
  # there, which is the loss this project has twice suffered.
  local want="$SESSION/exfil/EXPECTED.txt"
  {
    echo "$LEDGER"
    ls -1 "$SESSION"/logs/*.log 2>/dev/null
    if ledger_passed S1a; then
      echo "$SESSION/calibration/measured.sha256"
      ls -1 "$SESSION"/calibration/*.yaml 2>/dev/null
    fi
    if ledger_passed S5tar; then
      echo "$SESSION/exfil/ISA_CENSUS.txt"
      ls -1 "$SESSION"/exfil/ptx-*.tar.gz 2>/dev/null
    fi
    if ledger_passed 'S7-'; then
      ls -1 traces/*.npz 2>/dev/null
    fi
    # ADDED 2026-09-01. The first version of this manifest listed the calibration,
    # the PTX tarball and the traces, and omitted the outputs of steps 2, 3 and 4
    # -- which is the entire scientific payload of the session. A pod torn down
    # after a clean run would have taken the answer with it. This project has
    # already lost every published figure to an unanchored gitignore and its A100
    # PTX to a shared Triton cache; those were the same mistake.
    if ledger_passed S2; then
      ls -1 "$SESSION"/block_m/* 2>/dev/null
    fi
    if ledger_passed S3; then
      ls -1 "$SESSION"/group_m/* 2>/dev/null
    fi
    if ledger_passed S4; then
      ls -1 "$SESSION"/tuned_vs_fallback/* 2>/dev/null
    fi
    if ledger_passed S6; then
      ls -1 "$SESSION"/crossing_uniform/* 2>/dev/null
    fi
    # ADDED WITH STEP 2b AND P-nsys. The manifest omitted the entire scientific
    # payload once already today, and both of these are payload: the ablation is
    # the only alpha in this session that is independent of the byte model, and
    # the provenance file is the difference between a counted byte and a modelled
    # one. Both are cheap and both are irreplaceable after teardown.
    if ledger_passed SB; then
      ls -1 "$SESSION"/alias_ablation/* 2>/dev/null
    fi
    if ledger_passed SN; then
      ls -1 "$SESSION"/nsys/cell/probe.json 2>/dev/null
    fi
    # UNCONDITIONAL, unlike everything above it. These two are written by the
    # session itself rather than by a step, so there is no step whose PASS could
    # gate them, and their whole point is to be present even when a step did not
    # run: "traffic was INFERRED because nsys is not installed" and "only one
    # alpha exists" are results that must survive the pod.
    [[ -s "$SESSION/exfil/TRAFFIC_PROVENANCE.txt" ]] && echo "$SESSION/exfil/TRAFFIC_PROVENANCE.txt"
    [[ -s "$SESSION/exfil/ALPHA_RECONCILIATION.txt" ]] && echo "$SESSION/exfil/ALPHA_RECONCILIATION.txt"
    ls -1 "$SESSION"/nsys/probe.json 2>/dev/null
  } | grep -v '^$' | sort -u > "$want"
  note "$(wc -l < "$want" | tr -d ' ') artefact(s) expected: the steps that reported success, plus the provenance and reconciliation files the session writes whatever ran"
  while IFS= read -r item; do
    if [[ -s "$item" ]]; then
      present=$((present + 1))
      printf '    ok    %s (%s bytes)\n' "$item" "$(wc -c < "$item" | tr -d ' ')"
    else
      missing=$((missing + 1))
      printf '    GONE  %s\n' "$item"
    fi
  done < "$want"
  [[ "$missing" == "0" && "$present" -ge 1 ]]; verdict S8a "artefacts exist" $? \
    "$present present, $missing missing or empty" "0 missing, at least 1 present" soft \
    "A step produced nothing. Check its log before tearing down; re-running one step is minutes and re-renting the pod is an hour."

  head2 "the ruler did not move under the results"
  # On 2026-08-28 a recalibration overwrote measured_<device>.yaml between a
  # sweep finishing at 19:21 and its publish at 19:35. The two rulers disagree by
  # 9.9% on the compute ceiling, nothing downstream could tell, and claim C5 lost
  # its target for three days.
  if [[ -s "$SESSION/calibration/measured.sha256" ]]; then
    local sumfile now_sum then_sum name
    sumfile="$SESSION/calibration/measured.sha256"
    then_sum="$(awk '{print $1}' "$sumfile")"
    name="$(awk '{print $2}' "$sumfile")"
    now_sum="$( (cd moe/bench/hardware && shasum -a 256 "$name" 2>/dev/null || sha256sum "$name" 2>/dev/null) | awk '{print $1}')"
    [[ -n "$then_sum" && "$then_sum" == "$now_sum" ]]; verdict S8b "calibration unchanged since step 1" $? \
      "${then_sum:0:12} vs ${now_sum:0:12}" "identical" fatal \
      "moe/bench/hardware/$name changed after step 1, so publish_results.sh would copy a ruler these rows were never measured against. Do not publish. Restore $SESSION/calibration/$name first."
  elif ledger_passed S1a; then
    # Step 1 said it calibrated, so the snapshot MUST be there. Its absence is
    # the loss this gate exists to catch, not a reason to skip the gate.
    false; verdict S8b "calibration unchanged since step 1" $? \
      "no $SESSION/calibration/measured.sha256" "the snapshot step 1 took" fatal \
      "Step 1 recorded a PASS but its snapshot is gone, so nothing can prove the ruler under these rows is the one they were measured against. Re-run step 1, or do not publish this arm."
  else
    skipped S8b "calibration unchanged" "step 1 did not run"
  fi

  head2 "git will actually accept every one of these"
  # The check that would have caught the lost figures. `git add` reports nothing
  # when a rule swallows a file, so the rule is asked directly.
  local ignored=0
  while IFS= read -r item; do
    case "$item" in "$SESSION"/*) continue ;; esac   # session dir is outside the repo
    if git check-ignore -q "$item" 2>/dev/null; then
      ignored=$((ignored + 1))
      printf '    IGNORED  %s\n' "$(git check-ignore -v "$item" 2>/dev/null | tr '\t' ' ')"
    fi
  done < "$want"
  [[ "$ignored" == "0" ]]; verdict S8c "nothing is silently gitignored" $? \
    "$ignored ignored" "== 0" fatal \
    "git add would drop these without a word, which is how zero .png files came to be tracked under results/published/ across all ten arms. Rename the artefact or fix the rule before tearing down."

  head2 "publish"
  # Publishing is a DECISION, not a side effect: results/ is gitignored and
  # results/published/ is tracked precisely so that a run enters git only when
  # somebody chose it. So this refuses to publish a session that measured
  # nothing, which is also what stops an off-GPU rehearsal from committing to the
  # user's repo.
  if [[ "$DRY_RUN" == "1" ]]; then
    say "  \$ bash scripts/publish_results.sh --label $LABEL"
    skipped S8d "publish" "dry run"
  elif ! measured_anything; then
    skipped S8d "publish" "no step produced a measurement; there is nothing to publish"
  else
    local plog="$SESSION/logs/publish.log" pub_id
    # NAME THE RUN. publish_results.sh refuses to guess when several runs are
    # present, and a pod that has been used before has several: a sweep is one
    # experiment but each venv writes its own run_<id>_<env>.csv, and publishing
    # the newest CSV once published a third of a three-way and looked complete.
    pub_id="$(grep -m1 "^S6run" "$LEDGER" 2>/dev/null | cut -f4)"
    if [[ -n "$pub_id" ]]; then
      run_logged "$plog" bash scripts/publish_results.sh --label "$LABEL" --run-id "$pub_id"
    else
      run_logged "$plog" bash scripts/publish_results.sh --label "$LABEL"
    fi
    local rc=$?
    [[ "$rc" == "0" ]]; verdict S8d "publish_results" $? "exit $rc" "== 0" soft \
      "The commit may still exist locally even when the push failed: a public clone over HTTPS can pull but not push. Run gh auth login, or copy the session off with runpodctl send $SESSION."
    # Copy the session's own artefacts into the arm publish_results just made, so
    # the census, the logs and the ledger travel with the rows they explain.
    local arm
    arm="$(ls -1dt results/published/*/ 2>/dev/null | head -1)"
    if [[ -n "$arm" ]]; then
      mkdir -p "$arm/session"
      cp -f "$LEDGER" "$arm/session/" 2>/dev/null
      cp -f "$SESSION"/exfil/ISA_CENSUS.txt "$arm/session/" 2>/dev/null
      cp -f "$SESSION"/exfil/ptx-*.tar.gz "$arm/session/" 2>/dev/null
      cp -f "$SESSION"/logs/*.log "$arm/session/" 2>/dev/null
      cp -f "$SESSION"/calibration/measured.sha256 "$arm/session/" 2>/dev/null
      # The two files that say how much the rows beside them can be trusted, and
      # the ablation that is the only check on the number they are read against.
      cp -f "$SESSION"/exfil/TRAFFIC_PROVENANCE.txt "$arm/session/" 2>/dev/null
      cp -f "$SESSION"/exfil/ALPHA_RECONCILIATION.txt "$arm/session/" 2>/dev/null
      cp -f "$SESSION"/alias_ablation/report.md "$arm/session/alias_ablation.md" 2>/dev/null
      cp -f "$SESSION"/alias_ablation/cells.jsonl "$arm/session/alias_cells.jsonl" 2>/dev/null
      cp -f "$SESSION"/nsys/probe.json "$arm/session/nsys_probe.json" 2>/dev/null
      cp -f "$SESSION"/nsys/cell/probe.json "$arm/session/nsys_cell.json" 2>/dev/null
      git add "$arm/session" traces >/dev/null 2>&1
      git commit -q -m "session artefacts for $(basename "$arm")" >/dev/null 2>&1
      local tracked
      tracked="$(git ls-files "$arm/session" | wc -l | tr -d ' ')"
      [[ "${tracked:-0}" -ge 1 ]]; verdict S8e "session artefacts tracked in git" $? \
        "${tracked:-0} file(s) tracked under $arm/session" ">= 1" soft \
        "The census and the logs exist on the pod and nowhere else. They are what makes C1 and C3 checkable without a GPU."
    fi

    head2 "is it off the pod"
    local local_head remote_head branch
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    local_head="$(git rev-parse HEAD 2>/dev/null)"
    remote_head="$(git ls-remote origin "refs/heads/$branch" 2>/dev/null | awk '{print $1}')"
    [[ -n "$remote_head" && "$local_head" == "$remote_head" ]]; verdict S8f "pushed to the remote" $? \
      "local ${local_head:0:12}, remote ${remote_head:0:12}" "identical" soft \
      "The commit is local only and dies with the container filesystem. It does NOT die with the pod if the repo is cloned into the volume, but do not rely on that: push it, or 'runpodctl send $SESSION' now."
  fi

  head2 "checksums, so a copy can be verified after teardown"
  if [[ "$DRY_RUN" == "1" ]]; then
    skipped S8g "manifest" "dry run"
  else
    ( cd "$SESSION" && find . -type f ! -name MANIFEST.sha256 -print0 \
      | xargs -0 shasum -a 256 2>/dev/null || true ) > "$SESSION/MANIFEST.sha256"
    local nsum
    nsum="$(wc -l < "$SESSION/MANIFEST.sha256" | tr -d ' ')"
    [[ "${nsum:-0}" -ge 1 ]]; verdict S8g "manifest written" $? \
      "${nsum:-0} files checksummed" ">= 1" soft \
      "Without it a truncated copy off the pod is indistinguishable from a complete one."
  fi
}

# ==========================================================================
# main
# ==========================================================================
main() {
  head1 "MoE pod session  $STAMP  label=$LABEL"
  probe_capabilities
  if [[ "$DRY_RUN" == "1" ]]; then
    say "DRY RUN. Commands are printed, not executed. Preflight still runs for real,"
    say "because that is the half worth rehearsing on a laptop."
  fi

  preflight
  if halted; then
    head1 "STOPPED IN PRE-FLIGHT: $STOPPED"
    say "Nothing was spent. Fix the gate above and re-run; --preflight-only re-checks in about 5 minutes."
    summarise; return 2
  fi
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then summarise; return $?; fi

  SPENT=1
  wanted 0 && step0_download
  wanted 1 && { step1_calibrate; halted && { summarise; return 2; }; }
  # step2_nsys_cell is called BESIDE step2_block_m rather than from inside it,
  # so a resume that skips an already-passed sweep still gets its traffic
  # measurement. It carries its own idempotence on the SN ledger rows.
  wanted 2 && { step2_block_m; step2_nsys_cell; }
  # 2b, not 3: both estimate alpha and they must share one thermal state.
  wanted 2b && step2b_alias_ablation
  wanted 3 && step3_group_m
  wanted 4 && step4_tuned_vs_fallback
  wanted 5 && step5_isa
  wanted 6 && { step6_dense_grid; halted && { summarise; return 2; }; }
  wanted 7 && step7_traces
  wanted 8 && step8_exfil
  summarise
}

summarise() {
  # THE ANSWER GOES ABOVE THE SCOREBOARD. A session that measured the study's
  # central parameter twice and printed the disagreement four hundred lines up a
  # log would be a session that answered the question and told nobody, so the
  # reconciliation and the traffic provenance are the first things here and the
  # gate counts follow them.
  if [[ -s "$SESSION/exfil/ALPHA_RECONCILIATION.txt" ]]; then
    head1 "ALPHA, THE TWO ESTIMATES"
    sed 's/^/  /' "$SESSION/exfil/ALPHA_RECONCILIATION.txt" | grep -v '^  RECON '
  fi

  head1 "SUMMARY"
  say "  $(provenance_line)"
  say ""
  printf '  %-6s %s\n' "PASS" "$N_PASS"
  printf '  %-6s %s  (session continued; each names what it invalidates)\n' "FAIL" "$N_FAIL_SOFT"
  printf '  %-6s %s  (session stopped)\n' "FATAL" "$N_FAIL_FATAL"
  printf '  %-6s %s\n' "SKIP" "$N_SKIP"
  say ""
  say "  ledger    $LEDGER"
  say "  session   $SESSION"
  say "  logs      $SESSION/logs"
  say "  exfil     $SESSION/exfil"
  say "  alpha     $SESSION/exfil/ALPHA_RECONCILIATION.txt"
  say "  traffic   $SESSION/exfil/TRAFFIC_PROVENANCE.txt"
  say ""
  if [[ "$N_FAIL_FATAL" -gt 0 && "$SPENT" == "1" ]]; then
    say "  DO NOT TEAR DOWN. A fatal gate failed at: $STOPPED, and steps had already run."
    say "  Everything produced so far is under $SESSION, which is on the volume."
    return 2
  fi
  if [[ "$N_FAIL_FATAL" -gt 0 ]]; then
    say "  Stopped at: $STOPPED. Nothing was spent and nothing needs saving."
    return 2
  fi
  if [[ "$DRY_RUN" != "1" && "$HAVE_GPU" == "1" ]]; then
    say "  Before you terminate the pod, confirm step 8 says PASS on S8a, S8c and S8b."
    say "  Everything under $SESSION is on the Network Volume and survives termination;"
    say "  everything under / does not."
  fi
  [[ "$N_FAIL_SOFT" -gt 0 ]] && return 1
  return 0
}

main
exit $?
