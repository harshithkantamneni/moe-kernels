#!/usr/bin/env bash
# The alpha surface, as one unattended run.
#
# WHAT IT IS FOR. The 2026-09-01 session measured alpha four ways and got
# 0.92-1.02 at GROUP_SIZE_M=1 on mixtral, 0.58-0.62 at G>=8, and 0.71 on qwen2 at
# G=1. Those are not noise around a scalar, they are a surface over the two things
# that set REUSE DISTANCE -- how many M-tiles share one weight read (the swizzle)
# and how much else evicts it in between (the footprint). This measures that
# surface with the estimator that works: the memory-bound tread fit, which came in
# under 1% error on qwen2 and under 1.6% on mixtral.
#
# NOTE WHAT IS ALREADY REFUTED. "Does one expert fit in L2" is NOT the predictor.
# qwen2's expert is 37 MB against 60 MiB of L2 and still pays alpha 0.71.
#
# EVERY ARM IS INDEPENDENT AND NONE IS FATAL. One failing arm records its status
# and the rest continue, because a 35-minute unattended run that aborts at minute
# four on a shape that does not compile is worse than no automation at all.
set -uo pipefail          # NOT -e: an arm may fail and the run must continue

REPO="${REPO:-/workspace/repo}"
SESSION="${SESSION:-$(ls -1dt /workspace/session/*/ 2>/dev/null | head -1)}"
SESSION="${SESSION%/}"
[[ -n "$SESSION" && -d "$SESSION" ]] || { echo "no session dir; set SESSION="; exit 3; }
PY_VLLM="${PY_VLLM:-/workspace/venvs/vllm/bin/python}"
PY_BASE="${PY_BASE:-/workspace/venvs/base/bin/python}"
OUT="$SESSION/alpha_surface"
LOGS="$SESSION/logs"
LEDGER="$OUT/ARMS.tsv"
mkdir -p "$OUT" "$LOGS" || exit 3
[[ -f "$LEDGER" ]] || printf 'arm\tstatus\tseconds\tlog\n' > "$LEDGER"

say() { printf '\n==== %s ====\n' "$*"; }
note() { printf '  %s\n' "$*"; }

started=$(date -u +%s)

# --------------------------------------------------------------------------
# One arm. Skips itself if it already passed, so a re-run resumes rather than
# repeating 35 minutes of GPU time.
# --------------------------------------------------------------------------
arm() {
  local name="$1"; shift
  if grep -qP "^\Q$name\E\tPASS\t" "$LEDGER" 2>/dev/null; then
    note "SKIP $name (already PASS in $LEDGER)"; return 0
  fi
  local log="$LOGS/as_$name.log" t0 t1 rc
  note "-> $name   log $log"
  t0=$(date -u +%s)
  "$@" > "$log" 2>&1
  rc=$?
  t1=$(date -u +%s)
  local status=PASS
  [[ $rc -eq 0 ]] || status="EXIT$rc"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$((t1 - t0))" "$log" >> "$LEDGER"
  note "   $status in $((t1 - t0))s"
  # An arm that produced a report is useful even when its gates failed, and the
  # sweep exits non-zero on a failed gate BY DESIGN -- a refutation is a result.
  return 0
}

sweep() {   # sweep <name> <model> [extra args...]
  local name="$1" model="$2"; shift 2
  arm "$name" "$PY_VLLM" "$REPO/scripts/block_m_crossing_sweep.py" \
      --model "$model" --r-max 1024 --out "$OUT" "$@"
}

say "ALPHA SURFACE  session=$SESSION"
note "out    $OUT"
note "ledger $LEDGER"
note "arms are independent; a failure records its status and the run continues"

# --------------------------------------------------------------------------
# nsys, first, because it is the only arm whose result changes how the others
# are READ. P-nsys reported "no DRAM metric on this device" and that was NOT a
# statement about the H200: the importer binary is missing, so nsys cannot write
# a report at all here. Until that is fixed the DRAM question is OPEN, not
# closed, and docs/POD_RUNBOOK.md is wrong to say otherwise.
# --------------------------------------------------------------------------
say "0. is nsys repairable on this pod (cheap, and expected to fail)"
# MEASURED 2026-09-01 AND IT FAILED TWICE. `apt-get install nsight-systems-cli`
# reports "Unable to locate package", and the installed nsys captures a .qdstrm
# but cannot convert it: "The importer binary and its dependencies were not
# found", so --export=sqlite writes nothing and the kernel table comes back [].
#
# THIS MATTERS FOR HOW P-nsys IS READ. Pre-flight reported "no invocation of this
# nsys sampled a DRAM metric on this device" and the session took that as closing
# the question. It does not. nsys cannot write a report here AT ALL, so that line
# is a fact about a broken install and not about the H200. The DRAM question, and
# the occupancy-from-kernel-records question behind gate 2, both stay OPEN.
#
# One last cheap look for the importer, which ships separately from the CLI and
# is sometimes sitting in the CUDA tree. Two minutes, then the run moves on.
arm nsys_importer_hunt bash -c '
  echo "-- candidate importers --"
  find /opt /usr/local /usr/lib -name "QdstrmImporter*" -o -name "*qdstrm*import*" 2>/dev/null | head
  echo "-- nsys binaries --"
  ls -1 /opt/nvidia/nsight-systems/*/target-linux-x64/ 2>/dev/null | head -20
  echo "-- what apt does offer --"
  apt-cache search nsight 2>/dev/null | head
  echo
  echo "If nothing above names an importer, nsys is unusable on this pod and the"
  echo "DRAM and occupancy questions stay OPEN rather than answered. The .qdstrm"
  echo "files are still valid captures and can be imported on another machine."'

# --------------------------------------------------------------------------
# The swizzle axis. qwen2 FIRST and it is not a preference: GROUP_SIZE_M
# SATURATES once g reaches a rung's own num_pid_m, and past that every larger g
# is the same tile order. On mixtral num_pid_m is 4 at the small rungs, so
# G=8,16,32,64 collapse into ONE setting and the ladder cannot resolve them --
# which is exactly why step 3 identified none of its five. qwen2's E=64,k=8 puts
# num_pid_m near 550, so nothing saturates and the ladder is real.
# --------------------------------------------------------------------------
say "1. the swizzle axis on qwen2, where GROUP_SIZE_M does not saturate"
for g in 1 8 16 64; do sweep "qwen2_g$g" qwen2-57b-a14b --group-m "$g"; done

say "2. the swizzle axis on mixtral, the 235 MB/expert row of the surface"
note "G>=8 will partly saturate here; read it against arm 1, not on its own"
for g in 1 8 16 64; do sweep "mixtral_g$g" mixtral-8x7b --group-m "$g"; done

# --------------------------------------------------------------------------
# The footprint axis. deepseek-v2-lite is the third point, ~11 MB/expert against
# qwen2's 37 and mixtral's 235, with 60 MiB of L2 sitting between the second and
# third. Three points bracket the knee; two only give a direction.
# --------------------------------------------------------------------------
say "3. the footprint axis: a third model, ~11 MB per expert"
sweep v2lite_g1 deepseek-v2-lite
sweep v2lite_g16 deepseek-v2-lite --group-m 16

# --------------------------------------------------------------------------
# The activation confound. An extra M-tile re-reads ACTIVATIONS too, in the ratio
# BLOCK_M/BLOCK_N. This study quotes 0.25 for that, from BLOCK_M=16 over
# BLOCK_N=64 -- but the sweep pins BLOCK_N=64 while running BLOCK_M to 256, where
# the ratio is 4.0. So alpha above is NOT bounded the way the text says. At
# BLOCK_N=256 the ratio at BLOCK_M=64 is back to 0.25: if alpha holds, the
# weight-traffic reading survives; if it drops, part of 0.92-0.99 was activations
# and the headline needs rewriting.
# --------------------------------------------------------------------------
say "4. the activation confound, which is the cheapest way to lose all of this"
sweep qwen2_bn256 qwen2-57b-a14b --block-n 256
sweep mixtral_bn256 mixtral-8x7b --block-n 256

# --------------------------------------------------------------------------
say "THE SURFACE"
"$PY_BASE" "$REPO/scripts/alpha_surface.py" "$OUT" 2>&1 | tee "$OUT/SURFACE.txt"

say "ARMS"
cat "$LEDGER"
printf '\ntotal %s min\n' "$(( ($(date -u +%s) - started) / 60 ))"
printf 'surface  %s\n' "$OUT/SURFACE.txt"
printf 'reports  %s\n' "$OUT"
printf '\nNothing here is published. Commit and push from %s when you have read it.\n' "$REPO"
