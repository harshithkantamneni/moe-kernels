#!/usr/bin/env bash
# C3: does vLLM's decode path use Hopper's warpgroup tensor core, or not?
#
#   bash scripts/check_mma_path.sh
#   bash scripts/check_mma_path.sh --tokens 16 --model deepseek-v3
#
# THE QUESTION. Hopper's `wgmma.mma_async.m64nNk16` has M fixed at 64 by the
# instruction set, and Triton selects it only when BLOCK_M % 64 == 0 AND
# num_warps % 4 == 0 (supportMMA, the version==3 branch, triton release/3.7.x
# lib/Analysis/Utility.cpp). Not "BLOCK_M >= 64": 80 or 96 fail the modulo and
# compile to `mma.sync` anyway. And below compute capability 9.0
# getMMAVersionSafe returns {2} alone, so on the A100 arm no tile reaches the
# warpgroup instruction at any size and this script has nothing to ask there.
#
# WHICH TILE ACTUALLY RUNS. An earlier version of this header asserted that
# vLLM's tuned H200 config sets BLOCK_SIZE_M to 16 for every batch size from 1
# to 256. That ladder is real but belongs to E=128,N=512, which is no model in
# this study, and the error went unchallenged for days because the published
# CSVs carry no column recording the tile the kernel actually ran. The two study
# shapes that DO ship a tuned bf16 H200 file (vLLM v0.27.1, under
# vllm/model_executor/layers/fused_moe/configs/) both climb well before 256:
#
#   E=8,N=14336,device_name=NVIDIA_H200.json    mixtral-8x7b
#     1:16 2:32 4:16 8:16 16:16 24:16 32:16 48:32 64:32 96:32
#     128:64 256:128 512:128 1024:128 1536:128 2048:128 3072:128 4096:128
#   E=64,N=2560,device_name=NVIDIA_H200.json    qwen2-57b-a14b
#     1:16 2:16 4:16 8:16 16:16 24:16 32:16 48:16 64:16 96:32
#     128:32 256:64 512:128 1024:128 1536:128 2048:128 3072:128 4096:128
#
# Those are 2 of the 8 (model x card) cells in this study. NOTHING ships for
# NVIDIA_A100-SXM4-80GB at E=8,N=14336 / E=64,N=2560 / E=64,N=1408 /
# E=256,N=2048, and nothing ships for H200 at the last two either, so the other
# six cells take the hardcoded bf16 ladder in `get_default_config`:
# M<=32 -> 16, M<=96 -> 32, M<=512 -> 64, else 128. deepseek-v3 (E=256,N=2048),
# the default --model below, is one of the six: the 16 it runs at --tokens 16 is
# that fallback and not a grid-search optimum, and vLLM says so on the run log
# with "Using default MoE config. Performance might be sub-optimal!".
#
# Two more traps in that lookup. The key is chosen by NEAREST, not floor
# (`configs[min(configs.keys(), key=lambda x: abs(x - M))]`), so M=200 reads the
# 256 entry; and M is the rows entering the layer, not tokens x top_k.
# Separately, the fp8_w8a8 files for those same two H200 shapes sit at
# BLOCK_SIZE_M 64 with num_warps 4 from M=1 upward, which passes the predicate,
# so "vLLM declines the warpgroup instruction at decode" is a claim about the
# bf16 path only.
#
# Before this script the study recorded the instruction as an INFERENCE, not a
# measurement. This settles it: Triton writes its generated PTX to disk when
# asked, CUTLASS and PTX both carry the instruction name, and the answer is a
# grep.
#
# There is nothing to interpret. The instruction is in the file or it is not.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKSPACE="${WORKSPACE:-/workspace}"
DUMP_DIR="${MOE_PTX_DIR:-$WORKSPACE/ptx}"
MODEL="deepseek-v3"
TOKENS="16"
ENV_NAME="vllm"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokens) TOKENS="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --env)    ENV_NAME="$2"; shift 2 ;;
    --out)    DUMP_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PY="${MOE_PYTHON:-$WORKSPACE/venvs/$ENV_NAME/bin/python}"
[[ -x "$PY" ]] || { echo "no interpreter at $PY; run setup_runpod.sh" >&2; exit 1; }

log() { printf '[mma] %s\n' "$*"; }

#: Dropped inside a dump directory so a later run can recognise its own output.
DUMP_MARKER=".moe-ptx-dump"

# DUMP_DIR arrives from MOE_PTX_DIR or --out and is handed straight to `rm -rf`,
# so it is checked before it is deleted. MOE_PTX_DIR="" collapses the default,
# `--out` with a missing value swallows the next flag, and MOE_PTX_DIR=$HOME on a
# rented pod is one keystroke from a path whose recursive removal cannot be
# undone. A blacklist alone is not enough, so an EXISTING directory must also be
# empty or carry this script's own marker: that is what "clearly a dump
# directory" means here, and nothing else is deleted.
refuse() { echo "[mma] refusing to rm -rf '$DUMP_DIR': $*" >&2; exit 2; }
case "$DUMP_DIR" in
  "") refuse "empty path" ;;
  /*) : ;;
  *)  refuse "not an absolute path" ;;
esac
case "${DUMP_DIR%/}" in
  ""|"/"|"/root"|"/home"|"/tmp"|"/usr"|"/etc"|"/var"|"/workspace"|"$HOME"|"$WORKSPACE"|"$REPO_ROOT")
    refuse "that is a root, home, workspace or repo directory" ;;
esac
if [[ -e "$DUMP_DIR" ]]; then
  [[ -d "$DUMP_DIR" ]] || refuse "it exists and is not a directory"
  if [[ ! -e "$DUMP_DIR/$DUMP_MARKER" && -n "$(ls -A "$DUMP_DIR")" ]]; then
    refuse "it is not empty and carries no $DUMP_MARKER, so this script did not write it"
  fi
fi

DUMP_DIR="${DUMP_DIR%/}"   # so the -prune path below matches find's output exactly

rm -rf "$DUMP_DIR"
mkdir -p "$DUMP_DIR"
: > "$DUMP_DIR/$DUMP_MARKER"
log "dumping Triton IR and PTX to $DUMP_DIR"

# A CACHE HIT DUMPS NO PTX, and that is the failure this line exists to prevent.
# TRITON_KERNEL_DUMP asks for the dump, but Triton does not recompile a kernel it
# has already built, and setup_runpod.sh exports a SHARED
# TRITON_CACHE_DIR=$WORKSPACE/triton-cache that every earlier sweep on the pod has
# already populated with this exact fused_moe specialisation. Inheriting it means
# nothing compiles, nothing is written, and the script exits 1 reporting that the
# kernel never compiled -- which is very likely why the A100 was never
# successfully dumped. A per-run cache forces every specialisation to be built,
# exactly as scripts/tile_sweep.py arm_ptx_dump does.
CACHE_DIR="$DUMP_DIR/_cache"

# TRITON_KERNEL_DUMP=1 writes every compilation stage; TRITON_DUMP_DIR says where.
# A smoke-sized cell is enough: the instruction selection does not depend on how
# many times the kernel runs, only on the tile shape it was compiled for.
#
# TEE'd rather than piped to `tail -5`, and with no `|| true`. The discarded
# lines were the evidence: vLLM logs "Using configuration from FILE for MoE
# layer." on a tuned hit and "Using default MoE config. Performance might be
# sub-optimal!" on a miss, and that single line is what turns every tile
# statement in FINDINGS C3 and C5 from DERIVED into OBSERVED. `|| true` also
# meant a sweep that died -- OOM, missing vLLM, a bad --model -- still reached
# the PTX scan and reported on whatever stale files were lying around.
LOG="$DUMP_DIR/run.log"
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" TRITON_CACHE_DIR="$CACHE_DIR" \
  "$PY" -m moe.bench.cli --env "$ENV_NAME" --profile smoke \
        --groups baselines --models "$MODEL" --tokens "$TOKENS" \
        --out-dir "$WORKSPACE/results-ptx" 2>&1 | tee "$LOG"

echo
echo "=== which config vLLM resolved (quote this, do not derive it) ==="
grep -E 'Using configuration from|Using default MoE config' "$LOG" \
  || echo "  NEITHER LINE IN $LOG. vLLM logs one of them once per (E,N,dtype,device)
  via logger.info_once/warning_once, so a second cell in the same process is
  silent; check that this run was the first fused_experts call, and that the log
  level lets info through."

# `find` rather than `shopt -s globstar`, which is bash 4+ and therefore cannot
# be exercised on a macOS bash 3.2 laptop before the pod is rented. The -prune
# is load-bearing: the per-run Triton cache now lives UNDER the dump directory
# and a cache entry stores its own .ptx beside the cubin, so without it every
# kernel is counted twice, once under a hash-named path.
ptx=()
while IFS= read -r -d '' f; do
  ptx+=("$f")
done < <(find "$DUMP_DIR" -path "$CACHE_DIR" -prune -o -name '*.ptx' -print0)
if (( ${#ptx[@]} == 0 )); then
  echo "[mma] no .ptx under $DUMP_DIR (excluding $CACHE_DIR)." >&2
  echo "[mma] A stale cache is no longer a possible cause: TRITON_CACHE_DIR is" >&2
  echo "[mma] per-run and was empty a moment ago, so every kernel recompiled." >&2
  echo "[mma] What is left: the span never ran (check $LOG), vLLM is absent from" >&2
  echo "[mma] this env, or the cell fell back to a non-Triton path." >&2
  exit 1
fi
log "found ${#ptx[@]} PTX file(s)"

echo
echo "=== tensor core instructions, by kernel ==="
for f in "${ptx[@]}"; do
  w=$(grep -c 'wgmma' "$f" || true)
  m=$(grep -c 'mma\.sync' "$f" || true)
  l=$(grep -c 'ld\.global' "$f" || true)
  : "${w:=0}" "${m:=0}" "${l:=0}"
  (( w == 0 && m == 0 )) && continue
  printf '  %-52s wgmma=%-6s mma.sync=%-6s ld.global=%s\n' \
    "$(basename "$f")" "$w" "$m" "$l"
done

echo
echo "=== exact instruction shapes seen ==="
grep -ohE 'wgmma\.[a-z0-9_.]*|mma\.sync\.[a-z0-9_.]*' "${ptx[@]}" \
  | sort | uniq -c | sort -rn | head -20

echo
# `|| true` is load-bearing: grep exits 1 when it matches nothing, and under
# `set -o pipefail` that fails the pipeline, fails the assignment, and aborts the
# script before the verdict prints. Which is exactly what happened on the first
# real run, where wgmma=0 was the whole answer.
total_w=$( { grep -l 'wgmma' "${ptx[@]}" 2>/dev/null || true; } | wc -l | tr -d ' ')
total_m=$( { grep -l 'mma\.sync' "${ptx[@]}" 2>/dev/null || true; } | wc -l | tr -d ' ')
echo "=== verdict ==="
echo "  kernels containing wgmma   : $total_w"
echo "  kernels containing mma.sync: $total_m"
if (( total_w == 0 && total_m > 0 )); then
  echo "  -> C3 CONFIRMED: no warpgroup MMA. The decode path is on the"
  echo "     Ampere-era instruction, on Hopper silicon."
elif (( total_w > 0 )); then
  echo "  -> C3 REFUTED: warpgroup MMA is present. Triton is reaching M=64"
  echo "     some other way, and the inference from BLOCK_SIZE_M=16 was wrong."
else
  echo "  -> INCONCLUSIVE: neither instruction found. Check the fused_moe kernel"
  echo "     actually compiled, and that the right env was used."
fi
echo
echo "  PTX kept at $DUMP_DIR, run log at $LOG."
echo "  Both are transient pod output until they are committed; FINDINGS lists"
echo "  that as the reason C1 and C3 cannot be checked without a GPU."
