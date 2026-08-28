#!/usr/bin/env bash
# C3: does vLLM's decode path use Hopper's warpgroup tensor core, or not?
#
#   bash scripts/check_mma_path.sh
#   bash scripts/check_mma_path.sh --tokens 16 --model deepseek-v3
#
# THE QUESTION. Hopper's `wgmma.mma_async.m64nNk16` has M fixed at 64 by the
# instruction set. vLLM's tuned H200 config sets BLOCK_SIZE_M to 16 for every
# batch size from 1 to 256, the whole decode range. 16 < 64, so either Triton
# falls back to the Ampere-era `mma.sync` path, or it pads to 64 internally.
#
# The study currently records that as an INFERENCE, not a measurement. This
# settles it: Triton writes its generated PTX to disk when asked, CUTLASS and
# PTX both carry the instruction name, and the answer is a grep.
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

rm -rf "$DUMP_DIR"
mkdir -p "$DUMP_DIR"
log "dumping Triton IR and PTX to $DUMP_DIR"

# TRITON_KERNEL_DUMP=1 writes every compilation stage; TRITON_DUMP_DIR says where.
# A smoke-sized cell is enough: the instruction selection does not depend on how
# many times the kernel runs, only on the tile shape it was compiled for.
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" \
  "$PY" -m moe.bench.cli --env "$ENV_NAME" --profile smoke \
        --groups baselines --models "$MODEL" --tokens "$TOKENS" \
        --out-dir "$WORKSPACE/results-ptx" 2>&1 | tail -5 || true

shopt -s nullglob globstar
ptx=("$DUMP_DIR"/**/*.ptx)
if (( ${#ptx[@]} == 0 )); then
  echo "[mma] no .ptx under $DUMP_DIR." >&2
  echo "[mma] Triton only dumps kernels it COMPILES; a cached kernel is not" >&2
  echo "[mma] recompiled. Clear the cache and retry:" >&2
  echo "[mma]   rm -rf \${TRITON_CACHE_DIR:-~/.triton/cache}" >&2
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
echo "  PTX kept at $DUMP_DIR for the writeup."
