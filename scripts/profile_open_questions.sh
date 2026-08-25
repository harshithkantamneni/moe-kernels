#!/usr/bin/env bash
# The two questions the 2026-08-22 sweep could not answer without counters.
#
# Both need `ncu`, which is why this exists as a script rather than a note: the
# whole point of moving off a container tenancy was to be able to run it.
#
#   bash scripts/profile_open_questions.sh              # everything available
#   bash scripts/profile_open_questions.sh traffic      # just Q1 (needs ncu)
#   bash scripts/profile_open_questions.sh tile         # just Q2 (ncu or nsys)
#
# WHICH TOOL ANSWERS WHICH. Nsight Compute reads hardware performance counters
# and needs NVreg_RestrictProfilingToAdminUsers=0, a host kernel-module flag a
# container tenant cannot set: on RunPod it fails with ERR_NVGPUCTRPERM.
# Nsight Systems TRACES instead of reading counters, so it works there. That
# splits the two questions:
#
#   Q1 traffic  -> ncu ONLY. dram__bytes_read.sum is a counter; nothing traces it.
#   Q2 tile     -> either. The tile shape is in the kernel NAME, and a trace
#                  records names, so nsys is sufficient.
#
# So a pod that only has nsys can still settle the BLOCK_M claim, which is the
# one carried by three published posts.
#
# Q1. TRAFFIC MODEL. FINDINGS section 2 shows that on memory-bound rows the two
#     candidate models TIE (active experts 1.67x/14.3% CV, M-tiles 1.59x/14.5%),
#     so the sweep cannot say which one traffic scales with. One counter reading
#     settles it outright. On deepseek-v3, up stage, T=4096, uniform, one
#     expert's w1 is 58.72 MB:
#
#         256 active experts x 58.72 MB = 15.03 GB   -> traffic is active x W
#         370 M-tiles       x 58.72 MB = 21.73 GB   -> traffic is M_tiles x W
#
#     Those are 45% apart, so the reading is not marginal. Anything in between
#     is the partial re-read the sweep could not identify, and its position
#     fixes the alpha that could not be fitted from timing.
#
# Q2. TILE SHAPE. FINDINGS section 2 infers the incumbent's BLOCK_M is 128 from
#     timing alone: on the 54 discriminating memory-bound rows, BLOCK_M=64 puts
#     20 of them below the measured read ceiling, BLOCK_M=128 puts none. That
#     inference is load-bearing in three published posts. CUTLASS encodes the
#     tile shape in the kernel name, so a demangled name confirms or kills it.
#     Use T=1, where M-tiles and active experts are identical and nothing else
#     is moving.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WHICH="${1:-both}"
OUT="${MOE_PROFILE_DIR:-profiles}"
mkdir -p "$OUT"

# Both questions are about ONE kernel: the CUTLASS grouped GEMM that torch
# 2.13.0 dispatched to when the published rows were measured. A different torch
# ships a different CUTLASS, so profiling under one and quoting the answer
# against rows measured under the other silently answers a different question.
# The expected version is READ FROM the published CSV rather than hardcoded, so
# it cannot drift away from the run it is defending.
#
# RunPod's stock template is runpod-torch-v280, i.e. torch 2.8, which is exactly
# the mismatch this catches. Pin the venv instead:
#   MOE_BASE_TORCH='torch==2.13.0' \
#   MOE_TORCH_INDEX='https://download.pytorch.org/whl/cu130' \
#     bash scripts/setup_runpod.sh base
WANT_TORCH="$(python - <<'PY'
import csv, glob
rows = [next(csv.DictReader(open(p))) for p in sorted(glob.glob("results/published/*/merged.csv"))]
print(rows[-1]["torch_version"] if rows else "")
PY
)"
GOT_TORCH="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
echo "[torch] published=$WANT_TORCH  here=$GOT_TORCH"
if [[ -n "$WANT_TORCH" && "$GOT_TORCH" != "$WANT_TORCH" ]]; then
  if [[ "${MOE_ALLOW_TORCH_MISMATCH:-0}" == "1" ]]; then
    echo "[torch] MISMATCH overridden; whatever you learn describes $GOT_TORCH, not the published rows"
  else
    echo "REFUSING: this venv runs torch $GOT_TORCH, the published rows were" >&2
    echo "measured on $WANT_TORCH. A different torch ships a different CUTLASS," >&2
    echo "so the kernel you would profile is not the kernel being explained." >&2
    echo "Pin it (see the comment above), or set MOE_ALLOW_TORCH_MISMATCH=1 to" >&2
    echo "profile this torch on its own terms." >&2
    exit 2
  fi
fi

HAVE_NCU=0; HAVE_NSYS=0
command -v ncu  >/dev/null 2>&1 && HAVE_NCU=1
command -v nsys >/dev/null 2>&1 && HAVE_NSYS=1
echo "[tools] ncu=$HAVE_NCU nsys=$HAVE_NSYS"
if [[ "$HAVE_NCU" == "0" && "$HAVE_NSYS" == "0" ]]; then
  echo "neither ncu nor nsys is on PATH; install one:" >&2
  echo "  apt install -y nsight-compute   # counters, needs root + the module flag" >&2
  echo "  apt install -y nsight-systems   # tracing, works in a container" >&2
  exit 1
fi

# Counters are refused to non-admin users unless the module was loaded with
# NVreg_RestrictProfilingToAdminUsers=0. Check before spending a run.
ncu --version >/dev/null 2>&1 || NCU="sudo $(command -v ncu)"
NCU="${NCU:-$(command -v ncu)}"

# The `profile-cell` profile is one model, one routing, one L2 mode, one graph
# mode, so the matrix is a single cell and `--launch-count 1` cannot land on the
# wrong kernel. Token count is the only thing that varies between the two
# questions. `--groups baselines` keeps the reference python loop out.
cell() {  # tokens
  echo python -m moe.bench.cli --profile profile-cell --tokens "$1" \
    --groups baselines --impl torch_grouped_mm_up --out-dir "$OUT/rows"
}

if [[ "$WHICH" == "both" || "$WHICH" == "traffic" ]]; then
  if [[ "$HAVE_NCU" == "1" ]]; then
    echo "[Q1] dram__bytes_read.sum on deepseek-v3 up, T=4096, uniform"
    echo "     15.03 GB => active_experts x W ;  21.73 GB => M_tiles x W"
    $NCU --metrics dram__bytes_read.sum,dram__bytes_write.sum \
         --target-processes all --launch-count 1 \
         --export "$OUT/q1_traffic" --force-overwrite \
         -- $(cell 4096)
  else
    echo "[Q1] SKIPPED: needs ncu. A DRAM byte count is a hardware counter, and"
    echo "     no tracing tool can substitute for it. This question stays open"
    echo "     until you have a box where you are root."
  fi
fi

if [[ "$WHICH" == "both" || "$WHICH" == "tile" ]]; then
  echo "[Q2] CUTLASS tile shape from the demangled kernel name, T=1"
  if [[ "$HAVE_NCU" == "1" ]]; then
    $NCU --print-summary per-kernel --kernel-name-base demangled \
         --target-processes all --launch-count 1 \
         --export "$OUT/q2_tile" --force-overwrite \
         -- $(cell 1) \
      | tee "$OUT/q2_kernel_names.txt"
  else
    # Tracing is enough: the answer is in the NAME, not in a counter.
    echo "     using nsys (no counters needed for a kernel name)"
    nsys profile --trace=cuda --force-overwrite=true \
         --output "$OUT/q2_tile" \
         -- $(cell 1)
    # Report name has moved between nsys versions; try the current one first.
    nsys stats --report cuda_gpu_kern_sum "$OUT/q2_tile.nsys-rep" \
      2>/dev/null || nsys stats --report gpukernsum "$OUT/q2_tile.nsys-rep" \
      | tee "$OUT/q2_kernel_names.txt"
  fi
  echo "     grep the name for a tile shape, e.g. cutlass...128x128x64..."
  grep -oE "[0-9]+x[0-9]+x[0-9]+" "$OUT/q2_kernel_names.txt" 2>/dev/null \
    | sort | uniq -c || true
fi

echo
echo "Reports in $OUT. Both answers belong in FINDINGS section 2, which"
echo "currently records the question rather than an answer."
