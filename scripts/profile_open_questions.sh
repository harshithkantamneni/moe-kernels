#!/usr/bin/env bash
# The two questions the 2026-08-22 sweep could not answer without counters.
#
# Both need `ncu`, which is why this exists as a script rather than a note: the
# whole point of moving off a container tenancy was to be able to run it.
#
#   bash scripts/profile_open_questions.sh              # both
#   bash scripts/profile_open_questions.sh traffic      # just Q1
#   bash scripts/profile_open_questions.sh tile         # just Q2
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

command -v ncu >/dev/null 2>&1 || {
  echo "ncu is not on PATH. On a VM where you are root:" >&2
  echo "  sudo apt install -y cuda-nsight-compute-13-1" >&2
  exit 1; }

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
  echo "[Q1] dram__bytes_read.sum on deepseek-v3 up, T=4096, uniform"
  echo "     15.03 GB => active_experts x W ;  21.73 GB => M_tiles x W"
  $NCU --metrics dram__bytes_read.sum,dram__bytes_write.sum \
       --target-processes all --launch-count 1 \
       --export "$OUT/q1_traffic" --force-overwrite \
       -- $(cell 4096)
fi

if [[ "$WHICH" == "both" || "$WHICH" == "tile" ]]; then
  echo "[Q2] CUTLASS tile shape from the demangled kernel name, T=1"
  $NCU --print-summary per-kernel --kernel-name-base demangled \
       --target-processes all --launch-count 1 \
       --export "$OUT/q2_tile" --force-overwrite \
       -- $(cell 1) \
    | tee "$OUT/q2_kernel_names.txt"
  echo "     grep the name for a tile shape, e.g. cutlass...128x128x64..."
fi

echo
echo "Reports in $OUT. Both answers belong in FINDINGS section 2, which"
echo "currently records the question rather than an answer."
