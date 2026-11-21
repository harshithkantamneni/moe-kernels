#!/usr/bin/env bash
# The alpha surface across TWO cards, as one unattended run.
#
# RUN THE SAME SCRIPT ON BOTH. That is the whole design. The H200 and the A100
# differ in the one variable the mechanism is about -- L2 is 60 MiB against 40 MB
# -- and in the ridge, 163.7 against 145.7 FLOP/byte. Everything else is forced
# to match, because C5 was RETRACTED for comparing cards that had been running
# different kernels, and the fix is not to be more careful, it is to pin the
# kernel: BLOCK_M, GROUP_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K and num_stages are
# all forced, so the two cards run the same specialisation by construction.
#
# WHAT IT IS TESTING. alpha, the fraction of a weight re-read that misses L2,
# measured on H200 at 0.279 (deepseek-v2-lite, 11 MB/expert), 0.711-0.774 (qwen2,
# 37 MB) and 0.923-0.989 (mixtral, 235 MB). Monotone in footprint -- but "does
# the expert fit in L2" is REFUTED as the predictor, because qwen2's 37 MB fits
# inside 60 MiB and still pays 0.71. The candidate law is that alpha depends on
# footprint OVER L2. Six points, two cards:
#
#     model              MB/expert   phi on H200   phi on A100
#     deepseek-v2-lite        11        0.18          0.28
#     qwen2-57b-a14b          37        0.62          0.93
#     mixtral-8x7b           235        3.92          5.88
#
# If the six collapse onto one curve in phi, alpha is reuse-distance-limited and
# this is a law. If the A100 points sit off it, the L2 story is wrong and the
# cause is something else -- DRAM page locality, TLB reach, scheduler order.
# Either answer is worth the pod; one card cannot distinguish them.
set -uo pipefail          # NOT -e: an arm may fail and the run must continue

REPO="${REPO:-/workspace/repo}"
SESSION="${SESSION:-$(ls -1dt /workspace/session/*/ 2>/dev/null | head -1)}"
SESSION="${SESSION%/}"
if [[ -z "$SESSION" || ! -d "$SESSION" ]]; then
  SESSION="/workspace/session/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$SESSION" || { echo "cannot create $SESSION"; exit 3; }
fi
PY_VLLM="${PY_VLLM:-/workspace/venvs/vllm/bin/python}"
PY_BASE="${PY_BASE:-/workspace/venvs/base/bin/python}"
[[ -x "$PY_BASE" ]] || { echo "no base venv at $PY_BASE"; exit 3; }
[[ -x "$PY_VLLM" ]] || PY_VLLM="$PY_BASE"

say()  { printf '\n==== %s ====\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# --------------------------------------------------------------------------
# The card, and the stage count BOTH cards can run.
#
# num_stages is not a free choice. At BLOCK_M=256, BLOCK_N=64, BLOCK_K=64 in
# bf16 one pipeline stage holds (256*64 + 64*64) * 2 = 40,960 bytes, so four
# stages ask for 163,840 -- against 166,912 on an A100, which is 3 KB of
# headroom before Triton's own overhead. It will probably be refused, and a card
# that refuses ONE setting unpins the sweep: the remaining block sizes are no
# longer being compared at equal pipelining. So the count is computed from the
# device rather than assumed, and the SAME count has to be used on both cards
# for the cross-card row to mean anything.
# --------------------------------------------------------------------------
say "the card, and the stage count that fits it"
read -r CARD SMEM SLUG < <("$PY_BASE" - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
smem = getattr(p, "shared_memory_per_block_optin", 0) or getattr(
    p, "max_shared_memory_per_block_optin", 0) or p.shared_memory_per_block
slug = p.name.lower().replace(" ", "_").replace("-", "_")
print(p.name.replace(" ", "_"), int(smem), slug)
PY
) || { echo "no CUDA device"; exit 3; }

# THE MATCHED COUNT IS A STUDY DECISION, NOT A PER-CARD ONE. Sizing it to the
# card would pick 5 on an H200 and 3 on an A100 -- deeper pipelining on the
# bigger card, which is precisely the "different kernels per card" that got C5
# retracted. So it is FIXED at the deepest count the SMALLEST card in the study
# can run, and the per-card capacity is used only to refuse a count that will
# not fit rather than to choose one.
CROSS_CARD_STAGES=3
STAGES="${STAGES:-$CROSS_CARD_STAGES}"
fits="$("$PY_BASE" - "$SMEM" "$STAGES" <<'PY'
import sys
smem, stages = int(sys.argv[1]), int(sys.argv[2])
per_stage = (256 * 64 + 64 * 64) * 2      # BLOCK_M=256, BN=64, BK=64, bf16
# 0.90 rather than 1.0: Triton allocates beyond the operand tiles, and a sweep
# that dies at its last block size is worse than one that pipelines less.
need = per_stage * stages
deepest = max((s for s in (2, 3, 4, 5) if per_stage * s <= 0.90 * smem), default=0)
print(f"{'yes' if need <= 0.90 * smem else 'NO'} {need} {deepest}")
PY
)"
read -r STAGES_FIT STAGES_NEED STAGES_DEEPEST <<< "$fits"
if [[ "$STAGES_FIT" == "NO" ]]; then
  echo "REFUSING: num_stages=$STAGES needs $STAGES_NEED bytes of shared memory at" >&2
  echo "BLOCK_M=256 and this card offers $SMEM. The deepest that fits is" >&2
  echo "$STAGES_DEEPEST -- but lowering it HERE breaks the match with the other" >&2
  echo "card, so the other card has to be re-run at the lower count too. Set" >&2
  echo "STAGES=$STAGES_DEEPEST deliberately on BOTH, or drop BLOCK_M=256." >&2
  exit 3
fi
note "device        $CARD"
note "shared mem    $SMEM bytes per block (opt-in)"
note "num_stages    $STAGES  (needs $STAGES_NEED B; this card could do $STAGES_DEEPEST)"
note ""
note "THE H200 ROWS ALREADY IN HAND WERE TAKEN AT num_stages=4 AND DO NOT MATCH."
note "Run this same script on the H200 to produce its stages=$STAGES row; the"
note "run id carries -s$STAGES so the two never resume into each other."

OUT="$SESSION/cross_card/$SLUG-s$STAGES"
LOGS="$SESSION/logs"
LEDGER="$OUT/ARMS.tsv"
mkdir -p "$OUT" "$LOGS" || exit 3
[[ -f "$LEDGER" ]] || printf 'arm\tstatus\tseconds\tlog\n' > "$LEDGER"
started=$(date -u +%s)

arm() {
  local name="$1"; shift
  if grep -qP "^\Q$name\E\tPASS\t" "$LEDGER" 2>/dev/null; then
    note "SKIP $name (already PASS)"; return 0
  fi
  local log="$LOGS/cc_${SLUG}_$name.log" t0 t1 rc
  note "-> $name   log $log"
  t0=$(date -u +%s); "$@" > "$log" 2>&1; rc=$?; t1=$(date -u +%s)
  local status=PASS; [[ $rc -eq 0 ]] || status="EXIT$rc"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$((t1 - t0))" "$log" >> "$LEDGER"
  note "   $status in $((t1 - t0))s"
  return 0
}

sweep() {   # sweep <name> <model> [extra...]
  local name="$1" model="$2"; shift 2
  arm "$name" "$PY_VLLM" "$REPO/scripts/block_m_crossing_sweep.py" \
      --model "$model" --r-max 1024 --num-stages "$STAGES" --out "$OUT" "$@"
}

say "CROSS-CARD ALPHA SURFACE  card=$CARD  stages=$STAGES"
note "out    $OUT"
note "ledger $LEDGER"

# --------------------------------------------------------------------------
# Calibration FIRST, and it is not optional on a new pod. measured_*.yaml is
# matched by DEVICE NAME, so a second pod of the same part silently reuses the
# first one's ceilings -- and the ceilings are not stable: the H200's dense bf16
# moved 7.1% between 2026-08-28 and 2026-09-01 while its bandwidth reproduced to
# 0.014%. The ridge is compute over bandwidth, so that drift lands entirely on
# the ridge, which is the denominator of every crossing prediction below.
# --------------------------------------------------------------------------
say "1. calibrate THIS card"
arm calibrate "$PY_BASE" "$REPO/scripts/calibrate_hardware.py"
note "diff it before trusting anything: git -C $REPO diff moe/bench/hardware/"

# --------------------------------------------------------------------------
# The footprint axis, at the swizzle setting vLLM's fallback ladder actually
# holds through decode. G=1 first for all three models, because that is the row
# that pairs against the H200 numbers already in hand.
# --------------------------------------------------------------------------
say "2. the footprint axis at GROUP_SIZE_M=1, the fallback's decode setting"
note "H200 measured 0.279 / 0.711 / 0.975 for v2lite / qwen2 / mixtral"
note "a smaller L2 should raise ALL THREE, and qwen2 most: phi 0.62 -> 0.93"
sweep v2lite_g1  deepseek-v2-lite --group-m 1
sweep qwen2_g1   qwen2-57b-a14b   --group-m 1
sweep mixtral_g1 mixtral-8x7b     --group-m 1

say "3. the swizzle axis, on the model where GROUP_SIZE_M does not saturate"
note "mixtral's num_pid_m is 4 at the small rungs, so G=8,16,32,64 collapse to"
note "ONE setting there; qwen2's runs to ~550 and the ladder stays real"
for g in 8 16 64; do sweep "qwen2_g$g" qwen2-57b-a14b --group-m "$g"; done
sweep mixtral_g16 mixtral-8x7b --group-m 16

say "4. the activation confound"
note "the 0.25 bound this study quotes is BLOCK_M/BLOCK_N at 16/64, but the"
note "sweep pins BN=64 while running BM to 256, where the ratio is 4.0"
sweep qwen2_bn256 qwen2-57b-a14b --group-m 1 --block-n 256

# --------------------------------------------------------------------------
# The ISA census. On sm80 getMMAVersionSafe returns {2} alone, so NO tile should
# reach wgmma at any size -- against the H200, which showed wgmma=8 at BLOCK_M
# 64 and 128 and mma.sync only at 16. That is a one-line prediction and it is
# the second architecture C3's rescope has ever been tested on. The A100 has
# never been dumped successfully, very likely because of the shared Triton cache
# that pre-flight P5 now tests and that only started passing on 2026-09-01.
# --------------------------------------------------------------------------
say "5. the ISA census: sm80 should reach NO warpgroup MMA at any tile"
for cell in "deepseek-v3 16" "deepseek-v3 256" "mixtral-8x7b 256"; do
  set -- $cell
  arm "ptx_$1_T$2" bash "$REPO/scripts/check_mma_path.sh" \
      --model "$1" --tokens "$2" --out "$OUT/ptx/$1-T$2"
done
arm ptx_tarball bash -c "cd '$OUT' && tar czf '$SESSION/exfil_ptx-$SLUG.tar.gz' ptx 2>/dev/null && ls -l '$SESSION/exfil_ptx-$SLUG.tar.gz'"

say "THE SURFACE ON THIS CARD"
"$PY_BASE" "$REPO/scripts/alpha_surface.py" "$OUT" 2>&1 | tee "$OUT/SURFACE.txt"

say "BOTH CARDS, IF BOTH HAVE RUN"
"$PY_BASE" "$REPO/scripts/alpha_surface.py" "$SESSION/cross_card" 2>&1 \
  | tee "$SESSION/cross_card/SURFACE_ALL.txt" | tail -40

say "ARMS"
cat "$LEDGER"
printf '\ntotal %s min\n' "$(( ($(date -u +%s) - started) / 60 ))"
printf 'surface   %s\n' "$OUT/SURFACE.txt"
printf 'combined  %s\n' "$SESSION/cross_card/SURFACE_ALL.txt"
printf 'ptx       %s\n' "$SESSION/exfil_ptx-$SLUG.tar.gz"
printf '\nCOMMIT AND PUSH before terminating. Nothing here is in git yet, and the\n'
printf 'calibration this card just measured is the ruler every number above uses.\n'
