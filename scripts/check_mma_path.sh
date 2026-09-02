#!/usr/bin/env bash
# C3: does vLLM's decode path use Hopper's warpgroup tensor core, or not?
#
#   bash scripts/check_mma_path.sh
#   bash scripts/check_mma_path.sh --tokens 16 --model deepseek-v3
#   bash scripts/check_mma_path.sh --tokens 256 --block-m 16,64   # the switch, tile-forced
#   bash scripts/check_mma_path.sh --block-m 16,64 --dry-run      # the plan, no GPU needed
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
#
# --------------------------------------------------------------------------
# --block-m: THE LOOSE END, which is a different question from the one above.
#
# `docs/STUDY.md` item 3 is marked DONE with a tail: "confirm the instruction
# actually switched by re-running check_mma_path.sh under the override". The
# 2026-08-27 census DID see wgmma=0 at BLOCK_M=16 and wgmma=8 at 64 and 128 --
# but through vLLM's CONFIG LADDER, by asking for different token counts and
# letting the ladder pick a different tile at each. Two things move along that
# ladder at once. `get_default_config` sets num_warps to 4 at M<=128 and 8 above
# it, and num_warps is the OTHER half of Triton's warpgroup predicate, so a
# census taken across token counts cannot separate "the tile selects the
# instruction" from "the warp count does" -- or from "the batch size does",
# which is what a reader who does not know the ladder will assume.
#
# So --block-m holds the token count FIXED and forces each tile in turn through
# MOE_FORCE_TILE, with BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps and
# num_stages pinned identically across every arm. num_warps=8 in particular
# satisfies `num_warps % 4 == 0` at every tile this can run, which leaves
# `BLOCK_M % 64 == 0` as the only term in the predicate that moves. That is the
# whole difference between "the instruction correlates with batch size" and
# "the instruction is selected by the tile".
#
# THE PIN IS VERIFIED, NEVER ASSUMED. "The variable was set" and "the kernel ran
# that tile" are different facts, and the gap between them cost the 2026-09-01
# session its pinned crossing (gate S6a, FAIL, "observed tile_block_m = none").
# Each arm reads all six OBSERVED tile columns back out of its own fresh run CSV
# -- written by `_framework_config.recording_tile_config`, which reads the config
# out of vLLM during a real call -- and this script REFUSES to census an arm
# whose observed tile is not exactly the tile it asked for. A census attributed
# to a tile that did not run is worse than no census.
#
# EXIT CODES. 0 every gate passed. 1 a gate failed. 2 refused before measuring.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKSPACE="${WORKSPACE:-/workspace}"
DUMP_DIR="${MOE_PTX_DIR:-$WORKSPACE/ptx}"
MODEL="deepseek-v3"
TOKENS="16"
ENV_NAME="vllm"
BLOCK_M_LIST=""
DRY_RUN=0

# The five knobs held IDENTICAL across every forced arm, so BLOCK_SIZE_M is the
# only thing that can move. Same values scripts/tile_sweep.py and
# scripts/block_m_crossing_sweep.py pin, so a PTX census and a timing sweep
# describe the same kernel. num_warps=8 is load bearing: it clears the
# `num_warps % 4 == 0` half of Triton's warpgroup predicate at every tile, which
# is what leaves BLOCK_M as the only term that varies.
BLOCK_N="64"
BLOCK_K="64"
GROUP_M="1"
NUM_WARPS="8"
NUM_STAGES="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokens)   TOKENS="$2"; shift 2 ;;
    --model)    MODEL="$2"; shift 2 ;;
    --env)      ENV_NAME="$2"; shift 2 ;;
    --out)      DUMP_DIR="$2"; shift 2 ;;
    --block-m)  BLOCK_M_LIST="$2"; shift 2 ;;
    --block-n)  BLOCK_N="$2"; shift 2 ;;
    --block-k)  BLOCK_K="$2"; shift 2 ;;
    --group-m)  GROUP_M="$2"; shift 2 ;;
    --warps)    NUM_WARPS="$2"; shift 2 ;;
    --stages)   NUM_STAGES="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[mma] %s\n' "$*"; }
refuse() { echo "[mma] REFUSED: $*" >&2; exit 2; }

# --------------------------------------------------------------------------
# The plan and the prediction, printed before anything runs and before the
# interpreter is even looked for, so --dry-run works on a laptop.
# --------------------------------------------------------------------------
predicts_wgmma() {   # <block_m> <num_warps> -> echoes yes|no
  if (( $1 % 64 == 0 && $2 % 4 == 0 )); then echo yes; else echo no; fi
}

ARMS=()
if [[ -n "$BLOCK_M_LIST" ]]; then
  case "$TOKENS" in
    *,*) refuse "--block-m forces the tile at a FIXED token count, and --tokens
  '$TOKENS' names several. Splitting the batch across arms puts back exactly the
  confound this mode removes. Run one token count per invocation." ;;
  esac
  OLD_IFS="$IFS"; IFS=','; for bm in $BLOCK_M_LIST; do ARMS+=("$bm"); done
  IFS="$OLD_IFS"
  (( ${#ARMS[@]} >= 2 )) || refuse "--block-m needs at least two tiles: this mode
  is a COMPARISON at one token count, and one arm compares nothing. Try
  --block-m 16,64."

  yes_n=0; no_n=0
  for bm in "${ARMS[@]}"; do
    [[ "$bm" =~ ^[0-9]+$ ]] || refuse "--block-m '$bm' is not a number"
    if [[ "$(predicts_wgmma "$bm" "$NUM_WARPS")" == yes ]]; then
      yes_n=$((yes_n + 1)); else no_n=$((no_n + 1)); fi
  done
  # NON-VACUITY, at plan time. Every arm on the same side of `BLOCK_M % 64 == 0`
  # produces a table where the instruction never changes, which reads as a
  # finding and is a tautology.
  (( yes_n > 0 && no_n > 0 )) || refuse "every tile in --block-m '$BLOCK_M_LIST'
  falls on the SAME side of the predicate BLOCK_M % 64 == 0 at num_warps=$NUM_WARPS
  ($yes_n predicted to emit wgmma, $no_n not). The census would then be constant
  by construction and could not show a switch. Include one tile from each side,
  e.g. --block-m 16,64."

  echo "=== plan: the instruction at a FIXED token count, only BLOCK_SIZE_M moving ==="
  echo "  model $MODEL   tokens $TOKENS   env $ENV_NAME"
  echo "  pinned across every arm: BLOCK_SIZE_N=$BLOCK_N BLOCK_SIZE_K=$BLOCK_K"
  echo "                           GROUP_SIZE_M=$GROUP_M num_warps=$NUM_WARPS num_stages=$NUM_STAGES"
  echo "  forced through MOE_FORCE_TILE, and each arm's OBSERVED tile columns are"
  echo "  read back out of its own run CSV before its PTX is censused."
  echo
  echo "  PREDICTIONS, registered before the run. Triton emits wgmma only when"
  echo "  BLOCK_M % 64 == 0 AND num_warps % 4 == 0; num_warps is pinned at"
  echo "  $NUM_WARPS in every arm, so only the first term can move:"
  for bm in "${ARMS[@]}"; do
    if [[ "$(predicts_wgmma "$bm" "$NUM_WARPS")" == yes ]]; then
      printf '    BLOCK_SIZE_M=%-4s wgmma  > 0   (%s %% 64 == 0)\n' "$bm" "$bm"
    else
      printf '    BLOCK_SIZE_M=%-4s wgmma == 0   (%s %% 64 != 0, so mma.sync)\n' "$bm" "$bm"
    fi
  done
  echo
  echo "  GATES"
  echo "    G1 VALIDITY  every arm's OBSERVED six tile columns equal the forced ones"
  echo "    G2 VALIDITY  every arm dumped at least one PTX carrying a tensor-core instruction"
  echo "    G3 VALIDITY  the arms compiled DIFFERENT kernels (distinct PTX checksums)"
  echo "    G4 CLAIM     wgmma is present in exactly the arms the predicate names"
  echo "  A G1-G3 FAIL means no census on the page may be attributed to a tile."
  echo "  A G4 FAIL is a result: the instruction is not selected by the tile alone."
  echo "  dumps to $DUMP_DIR/bm<N>/"
else
  echo "=== plan: one cell on vLLM's own config ladder ==="
  echo "  model $MODEL   tokens $TOKENS   env $ENV_NAME   dump $DUMP_DIR"
  echo "  no tile is forced, so the tile is whatever the ladder resolves and the"
  echo "  instruction cannot be attributed to it alone. Use --block-m 16,64 for"
  echo "  the attribution."
fi
echo

if (( DRY_RUN )); then
  echo "[mma] --dry-run: nothing was executed and nothing was written."
  exit 0
fi

PY="${MOE_PYTHON:-$WORKSPACE/venvs/$ENV_NAME/bin/python}"
[[ -x "$PY" ]] || { echo "no interpreter at $PY; run setup_runpod.sh" >&2; exit 1; }

#: Dropped inside a dump directory so a later run can recognise its own output.
DUMP_MARKER=".moe-ptx-dump"

# DUMP_DIR arrives from MOE_PTX_DIR or --out and is handed straight to `rm -rf`,
# so it is checked before it is deleted. MOE_PTX_DIR="" collapses the default,
# `--out` with a missing value swallows the next flag, and MOE_PTX_DIR=$HOME on a
# rented pod is one keystroke from a path whose recursive removal cannot be
# undone. A blacklist alone is not enough, so an EXISTING directory must also be
# empty or carry this script's own marker: that is what "clearly a dump
# directory" means here, and nothing else is deleted.
case "$DUMP_DIR" in
  "") refuse "refusing to rm -rf an empty path" ;;
  /*) : ;;
  *)  refuse "refusing to rm -rf '$DUMP_DIR': not an absolute path" ;;
esac
case "${DUMP_DIR%/}" in
  ""|"/"|"/root"|"/home"|"/tmp"|"/usr"|"/etc"|"/var"|"/workspace"|"$HOME"|"$WORKSPACE"|"$REPO_ROOT")
    refuse "refusing to rm -rf '$DUMP_DIR': that is a root, home, workspace or repo directory" ;;
esac
if [[ -e "$DUMP_DIR" ]]; then
  [[ -d "$DUMP_DIR" ]] || refuse "refusing to rm -rf '$DUMP_DIR': it exists and is not a directory"
  if [[ ! -e "$DUMP_DIR/$DUMP_MARKER" && -n "$(ls -A "$DUMP_DIR")" ]]; then
    refuse "refusing to rm -rf '$DUMP_DIR': it is not empty and carries no $DUMP_MARKER, so this script did not write it"
  fi
fi

DUMP_DIR="${DUMP_DIR%/}"   # so the -prune path below matches find's output exactly

rm -rf "$DUMP_DIR"
mkdir -p "$DUMP_DIR"
: > "$DUMP_DIR/$DUMP_MARKER"
log "dumping Triton IR and PTX to $DUMP_DIR"

# --------------------------------------------------------------------------
# One arm: compile, then census. Set as globals rather than returned, because
# bash returns an exit status and these are four numbers.
#
# ARM_WGMMA / ARM_MMA  kernels containing each instruction
# ARM_PTX              PTX files found
# ARM_SUM              checksum over the PTX, for the "did the kernel change" assay
# --------------------------------------------------------------------------
run_arm() {   # <dump-subdir> <label> [force-json]
  local dir="$1" label="$2" force="${3:-}"
  local cache="$dir/_cache" results="$dir/results" log_file="$dir/run.log"
  mkdir -p "$dir" "$results"

  # A CACHE HIT DUMPS NO PTX, and that is the failure this line exists to
  # prevent. TRITON_KERNEL_DUMP asks for the dump, but Triton does not recompile
  # a kernel it has already built, and setup_runpod.sh exports a SHARED
  # TRITON_CACHE_DIR=$WORKSPACE/triton-cache that every earlier sweep on the pod
  # has already populated with this exact fused_moe specialisation. Inheriting it
  # means nothing compiles, nothing is written, and the script exits 1 reporting
  # that the kernel never compiled -- which is very likely why the A100 was never
  # successfully dumped. A per-ARM cache forces every specialisation to be built,
  # exactly as scripts/tile_sweep.py arm_ptx_dump does.
  #
  # The run CSV goes under the arm too, and that is not tidiness: the observed
  # tile is read back out of it, and a shared results directory accumulating
  # run_*.csv from previous arms and previous days would hand this arm another
  # arm's tile.
  #
  # TEE'd rather than piped to `tail -5`, and with no `|| true`. The discarded
  # lines were the evidence: vLLM logs "Using configuration from FILE for MoE
  # layer." on a tuned hit and "Using default MoE config. Performance might be
  # sub-optimal!" on a miss, and that single line is what turns every tile
  # statement in FINDINGS C3 and C5 from DERIVED into OBSERVED. `|| true` also
  # meant a sweep that died -- OOM, missing vLLM, a bad --model -- still reached
  # the PTX scan and reported on whatever stale files were lying around.
  if [[ -n "$force" ]]; then
    MOE_FORCE_TILE="$force" TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$dir" \
      TRITON_CACHE_DIR="$cache" \
      "$PY" -m moe.bench.cli --env "$ENV_NAME" --profile smoke \
            --groups baselines --models "$MODEL" --tokens "$TOKENS" \
            --out-dir "$results" 2>&1 | tee "$log_file"
  else
    TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$dir" TRITON_CACHE_DIR="$cache" \
      "$PY" -m moe.bench.cli --env "$ENV_NAME" --profile smoke \
            --groups baselines --models "$MODEL" --tokens "$TOKENS" \
            --out-dir "$results" 2>&1 | tee "$log_file"
  fi

  echo
  echo "=== [$label] which config vLLM resolved (quote this, do not derive it) ==="
  grep -E 'Using configuration from|Using default MoE config' "$log_file" \
    || echo "  NEITHER LINE IN $log_file. vLLM logs one of them once per
  (E,N,dtype,device) via logger.info_once/warning_once, so a second cell in the
  same process is silent; check that this run was the first fused_experts call,
  and that the log level lets info through."

  # `find` rather than `shopt -s globstar`, which is bash 4+ and therefore cannot
  # be exercised on a macOS bash 3.2 laptop before the pod is rented. The -prune
  # is load-bearing: the per-arm Triton cache lives UNDER the arm directory and a
  # cache entry stores its own .ptx beside the cubin, so without it every kernel
  # is counted twice, once under a hash-named path.
  ARM_FILES=()
  local f
  while IFS= read -r -d '' f; do
    ARM_FILES+=("$f")
  done < <(find "$dir" -path "$cache" -prune -o -name '*.ptx' -print0)
  ARM_PTX=${#ARM_FILES[@]}
  ARM_WGMMA=0
  ARM_MMA=0
  ARM_SUM="none"
  if (( ARM_PTX == 0 )); then
    return 0
  fi

  echo
  echo "=== [$label] tensor core instructions, by kernel ==="
  local w m l
  for f in "${ARM_FILES[@]}"; do
    w=$(grep -c 'wgmma' "$f" || true)
    m=$(grep -c 'mma\.sync' "$f" || true)
    l=$(grep -c 'ld\.global' "$f" || true)
    : "${w:=0}" "${m:=0}" "${l:=0}"
    # `if` rather than the `(( ... )) && continue` this line used to be. The
    # census now lives in a function, and a && list whose left side is false on
    # the last iteration leaves the loop -- and so the function -- returning
    # non-zero, which under `set -e` aborts the run at the call site with no
    # message at all.
    if (( w != 0 || m != 0 )); then
      printf '  %-52s wgmma=%-6s mma.sync=%-6s ld.global=%s\n' \
        "$(basename "$f")" "$w" "$m" "$l"
    fi
  done

  echo
  echo "=== [$label] exact instruction shapes seen ==="
  grep -ohE 'wgmma\.[a-z0-9_.]*|mma\.sync\.[a-z0-9_.]*' "${ARM_FILES[@]}" \
    | sort | uniq -c | sort -rn | head -20

  # `|| true` is load-bearing: grep exits 1 when it matches nothing, and under
  # `set -o pipefail` that fails the pipeline, fails the assignment, and aborts
  # the script before the verdict prints. Which is exactly what happened on the
  # first real run, where wgmma=0 was the whole answer.
  ARM_WGMMA=$( { grep -l 'wgmma' "${ARM_FILES[@]}" 2>/dev/null || true; } | wc -l | tr -d ' ')
  ARM_MMA=$( { grep -l 'mma\.sync' "${ARM_FILES[@]}" 2>/dev/null || true; } | wc -l | tr -d ' ')
  # cksum rather than sha1sum/shasum: POSIX, present on both the pod and a
  # macOS laptop, and this is an "are these two files identical" question with
  # no adversary in it. Sorted so file discovery order cannot change the digest.
  ARM_SUM=$( { cat "${ARM_FILES[@]}" 2>/dev/null || true; } | sort | cksum | awk '{print $1}')
}

# The six OBSERVED tile columns, read back out of one arm's own run CSV. Prints
# `bm=.. bn=.. bk=.. gm=.. warps=.. stages=.. src=.. rows=N`, with each field
# the comma-joined set of DISTINCT values seen. Distinct, not a mean or a first:
# two rows disagreeing about the tile is a fact this must show rather than
# average away.
observed_tile() {   # <results-dir>
  "$PY" - "$1" "$REPO_ROOT" <<'PYEOF'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])
from moe.bench.schema import UNRECORDED, read_csv

COLS = [("bm", "tile_block_m"), ("bn", "tile_block_n"), ("bk", "tile_block_k"),
        ("gm", "tile_group_m"), ("warps", "tile_num_warps"),
        ("stages", "tile_num_stages"), ("src", "tile_config_source")]
seen = {short: set() for short, _ in COLS}
rows = 0
for path in sorted(Path(sys.argv[1]).glob("run_*.csv")):
    for row in read_csv(path):
        rows += 1
        for short, column in COLS:
            value = row.get(column)
            if value not in (None, "", UNRECORDED, "0", 0):
                seen[short].add(str(value))
parts = [f"{short}={','.join(sorted(seen[short])) or 'none'}" for short, _ in COLS]
print(" ".join(parts) + f" rows={rows}")
PYEOF
}

# --------------------------------------------------------------------------
# UNFORCED: one cell on vLLM's own ladder. The original behaviour, unchanged,
# and still the right question when what is wanted is "what does this cell
# actually run".
# --------------------------------------------------------------------------
if [[ -z "$BLOCK_M_LIST" ]]; then
  run_arm "$DUMP_DIR" "ladder"
  if (( ARM_PTX == 0 )); then
    echo "[mma] no .ptx under $DUMP_DIR (excluding its cache)." >&2
    echo "[mma] A stale cache is no longer a possible cause: TRITON_CACHE_DIR is" >&2
    echo "[mma] per-run and was empty a moment ago, so every kernel recompiled." >&2
    echo "[mma] What is left: the span never ran (check the log), vLLM is absent" >&2
    echo "[mma] from this env, or the cell fell back to a non-Triton path." >&2
    exit 1
  fi
  log "found $ARM_PTX PTX file(s)"
  echo
  echo "=== verdict ==="
  echo "  kernels containing wgmma   : $ARM_WGMMA"
  echo "  kernels containing mma.sync: $ARM_MMA"
  if (( ARM_WGMMA == 0 && ARM_MMA > 0 )); then
    echo "  -> C3 CONFIRMED: no warpgroup MMA. The decode path is on the"
    echo "     Ampere-era instruction, on Hopper silicon."
  elif (( ARM_WGMMA > 0 )); then
    echo "  -> C3 REFUTED: warpgroup MMA is present. Triton is reaching M=64"
    echo "     some other way, and the inference from BLOCK_SIZE_M=16 was wrong."
  else
    echo "  -> INCONCLUSIVE: neither instruction found. Check the fused_moe kernel"
    echo "     actually compiled, and that the right env was used."
  fi
  echo
  echo "  THE TILE HERE IS THE LADDER'S, NOT AN OBSERVATION UNDER CONTROL. Along"
  echo "  the ladder num_warps moves with BLOCK_SIZE_M (4 at M<=128, 8 above), so"
  echo "  this cell cannot say which of the two selected the instruction. That is"
  echo "  what --block-m 16,64 is for."
  echo
  echo "  PTX kept at $DUMP_DIR, run log at $DUMP_DIR/run.log."
  echo "  Both are transient pod output until they are committed; FINDINGS lists"
  echo "  that as the reason C1 and C3 cannot be checked without a GPU."
  exit 0
fi

# --------------------------------------------------------------------------
# FORCED: one arm per tile, at one token count, with only BLOCK_SIZE_M moving.
# --------------------------------------------------------------------------
SUMMARY="$DUMP_DIR/summary.tsv"
: > "$SUMMARY"
g1_fail=0; g2_fail=0

for bm in "${ARMS[@]}"; do
  dir="$DUMP_DIR/bm$bm"
  force="{\"BLOCK_SIZE_M\":$bm,\"BLOCK_SIZE_N\":$BLOCK_N,\"BLOCK_SIZE_K\":$BLOCK_K,\"GROUP_SIZE_M\":$GROUP_M,\"num_warps\":$NUM_WARPS,\"num_stages\":$NUM_STAGES}"
  echo
  echo "############ arm BLOCK_SIZE_M=$bm  (T=$TOKENS, everything else pinned) ############"
  echo "MOE_FORCE_TILE=$force"
  run_arm "$dir" "BM=$bm" "$force"

  # G1's evidence, read before the census is believed. `|| true` so a failure to
  # read the CSV becomes the string "none" and a G1 FAIL, rather than aborting
  # under `set -e` and losing the arms already measured.
  obs="$(observed_tile "$dir/results" 2>/dev/null || true)"
  [[ -n "$obs" ]] || obs="bm=none bn=none bk=none gm=none warps=none stages=none src=none rows=0"
  want="bm=$bm bn=$BLOCK_N bk=$BLOCK_K gm=$GROUP_M warps=$NUM_WARPS stages=$NUM_STAGES src=vllm_override"
  got="${obs% rows=*}"
  [[ "$got" == "$want" ]] || g1_fail=$((g1_fail + 1))
  (( ARM_WGMMA + ARM_MMA > 0 )) || g2_fail=$((g2_fail + 1))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$bm" "$(predicts_wgmma "$bm" "$NUM_WARPS")" "$ARM_WGMMA" "$ARM_MMA" \
    "$ARM_PTX" "$ARM_SUM" "$obs" >> "$SUMMARY"
  echo
  echo "  observed tile: $obs"
  echo "  asked for    : $want"
done

echo
echo "=== the switch at T=$TOKENS, only BLOCK_SIZE_M moving ==="
printf '  %-9s %-11s %-9s %-11s %-6s %s\n' \
  BLOCK_M "predicted" "kernels" "kernels" "PTX" "observed tile"
printf '  %-9s %-11s %-9s %-11s %-6s %s\n' \
  "" "wgmma" "w/ wgmma" "w/ mma.sync" "files" ""
while IFS=$'\t' read -r bm pred w m n sum obs; do
  printf '  %-9s %-11s %-9s %-11s %-6s %s\n' \
    "$bm" "$( [[ "$pred" == yes ]] && echo '> 0' || echo '== 0' )" "$w" "$m" "$n" "$obs"
done < "$SUMMARY"
echo "  PTX checksums: $(cut -f6 "$SUMMARY" | tr '\n' ' ')"

# G3: did the arms compile DIFFERENT kernels. Identical PTX across two tile
# settings means the override did not reach the compile, which is the same
# failure the crossing sweep's gate 0 counts Triton artefacts for, and it looks
# from the outside exactly like a tidy null result.
distinct=$(cut -f6 "$SUMMARY" | sort -u | wc -l | tr -d ' ')
narms=${#ARMS[@]}
# G4: wgmma present in exactly the arms the predicate names.
g4_fail=0
while IFS=$'\t' read -r bm pred w m n sum obs; do
  if [[ "$pred" == yes ]]; then (( w > 0 )) || g4_fail=$((g4_fail + 1))
  else (( w == 0 )) || g4_fail=$((g4_fail + 1)); fi
done < "$SUMMARY"

echo
echo "=== gates ==="
verdict_line() {   # <tag> <kind> <ok> <claim> <measured> <gate> <consequence>
  local tag="$1" kind="$2" ok="$3" claim="$4" measured="$5" gate="$6" cons="$7"
  printf '%-3s %-8s %-4s %s\n' "$tag" "$kind" \
    "$( (( ok )) && echo PASS || echo FAIL )" "$claim"
  printf '             measured %s   gate %s\n' "$measured" "$gate"
  printf '             if this FAILS: %s\n' "$cons"
}
verdict_line G1 VALIDITY "$(( g1_fail == 0 ))" \
  "every arm ran the tile it was given" \
  "$(( narms - g1_fail ))/$narms arms observed exactly what they forced" \
  "all $narms" \
  "MOE_FORCE_TILE did not reach the kernel in some arm, so its census belongs to
             a tile nobody chose. Nothing on this page may be attributed to a tile.
             Check that moe/bench/force_tile.py is present in this checkout and that
             the impl under test exposes the force_tile_config hook."
verdict_line G2 VALIDITY "$(( g2_fail == 0 ))" \
  "every arm dumped a PTX carrying a tensor-core instruction" \
  "$(( narms - g2_fail ))/$narms arms" "all $narms" \
  "an arm compiled nothing, or nothing with a tensor core in it. An absent
             instruction and an absent kernel look identical in the table above."
verdict_line G3 VALIDITY "$(( distinct == narms ))" \
  "the arms compiled DIFFERENT kernels" \
  "$distinct distinct PTX checksum(s)" "== $narms" \
  "two arms produced byte-identical PTX, so the tile did not reach the compile
             and the census is one kernel compared with itself."
verdict_line G4 CLAIM "$(( g4_fail == 0 ))" \
  "wgmma appears in exactly the arms BLOCK_M % 64 == 0 names" \
  "$(( narms - g4_fail ))/$narms arms match the prediction" "all $narms" \
  "the instruction is NOT selected by the tile height alone at fixed num_warps,
             which is a result: Triton's predicate as read from supportMMA does not
             describe what this build emits."

echo
if (( g1_fail || g2_fail || distinct != narms )); then
  echo "READING IT. A validity gate failed. The census above may not be quoted,"
  echo "and STUDY.md item 3's loose end stays open."
  status=1
elif (( g4_fail == 0 )); then
  echo "READING IT. At a FIXED token count of $TOKENS, with num_warps pinned at"
  echo "$NUM_WARPS and every other tile knob equal, the warpgroup instruction"
  echo "appears exactly where BLOCK_M % 64 == 0 and nowhere else. THE INSTRUCTION"
  echo "IS SELECTED BY THE TILE, not by the batch size and not by the warp count."
  echo "That closes the loose end on STUDY.md item 3: the 2026-08-27 census saw"
  echo "the same switch through the config ladder, where the batch, the tile and"
  echo "num_warps all moved together."
  status=0
else
  echo "READING IT. The instruction did not follow the tile in $g4_fail arm(s)"
  echo "while every validity gate passed, so this is a measurement and not a"
  echo "broken run. Triton's warpgroup predicate as this study reads it does not"
  echo "describe this build; quote the table, not the predicate."
  status=1
fi

echo
echo "  PTX kept under $DUMP_DIR/bm<N>/, run logs beside it, summary at $SUMMARY."
echo "  *.ptx is gitignored at any depth on purpose, so the raw dumps leave a pod"
echo "  as a tarball and the table above leaves as text."
exit "$status"
