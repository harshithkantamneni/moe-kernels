#!/usr/bin/env python
"""Is `alpha` set by CONCURRENCY or by PROGRAM ORDER? Vary occupancy, hold the swizzle.

    python scripts/occupancy_vs_swizzle.py --audit      # the corpus, off GPU
    python scripts/occupancy_vs_swizzle.py --self-test  # three planted worlds
    python scripts/occupancy_vs_swizzle.py --dry-run    # the pod plan and its cost
    python scripts/occupancy_vs_swizzle.py --run        # the pod run
    python scripts/occupancy_vs_swizzle.py --replay DIR # re-report a finished run

WHY THIS EXISTS. Reuse-distance analysis is the standard way to predict cache
behaviour and it is what TileSight (arXiv:2607.22432) uses. Applied to this
kernel it says: the Triton swizzle groups `GROUP_SIZE_M` consecutive row-tiles
so they consume one weight block before the schedule moves on, so the weight
re-read fraction should fall roughly as `1/G`. At G=64 that predicts
`alpha_b ~ 0.016`. The published corpus measures 0.67 to 0.73. Capping the reuse
at tiles-per-expert -- which is the honest cap, since two M-tiles of DIFFERENT
experts share no weights at all -- only moves the prediction to about 0.12. The
standard predictor is wrong here by a factor of ten to forty, and it is wrong in
the direction that says the swizzle should have fixed this and it did not.

THE CANDIDATE EXPLANATION, which is the whole experiment. Reuse-distance
analysis assumes SEQUENTIAL execution: touch a block, touch other data, come
back, and count what was touched in between. A GPU does not execute that way.
132 SMs each hold several thread blocks, every one of them streaming its own
slices through ONE shared L2. Per CTA of this tiling the K loop streams
`(BM + BN) K b` bytes -- 1.31 MiB weighted over the two GEMMs at mixtral's
geometry -- so at four resident blocks per SM the recent history is
132 x 4 x 1.31 = 693 MiB against 50 MB of L2. Reordering a queue does not help
when its contents overflow the cache thirteen times over whatever the order.

If that is right, `alpha` tracks OCCUPANCY -- concurrently resident blocks --
and NOT `GROUP_SIZE_M`. It would also explain the study's cross-card null: both
cards are saturated, so a 1.25x L2 capacity difference (40 MB against 50 MB)
changes almost nothing.

THE LEVERS, and why these two. vLLM's config dict exposes exactly six knobs and
only two of them move residency without moving the tiling:

  * `num_stages` moves it. Triton multi-buffers the K loop, so one CTA holds
    `num_stages (BM BK + BK BN) b` bytes of shared memory -- 16 KiB per stage at
    this geometry -- and the card's per-SM shared memory then divides down to a
    resident-block count. On an H200 that is 6, 4, 3, 2 blocks per SM at 2, 3, 4
    and 5 stages; on an A100 it is 4, 3, 2, 2. THE TILE GEOMETRY IS UNTOUCHED:
    BM, BN and BK are pinned, so the bytes one CTA reads, the FLOPs it does and
    every term of the traffic model are identical across the ladder.
  * `num_warps` does NOT move it, and that is what makes it the control. At
    fixed shared memory the smem limit binds before the thread-slot limit, so
    four warps and eight warps give the SAME resident-block count and the same
    concurrent data footprint -- while halving or doubling the resident WARPS,
    i.e. the memory-level parallelism. The concurrency hypothesis is about
    FOOTPRINT, so it predicts alpha does not move here. A model that only says
    "more parallelism, more misses" predicts it does.

So the design is one lever that changes residency, one that changes warp-level
parallelism at fixed residency, and one that changes program order at fixed
both. Three arms, one fit, one verdict.

WHAT alpha IS HERE, stated because the name has caused trouble in this study. A
ladder fit at BLOCK_M returns

    alpha_fitted = alpha_b + alpha_a (BM / BN) + BM / K

and not the weight-side miss fraction alone. Every setting in this experiment
runs at the SAME BM, BN and K, so `alpha_a (BM/BN)` and `BM/K` are IDENTICAL
constants across the whole grid and cancel exactly out of every difference this
script computes. That is the reason the experiment is a comparison of settings
and never a level: the level carries two terms this design cannot separate, and
the differences carry only the cache term, which is the one under test.

WHY BLOCK_M=64 AND NOT 128. 128 is the production-relevant tile and is useless
here: its cap sits ON the ridge, the memory and compute branches are the same
line to about 1%, and `fit_ladder` discards the memory branch outright, so there
is no alpha at 128 to watch move. At BLOCK_M=64 the cap is 2 BM / (alpha b) ~ 71
FLOP/byte against calibrated ridges of 145.8 and 162.8, so the memory branch is
the steeper line, every tread is memory bound, and alpha is identifiable on 16
treads. The mechanism question -- what sets the re-read fraction -- is not a
question about a particular tile height, so it is asked where it can be answered.

THE THREE OUTCOMES, all of them reportable:

  CONCURRENCY   alpha moves with resident blocks and not with GROUP_SIZE_M.
                The standard predictor does not transfer to this regime, which
                is a correction to a published method.
  PROGRAM ORDER alpha moves with GROUP_SIZE_M and not with residency. The
                predictor transfers and the 40x gap is something else.
  NEITHER       a null. It is a legitimate result and is printed as one rather
                than forced into a branch: it would say the re-read fraction is
                fixed by something neither knob reaches -- DRAM scheduling, the
                replacement policy, or the sector granularity -- and it would
                retire both models at once.

Every one of those needs the VALIDITY gates to have passed first, and the
loudest of them is that the residency ladder actually happened: a grid whose
computed resident-block count is the same at every setting has swept nothing,
and would report a flat alpha as evidence for program order.

WHAT IS IMPORTED AND NEVER REIMPLEMENTED. The ladder fit, the compute
reference, its level checks, the tile-resource refusals, the timing loop, the
override hook and the balanced routing all come from
`scripts/block_m_crossing_sweep.py` by path. This script has to be scored by the
fit the study publishes; a private copy would drift and every number here would
become unattributable. What is NEW here is the residency arithmetic, which that
file does not have: it assumes one resident CTA per SM when it counts waves, and
says so.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402


def _load_sweep():
    """Load `block_m_crossing_sweep` BY PATH, and name what is missing.

    `scripts/` is not a package, so a bare import works only when this file is
    the entry point and fails silently when a test loads it by path. The symbol
    and SIGNATURE checks are not defensive noise: that file is under active
    edit by another workstream, and a rename must produce a sentence on a
    laptop rather than a TypeError thirty seconds into a metered pod session.
    """
    spec = importlib.util.spec_from_file_location(
        "block_m_crossing_sweep", ROOT / "scripts" / "block_m_crossing_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    needed = ("FIXED", "MIN_MEMORY_TREADS", "PARALLEL_BRANCH_TOLERANCE",
              "SMEM_PER_BLOCK_BYTES", "MAX_REGISTERS_PER_THREAD",
              "activation_slope_ms", "arm_triton_cache", "balanced_ids",
              "compute_reference", "count_new", "find_override", "fit_ladder",
              "ladder_points", "make_cell", "missing_gpu_stack", "model_ms",
              "results_root", "rows_quantum", "scaled_iters", "tile_resources",
              "time_call", "tokens_for_rows", "weight_bytes_per_expert")
    missing = [n for n in needed if not hasattr(module, n)]
    if missing:
        raise SystemExit(
            "scripts/block_m_crossing_sweep.py no longer exports "
            f"{', '.join(missing)}. This experiment is deliberately scored by "
            "that file's fit rather than a private copy, so the two move "
            "together. Re-point the import; do not fork the fit.")
    import inspect
    for name, required in (("compute_reference", ("cfg", "ridge",
                                                  "bandwidth_gbps", "b",
                                                  "pinned", "capability")),
                           ("fit_ladder", ("block_m",)),
                           ("tile_resources", ("pinned", "block_m",
                                               "dtype_bytes", "capability")),
                           ("make_cell", ("sm_count", "block_n"))):
        params = inspect.signature(getattr(module, name)).parameters
        gone = [p for p in required if p not in params]
        if gone:
            raise SystemExit(
                f"block_m_crossing_sweep.{name} no longer takes "
                f"{', '.join(gone)}. That file is under active edit and this "
                "one calls into it on the pod path; re-check the call sites in "
                "analyse and measure before spending GPU time.")
    return module


SWEEP = _load_sweep()

MIN_MEMORY_TREADS = SWEEP.MIN_MEMORY_TREADS

# --------------------------------------------------------------------------
# The numbers this experiment is arguing about, all stated before any code.
# --------------------------------------------------------------------------

#: The tile the ladder is fitted at. NOT 128: see the module docstring. At 64
#: the cap `2 BM / (alpha b)` is ~71 FLOP/byte against calibrated ridges of
#: 145.8 (A100) and 162.8 (H200), so the memory branch is the steeper line and
#: every tread is memory bound, which is the condition alpha is identifiable
#: under.
SUBJECT_BLOCK_M = 64

#: The compute reference. `C = 2 BM N / peak` is proportional to BLOCK_M with no
#: free parameter, so one compute-bound ladder gives the compute branch at every
#: block size, and membership at 64 is decided against it rather than by a split
#: search. Measured PER SETTING, because `C` depends on the achieved compute
#: rate and num_warps and num_stages both move that.
REFERENCE_BLOCK_M = 256

#: The occupancy ladder. Each value is a `num_stages`, and each `num_stages`
#: buys `(BM BK + BK BN) b` = 16 KiB of shared memory per CTA at this geometry,
#: which the card's per-SM shared memory then divides down to a resident-block
#: count. 5 is the top: at 6 stages the BLOCK_M=256 REFERENCE needs 240 KiB per
#: CTA, past both cards' per-block ceiling, and a setting with no reference has
#: no alpha to contribute.
DEFAULT_STAGES = (2, 3, 4, 5)

#: The warp control, run at these `num_stages`. num_warps changes resident WARPS
#: without changing resident BLOCKS while the shared-memory limit binds, which
#: is the whole point of it: same concurrent data footprint, half or double the
#: memory-level parallelism.
#:
#: 2 is excluded and not by taste. Triton's warpgroup predicate wants
#: `num_warps % 4 == 0` at `BLOCK_M % 64 == 0`, and at num_warps=2 the
#: BLOCK_M=256 reference's fp32 accumulator needs 256 registers per thread
#: against a hardware maximum of 255, so the reference would spill and its time
#: would not be the time of this tiling.
DEFAULT_CONTROL_WARPS = (4,)
CONTROL_WARP_STAGES = (2, 3)

#: The swizzle arm. `GROUP_SIZE_M` is a `tl.constexpr` that changes only the map
#: from program id to `(pid_m, pid_n)`: same grid, same tiles, same traffic
#: under the reuse-distance model, different order.
DEFAULT_GROUPS = (1, 8, 16, 64)

#: The setting every arm shares, so all three arms are anchored on one measured
#: ladder rather than on three.
BASE_STAGES, BASE_WARPS, BASE_GROUP = 3, 8, 1

#: How much of the concurrency model's own predicted alpha swing has to arrive
#: before "alpha moved with occupancy" is allowed to be said. Half: the model is
#: a one-parameter LRU caricature and the study has no business gating on its
#: exact value, but a mechanism that predicts a 0.09 swing and delivers under
#: 0.045 is not the mechanism.
OCCUPANCY_SWING_FRACTION = 0.5

#: The floor under that threshold, in units of the fitted alpha's own replicate
#: spread. A swing inside 3 sigma of the noise is not a swing whatever a model
#: predicted.
OCCUPANCY_SIGMA = 3.0

#: How much slack the reuse-distance prediction gets. `alpha(G) / alpha(1)` is
#: predicted at `1 / min(G, tiles per expert)`; the gate passes at twice that,
#: so the predictor is given a factor of two before it is called wrong. The
#: published corpus sits at 0.69-0.78 against a gate near 0.24.
ORDER_RATIO_TOLERANCE = 2.0

#: Distinct resident-block counts the occupancy ladder must actually produce,
#: and the span it must cover. NON-VACUITY: a grid that computes the same
#: residency at every setting has swept nothing, and a flat alpha over it would
#: read as evidence for program order when it is evidence of nothing.
MIN_RESIDENCY_LEVELS = 3
MIN_RESIDENCY_SPAN = 2.0

#: A setting whose treads move by more than this between repeats is not a
#: measurement of a setting, it is a measurement of the pod.
MAX_REPLICATE_SPREAD = 0.02

#: Treads a setting's ladder must classify as memory bound before its alpha may
#: enter a verdict. Imported, not chosen: it is the study's own bar, and two
#: points make a line with no residual.
MIN_SETTINGS_FOR_VERDICT = 3

VALIDITY = "VALIDITY"
CLAIM = "CLAIM"

VERDICT_CONCURRENCY = "CONCURRENCY"
VERDICT_ORDER = "PROGRAM ORDER"
VERDICT_BOTH = "BOTH"
VERDICT_NEITHER = "NEITHER (null)"
VERDICT_UNREADABLE = "UNREADABLE"

NO_CARD_SLUG = "nocard"


# --------------------------------------------------------------------------
# The card, as an occupancy calculator sees it.
# --------------------------------------------------------------------------

#: PER-SM shared memory, in bytes, by compute capability. NOT the per-block
#: opt-in ceiling, which `block_m_crossing_sweep.SMEM_PER_BLOCK_BYTES` already
#: holds and which answers a different question: that one asks whether ONE CTA
#: fits, this one asks how MANY fit. Both are needed and confusing them silently
#: triples the residency on an H200.
#:
#: UNKNOWN CAPABILITIES ARE NOT DEFAULTED. A missing entry makes residency
#: unknown, never 1 and never 32, because a guessed occupancy is the one number
#: this whole experiment is swept over.
SMEM_PER_SM_BYTES: dict[tuple[int, int], int] = {
    (7, 0): 98304,      # V100
    (7, 5): 65536,      # T4
    (8, 0): 167936,     # A100, 164 KiB
    (8, 6): 102400,     # A10 / A40 / RTX 30
    (8, 9): 102400,     # L4 / L40S / RTX 40
    (9, 0): 233472,     # H100 / H200, 228 KiB
    (10, 0): 233472,    # B200
}

#: Threads one SM may hold, which divided by the CTA's thread count is the
#: second residency limit.
MAX_THREADS_PER_SM: dict[tuple[int, int], int] = {
    (7, 0): 2048, (7, 5): 1024, (8, 0): 2048, (8, 6): 1536,
    (8, 9): 1536, (9, 0): 2048, (10, 0): 2048,
}

#: CTAs one SM may hold regardless of what they need. The third limit, and the
#: one that binds for very small blocks.
MAX_BLOCKS_PER_SM: dict[tuple[int, int], int] = {
    (7, 0): 32, (7, 5): 16, (8, 0): 32, (8, 6): 16,
    (8, 9): 24, (9, 0): 32, (10, 0): 32,
}

#: Shared memory the driver takes off the top for every resident thread block,
#: on Volta and later. 1 KiB. Small, and it is exactly what turns 233472/49152
#: from 4.75 into 4 at three stages -- which is the difference between a
#: residency ladder of 6/4/3/2 and one of 7/4/3/2.
RESERVED_SMEM_PER_BLOCK = 1024

#: The L2 a laptop is allowed to assume, and it is labelled every time it is
#: used. 50 MB is the published H100/H200 figure; the A100's is 40 MB. A
#: MEASURED run reads `L2_cache_size` off the attached device and REFUSES when
#: it cannot, for the same reason `resolve_ridge` refuses: this study has
#: already published seven reports scored against another machine's ceiling.
HYPOTHESIS_L2_BYTES = 50 * 1000 * 1000
HYPOTHESIS_L2_SOURCE = (
    "HYPOTHESIS: the published H100/H200 50 MB L2, which belongs to no "
    "attached device")

#: SM count a laptop is allowed to assume, labelled the same way.
HYPOTHESIS_SM_COUNT = 132
HYPOTHESIS_CAPABILITY = (9, 0)


@dataclass(frozen=True)
class CardLimits:
    """Everything the residency arithmetic needs, and where each number is from.

    Provenance travels with the numbers because three of the four decide the
    x axis of this whole experiment. A residency ladder computed against
    another card's shared memory is a ladder with the wrong rungs, and it would
    still plot.
    """

    capability: tuple[int, int]
    smem_per_sm: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    sm_count: int
    l2_bytes: int
    source: str

    def line(self) -> str:
        return (f"sm_{self.capability[0]}{self.capability[1]}  "
                f"{self.smem_per_sm / 1024:.0f} KiB smem/SM  "
                f"{self.max_threads_per_sm} threads/SM  "
                f"{self.max_blocks_per_sm} blocks/SM  {self.sm_count} SMs  "
                f"{self.l2_bytes / 1e6:.0f} MB L2   [{self.source}]")


class CardUnavailable(RuntimeError):
    """No occupancy limits this run is entitled to use, and none may be guessed.

    Raised rather than defaulted. Residency IS the swept axis here; a run that
    computes it from a capability table entry that does not exist has swept a
    quantity it invented.
    """


def card_limits(capability, sm_count: int | None, l2_bytes: int | None,
                source: str) -> CardLimits:
    """Assemble the limits, or refuse and say which one is missing."""
    if capability is None:
        raise CardUnavailable(
            "no compute capability for this run, so resident blocks per SM "
            "cannot be computed and the swept axis does not exist. Pass "
            "--capability MAJOR.MINOR, or run --dry-run, which assumes "
            f"sm_{HYPOTHESIS_CAPABILITY[0]}{HYPOTHESIS_CAPABILITY[1]} and says so.")
    cap = tuple(capability)
    if cap not in SMEM_PER_SM_BYTES:
        raise CardUnavailable(
            f"sm_{cap[0]}{cap[1]} is not in this file's per-SM resource "
            "tables, so resident blocks per SM is unknown. It is NOT defaulted: "
            "residency is the axis this experiment sweeps, and a guessed "
            "per-SM shared memory moves every rung of it. Add the row from the "
            "CUDA occupancy tables, or state the limits on the command line.")
    if not sm_count:
        raise CardUnavailable(
            "no SM count, so the concurrent footprint cannot be formed. Pass "
            "--sm-count.")
    if not l2_bytes:
        raise CardUnavailable(
            "no L2 capacity for this device, so the concurrency model has no "
            "denominator and its predicted alpha swing cannot be stated before "
            "the measurement. Pass --l2-bytes, or run --dry-run, which assumes "
            f"{HYPOTHESIS_L2_BYTES / 1e6:.0f} MB and says so.")
    return CardLimits(cap, SMEM_PER_SM_BYTES[cap], MAX_THREADS_PER_SM[cap],
                      MAX_BLOCKS_PER_SM[cap], sm_count, l2_bytes, source)


@dataclass(frozen=True)
class Residency:
    """Resident thread blocks per SM for one setting, and which limit bound.

    COMPUTED, not assumed, and computed from the three limits that can bind:

      * SHARED MEMORY. `num_stages (BM BK + BK BN) b` per CTA plus the driver's
        1 KiB reservation, into the SM's shared memory. This is the one the
        experiment steers with.
      * THREAD SLOTS. `32 num_warps` threads per CTA into the SM's thread
        capacity. This is the one `num_warps` would steer with if it bound --
        and the design is built so it does not, which is what makes num_warps a
        control instead of a second treatment.
      * CTA SLOTS, a flat per-architecture cap.

    THE REGISTER LIMIT IS NOT MODELLED and its absence is a stated bound, not an
    oversight: the per-thread register count is decided by ptxas and is not
    knowable from the pinned constants. It can only LOWER residency, so every
    number here is an UPPER BOUND, and `--probe-kernel` reads Triton's own
    reported shared memory and register count back off the compiled kernel to
    check the bound where the platform allows it. A run that cannot probe says
    so and its residency column is labelled a bound.
    """

    resident_blocks: int
    by_smem: int
    by_threads: int
    by_blocks: int
    smem_per_block: int
    binding: str

    def line(self) -> str:
        return (f"{self.resident_blocks} blocks/SM "
                f"(smem {self.by_smem}, threads {self.by_threads}, "
                f"cta {self.by_blocks}; {self.binding} binds) at "
                f"{self.smem_per_block / 1024:.0f} KiB/CTA")


def residency(pinned: dict, block_m: int, b: int, limits: CardLimits
              ) -> Residency:
    """Resident blocks per SM, from the pinned constants and the card's limits.

    Pure arithmetic, so `--dry-run` on a laptop prints the same ladder the pod
    will sweep -- which is the point: the predictions have to be registered with
    numbers, and the x axis is one of them.
    """
    res = SWEEP.tile_resources(pinned, block_m, b, limits.capability)
    per_block = res.smem_bytes + RESERVED_SMEM_PER_BLOCK
    by_smem = limits.smem_per_sm // per_block if per_block else 0
    by_threads = limits.max_threads_per_sm // (32 * pinned["num_warps"])
    by_blocks = limits.max_blocks_per_sm
    resident = min(by_smem, by_threads, by_blocks)
    binding = min((by_smem, "smem"), (by_threads, "threads"),
                  (by_blocks, "cta"), key=lambda t: t[0])[1]
    return Residency(resident, by_smem, by_threads, by_blocks,
                     res.smem_bytes, binding)


# --------------------------------------------------------------------------
# The two models, each stated as a number before anything is measured.
# --------------------------------------------------------------------------

def gemm_shapes(cfg) -> tuple[tuple[int, int], tuple[int, int]]:
    """`(K, N)` for the up GEMM and the down GEMM, in elements.

    The fused layer is two grouped GEMMs with different shapes and the per-CTA
    stream differs between them by 3.5x at mixtral, so a footprint that quotes
    only the first is wrong by a factor the experiment is trying to resolve.
    """
    return ((cfg.hidden_size, 2 * cfg.intermediate_size),
            (cfg.intermediate_size, cfg.hidden_size))


def per_cta_stream_bytes(cfg, block_m: int, block_n: int, b: int) -> float:
    """Bytes ONE CTA streams over its whole K loop, weighted over both GEMMs.

    A CTA at `(pid_m, pid_n)` walks the full K extent, reading `BM x K` of
    activations and `K x BN` of weights: `(BM + BN) K b` bytes. That is the unit
    of "recent history" the concurrency argument is made in -- one CTA-lifetime
    of streaming -- and it is weighted by how many CTAs each GEMM launches,
    because the up GEMM launches `2F/BN` N-tiles against the down GEMM's `H/BN`
    and at mixtral that is seven to one.
    """
    total = weight = 0.0
    for k, n in gemm_shapes(cfg):
        ctas = math.ceil(n / block_n)
        total += ctas * (block_m + block_n) * k * b
        weight += ctas
    return total / weight if weight else 0.0


def concurrent_footprint_bytes(cfg, block_m: int, block_n: int, b: int,
                               resident_blocks: int, sm_count: int) -> float:
    """`SMs x resident blocks x one CTA-lifetime of streaming`.

    The quantity the concurrency hypothesis says sets alpha. At mixtral's
    geometry with BM=BN=64 in bf16 it is 1.31 MiB per CTA, so 132 SMs at four
    resident blocks put 693 MiB through a 50 MB L2 -- thirteen times over,
    BEFORE program order gets a vote, which is the sentence this experiment
    exists to test.
    """
    return sm_count * resident_blocks * per_cta_stream_bytes(cfg, block_m,
                                                             block_n, b)


def alpha_concurrency(footprint_bytes: float, l2_bytes: float) -> float:
    """`1 - min(1, L2 / footprint)`: the concurrency model's predicted alpha.

    A one-parameter LRU caricature: a weight line survives to its re-read with
    probability equal to the fraction of the concurrent working set the cache
    can hold. It is not offered as a calibrated predictor and no gate is scored
    on its LEVEL -- only on the SWING it predicts across the residency ladder,
    which is what the experiment can actually see. Its level being close to the
    published 0.92 at four resident blocks is a coincidence worth noting and
    nothing to lean on.
    """
    if footprint_bytes <= 0 or l2_bytes <= 0:
        return 1.0
    return 1.0 - min(1.0, l2_bytes / footprint_bytes)


def alpha_order(alpha_base: float, group_m: int, tiles_cap: float) -> float:
    """Reuse distance's predicted alpha at `GROUP_SIZE_M = G`.

    `alpha(G) = alpha(1) / min(G, tiles per expert)`. The cap is not a
    concession, it is the mechanism: two M-tiles of DIFFERENT experts share no
    weight block at all, so grouping past one expert's tile count buys nothing
    and the honest form of the prediction says so. Without the cap the
    prediction at G=64 is `alpha(1)/64`, which is the 0.016 the measurement
    misses by forty.
    """
    return alpha_base / max(1.0, min(float(group_m), tiles_cap))


# --------------------------------------------------------------------------
# The grid.
# --------------------------------------------------------------------------

ARM_OCCUPANCY = "occupancy"
ARM_WARPS = "warp-control"
ARM_SWIZZLE = "swizzle"


def _arms(setting) -> str:
    """The arms a setting belongs to, short enough to stay in its column.

    A setting can be in all three -- the base is -- and the untruncated string
    overflows the column and shifts every number on the row, which is exactly
    how a table stops being readable at the moment it matters.
    """
    short = {ARM_OCCUPANCY: "occ", ARM_WARPS: "warp", ARM_SWIZZLE: "swz"}
    return "/".join(short.get(a, a) for a in setting.arms)


@dataclass(frozen=True)
class Setting:
    """One point of the grid: a `num_stages`, a `num_warps`, a `GROUP_SIZE_M`.

    Everything else -- BLOCK_M, BLOCK_N, BLOCK_K, model, dtype, routing, the
    row ladder -- is pinned across the whole experiment, which is what makes
    `alpha_a (BM/BN) + BM/K` a common constant that cancels out of every
    difference this script reports.
    """

    num_stages: int
    num_warps: int
    group_m: int
    arms: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"s{self.num_stages}w{self.num_warps}g{self.group_m}"

    def pinned(self, block_n: int, block_k: int) -> dict:
        return dict(SWEEP.FIXED, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
                    GROUP_SIZE_M=self.group_m, num_warps=self.num_warps,
                    num_stages=self.num_stages)


def build_settings(stages, control_warps, control_stages, groups,
                   base_stages: int = BASE_STAGES, base_warps: int = BASE_WARPS,
                   base_group: int = BASE_GROUP) -> list[Setting]:
    """The three arms, de-duplicated onto one shared base setting.

    The base setting belongs to all three arms rather than being measured three
    times: every arm's contrast is against the same measured ladder, so a base
    that drifted would move all three contrasts together and be visible, rather
    than moving one and looking like an effect.
    """
    arms: dict[tuple[int, int, int], set[str]] = {}
    for s in stages:
        arms.setdefault((s, base_warps, base_group), set()).add(ARM_OCCUPANCY)
    for s in control_stages:
        for w in (base_warps, *control_warps):
            arms.setdefault((s, w, base_group), set()).add(ARM_WARPS)
    for g in groups:
        arms.setdefault((base_stages, base_warps, g), set()).add(ARM_SWIZZLE)
    return [Setting(s, w, g, tuple(sorted(a)))
            for (s, w, g), a in sorted(arms.items())]


def ladder_rows(cfg, block_m: int, r_max: int) -> list[int]:
    """Exactly-full tile stacks only: `r = n BM`, zero padding, one per tread.

    REFUSES when the model's routing cannot form an integer token count at
    `n BM` rather than nudging the row count: a nudged row is not a full tile
    stack and a fit over partly filled treads is a fit over padding.
    """
    q = SWEEP.rows_quantum(cfg)
    rows = [n * block_m for n in range(1, r_max // block_m + 1)]
    bad = [r for r in rows if r % q]
    if bad:
        raise SystemExit(
            f"{cfg.num_experts} experts at top-k {cfg.top_k} need rows per "
            f"expert to be a multiple of {q}, and {bad[0]} is not. This model "
            f"cannot form an exactly full tile stack at BLOCK_M={block_m}; "
            "choose another --model.")
    if not rows:
        raise SystemExit(f"--r-max {r_max} is below one tile at "
                         f"BLOCK_M={block_m}; nothing to measure.")
    return rows


def median_tiles_per_expert(rows: list[int], block_m: int) -> float:
    """The reuse cap the order model gets, in tiles, over the ladder as swept.

    The cap is a median over a ladder and not a threshold, and the prediction
    that uses it says so where it prints.
    """
    return statistics.median([r / block_m for r in rows])


# --------------------------------------------------------------------------
# Gates.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """One pre-registered prediction and the number that settled it.

    `passed=None` prints UNKNOWN and never PASS: a check that examined nothing
    also reports zero failures, and this study has already published an arm
    whose eight blank cells read as a boring null when they were a broken
    reference. `invalidates` is required on a VALIDITY gate and says what may
    not be quoted if it fails, because a failed gate whose consequence is
    unstated gets read as a warning.
    """

    kind: str
    name: str
    prediction: str
    rule: str
    passed: bool | None
    observed: str
    invalidates: str = ""
    lines: list[str] = field(default_factory=list)

    def render(self) -> list[str]:
        tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}[self.passed]
        out = [f"[{tag}] {self.kind:8s} {self.name}  {self.prediction}",
               f"         gate: {self.rule}",
               f"         saw:  {self.observed}"]
        if self.passed is not True and self.invalidates:
            out.append(f"         a FAIL here invalidates: {self.invalidates}")
        out += [f"         {line}" for line in self.lines]
        return out

    def as_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name,
                "prediction": self.prediction, "rule": self.rule,
                "verdict": {True: "PASS", False: "FAIL",
                            None: "UNKNOWN"}[self.passed],
                "observed": self.observed, "invalidates": self.invalidates}


def render_gates(gates: list[Gate]) -> list[str]:
    out: list[str] = []
    for g in gates:
        out += g.render()
    npass = sum(1 for g in gates if g.passed is True)
    nfail = sum(1 for g in gates if g.passed is False)
    nunk = sum(1 for g in gates if g.passed is None)
    out += ["", f"{npass} PASS, {nfail} FAIL, {nunk} UNKNOWN"]
    return out


# --------------------------------------------------------------------------
# Predictions, registered with numbers before anything is measured.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Registered:
    """The two models evaluated on THIS plan, before a cell is timed.

    Held as a value rather than printed straight out so the gates are scored
    against exactly the numbers the report printed at the top, and so a test
    can assert that the two are the same object.
    """

    residency_by_setting: dict[str, Residency]
    footprint_by_setting: dict[str, float]
    concurrency_alpha: dict[str, float]
    occupancy_swing: float
    occupancy_threshold: float
    order_cap: float
    order_ratio: float
    order_gate: float
    #: `(floor, dead_lo, dead_hi)`: the window of alpha this design can read at
    #: all. See `identifiability_window`. Registered because P2's prediction has
    #: to be checked against it BEFORE the run: a predicted alpha inside the
    #: dead band cannot be measured here however clean the timings are.
    window: tuple[float, float, float]
    group_hi: int
    l2_bytes: int
    l2_source: str
    limits: CardLimits


def identifiability_window(ridge: float, b: int, block_m: int, treads: int
                           ) -> tuple[float, float, float]:
    """`(floor, dead_lo, dead_hi)`: which alphas this ladder can read at all.

    NOT a detail. The ladder fit reports alpha from the treads standing above
    the compute branch, and how many of those exist is fixed by the same
    identity everything else here turns on:

        B / C = alpha b ridge / (2 BM) = k alpha,      k = b ridge / (2 BM)

    and the memory prefix runs to `n* = k (1 - alpha) / (1 - k alpha)` treads.
    Two consequences, both of which have to be known before a prediction is
    registered against this design:

      * A FLOOR. Tread `n` is called memory bound when it stands above the
        compute branch BY THE MARGIN, so `1 + alpha(n-1) > n (1 + margin) / k`,
        and requiring that at `n = m` -- with `m` the study's
        `MIN_MEMORY_TREADS` -- gives

            alpha_floor = (m (1 + margin) / k - 1) / (m - 1)

        The fused layer's fixed cost cancels: it is on both sides of the
        comparison. At BLOCK_M=64 against an H200 ridge that is 0.102, and the
        capped reuse-distance prediction sits at 0.110 -- ABOVE it, so route one
        of P2 is available, but by only 8%. It is available because the subject
        ladder is swept 16 treads deep; at 4 treads it would not be.
      * A DEAD BAND. `fit_ladder` DISCARDS the memory branch when
        `|B/C - 1| <= PARALLEL_BRANCH_TOLERANCE`, because two branches within
        15% of each other are one line. That is `alpha` in
        `[(1 - tol)/k, (1 + tol)/k]`, which at BLOCK_M=64 on an H200 is
        0.334 to 0.452. An alpha landing in there is unmeasurable HERE however
        clean the timings are -- it is the same rejection that makes BLOCK_M=128
        useless for this question, arriving at a different alpha instead of at a
        different tile.

    Returned as three numbers so the report can print them and the gates can
    check a prediction against them instead of discovering the band afterwards.
    """
    if ridge <= 0 or block_m <= 0:
        return 0.0, 0.0, 0.0
    k = b * ridge / (2.0 * block_m)
    m = float(min(MIN_MEMORY_TREADS, max(2, treads)))
    margin = SWEEP.MEMORY_BRANCH_MARGIN
    floor = ((m * (1.0 + margin) / k - 1.0) / (m - 1.0)
             if k and m != 1.0 else 0.0)
    tol = SWEEP.PARALLEL_BRANCH_TOLERANCE
    return max(0.0, floor), (1.0 - tol) / k, (1.0 + tol) / k


def register(cfg, settings: list[Setting], limits: CardLimits, *, block_n: int,
             block_k: int, b: int, subject_rows: list[int], ridge: float,
             l2_source: str) -> Registered:
    """Evaluate both models on the plan, so every gate has a number in front."""
    res = {}
    foot = {}
    conc = {}
    for st in settings:
        r = residency(st.pinned(block_n, block_k), SUBJECT_BLOCK_M, b, limits)
        res[st.key] = r
        foot[st.key] = concurrent_footprint_bytes(
            cfg, SUBJECT_BLOCK_M, block_n, b, r.resident_blocks,
            limits.sm_count)
        conc[st.key] = alpha_concurrency(foot[st.key], limits.l2_bytes)
    occ_keys = [st.key for st in settings if ARM_OCCUPANCY in st.arms]
    swing = 0.0
    if occ_keys:
        lo = min(res[k].resident_blocks for k in occ_keys)
        hi = max(res[k].resident_blocks for k in occ_keys)
        at_lo = [conc[k] for k in occ_keys if res[k].resident_blocks == lo]
        at_hi = [conc[k] for k in occ_keys if res[k].resident_blocks == hi]
        swing = statistics.fmean(at_hi) - statistics.fmean(at_lo)
    cap = median_tiles_per_expert(subject_rows, SUBJECT_BLOCK_M)
    groups = [st.group_m for st in settings if ARM_SWIZZLE in st.arms]
    g_hi = max(groups) if groups else BASE_GROUP
    ratio = alpha_order(1.0, g_hi, cap)
    window = identifiability_window(ridge, b, SUBJECT_BLOCK_M,
                                    len(subject_rows))
    return Registered(res, foot, conc, swing,
                      OCCUPANCY_SWING_FRACTION * swing, cap, ratio,
                      ORDER_RATIO_TOLERANCE * ratio, window, g_hi,
                      limits.l2_bytes, l2_source, limits)


#: The published BLOCK_M=64 alpha at GROUP_SIZE_M=1, used ONLY to turn P2's
#: predicted RATIO into a predicted alpha so it can be checked against this
#: design's dead band before the run. It is a corpus median, it is labelled
#: wherever it prints, and no gate is scored on it.
CORPUS_ALPHA_AT_G1 = 0.93


def _window_verdict(reg: Registered) -> str:
    """One sentence: can P2 be settled by the ratio, or only by the collapse."""
    predicted = reg.order_ratio * CORPUS_ALPHA_AT_G1
    if predicted < reg.window[0]:
        return ("which is UNDER the floor, so a reuse-distance world shows up "
                "here as a collapse and not as a ratio.")
    if reg.window[1] <= predicted <= reg.window[2]:
        return ("which is INSIDE the dead band, so the fit would discard it "
                "and only route two can settle P2.")
    return (f"which is outside both, so P2 can be settled by the ratio "
            f"directly -- by {predicted / reg.window[0] - 1.0:.0%} over the "
            "floor, which is the ladder depth paying for itself.")


def k_of_window(window: tuple[float, float, float]) -> float:
    """Recover `k = b ridge / (2 BM)` from the dead band, so it prints once.

    The band's lower end is `(1 - tol) / k` by construction, so inverting it is
    exact rather than a second computation that could drift from the first.
    """
    lo = window[1]
    return (1.0 - SWEEP.PARALLEL_BRANCH_TOLERANCE) / lo if lo else 0.0


def predictions_text(reg: Registered, settings: list[Setting]) -> str:
    """The registered predictions, with this plan's own numbers in them."""
    rows = []
    for st in sorted(settings, key=lambda s: (s.num_stages, s.num_warps,
                                              s.group_m)):
        r = reg.residency_by_setting[st.key]
        rows.append(
            f"    {st.key:12s} {_arms(st):24.24s} "
            f"{r.smem_per_block / 1024:5.0f} KiB/CTA  "
            f"{r.resident_blocks:2d} blk/SM ({r.binding:7s})  "
            f"{reg.footprint_by_setting[st.key] / 2**20:8.0f} MiB in flight  "
            f"alpha_conc {reg.concurrency_alpha[st.key]:.3f}")
    first = settings[0].key
    per_cta = (reg.footprint_by_setting[first]
               / max(1, reg.limits.sm_count
                     * reg.residency_by_setting[first].resident_blocks))
    return "\n".join([
        "## Predictions, registered before anything is measured", "",
        "THE GRID, and the residency it computes to on the attached card:",
        f"    {reg.limits.line()}",
        f"    L2 provenance: {reg.l2_source}",
        f"    one CTA streams {per_cta / 2**20:.2f} MiB over its K loop, "
        "weighted over both GEMMs",
        "", *rows, "",
        "P1  CONCURRENCY. alpha rises with resident blocks per SM. The LRU",
        f"    caricature predicts a swing of {reg.occupancy_swing:+.3f} in alpha "
        "between the top and",
        "    bottom rungs of the residency ladder above. The gate takes half of",
        f"    that, {reg.occupancy_threshold:.3f}, floored at "
        f"{OCCUPANCY_SIGMA:.0f} sigma of the fitted alpha's own",
        "    replicate spread, which is measured and not assumed. A FAIL means",
        "    concurrency does not set alpha and the cross-card null was not",
        "    saturation.",
        "P2  PROGRAM ORDER. Reuse distance predicts "
        f"alpha(G={reg.group_hi}) / alpha(G=1) = "
        f"1/min(G, {reg.order_cap:.1f} tiles",
        f"    per expert) = {reg.order_ratio:.3f}; uncapped it would be "
        f"{1.0 / reg.group_hi:.4f}. The gate gives the",
        f"    predictor a factor of {ORDER_RATIO_TOLERANCE:.0f} and passes at "
        f"{reg.order_gate:.3f}. The published corpus",
        "    sits at 0.69-0.78, so this is expected to FAIL, and a FAIL here is",
        "    the interesting outcome: it says the standard predictor does not",
        "    transfer to this regime.",
        "P3  THE CONTROL. num_warps changes resident WARPS at fixed resident",
        "    BLOCKS, so the concurrent data footprint is unchanged and P1's",
        "    mechanism predicts NO movement. The gate is that the warp contrast",
        "    is smaller than the occupancy contrast. A FAIL says the residency",
        "    reading is confounded by something num_warps changes -- latency",
        "    hiding, register pressure, scheduling -- and P1 cannot be read as",
        "    a footprint effect.",
        "P4  THE NULL IS A RESULT. If P1 and P2 both FAIL the verdict is",
        f"    {VERDICT_NEITHER}, printed as such and not forced into a branch.",
        "    It would retire both models at once and point at DRAM scheduling,",
        "    the replacement policy or sector granularity instead.",
        "P2b WHAT THIS DESIGN CAN READ AT ALL, checked against P2 before the run.",
        "    The memory prefix runs n* = k(1-alpha)/(1-k alpha) treads with "
        f"k = b ridge / 2BM = {k_of_window(reg.window):.3f}, so:",
        f"      floor      alpha < {reg.window[0]:.3f} leaves fewer than "
        f"{MIN_MEMORY_TREADS} treads above the compute branch",
        "                 and the setting reports nothing;",
        f"      dead band  alpha in [{reg.window[1]:.3f}, {reg.window[2]:.3f}] "
        "is DISCARDED by the fit, because the two",
        "                 branches are then within "
        f"{SWEEP.PARALLEL_BRANCH_TOLERANCE:.0%} of each other and are one line.",
        f"    Taking a published alpha(1) near {CORPUS_ALPHA_AT_G1:.2f}, P2's "
        f"own prediction lands at {reg.order_ratio * CORPUS_ALPHA_AT_G1:.3f},",
        "    " + _window_verdict(reg),
        "    ROUTE TWO covers the case where a setting does fall in the dead "
        "band or under the",
        f"    floor: wide-G settings losing their memory branch while "
        f"G={BASE_GROUP} keeps its is scored as",
        "    a PASS for P2, and the report says which route it took.",
        "P5  THE LEVEL IS NOT PREDICTED, only the differences. alpha_fitted =",
        "    alpha_b + alpha_a (BM/BN) + BM/K, and this grid pins BM, BN and K,",
        "    so the two geometric terms are identical constants at every",
        "    setting and cancel out of every contrast above. No gate here is",
        "    scored on an absolute alpha.",
    ])


# --------------------------------------------------------------------------
# The plan.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """Everything the pod run will do, computable on a laptop."""

    model: str
    dtype: str
    settings: list[Setting]
    subject_rows: list[int]
    reference_rows: list[int]
    block_n: int
    block_k: int
    reps: int
    iters: int
    warmup: int
    cell_budget_ms: float
    estimated_seconds: float
    cells: int
    refused: dict[str, str]

    def lines(self, cfg) -> list[str]:
        out = [
            f"model        {self.model} E={cfg.num_experts} k={cfg.top_k} "
            f"{self.dtype}",
            f"pinned       BLOCK_SIZE_N={self.block_n} "
            f"BLOCK_SIZE_K={self.block_k}; BLOCK_SIZE_M is "
            f"{SUBJECT_BLOCK_M} (subject) and {REFERENCE_BLOCK_M} (reference)",
            f"subject      BLOCK_M={SUBJECT_BLOCK_M} at rows "
            f"{self.subject_rows[0]}..{self.subject_rows[-1]} "
            f"({len(self.subject_rows)} treads)",
            f"reference    BLOCK_M={REFERENCE_BLOCK_M} at rows "
            f"{self.reference_rows[0]}..{self.reference_rows[-1]} "
            f"({len(self.reference_rows)} treads), MEASURED PER SETTING because "
            "C depends on the achieved compute rate and both knobs move it",
            f"settings     {len(self.settings)}: "
            + ", ".join(s.key for s in self.settings),
            f"repeats      {self.reps} passes, settings shuffled inside each "
            "token count on the same tensors, so every contrast is paired",
            f"timing       {self.warmup} warmup + up to {self.iters} iters, cut "
            f"to keep a cell inside {self.cell_budget_ms:.0f} ms",
            f"cells        {self.cells} timings",
            f"estimate     {self.estimated_seconds:.0f} s of GPU at the model's "
            "own timings, excluding compiles and allocation",
        ]
        for key, why in sorted(self.refused.items()):
            out.append(f"REFUSED      {key}: {why}")
        return out


def build_plan(args, cfg, b: int, limits: CardLimits, *, alpha: float,
               ridge: float, bandwidth_gbps: float) -> Plan:
    """The grid, minus every setting whose tiles cannot physically run.

    A SETTING WITHOUT A REFERENCE IS DROPPED HERE, on the laptop, not diagnosed
    on the pod. The BLOCK_M=256 reference needs `num_stages x 40 KiB` of shared
    memory per CTA and both cards have a per-block ceiling; at 6 stages nothing
    fits, and at 5 the A100 does not. A setting whose reference spills produces
    a compute branch that is perfectly proportional and 40x too steep -- which
    is how 249.765 ms became this study's compute branch once already -- and
    every tread it classifies is then void.
    """
    settings = build_settings(args.stages, args.control_warps,
                              args.control_stages, args.groups)
    subject_rows = ladder_rows(cfg, SUBJECT_BLOCK_M, args.r_max)
    reference_rows = ladder_rows(cfg, REFERENCE_BLOCK_M, args.r_max)
    if len(reference_rows) < MIN_MEMORY_TREADS:
        raise SystemExit(
            f"--r-max {args.r_max} gives the BLOCK_M={REFERENCE_BLOCK_M} "
            f"reference only {len(reference_rows)} tread(s) and "
            f"`compute_reference` needs {MIN_MEMORY_TREADS} to qualify one. "
            f"Raise --r-max to at least {MIN_MEMORY_TREADS * REFERENCE_BLOCK_M}.")
    kept: list[Setting] = []
    refused: dict[str, str] = {}
    for st in settings:
        pinned = st.pinned(args.block_n, args.block_k)
        why = []
        for bm in (SUBJECT_BLOCK_M, REFERENCE_BLOCK_M):
            r = SWEEP.tile_resources(pinned, bm, b, limits.capability)
            if r.refusal:
                why.append(f"BLOCK_M={bm}: {r.refusal}")
        if why:
            refused[st.key] = "; ".join(why)
        else:
            kept.append(st)
    if not kept:
        raise SystemExit(
            "every setting in the grid is refused on this card's shared-memory "
            "or register limits, so there is nothing to measure. The refusals:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in sorted(refused.items())))
    total = 0.0
    cells = 0
    for _ in kept:
        for bm, rows in ((SUBJECT_BLOCK_M, subject_rows),
                         (REFERENCE_BLOCK_M, reference_rows)):
            for r in rows:
                ms = SWEEP.model_ms(cfg, r, bm, alpha=alpha, ridge=ridge,
                                    bandwidth_gbps=bandwidth_gbps, b=b)
                it = SWEEP.scaled_iters(ms, args.iters, args.cell_budget_ms)
                total += args.reps * ms * (args.warmup + it)
                cells += args.reps
    return Plan(args.model, args.dtype, kept, subject_rows, reference_rows,
                args.block_n, args.block_k, args.reps, args.iters, args.warmup,
                args.cell_budget_ms, total / 1e3, cells, refused)


# --------------------------------------------------------------------------
# Measurement.
# --------------------------------------------------------------------------

@dataclass
class Sample:
    """One timing of one tread of one setting in one repeat. The CSV row."""

    setting: str
    num_stages: int
    num_warps: int
    group_m: int
    block_m: int
    tiles: int
    rows_per_expert: int
    tokens: int
    rep: int
    ms_p50: float
    ms_min: float
    ms_stdev: float
    iters: int
    status: str = "ok"
    detail: str = ""


SAMPLE_FIELDS = list(Sample.__dataclass_fields__)


def append_sample(path: Path, sample: Sample) -> None:
    """One row, flushed. An abort costs the timing in flight and nothing else."""
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(asdict(sample))
        fh.flush()


def read_samples(path: Path) -> tuple[set[tuple[str, int, int, int]],
                                      list[Sample]]:
    """Timings already on disk, so a re-run resumes rather than repeats.

    Only the SUCCESSFUL ones count as done. A failed cell is usually a pod that
    lost its device, and a real failure fails again in milliseconds.
    """
    if not path.exists():
        return set(), []
    out: list[Sample] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Sample(
                setting=row["setting"], num_stages=int(row["num_stages"]),
                num_warps=int(row["num_warps"]), group_m=int(row["group_m"]),
                block_m=int(row["block_m"]), tiles=int(row["tiles"]),
                rows_per_expert=int(row["rows_per_expert"]),
                tokens=int(row["tokens"]), rep=int(row["rep"]),
                ms_p50=float(row["ms_p50"]), ms_min=float(row["ms_min"]),
                ms_stdev=float(row["ms_stdev"]), iters=int(row["iters"]),
                status=row.get("status", "ok"), detail=row.get("detail", "")))
    done = {(s.setting, s.block_m, s.tiles, s.rep)
            for s in out if s.status == "ok"}
    return done, out


def collapse(samples: list[Sample], setting: str, block_m: int
             ) -> tuple[list[tuple[int, float]], dict[int, list[float]],
                        float | None]:
    """Per-tread median across repeats, the repeats themselves, and the spread.

    The median across REPEATS rather than the single-pass median: a repeat is a
    fresh call at a fresh point in the pod's thermal history, and a single pass
    cannot tell a mechanism from a drift.
    """
    by: dict[int, list[float]] = {}
    for s in samples:
        if (s.setting == setting and s.block_m == block_m
                and s.status == "ok" and s.ms_p50 > 0):
            by.setdefault(s.tiles, []).append(s.ms_p50)
    points = [(n, statistics.median(v)) for n, v in sorted(by.items())]
    spreads = [statistics.pstdev(v) / statistics.median(v)
               for v in by.values() if len(v) > 1 and statistics.median(v) > 0]
    return points, by, statistics.median(spreads) if spreads else None


def token_plan(cfg, plan: Plan) -> list[tuple[int, list[tuple[int, int]]]]:
    """`(tokens, [(block_m, tiles)])`, so the inputs are built ONCE per batch.

    THE LOOP ORDER IS THE EXPERIMENT'S PAIRING. Every setting is timed at one
    token count back to back on the SAME tensors, in a shuffled order, before
    the batch moves on. Timing all of one setting's ladder and then the next
    puts the two settings at different points in the pod's thermal history and
    turns a drift into exactly the slope being fitted.
    """
    want: dict[int, list[tuple[int, int]]] = {}
    for bm, rows in ((SUBJECT_BLOCK_M, plan.subject_rows),
                     (REFERENCE_BLOCK_M, plan.reference_rows)):
        for r in rows:
            want.setdefault(SWEEP.tokens_for_rows(cfg, r), []).append(
                (bm, r // bm))
    return [(t, sorted(v)) for t, v in sorted(want.items())]


class KernelProbe:
    """Triton's OWN shared memory and register count for the kernel it compiled.

    WHY THIS EXISTS. `residency` computes resident blocks per SM from
    `num_stages (BM BK + BK BN) b`, which is a MODEL of what Triton allocates,
    and residency is the axis this whole experiment sweeps. If the real
    allocation differs -- an extra buffer, a different multi-buffering rule, a
    version that rounds -- every rung of the ladder is in the wrong place and
    the report would still plot. Triton knows the true number and puts it on the
    compiled kernel, so it is read rather than trusted.

    IT CAN ONLY REPORT, NEVER RAISE. This runs inside the metered loop, the
    attribute path has moved between vLLM versions, and a probe that threw would
    cost a pod session to learn nothing. Every failure returns a NOTE that names
    what was missing, and the gate that consumes it says UNKNOWN rather than
    PASS -- because a probe that examined nothing also reports zero
    disagreements.
    """

    def __init__(self) -> None:
        self.seen: set = set()
        self.note = ""
        self.by_setting: dict[str, dict] = {}

    def _kernel(self):
        import importlib
        for name in ("vllm.model_executor.layers.fused_moe.fused_moe",
                     "vllm.model_executor.layers.fused_moe"):
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            fn = getattr(mod, "fused_moe_kernel", None)
            if fn is not None and hasattr(fn, "cache"):
                return fn
        return None

    def _entries(self) -> dict:
        fn = self._kernel()
        if fn is None:
            self.note = ("vLLM exposes no fused_moe_kernel with a Triton cache "
                         "under either known path, so the compiled shared "
                         "memory could not be read")
            return {}
        out = {}
        for device in list(fn.cache.values()):
            if isinstance(device, dict):
                out.update(device)
        return out

    def record(self, key: str) -> None:
        """Attribute whatever compiled since the last call to `key`."""
        try:
            entries = self._entries()
        except Exception as exc:                        # noqa: BLE001
            self.note = f"probe raised {type(exc).__name__}: {exc}"
            return
        fresh = {k: v for k, v in entries.items() if k not in self.seen}
        self.seen.update(entries)
        for kernel in fresh.values():
            meta = getattr(kernel, "metadata", None)
            shared = getattr(meta, "shared", None)
            if shared is None:
                continue
            slot = self.by_setting.setdefault(
                key, {"shared": 0, "n_regs": 0, "n_spills": 0})
            slot["shared"] = max(slot["shared"], int(shared))
            slot["n_regs"] = max(slot["n_regs"],
                                 int(getattr(kernel, "n_regs", 0) or 0))
            slot["n_spills"] = max(slot["n_spills"],
                                   int(getattr(kernel, "n_spills", 0) or 0))
        if not self.by_setting and not self.note:
            self.note = ("nothing in the Triton cache carried a "
                         "metadata.shared, so the compiled shared memory is "
                         "unknown")


SMEM_PROBE_TOLERANCE = 0.10


def gate_compiled_smem(probe: dict[str, dict], note: str,
                       results: list[SettingResult]) -> Gate:
    """V9: does Triton's own shared memory agree with the residency model?

    A disagreement past `SMEM_PROBE_TOLERANCE` means the residency ladder's
    rungs are in the wrong places, which is not a detail -- it is the x axis.
    Spills are checked in the same gate because a spilled kernel's time is not
    the time of this tiling, and a spilled kernel still fits a straight line.
    """
    if not probe:
        return Gate(
            VALIDITY, "V9 compiled shared memory",
            "Triton allocated the shared memory the residency model assumed",
            f"every setting within {SMEM_PROBE_TOLERANCE:.0%} of "
            "num_stages (BM BK + BK BN) b, and zero register spills",
            None, f"not probed: {note or 'no probe was attempted'}",
            "nothing on its own, but it leaves the residency column a COMPUTED "
            "UPPER BOUND rather than a checked one: the register limit is not "
            "modelled and can only lower residency, so a report under a "
            "not-probed V9 may quote the ladder's ORDER and not its values")
    worst = 0.0
    spills = []
    for r in results:
        got = probe.get(r.setting.key)
        if not got or not got.get("shared"):
            continue
        rel = abs(got["shared"] / r.residency.smem_per_block - 1.0)
        worst = max(worst, rel)
        if got.get("n_spills"):
            spills.append(f"{r.setting.key}:{got['n_spills']}")
    return Gate(
        VALIDITY, "V9 compiled shared memory",
        "Triton allocated the shared memory the residency model assumed",
        f"every setting within {SMEM_PROBE_TOLERANCE:.0%}, and zero spills",
        worst <= SMEM_PROBE_TOLERANCE and not spills,
        f"worst disagreement {worst:.1%} over {len(probe)} settings; spills: "
        + (", ".join(spills) if spills else "none"),
        "every rung of the residency ladder, hence P1 and P3. The model is "
        "num_stages (BM BK + BK BN) b; if Triton allocates something else the "
        "resident-block counts are wrong and the report still plots")


def measure(args, cfg, plan: Plan, csv_path: Path, cache_root: Path,
            done, samples: list[Sample], probe: KernelProbe
            ) -> tuple[dict[str, int], dict[str, int]]:
    """The metered part. Appends every timing as it lands, so aborting keeps it."""
    import torch

    from moe.baselines._framework_config import vllm_call_kwargs
    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    override_config, where = SWEEP.find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    print(f"override hook: {where}.override_config")
    compiles: dict[str, int] = {s.key: 0 for s in plan.settings}
    executed: dict[str, int] = {s.key: 0 for s in plan.settings}
    seen: set[Path] = set()
    SWEEP.count_new(cache_root, seen)
    rng = random.Random(args.seed)
    batches = token_plan(cfg, plan)

    for rep in range(1, args.reps + 1):
        for tokens, wanted in batches:
            todo = [(st, bm, n) for st in plan.settings for bm, n in wanted
                    if (st.key, bm, n, rep) not in done]
            if not todo:
                continue
            spec = BenchSpec(cfg, num_tokens=tokens, dtype=args.dtype,
                             routing=RoutingSpec("uniform", 0.0), seed=args.seed)
            x, weights = make_inputs(spec, device="cuda")
            ids = SWEEP.balanced_ids(cfg, tokens, "cuda")
            w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                           device="cuda")
            kw = vllm_call_kwargs(spec)
            kw["activation"] = MoEActivation(kw["activation"])

            def call(_f=fused_experts, _x=x, _wt=weights, _w=w, _i=ids, _k=kw):
                return _f(hidden_states=_x, w1=_wt.w1, w2=_wt.w2,
                          topk_weights=_w, topk_ids=_i, **_k)

            rng.shuffle(todo)
            for st, bm, n in todo:
                rows = n * bm
                executed[st.key] += 1
                conf = dict(st.pinned(plan.block_n, plan.block_k),
                            BLOCK_SIZE_M=bm)
                # Per (setting, block size) cache directory, set BEFORE the
                # first compile of that pair. A warm cache dumping nothing has
                # already cost this project a PTX dump once.
                #
                # ATTRIBUTION DOES NOT DEPEND ON THIS REDIRECT TAKING EFFECT.
                # Some Triton versions snapshot the variable at import, so the
                # per-setting directory is a nicety; `count_new` runs over the
                # whole cache root immediately after THIS cell's first call, so
                # whatever compiled between the two counts belongs to this
                # setting wherever the files landed. V7 stays sound either way.
                arm = cache_root / f"{st.key}-bm{bm}"
                arm.mkdir(parents=True, exist_ok=True)
                os.environ["TRITON_CACHE_DIR"] = str(arm)
                try:
                    with override_config(conf):
                        call()
                        torch.cuda.synchronize()
                        compiles[st.key] += SWEEP.count_new(cache_root, seen)
                        probe.record(st.key)
                        ms0, _, _ = SWEEP.time_call(call, 1, 3)
                        iters = SWEEP.scaled_iters(ms0, args.iters,
                                                   args.cell_budget_ms)
                        ms, mn, sd = SWEEP.time_call(call, args.warmup, iters)
                    sample = Sample(st.key, st.num_stages, st.num_warps,
                                    st.group_m, bm, n, rows, tokens, rep, ms,
                                    mn, sd, iters)
                except Exception as exc:                # noqa: BLE001
                    sample = Sample(st.key, st.num_stages, st.num_warps,
                                    st.group_m, bm, n, rows, tokens, rep, 0.0,
                                    0.0, 0.0, 0, "failed",
                                    f"{type(exc).__name__}: {exc}")
                    print(f"  {st.key} BM={bm} n={n} rep={rep} FAILED "
                          f"{sample.detail}")
                samples.append(sample)
                append_sample(csv_path, sample)
                print(f"  rep {rep} T={tokens:6d} {st.key:12s} BM={bm:3d} "
                      f"n={n:2d}  {sample.ms_p50:9.4f} ms "
                      f"({sample.iters} iters)")
            del x, weights, ids, w, kw
            torch.cuda.empty_cache()
    return compiles, executed


# --------------------------------------------------------------------------
# Analysis: one fit per setting, three contrasts, one verdict.
# --------------------------------------------------------------------------

@dataclass
class SettingResult:
    """One setting's ladder, its fit, and everything a gate reads off it."""

    setting: Setting
    residency: Residency
    footprint_bytes: float
    predicted_alpha: float
    points: list[tuple[int, float]]
    reference_points: list[tuple[int, float]]
    memory_points: int
    alpha: float | None
    alpha_corrected: float | None
    alpha_sigma: float | None
    per_rep_alpha: list[float]
    spread: float | None
    mean_rel_err: float
    reference_note: str
    reference_block_m: int | None
    reference_refused: bool
    basis: str

    @property
    def usable(self) -> bool:
        """May this setting's alpha enter a verdict."""
        return (self.alpha_corrected is not None
                and self.memory_points >= MIN_MEMORY_TREADS
                and not self.reference_refused)


def _cells_for(cfg, points, block_m: int, block_n: int):
    return [SWEEP.make_cell(cfg, n * block_m, block_m, ms, sm_count=1,
                            block_n=block_n)
            for n, ms in points if ms > 0]


def _alpha_of(fit, cfg, bandwidth_gbps: float) -> tuple[float | None,
                                                        float | None]:
    """`(alpha, alpha with activation traffic subtracted)` from one fit.

    The correction is `block_m_crossing_sweep`'s own, called rather than
    reproduced. It is a common constant across this grid -- BM, BN and K are
    pinned -- so it cannot move a contrast; it is applied anyway so the numbers
    printed here are on the same scale as every other alpha the study quotes.
    """
    if fit.alpha is None or not fit.load_ms or fit.slope_memory is None:
        return None, None
    corr = ((fit.slope_memory
             - SWEEP.activation_slope_ms(cfg, fit.block_m, bandwidth_gbps))
            / fit.load_ms)
    return fit.alpha, corr


def _per_rep_alphas(samples, key: str, ref, cfg, block_n: int,
                    bandwidth_gbps: float) -> list[float]:
    """One alpha per repeat, against the SETTING'S POOLED reference.

    The spread of these is the only honest uncertainty this design has on
    alpha, and it is what the occupancy gate's noise floor is measured in. The
    reference is pooled rather than refitted per repeat on purpose: refitting it
    would put the reference's own noise into every repeat's alpha and inflate
    the floor, which would make a real swing unprovable rather than making a
    fake one provable.
    """
    reps = sorted({s.rep for s in samples if s.setting == key})
    out = []
    for rep in reps:
        pts = sorted((s.tiles, s.ms_p50) for s in samples
                     if s.setting == key and s.block_m == SUBJECT_BLOCK_M
                     and s.rep == rep and s.status == "ok" and s.ms_p50 > 0)
        if len(pts) < MIN_MEMORY_TREADS:
            continue
        fit = SWEEP.fit_ladder(pts, SUBJECT_BLOCK_M, ref)
        _, corr = _alpha_of(fit, cfg, bandwidth_gbps)
        if corr is not None and fit.memory_points >= MIN_MEMORY_TREADS:
            out.append(corr)
    return out


def analyse_settings(samples, cfg, plan: Plan, reg: Registered, b: int, *,
                     ridge: float, bandwidth_gbps: float) -> list[SettingResult]:
    """Fit every setting, each against ITS OWN measured compute reference."""
    out: list[SettingResult] = []
    for st in plan.settings:
        sub, _, spread = collapse(samples, st.key, SUBJECT_BLOCK_M)
        ref_pts, _, _ = collapse(samples, st.key, REFERENCE_BLOCK_M)
        cells = (_cells_for(cfg, sub, SUBJECT_BLOCK_M, plan.block_n)
                 + _cells_for(cfg, ref_pts, REFERENCE_BLOCK_M, plan.block_n))
        pinned = st.pinned(plan.block_n, plan.block_k)
        ref = SWEEP.compute_reference(
            cells, (SUBJECT_BLOCK_M, REFERENCE_BLOCK_M), cfg=cfg, ridge=ridge,
            bandwidth_gbps=bandwidth_gbps, b=b, pinned=pinned,
            capability=reg.limits.capability)
        fit = SWEEP.fit_ladder(sub, SUBJECT_BLOCK_M, ref)
        alpha, corrected = _alpha_of(fit, cfg, bandwidth_gbps)
        per_rep = _per_rep_alphas(samples, st.key, ref, cfg, plan.block_n,
                                  bandwidth_gbps)
        sigma = statistics.pstdev(per_rep) if len(per_rep) > 1 else None
        out.append(SettingResult(
            setting=st, residency=reg.residency_by_setting[st.key],
            footprint_bytes=reg.footprint_by_setting[st.key],
            predicted_alpha=reg.concurrency_alpha[st.key],
            points=sub, reference_points=ref_pts,
            memory_points=fit.memory_points, alpha=alpha,
            alpha_corrected=corrected, alpha_sigma=sigma, per_rep_alpha=per_rep,
            spread=spread, mean_rel_err=fit.mean_rel_err,
            reference_note=ref.note, reference_block_m=ref.block_m,
            reference_refused=ref.refused, basis=fit.basis))
    return out


@dataclass(frozen=True)
class Contrast:
    """One arm's answer: the two ends, the effect, and whether it is readable."""

    name: str
    lo_label: str
    hi_label: str
    lo_alpha: float | None
    hi_alpha: float | None
    sigma: float | None
    levels: int

    @property
    def swing(self) -> float | None:
        if self.lo_alpha is None or self.hi_alpha is None:
            return None
        return self.hi_alpha - self.lo_alpha

    @property
    def ratio(self) -> float | None:
        if self.lo_alpha is None or self.hi_alpha is None or not self.lo_alpha:
            return None
        return self.hi_alpha / self.lo_alpha


def occupancy_contrast(results: list[SettingResult]) -> Contrast:
    """alpha at the most-resident setting against alpha at the least.

    Averaged within a residency LEVEL rather than taken from one setting,
    because two `num_stages` can compute to the same resident-block count -- on
    an A100 four and five stages both give two -- and treating them as two rungs
    would put a pure num_stages effect on an axis that did not move.
    """
    usable = [r for r in results
              if ARM_OCCUPANCY in r.setting.arms and r.usable]
    by: dict[int, list[SettingResult]] = {}
    for r in usable:
        by.setdefault(r.residency.resident_blocks, []).append(r)
    if len(by) < 2:
        return Contrast("occupancy", "", "", None, None, None, len(by))
    lo, hi = min(by), max(by)
    lo_a = statistics.fmean([r.alpha_corrected for r in by[lo]])
    hi_a = statistics.fmean([r.alpha_corrected for r in by[hi]])
    sig = [r.alpha_sigma for r in usable if r.alpha_sigma is not None]
    return Contrast("occupancy", f"{lo} blocks/SM", f"{hi} blocks/SM", lo_a,
                    hi_a, statistics.fmean(sig) if sig else None, len(by))


def swizzle_contrast(results: list[SettingResult]) -> Contrast:
    """alpha at the widest swizzle against alpha at GROUP_SIZE_M=1."""
    usable = [r for r in results
              if ARM_SWIZZLE in r.setting.arms and r.usable]
    by = {r.setting.group_m: r for r in usable}
    if len(by) < 2:
        return Contrast("swizzle", "", "", None, None, None, len(by))
    lo, hi = min(by), max(by)
    sig = [r.alpha_sigma for r in usable if r.alpha_sigma is not None]
    return Contrast("swizzle", f"G={lo}", f"G={hi}", by[lo].alpha_corrected,
                    by[hi].alpha_corrected,
                    statistics.fmean(sig) if sig else None, len(by))


def warp_contrast(results: list[SettingResult]) -> Contrast:
    """The largest alpha gap between two num_warps at ONE resident-block count.

    Matched on residency and on num_stages, so the only thing that differs is
    the warp count -- and therefore the resident warps, not the resident blocks.
    """
    usable = [r for r in results if ARM_WARPS in r.setting.arms and r.usable]
    by: dict[tuple[int, int], dict[int, float]] = {}
    for r in usable:
        key = (r.setting.num_stages, r.residency.resident_blocks)
        by.setdefault(key, {})[r.setting.num_warps] = r.alpha_corrected
    pairs = [(k, v) for k, v in by.items() if len(v) >= 2]
    if not pairs:
        return Contrast("warp-control", "", "", None, None, None, 0)
    key, v = max(pairs, key=lambda kv: max(kv[1].values()) - min(kv[1].values()))
    lo_w, hi_w = min(v), max(v)
    sig = [r.alpha_sigma for r in usable if r.alpha_sigma is not None]
    return Contrast("warp-control", f"s{key[0]} w{lo_w}", f"s{key[0]} w{hi_w}",
                    v[lo_w], v[hi_w], statistics.fmean(sig) if sig else None,
                    len(pairs))


def monotone_in_residency(results: list[SettingResult]) -> tuple[int, int]:
    """`(rising steps, total steps)` across the residency ladder.

    The concurrency model does not merely predict a swing, it predicts a
    DIRECTION at every step: more resident blocks, more concurrent footprint,
    less survives to the re-read, higher alpha. A swing that arrives with the
    steps in the wrong order is not that mechanism, and reporting only the end
    points would hide it.
    """
    usable = [r for r in results
              if ARM_OCCUPANCY in r.setting.arms and r.usable]
    by: dict[int, list[float]] = {}
    for r in usable:
        by.setdefault(r.residency.resident_blocks, []).append(r.alpha_corrected)
    levels = sorted(by)
    if len(levels) < 2:
        return 0, 0
    means = [statistics.fmean(by[k]) for k in levels]
    rising = sum(1 for a, c in zip(means, means[1:], strict=False) if c > a)
    return rising, len(means) - 1


# --------------------------------------------------------------------------
# The gates. VALIDITY first: a claim gate read under a failed validity gate is
# a number with no referent.
# --------------------------------------------------------------------------

def gate_provenance(limits: CardLimits, l2_source: str, measured: bool) -> Gate:
    hypothesis = ("HYPOTHESIS" in limits.source or "HYPOTHESIS" in l2_source)
    return Gate(
        VALIDITY, "V1 card provenance",
        "the residency ladder and the L2 come from the ATTACHED card",
        "no HYPOTHESIS value reaches a measured run",
        (not hypothesis) if measured else None,
        f"limits: {limits.source}; L2: {l2_source}",
        "every rung of the residency ladder and the whole of P1's predicted "
        "swing: a per-SM shared memory or an L2 borrowed from another part "
        "puts the x axis and the threshold on a machine that was not measured",
        ["A hardcoded band from another machine has reached seven published "
         "reports in this repo once already."])


def gate_residency_swept(reg: Registered, results: list[SettingResult]) -> Gate:
    levels = sorted({r.residency.resident_blocks for r in results
                     if ARM_OCCUPANCY in r.setting.arms})
    span = (max(levels) / min(levels)) if levels and min(levels) else 0.0
    ok = len(levels) >= MIN_RESIDENCY_LEVELS and span >= MIN_RESIDENCY_SPAN
    return Gate(
        VALIDITY, "V2 residency swept",
        "the occupancy arm actually moved resident blocks per SM",
        f">= {MIN_RESIDENCY_LEVELS} distinct levels spanning >= "
        f"{MIN_RESIDENCY_SPAN:.1f}x",
        ok, f"levels {levels}, span {span:.2f}x",
        "P1 entirely, and P3 with it: a grid that computes one residency at "
        "every setting has swept nothing, and its flat alpha would read as "
        "evidence for program order when it is evidence of nothing",
        ["per-CTA shared memory across the arm: "
         + ", ".join(f"{r.setting.key} {r.residency.smem_per_block // 1024} KiB"
                     for r in sorted(results,
                                     key=lambda r: r.setting.num_stages)
                     if ARM_OCCUPANCY in r.setting.arms),
         f"the L2 they are compared against is {reg.l2_bytes / 1e6:.0f} MB "
         f"({reg.l2_source})"])


def gate_geometry_fixed(samples, plan: Plan) -> Gate:
    seen = sorted({s.block_m for s in samples})
    ok = set(seen) <= {SUBJECT_BLOCK_M, REFERENCE_BLOCK_M} and bool(seen)
    return Gate(
        VALIDITY, "V3 geometry pinned",
        "BLOCK_M, BLOCK_SIZE_N and BLOCK_SIZE_K are the same at every setting",
        f"block sizes measured are exactly "
        f"{{{SUBJECT_BLOCK_M}, {REFERENCE_BLOCK_M}}} at "
        f"BLOCK_SIZE_N={plan.block_n}, BLOCK_SIZE_K={plan.block_k}",
        ok if seen else None, f"block sizes in the CSV: {seen}",
        "P5, hence every contrast: alpha_fitted carries alpha_a (BM/BN) and "
        "BM/K, and those cancel out of a difference only while BM, BN and K "
        "are identical on both sides of it")


def gate_references(results: list[SettingResult]) -> Gate:
    bad = [r.setting.key for r in results if r.reference_refused]
    none = [r.setting.key for r in results if r.reference_block_m is None
            and not r.reference_refused]
    ok = not bad and not none and bool(results)
    return Gate(
        VALIDITY, "V4 compute reference",
        f"every setting qualified its own BLOCK_M={REFERENCE_BLOCK_M} ladder "
        "as a compute branch",
        "no refusals and no missing references",
        ok if results else None,
        (f"refused: {bad or 'none'}; absent: {none or 'none'}"),
        "the membership decision at every setting whose reference is missing "
        "or refused, hence that setting's alpha and every contrast it enters. "
        "A reference that is proportional but at the wrong level classifies "
        "every tread against a branch 40x too steep and prints blanks")


def gate_identifiable(results: list[SettingResult]) -> Gate:
    usable = [r for r in results if r.usable]
    return Gate(
        VALIDITY, "V5 alpha identifiable",
        "enough settings measured alpha over enough memory-bound treads",
        f">= {MIN_SETTINGS_FOR_VERDICT} settings with >= {MIN_MEMORY_TREADS} "
        "memory-bound treads and a fitted alpha",
        len(usable) >= MIN_SETTINGS_FOR_VERDICT,
        f"{len(usable)} of {len(results)} settings usable: "
        + ", ".join(f"{r.setting.key}:{r.memory_points}" for r in results),
        "every claim gate: a contrast between two settings needs both of them")


def gate_replication(results: list[SettingResult]) -> Gate:
    spreads = [r.spread for r in results if r.spread is not None]
    worst = max(spreads) if spreads else None
    return Gate(
        VALIDITY, "V6 replication",
        "a setting's treads reproduce between repeats",
        f"worst per-setting median relative spread <= "
        f"{MAX_REPLICATE_SPREAD:.1%}",
        None if worst is None else worst <= MAX_REPLICATE_SPREAD,
        "no setting had two repeats to compare" if worst is None
        else f"worst {worst:.2%} over {len(spreads)} settings",
        "every claim gate's noise floor, which is measured from these repeats "
        "and not assumed")


def gate_override(compiles: dict[str, int], executed: dict[str, int]) -> Gate:
    silent = sorted(k for k, v in compiles.items() if v < 1)
    ran = sum(executed.values())
    ok = not silent and ran > 0
    return Gate(
        VALIDITY, "V7 override applied",
        "every setting recompiled, so num_stages, num_warps and GROUP_SIZE_M "
        "reached the kernel",
        ">= 1 fresh Triton artefact per setting and > 0 timings executed",
        ok if compiles else None,
        f"{ran} timings; artefacts " + ", ".join(
            f"{k}:{v}" for k, v in sorted(compiles.items())),
        "everything. All three knobs are compile-time constants of the Triton "
        "kernel, so a setting that compiled nothing measured the previous "
        "setting again and its 'effect' is zero by construction")


def gate_non_vacuity(results: list[SettingResult],
                     contrasts: list[Contrast]) -> Gate:
    treads = sum(len(r.points) for r in results)
    reps = sum(len(r.per_rep_alpha) for r in results)
    formed = [c.name for c in contrasts if c.swing is not None]
    ok = treads > 0 and reps > 0 and len(formed) >= 2
    return Gate(
        VALIDITY, "V8 non-vacuity",
        "this report examined real treads and formed real contrasts",
        "> 0 treads, > 0 per-repeat fits, and >= 2 of the 3 contrasts formed",
        ok, f"{treads} treads, {reps} per-repeat alphas, contrasts formed: "
            f"{formed or 'none'}",
        "the entire report. A check that examined nothing also reports zero "
        "failures, which is the shape this study has already published once")


def _floor(contrast: Contrast, registered_threshold: float) -> float:
    """The occupancy threshold actually applied: registered, floored at noise.

    Registered before the run as half the model's predicted swing; raised at
    scoring time to `OCCUPANCY_SIGMA` times the measured per-repeat spread of
    alpha, because a swing inside the noise is not a swing whatever a model
    predicted. Raising a threshold with measured noise is not moving the
    goalposts: it can only make the claim harder.
    """
    noise = OCCUPANCY_SIGMA * contrast.sigma if contrast.sigma else 0.0
    return max(registered_threshold, noise)


def gate_occupancy(contrast: Contrast, reg: Registered,
                   rising: tuple[int, int]) -> Gate:
    threshold = _floor(contrast, reg.occupancy_threshold)
    swing = contrast.swing
    lines = [
        f"model predicted {reg.occupancy_swing:+.3f} across this ladder; "
        f"registered gate was half of it, {reg.occupancy_threshold:.3f}",
        f"noise floor {OCCUPANCY_SIGMA:.0f} x "
        + (f"{contrast.sigma:.4f}" if contrast.sigma is not None else "unknown")
        + f" -> threshold applied {threshold:.3f}",
        f"direction: {rising[0]} of {rising[1]} steps rise with residency",
    ]
    if swing is None:
        return Gate(CLAIM, "P1 occupancy", "alpha rises with resident blocks",
                    f"swing >= {threshold:.3f} and every step rising", None,
                    "the occupancy contrast could not be formed: fewer than "
                    "two usable residency levels", "", lines)
    moved = swing >= threshold
    return Gate(
        CLAIM, "P1 occupancy", "alpha rises with resident blocks per SM",
        f"swing >= {threshold:.3f} in alpha between {contrast.lo_label} and "
        f"{contrast.hi_label}, and every step rising",
        moved and rising[0] == rising[1],
        f"alpha {contrast.lo_alpha:.3f} -> {contrast.hi_alpha:.3f}, "
        f"swing {swing:+.3f}", "", lines)


def swizzle_collapse(results: list[SettingResult]) -> tuple[list[int], bool]:
    """`(groups whose alpha stopped being identifiable, was G=1 identifiable)`.

    The signature a reuse-distance world would leave in THIS design. It is not
    a fallback for a broken run: a reference that was refused is a different
    state, it is caught by V4, and settings in it are excluded here.
    """
    arm = [r for r in results if ARM_SWIZZLE in r.setting.arms
           and not r.reference_refused]
    base = next((r for r in arm if r.setting.group_m == BASE_GROUP), None)
    gone = sorted(r.setting.group_m for r in arm
                  if not r.usable and r.setting.group_m != BASE_GROUP)
    return gone, bool(base and base.usable)


def gate_order(contrast: Contrast, reg: Registered,
               collapse: tuple[list[int], bool]) -> Gate:
    """P2, with the two routes the design actually offers.

    ROUTE ONE is the ratio: `alpha(G_hi) / alpha(1)` at or under twice the
    capped reuse-distance prediction.

    ROUTE TWO is the collapse. Below `order_collapse_alpha` this block size is
    not memory bound at all and the fit returns nothing, so a world where the
    swizzle really does divide alpha by the group width cannot show up as a
    small ratio -- it shows up as wide-G settings losing their memory branch
    while G=1 keeps its. Scoring only route one would call that world a FAIL,
    which is the wrong answer for the right-looking reason.
    """
    ratio = contrast.ratio
    gone, base_ok = collapse
    lines = [
        f"reuse distance predicts {reg.order_ratio:.3f} with the "
        f"tiles-per-expert cap ({reg.order_cap:.1f} tiles, a median over the "
        "ladder and not a threshold)",
        f"uncapped it predicts {1.0 / reg.group_hi:.4f}, which is the 40x gap "
        "this experiment starts from",
        f"route two: this design reads alpha only above "
        f"{reg.window[0]:.3f} and outside [{reg.window[1]:.3f}, "
        f"{reg.window[2]:.3f}], so a collapse at wide G is the same claim",
    ]
    if gone and base_ok:
        return Gate(
            CLAIM, "P2 program order", "alpha falls as 1/GROUP_SIZE_M",
            f"ratio <= {reg.order_gate:.3f}, OR identifiability collapses at "
            f"wide G while G={BASE_GROUP} still fits",
            True,
            f"route two: G={gone} lost the memory branch while G={BASE_GROUP} "
            f"kept it, which is what an alpha outside "
            f"[{reg.window[0]:.3f}, {reg.window[1]:.3f}] looks like here",
            "", lines)
    if ratio is None:
        return Gate(CLAIM, "P2 program order",
                    "alpha falls as 1/GROUP_SIZE_M",
                    f"alpha({contrast.hi_label}) / alpha({contrast.lo_label}) "
                    f"<= {reg.order_gate:.3f}", None,
                    "the swizzle contrast could not be formed: fewer than two "
                    "usable GROUP_SIZE_M settings, and not by collapse "
                    f"(G={BASE_GROUP} usable: {base_ok})", "", lines)
    return Gate(
        CLAIM, "P2 program order", "alpha falls as 1/GROUP_SIZE_M",
        f"alpha({contrast.hi_label}) / alpha({contrast.lo_label}) <= "
        f"{reg.order_gate:.3f}, which is {ORDER_RATIO_TOLERANCE:.0f}x the "
        "prediction",
        ratio <= reg.order_gate,
        f"alpha {contrast.lo_alpha:.3f} -> {contrast.hi_alpha:.3f}, "
        f"ratio {ratio:.3f}", "", lines)


def gate_warp_control(warp: Contrast, occ: Contrast, threshold: float) -> Gate:
    """P3, and it is UNKNOWN rather than PASS when P1 did not move.

    "the warp effect is smaller than the occupancy effect" is a comparison of
    two noise measurements when the occupancy effect is itself inside the noise,
    and a comparison of two noise measurements passes about half the time. A
    gate that can pass on a null run is not a gate.
    """
    if warp.swing is None:
        return Gate(CLAIM, "P3 warp control",
                    "alpha tracks resident BLOCKS, not resident WARPS",
                    "the warp contrast is smaller than the occupancy contrast",
                    None, "no matched num_warps pair was usable", "",
                    ["num_warps changes warps per SM at fixed blocks per SM "
                     "only while the shared-memory limit binds; V2's table "
                     "says whether it did."])
    if occ.swing is None:
        return Gate(CLAIM, "P3 warp control",
                    "alpha tracks resident BLOCKS, not resident WARPS",
                    "the warp contrast is smaller than the occupancy contrast",
                    None, f"warp swing {warp.swing:+.3f} but no occupancy "
                          "contrast to compare it against")
    if abs(occ.swing) < threshold:
        return Gate(
            CLAIM, "P3 warp control",
            "alpha tracks resident BLOCKS, not resident WARPS",
            "|warp swing| < |occupancy swing|, and only where the occupancy "
            f"swing cleared its own threshold of {threshold:.3f}",
            None,
            f"occupancy swing {occ.swing:+.3f} did not clear {threshold:.3f}, "
            f"so the warp swing {warp.swing:+.3f} has nothing to be smaller "
            "than: two noise measurements compared would pass half the time",
            "", ["This gate is a CONTROL on P1 and is only readable when P1 "
                 "moved."])
    return Gate(
        CLAIM, "P3 warp control",
        "alpha tracks resident BLOCKS, not resident WARPS",
        "|warp swing| < |occupancy swing|",
        abs(warp.swing) < abs(occ.swing),
        f"warp {warp.lo_label} -> {warp.hi_label}: {warp.swing:+.3f}; "
        f"occupancy: {occ.swing:+.3f}",
        "", ["A warp effect at or above the occupancy effect says the "
             "residency reading is confounded by latency hiding or register "
             "pressure and P1 cannot be read as a footprint effect."])


def verdict_of(occupancy: Gate, order: Gate) -> str:
    """Which model the data picked, INCLUDING the null and the unreadable case."""
    if occupancy.passed is None and order.passed is None:
        return VERDICT_UNREADABLE
    if occupancy.passed and order.passed:
        return VERDICT_BOTH
    if occupancy.passed:
        return VERDICT_CONCURRENCY
    if order.passed:
        return VERDICT_ORDER
    return VERDICT_NEITHER


VERDICT_NOTE = {
    VERDICT_CONCURRENCY:
        "alpha is set by how many blocks are resident, not by the order they "
        "run in. Reuse-distance analysis -- the standard predictor, and what "
        "TileSight uses -- assumes sequential execution and does not transfer "
        "to a machine whose concurrent working set overflows L2 before program "
        "order gets a vote. That is a correction to a published method.",
    VERDICT_ORDER:
        "alpha is set by program order after all, and the 40x gap between "
        "1/GROUP_SIZE_M and the measured re-read fraction is something else -- "
        "most likely the expert boundary, which caps reuse at tiles per expert "
        "and is already in the capped form of the prediction.",
    VERDICT_BOTH:
        "both knobs move alpha. The two mechanisms are not exclusive and the "
        "effect sizes above are the finding; neither model may be reported as "
        "THE explanation.",
    VERDICT_NEITHER:
        "a null, and it is a result. Neither concurrency nor program order "
        "moves the re-read fraction by more than this design can resolve, "
        "which retires both models at once and points at DRAM scheduling, the "
        "replacement policy or sector granularity instead. The design's "
        "resolution is V6's spread and P1's applied threshold; quote both "
        "whenever this verdict is quoted.",
    VERDICT_UNREADABLE:
        "neither contrast could be formed, so no model was tested. Read the "
        "VALIDITY gates: this is a broken run, not a null.",
}


# --------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------

def analyse(samples, cfg, plan: Plan, reg: Registered, b: int, *, ridge: float,
            bandwidth_gbps: float, compiles: dict[str, int],
            executed: dict[str, int], l2_source: str, measured: bool,
            probe: dict[str, dict] | None = None, probe_note: str = ""
            ) -> tuple[list[str], list[Gate], dict]:
    """Every fit, every contrast, every gate, and the verdict."""
    results = analyse_settings(samples, cfg, plan, reg, b, ridge=ridge,
                               bandwidth_gbps=bandwidth_gbps)
    occ = occupancy_contrast(results)
    swz = swizzle_contrast(results)
    wrp = warp_contrast(results)
    rising = monotone_in_residency(results)

    lines = ["", "## What each setting measured", "",
             f"{'setting':12s} {'arms':24s} {'blk/SM':>7s} {'MiB flight':>11s} "
             f"{'pred':>6s} {'treads':>7s} {'mem':>4s} {'alpha':>7s} "
             f"{'corr':>7s} {'sigma':>7s} {'spread':>7s}  reference"]
    for r in sorted(results, key=lambda r: (r.residency.resident_blocks,
                                            r.setting.key)):
        lines.append(
            f"{r.setting.key:12s} {_arms(r.setting):24.24s} "
            f"{r.residency.resident_blocks:7d} "
            f"{r.footprint_bytes / 2**20:11.0f} "
            f"{r.predicted_alpha:6.3f} {len(r.points):7d} {r.memory_points:4d} "
            + (f"{r.alpha:7.3f}" if r.alpha is not None else "    n/a")
            + (f"{r.alpha_corrected:7.3f}" if r.alpha_corrected is not None
               else "    n/a")
            + (f"{r.alpha_sigma:7.4f}" if r.alpha_sigma is not None
               else "    n/a")
            + (f"{r.spread:7.2%}" if r.spread is not None else "    n/a")
            + "  " + (f"BM={r.reference_block_m}"
                      if r.reference_block_m else "REFUSED"))
    for r in results:
        if not r.usable:
            lines.append(f"  {r.setting.key} unusable: {r.basis}"
                         + (f" | {r.reference_note}" if r.reference_refused
                            else ""))

    lines += ["", "## The three contrasts", ""]
    for c in (occ, swz, wrp):
        if c.swing is None:
            lines.append(f"  {c.name:14s} NOT FORMED ({c.levels} usable level"
                         f"{'' if c.levels == 1 else 's'})")
        else:
            lines.append(
                f"  {c.name:14s} {c.lo_label:14s} {c.lo_alpha:.3f}  ->  "
                f"{c.hi_label:14s} {c.hi_alpha:.3f}   swing {c.swing:+.3f}   "
                f"ratio {c.ratio:.3f}")

    gates = [
        gate_provenance(reg.limits, l2_source, measured),
        gate_residency_swept(reg, results),
        gate_geometry_fixed(samples, plan),
        gate_references(results),
        gate_identifiable(results),
        gate_replication(results),
        gate_override(compiles, executed),
        gate_non_vacuity(results, [occ, swz, wrp]),
        gate_compiled_smem(probe or {}, probe_note, results),
    ]
    g_occ = gate_occupancy(occ, reg, rising)
    g_ord = gate_order(swz, reg, swizzle_collapse(results))
    gates += [g_occ, g_ord,
              gate_warp_control(wrp, occ, _floor(occ, reg.occupancy_threshold))]

    verdict = verdict_of(g_occ, g_ord)
    # FAIL and UNKNOWN are not the same standing and must not print the same
    # sentence. A FAILED validity gate says the verdict is wrong; an UNKNOWN
    # one says a specific thing about it was not checked, and V9's own
    # `invalidates` is written to say which.
    blocking = [g.name for g in gates
                if g.kind == VALIDITY and g.passed is False]
    unchecked = [g.name for g in gates
                 if g.kind == VALIDITY and g.passed is None]
    lines += ["", "## Verdict", "", f"  {verdict}", ""]
    lines += ["  " + line for line in _wrap(VERDICT_NOTE[verdict], 74)]
    if blocking:
        lines += ["", "  READ THIS FIRST: the VALIDITY gates "
                       f"{', '.join(blocking)} FAILED, so the verdict above is "
                       "not entitled to be quoted. Each names what it "
                       "invalidates."]
    if unchecked:
        lines += ["", f"  NOT CHECKED: {', '.join(unchecked)} returned "
                      "UNKNOWN, which is weaker than a PASS and is not a FAIL. "
                      "Each says what a reader may still quote."]

    payload = {
        "verdict": verdict,
        "settings": [
            {"key": r.setting.key, "arms": list(r.setting.arms),
             "num_stages": r.setting.num_stages,
             "num_warps": r.setting.num_warps, "group_m": r.setting.group_m,
             "resident_blocks": r.residency.resident_blocks,
             "residency_binding": r.residency.binding,
             "smem_per_block": r.residency.smem_per_block,
             "footprint_bytes": r.footprint_bytes,
             "predicted_alpha": r.predicted_alpha,
             "alpha": r.alpha, "alpha_corrected": r.alpha_corrected,
             "alpha_sigma": r.alpha_sigma, "per_rep_alpha": r.per_rep_alpha,
             "memory_points": r.memory_points, "treads": len(r.points),
             "spread": r.spread, "mean_rel_err": r.mean_rel_err,
             "reference_block_m": r.reference_block_m,
             "reference_refused": r.reference_refused,
             "reference_note": r.reference_note, "basis": r.basis,
             "points": r.points, "usable": r.usable}
            for r in results],
        "contrasts": {c.name: {"lo": c.lo_label, "hi": c.hi_label,
                               "lo_alpha": c.lo_alpha, "hi_alpha": c.hi_alpha,
                               "swing": c.swing, "ratio": c.ratio,
                               "sigma": c.sigma, "levels": c.levels}
                      for c in (occ, swz, wrp)},
        "rising_steps": rising,
        "compiled_smem": probe or {},
        "compiled_smem_note": probe_note,
        "registered": {"occupancy_swing": reg.occupancy_swing,
                       "occupancy_threshold": reg.occupancy_threshold,
                       "order_ratio": reg.order_ratio,
                       "order_gate": reg.order_gate,
                       "order_cap": reg.order_cap,
                       "l2_bytes": reg.l2_bytes, "l2_source": reg.l2_source,
                       "limits": asdict(reg.limits)},
        "gates": [g.as_dict() for g in gates],
    }
    return lines, gates, payload


def _wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    line = ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


# --------------------------------------------------------------------------
# The published corpus: what is already on disk about both models.
# --------------------------------------------------------------------------

#: PART SPECIFICATIONS, used ONLY by `--audit` and labelled wherever they print.
#: These are not measurements of the pods that produced the published rows;
#: they are the published SM count, L2 capacity and compute capability of the
#: part. A measured run never touches this table -- it reads the attached
#: device -- and the audit says so in its own heading.
CARD_SPECS: dict[str, tuple[tuple[int, int], int, int]] = {
    "a100": ((8, 0), 108, 40 * 1000 * 1000),
    "h200": ((9, 0), 132, 50 * 1000 * 1000),
}


@dataclass
class CorpusRow:
    """One published BLOCK_M=64 alpha, with the knobs that produced it."""

    session: str
    card: str
    model: str
    group_m: int
    block_n: int
    num_warps: int
    num_stages: int
    alpha: float
    memory_points: int
    path: str


def load_corpus(published: Path, dtype: str = "bf16"
                ) -> tuple[list[CorpusRow], list[str]]:
    """Every published BLOCK_M=64 alpha, and a NAMED reason for each skip.

    Skips are returned rather than swallowed: a corpus audit that quietly drops
    what it cannot parse reports a cleaner picture than the data supports.
    """
    rows: list[CorpusRow] = []
    skipped: list[str] = []
    for path in sorted(published.glob("*/*.report.json")):
        session = path.parent.name
        card = next((c for c in CARD_SPECS if c in session.lower()), None)
        if card is None:
            skipped.append(f"{session}/{path.name}: no known card in the "
                           "session name")
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            skipped.append(f"{session}/{path.name}: unreadable ({exc})")
            continue
        if payload.get("dtype") != dtype:
            skipped.append(f"{session}/{path.name}: dtype "
                           f"{payload.get('dtype')!r}, not {dtype!r}")
            continue
        fixed = payload.get("fixed") or {}
        rung = (payload.get("ladder") or {}).get(str(SUBJECT_BLOCK_M)) or {}
        alpha = rung.get("alpha_corrected") or rung.get("alpha")
        if not alpha:
            skipped.append(f"{session}/{path.name}: no identifiable "
                           f"BLOCK_M={SUBJECT_BLOCK_M} alpha")
            continue
        missing = [k for k in ("GROUP_SIZE_M", "BLOCK_SIZE_N", "num_warps",
                               "num_stages") if k not in fixed]
        if missing:
            skipped.append(f"{session}/{path.name}: report records no "
                           f"{', '.join(missing)}, so the knobs behind this "
                           "alpha are unknown and it cannot enter a contrast")
            continue
        rows.append(CorpusRow(
            session=session, card=card, model=str(payload.get("model") or "?"),
            group_m=int(fixed["GROUP_SIZE_M"]),
            block_n=int(fixed["BLOCK_SIZE_N"]),
            num_warps=int(fixed["num_warps"]),
            num_stages=int(fixed["num_stages"]), alpha=float(alpha),
            memory_points=int(rung.get("memory_points") or 0), path=str(path)))
    return rows, skipped


def corpus_group_contrasts(rows: list[CorpusRow]) -> list[tuple[str, int, int,
                                                                float, float]]:
    """`(label, G_lo, G_hi, alpha_lo, alpha_hi)` within one session and config.

    Matched on session, card, model, BLOCK_SIZE_N, num_warps and num_stages, so
    the only thing that differs across the pair is the swizzle. Cross-session
    pairs are NOT formed here: the two H200 sessions in this repo differ in
    num_stages and in calibration, and pairing across them would put a session
    effect on the swizzle axis.
    """
    by: dict[tuple, dict[int, float]] = {}
    for r in rows:
        by.setdefault((r.session, r.card, r.model, r.block_n, r.num_warps,
                       r.num_stages), {})[r.group_m] = r.alpha
    out = []
    for key, g in sorted(by.items()):
        if len(g) < 2:
            continue
        lo, hi = min(g), max(g)
        out.append((f"{key[1]} {key[2]} n{key[3]} w{key[4]} s{key[5]}",
                    lo, hi, g[lo], g[hi]))
    return out


def corpus_stage_contrasts(rows: list[CorpusRow]) -> list[tuple[str, int, int,
                                                                float, float]]:
    """`num_stages` pairs at matched card, model, G, BLOCK_N and num_warps.

    UNCONTROLLED, and that is the entire reason the pod run exists. The only
    num_stages contrast the corpus holds is between two SESSIONS run on
    different days against different calibrations, so a difference here carries
    the session as well as the residency. It is reported because it is the best
    prior available and because it is what the pod run has to beat, never as
    evidence.
    """
    by: dict[tuple, dict[int, float]] = {}
    for r in rows:
        by.setdefault((r.card, r.model, r.group_m, r.block_n, r.num_warps),
                      {})[r.num_stages] = r.alpha
    out = []
    for key, s in sorted(by.items()):
        if len(s) < 2:
            continue
        lo, hi = min(s), max(s)
        out.append((f"{key[0]} {key[1]} g{key[2]} n{key[3]} w{key[4]}",
                    lo, hi, s[lo], s[hi]))
    return out


def audit_report(rows: list[CorpusRow], skipped: list[str], block_n: int,
                 block_k: int, b: int, r_max: int
                 ) -> tuple[list[str], list[Gate], dict]:
    """Score both models against what is already on disk, off GPU."""
    lines = ["", "## The published corpus, off GPU", "",
             f"{len(rows)} published BLOCK_M={SUBJECT_BLOCK_M} alphas, "
             f"{len(skipped)} files skipped"]
    for why in skipped:
        lines.append(f"  skipped {why}")

    cap = median_tiles_per_expert(
        [n * SUBJECT_BLOCK_M
         for n in range(1, r_max // SUBJECT_BLOCK_M + 1)], SUBJECT_BLOCK_M)
    groups = corpus_group_contrasts(rows)
    lines += ["", "### GROUP_SIZE_M contrasts, within one session and config",
              "", f"{'pair':40s} {'G':>9s} {'alpha':>15s} {'ratio':>7s} "
                  f"{'predicted':>10s}"]
    ratios = []
    for label, lo, hi, a_lo, a_hi in groups:
        ratio = a_hi / a_lo if a_lo else math.nan
        ratios.append(ratio)
        lines.append(f"{label:40s} {lo:3d}->{hi:4d} {a_lo:7.3f}->{a_hi:7.3f} "
                     f"{ratio:7.3f} {alpha_order(1.0, hi, cap):10.3f}")

    stages = corpus_stage_contrasts(rows)
    lines += ["", "### num_stages contrasts -- UNCONTROLLED, ACROSS SESSIONS",
              "", f"{'pair':40s} {'stages':>9s} {'alpha':>15s} {'rel':>8s}"]
    stage_rel = []
    for label, lo, hi, a_lo, a_hi in stages:
        rel = a_hi / a_lo - 1.0 if a_lo else math.nan
        stage_rel.append(rel)
        lines.append(f"{label:40s} {lo:3d}->{hi:4d} {a_lo:7.3f}->{a_hi:7.3f} "
                     f"{rel:+8.1%}")

    lines += ["", "The residency these settings compute to, from PUBLISHED PART "
                  "SPECIFICATIONS", "and not from any attached device:", ""]
    seen = sorted({(r.card, r.num_stages, r.num_warps) for r in rows})
    for card, ns, nw in seen:
        capability, sms, l2 = CARD_SPECS[card]
        limits = CardLimits(capability, SMEM_PER_SM_BYTES[capability],
                            MAX_THREADS_PER_SM[capability],
                            MAX_BLOCKS_PER_SM[capability], sms, l2,
                            "published part specification")
        st = Setting(ns, nw, BASE_GROUP, (ARM_OCCUPANCY,))
        res = residency(st.pinned(block_n, block_k), SUBJECT_BLOCK_M, b, limits)
        foot = concurrent_footprint_bytes(
            MODEL_CONFIGS["mixtral-8x7b"], SUBJECT_BLOCK_M, block_n, b,
            res.resident_blocks, sms)
        lines.append(f"  {card:5s} s{ns} w{nw}   {res.line()}   "
                     f"{foot / 2**20:7.0f} MiB in flight   alpha_conc "
                     f"{alpha_concurrency(foot, l2):.3f}")

    predicted_stage: list[str] = []
    for card in sorted({r.card for r in rows}):
        capability, sms, l2 = CARD_SPECS[card]
        limits = CardLimits(capability, SMEM_PER_SM_BYTES[capability],
                            MAX_THREADS_PER_SM[capability],
                            MAX_BLOCKS_PER_SM[capability], sms, l2,
                            "published part specification")
        preds = {}
        for ns in sorted({r.num_stages for r in rows if r.card == card}):
            st = Setting(ns, BASE_WARPS, BASE_GROUP, (ARM_OCCUPANCY,))
            res = residency(st.pinned(block_n, block_k), SUBJECT_BLOCK_M, b,
                            limits)
            preds[ns] = alpha_concurrency(
                concurrent_footprint_bytes(MODEL_CONFIGS["mixtral-8x7b"],
                                           SUBJECT_BLOCK_M, block_n, b,
                                           res.resident_blocks, sms), l2)
        if len(preds) >= 2:
            lo, hi = min(preds), max(preds)
            predicted_stage.append(
                f"  {card:5s} s{lo}->s{hi}: concurrency predicts "
                f"{preds[hi] / preds[lo] - 1.0:+.1%} in alpha")
    if predicted_stage:
        lines += ["", "What the concurrency model would have predicted for "
                      "those num_stages pairs:", *predicted_stage,
                  "  -- against the measured medians above, which carry a "
                  "session difference as well."]

    median_ratio = statistics.median(ratios) if ratios else None
    gate_pred = ORDER_RATIO_TOLERANCE * alpha_order(
        1.0, max((hi for _, _, hi, _, _ in groups), default=1), cap)
    gates = [
        Gate(VALIDITY, "A0 corpus non-vacuity",
             "the audit examined published alphas and formed contrasts",
             f">= 2 GROUP_SIZE_M contrasts and >= 1 BLOCK_M={SUBJECT_BLOCK_M} "
             "alpha per side",
             len(groups) >= 2 and bool(rows),
             f"{len(rows)} alphas, {len(groups)} swizzle contrasts, "
             f"{len(stages)} num_stages contrasts",
             "the two audit gates below, which would otherwise be scoring an "
             "empty set and reporting zero failures"),
        Gate(CLAIM, "A1 reuse distance on the corpus",
             "published alpha falls as 1/GROUP_SIZE_M",
             f"median alpha ratio <= {gate_pred:.3f} "
             f"({ORDER_RATIO_TOLERANCE:.0f}x the capped prediction)",
             None if median_ratio is None else median_ratio <= gate_pred,
             "no contrast formed" if median_ratio is None
             else f"median ratio {median_ratio:.3f} over {len(ratios)} pairs",
             "", ["A FAIL here is what motivates the pod run: the standard "
                  "predictor misses by roughly the factor this experiment is "
                  "about."]),
        Gate(CLAIM, "A2 num_stages on the corpus",
             "the corpus can already answer the occupancy question",
             "IT CANNOT, and this gate is UNKNOWN by construction: every "
             "num_stages pair on disk spans two sessions",
             None,
             "no contrast formed" if not stage_rel
             else f"median {statistics.median(stage_rel):+.1%} over "
                  f"{len(stage_rel)} cross-session pairs",
             "", ["Reported as the prior the pod run has to beat, never as "
                  "evidence: the pair differs in session, calibration and day "
                  "as well as in num_stages."]),
    ]
    payload = {"rows": [asdict(r) for r in rows], "skipped": skipped,
               "group_contrasts": groups, "stage_contrasts": stages,
               "median_group_ratio": median_ratio, "order_gate": gate_pred}
    return lines, gates, payload


# --------------------------------------------------------------------------
# Self test: plant three worlds, check the verdict tells them apart.
# --------------------------------------------------------------------------

def planted_samples(cfg, plan: Plan, alpha_of, *, ridge: float,
                    bandwidth_gbps: float, b: int, noise: float = 0.0,
                    seed: int = 0) -> list[Sample]:
    """Timings generated FROM the model, so the analysis has a known answer.

    `alpha_of(setting)` is the world: a function of residency alone, of
    GROUP_SIZE_M alone, or of nothing. Everything downstream -- the reference,
    the fit, the contrasts, the gates, the verdict -- is the real code path.
    """
    rng = random.Random(seed)
    out: list[Sample] = []
    for st in plan.settings:
        alpha = alpha_of(st)
        for rep in range(1, plan.reps + 1):
            for bm, rows in ((SUBJECT_BLOCK_M, plan.subject_rows),
                             (REFERENCE_BLOCK_M, plan.reference_rows)):
                for r in rows:
                    ms = SWEEP.model_ms(cfg, r, bm, alpha=alpha, ridge=ridge,
                                        bandwidth_gbps=bandwidth_gbps, b=b,
                                        overhead_ms=0.05)
                    if noise:
                        ms *= math.exp(rng.gauss(0.0, noise))
                    out.append(Sample(
                        st.key, st.num_stages, st.num_warps, st.group_m, bm,
                        r // bm, r, SWEEP.tokens_for_rows(cfg, r), rep, ms, ms,
                        ms * noise, 0))
    return out


def self_test(cfg, plan: Plan, reg: Registered, b: int, *, ridge: float,
              bandwidth_gbps: float, noise: float, seed: int
              ) -> tuple[list[str], list[Gate]]:
    """Three planted worlds and the verdicts they MUST produce.

    The point is not that the code runs. It is that the verdict
    DISCRIMINATES: a pipeline that answers CONCURRENCY in a world built from
    GROUP_SIZE_M alone cannot settle anything, and neither can one that answers
    NEITHER in both.
    """
    base = 0.95
    worlds = [
        ("concurrency: alpha = 1 - L2/footprint(residency)",
         lambda st: reg.concurrency_alpha[st.key], VERDICT_CONCURRENCY),
        ("program order: alpha = alpha(1) / min(G, tiles per expert)",
         lambda st: alpha_order(base, st.group_m, reg.order_cap),
         VERDICT_ORDER),
        ("null: alpha is the same everywhere",
         lambda _st: base, VERDICT_NEITHER),
    ]
    lines = ["", "## Self test: planted worlds, real gates, real verdict", "",
             f"{'world':52s} {'occupancy':>12s} {'swizzle':>12s}  verdict"]
    gates: list[Gate] = []
    for name, alpha_of, expected in worlds:
        samples = planted_samples(cfg, plan, alpha_of, ridge=ridge,
                                  bandwidth_gbps=bandwidth_gbps, b=b,
                                  noise=noise, seed=seed)
        compiles = {s.key: 1 for s in plan.settings}
        executed = {s.key: plan.reps for s in plan.settings}
        _, world_gates, payload = analyse(
            samples, cfg, plan, reg, b, ridge=ridge,
            bandwidth_gbps=bandwidth_gbps, compiles=compiles,
            executed=executed, l2_source=reg.l2_source, measured=False)
        got = payload["verdict"]
        occ = payload["contrasts"]["occupancy"]["swing"]
        swz = payload["contrasts"]["swizzle"]["ratio"]
        lines.append(
            f"{name:52s} "
            + (f"{occ:+12.3f}" if occ is not None else f"{'collapsed':>12s}")
            + (f"{swz:12.3f}" if swz is not None else f"{'collapsed':>12s}")
            + f"  {got}")
        gates.append(Gate(
            VALIDITY, f"S {expected.split()[0].lower()}",
            f"a world built from {name.split(':')[0]} is called {expected}",
            f"verdict == {expected!r}", got == expected,
            f"verdict {got!r}",
            "the verdict machinery itself: a pipeline that answers the same in "
            "every planted world cannot settle this experiment",
            [f"  {g.name}: {'PASS' if g.passed else 'FAIL' if g.passed is False else 'UNKNOWN'}"
             for g in world_gates if g.kind == CLAIM]))
    return lines, gates


# --------------------------------------------------------------------------
# Where the run lands.
# --------------------------------------------------------------------------

def git_visibility(path: Path) -> str:
    """Say out loud whether git would keep this file.

    `.gitignore` excludes `results/*` and re-includes only
    `results/published/`, so a run that writes anywhere else under the repo
    produces files `git add -A` silently drops. Checked with `git check-ignore`
    rather than by re-implementing the pattern rules, because the pattern rules
    are what got it wrong.
    """
    try:
        proc = subprocess.run(["git", "check-ignore", "-q", str(path)],
                              cwd=ROOT, capture_output=True, timeout=15,
                              check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git check-ignore could not run ({exc}); path unverified"
    if proc.returncode == 0:
        return ("IGNORED by git. Nothing written here enters the repo. Publish "
                "with scripts/publish_results.sh, or point --out at "
                "results/published/<date>-<gpu>-occupancy-vs-swizzle")
    if proc.returncode == 1:
        return "git will keep this path"
    return (f"git check-ignore exited {proc.returncode}; path unverified "
            f"({proc.stderr.decode(errors='replace').strip()})")


def detect_card_slug() -> str | None:
    """Slug for the ATTACHED device, or None. Resolved before the run id."""
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
    except Exception:                                   # noqa: BLE001
        return None
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or None


def default_run_id(args, card: str) -> str:
    """Derived from EVERY swept parameter AND the card.

    THE FAILURE THIS PREVENTS HAS HAPPENED TWICE IN THIS REPO. A run id that
    omitted GROUP_SIZE_M had a G=16 run resume into the G=1 directory, find
    every cell present, skip all of them and print G=1's timings under a G=16
    heading. A run id that omitted the CARD had an A100 session resume into an
    H200 directory on the shared `/workspace` volume and report the H200's
    timings against the A100's ridge.
    """
    key = json.dumps({
        "card": card, "model": args.model, "dtype": args.dtype,
        "r_max": args.r_max, "block_n": args.block_n, "block_k": args.block_k,
        "stages": sorted(args.stages), "control_warps": sorted(args.control_warps),
        "control_stages": sorted(args.control_stages),
        "groups": sorted(args.groups), "reps": args.reps, "iters": args.iters,
        "warmup": args.warmup, "budget": args.cell_budget_ms, "seed": args.seed,
        "subject": SUBJECT_BLOCK_M, "reference": REFERENCE_BLOCK_M,
        # --capability PRUNES the grid: it sets smem_per_sm, which sets how many
        # blocks fit, which is the residency ladder. Asserting 8.0 on an sm_90
        # box drops the s=5 rung, and `stages` above is the REQUEST, not the
        # survivors -- so two runs differing only here derived one id and the
        # 8-setting run would resume into the 9-setting directory.
        # --sm-count and --l2-bytes are deliberately NOT here: they enter only
        # the analysis, and a knob that re-analyses the same timings must not
        # fork the directory, or a re-report becomes an empty resume.
        "capability": getattr(args, "capability", None)},
        sort_keys=True)
    stages = "_".join(str(s) for s in sorted(args.stages))
    warps = "_".join(str(w) for w in sorted({BASE_WARPS, *args.control_warps}))
    groups = "_".join(str(g) for g in sorted(args.groups))
    return (f"{card}-{args.model}-{args.dtype}-r{args.r_max}"
            f"-n{args.block_n}-k{args.block_k}-s{stages}-w{warps}-g{groups}"
            f"-x{args.reps}-{hashlib.sha1(key.encode()).hexdigest()[:6]}")


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in text.split(",") if v.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mixtral-8x7b",
                    choices=sorted(MODEL_CONFIGS),
                    help="mixtral by default: E/k=4 makes the whole "
                         "rows-per-expert ladder reachable at four times the "
                         "token count, and its per-CTA stream is the one the "
                         "concurrency argument is stated in")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"),
                    help="bf16 or fp16. Not fp8: every alpha in this study is "
                         "a bf16 statement and the fp8 call path needs a quant "
                         "config this script does not build")
    ap.add_argument("--r-max", type=int, default=1024,
                    help="deepest rows per expert. Sets both ladders: "
                         f"{SUBJECT_BLOCK_M} treads for the subject and "
                         f"{REFERENCE_BLOCK_M} for the reference, and the "
                         "reference needs at least three")
    ap.add_argument("--block-n", type=int, default=SWEEP.FIXED["BLOCK_SIZE_N"])
    ap.add_argument("--block-k", type=int, default=SWEEP.FIXED["BLOCK_SIZE_K"])
    ap.add_argument("--stages", type=_ints, default=DEFAULT_STAGES,
                    help="the occupancy ladder, as num_stages")
    ap.add_argument("--control-warps", type=_ints, default=DEFAULT_CONTROL_WARPS,
                    help="num_warps for the control arm, beside the base "
                         f"{BASE_WARPS}. Multiples of 4 only: Triton's "
                         "warpgroup predicate wants num_warps %% 4 == 0 at "
                         "BLOCK_M %% 64 == 0")
    ap.add_argument("--control-stages", type=_ints, default=CONTROL_WARP_STAGES,
                    help="which num_stages the warp control is run at")
    ap.add_argument("--groups", type=_ints, default=DEFAULT_GROUPS,
                    help="the swizzle arm, as GROUP_SIZE_M")
    ap.add_argument("--reps", type=int, default=5,
                    help="passes over the whole grid. Settings are shuffled "
                         "inside each token count on the same tensors, so "
                         "every contrast is paired and the spread of these is "
                         "the noise floor the claim gates are scored against")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cell-budget-ms", type=float, default=400.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.9,
                    help="COSTING ONLY: the alpha used to predict how many "
                         "milliseconds a cell takes, so --dry-run can price "
                         "the run. It never reaches a gate")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="Op/B. Omitted, a measured run reads the attached "
                         "device's own calibration and REFUSES when there is "
                         "none")
    ap.add_argument("--bandwidth-gbps", type=float, default=0.0)
    ap.add_argument("--sm-count", type=int, default=0)
    ap.add_argument("--capability", default="",
                    help="MAJOR.MINOR, e.g. 9.0. Read off the device when a "
                         "device is attached")
    ap.add_argument("--l2-bytes", type=int, default=0,
                    help="L2 capacity. Read off the device when a device is "
                         "attached; NEVER defaulted on a measured run")
    ap.add_argument("--card", default="",
                    help="card slug for the run id. May name a card that is "
                         "ABSENT, so a laptop can print the pod's real path; "
                         "it may never contradict one that is present")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--published", type=Path,
                    default=ROOT / "results" / "published")
    ap.add_argument("--noise", type=float, default=0.005,
                    help="lognormal timing noise for --self-test only")
    ap.add_argument("--run", action="store_true",
                    help="actually measure. Without it this prints the plan "
                         "and the registered predictions and stops, which is "
                         "what a metered pod session should read first")
    ap.add_argument("--dry-run", action="store_true",
                    help="the plan and the predictions, assuming an H200 where "
                         "no device is attached and saying so")
    ap.add_argument("--audit", action="store_true",
                    help="score both models against results/published, no GPU")
    ap.add_argument("--self-test", action="store_true",
                    help="three planted worlds; checks the verdict discriminates")
    ap.add_argument("--replay", type=Path, default=None,
                    help="re-report a finished run from its directory, no GPU")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="exit 1 when any gate is not PASS")
    ap.add_argument("--require-git-visible", action="store_true",
                    help="refuse to run when the output directory is gitignored")
    return ap


def resolve_device(args, *, synthetic: bool) -> tuple[CardLimits, str, str]:
    """The attached card's occupancy limits, or a labelled hypothesis, or refuse.

    Three states and they are not interchangeable:

      1. Values on the command line, which makes them the operator's assertion
         and puts them in this run's own argv.
      2. THE ATTACHED DEVICE, read from `torch.cuda.get_device_properties`.
         This is the default for a measured run.
      3. For `--dry-run` and `--self-test` ONLY, where nothing is measured and
         so nothing can be mislabelled, an H200 as a stated HYPOTHESIS.

    Anything else REFUSES. Residency is the swept axis and the L2 is the
    denominator of the registered prediction; a measured run that borrowed
    either from another part would sweep an axis that does not exist on the
    card it ran on.
    """
    capability = None
    if args.capability:
        major, _, minor = args.capability.partition(".")
        capability = (int(major), int(minor or 0))
    sm_count, l2 = args.sm_count, args.l2_bytes
    source = "given on the command line"
    l2_source = "given on the command line"
    device = ""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            device = torch.cuda.get_device_name(0)
            capability = capability or (props.major, props.minor)
            sm_count = sm_count or props.multi_processor_count
            probed = getattr(props, "L2_cache_size", 0)
            if not l2 and probed:
                l2 = int(probed)
                l2_source = f"read off {device} (L2_cache_size)"
            source = f"read off {device}"
    except Exception:                                   # noqa: BLE001
        pass
    if synthetic:
        capability = capability or HYPOTHESIS_CAPABILITY
        sm_count = sm_count or HYPOTHESIS_SM_COUNT
        if not l2:
            l2, l2_source = HYPOTHESIS_L2_BYTES, HYPOTHESIS_L2_SOURCE
        if not device:
            source = (f"HYPOTHESIS: sm_{capability[0]}{capability[1]}, "
                      f"{sm_count} SMs, which belong to no attached device")
    return card_limits(capability, sm_count, l2, source), l2_source, device


def resolve_roofline(args, *, synthetic: bool) -> tuple[float, float, str]:
    """`(ridge, bandwidth GB/s, source)` from THIS card's own calibration.

    The ridge and the bandwidth qualify the compute reference, which decides
    membership at every setting, so they are a MEASUREMENT decision and not a
    costing. A measured run with no calibration for its own device REFUSES:
    seven published A100 reports in this repo were scored against an H200 ridge
    because a missing calibration was allowed to fall back.
    """
    if args.ridge and args.bandwidth_gbps:
        return args.ridge, args.bandwidth_gbps, "given on the command line"
    try:
        from moe.bench.roofline import load_measured
        hw = load_measured()
    except Exception as exc:                            # noqa: BLE001
        hw, why = None, str(exc)
    else:
        why = "no calibration file for the attached device"
    if hw is not None:
        return (args.ridge or hw.ridge_point(args.dtype),
                args.bandwidth_gbps or hw.bandwidth_bytes_s / 1e9,
                f"measured on this device: {hw.name}")
    if synthetic:
        return (args.ridge or 162.8, args.bandwidth_gbps or 4374.5,
                "HYPOTHESIS: the 2026-09-01 H200 calibration, which belongs to "
                "no attached device")
    raise SystemExit(
        f"REFUSED: {why}. The ridge and the bandwidth qualify the compute "
        "reference at every setting, so they decide which treads are memory "
        "bound and therefore every alpha in this report.\n"
        "    Run:  python scripts/calibrate_hardware.py\n"
        "    or state the assertion yourself:  --ridge <Op/B> "
        "--bandwidth-gbps <GB/s>\n"
        "    off GPU, --dry-run and --self-test may assume the H200 and say so.")


def _main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    header = ("experiment  occupancy_vs_swizzle: is alpha set by CONCURRENCY "
              "or by PROGRAM ORDER?")
    if args.dry_run and args.run:
        print(f"{header}\n\nREFUSED: --dry-run and --run contradict each "
              "other. --dry-run prints the plan against an assumed H200 and "
              "measures nothing; --run measures. Nothing measured.")
        return 2
    synthetic = args.dry_run or not args.run

    if args.audit:
        rows, skipped = load_corpus(args.published, args.dtype)
        lines, gates, payload = audit_report(rows, skipped, args.block_n,
                                             args.block_k, b, args.r_max)
        print("\n".join([header] + lines + ["", "## Gates", ""]
                        + render_gates(gates)))
        return 1 if (args.fail_on_gate
                     and any(g.passed is not True for g in gates)) else 0

    try:
        limits, l2_source, device = resolve_device(args, synthetic=synthetic)
    except CardUnavailable as exc:
        # A SENTENCE, not a traceback. This is the ordinary path for `--run` on
        # a laptop and for a pod whose device this file's tables do not know,
        # and both readers need the next command rather than a stack.
        print(f"{header}\n\nREFUSED: {exc}\n"
              "    Nothing was measured. Off GPU, --audit, --self-test and "
              "--dry-run all still work.")
        return 2
    ridge, bandwidth, roof_source = resolve_roofline(args, synthetic=synthetic)
    plan = build_plan(args, cfg, b, limits, alpha=args.alpha, ridge=ridge,
                      bandwidth_gbps=bandwidth)
    reg = register(cfg, plan.settings, limits, block_n=args.block_n,
                   block_k=args.block_k, b=b, subject_rows=plan.subject_rows,
                   ridge=ridge, l2_source=l2_source)

    detected = detect_card_slug()
    card = args.card or detected or NO_CARD_SLUG
    if args.card and detected and args.card != detected:
        print(f"REFUSED: --card {args.card!r} but the attached device is "
              f"{detected!r}. --card may name a card that is ABSENT, so a "
              "laptop can print the pod's real path; it may never contradict "
              "one that is present. Nothing measured.")
        return 2
    run_id = args.run_id or default_run_id(args, card)
    out_dir = (args.replay or (args.out or SWEEP.results_root())
               / "occupancy_vs_swizzle" / run_id)
    csv_path = out_dir / "cells.csv"
    card_path = out_dir / "CARD"
    inputs_path = out_dir / "inputs.json"
    cache_root = out_dir / "triton-cache"

    lines = [header, "", predictions_text(reg, plan.settings), "",
             "## The plan", ""] + plan.lines(cfg) + [
        f"roofline     ridge {ridge:.3f} Op/B, bandwidth {bandwidth:.1f} GB/s "
        f"({roof_source})",
        f"card         {card}" + ("" if detected else
                                  f"  (NO DEVICE ATTACHED: the id above is the "
                                  f"{NO_CARD_SLUG!r} one and is not what a pod "
                                  "derives; pass --card <slug> for that)"),
        f"WRITES TO    {out_dir}",
        f"             {git_visibility(out_dir)}",
        "             cells.csv (one row per timing, flushed), CARD, "
        "inputs.json, report.txt, report.json, triton-cache/"]

    gates: list[Gate] = []
    payload: dict = {"predictions": predictions_text(reg, plan.settings),
                     "plan": {"model": args.model, "dtype": args.dtype,
                              "settings": [s.key for s in plan.settings],
                              "subject_rows": plan.subject_rows,
                              "reference_rows": plan.reference_rows,
                              "refused": plan.refused,
                              "cells": plan.cells,
                              "estimated_seconds": plan.estimated_seconds}}

    if args.self_test:
        more, g = self_test(cfg, plan, reg, b, ridge=ridge,
                            bandwidth_gbps=bandwidth, noise=args.noise,
                            seed=args.seed)
        lines += more
        gates += g

    if args.replay is not None:
        if not csv_path.exists():
            print("\n".join(lines))
            print(f"\nREFUSED: {csv_path} does not exist, so there is nothing "
                  "to replay.")
            return 2
        stored = json.loads(inputs_path.read_text()) if inputs_path.exists() \
            else {}
        if not stored:
            print("\n".join(lines))
            print(f"\nREFUSED: {inputs_path} is missing, so the ridge, "
                  "bandwidth, L2 and SM count this run was scored against are "
                  "unknown. Re-scoring it against THIS machine's numbers is "
                  "exactly the hybrid-of-two-machines failure this study has "
                  "already published once.")
            return 2
        ridge, bandwidth = stored["ridge"], stored["bandwidth_gbps"]
        limits = CardLimits(tuple(stored["limits"]["capability"]),
                            stored["limits"]["smem_per_sm"],
                            stored["limits"]["max_threads_per_sm"],
                            stored["limits"]["max_blocks_per_sm"],
                            stored["limits"]["sm_count"],
                            stored["limits"]["l2_bytes"],
                            stored["limits"]["source"] + " (replayed)")
        l2_source = stored["l2_source"] + " (replayed)"
        reg = register(cfg, plan.settings, limits, block_n=args.block_n,
                       block_k=args.block_k, b=b,
                       subject_rows=plan.subject_rows, ridge=ridge,
                       l2_source=l2_source)
        _, samples = read_samples(csv_path)
        counts: dict[str, int] = {}
        for s in samples:
            counts[s.setting] = counts.get(s.setting, 0) + 1
        more, g, pay = analyse(samples, cfg, plan, reg, b, ridge=ridge,
                               bandwidth_gbps=bandwidth,
                               compiles=stored.get("compiles", {}),
                               executed=counts, l2_source=l2_source,
                               measured=True,
                               probe=stored.get("compiled_smem") or {},
                               probe_note=stored.get("compiled_smem_note", ""))
        lines += more
        gates += g
        payload["run"] = pay
        print("\n".join(lines + ["", "## Gates", ""] + render_gates(gates)))
        return 1 if (args.fail_on_gate
                     and any(g.passed is not True for g in gates)) else 0

    if not args.run:
        lines += ["", "## Nothing was measured", "",
                  "  This is the plan and the registered predictions. Add "
                  "--run to measure,",
                  "  --self-test to check the verdict discriminates on planted "
                  "worlds, or",
                  "  --audit to score both models against results/published."]
        print("\n".join(lines + (["", "## Gates", ""] + render_gates(gates))
                        if gates else lines))
        return 1 if (args.fail_on_gate
                     and any(g.passed is not True for g in gates)) else 0

    visibility = git_visibility(out_dir)
    if args.require_git_visible and visibility.startswith("IGNORED"):
        print("\n".join(lines))
        print(f"\nREFUSING: {visibility}")
        return 2

    missing = SWEEP.missing_gpu_stack()
    if missing:
        print("\n".join(lines))
        print(f"\n{missing.split('.')[0]}.\n"
              "Off GPU, this script's whole argument is still available:\n"
              "  --audit      both models scored against results/published\n"
              "  --self-test  three planted worlds, checking the verdict "
              "discriminates\n"
              "  --dry-run    the plan, the residency ladder and the cost")
        return 2

    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)
    print("\n".join(lines))

    # THE RESUME GUARD, belt as well as braces. The card is already in the run
    # id, so another card lands in another directory and cannot normally reach
    # this cells.csv. This catches the ways it could anyway: an explicit --out
    # or --run-id aiming two cards at one place, or a directory copied between
    # pods on the shared volume.
    if csv_path.exists():
        written_by = card_path.read_text().strip() if card_path.exists() else ""
        if written_by != card:
            raise SystemExit(
                f"REFUSED to resume {csv_path}: written by card "
                f"{written_by or '<unrecorded>'!r} and this run is {card!r}. "
                "Resuming would score one card's treads against the other's "
                "residency ladder and the other's L2. Move or delete that "
                "directory deliberately. Nothing measured.")
    card_path.write_text(card + "\n")
    inputs_path.write_text(json.dumps(
        {"ridge": ridge, "bandwidth_gbps": bandwidth,
         "roofline_source": roof_source, "l2_source": l2_source,
         "device": device, "limits": asdict(limits)}, indent=2))

    done, samples = read_samples(csv_path)
    started = time.time()
    probe = KernelProbe()
    compiles, executed = measure(args, cfg, plan, csv_path, cache_root, done,
                                 samples, probe)
    print(f"\nmeasured in {time.time() - started:.0f} s")

    more, g, pay = analyse(samples, cfg, plan, reg, b, ridge=ridge,
                           bandwidth_gbps=bandwidth, compiles=compiles,
                           executed=executed, l2_source=l2_source,
                           measured=True, probe=probe.by_setting,
                           probe_note=probe.note)
    gates += g
    payload["run"] = pay
    payload["gpu"] = torch.cuda.get_device_name(0)
    inputs_path.write_text(json.dumps(
        {"ridge": ridge, "bandwidth_gbps": bandwidth,
         "roofline_source": roof_source, "l2_source": l2_source,
         "device": device, "limits": asdict(limits),
         "compiles": compiles, "compiled_smem": probe.by_setting,
         "compiled_smem_note": probe.note}, indent=2))
    tail = more + ["", "## Gates", ""] + render_gates(gates)
    print("\n".join(tail))
    (out_dir / "report.txt").write_text("\n".join(lines + tail))
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2,
                                                    default=str))
    for label, path in (("cells", csv_path), ("inputs", inputs_path),
                        ("report", out_dir / "report.txt"),
                        ("json", out_dir / "report.json")):
        print(f"{label:8s} {path}\n         {git_visibility(path)}")
    return 1 if (args.fail_on_gate
                 and any(g.passed is not True for g in gates)) else 0


def main(argv=None) -> int:
    """Convert a string SystemExit into exit code 2, which is what REFUSED means.

    `raise SystemExit("some sentence")` exits ONE. Every refusal in this file
    was written that way, so a run that refused before measuring anything --
    no calibration for the attached device, an import that drifted, a tile that
    cannot run as pinned -- exited with the same code a run that MEASURED and
    then failed a claim gate would have. The session driver could not tell them
    apart, and this script's own contract says 2 means refused and 1 is reserved
    for --fail-on-gate. Found in review on 2026-09-02, live, on this laptop.

    Caught here rather than at twenty raise sites so the contract holds for a
    caller of main() as well as for the CLI, and so a new refusal added later
    cannot reintroduce the bug by forgetting the code.
    """
    try:
        return _main(argv)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            msg = exc.code if exc.code.startswith("REFUSED") else f"REFUSED: {exc.code}"
            print(msg, file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
