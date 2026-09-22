#!/usr/bin/env python3
"""Separate RESIDENCY from PIPELINE DEPTH on the BLOCK_K diagonal.

    python scripts/blockk_diagonal.py --dry-run --capability 9.0   # laptop, free
    python scripts/blockk_diagonal.py --self-test depth            # the scorer, off GPU
    python scripts/blockk_diagonal.py --self-test residency
    python scripts/blockk_diagonal.py --self-test blockk
    python scripts/blockk_diagonal.py --self-test null
    python scripts/blockk_diagonal.py                              # the pod run

THE OPEN ITEM THIS CLOSES. `scripts/occupancy_vs_swizzle.py` gate P6 reads
UNKNOWN, in these words: "no two num_stages settings computed the same
resident-block count, so the pure pipeline-depth effect was not measured and
the confound in P1 is UNREMOVED on this card". That is not a property of the
H200, it is a property of a sweep that moved ONE knob. `num_stages` sets
resident blocks AND the software-pipeline prefetch depth of the K loop
together, so every rung of that ladder moves two mechanisms in lockstep and the
P1 residency null is confounded with depth. Three of the four modelling
attempts of the 2026-09-10 synthesis lean on that null.

THE LEVER, AND WHY IT IS THE ONLY ONE. BLOCK_SIZE_K is 64 in EVERY fit in this
corpus -- `block_m_crossing_sweep.FIXED`, every bn arm, every alias arm, every
published report. Triton multi-buffers the K loop, so one CTA holds
`num_stages` copies of the `BM x BK` A tile and the `BK x BN` B tile:

    smem(stages, BK) = stages x (BM x BK + BK x BN) x bytes
                     = stages x BK x (BM + BN) x bytes

The product `stages x BK` is what the card's per-SM shared memory divides down
to a resident-block count; `stages` ALONE is the prefetch depth. So a grid in
(num_stages, BLOCK_K) moves the two apart, and nothing else in this apparatus
can. At BLOCK_M = BLOCK_N = 64 in bf16 one stage-BK unit is 256 bytes, so

    (num_stages=3, BLOCK_K=64)  and  (num_stages=6, BLOCK_K=32)   48 KiB, depths 3 and 6
    (num_stages=4, BLOCK_K=64)  and  (num_stages=2, BLOCK_K=128)  64 KiB, depths 4 and 2

are two ISO-SHARED-MEMORY pairs whose depths differ by a factor of two, and

    (3, 32)   (3, 64)   (3, 128)      24 / 48 / 96 KiB, depth 3 THROUGHOUT

is a residency ladder at BYTE-IDENTICAL pipeline depth. Those six settings
plus (1,32) and (2,32) are the eight cells this arm times.

THE THIRD MECHANISM, NAMED BEFORE IT CAN CONTAMINATE ANYTHING. Holding smem
fixed forces `stages` and `BK` to move in opposite directions, so inside an
iso-smem pair "depth" and "BLOCK_K" are perfectly anti-collinear and a two-cell
contrast CANNOT tell them apart. The grid is therefore EIGHT cells and not
four, and the reading is a three-parameter fit

    ln w = b0 + bD log2(num_stages) + bK log2(BLOCK_K) + bR log2(resident blocks)

whose design matrix is full rank with 3 degrees of freedom left over. What
makes it full rank is that residency SATURATES: at the low-shared-memory end
the thread-slot and register limits bind instead, so two cells with different
`stages x BK` land on the SAME resident-block count and the collinearity
`log2 res = const - log2 stages - log2 BK` breaks. The plan prints the design
matrix, its rank and the smallest singular value, and REFUSES before any GPU
time if the design as realised on this card is singular.

    bD  the pure pipeline-depth coefficient, at fixed residency and fixed BK
    bK  the pure BLOCK_K coefficient: loop trip count and access granularity at
        fixed residency and fixed depth. PREDICTED ZERO -- a CTA reads
        `BM x K` of A and `K x BN` of B whatever BK chops K into, so under the
        reuse-distance model BLOCK_K is traffic-neutral. C3 is that prediction,
        and it is a prediction the design can refute.
    bR  the pure residency coefficient: the concurrency hypothesis' own axis.

WHY IT IS SCORED ON w AND NOT ON THE EXA RATIO, and this is not a taste.
`alpha = B / (A + B)` divides the ladder's per-tile slope by a LEVEL that is an
extrapolation of the ladder back to n = 0. On the 2026-09-10 H200 session that
denominator produced ten values above 1.0 across four arms and three NEGATIVE
fitted intercepts -- `moe/bench/weights.py`'s own record of that session, and
`tests/test_block_m_crossing_sweep.py`'s R9 note; `docs/STUDY.md` line 234
counts five and five off the committed cells, and the tree has not reconciled
the two -- and `bn_decomposition`'s session-3 EXA fit returned alpha_a = -0.8143
whose entire SIGN sits inside the reference fixed cost's own jackknife error.
The intercept `A` is far noisier than the slope `B` it is added to and goes
negative in two arms, so a residency effect of a few per cent in `B` arrives in
`B/(A+B)` as a sign flip and vanishes. `w` is the same slope `B` divided by a
MEASURED byte count over a rate the caller names:

    w = (ms per extra M-tile) / (ms to stream the routed expert weight set once)

no fitted level and no intercept. THE NOISE FIGURES THIS PARAGRAPH USED TO
CARRY -- "217 times noisier", "0.115% cold-replicate", "0.37% cross-session" --
ARE NOT IN THIS TREE AS LITERALS. They are the session-3 analysis's, quoted in
the brief that commissioned this arm, and no committed file holds any of them.
TWO OF THE THREE ARE NONETHELESS RECOMPUTABLE HERE, and NOISE_FLOOR.json is
the wrong file to have looked in: it records spreads of ALPHA, but the twelve
fresh-cache replicate reports under `results/published/2026-09-10-nvidia_h200-
gaps-session/results/gaps-nvidia_h200/replicate_noise_floor/nvidia_h200-fresh-
n3/` each carry a whole ladder, so per cell `B` is `slope_memory`, `A` is
`B(1-alpha)/alpha` exactly, and `w` is `B` over a load that is constant across
replicates (all twelve report the same calibrated bandwidth), which makes w's
relative spread IDENTICALLY the slope's. Over that arm's eight cells the
intercept's relative spread across three cold replicates runs 27x to 785x the
slope's, median 38x, and the slope's own runs 0.04% to 0.34%, median 0.15%.
THE DIRECTION THIS PARAGRAPH RESTS ON IS THE APPARATUS'S OWN, at every cell.
"217x" is not reproduced by any pooling of those rows -- the closest single
cell, mixtral G=1 BLOCK_M=64, gives 213x -- and three replicates cannot pin a
ratio of two spreads to three figures, so no such number is quoted here. What
stays unrecomputable is "0.37% cross-session": no committed replicate spans
sessions, and the noise-floor file says so of itself. They are stated as the
analysis's and not as this page's, and NO GATE READS ANY OF THEM -- the only spread this file scores
against is `ASSUMED_W_SPREAD`, which names its own source and is replaced by
this run's own measured spread the moment one cell is on disk. `w` is the one
statistic this apparatus is said to measure well, and the effect this arm is
looking for is a few per cent; whether the pod delivers the noise the analysis
claims is a thing this arm's own V4 replicate-spread check MEASURES rather than
assumes. `moe.bench.weights` owns the arithmetic and this file does not
reimplement any of it.

WHAT THE TWO READINGS MEAN, BOTH REGISTERED BEFORE THE RUN.

  IT FOLLOWS DEPTH. `bD` clears the threshold and `bR` does not. The occupancy
  arm's P1 residency null is CONFIRMED -- what P1 saw moving with `num_stages`
  was latency hiding and not a footprint, the concurrency family is closed, and
  the swizzle/reuse-distance reading of alpha stands unopposed by a footprint
  rival. C1 PASSES here; this is the registered prediction.

  IT FOLLOWS RESIDENCY. `bR` clears and `bD` does not. P1's null was an
  ARTEFACT OF THE LOCKSTEP: residency really does move the weight-stream cost,
  it was hidden in P1 because the depth effect ran the other way at the same
  time, and the concurrent-footprint hypothesis returns. C2 prints the size it
  returns at as a fraction of the concurrency model's OWN predicted swing
  across this run's own residency ladder, so "a sixth of what the model
  predicts" is a printed ratio and not an adjective. C1 FAILS here, and that
  FAIL is the finding.

  NEITHER. Both coefficients inside the threshold. C1 is UNKNOWN, not PASS: an
  arm that saw no effect of either cannot separate them, and the exit code says
  so.

THE REGISTER LIMIT IS THE ONE THING THIS DESIGN CANNOT KNOW OFF GPU, and it is
handled as a measurement rather than as an assumption. `occupancy_vs_swizzle`
computes residency from shared memory, thread slots and CTA slots and says in
its own docstring that the per-thread register count "is decided by ptxas and
is not knowable from the pinned constants. It can only LOWER residency, so
every number here is an UPPER BOUND". Here residency IS the axis being
contrasted, so an upper bound is not enough: this arm reads `n_regs` back off
the compiled kernel through that file's own `KernelProbe`, adds the fourth
limit, and V3 scores the ladder AS REALISED. If the probe cannot read it -- and
occupancy's V9 could not, on two separate pods -- V3 is UNKNOWN, the whole page
is INVALID, and nothing here is quotable. That is the honest outcome and it is
cheap: the probe runs inside the metered loop and costs nothing extra.

THE REGISTER LIMIT IS ALSO WHAT MAKES THE DESIGN IDENTIFIABLE, which is worth
saying out loud because it reads as a hazard. Residency is
`min(by_smem, by_threads, by_blocks, by_regs)`. The last three do not depend on
BLOCK_K at all, so every cell whose shared memory is slack piles up on the same
rung -- and two cells at one rung with different depths is exactly the
iso-residency contrast P6 could not form. The hazard is the opposite case: a
register bound so tight that EVERY cell lands on one rung, which leaves no
residency variation at all. V5 scores that directly, from the realised matrix.

EXIT CODES, THE RUN ID AND THE PROVENANCE BLOCK come from
`moe.bench.exit_codes`, `moe.bench.provenance.run_id` and
`moe.bench.provenance.provenance_block`. `--dry-run` prints the plan and then
exits 2 REFUSED: it scores no gate, so it prints no RESULT line, and
`classify_text` over its log raises `NoGatesScored`, which is what a REFUSED
log looks like from there.

THE CLOCK RULE IS THE APPARATUS STANDARD AND IS NOT RE-DECIDED HERE. DRIFT is
the exclusion: the clock moved while the tread was timed, so the median is a
blend of two operating points. BOTH sides of a LEVEL failure are KEPT with the
side recorded, because under the 700 W cap the under-load clock is an OUTCOME
of the tile -- and this arm sweeps eight cells, so it will see several
operating points by construction. TWO functions here read those verdicts and
`tests/test_blockk_diagonal.py` asserts by AST that it is exactly those two:
`clock_excluded`, which is the ONE function that EXCLUDES on them, and
`clock_state`, which only COUNTS. `CLOCK_VERDICT_READERS` is the list the test
compares against, so a third reader appearing anywhere goes red.
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench import weights as WEIGHTS  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402


def _load(name: str, needed: tuple[str, ...]):
    """Load a sibling script BY PATH and name what has drifted.

    `scripts/` is not a package, so a bare import works only when the sibling is
    the entry point. Both modules loaded here are under active edit by other
    workstreams and both are called on the POD path, so the names AND the
    signatures are probed on a laptop with a sentence that says which one moved,
    rather than as an AttributeError thirty seconds into a metered session.
    """
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    missing = [n for n in needed if not hasattr(module, n)]
    if missing:
        raise SystemExit(
            f"scripts/{name}.py no longer exports {', '.join(missing)}. This "
            "script is deliberately scored by that file's arithmetic rather "
            "than a private copy, so the two move together. Re-point the "
            "import; do not fork it.")
    return module


#: The sweep library: the tile resource bill, the row geometry, the ridge and
#: bandwidth resolvers, the override hook and the compile counter.
SWEEP = _load("block_m_crossing_sweep",
              ("FIXED", "SMEM_PER_BLOCK_BYTES", "MAX_REGISTERS_PER_THREAD",
               "MIN_MEMORY_TREADS", "NO_CARD_SLUG", "SYNTHETIC_INSTRUMENT",
               "DEFAULT_SM_COUNT", "tile_resources", "parse_capability",
               "resolve_capability", "tokens_for_rows", "rows_quantum",
               "useful_flops", "results_root", "detect_card_slug",
               "resolve_ridge", "RidgeUnavailable", "resolve_bandwidth",
               "BandwidthUnavailable", "missing_gpu_stack", "find_override",
               "count_new", "balanced_ids", "timing_basis", "observed_iters",
               "reference_clock_mhz"))

#: The arm whose P6 this one answers. Its occupancy arithmetic -- the per-SM
#: resource tables, `CardLimits` and `residency` -- is IMPORTED and not copied,
#: because a residency ladder computed here against a different per-SM shared
#: memory would be a ladder with the wrong rungs and it would still plot. Its
#: `KernelProbe` is imported for the same reason: it already knows where vLLM
#: keeps the kernel and where each Triton version keeps its compiled cache.
OCC = _load("occupancy_vs_swizzle",
            ("SMEM_PER_SM_BYTES", "MAX_THREADS_PER_SM", "MAX_BLOCKS_PER_SM",
             "RESERVED_SMEM_PER_BLOCK", "HYPOTHESIS_L2_BYTES",
             "HYPOTHESIS_L2_SOURCE", "CardLimits", "CardUnavailable",
             "card_limits", "resolve_device", "residency", "KernelProbe",
             "per_cta_stream_bytes", "concurrent_footprint_bytes",
             "alpha_concurrency", "OCCUPANCY_SWING_FRACTION"))


# --------------------------------------------------------------------------
# The numbers this experiment is arguing about, all stated before any code.
# --------------------------------------------------------------------------

#: The tile the ladders are fitted at. Imported from the occupancy arm's own
#: reasoning rather than re-chosen: at 64 the marginal arithmetic intensity of
#: one more M-tile is a factor of two or more below every calibrated ridge in
#: this study, so every tread is memory bound, which is the condition a
#: weight-stream slope means anything under. V4 MEASURES that rather than
#: assuming it, from this run's own w.
SUBJECT_BLOCK_M = 64

#: BLOCK_SIZE_N, GROUP_SIZE_M and num_warps are pinned at `SWEEP.FIXED`'s
#: values, which is the configuration every w in the published corpus was
#: measured at. Two knobs move in this arm and they are the two in its name.
PINNED_BLOCK_N = 64
PINNED_GROUP_M = 1
#: num_warps DEPARTS FROM `SWEEP.FIXED`, WHICH IS 8, AND THE REASON IS ON THE
#: PLAN PAGE. Residency is `min(by_smem, by_threads, by_blocks, by_regs)` and
#: `by_regs = registers per SM / (32 x num_warps x n_regs)`. At eight warps one
#: CTA is 256 threads, so an H200's 65536-register file holds `256 / n_regs`
#: CTAs and the whole residency ladder collapses onto one rung at any n_regs
#: above 64 -- which is where a Triton grouped GEMM with a 64x64 fp32
#: accumulator and a multi-buffered K loop actually lands. `--dry-run` prints
#: the sensitivity table at both counts: at four warps the design stays rank 4
#: up to n_regs 168 and at eight it is singular above 64.
#:
#: NOTHING HERE IS QUOTED AGAINST THE PUBLISHED CORPUS, which is what makes the
#: departure cheap. Every gate on this page is a contrast ACROSS THE CELLS OF
#: THIS RUN at one warp count; no w here is compared with a published w, so the
#: value only has to be the same for all eight cells, and it is. `--num-warps 8`
#: puts the arm back on the study's pinned value for a reader who wants the
#: levels comparable and is willing to risk the census refusing.
PINNED_WARPS = 4

#: The grid, as `num_stages x BLOCK_SIZE_K`. Eight cells, chosen so that
#: (a) the two ISO-SMEM pairs the design is named for are both present,
#: (b) a three-rung residency ladder stands at a fixed pipeline depth of 3,
#: (c) two cells sit where shared memory is slack, which is what breaks the
#:     stages/BK/residency collinearity, and
#: (d) no cell needs more than 96 KiB per CTA, so the grid is feasible on an
#:     A100 (163 KiB per-block ceiling) as well as on an H200 (227 KiB).
DEFAULT_CELLS = "1x32,2x32,3x32,6x32,3x64,4x64,2x128,3x128"

#: The two pairs the module docstring is about, registered by name so the
#: report cannot quietly compare a different pair. Each is `(from, to)` and NOT
#: `(low depth, high depth)`: the second registered pair runs 4 stages -> 2, so
#: a field called `low_key` named the DEEPER cell and `w_low` its w, and both
#: went into `report.json` through `asdict` for any later reader to misread.
#: The contrast arithmetic was right throughout -- `log2(to/from)` carries the
#: sign -- and only the names were backwards. The two members MUST have equal
#: computed shared memory; V2 asserts it and `--dry-run` prints the arithmetic.
ISO_SMEM_PAIRS = (("3x64", "6x32"), ("4x64", "2x128"))

#: The iso-depth residency ladder: three BLOCK_K at one num_stages. The one
#: contrast in this apparatus where residency moves and the software-pipeline
#: depth is byte-identical.
ISO_DEPTH_LADDER = ("3x32", "3x64", "3x128")

#: How many standard deviations of a coefficient's OWN bootstrap spread it has
#: to clear before this arm says the knob under it moved anything. Three, which
#: is `occupancy_vs_swizzle.OCCUPANCY_SIGMA`'s figure and is imported in spirit
#: rather than by name because that file applies it to an alpha swing and this
#: one to a log-slope: the same bar, on a different statistic.
K_SIGMA = 3.0

#: Treads a cell's ladder must contribute before its w may enter the fit. Two
#: points make a line with no residual; `SWEEP.MIN_MEMORY_TREADS` is 3, and
#: this arm asks for more because its whole content is a few per cent.
MIN_FIT_TREADS = 5

#: A cell whose treads move by more than this between repeats is a measurement
#: of the pod and not of the cell. Same figure the occupancy arm uses.
MAX_REPLICATE_SPREAD = 0.02

#: How far BELOW the calibrated ridge the MEASURED marginal arithmetic intensity
#: has to sit before a cell counts as memory bound. The marginal AI is
#: `useful flops per extra M-tile / (w x weight bytes)`, so it is computed from
#: this run's own w and not from the byte model: if a cell were compute bound
#: its slope would be a compute slope and `w` would not be a weight-traffic
#: statistic at all.
AI_BELOW_RIDGE = 0.75

#: Distinct resident-block counts the realised ladder must produce, and the
#: span it must cover, before the residency coefficient may be read. Below
#: these V5 says the design did not form, which is the outcome a hard register
#: bound produces and it must not read as a null.
MIN_RESIDENCY_LEVELS = 2
MIN_RESIDENCY_SPAN = 2.0

#: Register file per SM, in 32-bit registers, by compute capability. NOT the
#: per-BLOCK ceiling, which `bm128_roofline.REGISTERS_PER_BLOCK` already holds
#: and which answers a different question: that one asks whether ONE CTA's
#: accumulator fits, this one asks how MANY CTAs fit. They are the same number
#: on every part in this table and that coincidence is exactly why confusing
#: them is easy and silent.
#:
#: UNKNOWN CAPABILITIES ARE NOT DEFAULTED, for the reason the two sibling
#: tables give: residency is the axis this experiment contrasts, and a guessed
#: register file moves every rung of it.
REGISTERS_PER_SM: dict[tuple[int, int], int] = {
    (7, 0): 65536,      # V100
    (7, 5): 65536,      # T4
    (8, 0): 65536,      # A100
    (8, 6): 65536,      # A10 / A40 / RTX 30
    (8, 9): 65536,      # L4 / L40S / RTX 40
    (9, 0): 65536,      # H100 / H200
    (10, 0): 65536,     # B200
}

#: The per-cell relative spread of `w` this design's power is computed against
#: when no cells are on disk to measure it from. It is the conservative end of
#: the range `moe/bench/weights.py` records from the 2026-09-10 session -- "a
#: per-repeat sd of 0.002 to 0.005" -- over the median of ALL TWENTY-THREE of
#: that session's ladders, 1.168, and that is the number weights.py publishes.
#: IT IS NOT THE SUBJECT-RANGE MEDIAN, which this line used to call it: that
#: module went out of its way to separate the two, 0.683-1.368 being the range
#: over the SIXTEEN ladders at BLOCK_M <= 64, and it publishes no median for
#: that subset. This arm runs at BLOCK_M=64, i.e. inside the subject subset, so
#: the denominator is a SCALE taken from the wider set and not the subset's own.
#: Labelled ASSUMED wherever it is printed. `--plant-noise` overrides it; a
#: resumed or replayed directory measures it instead.
ASSUMED_W_SPREAD = 0.005 / 1.168
ASSUMED_W_SPREAD_SOURCE = (
    "ASSUMED: 0.005/1.168, the widest per-repeat sd of w in moe/bench/"
    "weights.py's record of the 2026-09-10 session over the median of ALL 23 "
    "of that session's ladders (NOT the subject subset, which publishes no "
    "median). No cell of THIS run was on disk to measure it from")

#: What a planted run calls its card, in the run id and nowhere else.
SYNTHETIC_CARD = "synthetic"

VALIDITY, CLAIM = "VALIDITY", "CLAIM"

VERDICT_DEPTH = "DEPTH"
VERDICT_RESIDENCY = "RESIDENCY"
VERDICT_BOTH = "BOTH"
VERDICT_NEITHER = "NEITHER (null)"


class RefusedBeforeMeasuring(SystemExit):
    """A refusal that costs nothing, carrying the code that says so.

    `raise SystemExit("some sentence")` sets `SystemExit.code` to the STRING and
    the interpreter turns that into exit 1, which is CLAIM_FAIL in the table
    this repo reads -- "measured; a pre-registered claim was refuted" -- for a
    run that measured nothing. Seven refusals in a sibling file exited 1 that
    way for a fortnight.
    """

    def __init__(self, message: str) -> None:
        print(f"REFUSED: {message}")
        super().__init__(exit_codes.REFUSED)


# --------------------------------------------------------------------------
# The grid: one cell is one (num_stages, BLOCK_SIZE_K).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One setting of the two swept knobs, and everything knowable off GPU."""

    num_stages: int
    block_k: int

    @property
    def key(self) -> str:
        return f"{self.num_stages}x{self.block_k}"

    def pinned(self, warps: int = PINNED_WARPS) -> dict:
        """The full config dict `override_config` is handed for this cell.

        Built from `SWEEP.FIXED` so that a knob added to the study's pinned set
        arrives here too, with only the two this arm sweeps overridden.

        `warps` is an ARGUMENT and not a constant because it is the one escape
        from the register collapse V5 refuses on: halving the CTA's thread count
        roughly doubles how many CTAs the register file holds. It is NOT a third
        treatment -- it is held at one value for the whole grid, it is in the run
        id, and `--dry-run` prints the register-sensitivity table at whichever
        value was given, so choosing it is a reviewable decision on the plan page
        and not a knob that moves between cells.
        """
        return dict(SWEEP.FIXED,
                    BLOCK_SIZE_M=SUBJECT_BLOCK_M,
                    BLOCK_SIZE_N=PINNED_BLOCK_N,
                    BLOCK_SIZE_K=self.block_k,
                    GROUP_SIZE_M=PINNED_GROUP_M,
                    num_warps=warps,
                    num_stages=self.num_stages)

    def smem_bytes(self, b: int) -> int:
        """`stages x (BM x BK + BK x BN) x bytes`, one CTA's K-loop buffers.

        NOT recomputed here: `SWEEP.tile_resources` is the study's one statement
        of this arithmetic and it is the function whose refusal already stopped
        a BN=256 arm. This property exists so the pairing check and the plan
        read the same number the refusal does.
        """
        return SWEEP.tile_resources(self.pinned(), SUBJECT_BLOCK_M, b,
                                    None).smem_bytes


def parse_cells(text: str) -> tuple[Cell, ...]:
    """`"3x64,6x32"` -> the two cells, refusing anything that is not a setting."""
    out = []
    for token in (t.strip() for t in text.split(",") if t.strip()):
        stages, _, block_k = token.partition("x")
        try:
            cell = Cell(int(stages), int(block_k))
        except ValueError:
            raise RefusedBeforeMeasuring(
                f"--cells entry {token!r} is not STAGESxBLOCK_K") from None
        if cell.num_stages < 1 or cell.block_k < 16:
            raise RefusedBeforeMeasuring(
                f"--cells entry {token!r}: num_stages must be at least 1 and "
                "BLOCK_SIZE_K at least 16, which is the smallest K a Triton "
                "`tl.dot` accepts")
        if cell.block_k & (cell.block_k - 1):
            raise RefusedBeforeMeasuring(
                f"--cells entry {token!r}: BLOCK_SIZE_K {cell.block_k} is not a "
                "power of two, and vLLM's kernel indexes the K loop assuming "
                "one")
        if cell in out:
            raise RefusedBeforeMeasuring(
                f"--cells names {token!r} twice; two rows for one setting would "
                "be counted as two cells by every gate below")
        out.append(cell)
    if not out:
        raise RefusedBeforeMeasuring("--cells is empty: there is no grid")
    return tuple(out)


def even_k_refusal(cfg, cell: Cell) -> str:
    """Empty when both GEMMs' K extents are exact multiples of this BLOCK_K.

    The fused layer is TWO grouped GEMMs with different K: the up GEMM walks
    `hidden_size` and the down GEMM walks `intermediate_size`. vLLM's kernel
    carries an `EVEN_K` constexpr and takes a MASKED load path when K is not a
    multiple of BLOCK_SIZE_K, which is a different kernel -- different
    instruction mix, different predication, a different time. A cell that
    compiles the masked path is not comparable with one that does not, and the
    difference would land in `bK`, the coefficient this arm registers a
    prediction of zero for. So it is refused at plan time and the arithmetic is
    printed, rather than measured and discounted afterwards.
    """
    bad = [(name, k) for name, k in (("hidden_size", cfg.hidden_size),
                                     ("intermediate_size", cfg.intermediate_size))
           if k % cell.block_k]
    if not bad:
        return ""
    return "; ".join(
        f"{name} {k} is not a multiple of BLOCK_SIZE_K {cell.block_k} "
        f"({k} = {k // cell.block_k} x {cell.block_k} + {k % cell.block_k}), so "
        "this cell takes the kernel's masked EVEN_K=False path"
        for name, k in bad)


def feasibility(cfg, cells, b: int, capability, warps: int) -> dict[str, str]:
    """Every cell's refusal, or the empty string, from the pinned constants.

    Two sources, both pure arithmetic and both computable on a laptop: the
    study's own one-CTA resource bill (shared memory against the card's
    per-block ceiling, fp32 accumulator against the 255-register hardware
    maximum) and the K-divisibility check above.
    """
    out = {}
    for cell in cells:
        why = []
        res = SWEEP.tile_resources(cell.pinned(warps), SUBJECT_BLOCK_M, b,
                                   capability)
        if res.refusal:
            why.append(res.refusal)
        even = even_k_refusal(cfg, cell)
        if even:
            why.append(even)
        out[cell.key] = "; ".join(why)
    return out


def ladder_rows(cfg, treads: int) -> list[int]:
    """Exactly-full tile stacks only: `r = n x BLOCK_M`, zero padding.

    REFUSES when the model's routing cannot form `n BM` rows as an integer
    token count rather than nudging, because a nudged row is a partly-filled
    tile and a slope fitted over padding is a slope of the padding.
    `rows_quantum` is 1 for mixtral and qwen2 and 3 for deepseek-v2-lite, which
    is the model that has already killed two unattended runs.
    """
    q = SWEEP.rows_quantum(cfg)
    out = []
    for n in range(1, treads + 1):
        r = n * SUBJECT_BLOCK_M
        if r % q:
            raise RefusedBeforeMeasuring(
                f"{cfg.name}: E={cfg.num_experts} at top-k {cfg.top_k} needs "
                f"rows per expert to be a multiple of {q}, and {r} (tread {n} "
                f"at BLOCK_M={SUBJECT_BLOCK_M}) is not. Choose another --model "
                "or another --treads.")
        out.append(r)
    return out


# --------------------------------------------------------------------------
# Residency, including the one limit the occupancy arm declares it cannot
# compute.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CellResidency:
    """Resident thread blocks per SM for one cell, and which limit bound.

    `by_smem`, `by_threads` and `by_blocks` come from
    `occupancy_vs_swizzle.residency` and are not recomputed here. `by_regs` is
    the fourth limit, which that file states it does not carry:

        "THE REGISTER LIMIT IS NOT MODELLED and its absence is a stated bound,
        not an oversight: the per-thread register count is decided by ptxas and
        is not knowable from the pinned constants. It can only LOWER residency,
        so every number here is an UPPER BOUND."

    Here residency is the contrast, so the bound is not enough and `n_regs` is
    READ BACK off the compiled kernel. `registers` is None before the kernel
    has compiled -- on a laptop, in `--dry-run`, and on a pod where the probe
    could not reach the Triton cache -- and then `bound` is True and every
    number in the row is an upper bound that V3 refuses to score.

    REGISTER ALLOCATION GRANULARITY IS NOT MODELLED EITHER, and it is the same
    direction: ptxas rounds the per-thread count up to a granule and rounds the
    per-warp total up again, so the true `by_regs` is at most this one. An
    upper bound on an upper bound is still an upper bound.
    """

    resident_blocks: int
    by_smem: int
    by_threads: int
    by_blocks: int
    by_regs: int | None
    smem_per_block: int
    registers: int | None
    spills: int | None
    binding: str

    @property
    def bound(self) -> bool:
        return self.by_regs is None

    def line(self) -> str:
        regs = ("regs ? (UPPER BOUND: no n_regs read)" if self.by_regs is None
                else f"regs {self.by_regs} (at {self.registers}/thread"
                     + (f", {self.spills} spills" if self.spills else "")
                     + ")")
        return (f"{self.resident_blocks} blocks/SM (smem {self.by_smem}, "
                f"threads {self.by_threads}, cta {self.by_blocks}, {regs}; "
                f"{self.binding} binds) at {self.smem_per_block / 1024:.0f} "
                "KiB/CTA")


def registers_per_sm(capability) -> int | None:
    """The card's register file, or None. NEVER a default: see the table."""
    return REGISTERS_PER_SM.get(tuple(capability)) if capability else None


def cell_residency(cell: Cell, b: int, limits, *, warps: int = PINNED_WARPS,
                   registers: int | None = None,
                   spills: int | None = None) -> CellResidency:
    """Resident blocks per SM for one cell, with the register limit folded in."""
    base = OCC.residency(cell.pinned(warps), SUBJECT_BLOCK_M, b, limits)
    regs_file = registers_per_sm(limits.capability)
    by_regs = None
    if registers and regs_file:
        per_cta = 32 * warps * int(registers)
        by_regs = max(0, regs_file // per_cta) if per_cta else 0
    candidates = [(base.by_smem, "smem"), (base.by_threads, "threads"),
                  (base.by_blocks, "cta")]
    if by_regs is not None:
        candidates.append((by_regs, "regs"))
    resident, binding = min(candidates, key=lambda t: t[0])
    return CellResidency(resident, base.by_smem, base.by_threads,
                         base.by_blocks, by_regs, base.smem_per_block,
                         registers, spills, binding)


# --------------------------------------------------------------------------
# The design matrix: computed, printed, and refused when it is singular.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Design:
    """The three-parameter design, as this card actually realises it.

    `rows` is one row per cell, `[1, log2 stages, log2 BLOCK_K, log2 resident]`.
    The intercept is column 0. `singular_values` is the full spectrum of the
    centred matrix, and `rank` is counted against a tolerance rather than
    asserted, because a rank taken from a determinant test is how a design that
    is one part in ten thousand from singular passes as identifiable.
    """

    keys: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    singular_values: tuple[float, ...]
    rank: int
    #: `sqrt(diag((X'X)^-1))` per parameter, the multiplier on the per-cell
    #: spread that gives each coefficient's standard error.
    se_multiplier: tuple[float, ...] | None

    @property
    def condition(self) -> float | None:
        if not self.singular_values or self.singular_values[-1] <= 0:
            return None
        return self.singular_values[0] / self.singular_values[-1]

    @property
    def identified(self) -> bool:
        return self.rank == 4 and self.se_multiplier is not None

    def lines(self) -> list[str]:
        out = ["  cell     log2 stages  log2 BK  log2 resident"]
        for key, row in zip(self.keys, self.rows, strict=True):
            out.append(f"  {key:8s} {row[1]:11.3f}  {row[2]:7.3f}  {row[3]:13.3f}")
        out.append("  singular values " + ", ".join(
            f"{v:.4f}" for v in self.singular_values)
            + f"; rank {self.rank} of 4"
            + ("" if self.condition is None
               else f"; condition {self.condition:.1f}"))
        if self.se_multiplier:
            names = ("intercept", "bD depth", "bK block_k", "bR residency")
            out.append("  standard error per unit of per-cell spread: " + ", ".join(
                f"{n} {m:.3f}" for n, m in zip(names, self.se_multiplier,
                                               strict=True)))
        else:
            out.append("  standard errors NOT STATEABLE: the design is singular, "
                       "so at least one coefficient is a combination of the "
                       "others and no amount of repeats resolves it")
        return out


def _matmul_t(a: list[list[float]]) -> list[list[float]]:
    n = len(a[0])
    return [[sum(r[i] * r[j] for r in a) for j in range(n)] for i in range(n)]


def _invert(m: list[list[float]]) -> list[list[float]] | None:
    """Gauss-Jordan with partial pivoting. None when the matrix is singular.

    Written out rather than taken from numpy because every other arithmetic
    path in this file runs on a laptop with no scientific stack, and a plan
    that cannot be printed without numpy is a plan that is not printed.
    """
    n = len(m)
    aug = [list(row) + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(m)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r == col or aug[r][col] == 0.0:
                continue
            factor = aug[r][col]
            aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col],
                                                     strict=True)]
    return [row[n:] for row in aug]


def _singular_values(a: list[list[float]]) -> list[float]:
    """Singular values of `a`, as the square roots of `A'A`'s eigenvalues.

    Jacobi eigenvalue iteration on the 4x4 symmetric `A'A`, which converges in a
    handful of sweeps at this size and needs no library.
    """
    m = [list(row) for row in _matmul_t(a)]
    n = len(m)
    for _ in range(100):
        off = max(((abs(m[i][j]), i, j) for i in range(n) for j in range(n)
                   if i != j), default=(0.0, 0, 0))
        if off[0] < 1e-14:
            break
        _, p, q = off
        if abs(m[p][p] - m[q][q]) < 1e-18:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * m[p][q], m[p][p] - m[q][q])
        c, s = math.cos(theta), math.sin(theta)
        rot = [row[:] for row in m]
        for k in range(n):
            rot[p][k] = c * m[p][k] + s * m[q][k]
            rot[q][k] = -s * m[p][k] + c * m[q][k]
        out = [row[:] for row in rot]
        for k in range(n):
            out[k][p] = c * rot[k][p] + s * rot[k][q]
            out[k][q] = -s * rot[k][p] + c * rot[k][q]
        m = out
    return sorted((math.sqrt(max(0.0, m[i][i])) for i in range(n)), reverse=True)


def build_design(keys, residencies: dict[str, CellResidency],
                 cells_by_key: dict[str, Cell]) -> Design:
    """The realised design, from the residency ladder this card actually has."""
    rows = []
    for key in keys:
        cell = cells_by_key[key]
        res = residencies[key].resident_blocks
        rows.append((1.0, math.log2(cell.num_stages), math.log2(cell.block_k),
                     math.log2(res) if res > 0 else 0.0))
    listed = [list(r) for r in rows]
    sv = _singular_values(listed) if len(listed) >= 4 else []
    rank = sum(1 for v in sv if v > (sv[0] * 1e-8 if sv else 0.0)) if sv else 0
    inverse = _invert(_matmul_t(listed)) if rank == 4 else None
    se = (tuple(math.sqrt(max(0.0, inverse[i][i])) for i in range(4))
          if inverse else None)
    return Design(tuple(keys), tuple(rows), tuple(sv), rank, se)


#: Per-thread register counts the sensitivity table walks. Not a prediction of
#: what ptxas will do: a span wide enough that the operator can see WHERE this
#: design stops being identifiable and read the census against it. 32 is about
#: the fp32 accumulator alone at eight warps; 255 is the hardware maximum.
REGISTER_PROBES = (32, 48, 64, 96, 128, 168, 224, 255)


def register_sensitivity(cells, b, limits, warps: int) -> list[str]:
    """Where the register file stops this design from separating anything.

    THE ONE TERM NEITHER THIS FILE NOR ITS SIBLING CAN COMPUTE, walked rather
    than assumed. Residency is `min(by_smem, by_threads, by_blocks, by_regs)`,
    and the last of those is `registers per SM / (32 x num_warps x n_regs)`.
    The first three do not depend on BLOCK_K at all, so a tight register bound
    flattens the WHOLE ladder onto one rung, `log2 resident` becomes a constant
    column, the design loses rank and every coefficient still prints.

    This table is why `--num-warps` exists and it is printed at the value given,
    so the operator chooses on the plan page. It is also what the compile census
    is read against before any timed cell is paid for.
    """
    file_size = registers_per_sm(limits.capability)
    out = ["REGISTER SENSITIVITY, the one limit this plan cannot compute. "
           f"At num_warps={warps} one CTA is {32 * warps} threads, so a card "
           + (f"with {file_size} registers per SM holds "
              f"{file_size // (32 * warps)} / n_regs CTAs."
              if file_size else
              "whose register file is unknown to this file holds an unknown "
              "number of CTAs, and the table below cannot be built."),
           "  n_regs  by_regs  resident-block ladder            rank  "
           "se(bD)  se(bK)  se(bR)"]
    if not file_size:
        return out
    for n_regs in REGISTER_PROBES:
        res = {c.key: cell_residency(c, b, limits, warps=warps,
                                     registers=n_regs) for c in cells}
        design = build_design([c.key for c in cells], res,
                              {c.key: c for c in cells})
        ladder = ",".join(str(res[c.key].resident_blocks) for c in cells)
        se = design.se_multiplier
        out.append(f"  {n_regs:6d}  {file_size // (32 * warps * n_regs):7d}  "
                   f"{ladder:32s} {design.rank:4d}  "
                   + ("  ".join(f"{v:6.3f}" for v in se[1:]) if se
                      else "   SINGULAR: this design separates nothing"))
    out.append("  A run whose measured n_regs lands on a SINGULAR row is "
               "refused by the compile census below, before a timed cell is "
               "paid for. The escape is --num-warps 4, which halves the "
               "threads per CTA and doubles by_regs at every n_regs.")
    return out


class CensusRefusal(RefusedBeforeMeasuring):
    """The compile census found a design that separates nothing.

    A SEPARATE CLASS because this refusal happens ON the pod, after CUDA is up
    and after one compile per cell, and it must be told apart in a log from the
    plan-time refusals that cost nothing. It still exits REFUSED: the arm
    measured no ladder and refuted no claim.
    """


def compile_census(args, cfg, cells, rows, cache_root: Path, probe, limits, b,
                   *, spread: float, spread_source: str):
    """One untimed call per cell, then read `n_regs` and re-check the design.

    WHY IT IS FREE. Every cell has to be compiled before it can be timed, so the
    census spends compiles the metered loop would spend anyway plus one call
    each. What it buys is the difference between a REFUSED log that cost two
    minutes and an INVALID one that cost the whole booking: if the register file
    has flattened the residency ladder onto one rung, V5 would fail after every
    timing was paid for, and nothing on that page would be quotable.

    IT REFUSES, IT DOES NOT ADAPT. Dropping a cell to rescue the rank, or
    nudging num_warps here, would change the design after the predictions were
    registered against it. The refusal names the measured `n_regs`, prints the
    realised ladder, and points at the flag.
    """
    import torch

    from moe.baselines._framework_config import vllm_call_kwargs
    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    override_config, _ = SWEEP.find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    tokens = SWEEP.tokens_for_rows(cfg, rows[0])
    spec = BenchSpec(cfg, num_tokens=tokens, dtype=args.dtype,
                     routing=RoutingSpec("uniform", 0.0), seed=args.seed)
    x, weights = make_inputs(spec, device="cuda")
    ids = SWEEP.balanced_ids(cfg, tokens, "cuda")
    tw = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                    device="cuda")
    kw = vllm_call_kwargs(spec)
    kw["activation"] = MoEActivation(kw["activation"])
    failed, compiles = {}, {}
    for cell in cells:
        arm_cache(cache_root, cell)
        seen: set[Path] = set()
        SWEEP.count_new(cache_root, seen)
        try:
            with override_config(cell.pinned(args.num_warps)):
                fused_experts(hidden_states=x, w1=weights.w1, w2=weights.w2,
                              topk_weights=tw, topk_ids=ids, **kw)
                torch.cuda.synchronize()
            # ATTRIBUTED HERE BECAUSE THE FIRST COMPILE HAPPENS HERE. V1 asks
            # whether every cell built its own kernel, and after the census the
            # metered loop finds the directory warm and counts zero for the
            # tile it just proved distinct -- so the count the gate reads is
            # this one plus whatever the loop adds for new token shapes.
            compiles[cell.key] = SWEEP.count_new(cache_root, seen)
            probe.record(cell.key)
        except Exception as exc:                        # noqa: BLE001
            failed[cell.key] = f"{type(exc).__name__}: {exc}"
            compiles.setdefault(cell.key, 0)
    del x, weights, ids, tw
    torch.cuda.empty_cache()

    observed = dict(probe.by_setting)
    residencies = {
        c.key: cell_residency(
            c, b, limits, warps=args.num_warps,
            registers=int(observed.get(c.key, {}).get("n_regs", 0) or 0) or None,
            spills=int(observed.get(c.key, {}).get("n_spills", 0) or 0))
        for c in cells}
    design = build_design([c.key for c in cells], residencies,
                          {c.key: c for c in cells})
    lines = ["", "## COMPILE CENSUS, one untimed call per cell, before the "
             "metered loop", ""]
    for cell in cells:
        lines.append(f"  {cell.key:8s} {residencies[cell.key].line()}"
                     + (f"   COMPILE FAILED: {failed[cell.key]}"
                        if cell.key in failed else ""))
    lines += design.lines()
    lines.append("  Triton entries built by the census: "
                 + ", ".join(f"{k}={v}" for k, v in sorted(compiles.items())))
    lines += [""] + mde_line(design, spread, spread_source)
    if probe.note:
        lines.append(f"  kernel probe: {probe.note}")
    print("\n".join(lines))
    if failed:
        raise CensusRefusal(
            f"{len(failed)} cell(s) could not be compiled as pinned: "
            + "; ".join(f"{k} ({v})" for k, v in sorted(failed.items()))
            + ". Every cell here is a rung of the design matrix, so dropping "
            "one changes its rank and the predictions were registered against "
            "the whole grid. Nothing was timed.")
    # THE READ-BACK IS WHAT V3 SCORES, AND THE CENSUS IS WHERE IT IS FREE.
    # `gate_residency` reads UNKNOWN when any cell carries only the residency
    # BOUND, and UNKNOWN there voids the page -- occupancy_vs_swizzle's V9 got
    # exactly that on two separate pods. Nothing in the metered loop supplies
    # n_regs, so a run that reaches the loop without it is fifteen minutes of
    # card bought to print INVALID. The two refusals below could not see it: a
    # ladder at the residency BOUND is still rank 4, so `design.identified`
    # is True and the compiles all succeeded. This costs nothing beyond the
    # compiles already paid for above.
    unread = sorted(c.key for c in cells
                    if not int(observed.get(c.key, {}).get("n_regs", 0) or 0))
    if unread:
        raise CensusRefusal(
            f"the kernel probe read no n_regs for {len(unread)} of "
            f"{len(cells)} cell(s): {', '.join(unread)}"
            + (f" (probe note: {probe.note})" if probe.note else "")
            + ". Residency is min(by_smem, by_threads, by_blocks, by_regs) and "
            "only the first depends on BLOCK_SIZE_K, so without n_regs every "
            "cell carries an upper BOUND and V3 reads UNKNOWN, which makes the "
            "whole page INVALID after the metered loop has been paid for. The "
            "probe reads `n_regs` off the compiled kernel object and `shared` "
            "off its metadata, and either can be absent on a Triton the probe "
            "does not know. Nothing was timed.")
    if not design.identified:
        raise CensusRefusal(
            "the design as this card REALISES it is singular: the register "
            "file has flattened the resident-block ladder and at least one of "
            "depth, BLOCK_K and residency is a combination of the others. "
            "Every coefficient would still print. Re-run with --num-warps "
            f"{max(1, args.num_warps // 2)}, which halves the threads per CTA "
            "and roughly doubles how many CTAs the register file holds; the "
            "register-sensitivity table above says at which n_regs each choice "
            "stops separating. Nothing was timed.")
    return observed, residencies, compiles


# --------------------------------------------------------------------------
# One timing of one tread of one cell: the CSV row.
# --------------------------------------------------------------------------

@dataclass
class Sample:
    """The instrument's own columns are part of the row, not of the report.

    `instrument`, `warmup_ms`, `iters`, `trials`, `l2_flush` and the seven clock
    fields come straight off the `moe.bench.timing.KernelTiming` that produced
    `ms_p50`. A row that carries them can be excluded by a later reader; a row
    that does not cannot. A planted row carries `SWEEP.SYNTHETIC_INSTRUMENT`, so
    "not measured" is a VALUE in the column and not an absence.

    The clock flags are Optional and None means NOT DETERMINED -- no NVML, a
    trial too short for the poller, no calibration to compare a level against --
    never "fine". `clock_level_side` says which way a LEVEL failure went and is
    a RECORD: since 2026-09-09 neither side excludes, because under the power
    cap the under-load clock is an outcome of the tile, and this arm sweeps
    eight cells.
    """

    cell: str
    num_stages: int
    block_k: int
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
    instrument: str = ""
    warmup_ms: float = 0.0
    trials: int = 0
    l2_flush: bool = False
    sm_clock_load_mhz: float | None = None
    sm_clock_start_mhz: float | None = None
    sm_clock_end_mhz: float | None = None
    clock_samples_mhz: str = ""
    power_w: float | None = None
    clock_level_ok: bool | None = None
    clock_drift_ok: bool | None = None
    host_bound: bool | None = None
    clock_level_side: str = ""
    #: What Triton compiled, read back where the platform allows it. Zero means
    #: NOT READ, never "no registers": `has_probe` is the predicate.
    compiled_smem: int = 0
    compiled_regs: int = 0
    compiled_spills: int = 0


SAMPLE_FIELDS = list(Sample.__dataclass_fields__)


def clock_excluded(level_ok: bool | None, side: str,
                   drift_ok: bool | None) -> bool:
    """Do a tread's clock verdicts exclude it. DRIFT does; NO LEVEL SIDE DOES.

    THE ONE FUNCTION IN THIS FILE THAT EXCLUDES ON THE CLOCK VERDICTS, and
    `tests/test_blockk_diagonal.py` asserts by AST that the only OTHER function
    reading them is `clock_state`, which counts and drops nothing. Written as
    "the one reader" this comment was false in its own file by one function.
    This
    repository's recurring defect is a rule applied at one of N call sites, and
    a clock rule is the exact shape of it: the sibling arm carried the old
    LEVEL-LOW exclusion into `fit_ladder` and left four gates scoring the
    unexcluded medians for a commit.

    THE RULE, which is the apparatus standard and is not re-decided here.
    DRIFT excludes: the clock MOVED while the tread was timed, so the median
    load is a blend of two operating points and the time is not a time at one
    of them. Both sides of a LEVEL failure are KEPT with `side` recorded,
    because under the 700 W cap the under-load clock is an outcome of the
    TILE -- and an eight-cell sweep will produce several operating points by
    construction, so a band around the calibration GEMM's own point would
    exclude cells and not defects. `level_ok` and `side` are taken here so that
    the signature states what the rule reads and what it declines to read.
    """
    del level_ok, side
    return drift_ok is False


def clock_state(samples: list[Sample]) -> dict:
    """How many timed treads sat where, in counts that PARTITION the treads.

    The four LEVEL counts are over the STEADY treads only, `drift` and
    `host_bound` take the rest, so the six sum to `timed` and no tread is
    called kept and excluded-shaped in one sentence. A tread that both drifted
    and sat HIGH was counted twice by the version of this block that filtered
    on LEVEL alone.

    HOST-BOUND IS COUNTED HERE BECAUSE `collapse` DROPS IT. The exclusion rule
    has three call sites -- `collapse` drops on `clock_excluded(...) or
    host_bound`, `measure` prints on the same disjunction, and this counter --
    and this one implemented the clock half alone, so V0's "timings kept",
    which is `timed - excluded`, reported sixteen kept timings that no fit had
    used and the CLOCK block printed "0 DRIFT failed" over a grid with nothing
    in it. `clock_excluded` is still the ONE function that decides the CLOCK
    half; host-bound is not a clock verdict and is read here directly.
    """
    from moe.bench import timing

    timed = [s for s in samples if s.status == "ok"]
    drifted = [s for s in timed if s.clock_drift_ok is False]
    host = [s for s in timed if s.clock_drift_ok is not False and s.host_bound]
    steady = [s for s in timed
              if s.clock_drift_ok is not False and not s.host_bound]

    def sides(rows: list[Sample]) -> dict:
        return {
            "level": sum(1 for s in rows if s.clock_level_ok is True),
            "low": sum(1 for s in rows if s.clock_level_ok is False
                       and s.clock_level_side != timing.LEVEL_HIGH),
            "high": sum(1 for s in rows if s.clock_level_ok is False
                        and s.clock_level_side == timing.LEVEL_HIGH),
            "unknown": sum(1 for s in rows if s.clock_level_ok is None),
        }

    kept, moved = sides(steady), sides(drifted)
    return {
        "timed": len(timed), "level": kept["level"], "low": kept["low"],
        "high": kept["high"], "drift": len(drifted), "host_bound": len(host),
        "unknown": kept["unknown"],
        "drift_level": moved["level"], "drift_low": moved["low"],
        "drift_high": moved["high"], "drift_unknown": moved["unknown"],
        # THE SAME DISJUNCTION `collapse` FITS ON, so "timings kept" is the
        # count of timings a fit could actually read.
        "excluded": sum(1 for s in timed
                        if clock_excluded(s.clock_level_ok,
                                          s.clock_level_side,
                                          s.clock_drift_ok) or s.host_bound),
        "rule": "DRIFT excludes and so does a HOST-BOUND interval, which "
                "bounds the kernel from above; BOTH LEVEL sides are kept with "
                "the side recorded; the four LEVEL counts are over the treads "
                "that are neither, and the six counts partition the timed "
                "treads",
    }


def clock_state_lines(state: dict) -> list[str]:
    out = [f"  {state['timed']} timed treads: {state['level']} steady level, "
           f"{state['low']} steady LOW (kept, side recorded), "
           f"{state['high']} steady HIGH (kept, side recorded), "
           f"{state['drift']} DRIFT failed (excluded, and not in the three "
           f"counts before it), {state['host_bound']} HOST-BOUND (excluded, "
           f"steady clock and an interval that bounds the kernel from above), "
           f"{state['unknown']} steady with LEVEL not determined"]
    if state["drift"]:
        out.append(f"  the {state['drift']} drifted treads by side: "
                   f"{state['drift_level']} level, {state['drift_low']} LOW, "
                   f"{state['drift_high']} HIGH, {state['drift_unknown']} with "
                   "LEVEL not determined; a tread whose clock moved is not "
                   "steady at any side, so none is counted as kept")
    return out


def append_sample(path: Path, sample: Sample, prov=None) -> None:
    """One row, flushed. An abort costs the timing in flight and nothing else.

    APPENDING TO A FILE WITH A DIFFERENT HEADER REFUSES. `DictWriter` writes the
    fieldnames it was given and never looks at the file, so a wider row appended
    under a narrower header shifts every field past the first difference and
    nothing downstream can tell: `clock_drift_ok` would read the LEVEL side, and
    a FAILED drift -- the one rule in this tree that excludes a tread -- would
    come back None, which every gate keeps.
    """
    new = not path.exists()
    row = asdict(sample)
    if prov is not None:
        row.update(prov.as_columns())
    if not new:
        with path.open(newline="") as fh:
            header = next(csv.reader(fh), [])
        if header and header != list(row):
            added = [c for c in row if c not in header]
            gone = [c for c in header if c not in row]
            raise RefusedBeforeMeasuring(
                f"cells.csv at {path} was written with a different set of "
                f"columns than this run writes (added: {added or 'none'}; "
                f"missing: {gone or 'none'}). Appending would put the new "
                "fields under the old header and misalign every row from here "
                "on. Re-run into a NEW directory rather than resuming this one.")
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def _opt_float(text: str | None) -> float | None:
    try:
        return float(text) if text not in (None, "", "None") else None
    except ValueError:
        return None


def _opt_bool(text: str | None) -> bool | None:
    if text in (None, "", "None"):
        return None
    return str(text).strip().lower() in ("1", "true", "yes")


def read_samples(path: Path) -> tuple[set[tuple[str, int, int]], list[Sample]]:
    """Timings already on disk, read BY HEADER NAME, so a re-run resumes.

    Only SUCCESSFUL timings count as done: the common failure is a pod that lost
    its device or a cell that ran out of shared memory, and a real failure fails
    again in milliseconds. Optional fields read `""` back as None and never as 0
    or False, so a file written before a column existed is visibly missing it.
    """
    if not path.exists():
        return set(), []
    out: list[Sample] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Sample(
                cell=row["cell"], num_stages=int(row["num_stages"]),
                block_k=int(row["block_k"]), tiles=int(row["tiles"]),
                rows_per_expert=int(row["rows_per_expert"]),
                tokens=int(row["tokens"]), rep=int(row["rep"]),
                ms_p50=float(row["ms_p50"]), ms_min=float(row["ms_min"]),
                ms_stdev=float(row["ms_stdev"]), iters=int(row["iters"]),
                status=row.get("status", "ok"), detail=row.get("detail", ""),
                instrument=row.get("instrument", ""),
                warmup_ms=_opt_float(row.get("warmup_ms")) or 0.0,
                trials=int(row.get("trials") or 0),
                l2_flush=bool(_opt_bool(row.get("l2_flush"))),
                sm_clock_load_mhz=_opt_float(row.get("sm_clock_load_mhz")),
                sm_clock_start_mhz=_opt_float(row.get("sm_clock_start_mhz")),
                sm_clock_end_mhz=_opt_float(row.get("sm_clock_end_mhz")),
                clock_samples_mhz=row.get("clock_samples_mhz", "") or "",
                power_w=_opt_float(row.get("power_w")),
                clock_level_ok=_opt_bool(row.get("clock_level_ok")),
                clock_drift_ok=_opt_bool(row.get("clock_drift_ok")),
                host_bound=_opt_bool(row.get("host_bound")),
                clock_level_side=row.get("clock_level_side", "") or "",
                compiled_smem=int(row.get("compiled_smem") or 0),
                compiled_regs=int(row.get("compiled_regs") or 0),
                compiled_spills=int(row.get("compiled_spills") or 0)))
    return ({(s.cell, s.tiles, s.rep) for s in out if s.status == "ok"}, out)


# --------------------------------------------------------------------------
# The fit: one w per cell, then three coefficients over the cells.
# --------------------------------------------------------------------------

def collapse(samples: list[Sample], cell: str, rng=None
             ) -> tuple[list[tuple[int, float]], float | None]:
    """Per-tread median across repeats, and the median across-repeat spread.

    The median across REPEATS rather than one pass's own median: a repeat is a
    fresh call at a fresh point in the pod's thermal history. With `rng` the
    repeats are resampled WITH REPLACEMENT, which is the bootstrap, and
    everything downstream -- the slope, w, the three coefficients and the
    verdict -- is recomputed on that draw.

    DRIFTED TREADS ARE DROPPED HERE AND NOWHERE ELSE, through the one reader.
    """
    by: dict[int, list[float]] = {}
    for s in samples:
        if (s.cell == cell and s.status == "ok" and s.ms_p50 > 0
                and not clock_excluded(s.clock_level_ok, s.clock_level_side,
                                       s.clock_drift_ok)
                and not s.host_bound):
            by.setdefault(s.tiles, []).append(s.ms_p50)
    points = []
    for n, vals in sorted(by.items()):
        draw = [rng.choice(vals) for _ in vals] if rng is not None else vals
        points.append((n, statistics.median(draw)))
    spreads = [statistics.pstdev(v) / statistics.median(v)
               for v in by.values() if len(v) > 1 and statistics.median(v) > 0]
    return points, (statistics.median(spreads) if spreads else None)


def ols(rows: list[list[float]], ys: list[float]) -> list[float] | None:
    """Ordinary least squares, or None when the normal equations are singular."""
    if len(rows) < len(rows[0]):
        return None
    inverse = _invert(_matmul_t(rows))
    if inverse is None:
        return None
    rhs = [sum(r[i] * y for r, y in zip(rows, ys, strict=True))
           for i in range(len(rows[0]))]
    return [sum(inverse[i][j] * rhs[j] for j in range(len(rhs)))
            for i in range(len(rhs))]


def line_fit(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    """`ms = A + B n` over the treads. Returns `(A, B)`, or None."""
    if len(points) < 2:
        return None
    beta = ols([[1.0, float(n)] for n, _ in points], [ms for _, ms in points])
    return (beta[0], beta[1]) if beta else None


@dataclass(frozen=True)
class CellFit:
    """One cell's ladder, its slope, and the slope in weight-stream units."""

    key: str
    cell: Cell
    treads: int
    intercept_ms: float
    slope_ms: float
    spread: float | None
    #: `moe.bench.weights.WeightStreamSlope`; `w` is `.streams`.
    w: object
    #: Marginal arithmetic intensity of one more M-tile, in Op/B, computed from
    #: the MEASURED w rather than from the byte model.
    marginal_ai: float
    inversions: int

    @property
    def streams(self) -> float:
        return self.w.streams


def fit_cell(cfg, key: str, cell: Cell, samples: list[Sample], *, dtype: str,
             bandwidth_gbps: float, bandwidth_source: str, rng=None
             ) -> CellFit | None:
    """Fit one cell's ladder and divide its slope by one weight stream."""
    points, spread = collapse(samples, key, rng=rng)
    fit = line_fit(points)
    if fit is None or len(points) < 2:
        return None
    intercept, slope = fit
    try:
        w = WEIGHTS.weight_streams_per_tile(
            slope, cfg, dtype, bandwidth_gbps, bandwidth_source=bandwidth_source)
    except WEIGHTS.WeightSetRefused:
        return None
    rows_per_tread = SUBJECT_BLOCK_M * cfg.num_experts
    flops = SWEEP.useful_flops(cfg, rows_per_tread)
    bytes_per_tread = abs(w.streams) * w.weight_bytes
    ai = flops / bytes_per_tread if bytes_per_tread > 0 else float("inf")
    drops = sum(1 for (_, a), (_, c) in zip(points, points[1:], strict=False)
                if c < a)
    return CellFit(key, cell, len(points), intercept, slope, spread, w, ai,
                   drops)


@dataclass(frozen=True)
class Coefficients:
    """The three-parameter reading, and the spread of each coefficient."""

    intercept: float
    depth: float
    block_k: float
    residency: float
    sd: dict[str, float | None]
    residual_rms: float | None
    dof: int

    def value(self, name: str) -> float:
        return {"depth": self.depth, "block_k": self.block_k,
                "residency": self.residency}[name]

    def moves(self, name: str) -> bool | None:
        """Did this knob move w by more than `K_SIGMA` of its OWN spread."""
        sd = self.sd.get(name)
        if sd is None or sd <= 0:
            return None
        return abs(self.value(name)) > K_SIGMA * sd

    def line(self, name: str) -> str:
        sd = self.sd.get(name)
        return (f"{self.value(name):+.4f}"
                + (" +/- NOT STATEABLE" if sd is None else f" +/- {sd:.4f}")
                + f" per doubling (gate {K_SIGMA:.0f} sigma)")


def fit_coefficients(fits: dict[str, CellFit],
                     residencies: dict[str, CellResidency],
                     sd: dict[str, float | None] | None = None
                     ) -> Coefficients | None:
    """`ln w = b0 + bD log2 stages + bK log2 BK + bR log2 resident`.

    In logs because every candidate mechanism here is multiplicative on a cost
    and because the coefficients then read as "fractional change in w per
    doubling", which is the unit both rival predictions are stated in.
    """
    rows, ys, keys = [], [], []
    for key, fit in sorted(fits.items()):
        res = residencies.get(key)
        if res is None or res.resident_blocks <= 0 or fit.streams <= 0:
            continue
        rows.append([1.0, math.log2(fit.cell.num_stages),
                     math.log2(fit.cell.block_k),
                     math.log2(res.resident_blocks)])
        ys.append(math.log(fit.streams))
        keys.append(key)
    if len(rows) < 4:
        return None
    beta = ols(rows, ys)
    if beta is None:
        return None
    resid = [y - sum(b * x for b, x in zip(beta, r, strict=True))
             for r, y in zip(rows, ys, strict=True)]
    dof = len(rows) - 4
    rms = (math.sqrt(sum(v * v for v in resid) / dof) if dof > 0 else None)
    return Coefficients(beta[0], beta[1], beta[2], beta[3],
                        sd or {"depth": None, "block_k": None,
                               "residency": None}, rms, dof)


@dataclass(frozen=True)
class PairContrast:
    """One iso-shared-memory pair, read model-free, in w.

    The headline table. The three-parameter fit pools eight cells and is the
    reading that separates BLOCK_K from depth; these two are the contrast the
    design is NAMED for and they are quoted without a model on top of them,
    because a pooled coefficient that disagrees with the pair it is supposed to
    summarise is a thing a reader has to be able to see.
    """

    #: `from` and `to` are the REGISTERED ORDER of the pair, which is not the
    #: order of their depths: `("4x64", "2x128")` runs 4 stages to 2. There is
    #: no `iso_smem` property here any more; it returned the literal True, had
    #: no call site, and read like a check. `gate_iso_smem` is the check.
    from_key: str
    to_key: str
    from_stages: int
    to_stages: int
    smem_bytes: int
    resident_from: int
    resident_to: int
    w_from: float
    w_to: float
    delta_per_doubling: float | None
    sd: float | None

    def line(self) -> str:
        delta = ("NOT STATEABLE" if self.delta_per_doubling is None
                 else f"{self.delta_per_doubling:+.4f}"
                      + ("" if self.sd is None else f" +/- {self.sd:.4f}"))
        return (f"  {self.from_key} -> {self.to_key}  "
                f"{self.smem_bytes / 1024:.0f} KiB/CTA both, depth "
                f"{self.from_stages} -> {self.to_stages}, resident "
                f"{self.resident_from} and {self.resident_to}: w "
                f"{self.w_from:.4f} -> {self.w_to:.4f}, "
                f"d ln w per depth doubling {delta}")


def pair_contrast(pair: tuple[str, str], fits: dict[str, CellFit],
                  residencies: dict[str, CellResidency], b: int,
                  sd: float | None = None) -> PairContrast | None:
    from_key, to_key = pair
    lo, hi = fits.get(from_key), fits.get(to_key)
    if lo is None or hi is None or lo.streams <= 0 or hi.streams <= 0:
        return None
    ratio = math.log2(hi.cell.num_stages / lo.cell.num_stages)
    delta = ((math.log(hi.streams) - math.log(lo.streams)) / ratio
             if ratio else None)
    return PairContrast(
        from_key, to_key, lo.cell.num_stages, hi.cell.num_stages,
        lo.cell.smem_bytes(b),
        residencies[from_key].resident_blocks if from_key in residencies else 0,
        residencies[to_key].resident_blocks if to_key in residencies else 0,
        lo.streams, hi.streams, delta, sd)


def verdict_of(coef: Coefficients | None) -> str:
    """Which knob the data picked, INCLUDING the null and the unreadable case."""
    if coef is None:
        return VERDICT_NEITHER
    depth, res = coef.moves("depth"), coef.moves("residency")
    if depth and res:
        return VERDICT_BOTH
    if depth:
        return VERDICT_DEPTH
    if res:
        return VERDICT_RESIDENCY
    return VERDICT_NEITHER


# --------------------------------------------------------------------------
# The bootstrap.
# --------------------------------------------------------------------------

def bootstrap(cfg, cells_by_key, samples, residencies, *, dtype, bandwidth_gbps,
              bandwidth_source, draws: int, seed: int, b: int
              ) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Resample REPEATS with replacement and refit everything on every draw.

    The interval on each coefficient then carries the instability of the whole
    chain -- the per-cell ladder fit, the division by a stream, the three-
    parameter regression -- and not only the scatter of the timings. Returns the
    coefficient spreads and the per-pair contrast spreads.
    """
    rng = random.Random(seed)
    coefs: dict[str, list[float]] = {"depth": [], "block_k": [], "residency": []}
    pairs: dict[str, list[float]] = {}
    for _ in range(max(0, draws)):
        fits = {}
        for key, cell in cells_by_key.items():
            fit = fit_cell(cfg, key, cell, samples, dtype=dtype,
                           bandwidth_gbps=bandwidth_gbps,
                           bandwidth_source=bandwidth_source, rng=rng)
            if fit is not None:
                fits[key] = fit
        coef = fit_coefficients(fits, residencies)
        if coef is not None:
            for name in coefs:
                coefs[name].append(coef.value(name))
        for pair in ISO_SMEM_PAIRS:
            got = pair_contrast(pair, fits, residencies, b)
            if got is not None and got.delta_per_doubling is not None:
                pairs.setdefault("/".join(pair), []).append(
                    got.delta_per_doubling)

    def sd(values: list[float]) -> float | None:
        return statistics.pstdev(values) if len(values) > 2 else None

    return ({k: sd(v) for k, v in coefs.items()},
            {k: sd(v) for k, v in pairs.items()})


# --------------------------------------------------------------------------
# Gates.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """One pre-registered prediction and the number that settled it.

    `passed=None` prints UNKNOWN and never PASS: a check that could not run is
    not a check that passed.
    """

    kind: str
    name: str
    prediction: str
    rule: str
    passed: bool | None
    observed: str
    invalidates: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def token(self) -> str:
        """The first word of `name`, so the printed heading and the machine
        line can never disagree about which gate they are."""
        return self.name.split()[0]

    def scored(self) -> tuple[str, str, str]:
        return (self.kind, self.token,
                {True: exit_codes.PASS, False: exit_codes.FAIL,
                 None: exit_codes.UNKNOWN}[self.passed])

    def result_line(self) -> str:
        detail = f"{self.prediction} | saw {self.observed} | gate {self.rule}"
        return exit_codes.result_line(*self.scored(), " ".join(detail.split()))

    def render(self) -> list[str]:
        tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}[self.passed]
        out = [self.result_line(),
               f"[{tag}] {self.kind:8s} {self.name}  {self.prediction}",
               f"         gate: {self.rule}",
               f"         saw:  {self.observed}"]
        if self.passed is not True and self.invalidates:
            out.append(f"         a non-PASS here invalidates: {self.invalidates}")
        return out + [f"         {line}" for line in self.lines]


def render_gates(gates: list[Gate]) -> list[str]:
    """Every gate, each with exactly one RESULT line, then a prose tally.

    The tally deliberately does not look like a result: a summary line that
    could be parsed as a verdict is how a count of gates becomes a gate.
    """
    out: list[str] = []
    for g in gates:
        out += g.render()
    if not gates:
        return out + ["", "NO GATES WERE SCORED. Nothing on this page is a "
                          "verdict."]
    npass = sum(1 for g in gates if g.passed is True)
    nfail = sum(1 for g in gates if g.passed is False)
    nunk = sum(1 for g in gates if g.passed is None)
    implied = exit_codes.classify(g.scored() for g in gates)
    return out + ["", f"{npass} PASS, {nfail} FAIL, {nunk} UNKNOWN",
                  f"the gates imply {exit_codes.describe(implied)}"]


def gate_non_vacuity(counts: dict[str, int]) -> Gate:
    """A check that examined nothing also reports zero failures.

    Every gate below can pass by having no input. This one asserts the input
    existed and names the counts, so a reader sees WHICH work happened.
    """
    empty = sorted(k for k, v in counts.items() if v <= 0)
    return Gate(VALIDITY, "V0 non-vacuity", "this report examined real work",
                "every counted quantity is above zero", not empty,
                ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                "every gate in this report: a check with no input reports no "
                "failures",
                [f"nothing was counted for: {', '.join(empty)}"] if empty else [])


def gate_geometry(compiles: dict[str, int], executed: dict[str, int],
                  observed: dict[str, dict]) -> Gate:
    """V1: did every cell compile its OWN kernel and run the geometry it was given.

    If `override_config` silently failed, all eight cells ran ONE kernel, w is
    identical across the grid by construction, all three coefficients come out
    exactly zero and the residual is exactly noise -- a tidy, false NEITHER on
    a page whose whole content is a comparison across those eight kernels.

    TWO FACTS, because either alone is satisfiable by an accident. Each cell's
    own Triton cache directory gained at least one entry while it was timed
    (so a distinct kernel was built for it), and where the compiled kernel's
    metadata could be read at all its shared memory differs across cells that
    differ in `stages x BLOCK_K` -- which is the one thing a shared kernel
    could not produce. The PER-CELL agreement between the read-back and the
    model is V2's; this gate asks only that the grid is not one kernel.

    WHAT THE READ-BACK CANNOT SEE, AND IT IS SAID HERE RATHER THAN LEFT TO BE
    NOTICED. The two members of a registered pair have EQUAL shared memory by
    construction -- that is what makes them a pair -- so `metadata.shared`
    cannot tell (6,32) from (3,64). If `override_config` delivered the wrong
    member of a pair, this gate and V2 would both pass. The probe this arm
    borrows records `shared`, `n_regs` and `n_spills` and not `num_stages`, so
    nothing here closes that; what does is upstream in the session, where the
    two `pin_probe` arms establish that the override reaches the kernel at all
    before any arm that pins a tile is run. An arm run outside that session is
    resting on the context manager's word.
    """
    silent = sorted(k for k, n in executed.items() if n > 0
                    and compiles.get(k, 0) <= 0)
    smems = {k: v.get("shared") for k, v in observed.items() if v.get("shared")}
    distinct = len(set(smems.values()))
    lines = ["compiles per cell: "
             + ", ".join(f"{k}={compiles.get(k, 0)}" for k in sorted(executed)),
             "compiled shared memory per cell: "
             + (", ".join(f"{k}={v}" for k, v in sorted(smems.items()))
                if smems else "NOT READ on this platform"),
             "the two members of a registered pair have EQUAL shared memory by "
             "construction, so this read-back cannot tell them apart; the pin "
             "itself is what the session's pin_probe arms establish"]
    if silent:
        return Gate(VALIDITY, "V1 geometry", "every cell ran its own kernel",
                    "at least one new Triton entry per timed cell",
                    False, f"{len(silent)} cell(s) compiled nothing: "
                           + ", ".join(silent),
                    "the whole page: a grid that ran one kernel has swept "
                    "nothing and every coefficient is zero by construction",
                    lines)
    if not smems:
        return Gate(VALIDITY, "V1 geometry", "every cell ran its own kernel",
                    "at least one new Triton entry per timed cell, and "
                    "distinct compiled shared memory across distinct settings",
                    None, "every cell compiled, but no compiled shared memory "
                          "could be read back to confirm the settings differed",
                    "the cross-check on the pin. The compile counts are "
                    f"consistent with {len(executed)} kernels and cannot "
                    "prove it",
                    lines)
    return Gate(VALIDITY, "V1 geometry", "every cell ran its own kernel",
                "at least one new Triton entry per timed cell, and at least "
                "two distinct compiled shared-memory sizes across the grid",
                distinct >= 2,
                f"{len(executed)} cells, all compiled; {distinct} distinct "
                f"compiled shared-memory size(s) over {len(smems)} read back",
                "the whole page: a grid that ran one kernel has swept nothing",
                lines)


def gate_iso_smem(pairs: list[PairContrast], cells_by_key, b: int,
                  observed: dict[str, dict], tolerance: float) -> Gate:
    """V2: do the paired cells really have equal shared memory.

    The design's entire claim to separate residency from depth is that each
    pair holds the shared-memory footprint fixed while moving the depth. If the
    computed footprints differ the pairing is not a pairing; if Triton allocates
    something other than `stages x (BM BK + BK BN) x b` the computed figure is
    the wrong number to have held fixed.

    BOTH ARE SCORED AND THE REPORT SAYS WHICH IT GOT. The computed equality is
    arithmetic and is always available. The read-back is a platform courtesy:
    `occupancy_vs_swizzle`'s V9 could not read it on either of two pods, and
    the honest rendering of that is UNKNOWN on this gate rather than a PASS
    taken from the half that is free.
    """
    lines = []
    computed_ok = True
    for pair in ISO_SMEM_PAIRS:
        lo, hi = (cells_by_key.get(k) for k in pair)
        if lo is None or hi is None:
            continue
        a, c = lo.smem_bytes(b), hi.smem_bytes(b)
        computed_ok &= (a == c)
        lines.append(
            f"{pair[0]}: {lo.num_stages} x {lo.block_k} x "
            f"({SUBJECT_BLOCK_M} + {PINNED_BLOCK_N}) x {b} = {a} B; "
            f"{pair[1]}: {hi.num_stages} x {hi.block_k} x "
            f"({SUBJECT_BLOCK_M} + {PINNED_BLOCK_N}) x {b} = {c} B"
            + ("  EQUAL" if a == c else "  NOT EQUAL"))
    if not pairs:
        return Gate(VALIDITY, "V2 iso-smem", "the paired cells hold the "
                    "shared-memory footprint fixed",
                    "computed equality within each registered pair, and "
                    f"read-back within {tolerance:.0%} where the platform "
                    "allows it",
                    None, "no registered pair was formed from this run's cells",
                    "the depth contrast: without a pair there is no contrast "
                    "at fixed footprint", lines)
    read = {k: v.get("shared") for k, v in observed.items() if v.get("shared")}
    worst = 0.0
    for key, got in read.items():
        cell = cells_by_key.get(key)
        if cell is None or not cell.smem_bytes(b):
            continue
        worst = max(worst, abs(got / cell.smem_bytes(b) - 1.0))
    if not computed_ok:
        return Gate(VALIDITY, "V2 iso-smem", "the paired cells hold the "
                    "shared-memory footprint fixed",
                    "computed equality within each registered pair",
                    False, "a registered pair's two cells do not have equal "
                           "computed shared memory",
                    "the depth contrast, and with it C1: a pair that moved the "
                    "footprint moved residency too", lines)
    if not read:
        return Gate(VALIDITY, "V2 iso-smem", "the paired cells hold the "
                    "shared-memory footprint fixed",
                    "computed equality within each registered pair, and "
                    f"read-back within {tolerance:.0%}",
                    None, "computed footprints are equal in every registered "
                          "pair; NO compiled shared memory could be read back "
                          "on this platform, exactly as occupancy_vs_swizzle's "
                          "V9 could not on two pods",
                    "the check that the model of Triton's allocation is the "
                    "right thing to have held fixed. The pairing is equal in "
                    "the model and unverified against the compiler", lines)
    lines.append(f"read back for {len(read)} of {len(cells_by_key)} cells; "
                 f"worst disagreement with the model {worst:.1%}")
    return Gate(VALIDITY, "V2 iso-smem", "the paired cells hold the "
                "shared-memory footprint fixed",
                "computed equality within each registered pair, and read-back "
                f"within {tolerance:.0%} of the model",
                worst <= tolerance,
                f"pairs equal as computed; worst read-back disagreement "
                f"{worst:.1%} over {len(read)} cells",
                "the check that the model of Triton's allocation is the right "
                "thing to have held fixed", lines)


def gate_residency(residencies: dict[str, CellResidency],
                   pairs: list[PairContrast]) -> Gate:
    """V3: is the resident-block ladder MEASURED, or still an upper bound.

    Residency is the axis this arm contrasts, so a computed upper bound is not
    a measurement of it. The register limit is the one term neither this file
    nor its sibling can compute, so it is read off the compiled kernel; a run
    that could not read `n_regs` anywhere has an unverified x axis and this
    gate is UNKNOWN, which makes the page INVALID and unquotable. That is the
    correct outcome: the pairing might be equal, or the register file might
    have flattened the whole ladder onto one rung, and nothing on the page can
    tell which.
    """
    lines = [f"{k}: {r.line()}" for k, r in sorted(residencies.items())]
    if not residencies:
        return Gate(VALIDITY, "V3 residency", "the resident-block ladder is "
                    "measured, not bounded",
                    "every cell's n_regs read back off its compiled kernel",
                    None, "no residency was computed at all",
                    "the x axis of the residency contrast", lines)
    bounded = sorted(k for k, r in residencies.items() if r.bound)
    if bounded:
        return Gate(VALIDITY, "V3 residency", "the resident-block ladder is "
                    "measured, not bounded",
                    "every cell's n_regs read back off its compiled kernel",
                    None,
                    f"{len(bounded)} of {len(residencies)} cells carry NO "
                    "register count, so their residency is an UPPER BOUND: "
                    + ", ".join(bounded),
                    "the x axis: an upper bound cannot say whether two cells "
                    "share a rung, which is the whole content of the depth "
                    "contrast", lines)
    tied = [p for p in pairs if p.resident_from == p.resident_to]
    return Gate(VALIDITY, "V3 residency", "the resident-block ladder is "
                "measured, not bounded",
                "every cell's n_regs read back, and every registered pair "
                "lands on ONE resident-block count",
                bool(pairs) and len(tied) == len(pairs),
                f"{len(residencies)} cells with a register count; "
                f"{len(tied)} of {len(pairs)} registered pairs share a "
                "resident-block count",
                "the depth contrast: a pair whose two cells sit on different "
                "rungs moved residency as well as depth, which is the confound "
                "this arm exists to remove", lines)


def gate_treads(fits: dict[str, CellFit], ridge: float, ridge_source: str,
                cells: tuple[Cell, ...]) -> Gate:
    """V4: enough MEMORY-BOUND treads per cell to fit a slope.

    Two facts, and the second is measured rather than assumed. Every cell
    contributed at least `MIN_FIT_TREADS` treads to its own fit after the
    DRIFT exclusion, and every cell's MARGINAL arithmetic intensity -- the
    useful flops one more M-tile adds, over the bytes its measured w says that
    tile costs -- sits below the calibrated ridge by the registered margin. A
    cell at or above the ridge has a compute slope, not a weight-traffic slope,
    and its w is not a fraction of a stream of anything.
    """
    lines = [f"{f.key}: {f.treads} treads, slope {f.slope_ms:.5f} ms/M-tile, "
             f"w {f.streams:.4f}, marginal AI {f.marginal_ai:.1f} Op/B "
             f"({f.marginal_ai / ridge:.2f} of the ridge), spread "
             + ("NOT STATEABLE" if f.spread is None else f"{f.spread:.4%}")
             + (f", {f.inversions} inversion(s)" if f.inversions else "")
             for f in sorted(fits.values(), key=lambda v: v.key)]
    lines.append(f"ridge {ridge:.3f} Op/B, {ridge_source}")
    missing = sorted(c.key for c in cells if c.key not in fits)
    thin = sorted(f.key for f in fits.values() if f.treads < MIN_FIT_TREADS)
    hot = sorted(f.key for f in fits.values()
                 if f.marginal_ai >= AI_BELOW_RIDGE * ridge)
    loud = sorted(f.key for f in fits.values()
                  if f.spread is not None and f.spread > MAX_REPLICATE_SPREAD)
    problems = []
    if missing:
        problems.append(f"{len(missing)} cell(s) produced no fit: "
                        + ", ".join(missing))
    if thin:
        problems.append(f"{len(thin)} cell(s) under {MIN_FIT_TREADS} treads: "
                        + ", ".join(thin))
    if hot:
        problems.append(f"{len(hot)} cell(s) at or above "
                        f"{AI_BELOW_RIDGE:.0%} of the ridge: " + ", ".join(hot))
    if loud:
        problems.append(f"{len(loud)} cell(s) over {MAX_REPLICATE_SPREAD:.0%} "
                        "across-repeat spread: " + ", ".join(loud))
    return Gate(VALIDITY, "V4 treads", "every cell fitted a memory-bound slope",
                f"at least {MIN_FIT_TREADS} kept treads per cell, marginal AI "
                f"under {AI_BELOW_RIDGE:.0%} of the ridge, and across-repeat "
                f"spread under {MAX_REPLICATE_SPREAD:.0%}",
                not problems,
                "; ".join(problems) if problems
                else f"all {len(fits)} cells fitted, "
                     f"{min(f.treads for f in fits.values())} treads at worst, "
                     f"marginal AI at most "
                     f"{max(f.marginal_ai for f in fits.values()) / ridge:.2f} "
                     "of the ridge",
                "every w on the page, and with it all three coefficients",
                lines)


def gate_identifiable(design: Design, residencies: dict[str, CellResidency],
                      coef: Coefficients | None) -> Gate:
    """V5: does the design AS REALISED separate the three mechanisms.

    The hazard this gate exists for is the register file. Residency is
    `min(by_smem, by_threads, by_blocks, by_regs)`; if the last of those binds
    at every cell the whole ladder collapses onto one rung, `log2 resident`
    becomes a constant column, the design loses rank and `bR` is a combination
    of the intercept. Every coefficient would still print.
    """
    levels = sorted({r.resident_blocks for r in residencies.values()})
    span = (max(levels) / min(levels) if levels and min(levels) > 0 else 0.0)
    lines = design.lines() + [
        f"realised resident-block levels {levels}, span {span:.1f}x "
        f"(needs {MIN_RESIDENCY_LEVELS} levels and {MIN_RESIDENCY_SPAN:.0f}x)"]
    if coef is not None and coef.residual_rms is not None:
        lines.append(f"residual rms {coef.residual_rms:.4f} in ln w on "
                     f"{coef.dof} degrees of freedom")
    enough = (len(levels) >= MIN_RESIDENCY_LEVELS and span >= MIN_RESIDENCY_SPAN)
    return Gate(VALIDITY, "V5 identifiable",
                "the realised design separates depth, BLOCK_K and residency",
                f"rank 4, at least {MIN_RESIDENCY_LEVELS} resident-block "
                f"levels spanning {MIN_RESIDENCY_SPAN:.0f}x",
                design.identified and enough,
                f"rank {design.rank} of 4"
                + ("" if design.condition is None
                   else f", condition {design.condition:.1f}")
                + f"; {len(levels)} resident-block level(s) {levels}, span "
                f"{span:.1f}x",
                "C1 and C2: a design that did not form cannot report a null. "
                "A flat residency ladder makes bR a combination of the "
                "intercept and it still prints",
                lines)


def gate_separation(coef: Coefficients | None, pairs: list[PairContrast],
                    verdict: str) -> Gate:
    """C1: the effect follows DEPTH, not RESIDENCY. The registered prediction.

    PASS is the reading that CONFIRMS the occupancy arm's P1 null: what moved
    with `num_stages` there was the software pipeline, not the footprint, and
    the concurrency family is closed. FAIL is the reading that reopens it, and
    that FAIL is this arm's finding rather than its failure. UNKNOWN is the
    null on both knobs, which cannot separate what it did not see.
    """
    lines = [p.line() for p in pairs]
    if coef is None:
        return Gate(CLAIM, "C1 depth-not-residency",
                    "w moves with pipeline DEPTH and not with RESIDENCY",
                    f"|bD| clears {K_SIGMA:.0f} sigma and exceeds |bR|",
                    None, "no three-parameter fit was formed",
                    "nothing on its own; the design did not form", lines)
    lines = [f"bD depth      {coef.line('depth')}",
             f"bK block_k    {coef.line('block_k')}",
             f"bR residency  {coef.line('residency')}"] + lines
    if verdict == VERDICT_NEITHER:
        return Gate(CLAIM, "C1 depth-not-residency",
                    "w moves with pipeline DEPTH and not with RESIDENCY",
                    f"|bD| clears {K_SIGMA:.0f} sigma and exceeds |bR|",
                    None,
                    "NEITHER: bD " + coef.line("depth") + " and bR "
                    + coef.line("residency")
                    + ", so neither clears its own spread (a spread that is "
                      "NOT STATEABLE does not clear it either, and is not a "
                      "null)",
                    "the separation. An arm that saw no effect of either knob "
                    "cannot say which of them P1 was reading", lines)
    dominates = abs(coef.depth) > abs(coef.residency)
    # THE PASS DOES NOT MEAN THE SAME THING IN THE BOTH WORLD, and until this
    # branch existed the page said it did: C1 PASS was printed as "P1's null is
    # CONFIRMED, the concurrency family is CLOSED" over a run whose C2 said
    # residency had moved w by more than the concurrency model predicts. The
    # gate's VERDICT is unchanged -- the registered claim is an inequality and
    # the inequality held -- and what changes is the sentence a PASS licenses.
    if verdict == VERDICT_BOTH:
        costs = ("nothing about P1's null, which is NOT confirmed here: both "
                 "knobs cleared their own spread and this PASS is the "
                 "registered INEQUALITY alone, |bD| > |bR|. Read C2 in the "
                 "same breath; the concurrency family is not closed by a page "
                 "on which residency moved w")
    else:
        costs = ("the confirmation of occupancy_vs_swizzle P1's residency "
                 "null. A FAIL here is the finding: P1's null was an artefact "
                 "of the lockstep and the concurrent-footprint hypothesis "
                 "returns")
    return Gate(CLAIM, "C1 depth-not-residency",
                "w moves with pipeline DEPTH and not with RESIDENCY",
                f"|bD| clears {K_SIGMA:.0f} sigma and exceeds |bR|",
                verdict in (VERDICT_DEPTH, VERDICT_BOTH) and dominates,
                f"{verdict}: |bD| {abs(coef.depth):.4f} against |bR| "
                f"{abs(coef.residency):.4f}",
                costs, lines)


def gate_residency_null(coef: Coefficients | None, predicted_swing: float | None,
                        swing_source: str) -> Gate:
    """C2: the residency null survives a design that could have broken it.

    Registered prediction: `bR` is inside noise. The size it comes in at is
    printed as a fraction of the concurrency model's OWN predicted swing across
    THIS run's realised residency ladder, so "the footprint hypothesis returns
    at a fraction of its predicted size" is a number on the page.
    """
    lines = []
    if predicted_swing is not None:
        lines.append(f"the concurrency model predicts {predicted_swing:+.4f} "
                     f"in ln w per doubling of residency over this run's own "
                     f"ladder ({swing_source})")
    if coef is None:
        return Gate(CLAIM, "C2 residency null", "residency does not move w",
                    f"|bR| inside {K_SIGMA:.0f} sigma of its own spread",
                    None, "no three-parameter fit was formed",
                    "nothing on its own", lines)
    moves = coef.moves("residency")
    if predicted_swing:
        lines.append(f"measured / predicted = "
                     f"{coef.residency / predicted_swing:+.3f}, and the "
                     "occupancy arm's own bar for calling a predicted swing "
                     f"delivered is {OCC.OCCUPANCY_SWING_FRACTION:.0%} of it")
    return Gate(CLAIM, "C2 residency null", "residency does not move w",
                f"|bR| inside {K_SIGMA:.0f} sigma of its own spread",
                None if moves is None else not moves,
                f"bR {coef.line('residency')}",
                "the reading of the whole concurrency family: a bR that clears "
                "its own noise is the footprint hypothesis returning at the "
                "size printed above", lines)


def gate_block_k(coef: Coefficients | None) -> Gate:
    """C3: BLOCK_K is traffic-neutral, which is what makes the pairs readable.

    A CTA at `(pid_m, pid_n)` walks the whole K extent and reads `BM x K` of A
    and `K x BN` of B whatever BLOCK_K chops K into, so under the reuse-distance
    model the DRAM traffic per CTA does not depend on BLOCK_K at all. What does
    depend on it is the loop trip count, the number of memory instructions and
    the contiguous extent of one B-tile row -- at BLOCK_K=32 in bf16 that extent
    is 64 B, half a cache line, and the other half is used on the next
    iteration, which is a latency question and not a traffic one.

    IF THIS FAILS THE PAIRS ARE NOT READABLE AS DEPTH CONTRASTS. Holding shared
    memory fixed forces `stages` and `BLOCK_K` to move in opposite directions
    inside a pair, so a real BLOCK_K effect sits in the pair's contrast and
    cannot be told from depth there. The pooled fit still separates them, which
    is why the grid is eight cells; the two-cell table above it does not.
    """
    if coef is None:
        return Gate(CLAIM, "C3 block-k-neutral",
                    "BLOCK_K does not move w at fixed depth and residency",
                    f"|bK| inside {K_SIGMA:.0f} sigma of its own spread",
                    None, "no three-parameter fit was formed",
                    "nothing on its own", [])
    moves = coef.moves("block_k")
    return Gate(CLAIM, "C3 block-k-neutral",
                "BLOCK_K does not move w at fixed depth and residency",
                f"|bK| inside {K_SIGMA:.0f} sigma of its own spread",
                None if moves is None else not moves,
                f"bK {coef.line('block_k')}",
                "the two-cell iso-smem table, where depth and BLOCK_K are "
                "perfectly anti-collinear. The eight-cell fit separates them "
                "and is what C1 is read off; a FAIL here says the PAIR lines "
                "may not be quoted as depth contrasts",
                [])


# --------------------------------------------------------------------------
# The plan and the registered predictions.
# --------------------------------------------------------------------------

def concurrency_swing(cfg, residencies: dict[str, CellResidency], b: int,
                      limits) -> tuple[float | None, str]:
    """The concurrency model's own predicted `d ln w` per doubling of residency.

    Borrowed whole from `occupancy_vs_swizzle`: `alpha = 1 - min(1, L2 /
    (SMs x resident blocks x one CTA-lifetime of streaming))`. It is a
    one-parameter LRU caricature and NO GATE IS SCORED ON ITS LEVEL. It is here
    so that C2's size can be quoted against something the rival hypothesis
    itself says, rather than against an adjective.
    """
    levels = sorted({r.resident_blocks for r in residencies.values()
                     if r.resident_blocks > 0})
    if len(levels) < 2:
        return None, "fewer than two resident-block levels on this card"
    lo, hi = levels[0], levels[-1]
    alphas = []
    for res in (lo, hi):
        footprint = OCC.concurrent_footprint_bytes(
            cfg, SUBJECT_BLOCK_M, PINNED_BLOCK_N, b, res, limits.sm_count)
        alphas.append(OCC.alpha_concurrency(footprint, limits.l2_bytes))
    if min(alphas) <= 0:
        return None, "the model's predicted alpha is zero at one end"
    swing = ((math.log(alphas[1]) - math.log(alphas[0]))
             / math.log2(hi / lo))
    return swing, (f"{lo} to {hi} blocks/SM at {limits.sm_count} SMs over "
                   f"{limits.l2_bytes / 1e6:.0f} MB of L2")


def mde_line(design: Design, spread: float, spread_source: str) -> list[str]:
    """A CONSERVATIVE BOUND on the smallest coefficient this design resolves.

    `sd(b_j) = sigma x sqrt(diag((X'X)^-1)_jj)`, where `sigma` must be the sd of
    ONE OBSERVATION of the regression -- and one observation here is one CELL's
    `ln w`, not one timing. WHAT IS ACTUALLY PASSED IN is `resolve_spread`: the
    median per-tread across-repeat relative spread of `ms_p50`, a per-TIMING
    figure, taken BEFORE the median over the repeats and BEFORE the slope fit
    over the treads. Two collapses stand between it and a cell's `ln w`, and
    both shrink it, so this line is an UPPER BOUND on `sd(b_j)` and the effects
    it says are unresolvable include effects this design does resolve: a
    planted `bD` of 0.015, under the 0.0235 the default plan prints, comes back
    at 0.0165 +/- 0.0032 and C1 reads DEPTH.

    THE SENTENCE THAT USED TO CLOSE THIS BLOCK said a coefficient smaller than
    its line "is not resolvable by this design at this spread, whatever it
    measures", and that is false by roughly the collapse factor. The bound is
    left conservative rather than re-derived because the two directions are not
    symmetric: a plan page that overstates its own power talks an owner into
    booking a card, and the number that settles it is the bootstrap sd this arm
    prints beside every coefficient AFTER the run, which carries the whole
    chain and needs no propagation argument.

    `resolve_spread` is the same figure `planted_samples` uses as a per-timing
    sd, where it IS one, and that double role is why the mismatch was invisible.
    Printed in the PLAN and not in the post-mortem, because the only cheap
    moment to find that a gate cannot resolve the effect it is registered
    against is before a pod is rented.
    """
    if not design.se_multiplier:
        return ["MINIMUM DETECTABLE EFFECT: not stateable. The design is "
                "singular, so no number of repeats resolves the coefficients."]
    names = ("intercept", "bD depth", "bK block_k", "bR residency")
    out = [f"MINIMUM DETECTABLE EFFECT at {K_SIGMA:.0f} sigma, per doubling, "
           f"in ln w, at a per-TIMING spread of {spread:.4%}:",
           f"             spread source: {spread_source}"]
    for name, mult in zip(names[1:], design.se_multiplier[1:], strict=True):
        out.append(f"             {name:12s} {K_SIGMA * spread * mult:.4f}")
    out.append("             AN UPPER BOUND, not the line it looks like. The "
               "sigma above is a per-TIMING spread and one observation of this "
               "regression is a CELL's ln w, which is a median over the "
               "repeats and then a slope over the treads; both collapses "
               "shrink it. A coefficient under its line here may still be "
               "resolved, and the number that says so is the bootstrap sd "
               "printed beside each coefficient after the run.")
    return out


#: Said wherever the MDE is printed off GPU. The plan's design matrix is built
#: from the residency BOUND -- no n_regs, so `by_regs` is absent -- which is the
#: top row of the register-sensitivity table and the worst-conditioned one at
#: every warp count this arm runs. The compile census reprints these lines at
#: the design the card actually realises.
MDE_BOUND_NOTE = (
    "             COMPUTED AT THE RESIDENCY BOUND. No n_regs has been read, so "
    "this is the top row of the register-sensitivity table above. THAT ROW IS "
    "NOT THE WORST-CONDITIONED ONE FOR EVERY COEFFICIENT, which this note used "
    "to claim: read the table's own bR column, where the deepest register "
    "rows are worse conditioned than the bound, so a kernel landing there has "
    "a bR standard error WIDER than the line above. The compile census "
    "reprints these lines at the measured n_regs before a timed cell is paid "
    "for, and those are the realised numbers.")


def predictions_text(cfg, cells, b, residencies, design, ridge, ridge_source,
                     swing, swing_source) -> list[str]:
    """Registered before the run and printed before any measurement."""
    out = [
        "## PREDICTIONS, registered before the run and printed before any "
        "measurement", "",
        "THE QUESTION. occupancy_vs_swizzle's P6 is UNKNOWN because no two "
        "num_stages settings in that corpus ever shared a resident-block "
        "count, so its P1 residency null is confounded with pipeline depth. "
        "This arm moves BLOCK_SIZE_K, which is 64 in every fit in the corpus, "
        "so that shared memory and prefetch depth come apart.", "",
        "THE FOUR WORLDS THE GATES DISCRIMINATE. Four, because `verdict_of` "
        "has four outcomes and this block registered three: the BOTH world was "
        "reachable, planted in the test file, and had no registered reading, "
        "so the page asserted C1's 'the concurrency family is CLOSED' and C2's "
        "'residency moved w by more than the model predicts' in one report.",
        "  FOLLOWS DEPTH      C1 PASS. P1's null is confirmed: what moved with "
        "num_stages was latency hiding, the concurrency family is closed, and "
        "the reuse-distance reading of alpha stands unopposed by a footprint "
        "rival.",
        "  FOLLOWS RESIDENCY  C1 FAIL. P1's null was an artefact of the "
        "lockstep, residency moves the weight-stream cost, and C2 prints the "
        "size it returns at as a fraction of the concurrency model's own "
        "predicted swing.",
        "  BOTH               C1 PASS and C2 FAIL, and the PASS is the "
        "REGISTERED INEQUALITY and nothing more: both knobs cleared their own "
        "spread and depth is the larger. P1's null is NOT confirmed -- "
        "residency moved w -- and the concurrency family is NOT closed. The "
        "quotable sentence is the ORDERING of the two coefficients, with C2's "
        "printed fraction of the predicted swing beside it.",
        "  NEITHER            C1 UNKNOWN, and the page is not a null on either "
        "knob: an arm that saw no effect of either cannot say which of them P1 "
        "was reading.", "",
        f"THE REGISTERED THRESHOLD is {K_SIGMA:.0f} sigma of each "
        "coefficient's OWN bootstrap spread, resampling repeats with "
        "replacement and refitting the whole chain on every draw.", "",
        "THE GRID, and the arithmetic every cell is chosen by. One CTA holds "
        f"num_stages x (BLOCK_M x BLOCK_K + BLOCK_K x BLOCK_N) x {b} bytes, "
        f"which at BLOCK_M = BLOCK_N = {SUBJECT_BLOCK_M} is "
        f"num_stages x BLOCK_K x {(SUBJECT_BLOCK_M + PINNED_BLOCK_N) * b} B.",
        "  cell     stages  BLOCK_K   smem/CTA   residency"]
    for cell in cells:
        res = residencies.get(cell.key)
        out.append(f"  {cell.key:8s} {cell.num_stages:6d}  {cell.block_k:7d}  "
                   f"{cell.smem_bytes(b) / 1024:7.0f} KiB   "
                   + (res.line() if res else "NOT COMPUTED"))
    out += ["", "THE TWO REGISTERED ISO-SHARED-MEMORY PAIRS, which are the "
            "contrast this design is named for:"]
    by_key = {c.key: c for c in cells}
    for lo_key, hi_key in ISO_SMEM_PAIRS:
        lo, hi = by_key.get(lo_key), by_key.get(hi_key)
        if lo is None or hi is None:
            out.append(f"  {lo_key} / {hi_key}: NOT IN THIS GRID")
            continue
        out.append(f"  {lo_key} / {hi_key}: {lo.smem_bytes(b) / 1024:.0f} KiB "
                   f"both, depths {lo.num_stages} and {hi.num_stages}, "
                   f"BLOCK_K {lo.block_k} and {hi.block_k}")
    out += ["", "AND THE ISO-DEPTH RESIDENCY LADDER, "
            + ", ".join(ISO_DEPTH_LADDER)
            + ": three BLOCK_K at one num_stages, which is the one contrast in "
            "this apparatus where residency moves and the software-pipeline "
            "depth is byte-identical.", "",
            "THE THIRD MECHANISM, named so it cannot contaminate the reading. "
            "Holding shared memory fixed forces num_stages and BLOCK_K to move "
            "in OPPOSITE directions, so inside a pair depth and BLOCK_K are "
            "perfectly anti-collinear and two cells cannot separate them. The "
            "grid is eight cells and the reading is a three-parameter fit; C3 "
            "registers bK = 0 and can refute it.", "",
            "THE DESIGN AS THIS CARD REALISES IT:"]
    out += design.lines()
    out += ["",
            "SCORED ON w, NOT ON B/(A+B). The EXA ratio's denominator carries "
            "a fitted intercept far noisier than the slope it is added to, and "
            "that intercept goes NEGATIVE in two of the 2026-09-10 session's "
            "arms, so a residency effect of a few per cent in the slope "
            "arrives in the ratio as a sign flip. w is the same slope over a "
            "byte count from moe.spec and a rate the caller names: no fitted "
            "level and no intercept. The per-cent noise figures the session-3 "
            "analysis quotes for w are ITS figures and are in no committed "
            "file here, so they are not printed as this page's; the spread "
            "this design is actually powered against is the line below, which "
            "says ASSUMED or says which of this run's own cells it measured.",
            "",
            f"ridge {ridge:.3f} Op/B, {ridge_source}. Used ONLY by V4, to ask "
            "whether each cell's MEASURED marginal arithmetic intensity is "
            "below it. No claim here is scored against a roof."]
    if swing is not None:
        out.append(f"the concurrency model predicts {swing:+.4f} in ln w per "
                   f"doubling of residency over {swing_source}; NO GATE IS "
                   "SCORED ON ITS LEVEL and C2 quotes bR against it only to "
                   "give the size a unit.")
    out += ["", "NOT A READOUT. Everything above is arithmetic over this "
            "repo's calibration and vLLM's resource model. NOT A PRODUCTION "
            "CLAIM: BLOCK_SIZE_K 32 and 128 are settings vLLM ships for other "
            "shapes, and this grid is chosen to separate two mechanisms, not "
            "to recommend a tile. NOT RUN: no cell below has been timed."]
    return out


def estimated_seconds(cells, treads: int, reps: int, warmup_ms: float,
                      trials: int, cell_budget_ms: float) -> float:
    """What the INSTRUMENT charges, which is not what a per-call loop charged.

    `time_kernel` warms for a fixed DURATION and then runs `trials` trials, each
    sized by `iters_for` to hold `cell_budget_ms` of kernel time, so one timing
    costs `warmup_ms + trials x cell_budget_ms` and is nearly independent of the
    kernel's own duration. The treads of this arm run from a few tenths of a
    millisecond to about seven at the top of the published subject range for
    `w`, which was written here as "between 0.6 and 5 ms" and understated the
    deep end. The conclusion is unchanged and has room to spare: `iters_for`
    clamps at `lo=10`, which binds only above 20 ms per call, so "nearly" is
    "exactly" here and the figure needs no caveat.
    """
    return (len(cells) * treads * reps
            * (warmup_ms + trials * cell_budget_ms) / 1e3)


# --------------------------------------------------------------------------
# Identity and provenance.
# --------------------------------------------------------------------------

def default_run_id(args, card: str) -> str:
    """Derived from EVERY swept parameter AND the card, so two settings cannot
    collide.

    `--cells` is the sweep, so the LIST is in the key and not a value: two runs
    over different grids are different experiments even when the grids overlap,
    because V5 reads a different rank off them.

    Every TIMING knob is in the key -- warmup, trials, the cell budget, the L2
    flush -- because each of them changes the measured milliseconds, and a
    flushed and an unflushed sweep must never share a directory.

    `--ridge`, `--bandwidth-gbps` and `--draws` stay OUT: they re-analyse one
    set of timings rather than change one, and two analyses of one sweep belong
    in one directory.
    """
    return PV.run_id(
        card=card,
        model=args.model,
        dtype=args.dtype,
        cells=tuple(c.key for c in parse_cells(args.cells)),
        bm=SUBJECT_BLOCK_M,
        bn=PINNED_BLOCK_N,
        g=PINNED_GROUP_M,
        warps=args.num_warps,
        treads=args.treads,
        reps=args.reps,
        warmup=args.warmup,
        trials=args.trials,
        flush=not args.no_l2_flush,
        budget=args.cell_budget_ms,
        seed=args.seed,
        plantnoise=("auto" if args.plant_noise is None else args.plant_noise),
    )


def arm_cache(root: Path, cell: Cell) -> Path:
    """A fresh Triton cache directory for THIS cell, before it compiles.

    Keyed on BOTH swept knobs. `block_m_crossing_sweep.arm_triton_cache` keys
    on BLOCK_M alone, which is right for a sweep whose only variable is BLOCK_M
    and wrong here: two cells at one BLOCK_M would share a directory, the second
    would find it warm, compile nothing, and be scored by V1 as a grid that ran
    one kernel.
    """
    directory = root / f"s{cell.num_stages}-k{cell.block_k}"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(directory)
    return directory


def git_visibility(path: Path) -> str:
    """Say out loud whether git would keep this file, asking git rather than
    re-implementing `.gitignore`."""
    try:
        done = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q",
                               str(path)], capture_output=True, timeout=10)
    except Exception as exc:                            # noqa: BLE001
        return f"git visibility UNVERIFIED ({type(exc).__name__}: {exc})"
    if done.returncode == 0:
        return "git IGNORES this path: nothing here will be committed by `git add`"
    if done.returncode == 1:
        return "git would KEEP this path"
    return (f"git visibility UNVERIFIED (check-ignore exited "
            f"{done.returncode}; 128 means outside the work tree, which is the "
            "pod default)")


# --------------------------------------------------------------------------
# The metered part.
# --------------------------------------------------------------------------

def measure(args, cfg, cells, rows, csv_path: Path, cache_root: Path, done,
            samples: list[Sample], probe, *, prov=None,
            reference_clock_mhz: float | None = None
            ) -> tuple[dict[str, int], dict[str, int], dict[str, dict]]:
    """Time every cell, `--reps` round-robin passes over its treads.

    ROUND ROBIN INSIDE THE CELL. Measuring tread 1 fifty times and then tread 8
    fifty times puts every tread at a different point in the pod's thermal
    history, and the resulting monotone drift IS a slope -- the quantity being
    fitted.

    THE INSTRUMENT IS `moe.bench.timing.time_kernel` AND NOTHING ELSE, which is
    the instrument the roof and every published w were measured with.

    A REFUSAL FROM THE INSTRUMENT IS NOT A FAILED CELL. `TimingRefused` says the
    measurement could not be made at all, and catching it per cell would write a
    grid of zeroed rows and exit DONE having learned it once.
    """
    import torch

    from moe.baselines._framework_config import vllm_call_kwargs
    from moe.bench import timing
    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    override_config, _ = SWEEP.find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    compiles: dict[str, int] = {}
    executed: dict[str, int] = {}
    built: dict[int, tuple] = {}

    for cell in cells:
        arm_cache(cache_root, cell)
        seen: set[Path] = set()
        SWEEP.count_new(cache_root, seen)
        compiles.setdefault(cell.key, 0)
        executed.setdefault(cell.key, 0)
        conf = cell.pinned(args.num_warps)
        for rep in range(1, args.reps + 1):
            for r in rows:
                tiles = r // SUBJECT_BLOCK_M
                if (cell.key, tiles, rep) in done:
                    continue
                tokens = SWEEP.tokens_for_rows(cfg, r)
                if tokens not in built:
                    spec = BenchSpec(cfg, num_tokens=tokens, dtype=args.dtype,
                                     routing=RoutingSpec("uniform", 0.0),
                                     seed=args.seed)
                    x, weights = make_inputs(spec, device="cuda")
                    ids = SWEEP.balanced_ids(cfg, tokens, "cuda")
                    tw = torch.full(ids.shape, 1.0 / cfg.top_k,
                                    dtype=torch.float32, device="cuda")
                    kw = vllm_call_kwargs(spec)
                    kw["activation"] = MoEActivation(kw["activation"])
                    built = {tokens: (x, weights, ids, tw, kw)}  # one cell live
                x, weights, ids, tw, kw = built[tokens]
                executed[cell.key] += 1

                def call(_f=fused_experts, _x=x, _wt=weights, _w=tw, _i=ids,
                         _k=kw):
                    return _f(hidden_states=_x, w1=_wt.w1, w2=_wt.w2,
                              topk_weights=_w, topk_ids=_i, **_k)

                try:
                    with override_config(conf):
                        call()
                        torch.cuda.synchronize()
                        compiles[cell.key] += SWEEP.count_new(cache_root, seen)
                        probe.record(cell.key)
                        t = timing.time_kernel(
                            call, warmup_ms=args.warmup,
                            target_ms=args.cell_budget_ms, trials=args.trials,
                            l2_flush=not args.no_l2_flush,
                            reference_clock_mhz=reference_clock_mhz)
                    got = probe.by_setting.get(cell.key, {})
                    sample = Sample(
                        cell.key, cell.num_stages, cell.block_k, tiles, r,
                        tokens, rep, t.ms_p50, t.ms_min, t.ms_std, t.iters,
                        instrument=t.instrument, warmup_ms=t.warmup_ms,
                        trials=t.trials, l2_flush=t.l2_flush,
                        sm_clock_load_mhz=t.sm_clock_load_mhz,
                        sm_clock_start_mhz=t.sm_clock_start_mhz,
                        sm_clock_end_mhz=t.sm_clock_end_mhz,
                        clock_samples_mhz=" ".join(
                            f"{v:.0f}" for v in (t.clock_samples_mhz or ())),
                        power_w=t.power_w,
                        clock_level_ok=t.clock_level_ok,
                        clock_drift_ok=t.clock_drift_ok,
                        host_bound=t.host_bound,
                        clock_level_side=getattr(t, "clock_level_side", "") or "",
                        compiled_smem=int(got.get("shared", 0) or 0),
                        compiled_regs=int(got.get("n_regs", 0) or 0),
                        compiled_spills=int(got.get("n_spills", 0) or 0))
                    # THE SIDE IS RECORDED HERE AND EXCLUDES NOTHING.
                    if clock_excluded(t.clock_level_ok,
                                      sample.clock_level_side,
                                      t.clock_drift_ok) or t.host_bound:
                        print(f"  ^ {t.clock_note or ''} "
                              f"{t.host_note or ''}".rstrip())
                    elif sample.clock_level_side:
                        print(f"  ^ kept (LEVEL {sample.clock_level_side} is "
                              f"recorded, not excluded): "
                              f"{t.clock_note or ''}".rstrip())
                except timing.TimingRefused:
                    raise
                except Exception as exc:                # noqa: BLE001
                    sample = Sample(cell.key, cell.num_stages, cell.block_k,
                                    tiles, r, tokens, rep, 0.0, 0.0, 0.0, 0,
                                    "failed", f"{type(exc).__name__}: {exc}")
                    print(f"  {cell.key} n={tiles} rep={rep} FAILED "
                          f"{sample.detail}")
                samples.append(sample)
                append_sample(csv_path, sample, prov)
                print(f"  {cell.key:8s} n={tiles:2d} rep={rep:2d} r={r:5d} "
                      f"T={tokens:6d}  {sample.ms_p50:9.4f} ms "
                      f"({sample.iters} iters)")
    return compiles, executed, dict(probe.by_setting)


# --------------------------------------------------------------------------
# Analysis: one function, called by the pod run, the self test and the tests.
# --------------------------------------------------------------------------

@dataclass
class Analysis:
    fits: dict[str, CellFit]
    residencies: dict[str, CellResidency]
    design: Design
    coef: Coefficients | None
    pairs: list[PairContrast]
    verdict: str
    coef_sd: dict[str, float | None]
    clock: dict


def analyse(cfg, cells, samples, *, dtype, bandwidth_gbps, bandwidth_source,
            limits, observed, b, draws: int, seed: int,
            warps: int = PINNED_WARPS) -> Analysis:
    """Fit every cell, build the realised design, fit the three coefficients.

    `observed` is the compiled-kernel read-back keyed by cell. The residency
    ladder is built from it, so a run whose probe read nothing gets a ladder of
    upper bounds and V3 says so rather than scoring them.
    """
    by_key = {c.key: c for c in cells}
    regs = {k: int(v.get("n_regs", 0) or 0) for k, v in observed.items()}
    spills = {k: int(v.get("n_spills", 0) or 0) for k, v in observed.items()}
    residencies = {
        c.key: cell_residency(c, b, limits, warps=warps,
                              registers=regs.get(c.key) or None,
                              spills=spills.get(c.key))
        for c in cells}
    fits = {}
    for key, cell in by_key.items():
        fit = fit_cell(cfg, key, cell, samples, dtype=dtype,
                       bandwidth_gbps=bandwidth_gbps,
                       bandwidth_source=bandwidth_source)
        if fit is not None:
            fits[key] = fit
    coef_sd, pair_sd = bootstrap(
        cfg, by_key, samples, residencies, dtype=dtype,
        bandwidth_gbps=bandwidth_gbps, bandwidth_source=bandwidth_source,
        draws=draws, seed=seed, b=b)
    coef = fit_coefficients(fits, residencies, coef_sd)
    design = build_design([c.key for c in cells if c.key in residencies],
                          residencies, by_key)
    pairs = []
    for pair in ISO_SMEM_PAIRS:
        got = pair_contrast(pair, fits, residencies, b,
                            pair_sd.get("/".join(pair)))
        if got is not None:
            pairs.append(got)
    return Analysis(fits, residencies, design, coef, pairs, verdict_of(coef),
                    coef_sd, clock_state(samples))


def build_gates(analysis: Analysis, cfg, cells, b, *, ridge, ridge_source,
                compiles, executed, observed, limits, tolerance: float
                ) -> list[Gate]:
    """Every gate, in the order a reader has to take them."""
    swing, swing_source = concurrency_swing(cfg, analysis.residencies, b, limits)
    return [
        gate_non_vacuity({
            "cells in grid": len(cells),
            "cells fitted": len(analysis.fits),
            "treads fitted": sum(f.treads for f in analysis.fits.values()),
            "timings kept": analysis.clock["timed"] - analysis.clock["excluded"],
            "registered pairs formed": len(analysis.pairs),
        }),
        gate_geometry(compiles, executed, observed),
        gate_iso_smem(analysis.pairs, {c.key: c for c in cells}, b, observed,
                      tolerance),
        gate_residency(analysis.residencies, analysis.pairs),
        gate_treads(analysis.fits, ridge, ridge_source, cells),
        gate_identifiable(analysis.design, analysis.residencies, analysis.coef),
        gate_separation(analysis.coef, analysis.pairs, analysis.verdict),
        gate_residency_null(analysis.coef, swing, swing_source),
        gate_block_k(analysis.coef),
    ]


def report_lines(analysis: Analysis) -> list[str]:
    out = ["## The reading", "",
           f"VERDICT  {analysis.verdict}", ""]
    if analysis.coef is not None:
        out += [f"  bD depth      {analysis.coef.line('depth')}",
                f"  bK block_k    {analysis.coef.line('block_k')}",
                f"  bR residency  {analysis.coef.line('residency')}"]
        if analysis.coef.residual_rms is not None:
            out.append(f"  residual rms  {analysis.coef.residual_rms:.4f} in "
                       f"ln w on {analysis.coef.dof} degrees of freedom")
    else:
        out.append("  no three-parameter fit was formed")
    out += ["", "THE REGISTERED ISO-SHARED-MEMORY PAIRS, model-free:"]
    out += [p.line() for p in analysis.pairs] or ["  none formed"]
    # THE RATE IS STATED ONCE FOR THE BLOCK, AND ONLY WHILE IT IS ONE RATE.
    # `WeightStreamSlope.render()` keeps the number and the rate on one line
    # because a w quoted without the rate it was divided by is not a
    # measurement -- w scales exactly 1:1 in it. One run has one bandwidth, so
    # repeating a 250-character provenance sentence at all eight cells buries
    # the eight numbers the page is about. The header below carries it once,
    # and the moment the fits do NOT agree on a rate this falls back to
    # `render()` per cell rather than printing a heading that is true of some
    # of the rows.
    rates = {(f.w.bandwidth_gbps, f.w.bandwidth_source)
             for f in analysis.fits.values()}
    one_rate = rates.pop() if len(rates) == 1 else None
    out += ["", "PER CELL, w = ms per extra M-tile / ms per complete stream of "
            "the routed expert weight set:"]
    if one_rate:
        first = next(iter(analysis.fits.values()))
        out.append(f"  all at {one_rate[0]:.1f} GB/s ({one_rate[1]}); one "
                   f"stream is {first.w.stream_ms:.4f} ms of "
                   f"{first.w.weight_bytes / 1e9:.4f} GB, {first.w.model} "
                   f"{first.w.dtype}")
    for key in sorted(analysis.fits):
        fit = analysis.fits[key]
        res = analysis.residencies.get(key)
        out.append(f"  {key:8s} w {fit.streams:8.4f}"
                   if one_rate else f"  {key:8s} {fit.w.render()}")
        out.append(f"           {res.line() if res else 'residency NOT COMPUTED'}")
    out += ["", "CLOCK:"] + clock_state_lines(analysis.clock)
    out.append(f"  rule: {analysis.clock['rule']}")
    return out


def report_payload(analysis: Analysis, gates: list[Gate], prov, *, run_id: str,
                   model: str, dtype: str, cells) -> dict:
    """`report.json`, assembled as a FUNCTION so it can be tested off GPU.

    The sibling arm records the failure this shape exists to prevent: a pod run
    "spent both ladders and then died between the last timing and the first
    `write_text`, leaving no report at all". Everything below is pure assembly
    over objects a planted world also produces, so a serialisation break is a
    unit test rather than a lost booking.

    `prov.stamp` nests the provenance block and lifts its top-level keys; it
    raises `ProvenanceCollision` on a conflicting key, which is why none of the
    keys chosen here is one of them.
    """
    return prov.stamp({
        "experiment": "blockk_diagonal",
        "run_id": run_id,
        "model": model,
        "dtype": dtype,
        "cells": [c.key for c in cells],
        "verdict": analysis.verdict,
        "coefficients": (None if analysis.coef is None else {
            "intercept": analysis.coef.intercept,
            "depth": analysis.coef.depth,
            "block_k": analysis.coef.block_k,
            "residency": analysis.coef.residency,
            "sd": analysis.coef.sd,
            "residual_rms": analysis.coef.residual_rms,
            "dof": analysis.coef.dof}),
        # EVERY w CARRIES THE RATE IT WAS DIVIDED BY, here as well as on the
        # page: w scales exactly 1:1 in it, so a persisted w without one is a
        # number a later reader cannot use.
        "w": {k: {"streams": f.streams, "slope_ms": f.slope_ms,
                  "treads": f.treads, "marginal_ai": f.marginal_ai,
                  "weight_bytes": f.w.weight_bytes,
                  "stream_ms": f.w.stream_ms,
                  "bandwidth_gbps": f.w.bandwidth_gbps,
                  "bandwidth_source": f.w.bandwidth_source}
              for k, f in sorted(analysis.fits.items())},
        "pairs": [asdict(p) for p in analysis.pairs],
        "residency": {k: asdict(r) for k, r in sorted(
            analysis.residencies.items())},
        "design": {"rank": analysis.design.rank,
                   "singular_values": list(analysis.design.singular_values),
                   "condition": analysis.design.condition,
                   "se_multiplier": (list(analysis.design.se_multiplier)
                                     if analysis.design.se_multiplier else None)},
        "clock": analysis.clock,
        "gates": [{"kind": g.kind, "name": g.name, "verdict": g.scored()[2],
                   "observed": g.observed} for g in gates],
    })


# --------------------------------------------------------------------------
# The planted worlds.
# --------------------------------------------------------------------------

#: The fixed cost planted under every world's ladder, in ms. It exists so the
#: planted ladders have a non-zero intercept and the slope is not recoverable
#: by dividing any single point: a world whose ms is exactly proportional to n
#: would let a broken fit pass.
PLANTED_INTERCEPT_MS = 0.12

#: The per-thread register count every planted world's kernels report. Chosen so
#: that the register limit does NOT bind on the planted card. AT THIS ARM'S OWN
#: WARP COUNT, which is 4 and not 8: a CTA is 128 threads, so 65536 / (128 x 32)
#: is 16 blocks by registers against a thread-slot limit of 16, and the
#: self-test prints exactly `threads 16, regs 16`. The arithmetic that stood
#: here was done at 256 threads and printed 8 against a thread-slot limit it
#: also called 8; the conclusion survived and neither number did. A self test
#: whose planted register file flattened the ladder would be testing V5's
#: refusal and nothing else. `test_blockk_diagonal.py` plants the other case
#: explicitly.
PLANTED_REGISTERS = 32


#: `w` at the reference cell of every planted world, before the world's own
#: coefficients move it. The middle of the published subject range.
PLANTED_W0 = 1.10


@dataclass(frozen=True)
class World:
    """A planted world, the verdicts it registers, and the exit code it implies.

    A self test that asserts nothing is a smoke test. Each world registers a
    verdict PER GATE and the exit code `classify` must return, which is the only
    thing a session driver can see. A mismatch exits ERROR (4), not INVALID and
    not CLAIM_FAIL, because it means the apparatus is broken rather than a claim
    refuted.
    """

    name: str
    why: str
    depth: float
    block_k: float
    residency: float
    expect: dict[str, str]
    exit_code: int


SELF_TEST_WORLDS = {
    "depth": World(
        "depth", "w depends on the pipeline depth alone: P1's null confirmed",
        0.12, 0.0, 0.0,
        {"C1": exit_codes.PASS, "C2": exit_codes.PASS, "C3": exit_codes.PASS},
        exit_codes.DONE),
    "residency": World(
        "residency", "w depends on the resident-block count alone: P1's null "
        "was an artefact of the lockstep",
        0.0, 0.0, -0.12,
        {"C1": exit_codes.FAIL, "C2": exit_codes.FAIL, "C3": exit_codes.PASS},
        exit_codes.CLAIM_FAIL),
    "blockk": World(
        "blockk", "w depends on BLOCK_K alone, which is the world C3 exists to "
        "catch and which would read as depth in the two-cell table",
        0.0, 0.12, 0.0,
        {"C1": exit_codes.UNKNOWN, "C2": exit_codes.PASS,
         "C3": exit_codes.FAIL},
        exit_codes.CLAIM_FAIL),
    "null": World(
        "null", "nothing moves w: the arm looked and could not separate them",
        0.0, 0.0, 0.0,
        {"C1": exit_codes.UNKNOWN, "C2": exit_codes.PASS,
         "C3": exit_codes.PASS},
        exit_codes.CLAIM_FAIL),
}


def planted_w(world: World, cell: Cell, resident: int) -> float:
    """`ln w = ln w0 + aD log2 stages + aK log2(BK/64) + aR log2(resident/4)`."""
    return PLANTED_W0 * math.exp(
        world.depth * math.log2(cell.num_stages)
        + world.block_k * math.log2(cell.block_k / 64.0)
        + world.residency * math.log2(max(1, resident) / 4.0))


#: The five clock states every planted world carries, in the order the repeats
#: cycle through them. FIVE and not four: the HIGH-and-DRIFT row exists because
#: every earlier world in this repo had its drifting row sitting LEVEL, so a
#: counter filtering on LEVEL alone counted that tread twice and stayed green.
PLANTED_CLOCKS = (
    ("level", True, True, ""),
    ("high", False, True, "LEVEL_HIGH"),
    ("low", False, True, "LEVEL_LOW"),
    ("drift", True, False, ""),
    ("high+drift", False, False, "LEVEL_HIGH"),
)


def planted_samples(cfg, cells, rows, world: World, residencies, *, reps: int,
                    noise: float, seed: int, stream_ms: float,
                    b: int) -> list[Sample]:
    """Rows of the shape `measure` writes, through the same CSV contract.

    Every row carries `SWEEP.SYNTHETIC_INSTRUMENT`, so "not measured" is a value
    in the column and not an absence, and the five clock states above so the
    exclusion rule is exercised rather than assumed.

    `b` is the dtype width and is a PARAMETER. Written as the literal 2 in the
    `compiled_smem` column it was a derived quantity typed in, and at any dtype
    but bf16 it wrote a CSV column disagreeing with the `observed` dict the
    same caller builds from the run's own width.
    """
    from moe.bench import timing

    rng = random.Random(seed)
    out: list[Sample] = []
    sides = {"LEVEL_HIGH": timing.LEVEL_HIGH, "LEVEL_LOW": timing.LEVEL_LOW,
             "": ""}
    for cell in cells:
        resident = residencies[cell.key].resident_blocks
        slope = planted_w(world, cell, resident) * stream_ms
        for rep in range(1, reps + 1):
            _, level, drift, side = PLANTED_CLOCKS[(rep - 1)
                                                   % len(PLANTED_CLOCKS)]
            for r in rows:
                tiles = r // SUBJECT_BLOCK_M
                ms = (PLANTED_INTERCEPT_MS + slope * tiles) * (
                    1.0 + rng.gauss(0.0, noise))
                # A DRIFTED TREAD IS PLANTED WRONG ON PURPOSE. It carries a
                # time 30% off the world's own ladder, so a fit that failed to
                # exclude it would miss the planted coefficients and the self
                # test would fail loudly rather than quietly agreeing.
                if drift is False:
                    ms *= 1.30
                out.append(Sample(
                    cell.key, cell.num_stages, cell.block_k, tiles, r,
                    SWEEP.tokens_for_rows(cfg, r), rep, ms, ms, 0.0, 100,
                    instrument=SWEEP.SYNTHETIC_INSTRUMENT, warmup_ms=0.0,
                    trials=0, l2_flush=True,
                    sm_clock_load_mhz=1485.0, sm_clock_start_mhz=1485.0,
                    sm_clock_end_mhz=1485.0, clock_samples_mhz="1485 1485",
                    power_w=690.0, clock_level_ok=level, clock_drift_ok=drift,
                    host_bound=False, clock_level_side=sides[side],
                    compiled_smem=cell.smem_bytes(b),
                    compiled_regs=PLANTED_REGISTERS, compiled_spills=0))
    return out


def self_test(args, cfg, cells, rows, limits, b, *, bandwidth_gbps,
              bandwidth_source, ridge, ridge_source, noise: float
              ) -> tuple[list[str], list[Gate], int]:
    """Run the pod's own `analyse` and `build_gates` over a planted world.

    It runs THE FUNCTIONS THE POD RUN CALLS. A self test that built its own
    `(tread, ms)` pairs and called one helper would leave every gate below it
    untouched, which is how a regression in the path that produced a 43.6x
    reference once passed one.
    """
    world = SELF_TEST_WORLDS[args.self_test]
    stream_ms = WEIGHTS.weight_stream_ms(cfg, args.dtype, bandwidth_gbps)
    observed = {c.key: {"shared": c.smem_bytes(b), "n_regs": PLANTED_REGISTERS,
                        "n_spills": 0} for c in cells}
    residencies = {c.key: cell_residency(c, b, limits, warps=args.num_warps,
                                         registers=PLANTED_REGISTERS, spills=0)
                   for c in cells}
    samples = planted_samples(cfg, cells, rows, world, residencies,
                              reps=args.reps, noise=noise, seed=args.seed,
                              stream_ms=stream_ms, b=b)
    analysis = analyse(cfg, cells, samples, dtype=args.dtype,
                       bandwidth_gbps=bandwidth_gbps,
                       bandwidth_source=bandwidth_source, limits=limits,
                       observed=observed, b=b, draws=args.draws,
                       seed=args.seed, warps=args.num_warps)
    gates = build_gates(analysis, cfg, cells, b, ridge=ridge,
                        ridge_source=ridge_source,
                        compiles={c.key: 1 for c in cells},
                        executed={c.key: len(rows) * args.reps for c in cells},
                        observed=observed, limits=limits,
                        tolerance=args.smem_tolerance)
    got = {g.token: g.scored()[2] for g in gates}
    mismatches = [f"{token}: registered {want}, got {got.get(token, 'ABSENT')}"
                  for token, want in world.expect.items()
                  if got.get(token) != want]
    code = exit_codes.classify(g.scored() for g in gates)
    if code != world.exit_code:
        mismatches.append(
            f"exit code: registered {exit_codes.CODE_NAMES[world.exit_code]}, got "
            f"{exit_codes.CODE_NAMES[code]}")
    lines = ["", f"## SELF TEST: {world.name}", f"   {world.why}",
             f"   planted aD {world.depth:+.3f}  aK {world.block_k:+.3f}  "
             f"aR {world.residency:+.3f} per doubling, on w0 {PLANTED_W0}, "
             f"at a per-rep spread of {noise:.4%}",
             "   every planted repeat cycles the five clock states "
             + ", ".join(name for name, *_ in PLANTED_CLOCKS)
             + "; the two DRIFT states carry a time 30% off the world's own "
               "ladder, so a fit that failed to exclude them misses the "
               "planted coefficients"] + report_lines(analysis)
    if mismatches:
        lines += ["", "REGISTRATION MISMATCH, which is the apparatus failing "
                  "and not a claim failing:"] + [f"  {m}" for m in mismatches]
    return lines, gates, (exit_codes.ERROR if mismatches else code)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Separate residency from pipeline depth on the BLOCK_K "
                    "diagonal.")
    p.add_argument("--model", default="mixtral-8x7b",
                   choices=sorted(MODEL_CONFIGS))
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--cells", default=DEFAULT_CELLS,
                   help="STAGESxBLOCK_K, comma separated. THE SWEEP.")
    p.add_argument("--treads", type=int, default=8,
                   help="exactly-full tile stacks per cell, n = 1..treads")
    p.add_argument("--reps", type=int, default=15,
                   help="round-robin passes per cell")
    p.add_argument("--warmup", type=float, default=300.0,
                   help="milliseconds of delivered GPU load before timing")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--cell-budget-ms", type=float, default=200.0)
    p.add_argument("--no-l2-flush", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--draws", type=int, default=200,
                   help="bootstrap draws over the repeats")
    p.add_argument("--plant-noise", type=float, default=None,
                   help="per-cell relative spread the planted worlds and the "
                        "MDE are computed at; measured off this run's own "
                        "cells when they exist")
    p.add_argument("--smem-tolerance", type=float, default=0.10,
                   help="how far the compiled shared memory may differ from "
                        "the model before V2 fails")
    p.add_argument("--capability", default="",
                   help="MAJOR.MINOR; may name an ABSENT card but may never "
                        "contradict a present one")
    p.add_argument("--num-warps", type=int, default=PINNED_WARPS,
                   help="warps per CTA, HELD AT ONE VALUE FOR THE WHOLE GRID. "
                        "It is the escape from the register collapse V5 refuses "
                        "on: halving the CTA's thread count roughly doubles how "
                        "many CTAs the register file holds. In the run id.")
    p.add_argument("--sm-count", type=int, default=0)
    p.add_argument("--l2-bytes", type=int, default=0)
    p.add_argument("--card", default="")
    p.add_argument("--ridge", type=float, default=0.0)
    p.add_argument("--ridge-band", default="")
    p.add_argument("--bandwidth-gbps", "--bandwidth", type=float, default=0.0,
                   dest="bandwidth_gbps")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--run-id", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", default=None, choices=sorted(SELF_TEST_WORLDS))
    p.add_argument("--new", action="store_true",
                   help="refuse to resume: write into a fresh directory")
    # ACCEPTED AND IGNORED. It softens a CLAIM_FAIL into DONE, and this arm's
    # registered outcomes include a CLAIM_FAIL that IS the finding -- the world
    # where w follows residency. Reporting that as 0 would tell a session
    # driver the opposite of what happened. `_exit_code` returns what
    # `classify` returns, always.
    p.add_argument("--fail-on-gate", action="store_true",
                   help="accepted and ignored; the exit code always comes from "
                        "moe.bench.exit_codes.classify")
    return p


def resolve_spread(explicit: float | None, samples: list[Sample]
                   ) -> tuple[float, str]:
    """The per-cell relative spread the MDE and the planted worlds use.

    From THIS run's own cells when there are any on disk -- a resume, or a
    replay of a pod directory pointed at with `--out` -- and otherwise from the
    published figure, labelled ASSUMED. The order matters: an argparse default
    in the middle of the published range is one number deciding whether the
    experiment looks worth paying for, while describing a pod quieter than half
    the corpus.
    """
    if explicit is not None:
        if explicit <= 0:
            raise RefusedBeforeMeasuring(
                "--plant-noise must be positive: a zero spread makes every "
                "minimum detectable effect zero and every gate resolvable")
        return explicit, "given on the command line"
    spreads = []
    for key in {s.cell for s in samples}:
        _, spread = collapse(samples, key)
        if spread:
            spreads.append(spread)
    if spreads:
        return statistics.median(spreads), (
            f"measured on this run's own {len(spreads)} cell(s) already on "
            "disk")
    return ASSUMED_W_SPREAD, ASSUMED_W_SPREAD_SOURCE


def _main(argv=None) -> int:                                    # noqa: C901
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    b = dtype_bytes(args.dtype)
    synthetic = bool(args.dry_run or args.self_test)
    cells = parse_cells(args.cells)
    if args.num_warps < 1 or args.num_warps & (args.num_warps - 1):
        raise RefusedBeforeMeasuring(
            f"--num-warps {args.num_warps} is not a positive power of two, "
            "which is what Triton's warp scheduler wants.")
    rows = ladder_rows(cfg, args.treads)

    if args.treads < MIN_FIT_TREADS:
        raise RefusedBeforeMeasuring(
            f"--treads {args.treads} is below the {MIN_FIT_TREADS} treads V4 "
            "requires of every cell, so this grid is refused at plan time "
            "rather than measured and then voided.")

    capability = SWEEP.resolve_capability(args, synthetic=synthetic)

    try:
        rr = SWEEP.resolve_ridge(args, synthetic=synthetic)
    except SWEEP.RidgeUnavailable as exc:
        raise RefusedBeforeMeasuring(str(exc)) from exc
    bw = SWEEP.resolve_bandwidth(args, synthetic=synthetic)

    # `SWEEP.detect_card_slug` RETURNS A STRING AND NEVER NONE: with no usable
    # device it returns `NO_CARD_SLUG`, which is the absence and not a card.
    # Read as a detected card it makes every `--card` on a laptop contradict a
    # device that is not there, which is the refusal saying the opposite of
    # what it means.
    probed = SWEEP.detect_card_slug()
    detected = "" if probed == SWEEP.NO_CARD_SLUG else probed
    card = args.card or detected or (SYNTHETIC_CARD if synthetic
                                     else SWEEP.NO_CARD_SLUG)
    # --card takes a NAME ('NVIDIA H200', as nvidia-smi and the driver's
    # dtype line spell it) or a slug; `detected` is a slug. Compared as
    # slugs, or the pod refused a test's --card 'NVIDIA H200' as
    # contradicting 'nvidia_h200' (session 4).
    if args.card and detected and PV.card_slug(args.card) != detected:
        raise RefusedBeforeMeasuring(
            f"--card {args.card!r} but the attached device is {detected!r}. "
            "--card may name a card that is ABSENT, so a laptop can print the "
            "pod's real path; it may never contradict one that is present. "
            "Nothing measured.")

    # THE CARD'S OCCUPANCY LIMITS COME FROM THE ARM WHOSE P6 THIS ONE ANSWERS.
    # `occupancy_vs_swizzle.resolve_device` is the study's one statement of the
    # order -- command line, then the ATTACHED DEVICE's own
    # `get_device_properties`, then a labelled hypothesis under --dry-run and
    # --self-test only, then refuse -- and a second copy of it here would be a
    # second place for the SM count and the L2 to come from. It also fills in
    # the capability off the device, so a pod run needs neither flag.
    try:
        limits, l2_source, device = OCC.resolve_device(args, synthetic=synthetic)
    except OCC.CardUnavailable as exc:
        raise RefusedBeforeMeasuring(str(exc)) from exc
    capability = capability or limits.capability

    run_id = args.run_id or default_run_id(args, card)
    out_dir = (args.out or SWEEP.results_root()) / "blockk_diagonal" / run_id
    csv_path = out_dir / "cells.csv"
    card_path = out_dir / "CARD"
    cache_root = out_dir / "triton-cache"

    if args.new and csv_path.exists():
        raise RefusedBeforeMeasuring(
            f"--new was given and {csv_path} already exists. This run would "
            "append a second copy of every tread under the first run's header "
            "and both copies would enter the same median. Move or delete that "
            "directory, or change a swept knob so the run id differs.")
    on_disk, prior = read_samples(csv_path)
    spread, spread_source = resolve_spread(args.plant_noise, prior)

    # RESIDENCY OFF GPU IS AN UPPER BOUND AND THE PLAN SAYS SO. `registers` is
    # None here at every cell, so `CellResidency.bound` is True and every rung
    # is labelled. The pod's own ladder is built from `n_regs` read back off
    # the compiled kernels and V3 refuses to score anything else.
    planned_res = {c.key: cell_residency(c, b, limits, warps=args.num_warps)
                   for c in cells}
    design = build_design([c.key for c in cells], planned_res,
                          {c.key: c for c in cells})
    swing, swing_source = concurrency_swing(cfg, planned_res, b, limits)
    refusals = feasibility(cfg, cells, b, capability, args.num_warps)
    secs = estimated_seconds(cells, args.treads, args.reps, args.warmup,
                             args.trials, args.cell_budget_ms)

    header = [
        "experiment  blockk_diagonal: separate residency from pipeline depth "
        "on the BLOCK_SIZE_K diagonal",
        f"run id      {run_id}", ""]
    header += predictions_text(cfg, cells, b, planned_res, design, rr.ridge,
                               rr.source, swing, swing_source)
    header += [
        "", "## The plan", "",
        f"model        {cfg.name} E={cfg.num_experts} k={cfg.top_k} "
        f"{args.dtype} ({b} B/element)",
        f"pinned       BLOCK_SIZE_M={SUBJECT_BLOCK_M} "
        f"BLOCK_SIZE_N={PINNED_BLOCK_N} GROUP_SIZE_M={PINNED_GROUP_M} "
        f"num_warps={args.num_warps}  (num_stages and BLOCK_SIZE_K are the "
        "sweep)",
        f"cells        {len(cells)}: " + ", ".join(c.key for c in cells),
        f"treads       {args.treads} exactly-full tile stacks per cell, "
        f"r = {rows[0]}..{rows[-1]} rows per expert, T = "
        f"{SWEEP.tokens_for_rows(cfg, rows[0])}.."
        f"{SWEEP.tokens_for_rows(cfg, rows[-1])} tokens",
        f"repeats      {args.reps} round-robin passes per cell",
        f"timing       {args.warmup:.0f} ms of warmup under load, then "
        f"{args.trials} trial(s) of {args.cell_budget_ms:.0f} ms of kernel "
        f"time each, L2 flush "
        f"{'OFF' if args.no_l2_flush else 'ON'}; the instrument sizes its own "
        "iteration count",
        f"timings      {len(cells) * args.treads * args.reps} "
        f"({len(cells)} cells x {args.treads} treads x {args.reps} reps)",
        f"estimate     {secs:.0f} s of GPU at what the instrument charges "
        "(warmup + trials x budget per timing), excluding compiles and "
        "allocation",
        "not priced   the two terms that figure leaves out, with what is known "
        "about each rather than a shrug. COMPILES: at most "
        f"{len(cells)} x {args.treads} = {len(cells) * args.treads} distinct "
        "Triton specialisations, and probably "
        f"{len(cells)} -- the kernel's constexprs are the tile, and every "
        "token count on this ladder is a multiple of 16, so the "
        "divisibility-by-16 specialisation key does not move across the "
        "treads. The COMPILE CENSUS pays the first one per cell before the "
        "metered loop and prints the count it actually built. ALLOCATION: the "
        "expert weight set is drawn ONCE, not once per tread -- "
        "moe.reference.torch_ref.make_inputs caches it on "
        "(model, dtype, seed, device, scale), which this arm never moves -- so "
        f"the {len(cells) * args.treads * args.reps} rebuilds redraw only the "
        f"activations, at most "
        f"{cfg.hidden_size * SWEEP.tokens_for_rows(cfg, rows[-1]) * b / 1e6:.0f}"
        " MB each.",
        f"card         {limits.line()}",
        f"card slug    {card}" + ("" if detected else
                                  "  (NO DEVICE: this id is the off-GPU one "
                                  "and is not what a pod derives; --card "
                                  "<slug> prints that)"),
        f"ridge        {rr.ridge:.3f} Op/B, {rr.source}",
        f"bandwidth    {bw.gbps:.1f} GB/s, {bw.detail}",
        "instrument   " + (SWEEP.timing_basis() or
                           "NOT NAMEABLE on this box (no importable torch), "
                           "which is also the case in which nothing here "
                           "measures anything"),
        "", "TILE RESOURCE BILL, one CTA, at "
        + (f"sm_{capability[0]}{capability[1]}" if capability
           else "an UNKNOWN device (--capability MAJOR.MINOR gives the "
                "shared-memory verdict; the register check runs regardless)")]
    for cell in cells:
        res = SWEEP.tile_resources(cell.pinned(args.num_warps),
                                   SUBJECT_BLOCK_M, b, capability)
        header.append(f"  {cell.key:8s}" + res.render())
    header += [""] + register_sensitivity(cells, b, limits, args.num_warps)
    header += ["",
               "WEIGHT-STREAM DENOMINATOR. One complete stream of this layer's "
               "routed expert weight set is "
               f"{WEIGHTS.weight_stream_ms(cfg, args.dtype, bw.gbps):.4f}"
               f" ms: "
               f"{WEIGHTS.routed_expert_weight_bytes(cfg, args.dtype) / 1e9:.4f}"
               f" GB at {bw.gbps:.1f} GB/s. w scales exactly 1:1 in "
               "that rate, so every w on this page names it.", ""]
    header += mde_line(design, spread, spread_source) + [MDE_BOUND_NOTE]
    header += ["",
               f"WRITES TO    {out_dir}",
               f"             {git_visibility(out_dir)}",
               "             cells.csv (one row per tread per repeat, "
               "flushed), CARD, report.txt, report.json, triton-cache/"]

    hard = {k: v for k, v in refusals.items() if v}
    if hard:
        header += ["", "REFUSED: a cell cannot physically run as pinned, and "
                   "no cell here is optional -- each is a rung of the design "
                   "matrix printed above, and dropping one changes its rank."]
        header += [f"  {k}: {v}" for k, v in sorted(hard.items())]
        print("\n".join(header))
        return exit_codes.REFUSED

    if not design.identified and not args.self_test:
        header += ["", "REFUSED: the design as this card realises it is "
                   "singular, so at least one of depth, BLOCK_K and residency "
                   "is a combination of the others and no number of repeats "
                   "separates them. Widen --cells so that two cells share a "
                   "resident-block count with different num_stages."]
        print("\n".join(header))
        return exit_codes.REFUSED

    if args.dry_run:
        print("\n".join(header))
        # REFUSED (2) AND NOT DONE (0). A dry run scores no gate, so it prints
        # no RESULT line, and `exit_codes.classify_text` over this log raises
        # `NoGatesScored` -- which that module documents as what a REFUSED log
        # looks like from there. DONE says "measured; every gate PASSED".
        print("\n".join(["", "=" * 72,
                         "REFUSED. Nothing was measured and nothing was "
                         "written.",
                         "  reason: --dry-run was given",
                         "  Everything above is arithmetic over this repo's "
                         "calibration and vLLM's",
                         "  resource model. No gate was scored, so no RESULT "
                         "line was printed and",
                         "  none of it is a result. Run --self-test <world> "
                         "for the planted worlds,",
                         "  or the bare command on the pod.",
                         "=" * 72]))
        return exit_codes.REFUSED

    if args.self_test:
        more, gates, code = self_test(
            args, cfg, cells, rows, limits, b,
            bandwidth_gbps=bw.gbps, bandwidth_source=bw.detail,
            ridge=rr.ridge, ridge_source=rr.source, noise=spread)
        print("\n".join(header + more + ["", "## Gates", ""]
                        + render_gates(gates)))
        return code

    missing = SWEEP.missing_gpu_stack()
    if missing:
        print("\n".join(header))
        print(f"\nREFUSED: {missing.split('.')[0]}.\n"
              "Off GPU this arm's whole argument is still available:\n"
              "  --dry-run                the plan, the design matrix and the "
              "cost\n"
              "  --self-test depth        the planted worlds, through the "
              "pod's own analysis\n"
              "  --self-test residency\n"
              "  --self-test blockk\n"
              "  --self-test null")
        return exit_codes.REFUSED

    out_dir.mkdir(parents=True, exist_ok=True)
    if card_path.exists() and card_path.read_text().strip() != card:
        raise RefusedBeforeMeasuring(
            f"{csv_path} was written on card "
            f"{card_path.read_text().strip()!r} and this run is on {card!r}. "
            "Resuming would fit one ladder across two machines.")
    card_path.write_text(card + "\n")

    prov = PV.provenance_block(
        instrument=SWEEP.timing_basis() or "",
        ridge=rr.ridge, ridge_source=rr.source,
        bandwidth=bw.gbps, bandwidth_source=bw.detail,
        warmup_ms=args.warmup, iters=None, target_ms=args.cell_budget_ms)

    reference_clock, clock_source = SWEEP.reference_clock_mhz()
    print("\n".join(header))
    print("\nreference clock  "
          + (f"{reference_clock:.0f} MHz, {clock_source}" if reference_clock
             else f"NOT DETERMINED ({clock_source}); every LEVEL verdict below "
                  "is None, which means not determined and excludes nothing"))

    probe = OCC.KernelProbe()
    # BEFORE ANY TIMED CELL. The census compiles each cell once, reads n_regs
    # back, rebuilds the residency ladder from it and REFUSES if the design the
    # predictions were registered against does not survive this card's register
    # file. It spends compiles the metered loop would spend anyway.
    _, _, census_compiles = compile_census(
        args, cfg, cells, rows, cache_root, probe, limits, b,
        spread=spread, spread_source=spread_source)
    samples = list(prior)
    loop_compiles, executed, observed = measure(
        args, cfg, cells, rows, csv_path, cache_root, on_disk, samples, probe,
        prov=prov, reference_clock_mhz=reference_clock)
    # ONE COUNT, FROM BOTH PLACES A KERNEL IS BUILT. The census builds each
    # cell's first specialisation and the metered loop builds one more per new
    # token shape; V1 reads the sum, because a gate that read only the loop
    # would score every cell as having compiled nothing the moment a census
    # was put in front of it.
    compiles = {c.key: census_compiles.get(c.key, 0)
                + loop_compiles.get(c.key, 0) for c in cells}
    if probe.note:
        print(f"\nkernel probe: {probe.note}")

    # THE ITERATION COUNT ON THE BLOCK IS THE ONE THE CELLS WERE TIMED AT, not
    # the one argv asked for. `time_kernel` sizes it per cell from
    # --cell-budget-ms, so a block carrying an argparse default contradicts
    # every row of cells.csv. `observed_iters` returns a NEW PROVENANCE BLOCK,
    # which is why it is reassigned here and not written into the payload.
    prov = SWEEP.observed_iters(prov, samples)
    print(SWEEP.iters_line(samples))

    analysis = analyse(cfg, cells, samples, dtype=args.dtype,
                       bandwidth_gbps=bw.gbps,
                       bandwidth_source=bw.detail, limits=limits,
                       observed=observed, b=b, draws=args.draws,
                       seed=args.seed, warps=args.num_warps)
    gates = build_gates(analysis, cfg, cells, b, ridge=rr.ridge,
                        ridge_source=rr.source, compiles=compiles,
                        executed=executed, observed=observed, limits=limits,
                        tolerance=args.smem_tolerance)
    body = report_lines(analysis) + ["", "## Gates", ""] + render_gates(gates)
    print("\n".join(body))

    report = "\n".join(header + [""] + body) + "\n"
    (out_dir / "report.txt").write_text(report)
    payload = report_payload(analysis, gates, prov, run_id=run_id,
                             model=cfg.name, dtype=args.dtype, cells=cells)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2,
                                                    default=str))
    for label, path in (("cells", csv_path), ("report", out_dir / "report.txt"),
                        ("json", out_dir / "report.json")):
        print(f"{label:8s} {path}\n         {git_visibility(path)}")
    return exit_codes.classify(g.scored() for g in gates)


def main(argv=None) -> int:
    """Convert a string `SystemExit` into REFUSED and a crash into ERROR.

    `raise SystemExit("some sentence")` exits ONE, which is CLAIM_FAIL in the
    table this repo reads, so a run that refused before measuring anything would
    be filed as a measured refutation. And an unplanned exception exits one as
    well, so a torch OOM and a refuted claim would arrive at a driver as one
    number: ERROR (4) is outside `FINISHED_CODES` precisely so "the apparatus
    broke" can be told from "the claim did not hold", and the arm may be re-run.
    """
    try:
        return _main(argv)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            msg = (exc.code if exc.code.startswith("REFUSED")
                   else f"REFUSED: {exc.code}")
            print(msg, file=sys.stderr)
            return exit_codes.REFUSED
        # AN INT CODE IS RETURNED, NOT RE-RAISED. `RefusedBeforeMeasuring`
        # carries `exit_codes.REFUSED`, and re-raising it would make
        # `main()` throw where `sys.exit(main())` happens to do the right
        # thing and every other caller -- this repo's own tests included --
        # sees an exception instead of a code. Bools are excluded because
        # `True == 1` and CLAIM_FAIL is not a thing to reach by accident.
        if isinstance(exc.code, int) and not isinstance(exc.code, bool):
            return exc.code
        raise
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        print("ERROR: blockk_diagonal crashed before it could reach a verdict. "
              "This is the apparatus failing, not a claim failing, so it exits "
              f"{exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: the "
              "traceback above is the thing to fix, and the arm may be re-run.",
              file=sys.stderr)
        return exit_codes.ERROR


#: Read by `tests/test_blockk_diagonal.py`, which parses THIS file and asserts
#: that these are exactly the functions testing the clock verdicts -- the one
#: that EXCLUDES on them and the one that COUNTS them. Kept here so the test
#: names the rule rather than a line number, and so the count lives in one
#: place: five sentences in this file said "exactly one" against this tuple's
#: two.
CLOCK_VERDICT_READERS = ("clock_excluded", "clock_state")


class _BlankStrings(ast.NodeTransformer):
    """Replace every string constant with an empty one.

    So that a function which MENTIONS the rule in a message, or carries the
    phrase as a search needle, is not counted as applying it. Stripping
    docstrings alone is not enough: the finder below holds the two needles as
    string literals in its own body and found itself.
    """

    def visit_Constant(self, node):                     # noqa: N802
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=""), node)
        return node


def _blank_strings(node):
    return _BlankStrings().visit(ast.parse(ast.unparse(node)))


def _clock_verdict_readers(path: Path | None = None) -> set[str]:
    """Which top-level functions in this file test a clock verdict.

    Docstrings are stripped before the search, so a function that DESCRIBES the
    rule is not counted as applying it -- which is the difference between the
    one reader and the several places that explain why there is one.
    """
    tree = ast.parse((path or Path(__file__)).read_text())
    owner: dict[ast.FunctionDef, str] = {}
    for top in tree.body:
        if isinstance(top, ast.FunctionDef):
            for node in ast.walk(top):
                if isinstance(node, ast.FunctionDef):
                    owner[node] = top.name
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = "\n".join(ast.unparse(_blank_strings(s)) for s in node.body
                         if not (isinstance(s, ast.Expr)
                                 and isinstance(s.value, ast.Constant)))
        # The needles are the SUFFIXES, so they match both the column name
        # (`s.clock_drift_ok is False`, which is how a counter reads a row) and
        # the parameter name (`drift_ok is False`, which is how the one reader
        # states the rule). Matching only the column name missed the reader.
        if "drift_ok is False" in body or "level_ok is False" in body:
            found.add(owner.get(node, node.name))
    return found


if __name__ == "__main__":
    sys.exit(main())
