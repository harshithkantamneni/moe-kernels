#!/usr/bin/env python
"""Which BLOCK_SIZE_M can reach the compute roof at all, and where does it cross?

    python scripts/block_m_crossing_sweep.py --self-test 0.558   # no GPU needed
    python scripts/block_m_crossing_sweep.py --self-test 0.10    # the retracted world
    python scripts/block_m_crossing_sweep.py --dry-run           # print the grid and stop
    python scripts/block_m_crossing_sweep.py                     # the pod run
    python scripts/block_m_crossing_sweep.py --model qwen2-57b-a14b --r-max 1536

WHY THIS EXISTS AND WHY `scripts/tile_sweep.py` CANNOT ANSWER IT. That script
runs deepseek-v3 at T=16/64/256, which is 0.5 to 8 rows per expert. At 0.5 to 8
rows EVERY expert is one M-tile at EVERY BLOCK_SIZE_M, so the tile count is
identical across the whole sweep, no weight re-read can be saved, and the only
thing a bigger tile can move is occupancy. It measured bigger tiles as slower,
and that measurement is sound -- of the single-tile regime. It says nothing
about the regime this study's claims live in. THIS sweep runs the MULTI-TILE
regime, where `ceil(rows_per_expert / BLOCK_M)` actually varies with BLOCK_M, and
sweeps far enough to bracket a crossing that is at a DIFFERENT batch for each
block size.

THE MODEL BEING FALSIFIED. One expert holding `r` rows is scheduled as
`n = ceil(r / BLOCK_M)` M-tiles. The first tile reads that expert's weights in
full; each extra tile re-reads them, discounted by L2 to a fraction `alpha`:

    Q(n) = 1 + alpha (n - 1)        AI(r) = (2 r / b) / Q(n)

Two consequences, and they are the whole experiment:

  * AI is BOUNDED at `2 BM / (alpha b)`. A block size whose cap sits below the
    hardware ridge can NEVER be compute bound, at any batch, ever.
  * the crossing solves `R = ridge b Q(R) / 2`, a step function on both sides,
    so it moves in jumps of `Q` as the block size changes which tread it lands
    in.

ALPHA WAS REFIT FROM 0.10 TO 0.558 (90% band 0.529-0.588, 10,813 rows, placebo
-0.002). The 0.10 was an estimator artefact. The two values are not a
quantitative disagreement, they are two different worlds, and this sweep is
built to tell them apart:

    BLOCK_M   AI cap    alpha=0.558                 alpha=0.10 (retracted)
       32       57.3    NO CROSSING EVER            crosses at R=304.6
       64      114.7    NO CROSSING EVER            crosses at R=208.4
      128      229.4    crosses at R=249.7          crosses at R=176.3
      256      458.8    crosses at R=160.3          crosses at R=160.3

At 0.558 the 128-to-256 crossing ratio is 1.558 and two of the four block sizes
never cross. At 0.10 the ratio is 1.10 and all four cross. Everything below is
arranged so the data has to pick one.

WHAT IS PINNED AND WHY. `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`,
`num_warps` and `num_stages` are held at the values `tile_sweep.py` pins, so the
two experiments are comparable and so BLOCK_SIZE_M is the only thing that can
move. GROUP_SIZE_M=1 in particular is load bearing twice over: it is the swizzle
setting whose L2 behaviour the refit's GROUP_SIZE_M=1 slice describes
(alpha 0.570 there against 0.558 pooled, a 2% difference), and leaving it free
would let the swizzle change the very re-read fraction being measured.

WHAT IS NOT PINNED AND CANNOT BE. `alpha` itself drifts with BLOCK_M -- 0.466 at
64, 0.625 at 128 -- which is what a swizzle-for-L2-reuse mechanism predicts and
which this sweep MEASURES per block size rather than assuming. The drift does not
rescue the retracted value: at 0.466 the BLOCK_M=64 cap is 137.3, still below the
ridge, so gate 4's prediction survives its own worst case.

ROUTING IS EXACTLY BALANCED, not sampled uniform, and that is a design decision
rather than a convenience. Sampled uniform routing at T=1024 on mixtral puts
about 15 rows of spread on a mean of 256, which smears every tile step across
+/-60 tokens and makes gate 1 unfalsifiable. `realize_counts` builds an
assignment whose per-expert histogram is EXACTLY `T k / E`, so rows per expert is
an integer the script knows in advance, tile steps land on a token count that can
be named before the run, and the sub-saturation hazard that forces
`crossing.py` to carry a `min_tokens` floor cannot arise: every expert holds the
same number of rows at every point of the grid.

WHAT IT WRITES, AND WHERE IT SURVIVES TEARDOWN. Everything lands under
`$MOE_RESULTS_DIR`, or `/workspace/results` when that exists (the RunPod network
volume, which outlives the pod), or `<repo>/results` on a laptop:

    <results>/block_m_crossing/<run-id>/cells.csv     one row per (BLOCK_M, T)
    <results>/block_m_crossing/<run-id>/report.json   predictions and gate verdicts
    <results>/block_m_crossing/<run-id>/report.txt    exactly what was printed
    <results>/block_m_crossing/<run-id>/triton-cache/ per-setting compile evidence

The exact path is printed at start AND at the end. `cells.csv` is appended and
flushed cell by cell, and a re-run with the same arguments resumes it, so
aborting costs only the cell in flight. The default run id is derived from the
arguments, so "the same experiment" means "the same directory" without anyone
having to remember an id.

THE COMPILE ASSAY, which is a gate and not a detail. If `override_config` fails
to take effect, all four settings run the SAME kernel, every gate reads a
difference of zero, and the report looks like a clean null result. The assay
against that is to count the Triton artefacts that appear during each setting: a
setting that compiled nothing new either did not change the kernel or was served
from a warm cache. Both are fatal to the experiment and both are silent
otherwise -- a stale cache is what cost this project its A100 PTX dump -- so
`TRITON_CACHE_DIR` is pointed at a fresh directory under the results path before
vLLM is imported, and at a per-setting subdirectory before each setting runs.
Gate 0 refuses to let the other four be read if a setting that ran cells
compiled nothing. A setting that ran NO cells, because a previous session
already measured them, is a different state: the assay belongs to that session
and gate 0 says so rather than scoring it.

OFF-GPU. `--self-test ALPHA` generates the cells from the physical model at that
alpha and runs the entire analysis on them, so the gates, the fits and the
report are exercised on a laptop, and so the claim "these gates can tell 0.558
from 0.10" is checkable rather than asserted. `--dry-run` prints the grid, the
predictions and the cost estimate without touching a GPU. Absent torch, CUDA or
vLLM the script says which one is missing and what to run instead.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

# --------------------------------------------------------------------------
# The numbers this script is arguing about. All of them stated up front so a
# reader can check what was assumed without reading the code.
# --------------------------------------------------------------------------

#: Refit 2026-08-31 against the derived tile with a per-group intercept.
ALPHA = 0.558
ALPHA_BAND = (0.529, 0.588)

#: The published value this refit replaces. Kept because every gate here is
#: designed to DISCRIMINATE against it, and a gate that cannot name what it
#: rules out is a gate nobody can check.
RETRACTED_ALPHA = 0.10

#: alpha measured per BLOCK_M. The scalar above is the pooled fit; these are the
#: slices, and they are what gate 4's worst case is run against.
ALPHA_BY_BLOCK_M = {64: 0.466, 128: 0.625}

#: THE 2026-08-26 H200 RIDGE BAND. Three calibrations of that card disagreed by
#: 9.9% on the compute term, and the two ends do not merely widen a band: they
#: change which TREAD the BLOCK_M=128 crossing lands in (2 at 160.3, 3 at 176.2).
#:
#: IT IS NOT A DEFAULT AND MUST NEVER BECOME ONE AGAIN. `--ridge` used to default
#: to `RIDGE_BAND[0]` and `scripts/cross_card_surface.sh` never passed `--ridge`,
#: so all 7 published A100 reports carry ridge=160.3 and ridge_band=[160.3,176.2]
#: -- a band belonging to NEITHER card. The A100's own contemporaneous
#: calibration is 262.371/1.79936 = 145.8 and the H200's is 712.259/4.37476 =
#: 162.8, so every printed `ridge x bandwidth` on the A100 was a hybrid of two
#: machines. `resolve_ridge` now reads the ATTACHED device's calibration and
#: REFUSES when there is none; this constant survives only as the hypothesis a
#: laptop planning run (--dry-run / --self-test) is allowed to assume, where
#: nothing was measured and so nothing can be mislabelled, and as the value
#: `scripts/tile_cap_test.py` imports.
RIDGE_BAND = (160.3, 176.2)

#: What a report says when its ridge is this constant rather than a measurement.
#: Carried into `report.json` so the provenance cannot be lost between the
#: printout and the file, which is how 160.3 reached seven A100 reports unnoticed.
HYPOTHESIS_RIDGE_SOURCE = (
    "HYPOTHESIS: the 2026-08-26 H200 band, which belongs to no attached device")

#: The card slug a run id carries when no device is attached: every --dry-run
#: and every --self-test on a laptop. Visible rather than blank, so a laptop
#: directory cannot be mistaken for the one a pod would write to.
NO_CARD_SLUG = "nocard"


def detect_card_slug() -> str:
    """Slug for the ATTACHED device, or `NO_CARD_SLUG`.

    Read at run-id time, because the card belongs IN the id. See
    `default_run_id` for the collision it prevents.
    """
    try:
        import torch
    except ImportError:
        return NO_CARD_SLUG
    try:
        if not torch.cuda.is_available():
            return NO_CARD_SLUG
        name = torch.cuda.get_device_name(0)
    except Exception:                                   # noqa: BLE001
        # A driver present but unusable is not a card identity, and naming it
        # `nocard` keeps the run out of a real card's directory.
        return NO_CARD_SLUG
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or NO_CARD_SLUG

#: Held fixed so BLOCK_SIZE_M is the only thing that can move. Same values as
#: `scripts/tile_sweep.py` pins, so the two experiments compose. num_warps=8
#: satisfies Triton's `BLOCK_M % 64 == 0 AND num_warps % 4 == 0` warpgroup
#: predicate at every setting that can reach it, leaving BLOCK_SIZE_M as the
#: only thing that flips the instruction.
FIXED = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
         "num_warps": 8, "num_stages": 4}

DEFAULT_BLOCK_SIZES = (32, 64, 128, 256)

#: H200. Only used to turn a block count into waves, and printed with its
#: source so a wave count is never mistaken for something the driver reported.
DEFAULT_SM_COUNT = 132

#: WHAT GATE 3 ACTUALLY TESTS. The midpoint between the retracted alpha (0.10)
#: and the refit one (0.558), rounded to the value the published gate used.
#:
#: The gate used to be phrased as a CROSSING RATIO and scored `1 + alpha_hat`
#: against 1.33. That "measured" field was an algebraic restatement of a fitted
#: alpha imported from a DIFFERENT BLOCK_M -- checked against all 22 published
#: reports that carry one, where measured == 1 + alpha_measured in 22 of 22 and
#: differs in 0 -- so the ratio was never an observation. Worse, the identity
#: `R_cross(128)/R_cross(256) = 1 + alpha` holds only if the 128 crossing lands
#: in tread 2, which at the A100's own alphas it does not in 3 of 6 arms; and
#: across all of `results/published` the ladder field `crosses` is False 41
#: times, null 61 times and True ZERO times. No crossing has ever been observed
#: by this study, so no gate may be phrased as though one was. The gate now
#: reports the fitted alpha itself, against this threshold.
GATE3_ALPHA_DISCRIMINATOR = 0.33

#: The same threshold in the ratio units the published gate printed. Kept so the
#: old verdict can be recomputed from a new report, and so `1 + alpha` never has
#: to be reconstructed by a reader guessing at what was compared.
GATE3_DISCRIMINATOR = 1.0 + GATE3_ALPHA_DISCRIMINATOR

#: Gate 2 asks a direction, and a direction needs a magnitude or noise answers
#: it. 1.5x at the largest common rows-per-expert is 5x the worst per-cell
#: timing spread this harness has produced.
GATE2_RATIO = 1.5

#: LEGACY, AND A UNIT ERROR. Gate 4 used to score `top > 0.85` where `top` is a
#: fraction of the ARM'S OWN measured plateau, while the 0.85's stated rationale
#: (cap/ridge = 0.716, "leaves room for the fused layer's non-GEMM work") is a
#: fraction of PEAK COMPUTE. Those are not the same denominator: across the 14
#: s3 reports the plateau is 50.5-71.8% of ridge x bandwidth and swings
#: 145.7-198.4 TFLOP/s inside ONE A100 session, so in the gate's own units the
#: model ceiling for the arm that "failed" it is 1.42 and 0.891 was never
#: evidence against anything. The two FAILs it ever produced are both qwen2 at
#: GROUP_SIZE_M=64, where the BLOCK_M=256 reference -- the plateau, the
#: DENOMINATOR -- drops to 41.36 rows/ms against 45.1-45.7 elsewhere while
#: BLOCK_M=64 is unchanged. That is a denominator artefact, not a crossing.
#:
#: `gate_4_no_crossing` now works in fractions of `ridge x bandwidth` on BOTH
#: sides and derives its threshold per run from the two worlds it separates.
#: This constant is retained only because `scripts/tile_cap_test.py` imports it.
GATE4_ROOF_FRACTION = 0.85

#: The two worlds gate 4 separates must actually be apart before a verdict means
#: anything. At alpha=0.558 the BLOCK_M=64 ceiling is cap/ridge = 0.70 (H200) or
#: 0.79 (A100) and the retracted alpha=0.10 puts it at 1.00, a gap of 0.21-0.30.
#: Below this the gate is not discriminating between them and says so instead of
#: scoring.
GATE4_MIN_SEPARATION = 0.10

#: A tread whose top throughput is this close to the plateau is compute bound.
COMPUTE_BOUND_FRACTION = 0.95

#: Treads a ladder needs before its alpha may be quoted in a verdict.
#:
#: Two points make a line with no residual, so a two-tread fit cannot notice
#: that one of its points was wrong -- and the tread most likely to be wrong is
#: the last one, which sits right where the two branches meet. At 1% timing
#: spread a two-tread fit at BLOCK_M=128 reported alpha 0.486 on cells planted
#: at 0.10, and gate 3 passed on it. Three is the fewest treads that can
#: disagree with themselves. Ladders below the threshold are still PRINTED,
#: because "this block size saw two memory-bound treads" is information; they
#: are just not allowed to decide anything.
MIN_MEMORY_TREADS = 3


# --------------------------------------------------------------------------
# The model. Pure arithmetic: no torch, no GPU, no CSV.
# --------------------------------------------------------------------------

def q_of_tiles(tiles: int, alpha: float) -> float:
    """`1 + alpha (n - 1)`: weight traffic in units of one full read."""
    return 1.0 + alpha * (tiles - 1)


def ai_cap(block_m: int, alpha: float, b: int = 2) -> float:
    """`2 BM / (alpha b)`, the arithmetic intensity this tile height cannot pass.

    The reason alpha is not a nuisance parameter. If this sits below the ridge,
    the block size cannot be compute bound at any batch size that exists.
    Infinite at alpha <= 0, which is the correct reading and not a guard: with
    no re-read cost `2r/b` is exact and unbounded.
    """
    return math.inf if alpha <= 0 else 2.0 * block_m / (alpha * b)


@dataclass(frozen=True)
class TilePrediction:
    """What one BLOCK_SIZE_M is predicted to do, before anything runs."""

    block_m: int
    alpha: float
    ridge: float
    dtype_bytes: int
    ai_cap: float
    #: The first M-tile count at which the expert is compute bound. None means
    #: NO CROSSING AT ALL, which is a prediction and not a missing value.
    first_compute_tread: int | None
    #: `ridge b Q(n*) / 2`, rows per expert. None when there is no crossing.
    crossing_rows: float | None

    @property
    def crosses(self) -> bool:
        return self.crossing_rows is not None

    def crossing_tokens(self, num_experts: int, top_k: int) -> float | None:
        """Rows per expert back into tokens: `R E / k`, saturated routing."""
        if self.crossing_rows is None:
            return None
        return self.crossing_rows * num_experts / top_k


def predict_tile(block_m: int, alpha: float, ridge: float, b: int = 2,
                 max_tiles: int = 4096) -> TilePrediction:
    """Solve `n BM >= ridge b Q(n) / 2` for the first tread that is compute bound.

    Scanned rather than solved in closed form because both sides step: the left
    is the padded row count, which only takes multiples of BM, and the right
    moves in units of alpha. A closed form would have to round, and rounding is
    where the two ends of the ridge band stop agreeing about which tread the
    crossing lands in.

    Terminates for the right reason. The right-hand side grows by
    `ridge b alpha / 2` per tread and the left by `BM`, so a solution exists iff
    `BM > ridge b alpha / 2`, which is exactly `ai_cap > ridge`. When the cap is
    below the ridge the gap widens forever and `max_tiles` only decides how long
    the script is willing to demonstrate that; None is then the answer, not a
    timeout.
    """
    cap = ai_cap(block_m, alpha, b)
    if cap <= ridge:
        return TilePrediction(block_m, alpha, ridge, b, cap, None, None)
    for n in range(1, max_tiles + 1):
        rows = ridge * b * q_of_tiles(n, alpha) / 2.0
        if n * block_m >= rows:
            return TilePrediction(block_m, alpha, ridge, b, cap, n, rows)
    raise AssertionError(                             # pragma: no cover
        f"cap {cap:.1f} exceeds ridge {ridge} but no tread crossed in "
        f"{max_tiles} tiles; the scan and the cap disagree")


def predictions(block_sizes, alpha: float, ridge: float, b: int = 2
                ) -> dict[int, TilePrediction]:
    return {bm: predict_tile(bm, alpha, ridge, b) for bm in block_sizes}


def crossing_ratio(preds: dict[int, TilePrediction], lo: int, hi: int
                   ) -> float | None:
    """`R_cross(lo) / R_cross(hi)`, gate 3's quantity. None if either misses."""
    a, c = preds.get(lo), preds.get(hi)
    if a is None or c is None or a.crossing_rows is None or c.crossing_rows is None:
        return None
    return a.crossing_rows / c.crossing_rows


# --------------------------------------------------------------------------
# Geometry: rows, tiles, waves, bytes, flops. Everything a cell knows before
# it is timed, so a report can be checked against the grid without the GPU.
# --------------------------------------------------------------------------

def rows_step(cfg) -> int:
    """Token step that keeps rows-per-expert an exact integer.

    `R = T k / E`, so T must be a multiple of `E / gcd(E, k)`. Stated as a
    function because getting it wrong does not raise: it produces a target
    histogram `realize_counts` refuses, halfway through a metered run.
    """
    return cfg.num_experts // math.gcd(cfg.num_experts, cfg.top_k)


def rows_quantum(cfg) -> int:
    """Rows-per-expert must be a multiple of this or the token count is not one.

    `T = R E / k`, so R must be a multiple of `k / gcd(E, k)`. It is 1 for
    mixtral (E=8, k=2) and for qwen2 (E=64, k=8), which is why nothing noticed
    -- and 3 for deepseek-v2-lite (E=64, k=6), whose default grid starts at
    r=28 and stepped by 32, hitting a legal row only by accident.

    build_grid did not consult it, so the model with the SMALLEST per-expert
    footprint -- the low-phi anchor of the whole alpha-versus-L2 curve -- died
    three seconds into an unattended run with `28 rows per expert is not an
    integer token count`, twice, while every other arm passed. A geometry
    constraint that excludes one model from a study is worse than a crash,
    because the surface still plots.
    """
    return cfg.top_k // math.gcd(cfg.num_experts, cfg.top_k)


def tokens_for_rows(cfg, rows: int) -> int:
    tokens = rows * cfg.num_experts / cfg.top_k
    if abs(tokens - round(tokens)) > 1e-9:
        raise ValueError(f"{rows} rows per expert is not an integer token count "
                         f"for E={cfg.num_experts} k={cfg.top_k}")
    return int(round(tokens))


def rows_for_tokens(cfg, tokens: int) -> float:
    return tokens * cfg.top_k / cfg.num_experts


def tiles_per_expert(rows: float, block_m: int) -> int:
    return max(1, math.ceil(rows / block_m))


def weight_bytes_per_expert(cfg, b: int) -> int:
    """up `[H, 2F]` plus down `[F, H]`, which is `3 F H` elements."""
    return 3 * cfg.intermediate_size * cfg.hidden_size * b


def activation_bytes_per_row(cfg, act_b: int = 2) -> int:
    """x_perm, h_up, h_act, y_perm: the traffic that grows WITH the batch.

    Named and counted because it is affine in the tile count too, so it lands
    in the fitted memory-branch slope and inflates the measured alpha. At
    mixtral it is 102 KB against a 352 MB weight read per expert, so the
    inflation runs from 1.7% of alpha at BLOCK_M=32 to 13% at 256 -- small, but
    not nothing, and the report's `alpha-corrected` column subtracts it.
    """
    cfg_terms = 2 * cfg.hidden_size + 3 * cfg.intermediate_size
    return cfg_terms * act_b


def useful_flops(cfg, rows_total: float) -> float:
    """`6 F H` per row: up is `2 F H` MACs, down is `F H`, two flops each."""
    return 6.0 * rows_total * cfg.intermediate_size * cfg.hidden_size


def waves(cfg, rows_per_expert: float, block_m: int, block_n: int,
          sm_count: int) -> tuple[float, float]:
    """CTA waves for the up and down GEMMs, one CTA resident per SM assumed.

    vLLM pads EACH EXPERT to a multiple of BLOCK_SIZE_M in
    `moe_align_block_size`, so the M-tile count is `E ceil(r / BM)` and not
    `ceil(E r / BM)`. Reported per cell because a time difference across a tile
    step is only about traffic if occupancy was saturated on BOTH sides of it,
    and one resident CTA per SM is the conservative reading: a real kernel
    fitting two would halve these and still be saturated.
    """
    m_tiles = cfg.num_experts * tiles_per_expert(rows_per_expert, block_m)
    up = m_tiles * math.ceil(2 * cfg.intermediate_size / block_n)
    down = m_tiles * math.ceil(cfg.hidden_size / block_n)
    return up / sm_count, down / sm_count


def model_ms(cfg, rows_per_expert: float, block_m: int, *, alpha: float,
             ridge: float, bandwidth_gbps: float, b: int = 2,
             overhead_ms: float = 0.0, activations: bool = True) -> float:
    """Predicted milliseconds: `overhead + max(traffic, padded compute)`.

    The generator for `--self-test`, and the source of every "predicted" column
    in the report. It is the model under test, so nothing that reads it may be
    read as evidence FOR it -- its job is to say what the data would look like
    in each of the two worlds, and the gates then ask which one arrived.

    The compute side is charged on PADDED rows. A tile computes `BM` rows
    whether or not they are useful, so a half-empty tile costs a full one, and
    that is why time is flat along a tread and steps at the tread boundary.
    """
    tiles = tiles_per_expert(rows_per_expert, block_m)
    bw = bandwidth_gbps * 1e9
    peak = ridge * bw
    weights = cfg.num_experts * weight_bytes_per_expert(cfg, b)
    traffic = weights * q_of_tiles(tiles, alpha)
    if activations:
        traffic += (cfg.num_experts * rows_per_expert
                    * activation_bytes_per_row(cfg))
    padded_rows = cfg.num_experts * tiles * block_m
    compute_s = useful_flops(cfg, padded_rows) / peak
    return overhead_ms + 1e3 * max(traffic / bw, compute_s)


# --------------------------------------------------------------------------
# The grid.
# --------------------------------------------------------------------------

def build_grid(cfg, block_sizes, r_max: int, row_step: int, step_probes: int
               ) -> list[int]:
    """Rows-per-expert to measure, as a sorted list.

    Two overlaid designs, because the four gates need different things.

    The BACKGROUND is every multiple of `row_step` (the smallest block size by
    default) up to `r_max`. That puts an EXACTLY-FULL tile stack -- `r = n BM`,
    zero padding -- on the grid for every block size, which is what the ladder
    fit reads, and it puts a common point for all four block sizes on every
    multiple of the largest one, which is what gate 2 compares across.

    The PROBES bracket each of the first `step_probes` tile boundaries per block
    size at `n BM +/- max(2, BM/8)`, because gate 1 is about where the step is
    and a grid that only samples tread tops cannot see a step at all: it has one
    point per tread and every interval it can form spans a boundary.
    """
    # Every point is snapped DOWN to a legal row count, then the illegal ones
    # are dropped rather than nudged: a probe at `edge - gap` that got rounded
    # onto `edge` would stop bracketing the boundary it exists to bracket, and
    # gate 1 would be reading a step that no point straddles.
    q = rows_quantum(cfg)
    row_step = max(row_step, q)
    if row_step % q:
        row_step += q - (row_step % q)
    grid = set(range(row_step, r_max + 1, row_step))
    for bm in block_sizes:
        gap = max(2, bm // 8)
        for n in range(1, step_probes + 1):
            edge = n * bm
            if edge > r_max:
                break
            # The point ABOVE the boundary is allowed one gap past `r_max`. A
            # boundary landing exactly on `r_max` -- which the largest block
            # size's last tread always does -- would otherwise have nothing
            # above it, and an unbracketed boundary is invisible to gate 1
            # however dense the rest of the grid is.
            for r in (edge - gap, edge, edge + gap):
                if 1 <= r <= r_max + gap and r % q == 0:
                    grid.add(r)
    return sorted(grid)


def estimated_seconds(cfg, grid, block_sizes, *, alpha: float, ridge: float,
                      bandwidth_gbps: float, b: int, iters: int, warmup: int,
                      cell_budget_ms: float) -> float:
    """Cost of the whole sweep at the model's own prediction, for --dry-run.

    Uses the SAME auto-scaling rule the runner uses, so the estimate is of the
    run that will actually happen rather than of a fixed iteration count that
    the budget would have cut.
    """
    total = 0.0
    for bm in block_sizes:
        for r in grid:
            ms = model_ms(cfg, r, bm, alpha=alpha, ridge=ridge,
                          bandwidth_gbps=bandwidth_gbps, b=b)
            total += ms * (warmup + scaled_iters(ms, iters, cell_budget_ms))
    return total / 1e3


def scaled_iters(ms: float, iters: int, cell_budget_ms: float,
                 floor: int = 5) -> int:
    """Iterations that keep one cell inside its time budget.

    An 11 ms cell at BLOCK_M=32 and 1024 rows per expert costs 50x what a 0.7 ms
    one does, and the sweep is 4 x 80 cells. Recorded per cell in the CSV, so a
    reader can see which numbers rest on 50 samples and which on 5.
    """
    if ms <= 0:
        return iters
    return max(floor, min(iters, int(cell_budget_ms / ms)))


# --------------------------------------------------------------------------
# A measured cell.
# --------------------------------------------------------------------------

@dataclass
class Cell:
    """One (BLOCK_SIZE_M, tokens) measurement, and everything derivable from it.

    `aligned` marks `rows_per_expert == n BLOCK_M` exactly, where padding is
    zero and useful throughput and padded throughput coincide. The ladder fit
    reads only aligned cells, because a partially-filled tread reports a
    throughput that depends on where in the tread it was sampled.
    """

    block_m: int
    tokens: int
    rows_per_expert: float
    tiles_per_expert: int
    padded_rows: int
    tile_eff: float
    aligned: bool
    waves_up: float
    waves_down: float
    ms_p50: float
    ms_min: float = 0.0
    ms_stdev: float = 0.0
    iters: int = 0
    useful_tflops: float = 0.0
    padded_tflops: float = 0.0
    status: str = "ok"
    detail: str = ""

    @property
    def rel_spread(self) -> float:
        return self.ms_stdev / self.ms_p50 if self.ms_p50 > 0 else 0.0


def make_cell(cfg, rows: float, block_m: int, ms: float, *, sm_count: int,
              block_n: int, ms_min: float = 0.0, ms_stdev: float = 0.0,
              iters: int = 0, status: str = "ok", detail: str = "") -> Cell:
    tiles = tiles_per_expert(rows, block_m)
    padded = cfg.num_experts * tiles * block_m
    rows_total = cfg.num_experts * rows
    up, down = waves(cfg, rows, block_m, block_n, sm_count)
    secs = ms * 1e-3
    return Cell(
        block_m=block_m, tokens=tokens_for_rows(cfg, int(rows)),
        rows_per_expert=float(rows), tiles_per_expert=tiles, padded_rows=padded,
        tile_eff=rows_total / padded if padded else 0.0,
        aligned=(rows % block_m == 0), waves_up=up, waves_down=down,
        ms_p50=ms, ms_min=ms_min, ms_stdev=ms_stdev, iters=iters,
        useful_tflops=(useful_flops(cfg, rows_total) / secs / 1e12) if secs > 0 else 0.0,
        padded_tflops=(useful_flops(cfg, padded) / secs / 1e12) if secs > 0 else 0.0,
        status=status, detail=detail)


def synthetic_cells(cfg, grid, block_sizes, *, alpha: float, ridge: float,
                    bandwidth_gbps: float, b: int, sm_count: int,
                    overhead_ms: float = 0.03, noise: float = 0.0,
                    seed: int = 0) -> list[Cell]:
    """Cells generated FROM the model, so the analysis has a known answer.

    This is what makes the whole report testable on a laptop, and it is what
    makes "these gates can tell 0.558 from 0.10" a check rather than a claim:
    generate at one alpha, read the gates, generate at the other, read them
    again. `noise` multiplies each cell by a lognormal draw so the gates are
    exercised against spread and not only against a clean curve.
    """
    import random
    rng = random.Random(seed)
    out = []
    for bm in block_sizes:
        for r in grid:
            ms = model_ms(cfg, r, bm, alpha=alpha, ridge=ridge,
                          bandwidth_gbps=bandwidth_gbps, b=b,
                          overhead_ms=overhead_ms)
            if noise:
                ms *= math.exp(rng.gauss(0.0, noise))
            out.append(make_cell(cfg, r, bm, ms, sm_count=sm_count,
                                 block_n=FIXED["BLOCK_SIZE_N"],
                                 ms_min=ms, ms_stdev=ms * noise, iters=0))
    return out


# --------------------------------------------------------------------------
# The ladder: time per tread, which is where every gate but the first is read.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderFit:
    """A max-affine fit of `t(n)` over exactly-full tile stacks at one BLOCK_M.

    THE MEASUREMENT THIS SCRIPT IS BUILT AROUND. At `r = n BM` the layer's time
    is, under the model,

        t(n) = D + max(L Q(n), n M) = D + max(A + B n, C n)

    with `L` one full weight read, `M` one tile's compute, and `D` the fused
    layer's fixed cost. BOTH branches are affine in `n`, so the whole curve is
    two lines and the fit is a split point. That matters because the two lines
    are the two mechanisms, separated:

      * `B = L alpha` is the marginal cost of one more M-tile. `alpha = B / L`
        is then a DIRECT measurement, needing no crossing and no ridge.
      * `C = M` is the marginal cost of one more tile's arithmetic, and is
        proportional to BLOCK_M by construction, which is checkable across
        settings and is checked.
      * whether the block size crosses AT ALL is `C > B`, one comparison of two
        fitted slopes, which is algebraically the same statement as
        `ai_cap > ridge` and is measured instead of assumed.

    WHICH TREADS ARE ON THE MEMORY BRANCH IS NOT DECIDED BY THE RESIDUAL, and
    that is the most load-bearing decision in this file. A free split search
    picks the split that fits best, and at BLOCK_M=128 the best-fitting split
    puts treads 1 and 2 on the memory branch and reports alpha = 0.597 -- from
    data generated at alpha=0.558 AND from data generated at alpha=0.10, because
    tread 2 is compute bound in both worlds and a line drawn through one memory
    point and one compute point answers the same thing whatever the memory
    branch was doing. A number that comes out at 0.6 under every hypothesis is
    unfalsifiable, and it would have arrived looking like a confirmation.

    So membership is decided by the COMPUTE branch instead. `C = 2 BM N / peak`
    is proportional to BLOCK_M with no free parameter, so a compute branch
    measured at one block size gives the compute branch at every block size, and
    a tread is memory bound when its time stands above that line by more than
    the margin. Treads that merely lie ON it carry no information about the
    re-read fraction and are excluded from the alpha fit rather than averaged
    into it.

    WHAT THAT COSTS, said here because the gate that needs it will otherwise
    look broken. At BLOCK_M=128 exactly ONE tread stands above the compute
    branch (tread 1; tread 2 is 256 padded rows against a 249.7 row crossing, a
    2.5% margin), so alpha at 128 is NOT IDENTIFIABLE from this sweep, in either
    world. `memory_points < 2` is that state and gate 3 says out loud that it
    imported alpha from a block size where the measurement exists. The block
    sizes that cannot cross are the ones that measure alpha best, which is a
    pleasing shape for an experiment about a ceiling.
    """

    block_m: int
    points: tuple[tuple[int, float], ...]
    #: Treads standing above the compute branch, which is the count that decides
    #: whether alpha is identifiable here at all.
    memory_points: int
    #: `A + B n`, the memory branch. Both None when fewer than 2 treads are on it.
    intercept: float | None
    slope_memory: float | None
    #: `C n` fitted through the origin on this block size's own compute-bound
    #: treads. None when it has none of its own.
    slope_compute: float | None
    #: `C` scaled from the reference block size by `C ~ BLOCK_M`. Present even
    #: where this block size never reaches its own compute branch, which is
    #: exactly the case gate 4 is about.
    slope_compute_ref: float | None
    mean_rel_err: float
    overhead_ms: float
    basis: str

    @property
    def load_ms(self) -> float | None:
        """`L = A + B`, one full weight read, the value the memory branch takes
        at a single tile."""
        if self.intercept is None or self.slope_memory is None:
            return None
        return self.intercept + self.slope_memory

    @property
    def alpha(self) -> float | None:
        """`B / L`, the fraction of a weight read an extra M-tile costs.

        TWO BIASES, both named, because their signs differ. The fused layer's
        fixed cost is inside `L` -- the branch is fitted on raw times -- which
        pushes this DOWN, and activation traffic is inside `B`, which pushes it
        UP. Only the second is correctable per block size, by
        `activation_slope_ms`, and the report prints that correction beside this
        number as `alpha-corrected`. What is NOT done is subtracting an
        extrapolated fixed cost first: on a 4-tread reference ladder under 1%
        timing spread that extrapolation wandered enough to move alpha from 0.56
        to 0.70 on data planted at 0.558. `alpha_upper` carries that end, and no
        gate is scored on it.
        """
        load = self.load_ms
        if load is None or load <= 0 or self.slope_memory is None:
            return None
        return self.slope_memory / load

    @property
    def alpha_upper(self) -> float | None:
        """`B / (L - D)`: alpha with the fused layer's fixed cost taken out.

        The high end of the range. Reported and never gated on, because `D` is
        an extrapolation to zero tiles and a 4-tread ladder under 1% timing
        spread extrapolates it to anywhere between 0.03 and 0.16 ms.
        """
        load = self.load_ms
        if load is None or self.slope_memory is None:
            return None
        net = load - self.overhead_ms
        return self.slope_memory / net if net > 0 else None

    @property
    def compute_slope(self) -> float | None:
        """This block size's `C`, measured if it has one and scaled if not."""
        return self.slope_compute if self.slope_compute else self.slope_compute_ref

    @property
    def crosses(self) -> bool | None:
        """`C > B`: does the compute branch ever overtake the memory branch.

        The same statement as `ai_cap > ridge`, made from two fitted slopes
        instead of from two assumed constants. None when a slope is missing,
        which is a different answer from "no" and is reported as a different one.
        """
        c = self.compute_slope
        if self.slope_memory is None or c is None:
            return None
        return c > self.slope_memory

    @property
    def first_compute_tread(self) -> int | None:
        return self.memory_points + 1 if self.memory_points < len(self.points) else None


def _line(xs, ys) -> tuple[float, float]:
    """Ordinary least squares `y = a + b x`. Two points give an exact line."""
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx
    return my - b * mx, b


def _through_origin(xs, ys) -> float:
    sxx = sum(x * x for x in xs)
    return sum(x * y for x, y in zip(xs, ys, strict=True)) / sxx if sxx else 0.0


def ladder_points(cells, block_m: int) -> list[tuple[int, float]]:
    """`(tiles, ms)` at exactly-full tile stacks, one point per tread.

    Aligned only. A tread sampled at 60% fill reports the same TIME as its top
    -- time is flat along a tread -- but a different throughput, and mixing the
    two is how a padding artefact enters a fit that is about traffic.
    """
    pts = {}
    for c in cells:
        if c.block_m != block_m or not c.aligned or c.status != "ok" or c.ms_p50 <= 0:
            continue
        pts[c.tiles_per_expert] = c.ms_p50
    return sorted(pts.items())


#: A tread has to stand this far above the compute branch to be called memory
#: bound. Below it the two mechanisms predict the same time and the tread
#: carries no information about the re-read fraction, whichever it is. A FLOOR,
#: not the value used: `analyse` raises it to three times the measured timing
#: spread, because the reference slope carries that spread too and a compute
#: branch estimated 2% low makes every compute-bound tread look memory bound.
MEMORY_BRANCH_MARGIN = 0.02

#: How far `B / C` has to sit from 1 before the two branches are two mechanisms.
#:
#: `B / C = ridge / ai_cap` exactly, so a memory branch running parallel to the
#: compute branch is a block size sitting precisely on its own crossing -- and
#: is far more often a fit that took a stretch of the compute branch for a
#: memory branch. At 2% timing spread that happened on 2 seeds in 20 at
#: BLOCK_M=128, reporting alpha 0.80 on cells planted at 0.10 and passing gate 3
#: with it. Rejecting the parallel case costs only block sizes whose ceiling
#: lands within 15% of the ridge, where the answer is undecidable anyway; the
#: four this sweep runs sit at ratios of 2.80, 1.40, 0.70 and 0.35.
PARALLEL_BRANCH_TOLERANCE = 0.15


#: How far ABOVE `ridge x bandwidth` a qualified compute reference is allowed to
#: sit, as a fraction of the roof.
#:
#: A compute branch cannot run faster than the compute roof, so the physical
#: bound is 1.0 and everything past it is slack for the ruler. The slack is
#: taken from the study's own ridge band: three calibrations of the same H200
#: disagreed by 9.9% on the compute term, so a reference within 10% of the roof
#: is inside the calibration's disagreement with itself. Beyond that the arm is
#: claiming to beat its card, which means the roof belongs to a different
#: machine or the FLOP count is wrong -- and both make every roof fraction, every
#: membership decision and every alpha in the report meaningless.
REFERENCE_ROOF_CEILING = 1.10

#: How much SLOWER than the fastest smaller block size, at a matched
#: exactly-full row count, a compute reference may be.
#:
#: At `r = n BM` there is no padding at any block size on the grid, so at a
#: matched `r` every block size does IDENTICAL useful arithmetic, and the larger
#: tile additionally does strictly FEWER weight re-reads (`Q(n)` falls as `n`
#: falls). Under the model the larger tile can therefore only be faster. Slower
#: at all is the model failing; the tolerance covers the timing spread and the
#: coarser wave quantum a big tile leaves at the end of a launch. 1.25 is set
#: above the worst value any sound published reference reaches -- 1.094, over 24
#: references on two cards -- and 4.6x below the smallest corrupt one, 5.723.
REFERENCE_LEVEL_TOLERANCE = 1.25


#: Maximum shared memory ONE thread block may opt into, in bytes, by compute
#: capability. Not the per-SM total: a CTA cannot exceed these however much the
#: SM holds. sm_80 has 164 KiB per SM and lets a block take 163; sm_90 has 228
#: and lets a block take 227. UNKNOWN CAPABILITIES ARE NOT DEFAULTED. A missing
#: entry makes the shared-memory verdict "unknown", never "fits", because a
#: guessed limit that is too generous is exactly the failure this table exists
#: to catch.
SMEM_PER_BLOCK_BYTES: dict[tuple[int, int], int] = {
    (7, 0): 98304,      # V100
    (7, 5): 65536,      # T4
    (8, 0): 166912,     # A100
    (8, 6): 101376,     # A10 / A40 / RTX 30
    (8, 9): 101376,     # L4 / L40S / RTX 40
    (9, 0): 232448,     # H100 / H200
    (10, 0): 232448,    # B200
}

#: A CUDA thread may address 255 registers, on every architecture this study can
#: run on. It is not a tuning parameter and there is no opt-in past it.
MAX_REGISTERS_PER_THREAD = 255


@dataclass(frozen=True)
class TileResources:
    """What one CTA of a given tile setting must be given, against what exists.

    THE CHECK THAT WOULD HAVE STOPPED THE BN=256 ARM BEFORE IT WAS TIMED. Two
    hard limits, computed from the pinned constants alone, so both are knowable
    off-GPU before a single cell runs:

      * SHARED MEMORY. Triton multi-buffers the K loop, so one CTA holds
        `num_stages` copies of the `BM x BK` A tile and the `BK x BN` B tile:
        `num_stages (BM BK + BK BN) b` bytes. At BM=BN=256, BK=64, bf16 and 3
        stages that is 192 KiB, which fits an H200's 227 KiB per-block ceiling
        and does NOT fit an A100's 163 KiB. At 4 stages it is 256 KiB and fits
        neither -- and in the two published num_stages=4 BN=256 arms the
        BLOCK_M=256 ladder is simply ABSENT, which is that cliff.

      * REGISTERS. `tl.dot` accumulates in fp32, so one CTA holds a `BM x BN`
        fp32 accumulator in registers: `BM BN / (32 num_warps)` registers per
        thread. At BM=BN=256 with num_warps=8 that is 256 per thread against a
        hardware maximum of 255, so the accumulator ALONE cannot be held and the
        kernel spills to local memory. This one is card-independent, which is
        why the setting is slow on BOTH cards; the A100 is 43.6x slow and the
        H200 only 3.92x because only the A100 also blows the shared-memory
        limit.

    WHY THIS IS A REFUSAL AND NOT A WARNING. A spilled kernel still returns a
    time, that time still fits a straight line through the origin, and that line
    still qualified as this study's compute reference at 0.2% error. Every tread
    in the arm was then classified against a compute branch 44x too steep. The
    setting has to be refused where it is chosen, not diagnosed afterwards.
    """

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    dtype_bytes: int
    smem_bytes: int
    acc_registers_per_thread: float
    smem_limit_bytes: int | None
    capability: tuple[int, int] | None

    @property
    def registers_fit(self) -> bool:
        return self.acc_registers_per_thread <= MAX_REGISTERS_PER_THREAD

    @property
    def smem_fits(self) -> bool | None:
        """None means the device is unknown, which is not the same as True."""
        if self.smem_limit_bytes is None:
            return None
        return self.smem_bytes <= self.smem_limit_bytes

    @property
    def refusal(self) -> str:
        """Empty when the setting may be timed; otherwise why it may not be."""
        why = []
        if not self.registers_fit:
            why.append(
                f"the {self.block_m}x{self.block_n} fp32 accumulator needs "
                f"{self.acc_registers_per_thread:.0f} registers per thread at "
                f"num_warps={self.num_warps}, against a hardware maximum of "
                f"{MAX_REGISTERS_PER_THREAD}. The accumulator alone does not "
                "fit, so the kernel spills to local memory and its time is not "
                "the time of the tiling this sweep is about")
        if self.smem_fits is False:
            why.append(
                f"one CTA needs {self.smem_bytes / 1024:.0f} KiB of shared "
                f"memory ({self.num_stages} stages x ({self.block_m}x"
                f"{self.block_k} + {self.block_k}x{self.block_n}) x "
                f"{self.dtype_bytes} B) against sm_{self.capability[0]}"
                f"{self.capability[1]}'s {self.smem_limit_bytes / 1024:.0f} KiB "
                "per-block ceiling")
        return "; ".join(why)

    def render(self) -> str:
        smem = f"{self.smem_bytes / 1024:6.0f} KiB"
        limit = ("      ?" if self.smem_limit_bytes is None
                 else f"{self.smem_limit_bytes / 1024:6.0f} KiB")
        return (f"  BLOCK_M={self.block_m:4d}  smem {smem} of {limit}  "
                f"acc {self.acc_registers_per_thread:5.0f} reg/thread of "
                f"{MAX_REGISTERS_PER_THREAD}  "
                + ("REFUSED" if self.refusal else "ok"))


def tile_resources(pinned: dict, block_m: int, dtype_bytes: int,
                   capability: tuple[int, int] | None) -> TileResources:
    """One CTA's shared-memory and accumulator-register bill for a setting.

    Pure arithmetic on the pinned constants, so `--dry-run` on a laptop prints
    the same numbers the pod would refuse on.
    """
    bn = pinned["BLOCK_SIZE_N"]
    bk = pinned["BLOCK_SIZE_K"]
    stages = pinned["num_stages"]
    warps = pinned["num_warps"]
    smem = stages * (block_m * bk + bk * bn) * dtype_bytes
    # fp32 accumulator, one register per element, spread over the CTA's threads.
    acc = block_m * bn / (32.0 * warps)
    return TileResources(
        block_m=block_m, block_n=bn, block_k=bk, num_stages=stages,
        num_warps=warps, dtype_bytes=dtype_bytes, smem_bytes=smem,
        acc_registers_per_thread=acc,
        smem_limit_bytes=SMEM_PER_BLOCK_BYTES.get(capability)
        if capability else None,
        capability=capability)


def parse_capability(text: str) -> tuple[int, int] | None:
    """`"9.0"` -> `(9, 0)`. Empty gives None, which means UNKNOWN, not "fine"."""
    if not text:
        return None
    major, _, minor = text.partition(".")
    return int(major), int(minor or 0)


def resolve_capability(args, *, synthetic: bool) -> tuple[int, int] | None:
    """The attached device's capability, or the one named on the command line.

    A synthetic run has no device, so the shared-memory limit is genuinely
    unknown there and is REPORTED as unknown. The register ceiling is 255 on
    every architecture this can run on, so the check that catches the BN=256
    accumulator still fires on a laptop.
    """
    named = parse_capability(args.capability)
    if named is not None or synthetic:
        return named
    try:
        import torch
        return tuple(torch.cuda.get_device_capability(0))
    except Exception:                                   # noqa: BLE001
        return None


def tile_resource_plan(pinned: dict, block_sizes, dtype_bytes: int,
                       capability: tuple[int, int] | None
                       ) -> tuple[dict[int, TileResources], dict[int, str]]:
    """Every block size's bill, and the refusals among them.

    Returned rather than printed so `main` can print it in the plan, drop the
    refused settings from the sweep, and carry the refusals into the report --
    three places that must agree, from one computation.
    """
    plan = {bm: tile_resources(pinned, bm, dtype_bytes, capability)
            for bm in block_sizes}
    return plan, {bm: r.refusal for bm, r in plan.items() if r.refusal}


@dataclass(frozen=True)
class ComputeReference:
    """The compute branch, and the fused layer's fixed cost, from one ladder.

    TWO THINGS THAT CANNOT BE MEASURED INSIDE A MEMORY-BOUND LADDER, taken from
    a ladder that is compute bound throughout instead.

    The FIXED COST `D` -- router, align, activation, launch -- enters every
    point of a memory-bound ladder identically, so `D` and the memory branch's
    intercept `A` are one number there and `alpha = B / (A + B)` comes out low
    by whatever fraction of the level `D` is. On a ladder that is compute bound
    at every tread, `t(n) = D + C n` has two free parameters and three or more
    points, so `D` separates.

    The COMPUTE SLOPE `C` is what decides membership for every other ladder, and
    `C = 2 BM N_w / peak` is proportional to BLOCK_M with no free parameter, so
    one measurement covers all of them.

    THE ASSUMPTION IS THAT THE LARGEST BLOCK SIZE IS COMPUTE BOUND THROUGHOUT,
    which is a prediction of the model under test, so it is checked rather than
    taken: the ladder has to be straight through a non-negative intercept. When
    it is not, `block_m` is None, alpha downstream becomes a LOWER bound, and
    the membership test falls back to the split search. Degrading to "cannot
    say" is the point; a reference taken from a ladder that was secretly memory
    bound would put `C` at the memory slope and quietly make every block size
    look identifiable.

    SHAPE IS NOT LEVEL, AND THE ORIGINAL QUALIFICATION ONLY TESTED SHAPE. A line
    44x too steep is still perfectly proportional. In the BN=256 arm the
    BLOCK_M=256 reference took 249.765 ms for one tile on the A100 against 5.724
    ms for the identical setting in its BN=64 twin, and qualified at 0.2% mean
    error, because through-origin residual is scale free. Every tread in the arm
    was then classified against that branch, no tread could stand above it, all
    8 cells came out "not identifiable", and the arm printed as a boring null.
    `refusals` carries the LEVEL checks that now have to pass as well, and
    `refused_block_m` names the ladder they rejected, so a reader can tell a
    refused reference from a sweep that never had a candidate.
    """

    block_m: int | None
    overhead_ms: float
    slope_per_tile: float | None
    mean_rel_err: float
    note: str
    #: The ladder the level checks REJECTED, when they rejected one. None both
    #: when a reference qualified and when no candidate existed at all, which is
    #: why `refusals` and not this field is what says a refusal happened.
    refused_block_m: int | None = None
    #: One line per failed level check, empty when the reference is usable.
    refusals: tuple[str, ...] = ()
    #: `C` at the roof over `C` measured: the fraction of `ridge x bandwidth`
    #: the reference ladder actually reached. 1.0 is the roof, and 0.013 is what
    #: the corrupt A100 BN=256 reference reached.
    roof_fraction: float | None = None
    #: `C` scaled to the smallest swept block size, over one full weight read.
    #: At or above 1 the reference makes memory-boundness impossible everywhere.
    vacuity_ratio: float | None = None
    #: Worst `t_reference(r) / min t_smaller(r)` over matched exactly-full row
    #: counts, and how many such comparisons there were. A zero count means the
    #: cross-ladder level check EXAMINED NOTHING and must not read as a pass.
    level_ratio: float | None = None
    level_comparisons: int = 0

    @property
    def refused(self) -> bool:
        """A candidate existed and the level checks threw it out."""
        return bool(self.refusals)

    def slope_for(self, block_m: int) -> float | None:
        """`C` at another block size, by `C ~ BLOCK_M`."""
        if self.slope_per_tile is None or self.block_m is None:
            return None
        return self.slope_per_tile * block_m / self.block_m

    def render(self) -> list[str]:
        """The qualification, as numbers against thresholds.

        Printed whether it passed or failed. The defect this exists to catch was
        invisible precisely because a refused reference and an uninformative
        sweep printed the same blanks downstream.
        """
        out = [f"  compute reference: {self.note}"]
        if self.roof_fraction is not None:
            out.append(
                f"    LEVEL roof fraction   {self.roof_fraction:8.3f}   "
                f"gate <= {REFERENCE_ROOF_CEILING:.2f} of ridge x bandwidth "
                f"(a compute branch cannot beat the roof)")
        if self.vacuity_ratio is not None:
            out.append(
                f"    LEVEL non-vacuity     {self.vacuity_ratio:8.3f}   "
                "gate <  1.00 of one full weight read, scaled to the smallest "
                "block size (at or above, NO tread anywhere can be memory "
                "bound and every alpha is unidentifiable by construction)")
        if self.level_comparisons:
            out.append(
                f"    LEVEL vs smaller BM   {self.level_ratio:8.3f}   "
                f"gate <= {REFERENCE_LEVEL_TOLERANCE:.2f} at matched "
                f"exactly-full rows, over {self.level_comparisons} comparison(s)")
        else:
            out.append(
                "    LEVEL vs smaller BM   NOT CHECKED   no smaller ladder "
                "shares an exactly-full row count with the reference, so this "
                "check examined nothing and is not a pass")
        for why in self.refusals:
            out.append(f"    REFUSED: {why}")
        return out


def _level_checks(cells, block_sizes, bm: int, c: float, *, cfg, ridge: float,
                  bandwidth_gbps: float, b: int, pinned: dict | None,
                  capability) -> tuple[list[str], dict]:
    """Is the candidate's per-tile slope the RIGHT SIZE, not just the right shape.

    Three independent readings of the same number, returned with the numbers so
    the report can print them whether they passed or not.

      1. ROOF CEILING. `C` at the roof is `6 E BM F H / (ridge x bandwidth)`.
         The measured slope cannot be smaller than that: nothing runs faster
         than the compute roof. Above `REFERENCE_ROOF_CEILING` the ruler belongs
         to another machine or the FLOP count is wrong.

      2. NON-VACUITY, and this is the one that catches the BN=256 arm. Scale `C`
         to the SMALLEST swept block size and compare it with ONE FULL WEIGHT
         READ `L = E 3 F H b / bandwidth`. A tread is memory bound when it
         stands above the compute branch, and the highest a memory branch can
         ever sit is `alpha = 1`, one full re-read per tile. So if the scaled
         compute branch is already at or above `L`, NO tread at ANY block size
         can be classified memory bound -- the report's blanks are then a
         property of the reference, not a measurement. Equivalently in roof
         units the floor is `2 BM_min / (b ridge)`, the smallest block size's
         own AI cap at alpha=1 over the ridge. The A100 BN=256 reference sits at
         15.9x this bound and the H200 one at 1.6x; every sound published
         reference sits between 0.31 and 0.52.

      3. AGAINST THE SWEEP'S OWN SMALLER LADDERS. At `r = n BM` nothing is
         padded at any block size, so a matched `r` is identical useful
         arithmetic with strictly fewer weight re-reads for the bigger tile: the
         reference can only be faster. `REFERENCE_LEVEL_TOLERANCE` is the slack.
         This check needs no roof and no calibration at all, which is what makes
         it worth having beside the other two -- and when the reference is the
         smallest ladder swept there is nothing to compare against, which is
         reported as NOT CHECKED rather than silently as a pass.

    A tile setting that cannot physically run is checked here too, because a
    spilled kernel produces a time that is proportional to its tile count and
    therefore sails through the shape test -- which is exactly how 249.765 ms
    became this study's compute branch.
    """
    why: list[str] = []
    nums: dict = {"roof_fraction": None, "vacuity_ratio": None,
                  "level_ratio": None, "level_comparisons": 0}

    if pinned is not None:
        res = tile_resources(pinned, bm, b, capability)
        if res.refusal:
            why.append(
                f"BLOCK_M={bm} cannot run as pinned: {res.refusal}. Its timing "
                "is not a measurement of this tiling and must not become the "
                "compute branch every other ladder is classified against")

    roof_flops = ridge * bandwidth_gbps * 1e9
    if roof_flops > 0:
        c_roof = 1e3 * useful_flops(cfg, cfg.num_experts * bm) / roof_flops
        nums["roof_fraction"] = c_roof / c
        if nums["roof_fraction"] > REFERENCE_ROOF_CEILING:
            why.append(
                f"BLOCK_M={bm} runs at {nums['roof_fraction']:.3f} of "
                f"ridge x bandwidth, past the {REFERENCE_ROOF_CEILING:.2f} "
                "ceiling. A compute branch cannot beat the compute roof, so "
                "either the roof was calibrated on a different machine or the "
                "FLOP count is wrong; either way nothing downstream is scored "
                "against a real ceiling")

    if bandwidth_gbps > 0 and block_sizes:
        bm_min = min(block_sizes)
        full_read_ms = (1e3 * cfg.num_experts * weight_bytes_per_expert(cfg, b)
                        / (bandwidth_gbps * 1e9))
        nums["vacuity_ratio"] = (c * bm_min / bm) / full_read_ms
        if nums["vacuity_ratio"] >= 1.0:
            why.append(
                f"BLOCK_M={bm}'s compute branch scaled to BLOCK_M={bm_min} is "
                f"{nums['vacuity_ratio']:.3f} of one full weight read. A memory "
                "branch cannot exceed one full re-read per tile (alpha <= 1), "
                "so no tread at any block size in this sweep could stand above "
                "this line: every 'not identifiable' below would be a property "
                "of the reference and not a measurement")

    ratios = []
    smaller = [s for s in block_sizes if s < bm]
    if smaller:
        ref_pts = dict(ladder_points(cells, bm))
        others = {s: dict(ladder_points(cells, s)) for s in smaller}
        for n, t_ref in ref_pts.items():
            rows = n * bm
            peers = [pts[rows // s] for s, pts in others.items()
                     if rows % s == 0 and (rows // s) in pts]
            if peers and t_ref > 0:
                ratios.append(t_ref / min(peers))
    if ratios:
        nums["level_ratio"] = max(ratios)
        nums["level_comparisons"] = len(ratios)
        if nums["level_ratio"] > REFERENCE_LEVEL_TOLERANCE:
            why.append(
                f"BLOCK_M={bm} is {nums['level_ratio']:.3f}x slower than the "
                f"best smaller block size at a matched exactly-full row count, "
                f"past the {REFERENCE_LEVEL_TOLERANCE:.2f} tolerance, over "
                f"{len(ratios)} comparison(s). At a matched full stack the "
                "bigger tile does the same arithmetic and strictly fewer weight "
                "re-reads, so under the model it cannot be slower at all")
    return why, nums


#: The machine-readable half of "why is this cell blank". Kept as fixed tokens
#: rather than prose so a downstream table can branch on them; the prose is in
#: `LadderFit.basis` and in `ComputeReference.render`.
NOT_IDENTIFIABLE_REFERENCE_REFUSED = "reference_refused"
NOT_IDENTIFIABLE_NO_REFERENCE = "no_compute_reference"
NOT_IDENTIFIABLE_TOO_FEW_TREADS = "too_few_memory_treads"
NOT_IDENTIFIABLE_IS_REFERENCE = "is_the_reference_ladder"


def _why_not_identifiable(fit, ref) -> str:
    """Empty when the fit IS identifiable, else the reason it is not.

    THE CONFLATION THIS EXISTS TO BREAK. All 8 cells of the BN=256 arm printed
    as blanks under a caption reading "fewer than 3 memory-bound treads". The
    treads were there; the compute reference was 44x too steep, so nothing could
    stand above it. Those two states have to be told apart at the row level,
    because the arm's whole defect was invisible while they were not.
    """
    if fit.memory_points >= MIN_MEMORY_TREADS and fit.alpha is not None:
        return ""
    if ref is not None and ref.refused:
        return NOT_IDENTIFIABLE_REFERENCE_REFUSED
    if ref is not None and ref.block_m == fit.block_m:
        return NOT_IDENTIFIABLE_IS_REFERENCE
    if ref is None or ref.block_m is None:
        return NOT_IDENTIFIABLE_NO_REFERENCE
    return NOT_IDENTIFIABLE_TOO_FEW_TREADS


def compute_reference(cells, block_sizes, max_err: float = 0.05, *,
                      cfg, ridge: float, bandwidth_gbps: float, b: int,
                      pinned: dict | None = None, capability=None
                      ) -> ComputeReference:
    """Qualify the largest ladder as a compute branch, or decline.

    THE QUALIFICATION IS PROPORTIONALITY, not the sign of an extrapolated
    intercept. `t = C n` through the origin is one parameter and stays put under
    noise; `t = D + C n` is two, and on the four treads BLOCK_M=256 gets at
    r_max=1024 a 3% timing spread pushed `D` to -0.097 ms and the slope 7% high.
    Rejecting on that sign threw the reference away exactly when the data was
    noisy, which sent every ladder to the split search and let BLOCK_M=256's own
    straight line be read as a memory branch with alpha 1.097.

    So `C` comes from the through-origin fit and the residual of THAT decides
    whether the ladder is a compute branch. It discriminates: a memory-bound
    ladder `A + B n` with a real intercept cannot be described by a line through
    the origin and misses by 11% on this grid, well past `max_err`.

    `C` then absorbs a little of the fixed cost, which biases the compute branch
    slightly HIGH and so makes membership slightly conservative: a tread has to
    clear a marginally higher line to be called memory bound. That is the
    direction to be wrong in, since every failure this file defends against is a
    tread wrongly called memory bound.

    PROPORTIONALITY IS NECESSARY AND NOWHERE NEAR SUFFICIENT, and believing
    otherwise cost this study 8 published cells. It is a test of SHAPE and a
    line 44x too steep has the right shape. `_level_checks` is the test of
    LEVEL, it runs after the shape test on the same candidate, and a candidate
    that fails it is REFUSED rather than warned about, because the reference is
    what every other ladder in the report is classified against: a bad one does
    not add noise, it decides the answer.

    A REFUSAL DOES NOT FALL THROUGH TO THE NEXT BLOCK SIZE. The next-largest
    ladder is measured under the same pinned constants on the same card, so the
    thing that broke the level is very likely still there; taking the runner-up
    would replace a loud refusal with a quiet, differently-wrong reference.
    """
    for bm in sorted(block_sizes, reverse=True):
        pts = ladder_points(cells, bm)
        if len(pts) < 3:
            continue
        xs = [float(n) for n, _ in pts]
        ys = [ms for _, ms in pts]
        c = _through_origin(xs, ys)
        err = statistics.fmean(abs(c * x - y) / y
                               for x, y in zip(xs, ys, strict=True))
        if c <= 0 or err > max_err:
            return ComputeReference(
                None, 0.0, None, err,
                f"BLOCK_M={bm} ladder is not proportional to its tile count "
                f"({err:.1%} mean error against a line through the origin), so "
                "it is not compute bound throughout and cannot provide a "
                "compute branch. Membership falls back to a split search and NO "
                "alpha may decide a verdict")
        why, nums = _level_checks(
            cells, block_sizes, bm, c, cfg=cfg, ridge=ridge,
            bandwidth_gbps=bandwidth_gbps, b=b, pinned=pinned,
            capability=capability)
        if why:
            return ComputeReference(
                None, 0.0, None, err,
                f"BLOCK_M={bm} ladder is proportional to {err:.1%} but its "
                "LEVEL is wrong, so it is REFUSED as a compute branch. "
                "Membership falls back to a split search, NO alpha may decide a "
                "verdict, and every 'not identifiable' in this report is "
                "CAUSED BY THIS REFUSAL rather than by a sweep that lacked "
                "treads",
                refused_block_m=bm, refusals=tuple(why), **nums)
        # Reported, and used only for `alpha_upper` and to shift the compute
        # branch. Clamped at zero because a negative fixed cost is a fitting
        # artefact and subtracting one would inflate every alpha.
        intercept, _ = _line(xs, ys)
        return ComputeReference(
            bm, max(0.0, intercept), c, err,
            f"BLOCK_M={bm} ladder, {len(pts)} treads, proportional to "
            f"{err:.1%}: compute branch {c:.4f} ms per tile, fixed cost "
            f"{max(0.0, intercept):.4f} ms",
            **nums)
    return ComputeReference(
        None, 0.0, None, math.inf,
        "no ladder had the 3 treads needed to qualify a compute branch. "
        "Membership falls back to a split search and NO alpha may decide a "
        "verdict")


def fit_ladder(points, block_m: int, ref: ComputeReference | None = None,
               margin: float = MEMORY_BRANCH_MARGIN) -> LadderFit:
    """Split the ladder into a memory branch and a compute branch.

    Membership comes from the reference compute branch when there is one: the
    memory branch is the leading run of treads standing more than `margin`
    above `C n`. Memory-boundness is a PREFIX property -- `Q` grows by `alpha`
    per tile and the compute branch by 1, so once compute is on top it stays
    there -- which is why the run is taken from the bottom and not as a
    scattered subset.

    Without a reference (`ref` absent or unusable) it falls back to searching
    every split for the smallest residual, which is what a reader would do by
    eye and carries the failure mode `LadderFit`'s docstring describes. The
    `basis` field says which happened, because the two answers are not
    interchangeable and one of them can invent an alpha.

    THE MEMORY BRANCH IS FITTED ON RAW TIMES, fixed cost included, and that is
    deliberate. Subtracting an extrapolated fixed cost before fitting hands its
    error straight to alpha: a 4-tread reference ladder under 1% timing spread
    put the extrapolated cost anywhere from 0.03 to 0.16 ms, which moved alpha
    from 0.56 to 0.70 on data planted at 0.558. Leaving it in makes
    `LadderFit.alpha` a LOWER BOUND with a known sign -- the fixed cost inflates
    the denominator and nothing else -- and every gate is scored on the bound.
    `alpha_upper` carries the other end for a reader who wants the range.
    """
    overhead = ref.overhead_ms if ref else 0.0
    pts = [(n, ms) for n, ms in points if ms > 0]
    if not pts:
        return LadderFit(block_m, tuple(points), 0, None, None, None, None,
                         math.inf, overhead, "no usable treads")
    xs = [float(n) for n, _ in pts]
    ys = [ms for _, ms in pts]
    c_ref = ref.slope_for(block_m) if ref else None

    if ref is not None and ref.block_m == block_m:
        # The reference ladder has NO memory branch, by the assumption that
        # qualified it as the reference. Letting it test its own points against
        # its own fitted line lets timing noise push the low treads above it:
        # at 1% spread BLOCK_M=256 reported two memory-bound treads and an alpha
        # of 0.96, which then won the "largest identifiable block size" contest
        # and turned gate 3 into a PASS on data planted at 0.10.
        k = 0
        basis = "the reference ladder itself: compute bound at every tread"
    elif c_ref:
        k = 0
        for x, y in zip(xs, ys, strict=True):
            if y <= overhead + c_ref * x * (1.0 + margin):
                break
            k += 1
        basis = (f"membership from the compute branch scaled off "
                 f"BLOCK_M={ref.block_m}")
    else:
        # THE TWO WAYS TO HAVE NO REFERENCE ARE NOT THE SAME STATE and printing
        # them the same way is what hid the BN=256 corruption: 8 cells read as
        # "not identifiable" under a caption blaming tread count, when the
        # actual cause was a compute branch 44x too steep. Say which happened.
        k = _best_split(xs, ys, overhead)
        basis = ("split search: the compute reference was REFUSED at "
                 f"BLOCK_M={ref.refused_block_m} on its level, so this ladder "
                 "is unidentifiable BECAUSE OF THE REFERENCE and not for want "
                 "of treads"
                 if ref is not None and ref.refused
                 else "split search: no usable compute reference")

    a = b = None
    if k >= 2:
        a, b = _line(xs[:k], ys[:k])
    c_own = (_through_origin(xs[k:], [y - overhead for y in ys[k:]])
             if k < len(pts) else None)
    c_eff = c_own if c_own else c_ref
    if b is not None and c_eff and abs(b / c_eff - 1.0) <= PARALLEL_BRANCH_TOLERANCE:
        # A memory branch parallel to the compute branch is not a second
        # mechanism. `B / C = ridge / ai_cap`, so this says the ceiling sits on
        # the ridge -- or, far more often, that the fit ran the prefix into the
        # compute branch and is about to report that branch's slope as alpha.
        a, b = None, None
        k = 0
        basis += (f"; memory branch DISCARDED, its slope was within "
                  f"{PARALLEL_BRANCH_TOLERANCE:.0%} of the compute branch and "
                  "the two are then the same line")
    err = _max_affine_error(xs, ys, a, b, c_eff, overhead)
    return LadderFit(block_m, tuple(points), k, a, b, c_own, c_ref, err,
                     overhead, basis)


def _best_split(xs, ys, overhead: float = 0.0) -> int:
    """Fallback membership: the split with the smallest max-affine residual."""
    best = None
    for k in range(0, len(xs) + 1):
        a = b = None
        if k >= 2:
            a, b = _line(xs[:k], ys[:k])
        c = (_through_origin(xs[k:], [y - overhead for y in ys[k:]])
             if k < len(xs) else None)
        if b is None and c is None:
            continue
        score = _max_affine_error(xs, ys, a, b, c, overhead)
        if best is None or score < best[0]:
            best = (score, k)
    return best[1] if best else 0


def _max_affine_error(xs, ys, a, b, c, overhead: float = 0.0) -> float:
    """Mean relative error of `max(a + b n, D + c n)` against the ladder.

    The fixed cost is added to the COMPUTE branch only. The memory branch is
    fitted on raw times and already carries it inside its intercept, which is
    what makes `LadderFit.alpha` a lower bound rather than a noisy point.
    """
    errs = []
    for x, y in zip(xs, ys, strict=True):
        branches = []
        if b is not None:
            branches.append(a + b * x)
        if c:
            branches.append(overhead + c * x)
        if not branches or y <= 0:
            return math.inf
        errs.append(abs(max(branches) - y) / y)
    return statistics.fmean(errs) if errs else math.inf


def activation_slope_ms(cfg, block_m: int, bandwidth_gbps: float) -> float:
    """The part of the per-tile slope that is activations, not weight re-reads.

    An extra tile carries `BM` more rows, and those rows move
    `activation_bytes_per_row` of x_perm / h_up / h_act / y_perm each. That is
    affine in `n` exactly like the re-read term, so it lands inside `B` and
    inflates the fitted alpha. Subtracting it is what the report's
    `alpha-corrected` column is, and gate 3 is scored on that column.
    """
    per_tile = cfg.num_experts * block_m * activation_bytes_per_row(cfg)
    return 1e3 * per_tile / (bandwidth_gbps * 1e9)


# --------------------------------------------------------------------------
# Gates.
# --------------------------------------------------------------------------

PASS, FAIL, UNDECIDED = "PASS", "FAIL", "UNDECIDED"

#: A gate about the INSTRUMENT: whether the run is readable at all. Gate 0 is
#: the only one, and a non-PASS there voids every claim gate below it.
VALIDITY = "VALIDITY"
#: A gate about the WORLD: it can be FAILed by hardware without the run being
#: broken, and a FAIL is a result rather than an error.
CLAIM = "CLAIM"

#: Where the number in `measured` came from. The distinction exists because
#: gate 3 printed a "measured" crossing ratio for 22 published reports that was
#: an algebraic restatement of a fitted alpha imported from a different
#: BLOCK_M, and the serialized JSON kept only {number, claim, verdict, measured,
#: gate} -- so nothing in the published file said the ratio was never observed.
OBSERVED = "OBSERVED"       # read off a timing in this run
DERIVED = "DERIVED"         # computed from a fit over this run's own timings
IMPORTED = "IMPORTED"       # computed from a fit over a DIFFERENT setting


@dataclass
class Gate:
    number: int
    claim: str
    verdict: str
    measured: str
    threshold: str
    lines: list[str] = field(default_factory=list)
    #: VALIDITY or CLAIM. Defaults to CLAIM because four of the five are.
    kind: str = CLAIM
    #: OBSERVED / DERIVED / IMPORTED, for the number in `measured`.
    basis: str = OBSERVED
    #: Machine-readable provenance, serialized beside the verdict. Anything a
    #: reader would need to tell an observation from a restatement goes here,
    #: because the printed detail lines do not survive into report.json.
    provenance: dict = field(default_factory=dict)

    def render(self) -> list[str]:
        out = [f"GATE {self.number}  {self.verdict:9s} [{self.kind}/"
               f"{self.basis}] {self.claim}",
               f"          measured {self.measured}   gate {self.threshold}"]
        out += [f"          {line}" for line in self.lines]
        return out


def gate_0_override(compiles: dict[int, int], executed: dict[int, int],
                    block_sizes) -> Gate:
    """Did `override_config` actually change the kernel.

    THE GATE THAT DECIDES WHETHER THE OTHER FOUR MEAN ANYTHING. If the override
    silently failed, all four settings ran one kernel, every difference is zero,
    and the report reads as a tidy null result. A setting that changed the tile
    constants MUST have compiled a new Triton specialisation, so counting the
    artefacts that appear while a setting runs is a direct assay. Zero means
    either the override did nothing or the cache served a previous run, and both
    are fatal in the same way.

    A RESUMED SETTING IS NOT A FAILED ONE. A run that finds every cell already
    in `cells.csv` executes nothing and therefore compiles nothing, which is not
    evidence that the override is broken -- it is the absence of the experiment
    that would have tested it. Those settings are named and the gate goes
    UNDECIDED, because the assay belongs to the session that ran the cells and
    this session cannot inherit it.
    """
    ran = [bm for bm in block_sizes if executed.get(bm, 0) > 0]
    resumed = [bm for bm in block_sizes if executed.get(bm, 0) <= 0]
    missing = [bm for bm in ran if compiles.get(bm, 0) <= 0]
    counts = ", ".join(f"BM={bm}:{compiles.get(bm, 0)}" for bm in block_sizes)
    if missing:
        return Gate(
            0, "override_config changed the kernel at every setting", FAIL,
            f"fresh Triton artefacts per setting: {counts}", ">= 1 per setting",
            [f"BLOCK_SIZE_M {missing} ran cells and compiled nothing new.",
             "Either override_config did not take effect, or TRITON_CACHE_DIR "
             "was warm.",
             "Every gate below is then a comparison of one kernel with itself. "
             "Do not read them."], kind=VALIDITY)
    if resumed:
        return Gate(
            0, "override_config changed the kernel at every setting", UNDECIDED,
            f"fresh Triton artefacts per setting: {counts}", ">= 1 per setting",
            [f"BLOCK_SIZE_M {resumed} ran no cells this session: every one was "
             "already in cells.csv.",
             "The compile assay belongs to the session that measured them and "
             "cannot be inherited. Delete cells.csv and re-run to assay it "
             "again, or read the gates knowing this one was not repeated."],
            kind=VALIDITY)
    return Gate(0, "override_config changed the kernel at every setting", PASS,
                f"fresh Triton artefacts per setting: {counts}",
                ">= 1 per setting", [], kind=VALIDITY)


def gate_1_steps(cells, cfg, preds, *, alpha: float, ridge: float,
                 bandwidth_gbps: float, b: int, noise: float) -> Gate:
    """Do the time steps land at `T = n BLOCK_M E / k`.

    Read as a jump against a tread, not as a slope. A slope over the 4-token
    interval that separates a full tile stack from the next tread would divide a
    0.5% timing difference by `log(1.004)` and report 1.2, so the log-log
    detector `crossing.py` uses is the wrong instrument at this resolution. The
    quantity here is the RATIO across the boundary against the ratio just below
    it, both over intervals of the same width, which needs no logarithm.
    """
    rows = {}
    for c in cells:
        if c.status == "ok" and c.ms_p50 > 0:
            rows.setdefault(c.block_m, {})[c.rows_per_expert] = c
    detail = []
    jumps, treads, predicted = [], [], []
    misplaced = 0
    for bm in sorted(rows):
        grid = sorted(rows[bm])
        for n in range(1, 64):
            edge = float(n * bm)
            if edge > max(grid):
                break
            at = rows[bm].get(edge)
            if at is None:
                continue
            below = max((r for r in grid if (n - 1) * bm < r < edge), default=None)
            above = min((r for r in grid if r > edge), default=None)
            if below is None or above is None:
                continue
            jump = rows[bm][above].ms_p50 / at.ms_p50 - 1.0
            tread = at.ms_p50 / rows[bm][below].ms_p50 - 1.0
            pj = (model_ms(cfg, above, bm, alpha=alpha, ridge=ridge,
                           bandwidth_gbps=bandwidth_gbps, b=b)
                  / model_ms(cfg, edge, bm, alpha=alpha, ridge=ridge,
                             bandwidth_gbps=bandwidth_gbps, b=b) - 1.0)
            jumps.append(jump)
            treads.append(tread)
            predicted.append(pj)
            if jump <= tread:
                misplaced += 1
            detail.append(
                f"BM={bm:3d} n={n:2d} T={tokens_for_rows(cfg, int(edge)):6d} "
                f"r={edge:6.0f}  tread {tread:+7.2%}  STEP {jump:+7.2%}  "
                f"(model {pj:+7.2%})  waves {at.waves_up:6.1f}/"
                f"{rows[bm][above].waves_up:6.1f}")
    if len(jumps) < 3:
        return Gate(1, "time steps at T = n x BLOCK_M x E/k", UNDECIDED,
                    f"{len(jumps)} bracketed boundaries", ">= 3",
                    ["No boundary had a point below, at and above it. Raise "
                     "--step-probes or lower --row-step."])
    measured = statistics.median(j - t for j, t in zip(jumps, treads, strict=True))
    gate = 0.5 * statistics.median(predicted)
    verdict = PASS if (measured >= gate and measured > 3 * noise) else FAIL
    return Gate(
        1, "time steps at T = n x BLOCK_M x E/k", verdict,
        f"median step minus tread {measured:+.2%}",
        f">= {gate:+.2%} (half the model's step) and > 3x noise ({3 * noise:.2%})",
        [f"{len(jumps)} boundaries bracketed, {misplaced} where the tread moved "
         f"at least as much as the step",
         f"median step {statistics.median(jumps):+.2%}, median tread "
         f"{statistics.median(treads):+.2%}, model step "
         f"{statistics.median(predicted):+.2%}",
         "waves are printed on both sides of every step so occupancy can be "
         "ruled out as the cause:"] + detail)


def gate_2_direction(cells, cfg, *, alpha: float, retracted: float, ridge: float,
                     bandwidth_gbps: float, b: int, block_sizes) -> Gate:
    """Does time move UP or DOWN with bigger BLOCK_M in the MULTI-TILE regime.

    Compared only at rows-per-expert that are an exact multiple of EVERY block
    size, where all four run zero-padding tile stacks. Anywhere else the
    comparison carries a padding difference of up to 8x and stops being about
    traffic.

    Both worlds predict a direction and they predict different magnitudes: at
    alpha=0.558 the smallest tile pays 18 weight reads where the largest pays a
    compute-bound 6.4, and at alpha=0.10 the smallest tile is ALSO compute bound
    by then and the ratio is 1.0. So this gate is not only a direction, it is a
    second, independent test of the same disagreement.
    """
    lo, hi = min(block_sizes), max(block_sizes)
    step = math.lcm(*block_sizes)
    by = {}
    for c in cells:
        if c.status == "ok" and c.aligned and c.ms_p50 > 0:
            by.setdefault(c.rows_per_expert, {})[c.block_m] = c
    common = sorted(r for r, d in by.items()
                    if r % step == 0 and all(bm in d for bm in block_sizes))
    if not common:
        return Gate(2, "time falls with BLOCK_M at equal rows per expert",
                    UNDECIDED, "no rows-per-expert common to every block size",
                    f"a multiple of {step}",
                    ["Raise --r-max to at least one multiple of the largest "
                     "block size."])
    detail = ["rows/expert  " + "  ".join(f"BM={bm:<9d}" for bm in block_sizes)
              + "   ms(lo)/ms(hi)"]
    for r in common:
        row = "  ".join(f"{by[r][bm].ms_p50:9.4f} ms" for bm in block_sizes)
        detail.append(f"{r:11.0f}  {row}   "
                      f"{by[r][lo].ms_p50 / by[r][hi].ms_p50:.3f}x")
    top = common[-1]
    ratio = by[top][lo].ms_p50 / by[top][hi].ms_p50
    def predicted(a):
        return (model_ms(cfg, top, lo, alpha=a, ridge=ridge,
                         bandwidth_gbps=bandwidth_gbps, b=b)
                / model_ms(cfg, top, hi, alpha=a, ridge=ridge,
                           bandwidth_gbps=bandwidth_gbps, b=b))
    direction = ("DOWN with BLOCK_M, as traffic requires" if ratio > 1.01 else
                 "UP with BLOCK_M, as occupancy would" if ratio < 0.99 else
                 "not at all with BLOCK_M, which is what a block size already "
                 "compute bound at the smallest tile looks like")
    return Gate(
        2, "time falls with BLOCK_M at equal rows per expert (traffic, not occupancy)",
        PASS if ratio >= GATE2_RATIO else FAIL,
        f"ms(BM={lo}) / ms(BM={hi}) = {ratio:.3f}x at {top:.0f} rows per expert",
        f">= {GATE2_RATIO:.2f}x",
        [f"time moves {direction}",
         f"model at alpha={alpha:.3f}: {predicted(alpha):.3f}x   "
         f"model at the retracted alpha={retracted:.2f}: {predicted(retracted):.3f}x",
         f"a FAIL here is the interesting answer: it says the padded arithmetic "
         f"or the lost occupancy outweighs {math.ceil(top / lo)} weight re-reads"]
        + detail)


def gate_3_alpha_discriminates(fits, preds_lo, preds_hi, cfg, *, lo: int,
                               hi: int, alpha_source: str,
                               alpha_hat: float | None,
                               alpha_source_bm: int | None,
                               ridge_band: tuple[float, float]) -> Gate:
    """Is the fitted re-read fraction above 0.33, the midpoint of the two worlds.

    WHAT THIS GATE USED TO CLAIM, AND WHY THAT WAS WITHDRAWN. It was phrased as
    "the BLOCK_M=128 crossing sits Q above the BLOCK_M=256 one" and its
    `measured` field was `1.0 + alpha_hat`. That is not a measurement of a
    crossing ratio; it is the model's identity `R_cross(lo)/R_cross(hi) = 1 +
    alpha` evaluated at a fitted alpha, and the alpha is usually IMPORTED from a
    different BLOCK_M because tread 2 at 128 is compute bound under both worlds
    and so carries no information about re-read. Three things make the old
    phrasing indefensible rather than merely loose:

      * it is ALGEBRA, not evidence. Recomputed over every published report that
        carries both fields, `measured == 1 + alpha_measured` in 22 of 22 and
        differs in 0. The gate restated its own input.
      * the identity needs the 128 crossing to land in tread 2. At the A100's
        own alphas the model predicts NO CROSSING AT ALL for 3 of 6 arms, so
        for those the ratio does not exist and 1 + alpha stands for nothing.
      * NO CROSSING HAS EVER BEEN OBSERVED. Across all of `results/published`
        the ladder field `crosses` is False 41 times, null 61 times and True
        zero times. A gate must not be phrased as though one was seen.

    So the claim is now what the arithmetic supports: the fitted alpha exceeds
    0.33. That is still the discrimination the sweep was built for -- 0.33 is
    the midpoint between the refit 0.558 and the retracted 0.10, so a PASS is
    incompatible with the retracted world and a FAIL is incompatible with the
    refit one -- and it no longer borrows the authority of an observation.

    The crossing ratio the old gate printed is still computed and still
    reported, under `provenance["restated_crossing_ratio"]`, so an old verdict
    can be recomputed from a new report. It is labelled as a restatement.
    """
    pred_lo = crossing_ratio(preds_lo, lo, hi)
    pred_hi = crossing_ratio(preds_hi, lo, hi)
    retracted_ratio = crossing_ratio(
        predictions((lo, hi), RETRACTED_ALPHA, ridge_band[0]), lo, hi)
    band = []
    for ridge_end, value in zip(ridge_band, (pred_lo, pred_hi), strict=True):
        band.append(f"ridge {ridge_end:.1f}: {value:.3f}x" if value
                    else f"ridge {ridge_end:.1f}: no crossing")
    # An ABSENCE stated from the data rather than from the model, because the
    # gate's old wording implied a crossing had been watched. `LadderFit.crosses`
    # is `C > B` read off two FITTED slopes: whether this ladder's own numbers
    # say a crossing exists at all. It is the weakest form of the claim -- it
    # does not require the crossing to have been reached -- and even so it has
    # never once been true: across all of results/published it is False 41
    # times, null 61 times and True zero times.
    crossed = sorted(bm for bm, f in fits.items() if f.crosses)
    undecided_cross = sorted(bm for bm, f in fits.items() if f.crosses is None)
    observed = (
        f"ladders whose own fitted slopes imply a crossing exists (C > B): "
        f"{crossed}; slopes missing at {undecided_cross}" if crossed else
        "NO ladder's fitted slopes imply a crossing exists (C > B is met "
        f"nowhere; slopes missing at {undecided_cross}), so nothing here is a "
        "crossing measurement and no crossing has been observed")
    claim = f"the fitted re-read fraction alpha exceeds {GATE3_ALPHA_DISCRIMINATOR:.2f}"
    threshold = (f"> {GATE3_ALPHA_DISCRIMINATOR:.2f} "
                 f"(midpoint of the refit {ALPHA} and the retracted "
                 f"{RETRACTED_ALPHA})")
    provenance = {
        "tests": "alpha_hat > threshold",
        "not_an_observed_crossing": True,
        "observed_crossing_ratio": None,
        "ladders_whose_slopes_imply_a_crossing": crossed,
        "ladders_with_a_missing_slope": undecided_cross,
        "alpha_hat": alpha_hat,
        "alpha_source": alpha_source,
        "alpha_source_block_m": alpha_source_bm,
        "imported_from_another_block_m": (
            None if alpha_source_bm is None else alpha_source_bm != lo),
        "restated_crossing_ratio": (
            None if alpha_hat is None else 1.0 + alpha_hat),
        "restated_crossing_ratio_note": (
            "1 + alpha_hat, the model's identity evaluated at the fitted alpha. "
            "It is what the retired ratio gate printed as 'measured'. It is not "
            "a measurement and it is only the crossing ratio at all when the "
            f"BLOCK_M={lo} crossing lands in tread 2."),
        "model_ratio_ridge_lo": pred_lo,
        "model_ratio_ridge_hi": pred_hi,
        "model_ratio_retracted": retracted_ratio,
        "ridge_band": list(ridge_band),
    }
    if alpha_hat is None:
        return Gate(3, claim, UNDECIDED,
                    "alpha not identifiable at any block size", threshold,
                    ["No ladder had two memory-bound treads, so no block size "
                     "measured the re-read fraction.",
                     "Lower --r-max is not the fix; a block size whose cap is "
                     "below the ridge is. 32 and 64 are those.",
                     observed,
                     "model's crossing ratio, for reference only: "
                     + "   ".join(band)],
                    basis=DERIVED, provenance=provenance)

    verdict = PASS if alpha_hat > GATE3_ALPHA_DISCRIMINATOR else FAIL
    own = fits.get(lo)
    imported = alpha_source_bm is not None and alpha_source_bm != lo
    if own is not None and own.memory_points >= MIN_MEMORY_TREADS and not imported:
        provenance_line = (
            f"alpha came from BLOCK_M={lo} itself, over {own.memory_points} "
            f"treads. Tread 2 there sits within a few percent of the compute "
            f"branch -- 256 padded rows against a {preds_lo[lo].crossing_rows:.0f} "
            "row crossing -- so it only just qualified as memory bound, and "
            "this alpha is the most fragile number in the report. Compare it "
            "with the ladders below, which have many more treads.")
    else:
        provenance_line = (
            f"alpha is IMPORTED from BLOCK_M={alpha_source_bm}: it is not "
            f"identifiable at BLOCK_M={lo} on this sweep "
            f"({own.memory_points if own else 0} tread(s) stand above the "
            f"compute branch, and a verdict needs {MIN_MEMORY_TREADS}). "
            "ALPHA_BY_BLOCK_M puts the cost of that import at about +/-25%, "
            "which is smaller than the gap being discriminated.")
    lines = [
        f"measured is the FITTED alpha ({alpha_hat:.3f}), {alpha_source}. It "
        "is not a crossing ratio and no crossing was measured to produce it.",
        observed,
        f"the same alpha restates the model's crossing ratio as "
        f"1 + {alpha_hat:.3f} = {1.0 + alpha_hat:.3f}x, which is what the "
        "retired ratio gate printed as its 'measured' value. That number is "
        "algebra over this line, not a second observation.",
        "model's crossing ratio: " + "   ".join(band),
        f"the retracted alpha={RETRACTED_ALPHA} would put that ratio at "
        + (f"{retracted_ratio:.3f}x" if retracted_ratio else "no crossing"),
        provenance_line]
    for bm, fit in sorted(fits.items()):
        if fit.alpha is not None:
            lines.append(f"  BLOCK_M={bm:3d}  alpha {fit.alpha:.3f} from "
                         f"{fit.memory_points} memory-bound treads, fit error "
                         f"{fit.mean_rel_err:.2%}")
        else:
            lines.append(f"  BLOCK_M={bm:3d}  alpha not identifiable "
                         f"({fit.memory_points} memory-bound tread(s))")
    return Gate(3, claim, verdict, f"alpha {alpha_hat:.3f}", threshold, lines,
                basis=IMPORTED if imported else DERIVED, provenance=provenance)


@dataclass(frozen=True)
class Bracketing:
    """Was the sweep long enough for "no crossing" to be evidence.

    An unbracketed sweep reporting an absence is worthless, so the three things
    that make the absence mean something are computed and printed rather than
    implied:

      * a POSITIVE CONTROL, some other block size that did cross inside the same
        grid, which proves the instrument can see a crossing;
      * a HORIZON, the largest rows-per-expert at which any competing hypothesis
        places this block size's crossing, times a safety factor;
      * SATURATION, the fraction of the modelled AI ceiling the last tread
        reached, since a curve still climbing has not finished rising.

    THE CONTROL IS SCORED AGAINST `ridge x bandwidth`, NOT AGAINST THE RUN'S OWN
    PLATEAU, and that is the change that makes it a control at all. Against the
    plateau -- the arm's own maximum -- something always reaches 1.00 by
    construction, because the plateau IS the maximum, so the control could never
    fail and the check examined nothing. Against the absolute roof it is a real
    question, and in all 26 published reports the answer is no: the plateau is
    46.5-75.6% of that card's own `ridge x bandwidth`, so nothing in any of
    those sweeps reached a compute roof and none of them was ever entitled to
    read an absence at BLOCK_M=64 as evidence about BLOCK_M=64.
    """

    reached_rows: float
    reached_tiles: int
    horizon_rows: float
    positive_control: int | None
    saturation: float
    last_gain: float
    #: The best roof fraction any OTHER block size reached, in units of
    #: `ridge x bandwidth`. Recorded even when it fails the control threshold,
    #: because "the best anything managed was 0.53" is the diagnosis.
    best_other_roof_fraction: float = 0.0
    #: `ridge x bandwidth` in TFLOP/s, the denominator every fraction here uses.
    roof_tflops: float = 0.0

    @property
    def sufficient(self) -> bool:
        return (self.reached_rows >= self.horizon_rows
                and self.positive_control is not None
                and self.saturation >= 0.90)

    def lines(self) -> list[str]:
        control = (f"BLOCK_M={self.positive_control} reached "
                   f"{self.best_other_roof_fraction:.2f} of ridge x bandwidth "
                   "inside this same grid"
                   if self.positive_control is not None
                   else "NOTHING reached the compute roof in this grid (best "
                        f"other block size {self.best_other_roof_fraction:.2f} "
                        f"of ridge x bandwidth = {self.roof_tflops:.0f} "
                        "TFLOP/s), so the sweep never demonstrated it can "
                        "detect a crossing at all")
        return [
            f"swept to {self.reached_rows:.0f} rows per expert "
            f"({self.reached_tiles} M-tiles)",
            f"horizon {self.horizon_rows:.0f} rows: 2x the crossing the "
            f"retracted alpha={RETRACTED_ALPHA} predicts for this block size",
            f"positive control: {control}",
            f"modelled AI at the last tread is {self.saturation:.1%} of the "
            f"ceiling, and the last tread gained {self.last_gain:.1%} of "
            "throughput over the one before it"]


def bracketing(cells, block_m: int, alpha: float, ridge: float, b: int,
               fits, roof_tflops: float, safety: float = 2.0) -> Bracketing:
    """`roof_tflops` is `ridge x bandwidth`, NOT the run's own plateau.

    Passing the plateau here is what made the positive control vacuous: the
    plateau is the maximum over the same cells the control is read from, so
    some block size always scores 1.00 against it.
    """
    pts = ladder_points(cells, block_m)
    reached_tiles = pts[-1][0] if pts else 0
    reached = float(reached_tiles * block_m)
    retracted = predict_tile(block_m, RETRACTED_ALPHA, ridge, b)
    horizon = safety * (retracted.crossing_rows or 0.0)
    # The control is MEASURED, not fitted: some other block size actually got
    # to the roof inside this same grid. A fitted `crosses` would make the
    # control depend on the same branch assignment gate 4 is arguing about, and
    # a fit that lost its memory branch to noise would silently remove the
    # control and turn a real absence into UNDECIDED.
    control = None
    best_other = 0.0
    for bm in sorted({c.block_m for c in cells}):
        if bm == block_m:
            continue
        tp_other = _throughput_ladder(cells, bm, roof_tflops)
        if not tp_other:
            continue
        best = max(v for _, v in tp_other)
        best_other = max(best_other, best)
        if best >= COMPUTE_BOUND_FRACTION:
            control = bm
    cap = ai_cap(block_m, alpha, b)
    ai = (2.0 * reached / b) / q_of_tiles(max(reached_tiles, 1), alpha) if reached else 0.0
    gain = 0.0
    tp = _throughput_ladder(cells, block_m, roof_tflops)
    if len(tp) >= 2 and tp[-2][1] > 0:
        gain = tp[-1][1] / tp[-2][1] - 1.0
    return Bracketing(reached, reached_tiles, horizon, control,
                      ai / cap if cap else 0.0, gain,
                      best_other_roof_fraction=best_other,
                      roof_tflops=roof_tflops)


def _throughput_ladder(cells, block_m: int, denominator: float):
    """`(tiles, useful throughput as a fraction of `denominator`)` per tread.

    The caller chooses the denominator and OWNS what the fraction then means.
    Gate 4 and `bracketing` pass `ridge x bandwidth`, so their fractions are
    fractions of peak compute and are comparable across arms and cards. A
    caller that passes the run's own plateau gets an ARM-RELATIVE number whose
    denominator moved 145.7-198.4 TFLOP/s inside one A100 session, which is not
    a quantity any threshold can be stated against.
    """
    out = []
    for c in sorted((c for c in cells
                     if c.block_m == block_m and c.aligned and c.status == "ok"),
                    key=lambda c: c.tiles_per_expert):
        if denominator > 0:
            out.append((c.tiles_per_expert, c.useful_tflops / denominator))
    return out


def gate_4_roof_fraction(*, block_m: int, alpha: float, ridge: float, b: int
                         ) -> tuple[float, float, float]:
    """`(model ceiling, retracted ceiling, threshold)`, all as fractions of peak.

    Both worlds cap the block size's attainable throughput at `ai_cap / ridge`
    of `ridge x bandwidth`, CLAMPED AT 1.0 -- a kernel cannot exceed the roof no
    matter how large its arithmetic intensity, and the retracted alpha puts the
    BLOCK_M=64 ceiling at 640/163 = 3.9 if the clamp is left out, which would
    make the midpoint 2.3 and the gate unfailable.

    The threshold is the midpoint of the two, so it is a DISCRIMINATOR in the
    same sense as gate 3's 0.33: above it the refit's ceiling is violated,
    below it the retracted world's is not reached.
    """
    model = min(ai_cap(block_m, alpha, b) / ridge, 1.0)
    retracted = min(ai_cap(block_m, RETRACTED_ALPHA, b) / ridge, 1.0)
    return model, retracted, 0.5 * (model + retracted)


def gate_4_no_crossing(cells, fits, *, block_m: int, plateau: float,
                       alpha: float, ridge: float, b: int, brack: Bracketing,
                       roof_tflops: float) -> Gate:
    """Does BLOCK_M=64 fail to reach the compute roof.

    BOTH SIDES ARE NOW FRACTIONS OF `ridge x bandwidth`, and that is the fix.
    The retired form scored `top > 0.85` where `top` was a fraction of the ARM'S
    OWN measured plateau, while the 0.85's stated rationale (`cap/ridge` = 0.716
    plus room for the fused layer's non-GEMM work) is a fraction of PEAK
    COMPUTE. Those denominators differ by the plateau's own shortfall, which
    across the 14 s3 reports runs 50.5-71.8% of `ridge x bandwidth`; in the
    gate's own units the model ceiling for the arm that "failed" it is 1.42, so
    0.891 was never evidence against anything. Both FAILs it ever produced are
    qwen2 at GROUP_SIZE_M=64, where the BLOCK_M=256 reference that SETS the
    plateau falls to 41.36 rows/ms against 45.1-45.7 elsewhere while BLOCK_M=64
    does not move: a denominator artefact, and the same config scored 0.871 FAIL
    on one run and 0.841 PASS on another.

    WHAT WAS NARROWED AS WELL AS RESCALED. A PASS is only allowed when the sweep
    demonstrated it could have produced a FAIL -- some OTHER block size actually
    reached `COMPUTE_BOUND_FRACTION` of `ridge x bandwidth` in this same grid.
    Without that the gate is a check that examined nothing: an absence measured
    by an instrument never shown to detect a presence. In every published report
    to date the plateau itself is 46.5-75.6% of that card's own roof, so no such
    control existed and the honest verdict there is UNDECIDED, not PASS.

    The arm-relative number the old gate scored is still printed, labelled, as a
    diagnostic. It is not the verdict.
    """
    fit = fits.get(block_m)
    tp = _throughput_ladder(cells, block_m, roof_tflops)
    model_ceiling, retracted_ceiling, threshold = gate_4_roof_fraction(
        block_m=block_m, alpha=alpha, ridge=ridge, b=b)
    separates = retracted_ceiling - model_ceiling >= GATE4_MIN_SEPARATION
    claim = (f"BLOCK_M={block_m} never reaches the compute roof "
             "(ridge x bandwidth)")
    gate_text = (f"< {COMPUTE_BOUND_FRACTION:.2f} of ridge x bandwidth"
                 + (f", and <= {threshold:.3f} (the midpoint of this run's "
                    f"ceiling {model_ceiling:.3f} and the rival world's "
                    f"{retracted_ceiling:.3f})" if separates else
                    f"; the ceiling test is SKIPPED because this run's ceiling "
                    f"{model_ceiling:.3f} and the rival's "
                    f"{retracted_ceiling:.3f} do not separate"))
    provenance = {
        "units": "fraction of ridge x bandwidth (peak compute)",
        "roof_tflops": roof_tflops,
        "plateau_tflops": plateau,
        "plateau_over_roof": (plateau / roof_tflops) if roof_tflops else None,
        "model_ceiling": model_ceiling,
        "retracted_ceiling": retracted_ceiling,
        "threshold": threshold,
        "positive_control_block_m": brack.positive_control,
        "best_other_roof_fraction": brack.best_other_roof_fraction,
        "retired_denominator": "the arm's own plateau; see GATE4_ROOF_FRACTION",
    }
    if not tp or fit is None or roof_tflops <= 0:
        why = ("no aligned cells at this block size" if not tp or fit is None
               else "no roof to measure against: ridge x bandwidth is zero")
        return Gate(4, claim, UNDECIDED, why, gate_text,
                    ["The sweep produced no exactly-full tile stack here."
                     if not tp or fit is None else
                     "ridge x bandwidth resolved to zero, so every fraction "
                     "below would divide by nothing. REFUSED."],
                    provenance=provenance)
    top = max(v for _, v in tp)
    provenance["peak_roof_fraction"] = top
    arm_top = (max(v for _, v in _throughput_ladder(cells, block_m, plateau))
               if plateau > 0 else None)
    provenance["peak_arm_relative_fraction"] = arm_top
    lines = [
        f"measured in fractions of ridge x bandwidth ({roof_tflops:.0f} "
        f"TFLOP/s); the refit caps this block size at cap/ridge = "
        f"{ai_cap(block_m, alpha, b):.1f}/{ridge:.1f} = {model_ceiling:.3f} "
        f"and the retracted alpha={RETRACTED_ALPHA} at {retracted_ceiling:.3f}",
        "throughput per tread as a fraction of that roof: "
        + ", ".join(f"n={n}:{v:.2f}" for n, v in tp),
        f"this arm's own plateau is {plateau:.1f} TFLOP/s = "
        + (f"{plateau / roof_tflops:.1%}" if roof_tflops else "n/a")
        + " of that roof. Against the plateau this block size peaks at "
        + (f"{arm_top:.3f}" if arm_top is not None else "n/a")
        + ", which is the number the retired gate scored against 0.85. It is "
          "ARM-RELATIVE and it is not the verdict.",
    ]
    if fit.slope_memory is not None and fit.compute_slope:
        c = fit.compute_slope
        src = ("measured on this block size's own compute-bound treads"
               if fit.slope_compute else "scaled from the reference by C ~ BLOCK_M")
        lines.append(
            f"per-tile slopes: memory B={fit.slope_memory:.4f} ms against "
            f"compute C={c:.4f} ms ({src}). C > B is the condition for a "
            f"crossing to exist at all, and it is "
            f"{'MET' if c > fit.slope_memory else 'NOT met'}.")
    lines += brack.lines()

    provenance["worlds_separate"] = separates
    provenance["reached_roof"] = top >= COMPUTE_BOUND_FRACTION
    provenance["exceeded_ceiling_midpoint"] = separates and top > threshold
    if not separates:
        # Reported, not scored around. Two ceilings that coincide say the run's
        # own alpha and the rival's put this block size in the same place, so
        # the ceiling comparison cannot discriminate. The DIRECT question --
        # did it reach the roof -- still can, and it is asked below.
        lines.insert(1, f"THE CEILING TEST IS SKIPPED AT BLOCK_M={block_m}: "
                        f"this run's ceiling {model_ceiling:.3f} and the "
                        f"retracted world's {retracted_ceiling:.3f} differ by "
                        f"{retracted_ceiling - model_ceiling:.3f}, under the "
                        f"{GATE4_MIN_SEPARATION:.2f} needed to tell them "
                        "apart. Only the direct roof question is scored.")

    # FAIL first, on the DIRECT question, judged by the same criterion the
    # positive control is judged by so that "reached the roof" means one thing
    # in this file. A block size that reached the roof crossed, and bracketing
    # governs an ABSENCE only: a sweep that watched the roof being reached is
    # bracketed by demonstration whatever the horizon says.
    if top >= COMPUTE_BOUND_FRACTION:
        return Gate(4, claim, FAIL, f"peak {top:.3f} of ridge x bandwidth",
                    gate_text,
                    [f"BLOCK_M={block_m} DID reach the roof, so it crosses and "
                     "the AI ceiling is not where this study put it."] + lines,
                    provenance=provenance)
    # The weaker falsification, and the one that needs the two worlds to be
    # apart: the block size did not reach the ROOF but did pass the ceiling its
    # own alpha puts on it, by more than half the distance to the rival world.
    if separates and top > threshold:
        return Gate(4, claim, FAIL, f"peak {top:.3f} of ridge x bandwidth",
                    gate_text,
                    [f"BLOCK_M={block_m} stayed below the roof but passed "
                     f"{threshold:.3f}, the midpoint between its modelled "
                     f"ceiling {model_ceiling:.3f} and the rival world's "
                     f"{retracted_ceiling:.3f}. The ceiling is not where this "
                     "run's alpha puts it."] + lines,
                    provenance=provenance)
    if not brack.sufficient:
        # NON-VACUITY. Without a control this gate is an absence reported by an
        # instrument never shown to detect a presence, and every published
        # report to date is in exactly this state.
        return Gate(4, claim, UNDECIDED, f"peak {top:.3f} of ridge x bandwidth",
                    gate_text,
                    ["THE SWEEP IS NOT BRACKETED, so an absence here is not "
                     "evidence of absence and this gate refuses to score it."]
                    + lines,
                    provenance=provenance)
    return Gate(4, claim, PASS, f"peak {top:.3f} of ridge x bandwidth",
                gate_text, lines, provenance=provenance)


# --------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------

@dataclass
class Report:
    lines: list[str]
    gates: list[Gate]
    payload: dict

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def analyse(cells, cfg, *, block_sizes, alpha: float, ridge: float,
            bandwidth_gbps: float, b: int, model_name: str, dtype: str,
            compiles: dict[int, int], executed: dict[int, int],
            sm_count: int, sm_source: str, pinned: dict | None = None,
            ridge_band: tuple[float, float] | None = None,
            ridge_source: str = "", ridge_band_source: str = "",
            capability=None, card: str = NO_CARD_SLUG) -> Report:
    """Everything between the timings and the verdicts. No GPU, no I/O.

    Kept pure and passed only cells so that `--self-test` and the test suite
    exercise the SAME code the pod run prints, rather than a second
    implementation that agrees with it until it does not.

    `ridge_band` IS NOT DEFAULTED TO `RIDGE_BAND`. That module constant is one
    machine's 2026-08-26 calibration, and defaulting to it is exactly how all 7
    published A100 reports came to carry a band belonging to neither card. When
    the caller does not state a band, the band is this run's own single ridge
    twice over, and the report says so -- a degenerate band is honest about
    being one calibration, a borrowed band is not.
    """
    if ridge_band is None:
        ridge_band = (ridge, ridge)
    ridge_band = (min(ridge_band), max(ridge_band))
    lines: list[str] = []
    ok = [c for c in cells if c.status == "ok" and c.ms_p50 > 0]
    aligned = [c for c in ok if c.aligned]
    plateau = max((c.useful_tflops for c in aligned), default=0.0)
    noise = statistics.median([c.rel_spread for c in ok]) if ok else 0.0

    # The roof and the pinned tile constants are passed in because the
    # qualification is a LEVEL test as well as a shape test, and level cannot be
    # judged without knowing what the card can do. There is no default: a
    # reference qualified against an unknown ceiling is the defect, not the fix.
    ref = compute_reference(ok, block_sizes, cfg=cfg, ridge=ridge,
                            bandwidth_gbps=bandwidth_gbps, b=b,
                            pinned=pinned or FIXED, capability=capability)
    # The margin scales with what the timing actually did. A fixed 2% was too
    # small at 2% spread: the reference slope carries the same spread, and a
    # compute branch estimated 2% low makes every compute-bound tread stand
    # "above" it, which is how a BLOCK_M=128 ladder planted at alpha=0.10
    # reported 0.80.
    margin = max(MEMORY_BRANCH_MARGIN, 3.0 * noise)
    fits = {bm: fit_ladder(ladder_points(ok, bm), bm, ref, margin)
            for bm in block_sizes}
    fits = {bm: f for bm, f in fits.items() if f.points}

    preds_lo = predictions(block_sizes, alpha, ridge_band[0], b)
    preds_hi = predictions(block_sizes, alpha, ridge_band[1], b)

    lines.append("")
    lines.append("PREDICTIONS, stated before the run and not adjusted after it")
    lines.append(f"  alpha {alpha:.3f} (band {ALPHA_BAND[0]}-{ALPHA_BAND[1]}), "
                 f"{dtype} at {b} bytes, ridge {ridge:.1f} Op/B, band "
                 f"{ridge_band[0]:.1f}-{ridge_band[1]:.1f} Op/B")
    lines.append(f"  ridge source: {ridge_source or 'NOT STATED by the caller'}")
    lines.append(f"  ridge band source: "
                 f"{ridge_band_source or 'NOT STATED by the caller'}")
    if ridge_band[0] == ridge_band[1]:
        lines.append("  the band is DEGENERATE: one calibration, so which tread "
                     "a crossing lands in is not bracketed by this run")
    lines.append(f"  BLOCK_M   AI cap   crossing @ridge {ridge_band[0]:<10.1f} "
                 f" crossing @ridge {ridge_band[1]:.1f}")
    for bm in block_sizes:
        p, ph = preds_lo[bm], preds_hi[bm]
        def fmt(pred):
            if pred.crossing_rows is None:
                return "NO CROSSING EVER    "
            tok = pred.crossing_tokens(cfg.num_experts, cfg.top_k)
            return f"r={pred.crossing_rows:7.1f} T={tok:8.0f} n={pred.first_compute_tread}"
        lines.append(f"  {bm:7d} {p.ai_cap:8.1f}   {fmt(p)}   {fmt(ph)}")
    r_lo = crossing_ratio(preds_lo, 128, 256)
    r_hi = crossing_ratio(preds_hi, 128, 256)
    if r_lo and r_hi:
        lines.append(f"  128-over-256 crossing ratio: {r_lo:.3f}x at the low "
                     f"ridge, {r_hi:.3f}x at the high one; the retracted "
                     f"alpha={RETRACTED_ALPHA} says 1.100x. MODEL, not "
                     "measurement: no crossing has been observed by this study.")
    # REGISTERED HERE, above the measurement, because both thresholds are
    # derived from this run's ridge and alpha rather than hardcoded, and a
    # derived threshold printed only beside its own verdict is a threshold a
    # reader cannot tell from a threshold chosen after the fact.
    null_bm = 64 if 64 in block_sizes else min(block_sizes)
    g4_model, g4_retracted, g4_threshold = gate_4_roof_fraction(
        block_m=null_bm, alpha=alpha, ridge=ridge, b=b)
    lines.append(f"  GATE 3 will test alpha > {GATE3_ALPHA_DISCRIMINATOR:.2f} "
                 f"(midpoint of {alpha:.3f} and {RETRACTED_ALPHA})")
    lines.append(
        f"  GATE 4 will test BLOCK_M={null_bm} against the roof in fractions "
        f"of ridge x bandwidth: this run's ceiling {g4_model:.3f}, the "
        f"retracted world's {g4_retracted:.3f}, so the gate is "
        f"{COMPUTE_BOUND_FRACTION:.2f} (reached the roof) and "
        f"{g4_threshold:.3f} (passed its own ceiling)")

    lines.append("")
    lines.append(f"MEASURED  {model_name} {dtype}  {len(ok)} cells, "
                 f"{len(aligned)} of them exactly-full tile stacks")
    model_roof = ridge * bandwidth_gbps * 1e9 / 1e12
    lines.append(f"  compute plateau {plateau:.1f} TFLOP/s useful, taken as the "
                 "roof every roof fraction below is against")
    # The plateau is the sweep's own maximum, so it is a roof only if something
    # in the sweep actually reached one. `ridge x bandwidth` is what the roof
    # should be; a plateau far below it means nothing here is compute bound and
    # every roof fraction is against a ceiling that does not exist.
    lines.append(f"  that plateau is {plateau / model_roof:.1%} of "
                 f"ridge x bandwidth ({model_roof:.0f} TFLOP/s). Far below 100% "
                 "means nothing in the sweep reached a roof and every roof "
                 "fraction is relative to something that is not one.")
    lines.append(f"  per-cell timing spread, median {noise:.2%}")
    lines += ref.render()
    lines.append(f"  {sm_count} SMs ({sm_source}), one resident CTA per SM "
                 "assumed for every wave count")
    if ok:
        lines.append(
            f"  up-GEMM waves run {min(c.waves_up for c in ok):.1f} to "
            f"{max(c.waves_up for c in ok):.1f} across the sweep. The smallest "
            "is the smallest tile stack, and even it is many waves deep, so no "
            "cell here is a partial-wave measurement and occupancy cannot be "
            "the thing that moved between two adjacent cells.")

    lines.append("")
    lines.append("THE LADDER: milliseconds per exactly-full tile stack, which is "
                 "where gates 2, 3 and 4 are read")
    lines.append("  alpha is a LOWER bound (the fused layer's fixed cost sits "
                 "in the denominator); alpha-hi takes that cost out, and "
                 "alpha-corrected also removes activation traffic")
    if ref.refused:
        # Said BEFORE the table, because the table is all n/a and a reader who
        # meets the blanks first will reach for the tread count -- which is what
        # happened to the BN=256 arm across two cards and eight published cells.
        lines.append("  EVERY BLANK BELOW IS CAUSED BY THE REFUSED REFERENCE "
                     f"ABOVE (BLOCK_M={ref.refused_block_m}), not by a shortage "
                     "of memory-bound treads. Withdraw this arm; do not table "
                     "it beside arms whose reference qualified.")
    lines.append("  BLOCK_M  treads  memory-bound  alpha   alpha-corrected  "
                 "alpha-hi  B ms/tile  C ms/tile  fit err")
    alpha_hat, alpha_source, alpha_source_bm = None, "", None
    alpha_corrected: dict[int, float] = {}
    for bm in sorted(fits):
        f = fits[bm]
        corr = None
        if f.alpha is not None and f.load_ms:
            corr = ((f.slope_memory - activation_slope_ms(cfg, bm, bandwidth_gbps))
                    / f.load_ms)
            alpha_corrected[bm] = corr
        hi = f.alpha_upper
        lines.append(
            f"  {bm:7d}  {len(f.points):6d}  {f.memory_points:12d}  "
            + (f"{f.alpha:5.3f}" if f.alpha is not None else "  n/a")
            + "   " + (f"{corr:13.3f}" if corr is not None else "          n/a")
            + "  " + (f"{hi:8.3f}" if hi is not None else "     n/a")
            + "  " + (f"{f.slope_memory:9.4f}" if f.slope_memory is not None else "      n/a")
            + "  " + (f"{f.slope_compute:9.4f}" if f.slope_compute is not None else "      n/a")
            + f"  {f.mean_rel_err:6.2%}")
        # Last eligible ladder wins, and the loop runs in ascending order, so
        # this is the LARGEST block size that measured alpha over enough treads
        # -- the closest to the 128 gate 3 has to import it to, and so the
        # shortest extrapolation across the drift `ALPHA_BY_BLOCK_M` records.
        eligible = (corr is not None
                    and ref.block_m is not None
                    and bm != ref.block_m
                    and f.memory_points >= MIN_MEMORY_TREADS)
        if eligible:
            alpha_hat, alpha_source_bm = corr, bm
            alpha_source = (
                f"measured at BLOCK_M={bm} over {f.memory_points} memory-bound "
                "treads, activation traffic subtracted")

    consistency = _compute_slope_consistency(fits)
    if consistency:
        lines.append("  consistency: " + consistency)

    gates = [
        gate_0_override(compiles, executed, block_sizes),
        gate_1_steps(ok, cfg, preds_lo, alpha=alpha, ridge=ridge,
                     bandwidth_gbps=bandwidth_gbps, b=b, noise=noise),
        gate_2_direction(ok, cfg, alpha=alpha, retracted=RETRACTED_ALPHA,
                         ridge=ridge, bandwidth_gbps=bandwidth_gbps, b=b,
                         block_sizes=block_sizes),
        gate_3_alpha_discriminates(fits, preds_lo, preds_hi, cfg, lo=128,
                                   hi=256, alpha_source=alpha_source,
                                   alpha_hat=alpha_hat,
                                   alpha_source_bm=alpha_source_bm,
                                   ridge_band=ridge_band),
    ]
    # `model_roof`, not `plateau`: see gate_4_no_crossing and Bracketing on why
    # scoring an absence against the run's own maximum examines nothing.
    brack = bracketing(ok, null_bm, alpha, ridge, b, fits, model_roof)
    gates.append(gate_4_no_crossing(ok, fits, block_m=null_bm, plateau=plateau,
                                    alpha=alpha, ridge=ridge, b=b, brack=brack,
                                    roof_tflops=model_roof))

    lines.append("")
    lines.append("GATES")
    for g in gates:
        lines += g.render()
        lines.append("")

    verdicts = {g.verdict for g in gates}
    if ref.refused:
        # Ranked ABOVE gate 0, because a refused reference is not a failed
        # prediction: it means the instrument, not the hypothesis, is what the
        # arm measured. Nothing in it is a result either way.
        lines.append(
            "READING IT. THE COMPUTE REFERENCE WAS REFUSED ON ITS LEVEL, so "
            "every ladder below was classified against a compute branch that "
            "is the wrong size. No alpha here is a measurement, no blank here "
            "is a null, and the gates are being scored against an instrument "
            "rather than against the hardware. WITHDRAW THIS ARM.")
    elif gates[0].verdict != PASS:
        lines.append("READING IT. Gate 0 failed, so the four settings may not "
                     "have been four kernels. Nothing below gate 0 is evidence.")
    elif verdicts == {PASS}:
        lines.append("READING IT. Every gate passed at alpha "
                     f"{alpha:.3f}. The AI ceiling is real, it is where the "
                     "refit put it, and BLOCK_M 32 and 64 cannot reach the "
                     "compute roof at any batch size.")
    else:
        failed = [g.number for g in gates if g.verdict != PASS]
        lines.append(f"READING IT. Gates {failed} did not pass. A FAIL here is "
                     "a result: it falsifies the tile-corrected roofline at "
                     f"alpha={alpha:.3f} in a specific, named place, which is "
                     "what the gate was built to do.")

    payload = {
        "alpha": alpha, "alpha_band": list(ALPHA_BAND),
        "retracted_alpha": RETRACTED_ALPHA, "ridge": ridge,
        "ridge_band": list(ridge_band),
        "ridge_source": ridge_source or "NOT STATED by the caller",
        "ridge_band_source": ridge_band_source or "NOT STATED by the caller",
        "ridge_band_degenerate": ridge_band[0] == ridge_band[1],
        "model_roof_tflops": model_roof, "dtype_bytes": b,
        "model": model_name, "dtype": dtype, "fixed": pinned or FIXED,
        "plateau_tflops": plateau, "timing_spread_median": noise,
        "overhead_ms": ref.overhead_ms, "compute_reference": asdict(ref),
        "sm_count": sm_count, "sm_source": sm_source,
        # THE CARD, IN THE ONLY MACHINE-READABLE ARTEFACT. `Cell` carries no
        # device column, so without this a cells.csv on a shared network volume
        # is unattributable after the pod is gone -- and this study's central
        # comparison is between two cards.
        "card": card,
        "alpha_measured": alpha_hat, "alpha_source": alpha_source,
        "predictions": {
            str(bm): {"ai_cap": preds_lo[bm].ai_cap,
                      "crossing_rows_ridge_lo": preds_lo[bm].crossing_rows,
                      "crossing_rows_ridge_hi": preds_hi[bm].crossing_rows,
                      "first_compute_tread": preds_lo[bm].first_compute_tread}
            for bm in block_sizes},
        "ladder": {str(bm): {"points": list(f.points),
                             "memory_points": f.memory_points,
                             "alpha": f.alpha,
                             "alpha_corrected": alpha_corrected.get(bm),
                             "alpha_upper": f.alpha_upper,
                             "slope_memory": f.slope_memory,
                             "slope_compute": f.slope_compute,
                             "slope_compute_ref": f.slope_compute_ref,
                             "crosses": f.crosses,
                             "basis": f.basis,
                             # A blank alpha has two causes and they are not
                             # interchangeable. Written per ladder so a reader
                             # of one row -- or a table generator like
                             # `scripts/alpha_surface.py` -- cannot print a
                             # refused reference as a sweep that lacked treads.
                             "identifiable": (f.memory_points
                                              >= MIN_MEMORY_TREADS
                                              and f.alpha is not None),
                             "unidentifiable_reason": _why_not_identifiable(
                                 f, ref),
                             "mean_rel_err": f.mean_rel_err}
                   for bm, f in fits.items()},
        "bracketing": asdict(brack),
        # THE DETAIL LINES AND THE PROVENANCE ARE SERIALIZED. They used not to
        # be, and that is how 22 published reports asserted a crossing ratio
        # with nothing in the file to say it was a restatement of a fitted
        # alpha imported from another BLOCK_M. A verdict a reader cannot trace
        # is a verdict a reader cannot check.
        "gates": [{"number": g.number, "claim": g.claim, "kind": g.kind,
                   "basis": g.basis, "verdict": g.verdict,
                   "measured": g.measured, "gate": g.threshold,
                   "detail": list(g.lines), "provenance": g.provenance}
                  for g in gates],
    }
    return Report(lines, gates, payload)


def _compute_slope_consistency(fits) -> str:
    """`C ~ BLOCK_M` across settings, which gate 4 leans on.

    A cross-check and not a gate: if the compute branch does not scale with the
    tile height, the scaled `C` gate 4 compares against is wrong, and a reader
    should be told before the verdict rather than after it.
    """
    have = [(bm, f.slope_compute) for bm, f in sorted(fits.items())
            if f.slope_compute]
    if len(have) < 2:
        return ""
    parts = []
    for (bm0, c0), (bm1, c1) in zip(have, have[1:], strict=False):
        parts.append(f"C({bm1})/C({bm0}) = {c1 / c0:.2f}x against "
                     f"{bm1 / bm0:.2f}x predicted")
    return "compute branch should scale with BLOCK_M -- " + "; ".join(parts)


# --------------------------------------------------------------------------
# The GPU half.
# --------------------------------------------------------------------------

def find_override():
    """vLLM's own tuning hook, probed rather than assumed.

    `try_get_optimal_moe_config` consults `get_config()` first and a truthy
    value bypasses the tuned file and the default ladder both. The import path
    has moved between versions, so a wrong guess would silently sweep nothing --
    which is precisely the failure gate 0 exists to catch, and this is the first
    line of that defence.
    """
    import importlib
    candidates = ["vllm.model_executor.layers.fused_moe",
                  "vllm.model_executor.layers.fused_moe.fused_moe",
                  "vllm.model_executor.layers.fused_moe.config"]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(mod, "override_config", None)
        if fn is not None:
            return fn, name
    raise SystemExit(
        "could not find vLLM's override_config in any of:\n  "
        + "\n  ".join(candidates)
        + "\nCheck the installed vLLM version; try_get_optimal_moe_config reads "
          "it via get_config(), so the hook exists under some name.")


def arm_triton_cache(root: Path, block_m: int) -> Path:
    """Point Triton at a fresh directory for THIS setting, before it compiles.

    Set before the first compile of the setting, which is the first timed call
    at that BLOCK_SIZE_M, because Triton reads these at compile time. Within one
    process each block size is a distinct specialisation and so a distinct cache
    entry anyway; the per-setting directory is what makes "did this setting
    compile anything" a countable question instead of an assumption. A warm
    cache dumping nothing is what cost this project its A100 PTX dump.
    """
    directory = root / f"bm{block_m}"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(directory)
    return directory


def count_new(root: Path, seen: set[Path]) -> int:
    fresh = [p for p in root.rglob("*") if p.is_file() and p not in seen]
    seen.update(fresh)
    return len(fresh)


def time_call(fn, warmup: int, iters: int):
    """Median, min and stdev milliseconds over CUDA events.

    No L2 flush, matching `tile_sweep.py`: the comparison is between tile
    settings on identical data, and a flush adds its own variance to both sides
    of every step this sweep is trying to resolve.
    """
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out), min(out), statistics.pstdev(out)


def _make_call(fused_experts, x, weights, w, ids, kw):
    """Bind explicitly rather than closing over the loop variables (ruff B023)."""
    def call():
        return fused_experts(hidden_states=x, w1=weights.w1, w2=weights.w2,
                             topk_weights=w, topk_ids=ids, **kw)
    return call


def balanced_ids(cfg, tokens: int, device: str):
    """Top-k ids whose per-expert histogram is EXACTLY `T k / E`.

    Not `sample_topk_ids(uniform)`. Sampled uniform routing puts about 15 rows
    of spread on a mean of 256 at mixtral T=1024, which smears every tile step
    across 60 tokens and makes gate 1 a matter of opinion. `realize_counts`
    builds the histogram exactly, so rows per expert is an integer this script
    knew before the pod was rented and every tile step lands where the grid was
    built to look for it.
    """
    from moe.routing.distributions import realize_counts
    per = tokens * cfg.top_k // cfg.num_experts
    if per * cfg.num_experts != tokens * cfg.top_k:
        raise ValueError(f"T={tokens} does not divide evenly over "
                         f"E={cfg.num_experts} at k={cfg.top_k}")
    return realize_counts([per] * cfg.num_experts, tokens, cfg.top_k,
                          device=device)


def run_sweep(args, cfg, grid, block_sizes, csv_path: Path, cache_root: Path,
              b: int, pinned: dict) -> tuple[list[Cell], dict[int, int],
                                              dict[int, int]]:
    """The metered part. Appends every cell as it lands, so aborting keeps it.

    BLOCK_SIZE_M IS THE OUTER LOOP, which is a trade. It makes the Triton cache
    attribution exact -- everything that compiles while a setting runs belongs
    to that setting -- and it makes a resumed run finish a setting before
    starting the next. It also means gate 2's comparison of four settings at the
    same rows-per-expert spans the whole sweep rather than a few seconds, so it
    carries whatever the clocks did in between. The sweep is about a minute of
    GPU on mixtral and the effect gate 2 is looking for is a factor of three, so
    that is a trade worth making; on a card that throttles it would not be.
    """
    import torch

    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    # BEFORE vLLM is imported. Triton may snapshot this variable at import in
    # some versions, and a warm cache dumps and compiles nothing -- the bug that
    # cost this project its A100 PTX dump. Pointing it at this run's own
    # directory first guarantees freshness relative to previous runs whatever
    # the per-setting redirect below manages, and the artefact count is taken
    # over the whole root so the assay works either way.
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)

    override_config, where = find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from moe.baselines._framework_config import vllm_call_kwargs

    print(f"override hook: {where}.override_config")
    print(f"triton cache: {cache_root} (fresh for this run)")
    done, cells = read_cells(csv_path)
    seen_files: set[Path] = set()
    compiles: dict[int, int] = {}
    executed: dict[int, int] = {}
    sm_count = args.sm_count or torch.cuda.get_device_properties(0).multi_processor_count

    inputs: dict[int, tuple] = {}
    for bm in block_sizes:
        arm_triton_cache(cache_root, bm)
        count_new(cache_root, seen_files)
        compiles.setdefault(bm, 0)
        executed.setdefault(bm, 0)
        for rows in grid:
            tokens = tokens_for_rows(cfg, rows)
            if (bm, tokens) in done:
                continue
            if tokens not in inputs:
                spec = BenchSpec(cfg, num_tokens=tokens, dtype=args.dtype,
                                 routing=RoutingSpec("uniform", 0.0),
                                 seed=args.seed)
                x, weights = make_inputs(spec, device="cuda")
                ids = balanced_ids(cfg, tokens, "cuda")
                w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                               device="cuda")
                kw = vllm_call_kwargs(spec)
                kw["activation"] = MoEActivation(kw["activation"])
                inputs = {tokens: (x, weights, ids, w, kw)}   # one cell live at a time
            x, weights, ids, w, kw = inputs[tokens]
            executed[bm] += 1
            conf = dict(pinned, BLOCK_SIZE_M=bm)
            call = _make_call(fused_experts, x, weights, w, ids, kw)
            try:
                with override_config(conf):
                    call()
                    torch.cuda.synchronize()
                    compiles[bm] += count_new(cache_root, seen_files)
                    ms0, _, _ = time_call(call, 1, 3)
                    iters = scaled_iters(ms0, args.iters, args.cell_budget_ms)
                    ms, mn, sd = time_call(call, args.warmup, iters)
                cell = make_cell(cfg, rows, bm, ms, sm_count=sm_count,
                                 block_n=pinned["BLOCK_SIZE_N"], ms_min=mn,
                                 ms_stdev=sd, iters=iters)
            except Exception as exc:                    # noqa: BLE001
                cell = make_cell(cfg, rows, bm, 0.0, sm_count=sm_count,
                                 block_n=pinned["BLOCK_SIZE_N"], status="failed",
                                 detail=f"{type(exc).__name__}: {exc}")
                print(f"  BM={bm} T={tokens} FAILED  {cell.detail}")
                if "shared memory" in str(exc).lower():
                    print(f"  ^ re-run the WHOLE sweep with --num-stages "
                          f"{max(1, pinned['num_stages'] - 1)}. Dropping stages "
                          "for one setting alone would unpin the thing this "
                          "sweep holds fixed.")
            cells.append(cell)
            append_cell(csv_path, cell)
            print(f"  BM={bm:3d} T={cell.tokens:6d} r={rows:6d} "
                  f"n={cell.tiles_per_expert:3d} waves {cell.waves_up:7.1f} "
                  f"{cell.ms_p50:9.4f} ms  {cell.useful_tflops:7.1f} TFLOP/s")
    return cells, compiles, executed


# --------------------------------------------------------------------------
# Persistence.
# --------------------------------------------------------------------------

CSV_FIELDS = [f for f in Cell.__dataclass_fields__]


def append_cell(path: Path, cell: Cell) -> None:
    """One row, flushed. An abort costs the cell in flight and nothing else."""
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(asdict(cell))
        fh.flush()


def read_cells(path: Path) -> tuple[set[tuple[int, int]], list[Cell]]:
    """Cells already measured, so a re-run resumes rather than repeats."""
    if not path.exists():
        return set(), []
    cells: list[Cell] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            kw = {}
            for name, f in Cell.__dataclass_fields__.items():
                raw = row.get(name, "")
                if f.type in ("int",):
                    kw[name] = int(float(raw or 0))
                elif f.type in ("float",):
                    kw[name] = float(raw or 0.0)
                elif f.type in ("bool",):
                    kw[name] = raw == "True"
                else:
                    kw[name] = raw
            cells.append(Cell(**kw))
    # Only the cells that SUCCEEDED count as done. A cell that failed is
    # retried on the next run, because the common failure here is a setting
    # that ran out of shared memory or a pod that lost its device, and both are
    # states a re-run can leave behind. A failure that is real fails again in
    # milliseconds.
    return {(c.block_m, c.tokens) for c in cells if c.status == "ok"}, cells


def results_root() -> Path:
    """`$MOE_RESULTS_DIR`, else the network volume, else the repo.

    Same order `scripts/run_all.sh` resolves it in, so this experiment lands
    beside every other arm on the volume that outlives the pod.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return Path(__file__).resolve().parents[1] / "results"


def default_run_id(args, card: str) -> str:
    """Derived from the arguments, so "the same experiment" resumes itself.

    A random id would make every re-run a new directory and turn the resume
    path into dead code the first time anyone used it.

    EVERY pinned knob has to be in this key. It used to omit GROUP_SIZE_M,
    BLOCK_SIZE_N and num_stages, which was safe only while all three were
    unreachable constants. The moment --group-m existed it became a silent
    overwrite: a G=16 run would derive the SAME id as the G=1 run, resume into
    its directory, find every cell already on disk, skip all of them, and report
    G=1's timings under a G=16 heading. Nothing would have looked wrong, because
    the report prints `pinned` from argv rather than from the cells it read. The
    knobs are in the visible name too, so two runs are distinguishable in `ls`
    and not only by a hash nobody can invert.

    THE SAME OMISSION SURVIVED IN FOUR MORE FIELDS UNTIL 2026-09-02, and the
    first of them had already been committed:

      * THE CARD. It is not swept by this script, it is swept by the operator
        moving to another pod, and `results_root()` prefers `$MOE_RESULTS_DIR`
        then `/workspace/results`, the network volume the runbook uses BECAUSE
        it outlives the pod. Two cards therefore derived one id, and the proof
        is in the repo: `results/published/2026-09-01-nvidia_h200-cross-card-s3`
        and `results/published/2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3`
        both contain `mixtral-8x7b-bf16-r1024-g1-n64-4867a2.report.json`, for
        `sm_count` 132 and 108. `read_cells` keys resume on `(block_m, tokens)`
        and `Cell` carries no device column, so nothing downstream would notice:
        the second card finds all cells present, skips them, spends no GPU time,
        and prints the first card's timings scored against its own ridge --
        145.7 against 162.8. `scripts/replicate_noise_floor.py:run_id_for`
        documents this defect and works around it locally; fixing it here is
        what stops the next caller inheriting it.
      * `--iters`, `--warmup` and `--cell-budget-ms`. These are not analysis
        knobs: they set the measured milliseconds of every cell. A `--iters 200`
        re-run after `--iters 50` landed in the same directory and printed the
        50-iteration numbers under the 200-iteration label, invisibly, because
        the report renders `pinned` and the arguments from argv rather than from
        the cells it read.

    `--ridge`, `--ridge-band`, `--alpha` and `--bandwidth-gbps` stay OUT of the
    key on purpose: they re-analyse a set of cells rather than change one, so
    two analyses of one sweep belong in one directory.
    """
    key = json.dumps({"card": card, "model": args.model, "dtype": args.dtype,
                      "tiles": args.tiles, "r_max": args.r_max,
                      "row_step": args.row_step, "probes": args.step_probes,
                      "seed": args.seed, "group_m": args.group_m,
                      "block_n": args.block_n, "num_stages": args.num_stages,
                      "iters": args.iters, "warmup": args.warmup,
                      "budget": args.cell_budget_ms},
                     sort_keys=True)
    return f"{card}-{args.model}-{args.dtype}-r{args.r_max}-" \
           f"g{args.group_m}-n{args.block_n}-" \
           f"{hashlib.sha1(key.encode()).hexdigest()[:6]}"


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mixtral-8x7b", choices=sorted(MODEL_CONFIGS),
                    help="mixtral by default: E/k=4 makes the whole "
                         "rows-per-expert range reachable at four times the "
                         "token count, where deepseek-v3 needs thirty-two")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"),
                    help="bf16 or fp16. Not fp8: the alpha refit, the ridge "
                         "band and the crossing table above are all bf16 "
                         "statements, and the fp8 call path needs a quant "
                         "config this sweep does not build")
    ap.add_argument("--tiles", default="32,64,128,256")
    ap.add_argument("--r-max", type=int, default=1024,
                    help="largest rows per expert. 1024 is 16 M-tiles at "
                         "BLOCK_M=64, which is 95%% of that block size's AI "
                         "ceiling and 4.9x the crossing the retracted alpha "
                         "predicts for it")
    ap.add_argument("--row-step", type=int, default=32)
    ap.add_argument("--num-stages", type=int, default=FIXED["num_stages"],
                    help="pipeline stages, applied to EVERY setting. The one "
                         "pinned parameter with a hard limit behind it: "
                         "BLOCK_SIZE_M=256 at 4 stages asks for about 164 KB of "
                         "shared memory, and a card that refuses it fails that "
                         "setting alone, which would unpin the sweep. Lower it "
                         "here and every setting moves together")
    ap.add_argument("--group-m", type=int, default=FIXED["GROUP_SIZE_M"],
                    help="the swizzle width, applied to EVERY setting. Pinned to "
                         "1 by default and by design: 1 is the setting vLLM's "
                         "FALLBACK ladder holds across the whole decode range, so "
                         "it is the one a deployment without a tuned file "
                         "actually runs. It is NOT a neutral choice, because "
                         "GROUP_SIZE_M is what groups consecutive M-tiles onto "
                         "one weight read, and alpha measured here is therefore "
                         "alpha AT THIS SWIZZLE rather than a property of the "
                         "kernel. Sweeping it is the point: the 2026-09-01 "
                         "session measured alpha 0.92-1.02 at 1 and 0.58-0.62 at "
                         "8 and above, so the ceiling 2*BM/(alpha*b) -- and "
                         "therefore whether a given tile can EVER reach the "
                         "compute roof -- moves with this number")
    ap.add_argument("--block-n", type=int, default=FIXED["BLOCK_SIZE_N"],
                    help="the N tile, applied to EVERY setting. Exists to bound "
                         "the ACTIVATION confound rather than to tune anything. "
                         "An extra M-tile re-reads activations as well as "
                         "weights, in the ratio BLOCK_M/BLOCK_N, so at the "
                         "default 64 a BLOCK_M=64 setting re-reads them one for "
                         "one and a BLOCK_M=256 setting four times over -- which "
                         "means alpha fitted across that grid is NOT bounded by "
                         "the 0.25 this study quotes elsewhere. Raise this to "
                         "256 and the ratio at BLOCK_M=64 falls to 0.25; if "
                         "alpha does not move, the weight-traffic reading holds")
    ap.add_argument("--step-probes", type=int, default=6,
                    help="tile boundaries per block size to bracket for gate 1")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cell-budget-ms", type=float, default=400.0,
                    help="iterations are cut so one cell stays inside this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sm-count", type=int, default=0,
                    help="0 asks the driver; only needed off-GPU")
    ap.add_argument("--capability", default="",
                    help="compute capability as MAJOR.MINOR, e.g. 8.0 for the "
                         "A100 or 9.0 for the H200. Empty asks the driver; "
                         "give it off-GPU to get the shared-memory verdict in "
                         "the tile-resource plan. The register check needs no "
                         "device and runs either way")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="Op/B. 0 (the default) reads the ATTACHED DEVICE's "
                         "own calibration and REFUSES if there is none. It used "
                         "to default to 160.3, an H200 figure, and "
                         "cross_card_surface.sh never passed it, so all 7 "
                         "published A100 reports were scored against a ridge "
                         "belonging to neither card")
    ap.add_argument("--ridge-band", default="",
                    help="LO,HI in Op/B, only meaningful with --ridge. Without "
                         "it an operator-asserted ridge gets a DEGENERATE band "
                         "and the report says so, rather than inheriting a "
                         "width measured on some other machine")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--bandwidth-gbps", type=float, default=0.0,
                    help="0 reads this machine's calibration, else 4374.5")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out", type=Path, default=None,
                    help="overrides the results root entirely")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid, the predictions and the cost, then stop")
    ap.add_argument("--self-test", type=float, default=None, metavar="ALPHA",
                    help="generate the cells from the model at this alpha and "
                         "run the whole analysis on them, off GPU")
    ap.add_argument("--self-test-noise", type=float, default=0.0,
                    help="lognormal sigma applied to every synthetic cell")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="exit non-zero unless every gate passes; off by "
                         "default because a falsified prediction is a "
                         "successful run, not a failed one")
    return ap


def missing_gpu_stack() -> str:
    """Which half of the stack is absent, and what to run instead.

    One function rather than three checks inline so the message a laptop gets
    is the same one the pod would get, and so the test suite can assert on it
    without a GPU. Empty string means the sweep can run.
    """
    try:
        import torch
    except ImportError:
        return ("no torch on this machine. --self-test 0.558 runs the whole "
                "analysis off GPU; --dry-run prints the grid.")
    if not torch.cuda.is_available():
        return ("no CUDA device. --self-test 0.558 runs the whole analysis off "
                "GPU; --dry-run prints the grid.")
    try:
        import vllm  # noqa: F401
    except ImportError:
        return ("vLLM is not importable in this environment, and it owns the "
                "kernel this sweep overrides.\nOn the pod: source the vllm venv "
                "(scripts/setup_runpod.sh vllm). Off GPU: --self-test 0.558.")
    return ""


class RidgeUnavailable(RuntimeError):
    """No ridge this run is entitled to use, and no constant may stand in.

    Raised rather than defaulted. `--ridge` used to default to `RIDGE_BAND[0]`
    and `scripts/cross_card_surface.sh` never passed it, so seven A100 reports
    were written against 160.3 Op/B -- a stale H200 figure -- and every
    `ridge x bandwidth` they printed was a hybrid of two machines. Nothing
    about that failure was visible in the output, which is precisely why the
    replacement refuses instead of choosing.
    """


@dataclass(frozen=True)
class ResolvedRidge:
    """The ridge, its band, and where each came from. Provenance travels with it."""

    ridge: float
    band: tuple[float, float]
    source: str
    band_source: str
    device: str


#: The phrase `moe/bench/calibrate.py` writes onto a bandwidth pattern it has
#: disowned. A ridge built from a disowned denominator is a ridge built from a
#: number the calibration itself says is not a ceiling.
DISOWNED_NOTE = "not a valid ceiling"


def _measured_yaml(gpu_name: str) -> dict:
    """THIS device's calibration yaml as a dict, or {}.

    Read directly rather than through `roofline.Hardware`, which carries only
    the headline peaks. The candidate order mirrors `load_measured` so this
    cannot end up describing a different file from the one the ridge came from,
    and the device name inside the file is checked again here: a detail block
    from another machine would put another machine's pattern spread on this
    run's band, which is the same class of error the whole change is about.
    """
    try:
        import yaml

        from moe.bench.roofline import HARDWARE_DIR, measured_slug
    except ImportError:                                   # pragma: no cover
        return {}
    stems = ([measured_slug(gpu_name)] if gpu_name else []) + ["measured"]
    for stem in stems:
        path = HARDWARE_DIR / f"{stem}.yaml"
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception:                                 # noqa: BLE001
            return {}
        named = str((data.get("detail") or {}).get("gpu_name") or "")
        if named and gpu_name:
            norm = lambda t: "".join(c for c in t.lower() if c.isalnum())  # noqa: E731
            if norm(named) not in norm(gpu_name) and norm(gpu_name) not in norm(named):
                return {}
        return data
    return {}


def _measured_detail(gpu_name: str) -> dict:
    """Just the `detail` block, which is where the per-pattern ridges live."""
    return _measured_yaml(gpu_name).get("detail") or {}


def calibration_stamp_line(doc: dict) -> str:
    """Which calibration SESSION the ridge came from, named in the report.

    The device match this replaces a hardcoded constant with is necessary and
    not sufficient: `measured_*.yaml` is keyed by DEVICE NAME, so a second pod
    of the same part inherits the first pod's ceilings, and the H200's dense
    bf16 moved 7.1% between 2026-08-28 and 2026-09-01 while its bandwidth
    reproduced to 0.014%. The ridge is compute over bandwidth, so that drift
    lands entirely on the ridge. Naming the session does not stop the reuse --
    `scripts/cross_card_surface.sh` recalibrating first is what stops it -- but
    it makes a report that inherited one say whose it was.
    """
    when = doc.get("checked_on")
    commit = str(doc.get("measured_commit") or "")[:8]
    dirty = " DIRTY TREE" if doc.get("measured_dirty") else ""
    if not when and not commit:
        return "calibration session unstamped"
    return f"calibrated {when or 'undated'} at {commit or 'no commit'}{dirty}"


def ridge_band_from_detail(detail: dict, ridge: float
                           ) -> tuple[tuple[float, float], str]:
    """A ridge band from THIS device's own bandwidth patterns, or a degenerate one.

    The band is the same silicon measured against several DRAM rulers, which is
    the honest width of a single calibration: the compute term is common to
    every end, so the spread is carried as a RATIO against the ceiling pattern
    and applied to `ridge`. Carrying the ratio rather than the stored
    `ridge_by_pattern` values keeps this correct for a dtype whose peak is not
    the bf16 peak those values were computed from.

    Patterns the calibration disowned are excluded: `calibrate.py` marks a
    pattern that came in below triad, which a read cannot legitimately do, and
    a band built from one is a band built from a number the file has already
    withdrawn.

    Returns a DEGENERATE band when fewer than two rulers survive. Degenerate is
    the honest answer for one calibration; borrowing another machine's band is
    not, and that is what this function exists to stop.
    """
    by_pattern = detail.get("ridge_by_pattern") or {}
    ceiling = detail.get("ceiling_pattern")
    base = by_pattern.get(ceiling)
    if not base or not ridge:
        return (ridge, ridge), ("degenerate: the calibration records no "
                                "per-pattern ridges, so this run has one "
                                "denominator and the band is one number twice")
    disowned = {p.get("pattern") for p in (detail.get("bandwidth_patterns") or [])
                if DISOWNED_NOTE in str(p.get("note") or "")}
    kept = {k: v for k, v in by_pattern.items() if k not in disowned and v}
    if len(kept) < 2:
        return (ridge, ridge), ("degenerate: fewer than two bandwidth patterns "
                                "survived this calibration's own disowning")
    scaled = sorted(ridge * v / base for v in kept.values())
    names = ", ".join(sorted(kept))
    return (scaled[0], scaled[-1]), (
        f"this device's own bandwidth patterns ({names}), carried as a ratio "
        f"against the {ceiling} ceiling; {sorted(disowned)} excluded as "
        "disowned by the calibration" if disowned else
        f"this device's own bandwidth patterns ({names}), carried as a ratio "
        f"against the {ceiling} ceiling")


def resolve_ridge(args, *, synthetic: bool) -> ResolvedRidge:
    """The ridge THIS run is entitled to quote, or a refusal.

    Order, and each step is a different kind of claim:

      1. `--ridge` (with optional `--ridge-band`), which makes the number the
         operator's assertion and puts it in the run's own command line.
      2. THE ATTACHED DEVICE'S OWN CALIBRATION: `peak(dtype) / bandwidth`, from
         the yaml `scripts/calibrate_hardware.py` wrote for this GPU. This is
         the default, and it is the whole point of the change: the A100's
         contemporaneous calibration puts its ridge at 145.8 and the H200's at
         162.8, and the reports that quoted 160.3 on the A100 were quoting
         neither.
      3. For `--dry-run` and `--self-test` ONLY, where nothing was measured and
         so nothing can be mislabelled, the module's H200 band as a stated
         HYPOTHESIS. The source string says so and is written into the report.
      4. Otherwise REFUSE. A measured run with no calibration for its own
         device does not get a ridge from anywhere else.
    """
    band_arg = getattr(args, "ridge_band", "") or ""
    if args.ridge:
        band = (args.ridge, args.ridge)
        band_source = ("degenerate: --ridge given as one number and no "
                       "--ridge-band with it")
        if band_arg:
            ends = tuple(float(v) for v in band_arg.split(","))
            if len(ends) != 2:
                raise RidgeUnavailable(
                    f"--ridge-band wants LO,HI; got {band_arg!r}")
            band = (min(ends), max(ends))
            band_source = "given on the command line"
        return ResolvedRidge(args.ridge, band, "given on the command line",
                             band_source, "")

    from moe.bench import roofline
    gpu_name = roofline.current_gpu_name()
    try:
        hw = roofline.load_measured(gpu_name or None)
    except roofline.HardwareMismatch as exc:
        raise RidgeUnavailable(str(exc)) from exc
    if hw is not None:
        try:
            ridge = hw.ridge_point(args.dtype)
        except ValueError as exc:
            raise RidgeUnavailable(
                f"{hw.name} has a measured bandwidth but no verified "
                f"{args.dtype} peak, so it cannot state a ridge: {exc}") from exc
        doc = _measured_yaml(gpu_name)
        band, band_source = ridge_band_from_detail(doc.get("detail") or {}, ridge)
        return ResolvedRidge(
            ridge, band,
            f"measured on this device: {hw.name}, "
            f"{hw.peak(args.dtype) / 1e12:.1f} TFLOP/s {args.dtype} over "
            f"{hw.bandwidth_bytes_s / 1e9:.1f} GB/s "
            f"({hw.ceiling_pattern or 'unnamed'} pattern); "
            f"{calibration_stamp_line(doc)}",
            band_source, hw.name or gpu_name)

    if synthetic:
        return ResolvedRidge(RIDGE_BAND[0], RIDGE_BAND, HYPOTHESIS_RIDGE_SOURCE,
                             HYPOTHESIS_RIDGE_SOURCE, gpu_name)

    raise RidgeUnavailable(
        f"no calibration for this device ({gpu_name or 'no CUDA device'}), so "
        "this run has no ridge it is entitled to quote.\n"
        "    Every roof fraction, every AI cap comparison and every crossing "
        "prediction below would be scored against another machine's ceiling: "
        f"the module constant is {RIDGE_BAND[0]} Op/B, which is a 2026-08-26 "
        "H200 figure and belongs to no attached device.\n"
        "    Run:  python scripts/calibrate_hardware.py\n"
        "    or state the assertion yourself:  --ridge <Op/B> "
        "[--ridge-band LO,HI]\n"
        "    off GPU, --dry-run and --self-test may assume the H200 band and "
        "say so in the report.")


def resolve_bandwidth(args) -> tuple[float, str]:
    """This machine's measured bandwidth, or the published H200 figure.

    Only ever used to turn the model into predicted milliseconds. Named with
    its source in the report so a predicted column measured against another
    machine's ceiling cannot pass for one measured against this one.
    """
    if args.bandwidth_gbps:
        return args.bandwidth_gbps, "given on the command line"
    try:
        from moe.bench.roofline import load_measured
        hw = load_measured()
        if hw is not None:
            return hw.bandwidth_bytes_s / 1e9, f"this machine's calibration ({hw.name})"
    except Exception as exc:                            # noqa: BLE001
        return 4374.5, f"published H200 triad ceiling; calibration unreadable ({exc})"
    return 4374.5, "published H200 triad ceiling (no calibration on this box)"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    block_sizes = tuple(int(v) for v in args.tiles.split(","))
    b = dtype_bytes(args.dtype)
    bandwidth, bw_source = resolve_bandwidth(args)
    # RESOLVED BEFORE ANY GPU TIME IS SPENT, and before the plan is printed, so
    # a run that has no ridge it may quote costs nothing and says why.
    try:
        rr = resolve_ridge(args, synthetic=bool(args.dry_run
                                                or args.self_test is not None))
    except RidgeUnavailable as exc:
        print(f"REFUSED: {exc}")
        return 2
    grid = build_grid(cfg, block_sizes, args.r_max, args.row_step,
                      args.step_probes)
    step = rows_step(cfg)
    pinned = dict(FIXED, num_stages=args.num_stages,
                  GROUP_SIZE_M=args.group_m, BLOCK_SIZE_N=args.block_n)

    card = detect_card_slug()
    run_id = args.run_id or default_run_id(args, card)
    out_dir = (args.out or results_root()) / "block_m_crossing" / run_id
    csv_path = out_dir / "cells.csv"
    cache_root = out_dir / "triton-cache"

    print(f"experiment  block_m_crossing / {run_id}")
    print(f"card        {card}"
          + ("   (no CUDA device: a plan or a replay, not a measurement)"
             if card == NO_CARD_SLUG else ""))
    print(f"model       {args.model} E={cfg.num_experts} k={cfg.top_k}  "
          f"{args.dtype} ({b} bytes)")
    print(f"pinned      {pinned}")
    print(f"grid        {len(grid)} rows-per-expert x {len(block_sizes)} block "
          f"sizes = {len(grid) * len(block_sizes)} cells")
    print(f"            r in [{grid[0]}, {grid[-1]}], tokens step {step}, "
          f"T in [{tokens_for_rows(cfg, grid[0])}, "
          f"{tokens_for_rows(cfg, grid[-1])}]")
    print(f"bandwidth   {bandwidth:.1f} GB/s, {bw_source}")
    print(f"ridge       {rr.ridge:.2f} Op/B, {rr.source}")
    print(f"ridge band  {rr.band[0]:.2f}-{rr.band[1]:.2f} Op/B, {rr.band_source}")
    # The run id deliberately does NOT include the ridge: it names the
    # MEASUREMENT, and the ridge changes only the analysis over it. Two runs of
    # the same grid at two ridges must share cells.csv, or the resume path
    # re-measures identical cells. What must not happen is a ridge reaching the
    # report without being named, which is what the two lines above prevent.
    print(f"WRITES TO   {out_dir}")
    print("            cells.csv (appended per cell), report.txt, report.json, "
          "triton-cache/")

    # THE PLAN'S RESOURCE BILL, PRINTED BEFORE ANY TIMING. A setting that cannot
    # hold its accumulator in registers, or its pipeline in shared memory, still
    # RUNS -- it spills, and returns a time proportional to its tile count that
    # sails through the compute reference's shape test. That is how a 249.765 ms
    # tile became this study's compute branch and cost 8 published cells. The
    # setting is refused here, where it is chosen.
    capability = resolve_capability(
        args, synthetic=bool(args.dry_run or args.self_test is not None))
    plan, tile_refusals = tile_resource_plan(pinned, block_sizes, b, capability)
    print("\nTILE RESOURCE PLAN, one CTA, at "
          + (f"sm_{capability[0]}{capability[1]}" if capability
             else "an UNKNOWN device (--capability MAJOR.MINOR gives the "
                  "shared-memory verdict; the register check runs regardless)"))
    for bm in block_sizes:
        print(plan[bm].render())
    if tile_refusals:
        for bm, why in tile_refusals.items():
            print(f"  REFUSED BLOCK_M={bm}: {why}")
        block_sizes = tuple(bm for bm in block_sizes if bm not in tile_refusals)
        if not block_sizes:
            print("REFUSED: every block size in --tiles is unrunnable as "
                  "pinned. Nothing to measure.")
            return 2
        print(f"  sweeping {list(block_sizes)} only. The refused settings are "
              "NOT missing data: they are settings this hardware cannot run, "
              "and a timing taken from one would not be a measurement of the "
              "tiling this sweep is about.")
        # THE GRID IS NOT REBUILT, deliberately. `build_grid` derives its row
        # counts and its step probes from the block sizes, so rebuilding it here
        # would make the SAME run id mean two different grids on two cards --
        # sm_80 refuses this tile where sm_90 does not -- and `cells.csv` is
        # resumed by run id. The refused settings simply contribute no rows.

    if args.dry_run:
        secs = estimated_seconds(cfg, grid, block_sizes, alpha=args.alpha,
                                 ridge=rr.ridge, bandwidth_gbps=bandwidth,
                                 b=b, iters=args.iters, warmup=args.warmup,
                                 cell_budget_ms=args.cell_budget_ms)
        print(f"\nestimated GPU time {secs:.0f} s at the model's own timings, "
              "excluding compiles and allocation")
        preds = predictions(block_sizes, args.alpha, rr.ridge, b)
        for bm in block_sizes:
            p = preds[bm]
            where = ("NO CROSSING EVER" if p.crossing_rows is None else
                     f"crosses at r={p.crossing_rows:.1f} "
                     f"(T={p.crossing_tokens(cfg.num_experts, cfg.top_k):.0f}), "
                     f"in the grid: {p.crossing_rows <= args.r_max}")
            print(f"  BLOCK_M={bm:3d} cap {p.ai_cap:7.1f}  {where}")
        return 0

    if args.self_test is None:
        missing = missing_gpu_stack()
        if missing:
            print("\n" + missing)
            return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test is not None:
        alpha = args.self_test
        sm_count = args.sm_count or DEFAULT_SM_COUNT
        cells = synthetic_cells(cfg, grid, block_sizes, alpha=alpha,
                                ridge=rr.ridge, bandwidth_gbps=bandwidth, b=b,
                                sm_count=sm_count, noise=args.self_test_noise,
                                seed=args.seed)
        compiles = {bm: 1 for bm in block_sizes}
        executed = dict(compiles)
        print(f"\nSELF TEST: cells GENERATED from the model at alpha={alpha}. "
              "Nothing here was measured.")
        print("The gates below are being run against a world we constructed, "
              "which tests the gates and not the hardware.")
    else:
        import torch
        alpha = args.alpha
        sm_count = args.sm_count or torch.cuda.get_device_properties(0).multi_processor_count
        started = time.time()
        cells, compiles, executed = run_sweep(
            args, cfg, grid, block_sizes, csv_path, cache_root, b, pinned)
        print(f"\nswept in {time.time() - started:.0f} s")

    sm_source = ("given on the command line" if args.sm_count
                 else "reported by the driver" if args.self_test is None
                 else f"assumed H200 default {DEFAULT_SM_COUNT}")
    report = analyse(cells, cfg, block_sizes=block_sizes, alpha=alpha,
                     ridge=rr.ridge, bandwidth_gbps=bandwidth, b=b,
                     model_name=args.model, dtype=args.dtype, compiles=compiles,
                     executed=executed, sm_count=sm_count, sm_source=sm_source,
                     pinned=pinned, ridge_band=rr.band, ridge_source=rr.source,
                     ridge_band_source=rr.band_source, capability=capability,
                     card=card)
    print(report.text())

    (out_dir / "report.txt").write_text(report.text())
    (out_dir / "report.json").write_text(json.dumps(report.payload, indent=2))
    print(f"cells    {csv_path}")
    print(f"report   {out_dir / 'report.txt'}")
    print(f"json     {out_dir / 'report.json'}")
    print("These survive pod teardown when the results root is on the network "
          "volume, which is what the WRITES TO line above says.")

    if args.fail_on_gate and any(g.verdict != PASS for g in report.gates):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
