#!/usr/bin/env python
"""Which BLOCK_SIZE_M can reach the compute roof at all, and where does it cross?

    python scripts/block_m_crossing_sweep.py --self-test 0.558   # no GPU needed
    python scripts/block_m_crossing_sweep.py --self-test 0.10    # the retracted world
    python scripts/block_m_crossing_sweep.py --self-test 0.90    # nothing crosses
    python scripts/block_m_crossing_sweep.py --self-test-world low-clock
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

ONE INSTRUMENT, AND WHAT IT COSTS TO HAVE HAD TWO. Until 2026-09-02 every cell
here was timed by a private `time_call` that created its CUDA events inside the
loop, recorded the start event on a stream it had just synchronised, and
synchronised again after every iteration, with no L2 flush and no clock read.
The roof every one of those cells is scored against was measured queue-deep
(`moe/bench/timing.time_eager`). The audit bounded the host prefix the old loop
exposed at 0.18 ms per fused_experts call on the H200 and 0.30 ms on the A100,
a bias in alpha of 8-16% at the smallest ladder cells and DIFFERENT PER CARD,
which is the same size as the cross-card effect this study registered. Every
cell is now timed by `moe.bench.timing.time_kernel` under `TIMING_BASIS`, and
the instrument name, the warmup duration, the iteration and trial counts, the
SM clock sampled UNDER LOAD, its two verdicts and the flush state are columns on
every row. A cell whose loaded clock came in below the clock the roof was
measured at is EXCLUDED from the ladder fit and counted, because a tread timed
on a throttled card is a tread at a different compute branch.

EXIT CODES AND THE ONE GREPPABLE LINE. `moe.bench.exit_codes` owns both. Every
scored gate prints exactly one `RESULT: KIND NAME VERDICT detail` line, the
process exit code comes from `exit_codes.classify` over the same gates, and a
refusal exits REFUSED before anything is measured. Nothing else in the output is
a gate result.

OFF-GPU. `--self-test ALPHA` generates the cells from the physical model at that
alpha and runs the entire analysis on them, so the gates, the fits and the
report are exercised on a laptop, and so the claim "these gates can tell 0.558
from 0.10" is checkable rather than asserted. `--self-test-world` plants the two
worlds that are not a single alpha: a low-clock tread, which must be excluded,
and a memory branch parallel to the compute branch, which must come out
UNDECIDED. `--dry-run` prints the grid, the predictions and the cost estimate
without touching a GPU. Absent torch, CUDA or vLLM the script says which one is
missing and what to run instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench import ai_model, exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.spec import MODEL_CONFIGS, dtype_bytes  # noqa: E402

#: `moe.bench.timing` is imported LAZILY, everywhere, and this comment is the
#: reason. That module imports torch at module scope; this one is documented to
#: run `--dry-run` and `--self-test` on a laptop with no torch at all, and an
#: import here would turn that documented path into an ImportError before
#: argparse ever ran. `exit_codes`, `provenance` and `ai_model` import nothing
#: heavier than the standard library, so they are imported normally above.


def timing_basis() -> str | None:
    """The name of the instrument this file times with, or None off-torch.

    None is not a default: it says the instrument could not be NAMED on this
    machine because torch is absent, which is exactly the laptop `--dry-run` and
    `--self-test` case where nothing was measured either. Every row a pod
    produces carries the string, and a row without it is a row from before
    2026-09-02 or a row from a laptop.
    """
    try:
        from moe.bench.timing import TIMING_BASIS
    except Exception:                                     # noqa: BLE001
        # Broad: a torch that is INSTALLED and broken raises OSError on a
        # missing libcudart rather than ImportError (moe/bench/provenance.py
        # records the same pod failure), and naming the instrument is never
        # worth taking the whole report down for.
        return None
    return TIMING_BASIS

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
#:
#: WHAT THE DRIFT WITH BLOCK_M IS, since this used to be filed as an unexplained
#: wobble that "cannot be pinned". These are `LadderFit.alpha` readings, and
#: that estimator is B/(A+B): the per-tile slope over the fitted level at one
#: tile. `moe/bench/ai_model.py` derives what that returns on three-term traffic
#: and the form has a name, EXA:
#:
#:     alpha_fitted = (alpha_b + phi) / (1 + phi + delta)
#:
#: with `alpha_b` the weight miss fraction the study NAMES, `phi` one M-tile's
#: activation-and-output cost in units of one full weight read, and `delta` the
#: fused layer's fixed cost in the same units. `phi` grows with BM/BN -- an
#: extra M-tile re-reads activations per N-tile, in exactly that ratio -- so
#: alpha_fitted rises with BLOCK_M at fixed BLOCK_N even when alpha_b does not
#: move at all. At BN=64 on mixtral, phi is 0.16 at BM=64 and 0.32 at BM=128,
#: which is the direction and roughly the size of the 0.466 -> 0.625 step here.
#: So these two numbers are FITTED QUANTITIES of a particular tiling, not two
#: measurements of one miss fraction that disagree, and 0.625 is not "a larger
#: fraction of the weights missed". Reading either as a miss fraction is what
#: `ai_model.alpha_b_from_fitted` exists to undo, and any cap taken as
#: 2*BM/(alpha_fitted*b) is HIGH by `ai_model.lin_overstatement` = 1 + phi +
#: delta, which this report prints beside every such cap.
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
#: by this study, so no gate may be phrased as though one was.
#:
#: RETIRED AS A VERDICT ON 2026-09-02, KEPT AS AN INFORMATIONAL LINE. Scoring
#: `alpha_hat > 0.33` is ONE-SIDED, and one-sided against the value the study
#: had already retracted. Every one of the 41 committed surface fits reads
#: 0.6-1.0 and every one of them passes this, while NOT ONE is within 0.05 of
#: the 0.558 the same reports predict; the one published sweep measured
#: 0.989/0.923 against a prediction of 0.558 and printed "it is where the refit
#: put it". A test that a pre-registered number cannot fail from above is not a
#: test of that number. `gate_3_alpha_discriminates` now scores the fitted
#: alpha's interval against `ALPHA_BAND` in BOTH directions and prints this
#: threshold beside the verdict as the retired discriminator it is.
GATE3_ALPHA_DISCRIMINATOR = 0.33

#: Resamples in the memory-branch bootstrap. 400 is enough for a 90% percentile
#: interval to be stable to about a third of a percent of alpha, and the whole
#: bootstrap runs in milliseconds on a ladder of at most a few dozen treads, so
#: there is nothing to buy by cutting it.
ALPHA_BOOTSTRAP_TRIALS = 400

#: Coverage of the percentile interval the bootstrap reports. 90% rather than
#: 95% because the interval is UNIONED with the two systematic ends below and a
#: wider random part would mostly widen an interval that is already dominated by
#: the systematic bracket at low noise.
ALPHA_BOOTSTRAP_COVERAGE = 0.90

#: Treads the bootstrap needs before it will resample at all. Below this a
#: resample draws the same one or two points over and over and reports an
#: interval of width zero, which would read as a precise measurement. The same
#: floor `MIN_MEMORY_TREADS` sets for quoting an alpha, for the same reason.
MIN_BOOTSTRAP_TREADS = 3

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

    THE STATE THE CELL WAS TIMED IN IS A COLUMN, not a property of the session.
    `instrument`, `warmup_ms`, `iters`, `trials`, `sm_clock_load_mhz`,
    `clock_level_ok`, `clock_drift_ok` and `l2_flush` are what
    `moe.bench.timing.time_kernel` reports about the measurement it just made,
    and they are written per row because they are what makes a row comparable
    with the roof or not. Before 2026-09-02 none of them existed here: the
    published cells carry an iteration count and nothing else, so a reader
    cannot tell a cell timed on a card at 1980 MHz from one timed at 1500, and
    every one of them was timed by a different instrument from the roof.

    THE THREE CLOCK FIELDS ARE OPTIONAL AND None MEANS "NOT DETERMINED", never
    "fine". A container without NVML, a trial too short for the poller to land a
    sample, and a laptop replay all produce None, and a filter that reads None
    as True would quietly re-admit exactly the rows this column exists to keep
    out. `clock_level_ok is False` is the only state `ladder_treads` excludes.
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
    #: `moe.bench.timing.TIMING_BASIS` of the loop that produced `ms_p50`.
    #: Empty means a row from before the instrument had a name.
    instrument: str = ""
    #: Milliseconds of DELIVERED GPU load the warmup ran for, not a call count.
    warmup_ms: float = 0.0
    trials: int = 0
    sm_clock_load_mhz: float | None = None
    clock_level_ok: bool | None = None
    clock_drift_ok: bool | None = None
    l2_flush: bool = False

    @property
    def rel_spread(self) -> float:
        return self.ms_stdev / self.ms_p50 if self.ms_p50 > 0 else 0.0

    @property
    def clock_excluded(self) -> bool:
        """Was this cell timed below the clock the roof was measured at.

        False for None on purpose, and this is the one place that reading is
        correct: an EXCLUSION has to be positively established. A row with no
        clock is a row whose comparability is unknown, and the report says how
        many of those there are rather than throwing them away.
        """
        return self.clock_level_ok is False


def make_cell(cfg, rows: float, block_m: int, ms: float, *, sm_count: int,
              block_n: int, ms_min: float = 0.0, ms_stdev: float = 0.0,
              iters: int = 0, status: str = "ok", detail: str = "",
              instrument: str = "", warmup_ms: float = 0.0, trials: int = 0,
              sm_clock_load_mhz: float | None = None,
              clock_level_ok: bool | None = None,
              clock_drift_ok: bool | None = None,
              l2_flush: bool = False) -> Cell:
    """One cell, with the state it was timed in.

    SIGNATURE EXTENDED 2026-09-02, never narrowed: every new argument has a
    default and the four sibling scripts that call this
    (`bm128_roofline`, `bm128_depth`, `bn_decomposition`, `occupancy_vs_swizzle`)
    keep working unchanged. A cell built without them carries the empty
    instrument and three None clock verdicts, which is the honest record of a
    row whose timing state nobody wrote down.
    """
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
        status=status, detail=detail, instrument=instrument,
        warmup_ms=warmup_ms, trials=trials,
        sm_clock_load_mhz=sm_clock_load_mhz, clock_level_ok=clock_level_ok,
        clock_drift_ok=clock_drift_ok, l2_flush=l2_flush)


#: What the synthetic cells claim as their instrument. A planted world is not a
#: measurement and must never carry `TIMING_BASIS`: a reader who greps a
#: cells.csv for the instrument has to be able to tell a pod row from a
#: generated one, and a self-test that stamped the real basis on its own
#: fabrications would make that impossible.
SYNTHETIC_INSTRUMENT = "synthetic/model-generated/not-measured"

#: Clock the synthetic cells claim to have run at, and the reference they are
#: scored against. Two numbers rather than one so the low-clock world below can
#: move the first without moving the second.
SYNTHETIC_CLOCK_MHZ = 1980.0


def synthetic_cells(cfg, grid, block_sizes, *, alpha: float, ridge: float,
                    bandwidth_gbps: float, b: int, sm_count: int,
                    overhead_ms: float = 0.03, noise: float = 0.0,
                    seed: int = 0, low_clock: tuple[int, int] | None = None,
                    warmup_ms: float = 0.0, trials: int = 0,
                    l2_flush: bool = True) -> list[Cell]:
    """Cells generated FROM the model, so the analysis has a known answer.

    This is what makes the whole report testable on a laptop, and it is what
    makes "these gates can tell 0.558 from 0.10" a check rather than a claim:
    generate at one alpha, read the gates, generate at the other, read them
    again. `noise` multiplies each cell by a lognormal draw so the gates are
    exercised against spread and not only against a clean curve.

    THE TIMING COLUMNS ARE PLANTED TOO, and that is not decoration. A cell whose
    `clock_level_ok` is False is excluded from every ladder fit, and an
    exclusion path that only ever runs on a pod is a path nobody has watched
    work. `low_clock=(BLOCK_M, tiles)` plants exactly one tread that failed the
    LEVEL flag, so the self-test can assert it was dropped, counted and named.
    Everything else carries `clock_level_ok=True` and `SYNTHETIC_INSTRUMENT`,
    which is deliberately NOT `TIMING_BASIS`: these rows were not measured.
    """
    rng = random.Random(seed)
    out = []
    for bm in block_sizes:
        for r in grid:
            ms = model_ms(cfg, r, bm, alpha=alpha, ridge=ridge,
                          bandwidth_gbps=bandwidth_gbps, b=b,
                          overhead_ms=overhead_ms)
            if noise:
                ms *= math.exp(rng.gauss(0.0, noise))
            slow = (low_clock is not None and bm == low_clock[0]
                    and tiles_per_expert(r, bm) == low_clock[1])
            out.append(make_cell(
                cfg, r, bm, ms, sm_count=sm_count,
                block_n=FIXED["BLOCK_SIZE_N"], ms_min=ms,
                ms_stdev=ms * noise, iters=0,
                instrument=SYNTHETIC_INSTRUMENT, warmup_ms=warmup_ms,
                trials=trials, l2_flush=l2_flush,
                sm_clock_load_mhz=(SYNTHETIC_CLOCK_MHZ * 0.7 if slow
                                   else SYNTHETIC_CLOCK_MHZ),
                clock_level_ok=not slow, clock_drift_ok=True))
    return out


# --------------------------------------------------------------------------
# The ladder: time per tread, which is where every gate but the first is read.
# --------------------------------------------------------------------------

#: What a ladder fit CONCLUDED, as a fixed token. Six states, and the whole
#: point of having six is that the four that are not `IDENTIFIED` used to arrive
#: as the same blank. `crosses=None` was printed for the BLOCK_M=128 row of
#: every H200 arm and read as "the sweep lacked treads", when the causes were
#: variously a membership rule anchored at n=1, a memory branch parallel to the
#: compute branch, and a compute reference 44x too steep.
IDENTIFIED = "identified"
#: The treads were there, the branch was fitted, and its slope is within
#: `PARALLEL_BRANCH_TOLERANCE` of the compute branch. `B / C = ridge / ai_cap`,
#: so this says the tile's cap sits ON the ridge: the sweep cannot decide, and a
#: roofline arm at this tile can. NOT None, and never imported over.
UNDECIDED_PARALLEL_BRANCH = "undecided_parallel_branch"
#: Enough treads were dropped for clock level that what remains is below
#: `MIN_MEMORY_TREADS`. The card, not the tile, is what this ladder measured.
UNDECIDED_LOW_CLOCK = "undecided_low_clock"
#: Fewer memory-bound treads than `MIN_MEMORY_TREADS`, with nothing excluded and
#: no reference problem: the tile really is compute bound this early.
NOT_IDENTIFIED_TOO_FEW = "too_few_memory_treads"
#: This ladder IS the compute reference, so by the assumption that qualified it
#: there is no memory branch here to fit.
NOT_IDENTIFIED_IS_REFERENCE = "is_the_reference_ladder"
#: No usable tread at all.
NOT_IDENTIFIED_NO_TREADS = "no_usable_treads"

LADDER_OUTCOMES = (IDENTIFIED, UNDECIDED_PARALLEL_BRANCH, UNDECIDED_LOW_CLOCK,
                   NOT_IDENTIFIED_TOO_FEW, NOT_IDENTIFIED_IS_REFERENCE,
                   NOT_IDENTIFIED_NO_TREADS)


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
    #: One of `LADDER_OUTCOMES`. `IDENTIFIED` is the only one that entitles the
    #: ladder's alpha to decide anything; every other value is a NAMED failure
    #: to identify, and the difference between them is what a reader needs.
    #: Defaulted so the four sibling scripts that build a `LadderFit` through
    #: `fit_ladder` keep working without naming it.
    outcome: str = ""
    #: Prose for `outcome`, empty when the outcome is `IDENTIFIED`. This is the
    #: sentence the report and report.json both print, so the text a reader sees
    #: and the text a table generator reads cannot drift apart.
    outcome_reason: str = ""
    #: Cells dropped from this ladder because their loaded clock was below the
    #: clock the roof was measured at. Counted rather than silently missing.
    excluded_low_clock: int = 0
    #: Index into `points` of the FIRST tread on the memory branch. Non-zero
    #: means the lowest tread(s) sat inside the margin and the branch starts
    #: above them; see `memory_branch_members` for why that is allowed and why
    #: the n=1 tread is the one it usually happens to.
    branch_start: int = 0

    @property
    def undecided(self) -> bool:
        """The two states where the sweep looked and could not say.

        Distinct from "not identifiable for want of treads": UNDECIDED means the
        treads were there and the QUESTION could not be settled by this
        instrument, which is a different thing to report and points at a
        different next experiment.
        """
        return self.outcome in (UNDECIDED_PARALLEL_BRANCH, UNDECIDED_LOW_CLOCK)

    @property
    def load_ms(self) -> float | None:
        """`L = A + B`, one full weight read, the value the memory branch takes
        at a single tile."""
        if self.intercept is None or self.slope_memory is None:
            return None
        return self.intercept + self.slope_memory

    @property
    def alpha(self) -> float | None:
        """`B / (A + B)`, the estimator this study calls alpha. EXA, not a miss
        fraction.

        WHAT THIS NUMBER IS. `moe/bench/ai_model.py` runs this exact estimator
        over three-term traffic and reports what comes back:

            alpha = B/(A+B) = (alpha_b + phi) / (1 + phi + delta)          (EXA)

        `alpha_b` is the weight miss fraction the study NAMES; `phi` is one
        M-tile's activation-and-output cost in units of one full weight read;
        `delta` is the fused layer's fixed cost in the same units. The level
        `A + B` is one full weight read PLUS one tile's activations and output
        PLUS the fixed cost, and every part of that lands in the denominator.
        So this is a FITTED QUANTITY of a particular tiling, not a fraction of
        the weights that missed, and it can be read as one only after
        `ai_model.alpha_b_from_fitted` has taken phi and delta back out.

        THE TWO BIASES, both named, because they are the two extra terms above
        and their signs differ. `delta` is inside the level, which pushes this
        DOWN; `phi` is inside both the slope and the level, and since phi < 1
        its net effect through the slope pushes this UP. Only the activation
        part of phi is correctable per block size, by `activation_slope_ms`, and
        the report prints that correction beside this number as
        `alpha-corrected`. What is NOT done is subtracting an extrapolated fixed
        cost first: on a 4-tread reference ladder under 1% timing spread that
        extrapolation wandered enough to move alpha from 0.56 to 0.70 on data
        planted at 0.558. `alpha_upper` carries that end, and no gate is scored
        on it alone; `alpha_interval` scores both ends at once.

        WHAT A CAP FROM THIS NUMBER COSTS. `2 BM / (alpha b)` treats the level
        as one full weight read, which it is not, so a cap taken that way is
        HIGH by `ai_model.lin_overstatement` = 1 + phi + delta: 16% at BM=64,
        32% at BM=128 and 64% at BM=256 with BN=64 on mixtral. At BM=128, the
        tile this study's claim is about, that is larger than the cap-to-ridge
        gap the cap was being used to decide. The report prints the factor
        beside every cap it derives from this number.
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


def ladder_treads(cells, block_m: int) -> tuple[list[tuple[int, float]], int]:
    """`(points, cells excluded for clock level)` at exactly-full tile stacks.

    Aligned only. A tread sampled at 60% fill reports the same TIME as its top
    -- time is flat along a tread -- but a different throughput, and mixing the
    two is how a padding artefact enters a fit that is about traffic.

    AND A CELL WHOSE LOADED CLOCK CAME IN LOW IS NOT ON THIS LADDER. Membership
    is decided against a compute branch `C = 2 BM N / peak`, and `peak` is the
    roof measured at the calibration's own clock. A tread timed on a card at
    1500 MHz against a roof measured at 1980 sits about 30% above that line for
    a reason that has nothing to do with weight re-reads, and it would be read
    as memory bound and fitted into alpha. `clock_level_ok is False` is the only
    exclusion; None (no NVML, too short a trial, a laptop replay) is NOT an
    exclusion, because an exclusion has to be positively established, and the
    count returned here is what the report says out loud so a ladder that lost
    half its treads to a hot box cannot look like a ladder that never had them.
    """
    pts: dict[int, float] = {}
    excluded = 0
    for c in cells:
        if c.block_m != block_m or not c.aligned or c.status != "ok" or c.ms_p50 <= 0:
            continue
        if c.clock_excluded:
            excluded += 1
            continue
        pts[c.tiles_per_expert] = c.ms_p50
    return sorted(pts.items()), excluded


def ladder_points(cells, block_m: int) -> list[tuple[int, float]]:
    """`ladder_treads` without the exclusion count. The four sibling scripts
    call this name and take a bare list; it stays exactly that."""
    return ladder_treads(cells, block_m)[0]


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


def memory_branch_members(xs, ys, c_ref: float, overhead: float,
                          margin: float) -> tuple[int, int, list[bool]]:
    """`(start index, count, per-tread verdicts)` of the memory branch.

    THE RULE THAT REPLACED THE PREFIX RULE ON 2026-09-02, and the row it was
    costing. A tread is memory bound when it stands more than `margin` above
    `overhead + C n`. The old rule then took the LEADING RUN FROM n=1 and
    stopped at the first tread that did not qualify, so a single tread at n=1
    inside the margin set `k = 0` and threw the entire branch away. In the H200
    mixtral G=1 arm the n=1 tread sits 3.30% above the scaled compute line
    against a 4.20% margin -- a miss of 0.003 ms -- while treads 2 through 8 all
    sit 7.3-9.3% above it. So `k = 0`, alpha was imported from BLOCK_M=64, and
    `crosses` came out None for BLOCK_M=128 in EVERY arm: the one tile the
    paper's claim is about has never been identified by its own ladder.

    WHY n=1 IS THE TREAD THAT GOES MISSING, and why it is right to let the
    branch start above it. The n=1 cell is the shortest kernel in the ladder and
    therefore the one carrying the largest HOST prefix as a fraction of its
    time: the audit bounded that prefix at 0.18 ms per fused_experts call on the
    H200 and 0.30 ms on the A100, which is 8-16% of a small tread and is a
    per-card constant, not traffic. It is also the tread whose elevation is
    G-dependent in a way the constant prefix does not explain, which is the
    thing `scripts/memory_branch_anchor.py` exists to measure. A rule that lets
    that one tread veto the other seven is a rule that hands the answer to the
    least trustworthy point on the ladder.

    So the branch is now the CONTIGUOUS run of memory-bound treads starting at
    the LOWEST such n. Contiguity is kept because memory-boundness really is a
    prefix property of the underlying curve -- `Q` grows by alpha per tile and
    the compute branch by 1, so once compute is on top it stays there -- and a
    scattered subset would be noise picking its own points. What is dropped is
    only the requirement that n=1 be in the run. Treads BELOW the run's start
    are neither on the branch nor evidence against it, and the caller is told
    where the run began.
    """
    above = [y > overhead + c_ref * x * (1.0 + margin)
             for x, y in zip(xs, ys, strict=True)]
    if not any(above):
        return 0, 0, above
    start = above.index(True)
    count = 0
    for flag in above[start:]:
        if not flag:
            break
        count += 1
    return start, count, above


def fit_ladder(points, block_m: int, ref: ComputeReference | None = None,
               margin: float = MEMORY_BRANCH_MARGIN,
               excluded_low_clock: int = 0) -> LadderFit:
    """Split the ladder into a memory branch and a compute branch.

    Membership comes from the reference compute branch when there is one, by the
    majority-and-contiguity rule `memory_branch_members` documents: the memory
    branch is the run of treads standing more than `margin` above `C n`,
    starting wherever the lowest such tread is rather than at n=1.

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

    SIGNATURE EXTENDED, never narrowed. `excluded_low_clock` is optional and
    defaults to 0, so the four sibling scripts that call
    `SWEEP.fit_ladder(points, bm, ref[, margin])` are unaffected; pass it and
    the fit can tell a ladder that never had treads from one whose treads were
    dropped for clock level, which is the difference between
    `NOT_IDENTIFIED_TOO_FEW` and `UNDECIDED_LOW_CLOCK`.
    """
    overhead = ref.overhead_ms if ref else 0.0
    pts = [(n, ms) for n, ms in points if ms > 0]
    if not pts:
        return LadderFit(block_m, tuple(points), 0, None, None, None, None,
                         math.inf, overhead, "no usable treads",
                         outcome=NOT_IDENTIFIED_NO_TREADS,
                         outcome_reason=(
                             "no exactly-full tile stack at this block size "
                             "carried a usable time"
                             + (f"; {excluded_low_clock} cell(s) were excluded "
                                "for clock level" if excluded_low_clock else "")),
                         excluded_low_clock=excluded_low_clock)
    xs = [float(n) for n, _ in pts]
    ys = [ms for _, ms in pts]
    c_ref = ref.slope_for(block_m) if ref else None
    start = 0

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
        start, k, _above = memory_branch_members(xs, ys, c_ref, overhead, margin)
        basis = (f"membership from the compute branch scaled off "
                 f"BLOCK_M={ref.block_m}"
                 + (f"; the branch starts at tread n={int(xs[start])}, not at "
                    "the first tread, which sits inside the margin"
                    if start else ""))
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

    stop = start + k
    a = b = None
    if k >= 2:
        a, b = _line(xs[start:stop], ys[start:stop])
    c_own = (_through_origin(xs[stop:], [y - overhead for y in ys[stop:]])
             if stop < len(pts) else None)
    c_eff = c_own if c_own else c_ref
    outcome, reason = "", ""
    if b is not None and c_eff and abs(b / c_eff - 1.0) <= PARALLEL_BRANCH_TOLERANCE:
        # A memory branch parallel to the compute branch is not a second
        # mechanism. `B / C = ridge / ai_cap`, so this says the ceiling sits on
        # the ridge -- or, far more often, that the fit ran the prefix into the
        # compute branch and is about to report that branch's slope as alpha.
        #
        # THE OUTCOME IS UNDECIDED AND IS NAMED. Until 2026-09-02 this branch
        # produced `alpha=None, crosses=None`, which downstream read as "the
        # sweep lacked treads" and let gate 3 IMPORT an alpha from another block
        # size over the top of it. Those are three different statements and only
        # one of them is true here: the ladder was measured, the branch was
        # fitted, and its slope sits on the ridge to within the tolerance. That
        # question is answerable, just not by this sweep.
        gap = abs(b / c_eff - 1.0)
        a, b = None, None
        k = 0
        outcome = UNDECIDED_PARALLEL_BRANCH
        reason = (
            f"UNDECIDED at BLOCK_M={block_m}: the tile's cap is within "
            f"{gap:.1%} of the ridge (the fitted memory slope B and the compute "
            f"slope C differ by that much, and B/C = ridge/ai_cap exactly), "
            f"inside the {PARALLEL_BRANCH_TOLERANCE:.0%} tolerance. This sweep "
            "cannot decide whether this tile crosses; a roofline arm at this "
            "tile can, because it measures the roof fraction directly instead "
            "of separating two slopes that are the same line. This is NOT a "
            "shortage of treads and NOT a null, and no alpha may be imported "
            "over it.")
        basis += (f"; memory branch DISCARDED, its slope was within "
                  f"{PARALLEL_BRANCH_TOLERANCE:.0%} of the compute branch and "
                  "the two are then the same line")
    err = _max_affine_error(xs, ys, a, b, c_eff, overhead)
    made = LadderFit(block_m, tuple(points), k, a, b, c_own, c_ref, err,
                     overhead, basis, outcome=outcome, outcome_reason=reason,
                     excluded_low_clock=excluded_low_clock,
                     branch_start=start if k else 0)
    if outcome:
        return made
    outcome, reason = _ladder_outcome(made, ref, excluded_low_clock)
    return LadderFit(block_m, tuple(points), k, a, b, c_own, c_ref, err,
                     overhead, basis, outcome=outcome, outcome_reason=reason,
                     excluded_low_clock=excluded_low_clock,
                     branch_start=start if k else 0)


def _ladder_outcome(fit: LadderFit, ref, excluded: int) -> tuple[str, str]:
    """Name what this ladder concluded, in the vocabulary of `LADDER_OUTCOMES`.

    Order matters. The clock exclusion is tested BEFORE the tread count,
    because a ladder that lost treads to a hot box and a ladder that never had
    them report the same count and mean opposite things: the first says the
    card was not at the roof's clock and the arm should be re-timed, the second
    says the tile really is compute bound this early. Reporting them as one
    number is how a throttled session becomes a physical finding.
    """
    if fit.memory_points >= MIN_MEMORY_TREADS and fit.alpha is not None:
        return IDENTIFIED, ""
    if excluded and fit.memory_points < MIN_MEMORY_TREADS:
        return UNDECIDED_LOW_CLOCK, (
            f"UNDECIDED at BLOCK_M={fit.block_m}: {excluded} cell(s) were "
            "excluded because their SM clock under load came in below the clock "
            "the roof was measured at, leaving "
            f"{fit.memory_points} memory-bound tread(s) against the "
            f"{MIN_MEMORY_TREADS} a verdict needs. What this ladder measured is "
            "the card's clock state, not the tile. Re-time the arm on a settled "
            "card; do not read the shortfall as a property of the tiling.")
    if ref is not None and ref.block_m == fit.block_m:
        return NOT_IDENTIFIED_IS_REFERENCE, (
            f"BLOCK_M={fit.block_m} is the compute reference, so by the "
            "assumption that qualified it there is no memory branch here to fit")
    return NOT_IDENTIFIED_TOO_FEW, (
        f"BLOCK_M={fit.block_m} has {fit.memory_points} memory-bound tread(s), "
        f"under the {MIN_MEMORY_TREADS} a verdict needs")


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


def cap_overstatement(cfg, block_m: int, block_n: int, b: int
                      ) -> tuple[float, float]:
    """`(lo, hi)` on `ai_model.lin_overstatement` for this tile. R8's label.

    THE FACTOR EVERY CAP IN THIS REPORT IS HIGH BY. `ai_cap` computes
    `2 BM / (alpha b)` from a FITTED alpha, and `moe/bench/ai_model.py` shows
    that a B/(A+B) fit returns `(alpha_b + phi) / (1 + phi + delta)`, so the cap
    that division gives is the exact cap times `1 + phi + delta`. At BM=128,
    BN=64 on mixtral that is about 1.32: a 32% overstatement, larger than the
    cap-to-ridge gap the cap is being used to decide.

    IT IS A BRACKET AND NOT A NUMBER, and the reason is the honest one:
    `alpha_a`, the miss fraction on the ACTIVATION re-read, has no measurement
    anywhere in this repository. `phi` depends on it, so this returns the factor
    at alpha_a = 0 and at alpha_a = 1 -- no re-read and a full one -- and the
    report prints both ends. `delta`, the fused layer's fixed cost, is taken as
    zero here, which makes both ends LOWER bounds on the overstatement; the
    report's own `overhead_ms` is the measured stand-in for delta and is printed
    beside them.

    The single-GEMM shape this is evaluated on is the up-projection,
    `N = 2 F` and `K = H`, because that is the GEMM whose B operand is the
    weight slab whose re-read the whole study is about.
    """
    n, k = 2 * cfg.intermediate_size, cfg.hidden_size
    ends = []
    for alpha_a in (0.0, 1.0):
        p = ai_model.phi(n, k, block_m=block_m, block_n=block_n,
                         alpha_a=alpha_a, b=b)
        ends.append(ai_model.lin_overstatement(phi=p, delta=0.0))
    return min(ends), max(ends)


def cap_note(cfg, block_m: int, block_n: int, b: int) -> str:
    """One labelled clause naming the overstatement beside a cap."""
    lo, hi = cap_overstatement(cfg, block_m, block_n, b)
    return (f"cap is 2*BM/(alpha*b) from a FITTED alpha and is HIGH by "
            f"ai_model.lin_overstatement = 1+phi+delta, {lo:.2f}-{hi:.2f}x here "
            f"(alpha_a unmeasured, delta taken as 0, so a lower bound)")


@dataclass(frozen=True)
class AlphaInterval:
    """What a fitted alpha is worth, as an interval, so it can be tested BOTH ways.

    A one-sided gate needs only a point estimate. A two-sided one needs to know
    how wide the estimate is, and this carries the two independent widths that
    apply, kept separate because they answer different objections:

      * RANDOM. `boot_lo`/`boot_hi` are a percentile interval from resampling
        the memory branch's own treads with replacement and refitting. It is
        zero-width on noiseless synthetic cells, which is correct: nothing about
        those treads is random.
      * SYSTEMATIC. `sys_lo`/`sys_hi` are the two ends the report has always
        printed. `alpha-corrected` subtracts the activation traffic that inflates
        the slope; `alpha-hi` subtracts the fused layer's fixed cost that
        inflates the level. `LadderFit.alpha` sits between them BY CONSTRUCTION
        and the two corrections point in opposite directions, so the pair is a
        bracket rather than an error bar.

    `lo` and `hi` are the union, and the union is the honest object to score a
    pre-registered band against: an interval that carried only the random part
    would be a two-sided test resting on the assumption that both named biases
    are zero, which is the assumption the two corrections exist because it is
    false. On mixtral at BN=64 the systematic bracket alone is about 0.05 wide
    at alpha 0.55, which is the size of `ALPHA_BAND` itself.
    """

    point: float
    lo: float
    hi: float
    sys_lo: float
    sys_hi: float
    boot_lo: float | None
    boot_hi: float | None
    treads: int
    trials: int

    def overlaps(self, band: tuple[float, float]) -> bool:
        return self.lo <= max(band) and min(band) <= self.hi

    def direction(self, band: tuple[float, float]) -> str:
        """"ABOVE", "BELOW" or "" for an interval that overlaps the band."""
        if self.overlaps(band):
            return ""
        return "ABOVE" if self.lo > max(band) else "BELOW"

    def render(self) -> str:
        boot = ("no bootstrap: fewer than "
                f"{MIN_BOOTSTRAP_TREADS} treads to resample"
                if self.boot_lo is None
                else f"bootstrap [{self.boot_lo:.3f}, {self.boot_hi:.3f}] over "
                     f"{self.trials} resamples of {self.treads} treads")
        return (f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]  "
                f"(systematic [{self.sys_lo:.3f}, {self.sys_hi:.3f}]: "
                f"activation-corrected to fixed-cost-corrected; {boot})")


def alpha_interval(fit: LadderFit, cfg, bandwidth_gbps: float, *,
                   trials: int = ALPHA_BOOTSTRAP_TRIALS,
                   coverage: float = ALPHA_BOOTSTRAP_COVERAGE,
                   seed: int = 0) -> AlphaInterval | None:
    """The fitted alpha with both its widths, or None when it has no alpha.

    The bootstrap resamples the memory branch's `(tiles, ms)` treads WITH
    REPLACEMENT and refits the line on each draw, which is the resampling that
    matches how the estimate was made: the treads are the observations and the
    line is the statistic. Draws that land on fewer than two distinct treads are
    discarded rather than fitted, because `_line` returns a zero slope on a
    degenerate x and a zero slope is an alpha of zero, which would drag the low
    end of the interval to somewhere no ladder ever suggested.

    The point estimate is the ACTIVATION-CORRECTED alpha, the same column gate 3
    has always scored, so the interval is centred on the number the report
    prints rather than on a second quantity computed only here.
    """
    if fit.alpha is None or fit.slope_memory is None or not fit.load_ms:
        return None
    act = activation_slope_ms(cfg, fit.block_m, bandwidth_gbps)
    point = (fit.slope_memory - act) / fit.load_ms
    upper = fit.alpha_upper
    sys_lo, sys_hi = (point, point if upper is None else max(point, upper))

    treads = [(n, ms) for n, ms in fit.points if ms > 0]
    # The branch is the run the fit used, and `branch_start` says where it
    # began: the run may start above the lowest tread, so slicing from zero
    # would resample points the fit deliberately left off the branch.
    branch = (treads[fit.branch_start:fit.branch_start + fit.memory_points]
              if fit.memory_points else [])
    boot_lo = boot_hi = None
    if len(branch) >= MIN_BOOTSTRAP_TREADS:
        rng = random.Random(seed)
        draws: list[float] = []
        for _ in range(trials):
            sample = [branch[rng.randrange(len(branch))]
                      for _ in range(len(branch))]
            xs = [float(n) for n, _ in sample]
            if len(set(xs)) < 2:
                continue
            a, slope = _line(xs, [ms for _, ms in sample])
            level = a + slope
            if level > 0:
                draws.append((slope - act) / level)
        if len(draws) >= max(20, trials // 10):
            draws.sort()
            tail = (1.0 - coverage) / 2.0
            boot_lo = draws[int(tail * (len(draws) - 1))]
            boot_hi = draws[int((1.0 - tail) * (len(draws) - 1))]
    lo = min([sys_lo] + ([boot_lo] if boot_lo is not None else []))
    hi = max([sys_hi] + ([boot_hi] if boot_hi is not None else []))
    return AlphaInterval(point=point, lo=lo, hi=hi, sys_lo=sys_lo,
                         sys_hi=sys_hi, boot_lo=boot_lo, boot_hi=boot_hi,
                         treads=len(branch), trials=trials)


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


#: The one-token name each gate answers to on its `RESULT:` line. A name is one
#: run of non-whitespace by `moe.bench.exit_codes.result_line`'s own rule, and
#: these are the words a driver greps for, so they are fixed here rather than
#: derived from the claim text.
GATE_NAMES = {0: "override_took_effect", 1: "tile_steps", 2: "time_falls_with_block_m",
              3: "alpha_in_band", 4: "no_crossing_at_the_null_tile"}


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

    @property
    def name(self) -> str:
        return GATE_NAMES.get(self.number, f"gate{self.number}")

    def result_line(self) -> str:
        """The ONE line a driver may grep for this gate.

        `moe.bench.exit_codes.result_line` renders it and `parse_result_lines`
        reads it back, anchored at column zero on the `RESULT: ` prefix. The
        `GATE n VERDICT` line below is for a human and for
        `scripts/pod_session.sh`, whose `gate_from_log` has matched that exact
        shape since 2026-09-01; both are kept because they have different
        readers, and only this one is the machine contract. Nothing else in this
        file's output starts with `RESULT: `.
        """
        detail = f"[{self.kind}/{self.basis}] {self.claim} | measured " \
                 f"{self.measured} | gate {self.threshold}"
        return exit_codes.result_line(
            *self.scored(), " ".join(detail.split()))

    def scored(self) -> tuple[str, str, str]:
        """`(kind, name, verdict)` in `moe.bench.exit_codes`'s vocabulary.

        This file has said UNDECIDED since it was written and the shared table
        says UNKNOWN; they are the same state and the table's spelling wins at
        the boundary, because `classify` refuses a verdict it does not know
        rather than letting it fall through a comparison and be scored as
        whatever the fallthrough happened to be. Both names mean "this gate did
        not decide", and both count AGAINST the gate.
        """
        return (exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
                self.name,
                exit_codes.UNKNOWN if self.verdict == UNDECIDED else self.verdict)

    def render(self) -> list[str]:
        out = [self.result_line(),
               f"GATE {self.number}  {self.verdict:9s} [{self.kind}/"
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
         f"a refutation here is the interesting answer: it says the padded "
         f"arithmetic or the lost occupancy outweighs "
         f"{math.ceil(top / lo)} weight re-reads"]
        + detail)


def no_crossing_reason(pred: TilePrediction) -> str:
    """Why this tile never crosses, when it never crosses. Empty when it does.

    "NEVER CROSSES AT THIS ALPHA" IS AN OUTCOME, NOT A MISSING VALUE, and until
    2026-09-02 this file did not have a sentence for it. `predict_tile` returned
    `crossing_rows=None` exactly when `ai_cap <= ridge`, which is the whole
    claim the study is making, and the report then formatted that None with
    `:.0f`. `--self-test 0.90` and `--self-test 1.0` -- the two worlds the
    measured alphas 0.92-1.02 actually describe -- died with a TypeError before
    report.json was written, so the analysis crashed in precisely the world the
    data pointed at.
    """
    if pred.crossing_rows is not None:
        return ""
    return (f"BLOCK_M={pred.block_m} never crosses at alpha={pred.alpha:.3f}: "
            f"its AI cap 2*BM/(alpha*b) = {pred.ai_cap:.1f} Op/B sits at or "
            f"below the ridge {pred.ridge:.1f} Op/B, so no batch size makes it "
            "compute bound. This is the study's own claim arriving, not a "
            "missing number")


def gate_3_alpha_discriminates(fits, preds_lo, preds_hi, cfg, *, lo: int,
                               hi: int, alpha_source: str,
                               alpha_hat: float | None,
                               alpha_source_bm: int | None,
                               ridge_band: tuple[float, float],
                               interval: AlphaInterval | None = None,
                               band: tuple[float, float] = ALPHA_BAND) -> Gate:
    """Does the fitted alpha's interval overlap the pre-registered ALPHA_BAND.

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

    AND WHAT REPLACED "alpha > 0.33" ON 2026-09-02. The one-sided form was still
    a gate a pre-registered number could not fail from above. `alpha_hat > 0.33`
    passes for every value in 0.33..1.0, so all 41 committed surface fits (0.6
    to 1.0) pass it while NOT ONE of them is within 0.05 of the 0.558 those same
    reports predict; the one published sweep measured 0.989 and 0.923 against a
    prediction of 0.558, passed, and printed "it is where the refit put it". A
    threshold at the midpoint of two hypotheses tests which of the two you are
    nearer, not whether either is right, and this study's registered number is
    the band and not the midpoint.

    THE TEST IS NOW TWO-SIDED AND IS SCORED ON AN INTERVAL. `alpha_interval`
    carries the estimate's random width (a bootstrap over the memory branch's
    own treads) unioned with its systematic bracket (the activation correction
    on one side, the fixed-cost correction on the other -- the two ends this
    report has always printed and never scored). PASS is that interval
    OVERLAPPING `ALPHA_BAND`; FAIL is disjoint, and the verdict says which
    DIRECTION, because "measured 0.99 [0.96, 1.02] is ABOVE the band
    [0.53, 0.59]" and "measured 0.10 is BELOW it" are opposite findings and the
    retired gate rendered the first as a confirmation.

    0.33 SURVIVES AS AN INFORMATIONAL LINE, not as the verdict, so a reader can
    still recompute what the retired gate would have said from a new report.

    The crossing ratio the old gate printed is still computed and still
    reported, under `provenance["restated_crossing_ratio"]`, so an old verdict
    can be recomputed from a new report. It is labelled as a restatement.
    """
    pred_lo = crossing_ratio(preds_lo, lo, hi)
    pred_hi = crossing_ratio(preds_hi, lo, hi)
    retracted_ratio = crossing_ratio(
        predictions((lo, hi), RETRACTED_ALPHA, ridge_band[0]), lo, hi)
    band_lines = []
    for ridge_end, value in zip(ridge_band, (pred_lo, pred_hi), strict=True):
        band_lines.append(f"ridge {ridge_end:.1f}: {value:.3f}x" if value
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
    claim = (f"the fitted alpha's interval overlaps the pre-registered band "
             f"{band[0]}-{band[1]}")
    threshold = (f"interval overlaps [{band[0]}, {band[1]}] "
                 f"(the refit's 90% band; a FAIL names the direction)")
    one_sided = (None if alpha_hat is None
                 else alpha_hat > GATE3_ALPHA_DISCRIMINATOR)
    provenance = {
        "tests": "alpha interval overlaps ALPHA_BAND, both sides",
        "alpha_band": list(band),
        "alpha_interval": (None if interval is None else asdict(interval)),
        "retired_one_sided_threshold": GATE3_ALPHA_DISCRIMINATOR,
        "retired_one_sided_verdict": one_sided,
        "retired_one_sided_note": (
            "alpha_hat > 0.33, the midpoint of the refit and the retracted "
            "value. INFORMATIONAL. It is one-sided and passes for everything "
            "from 0.33 to 1.0, which is why it is no longer the verdict."),
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
    if alpha_hat is None or interval is None:
        return Gate(3, claim, UNDECIDED,
                    "alpha not identifiable at any block size", threshold,
                    ["No ladder had enough memory-bound treads, so no block "
                     "size measured the re-read fraction.",
                     "Lower --r-max is not the fix; a block size whose cap is "
                     "below the ridge is. 32 and 64 are those.",
                     observed,
                     "model's crossing ratio, for reference only: "
                     + "   ".join(band_lines)],
                    basis=DERIVED, provenance=provenance)

    where = interval.direction(band)
    verdict = PASS if not where else FAIL
    own = fits.get(lo)
    imported = alpha_source_bm is not None and alpha_source_bm != lo
    # WHICH ALPHA WAS SCORED, in the verdict's own provenance line, because the
    # gate is about BLOCK_M=128 and the number is usually not from BLOCK_M=128.
    if own is not None and own.memory_points >= MIN_MEMORY_TREADS and not imported:
        crossing = preds_lo[lo].crossing_rows
        fragility = (
            f"Tread 2 there sits within a few percent of the compute branch -- "
            f"{2 * lo} padded rows against a {crossing:.0f} row crossing -- so "
            "it only just qualified as memory bound, and this alpha is the most "
            "fragile number in the report."
            if crossing is not None else
            f"At this alpha BLOCK_M={lo} has no crossing at all "
            f"(cap {preds_lo[lo].ai_cap:.1f} Op/B at or below the ridge "
            f"{preds_lo[lo].ridge:.1f}), so every tread on it is memory bound "
            "and the branch is as long as the sweep.")
        provenance_line = (
            f"SCORED ON BLOCK_M={lo}'s OWN alpha, over {own.memory_points} "
            f"memory-bound treads. {fragility} Compare it with the ladders "
            "below.")
    else:
        provenance_line = (
            f"SCORED ON AN alpha IMPORTED from BLOCK_M={alpha_source_bm}: it is "
            f"not identifiable at BLOCK_M={lo} on this sweep "
            f"({own.memory_points if own else 0} tread(s) stand above the "
            f"compute branch, and a verdict needs {MIN_MEMORY_TREADS}). "
            "ALPHA_BY_BLOCK_M records a drift of about +/-25% across block "
            "sizes, and that drift is `phi` growing with BM/BN rather than a "
            "different miss fraction, so an imported alpha is an alpha of "
            "ANOTHER TILING and the import is a bound, not a substitution.")
    lines = [
        f"measured is the FITTED alpha ({interval.render()}), {alpha_source}. "
        "It is not a crossing ratio and no crossing was measured to produce it.",
        (f"the interval is DISJOINT from the band and lies {where} it: "
         f"measured {interval.point:.3f} [{interval.lo:.3f}, {interval.hi:.3f}] "
         f"is {where} the band [{band[0]}, {band[1]}]"
         if where else
         f"the interval [{interval.lo:.3f}, {interval.hi:.3f}] overlaps the "
         f"band [{band[0]}, {band[1]}]"),
        (f"INFORMATIONAL, the retired one-sided gate: alpha "
         f"{'>' if one_sided else '<='} {GATE3_ALPHA_DISCRIMINATOR:.2f} would "
         f"have read {'PASS' if one_sided else 'FAIL'}. It is not the verdict: "
         "it passes for everything from 0.33 to 1.0, so it cannot fail from "
         "above and every committed surface fit clears it."),
        observed,
        f"the same alpha restates the model's crossing ratio as "
        f"1 + {alpha_hat:.3f} = {1.0 + alpha_hat:.3f}x, which is what the "
        "retired ratio gate printed as its 'measured' value. That number is "
        "algebra over this line, not a second observation.",
        "model's crossing ratio: " + "   ".join(band_lines),
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
                         f"({fit.memory_points} memory-bound tread(s)): "
                         f"{fit.outcome_reason or fit.outcome}")
    # `measured` stays the bare fitted alpha, which is the correction the
    # retired ratio gate already received and what a reader compares across
    # reports. The INTERVAL and the DIRECTION go in `threshold`, so the one
    # greppable RESULT line still carries both.
    threshold = (f"interval [{interval.lo:.3f}, {interval.hi:.3f}] against "
                 f"[{band[0]}, {band[1]}]"
                 + (f" -- DISJOINT, {where} the band" if where
                    else " -- overlaps"))
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


def null_block_m(block_sizes) -> int:
    """The tile gate 4 asks its absence question about: 64, or the smallest.

    One function rather than the same conditional in three places, because the
    predictions block registers the threshold, `bracketing` measures the horizon
    and `gate_4_no_crossing` scores the verdict, and those three reading
    different tiles would be a gate scored against a threshold registered for
    something else.
    """
    return 64 if 64 in block_sizes else min(block_sizes)


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
            capability=None, card: str = NO_CARD_SLUG,
            bandwidth_source: str = "", prov=None) -> Report:
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

    SIGNATURE EXTENDED, never narrowed. `bandwidth_source` and `prov` are
    optional: `bandwidth_source` names where the DENOMINATOR of every predicted
    millisecond came from, which `resolve_bandwidth` now refuses to leave
    unstated, and `prov` is the `moe.bench.provenance.Provenance` block that
    `stamp`s the payload with the commit, the card, the instrument and the two
    sources. Callers that pass neither (the test suite, and any script that
    only wants the analysis) get a report whose provenance says "not supplied
    by caller" for those fields, which is the honest record rather than a
    fabricated one.
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
    treads = {bm: ladder_treads(ok, bm) for bm in block_sizes}
    fits = {bm: fit_ladder(pts, bm, ref, margin, excluded_low_clock=dropped)
            for bm, (pts, dropped) in treads.items()}
    fits = {bm: f for bm, f in fits.items() if f.points}
    excluded_total = sum(dropped for _, dropped in treads.values())

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
    # THE DENOMINATOR OF EVERY PREDICTED MILLISECOND, named beside the ridge and
    # not below it. A ridge measured on this card divided by a bandwidth
    # inherited from a published H200 triad is a HYBRID roof, and until
    # 2026-09-02 `resolve_bandwidth` produced one silently whenever the
    # calibration was missing or unreadable.
    lines.append(f"  bandwidth source: {bandwidth_gbps:.1f} GB/s, "
                 f"{bandwidth_source or 'NOT STATED by the caller'}")
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
        why = no_crossing_reason(p)
        if why:
            lines.append(f"          {why}")
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
    null_bm = null_block_m(block_sizes)
    g4_model, g4_retracted, g4_threshold = gate_4_roof_fraction(
        block_m=null_bm, alpha=alpha, ridge=ridge, b=b)
    # LABELLED, PER TILE, BESIDE THE CAPS THEMSELVES. `ai_cap` is
    # `2 BM / (alpha b)` computed from a FITTED alpha, and a B/(A+B) fit returns
    # `(alpha_b + phi)/(1 + phi + delta)`, so every cap in the table above is
    # high by `1 + phi + delta`. At BM=128 that is larger than the cap-to-ridge
    # gap the cap is being used to decide, which is why it is printed here and
    # not left in a module docstring.
    bn = (pinned or FIXED)["BLOCK_SIZE_N"]
    lines.append(
        "  every AI cap above is 2*BM/(alpha*b) from a FITTED alpha and is HIGH "
        "by ai_model.lin_overstatement = 1+phi+delta (alpha_a is unmeasured "
        "anywhere in this repo and delta is taken as 0, so each range is a "
        "bracket and a lower bound): "
        + ", ".join(f"BM={bm} {lo:.2f}-{hi:.2f}x" for bm, (lo, hi) in
                    ((bm, cap_overstatement(cfg, bm, bn, b))
                     for bm in block_sizes)))
    lines.append(f"  GATE 3 will test alpha > {GATE3_ALPHA_DISCRIMINATOR:.2f} "
                 f"(midpoint of {alpha:.3f} and {RETRACTED_ALPHA}) as an "
                 f"INFORMATIONAL line; the VERDICT is whether the fitted "
                 f"alpha's interval overlaps the band "
                 f"[{ALPHA_BAND[0]}, {ALPHA_BAND[1]}], in both directions")
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
    lines.append("  alpha here is B/(A+B), which is (alpha_b+phi)/(1+phi+delta) "
                 "and NOT a weight miss fraction: see moe/bench/ai_model.py "
                 "and LadderFit.alpha")
    if excluded_total:
        lines.append(
            f"  {excluded_total} cell(s) excluded for clock level: their SM "
            "clock under load came in below the clock the roof was measured "
            "at, so they sit above a compute branch they were never comparable "
            "with. Excluded from every fit below and counted per ladder.")
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
            + f"  {f.mean_rel_err:6.2%}"
            + (f"   [{f.excluded_low_clock} excluded for clock level]"
               if f.excluded_low_clock else ""))
        if f.outcome_reason:
            lines.append(f"           {f.outcome_reason}")
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

    # The interval the two-sided gate 3 is scored on, built from the SAME ladder
    # the point estimate came from so the two cannot describe different fits.
    interval = (alpha_interval(fits[alpha_source_bm], cfg, bandwidth_gbps)
                if alpha_source_bm is not None else None)
    if interval is not None:
        lines.append(f"  alpha scored by gate 3: {interval.render()}")

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
                                   ridge_band=ridge_band, interval=interval),
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
        "bandwidth_gbps": bandwidth_gbps,
        "bandwidth_source": (bandwidth_source
                             or "NOT STATED by the caller"),
        "cells_excluded_for_clock_level": excluded_total,
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
        # "NEVER CROSSES AT THIS ALPHA" IS AN EXPLICIT OUTCOME HERE. `crosses`
        # is a bool and never absent, `crossing_rows` is null when there is no
        # crossing, and `no_crossing_reason` says in words that the cap sits at
        # or below the ridge. Before 2026-09-02 the only record of that state
        # was a null in `crossing_rows_ridge_lo`, which a report generator read
        # as a missing measurement -- and which this file formatted with `:.0f`
        # and crashed on at every alpha above about 0.79, i.e. at every alpha
        # the measured 0.92-1.02 ladders actually describe.
        "predictions": {
            str(bm): {"ai_cap": preds_lo[bm].ai_cap,
                      "crossing_rows_ridge_lo": preds_lo[bm].crossing_rows,
                      "crossing_rows_ridge_hi": preds_hi[bm].crossing_rows,
                      "first_compute_tread": preds_lo[bm].first_compute_tread,
                      "crosses": preds_lo[bm].crosses,
                      "crossing_rows": preds_lo[bm].crossing_rows,
                      "no_crossing_reason": no_crossing_reason(preds_lo[bm]),
                      "cap_overstatement": list(cap_overstatement(
                          cfg, bm, (pinned or FIXED)["BLOCK_SIZE_N"], b)),
                      "cap_overstatement_note": cap_note(
                          cfg, bm, (pinned or FIXED)["BLOCK_SIZE_N"], b)}
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
                             # THE NAMED OUTCOME, beside the older reason
                             # token. `undecided` is True only where the sweep
                             # LOOKED and could not say -- a branch parallel to
                             # the ridge, or treads lost to clock level -- which
                             # is a different report from "too few treads" and
                             # points at a different next experiment.
                             "outcome": f.outcome,
                             "outcome_reason": f.outcome_reason,
                             "undecided": f.undecided,
                             "branch_start": f.branch_start,
                             "excluded_low_clock": f.excluded_low_clock,
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
    # ONE PROVENANCE BLOCK, LAST, AND IT STAMPS RATHER THAN MERGES. `stamp`
    # raises `ProvenanceCollision` if any of the five audited top-level keys is
    # already present with a different value, so two provenance blocks can never
    # be layered over one report without someone noticing -- which is why the
    # caller passes `ridge_source` and `bandwidth_source` strings that ALREADY
    # begin with the block's own token (`main` builds them as
    # "<kind>: <prose>"), so the two writers agree and the stamp is a no-op on
    # those two keys rather than a conflict.
    if prov is not None:
        payload = prov.stamp(payload)
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


class RetiredInstrument(RuntimeError):
    """`time_call` was called. It no longer times anything, by design.

    See `time_call` for what to call instead and why this is a refusal rather
    than a redirect.
    """


def time_call(fn, warmup: int, iters: int):
    """RETIRED 2026-09-02. Raises `RetiredInstrument`; call `time_kernel`.

    WHAT THIS USED TO DO AND WHY IT MAY NOT DO IT AGAIN. It created a fresh
    CUDA event pair inside the loop, recorded the start event on a stream it had
    just synchronised, ran one call, and synchronised again -- once per
    iteration, with no L2 flush and no clock read. The ROOF every number it
    produced is scored against was measured by `moe.bench.timing.time_eager`,
    which is queue-deep with pre-primed events. Two instruments, one comparison.
    The audit bounded the host prefix this one let inside the measured interval
    at about 0.18 ms per fused_experts call on the H200 pod and 0.30 ms on the
    A100 pod, which is a bias in the fitted alpha of 8-16% at the smallest
    ladder cells, DIFFERENT PER CARD, and of the same order as the cross-card
    effect this study registered (+0.0117, MDE 0.030).

    WHY A REFUSAL AND NOT A WRAPPER. A wrapper would have to invent the two
    things `time_kernel` needs and this signature cannot supply: `warmup` here
    is a COUNT of calls and the instrument warms for a DURATION of delivered GPU
    load, and there is no honest conversion between them; and a caller passing a
    fixed `iters` has already decided a sample size that the instrument sizes
    from the warmup's own queue-deep per-call time. Guessing either would put a
    number in a report that no instrument produced.

    WHY THE NAME SURVIVES. `scripts/bm128_roofline.py`,
    `scripts/bm128_depth.py`, `scripts/bn_decomposition.py` and
    `scripts/occupancy_vs_swizzle.py` load this module by path and check
    `hasattr(module, "time_call")` before they will run at all. Deleting the
    symbol would abort those four at import with "no longer exports time_call",
    which says nothing about what actually changed. They keep importing, and the
    first cell any of them tries to time raises this, with the fix in the
    message. Migrating them onto `time_kernel` is their own phase; until then
    their numbers were never comparable with the roof, which is the finding, not
    a regression introduced here.
    """
    raise RetiredInstrument(
        "time_call has been retired: it timed with per-iteration synchronises "
        "and events created inside the loop, which is NOT the instrument the "
        "compute roof was measured with, and it exposed 0.18-0.30 ms of host "
        "enqueue time per call inside the measured interval (a per-card bias of "
        "8-16% in the fitted alpha at the smallest ladder cells).\n"
        "    Use moe.bench.timing.time_kernel(fn, warmup_ms=..., "
        "target_ms=..., trials=..., l2_flush=..., reference_clock_mhz=...), "
        "which is queue-deep, flushes L2 per iteration, samples the SM clock "
        "under load and returns a KernelTiming carrying its own instrument "
        "name.\n"
        f"    warmup={warmup} is a CALL COUNT and time_kernel warms for a "
        f"DURATION of delivered GPU load; iters={iters} is a sample size "
        "time_kernel derives from that warmup. Neither converts, which is why "
        "this refuses instead of wrapping.")


def reference_clock_mhz(gpu_name: str = "") -> tuple[float | None, str]:
    """The SM clock this card's roof was measured at, and where it came from.

    `timing.clock_flags` needs a reference before it can say anything about
    LEVEL: a level is relative to something, and `time_kernel` will not invent
    the something. That something is the clock the CALIBRATION ran its GEMM at,
    because the roof every cell here is scored against is that GEMM's number.

    Three places, in falling order of directness, all inside the card's own
    `measured_*.yaml`: the full `LoadedClock` record's median (samples taken
    WHILE the GEMM ran), the scalar `gemm_clock_mhz` every older consumer reads,
    and the compute settle's final plateau. `calibrate.clock_established` exists
    because those three have disagreed: eleven committed calibrations of one
    H200 recorded 1485-1935 MHz for the scalar while their own settle histories
    sat at 1455-1515. The order here prefers the number measured under the load
    that set the roof, and the returned string says which one was used, so a
    LEVEL exclusion can always be traced to a field in a file.

    Returns `(None, reason)` when the card has no calibration. That is not a
    failure: `clock_level_ok` is then None on every cell, which means "not
    determined", and `ladder_treads` excludes nothing. A guessed reference would
    exclude real treads.
    """
    try:
        from moe.bench.roofline import current_gpu_name
    except Exception as exc:                            # noqa: BLE001
        return None, f"roofline unavailable: {type(exc).__name__}: {exc}"
    name = gpu_name or current_gpu_name()
    detail = _measured_detail(name)
    if not detail:
        return None, (f"no calibration yaml for {name or 'this device'}, so the "
                      "clock the roof was measured at is not known here")
    median = (detail.get("gemm_clock") or {}).get("median_mhz")
    if median:
        return float(median), (f"{name}: median of the samples taken while the "
                               "calibration's dense GEMM ran")
    scalar = detail.get("gemm_clock_mhz")
    if scalar:
        return float(scalar), (f"{name}: gemm_clock_mhz, the scalar the "
                               "calibration published for its dense GEMM")
    plateau = (detail.get("settle") or {}).get("final_mhz")
    if plateau:
        return float(plateau), (f"{name}: the compute settle's final plateau; "
                                "the calibration recorded no GEMM clock")
    return None, (f"{name}: the calibration carries no clock at all, so LEVEL "
                  "cannot be scored against it")


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
              b: int, pinned: dict, prov=None) -> tuple[list[Cell],
                                                        dict[int, int],
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

    from moe.bench import timing
    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    # THE CLOCK THE ROOF WAS MEASURED AT, resolved once, before any cell. It is
    # the reference the LEVEL flag compares against; without it every cell's
    # `clock_level_ok` is None, which means "not determined" and excludes
    # nothing. Resolved here rather than per cell so the whole arm is scored
    # against one number and a mid-sweep yaml rewrite cannot move it.
    reference_clock, clock_source = reference_clock_mhz()
    print("reference clock: "
          + (f"{reference_clock:.0f} MHz, {clock_source}" if reference_clock
             else f"NOT RESOLVED ({clock_source}); every cell's clock LEVEL "
                  "verdict will be None and no cell can be excluded for it"))

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
                    t = timing.time_kernel(
                        call, warmup_ms=args.warmup,
                        target_ms=args.cell_budget_ms, trials=args.trials,
                        l2_flush=not args.no_l2_flush,
                        reference_clock_mhz=reference_clock)
                cell = make_cell(cfg, rows, bm, t.ms_p50, sm_count=sm_count,
                                 block_n=pinned["BLOCK_SIZE_N"],
                                 ms_min=t.ms_min, ms_stdev=t.ms_std,
                                 iters=t.iters, instrument=t.instrument,
                                 warmup_ms=t.warmup_ms, trials=t.trials,
                                 sm_clock_load_mhz=t.sm_clock_load_mhz,
                                 clock_level_ok=t.clock_level_ok,
                                 clock_drift_ok=t.clock_drift_ok,
                                 l2_flush=t.l2_flush)
                if t.clock_level_ok is False or t.host_bound:
                    print(f"  ^ {t.clock_note or ''} {t.host_note or ''}".rstrip())
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
            append_cell(csv_path, cell, prov)
            print(f"  BM={bm:3d} T={cell.tokens:6d} r={rows:6d} "
                  f"n={cell.tiles_per_expert:3d} waves {cell.waves_up:7.1f} "
                  f"{cell.ms_p50:9.4f} ms  {cell.useful_tflops:7.1f} TFLOP/s")
    return cells, compiles, executed


# --------------------------------------------------------------------------
# Persistence.
# --------------------------------------------------------------------------

CSV_FIELDS = [f for f in Cell.__dataclass_fields__]

#: Provenance columns appended to every cells.csv row, after the measurement
#: columns. `Provenance.as_columns` prefixes every one with `prov_`, so a
#: provenance column can never collide with a measurement column that happens
#: to share its name (`iters`, `gpu_name`). Written per ROW rather than once per
#: file because a resumed cells.csv is written by two processes on two days and
#: possibly two commits, and a header cannot say that.
PROVENANCE_COLUMNS = sorted(
    PV.Provenance().as_columns())


def append_cell(path: Path, cell: Cell, prov=None) -> None:
    """One row, flushed. An abort costs the cell in flight and nothing else.

    SIGNATURE EXTENDED, never narrowed: `prov` is optional and a caller that
    omits it writes the measurement columns alone, which is what the four
    sibling scripts and the test suite do. When it is given, the row also
    carries the commit, the card, the instrument and the two ruler sources that
    made it, so a cells.csv on a shared network volume stays attributable after
    the pod is gone.
    """
    new = not path.exists()
    fields = CSV_FIELDS + (PROVENANCE_COLUMNS if prov is not None else [])
    row = asdict(cell)
    if prov is not None:
        row.update(prov.as_columns())
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def _cell_value(type_name: str, raw: str):
    """One CSV field back into its declared type.

    THE OPTIONAL FORMS ARE HANDLED SEPARATELY AND THE EMPTY STRING IS None, not
    zero and not False. `sm_clock_load_mhz`, `clock_level_ok` and
    `clock_drift_ok` are tri-state by design: a cell timed on a host without
    NVML has no clock, and reading that blank back as `False` would mark it
    excluded while reading it as `True` would mark it comparable. Both are
    claims the row does not make.
    """
    optional = type_name.replace(" ", "").endswith("|None")
    base = type_name.replace(" ", "").removesuffix("|None")
    if optional and raw == "":
        return None
    if base == "int":
        return int(float(raw or 0))
    if base == "float":
        return float(raw or 0.0)
    if base == "bool":
        return raw == "True"
    return raw


def read_cells(path: Path) -> tuple[set[tuple[int, int]], list[Cell]]:
    """Cells already measured, so a re-run resumes rather than repeats."""
    if not path.exists():
        return set(), []
    cells: list[Cell] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            kw = {name: _cell_value(str(f.type), row.get(name, "") or "")
                  for name, f in Cell.__dataclass_fields__.items()}
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

    THE ID IS BUILT BY `moe.bench.provenance.run_id` AS OF 2026-09-02, not by a
    private hash here. That function raises `NoCard` on a missing card and
    `UnresolvedKnob` on a None or empty value, hashes the knobs in sorted order
    so the id does not depend on the order they were named in, and puts the card
    slug at the FRONT where `ls` shows it. Three scripts had each re-implemented
    a subset of this and each had left a different knob out; the collisions that
    cost are in that module's docstring. Two knobs are new to the key here and
    both set the measured milliseconds: `warmup` (now a duration of sustained
    load) and `l2_flush`, which changes whether every timed iteration starts
    with a cold L2.
    """
    return PV.run_id(
        card=card,
        model=args.model,
        dtype=args.dtype,
        tiles=tuple(int(v) for v in args.tiles.split(",")),
        g=args.group_m,
        n=args.block_n,
        stages=args.num_stages,
        r=args.r_max,
        rowstep=args.row_step,
        probes=args.step_probes,
        seed=args.seed,
        iters=args.iters,
        warmup=args.warmup,
        budget=args.cell_budget_ms,
        flush=not args.no_l2_flush,
        trials=args.trials,
        # Not a swept knob today and in the key anyway: `balanced_ids` builds an
        # EXACT per-expert histogram and `run_sweep` never asks for anything
        # else, so the day a sampled-routing arm appears its cells must not land
        # in a balanced arm's directory and be skipped as already measured.
        routing="balanced",
    )


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

#: One tread of the null tile's ladder is planted below the roof's clock. It
#: must be excluded by `ladder_treads`, counted on the ladder row, and named.
LOW_CLOCK_WORLD = "low-clock"
#: The subject tile's memory branch is planted ON the compute branch, which is
#: the regime `PARALLEL_BRANCH_TOLERANCE` refuses to decide. The outcome must be
#: `UNDECIDED_PARALLEL_BRANCH` with its reason, not a blank and not an import.
PARALLEL_WORLD = "parallel-branch"
SELF_TEST_WORLDS = (LOW_CLOCK_WORLD, PARALLEL_WORLD)


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
    ap.add_argument("--iters", type=int, default=50,
                    help="RETIRED as a timing knob on 2026-09-02 and kept in "
                         "the run id. moe.bench.timing.time_kernel sizes the "
                         "iteration count from --cell-budget-ms and the "
                         "warmup's own queue-deep per-call time, which is the "
                         "only sizing that can hold one trial to a duration. "
                         "It stays in the id because cells measured under a "
                         "different count exist on disk and a directory must "
                         "not be resumed into across that change, and it still "
                         "caps --dry-run's cost estimate")
    ap.add_argument("--warmup", "--warmup-ms", type=float, default=300.0,
                    dest="warmup", metavar="MS",
                    help="MILLISECONDS of delivered GPU load to warm up for, "
                         "not a call count. UNITS CHANGED 2026-09-02: a count "
                         "is the wrong unit and the ladders that compared cells "
                         "warmed at 5 against cells warmed at 20 were comparing "
                         "clock states. A 1 ms kernel needs hundreds of calls "
                         "before the governor reacts and a 30 ms GEMM needs "
                         "one, so the instrument warms for a duration measured "
                         "with the same events the trials use")
    ap.add_argument("--trials", type=int, default=3,
                    help="queue-deep trials per cell; the percentiles are over "
                         "iters x trials samples")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="do NOT evict L2 between timed iterations. Off by "
                         "default because the roof was measured flushed and a "
                         "warm-L2 cell is not comparable with it. Recorded per "
                         "cell and in the run id, so a flushed and an unflushed "
                         "sweep can never share a directory")
    ap.add_argument("--cell-budget-ms", type=float, default=400.0,
                    help="target measured KERNEL time per trial; the "
                         "instrument sizes its iteration count from it")
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
    ap.add_argument("--bandwidth-gbps", "--bandwidth", type=float, default=0.0,
                    dest="bandwidth_gbps", metavar="GBPS",
                    help="GB/s. 0 (the default) reads the ATTACHED DEVICE's "
                         "own calibration and REFUSES if there is none. It used "
                         "to fall back to 4374.5, an H200 triad figure, on a "
                         "missing OR unreadable calibration, so --ridge 145.8 "
                         "(an A100 ridge) built a roof of 145.8 x 4374.5 out of "
                         "two machines. Required alongside --ridge off a "
                         "calibrated card, because that is exactly the pairing "
                         "that produced the hybrid")
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
    ap.add_argument("--self-test-world", default="", choices=("",) + SELF_TEST_WORLDS,
                    help="a planted world that is not a single alpha. "
                         f"{LOW_CLOCK_WORLD} puts one tread below the roof's "
                         "clock, which must be excluded and counted; "
                         f"{PARALLEL_WORLD} puts the memory branch on the "
                         "compute branch, which must come out UNDECIDED rather "
                         "than blank. Implies --self-test at the refit alpha "
                         "unless one is given")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="exit with the moe.bench.exit_codes code for the gate "
                         "verdicts (CLAIM_FAIL 1, INVALID 3) instead of DONE. "
                         "Off by default because a falsified prediction is a "
                         "successful run, not a failed one")
    return ap


def self_test_cells(cfg, grid, block_sizes, *, alpha: float, ridge: float,
                    bandwidth_gbps: float, b: int, sm_count: int,
                    noise: float = 0.0, seed: int = 0, world: str = "",
                    warmup_ms: float = 0.0, trials: int = 0,
                    l2_flush: bool = True) -> list[Cell]:
    """The synthetic cells, plus the two worlds that are not a single alpha.

    A SELF-TEST THAT CANNOT FAIL IS NOT ONE, and until 2026-09-02 this script's
    self-test could only plant an alpha. The two paths that decide whether a
    BLOCK_M=128 row exists at all were therefore never exercised off-GPU:

      `low-clock` puts ONE tread of the null tile below the clock the roof was
      measured at, by setting its `clock_level_ok` False. `ladder_treads` must
      drop it, the ladder row must say how many were dropped, and if that takes
      the ladder under `MIN_MEMORY_TREADS` the outcome must be
      `UNDECIDED_LOW_CLOCK` and not "too few treads".

      `parallel-branch` puts the SUBJECT tile's whole ladder on a line whose
      slope is the compute slope, which is the regime
      `PARALLEL_BRANCH_TOLERANCE` refuses to decide. The outcome must be
      `UNDECIDED_PARALLEL_BRANCH` carrying its reason, rather than the `None`
      that used to be indistinguishable from a sweep that lacked treads.

    Both worlds are built by MODIFYING cells the model generated, not by
    hand-writing a ladder, so everything else in the report stays the world the
    alpha describes and only the planted thing differs.
    """
    if world and world not in SELF_TEST_WORLDS:
        raise ValueError(f"unknown self-test world {world!r}; "
                         f"choices are {list(SELF_TEST_WORLDS)}")
    null_bm = null_block_m(block_sizes)
    low = (null_bm, 2) if world == LOW_CLOCK_WORLD else None
    cells = synthetic_cells(cfg, grid, block_sizes, alpha=alpha, ridge=ridge,
                            bandwidth_gbps=bandwidth_gbps, b=b,
                            sm_count=sm_count, noise=noise, seed=seed,
                            low_clock=low, warmup_ms=warmup_ms, trials=trials,
                            l2_flush=l2_flush)
    if world != PARALLEL_WORLD:
        return cells
    # The compute branch scales as C ~ BLOCK_M off the reference ladder, so a
    # subject ladder whose time is the reference's time at the same ROW COUNT is
    # a ladder sitting exactly on the compute branch. Built from the reference's
    # own cells rather than from a formula, so it stays parallel whatever the
    # model does to the reference.
    subject = 128 if 128 in block_sizes else null_bm
    reference_bm = max(block_sizes)
    if subject == reference_bm:
        return cells
    by_rows = {c.rows_per_expert: c for c in cells if c.block_m == reference_bm}
    out = []
    for c in cells:
        peer = by_rows.get(c.rows_per_expert)
        if c.block_m != subject or peer is None or peer.ms_p50 <= 0:
            out.append(c)
            continue
        # A hair ABOVE the compute branch, so the tread qualifies as memory
        # bound and the fit gets far enough to notice the two slopes are one
        # line. Sitting ON it would be caught earlier, by membership, and would
        # test a different branch of the code.
        out.append(make_cell(
            cfg, c.rows_per_expert, subject, peer.ms_p50 * 1.10,
            sm_count=sm_count, block_n=FIXED["BLOCK_SIZE_N"],
            ms_min=peer.ms_p50 * 1.10, ms_stdev=peer.ms_p50 * 1.10 * noise,
            instrument=SYNTHETIC_INSTRUMENT, warmup_ms=warmup_ms,
            trials=trials, l2_flush=l2_flush,
            sm_clock_load_mhz=SYNTHETIC_CLOCK_MHZ, clock_level_ok=True,
            clock_drift_ok=True))
    return out


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
    #: `cli`, `calibration` or `hypothesis`: the SHORT token, beside the prose.
    #: The prose says which yaml and which session; the token is what a
    #: provenance block records and what an audit gate compares, because a gate
    #: that has to pattern-match a sentence is a gate that breaks on a reword.
    #: Defaulted so the three sibling scripts that read `ResolvedRidge` are
    #: unaffected.
    source_kind: str = ""


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
                             band_source, "", source_kind="cli")

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
            band_source, hw.name or gpu_name, source_kind="calibration")

    if synthetic:
        return ResolvedRidge(RIDGE_BAND[0], RIDGE_BAND, HYPOTHESIS_RIDGE_SOURCE,
                             HYPOTHESIS_RIDGE_SOURCE, gpu_name,
                             source_kind="hypothesis")

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


class BandwidthUnavailable(RuntimeError):
    """No bandwidth this run is entitled to use, and no constant may stand in.

    THE HYBRID ROOF, WHICH `resolve_ridge` ALREADY REFUSED AND THIS DID NOT.
    Every predicted millisecond, every `ridge x bandwidth` roof and therefore
    every roof fraction, every gate-4 verdict and the whole compute-reference
    LEVEL check divide by this number. Until 2026-09-02 `resolve_bandwidth`
    returned the published H200 triad figure, 4374.5 GB/s, whenever
    `load_measured` returned None OR raised ANY exception -- and `--ridge X`
    took the operator's assertion for the numerator while leaving that constant
    in the denominator. So `--ridge 145.8` on an A100 with no calibration
    produced `model_roof = 145.8 x 4374.5`: an A100 ridge times an H200
    bandwidth, printed as this card's roof, with a source line that said
    "published H200 triad ceiling" in a field nobody was reading.

    The two halves of a roof have to come from the same machine, or the roof is
    not a machine's. So this refuses on exactly the terms `RidgeUnavailable`
    does, and `--ridge` now requires `--bandwidth` beside it.
    """


@dataclass(frozen=True)
class ResolvedBandwidth:
    """The bandwidth, and where it came from. Provenance travels with it.

    `source` is the short token (`cli`, `calibration`, `hypothesis`) a
    provenance block records and an audit gate compares; `detail` is the prose
    that names the yaml and the pattern. Two fields because a gate that has to
    pattern-match a sentence breaks on a reword, and a report that prints only
    the token tells a reader nothing about which file it came from.

    THE COMPATIBLE PATH. This unpacks as the `(gbps, source_prose)` pair
    `resolve_bandwidth` used to return, so `scripts/tile_cap_test.py:1760`'s
    `bandwidth, bw_source = SWEEP.resolve_bandwidth(args)` keeps working
    unchanged. New callers take the fields by name and get the token as well.
    """

    gbps: float
    source: str
    detail: str
    device: str

    def __iter__(self):
        yield self.gbps
        yield self.detail


#: The figure that used to stand in silently. Kept ONLY so the refusal below can
#: name what it is refusing to substitute, and so a reader of an older report can
#: recognise the number. It is an H200 triad ceiling and belongs to no other
#: card; nothing in this file may return it as a value.
PUBLISHED_H200_TRIAD_GBPS = 4374.5


#: What a report says when its bandwidth is the module constant rather than a
#: measurement. Only reachable on a laptop plan or replay whose RIDGE is the
#: matching hypothesis, so the two halves of the roof still come from one
#: machine's 2026-08-26 calibration.
HYPOTHESIS_BANDWIDTH_SOURCE = (
    "HYPOTHESIS: the 2026-08-26 H200 triad ceiling, which belongs to no "
    "attached device, paired with the H200 hypothesis ridge")


def resolve_bandwidth(args, *, synthetic: bool | None = None) -> ResolvedBandwidth:
    """This card's measured bandwidth, the operator's assertion, or a refusal.

    Order, mirroring `resolve_ridge` step for step:

      1. `--bandwidth` / `--bandwidth-gbps`, the operator's assertion, source
         `cli`.
      2. THE ATTACHED DEVICE'S OWN CALIBRATION, source `calibration`.
      3. For `--dry-run` and `--self-test` ONLY, AND ONLY WHEN `--ridge` WAS NOT
         GIVEN, the module's H200 triad as a stated HYPOTHESIS, source
         `hypothesis`. The second condition is the whole point: a hypothesis
         bandwidth is safe only while the ridge beside it is the MATCHING
         hypothesis from the same 2026-08-26 calibration, so the roof is one
         machine's. The moment the operator asserts a ridge for another card,
         pairing it with this constant is the hybrid roof, and the escape
         closes.
      4. Otherwise REFUSE, on exactly the terms `resolve_ridge` refuses.

    RETURN TYPE CHANGED on 2026-09-02, from `(float, str)` to
    `ResolvedBandwidth`, so a caller cannot take the number and drop the source:
    that is how 4374.5 reached reports for a card whose own triad is 1799.4.
    The one in-repo caller outside this file monkeypatches this function in a
    test (`tests/test_tile_cap.py`), which passes its own callable and is
    unaffected by the shape.
    """
    if synthetic is None:
        # INFERRED, not defaulted to False. `scripts/tile_cap_test.py` calls
        # this with one argument, and defaulting to False would refuse its
        # documented laptop `--dry-run` -- a script this one does not own,
        # broken by a keyword it does not pass.
        synthetic = bool(getattr(args, "dry_run", False)
                         or getattr(args, "self_test", None) is not None)
    if args.bandwidth_gbps:
        return ResolvedBandwidth(
            args.bandwidth_gbps, "cli",
            "given on the command line as --bandwidth/--bandwidth-gbps", "")
    from moe.bench import roofline
    gpu_name = roofline.current_gpu_name()
    hw = None
    try:
        hw = roofline.load_measured(gpu_name or None)
    except Exception as exc:                            # noqa: BLE001
        # Broad, and it REFUSES rather than substituting. A yaml this process
        # cannot parse is not a licence to use another machine's ceiling; it is
        # a reason to stop and say which file failed.
        raise BandwidthUnavailable(
            f"this device's calibration could not be read "
            f"({type(exc).__name__}: {exc}), so this run has no bandwidth it is "
            "entitled to quote.\n"
            "    Run:  python scripts/calibrate_hardware.py\n"
            "    or state it yourself:  --bandwidth <GB/s>") from exc
    if hw is not None:
        return ResolvedBandwidth(
            hw.bandwidth_bytes_s / 1e9, "calibration",
            f"measured on this device: {hw.name}, "
            f"{hw.bandwidth_bytes_s / 1e9:.1f} GB/s "
            f"({hw.ceiling_pattern or 'unnamed'} pattern)",
            hw.name or gpu_name)
    ridge_arg = getattr(args, "ridge", 0.0)
    if synthetic and not ridge_arg:
        return ResolvedBandwidth(PUBLISHED_H200_TRIAD_GBPS, "hypothesis",
                                 HYPOTHESIS_BANDWIDTH_SOURCE, gpu_name)
    if ridge_arg:
        raise BandwidthUnavailable(
            f"--ridge {ridge_arg} Op/B was given and there is no bandwidth to "
            f"pair it with on this device ({gpu_name or 'no CUDA device'}).\n"
            "    The ridge is then YOUR assertion, for some card, and the only "
            f"bandwidth left is the {PUBLISHED_H200_TRIAD_GBPS} GB/s H200 "
            "constant: their product is a roof built out of two machines. That "
            "is the defect exactly -- --ridge 145.8 is an A100 figure and "
            "145.8 x 4374.5 is not any card's roof.\n"
            "    Every predicted millisecond, the ridge x bandwidth roof, every "
            "roof fraction and the compute reference's LEVEL check divide by "
            "this number.\n"
            "    Run:  python scripts/calibrate_hardware.py\n"
            "    or state the assertion yourself:  --bandwidth <GB/s>")
    # NO RIDGE EITHER, so no roof can be assembled and there is nothing to make
    # a hybrid OUT OF. The honest record is a bandwidth of zero with the source
    # `unresolved`, never the constant: zero is not a plausible ceiling, so any
    # arithmetic on it produces an obvious infinity rather than a roof that
    # looks right. `main` walls it below and refuses before anything is
    # measured, and `resolve_ridge` refuses first in this same state, so the
    # wall is a second lock on a door that is already shut.
    #
    # WHY THIS ONE IS RETURNED AND NOT RAISED, said plainly.
    # `scripts/tile_cap_test.py:1760` calls this function OUTSIDE a try and
    # then refuses on its own ridge two lines later; raising here would replace
    # that script's named refusal and exit code 2 with an uncaught exception,
    # in a file this one does not own. A zero it never reads costs nothing.
    # Every state where a hybrid roof COULD form -- an asserted ridge with no
    # bandwidth, an unreadable calibration -- still raises above.
    return ResolvedBandwidth(
        0.0, "unresolved",
        f"NO BANDWIDTH: this device ({gpu_name or 'no CUDA device'}) has no "
        "calibration, --bandwidth was not given, and this is not a --dry-run "
        "or --self-test. There is no ridge either, so no roof can be formed. "
        "Run scripts/calibrate_hardware.py, or state --bandwidth <GB/s>",
        gpu_name)


def main(argv=None) -> int:
    """The one entry point, and the one place an exit code is chosen.

    Every return here is a member of `moe.bench.exit_codes`'s table:
    REFUSED (2) before anything is measured, ERROR (4) for an exception nobody
    planned for, and otherwise `classify` over the scored gates. Without
    `--fail-on-gate` a CLAIM_FAIL is reported as DONE, because a falsified
    pre-registered claim is a successful run and the flag's whole purpose is to
    say when the caller wants otherwise; a VALIDITY failure is INVALID either
    way, since nothing on the page may be quoted after one.
    """
    args = build_parser().parse_args(argv)
    cfg = MODEL_CONFIGS[args.model]
    block_sizes = tuple(int(v) for v in args.tiles.split(","))
    b = dtype_bytes(args.dtype)
    if args.self_test_world and args.self_test is None:
        args.self_test = ALPHA
    synthetic = bool(args.dry_run or args.self_test is not None)
    # BOTH HALVES OF THE ROOF ARE RESOLVED BEFORE ANY GPU TIME IS SPENT, and
    # before the plan is printed, so a run that has no ruler it may quote costs
    # nothing and says why. Bandwidth refuses on the same terms the ridge does:
    # they are the numerator and the denominator of one roof and a run is not
    # entitled to take one from this card and the other from a constant.
    try:
        rr = resolve_ridge(args, synthetic=synthetic)
        rb = resolve_bandwidth(args, synthetic=synthetic)
    except (RidgeUnavailable, BandwidthUnavailable) as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    if rb.source == "unresolved":
        # The one state `resolve_bandwidth` reports rather than raises, because
        # a sibling script calls it outside a try. Nothing in THIS file may
        # proceed on it; see that function's tail comment for why.
        print(f"REFUSED: {rb.detail}")
        return exit_codes.REFUSED
    bandwidth, bw_source = rb.gbps, rb.detail
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
    print(f"bandwidth   {bandwidth:.1f} GB/s, source={rb.source}: {bw_source}")
    print(f"ridge       {rr.ridge:.2f} Op/B, source={rr.source_kind}: {rr.source}")
    print(f"ridge band  {rr.band[0]:.2f}-{rr.band[1]:.2f} Op/B, {rr.band_source}")
    if rr.source_kind != rb.source:
        # Not a refusal: `--ridge` beside a calibrated bandwidth is a deliberate
        # assertion and the operator may make it. It IS said out loud, because
        # the two halves of the roof then came from different places and the
        # audit's finding was that nobody could see that from the output.
        print(f"            NOTE: the roof's two halves have different sources "
              f"(ridge {rr.source_kind}, bandwidth {rb.source})")
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
    capability = resolve_capability(args, synthetic=synthetic)
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
            return exit_codes.REFUSED
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
            return exit_codes.REFUSED

    out_dir.mkdir(parents=True, exist_ok=True)

    # ONE PROVENANCE BLOCK PER RUN, built here and handed to both writers: the
    # report payload and every cells.csv row. Built AFTER the rulers resolve so
    # it carries their sources, and before any measurement so every row of one
    # run carries one block.
    # ONE STRING PER SOURCE, TOKEN FIRST, HANDED TO BOTH WRITERS. `stamp` puts
    # `ridge_source` and `bandwidth_source` at the payload's top level and
    # RAISES if the payload already carries a different value for either, so
    # the block and the report have to agree; making them literally the same
    # string is how they agree, and leading with the token
    # (`cli` / `calibration` / `hypothesis`) is what lets an audit gate compare
    # a source without pattern-matching a sentence.
    ridge_src = f"{rr.source_kind}: {rr.source}"
    bw_src = f"{rb.source}: {rb.detail}"
    prov = PV.provenance_block(
        instrument=timing_basis(), ridge=rr.ridge, ridge_source=ridge_src,
        bandwidth=bandwidth, bandwidth_source=bw_src,
        warmup_ms=args.warmup, iters=args.iters,
        target_ms=args.cell_budget_ms)

    if args.self_test is not None:
        alpha = args.self_test
        sm_count = args.sm_count or DEFAULT_SM_COUNT
        cells = self_test_cells(cfg, grid, block_sizes, alpha=alpha,
                                ridge=rr.ridge, bandwidth_gbps=bandwidth, b=b,
                                sm_count=sm_count, noise=args.self_test_noise,
                                seed=args.seed, world=args.self_test_world,
                                warmup_ms=args.warmup, trials=args.trials,
                                l2_flush=not args.no_l2_flush)
        compiles = {bm: 1 for bm in block_sizes}
        executed = dict(compiles)
        print(f"\nSELF TEST: cells GENERATED from the model at alpha={alpha}"
              + (f", world={args.self_test_world}" if args.self_test_world else "")
              + ". Nothing here was measured.")
        print("The gates below are being run against a world we constructed, "
              "which tests the gates and not the hardware.")
    else:
        import torch
        alpha = args.alpha
        sm_count = args.sm_count or torch.cuda.get_device_properties(0).multi_processor_count
        started = time.time()
        cells, compiles, executed = run_sweep(
            args, cfg, grid, block_sizes, csv_path, cache_root, b, pinned,
            prov=prov)
        print(f"\nswept in {time.time() - started:.0f} s")

    sm_source = ("given on the command line" if args.sm_count
                 else "reported by the driver" if args.self_test is None
                 else f"assumed H200 default {DEFAULT_SM_COUNT}")
    report = analyse(cells, cfg, block_sizes=block_sizes, alpha=alpha,
                     ridge=rr.ridge, bandwidth_gbps=bandwidth, b=b,
                     model_name=args.model, dtype=args.dtype, compiles=compiles,
                     executed=executed, sm_count=sm_count, sm_source=sm_source,
                     pinned=pinned, ridge_band=rr.band, ridge_source=ridge_src,
                     ridge_band_source=rr.band_source, capability=capability,
                     card=card, bandwidth_source=bw_src, prov=prov)
    print(report.text())

    (out_dir / "report.txt").write_text(report.text())
    (out_dir / "report.json").write_text(json.dumps(report.payload, indent=2))
    print(f"cells    {csv_path}")
    print(f"report   {out_dir / 'report.txt'}")
    print(f"json     {out_dir / 'report.json'}")
    print("These survive pod teardown when the results root is on the network "
          "volume, which is what the WRITES TO line above says.")

    # THE EXIT CODE COMES FROM THE SHARED TABLE, over the SAME gate objects that
    # printed the RESULT lines, so `exit_codes.classify_text` on this log
    # recomputes the code the process returned. Two integers meaning two
    # different things in two files is the defect that module is named against.
    rc = exit_codes.classify(g.scored() for g in report.gates)
    if rc == exit_codes.CLAIM_FAIL and not args.fail_on_gate:
        print(f"exit     {exit_codes.describe(exit_codes.DONE)} "
              f"(a claim gate did not pass, which is a RESULT; pass "
              f"--fail-on-gate to exit "
              f"{exit_codes.CLAIM_FAIL} CLAIM_FAIL on it)")
        return exit_codes.DONE
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
