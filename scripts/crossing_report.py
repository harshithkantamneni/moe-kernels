#!/usr/bin/env python3
"""Where does a sweep actually cross its ridge, and does C2 predict it?

Reads one or more result CSVs, aggregates to one median per (model, dtype,
token count), recovers the crossing from measured TIME via moe.bench.crossing,
and prints it beside the `2R/b` prediction.

The measured side never consults the byte model, so the prediction can be wrong.

    python scripts/crossing_report.py /workspace/results/run_h200fp8b_vllm.csv \
        --ridge 152.8 --impl vllm_fused_experts

152.8 is the H200's OWN triad ridge, off `moe/bench/hardware/
measured_nvidia_h200.yaml`. It read 162.8 until the 2026-09-09 calibration
sampled the dense GEMM's clock while it ran instead of after it, which is the
honest reading and the low one; a usage line is read as a recommendation, so
it names whatever that file holds. This line used to read `--ridge 160.3`, which is a
2026-08-26 H200 figure that ended up quoted for an A100 arm too; it was
withdrawn from all 26 published reports on 2026-09-02. `--ridge` is required
and has no default, so a run that cannot name its card's ridge does not start.

`--uncertainty` adds a 90% band, propagated from the replicate spread each token
count already carries. Off by default so existing output is unchanged, but every
crossing this study has quoted was quoted bare, and on a flat curve the
interpolation multiplies a 6% timing wobble into a 35% move in the answer.

SINCE 2026-09-02 IT BANDS EVERY QUOTED RATIO, not only the headline crossing:
each staircase step against the prediction, last over first, and the max-affine
comparison. `docs/FINDINGS.md`'s "1.149 +/- 0.069" is the sd of eight ratios
with none of their own intervals propagated, and `docs/STUDY.md` reads a 1-12%
span-vs-whole-layer agreement as a confirmation while each crossing in it
carries a 15-35% band. Both live inside the noise of what they are built from
(audit S39, fix B9). The header also states an MDE from the corpus's own
replicate spread, and the run prints the dirty share of the rows it admitted,
which no analysis path in this repository had ever read (A6).

EVERY crossing is printed, not the first. 8 of the 16 canonical uniform cells
cross 0.5 going up more than once, because the curve is a staircase in M-tiles
per expert rather than one flat-to-linear transition, and a cell that crosses
twice is labelled a staircase here rather than reduced to whichever step the
token grid sampled first. The M-tile count sits beside every token count so the
steps are visible in the table itself; `--block-m` names the tile it is counted
at, since the published arms predate the column that records the tile actually
run.

`--max-affine` adds the OTHER estimator beside this one: TEMPO's global
`t = max(a + b G, c + beta N)`, which has one inflection by construction and
therefore cannot have the multiple-crossing failure the slope threshold has. Off
by default, and the default output is byte for byte what it was without it. Read
`moe/bench/maxaffine.py` before quoting the number it prints: on the five-stage
cells it fits to a mean relative error of 24% and lands three times below the
ridge, so it removes the ambiguity without resolving it.

THE CLOCK GATE HERE IS `throttled`, AND ON A v6 ROW THAT WORD MEANS LOW OR
DRIFT. `moe/bench/driver.py` sets it from the under-load verdicts and keeps a
LEVEL failure on the HIGH side out of it (a cell boosted above the reference
clock is not a thermal event; its time is the kernel's), so a boosted
memory-bound cell is KEPT by this report and counted beside the kept total. What
a HIGH-side row cannot be quoted for is its FIXED-roof fraction, and this report
prints both compute-side fractions per cell: `pct_of_achieved_tflops` against the
calibration's roof, and `pct_of_roof_at_cell_clock` against the roof at the clock
the cell ran, which exists only on a v6 row the driver scored and reads "not
available (v<6 row)" on the committed corpus. Until 2026-09-08 no script read
that column, so the correction 03df2d4 wrote had no reader.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe.bench.crossing import (  # noqa: E402
    BAND_QUANTILES,
    DEFAULT_DRAWS,
    MIN_RELATIVE_SPREAD,
    ROUTING_COLUMN,
    STORED_TILE_EFF,
    crossing_interval,
    local_slopes,
    m_tiles_for_row,
    relative_spread,
    routing_domain,
    timed_rows,
    upcrossings,
)

# ONE PERCENTILE ESTIMATOR FOR EVERY BAND THIS REPORT PRINTS. `crossing_interval`
# bands the first crossing and `CellBands` below bands all of them, and the two
# appear within four lines of each other in the output. Two different quantile
# conventions -- this module's linear interpolation against
# `statistics.quantiles`' exclusive method -- put those numbers a percent or two
# apart on the same cell, for no reason a reader could ever work out. So the
# module's own estimator is imported, private name and all, rather than
# reimplemented next to it.
from moe.bench.crossing import _percentile as _band_percentile  # noqa: E402
from moe.bench.maxaffine import (  # noqa: E402
    PADDED_ROWS,
    RELATIVE,
    TOKENS,
    WEIGHTINGS,
    Comparison,
    MaxAffineFit,
    fit_rows,
)
from moe.bench.published import (  # noqa: E402
    dirty_share_line,
    filter_superseded,
    superseded_impls,
    superseded_reason,
    two_sample_mde,
)
from moe.bench.ridge import (  # noqa: E402
    crossing_batch,
    ridge_for_dtype,
    rows_per_expert,
    saturation_batch,
)
from moe.bench.schema import (  # noqa: E402
    UNRECORDED,
    TileConfigUnrecorded,
    has_cell_clock_roof,
)
from moe.routing.imbalance import TileEfficiencyUndetermined  # noqa: E402


class CellBands:
    """Every crossing in one cell, each with its own 90% band, from ONE shake.

    WHY EVERY RATIO NEEDS ONE. `docs/FINDINGS.md` quotes "1.149 +/- 0.069" for
    the whole-layer over span comparison; that 0.069 is the sd of eight ratios
    with no propagation of each crossing's own interval, which is 15-35% per
    crossing because the interpolation's leverage is `1/(s1 - s0)` and the curve
    is flat. `docs/STUDY.md` then reads a 1-12% agreement as a confirmation. Both
    numbers live inside the noise of the crossings they are built from, and
    neither had an interval that said so (audit S39, fix B9).

    ONE SHAKE, NOT ONE PER CROSSING. The crossings of a cell are functions of the
    same measured points, so `last / first` is a ratio of two dependent
    quantities: banding them separately and dividing the ends would give a band
    for a pair of independent crossings, which these are not. Every draw here
    perturbs the whole curve once and reads every crossing off the result, so a
    ratio's band is the distribution of the ratio.

    MATCHED BY STEP, NOT BY INDEX. A draw can lose or gain a crossing, so the
    k-th crossing of a draw is not necessarily the k-th of the medians. Each
    measured crossing is matched to the draw's crossing on the SAME grid
    interval, `Upcrossing.step_lo`/`step_hi`, and draws that produced none there
    are dropped. That makes every band CONDITIONAL on the crossing surviving,
    and `kept` says how often it did: a crossing recovered by a tenth of the
    draws is not a well-located crossing however tight the band around the
    survivors looks.
    """

    def __init__(self, replicates, min_tokens: float,
                 draws: int = DEFAULT_DRAWS, seed: int = 0):
        points, sigmas = [], []
        for t, reps in replicates:
            vals = [float(v) for v in reps if float(v) > 0.0]
            if not vals:
                continue
            points.append((float(t), statistics.median(vals)))
            sigmas.append(max(relative_spread(vals), MIN_RELATIVE_SPREAD))
        #: The medians and the per-point sigmas the shake was built from, kept
        #: so a reader chasing a band can see its inputs without re-deriving
        #: them from the CSVs.
        self.points = points
        self.sigmas = sigmas
        self.measured = upcrossings(points, min_tokens=min_tokens)
        rng = random.Random(seed)
        self.draws: list[list] = []
        for _ in range(draws):
            shaken = [(t, ms * rng.lognormvariate(0.0, sigma))
                      for (t, ms), sigma in zip(points, sigmas, strict=True)]
            self.draws.append(upcrossings(shaken, min_tokens=min_tokens))
        self.n_draws = len(self.draws)

    def band(self, index: int) -> tuple[float, float, int] | None:
        """`(lo, hi, draws that kept this crossing)`, or None if none did."""
        if index >= len(self.measured):
            return None
        step = (self.measured[index].step_lo, self.measured[index].step_hi)
        samples = sorted(u.tokens for draw in self.draws for u in draw
                         if (u.step_lo, u.step_hi) == step)
        if not samples:
            return None
        return (_band_percentile(samples, BAND_QUANTILES[0]),
                _band_percentile(samples, BAND_QUANTILES[1]), len(samples))

    def ratio_band(self, index: int, denominator: float
                   ) -> tuple[float, float, int] | None:
        """The band on `crossing / denominator`, for a FIXED denominator.

        The prediction `2R/b` is arithmetic over the model shape and the ridge,
        so it carries none of this cell's timing noise and divides straight
        through. A ratio against another MEASURED crossing is `last_over_first`,
        which does not.
        """
        band = self.band(index)
        if band is None or denominator <= 0:
            return None
        lo, hi, kept = band
        return lo / denominator, hi / denominator, kept

    def last_over_first(self) -> tuple[float, float, int] | None:
        """The band on `last crossing / first crossing`, jointly propagated.

        None when the medians give fewer than two crossings, or when no draw
        gave two. Only draws with at least two crossings contribute, so this is
        conditional on the staircase surviving, which is the same conditioning
        every band here carries and is stated for the same reason.
        """
        if len(self.measured) < 2:
            return None
        samples = sorted(draw[-1].tokens / draw[0].tokens
                         for draw in self.draws
                         if len(draw) >= 2 and draw[0].tokens > 0)
        if not samples:
            return None
        return (_band_percentile(samples, BAND_QUANTILES[0]),
                _band_percentile(samples, BAND_QUANTILES[1]), len(samples))


def tile_cell(counted: list[float] | None, previous: float | None) -> str:
    """One table cell: the median M-tile count, its step, and its disagreement.

    Medianed across replicates the same way `ms_p50` is, because the two have to
    describe the SAME cell: a tile count taken off one row beside a time taken
    off eight would put a step where the timed curve has none. Uniform routing
    is sampled per replicate, so the counts genuinely differ -- mixtral at
    T=1024 draws 19 tiles on two rows and 21 on four -- and `~` says so rather
    than letting a rounded median look exact.

    `--` when nothing determined it, never a blank: an empty cell in a column of
    numbers reads as zero steps, which is the claim this column exists to test.
    """
    if not counted:
        return "--"
    median = statistics.median(counted)
    spread = "~" if len({round(v) for v in counted}) > 1 else ""
    step = ""
    if previous is not None and round(median) != round(previous):
        step = f" ({round(median) - round(previous):+d})"
    return f"{round(median)}{spread}{step}"


def print_staircase(found: list, predicted: float,
                    cell_tiles: dict[int, list[float]],
                    bands: CellBands | None = None) -> None:
    """Every crossing, and what a cell with more than one of them is.

    Silent for a cell that crosses once, so the output of a single-crossing run
    keeps its shape. Loud for the rest, because the alternative -- printing one
    number off a curve that supplies several -- is how a tile step got quoted
    as a ridge crossing for the length of this study.

    Names no winner. The last crossing has the tighter distribution and to that
    extent the better claim on the ridge (rows per expert at the last: mean
    175.8, CV 21.2%; at the first: 123.4, CV 40.0%), and choosing on that here
    would bake a preference into a report whose job is to show that the sweep
    does not resolve it. A dense token grid resolves it.

    Those two means were quoted against a band of 160.3-176.2 until 2026-09-02,
    which flattered the last crossing by sitting it inside the band. Both ends
    of that band are H200 numbers from two calibrations of the same card whose
    compute ceilings disagree by 9.9%, so its width was the ceiling failing to
    reproduce and not the ridge being uncertain, and it was withdrawn from every
    published report. Against the H200's own band, 142.8-155.4 off its committed
    calibration (152.1-165.6 before the 2026-09-09 recalibration moved the
    card's ridge from 162.8 to 152.8), the last crossing sits ABOVE and the
    first below: the band contains neither, and the sweep pins the ridge even
    less well than the old figure made it look.
    """
    if len(found) < 2:
        return
    print(f"  STAIRCASE: the slope crosses 0.5 upward at {len(found)} token "
          "counts, not one.")
    for i, u in enumerate(found, 1):
        lo = tile_cell(cell_tiles.get(int(u.step_lo)), None)
        hi = tile_cell(cell_tiles.get(int(u.step_hi)), None)
        print(f"    {i} of {len(found)}: {u.tokens:8.0f} tokens   "
              f"{u.tokens / predicted:.2f}x predicted   "
              f"on the step T {u.step_lo:.0f} -> {u.step_hi:.0f}, "
              f"M-tiles {lo} -> {hi}")
        # EVERY RATIO ON THIS PAGE CARRIES ITS OWN BAND, not just the first
        # crossing's. These per-step lines are what the staircase argument is
        # read off, and until now they were the only bare numbers left in a
        # report that bands its headline.
        if bands is not None:
            rb = bands.ratio_band(i - 1, predicted)
            if rb is None:
                print("             90% band: no draw kept a crossing on this "
                      "step; the step does not survive its own noise")
            else:
                blo, bhi, kept = rb
                print(f"             90% band: {blo:.2f}-{bhi:.2f}x predicted "
                      f"({kept} of {bands.n_draws} draws kept this step)")
    ratio = found[-1].tokens / found[0].tokens
    print(f"    last over first: {ratio:.2f}x. M-tiles per expert is a step")
    # ON ITS OWN LINE, above the prose it belongs to, so the banded run is the
    # plain run plus lines: see the banner note in `main`. Jointly propagated,
    # because the two crossings are functions of the same measured points and a
    # ratio of two separately banded numbers would be a band for a pair that
    # does not exist.
    lof = bands.last_over_first() if bands is not None else None
    if lof is not None:
        blo, bhi, kept = lof
        print(f"      90% band on last over first: {blo:.2f}-{bhi:.2f}x "
              f"({kept} of {bands.n_draws} draws kept two crossings)")
    print("    function of T, and each extra tile is another pass over that "
          "expert's weights,")
    print("    so the slope spikes above 0.5 at every step and sags below it "
          "on every tread.")
    print("    A first crossing can be a tile step rather than a roofline "
          "transition; which")
    print("    of these is the ridge is not decided by this grid.")


#: The only columns `moe.bench.maxaffine` reads. Projected rather than keeping
#: the whole row for the same reason the routing census is: a published row is
#: 94 columns and there are up to 70k of them, and this script already refuses
#: to hold them. Nothing is retained at all unless `--max-affine` is passed.
AFFINE_COLUMNS = ("num_tokens", "ms_p50", "load_active_experts",
                  "load_total_rows", "load_max_rows",
                  "load_tile_eff_bm64", "load_tile_eff_bm128")


def print_max_affine(fit: MaxAffineFit | None, comparison: Comparison,
                     model: str, predicted: float,
                     bands: CellBands | None = None) -> dict | None:
    """The second estimator's answer for this cell, with its residuals attached.

    Prints the fit quality on the SAME lines as the inflection, never below a
    fold and never optional. A max-affine inflection is only worth more than the
    slope detector's several if the model describes the curve, and on the
    five-stage cells here it misses by a mean of 24% and a p95 of up to 263%. A
    reader who copies one line out of this report has to copy the residual with
    it, which is the lesson of every retraction in docs/FINDINGS.md.

    Returns the row the head-to-head table is built from, or None when there was
    no fit, so the summary is assembled from the same numbers that were printed
    rather than recomputed beside them.
    """
    if fit is None:
        print("  MAX-AFFINE: not fitted. Fewer than four token counts, or a "
              "padded regressor")
        print("    this row set cannot determine at the named BLOCK_M.")
        return None
    mem, comp = fit.memory, fit.compute
    n = len(fit.observations)
    b = ("unidentifiable, G never moved here" if mem.degenerate
         else f"{mem.slope:.6f} ms per active expert")
    beta = ("unidentifiable, N never moved here" if comp.degenerate
            else f"{comp.slope:.8f} ms per unit N")
    print("  MAX-AFFINE (TEMPO-style global fit): t = max(a + b G, c + beta N)")
    print(f"    G = active experts (observed column); {fit.compute_side.label}")
    print(f"    residuals weighted {fit.weighting}, split searched over all "
          f"{n - 3} contiguous partitions")
    print(f"    memory  branch:  a = {mem.intercept:9.4f} ms   b    = {b}"
          f"   [{fit.n_memory} of {n} points]")
    print(f"    compute branch:  c = {comp.intercept:9.4f} ms   beta = {beta}"
          f"   [{fit.n_compute} of {n} points]")
    inflection = fit.inflection
    rpe = None
    if inflection is None:
        print(f"    inflection:      none on this grid "
              f"({len(fit.inflections)} upward crossings)")
    else:
        rpe = rows_per_expert(model, inflection)
        print(f"    inflection:      {inflection:8.0f} tokens   "
              f"{inflection / predicted:.2f}x predicted   "
              f"{rpe:.1f} rows per expert")
    print(f"    fit quality:     mean |rel err| {fit.mean_rel_err:.1%}   "
          f"p95 {fit.p95_rel_err:.1%}   max {fit.max_rel_err:.1%}")
    if not fit.single_crossing:
        print(f"    NOT SINGLE-CROSSING: the fitted max also crosses back "
              f"{fit.reversals} time(s) at")
        print("      small T, where G rises steeply and N has barely moved. "
              "One inflection is a")
        print("      property of two planes, not of the path a sweep walks "
              "through them.")
    found = comparison.slope_crossings
    if found:
        listed = ", ".join(f"{t:.0f}" for t in found)
        print(f"    against the slope detector: {len(found)} crossing(s) "
              f"({listed})")
        first, last = comparison.ratio_to(comparison.first), \
            comparison.ratio_to(comparison.last)
        if first is not None:
            print(f"      max-affine is {first:.2f}x the first and "
                  f"{last:.2f}x the last")
            # The slope crossings in that ratio are measured and carry the
            # cell's timing noise; the max-affine inflection is a fit to the
            # same points and carries its own, which is NOT propagated here.
            # Inverting the measured crossing's band gives the half of the
            # uncertainty this report can compute, and the line says it is a
            # half rather than pretending to be the whole.
            infl = fit.inflection
            if bands is not None and infl is not None:
                # One crossing means first IS last, and printing the same band
                # twice under two headings reads as two pieces of evidence.
                wanted = [("first", 0)] if len(found) < 2 else \
                    [("first", 0), ("last", len(found) - 1)]
                for label, index in wanted:
                    band = bands.band(index)
                    if band is None:
                        continue
                    blo, bhi, kept = band
                    if blo <= 0 or bhi <= 0:
                        continue
                    print(f"      against the {label} crossing's own 90% band: "
                          f"{infl / bhi:.2f}-{infl / blo:.2f}x "
                          f"({kept} of {bands.n_draws} draws); the "
                          "inflection's own noise is not in this")
    return {"first": comparison.first, "last": comparison.last,
            "n_crossings": len(found), "inflection": inflection, "rpe": rpe,
            "ratio_first": comparison.ratio_to(comparison.first),
            "ratio_last": comparison.ratio_to(comparison.last),
            "mean": fit.mean_rel_err, "p95": fit.p95_rel_err,
            "single": fit.single_crossing}


def print_head_to_head(summary: list[tuple[str, dict]]) -> None:
    """Both estimators for every cell in one table, then what they disagree by.

    The per-cell blocks above are 40 lines apart, and a reader comparing two
    estimators across sixteen cells cannot hold them. This is the same numbers
    in one place -- assembled from the dicts those blocks returned, so the table
    and the blocks cannot drift.

    Reports the disagreement as a distribution rather than a mean, because the
    interesting part is its width: the two estimators agree within 20% on the
    smooth one-stage curves and differ by up to 5x on the stepped five-stage
    ones, and a single average over both would hide exactly that.
    """
    print("=== ESTIMATOR HEAD TO HEAD: slope threshold against max-affine ===")
    print("  The slope detector's crossings (all of them, first and last shown) "
          "against the")
    print("  max-affine inflection, on the SAME medians. R is rows per expert "
          "at the")
    print("  inflection, T k / E. `1x` is whether the fitted max crosses this "
          "grid once.")
    print(f"  {'cell':44} {'first':>7} {'last':>7} {'n':>2} "
          f"{'affine':>7} {'x1st':>6} {'xlast':>6} {'R':>7} "
          f"{'mean%':>6} {'p95%':>7}  1x")
    for label, row in summary:
        def num(value, fmt):
            return "--" if value is None else format(value, fmt)
        print(f"  {label:44} {num(row['first'], '7.0f'):>7} "
              f"{num(row['last'], '7.0f'):>7} {row['n_crossings']:>2} "
              f"{num(row['inflection'], '7.0f'):>7} "
              f"{num(row['ratio_first'], '6.2f'):>6} "
              f"{num(row['ratio_last'], '6.2f'):>6} "
              f"{num(row['rpe'], '7.1f'):>7} "
              f"{row['mean'] * 100:6.1f} {row['p95'] * 100:7.1f}  "
              f"{'yes' if row['single'] else 'no'}")
    rows = [r for _, r in summary]
    ambiguous = [r for r in rows if r["n_crossings"] > 1]
    resolved = [r for r in ambiguous if r["inflection"] is not None]
    print(f"\n  {len(ambiguous)} of {len(rows)} cells give the slope detector "
          f"more than one crossing.")
    print(f"  Max-affine returns exactly one inflection on {len(resolved)} of "
          f"those, and is")
    print(f"  single-crossing along the grid on "
          f"{sum(1 for r in rows if r['single'])} of {len(rows)} cells overall.")
    for name, key in (("first", "ratio_first"), ("last ", "ratio_last")):
        vals = sorted(r[key] for r in rows if r[key] is not None)
        if vals:
            print(f"  max-affine over the {name} crossing: median "
                  f"{statistics.median(vals):.2f}, range {vals[0]:.2f} to "
                  f"{vals[-1]:.2f} over {len(vals)} cells")
    means = sorted(r["mean"] for r in rows)
    p95s = sorted(r["p95"] for r in rows)
    print(f"  fit quality across cells: mean |rel err| median "
          f"{statistics.median(means):.1%} (range {means[0]:.1%} to "
          f"{means[-1]:.1%}),")
    print(f"  p95 median {statistics.median(p95s):.1%} (range {p95s[0]:.1%} to "
          f"{p95s[-1]:.1%}). A fit that misses by that much is not locating a "
          "ridge.")


#: The word the instrument writes in `clock_level_side` for a boosted cell:
#: `timing.LEVEL_HIGH`, spelled here because `moe/bench/timing.py` imports
#: torch at module scope and this report runs off-GPU from a CSV.
LEVEL_HIGH = "high"

#: What is printed for the corrected fraction on a row that predates it.
ROOF_PREDATES = "not available (v<6 row)"


def is_level_high(row: dict) -> bool:
    """Did this row's LEVEL verdict fail on the HIGH side? False on a pre-v6 row."""
    side = row.get("clock_level_side")
    return side is not None and side != UNRECORDED and str(side) == LEVEL_HIGH


def roof_fractions(row: dict) -> tuple[float | None, float | None, str]:
    """`(pct_of_achieved_tflops, pct_of_roof_at_cell_clock, why the second is absent)`.

    None, never 0.0, for a fraction that is not on the row: 0.0 is the driver's
    "not scored" for both columns and a median over it is a fraction of nothing.
    The first is None when the run had no compute ceiling for the dtype. The
    second exists only on a v6 row the driver scored (`has_cell_clock_roof`);
    the reason is the driver's own `roof_note` on a v6 row it refused,
    `ROOF_PREDATES` on a row from before the column, and "" when scored.
    """
    try:
        fixed = float(row.get("pct_of_achieved_tflops") or 0.0)
    except (TypeError, ValueError):
        fixed = 0.0
    fixed_or_none = fixed if fixed > 0.0 else None
    if has_cell_clock_roof(row):
        return fixed_or_none, float(row["pct_of_roof_at_cell_clock"]), ""
    note = row.get("roof_note")
    if "roof_at_cell_clock_tflops" not in row or note is None or note == UNRECORDED:
        return fixed_or_none, None, ROOF_PREDATES
    return fixed_or_none, None, (str(note) or "not scored, and the driver "
                                              "recorded no reason")


def print_roof_fractions(fixed: list[float], cell: list[float],
                         absent: collections.Counter, high: int, total: int) -> None:
    """The two compute-side fractions of one cell, medianed, side by side.

    Printed under every cell's table because the fraction of roof is what a
    reader takes off a row by hand, and the one they would take (the fixed
    roof) is the one that is inflated on a boosted cell.
    """
    print("  fraction of compute roof (median over the rows in this cell):")
    if fixed:
        print(f"    pct_of_achieved_tflops    (FIXED roof, calibration clock):  "
              f"{statistics.median(fixed):6.2f}%  over {len(fixed)} rows")
    else:
        print("    pct_of_achieved_tflops    (FIXED roof, calibration clock):  "
              "not scored (no compute ceiling for this dtype)")
    if cell:
        print(f"    pct_of_roof_at_cell_clock (roof AT THE CELL'S CLOCK):      "
              f"{statistics.median(cell):6.2f}%  over {len(cell)} rows")
    for note, count in absent.most_common():
        print(f"    pct_of_roof_at_cell_clock (roof AT THE CELL'S CLOCK):      "
              f"{note}  [{count} rows]")
    if high:
        print(f"    {high} of {total} rows failed LEVEL HIGH (boosted above the "
              "reference clock): kept,")
        print("    their fixed-roof fraction is not comparable; use "
              "pct_of_roof_at_cell_clock")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--ridge", type=float, required=True,
                    help="measured bf16 FLOP/byte from the calibration; other "
                         "dtypes are scaled by their FLOP rate, since a format "
                         "changes the peak as well as the bytes")
    ap.add_argument("--impl", default=None, help="restrict to one implementation")
    ap.add_argument("--routing", default=None, help="restrict to one routing kind")
    ap.add_argument("--include-throttled", action="store_true",
                    help="keep rows whose `throttled` column is True. On a v6 "
                         "row that word means the under-load LEVEL check failed "
                         "LOW or the DRIFT check failed; a LEVEL failure on the "
                         "HIGH side (a boosted cell) never sets it and such "
                         "rows are kept without this flag. On a pre-v5 row it "
                         "is the retired idle-instant flag")
    ap.add_argument("--l2-flush", choices=["true", "false"], default=None,
                    help="restrict to one L2 mode; default mixes both")
    ap.add_argument("--cuda-graph", choices=["true", "false"], default=None,
                    help="restrict to one capture mode; default mixes both")
    ap.add_argument("--block-m", type=int, default=128,
                    help="BLOCK_M the M-tile column is counted at (default "
                         "128). The published arms predate the column that "
                         "records the tile actually run, so this is a stated "
                         "assumption rather than a measurement; 64 and 128 read "
                         "a stored efficiency, any other value is "
                         "reconstructed and refuses once an expert spans tiles")
    ap.add_argument("--max-affine", action="store_true",
                    help="also fit TEMPO's global t = max(a + b G, c + beta N) "
                         "per cell and print both estimators side by side. Off "
                         "by default and the default output is unchanged; this "
                         "estimator has one inflection by construction, which "
                         "is the reason to try it and the reason to check its "
                         "residuals before quoting it")
    ap.add_argument("--max-affine-n", choices=[TOKENS, PADDED_ROWS],
                    default=TOKENS,
                    help="the compute-side regressor. `tokens` is TEMPO's own "
                         "and is a recorded column; `padded_rows` rounds up to "
                         "whole M-tiles at --block-m, fits better, and is "
                         "DERIVED -- below saturation it equals BLOCK_M x "
                         "active experts exactly, so the two branches go "
                         "collinear and the inflection stops meaning anything")
    ap.add_argument("--max-affine-weighting", choices=list(WEIGHTINGS),
                    default=RELATIVE,
                    help="`relative` weights each residual by 1/ms^2 so the "
                         "objective is relative error; `absolute` is textbook "
                         "least squares, which on a curve spanning 0.13 to "
                         "27.9 ms is a fit to the largest token count alone")
    ap.add_argument("--uncertainty", action="store_true",
                    help="add a 90%% band to each measured crossing, Monte "
                         "Carlo'd from the replicate spread of each token "
                         "count. Off by default so existing output is byte for "
                         "byte what it was")
    args = ap.parse_args()

    # A superseded arm holds the SAME measurements as the one that replaced it,
    # so reading both weights every one of its rows twice. Announced rather than
    # silent: a dropped input nobody sees is the same class of error.
    csvs, dropped = filter_superseded(args.csvs)
    for d in dropped:
        print(f"[skip] {d.parent.name}: {superseded_reason(d).splitlines()[0]}")
    partial = {}
    for c in csvs:
        names = superseded_impls(c)
        if names:
            partial[c] = names
            print(f"[skip] {c.parent.name}: {', '.join(sorted(names))} "
                  f"({superseded_reason(c).splitlines()[0]})")
    if dropped or partial:
        print()
    if not csvs:
        print("every input was superseded; nothing to report")
        return 1

    cells: dict[tuple[str, str], dict[int, list[float]]] = {}
    # One tiny mapping per kept row, so `routing_domain` sees rows and the
    # census is of what actually reached a median rather than of what the file
    # holds. Only the routing column is kept: the full rows are 200 columns
    # wide and there are up to 70k of them across the published arms.
    routing_rows: dict[tuple[str, str, str], list[dict]] = {}
    # M-tiles per row, reduced to a number at ingest for the same reason: the
    # count is one float and the row it came from is 200 columns.
    tiles: dict[tuple[str, str, str], dict[int, list[float]]] = {}
    # Projected rows for the max-affine fit, and only when it was asked for:
    # with the flag off nothing here is populated and the report allocates
    # exactly what it did before.
    affine: dict[tuple[str, str, str], dict[int, list[dict]]] = {}
    modes: collections.Counter = collections.Counter()
    # THE DIRTY COLUMN, COUNTED AS ROWS ARE ADMITTED, not re-read afterwards.
    # `git_dirty` is the only thing a row carries about whether the code that
    # produced it can be recovered, and no analysis path in this repository had
    # ever looked at it (audit A6). One tiny dict per kept row rather than the
    # row itself: a published row is 94 columns and there are up to 70k of them.
    admitted_dirty: list[dict] = []
    # THE READER FOR THE PER-ROW ROOF, reduced at ingest like everything else
    # here: two floats and a reason per kept row, keyed by cell.
    roof_fixed: dict[tuple[str, str, str], list[float]] = {}
    roof_cell: dict[tuple[str, str, str], list[float]] = {}
    roof_absent: dict[tuple[str, str, str], collections.Counter] = {}
    level_high: collections.Counter = collections.Counter()
    cell_rows: collections.Counter = collections.Counter()
    kept = skipped = untimed = tileless = 0
    for path in csvs:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
            timed = timed_rows(rows)
            untimed += len(rows) - len(timed)
            skip_impls = partial.get(path, set())
            for r in timed:
                if r["impl"] in skip_impls:
                    skipped += 1
                    continue
                if args.impl and r["impl"] != args.impl:
                    continue
                if args.routing and r["routing_kind"] != args.routing:
                    continue
                if r.get("correctness_passed") not in ("True", "true", "1", ""):
                    skipped += 1
                    continue
                if not args.include_throttled and r.get("throttled") in ("True", "true", "1"):
                    skipped += 1
                    continue
                if args.l2_flush is not None and \
                        str(r.get("l2_flush", "")).lower() != args.l2_flush:
                    continue
                if args.cuda_graph is not None and \
                        str(r.get("cuda_graph", "")).lower() != args.cuda_graph:
                    continue
                modes[(str(r.get("l2_flush")), str(r.get("cuda_graph")))] += 1
                try:
                    t, ms = int(r["num_tokens"]), float(r["ms_p50"])
                except (ValueError, KeyError):
                    continue
                # `impl` is in the key, not just an optional filter. Rows from
                # different implementations measure different SCOPES -- one
                # stage against five against a whole layer, 16.7x apart on the
                # published sweep -- so a median across them describes nothing.
                key = (r["model"], r["dtype"], r["impl"])
                cells.setdefault(key, {}).setdefault(t, []).append(ms)
                routing_rows.setdefault(key, []).append(
                    {ROUTING_COLUMN: r.get(ROUTING_COLUMN, "")})
                if args.max_affine:
                    affine.setdefault(key, {}).setdefault(t, []).append(
                        {c: r.get(c, "") for c in AFFINE_COLUMNS})
                admitted_dirty.append({"git_dirty": r.get("git_dirty", "")})
                fixed, at_cell, why = roof_fractions(r)
                if fixed is not None:
                    roof_fixed.setdefault(key, []).append(fixed)
                if at_cell is not None:
                    roof_cell.setdefault(key, []).append(at_cell)
                else:
                    roof_absent.setdefault(key, collections.Counter())[why] += 1
                cell_rows[key] += 1
                if is_level_high(r):
                    level_high[key] += 1
                try:
                    tiles.setdefault(key, {}).setdefault(t, []).append(
                        m_tiles_for_row(r, args.block_m))
                except (TileEfficiencyUndetermined, TileConfigUnrecorded,
                        KeyError, ValueError):
                    # A row with no load columns, or one whose experts span
                    # more tiles than the stored efficiencies pin down. Counted
                    # and announced: an empty column with no explanation reads
                    # as "no steps here", which is the opposite of the truth.
                    tileless += 1
                kept += 1

    print(f"kept {kept} rows, skipped {skipped} (throttled or failed), "
          f"{untimed} never timed (skipped graph mode: ms_p50 is 0.0, "
          f"which is not a measurement)")
    # `throttled` on a v6 row is LOW or DRIFT; a HIGH-side LEVEL failure is a
    # boosted cell, kept, and said so here rather than folded into either count.
    n_high = sum(level_high.values())
    print(f"  of the kept rows, {n_high} failed LEVEL HIGH (boosted above the "
          "reference clock): kept, their")
    print("  fixed-roof fraction is not comparable; use pct_of_roof_at_cell_clock")
    print(dirty_share_line(admitted_dirty, "admitted rows"))
    if len(modes) > 1:
        print("  timing modes mixed into each median (l2_flush, cuda_graph): "
              + ", ".join(f"{k}x{v}" for k, v in sorted(modes.items())))
        print("  pass --l2-flush/--cuda-graph to isolate one; the crossing is a "
              "slope, so mixing adds spread rather than bias")
    # The staircase is the reason a crossing can be a tile step, so the column
    # that shows it gets explained once, above the cells, rather than being an
    # unlabelled number a reader has to reverse-engineer.
    stored = " (a stored column)" if args.block_m in STORED_TILE_EFF else \
             " (reconstructed)"
    print(f"  M-tiles at BLOCK_M {args.block_m}{stored}: "
          "ceil(rows_per_expert / BLOCK_M) summed")
    print("  over active experts. Each extra tile is another pass over that "
          "expert's weights,")
    print("  so time steps where the count steps and flatlines where it holds. "
          "(+n) is the")
    print("  step from the row above; ~ marks replicates that drew different "
          "tile counts.")
    if tileless:
        print(f"  {tileless} rows carry no usable M-tile count and are absent "
              "from that column only:")
        print("  a row records no routing load, or an expert spans more tiles "
              "than the stored")
        print("  efficiencies pin down, and a reconstructed count there would "
              "be a guess.")
    if args.uncertainty:
        print(f"  bands are {DEFAULT_DRAWS} draws per cell, each token count "
              f"shaken by its own replicate spread; the crossing interpolates "
              f"between")
        print("  two slopes with leverage 1/(s1-s0), so it is widest exactly "
              "where the curve is flattest")
        # EVERY LINE OF THIS BANNER OPENS WITH "bands are ", and that is a
        # contract rather than a stylistic tic. `--uncertainty` is required to
        # ADD lines and rewrite none, and `tests/test_crossing_uncertainty.py`
        # enforces it by stripping the banded run of every line matching
        # "90% band", "bands are " or "two slopes with leverage" and comparing
        # what is left with the plain run. The same file also asserts that
        # EXACTLY ONE line carries "90% band" on a single-crossing cell, which
        # is the headline band, so a banner line may not use that phrase.
        print("  bands are on EVERY quoted ratio below, not just the headline "
              "crossing:")
        print("  bands are printed per staircase step, on last over first, and "
              "on max-affine")
    # Louder than the timing-mode note above, and deliberately not a change of
    # default: pooled rows do not make the crossing noisier, they make it a
    # crossing of nothing. Silently switching to `--routing uniform` here would
    # make this report answer differently than it did yesterday with no flag
    # changed, which is its own failure mode -- so it warns and names the flag.
    routing = routing_domain(itertools.chain.from_iterable(routing_rows.values()))
    banner = routing.warning_lines()
    if banner:
        print()
        for line in banner:
            print(line)
    print()
    if not cells:
        print("nothing to report")
        return 1

    # THE MDE, FROM A STATED NOISE ASSUMPTION, BEFORE THE FIRST CELL. Audit
    # B14: no arm in this study states one, so no ratio it prints has ever been
    # set against the smallest ratio the design could resolve. The assumption is
    # the corpus's own replicate spread, not a guess: `relative_spread` over
    # each token count's replicates, medianed across every cell about to be
    # reported. The crossing then AMPLIFIES that by its own leverage, which is
    # why this is a floor and the per-cell bands below are the real widths.
    spreads = []
    for by_t in cells.values():
        for reps in by_t.values():
            vals = [float(v) for v in reps if float(v) > 0.0]
            if vals:
                spreads.append(max(relative_spread(vals), MIN_RELATIVE_SPREAD))
    if spreads:
        sd_rel = statistics.median(spreads)
        print(f"  noise assumption: median relative replicate spread "
              f"{sd_rel:.2%} over {len(spreads)} token counts "
              f"(floor {MIN_RELATIVE_SPREAD:.1%})")
        print(f"  MDE {two_sample_mde(sd_rel):.1%} relative, on a TIME (two "
              "independent cells, 90% two-sided,")
        print("  80% power). A crossing is an interpolation between two "
              "slopes, so it multiplies")
        print("  this by 1/(s1-s0) -- 4.3x on the measured A100 qwen2 cell -- "
              "and the MDE on a")
        print("  crossing RATIO is the per-cell band below, never this line.")
    else:
        print("  MDE: NOT STATED -- no cell carries replicates, so this report "
              "has no noise")
        print("  model and no ratio in it is comparable with anything.")
    print()

    summary: list[tuple[str, dict]] = []
    for key_, by_t in sorted(cells.items()):
        model, dtype, impl = key_
        # The replicate lists survive to here rather than being medianed away at
        # the top of the loop: the band needs each token count's own scatter,
        # and a median has thrown that away.
        replicates = sorted(by_t.items())
        points = [(t, statistics.median(v)) for t, v in replicates]
        print(f"=== {model} / {dtype} / {impl} ===")
        slopes = dict(local_slopes(points))
        cell_tiles = tiles.get(key_, {})
        print(f"  {'T':>6} {'ms_p50':>9} {'slope':>7} {'M-tiles':>12}   regime")
        prev = None
        prev_tiles = None
        for t, ms in points:
            s = next((v for k, v in slopes.items() if prev and prev < k < t), None)
            tag = "" if s is None else ("weight-bound" if s < 0.5 else "compute-bound")
            counted = cell_tiles.get(t)
            # Padded rather than empty on the first row: an unpadded blank used
            # to be invisible because the regime column was blank too, and now
            # it would shift the M-tile column left and hide the step.
            print(f"  {t:>6} {ms:>9.4f} {'' if s is None else f'{s:.3f}':>7} "
                  f"{tile_cell(counted, prev_tiles):>12}   {tag}")
            prev = t
            if counted:
                prev_tiles = statistics.median(counted)

        # Below E/k a batch misses experts, so weight traffic grows with the
        # batch and the slope crosses for a reason unrelated to the ridge.
        sat = saturation_batch(model)
        # ONE shake per cell, built once and shared by every band printed for
        # it, so the crossing band, the per-step ratios and last-over-first all
        # come off the same draws and cannot disagree with each other.
        bands = CellBands(replicates, sat) if args.uncertainty else None
        # Every upcrossing, not the first. The first is what the line below
        # still prints, because published figures were read off it, but a cell
        # that crosses twice gets said so in as many words underneath.
        found = upcrossings(points, min_tokens=sat)
        measured = found[0].tokens if found else None
        dtype_ridge = ridge_for_dtype(args.ridge, dtype)
        predicted = crossing_batch(model, dtype_ridge, dtype)
        print(f"\n  saturation (E/k, floor):             {sat:8.0f} tokens")
        print(f"  ridge for {dtype:<9}                  {dtype_ridge:8.1f} FLOP/byte")
        print(f"  predicted (2R/b at that ridge):      {predicted:8.0f} tokens")
        if measured is None:
            print("  measured:                            not bracketed by this "
                  "token grid")
            print("  -> add token counts on both sides of the prediction")
        else:
            ratio = measured / predicted
            # `[1 of n]` on the line itself, because this line is the one that
            # gets copied out of a report and into a table, and a reader who
            # copies it has to carry the ambiguity with it.
            of_n = f"   [1 of {len(found)}]" if len(found) > 1 else ""
            print(f"  measured (slope crosses 0.5):        {measured:8.0f} tokens"
                  f"   {ratio:.2f}x predicted{of_n}")
            # Beside the number, not only in the header. A banner forty lines
            # up does not stop this figure being quoted on its own, and every
            # crossing this study has retracted was quoted on its own.
            for line in routing_domain(routing_rows[key_]).crossing_note():
                print(f"  {line}")
            if args.uncertainty:
                band = crossing_interval(replicates, min_tokens=sat)
                if band is None:
                    # The medians bracket the crossing and no perturbed draw
                    # does, so the number above is an artefact of the spread it
                    # was quoted without.
                    print("  90% band (replicate noise):          no draw kept "
                          "the bracket; the crossing does not survive its own "
                          "noise")
                else:
                    _, lo, hi = band
                    print(f"  90% band (replicate noise):          {lo:8.0f} - "
                          f"{hi:.0f} tokens"
                          f"   {lo / predicted:.2f}-{hi / predicted:.2f}x "
                          f"predicted")
            print_staircase(found, predicted, tiles.get(key_, {}), bands)
        print()
        print_roof_fractions(roof_fixed.get(key_, []), roof_cell.get(key_, []),
                             roof_absent.get(key_, collections.Counter()),
                             level_high[key_], cell_rows[key_])
        # Outside the else: a cell whose grid does not bracket a slope crossing
        # is exactly the cell where a second estimator is worth having, and
        # printing it only where the first one succeeded would hide that.
        if args.max_affine:
            print()
            fit = fit_rows(affine.get(key_, {}), args.max_affine_n,
                           args.block_m, args.max_affine_weighting)
            comparison = Comparison(tuple(u.tokens for u in found), fit)
            row = print_max_affine(fit, comparison, model, predicted, bands)
            if row is not None:
                summary.append((f"{model} / {dtype} / {impl}", row))
        print()
    if args.max_affine and summary:
        print_head_to_head(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
