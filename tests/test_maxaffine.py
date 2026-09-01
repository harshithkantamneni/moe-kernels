"""The estimator that cannot have our failure mode, and what it does instead.

`crossing.py` walks adjacent slopes and returns the first that passes 0.5, and
on half the canonical uniform cells that first passage is a tile step rather
than the roofline. TEMPO (arXiv:2608.13057) fits `t = max(a + b G, c + beta N)`
over the whole curve instead, which intersects once and therefore cannot report
a staircase at all.

These tests defend two separate things and it is worth saying which is which.
The synthetic ones pin the ESTIMATOR: that it recovers a max-affine curve it is
given, that it refuses rather than guesses when a regressor is unidentifiable or
a caller names a tile height no row records, and that its own honesty predicates
mean what they say. The published-data ones pin the RESULT, which is negative:
max-affine does return one answer where the slope detector returns two, and on
exactly those cells it fits to a mean relative error of 15 to 51% and lands
three times below the measured ridge. An estimator that removes an ambiguity by
being unable to see it has not settled anything, and these tests exist so that
conclusion cannot quietly soften.

The pool loader, the canonical arm list and the slope-detector crossings are
imported from `test_multiple_crossings` rather than rebuilt. A second copy of
"which rows are canonical" that drifted would let the two files disagree about
what they are comparing while both stayed green.
"""
from __future__ import annotations

import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
from test_multiple_crossings import (
    CANONICAL_IMPLS,
    FIVE_STAGE,
    ONE_STAGE,
    crossings,
    needs_published,
    pool,
)

from moe.bench.maxaffine import (
    ABSOLUTE,
    PADDED_ROWS,
    RELATIVE,
    TOKENS,
    Branch,
    Comparison,
    MaxAffineFit,
    Observation,
    build_compute_side,
    fit,
    fit_rows,
    observations,
)
from moe.bench.ridge import rows_per_expert, saturation_batch
from moe.routing.imbalance import TileEfficiencyUndetermined

REPO = Path(__file__).resolve().parent.parent

#: The H200 ridge band `docs/FINDINGS.md` scores every bf16 crossing against.
#: `2R/b` puts the crossing at `R = ridge` rows per expert, so this is also the
#: rows-per-expert window an estimator's answer has to land in to be the ridge.
RIDGE_BAND = (160.3, 176.2)


def grid(a: float, b: float, c: float, beta: float,
         cells: list[tuple[float, float, float]]) -> list[Observation]:
    """Observations that ARE `max(a + b G, c + beta N)`, exactly.

    A curve with a known right answer, so a failure here is the fitter and not
    the data. `cells` carries `(tokens, G, N)` because the two regressors are
    independent inputs: `G` saturating while `N` keeps rising is the whole shape
    of this problem and a helper that derived one from the other would fit a
    different model than the one under test.
    """
    return [Observation(tokens=t, g=g, n=n, ms=max(a + b * g, c + beta * n))
            for t, g, n in cells]


#: A saturating memory side and a rising compute side, which is what every cell
#: in the published pool looks like: `G` climbs to `E` and stops, `N` keeps going.
SATURATING = [(2.0 ** i, min(64.0, 2.0 ** (i + 1)), 2.0 ** i) for i in range(14)]

TOKEN_SIDE = build_compute_side(TOKENS, None)


def test_a_curve_that_is_exactly_max_affine_gives_both_branches_back():
    """The fitter has to solve the problem it claims to solve before any
    conclusion drawn from it means anything. Exact data, exact parameters."""
    got = fit(grid(0.05, 0.01, 0.02, 0.003, SATURATING), TOKEN_SIDE)
    assert got is not None
    assert got.memory.intercept == pytest.approx(0.05, rel=1e-6)
    assert got.memory.slope == pytest.approx(0.01, rel=1e-6)
    assert got.compute.intercept == pytest.approx(0.02, rel=1e-6)
    assert got.compute.slope == pytest.approx(0.003, rel=1e-6)
    assert got.mean_rel_err == pytest.approx(0.0, abs=1e-9)


def test_the_inflection_of_an_exact_curve_is_where_the_two_branches_meet():
    """`0.05 + 0.01 G` with G pinned at 64 is 0.69 ms, which `0.02 + 0.003 N`
    reaches at N = 223.33. The grid brackets it at 128 and 256, and the answer
    has to be the model's and not the grid's: interpolating in log T the way the
    slope detector must would report 214.5."""
    got = fit(grid(0.05, 0.01, 0.02, 0.003, SATURATING), TOKEN_SIDE)
    assert got.inflection == pytest.approx(0.67 / 0.003, rel=1e-9)
    assert got.single_crossing


def test_a_memory_regressor_that_never_moves_is_unidentifiable_not_zero():
    """Above `E/k` every expert is already active, so `G` is pinned at `E` and
    `b` is not constrained by anything. Reporting `b = 0` without saying it was
    never constrained would put a number in a table that no data produced."""
    flat = [(2.0 ** i, 64.0, 2.0 ** i) for i in range(6, 16)]
    got = fit(grid(0.05, 0.01, 0.02, 0.003, flat), TOKEN_SIDE)
    assert got.memory.degenerate
    assert got.memory.slope == 0.0
    assert not got.compute.degenerate


def test_neither_branch_is_ever_fitted_through_a_single_point():
    """Two points fit a line exactly, so a one-point branch would win the
    residual competition by describing nothing. The split search has to keep
    both sides at two or more whatever the curve looks like."""
    for cells in (SATURATING, [(2.0 ** i, min(64.0, 2.0 ** (i + 1)), 2.0 ** i)
                               for i in range(4)]):
        got = fit(grid(0.05, 0.01, 0.02, 0.003, cells), TOKEN_SIDE)
        assert got.n_memory >= 2 and got.n_compute >= 2


def test_four_token_counts_are_the_fewest_that_can_carry_two_branches():
    """Three points cannot give two branches two points each, and a fit of three
    points to four parameters is not a measurement of any of them."""
    assert fit(grid(0.05, 0.01, 0.02, 0.003, SATURATING[:3]), TOKEN_SIDE) is None
    assert fit(grid(0.05, 0.01, 0.02, 0.003, SATURATING[:4]),
               TOKEN_SIDE) is not None


def test_duplicate_token_counts_raise_rather_than_being_weighted_twice():
    """The same guard `crossing._clean` applies. A token count appearing twice is
    a caller that forgot to median its replicates, and silently letting one T
    contribute two residuals weights it double in a fit whose whole output is a
    weighted balance."""
    doubled = grid(0.05, 0.01, 0.02, 0.003, SATURATING) * 2
    with pytest.raises(ValueError, match="duplicate token counts"):
        fit(doubled, TOKEN_SIDE)


def test_padded_rows_without_a_block_m_is_refused_rather_than_defaulted():
    """No published row records the tile it ran, so a padded-rows regressor
    needs a BLOCK_M the caller names and owns. A default here would put a
    derived number into a fit that reads as measured, which is the single
    failure this study has spent a day correcting."""
    with pytest.raises(ValueError, match="BLOCK_M"):
        build_compute_side(PADDED_ROWS, None)
    assert build_compute_side(PADDED_ROWS, 128).derived
    assert not build_compute_side(TOKENS, 128).derived


def test_an_unknown_regressor_or_weighting_is_refused_at_the_edge():
    """`compute_side` takes the tokens branch for anything that is not
    PADDED_ROWS, so a typo'd kind would silently fit the wrong model and return
    a plausible number rather than an error."""
    with pytest.raises(ValueError, match="compute side"):
        build_compute_side("padded", 128)
    with pytest.raises(ValueError, match="weighting"):
        fit(grid(0.05, 0.01, 0.02, 0.003, SATURATING), TOKEN_SIDE, "huber")


def test_relative_weighting_keeps_the_batches_absolute_weighting_abandons():
    """One canonical cell spans 0.13 ms to 27.9 ms, so an unweighted residual at
    the top is worth more than the entire time at the bottom, and the memory
    branch -- the branch every decode claim in this study lives on -- is fitted
    to nothing.

    Shown on a curve that is NOT exactly max-affine, because on an exact one
    both objectives reach zero and the choice cannot show. Here the compute side
    grows as `T^1.2`, so a straight line has to compromise across the range and
    the weighting decides where. Absolute misses the smallest batch by 268% and
    puts the inflection at 1190 tokens; relative misses it by nothing and puts
    the inflection at 201."""
    curved = [Observation(2.0 ** i, min(64.0, 2.0 ** (i + 1)), 2.0 ** i,
                          max(0.05 + 0.01 * min(64.0, 2.0 ** (i + 1)),
                              0.5 * (2.0 ** i / 128.0) ** 1.2))
              for i in range(15)]
    relative = fit(curved, TOKEN_SIDE, RELATIVE)
    absolute = fit(curved, TOKEN_SIDE, ABSOLUTE)
    smallest = curved[0]
    assert abs(relative.predict(smallest) - smallest.ms) / smallest.ms < 0.01
    assert abs(absolute.predict(smallest) - smallest.ms) / smallest.ms > 1.0
    assert relative.mean_rel_err < absolute.mean_rel_err / 10
    assert relative.inflection < absolute.inflection / 5


def test_a_model_that_crosses_back_at_small_batches_is_not_single_crossing():
    """THE CLAIM THIS MODULE EXISTS TO CHECK. Two planes intersect once, but a
    sweep walks a path through them on which the max can flip twice: `G` rises
    steeply at tiny batches while `N` has barely moved, so a compute branch with
    a large intercept sits on top at T=1, loses by T=4 and wins again at the
    inflection. Built here as a TRUE curve, so this is a property of the model
    and not an artefact of fitting it. On the published pool 14 of 16 cells do
    exactly this."""
    got = fit(grid(0.05, 0.01, 0.50, 0.001, SATURATING), TOKEN_SIDE)
    assert got.reversals == 1
    assert got.inflection is not None      # one UPWARD crossing, still
    assert not got.single_crossing         # but not one crossing


def test_two_upward_inflections_report_none_rather_than_the_first():
    """Returning the first of several would rebuild the first-passage reduction
    this module exists to avoid, in the estimator advertised as immune to it."""
    two = MaxAffineFit(
        memory=Branch(0.0, 1.0, False), compute=Branch(0.0, 1.0, False),
        compute_side=TOKEN_SIDE, weighting=RELATIVE,
        observations=tuple(grid(0.05, 0.01, 0.02, 0.003, SATURATING)),
        split_tokens=256.0, n_memory=8, n_compute=6,
        inflections=(100.0, 900.0), reversals=1)
    assert two.inflection is None
    assert not two.single_crossing


def test_a_comparison_never_divides_by_a_crossing_that_was_not_found():
    """A cell whose grid brackets no slope crossing is exactly the cell a second
    estimator is worth having on, so the comparison has to survive one side
    being empty rather than raising or reporting a ratio to zero."""
    got = fit(grid(0.05, 0.01, 0.02, 0.003, SATURATING), TOKEN_SIDE)
    empty = Comparison(slope_crossings=(), fit=got)
    assert empty.first is None and empty.last is None
    assert empty.ratio_to(empty.first) is None
    assert not empty.slope_ambiguous and empty.affine_single
    assert not empty.resolves_ambiguity
    unfitted = Comparison(slope_crossings=(300.0, 800.0), fit=None)
    assert unfitted.slope_ambiguous and not unfitted.affine_single
    assert unfitted.ratio_to(300.0) is None


# ------------------------------------------------- against the published rows


def canonical_fits(kind: str = TOKENS, block_m: int | None = None) -> dict:
    """`(model, impl) -> MaxAffineFit` over the canonical uniform bf16 cells."""
    return {key: fit_rows(by_t, kind, block_m)
            for key, by_t in pool().items() if key[1] in CANONICAL_IMPLS}


@needs_published
def test_max_affine_answers_once_on_every_cell_the_slope_detector_splits():
    """Its advertised property, confirmed on the rows: all 8 of the canonical
    cells that give the slope detector two crossings give max-affine exactly one
    upward inflection. This is the result in max-affine's favour, and it is the
    only one."""
    fits = canonical_fits()
    ambiguous = [key for key in fits if len(crossings(*key)) > 1]
    assert len(ambiguous) == 8
    assert all(fits[key].inflection is not None for key in ambiguous)


@needs_published
def test_the_fitted_model_still_crosses_the_measured_grid_twice_on_most_cells():
    """"One crossing by construction" is a statement about two planes, not about
    the path a sweep walks through them. 14 of 16 canonical cells have a
    reversal at small T, so the estimator's headline property does not survive
    contact with a grid that starts at one token."""
    fits = canonical_fits()
    assert sum(1 for f in fits.values() if f.reversals) == 14
    assert sum(1 for f in fits.values() if f.single_crossing) == 2


@needs_published
def test_max_affine_does_not_describe_the_five_stage_staircase():
    """THE NEGATIVE RESULT. The one-stage CUTLASS spans hold BLOCK_M at 64 and
    their curves are smooth, so a two-piece model fits them: p95 relative error
    tops out at 47%. The five-stage Triton spans step, and there the same model
    is never better than 61% at p95 and reaches 263%. The two groups do not
    overlap on p95, which is what makes this a property of the curve shape
    rather than of any one cell."""
    fits = canonical_fits()
    five = [f.p95_rel_err for k, f in fits.items() if k[1] in FIVE_STAGE]
    one = [f.p95_rel_err for k, f in fits.items() if k[1] in ONE_STAGE]
    assert len(five) == len(one) == 8
    assert min(five) > max(one)
    assert statistics.median(five) > 1.0        # off by more than the time itself
    assert statistics.median(one) < 0.5


@needs_published
def test_the_five_stage_inflection_lands_far_below_the_ridge_it_should_find():
    """`2R/b` puts the crossing at `R = ridge` rows per expert, and the measured
    band is 160.3 to 176.2. Max-affine reaches 135 to 157 on the one-stage cells,
    just under the band, and 18 to 63 on the five-stage cells, three to nine
    times below it. So on the cells where it removes the slope detector's
    ambiguity it is not reporting the ridge, and the ambiguity was not the
    problem."""
    fits = canonical_fits()
    five = sorted(rows_per_expert(k[0], f.inflection)
                  for k, f in fits.items() if k[1] in FIVE_STAGE)
    one = sorted(rows_per_expert(k[0], f.inflection)
                 for k, f in fits.items() if k[1] in ONE_STAGE)
    assert max(five) < min(one)
    assert max(five) < RIDGE_BAND[0] / 2
    assert RIDGE_BAND[0] * 0.75 < statistics.fmean(one) < RIDGE_BAND[0]


@needs_published
def test_padded_rows_buys_its_better_fit_by_making_the_branches_collinear():
    """Rounding `N` up to whole M-tiles fits far better and means far less.
    Below saturation every expert holds fewer rows than one tile, so padded rows
    is EXACTLY `BLOCK_M x active experts` and the compute-side regressor is a
    constant multiple of the memory-side one. The residual improves because the
    inflection stops being identified: all eight five-stage inflections collapse
    below their own model's saturation batch, where no layer is compute bound."""
    padded = canonical_fits(PADDED_ROWS, 128)
    tokens = canonical_fits()
    for key, f in padded.items():
        assert f.mean_rel_err < tokens[key].mean_rel_err
    for key, f in padded.items():
        if key[1] in FIVE_STAGE:
            assert f.inflection < saturation_batch(key[0])
    side = build_compute_side(PADDED_ROWS, 128)
    obs = observations(pool()[("mixtral-8x7b", "vllm_fused_experts")], side)
    below = [o for o in obs if o.tokens <= saturation_batch("mixtral-8x7b")]
    assert len(below) >= 4
    assert all(o.n == pytest.approx(128 * o.g) for o in below)


@needs_published
def test_a_padded_inflection_on_a_fused_cell_is_set_by_the_assumed_tile():
    """The critical distinction, measured. Every published arm is schema v3 and
    records no tile configuration, so a padded-rows regressor is DERIVED from
    vLLM's source plus the row's `gpu_name`. Refitting the same cells at
    BLOCK_M 64 instead of 128 moves the five-stage inflections by a median
    factor of 48 -- mixtral goes from 2 tokens to 288 -- while the one-stage
    cells, whose curves the model actually describes, move by at most 11%. Where
    the answer swings that far with an assumption, the answer is the assumption.
    """
    at64, at128 = canonical_fits(PADDED_ROWS, 64), canonical_fits(PADDED_ROWS, 128)

    def swing(key) -> float:
        pair = [at64[key].inflection, at128[key].inflection]
        if None in pair:      # one tile finds no inflection at all: total
            return math.inf
        return max(pair) / min(pair)

    five = [swing(k) for k in at64 if k[1] in FIVE_STAGE]
    one = [swing(k) for k in at64 if k[1] in ONE_STAGE]
    assert statistics.median(five) > 4.0
    assert max(one) < 1.15


@needs_published
def test_a_padded_regressor_the_rows_cannot_determine_returns_no_fit():
    """At BLOCK_M 16 an expert spans several tiles well before the top of the
    grid, and the per-expert distribution that would settle the count is neither
    stored nor reproducible off the GPU. No fit is the answer; a fit over the
    token counts that happened to work is a fit of a different curve."""
    by_t = pool()[("mixtral-8x7b", "vllm_fused_experts")]
    with pytest.raises(TileEfficiencyUndetermined):
        observations(by_t, build_compute_side(PADDED_ROWS, 16))
    assert fit_rows(by_t, PADDED_ROWS, 16) is None


# ---------------------------------------------------------------- the report


def write_sweep(path: Path, points: list[tuple[int, float, int]]) -> Path:
    """A minimal sweep CSV whose ACTIVE EXPERT COUNT grows and then saturates.

    Not `test_multiple_crossings.write_sweep`, which pins `load_active_experts`
    at 8 for every row. A memory branch fitted on a constant `G` is degenerate
    by construction, so that fixture would test the one shape this report block
    must not assume.
    """
    from moe.bench.schema import COLUMNS
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for t, ms, active in points:
            for k in (-0.005, 0.0, 0.005):
                row = {c: "" for c in COLUMNS}
                row.update({"impl": "vllm_fused_experts", "model": "mixtral-8x7b",
                            "dtype": "bf16", "routing_kind": "uniform",
                            "num_tokens": t, "ms_p50": ms * (1 + k),
                            "correctness_passed": "True", "throttled": "False",
                            "load_total_rows": t * 2, "load_active_experts": active,
                            "load_max_rows": max(1, t // 4),
                            "load_tile_eff_bm64": t * 2 / (active * 64),
                            "load_tile_eff_bm128": t * 2 / (active * 128)})
                w.writerow(row)
    return path


@pytest.fixture
def sweep_csv(tmp_path) -> Path:
    """A flat memory regime that turns over, with `G` saturating at 8 by T=16."""
    points = [(1, 0.20, 2), (2, 0.29, 3), (4, 0.50, 6), (8, 0.52, 6),
              (16, 0.67, 8), (32, 0.70, 8), (64, 0.68, 8), (128, 0.72, 8),
              (256, 0.81, 8), (512, 1.22, 8), (1024, 1.91, 8), (2048, 3.30, 8)]
    return write_sweep(tmp_path / "sweep.csv", points)


def report(path: Path, *extra: str) -> str:
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "crossing_report.py"), str(path),
         "--ridge", "160.3", *extra],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_the_flag_is_off_by_default_and_changes_nothing_when_it_is(sweep_csv):
    """The report's default output is quoted in docs/FINDINGS.md line for line,
    so a new estimator that moved it by a byte would silently invalidate every
    figure regenerated from that command."""
    assert "MAX-AFFINE" not in report(sweep_csv)
    assert "max-affine" not in report(sweep_csv)


def test_every_default_line_survives_when_the_flag_is_on(sweep_csv):
    """`--max-affine` ADDS a block, it does not rewrite the table above it. Each
    default line still appears, in order, so the two runs can be read against
    each other."""
    default = report(sweep_csv).splitlines()
    with_affine = report(sweep_csv, "--max-affine").splitlines()
    remaining = iter(with_affine)
    for line in default:
        assert any(line == got for got in remaining), f"lost line: {line!r}"


def test_the_report_prints_the_fit_quality_beside_the_inflection(sweep_csv):
    """A max-affine inflection is worth more than the slope detector's several
    only if the model describes the curve, so the residual is printed on the
    line below the answer and not in a footnote. Both come out of the same
    block; a reader who copies one copies the other."""
    out = report(sweep_csv, "--max-affine")
    assert "MAX-AFFINE (TEMPO-style global fit)" in out
    assert "inflection:" in out
    assert "fit quality:     mean |rel err|" in out
    assert "ESTIMATOR HEAD TO HEAD" in out


def test_the_report_says_which_regressor_was_derived_and_which_was_observed(
        sweep_csv):
    """Padded rows needs a BLOCK_M no published row records. The output has to
    carry that where the number is, not only in the module that computed it."""
    tokens = report(sweep_csv, "--max-affine")
    assert "N = tokens (observed column)" in tokens
    padded = report(sweep_csv, "--max-affine", "--max-affine-n", "padded_rows",
                    "--block-m", "128")
    assert "DERIVED from vLLM's source, not recorded by these rows" in padded
