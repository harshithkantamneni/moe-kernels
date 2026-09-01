"""The curve crosses 0.5 more than once, and the first crossing is a tile step.

`crossing_from_points` returns the FIRST adjacent slope pair that brackets 0.5.
That is a first-passage detector, and it is only the right reduction if the
measured curve makes ONE transition from flat to linear. It does not. 8 of the
16 canonical uniform cells cross 0.5 going up twice, because time steps up when
an expert gains an M-tile and flatlines while the tile count holds, so the slope
spikes above 0.5 at every step and sags below it on every tread.

The bill is the study's headline: taking the last crossing instead of the first
moves the five-stage over one-stage separation from 0.560 to 0.889, and mixtral
and qwen2 from 0.56 and 0.46 to 1.01 and 1.00.

These tests do not decide which crossing is the ridge. They pin that there is
more than one, that the tile count steps where the slope spikes, and that the
report says so instead of printing a single number off a staircase.
"""
from __future__ import annotations

import csv
import functools
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

# `from_slopes` builds a curve with exactly the slopes a test asks for, and the
# staircase tests are about slopes. Imported rather than copied: a second copy
# that drifted would let the two files disagree about what a slope of 0.9 means
# while both kept passing.
from test_crossing_uncertainty import from_slopes

from moe.bench.crossing import (
    STORED_TILE_EFF,
    all_crossings_from_points,
    crossing_from_points,
    local_slopes,
    m_tiles_for_row,
    recorded_block_m,
    timed_rows,
    upcrossings,
)
from moe.bench.ridge import (
    crossing_batch,
    ridge_for_dtype,
    rows_per_expert,
    saturation_batch,
)
from moe.bench.schema import TileConfigUnrecorded
from moe.routing.imbalance import TileEfficiencyUndetermined

REPO = Path(__file__).resolve().parent.parent
PUBLISHED = REPO / "results/published"

#: The four-arm bf16 H200 pool every bf16 crossing in docs/FINDINGS.md comes
#: from. A pool rather than one run because no single arm carries all four model
#: geometries, and pooling only these four because the rest are superseded, on
#: another card, or at another dtype.
CANONICAL_POOL = (
    "2026-08-22-standard-sweep",
    "2026-08-26-nvidia_h200-full-three-way-recalibrated",
    "2026-08-28-nvidia_h200-ridge-resolution",
    "2026-08-28-nvidia_h200-h200-v2lite",
)

#: vLLM and SGLang time five stages of the layer; the two torch spans time one.
FIVE_STAGE = ("sglang_fused_experts", "vllm_fused_experts")
ONE_STAGE = ("torch_grouped_mm_down", "torch_grouped_mm_up")

#: 4 models x 4 implementations. `__pipeline__` is excluded because it is not a
#: kernel measurement, and its crossing is not one of the sixteen.
CANONICAL_IMPLS = FIVE_STAGE + ONE_STAGE

#: The H200 ridge that every ratio in docs/FINDINGS.md is scored against.
RIDGE = 160.3

needs_published = pytest.mark.skipif(
    not all((PUBLISHED / arm).exists() for arm in CANONICAL_POOL),
    reason="the canonical bf16 H200 pool is not in this checkout")


@functools.cache
def pool() -> dict[tuple[str, str], dict[int, tuple[dict, ...]]]:
    """`(model, impl) -> token count -> the rows`, filtered as the report filters.

    Uniform routing only, because `2R/b` is a uniform-routing statement and
    pooling regimes gives a crossing of a blend rather than of a layer. Rows
    kept whole rather than reduced to times: the M-tile count comes off the same
    row as the time it has to be lined up against.

    Cached because it reads 30,660 rows and every test here wants the same ones.
    """
    cells: dict[tuple[str, str], dict[int, list[dict]]] = {}
    for arm in CANONICAL_POOL:
        for path in sorted((PUBLISHED / arm).glob("run_*.csv")):
            with path.open(newline="") as fh:
                rows = timed_rows(list(csv.DictReader(fh)))
            for r in rows:
                if r["routing_kind"] != "uniform" or r["dtype"] != "bf16":
                    continue
                if r.get("correctness_passed") not in ("True", "true", "1", ""):
                    continue
                if r.get("throttled") in ("True", "true", "1"):
                    continue
                key = (r["model"], r["impl"])
                cells.setdefault(key, {}).setdefault(int(r["num_tokens"]), []).append(r)
    return {k: {t: tuple(v) for t, v in by_t.items()} for k, by_t in cells.items()}


def curve(model: str, impl: str) -> list[tuple[float, float]]:
    """One median time per token count, the way `crossing_report` aggregates."""
    return sorted((t, statistics.median(float(r["ms_p50"]) for r in rows))
                  for t, rows in pool()[(model, impl)].items())


def crossings(model: str, impl: str) -> list[float]:
    return all_crossings_from_points(curve(model, impl),
                                     min_tokens=saturation_batch(model))


def tiles(model: str, impl: str, block_m: int = 128) -> dict[int, float]:
    """Median M-tiles per token count. Median for the same reason the time is:
    uniform routing is redrawn per replicate, so the counts genuinely differ."""
    return {t: statistics.median(m_tiles_for_row(r, block_m) for r in rows)
            for t, rows in sorted(pool()[(model, impl)].items())}


def predicted(model: str) -> float:
    return crossing_batch(model, ridge_for_dtype(RIDGE, "bf16"), "bf16")


def separation(pick: int) -> tuple[float, float, float]:
    """`(five_stage_mean, one_stage_mean, five/one)` taking crossing `pick`.

    The FINDINGS quantity: each cell's crossing over what `2R/b` predicts for
    it, averaged within each span extent. The separation is the RATIO of the two
    means, which is what makes it survive the ridge band -- both sides divide by
    the same prediction, so the ridge cancels algebraically.
    """
    five, one = [], []
    for (model, impl), _ in sorted(pool().items()):
        if impl not in CANONICAL_IMPLS:
            continue
        found = crossings(model, impl)
        if not found:
            continue
        ratio = found[pick] / predicted(model)
        (five if impl in FIVE_STAGE else one).append(ratio)
    f, o = statistics.fmean(five), statistics.fmean(one)
    return f, o, f / o


# ------------------------------------------------- the cells that cross twice


@needs_published
def test_eight_of_the_sixteen_canonical_uniform_cells_cross_zero_point_five_twice():
    """The fact the first-passage detector hides. Half the study's cells supply
    two crossings and the report has been quoting whichever one the token grid
    happened to sample first."""
    counts = {(model, impl): len(crossings(model, impl))
              for (model, impl) in sorted(pool())
              if impl in CANONICAL_IMPLS}
    assert len(counts) == 16, sorted(counts)
    multi = sorted(k for k, n in counts.items() if n > 1)
    assert len(multi) == 8, multi
    assert all(n in (1, 2) for n in counts.values()), counts


@needs_published
@pytest.mark.parametrize("model,impl,first,last", [
    ("mixtral-8x7b", "vllm_fused_experts", 313, 800),
    ("mixtral-8x7b", "sglang_fused_experts", 313, 778),
    ("mixtral-8x7b", "torch_grouped_mm_up", 332, 789),
    ("qwen2-57b-a14b", "vllm_fused_experts", 730, 1573),
    ("qwen2-57b-a14b", "sglang_fused_experts", 730, 1574),
    ("deepseek-v3", "vllm_fused_experts", 2925, 6391),
    ("deepseek-v3", "sglang_fused_experts", 3104, 4941),
    ("deepseek-v3", "torch_grouped_mm_up", 2751, 6243),
])
def test_each_staircase_cell_reports_both_of_its_crossings(model, impl, first, last):
    """The eight, cell by cell, off the published rows. The gap between the two
    is 2.0x to 2.6x, which is far outside any band the replicate spread puts on
    either of them: this is not one crossing measured noisily."""
    found = crossings(model, impl)
    assert len(found) == 2, found
    assert round(found[0]) == first, found
    assert round(found[1]) == last, found
    assert last / first > 1.5


@needs_published
def test_the_other_eight_cells_cross_once_so_this_is_not_an_artefact_of_the_detector():
    """A detector that found extra crossings everywhere would be finding noise.
    Eight cells still cross exactly once, and deepseek-v2-lite -- whose token
    grid has no dense band at all -- crosses once on all four of its."""
    once = sorted(k for k in pool()
                  if k[1] in CANONICAL_IMPLS and len(crossings(*k)) == 1)
    assert len(once) == 8, once
    assert [impl for model, impl in once if model == "deepseek-v2-lite"] \
        == sorted(CANONICAL_IMPLS)


# --------------------------------------------------------- what it costs


@needs_published
def test_taking_the_last_crossing_moves_the_span_extent_separation_from_0_56_to_0_89():
    """The headline. The five-stage over one-stage separation is C4's number and
    the one the study calls robust, since it survives the ridge band and the
    routing restriction that killed C5. It does not survive the choice of which
    crossing to read: 0.560 on the first, 0.889 on the last, a 59% move."""
    five_first, one_first, sep_first = separation(0)
    five_last, one_last, sep_last = separation(-1)
    assert round(five_first, 3) == 0.553 and round(one_first, 3) == 0.987
    assert round(sep_first, 3) == 0.560, sep_first
    assert round(five_last, 3) == 1.032 and round(one_last, 3) == 1.161
    assert round(sep_last, 3) == 0.889, sep_last
    assert (sep_last - sep_first) / sep_first > 0.55


@needs_published
@pytest.mark.parametrize("model,first,last", [
    ("mixtral-8x7b", 0.56, 1.01),
    ("qwen2-57b-a14b", 0.46, 1.00),
])
def test_two_models_go_from_a_half_to_agreement_on_the_last_crossing(model, first, last):
    """Per model, the same quantity. On the first crossing mixtral and qwen2 say
    the five-stage span crosses at half the one-stage span's fraction of
    prediction; on the last they say the two agree. Those are different
    findings, from one input, separated only by which step was read."""
    def ratio(pick: int) -> float:
        by_extent = {}
        for impl in CANONICAL_IMPLS:
            found = crossings(model, impl)
            if found:
                group = "five" if impl in FIVE_STAGE else "one"
                by_extent.setdefault(group, []).append(found[pick] / predicted(model))
        return statistics.fmean(by_extent["five"]) / statistics.fmean(by_extent["one"])

    assert round(ratio(0), 2) == first
    assert round(ratio(-1), 2) == last


@needs_published
def test_rows_per_expert_at_the_last_crossing_lands_in_the_measured_ridge_band():
    """Why the ambiguity is worth resolving rather than averaging over. `2R/b`
    at bf16 puts the crossing at `rows_per_expert = ridge`, and the H200's ridge
    was measured three times at 160.3 to 176.2. The last crossings sit at a mean
    175.8 with CV 21.2%; the first at 123.4 with CV 40.0%, below the band and
    twice as scattered. Suggestive, not decisive -- it is one prediction scoring
    itself -- which is why nothing here picks a winner."""
    def spread(pick: int) -> tuple[float, float]:
        vals = [rows_per_expert(model, crossings(model, impl)[pick])
                for (model, impl) in sorted(pool())
                if impl in CANONICAL_IMPLS and crossings(model, impl)]
        mean = statistics.fmean(vals)
        return mean, statistics.stdev(vals) / mean

    first_mean, first_cv = spread(0)
    last_mean, last_cv = spread(-1)
    assert round(first_mean, 1) == 123.4 and round(first_cv, 3) == 0.400
    assert round(last_mean, 1) == 175.8 and round(last_cv, 3) == 0.212
    assert 160.3 <= last_mean <= 176.2
    assert first_mean < 160.3


# ------------------------------------------------------------ the mechanism


@needs_published
def test_the_mixtral_tile_count_is_12_16_16_16_16_19_on_published_rows():
    """The staircase, in the sequence the study quotes. mixtral holds 128, 144,
    160, 176, 192 then 256 rows per expert across T=512 to 1024, and at
    BLOCK_M 128 that is 12 tiles, then 16 for four token counts running, then
    19. Rows that read exactly this sequence exist at every one of those token
    counts."""
    counts = {t: {round(m_tiles_for_row(r, 128)) for r in rows}
              for t, rows in pool()[("mixtral-8x7b", "vllm_fused_experts")].items()}
    for t, expected in zip((512, 576, 640, 704, 768, 1024),
                           (12, 16, 16, 16, 16, 19), strict=True):
        assert expected in counts[t], (t, expected, sorted(counts[t]))


@needs_published
def test_the_replicate_median_tile_sequence_is_12_15_16_16_16_21_not_12_16_16_16_16_19():
    """The trap in the sequence above, and the reason the report medians. Uniform
    routing is SAMPLED per replicate, so two rows at one token count can round to
    different tile counts: T=576 draws 14, 15 and 16 across its six rows and
    T=1024 draws 19 on two and 21 on four. 12/16/16/16/16/19 is one realisation,
    not the cell -- quoting it as the cell puts a tile count beside a time taken
    over eight other draws. The flat tread and both steps survive either way,
    which is the part the mechanism rests on."""
    median = tiles("mixtral-8x7b", "vllm_fused_experts")
    got = [round(median[t]) for t in (512, 576, 640, 704, 768, 1024)]
    assert got == [12, 15, 16, 16, 16, 21], got


@needs_published
def test_time_flatlines_while_the_tile_count_holds_and_jumps_when_it_steps():
    """The mechanism itself, on the dense mixtral band. Across T=640 to 768 the
    tile count does not move and neither does the time: three token counts, 20%
    more work, 3% more milliseconds and slopes of 0.464, 0.223, 0.116. At the
    two token counts where a tile is added the slope is 0.731 and 0.971. A
    roofline transition does not go back down."""
    ms = dict(curve("mixtral-8x7b", "vllm_fused_experts"))
    tile = tiles("mixtral-8x7b", "vllm_fused_experts")
    slopes = {(t0, t1): math.log(ms[t1] / ms[t0]) / math.log(t1 / t0)
              for t0, t1 in zip((512, 576, 640, 704, 768), (576, 640, 704, 768, 1024),
                                strict=True)}
    assert round(tile[640]) == round(tile[704]) == round(tile[768]) == 16
    assert ms[768] / ms[640] < 1.05
    assert slopes[(640, 704)] < 0.25 and slopes[(704, 768)] < 0.25
    assert round(tile[576]) > round(tile[512])
    assert round(tile[1024]) > round(tile[768])
    assert slopes[(512, 576)] > 0.7 and slopes[(768, 1024)] > 0.9


@needs_published
def test_the_measured_slope_tracks_the_tile_growth_exponent_on_the_triton_cells():
    """The mechanism claim at full scale rather than on one band. Over the 78
    intervals of the eight five-stage Triton cells, `d(log ms)/d(log T)` and
    `d(log tiles)/d(log T)` correlate at r = 0.88 -- the measured time is
    tracking tile count, not token count.

    The one-stage spans are the control that stops this being arithmetic: they
    are CUTLASS `grouped_mm` with the tile fixed at 64 by the instruction set,
    the same computation over them gives r = 0.04, and tiles and tokens are
    related by construction in both. Something specific to the Triton schedule
    is what the correlation is picking up."""
    def pearson(impls: tuple[str, ...], block_m: int) -> float:
        xs, ys = [], []
        for (model, impl) in sorted(pool()):
            if impl not in impls:
                continue
            ms = [(t, m) for t, m in curve(model, impl) if t >= saturation_batch(model)]
            tile = tiles(model, impl, block_m)
            for (t0, m0), (t1, m1) in zip(ms, ms[1:], strict=False):
                xs.append(math.log(m1 / m0) / math.log(t1 / t0))
                ys.append(math.log(tile[t1] / tile[t0]) / math.log(t1 / t0))
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        cov = statistics.fmean((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
        return cov / (statistics.pstdev(xs) * statistics.pstdev(ys))

    assert pearson(FIVE_STAGE, 128) > 0.85
    assert pearson(ONE_STAGE, 64) < 0.20


@needs_published
def test_fifteen_of_the_sixteen_staircase_crossings_sit_on_a_tile_step():
    """What makes "the crossing is a tile step" a measurement rather than a
    story, INCLUDING the one that is not.

    Eight cells, two crossings each. For 15 of the 16 the interval whose slope
    rose through 0.5 is one where the tile count grows faster than the token
    count, which is what a step is and what smooth growth is not.

    The exception is deepseek-v3 / sglang's second crossing, and it is worth
    more than the rule. Its tile count is SATURATED there -- 511 to 512 across
    T=5120 to 5632, every one of the 256 experts already holding two tiles -- and
    what crosses is a slope that grazed the threshold from below, 0.473 then
    0.634. Same model on vLLM puts its second crossing 1450 tokens later, on the
    6144-to-8192 step. So one of these sixteen numbers is a threshold graze, not
    a step and not a ridge, and nothing here can tell a reader which of the two
    deepseek-v3 second crossings to believe. That is the ambiguity, stated.
    """
    grazing = []
    counted = 0
    for (model, impl) in sorted(pool()):
        found = upcrossings(curve(model, impl), min_tokens=saturation_batch(model))
        if impl not in CANONICAL_IMPLS or len(found) < 2:
            continue
        tile = tiles(model, impl)
        for u in found:
            lo, hi = int(u.step_lo), int(u.step_hi)
            counted += 1
            growth = math.log(tile[hi] / tile[lo]) / math.log(hi / lo)
            if growth <= 0.5:
                grazing.append((model, impl, lo, hi))
    assert counted == 16
    assert grazing == [("deepseek-v3", "sglang_fused_experts", 5120, 5632)], grazing


@needs_published
def test_the_annotated_interval_is_the_rising_slope_s_not_the_one_the_crossing_falls_in():
    """The off-by-one this annotation is written to avoid. qwen2's first
    crossing is 730 tokens, which falls inside the 512-to-1024 interval, but the
    slope that rose through 0.5 is the 1024-to-1152 one and that is where the
    tile step is: 94 tiles to 124. A slope is stamped at the geometric mid of
    its interval, so a crossing straddles the grid POINT two intervals share and
    can land on either side of it."""
    found = upcrossings(curve("qwen2-57b-a14b", "vllm_fused_experts"),
                        min_tokens=saturation_batch("qwen2-57b-a14b"))
    assert round(found[0].tokens) == 730
    assert (found[0].step_lo, found[0].step_hi) == (1024, 1152)
    assert found[0].step_lo > found[0].tokens
    tile = tiles("qwen2-57b-a14b", "vllm_fused_experts")
    assert round(tile[1024]) == 94 and round(tile[1152]) == 124


@needs_published
def test_the_annotated_slopes_are_the_ones_local_slopes_reports():
    """`upcrossings` recomputes the slope arithmetic so it can keep the interval
    endpoints attached, which `local_slopes` throws away. This pins the two
    equal on real rows, so the duplication cannot drift into two answers."""
    for model, impl in (("mixtral-8x7b", "vllm_fused_experts"),
                        ("deepseek-v3", "sglang_fused_experts")):
        points = [(t, ms) for t, ms in curve(model, impl)
                  if t >= saturation_batch(model)]
        reference = [s for _, s in local_slopes(points)]
        for u in upcrossings(points):
            assert u.slope_below in reference and u.slope_above in reference
            assert u.slope_below < 0.5 <= u.slope_above


# --------------------------------------------------- the functions in isolation


def test_a_staircase_returns_every_upcrossing_and_the_first_is_only_the_first():
    """Two steps and a tread between them, which is the shape mixtral makes."""
    points = from_slopes(64, 1.0, [0.1, 0.9, 0.1, 0.1, 0.9, 0.9])
    found = all_crossings_from_points(points)
    assert len(found) == 2, found
    assert found[0] < found[1]
    assert crossing_from_points(points) == found[0]


def test_a_slope_falling_back_through_the_threshold_is_not_a_crossing():
    """Downcrossings are the tops of steps. Counting them would report four
    crossings for mixtral's two, and would make "how many crossings" depend on
    whether the grid ends on a tread."""
    assert len(all_crossings_from_points(from_slopes(64, 1.0, [0.1, 0.9, 0.1, 0.9]))) == 2


def test_a_curve_that_crosses_once_gives_one_crossing_and_a_flat_one_gives_none():
    assert len(all_crossings_from_points(from_slopes(64, 1.0, [0.0, 0.0, 1.0, 1.0]))) == 1
    assert all_crossings_from_points(from_slopes(64, 1.0, [0.1, 0.1, 0.1])) == []
    assert crossing_from_points(from_slopes(64, 1.0, [0.1, 0.1, 0.1])) is None


def test_the_saturation_floor_applies_to_every_crossing_and_not_only_the_first():
    """Below `E/k` the slope crosses because weight traffic grows with the batch,
    which is not the ridge. Dropping the floor here would add a phantom crossing
    at the FRONT and silently renumber every later one, so a caller asking for
    "the last" would still get the right answer while "the first" changed."""
    # Flat then linear below the floor, which is the shape unsaturated routing
    # actually makes, and it brackets a crossing at about 4 tokens.
    early = [(1.0, 1.0), (2.0, 1.0), (4.0, 1.0), (8.0, 2.0), (16.0, 4.0), (32.0, 8.0)]
    late = from_slopes(64, 8.0, [0.1, 0.9, 0.1, 0.9])
    assert len(all_crossings_from_points(early + late, min_tokens=64)) == 2
    unfloored = all_crossings_from_points(early + late)
    assert len(unfloored) == 3 and unfloored[0] < 8


def test_a_single_pair_of_points_cannot_bracket_a_crossing():
    """One interval is one slope, and a crossing needs two to interpolate
    between. Returning something from it would be reading a transition off a
    curve with no shape."""
    assert all_crossings_from_points([(64, 1.0), (128, 4.0)]) == []
    assert all_crossings_from_points([]) == []


# ------------------------------------------------------------ the tile helper


def load_row(total: float, active: int, max_rows: float, **extra) -> dict:
    row = {"impl": "vllm_fused_experts", "load_total_rows": str(total),
           "load_active_experts": str(active), "load_max_rows": str(max_rows)}
    row.update({k: str(v) for k, v in extra.items()})
    return row


def test_the_tile_count_comes_off_the_stored_efficiency_when_there_is_one():
    """mixtral at T=512: 1024 rows over 8 experts at `tile_eff = 2/3` is 12
    tiles, not the 8 the mean rows-per-expert would suggest. Four of its experts
    drew more than 128 rows and each of those needs a second tile -- which is the
    whole reason the stored column is preferred over any reconstruction."""
    row = load_row(1024, 8, 147, load_tile_eff_bm128=2 / 3)
    assert m_tiles_for_row(row, 128) == pytest.approx(12.0)


def test_the_tile_count_is_reconstructed_at_a_block_size_with_no_stored_column():
    """32 is not one of the two stored efficiencies, and while every expert fits
    in one tile the count is just the active expert count."""
    assert 32 not in STORED_TILE_EFF
    row = load_row(200, 8, 30)
    assert m_tiles_for_row(row, 32) == pytest.approx(8.0)


def test_reconstruction_refuses_rather_than_guessing_once_an_expert_spans_tiles():
    """Above `max_rows > block_m` the count depends on the full per-expert
    distribution, which the CSV does not store and a GPU-seeded sampler cannot
    reproduce off-GPU. A plausible number here is indistinguishable from a
    measured one, which is how this study lost three days."""
    with pytest.raises(TileEfficiencyUndetermined):
        m_tiles_for_row(load_row(1024, 8, 147), 128)


def test_a_row_that_does_not_record_its_tile_refuses_to_have_one_assumed():
    """Every published arm is v3 and none of them records the tile that ran, so
    `block_m=None` raises rather than reading the UNRECORDED sentinel as a
    number. The caller has to name the block and own the assumption."""
    row = load_row(1024, 8, 147, load_tile_eff_bm128=2 / 3)
    assert recorded_block_m(row) is None
    with pytest.raises(TileConfigUnrecorded):
        m_tiles_for_row(row)
    with pytest.raises(TileConfigUnrecorded):
        m_tiles_for_row({**row, "tile_block_m": "<unrecorded>"})


def test_a_row_that_does_record_its_tile_is_counted_at_that_tile():
    """The v4 case this is written for. A row carrying `tile_block_m` needs no
    assumption at all, and the count uses the block the kernel actually ran."""
    row = load_row(1024, 8, 147, load_tile_eff_bm128=2 / 3, tile_block_m=128,
                   schema_version=4)
    assert recorded_block_m(row) == 128
    assert m_tiles_for_row(row) == pytest.approx(m_tiles_for_row(row, 128))


def test_a_row_with_no_routing_load_has_no_tile_count():
    """A synthetic or hand-written CSV leaves the load columns blank. That is
    absence, and it has to raise rather than report zero tiles."""
    with pytest.raises(TileEfficiencyUndetermined):
        m_tiles_for_row({"impl": "x", "load_total_rows": "", "load_max_rows": "",
                         "load_active_experts": ""}, 128)
    with pytest.raises(TileEfficiencyUndetermined):
        m_tiles_for_row(load_row(1024, 0, 0), 128)


def test_a_zero_or_negative_block_m_is_rejected():
    with pytest.raises(ValueError):
        m_tiles_for_row(load_row(1024, 8, 100), 0)


# ------------------------------------------------------------------ the report


def write_sweep(path: Path, points, tile_eff=None, model="mixtral-8x7b") -> Path:
    from moe.bench.schema import COLUMNS
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for t, ms in points:
            for k in (-0.005, 0.0, 0.005):
                row = {c: "" for c in COLUMNS}
                row.update({"impl": "vllm_fused_experts", "model": model,
                            "dtype": "bf16", "routing_kind": "uniform",
                            "num_tokens": int(t), "ms_p50": ms * (1 + k),
                            "correctness_passed": "True", "throttled": "False"})
                if tile_eff is not None:
                    row.update({"load_total_rows": int(t) * 2,
                                "load_active_experts": 8,
                                "load_max_rows": int(t) // 4,
                                "load_tile_eff_bm128": tile_eff(int(t)),
                                "load_tile_eff_bm64": tile_eff(int(t))})
                w.writerow(row)
    return path


def report(path: Path, *extra: str) -> str:
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "crossing_report.py"), str(path),
         "--ridge", "160.3", *extra],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.fixture
def staircase_csv(tmp_path) -> Path:
    """Two tile steps with a tread between them, and tile counts that step with
    the time. `tile_eff` is chosen so the count is 8 below T=512 and 16 above."""
    points = from_slopes(64, 1.0, [0.1, 0.9, 0.1, 0.1, 0.9, 0.9])
    return write_sweep(tmp_path / "staircase.csv", points,
                       tile_eff=lambda t: t * 2 / (8 * 128) if t >= 512 else t * 2 / (8 * 64))


def test_the_report_prints_every_crossing_and_calls_the_cell_a_staircase(staircase_csv):
    out = report(staircase_csv)
    assert "STAIRCASE: the slope crosses 0.5 upward at 2 token counts" in out
    assert "1 of 2:" in out and "2 of 2:" in out
    assert "last over first:" in out


def test_the_report_marks_the_headline_number_as_one_of_several(staircase_csv):
    """The `measured` line is what gets copied into a table, so the ambiguity
    has to travel on that line and not only in a block below it."""
    line = next(ln for ln in report(staircase_csv).splitlines()
                if "measured (slope crosses" in ln)
    assert line.rstrip().endswith("[1 of 2]"), line


def test_the_report_says_nothing_about_staircases_when_the_cell_crosses_once(tmp_path):
    """Silent in the ordinary case, so a single-crossing run keeps its shape."""
    single = write_sweep(tmp_path / "single.csv",
                         from_slopes(64, 1.0, [0.0, 0.0, 1.0, 1.0, 1.0]))
    out = report(single)
    assert "STAIRCASE" not in out
    assert "[1 of" not in out


def test_the_report_prints_the_tile_count_beside_every_token_count(staircase_csv):
    """The steps have to be readable in the table, not just asserted underneath
    it: a reader who can see the tile count holding across three token counts
    does not need to be told the tread is real."""
    out = report(staircase_csv)
    assert "M-tiles" in out
    table = [ln for ln in out.splitlines() if ln.startswith("    ") and "0." in ln]
    assert any("(+" in ln for ln in table), out


def test_the_tile_column_is_counted_at_the_block_the_flag_names(tmp_path):
    """`--block-m` is a stated assumption, so changing it has to change the
    column. Same rows at 64 give twice the tiles of the same rows at 128."""
    points = from_slopes(64, 1.0, [0.1, 0.9, 0.1, 0.1, 0.9, 0.9])
    path = write_sweep(tmp_path / "tiles.csv", points,
                       tile_eff=lambda t: 0.5)
    assert "M-tiles at BLOCK_M 128" in report(path)
    assert "M-tiles at BLOCK_M 64" in report(path, "--block-m", "64")


def test_the_report_says_so_when_no_row_determines_a_tile_count(tmp_path):
    """An empty column with no explanation reads as "no steps here", which is
    the opposite of what a missing load column means."""
    path = write_sweep(tmp_path / "bare.csv", from_slopes(64, 1.0, [0.1, 0.9, 0.1, 0.9]))
    out = report(path)
    assert "rows carry no usable M-tile count" in out
    assert "--" in out
