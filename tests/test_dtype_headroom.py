"""The dtype-gated headroom result, pinned to rows instead of to a prompt.

THE CLAIM UNDER TEST. Production fused MoE kernels, uniform routing, T <= 64:
in bf16 the incumbent sits close to its compulsory byte floor, and in fp8 at the
same cells it does not, because halving the weight bytes halves the floor without
halving the fixed dispatch and tile-quantisation costs.

THE BUG THIS FILE EXISTS FOR. That table was computed by hand as
`achieved_bw_gbps / compulsory_gbps`, because the fp8 arm's calibration measured
no fp8 ceiling and `driver.py` therefore never wrote its `implied_traffic_ratio`
column. Arithmetically identical, provenance absent. Every retraction this study
has made began exactly there, so the reconstruction is now a named route and the
eight headline numbers are asserted against the published rows.

WHAT THE TESTS BELOW SAY THAT THE HEADLINE DOES NOT. Three of them are
adversarial and two of those weaken the claim as it is currently written:

  - the bf16 side is NOT on the floor. It is at 1.14 to 1.16, which is the same
    "roughly 15% headroom" the supporting-results section of FINDINGS already
    reports. The result is that fp8 has MORE headroom than bf16, not that bf16
    has none.
  - the fp8 EAGER figure is 78% per-call host dispatch. The eager-to-graph gap is
    131 us in fp8 against 4.9 us in bf16, and a cost that were merely fixed in
    time would contribute at most 2x more to an fp8 ratio than to a bf16 one,
    not 27x. The fp8 code path does genuinely more host work per call, which a
    CUDA graph replays away. Production serving uses CUDA graphs.
  - what survives all of that is the GRAPHED residual, +0.337 in ratio, positive
    in 96% of matched cells and in every model. That is the defensible number.

The effect does survive matching: it is present on cells matched across the two
arms on (model, tokens, routing, seed, impl, l2_flush), in every model, in both
modes, and at every token count. It is not an arm artefact.
"""
from __future__ import annotations

import csv
import functools
import statistics
from pathlib import Path

import pytest

from moe.bench.efficiency import (
    FROM_CEILING_OVER_COMPULSORY,
    FROM_PUBLISHED_COLUMN,
    ROUTE_AGREEMENT_REL_TOL,
    TrafficRatioUnavailable,
    nearest_rank,
    summarise_traffic_ratios,
    traffic_ratio,
)
from moe.bench.schema import UNRECORDED, TileConfigUnrecorded
from scripts.dtype_headroom import CELL_KEY, cell_of, load_rows, mode_of

REPO = Path(__file__).resolve().parent.parent
PUBLISHED = REPO / "results/published"

#: The canonical bf16 H200 pool. Four arms because no single one carries all four
#: model geometries, which is why `published.py` can say what is RETIRED but not
#: what is COMPARABLE.
BF16_ARMS = (
    "2026-08-22-standard-sweep",
    "2026-08-26-nvidia_h200-full-three-way-recalibrated",
    "2026-08-28-nvidia_h200-ridge-resolution",
    "2026-08-28-nvidia_h200-h200-v2lite",
)

#: The fp8 arm is PARTIALLY superseded: its two torch spans quantised activations
#: inside the timed region and are replaced by `-fp8-refixed`, while its vLLM and
#: SGLang rows are current. `load_rows` drops the torch spans through
#: `superseded_impls`, which is asserted below rather than assumed.
FP8_ARMS = ("2026-08-28-nvidia_h200-h200-fp8-three-kernel",)


def _paths(arms) -> list[Path]:
    return [p for arm in arms for p in sorted((PUBLISHED / arm).glob("run_*.csv"))]


needs_published = pytest.mark.skipif(
    not all((PUBLISHED / a).exists() for a in BF16_ARMS + FP8_ARMS),
    reason="the published H200 arms are not in this checkout")


def row(**over):
    base = {"ms_p50": "1.0", "implied_traffic_ratio": "0.0",
            "achieved_bw_gbps": "0.0", "compulsory_gbps": "0.0",
            "arith_intensity_compulsory": "0.0"}
    base.update({k: str(v) for k, v in over.items()})
    return base


# --------------------------------------------------------------------------
# The helper: which route, and when it refuses
# --------------------------------------------------------------------------

def test_a_row_carrying_the_published_column_returns_it_unchanged():
    """Bit-identical by construction, which is the only way an analysis built on
    this helper can reproduce a published figure exactly."""
    got = traffic_ratio(row(implied_traffic_ratio="1.1751115524845386"))
    assert got.value == 1.1751115524845386
    assert got.route == FROM_PUBLISHED_COLUMN


def test_a_row_without_the_column_is_rebuilt_from_the_ceiling_over_the_floor():
    """The fp8 arm's whole case: no fp8 compute ceiling was measured, so the
    gate in driver.py never opened and the column stayed at its default."""
    got = traffic_ratio(row(achieved_bw_gbps=4377.2122, compulsory_gbps=3000.0))
    assert got.value == pytest.approx(4377.2122 / 3000.0)
    assert got.route == FROM_CEILING_OVER_COMPULSORY


def test_the_route_travels_with_the_value_so_a_mixed_comparison_cannot_hide_it():
    """The bf16 and fp8 halves of this comparison take DIFFERENT routes. A reader
    who cannot see that from the result cannot check it."""
    a = traffic_ratio(row(implied_traffic_ratio=1.2))
    b = traffic_ratio(row(achieved_bw_gbps=1200.0, compulsory_gbps=1000.0))
    assert a.value == pytest.approx(b.value)
    assert a.route != b.route


def test_a_zero_implied_traffic_ratio_is_never_returned_as_a_measurement():
    """`implied_traffic_ratio = 0.0` means the column does not apply, not that
    the kernel moved no traffic. 2,060 such rows once moved a published median
    from 1.16 to 1.13 and turned 82 sub-floor rows into 2,142."""
    got = traffic_ratio(row(implied_traffic_ratio=0.0,
                            achieved_bw_gbps=1200.0, compulsory_gbps=1000.0))
    assert got.value == pytest.approx(1.2)
    assert got.route == FROM_CEILING_OVER_COMPULSORY


def test_a_row_with_neither_route_is_refused_rather_than_scored_as_zero():
    with pytest.raises(TrafficRatioUnavailable):
        traffic_ratio(row())


def test_an_untimed_row_is_refused_before_either_route_is_tried():
    """`ms_p50 = 0.0` means the cell never ran. Its compulsory_gbps is a default,
    so the reconstruction would return a number for a cell with no measurement."""
    with pytest.raises(TrafficRatioUnavailable) as e:
        traffic_ratio(row(ms_p50=0.0, achieved_bw_gbps=1200.0,
                          compulsory_gbps=1000.0))
    assert "never ran" in str(e.value)


def test_a_compute_bound_row_is_refused_once_a_ridge_is_supplied():
    """Above the ridge the time is set by FLOPs, so `time x bandwidth` bounds no
    traffic at all and the reconstruction would be a number about nothing."""
    with pytest.raises(TrafficRatioUnavailable) as e:
        traffic_ratio(row(arith_intensity_compulsory=400.0,
                          achieved_bw_gbps=1200.0, compulsory_gbps=1000.0),
                      ridge=320.6)
    assert "compute bound" in str(e.value)


def test_without_a_ridge_the_compute_bound_case_cannot_be_detected():
    """Stated as a test because it is the helper's real limit: a missing
    implied_traffic_ratio has two unrelated causes -- compute-bound cell, or an
    arm with no ceiling for the dtype -- and no column distinguishes them."""
    got = traffic_ratio(row(arith_intensity_compulsory=400.0,
                            achieved_bw_gbps=1200.0, compulsory_gbps=1000.0))
    assert got.value == pytest.approx(1.2)


def test_a_nonpositive_ridge_is_refused_rather_than_letting_every_row_through():
    with pytest.raises(ValueError):
        traffic_ratio(row(implied_traffic_ratio=1.2), ridge=0.0)


def test_an_unrecorded_sentinel_raises_instead_of_reading_as_a_default():
    """The v4 sentinel discipline reaches this helper for free because it reads
    through `row_float`. A row stamped UNRECORDED must stop an analysis, not
    contribute a plausible number to a median."""
    with pytest.raises(TileConfigUnrecorded):
        traffic_ratio(row(ms_p50=UNRECORDED))


# --------------------------------------------------------------------------
# The percentile convention, which is not the one crossing.py uses
# --------------------------------------------------------------------------

def test_the_percentile_is_an_order_statistic_and_not_an_interpolation():
    """Every value reported should be a ratio some row actually measured."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert nearest_rank(values, 0.10) == 1.0
    assert nearest_rank(values, 0.90) == 5.0
    assert nearest_rank(values, 0.50) == 3.0


def test_it_differs_from_the_interpolating_percentile_the_crossing_code_uses():
    """Named so nobody swaps one for the other chasing a third-decimal drift.
    On the published fp8 eager tail the two conventions give 7.982 and 7.936."""
    from moe.bench.crossing import _percentile

    values = [float(i) for i in range(10)]
    assert nearest_rank(values, 0.90) == 8.0
    assert _percentile(sorted(values), 0.90) == pytest.approx(8.1)


def test_an_empty_sample_is_refused_rather_than_summarised():
    with pytest.raises(ValueError):
        nearest_rank([], 0.5)
    assert summarise_traffic_ratios([]) is None


def test_a_summary_counts_what_it_refused_instead_of_dropping_it_silently():
    """A silently discarded input is the same class of error as a double-counted
    one, which is why filter_superseded returns both halves."""
    s = summarise_traffic_ratios([row(implied_traffic_ratio=1.2), row()])
    assert s.n == 1 and s.refused == 1


# --------------------------------------------------------------------------
# The two routes, against the real rows
# --------------------------------------------------------------------------

@needs_published
def test_the_published_route_returns_the_column_bit_for_bit():
    rows, _ = load_rows(_paths(BF16_ARMS))
    assert rows
    for r in rows:
        got = traffic_ratio(r)
        assert got.route == FROM_PUBLISHED_COLUMN
        assert got.value == float(r["implied_traffic_ratio"])


@needs_published
def test_the_reconstruction_agrees_with_the_column_but_is_not_bit_identical():
    """THE HONEST VERSION OF "bit-identical". The two routes are the same algebra
    and they are NOT the same float: `achieved_bw_gbps` is `bandwidth_bytes_s`
    divided by 1e9 and written to a CSV, and binary floating point cannot invert
    that division, so the two associate their multiplications differently. Over
    every published row that carries the column they agree to 1 to 2 ulp and
    disagree at all on about half of them. Asserting bit-identity here would be a
    false precision claim; asserting nothing would let a real drift through."""
    exact = total = 0
    worst = 0.0
    for path in sorted(PUBLISHED.glob("*/run_*.csv")):
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                published = float(r["implied_traffic_ratio"] or 0.0)
                ceiling = float(r["achieved_bw_gbps"] or 0.0)
                floor = float(r["compulsory_gbps"] or 0.0)
                if published <= 0 or ceiling <= 0 or floor <= 0:
                    continue
                total += 1
                rebuilt = ceiling / floor
                exact += rebuilt == published
                worst = max(worst, abs(rebuilt - published) / published)
    assert total > 40000
    assert worst <= ROUTE_AGREEMENT_REL_TOL
    assert 0 < exact < total, "bit-identity is not achievable through the CSV"


@needs_published
def test_the_bf16_rows_take_one_route_and_every_fp8_row_takes_the_other():
    """The asymmetry the caveat is about, asserted rather than described: not one
    fp8 row carries the published column, so the fp8 half of the headline is
    entirely reconstruction."""
    bf16, _ = load_rows(_paths(BF16_ARMS))
    fp8, _ = load_rows(_paths(FP8_ARMS))
    assert {traffic_ratio(r).route for r in bf16} == {FROM_PUBLISHED_COLUMN}
    assert {traffic_ratio(r).route for r in fp8} == {FROM_CEILING_OVER_COMPULSORY}


@needs_published
def test_the_fp8_arms_partial_supersession_is_honoured_by_the_loader():
    """Its two torch spans timed a quantisation pass by mistake and are retired;
    its vLLM and SGLang rows are current. Reading the arm wholesale would put
    9,744 retracted rows into the headline."""
    fp8, notes = load_rows(_paths(FP8_ARMS),
                           impl=("vllm_fused_experts", "sglang_fused_experts",
                                 "torch_scaled_grouped_mm_up",
                                 "torch_scaled_grouped_mm_down"))
    assert any("torch_scaled_grouped_mm" in n for n in notes)
    assert not any(r["impl"].startswith("torch_scaled") for r in fp8)


# --------------------------------------------------------------------------
# The eight published numbers
# --------------------------------------------------------------------------

#: docs/FINDINGS.md, "The headroom is dtype-gated". `(n, p10, median, p90)`.
PUBLISHED_TABLE = {
    ("bf16", "eager"): (279, 1.030, 1.144, 2.833),
    ("bf16", "graph"): (182, 1.090, 1.162, 1.438),
    ("fp8_e4m3", "eager"): (275, 1.158, 1.959, 7.982),
    ("fp8_e4m3", "graph"): (284, 1.157, 1.361, 2.062),
}


@functools.lru_cache(maxsize=1)
def _pooled_summaries():
    rows = load_rows(_paths(BF16_ARMS))[0] + load_rows(_paths(FP8_ARMS))[0]
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["dtype"], mode_of(r)), []).append(r)
    return {k: summarise_traffic_ratios(v) for k, v in groups.items()}


@needs_published
@pytest.mark.parametrize("key", sorted(PUBLISHED_TABLE))
def test_the_published_dtype_headroom_row_reproduces_from_the_rows(key):
    n, p10, median, p90 = PUBLISHED_TABLE[key]
    s = _pooled_summaries()[key]
    assert s.n == n
    assert s.p10 == pytest.approx(p10, abs=5e-4)
    assert s.median == pytest.approx(median, abs=5e-4)
    assert s.p90 == pytest.approx(p90, abs=5e-4)


@needs_published
def test_the_row_counts_depend_on_the_throttle_filter_and_the_table_says_so():
    """279 / 182 / 275 / 284 is the UNTHROTTLED count. Including throttled rows
    gives 336 / 184 / 336 / 288 and moves the fp8 eager median from 1.959 to
    1.414, which is a third of the effect. A table that did not name the filter
    would not be reproducible."""
    rows = (load_rows(_paths(BF16_ARMS), include_throttled=True)[0]
            + load_rows(_paths(FP8_ARMS), include_throttled=True)[0])
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["dtype"], mode_of(r)), []).append(r)
    assert len(groups[("fp8_e4m3", "eager")]) == 336
    assert summarise_traffic_ratios(
        groups[("fp8_e4m3", "eager")]).median == pytest.approx(1.414, abs=5e-4)


# --------------------------------------------------------------------------
# Adversarial: does it survive matching, and what is it actually made of
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _matched():
    bf16, _ = load_rows(_paths(BF16_ARMS))
    fp8, _ = load_rows(_paths(FP8_ARMS))
    a = {(cell_of(r), mode_of(r)): r for r in bf16}
    b = {(cell_of(r), mode_of(r)): r for r in fp8}
    return a, b, sorted(set(a) & set(b))


@needs_published
def test_the_cells_really_are_matchable_across_the_two_arms():
    """The comparison's premise. The two halves come from different arms with
    different calibrations and different commits, so if they did not overlap on
    (model, tokens, routing, seed, impl, l2_flush) the pooled difference could be
    unequal cell coverage rather than dtype."""
    a, b, shared = _matched()
    assert "seed" in CELL_KEY and "l2_flush" in CELL_KEY
    assert len(shared) == 426
    assert len(shared) / len(a) > 0.9


@needs_published
def test_the_effect_is_larger_on_the_matched_subset_not_smaller():
    """The adversarial check that could have killed it. It does not: matching
    RAISES the paired fp8/bf16 median, so the pooled table was if anything
    understating the gap through cell composition."""
    a, b, shared = _matched()
    for mode, expected in (("eager", 1.79), ("graph", 1.27)):
        keys = [k for k in shared if k[1] == mode]
        paired = [traffic_ratio(b[k]).value / traffic_ratio(a[k]).value
                  for k in keys]
        assert statistics.median(paired) == pytest.approx(expected, abs=0.01)
        assert sum(p > 1 for p in paired) / len(paired) > 0.95


@needs_published
def test_it_survives_per_model_rather_than_being_carried_by_one():
    """deepseek-v2-lite is a third of the rows and has by far the largest gap, so
    a pooled median could be entirely its. Every model moves the same way, in
    both modes, which is what makes this a dtype effect and not one cell."""
    a, b, shared = _matched()
    models = {k[0][0] for k in shared}
    assert len(models) == 4
    for model in models:
        for mode in ("eager", "graph"):
            keys = [k for k in shared if k[0][0] == model and k[1] == mode]
            paired = [traffic_ratio(b[k]).value / traffic_ratio(a[k]).value
                      for k in keys]
            assert statistics.median(paired) > 1.1, (model, mode)


@needs_published
def test_the_bf16_side_is_not_actually_on_the_compulsory_floor():
    """AGAINST THE HEADLINE AS WRITTEN. "the incumbent sits essentially ON the
    compulsory byte floor and there is nothing to recover" is not what the rows
    say: bf16 is at 1.14 eager and 1.16 graph, which is the SAME "roughly 15%
    headroom" the supporting-results section of FINDINGS reports for these
    kernels. The result is that fp8 has more headroom than bf16, not that bf16
    has none."""
    summaries = _pooled_summaries()
    for mode in ("eager", "graph"):
        assert 1.10 < summaries[("bf16", mode)].median < 1.20


@needs_published
def test_most_of_the_fp8_eager_gap_is_dispatch_and_the_residual_is_much_smaller():
    """The decomposition the eager/graph split exists to produce. On cells
    measured in all four of (bf16, fp8) x (eager, graph):

        fp8_eager - bf16_eager
           = (fp8_graph - bf16_graph)                                 residual
           + [(fp8_eager - fp8_graph) - (bf16_eager - bf16_graph)]     dispatch

    Dispatch is +1.47 of a total +1.89, so the 1.959 headline is about three
    quarters per-call host overhead. The residual, +0.34, is the part a CUDA
    graph cannot replay away, and it is the number a serving system would feel.
    """
    a, b, shared = _matched()
    quad = [c for c in {k[0] for k in shared}
            if all((c, m) in a and (c, m) in b for m in ("eager", "graph"))]
    assert len(quad) == 170

    def v(src, cell, mode):
        return traffic_ratio(src[cell, mode]).value

    residual = [v(b, c, "graph") - v(a, c, "graph") for c in quad]
    dispatch = [(v(b, c, "eager") - v(b, c, "graph"))
                - (v(a, c, "eager") - v(a, c, "graph")) for c in quad]
    total = [v(b, c, "eager") - v(a, c, "eager") for c in quad]

    assert statistics.median(residual) == pytest.approx(0.337, abs=0.005)
    assert statistics.median(dispatch) == pytest.approx(1.474, abs=0.005)
    assert statistics.median(total) == pytest.approx(1.894, abs=0.005)
    assert statistics.median(dispatch) > 4 * statistics.median(residual)
    # The residual is what survives, and it survives nearly everywhere.
    assert sum(x > 0 for x in residual) / len(residual) > 0.95


@needs_published
def test_the_fp8_eager_overhead_is_too_large_to_be_the_halved_floor_alone():
    """THE FINDING THAT MOST WEAKENS THE HEADLINE. The stated mechanism is that a
    FIXED cost is divided by a floor that fp8 halves, which caps the fp8 penalty
    at 2x the bf16 one. In TIME the eager-to-graph gap is 4.9 us in bf16 and
    131 us in fp8 on the SAME cells: 27x, not 2x. So the fp8 code path does
    genuinely more host work per call, and it is not a byte-floor effect.

    The floor really is exactly halved (the byte model charges fp8 weights at one
    byte and, correctly, activations at bf16), which is why that side is asserted
    too: the excess is in the numerator, not the denominator.
    """
    a, b, shared = _matched()
    quad = [c for c in {k[0] for k in shared}
            if all((c, m) in a and (c, m) in b for m in ("eager", "graph"))]

    def gap_us(src, cell):
        return (float(src[cell, "eager"]["ms_p50"])
                - float(src[cell, "graph"]["ms_p50"])) * 1e3

    bf16_gap = statistics.median(gap_us(a, c) for c in quad)
    fp8_gap = statistics.median(gap_us(b, c) for c in quad)
    assert bf16_gap == pytest.approx(4.9, abs=0.5)
    assert fp8_gap == pytest.approx(131.3, abs=1.0)
    assert fp8_gap > 10 * bf16_gap, "a merely fixed cost could only reach 2x"

    byte_ratio = statistics.median(
        float(b[c, "graph"]["compulsory_bytes"])
        / float(a[c, "graph"]["compulsory_bytes"]) for c in quad)
    assert byte_ratio == pytest.approx(0.5, abs=0.01)


@needs_published
def test_in_graph_mode_fp8_is_faster_than_bf16_but_by_less_than_the_bytes():
    """The mechanism, stated where it is actually visible. Replayed from a graph,
    fp8 takes 0.68x the time for 0.50x the bytes on the same cells. That excess
    IS the fixed cost surviving a halved floor, and it is the whole of the
    defensible result."""
    a, b, shared = _matched()
    quad = [c for c in {k[0] for k in shared}
            if all((c, m) in a and (c, m) in b for m in ("eager", "graph"))]
    speed = statistics.median(float(b[c, "graph"]["ms_p50"])
                              / float(a[c, "graph"]["ms_p50"]) for c in quad)
    assert 0.5 < speed < 0.8
