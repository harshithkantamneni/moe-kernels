"""A crossing without an error bar overstates what the sweep resolved.

`crossing_from_points` interpolates between two adjacent slopes, so its
sensitivity to either of them is `1/(s1 - s0)`. Where the curve is flat that
denominator is small and the crossing amplifies timing noise about tenfold.

MEASURED on the published A100 qwen2-57b-a14b sweep (vllm_fused_experts,
uniform, unthrottled): the point estimate is 742 tokens off slopes of 0.492 and
0.725, and moving the single T=512 point by 6% -- no more than the 6.0% its own
four unthrottled replicates already span -- lands the crossing at 627 or at 886.
Times themselves reproduce to 0.2%. Every crossing quoted in this study so far
was quoted bare, and these tests exist so that stops being possible silently.
"""
from __future__ import annotations

import csv
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from moe.bench.crossing import (
    MIN_RELATIVE_SPREAD,
    all_crossings_from_points,
    crossing_from_points,
    crossing_interval,
    relative_spread,
    timed_rows,
)
from moe.bench.ridge import saturation_batch

REPO = Path(__file__).resolve().parent.parent
A100 = REPO / "results/published/2026-08-28-nvidia_a100_sxm4_80gb-a100-cross-card"
H200 = REPO / "results/published/2026-08-28-nvidia_h200-h200-whole-layer"


def published(arm: Path, model: str) -> list[tuple[int, list[float]]]:
    """One replicate list per token count, filtered the way the report filters.

    Uniform routing and one implementation because `impl` names a measured
    SCOPE: pooling two of them describes no kernel. Throttled rows out because a
    clocked-down row is a measurement of the cooler, and correctness-failed rows
    out because a wrong answer produced quickly is not a time.
    """
    by_t: dict[int, list[float]] = {}
    for path in sorted(arm.glob("*.csv")):
        if path.name == "merged.csv":          # the per-run files already cover it
            continue
        with path.open(newline="") as fh:
            rows = timed_rows(list(csv.DictReader(fh)))
        for r in rows:
            if r["impl"] != "vllm_fused_experts" or r["routing_kind"] != "uniform":
                continue
            if r["model"] != model or r["dtype"] != "bf16":
                continue
            if r.get("correctness_passed") not in ("True", "true", "1", ""):
                continue
            if r.get("throttled") in ("True", "true", "1"):
                continue
            by_t.setdefault(int(r["num_tokens"]), []).append(float(r["ms_p50"]))
    return sorted(by_t.items())


needs_published = pytest.mark.skipif(
    not (A100.exists() and H200.exists()),
    reason="published cross-card arms are not in this checkout")


def from_slopes(t0: float, ms0: float, slopes: list[float]) -> list[tuple[float, float]]:
    """A synthetic sweep with EXACTLY the requested `d(log ms)/d(log T)`.

    Each step doubles T, so a slope of `s` multiplies the time by `2**s`. Built
    this way because the leverage claim is about the slopes and nothing else,
    and a hand-written ms curve hides which slopes it actually has.
    """
    pts = [(t0, ms0)]
    for s in slopes:
        t, ms = pts[-1]
        pts.append((t * 2, ms * 2 ** s))
    return pts


def with_spread(points: list[tuple[float, float]], rel: float) -> list[tuple[float, list[float]]]:
    """Two replicates per point, straddling the median by `rel`."""
    return [(t, [ms * (1 - rel), ms * (1 + rel)]) for t, ms in points]


# ---------------------------------------------------------------- the pins


@needs_published
def test_the_a100_mixtral_crossing_is_229_tokens_with_a_band_of_216_to_241():
    """Published 2026-08-28 A100 cross-card arm. The point estimate matches what
    the bare report has always printed; the band is what was missing from it."""
    got = crossing_interval(published(A100, "mixtral-8x7b"),
                            min_tokens=saturation_batch("mixtral-8x7b"))
    assert got is not None
    point, lo, hi = got
    assert round(point) == 229, point
    assert 214 <= lo <= 219, lo
    assert 238 <= hi <= 244, hi


@needs_published
def test_the_a100_deepseek_v3_crossing_is_2848_tokens_with_a_band_of_2766_to_2969():
    """Same arm, the widest-T cell. Its band is 7% of the estimate against
    qwen2's 28% below, because deepseek-v3's slopes separate by 0.44 there where
    qwen2's separate by 0.23: the leverage, not the timing, sets the width."""
    got = crossing_interval(published(A100, "deepseek-v3"),
                            min_tokens=saturation_batch("deepseek-v3"))
    assert got is not None
    point, lo, hi = got
    assert round(point) == 2848, point
    assert 2740 <= lo <= 2790, lo
    assert 2940 <= hi <= 2995, hi


@needs_published
def test_the_a100_qwen2_crossing_of_742_tokens_carries_a_28_percent_band():
    """The cell the whole feature is named after. 742 tokens has been quoted
    bare; it is 742 +101/-110, and the flatness of the curve there -- slopes of
    0.492 and 0.725 -- is the entire reason."""
    got = crossing_interval(published(A100, "qwen2-57b-a14b"),
                            min_tokens=saturation_batch("qwen2-57b-a14b"))
    assert got is not None
    point, lo, hi = got
    assert round(point) == 742, point
    assert 625 <= lo <= 650, lo
    assert 835 <= hi <= 855, hi
    assert (hi - lo) / point > 0.20, (lo, hi)


@needs_published
def test_the_h200_and_a100_mixtral_bands_do_not_overlap():
    """Why the band has to exist rather than be nice to have. A100 measures 229
    and H200 316, and the claim that those differ survives only if the bands are
    disjoint. They are: 241 against 299."""
    a_point, _, a_hi = crossing_interval(
        published(A100, "mixtral-8x7b"), min_tokens=saturation_batch("mixtral-8x7b"))
    h_point, h_lo, _ = crossing_interval(
        published(H200, "mixtral-8x7b"), min_tokens=saturation_batch("mixtral-8x7b"))
    assert round(a_point) == 229 and round(h_point) == 316
    assert a_hi < h_lo, (a_hi, h_lo)


# -------------------------------------------- what the band is a band AROUND


@needs_published
def test_the_band_is_a_band_around_the_first_crossing_and_the_a100_mixtral_cell_has_two():
    """The caveat every band in this file carries, and the one above it most.

    `crossing_interval` bands `crossing_from_points`, which is the FIRST
    upcrossing, and the A100 mixtral cell crosses twice: at 229 tokens and again
    at 776. Its curve dips back to a slope of 0.479 over T=512 to 1024 before
    rising to 0.689, which is a tread between two tile steps rather than a
    single flat-to-linear transition.

    So 229 +/- 12 says the first step is well measured, and says nothing about
    whether the first step is the ridge -- a tight band on the wrong quantity is
    the more dangerous kind. `crossing.all_crossings_from_points` reports both,
    and `tests/test_multiple_crossings.py` carries the mechanism.
    """
    points = published(A100, "mixtral-8x7b")
    found = all_crossings_from_points(
        [(t, statistics.median(v)) for t, v in points],
        min_tokens=saturation_batch("mixtral-8x7b"))
    assert [round(x) for x in found] == [229, 776]
    banded, _, _ = crossing_interval(points,
                                     min_tokens=saturation_batch("mixtral-8x7b"))
    assert banded == found[0]


@needs_published
def test_the_cross_card_comparison_puts_a_first_crossing_against_a_lone_one():
    """Which makes the non-overlap above narrower than it reads. The A100 cell
    supplies two crossings and the H200 cell one, on the SAME octave token grid,
    so "229 against 316, bands disjoint" is comparing the A100's first step with
    the H200's only one. Taking the A100's last instead gives 776 against 316
    and reverses the sign of the difference. Nothing here says which is right;
    it says the claim needs the grid that settles it."""
    a100 = all_crossings_from_points(
        [(t, statistics.median(v)) for t, v in published(A100, "mixtral-8x7b")],
        min_tokens=saturation_batch("mixtral-8x7b"))
    h200 = all_crossings_from_points(
        [(t, statistics.median(v)) for t, v in published(H200, "mixtral-8x7b")],
        min_tokens=saturation_batch("mixtral-8x7b"))
    assert len(a100) == 2 and len(h200) == 1
    assert a100[0] < h200[0] < a100[-1]


# ------------------------------------------------------- the spread estimate


def test_the_spread_estimate_is_not_the_standard_deviation_of_the_replicates():
    """A100 deepseek-v3 at T=4096, the real three rows. The sample stdev calls
    this cell 1.24% noisy because squaring lets the one outlier dominate; the
    mean deviation from the median calls it 0.72%. Squared deviations put a
    single throttled replicate in charge of the whole band."""
    reps = [24.7540, 24.2318, 24.2342]
    med = statistics.median(reps)
    assert statistics.stdev(reps) / med == pytest.approx(0.0124, abs=0.0005)
    assert relative_spread(reps) == pytest.approx(0.0072, abs=0.0005)


def test_the_spread_estimate_is_not_the_median_absolute_deviation_either():
    """Same three rows. Their deviations from the median are 0.520, 0.002 and
    0.000, so the MAD is 0.002 -- 0.008% -- and would declare the cell exact
    while one of its three rows sits 2% away. With three replicates the MAD is
    just the middle deviation, which two agreeing rows drive to zero."""
    reps = [24.7540, 24.2318, 24.2342]
    med = statistics.median(reps)
    mad = statistics.median([abs(v - med) for v in reps]) / med
    assert mad < 0.0002, mad
    assert relative_spread(reps) > 30 * mad


def test_a_cell_with_one_replicate_reports_no_spread_so_the_caller_floors_it():
    assert relative_spread([1.234]) == 0.0
    assert relative_spread([]) == 0.0


def test_a_single_replicate_sweep_still_gets_a_band():
    """The floor is the point of the floor. One row per token count is not
    evidence that the timing is exact, it is the absence of evidence either
    way, and an unfloored sigma of 0 would report the crossing to infinite
    precision."""
    points = from_slopes(64, 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0])
    got = crossing_interval([(t, [ms]) for t, ms in points])
    assert got is not None
    point, lo, hi = got
    assert lo < point < hi
    assert hi / lo > 1.01, (lo, hi)


# ---------------------------------------------------------------- leverage


def test_a_flat_curve_gives_a_far_wider_band_than_a_steep_one_at_equal_noise():
    """The claim the whole module rests on. Both sweeps below cross 0.5 in the
    same interval and carry the same 1% replicate spread; only the separation of
    the two bracketing slopes differs. `1/(s1 - s0)` does the rest."""
    steep = with_spread(from_slopes(64, 1.0, [0.0, 0.0, 1.0, 1.0, 1.0]), 0.01)
    flat = with_spread(from_slopes(64, 1.0, [0.40, 0.45, 0.55, 0.60, 0.65]), 0.01)

    _, s_lo, s_hi = crossing_interval(steep)
    _, f_lo, f_hi = crossing_interval(flat)
    steep_width = math.log(s_hi / s_lo)
    flat_width = math.log(f_hi / f_lo)
    assert flat_width > 3 * steep_width, (steep_width, flat_width)


def test_more_replicate_spread_widens_the_band():
    """Sigma comes from the data, so doubling the observed scatter has to show
    up. A band that ignored its input would pass every other test here."""
    points = from_slopes(64, 1.0, [0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    _, tight_lo, tight_hi = crossing_interval(with_spread(points, 0.01))
    _, loose_lo, loose_hi = crossing_interval(with_spread(points, 0.04))
    assert math.log(loose_hi / loose_lo) > 2 * math.log(tight_hi / tight_lo)


def test_the_band_brackets_the_point_estimate():
    points = from_slopes(64, 1.0, [0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    point, lo, hi = crossing_interval(with_spread(points, 0.02))
    assert lo < point < hi
    assert crossing_from_points([(t, statistics.median(v))
                                 for t, v in with_spread(points, 0.02)]) == point


# ------------------------------------------------------------ reproducibility


def test_the_same_seed_gives_the_same_band_and_a_different_seed_does_not():
    points = with_spread(from_slopes(64, 1.0, [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]), 0.02)
    assert crossing_interval(points, seed=7) == crossing_interval(points, seed=7)
    assert crossing_interval(points, seed=7) != crossing_interval(points, seed=8)


def test_it_never_touches_the_global_random_stream():
    """`random.Random(seed)`, not the module-level `random`. A report that
    reseeded the global stream would silently change every other draw in the
    process, and a caller that seeded it would silently change this band."""
    points = with_spread(from_slopes(64, 1.0, [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]), 0.02)
    random.seed(1234)
    before = [random.random() for _ in range(3)]
    random.seed(1234)
    crossing_interval(points)
    after = [random.random() for _ in range(3)]
    assert before == after


def test_the_band_narrows_as_draws_rise_towards_a_stable_answer():
    """4000 draws is a choice, so it has to be shown to be enough: the edges
    move by under 2% between 4000 and 20000, which is far inside the width the
    spread itself produces."""
    points = with_spread(from_slopes(64, 1.0, [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]), 0.02)
    _, lo4, hi4 = crossing_interval(points, draws=4000)
    _, lo20, hi20 = crossing_interval(points, draws=20000)
    assert abs(lo4 - lo20) / lo20 < 0.02, (lo4, lo20)
    assert abs(hi4 - hi20) / hi20 < 0.02, (hi4, hi20)


# ------------------------------------------------------------- refusing to answer


def test_a_sweep_with_no_crossing_gets_no_interval():
    """Same rule as `crossing_from_points`: a flat or a linear grid does not
    bracket the transition, and a band around an invented number is worse than
    the invented number alone because it looks measured."""
    flat = [(t, [1.0, 1.0]) for t in (64, 128, 256, 512, 1024)]
    assert crossing_interval(flat) is None
    linear = [(t, [t / 100.0, t / 100.0]) for t in (1024, 2048, 4096, 8192)]
    assert crossing_interval(linear) is None


def test_untimed_replicates_are_dropped_rather_than_medianed_in():
    """A skipped graph mode writes ms_p50 = 0.0, which is not a fast run. Those
    are dropped by `timed_rows` upstream and again here, so a caller that
    assembles replicate lists by hand cannot reintroduce them. Medianing one in
    would move the point AND read the spread as 33% instead of 1%."""
    points = from_slopes(64, 1.0, [0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    clean = crossing_interval([(t, [ms, ms * 1.02]) for t, ms in points])
    zeroed = crossing_interval([(t, [ms, ms * 1.02, 0.0]) for t, ms in points])
    assert clean == zeroed
    assert relative_spread([1.0, 1.02, 0.0]) == relative_spread([1.0, 1.02])
    assert statistics.median([1.0, 1.02, 0.0]) != statistics.median([1.0, 1.02])


def test_the_min_tokens_floor_applies_to_every_draw_not_just_the_estimate():
    """Below `E/k` a batch misses experts and the slope crosses for a reason
    unrelated to the ridge. If the floor were applied only to the point
    estimate, the perturbed draws could re-bracket down there and the band would
    stretch to a crossing the claim's domain excludes."""
    # Flat then linear below the floor, which is the shape unsaturated routing
    # actually makes, and it brackets a crossing at about 4 tokens.
    early = [(1.0, [1.0]), (2.0, [1.0]), (4.0, [1.0]),
             (8.0, [2.0]), (16.0, [4.0]), (32.0, [8.0])]
    late = with_spread(from_slopes(64, 8.0, [0.0, 0.0, 1.0, 1.0, 1.0]), 0.01)
    point, lo, hi = crossing_interval(early + late, min_tokens=64)
    assert round(point) == 256
    assert lo > 64, lo
    assert hi < 4096, hi
    # Drop the floor from the draws only and the band collapses onto that
    # 4-token artefact, which is what this pins against.
    assert crossing_interval(early + late)[1] < 8


# ------------------------------------------------------------------ the report


def _row(t, ms, impl="vllm_fused_experts", model="mixtral-8x7b", dtype="bf16"):
    return {"impl": impl, "num_tokens": t, "ms_p50": f"{ms}", "model": model,
            "dtype": dtype, "routing_kind": "uniform",
            "correctness_passed": "True", "throttled": "False"}


def _write(path: Path, rows: list[dict]):
    from moe.bench.schema import COLUMNS
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            base = {c: "" for c in COLUMNS}
            base.update(r)
            w.writerow(base)


def _report(path: Path, *extra: str) -> str:
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "crossing_report.py"), str(path),
         "--ridge", "160.3", *extra],
        capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.fixture
def sweep_csv(tmp_path) -> Path:
    """Three replicates a point, 1% apart, over a sweep that does cross."""
    rows = []
    for t, ms in from_slopes(64, 1.0, [0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0]):
        for k in (-0.01, 0.0, 0.01):
            rows.append(_row(int(t), ms * (1 + k)))
    path = tmp_path / "sweep.csv"
    _write(path, rows)
    return path


def test_the_flag_is_off_by_default_and_changes_nothing_when_it_is(sweep_csv):
    """The existing output is what published figures were read off, so turning
    the band on must ADD lines and rewrite none."""
    added = ("90% band", "bands are ", "two slopes with leverage")
    plain = _report(sweep_csv)
    banded = _report(sweep_csv, "--uncertainty")
    assert "90% band" not in plain
    stripped = [ln for ln in banded.splitlines()
                if not any(a in ln for a in added)]
    assert stripped == plain.splitlines()


def test_the_report_prints_a_band_beside_the_point_estimate(sweep_csv):
    banded = _report(sweep_csv, "--uncertainty")
    measured = [ln for ln in banded.splitlines() if "measured (slope crosses" in ln]
    band = [ln for ln in banded.splitlines() if "90% band" in ln]
    assert len(measured) == 1 and len(band) == 1, banded
    # Same column as every other number in that block, and a ratio to the
    # prediction on both edges, since the ratio is what C2 is judged on.
    assert band[0].index("-") > 40, band[0]
    assert band[0].rstrip().endswith("x predicted"), band[0]


def test_the_report_bands_agree_with_the_function(sweep_csv):
    """The report must not re-derive the band with its own filters. Same
    replicate lists in, same numbers out."""
    with sweep_csv.open(newline="") as fh:
        by_t: dict[int, list[float]] = {}
        for r in csv.DictReader(fh):
            by_t.setdefault(int(r["num_tokens"]), []).append(float(r["ms_p50"]))
    _, lo, hi = crossing_interval(sorted(by_t.items()),
                                  min_tokens=saturation_batch("mixtral-8x7b"))
    band = [ln for ln in _report(sweep_csv, "--uncertainty").splitlines()
            if "90% band" in ln][0]
    assert f"{lo:8.0f} - {hi:.0f} tokens" in band, (band, lo, hi)


def test_the_floor_is_the_documented_half_percent():
    """Named rather than inlined because the report quotes it and the study's
    reproducibility claim (times repeat to 0.2%) is what justifies it."""
    assert MIN_RELATIVE_SPREAD == 0.005
