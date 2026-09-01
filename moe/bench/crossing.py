"""Read the ridge crossing off measured time.

The byte model's `arith_intensity_compulsory` crosses the ridge exactly where
arithmetic says it must, so comparing the two confirms nothing about hardware.
C2 is only tested if the crossing is recovered from measured TIME, with the
model consulted afterwards.

Below the crossing the layer is weight-bound: every active expert's weights are
read whatever the batch, so time barely moves as T grows and
`d(log ms)/d(log T)` tends to 0. Above it the layer is compute-bound and time
tracks work, so the slope tends to 1. The crossing is where the measured slope
passes halfway.

Returning None when the sweep does not bracket the transition is the point. A
grid that is entirely flat, or entirely linear, contains no crossing, and
producing a number from it would be inventing the result C2 is judged on.

THE CURVE CROSSES MORE THAN ONCE, and for eight of the sixteen canonical
uniform cells it does. `crossing_from_points` returns the FIRST one and the
first one is usually a tile step rather than the ridge, so
`all_crossings_from_points` is what an analysis should reach for.
`m_tiles_for_row` puts the tile count beside the token count, which is what
makes a staircase visible instead of arguable.

A crossing read this way inherits the noise of the two slopes it interpolates
between, amplified by `1/(s1 - s0)`, which is small exactly where the curve is
flat. `crossing_interval` pushes the measured replicate spread through the same
interpolation and returns a band, because a bare point estimate off a flat curve
claims a resolution the timing does not have.

The crossing also has a ROUTING domain, and it is narrower than the CSV makes it
look: `2R/b` is a uniform-routing statement and uniform is 14% of the published
cells. `routing_domain` reports which regimes a set of rows actually holds, so
that pooling them has to be a decision somebody made rather than the default
nobody noticed.
"""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..routing.imbalance import TileEfficiencyUndetermined, tile_efficiency_for_row
from .schema import UNRECORDED, TileConfigUnrecorded, has_tile_config, row_float, tile_field

#: Halfway between the weight-bound regime (0) and the compute-bound one (1).
DEFAULT_THRESHOLD = 0.5

#: The one routing regime `2R/b` is defined for. A domain, not a default.
UNIFORM_ROUTING = "uniform"

#: The CSV column that names the regime a row was measured under.
ROUTING_COLUMN = "routing_kind"


def timed_rows(rows: list[dict]) -> list[dict]:
    """Rows that carry a real timing.

    A skipped or uncapturable graph mode still writes a row, with `ms_p50` left
    at its 0.0 default. Those are not measurements of zero, and a median or a
    slope taken over them reports whatever the absent rows dictate. The first
    fp8 crossing report concluded deepseek-v3 crossed at 2 tokens because of it.

    Rejects "never ran", not "ran fast": any strictly positive time is kept.
    """
    out = []
    for r in rows:
        try:
            if float(r.get("ms_p50") or 0.0) > 0.0:
                out.append(r)
        except (TypeError, ValueError):
            continue
    return out


@dataclass(frozen=True)
class RoutingDomain:
    """Which routing kinds fed a crossing, and whether `2R/b` covers them.

    `inside` is the answer most callers want: true only when every row is
    uniform. `mixed` separates the two ways of being outside, because they fail
    differently. One skewed kind gives a crossing of the wrong thing; several
    kinds pooled give a crossing of no thing at all, since the median at each
    token count is then taken across regimes that are on opposite sides of the
    ridge.

    Built by `routing_domain`, whose docstring carries the reasoning.
    """

    counts: dict[str, int]

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self.counts))

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def uniform_rows(self) -> int:
        return self.counts.get(UNIFORM_ROUTING, 0)

    @property
    def uniform_fraction(self) -> float:
        """0.0 rather than a ZeroDivisionError on no rows: an empty selection is
        not in the domain, and the caller already has to handle having nothing
        to report."""
        return self.uniform_rows / self.total if self.total else 0.0

    @property
    def mixed(self) -> bool:
        """More than one regime pooled into the same median."""
        return len(self.counts) > 1

    @property
    def inside(self) -> bool:
        """Every row uniform, and there is at least one row."""
        return self.total > 0 and set(self.counts) == {UNIFORM_ROUTING}

    @property
    def census(self) -> str:
        """`uniform x1344, zipf x4032`, sorted, so the mix is a fact on the page
        rather than something the reader has to go and count."""
        return ", ".join(f"{k} x{self.counts[k]}" for k in self.kinds)

    def warning_lines(self) -> list[str]:
        """The banner for a report whose rows are outside the domain, unindented.

        Empty when `inside`, and empty when there are no rows at all, so a
        caller can print these unconditionally and stay silent in the one case
        where silence is correct.

        Says the mechanism and not just the rule, because the rule alone reads
        like a preference for cleaner data and the mechanism says why the number
        is not a measurement. Names the flag both callers spell the same way.
        """
        if not self.counts or self.inside:
            return []
        if self.mixed:
            return [
                f"ROUTING KINDS POOLED: {self.census}",
                f"  uniform is {self.uniform_fraction:.1%} of the {self.total} "
                "rows feeding this report, and `2R/b` covers",
                "  only uniform. `R = T*k/E` is a mean, and under skew no expert "
                "experiences",
                "  the mean: the busy experts are compute-bound while the quiet "
                "ones are still",
                "  memory-bound AT THE SAME BATCH, so the layer straddles the "
                "ridge and there",
                "  is no single crossing to find. Pooling these is INVALID, not "
                "merely noisy.",
                "  pass --routing uniform to restrict to the domain",
            ]
        return [
            f"ROUTING OUTSIDE THE DOMAIN: {self.census} (no uniform rows)",
            "  `2R/b` describes UNIFORM routing. `R = T*k/E` is a mean, and under "
            "skew",
            "  no expert experiences the mean: the busy experts are compute-bound "
            "while",
            "  the quiet ones are still memory-bound AT THE SAME BATCH, so the "
            "layer",
            "  straddles the ridge and there is no single crossing to find.",
            "  pass --routing uniform to restrict to the domain",
        ]

    def crossing_note(self) -> list[str]:
        """What a crossing read off these rows is, and is not, unindented.

        Separate from `warning_lines` because it belongs beside the number
        rather than in a header: a banner 40 lines above a figure does not stop
        the figure being quoted on its own, and every crossing this study has
        retracted was quoted on its own.
        """
        if not self.counts or self.inside:
            return []
        if self.mixed:
            return [
                f"^ POOLED OVER {len(self.counts)} ROUTING KINDS: that number is "
                "where a BLEND of regimes",
                "  passes the threshold, not a crossing, and not a noisy estimate "
                "of one:",
                "  pooled deepseek-v3 moves 238x under the saturation floor, where "
                "the",
                "  uniform answer does not move at all. --routing uniform gives "
                "the number",
                "  `2R/b` predicts.",
            ]
        return [
            f"^ {self.kinds[0].upper()} ROUTING: that number is where a skewed "
            "layer's blended slope",
            "  passes the threshold. `2R/b` does not predict it, so scoring it "
            "against the",
            "  prediction above measures the skew, not the ridge. --routing "
            "uniform gives",
            "  the comparable number.",
        ]


def routing_domain(rows: Iterable[Mapping], key: str = ROUTING_COLUMN
                   ) -> RoutingDomain:
    """Which routing kinds are present, and whether the set is in the domain.

    THE BUG THIS PREVENTS. `AI = 2R/b` describes UNIFORM routing and nothing
    else. `R = T*k/E` is a mean, and under skew no expert experiences the mean:
    the busy experts are compute-bound while the quiet ones are still
    memory-bound AT THE SAME BATCH, so the layer straddles the ridge and there
    is no single crossing for the slope detector to find. What it finds instead
    is where a blend of two regimes happens to pass the threshold, which moves
    with the mix rather than with the hardware.

    Uniform is 1344 of the 9408 cells in each cross-card arm, 14.3%. The other
    86% are outside the model's domain, so pooling them is INVALID rather than
    merely noisy. Two measured demonstrations, both reproduced from
    `results/published/` and both pinned in `tests/test_routing_domain.py`:

    - The cross-card ratio is unstable across routings even though both cards
      ran the IDENTICAL seven distributions. mixtral-8x7b, `vllm_fused_experts`,
      bf16: uniform 0.73, zipf 0.44, hot 0.46, dirichlet 1.92. That is a 4.3x
      spread on a quantity whose two candidate values are 0.83 (the ridge ratio,
      which is what C5 claims) and 0.82 (the SM-count rival). Those two differ
      by 0.01. The choice of routing regime moves the same number by 1.48, so
      the measurement resolves the analyst's flag and not the hardware.
    - Pooled deepseek-v3 on the H200 whole-layer arm crosses at 3474 tokens with
      the saturation floor and at 14.6 without it, a 238x swing, because the
      pooled curve is still steep where the floor cuts. Uniform gives 3010
      either way, to the last bit. A number that moves 238x under a floor the
      in-domain answer does not notice at all is not reading the ridge.

    Rows with no `routing_kind` count as UNRECORDED, never as uniform. A blank
    column is exactly the case this guard exists to catch, so reading it as the
    in-domain value would defeat it.

    Counts rather than a bare set, because "one stray zipf row" and "six sevenths
    of the input" are the same set and a reader has to be able to tell them
    apart.
    """
    counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get(key) or "").strip() or UNRECORDED
        counts[kind] = counts.get(kind, 0) + 1
    return RoutingDomain(counts)


def local_slopes(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """`d(log ms)/d(log T)` between each adjacent pair, at the geometric mid.

    Geometric because the token grid is log-spaced, so the arithmetic mean would
    place every slope closer to the upper point than the data warrants.
    """
    pts = _clean(points)
    out = []
    for (t0, m0), (t1, m1) in zip(pts, pts[1:], strict=False):
        if m0 <= 0 or m1 <= 0:
            continue
        slope = math.log(m1 / m0) / math.log(t1 / t0)
        out.append((math.sqrt(t0 * t1), slope))
    return out


@dataclass(frozen=True)
class Upcrossing:
    """One upward crossing, with the grid interval whose slope produced it.

    `step_lo` and `step_hi` are the two token counts bounding the interval whose
    slope rose through the threshold, and they are NOT the pair the crossing
    sits between. A slope is stamped at the geometric mid of its interval, so a
    crossing interpolated between two adjacent slopes straddles the grid POINT
    they share, and can land on either side of it: qwen2's first crossing is 730
    tokens and the tile step that caused it is the 1024-to-1152 interval, which
    starts 294 tokens later. An annotation that read the interval off the
    crossing's own value would name 512-to-1024 there and point at the wrong
    step.

    The slopes are kept because their separation is the crossing's leverage:
    `crossing_interval` amplifies timing noise by `1/(slope_above -
    slope_below)`, and a reader deciding whether to trust one of several
    crossings wants that number without recomputing it.
    """

    tokens: float
    step_lo: float
    step_hi: float
    slope_below: float
    slope_above: float


def upcrossings(points: list[tuple[float, float]],
                threshold: float = DEFAULT_THRESHOLD,
                min_tokens: float = 0.0) -> list[Upcrossing]:
    """Every upward crossing, each with the interval that produced it.

    The mechanism, the eight multi-crossing cells and the reason first-passage
    is the wrong reduction are all in `all_crossings_from_points`, which is this
    with the annotation dropped.
    """
    spans = _slope_spans([(t, ms) for t, ms in points if t >= min_tokens])
    found = []
    for (_, _, mid0, s0), (lo1, hi1, mid1, s1) in zip(spans, spans[1:],
                                                      strict=False):
        if not s0 < threshold <= s1:
            continue
        if s1 == s0:                           # pragma: no cover - guarded above
            tokens = mid1
        else:
            f = (threshold - s0) / (s1 - s0)   # interpolate in log T
            tokens = math.exp(math.log(mid0) + f * (math.log(mid1) - math.log(mid0)))
        # The interval annotated is the one the RISING slope spans, not the one
        # the crossing's own value falls in. See `Upcrossing`.
        found.append(Upcrossing(tokens, lo1, hi1, s0, s1))
    return found


def _slope_spans(points: list[tuple[float, float]]
                 ) -> list[tuple[float, float, float, float]]:
    """`(t_lo, t_hi, geometric mid, slope)` per adjacent pair.

    `local_slopes` drops a pair with a non-positive time WITHOUT dropping the
    point, so its output cannot be indexed back onto the token grid: one skipped
    pair and every interval an annotation names is off by one. This keeps the
    endpoints attached instead. The arithmetic is `local_slopes`', and
    `tests/test_multiple_crossings.py` pins the two equal on real rows so the
    duplication cannot drift.
    """
    pts = _clean(points)
    spans = []
    for (t0, m0), (t1, m1) in zip(pts, pts[1:], strict=False):
        if m0 <= 0 or m1 <= 0:
            continue
        spans.append((t0, t1, math.sqrt(t0 * t1),
                      math.log(m1 / m0) / math.log(t1 / t0)))
    return spans


def all_crossings_from_points(points: list[tuple[float, float]],
                              threshold: float = DEFAULT_THRESHOLD,
                              min_tokens: float = 0.0) -> list[float]:
    """EVERY token count where the measured slope crosses `threshold` upward.

    THE BUG THIS PREVENTS, and it moved this study's headline by 59%. The curve
    is not a single transition from flat to linear. It is a STAIRCASE, and a
    first-passage detector reads the first step of the staircase rather than the
    ridge. 8 of the 16 canonical uniform cells cross 0.5 going up more than once:

        mixtral-8x7b  vllm     313 then  800     qwen2  vllm   730 then 1573
        deepseek-v3   vllm    2925 then 6391     mixtral sglang 313 then  778

    Those are on the report's DEFAULT basis, which medians all four timing modes
    into each point. The `crossing-uniform` profile times only L2-warm eager, so
    the arm it produces is a different slice of the same cells; named here
    because `profiles.py` quotes that slice and a reader comparing the two
    should know why they differ. The staircase does not depend on the choice:
    warm-eager alone the same four read 315/800, 732/1564, 2925/6391 and
    316/780, every cell still crossing twice.

    THE MECHANISM IS M-TILE QUANTISATION, not the roofline. M-tiles per expert
    is `ceil(rows_per_expert / BLOCK_M)`, and each extra tile is another pass
    over that expert's weight matrix, so time steps up when a tile is added and
    flatlines while the tile count holds. mixtral, `vllm_fused_experts`,
    uniform, tiles at BLOCK_M 128 read off the rows' own load columns:

        T=512   128 rows/expert   12 tiles   1.2224 ms
        T=576   144              15-16       1.3323   slope 0.731   tiles JUMP
        T=640   160              16          1.3991   slope 0.464
        T=704   176              16          1.4292   slope 0.223
        T=768   192              16          1.4437   slope 0.116   tiles FLAT
        T=1024  256              19-21       1.9088   slope 0.971   tiles JUMP

    The slope therefore spikes above 0.5 at EVERY step and sags below it on
    every tread, and which step the detector lands on is set by where the token
    grid happens to sample, not by the hardware. Across the 78 intervals of the
    eight five-stage Triton cells the measured slope tracks the tile-growth
    exponent `d(log tiles)/d(log T)` at r = 0.88.

    128 is an ASSUMPTION, not a measurement: every published arm is schema v3
    and none of them records the tile that ran. It is not load-bearing here.
    The same rows counted at 64 read 20, 23, 24, 25, 28 then 37 tiles, which
    steps in the same places -- the staircase is a property of quantising rows
    into tiles at all, not of the particular block.

    WHICH CROSSING IS THE RIDGE IS NOT SETTLED HERE, and this function
    deliberately does not choose. What is on the record is that taking the LAST
    moves the five-stage over one-stage separation from 0.5602 to 0.8889, and
    that rows per expert at the last crossing (mean 175.8, CV 21.2%) sit inside
    the measured ridge band of 160.3-176.2 where the first (mean 123.4, CV
    40.0%) does not. A dense token grid is what settles it; reporting both is
    what stops a number being quoted before it is settled.

    `min_tokens` discards points below it. Pass the model's saturation batch:
    below `E/k` tokens a batch does not reach every expert, so active experts
    and weight traffic grow WITH the batch and time rises nearly linearly. That
    slope crosses the threshold for a reason that has nothing to do with the
    ridge, and without the floor mixtral reported a crossing at 5 tokens against
    a predicted 641. `2R/b` assumes all E experts are active, so those points
    are outside the claim's domain rather than evidence against it.

    Empty when the grid does not bracket a crossing at all, which is a real
    answer: it says the sweep needs different token counts, not that the
    crossing does not exist. Downward crossings are not returned -- the
    transition being measured runs flat to linear, and a slope falling back
    through 0.5 is the top of a step, not a regime change.
    """
    return [u.tokens for u in upcrossings(points, threshold, min_tokens)]


def crossing_from_points(points: list[tuple[float, float]],
                         threshold: float = DEFAULT_THRESHOLD,
                         min_tokens: float = 0.0) -> float | None:
    """The FIRST token count where the measured slope crosses `threshold`.

    Kept because published figures and pinned tests were read off it, and
    because `crossing_interval` bands the number those figures quote. It is not
    what a new analysis should call: on half the canonical uniform cells the
    first upcrossing is a tile step and there is a later one, so this function
    answers a question the data does not have a single answer to. Reach for
    `all_crossings_from_points`, whose docstring carries the staircase.

    None rather than an empty list, so the existing callers keep their shape.
    """
    found = all_crossings_from_points(points, threshold, min_tokens)
    return found[0] if found else None


# --- the staircase, overlaid on the slope table ------------------------------

#: The tile efficiencies the schema already stores, keyed by the BLOCK_M each
#: was computed at. Preferred over reconstruction because they were written from
#: the full per-expert distribution, which the CSV does not otherwise keep: above
#: `load_max_rows > block_m` an expert spans several tiles and the stored number
#: is the only thing that still knows how many.
STORED_TILE_EFF = {64: "load_tile_eff_bm64", 128: "load_tile_eff_bm128"}


def recorded_block_m(row: Mapping) -> int | None:
    """The tile the kernel ACTUALLY ran, or None when the row does not say.

    None for every v3 row, where the column did not exist and reads back as the
    UNRECORDED sentinel, and None for a v4 row whose observer never ran. The ten
    published arms are all v3, so in practice a caller has to name a BLOCK_M and
    own that choice -- which is the honest position, since `m_tiles_for_row`
    otherwise reports the staircase of a schedule that may not be the one timed.
    """
    if not has_tile_config(row):
        return None
    value = tile_field(row, "tile_block_m")
    return int(value) or None


def m_tiles_for_row(row: Mapping, block_m: int | None = None) -> float:
    """M-tiles this row's routing forces, summed over its active experts.

    THE NUMBER THE SLOPE TABLE IS MISSING. `ceil(rows_per_expert / BLOCK_M)`
    tiles per expert, each one another pass over that expert's weights, so the
    tile count is the staircase `all_crossings_from_points` describes and this
    is how it gets printed beside the token count. mixtral holds 16 tiles from
    T=576 to T=768 and its time moves 8% across that whole band, then gains
    three tiles by T=1024 and jumps 32%.

    `tile_eff = useful_rows / padded_rows` and `padded_rows = tiles * block_m`,
    so `tiles = total_rows / (tile_eff * block_m)` -- an exact integer, not an
    estimate, whenever `tile_eff` came from the real per-expert counts.

    Prefers the stored `load_tile_eff_bm{64,128}` column, because it was written
    from a distribution the CSV does not keep and stays right where an expert
    spans several tiles. Falls back to `tile_efficiency_for_row`, which
    reconstructs any block size while every expert still fits in one tile and
    raises `TileEfficiencyUndetermined` rather than guessing once one does not.

    `block_m = None` asks the row which tile it ran, and raises when the row
    predates the v4 column rather than reading the sentinel as a number. That
    refusal is the same rule the whole tile_* block exists for: a plausible
    number with no measurement behind it steered this study for days.
    """
    if block_m is not None and block_m <= 0:
        raise ValueError("block_m must be positive")
    if block_m is None:
        block_m = recorded_block_m(row)
        if block_m is None:
            raise TileConfigUnrecorded(
                "this row does not record the tile it ran (v3 predates the "
                "column), so name the BLOCK_M to overlay -- e.g. --block-m 128 "
                "-- rather than letting a default stand in for a measurement")
    total = row_float(row, "load_total_rows")
    if total <= 0.0:
        raise TileEfficiencyUndetermined(
            "load_total_rows is 0 or absent, so this row carries no routing "
            "realisation to count tiles over")
    eff = 0.0
    column = STORED_TILE_EFF.get(block_m)
    if column is not None:
        # `row_float` raises TileConfigUnrecorded on the sentinel rather than
        # coercing it, which is what makes a v3 row loaded through read_csv
        # fall through to reconstruction instead of reading a hole as a number.
        try:
            eff = row_float(row, column)
        except TileConfigUnrecorded:
            eff = 0.0
    if eff <= 0.0:
        eff = tile_efficiency_for_row(row, block_m)
    if eff <= 0.0:
        # `tile_efficiency_for_row` returns 0.0 for a row with rows but no
        # active experts. That is a malformed load column set, not a schedule
        # that computes nothing, and dividing by it would report inf tiles.
        raise TileEfficiencyUndetermined(
            f"tile efficiency at block_m={block_m} is 0, which means "
            "UNRECORDED and not a schedule that computes nothing")
    return total / (eff * block_m)


def _clean(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    seen = [t for t, _ in points]
    dupes = {t for t in seen if seen.count(t) > 1}
    if dupes:
        raise ValueError(
            f"duplicate token counts {sorted(dupes)}: aggregate to one median "
            "per token count before calling, rather than letting this pick one")
    return sorted(points)


#: Floor on a cell's relative spread. Two replicates agreeing to the fourth
#: decimal are two samples of ONE thermal and clock state, not proof the timing
#: is exact, and a single-replicate cell has no spread to observe at all.
#: Treating either as exact hands those points infinite weight in the slope, and
#: the crossing is a slope. 0.5% sits just above the 0.2% that repeated
#: measurements of the same cell reproduce to, so it is a floor on ignorance
#: rather than a claim about any particular cell's noise.
MIN_RELATIVE_SPREAD = 0.005

#: 5th and 95th percentile: a 90% band. Reported rather than a 95% one because
#: the tails are set by draws that re-bracket onto a different pair of slopes,
#: and those are the draws the replicate counts here resolve worst.
BAND_QUANTILES = (0.05, 0.95)

#: Enough draws that the band edges are stable to about 1% across seeds, which
#: is well inside the width the spread itself produces.
DEFAULT_DRAWS = 4000


def relative_spread(replicates: list[float]) -> float:
    """Mean absolute deviation from the median, over the median.

    Not the standard deviation. Squaring lets ONE throttled replicate set the
    whole band: A100 deepseek-v3 at T=4096 holds 24.754, 24.232 and 24.234 ms,
    where the sample stdev calls the spread 1.24% and the mean deviation calls
    it 0.72%.

    Not the median absolute deviation either. With three replicates the MAD is
    the middle deviation, and for that same cell the deviations are 0.520, 0.002
    and 0.000, so the MAD reports 0.008% and declares a cell exact when one of
    its three rows sits 2% away.

    Zero for fewer than two replicates, so the caller applies the floor.
    """
    vals = [v for v in replicates if v > 0.0]
    if len(vals) < 2:
        return 0.0
    med = statistics.median(vals)
    if med <= 0.0:
        return 0.0
    return statistics.fmean(abs(v - med) for v in vals) / med


def crossing_interval(points_with_replicates: list[tuple[float, list[float]]],
                      threshold: float = DEFAULT_THRESHOLD,
                      min_tokens: float = 0.0,
                      draws: int = DEFAULT_DRAWS,
                      seed: int = 0) -> tuple[float, float, float] | None:
    """`(point_estimate, lo, hi)`: the crossing with a 90% band on it.

    THE CROSSING AMPLIFIES TIMING NOISE, and every crossing quoted in this study
    so far has been quoted bare. `crossing_from_points` interpolates between two
    adjacent slopes, so its sensitivity to either of them is `1/(s1 - s0)`, and
    on a flat curve that denominator is small.

    MEASURED on A100 qwen2-57b-a14b (vllm_fused_experts, uniform, unthrottled,
    2026-08-28 cross-card arm). The point estimate is 742 tokens and the two
    bracketing slopes are 0.492 and 0.725, so the leverage is 4.3. Move the
    SINGLE T=512 point by 6% -- no more than the 6.0% its own four unthrottled
    replicates already span, 2.679 to 2.839 ms -- and the crossing lands at 627
    or at 886. That is a 259-token move, 35% of the estimate and 41% of its low
    end, out of a wobble a tenth that size. Times themselves reproduce to 0.2%.

    So the band comes from the data, not from a guessed sigma: the input carries
    every replicate, `relative_spread` reads each token count's own scatter, and
    each point is shaken independently by a lognormal of that width (floored at
    `MIN_RELATIVE_SPREAD`) before the crossing is recomputed. Lognormal because a
    time cannot go negative and the slopes are taken in log space, so the noise
    belongs there too.

    BANDS THE FIRST CROSSING, because that is the number published figures
    quote. On a staircase cell the first crossing is a tile step, and a tight
    band around a tile step says the step is well measured -- which it is --
    and nothing at all about whether it is the ridge. Check
    `all_crossings_from_points` before reading a band as a band on the answer.

    `random.Random(seed)` rather than the module-level `random`, so a report is
    reproducible and calling this does not move any other stream.

    None when the unperturbed points do not bracket the crossing at all, matching
    `crossing_from_points`. Draws that LOSE the bracket are dropped from the
    percentiles rather than clamped to an edge -- a draw with no crossing is not
    a draw with an extreme one -- which makes the band conditional on crossing,
    and it narrows the band when many draws are dropped. None as well if no draw
    keeps it, which says the crossing does not survive its own measurement noise.
    """
    points, sigmas = [], []
    for t, replicates in points_with_replicates:
        vals = [float(v) for v in replicates if float(v) > 0.0]
        if not vals:
            continue
        points.append((float(t), statistics.median(vals)))
        sigmas.append(max(relative_spread(vals), MIN_RELATIVE_SPREAD))

    point = crossing_from_points(points, threshold, min_tokens)
    if point is None:
        return None

    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        shaken = [(t, ms * rng.lognormvariate(0.0, sigma))
                  for (t, ms), sigma in zip(points, sigmas, strict=True)]
        got = crossing_from_points(shaken, threshold, min_tokens)
        if got is not None:
            samples.append(got)
    if not samples:
        return None

    samples.sort()
    lo, hi = (_percentile(samples, q) for q in BAND_QUANTILES)
    return point, lo, hi


def _percentile(ordered: list[float], q: float) -> float:
    """Linear interpolation between order statistics.

    Interpolated rather than `ordered[int(q * n)]` so the edge does not step by
    a whole sample when `draws` changes, which would look like a real change in
    the band.
    """
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])
