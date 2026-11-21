"""What ridge does a kernel actually meet, as opposed to the datasheet's?

THE OPEN ITEM THIS ADDRESSES. Measured crossings sit below `2R/b`'s prediction
by 0.63x in bf16 and 0.71x in fp8, across four models. One multiplicative factor
that consistent is structure rather than scatter.

A crossing is where arithmetic intensity meets the ridge, and the ridge is
`peak_FLOPS / bandwidth`. Datasheet peaks are not what a kernel reaches. Attain
fraction `f` of peak FLOPs and `g` of peak bandwidth and the ridge actually met
is `(f/g) x nominal`. The crossing is proportional to the ridge, so:

    measured_crossing / predicted_crossing  ==  effective_ridge / nominal_ridge

The left side is already measured. This module computes the right side, which
makes the hypothesis refutable: if the two disagree, achieved-versus-peak is not
the explanation and activation traffic comes back into play.

Both terms come from the SAME model, `flops` and `compulsory_bytes`, so this is
a roofline in the byte model's own units rather than a mix of measured traffic
and modelled work.

THE SECOND HALF OF THIS MODULE answers a different question with the same
columns: not "what ridge did the kernel meet" but "how far did it sit from its
own compulsory byte floor". `traffic_ratio` exists because the study's strongest
positive result, the dtype-gated headroom, had to be reconstructed by hand as
`achieved_bw_gbps / compulsory_gbps` -- the fp8 arm's calibration measured no fp8
ceiling, so `driver.py` never wrote its `implied_traffic_ratio` column. Same
arithmetic, no provenance, which is the shape of every retraction this project
has made. The reconstruction is now a route with a name and a test.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .crossing import timed_rows

#: TFLOP/s over GB/s is 1e12 FLOP/s over 1e9 byte/s.
_TFLOPS_PER_GBPS_TO_FLOP_PER_BYTE = 1000.0


@dataclass(frozen=True)
class Efficiency:
    peak_tflops: float
    peak_gbps: float
    effective_ridge: float
    n_rows: int

    def ratio_against(self, nominal_ridge: float) -> float:
        """`effective / nominal`, the quantity to compare with the crossing offset."""
        if nominal_ridge <= 0:
            raise ValueError("nominal ridge must be positive FLOP/byte")
        return self.effective_ridge / nominal_ridge


def efficiency_from_rows(rows) -> Efficiency | None:
    """Peak achieved FLOP rate and bandwidth over a family of cells.

    The PEAK of each, never the average: a kernel reaches its FLOP peak at large
    batch and its bandwidth peak at small batch, and never both in one cell, so
    an average describes neither end.

    None rather than a zero ridge when nothing usable is present. A zero would
    predict a crossing of zero and read like a finding.
    """
    tflops: list[float] = []
    gbps: list[float] = []
    for r in timed_rows(list(rows)):
        try:
            t, g = float(r["tflops"]), float(r["compulsory_gbps"])
        except (KeyError, TypeError, ValueError):
            continue
        if t > 0:
            tflops.append(t)
        if g > 0:
            gbps.append(g)
    if not tflops or not gbps:
        return None
    pt, pg = max(tflops), max(gbps)
    return Efficiency(
        peak_tflops=pt,
        peak_gbps=pg,
        effective_ridge=pt / pg * _TFLOPS_PER_GBPS_TO_FLOP_PER_BYTE,
        n_rows=max(len(tflops), len(gbps)),
    )


# --- how far a row sits from its own compulsory byte floor ---------------------
#
# THE BUG THIS SECTION EXISTS FOR. The strongest positive result in the study,
# "the headroom is dtype-gated", was computed at a terminal prompt as
# `achieved_bw_gbps / compulsory_gbps` because the fp8 arm's
# `implied_traffic_ratio` column is empty. The column is empty for a reason that
# has nothing to do with the fp8 rows being bad: `driver.py` writes it only when
# `hw.peak(dtype)` is non-zero, that arm's calibration measured no fp8 ceiling,
# so `achieved_peak_tflops` is 0.0 and the gate never opened. The quantity is the
# same either way, and this project has been wrong before precisely because a
# headline was reconstructed by hand from a column that was not there.
#
# So the reconstruction becomes a named route with a test behind it, rather than
# an expression retyped per analysis.

#: The row carried `implied_traffic_ratio`, which `driver.py` wrote at sweep time
#: from `ms x achieved_bandwidth / compulsory_bytes`. Returned verbatim, so this
#: route is bit-identical to the published column by construction.
FROM_PUBLISHED_COLUMN = "implied_traffic_ratio"

#: The column was absent or left at its 0.0 "does not apply" default, and the
#: ratio was rebuilt as `achieved_bw_gbps / compulsory_gbps`. Algebraically the
#: same expression: `compulsory_gbps` is `compulsory_bytes / ms`, so the ms and
#: the bytes cancel exactly as in the published formula.
FROM_CEILING_OVER_COMPULSORY = "achieved_bw_gbps / compulsory_gbps"

#: Agreement the two routes actually reach on the 44,688 published rows that
#: carry both. NOT zero, and pretending otherwise would be a false precision
#: claim: `achieved_bw_gbps` is `bandwidth_bytes_s / 1e9` written to a CSV, and
#: multiplying by 1e9 does not recover the original double, so the two routes
#: associate their multiplications differently and land 1 to 2 ulp apart.
#: Measured max relative difference is 4.5e-16; see tests/test_dtype_headroom.py.
ROUTE_AGREEMENT_REL_TOL = 1e-15


class TrafficRatioUnavailable(LookupError):
    """This row cannot say how far it sat from its compulsory byte floor.

    Raised rather than returning 0.0 or 1.0 for the reason `TileConfigUnrecorded`
    is raised: `implied_traffic_ratio = 0.0` ALREADY means "the column does not
    apply here", it survives a `float()`, it plots, and 2,060 such rows once
    moved a published median from 1.16 to 1.13 and turned 82 sub-floor rows into
    2,142 without a single warning. A second silent default on the same column
    is not something this module will add.
    """


@dataclass(frozen=True)
class TrafficRatio:
    """One row's achievable-bandwidth-over-compulsory-floor, and where it came from.

    `route` travels with the number because the bf16 and fp8 halves of the
    dtype-headroom comparison take DIFFERENT routes: every canonical bf16 row
    carries the published column and not one fp8 row does. A reader who cannot
    see that from the value cannot check it.
    """

    value: float
    route: str

    def __float__(self) -> float:
        return self.value


def traffic_ratio(row, ridge: float | None = None) -> TrafficRatio:
    """How many times the compulsory minimum traffic this row's time allows for.

    Two routes, tried in this order, and the one taken is reported:

      1. `implied_traffic_ratio` if the row carries a positive one. Returned
         unchanged, so an analysis built on this helper reproduces every
         published figure to the bit.
      2. `achieved_bw_gbps / compulsory_gbps`, which is the same expression with
         `ms` and `compulsory_bytes` cancelled.

    and a refusal when neither is available, because the alternative is a
    plausible-looking 0.0.

    WHAT THE SECOND ROUTE CANNOT TELL YOU, and it is the ambiguity this
    signature exists to resolve. A missing `implied_traffic_ratio` has two
    unrelated causes: the cell was compute-bound, where `time x bandwidth` is set
    by FLOPs and bounds nothing about traffic; or the arm's calibration had no
    ceiling for the row's dtype, which is the fp8 arm's case and says nothing
    about the cell at all. The row does not record which. Pass `ridge` -- in the
    row's OWN dtype, via `ridge.ridge_for_dtype` -- and the compute-bound case
    becomes a refusal instead of a number; leave it None and the caller has
    accepted that it does not know.

    An untimed row is refused before either route. `ms_p50 = 0.0` means the cell
    never ran, and its `compulsory_gbps` is a 0.0 default rather than a
    measurement of no bandwidth.
    """
    from .schema import row_float

    if row_float(row, "ms_p50") <= 0.0:
        raise TrafficRatioUnavailable(
            "ms_p50 is 0.0, so this cell never ran: a skipped or uncapturable "
            "graph mode still writes a row and every derived column on it is a "
            "default. Filter with crossing.timed_rows first.")

    if ridge is not None:
        if ridge <= 0:
            raise ValueError("ridge must be positive FLOP/byte")
        intensity = row_float(row, "arith_intensity_compulsory")
        if intensity >= ridge:
            raise TrafficRatioUnavailable(
                f"arith_intensity_compulsory {intensity:.1f} is at or above the "
                f"ridge {ridge:.1f}, so this cell is compute bound and its time "
                f"is set by FLOPs. `time x bandwidth` bounds no traffic here.")

    published = row_float(row, "implied_traffic_ratio")
    if published > 0.0:
        return TrafficRatio(published, FROM_PUBLISHED_COLUMN)

    ceiling = row_float(row, "achieved_bw_gbps")
    compulsory = row_float(row, "compulsory_gbps")
    if ceiling > 0.0 and compulsory > 0.0:
        return TrafficRatio(ceiling / compulsory, FROM_CEILING_OVER_COMPULSORY)

    raise TrafficRatioUnavailable(
        "neither route is available on this row: implied_traffic_ratio is "
        f"{published!r} and achieved_bw_gbps / compulsory_gbps is "
        f"{ceiling!r} / {compulsory!r}. A row with no measured bandwidth "
        "ceiling cannot be scored against one.")


def nearest_rank(ordered: list[float], q: float) -> float:
    """The `ceil(q*n)`-th order statistic. Not interpolated, and deliberately so.

    `crossing._percentile` interpolates, because it summarises a bootstrap whose
    draw count is a free parameter and a stepping edge there would read as a
    change in the measurement. This is the opposite situation: the sample is the
    measured rows themselves, and every value returned should be a ratio some row
    actually reported rather than a blend of two rows.

    It is also what the published dtype-headroom table used. The two conventions
    differ by up to 0.6% on the fp8 eager p90 (7.982 against 7.936), which is
    small and is exactly the kind of unexplained third-decimal drift that costs a
    day to chase, so the convention is named here rather than left to whichever
    helper an analysis happened to import.
    """
    if not ordered:
        raise ValueError("no samples")
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    index = max(1, math.ceil(q * len(ordered))) - 1
    return sorted(ordered)[index]


@dataclass(frozen=True)
class TrafficSummary:
    """A traffic-ratio distribution plus the census of how it was computed."""

    n: int
    p10: float
    median: float
    p90: float
    #: `route -> count`, so a summary that silently mixes the published column
    #: with the reconstruction says so on its face.
    routes: tuple[tuple[str, int], ...]
    #: Rows that refused, with the first reason. Counted rather than dropped:
    #: a silently discarded input is the same class of error as a double-counted
    #: one, which is why `filter_superseded` returns both halves.
    refused: int = 0

    @property
    def route(self) -> str:
        """The single route, or a joined label when the summary mixes them."""
        return " + ".join(name for name, _ in self.routes)


def summarise_traffic_ratios(rows, ridge: float | None = None) -> TrafficSummary | None:
    """p10 / median / p90 of `traffic_ratio` over a family of rows.

    None rather than a zero-filled summary when nothing scored, for the reason
    `efficiency_from_rows` returns None: a zero median reads like a kernel
    sitting below its own byte floor, which is a finding this study has actually
    reported and had to retract.
    """
    values: list[float] = []
    census: dict[str, int] = {}
    refused = 0
    for r in rows:
        try:
            got = traffic_ratio(r, ridge=ridge)
        except TrafficRatioUnavailable:
            refused += 1
            continue
        values.append(got.value)
        census[got.route] = census.get(got.route, 0) + 1
    if not values:
        return None
    values.sort()
    return TrafficSummary(
        n=len(values),
        p10=nearest_rank(values, 0.10),
        median=statistics.median(values),
        p90=nearest_rank(values, 0.90),
        routes=tuple(sorted(census.items())),
        refused=refused,
    )
