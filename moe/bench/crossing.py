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
"""
from __future__ import annotations

import math

#: Halfway between the weight-bound regime (0) and the compute-bound one (1).
DEFAULT_THRESHOLD = 0.5


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


def crossing_from_points(points: list[tuple[float, float]],
                         threshold: float = DEFAULT_THRESHOLD) -> float | None:
    """The token count where the measured slope first crosses `threshold`.

    None when the grid does not bracket it, which is a real answer: it says the
    sweep needs different token counts, not that the crossing does not exist.
    """
    slopes = local_slopes(points)
    if len(slopes) < 2:
        return None
    for (t0, s0), (t1, s1) in zip(slopes, slopes[1:], strict=False):
        if s0 < threshold <= s1:
            if s1 == s0:                       # pragma: no cover - guarded above
                return t1
            f = (threshold - s0) / (s1 - s0)   # interpolate in log T
            return math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
    return None


def _clean(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    seen = [t for t, _ in points]
    dupes = {t for t in seen if seen.count(t) > 1}
    if dupes:
        raise ValueError(
            f"duplicate token counts {sorted(dupes)}: aggregate to one median "
            "per token count before calling, rather than letting this pick one")
    return sorted(points)
