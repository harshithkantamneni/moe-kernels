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
                         threshold: float = DEFAULT_THRESHOLD,
                         min_tokens: float = 0.0) -> float | None:
    """The token count where the measured slope first crosses `threshold`.

    `min_tokens` discards points below it. Pass the model's saturation batch:
    below `E/k` tokens a batch does not reach every expert, so active experts
    and weight traffic grow WITH the batch and time rises nearly linearly. That
    slope crosses the threshold for a reason that has nothing to do with the
    ridge, and without the floor mixtral reported a crossing at 5 tokens against
    a predicted 641. `2R/b` assumes all E experts are active, so those points
    are outside the claim's domain rather than evidence against it.

    None when the grid does not bracket it, which is a real answer: it says the
    sweep needs different token counts, not that the crossing does not exist.
    """
    points = [(t, ms) for t, ms in points if t >= min_tokens]
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
