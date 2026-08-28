"""Finding the ridge crossing IN THE DATA, not in the model.

`arith_intensity_compulsory` crosses the ridge exactly where the byte model says
it will, so comparing it to the ridge tests arithmetic, not hardware. To confirm
C2 the crossing has to be read off measured TIME.

The signature is a change of slope. Below the crossing the layer is weight-bound
and every expert's weights are read whatever the batch, so total time barely
moves as T grows: `d(log ms)/d(log T) -> 0`. Above it the layer is compute-bound
and time is proportional to work: the slope goes to 1. The crossing is where the
measured slope passes halfway between the two regimes.

That definition never consults the byte model, so C2's prediction can fail.
"""
from __future__ import annotations

import math

import pytest

from moe.bench.crossing import crossing_from_points, local_slopes


def synth(crossing: float, floor_ms: float = 1.0):
    """time = max(floor, floor * T / crossing): flat, then linear. The idealised
    shape of a roofline, sampled where a real sweep samples."""
    pts = []
    for t in (64, 128, 256, 512, 1024, 2048, 4096):
        pts.append((t, max(floor_ms, floor_ms * t / crossing)))
    return pts


def test_slope_is_zero_below_and_one_above():
    s = dict(local_slopes(synth(512)))
    lo = [v for k, v in s.items() if k < 200]
    hi = [v for k, v in s.items() if k > 1500]
    assert all(abs(v) < 0.05 for v in lo), f"flat region has slope {lo}"
    assert all(abs(v - 1.0) < 0.05 for v in hi), f"linear region has slope {hi}"


@pytest.mark.parametrize("truth", [256, 512, 1024])
def test_it_recovers_a_known_crossing(truth):
    got = crossing_from_points(synth(truth))
    assert got is not None
    # Log-spaced sampling: within a factor of 1.5 is all the grid can resolve.
    assert 1 / 1.5 <= got / truth <= 1.5, f"truth {truth}, measured {got}"


def test_a_purely_flat_sweep_has_no_crossing():
    """Every point below the ridge. Reporting a number here would be inventing
    one, which is how a 2x prediction gets 'confirmed' by noise."""
    assert crossing_from_points([(t, 1.0) for t in (64, 128, 256, 512)]) is None


def test_a_purely_linear_sweep_has_no_crossing_either():
    """Every point above the ridge: the crossing is off the left of the grid."""
    pts = [(t, t / 100.0) for t in (1024, 2048, 4096, 8192)]
    assert crossing_from_points(pts) is None


def test_noise_does_not_move_it_much():
    """Real medians wobble. A definition that chases 2% jitter is useless."""
    clean = synth(512)
    noisy = [(t, ms * (1 + 0.02 * math.sin(t))) for t, ms in clean]
    a, b = crossing_from_points(clean), crossing_from_points(noisy)
    assert a is not None and b is not None
    assert abs(a - b) / a < 0.25


def test_it_refuses_too_few_points():
    assert crossing_from_points([(128, 1.0), (256, 1.0)]) is None


def test_duplicate_token_counts_are_rejected_not_averaged():
    """Two medians for one T means the caller forgot to aggregate, and silently
    picking one would hide it."""
    with pytest.raises(ValueError, match="duplicate"):
        crossing_from_points([(128, 1.0), (128, 2.0), (256, 1.0), (512, 2.0)])


# --- rows that were never timed --------------------------------------------

def test_an_untimed_row_is_not_a_measurement_of_zero():
    """MEASURED, H200 2026-08-28. The first fp8 crossing report printed
    `ms_p50 0.0000` for most of deepseek-v3 and concluded its crossing was at
    2 tokens.

    A skipped or uncapturable graph mode still writes a row, with ms_p50 left at
    its 0.0 default. Feeding those to a median drags it toward zero, and feeding
    them to a slope produces whatever the zeros dictate. Exactly the failure
    `schema.py::_schema_key` exists to warn about: a value that is absent read as
    a measurement of nothing.
    """
    from moe.bench.crossing import timed_rows
    rows = [
        {"ms_p50": "0.5", "capture_status": "eager"},
        {"ms_p50": "0.0", "capture_status": "graph_skipped"},
        {"ms_p50": "", "capture_status": "not_timed"},
        {"ms_p50": "0.7", "capture_status": "captured"},
    ]
    kept = timed_rows(rows)
    assert [r["ms_p50"] for r in kept] == ["0.5", "0.7"]


def test_a_genuinely_tiny_time_is_kept():
    """The filter must reject 'never ran', not 'ran fast'."""
    from moe.bench.crossing import timed_rows
    assert len(timed_rows([{"ms_p50": "0.0001"}])) == 1


def test_a_malformed_time_is_dropped_rather_than_crashing():
    from moe.bench.crossing import timed_rows
    assert timed_rows([{"ms_p50": "n/a"}, {"ms_p50": None}]) == []
