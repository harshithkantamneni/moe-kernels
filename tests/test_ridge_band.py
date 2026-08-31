"""The ridge is a band, and which findings survive it.

MEASURED, H200, two calibrations of the SAME machine:

    bandwidth   4377.2 -> 4374.5 GB/s      0.06% apart
    bf16 GEMM    701.6 ->  770.9 TFLOP/s   9.9% apart
    ridge        160.3 ->  176.2 FLOP/byte 9.9% apart

The bandwidth is reproducible; the compute term is not. Not because of the clock:
the three calibrations ran the GEMM at 1845, 1560 and 1530 MHz and reached 71.4%,
83.2% and 93.2% of their own clock's peak, so the clock moves 20.6% and the
achieved rate moves 9.9% the other way. The spread is in achieved efficiency. So
the ridge is a range, not the single 160.3 that STUDY's predicted crossings were
computed against, and every absolute measured/predicted figure carries it.

WHAT DOES NOT CARRY IT is the comparison between spans of different extent. Both
sides divide by the same predicted crossing, so the ridge cancels exactly. That
is why the five-stage-versus-one-stage result is the robust one and the
absolutes are not.
"""
from __future__ import annotations

import pytest

from moe.bench.ridge import crossing_batch

#: The two measured H200 ridges. Both are real measurements of one machine.
RIDGE_BAND = (160.3, 176.2)

#: Measured crossings, canonical published set. (five-stage, one-stage) pairs.
MEASURED = {
    "mixtral-8x7b": ((454, 464), (938, 409)),
    "qwen2-57b-a14b": ((810, 819), (1277, 1508)),
    "deepseek-v2-lite": ((922, 1025), (1794, 2027)),
    "deepseek-v3": ((3240, 3048), (6446, 6525)),
}


def _means(ridge: float) -> tuple[float, float]:
    five, one = [], []
    for model, (f5, o1) in MEASURED.items():
        p = crossing_batch(model, ridge, "bf16")
        five += [v / p for v in f5]
        one += [v / p for v in o1]
    return sum(five) / len(five), sum(one) / len(one)


def test_the_absolute_ratios_move_with_the_ridge():
    """The thing that is NOT safe to quote without the band."""
    lo_five, lo_one = _means(RIDGE_BAND[0])
    hi_five, hi_one = _means(RIDGE_BAND[1])
    assert lo_five == pytest.approx(0.63, abs=0.01)
    assert hi_five == pytest.approx(0.58, abs=0.01)
    assert lo_one == pytest.approx(1.13, abs=0.01)
    assert hi_one == pytest.approx(1.03, abs=0.01)


def test_the_separation_between_span_extents_is_ridge_INDEPENDENT():
    """The finding that survives. Both sides divide by the same predicted
    crossing, so the ridge cancels and a 9.9% calibration swing changes nothing."""
    ratios = [_means(r)[0] / _means(r)[1] for r in RIDGE_BAND]
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-9)
    assert ratios[0] == pytest.approx(0.561, abs=0.005)


@pytest.mark.parametrize("ridge", [80.0, 160.3, 176.2, 400.0])
def test_it_stays_independent_across_an_absurd_range(ridge):
    """Not a coincidence of two nearby numbers: it is algebraic."""
    five, one = _means(ridge)
    assert five / one == pytest.approx(0.561, abs=0.005)


def test_fp8_reaches_less_of_its_peak_than_bf16_does():
    """MEASURED: bf16 770.9 TFLOP/s at 1530 MHz is 93.2% of that clock's peak;
    fp8 1409.2 at 1740 MHz is 74.9%. So the per-clock advantage is 1.607, not
    the datasheet's 2.0, and the 1.828 in the calibration file conflates the
    format difference with the clock difference between the two measurements."""
    sm = 132
    bf16, bf16_clk, bf16_per = 770.916, 1530, 4096
    fp8, fp8_clk, fp8_per = 1409.168, 1740, 8192
    eff_bf16 = bf16 / (sm * bf16_per * bf16_clk * 1e6 / 1e12)
    eff_fp8 = fp8 / (sm * fp8_per * fp8_clk * 1e6 / 1e12)
    assert eff_bf16 == pytest.approx(0.932, abs=0.005)
    assert eff_fp8 == pytest.approx(0.749, abs=0.005)
    per_clock = (fp8 / fp8_clk) / (bf16 / bf16_clk)
    assert per_clock == pytest.approx(1.607, abs=0.01)
    assert per_clock < 2.0, "fp8 does not reach twice bf16 at equal clock"
