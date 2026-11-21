"""The 160.3 / 176.2 pair is a reproducibility spread, not a ridge band, and
which finding survives it.

MEASURED, H200, three calibrations of the SAME machine:

    bandwidth   4377.2 -> 4374.5 -> 4374.5 GB/s      0.06% apart
    bf16 GEMM    701.6 ->  770.9 ->  668.5 TFLOP/s   15.5% apart
    ridge        160.3 ->  176.2 ->  152.8 FLOP/byte 15.5% apart

The bandwidth is reproducible; the compute term is not, and it got WORSE on
2026-09-09: that session's calibration sampled the GEMM's clock UNDER LOAD
rather than after it (moe/bench/calibrate.py, LoadedClock), which is the
honest reading and the low one, and it puts the card's own ridge at 152.8,
BELOW both ends of the withdrawn pair rather than inside them. Retraction
(e) is stronger for it: the pair is not a band the card's ridge sits in, it
is two session artefacts that happen to bracket a third. Not because of the clock:
the three calibrations ran the GEMM at 1845, 1560 and 1530 MHz and reached 71.4%,
83.2% and 93.2% of their own clock's peak, so the clock moves 20.6% and the
achieved rate moves 9.9% the other way. The spread is in achieved efficiency.

WHAT THIS FILE USED TO SAY, AND WHY IT WAS WRONG. Its docstring opened "the
ridge is a band" and declared `RIDGE_BAND = (160.3, 176.2)` (withdrawn) as "the two measured
H200 ridges". That pair was withdrawn on 2026-09-02 as the ridge of any card
(`docs/FINDINGS.md` RETRACTIONS (e), `profiles.WITHDRAWN_RIDGE_BAND_WHY`): it
is how badly one card's compute ceiling reproduces across sessions, 26 ladder
reports on BOTH cards had been scored against it, and the H200's own committed
calibration put its ridge at 162.8 on 2026-09-02 and at 152.8 on 2026-09-09.
Every number below that depends on the ridge is read from the committed file
rather than pinned, so the next recalibration moves them without a green test
asserting a superseded one. The constant survives below
under a name that says what it is, because the ridge-independence test needs
two different ridges and these two are the ones the study's absolutes were
published at. And the crossings it pins were labelled the "canonical published
set" when they are the POOLED record: seven routing regimes, retracted as the
headline on 2026-09-02 in favour of the uniform-only 313 / 730 / 931 / 2925
(`docs/FINDINGS.md` C2 and C5). A green test that pins a retracted number is
the project's recurring defect in test form; each number here now carries
its label.

WHAT SURVIVES is the comparison between spans of different extent: both sides
divide by the same predicted crossing, so the ridge cancels exactly, and the
0.561 separation is the same at 160.3, at 176.2, at the card's own 162.8 and
at any number at all. That is algebra, and it is pinned. What it MEANS was
DOWNGRADED on 2026-09-01 (`docs/FINDINGS.md`, "0.563, and it is probably an
artefact"): the first-passage detector reads a BLOCK_M tile step, the two spans
use different tiles, and taking the last crossing instead of the first moves
the separation to 0.889. Ridge-independent is not the same as real.
"""
from __future__ import annotations

import pytest

from moe.bench import profiles, roofline
from moe.bench.ridge import crossing_batch

#: WITHDRAWN as a ridge band, RETRACTIONS (e). Two compute calibrations of one
#: H200, 9.9% apart; the number the absolutes below were published at (160.3)
#: and the number the whole-layer arm's shipped yaml carries (176.2). Kept
#: because the ridge-independence test needs two ridges that are not equal.
WITHDRAWN_H200_PAIR = (160.3, 176.2)

#: The H200's OWN bf16 ridge, from its committed calibration through the same
#: resolver every script uses, so the number has a file behind it.
RIDGE_H200 = roofline.load_hardware("measured_nvidia_h200").ridge_point("bf16")

#: Measured crossings, THE POOLED RECORD: seven routing regimes, `(five-stage,
#: one-stage)` pairs, retracted as the headline 2026-09-02 (pooling is invalid
#: for a crossing, C5 Defect 1). Kept as the record the published absolutes
#: were computed from and to pin the algebra; never quote a number from it as
#: current.
POOLED_CROSSINGS = {
    "mixtral-8x7b": ((454, 464), (938, 409)),
    "qwen2-57b-a14b": ((810, 819), (1277, 1508)),
    "deepseek-v2-lite": ((922, 1025), (1794, 2027)),
    "deepseek-v3": ((3240, 3048), (6446, 6525)),
}

#: The corrected five-stage crossings, `vllm_fused_experts` bf16, UNIFORM
#: routing only (`docs/FINDINGS.md` RETRACTIONS, "The C2 headline is at pooled
#: routing"). The uniform one-stage crossings are not tabulated per model in
#: the docs, so the uniform-only separation (0.5602) is not reproduced here;
#: what is pinned is how far pooling moved the five-stage absolute.
UNIFORM_FIVE_STAGE_VLLM = {
    "mixtral-8x7b": 313,
    "qwen2-57b-a14b": 730,
    "deepseek-v2-lite": 931,
    "deepseek-v3": 2925,
}


def _means(ridge: float, table: dict = POOLED_CROSSINGS) -> tuple[float, float]:
    five, one = [], []
    for model, (f5, o1) in table.items():
        p = crossing_batch(model, ridge, "bf16")
        five += [v / p for v in f5]
        one += [v / p for v in o1]
    return sum(five) / len(five), sum(one) / len(one)


def test_the_withdrawn_pair_is_no_cards_ridge_and_is_not_a_band_around_one():
    """THE RETRACTION, (e). Neither 160.3 nor 176.2 is the ridge of any
    committed calibration, and the card's own ridge is not inside the pair
    either: on 2026-09-02 it was 162.8, between the two, and on 2026-09-09 it
    is 152.8, BELOW both. A pair that brackets a card's ridge in one session
    and sits entirely above it in the next is a reproducibility spread and
    not a band. What is asserted here is the RELATION, read from whatever
    calibration is committed, not the number: the pair is separated from the
    card's own ridge, and the band `profiles` places grids from is the span
    of the cards' own ridges, which the pair is not.
    """
    lo, hi = WITHDRAWN_H200_PAIR
    assert RIDGE_H200 == pytest.approx(
        roofline.load_hardware("measured_nvidia_h200").ridge_point("bf16"))
    assert not lo < RIDGE_H200 < hi or abs(RIDGE_H200 - lo) > 1.0
    assert abs(RIDGE_H200 - lo) > 1.0 and abs(RIDGE_H200 - hi) > 1.0
    band = profiles.calibrated_ridge_band("bf16")
    assert tuple(round(v, 1) for v in band) != WITHDRAWN_H200_PAIR
    assert band[1] == pytest.approx(RIDGE_H200)
    assert "withdrawn" in profiles.WITHDRAWN_RIDGE_BAND_WHY


def test_the_absolute_ratios_move_with_the_ridge():
    """The thing that is NOT safe to quote without naming the ridge. At the
    two withdrawn ends the pooled absolutes are 0.63 / 0.58 (five-stage) and
    1.13 / 1.03 (one-stage). At the card's own ridge they were 0.62 and 1.11
    on the 2026-09-02 calibration, INSIDE that pair, and on 2026-09-09 they
    are 0.66 and 1.18, ABOVE both ends, because the ridge moved below both.
    The two ends are pinned because they are the numbers the study's
    absolutes were published at; the card's own pair is computed from the
    committed calibration, and what is asserted about it is that it moves
    with the ridge in the direction the algebra requires: a lower ridge
    predicts a smaller crossing, so the same measured crossing is a larger
    multiple of it.
    """
    lo_five, lo_one = _means(WITHDRAWN_H200_PAIR[0])
    hi_five, hi_one = _means(WITHDRAWN_H200_PAIR[1])
    assert lo_five == pytest.approx(0.63, abs=0.01)
    assert hi_five == pytest.approx(0.58, abs=0.01)
    assert lo_one == pytest.approx(1.13, abs=0.01)
    assert hi_one == pytest.approx(1.03, abs=0.01)
    own_five, own_one = _means(RIDGE_H200)
    assert hi_five < lo_five and hi_one < lo_one
    if RIDGE_H200 < WITHDRAWN_H200_PAIR[0]:
        assert own_five > lo_five and own_one > lo_one
    elif RIDGE_H200 > WITHDRAWN_H200_PAIR[1]:
        assert own_five < hi_five and own_one < hi_one
    else:
        assert hi_five < own_five < lo_five and hi_one < own_one < lo_one


def test_the_separation_between_span_extents_is_ridge_INDEPENDENT():
    """The algebra that survives. Both sides divide by the same predicted
    crossing, so the ridge cancels and a 9.9% calibration swing, or the move
    to the card's own ridge, changes nothing in the fourth decimal."""
    ridges = (*WITHDRAWN_H200_PAIR, RIDGE_H200)
    ratios = [_means(r)[0] / _means(r)[1] for r in ridges]
    for value in ratios[1:]:
        assert value == pytest.approx(ratios[0], rel=1e-9)
    assert ratios[0] == pytest.approx(0.561, abs=0.005)


@pytest.mark.parametrize("ridge", [80.0, 160.3, 176.2, 400.0])  # two points, withdrawn as a band
def test_it_stays_independent_across_an_absurd_range(ridge):
    """Not a coincidence of two nearby numbers: it is algebraic. 160.3 and
    176.2 appear here as two points of the range, not as anyone's band."""
    five, one = _means(ridge)
    assert five / one == pytest.approx(0.561, abs=0.005)


def test_pooling_moved_the_five_stage_absolute_by_more_than_the_ridge_did():
    """Why the pooled record is a record and not the headline. vLLM's
    five-stage absolute at 160.3 is 0.628 pooled and 0.543 uniform-only, a
    13% move, against the 9% the whole withdrawn ridge pair moves it. The
    restriction that killed C5 moves this number more than the ridge does."""
    vllm_pooled = {m: ((f5[0],), ()) for m, (f5, _) in POOLED_CROSSINGS.items()}
    vllm_uniform = {m: ((t,), ()) for m, t in UNIFORM_FIVE_STAGE_VLLM.items()}

    def five_only(table: dict, ridge: float) -> float:
        vals = [v / crossing_batch(m, ridge, "bf16")
                for m, (f5, _) in table.items() for v in f5]
        return sum(vals) / len(vals)

    pooled = five_only(vllm_pooled, WITHDRAWN_H200_PAIR[0])
    uniform = five_only(vllm_uniform, WITHDRAWN_H200_PAIR[0])
    assert pooled == pytest.approx(0.628, abs=0.005)
    assert uniform == pytest.approx(0.543, abs=0.005)
    ridge_move = 1 - WITHDRAWN_H200_PAIR[0] / WITHDRAWN_H200_PAIR[1]
    assert (pooled - uniform) / pooled > ridge_move


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
