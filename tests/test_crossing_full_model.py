"""Predict the crossing from the SAME byte model the rows are scored against.

THE INCONSISTENCY THIS CLOSES. Every CSV row records
`arith_intensity_compulsory` from `bytes_model`, which charges activations as
well as weights. Every PREDICTION came from `crossing_batch`, which solves
`2R/b = ridge` -- the weight-dominated approximation. So the measured column and
the predicted column used different models, and the gap is not negligible near
the crossings: `2R/b` overstates AI by about 4% for mixtral at T~640 and 7% for
deepseek-v3 at T~5100.

Overstating AI understates the batch needed to reach the ridge, so every
predicted crossing was systematically LOW.

`2R/b` is not wrong, it is a limit: `AI = 2MNK / ((MK + KN + MN)b)` reduces to
`2M/b` when the weight term `KN` dominates. That holds to under 2% at decode and
degrades as the batch grows, which is precisely the direction that moved the
predictions.
"""
from __future__ import annotations

import pytest

from moe.bench.ridge import crossing_batch, crossing_batch_full


@pytest.mark.parametrize("model", ["mixtral-8x7b", "qwen2-57b-a14b",
                                   "deepseek-v2-lite", "deepseek-v3"])
def test_the_full_model_always_predicts_a_LATER_crossing(model):
    """Activations add bytes without adding FLOPs, so the layer needs more rows
    to reach the same intensity. A full-model crossing below the approximate one
    would mean the byte model charges activations negatively."""
    approx = crossing_batch(model, 160.3, "bf16")
    full = crossing_batch_full(model, 160.3, "bf16")
    assert full > approx, f"{model}: full {full:.0f} <= approx {approx:.0f}"


@pytest.mark.parametrize("model", ["mixtral-8x7b", "deepseek-v3"])
def test_but_not_by_much_because_weights_still_dominate(model):
    """If this ever exceeded ~30% the approximation would be the wrong tool
    rather than a slightly optimistic one."""
    approx = crossing_batch(model, 160.3, "bf16")
    full = crossing_batch_full(model, 160.3, "bf16")
    assert full / approx < 1.30


def test_the_gap_is_between_5_and_20_percent_and_does_NOT_track_F_over_H():
    """This test first asserted that wide experts (large F/H) would show the
    biggest gap, since their activations look like a larger share of traffic.
    The data says otherwise:

        mixtral      F/H 3.50   gap 1.050   <- widest experts, SMALLEST gap
        deepseek-v3  F/H 0.29   gap 1.081
        qwen2        F/H 0.71   gap 1.095
        dsv2-lite    F/H 0.69   gap 1.183   <- largest gap

    Second time an F/H ordering has been reached for and not held; the first was
    retracted from STUDY for the same reason. The gap depends on the whole
    activation-to-weight ratio at the crossing, where the crossing itself scales
    with E/k, so F/H alone does not determine it. Pinned as a RANGE, which is
    what the data supports, rather than an ordering it does not."""
    gaps = {m: crossing_batch_full(m, 160.3, "bf16") / crossing_batch(m, 160.3, "bf16")
            for m in ("mixtral-8x7b", "qwen2-57b-a14b", "deepseek-v2-lite",
                      "deepseek-v3")}
    for m, g in gaps.items():
        assert 1.03 < g < 1.25, f"{m}: gap {g:.3f} outside the measured range"
    # And explicitly NOT ordered by F/H: mixtral is widest and gapped least.
    assert gaps["mixtral-8x7b"] < gaps["deepseek-v2-lite"]


def test_it_is_monotonic_in_the_ridge():
    a = crossing_batch_full("mixtral-8x7b", 100.0, "bf16")
    b = crossing_batch_full("mixtral-8x7b", 200.0, "bf16")
    assert b > a


def test_fp8_still_crosses_where_bf16_does_at_the_scaled_ridge():
    """C2's dtype-invariance must survive the better model, or the claim was an
    artefact of the approximation."""
    from moe.bench.ridge import ridge_for_dtype
    bf16 = crossing_batch_full("mixtral-8x7b", ridge_for_dtype(160.3, "bf16"), "bf16")
    fp8 = crossing_batch_full("mixtral-8x7b", ridge_for_dtype(160.3, "fp8_e4m3"),
                              "fp8_e4m3")
    # Not exactly equal now: activations stay bf16 in an fp8 cell, so they are a
    # LARGER share of a halved weight traffic. The invariance is approximate.
    assert fp8 / bf16 == pytest.approx(1.0, abs=0.15)


def test_a_ridge_no_batch_can_reach_returns_None():
    """Rather than a number off the end of the grid, which would look like a
    prediction."""
    assert crossing_batch_full("deepseek-v3", 1e9, "bf16") is None
