"""Predicting where a model crosses from memory-bound to compute-bound.

Claim C2 says arithmetic intensity is `2R/b`, with `R` the rows landing on each
active expert and `b` bytes per element, independent of the expert's architecture.
Once routing saturates and every expert is active, `R = T*k/E`, so

    AI = 2 T k / (E b)     and     AI = ridge  at  T = ridge * b * E / (2k)

This is the whole point of running a second device or a second dtype: the formula
makes a number BEFORE the sweep runs, and the sweep can falsify it. Halving `b`
must halve the crossing. Moving to a card with a lower ridge must lower it too.

The tests check the predictions against the brackets actually measured on the
H200, which is the only evidence that the formula describes this hardware and not
just a tidy derivation.
"""
from __future__ import annotations

import pytest

from moe.bench.ridge import crossing_batch, is_compute_bound, rows_per_expert

H200_MEASURED = 160.4      # 701.65 TFLOP/s / 4374.69 GB/s, measured on the card
H200_DATASHEET = 206.15    # Yun et al. Table I, 989.4 TFLOP/s / 4.8 TB/s
A100_DATASHEET = 153.02    # Yun et al. Table I

# What the 2026-08-26 sweep actually measured: the crossing lies in this bracket.
MEASURED_BRACKETS = {
    "mixtral-8x7b": (256, 1024),
    "qwen2-57b-a14b": (1024, 4096),
    "deepseek-v3": (4096, 8192),
}


@pytest.mark.parametrize("model,bracket", MEASURED_BRACKETS.items())
def test_the_prediction_lands_inside_the_measured_bracket(model, bracket):
    """Both the measured and the datasheet ridge must agree with the sweep."""
    lo, hi = bracket
    for ridge in (H200_MEASURED, H200_DATASHEET):
        got = crossing_batch(model, ridge, dtype="bf16")
        assert lo <= got <= hi, (
            f"{model} at ridge {ridge}: predicted crossing {got:.0f}, "
            f"sweep measured it between {lo} and {hi}")


def test_fp8_halves_the_crossing():
    """The falsifiable prediction the fp8 run exists to test."""
    for model in MEASURED_BRACKETS:
        bf16 = crossing_batch(model, H200_MEASURED, dtype="bf16")
        fp8 = crossing_batch(model, H200_MEASURED, dtype="fp8_e4m3")
        assert fp8 == pytest.approx(bf16 / 2, rel=1e-9), model


def test_a_lower_ridge_lowers_the_crossing():
    """The falsifiable prediction the A100 run exists to test."""
    h200 = crossing_batch("deepseek-v3", H200_DATASHEET, dtype="bf16")
    a100 = crossing_batch("deepseek-v3", A100_DATASHEET, dtype="bf16")
    assert a100 < h200
    assert a100 / h200 == pytest.approx(A100_DATASHEET / H200_DATASHEET, rel=1e-9)


def test_the_dilution_is_experts_over_topk():
    """Two models at the same ridge differ only by E/k."""
    a = crossing_batch("mixtral-8x7b", H200_MEASURED)      # E/k = 4
    d = crossing_batch("deepseek-v3", H200_MEASURED)       # E/k = 32
    assert d / a == pytest.approx(8.0, rel=1e-9)


def test_rows_per_expert_is_the_saturated_ratio():
    assert rows_per_expert("deepseek-v3", 8192) == pytest.approx(8192 * 8 / 256)
    assert rows_per_expert("mixtral-8x7b", 1024) == pytest.approx(1024 * 2 / 8)


def test_compute_bound_agrees_with_the_crossing():
    cross = crossing_batch("deepseek-v3", H200_MEASURED)
    assert not is_compute_bound("deepseek-v3", int(cross) - 1, H200_MEASURED)
    assert is_compute_bound("deepseek-v3", int(cross) + 1, H200_MEASURED)


def test_it_refuses_nonsense_rather_than_returning_a_number():
    with pytest.raises(KeyError):
        crossing_batch("not-a-model", H200_MEASURED)
    with pytest.raises(ValueError):
        crossing_batch("deepseek-v3", 0.0)
    with pytest.raises(ValueError):
        crossing_batch("deepseek-v3", H200_MEASURED, dtype="not-a-dtype")


def test_saturation_is_stated_not_assumed_silently():
    """Below saturation not every expert is active, so R = T*k/E understates the
    rows each active expert holds and the formula does not apply. The helper must
    say so rather than quietly extrapolating."""
    from moe.bench.ridge import saturation_batch
    # deepseek needs enough tokens for all 256 experts to be touched at k=8.
    assert saturation_batch("deepseek-v3") > saturation_batch("mixtral-8x7b")
    # Every predicted crossing must sit above its own saturation point, or the
    # prediction is being made in a regime where the formula is invalid.
    for model in MEASURED_BRACKETS:
        assert crossing_batch(model, H200_MEASURED) > saturation_batch(model), model
