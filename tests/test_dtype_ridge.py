"""The ridge is dtype-specific, and the crossing is therefore dtype-INVARIANT.

THE PREDICTION THIS FILE CORRECTS. The fp8 sweep was run to test a "2x crossing
shift": halve the weight bytes, halve the batch at which a model crosses its
ridge. That prediction was wrong, and wrong in my framing rather than in C2.

`crossing where AI = ridge`, with `AI = 2R/b` and `ridge = peak_FLOPS / bandwidth`.
Going bf16 -> fp8 halves `b` AND doubles `peak_FLOPS`, because the same silicon
runs fp8 tensor cores at twice the bf16 rate. So:

    fp8:   2R/1 = 2 * ridge_bf16   ->   R = ridge_bf16
    bf16:  2R/2 =     ridge_bf16   ->   R = ridge_bf16

The same rows-per-expert, so the same crossing batch. Holding the ridge fixed at
its bf16 value while changing the format is what produced the 2x figure, and the
H200 measurement agrees with the corrected version: deepseek-v2-lite crossed at
922 tokens in bf16 and 976 in fp8, not 1710 and 855.

MEASURED, H200 2026-08-28, mixtral at T=512: 1.1567 ms bf16 -> 0.6383 ms fp8,
0.55x. The times halve. Both SIDES of the roofline halve, so their intersection
does not move.
"""
from __future__ import annotations

import pytest

from moe.bench.ridge import crossing_batch, ridge_for_dtype


def test_fp8_doubles_the_ridge():
    """Same bandwidth, twice the FLOP rate."""
    assert ridge_for_dtype(160.3, "fp8_e4m3") == pytest.approx(320.6)
    assert ridge_for_dtype(160.3, "fp8_e5m2") == pytest.approx(320.6)


def test_bf16_and_fp16_are_the_reference_rate():
    for dt in ("bf16", "fp16"):
        assert ridge_for_dtype(160.3, dt) == pytest.approx(160.3)


def test_fp32_is_half_the_rate():
    """The other direction, so the scaling is a rule rather than an fp8 patch."""
    assert ridge_for_dtype(160.3, "fp32") == pytest.approx(80.15)


@pytest.mark.parametrize("model", ["mixtral-8x7b", "qwen2-57b-a14b",
                                   "deepseek-v2-lite", "deepseek-v3"])
def test_the_crossing_is_the_SAME_in_bf16_and_fp8(model):
    """The corrected prediction, and the one the sweep actually tested."""
    bf16 = crossing_batch(model, ridge_for_dtype(160.3, "bf16"), "bf16")
    fp8 = crossing_batch(model, ridge_for_dtype(160.3, "fp8_e4m3"), "fp8_e4m3")
    assert fp8 == pytest.approx(bf16, rel=1e-9), (
        f"{model}: bf16 {bf16:.0f} vs fp8 {fp8:.0f}; both sides of the roofline "
        "scale together, so the intersection must not move")


def test_the_old_two_x_prediction_is_what_a_FIXED_ridge_produces():
    """Pinning the bug itself, so nobody reintroduces it as a simplification."""
    naive = crossing_batch("mixtral-8x7b", 160.3, "fp8_e4m3")
    correct = crossing_batch("mixtral-8x7b", ridge_for_dtype(160.3, "fp8_e4m3"),
                             "fp8_e4m3")
    assert correct == pytest.approx(2 * naive)


def test_an_unknown_dtype_is_refused_rather_than_assumed_to_be_bf16():
    with pytest.raises(ValueError):
        ridge_for_dtype(160.3, "int4")
