import pytest
import torch

from moe.bench.tolerance import relative_error, tolerance
from moe.spec import MODEL_CONFIGS, BenchSpec


def spec(model="mixtral-8x7b", dtype="bf16"):
    return BenchSpec(MODEL_CONFIGS[model], num_tokens=128, dtype=dtype)


def test_lower_precision_gets_a_looser_budget():
    assert (tolerance(spec(dtype="fp32")).rel_max
            < tolerance(spec(dtype="fp16")).rel_max
            < tolerance(spec(dtype="bf16")).rel_max)


def test_budget_is_dominated_by_quantisation_not_by_geometry():
    """The old model scaled the budget by sqrt(K), which loosened tolerance as
    geometry grew. Tensor cores accumulate in fp32, so the dominant term is the
    one-time dtype quantisation and is K-independent."""
    small = tolerance(spec("toy")).rel_max
    big = tolerance(spec("deepseek-v3")).rel_max
    assert big == pytest.approx(small, rel=0.05)


def test_fp32_budget_does_grow_with_geometry():
    """At fp32 there is no quantisation term to dominate, so the fp32
    accumulation term is visible and geometry does matter."""
    assert (tolerance(spec("deepseek-v3", "fp32")).rel_max
            > tolerance(spec("toy", "fp32")).rel_max)


def test_bf16_budget_is_a_few_percent_not_a_few_hundred_percent():
    """Regression: the previous model produced rel tolerances above 2.0, which
    admitted sign-flipped and 3x-scaled outputs."""
    for name in MODEL_CONFIGS:
        t = tolerance(spec(name, "bf16"))
        assert t.rel_max < 0.05, f"{name}: rel_max {t.rel_max} is far too loose"


def test_uncalibrated_is_flagged():
    t = tolerance(spec())
    assert t.calibrated is False and t.basis == "analytic"


def test_measured_calibration_overrides_and_is_flagged():
    t = tolerance(spec(dtype="bf16"), calibration={"bf16": {"rel_max": 1e-3}})
    assert t.rel_max == 1e-3 and t.calibrated is True and t.basis == "measured"


def test_unknown_dtype_is_an_error():
    with pytest.raises(ValueError, match="no tolerance model"):
        tolerance(spec(dtype="fp8_e4m3"))


# --- the metric itself ------------------------------------------------------

def test_relative_error_is_scale_free():
    ref = torch.randn(64, 32)
    got = ref * 1.01
    small = relative_error(got * 1e-6, ref * 1e-6)
    large = relative_error(got * 1e6, ref * 1e6)
    assert small == pytest.approx(large, rel=1e-4)


def test_zeroed_output_scores_one():
    ref = torch.randn(16, 8)
    assert relative_error(torch.zeros_like(ref), ref) == pytest.approx(1.0)


def test_sign_flip_scores_two():
    ref = torch.randn(16, 8)
    assert relative_error(-ref, ref) == pytest.approx(2.0)


def test_exact_match_scores_zero():
    ref = torch.randn(16, 8)
    assert relative_error(ref.clone(), ref) == 0.0


def test_all_zero_reference_is_handled():
    z = torch.zeros(4, 4)
    assert relative_error(z, z) == 0.0
    assert relative_error(torch.ones(4, 4), z) == float("inf")
