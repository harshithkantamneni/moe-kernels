import pytest

from moe.bench.tolerance import tolerance
from moe.spec import MODEL_CONFIGS, BenchSpec


def spec(model="mixtral-8x7b", dtype="bf16"):
    return BenchSpec(MODEL_CONFIGS[model], num_tokens=128, dtype=dtype)


def test_lower_precision_gets_a_looser_budget():
    assert (tolerance(spec(dtype="fp32")).rtol
            < tolerance(spec(dtype="fp16")).rtol
            < tolerance(spec(dtype="bf16")).rtol)


def test_bigger_geometry_gets_a_looser_budget():
    small = tolerance(spec("toy")).rtol
    big = tolerance(spec("deepseek-v3")).rtol
    assert big > small


def test_uncalibrated_is_flagged():
    t = tolerance(spec())
    assert t.calibrated is False
    assert t.basis == "analytic"


def test_measured_calibration_overrides_and_is_flagged():
    cal = {"bf16": {"atol": 1e-3, "rtol": 2e-3}}
    t = tolerance(spec(dtype="bf16"), calibration=cal)
    assert (t.atol, t.rtol) == (1e-3, 2e-3)
    assert t.calibrated is True and t.basis == "measured"


def test_unknown_dtype_is_an_error():
    with pytest.raises(ValueError, match="no tolerance model"):
        tolerance(spec(dtype="fp8_e4m3"))
