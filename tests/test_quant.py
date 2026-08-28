"""Per-expert fp8 quantisation, in the form vLLM's fused_moe expects.

WHY fp8 IS IN THIS STUDY. Claim C2 says arithmetic intensity is `2R/b`. Halving
`b` must therefore DOUBLE intensity and HALVE the batch at which a model crosses
its ridge: deepseek-v3 from ~5,100 tokens to ~2,570. That is a 2x prediction,
which the existing powers-of-two token grid resolves trivially, where the A100's
1.1x ridge difference lands both predictions inside a single bin and resolves
nothing.

CONVENTION. `fp8_w8a8_moe_quant_config` requires `w1_scale` and `w2_scale`, so
the contract is `q.float() * scale == w`, one scale per expert. Not chosen for
accuracy, chosen to match what the kernel will do with it. With realistic
fan-in weights it is also marginally the better of the two options, since 27.6%
of elements would be subnormal in fp8 if left unscaled.
"""
from __future__ import annotations

import pytest
import torch

from moe.quant import (
    FP8_DTYPES,
    dequantize_per_expert,
    quantize_per_expert,
    round_trip_error,
)


@pytest.fixture
def weights():
    """Fan-in scaled, the shape make_inputs produces."""
    torch.manual_seed(0)
    return torch.randn(4, 256, 128) * (128 ** -0.5)


@pytest.mark.parametrize("dtype", sorted(FP8_DTYPES))
def test_round_trip_reconstructs_within_fp8_precision(weights, dtype):
    q, scale = quantize_per_expert(weights, dtype)
    back = dequantize_per_expert(q, scale)
    assert back.shape == weights.shape
    err = round_trip_error(weights, back)
    # e4m3 has 3 mantissa bits and e5m2 has 2, so a few percent is the floor.
    assert err < 0.12, f"{dtype}: RMS relative error {err}"
    assert err > 0.0, "lossless would mean the cast did not happen"


def test_the_contract_is_q_times_scale(weights):
    """vLLM multiplies the fp8 weight by the scale. If our convention were the
    reciprocal every fp8 cell would compute a different layer and still run."""
    q, scale = quantize_per_expert(weights, "fp8_e4m3")
    manual = q.float() * scale.reshape(-1, *([1] * (weights.ndim - 1)))
    assert torch.equal(manual, dequantize_per_expert(q, scale))


def test_one_scale_per_expert(weights):
    q, scale = quantize_per_expert(weights, "fp8_e4m3")
    assert scale.shape == (weights.shape[0],)
    assert scale.dtype == torch.float32
    assert (scale > 0).all()


def test_it_uses_the_fp8_range_rather_than_a_corner_of_it(weights):
    """The point of scaling: the largest element of each expert should land near
    the format's maximum, not at 0.2 where most of the exponent range is idle."""
    q, _ = quantize_per_expert(weights, "fp8_e4m3")
    peak = q.float().abs().amax(dim=(1, 2))
    top = torch.finfo(torch.float8_e4m3fn).max
    assert (peak > 0.5 * top).all(), f"peaks {peak.tolist()} against max {top}"


def test_an_all_zero_expert_does_not_divide_by_zero():
    w = torch.zeros(3, 8, 8)
    w[1] = torch.randn(8, 8)
    q, scale = quantize_per_expert(w, "fp8_e4m3")
    assert torch.isfinite(scale).all()
    assert (scale > 0).all()
    assert torch.equal(dequantize_per_expert(q, scale)[0], torch.zeros(8, 8))


def test_quantisation_is_deterministic(weights):
    a, sa = quantize_per_expert(weights, "fp8_e4m3")
    b, sb = quantize_per_expert(weights, "fp8_e4m3")
    assert torch.equal(a.float(), b.float()) and torch.equal(sa, sb)


def test_a_non_fp8_dtype_is_refused(weights):
    with pytest.raises(ValueError):
        quantize_per_expert(weights, "bf16")
    with pytest.raises(ValueError):
        quantize_per_expert(weights, "not-a-dtype")


def test_the_tolerance_model_knows_fp8():
    """`tolerance()` raised for any dtype not in its table, so an fp8 sweep
    would have died on the first cell rather than run with a wrong budget."""
    from moe.bench.tolerance import tolerance
    from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
    for dt in sorted(FP8_DTYPES):
        spec = BenchSpec(MODEL_CONFIGS["toy"], num_tokens=8, dtype=dt,
                         routing=RoutingSpec("uniform"), seed=0)
        tol = tolerance(spec)
        assert tol.rel_max > 0
        # fp8 must be looser than bf16, or the budget is not modelling the format
        bf16 = tolerance(BenchSpec(MODEL_CONFIGS["toy"], num_tokens=8,
                                   dtype="bf16", routing=RoutingSpec("uniform"),
                                   seed=0))
        assert tol.rel_max > bf16.rel_max, f"{dt} tolerance {tol.rel_max} <= bf16"
