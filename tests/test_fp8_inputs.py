"""fp8 weights, their scales, and the oracle that has to judge them.

The point of an fp8 sweep is C2's prediction: `AI = 2R/b`, so halving bytes must
halve the ridge crossing. deepseek-v3 from ~5,100 tokens to ~2,570. That is a 2x
shift, which the existing powers-of-two token grid separates without any custom
runs.

THE SUBTLE PART IS THE ORACLE, not the quantisation. If the reference computes
from the ORIGINAL bf16 weights while the kernel computes from fp8 ones, the
correctness gate measures quantisation error: a property of the format, identical
for every implementation, and about 2.6% RMS. That would either fail every fp8
cell or force a tolerance so wide it stops catching real kernel bugs.

So the oracle computes from the DEQUANTISED weights. Both sides then compute the
same mathematical function and differ only in arithmetic precision, which is what
the gate exists to measure.
"""
from __future__ import annotations

import pytest
import torch

from moe.quant import dequantize_per_expert
from moe.reference import torch_ref as R
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec


def spec(dtype: str, tokens: int = 16):
    return BenchSpec(MODEL_CONFIGS["toy"], num_tokens=tokens, dtype=dtype,
                     routing=RoutingSpec("uniform"), seed=0)


@pytest.mark.parametrize("dtype", ["fp8_e4m3", "fp8_e5m2"])
def test_fp8_weights_come_out_quantised_with_scales(dtype):
    _, w = R.make_inputs(spec(dtype))
    assert w.w1.dtype == getattr(torch, {"fp8_e4m3": "float8_e4m3fn",
                                         "fp8_e5m2": "float8_e5m2"}[dtype])
    assert w.w2.dtype == w.w1.dtype
    assert w.w1_scale is not None and w.w2_scale is not None
    assert w.w1_scale.shape == (MODEL_CONFIGS["toy"].num_experts,)
    assert w.w1_scale.dtype == torch.float32


def test_the_router_gate_stays_fp32():
    """Routing decides WHICH experts run. Quantising the gate would change the
    experiment rather than the arithmetic under test."""
    _, w = R.make_inputs(spec("fp8_e4m3"))
    assert w.wg.dtype == torch.float32


def test_bf16_is_untouched_and_carries_no_scales():
    """The fp8 path must not disturb the dtype every published row used."""
    _, w = R.make_inputs(spec("bf16"))
    assert w.w1.dtype == torch.bfloat16
    assert w.w1_scale is None and w.w2_scale is None


def test_dequantised_weights_are_close_to_what_bf16_would_have_held():
    """Same seed, same generator: fp8 weights are the bf16 draw, quantised. If
    they were a different draw the two dtypes would be different experiments."""
    _, bf16 = R.make_inputs(spec("bf16"))
    _, fp8 = R.make_inputs(spec("fp8_e4m3"))
    back = dequantize_per_expert(fp8.w1, fp8.w1_scale)
    ref = bf16.w1.float()
    rms = (back - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert rms < 0.10, f"RMS relative {rms}: not the same draw"
    assert rms > 0.0, "identical would mean no quantisation happened"


def test_the_oracle_judges_against_dequantised_weights(monkeypatch):
    """The whole design. golden_forward must see `q * scale`, not the original
    draw, or the gate measures the format instead of the kernel."""
    x, w = R.make_inputs(spec("fp8_e4m3"))
    seen = {}
    real = R.grouped_gemm_loop

    def spy(a, b, offsets, n):
        seen.setdefault("w1_dtype", b.dtype)
        return real(a, b.float() if b.dtype.is_floating_point else b, offsets, n)

    monkeypatch.setattr(R, "grouped_gemm_loop", spy)
    R.golden_forward(spec("fp8_e4m3"), w, x)
    assert seen["w1_dtype"] not in (torch.float8_e4m3fn, torch.float8_e5m2), (
        "the oracle was handed raw fp8; it must receive dequantised weights")


def test_the_fp8_oracle_agrees_with_an_explicit_dequantised_run():
    """Belt and braces: compute the golden output twice, once through the fp8
    path and once by dequantising by hand into a bf16 weight set. Identical
    means the oracle is doing exactly what this file claims."""
    s = spec("fp8_e4m3")
    x, w = R.make_inputs(s)
    from moe.state import MoEWeights
    manual = MoEWeights(w1=dequantize_per_expert(w.w1, w.w1_scale),
                        w2=dequantize_per_expert(w.w2, w.w2_scale),
                        wg=w.wg)
    a = R.golden_forward(s, w, x)
    b = R.golden_forward(s, manual, x)
    assert torch.allclose(a, b, rtol=0, atol=0), "oracle is not dequantising"


def test_validate_still_checks_shapes_for_fp8():
    s = spec("fp8_e4m3")
    _, w = R.make_inputs(s)
    w.validate(s)
    w.w1 = w.w1[:-1]
    with pytest.raises(ValueError, match="w1"):
        w.validate(s)


def test_vllm_kwargs_ask_for_an_fp8_quant_config():
    """The framework-free half, testable without vLLM installed."""
    from moe.baselines._framework_config import vllm_quant_spec
    assert vllm_quant_spec(spec("bf16")) is None
    q = vllm_quant_spec(spec("fp8_e4m3"))
    assert q is not None
    assert q["kind"] == "fp8_w8a8"
    # Per-tensor per-expert: no block shape, no per-token activation scaling.
    assert q["per_act_token_quant"] is False
    assert q["block_shape"] is None
