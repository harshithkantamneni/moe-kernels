"""In an fp8 cell the WEIGHTS are fp8 and the ACTIVATIONS are bf16.

MEASURED, H200 2026-08-27. 147 cells died inside vLLM on

    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

`fused_experts` takes activations at the compute dtype and quantises them
itself, because the activation scale depends on the values and is computed at
run time. Handing it fp8 activations skips the step the kernel owns.

That is also the right experiment. C2 says `AI = 2R/b` where R is rows per
expert and b is the WEIGHT dtype's width, because weights are `E*H*I` bytes and
activations are `T*H`: for mixtral at T=512 the weights outweigh the activations
by more than 300x. Halving the weight width is what moves the ridge crossing.
Halving the activation width would move nothing measurable and would change what
the kernel is asked to do.

So `spec.dtype` names the WEIGHT format. This file pins that, because the byte
model charged activations at the weight width and would have quietly reported an
arithmetic intensity built on 1-byte activations that were really 2.
"""
from __future__ import annotations

import pytest
import torch

from moe.bench.bytes_model import field_bytes, weight_bytes_for_stage
from moe.reference import torch_ref as R
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec, activation_dtype


def spec(dtype: str, tokens: int = 16, model: str = "toy"):
    return BenchSpec(MODEL_CONFIGS[model], num_tokens=tokens, dtype=dtype,
                     routing=RoutingSpec("uniform"), seed=0)


@pytest.mark.parametrize("dt", ["fp8_e4m3", "fp8_e5m2"])
def test_fp8_activations_are_bf16(dt):
    """The exact assertion 147 cells died on."""
    x, _ = R.make_inputs(spec(dt))
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), (
        f"vLLM's fused_experts rejects {x.dtype} activations")
    assert x.dtype is torch.bfloat16


@pytest.mark.parametrize("dt", ["fp8_e4m3", "fp8_e5m2"])
def test_the_weights_are_still_fp8(dt):
    """The half that must NOT change: fp8 weights are the whole point."""
    _, w = R.make_inputs(spec(dt))
    assert w.w1.dtype is not torch.bfloat16
    assert w.w1_scale is not None


def test_float_dtypes_are_unchanged():
    """Every published row is bf16. The fp8 path must not touch it."""
    for dt, want in [("bf16", torch.bfloat16), ("fp16", torch.float16),
                     ("fp32", torch.float32)]:
        assert activation_dtype(dt) == dt
        x, w = R.make_inputs(spec(dt))
        assert x.dtype is want and w.w1.dtype is want


def test_activation_dtype_maps_only_the_fp8_formats():
    assert activation_dtype("fp8_e4m3") == "bf16"
    assert activation_dtype("fp8_e5m2") == "bf16"
    assert activation_dtype("bf16") == "bf16"


# --- the byte model, which feeds the arithmetic-intensity column -------------

def test_activations_are_charged_two_bytes_in_an_fp8_cell():
    """The silent one. `field_bytes` used the weight width for activations, so
    an fp8 row would have reported activation traffic at half its real size."""
    assert field_bytes(spec("fp8_e4m3"))["x"] == field_bytes(spec("bf16"))["x"]


def test_weights_are_still_charged_one_byte_in_an_fp8_cell():
    """The other direction: if this halved too, fp8 would predict no shift at
    all and C2's test would be vacuous."""
    a = weight_bytes_for_stage(spec("fp8_e4m3"), "up_gemm", active_experts=4)
    b = weight_bytes_for_stage(spec("bf16"), "up_gemm", active_experts=4)
    assert a * 2 == b


def test_fp8_doubles_arithmetic_intensity_which_is_the_whole_prediction():
    """C2 end to end: same cell, one dtype changed, intensity must double."""
    from moe.bench.ridge import arithmetic_intensity
    lo = arithmetic_intensity("mixtral-8x7b", 512, "bf16")
    hi = arithmetic_intensity("mixtral-8x7b", 512, "fp8_e4m3")
    assert hi == pytest.approx(2 * lo, rel=0.02), (
        f"bf16 {lo:.1f} -> fp8 {hi:.1f}; C2 predicts exactly 2x")
