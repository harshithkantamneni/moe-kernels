"""SGLang's fp8 path: a THIRD independent kernel for the dtype-invariance claim.

WHY. The H200 sweep showed the ridge crossing does not move between bf16 and
fp8, measured on vLLM. One kernel makes that a fact about vLLM; three unrelated
kernels make it a fact about traffic, which is what C2 claims. SGLang is the
closest comparison of the three, covering the same five stages as vLLM, so if it
agrees then the 0.63x offset is not a property of that particular fusion.

WHAT THE API WANTS. Probed on SGLang 0.5.18, H200, 2026-08-28. Simpler than
vLLM's: `fused_experts` takes flat keyword arguments rather than a config
object, and `use_fp8_w8a8` is a plain bool.

    use_fp8_w8a8=True, w1_scale, w2_scale, per_channel_quant, block_shape
    a1_scale / a2_scale default to None, so SGLang quantises activations itself

Leaving the activation scales None is deliberate and matches vLLM: both
implementations then do their own activation quantisation inside the timed
region, so neither is charged for work the other avoids.
"""
from __future__ import annotations

import pytest

from moe.baselines._framework_config import sglang_quant_kwargs
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec


def spec(dtype: str):
    return BenchSpec(MODEL_CONFIGS["toy"], num_tokens=16, dtype=dtype,
                     routing=RoutingSpec("uniform"), seed=0)


def test_a_float_dtype_asks_for_no_quantisation():
    """bf16 is every published SGLang row. It must stay untouched."""
    assert sglang_quant_kwargs(spec("bf16")) == {}


def test_fp8_sets_the_w8a8_flag():
    kw = sglang_quant_kwargs(spec("fp8_e4m3"))
    assert kw["use_fp8_w8a8"] is True
    # The other quantisation flags must stay off, or fused_experts takes a
    # different kernel path entirely.
    for off in ("use_int8_w8a8", "use_int8_w8a16", "use_int4_w4a16"):
        assert kw.get(off, False) is False


def test_it_declares_per_tensor_scales_not_per_channel():
    """The harness quantises one scale per EXPERT. per_channel_quant=True would
    send the kernel looking for a scale per output channel and read shapes that
    do not exist."""
    assert sglang_quant_kwargs(spec("fp8_e4m3"))["per_channel_quant"] is False


def test_block_shape_is_stated_as_None_rather_than_left_out():
    """Block-wise scaling is a third layout again. Stating it keeps the call
    describing what this harness actually quantised."""
    kw = sglang_quant_kwargs(spec("fp8_e4m3"))
    assert "block_shape" in kw and kw["block_shape"] is None


def test_activation_scales_are_left_for_sglang_to_compute():
    """a1_scale/a2_scale=None means SGLang quantises activations itself, which
    is what vLLM does internally. Supplying them would charge the two
    implementations differently for the same work."""
    kw = sglang_quant_kwargs(spec("fp8_e4m3"))
    assert kw.get("a1_scale") is None
    assert kw.get("a2_scale") is None


def test_e5m2_is_refused_rather_than_silently_treated_as_e4m3():
    """SGLang's w8a8 path is built on e4m3. Passing e5m2 weights under an e4m3
    flag would run and compute a different layer."""
    with pytest.raises(ValueError, match="e4m3"):
        sglang_quant_kwargs(spec("fp8_e5m2"))


def test_the_span_declares_fp8_but_not_e5m2():
    from moe.baselines._framework_config import SGLANG_DTYPES
    assert "bf16" in SGLANG_DTYPES and "fp8_e4m3" in SGLANG_DTYPES
    assert "fp8_e5m2" not in SGLANG_DTYPES
