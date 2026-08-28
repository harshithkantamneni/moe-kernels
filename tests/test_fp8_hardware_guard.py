"""An fp8 row must have touched fp8 silicon.

fp8 tensor cores arrived with Ada (sm_89) and Hopper (sm_90). The A100 is
sm_80 and has none. Nothing stopped `--profile fp8 --envs vllm` from running
there: the span declared fp8_e4m3 with no architecture check, so vLLM would
accept the cell, most likely dequantise to bf16 and run a bf16 GEMM, and write
rows labelled fp8_e4m3 that never used an fp8 unit. In a merged CSV they would
be indistinguishable from the H200's.

This is the same failure `grouped_mm_support` already guards, in that guard's
own words: "a sweep on an A100 would time whatever torch falls back to and write
it under the same impl name as the published H200 rows, which is the silent
substitution this harness exists to refuse."

The reference spans keep fp8 deliberately. They dequantise and compute in fp32,
which is correct on any card and is the oracle's job. An fp8 sweep on an A100
should therefore plan zero MEASURED cells, which is the honest answer: that card
cannot do the experiment.
"""
from __future__ import annotations

import pytest

from moe.quant import fp8_hardware_support


def test_hopper_and_ada_have_fp8():
    assert fp8_hardware_support((9, 0)).supported
    assert fp8_hardware_support((10, 0)).supported
    assert fp8_hardware_support((8, 9)).supported


def test_ampere_does_not_and_says_which_card():
    v = fp8_hardware_support((8, 0))
    assert not v.supported
    assert "sm_80" in v.reason
    assert "fp8" in v.reason.lower()


def test_older_architectures_do_not_either():
    assert not fp8_hardware_support((7, 5)).supported
    assert not fp8_hardware_support((7, 0)).supported


def test_no_device_is_not_the_same_as_unsupported():
    """--dry-run builds the whole matrix on a laptop. Returning "unsupported"
    with no device would silently empty the plan, which is how the grouped_mm
    guard is written and for the same reason."""
    assert fp8_hardware_support(None) is None


@pytest.mark.parametrize("span", ["vllm_fused_experts", "sglang_fused_experts"])
def test_the_framework_spans_consult_the_guard(span, monkeypatch):
    """Both declare fp8_e4m3; neither may accept it on hardware without it."""
    import moe.quant as Q
    from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec

    monkeypatch.setattr(Q, "fp8_hardware_support",
                        lambda cap=None: Q.Fp8Support(False, "sm_80 has no fp8"))
    # Import lazily: neither framework is installed on a laptop, so this checks
    # the shared predicate the spans are built on rather than the spans.
    spec = BenchSpec(MODEL_CONFIGS["toy"], num_tokens=8, dtype="fp8_e4m3",
                     routing=RoutingSpec("uniform"), seed=0)
    assert Q.fp8_cell_supported(spec) is False
    bf16 = BenchSpec(MODEL_CONFIGS["toy"], num_tokens=8, dtype="bf16",
                     routing=RoutingSpec("uniform"), seed=0)
    assert Q.fp8_cell_supported(bf16) is True, "bf16 must be unaffected"


def test_a_missing_device_leaves_the_cell_supported():
    """On a laptop the plan must still build."""
    import moe.quant as Q
    from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
    spec = BenchSpec(MODEL_CONFIGS["toy"], num_tokens=8, dtype="fp8_e4m3",
                     routing=RoutingSpec("uniform"), seed=0)
    assert Q.fp8_cell_supported(spec) is True
