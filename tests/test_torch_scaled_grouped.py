"""torch's fp8 grouped GEMM, as a second independent test of C2.

WHY A SECOND IMPLEMENTATION. C2 says `AI = 2R/b`, so halving the weight width
halves the ridge crossing. Measured through vLLM alone, a 2x shift cannot be
told apart from a property of vLLM's kernel. Measured through two unrelated
kernels it is a statement about traffic, which is what the claim actually says.

This span is the cleanest of the three for that purpose: it covers ONE stage
(up_gemm or down_gemm), so the byte model applies directly, with no fusion or
permutation folded into the same measurement.

WHERE THE RISK IS. `_scaled_grouped_mm` needs BOTH operands in fp8, while the
harness hands out bf16 activations because vLLM asserts on anything else. So
this span quantises activations itself, inside the timed region. That is the
fair comparison rather than a thumb on the scale: vLLM's kernel quantises
activations internally too, so both implementations are charged for it.

The scale CONVENTION cannot be checked off-GPU, so it is not asserted here. If
`q * scale` is backwards the call still runs and returns the right shape, and
the harness's correctness gate against the fp32 oracle is what catches it.
"""
from __future__ import annotations

import pytest
import torch

from moe.baselines.torch_grouped_mm import scaled_grouped_args
from moe.quant import dequantize_per_expert


@pytest.fixture
def cell():
    torch.manual_seed(0)
    E, K, N, Ntot = 4, 32, 16, 10
    a = torch.randn(Ntot, K, dtype=torch.bfloat16)
    w = torch.randn(E, N, K) * (K ** -0.5)
    from moe.quant import quantize_per_expert
    wq, ws = quantize_per_expert(w, "fp8_e4m3")
    offs = torch.tensor([2, 5, 9, Ntot], dtype=torch.int32)
    return a, wq, ws, offs


def test_activations_are_quantised_to_fp8(cell):
    a, wq, ws, offs = cell
    args = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    assert args.a.dtype is torch.float8_e4m3fn, (
        "_scaled_grouped_mm needs both operands in fp8")


def test_there_is_one_activation_scale_per_ROW_not_per_tensor(cell):
    """Per-token scaling is what w8a8 means. A single tensor-wide scale would
    let one outlier token crush the resolution of every other one."""
    a, wq, ws, offs = cell
    args = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    assert args.scale_a.shape == (a.shape[0],)
    assert args.scale_a.dtype is torch.float32


def test_the_weight_scale_is_the_one_make_inputs_produced(cell):
    """Recomputing it here would silently measure a different layer from the
    one the oracle judges, since the oracle dequantises with ITS scale."""
    a, wq, ws, offs = cell
    args = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    assert torch.equal(args.scale_b, ws)


def test_the_weight_is_transposed_as_a_VIEW_not_copied(cell):
    """[E, N, K] -> [E, K, N] is a stride change. A copy would put a whole
    weight-sized allocation inside the timed region."""
    a, wq, ws, offs = cell
    args = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    assert args.b.shape == (wq.shape[0], wq.shape[2], wq.shape[1])
    assert args.b.data_ptr() == wq.data_ptr(), "transpose allocated"


def test_activation_quantisation_round_trips_within_fp8(cell):
    """If this drifts far the span is measuring a different layer, whatever the
    kernel does."""
    a, wq, ws, offs = cell
    args = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    back = dequantize_per_expert(args.a, args.scale_a)
    ref = a.float()
    rms = (back - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert rms < 0.12, f"activation round trip RMS {rms}"


def test_offsets_pass_through_unchanged(cell):
    a, wq, ws, offs = cell
    args = scaled_grouped_args(a, wq, ws, offs, "fp8_e4m3")
    assert torch.equal(args.offs, offs)
    assert args.offs.dtype is torch.int32


def test_the_spans_declare_fp8_and_are_registered():
    import moe
    from moe.stages import registry
    moe.bootstrap("baselines")
    reg = registry()
    for name in ("torch_scaled_grouped_mm_up", "torch_scaled_grouped_mm_down"):
        assert name in reg, f"{name} not registered"
        assert "fp8_e4m3" in reg[name].dtypes
        assert "bf16" not in reg[name].dtypes, (
            "the bf16 path is torch_grouped_mm_*; one span per dtype keeps the "
            "impl column meaning one thing")


def test_the_bf16_span_is_untouched():
    """Every published row came from these. They must not gain an fp8 dtype."""
    import moe
    from moe.stages import registry
    moe.bootstrap("baselines")
    reg = registry()
    assert reg["torch_grouped_mm_up"].dtypes == ("bf16",)
