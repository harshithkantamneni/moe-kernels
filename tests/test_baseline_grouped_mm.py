"""torch.nn.functional.grouped_mm as a baseline, checked against the oracle.

It ships inside torch 2.13.0 and runs on CPU, so unlike the vLLM and SGLang
baselines this one is fully verifiable before renting anything.
"""
import pytest
import torch

import moe
from moe import pipeline as P
from moe.bench import driver as D
from moe.reference import torch_ref as R
from moe.spec import MODEL_CONFIGS, BenchSpec, MoEConfig, RoutingSpec
from moe.stages import get
from moe.state import MoEState

moe.bootstrap("baselines")

pytestmark = pytest.mark.skipif(
    not hasattr(torch.nn.functional, "grouped_mm"),
    reason="this torch has no grouped_mm")

# Deliberately awkward: non-power-of-two F, group sizes that will not be
# multiples of any tile size, and enough experts to leave some empty.
GEOM = MoEConfig(name="gmm", hidden_size=128, intermediate_size=176,
                 num_experts=8, top_k=2, num_layers=1, verified=True)


def cell(routing="zipf", param=1.4, tokens=48):
    return BenchSpec(GEOM, tokens, "bf16", RoutingSpec(routing, param), seed=0)


def names(*impls):
    order = ["ref_router", "ref_permute", "ref_up_gemm", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    for impl in impls:
        stage = get(impl).covers[0]
        order[["router", "permute", "up_gemm", "act", "down_gemm",
               "unpermute"].index(stage)] = impl
    return order


def test_both_spans_are_registered():
    assert get("torch_grouped_mm_up").covers == ("up_gemm",)
    assert get("torch_grouped_mm_down").covers == ("down_gemm",)


def test_offsets_conversion_matches_the_csr_convention():
    """grouped_mm wants exclusive END offsets [E]; the harness carries CSR
    [E+1] starting at 0."""
    from moe.baselines.torch_grouped_mm import _offs
    st = MoEState(spec=cell(), weights=None)
    st.expert_offsets = torch.tensor([0, 3, 3, 7, 10], dtype=torch.int32)
    assert _offs(st).tolist() == [3, 3, 7, 10]
    assert _offs(st).dtype == torch.int32


@pytest.mark.parametrize("impl", ["torch_grouped_mm_up", "torch_grouped_mm_down"])
@pytest.mark.parametrize("routing,param", [("uniform", 0.0), ("zipf", 1.4),
                                           ("hot", 0.7)])
def test_matches_the_oracle_under_skew(impl, routing, param):
    """The whole point of a baseline: it has to pass the same gate a kernel does."""
    spec = cell(routing, param)
    x, weights = R.make_inputs(spec)
    cfg = D.RunConfig(device="cpu")
    result, _, _, _ = D.check_correctness(
        spec, P.build(names(impl), spec=spec), x, weights, None, cfg)
    assert result.passed, (f"{impl} under {routing}:{param} "
                           f"rel={result.rel_err:.3e} tol={result.tol_rel_max:.3e}")


def test_both_projections_together_match_the_oracle():
    spec = cell()
    x, weights = R.make_inputs(spec)
    cfg = D.RunConfig(device="cpu")
    pipe = P.build(names("torch_grouped_mm_up", "torch_grouped_mm_down"), spec=spec)
    result, _, _, _ = D.check_correctness(spec, pipe, x, weights, None, cfg)
    assert result.passed, f"rel={result.rel_err:.3e}"


def test_empty_experts_are_handled():
    """Under real skew most experts receive nothing, so a zero-length group has
    to be a normal case rather than an edge case."""
    spec = cell(tokens=32)
    forced = torch.zeros((spec.num_tokens, spec.model.top_k), dtype=torch.int32)
    forced[:, 1] = 1                      # only experts 0 and 1 receive rows
    x, weights = R.make_inputs(spec)
    cfg = D.RunConfig(device="cpu")
    pipe = P.build(names("torch_grouped_mm_up", "torch_grouped_mm_down"), spec=spec)
    result, st, _, _ = D.check_correctness(spec, pipe, x, weights, forced, cfg)
    from moe.routing.imbalance import counts_from_offsets
    counts = counts_from_offsets(st.expert_offsets)
    assert counts[2:] == [0] * (spec.model.num_experts - 2)
    assert result.passed, f"rel={result.rel_err:.3e}"


def test_the_weight_transpose_is_a_view_not_a_copy():
    """It sits inside the timed region, so it must not move data."""
    _, weights = R.make_inputs(cell())
    t = weights.w1.transpose(1, 2)
    assert t.shape == (GEOM.num_experts, GEOM.hidden_size,
                       2 * GEOM.intermediate_size)
    assert t.data_ptr() == weights.w1.data_ptr()
    assert not t.is_contiguous()


def test_group_count_limit_is_declared():
    """CUTLASS grouped GEMM caps group_count; a 1024-expert model must be
    reported unsupported rather than failing on the box."""
    big = MoEConfig(name="huge", hidden_size=64, intermediate_size=64,
                    num_experts=2048, top_k=2, verified=True)
    span = get("torch_grouped_mm_up")
    assert not span.supports(BenchSpec(big, 8, "bf16"))
    assert span.supports(BenchSpec(MODEL_CONFIGS["deepseek-v3"], 8, "bf16"))


# --- the incumbent does not exist on every architecture -----------------------

def test_grouped_mm_support_is_reported_per_architecture():
    """torch's grouped_mm dispatches to CUTLASS `bf16bf16_grouped_gemm_impl_
    sm90_sm100`. On anything outside that range the baseline is not the kernel
    the published numbers measured, and a sweep that silently timed a fallback
    would compare a hand-written kernel against a different incumbent."""
    from moe.baselines.torch_grouped_mm import grouped_mm_support

    assert grouped_mm_support((9, 0)).supported       # H100 / H200
    assert grouped_mm_support((10, 0)).supported      # B200
    assert not grouped_mm_support((8, 0)).supported   # A100
    assert not grouped_mm_support((12, 0)).supported  # RTX PRO 6000


def test_unsupported_architecture_explains_itself():
    """The reason is recorded so a row says why the baseline is absent, rather
    than the baseline quietly not appearing in the registry."""
    from moe.baselines.torch_grouped_mm import grouped_mm_support

    why = grouped_mm_support((8, 0)).reason
    assert "sm_80" in why
    assert "sm_90" in why and "sm_100" in why


def test_supported_architecture_has_no_complaint():
    from moe.baselines.torch_grouped_mm import grouped_mm_support

    assert grouped_mm_support((9, 0)).reason == ""
