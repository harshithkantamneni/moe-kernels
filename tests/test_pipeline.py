import pytest

from moe import pipeline as P
from moe.stages import StageSpan, register
from moe.state import MoEState


class _Noop(StageSpan):
    requires_cuda = False
    dtypes = ("fp32", "bf16")

    def __call__(self, st: MoEState) -> None:  # pragma: no cover - never run
        pass


@register
class _FusedPermuteUp(_Noop):
    name = "t_fused_permute_up"
    covers = ("permute", "up_gemm")


@register
class _VllmUpGemm(_Noop):
    name = "t_vllm_up_gemm"
    covers = ("up_gemm",)
    env = "vllm"


@register
class _SglangDownGemm(_Noop):
    name = "t_sglang_down_gemm"
    covers = ("down_gemm",)
    env = "sglang"


@register
class _Bf16Only(_Noop):
    name = "t_bf16_only_up"
    covers = ("up_gemm",)
    dtypes = ("bf16",)


REF = P.reference_pipeline_names()


def test_all_reference_tiling_is_valid():
    pipe = P.build(REF)
    assert pipe.env == "base"
    assert pipe.requires_cuda is False
    assert pipe.label.startswith("ref_router -> ref_permute")


def test_fused_tiling_is_valid_and_shorter():
    names = ["ref_router", "ref_permute", "ref_up_gemm", "ref_act",
             "ref_fused_down_scatter"]
    pipe = P.build(names)
    assert len(pipe.spans) == 5


def test_two_fusions_compose():
    names = ["ref_router", "ref_permute", "ref_fused_up_act", "ref_fused_down_scatter"]
    assert len(P.build(names).spans) == 4


def test_missing_stage_names_the_gap():
    with pytest.raises(P.PipelineError, match=r"uncovered.*unpermute"):
        P.build(REF[:-1])


def test_overlapping_spans_named():
    names = ["ref_router", "ref_permute", "ref_up_gemm", "ref_act",
             "ref_fused_down_scatter", "ref_unpermute"]
    with pytest.raises(P.PipelineError, match="re-covers"):
        P.build(names)


def test_out_of_order_tiling_rejected():
    names = ["ref_permute", "ref_router", "ref_up_gemm", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    with pytest.raises(P.PipelineError, match="out of canonical order"):
        P.build(names)


def test_fused_span_gets_its_live_outputs_computed_for_it():
    """A fused permute+up_gemm computes expert_offsets and perm_index
    internally and must expose both, because down_gemm and unpermute read them.
    That is derivable from the stage graph, so the author does not declare it:
    an earlier design required a `materialises` tuple and rejected the tiling
    when it was forgotten, which punished a correct kernel."""
    names = ["ref_router", "t_fused_permute_up", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    pipe = P.build(names)
    live = pipe.materialised_for(P.S.get("t_fused_permute_up"))
    assert live == {"expert_offsets", "perm_index", "h_up"}
    # x_perm is consumed inside the span, so it never reaches memory.
    assert "x_perm" not in live


def test_a_swallowed_intermediate_is_not_materialised():
    pipe = P.build(["ref_router", "ref_permute", "ref_fused_up_act",
                    "ref_down_gemm", "ref_unpermute"])
    live = pipe.materialised_for(P.S.get("ref_fused_up_act"))
    assert live == {"h_act"}
    assert "h_up" not in live


def test_the_layer_output_is_always_materialised():
    """No stage reads `y`, but the harness does, so liveness must keep it."""
    pipe = P.build(REF)
    assert pipe.materialised_for(P.S.get("ref_unpermute")) == {"y"}


def test_declared_extra_materialisation_is_charged_but_not_available():
    """ref_router stores logits nothing reads. That is traffic, not
    availability: the bytes model charges for the store, and liveness alone
    would not have."""
    pipe = P.build(REF)
    router = P.S.get("ref_router")
    assert "router_logits" in pipe.materialised_for(router)
    assert "router_logits" not in P.S.live_outputs(pipe.spans)[0]


def test_a_span_that_cannot_materialise_a_needed_field_is_rejected():
    """The opt-out for a kernel that physically fuses a field into registers.
    The tiling is genuinely invalid, and it says so by name at build time."""

    @register
    class _FusesOffsetsAway(_Noop):
        name = "t_fuses_offsets_away"
        covers = ("permute", "up_gemm")
        cannot_materialise = ("expert_offsets",)

    names = ["ref_router", "t_fuses_offsets_away", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    with pytest.raises(P.PipelineError, match=r"needs \['expert_offsets'\]"):
        P.build(names)


def test_mixed_framework_envs_rejected():
    names = ["ref_router", "ref_permute", "t_vllm_up_gemm", "ref_act",
             "t_sglang_down_gemm", "ref_unpermute"]
    with pytest.raises(P.PipelineError, match="mixes incompatible environments"):
        P.build(names)


def test_single_framework_env_propagates_to_pipeline():
    names = ["ref_router", "ref_permute", "t_vllm_up_gemm", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    assert P.build(names).env == "vllm"


def test_unsupported_dtype_rejected_when_spec_given(toy_spec):
    names = ["ref_router", "ref_permute", "t_bf16_only_up", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    P.build(names)  # fine without a spec
    with pytest.raises(P.PipelineError, match="does not support"):
        P.build(names, spec=toy_spec)  # toy_spec is fp32
