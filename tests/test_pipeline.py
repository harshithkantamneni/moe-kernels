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
class _FusedDownScatter(_Noop):
    name = "t_fused_down_scatter"
    covers = ("down_gemm", "unpermute")


@register
class _FusedUpAct(_Noop):
    name = "t_fused_up_act"
    covers = ("up_gemm", "act")


@register
class _FusedPermuteUp(_Noop):
    name = "t_fused_permute_up"
    covers = ("permute", "up_gemm")


@register
class _FusedPermuteUpMat(_Noop):
    name = "t_fused_permute_up_materialising"
    covers = ("permute", "up_gemm")
    materialises = ("expert_offsets", "perm_index")


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
             "t_fused_down_scatter"]
    pipe = P.build(names)
    assert len(pipe.spans) == 5


def test_two_fusions_compose():
    names = ["ref_router", "ref_permute", "t_fused_up_act", "t_fused_down_scatter"]
    assert len(P.build(names).spans) == 4


def test_missing_stage_names_the_gap():
    with pytest.raises(P.PipelineError, match=r"uncovered.*unpermute"):
        P.build(REF[:-1])


def test_overlapping_spans_named():
    names = ["ref_router", "ref_permute", "ref_up_gemm", "ref_act",
             "t_fused_down_scatter", "ref_unpermute"]
    with pytest.raises(P.PipelineError, match="re-covers"):
        P.build(names)


def test_out_of_order_tiling_rejected():
    names = ["ref_permute", "ref_router", "ref_up_gemm", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    with pytest.raises(P.PipelineError, match="out of canonical order"):
        P.build(names)


def test_fused_span_that_hides_a_needed_field_is_rejected():
    # A fused permute+up_gemm computes expert_offsets internally. If it does not
    # materialise them, the down GEMM later has no group boundaries to read.
    # This is a real design mistake, and it must surface at build time.
    names = ["ref_router", "t_fused_permute_up", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    with pytest.raises(P.PipelineError, match=r"ref_down_gemm reads \['expert_offsets'\]"):
        P.build(names)


def test_materialises_declaration_fixes_it():
    names = ["ref_router", "t_fused_permute_up_materialising", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    pipe = P.build(names)
    assert len(pipe.spans) == 5


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
