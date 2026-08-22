"""Shared fixtures and reference fusions used across the suite.

The two fused spans below are real torch implementations, not stubs. They exist
so that the fusion accounting in bytes_model and the numerical equivalence of
fused vs unfused tilings can both be tested on a laptop, without CUDA.
"""
import pytest

import moe
from moe.reference import torch_ref as R
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
from moe.stages import StageSpan, register
from moe.state import MoEState

moe.bootstrap("reference")


class _RefFused(StageSpan):
    requires_cuda = False
    dtypes = ("fp32", "fp16", "bf16")


@register
class RefFusedUpAct(_RefFused):
    """up_gemm + act in one span: h_up is never materialised."""

    name = "ref_fused_up_act"
    covers = ("up_gemm", "act")

    def __call__(self, st: MoEState) -> None:
        cfg = st.spec.model
        x_perm, offsets = st.require("x_perm", "expert_offsets")
        h_up = R.grouped_gemm_loop(x_perm, st.weights.w1, offsets,
                                   2 * cfg.intermediate_size)
        st.h_act = R.swiglu(h_up)


@register
class RefFusedDownScatter(_RefFused):
    """down_gemm + unpermute in one span: y_perm is never materialised.

    This is the shape of the kernel the bassrehab work could not express in
    Triton. Having a correct reference for it lets the harness compare against
    the right target before any CUDA is written.
    """

    name = "ref_fused_down_scatter"
    covers = ("down_gemm", "unpermute")

    def __call__(self, st: MoEState) -> None:
        cfg = st.spec.model
        h_act, offsets, perm, w = st.require(
            "h_act", "expert_offsets", "perm_index", "topk_weights")
        y_perm = R.grouped_gemm_loop(h_act, st.weights.w2, offsets, cfg.hidden_size)
        st.y = R.combine(y_perm, perm, w, st.spec.num_tokens, cfg.top_k)


@pytest.fixture
def toy_spec():
    return BenchSpec(MODEL_CONFIGS["toy"], num_tokens=32, dtype="fp32",
                     routing=RoutingSpec("uniform"), seed=0)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
