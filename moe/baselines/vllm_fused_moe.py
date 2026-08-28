"""vLLM's fused_moe as a baseline span.

`fused_experts` takes `topk_weights`/`topk_ids` rather than router logits, so it
covers exactly `permute -> up_gemm -> act -> down_gemm -> unpermute`: a
contiguous span, with the reference router upstream supplying the routing
decision and the combine weights. That matters for more than tidiness. The gate
rules are model-specific, and computing them upstream keeps DeepSeek-V3's
`routed_scaling_factor` of 2.5 applied exactly once and Qwen2's
`norm_topk_prob=false` respected, without this file knowing either fact.

Probed against vLLM 0.27.1:

    fused_experts(hidden_states, w1, w2, topk_weights, topk_ids,
                  activation=MoEActivation.SILU,
                  apply_router_weight_on_input=False,
                  global_num_experts=-1, expert_map=None, quant_config=None)

`quant_config=None` falls through to FUSED_MOE_UNQUANTIZED_CONFIG inside, which
is what bf16 wants. `MoEActivation.SILU` on a gated w1 is silu(gate) * up, the
same SwiGLU `reference/torch_ref.swiglu` computes; the enum has explicit
`SILU_NO_MUL` variants, so plain SILU is the gated one.

Weight layout is the harness's own: `w1` is `[E, 2F, H]` as `[gate | up]` and
`w2` is `[E, H, F]`, chosen to match this convention so no transpose is needed.
"""
from __future__ import annotations

from ..quant import fp8_cell_supported, fp8_hardware_support
from ..spec import BenchSpec
from ..stages import StageSpan, register
from ..state import MoEState
from ._framework_config import vllm_call_kwargs, vllm_quant_spec

# Import at module scope on purpose. `baselines.load_all()` catches the failure
# and skips this module with a warning, so on a laptop or in the base venv the
# span simply does not register, exactly like a kernel whose extension did not
# build.
from vllm.model_executor.layers.fused_moe import (  # noqa: E402  isort:skip
    fused_experts,
)
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402  isort:skip
    MoEActivation,
)


def quant_config(spec: BenchSpec, weights):
    """vLLM's quant config for this cell, or None for a float dtype.

    None is correct rather than lazy: `fused_experts` resolves it internally to
    FUSED_MOE_UNQUANTIZED_CONFIG, which is what bf16 wants.

    For fp8 the scales come from `MoEWeights`, which quantised them per expert
    with the contract `q * scale == original`. That is the direction this helper
    reconstructs in, so passing the reciprocal would produce a call that runs,
    returns the right shape, and computes a different layer.
    """
    q = vllm_quant_spec(spec)
    if q is None:
        return None
    if not weights.quantised:
        raise ValueError(
            f"{spec.dtype} cell but the weights carry no scales; make_inputs "
            "should have quantised them")
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config,
    )
    return fp8_w8a8_moe_quant_config(
        w1_scale=weights.w1_scale,
        w2_scale=weights.w2_scale,
        per_act_token_quant=q["per_act_token_quant"],
        per_out_ch_quant=q["per_out_ch_quant"],
        block_shape=q["block_shape"],
    )


def call_kwargs(spec: BenchSpec) -> dict:
    """The framework-free config, with the activation resolved to vLLM's enum.

    `_framework_config` holds the values so they can be tested on a laptop;
    only the enum reconstruction needs vLLM present.
    """
    kw = vllm_call_kwargs(spec)
    kw["activation"] = MoEActivation(kw["activation"])
    return kw


@register
class VllmFusedExperts(StageSpan):
    """vLLM's Triton fused MoE, from permute through the weighted combine."""

    name = "vllm_fused_experts"
    covers = ("permute", "up_gemm", "act", "down_gemm", "unpermute")
    env = "vllm"
    requires_cuda = True
    #: Unverified. The harness records capture_status per row, so this starts
    #: false and follows the evidence rather than leading it.
    cuda_graph_safe = False
    #: fp8_e4m3 only among the 8-bit formats: vLLM's w8a8 path is built on the
    #: e4m3 dtype, and e5m2 would be a different kernel. SGLang stays bf16-only
    #: because its runner takes a different config object and has not been
    #: probed for this.
    dtypes = ("bf16", "fp8_e4m3")

    def supports(self, spec: BenchSpec) -> bool:
        """fp8 needs fp8 silicon. Ampere has none, and vLLM would accept the
        cell anyway, most likely dequantise to bf16, and write a row labelled
        fp8_e4m3 that never touched an fp8 unit. Merged with the H200's it would
        be indistinguishable."""
        return super().supports(spec) and fp8_cell_supported(spec)

    def why_unsupported(self, spec: BenchSpec) -> str:
        if not fp8_cell_supported(spec):
            verdict = fp8_hardware_support()
            if verdict is not None:
                return verdict.reason
        return super().why_unsupported(spec)

    def __call__(self, st: MoEState) -> None:
        x, topk_ids, topk_weights = st.require("x", "topk_ids", "topk_weights")
        kw = call_kwargs(st.spec)
        kw["quant_config"] = quant_config(st.spec, st.weights)
        # Activations stay in the compute dtype. This is w8a8 in vLLM's naming
        # but the harness quantises WEIGHTS only, so `x` is handed over as
        # drawn; per_act_token_quant is False for exactly that reason.
        st.y = fused_experts(
            hidden_states=x,
            w1=st.weights.w1,
            w2=st.weights.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            **kw,
        )
