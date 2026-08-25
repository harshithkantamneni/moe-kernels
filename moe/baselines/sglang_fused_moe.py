"""SGLang's fused_moe as a baseline span.

Same span as the vLLM baseline, `permute -> up_gemm -> act -> down_gemm ->
unpermute`, but the call needs two config objects rather than plain arguments,
and three of their DEFAULTS are wrong for this harness. Each would produce a
call that runs, returns a tensor of the right shape, and computes a different
layer, which the fp32 oracle catches only as an unexplained number.

Probed against SGLang 0.5.18:

    fused_experts(hidden_states, w1, w2,
                  topk_output: StandardTopKOutput,
                  moe_runner_config: MoeRunnerConfig, ...)

    StandardTopKOutput = NamedTuple(topk_weights, topk_ids, router_logits)

`fused_experts` unpacks that as `topk_weights, topk_ids, _`, discarding the
logits, which is why this passes None for them: the span's contract does not
include `router_logits`, and reading a field it never declared would be exactly
the kind of undeclared dependency `pipeline.build` exists to reject.
"""
from __future__ import annotations

from ..spec import BenchSpec
from ..stages import StageSpan, register
from ..state import MoEState
from ._framework_config import sglang_runner_kwargs

# Module-scope import: baselines.load_all() skips this file with a warning
# wherever SGLang is absent.
from sglang.srt.layers.moe.fused_moe_triton import (  # noqa: E402  isort:skip
    fused_experts,
)
from sglang.srt.layers.moe.moe_runner.base import (  # noqa: E402  isort:skip
    MoeRunnerConfig,
)
from sglang.srt.layers.moe.topk import (  # noqa: E402  isort:skip
    StandardTopKOutput,
)

def _publish_runtime_context() -> None:
    """SGLang is a server, and its MoE path reads process-wide config.

    `fused_experts` reaches `get_exec().moe.*`, which raises
    "config namespace 'exec' not published" until a server has published one.
    Called as a library there is no server, so every cell crashed before
    reaching a kernel.

    Publishing DEFAULT ServerArgs is not a stub standing in for a real config.
    The MoE path reads exactly four leaves, and on SGLang 0.5.18 all four come
    out of the defaults at their single-GPU values, which is what a one-GPU
    grouped-GEMM benchmark wants:

        moe.enable_waterfill          False
        moe.enable_eplb               False
        moe.init_expert_location      "trivial"
        moe.ep_num_redundant_experts  0

    `model_path` is the only required field and nothing loads a model from it.
    The role is "test", which is a real entry in ROLE_NAMESPACE_SETS and is
    honest about what this process is; with SGLANG_ROLE_NAMESPACES off it is
    provenance only.

    CAVEAT worth carrying into any published comparison: a default publish is
    not identical to what a running SGLang server would publish. If some other
    default steers kernel selection differently from a real deployment, this
    measures a path production would not take. The four leaves above are the
    ones the MoE code actually reads, which is the evidence for that being a
    narrow risk rather than an open one.

    Idempotent, and never clobbers an existing context: publish is
    last-publish-wins, so re-publishing over a live engine would replace its
    config.
    """
    import sglang.srt.runtime_context as rc
    from sglang.srt.server_args import ServerArgs

    if rc._CONTEXT.is_config_namespace_published("exec"):
        return
    rc.publish(ServerArgs(model_path="moe-kernels-benchmark"), role="test")


_publish_runtime_context()


def runner_kwargs(spec: BenchSpec) -> dict:
    """Re-exported from `_framework_config`, which a laptop can import."""
    return sglang_runner_kwargs(spec)


@register
class SglangFusedExperts(StageSpan):
    """SGLang's Triton fused MoE, from permute through the weighted combine."""

    name = "sglang_fused_experts"
    covers = ("permute", "up_gemm", "act", "down_gemm", "unpermute")
    env = "sglang"
    requires_cuda = True
    #: Unverified; the capture_status column decides this, not this line.
    cuda_graph_safe = False
    dtypes = ("bf16",)

    def __call__(self, st: MoEState) -> None:
        x, topk_ids, topk_weights = st.require("x", "topk_ids", "topk_weights")
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            # Discarded by fused_experts, and outside this span's contract.
            router_logits=None,
        )
        st.y = fused_experts(
            hidden_states=x,
            w1=st.weights.w1,
            w2=st.weights.w2,
            topk_output=topk_output,
            moe_runner_config=MoeRunnerConfig(**runner_kwargs(st.spec)),
        )
