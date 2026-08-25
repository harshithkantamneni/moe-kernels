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

import atexit

from ..spec import BenchSpec
from ..stages import StageSpan, register
from ..state import MoEState
from ._framework_config import sglang_runner_kwargs

#: True only when this module created the process group, so teardown never
#: touches one a real distributed run owns.
_OWNS_PROCESS_GROUP = False

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

    `model_path` is the only required field of 472, and "dummy" is SGLang's own
    sentinel for "no real model": verified to construct with HF_HUB_OFFLINE=1,
    so it is special-cased rather than resolved over the network. Any other
    string is checked for real. A plausible name fails as a Hub lookup, and a
    genuine local directory carrying a valid config.json fails deeper still, in
    tokenizer resolution. "dummy" is the supported path, not a lucky one.
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
    rc.publish(ServerArgs(model_path="dummy"), role="test")


def _init_single_process_parallel() -> None:
    """SGLang's MoE path asserts a tensor-model-parallel group exists.

    `parallel_state.py:1951` is a bare assert, so with no group the call dies
    before any kernel. A server initialises this at startup; a library caller
    has to do it. Every parallel size defaults to 1 in
    `initialize_model_parallel`, so the single-GPU case is the default case and
    nothing here invents a topology.

    A free port is taken by binding to port 0 and reading back what the kernel
    assigned, rather than picking a number and hoping. There is a small race
    between closing that socket and torch binding it; acceptable for a
    single-process init, and it fails loudly rather than silently if lost.

    Guarded on torch.distributed.is_initialized(), because a second
    init_process_group raises and because a real distributed run must not have
    its group replaced by this one.
    """
    import socket

    import torch.distributed as dist
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if dist.is_initialized():
        return
    # Only tear down a group this module created. Without the flag an atexit
    # handler would destroy a group a real distributed run owns.
    global _OWNS_PROCESS_GROUP
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    _OWNS_PROCESS_GROUP = True
    atexit.register(_destroy_process_group)


def _destroy_process_group() -> None:
    """torch warns about a leaked group on every exit otherwise, which would be
    noise on every row of every sweep."""
    import torch.distributed as dist

    if _OWNS_PROCESS_GROUP and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001  - teardown must not mask a real error
            pass


# Config first, then the process group: that is the order a server starts in,
# and initialize_model_parallel reads config that publish projects.
_publish_runtime_context()
_init_single_process_parallel()


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
