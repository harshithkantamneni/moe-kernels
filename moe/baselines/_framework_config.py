"""Arguments the framework baselines must be handed, with no framework import.

The spans themselves import vLLM or SGLang at module scope, deliberately, so
`baselines.load_all()` skips them wherever the framework is absent rather than
letting a cell crash partway through a paid session. That makes the spans
untestable off the GPU box.

The argument construction is the part worth testing, and it does not need either
library. Every value here is one whose DEFAULT would produce a call that runs,
returns a tensor of the right shape, and computes a different layer than the
oracle is checking against. Those are the failures that survive a green test
suite, so they live in a module a laptop can import.
"""
from __future__ import annotations

from ..quant import FP8_DTYPES
from ..spec import BenchSpec


def vllm_call_kwargs(spec: BenchSpec) -> dict:
    """Non-tensor arguments for vLLM's `fused_experts`.

    `activation` is returned as its enum VALUE rather than the enum, so this
    module stays import-free; the span reconstructs `MoEActivation(value)`.
    """
    return {
        # SwiGLU: gated, hence not one of vLLM's explicit *_NO_MUL variants.
        "activation": "silu",
        # The reference combine() scales expert OUTPUTS by the combine weights.
        # Scaling inputs instead is a different computation wherever the expert
        # is nonlinear, which is everywhere.
        "apply_router_weight_on_input": False,
        # Defaults to -1 and is inferred. State it, so a shape disagreement is
        # an error rather than a silently different launch.
        "global_num_experts": spec.model.num_experts,
        # Single GPU: no expert-parallel remapping.
        "expert_map": None,
        # None resolves to FUSED_MOE_UNQUANTIZED_CONFIG inside fused_experts,
        # which is what bf16 wants.
        "quant_config": None,
    }


def sglang_runner_kwargs(spec: BenchSpec) -> dict:
    """MoeRunnerConfig fields, four of which fight their own defaults.

    routed_scaling_factor  defaults to None, which is correct, and is stated
        because it is load-bearing: gate_weights() has already applied
        DeepSeek-V3's 2.5x by the time any span runs, so a runner applying it
        again squares it to 6.25.
    gate_up_interleaved    defaults to TRUE, the opposite of this harness's
        layout. w1 is [E, 2F, H] as [gate | up] blocked, so interleaved pairs
        the wrong halves in every SwiGLU.
    inplace                defaults to TRUE and mutates hidden_states. `x` is
        built once per cell and reused by every timed iteration, so correctness
        would pass on the first call while the timing measured a decaying input.
    num_experts /          both default to None, and fused_experts computes
    num_local_experts      `filter_expert = num_experts is None or
        num_experts != num_local_experts`, so leaving them unset takes an
        expert-parallel masking path with no meaning on one GPU.
    """
    cfg = spec.model
    return {
        "num_experts": cfg.num_experts,
        "num_local_experts": cfg.num_experts,
        "hidden_size": cfg.hidden_size,
        "intermediate_size_per_partition": cfg.intermediate_size,
        "top_k": cfg.top_k,
        "activation": "silu",
        "is_gated": True,
        "apply_router_weight_on_input": False,
        "inplace": False,
        "no_combine": False,
        "routed_scaling_factor": None,
        "gate_up_interleaved": False,
    }


def vllm_quant_spec(spec: BenchSpec) -> dict | None:
    """What kind of quant config this cell needs, with no vLLM import.

    Returns None for float dtypes, which is what `fused_experts` wants: a
    `quant_config` of None resolves internally to FUSED_MOE_UNQUANTIZED_CONFIG.

    For fp8 the span builds `fp8_w8a8_moe_quant_config(w1_scale=..., w2_scale=...,
    **rest)`. The two flags below are stated rather than defaulted because both
    describe the quantisation this harness actually performed:

    per_act_token_quant  False. Activations are NOT quantised here, only weights.
        True would have vLLM expect a per-token activation scale that does not
        exist, and the shapes would be wrong rather than merely different.
    block_shape          None. Scales are per expert, one number for the whole
        tensor, not per [128, 128] block. A block shape sends the kernel down a
        blocked path expecting a 2-D scale grid.

    The scales themselves are tensors and so live on MoEWeights, not here; this
    module stays import-free so it can be tested on a laptop.
    """
    if spec.dtype not in FP8_DTYPES:
        return None
    return {
        "kind": "fp8_w8a8",
        "per_act_token_quant": False,
        "per_out_ch_quant": False,
        "block_shape": None,
    }
