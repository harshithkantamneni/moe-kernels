"""Configuration these baselines must be handed, testable without the frameworks.

vLLM and SGLang import only inside their own venvs, so the spans themselves
cannot be exercised on a laptop. What CAN be checked here is the argument
construction, and that is where the dangerous mistakes live: every trap below
produces a call that runs, returns a tensor, and computes a different layer.
"""
import pytest

from moe.spec import MODEL_CONFIGS, BenchSpec


@pytest.fixture
def deepseek():
    return BenchSpec(MODEL_CONFIGS["deepseek-v3"], num_tokens=8)


def test_sglang_runner_never_reapplies_the_routed_scaling_factor(deepseek):
    """DeepSeek-V3 multiplies combine weights by 2.5 AFTER renormalising, and
    gate_weights() has already done it by the time any span runs. SGLang's
    MoeRunnerConfig carries its own routed_scaling_factor and applies it inside
    the kernel, so setting it too squares the factor to 6.25."""
    from moe.baselines._framework_config import sglang_runner_kwargs as runner_kwargs

    kw = runner_kwargs(deepseek)
    assert deepseek.model.routed_scaling_factor == 2.5, "fixture must exercise this"
    assert kw["routed_scaling_factor"] is None


def test_sglang_runner_matches_our_blocked_gate_up_layout(deepseek):
    """w1 is [E, 2F, H] as [gate | up]. SGLang defaults gate_up_interleaved to
    True, which is the opposite layout, and would pair the wrong halves in
    every SwiGLU."""
    from moe.baselines._framework_config import sglang_runner_kwargs as runner_kwargs

    assert runner_kwargs(deepseek)["gate_up_interleaved"] is False


def test_sglang_runner_is_not_inplace(deepseek):
    """inplace defaults to True and mutates hidden_states. `x` is built once per
    cell and reused by every timed iteration, so the second iteration onward
    would run on corrupted input: correctness passes on the first call and the
    timing measures something else."""
    from moe.baselines._framework_config import sglang_runner_kwargs as runner_kwargs

    assert runner_kwargs(deepseek)["inplace"] is False


def test_sglang_runner_disables_the_expert_parallel_filter(deepseek):
    """fused_experts computes
    filter_expert = num_experts is None or num_experts != num_local_experts,
    so leaving either as None takes an expert-parallel masking path that has no
    meaning on one GPU."""
    from moe.baselines._framework_config import sglang_runner_kwargs as runner_kwargs

    kw = runner_kwargs(deepseek)
    E = deepseek.model.num_experts
    assert kw["num_experts"] == E and kw["num_local_experts"] == E


def test_vllm_is_told_the_global_expert_count(deepseek):
    """global_num_experts defaults to -1. Pass the real count rather than rely
    on it being inferred."""
    from moe.baselines._framework_config import vllm_call_kwargs as call_kwargs

    assert call_kwargs(deepseek)["global_num_experts"] == deepseek.model.num_experts


def test_vllm_applies_router_weights_on_the_output(deepseek):
    """The reference combine() scales expert OUTPUTS by the combine weights.
    apply_router_weight_on_input=True would scale the inputs instead, which is
    a different computation whenever an expert is nonlinear."""
    from moe.baselines._framework_config import vllm_call_kwargs as call_kwargs

    assert call_kwargs(deepseek)["apply_router_weight_on_input"] is False
