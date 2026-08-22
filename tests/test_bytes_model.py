import pytest

from moe import pipeline as P
from moe.bench import bytes_model as BM
from moe.spec import MODEL_CONFIGS, BenchSpec
from moe.stages import get

REF = P.reference_pipeline_names()
FUSED_UP = ["ref_router", "ref_permute", "ref_fused_up_act", "ref_down_gemm",
            "ref_unpermute"]
FUSED_DOWN = ["ref_router", "ref_permute", "ref_up_gemm", "ref_act",
              "ref_fused_down_scatter"]


def spans(names):
    return [get(n) for n in names]


def mixtral(tokens=512, dtype="bf16"):
    return BenchSpec(MODEL_CONFIGS["mixtral-8x7b"], num_tokens=tokens, dtype=dtype)


def test_grouped_gemm_flops_match_the_closed_form():
    spec = mixtral()
    cost = BM.grouped_gemm_only_cost(spec, active_experts=8)
    cfg = spec.model
    expected = 6.0 * spec.rows * cfg.intermediate_size * cfg.hidden_size
    assert cost.flops == pytest.approx(expected)


def test_flops_are_invariant_across_tilings():
    spec = mixtral()
    a = BM.pipeline_cost(spans(REF), spec, 8).flops
    b = BM.pipeline_cost(spans(FUSED_UP), spec, 8).flops
    c = BM.pipeline_cost(spans(FUSED_DOWN), spec, 8).flops
    assert a == pytest.approx(b) == pytest.approx(c)


def test_fusing_up_and_act_removes_exactly_two_h_up_traversals():
    spec = mixtral()
    unfused = BM.pipeline_cost(spans(REF), spec, 8).bytes_total
    fused = BM.pipeline_cost(spans(FUSED_UP), spec, 8).bytes_total
    h_up_bytes = BM.field_bytes(spec)["h_up"]
    assert unfused - fused == 2 * h_up_bytes


def test_fusing_down_and_scatter_removes_exactly_two_y_perm_traversals():
    spec = mixtral()
    unfused = BM.pipeline_cost(spans(REF), spec, 8).bytes_total
    fused = BM.pipeline_cost(spans(FUSED_DOWN), spec, 8).bytes_total
    y_perm_bytes = BM.field_bytes(spec)["y_perm"]
    assert unfused - fused == 2 * y_perm_bytes


def test_fusion_raises_arithmetic_intensity():
    spec = mixtral()
    assert (BM.pipeline_cost(spans(FUSED_UP), spec, 8).arithmetic_intensity
            > BM.pipeline_cost(spans(REF), spec, 8).arithmetic_intensity)


def test_weight_traffic_counts_only_active_experts():
    spec = BenchSpec(MODEL_CONFIGS["deepseek-v3"], num_tokens=64, dtype="bf16")
    few = BM.grouped_gemm_only_cost(spec, active_experts=8).bytes_total
    many = BM.grouped_gemm_only_cost(spec, active_experts=256).bytes_total
    assert many > few
    # Weight traffic is linear in the number of experts actually touched.
    w8 = BM.weight_bytes_for_stage(spec, "up_gemm", 8)
    w256 = BM.weight_bytes_for_stage(spec, "up_gemm", 256)
    assert w256 == 32 * w8


def test_small_batch_many_expert_regime_is_weight_dominated():
    """The wall this project targets: at DeepSeek geometry with few tokens,
    weight traffic swamps every activation term, so the grouped GEMM is memory
    bound no matter how good the tiling is."""
    spec = BenchSpec(MODEL_CONFIGS["deepseek-v3"], num_tokens=32, dtype="bf16")
    cost = BM.grouped_gemm_only_cost(spec, active_experts=min(256, spec.rows))
    weights = sum(c.weight_bytes for c in cost.spans)
    activations = sum(c.read_bytes + c.write_bytes for c in cost.spans)
    assert weights > 20 * activations
    assert cost.arithmetic_intensity < 20  # far below any Hopper ridge point


def test_large_batch_is_compute_dominated_by_comparison():
    small = BenchSpec(MODEL_CONFIGS["mixtral-8x7b"], num_tokens=16, dtype="bf16")
    large = BenchSpec(MODEL_CONFIGS["mixtral-8x7b"], num_tokens=16384, dtype="bf16")
    ai_small = BM.grouped_gemm_only_cost(small, 8).arithmetic_intensity
    ai_large = BM.grouped_gemm_only_cost(large, 8).arithmetic_intensity
    assert ai_large > 20 * ai_small


def test_fp8_halves_the_dtype_dependent_traffic():
    a = BM.grouped_gemm_only_cost(mixtral(dtype="bf16"), 8).bytes_total
    b = BM.grouped_gemm_only_cost(mixtral(dtype="fp8_e4m3"), 8).bytes_total
    assert b == pytest.approx(a / 2, rel=0.01)


def test_derived_rates_are_consistent():
    cost = BM.grouped_gemm_only_cost(mixtral(), 8)
    ms = 2.0
    assert cost.tflops(ms) == pytest.approx(cost.flops / 2e-3 / 1e12)
    assert cost.gbps(ms) == pytest.approx(cost.bytes_total / 2e-3 / 1e9)
