import pytest

from moe.spec import (MODEL_CONFIGS, ACTIVE_DTYPES, BenchSpec, MoEConfig,
                      RoutingSpec, dtype_bytes, sweep)


def test_dtype_bytes_covers_active_set():
    for dt in ACTIVE_DTYPES:
        assert dtype_bytes(dt) > 0


def test_unknown_dtype_rejected():
    with pytest.raises(ValueError, match="unknown dtype"):
        dtype_bytes("int4")


def test_topk_cannot_exceed_experts():
    with pytest.raises(ValueError, match="exceeds num_experts"):
        MoEConfig(name="bad", hidden_size=8, intermediate_size=8,
                  num_experts=2, top_k=4)


def test_weight_shapes_follow_vllm_layout():
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    assert cfg.w1_shape == (8, 2 * 14336, 4096)
    assert cfg.w2_shape == (8, 4096, 14336)


def test_rows_account_for_topk_replication():
    spec = BenchSpec(MODEL_CONFIGS["mixtral-8x7b"], num_tokens=512)
    assert spec.rows == 512 * 2
    assert spec.mean_rows_per_expert == 128.0


def test_trace_routing_requires_id():
    with pytest.raises(ValueError, match="requires a trace_id"):
        RoutingSpec("trace")


def test_trace_id_rejected_for_parametric_kinds():
    with pytest.raises(ValueError, match="meaningless"):
        RoutingSpec("uniform", trace_id="mixtral-wiki")


def test_hot_fraction_must_be_a_fraction():
    with pytest.raises(ValueError, match="hot-expert mass"):
        RoutingSpec("hot", param=1.5)


def test_routing_labels_are_stable_identifiers():
    assert RoutingSpec("uniform").label == "uniform"
    assert RoutingSpec("zipf", 1.2).label == "zipf:1.2"
    assert RoutingSpec("trace", trace_id="mix-wiki").label == "trace:mix-wiki"


def test_sweep_is_deterministic_and_complete():
    cells = list(sweep([MODEL_CONFIGS["toy"]], [16, 32], ["fp32", "bf16"],
                       [RoutingSpec("uniform"), RoutingSpec("zipf", 1.0)]))
    assert len(cells) == 1 * 2 * 2 * 2
    assert [c.label for c in cells] == [c.label for c in sweep(
        [MODEL_CONFIGS["toy"]], [16, 32], ["fp32", "bf16"],
        [RoutingSpec("uniform"), RoutingSpec("zipf", 1.0)])]


def test_published_configs_are_flagged_unverified_until_checked():
    # Guards against publishing a benchmark labelled with a model name whose
    # dimensions were never diffed against the upstream config.json.
    for name, cfg in MODEL_CONFIGS.items():
        if name == "toy":
            continue
        assert cfg.hf_repo, f"{name} needs an hf_repo to be verifiable"
