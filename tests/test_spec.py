import pytest

from moe.spec import (
    ACTIVE_DTYPES,
    MODEL_CONFIGS,
    BenchSpec,
    MoEConfig,
    RoutingSpec,
    dtype_bytes,
    sweep,
)


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


# --------------------------------------------------------------------------
# Tensor-parallel shard widths
# --------------------------------------------------------------------------
#
# Every published row in this study is TP=1, and `E=256,N=2048` -- unsharded
# DeepSeek-V3 -- is a shape vLLM ships no tuned config for on any device,
# because it is 1.3 TB of weights and nobody serves it. These tests defend the
# machinery that lets a config describe the widths that are served, and above
# all defend the four existing models against it.


def test_the_four_published_models_are_untouched_by_the_shard_field():
    """The load-bearing test. Ten published arms, 96,448 rows, all TP=1, and
    every one of them is interpreted through these five configs. A default that
    changed any of their shapes would silently reinterpret all of it."""
    for name, cfg in MODEL_CONFIGS.items():
        if cfg.tensor_parallel != 1:
            continue
        assert cfg.full_intermediate_size == cfg.intermediate_size, name
        assert cfg.w1_shape == (cfg.num_experts, 2 * cfg.intermediate_size,
                                cfg.hidden_size), name
        assert cfg.w2_shape == (cfg.num_experts, cfg.hidden_size,
                                cfg.intermediate_size), name
    ds = MODEL_CONFIGS["deepseek-v3"]
    assert (ds.hidden_size, ds.intermediate_size, ds.num_experts, ds.top_k) \
        == (7168, 2048, 256, 8)
    assert ds.tensor_parallel == 1
    assert ds.weight_bytes("bf16") == 256 * 3 * 2048 * 7168 * 2


def test_a_shard_divides_the_expert_width_and_nothing_else():
    from moe.spec import tensor_parallel_shard

    base = MODEL_CONFIGS["deepseek-v3"]
    tp8 = tensor_parallel_shard(base, 8)
    assert tp8.intermediate_size == 256
    assert tp8.full_intermediate_size == base.intermediate_size
    assert tp8.tensor_parallel == 8
    # Pure TP leaves the expert count and the hidden size whole. Sharding either
    # would be expert parallelism or sequence parallelism, which this does not
    # model, and would silently change what `2R/b` predicts.
    assert tp8.num_experts == base.num_experts
    assert tp8.hidden_size == base.hidden_size
    assert tp8.top_k == base.top_k
    assert (tp8.gate_fn, tp8.norm_topk_prob, tp8.routed_scaling_factor) == \
        (base.gate_fn, base.norm_topk_prob, base.routed_scaling_factor)
    assert tp8.num_layers == base.num_layers
    assert tp8.hf_repo == base.hf_repo


def test_the_shard_shows_up_where_vllm_looks_for_it():
    """vLLM's `try_get_optimal_moe_config` does `E, _, N = w2_shape` and keys its
    config file on that pair, so the shard has to land in `w2`'s third dim or it
    changes nothing about which kernel runs."""
    E, _, N = MODEL_CONFIGS["deepseek-v3-tp8"].w2_shape
    assert (E, N) == (256, 256)
    E, _, N = MODEL_CONFIGS["deepseek-v3-tp4"].w2_shape
    assert (E, N) == (256, 512)
    E, _, N = MODEL_CONFIGS["deepseek-v3"].w2_shape
    assert (E, N) == (256, 2048)


def test_a_width_that_does_not_divide_is_refused_not_floored():
    """`F // TP` produces a geometry no rank ever holds, and every byte,
    crossing and footprint downstream would then describe a layer that does not
    exist. DeepSeek-V2-Lite at 1408 over 3 is the case that reaches this."""
    from moe.spec import tensor_parallel_shard

    with pytest.raises(ValueError, match="not a shape any rank holds"):
        tensor_parallel_shard(MODEL_CONFIGS["deepseek-v2-lite"], 3)


def test_a_shard_width_below_one_is_refused():
    from moe.spec import tensor_parallel_shard

    with pytest.raises(ValueError, match="at least 1"):
        tensor_parallel_shard(MODEL_CONFIGS["deepseek-v3"], 0)


def test_sharding_twice_composes_rather_than_forgetting_the_first_split():
    """`full_intermediate_size` is derived as `intermediate_size *
    tensor_parallel`, so a shard of a shard has to multiply the factors or the
    unsharded width it reports is wrong by the first divisor."""
    from moe.spec import tensor_parallel_shard

    base = MODEL_CONFIGS["deepseek-v3"]
    twice = tensor_parallel_shard(tensor_parallel_shard(base, 2), 4)
    assert twice.intermediate_size == 256
    assert twice.tensor_parallel == 8
    assert twice.full_intermediate_size == base.intermediate_size


def test_shard_names_say_their_width_because_the_csv_has_nowhere_else_to_put_it():
    """The schema is at v4 and records `model` but no shard column, so the name
    is the only thing that distinguishes a TP=8 row from a TP=1 one in a
    published CSV."""
    for name, cfg in MODEL_CONFIGS.items():
        if cfg.tensor_parallel == 1:
            assert "-tp" not in name, name
        else:
            assert name.endswith(f"-tp{cfg.tensor_parallel}"), name


def test_a_shard_inherits_its_bases_verification_rather_than_asserting_its_own():
    """A shard is exact arithmetic over a base that WAS diffed against the
    upstream config.json, so it is as verified as its base and no more. An
    unverified base must not launder itself into a verified shard."""
    from moe.spec import tensor_parallel_shard

    unverified = MoEConfig(name="rumour", hidden_size=64, intermediate_size=128,
                           num_experts=4, top_k=2, verified=False)
    assert tensor_parallel_shard(unverified, 2).verified is False
    assert MODEL_CONFIGS["deepseek-v3-tp8"].verified is True
