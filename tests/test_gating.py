"""Gate-function fidelity per model.

These four models genuinely differ, and getting any of them wrong silently
changes the combine weights and therefore the layer output:
  Mixtral        softmax, renormalise (in code, not config), no scaling
  Qwen2-57B      softmax, NO renormalisation
  DeepSeek-V3    sigmoid, renormalise, then multiply by 2.5
  DeepSeek-V2-Lite softmax, NO renormalisation
"""
import pytest
import torch

from moe.reference.torch_ref import gate_scores, gate_weights, route
from moe.spec import MODEL_CONFIGS


def logits(T=8, E=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((T, E), generator=g)


def top_ids(scores, k):
    return torch.topk(scores, k, dim=-1).indices.to(torch.int32)


def test_softmax_scores_sum_to_one_across_experts():
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    s = gate_scores(logits(), cfg)
    assert torch.allclose(s.sum(-1), torch.ones(8), atol=1e-6)


def test_sigmoid_scores_do_not_sum_to_one():
    cfg = MODEL_CONFIGS["deepseek-v3"]
    s = gate_scores(logits(), cfg)
    assert not torch.allclose(s.sum(-1), torch.ones(8), atol=1e-2)
    assert bool(((s > 0) & (s < 1)).all())


def test_mixtral_combine_weights_sum_to_one():
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    lg = logits(E=cfg.num_experts)
    s = gate_scores(lg, cfg)
    w = gate_weights(s, top_ids(s, cfg.top_k), cfg)
    assert torch.allclose(w.sum(-1), torch.ones(lg.shape[0]), atol=1e-6)


def test_qwen2_does_not_renormalise():
    """norm_topk_prob is false in Qwen2's config.json. Its top-8 of 64 softmax
    probabilities must sum to less than 1, not be rescaled up to 1."""
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    assert cfg.norm_topk_prob is False
    lg = logits(E=cfg.num_experts)
    s = gate_scores(lg, cfg)
    w = gate_weights(s, top_ids(s, cfg.top_k), cfg)
    assert bool((w.sum(-1) < 0.999).all()), w.sum(-1)
    assert bool((w.sum(-1) > 0.0).all())


def test_deepseek_v3_scales_after_renormalising():
    """Renormalise to 1, then multiply by routed_scaling_factor 2.5. The order
    matters: scaling before renormalisation would cancel out entirely."""
    cfg = MODEL_CONFIGS["deepseek-v3"]
    assert cfg.routed_scaling_factor == 2.5
    lg = logits(E=cfg.num_experts)
    s = gate_scores(lg, cfg)
    w = gate_weights(s, top_ids(s, cfg.top_k), cfg)
    assert torch.allclose(w.sum(-1), torch.full((lg.shape[0],), 2.5), atol=1e-5)


def test_deepseek_v2_lite_does_not_renormalise_or_scale():
    cfg = MODEL_CONFIGS["deepseek-v2-lite"]
    lg = logits(E=cfg.num_experts)
    s = gate_scores(lg, cfg)
    w = gate_weights(s, top_ids(s, cfg.top_k), cfg)
    assert bool((w.sum(-1) < 0.999).all())


def test_scaling_would_be_a_no_op_if_applied_before_renormalisation():
    """Guards the ordering above by showing the alternative is degenerate."""
    cfg = MODEL_CONFIGS["deepseek-v3"]
    lg = logits(E=cfg.num_experts)
    s = gate_scores(lg, cfg)
    ids = top_ids(s, cfg.top_k)
    raw = torch.gather(s, 1, ids.long())
    wrong = raw * cfg.routed_scaling_factor
    wrong = wrong / (wrong.sum(-1, keepdim=True) + 1e-20)
    right = gate_weights(s, ids, cfg)
    assert not torch.allclose(wrong, right)
    assert torch.allclose(wrong.sum(-1), torch.ones(lg.shape[0]), atol=1e-5)


def test_all_zero_sigmoid_row_does_not_produce_nan():
    cfg = MODEL_CONFIGS["deepseek-v3"]
    s = torch.zeros((2, cfg.num_experts))
    w = gate_weights(s, top_ids(s, cfg.top_k), cfg)
    assert torch.isfinite(w).all()


def test_route_returns_consistent_ids_and_weights():
    cfg = MODEL_CONFIGS["toy"]
    g = torch.Generator().manual_seed(0)
    x = torch.randn((6, cfg.hidden_size), generator=g)
    wg = torch.randn((cfg.num_experts, cfg.hidden_size), generator=g)
    lg, ids, w = route(x, wg, cfg)
    assert lg.shape == (6, cfg.num_experts)
    assert ids.shape == (6, cfg.top_k) and ids.dtype == torch.int32
    assert w.shape == (6, cfg.top_k) and w.dtype == torch.float32
    for row in ids:
        assert len(set(row.tolist())) == cfg.top_k


@pytest.mark.parametrize("name", sorted(MODEL_CONFIGS))
def test_every_config_is_verified_against_upstream(name):
    """Guard: a benchmark labelled with a model name must not carry geometry
    that was never diffed against the upstream config.json."""
    assert MODEL_CONFIGS[name].verified is True
