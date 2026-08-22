import numpy as np
import pytest
import torch

from moe.routing import traces as TR
from moe.routing.distributions import expert_probs, feasible, realize_counts, sample_topk_ids
from moe.routing.imbalance import expert_load
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec


def counts_of(ids, E):
    return torch.bincount(ids.reshape(-1).long(), minlength=E).tolist()


# --- parametric distributions ---------------------------------------------

@pytest.mark.parametrize("routing", [
    RoutingSpec("uniform"),
    RoutingSpec("zipf", 1.2),
    RoutingSpec("hot", 0.6),
    RoutingSpec("dirichlet", 0.3),
])
def test_probs_are_a_distribution(routing):
    g = torch.Generator().manual_seed(0)
    p = expert_probs(routing, 16, g)
    assert p.shape == (16,)
    assert float(p.sum()) == pytest.approx(1.0)
    assert bool((p >= 0).all())


def test_trace_kind_has_no_parametric_form():
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="no parametric form"):
        expert_probs(RoutingSpec("trace", trace_id="x"), 8, g)


def test_sampled_ids_are_distinct_per_token():
    ids = sample_topk_ids(RoutingSpec("hot", 0.9), 128, 8, 4, seed=0)
    assert ids.shape == (128, 4)
    assert ids.dtype == torch.int32
    for row in ids:
        assert len(set(row.tolist())) == 4


def test_sampling_is_deterministic_in_the_seed():
    a = sample_topk_ids(RoutingSpec("zipf", 1.0), 64, 32, 4, seed=7)
    b = sample_topk_ids(RoutingSpec("zipf", 1.0), 64, 32, 4, seed=7)
    c = sample_topk_ids(RoutingSpec("zipf", 1.0), 64, 32, 4, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_skew_parameter_produces_monotonically_worse_balance():
    E, T, k = 64, 2048, 4
    cvs = []
    for s in (0.0, 0.6, 1.2, 2.0):
        ids = sample_topk_ids(RoutingSpec("zipf", s), T, E, k, seed=0)
        cvs.append(expert_load(counts_of(ids, E)).cv)
    assert all(a < b for a, b in zip(cvs, cvs[1:], strict=False)), (
        f"zipf exponent must STRICTLY worsen balance, got {cvs}. Equal values "
        "mean the sampler is degenerate and ignoring the distribution.")


def test_hot_routing_concentrates_mass_on_one_expert():
    E, T, k = 32, 4096, 2
    ids = sample_topk_ids(RoutingSpec("hot", 0.5), T, E, k, seed=1)
    load = expert_load(counts_of(ids, E))
    assert load.top1_share > 0.3
    assert load.max_over_mean > 5.0
    # The realised hot expert must be the one the distribution made hot, not
    # expert 0 by accident of index order.
    g = torch.Generator().manual_seed(1)
    p = expert_probs(RoutingSpec("hot", 0.5), E, g)
    assert int(np.argmax(counts_of(ids, E))) == int(p.argmax())


def test_sampler_actually_follows_the_distribution():
    """Regression: a NaN in the Gumbel keys made topk return experts [0..k-1]
    for every token, which silently passed weaker balance assertions."""
    E, T, k = 32, 4096, 4
    routing = RoutingSpec("zipf", 1.0)
    g = torch.Generator().manual_seed(3)
    p = expert_probs(routing, E, g)
    ids = sample_topk_ids(routing, T, E, k, seed=3)
    realised = np.array(counts_of(ids, E), dtype=float)

    assert int(ids.max()) > k, "sampler never selected an expert outside the first k"
    assert (realised > 0).sum() > k, "sampler collapsed onto a fixed expert set"
    # Empirical load must track the intended probabilities.
    corr = np.corrcoef(realised, p.numpy())[0, 1]
    assert corr > 0.9, f"realised load barely correlates with the target: {corr:.3f}"


def test_uniform_routing_is_actually_balanced():
    E, T, k = 16, 8192, 2
    ids = sample_topk_ids(RoutingSpec("uniform"), T, E, k, seed=0)
    load = expert_load(counts_of(ids, E))
    assert load.max_over_mean < 1.1
    assert load.entropy_norm > 0.99


def test_topk_cannot_exceed_experts():
    with pytest.raises(ValueError, match="exceeds num_experts"):
        sample_topk_ids(RoutingSpec("uniform"), 8, 4, 8)


def test_top_k_equal_to_num_experts_gives_every_expert_every_token():
    ids = sample_topk_ids(RoutingSpec("zipf", 2.0), 16, 4, 4, seed=0)
    assert counts_of(ids, 4) == [16, 16, 16, 16]


# --- exact realisation -----------------------------------------------------

def test_realisation_matches_the_target_histogram_exactly():
    target = [10, 0, 6, 1, 15]
    T, k = 16, 2
    ids = realize_counts(target, T, k)
    assert ids.shape == (T, k)
    assert counts_of(ids, len(target)) == target


def test_realisation_keeps_experts_distinct_within_a_token():
    target = [8, 8, 8, 0]
    ids = realize_counts(target, 8, 3)
    for row in ids:
        assert len(set(row.tolist())) == 3


def test_realisation_handles_maximum_concentration():
    # Every token must pick expert 0, plus one other.
    target = [8, 4, 4, 0]
    ids = realize_counts(target, 8, 2)
    assert counts_of(ids, 4) == target
    assert all(0 in row.tolist() for row in ids)


def test_infeasible_totals_rejected():
    ok, why = feasible([5, 5], num_tokens=4, top_k=2)
    assert not ok and "sum to" in why
    with pytest.raises(ValueError, match="infeasible"):
        realize_counts([5, 5], 4, 2)


def test_expert_demanding_more_rows_than_tokens_rejected():
    ok, why = feasible([9, 1, 0, 0, 0], num_tokens=5, top_k=2)
    assert not ok and "cannot pick the same expert twice" in why


def test_negative_counts_rejected():
    ok, why = feasible([-1, 5], 2, 2)
    assert not ok and "negative" in why


# --- rescaling -------------------------------------------------------------

def test_rescale_hits_the_exact_token_budget():
    captured = [312, 15, 0, 91, 4, 700, 22, 56]
    out = TR.rescale_counts(captured, num_tokens=256, top_k=2)
    assert sum(out) == 512
    assert max(out) <= 256


def test_rescale_preserves_the_shape_of_the_distribution():
    captured = [1000, 500, 250, 125, 60, 30, 15, 8]
    out = TR.rescale_counts(captured, num_tokens=512, top_k=2)
    # Ordering of experts by load must survive the rescale.
    assert np.argsort(out).tolist() == np.argsort(captured).tolist()


def test_rescale_caps_a_dominant_expert_and_stays_feasible():
    captured = [10_000, 1, 1, 1]
    out = TR.rescale_counts(captured, num_tokens=64, top_k=2)
    assert sum(out) == 128
    assert max(out) <= 64
    realize_counts(out, 64, 2)  # must not raise


def test_rescale_of_an_all_zero_capture_falls_back_to_uniform():
    out = TR.rescale_counts([0, 0, 0, 0], num_tokens=10, top_k=2)
    assert sum(out) == 20
    assert max(out) - min(out) <= 1


def test_rescale_needs_at_least_top_k_experts():
    with pytest.raises(ValueError, match="cannot route top_k"):
        TR.rescale_counts([1, 1], num_tokens=8, top_k=4)


# --- trace files -----------------------------------------------------------

def make_trace(tmp_path, E=8, B=2, L=3, trace_id="toy-chat-decode"):
    rng = np.random.default_rng(0)
    counts = np.zeros((B, L, E), dtype=np.int32)
    for b in range(B):
        for ln in range(L):
            p = rng.dirichlet(np.full(E, 0.4))
            counts[b, ln] = (p * 1000).astype(np.int32)
    meta = dict(trace_id=trace_id, model="toy", hf_repo="none", corpus="chat",
                phase="decode", num_experts=E, top_k=2,
                captured_at="2026-08-21T00:00:00")
    TR.write_trace(tmp_path / f"{trace_id}.npz", counts, meta)
    return counts


def test_trace_round_trip(tmp_path):
    counts = make_trace(tmp_path)
    ts = TR.TraceSet.load(tmp_path)
    assert len(ts) == 1
    trace, _, _ = ts.get("toy-chat-decode")
    assert trace.shape == counts.shape
    assert np.array_equal(trace.counts, counts)
    assert trace.meta["phase"] == "decode"


def test_trace_metadata_is_validated(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        TR.write_trace(tmp_path / "x.npz", np.zeros((1, 1, 4), dtype=np.int32),
                       {"trace_id": "x"})


def test_trace_expert_count_must_match_metadata(tmp_path):
    meta = dict(trace_id="x", model="m", hf_repo="r", corpus="c", phase="decode",
                num_experts=99, top_k=2, captured_at="now")
    with pytest.raises(ValueError, match="metadata says 99 experts"):
        TR.write_trace(tmp_path / "x.npz", np.zeros((1, 1, 4), dtype=np.int32), meta)


def test_slice_suffix_pins_batch_and_layer(tmp_path):
    counts = make_trace(tmp_path)
    ts = TR.TraceSet.load(tmp_path)
    trace, b, ln = ts.get("toy-chat-decode@b1l2")
    got, rb, rl = trace.select(b, ln)
    assert (rb, rl) == (1, 2)
    assert np.array_equal(got, counts[1, 2])


def test_unsuffixed_selection_varies_with_seed_and_is_reproducible(tmp_path):
    make_trace(tmp_path)
    ts = TR.TraceSet.load(tmp_path)
    assert ts.resolved_slice("toy-chat-decode", seed=0) == "toy-chat-decode@b0l0"
    assert ts.resolved_slice("toy-chat-decode", seed=1) == "toy-chat-decode@b1l1"
    assert ts.resolved_slice("toy-chat-decode", seed=1) == "toy-chat-decode@b1l1"


def test_replay_reproduces_the_captured_shape(tmp_path):
    counts = make_trace(tmp_path, E=8)
    ts = TR.TraceSet.load(tmp_path)
    cfg = MODEL_CONFIGS["toy"]
    assert cfg.num_experts == 4
    # toy has 4 experts, the trace has 8: that mismatch must be refused.
    spec = BenchSpec(cfg, num_tokens=32, dtype="fp32",
                     routing=RoutingSpec("trace", trace_id="toy-chat-decode"))
    with pytest.raises(ValueError, match="only meaningful for the model"):
        ts.forced_ids("toy-chat-decode", spec)

    # A matching trace replays, and the realised load ranks experts the same way.
    counts4 = counts[:, :, :4].copy()
    meta = dict(trace_id="toy4", model="toy", hf_repo="none", corpus="chat",
                phase="decode", num_experts=4, top_k=2, captured_at="now")
    TR.write_trace(tmp_path / "toy4.npz", counts4, meta)
    ts = TR.TraceSet.load(tmp_path)
    spec = BenchSpec(cfg, num_tokens=64, dtype="fp32", seed=0,
                     routing=RoutingSpec("trace", trace_id="toy4"))
    ids = ts.forced_ids("toy4", spec)
    assert ids.shape == (64, 2)
    realised = counts_of(ids, 4)
    assert sum(realised) == 128
    assert np.argsort(realised).tolist() == np.argsort(counts4[0, 0]).tolist()


def test_missing_trace_names_what_exists(tmp_path):
    make_trace(tmp_path)
    ts = TR.TraceSet.load(tmp_path)
    with pytest.raises(KeyError, match="capture_traces.py"):
        ts.get("nonexistent")
