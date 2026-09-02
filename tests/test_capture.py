"""CPU tests for the trace-capture machinery.

The capture script itself needs a GPU and a 93 GB model download, so it cannot
be run here. What CAN be verified on a laptop is everything that would waste
that expensive session: locating the gate modules in an unfamiliar model layout,
accumulating counts correctly, and writing a well-formed trace file.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from moe.routing.capture import routed_ids
from moe.routing.traces import TraceSet, load_trace
from moe.spec import MoEConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_traces.py"
spec = importlib.util.spec_from_file_location("capture_traces", SCRIPT)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)

CFG = MoEConfig(name="synthetic", hidden_size=32, intermediate_size=64,
                num_experts=8, top_k=2, num_layers=3, verified=True)


class Mixtralish(nn.Module):
    """Mixtral names it block_sparse_moe.gate; Qwen2 and DeepSeek use mlp.gate.
    Both layouts appear here, alongside decoys that must NOT be hooked."""

    def __init__(self, cfg, layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            block = nn.Module()
            moe = nn.Module()
            moe.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
            if i % 2 == 0:
                block.block_sparse_moe = moe
            else:
                block.mlp = moe
            # Decoys: right name, wrong width; and right width, wrong name.
            block.attn_gate = nn.Linear(cfg.hidden_size, cfg.num_experts)
            block.shared_gate = nn.Linear(cfg.hidden_size, 1)
            self.layers.append(block)


def test_finds_every_gate_across_naming_conventions():
    model = Mixtralish(CFG)
    found = capture.find_gate_modules(model, CFG.num_experts)
    names = [n for n, _ in found]
    assert len(found) == 3, names
    assert any("block_sparse_moe.gate" in n for n in names)
    assert any("mlp.gate" in n for n in names)
    # A suffix match would also grab these, and their counts are not
    # routed-expert counts.
    assert not any("attn_gate" in n for n in names)


def test_rejects_a_similarly_named_non_router():
    """Qwen2 has a `shared_expert_gate` alongside the real router; a suffix
    match on "gate" would hook it."""
    model = Mixtralish(CFG)
    names = [n for n, _ in capture.find_gate_modules(model, CFG.num_experts)]
    assert all(n.rsplit(".", 1)[-1] == "gate" for n in names)


def test_rejects_a_gate_of_the_wrong_width():
    model = Mixtralish(CFG)
    found = dict(capture.find_gate_modules(model, CFG.num_experts))
    assert not any(n.endswith("shared_gate") for n in found)


def test_finds_nothing_when_the_expert_count_disagrees():
    """Guards the script's own failure path: better to stop than to hook the
    wrong module and capture meaningless counts on a paid box."""
    assert capture.find_gate_modules(Mixtralish(CFG), 999) == []


def test_recorder_counts_match_an_independent_topk():
    model = Mixtralish(CFG, layers=1)
    gates = capture.find_gate_modules(model, CFG.num_experts)
    rec = capture.GateRecorder(CFG, len(gates))
    rec.attach(gates)

    torch.manual_seed(0)
    x = torch.randn(6, CFG.hidden_size)
    expected = []
    with torch.no_grad():
        for _, module in gates:
            logits = module(x)
            expected.append(torch.bincount(
                torch.topk(torch.softmax(logits.float(), -1), CFG.top_k, -1)
                .indices.reshape(-1), minlength=CFG.num_experts).long())
    rec.detach()

    assert int(rec.counts.sum()) == len(gates) * x.shape[0] * CFG.top_k
    for layer, want in enumerate(expected):
        assert torch.equal(rec.counts[layer], want), f"layer {layer}"


def test_recorder_uses_sigmoid_when_the_model_does():
    """DeepSeek-V3 scores with sigmoid. Softmax and sigmoid rank identically per
    row, so this checks the code path runs and totals are right rather than
    asserting a different argmax."""
    cfg = MoEConfig(name="ds", hidden_size=32, intermediate_size=64,
                    num_experts=8, top_k=2, gate_fn="sigmoid", verified=True)
    model = Mixtralish(cfg, layers=1)
    gates = capture.find_gate_modules(model, cfg.num_experts)
    rec = capture.GateRecorder(cfg, len(gates))
    rec.attach(gates)
    with torch.no_grad():
        for _, m in gates:
            m(torch.randn(4, cfg.hidden_size))
    rec.detach()
    assert int(rec.counts.sum()) == len(gates) * 4 * cfg.top_k


def test_reset_clears_between_batches():
    model = Mixtralish(CFG, layers=1)
    gates = capture.find_gate_modules(model, CFG.num_experts)
    rec = capture.GateRecorder(CFG, len(gates))
    rec.attach(gates)
    with torch.no_grad():
        for _, m in gates:
            m(torch.randn(4, CFG.hidden_size))
    first = int(rec.counts.sum())
    rec.reset()
    assert int(rec.counts.sum()) == 0
    with torch.no_grad():
        for _, m in gates:
            m(torch.randn(4, CFG.hidden_size))
    assert int(rec.counts.sum()) == first
    assert rec.snapshot().sum() == first
    rec.detach()


def test_detach_removes_the_hooks():
    model = Mixtralish(CFG, layers=1)
    gates = capture.find_gate_modules(model, CFG.num_experts)
    rec = capture.GateRecorder(CFG, len(gates))
    rec.attach(gates)
    rec.detach()
    with torch.no_grad():
        for _, m in gates:
            m(torch.randn(4, CFG.hidden_size))
    # snapshot() is the host view; it is all zeros because no hook ever fired,
    # and it works before any allocation has happened.
    assert rec.snapshot().sum() == 0
    assert rec.snapshot().shape == (len(gates), CFG.num_experts)


def test_every_builtin_corpus_is_usable():
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "capture_traces.py"
    spec_ = importlib.util.spec_from_file_location("capture_traces", script)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    for name, prompts in mod.CORPORA.items():
        assert prompts, name
        assert all(p.strip() for p in prompts), name


def test_a_written_trace_round_trips_and_is_tiny(tmp_path):
    """End-to-end of what the script writes, minus the model."""
    from moe.routing.traces import write_trace
    counts = np.random.default_rng(0).integers(0, 500, size=(4, 3, CFG.num_experts),
                                               dtype=np.int32)
    meta = dict(trace_id="synthetic-chat-decode", model=CFG.name, hf_repo="none",
                corpus="chat", phase="decode", num_experts=CFG.num_experts,
                top_k=CFG.top_k, captured_at="2026-08-21T00:00:00")
    path = write_trace(tmp_path / "synthetic-chat-decode.npz", counts, meta)
    assert path.stat().st_size < 8 * 1024, "traces must stay committable"

    trace = load_trace(path)
    assert np.array_equal(trace.counts, counts)
    assert len(trace.sha) == 16


def test_trace_provenance_identifies_the_exact_slice(tmp_path):
    from moe.routing.traces import write_trace
    from moe.spec import BenchSpec, RoutingSpec
    counts = np.random.default_rng(1).integers(1, 400, size=(4, 3, CFG.num_experts),
                                               dtype=np.int32)
    meta = dict(trace_id="prov", model=CFG.name, hf_repo="none", corpus="chat",
                phase="decode", num_experts=CFG.num_experts, top_k=CFG.top_k,
                captured_at="now")
    write_trace(tmp_path / "prov.npz", counts, meta)
    ts = TraceSet.load(tmp_path)

    spec_ = BenchSpec(CFG, num_tokens=64, dtype="fp32", seed=2,
                      routing=RoutingSpec("trace", trace_id="prov"))
    prov = ts.provenance(spec_)
    assert prov["trace_sha"] == ts.get("prov")[0].sha
    assert prov["trace_id"] == "prov@b2l2", prov
    assert ts.provenance(spec_.with_(routing=RoutingSpec("uniform"))) == {}


def test_trace_sha_changes_with_content(tmp_path):
    from moe.routing.traces import write_trace
    base = np.ones((2, 2, CFG.num_experts), dtype=np.int32)
    meta = dict(trace_id="a", model="m", hf_repo="r", corpus="c", phase="decode",
                num_experts=CFG.num_experts, top_k=2, captured_at="now")
    p1 = write_trace(tmp_path / "a.npz", base, meta)
    sha1 = load_trace(p1).sha
    p2 = write_trace(tmp_path / "a.npz", base * 2, meta)
    assert load_trace(p2).sha != sha1


# --------------------------------------------------------------------------
# What a gate module actually RETURNS
# --------------------------------------------------------------------------

#: DeepSeek-V2-Lite's real geometry, shrunk in hidden size only. The expert
#: count and top_k are the load-bearing part: 64 and 6 are what make the old
#: `reshape(-1, num_experts)` raise at decode and succeed at some prefills.
DSCFG = MoEConfig(name="ds-lite-ish", hidden_size=32, intermediate_size=64,
                  num_experts=64, top_k=6, num_layers=2, verified=True)


class FakeMoEGate(nn.Module):
    """DeepSeek's `MoEGate`, reduced to the two properties that broke capture.

    Checked against deepseek-ai/DeepSeek-V2-Lite `modeling_deepseek.py`:

      * it is NOT an `nn.Linear`. It holds a bare `nn.Parameter` of shape
        (n_routed_experts, hidden_size), so it has no `out_features` and
        `find_gate_modules` matches it on `weight.shape[0]`;
      * `forward` returns `(topk_idx, topk_weight, aux_loss)`, with aux_loss
        None outside training. Element 0 is int64 [N, top_k] expert IDS.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.randn(cfg.num_experts, cfg.hidden_size))

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, self.cfg.hidden_size)
        logits = nn.functional.linear(flat.float(), self.weight.float())
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.cfg.top_k, dim=-1,
                                           sorted=False)
        return topk_idx, topk_weight, None


class DeepSeekish(nn.Module):
    """`model.layers.N.mlp.gate`, the DeepSeek layout, with a MoEGate in it."""

    def __init__(self, cfg, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            block = nn.Module()
            mlp = nn.Module()
            mlp.gate = FakeMoEGate(cfg)
            block.mlp = mlp
            self.layers.append(block)


def test_a_deepseek_style_gate_is_hooked_despite_having_no_out_features():
    """It is found by `weight.shape[0]`, which is exactly why the old hook fed
    its ids to softmax rather than skipping it."""
    model = DeepSeekish(DSCFG)
    found = capture.find_gate_modules(model, DSCFG.num_experts)
    assert len(found) == 2, [n for n, _ in found]
    assert all(n.endswith("mlp.gate") for n, _ in found)
    assert all(not hasattr(m, "out_features") for _, m in found)


def test_a_deepseek_style_gate_records_the_ids_the_model_itself_chose():
    model = DeepSeekish(DSCFG, layers=1)
    gates = capture.find_gate_modules(model, DSCFG.num_experts)
    rec = capture.GateRecorder(DSCFG, len(gates))
    rec.attach(gates)

    torch.manual_seed(0)
    x = torch.randn(2, 5, DSCFG.hidden_size)
    with torch.no_grad():
        ids, _, _ = gates[0][1](x)
    rec.detach()

    want = torch.bincount(ids.reshape(-1), minlength=DSCFG.num_experts).long()
    assert torch.equal(rec.counts[0], want)
    assert int(rec.counts.sum()) == 2 * 5 * DSCFG.top_k


def test_a_deepseek_style_gate_at_decode_no_longer_raises():
    """4 sequences of top_k=6 is 24 ids against 64 experts. The old
    `.reshape(-1, num_experts)` cannot do that and blew up on the box."""
    model = DeepSeekish(DSCFG, layers=1)
    gates = capture.find_gate_modules(model, DSCFG.num_experts)
    rec = capture.GateRecorder(DSCFG, len(gates))
    rec.attach(gates)
    with torch.no_grad():
        gates[0][1](torch.randn(4, 1, DSCFG.hidden_size))
    rec.detach()
    assert int(rec.counts.sum()) == 4 * DSCFG.top_k


def test_index_values_are_never_softmaxed_into_the_trace():
    """The silent case, and the reason this is not merely a crash bug.

    32 rows of 6 ids is 192 values, which reshapes cleanly to (3, 64) and
    softmaxes without complaint. The old hook wrote THOSE counts. They are a
    well-formed histogram of nothing, and `write_trace` accepts them.
    """
    model = DeepSeekish(DSCFG, layers=1)
    gate = dict(capture.find_gate_modules(model, DSCFG.num_experts))["layers.0.mlp.gate"]
    torch.manual_seed(1)
    x = torch.randn(4, 8, DSCFG.hidden_size)   # 32 rows
    with torch.no_grad():
        ids, _, _ = gate(x)

    # Verbatim reconstruction of the old hook body.
    old = ids.detach().float().reshape(-1, DSCFG.num_experts)
    old_ids = torch.topk(torch.softmax(old, -1), DSCFG.top_k, dim=-1).indices
    old_counts = torch.bincount(old_ids.reshape(-1), minlength=DSCFG.num_experts)

    true_counts = torch.bincount(ids.reshape(-1), minlength=DSCFG.num_experts)
    assert old_counts.sum() != true_counts.sum(), (
        "the old path did not even preserve the number of routed rows")
    assert not torch.equal(old_counts, true_counts)

    rec = capture.GateRecorder(DSCFG, 1)
    rec.attach([("layers.0.mlp.gate", gate)])
    with torch.no_grad():
        gate(x)
    rec.detach()
    assert torch.equal(rec.counts[0], true_counts.long())


def test_a_float_gate_of_the_wrong_width_is_refused():
    cfg = MoEConfig(name="w", hidden_size=8, intermediate_size=8, num_experts=8,
                    top_k=2, verified=True)
    with pytest.raises(ValueError, match="not num_experts=8"):
        routed_ids(torch.randn(4, 5), cfg, where="layers.0.mlp.gate")


def test_an_integer_gate_of_the_wrong_arity_is_refused():
    with pytest.raises(ValueError, match="not top_k=6"):
        routed_ids(torch.zeros(4, 2, dtype=torch.long), DSCFG)


def test_a_gate_output_that_is_not_a_tensor_is_refused():
    with pytest.raises(TypeError, match="not a tensor"):
        routed_ids((None, None, None), DSCFG)


def test_a_gate_output_that_is_not_two_dimensional_is_refused():
    with pytest.raises(ValueError, match="2-D"):
        routed_ids(torch.randn(2, 3, DSCFG.num_experts), DSCFG)


def test_expert_ids_outside_the_configured_range_are_refused():
    """A config that says 64 experts against a model that routes to more is the
    one way the id path can be silently wrong."""
    ids = torch.full((4, DSCFG.top_k), DSCFG.num_experts, dtype=torch.long)
    with pytest.raises(ValueError, match=r"outside \[0, 63\]"):
        routed_ids(ids, DSCFG, check_values=True)


# --------------------------------------------------------------------------
# Padding must not be counted as routed tokens
# --------------------------------------------------------------------------

def test_padded_rows_are_excluded_from_the_counts():
    """A PAD position is scored by the router like any other token, and every
    PAD routes on the same embedding, so they pile onto one set of experts."""
    model = Mixtralish(CFG, layers=1)
    gates = capture.find_gate_modules(model, CFG.num_experts)
    rec = capture.GateRecorder(CFG, len(gates))
    rec.attach(gates)
    mask = torch.tensor([True, True, False, False, False, False])
    rec.set_token_mask(mask)
    with torch.no_grad():
        gates[0][1](torch.randn(6, CFG.hidden_size))
    rec.detach()
    assert int(rec.counts.sum()) == 2 * CFG.top_k, (
        "four padded rows were counted as routed tokens")


def test_clearing_the_token_mask_counts_every_row_again():
    model = Mixtralish(CFG, layers=1)
    gates = capture.find_gate_modules(model, CFG.num_experts)
    rec = capture.GateRecorder(CFG, len(gates))
    rec.attach(gates)
    rec.set_token_mask(torch.tensor([True, False, False, False]))
    with torch.no_grad():
        gates[0][1](torch.randn(4, CFG.hidden_size))
    rec.set_token_mask(None)
    with torch.no_grad():
        gates[0][1](torch.randn(4, CFG.hidden_size))
    rec.detach()
    assert int(rec.counts.sum()) == (1 + 4) * CFG.top_k


def test_a_stale_token_mask_is_refused_rather_than_broadcast():
    """The mask is set per forward pass. A leftover prefill mask meeting a
    decode step would otherwise index the wrong rows or silently broadcast."""
    model = Mixtralish(CFG, layers=1)
    gates = capture.find_gate_modules(model, CFG.num_experts)
    rec = capture.GateRecorder(CFG, len(gates))
    rec.attach(gates)
    rec.set_token_mask(torch.ones(16, dtype=torch.bool))
    with pytest.raises(ValueError, match="token mask covers 16 rows"):
        with torch.no_grad():
            gates[0][1](torch.randn(4, CFG.hidden_size))
    rec.detach()


# --------------------------------------------------------------------------
# The batch axis must carry more than one sample
# --------------------------------------------------------------------------

def test_the_old_modulo_indexing_collapsed_all_sixteen_batches_into_one():
    """`prompts[(b*batch_size + i) % len(prompts)]` with a 4-prompt corpus and
    batch_size 4 is `prompts[i]` for every b."""
    small = capture.CORPORA["chat"][:4]
    old = [[small[(b * 4 + i) % len(small)] for i in range(4)] for b in range(16)]
    assert capture.distinct_batch_count(old) == 1

    new = capture.build_batches(capture.CORPORA["chat"], 16, 4)
    assert capture.distinct_batch_count(new) == 16


def test_every_builtin_corpus_covers_the_default_run_without_repeating():
    """16 batches of 4 needs 64 distinct prompts. Anything less turns the batch
    axis into a regrouping of the same text."""
    for name, prompts in capture.CORPORA.items():
        assert len(set(prompts)) == len(prompts), f"{name} has duplicates"
        assert len(prompts) >= 16 * 4, f"{name} has only {len(prompts)}"
        built = capture.build_batches(prompts, 16, 4)
        assert len({p for batch in built for p in batch}) == 64, name


def test_build_batches_is_deterministic_and_the_seed_moves_it():
    a = capture.build_batches(capture.CORPORA["math"], 8, 4, seed=3)
    b = capture.build_batches(capture.CORPORA["math"], 8, 4, seed=3)
    c = capture.build_batches(capture.CORPORA["math"], 8, 4, seed=4)
    assert a == b
    assert a != c


def test_no_batch_repeats_a_prompt_inside_itself():
    built = capture.build_batches(capture.CORPORA["prose"], 16, 4, seed=1)
    for batch in built:
        assert len(set(batch)) == len(batch), batch


def test_a_short_corpus_regroups_instead_of_repeating_one_batch():
    prompts = capture.CORPORA["code"][:8]
    built = capture.build_batches(prompts, 6, 4, seed=0)
    assert len(built) == 6
    assert capture.distinct_batch_count(built) > 1, (
        "reshuffling between passes must at least regroup the prompts")


def test_a_corpus_too_small_for_one_batch_is_refused():
    with pytest.raises(ValueError, match="fewer than one batch"):
        capture.build_batches(["only", "three", "here"], 4, 4)


def test_identical_batches_are_refused_before_anything_is_written():
    one = np.random.default_rng(0).integers(0, 50, size=(3, CFG.num_experts),
                                            dtype=np.int32)
    repeated = np.broadcast_to(one, (16, 3, CFG.num_experts))
    with pytest.raises(SystemExit, match="identical"):
        capture.assert_batches_differ(repeated)


def test_a_varied_batch_axis_is_accepted():
    counts = np.random.default_rng(0).integers(0, 50, size=(16, 3, CFG.num_experts),
                                               dtype=np.int32)
    capture.assert_batches_differ(counts)          # must not raise
    capture.assert_batches_differ(counts[:1])      # a single batch cannot differ


# --------------------------------------------------------------------------
# Padding side and positions
# --------------------------------------------------------------------------

def test_decode_pads_on_the_left_and_prefill_on_the_right():
    """Right padding plus `logits[:, -1]` asks the model what follows the PAD."""
    assert capture.padding_side_for("decode") == "left"
    assert capture.padding_side_for("prefill") == "right"


def test_position_ids_skip_the_padding_they_are_given():
    attn = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    pos = capture.position_ids_from_mask(attn)
    assert pos.tolist() == [[0, 0, 0, 1], [0, 1, 2, 3]]
    # The first real token of a left-padded row must be position 0, not 2.
    assert pos[0, 2].item() == 0


# --------------------------------------------------------------------------
# The torchvision block. Step 7 lost both of its captures on 2026-09-01 to a
# torch/torchvision ABI mismatch that transformers reported as a MODEL problem.
# --------------------------------------------------------------------------

def _capture_traces_module():
    spec = importlib.util.spec_from_file_location(
        "capture_traces",
        Path(__file__).resolve().parents[1] / "scripts" / "capture_traces.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_disabling_torchvision_makes_it_unimportable_and_undetectable(
        monkeypatch, capsys):
    """Both halves matter and they fail differently.

    `import torchvision` must raise ImportError, which is what transformers
    already handles. And `find_spec` must return None rather than raising,
    because transformers' _is_package_available calls it FIRST and a raise there
    would replace one crash with another. On CPython 3.12 a None entry in
    sys.modules gives exactly that pair; this pins it, because the whole fix
    rests on that behaviour and it is not something the stdlib documents
    prominently.
    """
    import sys as _sys
    monkeypatch.delitem(_sys.modules, "torchvision", raising=False)
    mod = _capture_traces_module()
    mod._disable_torchvision()

    assert _sys.modules["torchvision"] is None
    with pytest.raises(ImportError):
        importlib.import_module("torchvision")
    assert importlib.util.find_spec("torchvision") is None, (
        "find_spec must report absence cleanly; if it raises, transformers' "
        "availability check crashes instead of degrading")
    assert "torchvision disabled" in capsys.readouterr().out


def test_a_working_torchvision_is_left_alone(monkeypatch):
    """The block is for a MISMATCHED torchvision, not a hostile act. If one is
    already imported and working, taking it away would break a caller that has
    every right to it."""
    import sys as _sys
    sentinel = object()
    monkeypatch.setitem(_sys.modules, "torchvision", sentinel)
    mod = _capture_traces_module()
    mod._disable_torchvision()
    assert _sys.modules["torchvision"] is sentinel
