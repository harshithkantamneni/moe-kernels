"""CPU tests for the trace-capture machinery.

The capture script itself needs a GPU and a 93 GB model download, so it cannot
be run here. What CAN be verified on a laptop is everything that would waste
that expensive session: locating the gate modules in an unfamiliar model layout,
accumulating counts correctly, and writing a well-formed trace file.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
