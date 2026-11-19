"""scripts/prod/moe_capture.py off the GPU: the ring's host logic on the numpy backend (the kernel's semantics),
drains, overflow, the reset, the step log; the binder against fake vLLM classes shaped as v0.27.1's
(MoERunner.layer_id / _quant_method / router.set_capture_fn); and static checks of the graph-safety rules."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "scripts" / "prod", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import moe_capture as MC  # noqa: E402
import prod_fakes as F  # noqa: E402
import prodlib as PL  # noqa: E402


def _cap(tmp_path, **kw):
    kw.setdefault("ids_cap", 4096)
    kw.setdefault("rec_cap", 64)
    return MC.Capture(tmp_path, top_k=4, n_moe_layers=2, backend="numpy", **kw)


def test_ring_round_trip_across_shards(tmp_path):
    cap = _cap(tmp_path, drain_every=1)
    cap.reset()
    rng = np.random.default_rng(0)
    sent = []
    for step, n in enumerate((3, 5, 1)):
        cap.on_step(num_tokens=n, num_tokens_padded=n, cudagraph_mode="NONE", dummy=False)
        for layer in (1, 2):
            ids = F.random_ids(rng, n, 16, 4)
            cap.launch(layer, ids)
            sent.append((layer, n, ids, step))
    cap.close()
    recs = PL.decode_shards(sorted((tmp_path / "ring").glob("*.npz")), 4)
    assert [(r["layer"], r["n"]) for r in recs] == [(s[0], s[1]) for s in sent]
    for r, s in zip(recs, sent, strict=True):
        assert np.array_equal(r["ids"], s[2])
    steps = PL.read_jsonl(tmp_path / "steps.jsonl")
    # each step's records land in the shard its row names (drain_every=1: a new shard per step)
    assert [s["shard_next"] for s in steps] == [0, 1, 2]
    assert [r["shard"] for r in recs] == [f"shard_{i:05d}.npz" for i in (0, 0, 1, 1, 2, 2)]
    man = json.loads((tmp_path / "capture_manifest.json").read_text())
    assert man["total_records"] == 6 and man["marker_kernel"] == MC.MARKER_KERNEL


def test_reset_drops_warmup_and_capture_calls(tmp_path):
    cap = _cap(tmp_path)
    cap.launch(1, np.zeros((7, 4), np.int64))          # warmup, before reset
    assert not cap.enabled
    cap.on_step(num_tokens=7, num_tokens_padded=8, cudagraph_mode="x", dummy=True)   # not logged before reset
    cap.reset()
    cap.on_step(num_tokens=2, num_tokens_padded=2, cudagraph_mode="x", dummy=False)
    cap.launch(1, np.ones((2, 4), np.int64))
    cap.launch(2, np.ones((2, 4), np.int64))
    cap.close()
    recs = PL.decode_shards(sorted((tmp_path / "ring").glob("*.npz")), 4)
    assert [r["n"] for r in recs] == [2, 2]
    assert len(PL.read_jsonl(tmp_path / "steps.jsonl")) == 1


def test_overflow_fails_loudly_and_poisons_the_shard(tmp_path):
    cap = _cap(tmp_path, ids_cap=40)
    cap.reset()
    cap.launch(1, np.zeros((8, 4), np.int64))           # 32 ids: fits
    cap.launch(2, np.zeros((8, 4), np.int64))           # 64 > 40: not written, counters advance
    with pytest.raises(PL.KitError, match="G-OVF"):
        cap.drain()
    with pytest.raises(PL.RingError, match="overflowed"):
        PL.decode_shards(sorted((tmp_path / "ring").glob("*.npz")), 4)


def test_record_overflow_is_caught_too(tmp_path):
    cap = _cap(tmp_path, rec_cap=2)
    cap.reset()
    for _ in range(3):
        cap.launch(1, np.zeros((1, 4), np.int64))
    with pytest.raises(PL.KitError, match="records 3 > capacity 2"):
        cap.drain()


def test_a_step_larger_than_the_ring_is_refused_before_it_runs(tmp_path):
    cap = _cap(tmp_path, ids_cap=100)
    cap.reset()
    with pytest.raises(PL.KitError, match="one step of T_pad 13"):
        cap.on_step(num_tokens=13, num_tokens_padded=13, cudagraph_mode="x", dummy=False)


def test_the_drain_is_scheduled_before_the_ring_can_fill(tmp_path):
    cap = _cap(tmp_path, ids_cap=1000, drain_every=10 ** 6)
    cap.reset()
    for _ in range(20):                                   # 2 layers x 4 x 50 = 400 ids per step; fill 0.5 -> 500
        cap.on_step(num_tokens=50, num_tokens_padded=50, cudagraph_mode="x", dummy=False)
        cap.launch(1, np.zeros((50, 4), np.int64))
        cap.launch(2, np.zeros((50, 4), np.int64))
    cap.close()
    recs = PL.decode_shards(sorted((tmp_path / "ring").glob("*.npz")), 4)
    assert len(recs) == 40                                  # nothing lost, no overflow
    assert cap.shard_idx > 10


def test_decode_shards_refuses_gaps_and_misaligned_offsets(tmp_path):
    cap = _cap(tmp_path)
    cap.reset()
    cap.launch(1, np.zeros((2, 4), np.int64))
    cap.launch(2, np.zeros((2, 4), np.int64))
    p = cap.drain()
    with np.load(p) as z:
        ids, recs, meta = z["ids"], z["recs"].copy(), z["meta"]
    recs[1, 3] = 5
    np.savez(p, ids=ids, recs=recs, meta=meta)
    with pytest.raises(PL.RingError, match="seq"):
        PL.decode_shards([p], 4)
    recs[1, 3] = 1
    recs[1, 2] = 3
    np.savez(p, ids=ids, recs=recs, meta=meta)
    with pytest.raises(PL.RingError, match="starts at"):
        PL.decode_shards([p], 4)


# ---------------------------------------------------------------- the binder, against fake vLLM classes

class FakeBaseRouter:
    def __init__(self):
        self.capture_fn = None

    def set_capture_fn(self, fn):           # router/base_router.py:185
        self.capture_fn = fn


class TritonExperts:
    pass


class FakeRunner:
    def __init__(self, layer_id, *, monolithic=False, backend="TRITON", router=None):
        self.layer_id = layer_id
        self.router = router or FakeBaseRouter()
        self._quant_method = SimpleNamespace(is_monolithic=monolithic,
                                             unquantized_backend=SimpleNamespace(value=backend),
                                             moe_kernel=SimpleNamespace(impl=SimpleNamespace(fused_experts=TritonExperts())))


class FakeModel:
    def __init__(self, mods):
        self.mods = mods

    def modules(self):
        yield SimpleNamespace()               # non-MoE modules are skipped
        yield from self.mods


def test_bind_sets_one_capture_fn_per_moe_layer_and_it_records_that_layer(tmp_path):
    cap = _cap(tmp_path)
    mods = [FakeRunner(1), FakeRunner(2)]
    rows = MC.bind(FakeModel(mods), cap, runner_cls=FakeRunner, router_cls=FakeBaseRouter)
    assert [r["layer"] for r in rows] == [1, 2] and {r["backend"] for r in rows} == {"TRITON"}
    assert rows[0]["experts_cls"] == "TritonExperts"
    cap.reset()
    mods[1].router.capture_fn(np.full((3, 4), 2, np.int64))   # what base_router.py:296-297 does
    recs = PL.decode_shards([cap.drain()], 4)
    assert recs[0]["layer"] == 2 and recs[0]["n"] == 3


@pytest.mark.parametrize("mods, why", [
    ([FakeRunner(1, monolithic=True), FakeRunner(2)], "monolithic"),
    ([FakeRunner(1), FakeRunner(2, router=SimpleNamespace(capture_fn=None, set_capture_fn=lambda f: None))],
     "not a BaseRouter"),
    ([FakeRunner(1)], "bound 1 MoE layers"),
    ([FakeRunner(2), FakeRunner(1)], "ascending"),
])
def test_bind_refuses_what_the_hook_would_not_cover(tmp_path, mods, why):
    with pytest.raises(PL.KitError, match=why):
        MC.bind(FakeModel(mods), _cap(tmp_path), runner_cls=FakeRunner, router_cls=FakeBaseRouter)


def test_bind_refuses_a_router_that_already_carries_a_capture_fn(tmp_path):
    r = FakeRunner(1)
    r.router.capture_fn = lambda ids: None   # vLLM's --enable-return-routed-experts binder got there first
    with pytest.raises(PL.KitError, match="already carries"):
        MC.bind(FakeModel([r, FakeRunner(2)]), _cap(tmp_path), runner_cls=FakeRunner, router_cls=FakeBaseRouter)


def test_install_refuses_the_multiprocess_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    with pytest.raises(PL.KitError, match="MULTIPROCESSING"):
        MC.install(tmp_path, top_k=6, n_moe_layers=27)


# ---------------------------------------------------------------- graph-safety rules, statically

def test_the_kernel_takes_everything_that_varies_as_unspecialised_runtime_arguments():
    src = inspect.getsource(MC._build_kernel)
    assert 'do_not_specialize=["n_tok", "layer", "stride0", "stride1", "ids_cap", "rec_cap"]' in src
    assert "tl.atomic_add(ctr_ptr, 1)" in src and "tl.atomic_add(ctr_ptr + 1, numel)" in src
    assert "if ok:" in src and "mask=m" in src           # an overflowing call writes nothing


def test_the_capture_fn_never_syncs_the_host():
    for fn in (MC.Capture.launch, MC.Capture.capture_fn):
        src = inspect.getsource(fn)
        for bad in (".item(", ".cpu(", "synchronize", ".tolist(", "numpy()"):
            assert bad not in src.split("if self.backend != \"torch\":")[-1], (fn.__name__, bad)
    drain = inspect.getsource(MC.Capture.drain)
    assert "_capturing()" in drain and "raise PL.KitError(\"drain() inside a stream capture\")" in drain


def test_the_marker_encodes_its_layer_in_grid_y_and_only_program_zero_works():
    src = inspect.getsource(MC._build_kernel)
    assert "if tl.program_id(1) == 0:" in src
    assert "kernel()[marker_grid(layer)]" in inspect.getsource(MC.Capture.launch)


def test_warm_gates_one_compiled_variant_per_ids_dtype():
    class Old:
        cache = {0: {"a": 1, "b": 2}}

    class New:
        device_caches = {0: ({"a": 1, "b": 2}, "target", "backend", "binder")}
    assert MC.kernel_variants(Old()) == 2 and MC.kernel_variants(New()) == 2 and MC.kernel_variants(object()) is None
    MC.check_variants(2, 2)
    for got in (1, 3, None):
        with pytest.raises(PL.KitError, match="G-VARIANTS"):
            MC.check_variants(got, 2)
