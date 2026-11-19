"""The routing capture hook, imported INSIDE the vLLM process (VLLM_ENABLE_V1_MULTIPROCESSING=0, so the worker, its
model runner and the model live in this interpreter and class patches reach them).

WHAT IT RECORDS, per routed MoE call: the call's final topk_ids (logical ids, before any EPLB mapping), its layer
and its row count n (the PADDED T the forward runs at: padded rows are routed and priced, CAPTURE.md); per engine
step: step id, unpadded T, padded T, the cudagraph mode the dispatcher chose.

WHERE IT HOOKS (vLLM v0.27.1, file:line of the tagged source):
  - router/base_router.py:185 `BaseRouter.set_capture_fn(capture_fn)`; base_router.py:296-297 calls it with the
    logical topk_ids right after `_compute_routing`, BEFORE the EPLB map (300) and the dtype conversion (303).
  - v1/worker/gpu_model_runner.py:7739-7772 `_bind_routed_experts_capturer`: vLLM's own routed-experts capture
    binds the same hook, per `MoERunner` module (`module.layer_id`), on `module.router` when the quant method is
    not monolithic (a monolithic kernel takes `fused_experts.set_capture_fn` instead). We bind the same way and
    REFUSE a monolithic quant method (the Triton path is modular) or a router that already carries a capture fn
    (vLLM's --enable-return-routed-experts would be silently overwritten).
  - v1/worker/gpu_worker.py:679 `Worker.compile_or_warm_up_model`: warmup `_dummy_run`s (706-708), kernel_warmup
    (713), then `capture_model()` (716-717). We wrap it and bind FIRST, so the warmup runs compile our kernel and
    the graph capture records our launch into every captured graph. (vLLM's own binder runs even later, from
    initialize_from_config, gpu_worker.py:667-668, after the profile run: the router call sits inside the opaque
    MoE custom op, so a capture fn bound after torch.compile still runs. VERIFIED ON THE VM by gate G-REC.)
  - v1/worker/gpu_model_runner.py:3932 `_determine_batch_execution_and_padding(num_tokens=...)` returns
    (cudagraph_mode, batch_descriptor, ...) and the forward runs at batch_descriptor.num_tokens (execute_model
    calls it at 4271 with num_tokens=, padded T read at 4289); _dummy_run (5830) calls it too (5936, keyword
    num_tokens=), so _dummy_run is wrapped to mark those calls as dummies.
  - routed_experts_capturer.py:120-215 (vLLM's own capturer) writes into a [max_tokens, layers, k] int32 buffer
    with a python slice assignment per call and D2Hs it every step (gpu_model_runner.py:3770, 4811). We do not
    reuse it: it keeps one step only, needs a per-step D2H, and its buffer is sized by unpadded tokens.

GRAPH SAFETY (CRITIQUE M8):
  - the capture fn launches ONE Triton kernel. Everything that varies per call is a kernel ARGUMENT: the topk_ids
    pointer and strides, n and the layer as runtime ints marked do_not_specialize (one compiled variant for every
    n: no JIT compile can fall inside a stream capture), the ring/counter pointers fixed for the process. Under
    graph capture the launch is recorded with its arguments; each replay re-executes the atomics, so every
    replayed call reserves a fresh slot.
  - no host sync inside the captured region: n comes from topk_ids.shape (host metadata), nothing calls .item(),
    .cpu() or synchronize in the capture fn. The ONLY syncs are in `drain()`, called between engine steps from
    the padding wrapper (never while a stream is capturing: asserted).
  - slot reservation is an atomic on a device counter (single program, so no contention); a call that would not
    fit writes NOTHING but still advances the counters, so the host sees counter > capacity at the next drain and
    the run FAILS (G-OVF). The drain is also scheduled from the step log's padded T so the ring never fills.
  - counters are reset after LLM() returns (`reset()`): warmup, profile and capture calls are excluded.
  - THE LAYER IS IN THE LAUNCH GRID (review 1, MAJOR 8): the kernel launches grid (1, layer + 1) and every program
    with program_id(1) > 0 returns at once, so nsys (and the torch profiler) record gridY = layer + 1 on every
    marker, graph replays included; nsys_extract's G-PAIR holds every call's gridY to its ring record's layer + 1.
    The grid is a launch parameter, not a specialisation key: no new compile. No NVTX range is pushed per call
    (graph replays would carry the capture-time range only; the secondary NVTX alignment claim is withdrawn).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from . import prodlib as PL  # type: ignore
except ImportError:  # run as a plain module from scripts/prod
    import prodlib as PL  # type: ignore

#: device ring capacity (int16 expert ids) and record capacity between two drains
IDS_CAP = 64 * 2 ** 20
REC_CAP = 2 ** 20
#: drain when the predicted ids in the ring would pass this share of capacity on the next step
DRAIN_FILL = 0.5
#: or at most every this many steps (bounded host memory per shard)
DRAIN_EVERY = 256
MARKER_KERNEL = "_moe_prod_capture_kernel"


# ------------------------------------------------------------------------------------------------------------
# the device kernel (built lazily: the laptop has no Triton)
# ------------------------------------------------------------------------------------------------------------

_KERNEL = None


def _build_kernel():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["n_tok", "layer", "stride0", "stride1", "ids_cap", "rec_cap"])
    def _moe_prod_capture_kernel(topk_ptr, ctr_ptr, ring_ptr, rec_ptr, n_tok, layer, stride0, stride1,
                                 ids_cap, rec_cap, K: tl.constexpr, BLOCK: tl.constexpr):
        # the grid's Y extent carries the layer (gridY = layer + 1): only program (0, 0) works, every other exits
        if tl.program_id(1) == 0:
            numel = n_tok * K
            rec = tl.atomic_add(ctr_ptr, 1)              # counter[0]: records reserved since the last drain
            off = tl.atomic_add(ctr_ptr + 1, numel)      # counter[1]: ids reserved since the last drain
            ok = ((off + numel) <= ids_cap) & (rec < rec_cap)
            for start in range(0, numel, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                m = (offs < numel) & ok
                row = offs // K
                col = offs % K
                v = tl.load(topk_ptr + row * stride0 + col * stride1, mask=m, other=0)
                tl.store(ring_ptr + off + offs, v.to(tl.int16), mask=m)
            if ok:
                base = rec_ptr + rec * 4
                tl.store(base + 0, layer)
                tl.store(base + 1, n_tok)
                tl.store(base + 2, off)
                tl.store(base + 3, rec)

    return _moe_prod_capture_kernel


def kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    return _KERNEL


def marker_grid(layer: int) -> tuple[int, int]:
    """The marker's launch grid: (1, layer + 1); the warm launches (layer -1) use (1, 1)."""
    return (1, max(int(layer), 0) + 1)


def reference_capture(topk, ctr, ring, recs, n_tok: int, layer: int, ids_cap: int, rec_cap: int, K: int) -> None:
    """The kernel's semantics in numpy (CPU tests drive the host logic through this)."""
    numel = n_tok * K
    rec = int(ctr[0])
    off = int(ctr[1])
    ctr[0] += 1
    ctr[1] += numel
    if off + numel <= ids_cap and rec < rec_cap:
        ring[off:off + numel] = np.asarray(topk, dtype=np.int64).reshape(-1)[:numel].astype(np.int16)
        recs[rec] = (layer, n_tok, off, rec)


# ------------------------------------------------------------------------------------------------------------
# the capture state
# ------------------------------------------------------------------------------------------------------------

@dataclass
class StepLog:
    path: Path
    fh: object = None

    def write(self, row: dict) -> None:
        if self.fh is None:
            self.fh = open(self.path, "a")
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


@dataclass
class Capture:
    """Device ring + host drain. `backend` is "torch" on the VM; tests pass "numpy" and drive `launch` directly."""
    out_dir: Path
    top_k: int
    n_moe_layers: int
    ids_cap: int = IDS_CAP
    rec_cap: int = REC_CAP
    drain_every: int = DRAIN_EVERY
    backend: str = "torch"
    enabled: bool = False             # True after reset(): steps are logged, drains happen
    shard_idx: int = 0
    step_id: int = 0
    steps_since_drain: int = 0
    predicted_ids: int = 0
    total_records: int = 0
    context: dict = field(default_factory=dict)
    bound_layers: list = field(default_factory=list)
    manifest: dict = field(default_factory=dict)
    in_dummy: bool = False
    #: the tag a scheduled (mid-run) drain gives its shard; serve_capture sets "profiler-window" while vLLM's
    #: torch profiler runs, so every record of the window is in a shard with that tag
    drain_tag: str = "periodic"

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        (self.out_dir / "ring").mkdir(parents=True, exist_ok=True)
        self.steps = StepLog(self.out_dir / "steps.jsonl")
        if self.backend == "torch":
            import torch
            self.torch = torch
            dev = torch.device("cuda", torch.cuda.current_device())
            self.ctr = torch.zeros(2, dtype=torch.int32, device=dev)
            self.ring = torch.zeros(self.ids_cap, dtype=torch.int16, device=dev)
            self.recs = torch.zeros((self.rec_cap, 4), dtype=torch.int32, device=dev)
        else:
            self.torch = None
            self.ctr = np.zeros(2, dtype=np.int64)
            self.ring = np.zeros(self.ids_cap, dtype=np.int16)
            self.recs = np.zeros((self.rec_cap, 4), dtype=np.int64)

    # -- the per-call launch (inside the forward; graph-safe) -------------------------------------------------
    def launch(self, layer: int, topk_ids) -> None:
        n = int(topk_ids.shape[0])            # host metadata, no sync
        if self.backend != "torch":
            reference_capture(topk_ids, self.ctr, self.ring, self.recs, n, layer, self.ids_cap, self.rec_cap,
                              self.top_k)
            return
        if topk_ids.dim() != 2 or topk_ids.shape[1] != self.top_k:
            raise PL.KitError(f"capture: topk_ids shape {tuple(topk_ids.shape)}, top_k {self.top_k}")
        if n:
            kernel()[marker_grid(layer)](topk_ids, self.ctr, self.ring, self.recs, n, int(layer), topk_ids.stride(0),
                                         topk_ids.stride(1), self.ids_cap, self.rec_cap, K=self.top_k, BLOCK=1024)

    def capture_fn(self, layer: int):
        def _fn(topk_ids, _layer=int(layer), _cap=self):
            _cap.launch(_layer, topk_ids)
        return _fn

    # -- host side -------------------------------------------------------------------------------------------
    def _capturing(self) -> bool:
        return self.torch is not None and self.torch.cuda.is_current_stream_capturing()

    def _sync(self) -> None:
        if self.torch is not None:
            self.torch.cuda.synchronize()

    def reset(self) -> None:
        """After LLM() init: drop everything the warmup/profile/capture runs wrote, start logging steps."""
        if self._capturing():
            raise PL.KitError("reset() inside a stream capture")
        self._sync()
        if self.torch is not None:
            self.ctr.zero_()
            self.torch.cuda.synchronize()
            self.torch.cuda.nvtx.mark("moe_capture:reset")
        else:
            self.ctr[:] = 0
        self.manifest["reset_monotonic_ns"] = time.monotonic_ns()
        self.enabled = True
        self.step_id = 0
        self.steps_since_drain = 0
        self.predicted_ids = 0

    def drain(self, tag: str = "") -> Path | None:
        """Copy [0, counter) of the ring and the records to an npz shard and zero the counters. Syncs the device:
        called between engine steps only (every DRAIN_EVERY steps or when the ring would pass DRAIN_FILL)."""
        if self._capturing():
            raise PL.KitError("drain() inside a stream capture")
        self._sync()
        if self.torch is not None:
            ctr = self.ctr.cpu().numpy().astype(np.int64)
        else:
            ctr = np.asarray(self.ctr, dtype=np.int64).copy()
        n_rec, n_ids = int(ctr[0]), int(ctr[1])
        overflow = ""
        if n_rec > self.rec_cap:
            overflow = f"records {n_rec} > capacity {self.rec_cap}"
        if n_ids > self.ids_cap:
            overflow = (overflow + "; " if overflow else "") + f"ids {n_ids} > capacity {self.ids_cap}"
        k_rec, k_ids = min(n_rec, self.rec_cap), min(n_ids, self.ids_cap)
        if self.torch is not None:
            ids = self.ring[:k_ids].cpu().numpy()
            recs = self.recs[:k_rec].cpu().numpy().astype(np.int32)
        else:
            ids = np.asarray(self.ring[:k_ids]).copy()
            recs = np.asarray(self.recs[:k_rec]).astype(np.int32)
        meta = {"schema": PL.RING_SCHEMA, "shard": self.shard_idx, "n_records": int(k_rec), "n_ids": int(k_ids),
                "counters": [n_rec, n_ids], "overflow": overflow, "base_record": self.total_records,
                "step_after": self.step_id, "tag": tag, "monotonic_ns": time.monotonic_ns()}
        path = self.out_dir / "ring" / f"shard_{self.shard_idx:05d}.npz"
        np.savez(path, ids=ids, recs=recs, meta=np.array(json.dumps(meta)))
        self.shard_idx += 1
        self.total_records += k_rec
        if self.torch is not None:
            self.ctr.zero_()
        else:
            self.ctr[:] = 0
        self.steps_since_drain = 0
        self.predicted_ids = 0
        if overflow:
            raise PL.KitError(f"G-OVF FAIL: the capture ring overflowed ({overflow}); shard {path.name} marks it")
        return path

    def on_step(self, *, num_tokens: int, num_tokens_padded: int, cudagraph_mode: str, dummy: bool) -> None:
        """Called by the padding wrapper before the step's forward is launched."""
        if not self.enabled:
            return
        need = self.n_moe_layers * self.top_k * int(num_tokens_padded)
        if need > self.ids_cap or self.n_moe_layers > self.rec_cap:
            raise PL.KitError(f"G-OVF FAIL: one step of T_pad {num_tokens_padded} needs {need} ids, the ring holds "
                              f"{self.ids_cap}")
        if (self.steps_since_drain >= self.drain_every
                or self.predicted_ids + need > DRAIN_FILL * self.ids_cap
                or (self.steps_since_drain + 1) * self.n_moe_layers > DRAIN_FILL * self.rec_cap):
            if not self._capturing():
                self.drain(tag=self.drain_tag)
        row = {"step": self.step_id, "T": int(num_tokens), "T_pad": int(num_tokens_padded),
               "cg_mode": str(cudagraph_mode), "dummy": bool(dummy), "shard_next": self.shard_idx,
               "t_ns": time.monotonic_ns(), **self.context}
        self.steps.write(row)
        self.step_id += 1
        self.steps_since_drain += 1
        self.predicted_ids += need

    def close(self) -> None:
        if self.enabled:
            self.drain(tag="final")
        self.steps.close()
        self.manifest.update({"bound_layers": self.bound_layers, "shards": self.shard_idx,
                              "total_records": self.total_records, "top_k": self.top_k,
                              "n_moe_layers": self.n_moe_layers, "ids_cap": self.ids_cap, "rec_cap": self.rec_cap,
                              "marker_kernel": MARKER_KERNEL})
        (self.out_dir / "capture_manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))


# ------------------------------------------------------------------------------------------------------------
# binding (vLLM classes injectable for the fake-vLLM test)
# ------------------------------------------------------------------------------------------------------------

def bind(model, cap: Capture, *, runner_cls=None, router_cls=None) -> list[dict]:
    """Bind cap's capture fn on every routed MoE layer of `model`, exactly as gpu_model_runner.py:7754-7772 does
    for vLLM's own capturer; refuse what that path would not cover. Returns one row per bound layer."""
    if runner_cls is None or router_cls is None:
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
        runner_cls, router_cls = runner_cls or MoERunner, router_cls or BaseRouter
    rows = []
    for module in model.modules():
        if not isinstance(module, runner_cls):
            continue
        layer_id = int(module.layer_id)
        qm = module._quant_method
        backend = getattr(qm, "unquantized_backend", None)
        backend = getattr(backend, "value", backend)
        impl = getattr(getattr(qm, "moe_kernel", None), "impl", None)
        experts = type(getattr(impl, "fused_experts", None)).__name__
        if qm.is_monolithic:
            raise PL.KitError(f"layer {layer_id}: monolithic MoE kernel {experts}: the router hook is not called")
        if not isinstance(module.router, router_cls):
            raise PL.KitError(f"layer {layer_id}: router {type(module.router).__name__} is not a BaseRouter")
        if getattr(module.router, "capture_fn", None) is not None:
            raise PL.KitError(f"layer {layer_id}: the router already carries a capture fn (routed-experts return?)")
        module.router.set_capture_fn(cap.capture_fn(layer_id))
        rows.append({"layer": layer_id, "backend": str(backend), "experts_cls": experts,
                     "quant_method": type(qm).__name__})
    if len(rows) != cap.n_moe_layers:
        raise PL.KitError(f"bound {len(rows)} MoE layers, the model has {cap.n_moe_layers}")
    if [r["layer"] for r in rows] != sorted(r["layer"] for r in rows):
        raise PL.KitError("MoE layers are not in ascending layer_id order: the step join assumes forward order")
    cap.bound_layers = rows
    return rows


def warm(cap: Capture, sizes, dtypes=("int32", "int64")) -> None:
    """Compile and run the kernel once per capture size and topk_ids dtype BEFORE any graph capture, then zero the
    counters (CRITIQUE M8: warm every capture size). do_not_specialize makes this one compile; it proves it."""
    if cap.backend != "torch":
        return
    torch = cap.torch
    for dt in dtypes:
        for n in sorted(set(int(s) for s in sizes)):
            x = torch.zeros((n, cap.top_k), dtype=getattr(torch, dt), device=cap.ctr.device)
            cap.launch(-1, x)
    torch.cuda.synchronize()
    cap.ctr.zero_()
    torch.cuda.synchronize()
    got = kernel_variants(kernel())
    cap.manifest["kernel_variants"] = got
    cap.manifest["kernel_variants_expected"] = len(dtypes)
    check_variants(got, len(dtypes))


def kernel_variants(jit_fn) -> int | None:
    """Compiled variants of a Triton JITFunction across devices: Triton <= 3.1 keeps `cache` {device: {key: kernel}};
    later Triton keeps `device_caches` {device: (kernel_cache, ...)}. None when neither reads."""
    try:
        dc = getattr(jit_fn, "device_caches", None)
        if dc:
            return int(sum(len(v[0]) for v in dc.values()))
        c = getattr(jit_fn, "cache", None)
        if c is not None:
            return int(sum(len(v) for v in c.values()))
    except (TypeError, IndexError, AttributeError):
        return None
    return None


def check_variants(got: int | None, expected: int) -> None:
    """The A3 gate (review 1, MINOR): after warm(), exactly one compiled variant per topk_ids dtype; any other count
    (or an unreadable cache) means a JIT compile could fall inside a graph capture: stop before capture."""
    if got != expected:
        raise PL.KitError(f"G-VARIANTS FAIL: the capture kernel has {got} compiled variants after warm(), want "
                          f"{expected} (one per topk_ids dtype): do_not_specialize did not hold")


# ------------------------------------------------------------------------------------------------------------
# install: patch vLLM's classes (before LLM() is constructed)
# ------------------------------------------------------------------------------------------------------------

_CAP: Capture | None = None


def current() -> Capture:
    if _CAP is None:
        raise PL.KitError("moe_capture.install() was not called")
    return _CAP


def install(out_dir, *, top_k: int, n_moe_layers: int, warm_sizes=None, drain_every: int = DRAIN_EVERY,
            ids_cap: int = IDS_CAP, rec_cap: int = REC_CAP) -> None:
    """Patch Worker.compile_or_warm_up_model (bind + warm first), GPUModelRunner._determine_batch_execution_and_
    padding (step log + drain) and GPUModelRunner._dummy_run (marks dummies). The Capture is created lazily inside
    the worker (the CUDA device is current there)."""
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise PL.KitError("VLLM_ENABLE_V1_MULTIPROCESSING must be 0: the hook lives in this process")
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    orig_warm = Worker.compile_or_warm_up_model
    orig_pad = GPUModelRunner._determine_batch_execution_and_padding
    orig_dummy = GPUModelRunner._dummy_run

    def compile_or_warm_up_model(self, *a, **kw):
        global _CAP
        _CAP = Capture(Path(out_dir), top_k=top_k, n_moe_layers=n_moe_layers, drain_every=drain_every,
                       ids_cap=ids_cap, rec_cap=rec_cap)
        bind(self.model_runner.model, _CAP)
        cg = getattr(self.vllm_config.compilation_config, "cudagraph_capture_sizes", None) or []
        _CAP.manifest["cudagraph_capture_sizes"] = list(cg)
        _CAP.manifest["cudagraph_mode"] = str(getattr(self.vllm_config.compilation_config, "cudagraph_mode", ""))
        warm(_CAP, sorted(set(list(cg) + list(warm_sizes or [1]))))
        return orig_warm(self, *a, **kw)

    def _determine_batch_execution_and_padding(self, *a, **kw):
        res = orig_pad(self, *a, **kw)
        cap = _CAP
        if cap is not None and cap.enabled:
            num_tokens = kw.get("num_tokens", a[0] if a else None)
            mode, desc = res[0], res[1]
            cap.on_step(num_tokens=int(num_tokens), num_tokens_padded=int(desc.num_tokens),
                        cudagraph_mode=str(mode), dummy=cap.in_dummy)
        return res

    def _dummy_run(self, *a, **kw):
        cap = _CAP
        if cap is not None:
            cap.in_dummy = True
        try:
            return orig_dummy(self, *a, **kw)
        finally:
            if cap is not None:
                cap.in_dummy = False

    Worker.compile_or_warm_up_model = compile_or_warm_up_model
    GPUModelRunner._determine_batch_execution_and_padding = _determine_batch_execution_and_padding
    GPUModelRunner._dummy_run = _dummy_run
