"""Off-GPU fakes for the production-check kit's tests: a synthetic serving run (capture ring shards, step log,
manifest, NVML clock log) written by the REAL moe_capture.Capture host logic on its numpy backend, and the GPU
timeline that run would leave in an `nsys export --type sqlite` file (the CUPTI/NVTX table schema nsys writes)
or in a torch-profiler chrome trace."""
from __future__ import annotations

import gzip
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "prod", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import moe_capture as MC  # noqa: E402
import prodlib as PL  # noqa: E402

STREAM = 7
NAMES = {
    "router": "void vllm::moe::topkGatingSoftmax<float, 8, 64>(float const*, bool const*, float*, int, int*, int)",
    "marker": MC.MARKER_KERNEL,
    "stepmark": MC.STEP_MARKER_KERNEL,
    "convert": "void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda>",
    "fill": "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >",
    "small": "void vllm::moe::moe_align_block_size_small_batch_expert_kernel<int>(int const*, int*, int*, int*, int, int, int, int)",
    "align": "void vllm::moe::moe_align_block_size_kernel<int>(int const*, int*, int*, int*, int, int, int, int, unsigned long, int*)",
    "sort": "void vllm::moe::count_and_sort_expert_tokens_kernel<int>(int const*, int*, int*, unsigned long)",
    "gemm": "fused_moe_kernel",
    "act": "void vllm::act_and_mul_kernel<c10::BFloat16, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(c10::BFloat16 const&)), true>",
    "reduce": "void vllm::moe::moe_sum_vec_kernel<c10::BFloat16, 8>(c10::BFloat16*, c10::BFloat16 const*, int)",
    "attn": "void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4> >",
    "shared": "nvjet_tst_128x64_64x6_2x1_v_bz_TNT",
}
DUR = {"router": 3000, "marker": 2000, "convert": 1000, "fill": 1000, "small": 3000, "align": 3000, "sort": 2000,
       "memset": 1000, "w1": 10000, "act": 2000, "w2": 8000, "reduce": 2000, "attn": 5000, "shared": 4000}
GAP = 500


def random_ids(rng, n: int, E: int, k: int) -> np.ndarray:
    return np.stack([rng.permutation(E)[:k] for _ in range(n)]).astype(np.int64) if n else np.zeros((0, k), np.int64)


def make_run(root: Path, model_key: str = "olmoe-1b-7b", steps=((5, 8), (200, 208), (600, 600)), *,
             seed: int = 0, drain_every: int = 2, warmup_calls: int = 3, tag_last: str | None = None,
             context=None) -> tuple[Path, list[dict]]:
    """A serving run of (T, T_pad) steps on the numpy Capture. Returns (run dir, calls in record order:
    {"layer", "n", "ids", "step"})."""
    model = PL.MODELS[model_key]
    cfg = model.cfg
    rng = np.random.default_rng(seed)
    cap = MC.Capture(root, top_k=cfg.top_k, n_moe_layers=model.moe_layers, ids_cap=1 << 22, rec_cap=4096,
                     drain_every=drain_every, backend="numpy")
    layers = list(range(cfg.first_moe_layer, cfg.first_moe_layer + model.moe_layers))
    cap.bound_layers = [{"layer": layer, "backend": "TRITON", "experts_cls": "TritonExperts"} for layer in layers]
    for _ in range(warmup_calls):                           # warmup/capture calls: dropped by reset()
        cap.launch(layers[0], random_ids(rng, 4, cfg.num_experts, cfg.top_k))
    cap.reset()
    calls = []
    for si, (T, Tp) in enumerate(steps):
        cap.context = dict(context or {"L": 8, "warmup": False})
        cap.on_step(num_tokens=T, num_tokens_padded=Tp, cudagraph_mode="CUDAGraphMode.PIECEWISE", dummy=False)
        for layer in layers:
            ids = random_ids(rng, Tp, cfg.num_experts, cfg.top_k)
            cap.launch(layer, ids)
            calls.append({"layer": layer, "n": Tp, "ids": ids, "step": si})
        if tag_last and si == len(steps) - 2:
            cap.drain(tag="pre-profile")
    if tag_last:
        cap.drain(tag=tag_last)
    cap.close()
    return Path(root), calls


SHARED_STREAM = 9
SHARED_ACT = "void vllm::act_and_mul_kernel<c10::BFloat16, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(c10::BFloat16 const&)), true>"


def call_events(cfg, call: dict, t: int, *, intruder: bool = False, bad_grid: bool = False, extra_gemm: bool = False,
                drop_marker: bool = False, shared_stream: bool = False, shared_same_stream: bool = False,
                extra_fill: bool = False, api_fills: int = 0) -> tuple[list[dict], int, dict]:
    """One call's kernels from time t: (events, next t, expected {kernel_sum, span, align}). Router first
    (excluded), the marker (gridY = layer + 1, as moe_capture launches it), a dtype conversion (excluded), the align
    path for (n, k, E), w1, act, w2, moe_sum, then an attention kernel of the next layer (outside).
    shared_stream: the shared experts' MLP (GEMM, SiluAndMul, GEMM) on ANOTHER stream, overlapping w1..w2 in time
    (the shared-experts stream on); shared_same_stream: that MLP on the marker's stream BEFORE the marker (stream
    off: it runs as its own op, never inside the routed call); extra_fill: a second fill kernel before w1;
    api_fills: that many fill kernels before the align run (the functional fused_experts' buffer fills, SMOKE-FIXES
    6: the replay's API path, which serving's modular path does not launch)."""
    n, k, E = call["n"], cfg.top_k, cfg.num_experts
    counts = PL.counts_from_ids(call["ids"], E)
    g1, g2 = PL.expected_grids(cfg, n, counts)
    ev = []

    def add(kind, name, dur, gx=1, mkind="kernel", gy=1):
        nonlocal t
        ev.append({"start": t, "end": t + dur, "stream": STREAM, "name": name, "gx": gx, "gy": gy, "kind": mkind,
                   "role": kind})
        t += dur + GAP
    if shared_same_stream:
        add("shared", NAMES["shared"], DUR["shared"])
        add("shared_act", SHARED_ACT, DUR["act"])
        add("shared", NAMES["shared"], DUR["shared"])
    add("router", NAMES["router"], DUR["router"])
    if not drop_marker:
        add("marker", NAMES["marker"], DUR["marker"], gy=int(call["layer"]) + 1)
    add("convert", NAMES["convert"], DUR["convert"])
    numel = n * k
    if numel * 4 <= E:
        parts = [("fill", NAMES["fill"], DUR["fill"], "kernel")]
    elif numel < 1024:
        parts = [("small", NAMES["small"], DUR["small"], "kernel")]
    else:
        parts = [("memset", "[memset]", DUR["memset"], "memset"), ("align", NAMES["align"], DUR["align"], "kernel"),
                 ("sort", NAMES["sort"], DUR["sort"], "kernel")]
    for _ in range(int(api_fills)):
        add("api_fill", NAMES["fill"], DUR["fill"])
    if extra_fill and numel * 4 <= E:
        add("fill", NAMES["fill"], DUR["fill"])
    s0 = t
    for role, name, d, mk in parts:
        add(role, name, d, mkind=mk)
    w1_start = t
    add("w1", NAMES["gemm"], DUR["w1"], gx=g1 + (1 if bad_grid else 0))
    if shared_stream:
        # another stream, in time beside w1 .. w2 (the act lands between the routed GEMMs in time order)
        ts = w1_start + 100
        for role, name, d in (("shared", NAMES["shared"], DUR["shared"]), ("shared_act", SHARED_ACT, DUR["act"]),
                              ("shared", NAMES["shared"], DUR["shared"])):
            ev.append({"start": ts, "end": ts + d, "stream": SHARED_STREAM, "name": name, "gx": 1, "gy": 1,
                       "kind": "kernel", "role": role})
            ts += d + GAP
    if intruder:
        add("shared", NAMES["shared"], DUR["shared"])
    add("act", NAMES["act"], DUR["act"])
    add("w2", NAMES["gemm"], DUR["w2"], gx=g2)
    if extra_gemm:
        add("w2b", NAMES["gemm"], DUR["w2"], gx=g2)
    e_end = t + DUR["reduce"]
    add("reduce", NAMES["reduce"], DUR["reduce"])
    add("attn", NAMES["attn"], DUR["attn"])
    ksum = sum(d for _r, _n, d, _m in parts) + DUR["w1"] + DUR["act"] + DUR["w2"] + DUR["reduce"]
    return ev, t, {"kernel_sum_ns": ksum, "span_ns": e_end - s0, "align_ns": sum(d for _r, _n, d, _m in parts),
                   "fills_ns": int(api_fills) * DUR["fill"]}


def timeline(cfg, calls, *, t0: int = 10_000_000, warmup_markers: int = 3, mutate=None, clk_every: int = 4,
             step_ids=None, drop_step_marks=()):
    """Events and NVTX marks for a run: warmup markers + GEMMs before the reset mark, the reset mark, then every
    call; a 'clk i' mark every clk_every calls. mutate(i) -> dict of call_events flags for call i. Before each step's
    first call, the step marker moe_capture launches from the step log (grid (1, step % STEP_MOD + 1)): step_ids is
    the list of step ids the markers carry, in order (default: the calls' own steps, as Capture.step_marks records
    them); drop_step_marks: positions whose marker is missing from the trace."""
    ev, marks, expect = [], [], []
    t = 1_000_000
    for _ in range(warmup_markers):
        for name, d in ((NAMES["marker"], 2000), (NAMES["gemm"], 9000), (NAMES["gemm"], 9000)):
            ev.append({"start": t, "end": t + d, "stream": STREAM, "name": name, "gx": 1, "kind": "kernel", "role": "w"})
            t += d + GAP
    marks.append((t0 - 1000, MC_RESET))
    t = t0
    ci = 0
    order = []
    for c in calls:
        if not order or order[-1] != c["step"]:
            order.append(c["step"])
    ids = list(step_ids) if step_ids is not None else order
    pos = -1
    for i, c in enumerate(calls):
        if i == 0 or calls[i - 1]["step"] != c["step"]:
            pos += 1
            if pos not in drop_step_marks:
                ev.append({"start": t, "end": t + 1000, "stream": STREAM, "name": NAMES["stepmark"], "gx": 1,
                           "gy": MC.step_grid(ids[pos])[1], "kind": "kernel", "role": "stepmark"})
            t += 1000 + GAP
        if i % clk_every == 0:
            marks.append((t - 100, f"clk {ci}"))
            ci += 1
        e, t, x = call_events(cfg, c, t, **((mutate(i) if mutate else None) or {}))
        ev += e
        expect.append(x)
    return ev, marks, expect


MC_RESET = "moe_capture:reset"


def write_clocks(run: Path, n: int, mhz: float = 1710.0, off=()):
    with open(run / "clocks.jsonl", "w") as fh:
        for i in range(n):
            fh.write(json.dumps({"i": i, "t_ns": i, "sm_mhz": (mhz - 60.0) if i in off else mhz, "power_w": 400.0}) + "\n")


def write_sqlite(path: Path, events, marks) -> Path:
    """The tables nsys's sqlite export writes (the columns this kit reads, with nsys's names and a few of the
    others so the reader must pick by name)."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER NOT NULL, end INTEGER NOT NULL, "
                 "deviceId INTEGER NOT NULL, contextId INTEGER NOT NULL, greenContextId INTEGER, streamId INTEGER NOT NULL, "
                 "correlationId INTEGER, globalPid INTEGER, demangledName INTEGER NOT NULL, shortName INTEGER NOT NULL, "
                 "mangledName INTEGER, launchType INTEGER, cacheConfig INTEGER, registersPerThread INTEGER NOT NULL, "
                 "gridX INTEGER NOT NULL, gridY INTEGER NOT NULL, gridZ INTEGER NOT NULL, blockX INTEGER NOT NULL, "
                 "blockY INTEGER NOT NULL, blockZ INTEGER NOT NULL, staticSharedMemory INTEGER NOT NULL, "
                 "dynamicSharedMemory INTEGER NOT NULL, localMemoryPerThread INTEGER NOT NULL, "
                 "localMemoryTotal INTEGER NOT NULL, gridId INTEGER NOT NULL, sharedMemoryExecuted INTEGER, "
                 "graphNodeId INTEGER, sharedMemoryLimitConfig INTEGER)")
    conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER NOT NULL, end INTEGER NOT NULL, "
                 "deviceId INTEGER NOT NULL, contextId INTEGER NOT NULL, greenContextId INTEGER, streamId INTEGER NOT NULL, "
                 "correlationId INTEGER, globalPid INTEGER, value INTEGER NOT NULL, bytes INTEGER NOT NULL, "
                 "graphNodeId INTEGER, memKind INTEGER)")
    conn.execute("CREATE TABLE NVTX_EVENTS (start INTEGER NOT NULL, end INTEGER, eventType INTEGER NOT NULL, "
                 "rangeId INTEGER, category INTEGER, color INTEGER, text TEXT, globalTid INTEGER, endGlobalTid INTEGER, "
                 "textId INTEGER, domainId INTEGER, uint64Value INTEGER, int64Value INTEGER, doubleValue REAL, "
                 "uint32Value INTEGER, int32Value INTEGER, floatValue REAL, jsonTextId INTEGER, jsonText TEXT)")
    ids: dict = {}

    def sid(s):
        if s not in ids:
            ids[s] = len(ids) + 1
            conn.execute("INSERT INTO StringIds VALUES (?, ?)", (ids[s], s))
        return ids[s]
    for i, e in enumerate(events):
        if e["kind"] == "memset":
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (e["start"], e["end"], 0, 1, None, e["stream"], i, 1, 0, 4096, None, 1))
            continue
        n = sid(e["name"])
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (" + ",".join(["?"] * 28) + ")",
                     (e["start"], e["end"], 0, 1, None, e["stream"], i, 1, n, n, n, 0, 0, 32, e["gx"], e.get("gy", 1), 1,
                      128, 1, 1, 0, 0, 0, 0, i, 0, 1000 + i, 0))
    for j, (s, txt) in enumerate(marks):
        # nsys writes registered strings via textId and ad-hoc ones in text: exercise both
        if j % 2:
            conn.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text, textId) VALUES (?, NULL, 34, NULL, ?)",
                         (s, sid(txt)))
        else:
            conn.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text) VALUES (?, NULL, 34, ?)", (s, txt))
    conn.commit()
    conn.close()
    return path


def write_torch_trace(d: Path, events, *, gz: bool = False) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    tev = []
    for i, e in enumerate(events):
        if e["kind"] == "memset":
            tev.append({"ph": "X", "cat": "gpu_memset", "name": "Memset (Device)", "ts": e["start"] / 1000.0,
                        "dur": (e["end"] - e["start"]) / 1000.0, "args": {"stream": e["stream"], "device": 0}})
        else:
            tev.append({"ph": "X", "cat": "kernel", "name": e["name"], "ts": e["start"] / 1000.0,
                        "dur": (e["end"] - e["start"]) / 1000.0,
                        "args": {"stream": e["stream"], "grid": [e["gx"], e.get("gy", 1), 1], "block": [128, 1, 1],
                                 "correlation": i, "device": 0}})
        tev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": e["start"] / 1000.0 - 5,
                    "dur": 3, "args": {}})
    doc = {"traceEvents": tev, "displayTimeUnit": "ms"}
    p = d / ("trace.pt.trace.json.gz" if gz else "trace.pt.trace.json")
    if gz:
        with gzip.open(p, "wt") as fh:
            json.dump(doc, fh)
    else:
        p.write_text(json.dumps(doc))
    return p
