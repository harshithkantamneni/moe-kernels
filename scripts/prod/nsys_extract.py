"""Per-call GPU times of the measured MoE call, from an nsys sqlite export (mode A) or a torch-profiler trace
(mode B), joined to the capture ring and the step log, with the gates that make the join trustworthy (KIT-SPEC 4).

    python scripts/prod/nsys_extract.py --run RUN_DIR --sqlite RUN_DIR/serve.sqlite --model ernie-4.5-21b-a3b
    python scripts/prod/nsys_extract.py --run RUN_DIR --torch-trace RUN_DIR/torch_trace --model ... --shard-tag profiler-window

THE MEASURED CALL (CAPTURE.md; v7's call): align .. moe_sum, the router EXCLUDED. On the marker's stream, between
the k-th capture-marker kernel and the next one:
  align   moe_align_block_size* / count_and_sort_expert_tokens* kernels and memsets before the first GEMM; on the
          naive path (T k 4 <= E) the torch.full fill kernel just before the first GEMM (fused_moe.py:1566-1576)
  gemm    the two fused_moe_kernel launches (w13, then w2)
  act     the activation kernel between them (act_and_mul / silu_and_mul)
  reduce  the first moe_sum kernel after the w2 GEMM (fused_moe.py:1848)
kernel-sum = the sum of those kernels' durations; span = first start to last end (CRITIQUE M7). Any other kernel
on the same stream inside the span is an INTRUDER (counted, named, excluded from kernel-sum, inside span): with
the shared-experts stream off, shared-expert GEMMs would show here. Kernels after the marker and before the
first align kernel (the topk_ids dtype conversion, EPLB) are router-side and excluded.

ALIGNMENT (primary): the k-th marker kernel (moe_capture.MARKER_KERNEL) after the reset mark is the k-th ring
record. NVTX "moeL{l}T{n}" ranges are secondary (graph replays carry the capture-time range only).

GATES (gates.json; any FAIL makes the exit 3 and is printed with its reason):
  G-REC   ring records == marker kernels after the reset
  G-GEMM  fused_moe_kernel launches after the reset == 2 x records
  G-STEP  records of every shard tile the shard's steps: n_layers records per step, layers ascending, n == T_pad
  G-IDS   every record's ids in [0, E) with k distinct experts per row (log integrity; sum(counts) == T_pad k)
  G-GRID  gridX of both GEMMs == cdiv(EM, BM) x cdiv(N, BN) for the logged padded T and the call's counts
  G-CLASS every call found its align, two GEMMs, activation and reduce
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prodlib as PL  # noqa: E402

MARKER = "_moe_prod_capture_kernel"
RESET_MARK = "moe_capture:reset"
CLASSES = (
    ("marker", re.compile(re.escape(MARKER))),
    # Triton names the kernel after the function; older Triton appended a signature suffix (_0d1d2d...)
    ("gemm", re.compile(r"(^|\W)fused_moe_kernel(_[0-9][0-9a-z]*)?($|\W)")),
    ("align", re.compile(r"moe_align_block_size|count_and_sort_expert_tokens")),
    ("act", re.compile(r"act_and_mul|silu_and_mul|swiglu|gelu_and_mul")),
    ("reduce", re.compile(r"moe_sum")),
    ("fill", re.compile(r"FillFunctor|fill_kernel|\bfill\b", re.I)),
)


def classify(name: str) -> str:
    for cls, rx in CLASSES:
        if rx.search(name):
            return cls
    return "other"


# ------------------------------------------------------------------------------------------------------------
# readers: a common event list (start_ns, end_ns, stream, name, gridX, kind) sorted by start
# ------------------------------------------------------------------------------------------------------------

def _cols(conn, table) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _tables(conn) -> set:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def read_sqlite(path) -> dict:
    """CUPTI_ACTIVITY_KIND_KERNEL (+ _MEMSET) and NVTX_EVENTS of an `nsys export --type sqlite` file."""
    conn = sqlite3.connect(str(path))
    try:
        tabs = _tables(conn)
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tabs:
            raise PL.KitError(f"{path}: no CUPTI_ACTIVITY_KIND_KERNEL table (no CUDA trace?)")
        strings = dict(conn.execute("SELECT id, value FROM StringIds")) if "StringIds" in tabs else {}
        kc = _cols(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
        name_col = next((c for c in ("demangledName", "shortName", "mangledName") if c in kc), None)
        if name_col is None:
            raise PL.KitError(f"{path}: kernel table has no name column ({kc})")
        gn = "graphNodeId" if "graphNodeId" in kc else "NULL"
        rows = conn.execute(f"SELECT start, end, streamId, {name_col}, gridX, gridY, gridZ, {gn} "
                            "FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
        ev = [(int(s), int(e), int(st), strings.get(n, str(n)), int(gx), int(gy), int(gz), gnode, "kernel")
              for s, e, st, n, gx, gy, gz, gnode in rows]
        if "CUPTI_ACTIVITY_KIND_MEMSET" in tabs:
            for s, e, st in conn.execute("SELECT start, end, streamId FROM CUPTI_ACTIVITY_KIND_MEMSET"):
                ev.append((int(s), int(e), int(st), "[memset]", 0, 0, 0, None, "memset"))
        marks = []
        if "NVTX_EVENTS" in tabs:
            nc = _cols(conn, "NVTX_EVENTS")
            txt = "text" if "text" in nc else "NULL"
            tid = "textId" if "textId" in nc else "NULL"
            for s, t, ti in conn.execute(f"SELECT start, {txt}, {tid} FROM NVTX_EVENTS"):
                label = t if t is not None else strings.get(ti)
                if label is not None:
                    marks.append((int(s), str(label)))
        ev.sort(key=lambda r: (r[0], r[1]))
        marks.sort()
        return {"events": ev, "marks": marks, "source": str(path), "kind": "nsys"}
    finally:
        conn.close()


def read_torch_trace(path) -> dict:
    """A torch-profiler chrome trace (.json or .json.gz; a directory: every trace in it): kernel and gpu_memset
    events, ts/dur in microseconds -> ns. No reset mark is needed: the window's records are one ring shard."""
    p = Path(path)
    files = sorted(list(p.rglob("*.json")) + list(p.rglob("*.json.gz"))) if p.is_dir() else [p]
    if not files:
        raise PL.KitError(f"no torch trace under {path}")
    ev = []
    for f in files:
        raw = gzip.open(f, "rt").read() if f.suffix == ".gz" else f.read_text()
        doc = json.loads(raw)
        for e in doc.get("traceEvents", []):
            cat = e.get("cat", "")
            if cat not in ("kernel", "gpu_memset") or e.get("ph") != "X":
                continue
            s = int(round(float(e["ts"]) * 1000.0))
            d = int(round(float(e.get("dur", 0)) * 1000.0))
            args = e.get("args", {}) or {}
            grid = args.get("grid") or [0, 0, 0]
            name = e.get("name", "") if cat == "kernel" else "[memset]"
            ev.append((s, s + d, int(args.get("stream", -1)), name, int(grid[0]), int(grid[1]), int(grid[2]), None,
                       "kernel" if cat == "kernel" else "memset"))
    ev.sort(key=lambda r: (r[0], r[1]))
    return {"events": ev, "marks": [], "source": str(path), "kind": "torch"}


# ------------------------------------------------------------------------------------------------------------
# per-call segmentation
# ------------------------------------------------------------------------------------------------------------

def segment_calls(events, *, cut_ns: int | None = None, max_after_w2: int = 16) -> tuple[list[dict], dict]:
    """One dict per marker (in start order) with the call's kernels classified; plus counts for the gates."""
    by_stream: dict = defaultdict(list)
    n_gemm = 0
    for e in events:
        if cut_ns is not None and e[0] < cut_ns:
            continue
        cls = "memset" if e[8] == "memset" else classify(e[3])
        if cls == "gemm":
            n_gemm += 1
        by_stream[e[2]].append((e, cls))
    markers = []
    for st, lst in by_stream.items():
        for i, (e, cls) in enumerate(lst):
            if cls == "marker":
                markers.append((e[0], st, i))
    markers.sort()
    nxt: dict = {}
    per_stream_marks: dict = defaultdict(list)
    for _s, st, i in markers:
        per_stream_marks[st].append(i)
    for st, idxs in per_stream_marks.items():
        for a, b in zip(idxs, idxs[1:] + [len(by_stream[st])], strict=True):
            nxt[(st, a)] = b
    calls = []
    for _s, st, i in markers:
        lst = by_stream[st]
        hi = nxt[(st, i)]
        win = lst[i + 1:hi]
        gem = [j for j, (_e, c) in enumerate(win) if c == "gemm"]
        call = {"stream": st, "marker_start": lst[i][0][0], "status": "ok", "names": []}
        if len(gem) < 2:
            call["status"] = f"{len(gem)} fused_moe_kernel in the window"
            calls.append(call)
            continue
        j1, j2 = gem[0], gem[1]
        if len(gem) > 2:
            call["status"] = f"{len(gem)} fused_moe_kernel in the window (chunked call?)"
        first_align = next((j for j in range(j1) if win[j][1] == "align"), None)
        if first_align is not None:
            s0 = first_align
            while s0 > 0 and win[s0 - 1][1] == "memset":     # the align op's own buffer memsets, just before it
                s0 -= 1
            align = [j for j in range(s0, j1) if win[j][1] in ("align", "memset")]
        else:
            fills = [j for j in range(j1) if win[j][1] == "fill"]
            align = [fills[-1]] if fills else []
            if not align:
                call["status"] = "no align kernel before w1"
        acts = [j for j in range(j1 + 1, j2) if win[j][1] == "act"]
        if len(acts) != 1:
            call["status"] = f"{len(acts)} activation kernels between the GEMMs"
        red = next((j for j in range(j2 + 1, min(len(win), j2 + 1 + max_after_w2)) if win[j][1] == "reduce"), None)
        if red is None:
            call["status"] = "no moe_sum after w2"
        parts = sorted(set(align + [j1] + acts + [j2] + ([red] if red is not None else [])))
        ks = [win[j][0] for j in parts]
        start, end = min(k[0] for k in ks), max(k[1] for k in ks)
        intr = [win[j][0][3] for j in range(len(win)) if j not in parts and win[j][0][0] >= start
                and win[j][0][1] <= end]
        call.update(kernel_sum_ns=int(sum(k[1] - k[0] for k in ks)), span_ns=int(end - start), start_ns=int(start),
                    end_ns=int(end), n_parts=len(parts), intruders=len(intr), intruder_names=sorted(set(intr))[:6],
                    gridX_w1=int(win[j1][0][4]), gridX_w2=int(win[j2][0][4]),
                    grid_yz=[int(win[j1][0][5]), int(win[j1][0][6])], graph_node=win[j1][0][7],
                    names=[win[j][0][3][:60] for j in parts],
                    align_ns=int(sum(win[j][0][1] - win[j][0][0] for j in align)),
                    w1_ns=int(win[j1][0][1] - win[j1][0][0]), w2_ns=int(win[j2][0][1] - win[j2][0][0]))
        calls.append(call)
    streams = Counter(st for _s, st, _i in markers)
    return calls, {"markers": len(markers), "gemms": n_gemm, "marker_streams": dict(streams)}


# ------------------------------------------------------------------------------------------------------------
# the joins
# ------------------------------------------------------------------------------------------------------------

def join_steps(records: list[dict], steps: list[dict], n_layers: int, layer_order: list[int]) -> tuple[list, list]:
    """Assign each record its step (records of shard s belong to steps with shard_next == s, n_layers each, in
    order). Returns (per-record step rows, problems)."""
    by_shard: dict = defaultdict(list)
    for pos, r in enumerate(records):
        by_shard[r["shard"]].append((pos, r))
    steps_by: dict = defaultdict(list)
    for s in steps:
        steps_by[int(s["shard_next"])].append(s)
    out: list = [None] * len(records)
    bad = []
    for shard, recs in sorted(by_shard.items()):
        idx = int(re.search(r"(\d+)", shard).group(1))
        sts = steps_by.get(idx, [])
        if len(recs) != n_layers * len(sts):
            bad.append(f"{shard}: {len(recs)} records, {len(sts)} steps x {n_layers} layers = {n_layers * len(sts)}")
            continue
        for si, st in enumerate(sts):
            chunk = [r for _p, r in recs[si * n_layers:(si + 1) * n_layers]]
            where = [p for p, _r in recs[si * n_layers:(si + 1) * n_layers]]
            layers = [c["layer"] for c in chunk]
            if layers != layer_order:
                bad.append(f"{shard} step {st['step']}: layers {layers[:4]}... not {layer_order[:4]}...")
            ns = {c["n"] for c in chunk}
            if ns != {int(st["T_pad"])}:
                bad.append(f"{shard} step {st['step']}: record n {sorted(ns)} != T_pad {st['T_pad']}")
            for p in where:
                out[p] = st
    return out, bad


def check_ids(rec: dict, E: int, k: int) -> str:
    ids = np.asarray(rec["ids"], dtype=np.int64)
    if ids.shape != (rec["n"], k):
        return f"ids shape {ids.shape}"
    if ids.size and (ids.min() < 0 or ids.max() >= E):
        return f"expert id out of range [{ids.min()}, {ids.max()}]"
    if ids.size:
        srt = np.sort(ids, axis=1)
        if (srt[:, 1:] == srt[:, :-1]).any():
            return "a token repeats an expert"
    return ""


def clock_at(marks, clocks: dict, t_ns: int):
    """The SM clock of the last 'clk <i>' NVTX mark at or before t (nsys timeline), from clocks.jsonl."""
    if not marks:
        return None
    i = bisect.bisect_right(marks, (t_ns, "￿")) - 1
    while i >= 0:
        lab = marks[i][1]
        if lab.startswith("clk "):
            row = clocks.get(int(lab[4:]))
            return None if row is None else row.get("sm_mhz")
        i -= 1
    return None


def extract(run_dir, trace: dict, model: PL.ServedModel, *, shard_tag: str | None = None) -> dict:
    run = Path(run_dir)
    cfg = model.cfg
    shards = sorted((run / "ring").glob("shard_*.npz"))
    records = PL.decode_shards(shards, cfg.top_k)
    if shard_tag is not None:
        records = [r for r in records if r["tag"] == shard_tag]
    steps = PL.read_jsonl(run / "steps.jsonl")
    man = json.loads((run / "capture_manifest.json").read_text())
    layer_order = [r["layer"] for r in man["bound_layers"]]
    cut = None
    if trace["kind"] == "nsys":
        resets = [s for s, t in trace["marks"] if t == RESET_MARK]
        if len(resets) != 1:
            raise PL.KitError(f"{len(resets)} '{RESET_MARK}' marks in the trace (want 1)")
        cut = resets[0]
    calls, cnt = segment_calls(trace["events"], cut_ns=cut)
    gates: dict = {}
    nrec = len(records)
    gates["G-REC"] = {"verdict": "PASS" if cnt["markers"] == nrec else "FAIL", "records": nrec,
                      "markers": cnt["markers"], "marker_streams": cnt["marker_streams"],
                      "why": "" if cnt["markers"] == nrec else f"{nrec} ring records, {cnt['markers']} marker kernels"}
    gates["G-GEMM"] = {"verdict": "PASS" if cnt["gemms"] == 2 * nrec else "FAIL", "gemms": cnt["gemms"],
                       "why": "" if cnt["gemms"] == 2 * nrec else f"{cnt['gemms']} fused_moe_kernel, 2 x {nrec} records"}
    sel_steps = steps
    if shard_tag is not None:
        keep = {int(re.search(r"(\d+)", r["shard"]).group(1)) for r in records}
        sel_steps = [s for s in steps if int(s["shard_next"]) in keep]
    step_of, bad_steps = join_steps(records, sel_steps, model.moe_layers, layer_order) if records else ([], [])
    gates["G-STEP"] = {"verdict": "PASS" if not bad_steps else "FAIL", "why": "; ".join(bad_steps[:5]),
                       "problems": len(bad_steps)}
    clocks = {r["i"]: r for r in PL.read_jsonl(run / "clocks.jsonl") if "i" in r}
    marks = trace["marks"]
    rows, counts_all = [], []
    bad_ids, bad_grid, bad_cls = [], [], []
    paired = min(len(calls), nrec)
    for k in range(paired):
        rec, call = records[k], calls[k]
        counts = PL.counts_from_ids(rec["ids"], cfg.num_experts) if not check_ids(rec, cfg.num_experts, cfg.top_k) \
            else [0] * cfg.num_experts
        why = check_ids(rec, cfg.num_experts, cfg.top_k)
        if why:
            bad_ids.append(f"record {rec['g']}: {why}")
        st = step_of[k] if k < len(step_of) else None
        row = {"g": rec["g"], "layer": rec["layer"], "T_pad": rec["n"], "T": None if st is None else st["T"],
               "step": None if st is None else st["step"], "L": None if st is None else st.get("L"),
               "warmup": None if st is None else st.get("warmup"), "cg_mode": None if st is None else st["cg_mode"],
               "tile": PL.tile_bin(rec["n"]), "branch": PL.branch(rec["n"]),
               "strata": PL.strata(rec["n"], cfg.top_k, cfg.num_experts),
               "counts_digest": PL.digest(counts), "ids_digest": PL.digest(rec["ids"]), **{
                   kk: call.get(kk) for kk in ("status", "kernel_sum_ns", "span_ns", "start_ns", "intruders",
                                               "intruder_names", "gridX_w1", "gridX_w2", "align_ns", "w1_ns", "w2_ns",
                                               "n_parts", "stream", "graph_node")}}
        if call["status"] != "ok":
            bad_cls.append(f"call {rec['g']}: {call['status']}")
        elif not why:
            want = PL.expected_grids(cfg, rec["n"], counts)
            row["grid_expected"] = list(want)
            if (call["gridX_w1"], call["gridX_w2"]) != want:
                bad_grid.append(f"call {rec['g']} T_pad {rec['n']}: gridX {call['gridX_w1']}/{call['gridX_w2']} "
                                f"!= {want[0]}/{want[1]}")
        row["sm_mhz"] = clock_at(marks, clocks, call.get("start_ns", call["marker_start"]))
        rows.append(row)
        counts_all.append(counts)
    gates["G-IDS"] = {"verdict": "PASS" if not bad_ids else "FAIL", "why": "; ".join(bad_ids[:5]),
                      "problems": len(bad_ids)}
    gates["G-GRID"] = {"verdict": "PASS" if not bad_grid else "FAIL", "why": "; ".join(bad_grid[:5]),
                       "problems": len(bad_grid)}
    gates["G-CLASS"] = {"verdict": "PASS" if not bad_cls else "FAIL", "why": "; ".join(bad_cls[:5]),
                        "problems": len(bad_cls)}
    seq = Counter(tuple(c.get("names", [])) for c in calls[:2000])
    return {"rows": rows, "counts": np.asarray(counts_all, dtype=np.int32).reshape(-1, cfg.num_experts),
            "gates": gates, "sequences": [{"names": list(k), "calls": v} for k, v in seq.most_common(12)],
            "source": trace["source"], "kind": trace["kind"]}


def write(res: dict, out_dir, prefix: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{prefix}calls.jsonl", "w") as fh:
        for r in res["rows"]:
            fh.write(json.dumps(r, default=str) + "\n")
    np.save(out / f"{prefix}counts.npy", res["counts"])
    (out / f"{prefix}sequences.json").write_text(json.dumps(res["sequences"], indent=2))
    (out / f"{prefix}gates.json").write_text(json.dumps(res["gates"], indent=2, default=str))
    bad = [k for k, v in res["gates"].items() if v["verdict"] == "FAIL"]
    for k, v in res["gates"].items():
        print(f"{v['verdict']:5s} {k}" + (f": {v['why']}" if v.get("why") else ""))
    print(f"{len(res['rows'])} calls -> {out / (prefix + 'calls.jsonl')}")
    return 3 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="serve_capture's output directory")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sqlite")
    g.add_argument("--torch-trace")
    ap.add_argument("--model", required=True, choices=sorted(PL.MODELS))
    ap.add_argument("--shard-tag", default=None, help="only records of shards with this tag (torch window)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="")
    a = ap.parse_args(argv)
    try:
        trace = read_sqlite(a.sqlite) if a.sqlite else read_torch_trace(a.torch_trace)
        tag = a.shard_tag if a.shard_tag is not None else ("profiler-window" if a.torch_trace else None)
        res = extract(a.run, trace, PL.MODELS[a.model], shard_tag=tag)
    except (PL.KitError, PL.RingError, OSError, sqlite3.Error) as exc:
        print(f"FAIL nsys_extract: {exc}")
        return 3
    return write(res, a.out or a.run, a.prefix)


if __name__ == "__main__":
    sys.exit(main())
