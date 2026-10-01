#!/usr/bin/env python
"""THE LAUNCH FLOOR: is R3's flat 0.24-0.29 ms at small n the instrument running
host-paced, or a GPU-side fixed time? (docs/registered, 2026-10-01; design
`design-launch/DESIGN.md` with the adversarial review's fixes.)

    python scripts/launch_floor.py --model granite-3.0-3b-a800m --treads 1,2,3,4,5,6,7,8,9 \\
        --modes E240,E0,E480,GR --trace-treads 1,2,4,6,9 --arms native,shared,private \\
        --group-m 4 --duty 0.25 --repeats 3 --seed 0 --out DIR [--dry-run]

WHAT IS TIMED. R3's own call: the weights, inputs, routing and closure come from
`private_weight_reference` (`build_private_weights`, `arm_inputs`, `arm_call`,
`declared_experts`, `pinned_config`, `counter_declaration`, `SWEEP.find_override`),
imported and never copied, so the call under test is the call on every
published page. R3's page path, gates and instrument string are untouched:
this script writes its own directory and no R3 page.

THE MODES, per (arm, n, repeat), all at the page's duty and lock:
  E240  R3 as published: `time_cell`'s duty sizing mirrored line for line
        (`mirror_time_cell`), then `clock_elasticity.time_duty` with an
        injected 240 MiB `L2Flusher` (4 x the 60 MiB L2, `flush_mb_for_device`)
        and `GAP_FROM_BURST`. The control: it must reproduce the published cells.
  E0    no flush. The lever's other side: a host-paced cell moves by about
        +(F - h_flush), a GPU-bound one by its warm-L2 speed-up only.
  E480  a 480 MiB flush: F roughly doubles; host-paced cells move by -dF.
  GR    one captured `fused_experts` call replayed as a CUDA graph (capture
        inside `override_config` after three side-stream warmups, as
        `moe.bench.driver.time_kernel_graph` does), timed by the same duty
        loop with the 240 MiB flush outside the graph. The host is out of the
        interval, so the reading is the call's GPU time. A `NotCapturable` is
        recorded as a finding and every GR row of that cell is a FAILED row.
  traces  one extra burst (not timed) of the identical loop body under
        `torch.profiler` (Kineto: CUPTI activity API, not the counter API, so
        the NVreg gate that blocks ncu does not apply), eager (TR-E) and graph
        (TR-G), each call inside `record_function("r3call")`, written as
        chrome traces (.json.gz). At every --trace-treads tread at E240, and
        at the lever treads (--lever-trace-treads, default 1,2: the P3 cells)
        at E0 and E480 as well (the review: E0 also moves D_tail).

`parse_trace` turns a trace into per call: the host span of `r3call`, the
host cost of the flush launch (h_flush) and of the event records, every
kernel's launch-API offset from the start-event record, every kernel's GPU
name and duration, the GPU idle between consecutive kernels, the flush
kernel's duration F, the launch latency lambda (kernel start minus its
launch-API return, least over the call), and D_tail (GPU time after the
call's last launch-API return). Nothing in this script is fitted.

WHAT IT WRITES under --out: cells.csv (one row per timed cell: R3's identity
columns, the mode, flush MiB, the duty timer's host_enqueue_ms and
host_backlog_iters on EVERY cell, not only host-bound ones, its host-bound
verdict, ms_p50/min/stdev, clocks and power), traces/*.json.gz,
traces/parsed.json, and manifest.json (versions, device, commit, the plan,
lscpu and the cpufreq governor, graph-capture findings).

EXIT CODES (moe/bench/exit_codes.py): 0 every cell and trace ran; 2 refused
(bad arguments, no CUDA or no vLLM; and --dry-run, which prints the plan and
its minutes); 3 some cells, modes or traces failed and the files are written.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import private_weight_reference as PW  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

SWEEP = PW.SWEEP
MODES = ("E240", "E0", "E480", "GR")
#: flush MiB per eager mode; GR replays under E240's flush (outside the graph)
FLUSH_MB = {"E240": None, "E0": 0, "E480": 480, "GR": None}
BLOCK_M = 32
#: R3's defaults for a timed cell (`private_weight_reference` --warmup,
#: --cell-budget-ms, --trials), every GH200 page's
WARMUP_MS = 300.0
CELL_BUDGET_MS = 200.0
TRIALS = 3
#: the profiled burst's calls (outside every timed burst)
TRACE_CALLS = 32
#: the plan's minute estimate: Granite's 729 timed duty-0.25 cells took 36 min
#: (2026-09-30), 2.96 s a cell; a trace burst with its export about 4 s; the
#: weight build, Triton compile and alignment warmup about 3 min a model
SEC_PER_CELL = 3.0
SEC_PER_TRACE = 4.0
SETUP_MIN = 3.0
GEMM_KERNEL = "fused_moe_kernel"
CSV_COLUMNS = (
    "model", "mode", "arm", "tiles", "repeat", "group_m", "block_m", "tokens",
    "copies", "experts_declared", "flush_mb", "duty", "status", "ms_p50", "ms_p90",
    "ms_min", "ms_stdev", "samples", "calls_per_burst", "bursts", "gap_ms",
    "duty_achieved", "host_bound", "host_enqueue_ms", "backlog", "sm_clock_load_mhz",
    "clock_samples_mhz", "clock_level_ok", "clock_level_side", "clock_drift_ok", "power_w",
    "mem_clock_mhz", "per_call_sizing_ms", "detail")


class Refused(Exception):
    """A precondition failed before anything was timed. Exits REFUSED."""


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def int_list(text: str) -> list[int]:
    out = [int(x) for x in str(text).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError(f"{text!r}: a comma-separated list of integers")
    return out


def name_list(choices):
    def parse(text: str) -> list[str]:
        out = [x.strip() for x in str(text).split(",") if x.strip()]
        bad = [x for x in out if x not in choices]
        if not out or bad:
            raise argparse.ArgumentTypeError(f"{text!r}: names from {', '.join(choices)}")
        return out
    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    p.add_argument("--treads", type=int_list, required=True)
    p.add_argument("--modes", type=name_list(MODES), default=list(MODES))
    p.add_argument("--trace-treads", type=int_list, default=[])
    p.add_argument("--lever-trace-treads", type=int_list, default=[1, 2],
                   help="treads also traced at E0 and E480 (the P3 cells), "
                        "where they are in --trace-treads")
    p.add_argument("--arms", type=name_list(PW.ARMS), default=list(PW.ARMS))
    p.add_argument("--group-m", type=int, default=4)
    p.add_argument("--duty", type=float, default=0.25)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def make_plan(args) -> dict:
    """The cells, traces and minutes; refuses what R3 would not time."""
    cfg = MODEL_CONFIGS[args.model]
    if not 0 < args.duty < 1:
        raise Refused(f"--duty {args.duty}: the launch floor is a duty-cycled "
                      "measurement (R3's pages ran at 0.25); 1.0 has no bursts")
    if args.repeats < 1 or args.group_m < 1:
        raise Refused("--repeats and --group-m must be at least 1")
    try:
        ladder = PW.counter_ladder(cfg, BLOCK_M)
        copies, why = PW.counter_declaration(cfg, BLOCK_M)
    except PW.PrivateWeightRefusal as exc:
        raise Refused(f"{args.model}: R3 refuses its ladder: {exc}") from None
    off = [n for n in args.treads if n not in ladder]
    if off:
        raise Refused(f"treads {off} are not on R3's counter ladder {ladder} for "
                      f"{args.model} (max {PW.COUNTER_MAX_TREADS}: the 9-copy "
                      "declaration every published page and C_reg use)")
    if max(args.treads) > copies:
        raise Refused(f"tread {max(args.treads)} is past the {copies} copies declared")
    bad_tr = [n for n in args.trace_treads if n not in args.treads]
    if bad_tr:
        raise Refused(f"--trace-treads {bad_tr} are not timed treads {args.treads}")
    cells = len(args.treads) * len(args.arms) * len(args.modes) * args.repeats
    traces = []
    for n in args.trace_treads:
        for arm in args.arms:
            flushes = ["E240"] + (["E0", "E480"] if n in args.lever_trace_treads else [])
            for fm in flushes:
                traces.append({"arm": arm, "n": n, "flush": fm, "kind": "TR-E"})
                if "GR" in args.modes:
                    traces.append({"arm": arm, "n": n, "flush": fm, "kind": "TR-G"})
    minutes = cells * SEC_PER_CELL / 60 + len(traces) * SEC_PER_TRACE / 60 + SETUP_MIN
    return {"model": args.model, "treads": list(args.treads), "modes": list(args.modes),
            "arms": list(args.arms), "group_m": args.group_m, "duty": args.duty,
            "repeats": args.repeats, "seed": args.seed, "block_m": BLOCK_M,
            "copies_declared": copies, "declaration": why,
            "declared_by_arm": {a: PW.declared_experts(a, cfg.num_experts, copies)
                                for a in PW.ARMS},
            "pinned": dict(PW.pinned_config(SWEEP.FIXED["BLOCK_SIZE_N"], args.group_m,
                                            SWEEP.FIXED["num_stages"]), BLOCK_SIZE_M=BLOCK_M),
            "timed_cells": cells, "traces": traces, "minutes": round(minutes, 1),
            "warmup_ms": WARMUP_MS, "cell_budget_ms": CELL_BUDGET_MS, "trials": TRIALS}


def plan_lines(plan: dict) -> list[str]:
    return [
        f"LAUNCH FLOOR PLAN {plan['model']}: treads {plan['treads']}, arms {plan['arms']}, "
        f"modes {plan['modes']}, G={plan['group_m']}, BLOCK_M {plan['block_m']}, duty "
        f"{plan['duty']}, {plan['repeats']} repeats, seed {plan['seed']}",
        f"  declaration: {plan['copies_declared']} copies ({plan['declaration']}); "
        f"declared {plan['declared_by_arm']}",
        f"  pinned config: {plan['pinned']}",
        f"  {plan['timed_cells']} timed cells at ~{SEC_PER_CELL:.1f} s, {len(plan['traces'])} "
        f"profiled bursts at ~{SEC_PER_TRACE:.0f} s, {SETUP_MIN:.0f} min weights and compile",
        f"  estimated {plan['minutes']:.1f} min",
        "  E240 = R3 as published (240 MiB flush); E0 no flush; E480 480 MiB flush; "
        "GR CUDA-graph replay under E240's flush",
    ]


# --------------------------------------------------------------------------
# the instrument: R3's time_cell at duty < 1, with the flusher injected
# --------------------------------------------------------------------------

def mirror_time_cell(call, *, flusher, l2_flush: bool, duty: float,
                     warmup_ms: float = WARMUP_MS, cell_budget_ms: float = CELL_BUDGET_MS,
                     trials: int = TRIALS, reference_clock_mhz: float | None = None,
                     timer=None, duty_timer=None):
    """`private_weight_reference.time_cell` below full duty, LINE FOR LINE,
    with one change: `flusher` reaches the sizing read and the duty timer
    (`time_cell` passes none, which is why this mirror exists). Returns
    (the duty timer's whole return, the sizing read's per-call ms), so
    host_enqueue_ms and host_backlog_iters are kept on every cell."""
    if duty >= 1.0:
        raise Refused("mirror_time_cell is the duty < 1 branch of time_cell only")
    import clock_elasticity as CE
    if timer is None:
        from moe.bench import timing
        timer = timing.time_kernel
    if duty_timer is None:
        duty_timer = CE.time_duty
    fk = {"flusher": flusher} if (l2_flush and flusher is not None) else {}
    sizing = timer(call, warmup_ms=min(warmup_ms, PW.DUTY_SIZING_MS),
                   target_ms=PW.DUTY_SIZING_MS, trials=1, l2_flush=l2_flush,
                   reference_clock_mhz=reference_clock_mhz, **fk)
    per_call = max(float(sizing.ms_p50), 1e-4)
    calls_per_burst = max(2, round(PW.DUTY_BURST_MS / per_call))
    bursts = max(1, round(cell_budget_ms / PW.DUTY_BURST_MS))
    t = duty_timer(call, duty=duty, calls_per_burst=calls_per_burst,
                   bursts=bursts, trials=trials, warm_ms=warmup_ms,
                   l2_flush=l2_flush, per_call_ms=per_call,
                   reference_clock_mhz=reference_clock_mhz,
                   gap_basis=CE.GAP_FROM_BURST, **fk)
    return t, per_call


def row_from(t, per_call: float | None, **ident) -> dict:
    """One cells.csv row from a DutyTiming (or a failure: t None)."""
    row = {k: "" for k in CSV_COLUMNS}
    row.update(ident)
    if t is None:
        return row
    clocks = getattr(t, "clock_samples_mhz", ()) or ()
    row.update(
        status="ok", ms_p50=t.ms_p50, ms_p90=t.ms_p90, ms_min=t.ms_min, ms_stdev=t.ms_std,
        samples=t.samples, calls_per_burst=t.calls_per_burst, bursts=t.bursts,
        gap_ms=t.gap_ms, duty_achieved=t.duty_achieved, host_bound=t.host_bound,
        host_enqueue_ms=t.host_enqueue_ms, backlog=t.host_backlog_iters,
        sm_clock_load_mhz=t.sm_clock_load_mhz,
        clock_samples_mhz=" ".join(f"{c:.0f}" for c in clocks),
        clock_level_ok=t.clock_level_ok, clock_level_side=getattr(t, "clock_level_side", ""),
        clock_drift_ok=t.clock_drift_ok,
        power_w=t.power_w, mem_clock_mhz=getattr(t, "mem_clock_mhz", None),
        per_call_sizing_ms=per_call, detail=t.clock_note or "")
    return row


# --------------------------------------------------------------------------
# graph replay and traces
# --------------------------------------------------------------------------

def graph_call(fn, *, override_config, conf):
    """One captured `fused_experts` call: three side-stream warmups, the
    capture, three replays, all inside `override_config(conf)` (vLLM resolves
    the tile at call time, so the capture must see the pin). Returns the
    graph's `replay`; raises `timing.NotCapturable`."""
    import torch

    from moe.bench import timing as T
    with override_config(conf):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                fn()
        except RuntimeError as exc:
            raise T.NotCapturable(str(exc)) from None
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph.replay


def trace_burst(fn, flusher, path: Path, calls: int = TRACE_CALLS) -> Path:
    """One profiled burst of the timed loop's body (flush, start event, call,
    end event), outside every timed burst, written as a gzipped chrome trace."""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(calls)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(calls)]
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(calls):
            if flusher is not None:
                with record_function("r3flush"):
                    flusher.flush()
            starts[i].record()
            with record_function("r3call"):
                fn()
            ends[i].record()
        torch.cuda.synchronize()
    raw = path.with_suffix("")          # x.json.gz -> x.json
    prof.export_chrome_trace(str(raw))
    with open(raw, "rb") as src, gzip.open(path, "wb") as dst:
        dst.write(src.read())
    raw.unlink()
    return path


def _events(trace) -> list[dict]:
    ev = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    return [e for e in ev if isinstance(e, dict) and e.get("ph") == "X"]


def _corr(e) -> int | None:
    a = e.get("args") or {}
    c = a.get("correlation", a.get("correlation id"))
    return int(c) if c is not None else None


def _is_kernel(e) -> bool:
    return str(e.get("cat", "")).lower() == "kernel"


def _is_runtime(e) -> bool:
    return str(e.get("cat", "")).lower() in ("cuda_runtime", "cuda_driver")


def _is_launch(e) -> bool:
    n = str(e.get("name", ""))
    return _is_runtime(e) and ("Launch" in n or "launch" in n)


def _is_event_record(e) -> bool:
    return _is_runtime(e) and "EventRecord" in str(e.get("name", ""))


def _is_annotation(e, name) -> bool:
    return (e.get("name") == name
            and str(e.get("cat", "")).lower() in ("user_annotation", "cpu_op"))


def parse_trace(trace) -> dict:
    """Per call of a profiled burst (see the module docstring), and medians.

    Host spans are `r3call` annotations; the flush is the `r3flush`
    annotation before it (h_flush its host span) and the kernel its launch
    correlates to (F); the start event is the last `cudaEventRecord` between
    the flush and the call. Kernels are matched to the call's launches by
    correlation id (a graph launch correlates every kernel it ran). Times in
    microseconds, Kineto's host-aligned clock."""
    ev = sorted(_events(trace), key=lambda e: float(e["ts"]))
    calls = [e for e in ev if _is_annotation(e, "r3call")]
    flushes = [e for e in ev if _is_annotation(e, "r3flush")]
    runtime = [e for e in ev if _is_runtime(e)]
    kernels = {}
    for e in ev:
        if _is_kernel(e) and _corr(e) is not None:
            kernels.setdefault(_corr(e), []).append(e)
    out = []
    prev_end = float("-inf")
    for i, c in enumerate(calls):
        t0, t1 = float(c["ts"]), float(c["ts"]) + float(c["dur"])
        fl = [f for f in flushes if prev_end <= float(f["ts"]) < t0]
        fl = fl[-1] if fl else None
        launches = [r for r in runtime if _is_launch(r) and t0 <= float(r["ts"]) <= t1]
        ks = sorted((k for r in launches for k in kernels.get(_corr(r), [])),
                    key=lambda k: float(k["ts"]))
        lo = (float(fl["ts"]) + float(fl["dur"])) if fl else prev_end
        rec = [r for r in runtime if _is_event_record(r) and lo <= float(r["ts"]) < t0]
        start_rec = float(rec[-1]["ts"]) if rec else t0
        F = h_flush = None
        if fl is not None:
            h_flush = float(fl["dur"])
            fls = [r for r in runtime if _is_launch(r)
                   and float(fl["ts"]) <= float(r["ts"]) <= float(fl["ts"]) + float(fl["dur"])]
            fk = [k for r in fls for k in kernels.get(_corr(r), [])]
            F = sum(float(k["dur"]) for k in fk) if fk else None
        by_corr = {_corr(r): r for r in launches}
        lam = [float(k["ts"]) - (float(by_corr[_corr(k)]["ts"]) + float(by_corr[_corr(k)]["dur"]))
               for k in ks if _corr(k) in by_corr]
        gaps = [float(b["ts"]) - (float(a["ts"]) + float(a["dur"]))
                for a, b in zip(ks, ks[1:], strict=False)]
        last_launch_end = max((float(r["ts"]) + float(r["dur"]) for r in launches),
                              default=None)
        gpu_end = max((float(k["ts"]) + float(k["dur"]) for k in ks), default=None)
        gemm = sum(float(k["dur"]) for k in ks if GEMM_KERNEL in str(k["name"]))
        ksum = sum(float(k["dur"]) for k in ks)
        out.append({
            "call": i, "host_span_us": t1 - t0, "h_flush_us": h_flush, "flush_us": F,
            "event_record_us": [float(r["dur"]) for r in rec],
            "launch_offsets_us": [float(r["ts"]) - start_rec for r in launches],
            "launch_api_us": [float(r["dur"]) for r in launches],
            "kernels": [{"name": str(k["name"])[:80], "dur_us": float(k["dur"]),
                         "start_us": float(k["ts"]) - start_rec} for k in ks],
            "gaps_us": gaps, "kernel_sum_us": ksum, "gemm_us": gemm,
            "non_gemm_us": ksum - gemm,
            "lambda_us": min(lam) if lam else None,
            "d_tail_us": (max(0.0, gpu_end - last_launch_end)
                          if gpu_end is not None and last_launch_end is not None else None),
            "gpu_span_us": (gpu_end - float(ks[0]["ts"])) if ks else None,
        })
        prev_end = t1

    def med(key):
        vals = [c[key] for c in out[1:] if c[key] is not None] or \
               [c[key] for c in out if c[key] is not None]
        return statistics.median(vals) if vals else None
    keys = ("host_span_us", "h_flush_us", "flush_us", "kernel_sum_us", "gemm_us",
            "non_gemm_us", "lambda_us", "d_tail_us", "gpu_span_us")
    return {"calls": out, "median": {k: med(k) for k in keys},
            "launch_api_share": (statistics.median(
                [sum(c["launch_api_us"]) / c["host_span_us"] for c in out[1:] or out
                 if c["host_span_us"] > 0]) if out else None)}


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run_refusal() -> str:
    """"" when a run can go on here, else why not (the checks R3's own
    counter child makes: torch, a CUDA device, vLLM)."""
    try:
        import torch
    except Exception as exc:                              # noqa: BLE001
        return f"no torch: {exc}"
    if not torch.cuda.is_available():
        return "no CUDA device: the launch floor times a GPU, it does not invent numbers"
    import importlib.util
    if importlib.util.find_spec("vllm") is None:
        return "no vLLM in this interpreter"
    return ""


def _sh(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception as exc:                              # noqa: BLE001
        return f"unreadable: {exc}"


def host_record() -> dict:
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    commit = _sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]).strip()
    dirty = _sh(["git", "-C", str(REPO), "status", "--porcelain"]).strip()
    return {"lscpu": _sh(["lscpu"]), "governor": gov.read_text().strip() if gov.exists()
            else "unreadable", "platform": platform.platform(), "commit": commit,
            "git_dirty": bool(dirty), "python": sys.version.split()[0]}


def run(args, plan: dict) -> int:
    import torch

    from moe.bench import timing as T
    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    cfg = MODEL_CONFIGS[args.model]
    os.environ.setdefault("TRITON_CACHE_DIR", str(out / "triton-cache"))
    override_config, where = SWEEP.find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    reference_clock, clock_source = SWEEP.reference_clock_mhz()
    copies = plan["copies_declared"]
    declared = plan["declared_by_arm"]
    conf = plan["pinned"]
    flush240 = T.L2Flusher(T.flush_mb_for_device())
    flushers = {"E240": flush240, "GR": flush240, "E0": None,
                "E480": T.L2Flusher(480)}
    w1, w2, delta = PW.build_private_weights(cfg, "bf16", copies, args.seed)
    nw1, nw2 = w1[::copies], w2[::copies]
    manifest = {"tool": "scripts/launch_floor.py", "plan": plan, "override_hook": where,
                "reference_clock_mhz": reference_clock, "reference_clock_source": clock_source,
                "flush_mb": {m: (int(f.megabytes) if f is not None else 0)
                             for m, f in flushers.items()},
                "weights_bytes": int(delta), "versions": PW._package_versions(),
                "device": {"name": torch.cuda.get_device_name(0),
                           "uuid": PW.device_identity()},
                "host": host_record(), "findings": [], "started": time.time()}
    csv_path = out / "cells.csv"
    failed = 0
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    parsed = {}
    try:
        for rep in range(args.repeats):
            treads = args.treads if rep % 2 == 0 else list(reversed(args.treads))
            for n in treads:
                tokens, x, ids_by_arm, weights, kw = PW.arm_inputs(
                    cfg, n, BLOCK_M, copies, args.seed, "bf16", w1.dtype)
                order = [args.arms[(i + rep) % len(args.arms)] for i in range(len(args.arms))]
                for arm in order:
                    call = PW.arm_call(fused_experts, arm, w1, w2, nw1, nw2, declared, x,
                                       ids_by_arm, weights, kw)
                    ident = dict(model=args.model, arm=arm, tiles=n, repeat=rep,
                                 group_m=args.group_m, block_m=BLOCK_M, tokens=tokens,
                                 copies=PW.copies_read(arm, n),
                                 experts_declared=declared[arm], duty=args.duty)
                    replay, why = None, ""
                    if "GR" in args.modes:
                        try:
                            replay = graph_call(call, override_config=override_config,
                                                conf=conf)
                        except Exception as exc:                  # noqa: BLE001
                            why = f"{type(exc).__name__}: {exc}"
                            manifest["findings"].append(
                                {"arm": arm, "n": n, "repeat": rep, "graph": why})
                    for mode in args.modes:
                        fl = flushers[mode]
                        fn = replay if mode == "GR" else call
                        row_id = dict(ident, mode=mode,
                                      flush_mb=int(fl.megabytes) if fl is not None else 0)
                        if mode == "GR" and replay is None:
                            writer.writerow(row_from(None, None, **row_id, status="failed",
                                                     detail=f"GR NOT CAPTURED: {why}"))
                            failed += 1
                            continue
                        try:
                            with override_config(conf):
                                fn()
                                torch.cuda.synchronize()
                                t, pc = mirror_time_cell(
                                    fn, flusher=fl, l2_flush=fl is not None, duty=args.duty,
                                    reference_clock_mhz=reference_clock)
                            writer.writerow(row_from(t, pc, **row_id))
                        except T.TimingRefused:
                            raise
                        except Exception as exc:                  # noqa: BLE001
                            writer.writerow(row_from(None, None, **row_id, status="failed",
                                                     detail=f"{type(exc).__name__}: {exc}"))
                            failed += 1
                        fh.flush()
                        print(f"  rep{rep} {mode:4s} {arm:8s} n={n}", flush=True)
                    if rep == 0 and n in args.trace_treads:
                        for tr in [t for t in plan["traces"] if t["arm"] == arm and t["n"] == n]:
                            name = f"{tr['kind']}-{tr['flush']}-{arm}-n{n}.json.gz"
                            fn = call if tr["kind"] == "TR-E" else replay
                            if fn is None:
                                failed += 1
                                continue
                            try:
                                with override_config(conf):
                                    trace_burst(fn, flushers[tr["flush"]], out / "traces" / name)
                                with gzip.open(out / "traces" / name, "rt") as f:
                                    parsed[name] = parse_trace(json.load(f))
                            except Exception as exc:              # noqa: BLE001
                                manifest["findings"].append({"trace": name, "error": str(exc)})
                                failed += 1
                    replay = None
    finally:
        fh.close()
        manifest["ended"] = time.time()
        manifest["failed"] = failed
        (out / "traces" / "parsed.json").write_text(json.dumps(parsed, indent=1))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"LAUNCH FLOOR {args.model}: {plan['timed_cells']} cells, {failed} failed; {out}")
    return exit_codes.INVALID if failed else exit_codes.DONE


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = make_plan(args)
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    print("\n".join(plan_lines(plan)))
    if args.dry_run:
        print("DRY RUN: nothing timed, nothing written (exit 2)")
        return exit_codes.REFUSED
    why = run_refusal()
    if why:
        print(f"REFUSED: {why}")
        return exit_codes.REFUSED
    return run(args, plan)


if __name__ == "__main__":
    sys.exit(main())
