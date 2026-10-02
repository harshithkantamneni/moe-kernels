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

THE TWO PROCESSES (rental 2, 2026-10-01; docs/registered rental2-launch2).
Rental 1 ran the traces in the timed process, and its host time drifted with
the profiler's state. `--phase` now splits the run:
  timed   every timed cell and the HOST PROBE, in a process where the profiler
          is GUARDED OFF: `import torch` loads torch.profiler, so "never
          imports" cannot hold; instead `torch.profiler.profile.__init__` and
          `torch.autograd.profiler.profile.__init__` are patched to raise
          (`install_profiler_guard`), and `torch.autograd._profiler_enabled()`
          is asserted False before every probe and every cell. `--trace-treads`
          is refused here. The host probe (`host_probe`) measures, for every
          (arm, n, mode) BEFORE any timed cell and again AFTER the last,
          H_pre: the host wall time per iteration of the timed body (flush
          launch, start record, call, end record), 64 iterations, 3 repeats,
          median, with the GPU held busy by a preceding `torch.cuda._sleep`
          sized to outlast the loop, so no backpressure enters it; and
          h_flush, h_event, h_call alone the same way. A probe whose slowest
          iteration exceeds 10 x its median (something in the body
          synchronised), or whose sleep ended before its loop (the GPU was not
          held), is retried once and then written REFUSED. Files:
          hostprobe.csv, cells.csv, hostprobe-post.csv, manifest.json (phase,
          pid, profiler guard, profiler_enabled_ever).
  trace   the traces only (TR-E, TR-G), traces/*.json.gz, traces/parsed.json
          and manifest-trace.json; no timed cell.
  both    rental 1's single process, kept so its record reads as it ran.
The driver runs `timed` and then `trace` as two processes under one lock.

EXIT CODES (moe/bench/exit_codes.py): 0 every cell and trace ran; 2 refused
(bad arguments, no CUDA or no vLLM; and --dry-run, which prints the plan and
its minutes); 3 some cells, modes, probes or traces failed and the files are
written; 4 the profiler was found on in the timed process.
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
#: the host probe (rental 2): four parts x 3 repeats of 64 iterations, each
#: behind a sleep sized to outlast its loop, about 1 s per (arm, n, mode), once
#: before the cells and once after
SEC_PER_PROBE = 1.0
PHASES = ("both", "timed", "trace")
PROBE_ITERS = 64
PROBE_REPEATS = 3
#: a probe iteration this many times its median says the body synchronised
PROBE_MAX_OVER_MEDIAN = 10.0
#: the probe's hold: the sleep lasts this many loop times plus a margin
PROBE_HOLD_FACTOR = 2.0
PROBE_HOLD_MARGIN_MS = 5.0
PROBE_COLUMNS = (
    "model", "when", "mode", "arm", "tiles", "flush_mb", "status", "H_pre_ms",
    "h_flush_ms", "h_event_ms", "h_call_ms", "iters", "repeats", "max_over_median",
    "held", "sleep_cycles", "detail")
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


class ProfilerForbidden(RuntimeError):
    """The profiler was asked for, or found on, in the timed process."""


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
    p.add_argument("--phase", choices=PHASES, default="both",
                   help="timed: the cells and the host probe, the profiler guarded "
                        "off; trace: the traces alone; both: rental 1's one process")
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
    phase = getattr(args, "phase", "both")
    if phase == "timed" and args.trace_treads:
        raise Refused("--phase timed takes no --trace-treads: the traces run in the "
                      "trace phase's own process, so the timed process never starts the "
                      "profiler")
    if phase == "trace" and not args.trace_treads:
        raise Refused("--phase trace needs --trace-treads: it writes traces and nothing "
                      "else")
    cells = (0 if phase == "trace" else
             len(args.treads) * len(args.arms) * len(args.modes) * args.repeats)
    probes = (2 * len(args.treads) * len(args.arms) * len(args.modes)
              if phase == "timed" else 0)
    traces = []
    for n in args.trace_treads:
        for arm in args.arms:
            flushes = ["E240"] + (["E0", "E480"] if n in args.lever_trace_treads else [])
            for fm in flushes:
                traces.append({"arm": arm, "n": n, "flush": fm, "kind": "TR-E"})
                if "GR" in args.modes:
                    traces.append({"arm": arm, "n": n, "flush": fm, "kind": "TR-G"})
    minutes = (cells * SEC_PER_CELL / 60 + len(traces) * SEC_PER_TRACE / 60
               + probes * SEC_PER_PROBE / 60 + SETUP_MIN)
    return {"model": args.model, "phase": phase, "probes": probes,
            "treads": list(args.treads), "modes": list(args.modes),
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
        f"  phase {plan.get('phase', 'both')}: {plan['timed_cells']} timed cells at "
        f"~{SEC_PER_CELL:.1f} s, {len(plan['traces'])} "
        f"profiled bursts at ~{SEC_PER_TRACE:.0f} s, {plan.get('probes', 0)} host probes at "
        f"~{SEC_PER_PROBE:.0f} s, {SETUP_MIN:.0f} min weights and compile",
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
# rental 2: the profiler guard and the host probe
# --------------------------------------------------------------------------

def install_profiler_guard(torch_mod) -> list[str]:
    """Make every profiler entry point raise in this process; returns what was
    patched. `import torch` already loads torch.profiler (torch 2.13: both
    `torch.profiler` and `torch.autograd.profiler` are in sys.modules right
    after it), so the invariant is not "never imported" but "never started"."""
    def refuse(self, *a, **k):
        raise ProfilerForbidden("the profiler was started in the timed process; the "
                                "traces belong to --phase trace")
    patched = []
    for owner, label in ((getattr(torch_mod, "profiler", None), "torch.profiler.profile"),
                         (getattr(getattr(torch_mod, "autograd", None), "profiler", None),
                          "torch.autograd.profiler.profile")):
        cls = getattr(owner, "profile", None) if owner is not None else None
        if cls is not None:
            cls.__init__ = refuse
            patched.append(label)
    return patched


def assert_profiler_off(torch_mod) -> None:
    """Raises ProfilerForbidden when autograd's profiler reads on."""
    if torch_mod.autograd._profiler_enabled():
        raise ProfilerForbidden("torch.autograd._profiler_enabled() is True in the timed "
                                "process")


class ProbeGPU:
    """What the probe needs of the device: a sleep that holds it, a test of
    whether the sleep is still running, and a synchronise. The suite plants
    one; on the box it is torch's."""

    def __init__(self, torch_mod):
        self.torch = torch_mod

    def cycles_per_ms(self, cycles: int = 2_000_000) -> float:
        t = self.torch
        a, b = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
        t.cuda.synchronize()
        a.record()
        t.cuda._sleep(int(cycles))
        b.record()
        t.cuda.synchronize()
        return cycles / max(a.elapsed_time(b), 1e-3)

    def hold(self, cycles: int):
        self.torch.cuda._sleep(int(cycles))
        ev = self.torch.cuda.Event()
        ev.record()
        return ev

    def still_busy(self, ev) -> bool:
        return not ev.query()

    def body_event(self):
        """An event of the timed loop's own kind (start and end records), for the
        probe's copy of that body. The probe times the HOST, never with these."""
        return self.torch.cuda.Event(enable_timing=True)

    def sync(self) -> None:
        self.torch.cuda.synchronize()


def _loop_ns(fn, iters: int, clock) -> list[int]:
    out = []
    t = clock()
    for _ in range(iters):
        fn()
        u = clock()
        out.append(u - t)
        t = u
    return out


def probe_part(fn, gpu, *, iters: int = PROBE_ITERS, repeats: int = PROBE_REPEATS,
               cycles_per_ms: float, clock=time.perf_counter_ns) -> dict:
    """One part of the host probe: `repeats` loops of `iters` calls of `fn`,
    each loop behind a sleep sized to outlast it. Returns the median per-call
    host ms over the repeats' medians, the worst max/median, whether every
    loop finished while the sleep still ran, and the sleep it used. A repeat
    whose sleep ended first, or whose max/median exceeds PROBE_MAX_OVER_MEDIAN,
    is retried once with twice the sleep; after that the part is REFUSED."""
    gpu.sync()
    dry = _loop_ns(fn, iters, clock)          # sizes the hold; never reported
    gpu.sync()
    loop_ms = sum(dry) / 1e6
    cycles = int((PROBE_HOLD_FACTOR * loop_ms + PROBE_HOLD_MARGIN_MS) * cycles_per_ms)
    meds, worst, held_all, why = [], 0.0, True, ""
    for _r in range(repeats):
        for attempt in (0, 1):
            ev = gpu.hold(cycles)
            ns = _loop_ns(fn, iters, clock)
            held = gpu.still_busy(ev)
            gpu.sync()
            med = statistics.median(ns)
            ratio = max(ns) / med if med > 0 else float("inf")
            if held and ratio <= PROBE_MAX_OVER_MEDIAN:
                break
            if attempt == 0:
                cycles *= 2
        meds.append(med / 1e6)
        worst = max(worst, ratio)
        if not held:
            held_all = False
            why = "the sleep ended before the loop: the GPU was not held"
        elif ratio > PROBE_MAX_OVER_MEDIAN:
            why = (f"an iteration took {ratio:.1f} x the median: something in the body "
                   "synchronised")
    ok = held_all and worst <= PROBE_MAX_OVER_MEDIAN
    return {"ms": statistics.median(meds) if ok else None, "max_over_median": worst,
            "held": held_all, "sleep_cycles": cycles, "status": "ok" if ok else "refused",
            "detail": why}


def host_probe(parts: dict, gpu, *, iters: int = PROBE_ITERS,
               repeats: int = PROBE_REPEATS, cycles_per_ms: float,
               clock=time.perf_counter_ns) -> dict:
    """The host probe for one (arm, n, mode). `parts` maps "body" (one whole
    iteration of the timed loop: flush launch, start record, call, end
    record), "flush" (None when the mode flushes nothing), "event" and "call"
    to zero-argument callables. Returns H_pre and h_flush, h_event, h_call in
    ms, and the probe's own checks; status "refused" when any part was."""
    out, status, detail, worst, held = {}, "ok", [], 0.0, True
    cycles_used = 0
    for key, name in (("body", "H_pre_ms"), ("flush", "h_flush_ms"),
                      ("event", "h_event_ms"), ("call", "h_call_ms")):
        fn = parts.get(key)
        if fn is None:
            out[name] = 0.0 if key == "flush" else None
            continue
        r = probe_part(fn, gpu, iters=iters, repeats=repeats, cycles_per_ms=cycles_per_ms,
                       clock=clock)
        out[name] = r["ms"]
        worst = max(worst, r["max_over_median"])
        held = held and r["held"]
        cycles_used = max(cycles_used, r["sleep_cycles"])
        if r["status"] != "ok":
            status = "refused"
            detail.append(f"{key}: {r['detail']}")
    out.update(status=status, iters=iters, repeats=repeats, max_over_median=worst,
               held=held, sleep_cycles=cycles_used, detail="; ".join(detail))
    return out


def probe_parts(fn, flusher, event_factory) -> dict:
    """The timed loop's body and its parts, as `mirror_time_cell`'s duty timer
    runs them (flush, start event, call, end event)."""
    start, end = event_factory(), event_factory()

    def body():
        if flusher is not None:
            flusher.flush()
        start.record()
        fn()
        end.record()
    return {"body": body, "flush": (flusher.flush if flusher is not None else None),
            "event": start.record, "call": fn}


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
    phase = plan.get("phase", "both")
    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    cfg = MODEL_CONFIGS[args.model]
    os.environ.setdefault("TRITON_CACHE_DIR", str(out / "triton-cache"))
    guard = install_profiler_guard(torch) if phase == "timed" else []
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
    manifest = {"tool": "scripts/launch_floor.py", "plan": plan, "phase": phase,
                "pid": os.getpid(), "profiler_guard": guard,
                "profiler_enabled_ever": False, "profiler_checks": 0,
                "override_hook": where,
                "reference_clock_mhz": reference_clock, "reference_clock_source": clock_source,
                "flush_mb": {m: (int(f.megabytes) if f is not None else 0)
                             for m, f in flushers.items()},
                "weights_bytes": int(delta), "versions": PW._package_versions(),
                "device": {"name": torch.cuda.get_device_name(0),
                           "uuid": PW.device_identity()},
                "host": host_record(), "findings": [], "started": time.time()}
    failed = 0
    parsed = {}

    def check_profiler():
        if phase != "timed":
            return
        manifest["profiler_checks"] += 1
        try:
            assert_profiler_off(torch)
        except ProfilerForbidden:
            manifest["profiler_enabled_ever"] = True
            raise

    def calls_for(n):
        tokens, x, ids_by_arm, weights, kw = PW.arm_inputs(
            cfg, n, BLOCK_M, copies, args.seed, "bf16", w1.dtype)
        return tokens, {arm: PW.arm_call(fused_experts, arm, w1, w2, nw1, nw2, declared, x,
                                         ids_by_arm, weights, kw) for arm in args.arms}

    def capture(call, arm, n, rep):
        try:
            return graph_call(call, override_config=override_config, conf=conf), ""
        except Exception as exc:                                  # noqa: BLE001
            why = f"{type(exc).__name__}: {exc}"
            manifest["findings"].append({"arm": arm, "n": n, "repeat": rep, "graph": why})
            return None, why

    def probe_all(when: str, path: Path) -> int:
        """The host probe for every (arm, n, mode), written to `path`."""
        bad = 0
        gpu = ProbeGPU(torch)
        cpm = gpu.cycles_per_ms()
        manifest.setdefault("probe_cycles_per_ms", {})[when] = cpm
        with open(path, "w", newline="") as ph:
            pw = csv.DictWriter(ph, fieldnames=PROBE_COLUMNS)
            pw.writeheader()
            for n in args.treads:
                _tokens, calls = calls_for(n)
                for arm in args.arms:
                    replay = None
                    if "GR" in args.modes:
                        replay, _why = capture(calls[arm], arm, n, -1)
                    for mode in args.modes:
                        fl = flushers[mode]
                        row = {k: "" for k in PROBE_COLUMNS}
                        row.update(model=args.model, when=when, mode=mode, arm=arm, tiles=n,
                                   flush_mb=int(fl.megabytes) if fl is not None else 0)
                        fn = replay if mode == "GR" else calls[arm]
                        if fn is None:
                            row.update(status="refused", detail="GR NOT CAPTURED")
                            bad += 1
                        else:
                            check_profiler()
                            with override_config(conf):
                                fn()
                                torch.cuda.synchronize()
                                r = host_probe(probe_parts(fn, fl, gpu.body_event),
                                               gpu, cycles_per_ms=cpm)
                            row.update({k: r[k] for k in r if k in row})
                            bad += r["status"] != "ok"
                        pw.writerow(row)
                        ph.flush()
                    replay = None
        return bad

    def time_cells(writer, fh) -> int:
        bad = 0
        for rep in range(args.repeats):
            treads = args.treads if rep % 2 == 0 else list(reversed(args.treads))
            for n in treads:
                tokens, calls = calls_for(n)
                order = [args.arms[(i + rep) % len(args.arms)] for i in range(len(args.arms))]
                for arm in order:
                    call = calls[arm]
                    ident = dict(model=args.model, arm=arm, tiles=n, repeat=rep,
                                 group_m=args.group_m, block_m=BLOCK_M, tokens=tokens,
                                 copies=PW.copies_read(arm, n),
                                 experts_declared=declared[arm], duty=args.duty)
                    replay, why = (capture(call, arm, n, rep) if "GR" in args.modes
                                   else (None, ""))
                    for mode in args.modes:
                        fl = flushers[mode]
                        fn = replay if mode == "GR" else call
                        row_id = dict(ident, mode=mode,
                                      flush_mb=int(fl.megabytes) if fl is not None else 0)
                        if mode == "GR" and replay is None:
                            writer.writerow(row_from(None, None, **row_id, status="failed",
                                                     detail=f"GR NOT CAPTURED: {why}"))
                            bad += 1
                            continue
                        check_profiler()
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
                            bad += 1
                        fh.flush()
                        print(f"  rep{rep} {mode:4s} {arm:8s} n={n}", flush=True)
                    if phase == "both" and rep == 0 and n in args.trace_treads:
                        bad += trace_cell(arm, n, call, replay)
                    replay = None
        return bad

    def trace_cell(arm, n, call, replay) -> int:
        bad = 0
        for tr in [t for t in plan["traces"] if t["arm"] == arm and t["n"] == n]:
            name = f"{tr['kind']}-{tr['flush']}-{arm}-n{n}.json.gz"
            fn = call if tr["kind"] == "TR-E" else replay
            if fn is None:
                bad += 1
                continue
            try:
                with override_config(conf):
                    trace_burst(fn, flushers[tr["flush"]], out / "traces" / name)
                with gzip.open(out / "traces" / name, "rt") as f:
                    parsed[name] = parse_trace(json.load(f))
            except Exception as exc:                              # noqa: BLE001
                manifest["findings"].append({"trace": name, "error": str(exc)})
                bad += 1
        return bad

    manifest_name = "manifest-trace.json" if phase == "trace" else "manifest.json"
    fh = None
    try:
        if phase == "trace":
            for n in args.trace_treads:
                _tokens, calls = calls_for(n)
                for arm in args.arms:
                    replay = (capture(calls[arm], arm, n, 0)[0] if "GR" in args.modes
                              else None)
                    failed += trace_cell(arm, n, calls[arm], replay)
        else:
            if phase == "timed":
                failed += probe_all("pre", out / "hostprobe.csv")
            fh = open(out / "cells.csv", "w", newline="")
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            failed += time_cells(writer, fh)
            fh.close()
            fh = None
            if phase == "timed":
                failed += probe_all("post", out / "hostprobe-post.csv")
    except ProfilerForbidden as exc:
        manifest["findings"].append({"profiler": str(exc)})
        manifest["profiler_enabled_ever"] = True
        print(f"PROFILER ON IN THE TIMED PROCESS: {exc}")
        failed = -1
    finally:
        if fh is not None:
            fh.close()
        manifest["ended"] = time.time()
        manifest["failed"] = failed
        if phase != "timed":
            (out / "traces" / "parsed.json").write_text(json.dumps(parsed, indent=1))
        (out / manifest_name).write_text(json.dumps(manifest, indent=1, default=str))
    if failed < 0:
        return exit_codes.ERROR
    print(f"LAUNCH FLOOR {args.model} phase {phase}: {plan['timed_cells']} cells, "
          f"{len(plan['traces']) if phase != 'timed' else 0} traces, {failed} failed; {out}")
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
