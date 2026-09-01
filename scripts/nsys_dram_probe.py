#!/usr/bin/env python
"""Can this pod measure DRAM traffic with nsys, since ncu will not let it?

    python scripts/nsys_dram_probe.py --explain            # the physics, no GPU
    python scripts/nsys_dram_probe.py                      # the probe, ~2 minutes
    python scripts/nsys_dram_probe.py --calibrate          # + a known-traffic check
    python scripts/nsys_dram_probe.py --measure            # + the MoE cell vs the model
    python scripts/nsys_dram_probe.py --report r.sqlite    # parse one you already have

WHY THIS EXISTS. `ncu` fails on RunPod with ERR_NVGPUCTRPERM because hardware
performance counters need `NVreg_RestrictProfilingToAdminUsers=0`, a host
kernel-module flag a container tenant cannot set.
`scripts/profile_open_questions.sh` concluded from that "Q1 traffic -> ncu ONLY.
dram__bytes_read.sum is a counter; nothing traces it", and docs/RUNPOD.md,
docs/POD_RUNBOOK.md and docs/FINDINGS.md have repeated it ever since.

`nsys --gpu-metrics-device` neither traces nor uses the CUPTI profiling API that
ncu is blocked on: it SAMPLES the GPU's hardware performance monitor through a
separate path. Whether that path is gated on the same flag is an empirical
question, `grep` finds zero references to the option anywhere in this repository,
and docs/FINDINGS.md calls it "the open path, not a closed door". This is the
probe that was never run.

WHAT IT WOULD BUY, WHICH IS WHY IT BELONGS IN PRE-FLIGHT. Every byte figure in
this study is compulsory-traffic arithmetic. `implied_traffic_ratio` is time x
achievable-bandwidth over MODELLED bytes, an inference and not a measurement,
and it carries the caveat attached to the study's strongest positive result.
`alpha`, refit 2026-08-31 from 0.10 to 0.558, is fitted through the same model,
and the tile-corrected roofline rests entirely on it. If nsys yields real DRAM
read bytes over a kernel window then alpha becomes directly measurable as the
extra bytes a second M-tile costs, the traffic ratio stops being an inference,
and the byte model gets validated or refuted for the first time.

WHAT THIS PROBE REFUSES TO DO. Sampled GPU metrics are DEVICE WIDE and coarse.
Three things follow and none of them are hidden here:

  * A sample covers the whole GPU. Isolation is a scheduling assumption, not a
    measurement, so every workload below goes idle for a beat before its timed
    loop and the parser reports the DRAM traffic the sampler saw during that
    idle gap. A non-zero idle baseline means the number is not ours. What stays
    uncontrolled is a NEIGHBOUR process arriving mid-window on a shared pod, so
    the report records `nvidia-smi`'s compute-apps list before and after.
  * The edge quantisation is per WINDOW, not per sample. A 54 us kernel, which
    is what a T=1 fused MoE cell costs on an H200, holds 0.5 samples at the nsys
    default of 10 kHz and about 11 at the 200 kHz ceiling. A single launch is
    not measurable at any rate the tool offers. Every workload here therefore
    enqueues its kernel back to back so the launches MERGE into one contiguous
    window, which is the only way the edge term shrinks.
  * The values may be a percentage of peak rather than bytes, in which case
    converting them needs the measured DRAM ceiling and the result inherits that
    calibration's uncertainty. It is still independent of the COMPULSORY BYTE
    MODEL, which is the thing under test, and `moe.bench.nsys_metrics` reports
    which route it took so the two can never be confused.

THE TIME BUDGET. Six rungs, each a fresh nsys plus a child that imports torch
and streams for 1.5 s plus an export, is roughly fifteen seconds a rung, so a
total refusal costs about ninety seconds and a success stops at the first rung
in about fifteen. The child's torch import dominates, not the profiling.

If the honest answer is that nsys cannot attribute traffic finely enough, this
prints that and exits 3. That is a result, and it closes an open question in
docs/FINDINGS.md either way.

HOW IT RUNS. The script is both the driver and the profiled child: `nsys profile
... python scripts/nsys_dram_probe.py --workload stream` keeps the thing being
measured inside the file that explains it, so no separate harness invocation can
drift away from the byte model the answer is scored against.

The parsing, the sampling arithmetic and the comparison all live in
`moe/bench/nsys_metrics.py` and are tested off GPU in `tests/test_nsys_metrics.py`.
This file owns the subprocesses and the CUDA, which is exactly the part a laptop
cannot exercise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench import nsys_metrics as nm  # noqa: E402
from moe.bench.bytes_model import field_bytes, weight_bytes_for_stage  # noqa: E402
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec  # noqa: E402

#: Long enough that a merged window holds thousands of samples even at the 10 kHz
#: default, short enough that the discovery ladder fits in the two-minute budget.
PROBE_SECONDS = 1.5

#: The gap the workload leaves between setup and the timed loop. It has to be
#: many sample periods so the idle baseline is itself measurable, and it is what
#: makes `longest_window` pick the timed loop rather than the warmup.
IDLE_GAP_S = 0.25

#: The rate the ladder falls back to when the requested one is refused. A build
#: that rejects the 200 kHz ceiling still answers the question at the rate nsys
#: uses unasked, just with a twenty-times worse edge term, and "the flag was
#: rejected" is a different answer from "the sampler is blocked". Distinguishing
#: them is the whole reason the ladder varies one thing per rung.

#: stderr fragments that say WHY an attempt failed, so the report can classify
#: rather than only quote. Order matters: the permission cases are checked first
#: because a permission failure often also prints a generic warning.
_STDERR_CLASSES = (
    ("ERR_NVGPUCTRPERM", "counters blocked by the host module flag, same wall as ncu"),
    ("insufficient permission", "permission denied to the GPU metrics sampler"),
    ("administrative privileges", "permission denied to the GPU metrics sampler"),
    ("permission", "permission denied to the GPU metrics sampler"),
    ("not supported", "this device or driver has no sampler for that metric set"),
    ("Unrecognized command line", "flag spelling wrong for this nsys build"),
    ("unrecognized option", "flag spelling wrong for this nsys build"),
    ("Invalid argument", "flag accepted but its value was rejected"),
    ("could not be started", "nsys could not launch the child at all"),
)


# --------------------------------------------------------------------------
# The workloads. These run as the CHILD, under nsys, and never in the driver.
# Both share a shape: allocate, warm up, go idle for a beat, then enqueue the
# kernel back to back with no synchronisation inside the loop, so the launches
# land contiguously and merge into one window.
# --------------------------------------------------------------------------


def _cuda_or_die():
    try:
        import torch
    except ImportError:
        print("no torch in this interpreter, so there is no workload to profile",
              file=sys.stderr)
        raise SystemExit(2) from None
    if not torch.cuda.is_available():
        print("no CUDA device visible to the child. nsys can profile it and will "
              "record nothing, which is not the question being asked.",
              file=sys.stderr)
        raise SystemExit(2)
    return torch


def workload_stream(buffer_mb: float, seconds: float) -> dict:
    """Read a buffer far larger than L2, back to back, for a KNOWN byte count.

    This is the calibration case and it is the load-bearing one: it is the only
    workload here whose true DRAM traffic is known without a model, so it tests
    the sampler, the unit interpretation and the window attribution all at once
    against an answer nobody can argue with. If this comes back at 0.4x or 3x,
    nothing else measured in the session is worth reading.

    The reduction runs along the CONTIGUOUS axis of a 2-D view, which is the
    shape C4 established is DRAM limited: `torch.sum` on a 1-D buffer into a
    scalar is a full tree reduction and bounds on ATen's reduction instead,
    which is the exact flaw that understated this project's read ceiling by
    1.85% and hid on the H200 for weeks.
    """
    torch = _cuda_or_die()
    cols = 4096
    rows = max(1, int(buffer_mb * 1e6 / (cols * 2)))
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    read_bytes = x.element_size() * x.numel()
    out_bytes = rows * x.element_size()

    for _ in range(3):
        x.sum(dim=1)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    x.sum(dim=1)
    torch.cuda.synchronize()
    per_launch = max(time.perf_counter() - t0, 1e-6)
    iters = max(32, int(seconds / per_launch))

    time.sleep(IDLE_GAP_S)          # the idle gap the baseline is read from

    started = time.perf_counter()
    for _ in range(iters):
        x.sum(dim=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "workload": "stream",
        "iters": iters,
        "seconds": elapsed,
        "per_launch_ms": elapsed / iters * 1e3,
        "known_read_bytes_per_launch": read_bytes,
        "known_write_bytes_per_launch": out_bytes,
        "buffer_gb": read_bytes / 1e9,
        "achieved_gbps": (read_bytes + out_bytes) * iters / elapsed / 1e9,
    }


def workload_grouped_mm(model: str, tokens: int, seconds: float) -> dict:
    """The MoE cell the byte model is being checked against: one up-stage GEMM.

    Deliberately the SAME cell `scripts/profile_open_questions.sh` reserved for
    ncu, deepseek-v3 up at T=4096 uniform, where one expert's w1 is 58.72 MB and
    the two candidate traffic models are 45% apart:

        256 active experts x 58.72 MB = 15.03 GB   -> traffic is active x W
        370 M-tiles        x 58.72 MB = 21.73 GB   -> traffic is M_tiles x W

    Routing is EXACTLY balanced rather than sampled uniform, for the reason
    `scripts/block_m_crossing_sweep.py` gives: sampled routing puts spread on
    rows per expert, which smears the tile count and therefore the very quantity
    the two candidate models disagree about.
    """
    torch = _cuda_or_die()
    grouped_mm = (getattr(torch.nn.functional, "grouped_mm", None)
                  or getattr(torch, "_grouped_mm", None))
    if grouped_mm is None:
        print(f"torch {torch.__version__} has neither "
              "torch.nn.functional.grouped_mm nor torch._grouped_mm",
              file=sys.stderr)
        raise SystemExit(2)

    cfg = MODEL_CONFIGS[model]
    e, h, f, k = cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k
    ntot = tokens * k
    per_expert = ntot // e
    if per_expert * e != ntot:
        print(f"T={tokens} k={k} E={e} does not divide evenly, so routing cannot "
              "be exactly balanced and the tile count would be smeared",
              file=sys.stderr)
        raise SystemExit(2)

    weight_bytes = e * h * 2 * f * 2
    free, total = torch.cuda.mem_get_info()
    need = weight_bytes + ntot * h * 2 + ntot * 2 * f * 2
    if need > free * 0.9:
        print(f"this cell needs about {need / 1e9:.1f} GB and {free / 1e9:.1f} GB "
              f"is free of {total / 1e9:.1f} GB. Use --tokens or --model to pick "
              "a smaller cell; the probe's conclusion does not depend on which "
              "cell it ran.", file=sys.stderr)
        raise SystemExit(2)

    wt = torch.randn(e, h, 2 * f, device="cuda", dtype=torch.bfloat16) * 0.02
    x_perm = torch.randn(ntot, h, device="cuda", dtype=torch.bfloat16)
    offs = torch.arange(1, e + 1, device="cuda", dtype=torch.int32) * per_expert

    for _ in range(3):
        grouped_mm(x_perm, wt, offs=offs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    grouped_mm(x_perm, wt, offs=offs)
    torch.cuda.synchronize()
    per_launch = max(time.perf_counter() - t0, 1e-6)
    iters = max(16, int(seconds / per_launch))

    time.sleep(IDLE_GAP_S)

    started = time.perf_counter()
    for _ in range(iters):
        grouped_mm(x_perm, wt, offs=offs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "workload": "grouped-mm",
        "model": model,
        "tokens": tokens,
        "rows_per_expert": per_expert,
        "active_experts": e,
        "iters": iters,
        "seconds": elapsed,
        "per_launch_ms": elapsed / iters * 1e3,
    }


def modelled_up_stage_bytes(model: str, tokens: int) -> dict:
    """Compulsory bytes for one up-stage GEMM, term by term.

    Spelled out rather than taken whole from `bytes_model.pipeline_cost` because
    the whole point of the comparison is to see WHICH term the hardware
    disagrees with, and a single total cannot say. The terms are exactly the
    ones `span_cost` sums for a bare `up_gemm`: its contract reads `x_perm` and
    `expert_offsets`, writes `h_up`, and pays one full weight read per ACTIVE
    expert.
    """
    cfg = MODEL_CONFIGS[model]
    spec = BenchSpec(cfg, num_tokens=tokens, dtype="bf16",
                     routing=RoutingSpec("uniform"), seed=0)
    fb = field_bytes(spec)
    weights = weight_bytes_for_stage(spec, "up_gemm", cfg.num_experts)
    terms = {
        "weights (active experts x w1)": weights,
        "read x_perm": fb["x_perm"],
        "read expert_offsets": fb["expert_offsets"],
        "write h_up": fb["h_up"],
    }
    return {"terms": terms, "total": sum(terms.values())}


# --------------------------------------------------------------------------
# Discovery. Nothing here assumes a flag spelling: the installed binary is asked
# what it offers, because branching on presence instead of capability is the
# mistake this repo already made once with ncu.
# --------------------------------------------------------------------------


@dataclass
class Shell:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def head(self, n: int = 6) -> str:
        lines = [ln for ln in (self.stderr or self.stdout).splitlines() if ln.strip()]
        return "\n".join(lines[:n]) or "(no output)"


def shell(argv, timeout: float = 90.0) -> Shell:
    t0 = time.perf_counter()
    try:
        p = subprocess.run(list(argv), capture_output=True, text=True,
                           timeout=timeout, check=False)
    except FileNotFoundError as exc:
        return Shell(tuple(argv), 127, "", str(exc), time.perf_counter() - t0)
    except subprocess.TimeoutExpired:
        return Shell(tuple(argv), -1, "", f"timed out after {timeout:.0f}s",
                     time.perf_counter() - t0, timed_out=True)
    return Shell(tuple(argv), p.returncode, p.stdout, p.stderr,
                 time.perf_counter() - t0)


def classify(text: str) -> str:
    low = text.lower()
    for needle, meaning in _STDERR_CLASSES:
        if needle.lower() in low:
            return meaning
    return "no recognised cause in the output"


@dataclass
class Attempt:
    """One candidate invocation, and everything needed to say why it lost.

    `metrics_rows` is the real verdict and the exit code is not: nsys exits 0
    while printing a warning and writing a report with no metrics table in it,
    which would read as success to anything that only checked the return code.
    """

    label: str
    argv: tuple[str, ...]
    returncode: int
    seconds: float
    stderr_head: str
    cause: str
    report: str | None = None
    sqlite: str | None = None
    metrics_rows: int = 0
    dram_metrics: tuple[str, ...] = ()
    verdict: str = ""


def nsys_binary(explicit: str | None) -> str:
    found = explicit or shutil.which("nsys")
    if not found:
        raise nm.NsysUnavailable(
            "no `nsys` on PATH. On a RunPod pod:\n"
            "  apt-get update && apt-get install -y nsight-systems\n"
            "Note that this pulls nvidia-profiler and therefore `ncu` with it, "
            "and ncu being PRESENT says nothing about counters being readable: "
            "that is the trap scripts/profile_open_questions.sh documents.")
    return found


@dataclass
class Discovery:
    binary: str
    version: str
    version_raw: str
    device_flag: str
    set_flag: str | None
    frequency_flag: str | None
    offered: tuple[str, ...]
    metric_sets: tuple[str, ...]
    chosen_set: str | None
    device_name: str
    compute_apps: str


def gpu_name() -> str:
    out = shell(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=20)
    return out.stdout.strip().splitlines()[0].strip() if out.ok and out.stdout.strip() \
        else "unknown"


def compute_apps() -> str:
    """Who else is on this device. The uncontrolled variable, recorded not fixed.

    A device-wide sampler cannot exclude a neighbour, so the only honest thing
    to do is state whether there was one at the time. On a shared pod a process
    that arrives mid-window is invisible to this snapshot, which is why the
    report takes it before AND after.
    """
    out = shell(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader"], timeout=20)
    if not out.ok:
        return "nvidia-smi unavailable"
    return out.stdout.strip() or "none"


def discover(binary: str) -> Discovery:
    version = nm.parse_nsys_version(
        (shell([binary, "--version"], timeout=30).stdout or "")
        + shell([binary, "--version"], timeout=30).stderr)
    help_text = shell([binary, "profile", "--help"], timeout=30)
    support = nm.parse_gpu_metrics_support(help_text.stdout + help_text.stderr)
    sets: tuple[nm.MetricSet, ...] = ()
    if support.set_flag:
        listed = shell([binary, "profile", f"{support.set_flag}=help"], timeout=30)
        sets = nm.parse_metric_sets(listed.stdout + listed.stderr)
    name = gpu_name()
    chosen = nm.select_metric_set(sets, name)
    return Discovery(
        binary=binary,
        version=str(version),
        version_raw=version.raw,
        device_flag=support.device_flag,
        set_flag=support.set_flag,
        frequency_flag=support.frequency_flag,
        offered=support.offered,
        metric_sets=tuple(f"[{s.index}] {s.chip} {s.description}" for s in sets),
        chosen_set=chosen.chip if chosen else None,
        device_name=name,
        compute_apps=compute_apps(),
    )


def candidate_invocations(d: Discovery, out_dir: Path, child_argv,
                          sample_hz: float) -> list[tuple[str, list[str]]]:
    """The ladder, most informative first, each one differing in ONE thing.

    No `--` separator anywhere: Nsight Systems 2022.4, which is what Ubuntu
    ships, rejects it as an end-of-options marker, and `scripts/kernel_name.py`
    already records that a tool which has to agree with you about its own flags
    is a dependency worth avoiding. The app and its arguments simply follow the
    options, which every version accepts.
    """
    base = [d.binary, "profile", "--force-overwrite", "true",
            "--trace", "cuda", "--sample", "none", "--cpuctxsw", "none"]
    out: list[tuple[str, list[str]]] = []

    def add(label: str, extra: list[str]) -> None:
        target = out_dir / f"probe-{len(out)}"
        argv = base + extra + ["--output", str(target)] + list(child_argv)
        out.append((label, argv))

    metrics = [f"{d.device_flag}=all"]
    if d.frequency_flag:
        metrics += [f"{d.frequency_flag}={int(sample_hz)}"]
    if d.set_flag and d.chosen_set:
        add(f"{d.device_flag}=all with {d.set_flag}={d.chosen_set} at "
            f"{sample_hz / 1e3:.0f} kHz",
            metrics + [f"{d.set_flag}={d.chosen_set}"])
    add(f"{d.device_flag}=all at {sample_hz / 1e3:.0f} kHz, nsys picks the set",
        metrics)
    add(f"{d.device_flag}=0, in case `all` is the part that is refused",
        [f"{d.device_flag}=0"]
        + ([f"{d.frequency_flag}={int(sample_hz)}"] if d.frequency_flag else []))
    if d.frequency_flag and sample_hz != nm.DEFAULT_SAMPLE_HZ:
        add(f"{d.device_flag}=all at the nsys default "
            f"{nm.DEFAULT_SAMPLE_HZ / 1e3:.0f} kHz, in case the rate is what is "
            "refused",
            [f"{d.device_flag}=all",
             f"{d.frequency_flag}={int(nm.DEFAULT_SAMPLE_HZ)}"])
    other = next((f for f in nm.GPU_METRICS_DEVICE_FLAGS if f != d.device_flag), None)
    if other:
        add(f"{other}=all, the other spelling, in case --help under-reports",
            [f"{other}=all"])
    add("no gpu metrics at all: the CONTROL, which says whether nsys itself works "
        "here", [])
    return out


def export_sqlite(binary: str, report: Path) -> tuple[Path | None, Shell]:
    target = report.with_suffix(".sqlite")
    if target.exists():
        target.unlink()
    out = shell([binary, "export", "--type", "sqlite", "--output", str(target),
                 str(report)], timeout=180)
    return (target if target.exists() else None), out


def inspect(sqlite_path: Path) -> tuple[int, tuple[str, ...], str]:
    """Rows in the metrics table and which DRAM metrics the catalogue names.

    The only test that distinguishes "nsys accepted the flag" from "nsys sampled
    something", which is the distinction the whole probe turns on.
    """
    conn = nm.open_report(sqlite_path)
    try:
        tables = nm.table_names(conn)
        if nm.GPU_METRIC_TABLE not in tables:
            return 0, (), (f"no {nm.GPU_METRIC_TABLE} table: nsys wrote a report "
                           "with no sampled metrics in it")
        rows = conn.execute(f"SELECT COUNT(*) FROM {nm.GPU_METRIC_TABLE}").fetchone()[0]
        if not rows:
            return 0, (), (f"{nm.GPU_METRIC_TABLE} exists and is EMPTY: the flag "
                           "was accepted and nothing was sampled")
        try:
            cat = nm.metric_catalogue(conn)
        except nm.NsysProbeRefused as exc:
            return int(rows), (), str(exc)
        dram = tuple(m.name for m in cat if "dram" in m.name.lower())
        if not dram:
            return int(rows), (), (f"{rows} samples over {len(cat)} metrics, none "
                                   "of which names DRAM")
        try:
            found = nm.find_dram_metrics(cat)
        except nm.MetricNotFound as exc:
            return int(rows), dram, str(exc)
        split = ("read and write are split" if found.write
                 else "READ ONLY, this set does not split writes")
        return int(rows), dram, (f"{rows} samples, DRAM metrics present, {split}, "
                                 f"unit reads as {found.read.unit_kind.value}")
    finally:
        conn.close()


def run_ladder(d: Discovery, out_dir: Path, child_argv, sample_hz: float,
               timeout: float) -> list[Attempt]:
    attempts: list[Attempt] = []
    for label, argv in candidate_invocations(d, out_dir, child_argv, sample_hz):
        res = shell(argv, timeout=timeout)
        report = next((Path(a) for i, a in enumerate(argv)
                       if i and argv[i - 1] == "--output"), None)
        rep_file = None
        for suffix in (".nsys-rep", ".qdrep", ""):
            cand = Path(str(report) + suffix) if report else None
            if cand and cand.is_file():
                rep_file = cand
                break
        att = Attempt(
            label=label, argv=tuple(argv), returncode=res.returncode,
            seconds=res.seconds, stderr_head=res.head(),
            cause=classify(res.stderr + res.stdout),
            report=str(rep_file) if rep_file else None,
        )
        if rep_file is None:
            att.verdict = "no report file was written"
            attempts.append(att)
            continue
        sqlite_path, exported = export_sqlite(d.binary, rep_file)
        if sqlite_path is None:
            att.verdict = f"export to sqlite failed: {exported.head(3)}"
            attempts.append(att)
            continue
        att.sqlite = str(sqlite_path)
        try:
            rows, dram, verdict = inspect(sqlite_path)
        except nm.NsysProbeRefused as exc:
            rows, dram, verdict = 0, (), str(exc)
        att.metrics_rows, att.dram_metrics, att.verdict = rows, dram, verdict
        attempts.append(att)
        if rows and dram:
            break                    # the question is answered, stop spending time
    return attempts


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------


@dataclass
class Report:
    started: str
    elapsed_s: float = 0.0
    discovery: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    single_launch: str = ""
    calibration: dict = field(default_factory=dict)
    cell: dict = field(default_factory=dict)
    verdict: str = ""
    compute_apps_after: str = ""


def explain() -> str:
    """The arithmetic that decides this before any GPU is rented."""
    lines = ["THE SAMPLING ARITHMETIC, which needs no GPU and settles the shape of",
             "the answer before anything is run.", ""]
    for hz in (nm.DEFAULT_SAMPLE_HZ, 100_000.0, nm.MAX_SAMPLE_HZ):
        r = nm.single_launch_verdict(nm.SHORT_KERNEL_US, hz)
        lines.append(f"  one {nm.SHORT_KERNEL_US:.0f} us launch at "
                     f"{hz / 1e3:6.0f} kHz -> {r.n_samples:6.2f} samples, "
                     f"{'usable' if r.ok else 'NOT usable'}")
    lines += ["", "So a single launch of the cell this study cares about cannot be "
              "measured", "at any rate nsys offers. Back-to-back launches merged "
              "into one window:", ""]
    for ms in (1.0, 10.0, 100.0, 1000.0):
        r = nm.resolve(int(ms * 1e6), 1, nm.DEFAULT_SAMPLE_HZ)
        lines.append(f"  {ms:7.1f} ms contiguous at 10 kHz -> {r.n_samples:8.1f} "
                     f"samples, edge {r.edge_error * 100:5.2f}%, alpha to "
                     f"+/-{r.alpha_band():.4f}")
    lines += ["", "and the SAME 10 ms of kernel time profiled as separate windows "
              "rather than merged:", ""]
    for n in (1, 10, 100, 1000):
        r = nm.resolve(int(10e6), n, nm.DEFAULT_SAMPLE_HZ)
        lines.append(f"  {n:5d} window(s) over 10 ms total -> edge "
                     f"{r.edge_error * 100:7.2f}%, "
                     f"{'usable' if r.ok else 'NOT usable'}")
    lines += ["",
              "The edge term is per WINDOW. Adding launches without merging them "
              "buys",
              "samples and no accuracy, which is the single thing to get right in "
              "the",
              "workload design and the reason every workload here goes idle before "
              "its",
              "timed loop and never synchronises inside it.",
              "",
              "WHAT PRECISION THE STUDY NEEDS, at two M-tiles per expert:",
              f"  tell alpha=0.558 from the retracted 0.10   traffic to "
              f"{nm.DISCRIMINATE_ALPHA_REL * 100:.0f}%",
              f"  pin alpha to +/-0.05                        traffic to "
              f"{nm.MEASURE_ALPHA_REL * 100:.1f}%", ""]
    lines.append("and what a given traffic precision buys, per M-tile count:")
    lines.append("  traffic err |" + "".join(f"  n={n:<4d}" for n in (2, 4, 8, 16)))
    for rel in (0.02, 0.05, 0.10, 0.20):
        band = "".join(f"  {nm.alpha_uncertainty(rel, n_tiles=n):6.3f}"
                       for n in (2, 4, 8, 16))
        lines.append(f"    {rel * 100:5.1f}%    |{band}")
    lines.append("More tiles per expert always gives a SHARPER alpha, so a probe "
                 "that cannot")
    lines.append("resolve a two-tile cell may still resolve an eight-tile one.")
    return "\n".join(lines)


def results_root() -> Path:
    """`$MOE_RESULTS_DIR`, else the network volume, else the repo.

    Same order `scripts/run_all.sh` and `scripts/block_m_crossing_sweep.py`
    resolve it in, so a probe run on a pod survives teardown alongside the arms.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return Path(__file__).resolve().parents[1] / "results"


def child_argv(args, workload: str) -> list[str]:
    argv = [sys.executable, str(Path(__file__).resolve()), "--workload", workload,
            "--seconds", str(args.seconds)]
    if workload == "stream":
        argv += ["--buffer-mb", str(args.buffer_mb)]
    else:
        argv += ["--model", args.model, "--tokens", str(args.tokens)]
    return argv


def rerun_with(winning_argv, target: Path, child) -> list[str]:
    """The winning invocation again, pointed at a new report and a new child.

    Reusing the argv that WON rather than rebuilding it means the measurement
    runs under exactly the flags the ladder proved work here, including whatever
    fallback rung it landed on. Rebuilding would quietly re-introduce the
    preferred rung that had already failed.
    """
    argv = list(winning_argv)
    cut = argv.index("--output")
    return argv[:cut] + ["--output", str(target)] + list(child)


def child_json(res: Shell) -> dict:
    """The workload's own summary, which it prints as one json line on stdout.

    Read from stdout rather than a file so nothing has to agree about a path
    across the nsys boundary, and matched on a leading brace so nsys's own
    banner cannot be mistaken for it.
    """
    for line in res.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return {}


def measure_from_report(sqlite_path: Path, sample_hz: float, peak_gbps: float | None,
                        modelled_per_launch: float, launches: int | None = None
                        ) -> dict:
    """Parse one exported report and score it against the byte model.

    Split out so `--report` can re-score a session's artefacts on a laptop
    months later, which is the difference between a probe and a reusable route.

    `launches` defaults to the count the TRACE carries, which is what makes the
    re-scoring route usable at all: a report handed over months later does not
    come with the workload's own iteration count. When the caller does supply
    one, both are reported and a disagreement is flagged rather than averaged,
    because the two counts disagreeing means the window selection picked
    something other than the timed loop and every byte below it is attributed to
    the wrong kernels.
    """
    conn = nm.open_report(sqlite_path)
    try:
        # The rate the tool DELIVERED, before anything is computed from the rate
        # it was ASKED for. `--gpu-metrics-frequency` is a request and a build
        # may clamp it without saying so; every sample count and edge term below
        # divides by the requested period, so an unnoticed clamp would leave the
        # traffic total right and its stated confidence wrong by the clamp
        # factor. Refusals here are recorded, not raised: a rate that could not
        # be measured must not stop a traffic figure being reported, it must
        # stop that figure being called resolved.
        rate_note, rate_ok = "", None
        try:
            cat = nm.metric_catalogue(conn)
            found = nm.find_dram_metrics(cat)
            rate = nm.observed_sample_hz(conn, found.read, sample_hz)
            rate_note, rate_ok = str(rate), rate.honoured
        except nm.NsysProbeRefused as exc:
            rate_note, rate_ok = f"delivered rate NOT verified: {exc}", None
        all_windows = nm.merge_windows(nm.kernel_windows(conn, None),
                                       nm.sample_period_ns(sample_hz))
        chosen = nm.longest_window(all_windows)
        traffic = nm.dram_traffic(
            conn, sample_hz=sample_hz,
            peak_bytes_per_s=(peak_gbps * 1e9) if peak_gbps else None,
            allow_unresolved=True, windows=chosen)
        comparison = nm.compare_to_model(traffic, modelled_per_launch, launches)
        mismatch = ""
        if launches is not None and traffic.launches and \
                abs(launches - traffic.launches) > max(1, 0.01 * launches):
            mismatch = (f"LAUNCH COUNT DISAGREES: the workload reported "
                        f"{launches} launches and the trace's chosen window "
                        f"holds {traffic.launches}. The window selection has "
                        "picked the wrong kernels and this ratio is void.")
        return {
            "launches_from_workload": launches,
            "launches_from_trace": traffic.launches,
            "mismatch": mismatch,
            "traffic": traffic.text(),
            "comparison": comparison.text(),
            "ratio": comparison.ratio,
            "observed_sample_hz": rate_note,
            "sample_rate_honoured": rate_ok,
            # A resolution verdict is only as good as the rate it was computed
            # from, so an unverified or clamped rate voids it rather than
            # narrowing it. False and None are different states and both are
            # kept: False means the tool clamped, None means we could not tell.
            "resolution_ok": bool(traffic.resolution.ok) and rate_ok is True,
            "resolution_ok_before_rate_check": traffic.resolution.ok,
            "edge_error": traffic.resolution.edge_error,
            "alpha_band_two_tiles": traffic.resolution.alpha_band(2),
            "route": traffic.route,
            "windows_in_report": len(all_windows),
            "measured_window_ms": chosen[0].duration_ns / 1e6 if chosen else 0.0,
            "calibration_verdict": nm.calibration_verdict(comparison),
        }
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--explain", action="store_true",
                    help="print the sampling arithmetic and stop. No GPU needed.")
    ap.add_argument("--calibrate", action="store_true",
                    help="after the ladder, measure a workload whose DRAM traffic "
                         "is KNOWN and check the sampler against it")
    ap.add_argument("--measure", action="store_true",
                    help="calibrate, then measure the MoE up-stage cell and score "
                         "it against the compulsory byte model")
    ap.add_argument("--report", type=Path,
                    help="skip the pod entirely and parse an exported .sqlite")
    ap.add_argument("--nsys", help="path to nsys, if it is not on PATH")
    ap.add_argument("--sample-hz", type=float, default=nm.MAX_SAMPLE_HZ,
                    help=f"GPU metrics sample rate (default the "
                         f"{nm.MAX_SAMPLE_HZ / 1e3:.0f} kHz ceiling; nsys itself "
                         f"defaults to {nm.DEFAULT_SAMPLE_HZ / 1e3:.0f} kHz)")
    ap.add_argument("--seconds", type=float, default=PROBE_SECONDS,
                    help="length of the child's timed loop")
    ap.add_argument("--buffer-mb", type=float, default=2048.0,
                    help="stream workload buffer, far larger than L2 on purpose")
    ap.add_argument("--model", default="deepseek-v3", choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--peak-gbps", type=float,
                    help="measured DRAM ceiling, needed only when the metric is a "
                         "percentage of peak. Use the measured_<device>.yaml "
                         "figure, never a datasheet one, and note that the byte "
                         "result then carries that calibration's uncertainty")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-attempt wall clock limit")
    ap.add_argument("--out", type=Path, help="where the report and traces land")
    ap.add_argument("--workload", choices=("stream", "grouped-mm"),
                    help="INTERNAL: run as the profiled child, not as the driver")
    args = ap.parse_args()

    if args.workload:
        if args.workload == "stream":
            info = workload_stream(args.buffer_mb, args.seconds)
        else:
            info = workload_grouped_mm(args.model, args.tokens, args.seconds)
        print(json.dumps(info))
        return 0

    if args.explain:
        print(explain())
        return 0

    out_dir = args.out or (results_root() / "nsys_dram_probe" /
                           time.strftime("%Y-%m-%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report(started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.perf_counter()
    print(f"WRITES TO   {out_dir}")
    print(explain())
    rep.single_launch = nm.single_launch_verdict().text()

    if args.report:
        modelled = modelled_up_stage_bytes(args.model, args.tokens)
        print(f"\nparsing {args.report} against the compulsory model for "
              f"{args.model} T={args.tokens}")
        for name, value in modelled["terms"].items():
            print(f"  {name:34s} {value / 1e9:8.3f} GB")
        try:
            got = measure_from_report(args.report, args.sample_hz, args.peak_gbps,
                                      modelled["total"])
        except nm.NsysProbeRefused as exc:
            print(f"\nREFUSED: {exc}")
            return 3
        print("\n" + got["traffic"] + "\n\n" + got["comparison"])
        if got["mismatch"]:
            print("\n" + got["mismatch"])
        return 0

    try:
        binary = nsys_binary(args.nsys)
        d = discover(binary)
    except nm.NsysProbeRefused as exc:
        print(f"\nREFUSED: {exc}")
        return 3

    rep.discovery = asdict(d)
    print("\nWHAT IS INSTALLED")
    print(f"  nsys        {d.binary}")
    print(f"  version     {d.version}   ({d.version_raw})")
    print(f"  device      {d.device_name}")
    print(f"  flags       device={d.device_flag} set={d.set_flag} "
          f"frequency={d.frequency_flag}")
    print(f"  offered     {', '.join(d.offered)}")
    print(f"  metric sets {len(d.metric_sets)} listed, chosen "
          f"{d.chosen_set or '(letting nsys decide)'}")
    print(f"  neighbours  {d.compute_apps}")

    print(f"\nTHE LADDER, at {args.sample_hz / 1e3:.0f} kHz, "
          f"{args.seconds:.1f} s of stream per attempt")
    attempts = run_ladder(d, out_dir, child_argv(args, "stream"), args.sample_hz,
                          args.timeout)
    rep.attempts = [asdict(a) for a in attempts]
    winner = None
    for a in attempts:
        state = "WORKED" if (a.metrics_rows and a.dram_metrics) else "no"
        print(f"\n  [{state}] {a.label}")
        print(f"    argv     {' '.join(a.argv)}")
        print(f"    exit     {a.returncode} in {a.seconds:.1f}s ({a.cause})")
        print(f"    verdict  {a.verdict}")
        if a.dram_metrics:
            print(f"    dram     {', '.join(a.dram_metrics)}")
        if a.returncode != 0:
            print("    stderr   " + a.stderr_head.replace("\n", "\n             "))
        if a.metrics_rows and a.dram_metrics:
            winner = a

    rep.compute_apps_after = compute_apps()
    if winner is None:
        rep.verdict = (
            "NO. No invocation of this nsys sampled a DRAM metric on this "
            "device. The open path in docs/FINDINGS.md is now a closed one for "
            "this pod, and DRAM traffic stays modelled rather than counted. "
            "The control attempt above says whether nsys itself works here, "
            "which separates 'no sampler' from 'no nsys'.")
        print(f"\nVERDICT: {rep.verdict}")
        rep.elapsed_s = time.perf_counter() - t0
        (out_dir / "probe.json").write_text(json.dumps(asdict(rep), indent=2))
        print(f"\nelapsed {rep.elapsed_s:.0f}s, report {out_dir / 'probe.json'}")
        return 3

    print(f"\nVERDICT SO FAR: nsys DOES sample DRAM here, via {winner.label}")

    if args.calibrate or args.measure:
        print("\nCALIBRATION: a workload whose DRAM traffic is known without a "
              "model")
        target = out_dir / "calibrate"
        res = shell(rerun_with(winner.argv, target, child_argv(args, "stream")),
                    timeout=args.timeout * 2)
        info = child_json(res)
        sqlite_path, _ = export_sqlite(d.binary, Path(str(target) + ".nsys-rep"))
        if sqlite_path and info:
            known = (info["known_read_bytes_per_launch"]
                     + info["known_write_bytes_per_launch"])
            got = measure_from_report(sqlite_path, args.sample_hz, args.peak_gbps,
                                      known, info["iters"])
            rep.calibration = {**info, **got}
            print(f"  known     {known * info['iters'] / 1e9:.3f} GB over "
                  f"{info['iters']} launches ({info['achieved_gbps']:.0f} GB/s "
                  "by the clock)")
            print("  RATE: " + got["observed_sample_hz"])
            print("  " + got["traffic"].replace("\n", "\n  "))
            print("  " + got["comparison"].replace("\n", "\n  "))
            if got["mismatch"]:
                print("  " + got["mismatch"])
            if got["sample_rate_honoured"] is not True:
                print("  RESOLUTION VOID: the delivered rate was not confirmed "
                      "to match the requested one, so the edge term and the "
                      "alpha band below were computed from the wrong period. "
                      "The traffic TOTAL still stands; its confidence does not.")
            print("  " + got["calibration_verdict"])
            if got["calibration_verdict"].startswith("FAIL"):
                rep.verdict = ("nsys sampled DRAM, and the sampler FAILED a case "
                               "with a known answer, so nothing from this session "
                               "should be quoted.")
                print(f"\nVERDICT: {rep.verdict}")
                rep.elapsed_s = time.perf_counter() - t0
                (out_dir / "probe.json").write_text(json.dumps(asdict(rep), indent=2))
                return 3
        else:
            rep.calibration = {"error": res.head()}
            print(f"  FAILED: {res.head()}")

    if args.measure:
        print(f"\nTHE CELL: {args.model} up-stage GEMM at T={args.tokens}, "
              "against the compulsory byte model")
        modelled = modelled_up_stage_bytes(args.model, args.tokens)
        for name, value in modelled["terms"].items():
            print(f"  {name:34s} {value / 1e9:8.3f} GB")
        target = out_dir / "cell"
        res = shell(rerun_with(winner.argv, target, child_argv(args, "grouped-mm")),
                    timeout=args.timeout * 4)
        info = child_json(res)
        sqlite_path, _ = export_sqlite(d.binary, Path(str(target) + ".nsys-rep"))
        if sqlite_path and info:
            got = measure_from_report(sqlite_path, args.sample_hz, args.peak_gbps,
                                      modelled["total"], info["iters"])
            rep.cell = {**info, **got}
            print("  RATE: " + got["observed_sample_hz"])
            if got["sample_rate_honoured"] is not True:
                print("  RESOLUTION VOID: alpha band below was computed from "
                      "the REQUESTED rate, which was not confirmed delivered.")
            print("  " + got["traffic"].replace("\n", "\n  "))
            print("  " + got["comparison"].replace("\n", "\n  "))
            if got["mismatch"]:
                print("  " + got["mismatch"])
            print(f"  at this edge term alpha lands at "
                  f"+/-{got['alpha_band_two_tiles']:.3f} on a two-tile cell, "
                  f"against the {nm.MEASURE_ALPHA_REL * 100:.1f}% traffic "
                  "precision +/-0.05 needs. That comparison, not the ratio, is "
                  "what says whether this route can carry the refit.")
        else:
            rep.cell = {"error": res.head()}
            print(f"  FAILED: {res.head()}")

    rep.verdict = ("nsys samples DRAM on this pod. Every number it produces is "
                   "device wide and edge quantised, so quote it with the "
                   "resolution line beside it.")
    rep.elapsed_s = time.perf_counter() - t0
    (out_dir / "probe.json").write_text(json.dumps(asdict(rep), indent=2))
    print(f"\nelapsed {rep.elapsed_s:.0f}s, report {out_dir / 'probe.json'}")
    print("Traces and their sqlite exports are beside it. `*.nsys-rep` is "
          "gitignored at any depth, so they leave as a tarball.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
