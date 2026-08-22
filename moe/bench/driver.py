"""Walks the benchmark matrix, gates on correctness, writes rows, and resumes.

Invariants this module enforces:
  - No timing number is written for an implementation that did not pass the
    golden-fp32 oracle on that exact cell in that same run.
  - The timer wraps the span under study, not the whole layer. Timing a full
    pipeline that contains python-loop reference stages would measure the
    reference, not the kernel.
  - Every row is flushed and fsynced before the next cell starts, so a killed
    pod loses at most one cell.
  - A completed unit of work is recorded in a manifest and skipped on re-run,
    so an interrupted sweep resumes instead of restarting.

The timing backend is injectable so the whole control flow above can be tested
on a laptop, before any of it runs on a metered GPU.
"""
from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..pipeline import Pipeline, PipelineError, build
from ..reference.torch_ref import golden_forward, make_inputs
from ..routing.imbalance import counts_from_offsets, expert_load
from ..spec import BenchSpec
from ..state import MoEState
from . import bytes_model as BM
from . import schema as SC
from . import timing as T
from .tolerance import tolerance

# Returns forced top-k expert ids for a cell, or None to let the router decide.
RoutingSource = Callable[[BenchSpec], "torch.Tensor | None"]

#: `impl` value meaning "time the entire tiling end to end" rather than one span.
PIPELINE_SCOPE = "__pipeline__"


@dataclass
class RunConfig:
    out_dir: Path = Path("results")
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    env_name: str = "base"
    warmup: int = 25
    trials: int = 3
    iters: int | None = None          # None: derive from FLOPs
    flush_mb: int = T.DEFAULT_FLUSH_MB
    l2_modes: tuple[bool, ...] = (True, False)
    graph_modes: tuple[bool, ...] = (False, True)
    calibration: dict | None = None
    validate_shapes: bool = True
    device: str = "cuda"

    # Injectable backends. Overridden in tests so the driver's control flow can
    # be verified without CUDA.
    timer_eager: Callable = T.time_eager
    timer_graph: Callable = T.time_graph
    clock_sampler: Callable = T.ClockState.sample

    @property
    def csv_path(self) -> Path:
        return self.out_dir / f"run_{self.run_id}_{self.env_name}.csv"

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / f"run_{self.run_id}_{self.env_name}.manifest.jsonl"


@dataclass
class CorrectnessResult:
    passed: bool
    max_abs_err: float
    max_rel_err: float
    atol: float
    rtol: float
    detail: str = ""


@torch.no_grad()
def check_correctness(spec: BenchSpec, pipe: Pipeline, x, weights, forced,
                      cfg: RunConfig) -> tuple[CorrectnessResult, MoEState]:
    """Run the tiling once and compare against golden fp32.

    Returns the populated state as well, which becomes the prologue for
    span-scoped timing: the span under study is then timed against real inputs
    produced by the real upstream stages.
    """
    tol = tolerance(spec, cfg.calibration)
    st = MoEState(spec=spec, weights=weights, x=x)
    st.forced_topk_ids = forced
    pipe.run(st, validate_shapes=cfg.validate_shapes)

    golden = golden_forward(spec, weights, x, forced_topk_ids=forced)
    got, ref = st.y.float(), golden.float()
    diff = (got - ref).abs()
    max_abs = float(diff.max())
    max_rel = float((diff / ref.abs().clamp_min(1e-6)).max())
    passed = bool(torch.allclose(got, ref, atol=tol.atol, rtol=tol.rtol))
    return CorrectnessResult(passed, max_abs, max_rel, tol.atol, tol.rtol), st


def _resolve_target(pipe: Pipeline, impl: str):
    if impl == PIPELINE_SCOPE:
        return None
    for s in pipe.spans:
        if s.name == impl:
            return s
    raise PipelineError(
        f"impl {impl!r} is not part of pipeline {pipe.label}; "
        f"available: {[s.name for s in pipe.spans]}"
    )


def _base_row(spec: BenchSpec, pipe: Pipeline, impl: str, span, cfg: RunConfig,
              info: dict, sha: str, dirty: bool) -> SC.Row:
    m = spec.model
    return SC.Row(
        run_id=cfg.run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_sha=sha,
        git_dirty=dirty,
        env_name=cfg.env_name,
        model=m.name,
        hidden_size=m.hidden_size,
        intermediate_size=m.intermediate_size,
        num_experts=m.num_experts,
        top_k=m.top_k,
        num_tokens=spec.num_tokens,
        rows=spec.rows,
        dtype=spec.dtype,
        routing_kind=spec.routing.kind,
        routing_param=spec.routing.param,
        trace_id=spec.routing.trace_id or "",
        seed=spec.seed,
        pipeline=pipe.label,
        impl=impl,
        scope="pipeline" if span is None else "span",
        covers="+".join(span.covers) if span is not None else "all",
        cuda_graph_safe=(pipe.cuda_graph_safe if span is None
                         else span.cuda_graph_safe),
        **{k: v for k, v in info.items() if k in SC.COLUMNS},
    )


def _apply_correctness(row: SC.Row, c: CorrectnessResult) -> None:
    row.correctness_passed = c.passed
    row.max_abs_err = c.max_abs_err
    row.max_rel_err = c.max_rel_err
    row.atol, row.rtol = c.atol, c.rtol


def _apply_cost(row: SC.Row, cost, ms: float | None) -> None:
    row.flops = cost.flops
    row.bytes_total = cost.bytes_total
    row.arithmetic_intensity = cost.arithmetic_intensity
    if ms:
        row.tflops = cost.tflops(ms)
        row.gbps = cost.gbps(ms)


@torch.no_grad()
def run_cell(spec: BenchSpec, pipeline_names: Sequence[str], impl: str,
             cfg: RunConfig, routing: RoutingSource,
             writer: SC.CsvWriter, manifest: SC.Manifest,
             info: dict, sha: str, dirty: bool) -> int:
    """Benchmark one (cell, tiling, target) across the configured timing modes."""
    pipe = build(pipeline_names, spec=spec)
    span = _resolve_target(pipe, impl)
    written = 0

    x, weights = make_inputs(spec, device=cfg.device)
    forced = routing(spec)
    if forced is not None:
        forced = forced.to(x.device)

    correctness, st = check_correctness(spec, pipe, x, weights, forced, cfg)

    counts = counts_from_offsets(st.expert_offsets.cpu())
    load = expert_load(counts)

    # Cost is scoped to what the timer wraps, so tflops and gbps describe the
    # same work the milliseconds describe.
    costed_spans = pipe.spans if span is None else [span]
    cost = BM.pipeline_cost(costed_spans, spec, active_experts=load.active_experts)
    iters = cfg.iters or T.iters_for(cost.flops)

    if not correctness.passed:
        row = _base_row(spec, pipe, impl, span, cfg, info, sha, dirty)
        _apply_correctness(row, correctness)
        _apply_cost(row, cost, None)
        for k, v in load.as_row().items():
            setattr(row, k, v)
        row.notes = "correctness failed; not timed"
        writer.write(row)
        manifest.record(SC.cell_key(row), "correctness_failed",
                        f"abs={correctness.max_abs_err:.3e}")
        return 1

    if span is None:
        def call():
            s = MoEState(spec=spec, weights=weights, x=x)
            s.forced_topk_ids = forced
            pipe.run(s, validate_shapes=False)
    else:
        def call():
            span(st)

    for use_graph in cfg.graph_modes:
        for l2 in cfg.l2_modes:
            row = _base_row(spec, pipe, impl, span, cfg, info, sha, dirty)
            row.l2_flush, row.cuda_graph = l2, use_graph
            key = SC.cell_key(row)
            if key in manifest:
                continue

            clocks_start = cfg.clock_sampler()
            try:
                fn = cfg.timer_graph if use_graph else cfg.timer_eager
                res = fn(call, warmup=cfg.warmup, iters=iters, trials=cfg.trials,
                         l2_flush=l2, flush_mb=cfg.flush_mb)
            except T.NotCapturable as e:
                # Not a failure: an implementation that cannot be graph-captured
                # cannot be used in real MoE inference, and that is a finding.
                manifest.record(key, "not_capturable", str(e)[:200])
                continue
            except RuntimeError as e:
                manifest.record(key, "error", str(e)[:200])
                _apply_correctness(row, correctness)
                _apply_cost(row, cost, None)
                row.notes = f"timing error: {str(e)[:180]}"
                writer.write(row)
                written += 1
                continue
            clocks_end = cfg.clock_sampler()
            drift, throttled = T.clock_drift(clocks_start, clocks_end)

            _apply_correctness(row, correctness)
            for k, v in load.as_row().items():
                setattr(row, k, v)
            row.warmup, row.iters, row.trials = res.warmup, res.iters, res.trials
            row.ms_p50, row.ms_p90 = res.ms_p50, res.ms_p90
            row.ms_min, row.ms_std = res.ms_min, res.ms_std
            row.jitter_p90_over_p50 = res.jitter_p90_over_p50
            _apply_cost(row, cost, res.ms_p50)
            row.sm_clock_start_mhz = clocks_start.sm_clock_mhz
            row.sm_clock_end_mhz = clocks_end.sm_clock_mhz
            row.temp_start_c, row.temp_end_c = clocks_start.temp_c, clocks_end.temp_c
            row.clock_drift_pct, row.throttled = drift, throttled

            writer.write(row)
            manifest.record(key)
            written += 1

    return written


def run_sweep(cells: Iterable[tuple[BenchSpec, Sequence[str], str]],
              cfg: RunConfig, routing: RoutingSource,
              info: dict | None = None) -> Path:
    """cells: (spec, pipeline span names, name of the impl under study)."""
    info = info if info is not None else T.gpu_info()
    sha, dirty = SC.git_provenance()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    with SC.CsvWriter(cfg.csv_path) as writer:
        manifest = SC.Manifest(cfg.manifest_path)
        try:
            for spec, names, impl in cells:
                try:
                    total += run_cell(spec, names, impl, cfg, routing, writer,
                                      manifest, info, sha, dirty)
                except PipelineError as e:
                    manifest.record(f"invalid|{spec.label}|{'+'.join(names)}|{impl}",
                                    "invalid_pipeline", str(e)[:200])
                    print(f"[warn] {spec.label} {impl}: {e}")
                except Exception as e:  # noqa: BLE001
                    manifest.record(f"crash|{spec.label}|{'+'.join(names)}|{impl}",
                                    "crash", traceback.format_exc()[-400:])
                    print(f"[warn] {spec.label} {impl}: {e}")
        finally:
            manifest.close()
    print(f"[driver] wrote {total} rows -> {cfg.csv_path}")
    return cfg.csv_path
