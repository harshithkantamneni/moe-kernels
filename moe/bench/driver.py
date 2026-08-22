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
from .tolerance import relative_error, tolerance

# Returns forced top-k expert ids for a cell, or None to let the router decide.
RoutingSource = Callable[[BenchSpec], "torch.Tensor | None"]

#: `impl` value meaning "time the entire tiling end to end" rather than one span.
PIPELINE_SCOPE = "__pipeline__"

_BANDWIDTH_CACHE: dict[str, float | None] = {}


def _bandwidth(cfg: RunConfig) -> float | None:
    if cfg.peak_bandwidth_bytes_s is not None:
        return cfg.peak_bandwidth_bytes_s
    if "bw" not in _BANDWIDTH_CACHE:
        from .roofline import peak_bandwidth
        _BANDWIDTH_CACHE["bw"] = peak_bandwidth()
    return _BANDWIDTH_CACHE["bw"]


def should_time_graph(cost, cfg: RunConfig) -> tuple[bool, str]:
    """Is isolating launch overhead worth a doubled sweep for this cell?

    Predict the roofline-minimum time from compulsory traffic. If a kernel
    launch is a smaller fraction of that than `graph_min_launch_share`, the
    graph/eager delta cannot be a first-order effect and the cell is not worth
    measuring twice. Errs toward measuring: an unknown bandwidth means yes.
    """
    if cfg.graph_min_launch_share <= 0:
        return True, ""
    bw = _bandwidth(cfg)
    if not bw:
        return True, ""
    predicted_ms = (cost.bytes_total / bw) * 1e3
    if predicted_ms <= 0:
        return True, ""
    share = cfg.launch_overhead_ms / predicted_ms
    if share >= cfg.graph_min_launch_share:
        return True, ""
    return False, (f"launch overhead is {share * 100:.2f}% of the "
                   f"{predicted_ms:.3f} ms roofline minimum, below the "
                   f"{cfg.graph_min_launch_share * 100:.1f}% threshold")


@dataclass
class RunConfig:
    out_dir: Path = Path("results")
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    env_name: str = "base"
    warmup: int = 25
    trials: int = 3
    iters: int | None = None          # None: derive from FLOPs
    flush_mb: int = T.DEFAULT_FLUSH_MB
    flush_mode: str = "read"
    input_scale: float = 1.0
    l2_modes: tuple[bool, ...] = (True, False)
    graph_modes: tuple[bool, ...] = (False, True)
    calibration: dict | None = None
    #: Optional callable returning per-cell routing provenance, e.g. the trace
    #: fingerprint and the resolved batch/layer slice.
    routing_info: Callable[[BenchSpec], dict] | None = None
    #: Skip CUDA-graph timing when predicted kernel time is so long that launch
    #: overhead cannot matter. Graph mode doubles a metered sweep; at DeepSeek
    #: geometry with a few tokens the roofline minimum is already ~0.8 ms
    #: against a ~5 us launch, so the axis would cost half the session to
    #: measure a sub-1% effect. Set to 0.0 to always time both.
    graph_min_launch_share: float = 0.01
    launch_overhead_ms: float = 0.005
    peak_bandwidth_bytes_s: float | None = None
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
    rel_err: float          # scale-free: max|got-ref| / max|ref|
    tol_rel_max: float
    calibrated: bool = False
    detail: str = ""


def compare(got, golden, tol) -> CorrectnessResult:
    max_abs = float((got.float() - golden.float()).abs().max())
    rel = relative_error(got, golden)
    return CorrectnessResult(rel <= tol.rel_max, max_abs, rel, tol.rel_max,
                             tol.calibrated)


@torch.no_grad()
def check_correctness(spec: BenchSpec, pipe: Pipeline, x, weights, forced,
                      cfg: RunConfig):
    """Run the tiling once and compare against golden fp32.

    Returns the populated state and the golden output as well. The state becomes
    the prologue for span-scoped timing, so the span under study is timed
    against real inputs produced by the real upstream stages; the golden output
    is reused to re-validate a CUDA-graph replay.
    """
    tol = tolerance(spec, cfg.calibration)
    st = MoEState(spec=spec, weights=weights, x=x)
    st.forced_topk_ids = forced
    pipe.run(st, validate_shapes=cfg.validate_shapes)
    golden = golden_forward(spec, weights, x, forced_topk_ids=forced)
    return compare(st.y, golden, tol), st, golden


def _downstream_of(pipe: Pipeline, span):
    """Spans that run after `span`, needed to turn its output back into a layer
    output that can be compared against golden."""
    if span is None:
        return []
    idx = [i for i, s in enumerate(pipe.spans) if s is span][0]
    return list(pipe.spans[idx + 1:])


def _expert_counts(st: MoEState, spec: BenchSpec, forced) -> tuple[list[int], str]:
    """Per-expert row counts, however they can be obtained.

    A span covering all six stages is a legal tiling and is exactly the shape of
    vLLM's and SGLang's fused_moe. Such a span never materialises
    `expert_offsets`, so reading them unconditionally crashed the cell, and
    run_sweep's broad handler turned that into a silent zero-row baseline.
    """
    E = spec.model.num_experts
    if st.expert_offsets is not None:
        return counts_from_offsets(st.expert_offsets.cpu()), "expert_offsets"
    ids = st.topk_ids if st.topk_ids is not None else forced
    if ids is not None:
        counts = torch.bincount(ids.reshape(-1).long().cpu(), minlength=E)
        return counts.tolist(), "topk_ids"
    return [0] * E, "unknown"


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
        input_init="fan_in",
        input_scale=cfg.input_scale,
        **{k: v for k, v in info.items() if k in SC.COLUMNS},
    )


def _apply_correctness(row: SC.Row, c: CorrectnessResult) -> None:
    row.correctness_passed = c.passed
    row.max_abs_err = c.max_abs_err
    row.rel_err = c.rel_err
    row.tol_rel_max = c.tol_rel_max
    row.tol_calibrated = c.calibrated


def _apply_cost(row: SC.Row, cost, ms: float | None) -> None:
    row.flops = cost.flops
    row.compulsory_bytes = cost.bytes_total
    row.arith_intensity_compulsory = cost.arithmetic_intensity
    if ms:
        row.tflops = cost.tflops(ms)
        row.compulsory_gbps = cost.gbps(ms)


@torch.no_grad()
def run_cell(spec: BenchSpec, pipeline_names: Sequence[str], impl: str,
             cfg: RunConfig, routing: RoutingSource,
             writer: SC.CsvWriter, manifest: SC.Manifest,
             info: dict, sha: str, dirty: bool) -> int:
    """Benchmark one (cell, tiling, target) across the configured timing modes."""
    pipe = build(pipeline_names, spec=spec)
    span = _resolve_target(pipe, impl)

    # Build every mode key up front. If they are all already done, skip the
    # cell entirely: re-running the fp32 python-loop oracle to produce zero
    # rows is the single most expensive way to resume a sweep.
    modes = [(g, l2) for g in cfg.graph_modes for l2 in cfg.l2_modes]
    keyed = []
    for use_graph, l2 in modes:
        row = _base_row(spec, pipe, impl, span, cfg, info, sha, dirty)
        row.l2_flush, row.cuda_graph = l2, use_graph
        keyed.append((use_graph, l2, row, SC.cell_key(row)))
    if all(key in manifest for _, _, _, key in keyed):
        return 0

    written = 0
    x, weights = make_inputs(spec, device=cfg.device, scale=cfg.input_scale)
    forced = routing(spec)
    if forced is not None:
        forced = forced.to(x.device)

    correctness, st, golden = check_correctness(spec, pipe, x, weights, forced, cfg)
    tol = tolerance(spec, cfg.calibration)

    routing_meta = cfg.routing_info(spec) if cfg.routing_info else {}
    counts, counts_source = _expert_counts(st, spec, forced)
    load = expert_load(counts)

    costed_spans = pipe.spans if span is None else [span]
    cost = BM.pipeline_cost(costed_spans, spec, active_experts=load.active_experts)

    if not correctness.passed:
        row = _base_row(spec, pipe, impl, span, cfg, info, sha, dirty)
        _apply_correctness(row, correctness)
        _apply_cost(row, cost, None)
        for k, v in load.as_row().items():
            setattr(row, k, v)
        row.notes = (f"correctness failed (rel={correctness.rel_err:.3e} > "
                     f"{correctness.tol_rel_max:.3e}); not timed")
        writer.write(row)
        # None of the timing modes can be run, and re-running would reproduce
        # this exact verdict, so every mode key for this cell is terminal.
        for _, _, _, key in keyed:
            manifest.record(key, "correctness_failed",
                            f"rel={correctness.rel_err:.3e}")
        return 1

    downstream = _downstream_of(pipe, span)

    if span is None:
        def call():
            s = MoEState(spec=spec, weights=weights, x=x)
            s.forced_topk_ids = forced
            pipe.run(s, validate_shapes=False)
    else:
        def call():
            span(st)

    # Re-earn the correctness verdict against the graph's own replayed output.
    # Graph replay reuses fixed, graph-private buffers, so a kernel that leaves
    # a tail tile or an empty-expert group unwritten sees the previous replay's
    # correct values still resident and would otherwise look fine.
    replay_verdict: dict = {}

    def verify_replay():
        if span is None:
            replay_verdict["result"] = compare(st.y, golden, tol) if st.y is not None \
                else correctness
            return
        for later in downstream:
            later(st)
        replay_verdict["result"] = compare(st.y, golden, tol)

    graph_ok, graph_skip_reason = should_time_graph(cost, cfg)

    for use_graph, l2, row, key in keyed:
        if key in manifest:
            continue
        for meta_key, meta_value in routing_meta.items():
            if meta_key in SC.COLUMNS:
                setattr(row, meta_key, meta_value)

        if use_graph and not graph_ok:
            _apply_correctness(row, correctness)
            _apply_cost(row, cost, None)
            for k, v in load.as_row().items():
                setattr(row, k, v)
            row.capture_status = "skipped"
            row.graph_skip_reason = graph_skip_reason
            row.notes = "graph timing skipped by cost policy"
            writer.write(row)
            manifest.record(key, "ok", "graph skipped by policy")
            written += 1
            continue

        clocks_start = cfg.clock_sampler()
        try:
            if use_graph:
                res = cfg.timer_graph(call, warmup=cfg.warmup, iters=cfg.iters,
                                      trials=cfg.trials, l2_flush=l2,
                                      flush_mb=cfg.flush_mb,
                                      flush_mode=cfg.flush_mode,
                                      on_captured=verify_replay)
            else:
                res = cfg.timer_eager(call, warmup=cfg.warmup, iters=cfg.iters,
                                      trials=cfg.trials, l2_flush=l2,
                                      flush_mb=cfg.flush_mb,
                                      flush_mode=cfg.flush_mode)
        except T.NotCapturable as e:
            # A finding, not a failure: an implementation that cannot be
            # graph-captured cannot be used in real MoE inference. It belongs in
            # the CSV, not only in a sidecar manifest, or every aggregate over
            # the published data is silently conditioned on capturability.
            _apply_correctness(row, correctness)
            _apply_cost(row, cost, None)
            for k, v in load.as_row().items():
                setattr(row, k, v)
            row.capture_status = "not_capturable"
            row.notes = f"not CUDA-graph capturable: {str(e)[:160]}"
            writer.write(row)
            manifest.record(key, "not_capturable", str(e)[:200])
            written += 1
            continue
        except RuntimeError as e:
            manifest.record(key, "error", str(e)[:200])
            _apply_correctness(row, correctness)
            _apply_cost(row, cost, None)
            row.notes = f"timing error: {str(e)[:160]}"
            writer.write(row)
            written += 1
            continue
        clocks_end = cfg.clock_sampler()
        drift, throttled = T.clock_drift(clocks_start, clocks_end)

        verdict = replay_verdict.get("result", correctness) if use_graph else correctness
        _apply_correctness(row, verdict)
        row.capture_status = "captured" if use_graph else "n/a"
        for k, v in load.as_row().items():
            setattr(row, k, v)
        if counts_source != "expert_offsets":
            row.notes = f"expert load derived from {counts_source}"
        row.warmup, row.iters, row.trials = res.warmup, res.iters, res.trials
        row.flush_mb = getattr(res, "flush_mb", 0)
        row.flush_mode = getattr(res, "flush_mode", "")
        row.ms_p50, row.ms_p90 = res.ms_p50, res.ms_p90
        row.ms_min, row.ms_std = res.ms_min, res.ms_std
        row.jitter_p90_over_p50 = res.jitter_p90_over_p50
        _apply_cost(row, cost, res.ms_p50)
        row.sm_clock_start_mhz = clocks_start.sm_clock_mhz
        row.sm_clock_end_mhz = clocks_end.sm_clock_mhz
        row.temp_start_c, row.temp_end_c = clocks_start.temp_c, clocks_end.temp_c
        row.clock_drift_pct, row.throttled = drift, throttled

        if not verdict.passed:
            row.notes = (f"REPLAYED output failed the oracle "
                         f"(rel={verdict.rel_err:.3e}); timing is not usable")
            writer.write(row)
            manifest.record(key, "correctness_failed", "graph replay")
            written += 1
            continue

        writer.write(row)
        manifest.record(key)
        written += 1

    return written


def run_sweep(cells: Iterable[tuple[BenchSpec, Sequence[str], str]],
              cfg: RunConfig, routing: RoutingSource,
              info: dict | None = None) -> Path:
    """cells: (spec, pipeline span names, name of the impl under study)."""
    info = info if info is not None else T.runtime_info()
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
