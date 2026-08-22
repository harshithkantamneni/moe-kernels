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
from ..reference.torch_ref import expert_counts, golden_forward, make_inputs
from ..routing.imbalance import counts_from_offsets, expert_load
from ..spec import BenchSpec
from ..stages import BASE_ENV
from ..state import MoEState
from . import bytes_model as BM
from . import schema as SC
from . import timing as T
from .roofline import Hardware
from .tolerance import relative_error, tolerance

# Returns forced top-k expert ids for a cell, or None to let the router decide.
RoutingSource = Callable[[BenchSpec], "torch.Tensor | None"]

#: `impl` value meaning "time the entire tiling end to end" rather than one span.
PIPELINE_SCOPE = "__pipeline__"

def should_time_graph(cost, cfg: RunConfig) -> tuple[bool, str]:
    """Is isolating launch overhead worth a doubled sweep for this cell?

    Predict the roofline-minimum time from compulsory traffic. If a kernel
    launch is a smaller fraction of that than `graph_min_launch_share`, the
    graph/eager delta cannot be a first-order effect and the cell is not worth
    measuring twice. Errs toward measuring: an unknown bandwidth means yes.
    """
    if cfg.graph_min_launch_share <= 0:
        return True, ""
    bw = cfg.hardware.bandwidth_bytes_s if cfg.hardware else None
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
    #: Reuse expert weights across cells that share (model, dtype, seed).
    #: Values are bit-identical either way; the contract is that no
    #: implementation writes to `weights`. Set False to opt out.
    reuse_weights: bool = True
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
    #: Measured ceilings from scripts/calibrate_hardware.py, as a Hardware.
    #: Without it the efficiency columns stay zero rather than being quoted
    #: against a spec peak this machine may never reach. It already knows its
    #: own ridge point and bound classification, so the driver does not.
    hardware: Hardware | None = None
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


def compare(got, golden, tol) -> CorrectnessResult:
    max_abs = float((got.float() - golden.float()).abs().max())
    rel = relative_error(got, golden)
    return CorrectnessResult(tol.passes(rel), max_abs, rel, tol.rel_max,
                             tol.calibrated)


@torch.no_grad()
def check_correctness(spec: BenchSpec, pipe: Pipeline, x, weights, forced,
                      cfg: RunConfig):
    """Run the tiling once and compare against golden fp32.

    Returns the populated state and the golden output as well. The state becomes
    the prologue for span-scoped timing, so the span under study is timed
    against real inputs produced by the real upstream stages; the golden output
    is reused to re-validate a CUDA-graph replay, and the tolerance is returned
    so callers do not rebuild it.
    """
    tol = tolerance(spec, cfg.calibration)
    st = MoEState(spec=spec, weights=weights, x=x)
    st.forced_topk_ids = forced
    pipe.run(st, validate_shapes=cfg.validate_shapes)
    golden = golden_forward(spec, weights, x, forced_topk_ids=forced)
    return compare(st.y, golden, tol), st, golden, tol


def _downstream_of(pipe: Pipeline, span):
    """Spans that run after `span`, needed to turn its output back into a layer
    output that can be compared against golden."""
    if span is None:
        return []
    return list(pipe.spans[pipe.spans.index(span) + 1:])


def _expert_counts(st: MoEState, spec: BenchSpec, forced) -> tuple[list[int], str]:
    """Per-expert row counts, from the most independent source available.

    Precedence matters and is not obvious. The forced routing decision is the
    experimental INPUT; `topk_ids` and `expert_offsets` in state are OUTPUTS of
    whatever implementation is under test. Reading state first meant a kernel
    covering `router` or `permute` derived its own load metrics from its own
    bug, and those metrics feed active_experts -> weight bytes ->
    compulsory_bytes -> arithmetic intensity. A benchmark axis must never be
    computed from the thing being measured while the ground truth is in hand.

    A span covering all six stages (the vLLM/SGLang fused_moe shape) never
    materialises `expert_offsets`, so no source is guaranteed and the chain has
    to end in something. It ends loudly: zeros are reported with a source of
    "unknown" so the row says the load is not known rather than saying it is 0.
    """
    E = spec.model.num_experts
    if forced is not None:
        return expert_counts(forced.cpu(), E).tolist(), "forced routing"
    if st.topk_ids is not None:
        return expert_counts(st.topk_ids.cpu(), E).tolist(), "topk_ids"
    if st.expert_offsets is not None:
        return counts_from_offsets(st.expert_offsets.cpu()), "expert_offsets"
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


def _apply_cost(row: SC.Row, cost, ms: float | None,
                cfg: RunConfig | None = None) -> None:
    row.flops = cost.flops
    row.compulsory_bytes = cost.bytes_total
    row.arith_intensity_compulsory = cost.arithmetic_intensity
    if not ms:
        return
    row.tflops = cost.tflops(ms)
    row.compulsory_gbps = cost.gbps(ms)
    if cfg is None:
        return

    hw = cfg.hardware
    if hw is None:
        return
    row.achieved_bw_gbps = hw.bandwidth_bytes_s / 1e9
    try:
        peak = hw.peak(row.dtype)
    except ValueError:
        peak = 0.0
    if peak:
        row.achieved_peak_tflops = peak / 1e12
        row.pct_of_achieved_tflops = 100.0 * row.tflops / row.achieved_peak_tflops

    # Only sound when the cell is genuinely memory bound. Compulsory intensity
    # is an UPPER bound on true intensity, so compulsory < ridge implies true <
    # ridge and the classification is conservative. Hardware owns that call.
    if peak and hw.bound(row.dtype, cost.arithmetic_intensity) == "memory":
        from .calibrate import implied_traffic_ratio
        row.implied_traffic_ratio = implied_traffic_ratio(
            cost.bytes_total, ms, hw.bandwidth_bytes_s)


#: Timing and derived columns, zeroed whenever a row did not earn them.
_TIMED_FIELDS = ("ms_p50", "ms_p90", "ms_min", "ms_std", "jitter_p90_over_p50",
                 "tflops", "compulsory_gbps", "pct_of_achieved_tflops",
                 "implied_traffic_ratio")


def _apply_load(row: SC.Row, load) -> None:
    for name, value in load.as_row().items():
        setattr(row, name, value)


def _apply_meta(row: SC.Row, meta: dict) -> None:
    for name, value in meta.items():
        if name in SC.COLUMNS:
            setattr(row, name, value)


def _emit(writer: SC.CsvWriter, manifest: SC.Manifest, row: SC.Row, key: str,
          status: str = SC.STATUS_OK, detail: str = "") -> int:
    """The single place a row reaches the CSV.

    Enforces the harness's headline invariant structurally rather than by
    convention: a row that did not pass the oracle leaves with its timing and
    derived columns zeroed, so `correctness_passed == False` implies
    `ms_p50 == 0` in the file itself. Consumer-side filters then become
    redundancy instead of the only defence.
    """
    if not row.correctness_passed:
        for name in _TIMED_FIELDS:
            setattr(row, name, 0.0)
    writer.write(row)
    manifest.record(key, status, detail)
    return 1


@torch.no_grad()
def run_cell(spec: BenchSpec, pipeline_names: Sequence[str], impl: str,
             cfg: RunConfig, routing: RoutingSource,
             writer: SC.CsvWriter, manifest: SC.Manifest,
             info: dict, sha: str, dirty: bool) -> int:
    """Benchmark one (cell, tiling, target) across the configured timing modes."""
    pipe = build(pipeline_names, spec=spec)
    span = _resolve_target(pipe, impl)
    if pipe.env not in (cfg.env_name, BASE_ENV):
        raise PipelineError(
            f"pipeline needs environment {pipe.env!r} but this process is "
            f"{cfg.env_name!r}; rows would claim a framework that did not run")

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
    x, weights = make_inputs(spec, device=cfg.device, scale=cfg.input_scale,
                             reuse_weights=cfg.reuse_weights)
    forced = routing(spec)
    if forced is not None:
        forced = forced.to(x.device)

    correctness, st, golden, tol = check_correctness(spec, pipe, x, weights,
                                                     forced, cfg)
    routing_meta = cfg.routing_info(spec) if cfg.routing_info else {}
    counts, counts_source = _expert_counts(st, spec, forced)
    load = expert_load(counts)

    # Cost is scoped to what the timer wraps, and materialisation is taken from
    # THIS tiling rather than from the span in isolation.
    if span is None:
        costed_spans, materialised = list(pipe.spans), list(pipe.materialised)
    else:
        costed_spans, materialised = [span], [pipe.materialised_for(span)]
    cost = BM.pipeline_cost(costed_spans, spec, load.active_experts, materialised)

    def prepare(row: SC.Row, verdict=None) -> SC.Row:
        """Everything every row carries, regardless of which path emitted it."""
        _apply_correctness(row, verdict or correctness)
        _apply_load(row, load)
        _apply_meta(row, routing_meta)
        _apply_cost(row, cost, None, cfg)
        if counts_source != "forced routing":
            row.notes = f"expert load derived from {counts_source}"
        return row

    if not correctness.passed:
        # One row, not four: none of the timing modes ran. capture_status says
        # so explicitly, since the l2_flush/cuda_graph columns would otherwise
        # read as a measured mode that happened to produce no numbers.
        row = prepare(_base_row(spec, pipe, impl, span, cfg, info, sha, dirty))
        row.capture_status = "not_timed"
        row.notes = (f"correctness failed (rel={correctness.rel_err:.3e} > "
                     f"{correctness.tol_rel_max:.3e}); not timed")
        written += _emit(writer, manifest, row, SC.cell_key(row),
                         SC.STATUS_CORRECTNESS_FAILED,
                         f"rel={correctness.rel_err:.3e}")
        # Re-running would reproduce this verdict exactly, so every mode key
        # for this cell is terminal.
        for _, _, _, key in keyed:
            manifest.record(key, SC.STATUS_CORRECTNESS_FAILED,
                            f"rel={correctness.rel_err:.3e}")
        return written

    downstream = _downstream_of(pipe, span)

    if span is None:
        # Built once, outside the timed region: allocating a MoEState per
        # iteration would measure python bookkeeping that is not under study.
        timed_state = MoEState(spec=spec, weights=weights, x=x)
        timed_state.forced_topk_ids = forced

        def call():
            pipe.run(timed_state, validate_shapes=False)
    else:
        # A span-scoped call mutates the prologue state in place.
        timed_state = st

        def call():
            span(st)

    # Re-earn the correctness verdict against the graph's own replayed output.
    # Graph replay reuses fixed, graph-private buffers, so a kernel that leaves
    # a tail tile or an empty-expert group unwritten sees the previous replay's
    # correct values still resident and would otherwise look fine.
    replay_verdict = None

    def verify_replay():
        """Compare the output the REPLAY produced, not the prologue's.

        `timed_state` is the state the timed callable actually writes into, and
        for a whole-layer span that is not the state check_correctness left
        behind. Comparing the wrong one made this a no-op for exactly the
        implementation shape (vLLM/SGLang fused_moe) it most needed to check.
        """
        nonlocal replay_verdict
        if span is not None:
            for later in downstream:
                later(timed_state)
        replay_verdict = (compare(timed_state.y, golden, tol)
                          if timed_state.y is not None else correctness)

    graph_ok, graph_skip_reason = should_time_graph(cost, cfg)

    for use_graph, l2, row, key in keyed:
        if key in manifest:
            continue
        prepare(row)

        if use_graph and not graph_ok:
            row.capture_status = "skipped"
            row.graph_skip_reason = graph_skip_reason
            row.notes = "graph timing skipped by cost policy"
            written += _emit(writer, manifest, row, key, SC.STATUS_OK,
                             "graph skipped by policy")
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
            row.capture_status = "not_capturable"
            row.notes = f"not CUDA-graph capturable: {str(e)[:160]}"
            written += _emit(writer, manifest, row, key,
                             SC.STATUS_NOT_CAPTURABLE, str(e)[:200])
            continue
        except RuntimeError as e:
            row.notes = f"timing error: {str(e)[:160]}"
            written += _emit(writer, manifest, row, key, SC.STATUS_ERROR,
                             str(e)[:200])
            continue
        clocks_end = cfg.clock_sampler()
        drift, throttled = T.clock_drift(clocks_start, clocks_end)

        verdict = replay_verdict if (use_graph and replay_verdict) else correctness
        _apply_correctness(row, verdict)
        row.capture_status = "captured" if use_graph else "n/a"
        row.warmup, row.iters, row.trials = res.warmup, res.iters, res.trials
        row.flush_mb, row.flush_mode = res.flush_mb, res.flush_mode
        row.ms_p50, row.ms_p90 = res.ms_p50, res.ms_p90
        row.ms_min, row.ms_std = res.ms_min, res.ms_std
        row.jitter_p90_over_p50 = res.jitter_p90_over_p50
        _apply_cost(row, cost, res.ms_p50, cfg)
        row.sm_clock_start_mhz = clocks_start.sm_clock_mhz
        row.sm_clock_end_mhz = clocks_end.sm_clock_mhz
        row.temp_start_c, row.temp_end_c = clocks_start.temp_c, clocks_end.temp_c
        row.clock_drift_pct, row.throttled = drift, throttled

        if not verdict.passed:
            # _emit zeroes the timing columns, so the file never carries a
            # measurement that failed the oracle.
            row.notes = (f"REPLAYED output failed the oracle "
                         f"(rel={verdict.rel_err:.3e}); timing discarded")
            written += _emit(writer, manifest, row, key,
                             SC.STATUS_CORRECTNESS_FAILED, "graph replay")
            continue

        written += _emit(writer, manifest, row, key)

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
                                    SC.STATUS_INVALID_PIPELINE, str(e)[:200])
                    print(f"[warn] {spec.label} {impl}: {e}")
                except Exception as e:  # noqa: BLE001
                    manifest.record(f"crash|{spec.label}|{'+'.join(names)}|{impl}",
                                    SC.STATUS_CRASH, traceback.format_exc()[-400:])
                    print(f"[warn] {spec.label} {impl}: {e}")
        finally:
            manifest.close()
    print(f"[driver] wrote {total} rows -> {cfg.csv_path}")
    return cfg.csv_path
