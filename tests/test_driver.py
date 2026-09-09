"""Driver control-flow tests, run on CPU with an injected timing backend.

Everything the driver promises (correctness gates timing, the timer wraps the
span and not the layer, resume skips finished work, one bad cell does not kill
a sweep) is verified here before any of it costs GPU minutes.
"""
import importlib.util
import itertools
import pathlib
import sys
from functools import partial

import pytest
import torch

from moe import pipeline as P
from moe.bench import bytes_model as BM
from moe.bench import cli
from moe.bench import driver as D
from moe.bench import exit_codes as EC
from moe.bench import profiles as PR
from moe.bench import roofline as RF
from moe.bench import schema as SC
from moe.bench import timing as T
from moe.reference import torch_ref as R
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
from moe.stages import StageSpan, get, register
from moe.state import MoEState

REF = P.reference_pipeline_names()
CALLS = {"count": 0}


@register
class CountingUpGemm(StageSpan):
    """Correct up_gemm that records how many times it was invoked."""

    name = "t_counting_up_gemm"
    covers = ("up_gemm",)
    requires_cuda = False
    dtypes = ("fp32", "bf16")

    def __call__(self, st: MoEState) -> None:
        CALLS["count"] += 1
        st.h_up = R.grouped_gemm_loop(
            st.x_perm, st.weights.w1, st.expert_offsets,
            2 * st.spec.model.intermediate_size)


@register
class WrongUpGemm(StageSpan):
    """Right shape, wrong numbers. Must be caught by the oracle."""

    name = "t_wrong_up_gemm"
    covers = ("up_gemm",)
    requires_cuda = False
    dtypes = ("fp32", "bf16")

    def __call__(self, st: MoEState) -> None:
        st.h_up = R.grouped_gemm_loop(
            st.x_perm, st.weights.w1, st.expert_offsets,
            2 * st.spec.model.intermediate_size) * 1.5


@register
class CrashingUpGemm(StageSpan):
    name = "t_crashing_up_gemm"
    covers = ("up_gemm",)
    requires_cuda = False
    dtypes = ("fp32", "bf16")

    def __call__(self, st: MoEState) -> None:
        raise ZeroDivisionError("kernel launch went sideways")


def fake_timer(fn, *, warmup_ms=300.0, target_ms=200.0, trials=1,
               l2_flush=True, reference_clock_mhz=None, flusher=None,
               on_captured=None, level=True, drift=True, host_bound=False):
    """A `time_kernel` stand-in: same call shape, same record, no CUDA.

    Returns a `KernelTiming` because that is what the driver's default timers
    return, so the control-flow tests exercise the column mapping the sweeps
    actually use. The three verdicts are parameters rather than constants:
    every one of them has a FAIL branch in the row, and a fake that could only
    say "ok" would leave all three untested.
    """
    for _ in range(3):
        fn()
    if on_captured is not None:
        on_captured()
    return T.KernelTiming(
        ms_p50=1.0, ms_p90=1.2, ms_min=0.9, ms_std=0.05,
        iters=2, trials=trials, warmup_ms=warmup_ms, l2_flush=l2_flush,
        sm_clock_load_mhz=1980.0, sm_clock_start_mhz=1980.0,
        sm_clock_end_mhz=1975.0, clock_level_ok=level, clock_drift_ok=drift,
        samples=6, warmup_calls=17,
        flush_mb=(flusher.megabytes if flusher is not None else 0),
        clock_samples=6, clock_source="injected", clock_poll_ms=0.01,
        host_bound=host_bound, host_enqueue_ms=0.2,
        host_note=("host-bound: the queue drained" if host_bound else ""))


def fake_graph_timer(fn, **kw):
    return fake_timer(fn, **kw)


def not_capturable(fn, **kw):
    raise T.NotCapturable("host sync during capture")


def legacy_timer(fn, warmup=1, iters=2, trials=1, l2_flush=True, flush_mb=8,
                 flush_mode="read", target_ms=200.0, on_captured=None,
                 graph=False):
    """The RETIRED instrument's shape, for the seam that still accepts it."""
    for _ in range(3):
        fn()
    if on_captured is not None:
        on_captured()
    return T.TimingResult(ms_p50=1.0, ms_p90=1.2, ms_min=0.9, ms_std=0.05,
                          jitter_p90_over_p50=1.2, warmup=warmup, iters=iters or 2,
                          trials=trials, l2_flush=l2_flush, cuda_graph=graph,
                          samples=3, flush_mb=flush_mb, flush_mode=flush_mode)


FAKE_INFO = {"gpu_name": "FakeH200", "gpu_count": 1, "torch_version": "x",
             "driver_version": "y", "cuda_version": "z", "triton_version": "w"}


def cfg_for(tmp_path, **kw):
    base = dict(
        out_dir=tmp_path, device="cpu", warmup_ms=5.0, trials=1, flush_mb=8,
        l2_modes=(True,), graph_modes=(False,),
        timer=fake_timer,
        graph_timer=fake_graph_timer,
        clock_sampler=lambda: T.ClockState(1980, 45),
    )
    base.update(kw)
    return D.RunConfig(**base)


def legacy_cfg_for(tmp_path, **kw):
    """A config on the RETIRED seam, which is opt-in and stamps its rows."""
    base = dict(timer_eager=legacy_timer, timer_graph=legacy_timer,
                warmup=1, iters=2)
    base.update(kw)
    return cfg_for(tmp_path, **base)


def spec():
    return BenchSpec(MODEL_CONFIGS["toy"], num_tokens=32, dtype="fp32",
                     routing=RoutingSpec("uniform"))


def names_with(impl):
    return ["ref_router", "ref_permute", impl, "ref_act", "ref_down_gemm",
            "ref_unpermute"]


def sweep(tmp_path, impl, **kw):
    cfg = cfg_for(tmp_path, **kw)
    path = D.run_sweep([(spec(), names_with(impl), impl)], cfg,
                       routing=lambda s: None, info=FAKE_INFO)
    return cfg, path


# --------------------------------------------------------------------------


def test_correct_impl_is_timed_and_recorded(tmp_path):
    _, path = sweep(tmp_path, "t_counting_up_gemm")
    rows = SC.read_csv(path)
    assert len(rows) == 1
    r = rows[0]
    assert r["correctness_passed"] == "True"
    assert float(r["ms_p50"]) == 1.0
    assert r["impl"] == "t_counting_up_gemm"
    assert r["scope"] == "span"
    assert r["covers"] == "up_gemm"
    assert r["gpu_name"] == "FakeH200"


def test_wrong_kernel_is_never_timed(tmp_path):
    _, path = sweep(tmp_path, "t_wrong_up_gemm")
    rows = SC.read_csv(path)
    assert len(rows) == 1
    r = rows[0]
    assert r["correctness_passed"] == "False"
    assert float(r["ms_p50"]) == 0.0
    assert float(r["tflops"]) == 0.0
    assert float(r["max_abs_err"]) > 0
    assert "correctness failed" in r["notes"]


def test_timer_wraps_only_the_span_under_study(tmp_path):
    """The bug this guards: timing the whole tiling would measure the python-loop
    reference stages, not the kernel."""
    CALLS["count"] = 0
    sweep(tmp_path, "t_counting_up_gemm")
    # 1 correctness run + 3 timed invocations from the fake timer.
    assert CALLS["count"] == 4


def test_span_scoped_cost_excludes_the_rest_of_the_layer(tmp_path):
    _, path = sweep(tmp_path, "t_counting_up_gemm")
    r = SC.read_csv(path)[0]
    s = spec()
    span_cost = BM.pipeline_cost([get("t_counting_up_gemm")], s, active_experts=4)
    pipe_cost = BM.pipeline_cost([get(n) for n in REF], s, active_experts=4)
    assert float(r["flops"]) == pytest.approx(span_cost.flops)
    assert float(r["flops"]) < pipe_cost.flops


def test_pipeline_scope_costs_the_whole_layer(tmp_path):
    cfg = cfg_for(tmp_path)
    path = D.run_sweep([(spec(), REF, D.PIPELINE_SCOPE)], cfg,
                       routing=lambda s: None, info=FAKE_INFO)
    r = SC.read_csv(path)[0]
    assert r["scope"] == "pipeline"
    assert r["covers"] == "all"
    pipe_cost = BM.pipeline_cost([get(n) for n in REF], spec(), active_experts=4)
    assert float(r["flops"]) == pytest.approx(pipe_cost.flops)


def test_impl_not_in_pipeline_is_a_clear_error(tmp_path):
    cfg = cfg_for(tmp_path)
    with pytest.raises(P.PipelineError, match="is not part of pipeline"):
        D.run_cell(spec(), REF, "t_counting_up_gemm", cfg, lambda s: None,
                   SC.CsvWriter(tmp_path / "x.csv"),
                   SC.Manifest(tmp_path / "x.jsonl"), FAKE_INFO, "", False)


def test_resume_skips_completed_work(tmp_path):
    cfg = cfg_for(tmp_path)
    cells = [(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")]
    D.run_sweep(cells, cfg, routing=lambda s: None, info=FAKE_INFO)
    assert len(SC.read_csv(cfg.csv_path)) == 1

    CALLS["count"] = 0
    D.run_sweep(cells, cfg, routing=lambda s: None, info=FAKE_INFO)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 1, "resume must not duplicate a finished cell"
    assert CALLS["count"] == 0, (
        "a fully completed cell must be skipped before the fp32 oracle runs; "
        "re-running it to produce zero rows is the most expensive way to resume")


def test_timing_modes_are_separate_units_of_work(tmp_path):
    cfg = cfg_for(tmp_path, l2_modes=(True, False), graph_modes=(False, True))
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 4
    modes = {(r["l2_flush"], r["cuda_graph"]) for r in rows}
    assert modes == {("True", "False"), ("False", "False"),
                     ("True", "True"), ("False", "True")}


def test_uncapturable_impl_still_gets_a_row(tmp_path):
    """Non-capturability is a finding about the implementation, so it must reach
    the CSV. Recording it only in a sidecar manifest would silently condition
    every published aggregate on capture-friendliness."""
    cfg = cfg_for(tmp_path, graph_modes=(True,), graph_timer=not_capturable)
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 1
    assert rows[0]["capture_status"] == "not_capturable"
    assert float(rows[0]["ms_p50"]) == 0.0
    assert "not_capturable" in cfg.manifest_path.read_text()


def test_a_crashing_kernel_does_not_kill_the_sweep(tmp_path):
    cfg = cfg_for(tmp_path)
    cells = [
        (spec(), names_with("t_crashing_up_gemm"), "t_crashing_up_gemm"),
        (spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm"),
    ]
    D.run_sweep(cells, cfg, routing=lambda s: None, info=FAKE_INFO)
    rows = SC.read_csv(cfg.csv_path)
    assert [r["impl"] for r in rows] == ["t_counting_up_gemm"]
    assert "crash" in cfg.manifest_path.read_text()


def test_invalid_tiling_is_recorded_not_raised(tmp_path):
    cfg = cfg_for(tmp_path)
    bad = ["ref_router", "ref_permute", "t_counting_up_gemm", "ref_act"]
    D.run_sweep([(spec(), bad, "t_counting_up_gemm")], cfg,
                routing=lambda s: None, info=FAKE_INFO)
    assert "invalid_pipeline" in cfg.manifest_path.read_text()


def test_forced_routing_is_reflected_in_the_load_columns(tmp_path):
    s = spec()
    forced = torch.zeros((s.num_tokens, s.model.top_k), dtype=torch.int32)
    forced[:, 1] = 1
    cfg = cfg_for(tmp_path)
    D.run_sweep([(s, names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda _: forced, info=FAKE_INFO)
    r = SC.read_csv(cfg.csv_path)[0]
    assert int(r["load_active_experts"]) == 2
    assert int(r["load_empty_experts"]) == s.model.num_experts - 2
    assert float(r["load_entropy_norm"]) < 1.0


# --- which instrument wrote the row ----------------------------------------

def test_the_default_config_measures_on_the_instrument(tmp_path):
    """THE FINDING THIS FILE EXISTS FOR SINCE 2026-09-02. The driver is the
    largest cell-writing path in the repository -- every one of the 100,144
    published rows came through it -- and until v5 it timed through
    `time_eager`/`time_graph` while the roof and every ladder script had moved
    on. A default that has to be opted INTO is a default nobody sets."""
    cfg = D.RunConfig()
    assert cfg.timer is T.time_kernel
    assert cfg.graph_timer is D.time_kernel_graph
    assert cfg.timer_eager is None and cfg.timer_graph is None


def test_every_timed_row_names_the_instrument_that_produced_it(tmp_path):
    _, path = sweep(tmp_path, "t_counting_up_gemm")
    r = SC.read_csv(path)[0]
    assert r["instrument"] == T.TIMING_BASIS
    assert SC.has_kernel_timing(r)
    # The warmup is a DURATION now, and the count is what it delivered.
    assert float(r["warmup_ms"]) == 5.0
    assert int(r["warmup"]) == 17
    assert float(r["sm_clock_load_mhz"]) == 1980.0
    assert r["clock_source"] == "injected"


def test_the_three_verdicts_reach_the_row_as_words_and_not_as_silence(tmp_path):
    for level, drift, host, expected in (
            (True, True, False, ("ok", "ok", "ok")),
            (False, True, False, ("failed", "ok", "ok")),
            (True, False, True, ("ok", "failed", "failed")),
            (None, None, None, ("undetermined",) * 3)):
        out = tmp_path / f"{level}-{drift}-{host}"
        cfg = cfg_for(out, timer=partial(fake_timer, level=level, drift=drift,
                                         host_bound=host))
        D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                      "t_counting_up_gemm")], cfg,
                    routing=lambda s: None, info=FAKE_INFO)
        r = SC.read_csv(cfg.csv_path)[0]
        got = tuple(SC.timing_verdict(r, c) for c in SC.TIMING_VERDICT_COLUMNS)
        assert got == expected, (level, drift, host, got)


def test_an_instrument_row_leaves_the_retired_clock_QUANTITIES_alone(tmp_path):
    """One column, one meaning, across the version boundary.

    `time_kernel` does read a first and a last clock sample, but UNDER LOAD, and
    `sm_clock_start_mhz`/`clock_drift_pct` hold IDLE-instant readings on all
    100,144 published rows. Refilling them here would silently re-point every
    reader at a different quantity. `throttled` is a verdict rather than a
    reading and is tested below."""
    _, path = sweep(tmp_path, "t_counting_up_gemm")
    r = SC.read_csv(path)[0]
    assert int(r["sm_clock_start_mhz"]) == 0
    assert int(r["sm_clock_end_mhz"]) == 0
    assert float(r["clock_drift_pct"]) == 0.0
    assert int(r["temp_start_c"]) == 0 and int(r["temp_end_c"]) == 0


@pytest.mark.parametrize("level,drift,throttled", [
    (True, True, "False"),      # both clock checks passed
    (False, True, "False"),     # LEVEL failed and DRIFT held: KEPT since
                                # 2026-09-09, whichever side it failed on
    (True, False, "True"),      # DRIFT failed: it moved while the trials ran
    (None, None, "False"),      # undetermined is not evidence, see below
])
def test_the_throttled_verdict_is_written_and_can_fail(tmp_path, level, drift,
                                                       throttled):
    """THE GATE THAT COULD NOT FAIL, AND WHAT IT NOW FAILS ON. Four consumers
    read `throttled` as the one bool for "this row's clock misbehaved, do not
    pool it": `scripts/pod_session.sh` gate S6d, `run_all.sh`,
    `publish_results.sh` and `scripts/efficiency_report.py`. Leaving it at its
    default on every v5 row made S6d compare 0.0% against "< 5%" on every card
    at every temperature, which is "a check that examined nothing reports zero
    failures", the shape the whole instrument exists to remove.

    Since 2026-09-09 the answer is DRIFT alone. The second row is that change:
    a LEVEL failure with a steady clock is a tile sitting at its own operating
    point under the board cap, which the H200 session measured over 750 cells,
    and excluding on it removed BM=128/N=64 and BM=64/G=1 from the study
    while removing nothing anywhere else.

    The last row is the deliberate asymmetry. An undetermined clock check (no
    NVML in the container) is not evidence against the number, and marking it
    throttled would empty `efficiency_report` and fail S6d for a whole session
    over a missing library. `alpha_refit.clock_gate` keeps those rows too."""
    cfg = cfg_for(tmp_path, timer=partial(fake_timer, level=level, drift=drift))
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                  "t_counting_up_gemm")], cfg, routing=lambda s: None,
                info=FAKE_INFO)
    r = SC.read_csv(cfg.csv_path)[0]
    assert r["throttled"] == throttled
    assert SC.row_bool(r, "throttled") is (throttled == "True")


def test_host_bound_is_not_a_thermal_event(tmp_path):
    """`host_bound_ok` says the CALLER could not keep the queue deep, which
    makes `ms_*` an upper bound rather than a hot box. Folding it into
    `throttled` would report a Python launcher as a thermal failure, and the
    four consumers above would drop every T=1 eager row in the study."""
    cfg = cfg_for(tmp_path, timer=partial(fake_timer, host_bound=True))
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                  "t_counting_up_gemm")], cfg, routing=lambda s: None,
                info=FAKE_INFO)
    r = SC.read_csv(cfg.csv_path)[0]
    assert SC.timing_verdict(r, "host_bound_ok") == SC.VERDICT_FAILED
    assert r["throttled"] == "False"


def test_the_retired_seam_stamps_its_rows_as_the_retired_instrument(tmp_path):
    """The seam survives for the tests and for reproducing an old row, and it is
    only safe because a row it writes says so. Nothing off it can be pooled with
    an instrument row by accident."""
    states = itertools.cycle([T.ClockState(1980, 40), T.ClockState(1600, 84)])
    cfg = legacy_cfg_for(tmp_path, clock_sampler=lambda: next(states))
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    r = SC.read_csv(cfg.csv_path)[0]
    assert r["instrument"] == SC.LEGACY_INSTRUMENT
    assert not SC.has_kernel_timing(r)
    # The retired clock pair is exactly what that path does record.
    assert r["throttled"] == "True"
    assert float(r["clock_drift_pct"]) > 5.0
    # And it made none of the checks the instrument makes. The columns exist
    # -- the file is v5 -- and every one of them says so in words rather than
    # sitting at a value a filter would read as a pass.
    assert float(r["warmup_ms"]) == 0.0
    assert all(SC.timing_verdict(r, c) == SC.VERDICT_UNDETERMINED
               for c in SC.TIMING_VERDICT_COLUMNS)


# --- a span covering the whole layer: the vLLM/SGLang fused_moe shape --------

@register
class FullLayerRef(StageSpan):
    """One span covering all six stages, like vLLM's fused_moe. It never
    materialises expert_offsets, which used to crash the driver's load metrics
    and, via run_sweep's broad handler, silently produce zero rows for the
    entire baseline."""

    name = "t_full_layer"
    covers = ("router", "permute", "up_gemm", "act", "down_gemm", "unpermute")
    requires_cuda = False
    dtypes = ("fp32", "bf16")

    def __call__(self, st: MoEState) -> None:
        st.y = R.golden_forward(st.spec, st.weights, st.x,
                                forced_topk_ids=st.forced_topk_ids)


def test_whole_layer_span_is_a_valid_tiling():
    pipe = P.build(["t_full_layer"])
    assert len(pipe.spans) == 1
    assert pipe.spans[0].writes == {"y"}


def test_whole_layer_span_benchmarks_without_crashing(tmp_path):
    cfg = cfg_for(tmp_path)
    D.run_sweep([(spec(), ["t_full_layer"], "t_full_layer")], cfg,
                routing=lambda s: None, info=FAKE_INFO)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 1, cfg.manifest_path.read_text()
    assert rows[0]["correctness_passed"] == "True"
    assert "crash" not in cfg.manifest_path.read_text()


def test_whole_layer_span_still_reports_expert_load(tmp_path):
    """expert_offsets is unavailable, so the load must come from the forced
    routing decision instead of silently reading as all-zero."""
    s = spec()
    forced = torch.zeros((s.num_tokens, s.model.top_k), dtype=torch.int32)
    forced[:, 1] = 1
    cfg = cfg_for(tmp_path)
    D.run_sweep([(s, ["t_full_layer"], "t_full_layer")], cfg,
                routing=lambda _: forced, info=FAKE_INFO)
    r = SC.read_csv(cfg.csv_path)[0]
    assert int(r["load_active_experts"]) == 2
    assert int(r["load_total_rows"]) == s.rows
    # The forced decision is available, so no fallback note is needed.
    assert "derived from" not in r["notes"]


@register
class LyingPermute(StageSpan):
    """Produces expert_offsets that disagree with the routing decision.

    A kernel covering `permute` is exactly the kind of implementation that can
    get this wrong, and the harness must not take its word for the load.
    """

    name = "t_lying_permute"
    covers = ("permute",)
    requires_cuda = False
    dtypes = ("fp32", "bf16")

    def __call__(self, st: MoEState) -> None:
        cfg = st.spec.model
        offsets, perm = R.build_permutation(st.topk_ids, cfg.num_experts)
        st.perm_index = perm
        st.x_perm = st.x[perm.long() // cfg.top_k]
        # Wrong: claims every row went to expert 0.
        bad = torch.zeros_like(offsets)
        bad[1:] = st.spec.rows
        st.expert_offsets = bad


def test_load_columns_come_from_the_input_not_the_implementation(tmp_path):
    """Regression: the load axis must not be derived from the thing under test.

    Reading expert_offsets first meant a buggy permute kernel computed its own
    load metrics, which feed active_experts -> weight bytes -> compulsory bytes
    -> arithmetic intensity. The routing decision is the ground truth and it is
    in hand, so it wins.
    """
    s = spec()
    forced = torch.zeros((s.num_tokens, s.model.top_k), dtype=torch.int32)
    forced[:, 1] = 1                       # two experts, evenly loaded
    names = ["ref_router", "t_lying_permute", "ref_up_gemm", "ref_act",
             "ref_down_gemm", "ref_unpermute"]
    cfg = cfg_for(tmp_path)
    D.run_sweep([(s, names, "t_lying_permute")], cfg,
                routing=lambda _: forced, info=FAKE_INFO)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 1, cfg.manifest_path.read_text()
    r = rows[0]
    # The lying span claims 1 active expert; the routing decision says 2.
    assert int(r["load_active_experts"]) == 2, "load was taken from the kernel"
    assert int(r["load_max_rows"]) == s.num_tokens


def test_graph_row_revalidates_the_replayed_output(tmp_path):
    """A graph row must earn its own correctness verdict against the replayed
    output, not inherit the eager one."""
    seen = {"verified": 0}

    def counting_graph_timer(fn, on_captured=None, **kw):
        if on_captured is not None:
            seen["verified"] += 1
        return fake_graph_timer(fn, on_captured=on_captured, **kw)

    cfg = cfg_for(tmp_path, graph_modes=(True,), graph_timer=counting_graph_timer)
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    assert seen["verified"] == 1
    assert SC.read_csv(cfg.csv_path)[0]["capture_status"] == "captured"


def test_transient_errors_stay_retryable(tmp_path):
    """A CUDA OOM must not permanently blank the cell from every future run."""
    calls = {"n": 0}

    def flaky(fn, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return fake_timer(fn, **kw)

    cfg = cfg_for(tmp_path, timer=flaky)
    cells = [(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")]
    D.run_sweep(cells, cfg, routing=lambda s: None, info=FAKE_INFO)
    assert "error" in cfg.manifest_path.read_text()

    D.run_sweep(cells, cfg, routing=lambda s: None, info=FAKE_INFO)
    rows = SC.read_csv(cfg.csv_path)
    timed = [r for r in rows if float(r["ms_p50"]) > 0]
    assert len(timed) == 1, "the retried cell should have produced a timing row"


def test_correctness_failure_is_terminal_and_not_retried(tmp_path):
    cfg = cfg_for(tmp_path)
    cells = [(spec(), names_with("t_wrong_up_gemm"), "t_wrong_up_gemm")]
    D.run_sweep(cells, cfg, routing=lambda s: None, info=FAKE_INFO)
    D.run_sweep(cells, cfg, routing=lambda s: None, info=FAKE_INFO)
    assert len(SC.read_csv(cfg.csv_path)) == 1


# --- graph-mode cost policy -------------------------------------------------

def cost_for(bytes_total):
    from moe.bench.bytes_model import PipelineCost
    return PipelineCost(flops=1.0, bytes_total=bytes_total)


def fake_hw(bw_bytes_s=4.8e12, bf16_tflops=835.5):
    from moe.bench.roofline import Hardware
    return Hardware(name="fake", bandwidth_bytes_s=bw_bytes_s,
                    peak_flops={"bf16": bf16_tflops * 1e12, "fp32": 60e12},
                    source="test")


def test_graph_mode_is_skipped_when_launch_overhead_cannot_matter():
    """At DeepSeek geometry with a few tokens the roofline minimum is ~0.8 ms
    against a ~5 us launch. Timing that cell twice spends half a session
    measuring a sub-1% effect."""
    cfg = D.RunConfig(hardware=fake_hw(), graph_min_launch_share=0.01)
    ok, reason = D.should_time_graph(cost_for(3.76e9), cfg)   # ~0.78 ms
    assert not ok
    assert "0.6" in reason or "below" in reason


def test_graph_mode_is_kept_when_launch_overhead_is_first_order():
    cfg = D.RunConfig(hardware=fake_hw(), graph_min_launch_share=0.01)
    ok, reason = D.should_time_graph(cost_for(4.8e5), cfg)     # ~0.0001 ms
    assert ok and reason == ""


def test_policy_can_be_disabled():
    cfg = D.RunConfig(hardware=fake_hw(), graph_min_launch_share=0.0)
    assert D.should_time_graph(cost_for(1e12), cfg)[0] is True


def test_policy_errs_toward_measuring_when_hardware_is_unknown():
    """No calibration means no prediction, so measure rather than guess."""
    cfg = D.RunConfig(hardware=None, graph_min_launch_share=0.01)
    assert D.should_time_graph(cost_for(1e12), cfg)[0] is True


def test_skipped_graph_row_is_written_not_dropped(tmp_path):
    cfg = cfg_for(tmp_path, graph_modes=(True,), l2_modes=(True,),
                  hardware=fake_hw(bw_bytes_s=1.0), graph_min_launch_share=0.5)
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 1
    assert rows[0]["capture_status"] == "skipped"
    assert rows[0]["graph_skip_reason"]
    assert float(rows[0]["ms_p50"]) == 0.0


def test_routing_provenance_lands_in_the_row(tmp_path):
    cfg = cfg_for(tmp_path,
                  routing_info=lambda s: {"trace_sha": "deadbeefcafe0001",
                                          "trace_id": "toy4@b1l2"})
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    r = SC.read_csv(cfg.csv_path)[0]
    assert r["trace_sha"] == "deadbeefcafe0001"
    assert r["trace_id"] == "toy4@b1l2"


def test_fixed_routing_caveat_is_recorded(tmp_path):
    _, path = sweep(tmp_path, "t_counting_up_gemm")
    assert SC.read_csv(path)[0]["routing_fixed_across_iters"] == "True"


# --- regressions for the review findings ------------------------------------

FLAKY = {"n": 0}


@register
class FlakyUpGemm(StageSpan):
    """Correct on its first call, wrong afterwards.

    Models the exact failure CUDA-graph re-validation exists to catch: an
    implementation that looks right during the eager prologue but leaves output
    unwritten on replay, where the graph's fixed buffers still hold the
    previous correct values.
    """

    name = "t_flaky_up_gemm"
    covers = ("up_gemm",)
    requires_cuda = False
    dtypes = ("fp32", "bf16")

    def __call__(self, st: MoEState) -> None:
        FLAKY["n"] += 1
        h = R.grouped_gemm_loop(st.x_perm, st.weights.w1, st.expert_offsets,
                                2 * st.spec.model.intermediate_size)
        st.h_up = h if FLAKY["n"] == 1 else torch.zeros_like(h)


def test_graph_replay_is_validated_for_pipeline_scoped_cells(tmp_path):
    """Regression: for `scope=pipeline` the timed callable runs into a separate
    state, but verify_replay compared the PROLOGUE's state, which the replays
    never touch. That made the check a no-op for exactly the implementation
    shape (a whole-layer fused_moe) it most needed to cover.
    """
    FLAKY["n"] = 0
    cfg = cfg_for(tmp_path, graph_modes=(True,), l2_modes=(True,))
    names = names_with("t_flaky_up_gemm")
    D.run_sweep([(spec(), names, D.PIPELINE_SCOPE)], cfg,
                routing=lambda s: None, info=FAKE_INFO)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 1, cfg.manifest_path.read_text()
    r = rows[0]
    assert r["scope"] == "pipeline"
    assert FLAKY["n"] > 1, "the timed callable must actually have run"
    assert r["correctness_passed"] == "False", (
        "the replayed output was wrong and must fail the oracle")
    assert "REPLAYED" in r["notes"]
    # And the timing it did produce must not survive into the file.
    assert float(r["ms_p50"]) == 0.0
    assert float(r["tflops"]) == 0.0
    assert float(r["pct_of_achieved_tflops"]) == 0.0


def test_span_scoped_replay_is_still_validated(tmp_path):
    """The span path shares the prologue state, so it was never affected; keep
    it covered so a future refactor cannot break it silently."""
    FLAKY["n"] = 0
    cfg = cfg_for(tmp_path, graph_modes=(True,), l2_modes=(True,))
    D.run_sweep([(spec(), names_with("t_flaky_up_gemm"), "t_flaky_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    r = SC.read_csv(cfg.csv_path)[0]
    assert r["scope"] == "span"
    assert r["correctness_passed"] == "False"
    assert float(r["ms_p50"]) == 0.0


def test_a_failed_row_never_carries_derived_efficiency(tmp_path):
    """_TIMED_FIELDS named a column that had been deleted and missed the one
    that replaced it, so a discarded measurement still published its
    efficiency."""
    row = SC.Row(correctness_passed=False, ms_p50=1.0, tflops=5.0,
                 pct_of_achieved_tflops=42.0, implied_traffic_ratio=2.5,
                 compulsory_gbps=99.0, jitter_p90_over_p50=1.1)
    written = D._emit(SC.CsvWriter(tmp_path / "x.csv"),
                      SC.Manifest(tmp_path / "x.jsonl"), row, "k")
    assert written == 1
    for name in D._TIMED_FIELDS:
        assert getattr(row, name) == 0.0, name
    assert all(f in SC.COLUMNS for f in D._TIMED_FIELDS), (
        "_TIMED_FIELDS must name real columns; a stray name silently zeroes "
        "nothing at all")


def test_machine_info_cannot_collide_with_columns_the_row_owns(tmp_path):
    """Regression: the CLI put env_name into the machine-info dict, which
    _base_row also sets from cfg. Row() then raised "got multiple values for
    keyword argument", run_sweep caught it as a crash, and an entire sweep
    wrote zero rows while reporting only warnings."""
    hostile = dict(FAKE_INFO)
    # Every column _base_row sets itself, deliberately jammed into info.
    for name in D._ROW_OWNED_BY_CALLER:
        hostile[name] = "COLLIDE"
    cfg = cfg_for(tmp_path)
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=hostile)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 1, cfg.manifest_path.read_text()
    # The row's own values win; the hostile ones are dropped.
    assert rows[0]["impl"] == "t_counting_up_gemm"
    assert rows[0]["env_name"] == "base"
    assert rows[0]["model"] == "toy"


def test_the_cli_builds_an_info_dict_the_row_accepts():
    """Exercises the exact construction cli.main() uses. Nothing tested that
    path before, which is why a one-line addition there broke every sweep."""
    from moe.bench import cli

    info = {"gpu_name": "FakeH200", "torch_version": "x", "sm_count": 132}
    info["env_version"] = cli.env_version("base")
    assert "env_name" not in info, (
        "env_name belongs to _base_row; putting it in info collides")
    for key in info:
        assert key not in D._ROW_OWNED_BY_CALLER, key
    assert "env_version" in SC.COLUMNS


# --- the sweep refuses another machine's ceilings -----------------------------

def test_measured_ceilings_refuse_a_calibration_from_another_device(tmp_path,
                                                                    monkeypatch):
    """Before this check, cli.measured_ceilings() loaded the committed H200
    measured.yaml unconditionally. A sweep on any other GPU scored every row
    against 4375 GB/s and 730 TFLOP/s it could not reach, and nothing noticed
    until plot time, after the GPU hours were spent."""
    from moe.bench import cli
    from moe.bench import roofline as RL

    (tmp_path / "measured.yaml").write_text(
        "name: NVIDIA H200 (measured)\n"
        "verified: true\n"
        "source: calibrate_hardware.py\n"
        "memory:\n  bandwidth_tb_s: 4.3756\n"
        "compute_dense_tflops:\n  bf16: 729.99\n")
    monkeypatch.setattr(RL, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(RL, "current_gpu_name", lambda: "NVIDIA A100-SXM4-80GB")

    with pytest.raises(RL.HardwareMismatch, match="calibrate_hardware"):
        cli.measured_ceilings()


def test_measured_ceilings_are_empty_when_no_calibration_exists(tmp_path,
                                                                monkeypatch):
    """Absent calibration is not an error: the efficiency columns stay zero
    rather than being quoted against a datasheet peak."""
    from moe.bench import cli
    from moe.bench import roofline as RL

    monkeypatch.setattr(RL, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(RL, "current_gpu_name", lambda: "NVIDIA A100-SXM4-80GB")
    assert cli.measured_ceilings() == {}


# --- what an UNTIMED row says about its instrument --------------------------
#
# Four of the driver's paths emit a row no timer ever saw. Each of them left
# `instrument` at the `Row` default, an empty string, which `instrument_of`
# refuses and `has_kernel_timing` -- the predicate an analysis is told to split
# a pool with BEFORE it reads any v5 column -- therefore raised on. Nothing
# broke only because no v5 data exists yet and `alpha_refit.collect` happens to
# drop `ms_p50 <= 0` a few lines before it asks. These four pin the answer at
# the writing end instead.

def test_a_correctness_failed_row_names_no_instrument_without_refusing(tmp_path):
    """No timing mode ran, so no apparatus may be claimed -- and the row must
    still answer the question rather than raise at whoever asks it."""
    _, path = sweep(tmp_path, "t_wrong_up_gemm")
    row = SC.read_csv(path)[0]
    assert row["capture_status"] == "not_timed"
    assert SC.instrument_of(row) == SC.NO_INSTRUMENT
    assert SC.has_kernel_timing(row) is False
    assert float(row["ms_p50"]) == 0.0


def test_an_uncapturable_row_names_no_instrument(tmp_path):
    cfg = cfg_for(tmp_path, graph_modes=(True,), graph_timer=not_capturable)
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    row = SC.read_csv(cfg.csv_path)[0]
    assert row["capture_status"] == "not_capturable"
    assert SC.instrument_of(row) == SC.NO_INSTRUMENT
    assert SC.has_kernel_timing(row) is False


def test_a_policy_skipped_graph_row_names_no_instrument(tmp_path):
    cfg = cfg_for(tmp_path, graph_modes=(True,), l2_modes=(True,),
                  hardware=fake_hw(bw_bytes_s=1.0), graph_min_launch_share=0.5)
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    row = SC.read_csv(cfg.csv_path)[0]
    assert row["capture_status"] == "skipped"
    assert SC.instrument_of(row) == SC.NO_INSTRUMENT
    assert SC.has_kernel_timing(row) is False


def test_a_row_whose_timer_raised_names_no_instrument(tmp_path):
    """The timer got as far as raising, which is not as far as measuring."""

    def exploding_timer(fn, **kw):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    cfg = cfg_for(tmp_path, timer=exploding_timer)
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    row = SC.read_csv(cfg.csv_path)[0]
    assert "timing error" in row["notes"]
    assert SC.instrument_of(row) == SC.NO_INSTRUMENT
    assert SC.has_kernel_timing(row) is False


def test_a_pool_of_driver_rows_splits_without_a_single_try_block(tmp_path):
    """The documented usage, end to end: read a sweep that produced both kinds
    of row and partition it with `has_kernel_timing`, catching nothing."""
    cfg = cfg_for(tmp_path)
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm"),
                 (spec(), names_with("t_wrong_up_gemm"), "t_wrong_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    rows = SC.read_csv(cfg.csv_path)
    measured = [r for r in rows if SC.has_kernel_timing(r)]
    untimed = [r for r in rows if not SC.has_kernel_timing(r)]
    assert len(measured) == 1 and len(untimed) == 1
    assert SC.instrument_of(measured[0]) == T.TIMING_BASIS
    assert SC.timing_verdict(measured[0], "clock_level_ok") == "ok"


# --- the retired instrument's knobs cannot be silently dropped --------------

def test_a_retired_knob_stops_the_sweep_instead_of_being_dropped(tmp_path):
    """`_instrument_kwargs` passes warmup_ms/target_ms/trials/l2_flush/
    reference_clock_mhz/flusher, so `warmup` (a CALL COUNT) and `iters` reach
    nothing -- while `moe/bench/cli.py` fills both from the profile on every
    run. `profile-cell` set warmup=5, trials=1, iters=1 and its note read "one
    cell, one launch: the shape ncu can read a counter off"; on the instrument
    the one launch became `iters_for(per_call_ms, 200)`, which is 10 to 2000,
    and nothing said so. A counter read off the wrong shape looks exactly like
    a counter read off the right one."""
    with pytest.raises(D.RetiredKnobRefused, match="RETIRED instrument knobs"):
        sweep(tmp_path, "t_counting_up_gemm", warmup=5, iters=1)


def test_the_refusal_leaves_the_sweep_rather_than_becoming_one_cell_s_crash(
        tmp_path, capsys):
    """`run_sweep` records any per-cell exception as a crash and carries on,
    which is right for a kernel and wrong for a config: the same config would
    fail every cell, so the sweep would print a warning per cell, write no rows
    and exit 0."""
    cfg = cfg_for(tmp_path, iters=1)
    with pytest.raises(D.RetiredKnobRefused):
        D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                      "t_counting_up_gemm")] * 3, cfg,
                    routing=lambda s: None, info=FAKE_INFO)
    assert "[warn]" not in capsys.readouterr().out
    assert not cfg.csv_path.exists() or SC.read_csv(cfg.csv_path) == []


def test_the_refusal_names_only_the_modes_that_would_drop_the_knob():
    """A run that measures only eager is not told about the graph timer, and a
    mode explicitly put back on the retired seam is honoured rather than
    refused: that timer does read a call count."""
    with pytest.raises(D.RetiredKnobRefused, match="the eager mode of this run"):
        D.refuse_dropped_retired_knobs(D.RunConfig(warmup=5, graph_modes=(False,)))
    with pytest.raises(D.RetiredKnobRefused, match="the graph mode of this run"):
        D.refuse_dropped_retired_knobs(D.RunConfig(warmup=5, graph_modes=(True,)))
    with pytest.raises(D.RetiredKnobRefused, match="iters=1"):
        D.refuse_dropped_retired_knobs(D.RunConfig(iters=1))
    cfg = D.RunConfig(warmup=5, iters=1, graph_modes=(False,),
                      timer_eager=legacy_timer)
    assert D.unhonourable_retired_knobs(cfg) == []
    D.refuse_dropped_retired_knobs(cfg)


def test_a_config_nothing_measures_with_is_never_refused(tmp_path):
    """WHY THE CHECK IS PER CELL AND NOT IN `RunConfig.__init__`. A knob is
    only dropped by a cell that is actually measured, and
    `tests/test_force_tile.py` drives the CLI purely to prove a force-tile plan
    is refused before anything is spent -- every cell declined by the pin, none
    of them timed. Refusing at construction would fail such a run over a knob no
    cell reached."""
    cfg = cfg_for(tmp_path, warmup=5, iters=10)
    assert D.unhonourable_retired_knobs(cfg)
    path = D.run_sweep([], cfg, routing=lambda s: None, info=FAKE_INFO)
    assert SC.read_csv(path) == []


def test_the_ordinary_sweep_is_not_refused():
    """The FAIL branch above is only worth planting because the PASS branch is
    the common one: a config that never asked for a call count asked for
    nothing, and passing the field default is not asking."""
    assert D.unhonourable_retired_knobs(D.RunConfig()) == []
    assert D.unhonourable_retired_knobs(D.RunConfig(warmup=25, iters=None)) == []
    assert D.unhonourable_retired_knobs(D.RunConfig(warmup_ms=5.0, trials=1)) == []


# --- and the refusal has to reach a caller as a RESULT, not a traceback -----

def cfg_the_cli_would_build(profile, **kw):
    """The RunConfig `cli.main` builds for a profile, without a GPU or a sweep.

    Mirrors `cli.main`'s `cfg_kw` exactly. Written out rather than imported
    because the point of the check is that the CLI's construction and the
    driver's refusal agree, and a helper shared with the code under test could
    not tell you that.
    """
    kwargs = dict(trials=profile.trials, l2_modes=profile.l2_modes,
                  graph_modes=profile.graph_modes)
    kwargs.update({name: value for name, value in
                   (("warmup_ms", profile.warmup_ms),
                    ("target_ms", profile.target_ms)) if value is not None})
    kwargs.update(kw)
    return D.RunConfig(**kwargs)


@pytest.mark.parametrize("name", sorted(PR.PROFILES))
def test_no_shipped_profile_asks_the_instrument_for_a_knob_it_cannot_honour(name):
    """EVERY documented session command went through one of these. `smoke` and
    `profile-cell` set `warmup`/`iters`, `cli` copied both into every RunConfig,
    and the refusal then fired on the first cell of `scripts/run_all.sh` line
    349 -- which runs `--profile smoke` under `set -euo pipefail` BEFORE the
    real sweep. `Profile` no longer carries the fields to copy."""
    profile = PR.get(name)
    assert not hasattr(profile, "warmup") and not hasattr(profile, "iters")
    assert D.unhonourable_retired_knobs(cfg_the_cli_would_build(profile)) == []


def test_the_two_quick_profiles_say_quick_in_the_instrument_s_units():
    """The FAIL branch of the check above, planted with the exact values the two
    profiles used to carry, so a revert cannot pass quietly."""
    with pytest.raises(D.RetiredKnobRefused, match="RETIRED instrument knobs"):
        D.refuse_dropped_retired_knobs(D.RunConfig(warmup=5, iters=10))
    with pytest.raises(D.RetiredKnobRefused, match="RETIRED instrument knobs"):
        D.refuse_dropped_retired_knobs(D.RunConfig(warmup=5, iters=1))

    smoke, cell = PR.get("smoke"), PR.get("profile-cell")
    assert (smoke.warmup_ms, smoke.target_ms, smoke.trials) == (25.0, 25.0, 1)
    # "one launch" is not sayable: `iters_for`'s floor is 10, and a target below
    # any real per-call time is the smallest honest ask there is.
    assert (cell.warmup_ms, cell.target_ms, cell.trials) == (25.0, 1.0, 1)
    assert T.iters_for(0.5, cell.target_ms) == 10


def test_the_cli_turns_the_refusal_into_REFUSED_and_not_a_traceback(tmp_path,
                                                                    monkeypatch,
                                                                    capsys):
    """THE WHOLE POINT OF THE REFUSAL BEING A REFUSAL. Uncaught, it leaves
    `main` as a traceback at process status 1, which `exit_codes` reads as
    CLAIM_FAIL: "measured; VALIDITY passed; a CLAIM gate did not... a RESULT,
    not a retry". Nothing was measured, so that reading is false in every field,
    and `scripts/run_all.sh` runs under `set -euo pipefail`, so the repository's
    top-level sweep script aborted before it measured anything."""
    def refuse(*a, **kw):
        raise D.RetiredKnobRefused("warmup=5 is a RETIRED instrument knob")

    monkeypatch.setattr(cli, "run_sweep", refuse)
    code = cli.main(["--profile", "smoke", "--out-dir", str(tmp_path),
                     "--groups", "reference"])
    assert code == EC.REFUSED
    assert code != EC.CLAIM_FAIL, "1 would say the world disagreed with a claim"
    captured = capsys.readouterr()
    assert "RETIRED instrument knob" in captured.err
    assert EC.ledger_state(code) != "RETRY"


def test_the_documented_smoke_invocation_reaches_the_sweep(tmp_path, monkeypatch):
    """`scripts/run_all.sh` line 349, verbatim minus the venv path. It is the
    first thing every session runs and the last thing that should be able to
    fail on a configuration error."""
    seen = {}

    def record(cells, cfg, routing, info=None):
        seen["cfg"] = cfg
        D.refuse_dropped_retired_knobs(cfg)   # what the first cell would do
        return cfg.csv_path

    monkeypatch.setattr(cli, "run_sweep", record)
    code = cli.main(["--profile", "smoke", "--out-dir", str(tmp_path),
                     "--groups", "reference,kernels"])
    assert code == EC.DONE
    cfg = seen["cfg"]
    assert (cfg.warmup_ms, cfg.target_ms, cfg.trials) == (25.0, 25.0, 1)
    # The retired fields are still there, still at their defaults, and therefore
    # still nothing the CLI asked for.
    assert cfg.warmup == 25 and cfg.iters is None


def test_the_gpu_driver_fixture_is_on_the_instrument_too():
    """tests/test_gpu.py's `make_cfg` built `RunConfig(warmup=3, iters=5)`, so
    every driver end-to-end test on the device refused before it timed anything
    -- and all 39 of them skip without CUDA, so no laptop run could show it.
    Read the file rather than import it: importing registers its spans into the
    global registry, and this check has to be cheap enough to always run."""
    import ast

    tree = ast.parse(pathlib.Path("tests/test_gpu.py").read_text())
    fixture = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "make_cfg")
    asked = {kw.arg for call in ast.walk(fixture)
             if isinstance(call, ast.Call) for kw in call.keywords}
    assert asked & {"warmup_ms", "target_ms"}, "it has to say something"
    assert not asked & set(D.RETIRED_KNOBS), sorted(asked)


# --- the graph timer, off the GPU ------------------------------------------
#
# `time_kernel_graph` is the default for every graph row the driver will ever
# publish and it is entirely new code, yet every driver test above replaces the
# whole function with `fake_graph_timer`. What follows drives the real function
# with `torch.cuda` faked out, so the capture, the priming order, the
# re-verification point and the refusal conversion are exercised on a laptop.
# It is NOT a substitute for a hardware test; tests/test_gpu.py still tests the
# retired `T.time_graph` it supersedes and nothing there touches this.

class _FakeStream:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits {other.name}")


def fake_cuda(monkeypatch, log, capture_error: str | None = None):
    """Enough of `torch.cuda` for `time_kernel_graph`, recording the order."""
    import contextlib

    current = _FakeStream(log, "current")
    side = _FakeStream(log, "side")

    class FakeGraph:
        def replay(self):
            log.append("replay")

    @contextlib.contextmanager
    def stream(s):
        log.append(f"on {s.name}")
        yield
        log.append(f"off {s.name}")

    @contextlib.contextmanager
    def graph(g):
        log.append("capture begin")
        if capture_error:
            raise RuntimeError(capture_error)
        yield
        log.append("capture end")

    monkeypatch.setattr(torch.cuda, "Stream", lambda: side)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda *a, **k: log.append("sync"))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "graph", graph)
    return log


def test_the_graph_timer_primes_captures_replays_then_verifies(monkeypatch):
    """The order is the point. The prime runs on a SIDE stream, the capture
    records one call, the warmup replays run BEFORE `on_captured`, and what is
    handed to the instrument is `graph.replay` and not the original callable --
    a graph row that timed `fn` directly would be an eager row wearing the
    label."""
    log: list[str] = []
    fake_cuda(monkeypatch, log)
    seen = {}

    def timed(fn, **kw):
        log.append("time_kernel")
        seen["fn"], seen["kw"] = fn, kw
        return "the timing"

    monkeypatch.setattr(T, "time_kernel", timed)
    out = D.time_kernel_graph(lambda: log.append("fn"),
                              on_captured=lambda: log.append("verify"),
                              warmup_ms=5.0, trials=1, l2_flush=False)

    assert out == "the timing"
    assert log == ["side waits current", "on side", "fn", "fn", "fn",
                   "off side", "current waits side", "sync",
                   "capture begin", "fn", "capture end",
                   "replay", "replay", "replay", "sync", "verify",
                   "time_kernel"]
    assert seen["fn"].__self__.__class__.__name__ == "FakeGraph"
    assert seen["kw"] == {"warmup_ms": 5.0, "trials": 1, "l2_flush": False}


def test_an_uncapturable_span_becomes_a_finding_not_a_crash(monkeypatch):
    """A RuntimeError out of the capture is converted, because an
    implementation that syncs with the host cannot be used in real MoE
    inference and the row has to record that rather than end the sweep. The
    verification never runs: there is no graph to have produced an output."""
    log: list[str] = []
    fake_cuda(monkeypatch, log, capture_error="operation not permitted during "
                                              "stream capture")
    monkeypatch.setattr(T, "time_kernel",
                        lambda *a, **k: pytest.fail("timed an uncaptured graph"))

    with pytest.raises(T.NotCapturable, match="not permitted"):
        D.time_kernel_graph(lambda: log.append("fn"),
                            on_captured=lambda: log.append("verify"),
                            warmup_ms=5.0)
    assert "verify" not in log
    assert log.count("replay") == 0


# --- the LEVEL reference: resolved from the card, or refused ----------------

def clock_from(mhz, *, card="NVIDIA H200", profile="NVIDIA H200 (measured)",
               source="", grade="", family=RF.BF16_FAMILY):
    """A resolver returning one planted `ReferenceClock`, the way `cfg_for`
    plants a timer. Both branches that matter -- a card with a calibration and a
    card without one -- need a card ATTACHED, and no test box has one.

    `grade` defaults to none, which REFUSES the per-row roof; a test that wants
    the roof written plants `RF.REFERENCE_UNDER_LOAD` and says so."""
    return lambda: RF.ReferenceClock(
        mhz, source or (f"{profile}: planted" if mhz else "planted refusal"),
        card=card, profile=profile if mhz else "", grade=grade if mhz else "",
        family=family)


def levelling_timer(fn, *, load_mhz, reference_clock_mhz=None, warmup_ms=300.0,
                    target_ms=200.0, trials=1, l2_flush=True, flusher=None,
                    on_captured=None):
    """`fake_timer`, except the two clock verdicts come from the real
    `timing.clock_flags` over a planted under-load clock.

    The point of this test section is the WIRE: that the number the driver
    resolved reaches `time_kernel`'s `reference_clock_mhz` and turns into a
    column. A fake that took `level=` as a parameter, as `fake_timer` does,
    would pass whether or not the wire existed.
    """
    fn()
    if on_captured is not None:
        on_captured()
    level, drift = T.clock_flags(load_mhz, load_mhz, load_mhz,
                                 reference_clock_mhz)
    return T.KernelTiming(
        ms_p50=1.0, ms_p90=1.2, ms_min=0.9, ms_std=0.05, iters=2, trials=trials,
        warmup_ms=warmup_ms, l2_flush=l2_flush, sm_clock_load_mhz=load_mhz,
        sm_clock_start_mhz=load_mhz, sm_clock_end_mhz=load_mhz,
        clock_level_ok=level, clock_drift_ok=drift, samples=6, warmup_calls=17,
        flush_mb=(flusher.megabytes if flusher is not None else 0),
        clock_samples=6, clock_source="injected", clock_poll_ms=0.01,
        host_bound=False, host_enqueue_ms=0.2)


def test_the_reference_clock_comes_from_the_attached_card_s_own_calibration():
    """THE FLAG THAT WAS PERMANENTLY UNDETERMINED. `RunConfig` left
    `reference_clock_mhz` at None under a comment reading "No committed
    calibration records it yet", while `measured_nvidia_h200.yaml` has carried
    a GEMM clock all along and `scripts/block_m_crossing_sweep.py`'s own
    `reference_clock_mhz()` has read it. Without a reference `clock_flags`
    leaves `clock_level_ok` None, so the LEVEL verdict could never fire on the
    path that wrote all 100,144 published rows -- and LEVEL is the entire reason
    `TIMING_BASIS` left v1. The number is the 2026-09-09 calibration's
    UNDER-LOAD median, 1485 MHz at 691 W, not the idle scalar the file used to
    carry."""
    cfg = D.RunConfig(reference_clock_resolver=lambda: RF.reference_clock("NVIDIA H200"))
    assert cfg.reference_clock_mhz == 1485.0
    assert "gemm_clock" in cfg.reference_clock_source
    assert cfg.missing == {}
    # And it is the same number, from the same field, as the sweep resolves.
    assert cfg.reference_clock_mhz == RF.reference_clock("NVIDIA H200").mhz


def test_a_low_clock_row_off_that_config_reports_LEVEL_failed(tmp_path):
    """The wire, end to end and into the CSV: resolved 1515 -> the instrument's
    `reference_clock_mhz` -> `clock_flags` -> the `clock_level_ok` column. A
    card at 1400 MHz against a roof measured at 1515 is 92.4%, under the
    band's snapped 1440 MHz edge, and its fixed-roof fraction is delivered
    throughput at a lower issue rate. The retired flag cannot see it at all:
    both its samples are 1400, so the drop is zero.

    AND OUT THE OTHER SIDE INTO `throttled`, WHICH IS NOW FALSE FOR BOTH ROWS.
    Until 2026-09-09 the LOW verdict set it; the H200 session then showed the
    LOW side is a tile's steady state under the 700 W cap, so the verdict is
    recorded and the row is kept. `throttled` says DRIFT, and the drift row
    beside these two is where the four consumers that read it (pod_session.sh
    S6d, run_all.sh, publish_results.sh, efficiency_report.py) see a True on
    a real sweep."""
    seen = {}
    for load, expected, flagged in ((1400.0, SC.VERDICT_FAILED, "False"),
                                    (1500.0, SC.VERDICT_OK, "False")):
        cfg = cfg_for(tmp_path / str(load),
                      timer=partial(levelling_timer, load_mhz=load),
                      reference_clock_resolver=clock_from(1515.0))
        D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                      "t_counting_up_gemm")], cfg, routing=lambda s: None,
                    info=FAKE_INFO)
        r = seen[load] = SC.read_csv(cfg.csv_path)[0]
        assert SC.timing_verdict(r, "clock_level_ok") == expected, load
        assert float(r["sm_clock_load_mhz"]) == load
        assert r["throttled"] == flagged, load
        assert SC.row_bool(r, "throttled") is (flagged == "True"), load
    # The retired flag cannot tell the two rows apart: it compares two
    # idle-instant samples, and this card sat at one clock for the whole cell.
    assert T.clock_drift(T.ClockState(1400, 50), T.ClockState(1400, 50))[1] is False
    assert seen[1400.0]["clock_drift_ok"] == seen[1500.0]["clock_drift_ok"]
    # The LOW row is KEPT and says where it sat; the side is the record.
    assert seen[1400.0]["clock_level_side"] == T.LEVEL_LOW
    assert SC.row_bool(seen[1400.0], "throttled") is False
    # And a row whose clock MOVED is the one that is excluded.
    drifted = cfg_for(tmp_path / "drift",
                      timer=partial(fake_timer, level=True, drift=False),
                      reference_clock_resolver=clock_from(1515.0))
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                  "t_counting_up_gemm")], drifted, routing=lambda s: None,
                info=FAKE_INFO)
    moved = SC.read_csv(drifted.csv_path)[0]
    assert SC.row_bool(moved, "throttled") is True


def test_a_card_with_no_calibration_refuses_rather_than_measuring_undetermined(
        tmp_path):
    """REFUSE RATHER THAN DEFAULT. A pod holding a card whose ruler was never
    measured is about to spend metered minutes on rows whose LEVEL column can
    never say anything, which is not a conservative sweep but an unexamined one.
    The reason is the one in the missing map, quoted rather than re-derived."""
    resolver = clock_from(None, card="NVIDIA B200",
                          source="no calibration for 'NVIDIA B200'")
    cfg = cfg_for(tmp_path, reference_clock_resolver=resolver)
    assert cfg.reference_clock_mhz is None
    assert cfg.missing["reference_clock_mhz"] == "no calibration for 'NVIDIA B200'"
    assert D.unreferenced_clock(cfg) == ["eager"]
    with pytest.raises(D.ReferenceClockRefused) as e:
        D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                      "t_counting_up_gemm")] * 3, cfg,
                    routing=lambda s: None, info=FAKE_INFO)
    assert "NVIDIA B200" in str(e.value)
    assert "no calibration for 'NVIDIA B200'" in str(e.value)
    assert "calibrate_hardware.py --publish" in str(e.value)
    assert "reference_clock_mhz=" in str(e.value)
    # Nothing was spent: the refusal is raised before the first cell writes.
    assert not cfg.csv_path.exists() or SC.read_csv(cfg.csv_path) == []


def test_the_refusal_exits_REFUSED_and_not_ERROR():
    """`RunConfig` is built in `cli._main` OUTSIDE the `except TimingRefused`
    that makes a refusal REFUSED(2), so resolving CANNOT raise: it records the
    reason and `run_sweep` raises. A `TimingRefused` subclass because that is
    the type the handler names; anything else reaches `cli.main`'s catch-all and
    exits ERROR(4), which is commit 366b4de's defect one file over."""
    assert issubclass(D.ReferenceClockRefused, T.TimingRefused)
    cfg = D.RunConfig(reference_clock_resolver=clock_from(None, card="NVIDIA B200"))
    assert cfg.reference_clock_mhz is None          # constructed, not raised
    assert EC.ledger_state(EC.REFUSED) != "RESULT"


def test_no_card_attached_records_the_reason_and_measures_anyway(tmp_path):
    """The other world, and why the refusal turns on the CARD rather than on
    the clock alone. Off a GPU there is nothing to be level against: the
    instrument itself refuses unless the caller injected the fakes that stand in
    for CUDA, and an injected clock sampler has no card behind it. Refusing here
    would fail every laptop test in this file over a machine fact."""
    cfg = cfg_for(tmp_path, reference_clock_resolver=RF.reference_clock,
                  timer=partial(levelling_timer, load_mhz=1500.0))
    assert cfg.reference_clock_card == ""
    assert cfg.reference_clock_mhz is None
    assert "no CUDA device" in cfg.missing["reference_clock_mhz"]
    assert D.unreferenced_clock(cfg) == []
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                  "t_counting_up_gemm")], cfg, routing=lambda s: None,
                info=FAKE_INFO)
    r = SC.read_csv(cfg.csv_path)[0]
    assert SC.timing_verdict(r, "clock_level_ok") == SC.VERDICT_UNDETERMINED
    assert SC.timing_verdict(r, "clock_drift_ok") == SC.VERDICT_OK


def test_a_non_positive_reference_is_not_a_reference(tmp_path):
    """`clock_flags` reads any value <= 0 as no reference at all, so a caller
    passing 0 would have bought the silent undetermined column this change
    exists to end. Dropped to None with the reason, and then refused on a card
    like any other missing reference.

    AND THE REFUSAL IS THE HALF THAT SHIPPED INERT. `unreferenced_clock` turns
    on TWO fields, a None clock and a NON-EMPTY CARD, and the first cut of this
    branch set the first and left the second at "": the wall built at one of
    the two ways in, this rebuild's recurring defect, reached through the very
    argument the refusal message offers as the way out. `RunConfig(
    reference_clock_mhz=0)` on a pod then dropped the number, recorded the
    reason, refused nothing, and wrote the whole sweep at
    `clock_level_ok = undetermined`. The resolver is consulted for the CARD on
    every path that ends without a clock, and not for a clock to stand in.
    """
    cfg = cfg_for(tmp_path, reference_clock_mhz=0.0,
                  reference_clock_resolver=clock_from(1515.0))
    assert cfg.reference_clock_mhz is None
    assert "is not a clock" in cfg.missing["reference_clock_mhz"]
    assert T.clock_flags(1500.0, 1500.0, 1500.0, 0.0)[0] is None
    # The card came from the resolver, so the refusal has both of its terms.
    assert cfg.reference_clock_card == "NVIDIA H200"
    assert D.unreferenced_clock(cfg) == ["eager"]
    with pytest.raises(D.ReferenceClockRefused) as e:
        D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                      "t_counting_up_gemm")], cfg, routing=lambda s: None,
                    info=FAKE_INFO)
    assert "NVIDIA H200" in str(e.value)
    assert "is not a clock" in str(e.value)
    assert not cfg.csv_path.exists() or SC.read_csv(cfg.csv_path) == []
    # 1515 was there to be read and was NOT read: a bad explicit number is
    # refused, never quietly replaced by the calibration it overrode.
    assert "1515" not in cfg.reference_clock_source
    # THE OTHER SIDE OF THE SAME BRANCH: off a card there is nothing to be
    # level against, so the same bad value records its reason and measures on.
    laptop = cfg_for(tmp_path / "laptop", reference_clock_mhz=0.0,
                     reference_clock_resolver=clock_from(None, card=""))
    assert laptop.reference_clock_card == ""
    assert "is not a clock" in laptop.missing["reference_clock_mhz"]
    assert D.unreferenced_clock(laptop) == []


def test_an_explicit_reference_wins_and_says_so(tmp_path):
    """The documented way out of the refusal. It is recorded as the caller's
    number rather than as the card's, so a write-up cannot cite a calibration
    that was never read."""
    cfg = cfg_for(tmp_path, reference_clock_mhz=1980.0,
                  reference_clock_resolver=clock_from(1515.0))
    assert cfg.reference_clock_mhz == 1980.0
    assert "given by the caller" in cfg.reference_clock_source
    assert D.unreferenced_clock(cfg) == []


def test_the_clock_and_the_roof_must_come_from_one_file(tmp_path):
    """TWO READINGS OF ONE CALIBRATION. `hardware` and the reference clock are
    both read out of `measured_<card>.yaml`, and resolving them separately is
    how a run ends up levelled against one card while scored against another. A
    datasheet roof was never measured at any clock, so pairing it with the
    calibration's GEMM clock would give LEVEL a left-hand side that belongs to
    something else."""
    datasheet = RF.load_hardware("h200_nvl", allow_unverified=True)
    cfg = D.RunConfig(hardware=datasheet,
                      reference_clock_resolver=clock_from(1515.0))
    assert cfg.reference_clock_mhz is None
    assert "NVIDIA H200 NVL" in cfg.missing["reference_clock_mhz"]
    assert D.unreferenced_clock(cfg) == ["eager", "graph"]
    # The measured profile the clock came from is accepted.
    measured = RF.load_hardware("measured_nvidia_h200")
    ok = D.RunConfig(hardware=measured, reference_clock_resolver=clock_from(1515.0))
    assert ok.reference_clock_mhz == 1515.0


def test_the_cli_s_own_config_resolves_the_clock_it_will_measure_against(
        tmp_path, monkeypatch):
    """The construction the CLI actually performs, with the resolution left at
    its default. On a laptop that is the no-card branch; the assertion that
    matters is that `cli` reaches `RunConfig` at all with the field unset, so
    the resolution is the CLI's behaviour and not something a script has to
    remember to pass."""
    seen = {}
    monkeypatch.setattr(cli, "run_sweep",
                        lambda cells, cfg, routing, info=None: seen.setdefault(
                            "cfg", cfg) or cfg.csv_path)
    assert cli.main(["--profile", "smoke", "--out-dir", str(tmp_path),
                     "--groups", "reference"]) == EC.DONE
    cfg = seen["cfg"]
    assert cfg.reference_clock_source == RF.reference_clock().source
    assert cfg.reference_clock_mhz == RF.reference_clock().mhz


def committed_calibrations():
    """`(card, doc)` for every calibration committed under `moe/bench/hardware`.

    The tests below are about the FILES, not about a planted document: a rule
    that holds on a fixture and not on the two yamls the pod will actually read
    is the shape of defect this section exists to close.
    """
    import yaml

    out = []
    for path in sorted(RF.HARDWARE_DIR.glob("measured_*.yaml")):
        doc = yaml.safe_load(path.read_text()) or {}
        card = str((doc.get("detail") or {}).get("gpu_name") or "")
        if card:
            out.append((card, doc))
    return out


def load_sweep_module():
    """`scripts/block_m_crossing_sweep.py`, loaded by path the way the rest of
    the suite loads a script. Registered in `sys.modules` before exec because
    `@dataclass` resolves annotations through `sys.modules[cls.__module__]`."""
    name = "block_m_crossing_sweep"
    if name in sys.modules:
        return sys.modules[name]
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        name, root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_two_resolvers_in_this_tree_read_one_clock_per_card():
    """THE SECOND CALL SITE, and it is in another file. This tree resolves the
    LEVEL reference twice: `roofline.reference_clock`, which the driver's config
    now uses, and `scripts/block_m_crossing_sweep.reference_clock_mhz`, which
    the sweep has used all along. Two copies of one rule is exactly how the
    driver came to believe no calibration recorded a clock while the sweep read
    1515 out of the committed yaml, and a divergence here would be worse than
    that was: the two paths would level rows against two different references
    and the exclusions would look like a property of the card.

    Numbers AND the quantity, because agreeing by accident is not agreeing:
    both name what they read in their reason, and the reason is what a LEVEL
    verdict is traced back to. Since the 2026-09-09 recalibration both read
    the under-load median rather than the post-hoc idle scalar, and both say
    so in the same words. Unifying the two is not this file's to make (the
    sweep is another owner's), so the wall goes here, where a future edit to
    either one fails the suite instead of the rental.
    """
    sweep = load_sweep_module()
    cards = [card for card, _ in committed_calibrations()]
    assert cards, "no committed calibration to compare the two resolvers over"
    for card in cards:
        ours, theirs = RF.reference_clock(card), sweep.reference_clock_mhz(card)
        assert ours.mhz == theirs[0], card
        assert ours.mhz and ours.mhz > 0, card
        # The QUANTITY, in the words both write. Since the 2026-09-09
        # recalibration the H200 answers from the under-load median and the
        # A100 still from the post-hoc idle scalar, so the phrase differs per
        # card; what may not differ is the two resolvers' account of the same
        # card. The sweep's reason, past its card prefix, is ours verbatim.
        assert theirs[1].split(": ", 1)[1] in ours.source, (card, theirs[1])
    # And the missing case is None on both, not a guess on either.
    assert RF.reference_clock("NVIDIA B200").mhz is None
    assert sweep.reference_clock_mhz("NVIDIA B200")[0] is None


def test_the_clock_cannot_come_out_of_a_file_the_roof_was_refused_from(tmp_path):
    """ONE CALIBRATION, TWO READINGS, AND THEY HAVE TO BE THE SAME FILE.
    `measured_doc` walked `load_measured`'s candidate NAMES in `load_measured`'s
    order and then took the first file that merely EXISTED and parsed, while
    `load_measured` reads each candidate through `load_hardware`, which SKIPS
    one on `verified: false`, on a null bandwidth and on a missing block. The
    two rules diverge on exactly the file that matters: an unverified
    `measured_<card>.yaml` beside a valid `measured.yaml` levelled the run
    against 1935 MHz while scoring it against the other file's roof, and the
    config's own clock-and-roof cross-check could not see it because both files
    carry the same `name:`. The candidate test is now `load_hardware` itself,
    so it cannot be restated wrongly.

    The remaining copy of this rule is `block_m_crossing_sweep._measured_yaml`,
    another owner's file, which still accepts an unverified candidate. It is
    named here so the next reader finds it; the pinning test above holds the two
    together on the committed, verified yamls.
    """
    import yaml

    card = "NVIDIA H200"
    def write(stem, *, verified, bf16, clock):
        (tmp_path / f"{stem}.yaml").write_text(yaml.safe_dump({
            "name": "NVIDIA H200 (measured)", "verified": verified,
            "memory": {"bandwidth_tb_s": 4.37},
            "compute_dense_tflops": {"bf16": bf16},
            "detail": {"gpu_name": card, "gemm_clock_mhz": clock}}))

    write(RF.measured_slug(card), verified=False, bf16=900.0, clock=1935)
    write("measured", verified=True, bf16=712.0, clock=1470)

    hw = RF.load_measured(card, directory=tmp_path)
    doc, reason = RF.measured_doc(card, directory=tmp_path)
    ref = RF.reference_clock(card, directory=tmp_path)
    assert reason == ""
    # The roof and the clock are two readings of ONE document, asserted as one
    # document and not as two numbers that happen to agree.
    assert hw.peak_flops["bf16"] == doc["compute_dense_tflops"]["bf16"] * 1e12
    assert ref.mhz == 1470.0
    # And the number that was there to be taken, and was not taken.
    unverified = yaml.safe_load(
        (tmp_path / f"{RF.measured_slug(card)}.yaml").read_text())
    assert unverified["detail"]["gemm_clock_mhz"] == 1935
    assert "1935" not in ref.source

    # THE FAIL BRANCH: the unverified file alone. `load_measured` has no roof,
    # so `reference_clock` must have no clock either, and say which file it
    # skipped rather than reading a number out of it.
    (tmp_path / "measured.yaml").unlink()
    assert RF.load_measured(card, directory=tmp_path) is None
    empty, why = RF.measured_doc(card, directory=tmp_path)
    assert empty == {}
    assert "UnverifiedHardware" in why and RF.measured_slug(card) in why
    bare = RF.reference_clock(card, directory=tmp_path)
    assert bare.mhz is None and bare.card == card
    # On a pod that is a refusal, not a sweep of undetermined rows.
    cfg = cfg_for(tmp_path / "run", reference_clock_resolver=lambda: bare)
    assert cfg.missing["reference_clock_mhz"] == why
    with pytest.raises(D.ReferenceClockRefused):
        D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                      "t_counting_up_gemm")], cfg, routing=lambda s: None,
                    info=FAKE_INFO)


def test_a_calibration_resolves_a_reference_its_own_plateau_can_clear(tmp_path):
    """WHAT TURNING THE FLAG ON COSTS IF THE REFERENCE IS A BOOST CLOCK. LEVEL
    excludes a cell whose loaded clock is under `LEVEL_FRACTION` of the
    reference, and until 2026-09-03 no row could be excluded because no
    reference existed. Now one does, so a calibration that publishes an IDLE
    BOOST as its GEMM clock would fail every cell of the arm it was measured
    for -- the S6d gate, `alpha_refit`'s clock gate and `efficiency_report` all
    at once -- and the exclusions would read as a hot card rather than as a
    bad ruler.

    The check is the calibration's own settle PLATEAU against the reference
    resolved out of the same file. The plateau and not the settle history: the
    history holds the transient (the A100's opens at 1245, 93.3% of its own
    reference) and `time_kernel` warms for a duration of sustained load before
    it samples, so the transient is not what any cell is timed at. Committed
    margins after the 2026-09-09 recalibration are H200 1470/1485 = 99.0% (it
    was 97.0% against the old idle scalar of 1515) and A100 1275/1335 = 95.5%,
    the A100 half a point inside the band, which is the number to watch when
    either card is recalibrated. The band no longer excludes anything, so this
    is a check on the RULER rather than on the cells: a calibration whose own
    plateau cannot clear its own reference has published the wrong clock.
    """
    for card, doc in committed_calibrations():
        ref = RF.reference_clock(card)
        plateau = ((doc.get("detail") or {}).get("settle") or {}).get("final_mhz")
        assert plateau, card
        assert plateau >= T.LEVEL_FRACTION * ref.mhz, (
            f"{card}: the reference {ref.mhz} MHz resolved from this file is "
            f"one its own plateau of {plateau} MHz cannot clear, so every cell "
            "of the next arm on this card would be flagged")
        assert T.clock_flags(float(plateau), float(plateau), float(plateau),
                             ref.mhz)[0] is True, card
    # THE FAIL BRANCH, planted: the same file with the idle boost published as
    # its GEMM clock. 1980 is this H200's own idle reading, recorded four times
    # over in its bandwidth patterns, and it is what `gemm_clock_mhz` held on
    # the calibrations whose scalars ran to 1935.
    import yaml

    card, doc = committed_calibrations()[-1]
    doc["detail"]["gemm_clock_mhz"] = 1980
    # The under-load median is the field that WINS the walk when the file has
    # one (the H200 does since 2026-09-09), so the planted boost has to go
    # where the resolver will actually read it.
    if isinstance(doc["detail"].get("gemm_clock"), dict):
        doc["detail"]["gemm_clock"]["median_mhz"] = 1980
    (tmp_path / f"{RF.measured_slug(card)}.yaml").write_text(yaml.safe_dump(doc))
    boosted = RF.reference_clock(card, directory=tmp_path)
    assert boosted.mhz == 1980.0
    plateau = doc["detail"]["settle"]["final_mhz"]
    assert plateau < T.LEVEL_FRACTION * boosted.mhz
    assert T.clock_flags(float(plateau), float(plateau), float(plateau),
                         boosted.mhz)[0] is False


# --- and a refusal out of the instrument itself is not one cell's error -----

def test_an_instrument_refusal_leaves_the_sweep_instead_of_being_recorded(
        tmp_path, capsys):
    """THE SECOND WAY IN. `timing.TimingRefused` subclasses `RuntimeError`, and
    `_run_modes` filed every RuntimeError out of the timer as this one cell's
    STATUS_ERROR row. So a refusal from the INSTRUMENT -- no CUDA and no
    injected fakes, `trials=0`, a warmup that makes the measurement meaningless
    -- became a zeroed row per cell and an exit of 0 (DONE), while
    `cli._main`'s `except TimingRefused` sat one frame up and could never see
    one. The config guards were given an escape hatch when they were written;
    the other door into the same room was not."""
    def refusing_timer(fn, **kw):
        raise T.TimingRefused("trials=0: a measurement needs at least one trial")

    cfg = cfg_for(tmp_path, timer=refusing_timer)
    with pytest.raises(T.TimingRefused, match="at least one trial"):
        D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                      "t_counting_up_gemm")] * 3, cfg,
                    routing=lambda s: None, info=FAKE_INFO)
    assert "[warn]" not in capsys.readouterr().out
    assert not cfg.csv_path.exists() or SC.read_csv(cfg.csv_path) == []


def test_a_kernel_s_own_RuntimeError_is_still_one_cell_s_error(tmp_path):
    """The PASS branch of the same door, and the reason it is a subclass check
    and not a blanket re-raise: a kernel that launched badly IS a per-cell fact,
    the row records it, and the sweep carries on to the next cell."""
    def broken_timer(fn, **kw):
        raise RuntimeError("CUDA error: an illegal memory access")

    cfg = cfg_for(tmp_path, timer=broken_timer)
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                  "t_counting_up_gemm")], cfg, routing=lambda s: None,
                info=FAKE_INFO)
    r = SC.read_csv(cfg.csv_path)[0]
    assert "timing error" in r["notes"] and "illegal memory access" in r["notes"]
    assert float(r["ms_p50"]) == 0.0


# --- v6/v7: the producer's own second sites ----------------------------------
# The LEVEL verdict went two-sided at ONE producer and the driver was taught
# to read the side. These pin every place the driver writes `throttled`,
# `clock_level_ok`, `clock_level_side`, `roof_at_cell_clock_tflops` and
# `pct_of_roof_at_cell_clock` to one rule, which since 2026-09-09 is: DRIFT ->
# throttled, and nothing else; both sides of LEVEL recorded and KEPT, with the
# per-row roof written on every shape of row; and the reference is the row's
# dtype family's GEMM.

import dataclasses  # noqa: E402

from moe.bench.bytes_model import PipelineCost  # noqa: E402

UNDER_LOAD = RF.REFERENCE_UNDER_LOAD
PLANTED_HW = RF.Hardware(
    name="NVIDIA H200 (measured)", bandwidth_bytes_s=4.37e12,
    peak_flops={"fp32": 700e12, "bf16": 700e12, "fp8_e4m3": 1447.7e12},
    source="planted", ceiling_pattern="triad")


def _timing(load, *, level, drift, side="", ref=1515.0):
    end = load if drift else load * 0.9
    return T.KernelTiming(
        ms_p50=1.0, ms_p90=1.1, ms_min=0.9, ms_std=0.01, iters=4, trials=1,
        warmup_ms=300.0, l2_flush=True, sm_clock_load_mhz=load,
        sm_clock_start_mhz=load, sm_clock_end_mhz=end,
        clock_level_ok=level, clock_drift_ok=drift, samples=4, warmup_calls=9,
        flush_mb=8, clock_samples=4, clock_source="injected", clock_poll_ms=0.01,
        host_bound=False, host_enqueue_ms=0.1, clock_level_side=side,
        reference_clock_mhz=ref)


def _timed_row(cfg, kt, dtype="bf16"):
    row = SC.Row(impl="base", model="mixtral-8x7b", dtype=dtype, num_tokens=64,
                 gpu_name="NVIDIA H200")
    D._apply_kernel_timing(row, kt, cfg)
    D._apply_cost(row, PipelineCost(flops=2e11, bytes_total=10**9), 1.0, cfg)
    return row


@pytest.mark.parametrize("name,load,level,drift,side,throttled", [
    ("clean", 1500.0, True, True, "", False),
    ("high", 1980.0, False, True, T.LEVEL_HIGH, False),
    ("low", 1400.0, False, True, T.LEVEL_LOW, False),
    ("drift", 1500.0, True, False, "", True),
])
def test_the_five_fields_agree_on_the_rule_for_every_shape_of_row(
        tmp_path, name, load, level, drift, side, throttled):
    """HIGH, LOW, DRIFT and clean, through `_apply_kernel_timing` then
    `_apply_cost`, against an under-load 1515 reference. The rule since
    2026-09-09: DRIFT is `throttled` and neither side of LEVEL is. A LOW cell
    is a tile holding a lower clock under the same board cap (the H200's
    BM=128/N=64 at 1395 MHz against a GEMM at 1485) and a HIGH cell is a
    memory-shaped one that boosted; both are measured, both carry the fixed
    fraction and the own-clock fraction, and the roof is written on EVERY
    shape. The roof is a number, `throttled` is the verdict, and a consumer
    excludes on the verdict."""
    cfg = cfg_for(tmp_path, hardware=PLANTED_HW,
                  reference_clock_resolver=clock_from(1515.0, grade=UNDER_LOAD))
    assert cfg.reference_clock_grade == UNDER_LOAD
    row = _timed_row(cfg, _timing(load, level=level, drift=drift, side=side))
    assert row.clock_level_ok == SC.verdict_word(level), name
    assert row.clock_level_side == side, name
    assert row.throttled is throttled, name
    assert row.roof_at_cell_clock_tflops == pytest.approx(700.0 * load / 1515), name
    assert row.pct_of_roof_at_cell_clock == pytest.approx(
        100.0 * row.tflops / (700.0 * load / 1515)), name
    # BOTH FRACTIONS ON EVERY ROW, and the note says which one a gate reads.
    assert row.roof_note == RF.ROOF_NOTE_SCORED, name
    assert "fixed roof" in row.roof_note and "gate input" in row.roof_note
    assert row.pct_of_achieved_tflops > 0 and row.pct_of_roof_at_cell_clock > 0
    assert row.reference_clock_mhz == 1515.0
    # And the ratio between the two fractions is exactly the clock ratio.
    assert (row.pct_of_achieved_tflops / row.pct_of_roof_at_cell_clock
            == pytest.approx(load / 1515))


def test_neither_side_of_level_is_throttled_and_the_side_is_still_derived(tmp_path):
    """A record that carries the verdict without the side (every fake before
    v6) still lands on the right side: the driver derives it from the clocks
    by the instrument's own rule. And the side is cleared when LEVEL did not
    fail, so a stale side on a record cannot outlive its verdict. Neither side
    sets `throttled`; only a moving clock does."""
    cfg = cfg_for(tmp_path, hardware=PLANTED_HW,
                  reference_clock_resolver=clock_from(1515.0, grade=UNDER_LOAD))
    high = _timed_row(cfg, _timing(1980.0, level=False, drift=True))
    assert high.clock_level_side == T.LEVEL_HIGH and high.throttled is False
    low = _timed_row(cfg, _timing(1400.0, level=False, drift=True))
    assert low.clock_level_side == T.LEVEL_LOW and low.throttled is False
    passed = _timed_row(cfg, _timing(1500.0, level=True, drift=True, side=T.LEVEL_HIGH))
    assert passed.clock_level_side == "" and passed.throttled is False
    moved = _timed_row(cfg, _timing(1500.0, level=True, drift=False))
    assert moved.clock_level_side == "" and moved.throttled is True


def test_a_v6_row_round_trips_through_the_csv_with_its_side_intact(tmp_path):
    """The wire, end to end: a boosted cell EARNS its HIGH verdict from a
    resolved under-load 1515 through `clock_flags`, is written, and reads back
    with the side, `throttled = False`, the roof at its clock and the
    reference on the row. The LOW row beside it reads back the same way, KEPT
    with its side recorded, since 2026-09-09. These are the rows every ladder
    and gate will see on the H200, where the boost is the normal state of a
    memory-bound cell and the low clock the normal state of a big tile."""
    seen = {}
    for load in (1980.0, 1400.0, 1500.0):
        cfg = cfg_for(tmp_path / str(int(load)), hardware=PLANTED_HW,
                      timer=partial(levelling_timer, load_mhz=load),
                      reference_clock_resolver=clock_from(1515.0, grade=UNDER_LOAD))
        D.run_sweep([(spec(), names_with("t_counting_up_gemm"),
                      "t_counting_up_gemm")], cfg, routing=lambda s: None,
                    info=FAKE_INFO)
        seen[load] = SC.read_csv(cfg.csv_path)[0]
    high, low, clean = seen[1980.0], seen[1400.0], seen[1500.0]
    assert SC.timing_verdict(high, "clock_level_ok") == SC.VERDICT_FAILED
    assert high["clock_level_side"] == T.LEVEL_HIGH
    assert SC.row_bool(high, "throttled") is False              # KEPT
    assert SC.has_cell_clock_roof(high)
    assert float(high["roof_at_cell_clock_tflops"]) == pytest.approx(700 * 1980 / 1515)
    assert float(high["pct_of_roof_at_cell_clock"]) > 0
    assert float(high["reference_clock_mhz"]) == 1515.0
    assert "planted" in high["reference_clock_source"]
    assert SC.timing_verdict(low, "clock_level_ok") == SC.VERDICT_FAILED
    assert low["clock_level_side"] == T.LEVEL_LOW
    assert SC.row_bool(low, "throttled") is False               # KEPT
    assert float(low["roof_at_cell_clock_tflops"]) == pytest.approx(700 * 1400 / 1515)
    assert clean["clock_level_side"] == "" and SC.row_bool(clean, "throttled") is False


def test_an_fp8_cell_is_levelled_and_roofed_against_the_fp8_gemm_s_clock(tmp_path):
    """TWO ROOFS, TWO CLOCKS. The instrument is handed the fp8 GEMM's clock
    for an fp8 cell and the bf16 GEMM's for everything else, from one call
    (`reference_for`) that also writes the reference onto the row. An fp8
    cell at 1905 is AT its roof's clock: level, side "", roof unchanged; the
    one-reference design filed it HIGH against 1515 and would have rescaled a
    peak measured at 1905 by 1905/1515."""
    cfg = cfg_for(
        tmp_path, hardware=PLANTED_HW,
        reference_clock_resolver=clock_from(1515.0, grade=UNDER_LOAD),
        reference_clock_resolvers={RF.FP8_FAMILY: clock_from(
            1905.0, grade=UNDER_LOAD, family=RF.FP8_FAMILY)})
    assert D._instrument_kwargs(cfg, True, "fp8_e4m3")["reference_clock_mhz"] == 1905.0
    assert D._instrument_kwargs(cfg, True, "bf16")["reference_clock_mhz"] == 1515.0
    assert D.unreferenced_clock(cfg, "fp8_e4m3") == []
    level, drift = T.clock_flags(1905.0, 1905.0, 1905.0, 1905.0)
    row = _timed_row(cfg, _timing(1905.0, level=level, drift=drift, ref=1905.0),
                     dtype="fp8_e4m3")
    assert row.clock_level_ok == SC.VERDICT_OK and row.clock_level_side == ""
    assert row.throttled is False
    assert row.reference_clock_mhz == 1905.0
    assert row.roof_at_cell_clock_tflops == pytest.approx(1447.7)
    assert row.roof_at_cell_clock_tflops != pytest.approx(1447.7 * 1905 / 1515)
    assert cfg.missing == {}
    # and a bf16 row on the same config is still against 1515
    bf16 = _timed_row(cfg, _timing(1980.0, level=False, drift=True), dtype="bf16")
    assert bf16.reference_clock_mhz == 1515.0 and bf16.clock_level_side == T.LEVEL_HIGH


def test_an_fp8_cell_on_a_card_with_no_fp8_reference_is_refused_by_name(tmp_path):
    """REFUSE rather than level against the other GEMM. The card is attached,
    the bf16 reference resolved, the fp8 one did not: the bf16 cells measure,
    the fp8 cell is a `ReferenceClockRefused` naming the family, the card and
    the reason, before anything is spent."""
    cfg = cfg_for(
        tmp_path, hardware=PLANTED_HW,
        reference_clock_resolver=clock_from(1515.0, grade=UNDER_LOAD),
        reference_clock_resolvers={RF.FP8_FAMILY: clock_from(
            None, card="NVIDIA H200", family=RF.FP8_FAMILY,
            source="the calibration wrote fp8_gemm_clock_mhz: 0")})
    assert D.unreferenced_clock(cfg, "bf16") == []
    assert D.unreferenced_clock(cfg) == []
    assert D.unreferenced_clock(cfg, "fp8_e4m3") == ["eager"]
    assert cfg.missing == {"reference_clock_mhz[fp8]":
                           "the calibration wrote fp8_gemm_clock_mhz: 0"}
    with pytest.raises(D.ReferenceClockRefused) as e:
        D.refuse_unreferenced_clock(cfg, "fp8_e4m3")
    msg = str(e.value)
    assert "fp8 GEMM" in msg and "fp8_e4m3" in msg and "NVIDIA H200" in msg
    assert "fp8_gemm_clock_mhz: 0" in msg
    assert "reference_clock_resolvers" in msg
    D.refuse_unreferenced_clock(cfg, "bf16")              # the PASS branch
    # A resolver that hands back the OTHER GEMM's clock is no reference
    # either: the family is the point of the number.
    wrong = cfg_for(tmp_path / "wrong", hardware=PLANTED_HW,
                    reference_clock_resolver=clock_from(1515.0, grade=UNDER_LOAD),
                    reference_clock_resolvers={RF.FP8_FAMILY: clock_from(1515.0)})
    assert wrong.reference_for("fp8_e4m3").mhz is None
    assert "another GEMM" in wrong.missing["reference_clock_mhz[fp8]"]
    # And a bf16 entry is not a second primary; it is dropped with the reason.
    two = cfg_for(tmp_path / "two", reference_clock_resolver=clock_from(1515.0),
                  reference_clock_resolvers={RF.BF16_FAMILY: clock_from(1980.0)})
    assert two.reference_clock_mhz == 1515.0
    assert two.reference_clock_resolvers == {}
    assert "second primary" in two.missing["reference_clock_resolvers"]


def test_the_fp8_reference_defaults_to_the_same_card_the_primary_came_from():
    """No injected fp8 resolver: the fp8 GEMM's clock is read from the SAME
    card's calibration as the bf16 one, and only when asked, so a bf16-only
    sweep never records the fp8 GEMM as missing."""
    cfg = D.RunConfig(reference_clock_resolver=clock_from(1515.0))
    assert cfg.family_references == {} and cfg.missing == {}
    fp8 = cfg.reference_for("fp8_e4m3")
    # The fp8 family's reference is the fp8 GEMM's OWN under-load median, 1395
    # MHz at 690 W on the committed calibration. It is a record of where that
    # family runs; nothing is excluded on it.
    assert fp8.mhz == 1395.0 and fp8.family == RF.FP8_FAMILY
    assert fp8.grade == RF.REFERENCE_UNDER_LOAD
    assert cfg.missing == {}
    assert cfg.reference_for("fp8_e5m2") is fp8          # resolved once
    # The A100's calibration has no fp8 GEMM, and says so on first use.
    a100 = D.RunConfig(reference_clock_resolver=clock_from(
        1335.0, card="NVIDIA A100-SXM4-80GB", profile="NVIDIA A100-SXM4-80GB (measured)"))
    assert a100.reference_for("fp8_e4m3").mhz is None
    assert "no fp8 GEMM clock" in a100.missing["reference_clock_mhz[fp8]"]
    # The clock and the roof must come from one file, per family too.
    datasheet = RF.load_hardware("h200_nvl", allow_unverified=True)
    mixed = D.RunConfig(hardware=datasheet, reference_clock_mhz=1515.0,
                        reference_clock_resolvers={RF.FP8_FAMILY: clock_from(
                            1905.0, grade=UNDER_LOAD, family=RF.FP8_FAMILY)})
    assert mixed.reference_for("fp8_e4m3").mhz is None
    assert "different profile" in mixed.missing["reference_clock_mhz[fp8]"]


def test_the_producer_persists_the_under_load_trace_the_watts_and_the_settle(
        tmp_path):
    """v7 AT THE PRODUCER. The instrument computed a first and a last
    under-load sample and a whole sample list, and the driver wrote none of
    them, so when the 2026-09-09 session flagged 135 cells as DRIFT only four
    could be examined. The pod's own settling cell is planted here: 1560 ->
    1650 MHz over three samples at 698 W, DRIFT failed, direction on the row.

    AND NOT INTO THE RETIRED COLUMNS. `sm_clock_start_mhz` holds an IDLE
    instant on 100,144 published rows; the under-load pair has its own names
    and this pins that they stay apart."""
    cfg = cfg_for(tmp_path, hardware=PLANTED_HW,
                  reference_clock_resolver=clock_from(1485.0, grade=UNDER_LOAD))
    kt = dataclasses.replace(
        _timing(1620.0, level=True, drift=False, ref=1485.0),
        sm_clock_start_mhz=1560.0, sm_clock_end_mhz=1650.0,
        clock_samples_mhz=(1560.0, 1620.0, 1650.0),
        clock_drift_direction=T.DRIFT_UP, power_w=698.0,
        settle_ms=50.0, settle_calls=50, clock_settled=True)
    row = _timed_row(cfg, kt)
    assert row.sm_clock_load_first_mhz == 1560.0
    assert row.sm_clock_load_last_mhz == 1650.0
    assert row.clock_samples_mhz == "1560 1620 1650"
    assert row.clock_drift_direction == T.DRIFT_UP
    assert row.power_w == 698.0
    assert row.warmup_settle_ms == 50.0 and row.warmup_settle_calls == 50
    assert row.warmup_clock_settled == SC.VERDICT_OK
    assert row.throttled is True                      # DRIFT, and only DRIFT
    # the retired idle-instant columns are left alone
    assert row.sm_clock_start_mhz == 0 and row.sm_clock_end_mhz == 0
    assert row.clock_drift_pct == 0.0
    # and it all survives the CSV
    with SC.CsvWriter(tmp_path / "v7.csv") as w:
        w.write(row)
    back = SC.read_csv(tmp_path / "v7.csv")[0]
    assert SC.has_load_clock_trace(back)
    assert back["clock_samples_mhz"] == "1560 1620 1650"
    assert SC.row_float(back, "power_w") == 698.0


def test_a_row_off_a_host_with_no_clock_reader_says_so_rather_than_settling(
        tmp_path):
    """No NVML, no settle loop, and `warmup_clock_settled` is undetermined --
    which is not "it settled at once". The row is still measured."""
    cfg = cfg_for(tmp_path, hardware=PLANTED_HW,
                  reference_clock_resolver=clock_from(1485.0, grade=UNDER_LOAD))
    row = _timed_row(cfg, _timing(1485.0, level=True, drift=True, ref=1485.0))
    assert row.warmup_clock_settled == SC.VERDICT_UNDETERMINED
    assert row.warmup_settle_ms == 0.0 and row.warmup_settle_calls == 0
    assert row.clock_samples_mhz == "" and row.power_w == 0.0
    assert row.throttled is False
