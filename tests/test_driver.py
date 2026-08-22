"""Driver control-flow tests, run on CPU with an injected timing backend.

Everything the driver promises (correctness gates timing, the timer wraps the
span and not the layer, resume skips finished work, one bad cell does not kill
a sweep) is verified here before any of it costs GPU minutes.
"""
import itertools

import pytest
import torch

from moe import pipeline as P
from moe.bench import bytes_model as BM
from moe.bench import driver as D
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


def fake_timer(fn, warmup=1, iters=2, trials=1, l2_flush=True, flush_mb=8,
               flush_mode="read", target_ms=200.0, on_captured=None,
               graph=False):
    for _ in range(3):
        fn()
    if on_captured is not None:
        on_captured()
    return T.TimingResult(ms_p50=1.0, ms_p90=1.2, ms_min=0.9, ms_std=0.05,
                          jitter_p90_over_p50=1.2, warmup=warmup, iters=iters or 2,
                          trials=trials, l2_flush=l2_flush, cuda_graph=graph,
                          samples=3, flush_mb=flush_mb, flush_mode=flush_mode)


def fake_graph_timer(fn, **kw):
    kw.pop("graph", None)
    return fake_timer(fn, graph=True, **kw)


def not_capturable(fn, **kw):
    raise T.NotCapturable("host sync during capture")


FAKE_INFO = {"gpu_name": "FakeH200", "gpu_count": 1, "torch_version": "x",
             "driver_version": "y", "cuda_version": "z", "triton_version": "w"}


def cfg_for(tmp_path, **kw):
    base = dict(
        out_dir=tmp_path, device="cpu", warmup=1, trials=1, iters=2,
        l2_modes=(True,), graph_modes=(False,),
        timer_eager=fake_timer,
        timer_graph=fake_graph_timer,
        clock_sampler=lambda: T.ClockState(1980, 45),
    )
    base.update(kw)
    return D.RunConfig(**base)


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
    cfg = cfg_for(tmp_path, graph_modes=(True,), timer_graph=not_capturable)
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


def test_clock_drift_is_recorded(tmp_path):
    states = itertools.cycle([T.ClockState(1980, 40), T.ClockState(1600, 84)])
    cfg = cfg_for(tmp_path, clock_sampler=lambda: next(states))
    D.run_sweep([(spec(), names_with("t_counting_up_gemm"), "t_counting_up_gemm")],
                cfg, routing=lambda s: None, info=FAKE_INFO)
    r = SC.read_csv(cfg.csv_path)[0]
    assert r["throttled"] == "True"
    assert float(r["clock_drift_pct"]) > 5.0


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
    assert "derived from topk_ids" in r["notes"]


def test_graph_row_revalidates_the_replayed_output(tmp_path):
    """A graph row must earn its own correctness verdict against the replayed
    output, not inherit the eager one."""
    seen = {"verified": 0}

    def counting_graph_timer(fn, on_captured=None, **kw):
        if on_captured is not None:
            seen["verified"] += 1
        return fake_graph_timer(fn, on_captured=on_captured, **kw)

    cfg = cfg_for(tmp_path, graph_modes=(True,), timer_graph=counting_graph_timer)
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

    cfg = cfg_for(tmp_path, timer_eager=flaky)
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


def test_graph_mode_is_skipped_when_launch_overhead_cannot_matter():
    """At DeepSeek geometry with a few tokens the roofline minimum is ~0.8 ms
    against a ~5 us launch. Timing that cell twice spends half a session
    measuring a sub-1% effect."""
    cfg = D.RunConfig(peak_bandwidth_bytes_s=4.8e12, graph_min_launch_share=0.01)
    ok, reason = D.should_time_graph(cost_for(3.76e9), cfg)   # ~0.78 ms
    assert not ok
    assert "0.6" in reason or "below" in reason


def test_graph_mode_is_kept_when_launch_overhead_is_first_order():
    cfg = D.RunConfig(peak_bandwidth_bytes_s=4.8e12, graph_min_launch_share=0.01)
    ok, reason = D.should_time_graph(cost_for(4.8e5), cfg)     # ~0.0001 ms
    assert ok and reason == ""


def test_policy_can_be_disabled():
    cfg = D.RunConfig(peak_bandwidth_bytes_s=4.8e12, graph_min_launch_share=0.0)
    assert D.should_time_graph(cost_for(1e12), cfg)[0] is True


def test_policy_errs_toward_measuring_when_bandwidth_is_unknown():
    cfg = D.RunConfig(peak_bandwidth_bytes_s=None, graph_min_launch_share=0.01)
    D._BANDWIDTH_CACHE["bw"] = None
    try:
        assert D.should_time_graph(cost_for(1e12), cfg)[0] is True
    finally:
        D._BANDWIDTH_CACHE.pop("bw", None)


def test_skipped_graph_row_is_written_not_dropped(tmp_path):
    cfg = cfg_for(tmp_path, graph_modes=(True,), l2_modes=(True,),
                  peak_bandwidth_bytes_s=1.0, graph_min_launch_share=0.5)
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
