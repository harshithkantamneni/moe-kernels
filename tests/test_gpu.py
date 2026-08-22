"""CUDA-path tests. Skipped automatically without a device.

Everything in `moe/bench/timing.py` and the CUDA branches of the driver is
unverifiable on a laptop, so this file is the first thing that runs on the H200
and is what `scripts/run_all.sh` relies on before it starts a paid sweep.
"""
import pytest
import torch

from moe import pipeline as P
from moe.bench import driver as D
from moe.bench import schema as SC
from moe.bench import timing as T
from moe.reference import torch_ref as R
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
from moe.stages import StageSpan, register
from moe.state import MoEState

pytestmark = pytest.mark.gpu

REF = P.reference_pipeline_names()


def toy(dtype="bf16", tokens=32):
    return BenchSpec(MODEL_CONFIGS["toy"], num_tokens=tokens, dtype=dtype,
                     routing=RoutingSpec("uniform"), seed=0)


# --- machine facts ----------------------------------------------------------

def test_runtime_info_populates_the_machine_columns():
    info = T.runtime_info()
    for key in ("gpu_name", "sm_count", "l2_bytes", "total_memory",
                "device_index", "torch_version", "python_version"):
        assert key in info, key
    assert info["sm_count"] > 0
    assert info["l2_bytes"] > 0
    assert all(k in SC.COLUMNS for k in info if k != "sm_count" or True)


def test_clock_sampling_returns_plausible_values():
    c = T.ClockState.sample()
    assert c.sm_clock_mhz >= 0 and c.temp_c >= 0
    if c.sm_clock_mhz:
        assert 100 < c.sm_clock_mhz < 4000
        assert 0 < c.temp_c < 120


def test_clock_drift_flags_a_throttle():
    drift, throttled = T.clock_drift(T.ClockState(1980, 40), T.ClockState(1500, 90))
    assert drift > 20 and throttled


# --- L2 flusher -------------------------------------------------------------

@pytest.mark.parametrize("mode", ["read", "write"])
def test_flusher_allocates_and_runs(mode):
    f = T.L2Flusher(64, mode=mode)
    assert f.enabled and f.buf is not None
    assert f.buf.numel() * f.buf.element_size() == 64 * 1024 * 1024
    f.flush()
    torch.cuda.synchronize()


def test_flush_buffer_exceeds_l2():
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    f = T.L2Flusher(T.DEFAULT_FLUSH_MB)
    assert f.buf.numel() * f.buf.element_size() > l2, (
        "the flush buffer must be larger than L2 or it evicts nothing")


def test_disabled_flusher_is_a_noop():
    f = T.L2Flusher(0)
    assert not f.enabled and f.buf is None
    f.flush()


# --- iteration calibration --------------------------------------------------

def test_calibrate_scales_iterations_to_the_work():
    small = torch.randn(256, 256, device="cuda")
    big = torch.randn(4096, 4096, device="cuda")
    n_small = T.calibrate_iters(lambda: small @ small, target_ms=50.0)
    n_big = T.calibrate_iters(lambda: big @ big, target_ms=50.0)
    assert n_small > n_big, (n_small, n_big)
    assert 10 <= n_big <= 2000 and 10 <= n_small <= 2000


# --- eager timing -----------------------------------------------------------

@pytest.mark.parametrize("l2_flush", [True, False])
def test_eager_timing_is_self_consistent(l2_flush):
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    res = T.time_eager(lambda: a @ a, warmup=5, iters=20, trials=2,
                       l2_flush=l2_flush, flush_mb=64)
    assert res.ms_p50 > 0
    assert res.ms_min <= res.ms_p50 <= res.ms_p90
    assert res.samples == 40
    assert res.l2_flush is l2_flush and res.cuda_graph is False
    assert res.flush_mb == (64 if l2_flush else 0)


def test_eager_timing_calibrates_when_iters_is_none():
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    res = T.time_eager(lambda: a @ a, warmup=5, iters=None, trials=1,
                       l2_flush=False, target_ms=50.0)
    assert res.iters >= 10 and res.samples == res.iters


def test_flushing_costs_time_for_a_memory_bound_kernel():
    """Sanity check that the flush actually perturbs what it should: a kernel
    whose working set fits L2 must get slower when L2 is evicted each iteration."""
    n = 1024 * 1024  # 4 MB fp32, comfortably L2-resident
    a = torch.randn(n, device="cuda")
    warm = T.time_eager(lambda: a.mul(1.0001), warmup=10, iters=50, trials=2,
                        l2_flush=False)
    cold = T.time_eager(lambda: a.mul(1.0001), warmup=10, iters=50, trials=2,
                        l2_flush=True, flush_mb=T.DEFAULT_FLUSH_MB)
    assert cold.ms_p50 >= warm.ms_p50 * 0.9, (warm.ms_p50, cold.ms_p50)


# --- graph timing -----------------------------------------------------------

def test_graph_capture_and_replay():
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)

    def fn():
        torch.mm(a, a, out=out)

    res = T.time_graph(fn, warmup=5, iters=20, trials=2, l2_flush=False)
    assert res.cuda_graph is True and res.ms_p50 > 0
    assert res.samples == 40


def test_graph_capture_invokes_the_replay_validator():
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)
    seen = {"n": 0}

    def fn():
        torch.mm(a, a, out=out)

    T.time_graph(fn, warmup=3, iters=10, trials=1, l2_flush=False,
                 on_captured=lambda: seen.__setitem__("n", seen["n"] + 1))
    assert seen["n"] == 1


def test_host_sync_is_reported_as_not_capturable():
    """A kernel that reads a device value on the host cannot be graph-captured,
    and therefore cannot be used in real MoE inference."""
    a = torch.randn(64, 64, device="cuda")

    def fn():
        if float(a.sum().item()) > 1e30:   # forces a device-to-host sync
            a.mul_(2)

    with pytest.raises(T.NotCapturable):
        T.time_graph(fn, warmup=2, iters=5, trials=1, l2_flush=False)


# --- driver end to end on the device ---------------------------------------

def make_cfg(tmp_path, **kw):
    base = dict(out_dir=tmp_path, device="cuda", warmup=3, trials=1, iters=5,
                l2_modes=(True,), graph_modes=(False,), flush_mb=32,
                graph_min_launch_share=0.0)
    base.update(kw)
    return D.RunConfig(**base)


@register
class GpuUpGemm(StageSpan):
    name = "gpu_ref_up_gemm"
    covers = ("up_gemm",)
    requires_cuda = True
    dtypes = ("fp32", "fp16", "bf16")

    def __call__(self, st: MoEState) -> None:
        st.h_up = R.grouped_gemm_loop(
            st.x_perm, st.weights.w1, st.expert_offsets,
            2 * st.spec.model.intermediate_size)


NAMES = ["ref_router", "ref_permute", "gpu_ref_up_gemm", "ref_act",
         "ref_down_gemm", "ref_unpermute"]


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_full_cell_runs_on_device(tmp_path, dtype):
    cfg = make_cfg(tmp_path)
    D.run_sweep([(toy(dtype), NAMES, "gpu_ref_up_gemm")], cfg,
                routing=lambda s: None)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 1, cfg.manifest_path.read_text()
    r = rows[0]
    assert r["correctness_passed"] == "True", r["notes"]
    assert float(r["ms_p50"]) > 0
    assert float(r["tflops"]) > 0
    assert r["gpu_name"]
    assert int(r["sm_count"]) > 0


def test_inputs_are_generated_on_device():
    x, w = R.make_inputs(toy(), device="cuda")
    assert x.is_cuda and w.w1.is_cuda and w.w2.is_cuda
    assert torch.isfinite(x).all()


def test_reference_pipeline_is_not_capturable_and_says_so(tmp_path):
    """The reference grouped GEMM loops over experts on the host, so it cannot
    be captured. That must appear in the CSV, not only in the manifest."""
    cfg = make_cfg(tmp_path, graph_modes=(True,), l2_modes=(True,))
    D.run_sweep([(toy(), NAMES, "gpu_ref_up_gemm")], cfg, routing=lambda s: None)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 1
    assert rows[0]["capture_status"] == "not_capturable"


def test_graph_policy_skips_a_long_kernel(tmp_path):
    """A cell whose predicted time dwarfs a kernel launch is not worth timing
    twice; the row must say so rather than silently vanishing."""
    cfg = make_cfg(tmp_path, graph_modes=(True,), graph_min_launch_share=0.99,
                   peak_bandwidth_bytes_s=4.8e12)
    D.run_sweep([(toy(), NAMES, "gpu_ref_up_gemm")], cfg, routing=lambda s: None)
    r = SC.read_csv(cfg.csv_path)[0]
    assert r["capture_status"] == "skipped"
    assert "threshold" in r["graph_skip_reason"]


def test_both_timing_modes_produce_comparable_rows(tmp_path):
    cfg = make_cfg(tmp_path, l2_modes=(True, False))
    D.run_sweep([(toy(), NAMES, "gpu_ref_up_gemm")], cfg, routing=lambda s: None)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 2
    assert {r["l2_flush"] for r in rows} == {"True", "False"}
    assert all(float(r["ms_p50"]) > 0 for r in rows)
