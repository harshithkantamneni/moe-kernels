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


def test_flush_barely_moves_a_dram_bound_kernel():
    """With a working set far larger than L2, nothing was cached, so evicting
    the cache should not materially change the time.

    This is the sound direction of the flush claim. The opposite direction (a
    cache-resident kernel getting SLOWER when flushed) is confounded at small
    sizes: see test_flush_can_appear_to_speed_up_a_tiny_kernel.
    """
    n = 512 * 1024 * 1024 // 4          # 512 MB, ~8x L2
    a = torch.empty(n, device="cuda", dtype=torch.float32).fill_(1.0)
    out = torch.empty_like(a)
    warm = T.time_eager(lambda: torch.mul(a, 1.0001, out=out), warmup=10,
                        iters=20, trials=2, l2_flush=False)
    cold = T.time_eager(lambda: torch.mul(a, 1.0001, out=out), warmup=10,
                        iters=20, trials=2, l2_flush=True,
                        flush_mb=T.DEFAULT_FLUSH_MB)
    ratio = cold.ms_p50 / warm.ms_p50
    assert 0.7 < ratio < 1.4, (
        f"flush changed a DRAM-bound kernel by {ratio:.2f}x "
        f"(warm {warm.ms_p50:.4f} ms, cold {cold.ms_p50:.4f} ms); nothing was "
        "cached, so this should be near 1.0")


def test_flush_can_appear_to_speed_up_a_tiny_kernel():
    """A real measurement hazard, pinned so it is not rediscovered as a bug.

    Measured on H200 SXM: a 4 MB kernel ran at 13.70 us unflushed and 7.07 us
    flushed. The flush made it FASTER, which no cache effect explains. The 128 MB
    flush is enough work to hold clocks up, while a loop of ~10 us kernels lets
    the GPU settle between iterations.

    Consequence for this harness: at microsecond scale the l2_flush axis is
    confounded with clock state, and that is exactly the small-batch decode
    regime the project targets. Every row records sm_clock_start/end per timing
    mode, so the confound is detectable in the data rather than invisible.

    The assertion is deliberately loose: the point is that the direction is NOT
    reliably "flushed is slower", so no analysis may assume it.
    """
    n = 1024 * 1024                      # 4 MB, comfortably L2-resident
    a = torch.empty(n, device="cuda", dtype=torch.float32).fill_(1.0)
    out = torch.empty_like(a)
    warm = T.time_eager(lambda: torch.mul(a, 1.0001, out=out), warmup=10,
                        iters=50, trials=2, l2_flush=False)
    cold = T.time_eager(lambda: torch.mul(a, 1.0001, out=out), warmup=10,
                        iters=50, trials=2, l2_flush=True,
                        flush_mb=T.DEFAULT_FLUSH_MB)
    assert warm.ms_p50 > 0 and cold.ms_p50 > 0
    assert warm.ms_p50 < 0.5 and cold.ms_p50 < 0.5, "expected microsecond scale"


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
    twice; the row must say so rather than silently vanishing.

    The policy triggers on PREDICTED time, so a toy cell on real H200 bandwidth
    is predicted to be microseconds and the policy correctly says "measure it".
    A deliberately slow hardware profile is what forces the skip branch.
    """
    from moe.bench.roofline import Hardware
    slow = Hardware(name="slow", bandwidth_bytes_s=1.0,
                    peak_flops={"bf16": 1e12, "fp32": 1e12}, source="test")
    cfg = make_cfg(tmp_path, graph_modes=(True,), graph_min_launch_share=0.5,
                   hardware=slow)
    D.run_sweep([(toy(), NAMES, "gpu_ref_up_gemm")], cfg, routing=lambda s: None)
    r = SC.read_csv(cfg.csv_path)[0]
    assert r["capture_status"] == "skipped"
    assert "threshold" in r["graph_skip_reason"]


def test_graph_policy_measures_a_short_kernel(tmp_path):
    """The other side: on real bandwidth a toy cell IS launch-dominated, so the
    policy must not skip it."""
    cfg = make_cfg(tmp_path, graph_modes=(True,),
                   graph_min_launch_share=0.01,
                   hardware=device_reference())
    D.run_sweep([(toy(), NAMES, "gpu_ref_up_gemm")], cfg, routing=lambda s: None)
    r = SC.read_csv(cfg.csv_path)[0]
    # The reference pipeline is not capturable, which is itself the finding.
    assert r["capture_status"] == "not_capturable"
    assert r["graph_skip_reason"] == ""


def test_both_timing_modes_produce_comparable_rows(tmp_path):
    cfg = make_cfg(tmp_path, l2_modes=(True, False))
    D.run_sweep([(toy(), NAMES, "gpu_ref_up_gemm")], cfg, routing=lambda s: None)
    rows = SC.read_csv(cfg.csv_path)
    assert len(rows) == 2
    assert {r["l2_flush"] for r in rows} == {"True", "False"}
    assert all(float(r["ms_p50"]) > 0 for r in rows)


# --- hardware calibration: the stand-in for unavailable counters -------------

#: The patterns every card must produce. `read_stream` is NOT here: it needs
#: Triton, and a machine without it records a refusal instead, which is a
#: legitimate result and not a failed measurement.
ALWAYS_MEASURED = {"read_reduce", "copy", "triad", "write"}


@pytest.mark.slow
def test_bandwidth_measurement_is_plausible():
    from moe.bench.calibrate import measure_bandwidth
    refusals = []
    results = measure_bandwidth(target_bytes=1 << 30, warmup=3, iters=10,
                                trials=2, refusals=refusals)
    names = {r.pattern for r in results}
    assert ALWAYS_MEASURED <= names
    # The optional one is present OR refused in writing. Neither absent-and-
    # silent nor present-as-a-zero is allowed.
    assert ("read_stream" in names) ^ any("read_stream" in r for r in refusals)
    for r in results:
        assert r.gbps > 50, f"{r.pattern} measured {r.gbps:.1f} GB/s, implausible"
        assert 0 < r.ms_p50 and r.ms_min <= r.ms_p50


def device_reference():
    """Ceilings for the card actually attached, or skip.

    These tests used `load_hardware("h200_sxm")` as their reference, which is a
    sanity bound against a DIFFERENT GPU. On an A100 that asks a card with a
    2447 GB/s pin rate to clear 40% of an H200's 4800, and the suite fails on
    hardware that is working perfectly. Since run_all.sh stops the session on a
    failing test, that blocked an entire sweep.
    """
    from moe.bench.roofline import load_measured
    hw = load_measured()
    if hw is None:
        pytest.skip("no measured calibration for this device; "
                    "run scripts/calibrate_hardware.py first")
    return hw

@pytest.mark.slow
@pytest.mark.parametrize("pattern", sorted(("read_reduce", "read_stream")))
def test_read_bandwidth_is_measured_and_not_optimised_away(pattern):
    """Every read pattern moves the bytes it claims to.

    Both formulations exist to consume a stream while writing almost nothing,
    which is exactly the shape a compiler can delete. `read_stream` is the more
    exposed of the two: its whole design is that nothing but one float per
    program leaves the kernel.
    """
    from moe.bench.calibrate import measure_bandwidth
    results = measure_bandwidth(target_bytes=1 << 30, warmup=3, iters=10,
                                trials=2)
    read = next((r for r in results if r.pattern == pattern), None)
    if read is None:
        pytest.skip(f"{pattern} was refused on this machine")
    peak = device_reference().bandwidth_bytes_s / 1e9
    # A pure read legitimately exceeds the triad ceiling: 1N against 2R+1W.
    # With the read measured on a 2-D view it reaches ~4472 where triad is
    # 4377, so a 1.02 bound fails on correct hardware. What this guards against
    # is the loads being elided entirely, which gives a figure several times
    # too high, so the bound is generous on purpose.
    assert read.gbps < peak * 2.0, (
        f"{pattern} measured {read.gbps:.0f} GB/s against a {peak:.0f} ceiling; "
        "the loads were probably optimised away")
    assert read.gbps > peak * 0.4


@pytest.mark.slow
def test_calibration_names_its_ceiling_and_records_every_pattern():
    """max() across patterns reported whichever pattern the hardware liked best,
    which is a property of the benchmark. The choice must be explicit and the
    alternatives kept."""
    from moe.bench.calibrate import calibrate
    cal = calibrate(target_bytes=1 << 30, gemm_n=2048, ceiling="triad",
                    settle=False)
    assert cal.ceiling_pattern == "triad"
    assert cal.achieved_bandwidth_gbps == cal.pattern("triad").gbps
    names = {p.pattern for p in cal.bandwidth_patterns}
    assert ALWAYS_MEASURED <= names <= (ALWAYS_MEASURED | {"read_stream"})
    # the ridge moves with the denominator, so every one is recorded
    ridges = cal.as_dict()["ridge_by_pattern"]
    assert set(ridges) == names
    assert all(v > 0 for v in ridges.values())
    # And the band, because a single ridge is the shape of the whole problem:
    # triad and the matched read ruler disagree by 2.2% on this card, which is
    # a whole tile tread at the crossings this study quotes.
    band = cal.ridge_band()
    assert band is None or band[0] < band[1]


@pytest.mark.slow
def test_an_unknown_ceiling_is_rejected():
    from moe.bench.calibrate import calibrate
    with pytest.raises(ValueError, match="unknown ceiling"):
        calibrate(target_bytes=1 << 28, gemm_n=1024, ceiling="nonsense",
                  settle=False)


@pytest.mark.slow
def test_settling_reaches_a_stable_clock():
    """A ceiling measured while the clock is still climbing is not a ceiling.
    Measured on this box: idle 840 MHz, boost 1980 MHz, and a calibration
    started cold walked the whole ramp across its four patterns."""
    from moe.bench.calibrate import settle_clocks
    info = settle_clocks(max_seconds=25.0)
    assert info["clock_history_mhz"], "no clock samples taken"
    assert info["final_mhz"] > 0
    if info["settled"]:
        recent = info["clock_history_mhz"][-3:]
        assert (max(recent) - min(recent)) / max(recent) * 100 <= 2.0, recent


@pytest.mark.slow
def test_the_gemm_clock_is_sampled_while_the_gemm_is_running():
    """The failure this replaces: one sample taken AFTER `time_eager` had
    synchronised, i.e. with the GPU idle and boosting back up. Across the eleven
    committed H200 calibrations that field ran 1485-1935 MHz on one card while
    the achieved rate moved 12%, and `gemm_efficiency_pct` published 87.4% and
    68.4% for the same kernel.

    Asserted here rather than only in the report because the wrong number was
    plausible: 1935 MHz is a real clock this part can reach, just not while it
    is saturating its tensor cores.
    """
    from moe.bench.calibrate import measure_bf16_gemm
    gemm = measure_bf16_gemm(n=2048, warmup=3, iters=10, trials=2)
    assert gemm.clock is not None, "no clock established for the GEMM"
    assert len(gemm.clock.samples) >= 3, "two samples cannot disagree"
    assert all(s > 0 for s in gemm.clock.samples), "a zero is not a clock"
    assert gemm.sm_clock_mhz == gemm.clock.median_mhz
    assert gemm.clock.after_idle_mhz > 0, (
        "the post-hoc sample is kept on purpose: the gap between it and the "
        "median is the size of the artefact")


@pytest.mark.slow
def test_a_clock_cannot_be_established_without_nvml(monkeypatch):
    """A typed refusal, not a zero. `ClockState.sample()` returns 0 MHz when the
    container forbids NVML, and a zero reaching `sustained_peak_tflops` makes
    the silicon's peak zero and the efficiency infinite, in a yaml where nothing
    looks wrong."""
    import moe.bench.timing as T
    from moe.bench.calibrate import ClockUnavailable, clock_under_load

    monkeypatch.setattr(T.ClockState, "sample",
                        classmethod(lambda cls: T.ClockState(0, 0)))
    with pytest.raises(ClockUnavailable, match="usable SM clock samples"):
        clock_under_load(lambda: None, "probe", samples=3,
                         seconds_per_sample=0.01)


@pytest.mark.slow
def test_calibration_records_per_pattern_clocks():
    """Each pattern carries the clock it was measured at, so a ramped run is
    detectable in the recorded data rather than only in a summary line."""
    from moe.bench.calibrate import calibrate
    cal = calibrate(target_bytes=1 << 29, gemm_n=1024, settle=False)
    for pat in cal.bandwidth_patterns:
        assert pat.sm_clock_start_mhz >= 0 and pat.sm_clock_end_mhz >= 0
    assert isinstance(cal.clock_ramped, bool)


@pytest.mark.slow
def test_measured_bandwidth_does_not_exceed_the_datasheet_peak():
    """A measured ceiling above the spec peak means the buffer fit in cache and
    the number is not a DRAM measurement at all."""
    from moe.bench.calibrate import measure_bandwidth
    peak_gbps = device_reference().bandwidth_bytes_s / 1e9
    best = max(r.gbps for r in measure_bandwidth(target_bytes=1 << 30,
                                                 warmup=3, iters=10, trials=2))
    # `best` is the max across ALL FOUR patterns and the reference is the named
    # ceiling, which is triad. Write legitimately exceeds triad: stores are
    # posted and on an A100 write reaches 1882 against triad's 1798. A 1.02
    # bound therefore failed on a card that was working perfectly.
    #
    # The failure this is for is cache residency, where the buffer never left
    # L2 and the figure comes out several times the real rate, not 5% over. A
    # generous bound catches that and does not fire on a legitimate pattern
    # ordering.
    assert best < peak_gbps * 2.0, (
        f"measured {best:.0f} GB/s against a {peak_gbps:.0f} GB/s ceiling; "
        "the working set was probably cache resident")
    assert best > peak_gbps * 0.4, (
        f"measured only {best:.0f} of {peak_gbps:.0f} GB/s; something is wrong")


@pytest.mark.slow
def test_bf16_gemm_ceiling_is_plausible():
    from moe.bench.calibrate import measure_bf16_gemm, sustained_peak_tflops
    gemm = measure_bf16_gemm(n=4096, warmup=3, iters=10, trials=2)
    assert gemm.shape == (4096, 4096, 4096)
    assert gemm.sm_clock_mhz >= 0

    # The bound is the SILICON at the clock this measurement ran at, not another
    # measurement taken at a different shape and clock. Measured on an H200: a
    # 4096 GEMM runs at 1980 MHz and reaches 782.9 TFLOP/s, while the
    # calibration's 712.4 came from an 8192 GEMM at 1845 MHz. Comparing them as
    # a hard ceiling failed on a card that was working perfectly.
    silicon = sustained_peak_tflops(gemm.sm_clock_mhz)
    if silicon is None:
        pytest.skip("no FLOP/SM/clk constant for this architecture")
    assert 0 < gemm.tflops < silicon * 1.02, (
        f"{gemm.tflops:.1f} TFLOP/s at {gemm.sm_clock_mhz} MHz, where the "
        f"silicon ceiling is {silicon:.1f}")
    assert gemm.tflops > silicon * 0.3, (
        f"only {gemm.tflops:.1f} of {silicon:.1f} TFLOP/s; something is wrong")


@pytest.mark.slow
def test_full_calibration_runs(tmp_path):
    from moe.bench.calibrate import calibrate
    cal = calibrate(target_bytes=1 << 30, gemm_n=2048, settle=False)
    assert cal.achieved_bandwidth_gbps > 0
    assert cal.achieved_bf16_tflops > 0
    assert cal.gpu_name
    assert len(cal.bandwidth_patterns) == 4
    assert cal.ridge_point() > 0


@pytest.mark.slow
def test_compute_ceiling_is_normalised_to_the_clock_it_was_measured_at():
    """The datasheet peak assumes a boost clock the part cannot hold under
    sustained dense tensor load. Measured on H200 SXM: it settles near 1455 MHz,
    where the silicon can do ~787 TFLOP/s, and cuBLAS reaches ~90% of that.
    Against the 989.5 datasheet the same result reads 71.5%, and the difference
    is the clock rather than the library."""
    from moe.bench.calibrate import calibrate, sustained_peak_tflops
    cal = calibrate(target_bytes=1 << 29, gemm_n=4096, settle=False)
    assert cal.gemm_clock_mhz > 0, "the GEMM's clock must be recorded"
    peak = cal.sustained_peak_tflops
    assert peak and peak > 0
    assert peak == pytest.approx(sustained_peak_tflops(cal.gemm_clock_mhz))
    # cuBLAS cannot exceed what the clock allows, and should be respectably close.
    assert cal.achieved_bf16_tflops < peak * 1.02
    assert cal.gemm_efficiency_pct > 50.0


def test_the_datasheet_clock_is_reproducible_from_first_principles():
    """132 SM x 4096 dense BF16 FLOP/SM/clk x 1830 MHz = 989.4 TFLOP/s, which is
    NVIDIA's published H200 SXM figure. That is what pins the FLOP/SM/clk
    constant and reveals the boost clock the datasheet assumes."""
    import torch

    from moe.bench.calibrate import _DENSE_BF16_FLOP_PER_SM_CLK

    # (boost MHz, published dense BF16 TFLOP/s) for the part this test knows.
    # The 1830 and the 989.5 are BOTH H200 SXM figures, so running this on any
    # other device compares that card's silicon to Hopper's headline: an A100
    # gives 108 x 2048 x 1830 MHz = 404.8 against an expected 989.5.
    #
    # Adding an sm_80 constant is what exposed it, because the `per_clk is None`
    # skip below used to catch every non-Hopper card by accident rather than by
    # intent. The device-independent version of this check, covering every entry
    # in the table against its own vendor figure, is
    # test_calibrate_settle.py::test_the_dense_bf16_rate_reproduces_each_vendor_headline.
    DATASHEET = {(9, 0): (1830e6, 989.5)}

    props = torch.cuda.get_device_properties(0)
    cap = (props.major, props.minor)
    per_clk = _DENSE_BF16_FLOP_PER_SM_CLK.get(cap)
    if per_clk is None or cap not in DATASHEET:
        pytest.skip(f"no datasheet figure recorded here for sm_{cap[0]}{cap[1]}")
    boost, _published = DATASHEET[cap]
    implied = props.multi_processor_count * per_clk * boost / 1e12
    assert implied == pytest.approx(989.5, abs=1.0)
