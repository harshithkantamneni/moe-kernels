"""Measure what this machine can actually do, since we cannot read its counters.

Nsight Compute needs GPU performance counters, which need a host-level
`NVreg_RestrictProfilingToAdminUsers=0` that a container tenant cannot set. On a
rented pod `ncu` fails with ERR_NVGPUCTRPERM, so DRAM traffic cannot be measured
directly.

What can be measured without any counter access is the machine's *achievable*
ceilings, using ordinary kernels and a clock. That turns the roofline from a
datasheet claim into a measured one. 92% of achievable bandwidth is a statement
about the kernel; 92% of a spec peak also carries whatever gap the part was
never going to close, and afterwards the two cannot be told apart.

WHICH BANDWIDTH NUMBER IS "THE" BANDWIDTH
-----------------------------------------
There isn't one. Read, write, copy and triad all measure real bandwidth and all
give different answers, because reads, writes and mixed traffic hit DRAM
differently. An earlier version of this file took `max()` across patterns. That
reports whichever pattern the hardware happens to like best, which is a property
of the benchmark rather than of the workload.

So every pattern is measured and recorded, and one is *named* as the ceiling
with the reason written down. The default is `triad`, the canonical STREAM
metric: mixed read/write traffic, widely published, and therefore comparable to
other people's numbers. Because all four are recorded, anyone (including you)
can recompute a roofline against a different denominator without re-running.

For a Mixture-of-Experts grouped GEMM at small batch, traffic is roughly 95%
streaming reads of expert weights, so a read-dominated denominator would be most
representative of that regime. But `read` here is a torch.sum tree reduction,
which reports ATen's reduction rate rather than the DRAM read rate, so it is
measured and recorded and guarded against ever becoming the ceiling.

HOW MUCH DOES THE CHOICE ACTUALLY MATTER
----------------------------------------
Across PROFILES['standard'] (105 specs) and PROFILES['full'] (882), zero cells
change memory/compute classification anywhere in the 4252.8-4656.9 GB/s range
measured on an H200 SXM: the highest memory-bound intensity is 162.6, the lowest
compute-bound one is 224.9, and every candidate ridge (170.8-187.0) falls in
that empty interval.

The honest caveat: the interval is empty because the token grid steps by 2x
straight through the ridge band, NOT because MoE arithmetic intensity is
bimodal. Off-grid runs (`--tokens 640,768,1536`) land squarely in it, and
nothing constrains the grid. The choice is real; the shipped sweeps happen not
to exercise it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from . import timing as T

#: Pattern whose figure becomes `achieved_bandwidth_gbps` unless overridden.
DEFAULT_CEILING = "triad"

#: Buffers must dwarf L2 (60 MiB on H200) and run long enough that launch and
#: clock ramp are not a meaningful fraction of the measurement.
DEFAULT_BUFFER_BYTES = 8 << 30


@dataclass(frozen=True)
class BandwidthResult:
    pattern: str
    bytes_moved: int
    ms_p50: float
    ms_min: float
    gbps: float
    gbps_peak_min: float      # from ms_min: the optimistic bound
    note: str = ""
    #: SM clock either side of THIS pattern. A ceiling measured while the clock
    #: is still ramping is not a ceiling.
    sm_clock_start_mhz: int = 0
    sm_clock_end_mhz: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Calibration:
    gpu_name: str
    achieved_bandwidth_gbps: float
    ceiling_pattern: str
    achieved_bf16_tflops: float
    bandwidth_patterns: tuple[BandwidthResult, ...]
    gemm_shape: tuple[int, int, int]
    gemm_clock_mhz: int = 0
    buffer_bytes: int = 0
    clocks: dict = field(default_factory=dict)
    settle: dict = field(default_factory=dict)
    #: The discarded first bandwidth pass. Kept because the gap between it and
    #: the reported pass is the size of the warm-up effect, and a large gap
    #: means the settle did not do its job.
    warmup_pass: dict = field(default_factory=dict)

    @property
    def warmup_drift_pct(self) -> float:
        """Largest per-pattern change between the discarded pass and this one."""
        if not self.warmup_pass:
            return 0.0
        worst = 0.0
        for pat in self.bandwidth_patterns:
            was = self.warmup_pass.get(pat.pattern)
            if was:
                worst = max(worst, abs(pat.gbps - was) / was * 100.0)
        return round(worst, 2)

    @property
    def clock_ramped(self) -> bool:
        """Did the SM clock move materially across the bandwidth patterns?

        True means the patterns were measured at different clocks and are not
        comparable to each other, let alone publishable as ceilings.
        """
        clks = [p.sm_clock_start_mhz for p in self.bandwidth_patterns
                if p.sm_clock_start_mhz > 0]
        if len(clks) < 2:
            return False
        return (max(clks) - min(clks)) / max(clks) > 0.05

    def pattern(self, name: str) -> BandwidthResult | None:
        for p in self.bandwidth_patterns:
            if p.pattern == name:
                return p
        return None

    def ridge_point(self, bandwidth_gbps: float | None = None) -> float:
        """FLOP/byte above which a kernel can be compute bound, as measured."""
        bw = bandwidth_gbps or self.achieved_bandwidth_gbps
        return (self.achieved_bf16_tflops * 1e12) / (bw * 1e9)

    @property
    def sustained_peak_tflops(self) -> float | None:
        """Silicon ceiling at the clock the GEMM was actually measured at."""
        return sustained_peak_tflops(self.gemm_clock_mhz)

    @property
    def gemm_efficiency_pct(self) -> float | None:
        """cuBLAS as a fraction of what this clock can deliver.

        The denominator is the clock the GEMM was measured at. Against the
        datasheet instead, the figure also carries a boost clock the part does
        not sustain, which is not a property of the kernel.
        """
        peak = self.sustained_peak_tflops
        return round(100.0 * self.achieved_bf16_tflops / peak, 1) if peak else None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bandwidth_patterns"] = [p.as_dict() for p in self.bandwidth_patterns]
        d["sustained_peak_tflops"] = (round(self.sustained_peak_tflops, 1)
                                      if self.sustained_peak_tflops else None)
        d["gemm_efficiency_pct"] = self.gemm_efficiency_pct
        # Ridge is the number that classifies every cell, and it moves with the
        # denominator. Record it for each pattern so the sensitivity is visible
        # rather than hidden behind one choice.
        d["ridge_by_pattern"] = {
            p.pattern: round(self.ridge_point(p.gbps), 1)
            for p in self.bandwidth_patterns
        }
        return d


def _elems(target_bytes: int) -> int:
    return int(target_bytes // 4)


def clocks_settled(history: list[int], tol_pct: float) -> bool:
    """Have the last three clock samples stopped moving?

    Pure so it can be tested without a GPU, which matters because the failure it
    guards against is silent: a ceiling measured mid-ramp is not a ceiling, and
    nothing in the number says so.

    A zero is not a plateau. `nvidia-smi` returns 0 when the query fails, and
    three zeros have a spread of zero, which would otherwise read as the most
    settled clock imaginable.
    """
    if len(history) < 3:
        return False
    recent = history[-3:]
    if min(recent) <= 0:
        return False
    return (max(recent) - min(recent)) / max(recent) * 100.0 <= tol_pct


def _load_compute():
    """Dense bf16 matmul: saturates tensor cores, high power, LOW clock."""
    a = torch.randn((8192, 8192), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)

    def step():
        for _ in range(32):
            torch.mm(a, a, out=out)
    return step


def _load_memory():
    """Streaming copy over a buffer far larger than L2: low power, HIGH clock.

    This is the load the bandwidth patterns actually impose, and its steady
    state is 500 MHz above the compute one on an H200 SXM, which is why
    settling under `_load_compute` and then measuring bandwidth measures the
    wrong plateau.
    """
    n = DEFAULT_BUFFER_BYTES // 4
    a = torch.empty(n, dtype=torch.float32, device="cuda").uniform_(0.0, 1.0)
    b = torch.empty_like(a)

    def step():
        for _ in range(8):
            b.copy_(a)
    return step


#: Which sustained load to hold while waiting for the clock to plateau. The
#: right choice is whichever resembles the measurement about to be taken.
SETTLE_LOADS = {"compute": _load_compute, "memory": _load_memory}


def settle_clocks(max_seconds: float = 30.0, tol_pct: float = 2.0,
                  poll_seconds: float = 1.5, load: str = "compute") -> dict:
    """Hold the GPU under load until its clock stops climbing.

    MEASURED, H200 SXM 2026-08-22: a calibration started from idle ran its first
    pattern at 840 MHz and its last at 1980 MHz. Patterns are measured in a
    fixed order, so every one of them sat at a different point on the ramp and
    the later ones looked faster for a reason that has nothing to do with DRAM.
    The same run's cuBLAS figure moved 795 -> 725 TFLOP/s against an earlier
    one, which is the same artefact from the other side.

    A ceiling measured mid-ramp is not a ceiling. So: run a sustained load,
    poll the clock, and return only once consecutive samples agree within
    `tol_pct` (or `max_seconds` elapses, which is itself worth recording).

    MEASURED, 2026-08-27, and the reason `load` exists. Settling under the
    matmul above converged at 1470 MHz and 64 C; the bandwidth patterns then ran
    at 1980 MHz and 52 C. The chip COOLED and BOOSTED, because streaming draws
    less power than saturating tensor cores. The settle was real, it was just
    the steady state of the wrong workload, and the read ceiling it produced was
    1.7% low. Settle under `memory` before bandwidth and `compute` before the
    GEMM.
    """
    import time

    step = SETTLE_LOADS[load]   # KeyError for an unknown load, before any work
    T.require_cuda()
    step = step()

    history: list[int] = []
    deadline = time.monotonic() + max_seconds
    settled = False
    while time.monotonic() < deadline:
        stop = time.monotonic() + poll_seconds
        # Enqueue deep and sync once per poll. Syncing every few kernels leaves
        # gaps the clock governor reacts to, and the settle then converges to a
        # partial-load plateau BELOW what the real measurement induces. Measured
        # on this box: settling this way reached 1575 MHz, and the bandwidth
        # patterns immediately drove it to 1980.
        while time.monotonic() < stop:
            step()
            torch.cuda.synchronize()
        history.append(T.ClockState.sample().sm_clock_mhz)
        if clocks_settled(history, tol_pct):
            settled = True
            break

    return {"settled": settled, "clock_history_mhz": history,
            "final_mhz": history[-1] if history else 0,
            "max_seconds": max_seconds, "load": load}


def measure_bandwidth(target_bytes: int = DEFAULT_BUFFER_BYTES, warmup: int = 5,
                      iters: int = 30, trials: int = 3,
                      l2_flush: bool = True) -> list[BandwidthResult]:
    """STREAM-style patterns on buffers far larger than L2.

    All four are measured because they genuinely differ, and a single figure
    would not show that. Byte counts are the compulsory traffic each pattern
    must move:

      read   1N   a full reduction; reads every element, writes one scalar
      write  1N   a fill; writes every element, reads none
      copy   2N   one read plus one write per element
      triad  3N   two reads plus one write per element

    The 1N write convention is PROVEN, not assumed. A read-for-ownership would
    make real traffic 2N, which on an H200 would imply 9316 GB/s, or 194% of the
    4.8 TB/s pin rate. Impossible, and the buffer is over 100x L2 so cache
    cannot be absorbing it. `fill_kernel_cuda` also has no memset path for CUDA
    tensors, and a warp of vectorised fp32 stores covers whole 128 B lines, so
    no sector fill occurs.

    Write is still not the ceiling: it is the least representative pattern for a
    read-dominated workload, and the only one whose timed window can shed
    writeback.
    """
    T.require_cuda()
    n = _elems(target_bytes)
    nbytes = n * 4

    # Contents are irrelevant to a bandwidth measurement, and randn would burn
    # several GB of curand for nothing.
    a = torch.empty(n, device="cuda", dtype=torch.float32).fill_(1.0)
    b = torch.empty(n, device="cuda", dtype=torch.float32).fill_(2.0)
    c = torch.empty_like(a)
    sink = torch.zeros((), device="cuda", dtype=torch.float32)

    def run(pattern, fn, moved, note="", flush=None):
        before = T.ClockState.sample()
        res = T.time_eager(fn, warmup=warmup, iters=iters, trials=trials,
                           l2_flush=l2_flush if flush is None else flush)
        after = T.ClockState.sample()
        return BandwidthResult(
            pattern=pattern, bytes_moved=moved, ms_p50=res.ms_p50,
            ms_min=res.ms_min, gbps=moved / (res.ms_p50 * 1e-3) / 1e9,
            gbps_peak_min=moved / (res.ms_min * 1e-3) / 1e9, note=note,
            sm_clock_start_mhz=before.sm_clock_mhz,
            sm_clock_end_mhz=after.sm_clock_mhz)

    out = [
        run("read", lambda: torch.sum(a, dim=0, out=sink), nbytes,
            "closest analogue to streaming expert weights"),
        run("copy", lambda: c.copy_(a), 2 * nbytes, ""),
        run("triad", lambda: torch.add(a, b, alpha=2.0, out=c), 3 * nbytes,
            "canonical STREAM metric; the default ceiling"),
        # Deliberately unflushed. The flusher runs BEFORE the start event, and
        # in read mode it evicts the previous fill's dirty lines during untimed
        # time, shedding writeback the timed window should have paid. That
        # biases write HIGH, the wrong direction for the figure already at the
        # top of the band. The flush is pointless here anyway: the buffer is
        # over 100x L2.
        run("write", lambda: c.fill_(1.0), nbytes,
            "1N proven (2N would exceed the pin rate by 94%); unflushed so the "
            "timed window cannot shed writeback", flush=False),
    ]
    # No `del` here: the timing lambdas close over these buffers, and unbinding
    # the names would leave those closures referring to deleted locals. They are
    # freed when this frame exits; the caller reclaims the memory.
    return out


#: Dense BF16 FLOP per SM per clock, by compute capability. The Hopper value
#: is confirmed exactly against the datasheet: 132 SM x 4096 x 1830 MHz =
#: 989.4 TFLOP/s, which is NVIDIA's published H200 SXM figure. That also reveals
#: what the datasheet number assumes: a 1830 MHz boost clock.
_DENSE_BF16_FLOP_PER_SM_CLK = {(9, 0): 4096}


def sustained_peak_tflops(sm_clock_mhz: float) -> float | None:
    """What the silicon can do AT THIS CLOCK, as opposed to at its boost clock.

    The datasheet peak assumes a clock the part cannot hold under sustained
    dense tensor load. Measured on an H200 SXM at 700 W: the clock settles at
    ~1455 MHz under continuous BF16 GEMMs, not the 1830 MHz the 989.5 TFLOP/s
    headline implies. Against the datasheet cuBLAS reads 71.5%; against what
    1455 MHz can deliver it reads ~90%. That 18 point difference is the clock,
    not the library, which is why this file records the clock it measured at.
    """
    import torch

    if not torch.cuda.is_available() or not sm_clock_mhz:
        return None
    props = torch.cuda.get_device_properties(0)
    per_clk = _DENSE_BF16_FLOP_PER_SM_CLK.get((props.major, props.minor))
    if per_clk is None:
        return None
    return props.multi_processor_count * per_clk * sm_clock_mhz * 1e6 / 1e12


@dataclass(frozen=True)
class GemmResult:
    """Named rather than a tuple on purpose.

    This started as `(tflops, shape)`, grew a clock, and silently broke a caller
    that unpacked two values — a failure that only shows up on the GPU, six
    minutes into a run. A dataclass makes the next added field free.
    """

    tflops: float
    shape: tuple[int, int, int]
    sm_clock_mhz: int


def measure_bf16_gemm(n: int = 8192, warmup: int = 5, iters: int = 20,
                      trials: int = 3) -> GemmResult:
    """Achievable dense BF16 through cuBLAS: the compute roof this box gives.

    A square GEMM at n=8192 is comfortably compute bound and is what a tuned
    library achieves, so it is the fair ceiling for a hand-written kernel.
    """
    T.require_cuda()
    a = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((n, n), device="cuda", dtype=torch.bfloat16)
    res = T.time_eager(lambda: torch.mm(a, b, out=out), warmup=warmup,
                       iters=iters, trials=trials, l2_flush=False)
    # Sample the clock DURING the measurement: it is what the result should be
    # normalised against, and it is not the boost clock.
    clk = T.ClockState.sample().sm_clock_mhz
    tflops = (2.0 * n * n * n) / (res.ms_p50 * 1e-3) / 1e12
    return GemmResult(tflops=tflops, shape=(n, n, n), sm_clock_mhz=clk)


def calibrate(target_bytes: int = DEFAULT_BUFFER_BYTES, gemm_n: int = 8192,
              ceiling: str = DEFAULT_CEILING, settle: bool = True,
              settle_seconds: float = 30.0) -> Calibration:
    T.require_cuda()
    # Settle FIRST. Measuring from idle put this machine's first pattern at
    # 840 MHz and its last at 1980, which is a bigger effect than anything the
    # choice of pattern argues about.
    settle_info = (settle_clocks(settle_seconds, load="compute") if settle
                   else {"settled": False, "skipped": True})
    before = T.ClockState.sample()

    # COMPUTE FIRST. Measured on an H200 SXM, the cuBLAS figure fell 795 -> 725
    # -> 687 TFLOP/s across three runs as each did more sustained memory work
    # before it. On a 700 W part, heavy HBM traffic eats the power budget the
    # tensor cores need, so a compute roof measured after a bandwidth sweep is a
    # power-limited number, not a ceiling. The settle above is matmul work, so
    # the GPU is already in the right state for exactly this measurement.
    gemm = measure_bf16_gemm(gemm_n)
    torch.cuda.empty_cache()

    # SETTLE AGAIN, UNDER A MEMORY LOAD. The settle above converged on the
    # matmul's steady state, which is the right one for the GEMM and the wrong
    # one for what follows: measured 2026-08-27 on an H200 SXM, the compute
    # plateau is 1470 MHz at 64 C and the bandwidth patterns then run at 1980
    # MHz and 52 C. The chip COOLS and BOOSTS, because streaming draws less
    # power than saturating tensor cores.
    #
    # The two-pass warmup below does not fix it. Both passes agreed with each
    # other (read 4385.6 then 4389.4) and both were 1.7% under what the same
    # torch.sum(dim=0) call measures on a chip that has not just run a GEMM.
    # Agreement between two measurements in the same wrong state is not
    # convergence on the right one.
    settle_mem = (settle_clocks(settle_seconds, load="memory") if settle
                  else {"settled": False, "skipped": True})

    # BANDWIDTH TWICE, keep the second. The first pass finishes ramping whatever
    # the settle did not, so pass one is warmup and pass two has all four
    # patterns in the same state. Without this the first pattern measured is
    # systematically different from the other three.
    first_pass = measure_bandwidth(target_bytes)
    patterns = measure_bandwidth(target_bytes)
    torch.cuda.empty_cache()
    after = T.ClockState.sample()

    chosen = next((p for p in patterns if p.pattern == ceiling), None)
    if chosen is None:
        raise ValueError(
            f"unknown ceiling pattern {ceiling!r}; "
            f"measured: {[p.pattern for p in patterns]}")

    # A pure-read stream cannot legitimately be slower than 2R+1W at the DRAM
    # level. If it is, torch.sum is reporting ATen's tree-reduction rate rather
    # than the DRAM read rate, and it is not a valid denominator.
    read = next((p for p in patterns if p.pattern == "read"), None)
    triad = next((p for p in patterns if p.pattern == "triad"), None)
    if read and triad and read.gbps < triad.gbps:
        patterns = tuple(
            BandwidthResult(**{**p.as_dict(),
                               "note": "reduction-limited, not DRAM-limited; "
                                       "not a valid ceiling"})
            if p.pattern == "read" else p for p in patterns)
        if ceiling == "read":
            raise ValueError(
                f"read measured {read.gbps:.0f} GB/s, below triad's "
                f"{triad.gbps:.0f}. A pure read cannot be slower than 2R+1W at "
                "the DRAM level, so that figure is ATen's reduction rate, not "
                "bandwidth. Use --ceiling triad.")
        chosen = next(p for p in patterns if p.pattern == ceiling)

    drift, throttled = T.clock_drift(before, after)
    return Calibration(
        gpu_name=torch.cuda.get_device_properties(0).name,
        achieved_bandwidth_gbps=chosen.gbps,
        ceiling_pattern=ceiling,
        achieved_bf16_tflops=gemm.tflops,
        bandwidth_patterns=tuple(patterns),
        gemm_shape=gemm.shape,
        gemm_clock_mhz=gemm.sm_clock_mhz,
        buffer_bytes=target_bytes,
        clocks={"sm_start_mhz": before.sm_clock_mhz, "sm_end_mhz": after.sm_clock_mhz,
                "temp_start_c": before.temp_c, "temp_end_c": after.temp_c,
                "drift_pct": round(drift, 2), "throttled": throttled},
        settle={**settle_info, "bandwidth_settle": settle_mem},
        warmup_pass={p.pattern: round(p.gbps, 1) for p in first_pass},
    )


def implied_traffic_ratio(compulsory_bytes: float, ms: float,
                          achieved_bw_bytes_s: float) -> float:
    """Upper bound on how much more traffic a kernel moved than it had to.

    With counters unavailable, actual DRAM bytes cannot be read. But if a cell
    is memory bound, `time x achievable_bandwidth` bounds the bytes that could
    have moved in that time, and dividing by the compulsory minimum gives a
    bound on the re-read factor.

    It bounds traffic CONDITIONAL on all of it being served from DRAM at no more
    than the named ceiling, and it is not a measurement: the same number absorbs
    low occupancy, latency stalls, and launch overhead. A ratio near 1 is strong
    evidence the kernel moves close to the minimum traffic; a large ratio says
    "something is costing you", not specifically "you are re-reading".

    Values BELOW 1.0 are legitimate, not violations: they mean L2 served part of
    the traffic, so less than the compulsory minimum reached DRAM. Read them
    against the row's l2_flush column and l2_absorbed_bytes.

    Worth more than the choice of denominator: the ridge that gates this column
    comes from a cuBLAS 8192-cubed peak, and an MoE grouped GEMM at 1-64 rows
    per expert cannot approach that shape. An overstated ridge means more cells
    qualify, and a launch- or latency-bound cell then shows a large ratio that
    reads as re-reads.

    Sensitive to which bandwidth ceiling was chosen, which is why the
    calibration records every pattern and names the one it used.
    """
    if compulsory_bytes <= 0 or ms <= 0 or achieved_bw_bytes_s <= 0:
        return 0.0
    return (ms * 1e-3 * achieved_bw_bytes_s) / compulsory_bytes


def l2_absorbed_bytes(ms_flushed: float, ms_warm: float,
                      achieved_bw_bytes_s: float) -> float:
    """Bytes L2 served rather than DRAM, inferred from the flush axis.

    The harness already times each cell with L2 flushed and with L2 warm. The
    time difference, multiplied by achievable bandwidth, estimates the traffic
    the cache absorbed. This is the counter-free stand-in for a cache hit-rate
    metric, and it comes free from an axis that is already swept.
    """
    return max(0.0, ms_flushed - ms_warm) * 1e-3 * achieved_bw_bytes_s
