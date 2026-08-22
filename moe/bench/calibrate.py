"""Measure what this machine can actually do, since we cannot read its counters.

Nsight Compute needs GPU performance counters, which need a host-level
`NVreg_RestrictProfilingToAdminUsers=0` that a container tenant cannot set. On a
rented pod `ncu` fails with ERR_NVGPUCTRPERM, so DRAM traffic cannot be measured
directly.

What can be measured without any counter access is the machine's *achievable*
ceilings, using ordinary kernels and a clock. That turns the roofline from a
datasheet claim into a measured one, which is strictly better: a kernel at 92%
of achievable bandwidth is a fact about your kernel, while the same number
against a spec peak silently blames your kernel for the 10-20% the hardware was
never going to give you.

WHICH BANDWIDTH NUMBER IS "THE" BANDWIDTH
-----------------------------------------
There isn't one. Read, write, copy and triad all measure real bandwidth and all
give different answers, because reads, writes and mixed traffic hit DRAM
differently. An earlier version of this file took `max()` across patterns, which
is indefensible: it reports whichever pattern the hardware happens to like best,
which is a property of the benchmark rather than of the workload.

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
    buffer_bytes: int
    clocks: dict = field(default_factory=dict)
    settle: dict = field(default_factory=dict)

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

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bandwidth_patterns"] = [p.as_dict() for p in self.bandwidth_patterns]
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


def settle_clocks(max_seconds: float = 30.0, tol_pct: float = 2.0,
                  poll_seconds: float = 1.5) -> dict:
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
    """
    import time

    T.require_cuda()
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)

    history: list[int] = []
    deadline = time.monotonic() + max_seconds
    settled = False
    while time.monotonic() < deadline:
        stop = time.monotonic() + poll_seconds
        while time.monotonic() < stop:
            for _ in range(8):
                torch.mm(a, a, out=out)
            torch.cuda.synchronize()
        clk = T.ClockState.sample().sm_clock_mhz
        history.append(clk)
        if len(history) >= 3 and history[-2] > 0:
            recent = history[-3:]
            spread = (max(recent) - min(recent)) / max(recent) * 100.0
            if spread <= tol_pct:
                settled = True
                break

    return {"settled": settled, "clock_history_mhz": history,
            "final_mhz": history[-1] if history else 0,
            "max_seconds": max_seconds}


def measure_bandwidth(target_bytes: int = DEFAULT_BUFFER_BYTES, warmup: int = 5,
                      iters: int = 30, trials: int = 3,
                      l2_flush: bool = True) -> list[BandwidthResult]:
    """STREAM-style patterns on buffers far larger than L2.

    All four are measured because they genuinely differ, and reporting only one
    hides that. Byte counts are the compulsory traffic each pattern must move:

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


def measure_bf16_gemm(n: int = 8192, warmup: int = 5, iters: int = 20,
                      trials: int = 3) -> tuple[float, tuple[int, int, int]]:
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
    tflops = (2.0 * n * n * n) / (res.ms_p50 * 1e-3) / 1e12
    return tflops, (n, n, n)


def calibrate(target_bytes: int = DEFAULT_BUFFER_BYTES, gemm_n: int = 8192,
              ceiling: str = DEFAULT_CEILING, settle: bool = True,
              settle_seconds: float = 30.0) -> Calibration:
    T.require_cuda()
    # Settle FIRST. Measuring from idle put this machine's first pattern at
    # 840 MHz and its last at 1980, which is a bigger effect than anything the
    # choice of pattern argues about.
    settle_info = settle_clocks(settle_seconds) if settle else {"settled": False,
                                                                "skipped": True}
    before = T.ClockState.sample()
    patterns = measure_bandwidth(target_bytes)
    torch.cuda.empty_cache()          # the buffers are now unreferenced
    tflops, shape = measure_bf16_gemm(gemm_n)
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
        achieved_bf16_tflops=tflops,
        bandwidth_patterns=tuple(patterns),
        gemm_shape=shape,
        buffer_bytes=target_bytes,
        clocks={"sm_start_mhz": before.sm_clock_mhz, "sm_end_mhz": after.sm_clock_mhz,
                "temp_start_c": before.temp_c, "temp_end_c": after.temp_c,
                "drift_pct": round(drift, 2), "throttled": throttled},
        settle=settle_info,
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
