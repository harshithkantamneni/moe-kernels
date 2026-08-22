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
streaming reads of expert weights, so `read` is arguably the most representative
denominator for that regime specifically. That is a judgement call, not a fact,
which is exactly why the choice is recorded rather than baked in.
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

    The write figure is the one to distrust: if any fill path issues a
    read-for-ownership, real traffic is 2N and the reported number is half the
    truth. It is recorded with that caveat rather than dropped, because a write
    figure implausibly close to datasheet peak is itself a useful signal.
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

    def run(pattern, fn, moved, note=""):
        res = T.time_eager(fn, warmup=warmup, iters=iters, trials=trials,
                           l2_flush=l2_flush)
        return BandwidthResult(
            pattern=pattern, bytes_moved=moved, ms_p50=res.ms_p50,
            ms_min=res.ms_min, gbps=moved / (res.ms_p50 * 1e-3) / 1e9,
            gbps_peak_min=moved / (res.ms_min * 1e-3) / 1e9, note=note)

    out = [
        run("read", lambda: torch.sum(a, dim=0, out=sink), nbytes,
            "closest analogue to streaming expert weights"),
        run("copy", lambda: c.copy_(a), 2 * nbytes, ""),
        run("triad", lambda: torch.add(a, b, alpha=2.0, out=c), 3 * nbytes,
            "canonical STREAM metric; the default ceiling"),
        run("write", lambda: c.fill_(1.0), nbytes,
            "distrust: counts 1N, but a read-for-ownership would make it 2N"),
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
              ceiling: str = DEFAULT_CEILING) -> Calibration:
    T.require_cuda()
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
    )


def implied_traffic_ratio(compulsory_bytes: float, ms: float,
                          achieved_bw_bytes_s: float) -> float:
    """Upper bound on how much more traffic a kernel moved than it had to.

    With counters unavailable, actual DRAM bytes cannot be read. But if a cell
    is memory bound, `time x achievable_bandwidth` bounds the bytes that could
    have moved in that time, and dividing by the compulsory minimum gives a
    bound on the re-read factor.

    It is an UPPER bound, not a measurement: the same number also absorbs low
    occupancy, latency stalls, and launch overhead. A ratio near 1 is strong
    evidence the kernel is moving close to the minimum traffic; a large ratio
    says "something is costing you", not specifically "you are re-reading".

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
