"""Measure what this machine can actually do, since we cannot read its counters.

Nsight Compute needs GPU performance counters, which need a host-level
`NVreg_RestrictProfilingToAdminUsers=0` that a container tenant cannot set. On a
rented pod `ncu` fails with ERR_NVGPUCTRPERM, so DRAM traffic cannot be measured
directly.

What can be measured without any counter access is the machine's *achievable*
ceilings, using ordinary kernels and a clock:

  bandwidth   large-buffer copy and triad, sized to defeat L2
  compute     a large square GEMM through cuBLAS

That turns the roofline from a datasheet claim into a measured one, which is
strictly better: a kernel at 92% of achievable bandwidth is a fact about your
kernel, while the same number against a spec peak silently blames your kernel
for the 15-20% the hardware was never going to give you.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from . import timing as T


@dataclass(frozen=True)
class BandwidthResult:
    pattern: str
    bytes_moved: int
    ms_p50: float
    gbps: float


@dataclass(frozen=True)
class Calibration:
    gpu_name: str
    achieved_bandwidth_gbps: float
    achieved_bf16_tflops: float
    bandwidth_patterns: tuple[BandwidthResult, ...]
    gemm_shape: tuple[int, int, int]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bandwidth_patterns"] = [asdict(b) for b in self.bandwidth_patterns]
        return d


def _buffer_elems(target_bytes: int, dtype=torch.float32) -> int:
    return int(target_bytes // torch.tensor([], dtype=dtype).element_size())


def measure_bandwidth(target_bytes: int = 2 << 30, warmup: int = 5,
                      iters: int = 30, trials: int = 3) -> list[BandwidthResult]:
    """STREAM-style patterns on buffers far larger than L2.

    Copy moves 2N bytes (one read, one write); triad moves 3N (two reads, one
    write). Both are reported because they do not always agree, and the higher
    of the two is the honest ceiling to quote.
    """
    T.require_cuda()
    n = _buffer_elems(target_bytes)
    # Contents are irrelevant to a bandwidth measurement, and randn would burn
    # several GB of curand for nothing.
    a = torch.empty(n, device="cuda", dtype=torch.float32).fill_(1.0)
    b = torch.empty(n, device="cuda", dtype=torch.float32).fill_(2.0)
    c = torch.empty_like(a)
    nbytes = n * 4

    def run(fn, moved):
        res = T.time_eager(fn, warmup=warmup, iters=iters, trials=trials,
                           l2_flush=False)
        return res.ms_p50, moved / (res.ms_p50 * 1e-3) / 1e9

    out = []
    ms, gbps = run(lambda: c.copy_(a), 2 * nbytes)
    out.append(BandwidthResult("copy", 2 * nbytes, ms, gbps))

    ms, gbps = run(lambda: torch.add(a, b, alpha=2.0, out=c), 3 * nbytes)
    out.append(BandwidthResult("triad", 3 * nbytes, ms, gbps))

    ms, gbps = run(lambda: c.fill_(1.0), nbytes)
    out.append(BandwidthResult("write", nbytes, ms, gbps))

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
    flops = 2.0 * n * n * n
    tflops = flops / (res.ms_p50 * 1e-3) / 1e12
    return tflops, (n, n, n)


def calibrate(target_bytes: int = 2 << 30, gemm_n: int = 8192) -> Calibration:
    T.require_cuda()
    patterns = measure_bandwidth(target_bytes)
    torch.cuda.empty_cache()          # 2 GB of buffers are now unreferenced
    tflops, shape = measure_bf16_gemm(gemm_n)
    torch.cuda.empty_cache()
    return Calibration(
        gpu_name=torch.cuda.get_device_properties(0).name,
        achieved_bandwidth_gbps=max(p.gbps for p in patterns),
        achieved_bf16_tflops=tflops,
        bandwidth_patterns=tuple(patterns),
        gemm_shape=shape,
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

    Only meaningful when the cell is genuinely memory bound. The caller checks
    that using compulsory intensity, which is an upper bound on true intensity,
    so `compulsory_AI < ridge` implies `true_AI < ridge` and the inference is
    sound in the conservative direction.
    """
    if compulsory_bytes <= 0 or ms <= 0 or achieved_bw_bytes_s <= 0:
        return 0.0
    implied_bytes = ms * 1e-3 * achieved_bw_bytes_s
    return implied_bytes / compulsory_bytes


def l2_absorbed_bytes(ms_flushed: float, ms_warm: float,
                      achieved_bw_bytes_s: float) -> float:
    """Bytes L2 served rather than DRAM, inferred from the flush axis.

    The harness already times each cell with L2 flushed and with L2 warm. The
    time difference, multiplied by achievable bandwidth, estimates the traffic
    the cache absorbed. This is the counter-free stand-in for a cache hit-rate
    metric, and it comes free from an axis that is already swept.
    """
    delta = max(0.0, ms_flushed - ms_warm)
    return delta * 1e-3 * achieved_bw_bytes_s
