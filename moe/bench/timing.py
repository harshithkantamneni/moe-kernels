"""GPU timing with the methodology recorded rather than assumed.

Three things most published MoE kernel numbers leave out, and which this module
makes explicit columns:

1. L2 residency. H200 has a large L2. Whether expert weights are already
   resident changes small-batch results by more than most kernel optimisations
   do, so the flush state is a parameter and a recorded fact.
2. Launch mode. At low token counts with many experts, kernel launch overhead
   is a first-order term. Eager and CUDA-graph replay are measured separately.
3. Clock and thermal drift. On shared rented hardware a "regression" is often
   just a hot box, so clocks are sampled before and after every cell.

Both timing modes use per-iteration CUDA events so the two are measured
identically and remain comparable. Event overhead (a few microseconds) is
therefore included in both, and is not subtracted.
"""
from __future__ import annotations

import statistics
import subprocess
from dataclasses import dataclass
from typing import Callable

import torch

DEFAULT_FLUSH_MB = 128  # comfortably larger than H200's L2


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device; timing must run on the GPU box")


# --------------------------------------------------------------------------
# machine facts
# --------------------------------------------------------------------------

def _nvidia_smi(query: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def gpu_info() -> dict:
    require_cuda()
    props = torch.cuda.get_device_properties(0)
    driver = _nvidia_smi("driver_version")
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = ""
    return {
        "gpu_name": props.name,
        "gpu_count": torch.cuda.device_count(),
        "driver_version": driver[0] if driver else "",
        "cuda_version": torch.version.cuda or "",
        "torch_version": torch.__version__,
        "triton_version": triton_version,
        "sm_count": props.multi_processor_count,
        "l2_bytes": getattr(props, "L2_cache_size", 0),
        "total_memory": props.total_memory,
    }


@dataclass(frozen=True)
class ClockState:
    sm_clock_mhz: int
    temp_c: int

    @classmethod
    def sample(cls) -> "ClockState":
        vals = _nvidia_smi("clocks.current.sm,temperature.gpu")
        if not vals:
            return cls(0, 0)
        try:
            sm, temp = (int(float(v)) for v in vals[0].split(","))
        except (ValueError, IndexError):
            return cls(0, 0)
        return cls(sm, temp)


def clock_drift(start: ClockState, end: ClockState) -> tuple[float, bool]:
    """Percent drop in SM clock across a cell, and whether it looks throttled."""
    if start.sm_clock_mhz <= 0:
        return 0.0, False
    drift = (start.sm_clock_mhz - end.sm_clock_mhz) / start.sm_clock_mhz * 100.0
    return drift, drift > 5.0


# --------------------------------------------------------------------------
# L2
# --------------------------------------------------------------------------

class L2Flusher:
    """Evicts L2 by writing a buffer larger than it between timed iterations."""

    def __init__(self, megabytes: int = DEFAULT_FLUSH_MB, device: str = "cuda"):
        self.enabled = megabytes > 0
        self.buf = (torch.empty(megabytes * 1024 * 1024, dtype=torch.int8, device=device)
                    if self.enabled else None)

    def flush(self) -> None:
        if self.buf is not None:
            self.buf.zero_()


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

@dataclass
class TimingResult:
    ms_p50: float
    ms_p90: float
    ms_min: float
    ms_std: float
    jitter_p90_over_p50: float
    warmup: int
    iters: int
    trials: int
    l2_flush: bool
    cuda_graph: bool
    samples: int


def _summarise(samples: list[float], **meta) -> TimingResult:
    s = sorted(samples)
    n = len(s)
    p50 = statistics.median(s)
    p90 = s[min(n - 1, int(round(0.9 * (n - 1))))]
    return TimingResult(
        ms_p50=p50,
        ms_p90=p90,
        ms_min=s[0],
        ms_std=statistics.stdev(s) if n > 1 else 0.0,
        jitter_p90_over_p50=(p90 / p50) if p50 > 0 else float("inf"),
        samples=n,
        **meta,
    )


def time_eager(
    fn: Callable[[], None],
    warmup: int = 25,
    iters: int = 100,
    trials: int = 3,
    l2_flush: bool = True,
    flush_mb: int = DEFAULT_FLUSH_MB,
) -> TimingResult:
    """Per-iteration CUDA-event timing of an eagerly launched callable."""
    require_cuda()
    flusher = L2Flusher(flush_mb if l2_flush else 0)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(trials):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            flusher.flush()
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        samples.extend(s.elapsed_time(e) for s, e in zip(starts, ends))

    return _summarise(samples, warmup=warmup, iters=iters, trials=trials,
                      l2_flush=l2_flush, cuda_graph=False)


class NotCapturable(RuntimeError):
    """Raised when a callable cannot be captured into a CUDA graph.

    This is a result, not a failure: an implementation that syncs with the host
    (an `.item()`, a `.tolist()`, a python loop over expert offsets) cannot be
    used in real MoE inference, and the harness records that fact.
    """


def time_graph(
    fn: Callable[[], None],
    warmup: int = 25,
    iters: int = 100,
    trials: int = 3,
    l2_flush: bool = True,
    flush_mb: int = DEFAULT_FLUSH_MB,
) -> TimingResult:
    """Capture `fn` into a CUDA graph and time replays.

    Isolates launch overhead, which dominates the many-expert small-batch regime
    that this project targets.
    """
    require_cuda()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            fn()
    except RuntimeError as e:
        raise NotCapturable(str(e)) from None

    flusher = L2Flusher(flush_mb if l2_flush else 0)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(trials):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            flusher.flush()
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
        samples.extend(s.elapsed_time(e) for s, e in zip(starts, ends))

    return _summarise(samples, warmup=warmup, iters=iters, trials=trials,
                      l2_flush=l2_flush, cuda_graph=True)


def iters_for(flops: float) -> int:
    """Iteration count that keeps timed work roughly constant across shapes.

    Carried over from the A100 harness. Keeps a sweep's wall-clock predictable,
    which matters when the box is metered.
    """
    if flops < 1e10:
        return 500
    if flops < 5e10:
        return 300
    if flops < 2e11:
        return 150
    if flops < 8e11:
        return 80
    return 40
