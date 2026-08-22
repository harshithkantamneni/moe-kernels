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
from collections.abc import Callable
from dataclasses import dataclass

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


def runtime_info() -> dict:
    """Machine and numerics facts that belong in every row.

    The torch numerics switches matter twice over: their defaults have moved
    between releases, and they change both the reference's numerics and the
    speed of any torch-backed baseline. A published row that omits them is not
    reproducible.
    """
    import platform
    import sys

    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = ""

    info = {
        "torch_version": torch.__version__,
        "triton_version": triton_version,
        "python_version": sys.version.split()[0],
        "host_cpu": platform.processor() or platform.machine(),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_bf16_reduced_reduction": bool(
            getattr(torch.backends.cuda.matmul,
                    "allow_bf16_reduced_precision_reduction", False)),
        "allow_fp16_reduced_reduction": bool(
            getattr(torch.backends.cuda.matmul,
                    "allow_fp16_reduced_precision_reduction", False)),
    }
    if not torch.cuda.is_available():
        return info

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    driver = _nvidia_smi("driver_version")
    info.update({
        "gpu_name": props.name,
        "gpu_count": torch.cuda.device_count(),
        "device_index": index,
        "driver_version": driver[0] if driver else "",
        "cuda_version": torch.version.cuda or "",
        "sm_count": props.multi_processor_count,
        "l2_bytes": getattr(props, "L2_cache_size", 0),
        "total_memory": props.total_memory,
    })
    return info


@dataclass(frozen=True)
class ClockState:
    sm_clock_mhz: int
    temp_c: int

    @classmethod
    def sample(cls) -> ClockState:
        """Sample SM clock and temperature.

        Prefers torch's NVML bindings, which cost tens of microseconds. The
        nvidia-smi fallback below runs twice per timing mode, eight times per
        cell, and a fork that initialises NVML costs tens of milliseconds on
        Linux: tens of minutes of a large sweep spent on process startup.
        """
        if torch.cuda.is_available():
            try:
                return cls(int(torch.cuda.clock_rate()),
                           int(torch.cuda.temperature()))
            except (AttributeError, RuntimeError, ValueError):
                pass  # older torch, or NVML unavailable inside this container
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
    """Evicts L2 between timed iterations by touching a buffer larger than it.

    The flush READS rather than writes. A write flush (the common `buf.zero_()`
    idiom) leaves up to a full L2 of dirty lines, and those writebacks land
    inside the NEXT timed interval, stealing roughly 11 microseconds of HBM
    bandwidth on a 50 MiB L2 at 4.8 TB/s. Irrelevant for a millisecond kernel,
    a 10-30% inflation on a sub-100-microsecond span, which is exactly the
    small-batch regime this project studies.
    """

    def __init__(self, megabytes: int = DEFAULT_FLUSH_MB, device: str = "cuda",
                 mode: str = "read"):
        self.enabled = megabytes > 0
        self.mode = mode
        self.megabytes = megabytes if self.enabled else 0
        if not self.enabled:
            self.buf = None
            self.out = None
            return
        elems = megabytes * 1024 * 1024 // 4
        self.buf = torch.empty((elems, 1), dtype=torch.float32, device=device)
        self.out = torch.zeros((1,), dtype=torch.float32, device=device)

    def flush(self) -> None:
        if self.buf is None:
            return
        if self.mode == "write":
            self.buf.zero_()
        else:
            torch.sum(self.buf, dim=0, out=self.out)


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

class _EventPairs:
    """Pre-created, pre-primed CUDA event pairs.

    torch creates the underlying cudaEvent lazily on first record(). Creating
    events inside the measured loop puts cudaEventCreateWithFlags between a
    kernel enqueue and the closing record, and whenever the GPU has drained
    ahead of the CPU (short spans, and always for graph replay) that CPU cost
    lands INSIDE the measured interval. A fixed offset on both arms is not
    harmless: it biases the eager/graph RATIO toward 1, which is precisely the
    number this project would publish.
    """

    def __init__(self, n: int):
        self.starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        self.ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for e in (*self.starts, *self.ends):
            e.record()          # forces creation of the underlying cudaEvent
        torch.cuda.synchronize()

    def elapsed(self, n: int) -> list[float]:
        return [self.starts[i].elapsed_time(self.ends[i]) for i in range(n)]


def calibrate_iters(fn: Callable[[], None], target_ms: float = 200.0,
                    lo: int = 10, hi: int = 2000) -> int:
    """Measure one warm iteration, then choose an iteration count.

    Bucketing on FLOPs (the previous approach) is wrong for this sweep: these
    cells are bandwidth bound, so at small token counts the time is set by
    weight traffic and is nearly independent of the FLOP count. That heuristic
    inverted its own goal, spending the most metered GPU time on the
    cheapest-FLOP cells.
    """
    require_cuda()
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    ms = max(start.elapsed_time(end), 1e-4)
    return max(lo, min(hi, int(target_ms / ms)))


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
    flush_mb: int = 0
    flush_mode: str = "read"


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
    iters: int | None = None,
    trials: int = 3,
    l2_flush: bool = True,
    flush_mb: int = DEFAULT_FLUSH_MB,
    flush_mode: str = "read",
    target_ms: float = 200.0,
) -> TimingResult:
    """Per-iteration CUDA-event timing of an eagerly launched callable."""
    require_cuda()
    flusher = L2Flusher(flush_mb if l2_flush else 0, mode=flush_mode)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if iters is None:
        iters = calibrate_iters(fn, target_ms)
    events = _EventPairs(iters)

    samples: list[float] = []
    for _ in range(trials):
        for i in range(iters):
            flusher.flush()
            events.starts[i].record()
            fn()
            events.ends[i].record()
        torch.cuda.synchronize()
        samples.extend(events.elapsed(iters))

    return _summarise(samples, warmup=warmup, iters=iters, trials=trials,
                      l2_flush=l2_flush, cuda_graph=False,
                      flush_mb=flusher.megabytes, flush_mode=flush_mode)


class NotCapturable(RuntimeError):
    """Raised when a callable cannot be captured into a CUDA graph.

    This is a result, not a failure: an implementation that syncs with the host
    (an `.item()`, a `.tolist()`, a python loop over expert offsets) cannot be
    used in real MoE inference, and the harness records that fact.
    """


def time_graph(
    fn: Callable[[], None],
    warmup: int = 25,
    iters: int | None = None,
    trials: int = 3,
    l2_flush: bool = True,
    flush_mb: int = DEFAULT_FLUSH_MB,
    flush_mode: str = "read",
    target_ms: float = 200.0,
    on_captured: Callable[[], None] | None = None,
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

    flusher = L2Flusher(flush_mb if l2_flush else 0, mode=flush_mode)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    # A replay writes into graph-private buffers that every replay reuses, so a
    # kernel leaving part of its output unwritten would show the PREVIOUS
    # replay's correct values. The caller re-checks the replayed result here,
    # while the graph is still the thing that produced it.
    if on_captured is not None:
        on_captured()

    if iters is None:
        iters = calibrate_iters(graph.replay, target_ms)
    events = _EventPairs(iters)

    samples: list[float] = []
    for _ in range(trials):
        for i in range(iters):
            flusher.flush()
            events.starts[i].record()
            graph.replay()
            events.ends[i].record()
        torch.cuda.synchronize()
        samples.extend(events.elapsed(iters))

    return _summarise(samples, warmup=warmup, iters=iters, trials=trials,
                      l2_flush=l2_flush, cuda_graph=True,
                      flush_mb=flusher.megabytes, flush_mode=flush_mode)
