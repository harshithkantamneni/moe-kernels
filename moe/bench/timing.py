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

ONE INSTRUMENT, AND WHY IT HAS A NAME
-------------------------------------
Until 2026-09-02 this repository had two timing instruments and compared their
outputs. The roof (`calibrate.measure_bf16_gemm` via `time_eager` below) was
measured queue-deep: events pre-primed, L2 flushed per iteration, one
synchronise per trial. Every ladder script carried a private `time_call` that
created its events inside the loop, recorded the start event on a stream that
had just been synchronised (an idle GPU, so the host's own enqueue cost sat
inside the interval), synchronised after every iteration, never flushed, and
read no clock. The audit that found it (AUDIT_REPORT A7) bounded the exposed
host prefix at ~0.18 ms per fused_experts call on the H200 pod and ~0.30 ms on
the A100 pod, a bias of 8-16% in alpha at the smallest ladder cells, different
per card, and of the order of the cross-card effect the study registered.

`time_kernel` is the one instrument now, and `TIMING_BASIS` is its name. Every
consumer writes `TIMING_BASIS` into its rows so a reader can tell at a glance
which apparatus produced a number, and a row without it is a row from before
the fix. Change the string when the instrument changes in a way that moves
numbers; never otherwise.

CLOCKS ARE READ UNDER LOAD, AND FLAGGED ON LEVEL AS WELL AS DRIFT
-----------------------------------------------------------------
The old `clock_drift` flag compared two idle-instant samples (both taken after
a synchronise) and fired only on a >5% DROP. On the published alpha-0558 arm it
flagged 91% of vLLM rows above T=4096 while flagged and unflagged replicates
in the same cell timed at ratio 0.998 with identical end clocks: it detected
whether the START sample had caught the idle boost, not throttling. And a
card sitting at 1500 MHz for the whole cell, with a roof measured at 1980,
passed it with drift 0.0. `time_kernel` polls the clock from a background
thread WHILE the trials run, reports the median as `sm_clock_load_mhz`, and
sets two flags: LEVEL (`clock_level_ok`: the loaded clock is within
`LEVEL_FRACTION` of the reference the roof was measured at) and DRIFT
(`clock_drift_ok`: first and last under-load samples agree within
`DRIFT_FRACTION`, in either direction). A rise is a defect too: it means the
warmup did not reach the operating point, so the trials were not at one clock.
"""
from __future__ import annotations

import statistics
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch

#: The name of the instrument `time_kernel` implements. Written into every row
#: it produces. Bump the version suffix when a change would move a published
#: number; the reader compares this string, not a commit hash.
TIMING_BASIS = "queue-deep/l2-flush/clock-under-load/v2"

#: LEVEL flag: the SM clock sampled under load must be at least this fraction
#: of the reference clock (the one the roof was measured at) for the cell to
#: be comparable with the roof. 0.95 because the H200's compute plateau moves
#: 1455-1515 MHz across sessions of one card (calibrate.py), a 4% band, and a
#: flag inside the band would fire on the card's own session-to-session noise.
LEVEL_FRACTION = 0.95

#: DRIFT flag: first and last under-load samples may differ by at most this
#: fraction of the first, in EITHER direction. The same 5% `clock_drift` used,
#: so one number means one thing here.
DRIFT_FRACTION = 0.05

#: Target duration of one warmup batch between synchronises. Short enough that
#: warmup overshoots `warmup_ms` by at most this much, long enough that the
#: queue stays deep: settling with a synchronise every few kernels converges to
#: a partial-load plateau below what the real measurement induces (measured
#: 1575 vs 1980 MHz, calibrate.settle_clocks).
WARMUP_BATCH_MS = 25.0

#: Largest warmup batch, in calls. A bound on host memory for the enqueue, not
#: a tuning knob: a 1 us kernel reaches it at 10 ms per batch.
WARMUP_BATCH_MAX_CALLS = 10_000

#: How often the background sampler polls the clock during the trials. NVML
#: costs tens of microseconds per read; at 50 ms a 600 ms cell yields ~12
#: samples and the poll thread is asleep 99.9% of the time.
CLOCK_POLL_SECONDS = 0.05

#: Fallback when the device cannot be queried. Prefer flush_mb_for_device().
DEFAULT_FLUSH_MB = 256


def flush_mb_for_device(multiple: float = 4.0, minimum_mb: int = 128) -> int:
    """Flush buffer sized from the device's actual L2, not a magic constant.

    A single pass over a buffer only slightly larger than L2 does not reliably
    evict it: replacement is not perfect LRU, and Hopper partitions its L2. The
    previous fixed 128 MB was only 2.1x an H200's 60 MiB, which is tight;
    Triton's do_bench uses 256 MB against smaller caches. 4x is comfortable
    without wasting much memory.
    """
    try:
        l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    except (RuntimeError, AssertionError):
        return DEFAULT_FLUSH_MB
    if not l2:
        return DEFAULT_FLUSH_MB
    return max(minimum_mb, int(l2 * multiple / 2 ** 20))


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device; timing must run on the GPU box")


class TimingRefused(RuntimeError):
    """`time_kernel` cannot produce a measurement here and will not fake one.

    Raised when CUDA is absent and the caller has not injected the fakes that
    stand in for it (events, clock sampler, and the flusher when flushing),
    or when an argument makes the measurement meaningless (a zero warmup). A
    named exception so a script can tell "no GPU" apart from "the kernel
    crashed", which both used to arrive as RuntimeError.
    """


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
        # The ARCHITECTURE, not just the card name. Every consumer that needed
        # sm80-vs-sm90 -- whether wgmma can be emitted at all, whether torch's
        # grouped_mm reaches its CUTLASS sm90/sm100 kernel -- had to infer it
        # from gpu_name, and gpu_name is a marketing string. props.major/minor
        # is exactly what torch.cuda.get_device_capability returns, which is
        # the pair torch_grouped_mm's own support check already tests against.
        "sm_capability": ".".join(
            str(v) for v in torch.cuda.get_device_capability(index)),
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
            except Exception:  # noqa: BLE001
                # Deliberately broad. torch routes this through pynvml, which
                # raises ModuleNotFoundError when absent and its own
                # NVMLError subclasses on vGPU or restricted containers.
                # A narrow clause here let those escape into run_sweep, where
                # every cell would be recorded as a crash and the sweep would
                # produce no rows at all.
                pass
        vals = _nvidia_smi("clocks.current.sm,temperature.gpu")
        if not vals:
            return cls(0, 0)
        try:
            sm, temp = (int(float(v)) for v in vals[0].split(","))
        except (ValueError, IndexError):
            return cls(0, 0)
        return cls(sm, temp)


def clock_drift(start: ClockState, end: ClockState) -> tuple[float, bool]:
    """Percent drop in SM clock across a cell, and whether it looks throttled.

    THE LEGACY FLAG, kept because the driver's rows carry it. It fires only on
    a drop, and when its two samples are taken after synchronises (as
    `driver.py` does) it measures whether the start sample caught the idle
    boost, not whether the cell throttled: a card at 1500 MHz for the whole
    cell against a 1980 MHz roof passes with drift 0.0. `time_kernel` replaces
    it with `clock_flags`, which tests LEVEL against a reference and DRIFT in
    both directions on samples taken under load.
    """
    if start.sm_clock_mhz <= 0:
        return 0.0, False
    drift = (start.sm_clock_mhz - end.sm_clock_mhz) / start.sm_clock_mhz * 100.0
    return drift, drift > 5.0


# --------------------------------------------------------------------------
# L2
# --------------------------------------------------------------------------

class L2Flusher:
    """Evicts L2 between timed iterations by touching a buffer larger than it.

    MEASURED HAZARD, H200 SXM, 2026-08-22. At microsecond scale the flush can
    make a kernel FASTER: a 4 MB kernel ran 13.70 us unflushed and 7.07 us
    flushed. No cache effect explains that. The flush is enough work to hold
    clocks up, while a loop of ~10 us kernels lets the GPU settle.

    So at small cell sizes the l2_flush axis is confounded with clock state, and
    that is precisely the small-batch decode regime this project targets. Do not
    read a flushed/warm delta as a cache effect without checking the
    sm_clock_start/end columns, which are recorded per timing mode for this
    reason. The confound shrinks as the kernel lengthens.

    WHAT IT DOES NOT TOUCH. There is no user-level instruction to invalidate
    L2 from CUDA, so this evicts by capacity: read a buffer several times L2 and
    every line ends up holding flush data. It does NOT reset DRAM row buffers or
    the TLB, and it does not need to touch L1 or shared memory: those are per-SM
    and are invalidated at every kernel launch, so each launch already starts
    with a cold L1. The exception is a persistent kernel, where one launch spans
    many logical iterations and L1 state does carry across them.

    The flush READS rather than writes. A write flush (the common `buf.zero_()`
    idiom) leaves up to a full L2 of dirty lines, and those writebacks land
    inside the NEXT timed interval, stealing roughly 11 microseconds of HBM
    bandwidth on a 50 MiB L2 at 4.8 TB/s. Irrelevant for a millisecond kernel,
    a 10-30% inflation on a sub-100-microsecond span, which is exactly the
    small-batch regime this project studies.
    """

    def __init__(self, megabytes: int | None = None, device: str = "cuda",
                 mode: str = "read"):
        if megabytes is None:
            megabytes = flush_mb_for_device()
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

    THE INJECTION SEAM. `time_kernel` takes an `events` factory with this
    class's shape: `events(n)` returns an object with `starts[i].record()`,
    `ends[i].record()`, `synchronize()` and `elapsed(n) -> list[float]`
    (milliseconds per pair). Priming happens in the constructor, so "the pairs
    were primed before the timed region" is the same statement as "the factory
    was called before the first start record". A fake that scripts `elapsed`
    exercises every branch of the instrument off-GPU (tests/test_timing.py).
    """

    def __init__(self, n: int):
        self.starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        self.ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for e in (*self.starts, *self.ends):
            e.record()          # forces creation of the underlying cudaEvent
        torch.cuda.synchronize()

    def synchronize(self) -> None:
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


def _stats(samples: list[float]) -> tuple[float, float, float, float]:
    """p50, p90, min, std of a sample list. One definition, three timers."""
    s = sorted(samples)
    n = len(s)
    p50 = statistics.median(s)
    p90 = s[min(n - 1, int(round(0.9 * (n - 1))))]
    return p50, p90, s[0], (statistics.stdev(s) if n > 1 else 0.0)


def _summarise(samples: list[float], **meta) -> TimingResult:
    p50, p90, lo, std = _stats(samples)
    return TimingResult(
        ms_p50=p50,
        ms_p90=p90,
        ms_min=lo,
        ms_std=std,
        jitter_p90_over_p50=(p90 / p50) if p50 > 0 else float("inf"),
        samples=len(samples),
        **meta,
    )


def _timed_trials(fn: Callable[[], None], iters: int, trials: int, events,
                  flush: Callable[[], None] | None) -> list[float]:
    """The queue-deep loop every timer in this module runs.

    Per trial: `iters` calls enqueued back to back, each bracketed by its own
    pre-primed event pair, the flush (when there is one) enqueued BEFORE the
    start record so it sits outside the interval, and ONE synchronise at the
    end. The host runs ahead of the GPU for the whole trial, so a start event
    is timestamped when the previous work finishes, not when the host got
    round to enqueueing the kernel. That is the difference between this loop
    and the per-iteration-synchronise `time_call` the ladders used to carry,
    and it is worth 0.18-0.30 ms per call on a fused_experts cell.

    Why per-iteration pairs rather than one pair around the trial: with the
    flush inside a single pair its ~50 us of L2-sized reads would be inside
    the measurement, and subtracting it back out would be a model, not a
    measurement. One pair per call keeps the flush out by construction and
    leaves `iters * trials` samples for the percentiles instead of `trials`.
    """
    samples: list[float] = []
    for _ in range(trials):
        for i in range(iters):
            if flush is not None:
                flush()
            events.starts[i].record()
            fn()
            events.ends[i].record()
        events.synchronize()
        samples.extend(events.elapsed(iters))
    return samples


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
    """Per-iteration CUDA-event timing of an eagerly launched callable.

    Runs the same `_timed_trials` loop as `time_kernel`, so the two agree on
    the measured interval; what differs is around it. This warms up for a
    COUNT of calls (`warmup`) and calibrates from one isolated call, and reads
    no clock; the driver depends on those semantics and on `TimingResult`, so
    they are unchanged. New consumers use `time_kernel`, which warms up for a
    duration of sustained load and samples the clock during the trials.
    """
    require_cuda()
    flusher = L2Flusher(flush_mb if l2_flush else 0, mode=flush_mode)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if iters is None:
        iters = calibrate_iters(fn, target_ms)
    samples = _timed_trials(fn, iters, trials, _EventPairs(iters), flusher.flush)

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
    samples = _timed_trials(graph.replay, iters, trials, _EventPairs(iters),
                            flusher.flush)

    return _summarise(samples, warmup=warmup, iters=iters, trials=trials,
                      l2_flush=l2_flush, cuda_graph=True,
                      flush_mb=flusher.megabytes, flush_mode=flush_mode)


# --------------------------------------------------------------------------
# the one instrument
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WarmupReport:
    """What the sustained-load warmup actually did, for the row and the tests.

    `per_call_ms` is the LAST batch's mean per call, measured queue-deep with
    the governor already responding to the load. It is what `time_kernel`
    calibrates `iters` from: an isolated single call (what `calibrate_iters`
    measures) includes launch latency on an idle GPU and reads a 5 us kernel
    as 15, so an iteration count sized from it lands at a third of the target.
    """

    delivered_ms: float
    calls: int
    batches: int
    per_call_ms: float


def warm_until(fn: Callable[[], None], warmup_ms: float, events,
               batch_ms: float = WARMUP_BATCH_MS) -> WarmupReport:
    """Run `fn` under sustained load until `warmup_ms` of GPU time has passed.

    A COUNT of warmup calls is the wrong unit, and the ladders that compared
    cells warmed at 5 against cells warmed at 20 were comparing clock states:
    a 1 ms kernel needs hundreds of calls before the governor reacts, a 30 ms
    GEMM needs one. So the warmup is a duration of delivered GPU time, measured
    with the same events the trials use. The first batch is one call, which
    sizes the rest to about `batch_ms` each so the queue stays deep between
    synchronises; the loop stops after the batch that carries the total past
    `warmup_ms`, so the overshoot is bounded by one batch.
    """
    if warmup_ms <= 0:
        raise TimingRefused(
            f"warmup_ms={warmup_ms}: the instrument warms up for a duration of "
            "sustained load and calibrates its iteration count from that load; "
            "a zero warmup measures a cold governor and calibrates from nothing")
    pair = events(1)
    delivered = 0.0
    calls = 0
    batches = 0
    batch = 1
    per_call = 0.0
    while delivered < warmup_ms:
        pair.starts[0].record()
        for _ in range(batch):
            fn()
        pair.ends[0].record()
        pair.synchronize()
        ms = max(pair.elapsed(1)[0], 1e-4)
        delivered += ms
        calls += batch
        batches += 1
        per_call = ms / batch
        batch = max(1, min(WARMUP_BATCH_MAX_CALLS, int(batch_ms / per_call)))
    return WarmupReport(delivered_ms=delivered, calls=calls, batches=batches,
                        per_call_ms=per_call)


def iters_for(per_call_ms: float, target_ms: float, lo: int = 10,
              hi: int = 2000) -> int:
    """Iterations per trial so that one trial lasts about `target_ms`."""
    return max(lo, min(hi, int(target_ms / max(per_call_ms, 1e-4))))


class BackgroundClockSampler:
    """Polls the SM clock from a thread while the calling thread keeps the GPU busy.

    The context-manager shape is the injection seam: `time_kernel` enters it
    before the first trial and exits it after the last synchronise, and reads
    `.samples` afterwards. A fake with the same shape returns a scripted trace.

    The first sample is taken `poll_seconds` AFTER entering, not at entry.
    Entry happens right after the events were primed, which ends in a
    synchronise, and a sample at that instant is the idle-boost reading the
    old flag was fooled by. Waiting first puts every sample inside the trials.
    """

    def __init__(self, sample: Callable[[], ClockState] = ClockState.sample,
                 poll_seconds: float = CLOCK_POLL_SECONDS):
        self._sample = sample
        self._poll = poll_seconds
        self._stop = threading.Event()
        self._got: list[ClockState] = []
        self._thread: threading.Thread | None = None
        self.samples: tuple[ClockState, ...] = ()

    def _run(self) -> None:
        while not self._stop.wait(self._poll):
            self._got.append(self._sample())

    def __enter__(self) -> BackgroundClockSampler:
        self._stop.clear()
        self._got = []
        self._thread = threading.Thread(target=self._run, name="clock-sampler",
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.samples = tuple(self._got)


def clock_flags(load_mhz: float | None, start_mhz: float | None,
                end_mhz: float | None, reference_mhz: float | None,
                ) -> tuple[bool | None, bool | None]:
    """LEVEL and DRIFT verdicts on under-load clock samples. Pure.

    LEVEL: the loaded clock is at least `LEVEL_FRACTION` of `reference_mhz`,
    the clock the roof was measured at. None when there is no reference or no
    load sample: the flag is a comparison and half a comparison is not a
    verdict.

    DRIFT: first and last under-load samples agree within `DRIFT_FRACTION` of
    the first, in either direction. A drop is throttling during the trials; a
    rise is a warmup that did not reach the operating point. Both mean the
    samples were not taken at one clock and the median is a blend.
    """
    level: bool | None
    drift: bool | None
    if load_mhz is None or reference_mhz is None or reference_mhz <= 0:
        level = None
    else:
        level = load_mhz >= LEVEL_FRACTION * reference_mhz
    if start_mhz is None or end_mhz is None or start_mhz <= 0:
        drift = None
    else:
        drift = abs(start_mhz - end_mhz) / start_mhz <= DRIFT_FRACTION
    return level, drift


@dataclass(frozen=True)
class KernelTiming:
    """One cell's measurement under `TIMING_BASIS`, with the state it ran in.

    The clock fields are Optional and None means "not determined", never
    "zero": NVML absent, a container that forbids it, or a trial too short for
    the poller to land a sample. `clock_note` says which. `clock_level_ok` and
    `clock_drift_ok` are the two verdicts `clock_flags` documents; a consumer
    that filters rows must test BOTH, since the old single drop-only flag is
    the defect this record exists to replace.
    """

    ms_p50: float
    ms_p90: float
    ms_min: float
    ms_std: float
    iters: int
    trials: int
    warmup_ms: float
    l2_flush: bool
    sm_clock_load_mhz: float | None
    sm_clock_start_mhz: float | None
    sm_clock_end_mhz: float | None
    clock_level_ok: bool | None
    clock_drift_ok: bool | None
    samples: int
    warmup_calls: int
    flush_mb: int
    clock_samples: int
    clock_note: str = ""
    instrument: str = TIMING_BASIS


def time_kernel(
    fn: Callable[[], None],
    *,
    warmup_ms: float,
    target_ms: float = 200.0,
    trials: int = 3,
    l2_flush: bool = True,
    reference_clock_mhz: float | None = None,
    clock_sampler=None,
    events=None,
    flusher=None,
) -> KernelTiming:
    """Time `fn` the one way this repository times anything it publishes.

    In order: warm up under sustained load until `warmup_ms` of GPU time has
    been delivered (`warm_until`); size `iters` so a trial lasts `target_ms`
    from the warmup's own queue-deep per-call time; prime one event pair per
    iteration; start the clock poller; run `trials` trials of `_timed_trials`
    (flush before each call when `l2_flush`, one synchronise per trial); stop
    the poller; summarise the `iters * trials` samples the way `time_eager`
    does and the clock samples the way `clock_flags` does.

    `reference_clock_mhz` is the clock the roof was measured at
    (`calibrate.LoadedClock.median_mhz`); without it the LEVEL flag is None,
    because a level is relative to something and this function will not
    invent the something.

    OFF-GPU. Refuses with `TimingRefused` unless every CUDA-touching part is
    injected: `events` (a factory with `_EventPairs`'s shape), `clock_sampler`
    (a context manager with `BackgroundClockSampler`'s shape) and, when
    `l2_flush`, `flusher` (an object with `flush()` and `megabytes`). With all
    of them present the full logic runs against the fakes; that is how the
    tests plant every PASS and FAIL branch of both clock flags.
    """
    if trials < 1:
        raise TimingRefused(f"trials={trials}: a measurement needs at least one trial")
    on_gpu = torch.cuda.is_available()
    missing = [name for name, given in (("events", events),
                                        ("clock_sampler", clock_sampler),
                                        ("flusher", flusher if l2_flush else True))
               if given is None]
    if missing and not on_gpu:
        raise TimingRefused(
            "no CUDA device, and no fake injected for " + ", ".join(missing)
            + "; time_kernel measures a GPU or runs against injected fakes, it "
            "does not invent numbers")
    if events is None:
        events = _EventPairs
    if clock_sampler is None:
        clock_sampler = BackgroundClockSampler()
    if l2_flush and flusher is None:
        flusher = L2Flusher(flush_mb_for_device())
    flush = flusher.flush if l2_flush else None
    flush_mb = int(flusher.megabytes) if l2_flush else 0

    warm = warm_until(fn, warmup_ms, events)
    iters = iters_for(warm.per_call_ms, target_ms)
    pairs = events(iters)
    with clock_sampler as poller:
        samples = _timed_trials(fn, iters, trials, pairs, flush)
    clocks = [c.sm_clock_mhz for c in poller.samples]
    usable = [c for c in clocks if c > 0]

    if usable:
        load: float | None = float(statistics.median(usable))
        start: float | None = float(usable[0])
        end: float | None = float(usable[-1])
        note = ""
        if len(usable) < 2:
            note = ("one usable clock sample during the trials; drift is not "
                    "determinable from one point")
            end = None
        if reference_clock_mhz is None:
            note = (note + "; " if note else "") + \
                "no reference clock given, level not determinable"
    else:
        load = start = end = None
        note = (f"no usable SM clock sample during the trials ({len(clocks)} "
                "polled, all zero or none landed); clocks not determinable")
    level_ok, drift_ok = clock_flags(load, start, end, reference_clock_mhz)

    p50, p90, lo, std = _stats(samples)
    return KernelTiming(
        ms_p50=p50, ms_p90=p90, ms_min=lo, ms_std=std,
        iters=iters, trials=trials, warmup_ms=warm.delivered_ms,
        l2_flush=l2_flush, sm_clock_load_mhz=load, sm_clock_start_mhz=start,
        sm_clock_end_mhz=end, clock_level_ok=level_ok, clock_drift_ok=drift_ok,
        samples=len(samples), warmup_calls=warm.calls, flush_mb=flush_mb,
        clock_samples=len(usable), clock_note=note,
    )
