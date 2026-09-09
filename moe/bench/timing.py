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

THE RESIDUAL THE EVENT PAIR CANNOT REMOVE, with its size and its sign. Each
interval holds the kernel plus the GPU-side inter-kernel gap between the
previous end record and this start record: ~1-3 us on Hopper. It is inside
every interval on both sides of a ratio and it does NOT cancel, because the
two sides are different lengths: on the roof's 1.4 ms GEMM it is ~0.2%, on a
50 us cell it is 2-6%. Direction: the cell reads SLOWER than the kernel, so
every fraction-of-roof this module feeds is UNDERSTATED by that amount. That
is the conservative direction, and it is a bias, not noise, so it is stated
here rather than absorbed into ms_std.

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

`time_kernel` is the instrument that replaces both, and `TIMING_BASIS` is its
name. The ladder scripts moved onto it in the phase after this header was
written (every `scripts/*.py` that times a kernel calls `time_kernel`; the one
remaining `def time_call`, in `block_m_crossing_sweep.py`, is the retired path
kept for reproducing old rows). A row is comparable with the roof only if it
carries `TIMING_BASIS`: a consumer on the instrument writes the string into
every row it produces, so a reader can tell at a glance which apparatus made
a number, and a row without it is a row from before the fix. Change the
string when the instrument changes in a way that moves numbers; never
otherwise.

v3 (2026-09-03) is such a change: the warmup now runs the FLUSHED loop when
the trials are flushed (see `warm_until`), which moves the operating point a
short cell is measured at, and `iters` is sized from a flushed per-iteration
probe rather than from the warm-L2 batch mean. `LOOP_SHAPE` below binds the
string to the loop's shape, and `tests/test_timing.py` pins both, so a change
to the flush position, the iteration cap or the warmup's shape fails the
suite until the string is bumped with it.

v4 (2026-09-09) is the next such change: the warmup does not stop at
`warmup_ms`, it stops when the CLOCK has stopped moving (two consecutive NVML
reads within one 15 MHz step, capped at three times `warmup_ms`), because the
H200 session's 135 DRIFT verdicts were all the governor still settling on the
first cell of a rep. That moves the operating point every short cell is
measured at, so the string moves with it.

THE HOST-BOUND CASE IS DETECTED, NOT ASSUMED AWAY
-------------------------------------------------
A queue-deep loop only measures the GPU while the queue is deep. When the
host takes longer to enqueue one call than the GPU takes to run it (the B0
evidence: vLLM's fused_experts at T=1 reads 0.18 ms per call through a
Python launcher and 0.04 ms as a graph replay) the queue drains, every start
event is timestamped on an idle GPU, and the interval carries the host's
enqueue time again, exactly as `time_call` did. No event structure can fix
that; it is a property of the callable. What the instrument can do is notice:
each trial's enqueue loop is wall-clocked against the whole trial, and a trial
whose GPU had less than `HOST_BOUND_BACKLOG_ITERS` iterations of work left
when the host finished enqueueing is marked `host_bound`. A host-bound number
is an upper bound on the kernel time, and the row says so.

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
sets two flags: LEVEL (`clock_level_ok`: the loaded clock is inside the band
[`LEVEL_FRACTION`, `LEVEL_HIGH_FRACTION`] around the reference the roof was
measured at, snapped to the `CLOCK_STEP_MHZ` grid, and `clock_level_side`
names which way it left the band) and DRIFT (`clock_drift_ok`: first and last
under-load samples agree within `DRIFT_FRACTION`, in either direction).

LEVEL IS A RECORD, DRIFT IS THE EXCLUSION (2026-09-09)
------------------------------------------------------
Until 2026-09-09 a LEVEL failure on the LOW side excluded the cell. The first
H200 session said what that rule actually is: across 750 cells the under-load
clock is set PER TILE by the kernel's own power draw under the 700 W cap.
BM=128/N=64 holds a median 1395 MHz over 215 cells, BM=64/G=1 1358, BM=256
1650, memory-shaped work 1950-1980; the calibration's own 8192^3 GEMM sits at
1485 MHz at 691 W, which is near the LOW end of what dense tensor work does on
this card rather than the middle of anything. So LEVEL-LOW named a TILE, not a
throttle: it removed the study's two primary tiles from measurability on this
card and removed nothing anywhere else. The verdict and the side are still
computed and still written on every row, because "this tile runs 6% under the
ruler's GEMM" is a finding; they exclude nothing. The exclusion is DRIFT
alone, and every one of the session's 135 DRIFTs was the governor settling on
the FIRST cell of a rep after a workload change, which is a property of the
INSTRUMENT and is fixed below rather than gated around.

Two more things follow from the same physics. The band edges snap to the
15 MHz NVML step (`snap_to_clock_step`): 0.95 x 1485 = 1410.75 sits inside one
step, so 1410 read LOW and 1425 read level for timings 0.1% apart, and 23 of
the session's 148 LOW verdicts sat within that step of the edge. And the roof
is normalised PER ROW: `sm_clock_load_mhz` is a row column, so a
fraction-of-compute-roof can also be scored against the roof AT THE CLOCK THE
CELL RAN (`roofline.roof_at_clock`, applied by the driver as
`roof_at_cell_clock_tflops`). Both fractions are on every row. The FIXED-roof
fraction is the compute-bound gate input, because the GEMM and every cell ran
under the same 700 W cap and it is therefore the fair delivered-throughput
comparison; the own-clock fraction is issue efficiency, carried beside it.

THE WARMUP SETTLES ON THE CLOCK, NOT ONLY ON A DURATION (v4)
------------------------------------------------------------
`warmup_ms` of delivered GPU time reaches the governor's fast response but does
not establish that it has STOPPED moving, and the session's DRIFTs were by
position exactly that: the first cell of a rep after the workload changed
(1875->1725, and 1560->1650 three times). So after `warmup_ms` the warmup keeps
running the trials' own loop until two consecutive NVML reads agree within one
`CLOCK_STEP_MHZ` step, capped at `SETTLE_CAP_MULTIPLE` x `warmup_ms` of
delivered time; `settle_ms` and the extra calls are on the record, and a DRIFT
that survives it names its direction (settling upward, dropping, oscillating).
The probe that sizes `iters` runs AFTER the settle, so the iteration count is
sized at the operating point the trials will run at.
"""
from __future__ import annotations

import math
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

#: The name of the instrument `time_kernel` implements. Written into every row
#: it produces. Bump the version suffix when a change would move a published
#: number; the reader compares this string, not a commit hash. v3: flushed
#: warmup and a flushed per-iteration probe for `iters` (see `warm_until`).
#: v4: the warmup runs on past `warmup_ms` until the SM clock has settled
#: (`warm_until`'s settle loop), so a cell is measured at the operating point
#: the governor stopped at rather than wherever it had reached at 300 ms.
TIMING_BASIS = "queue-deep/l2-flush/settled-warmup/clock-under-load/v4"

#: The name the RETIRED timers stamp on their `TimingResult`. `time_eager` and
#: `time_graph` run the same `_timed_trials` loop as `time_kernel` and differ
#: only around it (the seven differences are listed on `time_eager`), and a
#: record that did not say which apparatus produced it was the defect the
#: v5 schema was cut for. Deliberately NOT `TIMING_BASIS`: the calibration's
#: roof is measured through `time_eager`, and `scripts/calibrate_hardware.py`
#: asserts that it never claims the instrument's name.
RETIRED_TIMER_BASIS = "time_eager+time_graph/count-warmup/isolated-iters/no-clock"

#: The SM clock's quantum on this hardware. NVML reports Hopper SM clocks on a
#: 15 MHz grid: every raw reading in the H200 session's 750 cells is a multiple
#: of it, and the 23 distinct off-grid values in that corpus are all half
#: steps, medians of an even number of readings. So a threshold that lands
#: between two grid points is a threshold no reading can sit on: 0.95 x 1485 =
#: 1410.75 put a 1410 cell outside the band and a 1425 cell inside it for
#: timings 0.1% apart, and 23 of the session's 148 LOW verdicts sat in that one
#: step. Every band edge is snapped to this grid (`snap_to_clock_step`), and it
#: is also the agreement two consecutive warmup reads must reach before the
#: clock counts as settled.
CLOCK_STEP_MHZ = 15.0

#: LEVEL flag, LOW edge: the SM clock sampled under load must be at least this
#: fraction of the reference clock (the one the roof was measured at) for the
#: cell to be INSIDE the band. 0.95 because the H200's compute plateau moves
#: 1455-1515 MHz across sessions of one card (calibrate.py), a 4% band, and an
#: edge inside that band would fire on the card's own session-to-session noise.
#:
#: A CELL OUTSIDE THE BAND IS STILL MEASURED. Since 2026-09-09 neither side
#: excludes anything: the side is a record of which tile family the cell is in
#: (BM=128/N=64 sits at 1395 MHz under the 700 W cap, 6% under the ruler's own
#: GEMM, in every rep and every deep tread), and the exclusion is DRIFT.
LEVEL_FRACTION = 0.95

#: LEVEL flag, HIGH edge, the same 5% the other way. Symmetric because the
#: session-to-session plateau noise the low edge was sized for is symmetric
#: about the reference. A clock ABOVE the band is a cell that ran at a higher
#: tensor-core issue rate than the roof assumed (a memory-shaped cell that
#: boosted to 1980 against a power-limited 1515 roof), and its fixed-roof
#: fraction is inflated by the ratio; before 2026-09-03 that side passed.
LEVEL_HIGH_FRACTION = 1.05

#: DRIFT flag: first and last under-load samples may differ by at most this
#: fraction of the first, in EITHER direction. The same 5% `clock_drift` used,
#: so one number means one thing here. THIS is the exclusion: a cell whose
#: clock moved while the trials ran was not measured at one operating point,
#: and its median is a blend of two.
DRIFT_FRACTION = 0.05

#: How much delivered warmup the settle loop may add, as a multiple of
#: `warmup_ms`. The loop keeps running the trials' own work until two
#: consecutive clock reads agree within `CLOCK_STEP_MHZ`; the cap is what stops
#: a genuinely oscillating card from warming forever, and 3x of a 300 ms warmup
#: is 900 ms, which is under a second per cell against the 33-minute session
#: the DRIFTs came from. Reaching the cap is recorded, not silently accepted:
#: `WarmupReport.settled` is False and the note says so.
SETTLE_CAP_MULTIPLE = 3.0

#: Target duration of one warmup batch between synchronises. Short enough that
#: warmup overshoots `warmup_ms` by at most this much, long enough that the
#: queue stays deep: settling with a synchronise every few kernels converges to
#: a partial-load plateau below what the real measurement induces (measured
#: 1575 vs 1980 MHz, calibrate.settle_clocks).
WARMUP_BATCH_MS = 25.0

#: Largest warmup batch, in calls. A bound on host memory for the enqueue, not
#: a tuning knob: a 1 us kernel reaches it at 10 ms per batch.
WARMUP_BATCH_MAX_CALLS = 10_000

#: How often the background sampler polls the clock during the trials. The
#: sampler reads through torch's NVML bindings only (`nvml_clock_reader`),
#: tens of microseconds per read, so at 50 ms a 600 ms cell yields ~12 samples
#: and the poll thread is asleep 99.9% of the time. It never forks nvidia-smi
#: inside the timed region: that costs tens of milliseconds and an NVML init
#: per read, on the same host the enqueue thread needs to keep the queue deep.
#: Where NVML is unavailable the record says so and carries no clock.
CLOCK_POLL_SECONDS = 0.05
#: The shortest poll `clock_poll_for` will ask of NVML. A read is tens of
#: microseconds, so 5 ms stays far inside `CLOCK_POLL_BUDGET_FRACTION`.
CLOCK_POLL_FLOOR_SECONDS = 0.005

#: The sampler's per-read cost may take at most this fraction of the poll
#: interval before the record notes that the poller was competing with the
#: enqueue thread for the host. 10% of 50 ms is 5 ms, two orders above an
#: NVML read; only a slow or blocking reader reaches it.
CLOCK_POLL_BUDGET_FRACTION = 0.10

#: Fewest usable under-load samples a clock median is taken from. The same
#: floor `calibrate.clock_under_load` applies to the reference clock, so the
#: cell's `sm_clock_load_mhz` and the roof's `median_mhz` are one kind of
#: number: two cannot disagree with each other, and one is not a median.
#: DRIFT needs only a first and a last, so it is reported from two.
CLOCK_SAMPLE_FLOOR = 3

#: A trial is host-bound when the GPU had fewer than this many iterations of
#: work queued at the instant the host finished enqueueing it. A drained queue
#: leaves at most the last kernel in flight (one iteration); a queue that was
#: deep for the whole trial leaves the accumulated backlog, which is many. Two
#: is the smallest count that separates the two by construction rather than
#: by a chosen fraction.
HOST_BOUND_BACKLOG_ITERS = 2

#: Fallback when the device cannot be queried. Prefer flush_mb_for_device().
DEFAULT_FLUSH_MB = 256

#: Iterations of the flushed per-iteration probe `warm_until` runs after the
#: warmup, queue-deep, to size `iters` from. Enough samples that one slow
#: interval does not size the trials; small enough (200 x ~50 us of flush on
#: an H200) that the probe is a rounding error on the warmup it follows.
WARMUP_PROBE_CALLS = 200

#: THE LOOP'S SHAPE, bound to `TIMING_BASIS`. Every entry is a property of the
#: loop that moves numbers if it changes: where the flush sits relative to the
#: start record, the iteration cap `iters_for` applies, the warmup batch
#: length, whether the warmup runs flushed, the L2 multiple the flush buffer is
#: sized by, and where `iters` is sized from. `tests/test_timing.py` pins this
#: dict AND the basis string, and derives the live values from the code (the
#: `iters_for` signature, the source order of `_timed_trials`), so a change to
#: any of them fails the suite until the string is bumped with it. The knobs
#: a caller can set (warmup_ms, target_ms, trials, flush_mb) are row columns
#: and need no such guard: pooling across them is visible in the data.
LOOP_SHAPE = {
    "flush_position": "before start record",
    "iters_hi": 2000,
    "iters_lo": 10,
    "warmup_batch_ms": 25.0,
    "warmup_flushed": True,
    "iters_sized_from": "flushed per-iteration probe",
    "flush_l2_multiple": 4.0,
    "warmup_settles_on_clock": True,
    "settle_step_mhz": 15.0,
    "settle_cap_multiple": 3.0,
}


def snap_to_clock_step(mhz: float, step: float = CLOCK_STEP_MHZ) -> float:
    """A clock threshold moved onto the grid NVML actually reports on. Pure.

    Nearest grid point, halves up. A threshold between two grid points is a
    threshold no reading can equal, and the direction of the rounding decides
    a verdict for readings that differ by one step and time within 0.1% of
    each other; snapping to the nearest makes the edge itself a clock the card
    can report, so the verdict is about the reading and not about the
    arithmetic. Applied to both LEVEL edges by `level_side`.
    """
    if step <= 0:
        return float(mhz)
    return math.floor(float(mhz) / step + 0.5) * step


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


#: The readers `ClockState.sample` can have used, as `ClockState.source` names
#: them. `nvml` is torch's binding, tens of microseconds; `nvidia-smi` is a
#: forked process, tens of milliseconds plus an NVML init, and a sample that
#: came through it landed that long after the moment it was asked for.
CLOCK_SOURCE_NVML = "nvml"
CLOCK_SOURCE_NVIDIA_SMI = "nvidia-smi"
CLOCK_SOURCE_NONE = "none"


def nvml_power_w(device_index: int | None = None) -> float:
    """Board power in watts through torch's NVML binding, 0.0 when it cannot.

    `torch.cuda.power_draw` reports milliwatts and costs the same order as the
    clock read (tens of microseconds), so it can sit inside the same sample
    without perturbing what is being measured. 0.0 rather than a raise:
    nothing divides by it, it is recorded so a LOW clock can be told apart as
    a power-capped tile from a card in trouble, and a torch old enough not to
    have the binding should cost a row its watts and not its measurement.
    """
    try:
        return float(torch.cuda.power_draw(device_index)) / 1000.0
    except Exception:  # noqa: BLE001
        # As broad as the clock read beside it, and for the same reasons:
        # torch surfaces a missing pynvml as ModuleNotFoundError, NVML
        # refusals as its own NVMLError family, and an older torch as
        # AttributeError. None of them may stop a cell being timed.
        return 0.0


@dataclass(frozen=True)
class ClockState:
    sm_clock_mhz: int
    temp_c: int
    #: Which reader produced this sample. Defaults to NVML so the positional
    #: `ClockState(mhz, temp)` every fake and reader constructs keeps meaning
    #: what it meant; `sample()` sets it explicitly on every branch. It exists
    #: because a sample from the nvidia-smi fallback is not the same kind of
    #: number as one from NVML (see `calibrate.clock_under_load`, which refuses
    #: the fallback for the roof's reference), and a caller could not tell.
    source: str = CLOCK_SOURCE_NVML
    #: Board power at the instant of the clock read, W; 0.0 when the reader
    #: could not answer (no card ever draws zero, so the zero is unambiguous).
    #:
    #: READ AT THE SAME CALL as the clock, and that is the point of the field.
    #: The H200 session showed the under-load clock is set per tile by the
    #: kernel's own power draw under the 700 W cap (BM=128/N=64 1395 MHz, the
    #: calibration GEMM 1485 at 691 W), so a LOW clock is either a hungry tile
    #: at the cap or a card in trouble, and the two are only separable if the
    #: watts arrived beside the megahertz rather than from a later reading.
    #: `calibrate._power_draw_w` forks nvidia-smi and lands tens of
    #: milliseconds late; this comes through the same NVML path as the clock.
    power_w: float = 0.0

    @classmethod
    def sample(cls) -> ClockState:
        """Sample SM clock and temperature, and say which reader answered.

        Prefers torch's NVML bindings, which cost tens of microseconds. The
        nvidia-smi fallback below runs twice per timing mode, eight times per
        cell, and a fork that initialises NVML costs tens of milliseconds on
        Linux: tens of minutes of a large sweep spent on process startup.

        The fallback is kept for the idle-instant callers (the retired seam's
        two samples around a cell, the calibration's before/after pair) where a
        sample tens of milliseconds late is still an idle sample. It is NOT
        acceptable where the sample has to land while a queue is busy, and the
        `source` field is how such a caller tells: `clock_under_load` refuses a
        forked sample rather than publishing an idle reading as the clock the
        roof ran at, and `nvml_clock_reader` never forks at all.
        """
        if torch.cuda.is_available():
            try:
                return cls(int(torch.cuda.clock_rate()),
                           int(torch.cuda.temperature()),
                           source=CLOCK_SOURCE_NVML,
                           power_w=nvml_power_w())
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
            return cls(0, 0, source=CLOCK_SOURCE_NONE)
        try:
            sm, temp = (int(float(v)) for v in vals[0].split(","))
        except (ValueError, IndexError):
            return cls(0, 0, source=CLOCK_SOURCE_NONE)
        return cls(sm, temp, source=CLOCK_SOURCE_NVIDIA_SMI)


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
    """The RETIRED timers' record. Names its apparatus, as `KernelTiming` does.

    `instrument` is `RETIRED_TIMER_BASIS`, never `TIMING_BASIS`: the loop is
    the same `_timed_trials`, but the warmup, the iteration sizing and the
    absence of a clock are not, and a calibration row that said "the
    instrument" while it was measured by this would put the roof and the cells
    under one name for two apparatus. `ms_std` here, as in `KernelTiming`, is
    the dispersion of consecutive iterations in one thermal state: intra-run,
    not a between-replicate figure.
    """

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
    instrument: str = RETIRED_TIMER_BASIS


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


@dataclass(frozen=True)
class TrialWall:
    """Host wall clocks of one trial, the raw material of the host-bound verdict.

    `enqueue_s` is the perf_counter span of the enqueue loop alone; `wall_s`
    runs from the same start to the return of the synchronise. Their difference
    is the GPU's backlog at the instant the host finished: how long the GPU
    kept working on already-enqueued iterations after the host had nothing
    left to give it. That backlog, not the event intervals, is what tells a
    deep queue from a drained one, because a drained queue lets host time INTO
    the intervals and they cannot then witness against themselves.
    """

    enqueue_s: float
    wall_s: float


def _timed_trials(fn: Callable[[], None], iters: int, trials: int, events,
                  flush: Callable[[], None] | None,
                  ) -> tuple[list[float], list[TrialWall]]:
    """The queue-deep loop every timer in this module runs.

    Per trial: `iters` calls enqueued back to back, each bracketed by its own
    pre-primed event pair, the flush (when there is one) enqueued BEFORE the
    start record so it sits outside the interval, and ONE synchronise at the
    end. As long as each call costs the host less to enqueue than it costs the
    GPU to run, the queue deepens and a start event is timestamped when the
    previous work finishes, not when the host got round to enqueueing the
    kernel. That is the difference between this loop and the
    per-iteration-synchronise `time_call` the ladders used to carry, worth
    0.18-0.30 ms per call on a fused_experts cell.

    The condition is not always met. A callable whose host side is slower
    than its GPU side (a Python launcher over many small kernels at T=1)
    drains the queue no matter how the events are arranged, and the interval
    then holds host time again. This loop cannot prevent that; it records the
    wall clocks that let `host_bound_verdict` detect it, per trial.

    Why per-iteration pairs rather than one pair around the trial: with the
    flush inside a single pair its ~50 us of L2-sized reads would be inside
    the measurement, and subtracting it back out would be a model, not a
    measurement. One pair per call keeps the flush out by construction and
    leaves `iters * trials` samples for the percentiles instead of `trials`.
    """
    samples: list[float] = []
    walls: list[TrialWall] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for i in range(iters):
            if flush is not None:
                flush()
            events.starts[i].record()
            fn()
            events.ends[i].record()
        t1 = time.perf_counter()
        events.synchronize()
        t2 = time.perf_counter()
        samples.extend(events.elapsed(iters))
        walls.append(TrialWall(enqueue_s=t1 - t0, wall_s=t2 - t0))
    return samples, walls


def host_bound_verdict(walls: list[TrialWall], iters: int,
                       ) -> tuple[bool | None, float | None, float | None, str]:
    """Was any trial host-bound? Pure.

    Returns (verdict, enqueue_ms, backlog_iters, note). `backlog_iters` is the
    SMALLEST per-trial backlog in units of the trial's own per-iteration wall,
    the very number the verdict thresholds against `HOST_BOUND_BACKLOG_ITERS`,
    written out so a reader can see how far from the boundary a row sat
    rather than only which side of it. The verdict is binary and flips only
    when the host is slower than the GPU; analytically the backlog is about
    N(g - h) + g for a GPU-bound trial, so the ratio is what shows a callable
    that is nearly host-bound (backlog of 3 on a 2 threshold) beside one that
    is nowhere near (backlog of 200).

    A trial's GPU backlog at the end of its enqueue loop is `wall_s -
    enqueue_s`. If the host was slower than the GPU the backlog is at most the
    last iteration still in flight; if the host was faster it is everything
    the GPU has not yet reached, which grows through the trial. The verdict
    compares the backlog with `HOST_BOUND_BACKLOG_ITERS` iterations of the
    trial's own mean per-iteration wall (`wall_s / iters`, which bounds the
    GPU's per-iteration time from above, flush included). Any host-bound
    trial marks the record: its samples are pooled with the others.

    Why not compare the host wall with the event intervals, which is the
    obvious gate: when the queue drains the start event fires on an idle GPU
    and the interval absorbs the host's enqueue time, so the intervals sum to
    MORE than the host wall in exactly the case to be caught. The intervals
    are the contaminated witness.

    `enqueue_ms` is the median per-trial host wall of the enqueue loop, in
    milliseconds, reported so a reader can divide by `iters` and see the
    host's per-call cost beside the GPU's. None with a reason when no wall
    time elapsed (only a fake can manage that) or there were no trials.
    """
    if not walls or iters < 1:
        return None, None, None, "no trials; host-bound not determinable"
    if any(w.wall_s <= 0 for w in walls):
        return None, None, None, ("a trial spanned no wall time; host-bound "
                                  "not determinable")
    bound = []
    ratios = []
    for w in walls:
        backlog = w.wall_s - w.enqueue_s
        ratios.append(backlog / (w.wall_s / iters))
        bound.append(backlog < HOST_BOUND_BACKLOG_ITERS * (w.wall_s / iters))
    enqueue_ms = float(statistics.median(w.enqueue_s for w in walls)) * 1e3
    backlog_iters = float(min(ratios))
    if any(bound):
        n = sum(bound)
        return True, enqueue_ms, backlog_iters, (
            f"host-bound: in {n} of {len(walls)} trials the GPU had fewer than "
            f"{HOST_BOUND_BACKLOG_ITERS} iterations of work queued when the host "
            f"finished enqueueing (smallest backlog {backlog_iters:.2f} "
            f"iterations; host enqueue {enqueue_ms / iters:.4f} ms per "
            "call); the intervals include host enqueue time and the number is "
            "an upper bound on the kernel time; time the callable as a graph "
            "replay or through a fused launcher to measure the GPU alone")
    return False, enqueue_ms, backlog_iters, ""


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
    the measured interval; what differs is around it, and every difference is
    listed here so nobody has to diff the two functions to find one:

      1. warmup: a COUNT of back-to-back calls then one synchronise, not a
         duration of delivered GPU time. For the roof's 8192^3 GEMM the five
         calls are ~7 ms, and the roof is warm because `calibrate()` settled
         under compute load for up to 30 s first, not because of this;
      2. iters: caller-fixed (20 for the GEMMs) or `calibrate_iters` from ONE
         isolated call on an idle GPU, which reads a 5 us kernel as 15 us and
         sizes the trials at a third of the target; `time_kernel` sizes from
         a queue-deep flushed probe;
      3. no clock during the trials. The roof's clock is taken by a separate
         `calibrate.clock_under_load` run afterwards (same kernel, same
         state, NVML only);
      4. no host-bound verdict: the trial walls are discarded, so a
         host-bound `measure_bandwidth` pattern would not be marked;
      5. `flush_mb` defaults to `DEFAULT_FLUSH_MB` (256), not
         `flush_mb_for_device()` (240 MiB on an H200); the driver passes
         `cfg.flush_mb = DEFAULT_FLUSH_MB` on both paths, so cells and the
         bandwidth roof agree in practice and only direct callers differ;
      6. the warmup is UNFLUSHED whatever `l2_flush` is; `time_kernel` warms
         the loop it measures;
      7. the record is `TimingResult`, stamped `RETIRED_TIMER_BASIS`, with no
         clock or host fields; a row off it is a legacy-seam row.

    The driver depends on these semantics for the retired seam, so they are
    unchanged. New consumers use `time_kernel`. The GPU acceptance test
    (`tests/test_timing.py`, marked gpu) holds the two within 2% on a 55 us
    kernel; it runs only on the box.
    """
    require_cuda()
    flusher = L2Flusher(flush_mb if l2_flush else 0, mode=flush_mode)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if iters is None:
        iters = calibrate_iters(fn, target_ms)
    samples, _ = _timed_trials(fn, iters, trials, _EventPairs(iters), flusher.flush)

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
    samples, _ = _timed_trials(graph.replay, iters, trials, _EventPairs(iters),
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

    `per_call_ms` is the mean of the PROBE's per-iteration intervals: the
    kernel alone, measured queue-deep after the warmup, with the flush (when
    there is one) outside the interval exactly as the trials have it. It is
    what `time_kernel` calibrates `iters` from. An isolated single call (what
    `calibrate_iters` measures) includes launch latency on an idle GPU and
    reads a 5 us kernel as 15, so an iteration count sized from it lands at a
    third of the target; the last batch's mean (what this was until v3)
    measured the kernel warm in L2 while the trials measure it cold, so under
    a flush it sized the trials from the wrong number too.

    `calls` counts every call the warmup made, probe included, because every
    one of them was load the governor saw; `probe_calls` says how many of
    them were the probe. `delivered_ms` is GPU time under load: the batches'
    whole intervals (flush included when flushing, since the flush is load)
    plus the probe's kernel intervals (its flushes are outside those, so the
    probe is under-counted by its flush share; the direction is that the
    warmup ran slightly longer than the figure says).

    `settle_ms` and `settle_calls` are the EXTRA load the settle loop ran after
    `warmup_ms` was already delivered, waiting for two consecutive clock reads
    to agree within one `CLOCK_STEP_MHZ` step; both are inside `delivered_ms`
    and `calls` as well, because they were load the governor saw. `settled` is
    True when the reads agreed, False when the `SETTLE_CAP_MULTIPLE` cap
    stopped the loop first, and None when no clock reader was given (off-GPU,
    or a host without NVML), which is not the same as "did not settle".
    `settle_reads` is every clock the loop read, in order, so a cell that hit
    the cap shows what it was doing; `settle_note` says why the loop stopped
    when it did not settle.
    """

    delivered_ms: float
    calls: int
    batches: int
    per_call_ms: float
    probe_calls: int = 0
    settle_ms: float = 0.0
    settle_calls: int = 0
    settle_reads: tuple[float, ...] = ()
    settled: bool | None = None
    settle_note: str = ""


def warm_until(fn: Callable[[], None], warmup_ms: float, events,
               batch_ms: float = WARMUP_BATCH_MS,
               flush: Callable[[], None] | None = None,
               probe_calls: int = WARMUP_PROBE_CALLS,
               clock_read: Callable[[], ClockState] | None = None,
               ) -> WarmupReport:
    """Run the loop the trials will run until `warmup_ms` of GPU time has passed.

    A COUNT of warmup calls is the wrong unit, and the ladders that compared
    cells warmed at 5 against cells warmed at 20 were comparing clock states:
    a 1 ms kernel needs hundreds of calls before the governor reacts, a 30 ms
    GEMM needs one. So the warmup is a duration of delivered GPU time, measured
    with the same events the trials use. The first batch is one call, which
    sizes the rest to about `batch_ms` each so the queue stays deep between
    synchronises; the loop stops after the batch that carries the total past
    `warmup_ms`, so the overshoot is bounded by one batch. Batches are capped
    at `WARMUP_BATCH_MAX_CALLS` calls, so for a kernel under 2.5 us a batch
    delivers less than `batch_ms` and the loop takes more of them.

    THE WARMUP RUNS THE LOOP THE TRIALS RUN. Until v3 it ran unflushed while
    the trials ran flushed. For a 50 us kernel the flushed loop is roughly
    half kernel and half flush, and the flush is pure HBM streaming, which
    `L2Flusher`'s own hazard note records as holding clocks UP; the unflushed
    warmup therefore brought the card to the operating point of a DIFFERENT
    workload, and DRIFT could only catch the transition if it was still in
    progress after the first sample landed 50 ms into the trials. With
    `flush` given, every warmup call is preceded by a flush exactly as in
    `_timed_trials`, so the governor is responding to the load that is about
    to be measured.

    THEN THE CLOCK HAS TO STOP MOVING (2026-09-09). A duration of load is not
    the same claim as a settled governor, and the first H200 session paid for
    the difference: all 135 of its DRIFT verdicts were, by position, the first
    cell of a rep after the workload changed, the governor still walking from
    the previous tile's operating point to this one (1875->1725, 1560->1650
    three times). With a `clock_read` given, the loop keeps running the trials'
    own work, one batch at a time, until two consecutive reads agree within one
    `CLOCK_STEP_MHZ` step, and gives up at `SETTLE_CAP_MULTIPLE` x `warmup_ms`
    of delivered time with `settled=False` on the record rather than pretending
    it converged. Without one (off-GPU, or a host with no NVML) there is no
    settle loop and `settled` is None: the fix is at the instrument, so a host
    that cannot read the clock does not get it and the record says so.

    THEN A PROBE, to size `iters`. `probe_calls` iterations (or one batch's
    worth, whichever is fewer) through `_timed_trials` with the same flush,
    one interval per call, and `per_call_ms` is their mean: the kernel's own
    queue-deep, cold-L2-when-flushing time, which is the quantity `target_ms`
    budgets. The batch mean cannot give that under a flush, because one pair
    around a batch of `flush(); fn()` holds both.

    WHAT NONE OF THIS SEES: thermal ORDER. `warmup_ms` (300 ms by default)
    reaches the governor's fast response, not thermal equilibrium;
    `calibrate.settle_clocks` needs up to 30 s and documents an 840 to 1980
    MHz ramp from idle. In a sweep the thermal state a cell starts in is
    inherited from the cells before it, so cell order is a confound this
    warmup cannot remove and the row cannot show; the LEVEL and DRIFT
    verdicts are the instrument's only witnesses to it.
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
    per_batch_call = 0.0
    while delivered < warmup_ms:
        pair.starts[0].record()
        for _ in range(batch):
            if flush is not None:
                flush()
            fn()
        pair.ends[0].record()
        pair.synchronize()
        ms = max(pair.elapsed(1)[0], 1e-4)
        delivered += ms
        calls += batch
        batches += 1
        per_batch_call = ms / batch
        batch = max(1, min(WARMUP_BATCH_MAX_CALLS, int(batch_ms / per_batch_call)))

    # The settle: keep delivering the same load until the clock stops moving.
    settle_ms = 0.0
    settle_calls = 0
    reads: list[float] = []
    settled: bool | None = None
    settle_note = ""
    if clock_read is not None:
        settled = False
        cap_ms = SETTLE_CAP_MULTIPLE * warmup_ms
        previous: float | None = None
        while True:
            try:
                mhz = float(clock_read().sm_clock_mhz)
            except Exception as e:  # noqa: BLE001
                # As broad as every other reader in this module: torch reports
                # a missing pynvml, a revoked NVML and an old binding as three
                # unrelated types, and none of them may cost the cell its
                # measurement now that the warmup is already delivered.
                settle_note = (f"clock reader raised after {len(reads)} reads: "
                               f"{type(e).__name__}: {e}; settle abandoned")
                break
            if mhz > 0:
                reads.append(mhz)
                if previous is not None and abs(mhz - previous) <= CLOCK_STEP_MHZ:
                    settled = True
                    break
                previous = mhz
            else:
                # A zero is NVML declining to answer, not a clock. It cannot
                # pair with anything, so it does not become `previous`.
                settle_note = "a clock read came back zero during the settle"
            if delivered >= cap_ms:
                settle_note = (
                    f"the clock had not settled to within {CLOCK_STEP_MHZ:.0f} MHz "
                    f"after {delivered:.0f} ms of warmup (cap "
                    f"{SETTLE_CAP_MULTIPLE:g}x{warmup_ms:.0f} ms); reads "
                    + ", ".join(f"{r:.0f}" for r in reads))
                break
            pair.starts[0].record()
            for _ in range(batch):
                if flush is not None:
                    flush()
                fn()
            pair.ends[0].record()
            pair.synchronize()
            ms = max(pair.elapsed(1)[0], 1e-4)
            delivered += ms
            settle_ms += ms
            settle_calls += batch
            calls += batch
            batches += 1

    # The probe: the trials' own loop, one interval per call, flush outside.
    # AFTER the settle, so `iters` is sized at the operating point the trials
    # will run at rather than at whatever the governor had reached at
    # `warmup_ms`.
    n = max(1, min(int(probe_calls), batch))
    samples, _ = _timed_trials(fn, n, 1, events(n), flush)
    delivered += sum(samples)
    calls += n
    per_call = max(sum(samples) / n, 1e-4)
    return WarmupReport(delivered_ms=delivered, calls=calls, batches=batches,
                        per_call_ms=per_call, probe_calls=n,
                        settle_ms=settle_ms, settle_calls=settle_calls,
                        settle_reads=tuple(reads), settled=settled,
                        settle_note=settle_note)


def iters_for(per_call_ms: float, target_ms: float, lo: int = 10,
              hi: int = 2000) -> int:
    """Iterations per trial so that one trial holds `target_ms` of KERNEL time.

    `target_ms` budgets the measured intervals, not the trial's wall: with
    `l2_flush` each iteration also enqueues a flush (about 50 us of reads over
    a 240 MB buffer on an H200, outside the interval by construction) and a
    cold-L2 start for the kernel. For a 10 us kernel the 2000-iteration cap
    makes the trial about 120 ms of wall for 20 ms of measured time; for a
    1 ms kernel the flush is 5% on top. Same property as `calibrate_iters`,
    stated here because the old docstring implied the trial itself lasted
    `target_ms`.
    """
    return max(lo, min(hi, int(target_ms / max(per_call_ms, 1e-4))))


class ClockSourceUnavailable(RuntimeError):
    """The fast clock path cannot be used on this host; the message says why.

    Raised by `nvml_clock_reader` when torch's NVML bindings are absent
    (`nvidia-ml-py` not installed: the base pod venv ships without it) or
    refuse (a vGPU or a restricted container). `BackgroundClockSampler`
    catches it at entry and records the reason instead of sampling, so a
    `KernelTiming` on such a host carries `clock_source == "none"` and the
    reason in `clock_note`, never a zero and never a forked nvidia-smi inside
    the timed region.
    """


def nvml_clock_reader(device_index: int | None = None) -> Callable[[], ClockState]:
    """A reader for one device's SM clock through torch's NVML bindings only.

    Probed ONCE, here, before the timed region: a reader that fails on every
    poll would spend the whole region raising. The device is fixed at
    construction from the CALLING thread, because torch's current device is
    per host thread and a poll thread starts on device 0; a sweep that
    selected card k with `torch.cuda.set_device(k)` would otherwise read card
    0's clock. Nothing in the repo calls `set_device` today (selection is via
    CUDA_VISIBLE_DEVICES), so this is latent, and cheap to close.

    Deliberately no nvidia-smi fallback. `ClockState.sample` has one because
    its callers read twice per cell; a poller reads every 50 ms for the whole
    timed region, and a fork plus NVML init per read (tens of milliseconds,
    on the host the enqueue thread needs) is not a clock sample, it is a
    perturbation of the thing being measured.
    """
    if not torch.cuda.is_available():
        raise ClockSourceUnavailable("no CUDA device; there is no clock to read")
    index = torch.cuda.current_device() if device_index is None else int(device_index)

    def read() -> ClockState:
        return ClockState(int(torch.cuda.clock_rate(index)),
                          int(torch.cuda.temperature(index)),
                          power_w=nvml_power_w(index))

    try:
        read()
    except Exception as e:  # noqa: BLE001
        # Broad for the same reason ClockState.sample is: torch surfaces a
        # missing pynvml as ModuleNotFoundError and NVML refusals as its own
        # NVMLError family, and the reader has to name them all as one thing.
        raise ClockSourceUnavailable(
            f"torch.cuda.clock_rate(device={index}) failed: {type(e).__name__}: "
            f"{e}; install nvidia-ml-py in the pod venv to sample clocks under "
            "load, or accept rows without a clock") from e
    return read



def clock_poll_for(trials: int, iters: int, per_call_ms: float) -> float:
    """Poll interval, seconds, for the under-load sampler over one timed region.

    The region is `trials x iters x per_call_ms`, and `iters_for` truncates, so
    a trial holds a little UNDER `target_ms`. A fixed `CLOCK_POLL_SECONDS`
    (50 ms, first sample AT 50 ms) lands two or three samples in a ~150 ms
    region against `CLOCK_SAMPLE_FLOOR` of three, and two leave
    `sm_clock_load_mhz` None: LEVEL undetermined on a sound measurement, which
    alias_ablation's level gate reads as UNKNOWN, INVALID and latched
    (2026-09-09, 50 ms budget x 3 trials). Aim for TWICE the floor inside the
    expected region, never slower than `CLOCK_POLL_SECONDS`, never faster than
    `CLOCK_POLL_FLOOR_SECONDS`. Pure.
    """
    expected_s = trials * iters * per_call_ms / 1e3
    return max(CLOCK_POLL_FLOOR_SECONDS,
               min(CLOCK_POLL_SECONDS, expected_s / (2 * CLOCK_SAMPLE_FLOOR)))


class BackgroundClockSampler:
    """Polls the SM clock from a thread while the calling thread keeps the GPU busy.

    The context-manager shape is the injection seam: `time_kernel` enters it
    before the first trial and exits it after the last synchronise, and reads
    `.samples`, `.source`, `.note` and `.poll_cost_ms` afterwards. A fake with
    the same shape returns a scripted trace.

    THE SOURCE IS RESOLVED AT ENTRY AND NAMED. With no `sample` injected the
    sampler asks `nvml_clock_reader` for the device it was given (the
    calling thread's current device, passed by `time_kernel`); if that
    refuses, no thread starts, `source` is "none" and `note` carries the
    reason. It never falls back to nvidia-smi. An injected `sample` is
    "injected".

    The first sample is taken `poll_seconds` AFTER entering, not at entry.
    Entry happens right after the events were primed, which ends in a
    synchronise, and a sample at that instant is the idle-boost reading the
    old flag was fooled by. Waiting first puts every sample inside the trials.

    A reader that raises mid-region stops the poller and is REPORTED, not
    lost: the exception used to reach threading.excepthook and the record
    then said "one usable sample, drift not determinable", which misstates
    the cause. Now `note` names the exception and the count reached.

    Each read is wall-clocked; `poll_cost_ms` is the median. A reader whose
    cost exceeds `CLOCK_POLL_BUDGET_FRACTION` of the poll interval gets a
    note too, because a poll thread that is awake most of the time competes
    with the enqueue thread for the host, and that is the thread keeping the
    queue deep.
    """

    def __init__(self, sample: Callable[[], ClockState] | None = None,
                 poll_seconds: float = CLOCK_POLL_SECONDS,
                 device_index: int | None = None):
        self._sample = sample
        self._device = device_index
        self._poll = poll_seconds
        self._stop = threading.Event()
        self._got: list[ClockState] = []
        self._costs: list[float] = []
        self._thread: threading.Thread | None = None
        self._reader: Callable[[], ClockState] | None = None
        self.samples: tuple[ClockState, ...] = ()
        self.source: str = "unresolved"
        self.note: str = ""
        self.poll_cost_ms: float | None = None

    @property
    def poll_seconds(self) -> float:
        return self._poll

    def _run(self) -> None:
        assert self._reader is not None
        while not self._stop.wait(self._poll):
            t0 = time.perf_counter()
            try:
                state = self._reader()
            except Exception as e:  # noqa: BLE001
                # Anything the reader raises is the finding; a narrow clause
                # would send the rest to threading.excepthook and lose it.
                self.note = (f"clock sampler raised after {len(self._got)} "
                             f"samples: {type(e).__name__}: {e}; polling stopped")
                return
            self._costs.append(time.perf_counter() - t0)
            self._got.append(state)

    def __enter__(self) -> BackgroundClockSampler:
        self._stop.clear()
        self._got = []
        self._costs = []
        self._thread = None
        self.note = ""
        self.samples = ()
        self.poll_cost_ms = None
        if self._sample is None:
            try:
                self._reader = nvml_clock_reader(self._device)
            except ClockSourceUnavailable as e:
                self.source = "none"
                self.note = str(e)
                return self
            self.source = "nvml"
        else:
            self._reader = self._sample
            self.source = "injected"
        self._thread = threading.Thread(target=self._run, name="clock-sampler",
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.samples = tuple(self._got)
        if self._costs:
            self.poll_cost_ms = float(statistics.median(self._costs)) * 1e3
            budget_ms = CLOCK_POLL_BUDGET_FRACTION * self._poll * 1e3
            if self.poll_cost_ms > budget_ms:
                slow = (f"clock poll cost {self.poll_cost_ms:.2f} ms per read, "
                        f"over {CLOCK_POLL_BUDGET_FRACTION:.0%} of the "
                        f"{self._poll * 1e3:.0f} ms poll interval; the sampler "
                        "competed with the enqueue thread for the host")
                self.note = (self.note + "; " if self.note else "") + slow


#: What `level_side` answers with. Empty is INSIDE the band; the other two
#: name the direction a cell left it, and the direction is the finding: "low"
#: is the throttle the flag was built for, "high" is the boosted memory-shaped
#: cell whose fixed-roof fraction is inflated by the ratio. `None` is not a
#: side; it is "no comparison could be made".
LEVEL_LOW = "low"
LEVEL_HIGH = "high"


def level_band(reference_mhz: float) -> tuple[float, float]:
    """The LEVEL band's two edges in MHz, on the NVML grid. Pure.

    `[LEVEL_FRACTION, LEVEL_HIGH_FRACTION] * reference`, each edge snapped to
    `CLOCK_STEP_MHZ` so it is a clock the card can actually report. Against the
    session's 1485 MHz bf16 reference that is [1410, 1560]; unsnapped it was
    [1410.75, 1559.25], which put the 1410 readings outside the band and the
    1425 ones inside it on a 0.1% difference in time.
    """
    return (snap_to_clock_step(LEVEL_FRACTION * reference_mhz),
            snap_to_clock_step(LEVEL_HIGH_FRACTION * reference_mhz))


def level_side(load_mhz: float | None, reference_mhz: float | None) -> str | None:
    """Which side of the LEVEL band a loaded clock sits on. Pure.

    THE ONE DEFINITION of the band: `clock_flags` derives its LEVEL bool from
    this, so the flag and the side cannot disagree about where the edges are.
    Inside `level_band(reference)` is "", below is `LEVEL_LOW`, above is
    `LEVEL_HIGH`; None when there is no reference or no load sample, because
    half a comparison is not a verdict.

    A SIDE IS A RECORD AND NOT AN EXCLUSION since 2026-09-09. On the H200 the
    side names the tile family a cell is in: BM=128/N=64 and BM=64/G=1 sit
    below the band because they draw more power under the same 700 W cap than
    the ruler's GEMM does, memory-shaped work boosts above it. Callers exclude
    on `clock_drift_ok`; this answers "where did it sit", not "may it be used".
    """
    if load_mhz is None or reference_mhz is None or reference_mhz <= 0:
        return None
    low, high = level_band(reference_mhz)
    if load_mhz < low:
        return LEVEL_LOW
    if load_mhz > high:
        return LEVEL_HIGH
    return ""


#: What `drift_direction` answers with. The direction is the finding: a rise is
#: a warmup that had not reached the operating point (which is what the settle
#: loop in `warm_until` exists to remove), a fall is the card giving clock back
#: during the trials, and an oscillation is a governor hunting, which no longer
#: warmup can fix. Empty is "no direction": one sample, or a steady clock.
DRIFT_UP = "settling upward"
DRIFT_DOWN = "dropping"
DRIFT_OSCILLATING = "oscillating"


def drift_direction(samples, fraction: float = DRIFT_FRACTION) -> str:
    """Which way a cell's under-load clock moved, in words. Pure.

    Read from the WHOLE sample list rather than the first and last pair, which
    is the pair the DRIFT verdict itself is computed from. An excursion that
    came back reads as no drift at all on the endpoints, and calling it steady
    would hide a governor hunting between two states; here it is
    `DRIFT_OSCILLATING`, as is a trace that leaves the first reading's band in
    both directions. Otherwise the sign of last minus first decides, and
    endpoints inside the band with no excursion get "".
    """
    usable = [float(s) for s in samples if s and float(s) > 0]
    if len(usable) < 2:
        return ""
    first, last = usable[0], usable[-1]
    above = any(s > first * (1 + fraction) for s in usable)
    below = any(s < first * (1 - fraction) for s in usable)
    if above and below:
        return DRIFT_OSCILLATING
    if abs(last - first) <= fraction * first:
        return DRIFT_OSCILLATING if (above or below) else ""
    return DRIFT_UP if last > first else DRIFT_DOWN


def clock_flags(load_mhz: float | None, start_mhz: float | None,
                end_mhz: float | None, reference_mhz: float | None,
                ) -> tuple[bool | None, bool | None]:
    """LEVEL and DRIFT verdicts on under-load clock samples. Pure.

    LEVEL: the loaded clock is inside `level_band(reference_mhz)`, around the
    clock the roof was measured at, in EITHER direction (`level_side` names
    which). Until 2026-09-03 this was one-sided, `load >= 0.95 * reference`,
    and a cell boosted to 1980 against a 1515 roof passed with a fixed-roof
    fraction inflated by 31%. None when there is no reference or no load
    sample: the flag is a comparison and half a comparison is not a verdict.
    IT EXCLUDES NOTHING (2026-09-09): the side is the tile's power state under
    the 700 W cap, and the driver's `throttled` reads DRIFT alone.

    DRIFT: first and last under-load samples agree within `DRIFT_FRACTION` of
    the first, in either direction. A drop is the card giving clock back
    during the trials; a rise is a warmup that did not reach the operating
    point. Both mean the samples were not taken at one clock and the median is
    a blend of two states, which is why THIS is the verdict a consumer
    excludes on. `drift_direction` names which of the three it was.
    """
    level: bool | None
    drift: bool | None
    side = level_side(load_mhz, reference_mhz)
    level = None if side is None else side == ""
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
    the poller to land a sample. `clock_note` says which, and `clock_source`
    says which reader polled ("nvml", "injected", or "none"). The floors:
    `sm_clock_load_mhz` is a median of at least `CLOCK_SAMPLE_FLOOR` usable
    samples, the same floor the reference clock in `calibrate.clock_under_load`
    is held to, so LEVEL compares two numbers of one kind; `sm_clock_start_mhz`
    and `sm_clock_end_mhz` need two. `clock_level_ok` and `clock_drift_ok` are
    the two verdicts `clock_flags` documents, and since 2026-09-09 they do two
    different jobs: DRIFT is the EXCLUSION (the cell was not measured at one
    operating point), LEVEL and `clock_level_side` are a RECORD of where the
    cell sat relative to the ruler's GEMM. A consumer filters on DRIFT; a
    consumer that also drops LEVEL-LOW rows drops the two tiles this study is
    about, which is what the H200 session's 750 cells showed the old rule
    doing. Both fractions of the roof are on the driver's row either way.

    `host_bound` is the third verdict, from `host_bound_verdict`: True when
    any trial's queue had drained by the time the host finished enqueueing,
    in which case the intervals include host time and `ms_*` bound the kernel
    from above. `host_enqueue_ms` is the median per-trial host wall of the
    enqueue loop; divided by `iters` it is the host's per-call cost, the
    number to hold beside `ms_p50` when the flag is up. `host_backlog_iters`
    is the smallest per-trial backlog in iterations, the quantity the verdict
    thresholds at `HOST_BOUND_BACKLOG_ITERS`, so the distance from the
    boundary is on the record and not only the side. `host_note` says why.

    `clock_level_side` names which way a LEVEL failure went (`LEVEL_LOW`,
    `LEVEL_HIGH`, or "" when level or undetermined); `reference_clock_mhz` is
    the number LEVEL was scored against, kept on the record so a verdict can
    be re-derived from the row alone.

    `ms_std` IS INTRA-RUN. The `iters * trials` samples are consecutive
    iterations in one thermal state; the trials add samples, not
    independence, and there is no between-replicate variance or confidence
    interval in this record. That lives in `scripts/replicate_noise_floor.py`,
    which needs independent runs, and a reader who quotes `ms_std` as an
    uncertainty on the cell is quoting the wrong thing.
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
    clock_source: str
    clock_poll_ms: float | None
    host_bound: bool | None
    host_enqueue_ms: float | None
    clock_note: str = ""
    host_note: str = ""
    instrument: str = TIMING_BASIS
    clock_level_side: str = ""
    host_backlog_iters: float | None = None
    reference_clock_mhz: float | None = None
    #: Every usable under-load clock reading, in order, MHz. THE SAMPLES AND
    #: NOT ONLY THEIR SUMMARY: six of the seven ladder writers dropped the
    #: first and last pair and all seven dropped the list, so when the H200
    #: session produced 135 DRIFT verdicts only four cells could be examined
    #: for what they had actually done. A median and two endpoints cannot tell
    #: a settling ramp from a card hunting between two states, and
    #: `drift_direction` reads the list to say which.
    clock_samples_mhz: tuple[float, ...] = ()
    #: Median board power over the under-load samples, W, None when no sample
    #: carried one. Read at the same NVML call as the clock (`ClockState.
    #: power_w`), so a LOW clock can be told apart as a hungry tile at the
    #: 700 W cap from a card in trouble: the session's calibration GEMM held
    #: 1485 MHz at 691 W, and the BM=128/N=64 cells that sat at 1395 were at
    #: the same cap doing more work per clock.
    power_w: float | None = None
    #: Which way the clock moved when DRIFT failed, from `drift_direction`
    #: over the sample list; "" when DRIFT passed or was undetermined.
    clock_drift_direction: str = ""
    #: Delivered GPU time the settle loop added after `warmup_ms`, ms, and the
    #: calls it made. 0.0 with `clock_settled` None means no clock reader was
    #: available to settle against, which is not the same as settling at once.
    settle_ms: float = 0.0
    settle_calls: int = 0
    clock_settled: bool | None = None


def time_kernel(
    fn: Callable[[], None],
    *,
    warmup_ms: float,
    target_ms: float = 200.0,
    trials: int = 3,
    l2_flush: bool = True,
    reference_clock_mhz: float | None = None,
    clock_sampler=None,
    clock_reader=None,
    events=None,
    flusher=None,
) -> KernelTiming:
    """Time `fn` the one way this repository times anything it publishes.

    In order: warm up under sustained load, running the SAME loop the trials
    will run (flush before each call when `l2_flush`), until `warmup_ms` of
    GPU time has been delivered AND two consecutive clock reads agree within
    one `CLOCK_STEP_MHZ` step (`warm_until`); size `iters` so a trial holds
    `target_ms` of kernel time, from a flushed queue-deep per-iteration probe;
    prime one event pair per iteration; start the clock poller; run `trials`
    trials of `_timed_trials` (flush before each call when `l2_flush`, one
    synchronise per trial); stop the poller; summarise the `iters * trials`
    samples the way `time_eager` does and the clock samples the way
    `clock_flags` does.

    `reference_clock_mhz` is the clock the roof was measured at
    (`calibrate.LoadedClock.median_mhz`); without it the LEVEL flag is None,
    because a level is relative to something and this function will not
    invent the something. The verdict is two-sided, the record names the side,
    and NEITHER SIDE EXCLUDES the cell: a cell below the band is a tile that
    draws more power under the same cap, a cell above it is not "faster", and
    in both cases the fixed-roof fraction is off by `load / reference` while
    the driver's per-row roof (`roofline.roof_at_clock`) carries the other
    fraction beside it. DRIFT is the verdict that excludes.

    `clock_reader` is the settle loop's reader, a callable returning a
    `ClockState`. Left None on a CUDA host it is `nvml_clock_reader` for the
    calling thread's device; where that refuses there is no settle loop and
    the record says so in `clock_note`. Injected, it is how the tests drive
    the loop without a GPU.

    The default sampler is bound to the calling thread's current device and
    reads NVML only; on a host without `nvidia-ml-py` the record says so in
    `clock_note` and carries `clock_source == "none"`. The host-bound verdict
    is taken from the trials' wall clocks; see `host_bound_verdict`.

    OFF-GPU. Refuses with `TimingRefused` unless every CUDA-touching part is
    injected: `events` (a factory with `_EventPairs`'s shape), `clock_sampler`
    (a context manager with `BackgroundClockSampler`'s shape: `.samples`,
    `.source`, `.note`, `.poll_cost_ms` after exit) and, when `l2_flush`,
    `flusher` (an object with `flush()` and `megabytes`). With all of them
    present the full logic runs against the fakes; that is how the tests
    plant every PASS and FAIL branch of all three verdicts.
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
    if l2_flush and flusher is None:
        flusher = L2Flusher(flush_mb_for_device())
    flush = flusher.flush if l2_flush else None
    flush_mb = int(flusher.megabytes) if l2_flush else 0

    settle_note = ""
    if clock_reader is None and on_gpu:
        try:
            clock_reader = nvml_clock_reader(torch.cuda.current_device())
        except ClockSourceUnavailable as e:
            clock_reader = None
            settle_note = (f"the warmup could not settle on the clock: {e}")
    warm = warm_until(fn, warmup_ms, events, flush=flush,
                      clock_read=clock_reader)
    iters = iters_for(warm.per_call_ms, target_ms)
    if clock_sampler is None:
        # The device is read HERE, on the calling thread, and handed to the
        # poller: torch's current device is per thread and the poll thread
        # would otherwise start on device 0.
        #
        # THE POLL IS SIZED FROM THE REGION IT HAS TO LAND IN: see
        # `clock_poll_for`. On 2026-09-09 alias_ablation's 50 ms budget x 3
        # trials was ~150 ms of region against a fixed 50 ms poll, two or
        # three samples around a floor of three, LEVEL undetermined on a
        # sound rung, level gate UNKNOWN, INVALID, latched.
        poll = clock_poll_for(trials, iters, warm.per_call_ms)
        clock_sampler = BackgroundClockSampler(
            device_index=torch.cuda.current_device(), poll_seconds=poll)
    pairs = events(iters)
    with clock_sampler as poller:
        samples, walls = _timed_trials(fn, iters, trials, pairs, flush)
    clocks = [c.sm_clock_mhz for c in poller.samples]
    usable = [c for c in clocks if c > 0]
    powers = [float(c.power_w) for c in poller.samples
              if getattr(c, "power_w", 0.0) > 0]

    notes: list[str] = []
    if settle_note:
        notes.append(settle_note)
    if warm.settle_note:
        notes.append(warm.settle_note)
    if poller.note:
        notes.append(poller.note)
    load: float | None = None
    start: float | None = None
    end: float | None = None
    if len(usable) >= CLOCK_SAMPLE_FLOOR:
        load = float(statistics.median(usable))
        if reference_clock_mhz is None:
            notes.append("no reference clock given, level not determinable")
    if len(usable) >= 2:
        start, end = float(usable[0]), float(usable[-1])
    if not usable:
        notes.append(f"no usable SM clock sample during the trials ({len(clocks)} "
                     "polled, all zero or none landed); clocks not determinable")
    elif len(usable) < CLOCK_SAMPLE_FLOOR:
        what = ("drift needs two" if len(usable) < 2
                else "drift is reported from the first and last")
        notes.append(f"{len(usable)} usable clock sample(s) during the trials "
                     f"({len(clocks)} polled) is below the floor of "
                     f"{CLOCK_SAMPLE_FLOOR} a median is taken from; level not "
                     f"determinable, {what}")
    level_ok, drift_ok = clock_flags(load, start, end, reference_clock_mhz)
    side = level_side(load, reference_clock_mhz) or ""
    direction = drift_direction(usable) if drift_ok is False else ""
    if side == LEVEL_LOW:
        low_edge, _ = level_band(reference_clock_mhz)
        notes.append(
            f"LEVEL LOW (recorded, not an exclusion): loaded clock "
            f"{load:.0f} MHz is under the {low_edge:.0f} MHz edge of the band "
            f"around the {reference_clock_mhz:.0f} MHz reference; this tile "
            "draws more power under the same cap than the GEMM the roof was "
            "measured with, so its fixed-roof fraction is delivered throughput "
            f"and its issue efficiency is {load / reference_clock_mhz:.2f}x "
            "away, in roof_at_cell_clock_tflops")
    elif side == LEVEL_HIGH:
        _, high_edge = level_band(reference_clock_mhz)
        notes.append(
            f"LEVEL HIGH (recorded, not an exclusion): loaded clock "
            f"{load:.0f} MHz is over the {high_edge:.0f} MHz edge of the band "
            f"around the {reference_clock_mhz:.0f} MHz reference; the cell ran "
            f"at {load / reference_clock_mhz:.2f}x the issue rate the fixed "
            "compute roof assumed, so its fraction of that roof is inflated by "
            "the ratio; read roof_at_cell_clock_tflops")
    if direction:
        notes.append(
            f"DRIFT failed, {direction}: under-load clock went "
            f"{start:.0f} -> {end:.0f} MHz over {len(usable)} samples "
            f"({', '.join(f'{c:.0f}' for c in usable)}); the trials were not "
            "at one operating point and the median is a blend")
    host_bound, host_ms, backlog_iters, host_note = host_bound_verdict(walls, iters)

    p50, p90, lo, std = _stats(samples)
    return KernelTiming(
        ms_p50=p50, ms_p90=p90, ms_min=lo, ms_std=std,
        iters=iters, trials=trials, warmup_ms=warm.delivered_ms,
        l2_flush=l2_flush, sm_clock_load_mhz=load, sm_clock_start_mhz=start,
        sm_clock_end_mhz=end, clock_level_ok=level_ok, clock_drift_ok=drift_ok,
        samples=len(samples), warmup_calls=warm.calls, flush_mb=flush_mb,
        clock_samples=len(usable), clock_source=poller.source,
        clock_poll_ms=poller.poll_cost_ms, host_bound=host_bound,
        host_enqueue_ms=host_ms, clock_note="; ".join(notes),
        host_note=host_note, clock_level_side=side,
        host_backlog_iters=backlog_iters,
        reference_clock_mhz=reference_clock_mhz,
        clock_samples_mhz=tuple(float(c) for c in usable),
        power_w=float(statistics.median(powers)) if powers else None,
        clock_drift_direction=direction,
        settle_ms=warm.settle_ms, settle_calls=warm.settle_calls,
        clock_settled=warm.settled,
    )
