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
with the reason written down. Because all of them are recorded, anyone
(including you) can recompute a roofline against a different denominator without
re-running, which is what `moe/bench/recompute.py` exists to do.

THE READ CEILING IS A REDUCTION, AND IT IS NAMED AS ONE
-------------------------------------------------------
Until 2026-09-02 this file measured a pattern called `read` with `torch.sum`,
called it "the closest analogue to streaming expert weights", and left it at
that. A reduction is not a read. It is renamed `read_reduce`, because what it
measures is the rate at which ATen can CONSUME a stream, and a consumer that
also combines cannot be faster than one that only loads. So:

    read_reduce is a LOWER BOUND on this machine's DRAM read rate, not an
    estimate of it.

That bound has already moved once under its own shape. The 2026-08-27 fix
reduced a `[rows, n/rows]` view along the CONTIGUOUS axis instead of a 1-D
buffer to a single scalar, which removed the global combine and gained 1.7%
(4389.4 -> 4469.6 GB/s) with no change to the traffic. A shape change worth 1.7%
means the shape was still in the number, so `read_stream` is measured beside it:
a Triton kernel in `moe/bench/read_probe.py` where each program loads its own
tiles and stores ONE float, so there is no cross-CTA combine at all. The gap
between the two is what the reduction shape still costs. When Triton is
unavailable the pattern is ABSENT and the refusal is recorded; it is never a
zero and never silently the reduction's number under a different name.

WHICH DENOMINATOR THIS STUDY SHOULD USE, DECIDED RATHER THAN DEFAULTED
----------------------------------------------------------------------
The matched denominator is a READ denominator, and the argument is a traffic
mix, not a preference. An MoE layer at decode moves `3 F H b` bytes of expert
weights per expert against `(2 H + 3 F) act_b` bytes per row of activations; at
mixtral that is 352 MB against 102 KB per expert, and roughly half the
activation traffic is stores. The workload is therefore about 98.5% reads.
`triad` is 2 reads and 1 write, so 33% of it is a store stream this workload
does not have.

Using triad anyway is not neutral. On the H200, triad reads 4374.8 GB/s and
read_reduce 4469.6, so triad puts the ridge at 162.8 where the matched ruler
puts it at 159.4. A HIGHER ridge means more cells classify as memory bound and
crossings move to larger batch, which is the direction of this study's own
headline claim. The ruler in use flatters the conclusion by 2.2%, and if the
true read rate is above the reduction's lower bound the bias is larger still.

So the decision, and it has two halves:

  * `DEFAULT_CEILING` STAYS `triad`. Every published row in `results/published/`
    was stamped against it, and moving it silently would re-baseline eleven arms
    in a commit nobody could audit. triad is also the only figure comparable to
    anyone else's STREAM number.
  * `MATCHED_CEILING` names the ruler the traffic actually matches, and
    `ridge_band()` returns both ends. Any crossing quoted within one tile tread
    of the boundary must be quoted against the band, not against one end.

`scripts/ruler_rebaseline.py` measures what adopting the matched ruler would do
to the existing published rows. Adopt it there, on evidence, or not at all.

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

AND IT IS THE SMALLER HALF OF THE PROBLEM. Across the six distinct calibrations
of ONE H200 that ship inside the published arms, bandwidth reproduces to 0.06%
and the compute term moves 9.9% (docs/INSTRUMENTATION.md section 6). The
denominator question above is worth 2.2%; the numerator's own session-to-session
instability is worth four times that, and it is what the study's 160.3-176.2
ridge band actually is.

WHAT CLOCK WAS IT MEASURED AT
-----------------------------
Every ceiling here is normalised against a clock, and until 2026-09-02 the GEMM's
clock was a SINGLE `nvidia-smi` sample taken AFTER `time_eager` had already
synchronised -- that is, with the GPU idle and already boosting back up. Across
the eleven committed H200 calibrations that field reads 1485, 1500, 1515, 1530,
1530, 1560, 1560, 1845, 1845, 1905 and 1935 MHz, a 30% spread on one card, while
the compute settle plateau in the same files sits at 1455-1515 and the achieved
rate moves only 12%. `gemm_efficiency_pct` is linear in that number, so the same
card reports 87.4% and 68.4% efficiency for the same kernel.

`clock_under_load` therefore samples WHILE the queue is deep and the kernel is
running, returns the median with the spread, and raises `ClockUnavailable`
rather than returning a zero that would read as a measured zero. The old
post-hoc sample is still recorded as `clock_after_idle_mhz`, because the gap
between the two is the size of the artefact and deleting it would erase the
evidence that it was there.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import torch

from . import timing as T

#: Pattern whose figure becomes `achieved_bandwidth_gbps` unless overridden.
#: DELIBERATELY STILL triad. See the module docstring: every row in
#: `results/published/` was stamped against it, and the matched ruler below is
#: adopted on the evidence `scripts/ruler_rebaseline.py` produces, not by
#: changing a constant.
DEFAULT_CEILING = "triad"

#: The read patterns, best first. `matched_ceiling` walks this in order, so a
#: session that got a Triton probe uses it and one that did not falls back to
#: the reduction's lower bound rather than to triad.
READ_PATTERNS = ("read_stream", "read_reduce")

#: The denominator this study's traffic actually matches: ~98.5% reads, against
#: triad's 67%. NOT the default, on purpose. It is what `ridge_band()` puts at
#: the other end of the band and what the re-baseline script prices.
MATCHED_CEILING = READ_PATTERNS[0]

#: Buffers must dwarf L2 (60 MiB on H200) and run long enough that launch and
#: clock ramp are not a meaningful fraction of the measurement.
DEFAULT_BUFFER_BYTES = 8 << 30

#: How far the clock samples taken during one measurement may spread before the
#: measurement is not at a single clock at all. 5% is the same threshold
#: `clock_ramped` uses across patterns, so one number means one thing here.
CLOCK_SPREAD_TOL_PCT = 5.0

#: How far a GEMM's clock under load may sit from the settle plateau that
#: preceded it. Wider than the spread tolerance because the settle plateau is
#: itself sampled between polls and the GEMM is a different kernel; 10% is still
#: far tighter than the 30% the post-hoc sample was moving by.
CLOCK_VS_SETTLE_TOL_PCT = 10.0


#: The substring every disowned-read note ends with. STABLE ON PURPOSE: the
#: A100's committed calibration carries the 2026-08-28 wording
#: ("reduction-limited, ...") and anything that reads published yaml has to
#: recognise both. Only this tail is load bearing.
DISOWNED = "not a valid ceiling"


class ClockUnavailable(RuntimeError):
    """No usable SM clock was sampled while the load was actually running.

    A typed refusal. `ClockState.sample()` returns 0 MHz when NVML is absent or
    the container forbids it, and a zero flowing into `sustained_peak_tflops`
    would make the silicon's peak zero, `gemm_efficiency_pct` infinite, and none
    of it would look wrong in the yaml. The caller decides whether a calibration
    without a clock is still worth writing; this refuses to invent one.
    """


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
    #: Achieved dense fp8, and the clock it was measured at. None on hardware
    #: without fp8 tensor cores, which is not a failure: the A100 cannot run the
    #: format and a number here would be a peak for silicon that does not exist.
    achieved_fp8_tflops: float | None = None
    fp8_gemm_clock_mhz: int = 0
    buffer_bytes: int = 0
    clocks: dict = field(default_factory=dict)
    settle: dict = field(default_factory=dict)
    #: The discarded first bandwidth pass. Kept because the gap between it and
    #: the reported pass is the size of the warm-up effect, and a large gap
    #: means the settle did not do its job.
    warmup_pass: dict = field(default_factory=dict)
    #: Full `LoadedClock` records for the two GEMMs: the samples taken WHILE
    #: each was running, their spread, and the post-hoc idle sample the old code
    #: published. `gemm_clock_mhz` above is the median of the first of these,
    #: kept as a scalar because every consumer already reads it that way.
    gemm_clock: dict = field(default_factory=dict)
    fp8_gemm_clock: dict = field(default_factory=dict)
    #: Quantities this run could not measure, each with its reason. Never a
    #: substituted value: a missing `read_stream` is an absent pattern plus a
    #: line here, so a reader can tell "not measured" from "measured as slow".
    refusals: tuple[str, ...] = ()

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
        """Did the SM clock move materially ACROSS or WITHIN the patterns?

        True means the patterns were measured at different clocks and are not
        comparable to each other, let alone publishable as ceilings.

        Both ends of every pattern count. Reading only `sm_clock_start_mhz`
        missed the shape that actually matters: a pattern that STARTS at the
        plateau and ends 500 MHz lower has thrown its own average, and four such
        patterns all starting at 1980 would have reported a spread of zero.
        """
        clks = [c for p in self.bandwidth_patterns
                for c in (p.sm_clock_start_mhz, p.sm_clock_end_mhz) if c > 0]
        if len(clks) < 2:
            return False
        return (max(clks) - min(clks)) / max(clks) * 100.0 > CLOCK_SPREAD_TOL_PCT

    @property
    def clock_established(self) -> bool | None:
        """Was the GEMM's clock a plateau, or a number sampled off a ramp?

        None when no under-load samples were taken at all, which is the state a
        `--no-settle` smoke run is in and is not the same as a failure.

        Two conditions, and both are about the same 2026-09-02 finding. The
        samples taken during the GEMM must agree with each other, and their
        median must agree with the compute settle that preceded it. Eleven
        committed calibrations of one H200 recorded 1485-1935 MHz for this
        field while their own settle histories sat at 1455-1515, and nothing in
        any of those files says which number described the GEMM.
        """
        gemm = self.gemm_clock or {}
        median = gemm.get("median_mhz") or 0
        if not median:
            return None
        if (gemm.get("spread_pct") or 0.0) > CLOCK_SPREAD_TOL_PCT:
            return False
        plateau = (self.settle or {}).get("final_mhz") or 0
        if not plateau:
            return None
        return abs(median - plateau) / plateau * 100.0 <= CLOCK_VS_SETTLE_TOL_PCT

    def pattern(self, name: str) -> BandwidthResult | None:
        for p in self.bandwidth_patterns:
            if p.pattern == name:
                return p
        return None

    def matched_pattern(self) -> BandwidthResult | None:
        """The best available READ ruler, or None if no read pattern survived.

        Walks `READ_PATTERNS` in order, so the Triton stream probe wins when it
        ran and the reduction's lower bound is used when it did not. A pattern
        the guard demoted -- one that came in below triad, which a read cannot
        legitimately do -- is skipped, because the note on it says it is not a
        valid ceiling and a band built from it would be a band built from a
        number this file has already disowned.
        """
        for name in READ_PATTERNS:
            found = self.pattern(name)
            if found is not None and DISOWNED not in found.note:
                return found
        return None

    def ridge_band(self) -> tuple[float, float] | None:
        """(low, high) FLOP/byte across the named ceiling and the matched one.

        The honest form of the ridge. The two ends are the same silicon measured
        against two rulers that differ by which traffic mix they impose, and a
        crossing that lands between them is not resolved by this calibration.
        None when there is no read pattern to put at the other end, rather than
        a degenerate band of one number twice.
        """
        matched = self.matched_pattern()
        if matched is None:
            return None
        ends = (self.ridge_point(), self.ridge_point(matched.gbps))
        return (min(ends), max(ends))

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
        d["sustained_peak_fp8_tflops"] = (
            round(sustained_peak_tflops_fp8(self.fp8_gemm_clock_mhz), 1)
            if self.achieved_fp8_tflops and self.fp8_gemm_clock_mhz else None)
        # The number STUDY.md assumes is 2.0. Recording the measured ratio makes
        # the assumption checkable instead of inherited from a datasheet.
        d["fp8_over_bf16_achieved"] = (
            round(self.achieved_fp8_tflops / self.achieved_bf16_tflops, 3)
            if self.achieved_fp8_tflops and self.achieved_bf16_tflops else None)
        # Ridge is the number that classifies every cell, and it moves with the
        # denominator. Record it for each pattern so the sensitivity is visible
        # rather than hidden behind one choice.
        d["ridge_by_pattern"] = {
            p.pattern: round(self.ridge_point(p.gbps), 1)
            for p in self.bandwidth_patterns
        }
        # The band, written down, so nobody has to know which two of the four
        # patterns are the defensible ends. `null` when no read pattern
        # survived, which is a statement and not a missing value.
        band = self.ridge_band()
        d["ridge_band"] = [round(band[0], 1), round(band[1], 1)] if band else None
        matched = self.matched_pattern()
        d["matched_ceiling_pattern"] = matched.pattern if matched else None
        d["matched_ceiling_gbps"] = round(matched.gbps, 1) if matched else None
        d["clock_established"] = self.clock_established
        d["clock_ramped"] = self.clock_ramped
        d["refusals"] = list(self.refusals)
        return d

    def settle_lines(self) -> list[str]:
        """Both settles, each labelled with the measurement it governs.

        Exists because the memory settle was MEASURED from 2026-08-27 and never
        PRINTED: `scripts/calibrate_hardware.py` rendered only the top-level
        `settle` dict, so its output read "settle reached at 1500 MHz" followed
        by "clocks 1500 -> 1980 MHz", which is exactly what an unsettled ramp
        looks like. `docs/STUDY.md` still listed this work as not done five days
        after it was done, on the strength of that output. A measurement nobody
        can see in the report is not evidence.
        """
        def render(label: str, info: dict, governs: str) -> str:
            if not info:
                return f"{label:<18}NOT RUN            governs {governs}"
            if info.get("skipped"):
                return f"{label:<18}SKIPPED            governs {governs}"
            state = "reached" if info.get("settled") else "TIMED OUT"
            return (f"{label:<18}{state} at {info.get('final_mhz', 0)} MHz"
                    f"   governs {governs}\n"
                    f"{'':<18}history {info.get('clock_history_mhz') or []}")

        settle = self.settle or {}
        return [
            render("settle compute", {k: v for k, v in settle.items()
                                      if k != "bandwidth_settle"},
                   "the bf16 and fp8 GEMMs"),
            render("settle memory", settle.get("bandwidth_settle") or {},
                   "the four bandwidth patterns"),
        ]


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


@dataclass(frozen=True)
class LoadedClock:
    """The clock a measurement actually ran at, plus how sure of it we are.

    Every field is here because a scalar was not enough. `samples` is kept so a
    bimodal set (half at the plateau, half at boost) is visible rather than
    averaged into a plausible middle; `spread_pct` is the summary a gate reads;
    `after_idle_mhz` is the single post-hoc sample the old code published, kept
    beside the honest one so the size of the artefact is auditable.
    """

    label: str
    samples: tuple[int, ...]
    median_mhz: int
    spread_pct: float
    #: Post-hoc sample, taken after synchronise with the GPU idle. This is what
    #: `gemm_clock_mhz` used to be, and on the H200 it ran 1485-1935 across
    #: eleven calibrations of one card.
    after_idle_mhz: int
    temp_c: int = 0
    #: Board power under the load, W, or 0.0 when nvidia-smi did not answer.
    #: A GEMM pinned at the board limit is power capped, and a clock read near
    #: boost for such a GEMM is not credible whatever NVML said.
    power_w: float = 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["samples"] = list(self.samples)
        return d


def _power_draw_w() -> float:
    """Board power in watts, or 0.0 when it cannot be read.

    0.0 rather than a refusal ONLY because nothing divides by it: it is recorded
    to separate a power-capped GEMM from a clock-limited one, and an unavailable
    reading leaves that question open rather than answering it wrongly.
    """
    vals = T._nvidia_smi("power.draw")
    if not vals:
        return 0.0
    try:
        return float(vals[0].split()[0])
    except (ValueError, IndexError):
        return 0.0


def clock_under_load(step, label: str, samples: int = 5,
                     seconds_per_sample: float = 0.4) -> LoadedClock:
    """Sample the SM clock WHILE `step` is running, not after it has stopped.

    THE FAILURE THIS EXISTS FOR, measured 2026-09-02 across the eleven committed
    H200 calibrations. `measure_bf16_gemm` sampled the clock immediately after
    `time_eager` returned, and `time_eager` ends with a synchronise, so the
    sample was taken with the GPU idle and already climbing back toward its
    1980 MHz idle boost. That field reads 1485, 1500, 1515, 1530, 1530, 1560,
    1560, 1845, 1845, 1905 and 1935 MHz for the same card and the same kernel,
    a 30% spread, while the achieved rate moved 12% and the compute settle
    plateau in the same files sat at 1455-1515. `sustained_peak_tflops` is
    linear in it, so `gemm_efficiency_pct` reads 87.4% in one file and 68.4% in
    another for the same GEMM.

    The method is the one `settle_clocks` already proved: feed the queue for a
    while so the governor is responding to a real load, then sample with work
    still in flight, and only then synchronise. Sampling between two synchronises
    measures an idle GPU no matter how much work surrounds it.

    Raises `ClockUnavailable` rather than returning zeros. Three usable samples
    is the floor because two cannot disagree with each other.
    """
    import time

    T.require_cuda()
    step()
    torch.cuda.synchronize()

    got: list[int] = []
    temps: list[int] = []
    powers: list[float] = []
    for _ in range(samples):
        stop = time.monotonic() + seconds_per_sample
        while time.monotonic() < stop:
            step()
            torch.cuda.synchronize()
        # Queue work and sample BEFORE draining it: the point of the whole
        # function is that the GPU is busy at the instant of the sample.
        for _ in range(4):
            step()
        state = T.ClockState.sample()
        powers.append(_power_draw_w())
        torch.cuda.synchronize()
        if state.sm_clock_mhz > 0:
            got.append(state.sm_clock_mhz)
            temps.append(state.temp_c)

    after = T.ClockState.sample().sm_clock_mhz
    if len(got) < 3:
        raise ClockUnavailable(
            f"{label}: only {len(got)} usable SM clock samples out of {samples}; "
            "NVML returned zero or was unavailable. Nothing normalised against a "
            "clock may be quoted from this run.")
    spread = (max(got) - min(got)) / max(got) * 100.0
    live = [p for p in powers if p > 0]
    return LoadedClock(
        label=label, samples=tuple(got), median_mhz=int(statistics.median(got)),
        spread_pct=round(spread, 2), after_idle_mhz=after,
        temp_c=int(statistics.median(temps)) if temps else 0,
        power_w=round(statistics.median(live), 1) if live else 0.0)


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
                      iters: int = 30, trials: int = 3, l2_flush: bool = True,
                      refusals: list[str] | None = None) -> list[BandwidthResult]:
    """STREAM-style patterns on buffers far larger than L2.

    All of them are measured because they genuinely differ, and a single figure
    would not show that. Byte counts are the compulsory traffic each pattern
    must move:

      read_stream  1N  Triton; each program stores one float, no global combine
      read_reduce  1N  ATen; reads every element, writes `rows` floats
      write        1N  a fill; writes every element, reads none
      copy         2N  one read plus one write per element
      triad        3N  two reads plus one write per element

    THE TWO READS ARE NOT REDUNDANT. `read_reduce` is a reduction, so it bounds
    the machine's read rate from BELOW; `read_stream` removes the cross-CTA
    combine entirely and bounds it from closer. The gap between them is what the
    reduction shape still costs, which was 1.7% the last time this file changed
    a reduction's shape. When Triton is unavailable `read_stream` is ABSENT and
    its reason is appended to `refusals`; it is never zero and never quietly the
    reduction's number.

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
    # A view, not a copy: same memory, so copy and triad are unaffected. Rows
    # are chosen to divide n exactly; a remainder would silently shorten the
    # read and overstate its bandwidth.
    _rows = next((r for r in (4096, 2048, 1024, 512, 64, 8, 1) if n % r == 0), 1)
    a2d = a.view(_rows, n // _rows)
    sink_col = torch.zeros(_rows, device="cuda", dtype=torch.float32)

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
        # MEASURED 2026-08-27. This was `torch.sum(a, dim=0, out=scalar_sink)`
        # on a 1-D buffer: a full tree reduction to ONE value, which is the most
        # reduction-limited shape available and 1.7% below what the same op
        # reaches on a 2-D buffer. Reducing a [rows, n/rows] view along the
        # CONTIGUOUS axis instead gives `rows` independent reductions with no
        # global combine, and measures DRAM rather than ATen's reduction tree.
        # Traffic is still 1N: the output is `rows` floats against n read.
        #
        # RENAMED 2026-09-02 from `read`. It is a reduction, so it is a lower
        # bound on the read rate and not the read rate, and calling it `read`
        # was what let it be described as "the closest analogue to streaming
        # expert weights" while triad stayed the ceiling and nobody had to
        # reconcile the two.
        run("read_reduce", lambda: torch.sum(a2d, dim=1, out=sink_col), nbytes,
            "ATen reduction along the contiguous axis; a LOWER BOUND on the "
            "DRAM read rate, not the read rate"),
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

    # The probe goes LAST so a Triton compile failure costs nothing that was
    # already measured, and so its several-second compile does not sit between
    # the settle and the first pattern.
    from .read_probe import ProbeUnavailable, make_stream_read
    try:
        call, read_bytes, write_bytes = make_stream_read(a)
    except (ProbeUnavailable, ValueError) as exc:
        # An absent pattern plus a recorded reason. The alternative -- falling
        # back to the reduction under the name `read_stream` -- would put two
        # different measurements under one label across machines, which is how
        # a ruler stops being a ruler.
        if refusals is not None:
            refusals.append(f"read_stream: {exc}")
    else:
        # Byte count is the READ only. The per-program stores are counted and
        # asserted small rather than assumed: at the 8 GiB default they are
        # 0.01% of traffic, and if a future block size made them material this
        # note is where the number would stop being credible.
        out.insert(0, run(
            "read_stream", call, read_bytes,
            f"Triton, one store per program, no cross-CTA combine; stores are "
            f"{100.0 * write_bytes / read_bytes:.3f}% of traffic and are not "
            "counted"))
    # No `del` here: the timing lambdas close over these buffers, and unbinding
    # the names would leave those closures referring to deleted locals. They are
    # freed when this frame exits; the caller reclaims the memory.
    return out


#: Dense BF16 FLOP per SM per clock, by compute capability. The Hopper value
#: is confirmed exactly against the datasheet: 132 SM x 4096 x 1830 MHz =
#: 989.4 TFLOP/s, which is NVIDIA's published H200 SXM figure. That also reveals
#: what the datasheet number assumes: a 1830 MHz boost clock.
# Dense BF16 FLOP per SM per clock, by compute capability. Each is pinned by
# reproducing the vendor's headline figure from SM count and boost clock:
#   sm_90  132 SM x 4096 x 1830 MHz = 989.4 TFLOP/s   (H200 SXM)
#   sm_80  108 SM x 2048 x 1410 MHz = 311.8 TFLOP/s   (A100 SXM, published 312)
# Ampere does half of Hopper's bf16 per SM per clock. Absent entries return None
# rather than a guess, since the whole point is to normalise against silicon.
_DENSE_BF16_FLOP_PER_SM_CLK = {(9, 0): 4096, (8, 0): 2048}

#: The same table for fp8. Hopper and Blackwell run fp8 tensor cores at twice
#: the bf16 rate; Ampere has none, so sm_80 is ABSENT rather than zero. An entry
#: there would imply a peak for silicon that cannot run the format.
#:
#: This ratio is what STUDY.md's C2 currently assumes when it derives
#: ridge_fp8 = 2 * ridge_bf16. The constant is the datasheet relationship; the
#: MEASURED fp8 GEMM below is what can disagree with it, the way the bf16 GEMM
#: already disagrees with its own headline (701.6 achieved against 989.4).
_DENSE_FP8_FLOP_PER_SM_CLK = {(9, 0): 8192, (10, 0): 16384}


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
    that unpacked two values -- a failure that only shows up on the GPU, six
    minutes into a run. A dataclass makes the next added field free.
    """

    tflops: float
    shape: tuple[int, int, int]
    #: Median of the samples taken WHILE the GEMM was running. Not the post-hoc
    #: sample: see `clock_under_load` for the 30% spread that one had.
    sm_clock_mhz: int
    #: The full record, or None when the clock could not be established. None is
    #: a state a caller must handle, not a zero it can divide by.
    clock: LoadedClock | None = None


def sustained_peak_tflops_fp8(sm_clock_mhz: float) -> float | None:
    """fp8 silicon ceiling at this clock, or None where there is no fp8.

    None on Ampere is the point: the A100 has no fp8 tensor cores, and a number
    here would be a peak for a format the part cannot execute.
    """
    if not torch.cuda.is_available():
        return None
    per_clk = _DENSE_FP8_FLOP_PER_SM_CLK.get(torch.cuda.get_device_capability())
    if per_clk is None:
        return None
    props = torch.cuda.get_device_properties(0)
    return props.multi_processor_count * per_clk * sm_clock_mhz * 1e6 / 1e12


def measure_fp8_gemm(n: int = 8192, warmup: int = 5, iters: int = 20,
                     trials: int = 3,
                     refusals: list[str] | None = None) -> GemmResult | None:
    """Achievable dense fp8 through cuBLAS, or None where fp8 is unavailable.

    `torch._scaled_mm`, not `torch.mm`: fp8 tensors carry a scale and the plain
    path takes none.

    Per-tensor scales of 1.0, because this measures the RATE, not accuracy. The
    inputs are drawn small enough to stay in range, so no scaling is needed and
    none is charged to the timing.

    The result is what lets C2 stop assuming. STUDY.md derives
    ridge_fp8 = 2 * ridge_bf16 from the datasheet ratio; this measures whether
    fp8 reaches the same fraction of its peak that bf16 reaches of its own.
    """
    T.require_cuda()
    if sustained_peak_tflops_fp8(1000.0) is None:
        return None
    dt = torch.float8_e4m3fn
    # Small values so a unit scale is honest: fp8_e4m3 saturates at 448.
    a = (torch.randn((n, n), device="cuda") * 0.1).to(dt)
    b = (torch.randn((n, n), device="cuda") * 0.1).to(dt).t()
    one = torch.tensor(1.0, device="cuda")

    def step():
        torch._scaled_mm(a, b, one, one, out_dtype=torch.bfloat16)

    try:
        res = T.time_eager(step, warmup=warmup, iters=iters, trials=trials,
                           l2_flush=False)
    except RuntimeError as e:
        print(f"[calibrate] fp8 GEMM unavailable: {e}")
        return None
    tflops = (2.0 * n * n * n) / (res.ms_p50 * 1e-3) / 1e12
    return _with_clock(step, "fp8 GEMM", tflops, n, refusals)


def _with_clock(step, label: str, tflops: float, n: int,
                refusals: list[str] | None) -> GemmResult:
    """Attach the under-load clock to a finished GEMM, or record why there is none.

    The TFLOP/s is already measured by the time this runs and is valid without a
    clock: it is FLOP over seconds and nothing normalises it. Only
    `sustained_peak_tflops` and `gemm_efficiency_pct` need the clock, and both
    already return None when there is none. So an NVML-less box gets its
    ceilings, loses exactly the two figures that depend on a clock, and says so
    -- rather than losing the whole calibration to a refusal about one field.
    """
    try:
        clock = clock_under_load(step, label)
    except ClockUnavailable as exc:
        if refusals is not None:
            refusals.append(str(exc))
        return GemmResult(tflops=tflops, shape=(n, n, n), sm_clock_mhz=0,
                          clock=None)
    return GemmResult(tflops=tflops, shape=(n, n, n),
                      sm_clock_mhz=clock.median_mhz, clock=clock)


def measure_bf16_gemm(n: int = 8192, warmup: int = 5, iters: int = 20,
                      trials: int = 3,
                      refusals: list[str] | None = None) -> GemmResult:
    """Achievable dense BF16 through cuBLAS: the compute roof this box gives.

    A square GEMM at n=8192 is comfortably compute bound and is what a tuned
    library achieves, so it is the fair ceiling for a hand-written kernel.
    """
    T.require_cuda()
    a = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((n, n), device="cuda", dtype=torch.bfloat16)

    def step():
        torch.mm(a, b, out=out)

    res = T.time_eager(step, warmup=warmup, iters=iters, trials=trials,
                       l2_flush=False)
    tflops = (2.0 * n * n * n) / (res.ms_p50 * 1e-3) / 1e12
    # Sample the clock UNDER LOAD. The comment here used to say "DURING the
    # measurement" while the code sampled after `time_eager` had synchronised,
    # i.e. with the GPU idle; that field then moved 30% across eleven
    # calibrations of one card while the achieved rate moved 12%. It costs about
    # two seconds of extra GEMM to sample it properly.
    return _with_clock(step, "bf16 GEMM", tflops, n, refusals)


#: What a read pattern's note says once it has been disowned. One string, in one
#: place, because `matched_pattern` and the ceiling guard both test for it and a
#: typo in either would silently re-admit a figure this file rejected.
INVALID_READ_NOTE = f"consumer-limited, not DRAM-limited; {DISOWNED}"


def demote_invalid_reads(patterns) -> tuple[BandwidthResult, ...]:
    """Disown any read pattern that came in below triad.

    A stream that only reads cannot legitimately be slower than one that reads
    twice and writes once, so a read below triad is measuring its CONSUMER --
    ATen's reduction tree, or a Triton probe whose accumulator serialised the
    loads -- and not the DRAM read rate. It stays in the file, because "this
    formulation reaches 1744 GB/s" is a fact worth keeping (the A100's
    `read_reduce` is exactly that), but its note says it is not a denominator
    and `matched_pattern` will not build a ridge band out of it.

    Pure, and separate from `calibrate`, so the rule can be tested on a laptop.
    The version this replaces was inline, matched the literal name `read`, and
    would have silently stopped firing the moment that pattern was renamed.
    """
    patterns = tuple(patterns)
    triad = next((p for p in patterns if p.pattern == "triad"), None)
    if triad is None:
        return patterns
    return tuple(
        BandwidthResult(**{**p.as_dict(), "note": INVALID_READ_NOTE})
        if p.pattern in READ_PATTERNS and p.gbps < triad.gbps else p
        for p in patterns)


def calibrate(target_bytes: int = DEFAULT_BUFFER_BYTES, gemm_n: int = 8192,
              ceiling: str = DEFAULT_CEILING, settle: bool = True,
              settle_seconds: float = 30.0) -> Calibration:
    T.require_cuda()
    # Settle FIRST. Measuring from idle put this machine's first pattern at
    # 840 MHz and its last at 1980, which is a bigger effect than anything the
    # choice of pattern argues about.
    # ONE refusal list for the whole calibration, opened before the first
    # measurement. Anything that cannot be measured lands here with its reason
    # and is absent from the results; nothing anywhere below substitutes a
    # plausible value for something it did not measure.
    refusals: list[str] = []
    settle_info = (settle_clocks(settle_seconds, load="compute") if settle
                   else {"settled": False, "skipped": True})
    before = T.ClockState.sample()

    # COMPUTE FIRST. Measured on an H200 SXM, the cuBLAS figure fell 795 -> 725
    # -> 687 TFLOP/s across three runs as each did more sustained memory work
    # before it. On a 700 W part, heavy HBM traffic eats the power budget the
    # tensor cores need, so a compute roof measured after a bandwidth sweep is a
    # power-limited number, not a ceiling. The settle above is matmul work, so
    # the GPU is already in the right state for exactly this measurement.
    gemm = measure_bf16_gemm(gemm_n, refusals=refusals)
    torch.cuda.empty_cache()

    # fp8 IMMEDIATELY AFTER, for the reason the comment above gives: the GPU is
    # still in its compute steady state, and any bandwidth work before a compute
    # roof turns it into a power-limited number rather than a ceiling.
    #
    # This is what lets C2 stop assuming. STUDY.md derives
    # ridge_fp8 = 2 * ridge_bf16 from the datasheet ratio; measuring both says
    # whether fp8 reaches the same fraction of ITS peak that bf16 reaches of its
    # own. The bf16 figure is already 701.6 against a 989.4 headline, so the
    # datasheet ratio surviving to the achieved numbers is a claim, not a given.
    fp8_gemm = measure_fp8_gemm(gemm_n, refusals=refusals)
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
    # the settle did not, so pass one is warmup and pass two has every pattern
    # in the same state. Without this the first pattern measured is
    # systematically different from the rest. The first pass also absorbs the
    # Triton compile of the stream probe, which is seconds and would otherwise
    # land inside a timed window.
    first_pass = measure_bandwidth(target_bytes)
    patterns = measure_bandwidth(target_bytes, refusals=refusals)
    torch.cuda.empty_cache()
    after = T.ClockState.sample()

    patterns = demote_invalid_reads(patterns)
    chosen = next((p for p in patterns if p.pattern == ceiling), None)
    if chosen is None:
        raise ValueError(
            f"unknown ceiling pattern {ceiling!r}; "
            f"measured: {[p.pattern for p in patterns]}")
    if DISOWNED in chosen.note:
        raise ValueError(
            f"{ceiling} measured {chosen.gbps:.0f} GB/s, below triad. A read "
            "stream cannot be slower than 2R+1W at the DRAM level, so that "
            "figure is the consumer's rate and not bandwidth. Use "
            f"--ceiling {DEFAULT_CEILING}.")

    drift, throttled = T.clock_drift(before, after)
    return Calibration(
        gpu_name=torch.cuda.get_device_properties(0).name,
        achieved_bandwidth_gbps=chosen.gbps,
        ceiling_pattern=ceiling,
        achieved_bf16_tflops=gemm.tflops,
        achieved_fp8_tflops=(fp8_gemm.tflops if fp8_gemm else None),
        fp8_gemm_clock_mhz=(fp8_gemm.sm_clock_mhz if fp8_gemm else 0),
        bandwidth_patterns=tuple(patterns),
        gemm_shape=gemm.shape,
        gemm_clock_mhz=gemm.sm_clock_mhz,
        buffer_bytes=target_bytes,
        clocks={"sm_start_mhz": before.sm_clock_mhz, "sm_end_mhz": after.sm_clock_mhz,
                "temp_start_c": before.temp_c, "temp_end_c": after.temp_c,
                "drift_pct": round(drift, 2), "throttled": throttled},
        settle={**settle_info, "bandwidth_settle": settle_mem},
        warmup_pass={p.pattern: round(p.gbps, 1) for p in first_pass},
        gemm_clock=gemm.clock.as_dict() if gemm.clock else {},
        fp8_gemm_clock=(fp8_gemm.clock.as_dict()
                        if fp8_gemm and fp8_gemm.clock else {}),
        refusals=tuple(refusals),
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


# --- what a calibration FILE says about itself --------------------------------
#
# Everything above measures a machine. Everything below is about the file that
# measurement is written to, and specifically about telling one such file from
# another after both have left the pod.


@dataclass(frozen=True)
class CalibrationStamp:
    """The identifying half of a written calibration: which session produced it.

    `scripts/calibrate_hardware.py` writes `checked_on`, `measured_commit` and
    `measured_dirty` beside the ceilings, and until 2026-08-31 nothing read them
    back. They matter because `moe/bench/hardware/measured_<device>.yaml` is ONE
    FILE PER DEVICE that every new calibration overwrites, and
    `publish_results.sh` copies whatever is in it at publish time. A sweep that
    finishes before a recalibration and publishes after one therefore ships a
    ruler it never used, and nothing about the copied file looks wrong.

    THAT HAPPENED, and this is the file it happened to. Reconstructed from git,
    all times UTC:

        18:08-19:21  2026-08-28-...-h200-whole-layer sweeps at commit 873183a,
                     its rows stamped 701.61 TFLOP/s and 4377.21 GB/s from the
                     calibration then on disk (committed 04:12 as a6ee65d)
        19:29        89f9f7a overwrites measured_nvidia_h200.yaml with a fresh
                     calibration, 770.92 TFLOP/s and 4374.49 GB/s
        19:35        the arm is published and copies the NEW file

    The result is an arm whose measured.yaml is byte-identical to the one beside
    a different arm, `-fp8-refixed`, whose sweep started at 19:49. Nobody noticed
    for three days. The cost is claim C5: its target is the band 0.81-0.91
    instead of a number, because 176.2 and 160.3 are both defensible rulers for
    that arm and neither was measured in its session.

    A stamp only means something next to rows, so the comparison lives in
    `moe.bench.published.calibration_provenance`. This is the half that reads the
    file, and it is here because this module is what a calibration IS.
    """
    #: Where it was read from, so a verdict can name the file it judged.
    path: str
    name: str
    gpu_name: str
    #: `None` when absent or unparseable. Day resolution, which is why it did
    #: NOT catch the whole-layer swap: both sessions ran on 2026-08-28.
    checked_on: date | None
    #: Empty when the file records none. The A100 calibration is exactly that,
    #: so nothing in it can be compared against the sha its rows carry.
    measured_commit: str
    measured_dirty: bool
    bandwidth_gbps: float
    peak_tflops: dict[str, float]
    ceiling_pattern: str

    def peak_for(self, dtype: str) -> float | None:
        """TFLOP/s claimed for `dtype`, or None when the file claims none.

        None rather than 0.0, because absence is real: the A100 file has no fp8
        key at all, its silicon having no fp8 tensor cores, and a zero there
        would read as a measured zero. Rows swept in a dtype their calibration
        does not cover carry `achieved_peak_tflops = 0.0` for the same reason --
        all 19,908 of `-fp8-three-kernel` do.
        """
        value = self.peak_tflops.get(dtype)
        return float(value) if value else None

    def ridge(self, dtype: str) -> float | None:
        """FLOP per byte above which `dtype` can be compute bound on this ruler."""
        return ridge_flop_per_byte(self.peak_for(dtype), self.bandwidth_gbps)


def ridge_flop_per_byte(peak_tflops: float | None,
                        bandwidth_gbps: float | None) -> float | None:
    """The roofline ridge in the units the yaml and the CSV both already use.

    `roofline.Hardware.ridge_point` is the same quantity in SI: FLOP/s over
    byte/s. Both the calibration file and every published row hold TFLOP/s and
    GB/s instead, so converting each pair to SI just to divide them introduces
    two multiplications and a chance to drop one. The factor of 1000 is
    1e12 / 1e9.

    None when either term is missing or non-positive, so a caller that has no
    ridge is handed nothing rather than a zero it might quote.
    """
    if not peak_tflops or not bandwidth_gbps:
        return None
    if peak_tflops <= 0 or bandwidth_gbps <= 0:
        return None
    return peak_tflops * 1000.0 / bandwidth_gbps


def read_stamp(path) -> CalibrationStamp:
    """Read the identifying fields out of a calibration yaml.

    Tolerant where `recompute.load_calibration_hardware` is strict, and for the
    opposite reason. That function is about to divide every efficiency column by
    the bandwidth, so a missing field there is fatal. This one is about to report
    what can and cannot be established, so a missing field is an ANSWER -- the
    `unknown` verdict -- and raising would turn "I cannot tell" into "the file is
    broken". The A100 calibration ships an empty `measured_commit` and is read
    here without complaint.
    """
    import yaml

    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a mapping")
    detail = doc.get("detail") or {}
    memory = doc.get("memory") or {}
    peaks = {}
    for dtype, value in (doc.get("compute_dense_tflops") or {}).items():
        if value:
            peaks[str(dtype)] = float(value)
    bw_tb_s = memory.get("bandwidth_tb_s") or 0.0
    return CalibrationStamp(
        path=str(path),
        name=str(doc.get("name", "")),
        gpu_name=str(detail.get("gpu_name") or doc.get("name") or ""),
        checked_on=_as_date(doc.get("checked_on")),
        measured_commit=str(doc.get("measured_commit") or ""),
        measured_dirty=bool(doc.get("measured_dirty")),
        bandwidth_gbps=float(bw_tb_s) * 1000.0,
        peak_tflops=peaks,
        ceiling_pattern=str(detail.get("ceiling_pattern") or ""),
    )


def _as_date(value) -> date | None:
    """`checked_on` is written quoted, but a hand-edited file may not be.

    PyYAML resolves an unquoted 2026-08-28 to a `datetime.date` and a quoted one
    to a string, so both shapes reach here from files that look identical in a
    diff.
    """
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
