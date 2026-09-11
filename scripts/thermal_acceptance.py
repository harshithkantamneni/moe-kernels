#!/usr/bin/env python
"""Can THIS rented card hold its clock under sustained load, before we pay it?

    python scripts/thermal_acceptance.py --dry-run    # the plan and the cost, free
    python scripts/thermal_acceptance.py --self-test  # the scorer, on planted worlds
    python scripts/thermal_acceptance.py              # the pod run: load it and watch

WHAT THIS EXISTS FOR, measured on a RunPod H200 on 2026-09-11. The card boosted
to 1980 MHz, its maximum, then collapsed to 345 MHz, its floor, within ~30 s of
sustained bf16 GEMM, and stayed there. `nvidia-smi` clocks_event_reasons went
0x0 -> 0x20 (SwThermalSlowdown) -> 0x68 (HwSlowdown | SwThermalSlowdown |
HwThermalSlowdown). At the floor it drew only ~240 W of a 700 W limit and STILL
climbed from 87 C to 93 C, which is a cooling fault and not a workload.

`scripts/calibrate_hardware.py` ran to completion on that card, published a
tracked yaml, and its `not_throttled` CLAIM gate PASSED: that gate scored DRIFT,
and a card pinned flat at its floor has first == last. The calibration it
published reported ridge 73.6 where this card's real ridge is near 156, because
the compute peak collapsed with the clock while the memory side did not. Every
arm below it would have been scored against that ruler.

So this arm runs FIRST, ahead of the calibration, and it REFUSES the session
rather than warning about it. A warning costs a whole rental to ignore.

WHAT IT MEASURES. One sustained dense bf16 GEMM -- literally
`moe.bench.calibrate._load_compute`, the same load the calibration settles on,
so the probe stresses the card exactly as arm 1 will -- with the SM clock,
board power and temperature polled through NVML while the queue is busy. The
first `--settle-seconds` are discarded as the governor's ramp; the rest is the
scored window. The verdict is on the MEDIAN of that window, never on a sample:
healthy cells in `results/published` post individual readings down to 405 MHz on
a 1980 MHz part during one drain-and-ramp, and a per-sample gate would have no
margin left between that and the fault.

THE THRESHOLD IS A RELATION AND NOT A NUMBER. `timing.THERMAL_FLOOR_FRACTION`
of the card's OWN maximum SM clock, read off the device by
`timing.max_sm_clock_mhz` and snapped to the 15 MHz NVML grid. That constant's
comment carries the derivation: the lowest per-cell `sm_clock_load_mhz` median
anywhere in `results/published` is 1275 of 1980 (0.6439, drawn at 697.4 W of a
700 W cap, so a hungry tile and not a sick card) and the fault above is 345 of
1980 (0.1742); one third is the geometric midpoint of that window and lands on
660 MHz, a clock the grid can report. Nothing here is a clock literal, and the
same file on an A100 gates at 465 MHz without being told.

WHAT IT WRITES, under `$MOE_RESULTS_DIR` or `/workspace/results` or `<repo>/results`:

    <results>/thermal_acceptance/<run-id>/report.txt   exactly what was printed
    <results>/thermal_acceptance/<run-id>/report.json  the trace and the gates

Nothing is written into the tracked tree, by any flag. `git check-ignore` is run
on both paths and its answer is printed, because `results/*` is ignored with
only `!results/published/` excepted.

THE SLOWDOWN REASON MASK IS A RECORD AND NOT A GATE. NVML's
`clocks_event_reasons` is what NAMED the fault above, and every poll's mask is
recorded and printed. It is deliberately not scored: a driver or a container
that does not expose the field would then refuse a healthy card, which is an
expensive false refusal for a term the clock already answers. The clock is the
quantity that makes a ceiling wrong; the mask is why.

EXIT CODES are `moe.bench.exit_codes`. `--dry-run` measured nothing and scored
no gate, so it exits REFUSED (2) and prints no RESULT line; `--self-test` scores
one VALIDITY gate over planted worlds and exits through the table; the measuring
run exits DONE, CLAIM_FAIL when the card cannot hold its clock -- which is a
RESULT about this pod and never a retry -- or INVALID when the probe itself
could not be trusted.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench import exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench import timing as T  # noqa: E402

#: What actually ran, named rather than assumed. NOT `timing.TIMING_BASIS`:
#: nothing here is timed at all. The quantity is a clock, the instrument is
#: NVML, and the load is the calibration's own compute step; a page that
#: claimed the ladder's timing basis would be claiming an instrument it never
#: used, which is the confusion `dram_counter_route` keeps four separate
#: instrument strings to avoid.
INSTRUMENT = ("scripts/thermal_acceptance.py: sustained dense bf16 GEMM "
              "(moe.bench.calibrate._load_compute, 8192^3, 32 per step) with "
              "SM clock, board power and temperature polled through NVML while "
              "the queue is busy; NOT timing.TIMING_BASIS, nothing is timed")

#: Seconds of load discarded before the window is scored. THE GOVERNOR'S RAMP
#: IS NOT THE FAULT: `calibrate.settle_clocks` documents an 840 -> 1980 MHz walk
#: on this card and takes up to 30 s to plateau, so a window opened at t=0 would
#: read a rising clock as drift on every healthy card. 30 s is that function's
#: own budget, taken from it rather than invented here.
SETTLE_SECONDS = 30.0

#: Seconds of scored window. The 2026-09-11 card collapsed within ~30 s of load
#: reaching it, so this is four times the observed time-to-fault AFTER the ramp
#: has been discarded, and the whole arm still costs less than a calibration.
WINDOW_SECONDS = 120.0

#: Seconds between polls. `timing.CLOCK_POLL_SECONDS` is sized for a per-cell
#: region measured in hundreds of milliseconds; this window is two minutes and
#: a poll that often would add nothing but rows. At 2 s the scored window holds
#: 60 samples, twenty times the floor below.
POLL_SECONDS = 2.0

#: The non-vacuity floor on the scored window, in samples. Three times
#: `timing.CLOCK_SAMPLE_FLOOR`, because the window is read as three parts -- a
#: head, a middle and a tail -- and each of them is held to the same floor a
#: single under-load median is held to everywhere else in this repository. A
#: probe that sampled nothing also reports no throttling.
SCORED_SAMPLE_FLOOR = 3 * T.CLOCK_SAMPLE_FLOOR

#: How much of the scored window the head and the tail each take. A third, so
#: the two are disjoint and the middle is neither; the drift term compares the
#: card at the start of the window with the card at the end of it.
END_FRACTION = 1.0 / 3.0

#: The card label a run that measures NOTHING carries. `provenance.run_id`
#: refuses an id without a card, and `--dry-run` and `--self-test` have none.
NO_CARD = "no-card-nothing-measured"


# --------------------------------------------------------------------------
# Registered predictions. Printed before anything is measured, never rewritten.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    """One registered prediction: what it says, and what a FAIL would mean.

    The `fail` field is what makes it a prediction rather than a description: a
    gate whose failure has no stated consequence is a gate nobody has to
    honour.
    """
    number: int
    claim: str
    numbers: str
    fail: str

    def render(self) -> list[str]:
        return [f"P{self.number}  {self.claim}",
                f"      predicts  {self.numbers}",
                f"      a FAIL    {self.fail}"]


PREDICTIONS = (
    Prediction(
        1, "a healthy card holds a clock well clear of its own floor",
        f"the median SM clock over the scored window is at least "
        f"{T.THERMAL_FLOOR_FRACTION:.4f} of this card's maximum. Every one of "
        "the 5,260 published per-cell medians in this repository sits 1.93x "
        "above that line; the 2026-09-11 fault sat 1.91x below it",
        "THE SESSION IS REFUSED AND THE POD IS RETURNED. A card that cannot "
        "hold a clock produces a ruler whose compute peak has collapsed while "
        "its memory side has not, and every roof fraction, ridge and alpha "
        "measured against that ruler is wrong rather than merely low."),
    Prediction(
        2, "and it is still holding it at the end of the window, not on the "
           "way down",
        f"the last third of the window agrees with the first third within "
        f"{T.DRIFT_FRACTION:.2f} of it, the same fraction `timing.clock_flags` "
        "applies to a cell, after the first "
        f"{SETTLE_SECONDS:.0f} s have been discarded as the governor's ramp",
        "the card was still collapsing when the probe stopped. The median may "
        "be above the floor and the ceilings still unmeasurable, because the "
        "calibration that follows takes minutes and this window took two."),
)


# --------------------------------------------------------------------------
# Gates. A number against a threshold, PASS or FAIL, and what a FAIL costs.
# --------------------------------------------------------------------------

PASS, FAIL, UNKNOWN, UNDECIDED = "PASS", "FAIL", "UNKNOWN", "UNDECIDED"

#: A FAIL here means nothing on the page may be quoted.
VALIDITY = "VALIDITY"
#: A FAIL here IS a result: this pod cannot be used.
CLAIM = "CLAIM"


@dataclass
class Gate:
    kind: str
    number: str
    claim: str
    verdict: str
    measured: str
    threshold: str
    #: What a non-PASS on this gate voids. Printed only when it is not a PASS,
    #: so a reader of a failing page is told the consequence beside the number
    #: rather than having to find it in a docstring.
    invalidates: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def token(self) -> str:
        """The gate's identifier as ONE whitespace-free token, for `RESULT:`.

        `kind` and `number` are separate fields and a result line's name must
        be one token, so the two are joined: VALIDITY 1 is `V1`, CLAIM 1 is
        `C1`. Joined rather than truncated to the number, because a file with
        both a VALIDITY 1 and a CLAIM 1 would otherwise print two gates the
        driver cannot tell apart.
        """
        return f"{self.kind[0]}{self.number}"

    def result_line(self) -> str | None:
        """The one line the session driver may grep for this gate, or None.

        None for an UNDECIDED gate, which is NOT `exit_codes.UNKNOWN`. UNKNOWN
        means "scored, could not decide" and counts against the gate;
        UNDECIDED means the gate could not RUN on this machine at all, and
        printing UNKNOWN for it would classify a run that made no statement as
        a failed claim. `scored` applies the same rule, so the printed set and
        the classified set are identical by construction.
        """
        if self.verdict == UNDECIDED:
            return None
        return exit_codes.result_line(
            exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
            self.token, self.verdict,
            f"{self.claim}: measured {self.measured}")

    def scored(self) -> tuple[str, str, str] | None:
        if self.verdict == UNDECIDED:
            return None
        return (exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
                self.token, self.verdict)

    def render(self) -> list[str]:
        line = self.result_line()
        out = [line] if line else []
        out += [f"{self.kind} {self.number}  {self.verdict:9s} {self.claim}",
                f"{'':>11}measured {self.measured}   gate {self.threshold}"]
        if self.verdict != PASS and self.invalidates:
            out.append(f"{'':>11}a non-PASS here voids {self.invalidates}")
        out += [f"{'':>11}{text}" for text in self.lines]
        return out


# --------------------------------------------------------------------------
# The trace, and the arithmetic over it. Pure: no torch, no GPU, no clock.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """One poll of the card while the load was in flight."""
    #: Seconds since the load started, so the settle prefix can be dropped by
    #: TIME rather than by sample count: a poll that took longer than expected
    #: must not shift which part of the window is called the ramp.
    t: float
    mhz: int
    temp_c: int
    power_w: float
    #: `timing.CLOCK_SOURCE_*`. A forked nvidia-smi sample lands tens of
    #: milliseconds after it was asked for, which is fine for a static maximum
    #: and not fine for a reading that has to describe a card under load.
    source: str
    #: The NVML slowdown mask at this poll, decoded, or "" when the reader
    #: could not answer. RECORDED AND NOT SCORED; see the module docstring.
    reasons: str = ""


@dataclass(frozen=True)
class Trace:
    """A whole run of the probe, and every verdict derivable from it.

    Built by `sustain` on the pod and by `plant` in the self-test, so the
    scoring half is exercised off GPU against worlds whose answer is known.
    """
    samples: tuple[Sample, ...]
    settle_seconds: float
    window_seconds: float
    max_sm_clock_mhz: float | None
    max_sm_clock_source: str
    power_limit_w: float | None

    @property
    def scored(self) -> tuple[Sample, ...]:
        """The samples the verdict is taken over: everything after the ramp."""
        return tuple(s for s in self.samples if s.t >= self.settle_seconds)

    @property
    def spanned_seconds(self) -> float:
        got = self.scored
        return (got[-1].t - got[0].t) if len(got) >= 2 else 0.0

    @property
    def median_mhz(self) -> float | None:
        got = [s.mhz for s in self.scored if s.mhz > 0]
        return statistics.median(got) if got else None

    def _end_median(self, tail: bool) -> float | None:
        """Median of the first or last `END_FRACTION` of the scored window."""
        got = [s.mhz for s in self.scored if s.mhz > 0]
        if len(got) < SCORED_SAMPLE_FLOOR:
            return None
        n = max(T.CLOCK_SAMPLE_FLOOR, int(len(got) * END_FRACTION))
        return statistics.median(got[-n:] if tail else got[:n])

    @property
    def head_mhz(self) -> float | None:
        return self._end_median(tail=False)

    @property
    def tail_mhz(self) -> float | None:
        return self._end_median(tail=True)

    @property
    def floor_mhz(self) -> float | None:
        return T.thermal_floor_mhz(self.max_sm_clock_mhz)

    @property
    def floor_ok(self) -> bool | None:
        return T.clock_floor_ok(self.median_mhz, self.max_sm_clock_mhz)

    @property
    def steady_ok(self) -> bool | None:
        """Does the end of the window agree with its start, within DRIFT.

        `timing.clock_flags`' own DRIFT rule, applied to two MEDIANS rather
        than to two samples. The rule is the repository's; the pair it is
        applied to is this arm's, because one poll at each end of a two-minute
        window would make the verdict turn on two readings out of sixty.
        """
        head, tail = self.head_mhz, self.tail_mhz
        if head is None or tail is None or head <= 0:
            return None
        _level, drift = T.clock_flags(self.median_mhz, head, tail, None)
        return drift

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({s.source for s in self.scored}))

    @property
    def reason_masks(self) -> tuple[str, ...]:
        return tuple(sorted({s.reasons for s in self.samples if s.reasons}))

    @property
    def power_fraction(self) -> float | None:
        """Median board power over the window as a fraction of the limit.

        THE DISCRIMINATOR THAT NAMES THE CAUSE, and it is printed rather than
        gated. Of the 191 published rows whose clock sits below 0.70 of this
        card's maximum, the LOWEST board power is 685.6 W of a 700 W limit,
        0.979: a low clock in this corpus is always a hungry tile pinned at the
        power cap. The 2026-09-11 fault drew 240 W of 700, 0.343, at a clock
        five times lower. Low clock AND low power is a card in trouble; low
        clock at the cap is work.
        """
        got = [s.power_w for s in self.scored if s.power_w > 0]
        if not got or not self.power_limit_w:
            return None
        return statistics.median(got) / float(self.power_limit_w)


def gate_v1_non_vacuity(trace: Trace) -> Gate:
    """Did the probe actually watch a card under load for the time it booked.

    THE NON-VACUITY GATE, and it is first for the reason every model file in
    this repository puts one first: a probe that examined nothing also reports
    no throttling, and would hand the session a PASS earned by measuring
    nothing at all. Two conditions, because either one alone is satisfiable by
    a broken run: enough samples to take three medians from, and a window that
    actually spanned the seconds the plan charged for.
    """
    got = trace.scored
    want = trace.window_seconds * 0.9
    enough = len(got) >= SCORED_SAMPLE_FLOOR
    spanned = trace.spanned_seconds >= want
    return Gate(
        VALIDITY, "1", "the probe watched a loaded card for the booked window",
        PASS if (enough and spanned) else FAIL,
        f"{len(got)} scored samples over {trace.spanned_seconds:.0f} s "
        f"(of {len(trace.samples)} polled, {trace.settle_seconds:.0f} s of ramp "
        "discarded)",
        f">= {SCORED_SAMPLE_FLOOR} samples and >= {want:.0f} s",
        "every gate below, which would otherwise report a clean card from a "
        "window nobody measured")


def gate_v2_nvml(trace: Trace) -> Gate:
    """Did every scored sample come through NVML.

    `ClockState.sample` falls back to a forked `nvidia-smi` when pynvml is
    missing, and that fork plus an NVML init costs tens of milliseconds on
    Linux. `calibrate.clock_under_load` REFUSES such a sample for the roof's
    reference and `timing.nvml_clock_reader` never forks at all, both for the
    same reason: a reading that lands tens of milliseconds after it was asked
    for describes whatever the card was doing then, not while the queue was
    busy. A thermal verdict scored on forked samples has the same defect.
    """
    sources = trace.sources
    ok = bool(sources) and set(sources) == {T.CLOCK_SOURCE_NVML}
    return Gate(
        VALIDITY, "2", "every scored sample came through NVML",
        PASS if ok else FAIL,
        f"sources {', '.join(sources) or 'none'}",
        f"exactly {T.CLOCK_SOURCE_NVML!r}",
        "both claims below: a forked sample describes an idle card, and an "
        "idle card reads at its boost clock")


def gate_c1_floor(trace: Trace) -> Gate:
    """Is the median clock over the window above this card's thermal floor.

    THE ARM'S WHOLE PURPOSE. See `timing.THERMAL_FLOOR_FRACTION` for where the
    fraction comes from and why the reference is the card's own maximum rather
    than the clock any calibration reports: on a collapsed card the calibration
    collapses too, so every reference derived from it agrees with the fault.
    """
    floor, median = trace.floor_mhz, trace.median_mhz
    ok = trace.floor_ok
    lines = []
    fraction = trace.power_fraction
    if fraction is not None:
        lines.append(f"board power {100 * fraction:.1f}% of the "
                     f"{trace.power_limit_w:.0f} W limit over the same window. "
                     "A low clock AT the cap is a hungry kernel; a low clock "
                     "well under it is a card in trouble.")
    if trace.reason_masks:
        lines.append("NVML slowdown reasons seen: "
                     + "; ".join(trace.reason_masks)
                     + ". RECORDED, NOT SCORED: the clock is what makes a "
                       "ceiling wrong, this says why.")
    if ok is False:
        lines.append("DO NOT CALIBRATE THIS CARD. The ceilings would be "
                     "measured on silicon that is not clocking, the ridge "
                     "derived from them would be wrong rather than low, and "
                     "the 2026-09-11 pod published exactly that: ridge 73.6 "
                     "on a card whose ridge is near 156.")
    return Gate(
        CLAIM, "1", "the card holds a clock above its own thermal floor",
        PASS if ok else (FAIL if ok is False else UNKNOWN),
        (f"{median:.0f} MHz median over the window" if median is not None
         else "no usable clock sample"),
        (f">= {floor:.0f} MHz = {T.THERMAL_FLOOR_FRACTION:.4f} x this card's "
         f"{trace.max_sm_clock_mhz:.0f} MHz maximum, from "
         f"{trace.max_sm_clock_source}" if floor is not None else
         "no maximum SM clock could be read, so there is no floor to derive"),
        "the whole session: nothing measured on this pod is this card's "
        "ceiling and nothing scored against it may be quoted",
        lines)


def gate_c2_steady(trace: Trace) -> Gate:
    """Was the card still holding that clock when the window closed.

    A median can sit above the floor while the card is halfway down. The
    calibration that follows takes minutes and this window takes two, so a
    card that is still falling here is a card whose ceilings will be measured
    somewhere this probe never saw.
    """
    head, tail = trace.head_mhz, trace.tail_mhz
    ok = trace.steady_ok
    moved = (abs(tail - head) / head if (head and tail) else None)
    return Gate(
        CLAIM, "2", "the clock is steady from the start of the window to its end",
        PASS if ok else (FAIL if ok is False else UNKNOWN),
        (f"first third {head:.0f} MHz, last third {tail:.0f} MHz, "
         f"{100 * moved:.1f}% apart" if moved is not None else
         "too few samples to take two medians from"),
        f"within {T.DRIFT_FRACTION:.2f} of the first third, either direction",
        "the ceilings the calibration is about to measure, which will be "
        "taken minutes after this window closed")


def gates_for(trace: Trace) -> list[Gate]:
    """Every gate this run scores, in the order they are read.

    Flat and unconditional: a gate that vanishes when it cannot run reads as a
    page with fewer gates rather than as a page with an untested claim, and
    the claims below carry UNKNOWN for that instead.
    """
    return [gate_v1_non_vacuity(trace), gate_v2_nvml(trace),
            gate_c1_floor(trace), gate_c2_steady(trace)]


# --------------------------------------------------------------------------
# The measurement.
# --------------------------------------------------------------------------

class _ReasonReader:
    """NVML's slowdown mask, decoded, or "" forever if it cannot be read.

    Initialised ONCE, like `timing.nvml_clock_reader`'s probe, so a reader
    that cannot answer does not spend the whole window raising. Both
    generations of the constant name are accepted: NVML renamed
    `ClocksThrottleReasons` to `ClocksEventReasons`, and a pod's driver may
    ship either.
    """

    def __init__(self, index: int = 0) -> None:
        self._handle = None
        self._get = None
        self._bits: list[tuple[int, str]] = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
            for name in ("nvmlDeviceGetCurrentClocksEventReasons",
                         "nvmlDeviceGetCurrentClocksThrottleReasons"):
                self._get = getattr(pynvml, name, None)
                if self._get is not None:
                    break
            for label in ("GpuIdle", "ApplicationsClocksSetting", "SwPowerCap",
                          "HwSlowdown", "SyncBoost", "SwThermalSlowdown",
                          "HwThermalSlowdown", "HwPowerBrakeSlowdown",
                          "DisplayClockSetting"):
                for stem in ("nvmlClocksEventReason", "nvmlClocksThrottleReason"):
                    bit = getattr(pynvml, stem + label, None)
                    if bit:
                        self._bits.append((int(bit), label))
                        break
        except Exception:                               # noqa: BLE001
            # Broad for `ClockState.sample`'s reasons: a missing pynvml is a
            # ModuleNotFoundError and a restricted container raises NVML's own
            # family. Neither may stop the probe; the mask is a record.
            self._handle = None

    def read(self) -> str:
        if self._handle is None or self._get is None:
            return ""
        try:
            mask = int(self._get(self._handle))
        except Exception:                               # noqa: BLE001
            return ""
        if not mask:
            return "0x0 none"
        named = [label for bit, label in self._bits if mask & bit]
        return f"0x{mask:x} " + ("|".join(named) if named else "unnamed bits")

    def close(self) -> None:
        try:
            if self._handle is not None:
                self._pynvml.nvmlShutdown()
        except Exception:                               # noqa: BLE001
            pass


def sustain(settle_seconds: float, window_seconds: float,
            poll_seconds: float) -> Trace:
    """Load the card and watch it. The only function here that touches a GPU.

    The load is `moe.bench.calibrate._load_compute`, the calibration's own
    compute settle step, imported rather than re-written: a probe that stresses
    the card differently from the arm it gates is a probe that can pass a card
    the arm then fails. The clock is sampled with work STILL IN FLIGHT and
    before the queue is drained, which is the method `clock_under_load`
    documents; sampling between two synchronises measures an idle GPU however
    much work surrounds it.
    """
    import torch

    from moe.bench.calibrate import _load_compute
    from moe.bench.roofline import power_limit_w

    T.require_cuda()
    read = T.nvml_clock_reader()
    reasons = _ReasonReader()
    step = _load_compute()
    max_sm, max_source = T.max_sm_clock_mhz()
    try:
        started = time.monotonic()
        deadline = started + settle_seconds + window_seconds
        samples: list[Sample] = []
        while time.monotonic() < deadline:
            until = min(deadline, time.monotonic() + poll_seconds)
            while time.monotonic() < until:
                step()
                torch.cuda.synchronize()
            # Queue work and sample BEFORE draining it: the point of the whole
            # loop is that the GPU is busy at the instant of the reading.
            for _ in range(4):
                step()
            state = read()
            mask = reasons.read()
            torch.cuda.synchronize()
            samples.append(Sample(t=time.monotonic() - started,
                                  mhz=int(state.sm_clock_mhz),
                                  temp_c=int(state.temp_c),
                                  power_w=float(state.power_w),
                                  source=state.source, reasons=mask))
    finally:
        reasons.close()
    return Trace(samples=tuple(samples), settle_seconds=settle_seconds,
                 window_seconds=window_seconds, max_sm_clock_mhz=max_sm,
                 max_sm_clock_source=max_source, power_limit_w=power_limit_w())


# --------------------------------------------------------------------------
# The self-test: planted worlds whose answer is known.
# --------------------------------------------------------------------------

#: The maximum the planted worlds below are scored against: the SECOND element
#: of `timing.THERMAL_FAULT_OBSERVED_MHZ`, which is the 2026-09-11 card's own
#: maximum, taken from there rather than written here so this file carries no
#: clock of its own. Every world in the self-test is that card's, healthy or
#: not, which is what makes the `floored` world a replay and not a fiction.
PLANTED_MAX_MHZ = T.THERMAL_FAULT_OBSERVED_MHZ[1]


def plant(mhz, *, maximum: float | None = PLANTED_MAX_MHZ, power_w: float = 690.0,
          source: str = T.CLOCK_SOURCE_NVML, settle: float = SETTLE_SECONDS,
          window: float = WINDOW_SECONDS, poll: float = POLL_SECONDS) -> Trace:
    """A Trace built from a list of clocks, for the self-test. No GPU.

    The ramp is planted too, at the card's maximum, because that is what a real
    trace carries and because a scorer that read the whole list would then get
    a different answer from one that honours `settle_seconds`.
    """
    ramp = int(settle / poll)
    readings = [int(maximum or 0)] * ramp + [int(m) for m in mhz]
    return Trace(
        samples=tuple(Sample(t=i * poll, mhz=m, temp_c=60, power_w=power_w,
                             source=source)
                      for i, m in enumerate(readings)),
        settle_seconds=settle, window_seconds=window,
        max_sm_clock_mhz=maximum,
        max_sm_clock_source=T.CLOCK_SOURCE_NVML if maximum else T.CLOCK_SOURCE_NONE,
        power_limit_w=700.0)


#: Every planted world, as `(name, trace, {token: verdict}, why)`. The list is
#: the point: a self-test that plants only successes has never seen its own
#: refusals, and three of the six worlds below are refusals.
def self_test_worlds() -> list[tuple[str, Trace, dict[str, str], str]]:
    steady = [1470] * 60
    return [
        ("healthy", plant(steady),
         {"V1": PASS, "V2": PASS, "C1": PASS, "C2": PASS},
         "the committed H200 calibration's own GEMM clock, held for the window"),
        ("hungry-tile", plant([1275] * 60, power_w=697.4),
         {"V1": PASS, "V2": PASS, "C1": PASS, "C2": PASS},
         "the LOWEST per-cell median in results/published, at 697.4 W of 700. "
         "It must PASS: a card refused here is a session nobody can run"),
        ("floored", plant([345] * 60, power_w=240.0),
         {"V1": PASS, "V2": PASS, "C1": FAIL, "C2": PASS},
         "2026-09-11. C2 PASSES, which is the hole: flat at the floor is "
         "perfectly steady, and DRIFT was all the old gate scored"),
        ("collapsing", plant([1470] * 20 + [1100] * 20 + [420] * 20),
         {"V1": PASS, "V2": PASS, "C1": PASS, "C2": FAIL},
         "still falling when the window closed. The median is above the floor "
         "and the card is not usable, which is what C2 is for"),
        ("excursion", plant([1425, 1410, 405, 825, 1365, 1410, 1425] * 9),
         {"V1": PASS, "V2": PASS, "C1": PASS, "C2": PASS},
         "a healthy drain-and-ramp, from a published cells.csv. Individual "
         "samples go under the floor and the MEDIAN does not, which is why "
         "the verdict is taken on the median"),
        ("truncated", plant([1470] * 4),
         {"V1": FAIL, "V2": PASS, "C1": PASS, "C2": UNKNOWN},
         "a REFUSAL: four samples is not a window. V1 FAILs, which is INVALID, "
         "and C2 cannot take two medians from it"),
        ("forked-sampler", plant(steady, source=T.CLOCK_SOURCE_NVIDIA_SMI),
         {"V1": PASS, "V2": FAIL, "C1": PASS, "C2": PASS},
         "a REFUSAL: every reading came from a forked nvidia-smi and describes "
         "an idle card. INVALID, and the clocks are not quotable however good "
         "they look"),
        ("no-maximum", plant(steady, maximum=None),
         {"V1": PASS, "V2": PASS, "C1": UNKNOWN, "C2": PASS},
         "a REFUSAL: no maximum, so no floor, so the claim was NOT TESTED. "
         "UNKNOWN counts against the gate and never reads as a pass"),
    ]


def self_test() -> int:
    """Plant every world, score it, and name the rows that disagree.

    An estimator that has never been run against a known answer is an
    assertion. One `[PASS]`/`[FAIL]` line per world per gate, so a broken row
    is NAMED rather than merely counted, and the whole thing is ONE VALIDITY
    gate: a self-test that fails has refuted nothing about any card, it has
    declared that nothing this file computes may be quoted, and INVALID (3) is
    the table's word for that, not CLAIM_FAIL.
    """
    print("SELF TEST. Eight planted worlds; three of them are refusals, "
          "because a scorer that has only ever seen clean cards has never "
          "been shown to refuse one.")
    ok = True
    for name, trace, expect, why in self_test_worlds():
        print(f"\n  {name}: {why}")
        got = {g.token: g.verdict for g in gates_for(trace)}
        assert set(got) == set(expect), (sorted(got), sorted(expect))
        for token in sorted(expect):
            good = got[token] == expect[token]
            ok = ok and good
            print(f"    [{'PASS' if good else 'FAIL'}] {token} "
                  f"expected {expect[token]:8s} got {got[token]}")
    gate = Gate(VALIDITY, "0",
                "the scorer recovers the planted verdict in every world, "
                "refusals included",
                PASS if ok else FAIL, "every planted row above",
                "all rows PASS",
                "everything this file computes: a scorer that cannot tell a "
                "floored card from a hungry one cannot gate a session")
    print()
    print(gate.result_line())
    for line in gate.render()[1:]:
        print(line)
    rc = exit_codes.classify([gate.scored()])
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


# --------------------------------------------------------------------------
# Plumbing: where it writes, and what git thinks of that.
# --------------------------------------------------------------------------

def git_check_ignore(path: Path) -> bool | None:
    """Would git silently drop this path. True, False, or None for CANNOT ASK.

    Run rather than reasoned about, and the third return value is the point:
    `git check-ignore` returns 128 for a path outside the work tree, which is
    the POD DEFAULT, and reading that as "tracked" is how a gate came to print
    PASS on a machine where it had not been able to ask.
    """
    try:
        done = subprocess.run(["git", "check-ignore", "-q", str(path)],
                              cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode in (0, 1):
        return done.returncode == 0
    return None


def git_visibility(path: Path) -> str:
    answer = git_check_ignore(path)
    if answer is None:
        return "git could not be asked (rc 128 is a path outside the work tree)"
    return "IGNORED by git" if answer else "tracked"


def results_root() -> Path:
    """`$MOE_RESULTS_DIR`, else the network volume, else the repo.

    The same order every other arm resolves it in, so this lands beside them
    on the volume that outlives the pod.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return Path(__file__).resolve().parents[1] / "results"


def resolve_card(args) -> str:
    """`--card`, else the live device, else `NO_CARD`. The measuring path
    REFUSES under `NO_CARD`: a thermal verdict is about one piece of silicon,
    and both reports are written with `write_text`, which truncates."""
    if getattr(args, "card", None):
        return str(args.card)
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            if name:
                return str(name)
    except Exception:                                   # noqa: BLE001
        pass
    return NO_CARD


def default_run_id(args) -> str:
    """Card FIRST, then every knob that changes what is measured.

    A longer window sees a slower collapse and a shorter one may miss it, and
    the poll interval sets how many samples a median is taken over, so all
    three are in the key. `--out` is not: it re-files a measurement rather
    than changing it.
    """
    return PV.run_id(card=resolve_card(args),
                     **{"1settle": float(args.settle_seconds),
                        "2window": float(args.seconds),
                        "3poll": float(args.poll_seconds)})


def estimated_seconds(args) -> float:
    """Wall seconds this arm costs, from its own parts. Itemised deliberately:
    a single number nobody can decompose is a number nobody can check before
    spending it."""
    return float(args.settle_seconds) + float(args.seconds) + ALLOCATION_SECONDS


#: Seconds charged for allocating the two 8192^2 bf16 buffers and running the
#: first matmul, which is the one part of this arm that is not the window
#: itself. An allowance and it says so: `_load_compute` allocates 256 MiB and
#: cuBLAS picks a kernel on the first call, and neither has been timed here.
ALLOCATION_SECONDS = 10.0


def resolution_line(max_sm_clock_mhz: float | None, card: str = "this card") -> str:
    """The smallest difference this probe can see, and the one it is sized for.

    NOT AN MDE OVER A NOISE SIGMA, and the difference is worth stating. Every
    other arm in this study resolves a TIME against a measured run-to-run
    spread. This one resolves a CLOCK, which NVML reports on a fixed 15 MHz
    grid, so the resolution is a property of the reader rather than of the
    sample size, and quoting a sigma nobody measured would be the prior
    dressed as a bound that B14 found across the arms.
    """
    if not max_sm_clock_mhz:
        return ("RESOLUTION: REFUSED. No maximum SM clock, so neither a floor "
                "nor the distance to it can be stated.")
    floor = T.thermal_floor_mhz(max_sm_clock_mhz)
    steps = (max_sm_clock_mhz - floor) / T.CLOCK_STEP_MHZ
    return (f"RESOLUTION: {T.CLOCK_STEP_MHZ:.0f} MHz, the NVML grid, which is "
            f"{100 * T.CLOCK_STEP_MHZ / max_sm_clock_mhz:.2f}% of {card}'s "
            f"{max_sm_clock_mhz:.0f} MHz maximum. The gate sits {steps:.0f} "
            f"grid steps below that maximum ({floor:.0f} MHz), and the closest "
            "healthy reading in results/published is 1.93x above it. This "
            "probe cannot miss a difference of the size it is looking for.")


def trace_lines(trace: Trace) -> list[str]:
    """The trace, as a reader of `report.txt` sees it."""
    out = ["", "THE WINDOW"]
    out.append(f"  maximum SM clock  {trace.max_sm_clock_mhz:.0f} MHz from "
               f"{trace.max_sm_clock_source}"
               if trace.max_sm_clock_mhz else
               f"  maximum SM clock  UNREADABLE ({trace.max_sm_clock_source})")
    if trace.floor_mhz is not None:
        out.append(f"  thermal floor     {trace.floor_mhz:.0f} MHz "
                   f"= {T.THERMAL_FLOOR_FRACTION:.4f} of it")
    got = trace.scored
    clocks = [s.mhz for s in got if s.mhz > 0]
    if clocks:
        out.append(f"  clock             min {min(clocks)}  median "
                   f"{statistics.median(clocks):.0f}  max {max(clocks)} MHz "
                   f"over {len(clocks)} samples")
        out.append(f"  first third       {trace.head_mhz}   last third "
                   f"{trace.tail_mhz} MHz")
    temps = [s.temp_c for s in got if s.temp_c > 0]
    if temps:
        out.append(f"  temperature       {min(temps)} -> {max(temps)} C "
                   f"(median {statistics.median(temps):.0f})")
    powers = [s.power_w for s in got if s.power_w > 0]
    if powers:
        limit = (f" of a {trace.power_limit_w:.0f} W limit"
                 if trace.power_limit_w else "")
        out.append(f"  board power       median {statistics.median(powers):.0f} W"
                   + limit)
    out.append("  slowdown reasons  "
               + ("; ".join(trace.reason_masks) if trace.reason_masks
                  else "none reported, or the reader could not answer")
               + "   [RECORDED, NOT SCORED]")
    return out


def render(header: list[str], gates: list[Gate], body: list[str]) -> str:
    out = list(header) + body + ["", "=" * 78, "GATES"]
    for gate in gates:
        out += gate.render()
    out.append("")
    if any(g.kind == VALIDITY and g.verdict != PASS for g in gates):
        out.append("READING IT. A VALIDITY gate did not pass. No number on "
                   "this page may be quoted, and the card is neither accepted "
                   "nor refused by it; fix the probe and re-run.")
    else:
        failed = [g.token for g in gates if g.kind == CLAIM and g.verdict != PASS]
        out.append(f"READING IT. Validity holds. Claim gates not passed: "
                   f"{failed or 'none'}.")
        out.append("A failed CLAIM gate is a result, not a broken run: this "
                   "card cannot hold its clock, and no ceiling measured on it "
                   "is its ceiling.")
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the predictions, the paths and the "
                         "cost, then stop. Scores no gate, so it exits "
                         f"{exit_codes.REFUSED} REFUSED; --self-test is the "
                         "off-GPU mode that DOES score")
    ap.add_argument("--self-test", action="store_true",
                    help="score eight planted worlds, three of them refusals, "
                         "and check the scorer recovers each planted verdict. "
                         "Needs no GPU")
    ap.add_argument("--seconds", type=float, default=WINDOW_SECONDS,
                    help="seconds of SCORED window, after the ramp. Four times "
                         "the 2026-09-11 card's observed time to collapse")
    ap.add_argument("--settle-seconds", type=float, default=SETTLE_SECONDS,
                    help="seconds of load discarded as the governor's ramp "
                         "before the window opens. calibrate.settle_clocks' "
                         "own budget; below it a healthy card reads as drift")
    ap.add_argument("--poll-seconds", type=float, default=POLL_SECONDS,
                    help="seconds between NVML readings inside the window")
    ap.add_argument("--card", default=None,
                    help="the card this run is about, as nvidia-smi spells it. "
                         "Defaults to the live device; --dry-run and "
                         f"--self-test touch no GPU and are labelled {NO_CARD!r}. "
                         "The measuring path REFUSES without one")
    ap.add_argument("--out", type=Path, default=None,
                    help="overrides the results root entirely")
    ap.add_argument("--run-id", default="")
    return ap


def _main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    card = resolve_card(args)
    run_id = args.run_id or default_run_id(args)
    out_dir = (args.out or results_root()) / "thermal_acceptance" / run_id
    paths = {"report.txt": out_dir / "report.txt",
             "report.json": out_dir / "report.json"}

    header = [
        f"experiment  thermal_acceptance / {run_id}",
        f"card        {card}",
        f"window      {args.settle_seconds:.0f} s of ramp discarded, then "
        f"{args.seconds:.0f} s scored, polled every {args.poll_seconds:g} s",
        f"load        {INSTRUMENT}",
        f"WRITES TO   {out_dir}",
        "",
        "REGISTERED PREDICTIONS. Printed before anything is measured, and not "
        "rewritten afterwards.",
    ]
    for pred in PREDICTIONS:
        header += [""] + [f"  {line}" for line in pred.render()]
    header += ["", "=" * 78]
    print("\n".join(header))

    if args.dry_run:
        # THE PLAN'S FIGURE IS A WALL CLOCK AND SAYS SO. This arm compiles
        # nothing and times nothing: it holds a load for a stated number of
        # seconds, so the only thing outside the window is the allocation.
        print(f"\nestimated wall time {estimated_seconds(args):.0f} s "
              f"({estimated_seconds(args) / 60:.1f} min: "
              f"{args.settle_seconds:.0f} s ramp + {args.seconds:.0f} s window "
              f"+ {ALLOCATION_SECONDS:.0f} s allocation and first matmul)")
        print(f"samples            ~{args.seconds / args.poll_seconds:.0f} "
              f"scored, floor {SCORED_SAMPLE_FLOOR}")
        # The maximum is read even in a plan when a card is attached, because
        # the floor it implies is the whole threshold and an operator should
        # see it before renting rather than after.
        max_sm, source = T.max_sm_clock_mhz()
        if max_sm:
            print(f"this card           maximum {max_sm:.0f} MHz from {source}, "
                  f"so the floor is {T.thermal_floor_mhz(max_sm):.0f} MHz")
        else:
            print(f"this card           no maximum SM clock here ({source}); "
                  "on the pod it is read from the device and the measuring "
                  "path REFUSES without it")
        for label, path in paths.items():
            print(f"  {label:<14}{path}  {git_visibility(path)}")
        # OFF A GPU there is no maximum to state a resolution against, so the
        # line is stated against the 2026-09-11 card's maximum AND SAYS SO.
        # Printing it as "this card's" on a laptop would be the plan claiming
        # a device reading it never took.
        if max_sm:
            print(f"  {resolution_line(max_sm, card)}")
        else:
            print("  " + resolution_line(T.THERMAL_FAULT_OBSERVED_MHZ[1],
                                         "the 2026-09-11 H200")
                  + " (no card here, so this line is that card's, not one "
                    "this box read)")
        print("\n".join(["", "=" * 78,
                         "REFUSED. Nothing was measured and nothing was "
                         "written.",
                         "  reason: --dry-run was given",
                         "  No gate was scored, so no RESULT line was printed "
                         "and none of the above",
                         "  is a result. --self-test scores the planted worlds "
                         "off GPU; the bare",
                         "  command loads this card and watches it.",
                         "=" * 78]))
        return exit_codes.REFUSED

    if card == NO_CARD:
        print("\nREFUSED. Nothing was measured.")
        print("  This run would MEASURE and no card was named. The run id, the "
              "output directory and")
        print("  the floor itself are all per-card. --self-test scores the "
              "planted worlds without one.")
        return exit_codes.REFUSED

    max_sm, max_source = T.max_sm_clock_mhz()
    if not max_sm:
        # REFUSED AND NOT MEASURED-THEN-UNKNOWN. Without a maximum there is no
        # floor, so the window would be two minutes of load bought to answer
        # nothing. A refusal costs nothing and is decided before any of it is
        # spent, which is what REFUSED means in the shared table.
        print("\nREFUSED. Nothing was measured.")
        print(f"  This card's maximum SM clock could not be read ({max_source}). "
              "The floor is a fraction")
        print("  of it, so there is no threshold to score against and the "
              "window would buy nothing.")
        print("  Install nvidia-ml-py in this interpreter, or check that "
              "`nvidia-smi --query-gpu=clocks.max.sm` answers on this "
              "container.")
        return exit_codes.REFUSED

    print(f"\n[thermal] loading this card for "
          f"{args.settle_seconds + args.seconds:.0f} s and watching the clock")
    print(f"[thermal] maximum {max_sm:.0f} MHz from {max_source}; floor "
          f"{T.thermal_floor_mhz(max_sm):.0f} MHz")
    started = time.time()
    trace = sustain(args.settle_seconds, args.seconds, args.poll_seconds)
    print(f"[thermal] held the load for {time.time() - started:.0f} s")

    gates = gates_for(trace)
    body = trace_lines(trace) + ["", resolution_line(trace.max_sm_clock_mhz, card)]
    text = render(header, gates, body)
    print("\n".join(text.splitlines()[len(header):]))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(text + "\n")
    # THE PROVENANCE BLOCK GOES IN THE JSON AND NOT IN report.txt, on purpose:
    # it carries a UTC timestamp, and report.txt is the artefact two replays
    # are compared byte for byte.
    prov = PV.provenance_block(instrument=INSTRUMENT)
    payload = prov.stamp({
        "run_id": run_id, "card": card,
        "predictions": [asdict(p) for p in PREDICTIONS],
        "gates": [asdict(g) for g in gates],
        "trace": {**{k: v for k, v in asdict(trace).items() if k != "samples"},
                  "samples": [asdict(s) for s in trace.samples],
                  "median_mhz": trace.median_mhz,
                  "head_mhz": trace.head_mhz, "tail_mhz": trace.tail_mhz,
                  "floor_mhz": trace.floor_mhz,
                  "power_fraction": trace.power_fraction},
    })
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2))
    print(f"\nreport   {out_dir / 'report.txt'}   {git_visibility(paths['report.txt'])}")
    print(f"json     {out_dir / 'report.json'}")

    rc = exit_codes.classify(g.scored() for g in gates)
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


def main(argv=None) -> int:
    """AN UNPLANNED CRASH IS ERROR (4), which is the only retryable code.

    Left to propagate, an unexpected exception exits the interpreter ONE, and
    ONE is CLAIM_FAIL, which the shared table defines as a RESULT: it is in
    FINISHED_CODES, so the driver would file a crashed probe as "this card
    cannot hold its clock" and refuse the session over an arm that never
    measured. Wrapped around `_main` rather than installed at the `__main__`
    guard so the contract holds for importers and tests as well as for the
    CLI. `SystemExit` is a `BaseException` and passes through untouched: a
    refusal is not a crash.
    """
    try:
        return _main(argv)
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        print("ERROR: thermal_acceptance crashed before it could reach a "
              "verdict. This is the apparatus failing, not a card failing, so "
              f"it exits {exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: "
              "the traceback above is the thing to fix, and the arm may be "
              "re-run.", file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
