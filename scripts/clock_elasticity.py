#!/usr/bin/env python
"""How much of a measured millisecond is the CLOCK? d log ms / d log f, one kernel.

    python scripts/clock_elasticity.py --dry-run     # the priced plan and the registered bands
    python scripts/clock_elasticity.py --self-test   # the scorer and the estimator, off GPU
    python scripts/clock_elasticity.py --card 'NVIDIA H200'    # the pod run

WHY THIS ARM EXISTS. A 13-agent reading of the 2026-09-10 session concluded that
alpha, as this study defines it, is NOT IDENTIFIED by this apparatus, and one of
the three independent reasons is the clock. The card is power-capped at 689-695 W
in every cell of that session, so the SM clock is an ENDOGENOUS RESPONSE to the
tile and to the tread index rather than an input: 1462-1965 MHz across one grid.
Sweeping the admissible clock elasticity moves pooled EXA alpha_b from 0.974 to
0.897, which is 21x the quoted sd of 0.0037, and the 26.7-sigma monotone rise of
alpha_b with BLOCK_M breaks at elasticity 0.5 and INVERTS at 1.0.

NOTHING IN THE CORPUS MOVES THE CLOCK AT A BYTE-IDENTICAL KERNEL. Every clock
difference on record is confounded with the thing that caused it: a different
tile, a different tread, a different model. So the elasticity is a free
parameter, and a free parameter with that much leverage is the reason no alpha
in the study is quotable. This arm measures it.

WHAT IS MEASURED. One pinned cell -- BLOCK_SIZE_M 32, BLOCK_SIZE_N 64,
BLOCK_SIZE_K 64, GROUP_SIZE_M 16, num_warps 8, num_stages 4, mixtral-8x7b bf16 --
run as a ladder of exactly-full tile stacks, at three or more SUSTAINED DUTY
CYCLES. The tensors are built ONCE and reused in every state, so "the same
kernel over the same bytes" is true by construction and not by assertion, and the
rows carry the tile they ran under so it is checked off the rows as well.

WHY DUTY CYCLE AND NOT nvidia-smi -lgc. Setting a clock needs root and a rented
pod refuses it; `nvidia-smi -lgc` on a RunPod container returns "Insufficient
Permissions". Under a power cap the settled clock is a function of SUSTAINED
BOARD POWER, and sustained board power is delivered GPU time per unit wall time.
So a host-side idle gap after each burst of launches moves the clock at a kernel
whose bytes and instructions never change. The gap is OUTSIDE every measured
interval, and the FIRST call of every burst is discarded because it launches into
a drained queue and carries launch latency the other calls do not.

THE CLOCK IS READ WITH WORK IN FLIGHT, which is why this arm cannot use
`timing.BackgroundClockSampler`: a free-running poller at 10% duty lands in an
idle gap nine times in ten and reports the boost clock of an idle card. The
reader is `timing.nvml_clock_reader`, the same one `scripts/thermal_acceptance.py`
samples with, called once per burst after the last enqueue and BEFORE the
synchronise -- thermal_acceptance's own method, reused rather than rewritten.

WHAT IT CANNOT DO. It cannot make the card SLOWER than its full-duty clock at
this kernel: full duty is already the most sustained pressure a byte-identical
kernel can apply. So the swept range runs from the full-duty clock UP toward the
card's boost ceiling, and it is widest at the shallow treads (the 2026-09-10
corpus has this cell at 1485 MHz at tread 1 against 1815 at tread 8, on a part
whose maximum is 1980). V1 refuses a run whose states did not separate, which is
the honest outcome if the governor does not respond.

WHAT IT WRITES, under `$MOE_RESULTS_DIR` or `/workspace/results` or `<repo>/results`:

    <results>/clock_elasticity/<run-id>/cells.csv     one row per state x tread x repeat
    <results>/clock_elasticity/<run-id>/report.txt    exactly what was printed
    <results>/clock_elasticity/<run-id>/report.json   the fit, the gates, the provenance

NO RULER IS RESOLVED AND NONE IS NEEDED. An elasticity is a ratio of logs: it has
no ridge in it, no bandwidth, no compute peak and no fitted level. The reference
clock is resolved for the LEVEL RECORD only, and LEVEL -- on EITHER side -- never
excludes a row here, exactly as the repository's rule has stood since 2026-09-09.
DRIFT excludes, and so does a host-bound row, because a host-bound interval is an
upper bound and an upper bound that moves with the clock would BE the finding.

EXIT CODES are `moe.bench.exit_codes`. `--dry-run` measured nothing and scored no
gate, so it exits REFUSED (2) and prints no RESULT line. There is deliberately no
gate-softening flag in this file: `classify` over the gates is the exit code in
every scoring mode, so a failed claim is CLAIM_FAIL whether or not anyone
remembered a flag.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import block_m_crossing_sweep as SWEEP  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench import timing as T  # noqa: E402
from moe.spec import DTYPE_BYTES, MODEL_CONFIGS  # noqa: E402

# --------------------------------------------------------------------------
# What ran, named. NOT `timing.TIMING_BASIS`: that basis is a back-to-back
# queue-deep loop with a background clock poller, and this one is neither.
# Naming it TIMING_BASIS would claim an instrument this arm never used, which is
# the confusion `dram_counter_route` keeps four separate instrument strings to
# avoid.
# --------------------------------------------------------------------------
INSTRUMENT = (
    "scripts/clock_elasticity.py: bursts of N back-to-back fused_experts calls "
    "under one pinned tile, each call L2-flushed and bracketed by its own "
    "pre-primed event pair, the FIRST call of every burst discarded as a "
    "drained-queue launch, a host-side idle gap after each burst setting the "
    "sustained duty cycle, and the SM clock, board power and memory clock read "
    "through NVML with the burst still in flight; NOT timing.TIMING_BASIS")

#: The card label a run that measures NOTHING carries. `provenance.run_id`
#: refuses an id without a card, and --dry-run and --self-test have none.
NO_CARD = "no-card-nothing-measured"

# --------------------------------------------------------------------------
# The pinned cell. ONE tile for the whole arm, on purpose: the independent
# variable is the clock, and a second thing that moves is a second explanation.
# --------------------------------------------------------------------------

#: BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps and num_stages are `SWEEP.FIXED`'s,
#: taken from there rather than retyped so this arm's tile and the ladder arms'
#: tile cannot drift apart. GROUP_SIZE_M is 16 and NOT `FIXED`'s 1: 16 is the
#: swizzle the bn_decomposition arm ran and the swizzle every published alpha
#: this arm exists to qualify was fitted at, and `project_moe_kernels_alpha_
#: surface` records that alpha is a swizzle x footprint surface, so a G=1
#: elasticity would qualify an alpha nobody quoted.
PINNED = dict(SWEEP.FIXED, BLOCK_SIZE_M=32, GROUP_SIZE_M=16)

DEFAULT_MODEL = "mixtral-8x7b"
DEFAULT_DTYPE = "bf16"

# --------------------------------------------------------------------------
# The design. Every constant here is an experimental-design DECISION, printed
# on the plan page before the run and listed in the arm's report so it can be
# overruled; none of them is a calibration.
# --------------------------------------------------------------------------

#: Sustained GPU-busy fractions, one state each. Four rather than the three the
#: design needs, so a state that fails to separate still leaves a fit.
#:
#: 1.00 IS THE ANCHOR AND IS NOT QUITE THE LADDER ARMS' CADENCE, which is worth
#: saying plainly. It is the highest sustained pressure this instrument can
#: apply -- no host sleep at all -- and the corpus's ladders run the same
#: kernel back to back, so the anchor sits at or very near their operating
#: point. It is not byte-identical to them: `timing._timed_trials`
#: synchronises ONCE PER TRIAL and this loop synchronises once per burst, so
#: the anchor drains the queue more often. That difference is deliberate and it
#: is what makes the four states comparable: EVERY state, the anchor included,
#: launches each burst into a drained queue and discards that burst's lead
#: call, so the only thing that differs between states is the length of the
#: host sleep.
DUTY_LEVELS = (1.00, 0.50, 0.25, 0.10)

#: Treads on the ladder, exactly-full tile stacks `r = n * BLOCK_M`.
DEFAULT_TREADS = 8

#: Independent passes over the whole design. The interval on the elasticity is a
#: bootstrap over these, so this is the sample size that sets the resolution;
#: `mde_lines` prints the arithmetic that chose it.
DEFAULT_REPEATS = 13

#: Target GPU milliseconds inside one burst. Sets how many calls a burst holds,
#: hence how much of the burst the discarded lead call costs, and how finely the
#: within-burst steadiness gate (V5) can look. 40 ms puts at least 8 calls in a
#: burst at every tread of this ladder and keeps the lead call under 12% of them.
DEFAULT_BURST_MS = 40.0

#: Calls per burst, floor and ceiling. The floor is what V5 needs: it reads the
#: first and last QUARTER of a burst's kept calls, and a quarter of fewer than
#: eight calls is one call, which is a sample and not a median.
MIN_CALLS_PER_BURST = 8
MAX_CALLS_PER_BURST = 512

#: Kernel milliseconds per trial and trials per cell, the shared instrument's own
#: `--cell-budget-ms` and `--trials` defaults, kept identical so a row here and a
#: row from the ladder arms hold the same number of samples.
DEFAULT_TARGET_MS = 200.0
DEFAULT_TRIALS = 3

#: Milliseconds of delivered GPU load before a cell is timed, run AT THE STATE'S
#: OWN DUTY CYCLE and followed by the same two-consecutive-reads settle rule
#: `timing.warm_until` applies. 200 ms at duty 0.10 is 2 s of wall, which is the
#: per-tread part of holding an operating point.
DEFAULT_WARM_MS = 200.0

#: Seconds of the state's cadence run before the state's first cell, to move the
#: board's power average onto the new duty. The per-tread warm above handles the
#: fine adjustment; this handles the coarse one, and it is SECONDS rather than
#: milliseconds because a thermal transition is not a power-average transition.
DEFAULT_SETTLE_SECONDS = 10.0

#: Bootstrap draws for the interval. Re-analysis of one set of cells, so it is
#: OUT of the run id for the same reason --ridge and --alpha are.
DEFAULT_DRAWS = 2000

#: Depth a state must reach among KEPT rows before the fit will read it.
MIN_TREADS_PER_STATE = 4
MIN_REPEATS_PER_STATE = 3
#: States the fit needs. Two states give a slope with zero degrees of freedom and
#: no way to see curvature; the brief's own floor is three.
MIN_STATES = 3

#: Share of rows that may be dropped for DRIFT or for being host-bound before the
#: kept set stops describing the run that was paid for. One in five: above that,
#: what is left is a subsample chosen by the card's own behaviour.
EXCLUSION_CEILING = 0.20

# --------------------------------------------------------------------------
# THE REGISTERED BANDS. Written here, printed by --dry-run before the run, and
# NOT chosen after the fact. They come from the session-3 analysis: below 0.25
# the raw readings stand and pooled EXA alpha_b is near 0.97 with the BLOCK_M
# ladder monotone; above 0.40 the ladder is non-monotone, the pooled value is
# near 0.93, and C3's DIRECTION is retracted while its inequality survives.
#
# THE MIDDLE BAND IS THE ONE THE ANALYSIS DID NOT GIVE, and it is registered
# anyway. Two consequences and a gap between them is three outcomes, and an arm
# that landed in the gap with only two registered readings would have to invent
# the third after seeing its own number.
# --------------------------------------------------------------------------
BAND_LOW = 0.25
BAND_HIGH = 0.40

BANDS = (
    ("RAW-STANDS", None, BAND_LOW,
     "the raw readings stand: pooled EXA alpha_b is near 0.974 and the "
     "26.7-sigma monotone rise of alpha_b with BLOCK_M survives. The clock is "
     "not what is wrong with alpha; FORM and PHYSICS still are."),
    ("UNREGISTERED-GAP", BAND_LOW, BAND_HIGH,
     "NEITHER registered consequence is licensed. The analysis that produced "
     "the two readings below and above swept the elasticity past this band "
     "without stopping in it, so there is no pooled alpha_b to quote here and "
     "the honest report is the interval and the refusal to read it further."),
    ("CLOCK-CARRIES", BAND_HIGH, None,
     "the BLOCK_M ladder is NON-MONOTONE, pooled EXA alpha_b falls to about "
     "0.93, and C3's DIRECTION is RETRACTED while C3's inequality survives. "
     "Every alpha in the study is then a blend of traffic and clock and none "
     "of them may be quoted as traffic."),
)

#: The half-width the design is sized to reach: half the width of the
#: unregistered gap. An interval wider than the gap cannot be placed inside it,
#: so a design that cannot reach this cannot distinguish the three worlds.
RESOLUTION_TARGET = (BAND_HIGH - BAND_LOW) / 2.0

# --------------------------------------------------------------------------
# The one corpus fact this file carries, and it PRICES the run rather than
# scoring it.
# --------------------------------------------------------------------------

#: Per-tread median milliseconds of the cell this arm pins, measured by the
#: 2026-09-10 session's bn_decomposition arm at BLOCK_M=32, BLOCK_N=64,
#: GROUP_SIZE_M=16, mixtral-8x7b bf16, 17 repeats per tread, from
#: results/published/2026-09-10-nvidia_h200-gaps-session/results/bn_decomposition/
#: nvidia_h200-bm32_64_128-budget400.0-dtypebf16-flushtrue-g16-iters50-k64-
#: modelmixtral_8x7b-n32_64-b59b409f/cells.csv
#:
#: USED ONLY TO PRICE THE RUN. It sizes the burst, the iteration count and the
#: wall clock on the plan page so an operator buys pod time against a figure
#: derived the way the runner derives it. No gate reads it, no fit reads it, and
#: `tests/test_clock_elasticity.py` recomputes it from that committed file so it
#: cannot drift away from what was actually measured.
CORPUS_LADDER_MS = (0.7738, 1.3252, 1.8911, 2.4154, 3.0127, 3.5662, 4.1618,
                    4.6153)

#: Median across-repeat relative spread of those same 17-repeat treads, which is
#: the sd of log ms one cell of this design carries. Same file, same provenance,
#: same "prices only" rule: it is the input to the MDE arithmetic that CHOOSES
#: the clock-separation threshold, and the threshold is COMPUTED from it rather
#: than typed, so a revision of this number moves the gate with it.
CORPUS_REPEAT_SPREAD = 0.01071

#: The median's efficiency penalty against the mean for a normal sample,
#: sqrt(pi/2). The per-cell figure is a MEDIAN across repeats, so its standard
#: error is this much worse than the mean's and the MDE says so rather than
#: pretending the collapse was free.
MEDIAN_SE_PENALTY = math.sqrt(math.pi / 2.0)

#: Seconds charged for the weight build, the activations and the first compile.
#: An ALLOWANCE and it says so: mixtral-8x7b bf16 is 2.82 GB of expert weights
#: built once and reused by every tread, and Triton picks a kernel on the first
#: call under the pin; neither has been timed here.
ALLOCATION_SECONDS = 60.0


# --------------------------------------------------------------------------
# Registered predictions. Printed before anything is measured, never rewritten.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    """One registered prediction: what it says, and what a FAIL would mean."""

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
        1, "the duty cycle moves the clock at a byte-identical kernel",
        "every tread's under-load SM clock spans at least the ratio the design "
        "arithmetic below requires, across the duty states, with the same "
        "tensors, the same pinned tile and the same tread count in every state",
        "V1 FAILS and the arm is INVALID. Three states at one clock measure "
        "nothing, and the honest reading is that this container's governor does "
        "not respond to duty at this kernel -- not that the elasticity is zero."),
    Prediction(
        2, "the measured per-call time is BELOW 0.25 in elasticity: most of a "
           "millisecond at this cell is traffic, not issue rate",
        f"the 95% interval on d log ms / d log f lies wholly below {BAND_LOW}, "
        "which is the band in which the study's raw readings stand and pooled "
        "EXA alpha_b is near 0.974 with the BLOCK_M ladder monotone",
        f"a FAIL is a RESULT and not a retry. Above {BAND_HIGH} the BLOCK_M "
        "ladder is non-monotone, pooled alpha_b falls to about 0.93 and C3's "
        "DIRECTION is retracted; between the two the analysis registered no "
        "reading at all and the page says so instead of inventing one."),
    Prediction(
        3, "and the design resolves which of the three registered worlds it is in",
        f"the 95% interval lies wholly inside ONE of the three bands, which "
        f"needs a half-width under {RESOLUTION_TARGET:.3f}",
        "C1 FAILS. The apparatus worked and the answer is 'not resolved': the "
        "states separated too little or the repeats were too few, and NO "
        "registered consequence may be quoted off this page."),
)


# --------------------------------------------------------------------------
# Gates. A number against a threshold, PASS or FAIL, and what a FAIL costs.
# --------------------------------------------------------------------------

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
VALIDITY = "VALIDITY"
CLAIM = "CLAIM"


@dataclass
class Gate:
    kind: str
    number: str
    claim: str
    verdict: str
    measured: str
    threshold: str
    #: What a non-PASS here voids, printed only when it is not a PASS.
    invalidates: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def token(self) -> str:
        """The gate's identifier as ONE whitespace-free token, for `RESULT:`.

        `V1` and `C1` rather than `1` twice: a file with both a VALIDITY 1 and a
        CLAIM 1 would otherwise print two gates the driver cannot tell apart. A
        number that already begins with a letter is its own token, which is what
        keeps the self-test's S gates from printing as `VS1` and, more to the
        point, from colliding with a measuring run's `V1` in one log.
        """
        return (self.number if self.number[:1].isalpha()
                else f"{self.kind[0]}{self.number}")

    def result_line(self) -> str:
        return exit_codes.result_line(
            exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
            self.token, self.verdict,
            f"{self.claim} | measured {self.measured} | gate {self.threshold}")

    def scored(self) -> tuple[str, str, str]:
        return (exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
                self.token, self.verdict)

    def render(self) -> list[str]:
        out = [self.result_line(),
               f"{self.kind} {self.number}  {self.verdict:8s} {self.claim}",
               f"{'':>11}measured {self.measured}",
               f"{'':>11}gate     {self.threshold}"]
        if self.verdict != PASS and self.invalidates:
            out.append(f"{'':>11}a non-PASS here voids {self.invalidates}")
        out += [f"{'':>11}{text}" for text in self.lines]
        return out


# --------------------------------------------------------------------------
# One measured cell. Pure data: this is what the CSV holds and what every
# analysis below reads, so the self-test plants these and nothing else.
# --------------------------------------------------------------------------

@dataclass
class Row:
    """One (duty state, tread, repeat) measurement.

    The tile fields are on EVERY row rather than in a header: V2 reads the tile
    off the rows, and a tile recorded once per file cannot witness that all four
    states ran the same one.
    """

    duty_requested: float
    duty_achieved: float
    state_index: int
    repeat: int
    order_index: int
    model: str
    dtype: str
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int
    tiles: int
    rows_per_expert: int
    tokens: int
    calls_per_burst: int
    bursts: int
    trials: int
    ms_p50: float
    ms_min: float
    ms_stdev: float
    samples: int
    burst_ms: float
    gap_ms: float
    head_ms: float | None
    tail_ms: float | None
    within_burst_ok: bool | None
    sm_clock_load_mhz: float | None
    sm_clock_start_mhz: float | None
    sm_clock_end_mhz: float | None
    clock_samples_mhz: str
    clock_samples: int
    clock_source: str
    #: BOTH halves of the LEVEL verdict travel together. A failed LEVEL without
    #: its side reads as LOW to any consumer that assumes one, which is how the
    #: repository came to drop the boosted treads of a whole session.
    clock_level_ok: bool | None
    clock_level_side: str
    clock_drift_ok: bool | None
    clock_drift_direction: str
    reference_clock_mhz: float | None
    power_w: float | None
    mem_clock_mhz: float | None
    host_bound: bool | None
    host_enqueue_ms: float | None
    host_backlog_iters: float | None
    warmup_ms: float
    l2_flush: bool
    status: str = "ok"
    detail: str = ""
    instrument: str = INSTRUMENT


# --------------------------------------------------------------------------
# The arithmetic. Pure: no torch, no GPU, no clock. Everything below is what
# `--self-test` drives on planted rows.
# --------------------------------------------------------------------------

DROP_DRIFT = "drift"
DROP_HOST = "host_bound"
DROP_STATUS = "status"
DROP_NO_CLOCK = "no_clock"
DROP_NO_TIME = "no_time"


def exclusion(row: Row) -> str:
    """Why this row is not in the fit, or "" when it is.

    DRIFT EXCLUDES AND LEVEL DOES NOT, either side of it. That is the
    repository's rule since 2026-09-09 and it is exactly right here: a drifting
    cell was not measured at one operating point, so its median clock is a blend
    of two and the pair (ms, f) it contributes is a pair of averages over
    different worlds. A LEVEL failure means this tile sits away from the
    calibration GEMM's operating point under the same power cap, which is a
    RECORD about the tile -- and at 10% duty a LEVEL HIGH row is the whole point
    of the experiment, so a level filter here would drop precisely the states
    that carry the signal. `clock_level_side` is recorded and counted; it is
    never a reason.

    HOST-BOUND EXCLUDES, and that is this arm's own addition to the rule. A
    host-bound interval holds host enqueue time, so it bounds the kernel from
    above rather than measuring it, and a bound that moves with the clock is
    indistinguishable from the elasticity being measured.
    """
    if row.status != "ok":
        return DROP_STATUS
    if row.ms_p50 <= 0:
        return DROP_NO_TIME
    if row.clock_drift_ok is False:
        return DROP_DRIFT
    if row.host_bound:
        return DROP_HOST
    if not row.sm_clock_load_mhz or row.sm_clock_load_mhz <= 0:
        return DROP_NO_CLOCK
    return ""


def kept_rows(rows) -> list[Row]:
    return [r for r in rows if exclusion(r) == ""]


def level_counts(rows) -> dict[str, int]:
    """How many kept rows sat HIGH, LOW and level. A RECORD, printed, not scored."""
    out = {T.LEVEL_HIGH: 0, T.LEVEL_LOW: 0, "level": 0, "undetermined": 0}
    for r in rows:
        if r.clock_level_ok is None:
            out["undetermined"] += 1
        elif r.clock_level_ok:
            out["level"] += 1
        elif r.clock_level_side in (T.LEVEL_HIGH, T.LEVEL_LOW):
            out[r.clock_level_side] += 1
        else:
            out["undetermined"] += 1
    return out


def _duty_key(value: float) -> str:
    """A duty as a stable dictionary key. Floats read back off a CSV do not
    compare equal to the literals that wrote them often enough to rely on."""
    return f"{float(value):.4f}"


def collapse(rows, repeats=None) -> dict[tuple[int, str], tuple[float, float, int]]:
    """Per (tread, duty state) median ms and median under-load clock.

    `repeats` is a MULTISET of repeat indices. Passed the run's own repeats it is
    the point estimate; passed a draw with replacement it is one bootstrap
    replicate, and everything downstream -- the per-tread slopes, the pooled
    slope, the band the interval falls in -- is recomputed on that draw. That is
    `bn_decomposition.collapse`'s contract, in this arm's shape.
    """
    by: dict[tuple[int, str], dict[int, list[tuple[float, float]]]] = {}
    for r in kept_rows(rows):
        cell = by.setdefault((r.tiles, _duty_key(r.duty_requested)), {})
        cell.setdefault(r.repeat, []).append((r.ms_p50, float(r.sm_clock_load_mhz)))
    out = {}
    for key, per_repeat in by.items():
        want = sorted(per_repeat) if repeats is None else list(repeats)
        drawn = [pair for rep in want for pair in per_repeat.get(rep, [])]
        if not drawn:
            continue
        out[key] = (statistics.median(p[0] for p in drawn),
                    statistics.median(p[1] for p in drawn),
                    len(drawn))
    return out


#: THE SIGN CONVENTION, and it is a constant so it is applied ONCE.
#:
#: `d log ms / d log f` is NEGATIVE: a faster clock is a shorter call, and a
#: kernel whose time is entirely issue-rate-limited has slope -1. The study's
#: registered bands are POSITIVE numbers (0.25, 0.40) and the session-3 analysis
#: sweeps "the admissible clock elasticity" from 0 upward, so the quantity those
#: bands are about is the MAGNITUDE:
#:
#:     eta = - d log ms / d log f,   0 = pure traffic,  1 = pure issue rate
#:
#: Both are on the page and in report.json, because a conclusion that flips with
#: a sign is exactly the kind this study has already had to retract once. Every
#: gate reads `eta`; `Elasticity.slope` carries the signed regression slope.
ETA_SIGN = -1.0


def within_tread_slope(cells) -> tuple[float | None, dict[int, float], float, int]:
    """Pooled SIGNED d log ms / d log f AT FIXED TREAD, and the per-tread slopes.

    Signed: negative for a kernel whose time falls as the clock rises. `fit`
    turns it into `eta` through `ETA_SIGN`, in one place.

    A tread is one kernel over one set of bytes, so centring log ms and log f
    within a tread removes everything that differs BETWEEN treads -- the work,
    the footprint, the tile count -- and leaves only what the duty state moved.
    That is what "at a fixed kernel" means, and it is why this, and not a
    regression over the pooled cloud, is the estimator.

    Returns (pooled slope, {tread: its own slope}, the pooled Sxx, the number of
    (tread, state) cells that entered). `None` when no tread has two states.
    """
    by_tread: dict[int, list[tuple[float, float]]] = {}
    for (tread, _duty), (ms, mhz, _n) in cells.items():
        if ms > 0 and mhz > 0:
            by_tread.setdefault(tread, []).append((math.log(mhz), math.log(ms)))
    sxy = sxx = 0.0
    used = 0
    per_tread: dict[int, float] = {}
    for tread, points in sorted(by_tread.items()):
        if len(points) < 2:
            continue
        xbar = statistics.fmean(p[0] for p in points)
        ybar = statistics.fmean(p[1] for p in points)
        t_xx = sum((x - xbar) ** 2 for x, _ in points)
        t_xy = sum((x - xbar) * (y - ybar) for x, y in points)
        if t_xx <= 0:
            continue
        per_tread[tread] = t_xy / t_xx
        sxy += t_xy
        sxx += t_xx
        used += len(points)
    if sxx <= 0:
        return None, per_tread, 0.0, used
    return sxy / sxx, per_tread, sxx, used


def ladder_slope_elasticity(cells) -> tuple[float | None, int]:
    """THE COMPANION READING: how the per-M-tile cost itself responds to the clock.

    Fit `ms = a + b n` over the treads of each state, then regress log b on the
    state's mean log clock. It is a COMPANION and not the claim, for a stated
    reason: the clock is not constant across the treads of one state (the
    2026-09-10 corpus has this cell at 1485 MHz at tread 1 and 1815 at tread 8),
    so the "clock of a slope" is a summary and the elasticity of a summary is not
    the elasticity of a kernel. The claim is the fixed-tread quantity above.
    """
    by_state: dict[str, list[tuple[int, float, float]]] = {}
    for (tread, duty), (ms, mhz, _n) in cells.items():
        if ms > 0 and mhz > 0:
            by_state.setdefault(duty, []).append((tread, ms, mhz))
    points = []
    for _duty, rows in by_state.items():
        if len(rows) < 2:
            continue
        nbar = statistics.fmean(r[0] for r in rows)
        mbar = statistics.fmean(r[1] for r in rows)
        nxx = sum((r[0] - nbar) ** 2 for r in rows)
        if nxx <= 0:
            continue
        slope = sum((r[0] - nbar) * (r[1] - mbar) for r in rows) / nxx
        if slope <= 0:
            continue
        points.append((statistics.fmean(math.log(r[2]) for r in rows),
                       math.log(slope)))
    if len(points) < 2:
        return None, len(points)
    xbar = statistics.fmean(p[0] for p in points)
    ybar = statistics.fmean(p[1] for p in points)
    xx = sum((x - xbar) ** 2 for x, _ in points)
    if xx <= 0:
        return None, len(points)
    return sum((x - xbar) * (y - ybar) for x, y in points) / xx, len(points)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass(frozen=True)
class Elasticity:
    """The fit, its interval, and everything a reader needs to check it.

    `value` is ETA: the positive magnitude the registered bands are about,
    `-d log ms / d log f`. `slope` is the signed regression slope the estimator
    actually computed. Both are carried so a reader never has to infer which
    convention a number is in.
    """

    value: float | None
    slope: float | None
    lo: float | None
    hi: float | None
    per_tread: dict[int, float]
    sxx: float
    cells: int
    states: int
    treads: int
    repeats: int
    draws: int
    resampled: int
    ladder_slope: float | None
    ladder_states: int

    @property
    def half_width(self) -> float | None:
        if self.lo is None or self.hi is None:
            return None
        return (self.hi - self.lo) / 2.0


def fit(rows, *, draws: int = DEFAULT_DRAWS, seed: int = 0) -> Elasticity:
    """The point estimate and a bootstrap interval over REPEATS.

    The resample is over whole repeats, not over rows: a repeat is one pass
    through the whole design at one point in the pod's thermal history, and rows
    inside a repeat share whatever the card was doing. Resampling rows would
    treat 13 correlated passes as hundreds of independent draws and report an
    interval a factor of several too narrow, which is precisely the
    understatement the session-3 reading found in the published sd of 0.0037.
    """
    keep = kept_rows(rows)
    repeats = sorted({r.repeat for r in keep})
    cells = collapse(keep)
    slope, per_tread, sxx, used = within_tread_slope(cells)
    ladder, ladder_states = ladder_slope_elasticity(cells)
    states = len({d for _, d in cells})
    treads = len({t for t, _ in cells})
    lo = hi = None
    drawn = 0
    if slope is not None and draws > 0 and len(repeats) >= 2:
        rng = random.Random(seed)
        values = []
        for _ in range(draws):
            sample = [rng.choice(repeats) for _ in repeats]
            got, _pt, _xx, _n = within_tread_slope(collapse(keep, sample))
            if got is not None:
                values.append(ETA_SIGN * got)
        if values:
            lo = _percentile(values, 0.025)
            hi = _percentile(values, 0.975)
            drawn = len(values)
    return Elasticity(value=None if slope is None else ETA_SIGN * slope,
                      slope=slope, lo=lo, hi=hi,
                      per_tread={t: ETA_SIGN * v for t, v in per_tread.items()},
                      sxx=sxx, cells=used, states=states, treads=treads,
                      repeats=len(repeats), draws=draws, resampled=drawn,
                      ladder_slope=None if ladder is None else ETA_SIGN * ladder,
                      ladder_states=ladder_states)


def band_of(lo: float | None, hi: float | None) -> tuple[str, str] | None:
    """The registered band that WHOLLY contains [lo, hi], or None.

    Wholly, and that is the whole content of C1: an interval that crosses a
    boundary is consistent with two worlds whose consequences contradict each
    other, and picking the nearer one is how a pre-registration becomes a
    post-registration.
    """
    if lo is None or hi is None:
        return None
    for name, edge_lo, edge_hi, consequence in BANDS:
        if (edge_lo is None or lo >= edge_lo) and (edge_hi is None or hi <= edge_hi):
            return name, consequence
    return None


def clock_ratio_by_tread(cells) -> dict[int, float]:
    """Max/min under-load clock across the states, tread by tread."""
    by: dict[int, list[float]] = {}
    for (tread, _duty), (_ms, mhz, _n) in cells.items():
        if mhz > 0:
            by.setdefault(tread, []).append(mhz)
    return {t: max(v) / min(v) for t, v in by.items() if len(v) >= 2 and min(v) > 0}


def required_clock_ratio(*, repeats: int, treads: int, states: int,
                         spread: float = CORPUS_REPEAT_SPREAD,
                         target: float = RESOLUTION_TARGET) -> float:
    """The clock separation this design needs, COMPUTED and never typed.

    The arithmetic, which `mde_lines` prints in full:

        sigma_cell = MEDIAN_SE_PENALTY * spread / sqrt(repeats)
        se(eps)    = sigma_cell / (sqrt(treads * states) * sd of log f)
        half-width = 2 se(eps) <= target

    which needs `sd of log f >= 2 sigma_cell / (sqrt(treads*states) * target)`.
    For `states` points spread evenly in log f over a range `L`, the sd is
    `L * sqrt((states+1) / (12 (states-1)))`, so the required max/min ratio is
    `exp(L)`. Nothing here is a clock: it is a RATIO of two clocks on one card,
    which is why the same number governs an H200 and an A100 without being told
    which is attached.
    """
    if repeats < 1 or treads < 1 or states < 2 or target <= 0 or spread <= 0:
        return float("inf")
    sigma_cell = MEDIAN_SE_PENALTY * spread / math.sqrt(repeats)
    sd_needed = 2.0 * sigma_cell / (math.sqrt(treads * states) * target)
    shape = math.sqrt((states + 1) / (12.0 * (states - 1)))
    return math.exp(sd_needed / shape)


def registered_clock_ratio(args) -> tuple[float, str]:
    """The V1 threshold and where it came from. `--min-clock-ratio` overrides."""
    if getattr(args, "min_clock_ratio", None):
        return float(args.min_clock_ratio), "--min-clock-ratio, the operator's assertion"
    need = required_clock_ratio(repeats=args.repeats, treads=args.treads,
                                states=len(args.duty))
    rounded = math.ceil(need * 100.0) / 100.0
    return rounded, (f"the design's own requirement {need:.4f}, rounded up to "
                     f"the next hundredth, from {args.repeats} repeats x "
                     f"{args.treads} treads x {len(args.duty)} states against a "
                     f"{CORPUS_REPEAT_SPREAD:.5f} across-repeat spread")


# --------------------------------------------------------------------------
# The gates.
# --------------------------------------------------------------------------

def gate_v0_non_vacuity(rows, keep) -> Gate:
    """Did anything get measured at all.

    FIRST, for the reason every model file in this repository puts one first:
    every gate below can be satisfied by having no input. An empty run separates
    no states, contradicts no tile and drifts nowhere.
    """
    ok = bool(keep)
    drops = {}
    for r in rows:
        why = exclusion(r)
        if why:
            drops[why] = drops.get(why, 0) + 1
    detail = ", ".join(f"{k} {v}" for k, v in sorted(drops.items())) or "none"
    return Gate(
        VALIDITY, "0", "the run produced rows the fit can read",
        PASS if ok else FAIL,
        f"{len(keep)} kept of {len(rows)} rows (dropped: {detail})",
        ">= 1 kept row",
        "every gate below, which would otherwise report a clean run from a "
        "directory nobody wrote to")


def gate_v1_separation(cells, threshold: float, source: str) -> Gate:
    """Did the states actually SEPARATE in clock.

    THE GATE THE BRIEF NAMES: three states at one clock measure nothing. It is
    per tread because the regression is per tread: a tread whose clock never
    moved contributes zero to Sxx and the pooled slope simply ignores it, so an
    arm that let one through would quietly fit a design smaller than the one it
    booked. The threshold is COMPUTED by `required_clock_ratio` from this run's
    own repeats, treads and states, so a shallower run is held to a wider
    separation rather than to the same number.
    """
    ratios = clock_ratio_by_tread(cells)
    if not ratios:
        return Gate(VALIDITY, "1", "the duty states separated in clock at every tread",
                    FAIL, "no tread has two states with a usable clock",
                    f"every tread spans >= {threshold:.3f}x",
                    "the elasticity: with no clock separation there is no slope")
    worst_tread = min(ratios, key=lambda t: ratios[t])
    worst = ratios[worst_tread]
    ok = worst >= threshold
    return Gate(
        VALIDITY, "1", "the duty states separated in clock at every tread",
        PASS if ok else FAIL,
        f"narrowest tread {worst_tread} spans {worst:.4f}x "
        f"(widest {max(ratios.values()):.4f}x over {len(ratios)} treads)",
        f">= {threshold:.3f}x, from {source}",
        "the elasticity and both claims: a slope fitted across states that sat "
        "at one clock is a slope over nothing",
        ["per tread: " + ", ".join(f"n{t}={v:.4f}x"
                                    for t, v in sorted(ratios.items()))])


def gate_v2_one_kernel(keep) -> Gate:
    """Was it the SAME kernel in every state, read off the rows.

    Read off the rows and not asserted, which is the brief's own instruction.
    The live path builds the tensors once and pins one tile, so this can only
    fail if the pin did not reach the kernel or two runs were resumed into one
    directory -- and both of those are exactly the failures a run id and a
    docstring cannot catch. `calls_per_burst` is in the comparison because it is
    the loop shape: the same tile launched in bursts of 52 and of 9 carries a
    different share of one drained-queue lead call, and the lead call is
    discarded per burst rather than per run.
    """
    tiles = {(r.model, r.dtype, r.block_m, r.block_n, r.block_k, r.group_m,
              r.num_warps, r.num_stages) for r in keep}
    per_tread: dict[int, set[int]] = {}
    for r in keep:
        per_tread.setdefault(r.tiles, set()).add(r.calls_per_burst)
    mixed = sorted(t for t, v in per_tread.items() if len(v) > 1)
    ok = len(tiles) == 1 and not mixed
    return Gate(
        VALIDITY, "2", "one tile, one model, one dtype and one burst shape per tread",
        PASS if ok else FAIL,
        f"{len(tiles)} distinct tile tuples over {len(keep)} kept rows; "
        + (f"treads with more than one burst shape: {mixed}" if mixed
           else "one burst shape per tread"),
        "exactly 1 tile tuple and 1 burst shape per tread",
        "the whole arm: a slope across states that ran different kernels is a "
        "comparison of kernels, not of clocks",
        [f"tile: {sorted(tiles)[0]}"] if tiles else [])


def gate_v3_depth(keep, duties, threshold_treads: int, threshold_repeats: int,
                  min_states: int) -> Gate:
    """Enough treads and enough repeats, in EVERY state.

    Per state rather than in total: a design that measured one state deeply and
    the others once has the same row count and no second point to take a slope
    between.
    """
    per_state: dict[str, tuple[set[int], set[int]]] = {}
    for r in keep:
        treads, reps = per_state.setdefault(_duty_key(r.duty_requested), (set(), set()))
        treads.add(r.tiles)
        reps.add(r.repeat)
    good = {d: v for d, v in per_state.items()
            if len(v[0]) >= threshold_treads and len(v[1]) >= threshold_repeats}
    ok = len(good) >= min_states
    detail = "; ".join(f"duty {d}: {len(v[0])} treads x {len(v[1])} repeats"
                       for d, v in sorted(per_state.items())) or "no state"
    return Gate(
        VALIDITY, "3", "every state the fit reads is deep enough to be a state",
        PASS if ok else FAIL,
        f"{len(good)} of {len(duties)} planned states are deep enough ({detail})",
        f">= {min_states} states with >= {threshold_treads} treads and "
        f">= {threshold_repeats} repeats each",
        "the interval: a bootstrap over two repeats has two distinct draws")


def gate_v4_exclusions(rows, keep, ceiling: float) -> Gate:
    """Were the exclusions a trim or the measurement.

    DRIFT and host-bound are the two reasons `exclusion` gives, and above a
    ceiling what is left is not the run that was paid for but a subsample the
    card chose. LEVEL is NOT in this count on either side, and the line below
    prints the LEVEL sides so a reader can see that they were kept.
    """
    total = len(rows)
    drops = {DROP_DRIFT: 0, DROP_HOST: 0}
    other = 0
    for r in rows:
        why = exclusion(r)
        if why in drops:
            drops[why] += 1
        elif why:
            other += 1
    excluded = drops[DROP_DRIFT] + drops[DROP_HOST] + other
    share = excluded / total if total else 1.0
    ok = total > 0 and share <= ceiling
    sides = level_counts(keep)
    return Gate(
        VALIDITY, "4", "the excluded rows are a trim and not the measurement",
        PASS if ok else FAIL,
        f"{excluded} of {total} rows excluded ({100 * share:.1f}%): "
        f"drift {drops[DROP_DRIFT]}, host-bound {drops[DROP_HOST]}, other {other}",
        f"<= {100 * ceiling:.0f}% of all rows",
        "the fit: past this the kept set is a sample the card selected",
        [f"LEVEL among the KEPT rows, recorded and never a reason to exclude: "
         f"{sides[T.LEVEL_HIGH]} HIGH, {sides[T.LEVEL_LOW]} LOW, "
         f"{sides['level']} level, {sides['undetermined']} undetermined"])


def gate_v5_within_burst(keep) -> Gate:
    """Was the clock steady INSIDE a burst, not only across the run.

    The DRIFT verdict compares the first and last clock sample of a cell, and
    those samples are one per burst. A card that boosts at the start of every
    40 ms burst and sags by its end passes DRIFT on every burst and still
    delivers a per-call time that is an average over two operating points. This
    reads the burst itself: the median of the first quarter of a burst's kept
    calls against the median of the last quarter, against `timing.DRIFT_FRACTION`
    -- the repository's own rule, applied to the one pair this instrument adds.
    """
    scored = [r for r in keep if r.within_burst_ok is not None]
    bad = [r for r in scored if r.within_burst_ok is False]
    if not scored:
        return Gate(VALIDITY, "5", "the clock held inside each burst", UNKNOWN,
                    "no row carried enough calls per burst to take two quarters",
                    f"first and last quarter within {T.DRIFT_FRACTION:.2f}",
                    "the per-call time, which would be an average over two "
                    "operating points inside one burst")
    ok = not bad
    worst = ""
    if bad:
        pick = max(bad, key=lambda r: abs((r.tail_ms or 0) - (r.head_ms or 0)))
        worst = (f"worst: duty {pick.duty_requested:g} tread {pick.tiles}, "
                 f"{pick.head_ms:.4f} -> {pick.tail_ms:.4f} ms")
    return Gate(
        VALIDITY, "5", "the clock held inside each burst",
        PASS if ok else FAIL,
        f"{len(bad)} of {len(scored)} kept rows moved more than "
        f"{T.DRIFT_FRACTION:.2f} from first quarter to last",
        f"0 rows, at {T.DRIFT_FRACTION:.2f}, the repository's DRIFT fraction",
        "the per-call time: inside one burst it would be an average over two "
        "operating points, and the average moves with the duty cycle",
        [worst] if worst else [])


def gate_v6_memory_clock(keep) -> Gate:
    """Did the MEMORY clock stay put while the SM clock moved.

    THE CONFOUND THIS ARM WOULD OTHERWISE CARRY. `d log ms / d log f_sm` is only
    the SM clock's elasticity if nothing else moved with it. HBM on this part is
    not DVFS'd the way the SM domain is, so the expectation is a flat line, and
    that expectation is worth one gate rather than one sentence: if the memory
    clock moved with the duty cycle, the measured slope is a blend of issue rate
    and bandwidth and the whole arm is answering a different question.

    UNKNOWN when nothing carried a reading. On the live path that cannot happen
    where the SM clock read -- both come from the same NVML handle in the same
    call -- and this branch exists for the planted worlds and for a future
    reader swapped in here that does not report it.
    """
    seen = sorted({int(r.mem_clock_mhz) for r in keep
                   if r.mem_clock_mhz and r.mem_clock_mhz > 0})
    if not seen:
        return Gate(VALIDITY, "6", "the memory clock did not move with the SM clock",
                    UNKNOWN, "no kept row carried a memory clock",
                    "one memory clock across every state",
                    "the reading: an SM-clock elasticity measured while the "
                    "memory clock moved is not an SM-clock elasticity")
    span = max(seen) - min(seen)
    ok = span <= T.CLOCK_STEP_MHZ
    return Gate(
        VALIDITY, "6", "the memory clock did not move with the SM clock",
        PASS if ok else FAIL,
        f"{len(seen)} distinct values, {min(seen)}-{max(seen)} MHz "
        f"(span {span} MHz)",
        f"span <= {T.CLOCK_STEP_MHZ:.0f} MHz, one NVML grid step",
        "the reading: the slope would be a blend of issue rate and bandwidth")


def gate_c1_resolved(est: Elasticity) -> Gate:
    """Does the interval sit WHOLLY inside one registered band.

    A FAIL here is a RESULT about the DESIGN and not about the card: the
    apparatus worked, the states separated, and the answer is that this many
    repeats at this much separation cannot tell the three registered worlds
    apart. It is deliberately scored before C2, because C2's word means nothing
    until this one passes.
    """
    band = band_of(est.lo, est.hi)
    hw = est.half_width
    return Gate(
        CLAIM, "1", "the interval falls wholly inside ONE registered band",
        PASS if band else FAIL,
        (f"[{est.lo:.4f}, {est.hi:.4f}], half-width {hw:.4f}, "
         f"{'in ' + band[0] if band else 'straddling a registered boundary'}"
         if est.lo is not None else "no interval: the fit produced no slope"),
        f"wholly inside one of {[b[0] for b in BANDS]} "
        f"(edges {BAND_LOW}, {BAND_HIGH}); half-width under "
        f"{RESOLUTION_TARGET:.3f} is the design's own target",
        "both registered consequences: an interval across a boundary is "
        "consistent with two worlds whose readings contradict each other",
        [f"the band it fell in says: {band[1]}"] if band else
        ["NO registered consequence may be read off this page. Widen the duty "
         "range or add repeats; do not pick the nearer edge."])


def gate_c2_registered_reading(est: Elasticity) -> Gate:
    """THE PRE-REGISTERED CLAIM: the elasticity is below the RAW-STANDS edge.

    Registered in P2 before the run. A FAIL is a finding, not a retry: it says
    the clock carries enough of a measured millisecond that the study's C3
    direction is in question, and the page prints the registered consequence of
    whichever band the interval actually landed in rather than a sentence
    written afterwards.
    """
    band = band_of(est.lo, est.hi)
    ok = band is not None and band[0] == BANDS[0][0]
    lines = []
    if band and not ok:
        lines.append(f"the registered consequence of {band[0]}: {band[1]}")
    elif not band:
        lines.append("the interval straddles a boundary (C1), so this FAIL is "
                     "'not shown' and not 'shown false'.")
    return Gate(
        CLAIM, "2", f"eta = -d log ms / d log f is below {BAND_LOW} at this cell",
        PASS if ok else FAIL,
        (f"{est.value:.4f} [{est.lo:.4f}, {est.hi:.4f}]"
         if est.value is not None and est.lo is not None
         else f"{est.value if est.value is not None else 'no fit'}"),
        f"the whole 95% interval below {BAND_LOW}",
        "nothing: a FAIL here is the arm's result and the study's reading of "
        "every alpha changes with it",
        lines)


def gates_for(rows, args, threshold: float, source: str,
              est: Elasticity) -> list[Gate]:
    """Every gate this run scores, in the order they are read.

    Flat and unconditional. A gate that vanishes when it cannot run reads as a
    page with fewer gates rather than as a page with an untested claim; the ones
    that cannot decide carry UNKNOWN, which counts against them.
    """
    keep = kept_rows(rows)
    cells = collapse(keep)
    return [
        gate_v0_non_vacuity(rows, keep),
        gate_v1_separation(cells, threshold, source),
        gate_v2_one_kernel(keep),
        gate_v3_depth(keep, args.duty, MIN_TREADS_PER_STATE,
                      MIN_REPEATS_PER_STATE, MIN_STATES),
        gate_v4_exclusions(rows, keep, EXCLUSION_CEILING),
        gate_v5_within_burst(keep),
        gate_v6_memory_clock(keep),
        gate_c1_resolved(est),
        gate_c2_registered_reading(est),
    ]


# --------------------------------------------------------------------------
# The plan: the cost, the design arithmetic, and the registered bands.
# --------------------------------------------------------------------------

def corpus_call_ms(tread: int) -> float:
    """Modelled per-call ms at a tread, for PRICING only.

    Inside the corpus ladder it is the measured median; past it, the OLS line
    through those eight points, so a `--treads` above 8 is priced by the same
    rule rather than dropping off the end of a tuple.
    """
    if 1 <= tread <= len(CORPUS_LADDER_MS):
        return CORPUS_LADDER_MS[tread - 1]
    xs = list(range(1, len(CORPUS_LADDER_MS) + 1))
    xbar = statistics.fmean(xs)
    ybar = statistics.fmean(CORPUS_LADDER_MS)
    xx = sum((x - xbar) ** 2 for x in xs)
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, CORPUS_LADDER_MS,
                                                    strict=True)) / xx
    return max(1e-3, ybar + b * (tread - xbar))


def burst_shape(per_call_ms: float, burst_ms: float, target_ms: float
                ) -> tuple[int, int, int]:
    """(calls per burst, bursts per trial, kept calls per trial).

    THE ONE PLACE THIS IS DECIDED, called by the plan and by the runner alike, so
    the wall clock an operator books is computed from the same rule the pod
    executes. The kept-call target is `timing.iters_for`'s, so a cell here holds
    the same number of samples a cell of the ladder arms holds.
    """
    calls = max(MIN_CALLS_PER_BURST,
                min(MAX_CALLS_PER_BURST,
                    int(round(burst_ms / max(per_call_ms, 1e-4)))))
    wanted = T.iters_for(per_call_ms, target_ms)
    bursts = max(1, math.ceil(wanted / max(1, calls - 1)))
    return calls, bursts, bursts * (calls - 1)


def estimated_seconds(args) -> tuple[float, list[str]]:
    """Wall seconds, itemised. A single number nobody can decompose is a number
    nobody can check before spending it."""
    per_repeat_kernel_ms = 0.0
    for tread in range(1, args.treads + 1):
        ms = corpus_call_ms(tread)
        calls, bursts, _kept = burst_shape(ms, args.burst_ms, args.target_ms)
        per_repeat_kernel_ms += args.warm_ms + args.trials * bursts * calls * ms
    inflation = sum(1.0 / d for d in args.duty)
    settles = len(args.duty) * args.settle_seconds
    per_repeat = settles + per_repeat_kernel_ms * inflation / 1000.0
    total = args.repeats * per_repeat + ALLOCATION_SECONDS
    lines = [
        f"  kernel per state per repeat   {per_repeat_kernel_ms / 1000.0:7.2f} s "
        f"over {args.treads} treads ({args.warm_ms:.0f} ms warm + "
        f"{args.trials} x ~{args.target_ms:.0f} ms each)",
        f"  duty inflation                {inflation:7.2f}x "
        f"= sum of 1/duty over {', '.join(f'{d:g}' for d in args.duty)}",
        f"  settles per repeat            {settles:7.2f} s "
        f"= {len(args.duty)} states x {args.settle_seconds:.0f} s",
        f"  per repeat                    {per_repeat:7.2f} s",
        f"  x {args.repeats} repeats                 "
        f"{args.repeats * per_repeat:7.2f} s",
        f"  + weight build and first compile {ALLOCATION_SECONDS:5.0f} s "
        "(an ALLOWANCE: 2.82 GB of mixtral expert weights built once and reused "
        "by every tread, and Triton's first choice under the pin, neither timed here)",
    ]
    return total, lines


def mde_lines(args) -> list[str]:
    """What this design can resolve, and the separation it therefore demands.

    Printed on the plan page and not in the post-mortem: a resolution discovered
    after the run is a description of the run.
    """
    states = len(args.duty)
    sigma_cell = MEDIAN_SE_PENALTY * CORPUS_REPEAT_SPREAD / math.sqrt(args.repeats)
    need, source = registered_clock_ratio(args)
    shape = math.sqrt((states + 1) / (12.0 * (states - 1))) if states > 1 else 0.0
    sd_at_need = math.log(need) * shape
    se = (sigma_cell / (math.sqrt(args.treads * states) * sd_at_need)
          if sd_at_need > 0 else float("inf"))
    return [
        "RESOLUTION, computed from the design and not from the result.",
        f"  across-repeat spread of one cell   {CORPUS_REPEAT_SPREAD:.5f} of log ms, "
        "the median over the 8 treads of the 2026-09-10 bn_decomposition ladder "
        "at this exact tile, 17 repeats each",
        f"  x median penalty sqrt(pi/2)        {MEDIAN_SE_PENALTY:.4f}",
        f"  / sqrt({args.repeats} repeats)                    -> sigma per cell "
        f"{sigma_cell:.5f}",
        f"  design                             {args.treads} treads x {states} "
        f"states = {args.treads * states} cells, slope Sxx = sum of "
        "(log f - tread mean)^2",
        f"  target half-width                  {RESOLUTION_TARGET:.4f}, half the "
        f"width of the unregistered gap [{BAND_LOW}, {BAND_HIGH}]",
        f"  so the states must span            {need:.4f}x in clock at every "
        f"tread ({source})",
        f"  at exactly that span the interval  half-width ~{2 * se:.4f}",
        "  THE THRESHOLD IS A RATIO OF TWO CLOCKS ON ONE CARD, never a clock. "
        "The same arithmetic gates an A100 without being told which part is "
        "attached, and a shallower run is held to a WIDER separation rather "
        "than to the same number.",
    ]


def prediction_lines(args) -> list[str]:
    out = ["PREDICTIONS, registered before the run and printed before any "
           "measurement", ""]
    for pred in PREDICTIONS:
        out += [f"  {line}" for line in pred.render()] + [""]
    out.append("THE THREE REGISTERED BANDS, and the reading each licenses:")
    for name, lo, hi, consequence in BANDS:
        edge = (f"< {hi:.2f}" if lo is None else
                f"> {lo:.2f}" if hi is None else f"{lo:.2f} to {hi:.2f}")
        out.append(f"  {name:<18} {edge:<14} {consequence}")
    out += [
        "",
        "  NOT A READOUT. Nothing above was measured. The arm decides which band "
        "it is in by",
        "  where its 95% interval falls, and it may decide none of them (C1 "
        "FAIL), which is",
        "  a result about the design and not a licence to read the nearer edge.",
    ]
    return out


def plan_lines(args, run_id: str, out_dir: Path, card: str,
               paths: dict) -> list[str]:
    threshold, source = registered_clock_ratio(args)
    cfg = MODEL_CONFIGS[args.model]
    treads = [n * PINNED["BLOCK_SIZE_M"] for n in range(1, args.treads + 1)]
    quantum = SWEEP.rows_quantum(cfg)
    out = [
        f"experiment  clock_elasticity / {run_id}",
        f"card        {card}"
        + ("   (no CUDA device: this is a plan or a replay)"
           if card == NO_CARD else ""),
        f"model       {args.model} {args.dtype}: E={cfg.num_experts} "
        f"k={cfg.top_k} H={cfg.hidden_size} F={cfg.intermediate_size}, "
        f"rows per expert must be a multiple of {quantum}",
        "pinned      " + ", ".join(f"{k}={v}" for k, v in sorted(PINNED.items())),
        f"ladder      {args.treads} exactly-full tile stacks, rows per expert "
        f"{treads[0]}..{treads[-1]}, tokens "
        f"{SWEEP.tokens_for_rows(cfg, treads[0])}.."
        f"{SWEEP.tokens_for_rows(cfg, treads[-1])}",
        f"states      duty {', '.join(f'{d:g}' for d in args.duty)}, "
        f"{args.settle_seconds:.0f} s of the state's own cadence before its "
        "first cell",
        f"repeats     {args.repeats}, state order reversed on alternate repeats "
        "so a thermal trend does not line up with the state axis",
        f"burst       target {args.burst_ms:.0f} ms of GPU per burst, at least "
        f"{MIN_CALLS_PER_BURST} calls, the FIRST call of every burst discarded",
        f"instrument  {INSTRUMENT}",
        f"WRITES TO   {out_dir}",
    ]
    for label, path in paths.items():
        out.append(f"  {label:<12}{path}   {git_visibility(path)}")
    out += [
        "",
        "NO RULER IS RESOLVED. An elasticity is a ratio of logs: no ridge, no "
        "bandwidth, no",
        "compute peak, no fitted level. The reference clock is resolved on the "
        "pod for the",
        "LEVEL RECORD only, and LEVEL excludes nothing here on either side; "
        "DRIFT and",
        "host-bound are the two exclusions and V4 holds them to "
        f"{100 * EXCLUSION_CEILING:.0f}% of all rows.",
        "",
        "THE BURST TABLE, computed by the same `burst_shape` the pod calls:",
        "  tread   modelled ms   calls/burst   bursts/trial   kept calls/trial",
    ]
    for tread in range(1, args.treads + 1):
        ms = corpus_call_ms(tread)
        calls, bursts, kept = burst_shape(ms, args.burst_ms, args.target_ms)
        out.append(f"  {tread:5d}   {ms:11.4f}   {calls:11d}   {bursts:12d}   "
                   f"{kept:16d}")
    out.append("  the modelled ms PRICE this run and score nothing; they are "
               "the 2026-09-10")
    out.append("  bn_decomposition medians at this exact tile, named in the "
               "source.")
    out += ["", f"V1 THRESHOLD {threshold:.3f}x, from {source}"]
    return out


# --------------------------------------------------------------------------
# The measurement. Everything below this line touches a GPU or is injected.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DutyTiming:
    """One cell measured at one duty cycle. `time_duty`'s whole return."""

    ms_p50: float
    ms_p90: float
    ms_min: float
    ms_std: float
    samples: int
    dropped_leads: int
    bursts: int
    trials: int
    calls_per_burst: int
    burst_ms: float
    gap_ms: float
    duty_requested: float
    #: Measured GPU-busy fraction: the sum of the event intervals over the wall
    #: clock of the trials. A LOWER BOUND on the true busy fraction and not an
    #: equality: the L2 flush is GPU work that sits outside every interval by
    #: construction, and the last burst of a trial is followed by a gap the
    #: wall clock counts. A RECORD, printed beside the requested figure; the
    #: regressor is the measured CLOCK, never the duty.
    duty_achieved: float
    head_ms: float | None
    tail_ms: float | None
    within_burst_ok: bool | None
    warmup_ms: float
    warmup_calls: int
    flush_mb: int
    l2_flush: bool
    sm_clock_load_mhz: float | None
    sm_clock_start_mhz: float | None
    sm_clock_end_mhz: float | None
    clock_samples_mhz: tuple[float, ...]
    clock_source: str
    clock_level_ok: bool | None
    clock_level_side: str
    clock_drift_ok: bool | None
    clock_drift_direction: str
    reference_clock_mhz: float | None
    power_w: float | None
    mem_clock_mhz: float | None
    host_bound: bool | None
    host_enqueue_ms: float | None
    host_backlog_iters: float | None
    clock_note: str
    instrument: str = INSTRUMENT


def _quarter_medians(intervals: list[float]) -> tuple[float | None, float | None]:
    """Medians of the first and last quarter of one burst's KEPT calls."""
    n = len(intervals)
    if n < 4:
        return None, None
    q = max(1, n // 4)
    return statistics.median(intervals[:q]), statistics.median(intervals[-q:])


def _warm_at_duty(fn, *, warm_ms: float, calls: int, gap_s: float, events,
                  flush, clock_read, sleep) -> tuple[float, int, list[float]]:
    """Deliver `warm_ms` of GPU load AT THE STATE'S OWN CADENCE, then settle.

    `timing.warm_until` is the right function for a back-to-back loop and the
    wrong one here for the reason the whole arm exists: it runs bursts with no
    gap, so warming with it would hold the card at duty 1.0 and hand the trials
    a clock that belongs to a state nobody is measuring. The rule it applies is
    kept exactly -- deliver the load, then wait for two consecutive under-load
    reads to agree within one `CLOCK_STEP_MHZ`, capped at
    `SETTLE_CAP_MULTIPLE` times the warm budget -- and only the cadence changes.
    """
    pairs = events(calls)
    delivered = 0.0
    made = 0
    reads: list[float] = []
    cap = warm_ms * T.SETTLE_CAP_MULTIPLE
    settled = False
    while delivered < warm_ms or (not settled and delivered < cap):
        for i in range(calls):
            if flush is not None:
                flush()
            pairs.starts[i].record()
            fn()
            pairs.ends[i].record()
        state = clock_read() if clock_read is not None else None
        pairs.synchronize()
        got = pairs.elapsed(calls)
        delivered += sum(got)
        made += calls
        if state is not None and state.sm_clock_mhz > 0:
            reads.append(float(state.sm_clock_mhz))
            if (delivered >= warm_ms and len(reads) >= 2
                    and abs(reads[-1] - reads[-2]) <= T.CLOCK_STEP_MHZ):
                settled = True
        elif delivered >= warm_ms:
            settled = True
        if gap_s > 0:
            sleep(gap_s)
    return delivered, made, reads


def time_duty(fn, *, duty: float, calls_per_burst: int, bursts: int,
              trials: int, warm_ms: float, l2_flush: bool,
              per_call_ms: float, reference_clock_mhz: float | None,
              clock_read=None, mem_read=None, events=None, flusher=None,
              sleep=time.sleep) -> DutyTiming:
    """Time `fn` at a sustained duty cycle, sampling the clock in flight.

    The loop, per trial: `bursts` bursts; per burst, `calls_per_burst` calls
    enqueued back to back, each preceded by the L2 flush and bracketed by its own
    pre-primed event pair; then ONE NVML read while the burst is still queued;
    then one synchronise; then a host-side sleep of the gap. The FIRST call of
    every burst is discarded from the samples: it launched into a queue the
    previous gap had drained, so its interval carries launch latency the others
    do not, and that latency would otherwise enter the elasticity as a constant
    offset that does not cancel in a log slope.

    THE GAP IS THE INDEPENDENT VARIABLE AND IT IS OUTSIDE EVERY INTERVAL. It sets
    sustained board power, sustained board power sets the settled clock under the
    cap, and the kernel's bytes and instructions never change.

    EVERY STATE DRAINS THE QUEUE THE SAME WAY, the anchor at duty 1.0 included:
    the synchronise is per burst, not per trial, so every burst in every state
    starts on an idle GPU and loses exactly one lead call. That symmetry is why
    the discard does not have to be modelled -- it is the same in every state --
    and it is the reason this loop synchronises more often than
    `timing._timed_trials` does.

    OFF-GPU it refuses with `timing.TimingRefused` unless `events`, `clock_read`
    and (when flushing) `flusher` are all injected, on `time_kernel`'s own terms
    and for its own reason: an instrument that invents numbers when its device is
    missing is worse than one that stops.
    """
    if trials < 1 or bursts < 1 or calls_per_burst < 2:
        raise T.TimingRefused(
            f"trials={trials} bursts={bursts} calls_per_burst={calls_per_burst}: "
            "a burst needs at least two calls (one is discarded as the "
            "drained-queue lead) and a cell needs at least one of each")
    import torch

    on_gpu = torch.cuda.is_available()
    missing = [name for name, given in (("events", events),
                                        ("clock_read", clock_read),
                                        ("flusher", flusher if l2_flush else True))
               if given is None]
    if missing and not on_gpu:
        raise T.TimingRefused(
            "no CUDA device, and no fake injected for " + ", ".join(missing)
            + "; time_duty measures a GPU or runs against injected fakes, it "
            "does not invent numbers")
    if events is None:
        events = T._EventPairs
    if l2_flush and flusher is None:
        flusher = T.L2Flusher(T.flush_mb_for_device())
    if clock_read is None:
        clock_read = T.nvml_clock_reader(torch.cuda.current_device())
    flush = flusher.flush if l2_flush else None
    flush_mb = int(flusher.megabytes) if l2_flush else 0

    burst_gpu_ms = calls_per_burst * max(per_call_ms, 1e-4)
    gap_ms = burst_gpu_ms * (1.0 / max(duty, 1e-6) - 1.0)
    gap_s = max(0.0, gap_ms / 1000.0)

    delivered, warm_calls, _reads = _warm_at_duty(
        fn, warm_ms=warm_ms, calls=calls_per_burst, gap_s=gap_s, events=events,
        flush=flush, clock_read=clock_read, sleep=sleep)

    pairs = events(calls_per_burst)
    samples: list[float] = []
    heads: list[float] = []
    tails: list[float] = []
    walls: list[T.TrialWall] = []
    clocks: list[float] = []
    powers: list[float] = []
    mems: list[float] = []
    sources: set[str] = set()
    leads = 0
    busy_ms = 0.0
    wall_s = 0.0
    for _trial in range(trials):
        t_start = time.perf_counter()
        for _burst in range(bursts):
            t0 = time.perf_counter()
            for i in range(calls_per_burst):
                if flush is not None:
                    flush()
                pairs.starts[i].record()
                fn()
                pairs.ends[i].record()
            t1 = time.perf_counter()
            state = clock_read()
            if mem_read is not None:
                mem = mem_read()
                if mem:
                    mems.append(float(mem))
            pairs.synchronize()
            t2 = time.perf_counter()
            got = pairs.elapsed(calls_per_burst)
            busy_ms += sum(got)
            leads += 1
            kept = got[1:]
            samples.extend(kept)
            head, tail = _quarter_medians(kept)
            if head is not None:
                heads.append(head)
                tails.append(tail)
            walls.append(T.TrialWall(enqueue_s=t1 - t0, wall_s=t2 - t0))
            sources.add(state.source)
            if state.sm_clock_mhz > 0:
                clocks.append(float(state.sm_clock_mhz))
            if getattr(state, "power_w", 0.0) > 0:
                powers.append(float(state.power_w))
            if gap_s > 0:
                sleep(gap_s)
        wall_s += time.perf_counter() - t_start

    notes: list[str] = []
    load = start = end = None
    if len(clocks) >= T.CLOCK_SAMPLE_FLOOR:
        load = float(statistics.median(clocks))
        if reference_clock_mhz is None:
            notes.append("no reference clock given, level not determinable")
    elif clocks:
        notes.append(f"{len(clocks)} usable clock sample(s) over {leads} bursts "
                     f"is below the floor of {T.CLOCK_SAMPLE_FLOOR} a median is "
                     "taken from; this cell carries no under-load clock and the "
                     "fit cannot read it")
    else:
        notes.append(f"no usable SM clock sample over {leads} bursts; the fit "
                     "cannot read this cell")
    if len(clocks) >= 2:
        start, end = clocks[0], clocks[-1]
    level_ok, drift_ok = T.clock_flags(load, start, end, reference_clock_mhz)
    side = T.level_side(load, reference_clock_mhz) or ""
    direction = T.drift_direction(clocks) if drift_ok is False else ""
    if side:
        notes.append(
            f"LEVEL {side.upper()} (recorded, never an exclusion here): loaded "
            f"clock {load:.0f} MHz against the {reference_clock_mhz:.0f} MHz "
            "reference. At a duty cycle below 1.0 a HIGH reading is the "
            "experiment working, not a cell to drop.")
    if direction:
        notes.append(f"DRIFT failed, {direction}: {start:.0f} -> {end:.0f} MHz "
                     f"over {len(clocks)} bursts; this cell is EXCLUDED")
    host_bound, host_ms, backlog, host_note = T.host_bound_verdict(
        walls, calls_per_burst)
    if host_note:
        notes.append(host_note)

    head_ms = statistics.median(heads) if heads else None
    tail_ms = statistics.median(tails) if tails else None
    within_ok = None
    if head_ms and tail_ms and head_ms > 0:
        within_ok = abs(tail_ms - head_ms) / head_ms <= T.DRIFT_FRACTION
    p50, p90, lo, std = T._stats(samples) if samples else (0.0, 0.0, 0.0, 0.0)
    return DutyTiming(
        ms_p50=p50, ms_p90=p90, ms_min=lo, ms_std=std, samples=len(samples),
        dropped_leads=leads, bursts=bursts * trials, trials=trials,
        calls_per_burst=calls_per_burst,
        burst_ms=busy_ms / leads if leads else 0.0, gap_ms=gap_ms,
        duty_requested=duty,
        duty_achieved=(busy_ms / 1000.0 / wall_s) if wall_s > 0 else 0.0,
        head_ms=head_ms, tail_ms=tail_ms, within_burst_ok=within_ok,
        warmup_ms=delivered, warmup_calls=warm_calls, flush_mb=flush_mb,
        l2_flush=l2_flush, sm_clock_load_mhz=load, sm_clock_start_mhz=start,
        sm_clock_end_mhz=end, clock_samples_mhz=tuple(clocks),
        clock_source=(sorted(sources)[0] if sources else T.CLOCK_SOURCE_NONE),
        clock_level_ok=level_ok, clock_level_side=side, clock_drift_ok=drift_ok,
        clock_drift_direction=direction,
        reference_clock_mhz=reference_clock_mhz,
        power_w=float(statistics.median(powers)) if powers else None,
        mem_clock_mhz=float(statistics.median(mems)) if mems else None,
        host_bound=host_bound, host_enqueue_ms=host_ms,
        host_backlog_iters=backlog, clock_note="; ".join(notes))


class _MemClockReader:
    """NVML's memory clock, or 0.0 forever if it cannot be read.

    Initialised ONCE, like `thermal_acceptance._ReasonReader` and for the same
    reason: a reader that cannot answer must not spend the whole run raising.
    Separate from `timing.nvml_clock_reader` because that reader's `ClockState`
    is a shared record three other arms write rows from, and widening it for one
    arm's control is a change at every one of their call sites.
    """

    def __init__(self, index: int = 0) -> None:
        self._get = None
        self._handle = None
        self._pynvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
            self._get = pynvml.nvmlDeviceGetClockInfo
            self._which = pynvml.NVML_CLOCK_MEM
        except Exception:                               # noqa: BLE001
            # Broad for `ClockState.sample`'s reasons: a missing pynvml is a
            # ModuleNotFoundError and a restricted container raises NVML's own
            # family. Neither may stop the run; the memory clock is a control
            # and V6 reads UNKNOWN when nothing carried one.
            self._handle = None

    def __call__(self) -> float:
        if self._handle is None or self._get is None:
            return 0.0
        try:
            return float(self._get(self._handle, self._which))
        except Exception:                               # noqa: BLE001
            return 0.0

    def close(self) -> None:
        try:
            if self._handle is not None:
                self._pynvml.nvmlShutdown()
        except Exception:                               # noqa: BLE001
            pass


def build_ladder(cfg, args, treads):
    """One set of tensors for the whole arm, built once and reused everywhere.

    THE REASON THIS IS NOT A PER-CELL BUILD. "The same kernel over the same
    bytes in every state" is the arm's central control, and built once it is
    true by construction rather than by a comparison of shapes afterwards.
    `make_inputs` caches weights on (model, dtype, seed, device, scale), so the
    2.82 GB expert set is drawn once and every tread's activations point at it;
    the activations themselves are 8 MB at the deepest tread.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from moe.baselines._framework_config import vllm_call_kwargs
    from moe.reference.torch_ref import make_inputs
    from moe.spec import BenchSpec, RoutingSpec

    built = {}
    for tread in treads:
        rows = tread * PINNED["BLOCK_SIZE_M"]
        tokens = SWEEP.tokens_for_rows(cfg, rows)
        spec = BenchSpec(cfg, num_tokens=tokens, dtype=args.dtype,
                         routing=RoutingSpec("uniform", 0.0), seed=args.seed)
        x, weights = make_inputs(spec, device="cuda")
        ids = SWEEP.balanced_ids(cfg, tokens, "cuda")
        import torch
        w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32,
                       device="cuda")
        kw = vllm_call_kwargs(spec)
        kw["activation"] = MoEActivation(kw["activation"])
        built[tread] = (rows, tokens, x, weights, ids, w, kw)
    return built


def run_arm(args, cfg, csv_path: Path, cache_root: Path, prov=None) -> list[Row]:
    """The metered part. Appends every row as it lands, so aborting keeps it."""
    import torch
    from vllm.model_executor.layers.fused_moe import fused_experts

    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)
    override_config, where = SWEEP.find_override()
    print(f"override hook: {where}.override_config")
    print(f"triton cache:  {cache_root} (fresh for this run)")

    reference_clock, clock_source = SWEEP.reference_clock_mhz()
    print("reference clock: "
          + (f"{reference_clock:.0f} MHz, {clock_source}" if reference_clock
             else f"NOT RESOLVED ({clock_source}); every row's LEVEL verdict "
                  "will be None, which excludes nothing here because LEVEL "
                  "never excludes anything here"))

    T.require_cuda()
    treads = list(range(1, args.treads + 1))

    # WHAT IS ALREADY ON DISK. A resumed 40-minute arm that re-measured every
    # cell would double every row in the file and the medians would be taken
    # over two thermal histories with no column saying so.
    done = read_rows(csv_path)
    have = {(r.repeat, _duty_key(r.duty_requested), r.tiles)
            for r in done if r.status == "ok"}
    if have:
        print(f"resuming: {len(have)} cells already on disk in {csv_path}")

    built = build_ladder(cfg, args, treads)
    read = T.nvml_clock_reader(torch.cuda.current_device())
    mem_read = _MemClockReader(torch.cuda.current_device())
    flusher = T.L2Flusher(T.flush_mb_for_device())

    # THE BURST SHAPE IS SIZED ONCE, FOR THE WHOLE ARM, AT FULL DUTY. Sized per
    # state it would be a second thing that moves with the clock: a faster clock
    # is a shorter call, a shorter call is more calls per burst, and more calls
    # per burst is a smaller share of one discarded lead. V2 reads the shape off
    # the rows precisely because it must be the same in every state.
    shape: dict[int, tuple[int, int, float]] = {}
    print("\nsizing the burst at full duty, once, for every tread:")
    for tread in treads:
        _rows, tokens, x, weights, ids, w, kw = built[tread]
        call = SWEEP._make_call(fused_experts, x, weights, w, ids, kw)
        with override_config(dict(PINNED)):
            call()
            torch.cuda.synchronize()
            warm = T.warm_until(call, args.warm_ms, T._EventPairs,
                                flush=flusher.flush, clock_read=read)
        calls, bursts, kept = burst_shape(warm.per_call_ms, args.burst_ms,
                                          args.target_ms)
        # A RESUMED TREAD ADOPTS THE SHAPE ALREADY ON ITS ROWS. Re-sized from a
        # fresh warm the second process could land one call either side of the
        # rounding and V2 would then FAIL the whole arm for two burst shapes at
        # one tread -- correctly, because the discarded lead would be a
        # different share of two halves of one file. The recorded shape is the
        # run's, so the resume reads it back rather than re-deriving it.
        prior = {r.calls_per_burst for r in done
                 if r.tiles == tread and r.status == "ok" and r.calls_per_burst}
        adopted = ""
        if len(prior) == 1 and prior != {calls}:
            calls = prior.pop()
            bursts = max(1, math.ceil(
                T.iters_for(warm.per_call_ms, args.target_ms) / max(1, calls - 1)))
            kept = bursts * (calls - 1)
            adopted = "  [adopted from the rows already on disk]"
        shape[tread] = (calls, bursts, warm.per_call_ms)
        print(f"  tread {tread:2d}  T={tokens:5d}  {warm.per_call_ms:8.4f} ms/call "
              f"-> {calls:3d} calls x {bursts} bursts = {kept} kept calls/trial"
              + adopted)

    rows: list[Row] = []
    order = list(args.duty)
    try:
        for repeat in range(args.repeats):
            states = order if repeat % 2 == 0 else list(reversed(order))
            for index, duty in enumerate(states):
                calls, bursts, per_call = shape[treads[len(treads) // 2]]
                mid = built[treads[len(treads) // 2]]
                settle_call = SWEEP._make_call(fused_experts, mid[2], mid[3],
                                               mid[5], mid[4], mid[6])
                print(f"\nrepeat {repeat} state duty={duty:g}: settling "
                      f"{args.settle_seconds:.0f} s at this cadence")
                with override_config(dict(PINNED)):
                    _settle(settle_call, duty=duty, calls=calls,
                            per_call_ms=per_call, seconds=args.settle_seconds,
                            flusher=flusher)
                for tread in treads:
                    if (repeat, _duty_key(duty), tread) in have:
                        continue
                    rows_per_expert, tokens, x, weights, ids, w, kw = built[tread]
                    calls, bursts, per_call = shape[tread]
                    call = SWEEP._make_call(fused_experts, x, weights, w, ids, kw)
                    try:
                        with override_config(dict(PINNED)):
                            t = time_duty(
                                call, duty=duty, calls_per_burst=calls,
                                bursts=bursts, trials=args.trials,
                                warm_ms=args.warm_ms,
                                l2_flush=not args.no_l2_flush,
                                per_call_ms=per_call,
                                reference_clock_mhz=reference_clock,
                                clock_read=read, mem_read=mem_read,
                                flusher=flusher)
                        row = _row_from(t, args, duty, index, repeat, tread,
                                        rows_per_expert, tokens)
                    except T.TimingRefused:
                        # THE INSTRUMENT'S OWN REFUSAL IS NOT ONE CELL'S ERROR.
                        # Filed as a failed row, the arm would walk the whole
                        # design writing zeroes and exit through the gates as
                        # though it had measured.
                        raise
                    except Exception as exc:            # noqa: BLE001
                        row = _failed_row(args, duty, index, repeat, tread,
                                          rows_per_expert, tokens,
                                          f"{type(exc).__name__}: {exc}")
                        print(f"  FAILED duty={duty:g} tread={tread}: {row.detail}")
                    rows.append(row)
                    append_row(csv_path, row, prov)
                    print(f"  duty {duty:4.2f} tread {tread:2d} "
                          f"{row.ms_p50:9.4f} ms  "
                          f"clock {row.sm_clock_load_mhz or 0:7.1f} MHz  "
                          f"power {row.power_w or 0:6.1f} W  "
                          f"duty_seen {row.duty_achieved:5.3f}"
                          + (f"  [{exclusion(row)}]" if exclusion(row) else ""))
    finally:
        mem_read.close()
    # THE ROWS THE FIT READS ARE THE FILE'S, not this process's. A resume that
    # scored only the cells it happened to measure would report a design four
    # states wide as one, and V3 would refuse an arm that is in fact complete.
    return read_rows(csv_path) if have else rows


def _settle(fn, *, duty: float, calls: int, per_call_ms: float, seconds: float,
            flusher) -> None:
    """Run the state's cadence for `seconds` of wall, to move the power average.

    No clock is read and nothing is recorded: this is the coarse transition
    between two states, and the fine one is `_warm_at_duty` inside each cell,
    which does settle on the clock. Kept separate because the two have different
    time constants -- a board power average moves in tens of milliseconds and a
    die temperature in tens of seconds -- and a single loop tuned for one of them
    is wrong for the other.
    """
    import torch

    gap_s = max(0.0, (calls * per_call_ms / 1000.0) * (1.0 / max(duty, 1e-6) - 1.0))
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for _ in range(calls):
            flusher.flush()
            fn()
        torch.cuda.synchronize()
        if gap_s > 0:
            time.sleep(gap_s)


def _row_from(t: DutyTiming, args, duty, index, repeat, tread,
              rows_per_expert, tokens) -> Row:
    """One row. `state_index` is the duty's own place in `--duty` and
    `order_index` is where it fell in THIS repeat, which are different numbers
    on the reversed repeats and are both worth keeping: the first identifies the
    state, the second is what a reader checks a thermal trend against."""
    return Row(
        duty_requested=duty, duty_achieved=t.duty_achieved,
        state_index=list(args.duty).index(duty),
        repeat=repeat, order_index=index, model=args.model, dtype=args.dtype,
        block_m=PINNED["BLOCK_SIZE_M"], block_n=PINNED["BLOCK_SIZE_N"],
        block_k=PINNED["BLOCK_SIZE_K"], group_m=PINNED["GROUP_SIZE_M"],
        num_warps=PINNED["num_warps"], num_stages=PINNED["num_stages"],
        tiles=tread, rows_per_expert=rows_per_expert, tokens=tokens,
        calls_per_burst=t.calls_per_burst, bursts=t.bursts, trials=t.trials,
        ms_p50=t.ms_p50, ms_min=t.ms_min, ms_stdev=t.ms_std, samples=t.samples,
        burst_ms=t.burst_ms, gap_ms=t.gap_ms, head_ms=t.head_ms,
        tail_ms=t.tail_ms, within_burst_ok=t.within_burst_ok,
        sm_clock_load_mhz=t.sm_clock_load_mhz,
        sm_clock_start_mhz=t.sm_clock_start_mhz,
        sm_clock_end_mhz=t.sm_clock_end_mhz,
        clock_samples_mhz=" ".join(f"{c:.0f}" for c in t.clock_samples_mhz),
        clock_samples=len(t.clock_samples_mhz), clock_source=t.clock_source,
        clock_level_ok=t.clock_level_ok, clock_level_side=t.clock_level_side,
        clock_drift_ok=t.clock_drift_ok,
        clock_drift_direction=t.clock_drift_direction,
        reference_clock_mhz=t.reference_clock_mhz, power_w=t.power_w,
        mem_clock_mhz=t.mem_clock_mhz, host_bound=t.host_bound,
        host_enqueue_ms=t.host_enqueue_ms,
        host_backlog_iters=t.host_backlog_iters, warmup_ms=t.warmup_ms,
        l2_flush=t.l2_flush, status="ok", detail=t.clock_note)


def _failed_row(args, duty, index, repeat, tread, rows_per_expert, tokens,
                detail: str) -> Row:
    return Row(
        duty_requested=duty, duty_achieved=0.0,
        state_index=list(args.duty).index(duty), repeat=repeat,
        order_index=index, model=args.model, dtype=args.dtype,
        block_m=PINNED["BLOCK_SIZE_M"], block_n=PINNED["BLOCK_SIZE_N"],
        block_k=PINNED["BLOCK_SIZE_K"], group_m=PINNED["GROUP_SIZE_M"],
        num_warps=PINNED["num_warps"], num_stages=PINNED["num_stages"],
        tiles=tread, rows_per_expert=rows_per_expert, tokens=tokens,
        calls_per_burst=0, bursts=0, trials=args.trials, ms_p50=0.0, ms_min=0.0,
        ms_stdev=0.0, samples=0, burst_ms=0.0, gap_ms=0.0, head_ms=None,
        tail_ms=None, within_burst_ok=None, sm_clock_load_mhz=None,
        sm_clock_start_mhz=None, sm_clock_end_mhz=None, clock_samples_mhz="",
        clock_samples=0, clock_source=T.CLOCK_SOURCE_NONE, clock_level_ok=None,
        clock_level_side="", clock_drift_ok=None, clock_drift_direction="",
        reference_clock_mhz=None, power_w=None, mem_clock_mhz=None,
        host_bound=None, host_enqueue_ms=None, host_backlog_iters=None,
        warmup_ms=0.0, l2_flush=not args.no_l2_flush, status="failed",
        detail=detail)


# --------------------------------------------------------------------------
# Persistence. A module-level appender and NOT a `Store`: the Store tripwire in
# tests/test_shell_gates.py pins the set of Store classes to three, and a fourth
# would go red in another slice's file. The header check it exists to install is
# implemented here all the same, because the defect is the missing check and not
# the missing class name.
# --------------------------------------------------------------------------

ROW_FIELDS = list(Row.__dataclass_fields__)


class SchemaCollision(SystemExit):
    """A cells.csv on disk whose header is not the one these rows are written to.

    A `SystemExit` carrying REFUSED, so an uncaught one exits 2 and not the 1
    that means CLAIM_FAIL. `DictWriter` writes the fieldnames it was given and
    never looks at the file, so a wider row appended under a narrower header
    shifts every field past the first difference: `clock_drift_ok` would then be
    read out of `clock_level_side`, and a FAILED drift -- the one verdict in this
    file that excludes a row -- would come back None, which every gate keeps.
    """

    code = exit_codes.REFUSED


def append_row(path: Path, row: Row, prov=None) -> None:
    """Append one row, flushed, after checking the header already on disk."""
    fields = list(ROW_FIELDS)
    if prov is not None:
        fields += list(SWEEP.PROVENANCE_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="") as fh:
            on_disk = next(csv.reader(fh), [])
        if on_disk and on_disk != fields:
            added = [c for c in fields if c not in on_disk]
            gone = [c for c in on_disk if c not in fields]
            raise SchemaCollision(
                f"{path} already holds a header this run would not write.\n"
                f"  columns this run adds:   {added or 'none'}\n"
                f"  columns on disk and not here: {gone or 'none'}\n"
                "Appending under it would shift every field past the first "
                "difference. Use a fresh --out or --run-id.")
    payload = asdict(row)
    if prov is not None:
        payload.update(prov.as_columns())
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if fresh:
            writer.writeheader()
        writer.writerow(payload)
        fh.flush()


_OPT_FLOAT = {"head_ms", "tail_ms", "sm_clock_load_mhz", "sm_clock_start_mhz",
              "sm_clock_end_mhz", "reference_clock_mhz", "power_w",
              "mem_clock_mhz", "host_enqueue_ms", "host_backlog_iters"}
_OPT_BOOL = {"within_burst_ok", "clock_level_ok", "clock_drift_ok", "host_bound"}


def _opt_float(text: str) -> float | None:
    return float(text) if text not in ("", "None") else None


def _opt_bool(text: str) -> bool | None:
    if text in ("", "None"):
        return None
    return text.lower() in ("true", "1")


def read_rows(path: Path) -> list[Row]:
    """Read rows BY HEADER NAME, so a file written by an older column set comes
    back with those fields absent rather than defaulted to 0 or False."""
    if not path.exists():
        return []
    out = []
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            kwargs = {}
            for name, spec in Row.__dataclass_fields__.items():
                if name not in raw:
                    continue
                text = raw[name]
                if name in _OPT_FLOAT:
                    kwargs[name] = _opt_float(text)
                elif name in _OPT_BOOL:
                    kwargs[name] = _opt_bool(text)
                elif spec.type in ("int",):
                    kwargs[name] = int(float(text or 0))
                elif spec.type in ("float",):
                    kwargs[name] = float(text or 0.0)
                elif spec.type in ("bool",):
                    kwargs[name] = str(text).lower() in ("true", "1")
                else:
                    kwargs[name] = text
            out.append(Row(**kwargs))
    return out


# --------------------------------------------------------------------------
# The self-test: planted worlds whose answer is known.
# --------------------------------------------------------------------------

#: The clock a planted world's duty states sit at, MHz. NOT a calibration and
#: not this repository's H200: a set of four numbers spanning 1.30x, chosen so
#: the planted design clears its own V1 threshold, so that every world below
#: fails for the reason it was planted to fail for and not for a separation the
#: planter forgot to give it.
PLANTED_STATE_MHZ = (1500.0, 1650.0, 1800.0, 1950.0)
PLANTED_REFERENCE_MHZ = 1650.0
PLANTED_MEM_MHZ = 2619.0


def plant_rows(*, eps: float, duties=DUTY_LEVELS, mhz=PLANTED_STATE_MHZ,
               treads: int = DEFAULT_TREADS, repeats: int = DEFAULT_REPEATS,
               a_ms: float = 0.1936, b_ms: float = 0.5559,
               jitter: float = 0.0, seed: int = 7,
               block_m_at=None, drift_at=(), host_at=(),
               level_at=None, mem_at=None, burst_moves_at=(),
               calls_per_burst: int = 24) -> list[Row]:
    """Rows from a stated law: `ms(n, f) = (a + b n) (f_ref / f) ** eps`.

    THE LAW IS THE ONLY SOURCE OF TIME IN THESE ROWS, which is what makes the
    self-test a recovery test rather than a smoke test: the estimator either
    returns `eps` or it does not, and `SELF_TEST_WORLDS` registers which.

    `drift_at`, `host_at` and `level_at` plant the three verdicts separately, on
    purpose: LEVEL on either side must be KEPT and DRIFT must be EXCLUDED, and a
    planter that only ever put them on the same row could not tell a counter that
    filters on LEVEL from one that filters on DRIFT.
    """
    rng = random.Random(seed)
    level_at = level_at or {}
    block_m_at = block_m_at or {}
    out = []
    for repeat in range(repeats):
        for index, duty in enumerate(duties):
            f = mhz[index % len(mhz)]
            for tread in range(1, treads + 1):
                base = (a_ms + b_ms * tread) * (PLANTED_REFERENCE_MHZ / f) ** eps
                ms = base * (1.0 + rng.gauss(0.0, jitter) if jitter else 1.0)
                key = (index, tread, repeat)
                drift = (key in drift_at)
                samples = [f] * 6 if not drift else [f, f, f * 0.80]
                side = level_at.get(key, T.level_side(f, PLANTED_REFERENCE_MHZ) or "")
                level_ok = T.clock_flags(f, samples[0], samples[-1],
                                         PLANTED_REFERENCE_MHZ)[0]
                head = ms
                tail = ms * (1.20 if key in burst_moves_at else 1.0)
                out.append(Row(
                    duty_requested=duty, duty_achieved=duty, state_index=index,
                    repeat=repeat, order_index=index, model=DEFAULT_MODEL,
                    dtype=DEFAULT_DTYPE,
                    block_m=block_m_at.get(key, PINNED["BLOCK_SIZE_M"]),
                    block_n=PINNED["BLOCK_SIZE_N"],
                    block_k=PINNED["BLOCK_SIZE_K"],
                    group_m=PINNED["GROUP_SIZE_M"],
                    num_warps=PINNED["num_warps"],
                    num_stages=PINNED["num_stages"], tiles=tread,
                    rows_per_expert=tread * PINNED["BLOCK_SIZE_M"],
                    tokens=tread * PINNED["BLOCK_SIZE_M"] * 4,
                    calls_per_burst=calls_per_burst, bursts=18,
                    trials=DEFAULT_TRIALS, ms_p50=ms, ms_min=ms * 0.99,
                    ms_stdev=ms * 0.005, samples=300, burst_ms=40.0,
                    gap_ms=40.0 * (1.0 / duty - 1.0), head_ms=head,
                    tail_ms=tail,
                    within_burst_ok=abs(tail - head) / head <= T.DRIFT_FRACTION,
                    sm_clock_load_mhz=f, sm_clock_start_mhz=samples[0],
                    sm_clock_end_mhz=samples[-1],
                    clock_samples_mhz=" ".join(f"{c:.0f}" for c in samples),
                    clock_samples=len(samples),
                    clock_source=T.CLOCK_SOURCE_NVML, clock_level_ok=level_ok,
                    clock_level_side=side,
                    clock_drift_ok=not drift,
                    clock_drift_direction=(T.DRIFT_DOWN if drift else ""),
                    reference_clock_mhz=PLANTED_REFERENCE_MHZ, power_w=690.0,
                    mem_clock_mhz=(mem_at.get(key, PLANTED_MEM_MHZ)
                                   if mem_at else PLANTED_MEM_MHZ),
                    host_bound=(key in host_at), host_enqueue_ms=0.5,
                    host_backlog_iters=40.0, warmup_ms=200.0, l2_flush=True))
    return out


@dataclass(frozen=True)
class PlantedWorld:
    """One world, its registered verdicts, and the elasticity it must recover."""

    name: str
    why: str
    rows: list[Row]
    expect: dict[str, str]
    #: The planted elasticity, or None where the world is not about recovery.
    recovers: float | None = None
    tolerance: float = 0.05


def self_test_worlds(args) -> list[PlantedWorld]:
    """Every planted world. Four of the nine are REFUSALS: a scorer that has only
    ever seen a clean design has never been shown to refuse one."""
    duties = list(DUTY_LEVELS)
    n_states = len(duties)
    return [
        PlantedWorld(
            "traffic", "elasticity 0.05: the measured time is nearly all "
            "traffic. RAW-STANDS, so C2 PASSES and the study's raw readings "
            "hold.",
            plant_rows(eps=0.05, jitter=0.004),
            {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
             "V5": PASS, "V6": PASS, "C1": PASS, "C2": PASS}, 0.05),
        PlantedWorld(
            "clock-carries", "elasticity 0.60: past the upper edge. C2 FAILS, "
            "and that FAIL is the finding -- C3's DIRECTION is retracted. THE "
            "WORLD THAT MAKES C2'S FAIL BRANCH REACHABLE.",
            plant_rows(eps=0.60, jitter=0.004),
            {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
             "V5": PASS, "V6": PASS, "C1": PASS, "C2": FAIL}, 0.60),
        PlantedWorld(
            "unregistered-gap", "elasticity 0.32: inside the gap the analysis "
            "never stopped in. C1 PASSES -- the design resolved WHICH world -- "
            "and C2 FAILS with 'no registered consequence' rather than a "
            "sentence written afterwards.",
            plant_rows(eps=0.32, jitter=0.004),
            {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
             "V5": PASS, "V6": PASS, "C1": PASS, "C2": FAIL}, 0.32),
        PlantedWorld(
            "straddling", "elasticity 0.25 at ten times the noise: the interval "
            "crosses the lower edge. C1 FAILS, which is a result about the "
            "DESIGN -- and C2's FAIL then reads 'not shown', not 'shown false'. "
            "THE WORLD THAT MAKES C1'S FAIL BRANCH REACHABLE.",
            plant_rows(eps=0.25, jitter=0.04, seed=11),
            {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
             "V5": PASS, "V6": PASS, "C1": FAIL, "C2": FAIL}, None),
        PlantedWorld(
            "one-clock", "A REFUSAL: four duty states that all settled at the "
            "same clock. The elasticity is undefined and V1 says so; without "
            "this gate the arm would report a slope over nothing.",
            plant_rows(eps=0.20, mhz=(1650.0,) * n_states),
            {"V0": PASS, "V1": FAIL, "V2": PASS, "V3": PASS, "V4": PASS,
             "V5": PASS, "V6": PASS, "C1": FAIL, "C2": FAIL}, None),
        PlantedWorld(
            "two-tiles", "A REFUSAL: one row ran a different BLOCK_M. V2 reads "
            "the tile off the rows and fails; a slope across states that ran "
            "different kernels compares kernels, not clocks.",
            plant_rows(eps=0.05, jitter=0.004, block_m_at={(0, 1, 0): 64}),
            {"V0": PASS, "V1": PASS, "V2": FAIL, "V3": PASS, "V4": PASS,
             "V5": PASS, "V6": PASS, "C1": PASS, "C2": PASS}, 0.05),
        PlantedWorld(
            "thin", "A REFUSAL: two repeats and three treads. V3 fails on "
            "depth; the bootstrap over two repeats has three distinct draws and "
            "an interval from it is a decoration.",
            plant_rows(eps=0.05, treads=3, repeats=2),
            {"V0": PASS, "V1": PASS, "V2": PASS, "V3": FAIL, "V4": PASS,
             "V5": PASS, "V6": PASS, "C1": PASS, "C2": PASS}, None),
        PlantedWorld(
            "drifting", "A REFUSAL: a third of the rows drifted. V4 fails; what "
            "is left is a subsample the card chose. The DRIFT rows are the ones "
            "excluded, and the LEVEL rows beside them are NOT.",
            plant_rows(eps=0.05, jitter=0.004,
                       drift_at={(i, t, r) for i in range(n_states)
                                 for t in range(1, DEFAULT_TREADS + 1)
                                 for r in range(DEFAULT_REPEATS)
                                 if (t + r) % 3 == 0}),
            {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": FAIL,
             "V5": PASS, "V6": PASS, "C1": PASS, "C2": PASS}, None),
        PlantedWorld(
            "sagging-burst", "A REFUSAL: the clock held BETWEEN bursts and sagged "
            "20% INSIDE them. Every DRIFT verdict passes and the per-call time "
            "is still an average over two operating points; V5 is the only gate "
            "that can see it.",
            plant_rows(eps=0.05, jitter=0.004,
                       burst_moves_at={(0, 1, 0), (1, 2, 1), (2, 3, 2)}),
            {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
             "V5": FAIL, "V6": PASS, "C1": PASS, "C2": PASS}, None),
        PlantedWorld(
            "memory-clock-moved", "A REFUSAL: the HBM clock tracked the duty "
            "cycle. V6 fails; a slope fitted across states whose memory clock "
            "moved is a blend of issue rate and bandwidth and is not this arm's "
            "quantity.",
            plant_rows(eps=0.05, jitter=0.004,
                       mem_at={(i, t, r): PLANTED_MEM_MHZ + 200.0 * i
                               for i in range(n_states)
                               for t in range(1, DEFAULT_TREADS + 1)
                               for r in range(DEFAULT_REPEATS)}),
            {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
             "V5": PASS, "V6": FAIL, "C1": PASS, "C2": PASS}, None),
        PlantedWorld(
            "level-both-sides", "NOT a refusal, and that is the point. One row "
            "planted LEVEL HIGH, one LEVEL LOW and one DRIFTING, in one state. "
            "Both LEVEL rows are KEPT and the DRIFT row is EXCLUDED; a scorer "
            "that filtered on LEVEL would drop the boosted rows, which at a duty "
            "cycle below 1.0 are the experiment.",
            plant_rows(eps=0.05, jitter=0.004,
                       level_at={(0, 1, 0): T.LEVEL_HIGH, (0, 2, 0): T.LEVEL_LOW},
                       drift_at={(0, 3, 0)}),
            {"V0": PASS, "V1": PASS, "V2": PASS, "V3": PASS, "V4": PASS,
             "V5": PASS, "V6": PASS, "C1": PASS, "C2": PASS}, 0.05),
    ]


def _self_test_args(args):
    """The planted worlds are scored against the PLANTED design, not the CLI's.

    `--repeats 3` on the command line must not move the V1 threshold the planted
    worlds were registered against, or the registration would be a function of
    how the self-test was invoked.
    """
    class _P:
        pass
    p = _P()
    p.duty = list(DUTY_LEVELS)
    p.treads = DEFAULT_TREADS
    p.repeats = DEFAULT_REPEATS
    p.min_clock_ratio = getattr(args, "min_clock_ratio", None)
    return p


def self_test(args) -> int:
    """Plant every world, score it, and name the rows that disagree.

    Five VALIDITY gates over the worlds, not one: an estimator that recovers a
    planted number, a scorer that reproduces a planted verdict, a design that
    resolves, and PROOF THAT EACH CLAIM GATE CAN FAIL are four different
    statements, and one gate over all of them cannot say which broke.
    """
    print("SELF TEST. Eleven planted worlds; six of them are refusals, because "
          "a scorer that\nhas only ever seen a clean design has never been shown "
          "to refuse one.\n")
    planted = _self_test_args(args)
    threshold, source = registered_clock_ratio(planted)
    verdicts_ok = True
    recovered_ok = True
    resolves_ok = True
    seen_values: dict[str, float] = {}
    claim_fails: dict[str, bool] = {"C1": False, "C2": False}
    for world in self_test_worlds(args):
        est = fit(world.rows, draws=args.draws, seed=args.seed)
        gates = gates_for(world.rows, planted, threshold, source, est)
        got = {g.token: g.verdict for g in gates}
        print(f"  {world.name}: {world.why}")
        print(f"    fit {est.value if est.value is None else round(est.value, 4)}"
              f"  interval "
              + (f"[{est.lo:.4f}, {est.hi:.4f}]" if est.lo is not None else "none"))
        for token in sorted(world.expect):
            good = got.get(token) == world.expect[token]
            verdicts_ok = verdicts_ok and good
            if not good:
                print(f"    [FAIL] {token} expected {world.expect[token]} got "
                      f"{got.get(token)}")
        if all(got.get(k) == v for k, v in world.expect.items()):
            print("    [PASS] every registered verdict reproduced")
        for token in claim_fails:
            if got.get(token) == FAIL:
                claim_fails[token] = True
        if world.recovers is not None:
            if est.value is None or abs(est.value - world.recovers) > world.tolerance:
                recovered_ok = False
                print(f"    [FAIL] planted {world.recovers} not recovered "
                      f"({est.value})")
            else:
                print(f"    [PASS] planted {world.recovers} recovered as "
                      f"{est.value:.4f}")
                seen_values[world.name] = est.value
            if est.half_width is not None and est.half_width > RESOLUTION_TARGET:
                resolves_ok = False
        print()

    distinct = len({round(v, 3) for v in seen_values.values()})
    gates = [
        Gate(VALIDITY, "S1", "the scorer reproduces every registered verdict in "
             "every planted world",
             PASS if verdicts_ok else FAIL, "every planted row above",
             "all rows agree",
             "everything this file computes: a scorer that cannot tell a "
             "one-clock design from a separated one cannot gate a session"),
        Gate(VALIDITY, "S2", "the estimator RECOVERS the planted elasticity",
             PASS if recovered_ok else FAIL,
             f"{len(seen_values)} worlds with a planted value",
             "within the world's stated tolerance",
             "the claim: an estimator never run against a known answer is an "
             "assertion"),
        Gate(VALIDITY, "S3", "and a different planted elasticity gives a "
             "different answer",
             PASS if distinct >= 2 else FAIL,
             f"{distinct} distinct recovered values over "
             f"{len(seen_values)} worlds",
             ">= 2 distinct values",
             "the claim: an estimator that returns one number in every world is "
             "reporting its prior"),
        Gate(VALIDITY, "S4", "the design RESOLVES: the interval is inside the "
             "registered target on the recovery worlds",
             PASS if resolves_ok else FAIL,
             "every recovery world's half-width",
             f"<= {RESOLUTION_TARGET:.3f}",
             "C1, which would then fail on every real run for a reason the "
             "planted worlds could have shown first"),
        Gate(VALIDITY, "S5", "each CLAIM gate FAILS in at least one planted world",
             PASS if all(claim_fails.values()) else FAIL,
             "; ".join(f"{k} reached FAIL: {v}" for k, v in sorted(claim_fails.items())),
             "both C1 and C2 reach FAIL somewhere",
             "the claims: a gate no planted world can fail is satisfiable by a "
             "gate that fails nowhere, which is not the same as one that passes"),
    ]
    for gate in gates:
        for line in gate.render():
            print(line)
        print()
    rc = exit_codes.classify(g.scored() for g in gates)
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


# --------------------------------------------------------------------------
# Plumbing.
# --------------------------------------------------------------------------

def git_check_ignore(path: Path) -> bool | None:
    """Would git silently drop this path. True, False, or None for CANNOT ASK.

    Run rather than reasoned about, and the third value is the point: `git
    check-ignore` returns 128 for a path outside the work tree, which is the POD
    DEFAULT, and reading that as "tracked" is how a gate came to print PASS on a
    machine where it had not been able to ask.
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


def resolve_card(args) -> str:
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
    """Card first, then every knob that changes what is MEASURED.

    IN: the tile, the model, the dtype, the ladder depth, the duty list, the
    repeats, and every timing knob (`--burst-ms`, `--target-ms`, `--trials`,
    `--warm-ms`, `--settle-seconds`, the flush) because each of them changes the
    milliseconds on the row.

    OUT: `--draws`, `--seed-bootstrap` and `--min-clock-ratio`, which re-analyse
    one set of cells, and `--out`, `--run-id`, `--dry-run`, `--self-test`. Two
    analyses of one sweep belong in one directory.
    """
    return PV.run_id(
        card=resolve_card(args),
        model=args.model, dtype=args.dtype,
        tile="_".join(f"{k}{v}" for k, v in sorted(PINNED.items())),
        treads=int(args.treads), duty=[float(d) for d in args.duty],
        repeats=int(args.repeats), burstms=float(args.burst_ms),
        targetms=float(args.target_ms), trials=int(args.trials),
        warmms=float(args.warm_ms), settle=float(args.settle_seconds),
        l2flush=not bool(args.no_l2_flush), seed=int(args.seed))


def results_root() -> Path:
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return Path(__file__).resolve().parents[1] / "results"


def report_lines(rows, est: Elasticity, args) -> list[str]:
    keep = kept_rows(rows)
    cells = collapse(keep)
    sides = level_counts(keep)
    out = ["", "THE STATES, as measured"]
    out.append("  duty   rows   median clock   median power   median duty seen")
    for duty in args.duty:
        mine = [r for r in keep if _duty_key(r.duty_requested) == _duty_key(duty)]
        if not mine:
            out.append(f"  {duty:4.2f}      0   (no kept row)")
            continue
        out.append(
            f"  {duty:4.2f}   {len(mine):4d}   "
            f"{statistics.median(r.sm_clock_load_mhz for r in mine):9.1f} MHz   "
            f"{statistics.median(r.power_w or 0.0 for r in mine):9.1f} W   "
            f"{statistics.median(r.duty_achieved for r in mine):13.3f}")
    ratios = clock_ratio_by_tread(cells)
    if ratios:
        out.append("  clock span per tread: "
                   + ", ".join(f"n{t}={v:.4f}x" for t, v in sorted(ratios.items())))
    out += [
        "",
        f"  LEVEL among kept rows: {sides[T.LEVEL_HIGH]} HIGH, "
        f"{sides[T.LEVEL_LOW]} LOW, {sides['level']} level, "
        f"{sides['undetermined']} undetermined. NEITHER SIDE EXCLUDES ANYTHING "
        "HERE.",
        "",
        "THE FIT",
        "  eta = -d log ms / d log f, fixed tread "
        + (f"{est.value:.4f}" if est.value is not None else "no slope"),
        f"  95% interval over {est.repeats} repeats     "
        + (f"[{est.lo:.4f}, {est.hi:.4f}]  half-width {est.half_width:.4f}"
           if est.lo is not None else "none"),
        f"  bootstrap draws                     {est.resampled} of {est.draws}",
        f"  cells in the fit                    {est.cells} "
        f"({est.treads} treads x {est.states} states)",
        "  per tread                           "
        + ", ".join(f"n{t}={v:+.4f}" for t, v in sorted(est.per_tread.items())),
        "",
        "  COMPANION, not the claim: the elasticity of the ladder SLOPE, the "
        "per-M-tile cost",
        "  eta of the per-M-tile cost           "
        + (f"{est.ladder_slope:.4f} over {est.ladder_states} states"
           if est.ladder_slope is not None else "not determined"),
        "  The clock is not constant across the treads of one state, so the "
        "'clock of a slope'",
        "  is a summary and this number is read as one. The claim is the "
        "fixed-tread quantity.",
    ]
    return out


def report_tail(body: list[str], gates: list[Gate]) -> list[str]:
    """Everything after the header: the states, the fit, the gates, the reading.

    RETURNED AS LINES so stdout and report.txt are built from ONE list rather
    than from a string sliced by the header's length. The sliced version worked
    only while every header line happened to be one line, and a single embedded
    newline anywhere in the plan would have silently shifted what the pod's
    operator saw against what the artefact holds.
    """
    out = list(body) + ["", "=" * 78, "GATES"]
    for gate in gates:
        out += gate.render()
        out.append("")
    if any(g.kind == VALIDITY and g.verdict != PASS for g in gates):
        out.append("READING IT. A VALIDITY gate did not pass. No number on this "
                   "page may be quoted and no band is licensed; fix the "
                   "apparatus and re-run.")
    else:
        failed = [g.token for g in gates if g.kind == CLAIM and g.verdict != PASS]
        out.append(f"READING IT. Validity holds. Claim gates not passed: "
                   f"{failed or 'none'}.")
        if failed:
            out.append("A failed CLAIM gate is a result, not a broken run. Read "
                       "the C1 line for whether the design resolved the three "
                       "registered worlds, and the C2 line for the registered "
                       "consequence of the band it landed in.")
        else:
            out.append("The interval lies wholly below the RAW-STANDS edge: at "
                       "this cell the measured millisecond is traffic, not "
                       "issue rate, and the clock is not what is wrong with "
                       "alpha.")
    return out


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the registered bands, the design "
                         "arithmetic and the cost, then stop. Scores no gate, "
                         f"so it exits {exit_codes.REFUSED} REFUSED; "
                         "--self-test is the off-GPU mode that DOES score")
    ap.add_argument("--self-test", action="store_true",
                    help="score eleven planted worlds, six of them refusals, "
                         "and check the scorer reproduces each registered "
                         "verdict and the estimator recovers each planted "
                         "elasticity. Needs no GPU")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--dtype", default=DEFAULT_DTYPE,
                    choices=sorted(DTYPE_BYTES),
                    help="constrained here rather than discovered inside "
                         "make_inputs: a dtype this repository does not know is "
                         "an argparse 2 before the weights are drawn, not a "
                         "traceback after 2.82 GB of them")
    ap.add_argument("--treads", type=int, default=DEFAULT_TREADS,
                    help="exactly-full tile stacks on the ladder, r = n x "
                         f"BLOCK_M at BLOCK_M={PINNED['BLOCK_SIZE_M']}")
    ap.add_argument("--duty", type=float, nargs="+", default=list(DUTY_LEVELS),
                    help="sustained GPU-busy fractions, one state each. 1.0 is "
                         "no host sleep at all, the highest sustained "
                         "pressure a byte-identical kernel can apply")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                    help="passes over the whole design; the interval is a "
                         "bootstrap over these")
    ap.add_argument("--burst-ms", type=float, default=DEFAULT_BURST_MS,
                    help="target GPU ms inside one burst, which sets the calls "
                         "per burst and hence the cost of the discarded lead")
    ap.add_argument("--target-ms", type=float, default=DEFAULT_TARGET_MS,
                    help="kernel ms of kept calls per trial; timing.iters_for's "
                         "budget, so a cell here holds what a ladder cell holds")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--warm-ms", type=float, default=DEFAULT_WARM_MS,
                    help="delivered GPU load before a cell is timed, run AT THE "
                         "STATE'S OWN DUTY and settled on the clock")
    ap.add_argument("--settle-seconds", type=float, default=DEFAULT_SETTLE_SECONDS,
                    help="seconds of the state's cadence before its first cell")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="do not flush L2 before each call. Changes the "
                         "milliseconds, so it is in the run id")
    ap.add_argument("--seed", type=int, default=0,
                    help="routing and weight seed; in the run id")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                    help="bootstrap draws. Re-analysis of one set of cells, so "
                         "it is NOT in the run id")
    ap.add_argument("--seed-bootstrap", type=int, default=0,
                    help="seed for the bootstrap. Re-analysis, not in the run id")
    ap.add_argument("--min-clock-ratio", type=float, default=None,
                    help="override V1's threshold. Without it the threshold is "
                         "COMPUTED from this run's repeats, treads and states")
    ap.add_argument("--card", default=None,
                    help="the card this run is about, as nvidia-smi spells it. "
                         "Defaults to the live device; --dry-run and "
                         f"--self-test touch no GPU and are labelled {NO_CARD!r}")
    ap.add_argument("--out", type=Path, default=None,
                    help="overrides the results root entirely")
    ap.add_argument("--run-id", default="")
    return ap


def _main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.seed_bootstrap = int(args.seed_bootstrap)
    if args.self_test:
        return self_test(args)

    # THE DESIGN REFUSALS COME FIRST, BEFORE THE PLAN IS EVEN BUILT. They cost
    # nothing, they are decided from the flags alone, and a plan page for a
    # design that cannot exist is not a plan: `plan_lines` prints the token
    # count of the shallowest tread, and `tokens_for_rows` RAISES when the
    # routing cannot form one, so an illegal --model used to leave this file
    # with a traceback and exit 4 where the honest answer is a refusal and 2.
    if len(args.duty) < MIN_STATES:
        print(f"REFUSED before any GPU time. {len(args.duty)} duty states is "
              f"below the {MIN_STATES} this design needs: two states give a "
              "slope with no degrees of freedom and no way to see curvature.")
        return exit_codes.REFUSED
    if len(set(args.duty)) != len(args.duty):
        print("REFUSED before any GPU time. Two --duty values are the same, "
              "so two 'states' are one state wearing two labels and V1 would "
              "score a separation that is not there.")
        return exit_codes.REFUSED
    outside = [d for d in args.duty if not 0.0 < d <= 1.0]
    if outside:
        # AND NOT CLAMPED. `time_duty` takes max(0, gap), so a duty above 1
        # would silently become duty 1 and the run would carry two states at one
        # cadence under two labels -- which is the previous refusal wearing a
        # disguise the ledger cannot see.
        print(f"REFUSED before any GPU time. --duty {outside} is outside "
              "(0, 1]. A duty is a fraction of wall time the GPU is busy; 1.0 "
              "is no host sleep at all and is the most sustained pressure a "
              "byte-identical kernel can apply.")
        return exit_codes.REFUSED
    cfg = MODEL_CONFIGS[args.model]
    quantum = SWEEP.rows_quantum(cfg)
    bad = [n for n in range(1, args.treads + 1)
           if (n * PINNED["BLOCK_SIZE_M"]) % quantum]
    if bad:
        print(f"REFUSED before any GPU time. {args.model} at E="
              f"{cfg.num_experts} k={cfg.top_k} needs rows per expert to be a "
              f"multiple of {quantum}, and treads {bad} are not. A nudged row "
              "is not a full tile stack and this arm reads full stacks only.")
        return exit_codes.REFUSED

    card = resolve_card(args)
    run_id = args.run_id or default_run_id(args)
    out_dir = (args.out or results_root()) / "clock_elasticity" / run_id
    paths = {"cells.csv": out_dir / "cells.csv",
             "report.txt": out_dir / "report.txt",
             "report.json": out_dir / "report.json"}

    header = plan_lines(args, run_id, out_dir, card, paths) + ["", "=" * 78]
    header += prediction_lines(args) + [""] + mde_lines(args) + ["", "=" * 78]
    print("\n".join(header))

    if args.dry_run:
        total, items = estimated_seconds(args)
        print(f"\nestimated wall time {total:.0f} s ({total / 60:.1f} min), "
              "itemised:")
        for line in items:
            print(line)
        print(f"  rows                          "
              f"{args.repeats * len(args.duty) * args.treads} "
              f"({args.repeats} repeats x {len(args.duty)} states x "
              f"{args.treads} treads)")
        print("  EVERY TERM IS ON THE WALL CLOCK. This arm holds a cadence for "
              "a stated number of")
        print("  seconds per state and the gaps are the experiment, so there is "
              "no kernel-time")
        print("  figure to quote instead; the weight build and the first "
              "compile are charged above.")
        print("\n".join(["", "=" * 78,
                         "REFUSED. Nothing was measured and nothing was written.",
                         "  reason: --dry-run was given",
                         "  No gate was scored, so no RESULT line was printed "
                         "and none of the above",
                         "  is a result. --self-test scores the planted worlds "
                         "off GPU; the bare",
                         "  command with --card measures this card.",
                         "=" * 78]))
        return exit_codes.REFUSED

    if card == NO_CARD:
        print("\nREFUSED. Nothing was measured.")
        print("  This run would MEASURE and no card was named. The run id and "
              "the output directory")
        print("  are per-card, and an elasticity is a property of one piece of "
              "silicon's governor.")
        return exit_codes.REFUSED

    try:
        T.require_cuda()
    except Exception as exc:                            # noqa: BLE001
        print(f"\nREFUSED. Nothing was measured: {exc}")
        print("  " + SWEEP.missing_gpu_stack())
        return exit_codes.REFUSED
    try:
        T.nvml_clock_reader()
    except T.ClockSourceUnavailable as e:
        print("\nREFUSED. Nothing was measured.")
        print(f"  The under-load clock sampler could not be opened: {e}")
        print("  This arm's independent variable IS the clock. Without a "
              "reader there is no x")
        print("  axis, and a run without one would buy a ladder nobody can "
              "regress. Install")
        print("  nvidia-ml-py in THIS interpreter and re-run.")
        return exit_codes.REFUSED

    prov = PV.provenance_block(instrument=INSTRUMENT, warmup_ms=args.warm_ms,
                               target_ms=args.target_ms)
    out_dir.mkdir(parents=True, exist_ok=True)
    # THE CARD FILE IS THE RESUME GUARD THE RUN ID CANNOT BE. The id is keyed on
    # the card, so two cards cannot share a directory by accident -- but --out
    # and --run-id are both operator-supplied and both bypass it, and an
    # elasticity pooled over two pieces of silicon's governors is not an
    # elasticity.
    stamp = out_dir / "CARD"
    if stamp.exists() and stamp.read_text().strip() != card:
        print(f"REFUSED. {out_dir} already holds cells measured on "
              f"{stamp.read_text().strip()!r} and this run is on {card!r}. An "
              "elasticity pooled over two cards' governors is not an "
              "elasticity; use a fresh --out or --run-id.")
        return exit_codes.REFUSED
    stamp.write_text(card + "\n")
    rows = run_arm(args, cfg, paths["cells.csv"], out_dir / "triton-cache", prov)

    est = fit(rows, draws=args.draws, seed=args.seed_bootstrap)
    threshold, source = registered_clock_ratio(args)
    gates = gates_for(rows, args, threshold, source, est)
    tail = report_tail(report_lines(rows, est, args), gates)
    print("\n".join(tail))

    (out_dir / "report.txt").write_text("\n".join(header + tail) + "\n")
    payload = prov.stamp({
        "run_id": run_id, "card": card,
        "pinned": dict(PINNED), "duty": list(args.duty),
        "predictions": [asdict(p) for p in PREDICTIONS],
        "bands": [{"name": n, "lo": lo, "hi": hi, "consequence": c}
                  for n, lo, hi, c in BANDS],
        "v1_threshold": threshold, "v1_threshold_source": source,
        "elasticity": asdict(est),
        "gates": [asdict(g) for g in gates],
        "rows": len(rows), "kept": len(kept_rows(rows)),
    })
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\ncells    {paths['cells.csv']}   {git_visibility(paths['cells.csv'])}")
    print(f"report   {paths['report.txt']}")
    print(f"json     {paths['report.json']}")

    rc = exit_codes.classify(g.scored() for g in gates)
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


def main(argv=None) -> int:
    """AN UNPLANNED CRASH IS ERROR (4), which is the only retryable code.

    Left to propagate, an unexpected exception exits the interpreter ONE, and
    ONE is CLAIM_FAIL, which the shared table defines as a RESULT: the driver
    would file a crashed arm as "the elasticity is above the registered edge"
    and retract C3's direction over a run that never measured. `SystemExit`
    passes through untouched, so `SchemaCollision` and the sweep's own refusals
    keep their own codes.
    """
    try:
        return _main(argv)
    except SystemExit as e:
        # THE MESSAGE IS PRINTED HERE AND NOT LEFT TO THE INTERPRETER. A
        # `SystemExit` subclass that sets a class-level `code` -- which is how
        # this repository carries an exit code on a typed refusal -- SHADOWS the
        # slot the interpreter would have printed the message from, so the
        # remedy in the message would otherwise vanish and the operator would
        # get a bare exit 2.
        message = " ".join(str(a) for a in e.args if a)
        if message:
            print(message, file=sys.stderr)
        return (exit_codes.REFUSED if isinstance(e.code, str)
                else int(e.code or 0))
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        print("ERROR: clock_elasticity crashed before it could reach a verdict. "
              "This is the apparatus failing, not a finding about the card, so "
              f"it exits {exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}.",
              file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
