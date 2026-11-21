#!/usr/bin/env python
"""What does fixing the ruler do to every number already published against it?

    python scripts/ruler_rebaseline.py --dry-run      # plan + predictions, no GPU
    python scripts/ruler_rebaseline.py --corpus-only  # the shift, off GPU, hermetic
    python scripts/ruler_rebaseline.py                # the pod run: measure, then price

WHY THIS IS A SEPARATE SCRIPT AND NOT A COMMIT. `docs/STUDY.md:506-510` carries a
follow-up inside a DONE strikethrough: settle the clocks before the bandwidth
patterns, and stop naming a tree reduction as the read ceiling. Both change the
denominator of every efficiency column in `results/published/`, so they are a
re-baseline and not a patch, and a re-baseline that lands without a price tag is
indistinguishable from a silent one. This script is the price tag.

WHAT THE READ OF THE TREE FOUND, and it is not what the follow-up said.

  1. THE BANDWIDTH SIDE OF THE CLOCK PROBLEM WAS ALREADY FIXED, on 2026-08-27 in
     `1199139`, and the fix works: all eleven committed calibrations record
     1980 -> 1980 MHz on every bandwidth pattern, on both cards. What was NOT
     fixed is the REPORT. `scripts/calibrate_hardware.py` printed only the
     top-level `settle` dict, and the memory settle lives under
     `settle["bandwidth_settle"]`, so the 2026-09-01 session log reads

         settle    reached at 1500 MHz   history [1500, 1470, 1515, ...]
         clocks    1500 -> 1980 MHz

     which is exactly the signature of a calibration measured mid-ramp. STUDY.md
     still lists the work as outstanding on the strength of that output, five
     days after it was done. A measurement nobody can see in the report is not
     evidence, so `Calibration.settle_lines` now renders both, each labelled
     with the measurement it governs.

  2. THE CLOCK THAT IS STILL WRONG IS THE GEMM'S. `measure_bf16_gemm` sampled it
     ONCE, after `time_eager` had already synchronised, i.e. with the GPU idle
     and climbing back to its 1980 MHz idle boost. Across the eleven committed
     calibrations of ONE H200 that field reads

         1485 1500 1515 1530 1530 1560 1560 1845 1845 1905 1935  MHz

     a 30% spread, while the achieved rate moved 12% and the compute settle in
     the same files plateaued at 1455-1515. `sustained_peak_tflops` is linear in
     it, so the same card and the same cuBLAS kernel are published at 87.4%
     efficiency in one arm and 68.4% in another. `clock_under_load` now samples
     with work in flight and records the spread; the old post-hoc sample is kept
     beside it so the size of the artefact stays auditable.

  3. THE READ PATTERN IS RENAMED, not merely re-noted. It was `read`, described
     as "the closest analogue to streaming expert weights", and measured with
     `torch.sum`. A reduction bounds the read rate from BELOW. It is now
     `read_reduce`, and `read_stream` -- a Triton kernel with no cross-CTA
     combine, in `moe/bench/read_probe.py` -- is measured beside it to find out
     how much of the figure is still shape.

THE DENOMINATOR CHOICE, MADE RATHER THAN INHERITED. `calibrate.py` used triad as
the ceiling while calling read the closer analogue, and never reconciled the
two. The traffic settles it: an MoE layer at decode is ~98.5% reads, triad is
67% reads, and using triad puts the H200 ridge at 162.8 where the matched ruler
puts it at 159.4. A higher ridge classifies more cells as memory bound and
pushes crossings to larger batch, which is the direction of this study's own
headline. So the ruler in use flatters the conclusion, by 2.2%.

That is an argument for adopting the read ruler, and it is NOT an argument for
adopting it quietly, which is what this script exists to prevent. It prices two
levers separately and refuses to pool them, because they are not the same
question:

  LEVER 1, THE DENOMINATOR. Within each arm, hold the session fixed and swing
  the ceiling from that arm's own triad to that arm's own read pattern. Nothing
  else moves. This is what "adopting the matched ruler" costs.

  LEVER 2, THE SESSION. Hold the ruler's NAME fixed and swap the arm's whole
  calibration for the current committed one for that device. Both terms move at
  once, which is why it is reported separately: pooling it with lever 1 would
  average a 2% denominator question together with a 10% numerator one and report
  a single meaningless number.

WHAT IT WRITES. Everything lands under `$MOE_RESULTS_DIR`, or `/workspace/results`
when that exists, or `<repo>/results`:

    <results>/ruler_rebaseline/<run-id>/report.txt   exactly what was printed
    <results>/ruler_rebaseline/<run-id>/report.json  gates and per-arm shifts
    <results>/ruler_rebaseline/<run-id>/measured.yaml the new calibration, if any

`git check-ignore` is run on every one of them and the answer is printed, because
`results/*` is ignored with only `!results/published/` excepted and this project
has already lost a set of figures to exactly that rule.

OFF GPU. `--dry-run` prints the plan, the registered predictions and the cost.
`--corpus-only` runs the entire published-row comparison, which reads committed
CSVs and committed yaml and nothing else, so it is byte-identical on any machine
and never touches the device. Both need `torch` IMPORTABLE, because
`moe.bench.calibrate` imports it for the measurement half of the module, and
neither needs a CUDA device. Both are the reviewable half.

THE TWO OFF-GPU MODES END IN DIFFERENT CODES ON PURPOSE. `--corpus-only` SCORES
gates 4 and 5 over the committed rows, prints their RESULT lines and exits
whatever `exit_codes.classify` makes of them, which on the shipped corpus is
CLAIM_FAIL. `--dry-run` scores nothing, prints no RESULT line and exits REFUSED
(2); it returned the bare literal 0 until 2026-09-02, one paragraph after
printing "Nothing was measured", and DONE in the shared table means "measured;
every VALIDITY and CLAIM gate PASSED".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench import calibrate as calibrate_mod  # noqa: E402
from moe.bench import exit_codes  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench.calibrate import (  # noqa: E402
    DEFAULT_CEILING,
    DISOWNED,
    MATCHED_CEILING,
    READ_PATTERNS,
    ridge_flop_per_byte,
)
from moe.bench.published import (  # noqa: E402
    CEILINGS_DISAGREE,
    calibration_provenance,
    filter_superseded,
    superseded_impls,
)


def instrument_name() -> str:
    """What actually timed the ceilings this run publishes, named not assumed.

    THE SAME DISCIPLINE `scripts/calibrate_hardware.py` ALREADY HAS, and it is
    stated twice because `scripts/` is not a package and there is nothing to
    import it from. The two must return the same string;
    `tests/test_ruler_rebaseline.py` loads that file and asserts the equality,
    so a change to one is a failing test rather than a silent divergence.

    Until 2026-09-02 this file stamped `instrument=timing.TIMING_BASIS` onto a
    calibration measured by `moe.bench.calibrate.calibrate`, which times through
    `timing.time_eager` at `calibrate.py:718, :901, :952` and has never called
    `time_kernel`. The label named the ladder instrument on a number the ladder
    instrument never touched, and this is the script whose whole purpose is to
    re-baseline the ruler: with `--write-calibration` it writes the ceiling YAML
    that becomes the denominator of every published efficiency column, and the
    instrument label is the one field a downstream reader uses to decide whether
    a cell is comparable with the roof. Putting the roof and the ladders under
    one label is the confusion the audit measured at 12-16% in alpha. A
    provenance field that names the wrong instrument is worse than an absent
    one, because an absent one is read as a question and a wrong one is not
    read at all.

    Reads a `TIMING_BASIS` from `moe.bench.calibrate` when that module grows one
    (its migration onto `timing.time_kernel` is a separate phase), and otherwise
    says plainly which loop ran.
    """
    basis = getattr(calibrate_mod, "TIMING_BASIS", None)
    if basis:
        return str(basis)
    return ("moe.bench.calibrate via timing.time_eager (queue-deep, pre-primed "
            "events, L2 flush between iterations); NOT timing.TIMING_BASIS")


# --------------------------------------------------------------------------
# The numbers this script is arguing about, stated before anything runs.
# Every one is quoted from a committed file, and the file is named, so a reader
# can check the prior without trusting this docstring.
# --------------------------------------------------------------------------

#: `results/published/*/measured.yaml`, `detail.gemm_clock_mhz`, eleven files.
#: The post-hoc idle sample, and the reason prediction 1 exists.
POSTHOC_GEMM_CLOCKS_MHZ = (1485, 1500, 1515, 1530, 1530, 1560, 1560, 1845,
                           1845, 1905, 1935)

#: `detail.settle.clock_history_mhz` in the same files: where the COMPUTE settle
#: actually plateaued. A GEMM clock sampled properly has to land here.
COMPUTE_SETTLE_BAND_MHZ = (1455, 1515)

#: `moe/bench/hardware/measured_nvidia_h200.yaml`, 2026-09-09. What prediction 2
#: says a re-measurement must reproduce, and prediction 3 measures against.
#: Re-registered from the 2026-09-09 calibration, which is the first one to
#: carry `read_stream` under its own name rather than the `read` it was
#: renamed from on 2026-09-02. The 2026-09-02 figures were read_reduce 4469.6,
#: copy 4300.7, triad 4374.8, write 4682.4: every one of them within 0.05% of
#: what is registered now, which is the reproducibility this gate exists to
#: check and is why the bandwidth side of the ruler is the trusted half.
#: FOUR patterns, and `read_stream` is deliberately not one of them: gate 3
#: is the read-shape comparison and reads read_stream against read_reduce, so
#: registering read_stream here would make gate 2 re-judge the number gate 3
#: exists to judge, against a tolerance chosen for reproducibility rather
#: than for shape.
H200_PATTERNS_GBPS = {"read_reduce": 4471.4, "copy": 4300.8,
                      "triad": 4374.5, "write": 4680.2}

#: 3201 MHz x 2 for DDR x 6016 bits / 8. Nothing can exceed this, which is what
#: makes it the guard on a probe that might have had its loads deleted. 6016 is
#: the ENABLED bus NVML reports on the 141 GB part (6144 x 141/144); this
#: constant read 4916.7 off the unharvested 6144 until 2026-09-09, 2.1% high on
#: the one number in this file that is a hard physical bound.
H200_PIN_RATE_GBPS = 4814.3

#: WHICH CARD THE TWO CONSTANTS ABOVE BELONG TO. `_device_key` normalisation of
#: "NVIDIA H200", so the guard matches the same way the corpus survey does.
H200_DEVICE_KEY = "nvidiah200"

#: Theoretical pin rates, per card, from the committed calibrations'
#: `observed.pin_rate_gbps`: 4814.3 for the H200 and 2039.0 for the A100
#: (`moe/bench/hardware/measured_nvidia_h200.yaml` and
#: `measured_nvidia_a100_sxm4_80gb.yaml`). A card that is not in this table gets
#: no pin-rate guard and therefore no read_stream verdict, which is a REFUSAL
#: rather than a default: an elided-loads check run against another card's pin
#: rate is not a loose check, it is an inert one. On the A100 the H200's 4814.3
#: would pass a read_stream reporting 2.4x that card's entire bus.
PIN_RATE_GBPS = {"nvidiah200": H200_PIN_RATE_GBPS,
                 "nvidiaa100sxm480gb": 2039.0}

#: How far a re-measured pattern may sit from the figures above before the
#: bandwidth side is not reproducing. The two closest H200 calibrations differ
#: by 0.06%, so 0.5% is eight times the observed session-to-session spread.
GATE2_PATTERN_TOL_PCT = 0.5

#: Prediction 1's discriminator. The post-hoc sample and the under-load median
#: must differ by more than this on at least one GEMM for the artefact to be
#: real; 5% is the same threshold `clock_ramped` uses, and the observed spread
#: is 30%.
GATE1_CLOCK_DELTA_PCT = 5.0

#: Prediction 4. The numerator's session-to-session spread must be at least this
#: many times the denominator choice, or the re-baseline really is about the
#: denominator after all. 9.9% against 2.2% is 4.5x; 3x leaves room.
#: What the 2026-08-27 read-shape change was worth, as a fraction. Gate 3's
#: effect, and the number the MDE line is scored against.
READ_SHAPE_EFFECT = 0.017

GATE4_SPREAD_RATIO = 3.0

#: Prediction 5. How many published rows may change memory/compute
#: classification when the denominator swings. Zero, and the prior is
#: calibrate.py's own docstring: no cell changes classification anywhere in the
#: 4252.8-4656.9 GB/s range, which brackets every pattern on both cards.
GATE5_MAX_FLIPS = 0

#: Rows whose stored ceiling must agree with their own arm's yaml for the arm to
#: be comparable at all. Same tolerance `published.CEILING_REL_TOL` uses, for
#: the same reason: the value round-trips through TB/s and can land one ULP out.
IDENTITY_REL_TOL = 1e-9

#: Non-vacuity floors. A comparison that examined nothing reports zero flips too,
#: which is indistinguishable from a comparison that examined everything and
#: found none. These are what make gate 5 mean something.
MIN_ARMS = 5
MIN_ROWS = 10_000

#: The run-to-run timing spread this run's fresh calibration is sized against,
#: as a fraction of one measurement. MEASURED, not assumed: the median and worst
#: `timing_spread_median` over the 26 published `*.report.json` files in
#: `results/published`, which is every arm in this repository that records one
#: (min 0.0039, median 0.0077, max 0.0182 on 2026-09-02). The old convention in
#: this repo was to assume 0.5%, which sits below the whole measured range.
MEASURED_SPREAD_MEDIAN = 0.0077
MEASURED_SPREAD_MAX = 0.0182

#: Two-sided 5% at 80% power, the convention every MDE in this study is quoted
#: at, named here rather than inlined so a reader can see that nothing was
#: chosen to make a gate pass.
MDE_LEVEL = 0.05
MDE_POWER = 0.80

#: Trials behind one bandwidth pattern, from `calibrate.measure_bandwidth`'s
#: default. It is not a flag on this script, so the MDE is a property of the
#: apparatus rather than of the invocation, and printing it is the only way a
#: reader learns the design is fixed at this resolution before the pod is rented.
CALIBRATION_TRIALS = 3


@dataclass(frozen=True)
class Prediction:
    """One registered prediction: what it says, and what a FAIL would mean.

    Printed before any measurement and never rewritten from data. The `fail`
    field is the part that makes it a prediction rather than a description: a
    gate whose failure has no stated consequence is a gate nobody has to honour.
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
        1, "the GEMM clock was sampled in the wrong state",
        f"the under-load median lands in {COMPUTE_SETTLE_BAND_MHZ[0]}-"
        f"{COMPUTE_SETTLE_BAND_MHZ[1]} MHz (the compute settle plateau), and "
        f"differs from the same run's post-hoc idle sample by >"
        f"{GATE1_CLOCK_DELTA_PCT:.0f}% on at least one of the two GEMMs",
        "the post-hoc sample was not the artefact. The 87.4%-versus-68.4% "
        "efficiency swing on one card is then unexplained, and no "
        "efficiency-at-clock figure in the study may be quoted."),
    Prediction(
        2, "the bandwidth side is already settled and re-measuring changes it by "
           "nothing",
        "read_reduce 4469.6, copy 4300.7, triad 4374.8, write 4682.4 GB/s, each "
        f"within {GATE2_PATTERN_TOL_PCT}%; every pattern at 1980 -> 1980 MHz",
        "the memory settle does not hold, and every published GB/s is unstable "
        "at the size of the disagreement. The 2026-08-27 settle fix would then "
        "need re-opening, not just re-reporting."),
    Prediction(
        3, "the reduction shape still costs something",
        f"read_stream > read_reduce (4469.6) and < the {H200_PIN_RATE_GBPS} GB/s "
        "pin rate; the 2026-08-27 shape change was worth 1.7%, so 0-5%",
        "above the pin rate, the probe is broken -- dead-code elimination or a "
        "byte count -- and nothing from it may be used. Below read_reduce, ATen "
        "was not the limit and read_reduce stands as the read ruler."),
    Prediction(
        4, "the denominator question is the smaller half of the ruler problem",
        f"the compute term's spread across sessions is >= {GATE4_SPREAD_RATIO}x "
        "the triad-to-read denominator swing (9.9% against 2.2% in the "
        "committed files)",
        "the ruler CHOICE dominates its own instability, the re-baseline is "
        "about the denominator after all, and adopting the matched ruler is "
        "urgent rather than optional."),
    Prediction(
        5, "swinging the denominator changes no published row's bound",
        f"{GATE5_MAX_FLIPS} rows of the canonical pool change memory/compute "
        "classification when the ceiling moves from triad to the matched read "
        "pattern (calibrate.py's own docstring: no cell changes anywhere in "
        "4252.8-4656.9 GB/s)",
        "the 2.2% matters at row level. Every crossing in docs/FINDINGS.md is "
        "then quoted against a ruler that decides its own answer, and the "
        "re-baseline must be adopted before any of them is quoted again."),
)


# --------------------------------------------------------------------------
# Gates. A number against a threshold, PASS or FAIL, and what a FAIL costs.
# --------------------------------------------------------------------------

PASS, FAIL, UNDECIDED = "PASS", "FAIL", "UNDECIDED"

#: A FAIL here means nothing on the page may be quoted.
VALIDITY = "VALIDITY"
#: A FAIL here IS a result.
CLAIM = "CLAIM"


@dataclass
class Gate:
    kind: str
    number: str
    claim: str
    verdict: str
    measured: str
    threshold: str
    lines: list[str] = field(default_factory=list)

    @property
    def token(self) -> str:
        """The gate's identifier as ONE whitespace-free token, for `RESULT:`.

        `kind` and `number` are separate fields here, and a result line's name
        must be one token, so the two are joined: `VALIDITY 3` becomes `V3` and
        `CLAIM 2` becomes `C2`. Joined rather than truncated to the number,
        because this script has both a VALIDITY 3 and a CLAIM 3 and they would
        otherwise be indistinguishable to the driver.
        """
        return f"{self.kind[0]}{self.number}"

    def result_line(self) -> str | None:
        """The one line the session driver may grep for this gate, or None.

        None for an UNDECIDED gate, and the distinction is deliberate. UNDECIDED
        here does not mean "scored and could not decide", which is what
        `exit_codes.UNKNOWN` means and which counts against the gate; it means
        the gate could not RUN on this machine at all -- `--corpus-only` has no
        GPU, so gates 1 to 3 have no measurement to read. Printing UNKNOWN for
        those would classify every corpus-only run as a failed claim, and
        printing PASS would be the vacuous pass this project's rules forbid.
        They stay in the human table below, named and marked, and the closing
        summary says how many were never scored.
        """
        if self.verdict == UNDECIDED:
            return None
        return exit_codes.result_line(
            exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
            self.token, self.verdict,
            f"{self.claim}: measured {self.measured}")

    def scored(self) -> tuple[str, str, str] | None:
        """`(kind, name, verdict)` for `exit_codes.classify`, or None if the
        gate never ran. Same rule as `result_line`, so the log and the exit
        code are computed from exactly the same set of gates."""
        if self.verdict == UNDECIDED:
            return None
        return (exit_codes.VALIDITY if self.kind == VALIDITY else exit_codes.CLAIM,
                self.token, self.verdict)

    def render(self) -> list[str]:
        line = self.result_line()
        out = [line] if line else []
        out += [f"{self.kind} {self.number}  {self.verdict:9s} {self.claim}",
                f"{'':>11}measured {self.measured}   gate {self.threshold}"]
        out += [f"{'':>11}{line}" for line in self.lines]
        return out


# --------------------------------------------------------------------------
# Reading a calibration file and the rows beside it. Pure I/O plus arithmetic;
# no torch, no GPU, identical on every machine.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Ruler:
    """One calibration's ceilings, in the units the yaml and the CSV both use."""
    path: str
    name: str
    gpu_name: str
    checked_on: str
    #: `memory.bandwidth_tb_s * 1000`: the figure the rows were stamped with.
    bandwidth_gbps: float
    ceiling_pattern: str
    #: Every pattern in `detail`, by name. Empty for the oldest arms, which is a
    #: refusal condition and not a zero.
    patterns: dict[str, float]
    #: Patterns whose note disowns them. Kept so a band is never built out of a
    #: figure the calibration itself rejected.
    disowned: frozenset[str]
    peak_tflops: dict[str, float]
    gemm_clock_mhz: int

    def ridge(self, dtype: str, bandwidth_gbps: float | None = None) -> float | None:
        return ridge_flop_per_byte(self.peak_tflops.get(dtype),
                                   bandwidth_gbps or self.bandwidth_gbps)

    def matched_read(self) -> tuple[str, float] | None:
        """The best read pattern this file offers, or None.

        None rather than a fallback to triad: "this calibration has no read
        ruler" is the answer for the two 2026-08-22 arms and for the A100, whose
        `read_reduce` came in below triad and was disowned on the spot. A silent
        fallback would report a denominator swing of exactly zero for them and
        it would look like a measurement.
        """
        for name in READ_PATTERNS:
            if name in self.patterns and name not in self.disowned:
                return name, self.patterns[name]
        # The pre-rename name. Published arms carry `read`, and refusing to read
        # them would make the whole corpus comparison empty.
        if "read" in self.patterns and "read" not in self.disowned:
            return "read", self.patterns["read"]
        return None


def read_ruler(path: Path) -> Ruler:
    """Parse a calibration yaml into a `Ruler`, or raise.

    Raises rather than defaulting on the two fields everything divides by. A
    missing bandwidth here would propagate into every ridge on the page as a
    zero or an infinity, and the whole point of the exercise is that the
    denominator is visible.
    """
    import yaml

    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a mapping")
    detail = doc.get("detail") or {}
    bw_tb_s = ((doc.get("memory") or {}).get("bandwidth_tb_s")) or 0.0
    if not bw_tb_s:
        raise ValueError(f"{path}: no memory.bandwidth_tb_s to divide by")
    patterns, disowned = {}, set()
    for entry in detail.get("bandwidth_patterns") or []:
        name, gbps = str(entry.get("pattern", "")), float(entry.get("gbps") or 0.0)
        if not name or gbps <= 0:
            continue
        patterns[name] = gbps
        if DISOWNED in str(entry.get("note") or ""):
            disowned.add(name)
    peaks = {str(k): float(v)
             for k, v in (doc.get("compute_dense_tflops") or {}).items() if v}
    return Ruler(
        path=str(path), name=str(doc.get("name", "")),
        gpu_name=str(detail.get("gpu_name") or doc.get("name") or ""),
        checked_on=str(doc.get("checked_on") or ""),
        bandwidth_gbps=float(bw_tb_s) * 1000.0,
        ceiling_pattern=str(detail.get("ceiling_pattern") or ""),
        patterns=patterns, disowned=frozenset(disowned), peak_tflops=peaks,
        gemm_clock_mhz=int(detail.get("gemm_clock_mhz") or 0))


def arm_rows(arm: Path) -> list[dict]:
    """The per-venv CSVs, never merged.csv.

    merged.csv is rebuilt from `run_*.csv` and holds every row a second time.
    Reading both is the double-counting failure `moe/bench/published.py` exists
    to document, and it would inflate every count on this page without changing
    a single ratio, which is the hardest kind of error to notice.
    """
    rows: list[dict] = []
    for path in sorted(arm.glob("run_*.csv")):
        with path.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _f(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------
# The shift itself.
# --------------------------------------------------------------------------

@dataclass
class ArmShift:
    """What one lever does to one arm's rows.

    `unclassifiable` is a first-class count, not a filter. Rows whose dtype has
    no peak in the calibration beside them -- every fp8 row of
    `-fp8-three-kernel`, all 19,908 of which carry `achieved_peak_tflops = 0.0`
    -- have no ridge at all, so they cannot flip and must not be counted as
    "did not flip".
    """
    arm: str
    lever: str
    rows: int = 0
    classified: int = 0
    unclassifiable: int = 0
    flips: int = 0
    flip_examples: list[str] = field(default_factory=list)
    old_bw_gbps: float = 0.0
    new_bw_gbps: float = 0.0
    old_ceiling: str = ""
    new_ceiling: str = ""
    ridges: dict = field(default_factory=dict)
    refusal: str = ""

    @property
    def bw_shift_pct(self) -> float:
        if not self.old_bw_gbps:
            return 0.0
        return (self.new_bw_gbps / self.old_bw_gbps - 1.0) * 100.0

    @property
    def traffic_ratio_shift_pct(self) -> float:
        """`implied_traffic_ratio` is linear in the ceiling, so it moves with it.

        Stated as its own property rather than left implicit because it is the
        one published column whose VALUE changes with the denominator rather
        than just its classification, and a reader looking for "what actually
        moves on the page" should find it named.
        """
        return self.bw_shift_pct


def classify(rows, ruler: Ruler, bandwidth_gbps: float,
             peaks: dict[str, float] | None = None):
    """`(classified, unclassifiable, bounds)` for these rows at this ceiling.

    `bounds` is a list parallel to the classified rows, so two calls with two
    ceilings can be compared row by row rather than by count. Comparing counts
    would let one row crossing up and another crossing down cancel to zero.
    """
    peaks = peaks if peaks is not None else ruler.peak_tflops
    bounds, unclassifiable = [], 0
    for row in rows:
        dtype = str(row.get("dtype", ""))
        ai = _f(row, "arith_intensity_compulsory")
        ridge = ridge_flop_per_byte(peaks.get(dtype), bandwidth_gbps)
        if ridge is None or ai <= 0:
            unclassifiable += 1
            bounds.append(None)
            continue
        bounds.append("compute" if ai >= ridge else "memory")
    return len(rows) - unclassifiable, unclassifiable, bounds


def denominator_shift(arm: Path, ruler: Ruler, rows) -> ArmShift:
    """LEVER 1: same session, same peaks, ceiling swung triad -> matched read.

    The only thing that moves is the divisor, which is the point: this is what
    "adopt the matched ruler" costs, with nothing else varying to average it
    away.
    """
    shift = ArmShift(arm=arm.name, lever="denominator", rows=len(rows),
                     old_bw_gbps=ruler.bandwidth_gbps,
                     old_ceiling=ruler.ceiling_pattern or "unnamed")
    matched = ruler.matched_read()
    if matched is None:
        shift.refusal = (
            "no usable read pattern in this arm's own calibration"
            + (" (its read was disowned as consumer-limited)"
               if ruler.disowned else " (the file records no per-pattern detail)"))
        return shift
    shift.new_ceiling, shift.new_bw_gbps = matched
    _fill(shift, ruler, rows, ruler.bandwidth_gbps, shift.new_bw_gbps,
          ruler.peak_tflops, ruler.peak_tflops)
    return shift


def session_shift(arm: Path, ruler: Ruler, current: Ruler, rows) -> ArmShift:
    """LEVER 2: same ceiling NAME, the whole calibration replaced.

    Reported apart from lever 1 and never pooled with it. Both terms move here
    -- bandwidth by 0.06% and the compute peak by up to 9.9% -- so a single
    averaged "the ruler moved by X" over both levers would be a number about
    nothing.
    """
    shift = ArmShift(arm=arm.name, lever="session", rows=len(rows),
                     old_bw_gbps=ruler.bandwidth_gbps,
                     new_bw_gbps=current.bandwidth_gbps,
                     old_ceiling=ruler.ceiling_pattern or "unnamed",
                     new_ceiling=current.ceiling_pattern or "unnamed")
    _fill(shift, ruler, rows, ruler.bandwidth_gbps, current.bandwidth_gbps,
          ruler.peak_tflops, current.peak_tflops)
    return shift


def _fill(shift: ArmShift, ruler: Ruler, rows, old_bw: float, new_bw: float,
          old_peaks: dict, new_peaks: dict) -> None:
    """Classify under both rulers and count the rows that actually crossed."""
    classified, unclassifiable, before = classify(rows, ruler, old_bw, old_peaks)
    _, _, after = classify(rows, ruler, new_bw, new_peaks)
    shift.classified, shift.unclassifiable = classified, unclassifiable
    for row, was, now in zip(rows, before, after, strict=True):
        if was is None or now is None or was == now:
            continue
        shift.flips += 1
        if len(shift.flip_examples) < 3:
            shift.flip_examples.append(
                f"{row.get('model', '?')} {row.get('impl', '?')} "
                f"T={row.get('num_tokens', '?')} {row.get('dtype', '?')} "
                f"AI={_f(row, 'arith_intensity_compulsory'):.1f}  {was} -> {now}")
    for dtype in sorted(set(old_peaks) | set(new_peaks)):
        old = ridge_flop_per_byte(old_peaks.get(dtype), old_bw)
        new = ridge_flop_per_byte(new_peaks.get(dtype), new_bw)
        if old and new:
            shift.ridges[dtype] = [round(old, 1), round(new, 1),
                                   round((new / old - 1) * 100, 2)]


# --------------------------------------------------------------------------
# The identity that makes the comparison meaningful at all.
# --------------------------------------------------------------------------

def identity_check(rows, ruler: Ruler) -> tuple[int, int]:
    """`(agreeing, disagreeing)` rows whose stored ceiling is their own file's.

    If a row was stamped from some OTHER calibration, then swinging the pattern
    inside the file beside it prices a ruler the row never used. That is not
    hypothetical: `-h200-whole-layer` is exactly it, documented in
    `moe/bench/published.py`, and it is why claim C5 has a band for a target.
    """
    agree = disagree = 0
    for row in rows:
        stored = _f(row, "achieved_bw_gbps")
        if stored <= 0:
            continue
        if abs(stored - ruler.bandwidth_gbps) <= IDENTITY_REL_TOL * ruler.bandwidth_gbps:
            agree += 1
        else:
            disagree += 1
    return agree, disagree


# --------------------------------------------------------------------------
# The corpus.
# --------------------------------------------------------------------------

@dataclass
class Corpus:
    arms: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    #: Rows removed from an arm that is only PARTIALLY superseded. Its own list
    #: because it is not a refusal: the arm is read, and some of its rows are
    #: not, and collapsing the two would hide which happened.
    trimmed: list[str] = field(default_factory=list)
    rows: int = 0
    denominator: list[ArmShift] = field(default_factory=list)
    session: list[ArmShift] = field(default_factory=list)
    identity_bad: list[str] = field(default_factory=list)
    #: Every distinct ruler seen, keyed `device|bandwidth|bf16`, for
    #: prediction 4. Grouped by device so the spread within one card is never
    #: pooled with the difference between two cards.
    rulers: dict = field(default_factory=dict)
    #: arm name -> normalised device, so gate 4 can match a denominator swing
    #: to the device whose calibrations it belongs beside.
    device_of: dict = field(default_factory=dict)


def survey(published: Path, current_by_device: dict[str, Ruler]) -> Corpus:
    """Read every current published arm and price both levers on it.

    Superseded arms are DROPPED and named. They are still rows that were once
    published, but their ceilings are not what any current analysis divides by,
    and including them would weight the answer toward a directory the project
    has already retired.
    """
    out = Corpus()
    arm_dirs = sorted(p for p in published.iterdir() if p.is_dir())
    kept, dropped = filter_superseded([p / "merged.csv" for p in arm_dirs])
    keep_names = {p.parent.name for p in kept}
    out.dropped = sorted(p.parent.name for p in dropped)

    for arm in arm_dirs:
        if arm.name not in keep_names:
            continue
        yaml_path = arm / "measured.yaml"
        if not yaml_path.exists():
            out.refused.append(f"{arm.name}: no measured.yaml")
            continue
        rows = arm_rows(arm)
        # A PARTIALLY superseded arm keeps its current rows and loses the named
        # implementations. `-fp8-three-kernel` is the case: 10,164 good vLLM and
        # SGLang rows beside two torch spans that timed a quantisation pass by
        # mistake. Counting the retracted ones here would price the re-baseline
        # against rows the project has already withdrawn.
        retired = superseded_impls(arm / "merged.csv") or set()
        if retired:
            before = len(rows)
            rows = [r for r in rows if r.get("impl") not in retired]
            out.trimmed.append(
                f"{arm.name}: dropped {before - len(rows)} rows from superseded "
                f"impls {sorted(retired)}")
        if not rows:
            out.refused.append(f"{arm.name}: no run_*.csv rows")
            continue
        try:
            ruler = read_ruler(yaml_path)
        except (ValueError, KeyError) as exc:
            out.refused.append(f"{arm.name}: {exc}")
            continue

        out.arms.append(arm.name)
        out.rows += len(rows)
        # Keyed by DEVICE first. Two calibrations are only "the same ruler" if
        # they measured the same silicon, and the spread across them is only a
        # ruler-instability number within one device: pooling an A100's
        # 262 TFLOP/s with an H200's 716 reports a 66% "instability" that is
        # just two different cards, which is what the first version of gate 4
        # did.
        device = _device_key(ruler.gpu_name)
        key = (f"{device}|{ruler.bandwidth_gbps:.2f}|"
               f"{ruler.peak_tflops.get('bf16', 0.0):.2f}")
        entry = out.rulers.setdefault(key, {
            "device": device, "gpu_name": ruler.gpu_name,
            "bandwidth_gbps": round(ruler.bandwidth_gbps, 2),
            "bf16_tflops": round(ruler.peak_tflops.get("bf16", 0.0), 2),
            "arms": []})
        entry["arms"].append(arm.name)
        out.device_of[arm.name] = device

        agree, disagree = identity_check(rows, ruler)
        if disagree:
            # NAMED, not silently dropped, and only excused where the project
            # has already declared it. A new instance is a validity failure.
            verdict = calibration_provenance(arm).verdict
            out.identity_bad.append(
                f"{arm.name}: {disagree} of {agree + disagree} rows carry a "
                f"ceiling that is not this file's (provenance: {verdict})")
            if verdict != CEILINGS_DISAGREE:
                continue

        out.denominator.append(denominator_shift(arm, ruler, rows))
        current = current_by_device.get(_device_key(ruler.gpu_name))
        if current is None:
            out.refused.append(
                f"{arm.name}: no current calibration for {ruler.gpu_name!r}; "
                "the session lever cannot be priced for it")
        else:
            out.session.append(session_shift(arm, ruler, current, rows))
    return out


def _device_key(gpu_name: str) -> str:
    """Normalised device name, so 'NVIDIA H200' matches 'measured_nvidia_h200'."""
    return "".join(c for c in gpu_name.lower() if c.isalnum())


def load_current_rulers(hardware_dir: Path) -> dict[str, Ruler]:
    out = {}
    for path in sorted(hardware_dir.glob("measured_*.yaml")):
        try:
            ruler = read_ruler(path)
        except (ValueError, KeyError):
            continue
        out[_device_key(ruler.gpu_name)] = ruler
    return out


def spread_pct(values) -> float:
    """(max - min) / max, in percent. 0.0 for fewer than two values."""
    vals = [v for v in values if v > 0]
    if len(vals) < 2:
        return 0.0
    return (max(vals) - min(vals)) / max(vals) * 100.0


# --------------------------------------------------------------------------
# Gates over the corpus. These need no GPU and are identical on every machine.
# --------------------------------------------------------------------------

def gate_v1_non_vacuity(corpus: Corpus) -> Gate:
    """Did the comparison examine anything.

    A survey that read no rows reports zero classification flips, and zero flips
    is exactly what gate 5 passes on. Without this the strongest-looking result
    on the page is also the one an empty directory would produce.
    """
    ok = (len(corpus.arms) >= MIN_ARMS and corpus.rows >= MIN_ROWS
          and bool(corpus.denominator))
    priced = sum(1 for s in corpus.denominator if not s.refusal)
    return Gate(
        VALIDITY, "1", "the survey examined real rows", PASS if ok else FAIL,
        f"{len(corpus.arms)} arms, {corpus.rows} rows, {priced} priced on the "
        "denominator lever",
        f">= {MIN_ARMS} arms and {MIN_ROWS} rows",
        [] if ok else ["Nothing below examined enough to have found anything. "
                       "Every gate on this page is vacuous."])


def gate_v2_identity(corpus: Corpus) -> Gate:
    """Do the rows carry the ceiling of the file beside them.

    If they do not, this page prices a ruler those rows never used. One arm is
    already known to be in that state and is declared in
    `moe/bench/published.py`; a second one appearing is a validity failure and
    not a curiosity.
    """
    undeclared = [line for line in corpus.identity_bad
                  if f"provenance: {CEILINGS_DISAGREE}" not in line]
    return Gate(
        VALIDITY, "2", "every arm's rows were stamped from its own calibration",
        PASS if not undeclared else FAIL,
        f"{len(corpus.identity_bad)} arms disagree, {len(undeclared)} of them "
        "undeclared", "0 undeclared",
        corpus.identity_bad + ([] if not undeclared else [
            "An undeclared disagreement means the denominator swing below was "
            "priced against a file those rows were never stamped from."]))


def _meant_to_be_kept(path: Path, results_dir: Path) -> bool:
    """Would anyone expect git to track this file.

    A raw run directory under `results/` is ignored ON PURPOSE and its being
    ignored is not a defect. What IS a defect is a path outside that tree, or
    one under `results/published/`, coming back ignored: that is the shape of
    the rule which swallowed every figure of ten published arms, where
    `!results/published/` was excepted and an unanchored `plots/` re-ignored
    what sat inside it.
    """
    parts = path.resolve().parts
    if "published" in parts and "results" in parts:
        return True
    try:
        path.resolve().relative_to(results_dir.resolve())
    except ValueError:
        return True
    return False


def gate_v3_paths(paths: dict[str, Path], results_dir: Path) -> Gate:
    """Is anything being written where .gitignore will silently drop it.

    Run rather than reasoned about, on every path, and the verdict printed for
    each. The gate fails only for a path that is meant to be kept, because
    `results/*` being ignored is the design and reporting it as a failure would
    train the reader to ignore this gate.
    """
    verdicts, bad, unverified = [], [], []
    for label, path in paths.items():
        ignored = git_check_ignore(path)
        keep = _meant_to_be_kept(path, results_dir)
        state = ("UNVERIFIED" if ignored is None
                 else "IGNORED" if ignored else "tracked")
        verdicts.append(
            f"{label}: {path}  {state}"
            f"  ({'meant to be committed' if keep else 'raw output, ignored by design'})")
        if ignored is None:
            # A path git could not be asked about is NOT a path git keeps. On a
            # pod this is the normal case -- /workspace/results is outside the
            # work tree -- and it is reported as unverified for a path meant to
            # be kept rather than passed off as tracked.
            if keep:
                unverified.append(label)
        elif ignored and keep:
            bad.append(label)
    verdict = FAIL if bad else (UNDECIDED if unverified else PASS)
    extra = []
    if bad:
        extra.append(f"{bad} would be written and then silently not committed.")
    if unverified:
        extra.append(
            f"{unverified} could not be checked: git returned neither ignored "
            "nor tracked, which is what it does for a path outside this work "
            "tree (the pod's /workspace/results) or when there is no git "
            "binary. UNDECIDED rather than PASS, because a check that could not "
            "ask reports zero failures too.")
    return Gate(
        VALIDITY, "3", "no output lands where .gitignore drops it",
        verdict,
        f"{len(paths)} paths checked, {len(bad)} would be dropped, "
        f"{len(unverified)} unverifiable",
        "0 dropped that are meant to be kept, and every one of them checkable",
        verdicts + extra)


def gate_4_spread(corpus: Corpus) -> Gate:
    """Prediction 4: the numerator moves more than the denominator choice.

    MATCHED BY DEVICE, and that is not a detail. The first version of this gate
    pooled every calibration in the corpus and reported a 66.6% "compute term
    instability", which was an A100's 262 TFLOP/s sitting beside an H200's 716.
    Two cards are not two measurements of one ruler. The spread is computed
    within each device and the worst one is the gate's number, with every device
    printed so a single-calibration device cannot hide inside an average.
    """
    per_device: dict[str, dict] = {}
    for entry in corpus.rulers.values():
        row = per_device.setdefault(entry["device"],
                                    {"bw": [], "tf": [], "arms": 0})
        row["bw"].append(entry["bandwidth_gbps"])
        row["tf"].append(entry["bf16_tflops"])
        row["arms"] += len(entry["arms"])

    swings = {}
    for shift in corpus.denominator:
        if shift.refusal:
            continue
        device = corpus.device_of.get(shift.arm, "?")
        swings.setdefault(device, []).append(abs(shift.bw_shift_pct))

    lines, worst_ratio, worst_device = [], None, ""
    for device, row in sorted(per_device.items()):
        numerator = spread_pct(row["tf"])
        bw_spread = spread_pct(row["bw"])
        priced = swings.get(device, [])
        denominator = statistics.median(priced) if priced else 0.0
        if len(row["tf"]) < 2 or not denominator:
            lines.append(
                f"{device}: {len(row['tf'])} calibration(s), "
                f"{len(priced)} priced -- not enough to compare, NOT pooled "
                "into another device's answer")
            continue
        ratio = numerator / denominator
        lines.append(
            f"{device}: compute term {numerator:.1f}% over {len(row['tf'])} "
            f"calibrations, bandwidth {bw_spread:.2f}%, denominator swing "
            f"{denominator:.2f}% (median of {len(priced)}), ratio {ratio:.1f}x")
        if worst_ratio is None or ratio < worst_ratio:
            worst_ratio, worst_device = ratio, device

    if worst_ratio is None:
        return Gate(
            CLAIM, "4", "the compute term moves more than the denominator choice",
            UNDECIDED, "no device has two comparable calibrations", "n/a",
            lines + ["Unpairable rather than averaged away."])
    ok = worst_ratio >= GATE4_SPREAD_RATIO
    return Gate(
        CLAIM, "4", "the compute term moves more than the denominator choice",
        PASS if ok else FAIL,
        f"worst device {worst_device} at {worst_ratio:.1f}x",
        f">= {GATE4_SPREAD_RATIO:.0f}x on every device with two calibrations",
        lines + ["A FAIL is the interesting answer: it would mean the ruler "
                 "CHOICE dominates its own instability and the read ruler must "
                 "be adopted now."])


def gate_5_flips(corpus: Corpus) -> Gate:
    """Prediction 5: no published row changes bound under the denominator swing."""
    priced = [s for s in corpus.denominator if not s.refusal]
    flips = sum(s.flips for s in priced)
    classified = sum(s.classified for s in priced)
    unclassifiable = sum(s.unclassifiable for s in priced)
    lines = [f"{classified} rows classified, {unclassifiable} unclassifiable "
             "(no peak for their dtype in the calibration beside them)"]
    for s in priced:
        if s.flips:
            lines.append(f"{s.arm}: {s.flips} of {s.classified}")
            lines += [f"    {e}" for e in s.flip_examples]
    if not classified:
        return Gate(CLAIM, "5", "the denominator swing flips no published row",
                    UNDECIDED, "0 rows classified", f"<= {GATE5_MAX_FLIPS} flips",
                    lines + ["Nothing was classifiable, so nothing was tested."])
    return Gate(
        CLAIM, "5", "the denominator swing flips no published row",
        PASS if flips <= GATE5_MAX_FLIPS else FAIL,
        f"{flips} flips in {classified} classified rows",
        f"<= {GATE5_MAX_FLIPS}",
        lines + ([] if flips <= GATE5_MAX_FLIPS else [
            "A FAIL is the result: the 2.2% decides the answer for these rows, "
            "so the crossings computed from them inherit the ruler's bias."]))


# --------------------------------------------------------------------------
# Gates over a fresh measurement. UNDECIDED without one, never assumed.
# --------------------------------------------------------------------------

def gate_1_clock(cal) -> Gate:
    """Prediction 1: the post-hoc sample was reading an idle GPU."""
    records = [r for r in (cal.gemm_clock, cal.fp8_gemm_clock) if r]
    if not records:
        return Gate(CLAIM, "1", "the GEMM clock was sampled in the wrong state",
                    UNDECIDED, "no under-load samples", "n/a",
                    ["The clock could not be established this run."])
    lines, worst = [], 0.0
    in_band = True
    for rec in records:
        median, idle = rec["median_mhz"], rec["after_idle_mhz"]
        delta = abs(median - idle) / max(idle, 1) * 100.0
        worst = max(worst, delta)
        lo, hi = COMPUTE_SETTLE_BAND_MHZ
        band = lo <= median <= hi
        in_band = in_band and band
        lines.append(
            f"{rec['label']}: under load {median} MHz (spread "
            f"{rec['spread_pct']:.1f}%, samples {rec['samples']}), post-hoc idle "
            f"{idle} MHz, delta {delta:.1f}%"
            + ("" if band else f"  OUTSIDE the {lo}-{hi} settle band"))
    ok = worst > GATE1_CLOCK_DELTA_PCT
    if ok and not in_band:
        lines.append("The artefact reproduces but the under-load median is not "
                     "in the compute settle band, so the new number needs its "
                     "own explanation before it is published.")
    return Gate(
        CLAIM, "1", "the GEMM clock was sampled in the wrong state",
        PASS if ok else FAIL,
        f"largest post-hoc-vs-under-load delta {worst:.1f}%",
        f"> {GATE1_CLOCK_DELTA_PCT:.0f}%", lines)


def gate_2_bandwidth(cal) -> Gate:
    """Prediction 2: re-measuring the patterns reproduces the committed file.

    GUARDED ON THE DEVICE, and the guard is the point. `H200_PATTERNS_GBPS` is
    one card's committed figures; comparing an A100's 1758 GB/s copy against the
    H200's 4300.7 does not fail because the ruler moved, it fails because it is
    the wrong card, and a FAIL that means "wrong card" printed under the claim
    "the bandwidth patterns reproduce" is the study's own stale-H200-constant
    defect reappearing inside the script written to price it. The old
    `if not compared` escape could not catch this: it only fires when the
    pattern NAMES are missing, and they are present on every card.
    """
    device = _device_key(getattr(cal, "gpu_name", "") or "")
    if device != H200_DEVICE_KEY:
        return Gate(CLAIM, "2", "the bandwidth patterns reproduce and did not ramp",
                    UNDECIDED, f"attached device is {cal.gpu_name!r}",
                    f"an H200 (key {H200_DEVICE_KEY})",
                    ["H200_PATTERNS_GBPS is one card's committed figures and "
                     "this is not that card, so there is nothing here to "
                     "reproduce. Re-run this gate on an H200, or commit this "
                     "card's own patterns and key the constants by device.",
                     "This is UNDECIDED and not a FAIL on purpose: a FAIL would "
                     "read as 'the ruler moved' when it means 'wrong card'.",
                     f"clock_ramped={cal.clock_ramped} is measured here and "
                     "stands on its own, but it is not this gate's claim."])
    lines, worst, worst_name = [], 0.0, ""
    compared = 0
    for name, expected in H200_PATTERNS_GBPS.items():
        got = cal.pattern(name)
        if got is None:
            lines.append(f"{name}: not measured this run")
            continue
        compared += 1
        delta = abs(got.gbps - expected) / expected * 100.0
        if delta > worst:
            worst, worst_name = delta, name
        lines.append(f"{name}: {got.gbps:.1f} against {expected:.1f} committed, "
                     f"{delta:+.2f}%   clocks {got.sm_clock_start_mhz} -> "
                     f"{got.sm_clock_end_mhz} MHz")
    if not compared:
        return Gate(CLAIM, "2", "the bandwidth patterns reproduce", UNDECIDED,
                    "no comparable pattern measured", "n/a",
                    lines + ["This is not an H200, or the patterns were renamed "
                             "again. The committed figures are H200 only."])
    ok = worst <= GATE2_PATTERN_TOL_PCT and not cal.clock_ramped
    return Gate(
        CLAIM, "2", "the bandwidth patterns reproduce and did not ramp",
        PASS if ok else FAIL,
        f"worst pattern {worst_name} {worst:.2f}%, clock_ramped={cal.clock_ramped}",
        f"<= {GATE2_PATTERN_TOL_PCT}% and not ramped", lines)


def gate_3_read_shape(cal) -> Gate:
    """Prediction 3: the Triton stream beats the ATen reduction.

    THE PIN RATE COMES FROM THE ATTACHED CARD, not from a constant. The guard's
    whole job is to catch a probe whose loads were elided or whose byte count is
    wrong, and it can only do that against the bus the probe actually ran on.
    Hardcoding the H200's 4916.7 made the guard INERT on an A100, whose bus tops
    out at 2039: any read_stream up to 2.4x that card's entire bandwidth would
    have passed the one check that exists to reject exactly such a number. A
    card with no pin rate on file gets UNDECIDED, because a guard that cannot
    run must not be reported as a guard that passed.
    """
    stream, reduce_ = cal.pattern("read_stream"), cal.pattern("read_reduce")
    device = _device_key(getattr(cal, "gpu_name", "") or "")
    pin = PIN_RATE_GBPS.get(device)
    if reduce_ is None:
        return Gate(CLAIM, "3", "the reduction shape still costs something",
                    UNDECIDED, "no read_reduce measured", "n/a", [])
    if stream is None:
        return Gate(
            CLAIM, "3", "the reduction shape still costs something", UNDECIDED,
            "read_stream was refused", f"> {reduce_.gbps:.1f} GB/s",
            list(cal.refusals) + [
                "The reduction's lower bound stands as the read ruler for this "
                "session. That is a weaker ruler, not a wrong one."])
    gain = (stream.gbps / reduce_.gbps - 1.0) * 100.0
    if pin is None:
        return Gate(
            CLAIM, "3", "the reduction shape still costs something", UNDECIDED,
            f"read_stream over read_reduce {gain:+.2f}%, unguarded",
            "> 0% and < this card's pin rate",
            [f"read_stream {stream.gbps:.1f}, read_reduce {reduce_.gbps:.1f}, "
             f"gain {gain:+.2f}%",
             f"NO PIN RATE ON FILE for {cal.gpu_name!r} (key {device!r}); known: "
             f"{', '.join(sorted(PIN_RATE_GBPS)) or 'none'}.",
             "The elided-loads guard cannot run, so read_stream may not be used "
             "as a read ruler from this session whatever the gain says. Add this "
             "card's observed.pin_rate_gbps to PIN_RATE_GBPS and re-run."])
    over_pin = stream.gbps > pin
    ok = gain > 0 and not over_pin
    lines = [f"read_stream {stream.gbps:.1f}, read_reduce {reduce_.gbps:.1f}, "
             f"gain {gain:+.2f}%",
             f"pin rate {pin} GB/s ({cal.gpu_name}); "
             f"{100 * stream.gbps / pin:.1f}% of it"]
    if over_pin:
        lines.append("ABOVE THE PIN RATE. The probe's loads were elided or its "
                     "byte count is wrong. Nothing from read_stream may be used.")
    elif gain <= 0:
        lines.append("The probe did not beat ATen, so ATen's reduction was not "
                     "the limit and read_reduce is the read ruler.")
    return Gate(
        CLAIM, "3", "the reduction shape still costs something",
        PASS if ok else FAIL, f"read_stream over read_reduce {gain:+.2f}%",
        f"> 0% and < {pin} GB/s, this card's pin rate", lines)


# --------------------------------------------------------------------------
# Plumbing.
# --------------------------------------------------------------------------

def git_check_ignore(path: Path) -> bool | None:
    """Would git silently drop this path. True, False, or None for CANNOT ASK.

    Run rather than reasoned about. The rule that swallowed ten arms' worth of
    figures was an unanchored `plots/`, which nobody predicted by reading the
    file, and `git check-ignore` is the only thing that knows the answer.

    THE THIRD RETURN VALUE IS THE POINT. This used to return `False` -- read
    downstream as "tracked" -- for every outcome that was not a clean rc 0: no
    git binary, a timeout, and rc 128, which is what `git check-ignore` returns
    for a path outside the work tree. That last one is the POD DEFAULT: the
    results root is `$MOE_RESULTS_DIR` or `/workspace/results`, both outside
    this repo. So V3 -- a VALIDITY gate whose FAIL is supposed to mean "no
    number on the page may be quoted" -- computed `ignored=False`, found nothing
    bad, and printed PASS with the line `<path>  tracked`, on a machine where it
    had not been able to ask at all. A gate that cannot distinguish "git says
    this is kept" from "git could not be asked" passes vacuously, which is this
    project's non-vacuity rule and its refuse-rather-than-default rule in one
    function. `replicate_noise_floor.git_accepts` and `bm128_depth.
    git_visibility` both take the same shape.
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


def results_root() -> Path:
    """`$MOE_RESULTS_DIR`, else the network volume, else the repo.

    Same order `scripts/run_all.sh` resolves it in, so this lands beside every
    other arm on the volume that outlives the pod.
    """
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        return Path(env)
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    if workspace.is_dir():
        return workspace / "results"
    return Path(__file__).resolve().parents[1] / "results"


#: The card label a run that measures NOTHING carries. `--corpus-only` prices
#: the published rows of several devices at once and `--dry-run` prices none, so
#: neither has a card, and `provenance.run_id` refuses an id without one. This
#: is that card: a name no `nvidia-smi` can produce, so it can never be confused
#: with a real one, and it sorts its runs together in `ls`.
NO_CARD = "no-card-nothing-measured"


def resolve_card(args) -> str:
    """The card this run is about, or `NO_CARD` when there is not one.

    `--card`, else the live device, else `NO_CARD` for the two modes that touch
    no GPU. `main` refuses to MEASURE under `NO_CARD`: gates 1 to 3 compare a
    fresh calibration against this card's registered constants, and running
    them against a card nobody named is how a stale H200 band ended up in seven
    published A100 reports.
    """
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
    """Derived from EVERY argument that changes what is measured, card FIRST.

    A run id that omits a swept parameter is how two settings come to share a
    directory, and the second then reports the first's numbers under its own
    label. `--gemm-n` and `--buffer-gb` are in here because they change the
    ceilings; `--ceiling` because it changes which pattern the report calls the
    old ruler; `--corpus` because pointing at a different corpus is a different
    experiment with the same code.

    THE CARD WAS THE OMISSION (A5/P5), and it is the worst one this script could
    have had. The output directory is `<results root>/ruler_rebaseline/<id>`,
    the results root is `$MOE_RESULTS_DIR` or `/workspace/results` -- a network
    volume that outlives the pod on purpose -- and both `report.txt` and
    `report.json` are written with `write_text`, which TRUNCATES. So measuring
    the H200 and then the A100 with the same flags did not collide quietly: the
    second run silently overwrote the first's report, and the first card's
    measurement was gone. The card slug now leads the id.

    Built by `moe.bench.provenance.run_id` rather than hashed here: it refuses a
    missing card and a None value, sorts before hashing so the id does not
    depend on the order the knobs were named in, and puts the card at the front
    where `ls` shows it.
    """
    card = resolve_card(args)
    return PV.run_id(
        card=card,
        **{"1mode": "corpus" if args.corpus_only else "measured",
           "2buf": args.buffer_gb, "3gemm": args.gemm_n,
           "4ceil": args.ceiling, "5settle": args.settle,
           "6settlesec": args.settle_seconds, "7corpus": str(args.corpus)})


def estimated_seconds(args) -> float:
    """GPU seconds the measurement will cost, from its own parts.

    Two settles at `--settle-seconds` each, two GEMMs, two clock samples of
    about two seconds each, two bandwidth passes of five patterns, and a Triton
    compile. Deliberately itemised: a single number nobody can decompose is a
    number nobody can check before spending it.
    """
    if args.corpus_only:
        return 0.0
    settles = 2 * (args.settle_seconds if args.settle else 0.0)
    gemms = 2 * 3.0            # 20 iters x 3 trials at 8192^3, both dtypes
    clocks = 2 * 2.5           # clock_under_load: 5 samples x 0.4 s, plus tails
    passes = 2 * 12.0          # five patterns, 30 iters x 3 trials, flushed
    compile_s = 10.0           # first Triton compile of the stream probe
    return settles + gemms + clocks + passes + compile_s


def mde_of_pattern(spread: float, trials: int) -> float:
    """Smallest bandwidth SHIFT this run could resolve, as a fraction.

    The quantity gates 2 and 3 test is a difference between this run's GB/s for
    one pattern and a committed GB/s for the same pattern, and BOTH sides were
    measured, each as a median over `trials` trials, so the difference inherits
    two spreads: hence the sqrt(2). With sigma imported from the published
    corpus rather than estimated inside this run the test is a known-variance z
    test, which is the stricter of the two forms available at three trials.
    `statistics.NormalDist` supplies the quantiles so no distribution code is
    written twice in this repository.
    """
    if trials < 1:
        raise ValueError(f"an MDE needs at least one trial, got {trials}")
    if spread <= 0:
        raise ValueError(f"an MDE needs a positive spread, got {spread}")
    normal = statistics.NormalDist()
    z = normal.inv_cdf(1.0 - MDE_LEVEL / 2.0) + normal.inv_cdf(MDE_POWER)
    return z * spread * math.sqrt(2.0 / trials)


def mde_lines(spreads=None, trials: int = CALIBRATION_TRIALS) -> list[str]:
    """What the fresh calibration could see, printed BEFORE the pod is rented.

    B14: no arm in this study stated a minimum detectable effect, so a gate
    could pass or fail without anyone knowing whether the design could have
    resolved the difference either way. This one matters more than most, because
    two of its three measured gates are stated as TOLERANCES rather than as
    effects: gate 2 passes when every pattern lands within
    `GATE2_PATTERN_TOL_PCT`% of the committed figure, and a tolerance TIGHTER
    than the MDE turns a FAIL into a coin toss dressed as a refutation. Gate 3's
    effect is the 1.7% the 2026-08-27 shape change was worth.

    `spreads` overrides the measured pair, which is how the CANNOT-SEE branch is
    planted in the tests: on today's corpus this design cannot resolve gate 2's
    0.5% at either end, and that is a fact about the apparatus a reader has to
    be told rather than a branch nobody can reach.
    """
    spreads = spreads or (("median", MEASURED_SPREAD_MEDIAN),
                          ("worst", MEASURED_SPREAD_MAX))
    out = [
        "MDE. What a fresh calibration on this design could actually resolve.",
        "  noise assumption: run-to-run timing spread MEASURED over the 26 "
        "published reports,",
        f"                    median {MEASURED_SPREAD_MEDIAN:.2%}, worst "
        f"{MEASURED_SPREAD_MAX:.2%}; {trials} trials per pattern, both sides "
        f"measured.",
        f"  effects under test: gate 2's reproduction tolerance "
        f"{GATE2_PATTERN_TOL_PCT / 100:.2%}, gate 3's read-shape gain "
        f"{READ_SHAPE_EFFECT:.2%}.",
    ]
    for label, spread in spreads:
        mde = mde_of_pattern(spread, trials)
        for name, effect in (("gate 2", GATE2_PATTERN_TOL_PCT / 100.0),
                             ("gate 3", READ_SHAPE_EFFECT)):
            seen = "resolves it" if mde <= effect else "CANNOT resolve it"
            out.append(f"  at the {label:<6} spread {spread:.2%}: MDE "
                       f"{mde:.2%} against {name}'s {effect:.2%}  {seen}")
    out += [
        "  An MDE above the effect does not make a PASS wrong; it makes a FAIL "
        "uninformative.",
        "  The spread is imported from KERNEL arms, which is the only measured "
        "one this repository",
        "  has; a bandwidth pattern's own is plausibly tighter, so these are "
        "upper bounds and a",
        "  design that clears them clears its own. Nobody has measured the "
        "pattern's spread, and",
        "  until somebody does, gate 2's 0.5% tolerance is not known to be "
        "above this run's noise.",
        "  Gates 4 and 5 are arithmetic over committed rows and have no timing "
        "noise to clear.",
    ]
    return out


def missing_gpu_stack() -> str:
    """Which half of the stack is absent, and what to run instead."""
    try:
        import torch
    except ImportError:
        # Reached only if the module import above somehow succeeded without it,
        # which it cannot today: moe.bench.calibrate imports torch. Kept so the
        # message stays right if that ever stops being true.
        return "no torch on this machine; nothing here can run."
    if not torch.cuda.is_available():
        return ("no CUDA device. --corpus-only prices the published rows "
                "without one; --dry-run prints the plan.")
    return ""


# --------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------

def corpus_lines(corpus: Corpus) -> list[str]:
    out = ["", "LEVER 1  DENOMINATOR: this arm's triad -> this arm's read "
                "pattern, nothing else moved",
           f"  {'arm':<52}{'old':>9}{'new':>9}{'shift':>8}{'flips':>7}"
           f"{'rows':>8}"]
    for s in sorted(corpus.denominator, key=lambda s: s.arm):
        if s.refusal:
            out.append(f"  {s.arm:<52}  REFUSED: {s.refusal}")
            continue
        out.append(f"  {s.arm:<52}{s.old_bw_gbps:9.1f}{s.new_bw_gbps:9.1f}"
                   f"{s.bw_shift_pct:+7.2f}%{s.flips:7d}{s.classified:8d}")
        for dtype, (old, new, pct) in sorted(s.ridges.items()):
            out.append(f"  {'':<52}ridge {dtype:<9}{old:7.1f} -> {new:7.1f}"
                       f"  {pct:+.2f}%")

    out += ["", "LEVER 2  SESSION: this arm's whole calibration -> the current "
                "one for its device",
            "  Both terms move here. NOT pooled with lever 1: averaging a 2% "
            "denominator",
            "  question with a 10% numerator one produces a number about "
            "nothing.",
            f"  {'arm':<52}{'bw shift':>10}{'flips':>7}{'rows':>8}"]
    for s in sorted(corpus.session, key=lambda s: s.arm):
        out.append(f"  {s.arm:<52}{s.bw_shift_pct:+9.2f}%{s.flips:7d}"
                   f"{s.classified:8d}")
        for dtype, (old, new, pct) in sorted(s.ridges.items()):
            out.append(f"  {'':<52}ridge {dtype:<9}{old:7.1f} -> {new:7.1f}"
                       f"  {pct:+.2f}%")

    out += ["", "DISTINCT RULERS IN THE CORPUS, grouped by device because two "
                "cards are not",
            "two measurements of one ruler  (bandwidth GB/s, bf16 TFLOP/s)"]
    for key in sorted(corpus.rulers, key=lambda k: (corpus.rulers[k]["device"], k)):
        entry = corpus.rulers[key]
        out.append(f"  {entry['gpu_name']:<26}{entry['bandwidth_gbps']:9.1f}  "
                   f"{entry['bf16_tflops']:7.1f}   "
                   f"{', '.join(sorted(entry['arms']))}")
    if corpus.dropped:
        out += ["", f"DROPPED as superseded: {', '.join(corpus.dropped)}"]
    if corpus.trimmed:
        out += ["", "TRIMMED (arm kept, named implementations retired):"]
        out += [f"  {line}" for line in corpus.trimmed]
    if corpus.refused:
        out += ["", "REFUSED (named, not skipped):"]
        out += [f"  {line}" for line in corpus.refused]
    return out


def measurement_lines(cal) -> list[str]:
    out = ["", "THE NEW RULER", f"  device            {cal.gpu_name}",
           f"  ceiling           {cal.ceiling_pattern} "
           f"{cal.achieved_bandwidth_gbps:.1f} GB/s",
           f"  achieved bf16     {cal.achieved_bf16_tflops:.1f} TFLOP/s at "
           f"{cal.gemm_clock_mhz} MHz under load"]
    band = cal.ridge_band()
    if band:
        out.append(f"  ridge band        {band[0]:.1f} to {band[1]:.1f} FLOP/byte "
                   f"({100 * (band[1] / band[0] - 1):.1f}% wide)")
    else:
        out.append("  ridge band        REFUSED: no usable read pattern this run")
    out.append(f"  clock established {cal.clock_established}")
    out.append("")
    out += [f"  {line}" for line in cal.settle_lines()]
    out.append("")
    out.append(f"  {'pattern':<14}{'GB/s':>10}{'clk start':>11}{'clk end':>9}"
               "   note")
    for pat in cal.bandwidth_patterns:
        out.append(f"  {pat.pattern:<14}{pat.gbps:>10.1f}"
                   f"{pat.sm_clock_start_mhz:>11}{pat.sm_clock_end_mhz:>9}"
                   f"   {pat.note[:60]}")
    for line in cal.refusals:
        out.append(f"  REFUSED           {line}")
    return out


def render(header: list[str], gates: list[Gate], body: list[str]) -> str:
    out = list(header) + body + ["", "=" * 78, "GATES"]
    for gate in gates:
        out += gate.render()
    validity = [g for g in gates if g.kind == VALIDITY]
    claims = [g for g in gates if g.kind == CLAIM]
    out.append("")
    if any(g.verdict == FAIL for g in validity):
        out.append("READING IT. A VALIDITY gate failed. No number on this page "
                   "may be quoted; fix the gate and re-run.")
    else:
        failed = [g.number for g in claims if g.verdict == FAIL]
        undecided = [g.number for g in claims if g.verdict == UNDECIDED]
        out.append(f"READING IT. Validity holds. Claim gates failed: "
                   f"{failed or 'none'}; undecided: {undecided or 'none'}.")
        out.append("A failed CLAIM gate is a result, not a broken run: each one "
                   "says above what it costs.")
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=repo / "results" / "published",
                    help="published arms to price the re-baseline against")
    ap.add_argument("--hardware", type=Path,
                    default=repo / "moe" / "bench" / "hardware",
                    help="where the current per-device calibrations live; the "
                         "session lever compares each arm against the one for "
                         "its own device")
    ap.add_argument("--corpus-only", action="store_true",
                    help="price the published rows and stop. Needs no GPU and "
                         "is byte-identical on every machine")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, the predictions and the cost, then "
                         f"stop. Scores no gate, so it exits "
                         f"{exit_codes.REFUSED} REFUSED; --corpus-only is the "
                         "off-GPU mode that does score")
    ap.add_argument("--buffer-gb", type=float, default=8.0)
    ap.add_argument("--gemm-n", type=int, default=8192)
    ap.add_argument("--ceiling", default=DEFAULT_CEILING,
                    help="which pattern the new calibration NAMES as its "
                         "ceiling. Left at triad on purpose: this script exists "
                         "to price the change, not to make it")
    ap.add_argument("--settle-seconds", type=float, default=30.0)
    ap.add_argument("--no-settle", dest="settle", action="store_false",
                    help="skip both settles. Smoke checks only: the clock gate "
                         "has nothing to compare against without them")
    ap.add_argument("--write-calibration", type=Path, default=None,
                    help="also write the new calibration here. Off by default: "
                         "overwriting measured_<device>.yaml mid-session is how "
                         "an arm came to ship a ruler it never used")
    ap.add_argument("--card", default=None,
                    help="the card this run measures, as nvidia-smi names it. "
                         "Defaults to the live device; --corpus-only and "
                         "--dry-run touch no GPU and are labelled "
                         f"{NO_CARD!r}. The measuring path REFUSES without one")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out", type=Path, default=None,
                    help="overrides the results root entirely")
    ap.add_argument("--fail-on-gate", action="store_true",
                    help="RETIRED 2026-09-02 and accepted so old driver lines "
                         f"still parse. A failed CLAIM gate now always exits "
                         f"{exit_codes.CLAIM_FAIL} CLAIM_FAIL, which the ledger "
                         "reads as a finished result rather than a retry; "
                         "folding it into 0 made the log disagree with the "
                         "process")
    return ap


def _main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    card = resolve_card(args)
    run_id = args.run_id or default_run_id(args)
    out_dir = (args.out or results_root()) / "ruler_rebaseline" / run_id
    paths = {"report.txt": out_dir / "report.txt",
             "report.json": out_dir / "report.json"}
    if args.write_calibration:
        paths["calibration"] = args.write_calibration

    header = [
        f"experiment  ruler_rebaseline / {run_id}",
        f"corpus      {args.corpus}",
        f"hardware    {args.hardware}",
        f"mode        {'corpus only (no GPU)' if args.corpus_only else 'measure + price'}",
        f"ruler       named {args.ceiling}, matched {MATCHED_CEILING} "
        f"(fallbacks: {', '.join(READ_PATTERNS)})",
        f"WRITES TO   {out_dir}",
        "",
        "REGISTERED PREDICTIONS. Printed before anything is measured, and not "
        "rewritten afterwards.",
    ]
    for pred in PREDICTIONS:
        header += [""] + [f"  {line}" for line in pred.render()]
    header += [""] + mde_lines() + ["", "=" * 78]
    print("\n".join(header))

    if args.dry_run:
        print(f"\nestimated GPU time {estimated_seconds(args):.0f} s "
              "(two settles, two GEMMs, two clock samples, two bandwidth "
              "passes, one Triton compile)")
        n_arms = (len([p for p in args.corpus.iterdir() if p.is_dir()])
                  if args.corpus.is_dir() else 0)
        print(f"corpus arms        {n_arms}")
        for label, path in paths.items():
            print(f"  {label:<14}{path}  "
                  f"{'IGNORED by git' if git_check_ignore(path) else 'tracked'}")
        # REFUSED (2) AND NOT THE BARE LITERAL 0. This branch printed "Nothing
        # was measured." and then returned DONE, whose entry in the shared table
        # reads "measured; every VALIDITY and CLAIM gate PASSED" -- two
        # sentences apart, in the same eleven lines. It also scored no gate, so
        # it prints no RESULT line, and `exit_codes.classify_text` over this log
        # raises `NoGatesScored`, which that module documents as what a REFUSED
        # log looks like from there. The log and the code now say the same
        # thing, which is the whole point of there being one table.
        #
        # THE REPOSITORY DISAGREED WITH ITSELF AND THIS IS THE SIDE THAT WON.
        # On 2026-09-02 six scripts returned DONE from `--dry-run` --
        # `bm128_depth`, `bm128_roofline`, `bn_decomposition`, `tile_cap_test`,
        # and `block_m_crossing_sweep` and this file as the bare literal 0 --
        # and seven returned REFUSED (`calibrate_hardware`,
        # `dtype_tile_confound`, `memory_branch_anchor`, `occupancy_vs_swizzle`,
        # `rescore_published_reports`, `span_extent_separation`, `tile_sweep`).
        # REFUSED is the only one of the two that a log can be checked against.
        # The driver re-queues neither: in `--dry` mode `dry_state` maps 0 to
        # PLANNED and 2 to PLAN_REFUSED and `arm` retries neither state.
        print("\n".join(["", "=" * 78,
                         "REFUSED. Nothing was measured and nothing was "
                         "written.",
                         "  reason: --dry-run was given",
                         "  No gate was scored, so no RESULT line was printed "
                         "and none of the above",
                         "  is a result. --corpus-only prices the published "
                         "rows off GPU and DOES",
                         "  score gates 4 and 5; the bare command runs the pod "
                         "measurement.",
                         "=" * 78]))
        return exit_codes.REFUSED

    cal = None
    if not args.corpus_only:
        if card == NO_CARD:
            # BEFORE the calibration runs. Gates 1 to 3 compare a fresh
            # measurement against THIS CARD's registered constants, and both
            # reports are written with `write_text`, which truncates whatever a
            # differently-carded run left in the same directory.
            print("\nREFUSED. Nothing was measured.")
            print("  This run would MEASURE and no card was named. Pass --card "
                  "as nvidia-smi spells it:")
            print("  the run id, the output directory and the registered "
                  "constants gates 1 to 3 are")
            print("  scored against are all per-card. --corpus-only prices the "
                  "published rows without one.")
            return exit_codes.REFUSED
        missing = missing_gpu_stack()
        if missing:
            print("\n" + missing)
            return exit_codes.REFUSED
        from moe.bench.calibrate import calibrate
        started = time.time()
        cal = calibrate(int(args.buffer_gb * (1 << 30)), args.gemm_n,
                        args.ceiling, settle=args.settle,
                        settle_seconds=args.settle_seconds)
        print(f"\nmeasured in {time.time() - started:.0f} s")

    current = load_current_rulers(args.hardware)
    corpus = survey(args.corpus, current)

    gates = [gate_v1_non_vacuity(corpus), gate_v2_identity(corpus),
             gate_v3_paths(paths, args.out or results_root())]
    if cal is not None:
        gates += [gate_1_clock(cal), gate_2_bandwidth(cal), gate_3_read_shape(cal)]
    else:
        # UNDECIDED, not absent. A gate that vanishes when it cannot run reads
        # as a page with fewer gates rather than as a page with an untested
        # claim on it, and the second is what this is.
        for number, claim in (
                ("1", "the GEMM clock was sampled in the wrong state"),
                ("2", "the bandwidth patterns reproduce and did not ramp"),
                ("3", "the reduction shape still costs something")):
            gates.append(Gate(CLAIM, number, claim, UNDECIDED,
                              "no measurement this run", "needs a GPU",
                              ["--corpus-only cannot test this. Run the bare "
                               "command on the pod."]))
    gates += [gate_4_spread(corpus), gate_5_flips(corpus)]

    body = (measurement_lines(cal) if cal is not None else []) + corpus_lines(corpus)
    text = render(header, gates, body)
    print("\n".join(text.splitlines()[len(header):]))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(text + "\n")
    # PROVENANCE GOES IN THE JSON AND NOT IN report.txt, on purpose. The block
    # carries a UTC timestamp and a git dirty count, and `report.txt` is the
    # artefact two corpus-only runs are compared byte for byte to prove the
    # replay reads no hardware. A timestamp in there would make that comparison
    # fail for a reason that has nothing to do with reproducibility.
    prov = PV.provenance_block(instrument=instrument_name())
    payload = prov.stamp({
        "run_id": run_id, "card": card,
        "predictions": [asdict(p) for p in PREDICTIONS],
        "gates": [asdict(g) for g in gates],
        "corpus": {"arms": corpus.arms, "rows": corpus.rows,
                   "dropped": corpus.dropped, "refused": corpus.refused,
                   "identity_bad": corpus.identity_bad,
                   "trimmed": corpus.trimmed,
                   "rulers": corpus.rulers,
                   "denominator": [asdict(s) for s in corpus.denominator],
                   "session": [asdict(s) for s in corpus.session]},
        "calibration": cal.as_dict() if cal is not None else None,
    })
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2))
    print(f"\nreport   {out_dir / 'report.txt'}")
    print(f"json     {out_dir / 'report.json'}")

    if cal is not None and args.write_calibration:
        import yaml
        args.write_calibration.parent.mkdir(parents=True, exist_ok=True)
        args.write_calibration.write_text(yaml.safe_dump(cal.as_dict(),
                                                         sort_keys=False))
        print(f"calibration {args.write_calibration}")
        print("  NOTE this is the detail block only, not the full schema "
              "scripts/calibrate_hardware.py writes. Use that to publish.")

    # THE EXIT CODE COMES FROM THE SHARED TABLE, over the gates that were
    # actually SCORED -- `Gate.scored` returns None for an UNDECIDED gate, the
    # same set `result_line` prints -- so `exit_codes.classify_text` on this log
    # recomputes the code the process returned.
    #
    # A GATE THAT NEVER RAN IS NOT A FAILED CLAIM. `--corpus-only` cannot reach
    # gates 1 to 3 at all: there is no GPU and no fresh calibration to read.
    # Scoring them UNKNOWN would classify every corpus-only replay as
    # CLAIM_FAIL, which is a statement about the world rather than about the
    # machine, and this run made no such statement. They are named in the human
    # table and counted in the line below instead.
    scored = [g.scored() for g in gates if g.scored() is not None]
    undecided = [g.token for g in gates if g.verdict == UNDECIDED]
    if undecided:
        print(f"\nNOT SCORED: {', '.join(undecided)} could not run here, so "
              f"they print no RESULT line and are not in the exit code. "
              f"Run the bare command on the pod to decide them.")
    if not scored:
        print("\nNON-VACUITY: no gate could be scored at all, so nothing above "
              "is a result.")
        return exit_codes.REFUSED
    # NOTHING IS FOLDED INTO DONE, and `--fail-on-gate` is why this is a
    # paragraph and not a branch. Until 2026-09-02 a CLAIM_FAIL was described in
    # words and RETURNED AS 0 unless the flag was passed, so `--corpus-only`
    # printed `RESULT: CLAIM C5 FAIL 90 flips in 53188 classified rows` and the
    # process said DONE: the exact log-versus-exit-code split the comment above
    # claims cannot happen here, in the file that prints it. The masking was
    # obsolete anyway once the shared table landed. CLAIM_FAIL (1) is in
    # `FINISHED_CODES` and `ledger_state(1)` is "CLAIM_FAIL", so 1 already tells
    # the driver "this is a result, do not retry it" and 0 protects nothing. The
    # flag is accepted and ignored so an old driver line still parses.
    rc = exit_codes.classify(scored)
    print(f"exit     {exit_codes.describe(rc)}")
    return rc


def main(argv=None) -> int:
    """AN UNPLANNED CRASH IS ERROR (4), which is the only retryable code.

    Left to propagate, an unexpected exception exits the interpreter ONE, and
    ONE is CLAIM_FAIL, which `moe/bench/exit_codes.py` defines as a RESULT: it
    is in FINISHED_CODES, so the driver files the arm as finished, skips it on
    every resume, leaves RETRY_ARMS at zero and exits the session 0 over an arm
    that never measured. A torch OOM, a truncated report or a drifted import
    would be published as one of this experiment's registered outcomes. ERROR
    (4) is outside FINISHED_CODES precisely so the driver can tell "the
    apparatus broke" from "the claim did not hold", and the traceback is
    printed first rather than swallowed, because a code without one tells an
    operator nothing about what to fix.

    Wrapped around `_main` rather than installed at the `__main__` guard so the
    contract holds for a caller of `main()` -- the tests, and anything that
    imports this file -- as well as for the CLI. `SystemExit` is a
    `BaseException` and passes through untouched: a refusal is not a crash.
    """
    try:
        return _main(argv)
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        print("ERROR: ruler_rebaseline crashed before it could reach a "
              "verdict. This is the apparatus failing, not a claim "
              "failing, so it exits "
              f"{exit_codes.ERROR} and not {exit_codes.CLAIM_FAIL}: the "
              "traceback above is the thing to fix, and the arm may be re-run.",
              file=sys.stderr)
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
