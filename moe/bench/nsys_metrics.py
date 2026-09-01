"""Can a SAMPLED GPU-metrics trace stand in for the DRAM counter that ncu refuses?

WHY THIS MODULE EXISTS. Every byte figure in this study is compulsory-traffic
ARITHMETIC. `implied_traffic_ratio` is time x achievable-bandwidth over modelled
bytes, which is an inference and not a measurement, and `alpha` (the cost of an
extra M-tile as a fraction of a fresh weight read, refit 2026-08-31 from 0.10 to
0.558) is fitted through that same model. The whole tile-corrected roofline rests
on one regression against a byte model that has never been validated against a
byte. Nothing in this repository has ever counted one.

`ncu` would count them and cannot: `dram__bytes_read.sum` is a hardware
performance counter, and reading counters needs
`NVreg_RestrictProfilingToAdminUsers=0`, a host kernel-module flag a container
tenant cannot set. On RunPod it fails with ERR_NVGPUCTRPERM.
`scripts/profile_open_questions.sh` concluded from that "Q1 traffic -> ncu ONLY.
dram__bytes_read.sum is a counter; nothing traces it", and docs/RUNPOD.md,
docs/POD_RUNBOOK.md and docs/FINDINGS.md have all repeated it since.

THAT CONCLUSION SKIPS A MECHANISM. `nsys --gpu-metrics-device` neither traces
nor goes through the CUPTI profiling API that ncu is blocked on. It SAMPLES the
GPU's hardware performance monitor at a fixed rate through a separate path, and
whether that path is gated on the same module flag is an empirical question this
repository has never asked. `scripts/nsys_dram_probe.py` asks it on a pod. This
module is the half that is pure text and SQL and therefore testable on a laptop.

WHAT A SAMPLED METRIC IS NOT, and the three properties are why nothing here
returns a bare byte count:

  1. IT IS DEVICE WIDE. A sample covers the whole GPU, not a kernel and not a
     process. Anything else resident lands in it. "Attribution" here is no more
     than "sum the samples whose timestamp falls inside a window during which
     the only thing we launched was the kernel under test", which is a
     scheduling assumption. `DramTraffic` carries an idle baseline measured
     outside every kernel window precisely so a reader can see how much of the
     signal was there when nothing of ours was running.

  2. IT QUANTISES AT THE WINDOW EDGE. The first and last sample periods of a
     window straddle its boundary, so a window of `n` sample periods carries a
     systematic `2/n` uncertainty that no amount of averaging removes. At the
     nsys default of 10 kHz the period is 100 us and a 54 us kernel -- the T=1
     fused cell this study times -- does not contain a single sample. At the
     200 kHz maximum it contains about 11, for an edge term of 18%. THE
     CONSEQUENCE IS STRUCTURAL: a single short launch is not measurable at any
     rate nsys offers, and the only way to buy resolution is to make ONE
     CONTIGUOUS WINDOW out of many back-to-back launches, because the edge term
     is per window and not per sample. `merge_windows` exists for that and
     `Resolution` refuses when the arithmetic does not come out.

  3. IT MAY NOT BE IN BYTES. Depending on chip and metric set, nsys reports DRAM
     activity as a percentage of peak throughput rather than as a byte count.
     Converting a percentage to bytes needs the peak DRAM bandwidth, which is
     the very calibrated ceiling `implied_traffic_ratio` already depends on. A
     measurement taken that way is still independent of the COMPULSORY BYTE
     MODEL, which is what this study needs, but it is NOT independent of the
     calibration, and `DramTraffic.route` says which of the two happened so a
     reader can never mistake one for the other.

WHY THE SQLITE EXPORT AND NOT `nsys stats`. `nsys stats` reports are a moving
target: the report names, their columns and their availability have all changed
across versions, several of the GPU-metrics reports need a `--report` name that
does not exist on older builds, and every one of them summarises over the WHOLE
capture rather than over a window. This module needs per-window sums keyed to
kernel start and end times in the same timebase, which is exactly what the
sqlite export gives and no stats report does. The export costs one extra command
and buys a schema that can be introspected and refused on, which is why
`require_table` names what it found rather than raising an OperationalError.

Everything here is pure arithmetic, regex and sqlite over files nsys wrote. No
torch, no CUDA, no subprocess. The subprocess half lives in the script.
"""
from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# --------------------------------------------------------------------------
# Refusals. Every one of these is a state where returning a number would be
# worse than returning nothing, which is the same rule
# `efficiency.TrafficRatioUnavailable` was written under after a 0.0 default
# moved a published median.
# --------------------------------------------------------------------------


class NsysProbeRefused(RuntimeError):
    """This measurement cannot be made here, and the message says why."""


class NsysUnavailable(NsysProbeRefused):
    """No nsys binary, or one whose version string is unreadable."""


class GpuMetricsUnsupported(NsysProbeRefused):
    """This nsys build offers no GPU-metrics sampling flag at all."""


class ReportSchemaUnsupported(NsysProbeRefused):
    """The exported sqlite does not have the tables this parser reads.

    Named separately from "the metric is missing" because the two have opposite
    fixes: a missing table means the export or the nsys version is wrong, while
    a missing metric means sampling ran but this chip's metric set does not
    break DRAM out.
    """


class MetricNotFound(NsysProbeRefused):
    """Sampling ran, but no metric in the catalogue names DRAM read or write."""


class UnitUnknown(NsysProbeRefused):
    """The metric's unit does not say how to turn its values into bytes."""


class CeilingRequired(NsysProbeRefused):
    """A percent-of-peak metric was sampled and no peak bandwidth was supplied."""


class WindowTooShort(NsysProbeRefused):
    """The kernel window holds too few samples for the number to mean anything."""


# --------------------------------------------------------------------------
# What the installed nsys offers. Read out of the binary's OWN help text rather
# than inferred from its version, because a distro package can carry either
# spelling and guessing is exactly the failure this repo already recorded once:
# profile_open_questions.sh branched on `command -v ncu` being true and sent a
# question down a path the machine could not run.
# --------------------------------------------------------------------------

#: Renamed in Nsight Systems 2024.5. Newest spelling first; the first one the
#: help text actually mentions wins.
GPU_METRICS_DEVICE_FLAGS = ("--gpu-metrics-devices", "--gpu-metrics-device")
GPU_METRICS_SET_FLAGS = ("--gpu-metrics-set",)
GPU_METRICS_FREQUENCY_FLAGS = ("--gpu-metrics-frequency",)

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


@dataclass(frozen=True)
class NsysVersion:
    """What `nsys --version` said, parsed, with the raw line kept.

    The raw line travels with it because the probe's report has to be able to
    quote what it found rather than what it decided the string meant.
    """

    year: int
    minor: int
    patch: int
    raw: str

    @property
    def as_tuple(self) -> tuple[int, int, int]:
        return (self.year, self.minor, self.patch)

    def __str__(self) -> str:
        return f"{self.year}.{self.minor}.{self.patch}"


def parse_nsys_version(text: str) -> NsysVersion:
    """`NVIDIA Nsight Systems version 2024.6.1.90-...` -> 2024.6.1.

    Raises rather than returning None because a probe that cannot name the tool
    it ran has nothing to report: the whole point of this exercise is that the
    next reader can tell which nsys produced the answer.
    """
    for line in text.splitlines():
        m = _VERSION_RE.search(line)
        if m and int(m.group(1)) > 1000:   # a year, not a build counter
            return NsysVersion(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                               line.strip())
    raise NsysUnavailable(
        "could not find a version in the output of `nsys --version`:\n"
        + (text.strip() or "(no output)"))


@dataclass(frozen=True)
class GpuMetricsSupport:
    """Which gpu-metrics flags this build spells, and how."""

    device_flag: str
    set_flag: str | None
    frequency_flag: str | None
    #: Every gpu-metrics-ish flag seen in the help text, so a refusal can quote
    #: what WAS there instead of only what was missing.
    offered: tuple[str, ...]


def parse_gpu_metrics_support(help_text: str) -> GpuMetricsSupport:
    """Find the sampling flags in `nsys profile --help`.

    Refuses when no device flag is present. That is a real state and not a
    parsing failure: Nsight Systems builds for platforms without the sampler,
    and some slim container packages ship the CLI trace path only.
    """
    offered = tuple(sorted(set(re.findall(r"--gpu-metrics[\w-]*", help_text))))
    device = next((f for f in GPU_METRICS_DEVICE_FLAGS if f in help_text), None)
    if device is None:
        raise GpuMetricsUnsupported(
            "this nsys build's `profile --help` mentions no "
            f"{' or '.join(GPU_METRICS_DEVICE_FLAGS)} flag, so it cannot sample "
            "GPU hardware metrics at all. gpu-metrics flags it does offer: "
            + (", ".join(offered) or "(none)")
            + "\nTracing still works, and tracing answers kernel names and "
              "launch overhead, but not DRAM traffic.")
    return GpuMetricsSupport(
        device_flag=device,
        set_flag=next((f for f in GPU_METRICS_SET_FLAGS if f in help_text), None),
        frequency_flag=next((f for f in GPU_METRICS_FREQUENCY_FLAGS if f in help_text),
                            None),
        offered=offered,
    )


@dataclass(frozen=True)
class MetricSet:
    """One entry of `--gpu-metrics-set=help`: `[13] [gh100] General Metrics ...`."""

    index: int
    chip: str
    description: str


_SET_RE = re.compile(r"^\s*\[(\d+)\]\s*\[([\w.]+)\]\s*(.*?)\s*$")


def parse_metric_sets(text: str) -> tuple[MetricSet, ...]:
    """Every metric set the installed nsys will accept, in the order it listed."""
    found = []
    for line in text.splitlines():
        m = _SET_RE.match(line)
        if m:
            found.append(MetricSet(int(m.group(1)), m.group(2), m.group(3)))
    return tuple(found)


#: Device-name substring -> the chip tag nsys names its metric set after. Only
#: the parts this study has ever run on plus the two it might. Matched longest
#: first so "H100" does not shadow nothing and "A100" does not match "GA100".
CHIP_BY_DEVICE_SUBSTRING = (
    ("H200", "gh100"),
    ("H100", "gh100"),
    ("A100", "ga100"),
    ("L40", "ad10x"),
    ("B200", "gb100"),
)


def select_metric_set(sets: tuple[MetricSet, ...], device_name: str) -> MetricSet | None:
    """The set whose chip tag matches this device, or None to let nsys choose.

    None is not a failure. nsys picks a set on its own when none is given, and
    the only reason to name one is so the probe's report can say which was used
    rather than leaving it to whatever the tool defaulted to.
    """
    upper = device_name.upper()
    for needle, chip in CHIP_BY_DEVICE_SUBSTRING:
        if needle in upper:
            for s in sets:
                if s.chip.lower() == chip:
                    return s
    return None


# --------------------------------------------------------------------------
# The sampling arithmetic. This is the section that decides whether a run is
# worth taking seriously, and it needs no nsys, no GPU and no report file.
# --------------------------------------------------------------------------

#: nsys samples GPU metrics at 10 kHz unless told otherwise.
DEFAULT_SAMPLE_HZ = 10_000.0

#: The documented ceiling. Asking for more is rejected by the tool, so a probe
#: that needs more resolution than this needs a different tool, not a flag.
#:
#: THE CONCLUSION IS INSENSITIVE TO THIS NUMBER, which matters because it is the
#: one constant here taken from documentation rather than measured. A 54 us
#: kernel reaches MIN_SAMPLES_PER_WINDOW only above 370 kHz, and a 10% edge term
#: on a single launch would need 3.7 MHz. So even if a build turns out to accept
#: more than 200 kHz, a single short launch stays unmeasurable by two orders of
#: magnitude, and the ladder in `scripts/nsys_dram_probe.py` reports the rate the
#: installed tool actually accepted rather than trusting this value.
MAX_SAMPLE_HZ = 200_000.0

#: A window shorter than this many sample periods gets no number. Twenty is not
#: a statistical threshold, it is the point where the edge term (2/n) drops to
#: 10%, which is the coarsest a traffic figure can be and still tell the two
#: candidate alphas apart at n=2 tiles. See DISCRIMINATE_ALPHA_REL below.
MIN_SAMPLES_PER_WINDOW = 20

#: The systematic edge uncertainty this module will report a number at. Above
#: it, `dram_traffic` refuses.
EDGE_ERROR_LIMIT = 0.10

#: The kernel this study actually wants traffic for. A T=1 fused MoE cell on an
#: H200 runs about this long, and it is quoted here so the refusal message can
#: name the real case rather than a hypothetical short kernel.
SHORT_KERNEL_US = 54.0

#: Telling alpha=0.558 from the retracted alpha=0.10 means separating Q(2)=1.558
#: from Q(2)=1.100. Half that gap over the midpoint is 17%, so a traffic
#: measurement coarser than this cannot choose between the two worlds the study
#: is arguing about.
DISCRIMINATE_ALPHA_REL = 0.17

#: Pinning alpha to +/-0.05 at n=2 needs dQ/Q of 0.05/1.558.
MEASURE_ALPHA_REL = 0.032


def sample_period_ns(sample_hz: float) -> float:
    if sample_hz <= 0:
        raise ValueError("sample rate must be positive Hz")
    return 1e9 / sample_hz


def alpha_uncertainty(rel_error: float, alpha: float = 0.558, n_tiles: int = 2) -> float:
    """The +/- on alpha implied by measuring traffic to `rel_error` at `n_tiles`.

    This is the function that turns "the sampler is good to X%" into an answer
    to the question the study is actually asking. Traffic in units of one full
    weight read is `Q = 1 + alpha (n - 1)`, so `alpha = (Q - 1) / (n - 1)` and
    an error `rel_error * Q` on Q lands on alpha divided by `n - 1`.

    Note which way it points: MORE tiles per expert gives a SHARPER alpha, so a
    probe that cannot resolve a two-tile cell may still resolve an eight-tile
    one. That is the design advice this function exists to make quantitative.
    """
    if n_tiles < 2:
        raise ValueError("alpha is unidentifiable below two M-tiles: Q(1) = 1 "
                         "for every alpha, which is why the refit needed the "
                         "multi-tile regime")
    q = 1.0 + alpha * (n_tiles - 1)
    return rel_error * q / (n_tiles - 1)


@dataclass(frozen=True)
class Resolution:
    """Whether a sampled measurement over these windows means anything.

    `ok` is the gate and `reason` is what to print. Both are kept on the result
    rather than raised immediately, so a caller that wants the number anyway
    (`allow_unresolved=True`) still carries the verdict alongside it and cannot
    quote the figure without quoting the caveat.
    """

    window_ns: int
    n_windows: int
    sample_hz: float
    n_samples: float
    #: `2 * n_windows / n_samples`: two straddled sample periods per window, and
    #: it does NOT shrink by adding more launches unless they merge into one
    #: contiguous window.
    edge_error: float
    ok: bool
    reason: str

    @property
    def samples_per_window(self) -> float:
        return self.n_samples / self.n_windows if self.n_windows else 0.0

    def alpha_band(self, n_tiles: int = 2) -> float:
        return alpha_uncertainty(self.edge_error, n_tiles=n_tiles)

    def text(self) -> str:
        return (f"{self.n_windows} window(s), {self.window_ns / 1e6:.3f} ms total, "
                f"{self.sample_hz / 1e3:.1f} kHz -> {self.n_samples:.1f} samples "
                f"({self.samples_per_window:.1f} per window), edge term "
                f"{self.edge_error * 100:.1f}%: "
                f"{'usable' if self.ok else 'NOT USABLE'} -- {self.reason}")


def resolve(window_ns: int, n_windows: int, sample_hz: float,
            min_samples: int = MIN_SAMPLES_PER_WINDOW,
            edge_limit: float = EDGE_ERROR_LIMIT) -> Resolution:
    """Does a window of this length, sampled at this rate, support a number?

    Two independent tests, and a window has to pass both:

      * enough samples inside it at all, so the sum is not one or two reads;
      * a small enough EDGE term, which is the part people forget. Sampling a
        thousand separate 54 us launches gives plenty of samples in total and
        still cannot say what any of them moved, because every launch brings its
        own two straddled periods and the systematic error does not average
        down. Merge the launches into one contiguous window and the same total
        sample count comes with a hundredth of the edge error.
    """
    if n_windows <= 0:
        return Resolution(window_ns, 0, sample_hz, 0.0, math.inf, False,
                          "no kernel window matched, so there is nothing to sum")
    period = sample_period_ns(sample_hz)
    n_samples = window_ns / period
    edge = math.inf if n_samples <= 0 else 2.0 * n_windows / n_samples
    if n_samples < min_samples:
        return Resolution(
            window_ns, n_windows, sample_hz, n_samples, edge, False,
            f"{n_samples:.2f} samples at {sample_hz / 1e3:.0f} kHz is under the "
            f"{min_samples} this module will report on. One sample period is "
            f"{period / 1000:.1f} us and the window is {window_ns / 1000:.1f} us")
    if edge > edge_limit:
        return Resolution(
            window_ns, n_windows, sample_hz, n_samples, edge, False,
            f"edge term {edge * 100:.1f}% exceeds the {edge_limit * 100:.0f}% "
            f"limit: {n_windows} separate windows each contribute two straddled "
            "sample periods. Launch the kernel back to back so the windows "
            "merge, or profile a longer one")
    return Resolution(window_ns, n_windows, sample_hz, n_samples, edge, True,
                      f"edge term {edge * 100:.1f}%, which puts alpha at "
                      f"+/-{alpha_uncertainty(edge):.3f} at two M-tiles")


def single_launch_verdict(kernel_us: float = SHORT_KERNEL_US,
                          sample_hz: float = MAX_SAMPLE_HZ) -> Resolution:
    """What one launch of a short kernel can do, at the best rate nsys offers.

    Exists so the answer to "why not just profile the cell" is a computed line
    in the report rather than a claim in a docstring.
    """
    return resolve(int(kernel_us * 1000), 1, sample_hz)


# --------------------------------------------------------------------------
# Kernel windows.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A half-open `[start, end)` interval in the report's own timebase.

    Nanoseconds, and the same clock GPU_METRICS timestamps are in, because both
    come from the same nsys collection. That is the whole reason attribution
    goes through the kernel table rather than through wall-clock marks taken in
    the workload: a host timestamp would have to be mapped into nsys's timebase,
    and a mapping that is a few microseconds off is invisible and fatal at these
    window lengths.
    """

    start_ns: int
    end_ns: int
    n_launches: int = 1

    def __post_init__(self) -> None:
        if self.end_ns < self.start_ns:
            raise ValueError(f"window ends before it starts: {self!r}")

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


def merge_windows(windows, gap_ns: float) -> tuple[Window, ...]:
    """Fuse windows separated by less than `gap_ns` into single windows.

    THE POINT OF THE WHOLE MODULE IS IN THIS FUNCTION. The edge term is charged
    per window, so N back-to-back launches profiled as N windows carry N times
    the systematic error of the same N launches profiled as one. Merging is only
    legitimate when the gaps really are shorter than a sample period, because
    then the sampler could not have resolved them anyway and the fused window
    describes what the sampler actually saw.

    `gap_ns` should be one sample period. A larger tolerance starts folding
    genuinely idle time into the window, which biases traffic DOWN by counting
    duration that moved nothing.
    """
    ordered = sorted(windows, key=lambda w: (w.start_ns, w.end_ns))
    if not ordered:
        return ()
    out: list[Window] = []
    cur = ordered[0]
    for w in ordered[1:]:
        if w.start_ns - cur.end_ns <= gap_ns:
            cur = Window(cur.start_ns, max(cur.end_ns, w.end_ns),
                         cur.n_launches + w.n_launches)
        else:
            out.append(cur)
            cur = w
    out.append(cur)
    return tuple(out)


def duty_cycle(windows) -> float:
    """Union of the windows over the span they cover, in [0, 1].

    Reported rather than gated on, because it is the direct read on how much of
    the profiled interval the device was doing something else, or nothing. A low
    duty cycle with merged windows means the merge swallowed idle time.
    """
    ordered = merge_windows(windows, 0)
    if not ordered:
        return 0.0
    span = ordered[-1].end_ns - ordered[0].start_ns
    covered = sum(w.duration_ns for w in ordered)
    return covered / span if span > 0 else 1.0


def fill_fraction(raw_launches, selected) -> float:
    """How much of a MERGED window was really kernel, not sub-period gap.

    THE BIAS THIS CATCHES, and it is the one merging introduces. Merging fuses
    launches whose gaps are shorter than a sample period, which is legitimate
    because the sampler could not have resolved them, but the fused window is
    then LONGER than the kernel time inside it. Traffic is summed over the fused
    window while the byte model counts only the launches, so a 5% gap makes the
    measured-over-modelled ratio read 5% low for a reason that has nothing to do
    with the kernel. Below about 0.95 the ratio should be read as a lower bound.

    Distinct from `duty_cycle`, which is about the whole profiled interval,
    including the idle beat the workloads deliberately leave in it.
    """
    chosen = sorted(selected, key=lambda w: w.start_ns)
    span = sum(w.duration_ns for w in chosen)
    if span <= 0:
        return 0.0
    covered = 0
    for launch in raw_launches:
        for w in chosen:
            lo, hi = max(launch.start_ns, w.start_ns), min(launch.end_ns, w.end_ns)
            if hi > lo:
                covered += hi - lo
    return covered / span


# --------------------------------------------------------------------------
# The exported sqlite. Every table and column is introspected before it is read,
# so a schema change across nsys versions produces a message naming what was
# found instead of an OperationalError from three frames down.
# --------------------------------------------------------------------------

GPU_METRIC_TABLE = "GPU_METRICS"
GPU_METRIC_CATALOGUE = "TARGET_INFO_GPU_METRICS"
KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
STRING_TABLE = "StringIds"


def open_report(path) -> sqlite3.Connection:
    """Open an `nsys export --type sqlite` database read only.

    Read only on purpose: this file is the only artefact of a paid GPU minute
    and nothing in an analysis path should be able to write to it.
    """
    p = Path(path)
    if not p.is_file():
        raise ReportSchemaUnsupported(
            f"no sqlite report at {p}. Produce one with:\n"
            f"  nsys export --type sqlite --output {p} <report>.nsys-rep")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def table_names(conn: sqlite3.Connection) -> frozenset[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()
    return frozenset(r[0] for r in rows)


def column_names(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(r[1] for r in conn.execute(f"PRAGMA table_info({table})"))


def require_table(conn: sqlite3.Connection, table: str, why: str) -> None:
    present = table_names(conn)
    if table in present:
        return
    likely = sorted(t for t in present if "METRIC" in t.upper() or "KERNEL" in t.upper())
    raise ReportSchemaUnsupported(
        f"this report has no `{table}` table, which is where {why}.\n"
        f"Tables that look related: {', '.join(likely) or '(none)'}\n"
        f"Total tables in the export: {len(present)}.\n"
        "A report collected WITHOUT the gpu-metrics flag exports no metrics "
        "table at all, which is the most likely cause and is a run problem "
        "rather than a parse problem.")


# --------------------------------------------------------------------------
# Metrics, and what their values mean.
# --------------------------------------------------------------------------


class MetricUnit(Enum):
    """How to turn a sampled value into bytes, or the refusal to guess."""

    PERCENT_OF_PEAK = "percent of peak throughput"
    BYTES_PER_SAMPLE = "bytes accumulated in one sample period"
    BYTES_PER_SECOND = "bytes per second, instantaneous"
    UNKNOWN = "unknown"


_PERCENT_HINTS = ("%", "pct", "percent", "throughput %")
_RATE_HINTS = ("/s", "per second", "bytes/sec", "b/s")


def classify_unit(name: str, unit: str | None) -> MetricUnit:
    """Decide from the metric's own name and unit string, never from the chip.

    Deliberately conservative. UNKNOWN is a supported answer and the caller
    refuses on it, because the failure mode of guessing here is a byte figure
    wrong by the peak bandwidth, which is a factor of thousands and would look
    exactly like a spectacular finding.
    """
    blob = f"{name} {unit or ''}".lower()
    if any(h in blob for h in _PERCENT_HINTS):
        return MetricUnit.PERCENT_OF_PEAK
    if any(h in blob for h in _RATE_HINTS):
        return MetricUnit.BYTES_PER_SECOND
    if "byte" in blob:
        return MetricUnit.BYTES_PER_SAMPLE
    return MetricUnit.UNKNOWN


#: SI prefixes an nsys unit string can carry. A metric reported in GB/s whose
#: values are multiplied straight into bytes is wrong by 1e9, which would look
#: like a spectacular finding rather than a units bug, so the prefix is read
#: rather than assumed away. Longest first: "gb" must be tried before "b".
_SI_PREFIXES = (("tb", 1e12), ("gb", 1e9), ("mb", 1e6), ("kb", 1e3))


def unit_scale(unit: str | None) -> float:
    """Bytes per reported unit, from the unit string's SI prefix."""
    low = (unit or "").lower()
    for prefix, scale in _SI_PREFIXES:
        if prefix in low:
            return scale
    return 1.0


#: Which read metric to prefer when a set exposes more than one. A byte count
#: needs nothing but itself; a percentage needs the measured DRAM ceiling and
#: drags that calibration's uncertainty into the answer, so it is the last
#: resort rather than the first match in table order.
_UNIT_PREFERENCE = (MetricUnit.BYTES_PER_SAMPLE, MetricUnit.BYTES_PER_SECOND,
                    MetricUnit.PERCENT_OF_PEAK, MetricUnit.UNKNOWN)


@dataclass(frozen=True)
class Metric:
    type_id: int
    metric_id: int
    name: str
    unit: str | None = None

    @property
    def unit_kind(self) -> MetricUnit:
        return classify_unit(self.name, self.unit)

    @property
    def scale(self) -> float:
        return unit_scale(self.unit)

    @property
    def direction(self) -> str:
        """`read`, `write`, `combined` or `other`.

        `combined` is its own answer and not a fallback: several nsys metric
        sets expose one "DRAM Bandwidth" figure with no read/write split, and a
        study whose central question is how many times the WEIGHTS were READ
        cannot use a combined number as a read number.
        """
        low = self.name.lower()
        if "dram" not in low:
            return "other"
        has_read = "read" in low or "rd" in low.split()
        has_write = "write" in low or "wr" in low.split()
        if has_read and not has_write:
            return "read"
        if has_write and not has_read:
            return "write"
        return "combined"


def metric_catalogue(conn: sqlite3.Connection) -> tuple[Metric, ...]:
    """Every sampled metric the report describes, with its unit when recorded."""
    require_table(conn, GPU_METRIC_CATALOGUE,
                  "nsys records the name of each sampled metric")
    cols = column_names(conn, GPU_METRIC_CATALOGUE)
    name_col = next((c for c in ("metricName", "name") if c in cols), None)
    if name_col is None or "metricId" not in cols:
        raise ReportSchemaUnsupported(
            f"`{GPU_METRIC_CATALOGUE}` has columns {cols}, which carry no "
            "metric id and name pair this parser recognises.")
    unit_col = next((c for c in ("unit", "metricUnit") if c in cols), None)
    type_col = "typeId" if "typeId" in cols else None
    select = ", ".join(x for x in (type_col, "metricId", name_col, unit_col) if x)
    out = []
    for row in conn.execute(f"SELECT {select} FROM {GPU_METRIC_CATALOGUE}"):
        values = list(row)
        type_id = values.pop(0) if type_col else 0
        metric_id, name = values[0], values[1]
        unit = values[2] if unit_col else None
        out.append(Metric(int(type_id), int(metric_id), str(name),
                          None if unit is None else str(unit)))
    return tuple(out)


@dataclass(frozen=True)
class DramMetrics:
    read: Metric
    write: Metric | None


def find_dram_metrics(catalogue) -> DramMetrics:
    """The read metric, and the write metric when the set splits them.

    Refuses when only a combined DRAM figure exists, naming every metric that
    mentioned DRAM. That refusal is a real outcome of this probe: on a chip
    whose metric set does not split the direction, nsys cannot answer the
    question this study is asking however well the sampler works.
    """
    dram = [m for m in catalogue if "dram" in m.name.lower()]
    rank = {u: i for i, u in enumerate(_UNIT_PREFERENCE)}
    reads = sorted((m for m in dram if m.direction == "read"),
                   key=lambda m: rank[m.unit_kind])
    writes = sorted((m for m in dram if m.direction == "write"),
                    key=lambda m: rank[m.unit_kind])
    if not reads:
        raise MetricNotFound(
            "no sampled metric names a DRAM READ. Metrics mentioning DRAM: "
            + (", ".join(f"{m.name!r}" for m in dram) or "(none)")
            + f"\nThe catalogue holds {len(catalogue)} metrics in total. A "
              "combined DRAM figure cannot substitute: this study's question is "
              "how many times the WEIGHTS were read, and writes are a different "
              "term of the byte model.")
    return DramMetrics(reads[0], writes[0] if writes else None)


# --------------------------------------------------------------------------
# Reading the samples.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSum:
    total: float
    n_samples: int
    #: Samples landing within one period of a window edge. These are the ones
    #: whose period straddled the boundary, so they are the population the edge
    #: term describes, reported as a count so a reader can see it directly
    #: rather than trusting the 2/n arithmetic.
    n_boundary: int


def kernel_windows(conn: sqlite3.Connection, pattern: str | None = None
                   ) -> tuple[Window, ...]:
    """Every kernel launch in the report, optionally filtered by name regex.

    Matched against the demangled name when the export has one and the short
    name otherwise, and the regex is applied in PYTHON rather than in SQL so the
    same pattern behaves the same way whichever column the export carries.
    """
    require_table(conn, KERNEL_TABLE, "nsys records each kernel launch")
    cols = column_names(conn, KERNEL_TABLE)
    for needed in ("start", "end"):
        if needed not in cols:
            raise ReportSchemaUnsupported(
                f"`{KERNEL_TABLE}` has columns {cols} and no `{needed}`.")
    name_col = next((c for c in ("demangledName", "shortName", "name") if c in cols),
                    None)
    strings = {}
    if STRING_TABLE in table_names(conn):
        strings = {int(i): str(v) for i, v in
                   conn.execute(f"SELECT id, value FROM {STRING_TABLE}")}
    rx = re.compile(pattern) if pattern else None
    select = "start, end" + (f", {name_col}" if name_col else "")
    out: list[Window] = []
    unmatched = 0
    for row in conn.execute(f"SELECT {select} FROM {KERNEL_TABLE} ORDER BY start"):
        if rx is not None:
            raw = row[2] if name_col else None
            name = strings.get(int(raw), str(raw)) if isinstance(raw, int) else str(raw)
            if not rx.search(name or ""):
                unmatched += 1
                continue
        out.append(Window(int(row[0]), int(row[1])))
    if rx is not None and not out:
        raise MetricNotFound(
            f"no kernel name matched {pattern!r}; {unmatched} launches were "
            "examined. Run with no pattern to list what the report holds.")
    return tuple(out)


def sum_samples(conn: sqlite3.Connection, metric: Metric, windows,
                period_ns: float) -> SampleSum:
    """Total the metric's samples inside the windows, in one ordered pass.

    Fetched and bucketed in python rather than assembled into a SQL predicate
    with one OR per window, because a back-to-back run produces thousands of
    windows and the resulting statement is both slow and version-fragile.
    """
    require_table(conn, GPU_METRIC_TABLE, "nsys stores the sampled values")
    cols = column_names(conn, GPU_METRIC_TABLE)
    for needed in ("timestamp", "metricId", "value"):
        if needed not in cols:
            raise ReportSchemaUnsupported(
                f"`{GPU_METRIC_TABLE}` has columns {cols} and no `{needed}`.")
    where = "metricId = ?"
    params: list = [metric.metric_id]
    if "typeId" in cols:
        where += " AND typeId = ?"
        params.append(metric.type_id)
    ordered = sorted(windows, key=lambda w: w.start_ns)
    total = 0.0
    n = 0
    boundary = 0
    i = 0
    for ts, value in conn.execute(
            f"SELECT timestamp, value FROM {GPU_METRIC_TABLE} WHERE {where} "
            "ORDER BY timestamp", params):
        ts = int(ts)
        while i < len(ordered) and ordered[i].end_ns <= ts:
            i += 1
        if i >= len(ordered):
            break
        w = ordered[i]
        if w.start_ns <= ts < w.end_ns:
            total += float(value)
            n += 1
            if ts < w.start_ns + period_ns or ts >= w.end_ns - period_ns:
                boundary += 1
    return SampleSum(total, n, boundary)


def sum_samples_outside(conn: sqlite3.Connection, metric: Metric, windows,
                        period_ns: float) -> SampleSum:
    """The same metric where NO kernel of ours was running: the idle baseline.

    This is the isolation check. A device-wide sampler cannot tell our kernel
    from a neighbour's, but it can tell us what the device was reading when we
    had launched nothing, and a baseline that is not near zero means the number
    inside the windows is not ours either.
    """
    require_table(conn, GPU_METRIC_TABLE, "nsys stores the sampled values")
    cols = column_names(conn, GPU_METRIC_TABLE)
    where = "metricId = ?"
    params: list = [metric.metric_id]
    if "typeId" in cols:
        where += " AND typeId = ?"
        params.append(metric.type_id)
    ordered = sorted(windows, key=lambda w: w.start_ns)
    total = 0.0
    n = 0
    i = 0
    for ts, value in conn.execute(
            f"SELECT timestamp, value FROM {GPU_METRIC_TABLE} WHERE {where} "
            "ORDER BY timestamp", params):
        ts = int(ts)
        while i < len(ordered) and ordered[i].end_ns <= ts:
            i += 1
        inside = i < len(ordered) and ordered[i].start_ns <= ts < ordered[i].end_ns
        if not inside:
            total += float(value)
            n += 1
    return SampleSum(total, n, 0)


def to_bytes(unit: MetricUnit, total_value: float, n_samples: int, period_ns: float,
             peak_bytes_per_s: float | None = None, scale: float = 1.0) -> float:
    """Turn a summed sample value into bytes, or refuse to.

    `scale` is the SI prefix on the metric's unit, from `unit_scale`. It is a
    separate argument rather than folded into the value because the caller has
    to be able to see, in one place, that a GB/s metric was recognised as GB and
    not silently treated as bytes.

    The percent route needs the peak DRAM bandwidth and therefore inherits the
    calibration's uncertainty. It does NOT inherit the compulsory byte model,
    which is the model this whole exercise is trying to validate, so a percent
    metric is still an independent check of the thing that matters. Say which
    route was taken whenever the number is quoted.
    """
    period_s = period_ns / 1e9
    if unit is MetricUnit.BYTES_PER_SAMPLE:
        return total_value * scale
    if unit is MetricUnit.BYTES_PER_SECOND:
        return total_value * scale * period_s
    if unit is MetricUnit.PERCENT_OF_PEAK:
        if not peak_bytes_per_s or peak_bytes_per_s <= 0:
            raise CeilingRequired(
                "this metric is a percentage of peak DRAM throughput, so it "
                "cannot become bytes without a peak. Pass the MEASURED ceiling "
                "from moe/bench/hardware/measured_<device>.yaml, not a "
                "datasheet figure, and record that the byte figure now carries "
                "that calibration's uncertainty.")
        return total_value / 100.0 * peak_bytes_per_s * period_s
    raise UnitUnknown(
        "the metric's name and unit do not say what its values are, so no "
        "conversion to bytes is defensible. Guessing here is wrong by the peak "
        "bandwidth, a factor of thousands, and would read as a finding.")


@dataclass(frozen=True)
class DramTraffic:
    """Measured DRAM bytes over a kernel window, with everything needed to doubt it."""

    read_bytes: float
    write_bytes: float | None
    resolution: Resolution
    windows: tuple[Window, ...]
    launches: int
    duty_cycle: float
    #: Bytes per second the sampler saw while none of our kernels were running.
    #: The isolation check, in the same units as the measurement.
    idle_read_bytes_per_s: float
    #: Kernel time over merged-window time. Below ~0.95 the merge is charging
    #: the kernel for gaps it did not run in, and the ratio is a lower bound.
    fill: float
    #: `MetricUnit` value plus the metric names, so the percent route can never
    #: be mistaken for a byte counter.
    route: str
    n_samples: int
    n_boundary_samples: int

    @property
    def total_bytes(self) -> float:
        return self.read_bytes + (self.write_bytes or 0.0)

    def text(self) -> str:
        write = ("not split by this metric set"
                 if self.write_bytes is None else f"{self.write_bytes / 1e9:.3f} GB")
        return "\n".join([
            f"read      {self.read_bytes / 1e9:.3f} GB over {self.launches} launches",
            f"write     {write}",
            f"route     {self.route}",
            f"windows   {self.resolution.text()}",
            f"duty      {self.duty_cycle * 100:.1f}% of the profiled span was in "
            "one of our kernels",
            f"fill      {self.fill * 100:.1f}% of the MERGED window was kernel "
            "rather than sub-period gap",
            f"boundary  {self.n_boundary_samples} of {self.n_samples} samples sat "
            "within one period of a window edge",
            f"idle      {self.idle_read_bytes_per_s / 1e9:.1f} GB/s read while "
            "nothing of ours was running",
        ])


def longest_window(windows) -> tuple[Window, ...]:
    """The single longest merged window, which is the measured loop by design.

    The workloads in `scripts/nsys_dram_probe.py` allocate, warm up, go IDLE for
    a beat and only then run the timed loop back to back, so the longest
    contiguous run of kernels in the report IS the measurement and the setup
    kernels are somewhere else. Selecting by name would be more obviously
    correct and is less robust: torch's reduction and CUTLASS grouped-GEMM
    kernel names differ across versions, and a pattern that silently matches
    nothing is exactly the failure this module is built to make impossible.
    """
    if not windows:
        return ()
    return (max(windows, key=lambda w: w.duration_ns),)


def dram_traffic(conn: sqlite3.Connection, kernel_pattern: str | None = None,
                 sample_hz: float = DEFAULT_SAMPLE_HZ,
                 peak_bytes_per_s: float | None = None,
                 allow_unresolved: bool = False,
                 windows: tuple[Window, ...] | None = None) -> DramTraffic:
    """DRAM read and write bytes attributable to the matched kernels.

    Refuses by default when the windows cannot support a number, which is the
    common case for a single short launch and is the finding rather than an
    inconvenience. `allow_unresolved=True` returns it anyway with `ok=False` on
    the resolution, for a caller that is characterising the sampler itself.

    `windows` overrides which merged windows the numerator sums over, for a
    caller that has already selected them (see `longest_window`). The IDLE
    BASELINE is computed against every kernel in the report either way, so
    narrowing the numerator can never move time into the baseline that had a
    kernel running in it.
    """
    period = sample_period_ns(sample_hz)
    matched = kernel_windows(conn, kernel_pattern)
    all_kernels = kernel_windows(conn, None)
    merged = windows if windows is not None else merge_windows(matched, period)
    launches = (sum(w.n_launches for w in merged) if windows is not None
                else len(matched))
    total_ns = sum(w.duration_ns for w in merged)
    res = resolve(total_ns, len(merged), sample_hz)
    if not res.ok and not allow_unresolved:
        raise WindowTooShort(
            res.reason
            + f"\nThe matched kernels ran {len(matched)} launch(es) totalling "
              f"{total_ns / 1000:.1f} us, merging into {len(merged)} window(s) at "
              f"a {period / 1000:.1f} us sample period.\n"
              "A single short launch is not measurable at any rate nsys offers: "
              f"a {SHORT_KERNEL_US:.0f} us kernel holds "
              f"{SHORT_KERNEL_US * 1000 / sample_period_ns(MAX_SAMPLE_HZ):.1f} "
              f"samples even at the {MAX_SAMPLE_HZ / 1e3:.0f} kHz ceiling. "
              "Launch it back to back so the windows merge into one.")

    metrics = find_dram_metrics(metric_catalogue(conn))
    unit = metrics.read.unit_kind
    read_sum = sum_samples(conn, metrics.read, merged, period)
    read_bytes = to_bytes(unit, read_sum.total, read_sum.n_samples, period,
                          peak_bytes_per_s, metrics.read.scale)
    write_bytes = None
    if metrics.write is not None:
        w = sum_samples(conn, metrics.write, merged, period)
        write_bytes = to_bytes(metrics.write.unit_kind, w.total, w.n_samples,
                               period, peak_bytes_per_s, metrics.write.scale)

    outside = sum_samples_outside(conn, metrics.read, merge_windows(all_kernels, period),
                                  period)
    idle_bytes = (to_bytes(unit, outside.total, outside.n_samples, period,
                           peak_bytes_per_s, metrics.read.scale)
                  if outside.n_samples else 0.0)
    idle_seconds = outside.n_samples * period / 1e9
    route = (f"{unit.value} via {metrics.read.name!r}"
             + (f" and {metrics.write.name!r}" if metrics.write else ""))
    return DramTraffic(
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        resolution=res,
        windows=tuple(merged),
        launches=launches,
        duty_cycle=duty_cycle(matched),
        fill=fill_fraction(matched, merged),
        idle_read_bytes_per_s=idle_bytes / idle_seconds if idle_seconds > 0 else 0.0,
        route=route,
        n_samples=read_sum.n_samples,
        n_boundary_samples=read_sum.n_boundary,
    )


# --------------------------------------------------------------------------
# Against the byte model. The output of this module is never a byte count on its
# own: the study's question is whether the COMPULSORY MODEL is right, so the
# answer has the modelled figure beside the measured one or it is not an answer.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrafficComparison:
    """Modelled X, measured Y, ratio Z, and the band the ratio carries."""

    modelled_bytes: float
    measured_bytes: float
    launches: int
    ratio: float
    ratio_low: float
    ratio_high: float
    #: Everything in the error budget, named, so a reader can see which term
    #: dominates rather than being handed one interval.
    terms: tuple[tuple[str, float], ...]
    verdict: str

    def text(self) -> str:
        budget = ", ".join(f"{name} {v * 100:.1f}%" for name, v in self.terms)
        return "\n".join([
            f"modelled  {self.modelled_bytes / 1e9:.3f} GB "
            f"({self.modelled_bytes / max(self.launches, 1) / 1e6:.2f} MB per launch)",
            f"measured  {self.measured_bytes / 1e9:.3f} GB",
            f"ratio     {self.ratio:.3f}  [{self.ratio_low:.3f}, {self.ratio_high:.3f}]",
            f"budget    {budget}",
            f"verdict   {self.verdict}",
        ])


def compare_to_model(traffic: DramTraffic, modelled_bytes_per_launch: float,
                     launches: int | None = None,
                     extra_terms: tuple[tuple[str, float], ...] = ()
                     ) -> TrafficComparison:
    """`measured / modelled`, with the sampler's own uncertainty attached.

    The error budget is three named terms and no hidden fourth:

      * `edge`, the window quantisation from `Resolution`;
      * `idle`, the fraction of the measured signal that the same sampler saw
        while none of our kernels were running, which bounds contamination by
        anything else resident on a device-wide counter;
      * whatever the caller adds, which is where a percent-of-peak route puts
        the calibration's uncertainty.

    A ratio near 1.0 says the compulsory model counts the bytes the hardware
    moved. Above 1.0 is re-read, which is what alpha is. Below 1.0 with a band
    that excludes 1.0 means the model or the attribution is wrong, and on a
    device-wide sampler the attribution is the thing to suspect first.
    """
    n = launches if launches is not None else traffic.launches
    modelled = modelled_bytes_per_launch * n
    if modelled <= 0:
        raise ValueError("modelled bytes must be positive: a zero model would "
                         "make every measurement an infinite ratio")
    ratio = traffic.total_bytes / modelled
    idle_share = 0.0
    if traffic.total_bytes > 0 and traffic.resolution.window_ns > 0:
        idle_bytes = traffic.idle_read_bytes_per_s * traffic.resolution.window_ns / 1e9
        idle_share = min(1.0, idle_bytes / traffic.total_bytes)
    #: `fill` is the only ONE-SIDED term here: a merged window longer than the
    #: kernel time inside it charges the kernel for gaps it did not run in, which
    #: always biases the ratio DOWN. It is put in the symmetric budget anyway,
    #: because widening a band is conservative where narrowing one is not, and
    #: the direction is stated in the verdict.
    terms = (("edge", traffic.resolution.edge_error), ("idle", idle_share),
             ("fill", max(0.0, 1.0 - traffic.fill))) + extra_terms
    rel = math.sqrt(sum(v * v for _, v in terms))
    low, high = ratio * (1 - rel), ratio * (1 + rel)
    if not traffic.resolution.ok:
        verdict = "UNRESOLVED: the windows do not support this number"
    elif low <= 1.0 <= high:
        verdict = ("consistent with the compulsory model: the band spans 1.0, so "
                   "this cannot see re-read at all")
    elif low > 1.0:
        verdict = (f"above the compulsory floor by {(low - 1) * 100:.0f}% at the "
                   "low end, which is traffic the model does not account for")
    else:
        verdict = ("BELOW the compulsory floor, which is impossible for a correct "
                   "model and correct attribution: suspect the attribution first, "
                   "since a device-wide sampler over a low duty cycle undercounts")
    if traffic.fill < 0.95:
        verdict += (f". The merged window was only {traffic.fill * 100:.0f}% "
                    "kernel, and the gaps are charged to it, so the ratio is "
                    "biased DOWN by the remainder")
    if traffic.write_bytes is None:
        verdict += (". MEASURED READS ONLY: this metric set does not split "
                    "writes, while the compulsory model counts them, so the "
                    "ratio is a LOWER bound")
    return TrafficComparison(modelled, traffic.total_bytes, n, ratio, low, high,
                             terms, verdict)


def calibration_verdict(comparison: TrafficComparison, tolerance: float = 0.15) -> str:
    """Did a KNOWN-traffic workload come back at the traffic it is known to move?

    The self-check that has to pass before any MoE number from the same session
    is worth reading. A streaming read of a buffer far larger than L2 moves
    almost exactly its own size, so the measured-over-known ratio is a direct
    read on the sampler, the unit interpretation and the attribution all at
    once. It is the one place in this module where the modelled figure is not in
    question.
    """
    if not comparison.terms:
        return "no error budget"
    if abs(comparison.ratio - 1.0) <= tolerance:
        return (f"PASS: {comparison.ratio:.3f} of known traffic, within "
                f"{tolerance * 100:.0f}%. The unit interpretation and the window "
                "attribution both survive a case with a known answer.")
    return (f"FAIL: {comparison.ratio:.3f} of known traffic. Nothing measured in "
            "this session should be quoted. The likely causes in order are a "
            "wrong unit interpretation, a window that caught other work, and a "
            "buffer small enough to have been served by L2.")
