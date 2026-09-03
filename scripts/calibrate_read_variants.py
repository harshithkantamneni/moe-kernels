#!/usr/bin/env python
"""C4: is our read ceiling a ceiling, or just what one formulation achieved?

    python scripts/calibrate_read_variants.py
    python scripts/calibrate_read_variants.py --gib 8 --target-ms 300
    python scripts/calibrate_read_variants.py --self-test 1.03     (off GPU)

THE PROBLEM. `calibrate_hardware.py` measured the `read` pattern with
`torch.sum(a, dim=0, out=sink)`, and its own docstring already flagged it: "read
here is a torch.sum tree reduction". A reduction is not a pure streaming read.
It combines partial results across blocks, which costs traffic and
synchronisation that a weight stream does not pay.

That matters because 83 rows of the 2026-08-26 sweep imply a bandwidth ABOVE
that ceiling. Peak 4483.4 GB/s against a measured read of 4389.4. Nothing
exceeded the 4916.7 GB/s pin rate, so nothing impossible happened; the likely
reading is that the ceiling is too low, in which case every percent-of-ceiling
figure in the study is pessimistic by that margin, and so is anyone else's
computed the same way.

Every variant below reads the same buffer exactly once per iteration. They differ
only in what they do with the values, which is the part that must be cheap enough
not to matter and expensive enough that the compiler cannot delete the load.

THE LOOP THAT USED TO BE HERE
-----------------------------
This file carried the last private timing loop in the repository, and it was the
A7 shape exactly: a `torch.cuda.Event` pair created INSIDE the loop, a start
event recorded on a stream the previous iteration's `synchronize()` had just
drained, a `synchronize()` after every single iteration, five isolated warmup
calls, and no clock read at all. `moe/bench/timing.py` names what that costs:
the host's own enqueue sits inside the interval when the queue is empty, and a
loop that settles with a synchronise every few kernels converges to a
partial-load plateau, MEASURED at 1575 MHz where the real measurement induces
1980. The margin this script is here to resolve is 4483.4 / 4389.4 = 2.14%. A
governor sitting 20% low, and no column that would say so, is not an apparatus
that can resolve 2.14%.

It measures through `timing.time_kernel` now: queue-deep, one synchronise per
trial, L2 flushed before each timed call, warmup expressed as a DURATION of
delivered GPU load, the SM clock polled from a background thread WHILE the
trials run, and three verdicts per row (LEVEL, DRIFT, host-bound) that this
file scores rather than prints. `--iters` and `--warmup` are gone with the loop:
a call count does not convert into a duration, which is why `time_call` refuses
rather than wrapping.

THE COMPARISON IS WITHIN-INSTRUMENT, AND THE DECOMPOSITION IS THE POINT
----------------------------------------------------------------------
The old verdict compared this loop's GB/s against two constants produced by a
DIFFERENT loop (`calibrate.measure_bandwidth` via `time_eager`), at a margin of
2.14%, on an apparatus whose own audit bounded the cross-instrument bias at
8-16%. The 2026-08-27 pod run shows the confound in its own table
(`results/published/2026-08-26-nvidia_h200-full-three-way/FINDINGS.md`, the
five-row table under "MEASURED, 2026-08-27"): the formulation `calibrate.py`
itself used read 4463.0 GB/s in THIS loop and 4389.4 in the calibration's,
1.7% apart on ONE formulation, which is most of the 2.14% under test. The old
verdict would have called that 1.020x a formulation result.

So the same formulation is measured here as a baseline, and the total is
reported as a product of two terms that are each attributable:

    best_here / registered_read  =  (best_here / baseline_here)   FORMULATION
                                 x (baseline_here / registered_read)  INSTRUMENT

The formulation term is a ratio of two rows from one run under one instrument,
and it is what C4 asks about. The instrument term compares two loops and is
scored separately, only when the clock LEVEL says this run is comparable with
the roof at all. A run that cannot tell them apart says so instead of
attributing the difference to whichever of the two the author had in mind.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# `scripts/` is not a package, and `block_m_crossing_sweep` is where the shared
# reference-clock resolution and the synthetic instrument label already live.
# The same two-line idiom `tile_cap_test.py` uses, for the same reason: copying
# either of them would put two answers in the tree for one question.
sys.path.insert(0, str(HERE))

import block_m_crossing_sweep as SWEEP  # noqa: E402
import torch  # noqa: E402

from moe.bench import exit_codes as EX  # noqa: E402
from moe.bench import provenance as PV  # noqa: E402
from moe.bench import timing  # noqa: E402

#: What the 2026-08-26 sweep's fastest row implied, in GB/s. The number C4 has
#: to be resolved against.
ANOMALY_GBPS = 4483.4

#: What `calibrate_hardware.py` reported for `read` in that same session, before
#: the reduction shape was fixed. MEASURED BY `time_eager`, not by this file's
#: instrument, which is why it is only ever one end of the INSTRUMENT term and
#: never the denominator of a formulation ratio. The committed calibration for
#: the attached card is preferred over it when there is one, and the report
#: names which was used.
REGISTERED_READ_GBPS = 4389.4

#: The margin C4 is asked to explain: 4483.4 / 4389.4. A formulation that buys
#: less than this does not dissolve the anomaly on its own.
ANOMALY_MARGIN = ANOMALY_GBPS / REGISTERED_READ_GBPS

#: Theoretical pin rates, per card, from the committed calibrations'
#: `observed.pin_rate_gbps` (`moe/bench/hardware/measured_nvidia_h200.yaml` and
#: `measured_nvidia_a100_sxm4_80gb.yaml`), the same table and the same reason as
#: `ruler_rebaseline.PIN_RATE_GBPS`: a card that is not here gets NO pin-rate
#: guard and the gate is UNKNOWN rather than passed, because an elided-loads
#: check run against another card's pin rate is not a loose check, it is an
#: inert one. On the A100 the H200's 4916.7 would pass a read reporting 2.4x
#: that card's entire bus.
PIN_RATE_GBPS = {"nvidiah200": 4916.7, "nvidiaa100sxm480gb": 2039.0}

#: How far the two instruments' figures for ONE formulation may sit apart before
#: the instrument term is a finding rather than noise. `ruler_rebaseline`'s
#: gate 2 uses the same 0.5% for pattern reproduction, on the evidence that the
#: two closest H200 calibrations differ by 0.06%.
INSTRUMENT_TOL_PCT = 0.5

#: The formulation `calibrate.py` used, measured here so the formulation ratio
#: has a denominator from this run and this instrument. Must be a key of
#: `VARIANT_NAMES`.
BASELINE = "torch.sum(dim=0)"

#: Every formulation, in the order the report prints them. The first is the
#: baseline; `count_nonzero` is in the set because it is the one that lands an
#: order of magnitude off, which is how a reader can see the set is not rigged.
VARIANT_NAMES = (BASELINE, "torch.sum(dim=1)", "a.sum()", "a.amax()",
                 "a.count_nonzero()")

#: What each formulation is doing, printed beside it.
VARIANT_NOTES = {
    BASELINE: "what calibrate.py used when the anomaly was found",
    "torch.sum(dim=1)": "reduces along the contiguous axis",
    "a.sum()": "full reduction to one scalar",
    "a.amax()": "reduction with no accumulation width",
    "a.count_nonzero()": "integer reduction",
}


def _device_key(name: str) -> str:
    """Lowercase alphanumerics, the normalisation `ruler_rebaseline` keys on."""
    return "".join(c for c in name.lower() if c.isalnum())


@dataclass(frozen=True)
class Reading:
    """One formulation's rate, and the state the instrument says it ran in.

    Flattened out of `timing.KernelTiming` on purpose: the gates below are the
    part worth testing off GPU, and a gate that takes a `KernelTiming` can only
    be tested by building one. `level_ok`, `drift_ok` and `host_bound` are the
    three verdicts `time_kernel` returns, and None means NOT DETERMINED in all
    three, never False. `instrument` travels with the row so a planted reading
    can never be mistaken for a measured one.
    """

    name: str
    gbps: float
    ms_p50: float
    instrument: str
    host_bound: bool | None = None
    drift_ok: bool | None = None
    level_ok: bool | None = None
    clock_mhz: float | None = None
    note: str = ""

    @classmethod
    def from_timing(cls, name: str, nbytes: int, kt: timing.KernelTiming,
                    note: str = "") -> Reading:
        return cls(name=name, gbps=nbytes / (kt.ms_p50 * 1e-3) / 1e9,
                   ms_p50=kt.ms_p50, instrument=kt.instrument,
                   host_bound=kt.host_bound, drift_ok=kt.clock_drift_ok,
                   level_ok=kt.clock_level_ok, clock_mhz=kt.sm_clock_load_mhz,
                   note=note or kt.clock_note)


def pick(readings: list[Reading], name: str) -> Reading | None:
    for r in readings:
        if r.name == name:
            return r
    return None


def variants(a: torch.Tensor, sink_row: torch.Tensor, sink_scalar: torch.Tensor):
    """Formulations that each read every element of `a` exactly once.

    None of these is a clever kernel. They are the obvious ways to express
    "touch all of it" in torch, and the point is how far apart they land.
    """
    return [
        (BASELINE, lambda: torch.sum(a, dim=0, out=sink_row)),
        ("torch.sum(dim=1)", lambda: torch.sum(a, dim=1, out=sink_scalar)),
        ("a.sum()", lambda: torch.sum(a)),
        ("a.amax()", lambda: torch.amax(a)),
        ("a.count_nonzero()", lambda: torch.count_nonzero(a)),
    ]


def registered_read(gpu_name: str) -> tuple[float, str]:
    """The read figure this run's INSTRUMENT term is measured against, and where.

    The committed calibration for the attached card first, because that is the
    denominator the published efficiency columns actually use today, and it has
    moved: the H200's `read` reads 4469.6 GB/s in
    `moe/bench/hardware/measured_nvidia_h200.yaml` since the reduction shape was
    fixed, against the 4389.4 the anomaly was computed against. Falls back to
    that historical constant, named as historical, when the card has no
    calibration here. Either way the figure was measured by `time_eager` and the
    sentence says so, because the whole point of the term is that two loops are
    being compared.

    Reads through `block_m_crossing_sweep._measured_detail`, private and
    borrowed on purpose: it is the same candidate order `roofline.load_measured`
    uses AND it re-checks the device name inside the file, so this cannot end up
    quoting another machine's ceiling. A local re-implementation would be a
    second answer to one question.
    """
    detail = SWEEP._measured_detail(gpu_name) if gpu_name else {}
    for row in detail.get("bandwidth_patterns") or []:
        if row.get("pattern") == "read" and row.get("gbps"):
            return float(row["gbps"]), (
                f"{gpu_name}: the committed calibration's `read` pattern, "
                "measured by moe.bench.calibrate via timing.time_eager")
    return REGISTERED_READ_GBPS, (
        "the 2026-08-26 session's `read` figure, the one the anomaly was "
        "computed against; no committed calibration for this card, and it was "
        "measured by moe.bench.calibrate via timing.time_eager")


def measure(args, reference_mhz: float | None) -> tuple[list[Reading], str]:
    """Time every formulation through the one instrument. Returns the rows and a header.

    The buffer is allocated once and every variant reads it, so the formulation
    ratio is not confounded by allocation or by page state. `time_kernel` owns
    the warmup, the iteration count, the flush and the clock; nothing here
    reimplements any of them, which is the whole of the A7 fix.
    """
    nbytes = int(args.gib * (1 << 30))
    rows = 4096
    cols = nbytes // (rows * 4)
    a = torch.empty((rows, cols), dtype=torch.float32, device="cuda")
    a.uniform_(0.0, 1.0)
    real_bytes = a.numel() * a.element_size()
    sink_row = torch.empty(cols, dtype=torch.float32, device="cuda")
    sink_scalar = torch.empty(rows, dtype=torch.float32, device="cuda")

    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", 0)
    header = (f"device       {props.name}\n"
              f"buffer       {real_bytes / (1 << 30):.2f} GiB "
              f"({real_bytes / max(l2, 1):.0f}x L2)\n"
              f"flush        {'off' if args.no_flush else 'on'}   "
              f"warmup {args.warmup_ms:.0f} ms, target {args.target_ms:.0f} ms "
              f"per trial, {args.trials} trials")

    out = []
    for name, fn in variants(a, sink_row, sink_scalar):
        kt = timing.time_kernel(fn, warmup_ms=args.warmup_ms,
                                target_ms=args.target_ms, trials=args.trials,
                                l2_flush=not args.no_flush,
                                reference_clock_mhz=reference_mhz)
        out.append(Reading.from_timing(name, real_bytes, kt))
    return out, header


def plant(ratio: float, defect: str) -> list[Reading]:
    """A world with a KNOWN formulation ratio, so every branch below is reachable.

    The rates are the 2026-08-27 pod table's shape rather than round numbers, so
    a reader can see which world is planted; `ratio` scales the best formulation
    against the baseline and is the quantity C1 is scored on. `defect` plants the
    FAIL branch of each validity flag one at a time, because a gate that has
    only ever been seen to PASS has not been shown to be a gate. Every planted
    row carries `SWEEP.SYNTHETIC_INSTRUMENT`, never `TIMING_BASIS`.
    """
    base = 4463.0
    rates = {BASELINE: base, "torch.sum(dim=1)": base * ratio,
             "a.sum()": base * 0.982, "a.amax()": base * 0.981,
             "a.count_nonzero()": base * 0.152}
    if defect == "pin":
        rates["torch.sum(dim=1)"] = PIN_RATE_GBPS["nvidiah200"] * 1.01
    flags = {"host_bound": False, "drift_ok": True, "level_ok": True}
    if defect == "host-bound":
        flags["host_bound"] = True
    elif defect == "drift":
        flags["drift_ok"] = False
    elif defect == "level":
        flags["level_ok"] = False
    elif defect == "unlevelled":
        flags["level_ok"] = None
    return [Reading(name=name, gbps=gbps, ms_p50=8.0 * (1 << 30) / (gbps * 1e6),
                    instrument=SWEEP.SYNTHETIC_INSTRUMENT, clock_mhz=1980.0,
                    note=f"planted world, ratio {ratio:.4f}, defect {defect}",
                    **flags)
            for name, gbps in rates.items()]


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------

def score(readings: list[Reading], gpu_name: str,
          registered_gbps: float) -> list[tuple[str, str, str, str]]:
    """Four gates: two on whether the rows may be read at all, two on C4.

    V1 pin_rate      no formulation exceeds this card's bus. UNKNOWN for a card
                     with no registered pin rate, because the guard would be
                     inert rather than loose.
    V2 within_run    the two rows the formulation ratio is taken from are not
                     host-bound and did not drift. This is what makes a RATIO of
                     two rows from one run meaningful, and it needs no clock
                     level: a throttled card slows both ends of it together.
    C1 formulation   does re-formulating alone buy at least the 2.14% the
                     anomaly needs. This is the C4 question, and it is scored
                     entirely within one instrument.
    C2 instrument    does the registered `read` figure reproduce here, on the
                     SAME formulation, to within 0.5%. UNKNOWN unless the clock
                     LEVEL says this run is comparable with the roof, because
                     the two figures come from two loops and a throttled card
                     would be reported as an instrument difference.

    Returned as `(kind, name, verdict, detail)` tuples so `EX.classify` scores
    them and `EX.result_line` prints them: the exit code is the shared table's,
    never this file's opinion.
    """
    gates = []
    key = _device_key(gpu_name)
    pin = PIN_RATE_GBPS.get(key)
    best = max(readings, key=lambda r: r.gbps)
    baseline = pick(readings, BASELINE)

    if pin is None:
        gates.append((EX.VALIDITY, "pin_rate", EX.UNKNOWN,
                      f"no registered pin rate for {gpu_name or 'this device'} "
                      f"({key or 'no name'}); the elided-loads guard would be "
                      "inert, so it is not claimed"))
    elif best.gbps > pin:
        gates.append((EX.VALIDITY, "pin_rate", EX.FAIL,
                      f"{best.name} reads {best.gbps:.1f} GB/s above the "
                      f"{pin:.1f} GB/s bus: the byte count is wrong or the "
                      "compiler deleted the load"))
    else:
        gates.append((EX.VALIDITY, "pin_rate", EX.PASS,
                      f"best {best.gbps:.1f} GB/s is {100 * best.gbps / pin:.1f}% "
                      f"of the {pin:.1f} GB/s bus"))

    pair = [r for r in (baseline, best) if r is not None]
    if baseline is None:
        gates.append((EX.VALIDITY, "within_run", EX.UNKNOWN,
                      f"no {BASELINE} row, so the formulation ratio has no "
                      "denominator from this run"))
    elif any(r.host_bound for r in pair):
        named = ", ".join(r.name for r in pair if r.host_bound)
        gates.append((EX.VALIDITY, "within_run", EX.FAIL,
                      f"host-bound: {named}; the queue drained and the "
                      "intervals carry host enqueue time, so these are upper "
                      "bounds on the kernel and not rates"))
    elif any(r.drift_ok is False for r in pair):
        named = ", ".join(r.name for r in pair if r.drift_ok is False)
        gates.append((EX.VALIDITY, "within_run", EX.FAIL,
                      f"the clock moved during {named}, so the two ends of the "
                      "ratio were not measured at one operating point"))
    elif any(r.drift_ok is None for r in pair):
        gates.append((EX.VALIDITY, "within_run", EX.UNKNOWN,
                      "no clock samples landed, so nothing says the two ends of "
                      "the ratio were measured at one operating point"))
    else:
        gates.append((EX.VALIDITY, "within_run", EX.PASS,
                      f"{baseline.name} and {best.name} are queue-deep and did "
                      "not drift"))

    if baseline is None:
        gates.append((EX.CLAIM, "C4_formulation", EX.UNKNOWN,
                      "not scored without a baseline row"))
    else:
        ratio = best.gbps / baseline.gbps
        detail = (f"{best.name} / {BASELINE} = {ratio:.4f} against the "
                  f"{ANOMALY_MARGIN:.4f} the anomaly needs "
                  f"({ANOMALY_GBPS:.1f} / {REGISTERED_READ_GBPS:.1f})")
        gates.append((EX.CLAIM, "C4_formulation",
                      EX.PASS if ratio >= ANOMALY_MARGIN else EX.FAIL, detail))

    if baseline is None:
        gates.append((EX.CLAIM, "C4_instrument", EX.UNKNOWN,
                      "not scored without a baseline row"))
    elif baseline.level_ok is not True:
        why = ("the loaded clock was below the level the roof was measured at"
               if baseline.level_ok is False else
               "no reference clock, so LEVEL was not determined")
        gates.append((EX.CLAIM, "C4_instrument", EX.UNKNOWN,
                      f"{why}; two loops cannot be compared across a clock "
                      "nobody checked"))
    else:
        delta = 100 * (baseline.gbps / registered_gbps - 1)
        detail = (f"{BASELINE} reads {baseline.gbps:.1f} GB/s here against the "
                  f"registered {registered_gbps:.1f}, {delta:+.2f}% on one "
                  f"formulation, tolerance {INSTRUMENT_TOL_PCT:.1f}%")
        gates.append((EX.CLAIM, "C4_instrument",
                      EX.PASS if abs(delta) <= INSTRUMENT_TOL_PCT else EX.FAIL,
                      detail))
    return gates


def report(readings: list[Reading], gates, registered_gbps: float,
           registered_source: str) -> list[str]:
    """The table, the decomposition, and what each term means. No verdict prose.

    The two terms are printed as a product because that is the only form in
    which the total is attributable: the old report printed the total alone and
    the reader had to assume the whole of it was formulation.
    """
    lines = [f"{'variant':34} {'GB/s':>9} {'vs baseline':>12} {'clock':>8}  flags",
             "-" * 86]
    baseline = pick(readings, BASELINE)
    for r in readings:
        rel = f"{r.gbps / baseline.gbps:11.4f}x" if baseline else " " * 12
        clock = f"{r.clock_mhz:7.0f}" if r.clock_mhz else "      -"
        flags = ",".join(filter(None, [
            "HOST-BOUND" if r.host_bound else "",
            "LEVEL-BAD" if r.level_ok is False else "",
            "LEVEL-?" if r.level_ok is None else "",
            "DRIFT" if r.drift_ok is False else "",
        ])) or "ok"
        note = VARIANT_NOTES.get(r.name, "")
        lines.append(f"{r.name:34} {r.gbps:9.1f} {rel} {clock}  {flags}"
                     + (f"   [{note}]" if note else ""))

    best = max(readings, key=lambda r: r.gbps)
    lines += ["", "THE TWO TERMS", ""]
    if baseline:
        formulation = best.gbps / baseline.gbps
        instrument = baseline.gbps / registered_gbps
        lines += [
            f"  formulation  {formulation:.4f}x   {best.name} over {BASELINE}, "
            "both rows from this run and this instrument",
            f"  instrument   {instrument:.4f}x   {BASELINE} here over the "
            f"registered {registered_gbps:.1f} GB/s",
            f"               {registered_source}",
            f"  total        {formulation * instrument:.4f}x   against the "
            f"{ANOMALY_MARGIN:.4f} the anomaly needs",
            "",
            "  The anomaly is explained by whichever term carries it. Only the",
            "  first is a statement about formulations; the second is two loops",
            "  disagreeing about one formulation, which is a ruler defect and",
            "  not a kernel finding.",
        ]
    lines += ["", "## Gates", ""]
    lines += [EX.result_line(kind, name, verdict, detail)
              for kind, name, verdict, detail in gates]
    return lines


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gib", type=float, default=8.0,
                    help="buffer size; must be far larger than L2 (default 8)")
    # A DURATION, NOT A CALL COUNT. `time_kernel` warms until that much GPU work
    # has been delivered under sustained load and derives its own iteration
    # count from the per-call time it saw. The retired loop's `--iters 50` and
    # `--warmup 5` have no conversion into either.
    ap.add_argument("--warmup-ms", type=float, default=200.0,
                    help="GPU load delivered before timing (default 200)")
    ap.add_argument("--target-ms", type=float, default=200.0,
                    help="duration of one trial (default 200)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--no-flush", action="store_true",
                    help="do not flush L2 between timed calls")
    ap.add_argument("--reference-clock-mhz", type=float, default=None,
                    help="the clock the roof was measured at; read from this "
                         "card's calibration when not given")
    ap.add_argument("--self-test", type=float, default=None, metavar="RATIO",
                    help="plant a world whose best formulation is RATIO times "
                         "the baseline and score it off GPU")
    ap.add_argument("--self-test-defect", default="none",
                    choices=("none", "pin", "host-bound", "drift", "level",
                             "unlevelled"),
                    help="plant the FAIL branch of one validity flag")
    return ap


def _main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.reference_clock_mhz is not None and args.reference_clock_mhz <= 0:
        # A REFUSAL AND NOT A FALLBACK. A non-positive reference silently
        # falling back to the calibration's would make LEVEL a verdict about
        # a clock the operator did not ask for, in a run whose whole subject
        # is which apparatus produced which number.
        print(f"REFUSED: --reference-clock-mhz {args.reference_clock_mhz} "
              "is not a clock. LEVEL is scored against it, so a run that "
              "quietly used a different one would report a comparability "
              "verdict nobody asked for. Nothing measured.")
        return EX.REFUSED
    if args.self_test is not None:
        # A PLANTED WORLD NAMES ITSELF. The rows carry SYNTHETIC_INSTRUMENT and
        # the banner is printed before the table, so a self-test transcript
        # cannot be quoted as a measurement of a card.
        print(f"SYNTHETIC. Planted world, best/baseline = {args.self_test:.4f}, "
              f"defect {args.self_test_defect}. Nothing was measured; every "
              f"row below carries {SWEEP.SYNTHETIC_INSTRUMENT}.\n")
        readings = plant(args.self_test, args.self_test_defect)
        gpu_name, registered, source = "NVIDIA H200", REGISTERED_READ_GBPS, (
            "the 2026-08-26 session's `read` figure, PINNED for --self-test so "
            "the replay is identical on every machine")
    else:
        if not torch.cuda.is_available():
            print("REFUSED: no CUDA device. This script measures a card's read "
                  "ceiling and there is nothing here to measure; it does not "
                  "have an off-GPU mode that would produce a number. Use "
                  "--self-test to exercise the gates. Nothing measured.")
            return EX.REFUSED
        gpu_name = torch.cuda.get_device_properties(0).name
        reference, clock_source = (
            (args.reference_clock_mhz, "given by --reference-clock-mhz")
            if args.reference_clock_mhz is not None
            else SWEEP.reference_clock_mhz(gpu_name))
        print(f"reference clock  {reference if reference else 'NOT KNOWN'}"
              f"  ({clock_source})")
        readings, header = measure(args, reference)
        print(header + "\n")
        registered, source = registered_read(gpu_name)

    gates = score(readings, gpu_name, registered)
    print("\n".join(report(readings, gates, registered, source)))

    # THE SHARED BLOCK, PRINTED RATHER THAN WRITTEN. This script's artefact is
    # its transcript: it writes no report.json, so the instrument, the commit
    # and the driver are printed or they are nowhere. The instrument named here
    # is the one that timed these rows and never the other one, which is the
    # whole subject of the file.
    prov = PV.provenance_block(
        instrument=(SWEEP.SYNTHETIC_INSTRUMENT if args.self_test is not None
                    else timing.TIMING_BASIS),
        bandwidth=round(max(r.gbps for r in readings), 1),
        bandwidth_source=("planted" if args.self_test is not None
                          else "best formulation measured here"))
    print(f"\ninstrument   {prov.instrument}")
    print(f"commit       {prov.git_sha} (dirty {prov.git_dirty})")
    print(f"driver/cuda  {prov.driver_version} / {prov.cuda_version}")

    rc = EX.classify([(k, v) for k, _n, v, _d in gates])
    print(f"exit {EX.describe(rc)}")
    return rc


def main(argv=None) -> int:
    """AN UNPLANNED CRASH IS ERROR (4), which is the only retryable code.

    Left to propagate, an unexpected exception exits the interpreter ONE, and
    ONE is CLAIM_FAIL, which `moe/bench/exit_codes.py` defines as a RESULT: it
    is in FINISHED_CODES, so a caller files the arm as finished and never
    retries it. A torch OOM or a drifted import would be published as one of
    this experiment's registered outcomes. The traceback is printed first and
    not swallowed, because a code without one tells an operator nothing about
    what to fix.

    `SystemExit` is a `BaseException` and passes through untouched: a refusal is
    not a crash, and every refusal in this file returns `EX.REFUSED` anyway.
    """
    try:
        return _main(argv)
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        print("ERROR: calibrate_read_variants crashed before it could reach a "
              "verdict. This is the apparatus failing, not a claim failing, so "
              f"it exits {EX.ERROR} and not {EX.CLAIM_FAIL}: the traceback "
              "above is the thing to fix, and the arm may be re-run.",
              file=sys.stderr)
        return EX.ERROR


if __name__ == "__main__":
    sys.exit(main())
