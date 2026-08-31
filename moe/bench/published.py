"""Which published result arms an analysis should read.

`results/published/` accumulates arms, and some of them are REPLACEMENTS rather
than additions. The 2026-08-26 three-way sweep has a `-recalibrated` twin whose
17,640 ms_p50 values are identical to the original's; only ceiling columns
differ. Reading both weights every one of those rows twice.

That is not hypothetical. Pointed at both, mixtral's torch_grouped_mm_up crossing
reported 1.46x of prediction; pointed at one, 0.63x. Neither run announced which
file set it had used.

THOSE TWO NUMBERS NO LONGER REPRODUCE, and the mechanism still does. Re-checked
2026-08-31 on the current code: the canonical pool gives 938 tokens (1.46x) and
adding the superseded arm gives 936 (1.46x), so double-counting this particular
crossing now moves it by 0.2%. The saturation floor in `crossing_from_points` and
the untimed-row filter in `timed_rows` both landed after the original
observation, and between them they removed the points that made it swing. Read
the anecdote as the reason this module exists, not as a check to run: a
duplicated arm still doubles the weight of every row in it, and the next
analysis it distorts will not be this one.

A marker file inside the superseded directory makes the arm describe its own
status, so the rule lives with the data instead of in every caller's head.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

#: Dropped in a published arm that a later one replaces. Its contents are the
#: human-readable reason, which reports quote when they skip it.
SUPERSEDED_MARKER = "SUPERSEDED"


def is_superseded(csv_path: Path | str) -> bool:
    """Does this file live in an arm that a later one replaced?"""
    return (Path(csv_path).parent / SUPERSEDED_MARKER).exists()


def superseded_reason(csv_path: Path | str) -> str:
    marker = Path(csv_path).parent / SUPERSEDED_MARKER
    try:
        return marker.read_text().strip()
    except OSError:
        return ""


def superseded_impls(csv_path: Path | str) -> set[str] | None:
    """Which implementations in this arm a later one replaced.

    `set()`   nothing is superseded, the arm is current
    `None`    the WHOLE arm is superseded
    `{...}`   only these implementations are

    The middle case is the original one. The third exists because
    `2026-08-28-nvidia_h200-h200-fp8-three-kernel` holds valid vLLM and SGLang
    fp8 rows next to two torch spans that timed a quantisation pass by mistake,
    and `-fp8-refixed` replaces only the torch ones. Retiring the directory
    would discard 10,164 good rows to retract 9,744 bad ones.

    The marker declares it with a line `impls: name, name`. Without that line it
    means the whole arm, so existing markers keep their meaning.
    """
    marker = Path(csv_path).parent / SUPERSEDED_MARKER
    if not marker.exists():
        return set()
    for line in marker.read_text().splitlines():
        if line.lower().startswith("impls:"):
            names = {n.strip() for n in line.split(":", 1)[1].split(",")}
            return {n for n in names if n}
    return None


def filter_superseded(paths) -> tuple[list[Path], list[Path]]:
    """`(kept, dropped)`.

    Both halves are returned so a caller can ANNOUNCE what it dropped. A silently
    discarded input is the same class of error as a silently double-counted one.
    """
    kept, dropped = [], []
    for p in paths:
        # A PARTIALLY superseded arm is kept: its remaining rows are current,
        # and the caller drops the named implementations row by row.
        whole = is_superseded(p) and superseded_impls(p) is None
        (dropped if whole else kept).append(Path(p))
    return kept, dropped


# --- is an arm's calibration its own? -----------------------------------------
#
# The supersession rule above answers "may I read these rows". This one answers
# "against which ruler", and it is a separate question because an arm can be
# perfectly current and still ship a calibration from somebody else's session.
#
# THE BUG. `moe/bench/hardware/measured_<device>.yaml` is one file per device,
# every calibration overwrites it, and `publish_results.sh` copies whatever is in
# it at publish time. `2026-08-28-...-h200-whole-layer` swept 18:08-19:21 UTC,
# 89f9f7a overwrote that file at 19:29, and the arm published at 19:35 with the
# new one. Its measured.yaml is byte-identical (md5 db981ff9) to the one beside
# `-fp8-refixed`, whose sweep began at 19:49, twenty-eight minutes after the
# whole-layer rows stopped, at a different commit. Nothing checked, so nobody
# noticed for three days.
#
# The bill is claim C5. Its target is `2R/b`, which scales with the ridge, so it
# needs the H200's ridge; the whole-layer arm's rows say 160.3 and the file
# beside them says 176.2, and neither is a same-session number. C5 is therefore
# scored against the band 0.81-0.91 rather than against 0.83.

#: A calibration measured in the sweep's own session: the ceilings on the rows
#: are the ones in the file, the commit is one the rows carry, and the date sits
#: inside the sweep.
SAME_SESSION = "same_session"

#: The rows WERE stamped from this file -- the ceilings match to a float -- but
#: the commit or the date says the calibration was measured in some other
#: session. Usually benign and in fact the normal case (see
#: `calibration_provenance`), which is why it is its own verdict and not a
#: failure.
DIFFERENT_SESSION = "different_session"

#: The ceilings stamped on the rows are NOT the ceilings in the file beside
#: them, so the file is not the ruler the efficiency columns were computed
#: against. This is the whole-layer defect, and the only verdict that says the
#: published arm is internally inconsistent.
CEILINGS_DISAGREE = "ceilings_disagree"

#: Nothing recorded to decide with. An answer, not an error.
UNKNOWN = "unknown"

#: Dropped by `recompute_ceilings.py` into an arm it derives. Its first line is
#: the source arm's directory name.
DERIVED_MARKER = "DERIVED_FROM"

#: What `recompute_ceilings.py` has always written into the README of a derived
#: arm. Read as a fallback so the arm already on disk keeps working without
#: anyone editing a published directory -- but only as a fallback, because a
#: README is prose and gets hand-edited (the one in
#: `-full-three-way-recalibrated` already has been).
DERIVED_ATTRIBUTION = "recompute_ceilings.py"

#: How close a row's ceiling must be to the file's to count as the same number.
#: Not equality: the file stores bandwidth as TB/s and the row stores GB/s, so
#: the value makes a divide-by-1000 and a multiply-by-1000 round trip and can
#: come back one ULP away. Not a loose tolerance either: the two closest
#: calibrations of this H200 differ by 0.06%, which is eight orders of magnitude
#: above this, so nothing real is being smoothed over.
CEILING_REL_TOL = 1e-9

#: Columns a row carries that the calibration, and nothing else, determines.
#: `moe.bench.recompute.CEILING_COLUMNS` is the full list; these two are the
#: ones with a directly comparable value in the yaml.
ROW_CEILING_COLUMNS = ("achieved_peak_tflops", "achieved_bw_gbps")


@dataclass(frozen=True)
class CalibrationProvenance:
    """Whether an arm's `measured.yaml` belongs to the sweep that wrote its rows.

    A verdict and its evidence, never a bare bool. The three signals disagree
    with each other in useful ways and a caller has to be able to see which one
    fired: the ceilings are decisive, the commit is nearly always "different"
    even when everything is fine, and the date is too coarse to have caught the
    defect this exists for.
    """
    arm: str
    verdict: str
    #: Every comparison that was made, and the raw values it was made on.
    evidence: dict
    #: The arm this one was derived from, when it declares itself derived.
    #: `recompute_ceilings.py` output legitimately carries a calibration from a
    #: later session -- that is the point of it. The declaration excuses the LATE
    #: DATE and nothing else: a derived arm whose ceilings contradict its own
    #: calibration is a broken derivation, not a declared one.
    derived_from: str | None
    #: One line saying why publishing must stop, or None when it may proceed.
    #: A string rather than a flag so the refusal quotes its own reason.
    blocking_reason: str | None

    def summary(self) -> str:
        """One line for a log, with the declaration if there is one."""
        tail = f" (derived from {self.derived_from})" if self.derived_from else ""
        return f"{self.arm}: {self.verdict}{tail}"


def derived_from(arm: Path | str) -> str | None:
    """Which arm this one was recomputed from, if it says so.

    Two places, newest first. `recompute_ceilings.py` now writes a `DERIVED_FROM`
    marker, matching `SUPERSEDED` so the rule lives with the data rather than in
    prose. Before that it wrote only a README sentence, which is what the one
    derived arm on disk has, so that sentence is still read -- a published
    directory should not have to be edited to be understood.
    """
    arm = Path(arm)
    marker = arm / DERIVED_MARKER
    if marker.exists():
        for line in marker.read_text().splitlines():
            if line.strip():
                return line.strip()
        return arm.name
    readme = arm / "README.md"
    if not readme.exists():
        return None
    text = readme.read_text()
    if DERIVED_ATTRIBUTION not in text:
        return None
    # "Derived from `<arm>` by `scripts/recompute_ceilings.py`", possibly wrapped
    # across lines by a formatter, hence the DOTALL and the loose whitespace.
    match = re.search(r"[Dd]erived\s+from\s+`([^`]+)`\s+by\s+`[^`]*"
                      + re.escape(DERIVED_ATTRIBUTION), text, re.DOTALL)
    return match.group(1) if match else arm.name


def _row_files(arm: Path) -> list[Path]:
    """The per-venv arm CSVs, never merged.csv.

    merged.csv is rebuilt from them and holds every row twice over if both are
    read. `publish_results.sh` rebuilds it for exactly that reason.
    """
    return sorted(arm.glob("run_*.csv"))


def _read_rows(arm: Path) -> list[dict]:
    """Raw `csv.DictReader` rows, deliberately not `schema.read_csv`.

    `read_csv` refuses a schema version outside `READABLE_VERSIONS`, which is
    right for analysis and wrong here: an arm whose schema this code does not
    recognise is exactly the arm whose provenance you most want to read. Nothing
    below needs a coerced value, only `timestamp`, `git_sha`, `gpu_name`,
    `dtype`, `ms_p50` and the two ceiling columns, all of which have been in the
    schema since v1.
    """
    rows: list[dict] = []
    for path in _row_files(arm):
        with path.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _row_float(row: dict, key: str) -> float:
    """A ceiling column off a raw CSV row, or 0.0 when it says nothing.

    Deliberately NOT named `_f`, which is what `recompute.py` calls its own
    float reader two modules away. That one raises on an unparseable value and
    this one does not, and two private helpers with one name and opposite
    behaviour on bad input is how the bool readers drifted before `row_bool`
    collapsed them.

    Tolerant because 0.0 is the ANSWER here, not a failure: a missing or empty
    `achieved_peak_tflops` is exactly the fp8 arm swept against a bf16-only
    calibration, and `entitled_ridge` has to reach its refusal rather than
    raise on the way. Only ever called on v3 columns, so it cannot meet an
    UNRECORDED sentinel; `schema.tile_field` is the reader for the v4 ones and
    it raises on purpose.
    """
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _agree(row_value: float, file_value: float | None) -> bool:
    return bool(file_value) and math.isclose(row_value, file_value,
                                             rel_tol=CEILING_REL_TOL)


def calibration_provenance(arm: Path | str) -> CalibrationProvenance:
    """Decide whether `arm/measured.yaml` is the ruler `arm`'s rows were measured
    against.

    Three comparisons, in descending order of how much they prove.

    THE CEILINGS ARE DECISIVE. `driver.py` stamps `achieved_peak_tflops` and
    `achieved_bw_gbps` onto every timed row out of the calibration it loaded, so
    a row that matches the file to a float was measured against that file and a
    row that does not was not. This is the only signal that caught the
    whole-layer arm: its rows carry 701.61 / 4377.21 and its measured.yaml
    reports 770.92 / 4374.49.

    THE COMMIT IS WEAK, AND MISMATCH IS THE NORMAL CASE. `measured_commit` is
    the commit the calibration RAN at, and the usual workflow is calibrate, then
    commit the yaml, then sweep -- so the sweep runs one commit later by
    construction. Six of the ten published arms differ this way and none of the six
    is defective; two match, and two record no comparable commit. It is compared
    because it is the only field that can distinguish two calibrations taken on
    the same day, and because "matches" is real evidence even though "differs"
    is barely any.

    THE DATE IS TOO COARSE TO HAVE HELPED. `checked_on` has day resolution, and
    the whole-layer swap happened inside one day: the sweep stopped at 19:21 UTC
    and the calibration that replaced it was written by 19:29. It earns its place
    on the other side, where a calibration dated AFTER the last row is causally
    impossible for a same-session measurement, which is what a derived arm looks
    like and what an accidental one would look like too.

    A device disagreement is folded into `ceilings_disagree` rather than given
    its own verdict, because an A100 calibration beside H200 rows disagrees on
    the ceilings by a factor of three; the device is reported as evidence so the
    reason is not mysterious.
    """
    arm = Path(arm)
    evidence: dict = {"arm": arm.name}
    declared = derived_from(arm)

    cal = arm / "measured.yaml"
    if not cal.exists():
        evidence["calibration"] = "absent"
        return CalibrationProvenance(
            arm.name, UNKNOWN, evidence, declared,
            "no measured.yaml beside the rows, so the efficiency columns cannot "
            "be interpreted")

    # Deferred, and it has to stay deferred: calibrate.py imports torch at module
    # scope. `crossing_report.py` reaches this module for `filter_superseded`
    # alone and must keep importing on a box with no torch, so the cost lands on
    # the arms that actually have a calibration to read instead of on import.
    from .calibrate import read_stamp
    stamp = read_stamp(cal)
    evidence["calibration"] = cal.name
    evidence["yaml_gpu"] = stamp.gpu_name
    evidence["yaml_commit"] = stamp.measured_commit
    evidence["yaml_checked_on"] = stamp.checked_on.isoformat() if stamp.checked_on else ""
    evidence["yaml_bandwidth_gbps"] = stamp.bandwidth_gbps
    evidence["yaml_peak_tflops"] = dict(stamp.peak_tflops)

    rows = _read_rows(arm)
    timed = [r for r in rows if _row_float(r, "ms_p50") > 0.0]
    evidence["rows"] = len(rows)
    evidence["timed_rows"] = len(timed)
    if not timed:
        return CalibrationProvenance(
            arm.name, UNKNOWN, evidence, declared,
            "no timed rows, so nothing carries a ceiling to compare the "
            "calibration against")

    evidence["row_commits"] = sorted({r.get("git_sha", "") for r in timed} - {""})
    stamps = sorted({t for t in (r.get("timestamp", "") for r in timed) if t})
    evidence["row_span"] = (stamps[0], stamps[-1]) if stamps else ()
    evidence["row_gpus"] = sorted({r.get("gpu_name", "") for r in timed} - {""})
    evidence["row_dtypes"] = sorted({r.get("dtype", "") for r in timed} - {""})
    evidence["row_bandwidth_gbps"] = sorted({_row_float(r, "achieved_bw_gbps") for r in timed})
    evidence["row_peak_tflops"] = sorted(
        {(r.get("dtype", ""), _row_float(r, "achieved_peak_tflops")) for r in timed})

    # --- the ceilings ---------------------------------------------------------
    disagreements: list[str] = []
    compared = 0
    for value in evidence["row_bandwidth_gbps"]:
        if value <= 0.0:              # 0.0 is UNRECORDED, never a measured zero
            continue
        compared += 1
        if not _agree(value, stamp.bandwidth_gbps):
            disagreements.append(
                f"rows carry achieved_bw_gbps {value:.6f}, "
                f"{cal.name} reports {stamp.bandwidth_gbps:.6f}")
    stamped_bandwidth = any(v > 0.0 for v in evidence["row_bandwidth_gbps"])
    for dtype, value in evidence["row_peak_tflops"]:
        if value <= 0.0:
            # 0.0 is UNRECORDED. On a row that DID get its bandwidth stamped it
            # is a positive statement -- `recompute.ceiling_columns` writes the
            # bandwidth unconditionally and the peak only for a dtype the
            # calibration covers -- so the file loaded at sweep time had no peak
            # for this dtype. A file beside it that has one is a different file.
            # Guarded on the bandwidth so an arm predating the column, where
            # everything reads 0.0, comes back `unrecorded` rather than `wrong`.
            if stamped_bandwidth and stamp.peak_for(dtype):
                compared += 1
                disagreements.append(
                    f"rows carry no achieved_peak_tflops for dtype {dtype!r}, so "
                    f"the calibration they used had none, but {cal.name} reports "
                    f"{stamp.peak_for(dtype):.6f}")
            continue
        compared += 1
        if not _agree(value, stamp.peak_for(dtype)):
            claimed = stamp.peak_for(dtype)
            disagreements.append(
                f"rows carry achieved_peak_tflops {value:.6f} for dtype "
                f"{dtype!r}, {cal.name} reports "
                + (f"{claimed:.6f}" if claimed else "no peak for that dtype"))
    evidence["ceilings"] = ("disagree" if disagreements
                            else "agree" if compared else "unrecorded")
    evidence["ceiling_disagreements"] = disagreements

    device_ok = _device_agrees(stamp.gpu_name, evidence["row_gpus"])
    evidence["device"] = "agrees" if device_ok else "disagrees"

    if disagreements or not device_ok:
        reason = "; ".join(disagreements) or (
            f"rows were measured on {', '.join(evidence['row_gpus'])} and "
            f"{cal.name} describes {stamp.gpu_name}")
        # A declaration does NOT excuse this one. `recompute_ceilings.py`
        # restamps the rows from the calibration it then copies in, so a derived
        # arm whose ceilings disagree means the derivation itself is broken.
        return CalibrationProvenance(
            arm.name, CEILINGS_DISAGREE, evidence, declared,
            f"the calibration shipped with this arm is not the one its rows were "
            f"quoted against: {reason}")

    # --- the commit and the date ---------------------------------------------
    if not stamp.measured_commit:
        evidence["commit"] = "unrecorded"
    elif stamp.measured_commit in evidence["row_commits"]:
        evidence["commit"] = "matches"
    else:
        evidence["commit"] = "differs"

    evidence["date"], days_after = _date_verdict(stamp.checked_on, stamps)

    if evidence["ceilings"] == "unrecorded" and evidence["commit"] == "unrecorded" \
            and evidence["date"] == "unrecorded":
        return CalibrationProvenance(
            arm.name, UNKNOWN, evidence, declared,
            "the rows record no ceiling, no commit and no timestamp, so nothing "
            "can be established")

    if evidence["commit"] == "differs" or evidence["date"] != "within the sweep":
        blocking = None
        if days_after > 0 and not declared:
            # Causally impossible for a same-session calibration. A derived arm
            # is exactly this shape on purpose, which is why the declaration is
            # checked first.
            blocking = (
                f"the calibration is dated {stamp.checked_on}, {days_after} day(s) "
                f"after the last row was written ({stamps[-1]}), so it cannot be "
                f"the one this sweep ran against")
        return CalibrationProvenance(arm.name, DIFFERENT_SESSION, evidence,
                                     declared, blocking)
    return CalibrationProvenance(arm.name, SAME_SESSION, evidence, declared, None)


def _device_agrees(yaml_gpu: str, row_gpus: list[str]) -> bool:
    """Loose on purpose, the same way `roofline.device_matches` is: the yaml
    calls it "NVIDIA H200 (measured)" and the rows call it "NVIDIA H200"."""
    if not yaml_gpu or not row_gpus:
        return True                      # nothing to check against
    def norm(text: str) -> str:
        return "".join(c for c in text.lower() if c.isalnum())
    left = norm(yaml_gpu)
    return all(norm(g) in left or left in norm(g) for g in row_gpus)


def _date_verdict(checked_on: date | None, stamps: list[str]) -> tuple[str, int]:
    """`(phrase, days the calibration postdates the sweep)`.

    Postdating is the half that matters, so it is the half that comes back as a
    number. Predating is reported but not counted: a calibration must precede the
    sweep it serves, and one written at 23:50 the night before is both a day
    early and completely correct.
    """
    if checked_on is None or not stamps:
        return "unrecorded", 0
    try:
        first = date.fromisoformat(stamps[0][:10])
        last = date.fromisoformat(stamps[-1][:10])
    except ValueError:
        return "unrecorded", 0
    if checked_on > last:
        return f"{(checked_on - last).days} day(s) after the last row", (checked_on - last).days
    if checked_on < first:
        return f"{(first - checked_on).days} day(s) before the first row", 0
    return "within the sweep", 0


def entitled_ridge(arm: Path | str, dtype: str | None = None) -> tuple[float | None, str]:
    """`(ridge in FLOP/byte, why)`. The ridge this arm may be quoted against, or
    None and the reason it may not be.

    Same contract as `filter_superseded`: the reason comes back whether or not
    the number does, so an analysis that skips an arm ANNOUNCES it and one that
    uses an arm can print which ruler it used. A silent ridge is how C5 came to
    be scored against a target nobody could reproduce.

    Refuses on `ceilings_disagree`, and the refusal names both candidates,
    because that is the honest state of the whole-layer arm: 160.3 from its own
    rows, 176.2 from the file beside them, and the C5 target is the band those
    two produce.

    Refuses too when the calibration has no peak for the dtype the rows were
    swept in. `-fp8-three-kernel` is 19,908 fp8 rows next to a bf16-only
    calibration, which is why every one of them carries
    `achieved_peak_tflops = 0.0`, and quoting them against the bf16 ridge would
    be quoting fp8 work against half its roof.
    """
    arm = Path(arm)
    prov = calibration_provenance(arm)
    dtypes = prov.evidence.get("row_dtypes") or []
    if dtype is None:
        if len(dtypes) != 1:
            return None, (f"{arm.name}: rows carry {len(dtypes)} dtypes "
                          f"{dtypes}, so there is no single ridge; pass one")
        dtype = dtypes[0]

    if prov.verdict == CEILINGS_DISAGREE:
        rows = _rows_ridge(prov, dtype)
        stamp_ridge = _yaml_ridge(arm, dtype)
        return None, (
            f"{arm.name}: refused, {prov.blocking_reason or prov.verdict}. Its "
            f"rows give {_fmt(rows)} FLOP/byte and its measured.yaml gives "
            f"{_fmt(stamp_ridge)}; neither was measured in this arm's session")
    if prov.verdict == UNKNOWN:
        return None, f"{arm.name}: refused, {prov.blocking_reason or 'nothing recorded'}"

    # A blocked arm can still have a sound ridge: the ceilings agreeing means the
    # rows really were stamped from the file, whatever the date says about which
    # session wrote it. The caveat travels with the number rather than
    # suppressing it, so an analysis quotes the ridge AND the doubt.
    caveat = f". Note: {prov.blocking_reason}" if prov.blocking_reason else ""

    ridge = _rows_ridge(prov, dtype)
    if ridge is None:
        ridge = _yaml_ridge(arm, dtype)
        if ridge is None:
            return None, (f"{arm.name}: refused, no ceiling for dtype {dtype!r} on "
                          f"the rows or in measured.yaml, which is why every row "
                          f"carries achieved_peak_tflops 0.0")
        return ridge, (f"{arm.name}: {ridge:.1f} FLOP/byte for {dtype} from "
                       f"measured.yaml ({prov.verdict}); the rows carry no peak"
                       + caveat)
    return ridge, (f"{arm.name}: {ridge:.1f} FLOP/byte for {dtype} from the "
                   f"ceilings stamped on the rows ({prov.verdict})" + caveat)


def _rows_ridge(prov: CalibrationProvenance, dtype: str) -> float | None:
    from .calibrate import ridge_flop_per_byte  # deferred: see calibration_provenance
    peaks = [v for d, v in prov.evidence.get("row_peak_tflops", []) if d == dtype and v > 0]
    bandwidths = [v for v in prov.evidence.get("row_bandwidth_gbps", []) if v > 0]
    if len(peaks) != 1 or len(bandwidths) != 1:
        return None
    return ridge_flop_per_byte(peaks[0], bandwidths[0])


def _yaml_ridge(arm: Path, dtype: str) -> float | None:
    from .calibrate import read_stamp  # deferred: see calibration_provenance
    cal = arm / "measured.yaml"
    return read_stamp(cal).ridge(dtype) if cal.exists() else None


def _fmt(value: float | None) -> str:
    return f"{value:.1f}" if value else "no number"


#: How the committed report is regenerated. Named in the report itself, so the
#: next reader can rerun it instead of trusting a file with no provenance -- in
#: a document about provenance that would be a poor joke.
REPORT_COMMAND = (".venv/bin/python -m moe.bench.published results/published/*/ "
                  "> results/published/CALIBRATION_PROVENANCE.md")


def provenance_report(arms) -> str:
    """The verdict table for a set of arms, as markdown.

    Rendered by the code that decides, so the committed report in
    `results/published/CALIBRATION_PROVENANCE.md` cannot drift from the rule --
    `tests/test_calibration_provenance.py` regenerates it and compares byte for
    byte. Arm directory NAMES only, never paths, and no timestamp anywhere, so
    the same repo on two machines produces the same bytes.
    """
    paths = sorted(Path(a) for a in arms)
    verdicts = [(path, calibration_provenance(path)) for path in paths]
    blocked = [prov for _, prov in verdicts if prov.blocking_reason]
    out = [
        "# Does each published arm's calibration belong to its own sweep?",
        "",
        "Generated by `moe.bench.published.provenance_report`; regenerate with",
        "",
        f"    {REPORT_COMMAND}",
        "",
        "`measured.yaml` is copied out of `moe/bench/hardware/measured_<device>.yaml`",
        "at publish time, and that file is overwritten by every recalibration, so an",
        "arm can ship a ruler it never used. The rows themselves settle it: every timed",
        "row carries the `achieved_peak_tflops` and `achieved_bw_gbps` it was computed",
        "against, and those either are the file's numbers or are not.",
        "",
        "The commit is the weakest column here and `differs` is the NORMAL state: the",
        "workflow is calibrate, commit the yaml, then sweep, so the sweep runs one",
        "commit after the calibration by construction. `checked_on` has day resolution",
        "and did not catch the one real defect, which happened inside a single day.",
        "",
        "| arm | verdict | ceilings | commit | checked_on vs rows | ridge it may quote |",
        "|---|---|---|---|---|---|",
    ]
    refusals = []
    for path, prov in verdicts:
        ridge, why = entitled_ridge(path)
        if ridge is None:
            refusals.append(why)
        ev = prov.evidence
        out.append(
            f"| `{prov.arm}` | {prov.verdict}"
            + (f", derived from `{prov.derived_from}`" if prov.derived_from else "")
            + f" | {ev.get('ceilings', 'n/a')} | {ev.get('commit', 'n/a')}"
            + f" | {ev.get('date', 'n/a')} | "
            + (f"{ridge:.1f}" if ridge else "**refused**") + " |")
    out.append("")
    out.append(f"**{len(verdicts) - len(blocked)} of {len(verdicts)} arms pass**: "
               f"their calibration is either their own or declared derived. "
               f"{len(blocked)} does not.")
    for prov in blocked:
        out.append("")
        out.append(f"- `{prov.arm}`: {prov.blocking_reason}")
    if refusals:
        out += ["", "## Arms with no ridge to quote", ""]
        for why in refusals:
            out += [f"- {why}", ""]
        out += ["A `ceilings_disagree` refusal above is what costs claim C5 a",
                "target: a cross-card `2R/b` prediction scales with the ridge,",
                "and an arm with two candidate ridges gives C5 a band to be",
                "scored against rather than a number."]
    return "\n".join(out) + "\n"


def _main(argv: list[str]) -> int:
    """`python -m moe.bench.published <arm> ...` prints the report.

    An entry point rather than a script under `scripts/`, because the rule and
    the document that states it should not be able to drift apart, and because
    the report names this command as the way to regenerate itself.
    """
    arms = [Path(a) for a in argv if Path(a).is_dir()]
    if not arms:
        print(f"usage: {REPORT_COMMAND}")
        return 2
    print(provenance_report(arms), end="")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    import sys
    raise SystemExit(_main(sys.argv[1:]))
