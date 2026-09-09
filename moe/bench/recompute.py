"""Re-derive the ceiling columns of an existing sweep against a new calibration.

WHY THIS EXISTS. `calibrate.py` was found to settle the clock under a dense
matmul and then measure bandwidth, which are different power regimes on an H200
SXM (1470 MHz at 64 C against 1980 MHz at 52 C), and to name a `torch.sum` tree
reduction as its read ceiling. Both make the ceiling low, and every
percent-of-ceiling figure quoted against it pessimistic.

Fixing that does NOT require re-measuring the sweep. The calibration feeds
the columns in `CEILING_COLUMNS` into a row (`driver._apply_cost`) and
nothing else:

    achieved_bw_gbps            the ceiling itself
    bw_ceiling_pattern          which STREAM pattern named it
    achieved_peak_tflops        and pct_of_achieved_tflops, the FIXED roof
    implied_traffic_ratio       ceiling x time / compulsory bytes
    roof_at_cell_clock_tflops   the fixed roof rescaled to the clock THIS row
                                ran at (v6), with pct_of_roof_at_cell_clock
                                and roof_note, which names the FIXED-roof
                                fraction as the compute-bound gate input on a
                                scored row and says why the row is not scored
                                otherwise (roofline.ROOF_NOTE_SCORED)

Everything else, `ms_p50`, `tflops`, `compulsory_gbps`,
`arith_intensity_compulsory` and the under-load clock `sm_clock_load_mhz`, is
computed from the measured time, the byte model and the poller and never
touches the calibration. The timings were never wrong, so re-running three
hours of GPU to correct a 2% ceiling would be re-measuring things that do not
change.

THE PER-ROW ROOF WAS THE SECOND CALL SITE. When v6 added it the driver was
taught to write it and this mirror was not, so a recompute under a new peak
rewrote `achieved_peak_tflops` and left `roof_at_cell_clock_tflops` at the
old peak's value beside it: the v6 header survived, `schema.has_cell_clock_
roof` said True, and the column the schema tells a reader to quote from a v6
row was derived from a ruler the arm no longer carried. The roof needs the
calibration's REFERENCE CLOCK for the row's dtype family and its grade, which
`Hardware` did not carry; it does now (`Hardware.reference_clocks`, read from
the same file as the peaks by `load_hardware`), and both writers go through
`roofline.cell_clock_roof`, the one statement of the rule.

This mirrors what `_apply_cost` does rather than reimplementing it. The test
that the mirror is faithful is an identity: recomputing with the SAME
calibration must reproduce the stored columns exactly, on a v6 row as on a
v5 one.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import roofline as RF
from . import schema as SC
from .calibrate import implied_traffic_ratio
from .roofline import Hardware

#: The only columns a calibration determines. Anything outside this set is
#: measured, and must be left alone. The last three exist from schema v6; on
#: an older arm they are absent from the header and `rewrite_csv` leaves the
#: header as it found it, so a v5 arm is never given columns it cannot have
#: earned.
CEILING_COLUMNS = (
    "achieved_bw_gbps",
    "bw_ceiling_pattern",
    "achieved_peak_tflops",
    "pct_of_achieved_tflops",
    "implied_traffic_ratio",
    "roof_at_cell_clock_tflops",
    "pct_of_roof_at_cell_clock",
    "roof_note",
)

#: The v6 subset: the roof at the cell's own clock. Written only onto a row
#: that carries the columns, i.e. was measured under schema v6 or later.
CELL_CLOCK_ROOF_COLUMNS = ("roof_at_cell_clock_tflops",
                           "pct_of_roof_at_cell_clock", "roof_note")


def load_calibration_hardware(path: str | os.PathLike) -> Hardware:
    """Build a `Hardware` from a calibration YAML written by calibrate.py.

    Raises rather than defaulting when the file lacks what it needs: a silently
    zero bandwidth would divide every efficiency column into nonsense.
    """
    import yaml

    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a mapping")
    try:
        bw_tb_s = float(doc["memory"]["bandwidth_tb_s"])
        peaks = {k: float(v) * 1e12 for k, v in doc["compute_dense_tflops"].items()}
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"{path}: needs memory.bandwidth_tb_s and compute_dense_tflops; "
            f"missing {exc}") from exc
    if bw_tb_s <= 0:
        raise ValueError(f"{path}: bandwidth_tb_s must be positive")
    card = str((doc.get("detail") or {}).get("gpu_name") or "")
    return Hardware(
        name=str(doc.get("name", path)),
        bandwidth_bytes_s=bw_tb_s * 1e12,
        peak_flops=peaks,
        source=str(doc.get("source", str(path))),
        ceiling_pattern=str(doc.get("detail", {}).get("ceiling_pattern", "")),
        # The clock each roof was measured at, from THE SAME FILE, by the walk
        # the driver resolves with on the pod. Without it a v6 row's per-row
        # roof cannot be re-derived and the recompute refuses it by name.
        reference_clocks={
            family: ref for family in RF.REFERENCE_FAMILIES
            if (ref := RF.reference_clock_from_doc(doc, card, family)).mhz},
    )


def _f(row: dict, key: str) -> float:
    v = row.get(key)
    return float(v) if v not in (None, "") else 0.0


def carries_cell_clock_roof(row: dict) -> bool:
    """Was this row written under a schema that has the per-row roof (v6)?

    Keyed on the COLUMN being present in the row, not on its value: a v6 row
    the driver refused to score carries 0.0 and a `roof_note`, and must be
    re-derived (the refusal may lift under a recalibration that records the
    under-load median). A row read through `schema.read_csv` from an older
    CSV carries the `UNRECORDED` sentinel, which is "this row predates the
    column" and is left alone.
    """
    if "roof_at_cell_clock_tflops" not in row:
        return False
    return row.get("roof_at_cell_clock_tflops") != SC.UNRECORDED


def ceiling_columns(row: dict, hw: Hardware) -> dict:
    """The calibration-derived columns this row would carry under `hw`.

    Deliberately the same shape as `driver._apply_cost`, including its
    omissions: `implied_traffic_ratio` is written only when the cell is memory
    bound, because the bound it expresses is unsound otherwise, and a
    compute-bound row therefore gets no key at all rather than a zero.

    AND THE PER-ROW ROOF, on a row that carries it (`carries_cell_clock_roof`),
    through `roofline.cell_clock_roof` with `hw.reference_clocks` for the row's
    dtype family: the same call the driver makes, with the same three
    refusals written into `roof_note` (no under-load clock on the row, no
    reference for that family in this calibration, a reference not graded
    under-load). A row with no measured peak for its dtype gets the driver's
    note for that too. Nothing is left standing from the old ruler.
    """
    ms = _f(row, "ms_p50")
    if ms <= 0:
        return {}
    out: dict = {
        "achieved_bw_gbps": hw.bandwidth_bytes_s / 1e9,
        "bw_ceiling_pattern": hw.ceiling_pattern,
    }
    dtype = row.get("dtype", "")
    try:
        peak = hw.peak(dtype)
    except ValueError:
        peak = 0.0
    if peak:
        out["achieved_peak_tflops"] = peak / 1e12
        out["pct_of_achieved_tflops"] = 100.0 * _f(row, "tflops") / (peak / 1e12)
    if carries_cell_clock_roof(row):
        if peak:
            ref = hw.reference_clocks.get(RF.reference_family(dtype))
            roof, why = RF.cell_clock_roof(
                peak / 1e12, _f(row, "sm_clock_load_mhz"),
                ref.mhz if ref else None, ref.grade if ref else "")
            out["roof_at_cell_clock_tflops"] = roof
            out["pct_of_roof_at_cell_clock"] = (
                100.0 * _f(row, "tflops") / roof if roof else 0.0)
            out["roof_note"] = why
        else:
            out["roof_at_cell_clock_tflops"] = 0.0
            out["pct_of_roof_at_cell_clock"] = 0.0
            out["roof_note"] = f"no measured compute ceiling for dtype {dtype!r}"

    ai = _f(row, "arith_intensity_compulsory")
    if peak and hw.bound(dtype, ai) == "memory":
        # compulsory_bytes IS a column. An earlier version reconstructed it as
        # compulsory_gbps * time, which is algebraically the same and passed the
        # identity test, but it recomputed a value the row already carried and
        # its comment claimed the column did not exist.
        compulsory_bytes = _f(row, "compulsory_bytes")
        out["implied_traffic_ratio"] = implied_traffic_ratio(
            compulsory_bytes, ms, hw.bandwidth_bytes_s)
    return out


def rewrite_csv(src: str | os.PathLike, dst: str | os.PathLike,
                hw: Hardware) -> dict:
    """Write `src` to `dst` with the ceiling columns re-derived under `hw`.

    Never edits in place. The original arm is the record of what was measured
    against the ruler of the day, and overwriting it would erase the evidence
    that the ruler changed.
    """
    import csv

    src, dst = Path(src), Path(dst)
    with src.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = list(reader)

    changed = dict.fromkeys(CEILING_COLUMNS, 0)
    for row in rows:
        for col, value in ceiling_columns(row, hw).items():
            # `ceiling_columns` emits a v6 column only for a row that carries
            # it, so this cannot add a column the header lacks; asserted
            # rather than assumed, because DictWriter would raise on the
            # first row and a partial rewrite is worse than none.
            assert col in fields, (col, fields)
            before = row.get(col, "")
            after = value if isinstance(value, str) else repr(float(value))
            if str(before) != str(after):
                changed[col] += 1
            row[col] = after

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "changed": changed, "hardware": hw.name}
