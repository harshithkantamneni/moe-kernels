"""Re-derive the ceiling columns of an existing sweep against a new calibration.

WHY THIS EXISTS. `calibrate.py` was found to settle the clock under a dense
matmul and then measure bandwidth, which are different power regimes on an H200
SXM (1470 MHz at 64 C against 1980 MHz at 52 C), and to name a `torch.sum` tree
reduction as its read ceiling. Both make the ceiling low, and every
percent-of-ceiling figure quoted against it pessimistic.

Fixing that does NOT require re-measuring the sweep. The calibration feeds
exactly four things into a row (driver.py:262-289):

    achieved_bw_gbps          the ceiling itself
    bw_ceiling_pattern        which STREAM pattern named it
    achieved_peak_tflops      and pct_of_achieved_tflops
    implied_traffic_ratio     ceiling x time / compulsory bytes

Everything else, `ms_p50`, `tflops`, `compulsory_gbps` and
`arith_intensity_compulsory`, is computed from the measured time and the byte
model and never touches the calibration. The timings were never wrong, so
re-running three hours of GPU to correct a 2% ceiling would be re-measuring
things that do not change.

This mirrors what `_apply_cost` does rather than reimplementing it. The test
that the mirror is faithful is an identity: recomputing with the SAME
calibration must reproduce the stored columns exactly.
"""
from __future__ import annotations

import os
from pathlib import Path

from .calibrate import implied_traffic_ratio
from .roofline import Hardware

#: The only columns a calibration determines. Anything outside this set is
#: measured, and must be left alone.
CEILING_COLUMNS = (
    "achieved_bw_gbps",
    "bw_ceiling_pattern",
    "achieved_peak_tflops",
    "pct_of_achieved_tflops",
    "implied_traffic_ratio",
)


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
    return Hardware(
        name=str(doc.get("name", path)),
        bandwidth_bytes_s=bw_tb_s * 1e12,
        peak_flops=peaks,
        source=str(doc.get("source", str(path))),
        ceiling_pattern=str(doc.get("detail", {}).get("ceiling_pattern", "")),
    )


def _f(row: dict, key: str) -> float:
    v = row.get(key)
    return float(v) if v not in (None, "") else 0.0


def ceiling_columns(row: dict, hw: Hardware) -> dict:
    """The calibration-derived columns this row would carry under `hw`.

    Deliberately the same shape as `driver._apply_cost`, including its
    omissions: `implied_traffic_ratio` is written only when the cell is memory
    bound, because the bound it expresses is unsound otherwise, and a
    compute-bound row therefore gets no key at all rather than a zero.
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
