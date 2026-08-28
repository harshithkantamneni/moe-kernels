"""Re-deriving the ceiling columns without re-running a three-hour sweep.

The calibration feeds exactly four things into a row: `achieved_bw_gbps`,
`bw_ceiling_pattern`, `achieved_peak_tflops` with `pct_of_achieved_tflops`, and
`implied_traffic_ratio`. Everything else, `ms_p50`, `tflops`, `compulsory_gbps`,
`arith_intensity_compulsory`, comes from the timing and the byte model and never
touches it (driver.py:262-289).

So when `calibrate.py` was found to settle under the wrong workload and to name a
tree reduction as its read ceiling, the fix does NOT require re-measuring 17,640
cells. It requires re-deriving four columns. The timings were never wrong.

The test that this is faithful is an identity: recomputing with the SAME
calibration a row was measured against must reproduce that row exactly. If it
does not, the recompute does not model what the driver did and cannot be trusted
with a different calibration either.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from moe.bench.recompute import ceiling_columns, load_calibration_hardware

ARM = (Path(__file__).resolve().parents[1] / "results" / "published"
       / "2026-08-26-nvidia_h200-full-three-way")


def rows(limit: int = 2000) -> list[dict]:
    p = ARM / "merged.csv"
    if not p.exists():
        pytest.skip(f"no published sweep at {p}")
    with p.open(newline="") as fh:
        return [r for _, r in zip(range(limit), csv.DictReader(fh), strict=False)]


@pytest.fixture
def hardware():
    cal = ARM / "measured.yaml"
    if not cal.exists():
        pytest.skip("no calibration beside the published arm")
    return load_calibration_hardware(cal)


def test_recomputing_with_the_same_calibration_changes_nothing(hardware):
    """The identity check. Anything else means the recompute is not modelling
    what driver.py did."""
    checked = 0
    for row in rows():
        if not row.get("ms_p50") or float(row["ms_p50"]) <= 0:
            continue
        got = ceiling_columns(row, hardware)
        for col, value in got.items():
            stored = row.get(col, "")
            if col == "bw_ceiling_pattern":
                assert value == stored, (col, value, stored)
                continue
            if stored in ("", None):
                continue
            assert float(value) == pytest.approx(float(stored), rel=1e-9), (
                f"{col} on {row['impl']}/{row['model']}/T{row['num_tokens']}: "
                f"recomputed {value}, stored {stored}")
        checked += 1
    assert checked > 500, f"only {checked} rows exercised"


def test_a_faster_ceiling_lowers_the_implied_traffic_ratio(hardware):
    """implied_traffic_ratio is ceiling x time / compulsory bytes, so raising
    the ceiling raises it. The C4 correction raises the ceiling ~2%, which makes
    every published efficiency figure PESSIMISTIC, and this fixes the sign."""
    import dataclasses
    faster = dataclasses.replace(
        hardware, bandwidth_bytes_s=hardware.bandwidth_bytes_s * 1.02)
    moved = 0
    for row in rows(400):
        if not row.get("ms_p50") or float(row["ms_p50"]) <= 0:
            continue
        a = ceiling_columns(row, hardware).get("implied_traffic_ratio")
        b = ceiling_columns(row, faster).get("implied_traffic_ratio")
        if a in (None, "", 0.0) or b in (None, "", 0.0):
            continue
        assert float(b) == pytest.approx(float(a) * 1.02, rel=1e-9)
        moved += 1
    assert moved > 20, f"only {moved} rows carried a traffic ratio"


def test_the_ratio_is_omitted_for_compute_bound_rows(hardware):
    """driver.py only writes it when the cell is memory bound, because the
    bound it expresses is unsound otherwise. The recompute must agree."""
    seen_compute = False
    for row in rows():
        if not row.get("ms_p50") or float(row["ms_p50"]) <= 0:
            continue
        ai = float(row.get("arith_intensity_compulsory") or 0)
        if ai < hardware.ridge_point(row["dtype"]):
            continue
        seen_compute = True
        assert "implied_traffic_ratio" not in ceiling_columns(row, hardware)
    assert seen_compute, "no compute-bound rows in the sample; test proved nothing"


def test_an_unreadable_calibration_is_refused(tmp_path):
    bad = tmp_path / "nope.yaml"
    bad.write_text("name: something\n")
    with pytest.raises((KeyError, ValueError)):
        load_calibration_hardware(bad)
