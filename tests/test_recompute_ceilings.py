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

AND THE SCRIPT AROUND IT, from 2026-09-02. The arithmetic was always tested and
the CLI never was, which is how it came to write a new published arm that
recorded no commit of its own and to restamp H200 rows against an A100 ruler
without a word. The second half of this file runs `recompute_ceilings.main` and
checks what it leaves on disk.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from moe.bench import provenance as PV
from moe.bench.published import DERIVED_MARKER
from moe.bench.recompute import (
    CEILING_COLUMNS,
    ceiling_columns,
    load_calibration_hardware,
)


def _load_script():
    """The CLI, loaded by path. `scripts/` is not a package and never has been."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "recompute_ceilings", root / "scripts" / "recompute_ceilings.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RC = _load_script()
HARDWARE = Path(__file__).resolve().parents[1] / "moe" / "bench" / "hardware"

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


# --------------------------------------------------------------------------
# The rewrite has to be attributable, and it has to refuse the two ways it was
# found to go wrong when it was run twice.
# --------------------------------------------------------------------------

def _tiny_arm(root: Path, card: str = "NVIDIA H200") -> Path:
    """A one-row arm, so these tests do not depend on a published directory."""
    arm = root / "tiny-arm"
    arm.mkdir()
    row = {"impl": "base", "model": "mixtral-8x7b", "dtype": "bf16",
           "num_tokens": "512", "ms_p50": "1.0", "gpu_name": card,
           "git_sha": "0" * 40, "compulsory_bytes": "1000000",
           "flops": "2000000", "arith_intensity_compulsory": "2.0",
           # Every ceiling column is a FIELD from the start: `rewrite_csv`
           # writes into the header it was given, so an arm missing one of them
           # would fail for a reason that has nothing to do with these tests.
           **{column: "" for column in CEILING_COLUMNS}}
    with (arm / "run_a.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return arm


def test_the_rewrite_records_the_commit_that_did_it(tmp_path):
    """The rows keep the git_sha of the commit that MEASURED them, which is
    right and is also why nothing named the commit that REWROTE them:
    `check_published_shas` validated a commit that did not produce the numbers
    now in the file."""
    arm = _tiny_arm(tmp_path)
    out = tmp_path / "derived"
    cal = HARDWARE / "measured_nvidia_h200.yaml"
    assert RC.main(["--arm", str(arm), "--calibration", str(cal),
                    "--out", str(out)]) == 0

    doc = json.loads((out / RC.RECOMPUTE_JSON).read_text())
    for key in PV.TOP_LEVEL_KEYS:
        assert key in doc, key
    assert doc["provenance"]["git_sha"], "the commit that rewrote the rows"
    assert doc["provenance"]["utc"]
    assert doc["instrument"] == RC.INSTRUMENT
    # The ceiling every re-derived column comes from, and where it came from.
    assert doc["provenance"]["bandwidth"] > 0
    assert "measured_nvidia_h200.yaml" in doc["bandwidth_source"]
    assert doc["calibration_sha256"] == RC.calibration_sha(cal)
    assert doc["derived_from"] == arm.name and doc["row_cards"] == ["NVIDIA H200"]
    assert doc["csvs"]["run_a.csv"]["rows"] == 1
    # ...and the marker still leads with the source arm, which is what
    # `published.derived_from` reads.
    assert (out / DERIVED_MARKER).read_text().splitlines()[0] == arm.name


def test_another_cards_ruler_is_refused_rather_than_stamped(tmp_path):
    """`rewrite_csv` restamps every row from whatever calibration it is handed.
    Run against the H200 yaml then the A100 yaml, both derived
    `<arm>-recalibrated`, and H200 rows came back with achieved_bw_gbps
    4377.2 -> 1799.4 and implied_traffic_ratio 3.716 -> 1.527, with no refusal.
    """
    arm = _tiny_arm(tmp_path, card="NVIDIA H200")
    with pytest.raises(SystemExit, match="REFUSING"):
        RC.main(["--arm", str(arm), "--out", str(tmp_path / "x"),
                 "--calibration",
                 str(HARDWARE / "measured_nvidia_a100_sxm4_80gb.yaml")])
    # ...and the matching card is not refused, or the gate would be a wall.
    assert RC.main(["--arm", str(arm), "--out", str(tmp_path / "y"),
                    "--calibration",
                    str(HARDWARE / "measured_nvidia_h200.yaml")]) == 0


def test_rows_with_no_card_at_all_are_refused(tmp_path):
    """A check that examined nothing reports no failures."""
    arm = _tiny_arm(tmp_path, card="")
    with pytest.raises(SystemExit, match="no row"):
        RC.main(["--arm", str(arm), "--out", str(tmp_path / "z"),
                 "--calibration", str(HARDWARE / "measured_nvidia_h200.yaml")])


def test_a_second_ruler_for_the_same_card_may_not_overwrite_the_first(tmp_path):
    """The H200 was recalibrated six times to six distinct ceilings, and two
    calibrations of ONE card derive ONE default destination. The second run
    overwrote the first's CSVs, measured.yaml, README and marker in silence."""
    arm = _tiny_arm(tmp_path)
    out = tmp_path / "derived"
    first = HARDWARE / "measured_nvidia_h200.yaml"
    assert RC.main(["--arm", str(arm), "--calibration", str(first),
                    "--out", str(out)]) == 0

    # A DIFFERENT FILE FOR THE SAME CARD: same name, different contents, so a
    # path comparison would call these one ruler and a content hash does not.
    second = tmp_path / "measured_again.yaml"
    shutil.copy2(first, second)
    second.write_text(second.read_text() + "\n# recalibrated later that day\n")
    assert RC.calibration_sha(second) != RC.calibration_sha(first)
    with pytest.raises(SystemExit, match="REFUSING to overwrite"):
        RC.main(["--arm", str(arm), "--calibration", str(second),
                 "--out", str(out)])

    # Re-running with the SAME ruler is idempotent and must NOT be refused.
    assert RC.main(["--arm", str(arm), "--calibration", str(first),
                    "--out", str(out)]) == 0
