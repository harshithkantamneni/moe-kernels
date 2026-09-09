"""crossing_report reads the LEVEL side through `throttled`, and reads the roof.

Two things a rental would have exposed, both off-GPU (2026-09-08 pre-pod
verdict, F4's reader). First, the clock gate here is the `throttled` column,
and on a v6 row `moe/bench/driver.py` sets that from the under-load verdicts
with a LEVEL failure on the HIGH side kept OUT of it: a memory-bound cell
boosted to 1980 MHz against the 1515 MHz reference is not a thermal event and
its time is the kernel's. So a HIGH-side row must be KEPT by this report and a
LOW-side one (`throttled` True) skipped, and both are planted below. Second,
`pct_of_roof_at_cell_clock`, the fraction against the roof at the clock the
cell ran, had no reader under scripts/; this report prints it beside the fixed
`pct_of_achieved_tflops` per cell, and on a row from before the column says
"not available (v<6 row)" rather than printing a zero.

Every run is a subprocess over a hand-built CSV, the way the other
crossing_report tests in this directory work. No GPU.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from moe.bench.roofline import roof_at_clock
from moe.bench.schema import COLUMNS, COLUMNS_ADDED_IN, SCHEMA_VERSION

REPO = Path(__file__).resolve().parent.parent

#: The committed H200 calibration's two operating points: the bf16 GEMM
#: plateau LEVEL is scored against, and the clock under memory load.
REFERENCE_MHZ = 1515.0
BOOSTED_MHZ = 1980.0
PEAK_TFLOPS = 700.0

#: The columns a v3 row carries: everything v4, v5 and v6 added, removed. A
#: hand-built pre-v6 file has to LACK the roof columns, as a real one does,
#: or the reader would be asked about a column that is present and empty.
V3_COLUMNS = [c for c in COLUMNS
              if c not in {n for names in COLUMNS_ADDED_IN.values() for n in names}]


def write_csv(path: Path, rows: list[dict], fieldnames=COLUMNS) -> Path:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k in fieldnames})
    return path


def cell_row(t: int, ms: float, *, load: float = REFERENCE_MHZ,
             side: str = "", version: int = SCHEMA_VERSION,
             refused: str = "") -> dict:
    """One mixtral vLLM row at T tokens. `side` "high" plants a boosted cell,
    "low" a throttled one, "" a level one; `refused` is a driver `roof_note`
    on a v6 row whose roof was not scored."""
    tflops = 40.0 * t / 512
    roof = roof_at_clock(PEAK_TFLOPS, REFERENCE_MHZ, load)
    row = {"schema_version": version, "impl": "vllm_fused_experts",
           "model": "mixtral-8x7b", "dtype": "bf16", "num_tokens": t,
           "ms_p50": f"{ms}", "routing_kind": "uniform",
           "correctness_passed": "True", "tflops": tflops,
           "achieved_peak_tflops": PEAK_TFLOPS,
           "pct_of_achieved_tflops": 100.0 * tflops / PEAK_TFLOPS,
           "throttled": "True" if side == "low" else "False",
           "instrument": "queue-deep/l2-flush/clock-under-load/v3",
           "clock_level_ok": "failed" if side else "ok",
           "clock_level_side": side, "clock_drift_ok": "ok",
           "host_bound_ok": "ok", "sm_clock_load_mhz": load,
           "reference_clock_mhz": REFERENCE_MHZ,
           "roof_at_cell_clock_tflops": 0.0 if refused else roof,
           "pct_of_roof_at_cell_clock": 0.0 if refused else 100.0 * tflops / roof,
           "roof_note": refused}
    return row


def run_report(path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "crossing_report.py"), str(path),
         "--ridge", "162.8", *extra], capture_output=True, text=True, cwd=REPO)


GRID = (128, 256, 512, 1024, 2048, 4096)


def grid_rows(**over) -> list[dict]:
    return [cell_row(t, max(0.4, 0.4 * t / 512), **over) for t in GRID]


def test_a_boosted_row_is_kept_and_a_throttled_row_is_skipped(tmp_path):
    """THE PLANTED PAIR. Both fail LEVEL; only the LOW side is `throttled`."""
    rows = grid_rows()
    rows.append(cell_row(512, 0.41, load=BOOSTED_MHZ, side="high"))
    rows.append(cell_row(512, 0.60, load=1400.0, side="low"))
    got = run_report(write_csv(tmp_path / "v6.csv", rows))
    assert got.returncode == 0, got.stderr
    assert f"kept {len(GRID) + 1} rows, skipped 1 (throttled or failed)" in got.stdout
    assert ("of the kept rows, 1 failed LEVEL HIGH (boosted above the "
            "reference clock): kept") in got.stdout
    assert f"1 of {len(GRID) + 1} rows failed LEVEL HIGH" in got.stdout
    assert "use pct_of_roof_at_cell_clock" in got.stdout
    # The gate can also be lifted, and then the LOW row is in the pool.
    lifted = run_report(tmp_path / "v6.csv", "--include-throttled")
    assert f"kept {len(GRID) + 2} rows, skipped 0" in lifted.stdout


def test_a_level_corpus_reports_no_high_side_rows(tmp_path):
    """The PASS branch of the count: a corpus with nothing boosted says 0, so
    the line is a measurement of the pool and not a fixed banner."""
    got = run_report(write_csv(tmp_path / "level.csv", grid_rows()))
    assert got.returncode == 0, got.stderr
    assert "of the kept rows, 0 failed LEVEL HIGH" in got.stdout
    assert "rows failed LEVEL HIGH (boosted" not in got.stdout.split("===", 1)[1]


def test_both_roof_fractions_are_printed_per_cell_on_a_scored_v6_corpus(tmp_path):
    """The corrected fraction beside the fixed one, each labelled, with the
    row count it was medianed over. On a boosted row the two differ by the
    clock ratio; on the level rows they agree, and the medians here are over
    the same seven rows so the printed pair is a real comparison."""
    rows = grid_rows()
    rows.append(cell_row(512, 0.41, load=BOOSTED_MHZ, side="high"))
    got = run_report(write_csv(tmp_path / "v6.csv", rows))
    assert got.returncode == 0, got.stderr
    body = got.stdout.split("=== mixtral-8x7b / bf16 / vllm_fused_experts ===", 1)[1]
    assert "fraction of compute roof (median over the rows in this cell):" in body
    fixed = next(line for line in body.splitlines()
                 if "pct_of_achieved_tflops    (FIXED roof" in line)
    cell = next(line for line in body.splitlines()
                if "pct_of_roof_at_cell_clock (roof AT THE CELL'S CLOCK)" in line)
    assert fixed.endswith(f"over {len(GRID) + 1} rows")
    assert cell.endswith(f"over {len(GRID) + 1} rows")
    assert "not available" not in body


def test_a_pre_v6_corpus_says_the_corrected_fraction_is_not_available(tmp_path):
    """The branch the committed corpus (v3 to v5) exercises: the column does
    not exist on the row, and the report says so instead of printing 0.0."""
    rows = grid_rows(version=3)
    got = run_report(write_csv(tmp_path / "v3.csv", rows, fieldnames=V3_COLUMNS))
    assert got.returncode == 0, got.stderr
    assert "pct_of_achieved_tflops    (FIXED roof" in got.stdout
    assert (f"pct_of_roof_at_cell_clock (roof AT THE CELL'S CLOCK):      "
            f"not available (v<6 row)  [{len(GRID)} rows]") in got.stdout
    assert "of the kept rows, 0 failed LEVEL HIGH" in got.stdout


def test_a_v6_row_the_driver_refused_to_score_prints_the_drivers_reason(tmp_path):
    """Not "not available": the column exists, the driver declined, and its
    `roof_note` is the reason a reader needs (the committed calibrations are
    graded idle-scalar, which is exactly this refusal)."""
    why = "the reference is graded 'idle-scalar', not an under-load median"
    rows = grid_rows(refused=why)
    got = run_report(write_csv(tmp_path / "refused.csv", rows))
    assert got.returncode == 0, got.stderr
    assert f"{why}  [{len(GRID)} rows]" in got.stdout
    assert "not available (v<6 row)" not in got.stdout
