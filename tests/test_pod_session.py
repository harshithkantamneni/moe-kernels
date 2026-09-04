"""scripts/pod_session.sh: what it states as a prediction, and what S6d reads.

Two of the prose review's blocking findings against the session script were
the same defect in two shapes. The step 2 and step 6 texts carried a table
TYPED on 2026-09-01 -- caps as 2 BM / (alpha b) against "ridge 160.3" -- and
the S6d gate counted rows carrying `throttled` and called the rate "thermal
stability". The typed ridge was one H200 calibration and not the card the
session runs on; the typed cap is the LIN reading no estimator returns; and
on every published row `throttled` is the retired two-sample drift flag, which
detected an idle-boost catch (moe/bench/timing.py). A reader of the terminal
could not tell any of that from what was printed.

The repairs: the table is COMPUTED against the attached card's calibration
with the file named, brackets the cap over the unmeasured alpha_a, and REFUSES
when there is nothing to read; and S6d reads the LEVEL and DRIFT verdicts and
refuses rows that predate them. Both are exercised here off a GPU, in every
direction they can go. `--dry-run` drives the table; `--gate-sweep` drives the
row gates over CSVs written for the purpose, which is what the flag is for.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
POD = REPO / "scripts" / "pod_session.sh"
PY = sys.executable

sys.path.insert(0, str(REPO))
from moe.bench.schema import COLUMNS, SCHEMA_VERSION  # noqa: E402
from moe.bench.timing import TIMING_BASIS  # noqa: E402


def sh(*args: str, env: dict | None = None):
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "MOE_PYTHON": PY}
    base.update(env or {})
    return subprocess.run(["bash", str(POD), *args], cwd=REPO, text=True,
                          capture_output=True, env=base)


def line_for(out: str, gate: str) -> str:
    hits = [ln for ln in out.splitlines() if ln.startswith(gate + " ")]
    assert hits, f"no {gate} line in:\n{out}"
    return hits[0]


# --------------------------------------------------------------------------
# rows for --gate-sweep
# --------------------------------------------------------------------------

def _v5_rows(path: Path, verdicts: list[tuple[str, str]], run_id: str = "aa1") -> None:
    """Timed rows on the v5 instrument, one per (LEVEL, DRIFT) verdict pair."""
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for level, drift in verdicts:
            row = dict.fromkeys(COLUMNS, "")
            row.update(schema_version=SCHEMA_VERSION, run_id=run_id, env_name="base",
                       gpu_name="NVIDIA H200", impl="torch_grouped_mm", model="toy",
                       num_tokens=32, ms_p50=1.0, correctness_passed="True",
                       instrument=TIMING_BASIS, clock_level_ok=level,
                       clock_drift_ok=drift, host_bound_ok="ok",
                       throttled=str("failed" in (level, drift)))
            w.writerow(row)


def _legacy_rows(path: Path, n: int, run_id: str = "aa1") -> None:
    """Rows in the shape every published arm has: schema 3, no verdict columns,
    `throttled` set by the retired drift flag."""
    cols = ["schema_version", "run_id", "env_name", "gpu_name", "impl", "model",
            "num_tokens", "ms_p50", "correctness_passed", "throttled", "clock_drift_pct"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i in range(n):
            w.writerow(dict(schema_version=3, run_id=run_id, env_name="base",
                            gpu_name="NVIDIA H200", impl="torch_grouped_mm",
                            model="toy", num_tokens=32, ms_p50=1.0,
                            correctness_passed="True", throttled=str(i % 2 == 0),
                            clock_drift_pct=7.0))


def _gate(tmp_path: Path, results: Path):
    return sh("--gate-sweep", str(results), "aa1",
              "--session-dir", str(tmp_path / "session"))


def test_s6d_passes_when_the_under_load_verdicts_are_clean(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _v5_rows(results / "run_aa1_base.csv", [("ok", "ok")] * 10)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "PASS" in line and "0.0% of timed rows failed LEVEL or DRIFT" in line, line
    assert r.returncode == 0, r.stdout + r.stderr


def test_s6d_fails_on_a_high_failure_rate_and_says_which_verdict(tmp_path):
    """The planted FAIL: 2 of 10 rows failed LEVEL and 1 failed DRIFT, so 30%
    of timed rows are unquotable against the roof. The old gate would have
    reported a "throttled" percentage and called it thermal."""
    results = tmp_path / "results"
    results.mkdir()
    _v5_rows(results / "run_aa1_base.csv",
             [("failed", "ok"), ("failed", "ok"), ("ok", "failed")] + [("ok", "ok")] * 7)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "FAIL" in line, line
    assert "30.0% of timed rows failed LEVEL or DRIFT (2 level, 1 drift" in line, line
    assert "thermal stability" not in r.stdout
    assert r.returncode != 0


def test_s6d_refuses_rows_that_predate_the_verdicts(tmp_path):
    """Every published arm is in this shape, and the old gate scored it: the
    retired flag read as a thermal rate. A row with no verdict is not a row
    that passed, and the consequence says what the flag on it actually was."""
    results = tmp_path / "results"
    results.mkdir()
    _legacy_rows(results / "run_aa1_base.csv", 6)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "FAIL" in line and "6 of 6 timed rows carry no LEVEL/DRIFT verdict" in line, line
    assert "REFUSED" in r.stdout and "predate schema v5" in r.stdout
    assert "idle-boost catch" in r.stdout
    assert r.returncode != 0


def test_s6d_refuses_when_no_row_carries_a_determined_verdict(tmp_path):
    """No clock source is not a 0% failure rate. It is silence, and the gate
    says so rather than passing on it."""
    results = tmp_path / "results"
    results.mkdir()
    _v5_rows(results / "run_aa1_base.csv", [("undetermined", "undetermined")] * 4)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "FAIL" in line and "every one of 4 timed rows is undetermined" in line, line
    assert "P13d" in r.stdout
    assert r.returncode != 0


def test_s6d_refuses_an_empty_sweep_rather_than_scoring_nothing(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "FAIL" in line and "no timed rows" in line, line


# --------------------------------------------------------------------------
# the prediction table
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dry_run() -> str:
    r = subprocess.run(
        ["bash", str(POD), "--dry-run", "--skip-tests", "--no-download",
         "--session-dir", "/tmp/moe-test-pod-session-predictions"],
        cwd=REPO, text=True, capture_output=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
             "HOME": os.environ.get("HOME", "/tmp"), "MOE_PYTHON": PY})
    return r.stdout


def test_the_table_resolves_the_ridge_from_the_cards_calibration(dry_run):
    """The ridge is READ, with the file it came from, not typed. The committed
    H200 ruler says 162.8, not the 160.3 the text carried."""
    assert "ridge  162.8 FLOP/byte (bf16), read from" in dry_run, dry_run
    assert "measured_nvidia_h200.yaml" in dry_run
    # Rehearsal is labelled as rehearsal: no card is attached here.
    assert "REHEARSAL: no card attached" in dry_run


def test_the_table_brackets_the_cap_and_states_a_verdict_per_end(dry_run):
    """A cap at an unmeasured alpha_a is a bracket. Small tiles sit below the
    ridge at both ends; 128 and 256 straddle it, and the table says so rather
    than picking an end."""
    assert "alpha_a is unmeasured" in dry_run
    rows = [ln for ln in dry_run.splitlines() if "  mixtral-8x7b " in ln and "cap/ridge" not in ln]
    assert len(rows) >= 4, dry_run
    by_bm = {int(ln.split()[0]): ln for ln in rows}
    assert "NO CROSSING at any batch: below the ridge at both ends" in by_bm[32]
    assert "NO CROSSING at any batch: below the ridge at both ends" in by_bm[64]
    assert "UNDECIDED: the bracket straddles the ridge" in by_bm[128]
    assert "UNDECIDED: the bracket straddles the ridge" in by_bm[256]


def test_the_withdrawn_numbers_appear_only_as_withdrawn(dry_run):
    """160.3 may be named as the number that was withdrawn, and only so."""
    for ln in dry_run.splitlines():
        if "160.3" in ln:
            assert "withdrawn" in ln or "no card's own" in ln or "stood here" in ln, ln
    assert "PREDICTION at alpha = 0.558, ridge 160.3" not in dry_run
    assert "AI cap 229" not in dry_run
    assert "% of timed rows throttled" not in dry_run


def test_the_table_refuses_a_card_with_no_calibration(tmp_path):
    """The failing branch of the resolution: a card nothing has calibrated
    gets no prediction and no borrowed ridge."""
    r = sh("--dry-run", "--skip-tests", "--no-download",
           "--expect-gpu", "NVIDIA B200", "--session-dir", str(tmp_path / "s"))
    assert "REFUSED: no calibration for 'NVIDIA B200'" in r.stdout, r.stdout
    assert "measured_nvidia_b200.yaml" in r.stdout
    assert "cap/ridge" not in r.stdout, "a table was printed with no ridge to score it against"
    assert "ridge  162.8 FLOP/byte (bf16), read from" not in r.stdout, \
        "the H200 ridge was borrowed for another card"


def test_the_script_no_longer_types_the_prediction_or_counts_the_retired_flag():
    """Source-level, for the two-call-site defect: the typed table and the
    `throttled` count must not survive anywhere in the script."""
    text = POD.read_text()
    assert "ridge 160.3" not in text.replace('"ridge 160.3"', "")
    assert 'row_bool(r, "throttled")' not in text
    assert '"thermal stability"' not in text
    assert "150.0 <= r <= 185.0" not in text
    assert "3900.0 <= bw <= 4800.0" not in text
    # one computation for both call sites: two bare invocations, and neither
    # the definition nor the comment that lists the function counts as one
    assert len(re.findall(r"^\s+predictions_table$", text, re.M)) == 2
    assert text.count("sweep_clock_gates \"") == 2
