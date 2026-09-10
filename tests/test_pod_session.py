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

#: Three operating points this card really runs at, planted as rows. 1515 is
#: the bf16 GEMM reference the 2026-09-02 calibration recorded (the 2026-09-09
#: one records 1485; the ratios below are what matter, not the year) and 1980
#: is the clock it holds for the whole 30 s memory settle: 1980 / 1515 = 1.307
#: against a 1.05 band, so a memory-bound cell fails LEVEL on the HIGH side as
#: its normal state. COLD_MHZ
#: is the other steady state the 2026-09-09 census found: 1400 is the
#: BLOCK_M=128/BLOCK_N=64 family sitting under the reference in every rep,
#: because at 700 W a hungry tile clocks itself down. Both sides are planted
#: below and neither is an exclusion.
REFERENCE_MHZ = 1515
BOOSTED_MHZ = 1980
COLD_MHZ = 1400


def _v5_rows(path: Path, verdicts: list[tuple], run_id: str = "aa1",
             schema_version: int = SCHEMA_VERSION) -> None:
    """Timed rows on the under-load instrument, one per verdict tuple.

    A tuple is (LEVEL, DRIFT) or (LEVEL, DRIFT, side). A LEVEL failure with no
    side given is written as the driver writes it on a v6 row: side "low" with
    a cold clock. Side "high" gets the boosted clock. `throttled` follows the
    driver's rule (moe/bench/driver.py), which since 2026-09-09 is the DRIFT
    verdict and nothing else: neither side of a LEVEL failure is throttled.
    """
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for verdict in verdicts:
            level, drift = verdict[0], verdict[1]
            side = verdict[2] if len(verdict) > 2 else ("low" if level == "failed" else "")
            if level != "failed":
                side = ""
            load = {"high": BOOSTED_MHZ, "low": COLD_MHZ}.get(side, REFERENCE_MHZ)
            row = dict.fromkeys(COLUMNS, "")
            row.update(schema_version=schema_version, run_id=run_id, env_name="base",
                       gpu_name="NVIDIA H200", impl="torch_grouped_mm", model="toy",
                       num_tokens=32, ms_p50=1.0, correctness_passed="True",
                       instrument=TIMING_BASIS, clock_level_ok=level,
                       clock_drift_ok=drift, host_bound_ok="ok",
                       clock_level_side=side, sm_clock_load_mhz=load,
                       reference_clock_mhz=REFERENCE_MHZ,
                       throttled=str(drift == "failed"))
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
    assert "PASS" in line and "0.0% of timed rows failed DRIFT" in line, line
    assert "0 low and 0 high kept" in line, line
    assert r.returncode == 0, r.stdout + r.stderr


def test_s6d_fails_on_a_high_drift_rate_and_says_what_drift_is(tmp_path):
    """The planted FAIL: 3 of 10 rows failed DRIFT, so 30% of timed rows never
    reached one operating point. The old gate would have reported a
    "throttled" percentage and called it thermal; between 03df2d4 and
    2026-09-09 it would have added the LEVEL-low rows to this rate."""
    results = tmp_path / "results"
    results.mkdir()
    _v5_rows(results / "run_aa1_base.csv",
             [("ok", "failed"), ("ok", "failed"), ("ok", "failed")] + [("ok", "ok")] * 7)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "FAIL" in line, line
    assert "30.0% of timed rows failed DRIFT (3 drift; 0 low and 0 high kept" in line, line
    assert "thermal stability" not in r.stdout
    assert "the warmup never reached one operating point" in r.stdout
    assert "two consecutive clock reads agree within one 15 MHz step" in r.stdout
    assert r.returncode != 0


def test_s6d_keeps_the_high_side_and_passes_on_sixty_percent_boosted_rows(tmp_path):
    """THE PLANTED HIGH-SIDE ROWS, the fifteenth instance of the recurring
    defect. Six of ten rows are memory-bound cells that boosted to 1980 MHz
    against the 1515 MHz reference and fail LEVEL with side "high"; none fail
    DRIFT. Until 2026-09-08 this gate counted them as failed LEVEL and FAILED
    at 60% on the H200's normal decode state, with a consequence saying the
    card sat BELOW the roof's clock. They are KEPT: the rate is 0.0%, the gate
    PASSES, and the page says what was kept and which column to read them
    by."""
    results = tmp_path / "results"
    results.mkdir()
    _v5_rows(results / "run_aa1_base.csv",
             [("failed", "ok", "high")] * 6 + [("ok", "ok")] * 4)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "PASS" in line, line
    assert "0.0% of timed rows failed DRIFT (0 drift; 0 low and 6 high kept" in line, line
    assert ("S6d kept 0 of 10 timed rows that failed LEVEL on the LOW side "
            "and 6 on the HIGH side") in r.stdout
    assert "pct_of_roof_at_cell_clock" in r.stdout
    assert ("6 failed LEVEL high among the rows DRIFT kept (side recorded, "
            "neither excludes)") in r.stdout
    assert r.returncode == 0, r.stdout + r.stderr
    # And the rows themselves carry the driver's rule: neither side is
    # throttled, so nothing downstream that drops `throttled` loses them.
    import csv as _csv
    with (results / "run_aa1_base.csv").open() as fh:
        rows = list(_csv.DictReader(fh))
    assert all(r_["throttled"] == "False" for r_ in rows)


def test_s6d_keeps_the_low_side_too_and_passes_on_a_hungry_tile(tmp_path):
    """THE 2026-09-09 CHANGE, planted in the shape that forced it. Six of ten
    rows are BLOCK_M=128/BLOCK_N=64 cells holding 1395 MHz against a 1485 MHz
    reference: not throttled, not drifting, just a hungry tile at its own
    steady state under the 700 W cap. Under the rule in force until 2026-09-09
    this sweep FAILED at 60% and the study's primary tile was unmeasurable on
    this card. The rate is 0.0% now, both sides are reported, and neither is an
    exclusion."""
    results = tmp_path / "results"
    results.mkdir()
    _v5_rows(results / "run_aa1_base.csv",
             [("failed", "ok", "low")] * 6 + [("ok", "ok")] * 4)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "PASS" in line, line
    assert "0.0% of timed rows failed DRIFT (0 drift; 6 low and 0 high kept" in line, line
    assert ("S6d kept 6 of 10 timed rows that failed LEVEL on the LOW side "
            "and 0 on the HIGH side") in r.stdout
    assert "Neither side is an exclusion" in r.stdout
    # The consequence paragraph prints on a FAIL, and this sweep PASSES: the
    # rule is asserted where it fires, in the two FAIL tests below.
    assert r.returncode == 0, r.stdout + r.stderr
    import csv as _csv
    with (results / "run_aa1_base.csv").open() as fh:
        rows = list(_csv.DictReader(fh))
    assert all(r_["throttled"] == "False" for r_ in rows)


def test_s6d_counts_both_sides_beside_the_drift_rate(tmp_path):
    """Both sides and a drift in one sweep, and only the drift counts. One
    DRIFT out of ten is 10.0% and FAILS; the two LOW and three HIGH rows beside
    it are reported as kept and move nothing. The gate as it stood on
    2026-09-08 would have said 30%, and the one before that 60%."""
    results = tmp_path / "results"
    results.mkdir()
    _v5_rows(results / "run_aa1_base.csv",
             [("failed", "ok", "low"), ("failed", "ok", "low"), ("ok", "failed"),
              ("failed", "ok", "high"), ("failed", "ok", "high"), ("failed", "ok", "high")]
             + [("ok", "ok")] * 4)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "FAIL" in line, line
    assert "10.0% of timed rows failed DRIFT (1 drift; 2 low and 3 high kept" in line, line
    assert "LEVEL IS NOT IN THIS RATE ON EITHER SIDE" in r.stdout
    assert r.returncode != 0


def test_s6d_does_not_count_a_drifted_row_as_a_kept_level_row(tmp_path):
    """THE ROW THAT FAILS BOTH, and the count that called it kept. Three of ten
    rows here fail LEVEL on the LOW side AND fail DRIFT. DRIFT excludes them, so
    they are not kept by anything; until now this gate counted the LEVEL sides
    over EVERY timed row and then printed them under the word "kept", so the
    note, the gate line and the S6d summary all described the same three
    excluded rows as kept. Not hypothetical: memory_branch_anchor's 2026-09-09
    cells carry 18 LOW and 12 DRIFT with 4 rows in both.

    The kept counts are taken over the rows DRIFT kept, and the both-failed
    rows are reported on their own line as excluded on DRIFT."""
    results = tmp_path / "results"
    results.mkdir()
    _v5_rows(results / "run_aa1_base.csv",
             [("failed", "failed", "low")] * 3 + [("failed", "ok", "low")] * 2
             + [("ok", "ok")] * 5)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "FAIL" in line, line
    # 3 of 10 drifted: the rate is DRIFT's alone and the two undrifted LOW rows
    # are the only ones counted kept.
    assert ("30.0% of timed rows failed DRIFT (3 drift; 2 low and 0 high kept, "
            "3 LEVEL-and-drifted excluded on DRIFT") in line, line
    assert ("2 failed LEVEL low and 0 failed LEVEL high among the rows DRIFT "
            "kept (side recorded, neither excludes), 3 failed LEVEL AND drifted "
            "(excluded on DRIFT)") in r.stdout, r.stdout
    assert ("S6d kept 2 of 10 timed rows that failed LEVEL on the LOW side "
            "and 0 on the HIGH side") in r.stdout
    assert ("S6d found 3 of 10 timed rows that failed LEVEL AND drifted. They "
            "are EXCLUDED, on DRIFT") in r.stdout
    assert "kept 3" not in r.stdout and "3 low and 0 high kept" not in r.stdout
    assert r.returncode != 0


def test_s6d_notes_a_v6_level_failure_that_names_no_side_and_does_not_refuse(tmp_path):
    """NOTE RATHER THAN REFUSE, since 2026-09-09. A v6 row that fails LEVEL
    without a side was still not written by moe/bench/driver.py and is still
    worth finding, but the side now decides nothing, so refusing a whole sweep
    over it would exclude on the one column that excludes nothing. It is
    recorded and the sweep is scored on DRIFT."""
    results = tmp_path / "results"
    results.mkdir()
    _v5_rows(results / "run_aa1_base.csv",
             [("failed", "ok", "")] + [("ok", "ok")] * 9)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "PASS" in line, line
    assert "0.0% of timed rows failed DRIFT" in line, line
    assert "S6d found 1 of 10 timed rows failing LEVEL on a v6 row that names no side" in r.stdout
    assert "recorded here rather than refused" in r.stdout
    assert "will not guess which side" not in r.stdout
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_v5_level_failure_is_low_by_that_instruments_own_definition(tmp_path):
    """The v5 instrument (00f3324 to 03df2d4) had no high edge, so a v5 row
    that fails LEVEL fell below the reference by construction. It is counted
    LOW rather than unsided: that is the instrument's definition, not this
    gate's guess. Counted, and kept, like every other LEVEL failure."""
    results = tmp_path / "results"
    results.mkdir()
    _v5_rows(results / "run_aa1_base.csv",
             [("failed", "ok", "")] * 2 + [("ok", "ok")] * 8, schema_version=5)
    r = _gate(tmp_path, results)
    line = line_for(r.stdout, "S6d")
    assert "PASS" in line, line
    assert "0.0% of timed rows failed DRIFT (0 drift; 2 low and 0 high kept" in line, line
    assert r.returncode == 0, r.stdout + r.stderr


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
    """The ridge is READ, with the file it came from, not typed.

    THE EXPECTED NUMBER IS READ TOO, from the same committed yaml, because
    typing it here is the defect this test exists to catch one level up: it
    said 162.8 from 2026-09-03, the 2026-09-09 calibration moved the H200's
    dense bf16 ceiling and the ridge with it to 152.8, and this assertion went
    stale the moment the ruler it describes was replaced."""
    import yaml
    card = yaml.safe_load(
        (REPO / "moe" / "bench" / "hardware" / "measured_nvidia_h200.yaml").read_text())
    ridge = (card["compute_dense_tflops"]["bf16"] * 1000.0
             / (card["memory"]["bandwidth_tb_s"] * 1000.0))
    assert f"ridge  {ridge:.1f} FLOP/byte (bf16), read from" in dry_run, dry_run
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
