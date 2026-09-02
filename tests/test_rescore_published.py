"""The 26 published reports quoted another card's ridge, and the fix must be narrow.

`scripts/rescore_published_reports.py` edits committed evidence, which is the
most dangerous thing in this slice. So the tests are about CONFINEMENT and
REVERSIBILITY as much as about the new numbers:

- the four ridge-dependent fields change and NOTHING else does, checked by
  re-serialising the untouched half of every report;
- every gate verdict, `ai_cap`, every ladder fit and every plateau survive
  untouched, because the audit's refuters established those are ridge-
  independent and a rescoring that moved them would be a rewrite;
- running the tool twice changes nothing, including the timestamp;
- the FAIL branch of every gate exists and is reachable, planted rather than
  reasoned about;
- the A100 arm's committed files really do now carry 145.8 from
  `measured_nvidia_a100_sxm4_80gb.yaml`, which is the audit's own acceptance
  test for fix B4.

No GPU. Every test that writes works on a COPY of the arm in tmp_path; the two
that read the committed tree only read it.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe.bench import exit_codes as X  # noqa: E402

PUBLISHED = ROOT / "results" / "published"
A100_ARM = PUBLISHED / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
A100_MIXTRAL = A100_ARM / "mixtral-8x7b-bf16-r1024-g1-n64-4867a2.report.json"
H200_S4 = PUBLISHED / "2026-09-01-nvidia_h200-alpha-surface-s4"


def _load():
    spec = importlib.util.spec_from_file_location(
        "rescore_published_reports",
        ROOT / "scripts" / "rescore_published_reports.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RS = _load()
SWEEP = RS.load_sweep()


def _run(args, cwd=ROOT):
    return subprocess.run([sys.executable,
                           str(ROOT / "scripts" / "rescore_published_reports.py"),
                           *args], capture_output=True, text=True, cwd=cwd)


@pytest.fixture
def arm_copy(tmp_path) -> Path:
    """A writable copy of the A100 arm, under a directory of the same name.

    The NAME matters: the card is resolved from the arm directory, so copying
    into `tmp_path` directly would make every test here exercise the refusal
    path instead of the rescoring.
    """
    dest = tmp_path / A100_ARM.name
    shutil.copytree(A100_ARM, dest)
    return dest


# --------------------------------------------------------------------------
# the audit's acceptance test, on the committed tree
# --------------------------------------------------------------------------

def test_the_a100_report_carries_its_own_ridge_and_says_where_it_came_from():
    doc = json.loads(A100_MIXTRAL.read_text())
    assert doc["ridge"] == 145.8
    assert doc["ridge_source"] == "measured_nvidia_a100_sxm4_80gb.yaml"


def test_every_published_report_now_cites_its_own_card():
    expected = {"nvidia_a100_sxm4_80gb": 145.8, "nvidia_h200": 162.8}
    seen = set()
    for path in RS.report_paths(PUBLISHED):
        doc = json.loads(path.read_text())
        slug = doc["ridge_source"][len("measured_"):-len(".yaml")]
        assert doc["ridge"] == expected[slug], path
        assert doc["ridge_band"][0] < doc["ridge"] < doc["ridge_band"][1], path
        assert doc["rescored_from"]["ridge"] == 160.3
        seen.add(slug)
    assert seen == set(expected), "both cards must be represented"


def test_no_published_report_still_carries_the_borrowed_band():
    for path in RS.report_paths(PUBLISHED):
        doc = json.loads(path.read_text())
        assert doc["ridge_band"] != [160.3, 176.2], path


def test_the_a100_note_exists_and_names_both_numbers():
    note = (A100_ARM / "NOTE.md").read_text()
    assert "160.3" in note and "145.8" in note
    assert "gate" in note.lower()


def test_running_the_tool_on_the_committed_tree_now_changes_nothing():
    """Idempotence, on the real tree, as a plan that proposes no rewrite."""
    got = _run([])
    assert got.returncode == 0, got.stderr
    assert "0 to rewrite" in got.stdout
    assert "12 to rewrite" not in got.stdout


# --------------------------------------------------------------------------
# confinement: only the four fields move
# --------------------------------------------------------------------------

def test_only_the_registered_fields_differ_from_the_pre_rescore_report(arm_copy):
    """`untouched()` is the check the gate makes; here it is made from git."""
    original = subprocess.run(
        ["git", "show", f"HEAD:results/published/{A100_ARM.name}/"
                        f"{A100_MIXTRAL.name}"],
        capture_output=True, text=True, cwd=ROOT)
    assert original.returncode == 0, original.stderr
    before = json.loads(original.stdout)
    after = json.loads(A100_MIXTRAL.read_text())
    assert RS.untouched(before) == RS.untouched(after)
    # And the rescoring really did move something, or the check above is vacuous.
    assert before["ridge"] != after["ridge"]


def test_the_gate_verdicts_ai_caps_and_ladders_are_untouched():
    original = subprocess.run(
        ["git", "show", f"HEAD:results/published/{A100_ARM.name}/"
                        f"{A100_MIXTRAL.name}"],
        capture_output=True, text=True, cwd=ROOT)
    before = json.loads(original.stdout)
    after = json.loads(A100_MIXTRAL.read_text())
    assert before["gates"] == after["gates"]
    assert before["ladder"] == after["ladder"]
    assert before["plateau_tflops"] == after["plateau_tflops"]
    assert before["compute_reference"] == after["compute_reference"]
    for bm, pred in before["predictions"].items():
        assert pred["ai_cap"] == after["predictions"][bm]["ai_cap"]


def test_a_rescoring_that_moves_a_forbidden_field_fails_the_gate():
    """The FAIL branch of `fields_confined`, planted."""
    before = json.loads(A100_MIXTRAL.read_text())
    after = json.loads(A100_MIXTRAL.read_text())
    after["plateau_tflops"] = 1.0
    leaky = RS.Outcome(A100_MIXTRAL, before, after)
    assert not leaky.confined
    verdicts = {name: v for _k, name, v, _d in RS.gates([leaky], SWEEP, "t")}
    assert verdicts["fields_confined"] == "FAIL"


# --------------------------------------------------------------------------
# the arithmetic: the predictions are the sweep's, at the new ridge
# --------------------------------------------------------------------------

def test_the_rescored_predictions_are_what_the_sweep_predicts_at_that_ridge():
    doc = json.loads(A100_MIXTRAL.read_text())
    lo, hi = doc["ridge_band"]
    b = doc["dtype_bytes"]
    for bm_text, pred in doc["predictions"].items():
        bm = int(bm_text)
        want_lo = SWEEP.predict_tile(bm, doc["alpha"], lo, b)
        want_hi = SWEEP.predict_tile(bm, doc["alpha"], hi, b)
        assert pred["crossing_rows_ridge_lo"] == want_lo.crossing_rows
        assert pred["crossing_rows_ridge_hi"] == want_hi.crossing_rows
        assert pred["first_compute_tread"] == want_lo.first_compute_tread


def test_the_horizon_is_twice_the_retracted_alphas_crossing_at_the_new_ridge():
    doc = json.loads(A100_MIXTRAL.read_text())
    block_sizes = sorted(int(k) for k in doc["predictions"])
    null_bm = SWEEP.null_block_m(block_sizes)
    retracted = SWEEP.predict_tile(null_bm, SWEEP.RETRACTED_ALPHA, doc["ridge"],
                                   doc["dtype_bytes"])
    assert doc["bracketing"]["horizon_rows"] == \
        pytest.approx(2.0 * retracted.crossing_rows)


def test_lowering_the_ridge_moves_a_tile_into_crossing_rather_than_out():
    """The direction the borrowed H200 figure was hiding on the A100."""
    doc = json.loads(A100_MIXTRAL.read_text())
    at_old = SWEEP.predict_tile(128, doc["alpha"], 176.2, doc["dtype_bytes"])
    at_new = SWEEP.predict_tile(128, doc["alpha"], doc["ridge_band"][1],
                                doc["dtype_bytes"])
    assert at_new.crossing_rows < at_old.crossing_rows


# --------------------------------------------------------------------------
# refusal and idempotence, on a copy
# --------------------------------------------------------------------------

def test_an_arm_whose_name_names_no_calibrated_card_is_refused(tmp_path):
    dest = tmp_path / "2026-01-01-nvidia_gb200-invented"
    dest.mkdir()
    shutil.copy(A100_MIXTRAL, dest / "cell.report.json")
    got = _run([str(tmp_path)])
    assert got.returncode == X.INVALID
    assert "RESULT: VALIDITY card_resolved FAIL" in got.stdout
    assert "NOT WRITING" not in got.stdout        # no --write was passed


def test_a_refused_arm_is_not_written_even_with_write(tmp_path):
    dest = tmp_path / "2026-01-01-nvidia_gb200-invented"
    dest.mkdir()
    target = dest / "cell.report.json"
    shutil.copy(A100_MIXTRAL, target)
    before = target.read_text()
    got = _run(["--write", str(tmp_path)])
    assert got.returncode == X.INVALID
    assert "NOT WRITING" in got.stdout
    assert target.read_text() == before


def test_two_slugs_matching_one_arm_equally_well_refuse():
    with pytest.raises(RS.CardUnresolved) as excinfo:
        RS.profile_for_arm("2026-01-01-alpha-gamma-x",
                           {"alpha": "measured_alpha", "gamma": "measured_gamma"})
    assert "not decidable" in str(excinfo.value)


def test_the_longest_matching_slug_wins():
    got = RS.profile_for_arm(
        "2026-01-01-nvidia_h200_nvl-x",
        {"nvidia_h200": "measured_nvidia_h200",
         "nvidia_h200_nvl": "measured_nvidia_h200_nvl"})
    assert got == "measured_nvidia_h200_nvl"


def test_writing_twice_leaves_the_files_byte_identical(arm_copy, tmp_path):
    # Put it back to the pre-rescore state so the first pass has work to do.
    for path in RS.report_paths(arm_copy):
        doc = json.loads(path.read_text())
        for key in ("ridge_source", "ridge_band_source", "bandwidth_source",
                    "rescored_utc", "rescored_from"):
            doc.pop(key, None)
        doc["ridge"] = 160.3
        doc["ridge_band"] = [160.3, 176.2]
        path.write_text(json.dumps(doc, indent=2))

    first = _run(["--write", str(tmp_path)])
    assert first.returncode == 0, first.stderr
    assert "wrote 7 report(s)" in first.stdout
    snapshot = {p: p.read_bytes() for p in RS.report_paths(arm_copy)}

    second = _run(["--write", str(tmp_path)])
    assert second.returncode == 0, second.stderr
    assert "wrote 0 report(s)" in second.stdout
    assert {p: p.read_bytes() for p in RS.report_paths(arm_copy)} == snapshot


def test_a_second_pass_does_not_overwrite_the_history_it_recorded(arm_copy,
                                                                  tmp_path):
    _run(["--write", str(tmp_path)])
    for path in RS.report_paths(arm_copy):
        assert json.loads(path.read_text())["rescored_from"]["ridge"] == 160.3


def test_an_empty_root_refuses_rather_than_reporting_success(tmp_path):
    got = _run([str(tmp_path)])
    assert got.returncode == X.REFUSED
    assert "REFUSED" in got.stdout


# --------------------------------------------------------------------------
# the gates, and the self-test that plants each failure
# --------------------------------------------------------------------------

def test_the_self_test_passes():
    got = _run(["--self-test"])
    assert got.returncode == 0, got.stdout + got.stderr
    assert "[FAIL]" not in got.stdout


def test_self_test_fails_when_a_gate_is_broken(monkeypatch):
    """Non-vacuity for the self-test itself.

    `fields_confined` is neutered so it can no longer see a leak, and the
    self-test must notice, because a self-test that cannot return 1 proves
    nothing about the gates it exercises.
    """
    module = _load()
    real = module.untouched
    monkeypatch.setattr(module, "untouched", lambda payload: "identical")
    assert module.self_test() == 1
    monkeypatch.setattr(module, "untouched", real)
    assert module.self_test() == 0


def test_every_gate_is_scored_and_printed_exactly_once():
    got = _run([])
    lines = X.parse_result_lines(got.stdout)
    assert [r.name for r in lines] == [
        "card_resolved", "fields_confined", "idempotent", "ridge_is_own_card"]
    assert all(r.verdict == X.PASS for r in lines)
    assert X.classify_text(got.stdout) == got.returncode


def test_the_plan_states_an_mde_from_the_reports_own_noise():
    got = _run([])
    assert "noise assumption: median timing_spread_median" in got.stdout
    assert "MDE" in got.stdout
    assert "WITHIN one process, so it is a floor" in got.stdout


def test_the_mde_refuses_when_no_report_records_a_spread():
    before = json.loads(A100_MIXTRAL.read_text())
    before.pop("timing_spread_median")
    outcome = RS.Outcome(A100_MIXTRAL, before, None, "planted")
    lines = RS.mde_report([outcome])
    assert any("NOT STATED" in line for line in lines)


def test_the_run_report_carries_a_provenance_block(tmp_path):
    out = tmp_path / "rescore.json"
    got = _run(["--report", str(out)])
    assert got.returncode == 0, got.stderr
    doc = json.loads(out.read_text())
    assert doc["provenance"]["instrument"] == "analysis/rescore"
    assert "git_sha" in doc["provenance"]
    assert doc["reports"] == 26
    assert [g["name"] for g in doc["gates"]] == [
        "card_resolved", "fields_confined", "idempotent", "ridge_is_own_card"]
