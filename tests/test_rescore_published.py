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

THE PRE-RESCORE DOCUMENT IS A COMMITTED FIXTURE, NOT A `git show`. The first
version of these tests read the baseline with `git show HEAD:<report>`, which
was true only while HEAD was the commit BEFORE the rescoring. The moment the
rescoring was committed, `HEAD:` served the rescored file, `before` and `after`
became the same document, and the confinement test compared it with itself:
one assertion failed outright and its neighbour passed for no reason at all. A
baseline read through a moving ref is not a baseline. So the withdrawn document
is checked in beside these tests, and `test_the_fixture_is_the_real_pre_rescore
_document` proves it is the genuine pre-image by rescoring it and getting the
committed file back, which no fabricated fixture would do.
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

#: `A100_MIXTRAL` exactly as it stood before the rescoring, copied out of
#: `apparatus-standard` at 456e1e4 (blob ce590c80) and committed here. Pinned as
#: a file rather than fetched from a ref because every ref that named the
#: pre-rescore tree at the time of writing -- `HEAD`, `HEAD~1`,
#: `apparatus-standard` -- names the rescored tree once this work lands.
BEFORE_FIXTURE = Path(__file__).with_name("test_rescore_published_before.json")


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


# --- every rescored arm explains itself, from its own reports ----------------
#
# MEASURED, 2026-09-02 (audit B2). Only the A100 arm got a NOTE, while the two
# H200 arms were rescored 160.3 -> 162.8 with nothing inside their directories
# saying so -- and every report in all three points a reader at "this arm's
# NOTE.md", so on two of the three that pointer resolved to nothing. The three
# checks below are what a copied NOTE fails: it names the wrong directory, it
# states the wrong report count, and it quotes a ridge its own reports do not
# carry.

def _rescored_arms() -> dict[Path, list[Path]]:
    arms: dict[Path, list[Path]] = {}
    for path in RS.report_paths(PUBLISHED):
        arms.setdefault(path.parent, []).append(path)
    return arms


def test_every_rescored_arm_has_a_note_written_from_its_own_reports():
    arms = _rescored_arms()
    assert len(arms) == 3, sorted(a.name for a in arms)
    for arm, reports in arms.items():
        note = (arm / "NOTE.md")
        assert note.exists(), f"{arm.name} was rescored and says nothing about it"
        text = note.read_text()
        assert f"results/published/{arm.name}" in text, arm.name
        assert f"{len(reports)} `*.report.json`" in text, arm.name
        ridges = {json.loads(p.read_text())["ridge"] for p in reports}
        assert len(ridges) == 1, (arm.name, ridges)
        assert str(ridges.pop()) in text, arm.name


def test_every_report_carries_the_current_why():
    """`rescored_from` is stamped once and never rewritten, so it can drift.

    `stamp` preserves an existing `rescored_from` on purpose: a second pass
    reading its own output would otherwise overwrite the withdrawn ridge with
    the current one. The cost is that editing `WITHDRAWN_RIDGE_WHY` does NOT
    reach the 26 already-stamped reports, and nothing else would notice. This
    notices.

    The `2026-08-26` clause is checked by name because that is what the string
    said until this test existed, and it was false: the arms of that date quote
    160.4 and 162.8, while 160.3 is calibration md5 4d84542b from 2026-08-28.
    """
    assert "2026-08-26" not in RS.WITHDRAWN_RIDGE_WHY
    assert "NOTE.md" in RS.WITHDRAWN_RIDGE_WHY
    for path in RS.report_paths(PUBLISHED):
        doc = json.loads(path.read_text())
        assert doc["rescored_from"]["why"] == RS.WITHDRAWN_RIDGE_WHY, path


def test_the_invocation_the_docstring_opens_with_actually_runs():
    """The first documented command used to be `argparse` error 2.

    A usage line that cannot be typed is the same defect class this slice
    exists to close, so the flag is now real and the plan it prints is checked
    to be the same plan the no-flag default prints.
    """
    usage = RS.__doc__.splitlines()[2].split("#")[0].split()
    assert usage[:3] == ["python", "scripts/rescore_published_reports.py",
                         "--dry-run"]
    dry = _run(["--dry-run"])
    assert dry.returncode == 0, dry.stderr
    assert "0 to rewrite" in dry.stdout
    assert dry.stdout == _run([]).stdout


def test_asking_to_write_and_to_dry_run_at_once_refuses(arm_copy, tmp_path):
    """The FAIL branch of the new flag: a contradiction is not resolved.

    Checked on a COPY with real work to do, so a refusal that silently wrote
    anyway would show up as a changed file rather than as an opinion.
    """
    for path in RS.report_paths(arm_copy):
        doc = json.loads(path.read_text())
        doc["ridge"] = 160.3
        path.write_text(json.dumps(doc, indent=2))
    snapshot = {p: p.read_bytes() for p in RS.report_paths(arm_copy)}

    got = _run(["--write", "--dry-run", str(tmp_path)])
    assert got.returncode == X.REFUSED
    assert "REFUSED" in got.stdout
    assert "RESULT:" not in got.stdout
    assert {p: p.read_bytes() for p in RS.report_paths(arm_copy)} == snapshot


def test_running_the_tool_on_the_committed_tree_now_changes_nothing():
    """Idempotence, on the real tree, as a plan that proposes no rewrite."""
    got = _run([])
    assert got.returncode == 0, got.stderr
    assert "0 to rewrite" in got.stdout
    assert "12 to rewrite" not in got.stdout


# --------------------------------------------------------------------------
# confinement: only the four fields move
# --------------------------------------------------------------------------

def test_the_fixture_is_the_real_pre_rescore_document():
    """The fixture earns its place by being the pre-image, not by being asserted.

    Rescoring it reproduces the committed report exactly, apart from the
    timestamp, and the committed report's own `rescored_from` block quotes the
    two numbers the fixture carries. A fixture built by editing the rescored
    file backwards would fail the first check on the four recomputed
    quantities; a stale one would fail it on everything else.
    """
    before = json.loads(BEFORE_FIXTURE.read_text())
    assert before["ridge"] == 160.3
    assert before["ridge_band"] == [160.3, 176.2]
    assert "ridge_source" not in before and "rescored_from" not in before

    committed = json.loads(A100_MIXTRAL.read_text())
    assert committed["rescored_from"]["ridge"] == before["ridge"]
    assert committed["rescored_from"]["ridge_band"] == before["ridge_band"]

    rebuilt = RS.rescored_payload(before, A100_MIXTRAL, SWEEP, "2026-01-01T00:00:00Z")
    assert RS.without_utc(rebuilt) == RS.without_utc(committed)


def test_only_the_registered_fields_differ_from_the_pre_rescore_report():
    """`untouched()` is the check the gate makes, made here against the fixture.

    The four non-vacuity assertions are the point: each names one registered
    field and shows it actually moved, so `untouched()` agreeing is a statement
    about the rest of an 8 KB document rather than about two identical files.
    """
    before = json.loads(BEFORE_FIXTURE.read_text())
    after = json.loads(A100_MIXTRAL.read_text())
    assert RS.untouched(before) == RS.untouched(after)

    assert before["ridge"] == 160.3 and after["ridge"] == 145.8
    assert before["ridge_band"] != after["ridge_band"]
    assert before["bracketing"]["horizon_rows"] != after["bracketing"]["horizon_rows"]
    moved = [bm for bm, pred in before["predictions"].items()
             if pred["crossing_rows_ridge_lo"]
             != after["predictions"][bm]["crossing_rows_ridge_lo"]]
    assert moved, "no prediction moved, so confinement is being checked on a no-op"


def test_the_gate_verdicts_ai_caps_and_ladders_are_untouched():
    """Ridge independence, the claim NOTE.md rests on, against the real before.

    It re-asserts that the ridge moved first. Without that line this test would
    keep passing if a later rescoring quietly moved a gate AND the baseline it
    is compared against, which is exactly how its `git show HEAD:` ancestor
    passed while comparing the rescored file with itself.
    """
    before = json.loads(BEFORE_FIXTURE.read_text())
    after = json.loads(A100_MIXTRAL.read_text())
    assert before["ridge"] != after["ridge"], "nothing was rescored; see the fixture"
    assert before["gates"] == after["gates"]
    assert before["ladder"] == after["ladder"]
    assert before["plateau_tflops"] == after["plateau_tflops"]
    assert before["compute_reference"] == after["compute_reference"]
    assert before["predictions"].keys() == after["predictions"].keys()
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
