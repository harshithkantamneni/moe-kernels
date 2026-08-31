"""A published arm must be able to prove its calibration is its own.

MEASURED, and this is the state the check exists to make impossible to repeat.
`results/published/2026-08-28-nvidia_h200-h200-whole-layer/measured.yaml` is
BYTE-IDENTICAL to the one in `-fp8-refixed`, md5 db981ff9aadc08c61f06fb3965d95c3a
for both. Reconstructed from git, all times UTC:

    18:08:21-19:21:43  whole-layer sweeps at commit 873183a, every row stamped
                       701.6129 TFLOP/s and 4377.2122 GB/s out of the calibration
                       then on disk (md5 4d84542b, committed 04:12 as a6ee65d,
                       and the same file -h200-v2lite had used fifteen hours
                       earlier)
    19:29:01           89f9f7a overwrites moe/bench/hardware/measured_nvidia_h200
                       .yaml with a new calibration: 770.9163 and 4374.4897
    19:35:51           publish_results.sh publishes whole-layer and copies the
                       NEW file, because that copy has always taken whatever is
                       on disk at publish time
    19:49:38           -fp8-refixed starts sweeping, at commit 5687de8, against
                       the calibration that is now correctly beside it too

So the whole-layer arm ships a ruler measured after its own rows stopped, and
the ruler its rows actually used belongs to a different arm. NEITHER 176.2 nor
160.3 is a same-session ridge for it, which is why claim C5 is scored against
the band 0.81-0.91 instead of against 0.83. Nothing checked, so nobody noticed
for three days.

The ceilings are what catch it. The commit does not: `measured_commit` is the
commit the calibration RAN at and the sweep runs one commit later by
construction, so six of ten arms differ there and none of the six is defective.
`checked_on` does not either: it has day resolution and both sessions were on
2026-08-28.

Everything here runs off-GPU, against the committed rows.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest
import yaml

from moe.bench.calibrate import CalibrationStamp, read_stamp, ridge_flop_per_byte
from moe.bench.published import (
    CEILINGS_DISAGREE,
    DERIVED_MARKER,
    DIFFERENT_SESSION,
    REPORT_COMMAND,
    SAME_SESSION,
    UNKNOWN,
    calibration_provenance,
    derived_from,
    entitled_ridge,
    provenance_report,
)

REPO = Path(__file__).resolve().parents[1]
PUBLISHED = REPO / "results" / "published"
WHOLE_LAYER = PUBLISHED / "2026-08-28-nvidia_h200-h200-whole-layer"
V2LITE = PUBLISHED / "2026-08-28-nvidia_h200-h200-v2lite"
RECALIBRATED = PUBLISHED / "2026-08-26-nvidia_h200-full-three-way-recalibrated"

published_only = pytest.mark.skipif(
    not PUBLISHED.exists(), reason="results/published is not checked out here")


# --- writing a synthetic arm --------------------------------------------------
#
# Small enough to read, and shaped exactly like a real one: run_*.csv beside a
# measured.yaml. `merged.csv` is deliberately never written, because the check
# must read the per-venv arms and not the file rebuilt from them.


def write_arm(root: Path, *, name: str = "arm", peak: float = 700.0,
              bandwidth: float = 4300.0, yaml_peak: float | None = None,
              yaml_bandwidth: float | None = None, sha: str = "a" * 40,
              yaml_commit: str | None = None, timestamp: str = "2026-08-28T03:00:00",
              checked_on: str = "2026-08-28", gpu: str = "NVIDIA H200",
              yaml_gpu: str | None = None, dtype: str = "bf16",
              ms: float = 1.5, calibration: bool = True) -> Path:
    """One arm whose rows and calibration can be made to agree or not."""
    arm = root / name
    arm.mkdir(parents=True, exist_ok=True)
    fields = ["schema_version", "run_id", "timestamp", "git_sha", "gpu_name",
              "dtype", "ms_p50", "achieved_peak_tflops", "achieved_bw_gbps"]
    with (arm / "run_deadbeef_base.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"schema_version": 4, "run_id": "deadbeef",
                         "timestamp": timestamp, "git_sha": sha, "gpu_name": gpu,
                         "dtype": dtype, "ms_p50": ms,
                         "achieved_peak_tflops": peak, "achieved_bw_gbps": bandwidth})
    if calibration:
        peaks = yaml_peak if yaml_peak is not None else peak
        (arm / "measured.yaml").write_text(yaml.safe_dump({
            "name": f"{yaml_gpu or gpu} (measured)",
            "verified": True,
            "checked_on": checked_on,
            "measured_commit": sha if yaml_commit is None else yaml_commit,
            "measured_dirty": False,
            "memory": {"bandwidth_tb_s": (yaml_bandwidth if yaml_bandwidth
                                          is not None else bandwidth) / 1000.0},
            "compute_dense_tflops": {dtype: peaks} if peaks else {},
            "detail": {"gpu_name": yaml_gpu or gpu, "ceiling_pattern": "triad"},
        }, sort_keys=False))
    return arm


# --- the verdict --------------------------------------------------------------

def test_matching_ceilings_commit_and_date_are_a_same_session_calibration(tmp_path):
    arm = write_arm(tmp_path)
    assert calibration_provenance(arm).verdict == SAME_SESSION


def test_ceilings_that_differ_from_the_file_beside_them_are_reported_as_such(tmp_path):
    """The whole-layer defect, in miniature: the rows were stamped from one
    calibration and a different one was copied in beside them."""
    arm = write_arm(tmp_path, peak=701.61, yaml_peak=770.92)
    prov = calibration_provenance(arm)
    assert prov.verdict == CEILINGS_DISAGREE
    assert prov.blocking_reason
    assert "701.61" in prov.blocking_reason and "770.92" in prov.blocking_reason


def test_a_bandwidth_difference_alone_is_enough(tmp_path):
    """The compute ceiling moves 9.9% between H200 calibrations and bandwidth
    only 0.06%, so the smaller signal has to be checked too or an arm that
    differs only in bandwidth would pass."""
    arm = write_arm(tmp_path, bandwidth=4377.212185121149,
                    yaml_bandwidth=4374.48966354773)
    assert calibration_provenance(arm).verdict == CEILINGS_DISAGREE


def test_an_unrecorded_peak_beside_a_calibration_that_has_one_is_a_mismatch(tmp_path):
    """`achieved_peak_tflops = 0.0` on a row whose bandwidth WAS stamped says the
    calibration loaded at sweep time had no peak for that dtype. A file beside it
    that does have one is a different file. This is the shape `-fp8-three-kernel`
    would take if the fp8 calibration were copied in next to it."""
    arm = write_arm(tmp_path, dtype="fp8_e4m3", peak=0.0, yaml_peak=1409.17)
    prov = calibration_provenance(arm)
    assert prov.verdict == CEILINGS_DISAGREE
    assert "1409.17" in prov.blocking_reason


def test_an_arm_with_nothing_stamped_at_all_is_unrecorded_not_wrong(tmp_path):
    """An arm predating the ceiling columns reads 0.0 everywhere. That is missing
    information, not a contradiction, and calling it a mismatch would condemn
    every old arm on a guess."""
    arm = write_arm(tmp_path, peak=0.0, bandwidth=0.0, yaml_peak=770.92,
                    yaml_bandwidth=4374.49)
    prov = calibration_provenance(arm)
    assert prov.evidence["ceilings"] == "unrecorded"
    assert prov.verdict != CEILINGS_DISAGREE


def test_a_ceiling_survives_the_TB_per_second_round_trip(tmp_path):
    """The yaml stores TB/s and the row stores GB/s, so the value is divided by
    1000 and multiplied by 1000 again before the comparison and can come back an
    ULP away. Exact equality would call every arm in the repo defective."""
    arm = write_arm(tmp_path, bandwidth=4375.202563053733)
    assert calibration_provenance(arm).verdict == SAME_SESSION


def test_a_commit_that_differs_is_different_session_and_not_a_failure(tmp_path):
    """`measured_commit` is the commit the CALIBRATION ran at, and the workflow
    is calibrate, commit the yaml, sweep -- so the sweep runs one commit later by
    construction. Six of the ten published arms differ this way and none of the six
    is defective, so this must not block anything."""
    arm = write_arm(tmp_path, yaml_commit="b" * 40)
    prov = calibration_provenance(arm)
    assert prov.verdict == DIFFERENT_SESSION
    assert prov.blocking_reason is None


def test_a_calibration_dated_after_the_last_row_blocks(tmp_path):
    """Causally impossible for a same-session measurement: the sweep cannot have
    been quoted against a file that did not exist yet."""
    arm = write_arm(tmp_path, timestamp="2026-08-26T00:59:19", checked_on="2026-08-28")
    prov = calibration_provenance(arm)
    assert prov.verdict == DIFFERENT_SESSION
    assert prov.blocking_reason and "after the last row" in prov.blocking_reason


def test_a_derived_arm_declares_the_late_date_and_is_allowed(tmp_path):
    """`recompute_ceilings.py` restamps old rows against a NEW calibration on
    purpose, so its output is exactly the shape the rule above rejects."""
    arm = write_arm(tmp_path, timestamp="2026-08-26T00:59:19", checked_on="2026-08-28")
    (arm / DERIVED_MARKER).write_text("2026-08-26-nvidia_h200-full-three-way\n")
    prov = calibration_provenance(arm)
    assert prov.derived_from == "2026-08-26-nvidia_h200-full-three-way"
    assert prov.blocking_reason is None


def test_a_declaration_does_not_excuse_ceilings_that_disagree(tmp_path):
    """A derived arm's ceilings come FROM the calibration it then copies in, so
    disagreement there means the derivation is broken, not declared."""
    arm = write_arm(tmp_path, peak=701.61, yaml_peak=770.92)
    (arm / DERIVED_MARKER).write_text("some-source-arm\n")
    prov = calibration_provenance(arm)
    assert prov.verdict == CEILINGS_DISAGREE
    assert prov.blocking_reason


def test_the_readme_sentence_is_still_read_as_a_declaration(tmp_path):
    """The arm already on disk predates the marker, and a published directory
    should not have to be edited to be understood."""
    arm = write_arm(tmp_path)
    (arm / "README.md").write_text(
        "# derived\n\nDerived from `2026-08-26-nvidia_h200-full-three-way` by "
        "`scripts/recompute_ceilings.py`.\n")
    assert derived_from(arm) == "2026-08-26-nvidia_h200-full-three-way"


def test_a_readme_that_claims_nothing_is_not_a_declaration(tmp_path):
    arm = write_arm(tmp_path)
    (arm / "README.md").write_text("# an arm\n\nSome notes about the sweep.\n")
    assert derived_from(arm) is None


def test_a_device_disagreement_is_caught_and_named(tmp_path):
    """An A100 calibration beside H200 rows disagrees on the ceilings by a factor
    of three anyway; the device is reported so the reason is not mysterious."""
    arm = write_arm(tmp_path, gpu="NVIDIA H200", yaml_gpu="NVIDIA A100-SXM4-80GB",
                    yaml_peak=262.05, yaml_bandwidth=1798.49)
    prov = calibration_provenance(arm)
    assert prov.verdict == CEILINGS_DISAGREE
    assert prov.evidence["device"] == "disagrees"


def test_a_measured_yaml_that_is_not_there_is_unknown_not_a_pass(tmp_path):
    arm = write_arm(tmp_path, calibration=False)
    prov = calibration_provenance(arm)
    assert prov.verdict == UNKNOWN
    assert prov.blocking_reason


def test_an_arm_with_no_timed_rows_cannot_be_judged(tmp_path):
    """`ms_p50 = 0.0` means the cell never ran, so it carries no ceiling either.
    An arm made entirely of them has nothing to compare."""
    arm = write_arm(tmp_path, ms=0.0, peak=0.0, bandwidth=0.0)
    assert calibration_provenance(arm).verdict == UNKNOWN


def test_an_unrecorded_commit_does_not_by_itself_defeat_the_check(tmp_path):
    """The A100 calibration ships an empty `measured_commit`. The ceilings and
    the date still establish the calibration is the arm's own."""
    arm = write_arm(tmp_path, yaml_commit="")
    prov = calibration_provenance(arm)
    assert prov.verdict == SAME_SESSION
    assert prov.evidence["commit"] == "unrecorded"


def test_the_verdict_carries_its_evidence_rather_than_a_bare_bool(tmp_path):
    """Three signals that disagree with each other in useful ways: a caller has
    to be able to see which one fired."""
    prov = calibration_provenance(write_arm(tmp_path))
    for key in ("ceilings", "commit", "date", "device", "row_commits",
                "yaml_commit", "row_span", "timed_rows"):
        assert key in prov.evidence, key


# --- the ridge an arm may quote ----------------------------------------------

def test_an_arm_with_its_own_calibration_states_its_ridge(tmp_path):
    ridge, why = entitled_ridge(write_arm(tmp_path, peak=712.36, bandwidth=4376.97))
    assert ridge == pytest.approx(162.8, abs=0.05)
    assert "162.8" in why


def test_an_arm_whose_ceilings_disagree_is_refused_and_both_candidates_named(tmp_path):
    """The refusal has to reconstruct C5's problem, not hide it: 160.3 from the
    rows, 176.2 from the file, and no basis for choosing."""
    arm = write_arm(tmp_path, peak=701.6129057225834, bandwidth=4377.212185121149,
                    yaml_peak=770.9162924538318, yaml_bandwidth=4374.48966354773)
    ridge, why = entitled_ridge(arm)
    assert ridge is None
    assert "160.3" in why and "176.2" in why


def test_a_dtype_the_calibration_never_measured_is_refused(tmp_path):
    """19,908 fp8 rows sit next to a bf16-only calibration in
    `-fp8-three-kernel`, which is why every one of them carries
    achieved_peak_tflops 0.0. Quoting them against the bf16 ridge would score
    fp8 work against half its roof."""
    arm = write_arm(tmp_path, dtype="fp8_e4m3", peak=0.0, yaml_peak=0.0)
    ridge, why = entitled_ridge(arm)
    assert ridge is None
    assert "fp8_e4m3" in why


def test_the_reason_comes_back_whether_or_not_the_number_does(tmp_path):
    """Same contract as `filter_superseded`: an analysis that skips an arm
    announces it, and one that uses an arm can print which ruler it used."""
    for arm in (write_arm(tmp_path, name="ok"),
                write_arm(tmp_path, name="bad", peak=700.0, yaml_peak=800.0)):
        _, why = entitled_ridge(arm)
        assert why.strip()


def test_a_ridge_is_TFLOP_per_second_over_GB_per_second_times_1000():
    """1e12 / 1e9. The A100 arm is the arithmetic in one line."""
    assert ridge_flop_per_byte(262.0450496125275, 1798.4902071091922) == \
        pytest.approx(145.7, abs=0.05)
    assert ridge_flop_per_byte(0.0, 4377.2) is None
    assert ridge_flop_per_byte(701.6, 0.0) is None


# --- reading a calibration file ----------------------------------------------

def test_a_calibration_stamp_reads_the_fields_that_identify_a_session(tmp_path):
    arm = write_arm(tmp_path, sha="c" * 40, checked_on="2026-08-28")
    stamp = read_stamp(arm / "measured.yaml")
    assert isinstance(stamp, CalibrationStamp)
    assert stamp.measured_commit == "c" * 40
    assert stamp.checked_on.isoformat() == "2026-08-28"
    assert stamp.ceiling_pattern == "triad"


def test_an_absent_peak_reads_as_None_and_never_as_a_measured_zero(tmp_path):
    """The A100 file has no fp8 key because the silicon has no fp8 tensor
    cores."""
    arm = write_arm(tmp_path)
    stamp = read_stamp(arm / "measured.yaml")
    assert stamp.peak_for("fp8_e4m3") is None
    assert stamp.ridge("fp8_e4m3") is None


def test_an_unquoted_date_reads_the_same_as_a_quoted_one(tmp_path):
    """PyYAML resolves an unquoted 2026-08-28 to a date and a quoted one to a
    string, and the two files look identical in a diff."""
    path = tmp_path / "measured.yaml"
    path.write_text("name: x\nchecked_on: 2026-08-28\nmemory:\n"
                    "  bandwidth_tb_s: 4.3\ncompute_dense_tflops:\n  bf16: 700\n")
    assert read_stamp(path).checked_on.isoformat() == "2026-08-28"


def test_a_calibration_missing_its_provenance_fields_is_read_not_rejected(tmp_path):
    """`recompute.load_calibration_hardware` raises on a missing field because it
    is about to divide by it. This one has to be able to answer `unknown`."""
    path = tmp_path / "measured.yaml"
    path.write_text("name: x\nmemory:\n  bandwidth_tb_s: 4.3\n")
    stamp = read_stamp(path)
    assert stamp.measured_commit == "" and stamp.checked_on is None


# --- the ten arms actually published -----------------------------------------

@published_only
def test_the_ten_published_arms_have_these_verdicts():
    """The state this check exists for, and the place a new one gets declared on
    purpose rather than discovered by md5-ing directories three days later.

    Two arms can prove their calibration is their own. Seven differ on the
    commit or the date while their ceilings match exactly, which means the rows
    WERE stamped from the file beside them. One cannot: whole-layer.
    """
    expected = {
        "2026-08-22-first-smoke": DIFFERENT_SESSION,
        "2026-08-22-standard-sweep": DIFFERENT_SESSION,
        "2026-08-26-nvidia_h200-full-three-way": SAME_SESSION,
        "2026-08-26-nvidia_h200-full-three-way-recalibrated": DIFFERENT_SESSION,
        "2026-08-28-nvidia_a100_sxm4_80gb-a100-cross-card": SAME_SESSION,
        "2026-08-28-nvidia_h200-h200-fp8-refixed": DIFFERENT_SESSION,
        "2026-08-28-nvidia_h200-h200-fp8-three-kernel": DIFFERENT_SESSION,
        "2026-08-28-nvidia_h200-h200-v2lite": SAME_SESSION,
        "2026-08-28-nvidia_h200-h200-whole-layer": CEILINGS_DISAGREE,
        "2026-08-28-nvidia_h200-ridge-resolution": DIFFERENT_SESSION,
    }
    got = {p.name: calibration_provenance(p).verdict
           for p in sorted(PUBLISHED.iterdir()) if p.is_dir()}
    assert got == expected


@published_only
def test_exactly_one_published_arm_cannot_prove_its_calibration_is_its_own():
    """Nine of ten pass. The one that does not is the one C5 needed."""
    blocked = [p.name for p in sorted(PUBLISHED.iterdir()) if p.is_dir()
               and calibration_provenance(p).blocking_reason]
    assert blocked == ["2026-08-28-nvidia_h200-h200-whole-layer"]


@published_only
def test_the_whole_layer_arm_carries_the_v2lite_ruler_and_the_fp8_file():
    """The two halves of the swap, stated as numbers rather than as a story.

    Its rows were stamped from the calibration `-h200-v2lite` shipped fifteen
    hours earlier, and the file beside them is the one `-fp8-refixed` measured
    twenty-eight minutes after these rows stopped.
    """
    prov = calibration_provenance(WHOLE_LAYER)
    assert prov.evidence["row_peak_tflops"] == [("bf16", 701.6129057225834)]
    assert prov.evidence["row_bandwidth_gbps"] == [4377.212185121149]

    v2lite = calibration_provenance(V2LITE)
    assert v2lite.evidence["row_peak_tflops"] == prov.evidence["row_peak_tflops"]
    assert v2lite.evidence["row_bandwidth_gbps"] == prov.evidence["row_bandwidth_gbps"]

    beside = read_stamp(WHOLE_LAYER / "measured.yaml")
    refixed = read_stamp(PUBLISHED / "2026-08-28-nvidia_h200-h200-fp8-refixed"
                         / "measured.yaml")
    assert beside.peak_for("bf16") == refixed.peak_for("bf16") == 770.9162924538318
    assert beside.bandwidth_gbps == refixed.bandwidth_gbps


@published_only
def test_the_whole_layer_arm_is_entitled_to_no_ridge_which_is_what_C5_costs():
    """C5's target is the band 0.81-0.91 rather than 0.83 for exactly this
    reason: `2R/b` scales with the ridge, and this arm has two candidates."""
    ridge, why = entitled_ridge(WHOLE_LAYER)
    assert ridge is None
    assert "160.3" in why and "176.2" in why


@published_only
def test_the_recalibrated_arm_is_allowed_by_its_own_readme():
    """It is a derived arm and its calibration is two days later than its rows on
    purpose. The path `recompute_ceilings.py` has always written must keep
    working without anyone editing a published directory."""
    prov = calibration_provenance(RECALIBRATED)
    assert prov.derived_from == "2026-08-26-nvidia_h200-full-three-way"
    assert prov.blocking_reason is None
    assert prov.evidence["date"] == "2 day(s) after the last row"


@published_only
def test_the_published_ridges_are_the_ones_the_findings_table_quotes():
    """`docs/FINDINGS.md` names 162.8, 160.3 and 145.7 as the three rulers. They
    come out of the rows, so the table and the check cannot drift apart."""
    for arm, ridge in (
        ("2026-08-26-nvidia_h200-full-three-way-recalibrated", 162.8),
        ("2026-08-28-nvidia_h200-ridge-resolution", 162.8),
        ("2026-08-28-nvidia_h200-h200-v2lite", 160.3),
        ("2026-08-28-nvidia_a100_sxm4_80gb-a100-cross-card", 145.7),
    ):
        got, why = entitled_ridge(PUBLISHED / arm)
        assert got == pytest.approx(ridge, abs=0.05), why


@published_only
def test_the_committed_report_is_what_the_code_produces_today():
    """The artifact exists so the next reader inherits the answer instead of
    md5-ing ten directories. It is only worth inheriting if it is current, so it
    is regenerated here and compared byte for byte."""
    report = PUBLISHED / "CALIBRATION_PROVENANCE.md"
    assert report.exists(), f"missing; regenerate with:\n    {REPORT_COMMAND}"
    arms = [p for p in PUBLISHED.iterdir() if p.is_dir()]
    assert report.read_text() == provenance_report(arms), (
        f"stale. Regenerate with:\n    {REPORT_COMMAND}")


# --- the publish gate ---------------------------------------------------------

def run_publish(results: Path, publish_root: Path, *args: str):
    return subprocess.run(
        ["bash", str(REPO / "scripts" / "publish_results.sh"), "--dry-run", *args],
        cwd=REPO, text=True, capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(results.parent),
             "MOE_RESULTS_DIR": str(results), "MOE_PUBLISH_ROOT": str(publish_root)},
    )


@pytest.fixture
def foreign_calibration(tmp_path, monkeypatch):
    """A sweep whose rows were stamped from a calibration that has since been
    overwritten -- which is what had happened by 19:29 on 2026-08-28.

    The device slug is fixed by `gpu_name`, so the run has to claim a GPU whose
    `moe/bench/hardware/measured_*.yaml` exists for the copy to happen at all.
    """
    results = tmp_path / "results"
    results.mkdir()
    fields = ["schema_version", "run_id", "timestamp", "git_sha", "gpu_name",
              "dtype", "ms_p50", "achieved_peak_tflops", "achieved_bw_gbps",
              "correctness_passed", "l2_flush", "cuda_graph", "env_name",
              "model", "num_tokens", "impl"]
    with (results / "run_feedface_base.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"schema_version": 4, "run_id": "feedface",
                         "timestamp": "2026-08-28T18:08:21", "git_sha": "8" * 40,
                         "gpu_name": "NVIDIA H200", "dtype": "bf16", "ms_p50": 1.5,
                         "achieved_peak_tflops": 701.6129057225834,
                         "achieved_bw_gbps": 4377.212185121149,
                         "correctness_passed": "True", "l2_flush": "True",
                         "cuda_graph": "False", "env_name": "base",
                         "model": "toy", "num_tokens": 64, "impl": "toy_impl"})
    (results / "run_feedface_base.manifest.jsonl").write_text("{}\n")
    return results, tmp_path / "published"


def test_publishing_an_arm_whose_calibration_is_not_its_own_fails_loudly(
        foreign_calibration):
    """The gate that would have stopped 19:35:51 on 2026-08-28."""
    results, publish_root = foreign_calibration
    r = run_publish(results, publish_root, "--all", "--label", "whole-layer")
    assert r.returncode != 0, r.stdout
    assert "REFUSING TO PUBLISH" in r.stderr
    assert "4377.212185" in r.stderr and "4374.489664" in r.stderr


def test_the_refused_arm_is_left_on_disk_and_is_not_committed(foreign_calibration):
    """Staged so it can be looked at, never added to git. The fix is usually to
    copy in the calibration the rows actually used, and that needs the directory
    to still be there."""
    results, publish_root = foreign_calibration
    r = run_publish(results, publish_root, "--all", "--label", "whole-layer")
    assert r.returncode != 0
    dest = next(p for p in publish_root.iterdir() if p.is_dir())
    assert (dest / "measured.yaml").exists()
    assert "was NOT committed" in r.stdout


def test_the_override_publishes_and_records_the_admission(foreign_calibration):
    """`--allow-foreign-calibration` is not a way to make the problem quiet. The
    arm carries the verdict afterwards."""
    results, publish_root = foreign_calibration
    r = run_publish(results, publish_root, "--all", "--label", "whole-layer",
                    "--allow-foreign-calibration")
    assert r.returncode == 0, r.stderr
    dest = next(p for p in publish_root.iterdir() if p.is_dir())
    admission = (dest / "CALIBRATION_PROVENANCE.md").read_text()
    assert "ceilings_disagree" in admission
    assert "--allow-foreign-calibration" in admission


def test_every_published_arm_carries_its_verdict_from_now_on(foreign_calibration):
    """Pass or fail, the arm self-describes, the way a SUPERSEDED marker does.
    Re-deriving this from md5sums is what took three days last time."""
    results, publish_root = foreign_calibration
    run_publish(results, publish_root, "--all", "--label", "x",
                "--allow-foreign-calibration")
    dest = next(p for p in publish_root.iterdir() if p.is_dir())
    assert (dest / "CALIBRATION_PROVENANCE.md").exists()
