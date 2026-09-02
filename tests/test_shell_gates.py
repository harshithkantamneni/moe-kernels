"""The session shell: what it refuses, what it prints, and what it must not write.

Four scripts drive every pod session, and the 2026-09-02 audit found the same
defect in all of them: a gate that could not fail, a default that silently
narrowed the experiment, or a step that wrote into the tracked tree as a side
effect. None of it needed a GPU to go wrong and none of it needs one to test.

Every gate exercised here is exercised in BOTH directions. A test that only ever
sees the passing branch is the shape of check this whole apparatus is being
repaired for: `pod_session.sh` P7 compared a live census against the literal
"2 10" and had printed FAIL on every run for four arms, and nothing noticed,
because nothing had ever asked it to pass.

These run the real scripts as subprocesses. `--dry-run` everywhere, and
`git status --porcelain` is compared before and after, because "the plan writes
nothing" is itself one of the requirements.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_ALL = REPO / "scripts" / "run_all.sh"
SETUP = REPO / "scripts" / "setup_runpod.sh"
POD = REPO / "scripts" / "pod_session.sh"
PUBLISH = REPO / "scripts" / "publish_results.sh"
PY = sys.executable


def sh(script: Path, *args: str, env: dict | None = None, cwd: Path = REPO):
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "MOE_PYTHON": PY}
    base.update(env or {})
    return subprocess.run(["bash", str(script), *args], cwd=cwd, text=True,
                          capture_output=True, env=base)


def tree_state() -> str:
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                          capture_output=True, text=True).stdout


# --------------------------------------------------------------------------
# R1: the documented one-command session swept only torch
# --------------------------------------------------------------------------

def test_dry_run_names_the_environments_or_refuses(tmp_path):
    """The audit's own reproduction command, run verbatim.

    `run_all.sh --profile standard --dry-run` used to print neither the envs nor
    a refusal: `ENVS` defaulted to "base" and the dry run never looked at it, so
    the laptop pre-check could not expose the defect it exists to expose. Either
    answer is acceptable to this test and both are informative; silence is not.
    """
    r = sh(RUN_ALL, "--profile", "standard", "--dry-run",
           env={"MOE_VENV_ROOT": str(tmp_path / "venvs")})
    out = r.stdout + r.stderr
    assert "envs: base,vllm,sglang" in out or "REFUSE" in out, out


def test_a_box_without_the_framework_venvs_says_what_would_be_missing(tmp_path):
    """The failing branch: nothing installed, so the sweep would be torch only.

    A stranger burned a 45-minute H200 session this way and got an arm with no
    vLLM and no SGLang rows, which is none of the kernels the study's claims are
    about. The run exited 0 and printed a summary.
    """
    r = sh(RUN_ALL, "--profile", "standard", "--dry-run",
           env={"MOE_VENV_ROOT": str(tmp_path / "empty")})
    err = r.stderr
    assert "REFUSE" in err
    assert "vllm" in err and "sglang" in err
    assert "setup_runpod.sh" in err, "a refusal has to say how to proceed"


def test_the_same_shortfall_stops_the_session_where_it_would_cost_something(tmp_path):
    """And the other half: the refusal has to STOP something.

    A laptop rehearsal exits 0 after printing the shortfall, because a check
    that can only fail on the machine you rehearse on is one you learn to skip
    and nothing there is metered. With a card visible the same shortfall is a
    session about to spend pod minutes on a third of the experiment, and it
    stops. The card is faked, so the branch is exercised off-GPU.
    """
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "nvidia-smi").write_text("#!/bin/sh\necho 'NVIDIA H200'\n")
    (fake / "nvidia-smi").chmod(0o755)
    r = sh(RUN_ALL, "--profile", "standard", "--dry-run",
           env={"MOE_VENV_ROOT": str(tmp_path / "empty"),
                "PATH": f"{fake}:{os.environ.get('PATH', '/usr/bin:/bin')}"})
    assert r.returncode != 0, r.stdout + r.stderr
    assert "REFUSE" in r.stderr
    assert "stopping before anything is spent" in r.stderr


def test_naming_the_environments_explicitly_warns_and_continues(tmp_path):
    """The passing branch of the same gate, and the distinction it draws.

    `--envs base` is a decision an operator typed. An auto-detected shortfall is
    the box quietly sweeping a third of the experiment. The first gets a warning
    and runs; the second stops.
    """
    r = sh(RUN_ALL, "--profile", "standard", "--envs", "base", "--dry-run",
           env={"MOE_VENV_ROOT": str(tmp_path / "empty")})
    out = r.stdout + r.stderr
    assert "envs: base" in out
    assert "WARNING" in r.stderr and "REFUSE" not in r.stderr
    assert "planned cells" in r.stdout, "the plan itself must still print"


def test_the_plan_states_a_minimum_detectable_effect(tmp_path):
    """B14: no arm stated an MDE, so any difference could be read as real.

    The number has to come from a measured spread rather than a prior, so the
    line names where its sigma came from.
    """
    r = sh(RUN_ALL, "--profile", "standard", "--envs", "base", "--dry-run",
           env={"MOE_VENV_ROOT": str(tmp_path / "empty")})
    line = [ln for ln in r.stdout.splitlines() if "MDE:" in ln]
    assert line, r.stdout
    assert "sigma" in line[0] and "published rows" in line[0], line[0]


def test_no_dry_run_writes_anything_into_the_tree(tmp_path):
    """R8, over all three scripts that have a dry run.

    A rehearsal that dirties the checkout makes every row of the session that
    follows it carry git_dirty=True, which is how 44,872 of 100,144 published
    rows became unreproducible from the commit they name.
    """
    before = tree_state()
    sh(RUN_ALL, "--profile", "standard", "--dry-run",
       env={"MOE_VENV_ROOT": str(tmp_path / "venvs")})
    sh(SETUP, "--dry-run", "--base-python", "3.12")
    sh(POD, "--dry-run", "--skip-tests", "--no-download",
       "--session-dir", str(tmp_path / "session"))
    assert tree_state() == before, "a dry run modified the working tree"


# --------------------------------------------------------------------------
# R2: the environment pin was a hand edit, and the resolved sets were unread
# --------------------------------------------------------------------------

def test_the_pin_is_a_flag_and_names_the_resolved_set(tmp_path):
    """`docs/RUNPOD.md` told the operator to edit this script to pin torch.

    The edit dirtied a tracked file, so P1 failed and every row of the session
    was stamped git_dirty. The pin is a flag now, and the plan says which
    requirements file each environment would install from -- which no code path
    had ever read.
    """
    before = tree_state()
    r = sh(SETUP, "--dry-run", "--base-python", "3.12")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("resolved-base.txt") >= 1, r.stdout
    assert "isolated" in r.stdout
    assert tree_state() == before


def test_a_pinned_interpreter_with_inherited_site_packages_is_refused():
    """Refuse rather than pick one. The image's torch is built for the image's
    interpreter, and inheriting it into a venv on another one is the
    torchvision ABI trap `capture_traces.py` works around."""
    r = sh(SETUP, "--dry-run", "--base-python", "3.12", "--system-site-packages")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "REFUSING" in r.stderr


def _requirements(tmp_path: Path, plain: str, resolved: str) -> Path:
    d = tmp_path / "requirements"
    d.mkdir()
    (d / "base.txt").write_text(plain)
    (d / "resolved-base.txt").write_text(resolved)
    return d


def test_check_passes_when_the_resolved_set_agrees(tmp_path):
    d = _requirements(tmp_path, "numpy\ntransformers>=4.44,<4.54\n",
                      "numpy==2.1.0\ntransformers==4.53.3\ntorch==2.13.0+cu130\n")
    r = sh(SETUP, "--check", "base", env={"MOE_REQUIREMENTS_DIR": str(d)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "agrees with base.txt" in r.stdout


def test_check_fails_on_the_drift_that_is_actually_committed(tmp_path):
    """The real one: `resolved-base.txt` records transformers 5.15.1 against the
    `<4.54` cap `base.txt` grew on 2026-09-01, and it predates nvidia-ml-py."""
    d = _requirements(tmp_path, "numpy\ntransformers>=4.44,<4.54\nnvidia-ml-py\n",
                      "numpy==2.1.0\ntransformers==5.15.1\ntorch==2.13.0\n")
    r = sh(SETUP, "--check", "base", env={"MOE_REQUIREMENTS_DIR": str(d)})
    assert r.returncode == 1, r.stdout + r.stderr
    assert "STALE transformers" in r.stdout
    assert "nvidia-ml-py" in r.stdout
    # The missing +cu130 tag is true and cannot be fixed off a pod, so it is a
    # note rather than a reason to refuse: a check that can only fail is one
    # nobody reads.
    assert "note  torch==2.13.0 carries no local tag" in r.stdout


def test_the_committed_base_set_is_the_stale_one_the_audit_found():
    """Against the repo's own files, so the finding stays visible until it is
    fixed on a pod with `--fresh`."""
    r = sh(SETUP, "--check", "base")
    assert r.returncode == 1, r.stdout
    assert "transformers" in r.stdout


# --------------------------------------------------------------------------
# R9: the clock sampler's only source
# --------------------------------------------------------------------------

def test_nvidia_ml_py_is_a_declared_requirement():
    """Without it `torch.cuda.clock_rate` raises, every KernelTiming records
    clock_source "none", and the LEVEL flag is None on every row, which defeats
    the whole point of sampling the clock under load."""
    text = (REPO / "requirements" / "base.txt").read_text()
    assert "nvidia-ml-py" in text


def test_the_preflight_gate_for_the_clock_source_appears_and_can_fail(tmp_path):
    """Both branches. The PASS branch runs against the repo's own base.txt; the
    FAIL branch points the gate at a requirements directory without the line,
    which is the state of the pod venv the audit measured."""
    ok = sh(POD, "--dry-run", "--skip-tests", "--no-download",
            "--session-dir", str(tmp_path / "s1"))
    assert "P13c  PASS  nvidia-ml-py is a declared requirement" in ok.stdout

    d = tmp_path / "requirements"
    d.mkdir()
    (d / "base.txt").write_text("numpy\npandas\n")
    bad = sh(POD, "--dry-run", "--skip-tests", "--no-download",
             "--session-dir", str(tmp_path / "s2"),
             env={"MOE_REQUIREMENTS_DIR": str(d)})
    line = [ln for ln in bad.stdout.splitlines() if ln.startswith("P13c")]
    assert line and "FAIL" in line[0], bad.stdout
    assert "clock_source" in bad.stdout


# --------------------------------------------------------------------------
# R3: gates that could not pass
# --------------------------------------------------------------------------

def _repo_farm(tmp_path: Path) -> Path:
    """A throwaway repo root: the real code, a fixture `results/published`.

    P7 reads `results/published` relative to the repository root, and the point
    of the test is to change what is in it. Symlinking the code rather than
    copying keeps the scripts under test byte-identical to the ones that ship.
    """
    root = tmp_path / "farm"
    root.mkdir()
    for name in ("moe", "scripts", "tests", "requirements", "pyproject.toml"):
        (root / name).symlink_to(REPO / name)
    (root / "results" / "published").mkdir(parents=True)
    return root


def _fixture_arm(published: Path, name: str, *, dtype: str = "bf16") -> Path:
    """One published arm with enough of a row for `entitled_ridge` to answer."""
    arm = published / name
    arm.mkdir()
    from moe.bench.schema import COLUMNS, SCHEMA_VERSION
    with (arm / "run_x_base.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        row = dict.fromkeys(COLUMNS, "")
        row.update(schema_version=SCHEMA_VERSION, run_id="x", env_name="base",
                   gpu_name="NVIDIA H200", impl="torch_grouped_mm", model="toy",
                   num_tokens=32, ms_p50=1.0, correctness_passed="True",
                   dtype=dtype, achieved_bw_gbps=4374.0,
                   achieved_peak_tflops=716.0)
        w.writerow(row)
    return arm


def _census(published: Path, rows: dict[str, str]) -> None:
    """The generated census, in the shape `provenance_report` writes."""
    lines = ["# fixture census", "",
             "| arm | verdict | ceilings | commit | checked_on vs rows | ridge it may quote |",
             "|---|---|---|---|---|---|"]
    for arm, ridge in rows.items():
        lines.append(f"| `{arm}` | same_session | agree | matches | within the sweep | {ridge} |")
    (published / "CALIBRATION_PROVENANCE.md").write_text("\n".join(lines) + "\n")


def test_p7_passes_against_the_repositorys_own_census(tmp_path):
    """It could not, before. P7 compared the live refusal census against the
    literal "2 10", written when there were ten arms; there are fourteen and
    five refuse, so it printed FAIL on every run at HEAD and `main` returned 1
    whatever the GPU did."""
    r = sh(POD, "--dry-run", "--skip-tests", "--no-download",
           "--session-dir", str(tmp_path / "session"))
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("P7")]
    assert line, r.stdout
    assert "PASS" in line[0], line[0]
    # The two kinds of refusal are told apart, which the count could not do.
    assert "on provenance" in line[0] and "no rows" in line[0]


def test_p7_fails_when_an_arm_is_not_in_the_census(tmp_path):
    """The planted failure: an arm lands and the census is not regenerated.

    That is the state that made "5" look like a regression when four of the five
    refusals were arms that had never been listed.
    """
    root = _repo_farm(tmp_path)
    published = root / "results" / "published"
    _fixture_arm(published, "2026-01-01-fixture-listed")
    _fixture_arm(published, "2026-01-02-fixture-unlisted")
    _census(published, {"2026-01-01-fixture-listed": "163.7"})
    r = sh(root / "scripts" / "pod_session.sh", "--dry-run", "--skip-tests",
           "--no-download", "--session-dir", str(tmp_path / "session"),
           cwd=root, env={"PYTHONPATH": str(REPO)})
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("P7")]
    assert line, r.stdout
    assert "FAIL" in line[0], line[0]
    assert "fixture-unlisted" in line[0]


def test_the_dry_run_prints_the_download_it_would_actually_run(tmp_path):
    """It printed `huggingface-cli download ...`, which is neither the command
    the real branch runs nor on PATH: the CLI lives inside the venv."""
    r = sh(POD, "--dry-run", "--skip-tests", "--session-dir", str(tmp_path / "s"))
    assert "huggingface-cli" not in r.stdout
    assert "snapshot_download" in r.stdout


def test_run_all_does_not_hide_a_failed_pull_behind_a_pipeline():
    """`git pull --ff-only 2>&1 | tail -2 || echo skipped` reports whether TAIL
    succeeded, which it always does, so a failed pull was indistinguishable
    from a clean one and the session swept an unknown tree."""
    text = RUN_ALL.read_text()
    assert "git pull --ff-only 2>&1 | tail" not in text
    assert 'if pull_out="$(git pull --ff-only 2>&1)"' in text


# --------------------------------------------------------------------------
# R7: .gitignore dropped the raw evidence
# --------------------------------------------------------------------------

def _ignored(rel: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", rel],
                          capture_output=True).returncode == 0


@pytest.mark.parametrize("rel", [
    "results/published/x/ptx/kernel.ptx",
    "results/published/x/kernel.cubin",
    "results/published/x/counters.nsys-rep",
    "results/published/x/counters.ncu-rep",
    "results/published/x/plots/scaling.png",
    "results/published/x/models/routing.json",
])
def test_raw_evidence_under_published_is_committable(rel):
    """`git check-ignore` confirmed every one of these was dropped. The C3 PTX
    evidence survives only as tarballs because of it, and `git add` says
    nothing when a rule swallows a file."""
    assert not _ignored(rel), f"{rel} is still ignored"


@pytest.mark.parametrize("rel", [
    "ptx/kernel.ptx",
    "traces/raw/x.nsys-rep",
    "results/published/x/weights.safetensors",
    "results/published/x/model.bin",
])
def test_what_stays_ignored_stays_ignored(rel):
    """The other direction, which a blanket `!results/published/**` would have
    broken: raw dumps OUTSIDE the published tree are still noise, and weight
    formats are ignored everywhere including inside it, because one mixtral
    shard is gigabytes."""
    assert _ignored(rel), f"{rel} is no longer ignored"


# --------------------------------------------------------------------------
# R6: the missing commit and the unchecked git_sha column
# --------------------------------------------------------------------------

def test_the_sha_checker_self_test_passes():
    r = subprocess.run([PY, str(REPO / "scripts" / "check_published_shas.py"),
                        "--self-test"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_sha_checker_self_test_can_fail(monkeypatch):
    """The planted failure. A self-test that cannot return 1 proves nothing,
    so the branch is exercised by making every case report DONE: the two cases
    that must not be DONE then disagree with their expectation."""
    sys.path.insert(0, str(REPO / "scripts"))
    import check_published_shas as CH

    monkeypatch.setattr(CH, "run", lambda *a, **k: CH.EX.DONE)
    assert CH.self_test() == 1


def test_the_repositorys_own_published_tree_still_names_a_lost_commit():
    """2,100 ridge-resolution rows cite a commit rewritten by the `git pull
    --rebase` this project's own publish path recommends. Asserted so the
    finding cannot quietly disappear from the record."""
    r = subprocess.run([PY, str(REPO / "scripts" / "check_published_shas.py")],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 1, r.stdout
    assert "MISSING    7eecff427626" in r.stdout
    assert "RESULT: CLAIM shas_resolvable FAIL" in r.stdout


# --------------------------------------------------------------------------
# R5 and R6 through the publish path
# --------------------------------------------------------------------------

def _write_run(results: Path, run_id: str, env: str, rows, git_sha: str = ""):
    from moe.bench.schema import COLUMNS, SCHEMA_VERSION
    path = results / f"run_{run_id}_{env}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for flush, graph, ms in rows:
            row = dict.fromkeys(COLUMNS, "")
            row.update(schema_version=SCHEMA_VERSION, run_id=run_id,
                       env_name=env, gpu_name="NVIDIA H200",
                       impl="torch_grouped_mm", model="toy", num_tokens=32,
                       ms_p50=ms, correctness_passed="True", l2_flush=flush,
                       cuda_graph=graph, git_sha=git_sha)
            w.writerow(row)
    return path


def _publish(results: Path, published: Path, *args: str):
    return sh(PUBLISH, "--dry-run", *args,
              env={"MOE_RESULTS_DIR": str(results),
                   "MOE_PUBLISH_ROOT": str(published)})


def test_the_summary_names_the_bases_present_and_reports_medians(tmp_path):
    """The heading used to assert "L2-flushed eager rows" and take the MINIMUM
    ms_p50 per cell. The alpha-0558 arm has zero flushed rows, so its table was
    EMPTY under a heading claiming a basis it never ran, and any arm that did
    have them got a best-of table."""
    results, published = tmp_path / "results", tmp_path / "published"
    results.mkdir()
    _write_run(results, "aa1", "base",
               [("True", "False", 1.0), ("True", "False", 2.0),
                ("False", "True", 9.0)])
    r = _publish(results, published, "--label", "two-basis")
    assert r.returncode == 0, r.stdout + r.stderr
    text = next(iter(sorted(published.glob("*/SUMMARY.md")))).read_text()
    assert "L2-flushed eager" in text and "warm-L2 cuda-graph" in text
    assert "Median ms_p50" in text
    # the median of 1.0 and 2.0, not the minimum the old table printed
    assert "| 1.5000 |" in text
    assert "| 1.0000 |" not in text


def test_a_report_without_its_cells_is_refused(tmp_path):
    """`find results/published -name cells.csv` returned zero across every arm.
    A report is a verdict; the cells are the measurement it was read off."""
    results, published = tmp_path / "results", tmp_path / "published"
    results.mkdir()
    _write_run(results, "aa1", "base", [("True", "False", 1.0)])
    arm = tmp_path / "sweepdir"
    arm.mkdir()
    (arm / "report.json").write_text("{}")
    r = _publish(results, published, "--label", "no-cells", "--reports", str(arm))
    assert r.returncode != 0
    assert "cells.csv" in r.stderr


def test_a_report_with_its_cells_is_published_beside_them(tmp_path):
    results, published = tmp_path / "results", tmp_path / "published"
    results.mkdir()
    _write_run(results, "aa1", "base", [("True", "False", 1.0)])
    arm = tmp_path / "sweepdir"
    arm.mkdir()
    (arm / "report.json").write_text("{}")
    (arm / "cells.csv").write_text("block_m,ms_p50\n64,1.0\n")
    r = _publish(results, published, "--label", "with-cells", "--reports", str(arm))
    assert r.returncode == 0, r.stdout + r.stderr
    dest = next(iter(sorted(published.glob("*"))))
    assert (dest / "sweepdir.report.json").is_file()
    assert (dest / "sweepdir.cells.csv").is_file()


def test_publishing_a_row_whose_commit_is_lost_refuses_the_commit(tmp_path):
    """2,100 published rows cite a commit rewritten by the `git pull --rebase`
    this script recommends when a push is rejected, and nothing checked.

    A dry run stages and reports rather than refusing, because its contract is
    "touch no git" and the finding is worth more staged than withheld; the
    refusal is on the commit, which is the act that would make the arm
    permanent. Both halves are asserted here: the WOULD-REFUSE line, and the
    report landing inside the arm where the next reader finds it.
    """
    results, published = tmp_path / "results", tmp_path / "published"
    results.mkdir()
    _write_run(results, "aa1", "base", [("True", "False", 1.0)],
               git_sha="7eecff427626c40795b9543a10122a6d86595ab2")
    r = _publish(results, published, "--label", "lost-commit")
    assert "WOULD REFUSE TO COMMIT" in r.stdout, r.stdout
    assert "--allow-missing-sha" in r.stdout
    dest = next(iter(sorted(published.glob("*"))))
    assert "MISSING" in (dest / "GIT_SHA_CHECK.txt").read_text()


def test_the_refusal_is_wired_to_the_commit_not_only_to_the_message():
    """The line that does it, asserted directly: a refusal that only ever runs
    under --dry-run would be a message and not a gate."""
    text = PUBLISH.read_text()
    assert "REFUSING TO PUBLISH: a row names a commit" in text
    assert "SHA_OK == 0" in text
    # and the check itself runs before the commit, not after it
    assert text.index("check_published_shas.py --arm") < text.index('git add "$DEST"')


def test_the_override_writes_its_reason_into_the_summary(tmp_path):
    """An override that leaves no trace is the same as no check. The reason is
    mandatory, and it lands in the file a reader opens."""
    results, published = tmp_path / "results", tmp_path / "published"
    results.mkdir()
    _write_run(results, "aa1", "base", [("True", "False", 1.0)],
               git_sha="7eecff427626c40795b9543a10122a6d86595ab2")
    r = _publish(results, published, "--label", "lost-commit",
                 "--allow-missing-sha", "pod commit lost to a rebase, kept for the record")
    assert r.returncode == 0, r.stdout + r.stderr
    text = next(iter(sorted(published.glob("*/SUMMARY.md")))).read_text()
    assert "allow-missing-sha" in text
    assert "pod commit lost to a rebase" in text
    assert "MISSING" in text


def test_figures_are_drawn_from_the_arm_and_never_copied_from_the_repo_root():
    """`publish_results.sh:241-244` copied every PNG out of the repo-root
    `plots/` directory that `run_all.sh` had filled from the whole results
    volume: the ridge-resolution arm shipped a toy plot its CSVs do not
    contain, and a bf16-only arm shipped six fp8 figures."""
    text = PUBLISH.read_text()
    assert "cp plots/*.png" not in text
    assert 'scripts/plot.py --results "$DEST"' in text
