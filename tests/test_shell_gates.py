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


def test_no_dry_run_writes_anything_into_the_tree_except_pod_sessions_test_step(tmp_path):
    """R8, over all three scripts that have a dry run.

    A rehearsal that dirties the checkout makes every row of the session that
    follows it carry git_dirty=True, which is how 44,872 of 100,144 published
    rows became unreproducible from the commit they name.

    WHAT THIS DOES NOT COVER, SAID IN THE NAME BECAUSE IT WAS ONCE HIDDEN IN A
    FLAG. `pod_session.sh` is run here with `--skip-tests`, and that flag skips
    P12, the one dry-run step that still writes a tracked file: P12 runs the
    repository's own suite, and a test in it rescores into
    `results/published/ANCHOR_RESCORE.txt` through `scripts/memory_branch_anchor.py`,
    which this slice does not own and must not edit. Reading this test as proof
    of the whole requirement is exactly the mistake the flag invited. The rest
    of the requirement -- every dry-run step other than that one, P12 included
    when the suite it runs does not itself write -- is covered by
    `test_the_pod_dry_run_writes_nothing_when_its_test_step_runs_too`.
    """
    before = tree_state()
    sh(RUN_ALL, "--profile", "standard", "--dry-run",
       env={"MOE_VENV_ROOT": str(tmp_path / "venvs")})
    sh(SETUP, "--dry-run", "--base-python", "3.12")
    sh(POD, "--dry-run", "--skip-tests", "--no-download",
       "--session-dir", str(tmp_path / "session"))
    assert tree_state() == before, "a dry run modified the working tree"


def _stand_in_farm(tmp_path: Path) -> Path:
    """A repo checkout whose `tests/` is one trivial test and nothing else.

    P12 runs `pytest tests/` from the repository root, so pointing the root at a
    farm swaps the suite without touching either script. The scripts under test
    stay byte-identical: only the suite they run is a stand-in.
    """
    root = tmp_path / "standin"
    (root / "scripts").mkdir(parents=True)
    for name in ("moe", "requirements", "pyproject.toml"):
        (root / name).symlink_to(REPO / name)
    (root / "scripts" / "pod_session.sh").symlink_to(POD)
    (root / "tests").mkdir()
    (root / "tests" / "test_stand_in.py").write_text(
        "def test_stand_in():\n    assert True\n")
    (root / "results" / "published").mkdir(parents=True)
    (root / ".gitignore").write_text(
        "/moe\n/requirements\n/pyproject.toml\n/scripts\n__pycache__/\n"
        ".pytest_cache/\n")
    for args in (["init", "-q", "-b", "main"],
                 ["config", "user.email", "t@example.com"],
                 ["config", "user.name", "t"],
                 ["add", "-A"], ["commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(root), *args], check=True,
                       capture_output=True)
    return root


def test_the_pod_dry_run_writes_nothing_when_its_test_step_runs_too(tmp_path):
    """R8 without the `--skip-tests` scoping: P12 runs, and nothing is written.

    The step skipped by the other test is the one that writes, so a requirement
    tested only with it skipped is a requirement nobody tested. Here the whole
    dry run executes -- P12 included -- against a checkout whose suite is a
    single trivial test, and the checkout is asked afterwards whether anything
    changed. That isolates the two claims: what `pod_session.sh` itself writes
    (nothing, which is this test), and what the repository's own suite writes
    (`ANCHOR_RESCORE.txt`, whose writer lives in another slice).
    """
    root = _stand_in_farm(tmp_path)
    before_repo = tree_state()
    r = sh(root / "scripts" / "pod_session.sh", "--dry-run", "--no-download",
           "--session-dir", str(tmp_path / "session"), cwd=root,
           env={"PYTHONPATH": str(REPO)})
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("P12")]
    assert line and "PASS" in line[0], r.stdout
    assert "1 passed" in line[0], "the stand-in suite is what ran"
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    assert dirty == "", f"the dry run wrote into the checkout:\n{dirty}"
    assert tree_state() == before_repo, "and it wrote into the real one"


def _setup_farm(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A repo root whose `setup_runpod.sh` records its arguments and installs
    nothing.

    Returns `(root, log, venv_root)`. `run_all.sh` is symlinked, so the script under test is
    byte-identical to the one that ships; only the installer it calls is a stub,
    which is the only way to see what a FRESH pod would have been told to
    install without spending an hour installing it.
    """
    root = tmp_path / "setupfarm"
    (root / "scripts").mkdir(parents=True)
    (root / "moe").symlink_to(REPO / "moe")
    (root / "scripts" / "run_all.sh").symlink_to(RUN_ALL)
    log = tmp_path / "setup.log"
    stub = root / "scripts" / "setup_runpod.sh"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" > "$SETUP_LOG"\n')
    stub.chmod(0o755)
    # A base venv that exists and refuses everything. Past the setup step
    # `run_all.sh` switches to `$MOE_VENV_ROOT/base/bin/python` and runs the
    # calibration with --publish, which writes a TRACKED file -- and `moe/` here
    # is a symlink into the real checkout, so a real interpreter would write it
    # into the repository this suite is asserting about. A stub that exits 1
    # takes the same branches and cannot.
    venv = tmp_path / "stubvenv"
    (venv / "base" / "bin").mkdir(parents=True)
    py = venv / "base" / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 1\n")
    py.chmod(0o755)
    return root, log, venv


def test_a_fresh_pod_is_told_to_install_what_the_profile_is_priced_for(tmp_path):
    """The one-command session could not bootstrap itself.

    Setup was driven by `detect_envs` -- what is installed -- rather than by
    `profile_needs` -- what the profile is priced for. On a fresh pod that is a
    closed loop: nothing is installed, so `base` is detected, `setup_runpod.sh
    base` runs, `base` is re-detected, and the refusal fires on the very command
    `docs/RUNPOD.md` gives for every session. The only way out was
    `--envs base,vllm,sglang`, the undocumented flag R1 exists to remove.
    """
    root, log, venv = _setup_farm(tmp_path)
    r = sh(root / "scripts" / "run_all.sh", "--profile", "standard", cwd=root,
           env={"MOE_NO_PULL": "1", "SETUP_LOG": str(log),
                "MOE_VENV_ROOT": str(venv), "PYTHONPATH": str(REPO)})
    installed = sorted(log.read_text().split())
    assert installed == ["base", "sglang", "vllm"], log.read_text()
    # And the refusal keeps the job it is actually for: the stub installed
    # nothing, so the re-detection still finds only base and the session stops
    # rather than sweeping a third of the experiment.
    assert r.returncode != 0
    assert "REFUSE" in r.stderr and "stopping before anything is spent" in r.stderr


def test_naming_the_environments_never_widens_what_setup_installs(tmp_path):
    """The other branch. `--envs base` is a decision an operator typed, and it
    must not quietly grow a vLLM install because the profile would like one:
    the warning path says the arm is not comparable and runs what was asked."""
    root, log, venv = _setup_farm(tmp_path)
    before = tree_state()
    sh(root / "scripts" / "run_all.sh", "--profile", "standard",
       "--envs", "base", "--skip-tests", cwd=root,
       env={"MOE_NO_PULL": "1", "SETUP_LOG": str(log),
            "MOE_VENV_ROOT": str(venv), "PYTHONPATH": str(REPO)})
    assert log.read_text().split() == ["base"], log.read_text()
    assert tree_state() == before, "the run past the refusal wrote into the tree"


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


def test_the_checks_all_run_before_the_commit_and_not_after_it():
    """Source order, which no run can assert about a step it did not reach.

    Both sha checks must precede `git add`: the first because its refusal is
    what stops the commit, the second because its verdict has to be INSIDE the
    arm being committed. The remote half used to run after the push and append
    to `GIT_SHA_CHECK.txt`, a file already committed, so every successful
    publish ended with a tracked file modified in the working tree and the next
    thing measured on that pod stamped git_dirty=True on every row.
    """
    text = PUBLISH.read_text()
    assert "REFUSING TO PUBLISH: a row names a commit" in text
    assert "SHA_OK == 0" in text
    add = text.index('git add "$DEST"')
    assert text.index("check_published_shas.py --arm") < add
    assert text.rindex("check_published_shas.py --arm") < add
    assert text.index("--require-remote") < add


def _real_repo(tmp_path: Path, *, remote: str | None = None) -> tuple[Path, str]:
    """A throwaway checkout the publish path can really commit into.

    `publish_results.sh` takes its repository root from its own location, so a
    symlinked `scripts/` puts the real script in charge of a fixture repo. The
    code is symlinked and gitignored there; the only thing this repo can gain is
    the published arm, which is what the assertions are about.

    Returns `(root, head_sha)`. `remote` is the URL to add as `origin`: a bare
    repository makes the push succeed, a path to nothing makes it fail, and
    None leaves the caller to pass `--no-push`.
    """
    root = tmp_path / "realrepo"
    root.mkdir()
    for name in ("moe", "scripts", "requirements", "pyproject.toml"):
        (root / name).symlink_to(REPO / name)
    (root / ".gitignore").write_text(
        "/moe\n/scripts\n/requirements\n/pyproject.toml\n__pycache__/\n")
    for args in (["init", "-q", "-b", "main"],
                 ["config", "user.email", "t@example.com"],
                 ["config", "user.name", "t"],
                 ["add", ".gitignore"], ["commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(root), *args], check=True,
                       capture_output=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if remote:
        subprocess.run(["git", "-C", str(root), "remote", "add", "origin", remote],
                       check=True, capture_output=True)
    return root, head


def _status(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                          capture_output=True, text=True).stdout


def _commits(root: Path) -> int:
    out = subprocess.run(["git", "-C", str(root), "rev-list", "--count", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def _real_publish(root: Path, results: Path, *args: str):
    return sh(root / "scripts" / "publish_results.sh", *args, cwd=root,
              env={"MOE_RESULTS_DIR": str(results),
                   "MOE_PUBLISH_ROOT": str(root / "results" / "published"),
                   "PYTHONPATH": str(REPO)})


def test_a_lost_commit_stops_the_real_publish_at_the_commit(tmp_path):
    """The FAIL branch of the gate, EXECUTED rather than grepped for.

    Every publish test before this one passed `--dry-run`, so the `die` this
    requirement is about had never run: a refactor that moved the refusal below
    `git commit` would have left both string assertions green and published the
    arm anyway. Here a real publish into a real repository is asked to commit a
    row citing the commit that 2,100 published rows cite and nobody has.
    """
    root, _head = _real_repo(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    _write_run(results, "aa1", "base", [("True", "False", 1.0)],
               git_sha="7eecff427626c40795b9543a10122a6d86595ab2")
    before = _commits(root)
    r = _real_publish(root, results, "--no-push", "--label", "lost-commit")
    assert r.returncode != 0, r.stdout
    assert "REFUSING TO PUBLISH: a row names a commit" in r.stderr, r.stderr
    assert _commits(root) == before, "the arm was committed anyway"
    # and the arm is still on disk with its report in it, which is the whole
    # point of refusing the commit rather than the staging.
    dest = next(iter(sorted((root / "results" / "published").glob("*"))))
    assert "MISSING" in (dest / "GIT_SHA_CHECK.txt").read_text()


def test_a_real_publish_commits_pushes_and_leaves_no_tracked_file_modified(tmp_path):
    """The PASS branch of the same act, and R8's requirement on the publish path.

    The remote check used to run after the push and append to a file the commit
    above it already contained, so a publish that worked ended with a modified
    tracked file. Both halves are asserted: the tree is clean afterwards, and
    the reachability verdict is inside the commit rather than in a change made
    to it. PENDING is the verdict here because the row cites a commit that no
    remote had when the question was asked and that this publish's own push is
    what delivers.
    """
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    root, head = _real_repo(tmp_path, remote=str(bare))
    results = tmp_path / "results"
    results.mkdir()
    _write_run(results, "aa1", "base", [("True", "False", 1.0)], git_sha=head)
    r = _real_publish(root, results, "--label", "real-push")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "pushed" in r.stdout
    assert _status(root) == "", f"a successful publish dirtied the tree:\n{_status(root)}"
    name = next(iter(sorted((root / "results" / "published").glob("*")))).name
    committed = subprocess.run(
        ["git", "-C", str(root), "show",
         f"HEAD:results/published/{name}/GIT_SHA_CHECK.txt"],
        capture_output=True, text=True, check=True).stdout
    assert "reachability from a remote" in committed
    assert "RESULT: CLAIM shas_on_a_remote PASS" in committed
    assert f"PENDING    {head}" in committed
    summary = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:results/published/{name}/SUMMARY.md"],
        capture_output=True, text=True, check=True).stdout
    assert "shas_on_a_remote" in summary, "the verdict has to reach the file a reader opens"


def test_a_push_that_fails_says_the_pending_verdict_did_not_come_true(tmp_path):
    """PENDING is a promise about a push, so a push that fails has to retract it.

    Without this the operator reads "on no remote branch yet, but HEAD reaches
    it and this publish pushes that" in an arm whose push was rejected, which is
    the reassurance the old post-push check at least never gave.
    """
    root, head = _real_repo(tmp_path, remote=str(tmp_path / "nothing-here.git"))
    results = tmp_path / "results"
    results.mkdir()
    _write_run(results, "aa1", "base", [("True", "False", 1.0)], git_sha=head)
    r = _real_publish(root, results, "--label", "failed-push")
    assert "push failed" in r.stdout, r.stdout
    assert "calls PENDING reached the remote" in r.stdout
    # the commit is still local and complete, and the tree is still clean
    assert _commits(root) == 2
    assert _status(root) == ""


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


# --------------------------------------------------------------------------
# The `throttled` line in both generators, and the published files that carry it
# --------------------------------------------------------------------------
#
# `publish_results.sh` and `run_all.sh` printed "N rows throttled (clocks
# dropped >5% mid-cell)", and the first of those went into the SUMMARY.md of
# nine published arms. The rows carry the retired two-sample drift flag,
# which detected an idle-boost catch (moe/bench/timing.py); no detector ever
# measured a mid-cell drop. The generators now name the flag by the instrument
# that set it, and the published files carry a dated note beneath the line.

def _legacy_run(results: Path, run_id: str, flagged: int, total: int) -> Path:
    """A run in the shape every published arm has: schema 3, no verdict
    columns, `throttled` from the retired drift flag."""
    cols = ["schema_version", "run_id", "env_name", "gpu_name", "impl", "model",
            "num_tokens", "ms_p50", "correctness_passed", "l2_flush", "cuda_graph",
            "throttled", "clock_drift_pct"]
    path = results / f"run_{run_id}_base.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i in range(total):
            w.writerow(dict(schema_version=3, run_id=run_id, env_name="base",
                            gpu_name="NVIDIA H200", impl="torch_grouped_mm",
                            model="toy", num_tokens=32, ms_p50=1.0,
                            correctness_passed="True", l2_flush="True",
                            cuda_graph="False", throttled=str(i < flagged),
                            clock_drift_pct=7.0))
    return path


def _v5_run(results: Path, run_id: str, verdicts) -> Path:
    from moe.bench.schema import COLUMNS, SCHEMA_VERSION
    from moe.bench.timing import TIMING_BASIS
    path = results / f"run_{run_id}_base.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for level, drift in verdicts:
            row = dict.fromkeys(COLUMNS, "")
            row.update(schema_version=SCHEMA_VERSION, run_id=run_id, env_name="base",
                       gpu_name="NVIDIA H200", impl="torch_grouped_mm", model="toy",
                       num_tokens=32, ms_p50=1.0, correctness_passed="True",
                       l2_flush="True", cuda_graph="False", instrument=TIMING_BASIS,
                       clock_level_ok=level, clock_drift_ok=drift, host_bound_ok="ok",
                       throttled=str("failed" in (level, drift)))
            w.writerow(row)
    return path


def test_the_summary_names_the_retired_flag_on_pre_v5_rows(tmp_path):
    results, published = tmp_path / "results", tmp_path / "published"
    results.mkdir()
    _legacy_run(results, "aa1", flagged=2, total=5)
    r = _publish(results, published, "--label", "legacy")
    assert r.returncode == 0, r.stdout + r.stderr
    text = next(iter(sorted(published.glob("*/SUMMARY.md")))).read_text()
    assert "2 rows carry throttled=True from the retired pre-v5 drift flag" in text, text
    assert "idle-boost catch" in text
    assert "clocks dropped" not in text


def test_the_summary_names_level_and_drift_on_v5_rows(tmp_path):
    results, published = tmp_path / "results", tmp_path / "published"
    results.mkdir()
    _v5_run(results, "aa1", [("failed", "ok"), ("ok", "failed"), ("failed", "failed"),
                             ("ok", "ok")])
    r = _publish(results, published, "--label", "v5")
    assert r.returncode == 0, r.stdout + r.stderr
    text = next(iter(sorted(published.glob("*/SUMMARY.md")))).read_text()
    assert "3 rows carry throttled=True from the under-load clock check" in text, text
    assert "LEVEL failed on 2" in text and "DRIFT failed on 2" in text
    assert "retired pre-v5" not in text and "clocks dropped" not in text


def test_the_summary_says_when_the_instrument_cannot_be_read(tmp_path):
    """A v5-shaped row with an empty instrument is a file nothing sane wrote;
    the flag on it is reported as unattributable, not filed under either."""
    results, published = tmp_path / "results", tmp_path / "published"
    results.mkdir()
    from moe.bench.schema import COLUMNS, SCHEMA_VERSION
    with (results / "run_aa1_base.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        row = dict.fromkeys(COLUMNS, "")
        row.update(schema_version=SCHEMA_VERSION, run_id="aa1", env_name="base",
                   gpu_name="NVIDIA H200", impl="torch_grouped_mm", model="toy",
                   num_tokens=32, ms_p50=1.0, correctness_passed="True",
                   l2_flush="True", cuda_graph="False", throttled="True")
        w.writerow(row)
    r = _publish(results, published, "--label", "blank")
    assert r.returncode == 0, r.stdout + r.stderr
    text = next(iter(sorted(published.glob("*/SUMMARY.md")))).read_text()
    assert "1 rows carry throttled=True with an unreadable instrument column" in text, text


def test_run_all_summary_names_the_flag_the_same_way(tmp_path):
    """The second call site. `--summary-only` prints the end-of-run summary
    over existing rows, which is the only way to reach it without a sweep."""
    results = tmp_path / "results"
    results.mkdir()
    _legacy_run(results, "aa1", flagged=3, total=4)
    _v5_run(results, "bb2", [("failed", "ok")])
    r = sh(RUN_ALL, "--summary-only", str(results))
    assert r.returncode == 0, r.stdout + r.stderr
    assert ("CLOCK FLAG      3 rows carry throttled=True from the retired pre-v5 "
            "drift flag") in r.stdout, r.stdout
    assert ("CLOCK FLAG      1 rows carry throttled=True from the under-load clock "
            "check: LEVEL failed on 1") in r.stdout
    assert "clocks dropped" not in r.stdout
    assert "THROTTLED ROWS" not in r.stdout


def test_neither_generator_types_the_old_parenthetical():
    for script in (PUBLISH, RUN_ALL):
        for ln in script.read_text().splitlines():
            if "clocks dropped >5% mid-cell" in ln:
                assert ln.lstrip().startswith("#"), f"{script.name}: {ln}"


PUBLISHED = REPO / "results" / "published"


@pytest.mark.parametrize("summary", sorted(PUBLISHED.glob("*/SUMMARY.md")),
                         ids=lambda p: p.parent.name)
def test_every_published_summary_that_counts_the_flag_carries_the_dated_note(summary):
    """History is not rewritten: the line stays, its number stays, and the
    line beneath it says what the flag was and where the truth now lives."""
    lines = summary.read_text().splitlines()
    hits = [i for i, ln in enumerate(lines) if "rows throttled" in ln]
    if not hits:
        pytest.skip("this arm never printed the line")
    assert len(hits) == 1
    note = lines[hits[0] + 1]
    assert note.lstrip().startswith("- NOTE 2026-09-03:"), note
    assert "idle boost" in note and "moe/bench/timing.py" in note
    assert "clock_level_ok" in note and "00f3324" in note
    if "clocks dropped" in lines[hits[0]]:
        assert "withdrawn" in note


@pytest.mark.parametrize("surface", sorted(PUBLISHED.glob("*/SURFACE.txt")),
                         ids=lambda p: p.parent.name)
def test_every_published_surface_requalifies_the_candidate_line(surface):
    """ANCHOR_RESCORE.txt W3: the "0 of N fits within 0.05" count against
    0.558 is a property of an unidentified anchor. The generator still prints
    it, so the published file carries the requalification beneath the line."""
    lines = surface.read_text().splitlines()
    hits = [i for i, ln in enumerate(lines)
            if ln.startswith("  alpha = 0.558 (the 2026-09-01 pooled refit)")]
    assert len(hits) == 1, surface
    note = "\n".join(lines[hits[0] + 1:hits[0] + 9])
    assert note.startswith("  NOTE 2026-09-03: the line above is requalified"), note
    assert "ANCHOR_RESCORE" in note and "W4" in note
    assert "scripts/alpha_surface.py still prints" in note


def test_the_a100_surface_withdraws_the_direction_asserted_from_one_cell():
    surface = PUBLISHED / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3" / "SURFACE.txt"
    lines = surface.read_text().splitlines()
    i = lines.index("  paired change 1 -> 64: median +0.028   (every cell moves the same way: yes)")
    assert lines[i - 7].startswith("  MDE: only 1 matched cell(s)"), "the n=1 refusal moved"
    note = "\n".join(lines[i + 1:i + 8])
    assert note.startswith("  NOTE 2026-09-03: the line above is withdrawn as a direction"), note
    assert "n=1" in note and "retraction (d)" in note


def test_the_surface_generator_still_prints_a_direction_at_n_equals_1():
    """The defect the notes exist for, pinned so that fixing the generator
    turns this into a signal to regenerate the three files and drop the notes.
    Run against the published A100 arm, whose G lever has one matched cell."""
    arm = PUBLISHED / "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3"
    r = subprocess.run([PY, str(REPO / "scripts" / "alpha_surface.py"), str(arm)],
                       cwd=REPO, text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    i = next(k for k, ln in enumerate(out) if ln.startswith("  MDE: only 1 matched cell(s)"))
    block = "\n".join(out[i:i + 9])
    assert "paired change 1 -> 64: median +0.028   (every cell moves the same way: yes)" in block, (
        "scripts/alpha_surface.py now refuses a direction at n=1: regenerate the "
        "three published SURFACE.txt files and remove their 2026-09-03 notes")
    assert "alpha = 0.558 (the 2026-09-01 pooled refit): 0 of 12 fits within 0.05" in r.stdout
