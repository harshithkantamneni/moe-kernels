"""The unattended driver has to be checkable without a pod.

`scripts/h200_gaps_session.sh` spends about 102 minutes of rented H200 across
eight arms. Almost everything that can go wrong with it goes wrong SILENTLY and
is only visible an hour later: an arm marked finished having measured nothing, a
pipeline that recorded `tee`'s exit status, a flag the sibling script renamed, a
summary that prints nothing because a gate line moved. None of those raise.

So this file checks the four things that can be checked off GPU:

  1. THE SHELL ITSELF -- syntax, and the three habits this project has already
     been burned by (`set -e` aborting a long run, a pipeline masking an exit
     code, a PID variable in a shell that started no background job).
  2. THE ARM TABLE IS COMPLETE AND ORDERED. Every arm needs an estimate, a
     finished-code set, a gate regex and a statement of what it closes, and the
     arms whose results change how a later arm is READ have to come first.
  3. THE FLAGS STILL EXIST. A driver that passes `--densify` to a script that
     dropped it fails after the pod is already running.
  4. THE GATE REGEXES MATCH THE REAL OUTPUT. A summary that silently matches
     nothing reports no failures, which is this project's non-vacuity rule
     applied to its own reporting.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "h200_gaps_session.sh"
TEXT = DRIVER.read_text()
#: The same file with whole-line comments dropped. The structural checks below
#: are about what the shell RUNS, and this file's own comments name the very
#: constructs they forbid ("`cmd | tee log` would record tee's exit status",
#: "no `$!`"), so a check over the raw text matches its own documentation and
#: fails on a correct script.
CODE = "\n".join(ln for ln in TEXT.splitlines() if not ln.lstrip().startswith("#"))

# The seventeen arms, in the order their results are READ. Rewritten 2026-09-02
# after two parallel workflows each wrote a driver and the second one's arms --
# the four highest-priority items from the adversarial evaluation -- were left
# on disk with nothing scheduling them.
ARMS = ("calibrate", "pin_probe",
        "roofline", "bm128_depth", "noise_floor",
        "bn_g16", "bn_g1", "anchor_measure", "anchor_rescore", "occupancy",
        "mma_switch", "ruler", "cap_test", "dtype", "span", "span_dense",
        "counter_plan")


def run(args, cwd=None, session=None):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home())}
    if session:
        env["SESSION"] = str(session)
    return subprocess.run(["bash", str(DRIVER), *args], cwd=str(cwd or ROOT),
                          capture_output=True, text=True, timeout=600, env=env)


# --------------------------------------------------------------------------
# 1. the shell itself
# --------------------------------------------------------------------------

def test_the_driver_parses():
    done = subprocess.run(["bash", "-n", str(DRIVER)], capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_it_does_not_set_e_because_one_arm_must_not_end_the_run():
    """A 102-minute unattended run that aborts at minute six on a shape that
    does not compile is worse than no automation. Both reference drivers say so
    on the same line."""
    assert "set -uo pipefail" in TEXT
    assert not re.search(r"^set -[a-z]*e", TEXT, re.M)


def test_no_measured_command_is_run_through_a_pipeline():
    """FAILURE MODE 7. `cmd | tee log` reports the exit status of `tee`, and the
    exit code is the only thing that decides DONE from RETRY in the ledger, so
    every arm would be recorded as finished."""
    assert '"$@" > "$log" 2>&1' in CODE
    body = CODE.split("arm() {", 1)[1].split("\n}", 1)[0]
    assert "tee" not in body
    assert "|" not in body.replace("||", "")


def test_it_starts_no_background_job_and_reads_no_pid():
    """The other half of failure mode 7: a PID variable read in a shell that
    started nothing. There is no `$!` here because there is nothing to wait
    for."""
    assert "$!" not in CODE
    assert not re.search(r"[^&\s]\s*&\s*$", CODE, re.M)


# --------------------------------------------------------------------------
# 2. the arm table
# --------------------------------------------------------------------------

def test_every_arm_declares_a_cost_a_finished_code_a_regex_and_what_it_closes():
    listing = run(["--list"])
    assert listing.returncode == 0, listing.stderr
    for name in ARMS:
        assert re.search(rf"^  {name}\s+~\d+ min\s+finished on exit [\d,]+$",
                         listing.stdout, re.M), name
    # ...and a gate regex, which --list does not print.
    for name in ARMS:
        assert f"{name})" in TEXT or f"|{name})" in TEXT or f"{name}|" in TEXT


def test_every_arm_says_what_it_closes_and_what_it_leaves_open():
    """A summary that only reported PASS/FAIL would leave the reader to map
    eight arms onto the study's open items from memory."""
    listing = run(["--list"]).stdout
    for name in ARMS:
        block = listing.split(f"  {name} ", 1)[1].split("\n\n", 1)[0]
        assert len(block) > 100, name
    # The three arms that carry the claim say so in words a reader can find.
    assert "THE HEADLINE" in listing
    assert "noise floor" in listing.lower() or "replicate" in listing.lower()
    assert "DEMOTED" in listing, "the BLOCK_M=16 arm must say it no longer carries the claim"


def test_the_arms_whose_result_changes_a_later_reading_come_first():
    """ORDER IS THE ARGUMENT. The calibration sets the ridge every roof fraction
    below is scored against -- and `tile_cap_test.py` REFUSES without one for
    the attached device. The pin probe decides whether arm 4's forced census can
    be attributed to a tile at all. Both are cheap; both must precede what they
    gate."""
    order = re.search(r"^ARM_NAMES=\(([^)]*)\)", TEXT, re.M).group(1).split()
    assert order == list(ARMS)
    # Calibration and the pin probe gate everything: five arms refuse without
    # the first, and every forced-tile arm is meaningless without the second.
    assert order[0] == "calibrate" and order[1] == "pin_probe"
    # THE CLAIM COMES BEFORE ITS SUPPORTS. If BLOCK_M=128 reaches the roof,
    # depth, noise floor, decomposition and anchor are measuring a ceiling that
    # never binds -- and that is worth knowing at minute ten, not minute ninety.
    assert order.index("roofline") < order.index("bm128_depth")
    assert order.index("bm128_depth") < order.index("noise_floor")
    assert order.index("noise_floor") < order.index("bn_g16")
    # A measurement precedes the free re-scoring that reads it.
    assert order.index("anchor_measure") < order.index("anchor_rescore")
    # The docs' own backlog comes after the claim, and the demoted BLOCK_M=16
    # formula test comes after the BLOCK_M=128 production measurement.
    assert order.index("roofline") < order.index("cap_test")
    assert order.index("pin_probe") < order.index("mma_switch")
    # The free counter probe is last: its result changes how no arm above is
    # read, only what to type on a box where the route is open.
    assert order[-1] == "counter_plan"


def test_the_estimated_total_is_printed_before_anything_is_spent(tmp_path):
    got = run(["--dry-run"], session=tmp_path / "s")
    assert "WHAT THIS COMMITS YOU TO" in got.stdout
    body = got.stdout.split("WHAT THIS COMMITS YOU TO")[1]
    assert body.index("SESSION  card=") > 0
    for name in ARMS:
        assert re.search(rf"  {name}\s+~\s*\d+ min", body), name


# --------------------------------------------------------------------------
# 3. the flags still exist on the far side
# --------------------------------------------------------------------------

INVOKED = {
    "scripts/calibrate_hardware.py": (),
    "scripts/ruler_rebaseline.py": ("--dry-run",),
    "scripts/check_mma_path.sh": ("--block-m", "--tokens", "--model", "--out",
                                  "--dry-run"),
    "scripts/tile_cap_test.py": ("--dry-run", "--capability"),
    "scripts/dtype_tile_confound.py": ("--dry-run",),
    "scripts/span_extent_separation.py": ("--dry-run", "--densify",
                                          "--max-minutes"),
}


@pytest.mark.parametrize("script,flags", sorted(INVOKED.items()))
def test_every_script_the_driver_invokes_exists_and_takes_the_flags_it_is_given(
        script, flags):
    """A renamed flag fails an hour into a rented pod, in the one arm nobody is
    watching. Checked here for a few seconds instead."""
    path = ROOT / script
    assert path.exists(), script
    assert script in TEXT, f"{script} is not actually invoked any more"
    haystack = path.read_text()
    for flag in flags:
        assert flag in haystack, f"{script} no longer mentions {flag}"


def test_the_pin_probe_passes_the_two_flags_without_which_it_measures_nothing():
    """THE FIX THE REVIEW FOUND MISSING FROM pod_session.sh:1703.

    `profiles.candidate_impls` filters on `span.env` and `VllmFusedExperts`
    declares `env="vllm"`, so without `--env vllm` the probe plans only torch's
    CUTLASS spans, whose `tile_block_m` is 0. The gate then reads
    "observed = none" however good the hook is -- which is exactly the state the
    2026-09-01 session shipped in.
    """
    block = TEXT.split("PIN=", 1)[1].split("# ---", 1)[0]
    assert "--env vllm" in block
    assert "--impl vllm_fused_experts" in block


def test_the_pin_is_never_exported_over_the_other_arms():
    """Exported, MOE_FORCE_TILE would pin arms 5 to 7 silently and every one of
    them would be measuring a tile nobody asked it for."""
    assert "export MOE_FORCE_TILE" not in TEXT
    assert "env MOE_FORCE_TILE=" in TEXT


# --------------------------------------------------------------------------
# 4. the summary can actually find the gates
# --------------------------------------------------------------------------

REAL_GATE_LINES = {
    # verbatim shapes the scripts print; a regex that stops matching these
    # makes the closing summary silently empty.
    "cap_test": "V0  VALIDITY PASS      override_config changed the kernel",
    "mma_switch": "G1  VALIDITY PASS every arm ran the tile it was given",
    "dtype": "[PASS   ] CLAIM    C1 confound exists  the two dtypes resolve",
    "span": "[PASS] V0 assembly  the five launches recompute",
    "span_dense": "[FAIL] C2 kernel  the Triton-versus-CUTLASS kernel difference",
    "ruler": "VALIDITY 3  PASS      no output lands where .gitignore drops it",
    "pin_probe": "[force-tile] GATE F1  every row produced under the pin shows",
    "calibrate": "ridge       162.80 Op/B",
}


@pytest.mark.parametrize("arm,line", sorted(REAL_GATE_LINES.items()))
def test_each_arms_gate_regex_matches_the_shape_that_arm_really_prints(arm, line):
    # The function is lifted out and evaluated on its own: sourcing the driver
    # would run its `--list` branch, which exits, and would take the test with
    # it. This asks the SHIPPED function rather than a Python copy of the case
    # statement, which would agree with it until it did not.
    got = subprocess.run(
        ["bash", "-c",
         f'eval "$(sed -n \'/^arm_gate_regex()/,/^esac; }}/p\' "{DRIVER}")"; '
         f'arm_gate_regex {arm}'],
        capture_output=True, text=True, timeout=60)
    pattern = got.stdout.strip()
    assert pattern, f"no regex declared for {arm}"
    matched = subprocess.run(["grep", "-Eq", pattern], input=line, text=True)
    assert matched.returncode == 0, (
        f"{arm}'s summary regex {pattern!r} does not match the line its script "
        f"prints: {line!r}. The closing summary would report NO GATE LINE for "
        f"every {arm} run on the pod.")


# --------------------------------------------------------------------------
# 5. the two states a ledger must never confuse
# --------------------------------------------------------------------------

def test_a_dry_run_writes_a_separate_ledger_and_marks_nothing_done(tmp_path):
    """A PLAN IS NOT A MEASUREMENT, and the resume path reads the ledger.

    A dry run of an arm still produces an exit code. Written into the measuring
    ledger it would mark arms DONE having measured nothing, and the pod run
    would skip them -- ending with a confident summary of eight arms and no
    data.
    """
    session = tmp_path / "s"
    got = run(["--dry-run"], session=session)
    assert got.returncode == 0, got.stderr
    assert (session / "ARMS-dryrun.tsv").exists()
    assert not (session / "ARMS.tsv").exists()
    rows = (session / "ARMS-dryrun.tsv").read_text().splitlines()[1:]
    assert rows
    for row in rows:
        assert row.split("\t")[1] == "PLANNED", row


def test_a_measuring_run_refuses_without_a_card_and_writes_nothing(tmp_path):
    """Every verdict here is scored against a per-card calibrated ridge -- 145.7
    on the A100 against 162.8 on the H200 -- so there is no card-free measuring
    mode and none is offered."""
    session = tmp_path / "s"
    got = run([], session=session)
    assert got.returncode == 3
    assert "REFUSED: no CUDA device" in got.stdout
    assert not session.exists()


def test_the_resume_check_does_not_depend_on_a_gnu_only_grep():
    """`grep -qP` is a GNU extension. On a laptop it fails, the resume check
    silently says "not done" for every arm, and a re-run repeats work -- or
    worse, the same construct is trusted on the pod after being tested nowhere.
    Both reference drivers use it; this one uses awk."""
    body = TEXT.split("arm() {", 1)[1].split("\n}", 1)[0]
    assert "grep -qP" not in body
    assert "awk -F" in body


# --------------------------------------------------------------------------
# 6. git is asked, not remembered
# --------------------------------------------------------------------------

def test_it_asks_git_check_ignore_rather_than_asserting_the_rule():
    """FAILURE MODE 3. `.gitignore` ignores `results/*` and re-includes only
    `results/published/`, so the answer differs for three kinds of path and a
    remembered sentence is right for one of them. This repo has already lost
    every published plot of ten arms that way."""
    assert "git check-ignore" in TEXT
    assert "check-ignore -q" in TEXT
    # ...and the third answer exists: rc 128 is what git returns for a path
    # outside the work tree, which is the pod default /workspace/results.
    assert "UNVERIFIED" in TEXT
    assert "It is NOT 'tracked'" in TEXT


def test_the_git_verdict_is_asked_of_a_path_an_arm_really_writes(tmp_path):
    """`results/` itself is not ignored -- the pattern is `results/*` -- so
    asking about the ROOT answers the wrong question and reports "tracked" for a
    tree whose every child git drops."""
    got = run(["--dry-run"], session=tmp_path / "s")
    assert "/bm128_roofline" in got.stdout, (
        "the git verdict must be asked of a path an arm really writes, and the "
        "headline arm's directory is the one that matters most")


def test_it_never_commits_terminates_or_pushes(tmp_path):
    # These may appear only as printed ADVICE to the operator, inside the
    # closing heredoc or a comment, never as a command this script runs.
    runnable = [ln for ln in CODE.splitlines()
                if ln[:1] not in ("", " ") and not ln.startswith("cat <<")]
    for forbidden in ("git commit", "git push", "git add ", "runpodctl",
                      "shutdown", "poweroff", "rm -rf"):
        for line in runnable:
            assert forbidden not in line, line
    got = run(["--dry-run"], session=tmp_path / "s")
    assert "NOTHING HERE IS COMMITTED AND NOTHING IS PUSHED" in got.stdout


# --------------------------------------------------------------------------
# 7. non-vacuity in the driver's own reporting
# --------------------------------------------------------------------------

def test_an_arm_with_no_gate_line_is_reported_rather_than_omitted(tmp_path):
    """A CHECK THAT EXAMINED NOTHING REPORTS NO FAILURES. An arm whose log
    carries no gate line has not passed its gates; printing nothing for it would
    read as having found nothing wrong."""
    got = run(["--dry-run"], session=tmp_path / "s")
    assert "PLAN ONLY: --dry-run scores no gate for this arm." in got.stdout
    assert "NO GATE LINE MATCHED" in TEXT      # the measuring-mode wording
    assert "This arm was NOT scored" in TEXT
