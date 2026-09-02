"""The unattended driver has to be checkable without a pod.

`scripts/h200_gaps_session.sh` spends about 3.4 hours of rented GPU across
twenty arms. Almost everything that can go wrong with it goes wrong SILENTLY and
is only visible an hour later: an arm marked finished having measured nothing, a
pipeline that recorded `tee`'s exit status, a flag the sibling script renamed, a
summary that prints an imported constant out of a refused log as if this session
had measured it. None of those raise.

So this file checks what can be checked off GPU:

  1. THE SHELL ITSELF -- syntax, and the three habits this project has already
     been burned by (`set -e` aborting a long run, a pipeline masking an exit
     code, a PID variable in a shell that started no background job).
  2. THE ARM TABLE IS COMPLETE AND ORDERED. Every arm needs an estimate, a
     statement of what it closes, and the arms whose results change how a later
     arm is READ have to come first.
  3. THE FLAGS STILL EXIST. A driver that passes `--densify` to a script that
     dropped it fails after the pod is already running.
  4. THE STATES. The driver adopts `moe/bench/exit_codes.py`, and the shell
     mirror of that table is compared with the module code for code. The
     planting states (PLANNED / PLAN_REFUSED / BROKEN) are exercised with stub
     commands, because a `--dry-run` that tracebacks used to be recorded as a
     clean plan.
  5. THE SUMMARY GREPS ONLY `^RESULT: `. The previous summary grepped
     `floor|sigma` and printed an imported prior and a pre-registered
     expectation out of a REFUSED log under the heading "THE GATES". The two
     tests that pin this feed it exactly such a log.

HOW THE SHELL FUNCTIONS ARE TESTED. Sourcing the driver would run the session,
so the file marks a block of pure function definitions between
`# >>> LIFTABLE` and `# <<< LIFTABLE`, and `lift()` below evaluates that block
in a fresh bash. That asks the SHIPPED function rather than a Python copy of
its case statement, which would agree with it until it did not.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
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

sys.path.insert(0, str(ROOT))
from moe.bench import exit_codes  # noqa: E402

# The twenty arms, in the order their results are READ. Rewritten 2026-09-02
# after the standards audit: the headline roofline was running BLOCK_N=64 with
# GROUP_SIZE_M=1, a configuration vLLM never ships for a multi-tile BLOCK_M=128,
# so it could refute the ceiling and never confirm it. The configuration is now
# IN THE ARM NAME wherever two arms differ only by configuration.
ARMS = ("calibrate", "pin_probe-n64-g1", "pin_probe-n256-g16",
        "roofline-n64-g1", "roofline-n256-g16", "roofline-n256-g32",
        "bm128_depth", "noise_floor",
        "bn_g16", "bn_g1", "anchor_measure", "anchor_rescore", "occupancy",
        "mma_switch", "ruler", "cap_test", "dtype", "span_dense", "span",
        "counter_plan")


def run(args, cwd=None, session=None, env_extra=None):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home())}
    if session:
        env["SESSION"] = str(session)
    env.update(env_extra or {})
    return subprocess.run(["bash", str(DRIVER), *args], cwd=str(cwd or ROOT),
                          capture_output=True, text=True, timeout=900, env=env)


def lift(script: str, **variables):
    """Evaluate the driver's LIFTABLE function block, then `script`.

    Every global those functions read is passed explicitly, so a test that
    forgets one gets an unbound-variable error rather than a value the session
    happened to leave lying around.
    """
    setup = "\n".join(f'{k}={v!r}' for k, v in variables.items())
    body = (f"set -uo pipefail\n{setup}\n"
            f'eval "$(sed -n \'/^# >>> LIFTABLE/,/^# <<< LIFTABLE/p\' "{DRIVER}")"\n'
            f"{script}\n")
    return subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                          timeout=120)


# --------------------------------------------------------------------------
# 1. the shell itself
# --------------------------------------------------------------------------

def test_the_driver_parses():
    done = subprocess.run(["bash", "-n", str(DRIVER)], capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_it_does_not_set_e_because_one_arm_must_not_end_the_run():
    """A 3.4-hour unattended run that aborts at minute six on a shape that does
    not compile is worse than no automation."""
    assert "set -uo pipefail" in TEXT
    assert not re.search(r"^set -[a-z]*e", TEXT, re.M)


def test_no_measured_command_is_run_through_a_pipeline():
    """FAILURE MODE 7. `cmd | tee log` reports the exit status of `tee`, and the
    exit code is the only thing that decides one ledger state from another, so
    every arm would be recorded as finished."""
    assert '"$@" > "$log" 2>&1' in CODE
    body = CODE.split("\narm() {", 1)[1].split("\n}", 1)[0]
    assert "tee" not in body


def test_it_starts_no_background_job_and_reads_no_pid():
    """The other half of failure mode 7: a PID variable read in a shell that
    started nothing. There is no `$!` here because there is nothing to wait
    for."""
    assert "$!" not in CODE
    assert not re.search(r"[^&\s]\s*&\s*$", CODE, re.M)


# --------------------------------------------------------------------------
# 2. the arm table
# --------------------------------------------------------------------------

def test_every_arm_declares_a_cost_and_what_it_closes():
    listing = run(["--list"])
    assert listing.returncode == 0, listing.stderr
    for name in ARMS:
        assert re.search(rf"^  {re.escape(name)}\s+~\d+ min$", listing.stdout, re.M), name


def test_the_list_prints_the_one_exit_code_table_instead_of_per_arm_code_lists():
    """WHAT THE AUDIT FOUND. The driver kept a hand-written "finished on exit
    0,1" list per arm, in a tree where `memory_branch_anchor.py` used 2 for a
    VALIDITY FAIL after measuring and three other scripts refused with 3. One
    table replaces twelve lists, and `--list` prints the table."""
    listing = run(["--list"]).stdout
    assert "finished on exit" not in listing
    assert "arm_done_codes" not in TEXT
    for word in ("DONE", "CLAIM_FAIL", "REFUSED", "INVALID", "RETRY"):
        assert word in listing, word
    assert "exit_codes.py" in listing


def test_every_arm_says_what_it_closes_and_what_it_leaves_open():
    listing = run(["--list"]).stdout
    for name in ARMS:
        block = listing.split(f"  {name} ", 1)[1].split("\n\n", 1)[0]
        assert len(block) > 100, name
    assert "THE CLAIM, and the only arm that can confirm it" in listing
    assert "THE CONTROL" in listing
    assert "DEMOTED" in listing, "the BLOCK_M=16 arm must say it no longer carries the claim"


def test_the_arms_whose_result_changes_a_later_reading_come_first():
    """ORDER IS THE ARGUMENT. The calibration sets the ridge every roof fraction
    below is scored against. The pin probes decide whether a forced census can
    be attributed to a tile at all, and each probes the configuration the arms
    it gates actually run."""
    order = re.search(r"^ARM_NAMES=\(([^)]*)\)", TEXT, re.M).group(1).split()
    assert order == list(ARMS)
    assert order[0] == "calibrate"
    assert order[1].startswith("pin_probe") and order[2].startswith("pin_probe")
    # The control roofline runs first of the three: if BLOCK_M=128 reaches the
    # roof at the LEANEST configuration it reaches it at every richer one, so a
    # refutation there ends the session's whole middle at minute ten.
    assert order.index("roofline-n64-g1") < order.index("roofline-n256-g16")
    assert order.index("roofline-n256-g16") < order.index("roofline-n256-g32")
    assert order.index("roofline-n256-g32") < order.index("bm128_depth")
    assert order.index("bm128_depth") < order.index("noise_floor")
    assert order.index("noise_floor") < order.index("bn_g16")
    # A measurement precedes the free re-scoring that reads it.
    assert order.index("anchor_measure") < order.index("anchor_rescore")
    assert order.index("roofline-n64-g1") < order.index("cap_test")
    assert order[-1] == "counter_plan"


def test_the_dense_span_grid_runs_before_the_sparse_one():
    """AUDIT A11. `span_extent_separation --self-test kernel` on the SPARSE grid
    the driver schedules returns C2 FAIL, C5 FAIL and C3 UNKNOWN; with
    `--densify` it is 12/12 PASS. The dense grid is the only one on which C3's
    mechanism is observable, so a 45-minute sparse arm that runs first spends
    the budget on the grid that cannot answer."""
    order = re.search(r"^ARM_NAMES=\(([^)]*)\)", TEXT, re.M).group(1).split()
    assert order.index("span_dense") < order.index("span")


def test_a_claim_gate_failing_is_finished_and_never_re_run():
    """AUDIT A11, the other half: the driver used to accept only exit 0 for the
    span arms, so the outcome the kernel world PREDICTS landed in the ledger as
    RETRY and the arm was queued to spend its 45 minutes again. Exit 1 is
    CLAIM_FAIL for every arm alike now, and the resume check treats it as
    finished."""
    body = CODE.split("\narm() {", 1)[1].split("\n}", 1)[0]
    assert '$2 == "CLAIM_FAIL"' in body
    assert '$2 == "INVALID"' in body
    got = lift('ledger_state 1', REPO=str(ROOT))
    assert got.stdout.strip() == "CLAIM_FAIL"


def test_the_estimated_total_is_printed_before_anything_is_spent(tmp_path):
    got = run(["--dry-run"], session=tmp_path / "s")
    assert "WHAT THIS COMMITS YOU TO" in got.stdout
    body = got.stdout.split("WHAT THIS COMMITS YOU TO")[1]
    assert body.index("SESSION  card=") > 0
    for name in ARMS:
        assert re.search(rf"  {re.escape(name)}\s+~\s*\d+ min", body), name


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
    "scripts/bm128_roofline.py": ("--block-n", "--group-m", "--control",
                                  "--dry-run"),
    "scripts/memory_branch_anchor.py": ("--rescore", "--out-dir", "--dry-run"),
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


def test_the_production_roofline_arm_runs_the_configuration_vllm_ships():
    """AUDIT A8, THE FINDING THAT DECIDES THE PAPER. The headline arm ran
    `SWEEP.FIXED` -- BLOCK_N=64, GROUP_SIZE_M=1 -- under a heading about what
    production runs. vLLM 0.27.1's tuned entry for mixtral at BLOCK_M=128 is in
    this repo (moe/bench/hardware/vllm_configs/E=8,N=14336,device_name=
    NVIDIA_H200.json): BLOCK_N=256 with GROUP_SIZE_M=16 from 512 tokens up and
    32 at 2048. An arm at BN=64/G=1 can refute a ceiling and cannot confirm one,
    and the driver now says which arm is which."""
    plan = run(["--dry-run", "--only",
                "roofline-n64-g1,roofline-n256-g16,roofline-n256-g32"]).stdout
    assert len(re.findall(r"roofline.*-n256-g16", plan)) >= 1
    assert len(re.findall(r"roofline.*-n256-g32", plan)) >= 1
    for flags in ("--block-n 256 --group-m 16 --control 256",
                  "--block-n 256 --group-m 32 --control 256",
                  "--block-n 64 --group-m 1 --control 256"):
        assert flags in CODE, flags
    # The control arm is named a control in words, not only by its suffix.
    assert "can REFUTE the ceiling" in TEXT
    assert "CANNOT confirm one for production" in TEXT


def test_the_pin_is_probed_at_the_configurations_the_arms_run():
    """AUDIT A17. The probe pinned BLOCK_SIZE_N=128, which no arm runs: the
    control roofline, both bn arms, the anchor and the cap test pin 64, and the
    production rooflines pin 256. A pin honoured at one BLOCK_N is evidence
    about that BLOCK_N."""
    assert '"BLOCK_SIZE_N":128' not in CODE
    assert '"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":64' in CODE
    assert '"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":256' in CODE
    assert '"GROUP_SIZE_M":16' in CODE
    # ...and the two probes together still cost what the one probe cost.
    minutes = subprocess.run(
        ["bash", "-c",
         f'eval "$(sed -n \'/^arm_minutes()/,/^esac; }}/p\' "{DRIVER}")"; '
         f'arm_minutes pin_probe-n64-g1; arm_minutes pin_probe-n256-g16'],
        capture_output=True, text=True, timeout=60)
    assert sum(int(x) for x in minutes.stdout.split()) == 4


def test_the_pin_probe_passes_the_two_flags_without_which_it_measures_nothing():
    """THE FIX THE REVIEW FOUND MISSING FROM pod_session.sh:1703.

    `profiles.candidate_impls` filters on `span.env` and `VllmFusedExperts`
    declares `env="vllm"`, so without `--env vllm` the probe plans only torch's
    CUTLASS spans, whose `tile_block_m` is 0. The gate then reads
    "observed = none" however good the hook is -- which is exactly the state the
    2026-09-01 session shipped in.
    """
    block = TEXT.split("PIN_SWEPT=", 1)[1].split("# ---", 1)[0]
    assert block.count("--env vllm") >= 2
    assert block.count("--impl vllm_fused_experts") >= 2


def test_the_pin_is_never_exported_over_the_other_arms():
    """Exported, MOE_FORCE_TILE would pin every later arm silently and each one
    would be measuring a tile nobody asked it for."""
    assert "export MOE_FORCE_TILE" not in TEXT
    assert "env MOE_FORCE_TILE=" in TEXT


def test_the_anchor_rescore_is_kept_out_of_the_tracked_tree():
    """AUDIT A6. `--rescore` defaults its output to results/published/ and
    rewrote two TRACKED files, ANCHOR_RESCORE.txt and .json, on every
    invocation -- `--dry-run` included, since the driver ran it in both modes.
    P1 then passed at preflight and 3,696 published rows carry git_dirty=True."""
    assert '--rescore \\\n      --out-dir "$SESSION/anchor-rescore"' in TEXT
    plan = run(["--dry-run", "--only", "anchor_rescore"]).stdout
    assert "NOT PLANNED anchor_rescore" in plan
    assert "scores the committed corpus" in plan


# --------------------------------------------------------------------------
# 4. the states, and the table they all come from
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rc", [0, 1, 2, 3, 4, 5, 6, 127, 130])
def test_the_shell_ledger_state_agrees_with_the_module_code_for_code(rc):
    """THE DEFECT THIS PINS. Until 2026-09-02 the driver read 2 as REFUSED while
    `memory_branch_anchor.py` exited 2 for a VALIDITY gate that failed AFTER an
    eight-minute measurement, so a measured, unquotable run was logged as
    "REFUSED BEFORE MEASURING" and, because the anchor's stream check only runs
    on a freshly timed cell, every resume hit M0 FAIL and the arm could never
    reach DONE. Two integers meant two things in two files. This test is the
    only thing that keeps the shell copy honest."""
    got = lift(f'ledger_state {rc}', REPO=str(ROOT))
    assert got.stdout.strip() == exit_codes.ledger_state(rc), got.stderr


def test_the_driver_names_every_state_the_module_names():
    for state in exit_codes.LEDGER_STATES:
        assert state in TEXT, state


def test_a_dry_run_records_the_exit_code_and_a_tracebacking_plan_is_broken(tmp_path):
    """AUDIT A3. `:411-412` mapped EVERY dry-run exit code to PLANNED, so a
    `--dry-run` that ended in a traceback was written into the ledger as a clean
    plan and the operator flew to the pod with it. A plan gets three words now,
    and the one that means "this is not a plan" is planted here with a stub that
    tracebacks the way a python script does."""
    ledger = tmp_path / "ARMS-dryrun.tsv"
    logs = tmp_path / "logs"
    logs.mkdir()
    stub = tmp_path / "stub.py"
    stub.write_text("raise RuntimeError('the plan did not survive its own dry run')\n")
    got = lift(f'arm stub {sys.executable} {stub}',
               REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               DRY=1, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert got.returncode == 0, got.stderr
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert row[0] == "stub"
    assert row[1] == "BROKEN"
    assert row[2] == "1"
    assert "Traceback" in (logs / "stub.log").read_text()
    assert "BROKEN: this is a PLAN" in got.stdout


@pytest.mark.parametrize("rc,state", [(0, "PLANNED"), (2, "PLAN_REFUSED"),
                                      (1, "BROKEN"), (3, "BROKEN"),
                                      (4, "BROKEN"), (127, "BROKEN")])
def test_a_plan_gets_planning_words_and_a_refusal_is_not_a_breakage(rc, state):
    """A PLAN IS NOT A MEASUREMENT, so it does not get measuring words. rc 2 is
    the one non-zero code that is not a breakage: off a GPU box several of these
    plans are SUPPOSED to refuse, and a refusal costs nothing. Everything else
    -- a claim verdict, an INVALID, a traceback, a missing interpreter -- is a
    plan that did not print."""
    got = lift(f'dry_state {rc}', REPO=str(ROOT))
    assert got.stdout.strip() == state


def test_a_dry_run_writes_a_separate_ledger_and_marks_nothing_done(tmp_path):
    """A PLAN IS NOT A MEASUREMENT, and the resume path reads the ledger. Written
    into the measuring ledger a plan's exit code would mark arms finished having
    measured nothing, and the pod run would skip them, ending with a confident
    summary of twenty arms and no data."""
    session = tmp_path / "s"
    got = run(["--dry-run"], session=session)
    assert (session / "ARMS-dryrun.tsv").exists()
    assert not (session / "ARMS.tsv").exists()
    rows = [r.split("\t") for r in
            (session / "ARMS-dryrun.tsv").read_text().splitlines()[1:]]
    assert rows
    for row in rows:
        assert row[1] in ("PLANNED", "PLAN_REFUSED", "BROKEN", "NOT_PLANNED"), row
        assert row[1] not in exit_codes.LEDGER_STATES, row
    # A session with a broken plan exits INVALID and NAMES the arms; one with
    # none exits 0. Which of the two happens here depends on sibling scripts
    # whose refusals still exit 3, so both branches are checked for coherence
    # rather than one of them asserted.
    broken = [r[0] for r in rows if r[1] == "BROKEN"]
    if broken:
        assert got.returncode == exit_codes.INVALID
        assert f"{len(broken)} PLAN(S) BROKEN" in got.stdout
        for name in broken:
            assert re.search(rf"^  {re.escape(name)}\s+exited \d+ under --dry-run",
                             got.stdout, re.M), name
        assert "a refusal wearing an" in got.stdout
    else:
        assert got.returncode == 0, got.stdout[-2000:]


def test_the_session_exits_non_zero_when_a_plan_is_broken(tmp_path):
    """The other half of A3: recording BROKEN and then exiting 0 would leave the
    word in a file nobody reads. `--dry-run` is a gate, so it has a FAIL branch,
    and the branch is planted by pointing an arm at a stub that tracebacks."""
    ledger = tmp_path / "ARMS-dryrun.tsv"
    logs = tmp_path / "logs"
    logs.mkdir()
    stub = tmp_path / "stub.py"
    stub.write_text("raise RuntimeError('boom')\n")
    counted = lift(
        f'arm stub {sys.executable} {stub}\n'
        'if (( DRY )) && (( BROKEN_ARMS > 0 )); then exit 3; fi\nexit 0',
        REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
        DRY=1, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert counted.returncode == 3, counted.stdout
    # ...and the shipped script contains that same branch, not a variant.
    assert "if (( DRY )) && (( BROKEN_ARMS > 0 )); then" in CODE
    # The PASS branch: a plan that works exits 0 and counts no BROKEN arm.
    stub.write_text("print('a plan')\n")
    clean = lift(
        f'arm stub {sys.executable} {stub}\n'
        'if (( DRY )) && (( BROKEN_ARMS > 0 )); then exit 3; fi\nexit 0',
        REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
        DRY=1, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert clean.returncode == 0, clean.stdout


def test_an_invalid_arm_is_recorded_in_its_own_state_and_not_retried(tmp_path):
    """INVALID IS NOT REFUSED AND IS NOT A RETRY. It cost the whole arm, its
    cells are on disk and must not be scored, and repeating it repeats the
    failure unless the log says the cause was transient. The old driver had no
    word for it at all: exit 3 fell into "anything else", which it called
    RETRY."""
    ledger = tmp_path / "ARMS.tsv"
    logs = tmp_path / "logs"
    logs.mkdir()
    first = lift('arm stub bash -c "exit 3"',
                 REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                 DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert first.returncode == 0, first.stderr
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert row[1] == "INVALID"
    assert "Do NOT re-run and do NOT quote it" in first.stdout
    # A second pass over the same ledger must not spend the arm again.
    again = lift('arm stub bash -c "exit 3"',
                 REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                 DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert "SKIP stub (already INVALID" in again.stdout
    assert len(ledger.read_text().splitlines()) == 1


def test_a_refused_arm_is_re_attempted_because_refusing_costs_nothing(tmp_path):
    """The asymmetry the table implies. A REFUSED arm spent no pod minutes and
    the usual reason to re-run this driver is that the precondition it named was
    fixed, so REFUSED is the one finished state that is not skipped."""
    ledger = tmp_path / "ARMS.tsv"
    logs = tmp_path / "logs"
    logs.mkdir()
    for _ in range(2):
        got = lift('arm stub bash -c "exit 2"',
                   REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                   DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
        assert "SKIP stub" not in got.stdout
    rows = [r.split("\t") for r in ledger.read_text().splitlines()]
    assert [r[1] for r in rows] == ["REFUSED", "REFUSED"]


def test_the_ledger_records_the_dirty_file_count_after_every_arm(tmp_path):
    """AUDIT A6/P1. `calibrate_hardware.py` writes a TRACKED yaml and the anchor
    re-score rewrote two TRACKED files, both while the session ran, so a P1
    check at preflight passed and said nothing about the tree the later arms
    measured on. The count is re-asked after every arm and lands in its own
    column; a growth prints a warning naming the two file counts."""
    ledger = tmp_path / "ARMS.tsv"
    logs = tmp_path / "logs"
    logs.mkdir()
    got = lift('arm stub bash -c "exit 0"',
               REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert got.returncode == 0, got.stderr
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert row[4].isdigit(), row
    assert "dirty file(s)" in got.stdout
    # The FAIL branch: an arm that writes into the tree is named as having done
    # it. The stub writes an untracked file under the repo, which is what
    # --untracked-files=all is counted for.
    scratch = ROOT / "___dirty_probe_for_the_driver_test.tmp"
    try:
        dirtying = lift(f'arm dirtystub bash -c "echo x > {scratch}"',
                        REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs),
                        ONLY="", DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
        assert "WARNING: this arm dirtied the work tree" in dirtying.stdout
    finally:
        scratch.unlink(missing_ok=True)


def test_a_measuring_run_refuses_without_a_card_and_writes_nothing(tmp_path):
    """Every verdict here is scored against a per-card calibrated ridge -- 145.8
    on the A100 against 163.7 on the H200 -- so there is no card-free measuring
    mode and none is offered. The driver's own stop codes speak the same table
    as the arms': nothing measured is REFUSED, which is 2."""
    session = tmp_path / "s"
    got = run([], session=session)
    assert got.returncode == exit_codes.REFUSED
    assert "REFUSED: no CUDA device" in got.stdout
    assert not session.exists()


def test_the_resume_check_does_not_depend_on_a_gnu_only_grep():
    """`grep -qP` is a GNU extension. On a laptop it fails, the resume check
    silently says "not done" for every arm, and a re-run repeats work -- or
    worse, the same construct is trusted on the pod after being tested nowhere.
    Both reference drivers use it; this one uses awk."""
    body = CODE.split("\narm() {", 1)[1].split("\n}", 1)[0]
    assert "grep -qP" not in body
    assert "awk -F" in body


# --------------------------------------------------------------------------
# 5. the card is in every name
# --------------------------------------------------------------------------

def test_the_results_root_carries_the_card(tmp_path):
    """AUDIT A5. The comment said the card was "IN BOTH NAMES" while the export
    was a bare /workspace/results. That volume outlives the pod: two published
    arms, one for sm_count 132 and one for 108, contain a report file of the
    same name because a second card resumed the first card's directories."""
    got = run(["--dry-run"], session=tmp_path / "s")
    line = [ln for ln in got.stdout.splitlines() if "MOE_RESULTS_DIR" in ln
            and "results " in ln]
    assert line, got.stdout
    # Off a GPU box the card cannot be named, so the literal "nocard" is what
    # goes in the path AND the driver says so rather than leaving the reader to
    # infer it from an absence.
    assert "nocard" in line[0]
    assert "card      nocard -- " in got.stdout
    assert "the card is in the name" in line[0]


def test_a_results_root_that_does_not_name_the_card_is_refused(tmp_path):
    """The gate has both branches. An operator-supplied MOE_RESULTS_DIR is not
    silently rewritten -- that would put output where they are not looking --
    and it is not silently accepted either."""
    bad = run(["--dry-run"], session=tmp_path / "s",
              env_extra={"MOE_RESULTS_DIR": str(tmp_path / "plain")})
    assert bad.returncode == exit_codes.REFUSED
    assert "does not contain the card" in bad.stdout
    good = run(["--dry-run", "--only", "counter_plan"], session=tmp_path / "s2",
               env_extra={"MOE_RESULTS_DIR": str(tmp_path / "gaps-nocard")})
    assert good.returncode == 0, good.stdout[-2000:]
    assert "does not contain the card" not in good.stdout


# --------------------------------------------------------------------------
# 6. the summary may read only the one line a scored gate prints
# --------------------------------------------------------------------------

PROSE_LOG = """\
reading the imported floor: 0.0905 at the design this study runs
C1  the crossing ratio is above 1 + alpha  [PASS]
V3  VALIDITY PASS   sigma 0.0228 carried from NOISE_FLOOR.json
REFUSED: no calibration for this device. Nothing was measured.
"""

RESULT_LOG = """\
prose that mentions PASS and floor and sigma, at length
C1 ... [PASS]   <- a pre-registered expectation, not a result
RESULT: VALIDITY V0 PASS override_config changed the kernel
RESULT: CLAIM C1 FAIL plateau 0.47 of roof, control 0.52, gap 0.05 < 0.10
"""


def test_a_log_of_pure_prose_yields_an_empty_summary(tmp_path):
    """THE DEFECT, PLANTED. The old summary grepped `floor|sigma` for the
    noise-floor arm and `^\\[(PASS|FAIL)\\]` for others. Against a REFUSED log it
    matched eighteen times and printed `floor: 0.0905` -- a prior imported from
    a file -- and `C1 ... [PASS]` -- a pre-registered expectation -- under the
    heading "THE GATES", as this session's output. This log contains both and no
    RESULT line, and the summary must print NOTHING from it."""
    log = tmp_path / "arm.log"
    log.write_text(PROSE_LOG)
    got = lift(f'result_lines {log}', REPO=str(ROOT))
    assert got.stdout == ""
    scored = lift(f'summarize_arm noise_floor {log} DONE', REPO=str(ROOT))
    assert scored.returncode == 1
    assert "0.0905" not in scored.stdout
    assert "PASS" not in scored.stdout


def test_a_log_with_result_lines_yields_exactly_those(tmp_path):
    """And the PASS branch: the two RESULT lines are printed verbatim, the prose
    around them is not, and what comes back parses as the gates that were
    scored."""
    log = tmp_path / "arm.log"
    log.write_text(RESULT_LOG)
    got = lift(f'summarize_arm roofline-n64-g1 {log} CLAIM_FAIL', REPO=str(ROOT))
    assert got.returncode == 0
    lines = [ln for ln in got.stdout.splitlines() if ln]
    assert lines == [ln for ln in RESULT_LOG.splitlines()
                     if ln.startswith("RESULT: ")]
    parsed = exit_codes.parse_result_lines(got.stdout)
    assert [(p.kind, p.name, p.verdict) for p in parsed] == [
        ("VALIDITY", "V0", "PASS"), ("CLAIM", "C1", "FAIL")]
    assert exit_codes.classify_text(got.stdout) == exit_codes.CLAIM_FAIL


def test_the_free_text_greps_are_gone():
    """The regexes themselves, not only their effect: `floor|sigma` and the
    bracketed-verdict patterns were per-arm case entries, and a per-arm regex is
    a per-arm opportunity to match prose."""
    assert "arm_gate_regex" not in CODE
    assert "floor|sigma" not in CODE
    assert "grep -E '^RESULT: '" in CODE
    # The header still NAMES the old grep, as the defect it was; that is the
    # one place the string is allowed to survive.
    assert "`floor|sigma` matched a REFUSED noise-floor log" in TEXT


def test_an_invalid_arm_says_in_the_summary_that_nothing_may_be_quoted(tmp_path):
    log = tmp_path / "arm.log"
    log.write_text(RESULT_LOG)
    got = lift(f'summarize_arm anchor_measure {log} INVALID', REPO=str(ROOT))
    assert "MEASURED, THEN A VALIDITY GATE FAILED" in got.stdout
    assert "NOT auto-retried" in got.stdout
    # ...and it still prints the gate lines, because which VALIDITY gate failed
    # is the whole content of an INVALID arm.
    assert "RESULT: VALIDITY V0 PASS" in got.stdout


def test_an_arm_with_no_result_line_is_reported_rather_than_omitted(tmp_path):
    """A CHECK THAT EXAMINED NOTHING REPORTS NO FAILURES. An arm whose log
    carries no RESULT line has not passed its gates; printing nothing for it
    would read as having found nothing wrong."""
    got = run(["--dry-run"], session=tmp_path / "s")
    assert "PLAN ONLY: a --dry-run scores no gate" in got.stdout
    assert "This arm was NOT scored" in TEXT
    assert "do not read prose in it as a verdict" in TEXT


# --------------------------------------------------------------------------
# 7. the plan states its own detection limit
# --------------------------------------------------------------------------

def test_the_plan_prints_an_mde_from_a_stated_noise_assumption(tmp_path):
    """AUDIT B14. No arm in this study states an MDE, and the driver told the
    operator to "read every effect against the noise floor" while the floor
    itself has never been measured. The line is computed through
    `replicate_noise_floor.mde_external_sigma` -- the known-variance form, the
    only one that describes a design with one run per condition -- from the
    sigma in NOISE_FLOOR.json, and it says ASSUMED in the same breath because
    part (a) has not run on a card."""
    got = run(["--dry-run", "--only", "counter_plan"], session=tmp_path / "s")
    mde = [ln for ln in got.stdout.splitlines() if ln.strip().startswith("MDE ")]
    assert len(mde) == 1, got.stdout
    assert re.search(r"MDE 0\.\d{4} in fitted alpha", mde[0]), mde[0]
    sigma = [ln for ln in got.stdout.splitlines() if ln.strip().startswith("sigma ")]
    assert len(sigma) == 1 and "ASSUMED" in sigma[0], got.stdout
    assert "CONFOUNDS num_stages" in sigma[0]
    assert "0.0117, which is BELOW it" in got.stdout


def test_the_mde_line_refuses_rather_than_inventing_a_number(tmp_path):
    """The FAIL branch, planted: with no readable noise floor the session states
    NO detection limit and says every effect is unscored. A default sigma here
    would be a made-up denominator under every effect the session reports."""
    empty = tmp_path / "norepo"
    (empty / "results" / "published").mkdir(parents=True)
    got = lift('mde_line', REPO=str(empty), PY_BASE=sys.executable)
    assert got.returncode == 1
    assert "MDE UNAVAILABLE" in got.stdout
    assert "states no detection limit" in got.stdout


# --------------------------------------------------------------------------
# 8. the header says what the data say
# --------------------------------------------------------------------------

def test_the_regime_paragraph_is_counted_on_padded_rows_and_says_uniform_only():
    """AUDIT B3. The header said "128 is the only tile vLLM runs multi-tile (59
    of 87 cells, up to 32 tiles), at 16/32/64 it never does". Counted on
    `load_max_rows`, what `moe_align_block_size` actually pads to, it is 65 of
    87 and up to 33-34 tiles, BLOCK_M=16 fires in 1 of 24 cells and BLOCK_M=64
    in 5 of 112 -- isolated, not never. And every one of those counts is over
    UNIFORM routing; skewed routings, which is what production serves, were
    never counted at all."""
    header = TEXT.split("set -uo pipefail", 1)[0]
    assert "65 of 87" in header
    assert "33-34" in header
    assert "1 of 24" in header and "5 of 112" in header
    assert "SKEWED ROUTINGS WERE NEVER COUNTED" in header
    assert "uniform routing" in header.lower()
    assert "59 of 87" not in TEXT.replace(
        'The earlier "59 of 87, up to 32, and 16/32/64 never" was counted on', "")


def test_the_retracted_cap_identity_is_not_asserted_anywhere():
    """AUDIT B1. The header carried "cap = 2*BM/(alpha_fitted*b) is EXACTLY the
    three-term cap; the published cap numbers stand", and that sentence set this
    file's whole order. The test that "proved" it defined alpha_fitted by the
    formula under test. The estimator in use is B/(A+B), the (EXA) form in
    moe/bench/ai_model.py, and a cap built from a fitted alpha is 31% high at
    BLOCK_M=128 -- more than the cap-to-ridge gap it was deciding."""
    assert "the published cap numbers stand" not in CODE
    assert "EXACTLY the three-term cap" not in CODE
    assert "THE CAP IDENTITY IS RETRACTED" in TEXT
    # It survives in the header only inside the quoted sentence being retracted,
    # on the same line as the word that retracts it, so no reader can take it as
    # current.
    for line in TEXT.splitlines():
        if "the published cap numbers stand" in line:
            assert "was a tautology" in line, line
    assert "moe/bench/ai_model.py" in TEXT
    assert "(EXA)" in TEXT


def test_the_g1_decomposition_arm_does_not_claim_a_c2_it_cannot_measure():
    """AUDIT A9. `bn_decomposition --self-test --group-m 1` passes C2 in the
    PLANTED MISSING world -- chi2 1.78 against a 4.0 threshold -- so at
    GROUP_SIZE_M=1 C2 cannot fail whatever the card does, and the driver said
    the arm "still measures alpha_b, C2 and C3"."""
    assert "C2 is UNKNOWN there however the data fall" in TEXT
    listing = run(["--list"]).stdout
    block = listing.split("  bn_g1 ", 1)[1].split("\n\n", 1)[0]
    assert "NOT a second reading of C2" in block
    assert "alpha_b and C3" in block


# --------------------------------------------------------------------------
# 9. git is asked, not remembered
# --------------------------------------------------------------------------

def test_it_asks_git_check_ignore_rather_than_asserting_the_rule():
    """FAILURE MODE 3. `.gitignore` ignores `results/*` and re-includes only
    `results/published/`, so the answer differs for three kinds of path and a
    remembered sentence is right for one of them. This repo has already lost
    every published plot of ten arms that way."""
    assert "git check-ignore" in TEXT
    assert "check-ignore -q" in TEXT
    assert "UNVERIFIED" in TEXT
    assert "It is NOT 'tracked'" in TEXT


def test_the_git_verdict_is_asked_of_a_path_an_arm_really_writes(tmp_path):
    """`results/` itself is not ignored -- the pattern is `results/*` -- so
    asking about the ROOT answers the wrong question and reports "tracked" for a
    tree whose every child git drops."""
    got = run(["--dry-run"], session=tmp_path / "s")
    assert "/bm128_roofline" in got.stdout


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
# 10. the card probe, the capability parse, and the gate they feed
# --------------------------------------------------------------------------
#: The shipped read, lifted verbatim rather than retyped, so a change to the
#: delimiter or the field order fails here instead of on a pod.
READ_FIELDS = "IFS='|' read -r CARD CAPABILITY CARD_REASON"
#: The two shipped lines that turn a compute capability into a major number.
PARSE = re.search(r'^SM_MAJOR=""\n\[\[ "\$CAPABILITY" =~ [^\n]*\n', TEXT, re.M).group(0)


def read_probe(line):
    """Run the driver's own field split over one planted probe line."""
    body = (f'set -uo pipefail\n{READ_FIELDS} '
            f'< <(printf "%s\\n" {shlex.quote(line)})\n'
            'printf "%s\\n%s\\n%s\\n" "$CARD" "$CAPABILITY" "$CARD_REASON"\n')
    done = subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.split("\n")[:3]


def test_the_probe_is_pipe_delimited_because_a_prose_reason_ate_the_capability():
    """THE 2026-09-02 LIVE BUG, PLANTED. The probe used to split on a character
    the reasons themselves contain, so `compute capability unreadable:
    RuntimeError, no CUDA` put the words `no CUDA` in CAPABILITY. `pre_hopper`
    was then written as an arithmetic comparison, bash looked for a variable
    named CUDA, and `set -u` ended the session at the second arm. A pipe cannot
    appear in a reason these branches write, and the reason is the LAST field,
    so anything unexpected in it stays in it."""
    assert READ_FIELDS in CODE
    # The python side of the probe emits the same three pipe-joined fields on
    # every exit, including the two that give up.
    assert 'print(f"{card_slug(prov.gpu_name)}|{capability}|")' in TEXT
    assert TEXT.count('print(f"nocard||') == 2
    card, cap, reason = read_probe(
        "nocard||compute capability unreadable: RuntimeError, no CUDA, none")
    assert card == "nocard"
    assert cap == "", "the reason must not reach the capability field"
    assert reason == "compute capability unreadable: RuntimeError, no CUDA, none"
    # And the PASS side: a real card fills the first two and leaves the third empty.
    assert read_probe("nvidia_h200|9.0|") == ["nvidia_h200", "9.0", ""]


@pytest.mark.parametrize("capability,major", [
    ("9.0", "9"), ("8.0", "8"), ("10.0", "10"),
    ("", ""), ("no CUDA", ""), ("nine.zero", ""), ("9", ""),
])
def test_the_capability_major_is_taken_only_from_a_number(capability, major):
    """A capability that did not parse leaves SM_MAJOR EMPTY rather than
    guessing. `9` with no dot is deliberately empty: the field this driver reads
    is always `major.minor`, and a bare integer is a string from somewhere
    else."""
    body = (f"set -uo pipefail\nCAPABILITY={shlex.quote(capability)}\n"
            f'{PARSE}printf "%s\\n" "$SM_MAJOR"\n')
    done = subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == major


@pytest.mark.parametrize("sm_major,skips", [
    ("7", True), ("8", True), ("9", False), ("10", False), ("", False),
])
def test_pre_hopper_skips_a_pre_hopper_card_and_never_skips_on_a_string_it_could_not_read(
        sm_major, skips):
    """ALL THREE BRANCHES OF THE GATE THAT DECIDES FOUR ARMS. sm_80 skips them,
    because BLOCK_N=256 at four stages asks 192 KiB of shared memory against
    that card's 164. sm_90 runs them. An EMPTY SM_MAJOR runs them too: the gate
    exists to spare a card that provably cannot hold the shape, and a
    capability nobody could parse is not that proof. Skipping on an unreadable
    string would drop the session's only confirming arm and say nothing."""
    got = lift('if pre_hopper; then echo SKIP; else echo RUN; fi',
               REPO=str(ROOT), SM_MAJOR=sm_major)
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == ("SKIP" if skips else "RUN")


def test_an_unreadable_capability_travels_the_whole_chain_without_ending_the_run():
    """The probe, the parse and the gate together, on the exact string that
    ended a session. Under `set -u`, and the run must still be RUN and rc 0."""
    script = (f'{READ_FIELDS} < <(printf "%s\\n" '
              f'{shlex.quote("nocard||compute capability unreadable: no CUDA")})\n'
              f'{PARSE}'
              'if pre_hopper; then echo SKIP; else echo RUN; fi\n')
    got = lift(script, REPO=str(ROOT))
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == "RUN"


def test_the_pre_hopper_gate_decides_exactly_the_four_arms_that_need_the_shape():
    """The gate is not free: every arm behind it is an arm the session does not
    run. These four are the ones whose configuration a pre-Hopper card cannot
    hold; anything else appearing here is an arm silently dropped."""
    guarded = re.findall(r"if pre_hopper; then\n\s*skip_arm ([\w.-]+)", CODE)
    assert sorted(guarded) == sorted(["pin_probe-n256-g16", "roofline-n256-g16",
                                      "roofline-n256-g32", "mma_switch"])
    for name in guarded:
        assert 'compute capability $CAPABILITY' in CODE.split(f"skip_arm {name} ", 1)[1][:400] \
            or 'capability $CAPABILITY' in CODE.split(f"skip_arm {name} ", 1)[1][:400], name


# --------------------------------------------------------------------------
# 11. the state word is disclosed when the file that produced it uses another table
# --------------------------------------------------------------------------

def adopting_repo(tmp_path, rel):
    """A repo whose `rel` imports the module, for the caveat's silent branch."""
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("from moe.bench.exit_codes import classify\n")
    return tmp_path


def test_a_blocked_counter_route_is_not_reported_as_a_broken_instrument(tmp_path):
    """THE DEFECT, PLANTED. `dram_counter_route.py` ends `return 0 if verdict ==
    "OPEN" else 3`, so BLOCKED -- the answer that arm exists to obtain, and the
    expected one on a rented pod -- exits 3, which the adopted table reads as
    INVALID. The base driver carried `counter_plan) echo 0,3` for exactly this;
    R1 deleted the per-arm lists and that script is not on audit A4's fix list,
    so the row is wrong at merge and the only honest thing this file can do is
    say so beside it."""
    log = tmp_path / "counter_plan.log"
    log.write_text(RESULT_LOG)
    got = lift(f'summarize_arm counter_plan {log} INVALID', REPO=str(ROOT))
    assert "dram_counter_route.py returns 3 for" in got.stdout
    assert "not OPEN" in got.stdout
    assert "not a broken instrument" in got.stdout
    # and it still says what INVALID does to the ledger, because that is what
    # the operator has to undo.
    assert "delete this row from" in got.stdout


def test_an_unadopted_refusal_says_two_may_mean_the_opposite(tmp_path):
    """The other direction of the same collision. `memory_branch_anchor.py`
    documents 2 as a VALIDITY gate that failed AFTER an eight-minute
    measurement; this session reads 2 as REFUSED and prints "REFUSED BEFORE
    MEASURING". Until that script adopts the module the summary must not leave
    that sentence standing alone."""
    log = tmp_path / "anchor.log"
    log.write_text("REFUSED: no calibration for this device.\n")
    got = lift(f'summarize_arm anchor_measure {log} REFUSED', REPO=str(ROOT))
    assert "REFUSED BEFORE MEASURING" in got.stdout
    assert "DOCUMENTS 2 as a" in got.stdout
    assert "eight-minute measurement" in got.stdout


def test_the_caveat_is_silent_once_the_file_speaks_the_table(tmp_path):
    """THE PASS BRANCH, and the reason this is asked of the file rather than
    kept in a list here: a sibling slice landing the fix must turn the caveat
    off without anyone editing this driver."""
    repo = adopting_repo(tmp_path, "scripts/dram_counter_route.py")
    got = lift('contract_caveat counter_plan INVALID; echo "rc=$?"', REPO=str(repo))
    assert got.stdout.strip() == "rc=1"
    assert "CAVEAT" not in got.stdout


def test_a_file_the_driver_cannot_find_is_unknown_and_not_adopted(tmp_path):
    """REFUSE RATHER THAN DEFAULT. An arm whose script is missing gets the
    caveat, worded as UNKNOWN, because "could not check" is not "checked and
    fine"."""
    got = lift('contract_caveat counter_plan INVALID', REPO=str(tmp_path))
    assert "could not find" in got.stdout
    assert "UNKNOWN" in got.stdout
    rcs = lift('adopts_exit_codes scripts/nothing_here.py; echo "rc=$?"',
               REPO=str(tmp_path))
    assert rcs.stdout.strip() == "rc=2"


def test_the_measuring_run_prints_the_disclosure_the_dry_run_used_to_have_alone(tmp_path):
    """AUDIT A4, the half that was left in the cheap branch. The `--dry-run`
    banner said a refusal exiting 3 is "a refusal wearing an INVALID's number";
    the measuring path, where the mislabel costs an arm, said nothing. Both
    branches of the new section are planted: rows from unadopted files, and a
    ledger with none."""
    assert '(( DRY )) || contract_disclosure "$LEDGER"' in CODE
    ledger = tmp_path / "ARMS.tsv"
    ledger.write_text(
        "arm\tstate\trc\tseconds\tdirty\tlog\tnote\n"
        "counter_plan\tINVALID\t3\t61\t0\t/x.log\t\n"
        "anchor_measure\tREFUSED\t2\t480\t0\t/y.log\t\n"
        "ruler\tDONE\t0\t9\t0\t/z.log\t\n")
    got = lift(f'contract_disclosure {ledger}', REPO=str(ROOT))
    assert "THE ROWS WHOSE STATE MAY BE THE WRONG WORD" in got.stdout
    assert "counter_plan        INVALID" in got.stdout
    assert "anchor_measure      REFUSED" in got.stdout
    assert "ruler" not in got.stdout, "a DONE row has no state to disclose"

    clean = tmp_path / "CLEAN.tsv"
    clean.write_text("arm\tstate\trc\tseconds\tdirty\tlog\tnote\n"
                     "ruler\tDONE\t0\t9\t0\t/z.log\t\n")
    ok = lift(f'contract_disclosure {clean}', REPO=str(ROOT))
    assert "THE EXIT-CODE CONTRACT" in ok.stdout
    assert "WRONG WORD" not in ok.stdout
    assert "No REFUSED or INVALID row" in ok.stdout


def test_the_disclosure_restores_no_per_arm_state_map():
    """R1 DELETED THE PER-ARM DONE-CODE LISTS AND THIS PUTS NONE BACK. The new
    functions print words; they call `ledger_state` nowhere and assign `state`
    nowhere, so the ledger word still comes from the one table and from nothing
    else."""
    for fn in ("arm_script", "adopts_exit_codes", "contract_caveat",
               "contract_disclosure"):
        body = CODE.split(f"\n{fn}() ", 1)[1].split("\nesac; }", 1)[0] \
            if f"\n{fn}() {{ case" in CODE else \
            CODE.split(f"\n{fn}() {{", 1)[1].split("\n}", 1)[0]
        assert "ledger_state" not in body, fn
        assert not re.search(r'^\s*state=', body, re.M), fn
    assert "arm_done_codes" not in TEXT


def test_the_counter_arm_says_to_read_its_verdict_and_not_its_ledger_state():
    listing = run(["--list"]).stdout
    block = listing.split("  counter_plan ", 1)[1].split("\n\n", 1)[0]
    assert "READ ITS VERDICT LINE, NOT ITS LEDGER STATE" in block
    assert "BLOCKED is the ANSWER" in block
