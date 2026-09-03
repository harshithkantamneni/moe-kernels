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
  6. THE COST TABLE IS THE ARMS' OWN. Added 2026-09-03, after a pod-readiness
     check ran every arm the way this driver invokes it and found the table
     wrong by 2.1x overall and by 10x on the noise floor -- an operator sizing
     a rental off it under-booked by half a day, and the session it bought
     reached minute 12, entered a 240-minute arm booked at 25, and was killed
     inside it. Every booked figure is now checked against the plan that
     printed it, by RUNNING that plan, and the TOTAL is checked to be the sum
     of the column rather than a constant.
  7. A GATE THAT CANNOT FAIL IS NOT A GATE. Three measuring arms ran without
     the flag their own script needs to exit CLAIM_FAIL, and both
     `occupancy_vs_swizzle.exit_for` and `bn_decomposition.exit_for` downgrade
     a failed claim to exit 0 DONE without it. The check walks every arm's
     measuring invocation and asks the SCRIPT which gate flag it defines, and a
     second test reproduces the downgrade live rather than asserting it from
     memory. The same shape is checked on the advertised off-GPU commands,
     where the span line returned the same code whether the planted world was
     reproduced or not.
  8. A --dry-run THAT PREVIEWS A DIFFERENT RUN IS NOT A PREVIEW. Four arms
     planned something the pod does not execute -- a skipped calibrate, a
     rescore-mode anchor, a depth sweep at the wrong --r-max, a dtype arm with
     no card to name -- so the tests below plant the distinguishing flag and
     then run the whole dry session and check every arm reached a plan.

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
# bn_g1 was dropped on 2026-09-03: at GROUP_SIZE_M=1 `bn_decomposition.py
# --self-test` exits 3 INVALID and the arm's own plan says C1 reads UNKNOWN
# however the data fall, and no other pinning that self-test was checked at
# passes either, so there was nowhere to re-pin it to.
ARMS = ("calibrate", "pin_probe-n64-g1", "pin_probe-n256-g16",
        "roofline-n64-g1", "roofline-n256-g16", "roofline-n256-g32",
        "bm128_depth", "noise_floor",
        "bn_g16", "anchor_measure", "anchor_rescore", "occupancy",
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
    assert "bn_g1" not in order
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
    """THE ONE NUMBER THAT DECIDES WHETHER TO RENT USED TO BE PRINTED ONLY
    AFTER THE DECISION. `TOTAL ~$total minutes` sat inside the `else` of
    `if (( DRY ))`, so --dry-run listed twenty per-arm minutes and no sum, and
    the sum an operator would have reached by hand was wrong by nearly four
    hours. Both modes print it now, with the cumulative minute each arm starts
    at beside it, because "how long is the session" and "what is still
    unstarted when I release the pod" are different questions."""
    got = run(["--dry-run"], session=tmp_path / "s")
    assert "WHAT THIS COMMITS YOU TO" in got.stdout
    body = got.stdout.split("WHAT THIS COMMITS YOU TO")[1]
    assert body.index("SESSION  card=") > 0
    for name in ARMS:
        assert re.search(rf"  {re.escape(name)}\s+~\s*\d+ min", body), name
    total = re.search(r"TOTAL ~(\d+) minutes \(~(\d+)h (\d+)m\)", body)
    assert total, body
    minutes = int(total.group(1))
    assert minutes == int(total.group(2)) * 60 + int(total.group(3))
    # The sum is the arms', not a constant: add the column back up.
    listing = run(["--list"]).stdout
    booked = {n: int(re.search(rf"^  {re.escape(n)}\s+~(\d+) min$",
                               listing, re.M).group(1)) for n in ARMS}
    assert sum(booked.values()) == minutes
    # And every arm says the cumulative minute it starts at, which must be the
    # running sum of the arms above it and nothing else.
    running = 0
    for name in ARMS:
        found = re.search(
            rf"  {re.escape(name)}\s+~\s*\d+ min\s+starts at ~\s*(\d+) min", body)
        assert found, name
        assert int(found.group(1)) == running, (name, found.group(1), running)
        running += booked[name]


def test_no_arm_books_a_figure_this_file_invented(tmp_path):
    """THE COST TABLE WAS WRONG BY 2.1x OVERALL AND BY 10x ON THE NOISE FLOOR,
    and the fix is not a better guess: every row now names the command whose
    plan printed its figure, and what that figure leaves out. A row with no
    basis is a number somebody made up, which is the state the whole table was
    in."""
    body = run(["--dry-run"], session=tmp_path / "s").stdout.split(
        "WHAT THIS COMMITS YOU TO")[1].split("SESSION  card=")[0]
    for name in ARMS:
        basis = lift(f'arm_basis {shlex.quote(name)}', REPO=str(ROOT))
        assert basis.returncode == 0 and basis.stdout.strip(), name
        assert basis.stdout.strip() in body, name
    # The three arms whose own plan refuses before spending anything are booked
    # ZERO. Booking minutes for a refusal hides that the answer is already in.
    for name in ("roofline-n256-g16", "roofline-n256-g32", "span"):
        got = lift(f'arm_minutes {shlex.quote(name)}', REPO=str(ROOT))
        assert got.stdout.strip() == "0", name
        assert "REFUSE" in lift(f'arm_basis {shlex.quote(name)}',
                                REPO=str(ROOT)).stdout.upper(), name
    # And an arm whose figure excludes compiles says so where the figure is.
    assert "NOT IN THAT FIGURE" in body
    for name in ("span_dense", "dtype", "bn_g16"):
        assert lift(f'arm_unpriced {shlex.quote(name)}',
                    REPO=str(ROOT)).stdout.strip(), name
    # An arm whose plan already prints a WALL clock has nothing unpriced.
    for name in ("noise_floor", "anchor_measure", "ruler"):
        assert not lift(f'arm_unpriced {shlex.quote(name)}',
                        REPO=str(ROOT)).stdout.strip(), name


# --------------------------------------------------------------------------
# 3. the flags still exist on the far side
# --------------------------------------------------------------------------

INVOKED = {
    "scripts/calibrate_hardware.py": ("--publish", "--dry-run"),
    "scripts/ruler_rebaseline.py": ("--dry-run", "--fail-on-gate"),
    "scripts/check_mma_path.sh": ("--block-m", "--tokens", "--model", "--out",
                                  "--dry-run"),
    "scripts/tile_cap_test.py": ("--dry-run", "--capability", "--fail-on-gate"),
    "scripts/dtype_tile_confound.py": ("--dry-run", "--card", "--fail-on-claim"),
    "scripts/span_extent_separation.py": ("--dry-run", "--densify",
                                          "--no-densify", "--fail-on-world"),
    "scripts/bm128_roofline.py": ("--block-n", "--group-m", "--control",
                                  "--dry-run", "--fail-on-gate"),
    "scripts/bm128_depth.py": ("--dry-run", "--r-max", "--fail-on-gate"),
    "scripts/bn_decomposition.py": ("--dry-run", "--group-m", "--reps",
                                    "--capability", "--fail-on-gate"),
    "scripts/occupancy_vs_swizzle.py": ("--dry-run", "--run", "--fail-on-gate"),
    "scripts/replicate_noise_floor.py": ("--dry-run", "--replicates", "--arms",
                                         "--publish"),
    "scripts/memory_branch_anchor.py": ("--rescore", "--out-dir", "--dry-run",
                                        "--measure"),
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


# THE GATE FLAGS. A script that defines one and is not given it can only ever
# report the gate PASSING, and the measuring arms are exactly where that costs
# an hour of rented card. `--fail-on-world` is deliberately NOT in this set: it
# scores a --self-test's PLANTED WORLDS, not a measured run, so it belongs on
# the off-GPU line and nowhere near a pod invocation.
GATE_FLAGS = ("--fail-on-gate", "--fail-on-claim")


def measuring_invocation(arm_name):
    """The one command line the pod runs for `arm_name`, as shell words.

    Read out of the file rather than listed here, because a list of what the
    driver runs is a second copy of the driver and goes stale the day someone
    edits the first."""
    joined = re.sub(r"\\\n\s+", " ", CODE)
    found = [ln for ln in joined.splitlines()
             if re.match(rf"\s*arm {re.escape(arm_name)}\s", ln)]
    # The dry-run branch and the measuring branch, in that order in every arm.
    measuring = [ln for ln in found if "--dry-run" not in ln]
    assert len(measuring) == 1, (arm_name, found)
    return shlex.split(measuring[0])


@pytest.mark.parametrize("arm_name", [
    a for a in ARMS if a not in ("anchor_rescore",)])
def test_every_arm_is_given_the_gate_flag_its_own_script_defines(arm_name):
    """THE THREE ARMS THAT COULD NOT REACH THE STATE THEY WERE SCHEDULED FOR.
    occupancy and both bn arms ran without --fail-on-gate, and
    `occupancy_vs_swizzle.exit_for` / `bn_decomposition.exit_for` both downgrade
    a CLAIM_FAIL to exit 0 DONE when it is absent. So ~94 minutes of the old
    schedule could only ever land DONE, REFUSED or INVALID, and occupancy's P2
    -- which the driver says is EXPECTED to fail, and that FAIL is the finding
    -- had no code to land on.

    Asked of the SCRIPT rather than of a list kept here: if a file defines a
    gate flag, the arm that runs it has to pass it. A script that retires its
    flag keeps accepting it (three of them say so in their own --help), so
    passing it is also what stops a re-introduced downgrade from silently taking
    an arm's CLAIM_FAIL away again."""
    rel = lift(f"arm_script {shlex.quote(arm_name)}", REPO=str(ROOT)).stdout.strip()
    if not rel or not (ROOT / rel).exists():
        pytest.skip(f"{arm_name} runs no file under this repo")
    source = (ROOT / rel).read_text()
    defines = [f for f in GATE_FLAGS if f'"{f}"' in source]
    if not defines:
        pytest.skip(f"{rel} defines no gate flag")
    words = measuring_invocation(arm_name)
    for flag in defines:
        assert flag in words, (
            f"{arm_name} runs {rel}, which defines {flag}, without it. "
            f"Its exit_for downgrades a CLAIM_FAIL to 0 DONE when the flag is "
            f"absent, so the arm cannot report the state it is scheduled for.")


def test_the_downgrade_this_flag_closes_is_real_and_not_remembered():
    """The finding, reproduced rather than asserted. Off GPU, live:
    `occupancy_vs_swizzle.py --audit` prints RESULT: CLAIM ... FAIL and exits 0,
    while `exit_codes.classify_text` over that same log returns 1 -- a
    disagreement between the process and its own page, which
    moe/bench/exit_codes.py's docstring names as itself a defect. With the flag
    both are 1."""
    script = str(ROOT / "scripts" / "occupancy_vs_swizzle.py")
    without = subprocess.run([sys.executable, script, "--audit"],
                             capture_output=True, text=True, timeout=900,
                             cwd=str(ROOT))
    assert "RESULT: CLAIM" in without.stdout and "FAIL" in without.stdout
    assert exit_codes.classify_text(without.stdout) == exit_codes.CLAIM_FAIL
    assert without.returncode == exit_codes.DONE, "the downgrade is gone; drop this test"
    with_flag = subprocess.run([sys.executable, script, "--audit",
                                "--fail-on-gate"],
                               capture_output=True, text=True, timeout=900,
                               cwd=str(ROOT))
    assert with_flag.returncode == exit_codes.CLAIM_FAIL
    assert exit_codes.classify_text(with_flag.stdout) == with_flag.returncode


def test_the_advertised_off_gpu_gates_can_actually_fail():
    """A GATE THAT EXAMINED NOTHING REPORTS NO FAILURES, which is the shape
    exit_codes.py is named against. The span arms' advertised check was
    `span_extent_separation.py --self-test kernel|extent|neither --densify`,
    and all three exit 2 with ZERO `RESULT: ` lines: the command returned the
    same code whether the planted world was reproduced or not, and the script's
    own last line says why ("Pass --fail-on-world to score the S gates"). With
    the flag it is 15 RESULT lines per world. The occupancy line had the same
    shape for the same reason."""
    for arm_name, flag in (("span_dense", "--fail-on-world"),
                           ("span", "--fail-on-world"),
                           ("occupancy", "--fail-on-gate")):
        advertised = lift(f"arm_offgpu_gates {shlex.quote(arm_name)}",
                          REPO=str(ROOT)).stdout
        assert flag in advertised, (arm_name, advertised)
    script = str(ROOT / "scripts" / "span_extent_separation.py")
    for world in ("kernel", "extent", "neither"):
        bare = subprocess.run(
            [sys.executable, script, "--self-test", world, "--densify"],
            capture_output=True, text=True, timeout=900, cwd=str(ROOT))
        assert bare.returncode == exit_codes.REFUSED
        assert "RESULT: " not in bare.stdout, world
        scored = subprocess.run(
            [sys.executable, script, "--self-test", world, "--densify",
             "--fail-on-world"],
            capture_output=True, text=True, timeout=900, cwd=str(ROOT))
        assert scored.stdout.count("RESULT: ") > 0, world
        assert exit_codes.classify_text(scored.stdout) == scored.returncode


def test_the_noise_floor_is_bounded_published_and_booked_at_its_own_plan():
    """THE ARM THAT ATE THE SESSION. Booked 25 minutes against its own plan's
    240, run bare so the defaults stood, with no deadline or timeout anywhere in
    this file -- so a session sized off the old table entered it at minute 12
    and was still inside it when the pod was released, which fails that
    script's own V2 and makes even the partial floor unquotable. Bare also
    meant no --publish, and NOISE_FLOOR.json is written under that flag alone,
    so the sigma every future MDE line prints stayed ASSUMED.

    This runs the arm's own --dry-run with exactly the flags the driver passes
    and compares its TOTAL with the booking."""
    words = measuring_invocation("noise_floor")
    assert "--publish" in words, "the arm writes nothing without it"
    assert "--replicates" in words and words[words.index("--replicates") + 1] == "3"
    arms = words[words.index("--arms") + 1]
    assert arms == "$NOISE_ARMS", words
    named = re.search(r'^NOISE_ARMS=(\S+)$', CODE, re.M).group(1)
    models = {a.split("_")[0] for a in named.split(",")}
    assert len(models) >= 2, (
        "V7 is a VALIDITY gate: a single-model floor lands the whole arm on "
        "3 INVALID unless --single-model-floor is also given")
    for model in models:
        assert {f"{model}_g1", f"{model}_g16"} <= set(named.split(",")), (
            "C3 is a swizzle contrast and needs G=1 AND G=16 of the SAME "
            f"model; {model} has only one of them, so its C3 row would read "
            "G=1 against G=1")
    plan = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "replicate_noise_floor.py"),
         "--dry-run", "--replicates", "3", "--arms", named],
        capture_output=True, text=True, timeout=900, cwd=str(ROOT))
    total = re.search(r"^TOTAL: ~(\d+) min of GPU", plan.stdout, re.M)
    assert total, plan.stdout[-3000:]
    booked = lift("arm_minutes noise_floor", REPO=str(ROOT)).stdout.strip()
    assert booked == total.group(1), (booked, total.group(1))
    # And --publish is NOT on the dry-run line: verified live, a --dry-run given
    # --publish WRITES results/published/NOISE_FLOOR.json, which git tracks, and
    # a plan that dirties the tracked tree is the anchor-rescore defect again.
    dry = [ln for ln in CODE.splitlines()
           if re.match(r"\s*arm noise_floor\s", ln)
           or "replicate_noise_floor.py" in ln]
    dry_line = " ".join(ln for ln in dry if "--dry-run" in ln)
    assert "--publish" not in dry_line, dry_line


@pytest.mark.parametrize("arm_name,flag", [
    ("calibrate", "--dry-run"),
    ("anchor_measure", "--measure"),
    ("bm128_depth", "--r-max"),
    ("dtype", "--card"),
])
def test_the_dry_run_previews_the_run_the_pod_executes(arm_name, flag):
    """FOUR ARMS PREVIEWED SOMETHING ELSE. calibrate was skipped entirely with
    a reason that is false ("has no --dry-run": it has one, prints a nine-line
    plan and exits REFUSED) -- and it is the arm whose gate can end the session
    at minute 3. anchor_measure dry-ran without --measure, so it previewed
    rescore mode and 26 committed reports instead of "cells 128 ... estimated
    wall time 4.8 min". bm128_depth dry-ran at the default --r-max, previewing
    126 s and a -r1024- run id against the pod's 252 s and -r2048-. dtype
    dry-ran without --card, refused with NoCardToLabel and landed PLAN_REFUSED,
    which reads as a broken arm rather than as this laptop having no GPU."""
    joined = re.sub(r"\\\n\s+", " ", CODE)
    dry = [ln for ln in joined.splitlines()
           if re.match(rf"\s*arm {re.escape(arm_name)}\s", ln)
           and "--dry-run" in ln]
    assert dry, arm_name
    # dtype has two plan branches on purpose: with a device attached the script
    # names its own card and a --card flag would override it, and off a GPU box
    # it is given a NAMED HYPOTHETICAL, which the session labels as one. Every
    # other arm has a single plan branch and it must carry the flag.
    assert (any if arm_name == "dtype" else all)(
        flag in ln for ln in dry), (arm_name, flag, dry)
    assert not re.search(rf"^\s*skip_arm {re.escape(arm_name)}\b", CODE, re.M), \
        f"{arm_name} is skipped rather than planned"


def test_every_arm_that_has_a_plan_mode_plans_on_this_laptop(tmp_path):
    """The end-to-end form of the four fixes above, on a box with no GPU: the
    only arms without a PLANNED row are the two pin probes and the anchor
    rescore, each of which is NOT_PLANNED with a reason. calibrate used to be a
    fourth, skipped for a reason that was false, and dtype a fifth, landing
    PLAN_REFUSED because it was given no card to name."""
    got = run(["--dry-run"], session=tmp_path / "s")
    rows = dict(ln.split("\t")[:2] for ln in
                (tmp_path / "s" / "ARMS-dryrun.tsv").read_text().splitlines()[1:])
    # NOT_PLANNED: no plan mode reachable here, named with the reason.
    # PLAN_REFUSED: the two BLOCK_N=256 rooflines print their refusal on line
    # one and never plan, which is that arm's own finding -- no BLOCK_M=256
    # control fits at BLOCK_N=256 -- and not a defect in this driver.
    expected = dict.fromkeys(ARMS, "PLANNED")
    expected.update(dict.fromkeys(
        ("pin_probe-n64-g1", "pin_probe-n256-g16", "anchor_rescore"),
        "NOT_PLANNED"))
    expected.update(dict.fromkeys(
        ("roofline-n256-g16", "roofline-n256-g32"), "PLAN_REFUSED"))
    for name in ARMS:
        assert rows.get(name) == expected[name], (
            name, rows.get(name), got.stdout[-3000:])


def test_no_arm_is_skipped_in_a_dry_run_with_a_reason_that_is_false(tmp_path):
    """The skip that stood over calibrate said "calibrate_hardware.py is a
    measurement and has no --dry-run". It has one. This asks the script."""
    assert "has no --dry-run" not in TEXT
    plan = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "calibrate_hardware.py"),
         "--dry-run"], capture_output=True, text=True, timeout=900,
        cwd=str(ROOT))
    assert plan.returncode == exit_codes.REFUSED
    assert len([ln for ln in plan.stdout.splitlines() if ln.strip()]) >= 9
    got = run(["--dry-run", "--only", "calibrate"], session=tmp_path / "s")
    ledger = (tmp_path / "s" / "ARMS-dryrun.tsv").read_text().splitlines()
    row = [ln.split("\t") for ln in ledger if ln.startswith("calibrate\t")]
    assert row and row[0][1] == "PLANNED", (row, got.stdout[-3000:])


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
    assert got.stdout.strip() == state, "with no log to read, rc 2 printed nothing"


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


def test_no_arm_is_scheduled_at_a_pinning_its_own_design_gate_calls_invalid():
    """THE G=1 DECOMPOSITION ARM WAS, and this is the check that ran too late.
    `bn_decomposition.py --self-test --capability 9.0 --group-m 1 --reps 17
    --plant-noise 0.008` exits 3 INVALID: S4 sees sd(alpha_a) 0.1759 against a
    gate of 0.025, and S5 sees the planted MISSING world PASS C2 at chi2 1.78
    against a ceiling of 4.0, so neither of that arm's two readouts can be
    resolved at the pinning it was booked at. The same command at --group-m 16
    exits 0. The arm was dropped rather than re-pinned because the self-test
    also fails at 2, 4, 8, 32 and 64, so GROUP_SIZE_M=16 -- the arm already
    scheduled -- is the only pinning left.

    This runs the gate at the pinning the surviving arm ACTUALLY runs, which is
    the substitution the whole finding is about: the driver used to advertise
    the G=16 self-test as the off-GPU check for a G=1 arm."""
    names = re.search(r"^ARM_NAMES=\(([^)]*)\)", TEXT, re.M).group(1).split()
    assert "bn_g1" not in names and "bn_g16" in names
    assert not re.search(r"^  bn_g1\s", run(["--list"]).stdout, re.M)
    advertised = lift('arm_offgpu_gates bn_g16', REPO=str(ROOT)).stdout
    # The COMMAND, not the paragraph beside it: the paragraph names --group-m 1
    # on purpose, to say what the dropped arm's gate did.
    command = re.match(r"[^(]*", advertised).group(0)
    assert "--group-m 16" in command, command
    assert "--group-m 1 " not in command, command
    # And the invocation and the gate agree on the pinning.
    body = CODE.split("say \"4. alpha_a", 1)[1].split("say \"5.", 1)[0]
    assert "--group-m 16" in body
    assert "--group-m 1 " not in body
    for pinning, want in ((["--group-m", "1"], 3), (["--group-m", "16"], 0)):
        done = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "bn_decomposition.py"),
             "--self-test", "--capability", "9.0", *pinning,
             "--reps", "17", "--plant-noise", "0.008"],
            capture_output=True, text=True, timeout=900, cwd=str(ROOT))
        assert done.returncode == want, (pinning, done.returncode, done.stdout[-2000:])
        assert exit_codes.classify_text(done.stdout) == want, pinning


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


def unadopting_repo(tmp_path, rel):
    """A repo whose `rel` does NOT import the module, for the caveat's loud
    branch. Written rather than picked from the tree on purpose: which real
    script is unadopted changes every time a slice lands (the anchor's adoption
    broke two tests that had named it), and a test of the MECHANISM must not
    depend on that. `adopts_exit_codes` reads the file, so a file is what this
    gives it."""
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def main():\n    raise SystemExit(2)\n")
    return tmp_path


def test_a_blocked_counter_route_is_not_reported_as_a_broken_instrument(tmp_path):
    """THE DEFECT, PLANTED. `dram_counter_route.py` ended `return 0 if verdict ==
    "OPEN" else 3`, so BLOCKED -- the answer that arm exists to obtain, and the
    expected one on a rented pod -- exited 3, which the adopted table reads as
    INVALID. The base driver carried `counter_plan) echo 0,3` for exactly this;
    R1 deleted the per-arm lists, so the row was wrong and the only honest thing
    this file could do was say so beside it.

    AGAINST A PLANTED FILE SINCE 2026-09-02, for the reason `unadopting_repo`
    gives: that script has since adopted the table and its BLOCKED verdict now
    exits DONE, so the real tree no longer reaches this branch. The MECHANISM is
    what is under test and it must not depend on which script is unadopted this
    week. `test_an_unadopted_refusal_says_two_may_mean_the_opposite` was moved
    the same way when the anchor adopted, one function below.
    """
    repo = unadopting_repo(tmp_path, "scripts/dram_counter_route.py")
    log = tmp_path / "counter_plan.log"
    log.write_text(RESULT_LOG)
    got = lift(f'summarize_arm counter_plan {log} INVALID', REPO=str(repo))
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
    repo = unadopting_repo(tmp_path, "scripts/memory_branch_anchor.py")
    log = tmp_path / "anchor.log"
    log.write_text("REFUSED: no calibration for this device.\n")
    got = lift(f'summarize_arm anchor_measure {log} REFUSED', REPO=str(repo))
    assert "REFUSED BEFORE MEASURING" in got.stdout
    assert "used to document 2" in got.stdout
    assert "eight-minute measurement" in got.stdout
    # THE SAME ARM AGAINST THE REAL TREE IS SILENT, because the anchor adopted
    # the table in this same rebuild. That is the caveat working, not missing.
    real = lift(f'summarize_arm anchor_measure {log} REFUSED', REPO=str(ROOT))
    assert "used to document 2" not in real.stdout


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
        "mma_switch\tDONE\t0\t400\t0\t/w.log\t\n"
        "ruler\tDONE\t0\t9\t0\t/z.log\t\n")
    repo = unadopting_repo(tmp_path, "scripts/memory_branch_anchor.py")
    unadopting_repo(tmp_path, "scripts/check_mma_path.sh")
    adopting_repo(tmp_path, "scripts/ruler_rebaseline.py")
    (repo / "scripts" / "dram_counter_route.py").write_text(
        "def main():\n    raise SystemExit(3)\n")
    got = lift(f'contract_disclosure {ledger}', REPO=str(repo))
    assert "THE ROWS WHOSE STATE MAY BE THE WRONG WORD" in got.stdout
    assert "counter_plan        INVALID" in got.stdout
    assert "anchor_measure      REFUSED" in got.stdout
    # A DONE row IS disclosed when its file speaks another table: 0 there can be
    # a run that scored no gate. It was skipped until 2026-09-02, along with the
    # 1 and the 4 that the two live non-adopters actually spend.
    assert "mma_switch          DONE" in got.stdout
    assert "ruler" not in got.stdout, "a row from a file that has adopted the " \
        "table has nothing to disclose, in any state"
    # AGAINST THE REAL TREE, NOTHING IN THIS LEDGER IS DISCLOSED ANY MORE, and
    # that is the all-clear rather than a gap. Two slices landed together here:
    # one widened the disclosure to every state a non-adopter can spend (the
    # planted branch above), the other made `dram_counter_route.py` adopt the
    # table, which is what empties this branch. The planted repo keeps the loud
    # path covered no matter which real file adopts next; this branch asserts
    # the driver says the all-clear in a sentence instead of going silent.
    # AGAINST THE REAL TREE the answer is whatever has actually adopted today,
    # so it is ASKED rather than asserted. Hardcoding it is what broke this test
    # twice: once when the anchor adopted, once when the counter route did.
    real = lift(f'contract_disclosure {ledger}', REPO=str(ROOT))
    for arm, script in (("counter_plan", "scripts/dram_counter_route.py"),
                        ("anchor_measure", "scripts/memory_branch_anchor.py"),
                        ("mma_switch", "scripts/check_mma_path.sh")):
        adopted = lift(f'adopts_exit_codes {script}; echo "rc=$?"',
                       REPO=str(ROOT)).stdout.strip().endswith("rc=0")
        if adopted:
            assert arm not in real.stdout, (
                f"{script} adopts the table, so {arm} has nothing to disclose")
        else:
            assert arm in real.stdout, (
                f"{script} does not adopt the table, so {arm} must be disclosed "
                "in whatever state it lands, including DONE")
    # And the all-clear appears exactly when nothing was disclosed. Asserted on
    # the SECTION HEADING plus "No row", not on the sentence: the sentence was
    # rewritten when the disclosure widened past REFUSED and INVALID, and this
    # assertion pinned the old one and failed on the widening.
    if "WRONG WORD" not in real.stdout:
        assert "THE EXIT-CODE CONTRACT" in real.stdout
        assert "No row" in real.stdout

    clean = tmp_path / "CLEAN.tsv"
    clean.write_text("arm\tstate\trc\tseconds\tdirty\tlog\tnote\n"
                     "ruler\tDONE\t0\t9\t0\t/z.log\t\n")
    ok = lift(f'contract_disclosure {clean}', REPO=str(ROOT))
    assert "THE EXIT-CODE CONTRACT" in ok.stdout
    assert "WRONG WORD" not in ok.stdout
    assert "No row in this session, in ANY state" in ok.stdout


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


def test_a_ledger_of_one_claim_fail_row_does_not_print_the_all_clear(tmp_path):
    """THE DEFECT, PLANTED, AND IT IS THE ALL-CLEAR ITSELF. The disclosure fired
    on REFUSED and INVALID rows only, and the two arms whose files have not
    adopted the table spend 1 and 4. So a ledger holding nothing but
    `mma_switch CLAIM_FAIL 1` -- check_mma_path.sh, whose own header says "1 a
    gate failed" and which spends 1 on a VALIDITY reading, on "no interpreter"
    and on "no .ptx", three different things -- walked every row, matched none
    and printed a POSITIVE assurance that every state word was correct, over the
    one row where it was not."""
    ledger = tmp_path / "ARMS.tsv"
    ledger.write_text("arm\tstate\trc\tseconds\tdirty\tlog\tnote\n"
                      "mma_switch\tCLAIM_FAIL\t1\t400\t0\t/x.log\t\n")
    # THE UNADOPTING FILE IS PLANTED, NOT NAMED. check_mma_path.sh adopted the
    # table on 2026-09-02 and this assertion broke on the driver being right,
    # which is the third time a test here pinned which real script had adopted.
    # What is under test is the all-clear's reachability, not the census.
    repo = unadopting_repo(tmp_path, "scripts/check_mma_path.sh")
    got = lift(f'contract_disclosure {ledger}', REPO=str(repo))
    assert "THE EXIT-CODE CONTRACT" not in got.stdout, got.stdout
    assert "No row in this session" not in got.stdout
    assert "THE ROWS WHOSE STATE MAY BE THE WRONG WORD" in got.stdout
    assert "mma_switch          CLAIM_FAIL" in got.stdout
    assert "check_mma_path.sh" in got.stdout


@pytest.mark.parametrize("state,rc,tell", [
    ("DONE", "0", "A check\n  that examined nothing reports no failures."),
    ("CLAIM_FAIL", "1", "1 is three things"),
    ("UNKNOWN", "1", "1 is three things"),
    ("REFUSED", "2", "used to document 2"),
    ("INVALID", "3", "returns 3 for"),
    ("RETRY", "4", "may spend such a code on a REGISTERED ANSWER"),
])
def test_every_state_a_non_adopting_file_can_produce_is_disclosed(
        tmp_path, state, rc, tell):
    """COVER THE STATES, NOT THE TWO THAT WERE NOTICED FIRST. A file that speaks
    another table can land in any of these, and each one means something
    different when it does: 0 can be a run that scored no gate, 1 can be a
    crash, 4 can be a registered answer. The caveat has a branch for each and
    the ledger walk stops filtering to REFUSED and INVALID."""
    repo = unadopting_repo(tmp_path, "scripts/check_mma_path.sh")
    got = lift(f'contract_caveat mma_switch {state}; echo "rc=$?"', REPO=str(repo))
    assert "CAVEAT: this row came from scripts/check_mma_path.sh" in got.stdout
    assert tell in got.stdout, got.stdout
    assert got.stdout.strip().endswith("rc=0")
    # THE SILENT BRANCH, for the same state: the day that file adopts the table
    # the caveat stops printing without anyone editing this driver.
    quiet = lift(f'contract_caveat mma_switch {state}; echo "rc=$?"',
                 REPO=str(adopting_repo(tmp_path / "adopted",
                                        "scripts/check_mma_path.sh")))
    assert quiet.stdout.strip() == "rc=1"
    ledger = tmp_path / "ARMS.tsv"
    ledger.write_text("arm\tstate\trc\tseconds\tdirty\tlog\tnote\n"
                      f"mma_switch\t{state}\t{rc}\t9\t0\t/x.log\t\n")
    walked = lift(f'contract_disclosure {ledger}', REPO=str(repo))
    assert f"mma_switch          {state}" in walked.stdout, walked.stdout


def test_the_disclosure_walks_no_header_row_and_no_unplanned_arm(tmp_path):
    """The widened filter must not widen onto rows that are not exit codes. The
    ledger's first line is its column names and `skip_arm` writes NOT_PLANNED
    with an rc of `-`; neither is a state a command returned, and a caveat about
    either would be a sentence about nothing."""
    ledger = tmp_path / "ARMS.tsv"
    ledger.write_text("arm\tstate\trc\tseconds\tdirty\tlog\tnote\n"
                      "mma_switch\tNOT_PLANNED\t-\t0\t0\t-\tno GPU\n")
    got = lift(f'contract_disclosure {ledger}', REPO=str(
        unadopting_repo(tmp_path, "scripts/check_mma_path.sh")))
    assert "THE ROWS WHOSE STATE MAY BE THE WRONG WORD" not in got.stdout
    assert "No row in this session, in ANY state" in got.stdout


# --------------------------------------------------------------------------
# 12. an exit 1 this file cannot read is not a result
# --------------------------------------------------------------------------

def test_an_exit_one_from_a_file_that_does_not_speak_the_table_is_not_latched(tmp_path):
    """THE MOST EXPENSIVE MISLABEL AVAILABLE HERE. Python exits 1 for any
    exception that escapes `main`, and only three of the eleven arm scripts
    install the ERROR(4) handler that would say so. Under the table 1 is
    CLAIM_FAIL -- "measured, the world disagreed, a RESULT" -- and CLAIM_FAIL is
    one of the three words `arm()` LATCHES, so a crashed arm would be skipped on
    every later run and its silence reported as a finding. The driver cannot fix
    those scripts from here; what it can do is decline to read a word out of a
    table the file never agreed to."""
    ledger = tmp_path / "ARMS.tsv"
    logs = tmp_path / "logs"
    logs.mkdir()
    repo = unadopting_repo(tmp_path, "scripts/check_mma_path.sh")
    got = lift('arm mma_switch bash -c "exit 1"',
               REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert got.returncode == 0, got.stderr
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert row[1] == "UNKNOWN", row
    assert row[2] == "1"
    assert "NOT latched" in got.stdout
    assert "1 is three things" in got.stdout
    # ...and NOT latched means exactly that: the arm is attempted again.
    again = lift('arm mma_switch bash -c "exit 1"',
                 REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                 DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert "SKIP mma_switch" not in again.stdout
    assert len(ledger.read_text().splitlines()) == 2


def test_an_exit_one_from_a_file_that_does_speak_the_table_is_still_a_result(tmp_path):
    """THE OTHER BRANCH, and the one that must not move. From a file that HAS
    adopted the module, 1 means what the table says: the experiment worked and a
    pre-registered claim did not hold. That is the most valuable outcome this
    study has, it is finished, and re-running it until it passes is the failure
    mode the module is named against."""
    ledger = tmp_path / "ARMS.tsv"
    logs = tmp_path / "logs"
    logs.mkdir()
    repo = adopting_repo(tmp_path, "scripts/check_mma_path.sh")
    got = lift('arm mma_switch bash -c "exit 1"',
               REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert got.returncode == 0, got.stderr
    assert "CAVEAT" not in got.stdout
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert row[1] == "CLAIM_FAIL", row
    again = lift('arm mma_switch bash -c "exit 1"',
                 REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                 DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert "SKIP mma_switch (already CLAIM_FAIL" in again.stdout


def test_the_unknown_state_is_the_drivers_word_and_not_a_second_table():
    """R1 deleted the per-arm done-code lists and this puts none back:
    `ledger_state` still turns every exit code into a word by itself, and the
    UNKNOWN branch is a REFUSAL TO READ one, taken after that function has
    spoken and only for a file that has not adopted the module."""
    body = CODE.split("\narm() {", 1)[1].split("\n}", 1)[0]
    assert 'state="$(ledger_state "$rc")"' in body
    assert '[[ "$state" == "CLAIM_FAIL" ]] && ! adopts_exit_codes' in body
    assert "UNKNOWN" not in exit_codes.LEDGER_STATES
    for rc in (0, 1, 2, 3, 4):
        assert lift(f'ledger_state {rc}', REPO=str(ROOT)).stdout.strip() == \
            exit_codes.ledger_state(rc)


# --------------------------------------------------------------------------
# 13. a plan that printed is not a plan that refused
# --------------------------------------------------------------------------

PLANNED_LOG = """\
=== plan: four models, twelve cells ===
  mixtral-8x7b  T=1024  BLOCK_M 128
  the cost of the whole arm is 35 minutes
NOT A RESULT. Nothing was measured.
  reason: --dry-run was given
"""

REFUSED_LOG = """\
REFUSED before any GPU time, from the pinned constants alone:
  BLOCK_M=256: the accumulator needs 256 registers per thread against 255.
"""


def test_a_plan_that_printed_and_then_refused_is_planned_and_one_that_did_not_is_not(tmp_path):
    """THE DISTINCTION DRY MODE EXISTS TO DRAW, AND THE PHASE-3 CLOSE ERASED IT.
    Once every script adopted the table its own --dry-run began exiting REFUSED,
    which is right -- a plan measured nothing and scored no gate -- and eleven of
    the sixteen planned arms landed on PLAN_REFUSED, so the ledger could no
    longer say whether there was a plan on the page. The log still says it: one
    of these two printed a plan and then said it measured nothing, the other
    refused on line 1 and never planned."""
    planned = tmp_path / "planned.log"
    planned.write_text(PLANNED_LOG)
    refused = tmp_path / "refused.log"
    refused.write_text(REFUSED_LOG)
    assert lift(f'dry_state 2 {planned}', REPO=str(ROOT)).stdout.strip() == "PLANNED"
    assert lift(f'dry_state 2 {refused}', REPO=str(ROOT)).stdout.strip() == "PLAN_REFUSED"
    # An empty log is a refusal that printed nothing, and a missing one is not a
    # plan either: "could not read it" is not "it planned".
    (tmp_path / "empty.log").write_text("\n   \n")
    assert lift(f'dry_state 2 {tmp_path}/empty.log',
                REPO=str(ROOT)).stdout.strip() == "PLAN_REFUSED"
    assert lift(f'dry_state 2 {tmp_path}/gone.log',
                REPO=str(ROOT)).stdout.strip() == "PLAN_REFUSED"
    # ...and rc 0 and a traceback are still decided by the code alone.
    assert lift(f'dry_state 0 {refused}', REPO=str(ROOT)).stdout.strip() == "PLANNED"
    assert lift(f'dry_state 1 {planned}', REPO=str(ROOT)).stdout.strip() == "BROKEN"


def test_the_dry_run_ledger_still_distinguishes_the_two(tmp_path):
    """The same distinction end to end, against the real scripts rather than a
    planted log: some arm has to reach each word, or the ledger has one word
    for two things again."""
    session = tmp_path / "s"
    got = run(["--dry-run"], session=session)
    assert got.returncode in (0, exit_codes.INVALID), got.stdout[-2000:]
    states = {r.split("\t")[0]: r.split("\t")[1] for r in
              (session / "ARMS-dryrun.tsv").read_text().splitlines()[1:]}
    assert "PLANNED" in states.values(), states
    assert "PLAN_REFUSED" in states.values(), states
    assert 'dry_state "$rc" "$log"' in CODE


# --------------------------------------------------------------------------
# 14. the ruler the arms read is the one this session measured
# --------------------------------------------------------------------------

def calibration_yaml(path: Path, utc: str) -> Path:
    """A calibration file with only the field the gate reads, plus a `detail:`
    block carrying a `utc:` of its own. The real yaml nests a hundred keys under
    `detail`, and a grep for `utc:` over the whole file would find whichever
    came first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name: TESTCARD (measured)\n"
                    "detail:\n  utc: '1999-01-01T00:00:00+00:00'\n"
                    f"provenance:\n  utc: '{utc}'\n  git_dirty: false\n")
    return path


def test_the_session_publishes_the_calibration_it_measures():
    """THE WORST ONE IN THE SET. Arm 0 ran `calibrate_hardware.py` bare. Phase
    2's A6 fix had moved that script's output to an untracked session path and
    made the copy into the tracked tree a separate `--publish` decision -- "the
    copy is what makes the ruler visible to roofline.load_measured() and
    therefore to the sweep", in its own header. So arm 0 spent three minutes
    measuring THIS card's ridge where nothing reads it, and every later arm
    resolved its ridge from whatever measured_<card>.yaml a PREVIOUS rental left
    in the checkout, labelled "measured on this machine". The audit's own "a
    constant from another machine presented as a measurement", recreated by the
    fix for a different one."""
    assert 'scripts/calibrate_hardware.py" --publish' in CODE
    assert "--publish IS THE ARM" in TEXT
    # And the gate that follows it, with a stop that is not advice. It asks BOTH
    # questions: when the file was written, and whether the arm that wrote it
    # passed its own gates. OK is the only word that proceeds.
    gate = CODE.split('scripts/calibrate_hardware.py" --publish', 1)[1] \
               .split("\nsay ", 1)[0]
    assert 'CALIB_STATE="$(calibration_state "$CALIB_YAML" "$SESSION_SINCE")"' in gate
    assert 'CALIB_ROW="$(ledger_arm_state calibrate)"' in gate
    assert 'CALIB_VERDICT="$(calibration_verdict "$CALIB_ROW" "$CALIB_STATE")"' in gate
    assert '[[ "$CALIB_VERDICT" == "OK" ]]' in gate
    assert 'exit "$RC_REFUSED"' in gate
    # and the four yaml states are still named, now by the refusal it calls.
    refusal = CODE.split("calibration_refusal() {", 1)[1].split("\n}\n", 1)[0]
    for word in ("MISSING", "UNDATED", "STALE", "NO_BASELINE"):
        assert word in refusal, word


def test_a_fresh_stamp_is_not_a_ruler_the_arm_stood_behind(tmp_path):
    """THE DEFECT THE FIX INTRODUCED, planted through `arm()`. The first version
    of this gate asked only WHEN the tracked yaml was measured, and
    calibrate_hardware.py copies that yaml into the tree at :683 BEFORE it
    scores a gate at :712 -- its own --help says so: "A calibration whose clock
    could not be established is INVALID rather than DONE ... the yaml is still
    written". So a calibrate that fails VALIDITY clock_established ("the samples
    disagree with the settle plateau; nothing normalised by the clock may be
    quoted") published a fresh-stamped ruler, calibration_state said PUBLISHED,
    the gate printed "measured in THIS session", and 3.4 hours of arms scored
    every roof fraction, LEVEL flag and alpha against a ruler arm 0 itself
    refused to stand behind. Worse on a resume: INVALID is LATCHED, so the bad
    calibrate is never re-run and the gate keeps saying PUBLISHED.

    The stand-in publishes and then exits 3, exactly as the real script does."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    yaml = repo / "moe" / "bench" / "hardware" / "measured_testcard.yaml"
    yaml.parent.mkdir(parents=True)
    fake = repo / "scripts" / "calibrate_hardware.py"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        f'printf "name: T\\nprovenance:\\n  utc: \'$(date -u '
        '+%Y-%m-%dT%H:%M:%S)+00:00\'\\n" > '
        f'"{yaml}"\n'
        f'echo "[calibrate] PUBLISHED to {yaml}"\n'
        "echo 'RESULT: VALIDITY clock_established FAIL samples disagree with the plateau'\n"
        "exit 3\n")
    logs = tmp_path / "logs"
    logs.mkdir()
    ledger = tmp_path / "ARMS.tsv"
    ledger.write_text("arm\tstate\trc\tseconds\tdirty\tlog\tnote\n")
    got = lift('SESSION_SINCE="$(date -u +%Y%m%d%H%M%S)"\n'
               f'arm calibrate bash {fake} --publish >/dev/null\n'
               'ROW="$(ledger_arm_state calibrate)"\n'
               f'STATE="$(calibration_state {yaml} "$SESSION_SINCE")"\n'
               'echo "row=$ROW state=$STATE"\n'
               'calibration_verdict "$ROW" "$STATE"; echo "rc=$?"',
               REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
    assert got.returncode == 0, got.stderr
    lines = got.stdout.strip().splitlines()
    # The yaml IS this session's by date, and the arm is still not one to trust.
    assert lines[-3] == "row=INVALID state=PUBLISHED", got.stdout
    assert lines[-2] == "ARM INVALID", got.stdout
    assert lines[-1] == "rc=1", got.stdout


@pytest.mark.parametrize("row,state,verdict,rc", [
    ("DONE", "PUBLISHED", "OK", 0),
    ("DONE", "STALE", "YAML STALE", 1),
    ("DONE", "UNDATED", "YAML UNDATED", 1),
    ("DONE", "MISSING", "YAML MISSING", 1),
    ("INVALID", "PUBLISHED", "ARM INVALID", 1),
    ("CLAIM_FAIL", "PUBLISHED", "ARM CLAIM_FAIL", 1),
    ("REFUSED", "PUBLISHED", "ARM REFUSED", 1),
    ("UNKNOWN", "PUBLISHED", "ARM UNKNOWN", 1),
    ("", "PUBLISHED", "ARM NO_ROW", 1),
    ("", "UNDATED", "ARM NO_ROW", 1),
])
def test_one_pair_of_words_proceeds_and_every_other_pair_refuses(row, state, verdict, rc):
    """BOTH HALVES, AND THE FAIL BRANCH OF EACH. DONE alone is a run that wrote
    a ruler somewhere; PUBLISHED alone is a file of unknown standing. Only the
    pair means "this card's ruler, from an instrument that passed its own
    gates". The empty row prints NO_ROW rather than nothing, because an empty
    word inside a refusal reads as a bug in the refusal."""
    got = lift(f'calibration_verdict {row!r} {state!r}; echo "rc=$?"', REPO=str(ROOT))
    assert got.stdout.split("\n")[0] == verdict, got.stdout
    assert got.stdout.strip().splitlines()[-1] == f"rc={rc}", got.stdout


def test_the_gate_is_not_scoped_to_only_and_the_refusal_says_which_flag(tmp_path):
    """THE GATE IS UNCONDITIONAL AND ARM 0 IS NOT. `arm calibrate` returns early
    through `wanted` when --only names other arms, while the gate runs anyway,
    because every arm resolves its ridge through roofline.load_measured() no
    matter which subset was asked for. The cost of keeping it unconditional is
    that the file's own documented invocation must name calibrate, so it does,
    and the refusal an operator will actually see says which flag left arm 0
    out instead of reporting UNDATED on a file this session never touched."""
    assert "--only calibrate,roofline-n256-g16,noise_floor" in TEXT
    ledger = tmp_path / "ARMS.tsv"
    ledger.write_text("arm\tstate\trc\tseconds\tdirty\tlog\tnote\n")
    got = lift('V="$(calibration_verdict "$(ledger_arm_state calibrate)" UNDATED)"\n'
               'echo "$V"\n'
               'calibration_refusal "$V" testcard /nowhere/measured_testcard.yaml UNDATED',
               REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(tmp_path), ONLY="occupancy",
               PY_BASE="/usr/bin/python3", SESSION_SINCE="20260902134501")
    assert got.returncode == 0, got.stderr
    assert got.stdout.splitlines()[0] == "ARM NO_ROW", got.stdout
    assert "--only occupancy left arm 0 out" in got.stdout, got.stdout
    assert "--only calibrate,occupancy" in got.stdout, got.stdout
    # and a session with no --only is not told to blame one.
    alone = lift('calibration_refusal "ARM NO_ROW" testcard /nowhere/y.yaml UNDATED',
                 REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(tmp_path), ONLY="",
                 PY_BASE="/usr/bin/python3", SESSION_SINCE="20260902134501")
    assert "left arm 0 out" not in alone.stdout, alone.stdout
    assert "no --only to explain it" in alone.stdout, alone.stdout


def test_a_refusal_this_gate_cannot_read_refuses_rather_than_proceeding():
    """REFUSE RATHER THAN DEFAULT, over the one input neither half issues. A
    verdict word this refusal does not know is a bug in the gate, and the two
    states it decides between are "this card's ruler" and "another machine's",
    so there is no side to guess."""
    got = lift('calibration_refusal "SOMETHING_ELSE" testcard /nowhere/y.yaml UNDATED',
               REPO=str(ROOT), LEDGER="/nowhere/ARMS.tsv", LOGS="/nowhere", ONLY="",
               PY_BASE="/usr/bin/python3", SESSION_SINCE="20260902134501")
    assert got.stdout.startswith("REFUSED:"), got.stdout
    assert "SOMETHING_ELSE" in got.stdout


@pytest.mark.parametrize("state,utc", [
    ("PUBLISHED", "2026-09-02T14:00:00+00:00"),
    ("PUBLISHED", "2026-09-02T13:45:01+00:00"),
    ("STALE", "2026-09-02T13:45:00+00:00"),
    ("STALE", "2026-08-14T09:00:00+00:00"),
])
def test_a_calibration_measured_before_this_session_is_stale(tmp_path, state, utc):
    """BOTH BRANCHES OF THE GATE, and the boundary is the session's own start:
    one second earlier is the last rental's ruler. The H200's dense bf16 moved
    7.1% between two sessions while its bandwidth held to 0.014%, so a carried
    ridge is wrong by more than most effects this study reports."""
    yaml = calibration_yaml(tmp_path / "measured_testcard.yaml", utc)
    got = lift(f'calibration_state {yaml} 20260902134501', REPO=str(ROOT))
    assert got.stdout.strip() == state, got.stderr


def test_a_calibration_that_cannot_say_when_it_was_measured_is_not_this_ones(tmp_path):
    """REFUSE RATHER THAN DEFAULT, over the three ways the answer can be absent.
    The committed moe/bench/hardware/measured_nvidia_h200.yaml is the UNDATED
    case in the tree today: it predates the provenance block, so it cannot name
    the rental that measured it, and "cannot say" is not "this one"."""
    got = lift(f'calibration_state {ROOT}/moe/bench/hardware/'
               'measured_nvidia_h200.yaml 20260902134501', REPO=str(ROOT))
    assert got.stdout.strip() == "UNDATED", got.stdout
    gone = lift(f'calibration_state {tmp_path}/nothing.yaml 20260902134501',
                REPO=str(ROOT))
    assert gone.stdout.strip() == "MISSING"
    # A date with no time cannot decide which of two same-day rentals wrote it.
    coarse = tmp_path / "coarse.yaml"
    coarse.write_text("provenance:\n  utc: '2026-09-02'\n")
    assert lift(f'calibration_state {coarse} 20260902134501',
                REPO=str(ROOT)).stdout.strip() == "UNDATED"
    # And a driver that could not stamp its own start says so rather than
    # comparing against an empty string, which every number is greater than.
    fresh = calibration_yaml(tmp_path / "fresh.yaml", "2026-09-02T14:00:00+00:00")
    assert lift(f'calibration_state {fresh} ""',
                REPO=str(ROOT)).stdout.strip() == "NO_BASELINE"


def test_a_calibrate_that_does_not_publish_leaves_the_gate_refusing(tmp_path):
    """THE FIX, PROVED THROUGH `arm()` WITH A STAND-IN CALIBRATE, because the
    consequence is not that the flag is absent from a line -- it is that the
    file the arms read is not the file this session wrote. Two fakes, identical
    but for whether they honour `--publish`. The one that ignores it is the
    shipped behaviour before this fix: the arm lands DONE, the tree is clean,
    and every arm after it would score against whatever was already there."""
    logs = tmp_path / "logs"
    logs.mkdir()
    for honours, expected in ((True, "PUBLISHED"), (False, "MISSING")):
        repo = tmp_path / ("honours" if honours else "ignores")
        (repo / "scripts").mkdir(parents=True)
        fake = repo / "scripts" / "calibrate_hardware.py"
        body = ('printf "name: T\\nprovenance:\\n  utc: \'$(date -u '
                '+%Y-%m-%dT%H:%M:%S)+00:00\'\\n" > "$out"\n'
                if honours else 'echo "[calibrate] wrote a session path only"\n')
        fake.write_text(
            '#!/usr/bin/env bash\n'
            'set -uo pipefail\n'
            f'out="{repo}/moe/bench/hardware/measured_testcard.yaml"\n'
            'mkdir -p "$(dirname "$out")"\n'
            f'[[ "${{1:-}}" == "--publish" ]] && {body}'
            'echo "[calibrate] done"\n')
        ledger = tmp_path / f"ARMS-{expected}.tsv"
        got = lift(f'SESSION_SINCE="$(date -u +%Y%m%d%H%M%S)"\n'
                   f'arm calibrate bash {fake} --publish\n'
                   f'calibration_state {repo}/moe/bench/hardware/'
                   'measured_testcard.yaml "$SESSION_SINCE"',
                   REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                   DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0)
        assert got.returncode == 0, got.stderr
        assert ledger.read_text().splitlines()[-1].split("\t")[1] == "DONE"
        assert got.stdout.strip().splitlines()[-1] == expected, got.stdout


def test_the_session_start_stamp_survives_a_resume_into_the_same_directory(tmp_path):
    """A RESUME MUST NOT REFUSE AN HOUR OF FINISHED ARMS. The baseline is the
    session directory's own UTC stamp, not this process's start, so a session
    restarted into `gaps-<card>-<stamp>` still counts the calibration its first
    pass published. A SESSION the operator named by hand carries no stamp and
    falls back to now, which refuses until calibrate runs again: three minutes,
    not the session."""
    assert 'SESSION_SINCE="$(utc_stamp "${SESSION##*-}")" || SESSION_SINCE=' in CODE
    got = lift('utc_stamp "${SESSION##*-}"',
               REPO=str(ROOT), SESSION="/workspace/session/gaps-nvidia_h200-20260902T134501Z")
    assert got.stdout.strip() == "20260902134501"
    handmade = lift('utc_stamp "${SESSION##*-}"; echo "rc=$?"',
                    REPO=str(ROOT), SESSION=str(tmp_path / "s"))
    assert handmade.stdout.strip() == "rc=1"


# --------------------------------------------------------------------------
# 15. the two span arms are two arms
# --------------------------------------------------------------------------

def test_the_two_span_arms_do_not_derive_one_run_id():
    """THE SECOND ARM MEASURED NOTHING AND LANDED DONE. `--densify` became the
    default (BooleanOptionalAction) while the driver still booked `span_dense`
    with it and `span` bare, so both densified, both hashed to the same plan,
    and the second restored every row the first had written, timed nothing and
    was recorded finished with 30 minutes booked against it. `--max-minutes`
    could not have separated them either: it prices a run rather than defining
    one and is deliberately out of the key. This runs both plans and compares
    the ids the script itself prints."""
    assert '--dry-run --no-densify' in CODE
    # AND NEITHER ARM IS TRUNCATED. --max-minutes does not refuse: at
    # span_extent_separation.py:4602 it breaks out of the cell loop, records
    # "stopped after N minutes with K of M cells done" as prose that no gate
    # reads, and the reduction then scores whichever cells finished to the same
    # exit code a complete grid gets. Checked over the INVOCATIONS, not the
    # whole file: the paragraph above them names the flag it removes.
    invocations = [ln for ln in CODE.splitlines()
                   if ln.lstrip().startswith("arm span")]
    assert len(invocations) == 4, invocations
    for line in invocations:
        assert "--max-minutes" not in line, line
    ids = {}
    for flag in ("--densify", "--no-densify"):
        done = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "span_extent_separation.py"),
             "--dry-run", flag], capture_output=True, text=True, timeout=600,
            cwd=str(ROOT))
        found = re.search(r"^run id (\S+)$", done.stdout, re.M)
        assert found, done.stdout[-2000:]
        ids[flag] = found.group(1)
    assert ids["--densify"] != ids["--no-densify"], ids
    assert "densifytrue" in ids["--densify"]
    assert "densifyfalse" in ids["--no-densify"]


def test_the_sparse_span_arm_says_it_expects_to_refuse():
    """It is booked on the published powers-of-two grid, where every expert
    holds a power-of-two number of rows and the padding factor is exactly 1.00
    at every point, so `c2_grid_power` refuses before spending a minute. That is
    the honest answer for that grid and it is free -- but an operator reading
    `span REFUSED` in the ledger has to find it stated somewhere, or it reads as
    a broken arm."""
    block = run(["--list"]).stdout.split("  span ", 1)[1].split("\n\n", 1)[0]
    assert "IT REFUSES" in block
    assert "--no-densify" in block
    # And it is booked at zero, because a refusal spends nothing and booking
    # thirty minutes for one hides that the answer is already in.
    assert lift('arm_minutes span', REPO=str(ROOT)).stdout.strip() == "0"


def test_the_counter_arm_says_to_read_its_verdict_and_not_its_ledger_state():
    listing = run(["--list"]).stdout
    block = listing.split("  counter_plan ", 1)[1].split("\n\n", 1)[0]
    assert "READ ITS VERDICT LINE, NOT ITS LEDGER STATE" in block
    assert "BLOCKED is the ANSWER" in block
