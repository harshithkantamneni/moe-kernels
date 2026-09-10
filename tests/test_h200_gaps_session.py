"""The unattended driver has to be checkable without a pod.

`scripts/h200_gaps_session.sh` spends about four and a half hours of rented GPU
across twenty arms, and near seven once its KERNEL rows are put on a wall clock.
Almost everything that can go wrong with it goes wrong SILENTLY and is only
visible an hour later: an arm marked finished having measured nothing, a
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
  9. AN ARM THAT IS IN NO SESSION CLOSES NOTHING. `grep -c alias_ablation
     scripts/h200_gaps_session.sh` returned 0, and so did the same grep over
     all 28 branches, while `scripts/alias_ablation.py` is the only instrument
     in the tree that tests whether the per-tile slope every alpha is built
     from IS DRAM traffic. The tests below pin that it is scheduled, that it is
     booked at what its own plan prints (which, since the alias slice made the
     plan page charge the probe, is the pod's own figure and no longer 1.4
     minutes under it), that its dry branch cannot measure, and that the two
     rental subsets the banner prints are priced by the same two functions as
     the table rather than by a second copy of it.
 10. THE SECOND OPINION IS TAKEN. `exit_codes.classify_text` was written so the
     driver could recompute a script's verdict from the RESULT lines it printed
     and compare it with the exit code, and until 2026-09-03 nothing in the
     driver called it: `arm` read the integer and the summary grepped the lines
     for display. A `RESULT: CLAIM C1 FAIL` page under exit 0 was latched DONE.
     `second_opinion` is one function with a printed table, every row of it is
     planted here through the shipped `arm()`, and the four proofs the review
     ran (a FAIL line under exit 0, no line under exit 0, all-PASS lines under
     exit 1, and a Python import-time crash under the real pod command lines
     with a broken torch on PYTHONPATH) are the tests below.
 11. THE LATCH IS REACHABLE. The ledger lives in one session directory; a plain
     re-invocation used to open a fresh one beside it and re-run every row the
     latch protects, and nothing in --help said SESSION= was the way back.
     `session_choice` decides the directory in one place, so every branch
     (named, resumed, new, refused, latest-exists) is planted off GPU.

HOW THE SHELL FUNCTIONS ARE TESTED. Sourcing the driver would run the session,
so the file marks a block of pure function definitions between
`# >>> LIFTABLE` and `# <<< LIFTABLE`, and `lift()` below evaluates that block
in a fresh bash. That asks the SHIPPED function rather than a Python copy of
its case statement, which would agree with it until it did not.
"""
from __future__ import annotations

import math
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
# alias_ablation was added on 2026-09-03: it was in no session on any branch,
# and it is the only arm that tests the relabelling of the per-tile slope as
# DRAM traffic that every alpha, cap and roof fraction in the study rests on.
# bn_g1 was dropped on 2026-09-03: at GROUP_SIZE_M=1 `bn_decomposition.py
# --self-test` exits 3 INVALID and the arm's own plan says C1 reads UNKNOWN
# however the data fall, and no other pinning that self-test was checked at
# passes either, so there was nowhere to re-pin it to.
#: NO ARM IS EXEMPT FROM THE PLAN-READING TESTS, and between 2026-09-09 and now
#: one was. The depth arm was booked `--partner-block-m 32`, a flag no version
#: of `scripts/bm128_depth.py` has ever defined, and a `PENDING_SCRIPT_FLAGS`
#: dict here let three plan-reading tests skip that arm by name until it landed.
#: It was never going to land: the BM=32 scaling partner is UNCONDITIONAL in the
#: script (`SMALL_TILE_BLOCK_M`, carried in `BLOCK_SIZES` beside the subject and
#: the reference), so the exemption would have stood for good while the
#: measuring branch exited 2 on argparse and the ledger filed the one arm this
#: session exists to rescue as REFUSED. The flag is gone from the driver, the
#: dict is gone from here, and every arm is read off its own plan again.


ARMS = ("calibrate", "pin_probe-n64-g1", "pin_probe-n256-g16",
        "roofline-n64-g1", "roofline-n256-g16", "roofline-n256-g32",
        "bm128_depth", "alias_ablation", "noise_floor",
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
    """A cost, the clock that cost is on, and what the arm closes. The clock is
    part of the declaration: a figure whose unit is not stated is what let the
    session total add wall minutes to kernel minutes and print one number."""
    listing = run(["--list"])
    assert listing.returncode == 0, listing.stderr
    for name in ARMS:
        assert re.search(rf"^  {re.escape(name)}\s+~\d+ min (WALL|KERNEL|ALLOW|FREE)$",
                         listing.stdout, re.M), name


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
    assert "THE CLAIM'S CONFIGURATION, and NO ARM CAN CONFIRM IT ON sm_90" in listing
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
    # The ablation runs BEFORE the floor and before every alpha arm. Its result
    # changes what bn_g16, the anchor, occupancy, cap_test and both span arms
    # are decomposing; the floor changes only how each of them is scored. The
    # two are independent in both directions, so the order between them is a
    # budget decision, and a 13-minute arm that can retire the mechanism
    # sentence does not sit behind a 120-minute one.
    assert order.index("bm128_depth") < order.index("alias_ablation")
    assert order.index("alias_ablation") < order.index("noise_floor")
    assert order.index("alias_ablation") < order.index("bn_g16")
    assert order.index("alias_ablation") < order.index("anchor_measure")
    assert order.index("alias_ablation") < order.index("span_dense")
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
    booked = {n: int(re.search(rf"^  {re.escape(n)}\s+~(\d+) min \w+$",
                               listing, re.M).group(1)) for n in ARMS}
    assert sum(booked.values()) == minutes
    # And every arm says the cumulative minute it starts at, which must be the
    # running sum of the arms above it and nothing else. The column is a RANGE
    # once a KERNEL arm is above it, priced start to bounded start; the priced
    # half is the one that has to be the running sum.
    running = 0
    for name in ARMS:
        found = re.search(
            rf"  {re.escape(name)}\s+~\s*\d+ min \w+\s+starts at "
            rf"~\s*(\d+)(?:-(\d+))? min", body)
        assert found, name
        assert int(found.group(1)) == running, (name, found.group(1), running)
        if found.group(2):
            assert int(found.group(2)) >= running, name
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


#: The seven arms booked at their plans' own "excluding compiles and
#: allocation" kernel time, and the two booked at a wall clock. Written out here
#: so that a row moving between the two groups has to be a deliberate edit to
#: this list; the test below does not trust it, it re-derives every membership
#: from the arm's own printed plan and then checks this list against what it
#: found.
KERNEL_ARMS = ("roofline-n64-g1", "bm128_depth", "bn_g16", "occupancy",
               "cap_test", "dtype", "span_dense")


def test_the_cost_column_names_the_clock_each_figure_is_on(tmp_path):
    """THE TOTAL ADDED TWO DIFFERENT UNITS. Fixing every figure to be the arm's
    own left them in mixed clocks and said so nowhere: noise_floor's 120 and
    anchor_measure's 5 are WALL, while seven arms are booked at what their plans
    call "the model's own timings, excluding compiles and allocation". The
    column names the clock on every row now.

    THE MEMBERSHIP IS NOT A LIST KEPT HERE OR THERE. `arm_clock` reads
    `arm_unpriced`, so there is one set in the driver rather than two that drift
    -- the defect this whole slice keeps meeting -- and this test re-derives the
    same answer a third way, from what each arm's own --dry-run PRINTS, which is
    the only source neither of them can quietly disagree with."""
    session = tmp_path / "s"
    got = run(["--dry-run"], session=session)
    assert got.returncode == 0, got.stdout[-3000:]
    logs = session / "logs"
    found = {"KERNEL": [], "WALL": [], "ALLOW": [], "FREE": []}
    for name in ARMS:
        clock = lift(f'arm_clock {shlex.quote(name)}', REPO=str(ROOT)).stdout.strip()
        assert clock in found, (name, clock)
        found[clock].append(name)
        minutes = int(lift(f'arm_minutes {shlex.quote(name)}',
                           REPO=str(ROOT)).stdout.strip())
        unpriced = lift(f'arm_unpriced {shlex.quote(name)}', REPO=str(ROOT)).stdout
        log = logs / f"{name}.log"
        plan = log.read_text() if log.exists() else ""
        if minutes == 0:
            # A refusal, or a re-score that times nothing. There is no clock to
            # name, and calling it WALL would file a refusal as a wall figure.
            assert clock == "FREE", (name, clock)
            continue
        assert clock != "FREE", (name, minutes)
        if unpriced.startswith("everything:"):
            assert clock == "ALLOW", (name, clock)
            # The whole meaning of ALLOW: there was no plan to read a figure off.
            assert "estimat" not in plan.lower(), (name, plan[-1500:])
            continue
        # Everything else is decided by the arm's own words, in its own plan.
        disclaims = ("excluding compiles and allocation" in plan
                     or "not the wall clock" in plan.lower())
        assert plan, f"{name} printed no plan to read the clock off"
        assert (clock == "KERNEL") == disclaims, (name, clock, disclaims)
    assert tuple(found["KERNEL"]) == KERNEL_ARMS, found["KERNEL"]
    assert "noise_floor" in found["WALL"] and "anchor_measure" in found["WALL"]
    assert found["ALLOW"] and found["FREE"]
    # And the word reaches both places an operator reads the table.
    body = got.stdout.split("WHAT THIS COMMITS YOU TO")[1].split("SESSION  card=")[0]
    listing = run(["--list"]).stdout
    for clock, names in found.items():
        for name in names:
            assert re.search(rf"  {re.escape(name)}\s+~\s*\d+ min {clock}\b",
                             body), (name, clock)
            assert re.search(rf"^  {re.escape(name)}\s+~\d+ min {clock}$",
                             listing, re.M), (name, clock)


def test_the_total_bounds_the_kernel_rows_instead_of_leaving_them_unbounded(tmp_path):
    """"BOOK ABOVE THAT AND NEVER AT IT" WITH NO NUMBER TO BOOK ABOVE. The old
    banner printed one total over mixed units, computed the "starts at" column
    -- the one thing a rental is sized with -- from that sum, and then declined
    to bound it. Declining was defensible: this repo has ONE measured
    wall-over-model datum, on one small arm, and multiplying every row by it
    would be an invented number wearing a measurement's clothes. Printing no
    bound at all was not: it left the mixed sum as the only figure on the page.

    Both numbers are printed now, the second is arithmetic over the first, and
    no per-arm figure is touched by it."""
    got = run(["--dry-run"], session=tmp_path / "s")
    body = got.stdout.split("WHAT THIS COMMITS YOU TO")[1].split("SESSION  card=")[0]
    priced = int(re.search(r"TOTAL ~(\d+) minutes", body).group(1))
    kernel = re.search(r"of which ~(\d+) are KERNEL minutes", body)
    assert kernel, body
    kernel_min = int(kernel.group(1))
    listing = run(["--list"]).stdout
    booked = {n: int(re.search(rf"^  {re.escape(n)}\s+~(\d+) min \w+$",
                               listing, re.M).group(1)) for n in ARMS}
    assert kernel_min == sum(booked[n] for n in KERNEL_ARMS)
    assert 0 < kernel_min < priced, (kernel_min, priced)
    bound = re.search(r"~(\d+) minutes \(~(\d+)h (\d+)m\), the same table", body)
    assert bound, body
    bounded = int(bound.group(1))
    assert bounded == int(bound.group(2)) * 60 + int(bound.group(3))
    pct = int(lift("wall_over_model_pct", REPO=str(ROOT)).stdout.strip())
    assert bounded == priced - kernel_min + -(-kernel_min * pct // 100)
    assert bounded > priced, (bounded, priced)
    # The factor is the one this repo measured, and it is named where it is
    # used rather than left as a bare 2.35.
    assert "127 s logged against 54 s modelled" in body
    assert abs(pct / 100 - 127 / 54) < 0.01, pct
    assert "ILLUSTRATION" in body
    # THE OTHER DIRECTION, which is what says the bound is arithmetic and not a
    # constant: with no KERNEL minutes in it, a total is not inflated at all.
    assert lift("bounded_minutes 249 0", REPO=str(ROOT)).stdout.strip() == "249"
    assert lift("bounded_minutes 0 0", REPO=str(ROOT)).stdout.strip() == "0"
    # It rounds UP, because the number exists to be booked above and never at.
    assert lift("bounded_minutes 1 1", REPO=str(ROOT)).stdout.strip() == "3"
    # And no row was multiplied by it: the printed column still sums to the
    # priced total, not to the bound.
    assert sum(booked.values()) == priced


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
    # No --dry-run and no --fail-on-gate in this one, and both absences are the
    # design: a BARE invocation is its plan, `--run` is what makes it measure,
    # and its exit code comes straight from `exit_codes.classify` over the gates
    # it scored, so there is no downgrade for a flag to close.
    "scripts/alias_ablation.py": ("--run", "--models", "--alias-extent",
                                  "--compute", "--replicates", "--probe",
                                  "--dot-fallback", "--card", "--synthetic"),
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
    # alias_ablation.py has no --dry-run flag: a BARE invocation is its plan and
    # --run is what makes it measure, so its two plan branches carry no flag to
    # exclude on. Where more than one line survives, the one that names --run
    # is the pod's. This used to assert one survivor and was only ever reached
    # for arms whose script defines a gate flag, which that one does not.
    if len(measuring) > 1:
        measuring = [ln for ln in measuring if "--run" in ln]
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
                           ("occupancy", "--fail-on-gate"),
                           ("dtype", "--card"),
                           ("alias_ablation", "--synthetic")):
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


def test_the_dtype_gate_needs_the_card_the_dry_run_branch_was_already_given():
    """THE SAME DEFECT ONE ARM OVER, INSIDE THE COMMIT THAT NAMED IT. The word
    that makes dtype_tile_confound.py plan off a GPU box went into this file's
    dtype DRY-RUN branch and not into `arm_offgpu_gates`, so the command an
    operator runs before renting -- `--self-test 2.033 --self-test-alpha 0.2` --
    exits 2 NoCardToLabel with ZERO RESULT lines in all three planted worlds, the
    same shape --fail-on-world closed on span: a check that examined nothing
    reporting no failures.

    With --card the three worlds SEPARATE, which is the whole point of planting
    them, and they separate on the C3 line rather than on the exit code: a
    self-test observes no config, so every validity gate but V0 reads UNKNOWN and
    all three exit 3 INVALID. classify_text agreeing with that 3 is what says the
    3 is the table's word and not a crash."""
    script = str(ROOT / "scripts" / "dtype_tile_confound.py")
    seen = {}
    for world, verdict, tilt in (("2.033", "PASS", "1.023"),
                                 ("2.400", "FAIL", "1.208"),
                                 ("1.000", "FAIL", "0.503")):
        bare = subprocess.run(
            [sys.executable, script, "--self-test", world,
             "--self-test-alpha", "0.2"],
            capture_output=True, text=True, timeout=900, cwd=str(ROOT))
        assert bare.returncode == exit_codes.REFUSED, world
        assert "RESULT: " not in bare.stdout, world
        assert "NoCardToLabel" in bare.stdout, world
        carded = subprocess.run(
            [sys.executable, script, "--self-test", world,
             "--self-test-alpha", "0.2", "--card", "NVIDIA H200"],
            capture_output=True, text=True, timeout=900, cwd=str(ROOT))
        assert carded.stdout.count("RESULT: ") >= 10, world
        assert exit_codes.classify_text(carded.stdout) == carded.returncode
        c3 = re.search(r"^RESULT: CLAIM C3 (PASS|FAIL) median matched tilt "
                       r"([\d.]+)", carded.stdout, re.M)
        assert c3, carded.stdout[-2000:]
        seen[world] = (c3.group(1), c3.group(2))
        assert c3.group(1) == verdict, (world, c3.group(0))
        assert c3.group(2) == tilt, (world, c3.group(0))
    # A gate that says the same thing in every world is not a gate.
    assert len(set(seen.values())) == 3, seen


def test_every_advertised_off_gpu_command_is_run_by_this_guard_and_scores():
    """THE ANTIDOTE TO A FIX LANDING AT ONE OF TWO CALL SITES, which is how
    dtype's off-GPU line kept refusing for nine lines' distance from its own
    remedy. This walks EVERY arm's advertised off-GPU line, extracts the command
    it prints, RUNS it, and asks the log what it scored. A line an operator is
    told to run before paying for a pod is either a command that examines
    something, or a sentence saying no off-GPU check exists; there is no third
    kind, and the failure both defects had is the third kind wearing the first
    one's clothes.

    A SCORING mode (--self-test, --audit, --rescore, --corpus-only) must print
    RESULT lines and exit through the shared table. A PLANNING mode (--dry-run)
    is exempt from RESULT lines and must still print a plan rather than a
    traceback."""
    import tempfile
    scored_arms, planning_arms, prose_arms = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name in ARMS:
            line = lift(f"arm_offgpu_gates {shlex.quote(name)}",
                        REPO=str(ROOT)).stdout.strip()
            assert line, f"{name} advertises nothing at all"
            if "scripts/" not in line:
                prose_arms.append(name)
                continue
            cmd = line[line.index("scripts/"):]
            for cut in (", ", " and ", "  (", "("):
                if cut in cmd:
                    cmd = cmd[:cmd.index(cut)]
            # `kernel|extent|neither` is one command shown three ways; run the
            # first, since the worlds are covered arm by arm above.
            cmd = re.sub(r"([^\s|]+)(?:\|[^\s|]+)+", r"\1", cmd)
            # `<a path outside the tree>` is a placeholder for a directory.
            cmd = re.sub(r"<[^>]*>", tmp, cmd)
            words = shlex.split(cmd)
            assert (ROOT / words[0]).exists(), (name, cmd)
            got = subprocess.run([sys.executable, *words], capture_output=True,
                                 text=True, timeout=900, cwd=str(ROOT))
            # --synthetic is alias_ablation.py's planted-world mode and is a
            # SCORING one: it generates timings from a stated law, runs every
            # gate on them and exits through the shared table. Leaving it out
            # would have sent that arm's advertised line down the PLANNING
            # branch, where the guard demands a --dry-run the script does not
            # have, and the one arm added to close a missing-check defect would
            # have been checked by the wrong half of the guard.
            scoring = {"--self-test", "--audit", "--rescore", "--corpus-only",
                       "--synthetic"}
            if scoring & set(words):
                scored_arms.append(name)
                assert got.stdout.count("RESULT: ") > 0, (name, cmd, got.stdout[-2000:])
                assert exit_codes.classify_text(got.stdout) == got.returncode, (
                    name, cmd, got.returncode)
            else:
                planning_arms.append(name)
                assert "--dry-run" in words, (name, cmd)
                assert got.stdout.strip(), (name, cmd)
                assert "Traceback" not in got.stdout + got.stderr, (name, cmd)
    # The three groups are all non-empty, so none of the branches above is dead,
    # and every arm landed in exactly one of them.
    assert scored_arms and planning_arms and prose_arms
    assert len(scored_arms) + len(planning_arms) + len(prose_arms) == len(ARMS)
    assert "dtype" in scored_arms, scored_arms
    # An arm that advertises no command must say so in words rather than by
    # printing an empty line -- and it must not name a SCORING flag while
    # naming no program to run it with. That was the old dtype line's exact
    # shape ("C3 by --self-test 2.033|2.400|1.000"): an instruction to score
    # three planted worlds, with nothing for this guard to run and nothing for
    # the operator to run either without reconstructing the command.
    assert "alias_ablation" in scored_arms, scored_arms
    for name in prose_arms:
        line = lift(f"arm_offgpu_gates {shlex.quote(name)}",
                    REPO=str(ROOT)).stdout
        assert len(line.split()) >= 4, (name, line)
        for flag in ("--self-test", "--audit", "--rescore", "--corpus-only",
                     "--fail-on"):
            assert flag not in line, (name, flag, line)


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


@pytest.mark.parametrize("arm_name", ARMS)
def test_no_arm_is_given_a_flag_its_own_script_does_not_define(arm_name):
    """WHAT THE PENDING-FLAG EXEMPTION WAS HIDING. On 2026-09-09 both depth arm
    lines were written `--partner-block-m 32`, a flag no version of
    `scripts/bm128_depth.py` defines: the measuring branch would have exited 2
    on argparse and the arm this session exists to rescue would have been filed
    REFUSED having spent nothing. `INVOKED` above did not catch it because it is
    a hand-kept list of flags per script and that flag was simply not in it, so
    it guards against a RENAMED flag and not against an INVENTED one.

    This asks the question the other way round: every flag the driver actually
    writes on an `arm` line has to appear in the file that line runs. Derived
    from the driver's own text, so it cannot go stale against it."""
    rel = lift(f"arm_script {shlex.quote(arm_name)}", REPO=str(ROOT)).stdout.strip()
    if not rel or not (ROOT / rel).exists():
        pytest.skip(f"{arm_name} runs no file under this repo")
    source = (ROOT / rel).read_text()
    joined = re.sub(r"\\\n\s+", " ", CODE)
    lines = [ln for ln in joined.splitlines()
             if re.match(rf"\s*arm {re.escape(arm_name)}\s", ln)]
    assert lines, arm_name
    flags = sorted({w for ln in lines for w in shlex.split(ln)
                    if w.startswith("--")})
    assert flags, (arm_name, lines)
    for flag in flags:
        assert flag in source, (
            f"{arm_name} is given {flag}, which {rel} does not define: on the "
            "pod that is an argparse exit 2 and an arm that measured nothing")


def test_a_command_line_refusal_is_reported_as_one_and_not_as_a_blank_reason(tmp_path):
    """The other half of the usage-error fix. `summarize_arm` greps a refused
    log for the word REFUSED, which argparse never prints, so an arm refused by
    its own command line printed the heading "REFUSED BEFORE MEASURING" and
    then nothing at all, sending the reader to the log to find one line on
    stderr. It now prints that line and says whose fault it is: a flag this
    driver passed that the script does not define is not a refusal the script
    chose."""
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "bm128_depth.log"
    log.write_text(
        "usage: bm128_depth.py [-h] [--dry-run] [--r-max R_MAX]\n"
        "bm128_depth.py: error: unrecognized arguments: --partner-block-m 32\n")
    got = lift(f'summarize_arm bm128_depth {log} PLAN_REFUSED', REPO=str(ROOT),
               LEDGER=str(tmp_path / "ARMS.tsv"), LOGS=str(logs))
    assert "REFUSED BY ITS OWN COMMAND LINE" in got.stdout, got.stdout
    assert "unrecognized arguments: --partner-block-m 32" in got.stdout
    assert "not a refusal the script chose" in got.stdout
    # A script's own refusal still reads the way it did.
    own = logs / "span.log"
    own.write_text("plan line\nREFUSED: grid too sparse for C2\n")
    got = lift(f'summarize_arm span {own} PLAN_REFUSED', REPO=str(ROOT),
               LEDGER=str(tmp_path / "ARMS.tsv"), LOGS=str(logs))
    assert "REFUSED BEFORE MEASURING" in got.stdout, got.stdout
    assert "grid too sparse for C2" in got.stdout


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
               DRY=1, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
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
        DRY=1, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
    assert counted.returncode == 3, counted.stdout
    # ...and the shipped script contains that same branch, not a variant.
    assert "if (( DRY )) && (( BROKEN_ARMS > 0 )); then" in CODE
    # The PASS branch: a plan that works exits 0 and counts no BROKEN arm.
    stub.write_text("print('a plan')\n")
    clean = lift(
        f'arm stub {sys.executable} {stub}\n'
        'if (( DRY )) && (( BROKEN_ARMS > 0 )); then exit 3; fi\nexit 0',
        REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
        DRY=1, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
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
    # The stub NAMES the gate that failed, as a real INVALID does. A bare
    # `exit 3` used to stand here; since the second opinion an INVALID that
    # scored no gate is UNEARNED and not latched, which the last block below
    # plants as the FAIL branch of this same test.
    invalid = 'arm stub bash -c "echo RESULT: VALIDITY M0 FAIL stream check; exit 3"'
    first = lift(invalid,
                 REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                 DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
    assert first.returncode == 0, first.stderr
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert row[1] == "INVALID"
    assert "Do NOT re-run and do NOT quote it" in first.stdout
    # A second pass over the same ledger must not spend the arm again.
    again = lift(invalid,
                 REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                 DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
    assert "SKIP stub (already INVALID" in again.stdout
    assert len(ledger.read_text().splitlines()) == 1
    # UNEARNED: exit 3 over a page with no scored gate names no failed VALIDITY
    # gate, so there is nothing to latch. UNKNOWN, and re-attempted.
    bare = tmp_path / "BARE.tsv"
    for _ in range(2):
        got = lift('arm stub bash -c "exit 3"',
                   REPO=str(ROOT), LEDGER=str(bare), LOGS=str(logs), ONLY="",
                   DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
        assert "SKIP stub" not in got.stdout
        assert "UNEARNED INVALID" in got.stdout
    rows = [r.split("\t") for r in bare.read_text().splitlines()]
    assert [r[1] for r in rows] == ["UNKNOWN", "UNKNOWN"], rows


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
                   DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
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
               DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
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
                        ONLY="", DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
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
    # And the invocation and the gate agree on the pinning. ANCHORED ON THE
    # HEADING'S WORDS AND NOT ON ITS NUMBER: this read `say "4. alpha_a` until
    # 2026-09-03, when inserting alias_ablation at 3 renumbered every heading
    # below it and the split raised IndexError. A section number is a position
    # in the read order, which is the one thing about this file that is meant
    # to change; the sentence is what identifies the arm.
    body = CODE.split("alpha_a and alpha_b separated", 1)[1].split("\nsay ", 1)[0]
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

def plant_exit_codes(repo: Path) -> Path:
    """Make `moe.bench.exit_codes` importable from a planted repo.

    `arm()` takes its second opinion by running `exit_codes.classify_text` over
    the log through $PY_BASE with REPO on sys.path, so a planted repo with no
    such module makes every row UNREADABLE and therefore UNKNOWN, which is the
    refusal working and not the branch these tests are planting. The one file
    is linked rather than the package copied: the point is that the driver and
    the scripts read one line format from one module, and a copy would be a
    second one."""
    target = repo / "moe" / "bench" / "exit_codes.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Regular packages, not namespace portions: the venv has the real `moe`
    # installed, and a namespace portion at sys.path[0] does not shadow an
    # installed regular package, so without these two files the driver would
    # read the installed module and the planting would be decorative.
    for init in (repo / "moe" / "__init__.py", repo / "moe" / "bench" / "__init__.py"):
        if not init.exists():
            init.write_text("")
    if not target.exists():
        target.symlink_to(ROOT / "moe" / "bench" / "exit_codes.py")
    return repo


def adopting_repo(tmp_path, rel):
    """A repo whose `rel` imports the module, for the caveat's silent branch."""
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("from moe.bench.exit_codes import classify\n")
    return plant_exit_codes(tmp_path)


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
    return plant_exit_codes(tmp_path)


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
    # Told as history: that script adopted the table on 2026-09-02 and exits
    # DONE on BLOCKED, and a caveat that said "returns 3" in the present tense
    # was the recurring defect (a description of the old behaviour standing at
    # a second site) inside the function that exists to disclose it.
    assert "dram_counter_route.py used to return 3 for" in got.stdout
    assert "dram_counter_route.py returns 3 for" not in got.stdout
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
    ("INVALID", "3", "used to return 3 for"),
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
               DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
    assert got.returncode == 0, got.stderr
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert row[1] == "UNKNOWN", row
    assert row[2] == "1"
    assert "NOT latched" in got.stdout
    assert "1 is three things" in got.stdout
    # ...and NOT latched means exactly that: the arm is attempted again.
    again = lift('arm mma_switch bash -c "exit 1"',
                 REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                 DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
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
    # The stub prints the gate it failed, as an adopting script does: one
    # RESULT line per scored gate, then `classify`. A bare `exit 1` stood here
    # until the second opinion, and from an adopting file a 1 with no gate line
    # is a crash before the first gate, which is the test after this one.
    refuted = 'arm mma_switch bash -c "echo RESULT: CLAIM S6a FAIL observed none; exit 1"'
    got = lift(refuted,
               REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
    assert got.returncode == 0, got.stderr
    assert "CAVEAT" not in got.stdout
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert row[1] == "CLAIM_FAIL", row
    assert row[6].startswith("log agrees"), row
    again = lift(refuted,
                 REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                 DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
    assert "SKIP mma_switch (already CLAIM_FAIL" in again.stdout


def test_the_unknown_state_is_the_drivers_word_and_not_a_second_table():
    """R1 deleted the per-arm done-code lists and this puts none back:
    `ledger_state` still turns every exit code into a word by itself, and the
    UNKNOWN branch is a REFUSAL TO READ one, taken after that function has
    spoken and only for a file that has not adopted the module."""
    body = CODE.split("\narm() {", 1)[1].split("\n}", 1)[0]
    assert 'state="$(ledger_state "$rc")"' in body
    # The adoption question is asked ONCE, in `arm`, and handed to
    # `second_opinion`, which holds the non-adopter branch. Until 2026-09-03 it
    # was an inline demotion here and the only second look the row got.
    assert 'adopts_exit_codes "$(arm_script "$name")" || adopts=$?' in body
    assert 'second_opinion "$rc" "$state" "$(log_verdict "$log")" "$adopts"' in body
    opinion = CODE.split("\nsecond_opinion() {", 1)[1].split("\n}", 1)[0]
    assert "has not adopted moe/bench/exit_codes" in opinion
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
    # AN ARGPARSE USAGE ERROR IS NOT A PLAN, found on 2026-09-09 by handing an
    # arm a flag its script had not landed yet: argparse writes a usage block
    # and one error line to stderr, the driver captures both into the log, and
    # every line of that block counted as a line of plan printed before a
    # refusal marker that never comes, so the arm was filed PLANNED. The
    # signature is argparse's own error line, which no plan prints.
    usage = tmp_path / "usage.log"
    usage.write_text(
        "usage: bm128_depth.py [-h] [--dry-run] [--model MODEL] [--r-max R_MAX]\n"
        "                      [--reps REPS] [--group-m GROUP_M]\n"
        "bm128_depth.py: error: unrecognized arguments: --partner-block-m 32\n")
    assert lift(f'dry_state 2 {usage}', REPO=str(ROOT)).stdout.strip() == "PLAN_REFUSED"
    # And a plan that merely contains the word "error" in prose is still a plan.
    prose = tmp_path / "prose.log"
    prose.write_text(PLANNED_LOG.replace("\n", "\n", 1)
                     + "\n  the estimator error is 1.5%\n")
    assert lift(f'dry_state 2 {prose}', REPO=str(ROOT)).stdout.strip() == "PLANNED"


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
    plant_exit_codes(repo)
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
               DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
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


def test_the_claim_fail_refusal_reads_the_page_instead_of_naming_the_pin_rate(tmp_path):
    """THE PIN-RATE GATE WAS NAMED FOR ALL FOUR OF THEM, and backwards.
    `calibrate_hardware.py` scores four CLAIM gates
    (no_pattern_exceeds_the_pin_rate, write_rate_is_a_store_rate,
    clock_steady_across_patterns, not_throttled). Until 2026-09-09 this refusal
    told the operator, whichever had failed, that "no access pattern reached the
    pin rate", which is the wrong gate three times in four, and a
    misdescription of the fourth: that gate fails when a pattern EXCEEDS the
    derived pin rate. The refusal now reads the arm's own RESULT lines."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "calibrate.log").write_text(
        "RESULT: VALIDITY clock_established PASS the samples agree\n"
        "RESULT: CLAIM no_pattern_exceeds_the_pin_rate PASS write 4680.2 GB/s\n"
        "RESULT: CLAIM clock_steady_across_patterns FAIL two clock states: "
        "1980 -> 1650 MHz\n")
    ledger = tmp_path / "ARMS.tsv"
    ledger.write_text("calibrate\tCLAIM_FAIL\t1\t31\t1\t"
                      f"{logs / 'calibrate.log'}\tlog agrees\n")
    got = lift('calibration_refusal "ARM CLAIM_FAIL" testcard /nowhere/y.yaml MISSING',
               REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               PY_BASE=sys.executable, SESSION_SINCE="20260902134501")
    assert "RESULT: CLAIM clock_steady_across_patterns FAIL" in got.stdout, got.stdout
    assert "1980 -> 1650 MHz" in got.stdout
    # The gate that PASSED is not reported as the failure, and the old sentence
    # is gone from the driver entirely.
    assert "no access pattern reached the" not in TEXT
    assert "no_pattern_exceeds_the_pin_rate PASS" not in got.stdout
    # A page with no failing CLAIM line is a disagreement, and says so rather
    # than inventing a gate.
    (logs / "calibrate.log").write_text(
        "RESULT: VALIDITY clock_established PASS the samples agree\n")
    got = lift('calibration_refusal "ARM CLAIM_FAIL" testcard /nowhere/y.yaml MISSING',
               REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               PY_BASE=sys.executable, SESSION_SINCE="20260902134501")
    assert "holds no failing CLAIM line" in got.stdout, got.stdout
    # UNKNOWN counts: classify maps an UNKNOWN CLAIM to CLAIM_FAIL, so an arm
    # can wear the word with no FAIL line on its page.
    (logs / "calibrate.log").write_text(
        "RESULT: CLAIM not_throttled UNKNOWN no reference clock\n")
    got = lift('calibration_refusal "ARM CLAIM_FAIL" testcard /nowhere/y.yaml MISSING',
               REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               PY_BASE=sys.executable, SESSION_SINCE="20260902134501")
    assert "not_throttled UNKNOWN" in got.stdout, got.stdout


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

    Every case is PLANTED here. Until 2026-09-09 the UNDATED case was read off
    the committed `measured_nvidia_h200.yaml`, which predated the provenance
    block; the 2026-09-09 calibration carries `provenance.utc`, so that
    assertion became a statement about a file that had been replaced. A
    yaml with no provenance is planted instead, and the committed one is
    asserted to be what a yaml that CAN say when it was measured looks
    like."""
    undated = tmp_path / "undated.yaml"
    undated.write_text("name: testcard (measured)\nmemory:\n  bandwidth_tb_s: 4.0\n")
    got = lift(f'calibration_state {undated} 20260902134501', REPO=str(ROOT))
    assert got.stdout.strip() == "UNDATED", got.stdout
    # And the committed ruler is dated, so it is decided on its date and not
    # on "cannot say": measured 2026-09-09, so a session started before that
    # sees it as this rental's and one started after sees it as STALE.
    h200 = f"{ROOT}/moe/bench/hardware/measured_nvidia_h200.yaml"
    assert lift(f'calibration_state {h200} 20260909000000',
                REPO=str(ROOT)).stdout.strip() == "PUBLISHED"
    assert lift(f'calibration_state {h200} 20260910000000',
                REPO=str(ROOT)).stdout.strip() == "STALE"
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
        plant_exit_codes(repo)
        fake = repo / "scripts" / "calibrate_hardware.py"
        body = ('printf "name: T\\nprovenance:\\n  utc: \'$(date -u '
                '+%Y-%m-%dT%H:%M:%S)+00:00\'\\n" > "$out"\n'
                if honours else 'echo "[calibrate] wrote a session path only"\n')
        # Both fakes score a gate, as the real calibrate does before it exits
        # 0: DONE has to be EARNED by a RESULT line or the second opinion
        # records UNKNOWN, and what this test plants is the publish decision,
        # not the scoring.
        fake.write_text(
            '#!/usr/bin/env bash\n'
            'set -uo pipefail\n'
            f'out="{repo}/moe/bench/hardware/measured_testcard.yaml"\n'
            'mkdir -p "$(dirname "$out")"\n'
            f'[[ "${{1:-}}" == "--publish" ]] && {body}'
            'echo "RESULT: VALIDITY clock_established PASS plateau held"\n'
            'echo "[calibrate] done"\n')
        ledger = tmp_path / f"ARMS-{expected}.tsv"
        got = lift(f'SESSION_SINCE="$(date -u +%Y%m%d%H%M%S)"\n'
                   f'arm calibrate bash {fake} --publish\n'
                   f'calibration_state {repo}/moe/bench/hardware/'
                   'measured_testcard.yaml "$SESSION_SINCE"',
                   REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
                   DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
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


def test_the_counter_arm_says_its_ledger_word_is_earned_and_blocked_is_an_answer():
    """Until 2026-09-03 the row told the operator to read the probe's VERDICT
    line and distrust its ledger state, because the route exited 0 for OPEN
    and BLOCKED alike and printed no RESULT line. The gates slice made it
    score one gate per verdict and exit through the table, so the ledger word
    is earned and the row must say THAT, and must no longer send the operator
    around the second opinion. BLOCKED being the answer is unchanged."""
    listing = run(["--list"]).stdout
    block = listing.split("  counter_plan ", 1)[1].split("\n\n", 1)[0]
    assert "READ ITS RESULT LINES, AND THE LEDGER WORD IS EARNED" in block
    assert "READ ITS VERDICT LINE, NOT ITS LEDGER STATE" not in block
    assert "UNEARNED DONE" not in block
    assert "BLOCKED is the ANSWER" in block


# --------------------------------------------------------------------------
# 12. the alias ablation, and what a rental of a given length reaches
# --------------------------------------------------------------------------

#: The command whose plan the alias row is booked from. It carries --run, the
#: pod's own line; off a GPU box it prints the table and then refuses at the
#: probe having measured nothing. `report_cost` used to charge the probe's six
#: specialisations only under --run, and the test that runs this line now pins
#: that the bare plan prints the SAME figure. Every test below that runs it is
#: skipped on a machine with a CUDA device, because there the same command IS
#: the arm and would spend thirteen minutes of somebody's card.
#: The pod's own command, which since 2026-09-09 books `--dot-fallback refuse`:
#: the allow branch was bought once, on 2026-09-09, and returned P1 UNKNOWN
#: after 308 s. The fallback does not move the plan's figure, because both
#: price the whole ladder; what it changes is the cost of a probe miss.
ALIAS_POD_PLAN = (
    "scripts/alias_ablation.py --card 'NVIDIA H200' "
    "--models mixtral-8x7b,qwen2-57b-a14b,deepseek-v2-lite,deepseek-v3 "
    "--alias-extent block --compute sum --replicates 9 --probe "
    "--dot-fallback refuse --run")


def _no_cuda():
    try:
        import torch
    except Exception:                                             # noqa: BLE001
        return True
    try:
        return not torch.cuda.is_available()
    except Exception:                                             # noqa: BLE001
        return True


def _wall_minutes(stdout):
    found = re.search(r"^  WALL\s+([\d.]+) min", stdout, re.M)
    assert found, stdout[-2500:]
    return float(found.group(1))


def test_the_arm_that_tests_the_first_inferential_link_is_in_the_session():
    """THE DEFECT THIS SECTION EXISTS FOR, and it was an absence rather than a
    wrong number: `grep -c alias_ablation scripts/h200_gaps_session.sh` returned
    0, and the same grep over all 28 branches of this repository returned 0 on
    every one of them.

    Every alpha in this study is a slope per extra M-tile RELABELLED as a
    fraction of a fresh DRAM weight read. Every cap, every roof fraction and the
    sentence about a decode-configured kernel never reaching its compute roof is
    that relabelling carried forward, and `scripts/alias_ablation.py` is the only
    instrument in the tree that measures the same quantity without the byte
    model that does the relabelling. An arm in no session closes nothing."""
    assert "alias_ablation" in ARMS
    assert "scripts/alias_ablation.py" in TEXT
    rel = lift("arm_script alias_ablation", REPO=str(ROOT)).stdout.strip()
    assert rel == "scripts/alias_ablation.py"
    # It speaks the one exit-code table, so no row of its own can reach the
    # disclosure that says the state word may be a translation nobody agreed to.
    adopts = lift("adopts_exit_codes scripts/alias_ablation.py", REPO=str(ROOT))
    assert adopts.returncode == 0
    caveat = lift("contract_caveat alias_ablation INVALID", REPO=str(ROOT))
    assert caveat.returncode == 1 and not caveat.stdout.strip()
    # And what it closes states BOTH outcomes against the mechanism sentence,
    # plus the third state that is not an outcome at all.
    closes = lift("arm_closes alias_ablation", REPO=str(ROOT)).stdout
    assert "P1 PASS" in closes and "P1 FAIL" in closes
    assert "THE THIRD STATE IS NOT AN OUTCOME" in closes
    assert "0.529-0.588" in closes
    # The dependency on arm 0, and the direction that makes it expensive.
    assert "ARM 0" in closes and "DOES NOT REFUSE WITHOUT IT" in closes


@pytest.mark.skipif(not _no_cuda(), reason="the booking command measures on a GPU")
def test_the_alias_arm_is_booked_at_what_its_plan_prints_for_the_POD():
    """THE PLAN THE DRY BRANCH PREVIEWS IS THE POD'S OWN FIGURE. That script
    does not take a --dry-run flag at all: a bare invocation is its plan and
    `--run` is what makes it measure. Until 2026-09-03 `report_cost` charged
    the probe's six Triton specialisations only under `--run`, so the bare plan
    said WALL 11.6 while the pod spent WALL 13.0, and this test pinned that GAP
    as a disclosed fact. The alias slice then closed it from the script's side:
    the plan page charges the probe too, because an operator books a pod before
    they have one and the plan is the only page they can read. This test broke
    on the closing, which is the ninth time in this rebuild a fix at one site
    left a description of the old behaviour standing at another. It now pins
    the agreement: bare plan, pod figure and booking are one number.

    The dry branch is still bare, deliberately: a --dry-run carrying --run would
    MEASURE on a pod, and this file's rule is that a plan is free in every
    sense. Nothing needs disclosing because nothing differs."""
    pod = subprocess.run([sys.executable, *shlex.split(ALIAS_POD_PLAN)],
                         capture_output=True, text=True, timeout=900,
                         cwd=str(ROOT))
    assert pod.returncode == exit_codes.REFUSED, pod.stdout[-2000:]
    assert "BOOK THIS ONE" in pod.stdout
    pod_wall = _wall_minutes(pod.stdout)
    plan_words = [w for w in shlex.split(ALIAS_POD_PLAN) if w != "--run"]
    plan = subprocess.run([sys.executable, *plan_words], capture_output=True,
                          text=True, timeout=900, cwd=str(ROOT))
    plan_wall = _wall_minutes(plan.stdout)
    assert plan_wall == pod_wall, (plan_wall, pod_wall)
    booked = int(lift("arm_minutes alias_ablation", REPO=str(ROOT)).stdout.strip())
    assert booked == math.ceil(pod_wall), (booked, pod_wall)
    assert booked == math.ceil(plan_wall), (booked, plan_wall)
    # The row says where the figure came from, names the flag that separates the
    # two, and warns that on a GPU box the same command is the arm.
    basis = lift("arm_basis alias_ablation", REPO=str(ROOT)).stdout
    assert "--run" in basis and f"{pod_wall:.1f}" in basis, (basis, pod_wall)
    assert "11.6 is what the dry branch" not in basis, (
        "arm_basis still describes the gap the alias slice closed")
    assert "the two agree" in basis
    assert "on a box with no GPU" in basis
    # And the row says which fallback the figure was read at, because the two
    # cost the same to plan and different amounts to MISS.
    assert "--dot-fallback refuse" in basis, basis
    # WALL, not KERNEL: its plan charges the probe's compiles outright rather
    # than leaving them to the ratio, and says BOOK THIS ONE beside the figure.
    assert lift("arm_clock alias_ablation", REPO=str(ROOT)).stdout.strip() == "WALL"
    assert not lift("arm_unpriced alias_ablation", REPO=str(ROOT)).stdout.strip()


def test_no_dry_branch_of_the_alias_arm_can_measure():
    """A --dry-run must be free in every sense, and this is the one arm where
    the flag that would make its preview exact is also the flag that makes it
    spend a card. Both plan branches are therefore bare, and the pod branch is
    the only line in the file that gives that script --run."""
    joined = re.sub(r"\\\n\s+", " ", CODE)
    lines = [ln for ln in joined.splitlines()
             if re.match(r"\s*arm alias_ablation\s", ln)]
    assert len(lines) == 3, lines
    with_run = [ln for ln in lines if "--run" in ln]
    assert len(with_run) == 1, lines
    assert "PY_VLLM" in with_run[0], with_run[0]
    # Off a GPU box the plan is given a NAMED HYPOTHETICAL card, exactly as
    # dtype's is: without one the script labels the run
    # 'no-card-nothing-measured', prints NOT PRICED and cannot say what the arm
    # costs, which is the shape that made dtype's preview examine nothing.
    assert sum("--card" in ln for ln in lines if "--run" not in ln) == 1, lines
    assert 'ALIAS_PLAN_CARD="${ALIAS_PLAN_CARD:-NVIDIA H200}"' in CODE


def test_the_alias_planted_worlds_separate_and_the_fail_branch_is_planted():
    """A GATE THAT CANNOT FAIL IS NOT A GATE, checked on the arm's own advertised
    off-GPU command rather than asserted from its header. The four worlds land
    on three different exit codes, `classify_text` recomputes each one from the
    log, and the world that plants the FAILURE is `alias-blind`: the 2026-09-01
    apparatus, whose aliased arm never cleared the card's read roof, replayed as
    a law. That run's VOID was very nearly written up as a null result about
    DRAM, so a planted world in which the gates MUST refuse is the one this arm
    most needs."""
    script = str(ROOT / "scripts" / "alias_ablation.py")
    seen = {}
    for world, code in (("refit", exit_codes.DONE),
                        ("retracted", exit_codes.CLAIM_FAIL),
                        ("tempo", exit_codes.CLAIM_FAIL),
                        ("alias-blind", exit_codes.INVALID)):
        got = subprocess.run([sys.executable, script, "--synthetic", world],
                             capture_output=True, text=True, timeout=900,
                             cwd=str(ROOT))
        assert got.stdout.count("RESULT: ") > 0, world
        assert got.returncode == code, (world, got.returncode, got.stdout[-2000:])
        assert exit_codes.classify_text(got.stdout) == got.returncode, world
        p1 = re.search(r"^RESULT: CLAIM P1-\S+ (PASS|FAIL)", got.stdout, re.M)
        assert p1, (world, got.stdout[-2000:])
        seen[world] = (p1.group(1), got.returncode)
    # P1 alone does not separate the worlds and was never meant to: alias-blind
    # PASSES it on an interval so wide it contains everything, and what refuses
    # the page there is the VALIDITY half. Both halves are checked, because a
    # claim gate reading PASS out of an apparatus that could not see DRAM is the
    # exact failure this arm was rebuilt to make impossible.
    assert seen["refit"] == ("PASS", exit_codes.DONE)
    assert seen["retracted"] == ("FAIL", exit_codes.CLAIM_FAIL)
    assert seen["tempo"] == ("FAIL", exit_codes.CLAIM_FAIL)
    assert seen["alias-blind"][1] == exit_codes.INVALID
    blind = subprocess.run([sys.executable, script, "--synthetic", "alias-blind"],
                           capture_output=True, text=True, timeout=900,
                           cwd=str(ROOT))
    for gate in ("headroom", "attribution", "signal", "bracket"):
        assert re.search(rf"^RESULT: VALIDITY {gate}\S* FAIL", blind.stdout, re.M), gate


def test_the_rental_subsets_are_priced_by_the_table_and_not_by_a_second_copy():
    """WHAT THE SESSION COSTS AND WHAT FITS ARE DIFFERENT QUESTIONS, and the
    banner answered only the first. An operator with a two-hour budget could
    read that bn_g16 starts at minute 146 and then had to work out by hand which
    subset to name in --only.

    The two subsets are priced by `session_bound`, which walks the SAME
    `arm_minutes` and `arm_clock` the cost table walks, so a re-booked arm moves
    them by itself. A second copy of the cost model is this repo's recurring
    defect and there is not one here: this test re-derives both totals from the
    printed per-arm column."""
    listing = run(["--list"]).stdout
    booked = {n: int(re.search(rf"^  {re.escape(n)}\s+~(\d+) min (\w+)$",
                               listing, re.M).group(1)) for n in ARMS}
    clocks = {n: re.search(rf"^  {re.escape(n)}\s+~\d+ min (\w+)$",
                           listing, re.M).group(1) for n in ARMS}
    pct = int(lift("wall_over_model_pct", REPO=str(ROOT)).stdout.strip())
    for fn in ("rental_2h_arms", "rental_3h_arms"):
        names = lift(fn, REPO=str(ROOT)).stdout.split()
        assert names, fn
        assert len(names) == len(set(names)), (fn, names)
        for n in names:
            assert n in ARMS, (fn, n)
        priced, bound = lift(f"session_bound {' '.join(names)}",
                             REPO=str(ROOT)).stdout.split()
        want = sum(booked[n] for n in names)
        kern = sum(booked[n] for n in names if clocks[n] == "KERNEL")
        assert int(priced) == want, (fn, priced, want)
        assert int(bound) == want - kern + -(-kern * pct // 100), (fn, bound)
    # A name that is not an arm is a subset nobody can run, so it prices to
    # nothing and says so in its exit status rather than quietly summing to 0.
    bogus = lift("session_bound calibrate not_an_arm", REPO=str(ROOT))
    assert bogus.returncode != 0 and bogus.stdout.strip() == "0 0"


def test_the_short_rentals_buy_the_payload_and_leave_the_floor_out(tmp_path):
    """THE DECISION THIS BLOCK RECORDS. The owner's payload is bn_g16 and the
    alias ablation; the noise floor is 120 WALL minutes and sits above both of
    them in the read order. A rental that ENTERS the floor without finishing it
    is killed inside it, which fails that script's own V2 and makes even the
    partial floor unquotable, so a short booking has to leave it out rather than
    start it. Both subsets therefore carry both payload arms and neither carries
    the floor.

    And the alias arm is not what put them out of reach: it is thirteen minutes
    above the floor, so it moved every arm below the floor down by exactly that
    and moved nothing above it. bn_g16 was already past a two-hour booking."""
    two = lift("rental_2h_arms", REPO=str(ROOT)).stdout.split()
    three = lift("rental_3h_arms", REPO=str(ROOT)).stdout.split()
    for names in (two, three):
        assert "alias_ablation" in names and "bn_g16" in names
        assert "noise_floor" not in names
        # The calibration gate is deliberately NOT scoped to --only and refuses
        # the session without arm 0, and an unhonoured pin makes every
        # forced-tile arm below worthless, so both are in every set.
        assert "calibrate" in names
        assert {"pin_probe-n64-g1", "pin_probe-n256-g16"} <= set(names)
    assert set(two) < set(three)
    body = run(["--dry-run"], session=tmp_path / "s").stdout.split(
        "WHAT THIS COMMITS YOU TO")[1].split("SESSION  card=")[0]
    assert "WHAT A RENTAL OF A GIVEN LENGTH ACTUALLY REACHES" in body
    assert "NEITHER SET CONTAINS THE NOISE FLOOR" in body
    # The --only lines are printed ready to paste, not described.
    for names in (two, three):
        assert f"--only {','.join(names)}" in body, names
    # And both totals are in the banner, so the operator never has to add the
    # column back up to find out whether a booking reaches the payload.
    for fn in ("rental_2h_arms", "rental_3h_arms"):
        priced, bound = lift(f"session_bound $({fn})",
                             REPO=str(ROOT)).stdout.split()
        assert f"~{priced} priced / ~{bound} bounded min" in body, fn


def test_the_read_first_block_leads_with_the_arm_that_sets_the_units(tmp_path):
    """The roofline verdict is a fraction of a roof, and the alias ablation is
    what says the fraction is a fraction of the right thing. So it is read
    first, and the block says why rather than just listing it."""
    got = run(["--dry-run"], session=tmp_path / "s").stdout
    block = got.split("READ THESE FOUR FIRST")[1].split("WHAT TO COMMIT")[0]
    assert block.index("alias_ablation") < block.index("roofline-n256-g16")
    assert "BEFORE the roofline verdict" in block
    for name in ("alias_ablation", "roofline-n256-g16", "noise_floor", "bn_g16"):
        assert name in block, name
    # And the commit block says the arm writes nothing tracked, because an arm
    # with no --publish flag is one an operator can release a pod on top of.
    commit = got.split("WHAT TO COMMIT, AND WHAT NOT TO")[1]
    assert "THE ALIAS ABLATION WRITES NOTHING TRACKED" in commit
    assert "alias_ablation/<run id>" in commit


def test_both_end_of_rental_surfaces_disclose_the_dot_mode_state(tmp_path):
    """THE RECURRING DEFECT AGAIN, at the two surfaces read at the END of a
    rental. The body comment above the arm always disclosed what
    `--dot-fallback allow` buys: a dot ladder measures a LOWER BOUND, leaves P1
    UNKNOWN, exits 1 CLAIM_FAIL and is LATCHED by `arm`. The two surfaces the
    operator actually reads once the pod is nearly out of hours -- `arm_closes`
    and the READ-FIRST block -- enumerated three states (P1 PASS, P1 FAIL,
    headroom/attribution INVALID) and glossed exit 1 as the FAIL: "the interval
    says which of 0.10 or 0.33 it landed on instead". `choose_pinning` calls the
    fall to dot mode the LIKELY case rather than the corner, because the
    0.61-of-roof ceiling this arm exists to escape has the signature of the
    cross-lane `tl.sum` tree that `dot` removes. So the likeliest single reading
    of this arm's exit code was the state neither surface named, and the gloss
    they did carry is the retraction -- "alpha is not 0.558" for a run in which
    alpha was not asked -- that the sibling script exists to prevent. The state
    was documented where the arm is PLANNED and not where it is REPORTED.

    THE TWO EXIT-1 STATES ARE SEPARATED BY EXECUTION, NOT BY PROSE. The same
    gate builder renders FAIL in `sum` mode and UNKNOWN in `dot` mode, both
    classify to CLAIM_FAIL, and the verdict WORD is the only thing between a
    finding and an unasked question. That is why both surfaces have to send the
    operator to the RESULT line rather than to the exit code.

    SINCE 2026-09-09 THE ARM IS BOOKED `--dot-fallback refuse`, and the two
    surfaces have to disclose THAT: the fourth state was reached on 2026-09-09
    (308 s, P1 UNKNOWN at alpha >= 0.229, latched INVALID: the ledger reads
    `alias_ablation INVALID 3 308` and four validity gates failed), the bound it
    buys has been bought, and a probe miss now costs 2.0 min and exits 3. The
    state is still described, because an operator reading the page has to know
    what the flag is protecting them from; what may not stand is a surface
    saying the arm is booked at a flag it is not."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import alias_ablation as aa

    # The FAIL branch is planted beside the UNKNOWN one: a disjoint interval in
    # sum mode is the outcome the surfaces describe, and it is a different word
    # on the same line from the same builder.
    fail = aa.prediction_gate((0.05, 0.14), (0.04, 0.15), "sum")
    unknown = aa.prediction_gate((0.41, 0.62), (0.40, 0.63), "dot")
    assert fail.scored()[2] == exit_codes.FAIL
    assert unknown.scored()[2] == exit_codes.UNKNOWN
    assert unknown.result_line().startswith(
        f"RESULT: {exit_codes.CLAIM} P1-")
    assert "NOT A REFUTATION" in unknown.result_line()
    sound = exit_codes.result_line(exit_codes.VALIDITY, "headroom",
                                   exit_codes.PASS, "the pinning cleared")
    for gate in (fail, unknown):
        text = f"{sound}\n{gate.result_line()}\n"
        assert exit_codes.classify_text(text) == exit_codes.CLAIM_FAIL
    # And the state is REACHABLE from this driver: all three branches of the arm
    # name the flag that allows the fall, so it is not a corner of some other
    # invocation.
    joined = re.sub(r"\\\n\s+", " ", CODE)
    lines = [ln for ln in joined.splitlines()
             if re.match(r"\s*arm alias_ablation\s", ln)]
    assert lines and all("--dot-fallback refuse" in ln for ln in lines), lines

    closes = lift("arm_closes alias_ablation", REPO=str(ROOT)).stdout
    block = run(["--dry-run"], session=tmp_path / "s").stdout.split(
        "READ THESE FOUR FIRST")[1].split("WHAT TO COMMIT")[0]
    entry = block.split("roofline-n256-g16")[0]
    for surface in (closes, entry):
        # Flattened, because one surface is a heredoc wrapped at 76 columns and
        # a phrase that straddles two of its lines is still the phrase.
        flat = " ".join(surface.split())
        low = flat.lower()
        assert "--dot-fallback refuse" in flat, surface
        # The PAGE's figure for the probe, not a rounding of it: the plan prints
        # "PLUS 2.0 min charged outright for the probe". Both surfaces said 1.3
        # while arm_basis said 1.2, off the same page.
        assert "2.0 min" in flat, surface
        # The branch it replaced is named as history, not as the booking.
        assert "allow" in low, surface
        assert "unknown" in low and "not a refutation" in low, surface
        assert "lower bound" in low, surface
        assert "latch" in low, surface
        # The one instruction that separates the two exit-1 states, and the
        # guard on the gloss that was the misreading.
        assert "read the p1 result line" in low, surface
        assert "0.10-or-0.33" in flat, surface
    # The other two states are still stated, so the fourth was ADDED and did not
    # displace the ones that were right.
    assert "THE THIRD STATE IS NOT AN OUTCOME" in closes
    assert "P1 PASS" in closes and "P1 FAIL" in closes
    for word in ("P1 PASS", "P1 FAIL", "headroom or attribution FAIL"):
        assert word in entry, word


# --------------------------------------------------------------------------
# 16. the second opinion: the page is compared with the exit code, and no word
#     the page does not support is latched
# --------------------------------------------------------------------------

def arm_lift(script, repo, ledger, logs, **extra):
    """`lift` with every global `arm()` reads, in measuring mode."""
    return lift(script, REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs),
                ONLY="", DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0,
                PY_BASE=sys.executable, **extra)


def test_the_second_opinion_is_actually_called_and_not_only_described():
    """THE VERDICT OF THE 2026-09-03 REVIEW, PINNED. `classify_text` had four
    mentions in the driver and every one was a comment or an echo; `arm()`
    decided the state from the integer alone. The function has to be CALLED,
    from the one place a state is written, and the mention that matters is the
    one inside `log_verdict`'s python rather than in prose."""
    body = CODE.split("\narm() {", 1)[1].split("\n}", 1)[0]
    assert 'second_opinion "$rc" "$state" "$(log_verdict "$log")" "$adopts"' in body
    verdict = CODE.split("\nlog_verdict() {", 1)[1].split("\n}", 1)[0]
    assert "EC.classify_text(text)" in verdict
    assert "NoGatesScored" in verdict
    # ...and only in measuring mode: a plan scores no gate and gets planning
    # words, so a dry run must never reach the second opinion.
    assert 'if (( DRY )); then\n    state="$(dry_state "$rc" "$log")"\n  else' in body


@pytest.mark.parametrize("text,want", [
    ("RESULT: VALIDITY V0 PASS ok\nRESULT: CLAIM C1 PASS fine\n", "0"),
    ("prose\nRESULT: VALIDITY V0 PASS ok\nRESULT: CLAIM C1 FAIL gap\n", "1"),
    ("RESULT: VALIDITY V0 FAIL broke\nRESULT: CLAIM C1 PASS fine\n", "3"),
    ("RESULT: VALIDITY V0 PASS ok\nRESULT: CLAIM C1 UNKNOWN NOT A REFUTATION\n", "1"),
    ("C1 ... [PASS]  a pre-registered expectation\nfloor: 0.0905\n", "NONE"),
    ("", "NONE"),
])
def test_log_verdict_reads_the_result_lines_through_the_module(tmp_path, text, want):
    """The integer, or NONE, and the module's own regex rather than a shell
    copy of it: the prose log is the one the old summary matched eighteen
    times, and it must read as no gate at all."""
    log = tmp_path / "arm.log"
    log.write_text(text)
    got = lift(f"log_verdict {log}", REPO=str(ROOT), PY_BASE=sys.executable)
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == want


def test_log_verdict_says_unreadable_rather_than_none_when_it_cannot_ask(tmp_path):
    """"COULD NOT CHECK" IS NOT "CHECKED AND FOUND NOTHING". A missing log, and
    a REPO with no moe/bench/exit_codes to import, both answer UNREADABLE, and
    `second_opinion` refuses to latch on that word in every state."""
    gone = lift(f"log_verdict {tmp_path}/gone.log", REPO=str(ROOT),
                PY_BASE=sys.executable)
    assert gone.stdout.strip() == "UNREADABLE"
    log = tmp_path / "arm.log"
    log.write_text("RESULT: CLAIM C1 PASS fine\n")
    # A REPO whose exit_codes will not import. Planted as a regular package so
    # it shadows the `moe` the venv has installed; a bare tmp_path would not,
    # and the driver would quietly read the installed module and answer 0.
    broken = plant_exit_codes(tmp_path / "broken")
    (broken / "moe" / "bench" / "exit_codes.py").unlink()
    (broken / "moe" / "bench" / "exit_codes.py").write_text(
        "raise ImportError('planted: this checkout has no readable table')\n")
    bare = lift(f"log_verdict {log}", REPO=str(broken), PY_BASE=sys.executable)
    assert bare.stdout.startswith("UNREADABLE"), bare.stdout
    assert "ImportError" in bare.stdout
    # ...and the same page from a repo whose module DOES import reads 0.
    assert lift(f"log_verdict {log}", REPO=str(plant_exit_codes(tmp_path / "ok")),
                PY_BASE=sys.executable).stdout.strip() == "0"
    for rc in (0, 1, 2, 3):
        got = lift(f'second_opinion {rc} "$(ledger_state {rc})" '
                   f'{shlex.quote(bare.stdout.strip())} 0', REPO=str(ROOT))
        word, _, note = got.stdout.rstrip("\n").partition("\t")
        assert word == "UNKNOWN", (rc, got.stdout)
        assert "SECOND OPINION UNAVAILABLE" in note


#: The printed table in `second_opinion`'s own comment, row for row, plus the
#: adoption split. (rc, what the log implied, adopts_exit_codes rc) -> (state,
#: a phrase the note must carry). Every latched word here is EARNED by an
#: agreeing page; every other row is UNKNOWN or RETRY and says why.
def defect(rc, implied):
    """The note a DEFECT row carries: both codes, both words, in that order."""
    return (f"DEFECT: the process exited {rc} {exit_codes.ledger_state(rc)} but its "
            f"RESULT lines imply {implied} {exit_codes.ledger_state(implied)}")


SECOND_OPINION = [
    (0, "0", 0, "DONE", "log agrees"),
    (1, "1", 0, "CLAIM_FAIL", "log agrees"),
    (3, "3", 0, "INVALID", "log agrees"),
    (0, "1", 0, "UNKNOWN", defect(0, 1)),
    (1, "0", 0, "UNKNOWN", defect(1, 0)),
    (2, "0", 0, "UNKNOWN", defect(2, 0)),
    (3, "1", 0, "UNKNOWN", defect(3, 1)),
    (0, "3", 0, "UNKNOWN", defect(0, 3)),
    (0, "NONE", 0, "UNKNOWN", "UNEARNED DONE"),
    (1, "NONE", 0, "RETRY", "CRASH"),
    (1, "NONE", 1, "UNKNOWN", "1 is three things"),
    (1, "NONE", 2, "UNKNOWN", "1 is three things"),
    (1, "1", 1, "UNKNOWN", "has not adopted"),
    (2, "NONE", 0, "REFUSED", ""),
    (3, "NONE", 0, "UNKNOWN", "UNEARNED INVALID"),
    (4, "0", 0, "RETRY", "process code wins"),
    (4, "NONE", 0, "RETRY", ""),
    (127, "1", 0, "RETRY", "process code wins"),
    (130, "NONE", 0, "RETRY", ""),
]


@pytest.mark.parametrize("rc,implied,adopts,state,tell", SECOND_OPINION)
def test_the_second_opinion_table(rc, implied, adopts, state, tell):
    """Every row of the table, through the shipped function. The three rows
    that LATCH all carry an agreeing page; the DEFECT rows carry both codes in
    the note so the ledger says what disagreed with what; the crash row is
    RETRY only for a file that speaks the table, because from one that does not
    a 1 is still three things."""
    got = lift(f'second_opinion {rc} "$(ledger_state {rc})" {shlex.quote(implied)} {adopts}',
               REPO=str(ROOT))
    assert got.returncode == 0, got.stderr
    assert got.stdout.count("\n") == 1, "one line, read by one `read`"
    word, _, note = got.stdout.rstrip("\n").partition("\t")
    assert word == state, got.stdout
    assert tell in note, note
    if state in ("DONE", "CLAIM_FAIL", "INVALID"):
        assert note.startswith("log agrees"), note
    elif state == "UNKNOWN":
        assert "NOT latched" in note, note
    if note.startswith("DEFECT:"):
        assert f"exited {rc}" in note and f"imply {implied}" in note


@pytest.mark.parametrize("label,page,rc,state,tell,latched", [
    ("fail_line_exit_0", "RESULT: CLAIM C1 FAIL gap 0.05 < 0.10",
     0, "UNKNOWN", "DEFECT", False),
    ("no_line_exit_0", "measured something and printed no gate",
     0, "UNKNOWN", "UNEARNED DONE", False),
    ("pass_lines_exit_1", "RESULT: VALIDITY V0 PASS ok\nRESULT: CLAIM C1 PASS fine",
     1, "UNKNOWN", "DEFECT", False),
    ("no_line_exit_3", "nothing scored",
     3, "UNKNOWN", "UNEARNED INVALID", False),
    ("fail_line_exit_1", "RESULT: CLAIM C1 FAIL gap",
     1, "CLAIM_FAIL", "log agrees", True),
    ("pass_line_exit_0", "RESULT: CLAIM C1 PASS fine",
     0, "DONE", "log agrees", True),
    ("validity_fail_exit_3", "RESULT: VALIDITY V0 FAIL broke",
     3, "INVALID", "log agrees", True),
    ("scored_then_crash_4", "RESULT: CLAIM C1 PASS fine\nTraceback: boom",
     4, "RETRY", "process code wins", False),
])
def test_arm_latches_only_a_word_the_page_earned(tmp_path, label, page, rc, state,
                                                 tell, latched):
    """THE THREE PROOFS THE REVIEW RAN, PLUS THE ROWS AROUND THEM, through the
    shipped `arm()`. `RESULT: CLAIM C1 FAIL` under exit 0 used to land DONE and
    be SKIPPED on the next pass; no RESULT line under exit 0 landed DONE and
    was latched; all-PASS lines under exit 1 landed CLAIM_FAIL and was latched.
    Each is UNKNOWN now, carries its reason in the ledger note, and is
    re-attempted. The honest rows still latch, which is the branch that must
    not move: a CLAIM_FAIL with its FAIL line on the page is the most valuable
    outcome this study has and re-running it is the failure mode
    exit_codes.py is named against."""
    stub = tmp_path / f"{label}.sh"
    stub.write_text(f"#!/bin/bash\nprintf '%s\\n' {shlex.quote(page)}\nexit {rc}\n")
    ledger = tmp_path / "ARMS.tsv"
    logs = tmp_path / "logs"
    logs.mkdir()
    first = arm_lift(f"arm cap_test bash {stub}", ROOT, ledger, logs)
    assert first.returncode == 0, first.stderr
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert row[1] == state, (row, first.stdout)
    assert row[2] == str(rc)
    assert tell in row[6], row
    assert tell in first.stdout
    if row[6].startswith("DEFECT:"):
        assert f"exited {rc}" in row[6] and "imply" in row[6]
    if state == "RETRY":
        assert "last 5 lines of" in first.stdout
        assert "RETRY_ARMS=1" in arm_lift(
            f"arm cap_test bash {stub}; echo RETRY_ARMS=$RETRY_ARMS",
            ROOT, tmp_path / "again.tsv", logs).stdout
    # The lifted `arm()` writes rows only; the header is the session's. One
    # row after a latched pass, two after a re-attempted one.
    again = arm_lift(f"arm cap_test bash {stub}", ROOT, ledger, logs)
    if latched:
        assert f"SKIP cap_test (already {state}" in again.stdout
        assert len(ledger.read_text().splitlines()) == 1
    else:
        assert "SKIP cap_test" not in again.stdout
        assert len(ledger.read_text().splitlines()) == 2


def test_an_import_time_crash_is_retry_not_a_refuted_claim(tmp_path):
    """THE POD'S REAL 2026-09-01 SHAPE, through the real `arm()` with the real
    pod command lines. Python exits 1 for an exception that escapes, the
    scripts' ERROR(4) guards wrap `_main()` and cannot catch a failure at
    import, and every measuring arm has adopted the table, so the third pass's
    "UNKNOWN for a non-adopter" covered none of them: alias_ablation,
    pin_probe, calibrate and dtype all landed CLAIM_FAIL, latched, and were
    skipped on every resume. For calibrate that made the calibration gate say
    ARM CLAIM_FAIL and refuse every resume until the row was deleted by hand.

    A `torch.py` that raises ImportError is planted on PYTHONPATH, which is
    what an ABI drift looks like to the interpreter; each script crashes at
    `<module>` with zero RESULT lines, and the tracked tree is never touched
    because nothing gets past the import."""
    broken = tmp_path / "pp"
    broken.mkdir()
    (broken / "torch.py").write_text(
        "raise ImportError('planted ABI drift: torch was built for a different runtime')\n")
    session = tmp_path / "session"
    session.mkdir()
    logs = session / "logs"
    logs.mkdir()
    ledger = session / "ARMS.tsv"
    values = {
        "PY_VLLM": sys.executable, "PY_BASE": sys.executable, "REPO": str(ROOT),
        "SESSION": str(session),
        "PIN_SWEPT": re.search(r"^PIN_SWEPT='([^']*)'", CODE, re.M).group(1),
        "ALIAS_MODELS": re.search(r"^ALIAS_MODELS=(\S+)$", CODE, re.M).group(1),
    }

    def expand(word):
        def one(m):
            name = m.group(1)
            assert name in values, f"{word}: no planted value for ${name}"
            return values[name]
        return re.sub(r"\$\{?(\w+)\}?", one, word)

    tracked = ["git", "-C", str(ROOT), "status", "--porcelain", "--",
               "moe/bench/hardware", "results/published"]
    before = subprocess.run(tracked, capture_output=True, text=True, timeout=60).stdout
    for arm_name in ("alias_ablation", "pin_probe-n64-g1", "calibrate", "dtype"):
        words = [expand(w) for w in measuring_invocation(arm_name)]
        assert words[0] == "arm" and words[1] == arm_name, words
        command = ["env", f"PYTHONPATH={broken}",
                   f"MOE_RESULTS_DIR={tmp_path}/gaps-nocard", *words[2:]]
        got = arm_lift(f"arm {arm_name} {shlex.join(command)}", ROOT, ledger, logs)
        assert got.returncode == 0, got.stderr
        log = (logs / f"{arm_name}.log").read_text()
        assert "Traceback" in log and "planted ABI drift" in log, (arm_name, log[-1500:])
        assert "RESULT: " not in log, arm_name
        row = ledger.read_text().splitlines()[-1].split("\t")
        assert row[0] == arm_name and row[2] == "1", row
        assert row[1] == "RETRY", (arm_name, row)
        assert row[6].startswith("CRASH:"), row
        assert "planted ABI drift" in got.stdout, "the tail of the log is printed"
        again = arm_lift(f"arm {arm_name} {shlex.join(command)}", ROOT, ledger, logs)
        assert f"SKIP {arm_name}" not in again.stdout, "a crash is not latched"
    after = subprocess.run(tracked, capture_output=True, text=True, timeout=60).stdout
    assert before == after, "an import-time crash wrote into the tracked tree"
    # And the calibration gate no longer blocks every resume: the row is RETRY,
    # the gate refuses THIS pass (arm 0 did not stand behind a ruler) and the
    # next pass re-runs arm 0 instead of skipping a latched CLAIM_FAIL forever.
    verdict = lift('calibration_verdict "$(ledger_arm_state calibrate)" PUBLISHED; echo "rc=$?"',
                   REPO=str(ROOT), LEDGER=str(ledger))
    assert verdict.stdout.splitlines()[0] == "ARM RETRY", verdict.stdout
    assert verdict.stdout.strip().endswith("rc=1")


def test_a_defective_page_is_printed_under_its_own_heading_and_fails_the_session(tmp_path):
    """RECORDING THE WORD AND EXITING 0 WOULD LEAVE IT IN A FILE NOBODY READS.
    `defect_rows` is the one source for the heading and the exit code; last row
    per arm wins, so a defect that a later pass re-ran cleanly is gone and one
    a later pass did not touch is still shown."""
    ledger = tmp_path / "ARMS.tsv"
    ledger.write_text(
        "arm\tstate\trc\tseconds\tdirty\tlog\tnote\n"
        f"cap_test\tUNKNOWN\t0\t9\t0\t/a.log\t{defect(0, 1)}. x\n"
        f"ruler\tUNKNOWN\t1\t9\t0\t/b.log\t{defect(1, 0)}. y\n"
        "ruler\tDONE\t0\t9\t0\t/b.log\tlog agrees: RESULT lines imply 0\n"
        "dtype\tUNKNOWN\t0\t9\t0\t/c.log\tUNEARNED DONE: exit 0 with no RESULT line. z\n"
        "span\tREFUSED\t2\t0\t0\t/d.log\t\n")
    got = lift(f"defect_rows {ledger}", REPO=str(ROOT))
    lines = got.stdout.splitlines()
    assert len(lines) == 1, got.stdout
    assert lines[0].startswith("  cap_test") and "exit 0" in lines[0]
    assert "imply 1 CLAIM_FAIL" in lines[0]
    clean = tmp_path / "CLEAN.tsv"
    clean.write_text("arm\tstate\trc\tseconds\tdirty\tlog\tnote\n"
                     "ruler\tDONE\t0\t9\t0\t/z.log\tlog agrees: RESULT lines imply 0\n")
    assert lift(f"defect_rows {clean}", REPO=str(ROOT)).stdout == ""
    # The UNEARNED DONE row is not a defect; it is OWED, under the other
    # heading, and the same last-row rule applies (ruler re-ran cleanly).
    owed = lift(f"owed_rows {ledger}", REPO=str(ROOT)).stdout.splitlines()
    assert len(owed) == 1 and owed[0].startswith("  dtype") and "UNEARNED DONE" in owed[0]
    assert lift(f"owed_rows {clean}", REPO=str(ROOT)).stdout == ""
    # The shipped session reads both functions, prints both headings, and
    # exits through `session_rc` over the same ledger, so the summary and the
    # code have one source. Nothing on such a page is a verdict.
    assert 'DEFECTS="$(defect_rows "$LEDGER")"' in CODE
    assert 'OWED="$(owed_rows "$LEDGER")"' in CODE
    assert 'say "THE ROWS WHOSE PAGE AND EXIT CODE DISAGREE"' in CODE
    assert 'say "THE ROWS THIS SESSION STILL OWES"' in CODE
    tail = CODE.split('OWED="$(owed_rows "$LEDGER")"', 1)[1]
    assert 'SESSION_RC="$(session_rc "$LEDGER" "$RETRY_ARMS")"' in tail
    assert tail.count('exit "$SESSION_RC"') == 1
    # and no branch of the measuring tail exits by a constant any more
    measuring_tail = tail.split('SESSION_RC="$(session_rc', 1)[1].split("\nexit 0", 1)[0]
    assert 'exit "$RC_INVALID"' not in measuring_tail
    assert 'exit "$RC_RETRY"' not in measuring_tail


def test_the_summary_prints_the_reason_a_row_is_unknown_rather_than_one_paraphrase(tmp_path):
    """THE PARAPHRASE WAS THE RECURRING DEFECT. `summarize_arm` used to say
    "this arm exited 1 and the file it ran has not adopted ...", which was one
    of the five reasons a row can be UNKNOWN and wrong for the other four the
    moment they existed. The note on the ledger row is printed instead."""
    log = tmp_path / "arm.log"
    log.write_text(RESULT_LOG)
    note = "DEFECT: the process exited 0 DONE but its RESULT lines imply 1 CLAIM_FAIL."
    got = lift(f"summarize_arm cap_test {log} UNKNOWN {shlex.quote(note)}", REPO=str(ROOT))
    assert "STATE UNKNOWN, NOT LATCHED" in got.stdout
    assert note in got.stdout
    assert "has not adopted" not in got.stdout, "the old paraphrase is gone"
    # The RESULT lines are still printed after it: which gate disagreed with
    # the exit code is the whole content of a DEFECT row.
    assert "RESULT: CLAIM C1 FAIL" in got.stdout
    # A row with no note says so rather than inventing one.
    bare = lift(f"summarize_arm cap_test {log} UNKNOWN", REPO=str(ROOT))
    assert "the reason was not recorded" in bare.stdout
    # And the shipped loop passes the note through.
    assert 'summarize_arm "$n" "$log" "$state" "$reason"' in CODE


@pytest.mark.parametrize("arm_name", ARMS)
def test_every_measuring_invocation_runs_a_script_the_second_opinion_can_read(arm_name):
    """THE WALKER, EXTENDED TO THE SECOND OPINION. For every arm the pod runs:
    the file it runs speaks the table (so a 1 with no gate line is a crash and
    not "three things"), renders RESULT lines (so `log_verdict` has something
    to read and DONE can be earned), and exits through `classify` (so agreement
    is the expected shape and a disagreement is a defect in that file).

    counter_plan is the one arm that fails the middle check today, and the
    check is written to go RED when it stops failing: dram_counter_route.py
    prints no RESULT line in any mode (review finding 4, in a file this slice
    does not own), so on the pod its exit 0 is an UNEARNED DONE and lands
    UNKNOWN. `arm_closes counter_plan` says so; when that script prints one
    line per verdict, delete the special case here and that sentence there."""
    words = measuring_invocation(arm_name)
    rel = lift(f"arm_script {shlex.quote(arm_name)}", REPO=str(ROOT)).stdout.strip()
    assert rel and (ROOT / rel).exists(), (arm_name, rel)
    if rel == "moe/bench/cli.py":
        assert "-m" in words and words[words.index("-m") + 1] == "moe.bench.cli", words
    else:
        assert any(w.endswith(rel) for w in words), (arm_name, rel, words)
    adopts = lift(f"adopts_exit_codes {shlex.quote(rel)}; echo rc=$?",
                  REPO=str(ROOT)).stdout.strip()
    assert adopts.endswith("rc=0"), (arm_name, rel, adopts)
    source = (ROOT / rel).read_text()
    renders = "result_line(" in source or "RESULT: " in source
    assert "classify(" in source, (arm_name, rel)
    # counter_plan was special-cased here while dram_counter_route.py rendered
    # its gates as prose; since the gates slice it prints RESULT lines like every
    # other arm, so the second opinion can read it and no exemption remains.
    assert renders, (arm_name, rel, "the second opinion would read NONE on every run")


# --------------------------------------------------------------------------
# 17. the ledger is reachable: SESSION=, --resume-latest, --new
# --------------------------------------------------------------------------

def planted_root(tmp_path):
    """A session root with a dry-only directory, two measuring sessions for
    this card and one for another card. The newest for THIS card is the one
    with the March stamp; the April one is another card's and must never be
    picked."""
    root = tmp_path / "root"
    (root / "session-nocard-20260101T000000Z").mkdir(parents=True)
    for stamp in ("20260201T000000Z", "20260301T000000Z"):
        d = root / f"session-nocard-{stamp}"
        d.mkdir()
        (d / "ARMS.tsv").write_text("arm\tstate\trc\tseconds\tdirty\tlog\tnote\n")
    other = root / "session-othercard-20260401T000000Z"
    other.mkdir()
    (other / "ARMS.tsv").write_text("arm\tstate\trc\tseconds\tdirty\tlog\tnote\n")
    return root


def choose(dry, resume, new, explicit, root):
    got = lift(f'session_choice {dry} {resume} {new} {shlex.quote(explicit)} '
               f'{shlex.quote(str(root))} session-nocard-; echo "rc=$?"',
               REPO=str(ROOT))
    assert got.returncode == 0, got.stderr
    lines = got.stdout.rstrip("\n").splitlines()
    how, _, what = lines[0].partition("\t")
    return how, what, lines[-1]


def test_latest_session_is_the_newest_for_this_card_with_a_measuring_ledger(tmp_path):
    root = planted_root(tmp_path)
    got = lift(f"latest_session {root} session-nocard-; echo rc=$?", REPO=str(ROOT))
    assert got.stdout.splitlines() == [str(root / "session-nocard-20260301T000000Z"), "rc=0"]
    # A dry-only directory is not a session to resume into, and an empty root
    # is nothing at all rather than the literal glob.
    (root / "session-nocard-20260201T000000Z" / "ARMS.tsv").unlink()
    (root / "session-nocard-20260301T000000Z" / "ARMS.tsv").unlink()
    got = lift(f"latest_session {root} session-nocard-; echo rc=$?", REPO=str(ROOT))
    assert got.stdout.strip() == "rc=1", got.stdout
    empty = tmp_path / "empty"
    empty.mkdir()
    assert lift(f"latest_session {empty} session-nocard-; echo rc=$?",
                REPO=str(ROOT)).stdout.strip() == "rc=1"


def test_session_choice_every_branch(tmp_path):
    """EVERY BRANCH, PLANTED, because the one that matters (a measuring run
    finding a session it would have silently ignored) cannot be reached end to
    end on a box with no card: the driver refuses before it chooses."""
    root = planted_root(tmp_path)
    latest = str(root / "session-nocard-20260301T000000Z")
    # A measuring run without --new, with a latest: refused, naming it.
    assert choose(0, 0, 0, "", root) == ("LATEST_EXISTS", latest, "rc=1")
    # A dry run is free and skips nothing, so it opens a fresh one.
    how, what, rc = choose(1, 0, 0, "", root)
    assert (how, rc) == ("NEW", "rc=0") and what.startswith(f"{root}/session-nocard-2")
    assert what != latest
    # --new says so on purpose.
    how, what, rc = choose(0, 0, 1, "", root)
    assert (how, rc) == ("NEW", "rc=0") and what != latest
    # --resume-latest lands in the newest measuring session for THIS card.
    assert choose(0, 1, 0, "", root) == ("RESUMED", latest, "rc=0")
    assert choose(1, 1, 0, "", root) == ("RESUMED", latest, "rc=0")
    # SESSION= names it outright and wins.
    assert choose(0, 0, 0, "/x/y", root) == ("NAMED", "/x/y", "rc=0")
    # Contradictions are refused, not resolved.
    how, what, rc = choose(0, 1, 1, "", root)
    assert (how, rc) == ("REFUSED", "rc=1") and "contradict" in what
    for resume, new in ((1, 0), (0, 1)):
        how, what, rc = choose(0, resume, new, "/x/y", root)
        assert (how, rc) == ("REFUSED", "rc=1") and "names the directory" in what
    # Nothing to resume is a refusal that says so.
    empty = tmp_path / "empty"
    empty.mkdir()
    how, what, rc = choose(0, 1, 0, "", empty)
    assert (how, rc) == ("REFUSED", "rc=1") and "found no session" in what
    # With no latest, a measuring run opens a new one without being asked.
    how, what, rc = choose(0, 0, 0, "", empty)
    assert (how, rc) == ("NEW", "rc=0")


def test_resume_latest_and_new_end_to_end_in_dry_mode(tmp_path):
    """The flags reach the shipped driver, and the resumed directory is the one
    the ledger is read from and written into."""
    root = planted_root(tmp_path)
    latest = root / "session-nocard-20260301T000000Z"
    env = {"SESSION_ROOT": str(root)}
    got = run(["--dry-run", "--resume-latest", "--only", "counter_plan"], env_extra=env)
    assert got.returncode == 0, got.stdout[-2000:]
    assert f"session   {latest}" in got.stdout
    assert (latest / "ARMS-dryrun.tsv").exists()
    assert "(RESUMED)" in got.stdout
    assert "--resume-latest" in got.stdout.split("THE SESSION DIRECTORY IS")[1]
    # A bare dry run beside it opens a fresh directory, and says NEW.
    bare = run(["--dry-run", "--only", "counter_plan"], env_extra=env)
    assert bare.returncode == 0, bare.stdout[-2000:]
    assert f"session   {latest}" not in bare.stdout and "(NEW)" in bare.stdout
    # Refusals: nothing to resume, contradictory flags, SESSION= plus a flag.
    empty = tmp_path / "empty"
    empty.mkdir()
    none = run(["--dry-run", "--resume-latest"], env_extra={"SESSION_ROOT": str(empty)})
    assert none.returncode == exit_codes.REFUSED
    assert "REFUSED: --resume-latest found no session" in none.stdout
    both = run(["--dry-run", "--resume-latest", "--new"], env_extra=env)
    assert both.returncode == exit_codes.REFUSED and "contradict" in both.stdout
    named = run(["--dry-run", "--resume-latest"], session=tmp_path / "s", env_extra=env)
    assert named.returncode == exit_codes.REFUSED and "names the directory" in named.stdout
    assert not (tmp_path / "s").exists()
    # And the usage block, which is also --help, documents all three.
    helped = run(["--help"]).stdout
    for word in ("SESSION=", "--resume-latest", "--new"):
        assert word in helped, word
    assert "SESSION=" in TEXT.split("set -uo pipefail", 1)[0]


def test_the_measuring_refusal_names_all_three_ways_out():
    """The refusal's words, since the branch itself cannot run off GPU: an
    operator who reads it has to be able to paste the way forward."""
    block = CODE.split("LATEST_EXISTS)", 1)[1].split("exit \"$RC_REFUSED\"", 1)[0]
    assert "--resume-latest" in block and "--new" in block and "SESSION=" in block
    assert "EMPTY ledger" in block
    assert 'read -r SESSION_HOW SESSION_WHAT' in CODE
    assert 'session_choice "$DRY" "$RESUME_LATEST" "$NEW_SESSION" "${SESSION:-}"' in CODE


# --------------------------------------------------------------------------
# 18. descriptions of behaviour that has since changed are told as history
# --------------------------------------------------------------------------

def test_the_caveat_tells_the_adopters_old_codes_as_history():
    """THE RECURRING DEFECT, INSIDE THE FUNCTION THAT EXISTS TO DISCLOSE IT.
    `contract_caveat` said check_mma_path.sh spends 1 on refusals (2 since it
    adopted), that its --dry-run exits 0 (2), that cli.py returns 4 on a pin
    miss (INVALID through classify) and that dram_counter_route.py returns 3 on
    BLOCKED (DONE). Each is checked against the file it describes, not against
    memory."""
    caveat = CODE.split("\ncontract_caveat() {", 1)[1].split("\n}", 1)[0]
    for stale in ('documents "1 a gate failed"',
                  "returns 4 when the implementations",
                  "exits 0 from\\n",
                  "returns 3 for\\n"):
        assert stale not in caveat, stale
    for history in ('used to document "1 a gate failed"',
                    "used to return 4",
                    "used to exit 0",
                    "used to return 3",
                    "no longer reaches this caveat"):
        assert history in caveat, history
    mma = (ROOT / "scripts" / "check_mma_path.sh").read_text()
    assert 'refuse() { echo "[mma] REFUSED: $*" >&2; exit "$EXIT_REFUSED"; }' in mma
    assert 'exit "$EXIT_REFUSED"' in mma.split("--dry-run scored no gate", 1)[1][:400]
    cli = (ROOT / "moe" / "bench" / "cli.py").read_text()
    assert "rc = EC.classify(scored)" in cli
    route = (ROOT / "scripts" / "dram_counter_route.py").read_text()
    # The route used to spell its own two codes; since the gates slice it
    # exits through the table over the same Gate objects that print its lines.
    assert "exit_codes.classify(" in route
    assert "return exit_codes.REFUSED if verdict == REFUSE else exit_codes.DONE" not in route


def test_the_alias_booking_gap_survives_only_as_closed_history():
    """prose.md items 1 and 34. The 11.6-versus-13.0 gap was closed at
    00f3324; a heading in the alias arm's own body comment still stated it as
    current ("UNDER-BOOKS ITSELF BY 1.4 MINUTES") over a paragraph that said it
    was closed. Every remaining mention of 11.6 in the driver sits within two
    lines of a word that dates it, and the row states the agreement."""
    assert "UNDER-BOOKS ITSELF BY" not in TEXT
    assert "reads the difference rather than absorbing it" not in TEXT
    lines = TEXT.splitlines()
    hits = [i for i, ln in enumerate(lines) if "11.6" in ln]
    assert hits, "the history should still be told"
    for i in hits:
        window = " ".join(lines[max(0, i - 2): i + 3])
        assert re.search(r"[Uu]ntil|used to|closed", window), lines[i]
    basis = lift("arm_basis alias_ablation", REPO=str(ROOT)).stdout
    assert "the two agree" in basis and "the gap is closed" in basis
    assert "NO LONGER\n# UNDER-BOOKS ITSELF" in TEXT


# --------------------------------------------------------------------------
# 18. the fifteenth instance: the LEVEL side, and the four fixes beside it
# --------------------------------------------------------------------------

HEADER = "arm\tstate\trc\tseconds\tdirty\tlog\tnote\n"


def session_rc_of(tmp_path, rows: str, retry: int = 0) -> str:
    ledger = tmp_path / f"L{abs(hash(rows)) % 10**8}.tsv"
    ledger.write_text(HEADER + rows)
    got = lift(f"session_rc {ledger} {retry}", REPO=str(ROOT), RC_INVALID=3, RC_RETRY=4)
    assert got.returncode == 0, got.stderr
    return got.stdout.strip()


def test_the_session_exits_invalid_over_any_unknown_row_not_only_a_defect(tmp_path):
    """F10. The session's closing exit code reached INVALID only through
    `defect_rows`, whose awk selected UNKNOWN rows whose note begins DEFECT:.
    An UNEARNED DONE (exit 0, no RESULT line: the case the second opinion was
    written to demote), an exit 1 from a non-adopting file, a log the second
    opinion could not read: each left the pod session at exit 0, the runbook's
    next line was the exfil tar, and nothing machine-readable said an arm was
    still owed. The rule is one lifted function over the ledger now, planted in
    every direction: any UNKNOWN row is 3, a crash with no UNKNOWN row is 4,
    a clean ledger is 0, and last row per arm wins."""
    assert re.search(r"^RC_INVALID=3$", TEXT, re.M) and re.search(r"^RC_RETRY=4$", TEXT, re.M)
    done = "calibrate\tDONE\t0\t9\t1\t/a.log\tlog agrees: RESULT lines imply 0\n"
    unearned = ("dtype\tUNKNOWN\t0\t9\t0\t/c.log\tUNEARNED DONE: exit 0 with no RESULT "
                "line. A check that examined nothing reports no failures. NOT latched.\n")
    assert session_rc_of(tmp_path, done + unearned) == "3"
    assert session_rc_of(tmp_path,
                         done + f"cap_test\tUNKNOWN\t0\t9\t0\t/a.log\t{defect(0, 1)}.\n") == "3"
    assert session_rc_of(tmp_path, done + "ruler\tUNKNOWN\t1\t9\t0\t/b.log\texit 1 with no "
                         "RESULT line from a file that has not adopted moe/bench/exit_codes, "
                         "where 1 is three things at once. NOT latched.\n") == "3"
    assert session_rc_of(tmp_path, done + "span\tUNKNOWN\t3\t9\t0\t/d.log\tUNEARNED INVALID: "
                         "exit 3 with no RESULT line. NOT latched.\n") == "3"
    assert session_rc_of(tmp_path, done + "occupancy\tUNKNOWN\t0\t9\t0\t/e.log\tSECOND "
                         "OPINION UNAVAILABLE: exit 0, but classify_text could not run.\n") == "3"
    # an UNKNOWN row takes precedence over a crash count, and a crash alone is 4
    assert session_rc_of(tmp_path, done + unearned, retry=2) == "3"
    assert session_rc_of(tmp_path, done, retry=1) == "4"
    assert session_rc_of(tmp_path, done) == "0"
    # CLAIM_FAIL, INVALID and REFUSED are results or free, never owed
    assert session_rc_of(tmp_path, done + "ruler\tCLAIM_FAIL\t1\t9\t0\t/b.log\tlog agrees\n"
                         "span\tREFUSED\t2\t0\t0\t/d.log\t\n"
                         "bm128_depth\tINVALID\t3\t9\t0\t/f.log\tlog agrees\n") == "0"
    # last row per arm wins: an owed arm that a resume re-ran cleanly is paid
    paid = done + unearned + "dtype\tDONE\t0\t9\t0\t/c.log\tlog agrees: RESULT lines imply 0\n"
    assert session_rc_of(tmp_path, paid) == "0"
    # the constants are read, not defaulted: a lift that forgets them fails
    ledger = tmp_path / "unbound.tsv"
    ledger.write_text(HEADER + done + unearned)
    bare = lift(f"session_rc {ledger} 0", REPO=str(ROOT))
    assert bare.returncode != 0 and "RC_INVALID" in bare.stderr, bare


def test_the_owed_rows_are_printed_under_their_own_heading_by_the_shipped_tail():
    tail = CODE.split('OWED="$(owed_rows "$LEDGER")"', 1)[1]
    assert 'if [[ -n "$OWED" ]]; then' in tail
    assert ("STILL OWED: UNKNOWN, not latched, not a result. --resume-latest re-attempts them"
            in TEXT)
    assert "re-attempts every one. The session exits INVALID over these rows" in TEXT


def calibrate_stub(repo: Path, body: str) -> Path:
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    yaml = repo / "moe" / "bench" / "hardware" / "measured_testcard.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    plant_exit_codes(repo)
    fake = repo / "scripts" / "calibrate_hardware.py"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        f'printf "name: T\\nprovenance:\\n  utc: \'$(date -u '
        '+%Y-%m-%dT%H:%M:%S)+00:00\'\\n" > '
        f'"{yaml}"\n'
        f'echo "[calibrate] PUBLISHED to {yaml}"\n'
        + body)
    return fake


def test_the_calibration_refusal_prints_the_ledger_note_for_an_unknown_row(tmp_path):
    """F5, PROVED THROUGH `arm()`. A calibrate that publishes a fresh yaml,
    prints `RESULT: VALIDITY clock_established FAIL` and exits 0 lands UNKNOWN
    with a DEFECT note (page and exit code disagree), and the gate correctly
    refuses. Until 2026-09-08 the refusal then said "It exited 1 from a file
    this driver could not confirm speaks moe/bench/exit_codes" -- rc was 0 and
    calibrate_hardware.py adopts -- and sent the operator to the wrong cause
    at minute 3. The note on the row is printed instead, the way summarize_arm
    already does."""
    repo = tmp_path / "repo"
    fake = calibrate_stub(
        repo, "echo 'RESULT: VALIDITY clock_established FAIL samples disagree with the plateau'\n"
              "exit 0\n")
    logs = tmp_path / "logs"
    logs.mkdir()
    ledger = tmp_path / "ARMS.tsv"
    ledger.write_text(HEADER)
    yaml = repo / "moe" / "bench" / "hardware" / "measured_testcard.yaml"
    got = lift('SESSION_SINCE="$(date -u +%Y%m%d%H%M%S)"\n'
               f'arm calibrate bash {fake} --publish >/dev/null\n'
               'ROW="$(ledger_arm_state calibrate)"\n'
               f'STATE="$(calibration_state {yaml} "$SESSION_SINCE")"\n'
               'V="$(calibration_verdict "$ROW" "$STATE")"; echo "verdict=$V"\n'
               f'calibration_refusal "$V" testcard {yaml} "$STATE"',
               REPO=str(repo), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               DRY=0, BROKEN_ARMS=0, RETRY_ARMS=0, PY_BASE=sys.executable)
    assert got.returncode == 0, got.stderr
    assert "verdict=ARM UNKNOWN" in got.stdout, got.stdout
    assert "The ledger note says why:" in got.stdout
    assert ("DEFECT: the process exited 0 DONE but its RESULT lines imply 3 INVALID"
            in got.stdout), got.stdout
    assert "could not confirm speaks" not in got.stdout, "the stale one-reason paraphrase"
    assert "It exited 1" not in got.stdout
    # The other direction: a different UNKNOWN reason prints THAT note, and a
    # row with no note says so rather than inventing one.
    ledger.write_text(HEADER + "calibrate\tUNKNOWN\t0\t9\t1\t/a.log\tUNEARNED DONE: exit 0 "
                      "with no RESULT line. NOT latched.\n")
    got = lift(f'calibration_refusal "ARM UNKNOWN" testcard {yaml} PUBLISHED',
               REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               PY_BASE=sys.executable, SESSION_SINCE="20260902134501")
    assert "UNEARNED DONE: exit 0 with no RESULT line" in got.stdout, got.stdout
    ledger.write_text(HEADER + "calibrate\tUNKNOWN\t0\t9\t1\t/a.log\t\n")
    got = lift(f'calibration_refusal "ARM UNKNOWN" testcard {yaml} PUBLISHED',
               REPO=str(ROOT), LEDGER=str(ledger), LOGS=str(logs), ONLY="",
               PY_BASE=sys.executable, SESSION_SINCE="20260902134501")
    assert "(the reason was not recorded on the row)" in got.stdout, got.stdout


def test_the_contract_caveat_reads_the_rows_exit_code_and_note_rather_than_typing_1(tmp_path):
    """The second call site of F5: the CLAIM_FAIL|UNKNOWN caveat said "The
    command exited 1" for a state that can be reached at 0 and 3."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "check_mma_path.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    ledger = tmp_path / "ARMS.tsv"
    ledger.write_text(HEADER + "mma_switch\tUNKNOWN\t3\t9\t0\t/m.log\tUNEARNED INVALID: exit 3 "
                      "with no RESULT line. NOT latched.\n")
    got = lift("contract_caveat mma_switch UNKNOWN", REPO=str(repo), LEDGER=str(ledger))
    assert "The command exited 3; the ledger note reads: UNEARNED INVALID" in got.stdout, got.stdout
    assert "The command exited 1." not in got.stdout


def planted_ruler(directory: Path, clock: dict) -> Path:
    """A calibration yaml `roofline.load_hardware` accepts, carrying `clock`
    under `detail`, so `reference_clock` grades it the way it grades the real
    ones."""
    import yaml as _yaml
    doc = {"name": "testcard (measured)", "verified": True, "source": "planted",
           "memory": {"bandwidth_tb_s": 4.0},
           "compute_dense_tflops": {"bf16": 700.0},
           "detail": {"gpu_name": "testcard", **clock}}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "measured_testcard.yaml"
    path.write_text(_yaml.safe_dump(doc))
    return path


def grade_of(directory: Path) -> list[str]:
    got = lift(f"reference_grade testcard {directory}", REPO=str(ROOT), PY_BASE=sys.executable)
    assert got.returncode == 0, got.stderr
    return got.stdout.strip().split("|", 3)


def test_the_reference_grade_gate_passes_an_under_load_ruler_and_refuses_the_rest(tmp_path):
    """F4, THE DRIVER'S HALF. The calibration gate asked WHEN the yaml was
    written and whether arm 0 stood behind it, never what KIND of clock it
    carries. roofline.reference_clock grades the field it read, and only the
    under-load median may rescale the roof per row; against an idle scalar the
    driver refuses roof_at_cell_clock on every row and LEVEL is provisional,
    so nothing normalised by the clock is quotable, and no driver gate said
    so. Every grade is planted here, and the committed ruler is read rather
    than described: until 2026-09-09 this test asserted that the H200's own
    yaml was the REFUSED shape (idle-scalar, 1515 MHz), which was true of the
    2026-09-02 calibration and false the moment the 2026-09-09 one landed
    carrying `detail.gemm_clock.median_mhz`. The assertion is now that the
    committed ruler's grade agrees with what the file holds, whichever way
    that falls."""
    grade, mhz, usable, source = grade_of(
        tmp_path / "under" if planted_ruler(tmp_path / "under",
                                            {"gemm_clock": {"median_mhz": 1470}}) else None)
    assert (grade, mhz, usable) == ("under-load", "1470", "True"), (grade, mhz, usable, source)
    assert "median of the samples taken while the calibration" in source
    planted_ruler(tmp_path / "idle", {"gemm_clock_mhz": 1515})
    grade, mhz, usable, source = grade_of(tmp_path / "idle")
    assert (grade, mhz, usable) == ("idle-scalar", "1515", "False"), (grade, mhz, usable)
    assert "DISOWNED" in source
    planted_ruler(tmp_path / "settle", {"settle": {"final_mhz": 1470}})
    grade, mhz, usable, _ = grade_of(tmp_path / "settle")
    assert (grade, usable) == ("settle-plateau", "False"), (grade, mhz, usable)
    planted_ruler(tmp_path / "none", {})
    grade, mhz, usable, _ = grade_of(tmp_path / "none")
    assert (grade, mhz, usable) == ("NONE", "0", "False")
    (tmp_path / "empty").mkdir()
    grade, mhz, usable, source = grade_of(tmp_path / "empty")
    assert (grade, usable) == ("NONE", "False") and "no calibration for" in source
    # The committed H200 ruler, graded against what the file actually carries.
    # An under-load median is the one usable shape and the 2026-09-09
    # calibration carries one (1485 MHz, the clock its dense GEMM held); the
    # 2026-09-02 one carried only the idle scalar and was refused.
    import yaml as _yaml
    committed = _yaml.safe_load(
        (ROOT / "moe" / "bench" / "hardware" / "measured_nvidia_h200.yaml").read_text())
    detail = committed.get("detail") or {}
    got = lift(f"reference_grade nvidia_h200 {ROOT / 'moe' / 'bench' / 'hardware'}",
               REPO=str(ROOT), PY_BASE=sys.executable).stdout.strip().split("|", 3)
    if (detail.get("gemm_clock") or {}).get("median_mhz"):
        expect = ("under-load",
                  str(int(detail["gemm_clock"]["median_mhz"])), "True")
    elif detail.get("gemm_clock_mhz"):
        expect = ("idle-scalar", str(int(detail["gemm_clock_mhz"])), "False")
    else:
        expect = ("settle-plateau", got[1], "False")
    assert tuple(got[:3]) == expect, (got, expect)
    # And the refusal says the consequence in the words that matter.
    refusal = lift('reference_grade_refusal testcard /x/measured_testcard.yaml idle-scalar 1515 '
                   '"detail.gemm_clock_mhz = 1515 MHz, DISOWNED"',
                   REPO=str(ROOT), LOGS="/x/logs", LEDGER="/x/ARMS.tsv",
                   PY_BASE=sys.executable).stdout
    assert refusal.startswith(
        "REFUSED: the ruler for testcard carries no clock the roof can be rescaled against")
    assert "NOTHING NORMALISED BY THE CLOCK IS QUOTABLE" in refusal
    assert "usable_for_roof False" in refusal and "grade 'idle-scalar'" in refusal
    assert "calibrate_hardware.py --publish" in refusal


def test_the_reference_grade_gate_is_wired_after_the_calibration_gate_and_refuses():
    gate = CODE.split('scripts/calibrate_hardware.py" --publish', 1)[1].split("\nsay ", 1)[0]
    after = gate.split('calibration_refusal "$CALIB_VERDICT"', 1)[1]
    assert 'GRADE_LINE="$(reference_grade "$CARD" "$(dirname "$CALIB_YAML")")"' in after
    assert '[[ "${GRADE_USABLE:-}" == "True" ]]' in after
    branch = after.split("reference_grade_refusal", 1)[1]
    assert 'exit "$RC_REFUSED"' in branch.split("fi", 1)[0]
    # The PASS branch tells the operator which column and which side to read.
    assert "pct_of_roof_at_cell_clock beside pct_of_achieved_tflops" in TEXT
    assert "read clock_level_side on every LEVEL failure" in TEXT
    # The rule, in the words the instrument applies it in. Until 2026-09-09
    # this line said "only LOW or DRIFT excludes", which on this card excluded
    # the two tiles the study is about.
    assert "ONLY DRIFT EXCLUDES A ROW, since 2026-09-09" in TEXT
    assert "only LOW or DRIFT excludes" not in TEXT


#: What each KERNEL plan prints its figure as, in its own words. Three shapes,
#: because three scripts print three sentences; a fourth shape is a test
#: failure, not a fourth pattern added quietly.
PLAN_SECONDS = (r"estimate\s+(\d+) s of GPU", r"estimated GPU time (\d+) s",
                r"(\d+) s of timed kernel", r"Estimated KERNEL time (\d+) s")


def test_every_kernel_booking_is_the_ceiling_of_the_minutes_its_plan_prints(tmp_path):
    """F6. dtype was booked 6 KERNEL minutes "from ... 315 s of timed kernel"
    while its plan, run exactly as this driver runs it, prints 454 s (7.6
    min): d789b5f charged the warmup as time and the driver's three copies of
    the old figure were not updated, and no test pinned arm_minutes to the
    figure its plan prints. Every KERNEL booking is now read off the plan the
    dry run itself produced and must be that figure's ceiling in minutes."""
    session = tmp_path / "s"
    got = run(["--dry-run"], session=session)
    assert got.returncode == 0, got.stdout[-3000:]
    for name in KERNEL_ARMS:
        plan = (session / "logs" / f"{name}.log").read_text()
        hits = [m for pat in PLAN_SECONDS for m in re.finditer(pat, plan)]
        assert len(hits) == 1, (name, [h.group(0) for h in hits])
        seconds = int(hits[0].group(1))
        booked = int(lift(f"arm_minutes {shlex.quote(name)}", REPO=str(ROOT)).stdout.strip())
        assert booked == math.ceil(seconds / 60), (name, seconds, booked)
        assert str(seconds) in lift(f"arm_basis {shlex.quote(name)}",
                                    REPO=str(ROOT)).stdout, (name, seconds)
    assert lift("arm_minutes dtype", REPO=str(ROOT)).stdout.strip() == "8"
    assert "454 s of timed kernel" in (session / "logs" / "dtype.log").read_text()
    # The retired figure survives only on lines that retract it.
    for line in TEXT.splitlines():
        if "315 s" in line:
            assert "until d789b5f" in line, line


def test_the_production_arms_no_longer_promise_a_confirmation_no_card_can_give(tmp_path):
    """F7. `--list` still explained the BLOCK_N=256 refusal by its OLD reason
    (256 registers per thread against 255 at 8 warps; 256 KiB of shared memory
    against 227 at 16), said "one fix unblocks both", and the closing summary
    told the operator to read "the only arm here that can CONFIRM". The script
    itself says REFUSED AT EVERY WARP AND STAGE COUNT, 65536 of 65536 registers
    per block, NO BLOCK_SIZE_N confirms the headline on sm_90; the same command
    with --num-warps 16 --num-stages 3 also exits 2. The plan the owner books
    against may not promise a confirmation no reachable hardware can give."""
    listing = run(["--list"]).stdout
    assert not re.search(r"one fix unblocks|255 at 8 warps", listing)
    assert "NO ARM CAN CONFIRM IT ON sm_90" in listing
    assert "NO BLOCK_SIZE_N confirms the headline on this card" in listing
    assert "REFUSES AT EVERY WARP AND STAGE COUNT" in listing
    assert "65536 of 65536 registers per block" in listing
    assert "there is no fix on sm_90 that unblocks either" in listing
    assert "the only arm that can confirm it" not in listing
    body = run(["--dry-run"], session=tmp_path / "s").stdout
    block = body.split("READ THESE FOUR FIRST")[1].split("WHAT TO COMMIT")[0]
    assert "only arm here that can CONFIRM" not in block
    assert "CANNOT be confirmed on sm_90" in block
    assert "the paper has no confirming arm" in block
    # The header comments carry the same correction, and the retired reason
    # survives only on a line that retracts it.
    for line in TEXT.splitlines():
        if "255 at 8 warps" in line or "one fix unblocks" in line:
            assert "RETRACTED" in line, line
    assert "which can confirm." not in TEXT
    assert "no BLOCK_SIZE_N confirms the headline on this card" in TEXT


def test_the_help_and_the_closes_text_agree_on_what_the_counter_probe_exits():
    """F7, the other contradiction. --help said dram_counter_route.py exits DONE
    on BLOCKED and prints no RESULT line, so its exit 0 is UNEARNED; arm_closes
    said since 2026-09-03 it scores one gate per verdict and OPEN and BLOCKED
    land as the words the table gives them. Two surfaces in one file, one of
    them describing retracted behaviour, and --help is the one an operator
    reads first."""
    help_text = run(["--help"]).stdout
    assert "What it still does not do is print a RESULT line" not in help_text
    assert "its exit 0 is an UNEARNED DONE" not in help_text
    assert "OPEN lands DONE, BLOCKED lands CLAIM_FAIL" in help_text
    assert "exits 2 with no RESULT line and is re-attempted on every" in help_text
    caveat = CODE.split("contract_caveat() {", 1)[1].split("\n}\n", 1)[0]
    assert "it exits DONE on BLOCKED" not in caveat
    assert "OPEN DONE" in caveat and "BLOCKED CLAIM_FAIL" in caveat


def test_the_mma_arm_runs_under_the_drivers_vllm_interpreter():
    """F9. check_mma_path.sh chooses its interpreter from MOE_PYTHON or
    /workspace/venvs/vllm/bin/python, and the driver resolved PY_VLLM with
    fallbacks for every vLLM arm but this one: off a GPU the MMA arm refused
    on a missing /workspace path while every other arm refused on CUDA, and on
    a pod with PY_VLLM overridden it compiled under a different interpreter
    from the pin probes it is read beside."""
    words = measuring_invocation("mma_switch")
    assert words[:2] == ["arm", "mma_switch"], words
    assert words[2:5] == ["env", "MOE_PYTHON=$PY_VLLM", "bash"], words
    assert any(w.endswith("scripts/check_mma_path.sh") for w in words), words
    joined = re.sub(r"\\\n\s+", " ", CODE)
    dry = [ln for ln in joined.splitlines()
           if re.match(r"\s*arm mma_switch\s", ln) and "--dry-run" in ln]
    assert len(dry) == 1 and 'env MOE_PYTHON="$PY_VLLM" bash' in dry[0], dry


def test_the_empty_stage_array_is_guarded_for_bash_3():
    """F9's second half. `STAGES=()` followed by `"${STAGES[@]}"` is "unbound
    variable" under set -u on bash before 4.4 and would end the session at
    bn_g16. Both directions are executed on this machine's /bin/bash, which is
    3.2: the guard expands to nothing, the bare form dies."""
    assert '${STAGES[@]+"${STAGES[@]}"}' in CODE
    assert '"${STAGES[@]}"' not in CODE.replace('${STAGES[@]+"${STAGES[@]}"}', "")
    guarded = subprocess.run(
        ["/bin/bash", "-uc", 'STAGES=(); printf "[%s]" a ${STAGES[@]+"${STAGES[@]}"} b'],
        capture_output=True, text=True)
    assert guarded.returncode == 0 and guarded.stdout == "[a][b]", guarded
    filled = subprocess.run(
        ["/bin/bash", "-uc",
         'STAGES=(--num-stages 3); printf "[%s]" a ${STAGES[@]+"${STAGES[@]}"} b'],
        capture_output=True, text=True)
    assert filled.stdout == "[a][--num-stages][3][b]", filled
    version = subprocess.run(["/bin/bash", "-c", 'echo "${BASH_VERSINFO[0]}"'],
                             capture_output=True, text=True).stdout.strip()
    if version and int(version) < 4:
        bare = subprocess.run(["/bin/bash", "-uc", 'STAGES=(); printf "[%s]" a "${STAGES[@]}" b'],
                              capture_output=True, text=True)
        assert bare.returncode != 0 and "unbound variable" in bare.stderr, bare


# --------------------------------------------------------------------------
# 22. what the 2026-09-09 session changed: the arm lines, the clock probe, the
#     rerun booking, and the two sentences that named the wrong gate
# --------------------------------------------------------------------------

def test_the_cap_test_arm_carries_the_r_max_that_makes_its_own_v1_satisfiable():
    """THE ARM THAT WAS UNSATISFIABLE FROM ITS PLAN PAGE. `tile_cap_test.py`
    takes `r_max` from `depth.rows` when it is not given one; on the H200 band
    that is 688, 688 % 32 = 16 so the grid stops at 672, and the grid then holds
    two exactly-full BLOCK_M=256 stacks against V1's requirement of three
    aligned treads per tile. The 2026-09-09 run printed "BM=256:2" on its plan
    page and spent 141 s to fail V1.

    Both branches carried --r-max 1024 from 2026-09-09 until now, and that is
    the value the script's own plan-time V4 check REFUSES: it wants a 132-tile
    BLOCK_M=16 stack, 1024 gives 66, and the plan prints "raise --r-max to at
    least 2112" and no cost line at all. They carry 2112, the printed minimum,
    because r_max is in the grid, the cost and the run id."""
    joined = re.sub(r"\\\n\s+", " ", CODE)
    lines = [ln for ln in joined.splitlines()
             if re.match(r"\s*arm cap_test\s", ln)]
    assert len(lines) == 2, lines
    for ln in lines:
        assert "--r-max 2112" in ln, ln
    measuring = [ln for ln in lines if "--dry-run" not in ln]
    assert len(measuring) == 1 and "--fail-on-gate" in measuring[0], measuring
    # And the booking is the figure that plan prints, not the one it replaced.
    assert lift("arm_minutes cap_test", REPO=str(ROOT)).stdout.strip() == "5"
    basis = lift("arm_basis cap_test", REPO=str(ROOT)).stdout
    assert "--r-max 2112" in basis and "242 s" in basis, basis
    assert "688" in basis, "the row does not say what the default did"
    assert "143 s" in basis and "2026-09-09" in basis, (
        "the row does not retract the 1024 booking it replaced")


def test_the_alias_arm_refuses_at_the_probe_rather_than_buying_a_bound_twice():
    """--dot-fallback refuse on all three branches, and the dot lower bound
    named as the separate booking it is. The allow branch was bought on
    2026-09-09: 308 s, P1 UNKNOWN at alpha >= 0.229, latched INVALID (rc 3,
    four validity gates failed after measuring)."""
    joined = re.sub(r"\\\n\s+", " ", CODE)
    lines = [ln for ln in joined.splitlines()
             if re.match(r"\s*arm alias_ablation\s", ln)]
    assert len(lines) == 3, lines
    for ln in lines:
        assert "--dot-fallback refuse" in ln, ln
    closes = " ".join(lift("arm_closes alias_ablation", REPO=str(ROOT)).stdout.split())
    for needle in ("--cell-budget-ms 200", "--replicates 18", "33.4 WALL min",
                   "5500 GB/s", "6151"):
        assert needle in closes, needle


def test_the_depth_arm_books_r_max_and_leaves_the_partner_to_the_script():
    """Without a small tile in the SWEPT set the arm's own non-vacuity floor is
    0.838 of the roof and no BLOCK_M=256 ladder in the corpus reaches it, so the
    arm refuses its own reference on every card and under every clock rule: it
    is pre-registered INVALID, which is what 292 s bought on 2026-09-09.

    The partner that fixes that is UNCONDITIONAL in `scripts/bm128_depth.py`
    (`SMALL_TILE_BLOCK_M`, in `BLOCK_SIZES`) and there is no flag for it. Both
    arm lines carried `--partner-block-m 32` until now, which argparse rejects,
    so this asserts the flag is gone from both and that the driver still says
    where the partner actually lives. `--r-max` IS a flag and is still on both
    lines, because it is in the grid, the cost and the run id."""
    joined = re.sub(r"\\\n\s+", " ", CODE)
    lines = [ln for ln in joined.splitlines()
             if re.match(r"\s*arm bm128_depth\s", ln)]
    assert len(lines) == 2, lines
    for ln in lines:
        assert "--partner-block-m" not in ln, ln
        assert "--r-max 2048" in ln, ln
    # And the reason is on the page rather than in a commit message.
    assert "THE BM=32 PARTNER IS NOT A FLAG" in TEXT
    unpriced = lift("arm_unpriced bm128_depth", REPO=str(ROOT)).stdout
    assert "BM=32 scaling partner" in unpriced, unpriced
    assert "is not a flag" in unpriced, unpriced


def test_the_clock_probe_checks_what_it_says_and_survives_pipefail():
    """IT PRINTED THE OPPOSITE OF WHAT IT MEASURED. `nvidia-smi -q -d CLOCK |
    grep -q ...` under `set -o pipefail` returns 141 on a MATCH, because grep
    exits at the first hit and nvidia-smi takes SIGPIPE writing the rest: on
    2026-09-09 the session reported "does NOT report clocks" on a pod where the
    query reports four. And it probed the wrong reader: every LEVEL and DRIFT
    verdict comes from `torch.cuda.clock_rate()` in the arm's own interpreter,
    not from nvidia-smi. The shipped matcher is lifted and run over a large
    planted query, and the pipeline version is re-planted beside it to show the
    two disagree."""
    matcher = re.search(r'if \[\[ "\$SMI_CLOCK_Q" =~ ([^\n]+) \]\]; then', TEXT)
    assert matcher, "the clock probe no longer matches with bash's own regex"
    body = (
        'set -uo pipefail\n'
        'big() { for i in $(seq 1 5000); do '
        'echo "        Graphics                          : 1980 MHz"; done; }\n'
        'SMI_CLOCK_Q="$(big)"\n'
        f'if [[ "$SMI_CLOCK_Q" =~ {matcher.group(1)} ]]; then echo SHIPPED_OK; '
        'else echo SHIPPED_MISSED; fi\n'
        'if big | grep -qE "Graphics[[:space:]]*:[[:space:]]*[0-9]+ MHz"; '
        'then echo PIPE_OK; else echo "PIPE_MISSED $?"; fi\n')
    done = subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                          timeout=120)
    assert "SHIPPED_OK" in done.stdout, done.stdout
    assert "PIPE_MISSED 141" in done.stdout, (
        "the pipefail artefact no longer reproduces; if bash changed, the "
        "shipped probe must still be the one that does not use a pipeline")
    # No pipeline in the shipped probe, and the reader the arms use is asked.
    probe = TEXT.split("# CLOCK VISIBILITY")[1].split("# The session and results root")[0]
    assert "nvidia-smi -q -d CLOCK 2>/dev/null | grep" not in probe
    assert "torch.cuda.clock_rate()" in probe
    assert "$PY_VLLM" in probe, "the probe asks an interpreter no arm runs"


def test_the_next_session_booking_names_new_and_a_state_per_arm():
    """AN INVALID ROW IS LATCHED AND NO RESUME RE-RUNS IT, so a session that
    landed six INVALID arms resumes into nothing. The closing summary prints
    the --new command and what each arm is expected to reach, priced through
    the same `arm_minutes` and `arm_clock` as the cost table so a re-booked arm
    moves it by itself."""
    arms = lift("rerun_arms", REPO=str(ROOT)).stdout.split()
    assert arms == ["calibrate", "pin_probe-n64-g1", "roofline-n64-g1",
                    "cap_test", "bn_g16", "dtype", "bm128_depth",
                    "alias_ablation"], arms
    out = lift("next_session_booking", REPO=str(ROOT)).stdout
    assert "--new" in out and "--only " + ",".join(arms) in out, out
    priced, bound = lift(f"session_bound {' '.join(arms)}",
                         REPO=str(ROOT)).stdout.split()
    assert f"~{priced} priced / ~{bound} bounded minutes" in out, (out, priced)
    for arm in arms:
        expectation = lift(f"rerun_expectation {shlex.quote(arm)}",
                           REPO=str(ROOT)).stdout.strip()
        assert expectation, arm
        assert expectation in " ".join(out.split()), arm
    # The one arm expected to exit 1 is named as a RESULT, not as a failure.
    assert "CLAIM_FAIL, and that is the arm's result" in out
    assert "re-runs no INVALID and no CLAIM_FAIL row" in out


def test_the_commit_advice_names_the_ruler_gate_that_exists_and_its_polarity():
    """TWO THINGS WRONG IN ONE SENTENCE. It said "arm 9 (ruler) P1 FAILED" and
    `ruler_rebaseline.py` scores V1-V3 and C1-C5 with no P1 at all, so the
    operator was sent to a gate that is not on the page; and it read the verdict
    backwards. C1 is "the GEMM clock was sampled in the wrong state", gated
    above a 5% post-hoc-versus-under-load delta, so a PASS is the arm
    sustaining the charge and a FAIL is the designed null. On 2026-09-09 C1
    read FAIL at a measured delta of 0.0%, which is the state in which the
    calibration is committable."""
    ruler = (ROOT / "scripts" / "ruler_rebaseline.py").read_text()
    assert "P1" not in re.sub(r"P1[0-9]", "", ruler), "ruler now has a P1 gate"
    body = TEXT.split("WHAT TO COMMIT, AND WHAT NOT TO")[1]
    assert "arm 9 (ruler) C1 PASSED" in body, body[:1500]
    assert "P1 FAILED" not in body
    assert "a C1 FAIL is the designed null" in " ".join(body.split()).lower() or \
        "A C1 FAIL is the designed null" in " ".join(body.split()), body[:1500]


def test_the_summary_says_a_resume_will_not_re_run_an_invalid_row(tmp_path):
    """The latch is right and the advice around it was not: "to resume it" was
    the only route named, and a resume re-runs no INVALID row."""
    got = run(["--dry-run"], session=tmp_path / "s")
    flat = " ".join(got.stdout.split())
    assert "A RESUME RE-RUNS NO INVALID ROW AND NO CLAIM_FAIL ROW" in flat, flat[-3000:]
    assert "The next session for those arms is --new" in flat
