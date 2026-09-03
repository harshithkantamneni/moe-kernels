"""The ISA-switch arm speaks the shared exit-code table, in bash.

`scripts/check_mma_path.sh` is the driver's `mma_switch` arm. It scores four
named gates -- G1, G2 and G3 VALIDITY, G4 CLAIM -- and until 2026-09-02 it
declared its OWN three-code table at the top of the file ("0 every gate passed.
1 a gate failed. 2 refused before measuring") and never looked at
`moe/bench/exit_codes.py`. Two of the three states it named were inverted
against it:

  * a VALIDITY failure exited 1. 1 is CLAIM_FAIL, "measured, valid, and the
    world disagreed with the prediction", which the session records as a RESULT
    and LATCHES -- beside this script's own text saying the census may not be
    quoted. The state it meant is INVALID (3);
  * the two refusals, no interpreter at `$PY` and no PTX dumped, exited 1 as
    well, so a free precondition failure was ledgered as a refuted claim;
  * and the four gates printed no `RESULT: ` line at all, so the session
    summary reported four scored gates as "This arm was NOT scored".

It is bash and cannot import the module, so it MIRRORS it, and mirrors drift.
This file is the thing that stops them: the five integers are compared with the
module's constants, the shell's `result_line` is compared with
`exit_codes.result_line` character for character, and the shell's `classify` is
compared with `exit_codes.classify` over the same gate lists.

WHY THERE IS A `--self-test`. Every one of those defects lives in a branch that
only runs on a rented H200 with two real compiles in it. The script therefore
scores a PLANTED world off GPU, through the SHIPPED scorers, and three of its
six worlds plant a FAILING gate -- a gate that has only ever been seen passing
is the shape of check this apparatus is being repaired for. Each world asserts
the identity that matters: `classify_text` over what the script printed equals
the integer the process returned.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MMA = ROOT / "scripts" / "check_mma_path.sh"
TEXT = MMA.read_text()

sys.path.insert(0, str(ROOT))
from moe.bench import exit_codes as EC  # noqa: E402

#: name -> exit code the mirror declares.
CODES = {"EXIT_DONE": EC.DONE, "EXIT_CLAIM_FAIL": EC.CLAIM_FAIL,
         "EXIT_REFUSED": EC.REFUSED, "EXIT_INVALID": EC.INVALID,
         "EXIT_ERROR": EC.ERROR}

#: `--self-test <world>` -> the code the table says that world exits with.
WORLDS = {"forced-pass": EC.DONE, "forced-invalid": EC.INVALID,
          "forced-claim": EC.CLAIM_FAIL, "ladder-pass": EC.DONE,
          "ladder-refuted": EC.CLAIM_FAIL, "ladder-invalid": EC.INVALID}


def sh(*argv, env_extra=None, cwd=None):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "HOME": os.environ.get("HOME", "/tmp")}
    env.update(env_extra or {})
    return subprocess.run(["bash", str(MMA), *argv], cwd=str(cwd or ROOT),
                          capture_output=True, text=True, timeout=900, env=env)


def lift(script: str, **variables):
    """Evaluate the script's LIFTABLE block, then `script`, in a fresh bash.

    The same device `tests/test_h200_gaps_session.py` uses on the driver, and
    for the same reason: this asks the SHIPPED function rather than a python
    copy of its case statement, which would agree with it until it did not.
    Every global the block reads is passed explicitly, so a test that forgets
    one gets an unbound-variable error rather than a value left lying around.
    """
    setup = "\n".join(f"{k}={v!r}" for k, v in variables.items())
    body = (f"set -uo pipefail\n{setup}\n"
            f"eval \"$(sed -n '/^# >>> LIFTABLE/,/^# <<< LIFTABLE/p' \"{MMA}\")\"\n"
            f"{script}\n")
    return subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                          timeout=120)


def lifted(script: str):
    """`lift` with the five table constants already in scope."""
    return lift(script, **{name: code for name, code in CODES.items()})


# --------------------------------------------------------------------------
# the mirror
# --------------------------------------------------------------------------

def test_the_script_parses():
    done = subprocess.run(["bash", "-n", str(MMA)], capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_the_five_integers_are_the_modules():
    for name, want in CODES.items():
        found = re.search(rf"^{name}=(\d+)$", TEXT, re.M)
        assert found, f"{name} is not declared in the mirror"
        assert int(found.group(1)) == want, name


def test_it_no_longer_declares_a_table_of_its_own():
    """Two things: the header names the module as the authority, and no exit in
    the file chooses an integer of its own.

    The old sentence still appears once, inside the paragraph that repudiates
    it, which is deliberate: the header explains what was wrong. So this checks
    the executable statements rather than the prose. No `exit` in the file
    carries a literal any more -- every one of them names a mirrored constant or
    the code `classify` returned -- which is what makes the constants the single
    place a drift can happen, and the test above the thing that catches it.
    """
    assert "moe.bench.exit_codes" in TEXT
    exits = re.findall(r"^\s*exit (\S+)", TEXT, re.M)
    assert exits, "the script exits somewhere"
    allowed = {f'"${name}"' for name in CODES} | {'"$rc"'}
    assert set(exits) <= allowed, sorted(set(exits) - allowed)
    quoted = [ln for ln in TEXT.splitlines()
              if "0 every gate passed. 1 a gate failed." in ln]
    assert all(ln.lstrip().startswith("#") for ln in quoted), quoted


def test_the_shells_result_line_is_the_modules_character_for_character():
    """The format is the contract with the driver's summary, which keys on the
    `RESULT: ` prefix at column zero and on nothing else."""
    cases = [
        (EC.VALIDITY, "G1", EC.PASS, "[VALIDITY] every arm ran the tile it was given"),
        (EC.CLAIM, "G4", EC.FAIL, "measured 1/2 arms | gate all 2"),
        (EC.VALIDITY, "L1", EC.UNKNOWN, "  ragged   spacing\tand a tab  "),
        (EC.VALIDITY, "G3", EC.PASS, "2 distinct PTX checksum(s) | gate == 2"),
        (EC.CLAIM, "L2", EC.PASS, ""),
    ]
    for kind, name, verdict, detail in cases:
        got = lifted(f"result_line {kind} {name} {verdict} "
                     f"{shlex.quote(detail)}")
        assert got.returncode == 0, got.stderr
        want = EC.result_line(kind, name, verdict, " ".join(detail.split()))
        assert got.stdout.rstrip("\n") == want
        # And it survives the round trip the driver actually performs.
        back = EC.parse_result_lines(got.stdout)
        assert len(back) == 1 and back[0].name == name


def test_the_shells_result_line_refuses_what_the_parser_cannot_read_back():
    """The planted FAIL branch of the renderer. A name with a space in it, or a
    verdict outside the table, produces a line `parse_result_lines` skips, and a
    gate the driver cannot parse is a gate that disappears from the summary."""
    for bad in ("result_line VALIDITY 'G 1' PASS detail",
                "result_line VALIDITY G1 passed detail",
                "result_line WARNING G1 PASS detail",
                "result_line VALIDITY '' PASS detail"):
        got = lifted(bad)
        assert got.returncode == EC.ERROR, (bad, got.stdout, got.stderr)
        assert "result_line:" in got.stderr
        assert EC.parse_result_lines(got.stdout) == []


def test_the_shells_classify_is_the_modules_rule():
    """Including the two the module argues for at length: UNKNOWN counts against
    a gate exactly as FAIL does, and a VALIDITY failure outranks a CLAIM one."""
    tables = [
        [(EC.VALIDITY, "G1", EC.PASS), (EC.CLAIM, "G4", EC.PASS)],
        [(EC.VALIDITY, "G1", EC.PASS), (EC.CLAIM, "G4", EC.FAIL)],
        [(EC.VALIDITY, "G1", EC.FAIL), (EC.CLAIM, "G4", EC.PASS)],
        [(EC.VALIDITY, "G1", EC.FAIL), (EC.CLAIM, "G4", EC.FAIL)],
        [(EC.VALIDITY, "G1", EC.UNKNOWN)],
        [(EC.CLAIM, "G4", EC.UNKNOWN)],
        [(EC.VALIDITY, "G1", EC.PASS), (EC.VALIDITY, "G2", EC.PASS),
         (EC.VALIDITY, "G3", EC.PASS), (EC.CLAIM, "G4", EC.PASS)],
    ]
    for table in tables:
        calls = "\n".join(
            f"gate {name} {kind} {verdict} claim measured threshold consequence"
            for kind, name, verdict in table)
        got = lifted(f"{calls}\nclassify")
        assert got.returncode == 0, got.stderr
        assert int(got.stdout.strip().splitlines()[-1]) == EC.classify(
            [(kind, verdict) for kind, _name, verdict in table]), table


def test_classifying_nothing_is_not_done():
    """`classify([])` raises in the module rather than returning DONE, because
    "a check that examined nothing reports zero failures" is this project's
    documented failure shape. The shell's answer is a non-zero return, which
    `finish` turns into ERROR."""
    got = lifted("classify")
    assert got.returncode != 0
    assert "no gate was scored" in got.stderr
    with pytest.raises(EC.NoGatesScored):
        EC.classify([])


# --------------------------------------------------------------------------
# the planted worlds
# --------------------------------------------------------------------------

@pytest.mark.parametrize("world,want", sorted(WORLDS.items()))
def test_the_code_the_process_returns_is_the_code_its_result_lines_imply(world,
                                                                        want):
    got = sh("--self-test", world)
    out = got.stdout + got.stderr
    assert got.returncode == want, out
    assert EC.classify_text(out) == got.returncode, out


def test_every_gate_prints_exactly_one_result_line_and_nothing_else_does():
    forced = sh("--self-test", "forced-pass")
    lines = EC.parse_result_lines(forced.stdout)
    assert [(r.kind, r.name, r.verdict) for r in lines] == [
        (EC.VALIDITY, "G1", EC.PASS), (EC.VALIDITY, "G2", EC.PASS),
        (EC.VALIDITY, "G3", EC.PASS), (EC.CLAIM, "G4", EC.PASS)]
    assert forced.stdout.count("RESULT: ") == 4
    ladder = sh("--self-test", "ladder-pass")
    assert [(r.kind, r.name) for r in EC.parse_result_lines(ladder.stdout)] == [
        (EC.VALIDITY, "L1"), (EC.CLAIM, "L2")]


def test_the_self_test_plants_the_failing_branch_of_every_gate():
    """Three worlds, three failing verdicts, and each has to be the gate that
    world names -- otherwise the scorer is being exercised on its passing path
    with a different label on it."""
    def verdicts(world):
        return {(r.name, r.verdict)
                for r in EC.parse_result_lines(sh("--self-test", world).stdout)}

    assert ("G1", EC.FAIL) in verdicts("forced-invalid")
    assert ("G4", EC.FAIL) in verdicts("forced-claim")
    assert ("L2", EC.FAIL) in verdicts("ladder-refuted")
    # UNKNOWN is not a pass: nothing compiled, so L1 fails and L2 decided
    # nothing, and the arm is INVALID rather than a refuted claim.
    assert {("L1", EC.FAIL), ("L2", EC.UNKNOWN)} <= verdicts("ladder-invalid")


def test_a_claim_that_did_not_hold_is_a_result_and_not_a_retry():
    got = sh("--self-test", "forced-claim")
    assert got.returncode == EC.CLAIM_FAIL
    assert EC.ledger_state(got.returncode) == "CLAIM_FAIL"
    assert "quote the table, not the prediction" in got.stdout


def test_a_validity_failure_is_invalid_and_says_it_is_not_re_run():
    got = sh("--self-test", "forced-invalid")
    assert got.returncode == EC.INVALID
    assert EC.ledger_state(got.returncode) == "INVALID"
    assert "Nothing on this page may be quoted" in got.stdout
    assert "NOT auto-retried" in got.stdout


def test_an_unplanned_failure_is_error_rather_than_whatever_the_shell_returned():
    """A missing command exits 127, and a `grep` that matches nothing exits 1 =
    CLAIM_FAIL. Neither is a verdict, so the ERR trap names both ERROR."""
    got = sh("--self-test", "unhandled")
    assert got.returncode == EC.ERROR, got.stdout + got.stderr
    assert "UNHANDLED FAILURE" in got.stderr
    assert EC.parse_result_lines(got.stdout) == [], "a crash scores no gate"
    assert EC.ledger_state(got.returncode) == "RETRY"


def test_an_unknown_planted_world_refuses():
    got = sh("--self-test", "no-such-world")
    assert got.returncode == EC.REFUSED
    assert "REFUSED" in got.stderr


# --------------------------------------------------------------------------
# the real paths, as far as a laptop can drive them
# --------------------------------------------------------------------------

def test_a_missing_interpreter_refuses_free_rather_than_reporting_a_result(tmp_path):
    """The precondition failure the driver hits on any box without the vLLM
    venv. It exited 1 = CLAIM_FAIL, which the session ledgers as finished, so
    the arm could not be re-run once the venv existed."""
    got = sh("--model", "deepseek-v3", "--tokens", "256", "--block-m", "16,64",
             "--out", str(tmp_path / "ptx"),
             env_extra={"MOE_PYTHON": "/nonexistent/python"})
    assert got.returncode == EC.REFUSED, got.stdout + got.stderr
    assert "REFUSED: no interpreter" in got.stderr
    assert EC.parse_result_lines(got.stdout) == [], "a refusal scores no gate"
    assert not (tmp_path / "ptx").exists(), "a refusal writes nothing"


def test_the_sweeps_own_refusal_becomes_this_scripts_refusal(tmp_path):
    """Off the GPU box no framework span registers, so the first arm's sweep
    refuses under MOE_FORCE_TILE (exit 2, nothing measured) and this script
    adopts that code instead of reporting a census it never took. Before the
    repair the failing pipeline reached the ERR path and was reported as 4."""
    got = sh("--model", "toy", "--tokens", "8", "--block-m", "16,64",
             "--out", str(tmp_path / "ptx"),
             env_extra={"MOE_PYTHON": sys.executable})
    out = got.stdout + got.stderr
    assert got.returncode == EC.REFUSED, out
    assert "sweep refused before measuring" in out
    assert EC.parse_result_lines(got.stdout) == []


def test_the_ladder_arm_is_scored_instead_of_exiting_zero_unexamined(tmp_path):
    """The unforced cell used to exit 0 having scored nothing, which is DONE --
    "measured; every gate PASSED" -- out of a run where no gate existed to
    pass. On a laptop nothing compiles, so L1 fails and the arm is INVALID."""
    got = sh("--model", "toy", "--tokens", "8", "--out", str(tmp_path / "ptx"),
             env_extra={"MOE_PYTHON": sys.executable})
    out = got.stdout + got.stderr
    assert got.returncode == EC.INVALID, out
    assert EC.classify_text(out) == got.returncode
    assert ("L1", EC.FAIL) in [(r.name, r.verdict)
                               for r in EC.parse_result_lines(out)]


def test_the_dry_run_refuses_rather_than_reporting_every_gate_passed():
    """A plan scores no gate, so it prints no RESULT line, and `classify_text`
    over a log with none raises NoGatesScored -- which the module documents as
    exactly what a REFUSED log looks like from there. DONE would read "measured;
    every VALIDITY and CLAIM gate PASSED" one line under "nothing was executed".
    The driver re-queues neither state: `dry_state` maps 2 to PLAN_REFUSED."""
    got = sh("--block-m", "16,64", "--tokens", "256", "--dry-run")
    assert got.returncode == EC.REFUSED
    assert EC.parse_result_lines(got.stdout) == []
    assert "REFUSED: --dry-run" in got.stdout
    with pytest.raises(EC.NoGatesScored):
        EC.classify_text(got.stdout)


def test_the_driver_sees_this_arm_as_having_adopted_the_table():
    """`h200_gaps_session.sh:adopts_exit_codes` greps each arm's own file for
    the module's dotted name and prints a caveat beside every REFUSED or INVALID
    row that came out of a file which does not carry it: "the state word may not
    mean what this session reads it to mean". Asked of the SHIPPED function, in
    the shipped driver, because that grep is the thing that has to stop firing
    for this arm."""
    driver = ROOT / "scripts" / "h200_gaps_session.sh"
    body = (f"set -uo pipefail\nREPO={str(ROOT)!r}\n"
            f"eval \"$(sed -n '/^# >>> LIFTABLE/,/^# <<< LIFTABLE/p' \"{driver}\")\"\n"
            "adopts_exit_codes scripts/check_mma_path.sh; echo \"rc=$?\"\n")
    got = subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                         timeout=120)
    assert got.stdout.strip() == "rc=0", got.stdout + got.stderr


def stub_interpreter(tmp_path, name, body):
    """An `$PY` that fails the way the pod's does, without a pod.

    The sweep's own exit code is the only thing this script reads out of the
    child, and the two states that matter here -- a traceback that escaped
    `moe.bench.cli`, and a scorer that returned 1 -- cannot be planted from the
    laptop's real interpreter. So the interpreter is the plant.
    """
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def test_a_child_that_crashed_is_error_and_not_a_latched_invalid(tmp_path):
    """1 out of `moe.bench.cli` means a traceback, not a verdict.

    That module registers only VALIDITY gates, so `classify` over them returns
    DONE or INVALID and never CLAIM_FAIL, and every other return in it is 0 or
    2. Reading 1 as "it ran and then failed a gate of its own" printed that
    sentence over a log with no RESULT line in it and exited 3 -- INVALID, which
    `arm()` latches -- so one OOM marked this arm measured-and-unquotable for
    good. ERROR is the code that lets the session try again.
    """
    py = stub_interpreter(tmp_path, "crashpy",
                          'echo "Traceback (most recent call last):" >&2\n'
                          'echo "MemoryError" >&2\nexit 1\n')
    got = sh("--model", "toy", "--tokens", "8", "--block-m", "16,64",
             "--out", str(tmp_path / "ptx"), env_extra={"MOE_PYTHON": str(py)})
    out = got.stdout + got.stderr
    assert got.returncode == EC.ERROR, out
    assert EC.ledger_state(got.returncode) == "RETRY"
    assert "having scored no gate" in got.stderr
    assert "failed a gate of" not in out, "no RESULT line exists to point at"
    assert EC.parse_result_lines(got.stdout) == [], "a crash scores no gate"


def test_a_child_that_scored_its_own_gates_is_adopted_rather_than_retried(tmp_path):
    """The other half of the same branch, so the discriminator is exercised in
    both directions. A 1 whose log carries `RESULT: ` lines came out of a scorer
    and is a verdict about the pin, which is what this script attributes its
    census to, so it is adopted as INVALID and not queued for another rental."""
    py = stub_interpreter(tmp_path, "scoredpy",
                          "echo 'RESULT: VALIDITY F1 FAIL [VALIDITY] planted "
                          "| measured 1 | gate 0'\nexit 1\n")
    got = sh("--model", "toy", "--tokens", "8", "--block-m", "16,64",
             "--out", str(tmp_path / "ptx"), env_extra={"MOE_PYTHON": str(py)})
    out = got.stdout + got.stderr
    assert got.returncode == EC.INVALID, out
    assert "from its own scorer" in got.stderr
    assert "having scored no gate" not in got.stderr
