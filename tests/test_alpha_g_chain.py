"""scripts/alpha_g_chain.sh, checked without a pod.

The chain sequences the alpha(G) matrix session: preflight, the driver's
preconditions, the counter probe (informational), tests/test_gpu.py on the
card, the alignment probe's on-card check from the vLLM venv, the ratio at
seed 0 at every G (the first run's V8 read before anything else is bought),
the clock elasticity at every G at duty states 1.0 0.5 0.25, the later seeds
scored with the earlier ones at --duty 0.25, and, only when END_SUITE=run asks
for it, the whole suite at the end as a record. What this file pins: the three
shell habits this project has been burned by, the ledger's second opinion (the
driver's rule, lifted), every gate asking for DONE and not for "latched",
overrides that hold for every later pass, every arm under a hang cap off its
own price, the session a pass lands in, the card and the duty it is measured
at and the lock it holds, pairing only with reports that formed a ratio, a
skip worded by seed 0's V7 verdict and the follow-up a V7 FAIL prints, tables
rebuilt from the reports on disk (per run, and per G through R3's own
cross-run machinery) with their legend, the elasticity band read through the
arm's own `band_of` and withheld from a page its gates refused, what each
ratio reads as off that band, the bytes-rate bound per G at the ceilings of
the ruler the session measured, a counter probe that finds ncu off PATH too
and never gates or latches, and a laptop dry run that prices every step off
its own source and writes nothing into the tree.

The measuring path is driven end to end off GPU through two planted
interpreters: a base one that names a planted card and UUID, prints a planted
plan page for an arm's --dry-run, runs dram_counter_route.py's REAL do_probe
over a planted box for --probe, and is the real interpreter for everything
else; and an arm one that prints the plan line and the RESULT lines and writes
report.json, which is all the chain reads, and emulates
`private_weight_reference.py --probe-check` (one RESULT line; exit 0, 3, or 2
when it refuses).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scripts" / "alpha_g_chain.sh"
HELPERS = ROOT / "scripts" / "alpha_g_chain_helpers.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import alpha_g_chain_helpers as H  # noqa: E402
from _hermetic import laptop_env  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

CODE = CHAIN.read_text()
HEADER = "step\tstate\trc\tseconds\tdirty\tlog\tnote\n"
ARMS_HEADER = "arm\tstate\trc\tseconds\tdirty\tlog\tnote\n"
#: Every variable the chain reads from its environment, off its own line.
_KNOBS_LINE = re.search(r'^CHAIN_KNOBS="([^"]+)"$', CODE, re.M)
KNOBS = _KNOBS_LINE.group(1).split() if _KNOBS_LINE else []


def chain_env(**extra) -> dict:
    """laptop_env without the chain's knobs, then `extra`. On the pod this file
    runs inside the chain's end suite when END_SUITE=run buys it, and an
    operator's SESSION=<dir> or END_SUITE in the caller's environment must not
    steer a chain spawned here into the real session."""
    env = {k: v for k, v in laptop_env().items() if k not in KNOBS}
    env.update(extra)
    return env


def lift(script: str, **variables) -> subprocess.CompletedProcess:
    """Evaluate the chain's LIFTABLE block, then `script`, the driver's way."""
    setup = "\n".join(f"{k}={v!r}" for k, v in variables.items())
    body = (f"set -uo pipefail\n"
            f'eval "$(sed -n \'/^# >>> LIFTABLE/,/^# <<< LIFTABLE/p\' "{CHAIN}")"\n'
            f"{setup}\n{script}\n")
    return subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                          timeout=300, env=chain_env(REPO=str(ROOT)))


def _ledger(path: Path, *rows, header: str = HEADER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "".join("\t".join(r) + "\n" for r in rows))
    return path


def _row(name, state, rc="0", note=""):
    return (name, state, rc, "1", "0", "x.log", note)


# --------------------------------------------------------------------------
# the shell itself
# --------------------------------------------------------------------------

def test_the_chain_parses_and_avoids_the_three_habits():
    assert subprocess.run(["bash", "-n", str(CHAIN)], capture_output=True).returncode == 0
    assert re.search(r"^set -uo pipefail$", CODE, re.M)
    assert not re.search(r"^set -[a-z]*e", CODE, re.M), "set -e would abort a rented session"
    assert "errexit" not in CODE
    assert "$!" not in CODE, "a PID variable in a shell that starts no background job"
    body = CODE.split("\nrun_step_as() {", 1)[1].split("\n}", 1)[0]
    assert '"$@" > "$log" 2>&1 || rc=$?' in body, "the measured command's rc is captured directly"
    assert "| tee" not in body


def test_the_registered_duty_is_one_number_in_its_three_homes():
    """The owner's decision D1 (2026-09-22) lives in the chain's R3_DUTY
    default, the driver's standalone R3 arm (`private_duty`) and R3's own
    FLAT_DUTY, which its V7 remedy names. Three literals, so a relation."""
    import private_weight_reference as PW
    chain = re.search(r'^R3_DUTY="\$\{R3_DUTY:-([0-9.]+)\}"$', CODE, re.M)
    driver = re.search(r"^private_duty\(\) \{ echo ([0-9.]+); \}$",
                       (ROOT / "scripts" / "h200_gaps_session.sh").read_text(), re.M)
    assert chain and driver
    assert float(chain.group(1)) == float(driver.group(1)) == PW.FLAT_DUTY


def test_the_state_word_mirrors_the_exit_code_table():
    for rc, word in exit_codes.CODE_NAMES.items():
        got = lift(f"state_word {rc}")
        assert got.stdout.strip() == word, (rc, got.stdout)
    assert lift("state_word 9").stdout.strip() == "UNKNOWN"


@pytest.mark.parametrize("rc,implied,state,needle", [
    (0, "0", "DONE", "log agrees"),
    (1, "1", "CLAIM_FAIL", "log agrees"),
    (3, "3", "INVALID", "log agrees"),
    (0, "1", "UNKNOWN", "DEFECT: page implies 1"),
    (0, "NONE", "UNKNOWN", "UNEARNED DONE"),
    (2, "NONE", "REFUSED", ""),
    (3, "NONE", "UNKNOWN", "UNEARNED INVALID"),
    (4, "NONE", "ERROR", "crash"),
    (0, "UNREADABLE", "UNKNOWN", "SECOND OPINION UNAVAILABLE"),
])
def test_the_second_opinion_is_the_drivers_rule(rc, implied, state, needle):
    got = lift(f"state_for {rc} {implied}")
    word, _, note = got.stdout.rstrip("\n").partition("\t")
    assert word == state, got.stdout
    assert needle in note


def test_a_step_is_latched_only_on_a_result_state(tmp_path):
    ledger = _ledger(tmp_path / "CHAIN.tsv", _row("r1-g1", "REFUSED", "2"),
                     _row("r1-g1", "DONE"), _row("r3-g1-s0", "UNKNOWN"),
                     _row("r3-g4-s0", "CLAIM_FAIL", "1"), _row("r3-g4-s1", "SKIPPED", "-"))
    assert lift(f"latched r1-g1 {ledger!s} && echo yes").stdout.strip() == "yes"
    assert lift(f"latched r3-g4-s0 {ledger!s} && echo yes").stdout.strip() == "yes"
    assert lift(f"latched r3-g1-s0 {ledger!s} || echo no").stdout.strip() == "no"
    skipped = lift(f"latched r3-g4-s1 {ledger!s} || echo no")
    assert skipped.stdout.strip() == "no", "SKIPPED asks again"
    assert lift(f"latched never-ran {ledger!s} || echo no").stdout.strip() == "no"
    assert lift(f"newest_state r1-g1 {ledger!s}").stdout.strip() == "DONE"
    assert lift(f"newest_state never-ran {ledger!s}").stdout.strip() == ""
    assert lift(f"newest_state x {tmp_path / 'absent.tsv'!s}; echo rc=$?").stdout.strip() == "rc=0"


def test_run_step_writes_the_row_with_the_second_opinion_taken(tmp_path):
    ledger = _ledger(tmp_path / "CHAIN.tsv")
    page = tmp_path / "page.py"
    page.write_text("import sys\n"
                    "print('RESULT: CLAIM C1 FAIL [CLAIM] x | measured 0.9 | gate y')\n"
                    "sys.exit(1)\n")
    got = lift(f"run_step r3-g1-s0 {tmp_path / 'a.log'!s} {sys.executable} {page!s}; echo rc=$?",
               LEDGER=str(ledger))
    assert "rc=1" in got.stdout, got.stdout + got.stderr
    rows = ledger.read_text().splitlines()[1:]
    assert len(rows) == 1
    name, state, rc, _secs, _dirty, _log, note = rows[0].split("\t")
    assert (name, state, rc) == ("r3-g1-s0", "CLAIM_FAIL", "1")
    assert "log agrees" in note
    # exit 0 with no RESULT line is not a DONE anybody earned
    silent = tmp_path / "silent.py"
    silent.write_text("print('nothing scored')\n")
    lift(f"run_step preflight-r1 {tmp_path / 'b.log'!s} {sys.executable} {silent!s}",
         LEDGER=str(ledger))
    last = ledger.read_text().splitlines()[-1].split("\t")
    assert last[1] == "UNKNOWN" and "UNEARNED DONE" in last[6]


@pytest.mark.skipif(shutil.which("timeout") is None,
                    reason="no timeout(1) here: the chain runs its arms uncapped on such a box")
def test_an_arm_that_outlives_its_cap_is_an_unlatched_timed_out_error(tmp_path):
    """GPU-2. A timed-out arm is ERROR with TIMED OUT in its note, not latched,
    whatever RESULT lines it printed before the cap: both arms resume per
    cell, so --resume re-runs it."""
    ledger = _ledger(tmp_path / "CHAIN.tsv")
    hang = tmp_path / "hang.py"
    hang.write_text("import time\n"
                    "print('RESULT: VALIDITY V7 PASS [VALIDITY] x | measured y | gate z',"
                    " flush=True)\n"
                    "time.sleep(120)\n")
    got = lift(f"arm_step r3-g1-s0 {tmp_path / 'a.log'!s} 2 {sys.executable} {hang!s}; echo rc=$?",
               LEDGER=str(ledger))
    assert "rc=124" in got.stdout, got.stdout + got.stderr
    row = _rows(ledger)[-1]
    assert row[:3] == ["r3-g1-s0", "ERROR", "124"]
    assert "TIMED OUT after" in row[6] and "against a cap of 2 s" in row[6]
    assert lift(f"latched r3-g1-s0 {ledger!s} || echo no").stdout.strip() == "no"
    # under its cap, an arm is scored by its page as before
    page = tmp_path / "page.py"
    page.write_text("import sys\n"
                    "print('RESULT: CLAIM C1 FAIL [CLAIM] x | measured 0.9 | gate y')\n"
                    "sys.exit(1)\n")
    lift(f"arm_step r3-g4-s0 {tmp_path / 'b.log'!s} 60 {sys.executable} {page!s}",
         LEDGER=str(ledger))
    assert _rows(ledger)[-1][:3] == ["r3-g4-s0", "CLAIM_FAIL", "1"]


def test_an_arms_cap_is_a_multiple_of_its_own_price_with_a_floor():
    factor, floor = _const("ARM_CAP_FACTOR"), _const("ARM_CAP_FLOOR_S")
    for est in (1, floor // factor + 1, 876, 5000):
        cap, how = lift(f"cap_for {est}").stdout.rstrip("\n").split("\t")
        assert int(cap) == max(factor * est, floor), (est, cap, how)
        assert str(est) in how
    cap, how = lift('cap_for ""').stdout.rstrip("\n").split("\t")
    assert int(cap) == _const("ARM_CAP_UNPRICED_S") and "no price" in how


@pytest.mark.parametrize("rc,state,latches", [
    (0, "DONE", True), (2, "REFUSED", False), (3, "UNKNOWN", False), (4, "UNKNOWN", False),
])
def test_the_preconditions_step_latches_only_on_the_drivers_exit_0(tmp_path, rc, state, latches):
    """The driver exits 3 when a row is still owed and 4 when an arm crashed;
    both re-attempt on --resume only if the chain runs the driver again. Until
    2026-09-22 exit 3 was filed INVALID, which latched, so an owed pin_probe
    was never re-attempted although the driver said --resume would."""
    ledger = _ledger(tmp_path / "CHAIN.tsv")
    got = lift(f'run_step_as driver preconditions {tmp_path / "p.log"!s} bash -c "exit {rc}";'
               f" echo rc=$?", LEDGER=str(ledger))
    assert f"rc={rc}" in got.stdout, got.stdout + got.stderr
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert (row[0], row[1], row[2]) == ("preconditions", state, str(rc))
    if rc in (3, 4):
        assert f"driver exit {rc}" in row[6] and "--resume re-runs the driver" in row[6]
    again = lift(f"latched preconditions {ledger!s} && echo yes || echo no")
    assert again.stdout.strip() == ("yes" if latches else "no")


# --------------------------------------------------------------------------
# the gates: each asks for DONE, never for "latched"
# --------------------------------------------------------------------------

@pytest.mark.parametrize("r1,r3,goes", [
    ("DONE", "DONE", True),
    ("INVALID", "DONE", False),         # a scorer that failed its planted world: latched, not DONE
    ("DONE", "CLAIM_FAIL", False),
    ("DONE", None, False),
])
def test_the_preflight_gate_asks_for_done_not_latched(tmp_path, r1, r3, goes):
    rows = [_row("preflight-r1", r1, "3" if r1 == "INVALID" else "0")]
    if r3:
        rows.append(_row("preflight-r3", r3))
    ledger = _ledger(tmp_path / "CHAIN.tsv", *rows)
    got = lift("preflight_gate; echo rc=$?", LEDGER=str(ledger), LOGS=str(tmp_path))
    assert got.stdout.strip().endswith("rc=0" if goes else "rc=3"), got.stdout + got.stderr
    if not goes:
        bad = "preflight-r1" if r1 != "DONE" else "preflight-r3"
        assert f"STOP: {bad} is {r1 if r1 != 'DONE' else (r3 or 'absent')}, not DONE" in got.stdout
        assert "Fix the scorer" in got.stdout and "runs again on every pass" in got.stdout, \
            "the STOP names the way out"


@pytest.mark.parametrize("pre,thermal,calibrate,pin,goes,needle", [
    ("DONE", "DONE", "DONE", "DONE", True, ""),
    # pin_probe is recorded, never gated: neither arm reads MOE_FORCE_TILE
    ("UNKNOWN", "DONE", "DONE", "UNKNOWN", True, ""),
    ("DONE", "DONE", "DONE", None, True, ""),
    # the driver's own calibration or grade gate refused, with ARMS.tsv reading DONE
    ("REFUSED", "DONE", "DONE", None, False, "STOP: the driver REFUSED this card or its ruler"),
    ("DONE", "DONE", "INVALID", None, False, "the driver's calibrate row is INVALID, not DONE"),
    ("DONE", "DONE", "CLAIM_FAIL", None, False,
     "the driver's calibrate row is CLAIM_FAIL, not DONE"),
    ("DONE", "CLAIM_FAIL", None, None, False, "the driver's thermal row is CLAIM_FAIL, not DONE"),
    ("DONE", "DONE", None, None, False, "the driver's calibrate row is absent, not DONE"),
])
def test_the_preconditions_gate(tmp_path, pre, thermal, calibrate, pin, goes, needle):
    ledger = _ledger(tmp_path / "CHAIN.tsv",
                     _row("preconditions", pre, "2" if pre == "REFUSED" else "0"))
    rows = [_row(n, s) for n, s in (("thermal", thermal), ("calibrate", calibrate),
                                    ("pin_probe-n64-g1", pin)) if s]
    arms = _ledger(tmp_path / "ARMS.tsv", *rows, header=ARMS_HEADER)
    (tmp_path / "preconditions.log").write_text(
        "REFUSED: arm 0 did not stand behind a ruler: calibrate INVALID\n")
    got = lift(f"preconditions_gate {arms!s}; echo rc=$?", LEDGER=str(ledger), LOGS=str(tmp_path))
    assert got.stdout.strip().endswith("rc=0" if goes else "rc=3"), got.stdout + got.stderr
    assert needle in got.stdout
    if pre == "REFUSED":
        assert "arm 0 did not stand behind a ruler" in got.stdout, "the driver's own reason, quoted"


# --------------------------------------------------------------------------
# tests/test_gpu.py before the arms, gated; the whole suite after them, a record
# --------------------------------------------------------------------------

def _fake_pytest(tmp_path, name, tally, rc):
    page = tmp_path / name
    page.write_text(f"import sys\nprint('F..s')\nprint({tally!r})\nsys.exit({rc})\n")
    return page


@pytest.mark.parametrize("rc,tally,state,needle", [
    (0, "35 passed, 3 skipped in 100.00s (0:01:40)", "DONE",
     "pytest exit 0: 35 passed, 3 skipped"),
    # off a card every GPU test skips and pytest exits 0: that is not green
    (0, "38 skipped in 0.03s", "UNKNOWN", "UNEARNED DONE: pytest exit 0 but no test passed"),
    (1, "== 3 failed, 4769 passed, 57 skipped in 3050.12s (0:50:50) ==", "ERROR",
     "pytest exit 1: 3 failed, 4769 passed, 57 skipped in 3050.12s (0:50:50)"),
    (5, "no tests ran in 0.01s", "ERROR", "pytest exit 5: no tally in the log"),
    (124, "2 failed, 3000 passed in 5400.00s", "ERROR", "pytest TIMED OUT (exit 124)"),
])
def test_a_pytest_row_is_its_exit_code_and_tally(tmp_path, rc, tally, state, needle):
    """pytest prints no RESULT line, so the page's second opinion cannot apply:
    exit 0 with a test passed is DONE, anything else is not (never a latched
    word, so --resume re-runs it), and the note carries pytest's own tally."""
    ledger = _ledger(tmp_path / "CHAIN.tsv")
    page = _fake_pytest(tmp_path, "suite.py", tally, rc)
    got = lift(f"run_step_as suite gpu-tests {tmp_path / 's.log'!s} {sys.executable} {page!s};"
               f" echo rc=$?", LEDGER=str(ledger))
    assert f"rc={rc}" in got.stdout, got.stdout + got.stderr
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert (row[0], row[1], row[2]) == ("gpu-tests", state, str(rc))
    assert needle in row[6], row[6]
    assert "\t" not in row[6]
    latched = lift(f"latched gpu-tests {ledger!s} && echo yes || echo no")
    assert latched.stdout.strip() == ("yes" if state == "DONE" else "no")


def _priced(n: int, rate: float) -> int:
    """The chain's rounding, awk's `n * r + 0.5` truncated: half UP. Python's
    round() is half to even and disagrees at every count that is 25 mod 100
    at 0.66 s a test (4925 x 0.66 = 3250.5)."""
    return math.floor(n * rate + 0.5)


def test_a_collected_count_is_priced_half_up(tmp_path):
    log = tmp_path / "collect.log"
    log.write_text("tests/test_a.py::test_x\n\n4925 tests collected in 3.00s\n")
    got = lift(f"price_tests {log!s}", SUITE_S_PER_TEST="0.66")
    assert got.stdout.split() == ["4925", str(_priced(4925, 0.66))], got.stdout + got.stderr
    assert _priced(4925, 0.66) != round(4925 * 0.66), "the count the old assertion flaked on"


def test_the_pod_rate_is_the_quotient_of_the_run_it_cites():
    """SUITE_S_PER_TEST is a measured rate: it is checked against the run its
    comment cites (session 5's end suite, tests and seconds), not typed twice.
    The comment's first "N tests in S s" is that run, and where the tree
    carries the log it names, that log's own tally is the figure quoted."""
    block = re.search(r'#: The price per collected test ON A POD.*?\n'
                      r'SUITE_S_PER_TEST="\$\{SUITE_S_PER_TEST:-([0-9.]+)\}"', CODE, re.S)
    assert block, "the rate's comment and line"
    # the comment as one line: a wrapped path joins without a space
    text = re.sub(r"\n#: ", " ", re.sub(r"/\n#: ", "/", block.group(0)))
    n, secs = map(int, re.search(r"(\d+) tests in (\d+) s", text).groups())
    assert "session 5" in text.split(f"{n} tests in")[0]
    assert float(block.group(1)) == round(secs / n, 2), (block.group(1), secs, n)
    cited = re.search(r"(results/published/\S+/suite\.log)", text)
    assert cited, "the rate names the log it was read off"
    log = ROOT / cited.group(1)
    if log.exists():
        tally = log.read_text().strip().splitlines()[-1]
        ran = sum(int(k) for k in re.findall(r"(\d+) (?:failed|passed|skipped|errors?)\b", tally))
        took = float(re.search(r" in ([\d.]+)s", tally).group(1))
        assert (ran, round(took)) == (n, secs), tally


def test_a_red_gpu_tests_page_stops_the_chain_before_any_arm(tmp_path):
    ledger = _ledger(tmp_path / "CHAIN.tsv",
                     _row("gpu-tests", "ERROR", "1", "pytest exit 1: 3 failed"))
    got = lift("gpu_tests_gate 0; echo rc=$?", LEDGER=str(ledger), LOGS=str(tmp_path))
    assert "rc=3" in got.stdout, got.stdout + got.stderr
    assert "STOP: tests/test_gpu.py is ERROR, not green, on this card" in got.stdout
    assert "--resume --past-gpu-tests" in got.stdout
    assert "override" not in ledger.read_text()


def test_past_gpu_tests_goes_on_and_the_ledger_records_the_decision(tmp_path):
    ledger = _ledger(tmp_path / "CHAIN.tsv", _row("gpu-tests", "UNKNOWN", "0", "UNEARNED DONE"))
    got = lift("gpu_tests_gate 1; echo rc=$?", LEDGER=str(ledger), LOGS=str(tmp_path))
    assert "rc=0" in got.stdout, got.stdout + got.stderr
    row = ledger.read_text().splitlines()[-1].split("\t")
    assert (row[0], row[1]) == ("gpu-tests-override", "OVERRIDDEN")
    assert "--past-gpu-tests" in row[6] and "tests/test_gpu.py's newest row: UNKNOWN" in row[6]
    never = _ledger(tmp_path / "never" / "CHAIN.tsv")
    lift("gpu_tests_gate 1", LEDGER=str(never), LOGS=str(tmp_path))
    assert "newest row: never ran" in never.read_text().splitlines()[-1]


def test_a_green_gpu_tests_page_passes_without_an_override_row(tmp_path):
    ledger = _ledger(tmp_path / "CHAIN.tsv", _row("gpu-tests", "ERROR", "1"),
                     _row("gpu-tests", "DONE", "0", "pytest exit 0: 35 passed"))
    for past in (0, 1):
        got = lift(f"gpu_tests_gate {past}; echo rc=$?", LEDGER=str(ledger), LOGS=str(tmp_path))
        assert "rc=0" in got.stdout, got.stdout + got.stderr
    assert "override" not in ledger.read_text()


def test_the_tests_interpreter_must_not_import_vllm(tmp_path):
    """pod_session.sh P11c's rule: from a venv with vLLM an unplanted bare
    --run inside a test would MEASURE."""
    with_vllm = tmp_path / "py-with-vllm"
    with_vllm.write_text("#!/bin/sh\nexit 0\n")        # every import succeeds
    without = tmp_path / "py-without"
    without.write_text("#!/bin/sh\nexit 1\n")          # `import vllm` fails
    for f in (with_vllm, without):
        f.chmod(0o755)
    assert lift(f"suite_interpreter_ok {with_vllm!s} || echo refused").stdout.strip() == "refused"
    assert lift(f"suite_interpreter_ok {without!s} && echo ok").stdout.strip() == "ok"


def test_chain_knobs_names_every_variable_the_chain_takes_from_its_environment():
    """A knob missing from CHAIN_KNOBS would reach the end suite's tests. The
    chain's knobs are its self-defaulting assignments (`X="${X:-...}"`), SESSION
    (read as `${SESSION:-}` where the session is chosen) and CAPABILITY (read
    inline by a dry run); PY_BASE and PY_VLLM are replaced by the tests'
    laptop_env, and CARD is emptied before the probe fills it."""
    defaulted = set(re.findall(r'^\s*([A-Z0-9_]+)="\$\{\1:-', CODE, re.M))
    assert '"${SESSION:-}"' in CODE and "${CAPABILITY:-" in CODE
    assert set(KNOBS) == (defaulted - {"PY_BASE", "PY_VLLM", "CARD"}) | {"SESSION", "CAPABILITY"}


def test_the_opinion_is_an_argument_and_reaches_no_child(tmp_path, monkeypatch):
    """`OPINION=suite run_step ...` exported OPINION to every child of the
    step, so on the pod the end suite's own run of this file scored six of its
    rows with the suite's opinion and recorded them FAILED. The opinion is an
    argument now: no child sees one, and one in the chain's environment
    decides nothing."""
    monkeypatch.delenv("OPINION", raising=False)
    ledger = _ledger(tmp_path / "CHAIN.tsv")
    log = tmp_path / "probe.log"
    lift(f"run_step_as suite gpu-tests {log!s} bash -c"
         " 'echo opinion=${OPINION:-unset}; echo \"1 passed in 0.01s\"'", LEDGER=str(ledger))
    assert "opinion=unset" in log.read_text(), log.read_text()
    assert _rows(ledger)[-1][:2] == ["gpu-tests", "DONE"]
    page = tmp_path / "page.py"
    page.write_text("import sys\n"
                    "print('RESULT: CLAIM C1 FAIL [CLAIM] x | measured 0.9 | gate y')\n"
                    "sys.exit(1)\n")
    got = lift(f"export OPINION=suite; run_step r3-g1-s0 {tmp_path / 'a.log'!s}"
               f" {sys.executable} {page!s}", LEDGER=str(ledger))
    assert _rows(ledger)[-1][:3] == ["r3-g1-s0", "CLAIM_FAIL", "1"], got.stdout + got.stderr


def test_a_pytest_step_runs_without_the_chains_knobs(tmp_path, monkeypatch):
    """The suite's tests spawn this chain and the driver with os.environ merged
    in. An operator's `SESSION=<dir> bash scripts/alpha_g_chain.sh` left
    SESSION exported to the end suite, whose chain tests then wrote their
    ledgers into that directory and recorded 14 spurious failures."""
    monkeypatch.delenv("OPINION", raising=False)
    probe = tmp_path / "py"
    probe.write_text('#!/bin/bash\n[[ "$1" == -c ]] && exit 1\n'        # `import vllm` fails
                     'env | sed "s/=.*//" | sort > "$(dirname "$0")/seen.txt"\n'
                     'echo "5 passed in 1.00s"\n')
    probe.chmod(0o755)
    ledger = _ledger(tmp_path / "CHAIN.tsv")
    exports = " ".join(f'export {k}="${{{k}:-planted}}";' for k in KNOBS)
    got = lift(f"{exports} pytest_step suite 60 tests/",
               PY_BASE=str(probe), LEDGER=str(ledger), LOGS=str(tmp_path))
    seen = set((tmp_path / "seen.txt").read_text().split())
    assert "PATH" in seen and "MOE_RESULTS_DIR" in seen, "the environment, not an empty one"
    assert seen.isdisjoint([*KNOBS, "OPINION"]), sorted(seen & {*KNOBS, "OPINION"})
    assert _rows(ledger)[-1][:2] == ["suite", "DONE"], got.stdout + got.stderr


def test_the_order_is_the_owners_and_only_the_pre_arm_checks_are_gated():
    """D3: preflight, preconditions, the counter probe (informational, never
    gated), tests/test_gpu.py (gated), the probe check from the vLLM venv
    (gated), R3 seed 0 at every G with the pilot's V8 read, R1 at every G, R3's
    later seeds, then, on END_SUITE=run, the whole suite uncapped, from
    PY_BASE, gating nothing."""
    marks = ['echo "== preflight', 'echo "== preconditions', 'echo "== counter-probe',
             'echo "== tests/test_gpu.py', 'echo "== the alignment probe under the graph',
             'echo "== the ratio, shared over private, off the cap, seed',
             'echo "== clock elasticity',
             'echo "== the ratio, shared over private, the later seeds', 'echo "== the whole suite']
    at = [CODE.index(m) for m in marks]
    assert at == sorted(at), dict(zip(marks, at, strict=True))
    counter = CODE[at[2]:at[3]]
    assert 'counter_probe_step "$COUNTER_PROBE_S"' in counter
    assert "stop_chain" not in counter and "exit " not in counter, "the counter probe gates nothing"
    gpu = CODE[at[3]:at[4]]
    assert "tests/test_gpu.py -q -rfE -p no:cacheprovider" in gpu
    assert 'gpu_tests_gate "$PAST_GPU_TESTS" || stop_chain' in gpu
    probe = CODE[at[4]:at[5]]
    assert "$(probe_check_cmd)" in probe and 'probe_check_gate "$PAST_V8" || stop_chain' in probe
    assert re.search(r'^probe_check_cmd\(\) \{\n  echo "\$PY_VLLM" "\$REPO/scripts/private_weight_'
                     r'reference\.py" --probe-check \\\n\s+--model mixtral-8x7b --block-m 32\n\}',
                     CODE, re.M), "the NEW INTERFACE's command line, from the vLLM venv"
    pilot = CODE[at[5]:at[6]]
    assert 'v8_gate "$PAST_V8" || stop_chain' in pilot
    suite = CODE[at[8]:CODE.index("# 10. the table")]
    assert ('pytest_step suite "$SUITE_TIMEOUT_S" tests/ -q -rfE --durations=25'
            ' -p no:cacheprovider') in suite
    assert "--maxfail" not in suite and "-x " not in suite, "the suite on the pod is uncapped"
    assert "stop_chain" not in suite and "exit 3" not in suite, "the end suite gates nothing"
    assert 'R3_DUTY="${R3_DUTY:-0.25}"' in CODE


def test_help_prints_the_whole_header_and_no_code():
    got = subprocess.run(["bash", str(CHAIN), "--help"], capture_output=True, text=True,
                         timeout=60, env=chain_env(REPO=str(ROOT)))
    assert got.returncode == 0
    for flag in ("--past-gpu-tests", "--past-v8", "--new", "--resume", "END_SUITE=run"):
        assert flag in got.stdout, flag
    assert "OFF BY DEFAULT. END_SUITE=run buys the whole suite" in got.stdout
    assert "WHAT IT RUNS" in got.stdout and "THE REGIME WORD" in got.stdout
    for word in ("RAW-STANDS", "UNREGISTERED-GAP", "CLOCK-CARRIES", "STRADDLES", "withheld:<EXIT>"):
        assert word in got.stdout, word
    assert "CLOCK-CARRIES is only an upper bound" in got.stdout, "finding 28's caveat"
    assert "RUNS FOR THE RECORD AND IS NOT GATED" in got.stdout
    assert "nohup setsid bash scripts/alpha_g_chain.sh" in got.stdout
    assert ">> /workspace/alpha_g_chain.out 2>&1" in got.stdout, "a --resume appends"
    assert "> /workspace/alpha_g_chain.out" not in got.stdout.replace(">>", ""), \
        "no launch line truncates the console"
    assert "AN OVERRIDE HOLDS" in got.stdout and "probe-check" in got.stdout
    flat = " ".join(_header_prose().split())
    assert ("A V7 FAIL at 0.25 means that duty is not yet flat for that arm on this card; the"
            " page names a lower duty; the chain skips the G's later seeds and prints the"
            " follow-up command") in flat, "the owner's reading of a V7 FAIL (XS-1)"
    assert "a finding about the card" not in CODE
    assert "does NOT cover the graph probe R3's V8 stands on" in flat, "GPU-1: no false cover"
    assert "checkout -B r3-align origin/r3-align" in got.stdout
    assert "no PID variable" in got.stdout, "the header's last line"
    assert "set -uo pipefail" not in got.stdout


def _header_prose() -> str:
    """The header's comment, one line of prose: wrapped sentences are found
    whole."""
    head = CODE.split("\nset -uo pipefail\n", 1)[0]
    return " ".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#")
                    for ln in head.splitlines()[1:])


def test_the_resolution_sentence_is_session_4s_own_rescore():
    """T3. The header's figures for R1's resolution are session 4's G=16 cells
    re-scored by the arm's own fit at its default bootstrap, not typed."""
    import clock_elasticity as CE
    cells = ROOT / "tests" / "fixtures" / "2026-09-21-nvidia_h200-session4-clock_elasticity-g16"
    rows = CE.read_rows(cells / "cells.csv")

    def fitted(duties):
        want = {CE._duty_key(d) for d in duties}
        return CE.fit([r for r in rows if CE._duty_key(r.duty_requested) in want],
                      draws=CE.DEFAULT_DRAWS, seed=0)
    three, four = fitted((1.0, 0.5, 0.25)), fitted((1.0, 0.5, 0.25, 0.1))
    text = _header_prose()
    assert (f"half-width of {(three.hi - three.lo) / 2:.3f} over its states 1.0, 0.5 and 0.25"
            f" ({(four.hi - four.lo) / 2:.3f} over all four; the all-tread reading's was"
            f" {(four.per_tile_all_treads_hi - four.per_tile_all_treads_lo) / 2:.3f}) against"
            f" R1's {CE.RESOLUTION_TARGET:.3f} target") in text
    assert (three.hi - three.lo) / 2 > CE.RESOLUTION_TARGET, "the sentence's premise"


#: Session 5's R1 pages, where pod-h200-session5 publishes them: the directory
#: the header and the runbook cite for what R1 read at the chain's states.
S5_R1 = ("results/published/2026-09-23-nvidia_h200-session5/results/gaps-nvidia_h200/"
         "clock_elasticity/")
#: Its pages per G, by the run id that ends each directory's name, in the order
#: the texts give them: G=16's first page (INVALID on V5) was kept beside its re-run.
S5_R1_PAGES = {"1": ["0d8858eb"], "4": ["e1c429b7"],
               "16": ["a5a8fde2.first-v5-invalid", "a5a8fde2"], "64": ["a3d5cd3a"]}


def _s5_reading(g: str, pages: list[str]) -> str:
    """What the texts must say session 5's R1 read at G, off its own pages:
    each half-width, the word through the chain's own eta, and the edge a
    STRADDLES crossed or the validity gates that withheld the word."""
    import clock_elasticity as CE
    reps = []
    for page in pages:
        found = sorted((ROOT / S5_R1).glob(f"*-{page}"))
        assert len(found) == 1, (page, found)
        reps.append(found[0] / "report.json")
    loaded = [json.loads(r.read_text()) for r in reps]
    assert all(p["duty"] == [float(d) for d in _default("R1_DUTY").split()] for p in loaded), \
        "session 5's pages ran at the chain's states"
    halves = [f"{(p['elasticity']['hi'] - p['elasticity']['lo']) / 2:.4f}" for p in loaded]
    words = [H.eta(r)[3] for r in reps]
    if len(reps) == 1:
        tail = words[0]
        if tail == "STRADDLES":
            lo, hi = loaded[0]["elasticity"]["lo"], loaded[0]["elasticity"]["hi"]
            tail += " at " + " and ".join(f"{e:.2f}" for e in (CE.BAND_LOW, CE.BAND_HIGH)
                                          if lo < e < hi)
        return f"{halves[0]} at G={g} ({tail})"
    assert len(set(words)) == 1, words
    failed = [[f"V{gate['number']}" for gate in p["gates"]
               if gate.get("kind") == "VALIDITY" and gate.get("verdict") != "PASS"]
              for p in loaded]
    assert all(len(f) == 1 for f in failed), failed
    return (f"{' then '.join(halves)} at G={g} ({words[0]} both times, on"
            f" {' and then on '.join(f[0] for f in failed)})")


def test_the_resolution_paragraph_states_what_session_5_read_at_the_chains_states():
    """The header and the runbook expected STRADDLES at the chain's states
    ('not expected to resolve better', 'Expect STRADDLES'). Session 5 ran R1 at
    exactly those states, and G=4 and G=64 read CLOCK-CARRIES. Both texts say
    what it read and name its pages; where the tree carries them, every
    half-width, word, edge and failed gate is the page's own."""
    header = _header_prose().replace("/ ", "/")
    runbook = " ".join(_runbook_chain_section().split())
    ladder = re.search(r'^G_LADDER="\$\{G_LADDER:-([^}]*)\}"$', CODE, re.M).group(1).split()
    assert list(S5_R1_PAGES) == ladder, "one reading per G of the chain's ladder"
    names = [" then ".join(p) for p in S5_R1_PAGES.values()]
    pages = f"its pages {', '.join(names[:-1])}, and {names[-1]} under"
    for name, text in (("header", header), ("runbook", runbook)):
        for stale in ("not expected to resolve better", "Expect STRADDLES", "hard to reach"):
            assert stale not in text, (name, stale)
        assert "Session 5 ran R1 at those three states, the chain's, at every G" in text, name
        assert pages in text and S5_R1 in text, name
        for g, run in S5_R1_PAGES.items():
            if (ROOT / S5_R1).is_dir():
                assert _s5_reading(g, run) in text, (name, g, _s5_reading(g, run))
            else:
                halves = r" then ".join([r"\d\.\d{4}"] * len(run))
                assert re.search(rf"{halves} at G={g} \(", text), (name, g)


def test_the_secant_caveat_is_first_order_and_says_where_it_fails():
    """T4. A per-tile cost A + B/f gives an elasticity in [0, 1]; session 4's
    G=16 claim read above 1, where the caveat has no model under it."""
    text = _header_prose()
    assert "To first order, for a per-tile cost A + B/f" in text
    assert "cannot produce an elasticity above 1" in text
    assert "an hour or more apart" not in CODE, "the dry run prints the spacing"
    assert "SEED SPACING" in CODE


def test_no_text_says_r1_would_pool_two_cards():
    """XS-3. R1 carries its own UUID guard (device_guard, 858bf35); the chain
    said R1 resumes by card NAME and would pool two cards' cells."""
    import clock_elasticity as CE
    assert callable(CE.device_guard)
    for stale in ("resumes by card NAME", "keys its resume on the card NAME",
                  "R1's run id carries no UUID, so its resume would pool"):
        assert stale not in CODE, stale
    assert "R1's device_guard" in _header_prose()


def _runbook_chain_section() -> str:
    text = (ROOT / "docs" / "POD_RUNBOOK.md").read_text()
    start = text.index("## The alpha(G) chain")
    return text[start:text.index("\n## ", start + 1)]


def test_the_runbook_chain_section_says_what_the_chain_does():
    """Every behaviour this file pins has its runbook sentence: the appending
    launch, the probe check, overrides that hold, the V7 wording and its
    follow-up, the per-G table, the flock fallback, the caps."""
    sec = _runbook_chain_section()
    flat = " ".join(sec.split())
    assert "nohup setsid bash scripts/alpha_g_chain.sh >> /workspace/alpha_g_chain.out" in sec
    assert "> /workspace/alpha_g_chain.out" not in sec.replace(">>", "")
    assert "--probe-check" in sec and "skips from PY_BASE" in flat
    assert "graph primitives" not in flat
    assert "holds for every later pass" in flat
    assert ("V7 FAIL at 0.25 means that duty is not yet flat for that arm on this card; the"
            " page names a lower duty; the chain skips the G's later seeds and prints the"
            " follow-up command") in flat
    assert "a finding about the card" not in flat
    assert "could not be scored" in flat and "--duty 0.1" in flat
    assert "PAIRS-by-G.tsv" in sec and "exit_scope" in sec and "PAIRS-README.txt" in sec
    assert "ps -eo pid,etime,args" in sec
    assert "timeout --signal=INT --kill-after=60" in sec
    assert "resumes by card name" not in flat and "an hour or more apart" not in flat
    assert "makes every later ratio page INVALID" in flat and "SEEDS=0" in sec
    assert "To first order" in flat
    # session 5's three defaults, each where the runbook states it
    assert "The whole suite, only on `END_SUITE=run`: it is off by default" in flat
    assert "`END_SUITE=skip` drops" not in flat
    assert "three duty states (1.0, 0.5, 0.25" in flat and "(1.0, 0.7, 0.5)" not in flat
    assert "the clocks at 0.5 and 0.25" in flat and "clock at 0.7 and 0.5" not in flat
    rate = re.search(r'^SUITE_S_PER_TEST="\$\{SUITE_S_PER_TEST:-([0-9.]+)\}"$', CODE, re.M)
    assert f"session 5's pod rate of {rate.group(1)} s a test" in flat
    assert "0.66 s a test" not in flat


def test_the_help_says_what_each_regime_word_reads_as_and_the_bound():
    """The header's regime words say what the ratio beside each reads as,
    which both tables print, and CLOCK-CARRIES is not a time ratio but a
    blend (session 5's findings, 4.2 and 9); the bytes-rate bound is named
    with its form."""
    prose = " ".join(_header_prose().split())
    for label in (*H.READS_AS.values(), H.READS_AS_UNRESOLVED):
        assert label in prose, label
    assert "the ratio is a time ratio" not in prose, "a blend, not a time ratio (findings 9)"
    assert "the ratio is NOT alpha" in prose
    assert "alpha <= (t x C / W - 1) / (n - 1)" in prose
    assert "each rebuild prints it per G with the ceiling it used" in prose


def test_the_help_records_the_counter_probe_and_runpods_counter_history():
    """The counter probe's place and rules, where it looks (NCU_SEARCH's own
    default, not a second copy), and RunPod's record: the ERR_NVGPUCTRPERM
    refusals this repo commits, and session 4's probe finding no ncu on PATH."""
    got = subprocess.run(["bash", str(CHAIN), "--help"], capture_output=True, text=True,
                         timeout=60, env=chain_env(REPO=str(ROOT)))
    assert "counter-probe" in got.stdout and "ERR_NVGPUCTRPERM" in got.stdout, "--help prints it"
    flat = " ".join(_header_prose().split()).replace("/ ", "/")   # a wrapped path, joined
    runs = flat[flat.index("WHAT IT RUNS"):flat.index("EVERY ARM STEP")]
    assert runs.index("preconditions") < runs.index("counter-probe") < runs.index("gpu-tests")
    assert "counter-probe INFORMATIONAL: NEVER GATED AND NEVER LATCHED" in flat
    globs = _default("NCU_SEARCH")
    assert f"at NCU_SEARCH's globs ({' and '.join(globs.split())} by default)" in flat
    assert "scripts/dram_counter_route.py --probe, the driver's counter_plan probe" in flat
    for word in ("OPEN", "BLOCKED", "ABSENT", "UNTESTED", "ERROR"):
        assert word in runs, word
    for fact in ("two rented H200s attempted a counter read and both were refused with"
                 " ERR_NVGPUCTRPERM", "and on 2026-09-15, on a pod holding"
                 " neither CAP_SYS_ADMIN nor CAP_PERFMON",
                 "the 2026-09-09 and 2026-09-10 pods were never asked",
                 "in session 4 (2026-09-21) ncu was absent from the image as far as the probe"
                 " looked, which was PATH alone",
                 '"no ncu on PATH"', "gaps-nvidia_h200-20260921T235000Z/counter_route.json",
                 "session 5 attempted none", "Two refused pods are a record, not a fact about"
                 " the platform"):
        assert fact in flat, fact
    assert "COUNTERS.json" in flat and "$SESSION/COUNTERS" in flat
    assert "the counter probe runs on every pass (its INFO row is never latched)" in flat
    assert "The counter probe runs under the same rule over its price" in flat


def _committed_refusals() -> list[tuple[str, str]]:
    """Every ERR_NVGPUCTRPERM refusal a file under profiles/ commits, as (the
    file, the date its run's own rows carry): read off the log and the rows
    its run wrote, never typed. The 2026-09-15 refusal is committed only as
    prose (docs/FINDINGS.md's retraction), so it is not in this list."""
    found = []
    for log in sorted((ROOT / "profiles").glob("*.txt")):
        text = log.read_text(errors="replace")
        if "ERR_NVGPUCTRPERM" not in text:
            continue
        rows = re.search(r"wrote \d+ rows -> (\S+\.csv)", text)
        assert rows, f"{log.name} names no rows file to date it by"
        with open(ROOT / rows.group(1), newline="") as fh:
            row = next(csv.DictReader(fh))
        assert "H200" in row["gpu_name"], (log.name, row["gpu_name"])
        found.append((str(log.relative_to(ROOT)), row["timestamp"][:10]))
    return found


def test_runpods_counter_record_counts_every_refusal_this_repo_commits():
    """The chain's header (its --help) and every runbook passage that states
    RunPod's ncu record name each refusal a profile log commits, by its file
    and by its run's own date, and none calls the record one refusal on one
    pod: profiles/q2_kernel_names.txt holds an ERR_NVGPUCTRPERM from ncu over
    the harness's own CLI on an H200, three weeks before 2026-09-15."""
    refusals = _committed_refusals()
    assert refusals, "profiles/ commits the earlier refusal this record counts"
    header = " ".join(_header_prose().split()).replace("/ ", "/")
    runbook = " ".join((ROOT / "docs" / "POD_RUNBOOK.md").read_text().split())
    chain = " ".join(_runbook_chain_section().split())
    p10 = runbook[runbook.index("| P10 |"):runbook.index("| P11a |")]
    pnsys = runbook[runbook.index("### P-nsys, and why it moved to the front"):]
    pnsys = pnsys[:pnsys.index("question this project has never answered")]
    start = runbook.index("| `ncu` says ERR_NVGPUCTRPERM |")
    playbook = runbook[start:runbook.index("| override_config appears", start)]
    for name, text in (("header", header), ("chain section", chain), ("P10", p10),
                       ("P-nsys", pnsys), ("playbook", playbook)):
        for path, day in refusals:
            assert day in text, (name, day)
            if name != "P-nsys":
                assert path in text, (name, path)
        assert "2026-09-15" in text, name
        for stale in ("One refusal on one pod", "one refusal on one pod",
                      "one H200 refused", "one rented H200 refused"):
            assert stale not in text, (name, stale)


def test_the_runbook_states_the_reads_as_rule_and_the_bound():
    sec = " ".join(_runbook_chain_section().split())
    for label in (*H.READS_AS.values(), H.READS_AS_UNRESOLVED):
        assert label in sec, label
    assert "a time ratio, a blend of traffic and clock" not in sec
    assert "alpha <= (t x C / W - 1) / (n - 1)" in sec and "`ruler=<yaml>`" in sec
    assert "Both tables carry `reads_as`" in sec


def test_the_runbook_states_the_counter_probe_and_no_platform_fact():
    """The runbook's chain section and its counter rows say what the chain now
    does, and none of them states one pod's refusal as the platform's."""
    sec = " ".join(_runbook_chain_section().split())
    assert "The counter probe, INFORMATIONAL: never gated and never latched" in sec
    assert "`INFO`" in sec
    for glob_ in _default("NCU_SEARCH").split():
        assert f"`{glob_}`" in sec, glob_
    assert "2026-09-15" in sec and "ERR_NVGPUCTRPERM" in sec and "no ncu on PATH" in sec
    assert "`COUNTERS`" in sec and "`COUNTERS.json`" in sec
    text = " ".join((ROOT / "docs" / "POD_RUNBOOK.md").read_text().split())
    for stale in ("`ncu` fails on a rented pod with `ERR_NVGPUCTRPERM`",
                  "`ncu` is walled off on a rented pod by `ERR_NVGPUCTRPERM`"):
        assert stale not in text, stale
    start = text.index("| `ncu` says ERR_NVGPUCTRPERM |")
    playbook = text[start:text.index("| override_config appears", start)]
    assert "counter-probe" in playbook, "the playbook row names the chain's step"


# --------------------------------------------------------------------------
# which session, which card, and the lock
# --------------------------------------------------------------------------

def _dirs(root: Path, *specs):
    """Plant session directories: (name, ledger file or None)."""
    for name, ledger in specs:
        d = root / name
        d.mkdir(parents=True)
        if ledger:
            (d / ledger).write_text(HEADER)


def _choice(root, dry=0, resume=0, new=0, explicit=""):
    got = lift(f'session_choice {dry} {resume} {new} "{explicit}" {root!s} alpha_g-nvidia_h200-;'
               f" echo rc=$?")
    how_what, rc = got.stdout.rsplit("rc=", 1)
    how, _, what = how_what.strip().partition("\t")
    return how, what, int(rc)


def test_resume_takes_the_newest_session_holding_chain_tsv_never_a_dry_run(tmp_path):
    _dirs(tmp_path, ("alpha_g-nvidia_h200-20260923T080000Z", "CHAIN.tsv"),
          ("alpha_g-nvidia_h200-20260923T093000Z", "CHAIN-dryrun.tsv"),
          ("alpha_g-nocard-20260923T100000Z", "CHAIN.tsv"))
    how, what, rc = _choice(tmp_path, resume=1)
    assert (how, rc) == ("RESUMED", 0)
    assert what == str(tmp_path / "alpha_g-nvidia_h200-20260923T080000Z")


def test_resume_with_no_measuring_session_refuses_and_lists_what_it_found(tmp_path):
    _dirs(tmp_path, ("alpha_g-nvidia_h200-20260923T093000Z", "CHAIN-dryrun.tsv"))
    how, what, rc = _choice(tmp_path, resume=1)
    assert (how, rc) == ("REFUSED", 1)
    assert "found no session holding CHAIN.tsv" in what
    assert "alpha_g-nvidia_h200-20260923T093000Z" in what
    how, what, _ = _choice(tmp_path / "empty", resume=1)
    assert how == "REFUSED" and what.endswith(": none")


def test_a_bare_run_refuses_beside_a_measuring_session_and_new_opens_one(tmp_path):
    _dirs(tmp_path, ("alpha_g-nvidia_h200-20260923T080000Z", "CHAIN.tsv"))
    how, what, rc = _choice(tmp_path)
    assert (how, rc) == ("LATEST_EXISTS", 1)
    assert what == str(tmp_path / "alpha_g-nvidia_h200-20260923T080000Z")
    how, what, rc = _choice(tmp_path, new=1)
    assert (how, rc) == ("NEW", 0) and what.startswith(str(tmp_path / "alpha_g-nvidia_h200-2"))
    # a dry run opens a fresh directory beside it: it skips nothing and spends nothing
    assert _choice(tmp_path, dry=1)[0] == "NEW"
    # and with only a dry run's directory there, a bare run opens a fresh one
    _dirs(tmp_path / "b", ("alpha_g-nvidia_h200-20260923T093000Z", "CHAIN-dryrun.tsv"))
    assert _choice(tmp_path / "b")[0] == "NEW"


@pytest.mark.parametrize("kw,needle", [
    ({"dry": 1, "resume": 1}, "--dry-run plans into a fresh directory of its own"),
    ({"dry": 1, "explicit": "/x/alpha_g-nvidia_h200-1"}, "--dry-run plans into a fresh directory"),
    ({"resume": 1, "new": 1}, "--resume and --new contradict each other"),
    ({"new": 1, "explicit": "/x/s"}, "SESSION=/x/s names the directory outright"),
])
def test_contradictory_session_requests_are_refused(tmp_path, kw, needle):
    _dirs(tmp_path, ("alpha_g-nvidia_h200-20260923T080000Z", "CHAIN.tsv"))
    how, what, rc = _choice(tmp_path, **kw)
    assert (how, rc) == ("REFUSED", 1)
    assert needle in what


def test_a_named_session_is_taken_as_named(tmp_path):
    assert _choice(tmp_path, explicit="/x/s")[:2] == ("NAMED", "/x/s")


def test_the_device_file_is_written_once_and_a_resume_on_another_card_is_refused(tmp_path):
    s = tmp_path / "s"
    s.mkdir()
    (s / "CHAIN.tsv").write_text(HEADER)
    got = lift(f"device_check {s!s} 6b4b5fe6-e1ec; echo rc=$?")
    assert got.stdout.strip() == "rc=0"
    assert (s / "DEVICE").read_text() == "6b4b5fe6-e1ec\n"
    assert lift(f"device_check {s!s} 6b4b5fe6-e1ec; echo rc=$?").stdout.strip() == "rc=0"
    got = lift(f"device_check {s!s} 99999999-aaaa; echo rc=$?")
    assert got.stdout.strip().endswith("rc=2")
    assert "was measured on the card 6b4b5fe6-e1ec, and this card is 99999999-aaaa" in got.stdout
    assert "bash scripts/alpha_g_chain.sh --new" in got.stdout
    assert "R3's and R1's own UUID guards" in got.stdout
    assert (s / "DEVICE").read_text() == "6b4b5fe6-e1ec\n", "a refusal rewrites nothing"


def test_rows_with_no_device_file_and_an_unreadable_uuid_are_refused(tmp_path):
    s = tmp_path / "s"
    _ledger(s / "CHAIN.tsv", _row("r1-g1", "DONE"))
    got = lift(f"device_check {s!s} 6b4b5fe6; echo rc=$?")
    assert got.stdout.strip().endswith("rc=2") and "no DEVICE file" in got.stdout
    assert not (s / "DEVICE").exists()
    got = lift(f'device_check {s!s} ""; echo rc=$?')
    assert got.stdout.strip().endswith("rc=2") and "UUID could not be read" in got.stdout


def _hostname() -> str:
    return subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()


def _dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def test_the_mkdir_lock_refuses_a_live_holder_and_a_resume_takes_a_stale_one_over(tmp_path):
    s = tmp_path / "s"
    s.mkdir()
    got = lift(f'chain_lock {s!s} 0; echo "rc=$? held=$LOCK_DIR"', LOCK_TOOL="mkdir")
    assert f"rc=0 held={s / 'chain.lock.d'}" in got.stdout, got.stdout + got.stderr
    pid, host = (s / "chain.lock.d" / "owner").read_text().split()
    assert host == _hostname()
    # a live holder on this host: refused, on a resume too
    (s / "chain.lock.d" / "owner").write_text(f"{os.getpid()} {_hostname()}\n")
    for resuming in (0, 1):
        got = lift(f"chain_lock {s!s} {resuming}; echo rc=$?", LOCK_TOOL="mkdir")
        assert got.stdout.strip().endswith("rc=2"), got.stdout
        assert "another chain holds" in got.stdout and str(os.getpid()) in got.stdout
    # a dead pid here, or any pid on another host: a resume takes it over, a fresh pass does not
    for owner in (f"{_dead_pid()} {_hostname()}", f"{os.getpid()} some-other-pod"):
        (s / "chain.lock.d" / "owner").write_text(owner + "\n")
        fresh = lift(f"chain_lock {s!s} 0; echo rc=$?", LOCK_TOOL="mkdir")
        assert fresh.stdout.strip().endswith("rc=2"), "only a resume takes a lock over"
        got = lift(f"chain_lock {s!s} 1; echo rc=$?", LOCK_TOOL="mkdir")
        assert got.stdout.strip().endswith("rc=0"), got.stdout
        assert f"took over a stale lock (held by {owner})" in got.stdout
        assert (s / "chain.lock.d" / "owner").read_text().split()[1] == _hostname()


def test_the_lock_is_released_only_while_it_names_this_shell(tmp_path):
    s = tmp_path / "s"
    s.mkdir()
    got = lift(f"chain_lock {s!s} 0 && release_lock; echo rc=$?", LOCK_TOOL="mkdir")
    assert got.stdout.strip() == "rc=0" and not (s / "chain.lock.d").exists()
    # taken over by another holder since: this shell's exit leaves it standing
    got = lift(f'chain_lock {s!s} 0 && echo "999999 elsewhere" > "$LOCK_DIR/owner";'
               " release_lock; echo rc=$?", LOCK_TOOL="mkdir")
    assert got.stdout.strip() == "rc=0"
    assert (s / "chain.lock.d" / "owner").read_text() == "999999 elsewhere\n"


FLOCK_SHIM = """#!@PYTHON@
# util-linux's `flock [-n] FD`, for a box that has none: flock(2) on the
# descriptor this process inherited, which is the shell's open file
# description, so the lock outlives this process exactly as the real one's does
import fcntl
import sys

args = sys.argv[1:]
fd = int([a for a in args if a != "-n"][0])
try:
    fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if "-n" in args else 0))
except BlockingIOError:
    sys.exit(1)
"""


def _flock_path(tmp_path: Path) -> str:
    """A PATH with a flock on it: the box's own (the pod's), else the shim."""
    if shutil.which("flock"):
        return os.environ["PATH"]
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "flock").write_text(FLOCK_SHIM.replace("@PYTHON@", sys.executable))
    (shim / "flock").chmod(0o755)
    return f"{shim}{os.pathsep}{os.environ['PATH']}"


@pytest.mark.parametrize("recorded", ["a live chain", "a dead chain", "another host"])
def test_a_held_flock_is_refused_and_never_taken_over(tmp_path, recorded):
    """The kernel's answer is authoritative. When the chain the lock file
    records is dead, the process still holding the lock is the arm that chain
    left running (it inherited fd 9); until 2026-09-22 a --resume took the lock
    over and ran the same step, same run id, on the card beside it."""
    s = tmp_path / "s"
    s.mkdir()
    path = _flock_path(tmp_path)
    owner = {"a live chain": f"{os.getpid()} {_hostname()}",
             "a dead chain": f"{_dead_pid()} {_hostname()}",
             "another host": f"{os.getpid()} some-other-pod"}[recorded]
    # the holder EXECs its sleep, so killing it releases the lock: a forked
    # sleep would keep fd 9 and the lock with it
    holder = subprocess.Popen(["bash", "-c", f'exec 9>>"{s}/chain.lock"; flock 9 &&'
                               f' echo "{owner}" > "{s}/chain.lock" && exec sleep 30'],
                              env={**os.environ, "PATH": path})
    try:
        for _ in range(100):
            if (s / "chain.lock").exists() and (s / "chain.lock").read_text().strip():
                break
            subprocess.run(["sleep", "0.1"])
        fresh, resume = (lift(f"chain_lock {s!s} {resuming}; echo rc=$?", LOCK_TOOL="flock",
                              PATH=path) for resuming in (0, 1))
        for got in (resume, fresh):
            assert "took over" not in got.stdout, got.stdout
            assert got.stdout.strip().endswith("rc=2"), got.stdout + got.stderr
            assert f"another chain holds {s / 'chain.lock'} (it recorded {owner})" in got.stdout
            assert "never taken over" in got.stdout and f"fuser -v {s / 'chain.lock'}" in got.stdout
            assert "ps -eo pid,etime,args | grep -E" in got.stdout, "a fallback the image has"
            assert "Do not open a --new session meanwhile" in got.stdout
        assert (s / "chain.lock").read_text() == owner + "\n", "a refusal rewrote the holder"
    finally:
        holder.kill()
        holder.wait()
    got = lift(f"chain_lock {s!s} 1; echo rc=$?", LOCK_TOOL="flock", PATH=path)
    assert got.stdout.strip().endswith("rc=0"), got.stdout + got.stderr
    assert "took over" not in got.stdout, "a released lock is taken, not taken over"
    assert (s / "chain.lock").read_text().split()[1] == _hostname()


@pytest.mark.parametrize("code", [64, 65, 71])
def test_a_filesystem_that_cannot_flock_falls_back_to_mkdir_not_to_refusal(tmp_path, code):
    """util-linux flock exits 1 on a conflict and an EX_* code when it could
    not lock at all (ENOLCK, EOPNOTSUPP), which a network volume can answer.
    Until 2026-09-22 any nonzero exit read as 'another chain holds it', so a
    session root on such a volume refused every measuring run on the meter."""
    s = tmp_path / "s"
    s.mkdir()
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "flock").write_text(f"#!/bin/sh\nexit {code}\n")
    (shim / "flock").chmod(0o755)
    path = f"{shim}{os.pathsep}{os.environ['PATH']}"
    got = lift(f'chain_lock {s!s} 0; echo "rc=$? tool=$LOCK_TOOL held=$LOCK_DIR"',
               LOCK_TOOL="flock", PATH=path)
    assert f"rc=0 tool=mkdir held={s / 'chain.lock.d'}" in got.stdout, got.stdout + got.stderr
    assert f"exit {code}: this filesystem does not flock" in got.stdout
    assert "another chain holds" not in got.stdout
    assert (s / "chain.lock.d" / "owner").read_text().split()[0].isdigit()


# --------------------------------------------------------------------------
# the helpers: every read of a report, a log or a ledger
# --------------------------------------------------------------------------

def _report(tmp_path, name, *, G=1, seed=0, ratio=0.9551, lo=0.9544, hi=0.9695,
            synthetic=False, experiment="private_weight_reference", duty=0.25,
            replicates=None, table=None, v7="PASS", v8="PASS", extra=None):
    payload = {**(extra or {}),
               "experiment": experiment, "synthetic": synthetic, "model": "mixtral_8x7b",
               "block_m": 32, "treads": [1, 2, 3, 4, 5, 6], "repeats": 9,
               "pinned": {"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": G}, "seed": seed, "duty": duty,
               "run_id": name, "ratio": ratio,
               "ratio_interval": [lo, hi], "replicates": replicates,
               "treads_table": table or [],
               "gates": [{"tag": "V7", "kind": "VALIDITY", "verdict": v7},
                         {"tag": "V8", "kind": "VALIDITY", "verdict": v8},
                         {"tag": "C1", "kind": "CLAIM", "verdict": "FAIL"}]}
    p = tmp_path / name / "report.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(payload))
    return p


def test_pairs_admits_only_measured_reports_that_formed_a_ratio(tmp_path):
    a = _report(tmp_path, "run-a")
    b = _report(tmp_path, "run-b", seed=1, ratio=None, lo=None, hi=None)
    c = _report(tmp_path, "run-c", seed=2, synthetic=True)
    d = _report(tmp_path, "run-d", seed=2, experiment="clock_elasticity")
    assert H.pairs([str(a), str(b), str(c), str(d), str(tmp_path / "nope.json")]) == [str(a)]


def test_reading_carries_the_joint_reading_each_arms_clock_and_the_low_cells(tmp_path):
    import private_weight_reference as PWR

    from moe.bench.timing import LEVEL_LOW
    def cell(arm, n, mhz, sides):
        return {"arm": arm, "tiles": n, "sm_clock_load_mhz": mhz, "level_sides": sides}
    table = [cell(PWR.NATIVE, 1, 1965.0, []), cell(PWR.NATIVE, 2, 1980.0, []),
             cell(PWR.SHARED, 1, 1950.0, [LEVEL_LOW]), cell(PWR.PRIVATE, 1, None, [LEVEL_LOW])]
    rep = {"n": 3, "points": [0.95, 0.96, 0.97], "spread": 0.02, "sd": 0.01,
           "envelope": [0.94, 0.98], "verdict": "PASS"}
    a = _report(tmp_path, "run-a", G=16, seed=2, replicates=rep, table=table)
    row = H.reading(str(a))
    header = H.reading_header()
    assert len(row) == len(header)
    got = dict(zip(header, row, strict=True))
    assert row[:9] == ["16", "2", "0.9551", "0.9544", "0.9695", "CLAIM_FAIL", "envelope",
                       "0.25", "run-a"]
    assert (got["rep_n"], got["rep_spread"], got["rep_sd"], got["env_lo"], got["env_hi"],
            got["joint"]) == ("3", "0.0200", "0.0100", "0.9400", "0.9800", "PASS")
    assert got[f"clk_{PWR.NATIVE}"] == "1972"            # the median of 1965 and 1980
    assert got[f"clk_{PWR.SHARED}"] == "1950"
    assert got[f"clk_{PWR.PRIVATE}"] == "none", "no clock read is not a clock of zero"
    assert got["low_cells"] == "2"
    # a run scored alone, from a report with no level record
    alone = H.reading(str(_report(tmp_path, "run-b", table=[{"arm": PWR.NATIVE, "tiles": 1,
                                                             "sm_clock_load_mhz": 1965.0}])))
    got = dict(zip(header, alone, strict=True))
    assert [got[k] for k in ("rep_n", "rep_sd", "joint")] == ["none"] * 3
    assert got["exit_scope"] == "alone"
    assert got["low_cells"] == "none"
    assert H.reading(str(tmp_path / "missing.json")) == ["unreadable"]


def _r1(tmp_path, name, lo, hi, *verdicts):
    p = tmp_path / name / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    gates = [{"kind": k, "number": n, "verdict": v} for k, n, v in verdicts]
    p.write_text(json.dumps({"elasticity": {"value": (lo + hi) / 2 if lo is not None else None,
                                            "lo": lo, "hi": hi},
                             "duty": [1.0, 0.5, 0.25], "gates": gates}))
    return p


def test_eta_reads_the_band_through_the_arms_own_edges(tmp_path):
    import clock_elasticity as CE
    ok = ("CLAIM", "C1", "PASS")
    p = _r1(tmp_path, "a", 0.15, 0.24, ok)
    got = H.eta(str(p))
    assert got[:3] == ["0.1950", "0.1500", "0.2400"]
    assert got[3] == CE.band_of(0.15, 0.24)[0] and got[4] == "DONE"
    assert H.eta(str(_r1(tmp_path, "b", 0.20, 0.45, ok)))[3] == "STRADDLES"
    assert H.eta(str(_r1(tmp_path, "c", 0.30, 0.35, ("CLAIM", "C1", "FAIL"))))[3:] == [
        CE.band_of(0.30, 0.35)[0], "CLAIM_FAIL"]
    assert H.eta(str(_r1(tmp_path, "d", None, None, ok)))[3] == "unresolved"


@pytest.mark.parametrize("gates,word", [
    # an interval above V7's admissible edge (planted at the all-tread reading of
    # session 4's G=16 cells, which is printed beside the claim and never gated),
    # V7 FAIL, the page INVALID
    ((("VALIDITY", "V7", "FAIL"), ("CLAIM", "C1", "FAIL")), "INVALID"),
    ((("VALIDITY", "V1", "UNKNOWN"),), "INVALID"),
    ((), "unscored"),
])
def test_eta_withholds_the_band_from_a_page_its_gates_refused(tmp_path, gates, word):
    import clock_elasticity as CE
    p = _r1(tmp_path, "g16", 1.1335, 1.2861, *gates)
    assert CE.band_of(1.1335, 1.2861) is not None, "the arm alone would print a word here"
    got = H.eta(str(p))
    assert got[3] == f"withheld:{word}" and got[4] == word, got


def test_gate_reads_one_verdict_off_a_report(tmp_path):
    a = _report(tmp_path, "run-a", v8="FAIL")
    assert H.gate(str(a), "V8") == "FAIL"
    assert H.gate(str(a), "V7") == "PASS"
    assert H.gate(str(a), "V9") == "absent"
    assert H.gate(str(tmp_path / "none.json"), "V8") == "unreadable"


def test_run_id_and_estimate_come_off_the_plan_page(tmp_path):
    log = tmp_path / "r3.log"
    log.write_text("experiment  private_weight_reference / nvidia_h200-bm32-g4-abc123\n"
                   "estimated GPU time 154 s at the model's own timings, excluding compiles and"
                   " allocation; that includes the alignment probe's 9 s\n"
                   "WALL CLOCK at duty 0.25: the ladder's 145 s of kernel time takes about 581 s,"
                   " the idle gaps\n")
    assert H.run_id(str(log), "private_weight_reference") == "nvidia_h200-bm32-g4-abc123"
    assert H.run_id(str(log), "clock_elasticity") == ""
    # the wall figure, not the kernel one, PLUS the probe that wall line leaves out
    # (timed at full duty): the driver books the same arm at 581 + 9
    assert H.estimate(str(log)) == str(581 + 9)
    assert H.estimate_basis(str(log)) == ("581 s the ladder's wall at the plan's duty + 9 s the"
                                          " alignment probe, timed at full duty on top of it")
    r1 = tmp_path / "r1.log"
    r1.write_text("experiment  clock_elasticity / x-1cb0bac3\n"
                  "estimated wall time 876 s (14.6 min), itemised:\n")
    assert H.estimate(str(r1)) == "876"
    assert H.verdict(str(log)) == "NONE"
    log.write_text("RESULT: VALIDITY V7 PASS [VALIDITY] x | measured y | gate z\n"
                   "RESULT: CLAIM C1 FAIL [CLAIM] x | measured y | gate z\n")
    assert H.verdict(str(log)) == str(exit_codes.CLAIM_FAIL)


def test_the_calibration_dir_comes_off_calibrates_own_wrote_line(tmp_path):
    log = tmp_path / "calibrate.log"
    log.write_text("[calibrate] wrote /r/results/calibration/old-1/measured_nvidia_h200.yaml\n"
                   "[calibrate] wrote /r/results/calibration/run-2/measured_nvidia_h200.yaml\n"
                   "[calibrate] wrote /r/results/calibration/run-2/cells.csv\n"
                   "[calibrate] PUBLISHED to /r/moe/bench/hardware/measured_nvidia_h200.yaml\n")
    assert H.calibration_dir(str(log)) == "/r/results/calibration/run-2"
    assert H.calibration_dir(str(tmp_path / "absent.log")) == ""


def test_the_card_probe_prints_its_reason_off_a_card():
    got = subprocess.run([sys.executable, str(HELPERS), "card"], capture_output=True, text=True,
                         timeout=120, env=laptop_env())
    slug, reason = got.stdout.rstrip("\n").split("\t")
    assert slug == "nocard" and reason, got.stdout


def test_the_device_is_one_spelling_whichever_source_read_it(tmp_path, monkeypatch):
    import private_weight_reference as PWR
    monkeypatch.setattr(PWR, "device_identity", lambda: "6B4B5FE6-E1EC-B594")
    assert H.device() == "6b4b5fe6-e1ec-b594"
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho GPU-6b4b5fe6-e1ec-b594\n")
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(PWR, "device_identity", lambda: PWR.NO_UUID_PREFIX + "nvidia_h200")
    assert H.device() == "6b4b5fe6-e1ec-b594", "nvidia-smi's GPU- prefix is dropped"
    monkeypatch.setattr(PWR, "device_identity", lambda: "")
    assert H.device() == "6b4b5fe6-e1ec-b594"
    monkeypatch.setenv("PATH", str(tmp_path / "no-such-dir"))   # a pod's own nvidia-smi too
    assert H.device() == ""


def test_the_pairs_table_is_rebuilt_whole_and_joins_a_later_r1(tmp_path):
    session, results = tmp_path / "s", tmp_path / "res"
    logs = session / "chain-logs"
    logs.mkdir(parents=True)
    for g, seed in ((1, 0), (1, 1), (16, 0)):
        rid = f"rid-g{g}-s{seed}"
        _report(results / "private_weight_reference", rid, G=g, seed=seed)
        (logs / f"r3-g{g}-s{seed}.log").write_text(
            f"experiment  private_weight_reference / {rid}\n")
    settings = {"r1_treads": "--treads 8", "r3_duty": "--duty 0.25"}
    assert H.pairs_table(session, results, "1 4 16", "0 1 2", settings) == 3
    first = (session / "PAIRS.tsv").read_text()
    rows = [ln.split("\t") for ln in first.splitlines()]
    assert rows[0] == H.reading_header() + ["eta", "eta_lo", "eta_hi", "band", "eta_exit",
                                            "reads_as"]
    assert [(r[0], r[1]) for r in rows[1:]] == [("1", "0"), ("1", "1"), ("16", "0")]
    assert all(r[-3:] == ["unmeasured", "unscored", H.READS_AS_UNRESOLVED] for r in rows[1:])
    # the same disk, rebuilt again: byte-identical, no second row per run
    assert H.pairs_table(session, results, "1 4 16", "0 1 2", settings) == 3
    assert (session / "PAIRS.tsv").read_text() == first
    # R1 for G=1 lands on a later pass: every G=1 row takes it
    _r1(results / "clock_elasticity", "r1rid", 0.15, 0.24, ("CLAIM", "C1", "PASS"))
    (logs / "r1-g1.log").write_text("experiment  clock_elasticity / r1rid\n")
    H.pairs_table(session, results, "1 4 16", "0 1 2", settings)
    rows = [ln.split("\t") for ln in (session / "PAIRS.tsv").read_text().splitlines()]
    assert [r[-3:] for r in rows[1:]] == [["RAW-STANDS", "DONE", H.READS_AS["RAW-STANDS"]]] * 2 + [
        ["unmeasured", "unscored", H.READS_AS_UNRESOLVED]]
    fixed = {ln.split("\t")[0]: ln.split("\t")[1:]
             for ln in (session / "PAIRS-fixed.tsv").read_text().splitlines()[1:]}
    assert fixed["r3_duty"] == ["0.25", "report.json of 3 run(s)"]
    assert fixed["r3_pinned"][0] == "BLOCK_SIZE_N=64", "the swizzle is the G column, not fixed"
    assert fixed["r1_duty"] == ["1.0 0.5 0.25", "report.json of 1 run(s)"]
    assert fixed["r1_treads"] == ["--treads 8", "the chain's command line"]
    assert fixed["group_m"][0] == "varies: 1 4 16"
    assert not any(ln.startswith("#") for ln in (session / "PAIRS.tsv").read_text().splitlines())


def test_a_run_that_formed_no_ratio_quotes_no_joint_reading(tmp_path):
    """E2E-3. When a later seed forms no ratio of its own, R3 builds its
    replicates block from the OTHER runs alone, and PAIRS.tsv printed that
    block on the seed's row as if the seed were in it (n=2 PASS beside a
    ratio of none). The joint columns are filled only when the block is a
    reading this run is in."""
    others = {"n": 2, "spread": 0.01, "sd": None, "envelope": [0.94, 0.97], "verdict": "PASS",
              "runs": [{"run_id": "run-s0"}, {"run_id": "run-s1"}]}
    rep = _report(tmp_path, "run-s2", seed=2, ratio=None, lo=None, hi=None, replicates=others)
    got = dict(zip(H.reading_header(), H.reading(str(rep)), strict=True))
    assert [got[k] for k in ("rep_n", "env_lo", "env_hi", "joint")] == ["none"] * 4
    assert got["exit_scope"] == "alone", "C1 with no ratio of its own was not scored on an envelope"
    # a block that names other runs only is not this run's either
    rep = _report(tmp_path, "run-s2c", seed=2, replicates=others)
    assert dict(zip(H.reading_header(), H.reading(str(rep)), strict=True))["joint"] == "none"
    # the same shape on a run that formed its ratio and is in it: quoted, on the envelope
    mine = dict(others, n=3, runs=[{"run_id": "run-s2b"}, *others["runs"]])
    rep = _report(tmp_path, "run-s2b", seed=2, replicates=mine)
    got = dict(zip(H.reading_header(), H.reading(str(rep)), strict=True))
    assert (got["rep_n"], got["joint"], got["exit_scope"]) == ("3", "PASS", "envelope")


def _seeded(session, results, g, runs, duty=0.25):
    """Plant a G's ratio reports and the chain logs that name them."""
    logs = session / "chain-logs"
    logs.mkdir(parents=True, exist_ok=True)
    for seed, (ratio, lo, hi, v7) in runs.items():
        rid = f"rid-g{g}-s{seed}"
        _report(results / "private_weight_reference", rid, G=g, seed=seed, ratio=ratio, lo=lo,
                hi=hi, v7=v7, duty=duty if not isinstance(duty, dict) else duty[seed])
        (logs / f"r3-g{g}-s{seed}.log").write_text(
            f"experiment  private_weight_reference / {rid}\n")


def test_the_per_g_table_reads_every_seed_together_whatever_order_they_ran(tmp_path):
    """E2E-4 and T5. The chain pairs a seed only with the seeds BEFORE it, so a
    seed 0 re-run on --resume after seeds 1 and 2 latched was scored alone
    and no PAIRS.tsv row ever held all three; and an INVALID run inside a
    joint envelope was flagged nowhere. PAIRS-by-G.tsv reads every seed of a
    G that formed a ratio together, through R3's own cross-run machinery,
    whatever PAIRS.tsv's rows say, and names any INVALID run inside."""
    import private_weight_reference as PWR
    session, results = tmp_path / "s", tmp_path / "res"
    runs = {0: (0.560, 0.550, 0.570, "PASS"), 1: (0.600, 0.580, 0.610, "FAIL"),
            2: (0.555, 0.540, 0.565, "PASS")}
    _seeded(session, results, 4, runs)
    _seeded(session, results, 16, {0: (0.7, 0.69, 0.71, "PASS"), 1: (0.72, 0.71, 0.73, "PASS")},
            duty={0: 0.25, 1: 0.1})
    H.pairs_table(session, results, "1 4 16", "0 1 2", {})
    rows = [dict(zip(H.BY_G_HEADER, r, strict=True)) for r in _rows(session / "PAIRS-by-G.tsv")]
    assert [r["G"] for r in rows] == ["1", "4", "16"], "one row per G of the ladder"
    g1, g4, g16 = rows
    assert (g1["n"], g1["joint"]) == ("0", "none") and "formed a ratio" in g1["note"]
    points = [r[0] for r in runs.values()]
    assert (g4["n"], g4["seeds"]) == ("3", "0 1 2")
    assert (float(g4["env_lo"]), float(g4["env_hi"])) == (min(r[1] for r in runs.values()),
                                                          max(r[2] for r in runs.values()))
    assert float(g4["mean"]) == pytest.approx(statistics.mean(points), abs=5e-5)
    assert float(g4["sd"]) == pytest.approx(statistics.stdev(points), abs=5e-5)
    # the envelope reaches past ALPHA_BAND's upper edge without missing the band
    assert float(g4["env_lo"]) < PWR.ALPHA_BAND[1] <= float(g4["env_hi"])
    assert g4["joint"] == "UNKNOWN"
    assert g4["invalid_in_envelope"] == "seed 1", "seed 1's own page exited INVALID on V7"
    assert g4["eta"] == "none" and g4["band"] == "unmeasured", "R1's columns ride beside"
    # two duties are two designs: R3's load_replicates refuses them, and so does the table
    assert g16["joint"] == "REFUSED" and "duty" in g16["note"], g16


def test_the_tables_carry_a_legend_off_the_arms_own_constants(tmp_path):
    """T6. `exit` changed scope between seed 0 and seed 1 of a G, `joint` is a
    verdict against the refit band and not a quotability flag, and the two
    intervals are 90% and 95%: none of it was written down beside the table."""
    import clock_elasticity as CE
    import private_weight_reference as PWR
    session, results = tmp_path / "s", tmp_path / "res"
    _seeded(session, results, 1, {0: (0.95, 0.94, 0.96, "PASS")})
    H.pairs_table(session, results, "1", "0", {})
    legend = (session / "PAIRS-README.txt").read_text()
    for column in ("exit_scope", "rep_n", "joint", "eta_exit", "invalid_in_envelope",
                   "PAIRS-by-G.tsv", "PAIRS-fixed.tsv"):
        assert column in legend, column
    assert f"{PWR.INTERVAL_PCT:.0f}% percentile bootstrap over repeats" in legend
    assert f"ALPHA_BAND [{PWR.ALPHA_BAND[0]}, {PWR.ALPHA_BAND[1]})" in legend
    assert "NOT a quotability flag" in legend
    # R1's interval is the 2.5th to 97.5th percentile of its own bootstrap
    source = Path(CE.__file__).read_text()
    assert "_percentile(values, 0.025)" in source and "_percentile(values, 0.975)" in source
    assert "95% percentile bootstrap" in legend


def _default(knob: str) -> str:
    """A knob's default, off its own `KNOB="${KNOB:-default}"` line."""
    m = re.search(rf'^{knob}="\$\{{{knob}:-([^}}]*)\}}"$', CODE, re.M)
    assert m, knob
    return m.group(1)


def test_the_legend_names_the_states_r1s_secant_spans(tmp_path):
    """The legend called R1's word 'a SECANT between R1's capped duty
    states', written when the states were 1.0 0.7 0.5. At 1.0 0.5 0.25 only
    1.0 held session 5's H200 on its power cap, and R3's duty 0.25 is one of
    R1's own states. The legend names the states this session's R1 ran and
    R3's duty, as PAIRS-fixed.tsv reads them, and what the secant spans at the
    chain's own defaults, R1_DUTY's and R3_DUTY's."""
    session, results = tmp_path / "s", tmp_path / "res"
    _seeded(session, results, 1, {0: (0.95, 0.94, 0.96, "PASS")})
    # an R1 run at states other than the defaults: the legend names what ran
    (results / "clock_elasticity" / "rid-r1-g1").mkdir(parents=True)
    (results / "clock_elasticity" / "rid-r1-g1" / "report.json").write_text(json.dumps(
        {"elasticity": {"value": 0.5, "lo": 0.45, "hi": 0.55}, "duty": [1.0, 0.7, 0.5],
         "gates": []}))
    (session / "chain-logs" / "r1-g1.log").write_text(
        "experiment  clock_elasticity / rid-r1-g1\n")
    H.pairs_table(session, results, "1", "0", {"r1_duty": "--duty 1.0 0.6 0.3"})
    fixed = {ln.split("\t")[0]: ln.split("\t")[1]
             for ln in (session / "PAIRS-fixed.tsv").read_text().splitlines()[1:]}
    assert (fixed["r1_duty"], fixed["r3_duty"]) == ("1.0 0.7 0.5", "0.25"), fixed
    legend = " ".join((session / "PAIRS-README.txt").read_text().split())
    assert "capped duty states" not in legend
    assert "It is a SECANT across R1's duty states, not a local reading at R3's duty" in legend
    assert (f"This session's R1 states: {fixed['r1_duty']}; R3's duty: {fixed['r3_duty']}"
            " (PAIRS-fixed.tsv's r1_duty and r3_duty)") in legend
    r1, r3 = _default("R1_DUTY").split(), _default("R3_DUTY")
    assert r1[0] == "1.0" and r3 == r1[-1], "the sentence below is about these defaults"
    assert (f"At the chain's defaults, R1_DUTY {' '.join(r1)} and R3_DUTY {r3}, the secant"
            f" runs from the capped clock at {r1[0]} to the clocks at {' and '.join(r1[1:])}"
            ) in legend
    assert f"R3's {r3} is the top of that range" in legend
    # a session with no R1 report yet names the chain's command line, not the flag
    fresh = tmp_path / "fresh"
    _seeded(fresh, results, 4, {0: (0.7, 0.69, 0.71, "PASS")})
    H.pairs_table(fresh, results, "4", "0", {"r1_duty": f"--duty {' '.join(r1)}"})
    legend = " ".join((fresh / "PAIRS-README.txt").read_text().split())
    assert f"This session's R1 states: {' '.join(r1)}; R3's duty: {r3}" in legend


def test_the_fixed_table_says_mixed_when_the_runs_disagree(tmp_path):
    session, results = tmp_path / "s", tmp_path / "res"
    (session / "chain-logs").mkdir(parents=True)
    for seed, duty in ((0, 0.25), (1, 0.5)):
        _report(results / "private_weight_reference", f"rid{seed}", seed=seed, duty=duty)
        (session / "chain-logs" / f"r3-g1-s{seed}.log").write_text(
            f"experiment  private_weight_reference / rid{seed}\n")
    H.pairs_table(session, results, "1", "0 1", {})
    fixed = {ln.split("\t")[0]: ln.split("\t")[1]
             for ln in (session / "PAIRS-fixed.tsv").read_text().splitlines()[1:]}
    assert fixed["r3_duty"] == "MIXED: 0.25 | 0.5"


# --------------------------------------------------------------------------
# what a ratio reads as, and the bytes-rate bound (H200 session 5's findings)
# --------------------------------------------------------------------------

def test_reads_as_is_the_findings_rule_over_r1s_own_band_names():
    """A ratio is a re-read fraction ONLY beside RAW-STANDS on a page whose
    gates stood behind it; beside CLOCK-CARRIES it is a blend and not alpha
    (session 5 at G >= 4: the shared arm on a clock-scaled floor, the
    bytes-rate bound still proving reuse); beside every other word it is
    unresolved. The two keys are R1's own band names."""
    import clock_elasticity as CE
    names = [b[0] for b in CE.BANDS]
    assert set(H.READS_AS) <= set(names), (H.READS_AS, names)
    for word in ("DONE", "CLAIM_FAIL"):
        assert H.reads_as("RAW-STANDS", word) == "re-read fraction"
        assert H.reads_as("CLOCK-CARRIES", word) == ("blend (traffic and a clock-scaled on-chip"
                                                     " floor): not alpha")
    unlicensed = [n for n in names if n not in H.READS_AS] + [
        "STRADDLES", "withheld:INVALID", "unmeasured", "unresolved"]
    for band in unlicensed:
        assert H.reads_as(band, "CLAIM_FAIL") == H.READS_AS_UNRESOLVED, band
    for word in ("INVALID", "REFUSED", "ERROR", "unscored"):
        assert H.reads_as("RAW-STANDS", word) == H.READS_AS_UNRESOLVED, word
    got = subprocess.run([sys.executable, str(HELPERS), "reads-as", "CLOCK-CARRIES", "DONE"],
                         capture_output=True, text=True, timeout=120, env=laptop_env())
    assert got.stdout.strip() == H.READS_AS["CLOCK-CARRIES"]


def _r1_page(results: Path, logs: Path, g: int, lo, hi, *verdicts):
    """An R1 report for G and the chain log that names it."""
    rid = f"r1rid-g{g}"
    _r1(results / "clock_elasticity", rid, lo, hi, *verdicts)
    (logs / f"r1-g{g}.log").write_text(f"experiment  clock_elasticity / {rid}\n")


def test_both_tables_say_what_the_ratio_reads_as(tmp_path):
    """The relabel: PAIRS.tsv and PAIRS-by-G.tsv each gain `reads_as`, off the
    G's R1 page. Until now CLOCK-CARRIES rows sat in the table as if the ratio
    were the re-read fraction the header promised for RAW-STANDS."""
    session, results = tmp_path / "s", tmp_path / "res"
    ok, fail = ("CLAIM", "C1", "PASS"), ("CLAIM", "C1", "FAIL")
    for g in (1, 4, 16, 64, 128):
        _seeded(session, results, g, {0: (0.9, 0.89, 0.91, "PASS")})
    logs = session / "chain-logs"
    _r1_page(results, logs, 1, 0.10, 0.20, ok)                                  # RAW-STANDS
    _r1_page(results, logs, 4, 1.07, 1.20, fail)                                # CLOCK-CARRIES
    _r1_page(results, logs, 16, 1.09, 1.27, ("VALIDITY", "V7", "FAIL"), fail)   # withheld
    _r1_page(results, logs, 64, 0.387, 0.407, fail)                             # STRADDLES
    H.pairs_table(session, results, "1 4 16 64 128", "0", {})
    want = [H.READS_AS["RAW-STANDS"], H.READS_AS["CLOCK-CARRIES"]] + [H.READS_AS_UNRESOLVED] * 3
    header = H.reading_header() + ["eta", "eta_lo", "eta_hi", "band", "eta_exit", "reads_as"]
    per_run = [dict(zip(header, r, strict=True)) for r in _rows(session / "PAIRS.tsv")]
    assert [r["reads_as"] for r in per_run] == want
    per_g = [dict(zip(H.BY_G_HEADER, r, strict=True)) for r in _rows(session / "PAIRS-by-G.tsv")]
    assert [r["reads_as"] for r in per_g] == want
    assert [r["band"] for r in per_g] == ["RAW-STANDS", "CLOCK-CARRIES", "withheld:INVALID",
                                          "STRADDLES", "unmeasured"]
    legend = " ".join((session / "PAIRS-README.txt").read_text().split())
    for label in (*H.READS_AS.values(), H.READS_AS_UNRESOLVED):
        assert f"`{label}`" in legend, label
    assert "ONLY when the band is RAW-STANDS on a page whose gates stood behind it" in legend


def _ruler(path: Path, *, named: float, read: float | None, pin: float | None,
           read_note: str = "", reduce_gbps: float | None = None) -> Path:
    """A planted calibrate_hardware.py yaml: the named bandwidth, the read
    patterns, and the pin rate."""
    import yaml
    patterns = []
    if read is not None:
        patterns.append({"pattern": "read_stream", "gbps": read, "note": read_note})
    if reduce_gbps is not None:
        patterns.append({"pattern": "read_reduce", "gbps": reduce_gbps, "note": ""})
    patterns.append({"pattern": "triad", "gbps": named, "note": ""})
    doc = {"name": "planted", "verified": True, "checked_on": "2026-09-23",
           "measured_commit": "0123456789abcdef", "memory": {"bandwidth_tb_s": named / 1000},
           "detail": {"achieved_bandwidth_gbps": named, "ceiling_pattern": "triad",
                      "bandwidth_patterns": patterns},
           "observed": {"pin_rate_gbps": pin} if pin else {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc))
    return path


def _bound_reports(session: Path, results: Path, g: int, tops: dict[int, float],
                   *, named: float, w: int, treads: int = 6):
    """Ratio reports for G whose shared ladder tops out at `tops[seed]` ms."""
    logs = session / "chain-logs"
    logs.mkdir(parents=True, exist_ok=True)
    for seed, top in tops.items():
        rid = f"rid-g{g}-s{seed}"
        points = [[n, top * n / treads] for n in range(1, treads + 1)]
        _report(results / "private_weight_reference", rid, G=g, seed=seed,
                extra={"card": "nvidia_planted", "bandwidth_gbps": named,
                       "memory_plan": {"per_copy_bytes": w},
                       "ladders": {"shared": {"points": points}}})
        (logs / f"r3-g{g}-s{seed}.log").write_text(
            f"experiment  private_weight_reference / {rid}\n")


def test_the_bytes_rate_bound_is_read_off_the_reports_and_the_ruler_the_session_measured(tmp_path):
    """Findings 3.6: alpha <= (t x C / W - 1) / (n - 1), the one bound that
    needs no private arm, per G, at the ruler's read_stream and its pin rate,
    with t the mean of the G's shared top-tread times, W each report's own
    expert set and the ruler the yaml calibrate wrote in this session. At the
    bound the bytes W (1 + alpha (n - 1)) fill exactly t x C. 1 or above
    excludes nothing and the console says so."""
    session, results = tmp_path / "s", tmp_path / "res"
    named, read, pin, w = 4000.0, 4500.0, 4800.0, 2_000_000_000
    ruler = _ruler(tmp_path / "cal" / "run-1" / "measured_nvidia_planted.yaml",
                   named=named, read=read, pin=pin, reduce_gbps=4400.0)
    (session / "logs").mkdir(parents=True)
    (session / "logs" / "calibrate.log").write_text(
        f"[calibrate] wrote {tmp_path}/cal/old/measured_nvidia_planted.yaml\n"
        f"[calibrate] wrote {ruler}\n[calibrate] PUBLISHED to elsewhere.yaml\n")
    tops = {4: {0: 2.30, 1: 2.36}, 1: {0: 3.10, 1: 3.14}}
    for g, t in tops.items():
        _bound_reports(session, results, g, t, named=named, w=w)
    n, lines = H.pairs_table_lines(session, results, "1 4", "0 1", {})
    assert n == 4
    per_g = {r[0]: dict(zip(H.BY_G_HEADER, r, strict=True))
             for r in _rows(session / "PAIRS-by-G.tsv")}
    for g, t in tops.items():
        row, mean = per_g[str(g)], statistics.mean(t.values())
        assert row["top_tread"] == "6" and float(row["shared_top_ms"]) == pytest.approx(mean,
                                                                                        abs=5e-5)
        for col, c in (("bound_read_stream", read), ("bound_pin_rate", pin)):
            b = float(row[col])
            # the bound's meaning, not its formula: at it, the bytes fill the time
            assert w * (1 + b * 5) == pytest.approx(mean * 1e-3 * c * 1e9, rel=1e-4), (g, col)
        assert float(row["bound_pin_rate"]) > float(row["bound_read_stream"])
    fixed = {ln.split("\t")[0]: ln.split("\t")[1:]
             for ln in (session / "PAIRS-fixed.tsv").read_text().splitlines()[1:]}
    assert fixed["ruler"][0] == str(ruler)
    assert "calibrate wrote in this session" in fixed["ruler"][1]
    assert fixed["read_ceiling_gbps"][0] == f"{read:.1f}" and "read_stream" in fixed[
        "read_ceiling_gbps"][1]
    assert fixed["pin_rate_gbps"][0] == f"{pin:.1f}"
    assert fixed["expert_set_bytes"][0] == str(w)
    # the console: the ruler once, then each G with the ceilings it used
    assert lines[0].startswith(f"bytes-rate bound, at the ceilings of {ruler}")
    g4 = next(ln for ln in lines if ln.startswith("G=4 "))
    assert (f"alpha <= {per_g['4']['bound_read_stream']} at read_stream {read:.1f} GB/s, <= "
            f"{per_g['4']['bound_pin_rate']} at the pin rate {pin:.1f} GB/s") in g4
    assert "excludes nothing" not in g4
    g1 = next(ln for ln in lines if ln.startswith("G=1 "))
    assert float(per_g["1"]["bound_read_stream"]) >= 1, "the planted G=1 fits a full re-read"
    assert "1 or above: a full re-read per M-tile fits in that time" in g1
    assert "reads as: unresolved" in g1, "no R1 page yet"
    got = subprocess.run([sys.executable, str(HELPERS), "pairs-table", str(session), str(results),
                          "1 4", "0 1"], capture_output=True, text=True, timeout=120,
                         env=laptop_env())
    assert got.stdout.splitlines() == ["4 row(s)", *lines], got.stdout + got.stderr


def test_a_ruler_the_reports_were_not_scored_against_withholds_the_bound(tmp_path):
    """calibrate --publish overwrites one tracked file per card: on the laptop
    the tracked ruler may be a later calibration than the one the reports
    were scored against. Its named bandwidth must be the reports' own, or no
    G gets a bound; `ruler=` names the right one."""
    session, results = tmp_path / "s", tmp_path / "res"
    _bound_reports(session, results, 4, {0: 2.3}, named=4000.0, w=2_000_000_000)
    other = _ruler(tmp_path / "other.yaml", named=4000.5, read=4500.0, pin=4800.0)
    n, lines = H.pairs_table_lines(session, results, "4", "0", {"ruler": str(other)})
    assert lines[0].startswith("bytes-rate bound: NO RULER, so no G has one")
    assert "is not the ruler these reports were scored against" in lines[0]
    assert lines[1].startswith("G=4   no bytes-rate bound (no ruler, above)")
    row = dict(zip(H.BY_G_HEADER, _rows(session / "PAIRS-by-G.tsv")[0], strict=True))
    assert (row["bound_read_stream"], row["bound_pin_rate"]) == ("none", "none")
    right = _ruler(tmp_path / "right.yaml", named=4000.0, read=4500.0, pin=4800.0)
    H.pairs_table(session, results, "4", "0", {"ruler": str(right)})
    row = dict(zip(H.BY_G_HEADER, _rows(session / "PAIRS-by-G.tsv")[0], strict=True))
    assert row["bound_pin_rate"] != "none"
    # no ruler anywhere: said, not guessed
    bare, bare_results = tmp_path / "bare", tmp_path / "bare-res"
    _bound_reports(bare, bare_results, 4, {0: 2.3}, named=4000.0, w=2_000_000_000)
    _n, lines = H.pairs_table_lines(bare, bare_results, "4", "0", {})
    assert "pass ruler=<yaml> to pairs-table" in lines[0]


def test_the_read_ceiling_is_read_stream_and_never_read_reduce(tmp_path):
    """read_reduce is calibrate's LOWER bound on the read rate: a ceiling set
    too low makes an upper bound on alpha too tight, so a ruler without a
    usable read_stream gives no read-ceiling bound, and the pin-rate bound
    still stands."""
    from moe.bench import calibrate as CAL
    session, results = tmp_path / "s", tmp_path / "res"
    _bound_reports(session, results, 4, {0: 2.3}, named=4000.0, w=2_000_000_000)
    for name, kw in (("none", {"read": None}),
                     ("disowned", {"read": 4600.0,
                                   "read_note": f"came in below triad: {CAL.DISOWNED}"})):
        ruler = _ruler(tmp_path / f"{name}.yaml", named=4000.0, pin=4800.0, reduce_gbps=4400.0,
                       **kw)
        _n, lines = H.pairs_table_lines(session, results, "4", "0", {"ruler": str(ruler)})
        row = dict(zip(H.BY_G_HEADER, _rows(session / "PAIRS-by-G.tsv")[0], strict=True))
        assert row["bound_read_stream"] == "none" and row["bound_pin_rate"] != "none", name
        assert "no bound at read_stream" in lines[1] and "at the pin rate 4800.0 GB/s" in lines[1]


def test_the_ruler_row_names_its_measurement_or_says_it_has_none(tmp_path):
    """PAIRS-fixed.tsv's read-ceiling row cites the ruler's own date and
    commit; a ruler that records no commit says so, rather than a truncated
    "no comm" standing in for a sha."""
    import yaml
    session, results = tmp_path / "s", tmp_path / "res"
    _bound_reports(session, results, 4, {0: 2.3}, named=4000.0, w=2_000_000_000)
    ruler = _ruler(tmp_path / "r.yaml", named=4000.0, read=4500.0, pin=4800.0)
    doc = yaml.safe_load(ruler.read_text())

    def source():
        H.pairs_table(session, results, "4", "0", {"ruler": str(ruler)})
        return {ln.split("\t")[0]: ln.split("\t")[2] for ln in
                (session / "PAIRS-fixed.tsv").read_text().splitlines()[1:]}["read_ceiling_gbps"]
    assert f"measured {doc['checked_on']} at {doc['measured_commit'][:7]}" in source()
    doc["measured_commit"] = ""
    ruler.write_text(yaml.safe_dump(doc))
    assert source().endswith("at no commit recorded"), source()


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_ncu_is_looked_for_on_path_then_under_each_glob_newest_first(tmp_path):
    """Session 4's probe looked on PATH alone and read "no ncu on PATH"; an
    image can carry ncu under the CUDA toolkit or Nsight Compute's own
    directory with neither on PATH. PATH first (what a bare `ncu` runs), then
    each glob newest name first, a path once, executables only."""
    on_path = _exe(tmp_path / "bin" / "ncu")
    cuda = _exe(tmp_path / "cuda" / "bin" / "ncu")
    cuda124 = _exe(tmp_path / "cuda-12.4" / "bin" / "ncu")
    old = _exe(tmp_path / "nsight" / "2024.1" / "ncu")
    new = _exe(tmp_path / "nsight" / "2025.3" / "ncu")
    (tmp_path / "nsight" / "2026.1").mkdir()
    (tmp_path / "nsight" / "2026.1" / "ncu").write_text("not executable\n")
    globs = [f"{tmp_path}/cuda*/bin/ncu", f"{tmp_path}/nsight/*/ncu"]
    got = H.ncu_locate(globs, path_env=str(tmp_path / "bin"))
    assert got == [(str(on_path), "PATH"), (str(cuda), globs[0]), (str(cuda124), globs[0]),
                   (str(new), globs[1]), (str(old), globs[1])]
    # the PATH hit is listed once, as PATH
    got = H.ncu_locate(globs, path_env=str(tmp_path / "cuda" / "bin"))
    assert [p for p, _ in got].count(str(cuda)) == 1 and got[0] == (str(cuda), "PATH")
    assert H.ncu_locate([f"{tmp_path}/nowhere/*/ncu"], path_env=str(tmp_path / "empty")) == []
    env = laptop_env(PATH=str(tmp_path / "empty") + os.pathsep + "/usr/bin:/bin")
    cli = subprocess.run([sys.executable, str(HELPERS), "ncu-locate", " ".join(globs)],
                         capture_output=True, text=True, timeout=120, env=env)
    assert cli.stdout.rstrip("\n").split("\t") == [
        str(cuda), globs[0], " ".join(str(p) for p in (cuda, cuda124, new, old))]
    none = subprocess.run([sys.executable, str(HELPERS), "ncu-locate", f"{tmp_path}/x/*/ncu"],
                          capture_output=True, text=True, timeout=120, env=env)
    assert none.stdout.rstrip("\n").split("\t") == ["none", "none", "none"]


def test_the_counters_note_reads_a_crash_a_timeout_a_defect_and_no_device(tmp_path):
    """Whatever the probe does, the counter probe's note says it in one word
    first, and none of it is a latched word: ERROR for a crash, a timeout or
    an exit code its RESULT lines do not support; UNTESTED when ncu is here
    and no kernel launched."""
    s = tmp_path / "s"
    s.mkdir()
    log = s / "counter-probe.log"

    def note(rc, secs=5, cap=1800, binary="/x/ncu", where="PATH"):
        return H.counters(s, log, rc, cap, secs, binary, where, binary, "/g/*/ncu")
    log.write_text("Traceback (most recent call last): planted\n")
    assert note(4).startswith("ERROR: the probe wrote no COUNTERS.json (exit 4); its log ends:"
                              " Traceback")
    assert note(124, secs=1800).startswith("ERROR: TIMED OUT after 1800 s against a cap of 1800 s")
    assert note(137, secs=1900).startswith("ERROR: TIMED OUT after 1900 s")
    opened = {"verdict": "OPEN", "ncu": {"present": True, "counters_read": True,
                                         "cause": "counters readable: planted"}}
    (s / "COUNTERS.json").write_text(json.dumps(opened))
    log.write_text(exit_codes.result_line("CLAIM", "P1", "PASS", "planted") + "\n")
    assert note(0).startswith("OPEN: counters readable: planted; ncu /x/ncu (on PATH)")
    assert note(1).startswith("ERROR: DEFECT: the probe exited 1 and its RESULT lines imply 0")
    (s / "COUNTERS.json").write_text(json.dumps(
        {"verdict": "REFUSE", "ncu": {"present": True, "cause": "no CUDA device at all: planted"}}))
    log.write_text("ROUTE PROBE\n  VERDICT   REFUSE\n")
    assert note(2).startswith("UNTESTED: no CUDA device at all: planted")
    (s / "COUNTERS.json").write_text(json.dumps(
        {"verdict": "REFUSE", "ncu": {"present": False, "why": "no ncu on PATH"}}))
    assert note(2).startswith("ERROR: DEFECT: /x/ncu was found"), "found, yet the probe saw none"
    assert note(2, binary="none", where="none").startswith(
        "ABSENT: no ncu where the chain looks; ncu not on PATH and none at /g/*/ncu")
    text = (s / "COUNTERS").read_text()
    assert "verdict     ABSENT" in text and "informational" in text
    for word in ("DONE", "CLAIM_FAIL", "INVALID"):
        assert f"verdict     {word}" not in text


# --------------------------------------------------------------------------
# the measuring path, end to end, through planted interpreters
# --------------------------------------------------------------------------

#: The planted plan pages' prices: R3's ladder kernel seconds and its probe's,
#: and R1's wall. The chain prices every cap, the V8 STOP and a follow-up off
#: these; the tests recompute each figure from them.
STUB_LADDER_S, STUB_PROBE_S, STUB_R1_S = 150, 10, 900
#: The named bandwidth every planted R3 report says it was scored against, and
#: its expert set: planted inputs of the bytes-rate bound, which the tests
#: recompute from the reports rather than restate.
STUB_BANDWIDTH_GBPS, STUB_EXPERT_SET_BYTES = 4000.0, 2_000_000_000

STUB_BASE = r"""#!/bin/bash
# a planted base interpreter: the card and its UUID are the test's, an arm's
# --dry-run is a planted plan page carrying the price lines the chain reads,
# and everything else runs on the real one
if [[ "${1:-}" == */alpha_g_chain_helpers.py ]]; then
  case "${2:-}" in
    card)   printf '%s\t\n' "${STUB_CARD:-nvidia_testcard}"; exit 0 ;;
    device) printf '%s\n' "${STUB_DEVICE:-0a0a0a0a-1111-2222-3333-444444444444}"; exit 0 ;;
  esac
fi
if [[ " $* " == *" --dry-run "* ]]; then
  case "${1:-}" in
    */clock_elasticity.py)
      echo "experiment  clock_elasticity / planted-dry"
      echo "estimated wall time @R1@ s, itemised:"
      exit 2 ;;
    */private_weight_reference.py)
      duty=1.0; prev=""
      for a in "$@"; do [[ "$prev" == --duty ]] && duty="$a"; prev="$a"; done
      echo "experiment  private_weight_reference / planted-dry"
      echo "estimated GPU time $(( @LADDER@ + @PROBE@ )) s; that includes the alignment" \
           "probe's @PROBE@ s"
      w="$(awk -v d="$duty" 'BEGIN {printf "%d", @LADDER@ / d + 0.5}')"
      echo "WALL CLOCK at duty $duty: the ladder's @LADDER@ s of kernel time takes about $w s"
      exit 2 ;;
  esac
fi
# the counter probe: the REAL do_probe over a planted box (STUB_PROBE below)
if [[ "${1:-}" == */dram_counter_route.py && " $* " == *" --probe "* ]]; then
  shift
  exec @PYTHON@ @COUNTER_PROBE@ "$@"
fi
# a planted pytest, when the test asks for one: the names of the environment
# it was run with, and a tally
if [[ "${1:-}" == -m && "${2:-}" == pytest && -n "${STUB_PYTEST:-}" ]]; then
  printf 'pytest\t%s\n' "$(env | sed 's/=.*//' | sort | tr '\n' ' ')" >> "$STUB_PYTEST"
  echo "${STUB_PYTEST_TALLY:-5 passed in 1.00s}"
  exit "${STUB_PYTEST_RC:-0}"
fi
exec @PYTHON@ "$@"
"""

STUB_ARM = r'''#!@PYTHON@
"""A planted arm: the plan line, the RESULT lines, report.json and an exit code
that agrees with them, which is everything the chain reads, and a trace line."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "@ROOT@")
from moe.bench import exit_codes

script, args = Path(sys.argv[1]).name, sys.argv[2:]


def opt(name):
    return args[args.index(name) + 1] if name in args else None


def many(name):
    if name not in args:
        return []
    out = []
    for a in args[args.index(name) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


plan = json.loads(Path(os.environ["STUB_PLAN"]).read_text()) if os.environ.get("STUB_PLAN") else {}
STUB_BANDWIDTH_GBPS, STUB_EXPERT_SET_BYTES = @BANDWIDTH@, @EXPERT_SET@
if "--probe-check" in args:
    # the NEW INTERFACE of private_weight_reference.py --probe-check: one RESULT
    # line, exit 0 on PASS and 3 otherwise, REFUSED 2 with no card or no vLLM
    with open(os.environ["STUB_TRACE"], "a") as f:
        f.write("probe-check\t" + " ".join(args) + "\n")
    cfg = plan.get("probe-check", {})
    if cfg.get("refuse"):
        print("REFUSED: no CUDA device (a planted refusal)")
        sys.exit(2)
    v = cfg.get("P1", "PASS")
    print(exit_codes.result_line("VALIDITY", "P1", v, "[VALIDITY] planted probe | measured "
                                 "graph_calls 16 | gate graph_calls == 16"))
    sys.exit(0 if v == "PASS" else 3)
g, tag = int(opt("--group-m")), opt("--session-tag")
if script == "clock_elasticity.py":
    exp, step = "clock_elasticity", f"r1-g{g}"
else:
    exp, step = "private_weight_reference", f"r3-g{g}-s{opt('--seed')}"
rid = f"stub-{step}-{tag}"
with open(os.environ["STUB_TRACE"], "a") as f:
    f.write(step + "\t" + " ".join(args) + "\n")
cfg = plan.get(step, {})
print(f"experiment  {exp} / {rid}")
if cfg.get("crash"):
    print("Traceback (most recent call last): a planted crash before any gate")
    sys.exit(4)
if exp == "private_weight_reference":
    seed, reps = int(opt("--seed")), many("--replicate-of")
    ratio = cfg.get("ratio", 0.95 + 0.01 * seed)
    gates = [("VALIDITY", "V7", cfg.get("V7", "PASS")), ("VALIDITY", "V8", cfg.get("V8", "PASS")),
             ("CLAIM", "C1", cfg.get("C1", "PASS"))]
    clock = {"native": 1965.0, "shared": 1965.0, "private": 1950.0}
    table = [{"arm": arm, "tiles": n, "sm_clock_load_mhz": clock[arm],
              "level_sides": ["low"] if arm == "private" and n <= cfg.get("low", 0) else []}
             for arm in clock for n in range(1, int(opt("--treads")) + 1)]
    payload = {"experiment": exp, "synthetic": False, "model": "mixtral_8x7b", "block_m": 32,
               "pinned": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                          "GROUP_SIZE_M": g, "num_warps": 8, "num_stages": 4},
               "treads": list(range(1, int(opt("--treads")) + 1)),
               "repeats": int(opt("--repeats")), "duty": float(opt("--duty")),
               "run_id": rid, "seed": seed, "ratio": ratio,
               "ratio_interval": [ratio - 0.01, ratio + 0.01],
               "replicates": ({"n": 1 + len(reps), "spread": 0.01 * len(reps),
                               "sd": 0.005 if len(reps) >= 2 else None,
                               "envelope": [0.94, ratio + 0.01], "verdict": "PASS",
                               "runs": [{"run_id": rid}]
                               + [{"run_id": json.loads(Path(r).read_text())["run_id"]}
                                  for r in reps]}
                              if reps else None),
               "align_probe": {"synthetic": False, "note": cfg.get("probe_note", ""),
                               "cells": [{"graph_calls": 16, "host_bound": False}] * 3},
               "card": "nvidia_testcard", "bandwidth_gbps": STUB_BANDWIDTH_GBPS,
               "memory_plan": {"per_copy_bytes": STUB_EXPERT_SET_BYTES},
               "ladders": {"shared": {"points": [[n, 0.15 + (0.52 + 0.001 * seed) * n]
                                                 for n in range(1, int(opt("--treads")) + 1)]}},
               "treads_table": table,
               "gates": [{"tag": t, "kind": k, "verdict": v} for k, t, v in gates]}
else:
    lo, hi = cfg.get("eta", [0.15, 0.24])
    gates = [("VALIDITY", "V7", cfg.get("V7", "PASS")), ("CLAIM", "C1", cfg.get("C1", "PASS"))]
    payload = {"run_id": rid, "pinned": {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": g},
               "duty": [float(d) for d in many("--duty")],
               "elasticity": {"value": (lo + hi) / 2, "lo": lo, "hi": hi},
               "gates": [{"number": t, "kind": k, "verdict": v} for k, t, v in gates]}
for k, t, v in gates:
    print(exit_codes.result_line(k, t, v, "planted"))
if not cfg.get("no_report"):
    out = Path(os.environ["MOE_RESULTS_DIR"]) / exp / rid
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(payload))
sys.exit(exit_codes.classify((k, t, v) for k, t, v in gates))
'''

#: What the planted ncu prints for --version, and the metric value an OPEN world
#: reads back: the tests read both off here, not off a second literal.
STUB_NCU_VERSION = "Version 2026.1.0.0 (planted)"
STUB_ERR = ("==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access"
            " NVIDIA GPU Performance Counters on the target device 0.")

STUB_PROBE = r'''#!@PYTHON@
"""A planted `dram_counter_route.py --probe`: the REAL do_probe, with only the
box planted -- the two capabilities, the module flag, nsys, the card, and the
bytes ncu and the probe child print in STUB_PLAN's "counter-probe" world.
Whether ncu is found at all is the real shutil.which, on the PATH the chain
handed this process; STUB_COUNTERS, when set, records that PATH."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "@ROOT@")
sys.path.insert(0, "@ROOT@/scripts")
import dram_counter_route as D

plan = json.loads(Path(os.environ["STUB_PLAN"]).read_text()) if os.environ.get("STUB_PLAN") else {}
cfg = plan.get("counter-probe", {})
if os.environ.get("STUB_COUNTERS"):
    with open(os.environ["STUB_COUNTERS"], "a") as f:
        f.write("counter-probe\t" + os.environ.get("PATH", "") + "\n")
if cfg.get("crash"):
    print("Traceback (most recent call last): a planted crash before any gate")
    sys.exit(4)
launched = f"{D.PK.MARKER} {D.PK.LAUNCHED} planted card: one add_"
header = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
worlds = {
    "open": (0, launched, "", header + f'"0","probe","{D.NCU_PROBE_METRIC}","byte","4194304"\n'),
    "blocked": (1, launched, "@ERR@", ""),
    "nodevice": (1, f"{D.PK.MARKER} {D.PK.NO_CUDA_DEVICE} torch.cuda.is_available() is False",
                 "", ""),
}
rc, out, err, log_text = worlds[cfg.get("world", "open")]


def planted_run(argv, timeout=60):
    if "--version" in argv:
        return 0, "NVIDIA (R) Nsight Compute Command Line Profiler\n@VERSION@\n", ""
    Path(argv[argv.index("--log-file") + 1]).write_text(log_text)
    return rc, out, err


D._run = planted_run
D.probe_capabilities = lambda: {"available": True, "cap_eff": "0xa80425fb",
                                "cap_eff_field": "00000000a80425fb", "cap_eff_bits": 64,
                                "sys_admin": False, "perfmon": False}
D.probe_module_flag = lambda: {"available": False, "why": "the parameter is not listed"}
D.probe_nsys = lambda: {"present": False, "why": "no nsys on PATH"}
D.live_card = lambda: "nvidia_testcard"
sys.exit(D.main(sys.argv[1:]))
'''

CARD = "nvidia_testcard"
UUID = "0a0a0a0a-1111-2222-3333-444444444444"


class Pod:
    """A planted world: a session root, a results root, the two interpreters."""

    def __init__(self, root: Path):
        self.root = root
        self.sessions = root / "session"
        self.results = root / "results" / f"gaps-{CARD}"
        self.trace = root / "trace.txt"
        self.plan = root / "plan.json"
        self.base = root / "py-base"
        self.base.write_text(STUB_BASE.replace("@PYTHON@", sys.executable)
                             .replace("@LADDER@", str(STUB_LADDER_S))
                             .replace("@PROBE@", str(STUB_PROBE_S))
                             .replace("@R1@", str(STUB_R1_S)))
        self.arm = root / "py-arm"
        self.arm.write_text(STUB_ARM.replace("@PYTHON@", sys.executable)
                            .replace("@ROOT@", str(ROOT))
                            .replace("@BANDWIDTH@", repr(STUB_BANDWIDTH_GBPS))
                            .replace("@EXPERT_SET@", repr(STUB_EXPERT_SET_BYTES)))
        self.probe = root / "py-probe"
        self.probe.write_text(STUB_PROBE.replace("@PYTHON@", sys.executable)
                              .replace("@ROOT@", str(ROOT)).replace("@ERR@", STUB_ERR)
                              .replace("@VERSION@", STUB_NCU_VERSION))
        self.base.write_text(self.base.read_text().replace("@COUNTER_PROBE@", str(self.probe)))
        #: where the counter probe's globs look: a directory of this world's own,
        #: so no ncu the test box carries under /usr/local or /opt is found
        self.ncu_root = root / "ncu-planted"
        for f in (self.base, self.arm, self.probe):
            f.chmod(0o755)
        self.set_plan({})

    def set_plan(self, plan: dict):
        self.plan.write_text(json.dumps(plan))

    def session(self, *rows, arms=(("thermal", "DONE"), ("calibrate", "DONE"),
                                   ("pin_probe-n64-g1", "DONE")), device=UUID,
                stamp="20260923T000000Z") -> Path:
        """A measuring session the preliminaries already ran in."""
        s = self.sessions / f"alpha_g-{CARD}-{stamp}"
        (s / "chain-logs").mkdir(parents=True)
        rows = rows or tuple(_row(n, "DONE") for n in
                             ("preflight-r1", "preflight-r3", "preconditions", "gpu-tests"))
        _ledger(s / "CHAIN.tsv", *rows)
        _ledger(s / "ARMS.tsv", *(_row(n, st) for n, st in arms), header=ARMS_HEADER)
        if device:
            (s / "DEVICE").write_text(device + "\n")
        return s

    def run(self, *args, **env) -> subprocess.CompletedProcess:
        full = chain_env(REPO=str(ROOT), PY_BASE=str(self.base), PY_VLLM=str(self.arm),
                          SESSION_ROOT=str(self.sessions), RESULTS_ROOT=str(self.root / "results"),
                          MOE_RESULTS_DIR=str(self.results), WORKSPACE=str(self.root),
                          END_SUITE="skip", G_LADDER="1 16", SEEDS="0 1 2",
                          NCU_SEARCH=f"{self.ncu_root}/*/ncu",
                          STUB_PLAN=str(self.plan), STUB_TRACE=str(self.trace))
        full.update(env)
        # a knob given as None is unset: the pass takes the chain's own default
        full = {k: v for k, v in full.items() if v is not None}
        return subprocess.run(["bash", str(CHAIN), *args], capture_output=True, text=True,
                              timeout=900, env=full)

    def traced(self) -> list[tuple[str, str]]:
        if not self.trace.exists():
            return []
        return [tuple(ln.split("\t", 1)) for ln in self.trace.read_text().splitlines()]


def _rows(path: Path) -> list[list[str]]:
    return [ln.split("\t") for ln in path.read_text().splitlines()[1:]]


@pytest.fixture(scope="module")
def two_passes(tmp_path_factory):
    """Pass 1: r1-g1 crashes; G=4's R1 page is INVALID (planted at the
    all-tread reading of session 4's G=16 cells, above V7's admissible edge);
    G=16's seed 0 reads V7 FAIL. Pass 2, --resume: r1-g1 lands, and nothing
    else is bought."""
    pod = Pod(tmp_path_factory.mktemp("pod"))
    s = pod.session()
    plan = {"r3-g1-s0": {"low": 2}, "r3-g16-s0": {"V7": "FAIL"},
            "r1-g4": {"V7": "FAIL", "C1": "FAIL", "eta": [1.1335, 1.2861]}}
    pod.set_plan({**plan, "r1-g1": {"crash": True}})
    first = pod.run("--resume", G_LADDER="1 4 16")
    after_first = {"trace": pod.traced(), "pairs": (s / "PAIRS.tsv").read_text(),
                   "ledger": _rows(s / "CHAIN.tsv")}
    pod.set_plan(plan)
    second = pod.run("--resume", G_LADDER="1 4 16")
    return pod, s, first, after_first, second


def test_a_measuring_pass_runs_the_owners_order_at_duty_0_25(two_passes):
    pod, s, first, after, _second = two_passes
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-1500:]
    steps = [step for step, _ in after["trace"]]
    assert steps == ["probe-check", "r3-g1-s0", "r3-g4-s0", "r3-g16-s0", "r1-g1", "r1-g4",
                     "r1-g16", "r3-g1-s1", "r3-g4-s1", "r3-g1-s2", "r3-g4-s2"], steps
    assert dict(after["trace"])["probe-check"] == "--probe-check --model mixtral-8x7b --block-m 32"
    args = dict(after["trace"])
    tag = s.name
    for step in ("r3-g1-s0", "r3-g16-s0", "r3-g1-s1", "r3-g4-s2"):
        assert "--duty 0.25" in args[step] and f"--session-tag {tag}" in args[step], args[step]
    assert "--duty 1.0 0.5 0.25" in args["r1-g1"], "session 5's states, the owner's"
    def rep(st):
        return str(pod.results / "private_weight_reference" / f"stub-{st}-{tag}" / "report.json")
    assert "--replicate-of" not in args["r3-g1-s0"]
    assert args["r3-g1-s1"].endswith(f"--replicate-of {rep('r3-g1-s0')}")
    assert args["r3-g1-s2"].endswith(f"--replicate-of {rep('r3-g1-s0')} {rep('r3-g1-s1')}")


def test_a_seed_0_v7_failure_skips_that_gs_later_seeds_unlatched(two_passes):
    _pod, _s, first, after, second = two_passes
    by = {}
    for r in after["ledger"]:
        by.setdefault(r[0], []).append(r)
    for seed in (1, 2):
        (row,) = by[f"r3-g16-s{seed}"]
        assert row[1] == "SKIPPED" and "r3-g16-s0) read V7 FAIL" in row[6], row
        assert "the two ratio arms ran at different clocks at duty 0.25" in row[6]
        assert ("A V7 FAIL at 0.25 means that duty is not yet flat for that arm on this card;"
                " the page names a lower duty; the chain skips the G's later seeds and prints"
                " the follow-up command") in row[6]
    assert by["r3-g16-s0"][-1][1] == "INVALID"
    assert by["r1-g1"][-1][1] == "ERROR"
    assert by["suite"][-1][1] == "SKIPPED" and "END_SUITE=skip" in by["suite"][-1][6]
    assert "not requested" in by["suite"][-1][6]
    # SKIPPED is not latched: the next pass asks again and skips again
    assert re.search(r"^r3-g16-s1\s+SKIPPED", second.stdout, re.M), second.stdout


def test_a_seed_0_v7_fail_prints_the_follow_up_at_a_lower_duty(two_passes):
    """XS-1, T2(i). R3's page names a lower duty on a V7 FAIL at 0.25; the chain
    skipped the G's later seeds and said nothing more, and a hand re-run at
    0.1 is a new design key outside the ledger. It now prints, once per G, the
    exact follow-up with its own R3 flags and session tag, priced off R3's own
    plan at 0.1, to run after the chain and read with --read on the laptop."""
    pod, s, first, _after, _second = two_passes
    tag = s.name
    fu = (s / "followup-g16.txt").read_text()
    runs = [ln for ln in fu.splitlines()
            if not ln.startswith("#") and "private_weight_reference.py" in ln and " > " in ln]
    assert len(runs) == 3
    for seed, ln in zip((0, 1, 2), runs, strict=True):
        assert ln.startswith(f"{pod.arm} {ROOT}/scripts/private_weight_reference.py --model"), ln
        assert (f"--treads 6 --repeats 9 --group-m 16 --duty 0.1 --seed {seed}"
                f" --session-tag {tag}") in ln
        assert ln.endswith(f"r3-g16-s{seed}-duty0.1.log 2>&1") and "--dry-run" not in ln
    assert "--replicate-of" not in runs[0]
    assert all(ln.split(" > ")[0].endswith('--replicate-of "$S0"') for ln in runs[1:])
    assert f"S0=\"$MOE_RESULTS_DIR/private_weight_reference/$({pod.base} " in fu
    assert f"export MOE_RESULTS_DIR={pod.results}" in fu
    per = _r3_price(0.1) + _const("R3_RUN_OVERHEAD_S")
    assert f"~{-(-3 * per // 60)} min: 3 runs at {per} s each" in fu, "R3's own plan at 0.1"
    assert "RUN THEM AFTER THE CHAIN HAS FINISHED, never beside it" in fu
    assert "--read" in fu and "outside the chain's ledger and PAIRS.tsv" in fu
    note = [r for r in _rows(s / "CHAIN.tsv") if r[0] == "r3-g16-s1"][0][6]
    assert runs[0] in note and "--read on the laptop, outside PAIRS.tsv" in note
    assert str(s / "followup-g16.txt") in note
    assert first.stdout.count("Its follow-up, by hand AFTER the chain") == 1, "once per G"
    assert runs[2] in first.stdout, "the whole block on the console"
    assert not (s / "followup-g1.txt").exists() and not (s / "followup-g4.txt").exists()


def test_the_table_is_rebuilt_every_pass_and_joins_the_r1_a_resume_landed(two_passes):
    pod, s, _first, after, second = two_passes
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-1500:]
    assert [st for st, _ in pod.traced()][len(after["trace"]):] == ["r1-g1"], "only the owed arm"
    header = H.reading_header() + ["eta", "eta_lo", "eta_hi", "band", "eta_exit", "reads_as"]
    keys = [("1", "0"), ("1", "1"), ("1", "2"), ("4", "0"), ("4", "1"), ("4", "2"), ("16", "0")]
    before = [dict(zip(header, r, strict=True)) for r in _rows_text(after["pairs"])]
    assert [(r["G"], r["seed"]) for r in before] == keys
    assert [r["band"] for r in before] == ["unmeasured"] * 3 + ["withheld:INVALID"] * 3 + [
        "RAW-STANDS"]
    now = [dict(zip(header, r, strict=True)) for r in _rows(s / "PAIRS.tsv")]
    assert [(r["G"], r["seed"]) for r in now] == keys, "one row per run, however many passes"
    assert [(r["band"], r["eta_exit"]) for r in now] == [("RAW-STANDS", "DONE")] * 3 + [
        ("withheld:INVALID", "INVALID")] * 3 + [("RAW-STANDS", "DONE")]
    assert [r["reads_as"] for r in now] == [H.READS_AS["RAW-STANDS"]] * 3 + [
        H.READS_AS_UNRESOLVED] * 3 + [H.READS_AS["RAW-STANDS"]], "R1's word, read as"
    assert [r["rep_n"] for r in now] == ["none", "2", "3"] * 2 + ["none"]
    assert [r["joint"] for r in now] == ["none", "PASS", "PASS"] * 2 + ["none"]
    assert [r["exit_scope"] for r in now] == ["alone", "envelope", "envelope"] * 2 + ["alone"]
    assert [r["low_cells"] for r in now] == ["2"] + ["0"] * 6
    assert {r["clk_private"] for r in now} == {"1950"}
    assert [r["exit"] for r in now] == ["DONE"] * 6 + ["INVALID"]
    fixed = {ln.split("\t")[0]: ln.split("\t")[1]
             for ln in (s / "PAIRS-fixed.tsv").read_text().splitlines()[1:]}
    assert fixed["r3_duty"] == "0.25" and fixed["r1_treads"] == "--treads 8"
    assert "GROUP_SIZE_M" not in fixed["r3_pinned"]


def test_the_console_line_names_r1s_exit_beside_a_withheld_word(two_passes):
    _pod, _s, first, _after, _second = two_passes
    line = next(ln for ln in first.stdout.splitlines() if ln.strip().startswith("G=4 seed 1"))
    assert "eta 1.2098 [1.1335, 1.2861] withheld:INVALID  (R1 page: INVALID)" in line, line
    assert "CLOCK-CARRIES" not in first.stdout, "a word read off a page its gates refused"


def _rows_text(text: str) -> list[list[str]]:
    return [ln.split("\t") for ln in text.splitlines()[1:]]


def test_a_measuring_pass_keeps_its_device_and_releases_its_lock(two_passes):
    _pod, s, first, _after, second = two_passes
    assert (s / "DEVICE").read_text().strip() == UUID
    assert not (s / "chain.lock.d").exists(), "the lock outlived the pass"
    assert f"device      {UUID}" in first.stdout
    assert "(RESUMED)" in second.stdout


def test_the_pilots_v8_stops_the_chain_and_past_v8_goes_on_recorded(tmp_path):
    pod = Pod(tmp_path)
    s = pod.session()
    pod.set_plan({"r3-g1-s0": {"V8": "FAIL"}})
    got = pod.run("--resume", SEEDS="0 1")
    assert got.returncode == 3, got.stdout[-3000:] + got.stderr[-1000:]
    assert "STOP: r3-g1-s0's V8 is FAIL, not PASS" in got.stdout
    assert "--resume --past-v8" in got.stdout
    assert [st for st, _ in pod.traced()] == ["probe-check", "r3-g1-s0"], \
        "nothing after the pilot was bought"
    # T1: what going on buys, priced off the arms' own (planted) plans, and the
    # probe's own record off the pilot's report
    assert "makes every later ratio page INVALID: the probe re-runs on" in got.stdout
    n_g, n_s, per = 2, 2, _r3_price(0.25) + _const("R3_RUN_OVERHEAD_S")
    later = n_g * n_s - 1
    assert f"R1's {n_g} regime words (~{-(-n_g * STUB_R1_S // 60)} min off R1's own" in got.stdout
    assert (f"{later} ratio pages that cannot be quoted (~{-(-later * per // 60)} min at {per} s"
            " a page") in got.stdout
    assert f"limits the ratio pages to seed 0 ({n_g - 1}, ~{-(-(n_g - 1) * per // 60)} min)" \
        in got.stdout
    assert "SEEDS=0 bash scripts/alpha_g_chain.sh --resume --past-v8" in got.stdout
    assert "the pilot's probe: note: none; graph_calls 16; host-bound 0 of 3" in got.stdout
    assert [(r[0], r[1]) for r in _rows(s / "PAIRS.tsv")] == [("1", "0")], "rebuilt before the STOP"
    again = pod.run("--resume", "--past-v8", SEEDS="0 1")
    assert again.returncode == 0, again.stdout[-3000:]
    over = [r for r in _rows(s / "CHAIN.tsv") if r[0] == "v8-override"]
    assert len(over) == 1 and over[0][1] == "OVERRIDDEN"
    assert "--past-v8" in over[0][6] and "r3-g1-s0's V8 FAIL" in over[0][6]
    assert [st for st, _ in pod.traced()] == ["probe-check", "r3-g1-s0", "r3-g16-s0", "r1-g1",
                                              "r1-g16", "r3-g1-s1", "r3-g16-s1"]
    # XS-7, T8: THE DECISION HOLDS. A plain --resume after it (a killed chain
    # relaunched the runbook's way) went back to the STOP it had gone past.
    later = pod.run("--resume", SEEDS="0 1")
    assert later.returncode == 0, later.stdout[-3000:]
    assert "--past-v8 holds from an earlier pass (the ledger's v8-override row" in later.stdout
    assert len([r for r in _rows(s / "CHAIN.tsv") if r[0] == "v8-override"]) == 1, \
        "a held decision writes no second row"


def _const(name: str) -> int:
    """One of the chain's plain integer constants, off its own line."""
    return int(re.search(rf"^{name}=(\d+)$", CODE, re.M).group(1))


def _driver_minutes(arm: str) -> int:
    """The driver's own booking for one of its arms, lifted from its
    `arm_minutes` the way the chain lifts it."""
    driver = ROOT / "scripts" / "h200_gaps_session.sh"
    got = subprocess.run(
        ["bash", "-c", f'eval "$(sed -n \'/^arm_minutes()/,/^esac; }}/p\' "{driver}")"; '
                       f"arm_minutes {arm}"],
        capture_output=True, text=True, timeout=60).stdout.split()
    assert got, arm
    return int(got[0])


def _r3_price(duty: float) -> int:
    """What the planted R3 plan page prices a run at: its wall line at the duty
    (rounded as the stub's awk rounds) plus the probe that line leaves out."""
    return math.floor(STUB_LADDER_S / duty + 0.5) + STUB_PROBE_S


def test_past_gpu_tests_holds_on_every_later_pass(tmp_path):
    """XS-7, T8. --past-gpu-tests counted for the pass it was given on: a plain
    --resume after it re-ran tests/test_gpu.py (up to 15 min on a hung volume)
    and stopped at the same gate again."""
    pod = Pod(tmp_path)
    s = pod.session(_row("preflight-r1", "DONE"), _row("preflight-r3", "DONE"),
                    _row("preconditions", "DONE"), _row("gpu-tests", "ERROR", "1", "3 failed"))
    seen = tmp_path / "pytest-seen.txt"
    stub = {"G_LADDER": "1", "SEEDS": "0", "STUB_PYTEST": str(seen), "STUB_PYTEST_RC": "1",
            "STUB_PYTEST_TALLY": "3 failed, 32 passed in 20.00s"}
    first = pod.run("--resume", "--past-gpu-tests", **stub)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-800:]
    later = pod.run("--resume", **stub)
    assert later.returncode == 0, later.stdout[-3000:] + later.stderr[-800:]
    assert ("--past-gpu-tests holds from an earlier pass (the ledger's gpu-tests-override row"
            in later.stdout)
    assert not seen.exists(), "tests/test_gpu.py was bought again after the operator went past it"
    over = [r for r in _rows(s / "CHAIN.tsv") if r[0] == "gpu-tests-override"]
    assert len(over) == 1 and over[0][1] == "OVERRIDDEN"


def test_the_probe_check_stops_the_chain_before_the_pilot_and_runs_until_done(tmp_path):
    """GPU-1. tests/test_gpu.py's one on-card test of the graph probe imports
    vLLM and skips from PY_BASE, the only interpreter the chain ran it from, so
    a probe the capture refused was found by the pilot's ten-minute ladder,
    latched INVALID. The probe check runs that check alone from the vLLM venv
    first, and runs again on every pass until it is DONE."""
    pod = Pod(tmp_path)
    s = pod.session()
    pod.set_plan({"probe-check": {"P1": "UNKNOWN"}})
    got = pod.run("--resume", G_LADDER="1", SEEDS="0 1")
    assert got.returncode == 3, got.stdout[-3000:] + got.stderr[-800:]
    assert "STOP: the probe check is INVALID, not DONE, on this card" in got.stdout
    assert "RESULT: VALIDITY P1 UNKNOWN [VALIDITY] planted probe" in got.stdout, "the page's line"
    assert str(s / "chain-logs" / "probe-check.log") in got.stdout
    assert "makes every later ratio page INVALID" in got.stdout
    assert f"{2} ratio pages that cannot be quoted" in got.stdout, "none is on disk yet"
    assert [st for st, _ in pod.traced()] == ["probe-check"], "the pilot's ladder was not bought"
    row = [r for r in _rows(s / "CHAIN.tsv") if r[0] == "probe-check"][-1]
    assert row[1:3] == ["INVALID", "3"] and "log agrees" in row[6]
    # the fix is a code change: a --resume proves the probe again, and a PASS goes on
    pod.set_plan({})
    again = pod.run("--resume", G_LADDER="1", SEEDS="0 1")
    assert again.returncode == 0, again.stdout[-3000:]
    assert [st for st, _ in pod.traced()][:3] == ["probe-check", "probe-check", "r3-g1-s0"]
    done = pod.run("--resume", G_LADDER="1", SEEDS="0 1")
    assert [st for st, _ in pod.traced()].count("probe-check") == 2, "a DONE check is not re-bought"
    assert done.returncode == 0


def test_a_refused_probe_check_stops_on_the_interpreter_and_offers_no_override(tmp_path):
    """CH-1. A REFUSED check timed nothing: R3 refuses it with no card, no
    vLLM, or a vLLM op that did not import, so the interpreter or the card is
    wrong, not the probe. The STOP gave a FAIL's advice: a code change to the
    probe, and --past-v8 priced as if ratio pages would run, from the same
    interpreter that just refused, and held for every later pass once taken."""
    pod = Pod(tmp_path)
    s = pod.session()
    pod.set_plan({"probe-check": {"refuse": True}})
    got = pod.run("--resume", G_LADDER="1 4 16 64")
    assert got.returncode == 3, got.stdout[-3000:] + got.stderr[-800:]
    stop = " ".join(got.stdout[got.stdout.index("STOP:"):].split())
    assert "STOP: the probe check is REFUSED, not DONE, on this card" in stop
    assert "REFUSED: no CUDA device (a planted refusal)" in stop, "the page's own line"
    assert "timed nothing" in stop and "the interpreter or the card" in stop
    assert f"PY_VLLM here is {pod.arm}" in stop, "the interpreter the arms run from"
    assert "torch wheel against the driver" in stop
    assert "then --resume, which runs this check again" in stop
    # none of a FAIL's advice: no probe code change, no priced override
    for fail_word in ("did not earn PASS", "PROBE_CALLS_PER_REPLAY", "ratio pages that cannot be",
                      "regime words", "SEEDS=0 bash scripts/alpha_g_chain.sh --resume --past-v8"):
        assert fail_word not in stop, fail_word
    assert "--past-v8 buys nothing here" in stop
    assert sorted(p.name for p in (s / "chain-logs").iterdir()) == [
        "counter-probe.log", "probe-check.log"], \
        "the arms' plans were run to price an override that buys nothing"
    assert not [r for r in _rows(s / "CHAIN.tsv") if r[0].endswith("-override")]
    # the interpreter fixed: a plain --resume proves the probe and goes on
    pod.set_plan({})
    again = pod.run("--resume", G_LADDER="1", SEEDS="0")
    assert again.returncode == 0, again.stdout[-3000:]
    assert [st for st, _ in pod.traced()][:3] == ["probe-check", "probe-check", "r3-g1-s0"]


def test_past_v8_goes_past_the_probe_check_recorded_and_the_decision_holds(tmp_path):
    pod = Pod(tmp_path)
    s = pod.session()
    pod.set_plan({"probe-check": {"P1": "FAIL"}})
    got = pod.run("--resume", G_LADDER="1", SEEDS="0")
    assert got.returncode == 3, got.stdout[-3000:]
    assert "RESULT: VALIDITY P1 FAIL [VALIDITY] planted probe" in got.stdout
    past = pod.run("--resume", "--past-v8", G_LADDER="1", SEEDS="0")
    assert past.returncode == 0, past.stdout[-3000:]
    over = [r for r in _rows(s / "CHAIN.tsv") if r[0] == "probe-check-override"]
    assert len(over) == 1 and over[0][1] == "OVERRIDDEN" and "--past-v8" in over[0][6]
    assert "the probe check's newest row: INVALID" in over[0][6]
    later = pod.run("--resume", G_LADDER="1", SEEDS="0")
    assert later.returncode == 0, later.stdout[-3000:]
    assert ("--past-v8 holds from an earlier pass (the ledger's probe-check-override row"
            in later.stdout)
    assert [st for st, _ in pod.traced()].count("probe-check") == 1, \
        "a check the operator went past is not run again"


def test_a_seed_0_page_that_compared_no_clocks_skips_its_seeds_without_blaming_a_split(tmp_path):
    """E2E-2, T9. After a V8 FAIL R3 skips its sweep, times nothing and reads V7
    UNKNOWN; the skip note blamed a clock split between the ratio arms, the
    power state, for a card whose clocks R3 never read."""
    pod = Pod(tmp_path)
    s = pod.session()
    pod.set_plan({"r3-g1-s0": {"V8": "FAIL", "V7": "UNKNOWN"}})
    got = pod.run("--resume", "--past-v8", G_LADDER="1", SEEDS="0 1")
    assert got.returncode == 0, got.stdout[-3000:]
    (row,) = [r for r in _rows(s / "CHAIN.tsv") if r[0] == "r3-g1-s1"]
    assert row[1] == "SKIPPED"
    assert ("read V7 UNKNOWN: seed 0's V7 could not be scored (nothing timed, e.g. the sweep"
            " was skipped on V8, or a clock was unread)") in row[6]
    assert str(s / "chain-logs" / "r3-g1-s0.log") in row[6]
    assert "different clocks" not in row[6] and "follow-up" not in row[6]
    assert not (s / "followup-g1.txt").exists()


def test_the_session_records_r3s_duty_and_refuses_a_resume_at_another(tmp_path):
    """T2(ii). R3_DUTY=0.1 on a --resume of a 0.25 session measured nothing:
    seed 0 was latched at 0.25, the skip read the 0.25 page again, and every
    owed later seed was refused against 0.25 reports."""
    pod = Pod(tmp_path)
    s = pod.session()
    duty = re.search(r'^R3_DUTY="\$\{R3_DUTY:-([0-9.]+)\}"$', CODE, re.M).group(1)
    first = pod.run("--resume", G_LADDER="1", SEEDS="0")
    assert first.returncode == 0, first.stdout[-3000:]
    assert (s / "R3_DUTY").read_text() == duty + "\n"
    ledger, traced = (s / "CHAIN.tsv").read_text(), pod.traced()
    other = pod.run("--resume", G_LADDER="1", SEEDS="0", R3_DUTY="0.1")
    assert other.returncode == 2, other.stdout[-2000:]
    assert f"ratio runs are at --duty {duty}" in other.stdout
    assert "R3_DUTY=0.1 bash scripts/alpha_g_chain.sh --new" in other.stdout
    assert pod.traced() == traced and (s / "CHAIN.tsv").read_text() == ledger
    assert (s / "R3_DUTY").read_text() == duty + "\n", "a refusal rewrites nothing"
    same = pod.run("--resume", G_LADDER="1", SEEDS="0", R3_DUTY=f"{float(duty):.3f}")
    assert same.returncode == 0, "one duty however it is spelled"


def test_ratio_rows_with_no_duty_record_are_refused(tmp_path):
    s = tmp_path / "s"
    _ledger(s / "CHAIN.tsv", _row("preflight-r1", "DONE"))
    assert lift(f"duty_check {s!s} 0.25; echo rc=$?").stdout.strip() == "rc=0"
    assert (s / "R3_DUTY").read_text() == "0.25\n"
    t = tmp_path / "t"
    _ledger(t / "CHAIN.tsv", _row("r3-g1-s0", "DONE"))
    got = lift(f"duty_check {t!s} 0.25; echo rc=$?")
    assert got.stdout.strip().endswith("rc=2") and "no R3_DUTY file" in got.stdout
    assert not (t / "R3_DUTY").exists()


def test_every_arm_step_runs_under_a_cap_off_its_own_price(tmp_path):
    """GPU-2. Every arm ran as a bare command with no deadline, while the
    pytest steps were capped because the volume's MooseFS has hung. The probe
    check, R3 and R1 now run under timeout at max(3 x the arm's own price,
    30 min), the plan run again before the step."""
    pod = Pod(tmp_path)
    pod.session()
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "timeout").write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$STUB_TIMEOUTS"\n'
                                  'shift 3\nexec "$@"\n')
    (shim / "timeout").chmod(0o755)
    seen = tmp_path / "timeouts.txt"
    got = pod.run("--resume", G_LADDER="1", SEEDS="0", STUB_TIMEOUTS=str(seen),
                  PATH=f"{shim}{os.pathsep}{os.environ['PATH']}")
    assert got.returncode == 0, got.stdout[-3000:] + got.stderr[-800:]
    factor, floor = _const("ARM_CAP_FACTOR"), _const("ARM_CAP_FLOOR_S")
    want = {"counter-probe": max(factor * 60 * _driver_minutes("counter_plan"), floor),
            "probe-check": max(factor * _const("PROBE_CHECK_S"), floor),
            "r3-g1-s0": max(factor * _r3_price(0.25), floor),
            "r1-g1": max(factor * STUB_R1_S, floor)}
    for step, cap in want.items():
        assert f"{step}: capped at {cap} s" in got.stdout, step
    calls = [ln.split()[:3] for ln in seen.read_text().splitlines()]
    assert calls == [["--signal=INT", "--kill-after=60", str(cap)] for cap in want.values()]


def test_a_pilot_with_no_report_stops_the_chain(tmp_path):
    logs = tmp_path / "chain-logs"
    logs.mkdir()
    (logs / "r3-g1-s0.log").write_text("Traceback: died before its plan page\n")
    ledger = _ledger(tmp_path / "CHAIN.tsv")
    got = lift("v8_gate 0; echo rc=$?", LEDGER=str(ledger), LOGS=str(logs),
               RESULTS=str(tmp_path / "res"), PILOT="r3-g1-s0", TAG="t", G_LADDER="1",
               SEEDS="0")
    assert got.stdout.strip().endswith("rc=3"), got.stdout + got.stderr
    assert "STOP: r3-g1-s0's V8 is unread: no report.json, not PASS" in got.stdout


def test_a_callers_session_does_not_steer_a_chain_this_file_spawns(tmp_path, monkeypatch):
    """On the pod this file runs inside the chain's end suite when END_SUITE=run
    buys it. A SESSION= in the caller's environment turned every --resume here
    into a refusal, and every plain pass here into a pass in the caller's
    directory."""
    real = tmp_path / "the-operators-session"
    monkeypatch.setenv("SESSION", str(real))
    monkeypatch.setenv("END_SUITE", "run")
    pod = Pod(tmp_path)
    s = pod.session()
    got = pod.run("--resume", G_LADDER="1", SEEDS="0")
    assert got.returncode == 0, got.stdout[-2500:] + got.stderr[-800:]
    assert f"alpha(G) chain  {s.name}   (RESUMED)" in got.stdout
    assert not real.exists()


def test_the_end_suite_sees_neither_the_opinion_nor_the_operators_knobs(tmp_path, monkeypatch):
    """The operator's documented `SESSION=<dir> bash scripts/alpha_g_chain.sh`,
    end to end: pytest is started without SESSION and every other knob the
    chain reads, and without an OPINION."""
    monkeypatch.delenv("OPINION", raising=False)
    pod = Pod(tmp_path)
    s = pod.session()
    seen = tmp_path / "pytest-seen.txt"
    got = pod.run(SESSION=str(s), G_LADDER="1", SEEDS="0", END_SUITE="run",
                  STUB_PYTEST=str(seen))
    assert got.returncode == 0, got.stdout[-2500:] + got.stderr[-800:]
    (line,) = seen.read_text().splitlines()
    names = set(line.split("\t", 1)[1].split())
    assert "STUB_PLAN" in names, "the planted world reaches pytest"
    assert names.isdisjoint([*KNOBS, "OPINION"]), sorted(names & {*KNOBS, "OPINION"})
    assert _rows(s / "CHAIN.tsv")[-1][:2] == ["suite", "DONE"]


def test_a_preflight_that_is_not_done_runs_again_on_every_pass(tmp_path):
    """An INVALID self-test latched, so every --resume STOPPED on it the same
    way, even after the scorer was fixed and checked out: --new or an edited
    ledger was the only way on. It runs again until it is DONE; a DONE one is
    not bought again."""
    pod = Pod(tmp_path)
    s = pod.session(_row("preflight-r1", "DONE"), _row("preflight-r3", "INVALID", "3"),
                    _row("preconditions", "DONE"), _row("gpu-tests", "DONE"))
    got = pod.run("--resume", G_LADDER="1", SEEDS="0")
    assert got.returncode == 0, got.stdout[-3000:] + got.stderr[-1000:]
    rows = _rows(s / "CHAIN.tsv")
    assert [r[1] for r in rows if r[0] == "preflight-r3"] == ["INVALID", "DONE"]
    assert [r[1] for r in rows if r[0] == "preflight-r1"] == ["DONE"]
    assert [st for st, _ in pod.traced()] == ["probe-check", "r3-g1-s0", "r1-g1"]


@pytest.mark.parametrize("rc,tally,state,recorded", [
    (0, "5 passed in 1.00s", "DONE", True),
    # ran to its end, some tests red: a record of the box, not a gate
    (1, "2 failed, 3 passed in 1.00s", "ERROR", True),
    # timed out, or interrupted before its end: no record, bought again
    (124, "2 failed, 3000 passed in 5400.00s", "ERROR", False),
    (2, "Interrupted: 1 error during collection", "ERROR", False),
])
def test_an_end_suite_that_ran_to_its_tally_is_not_bought_again(tmp_path, rc, tally, state,
                                                                recorded):
    """END_SUITE=skip wrote a SKIPPED row over a DONE one, and only the newest
    row counts, so the next plain --resume bought the suite again (54 min at
    the pod's rate); a red suite re-ran on every --resume. Passes: the suite
    run, then END_SUITE=skip, then a plain --resume."""
    pod = Pod(tmp_path)
    s = pod.session()
    seen = tmp_path / "pytest-seen.txt"
    stub = {"G_LADDER": "1", "SEEDS": "0", "STUB_PYTEST": str(seen),
            "STUB_PYTEST_RC": str(rc), "STUB_PYTEST_TALLY": tally}
    passes = [pod.run("--resume", END_SUITE=end, **stub) for end in ("run", "skip", "run")]
    for got in passes:
        assert got.returncode == 0, got.stdout[-3000:] + got.stderr[-1000:]
    suite = [r[1] for r in _rows(s / "CHAIN.tsv") if r[0] == "suite"]
    ran = len(seen.read_text().splitlines())
    if recorded:
        assert (suite, ran) == ([state], 1)
        for got in passes[1:]:
            assert f"suite recorded, not bought again: {state}: pytest exit {rc}" in got.stdout
    else:
        assert (suite, ran) == ([state, "SKIPPED", state], 2)


def test_the_end_suite_is_bought_only_on_end_suite_run(tmp_path):
    """The owner's decision in session 5: on its pod the base-venv suite
    exercised nothing the arms depend on beyond tests/test_gpu.py and ran 2499
    tests in 3472 s before it was interrupted at 49%. A pass that does not ask
    buys no suite and writes a SKIPPED row saying it was not requested;
    END_SUITE=run buys it. tests/test_gpu.py before the arms is unchanged."""
    pod = Pod(tmp_path)
    s = pod.session()
    seen = tmp_path / "pytest-seen.txt"
    stub = {"G_LADDER": "1", "SEEDS": "0", "STUB_PYTEST": str(seen)}
    got = pod.run("--resume", END_SUITE=None, **stub)
    assert got.returncode == 0, got.stdout[-3000:] + got.stderr[-1000:]
    assert not seen.exists(), "a pass that did not ask for the end suite bought it"
    row = _rows(s / "CHAIN.tsv")[-1]
    assert row[:2] == ["suite", "SKIPPED"], row
    assert "not requested" in row[6] and "END_SUITE=run" in row[6], row[6]
    got = pod.run("--resume", END_SUITE="run", **stub)
    assert got.returncode == 0, got.stdout[-3000:] + got.stderr[-1000:]
    assert len(seen.read_text().splitlines()) == 1, "END_SUITE=run buys the suite once"
    assert _rows(s / "CHAIN.tsv")[-1][:2] == ["suite", "DONE"]


@pytest.mark.parametrize("value", ["yes", "RUN"])
def test_an_end_suite_value_other_than_run_or_skip_is_refused(tmp_path, value):
    """With skip the default, a mistyped opt-in would skip the suite it asked
    for without a word. Any value but run or skip is refused, exit 2, before a
    step runs or a row is written."""
    pod = Pod(tmp_path)
    s = pod.session()
    before = (s / "CHAIN.tsv").read_text()
    seen = tmp_path / "pytest-seen.txt"
    got = pod.run("--resume", END_SUITE=value, G_LADDER="1", SEEDS="0", STUB_PYTEST=str(seen))
    assert got.returncode == 2, got.stdout[-2000:]
    assert f"REFUSED: END_SUITE={value}: it takes run" in got.stdout
    assert pod.traced() == [] and not seen.exists()
    assert (s / "CHAIN.tsv").read_text() == before


def test_a_driver_refusal_stops_the_chain_before_any_arm(tmp_path):
    """The real driver, pointed at this session: it cannot name the planted
    card, refuses with exit 2, and the chain stops, although the ARMS.tsv a
    previous pass left reads thermal and calibrate DONE."""
    pod = Pod(tmp_path)
    s = pod.session(_row("preflight-r1", "DONE"), _row("preflight-r3", "DONE"))
    got = pod.run("--resume")
    assert got.returncode == 3, got.stdout[-3000:] + got.stderr[-1000:]
    assert "STOP: the driver REFUSED this card or its ruler (exit 2): REFUSED:" in got.stdout
    assert pod.traced() == [], "an arm ran after the driver refused"
    rows = _rows(s / "CHAIN.tsv")
    assert rows[-1][:3] == ["preconditions", "REFUSED", "2"]
    assert (s / "PAIRS.tsv").exists(), "the table is rebuilt before a STOP"


def test_a_resume_on_another_card_is_refused_before_anything_runs(tmp_path):
    pod = Pod(tmp_path)
    s = pod.session(device="ffffffff-9999-8888-7777-666666666666")
    before = (s / "CHAIN.tsv").read_text()
    got = pod.run("--resume")
    assert got.returncode == 2, got.stdout[-2000:]
    assert ("was measured on the card ffffffff-9999-8888-7777-666666666666,"
            f" and this card is {UUID}") in got.stdout
    assert "--new" in got.stdout
    assert pod.traced() == [] and (s / "CHAIN.tsv").read_text() == before
    assert not (s / "chain.lock.d").exists(), "a refused pass released its lock"


def test_a_bare_run_beside_a_measuring_session_is_refused_with_the_commands(tmp_path):
    pod = Pod(tmp_path)
    s = pod.session()
    got = pod.run()
    assert got.returncode == 2, got.stdout[-2000:]
    assert f"REFUSED: a chain session for {CARD} already exists" in got.stdout
    for line in ("bash scripts/alpha_g_chain.sh --resume",
                 f"SESSION={s} bash scripts/alpha_g_chain.sh",
                 "bash scripts/alpha_g_chain.sh --new"):
        assert line in got.stdout, line
    assert sorted(p.name for p in pod.sessions.iterdir()) == [s.name], "a fresh session was opened"


def test_a_concurrent_chain_on_the_session_is_refused(tmp_path):
    pod = Pod(tmp_path)
    s = pod.session()
    (s / "chain.lock.d").mkdir()
    (s / "chain.lock.d" / "owner").write_text(f"{os.getpid()} {_hostname()}\n")
    got = pod.run("--resume", LOCK_TOOL="mkdir")
    assert got.returncode == 2, got.stdout[-2000:]
    assert "another chain holds" in got.stdout
    assert pod.traced() == []
    assert (s / "chain.lock.d" / "owner").read_text().split()[0] == str(os.getpid()), \
        "a refused pass removed another chain's lock"


def test_a_measuring_run_off_a_card_is_refused_with_the_probes_reason(tmp_path):
    env = chain_env(REPO=str(ROOT), SESSION_ROOT=str(tmp_path / "session"),
                    RESULTS_ROOT=str(tmp_path / "results"), WORKSPACE=str(tmp_path))
    env.pop("MOE_RESULTS_DIR", None)
    got = subprocess.run(["bash", str(CHAIN)], capture_output=True, text=True, timeout=300, env=env)
    assert got.returncode == 2, got.stdout[-2000:]
    assert "REFUSED: no CUDA device this chain can name" in got.stdout
    said = re.search(r"said: (.+)$", got.stdout, re.M)
    assert said and said.group(1).strip(), got.stdout
    assert "nvidia-smi" in got.stdout
    assert not (tmp_path / "session").exists()


@pytest.mark.parametrize("args,env", [(["--dry-run", "--resume"], {}),
                                      (["--dry-run"], {"SESSION": "named"})])
def test_a_dry_run_into_an_existing_session_is_refused(tmp_path, args, env):
    s = tmp_path / "session" / "alpha_g-nocard-20260923T000000Z"
    (s / "chain-logs").mkdir(parents=True)
    (s / "CHAIN.tsv").write_text(HEADER)
    (s / "chain-logs" / "r3-g1-s0.log").write_text("experiment  private_weight_reference / REAL\n")
    if env.get("SESSION") == "named":
        env = {"SESSION": str(s)}
    full = chain_env(REPO=str(ROOT), SESSION_ROOT=str(tmp_path / "session"),
                     RESULTS_ROOT=str(tmp_path / "results"), WORKSPACE=str(tmp_path),
                     MOE_RESULTS_DIR=str(tmp_path / "results" / "gaps-nocard"), **env)
    got = subprocess.run(["bash", str(CHAIN), *args], capture_output=True, text=True,
                         timeout=300, env=full)
    assert got.returncode == 2, got.stdout[-2000:]
    assert "--dry-run plans into a fresh directory of its own" in got.stdout
    assert (s / "chain-logs" / "r3-g1-s0.log").read_text() == \
        "experiment  private_weight_reference / REAL\n"
    assert sorted(p.name for p in (tmp_path / "session").iterdir()) == [s.name]


def _path_without_ncu(*first: Path) -> str:
    """This box's PATH with every directory that holds an ncu dropped, and
    `first` put in front: a pod running this file may carry a real ncu."""
    keep = [d for d in os.environ["PATH"].split(os.pathsep)
            if d and not (Path(d) / "ncu").exists()]
    return os.pathsep.join([*(str(f) for f in first), *keep])


def _counter_rows(s: Path) -> list[list[str]]:
    return [r for r in _rows(s / "CHAIN.tsv") if r[0] == "counter-probe"]


def test_the_counter_probe_is_informational_and_asks_again_on_every_pass(tmp_path):
    """Right after the preconditions, on a box with no ncu anywhere the chain
    looks: an INFO row whose note says ABSENT and where it looked, COUNTERS
    beside the probe's own payload, and the chain goes on. INFO is not a
    latched word, so the next pass asks again: a counter route belongs to
    the pod."""
    pod = Pod(tmp_path)
    s = pod.session()
    path = _path_without_ncu()
    first = pod.run("--resume", G_LADDER="1", SEEDS="0", PATH=path)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-800:]
    (row,) = _counter_rows(s)
    assert row[1:3] == ["INFO", "2"], row
    assert row[6].startswith("ABSENT: no ncu where the chain looks; ncu not on PATH and none at"
                             f" {pod.ncu_root}/*/ncu"), row[6]
    assert "informational, gates nothing" in row[6] and str(s / "COUNTERS") in row[6]
    assert "CAP_PERFMON clear, CAP_SYS_ADMIN clear (CapEff 00000000a80425fb)" in row[6]
    assert json.loads((s / "COUNTERS.json").read_text())["ncu"]["why"] == "no ncu on PATH", \
        "the probe's own payload, from the probe's own code"
    text = (s / "COUNTERS").read_text()
    assert "verdict     ABSENT" in text and f"searched    PATH, then {pod.ncu_root}/*/ncu" in text
    assert "probe note  no ncu here" in text, "the probe's own notes ride along"
    assert [st for st, _ in pod.traced()] == ["probe-check", "r3-g1-s0", "r1-g1"]
    assert lift(f"latched counter-probe {s / 'CHAIN.tsv'!s} || echo no").stdout.strip() == "no"
    # the order: after the preconditions, before tests/test_gpu.py
    order = first.stdout.index
    assert order("== preconditions") < order("== counter-probe") < order("== tests/test_gpu.py")
    again = pod.run("--resume", G_LADDER="1", SEEDS="0", PATH=path)
    assert again.returncode == 0, again.stdout[-3000:]
    assert [r[1] for r in _counter_rows(s)] == ["INFO", "INFO"], "asked again, never latched"


def test_an_ncu_off_path_is_probed_with_its_directory_first_on_path(tmp_path):
    """An ncu under a searched directory, off PATH: the probe runs with its
    directory first on PATH, so the probe's own code finds it and reads the
    counter, and COUNTERS says the driver's counter_plan and --run need the
    same PATH."""
    pod = Pod(tmp_path)
    s = pod.session()
    ncu = _exe(pod.ncu_root / "2026.1" / "ncu")
    seen = tmp_path / "probe-path.txt"
    got = pod.run("--resume", G_LADDER="1", SEEDS="0", PATH=_path_without_ncu(),
                  STUB_COUNTERS=str(seen))
    assert got.returncode == 0, got.stdout[-3000:] + got.stderr[-800:]
    (probed,) = seen.read_text().splitlines()
    assert probed.split("\t", 1)[1].startswith(f"{ncu.parent}{os.pathsep}"), probed
    payload = json.loads((s / "COUNTERS.json").read_text())
    assert payload["verdict"] == "OPEN" and payload["ncu"]["binary"] == str(ncu)
    (row,) = _counter_rows(s)
    assert row[1:3] == ["INFO", "0"], row
    assert row[6].startswith(f"OPEN: {payload['ncu']['cause']}; ncu {ncu} (NOT on PATH; found by"
                             f" {pod.ncu_root}/*/ncu"), row[6]
    assert f"[{STUB_NCU_VERSION}]" in row[6], "the version the probe read"
    text = (s / "COUNTERS").read_text()
    assert f"PATH={ncu.parent}:$PATH" in text and "look on PATH only" in text


def test_a_refused_counter_quotes_the_exact_error_and_the_chain_goes_on(tmp_path):
    """ERR_NVGPUCTRPERM, as one pod gave it on 2026-09-15: the note carries
    ncu's line verbatim and the two capabilities, the row is INFO and not a
    CLAIM_FAIL, and every arm after it still runs."""
    pod = Pod(tmp_path)
    s = pod.session()
    ncu = _exe(tmp_path / "bin" / "ncu")
    pod.set_plan({"counter-probe": {"world": "blocked"}})
    got = pod.run("--resume", G_LADDER="1", SEEDS="0 1", PATH=_path_without_ncu(ncu.parent))
    assert got.returncode == 0, got.stdout[-3000:] + got.stderr[-800:]
    (row,) = _counter_rows(s)
    assert row[1:3] == ["INFO", str(exit_codes.CLAIM_FAIL)], "the probe's own exit, not the row's"
    assert row[6].startswith(f"BLOCKED: {STUB_ERR}; ncu {ncu} (on PATH)"), row[6]
    assert "CAP_PERFMON clear, CAP_SYS_ADMIN clear" in row[6]
    assert "counter-probe    INFO       rc=1" in got.stdout
    assert "--cap-add=PERFMON" in (s / "COUNTERS").read_text(), "the probe's own next ask"
    assert [st for st, _ in pod.traced()] == ["probe-check", "r3-g1-s0", "r1-g1", "r3-g1-s1"]


def test_every_rebuild_prints_the_bound_with_its_ceiling_and_what_the_ratio_reads_as(tmp_path):
    """End to end: the ruler the session's calibrate wrote, each G's bound at
    read_stream and at the pin rate off the reports on disk, and on each run's
    console line what its ratio reads as once R1 has spoken."""
    pod = Pod(tmp_path)
    s = pod.session()
    read, pin = 4400.0, 4800.0
    ruler = _ruler(tmp_path / "calibration" / "run-1" / f"measured_{CARD}.yaml",
                   named=STUB_BANDWIDTH_GBPS, read=read, pin=pin)
    (s / "logs").mkdir()
    (s / "logs" / "calibrate.log").write_text(f"[calibrate] wrote {ruler}\n")
    pod.set_plan({"r1-g1": {"eta": [0.55, 0.70], "C1": "FAIL"}})
    got = pod.run("--resume", G_LADDER="1", SEEDS="0 1")
    assert got.returncode == 0, got.stdout[-3000:] + got.stderr[-800:]
    line = next(ln for ln in got.stdout.splitlines() if ln.strip().startswith("G=1 seed 1"))
    assert line.endswith(f"reads as: {H.READS_AS['CLOCK-CARRIES']}"), line
    seed0 = next(ln for ln in got.stdout.splitlines() if ln.strip().startswith("G=1 seed 0"))
    assert seed0.endswith(f"reads as: {H.READS_AS_UNRESOLVED}"), "R1 had not run yet"
    tops = []
    for st in ("r3-g1-s0", "r3-g1-s1"):
        rep = json.loads((pod.results / "private_weight_reference" / f"stub-{st}-{s.name}"
                          / "report.json").read_text())
        tops.append(max(rep["ladders"]["shared"]["points"]))
    n, t = tops[0][0], statistics.mean(p[1] for p in tops)
    want = [H.bytes_rate_bound(t, n, STUB_EXPERT_SET_BYTES, c) for c in (read, pin)]
    flat = " ".join(got.stdout.split())
    assert f"bytes-rate bound, at the ceilings of {ruler}" in flat
    assert (f"G=1 alpha <= {want[0]:.4f} at read_stream {read:.1f} GB/s, <= {want[1]:.4f} at the"
            f" pin rate {pin:.1f} GB/s") in flat
    row = dict(zip(H.BY_G_HEADER, _rows(s / "PAIRS-by-G.tsv")[0], strict=True))
    assert (row["reads_as"], row["bound_pin_rate"]) == (H.READS_AS["CLOCK-CARRIES"],
                                                        f"{want[1]:.4f}")


# --------------------------------------------------------------------------
# the laptop dry run: priced off the arms' own plans, nothing written into the tree
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    root = tmp_path_factory.mktemp("chain")
    env = chain_env(REPO=str(ROOT), PY_BASE=sys.executable, PY_VLLM=sys.executable,
                    SESSION_ROOT=str(root / "session"), RESULTS_ROOT=str(root / "results"),
                    MOE_RESULTS_DIR=str(root / "results" / "gaps-nocard"), WORKSPACE=str(root))
    before = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    got = subprocess.run(["bash", str(CHAIN), "--dry-run"], capture_output=True,
                         text=True, timeout=1500, env=env)
    after = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    return got, root, before, after


def test_the_dry_run_prices_every_step_off_the_arms_own_plans(dry):
    got, root, _b, _a = dry
    assert got.returncode == 0, got.stdout[-2500:] + got.stderr[-800:]
    out = got.stdout
    assert "DRY RUN: every arm's own --dry-run is run and priced" in out
    for g in (1, 4, 16, 64):
        assert re.search(rf"^r1-g{g}\s", out, re.M), out
        for seed in (0, 1, 2):
            assert re.search(rf"^r3-g{g}-s{seed}\s", out, re.M), out
    assert "r1-g32" not in out
    assert out.count("off its own plan") == 4 + 12, "one priced line per arm, none for preflights"
    assert "PRICE, off the arms' own plans: 4 elasticity runs + 12 ratio runs" in out
    assert re.search(r"= \d+ min; at \$4\.59/h about \$\d+\.\d\d\. Book \d h\.", out), out
    assert "an elasticity run is its plan's wall figure at duty 1.0 0.5 0.25" in out
    # tests/test_gpu.py is COLLECTED and priced at the pod's rate, never run by a
    # dry run; the end suite, not requested by default, is neither
    m = re.search(r"tests/test_gpu\.py ~(\d+) s \((\d+) tests\) before them at ([\d.]+) s a"
                  r" test,\s+session 5's pod rate, and no end suite \(END_SUITE=skip, the default",
                  out)
    assert m, out
    rate = float(m.group(3))
    assert int(m.group(1)) == _priced(int(m.group(2)), rate) and int(m.group(2)) > 10
    assert "the end suite ~" not in out
    session = next((root / "session").glob("alpha_g-nocard-*"))
    ledger = (session / "CHAIN-dryrun.tsv").read_text().splitlines()
    names = [ln.split("\t")[0] for ln in ledger[1:]]
    assert names[:6] == ["preflight-r1", "preflight-r3", "preconditions", "counter-probe",
                         "gpu-tests", "probe-check"]
    assert names[6:10] == [f"r3-g{g}-s0" for g in (1, 4, 16, 64)]
    assert names[10:14] == ["r1-g1", "r1-g4", "r1-g16", "r1-g64"]
    assert names[14:22] == [f"r3-g{g}-s{s}" for s in (1, 2) for g in (1, 4, 16, 64)]
    assert names[22:] == ["suite"]
    rows = {ln.split("\t")[0]: ln.split("\t") for ln in ledger[1:]}
    assert rows["probe-check"][1] == "SKIPPED" and "a dry run does not run it" in rows[
        "probe-check"][6], "the probe check times the card: a dry run prices it, never runs it"
    assert rows["preflight-r1"][1] == "DONE" and "log agrees" in rows["preflight-r1"][6]
    assert rows["preconditions"][1] == "DONE" and "ARMS.tsv" in rows["preconditions"][6]
    assert rows["gpu-tests"][1] == "DONE" and "tests collected" in rows["gpu-tests"][6]
    assert rows["suite"][1] == "SKIPPED" and "not requested" in rows["suite"][6], rows["suite"]
    assert not (session / "chain-logs" / "suite.log").exists(), "the suite was collected"
    assert rows["r3-g1-s0"][1] == "REFUSED", "a dry-run plan scores no gate: REFUSED"
    assert not (session / "CHAIN.tsv").exists() and not (session / "DEVICE").exists()


def test_the_dry_run_prices_the_counter_probe_and_runs_nothing(dry):
    """No card, nothing measured: the counter probe is a SKIPPED row priced in
    minutes off the driver's own booking for the same probe, and the capped
    ceiling it would run under on the pod. Nothing it would write exists."""
    got, root, _b, _a = dry
    session = next((root / "session").glob("alpha_g-nocard-*"))
    rows = {ln.split("\t")[0]: ln.split("\t")
            for ln in (session / "CHAIN-dryrun.tsv").read_text().splitlines()[1:]}
    price = 60 * _driver_minutes("counter_plan")
    cap = max(_const("ARM_CAP_FACTOR") * price, _const("ARM_CAP_FLOOR_S"))
    row = rows["counter-probe"]
    assert row[1] == "SKIPPED" and "a dry run does not run it" in row[6], row
    assert (f"Priced ~{price} s, the driver's own arm_minutes for counter_plan (the same probe);"
            f" capped on the pod at {cap} s") in row[6]
    assert "informational, it gates nothing" in row[6]
    for made in ("COUNTERS", "COUNTERS.json", "chain-logs/counter-probe.log"):
        assert not (session / made).exists(), made


def test_the_dry_runs_price_names_every_term_and_draws_the_seed_spacing(dry):
    """XS-8, E2E-5, T7. The preconditions were 4 min against the driver's own 8,
    each ratio run left out the probe its plan prices on top of the wall line,
    and the prose said the seeds were 'an hour or more apart'. Every term is
    now read off its source, and the spacing is these prices' own timeline."""
    got, root, _b, _a = dry
    out = got.stdout
    logs = next((root / "session").glob("alpha_g-nocard-*")) / "chain-logs"
    page = (logs / "r3-g16-s2.log").read_text()
    wall = int(re.search(r"takes about (\d+) s", page).group(1))
    probe = int(re.search(r"includes the alignment probe's (\d+) s", page).group(1))
    assert int(H.estimate(logs / "r3-g16-s2.log")) == wall + probe
    pre = 60 * sum(_driver_minutes(a) for a in ("thermal", "calibrate", "pin_probe-n64-g1"))
    counter = 60 * _driver_minutes("counter_plan")
    assert (f"the counter probe ~{counter} s (the driver's own arm_minutes for counter_plan"
            in " ".join(out.split()))
    assert f"plus the preconditions ~{pre} s (the driver's own arm_minutes" in out
    priced = [int(x) for x in re.findall(r"priced (\d+) s off its own plan", out)]
    arms = int(re.search(r"= (\d+) s of arms", out).group(1))
    assert sum(priced) == arms
    gpu = int(re.search(r"tests/test_gpu\.py ~(\d+) s", out).group(1))
    total = int(re.search(r"\(an allowance\) = (\d+) s", out).group(1))
    overhead = _const("R3_RUN_OVERHEAD_S")
    assert total == (arms + gpu + pre + counter + _const("PROBE_CHECK_S") + 12 * overhead
                     + _const("EXFIL_S")), "no end suite in the default price"
    # the unpriced cap is four times the longest arm's price, R1's at its states
    r1 = [int(x) for x in re.findall(r"^r1-g\d+\s.*\n\s+priced (\d+) s off its own plan",
                                     out, re.M)]
    assert len(r1) == 4 and max(r1) == max(priced)
    assert _const("ARM_CAP_UNPRICED_S") >= 4 * max(r1)
    steps = re.findall(r"^(r[13]-g\d+(?:-s\d)?)\s", out, re.M)
    assert len(steps) == len(priced) == 16
    clock, start = pre + counter + gpu + _const("PROBE_CHECK_S"), {}
    for step, secs in zip(steps, priced, strict=True):
        start[step] = clock
        clock += secs + (overhead if step.startswith("r3") else 0)
    for a, b in ((0, 1), (1, 2)):
        gaps = {(start[f"r3-g{g}-s{b}"] - start[f"r3-g{g}-s{a}"] + 30) // 60
                for g in (1, 4, 16, 64)}
        assert len(gaps) == 1, gaps
        assert f"seed {a} to seed {b} ~{gaps.pop()} min" in out
    fu_log = logs / "followup-g1.price.log"
    assert "duty0.1" in H.run_id(fu_log, "private_weight_reference")
    per = int(H.estimate(fu_log)) + overhead
    assert f"~{-(-3 * per // 60)} min a G, 3 seeds at --duty 0.1 at {per} s each" in out


@pytest.fixture(scope="module")
def dry_with_suite(tmp_path_factory):
    """A dry run that asks for the end suite, over one G and one seed: the
    suite is collected and priced, never run."""
    root = tmp_path_factory.mktemp("chain-suite")
    env = chain_env(REPO=str(ROOT), PY_BASE=sys.executable, PY_VLLM=sys.executable,
                    SESSION_ROOT=str(root / "session"), RESULTS_ROOT=str(root / "results"),
                    MOE_RESULTS_DIR=str(root / "results" / "gaps-nocard"), WORKSPACE=str(root),
                    END_SUITE="run", G_LADDER="1", SEEDS="0")
    got = subprocess.run(["bash", str(CHAIN), "--dry-run"], capture_output=True,
                         text=True, timeout=1500, env=env)
    return got, root


def test_a_dry_run_on_end_suite_run_collects_and_prices_the_suite(dry_with_suite):
    got, root = dry_with_suite
    assert got.returncode == 0, got.stdout[-2500:] + got.stderr[-800:]
    out = got.stdout
    m = re.search(r"tests/test_gpu\.py ~(\d+) s \((\d+) tests\) before them and the end suite"
                  r" ~(\d+) s\s+\((\d+) tests\) after them \(END_SUITE=run\), at ([\d.]+) s a"
                  r" test, session 5's pod rate", out)
    assert m, out
    rate = float(m.group(5))
    gpu, suite = int(m.group(1)), int(m.group(3))
    assert gpu == _priced(int(m.group(2)), rate) and int(m.group(2)) > 10
    assert suite == _priced(int(m.group(4)), rate) and int(m.group(4)) > 1000
    arms = int(re.search(r"= (\d+) s of arms", out).group(1))
    pre = int(re.search(r"plus the preconditions ~(\d+) s", out).group(1))
    total = int(re.search(r"\(an allowance\) = (\d+) s", out).group(1))
    assert total == (arms + gpu + suite + pre + 60 * _driver_minutes("counter_plan")
                     + _const("PROBE_CHECK_S") + _const("R3_RUN_OVERHEAD_S") + _const("EXFIL_S"))
    session = next((root / "session").glob("alpha_g-nocard-*"))
    rows = {ln.split("\t")[0]: ln.split("\t")
            for ln in (session / "CHAIN-dryrun.tsv").read_text().splitlines()[1:]}
    assert rows["suite"][1] == "DONE" and "tests collected" in rows["suite"][6], rows["suite"]
    # the hang cap sits above the priced run: under it, an honest suite is interrupted
    timeout = re.search(r'^SUITE_TIMEOUT_S="\$\{SUITE_TIMEOUT_S:-(\d+)\}"$', CODE, re.M)
    assert timeout and int(timeout.group(1)) > suite, (timeout, suite)


def test_the_dry_run_lines_carry_the_duty_the_swizzle_and_the_seed(dry):
    got, root, _b, _a = dry
    session = next((root / "session").glob("alpha_g-nocard-*"))
    logs = session / "chain-logs"
    r3 = (logs / "r3-g16-s2.log").read_text()
    assert re.search(r"^duty\s+0\.25", r3, re.M), "the ratio arm plans at duty 0.25"
    assert "duty0.25" in H.run_id(logs / "r3-g16-s2.log", "private_weight_reference")
    assert "'GROUP_SIZE_M': 16" in r3        # the ratio arm prints its pinned dict
    r1 = (logs / "r1-g1.log").read_text()
    assert re.search(r"GROUP_SIZE_M=1\b", r1) and "GROUP_SIZE_M=16" not in r1   # R1 prints k=v
    assert "duty 1, 0.5, 0.25" in r1, "R1 plans at the chain's default states"
    # both preflights ran the scorer and said so
    assert "SELF-TEST OK" in (logs / "preflight-r3.log").read_text()
    assert re.search(r"^preflight-r1\s+DONE", got.stdout, re.M), got.stdout


def test_the_dry_run_prints_the_pod_checkout_the_detached_launch_and_the_exfil(dry):
    out = dry[0].stdout
    assert ("git -C /workspace/moe-kernels checkout --"
            " moe/bench/hardware/measured_nvidia_h200.yaml") in out
    assert "checkout -B r3-align origin/r3-align" in out and "log -1 --format=%h" in out
    assert ("nohup setsid bash scripts/alpha_g_chain.sh >> /workspace/alpha_g_chain.out 2>&1"
            " < /dev/null &") in out, "the launch APPENDS: a --resume keeps the first console"
    assert "tail -f /workspace/alpha_g_chain.out" in out
    tar = next(ln for ln in out.splitlines() if "exfil-alpha_g-nocard.tar.gz" in ln)
    assert '"moe/bench/hardware/measured_nocard.yaml"' in tar, "the ruler rides with the results"
    assert "results/calibration/" in tar, "and calibrate's own run directory"
    assert f'-C "{dry[1]}" "alpha_g_chain.out"' in tar, "and the console the launch appends to"
    assert "commit it with the results" in out


def test_a_results_dir_without_the_card_is_refused_before_anything_is_made(tmp_path):
    """The driver's own rule, applied by the chain first: a MOE_RESULTS_DIR that
    does not carry the card is refused, exit 2, and no session directory is
    opened. The suite's results sandbox is exactly such a directory, so without
    this the chain would open a session and then watch the driver refuse."""
    env = chain_env(REPO=str(ROOT), PY_BASE=sys.executable, PY_VLLM=sys.executable,
                    SESSION_ROOT=str(tmp_path / "session"), RESULTS_ROOT=str(tmp_path / "results"),
                    MOE_RESULTS_DIR=str(tmp_path / "plain"), WORKSPACE=str(tmp_path))
    got = subprocess.run(["bash", str(CHAIN), "--dry-run"], capture_output=True, text=True,
                         timeout=600, env=env)
    assert got.returncode == 2, got.stdout[-1500:]
    assert "REFUSED: MOE_RESULTS_DIR=" in got.stdout
    assert "does not contain the card 'nocard'" in got.stdout
    assert not (tmp_path / "session").exists(), "refused, yet a session directory was opened"
    assert "alpha(G) chain" not in got.stdout


def test_the_dry_run_writes_nothing_into_the_tree(dry):
    _got, _root, before, after = dry
    assert after == before, "the chain's dry run dirtied the checkout"

