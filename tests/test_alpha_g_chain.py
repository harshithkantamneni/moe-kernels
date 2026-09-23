"""scripts/alpha_g_chain.sh, checked without a pod.

The chain sequences the alpha(G) matrix session: preflight, the driver's
preconditions, tests/test_gpu.py on the card, the ratio at seed 0 at every G
(the first run's V8 read before anything else is bought), the clock
elasticity at every G, the later seeds scored with the earlier ones at
--duty 0.25, and the whole suite at the end as a record. What this file pins:
the three shell habits this project has been burned by, the ledger's second
opinion (the driver's rule, lifted), every gate asking for DONE and not for
"latched", the session a pass lands in, the card it is measured on and the lock
it holds, pairing only with reports that formed a ratio, a table rebuilt from
the reports on disk, the elasticity band read through the arm's own `band_of`
and withheld from a page its gates refused, and a laptop dry run that prices
every step off the arms' own plans and writes nothing into the tree.

The measuring path is driven end to end off GPU through two planted
interpreters: a base one that names a planted card and UUID and is the real
interpreter for everything else, and an arm one that prints the plan line and
the RESULT lines and writes report.json, which is all the chain reads.
"""
from __future__ import annotations

import json
import os
import re
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
    runs inside the chain's end suite, and an operator's SESSION=<dir> or
    END_SUITE=skip in the caller's environment must not steer a chain spawned
    here into the real session."""
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


def test_the_order_is_the_owners_and_only_test_gpu_is_gated():
    """D3: preflight, preconditions, tests/test_gpu.py (gated), R3 seed 0 at
    every G with the pilot's V8 read, R1 at every G, R3's later seeds, then
    the whole suite uncapped, from PY_BASE, gating nothing."""
    marks = ['echo "== preflight', 'echo "== preconditions', 'echo "== tests/test_gpu.py',
             'echo "== the re-read fraction, off the cap, seed', 'echo "== clock elasticity',
             'echo "== the re-read fraction, the later seeds', 'echo "== the whole suite']
    at = [CODE.index(m) for m in marks]
    assert at == sorted(at), dict(zip(marks, at, strict=True))
    gpu = CODE[at[2]:at[3]]
    assert "tests/test_gpu.py -q -rfE -p no:cacheprovider" in gpu
    assert 'gpu_tests_gate "$PAST_GPU_TESTS" || stop_chain' in gpu
    pilot = CODE[at[3]:at[4]]
    assert 'v8_gate "$PAST_V8" || stop_chain' in pilot
    suite = CODE[at[6]:CODE.index("# 8. the table")]
    assert ('pytest_step suite "$SUITE_TIMEOUT_S" tests/ -q -rfE --durations=25'
            ' -p no:cacheprovider') in suite
    assert "--maxfail" not in suite and "-x " not in suite, "the suite on the pod is uncapped"
    assert "stop_chain" not in suite and "exit 3" not in suite, "the end suite gates nothing"
    assert 'R3_DUTY="${R3_DUTY:-0.25}"' in CODE


def test_help_prints_the_whole_header_and_no_code():
    got = subprocess.run(["bash", str(CHAIN), "--help"], capture_output=True, text=True,
                         timeout=60, env=chain_env(REPO=str(ROOT)))
    assert got.returncode == 0
    for flag in ("--past-gpu-tests", "--past-v8", "--new", "--resume", "END_SUITE=skip"):
        assert flag in got.stdout, flag
    assert "WHAT IT RUNS" in got.stdout and "THE REGIME WORD" in got.stdout
    for word in ("RAW-STANDS", "UNREGISTERED-GAP", "CLOCK-CARRIES", "STRADDLES", "withheld:<EXIT>"):
        assert word in got.stdout, word
    assert "CLOCK-CARRIES is only an upper bound" in got.stdout, "finding 28's caveat"
    assert "RUNS FOR THE RECORD AND IS NOT GATED" in got.stdout
    assert "nohup setsid bash scripts/alpha_g_chain.sh" in got.stdout
    assert "checkout -B r3-align origin/r3-align" in got.stdout
    assert "no PID variable" in got.stdout, "the header's last line"
    assert "set -uo pipefail" not in got.stdout


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


@pytest.mark.skipif(subprocess.run(["bash", "-c", "command -v flock"],
                                   capture_output=True).returncode != 0,
                    reason="this box has no flock; the mkdir lock is the one it takes")
def test_the_flock_lock_refuses_while_held(tmp_path):
    s = tmp_path / "s"
    s.mkdir()
    holder = subprocess.Popen(["bash", "-c", f'exec 9>>"{s}/chain.lock"; flock 9; '
                               f'echo "$$ {_hostname()}" > "{s}/chain.lock"; sleep 30'])
    try:
        for _ in range(100):
            if (s / "chain.lock").exists() and (s / "chain.lock").read_text().strip():
                break
            subprocess.run(["sleep", "0.1"])
        got = lift(f"chain_lock {s!s} 1; echo rc=$?", LOCK_TOOL="flock")
        assert got.stdout.strip().endswith("rc=2") and "another chain holds" in got.stdout
    finally:
        holder.kill()
        holder.wait()
    got = lift(f"chain_lock {s!s} 1; echo rc=$?", LOCK_TOOL="flock")
    assert got.stdout.strip().endswith("rc=0"), got.stdout


# --------------------------------------------------------------------------
# the helpers: every read of a report, a log or a ledger
# --------------------------------------------------------------------------

def _report(tmp_path, name, *, G=1, seed=0, ratio=0.9551, lo=0.9544, hi=0.9695,
            synthetic=False, experiment="private_weight_reference", duty=0.25,
            replicates=None, table=None, v7="PASS", v8="PASS"):
    payload = {"experiment": experiment, "synthetic": synthetic, "model": "mixtral_8x7b",
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
    assert row[:8] == ["16", "2", "0.9551", "0.9544", "0.9695", "CLAIM_FAIL", "0.25", "run-a"]
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
    assert got["low_cells"] == "none"
    assert H.reading(str(tmp_path / "missing.json")) == ["unreadable"]


def _r1(tmp_path, name, lo, hi, *verdicts):
    p = tmp_path / name / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    gates = [{"kind": k, "number": n, "verdict": v} for k, n, v in verdicts]
    p.write_text(json.dumps({"elasticity": {"value": (lo + hi) / 2 if lo is not None else None,
                                            "lo": lo, "hi": hi},
                             "duty": [1.0, 0.7, 0.5], "gates": gates}))
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
    # session 4's G=16 shape: above R1's admissible ceiling, V7 FAIL, the page INVALID
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
                   "estimated GPU time 154 s at the model's own timings\n"
                   "WALL CLOCK at duty 0.25: the ladder's 145 s of kernel time takes about 581 s,"
                   " the idle gaps\n")
    assert H.run_id(str(log), "private_weight_reference") == "nvidia_h200-bm32-g4-abc123"
    assert H.run_id(str(log), "clock_elasticity") == ""
    assert H.estimate(str(log)) == "581"          # the wall figure, not the kernel one
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
    assert rows[0] == H.reading_header() + ["eta", "eta_lo", "eta_hi", "band", "eta_exit"]
    assert [(r[0], r[1]) for r in rows[1:]] == [("1", "0"), ("1", "1"), ("16", "0")]
    assert all(r[-2:] == ["unmeasured", "unscored"] for r in rows[1:])
    # the same disk, rebuilt again: byte-identical, no second row per run
    assert H.pairs_table(session, results, "1 4 16", "0 1 2", settings) == 3
    assert (session / "PAIRS.tsv").read_text() == first
    # R1 for G=1 lands on a later pass: every G=1 row takes it
    _r1(results / "clock_elasticity", "r1rid", 0.15, 0.24, ("CLAIM", "C1", "PASS"))
    (logs / "r1-g1.log").write_text("experiment  clock_elasticity / r1rid\n")
    H.pairs_table(session, results, "1 4 16", "0 1 2", settings)
    rows = [ln.split("\t") for ln in (session / "PAIRS.tsv").read_text().splitlines()]
    assert [r[-2:] for r in rows[1:]] == [["RAW-STANDS", "DONE"]] * 2 + [["unmeasured", "unscored"]]
    fixed = {ln.split("\t")[0]: ln.split("\t")[1:]
             for ln in (session / "PAIRS-fixed.tsv").read_text().splitlines()[1:]}
    assert fixed["r3_duty"] == ["0.25", "report.json of 3 run(s)"]
    assert fixed["r3_pinned"][0] == "BLOCK_SIZE_N=64", "the swizzle is the G column, not fixed"
    assert fixed["r1_duty"] == ["1.0 0.7 0.5", "report.json of 1 run(s)"]
    assert fixed["r1_treads"] == ["--treads 8", "the chain's command line"]
    assert fixed["group_m"][0] == "varies: 1 4 16"
    assert not any(ln.startswith("#") for ln in (session / "PAIRS.tsv").read_text().splitlines())


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
# the measuring path, end to end, through planted interpreters
# --------------------------------------------------------------------------

STUB_BASE = r"""#!/bin/bash
# a planted base interpreter: the card and its UUID are the test's, and
# everything else runs on the real one
if [[ "${1:-}" == */alpha_g_chain_helpers.py ]]; then
  case "${2:-}" in
    card)   printf '%s\t\n' "${STUB_CARD:-nvidia_testcard}"; exit 0 ;;
    device) printf '%s\n' "${STUB_DEVICE:-0a0a0a0a-1111-2222-3333-444444444444}"; exit 0 ;;
  esac
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


g, tag = int(opt("--group-m")), opt("--session-tag")
plan = json.loads(Path(os.environ["STUB_PLAN"]).read_text()) if os.environ.get("STUB_PLAN") else {}
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
                               "envelope": [0.94, ratio + 0.01], "verdict": "PASS"}
                              if reps else None),
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
        self.base.write_text(STUB_BASE.replace("@PYTHON@", sys.executable))
        self.arm = root / "py-arm"
        self.arm.write_text(STUB_ARM.replace("@PYTHON@", sys.executable)
                            .replace("@ROOT@", str(ROOT)))
        for f in (self.base, self.arm):
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
                          STUB_PLAN=str(self.plan), STUB_TRACE=str(self.trace))
        full.update(env)
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
    """Pass 1: r1-g1 crashes; G=4's R1 page is INVALID (session 4's G=16 shape,
    above the admissible ceiling); G=16's seed 0 reads V7 FAIL. Pass 2,
    --resume: r1-g1 lands, and nothing else is bought."""
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
    assert steps == ["r3-g1-s0", "r3-g4-s0", "r3-g16-s0", "r1-g1", "r1-g4", "r1-g16",
                     "r3-g1-s1", "r3-g4-s1", "r3-g1-s2", "r3-g4-s2"], steps
    args = dict(after["trace"])
    tag = s.name
    for step in ("r3-g1-s0", "r3-g16-s0", "r3-g1-s1", "r3-g4-s2"):
        assert "--duty 0.25" in args[step] and f"--session-tag {tag}" in args[step], args[step]
    assert "--duty 1.0 0.7 0.5" in args["r1-g1"]
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
    assert by["r3-g16-s0"][-1][1] == "INVALID"
    assert by["r1-g1"][-1][1] == "ERROR"
    assert by["suite"][-1][1] == "SKIPPED" and "END_SUITE=skip" in by["suite"][-1][6]
    # SKIPPED is not latched: the next pass asks again and skips again
    assert re.search(r"^r3-g16-s1\s+SKIPPED", second.stdout, re.M), second.stdout


def test_the_table_is_rebuilt_every_pass_and_joins_the_r1_a_resume_landed(two_passes):
    pod, s, _first, after, second = two_passes
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-1500:]
    assert [st for st, _ in pod.traced()][len(after["trace"]):] == ["r1-g1"], "only the owed arm"
    header = H.reading_header() + ["eta", "eta_lo", "eta_hi", "band", "eta_exit"]
    keys = [("1", "0"), ("1", "1"), ("1", "2"), ("4", "0"), ("4", "1"), ("4", "2"), ("16", "0")]
    before = [dict(zip(header, r, strict=True)) for r in _rows_text(after["pairs"])]
    assert [(r["G"], r["seed"]) for r in before] == keys
    assert [r["band"] for r in before] == ["unmeasured"] * 3 + ["withheld:INVALID"] * 3 + [
        "RAW-STANDS"]
    now = [dict(zip(header, r, strict=True)) for r in _rows(s / "PAIRS.tsv")]
    assert [(r["G"], r["seed"]) for r in now] == keys, "one row per run, however many passes"
    assert [(r["band"], r["eta_exit"]) for r in now] == [("RAW-STANDS", "DONE")] * 3 + [
        ("withheld:INVALID", "INVALID")] * 3 + [("RAW-STANDS", "DONE")]
    assert [r["rep_n"] for r in now] == ["none", "2", "3"] * 2 + ["none"]
    assert [r["joint"] for r in now] == ["none", "PASS", "PASS"] * 2 + ["none"]
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
    assert [st for st, _ in pod.traced()] == ["r3-g1-s0"], "nothing after the pilot was bought"
    assert [(r[0], r[1]) for r in _rows(s / "PAIRS.tsv")] == [("1", "0")], "rebuilt before the STOP"
    again = pod.run("--resume", "--past-v8", SEEDS="0 1")
    assert again.returncode == 0, again.stdout[-3000:]
    over = [r for r in _rows(s / "CHAIN.tsv") if r[0] == "v8-override"]
    assert len(over) == 1 and over[0][1] == "OVERRIDDEN"
    assert "--past-v8" in over[0][6] and "r3-g1-s0's V8 FAIL" in over[0][6]
    assert [st for st, _ in pod.traced()] == ["r3-g1-s0", "r3-g16-s0", "r1-g1", "r1-g16",
                                              "r3-g1-s1", "r3-g16-s1"]


def test_a_pilot_with_no_report_stops_the_chain(tmp_path):
    logs = tmp_path / "chain-logs"
    logs.mkdir()
    (logs / "r3-g1-s0.log").write_text("Traceback: died before its plan page\n")
    ledger = _ledger(tmp_path / "CHAIN.tsv")
    got = lift("v8_gate 0; echo rc=$?", LEDGER=str(ledger), LOGS=str(logs),
               RESULTS=str(tmp_path / "res"), PILOT="r3-g1-s0")
    assert got.stdout.strip().endswith("rc=3"), got.stdout + got.stderr
    assert "STOP: r3-g1-s0's V8 is unread: no report.json, not PASS" in got.stdout


def test_a_callers_session_does_not_steer_a_chain_this_file_spawns(tmp_path, monkeypatch):
    """On the pod this file runs inside the chain's end suite. A SESSION= in
    the caller's environment turned every --resume here into a refusal, and
    every plain pass here into a pass in the caller's directory."""
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
    # both pytest steps are COLLECTED and priced at the pod's rate, never run by a dry run
    m = re.search(r"tests/test_gpu\.py ~(\d+) s \((\d+) tests\) before them and the end suite"
                  r" ~(\d+) s\s+\((\d+) tests\) after them, at ([\d.]+) s a test", out)
    assert m, out
    rate = float(m.group(5))
    assert int(m.group(1)) == round(int(m.group(2)) * rate) and int(m.group(2)) > 10
    assert int(m.group(3)) == round(int(m.group(4)) * rate) and int(m.group(4)) > 1000
    session = next((root / "session").glob("alpha_g-nocard-*"))
    ledger = (session / "CHAIN-dryrun.tsv").read_text().splitlines()
    names = [ln.split("\t")[0] for ln in ledger[1:]]
    assert names[:4] == ["preflight-r1", "preflight-r3", "preconditions", "gpu-tests"]
    assert names[4:8] == [f"r3-g{g}-s0" for g in (1, 4, 16, 64)]
    assert names[8:12] == ["r1-g1", "r1-g4", "r1-g16", "r1-g64"]
    assert names[12:20] == [f"r3-g{g}-s{s}" for s in (1, 2) for g in (1, 4, 16, 64)]
    assert names[20:] == ["suite"]
    rows = {ln.split("\t")[0]: ln.split("\t") for ln in ledger[1:]}
    assert rows["preflight-r1"][1] == "DONE" and "log agrees" in rows["preflight-r1"][6]
    assert rows["preconditions"][1] == "DONE" and "ARMS.tsv" in rows["preconditions"][6]
    for step in ("gpu-tests", "suite"):
        assert rows[step][1] == "DONE" and "tests collected" in rows[step][6], rows[step]
    assert rows["r3-g1-s0"][1] == "REFUSED", "a dry-run plan scores no gate: REFUSED"
    assert not (session / "CHAIN.tsv").exists() and not (session / "DEVICE").exists()


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
    assert "duty 1, 0.7, 0.5" in r1
    # both preflights ran the scorer and said so
    assert "SELF-TEST OK" in (logs / "preflight-r3.log").read_text()
    assert re.search(r"^preflight-r1\s+DONE", got.stdout, re.M), got.stdout


def test_the_dry_run_prints_the_pod_checkout_the_detached_launch_and_the_exfil(dry):
    out = dry[0].stdout
    assert ("git -C /workspace/moe-kernels checkout --"
            " moe/bench/hardware/measured_nvidia_h200.yaml") in out
    assert "checkout -B r3-align origin/r3-align" in out and "log -1 --format=%h" in out
    assert ("nohup setsid bash scripts/alpha_g_chain.sh > /workspace/alpha_g_chain.out 2>&1"
            " < /dev/null &") in out
    assert "tail -f /workspace/alpha_g_chain.out" in out
    tar = next(ln for ln in out.splitlines() if "exfil-alpha_g-nocard.tar.gz" in ln)
    assert '"moe/bench/hardware/measured_nocard.yaml"' in tar, "the ruler rides with the results"
    assert "results/calibration/" in tar, "and calibrate's own run directory"
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

