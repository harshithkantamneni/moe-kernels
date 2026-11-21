"""The pin probe is an ARM, so it has to speak the ledger's vocabulary.

`scripts/h200_gaps_session.sh` runs `python -m moe.bench.cli` twice under
MOE_FORCE_TILE, as `pin_probe-n64-g1` and `pin_probe-n256-g16`, BEFORE the five
arms it guards -- the control roofline, both BN arms, the anchor and the cap
test, every one of which the driver's own notes call "worthless if the pin is
not honoured". The driver never reads the probe's prose. It reads the integer
and the `RESULT: ` lines, and until 2026-09-02 the probe emitted neither
correctly:

  * a plan holding no span able to pin printed the word REFUSED and exited 3.
    3 is INVALID, so the driver logged "MEASURED, THEN A VALIDITY GATE FAILED
    ... NOT auto-retried" and LATCHED the row (`arm()` skips a prior DONE,
    CLAIM_FAIL or INVALID), which made a free, zero-minute refusal whose own
    message names the fix impossible to re-run without hand-editing the ledger;
  * a row that ran pinned and did NOT show the pin -- gate F1, the S6a defect
    the probe exists to detect -- exited 4, which is not in the table and is
    therefore RETRY, so the driver would spend the minutes again next session;
  * a mistyped `--models` exited 1, CLAIM_FAIL: a refuted pre-registered claim,
    from a process that measured nothing;
  * and the two gates it scores printed `[force-tile] GATE F1 ... PASS`, which
    is prose. Nothing in the log began with `RESULT: `, so the session summary
    reported two scored VALIDITY gates as "This arm was NOT scored".

Every test here is off-GPU, because every one of those defects is.

WHAT IS TESTED ELSEWHERE. That the pin reaches the kernel at all, that an
unpinnable span is recorded rather than run, and that the ledger counts what it
counts, are `tests/test_force_tile.py`'s. This file is about the contract at the
boundary: which integer, and which lines, the driver gets.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "h200_gaps_session.sh"

sys.path.insert(0, str(ROOT))
from moe.bench import cli  # noqa: E402
from moe.bench import exit_codes as EC  # noqa: E402
from moe.bench import force_tile as FT  # noqa: E402

#: The tile `pin_probe-n64-g1` pins: block_m_crossing_sweep.FIXED, which is what
#: the control roofline, both BN arms, the anchor and the cap test run under.
PIN = ('{"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":64,'
       '"GROUP_SIZE_M":1,"num_warps":8,"num_stages":4}')


def forced():
    return FT.parse(PIN)


def ledger(*, pinned=0, resumed=0, skipped=0, unobserved=0):
    """A `ForceTileLedger` in one of the four states the probe can end in.

    Built by calling the recorders rather than by setting fields, so a test
    cannot describe a ledger the sweep could not have produced.
    """
    out = FT.ForceTileLedger()
    for _ in range(pinned):
        out.record_pinned()
    for _ in range(resumed):
        out.record_resumed()
    for _ in range(skipped):
        out.record_skip("torch_grouped_mm_up", "no force_tile_config hook")
    for _ in range(unobserved):
        out.record_unobserved("vllm_fused_experts",
                              "source='vllm_default' (expected 'vllm_override')")
    return out


def verdict(cfg_ledger, capsys, *, pin=True):
    """`(exit code, log text)` for one call of `force_tile_verdict`."""
    rc = cli.force_tile_verdict(forced() if pin else None, cfg_ledger)
    return rc, capsys.readouterr().out


def run_cli(*argv, env_extra=None, cwd=None):
    """The module as the driver runs it: a subprocess, read by exit code."""
    env = dict(os.environ)
    env.pop(FT.ENV_VAR, None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "moe.bench.cli", *argv],
                          cwd=str(cwd or ROOT), capture_output=True, text=True,
                          timeout=900, env=env)


# --------------------------------------------------------------------------
# the two gates, and the one line the driver may grep
# --------------------------------------------------------------------------

def test_each_scored_gate_prints_exactly_one_result_line(capsys):
    rc, out = verdict(ledger(pinned=3), capsys)
    lines = EC.parse_result_lines(out)
    assert [(r.kind, r.name, r.verdict) for r in lines] == [
        (EC.VALIDITY, "F1", EC.PASS), (EC.VALIDITY, "F2", EC.PASS)]
    assert rc == EC.DONE
    # The detail carries the measurement and the threshold, so the summary can
    # print the gate without opening the CSV.
    assert "measured" in lines[0].detail and "gate" in lines[0].detail
    # And nothing that is not a scored gate prints one: the human lines beside
    # them start with `[force-tile]`.
    assert out.count("RESULT: ") == 2


def test_the_code_the_process_returns_is_the_code_its_log_implies(capsys):
    """`classify_text` over the log has to recompute the integer. That is the
    whole reason the RESULT lines exist, and it is checked in all three states
    the probe can score."""
    for kw in ({"pinned": 2}, {"skipped": 4}, {"pinned": 1, "unobserved": 2}):
        rc, out = verdict(ledger(**kw), capsys)
        assert EC.classify_text(out) == rc, (kw, out)


def test_an_unpinned_sweep_scores_no_gate_and_prints_no_result_line(capsys):
    """A sweep with no pin makes no claim about pinning. It must not print a
    gate it did not score -- `classify` refuses an empty gate list rather than
    calling nothing-examined a pass, and this is the branch that would hand it
    one."""
    rc, out = verdict(ledger(pinned=1), capsys, pin=False)
    assert rc == EC.DONE
    assert EC.parse_result_lines(out) == []
    with pytest.raises(EC.NoGatesScored):
        EC.classify_text(out)


# --------------------------------------------------------------------------
# the failing branches: which of the five codes each one is
# --------------------------------------------------------------------------

def test_a_row_that_did_not_show_the_pin_is_invalid_and_not_a_retry(capsys):
    """Gate F1, the S6a defect. It exited 4 until 2026-09-02, and 4 is not in
    the table, so `ledger_state` reads RETRY and the driver spends the minutes
    again next session on a run whose instrument is known to be broken."""
    rc, out = verdict(ledger(pinned=6, unobserved=2), capsys)
    assert rc == EC.INVALID
    assert EC.ledger_state(rc) == "INVALID"
    assert EC.ledger_state(EC.ERROR) == "RETRY"
    assert ("VALIDITY", "F1", "FAIL") in [
        (r.kind, r.name, r.verdict) for r in EC.parse_result_lines(out)]
    assert "must NOT be scored" in out


def test_a_pin_that_stood_on_no_cell_at_all_is_invalid(capsys):
    """Gate F2. NON-VACUITY: a sweep that pinned nothing writes rows nobody can
    quote as pinned, and every check downstream examines zero pinned rows and
    reports zero failures. It is INVALID rather than REFUSED because the sweep
    had already run when the gate was scored: the minutes are spent and the
    cells are on disk, unscoreable."""
    rc, out = verdict(ledger(skipped=12), capsys)
    assert rc == EC.INVALID
    assert ("VALIDITY", "F2", "FAIL") in [
        (r.kind, r.name, r.verdict) for r in EC.parse_result_lines(out)]
    assert "12 cells were skipped as unpinnable" in out


def test_cells_already_complete_under_this_pin_are_not_a_vacuous_run(capsys):
    """Resuming a probe that has nothing left to do is not the same state as a
    probe that measured nothing, and only the second is a failure. They are
    distinguishable at all because the manifest key carries the pin's
    fingerprint."""
    rc, _ = verdict(ledger(resumed=5, skipped=2), capsys)
    assert rc == EC.DONE


def test_a_measured_verdict_never_prints_the_word_refused(capsys):
    """The driver prints the first line matching REFUSED out of a log it has
    already decided is a refusal, so the word on a measured page is a caption
    under the wrong picture. Both INVALID branches used to open with
    `REFUSED (3):` and `REFUSED (4):`."""
    for kw in ({"skipped": 3}, {"pinned": 1, "unobserved": 1}):
        _, out = verdict(ledger(**kw), capsys)
        assert "REFUSED" not in out, kw


# --------------------------------------------------------------------------
# the invocation the driver actually issues
# --------------------------------------------------------------------------

def test_the_probes_own_command_refuses_free_rather_than_latching(tmp_path):
    """The finding's reproduction, verbatim: the pin probe off the GPU box.

    No framework span registers here, which is also what a pod whose vLLM
    import failed looks like, so the plan holds nothing able to honour the pin
    and the run refuses BEFORE `run_sweep`. Nothing was measured, so the code
    is 2 and the arm is re-runnable the moment the span is there.
    """
    done = run_cli("--profile", "profile-cell", "--groups", "baselines",
                   "--env", "vllm", "--impl", "vllm_fused_experts",
                   "--out-dir", str(tmp_path), env_extra={FT.ENV_VAR: PIN})
    out = done.stdout + done.stderr
    assert done.returncode == EC.REFUSED, out
    assert "REFUSED" in out and "--env vllm" in out
    assert EC.parse_result_lines(out) == [], "a refusal scores no gate"
    assert list(tmp_path.glob("*.csv")) == [], "a refusal writes nothing"


def test_a_refusal_is_not_one_of_the_states_the_driver_latches():
    """Structural, against the shipped driver rather than a copy of it: `arm()`
    skips an arm whose ledger already says DONE, CLAIM_FAIL or INVALID. That
    list is why the probe's code matters more than its message -- 3 made a
    zero-cost refusal permanent, and 2 does not."""
    line = next(ln for ln in DRIVER.read_text().splitlines()
                if "$1 == n &&" in ln)
    latched = {state for state in EC.LEDGER_STATES if f'"{state}"' in line}
    assert latched == {"DONE", "CLAIM_FAIL", "INVALID"}
    assert EC.ledger_state(EC.REFUSED) not in latched


def test_the_driver_sees_the_probes_file_as_having_adopted_the_table():
    """`h200_gaps_session.sh:adopts_exit_codes` greps the arm's own file for the
    module's dotted name -- `arm_script` maps every `pin_probe-*` to
    moe/bench/cli.py -- and prints a caveat beside every REFUSED or INVALID row
    from a file that does not carry it: "the state word may not mean what this
    session reads it to mean". Asked of the SHIPPED function, because that grep
    is the thing that has to stop firing for this arm."""
    body = (f"set -uo pipefail\nREPO={str(ROOT)!r}\n"
            f"eval \"$(sed -n '/^# >>> LIFTABLE/,/^# <<< LIFTABLE/p' \"{DRIVER}\")\"\n"
            "adopts_exit_codes \"$(arm_script pin_probe-n64-g1)\"; echo \"rc=$?\"\n")
    got = subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                         timeout=120)
    assert got.stdout.strip() == "rc=0", got.stdout + got.stderr


def test_a_mistyped_argument_refuses_rather_than_refuting_a_claim(tmp_path):
    """`--models nope` returned 1 = CLAIM_FAIL, which the ledger records as a
    finished result: a pre-registered expectation tested and found false, by a
    process that never imported a benchmark."""
    done = run_cli("--profile", "smoke", "--dry-run", "--models", "nope")
    assert done.returncode == EC.REFUSED, done.stdout + done.stderr
    assert "REFUSED" in done.stderr and "unknown model" in done.stderr
    assert EC.parse_result_lines(done.stdout + done.stderr) == []


def test_a_plan_that_would_pin_nothing_is_a_plan_refused_not_a_broken_plan():
    """The dry run is the review step, and the driver reads its codes through
    `dry_state`: 0 PLANNED, 2 PLAN_REFUSED, anything else BROKEN ("this is a
    PLAN, and it did not survive its own --dry-run"). A plan that validated and
    reported an unpinnable matrix is refused, not broken."""
    done = run_cli("--profile", "smoke", "--dry-run", "--groups",
                   "reference,baselines", "--impl", "torch_grouped_mm_up",
                   env_extra={FT.ENV_VAR: PIN})
    assert done.returncode == EC.REFUSED, done.stdout + done.stderr
    assert "REFUSED" in done.stdout
    assert "No GPU was used" in done.stdout


def test_an_unplanned_exception_is_error_rather_than_a_refuted_claim(tmp_path):
    """The code the header advertised and the file could not emit.

    A traceback that escapes `main` exits 1, and 1 is CLAIM_FAIL: the driver
    ledgers a dead process as a pre-registered claim tested and found false, and
    LATCHES the arm, so an OOM or a vLLM import that died in `bootstrap` costs a
    second rental to discover. `--out-dir` under a regular file reproduces it
    with no GPU and no vLLM: the crash lands in `Path.mkdir`, past argument
    handling, which is exactly where the pod's crashes land.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    done = run_cli("--profile", "smoke", "--out-dir", str(blocker / "sub"))
    assert done.returncode == EC.ERROR, done.stdout + done.stderr
    assert EC.ledger_state(done.returncode) == "RETRY"
    assert "Traceback" in done.stderr, "the traceback is still printed in full"
    assert "must not be quoted" in done.stderr
    assert EC.parse_result_lines(done.stdout + done.stderr) == [], \
        "a crash scores no gate"


def test_the_contract_paragraph_describes_the_dry_run_this_file_ships():
    """Prose against behaviour, because the prose was the thing that was wrong.

    The header listed "a dry run" among the states that exit REFUSED, and
    `dry_run` returns DONE for a plan that validates. Both cannot be true, and
    the behaviour is the one to keep: `scripts/run_all.sh` execs this module and
    reads 0 as "the plan validated, proceed", and the driver reads a plan arm
    through `dry_state` (0 PLANNED, 2 PLAN_REFUSED) rather than through the gate
    table. So the paragraph has to say that a dry run is a plan row, and this
    test fails if either half drifts from the other.
    """
    doc = cli.__doc__
    refused_clause = doc.split("REFUSED 2 (", 1)[1].split(")", 1)[0]
    assert "dry run" not in refused_clause, refused_clause
    assert "dry_state" in doc and "PLAN_REFUSED" in doc
    assert "ERROR 4 (" in doc, "the header still names the code main now emits"

    done = run_cli("--profile", "smoke", "--dry-run")
    assert done.returncode == EC.DONE, done.stdout + done.stderr
    assert EC.parse_result_lines(done.stdout) == [], "a plan scores no gate"
    with pytest.raises(EC.NoGatesScored):
        EC.classify_text(done.stdout)
