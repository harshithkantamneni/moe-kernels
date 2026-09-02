"""The exit-code table, and the four ways two integers already lied.

`moe/bench/exit_codes.py` is one table that every experiment script and the
session driver read the same way. The tests pin the four things the 2026-09-02
audit found broken:

1. THE ANCHOR'S INVERTED CONTRACT. A VALIDITY gate that fails AFTER measuring is
   INVALID (3), not REFUSED (2). The driver used to log it REFUSED, print
   "nothing below is a gate", and leave the arm un-DONE forever.
2. A CLAIM FAILURE IS A RESULT. `ledger_state(1)` is CLAIM_FAIL, a finished
   state, never RETRY.
3. UNKNOWN IS NOT PASS. An UNKNOWN on either kind of gate lowers the code; an
   empty gate list refuses to be DONE.
4. PROSE IS NOT A RESULT. The driver's summary used to grep `floor|sigma` and
   print a pre-registered expectation as measured output. Only a line that
   begins `RESULT: ` at column zero and matches the format is a result.

Nothing here needs a GPU.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from moe.bench import exit_codes as X

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 1 + 2. the table and its ledger words
# --------------------------------------------------------------------------

def test_table_values_are_the_documented_ones():
    assert (X.DONE, X.CLAIM_FAIL, X.REFUSED, X.INVALID, X.ERROR) == (0, 1, 2, 3, 4)
    assert len({X.DONE, X.CLAIM_FAIL, X.REFUSED, X.INVALID, X.ERROR}) == 5


@pytest.mark.parametrize("rc,state", [
    (X.DONE, "DONE"), (X.CLAIM_FAIL, "CLAIM_FAIL"), (X.REFUSED, "REFUSED"),
    (X.INVALID, "INVALID"),
])
def test_every_finished_code_round_trips_through_ledger_state(rc, state):
    assert X.ledger_state(rc) == state
    assert X.CODE_NAMES[rc] == state
    assert state in X.LEDGER_STATES
    assert rc in X.FINISHED_CODES


@pytest.mark.parametrize("rc", [X.ERROR, 5, 127, 130, 255, -1])
def test_error_and_unlisted_codes_are_retry(rc):
    assert X.ledger_state(rc) == "RETRY"
    assert rc not in X.FINISHED_CODES


def test_ledger_state_refuses_non_integers():
    with pytest.raises(TypeError):
        X.ledger_state("2")
    with pytest.raises(TypeError):
        X.ledger_state(True)
    with pytest.raises(TypeError):
        X.ledger_state(None)


def test_claim_fail_is_finished_and_measured_not_retry():
    assert X.ledger_state(X.CLAIM_FAIL) != "RETRY"
    assert X.CLAIM_FAIL in X.MEASURED_CODES
    assert X.CLAIM_FAIL in X.FINISHED_CODES


def test_invalid_is_measured_but_refused_is_not():
    assert X.INVALID in X.MEASURED_CODES
    assert X.REFUSED not in X.MEASURED_CODES
    assert X.ERROR not in X.MEASURED_CODES


def test_describe_names_every_code_and_the_retry_form():
    for rc, name in X.CODE_NAMES.items():
        text = X.describe(rc)
        assert text.startswith(f"{rc} {name}:")
        assert X.CODE_MEANINGS[rc] in text
    assert X.describe(99).startswith("99 RETRY")


# --------------------------------------------------------------------------
# 3. classify
# --------------------------------------------------------------------------

V, C = X.VALIDITY, X.CLAIM
P, F, U = X.PASS, X.FAIL, X.UNKNOWN

CLASSIFY_TABLE = [
    # the anchor's exact case: eight minutes measured, then a VALIDITY gate failed
    ([(V, "M5", P), (V, "M0", F), (C, "M1", P), (C, "M2", P)], X.INVALID),
    # every gate passed
    ([(V, "V0", P), (V, "V1", P), (C, "C1", P)], X.DONE),
    # only a claim failed: a result
    ([(V, "V0", P), (C, "C1", F), (C, "C2", P)], X.CLAIM_FAIL),
    # claim UNKNOWN is not PASS
    ([(V, "V0", P), (C, "C1", U)], X.CLAIM_FAIL),
    # validity UNKNOWN is not PASS either, and outranks a claim failure
    ([(V, "V0", U), (C, "C1", F)], X.INVALID),
    # validity FAIL outranks everything
    ([(V, "V0", F), (V, "V1", P), (C, "C1", F), (C, "C2", U)], X.INVALID),
    # a script with only claim gates can still be DONE
    ([(C, "C1", P)], X.DONE),
    # a script with only validity gates can still be DONE
    ([(V, "V1", P), (V, "V2", P)], X.DONE),
    # two-tuples are accepted alongside three-tuples
    ([(V, P), (C, F)], X.CLAIM_FAIL),
]


@pytest.mark.parametrize("gates,want", CLASSIFY_TABLE)
def test_classify_table(gates, want):
    assert X.classify(gates) == want


def test_anchor_case_is_invalid_not_refused():
    """The row that was wrong in production, stated on its own."""
    rc = X.classify([(V, "M0", F), (C, "M1", P)])
    assert rc == X.INVALID
    assert rc != X.REFUSED
    assert X.ledger_state(rc) == "INVALID"


def test_classify_accepts_gate_like_objects():
    class Gate:
        def __init__(self, kind, verdict):
            self.kind, self.verdict = kind, verdict

    assert X.classify([Gate(V, P), Gate(C, P)]) == X.DONE
    assert X.classify([Gate(V, F)]) == X.INVALID


def test_classify_never_returns_refused_or_error():
    seen = set()
    for kinds in ((V,), (C,), (V, C)):
        for verdicts in ((P,), (F,), (U,), (P, F), (P, U), (F, U)):
            gates = [(k, v) for k in kinds for v in verdicts]
            seen.add(X.classify(gates))
    assert X.REFUSED not in seen
    assert X.ERROR not in seen
    assert seen == {X.DONE, X.CLAIM_FAIL, X.INVALID}


def test_classify_refuses_an_empty_list():
    with pytest.raises(X.NoGatesScored):
        X.classify([])


@pytest.mark.parametrize("bad", [
    [("VALIDITY", "pass")],
    [("validity", "PASS")],
    [("CLAIM", "Passed")],
    [("SANITY", "PASS")],
    [("VALIDITY",)],
    [("VALIDITY", "V1", "PASS", "extra")],
    [42],
])
def test_classify_refuses_malformed_gates(bad):
    with pytest.raises(X.MalformedGate):
        X.classify(bad)


# --------------------------------------------------------------------------
# 4. the one greppable line
# --------------------------------------------------------------------------

def test_result_line_exact_format():
    assert X.result_line(V, "V1", P, "spread 0.55%") == "RESULT: VALIDITY V1 PASS spread 0.55%"
    assert X.result_line(C, "C1", F) == "RESULT: CLAIM C1 FAIL"
    assert X.result_line(C, "C1", U, "  padded  ") == "RESULT: CLAIM C1 UNKNOWN padded"


@pytest.mark.parametrize("kind,name,verdict,detail", [
    (V, "V1", P, "spread 0.55% of pin"),
    (C, "M1", F, ""),
    (C, "C2", U, "floor 0.0905 sigma 0.032 PASS FAIL"),
    (V, "gate_0_override", P, "17 fresh artefacts"),
])
def test_result_line_round_trips(kind, name, verdict, detail):
    line = X.result_line(kind, name, verdict, detail)
    back = X.parse_result_lines(line)
    assert back == [X.ResultLine(kind, name, verdict, detail.strip())]
    assert back[0].render() == line


def test_parse_reads_lines_in_order_from_a_log():
    log = "\n".join([
        "importing vllm 0.27.1",
        X.result_line(V, "V0", P, "pin reached the kernel"),
        "cell 1/24 bm=32 g=1 n=1  1.95 ms",
        X.result_line(C, "C1", F, "t(1) spread 6.2% > 4%"),
        X.result_line(C, "C2", P),
        "exit 1",
    ])
    got = X.parse_result_lines(log)
    assert [(r.kind, r.name, r.verdict) for r in got] == [
        (V, "V0", P), (C, "C1", F), (C, "C2", P)]
    assert X.classify_text(log) == X.CLAIM_FAIL


@pytest.mark.parametrize("prose", [
    "floor: 0.0905",
    "C1 the floor is no wider than the proxy implied [PASS]",
    "sigma = 0.032 (imported proxy)",
    "GATE V1  VALIDITY PASS",
    "[PASS] VALIDITY V0 pin reached the kernel",
    "  RESULT: VALIDITY V1 PASS indented is not at column zero",
    "note: RESULT: VALIDITY V1 PASS mid-line is prose",
    "RESULT: VALIDITY V1 pass lower-case verdict",
    "RESULT: SANITY V1 PASS unknown kind",
    "RESULT: VALIDITY  V1 PASS double space",
    "RESULT:VALIDITY V1 PASS no space after colon",
    "RESULT: VALIDITY PASS",
    "RESULT: VALIDITY V1",
])
def test_prose_that_merely_contains_the_words_is_not_parsed(prose):
    assert X.parse_result_lines(prose) == []
    with pytest.raises(X.NoGatesScored):
        X.classify_text(prose)


def test_refused_log_yields_no_gates():
    """A REFUSED arm prints no RESULT line, so the summary has nothing to print
    for it. This is the noise-floor incident: 18 prose matches, zero results."""
    log = ("REFUSED: no CUDA device\n"
           "floor 0.0905 (imported proxy)\n"
           "C1 the floor is no wider than the proxy implied [PASS]\n"
           "sigma sigma sigma\n")
    assert X.parse_result_lines(log) == []


@pytest.mark.parametrize("kind,name,verdict,detail", [
    ("SANITY", "V1", P, ""),
    (V, "V1", "Passed", ""),
    (V, "V 1", P, ""),
    (V, "", P, ""),
    (V, "V1", P, "two\nlines"),
    (V, "V1", P, "carriage\rreturn"),
])
def test_result_line_refuses_what_it_cannot_read_back(kind, name, verdict, detail):
    with pytest.raises(X.MalformedResultLine):
        X.result_line(kind, name, verdict, detail)


def test_classify_text_agrees_with_classify_on_the_same_gates():
    for gates, want in CLASSIFY_TABLE:
        lines = []
        for i, g in enumerate(gates):
            kind, verdict = g[0], g[-1]
            name = g[1] if len(g) == 3 else f"G{i}"
            lines.append(X.result_line(kind, name, verdict, "detail"))
        assert X.classify_text("\n".join(lines)) == want


# --------------------------------------------------------------------------
# the self-test plants its own FAIL branch and exits 0 only when it holds
# --------------------------------------------------------------------------

def test_self_test_passes_and_prints_every_row(capsys):
    assert X.self_test() == 0
    out = capsys.readouterr().out
    assert "[FAIL]" not in out
    assert out.count("[PASS]") >= len(X.SELF_TEST_TABLE) + len(X.CODE_NAMES) + 2
    assert "planted VALIDITY FAIL is not DONE" in out


def test_self_test_fails_when_the_table_is_broken(monkeypatch, capsys):
    """The FAIL branch, planted: make classify say DONE for everything."""
    monkeypatch.setattr(X, "classify", lambda gates: X.DONE)
    assert X.self_test() == 1
    assert "[FAIL]" in capsys.readouterr().out


def test_module_main_self_test_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "moe.bench.exit_codes", "--self-test"],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[FAIL]" not in proc.stdout


def test_module_main_without_flag_prints_the_table():
    proc = subprocess.run([sys.executable, "-m", "moe.bench.exit_codes"],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    for rc, name in X.CODE_NAMES.items():
        assert f"{rc} {name}:" in proc.stdout
