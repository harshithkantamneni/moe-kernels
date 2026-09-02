"""One exit-code table, one result-line format, for every experiment script.

An experiment script ends in exactly one of five states, and the number it
exits with is the ONLY thing the session driver (`scripts/h200_gaps_session.sh`)
can see without reading prose. So the table lives here, in one module, and every
script maps its outcome through `classify` instead of choosing its own integers.

THE TABLE
---------
    DONE        0   measured; every VALIDITY gate PASSED; every CLAIM gate PASSED.
    CLAIM_FAIL  1   measured; VALIDITY passed; at least one CLAIM gate did not.
                    This is a RESULT, not a retry: the claim is refuted (or not
                    established) and the report says which.
    REFUSED     2   nothing was measured; a precondition was not met (no CUDA,
                    no calibration for this card, a tile that cannot run as
                    pinned, an import that drifted). Free: no pod minutes spent.
    INVALID     3   measured; a VALIDITY gate FAILED after measuring. Nothing on
                    the page may be quoted. NOT a retry: repeating the run
                    repeats the failure unless the cause was transient, and if
                    it was transient the log has to say so in words.
    ERROR       4   crashed; an exception the script did not plan for.

Anything else (a signal, a shell `127`, a Python traceback that escaped) is
outside the table and the ledger records it as RETRY, because a code nobody
chose carries no information about what happened.

WHY INVALID IS NOT REFUSED
--------------------------
Both are "nothing quotable", and that is where the resemblance ends. REFUSED
cost nothing and is the script protecting the pod from an arm that could not
have meant anything; INVALID cost the whole arm and is the instrument reporting
that it broke while in use. The driver treats them oppositely: a REFUSED arm
has nothing to resume into and its first stderr line says what to fix; an
INVALID arm has a directory full of cells that must NOT be scored and a gate
name that says which check failed while the pod was still rented. Folding the
two into one code hides which one you have, and hiding it costs a second rental
to find out.

WHY CLAIM_FAIL IS DONE
----------------------
A CLAIM gate is a pre-registered expectation about the world. When it fails,
the experiment worked and the world disagreed, which is the most valuable
outcome an experiment has. The ledger must mark the arm finished (DONE-shaped),
never queue it for another attempt, and the summary must print the failing
gate as a finding. "Re-run until it passes" is the failure mode this code is
named against.

WHY UNKNOWN COUNTS AGAINST THE GATE
-----------------------------------
A gate that could not decide has not passed. On a VALIDITY gate, UNKNOWN means
the instrument's soundness could not be shown, so the numbers are exactly as
unquotable as after a FAIL: INVALID. On a CLAIM gate, UNKNOWN means the claim
was not established: CLAIM_FAIL. The rule is one-directional on purpose, DONE
needs every gate to say PASS in so many words, because "a check that examined
nothing reports zero failures" is this project's documented failure shape and
UNKNOWN-as-PASS is that shape exactly. An EMPTY gate list is refused for the
same reason: `classify([])` raises rather than returning DONE.

THE HISTORY: THE ANCHOR'S INVERTED CONTRACT
-------------------------------------------
Until 2026-09-02 the driver's header said "every refusal path exits 2" and read
2 as REFUSED, with anything not in a per-arm done list as RETRY. In the same
tree, `scripts/memory_branch_anchor.py` documented the OPPOSITE contract, 2 for
"a VALIDITY gate failed" (after an eight-minute measurement) and 3 for "nothing
measured", while `dtype_tile_confound`, `replicate_noise_floor` and
`span_extent_separation` refused with 3. The consequences were measured by the
audit, not argued: a measured-and-invalid anchor run was logged REFUSED, printed
"REFUSED BEFORE MEASURING. Nothing below is a gate", and, because the anchor's
stream check runs only on a freshly timed cell, every resume hit gate M0 FAIL
and the arm could never reach DONE; the three scripts' genuine refusals were
queued as retries; and the driver's closing summary, which grepped free text
(`floor|sigma`), matched a REFUSED noise-floor log 18 times and printed the
imported proxy `floor: 0.0905` and a pre-registered `C1 ... [PASS]` as measured
output. None of that needed a GPU to go wrong; it needed two integers to mean
two different things in two files.

THE ONE LINE THE DRIVER MAY GREP
--------------------------------
    RESULT: <KIND> <NAME> <VERDICT> <detail>

`result_line` renders it and `parse_result_lines` reads it back; the driver's
summary keys on the `RESULT: ` prefix at column zero and on nothing else. A
line that merely CONTAINS "PASS" or "floor" is prose and is not a result. Every
scored gate prints exactly one such line; nothing that is not a scored gate
prints one. `classify_text` closes the loop: the exit code a log implies can be
recomputed from its RESULT lines and compared with the code the process
returned, and a disagreement is itself a defect.
"""
from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass

DONE = 0
CLAIM_FAIL = 1
REFUSED = 2
INVALID = 3
ERROR = 4

#: Codes the table names, in the order a reader should learn them.
CODE_NAMES: dict[int, str] = {
    DONE: "DONE",
    CLAIM_FAIL: "CLAIM_FAIL",
    REFUSED: "REFUSED",
    INVALID: "INVALID",
    ERROR: "ERROR",
}

#: What each code means, one line each, for `--help` text and ledgers.
CODE_MEANINGS: dict[int, str] = {
    DONE: "measured; every VALIDITY and CLAIM gate PASSED",
    CLAIM_FAIL: "measured; VALIDITY passed; a CLAIM gate did not (a result, not a retry)",
    REFUSED: "nothing measured; a precondition was not met (free)",
    INVALID: "measured; a VALIDITY gate failed after measuring; nothing quotable",
    ERROR: "crashed; an unplanned exception",
}

VALIDITY = "VALIDITY"
CLAIM = "CLAIM"
KINDS = (VALIDITY, CLAIM)

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
VERDICTS = (PASS, FAIL, UNKNOWN)

#: Ledger states, the vocabulary `ledger_state` speaks and the driver writes.
LEDGER_STATES = ("DONE", "CLAIM_FAIL", "REFUSED", "INVALID", "RETRY")

#: The codes after which the arm's cells exist on disk. INVALID is measured but
#: must not be scored; the driver uses this to decide whether a directory is
#: worth keeping, not whether it is worth quoting.
MEASURED_CODES = frozenset({DONE, CLAIM_FAIL, INVALID})

#: The codes the ledger treats as finished. A REFUSED arm is finished too: it
#: has nothing to resume into and re-running it without changing the world
#: refuses again.
FINISHED_CODES = frozenset({DONE, CLAIM_FAIL, REFUSED, INVALID})


class MalformedGate(ValueError):
    """A gate tuple whose kind or verdict is not in the table.

    Raised rather than coerced: a verdict spelled "pass" or "Passed" would
    otherwise fall through every branch and be scored as whatever the fallthrough
    happened to be.
    """


class NoGatesScored(ValueError):
    """`classify` was handed an empty list.

    An empty list is what a scoring path produces when nothing was examined,
    and DONE would be the wrong answer for the reason the module docstring
    gives: a check that examined nothing reports zero failures.
    """


class MalformedResultLine(ValueError):
    """`result_line` was asked to render something the format cannot carry.

    A name with whitespace or a detail with a newline would produce a line that
    `parse_result_lines` cannot read back, and a line the driver cannot parse is
    a gate that silently disappears from the summary.
    """


def _normalise(gate) -> tuple[str, str]:
    """Reduce a `(kind, verdict)` or `(kind, name, verdict)` tuple to `(kind, verdict)`.

    Objects carrying `.kind` and `.verdict` attributes (the Gate dataclasses the
    scripts already define) are accepted on the same terms.
    """
    if hasattr(gate, "kind") and hasattr(gate, "verdict"):
        kind, verdict = gate.kind, gate.verdict
    else:
        try:
            parts = tuple(gate)
        except TypeError as exc:
            raise MalformedGate(f"gate is not a tuple or Gate-like object: {gate!r}") from exc
        if len(parts) == 2:
            kind, verdict = parts
        elif len(parts) == 3:
            kind, _name, verdict = parts
        else:
            raise MalformedGate(
                f"gate tuple must be (kind, verdict) or (kind, name, verdict), got {gate!r}")
    if kind not in KINDS:
        raise MalformedGate(f"gate kind {kind!r} is not one of {KINDS}")
    if verdict not in VERDICTS:
        raise MalformedGate(f"gate verdict {verdict!r} is not one of {VERDICTS}")
    return kind, verdict


def classify(gates: Iterable) -> int:
    """Map a list of scored gates to the one exit code the table allows.

    `gates` is an iterable of `(kind, verdict)` or `(kind, name, verdict)`
    tuples, or of objects with `.kind` and `.verdict`; kind in `KINDS`, verdict
    in `VERDICTS`. Any other spelling raises `MalformedGate`.

    The rule, in the order it is applied:

        any VALIDITY gate not PASS   -> INVALID
        else any CLAIM gate not PASS -> CLAIM_FAIL
        else                         -> DONE

    "not PASS" covers FAIL and UNKNOWN alike; see the module docstring for why.
    `classify` never returns REFUSED or ERROR: those are decided BEFORE any gate
    is scored (a refusal is the absence of a measurement) or by the exception
    handler around the whole script, not by the gate table. An empty list raises
    `NoGatesScored`.
    """
    scored = [_normalise(g) for g in gates]
    if not scored:
        raise NoGatesScored("classify() was given no gates; nothing was examined, "
                            "so there is no verdict to exit with")
    if any(kind == VALIDITY and verdict != PASS for kind, verdict in scored):
        return INVALID
    if any(kind == CLAIM and verdict != PASS for kind, verdict in scored):
        return CLAIM_FAIL
    return DONE


def ledger_state(rc: int) -> str:
    """The word the driver writes beside an exit code.

    DONE, CLAIM_FAIL, REFUSED and INVALID are the four finished states and
    round-trip exactly. Everything else, ERROR included, is RETRY: the process
    did not reach a verdict, so the arm may be attempted again and the log has
    to be read to find out why.
    """
    if isinstance(rc, bool) or not isinstance(rc, int):
        raise TypeError(f"exit code must be an int, got {rc!r}")
    if rc in (DONE, CLAIM_FAIL, REFUSED, INVALID):
        return CODE_NAMES[rc]
    return "RETRY"


def describe(rc: int) -> str:
    """`"3 INVALID: measured; a VALIDITY gate failed ..."`, or the RETRY form.

    Same type guard as `ledger_state`: a bool is not an exit code even though
    `True == 1` would otherwise describe it as CLAIM_FAIL.
    """
    if isinstance(rc, bool) or not isinstance(rc, int):
        raise TypeError(f"exit code must be an int, got {rc!r}")
    if rc in CODE_NAMES:
        return f"{rc} {CODE_NAMES[rc]}: {CODE_MEANINGS[rc]}"
    return f"{rc} RETRY: not in the exit-code table; read the log"


# --------------------------------------------------------------------------
# the one greppable line
# --------------------------------------------------------------------------

RESULT_PREFIX = "RESULT: "

#: Anchored at column zero. A name is one run of non-whitespace; a detail is
#: whatever follows one space after the verdict, to end of line.
_RESULT_RE = re.compile(
    r"^RESULT: (?P<kind>VALIDITY|CLAIM) (?P<name>\S+) (?P<verdict>PASS|FAIL|UNKNOWN)"
    r"(?: (?P<detail>.*))?$"
)


@dataclass(frozen=True)
class ResultLine:
    kind: str
    name: str
    verdict: str
    detail: str = ""

    def render(self) -> str:
        return result_line(self.kind, self.name, self.verdict, self.detail)


def result_line(kind: str, name: str, verdict: str, detail: str = "") -> str:
    """Render `RESULT: <KIND> <NAME> <VERDICT> <detail>`.

    With an empty detail the line ends after the verdict (no trailing space).
    Refuses a name containing whitespace or a detail containing a newline,
    because both would produce a line the parser cannot read back.
    """
    if kind not in KINDS:
        raise MalformedResultLine(f"kind {kind!r} is not one of {KINDS}")
    if verdict not in VERDICTS:
        raise MalformedResultLine(f"verdict {verdict!r} is not one of {VERDICTS}")
    if not name or any(c.isspace() for c in name):
        raise MalformedResultLine(f"name must be one non-empty token, got {name!r}")
    if "\n" in detail or "\r" in detail:
        raise MalformedResultLine("detail must be a single line")
    detail = detail.strip()
    line = f"{RESULT_PREFIX}{kind} {name} {verdict}"
    return f"{line} {detail}" if detail else line


def parse_result_lines(text: str) -> list[ResultLine]:
    """Read back every RESULT line in `text`, in order.

    Only lines that begin with `RESULT: ` at column zero and match the format
    exactly are returned. A line that merely mentions PASS, FAIL, "floor" or
    "sigma" is prose, and prose is what the pre-2026-09-02 summary grep was
    reading pre-registered expectations out of.
    """
    out = []
    for raw in text.splitlines():
        m = _RESULT_RE.match(raw.rstrip("\r"))
        if m is None:
            continue
        out.append(ResultLine(m.group("kind"), m.group("name"), m.group("verdict"),
                              (m.group("detail") or "").strip()))
    return out


def classify_text(text: str) -> int:
    """The exit code a log's RESULT lines imply. Raises `NoGatesScored` if none.

    Lets the driver recompute a script's verdict from what it printed and
    compare it with the code the process actually returned. They differ in two
    cases and the driver must tell them apart. One: the script printed one
    thing and exited another, the defect this module exists to prevent. Two:
    the script printed every RESULT line and THEN crashed, so the log implies
    DONE or CLAIM_FAIL and the process returned ERROR (4) or a signal; that is
    a legitimate disagreement, the process code wins, and the arm is RETRY
    with the traceback as the reason. A log with no RESULT lines at all raises
    `NoGatesScored`, which is what a REFUSED log looks like from here.
    """
    return classify([(r.kind, r.verdict) for r in parse_result_lines(text)])


# --------------------------------------------------------------------------
# self-test: every branch of the table, including the ones that must FAIL
# --------------------------------------------------------------------------

#: (gates, expected code). The anchor's inverted case is the first row.
SELF_TEST_TABLE: tuple[tuple[tuple[tuple[str, str], ...], int], ...] = (
    (((VALIDITY, PASS), (VALIDITY, FAIL), (CLAIM, PASS)), INVALID),
    (((VALIDITY, PASS), (CLAIM, PASS)), DONE),
    (((VALIDITY, PASS), (CLAIM, FAIL)), CLAIM_FAIL),
    (((VALIDITY, PASS), (CLAIM, UNKNOWN)), CLAIM_FAIL),
    (((VALIDITY, UNKNOWN), (CLAIM, PASS)), INVALID),
    (((VALIDITY, FAIL), (CLAIM, FAIL)), INVALID),
    (((CLAIM, PASS),), DONE),
)


def self_test() -> int:
    """Exercise the table off-GPU. Returns 0 when every row agrees, 1 otherwise.

    Four checks: every row of `SELF_TEST_TABLE` through `classify`; every code
    through `ledger_state`; one RESULT line rendered, buried in prose that
    mentions PASS and floor, and read back alone; and a positive assertion that
    a VALIDITY FAIL does not classify as DONE. Each check prints one `[PASS]` or
    `[FAIL]` line so a broken row is named, not just counted. The proof that
    this function can itself return 1 is not in here: it lives in
    `tests/test_exit_codes.py::test_self_test_fails_when_the_table_is_broken`,
    which plants a wrong expectation in the table and asserts the 1.
    """
    bad = 0
    for gates, want in SELF_TEST_TABLE:
        got = classify(gates)
        ok = got == want
        bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] classify({list(gates)}) -> "
              f"{got} {CODE_NAMES[got]} (want {want} {CODE_NAMES[want]})")
    for rc in CODE_NAMES:
        state = ledger_state(rc)
        want = CODE_NAMES[rc] if rc != ERROR else "RETRY"
        ok = state == want
        bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] ledger_state({rc}) -> {state}")
    line = result_line(CLAIM, "C1", FAIL, "floor 0.0905 wider than proxy 0.032")
    back = parse_result_lines("prose that mentions PASS and floor\n" + line + "\n")
    ok = len(back) == 1 and back[0].render() == line
    bad += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] result_line round-trip: {line!r}")
    planted = classify([(VALIDITY, FAIL)])
    ok = planted != DONE
    bad += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] planted VALIDITY FAIL is not DONE ({planted})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    for code in CODE_NAMES:
        print(describe(code))
