"""The card-acceptance arm: what it refuses, and what it would have caught.

THE EVENT THIS FILE IS ABOUT. On 2026-09-11 a rented H200 boosted to its
1980 MHz maximum, collapsed to its 345 MHz floor within ~30 s of sustained
bf16 GEMM and stayed there, drawing ~240 W of a 700 W limit while climbing
from 87 C to 93 C. `scripts/calibrate_hardware.py` ran to completion on it,
published a tracked yaml whose ridge read 73.6 against a real ~156, and its
`not_throttled` gate PASSED, because that gate scored DRIFT and a card pinned
flat at its floor does not drift.

So every gate here is planted in BOTH directions against a Trace that needs no
GPU, and three of the planted worlds are refusals: a scorer that has only ever
been shown clean cards has never been shown to refuse one. The two that matter
most are `floored`, where the term the old rule scored still passes, and
`hungry-tile`, the lowest per-cell median in the whole published corpus, which
must NOT be refused -- a gate that refuses healthy cards costs a rental just as
surely as one that admits sick ones.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import thermal_acceptance as TA  # noqa: E402
from _hermetic import laptop_env  # noqa: E402

from moe.bench import exit_codes as EX  # noqa: E402
from moe.bench import timing as T  # noqa: E402

SCRIPT = REPO / "scripts" / "thermal_acceptance.py"

#: The maximum SM clock the planted worlds are scored against. A PROPERTY OF
#: THE PART, read from the card by `timing.max_sm_clock_mhz` on a real run and
#: never derived from a calibration, which is why it may stand as a literal
#: here where a ridge or a ceiling may not. Every assertion below about the
#: FLOOR is a relation to it and never the number 660.
H200_MAX_SM = 1980.0


def run_script(*args, cwd=None):
    # Laptop path on every box: the bare pod line would measure 160 s on a
    # card. NVML is NOT hidden by this, which is why the no-maximum test
    # below is a `no_gpu` test and not a hermetic one.
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=300,
                          cwd=str(cwd or REPO), env=laptop_env())


# --------------------------------------------------------------------------
# the gates, both branches of each
# --------------------------------------------------------------------------

def test_the_planted_worlds_name_every_gate_and_both_of_its_branches():
    """A gate added later cannot slip in untested: the set of tokens is
    asserted, and every one of them must be seen PASSING in some world and
    not-passing in another. `calibrate_hardware.score` shipped a docstring
    saying five gates while emitting six, and the sixth's FAIL branch had
    never run anywhere."""
    seen: dict[str, set[str]] = {}
    for name, trace, expect, _why in TA.self_test_worlds():
        got = {g.token: g.verdict for g in TA.gates_for(trace)}
        assert got == expect, (name, got, expect)
        for token, verdict in got.items():
            seen.setdefault(token, set()).add(verdict)
    assert set(seen) == {"V1", "V2", "C1", "C2"}
    for token, verdicts in seen.items():
        assert TA.PASS in verdicts, token
        assert verdicts - {TA.PASS}, f"{token} passes in every planted world"


def test_the_floored_card_fails_the_claim_the_old_rule_passed():
    """THE MOTIVATING EVENT, and the hole reproduced beside it.

    C2 is the repository's own DRIFT rule over two medians, and on a card flat
    at its floor it PASSES: first third 345, last third 345. That is what
    `calibrate_hardware`'s `not_throttled` scored, and it is why a pod that
    could not clock published a tracked ruler. C1 is what refuses it."""
    floored = TA.plant([345] * 60, power_w=240.0)
    gates = {g.token: g for g in TA.gates_for(floored)}
    assert gates["C2"].verdict == TA.PASS, "the term that used to decide"
    assert gates["C1"].verdict == TA.FAIL
    assert "345 MHz median" in gates["C1"].measured
    assert f"{T.thermal_floor_mhz(H200_MAX_SM):.0f} MHz" in gates["C1"].threshold
    rc = EX.classify(g.scored() for g in TA.gates_for(floored))
    # A RESULT about this pod, latched by the driver, never a retry.
    assert rc == EX.CLAIM_FAIL


def test_the_lowest_healthy_card_in_the_corpus_is_not_refused():
    """THE OTHER DIRECTION, and it is the expensive one to get wrong. 1275 MHz
    on a 1980 MHz part at 697.4 W of a 700 W cap is the lowest per-cell
    `sm_clock_load_mhz` median anywhere in `results/published`: a hungry tile
    pinned at the power cap, not a sick card. A threshold that refuses it
    refuses every session this study can run."""
    hungry = TA.plant([1275] * 60, power_w=697.4)
    assert all(g.verdict == TA.PASS for g in TA.gates_for(hungry))
    assert EX.classify(g.scored() for g in TA.gates_for(hungry)) == EX.DONE
    # And the power reading is printed as the discriminator it is.
    c1 = {g.token: g for g in TA.gates_for(hungry)}["C1"]
    assert any("board power" in ln for ln in c1.lines)
    assert any("99.6% of the 700 W limit" in ln for ln in c1.lines), c1.lines


def test_a_card_still_falling_when_the_window_closed_fails_the_steady_claim():
    """A median above the floor is not enough: the calibration that follows
    takes minutes and this window takes two."""
    falling = TA.plant([1470] * 20 + [1100] * 20 + [420] * 20)
    gates = {g.token: g for g in TA.gates_for(falling)}
    assert gates["C1"].verdict == TA.PASS, "the median is still above the floor"
    assert gates["C2"].verdict == TA.FAIL
    assert "first third" in gates["C2"].measured


def test_the_verdict_is_taken_on_the_median_and_not_on_a_sample():
    """Healthy cells in `results/published` post individual readings at
    405 MHz on a 1980 MHz part during one drain-and-ramp. A per-sample gate
    would need a fraction below 0.2045 against a fault at 0.1742 and would
    have no margin left; the median is what `clock_floor_ok` reads."""
    excursion = TA.plant([1425, 1410, 405, 825, 1365, 1410, 1425] * 9)
    floor = T.thermal_floor_mhz(H200_MAX_SM)
    assert min(s.mhz for s in excursion.scored) < floor
    assert excursion.median_mhz > floor
    assert {g.token: g.verdict for g in TA.gates_for(excursion)}["C1"] == TA.PASS


def test_a_probe_that_watched_nothing_does_not_report_a_clean_card():
    """NON-VACUITY, first gate. A window of four samples reports no throttling
    exactly as a healthy card does, and V1 FAILing is INVALID -- measured and
    unquotable -- rather than a claim about the card."""
    truncated = TA.plant([1470] * 4)
    gates = {g.token: g for g in TA.gates_for(truncated)}
    assert gates["V1"].verdict == TA.FAIL
    assert str(TA.SCORED_SAMPLE_FLOOR) in gates["V1"].threshold
    assert EX.classify(g.scored() for g in TA.gates_for(truncated)) == EX.INVALID
    # And the floor comes from the instrument rather than from this file.
    assert TA.SCORED_SAMPLE_FLOOR == 3 * T.CLOCK_SAMPLE_FLOOR


def test_a_forked_sampler_is_invalid_however_good_the_clocks_look():
    """`ClockState.sample` falls back to a forked nvidia-smi, which answers
    tens of milliseconds late and therefore describes an idle card at its
    boost clock. `calibrate.clock_under_load` refuses exactly that fallback
    for the roof's reference; a thermal verdict has the same duty."""
    forked = TA.plant([1470] * 60, source=T.CLOCK_SOURCE_NVIDIA_SMI)
    gates = {g.token: g for g in TA.gates_for(forked)}
    assert gates["V2"].verdict == TA.FAIL
    assert gates["C1"].verdict == TA.PASS, "the clocks themselves look fine"
    assert EX.classify(g.scored() for g in TA.gates_for(forked)) == EX.INVALID


def test_no_maximum_clock_leaves_the_claim_untested_and_never_passed():
    """UNKNOWN counts against a CLAIM gate. Mapping "there was no floor to
    compare against" to PASS is the shape of the hole this arm was added to
    close."""
    blind = TA.plant([1470] * 60, maximum=None)
    gates = {g.token: g for g in TA.gates_for(blind)}
    assert gates["C1"].verdict == TA.UNKNOWN
    assert "no maximum SM clock could be read" in gates["C1"].threshold
    assert EX.classify(g.scored() for g in TA.gates_for(blind)) == EX.CLAIM_FAIL


# --------------------------------------------------------------------------
# the threshold is a relation, never a literal (repo rule 2)
# --------------------------------------------------------------------------

def test_the_floor_moves_with_the_card_and_is_nowhere_a_literal():
    """R2. The same 345 MHz trace is refused on a 1980 MHz part and admitted
    on a part whose maximum is low enough, because the floor is a fraction of
    whatever the device reported. A gate carrying 660 as a number would be a
    gate that works on one part."""
    fault, fault_max = T.THERMAL_FAULT_OBSERVED_MHZ
    floored = TA.plant([int(fault)] * 60, maximum=fault_max)
    assert TA.gates_for(floored)[2].verdict == TA.FAIL
    # The low part is DERIVED, not the literal 900 it was until 2026-09-11: a
    # maximum whose floor is 0.9 x the fault clock, so this stays a PASS at any
    # fraction inside THERMAL_FLOOR_FRACTION's own admissible window rather
    # than only at one third.
    low_part = TA.plant([int(fault)] * 60,
                        maximum=0.9 * fault / T.THERMAL_FLOOR_FRACTION)
    assert TA.gates_for(low_part)[2].verdict == TA.PASS
    assert T.thermal_floor_mhz(1980.0) != T.thermal_floor_mhz(1410.0)
    # Every edge is a clock the grid can report.
    for maximum in (900.0, 1410.0, 1755.0, 1980.0):
        assert T.thermal_floor_mhz(maximum) % T.CLOCK_STEP_MHZ == 0


def test_the_source_carries_no_clock_literal_to_compare_against():
    """Asked of the file. The derivation quotes 1275, 1980 and 345 in PROSE,
    which is how this repository keeps a threshold auditable; what must not
    exist is executable code comparing a reading against one of them. Parsed
    rather than grepped, so a number inside a docstring or a comment is not
    mistaken for one inside an expression."""
    import ast

    tree = ast.parse(SCRIPT.read_text())
    # `self_test_worlds` is the one place clocks are written down, and they are
    # PLANTED DATA rather than thresholds: the worlds have to be some card's
    # readings or they test nothing. Every other function is checked.
    planted = {"self_test_worlds"}
    scoring = [node for node in tree.body
               if not (isinstance(node, ast.FunctionDef) and node.name in planted)]
    numbers = [n.value for node in scoring for n in ast.walk(node)
               if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool)]
    for forbidden in (1275, 1980, 345, 660, 1410, 465):
        assert forbidden not in numbers, (
            f"{forbidden} is a clock this file must never carry as a value: "
            "the floor is THERMAL_FLOOR_FRACTION of what the device reported")
    # And `plant` takes its clocks as arguments, so no scoring path can reach
    # one: the default maximum is the only clock in its signature and it is
    # only ever used by the planted worlds above.
    assert "THERMAL_FLOOR_FRACTION" in SCRIPT.read_text()
    assert all(g.threshold.count("MHz") for g in TA.gates_for(TA.plant([1470] * 60))
               if g.token == "C1")


# --------------------------------------------------------------------------
# the modes, end to end
# --------------------------------------------------------------------------

def test_the_self_test_scores_one_validity_gate_and_the_log_agrees_with_the_code():
    """A self-test that fails has refuted nothing about any card; it has said
    nothing on its page may be quoted, which is INVALID and not CLAIM_FAIL.
    And `classify_text` over the log must recompute the code the process
    returned, because the driver reads the page as a second opinion and a
    disagreement between the two costs the arm its latch."""
    got = run_script("--self-test")
    assert got.returncode == EX.DONE, got.stdout[-3000:]
    lines = EX.parse_result_lines(got.stdout)
    assert [(ln.kind, ln.name, ln.verdict) for ln in lines] == [
        (EX.VALIDITY, "V0", EX.PASS)]
    assert EX.classify_text(got.stdout) == got.returncode
    # Every planted world is NAMED on the page, so a broken row is identified
    # rather than merely counted.
    for name, _trace, _expect, _why in TA.self_test_worlds():
        assert name in got.stdout, name


def test_the_self_test_can_fail(monkeypatch, capsys):
    """Planted the other way. Without this the self-test could be a loop that
    compares a thing to itself and passes forever."""
    real = TA.self_test_worlds()

    def broken():
        name, trace, expect, why = real[0]
        return [(name, trace, {**expect, "C1": TA.FAIL}, why)]

    monkeypatch.setattr(TA, "self_test_worlds", broken)
    assert TA.self_test() == EX.INVALID
    out = capsys.readouterr().out
    assert "[FAIL] C1 expected FAIL     got PASS" in out


def test_a_dry_run_prints_a_plan_scores_nothing_and_refuses(tmp_path):
    """REFUSED (2) and not the bare literal 0. A plan measured nothing and
    scored no gate, and DONE in the shared table reads "measured; every
    VALIDITY and CLAIM gate PASSED". `classify_text` over this log raises
    rather than calling an empty gate list DONE, which is what a REFUSED log
    looks like from the table's side."""
    got = run_script("--dry-run")
    assert got.returncode == EX.REFUSED, got.stdout[-2000:]
    assert EX.parse_result_lines(got.stdout) == []
    with pytest.raises(EX.NoGatesScored):
        EX.classify_text(got.stdout)
    # The plan body comes BEFORE the refusal banner, which is what the
    # driver's `printed_a_plan` awk needs in order to file this PLANNED
    # rather than PLAN_REFUSED: it stops counting at the first REFUSED.
    body, _, banner = got.stdout.partition("REFUSED. Nothing was measured")
    assert "estimated wall time" in body and banner
    assert "REGISTERED PREDICTIONS" in body
    for needle in ("P1", "P2", "RESOLUTION"):
        assert needle in body, needle


def test_a_dry_run_writes_nothing_at_all(tmp_path):
    """The requirement, tested the way it failed elsewhere: run it and ask
    git. A laptop rehearsal of a session must not leave the tree dirty for the
    rows that follow it."""
    before = subprocess.run(["git", "status", "--porcelain",
                             "--untracked-files=all"], cwd=str(REPO),
                            capture_output=True, text=True).stdout
    run_script("--dry-run")
    after = subprocess.run(["git", "status", "--porcelain",
                            "--untracked-files=all"], cwd=str(REPO),
                           capture_output=True, text=True).stdout
    assert before == after
    assert not (REPO / "results" / "thermal_acceptance").exists()


def test_the_measuring_path_refuses_without_a_card_and_says_why():
    """Both reports are written with `write_text`, which truncates, and the
    floor itself is per-card. A measuring run under the no-card sentinel would
    file one card's verdict under another's name."""
    got = run_script()
    assert got.returncode == EX.REFUSED, got.stdout[-2000:]
    assert "no card was named" in got.stdout
    assert EX.parse_result_lines(got.stdout) == []
    assert TA.NO_CARD in got.stdout


@pytest.mark.no_gpu
def test_the_measuring_path_refuses_when_no_maximum_can_be_read():
    """REFUSED and not measured-then-UNKNOWN. Without a maximum there is no
    floor, so two minutes of load would be bought to answer nothing, and a
    refusal is decided BEFORE anything is spent. `no_gpu`, not hermetic: the
    maximum is read through pynvml / nvidia-smi, which CUDA_VISIBLE_DEVICES
    does not hide, so on a pod this door opens and the run refuses one door
    later with a different sentence."""
    got = run_script("--card", "NVIDIA H200")
    assert got.returncode == EX.REFUSED, got.stdout[-2000:]
    assert "maximum SM clock could not be read" in got.stdout
    assert EX.parse_result_lines(got.stdout) == []


# --------------------------------------------------------------------------
# where it writes, and what it stamps
# --------------------------------------------------------------------------

def test_the_run_id_carries_the_card_and_every_knob_that_changes_the_answer():
    """A longer window sees a slower collapse and a shorter one may miss it,
    and the poll interval sets how many samples the median is taken over. A
    run id that omits a swept knob is how two settings come to share a
    directory, and `write_text` truncates."""
    import argparse

    base = dict(card="NVIDIA H200", settle_seconds=30.0, seconds=120.0,
                poll_seconds=2.0, run_id="")
    first = TA.default_run_id(argparse.Namespace(**base))
    assert first.startswith("nvidia_h200")
    for knob, other in (("seconds", 60.0), ("poll_seconds", 1.0),
                        ("settle_seconds", 10.0)):
        moved = TA.default_run_id(argparse.Namespace(**{**base, knob: other}))
        assert moved != first, knob
    other_card = TA.default_run_id(
        argparse.Namespace(**{**base, "card": "NVIDIA A100-SXM4-80GB"}))
    assert other_card != first


def test_the_instrument_names_what_actually_ran_and_not_the_ladders():
    """`dram_counter_route` keeps four separate instrument strings precisely so
    an arithmetic mode cannot be stamped with a timing basis it never used.
    Nothing here is timed at all: the quantity is a clock and the reader is
    NVML."""
    assert "NOT timing.TIMING_BASIS" in TA.INSTRUMENT
    assert TA.INSTRUMENT != T.TIMING_BASIS
    assert "_load_compute" in TA.INSTRUMENT


def test_the_report_json_would_carry_the_whole_trace(tmp_path):
    """Every sample, not a summary. A bimodal window -- half at the plateau,
    half at the floor -- is visible in the list and averaged away in the
    median, and the calibration that follows is what the reader is deciding
    about."""
    trace = TA.plant([1470] * 30 + [345] * 30)
    payload = json.loads(json.dumps({
        "samples": [s.__dict__ for s in trace.samples],
        "median_mhz": trace.median_mhz, "floor_mhz": trace.floor_mhz}))
    assert len(payload["samples"]) == len(trace.samples)
    assert {s["mhz"] for s in payload["samples"]} >= {1470, 345}
    assert payload["floor_mhz"] == T.thermal_floor_mhz(H200_MAX_SM)


def test_an_unplanned_crash_is_error_and_not_a_failed_claim(monkeypatch, capsys):
    """Left to propagate, an exception exits the interpreter ONE, and ONE is
    CLAIM_FAIL, which is in FINISHED_CODES: the driver would file a crashed
    probe as "this card cannot hold its clock", refuse the session, and never
    retry the arm. ERROR (4) is outside FINISHED_CODES for exactly that."""
    def boom(argv=None):
        raise RuntimeError("planted")

    monkeypatch.setattr(TA, "_main", boom)
    assert TA.main([]) == EX.ERROR
    assert EX.ERROR not in EX.FINISHED_CODES
    assert "planted" in capsys.readouterr().err


# --------------------------------------------------------------------------
# the quantity the refusal gated on is the quantity the gate scores
# --------------------------------------------------------------------------

def test_the_maximum_is_read_once_and_handed_to_the_probe():
    """RULE 3, asked of the source, and it is the expensive direction.

    `_main` reads `max_sm_clock_mhz` and REFUSES for free when it is absent.
    `sustain` used to read it AGAIN and put THAT value on the Trace, so the
    quantity the free refusal was decided on was not the quantity C1 scored. A
    second read failing where the first succeeded spent the whole window and
    then exited CLAIM_FAIL -- latched, and printed by the driver as "RETURN
    THE POD AND RENT ANOTHER CARD" -- over a healthy card, because a C1 with
    no floor is UNKNOWN and UNKNOWN classifies as a failed claim.
    """
    import ast

    tree = ast.parse(SCRIPT.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "sustain")
    assert "max_sm" in [a.arg for a in fn.args.args], ast.unparse(fn.args)
    # A CALL to the reader, not the keyword it is stored under: `sustain` still
    # names `max_sm_clock_mhz=` when it builds the Trace, and must.
    inside = [ast.unparse(n) for n in ast.walk(fn)
              if isinstance(n, ast.Call)
              and getattr(n.func, "attr", "") == "max_sm_clock_mhz"]
    assert inside == [], inside
    # TWO calls in the whole file and both in `_main`, one per mode: the plan
    # prints the floor an operator should see before renting, and the measuring
    # path gates on it and then hands it down. Neither is on the other's path,
    # so the measuring run still reads the quantity exactly once.
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_main")
    everywhere = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and getattr(n.func, "attr", "") == "max_sm_clock_mhz"]
    in_main = [n for n in ast.walk(main) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", "") == "max_sm_clock_mhz"]
    assert len(everywhere) == 2 and len(in_main) == 2, (
        [ast.unparse(n) for n in everywhere])
    # and the one call site hands it both halves, so the page can say which
    # reader answered without asking a second time.
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "sustain")
    assert len(call.args) + len(call.keywords) == 5, ast.unparse(call)


def test_a_missing_nvml_sampler_refuses_before_the_load_rather_than_crashing():
    """The maximum has a forked `nvidia-smi` fallback and the under-load
    sampler deliberately has none, so a pod with nvidia-smi and no
    nvidia-ml-py passes the maximum's pre-flight and then fails one line into
    `sustain`. Unhandled, that reached `main`'s wrapper as ERROR (4) with a
    traceback and a RETRY row naming no cause -- for the same operator error
    the REFUSED branch above it already knows how to explain."""
    import ast

    tree = ast.parse(SCRIPT.read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_main")
    handler = next(
        (h for n in ast.walk(main) if isinstance(n, ast.Try)
         for h in n.handlers
         if h.type is not None
         and "ClockSourceUnavailable" in ast.unparse(h.type)),
        None)
    assert handler is not None, "the sampler is never probed before the load"
    # It REFUSES. Not ERROR, which is retryable and names no cause, and not
    # CLAIM_FAIL, which would latch an apparatus fault as a verdict on a card.
    returned = [ast.unparse(n.value) for n in ast.walk(handler)
                if isinstance(n, ast.Return) and n.value is not None]
    assert returned == ["exit_codes.REFUSED"], returned
    # BEFORE the load: a refusal only costs nothing if it is decided before
    # the window is spent.
    probe = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "nvml_clock_reader")
    load = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "sustain")
    assert probe.lineno < load.lineno, (probe.lineno, load.lineno)


# --------------------------------------------------------------------------
# what a page SAYS is what the page found
# --------------------------------------------------------------------------

def test_a_healthy_page_does_not_say_the_card_cannot_hold_its_clock():
    """The sentence sat outside `if failed`, so every accepted card's
    report.txt -- the artefact an operator reads and the driver greps --
    ended by announcing the failure that had not happened."""
    healthy = TA.render(["h"], TA.gates_for(TA.plant([1470] * 60)), [])
    assert "Claim gates not passed: none" in healthy
    assert "cannot hold its clock" not in healthy, healthy[-400:]
    floored = TA.render(
        ["h"],
        TA.gates_for(TA.plant([int(T.THERMAL_FAULT_OBSERVED_MHZ[0])] * 60)),
        [])
    assert "cannot hold its clock" in floored


def test_every_statement_of_the_planted_world_count_agrees_with_the_list():
    """Three sites say how many worlds there are and one said six over a list
    of eight. Nothing asserted the count, which is why the suite stayed green
    with it wrong."""
    import re

    n = len(TA.self_test_worlds())
    words = {6: "six", 8: "eight"}
    source = SCRIPT.read_text()
    assert n in words, n
    assert words[n] in source
    for wrong in set(words.values()) - {words[n]}:
        assert not re.search(rf"\b{wrong}\b (?:planted )?worlds", source), wrong


def test_the_margin_printed_beside_a_floor_is_computed_not_typed():
    """`resolution_line` derives the floor from whatever card is attached and
    then printed a hardcoded 1.93x beside it: on an A100 that is a 465 MHz
    floor with the H200's ratio next to it, and the published corpus holds no
    A100 row at all. The ratio is a property of the CORPUS, so it is derived
    from the recorded pair and the sentence names the part it is about."""
    healthy, healthy_max = T.THERMAL_HEALTHY_OBSERVED_MHZ
    expect = healthy / T.thermal_floor_mhz(healthy_max)
    assert f"{expect:.2f}x" in TA.resolution_line(healthy_max, "NVIDIA H200")
    a100 = TA.resolution_line(1410.0, "NVIDIA A100-SXM4-80GB")
    # the H200 ratio may appear, but only attached to the H200's own maximum
    assert f"{healthy_max:.0f} MHz part" in a100
    assert f"{T.thermal_floor_mhz(1410.0):.0f} MHz" in a100
    # and nothing here is a typed ratio
    assert "1.93x above it" not in a100
