"""The clock-elasticity arm: its estimator, its gates, its instrument and its exits.

WHAT THIS ARM IS FOR and therefore what these tests have to hold it to. The
session-3 reading found alpha unidentified for three independent reasons, one of
which is that the card is power-capped, the SM clock is an endogenous response to
the tile and the tread, and NOTHING in the corpus moves the clock at a
byte-identical kernel. Sweeping the admissible elasticity moves pooled EXA
alpha_b from 0.974 to 0.897, 21x the quoted sd. So the number this arm returns is
load-bearing for every alpha in the study, and the three things that can silently
make it wrong are:

    the SIGN     -- d log ms / d log f is negative and the registered bands are
                    positive, so a convention applied twice, or nowhere, flips a
                    retraction into a confirmation
    the EXCLUSION RULE -- LEVEL must be kept on BOTH sides (at a duty cycle below
                    1.0 the boosted rows ARE the experiment) and DRIFT must
                    exclude; a filter on the wrong verdict drops the signal
    the INTERVAL -- a bootstrap over ROWS rather than over REPEATS reports a
                    width several times too narrow, which is the understatement
                    the session-3 reading already found once

Every one of those has a test below that fails if it regresses.
"""
from __future__ import annotations

import ast
import csv
import dataclasses
import importlib.util
import inspect
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _hermetic import laptop_env  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.bench import timing as T  # noqa: E402

SCRIPT = ROOT / "scripts" / "clock_elasticity.py"
SOURCE = SCRIPT.read_text()


def _load():
    spec = importlib.util.spec_from_file_location("clock_elasticity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


CE = _load()


def run(args, timeout=900):
    # Laptop path on every box: the bare and --duty rows of OFF_GPU_MODES and
    # the measuring-run refusal would otherwise MEASURE a 40-minute ladder on
    # a pod with vLLM, and on session 4's base venv they crashed to ERROR 4
    # instead of REFUSED 2 (the script's stack door, this change).
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=timeout,
                          cwd=str(ROOT), env=laptop_env())


def parsed(args):
    return CE.build_parser().parse_args(args)


# --------------------------------------------------------------------------
# 1. the exit-code contract, in every off-GPU mode
# --------------------------------------------------------------------------

#: Every mode this file can be run in without a card, and what it must exit.
OFF_GPU_MODES = (
    (["--dry-run"], exit_codes.REFUSED),
    (["--dry-run", "--repeats", "3"], exit_codes.REFUSED),
    (["--self-test", "--draws", "200"], exit_codes.DONE),
    (["--duty", "1.0", "0.5"], exit_codes.REFUSED),
    (["--duty", "1.0", "0.5", "0.5"], exit_codes.REFUSED),
    ([], exit_codes.REFUSED),
)


@pytest.mark.parametrize("argv,code", OFF_GPU_MODES)
def test_every_off_gpu_mode_exits_the_code_its_own_page_implies(argv, code):
    """THE PROPERTY, over every mode: a log with RESULT lines recomputes the
    process's own exit code, and a log with none exits REFUSED. A script that
    prints one thing and exits another is the defect moe/bench/exit_codes.py is
    named against."""
    got = run(argv)
    assert got.returncode == code, (argv, got.stdout[-3000:], got.stderr[-2000:])
    lines = exit_codes.parse_result_lines(got.stdout)
    if lines:
        assert exit_codes.classify_text(got.stdout) == got.returncode
    else:
        assert got.returncode == exit_codes.REFUSED, argv


def test_a_dry_run_prints_a_plan_scores_nothing_and_writes_nothing(tmp_path):
    """A plan is not a measurement. It prints the registered bands, the design
    arithmetic and the priced cost, scores no gate, and exits REFUSED -- which is
    what `classify_text` raising NoGatesScored over its log means."""
    got = run(["--dry-run", "--out", str(tmp_path)])
    assert got.returncode == exit_codes.REFUSED
    assert "RESULT: " not in got.stdout
    with pytest.raises(exit_codes.NoGatesScored):
        exit_codes.classify_text(got.stdout)
    for want in ("PREDICTIONS, registered before the run", "RESOLUTION",
                 "estimated wall time", "THE THREE REGISTERED BANDS",
                 "V1 THRESHOLD"):
        assert want in got.stdout, want
    assert not list(tmp_path.rglob("*")), "a plan wrote something"


def test_every_gate_prints_exactly_one_result_line_and_nothing_else_does():
    got = run(["--self-test", "--draws", "200"])
    lines = exit_codes.parse_result_lines(got.stdout)
    raw = [ln for ln in got.stdout.splitlines() if ln.startswith("RESULT: ")]
    assert len(raw) == len(lines), "a RESULT line the parser cannot read back"
    assert [r.name for r in lines] == ["S1", "S2", "S3", "S4", "S5"]


def test_an_unplanned_crash_is_error_and_never_claim_fail(monkeypatch):
    """ERROR (4) is the only retryable code. Left to propagate an exception exits
    the interpreter ONE, and ONE is CLAIM_FAIL, which this table defines as a
    RESULT -- so a crashed arm would be filed as 'the clock carries the time' and
    C3's direction retracted over a run that never measured."""
    monkeypatch.setattr(CE, "_main", lambda argv=None: (_ for _ in ()).throw(
        RuntimeError("planted")))
    rc = CE.main([])
    assert rc == exit_codes.ERROR
    assert rc != exit_codes.CLAIM_FAIL
    assert rc not in exit_codes.FINISHED_CODES
    assert exit_codes.ledger_state(rc) == "RETRY"


def test_this_file_defines_no_gate_softening_flag():
    """bn_decomposition and occupancy_vs_swizzle both downgrade a CLAIM_FAIL to
    DONE when their gate flag is absent, and the driver has to remember to pass
    it. There is no such flag here: `classify` over the gates IS the exit code,
    so a failed claim is CLAIM_FAIL whether or not anyone remembered."""
    for flag in ("--fail-on-gate", "--fail-on-claim", "--fail-on-world"):
        assert f'"{flag}"' not in SOURCE, flag
    known = {a.option_strings[0] for a in CE.build_parser()._actions
             if a.option_strings}
    assert not {f for f in known if f.startswith("--fail-on")}


# --------------------------------------------------------------------------
# 2. the corpus figures PRICE the run and must still be the corpus's
# --------------------------------------------------------------------------

PUBLISHED = (ROOT / "results" / "published" /
             "2026-09-10-nvidia_h200-gaps-session" / "results" /
             "bn_decomposition" /
             "nvidia_h200-bm32_64_128-budget400.0-dtypebf16-flushtrue-g16-"
             "iters50-k64-modelmixtral_8x7b-n32_64-b59b409f" / "cells.csv")


def _published_treads():
    by = {}
    with PUBLISHED.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if (row["status"] == "ok" and row["block_n"] == "64"
                    and row["block_m"] == "32"):
                by.setdefault(int(row["tiles"]), []).append(float(row["ms_p50"]))
    return by


@pytest.mark.skipif(not PUBLISHED.exists(), reason="the published arm is absent")
def test_the_priced_ladder_is_still_the_one_that_was_published():
    """CORPUS_LADDER_MS and CORPUS_REPEAT_SPREAD price this arm and, through the
    MDE, CHOOSE its V1 threshold. Recomputed here from the committed file rather
    than trusted, so neither can drift away from what was measured -- which is
    the difference between a number with a provenance and a number with a
    comment."""
    by = _published_treads()
    got = [statistics.median(by[n]) for n in sorted(by)]
    assert len(got) >= len(CE.CORPUS_LADDER_MS)
    for want, saw in zip(CE.CORPUS_LADDER_MS, got, strict=False):
        assert abs(want - saw) < 5e-4, (want, saw)
    spreads = [statistics.pstdev(v) / statistics.median(v)
               for v in by.values() if len(v) > 1]
    assert abs(statistics.median(spreads) - CE.CORPUS_REPEAT_SPREAD) < 5e-5


def test_the_priced_ladder_scores_nothing():
    """It is an input to the PRICE and to the design arithmetic, and to nothing
    that decides a verdict. Asked of the source: no gate function may mention
    it."""
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("gate_"):
            body = ast.unparse(node)
            assert "CORPUS_LADDER_MS" not in body, node.name
            assert "corpus_call_ms" not in body, node.name


# --------------------------------------------------------------------------
# 3. the sign convention, applied ONCE
# --------------------------------------------------------------------------

def test_the_reported_elasticity_is_positive_and_the_slope_is_negative():
    """d log ms / d log f is NEGATIVE and the registered bands are POSITIVE.
    Both numbers are on the record, and this is the test that says which is
    which: a planted 0.60 comes back as eta +0.60 and slope -0.60."""
    est = CE.fit(CE.plant_rows(eps=0.60, jitter=0.0), draws=0)
    assert est.value == pytest.approx(0.60, abs=1e-6)
    assert est.slope == pytest.approx(-0.60, abs=1e-6)
    assert est.value == pytest.approx(CE.ETA_SIGN * est.slope)


def test_the_sign_is_applied_at_one_place():
    """THE RECURRING DEFECT, in the one form that would be silent here: a second
    negation somewhere downstream turns a retraction into a confirmation and
    every printed number still looks reasonable. `ETA_SIGN` appears in `fit` and
    nowhere else that computes a value."""
    tree = ast.parse(SOURCE)
    users = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # CODE, not docstrings: a docstring that NAMES the convention is the
        # documentation working, and a test that counted it would push the
        # explanation out of the file it explains.
        body = "\n".join(
            ast.unparse(stmt) for stmt in node.body
            if not (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)))
        if "ETA_SIGN" in body:
            users.add(node.name)
    assert users == {"fit"}, users


@pytest.mark.parametrize("eps", [0.0, 0.05, 0.32, 0.60, 1.0])
def test_the_estimator_recovers_what_was_planted(eps):
    est = CE.fit(CE.plant_rows(eps=eps, jitter=0.0), draws=0)
    assert est.value == pytest.approx(eps, abs=1e-6)


def test_a_pure_bandwidth_world_reads_zero_and_a_pure_issue_rate_world_reads_one():
    """The two physical anchors. A time that does not move with the clock is
    traffic (0); a time inversely proportional to the clock is issue rate (1).
    Without these the estimator could be off by a factor and every planted world
    in between would still agree with it."""
    assert CE.fit(CE.plant_rows(eps=0.0, jitter=0.0), draws=0).value == \
        pytest.approx(0.0, abs=1e-9)
    assert CE.fit(CE.plant_rows(eps=1.0, jitter=0.0), draws=0).value == \
        pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# 4. the exclusion rule: DRIFT excludes, LEVEL does not, on either side
# --------------------------------------------------------------------------

def test_both_sides_of_level_are_kept_and_drift_is_excluded():
    """THE FIVE-ROW PLANTING the repository's side-blind tripwire asks for, and
    the fifth row is the one every earlier world in this tree was missing: a
    tread that is LEVEL HIGH *and* DRIFTING at once. Counting on LEVEL alone
    kept that tread twice and stayed green.

    At a duty cycle below 1.0 a LEVEL HIGH row is the experiment working. A
    filter on LEVEL here would drop exactly the states that carry the signal."""
    rows = CE.plant_rows(
        eps=0.05, jitter=0.0,
        level_at={(0, 1, 0): T.LEVEL_HIGH, (0, 2, 0): T.LEVEL_LOW,
                  (0, 3, 0): T.LEVEL_HIGH},
        drift_at={(0, 3, 0), (0, 4, 0)},
        host_at={(0, 5, 0)})
    by = {(r.state_index, r.tiles, r.repeat): r for r in rows}
    assert CE.exclusion(by[(0, 1, 0)]) == "", "a LEVEL HIGH row was excluded"
    assert CE.exclusion(by[(0, 2, 0)]) == "", "a LEVEL LOW row was excluded"
    assert CE.exclusion(by[(0, 3, 0)]) == CE.DROP_DRIFT, "HIGH *and* DRIFTING"
    assert CE.exclusion(by[(0, 4, 0)]) == CE.DROP_DRIFT
    assert CE.exclusion(by[(0, 5, 0)]) == CE.DROP_HOST
    counts = CE.level_counts(CE.kept_rows(rows))
    assert counts[T.LEVEL_HIGH] >= 1 and counts[T.LEVEL_LOW] >= 1


def test_only_one_function_decides_whether_a_row_is_in_the_fit():
    """The 2026-09-15 defect in this tree's own words: a guard applied at one of
    N call sites. Every reader of `clock_drift_ok` and `host_bound` that could
    DROP a row has to be `exclusion`, so a rule changed there changes it
    everywhere."""
    tree = ast.parse(SOURCE)
    owner = {}
    for top in tree.body:
        if isinstance(top, ast.FunctionDef):
            for node in ast.walk(top):
                if isinstance(node, ast.FunctionDef):
                    owner[node] = top.name
    readers = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = "\n".join(
            ast.unparse(stmt) for stmt in node.body
            if not (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)))
        if "clock_drift_ok is False" in body or "row.host_bound" in body:
            readers.add(owner.get(node, node.name))
    assert readers == {"exclusion"}, readers


def test_this_file_reads_the_side_of_every_level_verdict():
    """The repository-wide tripwire, asserted here too so the reason lives beside
    the code: a reader of `clock_level_ok` that never reads `clock_level_side`
    reads a HIGH failure as a LOW one."""
    assert "clock_level_ok" in SOURCE and "clock_level_side" in SOURCE
    assert "T.LEVEL_HIGH" in SOURCE and "T.LEVEL_LOW" in SOURCE, (
        "the sides are read through timing's own names, so a rename moves them "
        "here too")
    assert T.LEVEL_HIGH == "high" and T.LEVEL_LOW == "low", (
        "if the values change, this file's planted worlds move with them")


# --------------------------------------------------------------------------
# 5. the interval: a bootstrap over REPEATS, not over rows
# --------------------------------------------------------------------------

def test_the_bootstrap_resamples_repeats_and_not_rows():
    """Rows inside one repeat share whatever the card was doing, so resampling
    rows treats 13 correlated passes as hundreds of independent draws and reports
    an interval a factor of several too narrow. Read off the source: the thing
    the bootstrap draws from is the repeat list."""
    body = inspect.getsource(CE.fit)
    assert "rng.choice(repeats)" in body
    assert "rng.choice(keep)" not in body and "rng.choice(rows)" not in body


def test_fewer_repeats_give_a_wider_interval():
    wide = CE.fit(CE.plant_rows(eps=0.30, jitter=0.02, repeats=3, seed=5),
                  draws=600, seed=1)
    tight = CE.fit(CE.plant_rows(eps=0.30, jitter=0.02, repeats=21, seed=5),
                   draws=600, seed=1)
    assert wide.half_width > tight.half_width


def test_the_interval_is_a_95_percent_one_and_covers_the_planted_value():
    est = CE.fit(CE.plant_rows(eps=0.30, jitter=0.01, repeats=13, seed=3),
                 draws=1000, seed=2)
    assert est.lo < 0.30 < est.hi
    assert est.resampled == 1000


# --------------------------------------------------------------------------
# 6. the gates, each planted both ways
# --------------------------------------------------------------------------

def _score(rows, **overrides):
    args = CE._self_test_args(CE.build_parser().parse_args(["--dry-run"]))
    for key, value in overrides.items():
        setattr(args, key, value)
    threshold, source = CE.registered_clock_ratio(args)
    est = CE.fit(rows, draws=400, seed=0)
    return {g.token: g for g in CE.gates_for(rows, args, threshold, source, est)}


def test_v1_passes_on_a_separated_design_and_fails_on_one_clock():
    ok = _score(CE.plant_rows(eps=0.05, jitter=0.0))
    assert ok["V1"].verdict == CE.PASS
    flat = _score(CE.plant_rows(eps=0.05, mhz=(1650.0,) * 4))
    assert flat["V1"].verdict == CE.FAIL
    assert "1650" not in flat["V1"].threshold, "the threshold is a RATIO, not a clock"


def test_v1_fails_when_one_tread_out_of_eight_did_not_separate():
    """Per tread, not pooled. A tread whose clock never moved contributes nothing
    to Sxx and the pooled slope simply ignores it, so a pooled-only gate would
    quietly fit a design smaller than the one that was booked."""
    rows = [r for r in CE.plant_rows(eps=0.05, jitter=0.0)
            if not (r.tiles == 4 and r.state_index > 0)]
    stuck = [r for r in CE.plant_rows(eps=0.05, jitter=0.0)
             if r.tiles == 4 and r.state_index > 0]
    for r in stuck:
        r.sm_clock_load_mhz = CE.PLANTED_STATE_MHZ[0]
    assert _score(rows + stuck)["V1"].verdict == CE.FAIL


def test_v2_reads_the_tile_off_the_rows():
    assert _score(CE.plant_rows(eps=0.05, jitter=0.0))["V2"].verdict == CE.PASS
    mixed = _score(CE.plant_rows(eps=0.05, jitter=0.0,
                                 block_m_at={(0, 1, 0): 64}))
    assert mixed["V2"].verdict == CE.FAIL
    shapes = CE.plant_rows(eps=0.05, jitter=0.0)
    shapes[0].calls_per_burst = 9
    assert _score(shapes)["V2"].verdict == CE.FAIL, (
        "one tread launched in two burst shapes carries two shares of one "
        "discarded lead call")


def test_v3_fails_on_a_design_that_is_deep_in_one_state_only():
    deep = CE.plant_rows(eps=0.05, jitter=0.0, repeats=13)
    thin = [r for r in deep
            if r.state_index == 0 or (r.repeat < 2 and r.tiles < 3)]
    assert _score(thin)["V3"].verdict == CE.FAIL
    assert _score(deep)["V3"].verdict == CE.PASS


def test_v4_counts_drift_and_host_bound_and_never_level():
    rows = CE.plant_rows(eps=0.05, jitter=0.0)
    assert _score(rows)["V4"].verdict == CE.PASS
    every = {(i, t, r) for i in range(4) for t in range(1, 9) for r in range(13)}
    all_high = _score(CE.plant_rows(
        eps=0.05, jitter=0.0,
        level_at=dict.fromkeys(every, T.LEVEL_HIGH)))
    assert all_high["V4"].verdict == CE.PASS, (
        "a run where every row sat LEVEL HIGH is a run at a boosted clock, "
        "which is the experiment, not an exclusion")
    drifted = _score(CE.plant_rows(
        eps=0.05, jitter=0.0,
        drift_at={k for k in every if (k[1] + k[2]) % 3 == 0}))
    assert drifted["V4"].verdict == CE.FAIL


def test_v5_sees_a_sag_inside_a_burst_that_every_drift_verdict_passes():
    """A card that boosts at the start of each 40 ms burst and sags by its end
    has first == last on every burst-to-burst comparison, so DRIFT passes
    everywhere and the per-call time is still an average over two operating
    points. V5 is the only gate that can see it."""
    sagging = CE.plant_rows(eps=0.05, jitter=0.0,
                            burst_moves_at={(0, 1, 0)})
    gates = _score(sagging)
    assert gates["V5"].verdict == CE.FAIL
    assert all(r.clock_drift_ok is not False for r in sagging)
    assert _score(CE.plant_rows(eps=0.05, jitter=0.0))["V5"].verdict == CE.PASS


def test_v6_catches_a_memory_clock_that_moved_with_the_duty_cycle():
    every = {(i, t, r) for i in range(4) for t in range(1, 9) for r in range(13)}
    moved = _score(CE.plant_rows(
        eps=0.05, jitter=0.0,
        mem_at={k: CE.PLANTED_MEM_MHZ + 200.0 * k[0] for k in every}))
    assert moved["V6"].verdict == CE.FAIL
    assert _score(CE.plant_rows(eps=0.05, jitter=0.0))["V6"].verdict == CE.PASS


def test_v0_refuses_a_run_that_measured_nothing():
    assert CE.gate_v0_non_vacuity([], []).verdict == CE.FAIL


@pytest.mark.parametrize("eps,band,c1,c2", [
    (0.05, "RAW-STANDS", CE.PASS, CE.PASS),
    (0.32, "UNREGISTERED-GAP", CE.PASS, CE.FAIL),
    (0.60, "CLOCK-CARRIES", CE.PASS, CE.FAIL),
])
def test_each_registered_band_is_reachable_and_names_its_own_consequence(
        eps, band, c1, c2):
    gates = _score(CE.plant_rows(eps=eps, jitter=0.002, seed=4))
    assert gates["C1"].verdict == c1
    assert gates["C2"].verdict == c2
    assert band in gates["C1"].measured
    named = [b for b in CE.BANDS if b[0] == band][0]
    page = "\n".join(gates["C1"].render() + gates["C2"].render())
    assert named[3][:40] in page, "the band's registered consequence is printed"


def test_c1_fails_when_the_interval_crosses_a_registered_boundary():
    """A FAIL here is a result about the DESIGN, not about the card: an interval
    across a boundary is consistent with two worlds whose readings contradict
    each other, and picking the nearer one is how a pre-registration becomes a
    post-registration."""
    gates = _score(CE.plant_rows(eps=0.25, jitter=0.04, seed=11))
    assert gates["C1"].verdict == CE.FAIL
    assert gates["C2"].verdict == CE.FAIL
    assert "not shown" in "\n".join(gates["C2"].lines)


def test_band_of_refuses_an_interval_that_straddles():
    assert CE.band_of(0.05, 0.10)[0] == "RAW-STANDS"
    assert CE.band_of(0.20, 0.30) is None
    assert CE.band_of(0.30, 0.50) is None
    assert CE.band_of(0.45, 0.90)[0] == "CLOCK-CARRIES"
    assert CE.band_of(None, None) is None


# --------------------------------------------------------------------------
# 7. the V1 threshold is COMPUTED, never asserted
# --------------------------------------------------------------------------

def test_a_shallower_design_is_held_to_a_wider_separation():
    """The threshold is the MDE arithmetic's own answer, so it moves with the
    design. A constant would let a three-repeat run claim the resolution a
    thirteen-repeat run was sized for."""
    few = CE.required_clock_ratio(repeats=3, treads=8, states=4)
    many = CE.required_clock_ratio(repeats=21, treads=8, states=4)
    assert few > many > 1.0
    narrow = CE.required_clock_ratio(repeats=13, treads=2, states=4)
    assert narrow > CE.required_clock_ratio(repeats=13, treads=8, states=4)


def test_the_threshold_on_the_plan_page_is_the_one_the_gate_uses():
    plan = run(["--dry-run", "--repeats", "3"]).stdout
    threshold, _ = CE.registered_clock_ratio(parsed(["--repeats", "3"]))
    assert f"V1 THRESHOLD {threshold:.3f}x" in plan
    other = run(["--dry-run", "--repeats", "21"]).stdout
    assert plan.split("V1 THRESHOLD")[1][:8] != other.split("V1 THRESHOLD")[1][:8]


def test_the_operator_can_overrule_the_threshold_and_the_page_says_who_did():
    args = parsed(["--min-clock-ratio", "1.5"])
    threshold, source = CE.registered_clock_ratio(args)
    assert threshold == 1.5 and "operator" in source


# --------------------------------------------------------------------------
# 8. the instrument, driven off GPU through its injection seams
# --------------------------------------------------------------------------

class FakeEvents:
    """`_EventPairs`' shape, scripted. Records the order of every operation so
    the flush/record/gap arrangement can be asserted rather than assumed."""

    log: list = []
    per_call_ms = 2.0
    lead_ms = 8.0

    def __init__(self, n):
        self.n = n
        self.starts = [_Rec(self, "start", i) for i in range(n)]
        self.ends = [_Rec(self, "end", i) for i in range(n)]

    def synchronize(self):
        FakeEvents.log.append(("sync", -1))

    def elapsed(self, n):
        # The lead call launches into a drained queue and costs more; every
        # other call is the kernel alone. That difference is what the discard
        # exists for, so the fake has to have it.
        return [FakeEvents.lead_ms if i == 0 else FakeEvents.per_call_ms
                for i in range(n)]


class _Rec:
    def __init__(self, owner, kind, index):
        self.owner, self.kind, self.index = owner, kind, index

    def record(self):
        FakeEvents.log.append((self.kind, self.index))


class FakeFlusher:
    megabytes = 256

    def flush(self):
        FakeEvents.log.append(("flush", -1))


def _clock_reader(values):
    it = iter(values)
    last = [values[-1]]

    def read():
        try:
            mhz = next(it)
        except StopIteration:
            mhz = last[0]
        FakeEvents.log.append(("clock", int(mhz)))
        return T.ClockState(int(mhz), 60, source=T.CLOCK_SOURCE_NVML,
                            power_w=690.0)
    return read


def _drive(duty=0.25, calls=8, bursts=3, trials=2, clocks=(1800,) * 12,
           per_call=2.0):
    FakeEvents.log = []
    FakeEvents.per_call_ms = per_call
    slept = []
    got = CE.time_duty(
        lambda: FakeEvents.log.append(("call", -1)),
        duty=duty, calls_per_burst=calls, bursts=bursts, trials=trials,
        warm_ms=0.0, l2_flush=True, per_call_ms=per_call,
        reference_clock_mhz=1650.0, clock_read=_clock_reader(list(clocks)),
        mem_read=lambda: 2619.0, events=FakeEvents, flusher=FakeFlusher(),
        sleep=slept.append)
    return got, slept


def test_a_box_with_a_card_and_no_vllm_is_refused_not_crashed(tmp_path, monkeypatch, capsys):
    """Session 4's base venv: a card, no vLLM. The bare run got past the device
    door and crashed to ERROR 4 (the driver's retryable code) with a traceback;
    the stack door now refuses with the missing half named and writes nothing."""
    monkeypatch.setattr(CE, "resolve_card", lambda args: "NVIDIA H200")
    monkeypatch.setattr(CE.SWEEP, "missing_gpu_stack",
                        lambda: "vLLM is not importable from this interpreter; "
                                "source the vllm venv")
    monkeypatch.setattr(CE.T, "require_cuda", lambda: None)
    rc = CE.main(["--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED, out[-800:]
    assert "vLLM is not importable" in out
    assert "Traceback" not in out
    assert not list(tmp_path.rglob("CARD")), "the refused run wrote a CARD file"


def test_time_duty_refuses_off_gpu_unless_every_seam_is_injected(no_cuda):
    """`time_kernel`'s own terms, for its own reason: an instrument that invents
    numbers when its device is missing is worse than one that stops."""
    with pytest.raises(T.TimingRefused) as caught:
        CE.time_duty(lambda: None, duty=0.5, calls_per_burst=8, bursts=1,
                     trials=1, warm_ms=0.0, l2_flush=True, per_call_ms=1.0,
                     reference_clock_mhz=None)
    assert "events" in str(caught.value) and "clock_read" in str(caught.value)


def test_a_burst_needs_two_calls_because_one_of_them_is_discarded():
    with pytest.raises(T.TimingRefused):
        CE.time_duty(lambda: None, duty=0.5, calls_per_burst=1, bursts=1,
                     trials=1, warm_ms=0.0, l2_flush=True, per_call_ms=1.0,
                     reference_clock_mhz=None, clock_read=_clock_reader([1800]),
                     events=FakeEvents, flusher=FakeFlusher())


def test_the_lead_call_of_every_burst_is_discarded():
    """It launched into a queue the previous gap had drained, so its interval
    carries launch latency the others do not. A constant additive offset does
    NOT cancel in a log slope, which is why it is discarded rather than
    tolerated."""
    got, _ = _drive(calls=8, bursts=3, trials=2, per_call=2.0)
    assert got.samples == 3 * 2 * 7, "one call per burst must be dropped"
    assert got.dropped_leads == 6
    assert got.ms_p50 == pytest.approx(2.0), (
        "the 8 ms lead reached the percentiles")
    assert got.ms_min == pytest.approx(2.0)


def test_the_gap_is_outside_every_measured_interval_and_sets_the_duty():
    """The gap is the independent variable and it is a host-side sleep AFTER the
    synchronise, so no event pair can contain it."""
    got, slept = _drive(duty=0.25, calls=8, bursts=3, trials=2, per_call=2.0)
    assert got.gap_ms == pytest.approx(8 * 2.0 * 3.0)
    assert len(slept) == 6 and all(s == pytest.approx(got.gap_ms / 1000.0)
                                   for s in slept)
    order = FakeEvents.log
    first_sync = order.index(("sync", -1))
    assert ("clock", 1800) in order[:first_sync], (
        "the clock must be read BEFORE the synchronise, with the burst in "
        "flight; a reading taken after it describes an idle card")
    # flush, start, call, end -- the flush is outside the pair by construction.
    window = order[order.index(("flush", -1)):]
    assert window[:4] == [("flush", -1), ("start", 0), ("call", -1), ("end", 0)]


def test_duty_one_asks_for_no_gap_at_all():
    got, slept = _drive(duty=1.0)
    assert got.gap_ms == pytest.approx(0.0)
    assert slept == []


def test_one_clock_sample_per_burst_and_the_verdicts_come_from_them():
    got, _ = _drive(bursts=3, trials=2, clocks=(1800,) * 6)
    assert len(got.clock_samples_mhz) == 6
    assert got.sm_clock_load_mhz == pytest.approx(1800.0)
    assert got.clock_drift_ok is True
    assert got.clock_level_side == T.LEVEL_HIGH, (
        "1800 against a 1650 reference is HIGH, and at a duty below 1.0 that "
        "is the experiment working")
    assert got.mem_clock_mhz == pytest.approx(2619.0)


def test_a_drifting_cell_is_flagged_and_names_its_direction():
    got, _ = _drive(bursts=3, trials=2,
                    clocks=(1800, 1800, 1790, 1500, 1400, 1300))
    assert got.clock_drift_ok is False
    assert got.clock_drift_direction == T.DRIFT_DOWN
    assert "EXCLUDED" in got.clock_note


def test_the_within_burst_quarters_are_taken_from_the_kept_calls():
    FakeEvents.log = []
    FakeEvents.per_call_ms = 2.0
    got, _ = _drive(calls=8, bursts=2, trials=1)
    assert got.head_ms == pytest.approx(2.0) and got.tail_ms == pytest.approx(2.0)
    assert got.within_burst_ok is True


def test_a_cell_with_too_few_clock_samples_carries_no_clock_and_says_so():
    got, _ = _drive(bursts=1, trials=1, clocks=(0, 0, 0))
    assert got.sm_clock_load_mhz is None
    assert "cannot read this cell" in got.clock_note
    row = CE.plant_rows(eps=0.05, jitter=0.0)[0]
    row.sm_clock_load_mhz = None
    assert CE.exclusion(row) == CE.DROP_NO_CLOCK


def test_the_burst_shape_rule_is_called_by_the_plan_and_by_the_runner():
    """One rule, two callers. A plan that sized bursts differently from the pod
    would price a run nobody is going to make -- which is the unit defect
    `block_m_crossing_sweep.estimated_seconds` carries its own note about."""
    tree = ast.parse(SOURCE)
    callers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and "burst_shape(" in ast.unparse(node):
            callers.add(node.name)
    assert {"estimated_seconds", "plan_lines", "run_arm"} <= callers, callers


@pytest.mark.parametrize("per_call", [0.7738, 2.4154, 4.6153, 20.0])
def test_every_burst_holds_enough_calls_for_the_quarters_v5_reads(per_call):
    calls, bursts, kept = CE.burst_shape(per_call, CE.DEFAULT_BURST_MS,
                                         CE.DEFAULT_TARGET_MS)
    assert calls >= CE.MIN_CALLS_PER_BURST
    # WHAT THE ASSERTION USED TO SAY AND DID NOT ESTABLISH: `calls - 1 >= 4`
    # was labelled "V5 takes a median of a quarter, not one sample", but a
    # quarter of four kept calls is ONE call. The quarter V5 actually reads is
    # `max(1, kept // 4)` over `got[1:]`, so a two-sample quarter needs eight
    # kept and nine launched. This asserts what the shape gives rather than a
    # sentence about it, and prints the quarter so a reader sees which floor a
    # tread landed on.
    kept_calls = calls - 1
    assert kept_calls >= 4, (calls, kept_calls)
    assert max(1, kept_calls // 4) >= 1
    assert kept >= T.iters_for(per_call, CE.DEFAULT_TARGET_MS)
    assert bursts * trials_floor() >= T.CLOCK_SAMPLE_FLOOR


def trials_floor():
    return CE.DEFAULT_TRIALS


# --------------------------------------------------------------------------
# 9. the row store
# --------------------------------------------------------------------------

def test_a_row_survives_the_csv_with_its_absences_intact(tmp_path):
    """`""` must read back as None and never as 0 or False. A `clock_drift_ok`
    that came back False where it was unknown would exclude every cell; one that
    came back True where it was FAILED would keep every drifting one."""
    path = tmp_path / "cells.csv"
    row = CE.plant_rows(eps=0.1, jitter=0.0, treads=1, repeats=1)[0]
    row.power_w = None
    row.clock_drift_ok = None
    row.within_burst_ok = None
    CE.append_row(path, row)
    back = CE.read_rows(path)
    assert len(back) == 1
    assert back[0].power_w is None
    assert back[0].clock_drift_ok is None
    assert back[0].within_burst_ok is None
    assert back[0].ms_p50 == pytest.approx(row.ms_p50)
    assert back[0].duty_requested == pytest.approx(row.duty_requested)
    assert back[0].clock_level_side == row.clock_level_side


def test_the_store_refuses_a_narrower_header_already_on_disk(tmp_path):
    """`DictWriter` writes the fieldnames it was given and never looks at the
    file, so a wider row appended under a narrower header shifts every field past
    the first difference: `clock_drift_ok` would then be read out of
    `clock_level_side`, and a FAILED drift -- the one verdict here that excludes
    a row -- would come back None, which every gate keeps."""
    path = tmp_path / "cells.csv"
    path.write_text("duty_requested,ms_p50\n1.0,2.0\n")
    row = CE.plant_rows(eps=0.1, jitter=0.0, treads=1, repeats=1)[0]
    with pytest.raises(CE.SchemaCollision) as caught:
        CE.append_row(path, row)
    assert caught.value.code == exit_codes.REFUSED
    assert "columns this run adds" in str(caught.value)


def test_a_fresh_file_is_written_with_the_header_and_appends_match_it(tmp_path):
    path = tmp_path / "cells.csv"
    for row in CE.plant_rows(eps=0.1, jitter=0.0, treads=2, repeats=1):
        CE.append_row(path, row)
    header = path.read_text().splitlines()[0].split(",")
    assert header == CE.ROW_FIELDS
    assert len(CE.read_rows(path)) == 2 * len(CE.DUTY_LEVELS)


def test_this_file_defines_no_store_class():
    """tests/test_shell_gates.py pins the set of `class Store` definitions under
    scripts/ to exactly three, because the 2026-09-15 header guard landed at two
    of them while its own prose said there were two. A fourth would turn that
    test red in another slice's file, so this appender is a module-level function
    -- with the header check that guard exists to install, which is the part that
    matters."""
    assert "class Store" not in SOURCE
    assert 'path.open("a", newline="")' in SOURCE
    assert "next(csv.reader(fh), [])" in SOURCE


# --------------------------------------------------------------------------
# 10. the run id
# --------------------------------------------------------------------------

#: Every parser dest, and whether it belongs in the key. IN: anything that
#: changes the milliseconds on a row. OUT: re-analysis of one set of cells, and
#: where the cells are filed.
ID_EXEMPT = {"dry_run", "self_test", "out", "run_id", "card", "draws",
             "seed_bootstrap", "min_clock_ratio"}


def test_every_knob_that_changes_a_measurement_is_in_the_run_id():
    args = parsed(["--card", "NVIDIA H200"])
    base = CE.default_run_id(args)
    for action in CE.build_parser()._actions:
        if not action.option_strings or action.dest in ("help",):
            continue
        if action.dest in ID_EXEMPT:
            continue
        moved = parsed(["--card", "NVIDIA H200"])
        current = getattr(moved, action.dest)
        if isinstance(current, bool):
            setattr(moved, action.dest, not current)
        elif isinstance(current, list):
            setattr(moved, action.dest, [*current, 0.75])
        elif isinstance(current, int):
            setattr(moved, action.dest, current + 1)
        elif isinstance(current, float):
            setattr(moved, action.dest, current + 1.0)
        else:
            setattr(moved, action.dest, "qwen2-57b-a14b")
        assert CE.default_run_id(moved) != base, (
            f"--{action.dest} changes the measurement and not the run id")


def test_re_analysis_knobs_are_not_in_the_run_id():
    """Two analyses of one sweep belong in one directory, which is why --ridge
    and --alpha are out of every other arm's key and why --draws is out of this
    one's."""
    base = CE.default_run_id(parsed(["--card", "NVIDIA H200"]))
    for flag, value in (("--draws", "17"), ("--seed-bootstrap", "9"),
                        ("--min-clock-ratio", "1.4")):
        assert CE.default_run_id(
            parsed(["--card", "NVIDIA H200", flag, value])) == base, flag


def test_the_run_id_carries_the_card_and_refuses_without_one():
    got = CE.default_run_id(parsed(["--card", "NVIDIA H200"]))
    assert got.startswith("nvidia_h200-")


# --------------------------------------------------------------------------
# 11. the design refusals, decided before any GPU time
# --------------------------------------------------------------------------

def test_two_states_are_refused_before_anything_is_spent():
    got = run(["--duty", "1.0", "0.5", "--card", "NVIDIA H200"])
    assert got.returncode == exit_codes.REFUSED
    assert "below the 3 this design needs" in got.stdout
    assert got.stdout.startswith("REFUSED before any GPU time"), (
        "a design refusal must be decided before a plan is even built")
    assert "RESULT: " not in got.stdout


def test_two_identical_duty_values_are_refused():
    got = run(["--duty", "1.0", "0.5", "0.5", "--card", "NVIDIA H200"])
    assert got.returncode == exit_codes.REFUSED
    assert "one state wearing two labels" in got.stdout


@pytest.mark.parametrize("duty", ["1.5", "0", "-0.5"])
def test_a_duty_outside_zero_to_one_is_refused_and_never_clamped(duty):
    """`time_duty` takes max(0, gap), so a duty above 1 would silently become
    duty 1 and the run would carry two states at one cadence under two labels --
    the duplicate-state refusal wearing a disguise the ledger cannot see."""
    got = run(["--duty", "1.0", "0.5", duty, "--card", "NVIDIA H200"])
    assert got.returncode == exit_codes.REFUSED
    assert "outside (0, 1]" in got.stdout
    assert "RESULT: " not in got.stdout


def test_a_model_whose_routing_cannot_form_a_full_stack_is_refused():
    """`rows_quantum` is 3 for deepseek-v2-lite, and BLOCK_M=32 stacks land on a
    legal row only by accident. Refused rather than nudged: a nudged row is not a
    full tile stack and the ladder here reads full stacks only."""
    got = run(["--model", "deepseek-v2-lite", "--card", "NVIDIA H200"])
    assert got.returncode == exit_codes.REFUSED
    assert "multiple of 3" in got.stdout


def test_a_measuring_run_without_a_card_is_refused():
    got = run([])
    assert got.returncode == exit_codes.REFUSED
    assert "no card was named" in got.stdout


# --------------------------------------------------------------------------
# 11b. the claim is the per-M-tile cost's elasticity (2026-09-22)
# --------------------------------------------------------------------------

def test_the_gated_reading_is_the_per_tile_costs_and_the_per_call_one_is_beside_it():
    """One elasticity for the whole planted call: every reading agrees, and the
    fields beside the claim carry the all-tread and per-call readings with
    intervals of their own."""
    est = CE.fit(CE.plant_rows(eps=0.60, jitter=0.0), draws=50)
    assert est.value == pytest.approx(0.60, abs=1e-6)
    assert est.fixed_tread == pytest.approx(0.60, abs=1e-6)
    assert est.per_tile_all_treads == pytest.approx(0.60, abs=1e-6)
    assert est.claim_min_tread == CE.CLAIM_MIN_TREAD == 2
    assert est.per_tile_treads == CE.DEFAULT_TREADS - CE.CLAIM_MIN_TREAD + 1
    assert est.lo is not None and est.fixed_tread_lo is not None
    assert est.per_tile_all_treads_lo is not None
    assert est.lo <= est.value <= est.hi
    assert est.fixed_tread_lo <= est.fixed_tread <= est.fixed_tread_hi
    assert (est.per_tile_all_treads_lo <= est.per_tile_all_treads
            <= est.per_tile_all_treads_hi)


def test_every_caller_of_the_per_tile_estimator_names_its_treads():
    """THE RECURRING DEFECT, at the estimator: the claim and the reading beside
    it are one function at two tread sets, and a default is how a second call
    site silently reads the other one. `min_tread` has none."""
    param = inspect.signature(CE.per_tile_slope).parameters["min_tread"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty
    for fn in (CE.tread_levels, CE.tread_one_departure):
        param = inspect.signature(fn).parameters["min_tread"]
        assert param.default is inspect.Parameter.empty, fn.__name__


def test_an_intercept_with_its_own_elasticity_does_not_enter_the_claim():
    """An intercept with its own elasticity, ON the additive law: eps_a=0.17
    plants the INTERCEPT's elasticity, and the one-tile call then reads 0.78,
    not 0.17, because at tread 1 the per-tile cost is still most of the call.
    Every tread lies on a' + b' n, tread 1 included, so the claim and the
    all-tread reading both return the per-tile cost's 1.00 and tread 1 sits
    on the claim's lines. The off-law world, which is the 2026-09-21 card, is
    the next test."""
    rows = CE.plant_rows(eps=1.00, eps_a=0.17, jitter=0.0)
    est = CE.fit(rows, draws=0)
    assert est.value == pytest.approx(1.00, abs=0.02), est.value
    assert est.per_tile_all_treads == pytest.approx(1.00, abs=0.02)
    assert est.fixed_tread is not None and 0.5 < est.fixed_tread < 0.95
    # The per-tread readings rise with depth, as the additive law says they
    # must: shallow treads carry more intercept.
    per = est.per_tread
    assert per[1] < per[CE.DEFAULT_TREADS]
    assert per[1] > 0.5, "an on-law intercept cannot pull tread 1 to 0.17"
    assert est.tread1_sensitivity_ms == pytest.approx(
        est.tread1_sensitivity_on_line_ms, rel=1e-3)
    assert est.tread1_level_ms == pytest.approx(est.tread1_level_on_line_ms,
                                                rel=1e-3)
    # The old world with one elasticity still plants one number everywhere.
    one = CE.fit(CE.plant_rows(eps=0.30, jitter=0.0), draws=0)
    assert one.value == pytest.approx(one.fixed_tread, abs=1e-6)


def test_a_tread_one_off_the_additive_law_moves_the_all_tread_reading_and_never_the_claim():
    """THE 2026-09-21 CARD, PLANTED: the one-tile call reading 0.17 per call
    beside a per-tile cost of 1.00, which no intercept elasticity inside
    a(f) + b(f) n produces. The claim over treads 2 and deeper recovers the
    planted 1.00 and its interval, drawn over the same treads, covers it; the
    all-tread reading is displaced above it by more than the design's own
    half-width, which is the defect the registered tread set removes."""
    # No physical intercept elasticity puts tread 1 anywhere near 0.17 while
    # the per-tile cost is 1.00: that is what "off the law" means here.
    for eps_a in (0.0, 0.17, 0.5, 1.0):
        on = CE.fit(CE.plant_rows(eps=1.00, eps_a=eps_a, jitter=0.0), draws=0)
        assert on.per_tread[1] > 0.5, (eps_a, on.per_tread[1])

    exact = CE.fit(CE.plant_rows(eps=1.00, off_law={1: 0.17}, jitter=0.0),
                   draws=0)
    assert exact.per_tread[1] == pytest.approx(0.17, abs=1e-9)
    assert exact.value == pytest.approx(1.00, abs=1e-6)
    assert exact.per_tile_all_treads > exact.value + CE.RESOLUTION_TARGET
    assert exact.tread1_sensitivity_ms < exact.tread1_sensitivity_on_line_ms / 2

    noisy = CE.fit(CE.plant_rows(eps=1.00, off_law={1: 0.17}, jitter=0.004,
                                 seed=3), draws=400, seed=1)
    assert noisy.lo < 1.00 < noisy.hi, (noisy.lo, noisy.hi)
    assert noisy.lo <= noisy.value <= noisy.hi, (
        "the interval was drawn over a different tread set from the point")
    assert noisy.per_tile_all_treads_lo > noisy.hi, (
        "the all-tread interval lies wholly above the claim's")


def test_the_page_names_the_claim_and_prints_the_per_call_reading_beside_it():
    rows = CE.plant_rows(eps=0.30, jitter=0.0)
    est = CE.fit(rows, draws=20)
    text = "\n".join(CE.report_lines(rows, est, parsed(["--self-test"])))
    assert "THE FIT: eta of the per-M-tile cost, -d log b / d log f" in text
    assert "over treads 2 and deeper" in text
    assert "eta, THE CLAIM                      0.3000 over 7 treads" in text
    assert "PRINTED BESIDE IT, never gated: the same estimator over EVERY tread" in text
    assert "eta, per-M-tile, every tread        0.3000" in text
    assert "tread 1 against the claim's lines" in text
    assert "PRINTED BESIDE IT, not the claim: eta of the per-CALL time" in text
    assert "eta, fixed tread, pooled            0.3000" in text
    assert "COMPANION: the elasticity of the ladder SLOPE" in text
    gate = CE.gate_c2_registered_reading(est)
    assert "per-M-tile cost" in gate.claim
    assert "over treads 2 and deeper" in gate.claim
    assert gate.verdict == exit_codes.FAIL          # 0.30 is in the gap


def test_no_description_calls_tread_one_the_swizzle_engaging():
    """The launch arithmetic refutes it: num_pid_m = cdiv(numel + E(BM-1), BM)
    = 8n+8 on the 2026-09-21 ladder, so at G=16 the real M-tiles sit in ONE
    group at n=1 and at n=2 alike, and n=2 already reads like every deeper
    tread. The page says what the data shows instead, and says the cause is
    not established."""
    for stale in ("swizzle engaging", "N-outer stream", "one group, an N-outer"):
        assert stale not in SOURCE, stale
    rows = CE.plant_rows(eps=0.30, jitter=0.0)
    text = "\n".join(CE.report_lines(rows, CE.fit(rows, draws=0),
                                     parsed(["--self-test"])))
    assert "off the additive law a(f) + b(f) n" in text
    assert "NOT ESTABLISHED, and it is not the swizzle" in text
    assert "cdiv(numel + E(BM-1), BM)" in text


def test_the_module_says_what_is_gated_and_what_is_printed_beside_it():
    """Finding 19's two sites in this file: the docstring's first lines and the
    sign convention both defined eta as -d log ms / d log f after the gate
    had moved to the per-M-tile cost."""
    head = CE.__doc__.split("\n\n")[0]
    assert "-d log b / d log f" in head
    assert "-d log ms / d log f" not in head
    assert "treads 2 and deeper" in CE.__doc__
    sign = SOURCE.split("#: THE SIGN CONVENTION")[1].split("ETA_SIGN = ")[0]
    assert "eta_b = - d log b / d log f" in sign
    assert "Every gate reads `eta`;" not in sign


def test_a_passing_page_reads_the_per_tile_cost_and_not_the_whole_millisecond():
    """Two more of finding 19's sites: with every gate passed the page's reading
    said 'the measured millisecond is traffic', and C2's docstring said a FAIL
    means the clock carries 'a measured millisecond'. The claim is the per-M-tile
    cost over treads 2 and deeper, and tread 1 is left out of it because it sits
    off the law, so a PASS says nothing about the whole per-call millisecond."""
    rows = CE.plant_rows(eps=0.05, jitter=0.004)
    args = CE._self_test_args(CE.build_parser().parse_args(["--dry-run"]))
    threshold, source = CE.registered_clock_ratio(args)
    gates = CE.gates_for(rows, args, threshold, source,
                         CE.fit(rows, draws=400, seed=0))
    assert all(g.verdict == CE.PASS for g in gates), {
        g.token: g.verdict for g in gates}
    reading = "\n".join(CE.report_tail([], gates)).split("READING IT.")[1]
    assert ("at this cell the per-M-tile cost, over treads 2 and deeper, is "
            "traffic, not issue rate") in reading
    assert "millisecond" not in reading
    c2 = " ".join(CE.gate_c2_registered_reading.__doc__.split())
    assert "millisecond" not in c2
    assert "the per-M-tile cost over treads 2 and deeper" in c2


# --------------------------------------------------------------------------
# 11c. the claim's tread set, pinned on the cells that motivated it
# --------------------------------------------------------------------------

SESSION4_G16 = (ROOT / "tests" / "fixtures" /
                "2026-09-21-nvidia_h200-session4-clock_elasticity-g16" /
                "cells.csv")

#: The chain's R1 flags (scripts/alpha_g_chain.sh r1_cmd) that shape the plan,
#: less --duty, which is the subset under test.
CHAIN_R1 = ["--model", "mixtral-8x7b", "--dtype", "bf16", "--group-m", "16",
            "--treads", "8", "--repeats", "13", "--burst-ms", "40",
            "--target-ms", "200", "--trials", "3", "--warm-ms", "200",
            "--settle-seconds", "10"]


def _session4(duties):
    want = {CE._duty_key(d) for d in duties}
    rows = [r for r in CE.read_rows(SESSION4_G16)
            if CE._duty_key(r.duty_requested) in want]
    args = parsed(CHAIN_R1 + ["--duty", *(str(d) for d in duties)])
    threshold, source = CE.registered_clock_ratio(args)
    est = CE.fit(rows, draws=1000, seed=0)
    return rows, est, {g.token: g for g in CE.gates_for(rows, args, threshold,
                                                        source, est)}


def test_the_session4_fixture_is_the_published_run():
    """Read by the arm's own reader, so the columns it carries are the ones
    the fit and the gates read; its README names the source and the commit."""
    rows = CE.read_rows(SESSION4_G16)
    assert len(rows) == 13 * 4 * 8
    assert {r.group_m for r in rows} == {16}
    assert {CE._duty_key(r.duty_requested) for r in rows} == {
        CE._duty_key(d) for d in (1.0, 0.5, 0.25, 0.1)}
    readme = (SESSION4_G16.parent / "README.md").read_text()
    assert "2c19a4ce7617a6aeaa025614cd97e45a24024723" in readme
    assert "pod-h200-session4" in readme


@pytest.mark.parametrize("duties", [(1.0, 0.5, 0.25, 0.1), (1.0, 0.5, 0.25),
                                    (1.0, 0.5)])
def test_session4_g16_rescored_is_admissible_and_clock_carries(duties):
    """THE BLOCKER, pinned. Read over every tread, these cells put the claim's
    interval wholly above V7's admissible edge and the page exits INVALID; read
    over treads 2 and deeper, as registered, V7 passes and the interval sits
    wholly in CLOCK-CARRIES. The all-tread reading lies above the claim, and V7
    applied to ITS interval still fails, which is why it is printed and not
    gated. {1.0, 0.5} is two states, below MIN_STATES, so V3 is not asked of
    it; the others are whole pages and read CLAIM_FAIL, a result."""
    _rows, est, gates = _session4(duties)
    assert gates["V7"].verdict == CE.PASS, gates["V7"].measured
    assert gates["C1"].verdict == CE.PASS
    assert CE.band_of(est.lo, est.hi)[0] == "CLOCK-CARRIES"
    assert "CLOCK-CARRIES" in gates["C1"].measured
    assert est.per_tile_all_treads > est.value
    assert est.per_tile_all_treads_lo > est.lo
    every = dataclasses.replace(est, lo=est.per_tile_all_treads_lo,
                                hi=est.per_tile_all_treads_hi)
    assert CE.gate_v7_admissible(every).verdict == CE.FAIL
    # Tread 1 is off the claim's line, below it in sensitivity and above it in
    # level, which is the departure the page prints.
    assert est.tread1_sensitivity_ms < est.tread1_sensitivity_on_line_ms
    assert est.tread1_level_ms > est.tread1_level_on_line_ms
    if len(duties) >= CE.MIN_STATES:
        validity = [t for t, g in gates.items() if g.kind == CE.VALIDITY]
        assert all(gates[t].verdict == CE.PASS for t in validity), {
            t: gates[t].verdict for t in validity}
        rc = exit_codes.classify(g.scored() for g in gates.values())
        assert rc == exit_codes.CLAIM_FAIL


def test_the_session4_page_prints_tread_one_beside_the_claim():
    rows, est, gates = _session4((1.0, 0.5, 0.25, 0.1))
    args = parsed(CHAIN_R1 + ["--duty", "1.0", "0.5", "0.25", "0.1"])
    text = "\n".join(CE.report_tail(CE.report_lines(rows, est, args),
                                    list(gates.values())))
    assert f"{est.value:.4f} over 7 treads from tread 2" in text
    assert (f"{est.per_tile_all_treads:.4f}  [{est.per_tile_all_treads_lo:.4f}, "
            f"{est.per_tile_all_treads_hi:.4f}]") in text
    assert (f"{est.tread1_sensitivity_ms:.4f} ms against "
            f"{est.tread1_sensitivity_on_line_ms:.4f} ms on the line") in text
    assert "RESULT: VALIDITY V7 PASS" in text


def _helpers():
    spec = importlib.util.spec_from_file_location(
        "alpha_g_chain_helpers", ROOT / "scripts" / "alpha_g_chain_helpers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_report_shape_on_disk_still_loads_through_the_chains_reader(tmp_path):
    """Three shapes of `elasticity` are on disk and the chain's reader takes
    `value`, `lo` and `hi` from any of them. The new keys ADD to the block and
    rename nothing that reader reads; the Elasticity docstring says how to
    tell the three apart."""
    H = _helpers()
    session4 = {"value": 0.7436, "slope": -0.7436, "lo": 0.7277, "hi": 0.7559,
                "per_tread": {"1": 0.17}, "sxx": 0.16, "cells": 32,
                "states": 4, "treads": 8, "repeats": 13, "draws": 2000,
                "resampled": 2000, "ladder_slope": 0.70, "ladder_states": 4}
    every_tread = dict(session4, value=1.2082, lo=1.1335, hi=1.2865,
                       fixed_tread=0.7436, per_tile_from2=1.1075,
                       per_tile_treads=8)
    est = CE.fit(CE.plant_rows(eps=0.60, jitter=0.002), draws=50)
    now = dataclasses.asdict(est)
    assert now["claim_min_tread"] == CE.CLAIM_MIN_TREAD
    for key in ("per_tile_all_treads", "per_tile_all_treads_lo",
                "per_tile_all_treads_hi", "tread1_sensitivity_ms",
                "tread1_sensitivity_on_line_ms", "tread1_level_ms",
                "tread1_level_on_line_ms", "fixed_tread"):
        assert now[key] is not None, key
    assert "per_tile_from2" not in now, "the claim is that reading now"
    for name, block in (("s4", session4), ("8d4eb78", every_tread),
                        ("now", now)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"elasticity": block, "gates": []},
                                   default=str))
        got = H.eta(str(path))
        assert got[0] != "unreadable", name
        assert got[:3] == [f"{float(block[k]):.4f}" for k in ("value", "lo", "hi")]


def test_group_m_pins_the_swizzle_and_enters_the_run_id():
    base = CE.default_run_id(parsed(["--dry-run"]))
    assert CE.default_run_id(parsed(["--dry-run", "--group-m", "16"])) == base
    assert CE.default_run_id(parsed(["--dry-run", "--group-m", "1"])) != base
    assert CE.pinned_for(1)["GROUP_SIZE_M"] == 1
    assert CE.pinned_for(1)["BLOCK_SIZE_M"] == CE.PINNED["BLOCK_SIZE_M"]
    assert CE.observed_tile(None, CE.pinned_for(1))["GROUP_SIZE_M"] == 1
    assert CE.observed_tile(None)["GROUP_SIZE_M"] == 16
    got = run(["--dry-run", "--group-m", "1"])
    assert re.search(r"GROUP_SIZE_M=1\b", got.stdout), got.stdout[-800:]
    assert "GROUP_SIZE_M=16" not in got.stdout


def test_the_session_tag_enters_the_run_id_only_when_given():
    bare = CE.default_run_id(parsed(["--dry-run"]))
    assert CE.default_run_id(parsed(["--dry-run", "--session-tag", ""])) == bare
    assert CE.default_run_id(parsed(["--dry-run", "--session-tag", "gaps-x"])) != bare
    got = run(["--dry-run", "--session-tag", "gaps-x"])
    assert "session     gaps-x" in got.stdout


# --------------------------------------------------------------------------
# 12. the self-test's own gates can fail
# --------------------------------------------------------------------------

def test_the_self_test_can_fail(monkeypatch, capsys):
    """A self-test that plants only worlds it passes is a smoke test. Break the
    estimator and S1, S2 and S3 have to go red and the mode has to exit
    INVALID -- not DONE, and not CLAIM_FAIL."""
    monkeypatch.setattr(CE, "within_tread_slope",
                        lambda cells: (-0.99, {}, 1.0, 4))
    rc = CE.self_test(parsed(["--self-test", "--draws", "50"]))
    out = capsys.readouterr().out
    assert rc == exit_codes.INVALID
    assert exit_codes.classify_text(out) == rc
    names = {r.name: r.verdict for r in exit_codes.parse_result_lines(out)}
    assert names["S2"] == exit_codes.FAIL
    assert names["S3"] == exit_codes.FAIL


def test_the_planted_worlds_cover_every_gate_and_both_verdicts():
    """Each gate has to FAIL in at least one world and PASS in at least one, or
    the world list is a list of successes."""
    args = parsed(["--self-test"])
    seen: dict[str, set[str]] = {}
    planted = CE._self_test_args(args)
    threshold, source = CE.registered_clock_ratio(planted)
    for world in CE.self_test_worlds(args):
        # WITH DRAWS: C1 reads an INTERVAL, and at draws=0 there is none, so a
        # coverage test run without them would report C1 as a gate that never
        # passes and hide that it is the design's own resolution being scored.
        est = CE.fit(world.rows, draws=300, seed=0)
        for gate in CE.gates_for(world.rows, planted, threshold, source, est):
            seen.setdefault(gate.token, set()).add(gate.verdict)
    assert set(seen) == {"V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7",
                         "C1", "C2"}
    for token, verdicts in seen.items():
        if token == "V0":
            continue          # V0 is planted empty in its own unit test above
        assert CE.FAIL in verdicts, f"{token} never fails in any planted world"
        assert CE.PASS in verdicts, f"{token} never passes in any planted world"


# --------------------------------------------------------------------------
# 13. the resume, which a 40-minute arm on a rented pod will meet
# --------------------------------------------------------------------------

def test_a_resume_adopts_the_burst_shape_already_on_its_rows(tmp_path):
    """Re-sized from a fresh warm, the second process can land one call either
    side of the rounding, and V2 would then FAIL the whole arm for two burst
    shapes at one tread -- correctly, because the discarded lead would be a
    different share of two halves of one file. The recorded shape is the run's.

    Asserted against `burst_shape` rather than against a remembered number: the
    rounding is the thing that moves, so the test asks the rule."""
    ms = CE.corpus_call_ms(1)
    fresh, _b, _k = CE.burst_shape(ms, CE.DEFAULT_BURST_MS, CE.DEFAULT_TARGET_MS)
    # A cell measured by an earlier process at one call fewer.
    row = CE.plant_rows(eps=0.1, jitter=0.0, treads=1, repeats=1)[0]
    row.calls_per_burst = fresh - 1
    path = tmp_path / "cells.csv"
    CE.append_row(path, row)
    prior = {r.calls_per_burst for r in CE.read_rows(path)
             if r.tiles == 1 and r.status == "ok"}
    assert prior == {fresh - 1}, "the shape has to survive the round trip"
    assert "adopted from the rows already on disk" in inspect.getsource(CE.run_arm)


def test_the_resume_key_is_the_cell_and_not_the_row_count():
    """Skipping N rows because N are on disk would resume the wrong cells the
    moment one in the middle failed. The key is (repeat, duty, tread)."""
    body = inspect.getsource(CE.run_arm)
    assert "(r.repeat, _duty_key(r.duty_requested), r.tiles)" in body
    assert "if (repeat, _duty_key(duty), tread) in have:" in body


def test_a_resumed_arm_scores_the_whole_file_and_not_only_what_it_measured():
    """A resume that scored only its own cells would report a design four states
    wide as one, and V3 would refuse an arm that is in fact complete."""
    assert "return read_rows(csv_path) if have else rows" in \
        inspect.getsource(CE.run_arm)


# --------------------------------------------------------------------------
# 13b. the resume guard is the card's UUID, not its name
# --------------------------------------------------------------------------

def test_the_device_guard_refuses_a_second_card_and_resumes_the_first(tmp_path):
    """`private_weight_reference.device_guard`'s rule, one arm over: every H200
    is 'NVIDIA H200', so the CARD stamp alone let a replacement pod's card fill
    the holes in the first pod's ladder."""
    out = tmp_path / "run"
    assert CE.device_guard(out, "GPU-aaaa") == ""
    assert (out / CE.DEVICE_FILE).read_text().strip() == "GPU-aaaa"
    (out / "cells.csv").write_text("x\n")
    assert CE.device_guard(out, "GPU-aaaa") == ""
    refused = CE.device_guard(out, "GPU-bbbb")
    assert "GPU-aaaa" in refused and "GPU-bbbb" in refused
    assert "--session-tag" in refused and "--run-id" in refused, (
        "the refusal names the fix")
    assert CE.device_guard(out, "") != "", "an unreadable UUID proves nothing"


def test_the_device_guard_accepts_a_no_uuid_identity_only_against_itself(tmp_path):
    out = tmp_path / "run"
    weak = CE.NO_UUID_PREFIX + "NVIDIA H200"
    assert CE.device_guard(out, weak) == ""
    (out / "cells.csv").write_text("x\n")
    assert CE.device_guard(out, weak) == ""
    assert CE.device_guard(out, "GPU-aaaa") != ""


def test_an_old_card_stamp_is_refused_with_cells_and_upgraded_without(tmp_path):
    """THE DECISION FOR A STAMP FROM BEFORE 2026-09-22, which names the card and
    carries no UUID: with cells beside it, the card they came from cannot be
    shown to be this one, so it is REFUSED (the ratio arm's rule for cells with
    no DEVICE file); with no cells there is nothing to pool, so it is stamped
    and measured."""
    old = tmp_path / "old"
    old.mkdir()
    (old / "CARD").write_text("NVIDIA H200\n")
    (old / "cells.csv").write_text("x\n")
    refused = CE.device_guard(old, "GPU-aaaa")
    assert "no DEVICE file" in refused and "--session-tag" in refused
    assert not (old / CE.DEVICE_FILE).exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "CARD").write_text("NVIDIA H200\n")
    assert CE.device_guard(empty, "GPU-aaaa") == ""
    assert (empty / CE.DEVICE_FILE).read_text().strip() == "GPU-aaaa"


class _FakeCuda:
    def __init__(self, uuid=None):
        self._uuid = uuid

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def get_device_name(self, index):
        return "NVIDIA H200"

    def get_device_properties(self, index):
        if self._uuid is None:
            raise RuntimeError("this torch exposes no uuid")
        return type("Props", (), {"uuid": self._uuid})()


def test_the_identity_is_the_uuid_and_names_itself_weaker_without_one(monkeypatch):
    fake = type("Torch", (), {})()
    fake.cuda = _FakeCuda("GPU-6b4b5fe6")
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert CE.device_identity() == "GPU-6b4b5fe6"
    fake.cuda = _FakeCuda(None)
    assert CE.device_identity() == CE.NO_UUID_PREFIX + "NVIDIA H200"


def _measuring_run(monkeypatch, out, *, identity, rows):
    """`main` down to the measurement with every GPU door planted open and
    `run_arm` replaced, so the guard is exercised on the path a pod takes."""
    calls = []
    monkeypatch.setattr(CE, "resolve_card", lambda args: "NVIDIA H200")
    monkeypatch.setattr(CE.SWEEP, "missing_gpu_stack", lambda: "")
    monkeypatch.setattr(CE.T, "require_cuda", lambda: None)
    monkeypatch.setattr(CE.T, "nvml_clock_reader", lambda *a, **k: None)
    # raising=False so the parent commit, which has no UUID guard, runs this
    # same path and shows what it did: it measured into the other card's ladder.
    monkeypatch.setattr(CE, "device_identity", lambda: identity, raising=False)

    def fake_run_arm(*args, **kwargs):
        calls.append(args)
        return rows
    monkeypatch.setattr(CE, "run_arm", fake_run_arm)
    rc = CE.main(["--out", str(out), "--run-id", "r", "--draws", "50"])
    return rc, calls


def test_a_resume_on_another_card_of_the_same_name_is_refused(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "clock_elasticity" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "CARD").write_text("NVIDIA H200\n")
    (run_dir / "DEVICE").write_text("GPU-aaaa\n")
    (run_dir / "cells.csv").write_text("x\n")
    rc, calls = _measuring_run(monkeypatch, tmp_path, identity="GPU-bbbb",
                               rows=[])
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED, out[-800:]
    assert not calls, "the replacement card measured into the first card's ladder"
    assert "GPU-aaaa" in out and "GPU-bbbb" in out and "--run-id" in out


def test_a_resume_into_an_old_stamp_with_cells_is_refused(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "clock_elasticity" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "CARD").write_text("NVIDIA H200\n")
    (run_dir / "cells.csv").write_text("x\n")
    rc, calls = _measuring_run(monkeypatch, tmp_path, identity="GPU-aaaa",
                               rows=[])
    out = capsys.readouterr().out
    assert rc == exit_codes.REFUSED, out[-800:]
    assert not calls
    assert "no DEVICE file" in out


def test_a_resume_on_the_same_card_measures_and_the_report_names_it(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "clock_elasticity" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "CARD").write_text("NVIDIA H200\n")
    rows = CE.plant_rows(eps=0.05, jitter=0.004)
    rc, calls = _measuring_run(monkeypatch, tmp_path, identity="GPU-aaaa",
                               rows=rows)
    out = capsys.readouterr().out
    assert calls, out[-800:]
    assert rc == exit_codes.DONE, out[-1500:]
    assert CE.DEVICE_FILE == "DEVICE", "the ratio arm's name for the same file"
    assert (run_dir / "DEVICE").read_text().strip() == "GPU-aaaa"
    payload = json.loads((run_dir / "report.json").read_text())
    assert payload["device"] == "GPU-aaaa"
    assert payload["elasticity"]["claim_min_tread"] == CE.CLAIM_MIN_TREAD
    # And the second run on the same card resumes rather than refusing.
    (run_dir / "cells.csv").write_text("x\n")
    rc, calls = _measuring_run(monkeypatch, tmp_path, identity="GPU-aaaa",
                               rows=rows)
    assert rc == exit_codes.DONE and len(calls) == 1


# --------------------------------------------------------------------------
# 13c. the plan page says what V1 is sized for
# --------------------------------------------------------------------------

def test_the_plan_prints_the_claims_own_resolution_at_the_design_requested():
    """V1's threshold comes from the pooled per-call estimator's standard
    error, and the gated claim fits one slope per tread over treads 2 and
    deeper, so at V1's span its interval is several times wider. The page says
    so for the design actually requested, the chain's three states included,
    and V1 does not move."""
    chain = CHAIN_R1 + ["--duty", "1.0", "0.7", "0.5"]
    args = parsed(chain)
    threshold, _source = CE.registered_clock_ratio(args)
    factor = CE.per_tile_se_factor(8, min_tread=CE.CLAIM_MIN_TREAD)
    assert factor > 1.0
    assert CE.per_tile_se_factor(8, min_tread=1) < factor, (
        "dropping the shallowest tread costs leverage")
    need = CE.required_clock_ratio(repeats=13, treads=8, states=3,
                                   se_factor=factor)
    assert need > threshold, "the claim needs more span than V1 asks for"
    assert CE.required_clock_ratio(repeats=13, treads=8, states=3) == \
        CE.required_clock_ratio(repeats=13, treads=8, states=3, se_factor=1.0)
    page = run(["--dry-run", *chain]).stdout
    assert f"V1 THRESHOLD {threshold:.3f}x" in page
    assert "V1'S THRESHOLD IS SIZED FOR THAT ESTIMATOR AND NOT FOR THE GATED ONE" in page
    assert "(13 repeats x 8 treads x 3 states)" in page
    assert f"a span of {need:.4f}x at every tread" in page
    assert f"its interval is {factor:.2f}x wider" in page
    # A ladder too shallow to hold two treads from tread 2 says so, rather
    # than printing a span of inf.
    assert CE.per_tile_se_factor(2, min_tread=CE.CLAIM_MIN_TREAD) == float("inf")
    shallow = "\n".join(CE.mde_lines(parsed(["--treads", "2"])))
    assert "cannot be formed at all" in shallow and "inf" not in shallow


def test_the_claims_predicted_resolution_is_what_the_estimator_delivers():
    """The printed factor is arithmetic, so it is checked against the
    estimator: a planted 3-state design at V1's own span, with the corpus
    spread as its jitter, bootstraps to a claim interval near the predicted
    one and several times the pooled one."""
    states, repeats = 3, 13
    need = CE.required_clock_ratio(repeats=repeats, treads=8, states=states)
    span = math.log(need)
    mhz = tuple(1500.0 * math.exp(span * i / (states - 1)) for i in range(states))
    xs = list(range(1, 9))
    b = CE._line_slope(xs, list(CE.CORPUS_LADDER_MS))
    a = statistics.fmean(CE.CORPUS_LADDER_MS) - b * statistics.fmean(xs)
    claim, pooled = [], []
    for seed in range(4):
        est = CE.fit(CE.plant_rows(eps=0.5, duties=CE.DUTY_LEVELS[:states],
                                   mhz=mhz, repeats=repeats, a_ms=a, b_ms=b,
                                   jitter=CE.CORPUS_REPEAT_SPREAD, seed=seed),
                     draws=400, seed=1)
        claim.append(est.half_width)
        pooled.append((est.fixed_tread_hi - est.fixed_tread_lo) / 2.0)
    factor = CE.per_tile_se_factor(8, min_tread=CE.CLAIM_MIN_TREAD)
    seen = statistics.median(claim) / statistics.median(pooled)
    assert 0.75 * factor < seen < 1.33 * factor, (seen, factor)
    predicted = CE.RESOLUTION_TARGET * factor
    assert 0.75 * predicted < statistics.median(claim) < 1.5 * predicted


# --------------------------------------------------------------------------
# 14. the booking in the session driver is this plan's own figure
# --------------------------------------------------------------------------

DRIVER = ROOT / "scripts" / "h200_gaps_session.sh"
ARM = "elasticity-m32-n64-g16"


def _lift(script):
    body = ('set -uo pipefail\n'
            f'REPO={str(ROOT)!r}\n'
            f'eval "$(sed -n \'/^# >>> LIFTABLE/,/^# <<< LIFTABLE/p\' "{DRIVER}")"\n'
            f'{script}\n')
    return subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                          timeout=120).stdout


def test_the_booked_minutes_are_above_this_plan_and_never_at_it():
    """"Book above it and never at it" needs the figure it is above to be the
    one this plan prints, with the flags the arm line passes. A booking read off
    a different command is a number somebody made up, which is the state the
    whole cost table was in before it named its commands."""
    words = _lift(f"arm_basis {ARM}").strip()
    assert words, "the arm has no basis"
    quoted = re.search(r"'estimated wall time (\d+) s \(([\d.]+) min\)'", words)
    assert quoted, words[:400]
    plan = run(["--dry-run", "--model", "mixtral-8x7b", "--dtype", "bf16",
                "--group-m", "16",
                "--treads", "8", "--duty", "1.0", "0.5", "0.25", "0.1",
                "--repeats", "13", "--burst-ms", "40", "--target-ms", "200",
                "--trials", "3", "--warm-ms", "200", "--settle-seconds", "10"])
    assert f"estimated wall time {quoted.group(1)} s " in plan.stdout
    assert f"({quoted.group(2)} min)" in plan.stdout
    booked = int(_lift(f"arm_minutes {ARM}").strip())
    assert booked > float(quoted.group(2)), (booked, quoted.group(2))


def test_the_booking_is_on_the_wall_clock_and_the_page_charges_everything():
    """WALL means the plan charged every term, including the compiles. The
    driver derives the word from `arm_unpriced` and the session's own test
    re-derives it a third way, from what the plan PRINTS, so this asserts the
    thing both of them read: the page carries no kernel-time disclaimer."""
    assert _lift(f"arm_unpriced {ARM}").strip() == ""
    assert _lift(f"arm_clock {ARM}").strip() == "WALL"
    plan = run(["--dry-run"]).stdout.lower()
    assert "excluding compiles and allocation" not in plan
    assert "not the wall clock" not in plan
    assert "weight build and first compile" in plan


def test_the_arm_line_carries_every_flag_that_moves_the_plan():
    """A dry branch missing --duty or --repeats previews a different sweep, a
    different V1 threshold and a different run id. Read out of the driver rather
    than listed here, so a list cannot go stale against it."""
    code = DRIVER.read_text()
    joined = re.sub(r"\\\n\s+", " ", code)
    lines = [ln for ln in joined.splitlines()
             if re.match(rf"\s*arm {re.escape(ARM)}\s", ln)]
    assert len(lines) == 2, lines
    for line in lines:
        for flag in ("--model", "--dtype", "--treads", "--duty", "--repeats",
                     "--burst-ms", "--target-ms", "--trials", "--warm-ms",
                     "--settle-seconds"):
            assert flag in line, (flag, line)
        assert "--publish" not in line
    dry = [ln for ln in lines if "--dry-run" in ln]
    assert len(dry) == 1
    # And the measuring branch runs the interpreter that has vLLM in it.
    measuring = [ln for ln in lines if "--dry-run" not in ln][0]
    assert "$PY_VLLM" in measuring, measuring
