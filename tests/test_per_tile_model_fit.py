"""The per-M-tile model fit: ADDITIVE against OVERLAP on planted cells.

What the script is for decides what these tests hold it to. It exists to say
which of two model families describes the R1 cells and how much of alpha(G) the
cells can see, so the three things that can make its page wrong are:

    the FITTER     -- a local minimum reported as the fit, or a containing model
                      reported worse than the model it contains
    the WORLDS     -- an overlap world must come back overlap with its planted
                      parameters, an additive world additive, and the scan must
                      be flat exactly where the traffic branch is hidden
    the NUMBERS    -- a parameter or a bandwidth that the fitted cells leave
                      free (an invisible direction) must print as not
                      identified, never as where the fitter stopped
    the INPUTS     -- the arm's own kept rows, INVALID pages out by default, a
                      gate spelled outside exit_codes' table refused, and the
                      pin rate read from the session's ruler, never typed

Every planted number here is a test design choice and not a calibration; the
pin rate a test asserts against is the one its own planted ruler carries.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _hermetic import laptop_env  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.bench.weights import routed_expert_weight_bytes  # noqa: E402

SCRIPT = ROOT / "scripts" / "per_tile_model_fit.py"


def _load():
    spec = importlib.util.spec_from_file_location("per_tile_model_fit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


M = _load()
CE = M.CE
BY_KEY = {s.key: s for s in M.STRUCTURES}

MODEL, DTYPE = "mixtral-8x7b", "bf16"
W = routed_expert_weight_bytes(MODEL, DTYPE)
#: A planted card, deliberately not the H200's numbers, and a planted F_REF.
PIN_GBPS, READ_GBPS, TRIAD_GBPS, F_REF = 4000.0, 3800.0, 3600.0, 2000.0
CTX = M.Context(weight_bytes=W, experts=8, tau_pin_ms=M.tau_ms_at(W, PIN_GBPS),
                f_ref=F_REF)
GROUPS = (1, 4, 16, 64)
TREADS = range(1, 9)
#: Three duty states; the capped one climbs along its treads the way a
#: power-capped card's does, the other two sit near the top.
STATES = (lambda n: 1450.0 + 40.0 * n, 1880.0, 1960.0)
DUTIES = (1.0, 0.5, 0.25)

OVERLAP_WORLD = {"T0": 0.08, "tau": 0.80, "c0": 0.52, "e": 1.1}
ADDITIVE_WORLD = {"T0": 0.15, "tau": 0.90, "c0": 0.25, "e": 1.0,
                  "alpha(4)": 0.6, "alpha(16)": 0.5, "alpha(64)": 0.4}
NOISE = 0.002


def overlap_cells(seed=1, noise=NOISE):
    return M.planted_cells(BY_KEY["ovl.group"], OVERLAP_WORLD, CTX, groups=GROUPS,
                           treads=TREADS, states=STATES, noise=noise, seed=seed)


def additive_cells(seed=2, world=ADDITIVE_WORLD):
    return M.planted_cells(BY_KEY["add.split.pin"], world, CTX, groups=GROUPS,
                           treads=TREADS, states=STATES, noise=NOISE, seed=seed)


def _best(fits, family):
    return min(f.rms for f in fits if f.structure.family == family)


# --------------------------------------------------------------------------
# 1. the traffic count and the formulas, by hand
# --------------------------------------------------------------------------

@pytest.mark.parametrize("g,n,expect", [
    (1, 3, 3.0),          # no reuse: every M-tile streams its expert again
    (4, 1, 1.0),          # one tile per expert: each expert once
    (4, 2, 1.0),          # groups {0,0,1,1}: each expert once
    (4, 3, 1.5),          # {0,0,0,1} {1,1,2,2} {2,3,3,3} ...: 12 reads of 8
    (16, 3, 1.125),       # {0..4,5} then {5,6,7}: 9 reads of 8
    (64, 8, 1.0),         # one group holds every expert
])
def test_group_traffic_counts_the_distinct_experts_of_each_group(g, n, expect):
    assert M.group_traffic(g, n, 8) == pytest.approx(expect)


def test_every_structure_evaluates_its_printed_formula():
    """One cell per G, each structure's prediction against the arithmetic its
    `formula()` prints, done by hand here."""
    cells = [M.Cell(page="p", arm="R1", state="0", group_m=g, tread=3, mhz=1600.0, ms=1.0)
             for g in (1, 4)]
    data = M.Data.of(cells, CTX)
    t0, tau, c0, e, a1, a4 = 0.1, 0.7, 0.5, 1.2, 0.9, 0.3
    phi = (CTX.f_ref / 1600.0) ** e
    d_group = [M.group_traffic(1, 3, 8), M.group_traffic(4, 3, 8)]
    for s in M.STRUCTURES:
        lay = M.layout(s, data, CTX)
        vals = {"T0": t0, "tau": max(tau, lay.lb[1]), "c0": c0, "e": e,
                "alpha(1)": a1, "alpha(4)": a4}
        p = np.array([vals[n] for n in lay.names])
        t = vals["tau"]
        got = M.predict(s, lay, p, data, CTX)
        alpha = [a1 if s.alpha_one == "free" else 1.0, a4]
        for i in range(2):
            if s.traffic == "group":
                d = d_group[i]
            elif s.traffic == "per_tile":
                d = alpha[i] * 3
            else:
                d = 1 + alpha[i] * 2
            if s.combine == "max":
                want = t0 + max(d * t, 3 * c0 * phi)
            elif s.phi == "all":
                want = t0 + (d * t + 3 * c0) * phi
            else:
                want = t0 + d * t + 3 * c0 * phi
            assert got[i] == pytest.approx(want), (s.key, i)


def test_a_private_cell_reads_one_weight_set_per_tile_at_every_g():
    cells = [M.Cell(page="p", arm="R3 private", state="0.25", group_m=64, tread=5,
                    mhz=1960.0, ms=1.0, private=True)]
    data = M.Data.of(cells, CTX)
    s = BY_KEY["ovl.group"]
    lay = M.layout(s, data, CTX)
    got = M.predict(s, lay, np.array([0.0, 0.8, 0.01, 1.0]), data, CTX)
    assert got[0] == pytest.approx(5 * 0.8)


# --------------------------------------------------------------------------
# 2. the fitter
# --------------------------------------------------------------------------

def test_bounded_lsq_finds_an_interior_minimum_and_stops_at_an_active_bound():
    target = np.array([0.3, -0.2])

    def fun(p):
        return np.array([p[0] - target[0], p[1] - target[1], 0.5 * (p[0] - target[0])])

    p, _r, cost = M.bounded_lsq(fun, np.array([2.0, 2.0]),
                                np.array([-1.0, 0.0]), np.array([1.0, 1.0]))
    assert p[0] == pytest.approx(0.3, abs=1e-8)
    assert p[1] == pytest.approx(0.0, abs=1e-12)      # the bound, not -0.2
    assert cost == pytest.approx(0.04, rel=1e-6)


def _inversions(fits) -> list[str]:
    """Containing models whose SSR ended above a model they contain: tau free
    contains tau >= W/pin, alpha(1) free contains alpha(1) = 1."""
    by = {f.structure.key: f for f in fits}
    bad = [k for k in by if k.endswith(".pin") and by[k[:-4]].ssr > by[k].ssr * (1 + 1e-9)]
    for wide, narrow in (("ovl.first.a1", "ovl.first"), ("ovl.first.a1.pin", "ovl.first.pin")):
        if by[wide].ssr > by[narrow].ssr * (1 + 1e-9):
            bad.append(wide)
    return bad


def test_a_containing_model_never_ends_above_the_model_it_contains(monkeypatch):
    """On session 5's cells a single fit of ovl.first.a1 (tau free) stopped 0.9%
    of SSR above its own .pin variant. With the polish off, this world does the
    same to independent fits at two starts; `fit_all` restarts each structure
    from every solution that is a point of it, so the order cannot invert."""
    monkeypatch.setattr(M, "POLISH_ROUNDS", 0)
    world = dict(OVERLAP_WORLD, **{f"alpha({g})": 0.35 for g in GROUPS if g != 1})
    cells = M.planted_cells(BY_KEY["ovl.first.pin"], world, CTX, groups=GROUPS,
                            treads=TREADS, states=STATES, noise=0.004, seed=10)
    alone = [M.fit(s, cells, CTX, starts=2) for s in M.STRUCTURES]
    assert _inversions(alone), "the premise: independent fits invert on this world"
    assert _inversions(M.fit_all(cells, CTX, starts=2)) == []
    monkeypatch.undo()
    assert _inversions(M.fit_all(overlap_cells(seed=5), CTX)) == []


def test_the_polish_leaves_a_point_where_the_max_kink_stalled_the_fitter(monkeypatch):
    """A forward-difference Jacobian taken on one side of the max() kink can
    stop Levenberg-Marquardt short: on session 5's cells one scan point sat
    0.19% of SSR above the minimum. Here, from one start and no polish, the
    fitter stops 50% above; the kicks reach what four starts reach."""
    world = {"T0": 0.07, "tau": 0.706, "c0": 0.495, "e": 1.03,
             **{f"alpha({g})": 0.47 for g in GROUPS if g != 1}}
    states = (STATES[0], 1900.0, 1960.0)
    cells = M.planted_cells(BY_KEY["ovl.first.pin"], world, CTX, groups=GROUPS,
                            treads=TREADS, states=states, noise=0.006, seed=45)
    scan_col = next(s for s in M.SCAN_STRUCTURES
                    if s.combine == "max" and s.alpha_one == "free")
    polished = M.fit(scan_col, cells, CTX, fixed_rest=0.4, starts=1)
    many = M.fit(scan_col, cells, CTX, fixed_rest=0.4, starts=4)
    monkeypatch.setattr(M, "POLISH_ROUNDS", 0)
    stalled = M.fit(scan_col, cells, CTX, fixed_rest=0.4, starts=1)
    assert stalled.ssr > 1.2 * polished.ssr
    assert polished.ssr == pytest.approx(many.ssr, rel=1e-9)


def test_transfer_refuses_a_point_outside_the_target_bounds():
    cells = overlap_cells()
    free = M.fit(BY_KEY["add.split"], cells, CTX)
    data = M.Data.of(cells, CTX)
    pin_layout = M.layout(BY_KEY["add.split.pin"], data, CTX)
    assert free.value("tau") < CTX.tau_pin_ms          # this world's free fit
    assert M.transfer(free, pin_layout) is None
    pinned = M.fit(BY_KEY["add.split.pin"], cells, CTX)
    assert M.transfer(pinned, M.layout(BY_KEY["add.split"], data, CTX)) is not None


# --------------------------------------------------------------------------
# 3. the planted worlds
# --------------------------------------------------------------------------

def test_an_overlap_world_is_recovered_and_preferred():
    fits = {f.structure.key: f for f in M.fit_all(overlap_cells(), CTX)}
    got = fits["ovl.group"]
    for name, planted in OVERLAP_WORLD.items():
        assert got.value(name) == pytest.approx(planted, rel=0.02), name
    assert got.rms < 2 * NOISE
    assert got.rank == got.movable == 4
    assert not got.blind and got.bandwidth_identified and all(got.identified)
    assert _best(fits.values(), M.OVERLAP) < _best(fits.values(), M.ADDITIVE) / 5


def test_an_additive_world_is_preferred_as_additive():
    fits = {f.structure.key: f for f in M.fit_all(additive_cells(), CTX)}
    assert fits["add.split.pin"].rms < 2 * NOISE
    assert _best(fits.values(), M.ADDITIVE) < _best(fits.values(), M.OVERLAP) / 5


def test_planting_outside_a_structures_bounds_is_refused():
    below_pin = dict(ADDITIVE_WORLD, tau=0.5 * CTX.tau_pin_ms)
    with pytest.raises(ValueError, match="outside"):
        additive_cells(world=below_pin)


def test_the_rank_flags_a_parameter_combination_the_cells_cannot_see():
    """On one G the paper form sees only alpha(G) tau, never tau alone."""
    one_g = [c for c in overlap_cells() if c.group_m == 16]
    f = M.fit(BY_KEY["add.paper"], one_g, CTX)
    assert f.rank < f.movable and f.blind
    assert not f.bandwidth_identified and not f.is_identified("alpha(16)")


#: A world on the T0-tau valley of the compulsory-first-read form: T0 + d,
#: tau - d and every alpha(G) x tau / (tau - d) give the same cells, for as
#: long as the largest alpha stays <= 1. T0 = 0 is the valley's lower end.
VALLEY_WORLD = {"T0": 0.0, "tau": 0.5, "c0": 0.4, "e": 0.8, "alpha(1)": 0.6,
                "alpha(4)": 0.3, "alpha(16)": 0.2, "alpha(64)": 0.1}


def _fit_at(monkeypatch, s, cells, world):
    """`fit` started at `world` alone: on an exact valley the fitter stops
    wherever its start meets the valley, so a test that needs the end of one
    has to start there."""
    data = M.Data.of(cells, CTX)
    lay = M.layout(s, data, CTX)
    start = np.array([world[n] for n in lay.names])
    monkeypatch.setattr(M, "_start", lambda *_a, **_k: start.copy())
    return M.fit(s, cells, CTX, starts=1)


def test_a_valley_that_ends_on_a_bound_is_flagged_and_its_bandwidth_is_not_printed(
        monkeypatch):
    """Session 5 with --include-invalid: add.first stopped at T0 = 0 on its
    lower bound and printed rank 6/6, no !, and BW 14058 GB/s, while T0 += d,
    tau -= d, alpha(G) x tau / (tau - d) left the SSR unchanged out to 18728.
    The rank counted only the parameters off their bounds, and the valley
    leaves through T0's."""
    s = BY_KEY["add.first"]
    cells = M.planted_cells(s, VALLEY_WORLD, CTX, groups=GROUPS, treads=TREADS,
                            states=STATES)
    f = _fit_at(monkeypatch, s, cells, VALLEY_WORLD)
    assert f.value("T0") == 0.0 and f.at_lower[0], "the premise: the fit ends on T0's bound"
    assert f.ssr < 1e-20
    text = "\n".join(M.fit_lines([f], CTX, M.Ruler("planted", PIN_GBPS, {}, "triad"),
                                 list(GROUPS), cells))
    row = next(line for line in text.splitlines() if line.strip().startswith("ADDITIVE"))
    assert "!" in row.split()[5]                       # the rank column
    assert "not identified" in row
    assert "GB/s" not in row and "x pin" not in row
    params = next(line for line in text.splitlines() if line.strip().startswith("add.first "))
    for name in ("T0", "tau", "alpha(1)"):
        assert re.search(rf"{re.escape(name)} [0-9.]+ \[[^]]*not identified\]", params), name
    assert "c0 0.4000," in params and "e 0.8000," in params    # off the valley
    assert M._fit_json(f)["bandwidth_gbps"] is None


def test_a_valley_blocked_at_both_ends_is_not_flagged(monkeypatch):
    """The same valley with alpha(1) = 1 at T0 = 0: moving along it takes T0
    below 0 one way and alpha(1) above 1 the other, so nothing moves and the
    fit is determined although the Jacobian's rank is short."""
    s = BY_KEY["add.first"]
    world = dict(VALLEY_WORLD, **{"alpha(1)": 1.0})
    cells = M.planted_cells(s, world, CTX, groups=GROUPS, treads=TREADS, states=STATES)
    f = _fit_at(monkeypatch, s, cells, world)
    assert f.at_lower[0] and f.at_upper[f.names.index("alpha(1)")]
    assert f.rank < f.movable
    assert not f.blind and f.bandwidth_identified and all(f.identified)


# --------------------------------------------------------------------------
# 4. the identifiability scan and leave-one-G-out
# --------------------------------------------------------------------------

HIDDEN_ALPHA = 0.2
TIED_ALPHA = 0.5


def _scan(cells):
    cols = M.scan_all(cells, CTX, M.scan_grid(0.05), [], M.FLAT_TOL)
    return {c.structure.key: c for c in cols}


def test_the_scan_is_flat_where_the_traffic_branch_is_under_the_floor():
    """G>1 traffic (1 + alpha(n-1)) tau sits under n c0 Phi at every tread of
    this world for any alpha up to about 0.3, so every such alpha fits the
    same: the window starts at 0 and runs past the planted value."""
    world = dict(OVERLAP_WORLD, **{f"alpha({g})": HIDDEN_ALPHA for g in GROUPS if g != 1})
    cells = M.planted_cells(BY_KEY["ovl.first.pin"], world, CTX, groups=GROUPS,
                            treads=TREADS, states=STATES, noise=NOISE, seed=3)
    col = _scan(cells)["ovl a1=1"]
    lo, hi = col.window
    assert lo == 0.0 and col.contiguous
    assert hi >= HIDDEN_ALPHA + 0.05
    assert hi - lo >= 0.25


def test_the_scan_is_not_flat_where_alpha_is_identified():
    """The same scan on an additive world with a tied alpha: a window around
    the planted value, narrower than the floor above and bounded away from 0.
    A scan that were flat everywhere would pass the test above and fail this."""
    world = dict(ADDITIVE_WORLD, **{f"alpha({g})": TIED_ALPHA for g in GROUPS if g != 1})
    cols = _scan(additive_cells(world=world))
    lo, hi = cols["add a1=1"].window
    assert lo <= TIED_ALPHA <= hi
    assert hi - lo < 0.25
    assert lo >= TIED_ALPHA - 0.1
    assert cols["ovl a1=1"].window[0] > 0.0


def test_the_scan_marks_an_alpha_1_the_cells_cannot_see():
    """The scan prints the fitted alpha(1) beside each held alpha(G>1), a
    parameter value like any other: where the floor n c0 Phi covers G=1's
    traffic n tau at every cell, alpha(1) is flat and the printed value is
    where the fitter stopped."""
    world = dict(OVERLAP_WORLD, c0=1.0)
    assert world["c0"] * min(CTX.f_ref / f for f in (STATES[0](8), *STATES[1:])) > world["tau"]
    cells = M.planted_cells(BY_KEY["ovl.group.pin"], world, CTX, groups=GROUPS,
                            treads=TREADS, states=STATES, noise=NOISE, seed=4)
    cols = M.scan_all(cells, CTX, M.scan_grid(0.25), [], M.FLAT_TOL)
    lines = M.scan_lines(cols, M.FLAT_TOL, {})
    rows = [line for line in lines if re.match(r"\s+\d\.\d{3}\s", line)]
    assert len(rows) == len(M.scan_grid(0.25))
    for line in rows:
        # ovl a1 free's alpha(1) is marked; add a1 free's is seen, and is not.
        assert len(re.findall(r"a1 \d\.\d{3}\?", line)) == 1, line
        assert len(re.findall(r"a1 \d\.\d{3}", line)) == 2, line


def test_leave_one_g_out_predicts_a_held_out_g_on_an_overlap_world():
    per = M.leave_one_g_out(BY_KEY["ovl.group.pin"], overlap_cells(), CTX)
    assert set(per) == set(GROUPS)
    for g in GROUPS:
        rms, worst = per[g]
        assert rms < 3 * NOISE, g
        assert worst >= rms


def test_leave_one_g_out_of_a_per_g_model_ties_alpha_and_predicts():
    per = M.leave_one_g_out(BY_KEY["add.split.pin"], additive_cells(), CTX)
    assert all(per[g] is not None for g in GROUPS if g != 1)


# --------------------------------------------------------------------------
# 5. the inputs: pages, labels, kept rows and the ruler, on disk
# --------------------------------------------------------------------------

def _ruler(path: Path, pin=PIN_GBPS):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "name": "planted card", "verified": True,
        "memory": {"bandwidth_tb_s": TRIAD_GBPS / 1000.0},
        "compute_dense_tflops": {"bf16": 500.0},
        "detail": {"gpu_name": "planted", "ceiling_pattern": "triad",
                   "bandwidth_patterns": [{"pattern": "read_stream", "gbps": READ_GBPS},
                                          {"pattern": "triad", "gbps": TRIAD_GBPS}]},
        "observed": {"pin_rate_gbps": pin}}))
    return path


def _gates(v7="PASS"):
    gates = [{"kind": "VALIDITY", "number": str(i), "verdict": "PASS"} for i in range(7)]
    gates.append({"kind": "VALIDITY", "number": "7", "verdict": v7})
    gates.append({"kind": "CLAIM", "number": "1", "verdict": "FAIL"})
    return {"gates": gates}


_TEMPLATE = CE.plant_rows(eps=1.0, duties=(1.0,), mhz=(1500.0,), treads=1, repeats=1)[0]


def _r1_page(path: Path, group_m: int, *, report=None, repeats=3, drift_ms=None):
    """A clock_elasticity page whose cells are the overlap world at `group_m`.
    `drift_ms` adds one DRIFTING row at tread 2 of state 0 with that time."""
    cells = [c for c in overlap_cells(noise=0.0) if c.group_m == group_m]
    for rep in range(repeats):
        for c in cells:
            i = int(c.state)
            CE.append_row(path / "cells.csv", dataclasses.replace(
                _TEMPLATE, duty_requested=DUTIES[i], duty_achieved=DUTIES[i],
                state_index=i, repeat=rep, group_m=group_m, tiles=c.tread,
                rows_per_expert=32 * c.tread, ms_p50=c.ms, sm_clock_load_mhz=c.mhz,
                sm_clock_start_mhz=c.mhz, sm_clock_end_mhz=c.mhz, clock_drift_ok=True))
    if drift_ms is not None:
        c = next(c for c in cells if c.state == "0" and c.tread == 2)
        CE.append_row(path / "cells.csv", dataclasses.replace(
            _TEMPLATE, duty_requested=DUTIES[0], state_index=0, repeat=repeats,
            group_m=group_m, tiles=2, rows_per_expert=64, ms_p50=drift_ms,
            sm_clock_load_mhz=c.mhz, clock_drift_ok=False))
    if report is not None:
        (path / "report.json").write_text(json.dumps(report))
    return path


@pytest.fixture
def session(tmp_path):
    """A published-session layout: the ruler above, one page per G, the G=16
    page INVALID on V7 and one page with no report.json."""
    root = tmp_path / "2026-09-99-planted-session"
    _ruler(root / "calibration" / "measured_planted.yaml")
    arm = root / "results" / "gaps-planted" / "clock_elasticity"
    for g in (1, 4, 64):
        _r1_page(arm / f"planted-g{g}-seed0-{g:08x}", g, report=_gates())
    _r1_page(arm / "planted-g16-seed0-0000abcd", 16, report=_gates(v7="FAIL"))
    _r1_page(arm / "planted-g4-seed0-0000ffff", 4)
    return root


def _run(argv):
    lines, doc = M.run(M.build_parser().parse_args([str(a) for a in argv]))
    return "\n".join(lines), doc


def test_invalid_pages_are_listed_and_excluded_by_default(session):
    text, doc = _run([session, "--no-scan", "--no-logo"])
    assert "NOT VALID" not in text
    pages = {p["run"]: p for p in doc["pages"]}
    assert pages["0000abcd"]["label"] == M.INVALID
    assert pages["0000abcd"]["failed"] == ["V7"]
    assert not pages["0000abcd"]["fitted"]
    assert pages["0000ffff"]["label"] == M.UNSCORED and not pages["0000ffff"]["fitted"]
    assert {p["group_m"] for p in doc["pages"] if p["fitted"]} == {1, 4, 64}
    assert "INVALID (V7 not PASS)" in text and "--include-invalid to fit" in text
    assert {c["group_m"] for c in doc["cells"]} == {1, 4, 64}


def test_include_invalid_fits_the_invalid_page_and_still_labels_it(session):
    text, doc = _run([session, "--no-scan", "--no-logo", "--include-invalid"])
    assert "FITTED PAGES THAT ARE NOT VALID: 0000abcd (INVALID: V7)" in text
    pages = {p["run"]: p for p in doc["pages"]}
    assert pages["0000abcd"]["fitted"] and pages["0000abcd"]["label"] == M.INVALID
    assert {c["group_m"] for c in doc["cells"]} == {1, 4, 16, 64}
    assert not pages["0000ffff"]["fitted"]
    _text, doc = _run([session, "--no-scan", "--no-logo", "--include-unscored"])
    assert {p["run"]: p for p in doc["pages"]}["0000ffff"]["fitted"]


def test_r1_cells_are_the_arms_own_collapse_of_its_own_kept_rows(tmp_path):
    page = _r1_page(tmp_path / "run-00001234", 4, report=_gates(), drift_ms=50.0)
    loaded = M.load_r1(page)
    rows = CE.read_rows(page / "cells.csv")
    assert any(CE.exclusion(r) == CE.DROP_DRIFT for r in rows)
    want = {(t, d): (ms, mhz) for (t, d), (ms, mhz, _n) in CE.collapse(rows).items()}
    got = {(c.tread, c.state): (c.ms, c.mhz) for c in loaded.cells}
    assert got == want
    planted = {(c.tread, c.state): c.ms for c in overlap_cells(noise=0.0) if c.group_m == 4}
    drifted = next(c for c in loaded.cells if c.tread == 2 and float(c.state) == DUTIES[0])
    assert drifted.ms == pytest.approx(planted[(2, "0")])      # 50 ms never entered


def test_the_pin_rate_is_the_rulers_and_the_ruler_decides_the_tau_floor(session):
    _text, doc = _run([session, "--no-scan", "--no-logo"])
    assert doc["ruler"]["pin_gbps"] == PIN_GBPS
    assert doc["tau_pin_ms"] == pytest.approx(M.tau_ms_at(W, PIN_GBPS))
    fastest = max(c["mhz"] for c in doc["cells"])
    assert doc["f_ref_mhz"] == doc["f_top_mhz"] == fastest
    for f in doc["fits"]:
        if f["model"].endswith(".pin"):
            assert f["parameters"]["tau"]["lower"] == pytest.approx(doc["tau_pin_ms"])
    other = _ruler(session.parent / "other.yaml", pin=2 * PIN_GBPS)
    _text, doc2 = _run([session, "--no-scan", "--no-logo", "--ruler", other])
    assert doc2["tau_pin_ms"] == pytest.approx(doc["tau_pin_ms"] / 2)


def test_f_ref_moves_c0_and_never_an_rms(session):
    """--help says so for every structure. It was false for add.paper.pin while
    that form's Phi used f_ref: Phi multiplies tau there, so tau >= W/pin at
    f_ref is a different bound at a different f_ref, and the rms moved."""
    _t, a = _run([session, "--no-scan", "--no-logo"])
    for moved in (a["f_top_mhz"] + 20.0, a["f_top_mhz"] - 300.0):
        _t, b = _run([session, "--no-scan", "--no-logo", "--f-ref", str(moved)])
        assert b["f_top_mhz"] == a["f_top_mhz"]
        fa = {f["model"]: f for f in a["fits"]}
        for f in b["fits"]:
            assert f["rms"] == pytest.approx(fa[f["model"]]["rms"], rel=1e-6), f["model"]
        e = fa["ovl.group"]["parameters"]["e"]["value"]
        c0 = [d["ovl.group"]["parameters"]["c0"]["value"]
              for d in (fa, {f["model"]: f for f in b["fits"]})]
        assert c0[1] / c0[0] == pytest.approx((a["f_ref_mhz"] / moved) ** e, rel=1e-4)


def test_the_paper_forms_pin_floor_holds_the_rate_at_every_cell(session):
    """tau Phi(f) >= W/pin at every cell: Phi is referenced to the fastest one."""
    _t, doc = _run([session, "--no-scan", "--no-logo", "--f-ref", "3000"])
    f = {x["model"]: x for x in doc["fits"]}["add.paper.pin"]
    tau = f["parameters"]["tau"]["value"]
    e = f["parameters"]["e"]["value"]
    for c in doc["cells"]:
        assert tau * (doc["f_top_mhz"] / c["mhz"]) ** e >= doc["tau_pin_ms"] * (1 - 1e-12)


@pytest.mark.parametrize("gate", [
    {"kind": "VALIDITY", "number": "7", "verdict": "Fail"},
    {"kind": "Validity", "number": "7", "verdict": "FAIL"},
    {"kind": "VALIDITY", "number": "7"},
])
def test_a_gate_spelled_outside_the_exit_code_table_refuses(session, capsys, gate):
    """The page label skipped such a gate, so a failing VALIDITY gate spelled
    'Fail' labelled the page VALID and it was fitted by default.
    exit_codes.classify raises MalformedGate on the same gates."""
    page = session / "results" / "gaps-planted" / "clock_elasticity" / "planted-g4-seed0-00000004"
    report = _gates()
    report["gates"][7] = gate
    (page / "report.json").write_text(json.dumps(report))
    assert M.main([str(session), "--no-scan", "--no-logo"]) == exit_codes.REFUSED
    err = capsys.readouterr().err
    assert str(page / "report.json") in err and "not one of" in err


def test_no_ruler_refuses_and_two_rulers_refuse(tmp_path, capsys):
    page = _r1_page(tmp_path / "lone" / "run-00000001", 1, report=_gates())
    assert M.main([str(page)]) == exit_codes.REFUSED
    assert "--ruler" in capsys.readouterr().err
    a = tmp_path / "a"
    b = tmp_path / "b"
    _ruler(a / "calibration" / "measured_x.yaml")
    _ruler(b / "calibration" / "measured_x.yaml", pin=PIN_GBPS * 1.1)
    _r1_page(a / "run-0000000a", 1, report=_gates())
    _r1_page(b / "run-0000000b", 4, report=_gates())
    assert M.main([str(a), str(b), "--no-scan", "--no-logo"]) == exit_codes.REFUSED
    assert "2 rulers" in capsys.readouterr().err


def test_a_directory_with_no_pages_refuses(tmp_path):
    assert M.main([str(tmp_path)]) == exit_codes.REFUSED


def test_it_writes_nothing_unless_out_is_given(session, tmp_path):
    before = sorted(p for p in tmp_path.rglob("*"))
    out = subprocess.run([sys.executable, str(SCRIPT), str(session), "--no-scan", "--no-logo"],
                         capture_output=True, text=True, cwd=str(tmp_path),
                         env=laptop_env(), timeout=600)
    assert out.returncode == exit_codes.DONE, out.stderr[-2000:]
    assert "FITS:" in out.stdout
    assert sorted(p for p in tmp_path.rglob("*")) == before
    dest = tmp_path / "fit.json"
    assert M.main([str(session), "--no-scan", "--no-logo", "--out", str(dest)]) == exit_codes.DONE
    doc = json.loads(dest.read_text())
    assert {f["model"] for f in doc["fits"]} == set(BY_KEY)


def test_the_page_prints_the_scan_the_registered_alphas_and_leave_one_g_out(session):
    text, doc = _run([session, "--scan-step", "0.1"])
    for label in M.registered_alphas():
        assert label in text
    assert "FLAT WINDOW" in text and "LEAVE ONE G OUT" in text
    assert {c["model"] for c in doc["scan"]} == {s.key for s in M.SCAN_STRUCTURES}
    assert set(doc["leave_one_g_out"]) == set(M.LOGO_KEYS)


def test_predict_scores_other_pages_without_fitting_them(session, tmp_path):
    other = tmp_path / "other-session"
    _ruler(other / "calibration" / "measured_planted.yaml")
    _r1_page(other / "results" / "clock_elasticity" / "run-0000cafe", 16, report=_gates())
    text, doc = _run([session, "--no-scan", "--no-logo", "--predict", other])
    assert doc["predict"]["pages"] == ["0000cafe"]
    assert doc["predict"]["rms"]["ovl.group"] < 1e-6       # noise-free, same world
    assert {c["group_m"] for c in doc["cells"]} == {1, 4, 64}    # never fitted
    assert "never refitted" in text


def _r3_page(path: Path, group_m: int, duty=0.25):
    import csv
    path.mkdir(parents=True, exist_ok=True)
    world = overlap_cells(noise=0.0)
    cols = ["arm", "repeat", "block_m", "tiles", "rows_per_expert", "tokens", "copies",
            "experts_declared", "ms_p50", "sm_clock_load_mhz", "clock_drift_ok",
            "status", "duty"]
    with (path / "cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for arm in ("native", "shared", "private"):
            for c in world:
                if c.group_m != group_m or c.state != "2" or c.tread > 6:
                    continue
                ms = c.ms if arm != "private" else OVERLAP_WORLD["T0"] + c.tread * 0.9
                for rep in range(3):
                    w.writerow({"arm": arm, "repeat": rep, "block_m": 32, "tiles": c.tread,
                                "rows_per_expert": 32 * c.tread, "tokens": 128 * c.tread,
                                "copies": c.tread if arm == "private" else 1,
                                "experts_declared": 8, "ms_p50": ms,
                                "sm_clock_load_mhz": c.mhz, "clock_drift_ok": "True",
                                "status": "ok", "duty": duty})
    report = {"model": MODEL, "dtype": DTYPE, "block_m": 32, "duty": duty,
              "pinned": dict(CE.PINNED, GROUP_SIZE_M=group_m),
              "gates": [{"tag": f"V{i}", "kind": "VALIDITY", "verdict": "PASS"}
                        for i in range(9)]}
    (path / "report.json").write_text(json.dumps(report))
    return path


def test_r3_ladders_join_only_with_the_flag_and_private_cells_read_n_sets(session):
    arm = session / "results" / "gaps-planted" / "private_weight_reference"
    _r3_page(arm / "r3-g4-0000beef", 4)
    _t, doc = _run([session, "--no-scan", "--no-logo"])
    assert {c["arm"] for c in doc["cells"]} == {"R1"}
    _t, doc = _run([session, "--no-scan", "--no-logo", "--r3",
                    "--r3-arms", "native,shared,private"])
    arms = {c["arm"] for c in doc["cells"]}
    assert arms == {"R1", "R3 native", "R3 shared", "R3 private"}
    private = [c for c in doc["cells"] if c["arm"] == "R3 private"]
    assert private and all(c["private"] for c in private)
    assert "rms_by_arm" in doc["fits"][0]


def test_an_unknown_r3_arm_refuses(session):
    assert M.main([str(session), "--r3", "--r3-arms", "native,bogus"]) == exit_codes.REFUSED


def test_every_flag_the_header_names_is_a_flag_help_lists():
    """The header is the description a reader meets first; a flag it names
    that --help does not list is a description of something that is not
    there."""
    import re

    out = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True,
                         text=True, env=laptop_env(), timeout=300)
    assert out.returncode == 0
    named = set(re.findall(r"(--[a-z][a-z0-9-]+)", M.__doc__ or ""))
    assert {"--include-invalid", "--include-unscored", "--r3", "--ruler", "--out"} <= named
    for flag in named:
        assert flag in out.stdout, flag
