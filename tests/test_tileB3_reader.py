"""Tile rental B3's rules, gates, reader and registration (scripts/scoring/tileB3/, scripts/tileB3_gate.py,
docs/registered/2026-11-04-tileB3-gh200): the pure part on synthetic inputs with known answers (the working-model
bar at its edges, deepseek-v2-lite in the pool only, CLEAN certification, the parametrised power equal to read_c2's at
B2's bar, the estimate at both bars against the registration); the sag estimator against the frozen chain's
registered fits on A2's, C2's and C4's published f128 pages and on A2's f64w8 pages, both sag-fit ranges clause by
clause, NaN pages and NaN fits, the f64w8 clock clause, the par tolerance; the census gate on A2's own census and on
drifted copies; the gate CLI's exit codes; the registration (its
constants are the reader's, its pins hold, its cells are B2's, every one priced at three pars, only Phi 64g1 off the
fingerprint, the per-call risk disclosed and the align amendment not adopted); then (slow, where the round-5 work folder and the exfil trees are on this machine) one chain job:
finalise on a stand-in tree reproducing the registered C4-par predictions, the score with no blind page, the score's
VALID path on synthetic times (HOLDS, and FAILS on Phi), G3's per-stratum and homogeneous voids, an edited predictions
file refused, a missing or failed census record NOT SCORED and a missing or failed sag64 record refused, the
timed-page and blind-byte-page guard (the census allowed), the fingerprint refusal, the work files the chain opens all pinned, and the registration rebuilt."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
B3D = REPO / "scripts" / "scoring" / "tileB3"
for p in (REPO, REPO / "scripts", B3D, REPO / "scripts" / "scoring" / "cal2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import b3common as B  # noqa: E402
import read_c2 as RC  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / f"{B.NAME}.json").read_text())
PUB = REPO / "results" / "published"
BOARDS = {"A2": PUB / "2026-10-21-nvidia_gh200_480gb-tileA2-session" / "results",
          "C2": PUB / "2026-10-21-nvidia_gh200_480gb-cal2-session" / "results",
          "C4": PUB / "2026-10-27-nvidia_gh200_480gb-cal4-session" / "results"}
WORK = Path.home() / "code" / "moe-kernels-work"
EXFIL = Path.home() / "moe-kernels-exfil" / "runs"


def _frozen_files() -> tuple:
    import ast
    tree = ast.parse((B3D / "read_b3.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "FROZEN_FILES":
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("no FROZEN_FILES")


FROZEN = _frozen_files()
HAVE_CHAIN = all((WORK / f).is_file() for f in FROZEN) and all(
    (EXFIL / d / "tree/vm/results").is_dir() for d in ("gh200-tile2-20261020T2242Z", "gh200-tile2-20261021T1842Z",
                                                       "gh200-tile1-20261021T0704Z", "gh200-tile4-20261027T2238Z"))
chain_only = pytest.mark.skipif(not HAVE_CHAIN, reason="the round-5 work folder or the A2, S1, C2 and C4 exfil trees are not here")


def rows(model_e: dict) -> list:
    return [{"model": m, "e": e} for m, es in model_e.items() for e in es]


# --------------------------------------------------------------------------
# the bar and the verdict
# --------------------------------------------------------------------------

def test_the_bars_are_the_owners_and_b2s_and_the_registrations():
    assert B.BAR == {"rms_max": 0.03, "max_abs": 0.08, "mean_max": 0.015} == REG["bar"]["values"]
    assert B.B2_BAR == {"rms_max": 0.02, "max_abs": 0.05, "mean_max": 0.01} == REG["bar"]["B2's bar (DESCRIPTIVE beside)"]
    b2 = json.loads((REPO / "docs" / "registered" / f"{B.B2_NAME}.json").read_text())
    assert "rms <= 2%, worst <= 5%, |mean| <= 1%" in b2["tests"]["E1-TILE (primary, D2)"]["rule"]
    for word in ("TileSight", "10.35%", "PipeWeave", "6.0%", "11.5%", "NeuSight", "8.9%", "pre-registered"):
        assert word in REG["bar"]["reason"], word
    rel = REG["bar"]["relation to the round-5 gate (stated plainly)"]
    assert "FAILED it" in rel and "that FAIL stands" in rel and "A 0.428" in rel and "spends" in rel


@pytest.mark.parametrize("es, verdict", [([0.03, -0.03] * 2, "HOLDS"), ([0.0301, -0.0301] * 2, "FAILS"),
                                         ([0.015, 0.015], "HOLDS"), ([0.0151, 0.0151], "FAILS"),
                                         ([0.08, -0.08] + [0.0] * 6, "FAILS"), ([0.05, -0.05] + [0.0] * 6, "HOLDS"),
                                         ([0.08] + [0.0] * 99, "HOLDS"), ([-0.081] + [0.0] * 99, "FAILS")])
def test_the_working_model_bar_at_its_edges(es, verdict):
    assert B.bar(es)["verdict"] == verdict, (es, B.stats(es))


def test_e1_holds_only_when_the_pool_and_every_blind_model_hold_and_deepseek_is_pool_only():
    ok = {m: [0.01, -0.005, 0.012] for m in B.BLIND}
    r = B.e1_rule(rows({**ok, B.DSV: [0.04, -0.04]}), {m: ["p1", "p2"] for m in B.BLIND}, "CLEAN")
    assert r["per_model"][B.DSV]["verdict"] == "FAILS" and "DESCRIPTIVE" in r["per_model"][B.DSV]["label"]
    assert r["pooled"]["verdict"] == "HOLDS" and r["verdict"] == "HOLDS"
    assert all(r["per_model"][m]["certification"] == "CERTIFIED" for m in B.BLIND)
    bad = dict(ok, **{"phi-3.5-moe": [0.02, 0.02, 0.02]})
    r = B.e1_rule(rows(bad), {m: ["p1", "p2"] for m in B.BLIND}, "CLEAN")
    assert r["verdict"] == "FAILS" and r["per_model"]["phi-3.5-moe"]["certification"].startswith("NOT CERTIFIED")
    one = B.e1_rule(rows(ok), {"phi-3.5-moe": ["p1"], "qwen1.5-moe-a2.7b": ["a", "b"], "jetmoe-8b": ["a", "b"]}, "CLEAN")
    assert one["per_model"]["phi-3.5-moe"]["certification"] == "NOT CERTIFIED (1 pages in CLEAN, bar HOLDS)"
    lost = B.e1_rule(rows({m: v for m, v in ok.items() if m != "jetmoe-8b"}), {}, "ALL")
    assert lost["verdict"].startswith("NOT SCORED (incomplete") and lost["models_without_a_valid_cell"] == ["jetmoe-8b"]
    lost = B.e1_rule(rows({"phi-3.5-moe": [0.05] * 3, "qwen1.5-moe-a2.7b": [0.0]}), {}, "ALL")
    assert lost["verdict"].startswith("FAILS (and incomplete")
    assert B.e1_rule([], {}, "ALL")["verdict"].startswith("NOT SCORED")


def test_certification_needs_the_registered_verdict_to_hold():
    clean = {"per_model": {m: {"certification": "CERTIFIED"} for m in B.BLIND}}
    assert B.certification(clean, "HOLDS") == {m: "CERTIFIED" for m in B.BLIND}
    got = B.certification(clean, "INCONCLUSIVE (ALL HOLDS; CLEAN FAILS)")
    assert all(v.startswith("NOT CERTIFIED (E1-TILE reads INCONCLUSIVE") for v in got.values())
    assert B.certification({}, "HOLDS") == {m: "NOT CERTIFIED (no CLEAN cell)" for m in B.BLIND}


# --------------------------------------------------------------------------
# the estimate
# --------------------------------------------------------------------------

def test_the_parametrised_power_is_read_c2s_at_b2s_bar():
    cells = RC.b2_cells()
    b = [RC.to_b(x) for x in B.ROUND5_MEANS.values()]
    import statistics as st
    for bc, sd in (([sum(b) / 5] * 3, st.stdev(b)), ([0.0] * 3, 0.01), ([0.01, -0.005, 0.0], 0.004)):
        assert B.power(cells, bc, sd, B.B2_BAR, R=40_000) == RC.power(cells, bc, sd, R=40_000)


def test_the_estimate_at_both_bars_is_the_registrations():
    pe = REG["pass_estimate"]
    assert B.pass_estimates(rule=B.BAR) == pe["at the working-model bar (3 / 8 / 1.5)"]
    got = B.pass_estimates(rule=B.B2_BAR)
    assert got == pe["at B2's bar (2 / 5 / 1), reproduced"]
    assert {k: v["E1-TILE"] for k, v in got.items()} == {"A (5 models)": 0.124, "B (5 models)": 0.1903,
                                                         "D (O, Q, 8x22B, tp2; 8x7B G1 aside)": 0.1381}
    a = pe["at the working-model bar (3 / 8 / 1.5)"]["A (5 models)"]["E1-TILE"]
    assert 0.40 < a < 0.45 and "about 0.43 (A)" in pe["reading"] and "0.30 to 0.40" in pe["reading"]


# --------------------------------------------------------------------------
# the sag estimator and its gate
# --------------------------------------------------------------------------

FROZEN_FITS = {"A2": {"olmoe-1b-7b": (29.65, 300.0), "mixtral-8x7b": (54.39, 550.0)},
               "C2": {"olmoe-1b-7b": (30.6, 325.0), "mixtral-8x7b": (52.58, 500.0)},
               "C4": {"olmoe-1b-7b": (40.95, 525.0), "mixtral-8x7b": (67.55, 650.0)}}


@pytest.mark.parametrize("board", sorted(BOARDS))
def test_the_repo_estimator_gives_the_frozen_chains_fits_on_three_boards_and_every_board_passes(board):
    g = B.sag_gate(BOARDS[board])
    assert g["verdict"] == "PASS", g["why"]
    for m, (A, tau) in FROZEN_FITS[board].items():
        assert (g["per_page"][m]["A_us"], g["per_page"][m]["tau_us"]) == (A, tau)
        assert g["per_page"][m]["cells"] == 54
    want = B.par(FROZEN_FITS[board].values())
    assert g["par"] == [want[0], want[1]]
    ref = REG["measured_on_board"]["board references, SEEN (gate b3sag run on the published pages)"][board]
    assert ref["par"] == g["par"] and ref["gate"] == "PASS" and ref["ratio_1ms"] == g["ratio_1ms"]
    assert g["inside_held_out_par_span (DESCRIPTIVE)"] is True


def test_the_estimator_is_design_v5s_sag_py_verbatim():
    import ast
    src = (WORK / "2026-10-21" / "design-v5" / "work" / "sag.py")
    if not src.is_file():
        pytest.skip("design-v5's work folder is not here")
    want = {n.name: ast.dump(n) for n in ast.parse(src.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ("pts_of", "fit")}
    got = {n.name: ast.dump(n) for n in ast.parse((B3D / "b3common.py").read_text()).body
           if isinstance(n, ast.FunctionDef) and n.name in ("pts_of", "fit")}
    assert got == want and len(got) == 2


def _fit(A, tau, rms=5.0, cells=54):
    return {"A_us": A, "tau_us": tau, "rms_us": rms, "cells": cells}


@pytest.mark.parametrize("o, m, clause", [
    (_fit(40.95, 525.0), _fit(67.55, 650.0), None),
    (_fit(40.95, 525.0, cells=47), _fit(67.55, 650.0), "47 points"),
    (_fit(14.0, 525.0), _fit(67.55, 650.0), "A 14.0"),
    (_fit(40.95, 525.0), _fit(121.0, 650.0), "A 121.0"),
    (_fit(40.95, 75.0), _fit(67.55, 650.0), "tau 75.0"),
    (_fit(40.95, 525.0), _fit(67.55, 4975.0), "tau 4975.0"),
    (_fit(40.95, 525.0), _fit(67.55, 650.0, rms=20.5), "rms 20.5"),
    (_fit(40.95, 525.0), _fit(40.0, 525.0), "outside [1.0, 2.5]"),
    (_fit(20.0, 525.0), _fit(80.0, 525.0), "outside [1.0, 2.5]"),
    (_fit(16.0, 525.0), _fit(30.0, 525.0), "par A 23.000"),
])
def test_the_sag_fit_range_clause_by_clause(o, m, clause):
    r = B.sag_check({"olmoe-1b-7b": o, "mixtral-8x7b": m})
    if clause is None:
        assert r["verdict"] == "PASS" and r["why"] == ""
    else:
        assert r["verdict"] == "FAIL" and clause in r["why"], r["why"]


def test_the_64x128_sag_on_a2s_f64w8_pages_is_the_frozen_chains_and_knee_v5s_par():
    g = B.sag_gate(BOARDS["A2"], B.SAG64_UNIT)
    assert g["verdict"] == "PASS" and g["gate"] == "sag64", g["why"]
    assert {m: (f["A_us"], f["tau_us"], f["cells"]) for m, f in g["per_page"].items()} == {
        "olmoe-1b-7b": (24.79, 275.0, 54), "mixtral-8x7b": (35.94, 325.0, 54)}
    assert g["par"] == [30.365, 300.0] and abs(g["par"][0] - B.SAG64_A2[0]) <= 0.005 + 1e-9 and g["par"][1] == B.SAG64_A2[1]
    ref = REG["measured_on_board"]["64x128 board reference, SEEN (gate b3sag64 run on A2's f64w8 pages)"]
    assert ref["par"] == g["par"] and ref["gate"] == "PASS" and ref["ratio_1ms"] == g["ratio_1ms"]
    for b in ("C2", "C4"):   # no f64w8 page on C2's or C4's board
        assert B.sag_gate(BOARDS[b], B.SAG64_UNIT)["verdict"] == "FAIL"


@pytest.mark.parametrize("o, m, clause", [
    (_fit(24.79, 275.0), _fit(35.94, 325.0), None),
    (_fit(24.79, 275.0, cells=47), _fit(35.94, 325.0), "47 points"),
    (_fit(9.0, 275.0), _fit(35.94, 325.0), "A 9.0"),
    (_fit(24.79, 275.0), _fit(81.0, 325.0), "A 81.0"),
    (_fit(24.79, 75.0), _fit(35.94, 325.0), "tau 75.0"),
    (_fit(24.79, 275.0), _fit(35.94, 325.0, rms=20.5), "rms 20.5"),
    (_fit(24.79, 275.0), _fit(20.0, 275.0), "outside [1.0, 2.5]"),
    (_fit(10.0, 275.0), _fit(19.0, 275.0), "par A 14.500"),
])
def test_the_64x128_sag_fit_range_clause_by_clause(o, m, clause):
    r = B.sag_check({"olmoe-1b-7b": o, "mixtral-8x7b": m}, B.SAG64_RANGE, B.SAG64_UNIT)
    if clause is None:
        assert r["verdict"] == "PASS" and r["why"] == "" and "inside_held_out_par_span (DESCRIPTIVE)" not in r
    else:
        assert r["verdict"] == "FAIL" and clause in r["why"], r["why"]


@pytest.mark.parametrize("bad", [{"A_us": float("nan")}, {"tau_us": float("nan")}, {"rms_us": float("nan")},
                                 {"A_us": float("inf")}])
@pytest.mark.parametrize("R, unit, o, m", [(B.SAG_RANGE, B.EST_UNIT, (40.95, 525.0), (67.55, 650.0)),
                                           (B.SAG64_RANGE, B.SAG64_UNIT, (24.79, 275.0), (35.94, 325.0))])
def test_a_nan_fit_fails_every_range(bad, R, unit, o, m):
    ok = B.sag_check({"olmoe-1b-7b": _fit(*o), "mixtral-8x7b": _fit(*m)}, R, unit)
    assert ok["verdict"] == "PASS"
    r = B.sag_check({"olmoe-1b-7b": {**_fit(*o), **bad}, "mixtral-8x7b": _fit(*m)}, R, unit)
    assert r["verdict"] == "FAIL" and "olmoe-1b-7b" in r["why"], r["why"]


def _nan_board(tmp_path, unit, field):
    """A2's two pages of a unit copied with one GEMM time per cell made NaN (field: the time or the cycles)."""
    for f in sorted(BOARDS["A2"].glob(f"*-{unit}-r3-counters/lock1410/r3c-g*.json")):
        d = tmp_path / f.parent.parent.name / "lock1410"
        d.mkdir(parents=True)
        p = json.loads(f.read_text())
        for c in p["cells"]:
            pg = c["per_gemm"]["w1"]
            if field == "time":
                pg["gpu__time_duration.sum" if "gpu__time_duration.sum" in pg else "gpu_time_ns"] = float("nan")
            else:
                (c.get("recorded") or {}).get("w1", pg)["sm__cycles_elapsed.avg"] = float("nan")
        (d / f.name).write_text(json.dumps(p))
    return tmp_path


@pytest.mark.parametrize("field", ["time", "cycles"])
def test_a_nan_page_fails_the_gate_and_does_not_crash(tmp_path, field):
    g = B.sag_gate(_nan_board(tmp_path, "f128", field))
    assert g["verdict"] == "FAIL" and g["why"], g
    for m in B.EST_MODELS:          # either the fit refused the page (listed in why) or it is non-finite or on the grid edge
        f = g["per_page"].get(m)
        assert f is None and f"{m}:" in g["why"] or (not math.isfinite(f["A_us"]) or not math.isfinite(f["rms_us"])
                                                      or f["tau_us"] == 50.0)


def test_the_f64w8_clock_clause_on_a2_and_its_threshold(monkeypatch):
    c = B.clock64(BOARDS["A2"])
    assert c["verdict"] == "PASS" and c["band"] == [1352.19, 1410.0] and c["band_which"].startswith("fallback")
    assert {m: (v["check_page"], v["cells_in_band"], v["cells"]) for m, v in c["per_page"].items()} == {
        "olmoe-1b-7b": ("PASS", 26, 27), "mixtral-8x7b": ("PASS", 22, 27)}
    assert B.CLOCK64 == {"cells_in_band_min": 14, "check_page": "PASS"}
    assert REG["stop_rule"]["64x128 sag-fit range"]["clock clause"]["cells_in_band_min"] == 14
    assert REG["measured_on_board"]["64x128 board reference, SEEN (gate b3sag64 run on A2's f64w8 pages)"]["clock"] == c
    monkeypatch.setitem(B.CLOCK64, "cells_in_band_min", 23)
    c = B.clock64(BOARDS["A2"])
    assert c["verdict"] == "FAIL" and "mixtral-8x7b f64w8: 22 cells in the band in force, under 23" in c["why"]
    g = B.sag_gate(BOARDS["A2"], B.SAG64_UNIT)
    assert g["verdict"] == "FAIL" and "22 cells in the band" in g["why"]


def test_the_f64w8_clock_clause_fails_without_f128_pages_or_f64w8_pages(tmp_path):
    (tmp_path / "x").mkdir()
    g = B.sag_gate(tmp_path, B.SAG64_UNIT)
    assert g["verdict"] == "FAIL"
    for d in BOARDS["C4"].glob("*-f128-r3-counters"):     # f128 pages but no f64w8 page
        (tmp_path / d.name).symlink_to(d)
    c = B.clock64(tmp_path)
    assert c["verdict"] == "FAIL" and "0 olmoe-1b-7b f64w8 pages" in c["why"]


def test_the_par_tolerance():
    assert B.PAR_RTOL == 1e-9
    assert B.par_close([54.25, 587.5], [54.25 * (1 + 5e-10), 587.5]) and not B.par_close([54.25, 587.5], [54.2500001, 587.5])
    assert not B.par_close([float("nan"), 1.0], [float("nan"), 1.0]) and not B.par_close(None, [1.0, 2.0])
    assert not B.par_close([1.0], [1.0, 2.0])
    assert "relative 1e-9" in REG["finalisation"]["after the rental, before any blind page is read"]


def test_a_missing_sag_page_fails_the_gate(tmp_path):
    g = B.sag_gate(tmp_path)
    assert g["verdict"] == "FAIL" and "0 olmoe-1b-7b f128 pages" in g["why"] and "0 mixtral-8x7b f128 pages" in g["why"]
    (tmp_path / "x-olmoe-1b-7b-f128-r3-counters").symlink_to(next((BOARDS["C4"]).glob("*-olmoe-1b-7b-f128-r3-counters")))
    g = B.sag_gate(tmp_path)
    assert g["verdict"] == "FAIL" and "0 mixtral-8x7b f128 pages" in g["why"] and "0 olmoe-1b-7b" not in g["why"]
    assert g["per_page"]["olmoe-1b-7b"]["A_us"] == 40.95 and g["par"] is None


# --------------------------------------------------------------------------
# the census gate
# --------------------------------------------------------------------------

def _census():
    TCA = B._census_mods()
    ref = json.loads((REPO / B.A2_MERGED_CENSUS).read_text())
    got = json.loads(B.find_census(BOARDS["A2"], "ta2").read_text())
    return TCA, ref, got


def test_the_census_gate_passes_a2s_own_census_and_names_two_ctas_per_sm_at_128():
    TCA, ref, got = _census()
    r = B.census_check(ref, got, TCA.facts, TCA.FIELDS)
    assert r["verdict"] == "PASS" and r["compared"] == 16 and r["deepseek"] == "admitted"
    for m in B.MODELS:
        assert r["ctas_per_sm"][f"{m}|128x128x64w8s3|w1"] == r["ctas_per_sm"][f"{m}|128x128x64w8s3|w2"] == 2
        assert r["ctas_per_sm"][f"{m}|64x128x64w8s3|w1"] == 3
    ref_reg = REG["measured_on_board"]["census reference, SEEN (gate b3census's rule on A2's own census, label ta2)"]
    assert ref_reg == {"verdict": "PASS", "compared": 16, "ctas_per_sm": r["ctas_per_sm"]}


def test_a_blind_models_drift_or_a_path_mismatch_fails_and_deepseeks_drift_does_not():
    TCA, ref, got = _census()
    g = copy.deepcopy(got)
    e = next(x for x in g["entries"] if x["model"] == "phi-3.5-moe" and x["tile"] == "128x128x64w8s3" and x["gemm"] == "w1")
    e["occupancy"] = dict(e["occupancy"], ctas_per_sm=1)
    r = B.census_check(ref, g, TCA.facts, TCA.FIELDS)
    assert r["verdict"] == "FAIL" and "phi-3.5-moe|128x128x64w8s3|w1: occupancy differ" in r["why"]
    g = copy.deepcopy(got)
    g["entries"] = [x for x in g["entries"] if not (x["model"] == "jetmoe-8b" and x["tile"] == "64x128x64w8s3")]
    assert "not compiled on B3's board" in B.census_check(ref, g, TCA.facts, TCA.FIELDS)["why"]
    g = copy.deepcopy(got)
    e = next(x for x in g["entries"] if x["model"] == B.DSV)
    e["regs"] = (e.get("regs") or 0) + 8
    r = B.census_check(ref, g, TCA.facts, TCA.FIELDS)
    assert r["verdict"] == "PASS" and r["deepseek"] == "NOT SCORED (census drift)"
    g = copy.deepcopy(got)
    g["path_mismatch"] = ["qwen1.5-moe-a2.7b 128x128x64w8s3 w1: mma.sync"]
    assert B.census_check(ref, g, TCA.facts, TCA.FIELDS)["verdict"] == "FAIL"


def _gate(*args):
    return subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "tileB3_gate.py"), *args],
                          capture_output=True, text=True, timeout=120)


def test_the_gate_cli_exit_codes_and_its_ledger_line(tmp_path):
    rec = tmp_path / "gate-b3sag-1410.json"
    got = _gate("sag", "--results", str(BOARDS["C4"]), "--lock", "1410", "--record", str(rec), "--session", str(tmp_path))
    assert got.returncode == 0, got.stdout + got.stderr
    line = got.stdout.strip()
    assert line.startswith("RESULT: GATE b3sag PASS par A 54.2500 us tau 587.5 us ratio 1.5219") and len(line) < 220
    r = json.loads(rec.read_text())
    assert r["verdict"] == "PASS" and r["per_page"]["mixtral-8x7b"]["sha256"][:12] in line
    assert _gate("sag", "--results", str(tmp_path)).returncode == 3
    got = _gate("census", "--results", str(tmp_path))
    assert got.returncode == 3 and "no single B3 tile-census.json (label tb)" in got.stdout
    assert _gate("sag", "--results", str(BOARDS["C4"]), "--lock", "1710").returncode == 2
    assert _gate("sag").returncode == 2
    got = _gate("sag64", "--results", str(BOARDS["A2"]), "--lock", "1410")
    assert got.returncode == 0 and got.stdout.startswith("RESULT: GATE b3sag64 PASS par A 30.3650 us tau 300 us ratio 1.4204")
    assert _gate("sag64", "--results", str(BOARDS["C4"])).returncode == 3
    assert _gate("align", "--results", str(BOARDS["A2"])).returncode == 2   # no align gate in B3 v3


# --------------------------------------------------------------------------
# the registration
# --------------------------------------------------------------------------

def test_the_registration_is_frozen_with_the_readers_files_and_its_pins_hold():
    f = REG["frozen"]
    assert set(f["files"]) == set(FROZEN) and len(FROZEN) == 32
    for p in ("2026-10-28/round5/work/r7.py", "2026-10-28/round5/work/DECLARED5.txt", "2026-10-22/refit4/work/refit4.py"):
        assert p in f["files"]
    if all((WORK / p).is_file() for p in FROZEN):
        assert {p: hashlib.sha256((WORK / p).read_bytes()).hexdigest() for p in FROZEN} == f["files"]
    assert f["files"]["2026-10-28/round5/work/DECLARED5.txt"].startswith("e2dd41fe")
    assert RC.check_pins(REG, REPO) == []
    for p in ("scripts/gh200_tileB3_session.sh", "scripts/tileB3_gate.py", "scripts/scoring/tileB3/b3common.py",

              "scripts/scoring/tileB3/read_b3.py", "scripts/scoring/tileB3/register_b3.py", "scripts/gh200_cal2_session.sh",
              "scripts/scoring/cal2/read_c2.py", "scripts/tileA2_gate.py", "scripts/tile_census_a2.py"):
        assert p in REG["code_pins"], p
    for p in f["chain_repo_modules"]:
        assert p in REG["code_pins"], p
    assert f["round 5 selected table, re-priced by the frozen code at registration"] == {
        **B.ROUND5_MEANS, "mixtral-8x7b C2": B.ROUND5_M_C2}
    c = f["constants"]
    assert "(6.0, 2.75)" in c["K1 knee (p_128, S0)"] and "60 MiB" in c["L2 capacity"] and "0.0375" in c["w2 byte law W2"]
    assert "MEASURED-ON-BOARD" in c["F1 (A, tau) at 128x128"] and "30.37 us / 300 us" in c["64x128 cells (Phi-3.5 64g1, 6 cells)"]
    assert "MEASURED-ON-BOARD" in c["64x128 cells (Phi-3.5 64g1, 6 cells)"] and "s_block -7.07" in c["T0, c, bw, F4"]
    assert not any("align" in k for k in c) and "scripts/launch_floor.py" not in REG["code_pins"]
    assert c["JetMoE-8B 128g1 n2 PT (B2 index 72)"].startswith("PRICED")


def test_the_cells_are_b2s_84_in_b2s_order_every_one_priced_at_three_pars():
    cells = RC.b2_cells()
    lst = REG["cells"]["list"]
    assert REG["cells"]["count"] == 84 == len(lst) and REG["cells"]["unpriced"] == []
    assert [(x["model"], x["page"], x["cls"]) for x in lst] == [(c["model"], c["page"], c["cls"]) for c in cells]
    assert (lst[72]["model"], lst[72]["cls"], lst[72]["n"], lst[72]["label"]) == ("jetmoe-8b", "128g1", 2, "PT")
    pre = REG["frozen"]["predictions before the rental (frozen model at three boards' 128x128 pars and A2's 64x128 par; "
                         "the rental's own pars finalise)"]
    assert set(pre) == {"A2", "C2", "C4"}
    for b, v in pre.items():
        assert len(v["ms"]) == 84 and all(isinstance(x, float) and 0.2 < x < 5.0 for x in v["ms"]), b
    fp = REG["frozen"]["fingerprint"]["ms"]
    assert pre["A2"]["par"] == [42.02, 425.0] and pre["A2"]["par64"] == [30.365, 300.0]
    moved = {i for i, (x, a, f) in enumerate(zip(lst, pre["A2"]["ms"], fp, strict=True)) if a != f}
    assert moved and moved <= {i for i, x in enumerate(lst) if x["cls"] == "64g1"}   # only 64x128 cells take A2's estimator par
    assert pre["C4"]["par"] == [54.25, 587.5] and pre["C2"]["par"] == [41.59, 412.5]
    by = {}
    for x, a, c4 in zip(lst, pre["A2"]["ms"], pre["C4"]["ms"]):
        by.setdefault(x["cls"], []).append(c4 / a - 1)
    assert by["64g1"] == [0.0] * 6                          # the 64x128 par is A2's at all three 128x128 pars
    assert all(abs(v) < 0.02 for vs in by.values() for v in vs)
    sk = REG["frozen"]["blind stakes, reproduced (round 5 PART D: frozen model against v5'' and round 4, A2's par)"]
    assert sk["phi-3.5-moe 128g1"]["vs_v5pp"]["mean"] == 3.64 and sk["phi-3.5-moe 128g1"]["vs_round4"]["mean"] == 6.13


def test_the_registration_text_and_scope_and_stop_rule():
    text = (REPO / "docs" / "registered" / f"{B.NAME}.txt").read_text()
    assert chr(0x2014) not in text and chr(0x2013) not in text and chr(0x2014) not in json.dumps(REG, ensure_ascii=False)
    for s in ("FROZEN", "STOP RULE", "SCOPE:", "PASS ESTIMATE (ESTIMATE)", "FINALISATION", "SEEN"):
        assert s in text, s
    sc = REG["certification"]["scope"]
    for s in ("1410 MHz", "700 W", "128x128x64 w8 s3 at G 1 and G 16", "64x128x64 w8 s3 at G 1", "B2's 84",
              "does not certify B2's bar"):
        assert s in sc, s
    sr = REG["stop_rule"]
    assert set(sr) >= {"power", "lock", "census", "band", "sag-fit range", "64x128 sag-fit range", "order"}
    assert "align range" not in sr
    assert sr["sag-fit range"]["range"] == {k: (list(v) if isinstance(v, tuple) else v) for k, v in B.SAG_RANGE.items()}
    assert sr["64x128 sag-fit range"]["range"] == {k: (list(v) if isinstance(v, tuple) else v) for k, v in B.SAG64_RANGE.items()}
    assert sr["order"][-2] == "gate b3sag64" and not any("align" in s for s in sr["order"])
    seen = " ".join(REG["seen"])
    for s in ("A2", "S1", "C2", "C3", "C4", "rounds 2 to 5", "0.03 points", "OD13", "+3.64", "SEEN-FITTED",
              "launch-floor rentals 1 to 3", "Granite-3.0-1B-A400M", "Granite-3.0-3B-A800M", "Mixtral-8x7B-tp8",
              "JetMoE-8B traces at R3's old tile", "per-call check", "decode-addendum4", "bfc45176", "B3 v2",
              "NOT adopted", "did not open the DS pages"):
        assert s in seen, s
    pc = REG["per-call risk (disclosed, not modelled)"]
    assert "does NOT price the block-scan align by numel" in pc["statement"] and "36.16 us" in pc["statement"]
    chk = pc["the B3 per-call check (moe-kernels-work 2026-11-04 b3-percall REPORT.md, offline, read-only)"]
    assert "0.30 to 0.40" in chk["estimate A at the working-model bar with the per-call shifts"]
    na = pc["the amendment tried, NOT ADOPTED (B3 v2, offline, 2026-11-04)"]
    assert "OLMoE +0.45 -> +2.01" in na and "worsened the held-out fit" in na and "not adopt" in na
    assert "amendment" not in REG and "NOT ADOPTED" in text and "PER-CALL RISK" in text
    assert REG["plan"]["sha256"] == hashlib.sha256((REPO / B.PLAN).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# the frozen chain (slow)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def job(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("b3job")
    got = subprocess.run([sys.executable, "-B", str(REPO / "tests" / "tileB3_chain_job.py"), str(tmp)],
                         capture_output=True, text=True, timeout=1500, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    line = [x for x in got.stdout.splitlines() if x.startswith("JSON ")]
    assert got.returncode == 0 and line, got.stdout[-3000:] + got.stderr[-3000:]
    return json.loads(line[-1][5:])


@chain_only
def test_finalise_on_a_stand_in_board_reproduces_the_registered_predictions_at_its_par(job):
    s = job["standin"]
    assert s["equals_registered_C4"] is True and s["par"] == [54.25, 587.5] and s["n"] == 84 and s["unpriced"] == []
    assert s["par64"] == [30.365, 300.0]
    assert s["validity"] == {"valid": True, "why": ""}
    assert s["E1-TILE"].startswith("NOT SCORED (incomplete: no valid cell of phi-3.5-moe, qwen1.5-moe-a2.7b, jetmoe-8b")
    assert "0 OLMoE f128r pages" in s["drift"]["why"]


@chain_only
def test_the_score_holds_on_small_errors_and_fails_on_a_phi_bias(job):
    v = job["valid_small"]
    assert v["verdict"] == "HOLDS" and v["certification"] == {m: "CERTIFIED" for m in B.BLIND}
    assert v["pooled"]["cells"] == 84 and v["b2_bar_reading"] == "HOLDS"
    p = job["valid_phi"]
    assert p["verdict"] == "FAILS" and p["per_model"]["phi-3.5-moe"]["verdict"] == "FAILS"
    assert all(p["per_model"][m]["verdict"] == "HOLDS" for m in ("qwen1.5-moe-a2.7b", "jetmoe-8b"))
    assert all(c.startswith("NOT CERTIFIED") for c in p["certification"].values())
    assert math.isclose(p["per_model"]["phi-3.5-moe"]["mean"], 0.02, abs_tol=1e-6)


@chain_only
def test_g3s_per_stratum_voids_are_applied_and_the_homogeneous_fail_voids_every_histogram_cell(job):
    s = job["g3_stratum"]
    assert s["verdict"] == "HOLDS" and len(s["views"]) == 2
    for v in s["views"]:
        assert 0 < v["voided"] < 84 and v["kept"] == 84 - v["voided"] and "balanced" not in v["voided_labels"]
        assert v["without_voids"] == "HOLDS"
    a = job["g3_all"]
    assert a["verdict"].startswith("NOT SCORED: G3 FAILS (homogeneous)") and a["validity"]["valid"] is True
    assert all(v["E1"].startswith("NOT SCORED: G3 FAILS (homogeneous)") for v in a["views"])


@chain_only
def test_score_refuses_an_edited_predictions_file(job):
    assert job["edited"] == "REFUSED: the predictions file does not equal the finalised file on these pages"


@chain_only
def test_a_missing_or_failed_census_record_reads_not_scored_and_a_sag64_record_refuses(job):
    r = job["records"]
    for how, read in (("missing", "None"), ("FAIL", "'FAIL'")):
        c = r[f"census {how}"]
        assert c["E1-TILE"].startswith("NOT SCORED: the board or a gate is not valid") and f"gate census reads {read}" in c["E1-TILE"]
        assert c["validity"]["valid"] is False and all(v == "NOT CERTIFIED (not scored)" for v in c["certification"].values())
        assert r[f"sag64 {how}"]["refused"].startswith("REFUSED: the sag64 gate did not PASS on this board")


@chain_only
def test_finalise_refuses_a_timed_page_and_a_moved_fingerprint(job):
    assert job["guard"] and job["guard"].startswith("REFUSED: finalisation opened a blind page") and "/gaps-" in job["guard"]
    assert job["guard_bytes"].startswith("REFUSED: finalisation opened a blind page") and "phi-3.5-moe-b128g1" in job["guard_bytes"]
    assert job["guard_census"] is None           # its path names phi-3.5-moe: the census exemption, exercised
    assert job["refuse_clock"].startswith("REFUSED: the f64w8 pages' clock clause fails here")
    assert job["refuse"] == "REFUSED: the frozen round-5 model does not reproduce the registered fingerprint"


@chain_only
def test_every_work_file_the_chain_opens_is_pinned_and_every_repo_module_is_registered(job):
    t = job["trace"]
    assert t["not_pinned"] == [] and t["repo_modules_bad"] == []
    opened = set(t["opened_work_files"])
    assert {"2026-10-28/round5/work/r7.py", "2026-10-22/refit4/work/refit4.py", "2026-10-27/round3/work/r6.py"} <= opened


@chain_only
def test_the_registration_rebuilds_from_the_frozen_chain(job):
    assert job["register_check"] == {"json": True, "txt": True}
