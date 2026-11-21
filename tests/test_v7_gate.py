"""Model v7's gate registration (docs/registered/2026-11-17-v7-gate-gh200) and its scorer (scripts/scoring/v7/gate_v7.py),
on SYNTHETIC fixtures only: the gate runs on the published pages only after the registration is public, so no test
here prices a gate page. The registration recomputes to its committed files and freezes v7's code, the vendored round
5, the readings and the quoted registrations; the scorer refuses every regime on a broken freeze, reads each clause
at its registered bar and edge, keeps each regime's GO independent, and prints one line per regime."""
from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "v7", REPO / "scripts" / "scoring" / "decode"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gate_v7 as GV  # noqa: E402
import r3_timing_model_v7 as V7  # noqa: E402
import register_v7 as RG  # noqa: E402

REGF = REPO / "docs" / "registered" / f"{GV.NAME}.json"


def _reg() -> dict:
    return json.loads(REGF.read_text())


GOOD_MEANS = {"olmoe-1b-7b": 0.4, "qwen2-57b-a14b": -0.9, "mixtral-8x7b": 1.0, "mixtral-8x22b": -1.0,
              "mixtral-8x7b-tp2": 0.2, "mixtral-8x7b C2": 0.99}


# ---------------------------------------------------------------- the registration

def test_the_registration_recomputes_to_its_committed_files(capsys):
    assert RG.main(["--check"]) == 0
    assert "DIFFERS" not in capsys.readouterr().out


def test_the_freeze_holds_on_this_tree_and_names_v7_the_gate_round_5_and_the_quoted_registrations():
    reg = _reg()
    fz = reg["freeze"]
    assert GV.verify_freeze(reg) == []
    assert fz["constants_sha256"] == V7.constants_sha256() and fz["constants"] == V7.constants()
    for p in ("scripts/r3_timing_model_v7.py", "scripts/round5_chain.py", "scripts/scoring/v7/gate_v7.py",
              "scripts/r3_timing_model_v6d.py", "scripts/r3_timing_model_v4.py", "moe/spec.py",
              "scripts/scoring/pc/pc.read.json", "scripts/scoring/decode/d6_build/pc_spans.json",
              "docs/registered/2026-11-04-tileB3-gh200.json", "docs/registered/2026-10-27-cal4-gh200.txt",
              "docs/registered/2026-11-10-decode-addendum6-gh200.json", "moe/routing/distributions.py",
              "moe/__init__.py", "moe/bench/tile_resolve.py", "scripts/launch_floor.py"):
        assert p in fz["files"], p
    code = [p for p in fz["files"] if p.endswith(".py") and not p.startswith("scripts/round5_frozen/")]
    assert code == list(RG.freeze_code()) and "scripts/scoring/v7/register_v7.py" not in code
    vend = [p for p in fz["files"] if p.startswith("scripts/round5_frozen/")]
    b3 = json.loads((REPO / "docs/registered/2026-11-04-tileB3-gh200.json").read_text())["frozen"]["files"]
    assert len(vend) == 32 and {p.split("round5_frozen/")[1]: fz["files"][p] for p in vend} == b3


def test_the_registration_states_the_old_bars_the_go_rule_and_the_disclosures():
    reg = _reg()
    g = reg["gate"]
    assert "each regime's GO independent" in g["GO rule"] and "ONE blind only on its own GO" in g["GO rule"]
    assert "no bar may be changed after this file is public" in g["GO rule"]
    assert "300 registered v3 predictions" in g["R3"]["G-R3"] and "bit for bit" in g["R3"]["G-R3"]
    assert "within 1.0%" in g["PREFILL"]["G1"] and "leave-one-model-out exactly as round 5" in g["PREFILL"]["G1"]
    assert "P(A) >= 0.5" in g["PREFILL"]["G2"] and "seed 20261022" in g["PREFILL"]["G2"]
    assert "never a clause" in next(k for k in g["PREFILL"] if "never" in k)
    assert "3% rms, 8% worst cell, 1.5% |mean|" in g["DECODE"]["CAL (addendum 6's clause 1)"]
    assert "under 4.0% over the committed 220 cells" in g["DECODE"]["RE-SCORE (addendum 6's clause 2)"]
    d = reg["disclosures"]
    seen = " ".join(d["every v7 constant and rule is SEEN"])
    assert "AFTER pricing launch-layer variants on SEEN pages" in seen and "D6-24 defect was found on SEEN pages" in seen
    assert "five declared rounds" in seen
    est = d["the gate is a registered statement of a known estimate, not a blind test"]
    assert "POST-HOC" in est and "NO-GO as estimated" in est and "GO only with the D6-24 fix" in est
    assert "2026-11-17" in d["the owner's reopening (2026-11-17)"] or "reopened" in d["the owner's reopening (2026-11-17)"]
    assert any("CAL2 under v7 reads GO" in x for x in d["computed by the build before this registration (POST-HOC, SEEN; "
                                                         "disclosed, not a gate reading)"])
    assert (GV.G1_MAX_ABS_MEAN_PCT, GV.G2_PA_MIN, GV.RESCORE_RMS_MAX, GV.RESCORE_CELLS, GV.R3_PREDICTIONS) == (
        1.0, 0.5, 0.04, 220, 300)
    assert GV.CAL_BARS == {"rms_max": 0.03, "max_abs": 0.08, "mean_max": 0.015}


def test_the_clause_bars_are_addendum_6s_and_c4s():
    import d6common as D6
    assert (D6.RESCORE_RMS_MAX, D6.RESCORE_CELLS) == (GV.RESCORE_RMS_MAX, GV.RESCORE_CELLS)
    assert {k: D6.BARS_OWNER[k] for k in GV.CAL_BARS} == GV.CAL_BARS
    c4 = (REPO / "docs/registered/2026-10-27-cal4-gh200.txt").read_text()
    assert "G1 every held-out 128 G 1 mean within 1.0%, Mixtral-8x7B on C2's board as well" in c4
    assert "G2 estimated P(A) >= 0.5" in c4


# ---------------------------------------------------------------- the freeze

def test_a_changed_frozen_file_or_constant_refuses_every_regime_and_prices_nothing(tmp_path):
    reg = _reg()
    bad = copy.deepcopy(reg)
    p = next(iter(bad["freeze"]["files"]))
    bad["freeze"]["files"][p] = "0" * 64
    assert any(p in x for x in GV.verify_freeze(bad))
    assert GV.verify_freeze(reg, constants_sha="f" * 64)
    called = []

    def comp(sw):
        called.append(sw)
        raise AssertionError("priced under a broken freeze")
    doc = GV.evaluate(bad, compute={k: comp for k in GV.REGIMES})
    assert doc["freeze"]["verdict"] == "REFUSED" and not called
    assert all(doc[k]["verdict"] == "NO-GO" for k in GV.REGIMES)
    assert GV.lines(doc)[-1].startswith("FREEZE: REFUSED")
    gone = copy.deepcopy(reg)
    gone["freeze"]["files"]["scripts/no_such_file.py"] = "0" * 64
    assert any("missing" in x for x in GV.verify_freeze(gone))


# ---------------------------------------------------------------- the clauses at their bars

def test_g_r3_needs_all_300_and_no_difference():
    assert GV.r3_clause({"compared": 300, "unequal_registered": [], "unequal_pricer": []})["verdict"] == "GO"
    assert GV.r3_clause({"compared": 299, "unequal_registered": [], "unequal_pricer": []})["verdict"] == "NO-GO"
    assert GV.r3_clause({"compared": 300, "unequal_registered": [{}], "unequal_pricer": []})["verdict"] == "NO-GO"
    assert GV.r3_clause({"compared": 300, "unequal_registered": [], "unequal_pricer": [{}]})["verdict"] == "NO-GO"


@pytest.mark.parametrize("k,v,go", [("mixtral-8x7b C2", 1.0, True), ("mixtral-8x7b C2", -1.0, True),
                                    ("mixtral-8x7b C2", 1.01, False), ("mixtral-8x7b-tp2", -1.01, False),
                                    ("olmoe-1b-7b", math.nan, False)])
def test_g1_reads_every_held_out_mean_at_1_percent_inclusive(k, v, go):
    got = GV.prefill_clauses(dict(GOOD_MEANS, **{k: v}), 0.6)
    assert (got["verdict"] == "GO") is go and (got["G1"]["verdict"] == "PASSES") is go


def test_g1_fails_without_every_set_and_g2_at_its_edge():
    m = dict(GOOD_MEANS)
    del m["mixtral-8x7b C2"]
    assert GV.prefill_clauses(m, 0.9)["verdict"] == "NO-GO"
    assert GV.prefill_clauses(GOOD_MEANS, 0.5)["verdict"] == "GO"
    for pa in (0.4999, None, math.nan):
        got = GV.prefill_clauses(GOOD_MEANS, pa)
        assert got["verdict"] == "NO-GO" and got["G2"]["verdict"] == "FAILS" and got["G1"]["verdict"] == "PASSES"


@pytest.mark.parametrize("cal,rms,cells,go", [("GO", 0.0399, 220, True), ("GO", 0.04, 220, False),
                                              ("STOP", 0.02, 220, False), ("GO", 0.02, 219, False),
                                              ("GO", math.nan, 220, False), (None, 0.02, 220, False)])
def test_the_decode_clauses_at_their_bars(cal, rms, cells, go):
    got = GV.decode_clauses(cal, {"rms": 0.018}, {"rms": rms, "cells": cells})
    assert (got["verdict"] == "GO") is go
    assert GV.decode_clauses("GO", None, None)["verdict"] == "NO-GO"


# ---------------------------------------------------------------- the regimes are independent

def _synthetic(r3_ok=True, means=None, pa=0.7, cal="GO", rms=0.03, boom=None):
    calls = []

    def r3(sw):
        calls.append(("r3", sw))
        return {"compared": 300 if r3_ok else 12, "unequal_registered": [], "unequal_pricer": []}

    def prefill(sw):
        calls.append(("prefill", sw))
        if boom == "prefill":
            raise RuntimeError("synthetic failure")
        return {"means": means or GOOD_MEANS, "P_A": pa, "estimate": {"A (5 models)": pa}, "estimate_cells": 84,
                "b3_information": {"pooled": {"n": 84, "mean": 0.1, "rms": 1.0, "worst": 2.0}}}

    def decode(sw):
        calls.append(("decode", sw))
        return {"cal_verdict": cal, "cal_stats": {"rms": 0.016}, "pooled": {"rms": rms, "cells": 220}, "rho": "fixed"}
    return {"r3": r3, "prefill": prefill, "decode": decode}, calls


def test_each_regime_reads_its_own_go_and_the_ablation_runs_with_every_switch_off():
    comp, calls = _synthetic(means=dict(GOOD_MEANS, **{"mixtral-8x7b-tp2": -1.99}))
    doc = GV.evaluate(_reg(), compute=comp)
    assert (doc["r3"]["verdict"], doc["prefill"]["verdict"], doc["decode"]["verdict"]) == ("GO", "NO-GO", "GO")
    assert ("prefill", V7.OFF) in calls and ("decode", V7.OFF) in calls and ("r3", V7.OFF) not in calls
    assert {sw for _k, sw in calls} == {V7.V7, V7.OFF}
    assert "ablation (DESCRIPTIVE)" in doc["prefill"] and "B3 84 cells (SEEN, never a clause)" in doc["prefill"]["information"]
    L = GV.lines(doc)
    assert L[0].startswith("R3: GO") and "ONE blind" in L[0]
    assert L[1].startswith("PREFILL: NO-GO: G1") and "mixtral-8x7b-tp2 -1.99" in L[1] and L[1].endswith("no blind for this regime")
    assert L[2].startswith("DECODE: GO")


def test_a_failing_computation_is_that_regimes_no_go_and_no_other_regimes():
    comp, _calls = _synthetic(boom="prefill")
    doc = GV.evaluate(_reg(), compute=comp, ablation=False)
    assert doc["prefill"]["verdict"] == "NO-GO" and doc["prefill"]["why"][0].startswith("REFUSED: RuntimeError")
    assert doc["r3"]["verdict"] == "GO" and doc["decode"]["verdict"] == "GO"


def test_the_cli_exit_codes_lines_and_files(tmp_path, monkeypatch, capsys):
    comp, _ = _synthetic()
    real = GV.evaluate
    monkeypatch.setattr(GV, "evaluate", lambda reg: real(reg, compute=comp))
    assert GV.main(["--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [x.split(":")[0] for x in out] == ["R3", "PREFILL", "DECODE"]
    assert json.loads((tmp_path / "o" / "gate.json").read_text())["decode"]["verdict"] == "GO"
    assert (tmp_path / "o" / "gate.txt").read_text().splitlines() == out
    comp2, _ = _synthetic(cal="STOP")
    monkeypatch.setattr(GV, "evaluate", lambda reg: real(reg, compute=comp2))
    assert GV.main([]) == 1
    assert json.loads((tmp_path / "o" / "gate.json").read_text())["registration_sha256"] == GV.sha256(REGF)


def test_the_gate_reads_only_the_committed_registration(tmp_path, monkeypatch, capsys):
    """No option names another registration (a substituted file could drop paths from the freeze); a missing
    registration at the committed path is a usage refusal."""
    with pytest.raises(SystemExit) as ex:
        GV.main(["--registration", str(tmp_path / "other.json")])
    assert ex.value.code == 2
    monkeypatch.setattr(GV, "REG_PATH", tmp_path / "none.json")
    assert GV.main([]) == 2 and "REFUSED: no registration" in capsys.readouterr().out


# ---------------------------------------------------------------- house rules

def test_no_em_dash_and_no_trailer_in_the_gate_files():
    paths = [REPO / "scripts/scoring/v7/gate_v7.py", REPO / "scripts/scoring/v7/register_v7.py", Path(__file__),
             REGF, REGF.with_suffix(".txt")]
    for p in paths:
        t = p.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, p
        assert not re.search("|".join(("Co-" + "Authored-By", "Claude " + "Code", "Generated " + "with", "Anth" + "ropic")), t), p


# ---------------------------------------------------------------- the computations' plumbing (stubs, no page priced)

def test_compute_prefill_wires_the_six_sets_the_estimate_and_b3s_information(monkeypatch):
    """Every pricing call is stubbed (synthetic rows, a fixed price): the six sets' means reach G1 by name, the five
    models' means and the priced power cells reach the estimate, and B3's information pools by model."""
    import types

    import round5_chain as R5
    rows = [{"model": "phi-3.5-moe", "page": "phi-3.5-moe-128g1", "n": 1, "label": "PT", "doc": {"block_m": 128}},
            {"model": "jetmoe-8b", "page": "jetmoe-8b-64g1", "n": 2, "label": "PW", "doc": {"block_m": 64}}]
    cells = [{"model": "phi-3.5-moe"}, {"model": "jetmoe-8b"}]
    fake = {"r4": types.SimpleNamespace(summ=lambda e: GV.summ(e)),
            "RF": types.SimpleNamespace(RC=types.SimpleNamespace(b2_cells=lambda: cells))}
    sets = {k: ([{"meas": 1.0, "board": "A2"}, {"meas": 2.0, "board": "A2"}], k) for k in GV.HELD_OUT}
    seen = {}
    monkeypatch.setattr(R5, "chain", lambda: fake)
    monkeypatch.setattr(R5, "held_out_sets", lambda: sets)
    monkeypatch.setattr(R5, "held_out_knee", lambda hm: R5.K1_SEEN)
    monkeypatch.setattr(R5, "held_out_sag", lambda r, hm: (40.0, 400.0))
    monkeypatch.setattr(R5, "b2_rows", lambda: rows)
    monkeypatch.setattr(R5, "reads", lambda r, p: {"w1": 1.0, "w2": 1.0})
    monkeypatch.setattr(R5, "pass_estimate", lambda means, c: seen.update(means=means, cells=c) or {"A (5 models)": 0.3})
    monkeypatch.setattr(V7, "price_prefill", lambda r, **kw: {"ms": 1.01 * r.get("meas", 1.0)})
    monkeypatch.setattr(GV, "prefill_trees", lambda: [])
    got = GV.compute_prefill()
    assert set(got["means"]) == set(GV.HELD_OUT) and all(abs(v - 1.0) < 1e-9 for v in got["means"].values())
    assert list(seen["means"]) == list(GV.FIVE) and seen["cells"] == cells and got["P_A"] == 0.3
    assert got["estimate_cells"] == 2
    b3 = json.loads((REPO / "scripts/scoring/tileB3/tileB3.score.json").read_text())
    have = {tuple(c["cell"].split("/")[:4]) for c in b3["views"]["cells"]}
    want = {m for m in ("phi-3.5-moe", "jetmoe-8b") if any(k[0] == m and k[1] == r["page"] and int(k[2]) == r["n"]
                                                          and k[3] == r["label"] for k in have for r in rows)}
    assert set(got["b3_information"]) == (want | {"pooled"} if want else set())
    assert GV.prefill_clauses(got["means"], got["P_A"])["verdict"] == "NO-GO"


def test_the_prefill_trees_refusal_reads_round_5s_table_and_b3s_84_predictions(monkeypatch):
    """Stubbed pricing (no page priced): with every switch off the six means must equal tileB3.score.json's round 5
    table and every B2 row must give B3's committed ms and reads; one difference anywhere and compute_prefill is
    REFUSED (TREES) before any clause, and the gate reads that regime NO-GO."""
    import types

    import round5_chain as R5
    want = json.loads((REPO / "scripts/scoring/tileB3/tileB3.score.json").read_text())["preflight"]["round5_table"]
    fin = json.loads((REPO / "scripts/scoring/tileB3/b3.predictions.json").read_text())
    sets = {k: ([{"meas": 1.0, "set": k}], k) for k in want}
    rows = [{"b3": c} for c in fin["cells"]]
    seen_sw = set()

    def price(r, **kw):
        seen_sw.add(kw["switches"])
        if "set" in r:
            return {"ms": 1.0 + want[r["set"]] / 100}
        return {"ms": r["b3"]["ms"], "reads": dict(r["b3"]["reads"])}
    monkeypatch.setattr(R5, "chain", lambda: {"r4": types.SimpleNamespace(summ=lambda e: GV.summ(e))})
    monkeypatch.setattr(R5, "held_out_sets", lambda: sets)
    monkeypatch.setattr(R5, "held_out_knee", lambda hm: R5.K1_SEEN)
    monkeypatch.setattr(R5, "held_out_sag", lambda r, hm: (40.0, 400.0))
    monkeypatch.setattr(R5, "b2_rows", lambda: rows)
    monkeypatch.setattr(V7, "price_prefill", price)
    assert GV.prefill_trees() == [] and seen_sw == {V7.OFF}
    rows[5] = {"b3": dict(fin["cells"][5], reads={"w1": 1, "w2": 2})}
    bad = GV.prefill_trees()
    assert len(bad) == 1 and bad[0].startswith("B3 cell 5")
    rows[5] = {"b3": fin["cells"][5]}
    k0 = next(iter(want))
    sets[k0] = ([{"meas": 1.0, "set": k0}, {"meas": 1.0, "set": k0, "extra": 1}], k0)
    monkeypatch.setattr(V7, "price_prefill", lambda r, **kw: {"ms": 1.5} if "extra" in r else price(r, **kw))
    bad = GV.prefill_trees()
    assert len(bad) == 1 and bad[0].startswith("round 5's held-out table")
    with pytest.raises(V7.Refused, match="^TREES: "):
        GV.compute_prefill()
    comp, _ = _synthetic()
    comp["prefill"] = GV.compute_prefill
    doc = GV.evaluate(_reg(), compute=comp, ablation=False)
    assert doc["prefill"]["verdict"] == "NO-GO" and doc["prefill"]["why"][0].startswith("REFUSED: Refused: TREES")
    assert doc["r3"]["verdict"] == "GO" and doc["decode"]["verdict"] == "GO"


def test_compute_decode_wires_the_fix_fit_the_cal_score_and_the_re_score(monkeypatch):
    """The trees, the scorers and the prediction are stubbed (no page is read beyond the committed readings): with
    D6FIX on the fix fit is read first with v4d' on every naive cell, its prm reaches the CAL score and the re-score,
    and the clauses get the stop verdict and the pooled stats."""
    import d4common as D4
    import finalise_decode as FN
    import gate_d6 as G
    import score_blind_d3 as SB3
    import score_cal_d3 as SC3
    seen = []
    D3 = V7.decode_modules()["D3"]

    def sc3(tree, reg, tp, src, settle=False):
        seen.append(("cal", D3.price_cell_v4dp.__closure__ is not None))
        form = {"T0_GR_ms": -0.002, "s_naive_GR_ms": 0.0095, "s_block_GR_ms": -0.0007}
        return {"FIT": {"selected": "scaled", "forms": {"scaled": form}}, "STOP": {"verdict": "GO", "why": []},
                "E1_CAL_IN_SAMPLE": {"stats": {"cells": 91, "rms": 0.016}}}
    monkeypatch.setattr(D4, "reproduce", lambda *a, **k: {"verdict": "REPRODUCES"})
    monkeypatch.setattr(SC3, "score", sc3)
    monkeypatch.setattr(G, "frozen_at", lambda c: {})
    monkeypatch.setattr(FN, "predict", lambda fz, models: seen.append(("predict", tuple(models))) or [])
    monkeypatch.setattr(SB3, "score", lambda tree, reg, fin: {"E1_DEC": {"pooled": {"stats": {"cells": 220, "rms": 0.03}},
                                                                         "per_model": {}}})
    got = GV.compute_decode()
    assert [s[0] for s in seen] == ["cal", "cal", "predict"]
    assert got["fix"]["rho"] == "scaled" and got["cal_verdict"] == "GO" and got["pooled"] == {"cells": 220, "rms": 0.03}
    assert GV.decode_clauses(got["cal_verdict"], got["cal_stats"], got["pooled"])["verdict"] == "GO"
    seen.clear()
    off = GV.compute_decode(V7.OFF)
    assert [s[0] for s in seen] == ["cal", "predict"] and off["fix"] is None
