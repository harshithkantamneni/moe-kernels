"""The decode stage's sixth addendum (docs/registered/2026-11-10-decode-addendum6-gh200), off the GPU: model v6d, its
gate and addendum 5's new blind under it. The registration recomputes to its committed files and pins every earlier
decode file at its bytes; the freeze record's hashes hold; v6d with the law off is v5d on every registered page cell,
and its constants are the readings by the registered rules (d6common.law_from), never refitted after them. The build
law with switches prices every GEMM as the model at the candidate. The gate prints one line, reads NO-GO on the
published trees (clause 1 GO, clause 2 FAILS), refuses readings that are BLOCKING or that do not give the registered
v6d, and the finalisation refuses unless it reads GO; the blind scorer refuses anything else. The new blind's pages are
addendum 5's, its plan is not written, and no page of a new model has been timed."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "decode",
          REPO / "scripts" / "scoring" / "decode" / "d6_build"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import d4common as D4  # noqa: E402
import d5common as D5  # noqa: E402
import d6common as D6  # noqa: E402
import d6v  # noqa: E402
import dcommon as DC  # noqa: E402
import finalise_decode_d6 as FN6  # noqa: E402
import gate_d6 as G6  # noqa: E402
import register_decode_d6 as RG6  # noqa: E402
import score_blind_d6 as SB6  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

RD, V4DP, V5D, V6D = DC.RD, D6.V4DP, D6.V5D, D6.V6D
REG = REPO / "docs" / "registered"
CAL2 = REPO / D4.CAL2_PUBLISHED / "results"
BLIND = REPO / D4.BLIND_PUBLISHED / "results"


def _sha(p) -> str:
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def _reg() -> dict:
    return json.loads((REG / f"{D6.NAME}.json").read_text())


def _trees():
    if not CAL2.is_dir() or not BLIND.is_dir():
        pytest.skip("the CAL2 or decode blind session is not published in this checkout")


@pytest.fixture(scope="module")
def rdir(tmp_path_factory) -> Path:
    return G6.readings_dir_committed(tmp_path_factory.mktemp("readings"))


@pytest.fixture(scope="module")
def gate_doc(rdir):
    _trees()
    return G6.gate(rdir, CAL2, BLIND)


# ---------------------------------------------------------------- registration and freeze

def test_the_addendum_recomputes_to_its_committed_files(capsys):
    assert RG6.main(["--check"]) == 0, capsys.readouterr().out


def test_the_addendum_pins_every_earlier_decode_file_and_its_own_code_at_their_bytes():
    pins = _reg()["code_pins"]
    prev = json.loads((REG / f"{D5.NAME}.json").read_text())["code_pins"]
    assert set(prev) <= set(pins)
    for p in tuple(RG6.FROZEN) + tuple(RG6.D6_CODE) + tuple(RG6.BUILD):
        assert p in pins, p
    for p, s in pins.items():
        assert _sha(p) == s, p
    for p, s in prev.items():
        assert pins[p] == s, p


def test_the_freeze_record_hashes_the_model_and_its_constants_before_the_gate():
    t = (REPO / RG6.WORK / "FROZEN.txt").read_text().splitlines()
    assert t[0].startswith(f"v6d (addendum 6) FROZEN {RG6.FREEZE_TS}") and "before gate_d6 ran" in t[0]
    assert RG6.FREEZE_TS < RG6.GATE_TS
    got = dict(reversed(ln.split(None, 1)) for ln in t if re.match(r"^[0-9a-f]{64}  ", ln))
    for path, sha in got.items():
        path = path.split(" (")[0]
        assert _sha(path) == sha, path
    assert {p.split(" (")[0] for p in got} == {"scripts/r3_timing_model_v6d.py", f"{RG6.WORK}/v6d_constants.json",
                                               f"{RG6.WORK}/pc_spans.json", "scripts/scoring/decode/d6common.py"}
    c = json.loads((REPO / RG6.WORK / "v6d_constants.json").read_text())
    assert D6.same_law(c["law"], D6.registered_law()) == []
    assert c["terms (v5d's, regimes and W1 emptied)"] == D6.registered_terms().as_dict()


def test_the_earlier_decode_models_and_scorers_are_untouched():
    pins = json.loads((REG / f"{D5.NAME}.json").read_text())["code_pins"]
    for p in ("scripts/r3_timing_model_v5d.py", "scripts/r3_timing_model_v4dp.py", "scripts/scoring/decode/d5common.py",
              "scripts/scoring/decode/gate_d5.py", "scripts/scoring/decode/score_blind_d3.py",
              "scripts/scoring/decode/score_cal_d3.py", "scripts/decode_synth.py"):
        assert _sha(p) == pins[p], p


def test_the_law_is_the_readings_by_the_registered_rules():
    pc, pc2, sp = D6.committed()
    assert D6.blocking(pc, pc2, sp) == []
    assert D6.same_law(D6.law_from(pc, pc2, sp), D6.registered_law()) == []
    law = V6D.LAW
    assert law.c0[V6D.T16] == ((8, pc2["PC2-R1"]["R1b F ladders"]["G 8"]["c0_us"]),
                               (128, pc["PC-R2"]["G128 (Mixtral tp8, tp4, tp2, 8x7B)"]["c0_us"]))
    assert law.c0[V6D.T64][0][1] == pc2["PC2-R2"]["R2a c0 per G (F 256 to 14336)"]["G 8"]["c0_us"]
    assert law.x[V6D.T16] == pc2["PC2-R1"]["R1d X_sat"]["X_sat"] and law.x[V6D.T64] == pc["PC-R7"]["X_sat"]
    assert law.n_if == {V6D.T16: 3, V6D.T32A: 2, V6D.T32B: 2, V6D.T64: 2}
    assert law.q(V6D.T16) == 48 * 1024 and law.q(V6D.T64) == 32 * 1024
    for t in (V6D.T32A, V6D.T32B):
        g528 = next(r for r in pc2["PC2-R3"][t]["rows"] if r["G"] == 528 and "-F" not in r["cell"])
        assert law.kap[t][-1] == (g528["n"] / 528, g528["kappa"])
    assert law.kap[V6D.T16] == law.kap[V6D.T64]


def test_the_terms_are_addendum_5s_with_the_regimes_and_w1_emptied():
    t = D6.registered_terms()
    base = D5.registered_terms()
    assert t.regime == {x: (None, None) for x in V6D.TILES} and t.w1_scale == {}
    assert t.call == base.call and t.align == base.align and t.slope == base.slope


# ---------------------------------------------------------------- model v6d

def test_v6d_with_the_law_off_is_v5d_and_keeps_v4d_prime_on_follower_naive_cells():
    red = _reg()["model v6d"]["reduction (COMPUTED)"]
    assert red["verdict"] == "REDUCES" and red["cells"] == red["identical to v5d with the law off"] == 350
    assert red["naive cells with a non-fetching follower"] == red["of them keeping v4d''s naive GEMM prices"] > 0


def test_the_build_law_at_the_candidate_prices_every_gemm_as_the_model():
    lw = d6v.variant()
    worst = 0.0
    for t in V6D.TILES:
        S = 132 * (4 if t.startswith("32") else 3)
        for n in list(range(1, 1700, 37)) + [S - 1, S, S + 1, 2 * S]:
            for b in (65536.0, 262144.0, 1835008.0, 4194304.0):
                for fl in (0.0, 3.0):
                    worst = max(worst, abs(V6D.LAW.gemm_us(t, n, S, b, fl) / lw.gemm_us(t, n, b, fl, S) - 1))
    assert worst < 1e-8


def test_c0_is_linear_in_g_between_its_two_reads_and_a_multi_wave_gemm_runs_at_c0_s():
    law = V6D.LAW
    (g0, a), (g1, b) = law.c0[V6D.T64]
    assert law.c0_at(V6D.T64, 2) == a and law.c0_at(V6D.T64, 500) == b
    assert math.isclose(law.c0_at(V6D.T64, 68), a + (b - a) * 60 / 120)
    S, slab = 396, 65536.0
    k = 52
    lo = law.c0.copy()
    flat = V6D.Law({"C0": {t: ((8, b), (128, b)) for t in lo}})
    # past one wave the remainder is priced at c0(S): the same as a law whose c0 is b everywhere
    assert math.isclose(law.gemm_us(V6D.T64, S + k, S, slab), flat.gemm_us(V6D.T64, S + k, S, slab))
    assert law.gemm_us(V6D.T64, k, S, slab) < flat.gemm_us(V6D.T64, k, S, slab)


def test_kappa_is_read_by_fill_and_held_outside():
    law = V6D.LAW
    pts = law.kap[V6D.T32B]
    assert law.kappa_at(V6D.T32B, 1.0) == pts[-1][1] and law.kappa_at(V6D.T32B, 0.0) == pts[0][1]
    assert law.kappa_at(V6D.T32A, 396 / 528) > 1.09 and law.kappa_at(V6D.T32A, 1.0) < 1.03


def test_a_naive_cell_with_a_non_fetching_follower_keeps_v4d_prime_and_one_without_takes_the_law():
    prm = D4.ds_prm()
    terms = D6.registered_terms()
    m = "olmoe-1b-7b"
    seen = set()
    for c in DC.page_doc(m, "main")["cells"]:
        if RD.geometry(MODEL_CONFIGS[m], int(c["T"]), c["counts"]).path != RD.NAIVE:
            continue
        ids = V4DP.ids_of(m, c)
        a = V5D.price(m, int(c["T"]), c["counts"], prm, ids=ids, terms=terms)
        b = V6D.price(m, int(c["T"]), c["counts"], prm, ids=ids, terms=terms)
        if (b.get("v6d") or {}).get("gemm") is None:
            seen.add("v4d'")
            assert all(b["gemm"][g]["gemm_ms"] == a["gemm"][g]["gemm_ms"] for g in ("w1", "w2"))
        else:
            seen.add("law")
            assert all(b["gemm"][g]["gemm_ms"] == b["v6d"]["gemm"][g]["law_ms"] for g in ("w1", "w2"))
    assert seen == {"law", "v4d'"}


def test_the_build_records_reproduce(capsys):
    import subprocess
    r = subprocess.run([sys.executable, str(REPO / RG6.WORK / "run.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count(": same") == 10


# ---------------------------------------------------------------- the gate

def test_the_gate_reads_no_go_on_clause_2_as_registered(gate_doc):
    assert gate_doc["verdict"] == "NO-GO"
    line = G6.line(gate_doc)
    assert line.startswith("NO-GO: RE-SCORE: the SEEN decode blind under v6d reads pooled rms 4.52%") and "\n" not in line
    reg = _reg()["gate"]["the gate on the published trees (COMPUTED after the freeze)"]
    assert reg["line"] == line
    assert gate_doc["CAL"]["verdict"] == "GO" and gate_doc["CAL"]["E1_CAL_IN_SAMPLE"]["rms"] < 0.03
    assert gate_doc["RESCORE"]["pooled"]["cells"] == D6.RESCORE_CELLS == 220
    assert gate_doc["RESCORE"]["pooled"]["rms"] >= D6.RESCORE_RMS_MAX
    assert set(gate_doc["RESCORE"]["per_model"]) == set(DC.BLIND_MODELS)


def test_the_committed_gate_output_reproduces_on_the_published_trees(gate_doc):
    assert (REPO / RG6.GATE_OUT).read_text() == "\n".join(G6.lines(gate_doc)) + "\n"


def test_the_gate_cli_prints_one_line_and_exits_1_on_no_go(rdir, tmp_path, capsys):
    _trees()
    assert G6.main([str(rdir), str(CAL2), str(BLIND), str(tmp_path)]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("NO-GO: ")
    assert json.loads((tmp_path / "gate.json").read_text())["verdict"] == "NO-GO"


def _edited(rdir: Path, d: Path, name: str, edit) -> Path:
    d.mkdir()
    for f in rdir.iterdir():
        (d / f.name).write_bytes(f.read_bytes())
    doc = json.loads((d / name).read_text())
    edit(doc)
    (d / name).write_text(json.dumps(doc))
    return d


def test_the_gate_refuses_blocking_or_changed_readings_before_any_tree(rdir, tmp_path):
    def unread(doc):
        doc["PC2-R1"]["R1d X_sat"]["status"] = "NOT READ"
    g = G6.gate(_edited(rdir, tmp_path / "a", "pc2.read.json", unread), CAL2, BLIND)
    assert g["verdict"] == "NO-GO" and g["why"][0].startswith("REFUSED: BLOCKING: PC2-R1d X_sat")

    def move(doc):
        doc["PC2-R1"]["R1d X_sat"]["X_sat"] = 3700.0
    g = G6.gate(_edited(rdir, tmp_path / "b", "pc2.read.json", move), CAL2, BLIND)
    assert g["verdict"] == "NO-GO" and "is not the registered v6d" in g["why"][0]

    def other(doc):
        doc["registration"] = "2026-11-06-pc-gh200"
    assert "registered readers" in G6.gate(_edited(rdir, tmp_path / "c", "pc2.read.json", other), CAL2, BLIND)["why"][0]
    assert G6.gate(tmp_path / "missing", CAL2, BLIND)["why"][0].startswith("REFUSED: readings")


def test_the_gate_line_on_go_names_both_clauses_and_the_last_blind():
    g = {"verdict": "GO", "rescore_rms_max": 0.04,
         "CAL": {"E1_CAL_IN_SAMPLE": {"rms": 0.018, "worst": 0.06}}, "RESCORE": {"pooled": {"rms": 0.031, "cells": 220}}}
    line = G6.line(g)
    assert line.startswith("GO: ") and "never a test" in line and "last decode blind" in line


def test_the_finalisation_refuses_on_no_go_and_writes_nothing(rdir, tmp_path, capsys):
    _trees()
    out, plan = tmp_path / "fin.json", tmp_path / "plan.plan"
    assert FN6.main([str(rdir), str(CAL2), str(BLIND), str(out), "--plan-out", str(plan)]) == 2
    assert capsys.readouterr().out.startswith("REFUSED: the launch gate reads NO-GO: RE-SCORE")
    assert not out.exists() and not plan.exists() and not (REPO / D6.BLIND6_PLAN).exists()


def test_the_blind_scorer_refuses_anything_but_a_v6d_finalisation_on_go(tmp_path, capsys):
    fin = {"addendum": D6.NAME, "frozen": {"model": {"name": "v6d", "law": D6.registered_law()}}, "GATE": {"verdict": "NO-GO"}}
    (tmp_path / "fin.json").write_text(json.dumps(fin))
    assert SB6.main([str(tmp_path), str(tmp_path / "fin.json"), str(tmp_path / "out")]) == 2
    assert capsys.readouterr().out.startswith("REFUSED")
    fin["GATE"]["verdict"] = "GO"
    fin["frozen"]["model"]["law"] = dict(D6.registered_law(), X_SAT=dict(D6.registered_law()["X_SAT"], **{V6D.T64: 1.0}))
    (tmp_path / "fin.json").write_text(json.dumps(fin))
    assert SB6.main([str(tmp_path), str(tmp_path / "fin.json"), str(tmp_path / "out")]) == 2
    assert "not the registered v6d" in capsys.readouterr().out
    assert SB6.registration()["pages"]["dir"] == D5.PAGES5


# ---------------------------------------------------------------- the new blind

def test_the_new_blind_reuses_addendum_5s_pages_units_bars_and_rental_and_is_not_run():
    reg = _reg()
    b = reg[SB6.BLOCK]
    assert b["pages"]["files"] == json.loads((REG / f"{D5.NAME}.json").read_text())[
        "blind (owner, 2026-11-04, D4-1; registered in addendum 5)"]["pages"]["files"]
    assert b["units"] == D5.blind_units() and b["plan_units"] == D5.blind_plan_units()
    assert {k: b["bars"][k] for k in ("rms_max", "max_abs", "mean_max")} == {"rms_max": 0.03, "max_abs": 0.08, "mean_max": 0.015}
    r = reg["rentals"]["new blind"]
    assert (r["units_min"], r["wall_min"], r["cost_usd"], r["hard_cap_after_launch_min"], r["cost_at_hard_cap_usd"]) == (
        72, 111, 4.24, 140, 5.34)
    assert b["status"].startswith("NOT RUN") and reg["rentals"]["status"].startswith("NOT RUN")
    assert not (REPO / D6.BLIND6_PLAN).exists() and not (REPO / D5.BLIND5_PLAN).exists()
    for m in D6.NEW_BLIND:
        assert D4.no_decode_page(m)["verdict"] == "NO DECODE PAGE", m
    assert "the last decode round" in reg["model v6d"]["forking paths"]["the last round"]


def test_the_registration_discloses_every_forking_path_and_the_post_gate_defect():
    m = _reg()["model v6d"]
    d = m["decisions"]
    for k in ("D6-18", "D6-19", "D6-20", "D6-21", "D6-22", "D6-23", "D6-24"):
        assert any(x.startswith(k) for x in d), k
    assert "THE ONE PLACE THE BRIEF" in next(v for k, v in d.items() if k.startswith("D6-20"))
    assert "naive_nofollow6" in next(v for k, v in d.items() if k.startswith("D6-24"))
    f = m["forking paths"]
    assert "no duty 3.99%" in " ".join(f["design-D6's post-freeze variants, not taken (re-scores POST-HOC, design-D6 section 6.4)"])
    assert "were not opened" in f["the old decode blind was seen"]
    a = _reg()["after the gate"]
    assert "NOT fixed" in next(k for k in a if k.startswith("a defect"))


# ---------------------------------------------------------------- house rules

def test_no_em_dash_and_no_trailer_in_the_addendum_files():
    paths = [REPO / p for p in RG6.D6_CODE + RG6.BUILD if not p.endswith(".json")] + [REG / f"{D6.NAME}.{x}" for x in ("json", "txt")]
    paths += [Path(__file__)]
    for p in paths:
        t = p.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, p
        assert not re.search("|".join(("Co-" + "Authored-By", "Claude " + "Code", "Generated " + "with", "Anth" + "ropic")), t), p
