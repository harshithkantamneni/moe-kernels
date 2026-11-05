"""The decode stage's fifth addendum (docs/registered/2026-11-04-decode-addendum5-gh200), off the GPU: model v5d, its
gate and the new blind. The registration recomputes to its committed files and pins every earlier decode file at its
bytes; v5d with every term off is v4d' on every registered page cell, and its terms are DS's committed readings by
the registered rules (the regimes per tile, the w1 scale on the larger tiles, the CALL rule, the align points and the
slope), never refitted after them. The GEMM terms act on the aligned paths only. The gate prints one line, reads
NO-GO on the published trees on both clauses (as registered), refuses readings that are BLOCKING or that do not give
the registered terms, and the finalisation refuses unless it reads GO; the blind scorer refuses anything else. The new
blind's pages are decode_synth's frozen generators, its plan is not written before a GO, its rental and bars are the
registered ones, and no page of a new model has been timed."""
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
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "decode"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import d3common as D3  # noqa: E402
import d4common as D4  # noqa: E402
import d5common as D5  # noqa: E402
import dcommon as DC  # noqa: E402
import finalise_decode_d5 as FN5  # noqa: E402
import gate_d5 as G5  # noqa: E402
import register_decode_d5 as RG5  # noqa: E402
import score_blind_d5 as SB5  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

RD, V4DP, V5D = DC.RD, D3.V4DP, D5.V5D
REG = REPO / "docs" / "registered"
CAL2 = REPO / D4.CAL2_PUBLISHED / "results"
BLIND = REPO / D4.BLIND_PUBLISHED / "results"
READINGS_DIR = REPO / Path(D5.DS_READINGS_COMMITTED).parent


def _sha(p) -> str:
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def _reg() -> dict:
    return json.loads((REG / f"{D5.NAME}.json").read_text())


def _trees():
    if not CAL2.is_dir() or not BLIND.is_dir():
        pytest.skip("the CAL2 or decode blind session is not published in this checkout")


def _counts(model: str, T: int, A: int) -> list[int]:
    import decode_synth as SY
    return SY.balanced_over(model, T, list(range(A)))


@pytest.fixture(scope="module")
def terms() -> V5D.Terms:
    return D5.registered_terms()


@pytest.fixture(scope="module")
def gate_doc():
    _trees()
    return G5.gate(READINGS_DIR, CAL2, BLIND)


# ---------------------------------------------------------------- registration

def test_the_addendum_recomputes_to_its_committed_files(capsys):
    assert RG5.main(["--check"]) == 0, capsys.readouterr().out


def test_the_addendum_pins_every_earlier_decode_file_and_its_own_code_at_their_bytes():
    pins = _reg()["code_pins"]
    prev = json.loads((REG / f"{D4.NAME}.json").read_text())["code_pins"]
    assert set(prev) <= set(pins)
    for p in tuple(RG5.FROZEN) + tuple(RG5.D5_CODE) + tuple(RG5.BUILD):
        assert p in pins, p
    for p, s in pins.items():
        assert _sha(p) == s, p


def test_the_earlier_decode_modules_and_models_are_untouched():
    pins = json.loads((REG / f"{D4.NAME}.json").read_text())["code_pins"]
    for p in ("scripts/r3_timing_model_v4dp.py", "scripts/r3_timing_model_v4d.py", "scripts/scoring/decode/d3common.py",
              "scripts/scoring/decode/d4common.py", "scripts/scoring/decode/score_blind_d3.py",
              "scripts/scoring/decode/score_cal_d3.py", "scripts/decode_synth.py", "scripts/gh200_decode_session.sh"):
        assert _sha(p) == pins[p], p
    assert math.isclose(V4DP.W1_SCALE, 0.9184877841384885)


def test_the_registered_terms_are_ds_readings_by_the_registered_rules(terms):
    rd = D5.committed_readings()
    assert D5.blocking(rd) == []
    assert D5.same_terms(D5.terms_from(rd), terms) == []
    for t in DC.DECODE_TILES:
        assert terms.regime[t] == (rd["SUBWAVE"]["tiles"][t]["r_cta_GBps"], rd["DRAIN"]["tiles"][t]["r1_GBps"])
    assert set(terms.w1_scale) == set(D5.LARGER_TILES) and V4DP.W1_TILE not in terms.w1_scale
    for t, s in terms.w1_scale.items():
        assert math.isclose(s * rd["W1RATE"]["tiles"][t]["counter_GBps"], D4.V4D_GBPS)
    assert terms.slope == rd["ALIGN"]["SLOPE"]["slope_us"]
    assert [list(x) for x in terms.align[RD.SMALL_BATCH]] == rd["ALIGN"]["points"][RD.SMALL_BATCH]
    assert [list(x) for x in terms.align[RD.BLOCK_SCAN]] == rd["ALIGN"]["points"][RD.BLOCK_SCAN]
    assert {c: tuple(v) for c, v in terms.call.items()} == {c: tuple(sorted(V5D.CALL_FORMS[c])) for c in V5D.CALL_CLASSES}


def test_the_call_rule_is_ols_on_ds_native_rows_with_its_leave_one_model_out_beside():
    fit = D5.fit_call_rule(D5.call_rows(D5.committed_readings()))
    for x in fit.values():
        assert x["verdict"].startswith("READ") and x["rows"] == 69 and len(x["models"]) == 6
        assert x["lomo_rms_us (DESCRIPTIVE)"] >= x["rms_us"]
    assert fit["silu_and_mul"]["form"] == ["1", "B_MB"] and fit["moe_sum"]["form"] == ["1", "k", "B_MB"]
    few = [r for r in D5.call_rows(D5.committed_readings()) if r["model"] in ("jetmoe-8b", "phi-3.5-moe")]
    assert all(not x["verdict"].startswith("READ") for x in D5.fit_call_rule(few).values())


def test_the_registration_states_the_decisions_the_disclosure_and_the_forking_paths():
    m = _reg()["model v5d"]
    d = m["decisions"]
    assert [k.split(" ")[0] for k in d if k.startswith("D5-")] == [f"D5-{i}" for i in range(1, 9)]
    assert "FORKING PATH" in d["D5-2 the paths"] and "3.52%" in d["D5-2 the paths"]
    assert "forking path on DS's pages" in d["D5-4 CALL"]
    weak = d["D5-7 the three weak DS readings (coordinator, 2026-11-04; COMPUTED in d5_build/weak.py, weak.out)"]
    assert "binds on 0 of the 143 new main cells" in weak and "no refit" in weak
    assert any("diagnoses" in x for x in d["D5-8 choices informed by the diagnosis of the old blind pages (listed as such)"])
    assert m["reduction (COMPUTED)"]["verdict"] == "REDUCES"
    assert "WEAK SANITY FLOOR" in _reg()["gate"]["clause 2 label"]


# ---------------------------------------------------------------- model v5d

def test_v5d_with_every_term_off_is_v4d_prime_exactly():
    prm = D4.ds_prm()
    for m, T, A in (("mixtral-8x7b", 4, 2), ("olmoe-1b-7b", 128, 27), ("granite-3.0-3b-a800m", 64, 16),
                    ("jetmoe-8b", 96, 6)):
        c = _counts(m, T, A)
        assert V5D.price(m, T, c, prm) == V4DP.price(m, T, c, prm), (m, T, A)
    red = D5.reduction(D5.registered_terms())
    assert red["verdict"] == "REDUCES" and red["cells"] == red["identical with every term off"] == 350


def test_v5d_through_its_hook_with_zero_effect_terms_is_v4d_prime_on_the_350_cells():
    """Not the short-circuit: a w1 scale of exactly 1 makes Terms.any() true, so every price runs the regime hook (the
    gemm_ms recorder), the per-GEMM loop, CALL, ALIGN and SLOPE with nothing to add; each must equal v4d' exactly."""
    zero = V5D.Terms(w1_scale={t: 1.0 for t in D5.LARGER_TILES})
    assert zero.any()
    prm = D4.ds_prm()
    n = 0
    for m in DC.CAL_MODELS + DC.BLIND_MODELS:
        cfg = MODEL_CONFIGS[m]
        for c in DC.page_doc(m, "main")["cells"]:
            ids = V4DP.ids_of(m, c) if RD.geometry(cfg, int(c["T"]), c["counts"]).path == RD.NAIVE else None
            try:
                a = V4DP.price(m, int(c["T"]), c["counts"], prm, ids=ids)
            except V4DP.TM.Refused:
                continue
            b = V5D.price(m, int(c["T"]), c["counts"], prm, ids=ids, terms=zero)
            assert "v5d" in b, (m, c["label"])
            assert (b["body_ms"], b["bytes"], b["path"]) == (a["body_ms"], a["bytes"], a["path"]), (m, c["label"])
            assert all(b["gemm"][g]["gemm_ms"] == a["gemm"][g]["gemm_ms"] for g in ("w1", "w2")), (m, c["label"])
            n += 1
    assert n == 350


@pytest.mark.slow
def test_the_build_records_reproduce(capsys):
    """scripts/scoring/decode/d5_build: every committed .out (the alternatives, the every-path switch, the ablation, the
    call-form search, the D5-7 sensitivity, the ladder excesses) recomputes byte for byte (about 45 s, 0.3 GB)."""
    _trees()
    import subprocess
    r = subprocess.run([sys.executable, str(REPO / RG5.WORK / "run.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count(": same") == 6


def test_the_regimes_price_a_gemm_at_bandwidth_with_the_per_cta_ceiling_and_the_drain(terms):
    prm = D4.ds_prm()
    p = V5D.price("jetmoe-8b", 4, _counts("jetmoe-8b", 4, 2), prm, terms=terms.only("regime"))
    w2 = p["v5d"]["gemm"]["w2"]
    r_cta, r1 = terms.regime["16x64x128w4s4"]
    assert w2["n"] == 64 and w2["S"] == 396 and w2["ceiling_binds"]
    assert math.isclose(w2["ceiling_ms"], w2["b_bytes"] / (r_cta * 1e6)) and w2["drain_ms"] == 0.0
    assert math.isclose(p["gemm"]["w2"]["gemm_ms"], w2["ceiling_ms"])
    assert p["v5d"]["gemm"]["w1"]["n"] == 352 and p["v5d"]["gemm"]["w1"]["drain_ms"] == 0.0     # one wave: no drain
    p3 = V5D.price("jetmoe-8b", 4, _counts("jetmoe-8b", 4, 3), prm, terms=terms.only("regime"))
    w1 = p3["v5d"]["gemm"]["w1"]
    assert w1["n"] == 528 > w1["S"] and math.isclose(w1["drain_ms"], w1["b_bytes"] / (r1 * 1e6))
    assert math.isclose(p3["gemm"]["w1"]["gemm_ms"], max(w1["bw_ms"], w1["ceiling_ms"]) + w1["drain_ms"])
    q = V4DP.price("jetmoe-8b", 4, _counts("jetmoe-8b", 4, 2), prm)
    assert p["gemm"]["w2"]["gemm_ms"] < q["gemm"]["w2"]["gemm_ms"]          # the CORES rho tail is gone


def test_w1_scales_only_its_bandwidth_part_on_a_larger_tile(terms):
    prm = D4.ds_prm()
    c = _counts("mixtral-8x7b", 128, 4)
    a = V5D.price("mixtral-8x7b", 128, c, prm, terms=terms.only("regime"))
    b = V5D.price("mixtral-8x7b", 128, c, prm, terms=terms.only("regime", "w1_scale"))
    s = terms.w1_scale["64x128x64w4s3"]
    x, y = a["v5d"]["gemm"]["w1"], b["v5d"]["gemm"]["w1"]
    assert math.isclose(y["bw_ms"], s * x["bw_ms"]) and y["drain_ms"] == x["drain_ms"] and y["ceiling_ms"] == x["ceiling_ms"]
    assert b["v5d"]["gemm"]["w2"] == a["v5d"]["gemm"]["w2"]
    t = V5D.price("mixtral-8x7b", 4, _counts("mixtral-8x7b", 4, 2), prm, terms=terms.only("w1_scale"))
    assert t["v5d"]["w1_scale"] is None                                         # fix R's tile keeps fix R


def test_the_naive_path_keeps_v4d_primes_gemm_prices_and_takes_call_but_no_align(terms):
    prm = D4.ds_prm()
    m, cell = "olmoe-1b-7b", {"T": 1, "counts": _counts("olmoe-1b-7b", 1, 8), "shuffle": True}
    ids = V4DP.ids_of(m, cell)
    a = V4DP.price(m, 1, cell["counts"], prm, ids=ids)
    b = V5D.price(m, 1, cell["counts"], prm, ids=ids, terms=terms)
    assert a["path"] == RD.NAIVE and not b["v5d"]["aligned"]
    assert all(b["gemm"][g]["gemm_ms"] == a["gemm"][g]["gemm_ms"] for g in ("w1", "w2"))
    assert b["v5d"]["align_ms"] == 0.0 and b["v5d"]["slope_ms"] == 0.0
    assert math.isclose(b["body_ms"] - a["body_ms"], b["v5d"]["call_ms"] - a["other_ms"])


def test_align_is_measured_interpolated_or_held_never_extrapolated(terms):
    sb, bs = RD.SMALL_BATCH, RD.BLOCK_SCAN
    assert V5D.align_us(terms, sb, 32) == 7.264 and V5D.align_us(terms, sb, 64) == 11.632
    assert math.isclose(V5D.align_us(terms, sb, 48), 7.776 + (9.184 - 7.776) * 8 / 20)
    assert V5D.align_us(terms, sb, 4) == 4.752 and V5D.align_us(terms, sb, 128) == 11.632
    assert V5D.align_us(terms, bs, 32) == 8.704 and V5D.align_us(terms, bs, 64) == 8.128
    assert V5D.align_us(terms, RD.NAIVE, 32) == 0.0
    assert V5D.iterations(96 * 8, 32) == 24 and V5D.iterations(8, 64) == 1


def test_call_us_is_the_registered_form_at_the_models_shape(terms):
    cfg = MODEL_CONFIGS["qwen2-57b-a14b"]
    T = 64
    want = 0.0
    for cls in V5D.CALL_CLASSES:
        x = V5D.call_regressors(cfg, T, cls)
        want += sum(c * x[q] for q, c in terms.call[cls].items())
    assert math.isclose(V5D.call_us(terms, cfg, T), want)
    assert math.isclose(V5D.call_regressors(cfg, T, "silu_and_mul")["B_MB"], T * 8 * 3 * 2560 * 2 / 1e6)
    assert math.isclose(V5D.call_regressors(cfg, T, "moe_sum")["B_MB"], (T * 8 * 3584 + T * 3584) * 2 / 1e6)
    assert V5D.call_us(V5D.NONE, cfg, T) is None


def test_terms_round_trip_and_only():
    t = D5.registered_terms()
    assert V5D.Terms.from_dict(t.as_dict()) == t
    assert not V5D.Terms().any() and t.only().as_dict() == V5D.NONE.as_dict()
    assert t.only("align").slope == t.slope and not t.only("call").slope


# ---------------------------------------------------------------- the gate

def test_the_gate_reads_no_go_on_both_clauses_as_registered(gate_doc):
    assert gate_doc["verdict"] == "NO-GO"
    line = G5.line(gate_doc)
    assert line.startswith("NO-GO: CAL: CAL2 under v5d reads STOP") and "RE-SCORE" in line and "\n" not in line
    reg = _reg()["gate"]["the gate on the published trees at registration (COMPUTED)"]
    assert reg["line"] == line
    assert gate_doc["CAL"]["E1_CAL_IN_SAMPLE"]["worst"] < -D5.BARS_OWNER["max_abs"]
    assert gate_doc["RESCORE"]["pooled"]["cells"] == D5.RESCORE_CELLS == 220
    assert gate_doc["RESCORE"]["pooled"]["rms"] >= D5.RESCORE_RMS_MAX
    assert set(gate_doc["RESCORE"]["per_model"]) == set(DC.BLIND_MODELS)


def test_the_committed_gate_output_reproduces_on_the_published_trees(gate_doc):
    """scripts/scoring/decode/gate_d5.out.txt is gate_d5.lines on the published trees, recomputed here."""
    assert (REPO / RG5.GATE_OUT).read_text() == "\n".join(G5.lines(gate_doc)) + "\n"


def test_the_new_blind_is_registered_and_not_run():
    reg = _reg()
    assert "NOT RUN" in reg["status"] and "calibrated, not certified" in reg["decision (owner, 2026-11-04)"]
    assert reg[SB5.BLOCK]["status"].startswith("NOT RUN") and reg["rentals"]["status"].startswith("NOT RUN")
    assert not (REPO / D5.BLIND5_PLAN).exists()
    t = (REG / f"{D5.NAME}.txt").read_text()
    for phrase in ("unless the owner", "only on GO", "if the owner launched", "cost if launched"):
        assert phrase not in t, phrase


def test_the_gate_cli_prints_one_line_and_exits_1_on_no_go(tmp_path, capsys):
    _trees()
    assert G5.main([str(READINGS_DIR), str(CAL2), str(BLIND), str(tmp_path)]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("NO-GO: ")
    assert json.loads((tmp_path / "gate.json").read_text())["verdict"] == "NO-GO"
    assert (tmp_path / "gate.txt").read_text().splitlines()[0] == out[0]


def _readings_dir(d: Path, edit) -> Path:
    rd = copy.deepcopy(D5.committed_readings())
    edit(rd)
    (d / D5.DS_READINGS).write_text(json.dumps(rd))
    return d


def test_the_gate_refuses_blocking_or_changed_readings_before_any_tree(tmp_path):
    def drop(rd):
        rd["DRAIN"]["tiles"]["32x64x128w4s3"] = {"verdict": "NOT IDENTIFIED: 1 mixed ladders from 1 models"}
    (tmp_path / "a").mkdir()
    g = G5.gate(_readings_dir(tmp_path / "a", drop), CAL2, BLIND)
    assert g["verdict"] == "NO-GO" and g["why"][0].startswith("REFUSED: BLOCKING: DRAIN 32x64x128w4s3")

    def move(rd):
        rd["DRAIN"]["tiles"]["16x64x128w4s4"]["r1_GBps"] = 95.0
    (tmp_path / "b").mkdir()
    g = G5.gate(_readings_dir(tmp_path / "b", move), CAL2, BLIND)
    assert g["verdict"] == "NO-GO" and "is not the registered v5d" in g["why"][0]

    def other(rd):
        rd["addendum"] = "2026-10-29-decode-addendum3-gh200"
    (tmp_path / "c").mkdir()
    assert "not written by read_ds.py" in G5.gate(_readings_dir(tmp_path / "c", other), CAL2, BLIND)["why"][0]
    assert "no " in G5.gate(tmp_path / "missing", CAL2, BLIND)["why"][0]


def test_the_gate_line_on_go_names_both_clauses_and_the_weak_floor():
    g = {"verdict": "GO", "rescore_rms_max": 0.04,
         "CAL": {"E1_CAL_IN_SAMPLE": {"rms": 0.021, "worst": -0.05}}, "RESCORE": {"pooled": {"rms": 0.031, "cells": 220}}}
    line = G5.line(g)
    assert line.startswith("GO: ") and "weak sanity floor on seen pages, never a test" in line and "220 cells" in line


def test_the_finalisation_refuses_on_no_go_and_writes_nothing(tmp_path, capsys):
    _trees()
    out = tmp_path / "fin.json"
    plan = tmp_path / "plan.plan"
    assert FN5.main([str(READINGS_DIR), str(CAL2), str(BLIND), str(out), "--plan-out", str(plan)]) == 2
    assert capsys.readouterr().out.startswith("REFUSED: the launch gate reads NO-GO: CAL:")
    assert not out.exists() and not plan.exists() and not (REPO / D5.BLIND5_PLAN).exists()


def test_the_blind_scorer_refuses_anything_but_a_v5d_finalisation_on_go(tmp_path, capsys):
    fin = {"addendum": D5.NAME, "frozen": {"model": {"name": "v5d"}}, "GATE": {"verdict": "NO-GO"}}
    (tmp_path / "fin.json").write_text(json.dumps(fin))
    assert SB5.main([str(tmp_path), str(tmp_path / "fin.json"), str(tmp_path / "out")]) == 2
    assert capsys.readouterr().out.startswith("REFUSED")
    assert SB5.registration()["pages"]["dir"] == D5.PAGES5


# ---------------------------------------------------------------- the new blind

def test_the_new_blind_pages_are_the_frozen_generators_and_untimed():
    import decode_synth as SY
    files = sorted(p.name for p in (REPO / D5.PAGES5).glob("*.json"))
    assert files == sorted(f"{m}-{k}.json" for m, k in D5.PAGE_KEYS)
    cells = {}
    for m, k in D5.PAGE_KEYS:
        doc = json.loads((REPO / D5.page5_path(m, k)).read_text())
        assert doc["cells"] == SY.page(m, k)["cells"], (m, k)
        assert all(c["arms"] == ["native"] for c in doc["cells"])
        if k == "main":
            cells[m] = len(doc["cells"])
    assert cells == {"mixtral-8x22b": 47, "qwen2-57b-a14b": 48, "granite-3.0-1b-a400m": 48}
    for m in D5.NEW_BLIND:
        assert D4.no_decode_page(m)["verdict"] == "NO DECODE PAGE", m


def test_the_blind_is_registered_with_the_owners_bars_rental_and_plan_lines():
    b = _reg()[SB5.BLOCK]
    assert {k: b["bars"][k] for k in ("rms_max", "max_abs", "mean_max")} == {"rms_max": 0.03, "max_abs": 0.08, "mean_max": 0.015}
    r = _reg()["rentals"]["new blind"]
    assert (r["units_min"], r["wall_min"], r["cost_usd"], r["deadline_after_launch_min"], r["hard_cap_after_launch_min"],
            r["cost_at_hard_cap_usd"]) == (72, 111, 4.24, 125, 140, 5.34)
    assert "2.29" in r["rate"]
    assert b["plan_units"] == D5.blind_plan_units() and len(b["plan_units"]) == 7
    assert all(D5.PAGES5 in ln for ln in b["plan_units"][2:])
    assert not (REPO / D5.BLIND5_PLAN).exists()
    hdr = D5.blind_plan({"constants": {"T0_GR_ms": 0.008, "s_naive_GR_ms": None, "s_block_GR_ms": -0.002},
                         "rho_form": "fixed", "bars": D5.BARS_OWNER}).splitlines()
    assert any("gate_d5.py prints GO" in ln for ln in hdr) and any("T0_GR 8.00 us" in ln for ln in hdr)


def test_gd5_offline_names_the_lookups_and_reads_a_vllm_tree_when_present(tmp_path):
    assert D5.gd5_file("granite-3.0-1b-a400m") == "E=32,N=512,device_name=NVIDIA_GH200_480GB.json"
    reg = _reg()["new blind design"]["new models"]["GD5 offline (tuned config)"]
    assert reg["verdict"].startswith("PASS offline") and reg == json.loads(json.dumps(D5.GD5_REGISTERED))
    (tmp_path / "E=64,N=2560,device_name=NVIDIA_GH200_480GB.json").write_text("{}")
    (tmp_path / "E=64,N=2560,device_name=NVIDIA_H200.json").write_text("{}")
    got = D5.gd5_offline({"fake": tmp_path})["fake"]
    assert got["qwen2-57b-a14b"].startswith("TUNED FILE PRESENT") and got["mixtral-8x22b"].startswith("NO TUNED FILE")


def test_the_new_models_reach_names_every_align_rule():
    reach = _reg()["new blind design"]["new models"]["reach under v5d (COMPUTED)"]
    assert reach["mixtral-8x22b"]["align"]["small-batch"]["rule"] == "MEASURED point"
    assert reach["qwen2-57b-a14b"]["align"]["block-scan"]["rule"] == "MEASURED point"
    assert reach["granite-3.0-1b-a400m"]["align"]["small-batch"]["rule"] == "MEASURED point"
    assert reach["granite-3.0-1b-a400m"]["align"]["block-scan"]["rule"] == "HELD at the nearest measured E"
    assert max(reach["granite-3.0-1b-a400m"]["small-batch iterations"]) == 24


# ---------------------------------------------------------------- house rules

def test_no_em_dash_and_no_trailer_in_the_addendum_files():
    paths = [REPO / p for p in RG5.D5_CODE + RG5.BUILD if not p.endswith(".json")] + [REG / f"{D5.NAME}.{x}" for x in ("json", "txt")]
    paths += [REPO / D5.page5_path(m, k) for m, k in D5.PAGE_KEYS] + [Path(__file__)]
    for p in paths:
        t = p.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, p
        assert not re.search("|".join(("Co-" + "Authored-By", "Claude " + "Code", "Generated " + "with", "Anth" + "ropic")), t), p
