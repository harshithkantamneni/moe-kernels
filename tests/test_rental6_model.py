"""Rental 6's model v3 off the GPU: term (b) in scripts/r3_timing_model_v3.py, MODEL M4 (content
keys on histogram cells) in scripts/wave_split_bytes_v3.py, the rental-6 draws (skew_synth
--draw6) and the four registrations' --check (docs/registered/2026-10-09-rental6-*). The pinned
rental-5 modules stay byte-identical to rental5-skew's code_pins (owner decision 2026-10-09)."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
R6DIR = REPO / "scripts" / "scoring" / "rental6"
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "rental5", R6DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cores_heldout_predict as CP  # noqa: E402
import cross_model_predict as CMP  # noqa: E402
import hashlib  # noqa: E402

import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v3 as TV3  # noqa: E402
import r6model as M6  # noqa: E402
import skew_synth as K  # noqa: E402
import wave_split_bytes as W  # noqa: E402
import wave_split_bytes_v3 as WV3  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

REG = REPO / "docs" / "registered"


@pytest.fixture(scope="module")
def card():
    return W.load_card(CP.C27)


def _geom(card, model, copies=9):
    src, tgt = MODEL_CONFIGS[card.geom.model], MODEL_CONFIGS[model]
    same = (src.num_experts, src.top_k) == (tgt.num_experts, tgt.top_k)
    g = CMP.target_geometry(card.geom, model, None if same else CMP.target_design(model, 32))
    decl = dict(g.declared)
    decl["shared"] = decl["private"] = tgt.num_experts * copies
    return dataclasses.replace(g, copies_declared=copies, declared=tuple(sorted(decl.items())))


def _cell(arm="shared", path=TM.BLOCK_SCAN):
    return TM.Cell(arm=arm, G=8, n=2, ms=float("nan"), mhz=None, path=path, declared=72, runs=(), source="t",
                   reads=None, sigma={"w1": 1.0, "w2": 1.0}, fit=False)


# --------------------------------------------------------------------------
# the pinned modules are untouched
# --------------------------------------------------------------------------

def test_the_files_rental5_pins_are_byte_identical_to_its_registered_sha256():
    pins = json.loads((REG / "2026-10-07-rental5-skew-gh200.json").read_text())["code_pins"]
    for f in ("scripts/r3_timing_model.py", "scripts/wave_split_bytes.py", "scripts/scoring/rental5/skewmodel.py"):
        assert hashlib.sha256((REPO / f).read_bytes()).hexdigest() == pins[f], f
    assert not hasattr(TM, "V3_H_MS") and not hasattr(TM, "timing_version")


# --------------------------------------------------------------------------
# term (b)
# --------------------------------------------------------------------------

def test_term_b_only_where_native_takes_block_scan_and_h_only_on_shared():
    A = 0.0004
    assert TV3.V3_H_MS == pytest.approx(-0.00516)
    assert TV3.v3_b_ms("native", TM.BLOCK_SCAN, A) == 0.0
    assert TV3.v3_b_ms("shared", TM.SMALL_BATCH, A) == 0.0
    assert TV3.v3_b_ms("shared", TM.BLOCK_SCAN, A) == pytest.approx(A + TV3.V3_H_MS)
    assert TV3.v3_b_ms("private", TM.BLOCK_SCAN, A) == pytest.approx(A)
    with pytest.raises(TM.Refused):
        TV3.v3_b_ms("shared", TM.BLOCK_SCAN, None)


def test_v3_call_ms_is_v2s_plus_term_b():
    ctx = TM.Context(card="t", device="t", sms=132, sms_source="t", occupancy={"w1": 5, "w2": 4}, occupancy_source="t",
                     clock_mhz=1710.0, locked=True, bandwidth_gbps=3600.0, byte_label="t")
    x = np.array([0.0432, 202.5, 3597.8, 0.013, -0.00707])
    old = TM.set_model("mixtral-8x7b")
    try:
        with TM.dead_model("v2"):
            a = TM.call_ms(x, _cell(), ctx, 0.5)
            c = TV3.call_ms(x, _cell(), ctx, 0.5, native_path=TM.BLOCK_SCAN, align_ms=0.003)
            d = TV3.call_ms(x, _cell(), ctx, 0.5, native_path=TM.SMALL_BATCH, align_ms=0.003)
        assert c == pytest.approx(a + 0.003 + TV3.V3_H_MS) and d == a
    finally:
        TM.set_model(old)


# --------------------------------------------------------------------------
# MODEL M4: content keys on histogram cells
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model,n", [("olmoe-1b-7b", 3), ("mixtral-8x7b", 3), ("phi-3.5-moe", 2)])
def test_m4_on_the_uniform_histogram_equals_content_events_on_the_balanced_cell(card, model, n):
    g = _geom(card, model)
    prm = W.analyse(card).params(W.REGISTERED_VIEW)
    E = MODEL_CONFIGS[model].num_experts
    uni = tuple([n * 32] * E)
    bal = W.Model(g, content_a=True).evaluate(prm, [("shared", 8, n, "w1")])[0]["q"]
    m4 = WV3.ModelV3(g)
    for seed in (None, 20261010):
        q = m4.evaluate(prm, [("shared", 8, n, "w1", uni, seed)])[0]["q"]
        assert q == pytest.approx(bal, rel=1e-12), (seed, q, bal)
    ev_b = W.Model(g, content_a=True).events("shared", 8, n, "w1")
    ev_u = m4.events("shared", 8, n, "w1", uni, 20261010)
    assert ev_u.slabs == pytest.approx(ev_b.slabs)
    assert np.allclose(ev_u.a_D, ev_b.a_D) and np.allclose(ev_u.a_k, ev_b.a_k) and np.allclose(ev_u.a_ws, ev_b.a_ws)


def test_m4_on_a_skewed_cell_moves_q_only_through_partial_tiles(card):
    g = _geom(card, "olmoe-1b-7b")
    prm = W.analyse(card).params(W.REGISTERED_VIEW)
    doc = json.loads((REG / "2026-10-09-rental6-skew-hist" / "olmoe-1b-7b-A.json").read_text())
    pt = tuple(next(c for c in doc["cells"] if c["label"] == "PT" and c["n"] == 6)["counts"])
    base = W.Model(g).evaluate(prm, [("shared", 8, 6, "w1", pt)])[0]["q"]
    # the pinned model with content_a still keys a histogram cell by M-tile
    assert W.Model(g, content_a=True).evaluate(prm, [("shared", 8, 6, "w1", pt)])[0]["q"] == base
    q4 = WV3.ModelV3(g).evaluate(prm, [("shared", 8, 6, "w1", pt, doc["shuffle_seed"])])[0]["q"]
    assert q4 != base and q4 == pytest.approx(base, rel=2e-3)


def test_the_rental6_pages_recompute_byte_for_byte_and_rental5s_are_untouched():
    assert K.main(["--check6"]) == 0
    assert K.main(["--check"]) == 0


def test_the_draw_seeds_and_phis_carried_shapes():
    assert K.draw6_seed("phi-3.5-moe", 12, "PW2") == 20261010 + 2000 + 120 + 3
    fit = json.loads(K.FIT.read_text())
    kind, prm, rule = K.shape_param6(fit, "mixtral-8x7b", "PW", K.tokens_for("phi-3.5-moe", 4))
    assert (kind, prm) == ("dirichlet", 5.0) and "1024" in rule    # Phi n 4 is B 1024: Mixtral's fitted point
    doc = json.loads((REG / "2026-10-09-rental6-skew-hist" / "phi-3.5-moe-A.json").read_text())
    assert {c["provenance"]["source_model"] for c in doc["cells"] if c.get("provenance")} == {"mixtral-8x7b"}
    assert not any(c["label"].startswith("PW-") for c in doc["cells"]), "no permutation cells (owner decision 6)"


def test_the_proxy_reads_a_at_its_tread_and_extrapolates_past_nine():
    tab = {"m|72": {"shared": {str(n): float(n) for n in range(1, 10)}}}
    assert M6.proxy_A_us(tab, "mixtral-8x7b", "native", 3, 9)[0] == 0.0
    assert M6.proxy_A_us({"mixtral-8x7b|72": tab["m|72"]}, "mixtral-8x7b", "shared", 16, 9)[0] == pytest.approx(16.0)
    a, how = M6.proxy_A_us({"mixtral-8x7b|72": tab["m|72"]}, "mixtral-8x7b", "shared", 4, 15)
    assert a == 4.0 and how.startswith("c9 table")


def test_reprice_moves_only_block_scan_non_native_cells():
    assert M6.reprice(1.0, 1.0, 3.0, "shared", TM.BLOCK_SCAN) == pytest.approx(1.002)
    assert M6.reprice(1.0, 1.0, 3.0, "shared", TM.SMALL_BATCH) == 1.0
    assert M6.reprice(1.0, 1.0, 3.0, "shared", TM.SMALL_BATCH, "v3_all") == pytest.approx(1.002)
    assert M6.reprice(1.0, 1.0, 3.0, "native", TM.BLOCK_SCAN) == 1.0
    assert M6.reprice(1.0, 1.0, None, "shared", TM.BLOCK_SCAN) == 1.0


# --------------------------------------------------------------------------
# the registrations: the design's SEEN numbers, and --check
# --------------------------------------------------------------------------

def test_v3_reproduces_the_designs_seen_numbers():
    d = json.loads((REG / "2026-10-09-rental6-v3-gh200.json").read_text())
    s = d["seen_reproduction"]
    assert s["cells"] == 72
    v2, v3, h8 = (s["rental5_E1_SKEW_72"][k] for k in ("v2", "v3", "H8"))
    assert (round(100 * v2["rms"], 2), round(100 * v2["mean"], 2), round(100 * v2["worst"], 2)) == (1.26, 0.40, 4.38)
    assert (round(100 * v3["rms"], 2), round(100 * v3["mean"], 2), round(100 * v3["worst"], 2)) == (0.90, 0.09, 3.08)
    assert round(100 * h8["rms"], 2) == 1.40
    assert round(100 * s["cal_8x7b_fitted_cells_v3"]["rms"], 3) == 0.249
    assert d["h"]["value_us"] == -5.16 and d["h"]["label"].startswith("SEEN-FITTED")
    assert round(d["h"]["rival_H8"]["h_us"], 2) == -0.65


def test_every_seen_informed_choice_is_stated():
    d = json.loads((REG / "2026-10-09-rental6-v3-gh200.json").read_text())
    text = " ".join(d["seen_informed_choices"]) + json.dumps(d["h"])
    for must in ("SEEN-FITTED", "SEEN-INFORMED", "SEEN-SELECTED", "n = 1", "IN-SAMPLE", "-8.03"):
        assert must in text, must
    assert "8x22B, Qwen2-57B and OLMoE" in d["h"]["in_sample"]


def test_the_registered_predictions_carry_g3_null_and_the_designs_rival_gaps():
    d = json.loads((REG / "2026-10-09-rental6-skew-gh200.json").read_text())
    assert max(abs(r["delta_v3"]) for r in d["predictions"] if "delta_v3" in r) == 0.0
    g = d["rival_gaps_on_E1"]
    assert round(g["v2"]["rms_pct"], 2) == 0.22 and round(g["LT3"]["rms_pct"], 2) == 0.97 and round(g["H8"]["rms_pct"], 2) == 0.35
    assert all(r["host_ok"] for r in d["predictions"])
    assert not any(r["model"] == "mixtral-8x7b" for r in d["byte_predictions"]), "Mixtral's byte page is dropped"


def test_q1_jetmoe_small_batch_is_host_bound_and_q1p_is_descriptive():
    d = json.loads((REG / "2026-10-09-rental6-q1-gh200.json").read_text())
    assert sorted(d["tests"]["Q1-P"]["host_bound_excluded"]) == sorted(
        f"jetmoe-8b/c{c}/n{n}" for c in (9, 15) for n in (1, 2))
    assert d["tests"]["Q1-P"]["status"] == "DESCRIPTIVE"
    assert all(p["label"] in ("uniform", "balanced") for p in d["tests"]["Q1-H"]["pairs"])
    assert {(p["model"], p["n"]) for p in d["tests"]["Q1-H"]["pairs"]} == {
        ("phi-3.5-moe", 2), ("olmoe-1b-7b", 3), ("qwen1.5-moe-a2.7b", 2), ("jetmoe-8b", 4)}
    assert d["power"]["Q1-H | replicate sigma | v3 true"]["P(v3 HOLDS)"] > 0.95
    assert d["power"]["Q1-H | replicate sigma | v2 true"]["P(v3 HOLDS)"] < 0.01


def test_the_registrations_recompute():
    got = subprocess.run([sys.executable, str(R6DIR / "register.py"), "--check"], capture_output=True, text=True,
                         cwd=REPO, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    assert got.returncode == 0, got.stdout + got.stderr
