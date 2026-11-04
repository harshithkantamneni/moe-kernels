"""The decode stage's fourth addendum (docs/registered/2026-11-04-decode-addendum4-gh200), off the GPU: a MEASUREMENT
registration only (owner, 2026-11-04, D4-7). The registration recomputes to its committed files, pins every earlier
decode file, states that it registers no model form and when the form will be chosen, discloses that the earlier blind
pages were seen, and records the future gate as intent only; no model, gate, finalisation or blind scorer of this
addendum exists and no new blind page is written. The DS pages hold SEEN models only, every unit is nodrop, and their
ladders identify both regimes (sub-wave few-CTA long-slab cells and the wave-boundary drain) on every decode tile, with
a third DRAIN model on 32x128x64w4s3 and 64x128x64w4s3 from the owner's D4-8 cells; the DS plan, with DS's own tile
census, parses under the D3 driver. On SYNTHETIC trees (decode_fakes' card) whose counters follow the readings' own
forms at a planted truth, every DESCRIPTIVE reading recovers it; the stop rules (POWER, PAGE, the census gate on DS's
own census, CENSUS, and the cross-check against CAL2's census) hold; a missing unit reads NOT IDENTIFIED, never a
STOP. The published CAL2 and decode blind trees reproduce their committed scores."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "decode"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import d3common as D3  # noqa: E402
import d4common as D4  # noqa: E402
import dcommon as DC  # noqa: E402
import decode_fakes as F  # noqa: E402
import read_ds as RDS  # noqa: E402
import register_decode_d4 as RG4  # noqa: E402
import reproduce_d4 as RP4  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

V4, V4D, RD, V4DP = DC.V4, DC.V4D, DC.RD, D3.V4DP
REG = REPO / "docs" / "registered"
#: the planted DS truth: a per-CTA ceiling on the registered grid, a wave-boundary drain at 90 GB/s on every tile, a
#: w1 counter rate per tile, the call kernels on their summary lines, an align kernel linear in E, and the slope
R_CTA = D4.R_CTA_GRID[51]              # 10 x 1.02^50 GB/s, about 26.9
RATE = 90.0
W1C = {"16x64x128w4s4": V4DP.W1_COUNTER_GBPS, "32x64x128w4s3": 3890.0, "32x128x64w4s3": 3760.0, "64x128x64w4s3": 3975.0}
CALL = {"silu_and_mul": (3.0, 1.5e-4, 0.2), "moe_sum": (2.0, 7e-5, 0.22)}
ALIGN = lambda E: 3.9 + 0.11 * E          # noqa: E731
ALIGN_BS = lambda E: 5.0 + 0.04 * E       # noqa: E731
SLOPE = 0.6
POWER_TXT = """GPU 00000000:DD:00.0
    GPU Power Readings
        Current Power Limit               : {gpu} W
    Module Power Readings
        Current Power Limit               : 1000.00 W
"""
REMOVED = ("scripts/r3_timing_model_v4dpp.py", "scripts/scoring/decode/gate_d4.py",
           "scripts/scoring/decode/finalise_decode_d4.py", "scripts/scoring/decode/score_blind_d4.py",
           "scripts/plans/rentalD4-blind-2026-11.plan")


def _sha(p) -> str:
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def _reg() -> dict:
    return json.loads((REG / f"{D4.NAME}.json").read_text())


# ---------------------------------------------------------------- registration

def test_the_addendum_recomputes_to_its_committed_files(capsys):
    assert RG4.main(["--check"]) == 0, capsys.readouterr().out


def test_the_addendum_pins_every_earlier_decode_file_and_itself_at_their_bytes():
    pins = _reg()["code_pins"]
    for p, s in pins.items():
        assert _sha(p) == s, p
    prev = json.loads((REG / f"{D3.NAME}.json").read_text())["code_pins"]
    for p, s in prev.items():
        if p != f"docs/registered/{D4.NAME}.json":
            assert pins.get(p) == s, p
    for p in RG4.D4_CODE + RG4.FROZEN + ("scripts/r3_timing_model_v4dp.py", "scripts/r3_timing_model_v4d.py",
                                         D4.CAL2_SCORE, D4.FINALISED_D3, D4.CAL2_CENSUS, "scripts/gh200_decode3_session.sh"):
        assert p in pins, p
    assert not any(p in pins for p in REMOVED)


def test_the_rental_states_minutes_wall_cost_deadline_and_cap():
    r = _reg()["rentals"]
    assert (r["ds"]["units_min"], r["ds"]["wall_min"], r["ds"]["cost_usd"]) == (49, 83, 3.17)
    assert (r["ds"]["deadline_after_launch_min"], r["ds"]["hard_cap_after_launch_min"]) == (100, 115)
    assert r["ds"]["cost_at_hard_cap_usd"] == 4.39 and "DS census 6" in r["ds"]["composition"]
    assert r["ds"]["cost_usd"] == round(r["ds"]["wall_min"] / 60 * 2.29, 2) and r["ds"]["boot_setup_min"] == 25
    assert "not planned here" in r["new blind"] and set(r) == {"ds", "operator power step", "new blind"}
    a = _reg()["measurement registration (owner, 2026-11-04, D4-7)"]["cells added (COMPUTED)"]
    assert a["units min (before, with the cells, with the cells and the census)"] == [40, 43, 49]
    assert a["unit min added (cells, census)"] == [3, 6] and a["cost added at $2.29/h"] == 0.38
    assert len(a["cells"]) == 8 and all(c["profiled_s"] == 18.0 and c["enters_gemm_readings"] for c in a["cells"])


def test_the_registration_is_a_measurement_registration_only():
    d = _reg()
    m = d["measurement registration (owner, 2026-11-04, D4-7)"]
    assert "no model form and no v4d'' constants" in m["decision (owner, 2026-11-04, D4-7)"]
    assert "chosen after DS, from DS's and CAL2's pages" in m["the model form"]
    assert "frozen and registered before any page of the three new blind models" in m["the model form"]
    assert "remain unseen" in m["the model form"]
    assert "were seen during the diagnosis" in m["disclosure (plain words)"]
    rd = d["readings (registered before any DS page)"]
    for k in ("LADDERS", "SUBWAVE", "DRAIN", "W1RATE", "CALLS", "ALIGN", "BLOCK", "SLOPE"):
        key = next(x for x in rd if x.startswith(k + " "))
        assert key.startswith(f"{k} (DESCRIPTIVE;"), key
    assert "none is a model term, none is a STOP" in rd["every reading"]
    assert not any(x.startswith(("CALL ", "TAIL", "GROWTH")) for x in rd)
    assert "not a model term and not a rule for any other model" in next(v for k, v in rd.items() if k.startswith("CALLS"))
    sr = d["stop rules (registered before any DS page)"]
    for k in ("prelude", "lock", "POWER", "PAGE", "CENSUS GATE", "CENSUS (", "CENSUS CROSS-CHECK", "G-BAND", "budget"):
        assert any(x.startswith(k) for x in sr), k
    cx = next(v for k, v in sr.items() if k.startswith("CENSUS CROSS-CHECK"))
    assert "reported, not gating" in cx and "DIFFERS is a STOP for the per-CTA readings" in cx
    assert "a missing census or a path mismatch is a STOP" in next(v for k, v in sr.items() if k.startswith("CENSUS GATE"))
    assert "does not apply to DS" in sr["G-BAND (not a DS rule)"]
    i = d["next round (RECORDED INTENT, not code; owner, 2026-11-04, D4-7)"]
    assert i["label"].startswith("RECORDED INTENT, NOT CODE")
    g = i["future gate (owner, 2026-11-04)"]
    assert "ONLY on GO" in g and "4.0%" in g and "WEAK SANITY FLOOR" in g and "CAL2 passes under the post-DS model" in g
    for k in ("launch gate (owner, 2026-11-04, D4-4)", "model v4d'' (owner, 2026-11-04, D4-2b)",
              "blind (owner, 2026-11-04, D4-2c)", "re-registered readings (owner, 2026-11-04, D4-6)"):
        assert k not in d, k
    t = (REG / f"{D4.NAME}.txt").read_text()
    assert "the model form: The decode model form for the new blind will be chosen after DS" in t


def test_no_model_gate_finalisation_or_blind_page_of_this_addendum_exists():
    for p in REMOVED:
        assert not (REPO / p).exists(), p
    assert sorted(p.name for p in (REPO / D4.PAGES4).glob("*.json")) == sorted(f"{m}-ds.json" for m, _d in D4.DS_ORDER)
    for m in D4.NEW_BLIND:
        assert D4.no_decode_page(m)["verdict"] == "NO DECODE PAGE", m
    src = (REPO / "scripts/scoring/decode/d4common.py").read_text()
    for word in ("v4dpp", "terms_from", "BLOCKING", "RESCORE_RMS_MAX", "BARS_OWNER"):
        assert word not in src, word


# ---------------------------------------------------------------- the DS pages and plan

def test_the_earlier_non_decode_pages_are_frozen_at_b2fb3402():
    """The registration carries EARLIER_PAGES_B2FB3402, not a live scan: a later publish cannot move its --check. Every
    frozen session is still published (sessions are only ever added)."""
    seen = _reg()["seen"]["earlier non-decode pages of the new blind models (unused by addendum 4)"]
    assert seen["sessions (COMPUTED at b2fb3402, FROZEN)"] == D4.EARLIER_PAGES_B2FB3402
    assert [len(D4.EARLIER_PAGES_B2FB3402[m]) for m in D4.NEW_BLIND] == [2, 20, 3]
    for m, v in D4.EARLIER_PAGES_B2FB3402.items():
        assert set(v) <= set(D4.earlier_pages(m)), m
    assert "'unseen' in this addendum means no decode page" in seen["statement"]
    keep = D4.earlier_pages
    try:
        D4.earlier_pages = lambda m: ["a-later-session"]                       # a later publish
        assert RG4.build()["seen"] == _reg()["seen"]
    finally:
        D4.earlier_pages = keep


def test_ds_touches_seen_models_only_and_no_new_blind_shape():
    t = D4.ds_touches()
    assert t["all_seen"] and t["new_blind_models_loaded"] == [] and t["new_blind_shapes_measured"] == []
    assert t["declared_E"] == [8, 16, 32, 40, 60, 64]
    for m, _d in D4.DS_ORDER:
        page = RD.page_from(D4.page_doc4(m), MODEL_CONFIGS[m], m)
        assert page.cells and m not in D4.NEW_BLIND


def test_ds_identifies_both_regimes_on_every_tile_and_every_other_reading():
    ident = D4.ds_identification()
    assert ident["both_regimes_on_every_tile"] and ident["all_identified"], ident
    assert set(ident["SUBWAVE"]) == set(ident["DRAIN"]) == set(ident["W1RATE"]) == set(DC.DECODE_TILES)
    assert ident["SUBWAVE"]["32x128x64w4s3"]["models"] == ["phi-3.5-moe", "qwen1.5-moe-a2.7b"]
    assert ident["DRAIN"]["32x128x64w4s3"]["models"] == ["jetmoe-8b", "phi-3.5-moe", "qwen1.5-moe-a2.7b"]
    assert ident["DRAIN"]["64x128x64w4s3"]["models"] == ["jetmoe-8b", "olmoe-1b-7b", "qwen1.5-moe-a2.7b"]
    assert ident["BLOCK"]["block_scan_E"] == [40, 64]
    its = D4.ds_iterations()
    assert max(its[40]) == 24 and sum(len(v) >= 2 for v in its.values()) == 6
    assert set(dict(D4.DS_ORDER).values()) == {"nodrop"}
    # without the three SUBWAVE cells, 32x128x64w4s3 lacks the sub-wave regime; without the owner's four DRAIN cells,
    # the two s3 tiles keep two DRAIN models
    for drop, check in (("SUBWAVE", lambda i: not i["SUBWAVE"]["32x128x64w4s3"]["identified"]),
                        ("DRAIN", lambda i: [len(i["DRAIN"][t]["models"]) for t in ("32x128x64w4s3", "64x128x64w4s3")]
                         == [2, 2])):
        keep = dict(D4.DS_SPECS)
        try:
            for m, T, A, why in D4.DS_ADDED_V4:
                if why == drop:
                    dc, spec = D4.DS_SPECS[m]
                    D4.DS_SPECS[m] = (dc, tuple(x for x in spec if (x[0], x[1]) != (T, A)))
            assert check(D4.ds_identification()), drop
        finally:
            D4.DS_SPECS.clear()
            D4.DS_SPECS.update(keep)


def test_the_replaced_cells_are_past_block_m_and_the_owner_d4_8_cells_are_not():
    """D4-8: Phi T 96 A 5 and T 128 A 3 (every expert past BLOCK_M, no GEMM reading) are gone; Phi T 80 A 5 and A 6 make
    one ladder across one boundary on 32x128x64w4s3, and OLMoE T 128 A 16 and A 32 a sub-wave row and a one-boundary
    crossing row beside T 128 A 64 on 64x128x64w4s3, each with no expert past BLOCK_M."""
    assert D4.DS_REPLACED == (("phi-3.5-moe", 96, 5), ("phi-3.5-moe", 128, 3))
    for m, T, A in D4.DS_REPLACED:
        c = D4.cell_check(m, T, A)
        assert c["experts_over_BLOCK_M"] == A and not c["enters_gemm_readings"]
        assert (T, A) not in {(x[0], x[1]) for x in D4.DS_SPECS[m][1]}
    drain = [(m, T, A) for m, T, A, why in D4.DS_ADDED_V4 if why == "DRAIN"]
    assert drain == [("phi-3.5-moe", 80, 5), ("phi-3.5-moe", 80, 6), ("olmoe-1b-7b", 128, 16), ("olmoe-1b-7b", 128, 32)]
    c5, c6, co, c32 = (D4.cell_check(*x) for x in drain)
    assert (c5["tile"], c5["BLOCK_M"], c5["rows_per_expert"]) == ("32x128x64w4s3", 32, [32, 32])
    assert (c5["gemms"]["w1"]["n"], c5["gemms"]["w1"]["wave"]) == (500, "sub-wave")
    assert (c6["rows_per_expert"], c6["gemms"]["w1"]["n"], c6["gemms"]["w1"]["wave"]) == ([26, 27], 600, "crossing")
    assert (co["tile"], co["BLOCK_M"], co["rows_per_expert"], co["path"]) == ("64x128x64w4s3", 64, [64, 64], RD.BLOCK_SCAN)
    assert [co["gemms"][g]["n"] for g in RD.GEMMS] == [256, 256] and co["gemms"]["w1"]["wave"] == "sub-wave"
    assert (c32["tile"], c32["rows_per_expert"], c32["path"]) == ("64x128x64w4s3", [32, 32], RD.BLOCK_SCAN)
    assert all(c32["gemms"][g]["n"] == 512 and c32["gemms"][g]["wave"] == "crossing" for g in RD.GEMMS)
    assert all(c["enters_gemm_readings"] for c in (c5, c6, co, c32))
    rows = D4.gemm_rows(D4.ds_prm())
    assert "phi-3.5-moe/T80/w1" in D4.mixed_ladders([r for r in rows if r["tile"] == "32x128x64w4s3"])
    assert {"olmoe-1b-7b/T128/w1", "olmoe-1b-7b/T128/w2"} <= set(D4.mixed_ladders([r for r in rows
                                                                                  if r["tile"] == "64x128x64w4s3"]))


def dry(*args, driver):
    return subprocess.run(["bash", str(driver), "--dry-run", *args], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": "/nonexistent"})


def test_the_ds_plan_parses_under_the_d3_driver_in_49_minutes_with_its_own_census():
    assert (REPO / D4.DS_PLAN).read_text() == D4.ds_plan()
    got = dry("--plan", str(REPO / D4.DS_PLAN), driver=REPO / "scripts" / D4.DS_DRIVER)
    assert got.returncode == exit_codes.REFUSED, got.stdout + got.stderr
    assert "a PLAN of 9 units" in got.stdout and "49 min of units" in got.stdout
    cen = next(ln for ln in got.stdout.splitlines() if " tilecensus " in ln)
    assert "label=ds" in cen and "tile-census-mixtral-8x7b-ds" in cen and "[nodrop]" in cen
    assert f"models={','.join(m for m, _d in D4.DS_ORDER)}" in cen and f"tiles={','.join(DC.DECODE_TILES)}" in cen
    tail = got.stdout.split("drop order when time runs short:", 1)[1].splitlines()
    assert tail[1].strip() == "", tail[:2]                                      # every unit nodrop: nothing to drop
    assert got.stdout.count("dram_counter_route_decode3.py --kernels wide") == 6
    assert _reg()["settling rental DS (owner, 2026-11-04, D4-2a)"]["plan_sha256"] == _sha(D4.DS_PLAN)


# ---------------------------------------------------------------- synthetic DS trees

def _power(run: Path, gpu="700.00"):
    (run / "session").mkdir(parents=True, exist_ok=True)
    (run / "session" / "power-set.txt").write_text(POWER_TXT.format(gpu=gpu))


def _ds_census_doc() -> dict:
    """DS's own census as the pinned tile_census.py would write it on the six DS models, with CAL2's compile records
    (the same versions): what a matching DS census reads."""
    cal2 = json.loads((REPO / D4.CAL2_CENSUS).read_text())
    models = [m for m, _d in D4.DS_ORDER]
    return dict(cal2, models=models, entries=[e for e in cal2["entries"] if e["model"] in models], errors=[],
                path_mismatch=[], verdict="COMPLETE")


def _census(root: Path, doc):
    d = root / f"2026-11-05-{F.CARD}-tile-census-mixtral-8x7b-ds"
    d.mkdir(parents=True, exist_ok=True)
    if doc is not None:
        (d / "tile-census.json").write_text(json.dumps(doc))


def _ds_tree(root: Path, *, decl_shift_us=0.0, versions=None, census="match"):
    """A synthetic DS tree whose counters follow the readings' own forms at the planted truth: per GEMM the counted
    bytes B = n b and an ncu time c0 + max(B / c, F, b / R_CTA) + d b / RATE, c0 a per-ladder offset of 9 to 14 us (as
    the published counter pages read), d = 1 past a wave boundary, c w1's planted rate on its tile (W1C) and 3895.49
    otherwise; the call classes on CALL's lines; the align class at ALIGN(E) + SLOPE x (iterations - 1) (block-scan
    ALIGN_BS(E)); DECL rows + decl_shift_us. Each page carries the registered page's sha256, the grid gate PASS, the
    wide filter and the CAL2 session's versions. DS's census: `census` ("match": _ds_census_doc; None: none written)."""
    _census(root, _ds_census_doc() if census == "match" else census)
    prm = D4.ds_prm()
    for m, _d in D4.DS_ORDER:
        cfg = MODEL_CONFIGS[m]
        doc = D4.page_doc4(m)
        rows = []
        for c in doc["cells"]:
            recs = D4.gemm_records(m, c, prm)
            g = RD.geometry(cfg, c["T"], c["counts"])
            for arm in c["arms"]:
                E = cfg.num_experts * (doc["decl_copies"] if arm == "decl" else 1)
                ga = RD.geometry(cfg, c["T"], c["counts"], None if arm == "native" else E)
                per = {}
                for gm in RD.GEMMS:
                    r = recs[gm]
                    n, S = r["n"], r["S"]
                    rate = W1C[g.tile] if gm == "w1" else D4.COUNTER_GBPS
                    B = n * r["b"]
                    c0 = 9.0 + (c["T"] % 5) + (1.0 if gm == "w2" else 0.0)
                    L = c0 + max(B / rate / 1e3, r["F_us"], r["b"] / (R_CTA * 1e3)) + (r["b"] / (RATE * 1e3) if n > S else 0.0)
                    per[gm] = {"dram__bytes_read.sum": B, "gpu__time_duration.sum": L * 1e3}
                if ga.path == RD.BLOCK_SCAN:
                    al = ALIGN_BS(E)
                else:
                    al = ALIGN(E) + SLOPE * (D4.iterations(ga.numel, E) - 1)
                al += decl_shift_us if arm == "decl" else 0.0
                pcl = {"align": {"gpu__time_duration.sum": al * 1e3}}
                for cls, (a_, b_, g_) in CALL.items():
                    x, Bc = D4.call_x(cfg, c["T"], cls)
                    pcl[cls] = {"gpu__time_duration.sum": (a_ + b_ * x + g_ * Bc / 1e6) * 1e3}
                rows.append({"key": f"{arm}/{c['T']}/{c['label']}", "arm": arm, "T": c["T"], "label": c["label"],
                             "per_gemm": per, "per_class": pcl})
        d = root / f"2026-11-05-{F.CARD}-{m}-ds-r3-counters" / "lock1710"
        d.mkdir(parents=True, exist_ok=True)
        (d / "r3d.json").write_text(json.dumps({
            "family": "r3-decode", "kernels": "wide", "cells": rows,
            "manifest": {"page_sha256": _sha(D4.page4_path(m)), "versions": dict(versions or D4.CAL2_VERSIONS)},
            "gates": {"V_ALIGN_grid": "PASS (every GEMM launch's grid is the child's readback)", "refused_cells": []}}))


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    run = tmp_path_factory.mktemp("ds")
    _ds_tree(run / "results")
    _power(run)
    out = run / "out"
    assert RDS.main([str(run / "results"), str(out)]) == 0
    return run, json.loads((out / "ds.readings.json").read_text())


def test_ds_readings_recover_the_planted_truth(ds):
    _run, r = ds
    assert all(x["verdict"] == "READ" for x in r["PAGE"].values()) and all(x["verdict"] == "READ" for x in r["CENSUS"].values())
    assert r["CENSUS GATE"]["verdict"] == "READ (COMPLETE)" and "tile-census-mixtral-8x7b-ds" in r["CENSUS GATE"]["file"]
    assert r["CENSUS CROSS-CHECK"]["verdict"] == "SAME" and r["CENSUS CROSS-CHECK"]["entries_compared"] == 48
    assert set(r["CENSUS CROSS-CHECK"]["CAL2 census (occupancy rule, reported)"].values()) == {"READ"}
    assert r["PER-CTA STOP"]["verdict"] == "none"
    for t in DC.DECODE_TILES:
        x = r["W1RATE"]["tiles"][t]
        assert x["verdict"].startswith("READ: OLS") and len(x["models"]) >= 2, (t, x)
        assert x["counter_GBps"] == pytest.approx(W1C[t], rel=2e-4), (t, x)
        s = r["SUBWAVE"]["tiles"][t]
        assert s["verdict"].startswith("READ: grid") and len(s["models"]) >= 2, (t, s)
        assert s["r_cta_GBps"] == pytest.approx(R_CTA, rel=1e-6) and s["binding_rows"], (t, s)
        assert s["rms_rel"] < 1e-4 < s["rms_rel_no_ceiling"], (t, s)
        d = r["DRAIN"]["tiles"][t]
        assert d["verdict"].startswith("READ: weighted") and len(d["models"]) >= 2, (t, d)
        assert d["r1_GBps"] == pytest.approx(RATE, rel=2e-3) and d["r_cta_GBps_used"] == pytest.approx(R_CTA), (t, d)
        assert d["rms_rel"] < 1e-4 < d["rms_rel_no_drain"], (t, d)
        for lad, c0 in d["offsets_us"].items():
            assert 8.0 <= c0 <= 15.5, (lad, c0)
        lad = r["LADDERS"]["tiles"][t]
        assert set(lad["mixed"]) == set(d["ladders"]) and set(lad["subwave"]) == set(s["qualifying_ladders"])
    for cls, (a_, b_, g_) in CALL.items():
        x = r["CALLS"]["summary_lines"][cls]
        assert x["verdict"].startswith("READ") and len(x["models"]) == 6, (cls, x)
        assert (x["a_us"], x["b_us_per_x"], x["g_us_per_MB"]) == (pytest.approx(a_, abs=1e-5), pytest.approx(b_, rel=1e-5),
                                                                    pytest.approx(g_, rel=1e-5))
    assert set(r["CALLS"]["per_model"]) == {m for m, _d in D4.DS_ORDER}
    assert set(r["CALLS"]["per_model"]["olmoe-1b-7b"]) == {"align", "silu_and_mul", "moe_sum"}
    a = r["ALIGN"]
    assert a["DECL_TRANSFER"]["verdict"] == "TRANSFERS" and a["BLOCK"]["verdict"].startswith("READ")
    assert [E for E, _v in a["points"][RD.SMALL_BATCH]] == [8, 16, 32, 40, 60, 64]
    for E, v in a["points"][RD.SMALL_BATCH]:
        assert v == pytest.approx(ALIGN(E), abs=1e-6)
    assert [(E, round(v, 6)) for E, v in a["points"][RD.BLOCK_SCAN]] == [(40, round(ALIGN_BS(40), 6)), (64, round(ALIGN_BS(64), 6))]
    s = a["SLOPE"]
    assert s["verdict"].startswith("READ") and s["slope_us"] == pytest.approx(SLOPE, abs=1e-6)
    assert s["median_per_E_us"] == pytest.approx(SLOPE, abs=1e-6) and s["iterations_read"] == [1, 24]
    assert s["check_against_0"]["label"] == "DESCRIPTIVE"
    assert r["NOT_IDENTIFIED"] == [] and "terms" not in r and "BLOCKING" not in r
    assert "remain unseen" in r["label"]


def test_a_missing_unit_reads_not_identified_never_a_stop(tmp_path):
    """Qwen1.5's and OLMoE's units missing: DRAIN on 64x128x64w4s3 and SUBWAVE on 32x128x64w4s3 keep one model, BLOCK
    one E; each reads NOT IDENTIFIED, and the readings still read. DRAIN on 32x128x64w4s3 keeps JetMoE and Phi (the
    third model D4-8 added) and still reads."""
    _ds_tree(tmp_path / "results")
    _power(tmp_path)
    for m in ("qwen1.5-moe-a2.7b", "olmoe-1b-7b"):
        for d in (tmp_path / "results").glob(f"*-{m}-ds-r3-counters"):
            (d / "lock1710" / "r3d.json").unlink()
    r = D4.read_ds(tmp_path / "results", D4.ds_prm())
    why = " ".join(r["NOT_IDENTIFIED"])
    assert r["verdict"] == "READ"
    for k in ("DRAIN 64x128x64w4s3: NOT IDENTIFIED", "SUBWAVE 32x128x64w4s3: NOT IDENTIFIED", "BLOCK: NOT IDENTIFIED"):
        assert k in why, k
    assert r["DRAIN"]["tiles"]["16x64x128w4s4"]["verdict"].startswith("READ")
    assert r["DRAIN"]["tiles"]["32x128x64w4s3"]["models"] == ["jetmoe-8b", "phi-3.5-moe"]
    assert r["DRAIN"]["tiles"]["32x128x64w4s3"]["r1_GBps"] == pytest.approx(RATE, rel=2e-3)
    assert D4.fit_w1rate([])["verdict"].startswith("NOT IDENTIFIED") and D4.fit_call([])["verdict"].startswith("NOT IDENTIFIED")


def test_the_page_stop_rule(tmp_path):
    _ds_tree(tmp_path / "results", versions=dict(D4.CAL2_VERSIONS, triton="3.8.0"))
    _power(tmp_path)
    r = D4.read_ds(tmp_path / "results", D4.ds_prm())
    assert all("versions" in x["verdict"] and x["verdict"].startswith("NOT READ") for x in r["PAGE"].values())
    assert r["verdict"] == "NOT READ"
    txt = "\n".join(RDS.lines(r))                     # no page read: the reasons and the model form still print
    assert "PAGE jetmoe-8b: NOT READ: versions" in txt and txt.splitlines()[-1].startswith("model form: the decode")
    _ds_tree(tmp_path / "results")
    p = next((tmp_path / "results").glob("*-phi-3.5-moe-ds-r3-counters")) / "lock1710" / "r3d.json"
    doc = json.loads(p.read_text())
    doc["manifest"]["page_sha256"] = "0" * 64
    p.write_text(json.dumps(doc))
    r = D4.read_ds(tmp_path / "results", D4.ds_prm())
    assert r["PAGE"]["phi-3.5-moe"]["verdict"].startswith("NOT READ: page_sha256")
    assert r["pages"]["phi-3.5-moe"].startswith("NOT READ")
    assert not any(x["model"] == "phi-3.5-moe" for x in r["CALLS"]["rows"])


def _page(root: Path, model: str) -> Path:
    return next(root.glob(f"*-{model}-ds-r3-counters")) / "lock1710" / "r3d.json"


def _mutate(root: Path, model: str, fn):
    p = _page(root, model)
    doc = json.loads(p.read_text())
    fn(doc)
    p.write_text(json.dumps(doc))


def test_page_refuses_another_filter_or_a_grid_gate_not_pass_and_compares_python_on_major_minor(tmp_path):
    _ds_tree(tmp_path / "results", versions=dict(D4.CAL2_VERSIONS, python="3.12.17"))
    _power(tmp_path)
    _mutate(tmp_path / "results", "jetmoe-8b", lambda d: d.update(kernels="gemm"))
    _mutate(tmp_path / "results", "granite-3.0-3b-a800m", lambda d: d["gates"].update(V_ALIGN_grid="FAIL (2 launches)"))
    r = D4.read_ds(tmp_path / "results", D4.ds_prm())
    assert r["PAGE"]["jetmoe-8b"]["verdict"] == "NOT READ: kernels reads 'gemm', not wide"
    assert r["PAGE"]["granite-3.0-3b-a800m"]["verdict"] == "NOT READ: V_ALIGN_grid reads 'FAIL (2 launches)'"
    assert not any(x["model"] in ("jetmoe-8b", "granite-3.0-3b-a800m") for x in r["CALLS"]["rows"])
    assert r["PAGE"]["mixtral-8x7b"]["verdict"] == "READ"                     # python 3.12.17 reads as 3.12
    assert r["PAGE"]["mixtral-8x7b"]["python_patch"] == {"label": "DESCRIPTIVE", "page": "3.12.17", "CAL2": "3.12.16"}
    assert "PAGE mixtral-8x7b: READ; python 3.12.17 (CAL2 3.12.16; patch level DESCRIPTIVE)" in "\n".join(RDS.lines(r))
    for v, ok in (("3.13.0", False), ("3.12", True), (None, False)):
        assert D4.versions_match(dict(D4.CAL2_VERSIONS, python=v)) is ok, v
    assert not D4.versions_match(dict(D4.CAL2_VERSIONS, torch="2.13.1"))


def test_refused_cells_are_left_out_of_every_reading(tmp_path):
    _ds_tree(tmp_path / "results")
    _power(tmp_path)
    key = "native/4/S-T4-A2"
    _mutate(tmp_path / "results", "mixtral-8x7b", lambda d: d["gates"].update(refused_cells=[key]))
    r = D4.read_ds(tmp_path / "results", D4.ds_prm())
    assert r["PAGE"]["mixtral-8x7b"]["verdict"] == "READ" and r["PAGE"]["mixtral-8x7b"]["refused_cells"] == [key]
    assert not any(x["model"] == "mixtral-8x7b" and x["label"] == "S-T4-A2" and x["arm"] == "native"
                   for x in r["CALLS"]["rows"] + r["ALIGN"]["rows"])
    lad = r["LADDERS"]["tiles"]["16x64x128w4s4"]["ladders"]["mixtral-8x7b/T4/w2"]
    assert [x["label"] for x in lad] == ["S-T4-A3", "S-T4-A4"]


#: the garbled rows: (cell key, GEMM) and the reason ROW SANITY gives
GARBLED = {"nan bytes": ("native/4/S-T4-A2", "w2", "bytes nan are not finite"),
           "zero bytes": ("native/4/S-T4-A2", "w2", "bytes 0 are not finite and positive"),
           "negative time": ("native/4/S-T4-A2", "w2", "time -5000.0 is not finite"),
           "zero time": ("native/4/S-T4-A2", "w2", "time 0 is not finite"),
           "bytes off the band": ("native/4/S-T4-A2", "w2", "x n b, outside 0.5 to 2"),
           "bytes x1.99": ("native/96/S-T96-A8", "w1", "GB/s is above the 4022 GB/s ceiling"),
           "time x0.01": ("native/96/S-T96-A8", "w1", "GB/s is above the 4022 GB/s ceiling")}


@pytest.mark.parametrize("case", list(GARBLED))
def test_a_garbled_row_is_refused_and_listed_and_the_readings_still_read(tmp_path, case):
    """ROW SANITY: one GEMM row garbled (Mixtral T 4 A 2 w2; for the rate ceiling, Mixtral T 96 A 8 w1 with bytes x 1.99,
    inside the band, or time x 0.01); the row is refused and listed with its reason, no fit reads a non-finite number,
    nothing crashes, and SUBWAVE, DRAIN and W1RATE recover the planted truth on every tile. "zero time" also zeroes that
    cell's align and moe_sum times: they are refused and CALLS, ALIGN and SLOPE still read. Without T 96 A 8, Mixtral's
    T 96 w1 ladder keeps A 6 and A 7 (237 MB apart), so W1RATE on 32x128x64w4s3 still reads from two models."""
    _ds_tree(tmp_path / "results")
    _power(tmp_path)
    key, gm, want = GARBLED[case]

    def garble(d):
        c = next(x for x in d["cells"] if x["key"] == key)
        g = c["per_gemm"][gm]
        if case == "nan bytes":
            g["dram__bytes_read.sum"] = float("nan")
        elif case == "zero bytes":
            g["dram__bytes_read.sum"] = 0
        elif case == "negative time":
            g["gpu__time_duration.sum"] = -5.0e3
        elif case == "zero time":
            g["gpu__time_duration.sum"] = 0
            c["per_class"]["align"]["gpu__time_duration.sum"] = 0
            c["per_class"]["moe_sum"]["gpu__time_duration.sum"] = 0
        elif case == "bytes off the band":
            g["dram__bytes_read.sum"] *= 3
        elif case == "bytes x1.99":
            g["dram__bytes_read.sum"] *= 1.99
        else:
            g["gpu__time_duration.sum"] *= 0.01
    _mutate(tmp_path / "results", "mixtral-8x7b", garble)
    r = D4.read_ds(tmp_path / "results", D4.ds_prm())
    rf = r["REFUSED ROWS"]["rows"]
    gem = [x for x in rf if x.get("gemm")]
    label = key.split("/")[-1]
    assert [(x["model"], x["label"], x["gemm"]) for x in gem] == [("mixtral-8x7b", label, gm)], rf
    assert want in gem[0]["why"], gem[0]["why"]
    if case == "zero time":
        assert sorted(x["class"] for x in rf if x.get("class")) == ["align", "moe_sum"]
    tile = "32x128x64w4s3" if gm == "w1" else "16x64x128w4s4"
    lad = r["LADDERS"]["tiles"][tile]["ladders"][f"mixtral-8x7b/T{key.split('/')[1]}/{gm}"]
    assert label not in [x["label"] for x in lad]
    for t in DC.DECODE_TILES:
        s_, d_ = r["SUBWAVE"]["tiles"][t], r["DRAIN"]["tiles"][t]
        assert s_["r_cta_GBps"] == pytest.approx(R_CTA, rel=1e-6), (t, s_["verdict"])
        assert d_["r1_GBps"] == pytest.approx(RATE, rel=2e-3), (t, d_["verdict"])
        assert r["W1RATE"]["tiles"][t]["counter_GBps"] == pytest.approx(W1C[t], rel=2e-4), t
    for c in D4.CALL_CLASSES:
        assert r["CALLS"]["summary_lines"][c]["verdict"].startswith("READ"), c
    assert r["ALIGN"]["SLOPE"]["slope_us"] == pytest.approx(SLOPE, abs=1e-6)
    assert r["ALIGN"]["BLOCK"]["verdict"].startswith("READ")
    txt = "\n".join(RDS.lines(r))
    assert f"  mixtral-8x7b {label} {gm}: " in txt and "nan" not in RDS.lines(r)[-2].lower()
    json.dumps(r, allow_nan=False, default=str)                                # no NaN anywhere in the readings


def test_no_reading_prices_w1_at_a_fallback_rate(tmp_path):
    """NO FALLBACK (owner, 2026-11-04): Mixtral T 96 A 7 and A 8 w1 both garbled (time x 0.01) leave Mixtral's T 96 w1
    ladder one row, so W1RATE on 32x128x64w4s3 keeps one model and reads NOT IDENTIFIED. There, SUBWAVE (its fitted
    ladders hold Qwen1.5's T 96 w1) and DRAIN (its mixed ladders hold w1) read NOT IDENTIFIED, and LADDERS shows the w1
    rows unpriced; no w1 row anywhere is priced at 3895.49 GB/s. The other tiles read."""
    _ds_tree(tmp_path / "results")
    _power(tmp_path)

    def garble(d):
        for c in d["cells"]:
            if c["key"] in ("native/96/S-T96-A7", "native/96/S-T96-A8"):
                c["per_gemm"]["w1"]["gpu__time_duration.sum"] *= 0.01
    _mutate(tmp_path / "results", "mixtral-8x7b", garble)
    r = D4.read_ds(tmp_path / "results", D4.ds_prm())
    assert len([x for x in r["REFUSED ROWS"]["rows"] if x.get("gemm") == "w1"]) == 2
    t = "32x128x64w4s3"
    assert r["W1RATE"]["tiles"][t]["verdict"] == "NOT IDENTIFIED: 1 usable w1 ladders from 1 models (needs 2 from 2)"
    d = r["DRAIN"]["tiles"][t]
    assert d["verdict"].startswith("NOT IDENTIFIED: the mixed ladders hold w1 rows") and d["r1_GBps"] is None, d
    assert d["verdict"].endswith(D4.NO_FALLBACK) and f"DRAIN {t}: NOT IDENTIFIED" in " ".join(r["NOT_IDENTIFIED"])
    sw = r["SUBWAVE"]["tiles"][t]
    assert sw["verdict"].startswith("NOT IDENTIFIED: the fitted ladders hold w1 rows") and sw["r_cta_GBps"] is None, sw
    assert sw["verdict"].endswith(D4.NO_FALLBACK)
    lad = r["LADDERS"]["tiles"][t]
    assert lad["verdict"] == "READ" and lad["w1_unpriced"].startswith("w1 rows not priced")
    w1rows = [x for k, v in lad["ladders"].items() if k.endswith("/w1") for x in v]
    assert w1rows and all(x["Dc_us"] is None and x["X_us"] is None for x in w1rows)
    assert all(x["Dc_us"] is not None for k, v in lad["ladders"].items() if k.endswith("/w2") for x in v)
    assert f"LADDERS {t}: READ" in "\n".join(RDS.lines(r)) and "w1 rows not priced" in "\n".join(RDS.lines(r))
    for u in DC.DECODE_TILES:
        if u != t:
            assert r["SUBWAVE"]["tiles"][u]["r_cta_GBps"] == pytest.approx(R_CTA, rel=1e-6), u
            assert r["LADDERS"]["tiles"][u]["w1_unpriced"] is None, u
    # dc_rows: w1 takes the tile's c_t or nothing; w2 the counters' measured rate
    rows = [{"gemm": g, "tile": t, "B_bytes": 1e9, "L_us": 500.0, "F_us": 0.0} for g in RD.GEMMS]
    got = D4.dc_rows(rows, {"tiles": {t: {"verdict": "NOT IDENTIFIED"}}})
    assert (got[0]["c_GBps"], got[0]["Dc_us"], got[0]["w1_unpriced"]) == (None, None, True)
    assert (got[1]["c_GBps"], got[1]["w1_unpriced"]) == (pytest.approx(D4.COUNTER_GBPS), False)
    got = D4.dc_rows(rows, {"tiles": {t: {"counter_GBps": 3760.0}}})
    assert got[0]["c_GBps"] == 3760.0 and got[0]["Dc_us"] == pytest.approx(1e9 / 3760.0 / 1e3)
    for u in DC.DECODE_TILES:
        if u != t:
            assert r["DRAIN"]["tiles"][u]["r1_GBps"] == pytest.approx(RATE, rel=2e-3), u
    rows = [{"model": m, "ladder": f"{m}/T4/{g}", "gemm": g, "n": n, "S": 396, "cross": n > 396, "b_bytes": 1e6,
             "F_us": 0.0, "Dc_us": 1.0, "L_us": 10.0 + (1.0 if n > 396 else 0.0)} for m in "ab" for g in ("w2",)
            for n in (300, 500)]
    assert D4.fit_drain(rows)["verdict"].startswith("READ")                   # w2 only: the rule does not apply


def test_a_non_finite_or_non_positive_fit_reads_not_identified():
    rows = []
    for m in ("a", "b"):
        for T, (n1, n2) in ((4, (500, 900)), (8, (450, 1000))):
            for n, L in ((n1, 30.0), (n2, 20.0)):                              # more bytes, less time: u < 0
                rows.append({"model": m, "T": T, "gemm": "w1", "cross": True, "B_bytes": n * 1e6 / 2, "L_us": L})
    x = D4.fit_w1rate(rows)
    assert x["verdict"].startswith("NOT IDENTIFIED: the marginal time per byte") and x["counter_GBps"] is None
    assert D4.RATE_CEILING_GBPS == 4022.0                                     # the GH200's HBM3 spec peak
    fast = [dict(x, L_us=10.0 + x["B_bytes"] / 4100.0 / 1e3 + (1.0 if x["T"] == 8 else 0.0)) for x in rows]
    x = D4.fit_w1rate(fast)                                                   # c 4100 GB/s, past the ceiling
    assert x["verdict"] == "NOT IDENTIFIED: c 4100 GB/s is above the 4022 GB/s ceiling" and x["counter_GBps"] is None
    assert D4.fit_w1rate([dict(x, L_us=10.0 + x["B_bytes"] / 3950.0 / 1e3) for x in rows])["counter_GBps"] == \
        pytest.approx(3950.0)
    assert D4.row_sanity(1e3, 4.05e6, 4, 1e6).startswith("implied DRAM rate 4050 GB/s is above the 4022")
    assert D4.row_sanity(1e3, 3.95e6, 4, 1e6) is None
    rows[0]["L_us"] = float("nan")
    assert D4.fit_w1rate(rows)["verdict"] == D4.NONFINITE
    sub = [{"model": m, "ladder": f"{m}/T4/w2", "n": n, "S": 396, "cross": False, "b_bytes": 1e6, "F_us": 0.0,
            "Dc_us": 1.0, "L_us": float("nan") if (m, n) == ("a", 100) else 10.0} for m in "ab" for n in (100, 200, 300)]
    assert D4.fit_subwave(sub)["verdict"] == D4.NONFINITE
    mix = [dict(x, cross=x["n"] > 200) for x in sub]
    assert D4.fit_drain(mix)["verdict"] == D4.NONFINITE
    calls = [{"model": m, "x": x, "B_bytes": 1e6 * x, "us": float("inf") if m == "c" else 3.0 + x}
             for m, x in (("a", 1.0), ("b", 2.0), ("c", 3.0))]
    assert D4.fit_call(calls)["verdict"] == D4.NONFINITE
    slope = [{"E": E, "iterations": i, "align_us": float("nan") if (E, i) == (8, 3) else 4.0 + i} for E in (8, 64)
             for i in (1, 3)]
    assert D4.fit_slope(slope)["verdict"] == D4.NONFINITE


def _per_cta_stopped(r, why):
    assert r["PER-CTA STOP"]["verdict"].startswith("STOP") and why in r["PER-CTA STOP"]["verdict"], r["PER-CTA STOP"]
    for k in ("W1RATE", "SUBWAVE", "DRAIN"):
        assert all(x["verdict"].startswith("STOP") for x in r[k]["tiles"].values()), k
    assert r["LADDERS"]["verdict"].startswith("STOP")
    assert all(x["verdict"].startswith("STOP") and x["ladders"] == {} for x in r["LADDERS"]["tiles"].values())
    assert r["CALLS"]["summary_lines"]["silu_and_mul"]["verdict"].startswith("READ")      # CALLS takes no S
    assert r["ALIGN"]["SLOPE"]["verdict"].startswith("READ") and r["ALIGN"]["BLOCK"]["verdict"].startswith("READ")


def test_a_missing_or_refused_ds_census_is_a_stop_for_the_per_cta_readings(tmp_path):
    """The census gate CAL used: no DS census, or one with an MMA path mismatch, STOPS LADDERS, SUBWAVE, DRAIN and
    W1RATE (never a fallback to CAL2's census); CALLS, ALIGN, BLOCK and SLOPE still read."""
    _ds_tree(tmp_path / "a" / "results", census=None)
    _power(tmp_path / "a")
    r = D4.read_ds(tmp_path / "a" / "results", D4.ds_prm())
    assert r["verdict"] == "READ" and r["CENSUS GATE"]["verdict"].startswith("STOP: no DS census (0 DS census files)")
    assert r["CENSUS CROSS-CHECK"]["verdict"] == "NOT READ: no DS census"
    _per_cta_stopped(r, "no DS census")
    bad = dict(_ds_census_doc(), path_mismatch=["jetmoe-8b 32x64x128w4s3 w1: wgmma, not mma.sync"],
               verdict="REFUSED (path mismatch)")
    _ds_tree(tmp_path / "b" / "results", census=bad)
    _power(tmp_path / "b")
    r = D4.read_ds(tmp_path / "b" / "results", D4.ds_prm())
    assert r["CENSUS GATE"]["verdict"] == "STOP: REFUSED (path mismatch): jetmoe-8b 32x64x128w4s3 w1: wgmma, not mma.sync"
    _per_cta_stopped(r, "path mismatch")
    assert any("STOP" in w for w in r["NOT_IDENTIFIED"])


def _occ(doc: dict, model: str, tile: str, gemm: str, occ: int) -> dict:
    """The census doc with one (model, tile, GEMM) compiled at another occupancy, in its entry and its summary."""
    doc = json.loads(json.dumps(doc))
    e = next(x for x in doc["entries"] if (x["model"], x["tile"], x["gemm"]) == (model, tile, gemm))
    e["occupancy"] = dict(e["occupancy"], ctas_per_sm=occ)
    doc["summary"][tile]["by"][f"{model}|{gemm}"] = occ
    return doc


def test_a_ds_census_that_differs_from_cal2s_is_a_stop_with_the_details(tmp_path):
    """DS's census reads another occupancy for JetMoE's w2 on 32x64x128w4s3: the tile's occupancy rule reads NOT READ,
    the cross-check reads DIFFERS with the entry named, and every per-CTA reading STOPS on every tile."""
    _ds_tree(tmp_path / "results", census=_occ(_ds_census_doc(), "jetmoe-8b", "32x64x128w4s3", "w2", 3))
    _power(tmp_path)
    r = D4.read_ds(tmp_path / "results", D4.ds_prm())
    assert r["CENSUS"]["32x64x128w4s3"]["verdict"].startswith("NOT READ: jetmoe-8b|w2: census 3 against 4")
    assert r["CENSUS"]["16x64x128w4s4"]["verdict"] == "READ"
    x = r["CENSUS CROSS-CHECK"]
    assert x["verdict"] == "DIFFERS" and len(x["differences"]) == 1
    assert x["differences"][0].startswith("jetmoe-8b 32x64x128w4s3 w2 config") and "occupancy" in x["differences"][0]
    _per_cta_stopped(r, "DS's census differs from CAL2's: jetmoe-8b 32x64x128w4s3 w2")
    txt = "\n".join(RDS.lines(r))
    assert "CENSUS CROSS-CHECK (DS's census against CAL2's, reported): DIFFERS over 48 entries" in txt
    assert "\n  jetmoe-8b 32x64x128w4s3 w2 config" in txt and "PER-CTA STOP (LADDERS, SUBWAVE, DRAIN, W1RATE): STOP" in txt


def test_cal2s_census_is_a_reported_cross_check_not_a_gate(tmp_path):
    """CAL2's census is read under the occupancy rule and reported; it gates nothing. When DS's census and CAL2's read
    the same off-rule occupancy on a tile, the cross-check reads SAME and DS's own census leaves that tile NOT READ by
    the occupancy rule alone (no STOP); CAL2's summary off the rule, its entries matching DS's, changes no reading."""
    _ds_tree(tmp_path / "results")
    _power(tmp_path)
    cal2 = json.loads((REPO / D4.CAL2_CENSUS).read_text())
    both = ("jetmoe-8b", "32x64x128w4s3", "w2", 3)
    r = D4.read_ds(tmp_path / "results", D4.ds_prm(), census=_occ(_ds_census_doc(), *both), cal2_census=_occ(cal2, *both))
    x = r["CENSUS CROSS-CHECK"]
    assert x["verdict"] == "SAME" and x["CAL2 census (occupancy rule, reported)"]["32x64x128w4s3"].startswith("NOT READ")
    assert r["PER-CTA STOP"]["verdict"] == "none"
    assert r["CENSUS"]["32x64x128w4s3"]["verdict"].startswith("NOT READ: jetmoe-8b|w2")
    for k in ("W1RATE", "SUBWAVE", "DRAIN", "LADDERS"):
        assert r[k]["tiles"]["32x64x128w4s3"]["verdict"].startswith("NOT READ: CENSUS"), k
        assert r[k]["tiles"]["16x64x128w4s4"]["verdict"].startswith("READ"), k
    assert r["LADDERS"]["tiles"]["32x64x128w4s3"]["ladders"] == {}
    assert "LADDERS 32x64x128w4s3: NOT READ: CENSUS" in "\n".join(RDS.lines(r))
    cal2["summary"]["32x64x128w4s3"]["by"]["jetmoe-8b|w2"] = 3
    r = D4.read_ds(tmp_path / "results", D4.ds_prm(), cal2_census=cal2)
    assert r["CENSUS CROSS-CHECK"]["CAL2 census (occupancy rule, reported)"]["32x64x128w4s3"].startswith("NOT READ")
    assert r["CENSUS CROSS-CHECK"]["verdict"] == "SAME" and r["PER-CTA STOP"]["verdict"] == "none"
    assert all(v["verdict"] == "READ" for v in r["CENSUS"].values())
    assert r["SUBWAVE"]["tiles"]["32x64x128w4s3"]["verdict"].startswith("READ")
    assert all(x["verdict"] == "READ" for t, x in D4.census_check(D4.ds_prm(), json.loads(
        (REPO / D4.CAL2_CENSUS).read_text())).items())


def test_ds_drops_the_decl_points_when_decl_does_not_transfer_and_reads_nothing_off_power(tmp_path):
    _ds_tree(tmp_path / "results", decl_shift_us=1.5)
    _power(tmp_path)
    r = D4.read_ds(tmp_path / "results", D4.ds_prm())
    assert r["ALIGN"]["DECL_TRANSFER"]["used"] is False
    assert [E for E, _v in r["ALIGN"]["points"][RD.SMALL_BATCH]] == [8, 16, 40, 60, 64]
    _power(tmp_path, gpu="900.00")
    assert D4.read_ds(tmp_path / "results", D4.ds_prm())["verdict"] == "NOT READ"


def test_subwave_reads_no_ceiling_when_none_binds():
    rows = []
    for m, b in (("a", 1.0e6), ("b", 2.0e6)):
        for n in (100, 200, 300):
            dc = n * b / D4.COUNTER_GBPS / 1e3
            rows.append({"model": m, "ladder": f"{m}/T4/w2", "n": n, "S": 396, "cross": False, "b_bytes": b, "F_us": 0.0,
                         "Dc_us": dc, "L_us": 10.0 + dc})
    x = D4.fit_subwave(rows)
    assert x["verdict"].endswith("no ceiling binds") and x["r_cta_GBps"] is None and x["binding_rows"] == []
    assert D4.fit_subwave(rows[:3])["verdict"].startswith("NOT IDENTIFIED")


def test_fit_slope_reads_one_slope_over_per_e_intercepts_and_checks_it_against_0():
    rows = [{"E": E, "iterations": i, "align_us": a + 0.5 * (i - 1)} for E, a in ((8, 4.0), (64, 11.0)) for i in (1, 3, 8)]
    s = D4.fit_slope(rows)
    assert s["verdict"].startswith("READ") and s["slope_us"] == pytest.approx(0.5)
    assert s["intercepts_us"] == {"8": pytest.approx(4.0), "64": pytest.approx(11.0)} and s["iterations_read"] == [1, 8]
    assert D4.fit_slope([r for r in rows if r["E"] == 8])["verdict"].startswith("NOT IDENTIFIED")
    flat = [dict(r, align_us=r["align_us"] - 0.5 * (r["iterations"] - 1) + (0.05 if k % 2 else -0.05))
            for k, r in enumerate(rows)]
    assert D4.fit_slope(flat)["check_against_0"]["verdict"] == "NOT DISTINGUISHABLE FROM 0"
    noisy = [dict(r, align_us=r["align_us"] + (0.05 if k % 2 else -0.05)) for k, r in enumerate(rows)]
    assert D4.fit_slope(noisy)["check_against_0"]["verdict"] == "DIFFERS FROM 0"


def test_the_published_trees_reproduce_their_committed_scores(capsys):
    cal2, blind = REPO / D4.CAL2_PUBLISHED / "results", REPO / D4.BLIND_PUBLISHED / "results"
    if not cal2.is_dir() or not blind.is_dir():
        pytest.skip("the CAL2 or decode blind session is not published in this checkout")
    assert RP4.main([str(cal2), str(blind)]) == 0, capsys.readouterr().out
    assert D4.blind_check({"E1_DEC": {"pooled": {"stats": {"cells": 1}}}}, json.loads((REPO / D4.BLIND_SCORE).read_text()))


def test_the_cal2_versions_are_the_published_sessions():
    for p in (REPO / D4.CAL2_PUBLISHED / "results").glob("*-r3-counters/lock1710/r3d.json"):
        assert json.loads(p.read_text())["manifest"]["versions"] == D4.CAL2_VERSIONS, p


# ---------------------------------------------------------------- house rules

def test_no_em_dash_and_no_trailer_in_the_addendum_files():
    paths = [REPO / p for p in RG4.D4_CODE] + [REG / f"{D4.NAME}.{x}" for x in ("json", "txt")] + [Path(__file__)]
    paths += [REPO / D4.page4_path(m, k) for m, k in D4.PAGE_KEYS]
    for p in paths:
        t = p.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, p
        assert not re.search("|".join(("Co-" + "Authored-By", "Claude " + "Code", "Generated " + "with", "Anth" + "ropic")), t), p


def test_the_earlier_decode_modules_are_untouched():
    pins = json.loads((REG / f"{D3.NAME}.json").read_text())["code_pins"]
    for p in ("scripts/r3_timing_model_v4dp.py", "scripts/r3_timing_model_v4d.py", "scripts/scoring/decode/d3common.py",
              "scripts/scoring/decode/score_blind_d3.py", "scripts/gh200_decode3_session.sh", "scripts/decode_synth.py"):
        assert _sha(p) == pins[p], p
    assert math.isclose(V4DP.W1_SCALE, 0.9184877841384885)
