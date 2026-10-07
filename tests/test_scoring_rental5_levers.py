"""Rental 5's lever scorers (scripts/scoring/rental5/score_{secondk,bk128}.py) and the v2
re-prediction scorer's rules, written before any rental-5 page: each scorer is driven on
SYNTHETIC pages built in tmp_path, never on a published one, and must return the verdict its
registration's rule gives for the world planted. The registrations come from their builders
(scripts/scoring/rental5/reg_levers.py) into a scratch repo beside the addendum they need, so
these tests do not depend on the committed JSON (register.py --check holds that)."""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
R5DIR = REPO / "scripts" / "scoring" / "rental5"
for p in (REPO, REPO / "scripts", R5DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r5common as C5  # noqa: E402
import reg_levers as RL  # noqa: E402
import score_bk128 as SB  # noqa: E402
import score_secondk as SK  # noqa: E402
import score_v2seen as SV  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

C4 = C5.C4
CARD = C5.CARD
DAY = "2026-10-08"


@pytest.fixture(scope="module")
def regs():
    return {p: RL.BUILDERS[p]() for p in ("secondk", "bk128")}


@pytest.fixture()
def repo(tmp_path, regs):
    """A scratch repo: the rental-2 addendum (two_views reads it) and the built registrations."""
    r = tmp_path / "repo"
    (r / "docs" / "registered").mkdir(parents=True)
    add = C4.CM2.ADDENDUM
    shutil.copy(REPO / "docs" / "registered" / f"{add}.json", r / "docs" / "registered")
    for p, d in regs.items():
        (r / "docs" / "registered" / f"{C5.NAMES[p]}.json").write_text(json.dumps(d, default=str))
    return r


# --------------------------------------------------------------------------
# the registrations
# --------------------------------------------------------------------------

def test_the_builders_reproduce_the_designs_numbers(regs):
    sk = regs["secondk"]["predicted_ratio"]
    assert sk["w1"]["H_ITER"] == pytest.approx(1.406, abs=5e-4) and sk["w2"]["H_ITER"] == pytest.approx(1.231, abs=5e-4)
    assert sk["w1"]["H_CTA"] == pytest.approx(1.232, abs=5e-4) and sk["w2"]["H_CTA"] == pytest.approx(1.739, abs=5e-4)
    assert sk["w2"]["S32"] == 10 and sk["w2"]["S64"] == 5
    for v in regs["bk128"]["predicted_q"].values():
        assert 0.985 <= v["H_C"] <= 1.0 and 0.55 <= v["SYNC"] <= 0.65
        assert abs(v["PS"] / v["H_C"] - 1) < 0.006


def test_the_registrations_carry_their_labels_and_no_typographic_dash(regs):
    for p, d in regs.items():
        text = json.dumps(d)
        assert "\u2014" not in text and "\u2013" not in text, p
        assert d["status"].startswith("REGISTERED") and d["seen_data"] and d["labels"], p
    assert regs["secondk"]["labels"]["k32s4"] == "BLIND"
    assert "BLIND-CALMODEL" in regs["bk128"]["labels"]["time"] and "not a lever" in regs["bk128"]["status"]
    assert "tp2" not in RL.BUILDERS and "tp2" not in C5.PARTS   # CUT by the owner, 2026-10-07


def test_v2_registration_labels_and_expected_table():
    d = RL.v2()
    assert d["model"]["refit_8x7b"]["v2"]["T0"] == pytest.approx(0.04323, abs=1e-5)
    assert d["model"]["refit_8x7b"]["M"]["T0"] == pytest.approx(0.04424, abs=1e-5)
    assert d["model"]["d_kappa_fit"]["set"].startswith("CAL-counters (4 models")
    assert "IN-SAMPLE (d, kappa)" in json.dumps(d["re_predictions"]["labels"])
    assert "BLIND-CALMODEL only because" in d["labels"]["olmoe_skew"]
    assert d["expected_rms_pct"]["rental3 E2 (counted)"] == [2.52, 0.97]


def test_v2seen_check_flags_a_number_off_by_more_than_a_hundredth():
    reg = {"expected_rms_pct": {"a": [1.00, 2.00]}}
    tt = {"m": {"tests": {"a": {"rms": 0.010049}}}, "v2": {"tests": {"a": {"rms": 0.020049}}}}
    assert SV.check(reg, tt) == []
    tt["v2"]["tests"]["a"]["rms"] = 0.0202
    assert len(SV.check(reg, tt)) == 1
    del tt["m"]["tests"]["a"]
    assert len(SV.check(reg, tt)) == 2
    assert SV.label_of("olmoe-1b-7b").startswith("IN-SAMPLE") and SV.label_of("jetmoe-8b").startswith("held-out")


# --------------------------------------------------------------------------
# secondk: synthetic counter pages
# --------------------------------------------------------------------------

QW = MODEL_CONFIGS["qwen2-57b-a14b-tp8"]


def sk_page(bk, c, *, dram_frac=0.2, design_bk=None):
    E = QW.num_experts
    K = {"w1": QW.hidden_size, "w2": QW.intermediate_size}
    npn = {"w1": 2 * QW.intermediate_size // 64, "w2": QW.hidden_size // 64}
    cells = []
    for arm, D in (("native", E), ("shared", 9 * E)):
        for n in range(1, 10):
            pg, rec = {}, {}
            for g in ("w1", "w2"):
                R = math.ceil((E * n * 32 + D * 31) / 32)
                live = E * n * npn[g]
                cyc = (5000 if arm == "native" else 9000) + math.ceil(live / 132) * (K[g] // bk * c[g] + C4.F_CTA[g])
                pg[g] = {"grid_size": R * npn[g], "gpu_time_ns": 1.0e6, "dram_bytes_read": dram_frac * 4022.0 * 1.0e6}
                rec[g] = {"launch__occupancy_limit_registers": 5, "launch__occupancy_limit_warps": 8,
                          "sm__cycles_elapsed.avg": cyc}
            cells.append({"arm": arm, "n": n, "declared": D, "per_gemm": pg, "recorded": rec})
    return {"family": "r3-arms", "design": {"model": "qwen2-57b-a14b-tp8", "block_m": 32, "block_n": 64,
                                            "block_k": design_bk or bk, "num_stages": 4, "group_m": 64},
            "card": {"slug": CARD, "sm_count": 132}, "cells": cells, "gates": []}


def put_bytes(tree, model, label, page):
    d = tree / f"{DAY}-{CARD}-{model}-{label}-r3-counters" / "lock1710"
    d.mkdir(parents=True, exist_ok=True)
    (d / "r3c-g64.json").write_text(json.dumps(page))


def c32_for(reg, h):
    out = {}
    for g in ("w1", "w2"):
        p = reg["predicted_ratio"][g]
        u64 = p["S64"] * RL.C64[g] + C4.F_CTA[g]
        out[g] = (p[h] * u64 - C4.F_CTA[g]) / p["S32"]
    return out


@pytest.mark.parametrize("h", ["H_ITER", "H_CTA"])
def test_secondk_selects_the_hypothesis_its_pages_were_built_under(tmp_path, repo, regs, h):
    put_bytes(tmp_path, "qwen2-57b-a14b-tp8", "k64s4", sk_page(64, RL.C64))
    put_bytes(tmp_path, "qwen2-57b-a14b-tp8", "k32s4", sk_page(32, c32_for(regs["secondk"], h)))
    res = SK.score(repo, tmp_path)
    assert res["verdict"] == f"SELECTED {h}", res["verdict"]
    assert abs(res["gemms"]["w1"]["e"][h]) < 1e-6


def test_secondk_split_readings_are_undecided_and_a_missing_or_wrong_page_not_scored(tmp_path, repo, regs):
    c = c32_for(regs["secondk"], "H_ITER")
    c["w2"] = c32_for(regs["secondk"], "H_CTA")["w2"]
    put_bytes(tmp_path, "qwen2-57b-a14b-tp8", "k64s4", sk_page(64, RL.C64))
    put_bytes(tmp_path, "qwen2-57b-a14b-tp8", "k32s4", sk_page(32, c))
    assert SK.score(repo, tmp_path)["verdict"].startswith("UNDECIDED")
    put_bytes(tmp_path, "qwen2-57b-a14b-tp8", "k32s4", sk_page(32, c, design_bk=64))
    res = SK.score(repo, tmp_path)
    assert res["verdict"].startswith("NOT SCORED") and res["pages"]["k32s4"].startswith("NOT SCORED")
    assert SK.score(repo, tmp_path / "empty")["verdict"].startswith("NOT SCORED")


def test_secondk_reads_no_dram_bound_cell(tmp_path, repo, regs):
    put_bytes(tmp_path, "qwen2-57b-a14b-tp8", "k64s4", sk_page(64, RL.C64, dram_frac=0.9))
    put_bytes(tmp_path, "qwen2-57b-a14b-tp8", "k32s4", sk_page(32, c32_for(regs["secondk"], "H_ITER")))
    res = SK.score(repo, tmp_path)
    assert res["gemms"]["w1"]["verdict"].startswith("NOT SCORED") and res["verdict"].startswith("NOT SCORED")


# --------------------------------------------------------------------------
# bk128: synthetic R3 pages
# --------------------------------------------------------------------------

def put_timed(tree, model, label, ms_of, *, G=8, copies=9, bk=64, stages=4, align_of=None, fail=()):
    d = tree / f"gaps-{CARD}-{model}-{label}" / "private_weight_reference" / f"run-{label}"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"arm": a, "tiles": n, "ms_p50": ms_of(a, n)} for a in ("native", "shared", "private") for n in range(1, 10)]
    align_of = align_of or (lambda a, n: 0.0055)
    align = [{"label": a, "tread": n, "repeat": r, "ms": align_of(a, n)} for a in ("native", "shared", "private")
             for n in range(1, 10) for r in range(2)]
    gates = [{"tag": f"V{i}", "kind": "VALIDITY", "verdict": "FAIL" if f"V{i}" in fail else "PASS"} for i in range(9)]
    (d / "report.json").write_text(json.dumps({
        "session_tag": f"gh200-x-{label}-lock1710", "copies_declared": copies,
        "pinned": {"GROUP_SIZE_M": G, "BLOCK_SIZE_K": bk, "num_stages": stages, "BLOCK_SIZE_N": 64},
        "treads_table": rows, "align_probe": {"cells": align}, "gates": gates}))


def bk_world(tree, regs, rival, *, bk128_bk=128, drop=()):
    q = {int(n): v[rival] for n, v in regs["bk128"]["predicted_q"].items()}
    put_timed(tree, "olmoe-1b-7b", "bk64", lambda a, n: 0.2 + 0.1 * n, G=64)
    put_timed(tree, "olmoe-1b-7b", "bk128",
              lambda a, n: None if n in drop else (0.2 + 0.1 * n) * q.get(n, 1.0), G=64, bk=bk128_bk)


def test_bk128_at_h_c_holds_and_excludes_sync(tmp_path, repo, regs):
    bk_world(tmp_path, regs, "H_C")
    res = SB.score(repo, tmp_path)
    assert res["H_C"]["verdict"] == "HOLDS" and res["SYNC"]["verdict"] == "EXCLUDED", res["verdict"]


def test_bk128_at_sync_fails_h_c(tmp_path, repo, regs):
    bk_world(tmp_path, regs, "SYNC")
    res = SB.score(repo, tmp_path)
    assert res["H_C"]["verdict"] == "FAILS" and res["SYNC"]["verdict"] == "NOT EXCLUDED"


def test_bk128_refuses_a_wrong_block_k_and_too_few_cells(tmp_path, repo, regs):
    bk_world(tmp_path, regs, "H_C", bk128_bk=64)
    assert SB.score(repo, tmp_path)["verdict"].startswith("NOT SCORED")
    t2 = tmp_path / "t2"
    bk_world(t2, regs, "H_C", drop=(4, 5, 6))
    assert SB.score(repo, t2)["verdict"].startswith("NOT SCORED: 3 of")


def test_every_lever_scorer_writes_its_files_on_an_empty_tree(tmp_path, repo):
    for mod, part in ((SK, "secondk"), (SB, "bk128")):
        assert mod.main([str(repo), str(tmp_path / "none"), str(tmp_path / "out")]) == 0
        assert (tmp_path / "out" / f"{part}.score.json").exists() and (tmp_path / "out" / f"{part}.score.txt").exists()
