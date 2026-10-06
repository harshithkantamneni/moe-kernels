"""Rental 3's scorers (scripts/scoring/rental3/), written before any rental-3 page:
each is driven here on SYNTHETIC pages built in tmp_path, never on a published one,
and must return the verdict the registration's rule gives for the world planted.
Also the offline pieces they read (floor_estimator's joint fit and rulers, the
lock-in-force check) and the registrations' own consistency (register.py --check)."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
R3DIR = REPO / "scripts" / "scoring" / "rental3"
for p in (REPO, REPO / "scripts", R3DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dram_counter_route as DCR  # noqa: E402
import floor_estimator as FE  # noqa: E402
import l2_survival as L2S  # noqa: E402
import r3common as C3  # noqa: E402
import score_e2e as SE  # noqa: E402
import score_floorlaw as SF  # noqa: E402
import score_flush as SH  # noqa: E402
import score_replicate as SR  # noqa: E402
import score_stages as SS  # noqa: E402
import score_zform as SZ  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

E2E = C3.registration(REPO, "e2e")
FLR = C3.registration(REPO, "floorlaw")
ZF = C3.registration(REPO, "zform")
STG = C3.registration(REPO, "stages")
FLU = C3.registration(REPO, "flush")
REP = C3.registration(REPO, "replicate")
DAY = "2026-10-06"
CARD = C3.CARD
F_MEAS = {"base": 1375.0, "lock1710": 1690.0, "lock1005": 991.0, "lock1410": 1390.0}
F_RULER = {"base": 1375.0, "lock1710": 1710.0, "lock1005": 1005.0, "lock1410": 1410.0}
L2_MHZ = {"base": 1704.0, "lock1710": 1704.0, "lock1005": 1125.0, "lock1410": 1400.0}
STEM = {"base": "r3f-g64", "lock1710": "r3f-g64-lock1710", "lock1005": "r3f-g64-lock1005",
        "lock1410": "r3f-g64-lock1410"}


# --------------------------------------------------------------------------
# the registrations
# --------------------------------------------------------------------------

def test_register_check_recomputes_every_committed_number():
    got = subprocess.run([sys.executable, str(R3DIR / "register.py"), "--check"], capture_output=True,
                         text=True, timeout=900, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin",
                                                      "HOME": str(Path.home())})
    assert got.returncode == 0, got.stdout + got.stderr
    assert got.stdout.count(": same") == 12


def test_the_registrations_carry_their_labels_and_no_typographic_dash():
    for part in C3.PARTS:
        for ext in ("json", "txt"):
            text = (REPO / "docs" / "registered" / f"{C3.NAMES[part]}.{ext}").read_text()
            assert "\u2014" not in text and "\u2013" not in text, (part, ext)
        doc = C3.registration(REPO, part)
        assert doc["status"].startswith("REGISTERED BEFORE ANY PAGE")
        assert doc["seen_data"], part
    assert "NOT a test of transfer across expert counts" in E2E["scope"]
    assert set(E2E["parent_inputs"]) >= {"routing and declaration", "w2 grid", "w1 per-CTA unit",
                                         "dead-CTA term", "what is not the parent's"}


def test_the_e2e_population_is_the_host_rules_31_core_cells():
    pop = E2E["population"]
    assert pop["core_cells"] == 31
    for c in pop["core"]:
        assert c["T_M"] + E2E["host_rule"]["F240_ms"] >= 0.40
    got = {(c["arm"], c["G"]): sorted(x["n"] for x in pop["core"] if (x["arm"], x["G"]) == (c["arm"], c["G"]))
           for c in pop["core"]}
    assert got[("shared", 3)] == [3, 4, 5, 6, 7, 8] and got[("private", 3)] == [2, 3, 4, 5, 6, 7, 8]
    for G in (8, 32):
        assert got[("shared", G)] == [3, 4, 5, 6] and got[("private", G)] == [2, 3, 4, 5, 6]


def test_the_timing_models_gemm_constant_is_the_registered_cal_aff_fit():
    import r3_timing_model as TM
    g, b = ZF["fits_CAL"]["AFF"]["coef"]
    assert TM.GEMM_CONST_CYCLES == pytest.approx(g, abs=1.0)
    assert TM.GEMM_CONST_PER_U == pytest.approx(b, abs=5e-5)
    assert E2E["gemm_const"]["cycles"] == TM.GEMM_CONST_CYCLES


def test_e6s_bands_separate_the_dead_term_from_the_parents_count():
    b = E2E["E6"]["bands_w2"]
    assert b["FIXED"][1] < b["DEAD"][0]
    assert E2E["E6"]["predicted"]["w1"]["pred_us"] == 0.0


def test_every_one_page_pair_is_registered_and_the_patterns_are_distinct():
    pats = STG["patterns_at_expected"]
    knob = [p for p in STG["pages"]["knobs"] if p != "k64s3"]
    keys = {h: tuple(pats[h][p] for p in knob) for h in pats}
    assert len(set(keys.values())) == len(keys)
    assert set(STG["pairs"]["one_page_pairs"]) == {"OCC/U-OCC", "WIDTH/U-OCC", "U-OCC/U-DEPTH"}


# --------------------------------------------------------------------------
# floor_estimator: the joint fit and the rulers
# --------------------------------------------------------------------------

def test_joint_theta_recovers_a_planted_law_and_its_covariance_shrinks_with_noise():
    rng = np.random.default_rng(0)
    q = np.linspace(2.3, 30.7, 14)
    frac = np.ceil(q) - q
    y = 0.3 + 1.0 * q + 0.6 * frac + rng.normal(0, 0.01, len(q))
    jt = FE.joint_theta(y, q, frac)
    assert jt["theta"] == pytest.approx(0.6, abs=0.05) and jt["b"] == pytest.approx(1.0, abs=0.01)
    assert 0 < jt["se_theta"] < 0.05
    with pytest.raises(ValueError):
        FE.ols([1, 2, 3], [[1, 1, 1], [1, 2, 3], [0, 1, 2]])


def test_rulers_read_dur_x_and_gap_in_cycles():
    pg = {"sm__cycles_elapsed.avg": 10000.0, "sm__cycles_elapsed.max": 10100.0, "sm__cycles_active.max": 9500.0,
          "sm__cycles_active.avg": 9000.0, "gpu__time_duration.sum": 6000.0, "lts__cycles_elapsed.avg": 10224.0}
    r = FE.rulers(pg, 1710.0)
    assert r["DUR"] == pytest.approx(10260.0) and r["X"] == pytest.approx(260.0) and r["GAP"] == 500.0
    assert r["L2_mhz"] == pytest.approx(1704.0)
    assert FE.rulers(pg, None)["DUR"] is None


# --------------------------------------------------------------------------
# synthetic floor captures
# --------------------------------------------------------------------------

def floor_cells(model):
    return FLR["cells_registered"][model]


def make_floor(tree, model, label, clk, law, *, null_rel=-0.014, l2=None, smi=None, fl1="PASS",
               v1="PASS", seed=0, treads=None):
    """A G = 64 NATIVE floor capture whose per-GEMM counters follow `law(c, clk)` ->
    dict(ACT.avg, ACT.max, EL.avg, DUR) in cycles (c a registered cell)."""
    rng = np.random.default_rng(seed)
    lock = None if clk == "base" else float(clk[4:])
    by_n = {}
    for c in floor_cells(model):
        if treads and c["n"] not in treads:
            continue
        m = law(c, clk)
        u = c["u"]
        noise = lambda: float(rng.normal(0, 0.002 * u))  # noqa: E731
        aa, am = m["ACT.avg"] + noise(), m["ACT.max"] + noise()
        el = m["EL.avg"] + noise()
        dur_cyc = m["DUR"] + noise()
        dur = dur_cyc / F_RULER[clk] * 1e3
        pg = {"sm__cycles_active.avg": aa, "sm__cycles_active.max": am, "sm__cycles_elapsed.avg": el,
              "sm__cycles_elapsed.max": el * 1.005, "gpu__time_duration.sum": dur,
              "lts__cycles_elapsed.avg": (l2 or L2_MHZ[clk]) * dur / 1e3, "sm_clock_mhz": F_MEAS[clk]}
        cell = by_n.setdefault(c["n"], {"arm": "native", "n": c["n"], "per_gemm": {}})
        cell["per_gemm"][c["gemm"]] = pg
        if lock:
            cell["null"] = {"sm_clock_mhz": lock * (1 + null_rel)}
    page = {"plan": {"model": model}, "cells": [by_n[n] for n in sorted(by_n)],
            "gates": [{"number": "FL1", "kind": "VALIDITY", "verdict": fl1},
                      {"number": "V1", "kind": "VALIDITY", "verdict": v1}]}
    if lock:
        s = smi if smi is not None else lock
        page["clock"] = {"lock_mhz": lock, "smi_before": {"rows": [{"clocks.sm": str(s)}]},
                         "smi_after": {"rows": [{"clocks.sm": str(s)}]}}
    d = tree / f"{DAY}-{CARD}-{model}-{label}-r3-counters"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{STEM[clk]}.json").write_text(json.dumps(page))
    return page


def ceil_law(Z=3000.0, theta_el=1.0, X0=800.0, xfrac=0.0):
    """ACT.max on the ceil law, EL.avg = ACT.max + gap - (1 - theta_el) frac u, DUR = EL + X."""
    def law(c, clk):
        u, q, fr = c["u"], c["q"], c["frac"]
        aa = Z + q * u
        am = Z + math.ceil(q) * u
        el = am + 2000.0 - (1 - theta_el(clk) if callable(theta_el) else 1 - theta_el) * fr * u
        x = X0 + xfrac * fr * u
        return {"ACT.avg": aa, "ACT.max": am, "EL.avg": el, "DUR": el + x}
    return law


def fluid_law(c, clk):
    u, q = c["u"], c["q"]
    am = 3000.0 + q * u + 500.0
    return {"ACT.avg": 3000.0 + q * u, "ACT.max": am, "EL.avg": am + 2000.0, "DUR": am + 2800.0}


def all_captures(tree, law, models=None, **kw):
    for model, labels in FLR["captures"].items():
        if models and model not in models:
            continue
        for label, spec in labels.items():
            for clk in spec["clocks"]:
                make_floor(tree, model, label, clk, law, **kw)


def clock_theta(clk):
    return 0.35 if clk in ("lock1710", "base") else 1.0


def test_lock_in_force_reads_the_null_kernel_the_l2_at_1710_only_and_nvidia_smi(tmp_path):
    rules = FLR["gates"]["floor"]
    law = ceil_law()
    ok = make_floor(tmp_path, "mixtral-8x7b-tp4", "a", "lock1710", law)
    assert C3.lock_in_force(ok, rules)["ok"] is True
    assert C3.lock_in_force(make_floor(tmp_path, "mixtral-8x7b-tp4", "b", "lock1710", law, l2=1600.0),
                            rules)["ok"] is False
    # at 1005 the L2 follows the lock down: printed, not gated
    assert C3.lock_in_force(make_floor(tmp_path, "mixtral-8x7b-tp4", "c", "lock1005", law), rules)["ok"] is True
    assert C3.lock_in_force(make_floor(tmp_path, "mixtral-8x7b-tp4", "d", "lock1710", law, null_rel=0.056),
                            rules)["ok"] is False
    assert C3.lock_in_force(make_floor(tmp_path, "mixtral-8x7b-tp4", "e", "lock1710", law, smi=1980),
                            rules)["ok"] is False
    assert C3.lock_in_force(make_floor(tmp_path, "mixtral-8x7b-tp4", "f", "base", law), rules) is None


def test_a_short_k_capture_prints_the_l2_leg_and_the_null_kernel_and_smi_still_gate(tmp_path):
    rules = FLR["gates"]["floor"]
    # short cells read low (SEEN: a fixed offset of about 0.75 us); never gated on them
    page = make_floor(tmp_path, "granite-3.0-1b-a400m", "floor", "lock1710", ceil_law(), l2=1676.0)
    longest = max(pg["gpu__time_duration.sum"] for c in page["cells"] for pg in c["per_gemm"].values())
    assert longest < 5e5
    assert C3.l2_clock_mhz(page, 5e5) is None
    chk = C3.lock_in_force(page, rules)
    assert chk["ok"] is True and "printed, not gated" in chk["why"]
    bad = make_floor(tmp_path, "granite-3.0-1b-a400m", "floor2", "lock1710", ceil_law(), smi=1980)
    assert C3.lock_in_force(bad, rules)["ok"] is False
    bad = make_floor(tmp_path, "granite-3.0-1b-a400m", "floor3", "lock1710", ceil_law(), null_rel=0.056)
    assert C3.lock_in_force(bad, rules)["ok"] is False


def test_the_floor_law_selects_clock_where_the_1710_flattening_sits_in_gap(tmp_path):
    all_captures(tmp_path, ceil_law(theta_el=clock_theta))
    res = SF.score(REPO, tmp_path)
    assert res["hypotheses"] == {"CLOCK": "HOLDS", "CEIL": "FALSIFIED", "FLUID": "FALSIFIED"}, res["tests"]
    assert res["verdict"] == "CLOCK (ncu-measured)"
    assert res["tests"]["B0"]["verdict"] == "HOLDS"
    assert res["tests"]["B6"]["verdict"].startswith("NOT RUN")


def test_the_floor_law_selects_ceil_and_fluid_in_their_worlds(tmp_path):
    all_captures(tmp_path / "c", ceil_law(theta_el=1.0))
    assert SF.score(REPO, tmp_path / "c")["verdict"] == "CEIL"
    all_captures(tmp_path / "f", fluid_law)
    res = SF.score(REPO, tmp_path / "f")
    assert res["verdict"] == "FLUID", res["hypotheses"]


def test_b4s_clock_clause_needs_a_blind_gemm_not_the_tp4_replicate_alone(tmp_path):
    """Only tp4 flattens at 1710 (the SEEN-chosen narrowing's case); the blind granite-1B
    and qwen2-tp8 GEMMs stay ceil: CLOCK's B4 clause fails, so CLOCK is not selected."""
    all_captures(tmp_path, ceil_law(theta_el=1.0))
    all_captures(tmp_path, ceil_law(theta_el=clock_theta), models={"mixtral-8x7b-tp4"})
    res = SF.score(REPO, tmp_path)
    b4 = res["tests"]["B4"]
    assert sum(v["theta_DUR"] <= v["theta_ACT_max"] - 0.20 for k, v in b4["per_gemm"].items()
               if k.startswith("mixtral-8x7b-tp4")) >= 2
    assert b4["CLOCK_clause"] is False
    assert res["hypotheses"]["CLOCK"] != "HOLDS" and not res["verdict"].startswith("CLOCK")


def test_b0_falsifies_an_x_that_moves_with_frac(tmp_path):
    all_captures(tmp_path, ceil_law(theta_el=clock_theta, xfrac=-0.5))
    res = SF.score(REPO, tmp_path)
    assert res["tests"]["B0"]["verdict"] == "FALSIFIED"
    assert res["verdict"] == "INCONCLUSIVE"


def test_a_capture_whose_lock_was_not_in_force_leaves_clean_and_the_views_disagree(tmp_path):
    all_captures(tmp_path, ceil_law(theta_el=1.0))
    # every 1710 capture read 1980 on nvidia-smi: CLEAN loses B4's GEMMs
    for model, labels in FLR["captures"].items():
        for label, spec in labels.items():
            if "lock1710" in spec["clocks"]:
                make_floor(tmp_path, model, label, "lock1710", ceil_law(theta_el=1.0), smi=1980)
    res = SF.score(REPO, tmp_path)
    assert res["verdict"].startswith("NOT SCORED (CLEAN has no data); ALL")
    assert "CEIL" in res["verdict"]


def wall_csv(tree, theta, mode="GR"):
    d = tree / f"{DAY}-{CARD}-launch-floor-wall" / "mixtral-8x7b-tp4"
    d.mkdir(parents=True, exist_ok=True)
    W = {int(k): v for k, v in FLR["tests"]["B6"]["W_ms"].items()}
    with open(d / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "arm", "tiles", "repeat", "status", "ms_p50"])
        w.writeheader()
        for n, x in W.items():
            for rep in range(3):
                for m in ("GR", "E240"):
                    w.writerow({"mode": m, "arm": "native", "tiles": n, "repeat": rep, "status": "ok",
                                "ms_p50": 0.05 + 0.03 * n + theta * x + (0.03 if m == "E240" else 0.0)})
    (d / "manifest.json").write_text(json.dumps({"phase": "timed", "profiler_enabled_ever": False}))


@pytest.mark.parametrize("theta, want", [(1.0, "NCU REPLAY ONLY"), (0.0, "IN WALL TIME"), (0.6, "INCONCLUSIVE")])
def test_b6_reads_the_wall_time_floor_outside_ncu(tmp_path, theta, want):
    wall_csv(tmp_path, theta)
    got = SF.wall_fit(FLR, tmp_path)
    assert got["verdict"] == want and got["GR"]["theta_wall"] == pytest.approx(theta, abs=1e-6)


def test_b6_labels_a_clock_selection(tmp_path):
    all_captures(tmp_path, ceil_law(theta_el=clock_theta))
    wall_csv(tmp_path, 0.0)
    assert SF.score(REPO, tmp_path)["verdict"] == "CLOCK (in wall time)"


# --------------------------------------------------------------------------
# part A on synthetic captures
# --------------------------------------------------------------------------

def zlaw(form, *, base_scale=1.0):
    def law(c, clk):
        u, q = c["u"], c["q"]
        z = SZ.zpred(ZF, form, u, F_MEAS[clk]) * (base_scale if clk == "base" else 1.0)
        aa = z + q * u
        am = z + math.ceil(q) * u
        return {"ACT.avg": aa, "ACT.max": am, "EL.avg": am + 2000.0, "DUR": am + 2800.0}
    return law


def z_captures(tree, law):
    for model in ("qwen2-57b-a14b-tp8", "granite-3.0-1b-a400m"):
        make_floor(tree, model, "floor", "base", law)
        make_floor(tree, model, "floor", "lock1710", law)
        make_floor(tree, model, "floor1005", "lock1005", law)


@pytest.mark.parametrize("form", ["AFF", "MIX"])
def test_part_a_selects_the_form_its_captures_were_built_under(tmp_path, form):
    z_captures(tmp_path, zlaw(form))
    res = SZ.score(REPO, tmp_path)
    assert res["verdict"] == form, res["forms"]
    assert res["forms"]["PROP"]["status"] == "FALSIFIED"


def test_a_constant_in_ns_gives_the_clock_ratio_and_prop_fails_a1_at_small_u(tmp_path):
    z_captures(tmp_path, zlaw("MIX"))
    res = SZ.score(REPO, tmp_path)
    for key in ("qwen2-57b-a14b-tp8 w2", "granite-3.0-1b-a400m w2"):
        g = res["gemms"][key]
        assert g["A2"]["r"] == pytest.approx(g["A2"]["r_MIX"], abs=0.02)
        assert g["A1"]["PROP"]["inside"] is False


def test_mid_is_empty_on_the_primary_lever_and_read_on_the_printed_1005_one(tmp_path):
    for key in ("qwen2-57b-a14b-tp8 w2", "granite-3.0-1b-a400m w2", "granite-3.0-1b-a400m w1"):
        pw = ZF["separating_gemms"][key]["power"]
        assert pw["base"]["MID_possible"] is False and pw["lock1005"]["MID_possible"] is True

    def law(c, clk):
        base = zlaw("AFF")(c, clk)
        if clk == "lock1005":
            u = c["u"]
            mid = 0.5 * (1 + SZ.zpred(ZF, "MIX", u, F_MEAS["lock1005"]) / SZ.zpred(ZF, "MIX", u, F_MEAS["lock1710"]))
            z = SZ.zpred(ZF, "AFF", u, 0) * mid
            base = dict(base, **{"ACT.avg": z + c["q"] * u})
        return base
    z_captures(tmp_path, law)
    res = SZ.score(REPO, tmp_path)
    assert res["MID_1005_printed"]["reads_MID"] is True, res["MID_1005_printed"]
    assert res["verdict"] == "AFF"      # the primary lever and A1 read AFF


def test_a_curved_intercept_is_not_scored(tmp_path):
    def law(c, clk):
        m = zlaw("AFF")(c, clk)
        if c["n"] >= 8:
            m["ACT.avg"] += 4000.0 + 300.0 * c["n"]
        return m
    z_captures(tmp_path, law)
    res = SZ.score(REPO, tmp_path)
    assert res["gemms"]["qwen2-57b-a14b-tp8 w2"]["verdict"].startswith("NOT SCORED: the half-fits")


# --------------------------------------------------------------------------
# part C on synthetic G = 1 byte pages
# --------------------------------------------------------------------------

def r3c_page(model, *, s=None, ctas=None, block_k=64, num_stages=4, v6="PASS", fmn=0.53, far=0.25):
    """rental 2's synthetic G = 1 page (tests/test_scoring_rental2.py r3c_page), with
    the far share of SHARED's extra hits planted."""
    cfg = MODEL_CONFIGS[model]
    bm = DCR.r3_byte_model(cfg, "bf16", 32)
    W = {"w1": bm["W_w1"], "w2": bm["W_w2"]}
    op = {"w1": bm["operand_per_tile_w1"], "w2": bm["operand_per_tile_w2"]}
    ctas = ctas or {"w1": 5, "w2": 4}
    s = s or {"w1": {}, "w2": {}}
    cells = []
    for arm in ("native", "shared", "private"):
        for n in range(1, 10):
            per_gemm, rec = {}, {}
            for g in ("w1", "w2"):
                b_priv = n * W[g] + n * op[g]
                b = b_priv if arm == "private" else b_priv - (s[g].get(n, 0.0) * (n - 1) * W[g] if n >= 2 else 0.0)
                T, Hh = 1.0e6, (4.0e5 if arm == "private" else 4.0e5 + 1.0e5 * (1 - far))
                Mn = T - Hh
                F = fmn * Mn if arm == "private" else fmn * (T - 4.0e5) + (-1.0e5 * far)
                per_gemm[g] = {"dram_bytes_read": b, "l2_tex_read_sectors": T, "l2_tex_read_hit_sectors": Hh,
                               "l2_read_miss_sectors": (F - 0.5 * F) + Mn, "grid_size": 1}
                rec[g] = {"launch__occupancy_limit_blocks": 32, "launch__occupancy_limit_registers": ctas[g],
                          "launch__occupancy_limit_shared_mem": 32, "launch__occupancy_limit_warps": 8,
                          L2S.FABRIC: F, L2S.FABRIC_HIT: 0.5 * F}
            cells.append({"arm": arm, "n": n, "per_gemm": per_gemm, "recorded": rec,
                          "per_call": {"dram_bytes_read": sum(per_gemm[g]["dram_bytes_read"] for g in per_gemm)}})
    E = cfg.num_experts
    return {"family": "r3-arms", "run_id": f"synthetic-{model}",
            "design": {"model": model, "dtype": "bf16", "block_m": 32, "block_n": 64, "block_k": block_k,
                       "num_warps": 8, "num_stages": num_stages, "copies_declared": 9, "group_m": 1,
                       "declared_by_arm": {"native": E, "shared": 9 * E, "private": 9 * E}},
            "card": {"name": "NVIDIA GH200 480GB", "slug": CARD, "uuid": "GPU-synthetic",
                     "capability": "9.0", "sm_count": 132, "l2_bytes": 62914560},
            "cells": cells, "gates": [{"number": "V6", "kind": "VALIDITY", "verdict": v6},
                                      {"number": "V10", "kind": "VALIDITY", "verdict": "FAIL"}]}


def put_bytes(tree, model, label, page, G=1):
    d = tree / f"{DAY}-{CARD}-{model}-{label}-r3-counters" / "lock1710"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"r3c-g{G}.json").write_text(json.dumps(page))


S_BASE = {n: float(v) for n, v in STG["controls"]["board_check"]["rental2_s_w2_SEEN"].items()}


def stage_pages(tree, truth, *, ctas_override=None, flag=None, control_lift=False, base_shift=0.0):
    base = {"w1": {int(n): 0.05 for n in S_BASE}, "w2": {int(n): v + base_shift for n, v in S_BASE.items()}}
    put_bytes(tree, SS.MODEL, "l2base", r3c_page(SS.MODEL, s=base))
    for label, kn in STG["pages"]["knobs"].items():
        ctas = (ctas_override or {}).get(label, STG["occupancy"]["expected"][label])
        lifted = SS.classify(truth, ctas, kn["num_stages"], kn["block_k"]) == "lifted"
        if label == "k64s3" and control_lift:
            lifted = True
        s2 = {int(n): v + base_shift + (0.2 if lifted else 0.0) for n, v in S_BASE.items()}
        page = r3c_page(SS.MODEL, s={"w1": base["w1"], "w2": s2}, ctas={"w1": 5, "w2": ctas},
                        block_k=kn["block_k"], num_stages=kn["num_stages"],
                        v6="FAIL" if flag == label else "PASS")
        put_bytes(tree, SS.MODEL, label, page)


@pytest.mark.parametrize("truth", ["OCC", "DEPTH", "WIDTH", "U-OCC", "U-DEPTH"])
def test_part_c_selects_the_pattern_its_pages_encode(tmp_path, truth):
    stage_pages(tmp_path, truth)
    res = SS.score(REPO, tmp_path)
    assert res["verdict"] == truth, (res.get("hypotheses"), res.get("pages"))


def test_an_unexpected_occupancy_rekeys_the_column_and_unseparates_its_pair(tmp_path):
    """k128s3 records 2 CTAs/SM, not 3: OCC and U-OCC then predict the same everywhere."""
    stage_pages(tmp_path, "OCC", ctas_override={"k128s3": 2})
    res = SS.score(REPO, tmp_path)
    assert res["pages"]["k128s3"]["ctas_recorded"] == 2 and "k128s3_check" in res["pages"]["k128s3"]
    assert res["pairs"]["OCC/U-OCC"] == "NO SEPARATION"
    assert res["verdict"] == "INCONCLUSIVE" and set(res["survivors"]) == {"OCC", "U-OCC"}


def test_a_v6_flagged_page_leaves_the_count(tmp_path):
    stage_pages(tmp_path, "WIDTH", flag="k64s7")
    res = SS.score(REPO, tmp_path)
    assert "k64s7" not in res["classified"] and res["pages"]["k64s7"]["use"].startswith("printed")
    # k64s7 alone separates WIDTH from U-OCC
    assert res["pairs"]["WIDTH/U-OCC"] == "NO SEPARATION" and res["verdict"] == "INCONCLUSIVE"


def test_a_failed_board_check_turns_every_verdict_inconclusive(tmp_path):
    stage_pages(tmp_path, "DEPTH", base_shift=0.12)
    res = SS.score(REPO, tmp_path)
    assert res["board_check"]["ok"] is False and res["verdict"].startswith("INCONCLUSIVE")


def test_a_lifted_control_falsifies_every_hypothesis(tmp_path):
    stage_pages(tmp_path, "U-DEPTH", control_lift=True)
    assert SS.score(REPO, tmp_path)["verdict"].startswith("NONE")


# --------------------------------------------------------------------------
# part E on synthetic timed, byte and floor pages
# --------------------------------------------------------------------------

def predicted(arm, G, n, mz=False):
    for c in E2E["population"]["core"] + E2E["population"]["deep"]:
        if (c["arm"], c["G"], c["n"]) == (arm, G, n):
            return c["T_MZ"] if mz else c["T_M"]
    return 0.2   # host-bound cells the rule excludes


def put_r3(tree, label, tag, G, treads, ms_of, *, fail=(), run=None):
    d = (tree / f"gaps-{CARD}-{C3.TARGET}-{label}" / "private_weight_reference"
         / (run or f"run-{tag}-g{G}"))
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"arm": arm, "tiles": n, "ms_p50": ms_of(arm, G, n), "sm_clock_load_mhz": 1710.0}
            for arm in ("native", "shared", "private") for n in range(1, treads + 1)]
    gates = [{"tag": f"V{i}", "kind": "VALIDITY", "verdict": "FAIL" if f"V{i}" in fail else "PASS"} for i in range(9)]
    (d / "report.json").write_text(json.dumps({"session_tag": f"gh200-x-{tag}-lock1710",
                                               "pinned": {"GROUP_SIZE_M": G}, "treads_table": rows,
                                               "gates": gates}))


def e2e_pages(tree, ms_of, *, fail=None, rep_scale=1.0):
    fail = fail or {}
    put_r3(tree, "e2e", "p2", 8, 6, ms_of, fail=fail.get(8, ()))
    put_r3(tree, "e2e", "p2", 32, 6, ms_of, fail=fail.get(32, ()))
    put_r3(tree, "e2e", "p5", 3, 8, ms_of, fail=fail.get(3, ()))
    for G in (4, 2):
        put_r3(tree, "e2e", "deep", G, 9, ms_of)
    put_r3(tree, "rep8", "p2b", 8, 6, lambda a, G, n: ms_of(a, G, n) * rep_scale)


def noisy(scale=1.0, sd=0.01, seed=1, mz=False):
    rng = np.random.default_rng(seed)
    cache = {}

    def ms(arm, G, n):
        k = (arm, G, n)
        if k not in cache:
            cache[k] = predicted(arm, G, n, mz) * scale * (1 + float(rng.normal(0, sd)))
        return cache[k]
    return ms


def no_e2(*a, **k):
    return {"verdict": "NOT SCORED: E2 not run in this test"}


def test_e1_holds_on_pages_at_the_prediction_with_one_percent_noise(tmp_path):
    e2e_pages(tmp_path, noisy(sd=0.007))
    res = SE.score(REPO, tmp_path, e2=no_e2)
    assert res["E1"]["verdict"] == "HOLDS", res["E1"]["stats"]
    assert res["E1"]["stats"]["cells"] == 31
    assert res["E1"]["label"].startswith(("resolved", "UNRESOLVED"))


def test_e1_falsifies_a_three_percent_offset(tmp_path):
    e2e_pages(tmp_path, noisy(scale=1.03, sd=0.002))
    assert SE.score(REPO, tmp_path, e2=no_e2)["E1"]["verdict"] == "FALSIFIED"


def test_a_host_bound_n1_cell_is_excluded_by_the_rule_not_by_its_data(tmp_path):
    base = noisy(sd=0.002)

    def ms(arm, G, n):
        return base(arm, G, n) * (3.0 if n == 1 else 1.0)
    e2e_pages(tmp_path, ms)
    res = SE.score(REPO, tmp_path, e2=no_e2)
    assert res["E1"]["verdict"] == "HOLDS"
    assert all(c["n"] >= 2 for c in res["E1"]["cells"])


def test_a_page_whose_only_failure_is_v5_is_counted_and_a_v1_failure_drops_it_in_both_views(tmp_path):
    e2e_pages(tmp_path / "a", noisy(sd=0.002), fail={8: ("V5",), 32: ("V5",), 3: ("V5",)})
    res = SE.score(REPO, tmp_path / "a", e2=no_e2)
    assert res["E1"]["verdict"] == "HOLDS" and res["E1"]["stats"]["cells"] == 31
    e2e_pages(tmp_path / "b", noisy(sd=0.002), fail={3: ("V1",)})
    res = SE.score(REPO, tmp_path / "b", e2=no_e2)
    assert res["E1"]["stats"]["cells"] == 18
    pages = res["addendum"]["pages"]
    k = next(k for k in pages if k.endswith("run-p5-g3"))
    assert pages[k]["ALL"]["use"].startswith("unusable") and pages[k]["CLEAN"]["use"].startswith("unusable")


def test_a_v8_failure_drops_the_page_in_clean_only(tmp_path):
    e2e_pages(tmp_path, noisy(sd=0.002), fail={3: ("V8",), 8: ("V8",), 32: ("V8",)})
    res = SE.score(REPO, tmp_path, e2=no_e2)
    assert res["E1"]["verdict"].startswith("NOT SCORED (CLEAN has no data); ALL")


def test_mz_reads_lower_rms_on_pages_built_from_mz_and_e5_stays_printed(tmp_path):
    e2e_pages(tmp_path, noisy(sd=0.001, mz=True))
    res = SE.score(REPO, tmp_path, e2=no_e2)
    e5 = res["E5_printed"]
    assert e5["rms_MZ"] < e5["rms_M"] and "PRINTED" in e5["status"]


def test_rep8_gives_sigma_page_and_the_resolution(tmp_path):
    e2e_pages(tmp_path, noisy(sd=0.0), rep_scale=1.004)
    sp = SR.sigma_page(REPO, tmp_path, C3.TimedView("ALL"))
    assert sp["sigma_page"] == pytest.approx(0.004, rel=1e-6) and len(sp["cells"]) == 9
    res = SE.score(REPO, tmp_path, e2=no_e2)
    sb = REP["R"]["sigma_board"]["rms"]
    assert res["E1"]["resolution"] == pytest.approx(math.sqrt(0.004 ** 2 + sb ** 2))


def test_e2_reads_the_injected_counted_bytes_score(tmp_path):
    e2e_pages(tmp_path, noisy(sd=0.0))
    for G in (3, 8, 32):
        put_bytes(tmp_path, C3.TARGET, "e2e", qwen_bytes(G), G=G)
    calls = []

    def fake(reg, repo, dirs, counters):
        calls.append((len(dirs), counters.name))
        return {"cells": {(c["arm"], c["G"], c["n"]): 0.004 for c in E2E["population"]["core"]}}
    res = SE.score(REPO, tmp_path, e2=fake)
    assert res["E2"]["verdict"] == "HOLDS" and calls and calls[0] == (3, "lock1710")


def qwen_bytes(G, *, q_scale=1.0, dead_us=None, v10="PASS"):
    """A byte page of the target whose PRIVATE / SHARED q is the registered prediction
    times q_scale and whose SHARED - NATIVE w2 in-kernel time is dead_us."""
    cfg = MODEL_CONFIGS[C3.TARGET]
    bm = DCR.r3_byte_model(cfg, "bf16", 32)
    dead_us = E2E["E6"]["predicted"]["w2"]["pred_us"] if dead_us is None else dead_us
    cells = []
    for arm in ("native", "shared", "private"):
        for n in range(1, 10):
            per_gemm, rec = {}, {}
            for g in ("w1", "w2"):
                qa = "shared" if arm == "native" else arm
                q = E2E["q_pred"][f"{qa} G={G} n={n}"][g] * q_scale
                per_gemm[g] = {"dram_bytes_read": q * bm[f"W_{g}"] + n * bm[f"operand_per_tile_{g}"]}
                el = 50000.0 + (dead_us * 1710.0 if (arm == "shared" and g == "w2") else 0.0)
                rec[g] = {"sm__cycles_elapsed.avg": el}
            cells.append({"arm": arm, "n": n, "per_gemm": per_gemm, "recorded": rec,
                          "per_call": {"dram_bytes_read": sum(v["dram_bytes_read"] for v in per_gemm.values())}})
    return {"design": {"model": C3.TARGET, "dtype": "bf16", "block_m": 32, "group_m": G}, "cells": cells,
            "gates": [{"number": "V10", "kind": "VALIDITY", "verdict": v10}]}


def test_a_v10_failure_drops_es_byte_pages_in_clean(tmp_path):
    """V10 gates CLEAN for E3, E3s and E6 (E6 converts cycles at 1710 MHz): a page whose
    only failure is V10 is counted by ALL and not by CLEAN, so the verdicts read NOT SCORED."""
    for G in (3, 8, 32):
        put_bytes(tmp_path, C3.TARGET, "e2e", qwen_bytes(G, v10="FAIL"), G=G)
    res = SE.score(REPO, tmp_path, e2=no_e2)
    for k in ("E3", "E6"):
        assert res[k]["verdict"].startswith("NOT SCORED (CLEAN has no data); ALL"), res[k]


def test_e3_holds_at_the_predicted_bytes_and_falsifies_a_ten_percent_miss(tmp_path):
    for G in (3, 8, 32):
        put_bytes(tmp_path / "a", C3.TARGET, "e2e", qwen_bytes(G), G=G)
        put_bytes(tmp_path / "b", C3.TARGET, "e2e", qwen_bytes(G, q_scale=1.10), G=G)
    a = SE.score(REPO, tmp_path / "a", e2=no_e2)
    assert a["E3"]["verdict"] == "HOLDS" and a["E3"]["cells"] == 54
    assert SE.score(REPO, tmp_path / "b", e2=no_e2)["E3"]["verdict"] == "FALSIFIED"


@pytest.mark.parametrize("which, want", [("DEAD", "DEAD"), ("FIXED", "FIXED")])
def test_e6_separates_the_dead_term_from_the_parents_exposure(tmp_path, which, want):
    us = (E2E["E6"]["predicted"]["w2"]["pred_us"] if which == "DEAD"
          else E2E["E6"]["parent_counted_SEEN_us"]["w2"])
    for G in (3, 8, 32):
        put_bytes(tmp_path, C3.TARGET, "e2e", qwen_bytes(G, dead_us=us), G=G)
    assert SE.score(REPO, tmp_path, e2=no_e2)["E6"]["verdict"] == want


def test_e4_reads_the_w2_unit_on_the_1005_dur_and_its_f0_rival(tmp_path):
    make_floor(tmp_path / "m", C3.TARGET, "floor1005", "lock1005", ceil_law())
    assert SE.score(REPO, tmp_path / "m", e2=no_e2)["E4"]["verdict"].startswith("M HOLDS")
    b0 = E2E["E4"]["F0_rival_b"]

    def f0(c, clk):
        m = ceil_law()(c, clk)
        if c["gemm"] == "w2":
            m["DUR"] = 2000.0 + math.ceil(c["q"]) * c["u"] * b0
        return m
    make_floor(tmp_path / "f", C3.TARGET, "floor1005", "lock1005", f0)
    assert SE.score(REPO, tmp_path / "f", e2=no_e2)["E4"]["verdict"].startswith("F0 HOLDS")


# --------------------------------------------------------------------------
# part D on a synthetic launch-floor directory
# --------------------------------------------------------------------------

def flush_dir(tree, phi_of):
    d = tree / f"{DAY}-{CARD}-launch-floor-r3" / "mixtral-8x7b-tp8"
    d.mkdir(parents=True, exist_ok=True)
    F = FLU["F_gpu_ms"]
    modes = ("E0", "E240", "E360", "E480")
    cr = FLU["C_reg_ms"]
    with open(d / "cells.csv", "w", newline="") as f, open(d / "hostprobe.csv", "w", newline="") as hp:
        w = csv.DictWriter(f, fieldnames=["mode", "arm", "tiles", "repeat", "status", "ms_p50", "host_enqueue_ms",
                                          "calls_per_burst", "host_bound"])
        p = csv.DictWriter(hp, fieldnames=["mode", "arm", "tiles", "status", "H_pre_ms"])
        w.writeheader()
        p.writeheader()
        for arm in ("native", "shared", "private"):
            for n in (1, 2, 3):
                H = float(cr[arm][str(n)]) + F["E480"] + 0.10
                for m in modes:
                    p.writerow({"mode": m, "arm": arm, "tiles": n, "status": "ok", "H_pre_ms": H})
                    for rep in range(3):
                        w.writerow({"mode": m, "arm": arm, "tiles": n, "repeat": rep, "status": "ok",
                                    "ms_p50": H - phi_of(F[m], n), "host_enqueue_ms": H * 40,
                                    "calls_per_burst": 40, "host_bound": True})
    (d / "hostprobe-post.csv").write_text((d / "hostprobe.csv").read_text())
    (d / "manifest.json").write_text(json.dumps({"phase": "timed", "profiler_checks": 9,
                                                 "profiler_enabled_ever": False}))


def test_part_d_selects_overlap_and_sat_in_their_worlds(tmp_path):
    flush_dir(tmp_path / "o", lambda F, n: F - 0.010 + (0.03 if n == 1 else 0.005))
    res = SH.score(REPO, tmp_path / "o")
    assert res["verdict"] == "OVERLAP", res.get("hypotheses")
    assert all(abs(r["slope"] - 1) < 1e-6 for r in res["cells"] if "slope" in r)
    flush_dir(tmp_path / "s", lambda F, n: min(F, 0.0675) + 0.02)
    assert SH.score(REPO, tmp_path / "s")["verdict"] == "SAT"


def test_part_d_is_not_tested_without_three_qualifying_cells(tmp_path):
    flush_dir(tmp_path, lambda F, n: F)
    cells = (next(tmp_path.rglob("cells.csv")))
    rows = list(csv.DictReader(open(cells)))
    for r in rows:
        r["host_bound"] = "False"
    with open(cells, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert SH.score(REPO, tmp_path)["verdict"].startswith("NOT TESTED")


# --------------------------------------------------------------------------
# RK: the K1 / K4 replicate
# --------------------------------------------------------------------------

def test_rk_forms_a_noise_floor_and_resolves_a_pool_far_from_its_thresholds(tmp_path):
    law = ceil_law()
    make_floor(tmp_path, "mixtral-8x7b-tp4", "floor", "lock1710", law, seed=1)
    make_floor(tmp_path, "mixtral-8x7b-tp4", "floorrep", "lock1710", law, seed=2)
    res = SR.score(REPO, tmp_path)
    rk = res["RK"]
    for g in ("w1", "w2"):
        # three noisy counters, each 0.002 u, enter L and D_imb: sigma about sqrt(2) x 0.002 u
        u = next(c["u"] for c in floor_cells("mixtral-8x7b-tp4") if c["gemm"] == g)
        assert 0.5 * 0.002 * u < rk["per_gemm"][g]["sigma_L"] < 3 * 0.002 * u
    assert rk["rental3_tp4_K1"]["verdict"] == "RESOLVED"
    assert rk["rental3_tp4_K4"]["verdict"] == "RESOLVED"


def test_k1_resolution_flips_the_ambiguous_cells_both_ways():
    sig = {"w1": 300.0}
    near = [(5000.0 + 100.0, "w1")] * 3 + [(1000.0, "w1")] * 7
    assert SR.k1_resolution(near, sig, 5000.0)["verdict"] == "UNRESOLVED"
    far = [(9000.0, "w1")] * 3 + [(1000.0, "w1")] * 7
    assert SR.k1_resolution(far, sig, 5000.0)["verdict"] == "RESOLVED"


def test_rk_is_not_run_without_its_pair(tmp_path):
    make_floor(tmp_path, "mixtral-8x7b-tp4", "floor", "lock1710", ceil_law())
    assert SR.score(REPO, tmp_path)["RK"]["verdict"].startswith("NOT RUN")
