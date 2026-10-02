"""Rental 2's scorers (scripts/scoring/rental2/), written before any rental-2 page:
each is driven here on SYNTHETIC pages built in tmp_path, never on a published one,
and must return the verdict the registration's rule gives for the world planted.
Also the offline modules they read (scripts/l2_survival.py, scripts/floor_estimator.py),
the verbatim copy of rental 1's P6 function, and the registrations' own consistency."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "rental2"):
    sys.path.insert(0, str(p))

import common as CM  # noqa: E402
import dram_counter_route as DCR  # noqa: E402
import f4_implied as F4  # noqa: E402
import floor_estimator as FE  # noqa: E402
import l2_survival as L2S  # noqa: E402
import score_const as SC  # noqa: E402
import score_knobs as SK  # noqa: E402
import score_launch as SL  # noqa: E402
import score_w1floor as SW  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

KNOBS = CM.registration(REPO, "knobs")
LAUNCH = CM.registration(REPO, "launch2")
CONST = CM.registration(REPO, "const")
W1F = CM.registration(REPO, "w1floor")
DAY = "2026-10-02"


# --------------------------------------------------------------------------
# synthetic r3c byte pages
# --------------------------------------------------------------------------

def r3c_page(model, *, G=1, s=None, ctas=None, block_k=64, num_stages=4, v6="PASS",
             fmn=0.53, q_private=None, direct_ratio=1.0, private_scale=1.0):
    """A G page whose PRIVATE and SHARED per-GEMM bytes encode `s[g][n]` (and PRIVATE
    w2 `q_private[n]`), the recorded occupancy giving `ctas` CTAs per SM, the fabric
    sectors giving PRIVATE F / Mn = `fmn` and direct fabric hits `direct_ratio` x the
    subtraction's."""
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
                qp = (q_private or {}).get(g, {}).get(n, n) if arm == "private" else None
                bp = (qp if qp is not None else n) * W[g] * (private_scale if arm == "private" else 1)
                b_priv = n * W[g] + n * op[g]
                if arm == "private":
                    b = bp + n * op[g]
                else:
                    b = b_priv - (s[g].get(n, 0.0) * (n - 1) * W[g] if n >= 2 else 0.0)
                T, Hh = 1.0e6, 4.0e5
                Mn = T - Hh
                F = fmn * Mn
                far = 0.5 * F
                M = (F - far) + Mn
                per_gemm[g] = {"dram_bytes_read": b, "l2_tex_read_sectors": T,
                               "l2_tex_read_hit_sectors": Hh, "l2_read_miss_sectors": M,
                               "grid_size": 1}
                rec[g] = {"launch__occupancy_limit_blocks": 32,
                          "launch__occupancy_limit_registers": ctas[g],
                          "launch__occupancy_limit_shared_mem": 32,
                          "launch__occupancy_limit_warps": 8,
                          L2S.FABRIC: F, L2S.FABRIC_HIT: direct_ratio * far}
            cells.append({"arm": arm, "n": n, "per_gemm": per_gemm, "recorded": rec,
                          "per_call": {"dram_bytes_read": sum(per_gemm[g]["dram_bytes_read"]
                                                              for g in ("w1", "w2"))}})
    E = cfg.num_experts
    return {"family": "r3-arms", "run_id": f"synthetic-{model}-g{G}",
            "design": {"model": model, "dtype": "bf16", "block_m": 32, "block_n": 64,
                       "block_k": block_k, "num_warps": 8, "num_stages": num_stages,
                       "copies_declared": 9, "group_m": G,
                       "declared_by_arm": {"native": E, "shared": 9 * E, "private": 9 * E}},
            "card": {"name": "NVIDIA GH200 480GB", "slug": CM.CARD, "uuid": "GPU-synthetic",
                     "capability": "9.0", "sm_count": 132, "l2_bytes": 62914560},
            "cells": cells, "gates": [{"number": "V6", "kind": "VALIDITY", "verdict": v6}]}


def put_page(tree: Path, model: str, label: str, page: dict) -> Path:
    G = page["design"]["group_m"]
    d = tree / f"{DAY}-{CM.CARD}-{model}-{label}-r3-counters" / "lock1710"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"r3c-g{G}.json"
    p.write_text(json.dumps(page))
    return p


S_BASE = {"w1": {n: 0.05 for n in range(2, 10)}, "w2": {n: 0.55 for n in range(2, 10)}}


def knob_s(model, gemm, lam_key, wc_knob, s_base, *, wc_base=None):
    """s at a knob page under one hypothesis: 'L2r' (the central lambda), 'NL', 'ST'."""
    cfg = MODEL_CONFIGS[model]
    k, ncols = L2S.gemm_shape(cfg, gemm)
    P = ncols // 64
    x = (P - 1) * 64 * k * 2 / L2S.L2_REF_BYTES
    wc_base = wc_base or (660 if gemm == "w1" else 528)
    dd = P / wc_knob - P / wc_base
    out = {}
    for n in range(2, 10):
        lam = (KNOBS["law_L2r"]["lambda_by_n_SEEN"][str(n)]["lam"] if lam_key == "L2r" else
               KNOBS["rivals"]["lambda_steep_by_n"][str(n)])
        cap = CM.s_cap(x, KNOBS["law_L2r"]["s_cap_knots_by_n"][str(n)])
        out[n] = (s_base[gemm][n] if lam_key == "NL" else
                  CM.law_value(s_base[gemm][n], cap, dd, lam, x >= 0.92))
    return out


def tp2_tree(tmp_path, hyp, *, ctas_knob=None, v6="PASS", tp8_w2=None):
    tree = tmp_path / "tree"
    put_page(tree, "mixtral-8x7b-tp2", "l2base", r3c_page("mixtral-8x7b-tp2", s=S_BASE))
    ck = ctas_knob or {"w1": 2, "w2": 2}
    s = {g: knob_s("mixtral-8x7b-tp2", g, hyp, 132 * ck[g], S_BASE) for g in ("w1", "w2")}
    put_page(tree, "mixtral-8x7b-tp2", "l2s8",
             r3c_page("mixtral-8x7b-tp2", s=s, ctas=ck, num_stages=8, v6=v6))
    if tp8_w2 is not None:
        put_page(tree, "mixtral-8x7b-tp8", "l2s8",
                 r3c_page("mixtral-8x7b-tp8", s={"w1": {n: 1.0 for n in range(2, 10)},
                                                 "w2": {n: tp8_w2 for n in range(2, 10)}},
                          ctas={"w1": 2, "w2": 2}, num_stages=8))
    return tree


PRIMARY = "tp2 w2 s8 (PRIMARY)"


@pytest.mark.parametrize("hyp", ["L2r", "NL", "ST"])
def test_knobs_selects_each_hypothesis_on_the_primary_page(tmp_path, hyp):
    res = SK.score(REPO, tp2_tree(tmp_path, hyp))
    t = res["tests"][PRIMARY]
    assert t["verdict"] == hyp, t
    assert t["W_c"] == {"base": 528, "knob": 264}
    assert t["dd"] == pytest.approx(64 / 264 - 64 / 528)
    others = [h for h in ("L2r", "NL", "ST") if h != hyp]
    assert all(t["per_hypothesis"][h]["status"] == "FALSIFIED" for h in others)


def test_knobs_reprices_d_on_the_pages_own_window(tmp_path):
    """A knob page whose occupancy gives W_c 396, not the 264 a num_stages 8 build
    was expected to give: d is re-priced from the page and L2r is still selected."""
    res = SK.score(REPO, tp2_tree(tmp_path, "L2r", ctas_knob={"w1": 3, "w2": 3}))
    t = res["tests"][PRIMARY]
    assert t["W_c"]["knob"] == 396 and t["dd"] == pytest.approx(64 / 396 - 64 / 528)
    assert t["verdict"] == "L2r"


def test_a_v6_failing_page_is_flagged_and_reads_inconclusive(tmp_path):
    res = SK.score(REPO, tp2_tree(tmp_path, "L2r", v6="FAIL"))
    t = res["tests"][PRIMARY]
    assert t["verdict"] == "INCONCLUSIVE (FLAGGED V6)"
    assert any("V6" in d for d in t["demoted_by"])


def test_the_negative_control_demotes_every_selected_verdict(tmp_path):
    held = SK.score(REPO, tp2_tree(tmp_path / "a", "L2r", tp8_w2=1.0))
    assert held["controls"]["tp8 w2 s8"]["ok"] is True
    assert held["tests"][PRIMARY]["verdict"] == "L2r"
    failed = SK.score(REPO, tp2_tree(tmp_path / "b", "L2r", tp8_w2=0.80))
    t = failed["tests"][PRIMARY]
    assert failed["controls"]["tp8 w2 s8"]["ok"] is False
    assert t["verdict"] == "INCONCLUSIVE" and t["verdict_before_controls"] == "L2r"


def test_a_fabric_ratio_out_of_band_demotes_the_page(tmp_path):
    tree = tp2_tree(tmp_path, "L2r")
    s = {g: knob_s("mixtral-8x7b-tp2", g, "L2r", 264, S_BASE) for g in ("w1", "w2")}
    put_page(tree, "mixtral-8x7b-tp2", "l2s8", r3c_page("mixtral-8x7b-tp2", s=s,
             ctas={"w1": 2, "w2": 2}, num_stages=8, fmn=0.60))
    t = SK.score(REPO, tree)["tests"][PRIMARY]
    assert t["verdict"] == "INCONCLUSIVE" and any("F/Mn" in d for d in t["demoted_by"])


def test_the_partition_cross_check_agrees_and_disagrees(tmp_path):
    tree = tp2_tree(tmp_path, "L2r")
    res = SK.score(REPO, tree)
    assert res["partition_cross_check"]["mixtral-8x7b-tp2 l2base"]["verdict"] == "AGREE"
    put_page(tree, "mixtral-8x7b-tp2", "l2base",
             r3c_page("mixtral-8x7b-tp2", s=S_BASE, direct_ratio=1.2))
    res = SK.score(REPO, tree)
    assert res["partition_cross_check"]["mixtral-8x7b-tp2 l2base"]["verdict"].startswith("DISAGREE")


def t5_pages(ratio, f32=0.33, f64=0.5802):
    def q(f):
        return 9 + f * 63 * 9 / 128
    return {"BK64": r3c_page("mixtral-8x7b-tp2", G=64, q_private={"w2": {9: q(f64), 8: 8.5}}),
            "BK32": r3c_page("mixtral-8x7b-tp2", G=64, block_k=32, q_private={"w2": {9: q(f32), 8: 8.5}}),
            "BK128": r3c_page("mixtral-8x7b-tp2", G=64, block_k=128, num_stages=2,
                              q_private={"w2": {9: q(f32 * ratio), 8: 8.5}})}


@pytest.mark.parametrize("ratio, held", [(1.00, {"R0", "R1"}), (1.29, {"R2h"}), (1.67, {"R2"})])
def test_t5_selects_by_the_measured_rho(ratio, held):
    """Each candidate re-priced on the synthetic pages' own geometry (wave_split_bytes),
    which at the default W_c reproduces the registered centres."""
    res = SK.score_t5(KNOBS, t5_pages(ratio))
    assert res["rho_BK"] == pytest.approx(ratio, rel=1e-9)
    got = {k for k, v in res["candidates"].items() if v["verdict"] == "HELD"}
    assert got == held, res["candidates"]
    for k, v in res["candidates"].items():
        assert v["centre"] == pytest.approx(KNOBS["T5"]["registered_at_default_Wc"][k]["rho_BK"], abs=1e-4)


def test_t5_board_check_failure_makes_r2h_inconclusive():
    res = SK.score_t5(KNOBS, t5_pages(1.29, f64=0.50))
    assert not res["board_check"]["ok"]
    assert res["candidates"]["R2h"]["verdict"].startswith("INCONCLUSIVE")
    assert res["candidates"]["R1"]["verdict"] == "FALSIFIED"


def test_score_knobs_main_writes_its_files_on_a_partial_tree(tmp_path):
    tree = tp2_tree(tmp_path, "NL")
    out = tmp_path / "out"
    assert SK.main([str(REPO), str(tree), str(out)]) == 0
    res = json.loads((out / "knobs.score.json").read_text())
    assert res["tests"][PRIMARY]["verdict"] == "NL"
    assert res["tests"]["tp4 w1 s8 (PRIMARY, NL vs lag)"]["verdict"].startswith("NOT SCORED")
    assert res["T5"]["verdict"].startswith("NOT SCORED")
    assert "RENTAL 2 PART 1" in (out / "knobs.score.txt").read_text()


def test_tp4_s8_is_nl_against_lag_only(tmp_path):
    tree = tmp_path / "tree"
    base = {"w1": {n: 0.40 for n in range(2, 10)}, "w2": {n: 0.85 for n in range(2, 10)}}
    put_page(tree, "mixtral-8x7b-tp4", "l2base", r3c_page("mixtral-8x7b-tp4", s=base))
    for hyp in ("ST", "L2r"):
        s = {g: knob_s("mixtral-8x7b-tp4", g, hyp, 264, base) for g in ("w1", "w2")}
        put_page(tree, "mixtral-8x7b-tp4", "l2s8",
                 r3c_page("mixtral-8x7b-tp4", s=s, ctas={"w1": 2, "w2": 2}, num_stages=8))
        t = SK.score(REPO, tree)["tests"]["tp4 w1 s8 (PRIMARY, NL vs lag)"]
        assert set(t["per_hypothesis"]) == {"LAG", "NL"} and t["verdict"] == "LAG", (hyp, t)


# --------------------------------------------------------------------------
# l2_survival and floor_estimator
# --------------------------------------------------------------------------

def test_survival_recovers_the_planted_s_and_the_window():
    s = {"w1": {n: 0.1 + 0.01 * n for n in range(2, 10)}, "w2": {n: 0.5 for n in range(2, 10)}}
    out = L2S.survival(r3c_page("mixtral-8x7b-tp2", s=s, ctas={"w1": 5, "w2": 3}, fmn=0.52))
    assert out["w2"]["W_c"] == 396 and out["w1"]["W_c"] == 660
    assert out["w2"]["P"] == 64 and out["w2"]["d"] == pytest.approx(64 / 396)
    assert out["w1"]["x"] == pytest.approx((224 - 1) * 64 * 4096 * 2 / 62914560)
    for g in ("w1", "w2"):
        for n in range(2, 10):
            assert out[g]["s"][n] == pytest.approx(s[g][n], abs=1e-12)
            assert out[g]["fabric_over_mn"][n] == pytest.approx(0.52)
    assert out["w2"]["partition"]["private/5"] == pytest.approx(1.0)
    assert out["v6"] == "PASS"


def test_implied_slope_is_f4s_polyfit_and_its_weights_reproduce_it():
    cells = {n: (448 * n + 448, 22000.0 * n + 1234.0 + 50 * (n % 3)) for n in (6, 8, 10, 13)}
    xs = [FE.estimator_x(cells[n][0], 64) for n in cells]
    want = float(np.polyfit(xs, [cells[n][1] for n in cells], 1)[0])
    assert FE.implied_slope(cells, list(cells), 64) == pytest.approx(want)
    w = FE.slope_weights({n: c[0] for n, c in cells.items()}, list(cells), 64)
    assert sum(w[n] * cells[n][1] for n in cells) == pytest.approx(want)
    with pytest.raises(KeyError):
        FE.implied_slope(cells, [6, 7], 64)


def test_theta_delta_recovers_a_planted_line_and_its_weights_are_exact():
    sets = [{"slope": 352.2 * (1 + 0.01 + 0.7 * b), "asymptote": 352.2, "bias": b}
            for b in (-0.11, -0.004, 0.038)]
    td = FE.theta_delta(sets)
    assert td["theta"] == pytest.approx(0.7) and td["delta"] == pytest.approx(0.01)
    # the weights are the linear map from cells to theta and delta
    grids = {n: 448 * n + 448 for n in range(10, 17)}
    tsets = [{"treads": t, "asymptote": 352.2, "bias": b}
             for t, b in (((13, 14, 15), -0.11), ((11, 15, 16), -0.004), ((11, 12, 13), 0.038))]
    rng = np.random.default_rng(0)
    y = {n: float(rng.normal(1e5 * n, 3000)) for n in grids}
    slopes = [{"slope": FE.implied_slope({n: (grids[n], y[n]) for n in grids}, s["treads"], 64),
               "asymptote": 352.2, "bias": s["bias"]} for s in tsets]
    td = FE.theta_delta(slopes)
    wt, wd = FE.theta_delta_weights(tsets, grids, 64)
    b = np.array([s["bias"] for s in tsets])
    # theta is linear in y: theta(y) - theta(0) = sum w y; the constant part is mean-free here
    lin_t = sum(wt[n] * y[n] for n in wt)
    lin_d = sum(wd[n] * y[n] for n in wd)
    assert td["theta"] == pytest.approx(lin_t, rel=1e-9, abs=1e-9)
    assert td["delta"] == pytest.approx(lin_d - 1, rel=1e-9, abs=1e-9)
    assert FE.norm(wt) > 0 and b.size == 3


def test_intercept_and_its_sigma_and_form_rms():
    q = [10.0, 20.0, 30.0, 40.0]
    z, slope = FE.intercept(q, [5000 + 22000 * v for v in q])
    assert z == pytest.approx(5000) and slope == pytest.approx(22000)
    x = np.array(q)
    want = 1.0 * math.sqrt(1 / 4 + x.mean() ** 2 / ((x - x.mean()) ** 2).sum())
    assert FE.intercept_sigma(q, 1.0) == pytest.approx(want)
    assert FE.form_rms([10, 12, 14], [0, 2, 4]) == pytest.approx(0.0)
    assert FE.form_rms([0, 0], [1, -1]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# part 2: the launch floor, on a toy host/GPU model
# --------------------------------------------------------------------------

HF = 0.004          # the probe's h_flush, ms
H2C = "H2c BLOCK_K at G=1 (tail)"


@pytest.mark.parametrize("v6", ["PASS", "FAIL"])
def test_h2c_applies_the_registered_v6_rule(tmp_path, v6):
    """Post-page fix (2026-10-02): the H2c branch reads the page flags, so a V6-failing
    BLOCK_K page reads INCONCLUSIVE, as the registration's V6 control says."""
    tree = tmp_path / "tree"
    put_page(tree, "mixtral-8x7b-tp2", "l2base", r3c_page("mixtral-8x7b-tp2", s=S_BASE))
    put_page(tree, "mixtral-8x7b-tp2", "l2bk32",
             r3c_page("mixtral-8x7b-tp2", s=S_BASE, block_k=32, v6=v6))
    t = SK.score(REPO, tree)["tests"][H2C]
    for g in ("w1", "w2"):
        r = t[f"l2bk32 {g}"]
        if v6 == "FAIL":
            assert r["verdict"] == "INCONCLUSIVE (FLAGGED V6)", r
            assert r["verdict_before_controls"] == "H2c FALSIFIED (no k-step effect)"
            assert any("V6" in d for d in r["demoted_by"])
        else:
            assert r["verdict"] == "H2c FALSIFIED (no k-step effect)", r
            assert "demoted_by" not in r


PHI = 0.03          # host time not inside the interval, ms
CPB = 100


def toy_cells(model, H=0.35):
    creg = LAUNCH["C_reg_ms"][model]
    F = LAUNCH["F_ms"]
    rows, probe = [], []
    for arm in ("native", "shared", "private"):
        for n in range(1, 10):
            C = creg[arm][str(n)]
            for mode in ("E240", "E0", "E480", "GR"):
                Hm = H - HF if mode == "E0" else H
                if mode == "GR":
                    I, hb = C, False
                elif C + F[mode] < Hm:
                    I, hb = Hm - F[mode] - PHI, True
                else:
                    I, hb = C + 0.004, False
                for rep in range(3):
                    rows.append({"model": model, "mode": mode, "arm": arm, "tiles": n, "repeat": rep,
                                 "status": "ok", "ms_p50": I, "host_enqueue_ms": Hm * CPB,
                                 "calls_per_burst": CPB, "host_bound": hb})
                probe.append({"model": model, "when": "pre", "mode": mode, "arm": arm, "tiles": n,
                              "status": "ok", "H_pre_ms": Hm, "h_flush_ms": HF if mode != "E0" else 0.0})
    return rows, probe


def write_launch(tree, model, rows, probe, *, post=None, manifest=None, h_flush_trace=HF * 1e3):
    d = tree / f"{DAY}-{CM.CARD}-launch-floor-r2" / model
    (d / "traces").mkdir(parents=True, exist_ok=True)
    for name, rs in (("cells.csv", rows), ("hostprobe.csv", probe),
                     ("hostprobe-post.csv", post if post is not None else probe)):
        with open(d / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    (d / "manifest.json").write_text(json.dumps(manifest or {
        "phase": "timed", "profiler_enabled_ever": False, "profiler_checks": len(rows)}))
    parsed = {}
    for arm in ("native", "shared", "private"):
        for n in (1, 2):
            for mode, fl in (("E240", 67.6), ("E480", 135.1), ("E0", None)):
                parsed[f"TR-E-{mode}-{arm}-n{n}.json.gz"] = {"median": {
                    "flush_us": fl, "h_flush_us": h_flush_trace, "kernel_sum_us": 80.0,
                    "non_gemm_us": 11.0}, "launch_api_share": 0.11}
                parsed[f"TR-G-{mode}-{arm}-n{n}.json.gz"] = {"median": {
                    "flush_us": fl, "kernel_sum_us": 80.0, "non_gemm_us": 11.0}}
    (d / "traces" / "parsed.json").write_text(json.dumps(parsed))
    return d


def toy_tree(tmp_path, mutate=None, **kw):
    tree = tmp_path / "tree"
    for m in ("mixtral-8x7b-tp8", "granite-3.0-3b-a800m"):
        rows, probe = toy_cells(m)
        if mutate:
            rows, probe, kw2 = mutate(m, rows, probe)
        else:
            kw2 = {}
        write_launch(tree, m, rows, probe, **{**kw, **kw2})
    return tree


def test_the_toy_host_gpu_model_passes_p0_to_p5_and_p8(tmp_path):
    res = SL.score(REPO, toy_tree(tmp_path))
    for m in ("mixtral-8x7b-tp8", "granite-3.0-3b-a800m"):
        P = res["models"][m]["P"]
        for k in ("P0", "P1", "P2", "P4", "P5", "P8"):
            assert P[k]["verdict"] == "HELD", (m, k, P[k])
        assert P["P2"]["cells"] and P["P5"]["of"] > 0
    assert res["P3_pooled"]["cells"] >= 8 and res["P3_pooled"]["verdict"] == "HELD"
    assert res["models"]["jetmoe-8b"]["verdict"].startswith("NOT RUN")


def test_p1_is_not_scored_when_a_registered_tread_was_never_planned(tmp_path):
    """Post-page fix (2026-10-02): a unit planned at n = 1..4 cannot be read by P1's
    n <= 5 cell set and increment, so P1 reads NOT SCORED rather than counting the
    absent n = 5 cells as outside."""
    def four_treads(m, rows, probe):
        if m != "granite-3.0-3b-a800m":
            return rows, probe, {}
        return ([r for r in rows if r["tiles"] <= 4], [p for p in probe if p["tiles"] <= 4], {})
    P = SL.score(REPO, toy_tree(tmp_path, four_treads))["models"]
    p1 = P["granite-3.0-3b-a800m"]["P"]["P1"]
    assert p1["verdict"] == "NOT SCORED: registered treads [5] are not on the plan", p1
    assert p1["outside_of_measured"] == 0 and len(p1["cells"]) == 12
    assert P["mixtral-8x7b-tp8"]["P"]["P1"]["verdict"] == "HELD"


def test_a_host_step_after_half_the_cells_fails_p0(tmp_path):
    def step(m, rows, probe):
        half = len(rows) // 2
        for r in rows[half:]:
            r["host_enqueue_ms"] = (r["host_enqueue_ms"] / CPB + 0.08) * CPB
        post = [dict(p, H_pre_ms=p["H_pre_ms"] + 0.08) for p in probe]
        return rows, probe, {"post": post}
    res = SL.score(REPO, toy_tree(tmp_path, step))
    P = res["models"]["mixtral-8x7b-tp8"]["P"]
    assert P["P0"]["verdict"] == "FAILED"
    assert any("drift" in w for w in P["P0"]["why"]) and any("H_cell/H_pre" in w for w in P["P0"]["why"])
    for k in ("P2", "P4", "P5", "P8"):
        assert P[k]["verdict"].endswith("(host drift)")


def test_a_profiler_flag_fails_p0(tmp_path):
    tree = toy_tree(tmp_path, manifest={"phase": "timed", "profiler_enabled_ever": True,
                                        "profiler_checks": 10})
    P0 = SL.score(REPO, tree)["models"]["mixtral-8x7b-tp8"]["P"]["P0"]
    assert P0["verdict"] == "FAILED" and any("profiler" in w for w in P0["why"])


def test_an_edge_cell_is_excluded(tmp_path):
    base = SL.score(REPO, toy_tree(tmp_path / "a"))["models"]["mixtral-8x7b-tp8"]["P"]

    def edge(m, rows, probe):
        if m == "mixtral-8x7b-tp8":
            for r in rows:
                if (r["mode"], r["arm"], r["tiles"], r["repeat"]) == ("E240", "native", 7, 0):
                    r["host_bound"] = True        # 1 of 3 repeats: fraction 0.33, an edge cell
        return rows, probe, {}
    got = SL.score(REPO, toy_tree(tmp_path / "b", edge))["models"]["mixtral-8x7b-tp8"]["P"]
    assert got["P5"]["of"] == base["P5"]["of"] - 1
    assert ("native", 7) not in [(c["arm"], c["n"]) for c in got["P2"]["cells"]]
    assert ("native", 7) in [(c["arm"], c["n"]) for c in base["P2"]["cells"]]


def test_p3s_band_uses_the_probes_h_flush_not_the_traces(tmp_path):
    """The traces' host flush (inflated by the profiler) is 30 us; the probe's is 4 us,
    the toy's truth. P3 holds on the probe's, and its rows say which it used."""
    res = SL.score(REPO, toy_tree(tmp_path / "a", h_flush_trace=30.0))
    assert res["P3_pooled"]["verdict"] == "HELD"
    rows = res["models"]["mixtral-8x7b-tp8"]["P"]["P3"]["cells"]
    assert rows and all(r["h_flush_probe_ms"] == pytest.approx(HF) for r in rows)

    def wrong_probe(m, rows, probe):
        return rows, [dict(p, h_flush_ms=0.030) for p in probe], {}
    assert SL.score(REPO, toy_tree(tmp_path / "b", wrong_probe))["P3_pooled"]["verdict"] == "FALSIFIED"


def test_launch_mp_is_rental1s_p6_function_verbatim():
    r1 = (REPO / "scripts/scoring/rental1/score_launch.py").read_text()
    r2 = (REPO / "scripts/scoring/rental2/launch_mp.py").read_text()
    body = r1[r1.index("def timeline(path):"):r1.index('p("\\n== P6')].rstrip()
    assert body in r2
    assert hashlib.sha256(r2[r2.index("def timeline(path):"):].rstrip().encode()).hexdigest() == \
        hashlib.sha256(body.encode()).hexdigest()


# --------------------------------------------------------------------------
# part 3: the per-GEMM constant, on synthetic floor captures
# --------------------------------------------------------------------------

def floor_file(model, *, Zu=0.4, L=1000.0, D_u=0.0, mhz=1690.0, z_scale=1.0, L0=1500.0):
    cells = {}
    for c in CONST["cells_registered"][model]:
        q, u, ce = c["q"], c["u"], c["ceil"]
        ac = q * u + Zu * u * z_scale
        amax = ac + (ce - q) * u + D_u * u
        el = amax + L
        cell = cells.setdefault(c["n"], {"arm": "native", "n": c["n"], "grid": {}, "per_gemm": {},
                                          "null": {"sm__cycles_elapsed.avg": 1000 + L0}})
        cell["grid"][c["gemm"]] = c["grid"]
        cell["per_gemm"][c["gemm"]] = {
            "sm__cycles_active.avg": ac, "sm__cycles_active.max": amax,
            "sm__cycles_elapsed.avg": el, "sm_clock_mhz": mhz,
            "gpu__time_duration.sum": el / mhz * 1e3, "launch__grid_size": c["grid"],
            "sm__ctas_launched.max": math.ceil(c["grid"] / 132) + 1,
            "sm__ctas_active.sum": 3.0 * 132 * ac}
    return {"plan": {"model": model, "group_m": 64}, "cells": list(cells.values())}


def put_floor(tree, model, label, stem, page):
    d = tree / f"{DAY}-{CM.CARD}-{model}-{label}-r3-counters"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.json").write_text(json.dumps(page))


def const_tree(tmp_path, *, ns_form=True, **kw):
    tree = tmp_path / "tree"
    for m, lab in (("mixtral-8x7b-tp8", "floor2"), ("mixtral-8x7b-tp4", "floor"), ("mixtral-8x7b-tp2", "floor")):
        put_floor(tree, m, lab, "r3f-g64", floor_file(m, mhz=1360.0, **kw))
        put_floor(tree, m, lab, "r3f-g64-lock1710", floor_file(m, mhz=1690.0, **kw))
    put_floor(tree, "mixtral-8x7b-tp4", "floor1005", "r3f-g64-lock1005",
              floor_file("mixtral-8x7b-tp4", mhz=1005.0, z_scale=(1005 / 1690 if ns_form else 1.0), **kw))
    return tree


def test_const_identifies_the_transient(tmp_path):
    res = SC.score(REPO, const_tree(tmp_path))
    K = res["K"]
    assert K["K1"]["verdict"] == "HOLDS" and K["K2"]["verdict"] == "HOLDS"
    assert K["K2"]["pooled"] == pytest.approx(0.4, abs=1e-6)
    assert K["K4"]["verdict"] == "HOLDS"
    assert K["K3"]["pooled_r"] == pytest.approx(1005 / 1690, abs=1e-6)
    assert K["K3"]["verdict"] == "ns form HOLDS, cycle form FALSIFIED"
    assert res["identification"] == "H_ZT"
    tp4 = res["captures"]["mixtral-8x7b-tp4 lock1710"]["w2"]
    assert tp4["Z_f_over_u"] == pytest.approx(0.4, abs=1e-6)
    assert all(r["L0"] == pytest.approx(1500.0) for r in tp4["rows"])


def test_const_identifies_launch_drain(tmp_path):
    res = SC.score(REPO, const_tree(tmp_path, L=20000.0))
    assert res["K"]["K1"]["verdict"] == "FALSIFIED" and res["identification"] == "H_LD"


def test_const_identifies_imbalance(tmp_path):
    res = SC.score(REPO, const_tree(tmp_path, D_u=0.5))
    assert res["K"]["K1"]["verdict"] == "HOLDS"
    assert res["K"]["K4"]["verdict"].startswith("FALSIFIED") and res["identification"] == "H_IMB"


def test_k3_reads_about_one_for_a_constant_in_cycles(tmp_path):
    res = SC.score(REPO, const_tree(tmp_path, ns_form=False))
    assert res["K"]["K3"]["pooled_r"] == pytest.approx(1.0, abs=1e-6)
    assert res["K"]["K3"]["verdict"] == "cycle form HOLDS, ns form FALSIFIED"
    assert res["identification"] == "INCONCLUSIVE"


def test_const_without_shape_metrics_scores_nothing_it_cannot(tmp_path):
    tree = tmp_path / "tree"
    page = floor_file("mixtral-8x7b-tp4")
    for c in page["cells"]:
        for pg in c["per_gemm"].values():
            pg.pop("sm__cycles_active.max")
    put_floor(tree, "mixtral-8x7b-tp4", "floor", "r3f-g64-lock1710", page)
    res = SC.score(REPO, tree)
    assert res["K"]["K1"]["verdict"].startswith("NOT SCORED")
    assert res["K"]["K3"]["verdict"].startswith("NOT SCORED")


# --------------------------------------------------------------------------
# part 4: tp8's w1 floor, on synthetic base captures
# --------------------------------------------------------------------------

def w1_file(model, form, a=5000.0):
    cells = {}
    for c in CONST["cells_registered"][model]:
        g = c["gemm"]
        S = W1F["geometry"][model][g]["ksteps"]
        asym = W1F["geometry"][model][g]["asymptote"]
        y = {"H_EST": c["cores"] + a, "H_NPN": 0.964 * c["cores"] + a,
             "FLUID": FE.estimator_x(c["grid"], S) * asym + a}[form]
        cell = cells.setdefault(c["n"], {"arm": "native", "n": c["n"], "grid": {}, "per_gemm": {}})
        cell["grid"][g] = c["grid"]
        cell["per_gemm"][g] = {"sm__cycles_elapsed.avg": y, "launch__grid_size": c["grid"]}
    return {"plan": {"model": model, "group_m": 64}, "cells": list(cells.values())}


def w1_tree(tmp_path, form):
    tree = tmp_path / "tree"
    for m, lab in (("mixtral-8x7b-tp8", "floor2"), ("mixtral-8x7b-tp4", "floor"), ("mixtral-8x7b-tp2", "floor")):
        put_floor(tree, m, lab, "r3f-g64", w1_file(m, form))
    return tree


def test_w1floor_selects_h_est_on_the_models_own_cells(tmp_path):
    res = SW.score(REPO, w1_tree(tmp_path, "H_EST"))
    assert res["H_EST_recomputed"] == "agree"
    fam = res["base (PRIMARY)"]["families"]
    assert fam["H_EST"] == "HOLDS" and fam["FLUID"] == "FALSIFIED" and fam["FLUID_LOW"] == "FALSIFIED"
    assert fam["verdict"] == "H_EST" and fam["co_primary"] == "H_EST supported"
    g = res["base (PRIMARY)"]["gemms"]["mixtral-8x7b-tp8 w1"]
    assert g["theta"] == pytest.approx(1.0, abs=1e-6) and g["delta"] == pytest.approx(0.0, abs=1e-6)
    assert res["base (PRIMARY)"]["gemms"]["mixtral-8x7b-tp4 w2"]["w2_offset"]["H_EST"] == "HOLDS"
    assert res["lock1710 (printed)"]["families"]["verdict"].startswith("NOT SCORED")


def test_w1floor_selects_fluid_on_a_set_independent_slope(tmp_path):
    fam = SW.score(REPO, w1_tree(tmp_path, "FLUID"))["base (PRIMARY)"]
    assert fam["families"]["verdict"] == "FLUID", fam["families"]
    assert fam["families"]["co_primary"] == "FLUID supported"
    assert fam["gemms"]["mixtral-8x7b-tp8 w1"]["theta"] == pytest.approx(0.0, abs=1e-6)
    assert fam["gemms"]["mixtral-8x7b-tp2 w2"]["w2_offset"]["FLUID"] == "HOLDS"


def test_w1floor_reads_the_npn_deficit_as_h_npn_with_every_main_family_failing(tmp_path):
    fam = SW.score(REPO, w1_tree(tmp_path, "H_NPN"))["base (PRIMARY)"]["families"]
    assert fam["H_NPN (secondary)"] == "HOLDS"
    assert fam["H_EST"] == "FALSIFIED" and fam["verdict"] == "NEITHER"


def test_w1floor_refuses_a_grid_that_is_not_the_registered_one(tmp_path):
    tree = w1_tree(tmp_path, "H_EST")
    page = w1_file("mixtral-8x7b-tp8", "H_EST")
    page["cells"][0]["grid"]["w1"] += 1
    put_floor(tree, "mixtral-8x7b-tp8", "floor2", "r3f-g64", page)
    with pytest.raises(ValueError, match="registered"):
        SW.score(REPO, tree)


# --------------------------------------------------------------------------
# the registrations' own consistency and the F4 diagnosis's capture match
# --------------------------------------------------------------------------

def test_the_registrations_are_internally_consistent():
    for k, v in KNOBS["T5"]["registered_at_default_Wc"].items():
        assert v["band_at_default_Wc"][0] == pytest.approx(v["rho_BK"] - v["half_width"], abs=2e-6)
        assert v["half_width"] == pytest.approx(2 * v["sigma_rho"], abs=2e-6)
    for n, v in KNOBS["law_L2r"]["lambda_by_n_SEEN"].items():
        assert v["lam"] > 0 and v["sigma"] > 0
    assert SW.check_registration(W1F, CONST["cells_registered"]) == []
    for model, cells in CONST["cells_registered"].items():
        for c in cells:
            assert c["floor_bound"] == (c["floor_share_sigma1"] >= 0.97 and c["waves"] >= 2)
    # every scored tread set is unseen and >= 4 waves
    for key, s in W1F["sets"].items():
        model = key.split(" ")[0]
        assert all(n >= W1F["unseen_min_tread"][model] for v in s["sets"].values() for n in v["treads"])
    assert LAUNCH["F_ms"]["E240"] == pytest.approx(0.0676, abs=1e-4)
    assert set(LAUNCH["C_reg_ms"]["mixtral-8x7b-tp8"]) == {"native", "shared", "private"}


def test_f4_implied_reads_exactly_one_base_capture(tmp_path):
    sess = tmp_path / "sess"
    page = w1_file("mixtral-8x7b-tp8", "H_EST")
    put_floor(sess, "mixtral-8x7b-tp8", "floor", "r3f-g64", page)
    got = F4.measured_cells(sess, "mixtral-8x7b-tp8")
    assert set(n for n, _g in got) == {c["n"] for c in page["cells"]}
    put_floor(sess, "mixtral-8x7b-tp8", "floorb", "r3f-g64", page)
    assert F4.measured_cells(sess, "mixtral-8x7b-tp8") == {}


def test_no_page_of_another_block_k_reaches_the_timing_model(tmp_path):
    """The review, section 1: the timing model refuses a BK != 64 page, so no gate or
    scorer can feed one to TM.build; T5 prices such pages with wave_split_bytes only."""
    import r3_timing_model as TM
    p = put_page(tmp_path, "mixtral-8x7b-tp2", "atbk32",
                 r3c_page("mixtral-8x7b-tp2", G=64, block_k=32))
    old = TM.set_model("mixtral-8x7b-tp2")
    try:
        with pytest.raises(TM.Refused, match="constants are"):
            TM.load_counter_pages(p.parent)
    finally:
        TM.set_model(old)


# --------------------------------------------------------------------------
# the gate addendum (docs/registered/2026-10-02-rental2-addendum-gates.json),
# on synthetic pages only
# --------------------------------------------------------------------------

ADD = CM.addendum(REPO)
LOCKED = (("mixtral-8x7b-tp8", "floor2"), ("mixtral-8x7b-tp4", "floor"), ("mixtral-8x7b-tp2", "floor"))


def gated(page, *, lock=None, **verdicts):
    """`page` carrying the named VALIDITY gates (FL1="FAIL", V1="PASS", ...) and,
    for a lock capture, its clock block."""
    page["gates"] = [{"number": k, "kind": "VALIDITY", "verdict": v} for k, v in verdicts.items()]
    if lock is not None:
        page["clock"] = {"control": "none", "lock_mhz": float(lock), "band_mhz": 15.0}
    return page


def cell_of(page, n, g):
    return next(c for c in page["cells"] if c["n"] == n)["per_gemm"][g]


def with_null_clock(page, mhz):
    """`page` whose every cell's null kernel reads `mhz` (None: no null clock)."""
    if mhz is not None:
        for c in page["cells"]:
            c["null"]["sm_clock_mhz"] = float(mhz)
    return page


def gated_const_tree(tmp_path, *, fl1=("PASS", "PASS", "PASS"), v1_1005="PASS", zu_tp2=0.4,
                     mutate=None, null=(None, None, None), null_1005=None):
    """const_tree's world, each 1710 capture carrying FL1 and V1 as given and its
    null kernel's clock `null` (None: a capture without one)."""
    tree = tmp_path / "tree"
    for (m, lab), f, nk in zip(LOCKED, fl1, null, strict=True):
        put_floor(tree, m, lab, "r3f-g64", gated(floor_file(m, mhz=1360.0), V1="PASS"))
        page = gated(floor_file(m, mhz=1690.0, Zu=zu_tp2 if m.endswith("tp2") else 0.4),
                     lock=1710, FL1=f, V1="PASS")
        with_null_clock(page, nk)
        if mutate:
            mutate(m, page)
        put_floor(tree, m, lab, "r3f-g64-lock1710", page)
    put_floor(tree, "mixtral-8x7b-tp4", "floor1005", "r3f-g64-lock1005",
              with_null_clock(gated(floor_file("mixtral-8x7b-tp4", mhz=1005.0, z_scale=1005 / 1690),
                                    lock=1005, FL1="PASS", V1=v1_1005), null_1005))
    return tree


def test_the_addendum_leaves_the_four_registrations_byte_identical():
    for stem, h in ADD["amends"].items():
        for ext, digest in h.items():
            path = REPO / "docs" / "registered" / f"{stem}.{ext}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path.name
    assert set(ADD["amends"]) == {CM.NAMES[p] for p in ("knobs", "launch2", "const", "w1floor")}
    assert ADD["constants"] == {"step_mhz": 15.0, "unusable_gates": ["V1"], "lock_check_tol": 0.03,
                                "lock_gate_set_aside": "FL1"}
    assert ADD["per_part"]["knobs"]["gates_not_gating"] == ["V6", "V10"]
    txt = (REPO / "docs" / "registered" / f"{CM.ADDENDUM}.txt").read_text()
    assert "before any rental-2 score" in txt and "\u2014" not in txt


def test_combine_verdict_agreement_disagreement_and_clean_empty():
    assert CM.combine_verdict("HOLDS", "HOLDS") == "HOLDS"
    assert CM.combine_verdict("FALSIFIED", "HOLDS") == "INCONCLUSIVE (ALL FALSIFIED; CLEAN HOLDS)"
    v = CM.combine_verdict("HOLDS", "NOT SCORED: no L readable")
    assert v.startswith("NOT SCORED") and v.endswith("reading only gate-failed pages: HOLDS")
    assert CM.combine_verdict("NOT SCORED: x", None) == "NOT SCORED: x"
    assert CM.failed_gates({"gates": [{"number": "V7", "kind": "VALIDITY", "verdict": "REFUSE"},
                                      {"number": "C6", "kind": "CLAIM", "verdict": "FAIL"},
                                      {"number": "V1", "kind": "VALIDITY", "verdict": "PASS"}]}) == ["V7"]


def test_a_cell_above_its_lock_by_more_than_one_step_is_dropped_and_recorded(tmp_path):
    def mutate(m, page):
        if m == "mixtral-8x7b-tp4":
            cell_of(page, 1, "w2")["sm_clock_mhz"] = 1785.0     # rental 2's tp4 n=1 w2
            cell_of(page, 4, "w2")["sm_clock_mhz"] = 1725.5     # just over one step
            cell_of(page, 6, "w2")["sm_clock_mhz"] = 1725.0     # exactly one step: kept
    res = SC.score(REPO, gated_const_tree(tmp_path, mutate=mutate))
    pages = res["addendum"]["pages"]
    tp4 = next(v for k, v in pages.items() if "tp4-floor-r3" in k and k.endswith("lock1710.json"))
    assert tp4["failed_gates"] == [] and tp4["ALL"] == tp4["CLEAN"] == "counted"
    assert [d.split(" MHz")[0] for d in tp4["dropped_above_lock"]] == ["n=1 w2 1785.0", "n=4 w2 1725.5"]
    rows = res["captures"]["mixtral-8x7b-tp4 lock1710"]["w2"]["rows"]
    assert 4 not in {r["n"] for r in rows} and 6 in {r["n"] for r in rows}
    k5 = res["K"]["K5"]["cells"]
    assert "mixtral-8x7b-tp4 lock1710 n=1 w2" not in k5 and "mixtral-8x7b-tp4 lock1710 n=1 w1" in k5
    # a base capture carries no lock and drops nothing
    base = next(v for k, v in pages.items() if "tp4-floor-r3" in k and k.endswith("r3f-g64.json"))
    assert base["dropped_above_lock"] == []
    assert any("dropped n=1 w2 1785.0 MHz" in ln for ln in SC.lines(res))


def test_all_and_clean_agree_and_the_common_verdict_is_registered(tmp_path):
    """tp2's 1710 capture fails FL1 and reads like the others: K1, K2 and K4 agree."""
    res = SC.score(REPO, gated_const_tree(tmp_path, fl1=("PASS", "PASS", "FAIL")))
    assert res["K"]["K1"]["verdict"] == "HOLDS" and res["K"]["K2"]["verdict"] == "HOLDS"
    assert res["identification"] == "H_ZT"
    clean = res["addendum"]["CLEAN"]
    assert clean["captures"]["mixtral-8x7b-tp2 lock1710"] == {"verdict": "missing"}
    assert res["K"]["K2"]["values"] == [pytest.approx(0.4), pytest.approx(0.4)]   # ALL's numbers
    assert clean["K"]["K2"]["values"] == [pytest.approx(0.4)]


def test_all_and_clean_disagree_and_the_verdict_is_inconclusive(tmp_path):
    """tp2's FL1-failed capture alone carries a w2 Z / u outside K2's band: the
    median of the two (ALL) falls outside, tp4 alone (CLEAN) inside."""
    res = SC.score(REPO, gated_const_tree(tmp_path, fl1=("PASS", "PASS", "FAIL"), zu_tp2=1.2))
    k2 = res["K"]["K2"]["verdict"]
    assert k2 == "INCONCLUSIVE (ALL FALSIFIED; CLEAN HOLDS)"
    row = next(r for r in res["addendum"]["verdicts"] if r["path"] == "K / K2 / verdict")
    assert (row["ALL"], row["CLEAN"]) == ("FALSIFIED", "HOLDS")
    assert any("K / K2 / verdict: ALL FALSIFIED | CLEAN HOLDS" in ln for ln in SC.lines(res))


def test_clean_empty_reads_not_scored_with_alls_verdict_labelled(tmp_path):
    """Rental 2's known outcome: every 1710 capture fails FL1, so CLEAN holds none."""
    res = SC.score(REPO, gated_const_tree(tmp_path, fl1=("FAIL", "FAIL", "FAIL")))
    for k in ("K1", "K2", "K3", "K4"):
        v = res["K"][k]["verdict"]
        assert v.startswith("NOT SCORED (CLEAN has no data); ALL, reading only gate-failed pages: "), (k, v)
    assert res["K"]["K1"]["verdict"].endswith(": HOLDS")
    assert res["K"]["K3"]["verdict"].endswith("ns form HOLDS, cycle form FALSIFIED")
    assert res["identification"].endswith("gate-failed pages: H_ZT")
    assert res["K"]["K5"]["verdict"] == "RECORD"
    # no null kernel on these captures: CLEAN fell back to FL1
    p = next(v for k, v in res["addendum"]["pages"].items() if "tp8-floor2" in k and "lock1710" in k)
    assert p["lock_check"]["ok"] is None and "falls back to FL1" in p["lock_check"]["why"]
    assert p["clean_failed"] == ["FL1"] and p["CLEAN"] == "excluded"


def test_a_held_lock_by_the_null_kernel_keeps_an_fl1_failed_capture_in_clean(tmp_path):
    """Rental 2's medians: 1684 / 1687 / 1688 MHz at 1710 and 991 at 1005 (-1.4%) pass the
    3% check, so CLEAN keeps every capture FL1 failed and the views agree."""
    res = SC.score(REPO, gated_const_tree(tmp_path, fl1=("FAIL", "FAIL", "FAIL"),
                                          null=(1684.0, 1687.0, 1688.0), null_1005=991.0))
    for k in ("K1", "K2", "K4"):
        assert res["K"][k]["verdict"] == "HOLDS", (k, res["K"][k]["verdict"])
    assert res["K"]["K3"]["verdict"] == "ns form HOLDS, cycle form FALSIFIED"
    assert res["identification"] == "H_ZT"
    p = next(v for k, v in res["addendum"]["pages"].items() if "tp8-floor2" in k and "lock1710" in k)
    assert p["failed_gates"] == ["FL1"] and p["clean_failed"] == [] and p["CLEAN"] == "counted"
    assert p["lock_check"]["ok"] is True and p["lock_check"]["null_median_mhz"] == pytest.approx(1684.0)
    assert any("lock check: null kernel median 1684.0 MHz" in ln for ln in SC.lines(res))


def test_a_lock_not_in_force_by_the_null_kernel_is_out_of_clean(tmp_path):
    """tp2's null kernel reads +5.6% (the 8x22B floor's lock not in force): out of CLEAN
    even though its FL1 passed; its own w2 Z / u then splits K2."""
    res = SC.score(REPO, gated_const_tree(tmp_path, fl1=("FAIL", "FAIL", "PASS"), zu_tp2=1.2,
                                          null=(1684.0, 1687.0, 1710.0 * 1.056)))
    p = next(v for k, v in res["addendum"]["pages"].items() if "tp2-floor" in k and "lock1710" in k)
    assert p["failed_gates"] == [] and p["lock_check"]["ok"] is False
    assert p["clean_failed"] == ["lock not in force (null kernel)"]
    assert (p["ALL"], p["CLEAN"]) == ("counted", "excluded")
    assert res["K"]["K2"]["verdict"] == "INCONCLUSIVE (ALL FALSIFIED; CLEAN HOLDS)"
    def at(mhz):
        return CM.lock_check({"clock": {"lock_mhz": 1710.0}, "cells": [{"null": {"sm_clock_mhz": mhz}}]},
                             0.03)["ok"]
    assert at(1710 * 0.971) is True and at(1710 * 0.969) is False and at(1710 * 1.029) is True


def test_a_v1_failure_makes_the_page_unusable_in_both_views(tmp_path):
    res = SC.score(REPO, gated_const_tree(tmp_path, v1_1005="FAIL"))
    p = next(v for k, v in res["addendum"]["pages"].items() if "floor1005" in k)
    assert p["failed_gates"] == ["V1"] and p["ALL"].startswith("unusable") and p["CLEAN"].startswith("unusable")
    assert res["K"]["K3"]["verdict"] == "NOT SCORED: a tp4 capture is missing"
    assert res["K"]["K1"]["verdict"] == "HOLDS"


def test_w1floor_primary_agrees_while_its_lock_view_reads_not_scored(tmp_path):
    tree = w1_tree(tmp_path, "H_EST")
    for m, lab in LOCKED:
        put_floor(tree, m, lab, "r3f-g64-lock1710",
                  gated(w1_file(m, "H_EST"), lock=1710, FL1="FAIL", V1="PASS"))
    res = SW.score(REPO, tree)
    assert res["base (PRIMARY)"]["families"]["verdict"] == "H_EST"
    lock = res["lock1710 (printed)"]["families"]
    assert lock["verdict"] == "NOT SCORED (CLEAN has no data); ALL, reading only gate-failed pages: H_EST"
    assert lock["co_primary"].endswith("gate-failed pages: H_EST supported")


def test_w1floor_v1_failure_on_the_primary_scores_nothing_from_that_page(tmp_path):
    tree = w1_tree(tmp_path, "H_EST")
    put_floor(tree, "mixtral-8x7b-tp8", "floor2", "r3f-g64",
              gated(w1_file("mixtral-8x7b-tp8", "H_EST"), V1="FAIL"))
    res = SW.score(REPO, tree)
    assert res["base (PRIMARY)"]["families"]["verdict"] == "NOT SCORED: a w1 GEMM lacks its three sets"
    assert res["base (PRIMARY)"]["gemms"]["mixtral-8x7b-tp8 w1"] == {"verdict": "NOT SCORED: capture missing"}


@pytest.mark.parametrize("gate, want", [("V10", "L2r"), ("V7", "NOT SCORED"), ("V1", "NOT SCORED: a page")])
def test_knobs_gates_v10_not_at_all_others_by_the_two_views(tmp_path, gate, want):
    """V10 gates no part-1 score (as registered); a V7 failure goes through ALL and
    CLEAN; a V1 failure makes the page unusable in both."""
    tree = tp2_tree(tmp_path, "L2r")
    s = {g: knob_s("mixtral-8x7b-tp2", g, "L2r", 264, S_BASE) for g in ("w1", "w2")}
    page = r3c_page("mixtral-8x7b-tp2", s=s, ctas={"w1": 2, "w2": 2}, num_stages=8)
    page["gates"].append({"number": gate, "kind": "VALIDITY", "verdict": "FAIL"})
    put_page(tree, "mixtral-8x7b-tp2", "l2s8", page)
    v = SK.score(REPO, tree)["tests"][PRIMARY]["verdict"]
    assert v.startswith(want), v
    if gate == "V7":
        assert v.endswith("reading only gate-failed pages: L2r")


def test_score_mains_print_the_addendum(tmp_path):
    tree = gated_const_tree(tmp_path, fl1=("FAIL", "FAIL", "FAIL"))
    out = tmp_path / "out"
    assert SC.main([str(REPO), str(tree), str(out)]) == 0
    res = json.loads((out / "const.score.json").read_text())
    assert res["addendum"]["registration"] == CM.ADDENDUM
    assert f"addendum {CM.ADDENDUM}" in (out / "const.score.txt").read_text()
