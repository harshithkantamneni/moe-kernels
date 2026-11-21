"""Rental 5's skew and c15 scorers on SYNTHETIC pages only (no rental-5 page exists):
scripts/scoring/rental5/score_skew.py and score_c15.py against
docs/registered/2026-10-07-rental5-{skew,c15}-gh200.json, and r5common's rules (the G3
TOST with a cluster-t interval and Cochran's Q, the log calibration line, gamma)."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
R5 = REPO / "scripts" / "scoring" / "rental5"
for p in (REPO, REPO / "scripts", R5):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r5common as C5  # noqa: E402
import reg_skew as RS  # noqa: E402
import score_c15 as SC15  # noqa: E402
import score_skew as SK  # noqa: E402

CARD = C5.CARD
REG = C5.registration(REPO, "skew")
REG15 = C5.registration(REPO, "c15")


# --------------------------------------------------------------------------
# r5common's rules
# --------------------------------------------------------------------------

def test_the_t_and_chi2_tails_need_no_scipy():
    assert C5.t_ppf(0.95, 5) == pytest.approx(2.015048, abs=1e-5)
    assert C5.t_ppf(0.975, 10) == pytest.approx(2.228139, abs=1e-5)
    assert C5.chi2_sf(3.841459, 1) == pytest.approx(0.05, abs=1e-5)
    assert C5.chi2_sf(35.0, 35) == pytest.approx(0.4682, abs=1e-3)


def test_g3_passes_noise_fails_a_pooled_shift_and_sends_an_n_local_effect_to_its_stratum():
    rng = random.Random(7)
    noise = [{"cluster": p, "stratum": n % 3, "d": rng.gauss(0, 0.0025)} for p in range(6) for n in range(6)]
    assert C5.g3_rule(noise)["verdict"] == "PASS"
    shifted = [dict(r, d=r["d"] + 0.006) for r in noise]
    out = C5.g3_rule(shifted)
    assert out["verdict"].startswith("FAIL") and out["void_strata"] == "ALL"
    local = [dict(r, d=r["d"] + (0.012 if r["stratum"] == 2 else 0.0)) for r in noise]
    out = C5.g3_rule(local)
    assert out["Q"]["heterogeneous"] and out["void_strata"] == ["2"], out
    assert C5.g3_rule([])["verdict"].startswith("NOT SCORED")


def test_the_calibration_line_is_descriptive_under_six_clusters_and_gamma_drops_prediction_ties():
    rows = [{"pred": 1 + 0.1 * i, "meas": (1 + 0.1 * i) * 1.001, "cluster": i % 3, "stratum": 0}
            for i in range(12)]
    cal = C5.calibration(rows)
    assert cal["verdict"].startswith("NOT RESOLVED (3 clusters") and cal["slope"] == pytest.approx(1.0, abs=1e-6)
    tied = [{"pred": 1.0, "meas": 1.0 + 0.01 * i, "cluster": i, "stratum": 0} for i in range(8)]
    g = C5.gamma(tied)
    assert g["concordant"] == g["discordant"] == 0 and g["verdict"].startswith("NOT RESOLVED")
    rng = random.Random(5)
    six = [{"pred": 1 + 0.1 * i, "meas": (1 + 0.1 * i) * (1 + rng.gauss(0, 0.002)), "cluster": i % 6}
           for i in range(24)]
    assert C5.calibration(six, boot=300)["verdict"] == "CONSISTENT"
    off = [dict(r, meas=r["pred"] * 1.05) for r in six]   # a 5% level miss with no spread
    assert C5.calibration(off, boot=300)["verdict"] == "NOT CONSISTENT"


# --------------------------------------------------------------------------
# synthetic pages
# --------------------------------------------------------------------------

def _hist_doc(name):
    return json.loads((REPO / REG["histograms"]["files"][name]["path"]).read_text())


def _write_page(tree: Path, model: str, label: str, file: str, rows: list, copies: int = 9, g1="PASS"):
    d = tree / "results" / f"gaps-{CARD}-{model}-{label}" / "private_weight_reference" / f"run-{label}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps({
        "experiment": "private_weight_reference", "kind": "histogram-page", "session_tag": f"t-{label}",
        "pinned": {"GROUP_SIZE_M": 8}, "copies_declared": copies,
        "histogram": {"file_sha256": REG["histograms"]["files"][file]["sha256"],
                      "shuffle_seed": REG["histograms"]["files"][file]["shuffle_seed"]},
        "histogram_gates": {"G1_lock_thermal": {"verdict": g1}}, "gates": [], "treads_table": rows}))


def _rows(model, page, ms_of, copies=9, file=None):
    doc = _hist_doc(file or f"{model}-{page}.json")
    E = len(next(c["counts"] for c in doc["cells"] if c["counts"]))
    out = []
    for c in doc["cells"]:
        for arm in c.get("arms") or ("native", "shared"):
            ms = ms_of(model, page, c["n"], c["label"], arm)
            if ms is None:
                continue
            out.append({"arm": arm, "tiles": c["n"], "ms_p50": ms, "sm_clock_load_mhz": 1710.0,
                        "experts_declared": E if arm == "native" else E * copies, "histogram": c["label"],
                        "counts_sha256": RS.sha_counts(c["counts"]),
                        "bincount_ok": True, "host_bound": False,
                        "shuffle_seed": None if c["counts"] is None else doc["shuffle_seed"]})
    return out


PRED = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r for r in REG["predictions"]}


def _ms_S(noise=0.0005, seed=1):
    rng = random.Random(seed)
    return lambda m, p, n, lab, arm: PRED[(m, p, n, lab, arm)]["ms"]["S"] * (1 + rng.gauss(0, noise))


def _counters(tree: Path, model: str, factor=1.0, mhz=1710.0, slow=None, bad_sha=False):
    import r3_timing_model as TM
    old = TM.set_model(model)
    bm = dict(TM.BYTE_MODEL)
    TM.set_model(old)
    doc = _hist_doc(f"{model}-C.json")
    counts = {(c["label"], c["n"]): c["counts"] for c in doc["cells"]}
    cells = []
    for r in REG["byte_predictions"]:
        if r["model"] != model or r["q_pred"] is None:
            continue
        clk = slow if slow and (r["label"], r["n"], r["arm"]) == slow[0] else mhz
        if slow and (r["label"], r["n"], r["arm"]) == slow[0]:
            clk = slow[1]
        cells.append({"arm": r["arm"], "n": r["n"], "histogram": r["label"],
                      "counts_sha256": "0" * 64 if bad_sha else r["counts_sha256"],
                      "bincount": counts[(r["label"], r["n"])],
                      "recorded": {g: {"sm__cycles_elapsed.avg": clk * 1e3} for g in ("w1", "w2")},
                      "per_gemm": {g: {"dram_bytes_read": r["q_pred"][g] * factor * bm[f"W_{g}"]
                                       + r["n"] * bm[f"operand_per_tile_{g}"], "gpu_time_ns": 1e6}
                                   for g in ("w1", "w2")}})
    d = tree / "results" / f"2026-10-08-{CARD}-{model}-skc-r3-counters" / "lock1710"
    d.mkdir(parents=True, exist_ok=True)
    (d / "r3c-g8.json").write_text(json.dumps({"design": {"histogram": {
        "file_sha256": REG["histograms"]["files"][f"{model}-C.json"]["sha256"],
        "shuffle_seed": REG["histograms"]["files"][f"{model}-C.json"]["shuffle_seed"]}, "histogram_page": True},
        "cells": cells}))


def _tree(tmp_path, ms_of, counters=True, factor=1.0, **ckw):
    for model in SK.RANK:
        for page in ("A", "B"):
            _write_page(tmp_path, model, f"sk{page.lower()}", f"{model}-{page}.json", _rows(model, page, ms_of))
        if counters and model != "qwen1.5-moe-a2.7b":
            _counters(tmp_path, model, factor, **ckw)
    return tmp_path


@pytest.fixture(scope="module")
def s_world(tmp_path_factory):
    tree = _tree(tmp_path_factory.mktemp("s"), _ms_S())
    return SK.score(REPO, tree)


def test_pages_that_follow_s_pass_g3_and_hold_s(s_world):
    res = s_world
    assert res["G3"]["verdict"] == "PASS", res["G3"]
    assert res["SKEW-RATIO"]["S"]["verdict"] == "HOLDS"
    assert res["SKEW-RATIO"]["rivals"]["U"]["verdict"] == "EXCLUDED"
    assert res["SKEW-RATIO"]["rivals"]["PW"]["verdict"] == "EXCLUDED"
    assert res["E1-SKEW"]["verdict"] == "HOLDS" and res["E1-UNI-EXT"]["verdict"] == "HOLDS"
    assert res["perm_control"]["verdict"] == "PASS"
    assert res["B-SKEW"]["verdict"] == "HOLDS" and res["E2-SKEW"]["verdict"] == "HOLDS"
    assert len(res["G3"]["d"]) == 36 and res["G3"]["pooled"]["clusters"] == 6
    assert res["secondary"]["calibration"]["verdict"].startswith("NOT RESOLVED (3 clusters")
    assert res["secondary"]["top1_regret"].startswith("NOT APPLICABLE")
    text = "\n".join(SK.lines(res))
    assert "G3: PASS" in text and chr(0x2014) not in text


def test_pages_with_no_skew_effect_fail_s_and_keep_u(tmp_path):
    def u_world(m, p, n, lab, arm):
        key = (m, p, n, "uniform" if lab in ("PT", "PW", "DW") or lab.startswith("PW-") else lab, arm)
        return PRED[key]["ms"]["S"]
    res = SK.score(REPO, _tree(tmp_path, u_world, counters=False))
    assert res["SKEW-RATIO"]["S"]["verdict"] == "FAILS"
    assert res["SKEW-RATIO"]["rivals"]["U"]["verdict"] == "NOT EXCLUDED"
    assert res["B-SKEW"]["verdict"].startswith("NOT SCORED") and res["E2-SKEW"]["verdict"].startswith("NOT SCORED")


def test_a_uniform_cell_off_its_balanced_cell_fails_g3_and_voids_the_histogram_cells(tmp_path):
    base = _ms_S(0.0002, 3)

    def shifted(m, p, n, lab, arm):
        v = base(m, p, n, lab, arm)
        return v * (1.01 if lab != "balanced" else 1.0)
    res = SK.score(REPO, _tree(tmp_path, shifted, counters=False))
    assert res["G3"]["verdict"].startswith("FAIL") and res["G3"]["void_strata"] == "ALL"
    assert res["SKEW-RATIO"]["S"].startswith("NOT SCORED") and res["E1-SKEW"]["verdict"].startswith("NOT SCORED")


def test_a_page_with_another_histogram_file_is_not_scored(tmp_path):
    _write_page(tmp_path, "mixtral-8x7b", "ska", "mixtral-8x7b-B.json", _rows("mixtral-8x7b", "A", _ms_S()))
    res = SK.score(REPO, tmp_path)
    assert "G5" in res["pages"]["mixtral-8x7b-A"]
    assert res["G3"]["verdict"].startswith("NOT SCORED")


def test_a_g1_failed_page_is_ruled_by_the_two_views(tmp_path):
    """Post-page fix 2026-10-09 (rental 5's Qwen1.5 skb, G1 FAIL on a drift flag): G1 is a
    VALIDITY gate the page records in `gates`, so ALL counts the page and CLEAN excludes it
    (the registration's `views`); a page that records G1 nowhere in `gates` is still refused."""
    tree = _tree(tmp_path, _ms_S(), counters=False)
    page = next((tree / "results").glob(f"gaps-{CARD}-qwen1.5-moe-a2.7b-skb/*/*/report.json"))
    rep = json.loads(page.read_text())
    rep["histogram_gates"]["G1_lock_thermal"] = {"verdict": "FAIL"}
    rep["gates"] = [{"kind": "VALIDITY", "tag": "G1_lock_thermal", "verdict": "FAIL"}]
    page.write_text(json.dumps(rep))
    res = SK.score(REPO, tree)
    rec = res["addendum"]["pages"]["qwen1.5-moe-a2.7b-skb"]
    assert rec["ALL"]["use"] == "counted" and rec["CLEAN"]["use"] == "excluded"
    assert res["pages"]["qwen1.5-moe-a2.7b-B"] == "counted"
    assert res["G3"]["pooled"]["clusters"] == 6
    assert res["addendum"]["CLEAN"]["G3"]["pooled"]["clusters"] == 5
    rep["gates"] = []
    page.write_text(json.dumps(rep))
    res = SK.score(REPO, tree)
    assert res["pages"]["qwen1.5-moe-a2.7b-B"].startswith("NOT SCORED (G1)")


def test_no_pages_score_nothing(tmp_path):
    res = SK.score(REPO, tmp_path)
    assert res["G3"]["verdict"].startswith("NOT SCORED") and res["SKEW-RATIO"]["S"].startswith("NOT SCORED")
    assert res["B-SKEW"]["verdict"].startswith("NOT SCORED")


# --------------------------------------------------------------------------
# c15
# --------------------------------------------------------------------------

def _c15_tree(tmp_path, delta_us):
    m = "olmoe-1b-7b"
    base = {}
    for r in REG15["predictions"]:
        base[(r["label"], r["n"])] = r["S9_ms"]

    def c9(model, page, n, lab, arm):
        return PRED[(model, page, n, lab, arm)]["ms"]["S"]

    def c15(model, page, n, lab, arm):
        if arm == "native":
            return PRED[(m, "A", n, "uniform", "native")]["ms"]["S"]
        c9page = "B" if lab == "DW" else "A"
        return PRED[(m, c9page, n, lab, "shared")]["ms"]["S"] + delta_us * 1e-3
    _write_page(tmp_path, m, "ska", f"{m}-A.json", _rows(m, "A", c9))
    _write_page(tmp_path, m, "skb", f"{m}-B.json", _rows(m, "B", c9))
    _write_page(tmp_path, m, "sk15", f"{m}-c15.json", _rows(m, "c15", c15, copies=15, file=f"{m}-c15.json"), copies=15)
    return tmp_path


def test_c15_holds_d_at_its_dead_increment_and_excludes_m_and_fixed(tmp_path):
    d = REG15["predicted_median_us"]["D"]
    res = SC15.score(REPO, _c15_tree(tmp_path, d))
    assert res["verdict"] == "HOLDS" and res["M"] == "EXCLUDED" and res["FIXED"] == "EXCLUDED", res
    assert res["SKEW-DEAD"]["verdict"] == "NOT SHOWN" and res["alignment"].startswith("RAW")
    assert math.isfinite(res["two_regressor"]["VIF"])


def test_c15_fails_d_at_m_s_increment(tmp_path):
    res = SC15.score(REPO, _c15_tree(tmp_path, REG15["predicted_median_us"]["M"]))
    assert res["verdict"] == "FAILS" and res["M"] == "NOT EXCLUDED"


def test_c15_with_a_page_missing_is_not_scored(tmp_path):
    res = SC15.score(REPO, tmp_path)
    assert res["verdict"].startswith("NOT SCORED")


# --------------------------------------------------------------------------
# build-r5-review F5, F6 and the shuffle-seed G5 check
# --------------------------------------------------------------------------

def test_the_byte_leg_drops_a_cell_under_1705_mhz_and_refuses_a_wrong_counts_sha(tmp_path):
    slow = (("PT", 2, "native"), 1650.0)
    res = SK.score(REPO, _tree(tmp_path / "a", _ms_S(), slow=slow))
    assert "PT/native/n2: G4 clock 1650.0" in res["byte_pages"]["mixtral-8x7b-dropped"]
    assert not any(r["cell"] == "mixtral-8x7b/n2/PT/native" for r in res["B-SKEW"]["rows"])
    assert res["B-SKEW"]["verdict"] == "HOLDS"
    res = SK.score(REPO, _tree(tmp_path / "b", _ms_S(), bad_sha=True))
    assert res["byte_pages"]["olmoe-1b-7b"].startswith("NOT SCORED (G5)")
    assert res["B-SKEW"]["verdict"].startswith("NOT SCORED")


def test_a_g3_void_drops_the_byte_cells_too(tmp_path):
    base = _ms_S(0.0002, 3)

    def shifted(m, p, n, lab, arm):
        return base(m, p, n, lab, arm) * (1.01 if lab != "balanced" else 1.0)
    res = SK.score(REPO, _tree(tmp_path, shifted))
    assert res["G3"]["void_strata"] == "ALL"
    assert all("G3 void" in d for d in res["byte_pages"]["mixtral-8x7b-dropped"])
    assert res["B-SKEW"]["verdict"].startswith("NOT SCORED")


def test_no_g3_row_scores_no_histogram_cell(tmp_path):
    def no_balanced(m, p, n, lab, arm):
        return None if lab == "balanced" else PRED[(m, p, n, lab, arm)]["ms"]["S"]
    res = SK.score(REPO, _tree(tmp_path, no_balanced))
    assert res["G3"]["verdict"].startswith("NOT SCORED")
    assert res["SKEW-RATIO"]["S"].startswith("NOT SCORED: no G3 row")
    assert res["E1-SKEW"]["verdict"].startswith("NOT SCORED")
    assert res["B-SKEW"]["verdict"].startswith("NOT SCORED: no G3 row") and res["E2-SKEW"]["verdict"].startswith("NOT SCORED")
    assert res["perm_control"]["verdict"].startswith("NOT SCORED")


def test_a_page_without_its_shuffle_seed_is_not_scored(tmp_path):
    rows = _rows("olmoe-1b-7b", "A", _ms_S())
    for r in rows:
        r["shuffle_seed"] = None
    _write_page(tmp_path, "olmoe-1b-7b", "ska", "olmoe-1b-7b-A.json", rows)
    res = SK.score(REPO, tmp_path)
    assert res["pages"]["olmoe-1b-7b-A"].startswith("NOT SCORED (G5): row")


def test_the_registration_states_coverage_power_and_the_fixed_margin():
    g = REG["gates"]
    assert "0.4%" in g["G3_power"] and "5.2%" in g["G3_power"] and "38%" in g["G3_power"]
    assert "fixed at +-0.36%" in g["G3_margin_fixed"] and "NOT SCORED" in g["G3_no_rows"]
    assert "p90" in REG["histograms"]["coverage"] and "least-skewed" in REG["histograms"]["coverage"]
    bal = [r for r in REG["predictions"] if r["label"] == "balanced"]
    assert all(r["label_blind"].startswith(("CAL", "SEEN")) for r in bal if r["n"] <= 6)
    assert all("BLIND" in r["label_blind"] for r in bal if r["n"] > 6)
    assert REG15["noise"]["sigma_noise_us"] == pytest.approx(1.29, abs=0.01)
    sep = float(REG15["rules"]["separation"].split("D against M ")[1].split(" sigma")[0])
    assert 6.1 <= sep <= 6.3
