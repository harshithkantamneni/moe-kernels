"""Rental 7's scorers on SYNTHETIC pages only (no rental-7 page exists):
scripts/scoring/rental7/score_skew.py, score_q1.py and score_l2open.py against
docs/registered/2026-10-13-rental7-*.json, and r7common's rules (G1 under r7-median-run3 with the
scorer's recomputation, the P20 view, the per-shape E1 clause and the CLEAN certification, Q1-H's
three clusters and its rivals scored apart from the copies clause, the l2open naming rule)."""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
R7DIR = REPO / "scripts" / "scoring" / "rental7"
for p in (REPO, REPO / "scripts", R7DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r7common as C7  # noqa: E402

SS = C7._load("rental7_score_skew", R7DIR / "score_skew.py")
SQ = C7._load("rental7_score_q1", R7DIR / "score_q1.py")
SL = C7._load("rental7_score_l2open", R7DIR / "score_l2open.py")

from moe.spec import MODEL_CONFIGS  # noqa: E402

CARD = C7.CARD
REGS = {p: C7.registration(REPO, p) for p in C7.PARTS}
PRED = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r
        for r in REGS["skew"]["predictions"] + REGS["q1"]["predictions"]}
RULE = C7.RULE


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "docs" / "registered").mkdir(parents=True)
    add = C7.C4.CM2.ADDENDUM
    shutil.copy(REPO / "docs" / "registered" / f"{add}.json", r / "docs" / "registered")
    for p in C7.PARTS:
        shutil.copy(REPO / "docs" / "registered" / f"{C7.NAMES[p]}.json", r / "docs" / "registered")
    return r


def _hist(name):
    files = {**REGS["skew"]["histograms"]["files"], **REGS["q1"]["pages"]["files"], **REGS["l2open"]["page"]["file"]}
    return files[name], json.loads((REPO / files[name]["path"]).read_text())


CSV_FIELDS = ("arm", "repeat", "tiles", "ms_p50", "sm_clock_load_mhz", "clock_drift_ok", "status", "detail",
              "clock_samples_mhz", "host_bound")


def write_timed(tree: Path, model, label, page, name, ms_of, *, copies=9, probe=True, clean=None, flagged=None,
                min_clean=0, short=None):
    """A rental-7 R3 page as rental 6's R3 writes it at --slip-policy cell --min-clean-repeats
    `min_clean`: cells.csv with 9 repeats per cell (each with 15 reads), and report.json whose stored
    G1 is r5-any's (FAIL here whenever a flag is planted) and whose stored medians are garbage, since
    the scorer must read neither. `clean` {(label, n, arm): k}: 9 - k repeats carry a run of 3
    dipped reads; `flagged` {(label, n, arm): f}: f of the clean repeats carry one endpoint dip
    (drift flag); `short` {(label, n, arm): m}: cells.csv holds only m repeats (a defect)."""
    import csv
    reg_file, doc = _hist(name)
    E = MODEL_CONFIGS[model].num_experts
    rows, raw = [], []
    for c in doc["cells"]:
        for arm in c.get("arms") or ("native", "shared"):
            ms = ms_of(model, page, c["n"], c["label"], arm)
            if ms is None:
                continue
            k = (c["label"], c["n"], arm)
            cr, fl = (clean or {}).get(k, 9), (flagged or {}).get(k, 0)
            for rep in range((short or {}).get(k, 9)):
                reads = [1710] * 15
                if rep >= cr:
                    reads[5:8] = [1600] * 3
                drift = not (rep < fl)
                if not drift:
                    reads[-1] = 1600
                raw.append({"arm": arm, "repeat": rep, "tiles": c["n"], "ms_p50": repr(ms), "sm_clock_load_mhz": "1710.0",
                            "clock_drift_ok": str(drift), "status": "ok", "detail": f"histogram={c['label']}; LEVEL HIGH",
                            "clock_samples_mhz": " ".join(map(str, reads)), "host_bound": "False"})
            rows.append({"repeat_rows": 9, "lock_slip_repeats": 0, "clean_repeats": 9, "worst_repeat_mhz": 1710.0,
                         "lost": False, "arm": arm, "tiles": c["n"], "ms_p50": 99.0, "sm_clock_load_mhz": 1200.0,
                         "experts_declared": E if arm == "native" else E * copies, "repeats": 9, "host_bound": False,
                         "histogram": c["label"], "counts_sha256": next(x["counts_sha256"] for x in reg_file["cells"]
                                                                        if x["label"] == c["label"] and x["n"] == c["n"]),
                         "bincount_ok": True, "shuffle_seed": None if c["counts"] is None else doc["shuffle_seed"]})
    g1 = "FAIL" if (flagged or clean) else "PASS"
    gates = {"G1_lock_thermal": {"verdict": g1}, "G5_provenance": {"verdict": "PASS", "missing": []}}
    probe_cells = []
    for n in sorted({c["n"] for c in doc["cells"]}):
        a = PRED.get((model, page, n, "balanced", "shared"), {}).get("A_proxy_us") or 0.0
        probe_cells += [{"label": "native", "tread": n, "ms": 0.005}, {"label": "shared", "tread": n, "ms": 0.005 + a * 1e-3}]
    d = tree / "results" / f"gaps-{CARD}-{model}-{label}" / "private_weight_reference" / f"run-{label}"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(raw)
    (d / "report.json").write_text(json.dumps({
        "experiment": "private_weight_reference", "kind": "histogram-page", "session_tag": f"t-{model}-{label}",
        "pinned": {"GROUP_SIZE_M": 8}, "copies_declared": copies,
        "histogram": {"file_sha256": reg_file["sha256"], "shuffle_seed": reg_file["shuffle_seed"]},
        "histogram_gates": gates, "treads_table": rows,
        "gates": [{"kind": "VALIDITY", "tag": k, "verdict": v["verdict"]} for k, v in gates.items()],
        "align_probe": {"cells": probe_cells} if probe else None, "slip_policy": "cell", "lock_mhz": 1710,
        "min_clean_repeats": min_clean, "repeat_rows": 9 * len(rows), "lock_slip_rows": 0, "slipped_repeats": []}))


def ms_v3(noise=0.0003, seed=1, variant="v3", arms=("native", "shared"), bias=None):
    rng = random.Random(seed)

    def f(m, p, n, lab, arm):
        r = PRED[(m, p, n, lab, arm)]
        v = variant if arm in arms else "v3"
        return r["ms"][v] * (1 + rng.gauss(0, noise)) * (1 + ((bias or {}).get(m, 0.0) if lab in SS.SKEWED else 0.0))
    return f


def all_timed(tree, ms_of=None, per=None, q1=True, q15=True):
    ms_of = ms_of or ms_v3()
    kw = per or {}
    for m in SS.RANK:
        for page in SS.PAGES:
            if kw.get((m, page)) == "missing":
                continue
            write_timed(tree, m, f"sk{page.lower()}", page, f"{m}-{page}.json", ms_of, **kw.get((m, page), {}))
    if q1:
        for m in REGS["q1"]["pages"]["models"]:
            for c in ((9, 15) if q15 else (9,)):
                write_timed(tree, m, f"q{c}", f"c{c}", f"q1-{m}.json", ms_of, copies=c)


# --------------------------------------------------------------------------
# G1 under r7-median-run3 (r7common)
# --------------------------------------------------------------------------

def _page(tmp_path, **kw):
    write_timed(tmp_path, "phi-3.5-moe", "ska", "A", "phi-3.5-moe-A.json", ms_v3(), **kw)
    d = next(tmp_path.rglob("report.json")).parent
    rep, raw = C7.read_page(d)
    return rep, raw


def test_the_scorer_rebuilds_g1_from_cells_csv_and_ignores_the_stored_one(tmp_path):
    rep, raw = _page(tmp_path, flagged={("PT", 2, "native"): 4})
    assert rep["histogram_gates"]["G1_lock_thermal"]["verdict"] == "FAIL"   # r5-any on one flag
    r, why = C7.rebuild(rep, raw)
    assert r is not None and r["g1_r7"]["verdict"] == "PASS" and r["g1_p20"]["verdict"] == "PASS", why
    row = next(t for t in r["treads_table"] if (t["histogram"], t["tiles"], t["arm"]) == ("PT", 2, "native"))
    assert row["ms_p50"] != 99.0 and row["sm_clock_load_mhz"] == 1710.0 and row["clean_repeats"] == 9
    assert row["clean_repeats_p20"] == 5 and row["lost_p20"] is True and row["lost"] is False


@pytest.mark.parametrize("last, state", [(6, "PASS"), (5, "FAIL")])
def test_the_ten_percent_cap_counts_every_row(tmp_path, last, state):
    """Phi A: 24 cells x 9 = 216 rows, the cap 21.6: two cells with no clean repeat and one with
    `last` clean is 18 + 9 - last non-clean rows."""
    rep, raw = _page(tmp_path, clean={("PT", 2, "native"): 0, ("PT", 2, "shared"): 0, ("PW", 2, "native"): last})
    r, _ = C7.rebuild(rep, raw)
    g = r["g1_r7"]
    assert g["repeat_rows"] == 216 and g["non_clean_rows"] == 18 + 9 - last and g["verdict"] == state


# --------------------------------------------------------------------------
# rental7-skew
# --------------------------------------------------------------------------

def test_pages_that_follow_v3_hold_every_test_and_certify_both_shapes(tmp_path, repo):
    all_timed(tmp_path)
    res = SS.score(repo, tmp_path)
    assert res["G3"]["verdict"] == "PASS", res["G3"]
    assert res["E1-SKEW"]["verdict"] == "HOLDS" and res["E1-UNI"]["verdict"] == "HOLDS"
    assert all(v["verdict"] == "HOLDS" for v in res["E1-SKEW"]["per_shape"].values())
    assert res["certification"] == {"mixtral-8x7b": "CERTIFIED", "phi-3.5-moe": "CERTIFIED"}
    assert res["SKEW-RATIO"]["v3"]["verdict"] == "HOLDS" and res["SKEW-RATIO"]["U"]["verdict"] == "EXCLUDED"
    assert res["E1-rivals"]["LT3"]["verdict"] == "EXCLUDED"
    for R in ("v2", "H8", "v3_all", "v3_noca"):
        assert res["E1-rivals"][R]["verdict"].startswith("UNDECIDED"), R
    assert res["P20_view"]["label"].startswith("DESCRIPTIVE") and res["P20_view"]["combined"]["E1-SKEW"]["verdict"] == "HOLDS"
    txt = "\n".join(SS.lines(res))
    assert "certification (CLEAN)" in txt and "P20 view (DESCRIPTIVE)" in txt
    empty = SS.score(repo, tmp_path / "none")
    assert empty["E1-SKEW"]["verdict"].startswith("NOT SCORED")
    assert empty["certification"]["phi-3.5-moe"].startswith("NOT CERTIFIED")


def test_one_shape_off_the_bar_falsifies_e1_while_the_pooled_bar_holds(tmp_path, repo):
    """The per-shape clause: Phi's skewed cells 2% fast against v3 (Phi's mean e about +1.2% over
    its 42 cells, past the 1% bar), Mixtral's on v3. Pooled the bar holds (mean about +0.6%);
    E1-SKEW is FALSIFIED and Phi is NOT CERTIFIED."""
    all_timed(tmp_path, ms_v3(bias={"phi-3.5-moe": -0.02}))
    res = SS.score(repo, tmp_path)
    e1 = res["E1-SKEW"]
    assert abs(e1["stats"]["mean"]) < 0.01 and e1["stats"]["rms"] < 0.02, e1["stats"]
    assert e1["per_shape"]["phi-3.5-moe"]["verdict"] == "FALSIFIED" and e1["per_shape"]["mixtral-8x7b"]["verdict"] == "HOLDS"
    assert e1["verdict"].startswith("FALSIFIED")
    assert res["certification"]["phi-3.5-moe"].startswith("NOT CERTIFIED")
    assert res["certification"]["mixtral-8x7b"].startswith("NOT CERTIFIED (E1-SKEW's registered verdict")


def test_a_page_over_the_cap_counts_in_all_and_two_out_leave_a_shape_uncertified(tmp_path, repo):
    over = {("PT", n, a): 0 for n in (2, 4, 12) for a in ("native", "shared")}
    all_timed(tmp_path, per={("phi-3.5-moe", "A"): {"clean": over}, ("phi-3.5-moe", "B"): "missing"})
    res = SS.score(repo, tmp_path)
    pg = res["addendum"]["pages"]["phi-3.5-moe-ska"]
    assert pg["ALL"]["use"] == "counted" and pg["CLEAN"]["use"].startswith("excluded"), pg
    assert "over the 10% cap" in res["pages"]["phi-3.5-moe-A"] and pg["CLEAN"]["g1_r7"].startswith("G1 FAIL (rebuilt)")
    assert res["pages"]["phi-3.5-moe-B"].startswith("NOT SCORED")
    clean_shape = res["addendum"]["CLEAN"]["E1-SKEW"]["per_shape"]["phi-3.5-moe"]
    assert clean_shape["pages_in_view"] == ["C"] and clean_shape["certification"].startswith("NOT CERTIFIED (1 of 3")
    assert res["certification"]["phi-3.5-moe"].startswith("NOT CERTIFIED")
    assert res["certification"]["mixtral-8x7b"] == "CERTIFIED"


def test_a_page_at_the_old_min_clean_or_with_short_cells_csv_is_not_scored(tmp_path, repo):
    all_timed(tmp_path, per={("mixtral-8x7b", "A"): {"min_clean": 6}, ("mixtral-8x7b", "B"): {"short": {("DW", 3, "native"): 8}}})
    res = SS.score(repo, tmp_path)
    assert res["pages"]["mixtral-8x7b-A"].startswith("NOT SCORED: the page did not run --slip-policy cell --min-clean-repeats 0")
    assert res["pages"]["mixtral-8x7b-B"].startswith("NOT SCORED (rebuild): cells.csv holds 8 repeats")
    assert res["addendum"]["CLEAN"]["E1-SKEW"]["per_shape"]["mixtral-8x7b"]["certification"].startswith("NOT CERTIFIED (1 of 3")


def test_the_p20_view_drops_flagged_repeats_and_a_page_past_twenty_percent(tmp_path, repo):
    """Endpoint flags are recorded and kept under r7; the P20 view takes them out: 5 flags in each
    of Phi A's PT and PW cells lose those cells in the view (4 clean of 9) and put the page past
    20% (60 of 216 rows), so the view drops it from CLEAN while the registered view keeps it."""
    fl = {(lab, n, a): 5 for lab in ("PT", "PW") for n in (2, 4, 12) for a in ("native", "shared")}
    all_timed(tmp_path, per={("phi-3.5-moe", "A"): {"flagged": fl}})
    res = SS.score(repo, tmp_path)
    assert res["addendum"]["pages"]["phi-3.5-moe-ska"]["CLEAN"]["use"] == "counted"
    p20 = res["P20_view"]["pages"]["phi-3.5-moe-ska"]
    assert p20["CLEAN"]["use"].startswith("excluded") and p20["ALL"]["use"] == "counted", p20
    drop = res["P20_view"]["combined"]["dropped_rows"]["phi-3.5-moe-A"]
    assert len([d for d in drop if "4 clean repeats" in d[1]]) == 12


# --------------------------------------------------------------------------
# rental7-q1
# --------------------------------------------------------------------------

def _q1c(base):
    r = REGS["q1"]["tests"]["Q1-C"]["dead_ratio"]

    def ms(m, p, n, lab, arm):
        v = base(m, p, n, lab, arm)
        if m == "qwen1.5-moe-a2.7b" and n == 1 and arm == "shared":
            v += (-6.6 if p == "c9" else -6.6 * r) * 1e-3
        return v
    return ms


def test_q1_holds_on_v3_with_five_clusters_and_excludes_v2_and_h8(tmp_path, repo):
    all_timed(tmp_path, _q1c(ms_v3(noise=0.0)))
    res = SQ.score(repo, tmp_path)
    h = res["Q1-H"]
    assert h["verdict"] == "HOLDS" and abs(h["mean_g_us"]) < 1e-6 and h["clusters"] == 5, h
    assert h["pairs"] == len(REGS["q1"]["tests"]["Q1-H"]["pairs"]) == 24
    assert h["rivals"]["v2"]["verdict"] == "EXCLUDED" and h["rivals"]["H8"]["verdict"] == "EXCLUDED"
    assert h["rivals"]["v3_all"]["verdict"] == "NOT EXCLUDED"
    assert h["label"].startswith("REPLICATE") and h["design_population_printed"]["reads_as"] == "HOLDS"
    assert res["Q1-P"]["verdict"].startswith("DESCRIPTIVE") and res["Q1-C"]["verdict"] == "SELECTED DEAD", res["Q1-C"]
    assert isinstance(res["Q1-SK"]["phi-3.5-moe n 2"], dict)


def test_the_rivals_are_scored_when_the_copies_clause_is_undecided(tmp_path, repo):
    """Owner decision 2: with no c15 page the copies clause is UNDECIDED and so is v3, but the
    rivals are scored on the valid pairs (rental 6 left them unread)."""
    all_timed(tmp_path, ms_v3(noise=0.0), q15=False)
    h = SQ.score(repo, tmp_path)["Q1-H"]
    assert h["verdict"].startswith("UNDECIDED") and h["copies_clause"]["verdict"].startswith("UNDECIDED"), h
    assert h["rivals"]["v2"]["verdict"] == "EXCLUDED" and h["rivals"]["H8"]["verdict"] == "EXCLUDED"


def test_q1h_needs_three_clusters(tmp_path, repo):
    """Phi's pages alone (Phi n 2 and n 4): 14 pairs but 2 clusters, so the CR1 se is not defined
    and Q1-H, rivals included, is UNDECIDED."""
    for page in SS.PAGES:
        write_timed(tmp_path, "phi-3.5-moe", f"sk{page.lower()}", page, f"phi-3.5-moe-{page}.json", ms_v3(noise=0.0))
    for c in (9, 15):
        write_timed(tmp_path, "phi-3.5-moe", f"q{c}", f"c{c}", "q1-phi-3.5-moe.json", ms_v3(noise=0.0), copies=c)
    h = SQ.score(repo, tmp_path)["Q1-H"]
    assert h["pairs"] == 14 and h["clusters"] == 2 and h["verdict"].startswith("UNDECIDED"), h
    assert all(v["verdict"].startswith("UNDECIDED") for v in h["rivals"].values())


def test_q1_fails_when_the_shared_gap_is_v2s(tmp_path, repo):
    all_timed(tmp_path, ms_v3(noise=0.0, variant="v2", arms=("shared",)))
    h = SQ.score(repo, tmp_path)["Q1-H"]
    assert h["verdict"] == "FAILS" and h["rivals"]["v2"]["verdict"] == "NOT EXCLUDED"


def test_the_copies_clause_matches_by_model_and_equals_the_unmatched_mean_with_every_page():
    rows = [{"model": m, "copies": c, "q1page": True, "g": g, "cluster": f"{m}|n2", "g_R": {}}
            for m, (a, b) in {"x": (0.0, 1.0), "y": (2.0, 2.5), "z": (-1.0, -0.5)}.items() for c, g in ((9, a), (15, b))]
    cc = C7.copies_clause(rows)
    unmatched = sum(r["g"] for r in rows if r["copies"] == 15) / 3 - sum(r["g"] for r in rows if r["copies"] == 9) / 3
    assert cc["c15_minus_c9_us"] == pytest.approx(unmatched) and cc["verdict"] == "HOLDS"
    # one model without its c15 pair: it leaves the clause instead of biasing it
    cc2 = C7.copies_clause([r for r in rows if not (r["model"] == "y" and r["copies"] == 15)])
    assert cc2["models"].keys() == {"x", "z"} and cc2["c15_minus_c9_us"] == pytest.approx(0.75)


def test_the_registered_q1h_power_matches_the_reviews_monte_carlo():
    """design-r7-review work/mc_q1h.json (R 400,000): the registered population (+ Mixtral n 6 +
    Phi n 4) reads P(v3 HOLDS) 0.983 at tau 0.9 and 0.929 at tau 1.3; the design's 0.886, 0.791."""
    pts = REGS["q1"]["tests"]["Q1-H"]["power"]["points"]
    assert pts["registered population | v3 | tau 0.9"]["P(v3 HOLDS)"] == pytest.approx(0.983, abs=0.003)
    assert pts["registered population | v3 | tau 1.3"]["P(v3 HOLDS)"] == pytest.approx(0.929, abs=0.003)
    assert pts["design population only (3 clusters) | v3 | tau 0.9"]["P(v3 HOLDS)"] == pytest.approx(0.886, abs=0.003)
    assert pts["design population only (3 clusters) | v3 | tau 1.3"]["P(v3 HOLDS)"] == pytest.approx(0.791, abs=0.003)
    assert pts["registered population | v2 true | tau 0.9"]["P(v3 HOLDS)"] == 0.0


# --------------------------------------------------------------------------
# rental7-l2open
# --------------------------------------------------------------------------

def write_l2open(tree, truth="v3", *, noise=0.003, mhz=1700.0, smi=1710.0, seed=3, wrong_sha=False):
    import r3_timing_model as TM
    old = TM.set_model("olmoe-1b-7b")
    bm = dict(TM.BYTE_MODEL)
    TM.set_model(old)
    reg_file, _doc = _hist("olmoe-1b-7b-l2open.json")
    rng = random.Random(seed)
    cells = []
    for r in REGS["l2open"]["byte_predictions"]:
        q = {"w1": r["q_pred"][truth]["w1"] * (1 + rng.gauss(0, noise)), "w2": r["q_pred"]["v3"]["w2"]}
        ns = 1.0e6
        cells.append({"arm": r["arm"], "n": r["n"], "histogram": r["label"],
                      "counts_sha256": ("0" * 64 if wrong_sha else r["counts_sha256"]),
                      "recorded": {g: {"sm__cycles_elapsed.avg": mhz * ns / 1e3} for g in ("w1", "w2")},
                      "per_gemm": {g: {"dram_bytes_read": q[g] * bm[f"W_{g}"] + r["n"] * bm[f"operand_per_tile_{g}"],
                                       "gpu_time_ns": ns} for g in ("w1", "w2")}})
    d = tree / "results" / f"2026-10-14-{CARD}-olmoe-1b-7b-l2open-r3-counters" / "lock1710"
    d.mkdir(parents=True, exist_ok=True)
    smi_rec = {"rows": [{"clocks.sm": smi}]}
    (d / "r3c-g8.json").write_text(json.dumps({
        "design": {"model": "olmoe-1b-7b", "histogram": {"file_sha256": reg_file["sha256"], "shuffle_seed": reg_file["shuffle_seed"]}},
        "ncu": {"lock_mhz": 1710, "smi_before": smi_rec, "smi_after": smi_rec}, "cells": cells, "gates": []}))


@pytest.mark.parametrize("truth", ["v3", "ROWKEY39.5", "ROWKEY60"])
def test_l2open_names_the_rival_the_page_follows(tmp_path, repo, truth):
    write_l2open(tmp_path, truth)
    res = SL.score(repo, tmp_path)
    assert res["L2OPEN"]["verdict"] == f"NAMED {truth}", res["L2OPEN"]
    assert res["status"].startswith("OPEN") and res["L2OPEN"]["skewed_rows"] == 10
    assert len(res["rows"]) == len(REGS["l2open"]["byte_predictions"]) == 20


def test_l2open_names_nothing_between_the_two_capacities_or_off_the_lock(tmp_path, repo):
    """A page halfway between ROWKEY39.5 and ROWKEY60 in q is near both: NOT SEPARATED. A page
    off the lock, or carrying another histogram, is NOT SCORED."""
    import r3_timing_model as TM
    write_l2open(tmp_path, "ROWKEY60", noise=0.0)
    p = next(tmp_path.rglob("r3c-g8.json"))
    d = json.loads(p.read_text())
    old = TM.set_model("olmoe-1b-7b")
    bm = dict(TM.BYTE_MODEL)
    TM.set_model(old)
    pred = {(r["n"], r["label"], r["arm"]): r for r in REGS["l2open"]["byte_predictions"]}
    for c in d["cells"]:
        r = pred[(c["n"], c["histogram"], c["arm"])]
        q = 0.5 * (r["q_pred"]["ROWKEY60"]["w1"] + r["q_pred"]["ROWKEY39.5"]["w1"])
        c["per_gemm"]["w1"]["dram_bytes_read"] = q * bm["W_w1"] + c["n"] * bm["operand_per_tile_w1"]
    p.write_text(json.dumps(d))
    assert SL.score(repo, tmp_path)["L2OPEN"]["verdict"].startswith("NOT SEPARATED")
    off = tmp_path / "off"
    write_l2open(off, "ROWKEY60", smi=1980.0)
    assert SL.score(repo, off)["L2OPEN"]["verdict"].startswith("NOT SCORED (clock rule)")
    wrong = tmp_path / "wrong"
    write_l2open(wrong, "ROWKEY60", wrong_sha=True)
    assert SL.score(repo, wrong)["L2OPEN"]["verdict"].startswith("NOT SCORED (G5)")
    assert SL.score(repo, tmp_path / "none")["L2OPEN"]["verdict"].startswith("NOT SCORED")


def test_the_five_percent_clause_reads_skewed_rows_only():
    rows = [{"skewed": True, "e": {"v3": 0.15, "ROWKEY39.5": 0.06, "ROWKEY60": 0.01}} for _ in range(4)]
    rows += [{"skewed": False, "e": {"v3": 0.02, "ROWKEY39.5": 0.07, "ROWKEY60": 0.065}}]
    v = C7.l2open_rule(rows)
    assert v["verdict"] == "NAMED ROWKEY60", v      # the uniform row's +6.5% does not stop it
    rows[0]["e"]["ROWKEY60"] = 0.051
    assert C7.l2open_rule(rows)["verdict"].startswith("NOT SEPARATED")
