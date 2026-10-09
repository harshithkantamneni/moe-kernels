"""Rental 6's scorers on SYNTHETIC pages only (no rental-6 page exists):
scripts/scoring/rental6/score_skew.py, score_q1.py and score_secondk.py against
docs/registered/2026-10-09-rental6-*.json, and r6common's rules (6 of 9 clean repeats, the
scorer's G1 with slipped repeats, the host rule, the counter-page clock rule on the PUBLISHED
pages, the cluster-robust gap rule, Q1-C on D)."""
from __future__ import annotations

import json
import math
import random
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
R6DIR = REPO / "scripts" / "scoring" / "rental6"
for p in (REPO, REPO / "scripts", R6DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r6common as C6  # noqa: E402

# rental 5 has scorers of the same file names (score_skew, score_secondk): load rental 6's under
# their own module names so one pytest worker can import both
SQ = C6._load("rental6_score_q1", R6DIR / "score_q1.py")
SKK = C6._load("rental6_score_secondk", R6DIR / "score_secondk.py")
SS = C6._load("rental6_score_skew", R6DIR / "score_skew.py")

from moe.spec import MODEL_CONFIGS  # noqa: E402

CARD = C6.CARD
REGS = {p: C6.registration(REPO, p) for p in C6.PARTS}
PRED = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r
        for r in REGS["skew"]["predictions"] + REGS["q1"]["predictions"]}


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "docs" / "registered").mkdir(parents=True)
    add = C6.C4.CM2.ADDENDUM
    shutil.copy(REPO / "docs" / "registered" / f"{add}.json", r / "docs" / "registered")
    for p in C6.PARTS:
        shutil.copy(REPO / "docs" / "registered" / f"{C6.NAMES[p]}.json", r / "docs" / "registered")
    return r


# --------------------------------------------------------------------------
# r6common's rules
# --------------------------------------------------------------------------

def _row(**kw):
    r = {"ms_p50": 1.0, "clean_repeats": 9, "repeats": 9, "host_bound": False, "sm_clock_load_mhz": 1710.0}
    r.update(kw)
    return r


def test_a_cell_needs_six_clean_repeats_and_the_host_and_clock_rules():
    assert C6.row_verdict(_row(), 1.0) is None
    assert C6.row_verdict(_row(clean_repeats=6), 1.0) is None
    assert "5 clean repeats" in C6.row_verdict(_row(clean_repeats=5), 1.0)
    assert "host-bound by rule" in C6.row_verdict(_row(), 0.30)          # 0.30 + 0.0676 < 0.40
    assert C6.row_verdict(_row(), 0.34) is None                          # 0.4076 >= 0.40
    assert "timer flag" in C6.row_verdict(_row(host_bound=True), 1.0)
    assert "G4" in C6.row_verdict(_row(sm_clock_load_mhz=1700.0), 1.0)   # [1695, 1705): the cell goes
    assert C6.row_verdict(_row(sm_clock_load_mhz=1705.0), 1.0) is None
    # a page from before rental 6 (no slip fields): its usable repeats are its clean ones
    assert C6.clean_repeats({"repeats": 7}) == 7


def test_the_scorers_g1_takes_r3s_g1_and_at_most_ten_percent_slipped_rows():
    rep = {"histogram_gates": {"G1_lock_thermal": {"verdict": "PASS"}}, "repeat_rows": 100, "lock_slip_rows": 10,
           "slipped_repeats": [{"histogram": "PT", "n": 3, "arm": "native", "repeat": 2, "mhz": 1680.0}]}
    ok, why = C6.page_g1(rep)
    assert ok and "10 of 100" in why
    assert not C6.page_g1(dict(rep, lock_slip_rows=11))[0]
    assert not C6.page_g1(dict(rep, histogram_gates={"G1_lock_thermal": {"verdict": "FAIL"}}))[0]
    sl = C6.page_slips(rep)
    assert sl["slipped"][0]["mhz"] == 1680.0 and sl["slipped"][0]["cell"] == "PT/native/n3"


def test_the_counter_clock_rule_passes_held_pages_and_fails_the_8x22b_floor():
    import lock_gate as LG
    tot = ok = 0
    off = []
    for p in LG.published_lock_pages(LG.PUBLISHED):
        page = json.loads(p.read_text())
        if LG.page_lock(page) != 1710:
            continue
        if not LG.held_by_smi(page):
            off.append(C6.clock_rule(page, p))
            continue
        if LG.page_kind(page) != "page":
            continue
        cs = [c for c in C6.clock_rule(page, p)["cells"] if c["mhz"] is not None]
        tot += len(cs)
        ok += sum(c["ok"] for c in cs)
    assert tot > 3000 and 0.995 <= ok / tot <= 0.999, (ok, tot)        # 3025 of 3033, 99.74%
    assert len(off) == 1 and not off[0]["page_ok"] and off[0]["cells_ok"] == 0


def test_the_counter_clock_rule_drops_every_cell_of_the_under_lock_floor_pages():
    """The under-lock negative control: the published floor1005 and floor1410 pages pass
    check_page at their own lock, and the band [1640, 1710] MHz takes none of their cells."""
    pub = REPO / "results" / "published"
    pages = sorted(pub.glob("*/results/*-floor1005-r3-counters/r3f-g64-lock1005.json")) + \
        sorted(pub.glob("*/results/*-floor1410-r3-counters/r3f-g64-lock1410.json"))
    assert len(pages) == 7, pages
    clocked = 0
    for p in pages:
        r = C6.clock_rule(json.loads(p.read_text()), p)
        assert r["page_ok"], (p.name, r["why"])          # the lock in force is the page's own
        assert r["cells"] and r["cells_ok"] == 0, p
        mhz = [c["mhz"] for c in r["cells"] if c["mhz"] is not None]
        assert all(m < C6.BAND_MHZ[0] - 200 for m in mhz), (p, mhz)
        clocked += len(mhz)
    assert clocked >= 50
    reg = REGS["skew"]["counter_clock_rule"]
    assert "floor1005" in reg["under_lock_control"] and "NO UNDER-LOCK" not in json.dumps(reg)


def test_the_cluster_robust_se_and_the_gap_rule():
    g = [1.0, 1.2, -0.8, -1.0]
    m, se, C = C6.cluster_mean_se(g, ["a", "a", "b", "b"])
    assert m == pytest.approx(0.1) and C == 2
    assert se == pytest.approx(math.sqrt(2 * ((1.0 + 1.2 - 0.2) ** 2 + (-0.8 - 1.0 - 0.2) ** 2)) / 4)
    rows = [{"g": x, "cluster": c, "copies": cp, "q1page": q, "g_R": {"v2": 4.6}}
            for x, c, cp, q in [(0.1, "a", 9, True), (0.2, "a", 15, True), (-0.1, "b", 9, True),
                                (0.0, "b", 15, True), (0.3, "c", 9, False), (-0.2, "c", 9, False)]]
    out = C6.q1_rule(rows, ("v2",))
    assert out["verdict"] == "HOLDS" and out["rivals"]["v2"]["verdict"] == "EXCLUDED"
    assert C6.q1_rule(rows[:5])["verdict"].startswith("UNDECIDED")
    no15 = [dict(r, copies=9) for r in rows]
    assert C6.q1_rule(no15)["verdict"].startswith("UNDECIDED: the copies clause")
    shifted = [dict(r, g=r["g"] + (2.0 if r["copies"] == 15 else 0.0)) for r in rows]
    assert C6.q1_rule(shifted)["verdict"] == "FAILS"


def test_q1c_reads_d_against_positions_set_by_the_measured_x9():
    r = 66272 / 39824
    assert C6.q1c_rule(-6.6, -6.6 * r, r, 0.64)["verdict"] == "SELECTED DEAD"
    assert C6.q1c_rule(-6.6, -6.4, r, 0.64)["verdict"] == "SELECTED CALL"
    assert C6.q1c_rule(-6.6, -8.8, r, 2.0)["verdict"].startswith("NOT SEPARATED")
    assert C6.q1c_rule(None, -6.6, r, 0.64)["verdict"].startswith("NOT SCORED")


# --------------------------------------------------------------------------
# synthetic pages
# --------------------------------------------------------------------------

def _hist(name):
    files = {**REGS["skew"]["histograms"]["files"], **REGS["q1"]["pages"]["files"]}
    return files[name], json.loads((REPO / files[name]["path"]).read_text())


def write_timed(tree: Path, model, label, page, name, ms_of, *, copies=9, probe=True, slips=None, g1="PASS",
                clean=None):
    reg_file, doc = _hist(name)
    E = MODEL_CONFIGS[model].num_experts
    rows, slipped = [], []
    for c in doc["cells"]:
        for arm in c.get("arms") or ("native", "shared"):
            ms = ms_of(model, page, c["n"], c["label"], arm)
            if ms is None:
                continue
            k = (c["label"], c["n"], arm)
            cr = (clean or {}).get(k, 9)
            ns = 9 - cr
            rows.append({"arm": arm, "tiles": c["n"], "ms_p50": ms, "sm_clock_load_mhz": 1710.0,
                         "experts_declared": E if arm == "native" else E * copies, "repeats": cr, "clean_repeats": cr,
                         "lock_slip_repeats": ns, "repeat_rows": 9, "lost": cr < 6, "host_bound": False,
                         "histogram": c["label"], "counts_sha256": next(x["counts_sha256"] for x in reg_file["cells"]
                                                                        if x["label"] == c["label"] and x["n"] == c["n"]),
                         "bincount_ok": True, "shuffle_seed": None if c["counts"] is None else doc["shuffle_seed"]})
            slipped += [{"histogram": c["label"], "n": c["n"], "arm": arm, "repeat": i, "mhz": 1680.0} for i in range(ns)]
    total = sum(r["repeat_rows"] for r in rows)
    nslip = sum(r["lock_slip_repeats"] for r in rows) if slips is None else slips
    probe_cells = []
    for n in sorted({c["n"] for c in doc["cells"]}):
        a = PRED.get((model, page, n, "balanced", "shared"), {}).get("A_proxy_us") or 0.0
        probe_cells += [{"label": "native", "tread": n, "ms": 0.005}, {"label": "shared", "tread": n, "ms": 0.005 + a * 1e-3}]
    d = tree / "results" / f"gaps-{CARD}-{model}-{label}" / "private_weight_reference" / f"run-{label}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps({
        "experiment": "private_weight_reference", "kind": "histogram-page", "session_tag": f"t-{model}-{label}",
        "pinned": {"GROUP_SIZE_M": 8}, "copies_declared": copies,
        "histogram": {"file_sha256": reg_file["sha256"], "shuffle_seed": reg_file["shuffle_seed"]},
        "histogram_gates": {"G1_lock_thermal": {"verdict": g1}}, "gates": [], "treads_table": rows,
        "align_probe": {"cells": probe_cells} if probe else None, "slip_policy": "cell", "lock_mhz": 1710,
        "min_clean_repeats": 6, "repeat_rows": total, "lock_slip_rows": nslip, "slipped_repeats": slipped}))


def ms_v3(noise=0.0003, seed=1, variant="v3", arms=("native", "shared")):
    rng = random.Random(seed)

    def f(m, p, n, lab, arm):
        r = PRED[(m, p, n, lab, arm)]
        v = variant if arm in arms else "v3"
        return r["ms"][v] * (1 + rng.gauss(0, noise))
    return f


def all_timed(tree, ms_of=None, per=None):
    ms_of = ms_of or ms_v3()
    kw = per or {}
    for m in SS.RANK:
        for page in SS.PAGES:
            write_timed(tree, m, f"sk{page.lower()}", page, f"{m}-{page}.json", ms_of, **kw.get((m, page), {}))
    for m in REGS["q1"]["pages"]["models"]:
        for c in (9, 15):
            write_timed(tree, m, f"q{c}", f"c{c}", f"q1-{m}.json", ms_of, copies=c)


def write_bytes(tree, model, *, factor=1.0, mhz=1700.0, smi=1710.0, rowkey=False):
    import r3_timing_model as TM
    old = TM.set_model(model)
    bm = dict(TM.BYTE_MODEL)
    TM.set_model(old)
    reg_file, doc = _hist(f"{model}-bytes.json")
    cells = []
    for r in REGS["skew"]["byte_predictions"]:
        if r["model"] != model:
            continue
        q = dict(r["q_pred"])
        if rowkey and r.get("rowkey_w1_q"):
            q["w1"] = r["rowkey_w1_q"]["C39.5"]
        ns = 1.0e6
        cells.append({"arm": r["arm"], "n": r["n"], "histogram": r["label"], "counts_sha256": r["counts_sha256"],
                      "recorded": {g: {"sm__cycles_elapsed.avg": mhz * ns / 1e3} for g in ("w1", "w2")},
                      "per_gemm": {g: {"dram_bytes_read": q[g] * factor * bm[f"W_{g}"] + r["n"] * bm[f"operand_per_tile_{g}"],
                                       "gpu_time_ns": ns} for g in ("w1", "w2")}})
    d = tree / "results" / f"2026-10-10-{CARD}-{model}-skbytes-r3-counters" / "lock1710"
    d.mkdir(parents=True, exist_ok=True)
    smi_rec = {"rows": [{"clocks.sm": smi}]}
    (d / "r3c-g8.json").write_text(json.dumps({
        "design": {"model": model, "histogram": {"file_sha256": reg_file["sha256"], "shuffle_seed": reg_file["shuffle_seed"]}},
        "ncu": {"lock_mhz": 1710, "smi_before": smi_rec, "smi_after": smi_rec}, "cells": cells, "gates": []}))


def test_pages_that_follow_v3_hold_every_test_and_no_page_scores_not_scored(tmp_path, repo):
    all_timed(tmp_path)
    for m in SS.BYTE_MODELS:
        write_bytes(tmp_path, m)
    res = SS.score(repo, tmp_path)
    assert res["G3"]["verdict"] == "PASS", res["G3"]
    assert res["E1-SKEW"]["verdict"] == "HOLDS" and res["E1-UNI"]["verdict"] == "HOLDS"
    assert res["SKEW-RATIO"]["v3"]["verdict"] == "HOLDS"
    assert res["E1-rivals"]["LT3"]["verdict"] == "EXCLUDED"
    assert res["B-SKEW"]["verdict"] == "HOLDS" and res["B-CA"]["v3"] == "HOLDS"
    assert res["OPEN-L2SKEW"]["status"] == "OPEN" and res["OPEN-L2SKEW"]["v3"] == "HOLDS"
    assert res["E2-SKEW"]["verdict"] == "HOLDS", res["E2-SKEW"]
    empty = SS.score(repo, tmp_path / "none")
    assert empty["E1-SKEW"]["verdict"].startswith("NOT SCORED") and empty["B-SKEW"]["verdict"].startswith("NOT SCORED")


def test_a_g3_failure_voids_every_histogram_cell_and_prints_e1_descriptively(tmp_path, repo):
    """Owner decision 2026-10-09: G3 FAILS with Q not rejecting (every uniform cell 1% slow, a
    homogeneous token-order effect): E1-SKEW is NOT SCORED, and E1 on the voided cells is
    printed beside it, labelled DESCRIPTIVE, NOT A VERDICT."""
    base = ms_v3()

    def slow_uniform(m, p, n, lab, arm):
        return base(m, p, n, lab, arm) * (1.01 if lab == "uniform" else 1.0)
    all_timed(tmp_path, ms_of=slow_uniform)
    res = SS.score(repo, tmp_path)
    g3 = res["G3"]
    assert g3["void_strata"] == "ALL" and not g3["Q"].get("heterogeneous"), g3
    e1 = res["E1-SKEW"]
    assert e1["verdict"].startswith("NOT SCORED: G3 FAILS"), e1
    assert "stats" not in e1, "no verdict statistic outside the descriptive block"
    d = e1["descriptive"]
    assert d["label"].startswith("DESCRIPTIVE, NOT A VERDICT")
    hist = [k for k, r in PRED.items() if r["test"] == "E1" and r["label"] != "balanced"]
    assert d["stats"]["cells"] == len(hist)
    # the PT / PW / PW2 / DW cells follow v3 (e near 0), the uniform ones read about -1%
    assert d["reads_as"] == "would hold" and -0.006 < d["stats"]["mean"] < -0.001, d
    assert res["E1-rivals"]["LT3"]["verdict"].startswith("UNDECIDED")
    txt = "\n".join(SS.lines(res))
    assert "E1-SKEW: NOT SCORED: G3 FAILS" in txt and "E1 DESCRIPTIVE, NOT A VERDICT" in txt
    # a passing G3 prints no descriptive block
    ok = tmp_path / "ok"
    all_timed(ok)
    assert "descriptive" not in SS.score(repo, ok)["E1-SKEW"]


def test_slips_drop_a_cell_under_six_clean_and_a_page_over_ten_percent_leaves_clean(tmp_path, repo):
    lost = {("PT", 3, "native"): 5}
    many = {(lab, n, a): 7 for lab in ("PT", "PW", "uniform", "balanced") for n in (3, 6, 12) for a in ("native", "shared")}
    all_timed(tmp_path, per={("mixtral-8x7b", "A"): {"clean": lost}, ("olmoe-1b-7b", "A"): {"clean": many}})
    res = SS.score(repo, tmp_path)
    ad = res["addendum"]
    allv = ad["pages"]["olmoe-1b-7b-ska"]["ALL"]
    cln = ad["pages"]["olmoe-1b-7b-ska"]["CLEAN"]
    assert allv["use"] == "counted" and cln["use"].startswith("excluded (G1"), (allv, cln)
    dropped = [d for d in res["dropped_rows"]["mixtral-8x7b-A"] if "5 clean repeats" in d[1]]
    assert dropped and dropped[0][0] == "mixtral-8x7b/A/3/PT/native"
    sl = res["slips"]["olmoe-1b-7b-A"]
    assert (sl["slipped_rows"], sl["repeat_rows"]) == (48, 216) and all(x["mhz"] == 1680.0 for x in sl["slipped"])


def test_a_skewed_olmoe_w1_miss_stays_open_and_the_rowkey_rival_reads_better(tmp_path, repo):
    all_timed(tmp_path)
    write_bytes(tmp_path, "olmoe-1b-7b", rowkey=True)
    write_bytes(tmp_path, "phi-3.5-moe")
    res = SS.score(repo, tmp_path)
    l2 = res["OPEN-L2SKEW"]
    assert l2["v3"] == "FAILS" and l2["ROWKEY"]["verdict"] == "BETTER", l2
    assert res["B-SKEW"]["verdict"] == "HOLDS", "the open rows do not decide the pooled verdict"


def test_the_counter_clock_rule_drops_a_page_off_the_lock_and_a_cell_out_of_band(tmp_path, repo):
    all_timed(tmp_path)
    write_bytes(tmp_path, "olmoe-1b-7b", smi=1980.0)
    write_bytes(tmp_path, "phi-3.5-moe", mhz=1630.0)
    res = SS.score(repo, tmp_path)
    assert res["byte_pages"]["olmoe-1b-7b"].startswith("NOT SCORED (clock rule)")
    assert res["byte_pages"]["phi-3.5-moe-dropped"] and res["B-SKEW"]["verdict"].startswith("NOT SCORED")


def test_a_page_without_a_probe_scores_e1_on_the_proxy_and_flags_it(tmp_path, repo):
    all_timed(tmp_path, per={("phi-3.5-moe", "B"): {"probe": False}})
    res = SS.score(repo, tmp_path)
    assert res["E1-SKEW"]["no_probe_pages"] == ["phi-3.5-moe-B"] and res["E1-SKEW"]["verdict"] == "HOLDS"


def test_q1_holds_on_v3_excludes_v2_and_h8_and_reads_q1c(tmp_path, repo):
    r = REGS["q1"]["tests"]["Q1-C"]["dead_ratio"]
    base = ms_v3(noise=0.0)

    def ms(m, p, n, lab, arm):
        v = base(m, p, n, lab, arm)
        if m == "qwen1.5-moe-a2.7b" and n == 1 and arm == "shared":
            v += (-6.6 if p == "c9" else -6.6 * r) * 1e-3
        return v
    all_timed(tmp_path, ms)
    res = SQ.score(repo, tmp_path)
    h = res["Q1-H"]
    assert h["verdict"] == "HOLDS" and abs(h["mean_g_us"]) < 1e-6, h
    assert h["rivals"]["v2"]["verdict"] == "EXCLUDED" and h["rivals"]["H8"]["verdict"] == "EXCLUDED"
    assert h["rivals"]["v3_all"]["verdict"] == "NOT EXCLUDED"
    assert res["Q1-P"]["verdict"].startswith("DESCRIPTIVE")
    assert res["Q1-C"]["verdict"] == "SELECTED DEAD", res["Q1-C"]


def test_q1_fails_when_the_shared_gap_is_v2s(tmp_path, repo):
    all_timed(tmp_path, ms_v3(noise=0.0, variant="v2", arms=("shared",)))
    res = SQ.score(repo, tmp_path)
    assert res["Q1-H"]["verdict"] == "FAILS" and res["Q1-H"]["mean_g_us"] == pytest.approx(
        REGS["q1"]["power"]["Q1-H | replicate sigma | v3 true"]["g_R v2 us"], abs=0.05)


def test_q1_not_scored_without_pages_and_a_probe_less_page_skips_its_pairs(tmp_path, repo):
    assert SQ.score(repo, tmp_path / "none")["Q1-H"]["verdict"].startswith("UNDECIDED")
    all_timed(tmp_path, per={("olmoe-1b-7b", "A"): {"probe": False}})
    res = SQ.score(repo, tmp_path)
    assert any("no align_probe" in s for s in res["Q1-H"]["skipped"])


# --------------------------------------------------------------------------
# second K
# --------------------------------------------------------------------------

TP4 = MODEL_CONFIGS["mixtral-8x7b-tp4"]


def sk_page(bk, c, *, mhz=1700.0, v7=False, design_bk=None):
    E = TP4.num_experts
    K = {"w1": TP4.hidden_size, "w2": TP4.intermediate_size}
    npn = {"w1": 2 * TP4.intermediate_size // 64, "w2": TP4.hidden_size // 64}
    cells = []
    for arm, D in (("native", E), ("shared", 9 * E)):
        for n in range(1, 10):
            pg, rec = {}, {}
            for g in ("w1", "w2"):
                R = math.ceil((E * n * 32 + D * 31) / 32)
                live = E * n * npn[g]
                cyc = (5000 if arm == "native" else 9000) + math.ceil(live / 132) * (K[g] // bk * c[g] + C6.C4.F_CTA[g])
                ns = cyc * 1e3 / mhz
                pg[g] = {"grid_size": R * npn[g], "gpu_time_ns": ns, "dram_bytes_read": 0.2 * 4022.0 * ns}
                rec[g] = {"launch__occupancy_limit_registers": 5, "launch__occupancy_limit_warps": 8,
                          "sm__cycles_elapsed.avg": cyc}
            cells.append({"arm": arm, "n": n, "declared": D, "per_gemm": pg, "recorded": rec})
    smi = {"rows": [{"clocks.sm": 1710.0}]}
    return {"family": "r3-arms", "design": {"model": "mixtral-8x7b-tp4", "block_m": 32, "block_n": 64,
                                            "block_k": design_bk or bk, "num_stages": 4, "group_m": 64},
            "ncu": {"lock_mhz": 1710, "smi_before": smi, "smi_after": smi},
            "card": {"slug": CARD, "sm_count": 132}, "cells": cells,
            "gates": [{"number": "V7", "verdict": "FAIL"}] if v7 else []}


def put_bytes(tree, label, page):
    d = tree / f"2026-10-10-{CARD}-mixtral-8x7b-tp4-{label}-r3-counters" / "lock1710"
    d.mkdir(parents=True, exist_ok=True)
    (d / "r3c-g64.json").write_text(json.dumps(page))


def c32_for(h):
    import reg6 as R6
    out = {}
    for g in ("w1", "w2"):
        p = REGS["secondk"]["predicted_ratio"][g]
        u64 = p["S64"] * R6.C64[g] + C6.C4.F_CTA[g]
        out[g] = (p[h] * u64 - C6.C4.F_CTA[g]) / p["S32"]
    return out


@pytest.mark.parametrize("h", ["H_ITER", "H_CTA"])
def test_secondk_selects_the_hypothesis_its_pages_were_built_under(tmp_path, repo, h):
    import reg6 as R6
    put_bytes(tmp_path, "k64s4", sk_page(64, R6.C64))
    put_bytes(tmp_path, "k32s4", sk_page(32, c32_for(h)))
    res = SKK.score(repo, tmp_path)
    assert res["verdict"] == f"SELECTED {h}", res
    # a V7 failure reads the NATIVE cells only, and says so
    put_bytes(tmp_path, "k32s4", sk_page(32, c32_for(h), v7=True))
    res = SKK.score(repo, tmp_path)
    assert res["verdict"] == f"SELECTED {h} (V7: native only)", res["verdict"]


def test_secondk_needs_both_pages_in_band_and_clocks_within_one_percent(tmp_path, repo):
    import reg6 as R6
    put_bytes(tmp_path, "k64s4", sk_page(64, R6.C64))
    put_bytes(tmp_path, "k32s4", sk_page(32, c32_for("H_ITER"), mhz=1660.0))
    assert SKK.score(repo, tmp_path)["verdict"].startswith("NOT SCORED: the two pages' clocks")
    put_bytes(tmp_path, "k32s4", sk_page(32, c32_for("H_ITER"), mhz=1630.0))
    res = SKK.score(repo, tmp_path)
    assert res["verdict"].startswith("NOT SCORED")
    assert SKK.score(repo, tmp_path / "none")["verdict"].startswith("NOT SCORED")
