"""The tile stage's scorers on SYNTHETIC pages only (no tile-stage page exists):
scripts/scoring/tileA/score_{census,floor,bytes,rep,e1cal,dead,tileB}.py and finalise.py against
docs/registered/2026-10-14-tileA-* and tileB-*, and tcommon's rules (the joint c and F estimator,
the accounting and F-form selections, the A key, the stop rule, E1-TILE with the per-model clause and
the CLEAN certification, the rivals, the dead lever, the replicate)."""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TDIR = REPO / "scripts" / "scoring" / "tileA"
for p in (REPO, REPO / "scripts", TDIR, REPO / "scripts" / "scoring" / "rental6", REPO / "scripts" / "scoring" / "rental7"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402
import tcommon as TC  # noqa: E402

#: the scorers under names of their own: rental 1 and rental 4 have a score_floor.py and a score_dead.py
FN = TC._load("finalise", TDIR / "finalise.py")
RG = TC._load("regtile", TDIR / "regtile.py")
SB, SC, SD, SE, SF, SR, STB = (TC._load(f"tile_{n}", TDIR / f"{n}.py") for n in
                               ("score_bytes", "score_census", "score_dead", "score_e1cal", "score_floor", "score_rep",
                                "score_tileB"))

from moe.spec import MODEL_CONFIGS  # noqa: E402

CARD = TC.CARD
REGS = {p: TC.registration(REPO, p) for p in TC.PARTS}
SMI = {"rows": [{"clocks.sm": 1710.0}]}
NOMINAL = {"c": 900.0, "F": {"w1": 520.0, "w2": 979.0}}


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "docs" / "registered").mkdir(parents=True)
    add = TC.C4.CM2.ADDENDUM
    shutil.copy(REPO / "docs" / "registered" / f"{add}.json", r / "docs" / "registered")
    for p in TC.PARTS:
        shutil.copy(REPO / "docs" / "registered" / f"{TC.NAMES[p]}.json", r / "docs" / "registered")
    return r


def _tile_of(bm, warps=8):
    return {128: V4.CAL_TILE, 64: V4.T64W8 if warps == 8 else V4.T64W4, 32: V4.T32W4}[bm]


# --------------------------------------------------------------------------
# synthetic counter pages
# --------------------------------------------------------------------------

def write_counter(tree, model, label, G, tile, *, c=None, F=None, reads=None, hist=None, ns=range(1, 10), occ=None,
                  seed=5, noise=0.0, wrong_tile=False, mhz=1700.0):
    """A G-`G` counter page at `tile`: floor cycles = 5000 + ceil(N_live / 132) (S c + F_g) per GEMM
    (when c given), each GEMM's DRAM rate a tenth of the peak, the clock in band; `reads(cell)` the
    per-GEMM read bytes of a histogram cell (byte pages)."""
    rng = random.Random(seed)
    cfg = MODEL_CONFIGS[model]
    E, bm = cfg.num_experts, tile.bm
    cells = []
    hc = None
    if hist is not None:
        reg_file, doc = hist
        hc = [(x["label"], int(x["n"]), arm) for x in doc["cells"] for arm in x["arms"]]
    keys = hc or [(None, n, arm) for n in ns for arm in ("native", "shared")]
    for lab, n, arm in keys:
        D = E if arm == "native" else E * 9
        R = math.ceil((E * n * bm + D * (bm - 1)) / bm)
        rec, pg = {}, {}
        for g in ("w1", "w2"):
            N = 2 * cfg.intermediate_size if g == "w1" else cfg.hidden_size
            K = cfg.hidden_size if g == "w1" else cfg.intermediate_size
            npn = -(-N // tile.bn)
            live = E * n * npn
            S = K // tile.bk
            cyc = 5000.0 + math.ceil(live / 132) * (S * c + F[g]) if c is not None else 4.0e6
            cyc *= 1 + rng.gauss(0, noise)
            ns_ = cyc / (mhz / 1e3)
            b = reads(lab, n, arm, g) if reads else 0.1 * 4022.0 * ns_
            rec[g] = {"sm__cycles_elapsed.avg": cyc, "launch__occupancy_limit_shared_mem": occ or tile.occ_shared()}
            pg[g] = {"grid_size": R * npn, "dram_bytes_read": b, "gpu_time_ns": ns_}
        cell = {"arm": arm, "n": n, "declared": D, "recorded": rec, "per_gemm": pg}
        if hist is not None:
            cell["histogram"] = lab
            cell["counts_sha256"] = next(x["counts_sha256"] for x in reg_file["cells"] if x["label"] == lab and x["n"] == n)
        cells.append(cell)
    de = {"model": model, "block_m": bm if not wrong_tile else 32, "block_n": tile.bn, "block_k": tile.bk,
          "num_warps": tile.warps, "num_stages": tile.stages}
    if hist is not None:
        de["histogram"] = {"file_sha256": hist[0]["sha256"], "shuffle_seed": hist[0]["shuffle_seed"]}
    d = tree / "results" / f"2026-10-14-{CARD}-{model}-{label}-r3-counters" / "lock1710"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"r3c-g{G}.json").write_text(json.dumps({"design": de, "ncu": {"lock_mhz": 1710, "smi_before": SMI, "smi_after": SMI},
                                                  "cells": cells, "gates": []}))


def floor_pages(tree, *, c64=None, F64=None, cal=NOMINAL):
    for m in ("olmoe-1b-7b", "mixtral-8x7b"):
        write_counter(tree, m, "f128", 64, V4.CAL_TILE, c=cal["c"], F=cal["F"])
        if c64 is not None:
            write_counter(tree, m, "f64w8", 64, V4.T64W8, c=c64, F=F64)


def w_variant_64(cal=NOMINAL, variant="W", form="F-step"):
    prm = V4.Params(c128=cal["c"], F_wg=dict(cal["F"]), variant=variant, f_form=form)
    c64 = V4.c_counter(V4.T64W8, "w1", prm)
    return c64, {g: V4.phi(V4.T64W8, g, prm) * c64 for g in ("w1", "w2")}


# --------------------------------------------------------------------------
# tA-floor
# --------------------------------------------------------------------------

def test_the_joint_estimator_recovers_c_and_both_f_from_two_depths(tmp_path):
    floor_pages(tmp_path)
    pages = SF.pages_of(tmp_path, "f128")[0]
    fit = TC.joint_cF(pages)
    assert fit["c"] == pytest.approx(900.0, rel=1e-9) and fit["F"]["w1"] == pytest.approx(520.0, abs=1e-6)
    assert fit["F"]["w2"] == pytest.approx(979.0, abs=1e-6) and fit["models"] == ["mixtral-8x7b", "olmoe-1b-7b"]
    assert TC.joint_cF(pages[:1]) is None          # one model: c and the two F are not separable


@pytest.mark.parametrize("truth", ["W", "T"])
def test_the_floor_selects_the_accounting_the_64x128_pages_follow_and_f_step(tmp_path, repo, truth):
    c64, F64 = w_variant_64(variant=truth)
    floor_pages(tmp_path, c64=c64, F64=F64)
    res = SF.score(repo, tmp_path, tmp_path / "out")
    assert res["calibration"]["verdict"] == "CAL-A" and res["calibration"]["c"] == pytest.approx(900.0)
    assert res["selection"]["verdict"] == f"SELECTED {truth}", res["FLOOR-TEST (DESCRIPTIVE)"]
    assert res["f_form"]["selected"] == "F-step"
    assert res["FLOOR-TEST (DESCRIPTIVE)"]["B"]["verdict"] == "FAILS"


def test_f_cyc_replaces_f_step_only_when_the_pages_say_so(tmp_path, repo):
    c64, F64 = w_variant_64(form="F-cyc")
    for m in ("olmoe-1b-7b", "mixtral-8x7b"):
        write_counter(tmp_path, m, "f128", 64, V4.CAL_TILE, c=900.0, F=NOMINAL["F"])
        write_counter(tmp_path, m, "f64w8", 64, V4.T64W8, c=c64, F=F64, noise=0.0002, seed=hash(m) % 97)
    res = SF.score(repo, tmp_path, tmp_path / "out")
    assert res["f_form"]["selected"] == "F-cyc", res["f_form"]


def test_no_accounting_holding_takes_v4cal_and_no_calibration_stops_everything(tmp_path, repo):
    c64, F64 = w_variant_64()
    floor_pages(tmp_path, c64=c64 * 0.85, F64=F64)          # under every accounting
    res = SF.score(repo, tmp_path, tmp_path / "out")
    assert res["selection"]["selected"] is None and "v4cal" in res["selection"]["v4_at_64x128"]
    empty = SF.score(repo, tmp_path / "none", tmp_path / "out2")
    assert empty["calibration"]["verdict"].startswith("NOT SCORED")


def test_a_page_at_another_tile_or_off_the_lock_is_not_scored(tmp_path, repo):
    write_counter(tmp_path, "olmoe-1b-7b", "f128", 64, V4.CAL_TILE, c=900.0, F=NOMINAL["F"], wrong_tile=True)
    write_counter(tmp_path, "mixtral-8x7b", "f128", 64, V4.CAL_TILE, c=900.0, F=NOMINAL["F"], mhz=1500.0)
    why = SF.pages_of(tmp_path, "f128")[1]
    assert why["olmoe-1b-7b"].startswith("NOT SCORED: the page ran tile")
    assert why["mixtral-8x7b"] != "counted"


# --------------------------------------------------------------------------
# tA-census
# --------------------------------------------------------------------------

def _census(tree, entries, mismatch=()):
    d = tree / "results" / f"2026-10-14-{CARD}-tile-census-mixtral-8x7b-ta"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tile-census.json").write_text(json.dumps({"entries": entries, "path_mismatch": list(mismatch), "errors": [],
                                                    "verdict": "COMPLETE"}))


def test_the_census_score_carries_common_occupancies_and_refuses_a_path_mismatch(tmp_path, repo):
    e = [{"model": m, "tile": "128x128x64w8s3", "gemm": g, "occupancy": {"ctas_per_sm": 2},
          "ptx_counts": {"path": "wgmma", "instruction": "m64n128k16", "per_warpgroup": 4, "warpgroups": 2}}
         for m in ("olmoe-1b-7b", "phi-3.5-moe") for g in ("w1", "w2")]
    e += [{"model": "olmoe-1b-7b", "tile": "64x128x64w8s3", "gemm": g, "occupancy": {"ctas_per_sm": 2}, "ptx_counts": {}}
          for g in ("w1", "w2")]
    _census(tmp_path, e)
    res = SC.score(repo, tmp_path)
    assert res["CENSUS"]["verdict"].startswith("VALID") and res["occupancy"]["128x128x64w8s3"] == {"w1": 2, "w2": 2}
    assert res["differs_from_design"] == {"64x128x64w8s3": {"design": 3, "census": {"w1": 2, "w2": 2}}}
    assert res["mechanical_steps"] and res["counts"]["128x128x64w8s3"]["instruction"] == "m64n128k16"
    bad = tmp_path / "bad"
    _census(bad, e, ["x 128x128x64w8s3 w1: compiled mma.sync"])
    r2 = SC.score(repo, bad)
    assert r2["CENSUS"]["verdict"].startswith("REFUSED") and r2["occupancy"] == {}
    assert SC.score(repo, tmp_path / "none")["CENSUS"]["verdict"].startswith("NOT SCORED")


# --------------------------------------------------------------------------
# tA-bytes
# --------------------------------------------------------------------------

def _byte_reads(truth, name, model, bm, noise=0.003, seed=3, mid=None):
    rng = random.Random(seed)
    with V4.at_tile(model, _tile_of(bm)):
        bmod = dict(TM.BYTE_MODEL)
    pred = {(r["n"], r["label"], r["arm"]): r for r in REGS["bytes"]["byte_predictions"] if r["page"] == name}

    def f(lab, n, arm, g):
        q = pred[(n, lab, arm)]["q_pred"]
        if mid:
            a, b = mid
            v = 0.5 * (q[a][g] + q[b][g]) if (q.get(a) and q.get(b) and g in q[a] and g in q[b]) else q["v3"][g]
        else:
            v = q[truth][g] if (q.get(truth) and g in q[truth]) else q["v3"][g]
        return v * (1 + rng.gauss(0, noise)) * bmod[f"W_{g}"] + n * bmod[f"operand_per_tile_{g}"]
    return f


def byte_pages(tree, truth="CLAMP", **kw):
    files = REGS["bytes"]["pages"]["files"]
    for label, model, name in SB.UNITS:
        reg_file = files[f"{name}.json"]
        doc = json.loads((REPO / reg_file["path"]).read_text())
        write_counter(tree, model, label, reg_file["group_m"], _tile_of(reg_file["tile"][0]), hist=(reg_file, doc),
                      reads=_byte_reads(truth, name, model, reg_file["tile"][0], **kw))


@pytest.mark.parametrize("truth", ["v3", "CLAMP", "ROWKEY60", "ROWKEY30"])
def test_the_byte_pages_name_the_key_they_follow(tmp_path, repo, truth):
    byte_pages(tmp_path, truth)
    res = SB.score(repo, tmp_path)
    assert all(v == "counted" for v in res["pages"].values()), res["pages"]
    assert res["KEY"]["verdict"] == f"NAMED {truth}", res["KEY"]
    assert res["KEY"]["key"] == truth and res["KEY"]["rows"] == len(res["rows"])
    assert {r["label"] for r in res["rows"]} == {"PT", "PW", "uniform", "balanced"}       # every label scored
    assert res["printed_G16_rows"] and all(r["page"] == "mixtral-8x7b-128g16" for r in res["printed_G16_rows"])


def test_nothing_named_falls_back_to_clamp_not_v3(tmp_path, repo):
    byte_pages(tmp_path, mid=("v3", "ROWKEY60"), noise=0.0)
    res = SB.score(repo, tmp_path)
    assert res["KEY"]["verdict"].startswith("NOT NAMED") and res["KEY"]["key"] == "CLAMP"
    rows = [{"label": "uniform", "e": {"v3": 0.001, "CLAMP": 0.04, "ROWKEY60": 0.2, "ROWKEY30": 0.3}}] * 3
    rows += [{"label": "uniform", "e": {"v3": 0.06, "CLAMP": 0.06, "ROWKEY60": 0.2, "ROWKEY30": 0.3}}]
    assert TC.named_key(rows)["key"] == "CLAMP"         # the uniform row's 6% stops v3 being NAMED


# --------------------------------------------------------------------------
# synthetic timed pages (R3 at --slip-policy cell --min-clean-repeats 0)
# --------------------------------------------------------------------------

CSV_FIELDS = ("arm", "repeat", "tiles", "ms_p50", "sm_clock_load_mhz", "clock_drift_ok", "status", "detail",
              "clock_samples_mhz", "host_bound")


def write_timed(tree, model, label, reg_file, ms_of, *, copies=9, A_of=None, clean=None):
    doc = json.loads((REPO / reg_file["path"]).read_text())
    E = MODEL_CONFIGS[model].num_experts
    bm, bn, bk, w, s = reg_file["tile"]
    rows, raw = [], []
    for c in doc["cells"]:
        for arm in c["arms"]:
            ms = ms_of(c["n"], c["label"], arm)
            if ms is None:
                continue
            cr = (clean or {}).get((c["label"], c["n"], arm), 9)
            for rep in range(9):
                reads = [1710] * 15
                if rep >= cr:
                    reads[5:8] = [1600] * 3
                raw.append({"arm": arm, "repeat": rep, "tiles": c["n"], "ms_p50": repr(ms), "sm_clock_load_mhz": "1710.0",
                            "clock_drift_ok": "True", "status": "ok", "detail": f"histogram={c['label']}; LEVEL HIGH",
                            "clock_samples_mhz": " ".join(map(str, reads)), "host_bound": "False"})
            rows.append({"repeat_rows": 9, "lock_slip_repeats": 0, "clean_repeats": 9, "lost": False, "arm": arm,
                         "tiles": c["n"], "ms_p50": 99.0, "sm_clock_load_mhz": 1200.0, "repeats": 9, "host_bound": False,
                         "experts_declared": E if arm == "native" else E * copies, "histogram": c["label"],
                         "counts_sha256": next(x["counts_sha256"] for x in reg_file["cells"] if x["label"] == c["label"] and x["n"] == c["n"]),
                         "bincount_ok": True})
    probe = []
    for n in sorted({c["n"] for c in doc["cells"]}):
        a = (A_of or {}).get(n, 0.0)
        probe += [{"label": "native", "tread": n, "ms": 0.005}, {"label": "shared", "tread": n, "ms": 0.005 + a * 1e-3}]
    gates = {"G1_lock_thermal": {"verdict": "PASS"}, "G5_provenance": {"verdict": "PASS", "missing": []}}
    d = tree / "results" / f"gaps-{CARD}-{model}-{label}" / "private_weight_reference" / f"run-{label}"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "cells.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        wr.writeheader()
        wr.writerows(raw)
    (d / "report.json").write_text(json.dumps({
        "experiment": "private_weight_reference", "kind": "histogram-page", "session_tag": f"t-{model}-{label}",
        "block_m": bm, "pinned": {"GROUP_SIZE_M": doc["group_m"], "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "num_warps": w,
                                  "num_stages": s},
        "copies_declared": copies, "histogram": {"file_sha256": reg_file["sha256"], "shuffle_seed": reg_file["shuffle_seed"]},
        "histogram_gates": gates, "treads_table": rows,
        "gates": [{"kind": "VALIDITY", "tag": k, "verdict": v["verdict"]} for k, v in gates.items()],
        "align_probe": {"cells": probe}, "slip_policy": "cell", "lock_mhz": 1710, "min_clean_repeats": 0,
        "repeat_rows": 9 * len(rows), "lock_slip_rows": 0, "slipped_repeats": []}))


def a_scores(out: Path, *, key="CLAMP", variant="W", sigma=None):
    """Synthetic rental-A selections: the score files finalise.selections reads."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "floor.score.json").write_text(json.dumps({"calibration": {"c": NOMINAL["c"], "F": NOMINAL["F"]},
                                                      "selection": {"selected": variant}, "f_form": {"selected": "F-step"}}))
    (out / "bytes.score.json").write_text(json.dumps({"KEY": {"key": key}}))
    (out / "census.score.json").write_text(json.dumps({"CENSUS": {"verdict": "VALID (COMPLETE)"}, "occupancy": {}, "counts": {},
                                                       "occupancy_by_model": {}}))
    if sigma:
        (out / "rep.score.json").write_text(json.dumps({"REP": {"sigma_page": sigma}}))


_PRED: dict = {}


def a_preds(out: Path) -> dict:
    if "A" not in _PRED:
        rows = FN.predict_pages(RG.A_TIMED_PAGES, "A", FN.selections(out), rivals=False)
        _PRED["A"] = rows
    return {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r for r in _PRED["A"]}


def e1cal_pages(tree, preds, *, bias=None, noise=0.002, skip=(), seed=7):
    rng = random.Random(seed)
    files = REGS["e1cal"]["pages"]["files"]
    for label, model, name in RG.A_TIMED_PAGES:
        if (model, label) in skip:
            continue
        A = {}
        for k, r in preds.items():
            if k[0] == model and k[1] == name and k[4] == "shared":
                A[k[2]] = r["A_proxy_us"]

        def ms_of(n, lab, arm, model=model, name=name):
            r = preds[(model, name, n, lab, arm)]
            b = (bias or {}).get((model, name), 0.0)
            return r["ms"]["v4"] / (1 + b) * (1 + rng.gauss(0, noise))
        write_timed(tree, model, label, files[f"{name}.json"], ms_of, A_of=A)


def test_e1cal_holds_on_pages_that_follow_v4_and_the_stop_rule_reads_go(tmp_path, repo):
    out = tmp_path / "out"
    a_scores(out)
    e1cal_pages(tmp_path, a_preds(out))
    res = SE.score(repo, tmp_path, out)
    assert res["E1-CAL"]["verdict"] == "HOLDS", res["E1-CAL"]
    assert res["STOP"]["verdict"].startswith("GO"), res["STOP"]
    cm = res["STOP"]["class_means"]
    assert set(cm) == {"64g1", "128g1", "128g16"} and len(cm["128g1"]["by_model"]) == 3
    assert res["OLS_fit"]["verdict"] == "FITTED" and res["OLS_fit"]["cells"] >= 12


def test_a_class_mean_past_three_quarters_of_a_percent_stops_rental_b(tmp_path, repo):
    out = tmp_path / "out"
    a_scores(out)
    e1cal_pages(tmp_path, a_preds(out), bias={("mixtral-8x7b", "mixtral-8x7b-128g16"): 0.012})
    res = SE.score(repo, tmp_path, out)
    assert res["STOP"]["verdict"].startswith("STOP")
    assert any("class 128g16" in w for w in res["STOP"]["why"]), res["STOP"]["why"]


def test_two_cal_models_apart_by_over_one_percent_stop_rental_b(tmp_path, repo):
    out = tmp_path / "out"
    a_scores(out)
    e1cal_pages(tmp_path, a_preds(out), bias={("olmoe-1b-7b", "olmoe-1b-7b-128g1"): 0.007,
                                              ("qwen2-57b-a14b", "qwen2-57b-a14b-128g1"): -0.007})
    res = SE.score(repo, tmp_path, out)
    assert any("CAL models' means differ" in w for w in res["STOP"]["why"]), res["STOP"]


def test_a_skipped_e1cal_page_is_a_stop(tmp_path, repo):
    out = tmp_path / "out"
    a_scores(out)
    e1cal_pages(tmp_path, a_preds(out), skip={("qwen2-57b-a14b", "e128g1")})
    res = SE.score(repo, tmp_path, out)
    assert res["STOP"]["verdict"].startswith("STOP")
    assert any("qwen2-57b-a14b-e128g1" in w for w in res["STOP"]["why"])


def test_no_calibration_means_no_e1cal_and_a_stop(tmp_path, repo):
    out = tmp_path / "out"
    out.mkdir()
    res = SE.score(repo, tmp_path, out)
    assert res["E1-CAL"]["verdict"].startswith("NOT SCORED") and res["STOP"]["verdict"].startswith("STOP")


# --------------------------------------------------------------------------
# tA-rep and tA-dead
# --------------------------------------------------------------------------

def test_the_replicate_reads_sigma_page_and_its_absence_falls_back(tmp_path, repo):
    f = REGS["e1cal"]["pages"]["files"]["olmoe-1b-7b-128g1.json"]
    for label, s in (("e128g1", 1), ("e128g1r", 2)):
        rng = random.Random(s)
        write_timed(tmp_path, "olmoe-1b-7b", label, f, lambda n, lab, arm, rng=rng: 1.0 * (1 + rng.gauss(0, 0.002)))
    res = SR.score(repo, tmp_path)
    assert res["REP"]["verdict"] == "MEASURED" and 0.0008 < res["REP"]["sigma_page"] < 0.0035
    none = SR.score(repo, tmp_path / "none")
    assert none["sigma_page_used"] == TC.SIGMA_PAGE_R3 and "0.18%" in none["source"]


@pytest.mark.parametrize("law, verdict", [("D", "D HOLDS, SLOT EXCLUDED"), ("SLOT", "SLOT HOLDS, D EXCLUDED")])
def test_the_dead_lever_separates_d_from_slot_at_occupancy_two(tmp_path, repo, law, verdict):
    out = tmp_path / "out"
    out.mkdir()
    (out / "floor.score.json").write_text(json.dumps({"calibration": {"c": 900.0, "F": NOMINAL["F"]}}))
    inc = RG.dead_increments(900.0)[law]
    f = REGS["dead"]["pages"]["files"]["olmoe-1b-7b-dead.json"]
    for label, extra in (("d9", 0.0), ("d15", inc * 1e-3)):
        write_timed(tmp_path, "olmoe-1b-7b", label, f, lambda n, lab, arm, extra=extra: 0.5 + (0.01 + extra if arm == "shared" else 0.0),
                    copies=15 if label == "d15" else 9)
    res = SD.score(repo, tmp_path, out)
    assert res["DEAD"]["D_vs_SLOT"] == verdict, res["DEAD"]
    assert res["DEAD"]["laws"]["kappa 0"]["verdict"] in ("EXCLUDED", "NOT EXCLUDED")   # c_128 900: scored


# --------------------------------------------------------------------------
# finalise and rental B
# --------------------------------------------------------------------------

def test_finalise_reads_the_selections_and_refuses_without_a_calibration(tmp_path):
    out = tmp_path / "out"
    a_scores(out, sigma=0.0021)
    sel = FN.selections(out)
    assert sel["key"] == "CLAMP" and sel["variant"] == "W" and sel["sigma_page"] == 0.0021 and sel["f_form"] == "F-step"
    (out / "floor.score.json").write_text(json.dumps({"calibration": {"c": 900.0, "F": NOMINAL["F"]}, "selection": {"selected": None},
                                                      "fit_64x128_w8": {"c": 530.0, "F": {"w1": 300.0, "w2": 560.0}}}))
    sel = FN.selections(out)
    assert sel["variant"] is None and sel["v4cal"][V4.T64W8.key]["c"] == 530.0
    assert V4.c_counter(V4.T64W8, "w1", FN.params(sel)) == 530.0
    (out / "floor.score.json").write_text(json.dumps({"calibration": {"verdict": "NOT SCORED"}}))
    with pytest.raises(FN.Refused):
        FN.selections(out)


_B: dict = {}


def b_predictions(tmp_path_factory):
    if "doc" not in _B:
        out = tmp_path_factory.mktemp("ascores")
        a_scores(out, sigma=0.0018)
        (out / "e1cal.score.json").write_text(json.dumps({"OLS_fit": {"verdict": "NOT SCORED: synthetic"},
                                                          "STOP": {"verdict": "GO: rental B may run", "why": []}}))
        doc = FN.finalise(REPO, out)
        p = out / "tileB-predictions.json"
        p.write_text(json.dumps(doc, default=str))
        _B["doc"], _B["path"] = doc, p
    return _B["doc"], _B["path"]


def b_census(tree, occ=None):
    """rental B's own census (label tb), every (model, tile, GEMM) at rental A's design occupancy unless
    `occ` {(model, tile key): ctas} says otherwise."""
    e = []
    for u in RG.B_TIMED_PAGES:
        model, name = u[1], u[2]
        t = RG.PAGE_TILE[int(RG.hist("B", name)["block_m"])]
        for g in ("w1", "w2"):
            e.append({"model": model, "tile": t.key, "gemm": g, "occupancy": {"ctas_per_sm": (occ or {}).get((model, t.key), t.occ_shared())},
                      "ptx_counts": {}})
    d = tree / "results" / f"2026-10-14-{CARD}-tile-census-mixtral-8x7b-tb"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tile-census.json").write_text(json.dumps({"entries": e, "path_mismatch": [], "errors": [], "verdict": "COMPLETE"}))


def b_pages(tree, doc, *, bias=None, noise=0.002, seed=11, lost=(), census=True):
    if census:
        b_census(tree)
    rng = random.Random(seed)
    files = REGS["e1tile"]["pages"]["files"]
    preds = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r for r in doc["cells"]}
    for u in RG.B_TIMED_PAGES:
        label, model, name = u[:3]
        if model in lost:
            continue
        A = {k[2]: r["A_proxy_us"] for k, r in preds.items() if k[0] == model and k[1] == name and k[4] == "shared"}

        def ms_of(n, lab, arm, model=model, name=name):
            r = preds[(model, name, n, lab, arm)]
            return r["ms"]["v4"] / (1 + (bias or {}).get(model, 0.0)) * (1 + rng.gauss(0, noise))
        write_timed(tree, model, label, files[f"{name}.json"], ms_of, A_of=A)
    for u in RG.B_BYTE_PAGES:
        label, model, name = u[:3]
        reg_file = files[f"{name}.json"]
        hdoc = json.loads((REPO / reg_file["path"]).read_text())
        write_counter(tree, model, label, reg_file["group_m"], _tile_of(reg_file["tile"][0]), hist=(reg_file, hdoc),
                      reads=lambda lab, n, arm, g, model=model, name=name: preds[(model, name, n, lab, arm)]["reads"][g])


def test_rental_b_holds_and_certifies_on_pages_that_follow_the_finalised_v4(tmp_path, tmp_path_factory, repo):
    doc, path = b_predictions(tmp_path_factory)
    assert len([r for r in doc["cells"] if r["arm"] == "native"]) == 84
    b_pages(tmp_path, doc)
    res = STB.score(repo, tmp_path, path)
    assert res["E1-TILE"]["verdict"] == "HOLDS", res["E1-TILE"]
    assert res["certification"] == {m: "CERTIFIED" for m in RG.BLIND_TILE}, res["certification"]
    assert res["E1-TILE"]["per_model"][RG.DSV]["label"].startswith("DESCRIPTIVE")
    assert res["bytes"]["B-BYTES"]["verdict"] == "HOLDS" and res["bytes"]["E2-TILE"]["verdict"] == "HOLDS"
    riv = res["rivals"]
    assert riv["roofline"]["verdict"] == "EXCLUDED" and riv["v3own"]["verdict"] == "EXCLUDED", riv
    assert riv["eta_mma rival"]["verdict"].startswith("NOT SEPARATED")


def test_one_blind_model_off_the_bar_fails_e1_tile_and_deepseek_alone_only_moves_the_pool(tmp_path, tmp_path_factory, repo):
    doc, path = b_predictions(tmp_path_factory)
    b_pages(tmp_path / "j", doc, bias={"jetmoe-8b": 0.02})
    res = STB.score(repo, tmp_path / "j", path)
    assert res["E1-TILE"]["verdict"] == "FAILS" and res["E1-TILE"]["per_model"]["jetmoe-8b"]["verdict"] == "FAILS"
    assert res["certification"]["phi-3.5-moe"].startswith("NOT CERTIFIED")
    b_pages(tmp_path / "d", doc, bias={RG.DSV: 0.02})
    res = STB.score(repo, tmp_path / "d", path)
    assert res["E1-TILE"]["per_model"][RG.DSV]["verdict"] == "FAILS"
    assert res["E1-TILE"]["verdict"] == "HOLDS", res["E1-TILE"]["pooled"]      # the pool absorbs one page


def test_the_stop_rule_power_is_the_reviews_at_zero_drift():
    pts = REGS["stop"]["power (from tileB-e1tile, CPU float64)"]
    z = pts["common 0 / model-specific 0"]
    assert 0.9 < z["P(stop rule A2 passes)"] < 0.95 and z["P(E1-TILE)"] > 0.95
    assert REGS["e1tile"]["power"]["mps_check"]["max_abs_diff"] < 1e-4


def test_a_lost_blind_model_reads_not_scored_not_fails(tmp_path, tmp_path_factory, repo):
    doc, path = b_predictions(tmp_path_factory)
    b_pages(tmp_path, doc, lost=("jetmoe-8b",))
    res = STB.score(repo, tmp_path, path)
    assert res["E1-TILE"]["verdict"].startswith("NOT SCORED (incomplete"), res["E1-TILE"]["verdict"]
    assert res["E1-TILE"]["models_without_a_valid_cell"] == ["jetmoe-8b"]


def test_rental_bs_census_gates_its_pages(tmp_path, tmp_path_factory, repo):
    """ambiguity 11: a (model, tile) at another occupancy on rental B's board is NOT SCORED; no rental-B
    census leaves every page NOT SCORED."""
    doc, path = b_predictions(tmp_path_factory)
    b_pages(tmp_path / "x", doc, census=False)
    b_census(tmp_path / "x", {("phi-3.5-moe", "128x128x64w8s3"): 1})
    res = STB.score(repo, tmp_path / "x", path)
    assert res["census"]["excluded"] == [("phi-3.5-moe", "128x128x64w8s3")]
    pv = res["E1-TILE"]["per_model"]["phi-3.5-moe"]
    assert pv["stats"]["cells"] == 6 and res["certification"]["phi-3.5-moe"].startswith("NOT CERTIFIED")
    b_pages(tmp_path / "none", doc, census=False)
    res = STB.score(repo, tmp_path / "none", path)
    assert res["census"]["excluded"] == "ALL" and res["E1-TILE"]["verdict"].startswith("NOT SCORED")


def test_a_census_path_mismatch_stops_rental_b_and_finalise_refuses(tmp_path, repo):
    out = tmp_path / "out"
    a_scores(out)
    e1cal_pages(tmp_path, a_preds(out))
    ok = SE.score(repo, tmp_path, out)
    assert ok["STOP"]["verdict"].startswith("GO")
    (out / "census.score.json").write_text(json.dumps({"CENSUS": {"verdict": "REFUSED (path mismatch): x"}, "occupancy": {}}))
    with pytest.raises(FN.Refused):
        FN.selections(out)
    st = TC.stop_rule({"pages": {}, "rows": [], "bar": {"verdict": "HOLDS"}}, {"CENSUS": {"verdict": "REFUSED (path mismatch)"}})
    assert any("census" in w for w in st["why"])
    import tcommon
    tcommon.use_census(None)


def test_finalise_refuses_on_a_stop_and_writes_nothing(tmp_path):
    out = tmp_path / "out"
    a_scores(out)
    (out / "e1cal.score.json").write_text(json.dumps({"STOP": {"verdict": "STOP: rental B does not run",
                                                               "why": ["class 128g16: mean residual +1.2%"]}}))
    dst = tmp_path / "p.json"
    assert FN.main([str(REPO), str(out), str(dst)]) == 2 and not dst.exists()
    (out / "e1cal.score.json").unlink()
    assert FN.main([str(REPO), str(out), str(dst)]) == 2 and not dst.exists()


def test_the_generated_rental_b_plan_parses_and_drops_byt_then_dsv(tmp_path):
    import os
    import re
    import subprocess
    f = tmp_path / "rentalB.plan"
    f.write_text(FN.plan_b())
    got = subprocess.run(["bash", str(REPO / "scripts" / "gh200_tile_session.sh"), "--dry-run", "--plan", str(f)],
                         capture_output=True, text=True, timeout=60, env={**os.environ, "HOME": "/nonexistent"})
    assert got.returncode == 2 and "a PLAN of 14 units" in got.stdout, got.stderr[-2000:]
    order = re.search(r"drop order when time runs short: .*\n\s+([\d ]+)", got.stdout).group(1).split()
    assert order == ["14", "13", "12", "11"]
    body = f.read_text()
    for u in RG.B_TIMED_PAGES:
        assert f"{u[1]} timed label={u[0]} " in body
    assert body.count("nodrop=1") == 9 and body.count("drop-group=byt") == 3 and body.count("drop-group=dsv") == 1


def test_the_floor_re_keys_a_page_at_its_recorded_occupancy(tmp_path, repo):
    """ambiguity 11, rental A's half: a 64x128 page recording another occupancy than the census is
    predicted at its recorded one."""
    prm3 = V4.Params(c128=900.0, F_wg=dict(NOMINAL["F"]), variant="W")
    prm2 = V4.Params(c128=900.0, F_wg=dict(NOMINAL["F"]), variant="W", occupancy={V4.T64W8.key: {"w1": 2, "w2": 2}})
    c3, c2 = V4.c_counter(V4.T64W8, "w1", prm3), V4.c_counter(V4.T64W8, "w1", prm2)
    assert c2 != c3
    for m in ("olmoe-1b-7b", "mixtral-8x7b"):
        write_counter(tmp_path, m, "f128", 64, V4.CAL_TILE, c=900.0, F=NOMINAL["F"])
        write_counter(tmp_path, m, "f64w8", 64, V4.T64W8, c=c2, F={g: 520.0 / 900.0 * c2 if g == "w1" else 979.0 / 900.0 * c2
                                                                     for g in ("w1", "w2")}, occ=2)
    res = SF.score(repo, tmp_path, tmp_path / "out")
    assert res["re_key"]["f64w8 recorded"] == {"olmoe-1b-7b": 2, "mixtral-8x7b": 2}
    assert res["FLOOR-TEST (DESCRIPTIVE)"]["W"]["verdict"] == "HOLDS", res["FLOOR-TEST (DESCRIPTIVE)"]["W"]
