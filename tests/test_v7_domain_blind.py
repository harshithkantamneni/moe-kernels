"""Model v7's decode DOMAIN blind (docs/registered/2026-11-18-v7-domain-blind-gh200), off the GPU and on SYNTHETIC
trees and SYNTHETIC map readouts only: the registration, its six pages and its plan recompute to their committed
files; v7's predictions are the v7 blind's frozen fit on the four models (Phase 0's pricing); the census finds no decode
page of any of the four and the map's shapes equal none of theirs; the bars and the in-domain minimum are registered
before D is known; --final refuses without a committed, verified map readout, a tampered or foreign readout, an empty
D and a D under which no model reaches the minimum, and labels each cell by D (a path with no X_p, X_p, the slab
guard's S_p, the second tile); the scorer scores only IN-DOMAIN cells, applies the minimum, and handles off-lock cells,
a missing page and a failed power check; the launcher is vm_run_v7b.sh plus lines marked V7D, delivers the plan and
pages on a stub VM and refuses a DRAFT plan."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "decode",
          REPO / "scripts" / "scoring" / "v7blind", REPO / "scripts" / "scoring" / "v7domain",
          REPO / "scripts" / "scoring" / "v7dblind"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import decode_fakes as F  # noqa: E402
import map_shapes as MS  # noqa: E402
import read_map as RM  # noqa: E402
import register_v7d_blind as RG  # noqa: E402
import score_blind_v7d as SV  # noqa: E402
import test_gh200_model_session as TG  # noqa: E402
import test_rr_driver as TRR  # noqa: E402
import v7dcommon as V  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

REGF = REPO / V.REG_JSON
PLAN = REPO / V.PLAN
DRIVER = REPO / "scripts" / V.DRIVER
LAUNCHER = REPO / V.LAUNCHER
PARENT = REPO / V.PARENT_LAUNCHER
V7B_REG = REPO / "docs/registered/2026-11-17-v7-decode-blind-gh200.json"
POWER = REPO / "results/published/2026-11-03-nvidia_gh200_480gb-decodeCAL2-session/session/power-set.txt"


def _reg() -> dict:
    return json.loads(REGF.read_text())


def _sessions() -> list[str]:
    return _reg()["models"]["census (COMPUTED)"]["sessions scanned"]


@pytest.fixture(scope="module")
def pred():
    return V.predictions()


# ---------------------------------------------------------------- the registration, pages and plan

def test_the_registration_pages_and_plan_recompute_to_their_committed_files(pred):
    reg = _reg()
    dom = sha = None
    if not reg["draft"]:
        dom, sha = json.loads((REPO / V.DOMAIN).read_text()), V.sha256(REPO / V.DOMAIN)
    doc = json.loads(json.dumps(RG.build(reg["draft"], pred, _sessions(), dom, sha), default=str))
    assert RG.close(doc, reg) == []
    assert (REPO / V.REG_TXT).read_text() == "\n".join(RG.text(doc)) + "\n"
    assert PLAN.read_text() == V.plan_text(doc["predictions"]["frozen"], doc["draft"])
    assert reg["rental"]["plan_sha256"] == hashlib.sha256(PLAN.read_bytes()).hexdigest()
    for p, t in V.page_texts().items():
        assert (REPO / p).read_text() == t, p
    assert sorted(x.name for x in (REPO / V.PAGES).iterdir()) == sorted(f"{m}-{k}.json" for m, k in V.PAGE_KEYS)
    assert V.verify_freeze(reg) == []


def test_v7_is_the_v7_blinds_frozen_fit_and_prices_every_cell_as_phase_0_did(pred):
    v7b = json.loads(V7B_REG.read_text())["predictions"]["frozen"]
    assert RG.close(json.loads(json.dumps(pred["frozen"], default=str)), v7b) == []
    fz = pred["frozen"]
    assert fz["model"]["switches"] == {"RR": True, "D6FIX": True} and fz["model"]["constants_sha256"] == V.V7.constants_sha256()
    assert pred["CAL2 refit under v7"]["verdict"] == "GO" and pred["not_priced"] == 0
    assert len(pred["cells"]) == 192 and len(pred["descriptive_cells"]) == 20
    assert pred["pricers agree"].endswith("within 0 relative on every cell") or "e-1" in pred["pricers agree"]
    # Phase 0 (moe-kernels-work 2026-11-18 v7-domain/REPORT.md, the four still-unseen models): 43 of 48 cells in scope
    # each, the paths and the T_v7 ranges
    want = {"deepseek-v3": ({"naive": 19, "block-scan": 24}, (207.0, 6292.7)),
            "deepseek-v3-tp4": ({"naive": 19, "block-scan": 24}, (63.0, 1622.2)),
            "deepseek-v3-tp8": ({"naive": 19, "block-scan": 24}, (42.3, 821.2)),
            "qwen2-57b-a14b-tp8": ({"naive": 7, "small-batch": 31, "block-scan": 5}, (29.3, 160.6))}
    for m, (paths, rng) in want.items():
        b = pred["per model"][m]
        assert b["cells"] == 48 and b["second tile (DESCRIPTIVE)"] == 5 and b["paths"] == paths
        assert (round(b["T_v7 range us"][0], 1), round(b["T_v7 range us"][1], 1)) == rng
    for r in pred["cells"]:
        assert r["geometry"]["path"] == r["path"] and r["geometry"]["T_v7_us"] == round(r["ms"]["v4d"] * 1e3, 6)
        assert r["geometry"]["second_tile"] == (f"{r['model']}/{r['label']}" in pred["descriptive_cells"])


def test_the_cells_keep_the_v7_blinds_recipe():
    reg = _reg()["cells"]
    v7b = json.loads(V7B_REG.read_text())["cells"]
    old = json.loads((REPO / v7b["pages"]["files"]["qwen2-57b-a14b-main.json"]["path"]).read_text())
    for m in V.MODELS:
        doc = json.loads((REPO / V.page_path(m, "main")).read_text())
        assert doc["shuffle_seed"] == old["shuffle_seed"] == 20261014 and all(c["arms"] == ["native"] for c in doc["cells"])
        assert {c["T"] for c in doc["cells"]} == {c["T"] for c in old["cells"]}
        assert {c["role"] for c in doc["cells"]} == {c["role"] for c in old["cells"]}
        assert [c["label"].split("-")[0] for c in doc["cells"]] == [c["label"].split("-")[0] for c in old["cells"]]
    for m in V.CTL:
        main = {c["label"] for c in json.loads((REPO / V.page_path(m, "main")).read_text())["cells"]}
        ctl = json.loads((REPO / V.page_path(m, "ctl")).read_text())["cells"]
        assert {c["label"] for c in ctl} <= main and {c["role"] for c in ctl} == {"BALANCED", "UNIFORM", "LADDER-BOTTOM"}
    assert [(u["label"], u["inst"], u["drop"]) for u in reg["units"]] == \
        [("dm", "GRR", "nodrop")] * 4 + [("dc", "GR1", "gr1")] * 2
    assert reg["drop_order"] == v7b["drop_order"] == ["gr1"]


def test_the_plan_lists_every_unit_and_the_pinned_driver_dry_runs_it():
    reg = _reg()
    units = [ln for ln in PLAN.read_text().splitlines() if ln and not ln.startswith("#")]
    assert units == reg["cells"]["plan_units"] == V.unit_lines()
    pages = re.findall(r"histogram=(\S+)", PLAN.read_text())
    assert sorted(pages) == sorted(v["path"] for v in reg["cells"]["pages"]["files"].values())
    for p in pages:
        assert hashlib.sha256((REPO / p).read_bytes()).hexdigest() == \
            reg["cells"]["pages"]["files"][Path(p).name]["sha256"] == reg["freeze"]["own"][p]
    got = subprocess.run(["bash", str(DRIVER), "--dry-run", "--plan", str(PLAN), "--deadline", "2000000000"],
                         capture_output=True, text=True, timeout=60, env={**os.environ, "HOME": "/nonexistent"}, cwd=REPO)
    assert got.returncode == 2, got.stdout + got.stderr
    assert "a PLAN of 8 units" in got.stdout and "89 min of units" in got.stdout
    assert got.stdout.count(" r3decode ") == 6 and re.search(r"drop order when time runs short.*\n\s+8 7\n", got.stdout)
    r = reg["rental"]
    assert (r["units_min"], r["wall_min"], r["deadline_after_launch_min"], r["hard_cap_after_launch_min"]) == \
        (89, 131, 145, 160)
    assert (r["cost_usd"], r["cost_usd_at_list"], r["cost_at_hard_cap_usd"], r["cost_at_hard_cap_usd_at_list"]) == \
        (5.0, 5.65, 6.11, 6.91)
    assert r["without the gr1 pages (DESCRIPTIVE)"]["wall_min"] == 107


# ---------------------------------------------------------------- the models

def test_no_model_has_a_decode_page_and_the_census_reads_what_it_should():
    reg = _reg()["models"]
    cen = reg["census (COMPUTED)"]
    assert reg["every model with no decode page"] == list(V.MODELS)
    assert "2026-11-18-nvidia_gh200_480gb-v7b-session" in cen["sessions scanned"]
    q = cen["per model"]["qwen2-57b-a14b-tp8"]
    # its R3 gap pages are found and classified: the R3 tile, not a decode page
    assert len(q["published gaps directories for it or its shape"]) == 4
    assert all(not g["decode"] and g["tiles"] == ["32x64x64w8s4"] for g in q["published gaps directories for it or its shape"].values())
    assert q["decode tiles among them"] == [] and q["tiles named by the JSON files whose names name it"]
    d = cen["per model"]["deepseek-v3"]["not pages: merged.csv rows naming it (earlier timings, other cards)"]
    assert {g for x in d.values() for g in x["gpu"]} == {"NVIDIA H200", "NVIDIA A100-SXM4-80GB"}
    for m in ("deepseek-v3-tp4", "deepseek-v3-tp8"):
        x = cen["per model"][m]
        assert not x["not pages: published file names naming it, per session"]
        assert not x["not pages: merged.csv rows naming it (earlier timings, other cards)"]
    # the classifier's positive control: the v7 blind's pages are decode pages
    g = REPO / "results/published/2026-11-18-nvidia_gh200_480gb-v7b-session/results/gaps-nvidia_gh200_480gb-mixtral-8x22b-dm"
    assert V.gaps_kind(g)["decode"] and "decode-page" in V.gaps_kind(g)["kinds"]
    # the map's shapes equal none of the four models' (E, H, I, k)
    mv = reg["the map's synthetic shapes against these models (COMPUTED)"]
    assert all(x["map shapes with this (E, H, I, k)"] == [] for x in mv.values())
    assert reg["GD5 offline (vLLM tuned configs)"]["verdict"].startswith("PASS offline")


def test_names_are_matched_exactly():
    assert V.names_model("gaps-x-deepseek-v3-dm", "deepseek-v3")
    assert not V.names_model("gaps-x-deepseek-v3-tp4-dm", "deepseek-v3")
    assert V.names_model("census-deepseek_v3_tp8.json", "deepseek-v3-tp8")
    assert not V.names_model("qwen2-57b-a14b-dm", "qwen2-57b-a14b-tp8")


def test_the_bars_the_minimum_and_the_disclosures_are_registered_before_d():
    reg = _reg()
    want = {"rms_max": 0.03, "max_abs": 0.08, "mean_max": 0.015}
    assert {k: reg["bars"][k] for k in want} == want == V.BARS
    assert "a model is CERTIFIED when its own bar and the pooled bar hold" in reg["certification"]
    assert reg["minimum"] == V.MIN_RULE and V.MIN_IN_DOMAIN == 10
    assert "before D is known" in V.MIN_RULE and "BELOW THE MINIMUM" in V.MIN_RULE
    t = " ".join(reg["disclosures"])
    for s in ("second reopening of decode (owner, 2026-11-18)", "POST-HOC origin", "v7 is frozen and unchanged",
              "Phase 0 priced these four models offline", "before any map page", V.MAP_REG_COMMIT):
        assert s in t, s
    assert reg["domain"]["rule (the map registration's, verbatim)"] == V.VM.RULE_TEXT
    assert V.sha256(REPO / f"docs/registered/{V.MAP_NAME}.json") == V.MAP_REG_SHA256


def test_the_committed_draft_or_final_matches_whether_the_readout_is_committed(tmp_path, capsys):
    reg = _reg()
    have = (REPO / V.DOMAIN).is_file()
    if reg["draft"]:
        assert reg["status"].startswith("DRAFT") and reg["domain"]["status"].startswith("NOT KNOWN")
        assert V.DRAFT_LINE in PLAN.read_text().splitlines() and (REPO / V.REG_TXT).read_text().startswith("DRAFT")
        assert SV.refusals(reg)[0].startswith("the registration is a DRAFT")
        if not have:
            assert RG.main(["--write", "--final", "--root", str(tmp_path)]) == 2
            assert "does not exist" in capsys.readouterr().out and not (tmp_path / V.REG_JSON).exists()
    else:
        assert have and V.is_committed(V.DOMAIN) and "# DRAFT" not in PLAN.read_text()
        assert reg["domain"]["readout"]["sha256"] == V.sha256(REPO / V.DOMAIN)
        assert SV.refusals(reg) == []


# ---------------------------------------------------------------- D from synthetic map readouts

def _row(shape, path, pred, e, *, fam="A", W=6.0, waves=1.5, T=8, meas=None, ub=None):
    return {"model": shape, "label": f"x{pred}", "path": path, "family": fam, "pred_us": pred, "pred_ms": pred / 1e3,
            "meas_us": pred / (1 + e) if meas is None else meas, "e": e, "e_ub": abs(e) + 0.004 if ub is None else ub,
            "in_scope": True, "valid": True, "W_e_MB": W, "waves": waves, "T": T}


def _readout(kind: str = "good") -> dict:
    """A synthetic map readout: naive good past 100 us, small-batch good, block-scan never. 'slab': small-batch is
    family B under a wave at W_e 8 to 37 MB with a residual falling with W_e (the guard triggers; S_p >= 8 MB, past
    Qwen2-57B-tp8's 6.88 MB); 'empty': every path fails; 'high': naive's X_p past every cell of the four models and
    small-batch failing."""
    rows = []
    for i in range(42):
        s = MS.NAMES[i % 14]
        pn = 20.0 + 10 * i
        rows.append(_row(s, "naive", pn if kind != "high" else 7000 + pn, 0.001 if pn >= 100 and kind != "empty" else 0.07))
        ps = 15.0 + 5 * i
        if kind == "slab":
            W = 8.0 + 0.7 * i
            rows.append(_row(MS.NAMES[6 + i % 6], "small-batch", ps, 0.001, fam="B", W=W, waves=0.5,
                             meas=ps - (4.0 - 0.12 * W)))
        else:
            rows.append(_row(s, "small-batch", ps, 0.001 if kind not in ("empty", "high") else 0.07))
        rows.append(_row(s, "block-scan", 30.0 + 20 * i, 0.07))
    dom = {"what": "synthetic", "registration": V.MAP_NAME, "verdict": "READ",
           "provenance": {"registration_sha256": V.MAP_REG_SHA256, "registration_path": V.VM.REG_JSON},
           "cells": {"registered": len(rows), "valid": len(rows)}, "domain": RM.domain(rows), "rows": rows}
    return json.loads(json.dumps(dom))


@pytest.fixture()
def readout(tmp_path, monkeypatch):
    """Points --final at a synthetic readout file held committed, the map session held published."""
    f = tmp_path / "map.domain.json"
    monkeypatch.setattr(V, "DOMAIN", str(f))
    monkeypatch.setattr(V, "is_committed", lambda rel, root=V.REPO: True)
    monkeypatch.setattr(V, "map_sessions", lambda sessions: ["synthetic-map-session"])

    def put(dom):
        f.write_text(json.dumps(dom, indent=1) + "\n")
        return f
    return put


def _expect_in(geom: dict, D: dict) -> bool:
    p = D["per_path"][geom["path"]]
    return (not geom["second_tile"] and p["in"] and geom["T_v7_us"] >= p["X_p_us"]
            and (p["S_p_MB"] is None or geom["W_e_MB"] >= p["S_p_MB"]))


def test_final_labels_every_cell_by_d_and_a_path_with_no_x_p_is_out(readout, pred, tmp_path):
    readout(_readout("good"))
    dom, sha, why = RG.final_refusals(_sessions(), pred)
    assert why == []
    D = V.domain_of(dom)
    assert D["per_path"]["block-scan"]["X_p_us"] is None and not D["per_path"]["block-scan"]["in"]
    assert D["per_path"]["naive"]["X_p_us"] >= 100 and D["per_path"]["small-batch"]["in"]
    doc = RG.build(False, pred, _sessions(), dom, sha)
    lab = doc["domain"]["labels"]
    for r in pred["cells"]:
        x = lab["cells"][f"{r['model']}/{r['label']}"]
        assert x["in"] == _expect_in(r["geometry"], D), (r["label"], x)
        if r["geometry"]["path"] == "block-scan" and not r["geometry"]["second_tile"]:
            assert not x["in"] and "outside D" in x["why"]
        if r["geometry"]["second_tile"]:
            assert not x["in"] and "second M-tile" in x["why"]
    assert lab["in-domain cells per model"]["deepseek-v3"] == 19            # its naive cells, all over 207 us
    assert doc["draft"] is False and doc["freeze"]["own"][V.DOMAIN] == sha
    assert "# DRAFT" not in V.plan_text(doc["predictions"]["frozen"], False)
    assert RG.main(["--write", "--final", "--root", str(tmp_path)]) == 0
    assert json.loads((tmp_path / V.REG_JSON).read_text())["draft"] is False
    assert (tmp_path / V.REG_TXT).read_text().startswith("REGISTERED ")


def test_the_slab_guard_sets_s_p_and_cells_under_it_are_out(readout, pred):
    readout(_readout("slab"))
    dom, sha, why = RG.final_refusals(_sessions(), pred)
    assert why == [] and dom["domain"]["slab_guard"]["triggered"]
    D = V.domain_of(dom)
    sp = D["per_path"]["small-batch"]["S_p_MB"]
    assert sp is not None and sp >= 8.0 > 6.88128
    lab = V.labels(pred["cells"], D)
    q = [r for r in pred["cells"] if r["model"] == "qwen2-57b-a14b-tp8" and r["geometry"]["path"] == "small-batch"
         and not r["geometry"]["second_tile"]]
    assert q and all(not lab["cells"][f"{r['model']}/{r['label']}"]["in"] and "under S_p" in
                     lab["cells"][f"{r['model']}/{r['label']}"]["why"] for r in q if r["geometry"]["T_v7_us"] >=
                     D["per_path"]["small-batch"]["X_p_us"])
    for r in pred["cells"]:
        assert lab["cells"][f"{r['model']}/{r['label']}"]["in"] == _expect_in(r["geometry"], D)


def test_final_refuses_a_tampered_foreign_or_unread_readout_an_empty_d_and_a_d_no_model_can_pass(readout, pred, monkeypatch):
    good = _readout("good")
    t = copy.deepcopy(good)
    t["domain"]["per_path"]["naive"]["X_p_us"] = 30.0                      # D edited after the rule ran
    readout(t)
    assert any("read_map.domain on the readout's rows" in w for w in RG.final_refusals(_sessions(), pred)[2])
    t = copy.deepcopy(good)
    t["provenance"]["registration_sha256"] = "0" * 64
    readout(t)
    assert any("another map registration sha256" in w for w in RG.final_refusals(_sessions(), pred)[2])
    readout(dict(copy.deepcopy(good), verdict="NOT READ"))
    assert any("not READ" in w for w in RG.final_refusals(_sessions(), pred)[2])
    readout(_readout("empty"))
    assert any(w.startswith("D is EMPTY") for w in RG.final_refusals(_sessions(), pred)[2])
    readout(_readout("high"))
    assert any(w.startswith("no model has 10 registered in-domain cells") for w in RG.final_refusals(_sessions(), pred)[2])
    readout(good)
    monkeypatch.setattr(V, "is_committed", lambda rel, root=V.REPO: False)
    assert any("is not committed" in w for w in RG.final_refusals(_sessions(), pred)[2])
    monkeypatch.setattr(V, "is_committed", lambda rel, root=V.REPO: True)
    monkeypatch.setattr(V, "map_sessions", lambda sessions: [])
    assert any("no published session holds a map page" in w for w in RG.final_refusals(_sessions(), pred)[2])
    # the map registration at other bytes
    mreg = json.loads((REPO / f"docs/registered/{V.MAP_NAME}.json").read_text())
    assert any("not the bytes registered" in w for w in V.domain_refusals(good, "x", mreg, "0" * 64))
    mreg["rental"]["plan_sha256"] = "0" * 64
    assert any("does not verify" in w for w in V.domain_refusals(good, "x", mreg, V.MAP_REG_SHA256))


# ---------------------------------------------------------------- the scorer, on synthetic trees

@pytest.fixture(scope="module")
def final_reg(pred, tmp_path_factory):
    """A FINAL document built in the test from the 'good' synthetic readout (never committed)."""
    f = tmp_path_factory.mktemp("dom") / "map.domain.json"
    dom = _readout("good")
    f.write_text(json.dumps(dom))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(V, "DOMAIN", str(f))
        doc = RG.build(False, pred, _sessions(), dom, V.sha256(f))
    return json.loads(json.dumps(doc, default=str))


def _page(tree: Path, m: str, label: str, inst: str, path: Path, truth, slip=()):
    RD = F.RD
    cfg = F.MODEL_CONFIGS[m]
    page = RD.load_page(path, cfg, m)
    out = tree / f"gaps-{F.CARD}-{m}-{label}" / "private_weight_reference" / f"run-{label}-{inst}"
    out.mkdir(parents=True, exist_ok=True)
    store = RD.Store(out / "cells.csv")
    be = F.FakeBackend(cfg, page, truth=truth, noise=0.0005, seed=7, slip=slip)
    res = RD.run_page(cfg, page, be, inst=inst, repeats=9, store=store, lock_mhz=1710, min_clean=0, say=lambda s: None)
    store.close()
    args = argparse.Namespace(model=m, dtype="bf16", inst=inst, duty=0.25, repeats=9, group_m=1, seed=0,
                              session_tag="T-lock1710", min_clean_repeats=0, rot="auto")
    RD.write_report(args, cfg, page, res, out_dir=out, run_id=out.name, card=F.CARD, R=1, calls=1, copies=1,
                    slip_lock=1710, header=["DECODE PAGE (fake)"], seconds=1.0)


def _tree(root: Path, reg: dict, meas, *, skip=(), slip=None) -> Path:
    """Every unit's page, measured time meas(model, label, pred_ms, in_domain) in ms; GR1 pages 2 us over GRR."""
    pr = {(r["model"], r["label"]): r["ms"]["v4d"] for r in reg["predictions"]["cells"]}
    lab = reg["domain"]["labels"]["cells"]
    (root / "session").mkdir(parents=True)
    shutil.copy(POWER, root / "session" / "power-set.txt")
    tree = root / "results"
    for u in reg["cells"]["units"]:
        m = u["model"]
        if (m, u["label"]) in skip:
            continue
        plus = 0.002 if u["inst"] == "GR1" else 0.0
        _page(tree, m, u["label"], u["inst"], REPO / reg["cells"]["pages"]["files"][u["page"]]["path"],
              lambda c, a, m=m, plus=plus: meas(m, c.label, pr[(m, c.label)], lab[f"{m}/{c.label}"]["in"]) + plus,
              (slip or {}).get((m, u["label"]), ()))
    return tree


#: OUT cells 20% off: they must not move the registered reading
def _good(m, lab, p, inn):
    return p if inn else p / 1.2


def _slow(m, lab, p, inn):
    return p * 1.035 if m == "deepseek-v3-tp4" else p


@pytest.fixture(scope="module")
def scored(final_reg, tmp_path_factory):
    reg = final_reg
    good = SV.score(_tree(tmp_path_factory.mktemp("good"), reg, _good), reg)
    slow = SV.score(_tree(tmp_path_factory.mktemp("slow"), reg, _slow), reg)
    # deepseek-v3's main page missing; one deepseek-v3-tp8 cell off-lock (5 of 9 repeats); every repeat of six
    # qwen2-57b-a14b-tp8 cells off-lock (54 of 432 rows, over G1's 10% cap: G1 voids that page)
    bad_tp8 = next(r["label"] for r in reg["predictions"]["cells"] if r["model"] == "deepseek-v3-tp8"
                   and reg["domain"]["labels"]["cells"][f"deepseek-v3-tp8/{r['label']}"]["in"])
    slip = {("deepseek-v3-tp8", "dm"): tuple((bad_tp8, "native", n) for n in range(1, 6)),
            ("qwen2-57b-a14b-tp8", "dm"): tuple((lab, "native", n) for lab in ("L-T32-A64", "L-T64-A64", "L-T96-A64",
                                                                               "L-T128-A64", "U-T64", "U-T96") for n in range(1, 10))}
    adv_tree = _tree(tmp_path_factory.mktemp("adv"), reg, _good, skip={("deepseek-v3", "dm")}, slip=slip)
    adv = SV.score(adv_tree, reg)
    low = tmp_path_factory.mktemp("lowpower")
    (low / "session").mkdir()
    (low / "session" / "power-set.txt").write_text(POWER.read_text().replace(
        "Current Power Limit               : 700.00 W", "Current Power Limit               : 600.00 W"))
    (low / "results").symlink_to(adv_tree)
    return {"good": good, "slow": slow, "adv": adv, "adv_bad": bad_tp8, "low": SV.score(low / "results", reg)}


def test_the_scorer_reads_only_in_domain_cells_and_certifies_what_v7_prices(scored, final_reg):
    s = scored["good"]
    n = final_reg["domain"]["labels"]["in-domain cells per model"]
    assert s["POWER"]["verdict"] == "READ" and s["bars"] == V.BARS
    r = s["E1_DEC_IN_DOMAIN"]
    assert r["in-domain cells scored"] == n and r["pooled"]["verdict"] == "HOLDS" and abs(r["pooled"]["stats"]["rms"]) < 0.002
    for m in V.MODELS:
        assert s["certified"][m] == ("CERTIFIED" if n[m] >= 10 else
                                     f"NOT CERTIFIED: BELOW THE MINIMUM ({n[m]} in-domain cells scored, under 10)")
    # the OUT cells, 20% off, read FAILS beside, DESCRIPTIVE; the frozen scorer's all-cell reading fails with them
    assert r["OUT cells (DESCRIPTIVE)"]["pooled"]["verdict"] == "FAILS"
    assert s["ALL_CELLS_DESCRIPTIVE"]["E1_DEC"]["pooled"]["verdict"] == "FAILS"
    assert {x["domain"] for x in s["rows"]} == {"IN", "OUT"} and len(s["rows"]) == 192 - 20
    assert {m: s["GD2"][m]["verdict"] for m in V.CTL} == {m: "PASS" for m in V.CTL}
    assert SV.lines(s)[0].startswith("MODEL v7 DECODE DOMAIN BLIND")


def test_a_model_v7_misprices_in_domain_is_not_certified(scored, final_reg):
    s = scored["slow"]
    n = final_reg["domain"]["labels"]["in-domain cells per model"]
    assert n["deepseek-v3-tp4"] >= 10
    assert s["E1_DEC"]["per_model"]["deepseek-v3-tp4"]["verdict"] == "FAILS"
    assert s["certified"]["deepseek-v3-tp4"] == "NOT CERTIFIED"
    if s["E1_DEC"]["pooled"]["verdict"] == "HOLDS":
        assert s["certified"]["deepseek-v3"] == "CERTIFIED"


def test_a_model_under_the_minimum_is_reported_and_not_certified(scored, final_reg, tmp_path):
    reg = copy.deepcopy(final_reg)
    cells = reg["domain"]["labels"]["cells"]
    keep = [k for k, v in cells.items() if k.startswith("deepseek-v3/") and v["in"]][:9]
    for k, v in cells.items():
        if k.startswith("deepseek-v3/") and k not in keep:
            v["in"] = False
    s = SV.in_domain([dict(r, e=0.0) for r in scored["good"]["rows"]], reg)
    assert s["in-domain cells scored"]["deepseek-v3"] == 9 and s["per_model"]["deepseek-v3"]["verdict"] == "HOLDS"
    c = SV.certify(s, {})
    assert c["deepseek-v3"] == "NOT CERTIFIED: BELOW THE MINIMUM (9 in-domain cells scored, under 10)"
    assert "deepseek-v3" in s["pooled_over"]                                 # its cells still count in the pooled bar


def test_off_lock_cells_a_voided_page_and_a_missing_page(scored):
    s = scored["adv"]
    assert s["units"]["deepseek-v3-dm"]["state"].startswith("NOT SCORED")
    assert s["E1_DEC_IN_DOMAIN"]["in-domain cells scored"]["deepseek-v3"] == 0
    assert s["certified"]["deepseek-v3"].startswith("NOT CERTIFIED: BELOW THE MINIMUM (0 ")
    rows = {(r["model"], r["label"]) for r in s["rows"]}
    assert ("deepseek-v3-tp8", scored["adv_bad"]) not in rows                # 4 clean repeats, under 6: not read
    assert s["units"]["qwen2-57b-a14b-tp8-dm"]["state"].startswith("G1 FAIL")
    assert not any(m == "qwen2-57b-a14b-tp8" for m, _ in rows)
    assert "deepseek-v3" not in s["E1_DEC_IN_DOMAIN"]["pooled_over"]


def test_a_failed_power_check_scores_nothing(scored):
    s = scored["low"]
    assert s["verdict"] == "NOT SCORED" and "600" in s["POWER"]["verdict"]
    assert SV.lines(s)[0].endswith("NOT SCORED")


def test_the_scorer_reads_only_the_committed_registration_and_refuses_a_draft_and_a_broken_freeze(tmp_path, final_reg):
    with pytest.raises(SystemExit):
        SV.main([str(tmp_path), str(tmp_path / "o"), "--registration", str(REGF)])   # no such option
    d = dict(copy.deepcopy(final_reg), draft=True)
    assert SV.refusals(d)[0].startswith("the registration is a DRAFT")
    reg = _reg()
    if reg["draft"]:
        assert SV.main([str(tmp_path), str(tmp_path / "o")]) == 2
    f = copy.deepcopy(final_reg)
    f["freeze"] = V.freeze(final=False)
    assert SV.refusals(f) == []
    p = next(iter(f["freeze"]["own"]))
    f["freeze"]["own"][p] = "0" * 64
    assert SV.refusals(f)[0].startswith("the freeze does not hold")
    assert SV.refusals(dict(final_reg, name="another"))[0].startswith("the registration is not")
    pv = SV.provenance(REGF)
    assert pv["registration_sha256"] == hashlib.sha256(REGF.read_bytes()).hexdigest() and pv["registration_path"] == V.REG_JSON


# ---------------------------------------------------------------- the launcher

def test_the_launcher_is_v7bs_plus_lines_marked_v7d():
    changed = TRR._held_to_parent(PARENT, LAUNCHER, None, mark="V7D")
    assert sum("^# DRAFT" in x and "refuse" in x and V.NAME in x for x in changed) == 1
    assert "PYTHONPATH" not in "\n".join(changed)
    assert (LAUNCHER.stat().st_mode & 0o777) == (PARENT.stat().st_mode & 0o777)
    assert V.sha256(PARENT) == _reg()["freeze"]["own"][V.PARENT_LAUNCHER]


def _laptop(tmp_path, plan_text: str):
    pages = sorted(set(re.findall(r"histogram=(\S+)", plan_text)))
    lap = TG.Laptop(tmp_path)
    (lap.clone / "scripts" / "plans").mkdir()
    (lap.clone / V.PLAN).write_text(plan_text)
    for f in (DRIVER, LAUNCHER):
        shutil.copy(f, lap.clone / "scripts" / f.name)
    for p in pages:
        (lap.clone / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / p, lap.clone / p)
    g = ["git", "-C", str(lap.clone)]
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "plan"], check=True)
    subprocess.run([*g, "push", "-q", "origin", "HEAD:refs/heads/model-t"], check=True, capture_output=True)
    lap.make_branch()
    env = {**os.environ, "MOE_VM_SSH": str(lap.bin / "ssh"), "MOE_VM_SCP": str(lap.bin / "scp"),
           "MOE_VM_GH": str(lap.bin / "gh"), "MOE_VM_STATE_ROOT": str(lap.state), "MOE_VM_POLL_S": "0",
           "MOE_VM_WAIT_S": "5", "FAKE_VM": str(lap.vm), "FAKE_GH": str(lap.gh)}

    def run(*args):
        return subprocess.run(["bash", str(lap.clone / V.LAUNCHER), *args], env=env, capture_output=True, text=True,
                              timeout=120)
    assert run("prepare", "--ip", "1.2.3.4", "--run-id", "r", "--branch", "run-gh200-t").returncode == 0
    return lap, run, pages


def test_start_delivers_the_driver_the_final_plan_and_every_page_byte_for_byte(tmp_path):
    final = V.plan_text(_reg()["predictions"]["frozen"], draft=False)
    assert "# DRAFT" not in final
    lap, run, pages = _laptop(tmp_path, final)
    real = lap.bin / "ssh.real"
    (lap.bin / "ssh").rename(real)
    TG._exe(lap.bin / "ssh", f'#!/bin/bash\ncat > /dev/null\nexec "{real}" "$@"\n')
    vm = tmp_path / "vm-home"
    vm.mkdir()
    TG._exe(lap.bin / "scp", (
        '#!/bin/bash\nsrc="${@: -2:1}"; dst="${@: -1}"\n'
        '[[ "$dst" == *:* ]] || exit 0\n'
        f'rel="${{dst#*:}}"; [[ -n "$rel" ]] || rel="$(basename "$src")"\n'
        f'mkdir -p "$(dirname "{vm}/$rel")" && cp "$src" "{vm}/$rel"\nexit 0\n'))
    run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", V.PLAN)
    assert V.sha256(vm / PLAN.name) == hashlib.sha256(final.encode()).hexdigest() and V.sha256(vm / DRIVER.name) == V.sha256(DRIVER)
    assert len(pages) == 6
    for p in pages:
        assert (vm / p).is_file() and V.sha256(vm / p) == V.sha256(REPO / p), p
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "gh200_decode_session.sh --dry-run" in cmds and "gh200_decode_session.sh --setup" in cmds
    assert "PYTHONPATH" not in cmds


def test_start_refuses_a_draft_plan_and_starts_nothing(tmp_path):
    draft = V.plan_text(_reg()["predictions"]["frozen"], True)
    lap, run, _pages = _laptop(tmp_path, draft)
    got = run("start", "--ip", "1.2.3.4", "--run-id", "r", "--deadline", "2000000000", "--plan", V.PLAN)
    assert got.returncode == exit_codes.REFUSED and "carries a DRAFT line" in got.stdout + got.stderr
    assert V.NAME in got.stdout + got.stderr
    cmds = " ".join(json.dumps(c) for c in lap.vm_cmds())
    assert "--dry-run" not in cmds and "--setup" not in cmds


def test_no_em_dash_and_no_trailer_in_the_blind_files():
    files = [REPO / V.REG_JSON, REPO / V.REG_TXT, PLAN, LAUNCHER, Path(__file__),
             *(REPO / "scripts/scoring/v7dblind").glob("*.py"), *(REPO / V.PAGES).glob("*.json")]
    for f in files:
        t = f.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t, f
        assert ("Co-" + "Authored") not in t and ("Clau" + "de") not in t, f
