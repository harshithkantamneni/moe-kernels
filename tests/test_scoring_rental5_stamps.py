"""Rental 5's timestamps-v2 registration and scorer (scripts/scoring/rental5/reg_stamps2.py,
score_stamps2.py) on SYNTHETIC stamp pages only: no page of this kernel exists."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
R5 = REPO / "scripts" / "scoring" / "rental5"
sys.path.insert(0, str(REPO))


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RS = _load("rental5_reg_stamps2", R5 / "reg_stamps2.py")
SS = _load("rental5_score_stamps2", R5 / "score_stamps2.py")
import moe.instrumented as I  # noqa: E402

CARD = SS.C5.CARD


@pytest.fixture(scope="module")
def reg():
    return json.loads(json.dumps(RS.stamps2(), default=str))


def _repo(tmp_path, reg):
    d = tmp_path / "repo" / "docs" / "registered"
    d.mkdir(parents=True)
    (d / f"{SS.C5.NAMES['stamps2']}.json").write_text(json.dumps(reg))
    return tmp_path / "repo"


def test_the_registration_carries_the_owner_gate_and_the_mod_17_variants(reg):
    assert reg["name"] == "2026-10-07-rental5-stamps2-gh200"
    assert reg["gate"]["tolerance"] == {"median_max": 0.01, "worst_max": 0.02, "occupancy": "identical CTAs per SM"}
    assert reg["variants"]["texts"] == I.R5_VARIANTS
    assert "before any timed unit" in reg["gate"]["regcheck"]
    assert reg["K2"]["rivals_cycles"]["1"] == {"MVA2": 591.7, "LK": 350.5, "PS": 344.1, "OCC": 502.0}
    assert reg["K2"]["rivals_cycles"]["2"]["OCC"] == 804.0 and reg["K2"]["bin_minimum_ctas"] == 30
    assert reg["F2"]["delta_epi"]["rivals_cycles"] == {"NEAR": 260.6, "FAR": 470.3, "DRAM": 592.4}
    assert reg["D2"]["life"]["rivals_cycles"]["DRAM"] == 667.4
    assert reg["D2"]["dispatch"]["rivals_ns"]["2"] == {"DISP": 0.995, "MIX": 1.53, "SLOT": 2.025}
    assert reg["instrument"]["mark_k"]["mixtral-8x7b"]["w1"]["mark_k"] == [0, 15, 31, 47, 63]
    assert {u["drop_group"] for u in reg["units"].values()} == {"st", "stt", "std"}
    assert all("BLIND-CALMODEL" in v for v in reg["labels"].values())
    assert chr(0x2014) not in json.dumps(reg, ensure_ascii=False)


# --------------------------------------------------------------------------
# synthetic pages
# --------------------------------------------------------------------------

def _rows(S, T, P, epi, n_live, n_dead=0, mods=(1, 1, 1), every=16, smid=None, dead_t0=None):
    """Live CTA i: start 10 i, tops at start + P + k T, end top(S - 1) + T + epi; dead CTAs
    after them. Sampling as the copy samples (cta, iter, dead moduli)."""
    ks = I.sample_ks(S, every)
    cm, im, dm = mods
    out = []
    for pid in range(n_live + n_dead):
        r = [-1] * (I.HDR + 2 * len(ks))
        if pid < n_live:
            if pid % cm:
                out.append(r)
                continue
            s = 10 * pid
            r[0] = pid if smid is None else smid(pid)
            r[1], r[2] = s, s
            if pid % im == 0:
                for j, k in enumerate(ks):
                    r[I.HDR + 2 * j] = s + P + k * T
            end = s + P + (S - 1) * T + T + epi
            r[7], r[8], r[9] = end, end, 1
        else:
            if pid % dm:
                out.append(r)
                continue
            t = (dead_t0 or 0) + (pid - n_live)
            r[0], r[1], r[2], r[7], r[8], r[9] = 0, t, t, t + 1, t + 150, 2
        out.append(r)
    return np.array(out, dtype=np.int64), ks


def _unit(tree, model, label, launches, vid, spec, sass_ok=True):
    d = tree / f"2026-10-07-{CARD}-instr-{model}-{label}"
    (d / "stamps").mkdir(parents=True)
    recs = []
    for i, (arr, ks, extra) in enumerate(launches):
        name = f"l{i}.npy"
        np.save(d / "stamps" / name, arr)
        recs.append({"file": name, "marks": len(ks), "mark_k": ks, **extra})
    chk = {"start_before_first_ldg": sass_ok, "epi_bracket": True, "loop_clock_reads": 4}
    man = {"variant": vid, "spec_text": I.parse_spec(spec).text(), "launches": recs,
           "sass": [{"MUL_ROUTED_WEIGHT": False, "checks": chk}, {"MUL_ROUTED_WEIGHT": True, "checks": chk}]}
    (d / "stamps.json").write_text(json.dumps(man))
    (d / "CHOSEN_VARIANT.txt").write_text(f"id={vid}\nspec={spec}\n")
    return d


def _envs(tree, gates: dict, regs: dict, variants: list):
    p = tree / f"2026-10-07-{CARD}-perturb-pt"
    p.mkdir(parents=True, exist_ok=True)
    (p / "gate.env").write_text("".join(f"GATE_{SS.key_of(k)}={v} median=0.001 worst=0.002\n" for k, v in gates.items()))
    r = tree / f"2026-10-07-{CARD}-regcheck-rc"
    r.mkdir(parents=True, exist_ok=True)
    (r / "regcheck.env").write_text("".join(f"REGCHECK_{SS.key_of(k)}={v}\n" for k, v in regs.items()))
    (tree / "instr-variants.json").write_text(json.dumps(variants))


def _stf(tree, vid="u4-stf-v2", spec=None, gates=None, regs=None):
    spec = spec or I.R5_VARIANTS["v2"]
    launches = []
    for n in (2, 4, 6):
        # w1: S 64, T 1720 (5 x 344), epi 100, T_fix = lifetime - S T = P + epi = 2600 (PS); w2: S 224,
        # T 1376, epi 570, T_fix 3916 (PS); Delta_epi 470 (FAR)
        a, ks = _rows(64, 1720, 2500, 100, 1500, mods=(1, 17, 17))
        launches.append((a, ks, {"n": n, "gemm": "w1", "arm": "native"}))
        b, ks2 = _rows(224, 1376, 3346, 570, 1200, mods=(1, 17, 17))
        launches.append((b, ks2, {"n": n, "gemm": "w2", "arm": "native"}))
    _unit(tree, "mixtral-8x7b", "stf", launches, vid, spec)
    ids = ["u4-stf-v1", "u4-stf-v2", "u4-stf-v3"]
    _envs(tree, gates or {"u4-stf-v1": "FAIL", "u4-stf-v2": "PASS", "u4-stf-v3": "PASS"},
          regs or {i: "PASS" for i in ids},
          [{"id": i, "kind": "stamps", "model": "mixtral-8x7b"} for i in ids])


def test_zero_pages_score_every_block_not_scored(tmp_path, reg):
    res = SS.score(_repo(tmp_path, reg), tmp_path / "tree")
    assert all(v["use"].startswith("NOT SCORED") for v in res["units"].values())
    assert res["F2"]["verdict"] == "NOT SCORED" and res["K2"]["verdict"].startswith("NOT SCORED")
    assert all(v["verdict"] == "NOT SCORED" for v in res["D2"].values())


def test_f2_reads_delta_epi_and_t_fix_off_the_sample_marks(tmp_path, reg):
    tree = tmp_path / "tree"
    _stf(tree)
    res = SS.score(_repo(tmp_path, reg), tree)
    assert res["units"]["stf"]["use"].startswith("counted (u4-stf-v2")
    de = res["F2"]["delta_epi"]
    assert de["cycles"] == pytest.approx(470.0) and de["verdict"] == "FAR"
    assert res["F2"]["t_fix"]["w1"]["cycles"] == pytest.approx(2600.0) and res["F2"]["t_fix"]["w1"]["verdict"] == "PS"
    assert res["F2"]["t_fix"]["w2"]["verdict"] == "PS"
    out = tmp_path / "out"
    text = SS.C5.write_score(out, "stamps2", res, SS.lines(res))
    assert "F2 Delta_epi: FAR" in text and (out / "stamps2.score.json").exists()


def test_a_variant_other_than_the_registered_choice_is_not_scored(tmp_path, reg):
    tree = tmp_path / "tree"
    # v1 passes both, so the rule chooses v1; the unit ran v2
    _stf(tree, gates={"u4-stf-v1": "PASS", "u4-stf-v2": "PASS", "u4-stf-v3": "PASS"})
    res = SS.score(_repo(tmp_path, reg), tree)
    assert "variant mismatch" in res["units"]["stf"]["use"]
    tree2 = tmp_path / "tree2"
    _stf(tree2, regs={"u4-stf-v1": "PASS", "u4-stf-v2": "FAIL x", "u4-stf-v3": "PASS"})
    assert "regcheck FAIL" in SS.score(_repo(tmp_path / "b", reg), tree2)["units"]["stf"]["use"]


def test_a_unit_never_run_names_its_failed_lines(tmp_path, reg):
    tree = tmp_path / "tree"
    ids = ["u5-stt-v1", "u5-stt-v2"]
    _envs(tree, {i: "FAIL" for i in ids}, {i: "PASS" for i in ids},
          [{"id": i, "kind": "stamps", "model": "mixtral-8x7b"} for i in ids])
    res = SS.score(_repo(tmp_path, reg), tree)
    assert "NOT RUN" in res["units"]["stt"]["use"]


def test_k2_residency_bins_and_the_class_rule(reg):
    # two CTAs on one SM for the whole window: j = 2
    a, ks = _rows(64, 688, 0, 0, 2, smid=lambda pid: 0)
    a[1, 1:3] = a[0, 1:3]
    a[1, I.HDR::2] = a[0, I.HDR::2]
    a[1, 7:9] = a[0, 7:9]
    cols = I.decode(a, len(ks), ks)
    w = SS.windows(cols)
    assert w and all(abs(j - 2) < 0.05 for _r, j, _i in w) and all(r == 688 for r, _j, _i in w)
    v = SS.rj_verdict(reg, {1: 345.0, 2: 690.0, 5: 1721.0, 4: 1377.0})
    assert v["verdict"] == "SELECTED PS~LK"
    assert SS.rj_verdict(reg, {1: 547.0, 2: 700.0})["verdict"] == "UNDECIDED"
    assert SS.rj_verdict(reg, {2: 690.0})["verdict"].startswith("NOT SCORED")


def test_d2_dispatch_and_kappa_with_the_separation_rule(reg):
    # 1000 live CTAs ending by 10 x 999 + ..., dead sampled every 17 pids after them
    a, ks = _rows(32, 10, 0, 0, 400, n_dead=17 * 600, mods=(1, 1, 17), dead_t0=0)
    cols = I.decode(a, len(ks), ks)
    last = cols["end_ns"][cols["kind"] == 1].max()
    r = SS.dead_reads(cols, 17)
    assert r["hidden"] == int(((cols["kind"] == 2) & (cols["start_ns"] <= last)).sum())
    assert r["dispatch"] == pytest.approx(1.0) and r["qualifying"] >= 100
    assert r["life"] == 150
    # 0.31 sits between D2 0.30 and D 0.325, within 2 sigma of both: NOT SEPARATED
    v = SS.separated(0.31, 0.02, reg["D2"]["hiding"]["rivals"])
    assert v["verdict"].startswith("NOT SEPARATED") and {v["nearest"], v["second"]} == {"D2", "D"}
    assert SS.separated(0.0, 0.02, reg["D2"]["hiding"]["rivals"])["verdict"] == "K0"
    # too few sampled dead CTAs after the last live end: no dispatch reading
    b, ks = _rows(32, 10, 0, 0, 400, n_dead=17 * 50, mods=(1, 1, 17), dead_t0=10 ** 9)
    assert SS.dead_reads(I.decode(b, len(ks), ks), 17)["dispatch"] is None


def test_a_failed_sass_precondition_is_not_scored(tmp_path, reg):
    tree = tmp_path / "tree"
    a, ks = _rows(32, 10, 0, 0, 10)
    _unit(tree, "olmoe-1b-7b", "std4", [(a, ks, {"n": 2, "gemm": "w1", "arm": "shared"})], "u8-std4-v2",
          I.R5_VARIANTS["v2"], sass_ok=False)
    _envs(tree, {"u8-std4-v2": "PASS"}, {"u8-std4-v2": "PASS"},
          [{"id": "u8-std4-v2", "kind": "stamps", "model": "olmoe-1b-7b"}])
    res = SS.score(_repo(tmp_path, reg), tree)
    assert "SASS precondition" in res["units"]["std4"]["use"]


def test_the_k2_minimum_counts_ctas_not_windows_and_the_occupancy_bins_are_registered(tmp_path, reg):
    """build-r5-review F7: one CTA gives many 16-step windows; 20 CTAs of 64 k-steps (3 windows
    each, 60 windows) stay under the 30-CTA minimum. The occupancy bins and Z's divisor are
    the registration's, and the tail rule is registered with its sampled counts."""
    assert reg["K2"]["occ_bins"] == [4, 5] and reg["K2"]["occ_z"] == 5
    tail = reg["K2"]["tail_ctas"]
    assert tail["stt8x22 n1 w2"]["tail_ctas"] == 240 and tail["stt8x22 n1 w2"]["sampled_1_in_17"] < 30
    assert "stt n1 w1" in reg["K2"]["tail_printed_only"] and "stt n1 w2" not in reg["K2"]["tail_printed_only"]
    assert "printed only" in reg["K2"]["tail_rule"].lower()
    tree = tmp_path / "tree"
    a, ks = _rows(64, 344, 0, 0, 20, smid=lambda pid: pid)   # one CTA per SM: j = 1
    _unit(tree, "mixtral-8x7b", "stt", [(a, ks, {"n": 1, "gemm": "w1", "arm": "native"})], "u5-stt-v1", I.R5_VARIANTS["v1"])
    _envs(tree, {"u5-stt-v1": "PASS"}, {"u5-stt-v1": "PASS"}, [{"id": "u5-stt-v1", "kind": "stamps", "model": "mixtral-8x7b"}])
    res = SS.score(_repo(tmp_path, reg), tree)
    b = res["K2"]["bins"]["1"]
    assert b["ctas"] == 20 and b["windows"] == 60 and b["use"].startswith("NOT SCORED: 20 CTAs")
    tree2 = tmp_path / "tree2"
    a, ks = _rows(64, 344, 0, 0, 40, smid=lambda pid: pid)
    _unit(tree2, "mixtral-8x7b", "stt", [(a, ks, {"n": 1, "gemm": "w1", "arm": "native"})], "u5-stt-v1", I.R5_VARIANTS["v1"])
    _envs(tree2, {"u5-stt-v1": "PASS"}, {"u5-stt-v1": "PASS"}, [{"id": "u5-stt-v1", "kind": "stamps", "model": "mixtral-8x7b"}])
    res2 = SS.score(_repo(tmp_path / "b", reg), tree2)
    assert res2["K2"]["bins"]["1"]["use"] == "counted" and res2["K2"]["bins"]["1"]["R"] == 344


def test_v2_tail_bins_of_the_stt_units_are_printed_only(tmp_path, reg):
    tree = tmp_path / "tree"
    a, ks = _rows(64, 344, 0, 0, 17 * 40, mods=(1, 17, 17), smid=lambda pid: pid)
    _unit(tree, "mixtral-8x7b", "stt", [(a, ks, {"n": 1, "gemm": "w1", "arm": "native"})], "u5-stt-v2", I.R5_VARIANTS["v2"])
    _envs(tree, {"u5-stt-v1": "FAIL", "u5-stt-v2": "PASS"}, {"u5-stt-v1": "PASS", "u5-stt-v2": "PASS"},
          [{"id": i, "kind": "stamps", "model": "mixtral-8x7b"} for i in ("u5-stt-v1", "u5-stt-v2")])
    res = SS.score(_repo(tmp_path, reg), tree)
    b = res["K2"]["bins"]["1"]
    assert "printed_only" in b and "ctas" not in b and res["K2"]["verdict"].startswith("NOT SCORED")


def test_dispatch_and_kappa_are_not_scored_when_live_ctas_are_sampled(tmp_path, reg):
    """build-r5-review F2: under v3 (cta_mod 17) the last live end is read off sampled CTAs."""
    assert "cta_mod != 1" in reg["D2"]["dispatch"]["cta_mod_rule"]
    tree = tmp_path / "tree"
    a, ks = _rows(32, 10, 0, 0, 17 * 30, n_dead=17 * 600, mods=(17, 17, 17), dead_t0=0)
    _unit(tree, "olmoe-1b-7b", "std4", [(a, ks, {"n": 2, "gemm": g, "arm": "shared"}) for g in ("w1", "w2")],
          "u8-std4-v3", I.R5_VARIANTS["v3"])
    ids = ["u8-std4-v2", "u8-std4-v3"]
    _envs(tree, {"u8-std4-v2": "FAIL", "u8-std4-v3": "PASS"}, {i: "PASS" for i in ids},
          [{"id": i, "kind": "stamps", "model": "olmoe-1b-7b"} for i in ids])
    res = SS.score(_repo(tmp_path, reg), tree)
    e = res["D2"]["std4"]["w1"]
    assert e["dispatch"]["verdict"].startswith("NOT SCORED: the variant samples live CTAs")
    assert e["kappa"]["verdict"].startswith("NOT SCORED") and e["life"]["verdict"] != "NOT SCORED"
