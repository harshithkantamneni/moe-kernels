"""Rental 4's scorers (scripts/scoring/rental4/), written before any rental-4 page: each is
driven here on SYNTHETIC pages built in tmp_path, never on a published one, and must return
the verdict its registration's rule gives for the world planted. Also the registrations' own
consistency (register.py --check) and the gate rule's one implementation on both sides."""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
R4DIR = REPO / "scripts" / "scoring" / "rental4"
for p in (REPO, REPO / "scripts", R4DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import instr_probe as IP  # noqa: E402
import r4common as C4  # noqa: E402
import score_dead as SD  # noqa: E402
import score_hw as SH  # noqa: E402
import score_occlaw as SO  # noqa: E402
import score_perturb as SP  # noqa: E402
import score_stamps as SST  # noqa: E402

from moe import instrumented as I  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

CARD = C4.CARD
DAY = "2026-10-07"
DEAD = C4.registration(REPO, "dead")
OCCL = C4.registration(REPO, "occlaw")
PERT = C4.registration(REPO, "perturb")
STMP = C4.registration(REPO, "stamps")
HW = C4.registration(REPO, "hw")


# --------------------------------------------------------------------------
# the registrations
# --------------------------------------------------------------------------

def test_register_check_recomputes_every_committed_number():
    got = subprocess.run([sys.executable, str(R4DIR / "register.py"), "--check"], capture_output=True,
                         text=True, timeout=600, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin",
                                                      "HOME": str(Path.home())})
    assert got.returncode == 0, got.stdout + got.stderr
    assert got.stdout.count(": same") == 2 * len(C4.PARTS)


def test_the_registrations_carry_their_labels_status_and_no_typographic_dash():
    for part in C4.PARTS:
        for ext in ("json", "txt"):
            text = (REPO / "docs" / "registered" / f"{C4.NAMES[part]}.{ext}").read_text()
            assert "\u2014" not in text and "\u2013" not in text, (part, ext)
        doc = C4.registration(REPO, part)
        assert doc["status"].startswith("REGISTERED"), part
        assert doc["seen_data"], part
    assert "CUT by the owner" in DEAD["design"]
    assert "NO RENTAL-4 UNIT USES IT" in C4.registration(REPO, "nativegates")["status"]


def test_the_dead_predictions_reproduce_the_designs_and_the_cal_fit():
    assert DEAD["fit_D_CAL"]["d_ns"] == pytest.approx(0.995, abs=0.003)
    assert DEAD["fit_D_CAL"]["kappa"] == pytest.approx(0.325, abs=0.003)
    a1, a2 = DEAD["predictions_us"]["A1"]["rivals"], DEAD["predictions_us"]["A2"]["rivals"]
    assert a1["D"]["total"] == pytest.approx(20.73, abs=0.05) and a1["M"]["total"] == pytest.approx(27.77, abs=0.05)
    assert a2["D"]["total"] == pytest.approx(23.69, abs=0.05) and a2["SLOT"]["total"] == pytest.approx(21.70, abs=0.05)
    assert a1["K0"]["total"] == pytest.approx(24.43, abs=0.05) and a1["D2"]["total"] == pytest.approx(21.46, abs=0.05)
    assert "A3" not in DEAD["predictions_us"]


def test_the_laws_reproduce_the_reanalysis_fit_on_the_seen_points():
    rms = OCCL["laws_on_seen_points"]["rms_by_law"]
    assert rms["MVA2"]["points"] == 10 and rms["MVA2"]["rms"] == pytest.approx(0.0128, abs=0.0005)
    assert rms["LK"]["rms"] == pytest.approx(0.0553, abs=0.0005)


def test_one_gate_rule_on_both_sides_and_the_owners_tolerance():
    assert PERT["tolerance"] == IP.PERTURB_TOL == {"median_max": 0.01, "worst_max": 0.02, "occupancy": "identical"}
    rng = random.Random(4)
    for _ in range(100):
        r = [1 + rng.gauss(0, 0.012) for _ in range(rng.randint(1, 8))]
        cfg = [{"config": [1], "equal": rng.choice([True, True, False, None]), "plain": {}, "copy": {}}]
        kw = dict(upstream_ok=rng.random() < 0.9, sass_equal=rng.choice([True, False, None]),
                  hint_ok=rng.choice([None, True, False]))
        assert IP.gate_verdict(r, cfg, **kw) == C4.perturb_verdict(r, cfg, **kw)
    assert "eviction_hints" in PERT and "DEFERRED" in PERT["eviction_hints"]["deferred"]
    assert "evict" not in C4.PARTS and not (REPO / "scripts/scoring/rental4/score_evict.py").exists()


# --------------------------------------------------------------------------
# dead: synthetic R3 pages
# --------------------------------------------------------------------------

def put_timed(tree, model, label, copies, ms_of, align_of, *, fail=()):
    d = tree / f"gaps-{CARD}-{model}-{label}" / "private_weight_reference" / f"run-{label}"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"arm": a, "tiles": n, "ms_p50": ms_of(a, n)} for a in ("native", "shared", "private")
            for n in range(1, 10)]
    align = [{"label": a, "tread": n, "repeat": r, "ms": align_of(a, n)} for a in ("native", "shared", "private")
             for n in range(1, 10) for r in range(2)]
    gates = [{"tag": f"V{i}", "kind": "VALIDITY", "verdict": "FAIL" if f"V{i}" in fail else "PASS"} for i in range(9)]
    (d / "report.json").write_text(json.dumps({"session_tag": f"gh200-x-{label}-lock1710", "copies_declared": copies,
                                               "pinned": {"GROUP_SIZE_M": 8}, "treads_table": rows,
                                               "align_probe": {"cells": align}, "gates": gates}))


GAP9_MS = 0.030   # SHARED - NATIVE at 9 copies, the planted world's


def _nat(n):
    return 0.30 + 0.05 * n


def dead_world(tree, deltas, *, native_drift=0.0, c15_declares=15, align_extra_us=0.3):
    """Pages whose Delta (less the alignment growth) is `deltas[contrast]` us."""
    for name, v in DEAD["predictions_us"].items():
        model, c15 = v["model"], v["copies"][1]
        put_timed(tree, model, "c9", 9,
                  lambda a, n: _nat(n) + (GAP9_MS if a == "shared" else 0.02 if a == "private" else 0),
                  lambda a, n: 0.0055 if a == "native" else 0.0059)
        d = (deltas[name] + align_extra_us) / 1e3
        put_timed(tree, model, f"c{c15}", c15_declares,
                  lambda a, n, d=d: _nat(n) * (1 + native_drift)
                  + (GAP9_MS + d if a == "shared" else 0.02 if a == "private" else 0),
                  lambda a, n: 0.0055 if a == "native" else 0.0059 + align_extra_us / 1e3)


def rival(name):
    return {k: v["rivals"][name]["total"] for k, v in DEAD["predictions_us"].items()}


def test_dead_at_ds_prediction_holds_d_and_excludes_m_fixed_and_k0(tmp_path):
    dead_world(tmp_path, rival("D"))
    res = SD.score(REPO, tmp_path)
    v = res["verdicts"]
    assert v["D"]["verdict"] == "HOLDS" and v["M"]["verdict"].startswith("EXCLUDED")
    assert v["FIXED"]["verdict"].startswith("EXCLUDED") and v["K0"]["verdict"] == "EXCLUDED on ['A1']"
    assert v["SLOT"]["verdict"] == "NOT EXCLUDED" and v["D_vs_SLOT"]["verdict"] == "UNDECIDED"
    assert v["K5"]["verdict"].startswith("NOT TESTED")
    c = res["contrasts"]["A1"]
    assert c["Delta_us"] == pytest.approx(rival("D")["A1"], abs=1e-6)
    assert c["Delta_raw_us"] == pytest.approx(rival("D")["A1"] + 0.3, abs=1e-6)
    assert c["null_control"]["verdict"] == "PASS" and "NOT JOINABLE" in c["label"]
    assert res["per_gemm_d_printed"]["d_w2_ns"] == pytest.approx(rival("D")["A1"] / (52080 - 31248) * 1e3)


def test_dead_at_ms_prediction_fails_d(tmp_path):
    dead_world(tmp_path, rival("M"))
    v = SD.score(REPO, tmp_path)["verdicts"]
    assert v["D"]["verdict"] == "FAILS on ['A1', 'A2']" and v["M"]["verdict"] == "NOT EXCLUDED"


def test_a_drifting_native_turns_every_verdict_inconclusive(tmp_path):
    dead_world(tmp_path, rival("D"), native_drift=0.012)
    v = SD.score(REPO, tmp_path)["verdicts"]
    assert all(x["verdict"].startswith("INCONCLUSIVE") for x in v.values())


def test_a_page_declaring_the_wrong_copies_is_not_scored(tmp_path):
    dead_world(tmp_path, rival("D"), c15_declares=9)
    res = SD.score(REPO, tmp_path)
    assert res["contrasts"]["A1"]["verdict"].startswith("NOT SCORED")
    assert res["verdicts"]["D"]["verdict"].startswith("NOT SCORED")


def test_a_gate_failed_page_leaves_clean_and_a1_alone_carries_clean(tmp_path):
    dead_world(tmp_path, rival("D"))
    d = next(tmp_path.glob(f"gaps-{CARD}-olmoe-1b-7b-c15/private_weight_reference/*"))
    j = json.loads((d / "report.json").read_text())
    j["gates"][7]["verdict"] = "FAIL"
    (d / "report.json").write_text(json.dumps(j))
    res = SD.score(REPO, tmp_path)
    assert res["addendum"]["CLEAN"]["contrasts"]["A2"]["verdict"].startswith("NOT SCORED")
    assert "Delta_us" in res["contrasts"]["A2"]          # ALL counts the page
    assert res["addendum"]["CLEAN"]["verdicts"]["D"]["z"].keys() == {"A1"}
    assert res["verdicts"]["D"]["verdict"] == "HOLDS"


# --------------------------------------------------------------------------
# occlaw: synthetic OLMoE G = 64 counter pages under a planted law
# --------------------------------------------------------------------------

OLMOE = MODEL_CONFIGS["olmoe-1b-7b"]


def occ_page(label, law, *, occ=None, dram_frac=0.2, design=None):
    bk, stg = C4.KNOBS[label]
    occ = occ or {"w1": C4.OCC[label][0], "w2": C4.OCC[label][1]}
    E = OLMOE.num_experts
    K = {"w1": OLMOE.hidden_size, "w2": OLMOE.intermediate_size}
    npn = {"w1": 2 * OLMOE.intermediate_size // 64, "w2": OLMOE.hidden_size // 64}
    cells = []
    for arm, D in (("native", E), ("shared", 9 * E), ("private", 9 * E)):
        for n in range(1, 10):
            pg, rec = {}, {}
            for g in ("w1", "w2"):
                R = math.ceil((E * n * 32 + D * 31) / 32)
                grid = R * npn[g]
                live = E * n * npn[g]
                c = C4.law_c(law, OCCL["laws"][law], bk, stg, occ[g])
                cyc = (5000 if arm == "native" else 9000) + math.ceil(live / 132) * (K[g] // bk * c + C4.F_CTA[g])
                pg[g] = {"grid_size": grid, "gpu_time_ns": 1.0e6, "dram_bytes_read": dram_frac * 4022.0 * 1.0e6}
                rec[g] = {"launch__occupancy_limit_registers": occ[g], "launch__occupancy_limit_shared_mem": 9,
                          "launch__occupancy_limit_warps": 8, "launch__occupancy_limit_blocks": 32,
                          "sm__cycles_elapsed.avg": cyc}
            cells.append({"arm": arm, "n": n, "declared": D, "per_gemm": pg, "recorded": rec})
    return {"family": "r3-arms", "design": design or {"model": "olmoe-1b-7b", "block_m": 32, "block_n": 64,
                                                      "block_k": bk, "num_stages": stg, "group_m": 64},
            "card": {"slug": CARD, "sm_count": 132}, "cells": cells,
            "gates": [{"number": "V7", "kind": "VALIDITY", "verdict": "FAIL"}]}


def put_bytes(tree, model, label, page, G):
    d = tree / f"{DAY}-{CARD}-{model}-{label}-r3-counters" / "lock1710"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"r3c-g{G}.json").write_text(json.dumps(page))


@pytest.mark.parametrize("law", ["MVA2", "LK"])
def test_occlaw_selects_the_law_its_pages_were_built_under(tmp_path, law):
    for label in C4.KNOBS:
        put_bytes(tmp_path, "olmoe-1b-7b", label, occ_page(label, law), 64)
    res = SO.score(REPO, tmp_path)
    assert res["verdict"] == f"SELECTED {law}", (res["verdict"], {k: v["status"] for k, v in res["laws"].items()})
    assert res["laws"][law]["rms"] == pytest.approx(0, abs=1e-5)   # the registered ratios carry 6 decimals
    assert res["laws"]["PS"]["status"] == "FALSIFIED"


def test_occlaw_rekeys_a_page_whose_recorded_occupancy_is_not_the_registered(tmp_path):
    for label in C4.KNOBS:
        occ = {"w1": 4, "w2": 4} if label == "k64s6" else None
        put_bytes(tmp_path, "olmoe-1b-7b", label, occ_page(label, "MVA2", occ=occ), 64)
    res = SO.score(REPO, tmp_path)
    assert res["pages"]["k64s6"]["w1"]["rekeyed"] is True
    assert res["laws"]["MVA2"]["status"] == "consistent"


def test_occlaw_reads_no_dram_bound_cell_and_refuses_a_wrong_design(tmp_path):
    for label in C4.KNOBS:
        page = occ_page(label, "MVA2", dram_frac=0.8 if label == "k32s4" else 0.2)
        if label == "k128s4":
            page["design"]["block_k"] = 64
        put_bytes(tmp_path, "olmoe-1b-7b", label, page, 64)
    res = SO.score(REPO, tmp_path)
    assert res["pages"]["k32s4"]["w1"]["use"].startswith("NOT SCORED")
    assert res["pages"]["k128s4"]["use"].startswith("NOT SCORED")


def test_occlaw_without_its_base_is_not_scored(tmp_path):
    put_bytes(tmp_path, "olmoe-1b-7b", "k32s4", occ_page("k32s4", "MVA2"), 64)
    assert SO.score(REPO, tmp_path)["verdict"].startswith("NOT SCORED")


# --------------------------------------------------------------------------
# perturb
# --------------------------------------------------------------------------

def put_perturb(tree, ratios, *, occ_copy=(5, 4), upstream=True, vm=None, sass=True, hint=None):
    d = tree / f"{DAY}-{CARD}-perturb-pt"
    d.mkdir(parents=True, exist_ok=True)
    f = {(32, 64, 64, 8, 4, 8, mrw, 8): {"regs": 48 + 7 * mrw, "shared": 36864, "num_warps": 8,
                                        "occupancy": {"ctas_per_sm": o}} for mrw, o in ((False, 5), (True, 4))}
    c = {k: dict(v, occupancy={"ctas_per_sm": o}) for (k, v), o in zip(f.items(), occ_copy, strict=True)}
    variants = {vid: {"kind": "stamps", "cells": [{"ratio": x} for x in rs], "configs": IP.compare_configs(f, c),
                      "sass_equal": sass, "hint_ok": hint} for vid, rs in ratios.items()}
    (d / "perturb.json").write_text(json.dumps({"upstream": {"file_matches": upstream, "excerpt_matches": True},
                                                "variants": variants}))
    vm = vm or {}
    (d / "gate.env").write_text("".join(f"GATE_{IP.gate_key(k)}={vm.get(k, 'PASS')} x\n" for k in ratios))


def test_perturb_passes_within_tolerance_and_names_the_void_units(tmp_path):
    put_perturb(tmp_path, {"u8-stf": [1.004, 0.997, 1.008, 1.001], "u9-stk64s4": [1.004, 1.025]},
                vm={"u9-stk64s4": "FAIL"})
    res = SP.score(REPO, tmp_path)
    assert res["variants"]["u8-stf"]["verdict"] == "PASS"
    assert res["variants"]["u9-stk64s4"]["verdict"] == "FAIL" and res["void_units"] == ["u9-stk64s4"]
    assert all(v["agrees_with_vm"] for v in res["variants"].values())


@pytest.mark.parametrize("kw, why", [({"occ_copy": (4, 4)}, "config"), ({"upstream": False}, "upstream"),
                                     ({"sass": False}, "SASS differs"), ({"sass": None}, "SASS leg unread"),
                                     ({"hint": False}, "dropped the eviction hint")])
def test_perturb_fails_on_occupancy_or_a_foreign_install(tmp_path, kw, why):
    put_perturb(tmp_path, {"u8-stf": [1.0, 1.0]}, **kw)
    v = SP.score(REPO, tmp_path)["variants"]["u8-stf"]
    assert v["verdict"] == "FAIL" and any(why in w for w in v["why"])
    assert v["agrees_with_vm"] is False


def test_perturb_median_rule(tmp_path):
    put_perturb(tmp_path, {"u8-stf": [1.012, 1.013, 1.011, 0.99]})
    v = SP.score(REPO, tmp_path)["variants"]["u8-stf"]
    assert v["verdict"] == "FAIL" and "median" in v["why"][0]


# --------------------------------------------------------------------------
# stamps: synthetic per-CTA rows
# --------------------------------------------------------------------------

def rows(n_live, *, prologue=300, loop_iters=0, it=0, epi=200, start_gap_ns=2.0, n_dead=0, life_dead=451,
         dead_gap_ns=1.53, marks=0, tail_from=None, tail_it=None):
    out = []
    for i in range(n_live):
        s_ns, s_clk = 1000 + i * start_gap_ns, 10_000 + i * 3
        lit = tail_it if (tail_from is not None and i >= tail_from) else it
        loop = loop_iters * lit
        r = [i % 132, s_ns, s_clk, s_ns + 100, s_clk + prologue, s_ns + 200, s_clk + prologue + loop,
             s_ns + 300, s_clk + prologue + loop + epi, 1]
        tops = [s_clk + prologue + k * lit if k < loop_iters else -1 for k in range(marks)]
        ends = [t + lit - 5 if t >= 0 else -1 for t in tops]
        r += [x for pair in zip(tops, ends, strict=True) for x in pair]
        out.append(r)
    for j in range(n_dead):
        s_ns = 50_000 + j * dead_gap_ns
        out.append([j % 132, s_ns, 90_000, -1, -1, -1, -1, s_ns + 300, 90_000 + life_dead, 2] + [-1] * (2 * marks))
    return np.array(out, dtype=np.int64)


GOOD_SASS = {"start_before_first_ldg": True, "loop_clock_reads": 2, "epi_bracket": True}


def put_stamps(tree, model, label, vid, launches, *, gate="PASS", sass=None):
    d = tree / f"{DAY}-{CARD}-instr-{model}-{label}"
    (d / "stamps").mkdir(parents=True, exist_ok=True)
    idx = []
    for i, (meta, arr) in enumerate(launches):
        name = f"l{i}.npy"
        np.save(d / "stamps" / name, arr)
        idx.append({**meta, "file": name, "call": meta.get("call", 0)})
    sass = GOOD_SASS if sass is None else sass
    rows_s = [{"config": [mrw], "MUL_ROUTED_WEIGHT": mrw, "file": "x.sass",
               "checks": dict(sass, epi_bracket=sass["epi_bracket"] if mrw else None)} for mrw in (False, True)]
    (d / "stamps.json").write_text(json.dumps({"variant": vid, "launches": idx, "sass": rows_s}))
    pd = tree / f"{DAY}-{CARD}-perturb-pt"
    pd.mkdir(parents=True, exist_ok=True)
    with open(pd / "gate.env", "a") as f:
        f.write(f"GATE_{IP.gate_key(vid)}={gate} x\n")


def stamps_world(tree, *, far=433.3, law="MVA2", tail_ns=1070.0, life=451, dead_gap=1.53, gate="PASS",
                 stf_sass=None):
    put_stamps(tree, "mixtral-8x7b", "stf", "u8-stf",
               [({"G": 8, "arm": "native", "n": n, "gemm": g, "marks": 0},
                 rows(400, epi=200 + (far if g == "w2" else 0))) for n in (2, 4, 6) for g in ("w1", "w2")],
               gate=gate, sass=stf_sass)
    kp = STMP["K"]["predicted_T_iter_cycles"]
    for i, lab in enumerate(("stk64s4", "stk32s4", "stk128s4", "stk64s8")):
        L = []
        for n in (4, 6):
            for g in ("w1", "w2"):
                t = int(round(kp[lab][g]["T_iter_cycles"][law]))
                iters = kp[lab][g]["iterations"]
                L.append(({"G": 64, "arm": "native", "n": n, "gemm": g, "marks": 64},
                          rows(300, loop_iters=iters, it=t, marks=64)))
        put_stamps(tree, "olmoe-1b-7b", lab, f"u{9 + i}-{lab}", L)
    tcyc = int(round(tail_ns * C4.LOCK_MHZ / 1e3))
    for model, vid in (("mixtral-8x22b", "u13-sttail"), ("mixtral-8x7b", "u14-sttail")):
        L = [({"G": 8, "arm": "native", "n": 1, "gemm": g, "marks": 112},
              rows(132 * SST.OCC_T[g] * 2 + 60, loop_iters=40, it=400, marks=112,
                   tail_from=132 * SST.OCC_T[g] * 2, tail_it=tcyc))
             for g in ("w1", "w2")]
        put_stamps(tree, model, "sttail", vid, L)
    for lab, vid in (("stdead4", "u15-stdead4"), ("stdead2", "u16-stdead2")):
        L = [({"G": 8, "arm": a, "n": n, "gemm": g, "marks": 0},
              rows(50, n_dead=500, life_dead=life, dead_gap_ns=dead_gap))
             for a in ("native", "shared") for n in (2, 4) for g in ("w1", "w2")]
        put_stamps(tree, "olmoe-1b-7b", lab, vid, L)


def test_stamps_read_the_planted_world(tmp_path):
    stamps_world(tmp_path)
    res = SST.score(REPO, tmp_path)
    u = res["units"]
    assert u["stf"]["verdict"] == "FAR" and u["stf"]["delta_epi_cycles"] == pytest.approx(433, abs=1)
    assert res["K_laws"]["MVA2"]["status"] == "consistent" and res["K_laws"]["PS"]["status"] == "FALSIFIED"
    assert res["K_classes"]["LK=LITTLE"]["status"] == "FALSIFIED"
    assert res["K_verdict"]["verdict"] == "SELECTED class MVA2~OCC"   # never MVA2 alone
    assert u["sttail-mixtral-8x22b"]["verdict"] == "R1" and u["sttail-mixtral-8x7b"]["verdict"] == "R1"
    assert u["stdead2"]["w1"]["life"]["verdict"] == "MIX" and u["stdead2"]["w1"]["dispatch"]["verdict"] == "MIX"
    assert res["F_rivals"]["source"].startswith("the published RRZE")


def test_stamps_take_this_rentals_gpubench_latencies_and_refuse_an_ungated_unit(tmp_path):
    stamps_world(tmp_path, far=600)
    d = tmp_path / f"{DAY}-{CARD}-gpubench-r4"
    d.mkdir()
    (d / "constants.json").write_text(json.dumps({"cycles_at_lock": {"near_l2_ns": 240.0, "far_l2_ns": 420.0,
                                                                      "dram_ns": 600.0}}))
    res = SST.score(REPO, tmp_path)
    assert res["F_rivals"]["source"] == "this rental's gpubench constants" and res["units"]["stf"]["verdict"] == "DRAM"
    t2 = tmp_path / "other"
    stamps_world(t2, gate="FAIL")
    res = SST.score(REPO, t2)
    assert res["units"]["stf"]["use"].startswith("NOT SCORED: the perturb gate")


def test_k_classes_merge_what_the_cells_cannot_separate():
    k = STMP["K"]
    assert k["classes"] == {"PS": ["PS"], "LK=LITTLE": ["LK", "LITTLE"], "MVA2~OCC": ["MVA2", "OCC"]}
    p = k["predicted_T_iter_cycles"]
    for lab in p:
        for g in ("w1", "w2"):
            t = p[lab][g]["T_iter_cycles"]
            assert t["LK"] == pytest.approx(t["LITTLE"])
            if "OCC" in t:
                assert abs(t["OCC"] / t["MVA2"] - 1) < 0.06
    assert "SEEN" in k["labels"]["stk64s4"] and "ONE test" in OCCL["one_test"]


@pytest.mark.parametrize("bad", [{"epi_bracket": False}, {"start_before_first_ldg": False}, None])
def test_a_unit_whose_sass_misplaces_the_clock_reads_is_not_scored(tmp_path, bad):
    sass = dict(GOOD_SASS, **bad) if bad else {"start_before_first_ldg": None, "loop_clock_reads": None,
                                                  "epi_bracket": None}
    stamps_world(tmp_path, stf_sass=sass)
    u = SST.score(REPO, tmp_path)["units"]["stf"]
    assert u["use"].startswith("NOT SCORED: the SASS precondition fails"), u


def test_stamps_tail_reads_the_final_partial_wave_and_the_device_share_rival(tmp_path):
    stamps_world(tmp_path, tail_ns=900.0)
    u = SST.score(REPO, tmp_path)["units"]["sttail-mixtral-8x7b"]
    x = next(v for k, v in u.items() if k.startswith("w1-"))
    assert x["tail_ctas"] == 60 and "FS" in x["rivals"]
    assert x["rivals"]["FS"] == pytest.approx(C4.STAGE_BYTES * 60 / 3598e9 * 1e9)


def test_dispatch_reads_only_dead_ctas_after_the_live_drain_and_needs_timer_ticks():
    import numpy as np
    from moe import instrumented as I2
    arr = rows(50, n_dead=200, dead_gap_ns=1.0)
    arr[50:150, 1] = 1100          # half the dead CTAs start inside the live drain
    L = {"cols": I2.decode(arr, 0)}
    r = C4.dead_dispatch_ns(L)
    assert r["qualifying"] == 100 and r["value"] == pytest.approx(1.0, abs=0.02)
    coarse = rows(50, n_dead=200, dead_gap_ns=1.0)
    coarse[50:, 1] = (coarse[50:, 1] // 1000) * 1000      # a microsecond timer
    r = C4.dead_dispatch_ns({"cols": I2.decode(coarse, 0)})
    assert r["value"] is None and "timer ticks" in r["why"]
    assert C4.dead_dispatch_ns({"cols": I2.decode(rows(50, n_dead=20), 0)})["value"] is None
    assert np.isfinite(C4.timer_resolution_ns({"cols": I2.decode(arr, 0)}))


def test_stamps_dead_ctas_at_the_dispatch_rate(tmp_path):
    stamps_world(tmp_path, life=150, dead_gap=0.995)
    u = SST.score(REPO, tmp_path)["units"]
    assert u["stdead2"]["w2"]["life"]["verdict"] == "FAST" and u["stdead2"]["w2"]["dispatch"]["verdict"] == "DISP"


# --------------------------------------------------------------------------
# hw
# --------------------------------------------------------------------------

def test_hw_checks_read_the_units_files(tmp_path):
    import gpubench as GB
    d = tmp_path / f"{DAY}-{CARD}-gpubench-r4"
    d.mkdir()
    text_lat = "".join(f"  1000000  1710 {kb:8.1f}  1.0  {cyc:.1f} {cyc:.1f} {cyc:.1f} {cyc:.1f}\n"
                       for kb, cyc in ((8_000, 242.0), (12_000, 242.0), (45_000, 433.0), (50_000, 433.0),
                                       (200_000, 600.0), (300_000, 600.0)))
    text_st = "".join(f" {b:4d}  1  2  {o:5.1f}%     |  GB/s:  3900 2700 3100 {t} 3100 3000\n"
                      for b, o, t in ((256, 25.0, 2400), (1024, 100.0, 3790)))
    cons = GB.constants({"gpu-latency": text_lat, "gpu-stream": text_st}, l2_bytes=60 * GB.MIB, lock_mhz=1710)
    (d / "constants.json").write_text(json.dumps(cons))
    r = tmp_path / f"{DAY}-{CARD}-rulers-r4"
    r.mkdir()
    (r / "rulers.json").write_text(json.dumps({"device": {"l2_cache_size": 62914560},
                                               "rates": {"read2d": {"gbps": 3700.0}, "matmul": {"tflops": 700.0}}}))
    res = SH.score(REPO, tmp_path)
    assert res["checks"]["far_l2"]["verdict"] == "CONSISTENT" and res["checks"]["triad"]["verdict"] == "CONSISTENT"
    assert res["checks"]["dram"]["verdict"] == "CONSISTENT" and res["checks"]["l2_cache_size"]["verdict"] == "CONSISTENT"
    assert res["printed"]["eta_mix"]["read2d"] == pytest.approx(3598 / 3700)


def test_every_scorer_writes_its_files_on_an_empty_tree(tmp_path):
    for mod in (SD, SO, SP, SST, SH):
        out = tmp_path / "out"
        assert mod.main([str(REPO), str(tmp_path), str(out)]) == 0
        assert (out / f"{mod.PART}.score.json").exists() and (out / f"{mod.PART}.score.txt").exists()


def test_a_refused_stamps_unit_reads_not_run_with_its_gate_numbers(tmp_path):
    """Rental 4's post-page fix (reading code): a unit the driver refused has no stamps.json;
    the scorer names it NOT RUN with the gate's numbers off gate.env, the verdict NOT SCORED."""
    pd = tmp_path / f"{DAY}-{CARD}-perturb-pt"
    pd.mkdir(parents=True)
    (pd / "gate.env").write_text("GATE_u8_stf=FAIL median=0.0168 worst=0.0222\n"
                                 "GATE_u13_sttail=FAIL median=0.0753 worst=0.1668\n"
                                 "GATE_u14_sttail=PASS median=0.001 worst=0.002\n")
    drv = tmp_path / "session" / "gh200-driver"
    drv.mkdir(parents=True)
    (drv / "instr-variants.json").write_text(json.dumps([
        {"id": "u8-stf", "kind": "stamps", "model": "mixtral-8x7b"},
        {"id": "u13-sttail", "kind": "stamps", "model": "mixtral-8x22b"},
        {"id": "u14-sttail", "kind": "stamps", "model": "mixtral-8x7b"}]))
    u = SST.score(REPO, tmp_path)["units"]
    assert u["stf"]["use"] == ("NOT SCORED: NOT RUN (gate FAIL: median 1.68%, worst 2.22%; "
                               "variant u8-stf, refused by the driver)")
    assert "NOT RUN (gate FAIL: median 7.53%, worst 16.68%" in u["sttail-mixtral-8x22b"]["use"]
    # a PASS gate with no page is not a refusal: the old reading stands
    assert u["sttail-mixtral-8x7b"]["use"] == "NOT SCORED: 0 stamps.json for mixtral-8x7b sttail"
    assert u["stk64s4"]["use"] == "NOT SCORED: 0 stamps.json for olmoe-1b-7b stk64s4"
