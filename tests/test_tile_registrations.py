"""The tile stage's registrations (docs/registered/2026-10-14-tileA-* and tileB-*): `register.py --check`
recomputes all eight from committed files, and the review's fixes are in them as registered text and
numbers (the v4 reduction, Luo's rate and its rival, the descriptive accounting reading and the
parameter count, the reachability and leakage labels, the stop rule, the bytes at G 1, the plumbing,
the dead lever, the drop order and the 14 resolved ambiguities)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TDIR = REPO / "scripts" / "scoring" / "tileA"
for p in (REPO, REPO / "scripts", TDIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tcommon as TC  # noqa: E402

REGS = {p: TC.registration(REPO, p) for p in TC.PARTS}


def test_register_check_recomputes_all_eight_files():
    got = subprocess.run([sys.executable, str(TDIR / "register.py"), "--check"], capture_output=True, text=True,
                         cwd=REPO, timeout=1800)
    assert got.returncode == 0, got.stdout[-3000:] + got.stderr[-3000:]
    assert got.stdout.count(": same") == 16 and "DIFFERS" not in got.stdout


def test_every_registration_is_dated_before_any_page_and_carries_the_14_resolved_ambiguities():
    for p, d in REGS.items():
        assert d["registered"] == "2026-10-14, before any tile-stage page" and d["name"] == TC.NAMES[p]
        assert len(d["scorer_ambiguities"]) == 14
        assert d["code_pins"]["scripts/dram_counter_route.py"].startswith("130d5dc")
    assert REGS["e1tile"]["status"].startswith("FROZEN BEFORE ANY TILE PAGE; FINALISED MECHANICALLY")


def test_the_floor_registers_the_reduction_luos_rates_and_the_parameter_count():
    f = REGS["floor"]
    m = f["model_v4"]
    assert "300 registered v3 predictions" in m["reduction_to_v3 (design-tile-review 1a)"]
    assert m["eta_mma"]["registered"]["m64n128k16 (128x128 w8)"] == 0.982568
    assert "H800 PCIe" in m["eta_mma"]["what_it_is"] and "FP16" in m["eta_mma"]["what_it_is"]
    assert "2501.12084" in m["eta_mma"]["source"] and "BF16 on the GH200" in m["eta_mma"]["assumption"]
    assert m["eta_mma"]["rival"]["value"] == {"m64n128k16": 0.872174, "m64n64k16": 0.856973}
    assert "UNTESTED" in m["eta_smem"]["status"] and "Table 11" in m["eta_mma"]["source"]
    sens = m["eta_mma"]["sensitivity (build-tile-review 1)"]
    assert sens["128x128_T_shift_pct"] == 0.0 and sens["phi_64x128_max_T_shift_pct"] < 2.0 and sens["max_c64_shift_pct"] > 2.0
    assert f["model_v4"]["F_forms"]["selection_rule"].startswith("F-step unless")
    p = f["parameters"]
    assert len(p["fitted, new in the timing model: 6"]) == 2 and len(p["selections: 3"]) == 3
    assert f["test (BLIND-CALMODEL, DESCRIPTIVE)"]["status"].startswith("DESCRIPTIVE")
    lev = f["selection_for_rental_B (D3)"]["lever_counts_on_B"]
    assert all(v["lever_cells"] < 6 and v["verdict_by_construction"].startswith("NOT SEPARATED") for v in lev.values())


def test_the_census_lists_only_two_reachable_tiles_and_the_seen_ptx():
    t = {x["tile"]: x for x in REGS["census"]["tiles"]}
    assert {k for k, v in t.items() if v["reachable_by_R3"]} == {"128x128x64w8s3", "64x128x64w8s3"}
    assert t["128x128x64w8s3"]["design_occupancy"] == 2 and t["64x128x64w8s3"]["design_occupancy"] == 3
    seen = {x["tile"]: x["counts"] for x in REGS["census"]["seen_ptx"]["files"]}
    assert seen["128x128x64w8s4"]["instruction"] == "m64n128k16" and seen["32x64x64w8s4"]["wavefronts"] == 280.0


def test_bytes_score_every_label_and_fall_back_to_clamp_and_withdraw_fan_et_al():
    b = REGS["bytes"]
    assert "uniform, balanced and skewed" in b["rule (D3; design-tile-review 5)"] and "CLAMP" in b["rule (D3; design-tile-review 5)"]
    assert "withdrawn" in b["keys"]["ROWKEY30"] and "60 MiB" in b["keys"]["ROWKEY30"]
    scored = [r for r in b["byte_predictions"] if r["scored_for_key"]]
    assert {r["label"] for r in scored} == {"PT", "PW", "uniform", "balanced"} and len(scored) == 56
    for r in scored:
        assert r["q_pred"]["ROWKEY60"] and r["q_pred"]["ROWKEY30"] and r["q_pred"]["CLAMP"]["w1"] >= r["q_ideal"] - 1e-6
    assert all(r["G"] == 16 for r in b["byte_predictions"] if not r["scored_for_key"])


def test_rental_b_cells_labels_and_the_drop_order():
    e = REGS["e1tile"]
    nat = [r for r in e["nominal_cells"] if r["arm"] == "native"]
    assert len(nat) == 84 and len(e["power"]["cells"]) == 84
    assert {r["model"] for r in nat} == {"phi-3.5-moe", "qwen1.5-moe-a2.7b", "jetmoe-8b", "deepseek-v2-lite"}
    assert all(r["label_blind"].startswith("SEEN-SHAPE") for r in nat if r["model"] == "deepseek-v2-lite")
    assert e["leakage (design-tile-review 3)"]["rows"] == {"SEEN-WORKLOAD": 14, "BLIND counts": 70}
    seen = {(r["page"], r["n"]) for r in nat if r["label_blind"].endswith("only the tile is new)")}
    assert seen == {("phi-3.5-moe-64g1", 1), ("phi-3.5-moe-128g1", 1), ("phi-3.5-moe-128g1", 2), ("phi-3.5-moe-128g16", 3),
                    ("qwen1.5-moe-a2.7b-128g1", 1), ("qwen1.5-moe-a2.7b-128g1", 2), ("jetmoe-8b-128g1", 2)}
    phi = [r for r in nat if r["model"] == "phi-3.5-moe" and r["label"] in ("uniform", "balanced") and r["page"] != "phi-3.5-moe-128g16"]
    assert phi and all(r["label_blind"].endswith("only the tile is new)") for r in phi if r["n"] == 1)
    plan = e["pages"]["plan"]
    assert plan["drop order, last first (design-tile-review 8, owner fix 10)"] == ["byt (the three byte pages)", "dsv"]
    drops = {u["unit"]: u["drop"] for u in e["pages"]["timed_units"]}
    assert drops["jetmoe-8b timed label=t128g1"] == "nodrop" and drops["qwen1.5-moe-a2.7b timed label=t128g16"] == "nodrop"
    assert all(r["host_ok_nominal"] for r in nat)
    assert "0.40 ms" in e["gates"]["G2 host (design-tile-review 9.7)"]


def test_the_olmoe_n8_deviation_is_registered_where_it_bites():
    for p in ("bytes", "e1cal"):
        assert "OLMoE runs n 1, 2, 4" in REGS[p]["deviation"] and "HOPPER_G1_N" in REGS[p]["deviation"]


def test_the_stop_rule_is_registered_with_its_clauses_and_power():
    s = REGS["stop"]["rule (D6; design-tile-review 4 and 8; owner fixes 6 and 10)"]
    assert "0.75%" in s["A2"] and "1%" in s["A3"] and "SKIPPED, LOST" in s["pages"]
    assert "STOP" in s["census"] and "refuses" in s["finalise"]
    pts = REGS["stop"]["power (from tileB-e1tile, CPU float64)"]
    assert len(pts) == 9 and pts["common 0.01 / model-specific 0"]["P(stop rule A2 passes)"] < 0.2


def test_the_dead_lever_states_what_it_cannot_separate():
    d = REGS["dead"]["rule"]
    assert "c_128 >= 900" in d["kappa 0.325 against smaller kappa"]
    assert len(d["NOT SEPARABLE (stated before the page)"]) == 2
    assert REGS["dead"]["pages"]["tread"].startswith("n 1")
