"""Rental PC2's reader off the GPU (scripts/scoring/pc2/read_pc2.py): a whole synthetic PC2 tree planted from a known
world (tests/pc2_fakes.py: life = C0 + b / r) reads back its answers (c0 on every F ladder, the L table, X_sat on each
tile, SATURATED at G 528, kappa, the twin and pair ratios, leaders and followers, the replicate); then corrupted copies
(a NaN overhead, a zero stamp, a non-positive life, an over-ceiling rate, an L2-served CTA, a wrong page, wrong versions,
a python minor version, a lock that did not read back, the perturb unit's lock, a failed regcheck, a failed hard leg, a
PARTIAL page, a stamped compile whose occupancy is not the registered one, the census, the board) each read NOT READ or
NOT IDENTIFIED with its reason, never a number; the freeze refusals; and the stand-in readout on PC's published pages."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "pc2", REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pc2_fakes as F  # noqa: E402
import pc2common as P  # noqa: E402
import read_pc2 as R  # noqa: E402

PC = REPO / "results" / "published" / "2026-11-06-nvidia_gh200_480gb-pc-session"
C0_US = F.C0_NS / (1 + F.O) / 1e3


def read(tmp_path, **kw):
    pc2 = F.tree(tmp_path, **kw)
    return R.read(pc2, _skip_commit_check_for_tests=True)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return read(tmp_path_factory.mktemp("pc2clean"))


# ================================================================== the clean world reads back its answers

def test_every_unit_is_valid_and_its_page_read(clean):
    for lab, u in clean["PC2-R0"].items():
        assert u["valid"] and u["why"] == "" and u["occupancy_leg"] == "PASS" and u["page_label"] == "CLEAN", (lab, u)
    assert clean["census"]["registered_check"] == {t: {"registered": o, "census": o, "ok": True}
                                                   for t, o in P.OCCUPANCY.items()}
    assert all(r["status"] == "READ" for r in clean["PC2-R1"]["R1a rate curve"])


def test_every_f_ladder_reads_the_planted_c0(clean):
    r1, r2 = clean["PC2-R1"]["R1b F ladders"], clean["PC2-R2"]
    for lad in list(r1.values()) + list(r2["R2a c0 per G (F 256 to 14336)"].values()) + \
            list(r2["R2b c0 per G (short rungs alone, 66 to 459 KB)"].values()):
        assert lad["status"] == "READ" and math.isclose(lad["c0_us"], C0_US, rel_tol=2e-4), lad
    assert set(r2["R2a c0 per G (F 256 to 14336)"]) == {"G 8", "G 32", "G 128", "G 256"}
    split = r2["R2d life split"]["S-G8-F256"]
    assert split["status"] == "READ" and math.isclose(split["fixed_us"], C0_US, rel_tol=2e-3)


def test_the_l_table_x_sat_and_the_big_wave_reading(clean):
    rows = clean["PC2-R1"]["R1c L table (c0 = R1b G 128)"]
    assert [r["cell"] for r in rows][0] == "D-G4" and all(r["status"] == "READ" for r in rows)
    q = F.q_b(P.D16)
    assert all(r["Q_B"] == q for r in rows)
    lo = [r["L_us"] for r in rows if r["cell"] in ("D-G4", "D-G8", "D-G16", "D-G32")]
    assert all(math.isclose(x, F.L0_NS / 1e3 / (1 + F.O), rel_tol=2e-3) for x in lo), lo   # L = Q_B / r = L0, o-scaled
    assert clean["PC2-R1"]["R1d X_sat"]["status"] == "READ" and len(clean["PC2-R1"]["R1d X_sat"]["pool"]) >= 2
    for t in ("32x64x128w4s3", "32x128x64w4s3"):
        x = clean["PC2-R3"][t]
        assert x["X_sat"]["status"] == "READ" and x["G 528"]["status"] == "SATURATED", x["G 528"]
        assert x["kappa 528 / kappa 396"]["status"] == "HELD" and x["v6d term 3"]["label"] == "HOLDS"
        assert all(" w1 " in lab or (" w2 G " in lab and int(lab.split(" G ")[1].split()[0]) > 528 and "-F" not in lab)
                   for lab, _x in x["X_sat"]["pool"])
        assert not any("G528" in lab for lab, _x in x["X_sat"]["pool"]) and x["X_sat"]["composition"]["excluded"]
        g528 = next(r for r in x["rows"] if r["cell"].endswith("-G528"))
        assert g528["per_sm_max"] == 4 and g528["distinct_sms"] == 132 and g528["kappa"] is not None
        past = next(r for r in x["rows"] if r["cell"].endswith("-G594"))
        assert past["kappa"] is None and "past one wave" in past["span_status"]


def test_kappa_is_the_planted_span_over_the_life(clean):
    sp = clean["PC2-R2"]["R2c SPAN at short slabs, 64x128x64"]["S-G8-F256"]
    b = 256 * 128 * 2
    life = F.C0_NS + b / F.world(P.D64, 8)
    assert sp["status"] == "READ" and math.isclose(sp["kappa"], (life + 2 * 7) / life, rel_tol=1e-3)
    assert math.isclose(sp["t_us"], (life + 14) / (1 + F.O) / 1e3, rel_tol=1e-3)


def test_the_naive_twins_pairs_leaders_and_followers(clean):
    r4 = clean["PC2-R4"]
    assert set(r4["twins"]) == {f"{c['label']} / {c['twin']}" for c in P.block_n() if "twin" in c}
    assert set(r4["pairs"]) == {f"{c['label']} / {c['pair']}" for c in P.block_n() if "pair" in c}
    for k, v in list(r4["twins"].items()) + list(r4["pairs"].items()):
        for g in ("w1", "w2"):
            assert v[g]["status"] == "READ" and math.isclose(v[g]["t_us"], 1.0, rel_tol=1e-3), (k, g, v)
    f = r4["cells"]["N-PH-T2-A2"]["w2"]
    assert f["followers"] == f["leaders"] > 0
    n = r4["cells"]["N-PH-T1"]["w2"]
    assert n["followers"] == 0 and n["leaders"] > 0
    assert r4["cells"]["N-MX-T2"]["w2"].get("followers") is None   # an aligned twin has no leader split
    assert all(abs(v["r"]) < 1e-3 and abs(v["t"]) < 1e-3 for v in clean["PC2-R5"].values())


def test_plain_is_printed_with_its_limit_beside_every_block(clean):
    k = next(x for x in clean["PC2-R1"] if x.startswith("R1h PLAIN"))
    assert "no per-CTA life" in k and clean["PC2-R1"][k]["D-G4 w2"]["plain_us"] == pytest.approx(100.0)
    assert any(x.startswith("R2f PLAIN") for x in clean["PC2-R2"]) and any(x.startswith("R3p PLAIN")
                                                                            for x in clean["PC2-R3"])


# ================================================================== corrupted pages and refused units

def _sd(tmp_path, **ov):
    return read(tmp_path, labels=("sd",), overrides=ov)


@pytest.mark.parametrize("change, why", [
    (lambda a: a.__setitem__((0, 1), 0), "start or end stamp is not positive"),
    (lambda a: a.__setitem__((1, 7), a[1, 1] - 5), "end at or before they start"),
    (lambda a: a.__setitem__((2, 8), a[2, 2] + 10 ** 9), "over the 1980 MHz ceiling"),
    (lambda a: a.__setitem__((3, 7), a[3, 1] + 100), "over the 4022 GB/s ceiling"),
    (lambda a: a.__setitem__((4, 9), -1), "rows not stamped live or dead"),
    (lambda a: a.__setitem__((5, 7), a[5, 1] + 9000), "L2-served"),
])
def test_a_corrupted_launch_leaves_its_cell_not_read(tmp_path, change, why):
    d = _sd(tmp_path, corrupt=("D-G128", "w2", change), corrupt_unit="sd")
    row = next(r for r in d["PC2-R1"]["R1a rate curve"] if r["cell"] == "D-G128")
    assert row["status"].startswith("NOT READ") and why in row["status"] and row["r"] is None, row
    lad = d["PC2-R1"]["R1b F ladders"]["G 128"]          # 5 of its 6 rungs still READ: PC's 4-rung rule holds
    assert lad["status"] == "READ" and [x for x in lad["rungs"] if x[0] == "D-G128"][0][2] is None
    lt = {r["cell"]: r["status"] for r in d["PC2-R1"]["R1c L table (c0 = R1b G 128)"]}
    assert lt["D-G128"].startswith("NOT READ") and lt["D-G64"] == "READ"


def test_a_nan_overhead_reads_not_read_never_zero(tmp_path):
    d = _sd(tmp_path, nan_overhead=("sd", "D-G8"))
    row = next(r for r in d["PC2-R1"]["R1a rate curve"] if r["cell"] == "D-G8")
    assert row["status"].startswith("NOT READ") and "overhead" in row["status"] and row["r"] is None


@pytest.mark.parametrize("ov, why", [
    ({"unit": {"sd": {"page_sha": "0" * 64}}}, "the page is not the registered file"),
    ({"unit": {"sd": {"versions": {**P.VERSIONS, "torch": "2.12.0"}}}}, "versions"),
    ({"unit": {"sd": {"versions": {**P.VERSIONS, "python": "3.11.9"}}}}, "versions"),
    ({"unit": {"sd": {"partial": True}}}, "PARTIAL page"),
    ({"unit": {"sd": {"occupancy": {"16x64x128w4s4": 2}}}}, "holds 2 CTAs per SM, the registered (plain) occupancy is 3"),
    ({"unit": {"sd": {"page_label": "PERTURBED"}}}, "the page reads PERTURBED, the gate CLEAN"),
    ({"after": {"sd": "1695"}}, "the unit's lock read back after the tool"),
    ({"after": {"pt": "1695"}}, "its perturb unit's lock read back after the tool"),
    ({"regcheck": {"sd": "FAIL 2 CTAs/SM against 3"}}, "regcheck FAIL"),
    ({"hard": "FAIL"}, "gate hard legs FAIL"),
    ({"power": "C2_POWER_W=650\n"}, "C2_POWER_W=700"),
    ({"locks": "LOCK_PROBE_1410=held\n"}, "LOCK_PROBE_1710=held"),
])
def test_a_refused_unit_reads_not_read_with_its_reason(tmp_path, ov, why):
    d = _sd(tmp_path, **ov)
    u = d["PC2-R0"]["sd"]
    assert not u["valid"] and why in u["why"], u["why"]
    assert all(r["status"].startswith("NOT READ") for r in d["PC2-R1"]["R1a rate curve"])
    assert all(v["status"].startswith("NOT IDENTIFIED") for v in d["PC2-R1"]["R1b F ladders"].values())


def test_a_python_patch_version_is_not_a_refusal(tmp_path):
    d = _sd(tmp_path, unit={"sd": {"versions": {**P.VERSIONS, "python": "3.12.3"}}})
    assert d["PC2-R0"]["sd"]["valid"], d["PC2-R0"]["sd"]["why"]


def test_a_census_off_the_registered_occupancy_leaves_the_wave_readings_not_identified(tmp_path):
    d = _sd(tmp_path, census={"16x64x128w4s4": 2})
    assert d["PC2-R1"]["R1d X_sat"]["status"].startswith("NOT IDENTIFIED: the census reads 2")
    row = next(r for r in d["PC2-R1"]["R1a rate curve"] if r["cell"] == "D-G8")
    assert row["status"] == "READ" and row["kappa"] is None and row["span_status"].startswith("NOT IDENTIFIED")


def test_a_missing_unit_reads_not_read_and_nothing_falls_back(tmp_path):
    d = read(tmp_path, labels=("sd",))
    assert not d["PC2-R0"]["sw"]["valid"] and "no stamp page" in d["PC2-R0"]["sw"]["why"]
    for t in ("32x64x128w4s3", "32x128x64w4s3"):
        assert d["PC2-R3"][t]["X_sat"]["status"].startswith("NOT IDENTIFIED")
        assert d["PC2-R3"][t]["G 528"]["status"].startswith("NOT IDENTIFIED")
    assert all(v["w2"]["status"].startswith("NOT READ") for v in d["PC2-R4"]["twins"].values())


def test_the_occupancy_leg_reads_every_compile():
    man = {"sass": [{"config": [16, 64, 128, 1, 4, 4, False, 2], "ctas_per_sm": 3},
                    {"config": [16, 64, 128, 1, 4, 4, True, 1], "ctas_per_sm": 3}], "launches": [{}]}
    assert R.occupancy_leg(man) == []
    man["sass"][1]["ctas_per_sm"] = 2
    assert "w2) holds 2 CTAs per SM" in R.occupancy_leg(man)[0]
    man["sass"].append({"config": [16, 64, 128, 1, 2, 4, True, 1], "ctas_per_sm": 7})
    assert any("16x64x128w4s2 compile" in w and "is None" in w for w in R.occupancy_leg(man))
    assert R.occupancy_leg({"sass": [], "launches": [{}]}) == ["no stamped compile recorded"]


def test_a_g528_rung_below_the_plateau_reads_below_and_the_short_siblings_stay_out_of_x_sat(tmp_path):
    """A G 528 long-slab rung planted 20% slow, with its short-slab siblings at plateau speed: X_sat's pool takes neither
    (G 528 is excluded from its own reference, short slabs never enter), so G 528 reads BELOW and term 3 DOES NOT HOLD."""
    d = read(tmp_path, labels=("sw",), overrides={"unit": {"sw": {"slow": {("W64-G528", "w2"): 1.25}}}})
    x = d["PC2-R3"]["32x64x128w4s3"]
    assert x["X_sat"]["status"] == "READ" and x["G 528"]["status"] == "BELOW", x["G 528"]
    assert x["G 528"]["X_528_over_X_sat"] < 0.97
    labs = [lab for lab, _v in x["X_sat"]["pool"]]
    assert labs and not any(s in lab for lab in labs for s in ("W64-G528", "-F1024", "-F512"))
    assert x["X_sat"]["composition"]["w2 long-slab rungs past one wave"] == 2   # G 594 and G 1056
    assert x["v6d term 3"]["label"] == "DOES NOT HOLD" and "G 528 BELOW" in x["v6d term 3"]["why"]
    assert d["PC2-R3"]["32x128x64w4s3"]["v6d term 3"]["label"] == "HOLDS"


def test_term3_reads_its_registered_label():
    sat, held = {"status": "SATURATED"}, {"status": "HELD", "ratio": 1.01}
    assert R.term3(sat, held)["label"] == "HOLDS"
    assert R.term3(sat, {"status": "MOVED", "ratio": 1.05})["label"] == "DOES NOT HOLD"
    assert R.term3({"status": "ABOVE"}, held)["label"] == "DOES NOT HOLD"
    assert R.term3({"status": "NOT IDENTIFIED: x"}, held)["label"] == "NOT IDENTIFIED"
    assert R.term3(sat, {"status": "NOT IDENTIFIED: y"})["label"] == "NOT IDENTIFIED"


def test_exact_time_not_applied_when_a_stamp_stays_past_2_40_after_the_rebase(tmp_path):
    def far(a):   # one live CTA's end on another absolute base: the unit's stamps no longer share one
        a[0, 7] = a[0, 7] + (1 << 50)
    d = read(tmp_path, labels=("sd",), overrides={"corrupt": ("D-G8", "w2", far), "corrupt_unit": "sd"})
    u = d["PC2-R0"]["sd"]
    assert u["exact_time"].startswith("NOT APPLIED: a stamp reads") and "2^40" in u["exact_time"]
    clean = read(tmp_path / "c", labels=("sd",))["PC2-R0"]["sd"]
    assert clean["exact_time"].startswith("APPLIED (offset") and clean["time_offset_ns"] > 0
    assert R.exact_time_check([np.zeros((2, 10), np.int64) + [[0, 5, 0, 0, 0, 0, 0, 9, 0, 1]]], 0).startswith("NOT NEEDED")


# ================================================================== EXACT TIME and TIMER

def _abs_rows(n=8, life=5000.0, base=1_793_988_699_408_532_736, step=32):
    tile = P.D64
    s = np.array([base + step * i for i in range(n)], np.int64)
    e = s + int(life)
    a = np.full((n, 10), -1, np.int64)
    a[:, 0], a[:, 1], a[:, 7], a[:, 9] = np.arange(n), s, e, 1
    a[:, 2], a[:, 8] = 1_000_000, 1_000_000 + int(life * 1.7)
    return a, tile


def test_exact_time_rebases_absolute_stamps_and_keeps_a_zero_stamp_refused():
    a, tile = _abs_rows(life=5001.0)
    raw = R.RP.launch(a, [0], 64, 128 * 8, 256, tile, o=0.0, expected_live=8)
    off = R.time_offset([a])
    assert off == int(a[:, 1].min()) - (1 << 20) and R.time_offset([np.abs(a) % 10_000]) == 0
    junk = a.copy()
    junk[3, 7] = 12345                                   # one garbled small stamp does not switch the rebase off
    assert R.time_offset([junk]) == off
    ex = R.RP.launch(R.rebase(a, off), [0], 64, 128 * 8, 256, tile, o=0.0, expected_live=8)
    assert ex["valid"] and ex["L_ns"] == 5001.0 and ex["timer_res_ns"] == 32
    assert raw["L_ns"] != 5001.0 or raw["timer_res_ns"] != 32          # the float64 step at 1.8e18 ns is 256 ns
    bad = a.copy()
    bad[0, 1] = 0
    got = R.RP.launch(R.rebase(bad, R.time_offset([bad])), [0], 64, 128 * 8, 256, tile, o=0.0, expected_live=8)
    assert not got["valid"] and "not positive" in got["why"]


def test_the_unit_timer_takes_the_cards_resolution_only_when_the_unit_agrees():
    tick = [_abs_rows(step=32)[0] for _ in range(12)]
    one = _abs_rows(step=0)[0]                       # every CTA starts in one tick: no resolution of its own
    ref = R.unit_timer(tick + [one])
    assert ref["res"] == 32 and ref["agree"] == 12
    assert R.unit_timer(tick[:5])["res"] is None                     # under 10 launches report one
    mixed = [_abs_rows(step=32)[0] for _ in range(5)] + [_abs_rows(step=96)[0] for _ in range(6)]
    assert R.unit_timer(mixed)["res"] == 96 and R.unit_timer(mixed[:9])["res"] is None   # 9 launches: under 10
    split = [_abs_rows(step=s)[0] for s in (32, 64, 96, 128) for _ in range(3)]
    assert R.unit_timer(split)["res"] is None                        # no value holds half
    tile = P.D64
    a = R.rebase(one, R.time_offset([one]))
    alone = R.RP.launch(a, [0], 64, 128 * 8, 256, tile, o=0.0, expected_live=8)
    assert not alone["valid"] and "resolution None" in alone["why"]
    rescued = R.RP.launch(a, [0], 64, 128 * 8, 256, tile, o=0.0, expected_live=8, timer_ref=ref)
    assert rescued["valid"] and rescued["timer_res_ns"] == 32


# ================================================================== the freeze

def test_the_reader_refuses_a_tree_whose_commit_is_not_on_its_ledger_or_is_not_this_tree(tmp_path):
    pc2 = F.tree(tmp_path, labels=("sd",))
    with pytest.raises(SystemExit, match="the rental's commit is not on its ledger"):
        R.read(pc2)
    (pc2 / "session" / "gh200-driver" / "status").write_text(
        "2026-11-10T00:00:00Z driver start: commit 6c800f845af59e50e118083760cd28bb5a548cea, steps of the plan x\n")
    with pytest.raises(SystemExit, match="the working tree is not the rental's commit"):
        R.read(pc2)


def test_a_reserved_shape_on_a_page_is_refused_by_the_reader_too():
    with pytest.raises(P.Refused, match="reserved"):
        R.cell_cfg({"shape": "granite-3.0-1b-a400m"}, P.load_shapes())


# ================================================================== the stand-in on PC's published pages

@pytest.mark.skipif(not (PC / "results").is_dir(), reason="PC's published session is not in this checkout")
def test_the_stand_in_reads_pcs_published_pages():
    d = R.standin(PC)
    sp = d["SPAN on PC's block A (kappa: single-wave launches only)"]
    assert sp["A-G128 w2"]["status"] == "READ" and sp["A-G128 w2"]["kappa"] == pytest.approx(1.0613, abs=5e-4)
    assert sp["A-G384 w2"]["kappa"] == pytest.approx(1.0328, abs=5e-4)       # v6d's kappa(384), 1.032
    assert sp["A-G512 w2"]["kappa"] is None and "past one wave" in sp["A-G512 w2"]["status"]
    plain = next(v for k, v in d.items() if k.startswith("PLAIN on PC's refused block D"))
    assert plain["D-G4 w2 16x64x128w4s4"]["plain_us"] == pytest.approx(40.688, abs=1e-3)
    assert plain["D-G16-s2 w2 16x64x128w4s2"]["plain_us"] == pytest.approx(108.97, abs=1e-2)
    tm = d["PC2's timer rule on PC's launches that had no resolution (PC: NOT READ)"]
    assert tm["unit resolution (exact stamps)"]["res"] == 32
    assert all(v["valid"] for k, v in tm.items() if k.startswith("A-"))
    assert {k.rsplit(" call", 1)[0] for k in tm if k.startswith("A-")} == {"A-F7168-G8 w1", "A-F7168-G8 w2", "A-TP8-G128 w1"}
    q = d["EXACT TIME on PC's block A: the median life, PC's float64 stamps against exact ones"]
    assert q["launches"] >= 70 and 0 < q["max_abs_ratio_minus_1"] < 0.05
    pts = d["PLAIN D16 w2 at F 14336 against G (single wave, DESCRIPTIVE)"]
    assert [g for g, _ in pts] == [4, 16, 64, 128, 256, 384] and all(np.isfinite(v) for _, v in pts)
