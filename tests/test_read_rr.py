"""Rental RR's reader off the GPU (scripts/scoring/rr/read_rr.py): the per-pair integral against brute force; a synthetic
RR tree planted from a known world (tests/rr_fakes.py: list-scheduled launches, D = C_u + m x R with m the primary form's
m_hat) reads back its answers (the primary form's residual zero on every cell, the refit's constant at 42.5 MiB, the
replicate, the wave spreads); the wave label on constructed curves; corrupted copies (a zero stamp, an end before its
start, a NaN or over-bound overhead, a wrong page, wrong versions, a PARTIAL page, a stamped compile off the registered
occupancy, a census off it, the board, a lock that did not read back, a failed regcheck or hard leg; on a counter page a
spread over 3%, bytes under the compulsory floor, a wrong page, a failed attribution, an incomplete page, another lock, a
missing page) each read NOT READ or NOT IDENTIFIED with its reason, never a number; the freeze refusal; and the stand-in
on PC's and PC2's published pages."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "rr", REPO / "scripts" / "scoring" / "pc2",
          REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import read_rr as R  # noqa: E402
import rr_fakes as F  # noqa: E402
import rrcommon as P  # noqa: E402

PUB = REPO / "results" / "published"
PC = PUB / "2026-11-06-nvidia_gh200_480gb-pc-session"
PC2 = PUB / "2026-11-10-nvidia_gh200_480gb-pc2-session"
B3 = PUB / "2026-11-05-nvidia_gh200_480gb-tileB3-session"
SMALL = ["A128-P20-m2", "A128-P100-m4", "A128-P224-m8", "C128-P64-skew", "B128-P24-m4", "A64-P100-m4", "A64-P224-m8",
         "B64-P40-m4", "A64-P36-m8", "C64-P36-skew"]
F1, F2, F3, F4 = P.FORMS


def read(tmp_path, **kw):
    return R.read(F.tree(tmp_path, **kw), _skip_commit_check_for_tests=True)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return read(tmp_path_factory.mktemp("rrclean"), cells=SMALL)


# ================================================================== the pair geometry, pure

def brute_bytes(t, s, e, b):
    return float(np.sum(b * np.clip((t - s) / (e - s), 0.0, 1.0)))


def test_the_pair_integral_is_the_brute_force_bytes_between_partners():
    tile = P.T128
    counts = [256, 384, 0, 128, 0, 0, 0, 0]          # m_e 2, 3, 1
    N, K = 128 * 6, 512
    ex = F.experts_of(counts, 128)
    live = len(ex) * 6
    s, e = F.schedule(live, 7, K * 128 * 2, 0.0)
    rows = F.rows_of(s, e)
    pr = R.pair_read(rows, ex, len(ex) * 128, N, K, tile)
    assert pr["status"] == "READ" and pr["pairs"] == (1 + 2) * 6
    r = np.array([m * 6 + j for m in range(1, len(ex)) if ex[m] == ex[m - 1] for j in range(6)])
    b = np.full(live, K * 128 * 2.0)
    ss, ee = np.round(s), np.round(e)
    want = [brute_bytes(ss[i], ss, ee, b) - brute_bytes(ss[i - 6], ss, ee, b) for i in r]
    assert np.allclose(pr["_d_start"], want, rtol=1e-9, atol=1e-3)
    mid = (ss + ee) / 2
    want_m = [brute_bytes(mid[i], ss, ee, b) - brute_bytes(mid[i - 6], ss, ee, b) for i in r]
    assert np.allclose(pr["_d_mid"], want_m, rtol=1e-9, atol=1e-3)
    for f in P.FORMS:
        dist, law, C, _ = P.FORMS[f]
        d = np.array(want if dist == "start" else want_m)
        m = np.mean(d > C * P.MIB) if law == "step" else np.mean(1 - np.exp(-np.maximum(d, 0) / (C * P.MIB)))
        assert math.isclose(pr["m_hat"][f], float(m), rel_tol=1e-9, abs_tol=1e-12), f


def test_wave_spreads_and_the_numbers_on_constructed_curves_with_no_label():
    s = np.concatenate([np.full(10, 0.0), np.linspace(100, 190, 10), np.arange(10.0)])
    assert R.wave_spreads(s, 100.0, 10) == pytest.approx([0.0, 0.72, 0.072])
    up = [0.0, 0.3, 0.5, 0.58, 0.6, 0.6, 0.61, 0.6, 0.59, 0.6]
    n = R.wave_numbers(up)
    assert n["status"] == "READ" and n["w_inf"] == 0.6 and n["k_90"] == 3 and abs(n["slope_flat"]) < 0.005
    assert set(n) == {"w_inf", "k_90", "slope_flat", "waves", "status"}           # numbers only: no label, no band
    rising = [0.0, 0.3, 0.45, 0.5, 0.55, 0.5, 0.55, 0.6, 0.65, 0.7]
    assert R.wave_numbers(rising)["slope_flat"] == pytest.approx(0.05)
    assert R.wave_numbers([0.0, 0.3, 0.5, 0.6, 0.6, 0.6])["status"].startswith(R.NI)
    assert R.pooled_curve([[0, 1], [0, 2], [0, 3]]) == [0.0, 2.0] and R.pooled_curve([[0], [0]]) == [None]
    src = (REPO / "scripts/scoring/rr/read_rr.py").read_text() + (REPO / "scripts/scoring/rr/rrcommon.py").read_text()
    assert "AS FOUND" not in src and "AS_FOUND" not in src and "DIFFERENT" not in src


def test_counted_m_and_its_refusals():
    assert P.counted_m(150.0, 100.0, 100.0) == 0.5
    for bad in ((float("nan"), 1, 1), (1, 1, 0), (1, None, 1)):
        assert P.counted_m(*bad) is None


# ================================================================== the clean world reads back its answers

def test_every_unit_is_valid(clean):
    for lab, u in clean["RR-R0"]["stamps"].items():
        assert u["valid"] and u["occupancy_leg"] == "PASS" and u["page_label"] == "CLEAN", (lab, u)
        assert u["exact_time"].startswith("APPLIED")
    for lab, c in clean["RR-R0"]["counters"].items():
        assert c["valid"] and c["why"] == "", (lab, c)
    assert all(v["ok"] for v in clean["census"]["registered_check"].values())


def test_the_primary_form_reads_the_planted_world_and_the_refit_finds_its_constant(clean):
    r2 = clean["RR-R2"]
    for t in ("128x128x64w8s3", "64x128x64w8s3"):
        rows = [r for r in r2[t]["cells"] if r["cell"] in SMALL]
        assert rows and all(r["status"] == "READ" for r in rows), [r["status"] for r in rows]
        assert all(abs(r["resid"][F1]) < 1e-12 for r in rows)
        assert any(abs(r["resid"][F3]) > 0.01 for r in rows)
    both = r2["pooled, both tiles"]
    assert both["cells"] == len(SMALL) and both[F1]["rms"] < 1e-12 and both["lowest rms (DESCRIPTIVE)"] == F1
    assert r2["pooled, FRESH synthetic cells (P not 100 or 224)"]["cells"] == 6
    assert r2["pooled, seen-anchor cells (P 100, 224)"]["cells"] == 4
    ref = r2["refit (DESCRIPTIVE), both tiles"]["step between starts"]
    assert ref["rms"] < 1e-12 and ref["C_star_mib"] <= 42.5
    assert "never a registered one" in r2["refit (DESCRIPTIVE), both tiles"]["status"]
    absent = [r for r in r2["128x128x64w8s3"]["cells"] if r["cell"] not in SMALL]
    assert absent and all(r["status"].startswith(R.NR) and "not counted" in r["status"] for r in absent)


def test_rr_r1_reads_m_beside_its_extras(clean):
    c = clean["RR-R1"]["c128"]["A128-P224-m8"]
    assert c["status"] == "READ" and 0.0 <= c["m"] <= 1.0 and c["spread"] == 0.0
    assert c["clock_mhz"] == 1330.0 and c["w2_D_over_compulsory_minus_1"] == pytest.approx(0.001)


def test_the_wave_reading_and_the_replicate(clean):
    w = clean["RR-R3"]["128x128x64w8s3"]
    assert w["status"] == "READ" and w["pooled"]["pooled_curve"][0] == pytest.approx(0.0, abs=1e-3)
    assert w["pooled"]["numbers"]["status"] == "READ" and "label" not in json.dumps(w)
    assert set(w["by series"]) == {"A", "B", "C"} and w["by series"]["A"]["cells"] == 3
    assert sum(g["cells"] for g in w["by W_c / P"].values()) == 5
    cell = w["cells"]["A128-P224-m8"]
    assert len(cell["wave_spread"]) == 54 and cell["numbers"]["status"] == "READ" and cell["series"] == "A"
    beside = clean["RR-R3"]["stand-in beside (SEEN pages, frozen at registration; never used by a reading)"]
    assert beside is None or set(beside) == {s for _a, s, _m, _b in R.STANDIN_STAMPS}
    rep = clean["RR-R4"]
    assert rep["status"] == "READ" and rep["br unit"]["valid"]
    rows = [v for v in rep["cells"].values() if "resid" in v]
    assert len(rows) == 10 and all(abs(v["resid"][F1]) < 1e-12 for v in rows)   # balanced n1, uniform n1: no re-read
    r2 = clean["RR-R2"]
    assert "share the lock, not the clock" in r2["clock note"]
    assert all(r["page_label"] == "CLEAN" for t_ in ("128x128x64w8s3", "64x128x64w8s3") for r in r2[t_]["cells"])
    assert '"_d' not in json.dumps(clean, default=str)   # the private distance arrays never reach the reading


# ================================================================== corrupted pages and refused units

ONE = ["A128-P20-m2"]


def _cell(d, tile="128x128x64w8s3", cell="A128-P20-m2"):
    return next(r for r in d["RR-R2"][tile]["cells"] if r["cell"] == cell)


@pytest.mark.parametrize("change, why", [
    (lambda a: a.__setitem__((0, 1), 0), "not positive"),
    (lambda a: a.__setitem__((1, 7), a[1, 1] - 5), "end at or before they start"),
])
def test_a_corrupted_launch_leaves_its_cell_not_read(tmp_path, change, why):
    d = read(tmp_path, cells=ONE, overrides={"corrupt_unit": "s128", "corrupt": ("A128-P20-m2", "w1", change)})
    r = _cell(d)
    assert r["status"].startswith(R.NR) and "m_hat" not in r and why in r["status"], r["status"]


@pytest.mark.parametrize("o, why", [(float("nan"), "overhead is unread"), (0.2, "over C3's NOT-READ bound")])
def test_an_unread_or_over_bound_overhead_reads_not_read(tmp_path, o, why):
    d = read(tmp_path, cells=ONE, overrides={"overhead": {("s128", "A128-P20-m2"): o}})
    r = _cell(d)
    assert r["status"].startswith(R.NR) and why in r["status"] and "resid" not in r


@pytest.mark.parametrize("ov, why", [
    ({"unit": {"s128": {"page_sha": "0" * 64}}}, "not the registered file"),
    ({"unit": {"s128": {"versions": {**P.VERSIONS, "vllm": "0.28.0"}}}}, "versions"),
    ({"unit": {"s128": {"versions": {**P.VERSIONS, "python": "3.13.0"}}}}, "versions"),
    ({"unit": {"s128": {"partial": True}}}, "PARTIAL page"),
    ({"unit": {"s128": {"occupancy": {"128x128x64w8s3": 1}}}}, "holds 1 CTAs per SM"),
    ({"unit": {"s128": {"page_label": "PERTURBED"}}}, "the page reads PERTURBED"),
    ({"census": {"128x128x64w8s3": 1}}, "the census reads 1 CTAs per SM"),
    ({"power": "C2_POWER_W=900\n"}, "C2_POWER_W=700"),
    ({"locks": "LOCK_TIMED_OK=1\n"}, "LOCK_PROBE_1410=held"),
    ({"after": {"s128": 1305}}, "the unit's lock read back after the tool"),
    ({"after": {"pt": 1305}}, "its perturb unit's lock read back"),
    ({"regcheck": {"s128": "FAIL 1 against 2"}}, "regcheck FAIL"),
    ({"hard": "FAIL"}, "gate hard legs FAIL"),
])
def test_a_refused_stamps_unit_reads_not_read_with_its_reason(tmp_path, ov, why):
    d = read(tmp_path, cells=ONE, overrides=ov)
    u = d["RR-R0"]["stamps"]["s128"]
    assert not u["valid"] and why in u["why"], u["why"]
    r = _cell(d)
    assert r["status"].startswith(R.NR) and "m_hat" not in r
    assert d["RR-R2"]["128x128x64w8s3"]["pooled"]["status"].startswith(R.NI)


def test_a_python_patch_version_is_not_a_refusal(tmp_path):
    d = read(tmp_path, cells=ONE, overrides={"unit": {"s128": {"versions": {**P.VERSIONS, "python": "3.12.99"}}}})
    assert d["RR-R0"]["stamps"]["s128"]["valid"] and _cell(d)["status"] == "READ"


@pytest.mark.parametrize("cov, why", [
    ({"per_call": {"A128-P20-m2": [1.0e9, 1.0e9, 1.05e9]}}, "spread"),
    ({"per_call": {"A128-P20-m2": [1.0, 1.0, 1.0]}}, "counter sanity"),
    ({"per_call": {"A128-P20-m2": [1.0e9, float("nan"), 1.0e9]}}, "finite"),
    ({"per_call": {"A128-P20-m2": [1.0e9, 1.0e9]}}, "2 counted w1 calls"),
    ({"page_sha": "0" * 64}, "not the registered file"),
    ({"grid": "FAIL"}, "attribution"),
    ({"complete": False}, "not complete"),
    ({"lock_mhz": 1710}, "counted at 1710"),
    ({"versions": {**P.VERSIONS, "torch": "2.14.0"}}, "versions"),
    ({"schema": "moe-kernels/decode-page/1"}, "schema"),
    ({"missing": True}, "no counter page"),
])
def test_a_refused_counter_page_or_cell_reads_not_read(tmp_path, cov, why):
    d = read(tmp_path, cells=ONE, overrides={"counter": {"c128": cov}})
    c = d["RR-R1"]["c128"].get("A128-P20-m2") or {}
    blob = json.dumps(d["RR-R0"]["counters"]["c128"]) + json.dumps(c) + _cell(d)["status"]
    assert why in blob, blob[:600]
    assert "m" not in c and _cell(d)["status"].startswith(R.NR)


def test_a_counter_lock_that_did_not_read_back_refuses_the_counter_unit(tmp_path):
    d = read(tmp_path, cells=ONE, overrides={"after": {"c128": 1305}})
    assert not d["RR-R0"]["counters"]["c128"]["valid"] and _cell(d)["status"].startswith(R.NR)


def test_the_occupancy_leg_reads_every_compile():
    man = {"sass": [{"config": [128, 128, 64, 1, 3, 8, False, 2], "ctas_per_sm": 2},
                    {"config": [128, 128, 64, 1, 3, 8, True, 1], "ctas_per_sm": 1}], "launches": [1]}
    bad = R.occupancy_leg(man)
    assert len(bad) == 1 and "(w2) holds 1" in bad[0]
    assert R.occupancy_leg({"sass": [], "launches": [1]}) == ["no stamped compile recorded"]


@pytest.mark.parametrize("bov, why", [
    ({"per_call": {"PT-n2": [1.0e9, 1.0e9, 1.1e9]}}, "spread"),
    ({"per_call": {"PT-n2": [1.0, 1.0, 1.0]}}, "counter sanity"),
    ({"per_call": {"PT-n2": [2.0e9, 2.0e9]}}, "2 counted w1 calls"),
    ({"sha": "0" * 64}, "not the registered Phi-3.5"),
    ({"versions": {**P.VERSIONS, "vllm": "0.28.0"}}, "versions"),
    ({"lock_mhz": 1710}, "counted at 1710"),
    ({"status": "G=1 exit 3 12:00:00 30 s\n"}, "status line reads G=1 exit 3"),
    ({"missing": True}, "no counter page"),
])
def test_the_replicates_bytes_take_rr_r1s_rules_and_a_refused_unit_reads_no_number(tmp_path, bov, why):
    d = read(tmp_path, labels=("sr",), overrides={"counter": {"br": bov}})
    rep = d["RR-R4"]
    blob = json.dumps(rep["br unit"]) + json.dumps(rep["cells"]["PT-n2-native"])
    assert why in blob, blob[:800]
    assert "m" not in rep["cells"]["PT-n2-native"] and "resid" not in rep["cells"]["PT-n2-native"]


def test_the_replicate_reads_no_number_on_an_invalid_board(tmp_path):
    d = read(tmp_path, labels=("sr",), overrides={"power": "C2_POWER_W=900\n"})
    assert not d["RR-R4"]["br unit"]["valid"] and "board" in d["RR-R4"]["br unit"]["why"]
    assert not any("m" in r for r in d["RR-R4"]["cells"].values())


def test_the_reader_refuses_a_tree_whose_commit_is_not_on_its_ledger(tmp_path):
    t = F.tree(tmp_path, labels=("s128",), cells=ONE)
    with pytest.raises(SystemExit, match="REFUSED: the rental's commit is not on its ledger"):
        R.read(t)


# ================================================================== the stand-in on SEEN published pages

@pytest.mark.skipif(not (PC.is_dir() and PC2.is_dir() and B3.is_dir()), reason="published pages absent")
def test_the_stand_in_reads_both_boards_within_two_thousandths_of_the_designs_fit():
    d = R.standin(PUB)
    assert d["pooled (every counted board)"]["rows"] == 54                # design-P1's 54 (cell, board) rows
    for sel, figs in R.DESIGN_FIT.items():
        for f, want in figs.items():
            got = d["read against design-P1"][sel if sel == "Phi-3.5 only" else "pooled (every counted board)"][f]
            assert abs(got["here"] - want) <= 0.002, (sel, f, got)
    assert set(d["by family"]) == {"mixtral-8x7b", "qwen2-57b-a14b", "phi-3.5-moe"}
    assert d["same board (stamps and bytes on one board: PC's Phi-3.5)"]["rows"] == 14
    w = d["RR-R3 numbers per stamped family"]
    assert set(w) == {s for _a, s, _m, _b in R.STANDIN_STAMPS} and w["instr-qwen2-57b-a14b-stq"]["cells"] == 6
    reg = json.loads((REPO / "docs/registered" / f"{P.NAME}.json").read_text())
    assert R.standin_waves(d) == P.STANDIN_WAVE == reg["stand-in wave numbers (SEEN pages, frozen)"]
    p2 = d["PC2 wave spread (decode tiles; RR-R3's rule on SEEN pages of another regime)"]
    assert p2["sd"]["cells"] and all(len(c["wave_spread"]) >= 7 for c in p2["sd"]["cells"].values())
    assert '"_d' not in json.dumps(d, default=str)
