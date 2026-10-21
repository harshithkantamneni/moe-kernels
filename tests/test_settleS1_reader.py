"""The settle rental S1's reader (scripts/scoring/settleS1/read_s1.py) and registration
(docs/registered/2026-10-21-settleS1-gh200): each function on synthetic pages with a known answer, then on real
counter pages as stand-ins where the format allows (rental A's published 1710 MHz pages, always in the checkout; A2's
1410 MHz pages once published), reproducing DIAGNOSIS-A2's numbers and the registered scorers' byte errors. The
registration: its reading constants are the reader's, its pins hold, its text is its JSON's, and (slow) the
generator reproduces it, v4's byte predictions for S1-M3 included."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
S1 = REPO / "scripts" / "scoring" / "settleS1"
for p in (REPO, REPO / "scripts", S1):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import read_s1 as R  # noqa: E402
import register_s1 as RG  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / f"{RG.NAME}.json").read_text())
PUB_A = REPO / "results" / "published" / "2026-10-20-nvidia_gh200_480gb-tileA-session"
A2_PAGES = sorted((REPO / "results" / "published").glob("*tileA2*/**/*-r3-counters/lock1410/r3c-g*.json"))


# --------------------------------------------------------------------------
# synthetic pages
# --------------------------------------------------------------------------

def page(cells, lock=1410.0, replay="kernel", model="olmoe-1b-7b", bm=128, G=1, arms=("native",), bk=64, bn=128):
    return {"ncu": {"lock_mhz": lock, "replay_mode": replay}, "cells": cells,
            "design": {"model": model, "dtype": "bf16", "block_m": bm, "block_n": bn, "block_k": bk, "group_m": G,
                       "arms": list(arms), "num_warps": 8}}


def cell(n, label, t_us, off_us, lock=1410.0, arm="native", rec=None, dram=None, bincount=None):
    """A cell whose w1 and w2 last t_us[g] with off_us[g] of it not covered by cycles at the lock."""
    pg, recd = {}, {}
    for g in ("w1", "w2"):
        pg[g] = {"gpu_time_ns": t_us[g] * 1e3, "dram_bytes_read": (dram or {}).get(g, 1e9), "grid_size": 100}
        recd[g] = {"sm__cycles_elapsed.avg": (t_us[g] - off_us[g]) * lock, **((rec or {}).get(g, {}))}
    c = {"arm": arm, "n": n, "histogram": label, "per_gemm": pg, "recorded": recd}
    if bincount is not None:
        c["bincount"] = bincount
    return c


def sag_cells(A, tau, lock, ts=(100, 250, 400, 700, 1000, 1500, 2500, 4000)):
    out = []
    for i, t in enumerate(ts):
        o = A * (1 - math.exp(-t / tau))
        out.append(cell(i + 1, "PT", {"w1": t, "w2": t * 0.6}, {"w1": o, "w2": A * (1 - math.exp(-t * 0.6 / tau))}, lock))
    return out


def test_gemm_rows_reads_the_offset_against_the_lock():
    p = page([cell(1, "PT", {"w1": 500.0, "w2": 300.0}, {"w1": 20.0, "w2": 12.0})])
    rows = {r["gemm"]: r for r in R.gemm_rows(p)}
    assert rows["w1"]["off_us"] == pytest.approx(20.0) and rows["w2"]["off_us"] == pytest.approx(12.0)
    assert rows["w1"]["f_eff_mhz"] == pytest.approx(1410.0 * 480 / 500)
    with pytest.raises(ValueError, match="no nvidia-smi lock"):
        R.gemm_rows({"ncu": {}, "cells": []})


def test_sag_fit_recovers_a_planted_saturating_sag():
    fit = R.sag_fit(R.gemm_rows(page(sag_cells(40.0, 600.0, 1410.0))))
    assert fit["A_us"] == pytest.approx(40.0, abs=0.01) and fit["tau_us"] == 600.0 and fit["rms_us"] < 1e-6
    assert fit["cells"] == 16 and set(fit["bins"]) <= {f"{lo:g}-{hi:g} us" for lo, hi in R.BINS_US}
    assert 1300 < fit["linear"]["f_mhz"] < 1410   # a sag reads as a slower linear clock


@pytest.mark.parametrize("shift, reading", [(0.0, "INSTRUMENT-INDEPENDENT"), (-15.0, "INSTRUMENT-INFLATED"),
                                            (+15.0, "INSTRUMENT-DEFLATED")])
def test_replay_contrast_reads_the_registered_bands(shift, reading):
    k = page([cell(1, "PT", {"w1": 400.0, "w2": 300.0}, {"w1": 20.0, "w2": 15.0}),
              cell(4, "PT", {"w1": 1200.0, "w2": 900.0}, {"w1": 30.0, "w2": 25.0})])
    a = copy.deepcopy(k)
    a["ncu"]["replay_mode"] = "application"
    for c in a["cells"]:
        c["per_gemm"]["w1"]["gpu_time_ns"] += shift * 1e3
    got = R.replay_contrast(k, a)
    assert got["reading"] == reading and got["median_d_cell_us"] == pytest.approx(shift)
    assert [c["cell"] for c in got["cells"]] == ["n1/PT/native", "n4/PT/native"] and not got["unmatched"]
    with pytest.raises(ValueError, match="replay_mode"):
        R.replay_contrast(a, k)


def test_replay_contrast_mixed_when_a_cell_moves_past_its_band():
    k = page([cell(1, "PT", {"w1": 400.0, "w2": 300.0}, {"w1": 10.0, "w2": 10.0}),
              cell(2, "PT", {"w1": 400.0, "w2": 300.0}, {"w1": 10.0, "w2": 10.0}),
              cell(4, "PT", {"w1": 400.0, "w2": 300.0}, {"w1": 10.0, "w2": 10.0})])
    a = copy.deepcopy(k)
    a["ncu"]["replay_mode"] = "application"
    a["cells"][2]["per_gemm"]["w1"]["gpu_time_ns"] += 30e3
    assert R.replay_contrast(k, a)["reading"] == "MIXED"


def test_lock_ladder_reads_gone_reduced_and_present():
    pages = {1410.0: page(sag_cells(50.0, 750.0, 1410.0)), 1305.0: page(sag_cells(20.0, 500.0, 1305.0), lock=1305.0),
             1005.0: page(sag_cells(2.0, 300.0, 1005.0), lock=1005.0)}
    got = R.lock_ladder(pages)
    assert got["readings"] == {"1305": "REDUCED", "1005": "GONE"} and got["ref_source"].startswith("this board")
    assert got["fits"]["1305"]["A_us"] == pytest.approx(20.0, abs=0.01)
    pages[1305.0] = page(sag_cells(40.0, 500.0, 1305.0), lock=1305.0)
    del pages[1410.0]
    got = R.lock_ladder(pages)
    assert got["readings"]["1305"] == "PRESENT" and got["ref_A_us"] == R.A2_REF_A_US and "another board" in got["ref_source"]
    assert "olmoe-1b-7b-f128" in got["ref_source"]
    with pytest.raises(ValueError, match="records lock"):
        R.lock_ladder({1305.0: page(sag_cells(5.0, 300.0, 1005.0), lock=1005.0)})


def test_m3_bytes_reads_q_against_the_registered_predictions():
    bm = {"W_w1": 1000.0, "W_w2": 500.0, "operand_per_tile_w1": 10.0, "operand_per_tile_w2": 5.0}
    pred = [{"n": 2, "label": "PT", "arm": "native", "q_pred": {"v3": {"w1": 2.2, "w2": 1.5}, "CLAMP": {"w1": 2.2, "w2": 1.5}}}]
    p = page([cell(2, "PT", {"w1": 1, "w2": 1}, {"w1": 0, "w2": 0}, dram={"w1": 2.0 * 1000 + 20, "w2": 1.5 * 500 + 10})])
    got = R.m3_bytes(p, pred, bm)
    assert got["rows"][0]["q_counted"] == {"w1": 2.0, "w2": 1.5}
    assert got["mean_e"]["CLAMP"]["w1"] == pytest.approx(0.1) and got["reading"] == "W1-OVER"
    pred[0]["q_pred"]["CLAMP"]["w1"] = 2.02
    assert R.m3_bytes(p, pred, bm)["reading"] == "W1-EXACT"
    pred[0]["q_pred"]["CLAMP"]["w1"] = 2.06
    assert R.m3_bytes(p, pred, bm)["reading"] == "W1-PARTIAL"


def ladder_page(e_P, q_S, treads=(3, 4, 8), G=16):
    """A Mixtral 128x128 ladder page whose PRIVATE excess over n is e_P[g] and SHARED's q is q_S[g] at every tread."""
    import dram_counter_route as DCR
    from moe.spec import MODEL_CONFIGS
    bm = DCR.r3_byte_model(MODEL_CONFIGS["mixtral-8x7b"], "bf16", 128)
    cells = []
    for n in treads:
        for arm in ("native", "shared", "private"):
            q = {g: (n + e_P[g] if arm == "private" else q_S[g]) for g in ("w1", "w2")}
            pg = {g: {"dram_bytes_read": q[g] * bm[f"W_{g}"] + n * bm[f"operand_per_tile_{g}"]} for g in ("w1", "w2")}
            tot = sum(v["dram_bytes_read"] for v in pg.values())
            cells.append({"arm": arm, "n": n, "per_gemm": pg, "per_call": {"dram_bytes_read": tot},
                          "per_call_values": [tot] * 3, "per_gemm_values": {g: [pg[g]["dram_bytes_read"]] * 3 for g in pg}})
    return page(cells, model="mixtral-8x7b", G=G, arms=("native", "shared", "private"))


def test_private_split_reads_a_against_b_rereads():
    got = R.private_split(ladder_page({"w1": 0.01, "w2": 0.30}, {"w1": 1.5, "w2": 1.4}))
    assert got["readings"] == {"w1": "B-DOMINATED", "w2": "A-DOMINATED"} and got["treads"] == [3, 4, 8]
    row = got["gemms"]["w2"][0]
    assert row["e_P"] == pytest.approx(0.30, abs=1e-6) and row["shared_reread"] == pytest.approx(0.4, abs=1e-6)
    assert row["a_share_max"] == pytest.approx(0.75, abs=1e-6) and row["e_P_highest_call"] == pytest.approx(0.30, abs=1e-6)
    assert R.private_split(ladder_page({"w1": 0.12, "w2": 0.12}, {"w1": 1.4, "w2": 1.4}))["readings"]["w1"] == "MIXED"
    with pytest.raises(ValueError, match="SHARED and PRIVATE"):
        R.private_split(page([], arms=("native", "shared")))


def test_spearman_handles_ties_and_too_few_points():
    assert R.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert R.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert R.spearman([1, 1, 2, 3], [5, 5, 6, 7]) == pytest.approx(1.0)
    assert R.spearman([1, 2], [1, 2]) is None and R.spearman([1, 1, 1], [1, 2, 3]) is None


@pytest.mark.parametrize("rising, reading", [("ls", "LATENCY"), ("bar", "BARRIER"), (None, "NONE-SEEN")])
def test_stall_table_reads_the_registered_bands(rising, reading):
    consts = REG["constants"]["M5"]
    cells = []
    for i, d in enumerate((2e8, 4e8, 6e8, 8e8, 1e9, 1.2e9)):
        ls = 10 + i * 5 if rising == "ls" else 20 - (i % 2)
        bar = 5 + i * 3 if rising == "bar" else 8 - (i % 2)
        ten = 80 - i * 6 if rising == "ls" else 60 + (i % 2)
        rec = {g: {R.STALL_LS: ls, R.STALL_BAR: bar, R.TENSOR: ten} for g in ("w1", "w2")}
        cells.append(cell(1, f"c{i}", {"w1": 500.0, "w2": 300.0}, {"w1": 10.0, "w2": 8.0}, rec=rec,
                          dram={"w1": d, "w2": d / 2}, bincount=[128] * 64))
    got = R.stall_table(page(cells), consts)
    assert got["reading"] == reading, got["rho_x"]
    assert len(got["rows"]) == 12 and all(r["x_s1"] > 0 for r in got["rows"])
    r0 = got["rows"][0]
    # OLMoE w1 at n 1 uniform: 64 M-tiles x 16 N-tiles = 1024 CTAs, q 8, S 2048 / 64 = 32
    floor = 8 * (32 * consts["c_cycles"] + consts["F_cycles"]["w1"])
    assert r0["x_s1"] == pytest.approx(2e8 / (consts["bw_gbps"] * 1e9) * 1410e6 / floor, rel=1e-5)


def test_a_page_without_the_stall_metrics_reads_none():
    got = R.stall_table(page([cell(1, "PT", {"w1": 500.0, "w2": 300.0}, {"w1": 10.0, "w2": 8.0}, bincount=[128] * 64)]),
                        REG["constants"]["M5"])
    assert got["reading"] is None and got["rows"][0]["long_scoreboard_pct"] is None


def pp_csv(watts, cap=False, sm=1980, mx=1980, enforced=900.0):
    reasons = "0x0000000000000004" if cap else "0x0000000000000000"
    return "\n".join(f"2026/10/21 00:00:{i // 10:02d}.{i % 10}00, {w:.2f}, {enforced:.2f}, 900.00, {sm}, {mx}, {reasons}, 60"
                     for i, w in enumerate(watts)) + "\n"


@pytest.mark.parametrize("watts, cap, sm, reading", [
    ([150.0] * 20 + [800.0] * 200, False, 1980, "ABOVE-700"),
    ([150.0] * 20 + [700.0] * 300, True, 1700, "AT-700"),
    ([150.0] * 20 + [600.0] * 300, True, 1500, "CAP-BELOW-680"),
    ([150.0] * 20 + [600.0] * 300, False, 1980, "UNDECIDED"),
    ([800.0] * 50, False, 1980, "UNDECIDED"),   # 4.1 s over 720 W: under the registered 10 s
])
def test_power_proof_reads_the_registered_bands(watts, cap, sm, reading):
    got = R.power_proof(pp_csv(watts, cap, sm))
    assert got["reading"] == reading, got
    assert got["samples"] == len(watts) and got["enforced_limit_w"] == [900.0]


def test_power_proof_skips_unreadable_rows_and_reads_none_without_samples():
    assert R.power_proof("garbage\n1, 2\n")["reading"] is None
    got = R.power_proof(pp_csv([800.0] * 150) + "2026/10/21 00:01:00.000, [N/A], 900.00, 900.00, 1980, 1980, 0x0, 60\n")
    assert got["samples"] == 150 and got["reading"] == "ABOVE-700" and got["seconds_over_720_w"] == pytest.approx(14.1)


def test_power_proof_reads_a2s_own_probe_draws_as_no_proof():
    """A2's per-second -q samples are not this CSV; the reader takes only the registered columns."""
    assert R.M0_COLUMNS[1] == "power.draw" and R.M0_COLUMNS[2] == "enforced.power.limit"


# --------------------------------------------------------------------------
# real pages as stand-ins
# --------------------------------------------------------------------------

def pub_a(unit, model=None):
    pat = f"*-{model}-{unit}-r3-counters" if model else f"*-{unit}-r3-counters"
    return [json.loads(p.read_text()) for p in sorted(PUB_A.rglob(f"{pat}/lock1710/r3c-g*.json"))]


def test_rental_a_f128_pages_give_diagnosis_a2s_1710_sag():
    """DIAGNOSIS-A2 Q5 (work/transient_1710.py): rental A's 128x128 f128 pages at 1710 MHz, A 82.3 us, tau 1.125 ms."""
    pages = pub_a("f128")
    assert len(pages) == 2
    rows = [r for p in pages for r in R.gemm_rows(p)]
    fit = R.sag_fit(rows)
    assert fit["cells"] == 108 and round(fit["A_us"], 1) == 82.3 and fit["tau_us"] == 1125.0


def test_rental_a_byte_rows_are_the_registered_scorers():
    """m3_bytes on rental A's byte pages with tileA-bytes' predictions reproduces score_bytes' q and e (w1, CLAMP)
    on every row rental A's scorer kept (scripts/scoring/tileA/bytes.score.json)."""
    reg = json.loads((REPO / "docs" / "registered" / "2026-10-14-tileA-bytes-gh200.json").read_text())
    sc = json.loads((REPO / "scripts" / "scoring" / "tileA" / "bytes.score.json").read_text())
    want = {(r["page"], r["cell"]): r for r in sc["rows"] + sc.get("printed_G16_rows", [])}
    assert len(want) >= 19
    hits = 0
    for unit, model, name in (("b128g1", "olmoe-1b-7b", "olmoe-1b-7b-128g1"), ("b64g1", "mixtral-8x7b", "mixtral-8x7b-64g1"),
                              ("b128g16", "mixtral-8x7b", "mixtral-8x7b-128g16"),
                              ("b128g1", "qwen2-57b-a14b", "qwen2-57b-a14b-128g1")):
        (p,) = pub_a(unit, model)
        got = R.m3_bytes(p, [r for r in reg["byte_predictions"] if r["page"] == name])
        for row in got["rows"]:
            w = want.get((name, row["cell"]))
            if w is None:
                continue
            hits += 1
            assert row["e"]["CLAMP"]["w1"] == pytest.approx(w["e"]["CLAMP"], abs=2e-6), (name, row["cell"])
            assert row["q_counted"]["w1"] == pytest.approx(w["q_counted"]["w1"], abs=2e-6)
    assert hits == len(want)


def test_rental_a_private_ladder_splits_and_matches_the_routes_own_q():
    import dram_counter_route as DCR
    (p,) = pub_a("f128", "mixtral-8x7b")
    got = R.private_split(p)
    q = DCR.r3_q(p)
    assert got["treads"] == list(range(1, 10)) and set(got["readings"]) == {"w1", "w2"}
    for g in ("w1", "w2"):
        for row in got["gemms"][g]:
            assert row["q_private"] == pytest.approx(q["private"][g][row["n"]], abs=1e-6)
            assert row["e_P"] == pytest.approx(max(q["private"][g][row["n"]] - row["n"], 0.0), abs=1e-6)
            assert row["e_P_highest_call"] >= row["e_P"] - 1e-9


def test_a_page_contrasted_with_itself_reads_independent():
    (p,) = pub_a("b128g1", "qwen2-57b-a14b")
    a = copy.deepcopy(p)
    a["ncu"]["replay_mode"] = "application"
    got = R.replay_contrast(p, a)
    assert got["reading"] == "INSTRUMENT-INDEPENDENT" and got["max_abs_d_cell_us"] == 0.0 and len(got["cells"]) == 16


@pytest.mark.skipif(not A2_PAGES, reason="A2's pages are not published in this checkout yet")
def test_m2s_fallback_reference_is_the_reader_on_a2s_own_olmoe_f128_page():
    fit = R.sag_fit(R.gemm_rows(json.loads((REPO / R.A2_REF_PAGE).read_text())))
    assert fit["cells"] == 54 and fit["A_us"] == R.A2_REF_A_US and fit["tau_us"] == R.A2_REF_TAU_US
    assert REG["constants"]["M2"]["a2_ref_A_us"] == R.A2_REF_A_US and REG["constants"]["M2"]["a2_ref_page"] == R.A2_REF_PAGE


def test_session_power_reads_each_registered_outcome(tmp_path):
    def tree(name, env, sets):
        d = tmp_path / name / "session"
        (d / "gh200-driver").mkdir(parents=True)
        if env is not None:
            (d / "gh200-driver" / "s1-power.env").write_text(env)
        for f, w in sets.items():
            (d / f).write_text(f"    GPU Power Readings\n        Current Power Limit               : {w:.2f} W\n"
                               "    Module Power Readings\n        Current Power Limit               : 1000.00 W\n")
        return tmp_path / name
    ok = R.session_power(tree("a", "S1_POWER_W=900\n", {"power-set.txt": 900}))
    assert ok["outcome"] == "900 W read back" and ok["limit_w"] == 900.0 and ok["power_set_gpu_current_w"] == {"power-set.txt": 900.0}
    fb = R.session_power(tree("b", "S1_POWER_900=refused\nS1_POWER_W=700\n", {"power-set.txt": 700, "power-set-unsettled-900.txt": 700}))
    assert fb["outcome"] == "900 W refused, 700 W read back" and fb["limit_w"] == 700.0
    assert R.session_power(tree("c", "S1_POWER_900=refused\nS1_POWER_700=refused\n", {}))["outcome"] == "both refused"
    assert R.session_power(tree("d", None, {}))["outcome"] is None


def test_a2_pages_give_diagnosis_a2s_1410_sag():
    """DIAGNOSIS-A2 Q5 (work/transient.py): A2's 128x128 w8 pages (f128, lp1410f, b128*) at 1410, A 50.1 us, tau 0.75 ms."""
    pick = [p for p in A2_PAGES if any(f"-{u}" in p.parent.parent.name for u in ("f128", "lp1410f", "b128"))]
    rows = [r for p in pick for r in R.gemm_rows(json.loads(p.read_text()))]
    fit = R.sag_fit(rows)
    assert fit["cells"] == 290 and round(fit["A_us"], 1) == 50.1 and fit["tau_us"] == 750.0


def test_read_tree_runs_on_a_tree_of_stand_ins(tmp_path):
    """The whole reader over a tree laid out as S1 writes it, built from rental A's pages (relabelled): every
    measurement reads, and the CLI writes its two files."""
    import shutil
    tree = tmp_path / "tree"

    def put(src_unit, model, label, G, lock=1710.0, mutate=None):
        (src,) = sorted(PUB_A.rglob(f"*-{model}-{src_unit}-r3-counters/lock1710/r3c-g{G}.json"))
        d = tree / f"2026-10-21-nvidia_gh200_480gb-{model}-{label}-r3-counters" / f"lock{int(lock)}"
        d.mkdir(parents=True)
        p = json.loads(src.read_text())
        if mutate:
            mutate(p)
        (d / f"r3c-g{G}.json").write_text(json.dumps(p))

    def app(p):
        p["ncu"]["replay_mode"] = "application"
    for model in ("olmoe-1b-7b", "qwen2-57b-a14b"):
        put("b128g1", model, "kr", 1)
        put("b128g1", model, "ar", 1, mutate=app)
        put("b128g1", model, "st", 1)
    put("f128", "olmoe-1b-7b", "f1410", 64)
    put("f128", "mixtral-8x7b", "p128g16", 64)
    out = tmp_path / "out"
    got = subprocess.run([sys.executable, "-B", str(S1 / "read_s1.py"), str(REPO), str(tree), str(out)],
                         capture_output=True, text=True, timeout=300)
    assert got.returncode == 0, got.stdout + got.stderr
    res = json.loads((out / "settleS1.read.json").read_text())
    assert res["power"]["outcome"] is None and res["M0"] is None and res["M1"]["olmoe-1b-7b"]["reading"] == "INSTRUMENT-INDEPENDENT"
    assert res["M2"]["fits"]["1710"]["cells"] == 54 and res["M3"] is None
    assert res["M4"]["p128g16"] is None   # rental A's f128 Mixtral page is G 64, not G 16: no page at the S1 name
    assert res["M5"]["qwen2-57b-a14b"]["reading"] is None and res["M5"]["qwen2-57b-a14b"]["rows"]
    assert (out / "settleS1.read.txt").read_text().startswith(f"READ against {RG.NAME}")
    import os
    for seed in ("0", "12345"):   # the output never depends on the hash seed (keys sorted)
        o2 = tmp_path / f"out{seed}"
        subprocess.run([sys.executable, "-B", str(S1 / "read_s1.py"), str(REPO), str(tree), str(o2)], check=True,
                       capture_output=True, timeout=300, env={**os.environ, "PYTHONHASHSEED": seed})
    assert (tmp_path / "out0" / "settleS1.read.json").read_bytes() == (tmp_path / "out12345" / "settleS1.read.json").read_bytes()
    assert (out / "settleS1.read.json").read_bytes().replace(str(tree).encode(), b"") == \
        (tmp_path / "out0" / "settleS1.read.json").read_bytes().replace(str(tree).encode(), b"")
    shutil.rmtree(tree)


# --------------------------------------------------------------------------
# the registration
# --------------------------------------------------------------------------

def test_the_registrations_reading_constants_are_the_readers():
    c = REG["constants"]
    assert c["M1"] == {"abs_us": R.M1_ABS_US, "rel": R.M1_REL, "slack_us": R.M1_SLACK_US}
    assert (c["M2"]["gone_us"], c["M2"]["reduced_frac"]) == (R.M2_GONE_US, R.M2_REDUCED)
    assert c["M3"] == {"over": R.M3_OVER, "exact": R.M3_EXACT}
    assert c["M4"] == {"a_dom": R.M4_A_DOM, "b_dom": R.M4_B_DOM, "tiny": R.M4_TINY}
    assert c["M5"]["rho"] == R.M5_RHO and c["M5"]["sms"] == R.SMS and c["M5"]["c_cycles"] == RG.CAL_A["c_cycles"]
    assert REG["status"].startswith("DESCRIPTIVE") and "no model verdict" in REG["status"]
    for shape in ("JetMoE", "Phi-3.5", "Qwen1.5"):
        assert shape in REG["seen"]
    assert REG["power_and_lock"]["power_limit_w"] == 900 and REG["power_and_lock"]["session_lock_mhz"] == 1410
    assert "no lock-hold probe" in REG["power_and_lock"]["probe"]
    assert "-pl 700 -sc 0" in REG["power_and_lock"]["fallback"] and "-sc 2 is not used" in REG["power_and_lock"]["fallback"]
    assert set(REG["power_and_lock"]["outcomes"]) == {"900 W read back", "900 W refused, 700 W read back", "both refused"}
    assert "MIXED" in REG["measurements"]["M1"]["follow_up"] and "after S1" in REG["measurements"]["M2"]["decided_after"]


def test_the_registrations_pins_hold_and_its_text_is_its_json():
    for p, s in REG["code_pins"].items():
        assert hashlib.sha256((REPO / p).read_bytes()).hexdigest() == s, p
    assert set(REG["code_pins"]) == set(RG.PINNED)
    txt = (REPO / "docs" / "registered" / f"{RG.NAME}.txt").read_text()
    assert txt == "\n".join(RG.txt(REG)) + "\n"
    assert "\u2014" not in txt and "\u2014" not in json.dumps(REG)
    assert [(r["n"], r["label"], r["arm"]) for r in REG["byte_predictions_m3"]] == [
        (2, "PT", "native"), (2, "PW", "native"), (2, "uniform", "native"), (2, "balanced", "native")]


@pytest.mark.slow
def test_the_generator_reproduces_the_registration():
    got = subprocess.run([sys.executable, "-B", str(S1 / "register_s1.py"), "--check"], capture_output=True, text=True,
                         timeout=600)
    assert got.returncode == 0, got.stdout + got.stderr
