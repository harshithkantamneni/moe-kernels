"""Rental PC's reader off the GPU (scripts/scoring/pc/read_pc.py): the per-CTA functions on planted arrays (concurrency,
sharing, the steady window, the timer fallback, the plateau), ROW SANITY on corrupted rows (NaN, zero, negative,
out-of-ceiling), the fits on planted numbers, H1's label for every falsifier and band, then whole synthetic PC trees
planted from three worlds (H1 holds: SUPPORTED; a clock-bound rate: FALSIFIED; a warps effect: MIXED) and corrupted
copies of the first (a NaN overhead, a zero stamp, a non-positive life, an over-ceiling rate, an L2-served CTA, a
wrong page, wrong versions, a lock that did not read back, a failed regcheck, the board's clauses, the census); and the
stand-in readout reproducing the design's section 1.1 table on C3's published stamp pages."""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "tests", REPO / "scripts" / "scoring" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pc_fakes as F  # noqa: E402
import pccommon as P  # noqa: E402
import read_pc as R  # noqa: E402

C3 = REPO / "results" / "published" / "2026-10-27-nvidia_gh200_480gb-cal3-session"


def rows_of(starts, ends, sms=None, f_mhz=1400.0, kinds=None):
    s, e = np.asarray(starts, np.float64), np.asarray(ends, np.float64)
    n = len(s)
    a = np.full((n, 10), -1, np.int64)
    a[:, 0] = np.arange(n) % P.SMS if sms is None else sms
    a[:, 1], a[:, 7] = np.round(s), np.round(e)
    a[:, 2], a[:, 8] = np.round(s * f_mhz / 1e3), np.round(e * f_mhz / 1e3)
    a[:, 9] = 1 if kinds is None else kinds
    return a


# ================================================================== per-CTA functions

def test_concurrency_and_sharing_are_exact_on_planted_intervals():
    # A [0, 10], B [0, 10], C [5, 15]: A and B see 2 for 5 ns and 3 for 5 ns (2.5); C sees 3 then 1 (2.0)
    n, peak = R.concurrency([0, 0, 5], [10, 10, 15])
    assert np.allclose(n, [2.5, 2.5, 2.0]) and peak == 3
    m = R.sharing([0, 0, 5], [10, 10, 15], [0, 1, 0])
    assert np.allclose(m, [1.5, 1.0, 1.5])
    X, w = R.steady_x([0, 0, 5], [10, 10, 15], [100, 100, 100], frac=0.9)
    assert w == 5 and math.isclose(X, (50 + 50 + 50) / 5)   # in [5, 10] all three stream 10 B/ns
    assert R.rank_corr([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert R.rank_corr([0, 1], [1, 0]) is None


def test_a_launchs_rates_lives_and_plateau_come_back_exact():
    tile, N, K = P.D64, 128 * 8, 14336
    b = K * 128 * 2
    s = [1000 + 2 * i for i in range(8)]
    e = [x + b / 40.0 for x in s]   # 40 GB/s each
    lr = R.launch(rows_of(s, e), [0], 64, N, K, tile, o=0.0, expected_live=8, q_guard=P.q_bytes(tile, 2))
    assert lr["valid"], lr["why"]
    assert np.allclose(lr["r"], 40.0, rtol=1e-4) and lr["plateau"].all() and lr["peak"] == 8
    assert lr["path"] == "globaltimer" and lr["timer_res_ns"] == 2 and math.isclose(lr["X"], 320.0, rel_tol=1e-3)
    o = R.launch(rows_of(s, e), [0], 64, N, K, tile, o=0.05, expected_live=8)
    assert np.allclose(o["r"], 40.0 * 1.05, rtol=1e-4)


def test_a_coarse_globaltimer_takes_the_clock64_path():
    tile, N, K = P.D64, 128 * 4, 2048
    b = K * 128 * 2
    s = np.array([0, 1024, 2048, 3072]) + 10    # starts 1024 ns apart: res 1024 over a tenth of the median life
    life = np.array([5000.0, 5000.0, 5000.0, 20000.0])   # one CTA lives 10 resolutions: its clock is f_l
    lr = R.launch(rows_of(s, s + life), [0], 64, N, K, tile, o=0.0, expected_live=4)
    assert lr["valid"] and lr["path"].startswith("clock64 at 1400") and np.allclose(lr["r"], b / life, rtol=1e-3)
    short = R.launch(rows_of(s, s + 5000.0), [0], 64, N, K, tile, o=0.0, expected_live=4)
    assert not short["valid"] and "no clock64 path" in short["why"]


@pytest.mark.parametrize("change, why", [
    (lambda a: a.__setitem__((0, 1), 0), "start or end stamp is not positive"),
    (lambda a: a.__setitem__((1, 7), a[1, 1] - 5), "end at or before they start"),
    (lambda a: a.__setitem__((2, 8), a[2, 2] + 10 ** 9), "over the 1980 MHz ceiling"),
    (lambda a: a.__setitem__((3, 7), a[3, 1] + 100), "over the 4022 GB/s ceiling"),
    (lambda a: a.__setitem__((4, 9), -1), "rows not stamped live or dead"),
    (lambda a: a.__setitem__((5, 9), 2), "the live CTAs are not the pids"),
])
def test_row_sanity_refuses_a_corrupted_launch(change, why):
    tile, N, K = P.D64, 128 * 8, 14336
    b = K * 128 * 2
    s = [1000 + 2 * i for i in range(8)]
    a = rows_of(s, [x + b / 40.0 for x in s])
    change(a)
    lr = R.launch(a, [0], 64, N, K, tile, o=0.0, expected_live=8, q_guard=P.q_bytes(tile, 2))
    assert not lr["valid"] and why in lr["why"], lr["why"]


@pytest.mark.parametrize("kw, why", [
    ({"o": float("nan")}, "overhead is unread"),
    ({"o": None}, "overhead is unread"),
    ({"launch_ms": float("nan")}, "launch_ms nan"),
    ({"launch_ms": 0.0}, "launch_ms 0.0"),
    ({"launch_ms": -1.0}, "launch_ms -1.0"),
    ({"expected_live": 16}, "the cell's geometry gives 16"),
    ({"use_td": True}, "USE_TD True"),
    ({"late": 2}, "late_buffers 2"),
])
def test_a_launch_with_an_unread_or_impossible_record_is_not_read(kw, why):
    tile, N, K = P.D64, 128 * 8, 14336
    b = K * 128 * 2
    s = [1000 + 2 * i for i in range(8)]
    a = rows_of(s, [x + b / 40.0 for x in s])
    args = {"o": 0.0, "expected_live": 8, "launch_ms": 0.1, **kw}
    lr = R.launch(a, [0], 64, N, K, tile, **args)
    assert not lr["valid"] and why in lr["why"], lr["why"]


def test_an_l2_served_cta_voids_a_flushed_launch_but_not_the_flush_off_control():
    tile, N, K = P.D64, 128 * 8, 14336
    b = K * 128 * 2
    s = [1000 + 2 * i for i in range(8)]
    a = rows_of(s, [x + b / 200.0 for x in s])   # 200 GB/s: over Q_B / 350 ns = 93.6 GB/s
    lr = R.launch(a, [0], 64, N, K, tile, o=0.0, expected_live=8, q_guard=P.q_bytes(tile, 2))
    assert not lr["valid"] and "L2-served" in lr["why"]
    off = R.launch(a, [0], 64, N, K, tile, o=0.0, expected_live=8, q_guard=P.q_bytes(tile, 2), flush=False)
    assert off["valid"]


def test_x_over_the_ceiling_voids_a_slab_once_launch_but_not_a_tread_launch():
    tile, N, K = P.D64, 128 * 64, 14336
    b = K * 128 * 2
    s = [1000 + 2 * i for i in range(64)]
    a = rows_of(s, [x + b / 80.0 for x in s])   # 64 x 80 GB/s = 5.1 TB/s
    assert "X 5" in R.launch(a, [0], 64, N, K, tile, o=0.0, expected_live=64)["why"]
    assert R.launch(a, [0], 64, N, K, tile, o=0.0, slab_once=False)["valid"]


def test_cell_gemm_reads_the_overhead_bound_the_spread_and_the_8_cta_rule():
    tile, N, K = P.D64, 128 * 8, 14336
    b = K * 128 * 2

    def lrs(rate_per_call, n=8):
        out = []
        for c, r in enumerate(rate_per_call):
            s = [1000 + 2 * i for i in range(n)]
            out.append(R.launch(rows_of(s, [x + b / r for x in s]), [0], 64, 128 * n, K, tile, o=0.0, expected_live=n))
        return out
    ok = R.cell_gemm(lrs([40, 40, 40]), 8, {"median": 0.01, "worst": 0.02})
    assert ok["status"] == "READ" and math.isclose(ok["r"], 40.0, rel_tol=1e-4)
    assert R.cell_gemm(lrs([40, 40, 40]), 8, {"median": 0.11, "worst": 0.12})["status"].startswith("NOT READ: overhead")
    assert "over C3's NOT-READ bound" in R.cell_gemm(lrs([40, 40, 40]), 8, {"median": 0.01, "worst": 0.21})["status"]
    assert "unread" in R.cell_gemm(lrs([40, 40, 40]), 8, {"median": float("nan"), "worst": 0.0})["status"]
    assert "per-call spread" in R.cell_gemm(lrs([40, 40, 44]), 8, {"median": 0.0, "worst": 0.0})["status"]
    assert R.cell_gemm(lrs([40, 40, 41.6]), 8, {"median": 0.0, "worst": 0.0})["status"] == "READ"      # 4% <= 5% at G 8
    assert "per-call spread" in R.cell_gemm(lrs([40, 40, 41.6], 16), 16, {"median": 0.0, "worst": 0.0})["status"]
    small = R.cell_gemm(lrs([40, 40, 40], 2), 2, {"median": 0.0, "worst": 0.0})
    assert "6 plateau CTAs, under 8" in small["status"]
    assert R.cell_gemm(lrs([40] * 6, 2), 2, {"median": 0.0, "worst": 0.0})["status"] == "READ"   # 6 calls


# ================================================================== the fits

def test_f_ladder_beta_q_beta_f_and_transition_on_planted_numbers():
    rungs = [(f"F{b}", b, 5000.0 + b / 40.0) for b in (1e5, 2e5, 4e5, 8e5)]
    fl = R.f_ladder(rungs)
    assert fl["status"] == "READ" and math.isclose(fl["c0_us"], 5.0) and math.isclose(fl["r_inf_gbps"], 40.0)
    assert R.f_ladder(rungs[:3])["status"].startswith("NOT IDENTIFIED: 3 of 4")
    noisy = [(n, b, lam * (1 + 0.3 * (-1) ** i)) for i, (n, b, lam) in enumerate(rungs)]
    assert "standard error" in R.f_ladder(noisy)["status"]
    bq = R.beta_q([("a", 10.0, 16384), ("b", 20.0, 32768), ("c", 30.0, 49152)], 3)
    assert bq["status"] == "READ" and math.isclose(bq["beta_Q"], 1.0, rel_tol=1e-9) and bq["q_ratio"] == 3.0
    assert R.beta_q([("a", 10.0, 16384), ("b", None, 32768), ("c", 30.0, 49152)], 3)["status"].startswith("NOT")
    assert R.beta_q([("a", 10.0, None), ("b", 20.0, 32768), ("c", 30.0, 49152)], 3)["status"].startswith("NOT")
    assert math.isclose(R.beta_f(30.0 * 1395 / 1693, 1395, 30.0, 1693), 1.0)
    assert math.isclose(R.beta_f(30.0, 1395, 30.0, 1693), 0.0, abs_tol=1e-12)
    assert R.beta_f(None, 1, 2, 3) is None and R.beta_f(1.0, 1400, 1.0, 1400) is None
    lad = [(f"A-G{g}", g, min(47.0, 3600.0 / g), g * min(47.0, 3600.0 / g), float(g), 32768) for g in
           (2, 4, 8, 16, 32, 64, 128, 256, 512)]
    tr = R.transition(lad, [("w1", 3600.0), ("w2", 3600.0)])
    assert tr["status_plateau"] == "READ" and tr["status_sat"] == "READ" and tr["X_sat"] == 3600.0 and tr["G_90"] == 128
    assert tr["r0"] == 47.0 and all(abs(rw["deficit"]) < 1e-9 for rw in tr["rows"])   # the sharp knee: d 0
    assert R.l_gpu_at(tr["rows"], 1000.0) is not None and R.l_gpu_at(tr["rows"], 1e6) is None
    assert R.transition(lad[:2], [])["status_sat"].startswith("NOT IDENTIFIED")


def test_life_packing_is_an_exact_split():
    x = R.life_packing(0.5, 0.45, 0.02, 2500, 396, 0.01)
    assert math.isclose(x["life"] + x["packing"], x["total"], rel_tol=1e-12)
    assert R.life_packing(0.5, None, 0.02, 2500, 396, 0.0) is None


def test_reread_pairs_partners_and_the_zero_constant_rule():
    tile, N, K = P.P128, 12800, 4096
    Pn = N // 128
    experts = np.array([0, 0, 1])                       # expert 0 holds two M-tiles: P re-reads
    s = np.arange(3 * Pn, dtype=np.float64) * 10 + 1000
    e = s + 20_000
    rows = rows_of(s, e)
    out = R.reread(rows, experts, 3 * 128, N, K, tile, w_c=264, counts=[200, 100] + [0] * 14, H=4096, D=2.0e9)
    assert out["status"] == "READ" and out["pairs"] == Pn and out["R_bytes"] == Pn * K * 128 * 2
    assert out["C_bytes"] == 2 * N * K * 2 + 300 * 4096 * 2 and out["overlap_share"] == 1.0
    assert out["rule_miss_share"] == 0.0 and out["rule_bytes"] == out["C_bytes"]
    assert math.isclose(out["phi_meas"], (2.0e9 - out["C_bytes"]) / out["R_bytes"])


# ================================================================== H1's label

@pytest.mark.parametrize("args, label, why", [
    ((1.0, 0.0, 1.0, 1.0, 1.0, 0.2, 1.0, 0.1), "SUPPORTED", "every support band"),
    ((0.4, 0.0, 1.0, 1.0, 1.0, 0.2, 1.0, 0.1), "FALSIFIED", "beta_Q 0.400 < 0.5"),
    ((1.0, 0.8, 1.0, 1.0, 1.0, 0.2, 1.0, 0.1), "FALSIFIED", "beta_f 0.800 > 0.5"),
    ((1.0, 0.0, 1.3, 1.0, 1.0, 0.2, 1.0, 0.1), "FALSIFIED", "omega 1.300 > 1.25"),
    ((1.0, 0.0, 1.0, 1.6, 1.6, 0.2, 1.0, 0.1), "FALSIFIED", "no plateau"),
    ((1.0, 0.0, 1.0, 1.6, 1.6, 0.3, 1.0, 0.1), "MIXED", "plateau"),               # the ratio at a load past 0.25 X_sat
    ((1.0, 0.0, 1.0, 1.0, 1.0, 0.2, 0.5, 0.1), "FALSIFIED", "sigma 0.500 < 0.6"),
    ((1.0, 0.0, 1.2, 1.0, 1.0, 0.2, 1.0, 0.1), "MIXED", "omega"),
    ((0.6, 0.0, 1.0, 1.0, 1.0, 0.2, 1.0, 0.1), "MIXED", "beta_Q"),
    ((1.0, 0.4, 1.0, 1.0, 1.0, 0.2, 1.0, 0.1), "MIXED", "beta_f"),
    ((1.0, 0.0, 1.0, 1.2, 1.2, 0.2, 1.0, 0.1), "MIXED", "plateau"),
    ((1.0, 0.0, 1.0, 1.0, 1.0, 0.2, 0.7, 0.1), "MIXED", "sigma"),
    ((1.0, None, 1.0, 1.0, 1.0, 0.2, 1.0, 0.1), "NOT IDENTIFIED", "unread: beta_f"),
    ((1.0, float("nan"), 1.0, 1.0, 1.0, 0.2, 1.0, 0.1), "NOT IDENTIFIED", "unread: beta_f"),
    ((1.0, 0.0, 1.0, 1.0, 1.0, 0.2, 1.0, 0.3), "NOT IDENTIFIED", "G 8 is not low load"),
    ((1.0, 0.0, 1.0, 1.0, 1.0, None, 1.0, None), "NOT IDENTIFIED", "the low-load premise"),
    ((0.4, None, None, None, None, None, None, None), "FALSIFIED", "beta_Q"),   # a read falsifier wins over unread inputs
    ((1.0, 0.0, 1.0, 1.6, 0.9, 0.2, 1.0, 0.1), "MIXED", "plateau"),   # rising with G: no falsifier, outside the band
])
def test_h1s_label_on_planted_inputs(args, label, why):
    got = P.h1_label(*args)
    assert got["label"] == label and any(why in w for w in got["why"]), got


# ================================================================== whole synthetic trees

@pytest.fixture(scope="module")
def supported(tmp_path_factory):
    pc = F.tree(tmp_path_factory.mktemp("sup"))
    return pc, R.read(pc, _skip_commit_check_for_tests=True)


def test_a_world_where_h1_holds_reads_supported_with_every_planted_number(supported):
    _pc, d = supported
    assert d["H1"]["label"] == "SUPPORTED", d["H1"]
    assert all(u["valid"] for u in d["PC-R0"].values()), {k: u["why"] for k, u in d["PC-R0"].items()}
    i = d["H1"]["inputs"]
    assert math.isclose(i["beta_Q"], 1.0, rel_tol=1e-4) and abs(i["beta_f"]) < 1e-9 and i["omega"] == 1.0
    assert i["plateau_ratio"] == 1.0 and i["sigma"] == 1.0
    r0 = 2 * 64 * 128 * 2 / F.L0_NS * 1.004   # the overhead correction (o 0.004) lifts every rate by 1 + o
    assert math.isclose(d["PC-R7"]["r0"], r0, rel_tol=1e-4)
    assert math.isclose(d["PC-R7"]["X_sat"], F.X_SAT * 1.004, rel_tol=1e-3) and d["PC-R7"]["G_90"] == 128
    assert d["PC-R2"]["G8 (synthetic F ladder)"]["status"] == "READ"
    assert abs(d["PC-R2"]["G8 (synthetic F ladder)"]["c0_us"]) < 1e-6
    assert d["PC-R9"]["ratio flush off / on"] == 1.0 and d["PC-R8"]["A-MX-T128-A4 / A-G128"]["t"] == 0.0
    assert d["PC-R8"]["does the curve price the CAL2 cell (Mixtral T128 A4)"]["reading"].startswith("CARRIES")
    assert d["PC-R6"]["A-G192"]["status"] == "READ" and d["PC-R6"]["A-G288"]["status"].startswith("NOT IDENTIFIED")
    for k in ("E1 (128x128, W_c 264)", "E2 (64x128, W_c 396)"):
        assert d["PC-R10"][k]["status"] == "READ" and all(c["status"] == "READ" for c in d["PC-R10"][k]["cells"].values())
    assert d["PC-R10"]["E1 (128x128, W_c 264)"]["cells"]["uniform-n1-native"]["pairs"] == 0
    assert d["PC-R10"]["E1 (128x128, W_c 264)"]["cells"]["uniform-n2-native"]["pairs"] == 16 * 100
    assert d["PC-R12"]["page"] == "READ" and d["PC-R12"]["rows"]["L-T128-A4 w2"]["dram_over_construction"] == 1.0
    assert "confounded" in d["PC-R12"]["confound"]
    assert all(v == 0.0 for v in d["replicate"].values())
    assert all(x["flagged_under_0.97"] == 0 for x in d["PC-R13"].values())
    assert d["PC-R3"]["G 8"]["BK128 (s2, s3)"]["status"] == "READ"
    assert d["PC-R3"]["G 384"]["BK64 (s2, s3, s4)"]["rates"]["A-G384"].startswith("uncorrected")


def test_a_clock_bound_world_reads_falsified(tmp_path):
    d = R.read(F.tree(tmp_path, F.world_clock), _skip_commit_check_for_tests=True)
    assert d["H1"]["label"] == "FALSIFIED" and any("beta_f" in w for w in d["H1"]["why"]), d["H1"]
    assert math.isclose(d["H1"]["inputs"]["beta_f"], 1.0, rel_tol=1e-3)


def test_a_warps_world_reads_mixed(tmp_path):
    d = R.read(F.tree(tmp_path, F.world_warps), _skip_commit_check_for_tests=True)
    assert d["H1"]["label"] == "MIXED" and "omega" in d["H1"]["why"][0], d["H1"]
    assert math.isclose(d["H1"]["inputs"]["omega"], 1.2, rel_tol=1e-4)
    text = R.txt(d)
    assert "H1 (per-CTA memory-level parallelism): MIXED [DESCRIPTIVE]" in text and chr(0x2014) not in text


def _copy(supported, tmp_path) -> Path:
    pc, _ = supported
    dst = tmp_path / "pc"
    shutil.copytree(pc, dst)
    return dst


def _unit(pc: Path, suffix: str) -> Path:
    return next((pc / "results").glob(f"*-{suffix}"))


def _rows(pc, suffix, name):
    return _unit(pc, suffix) / "stamps" / name


@pytest.mark.parametrize("what", ["nan overhead", "zero stamp", "negative life", "over ceiling", "l2 served"])
def test_a_corrupted_page_reads_not_read_and_h1_not_identified_never_a_fallback(supported, tmp_path, what):
    pc = _copy(supported, tmp_path)
    if what == "nan overhead":
        f = _unit(pc, "perturb-ptp") / "perturb.json"
        doc = json.loads(f.read_text())
        for c in doc["variants"][P.VARIANTS["sa"]]["cells"]:
            if c["stem"] == "A-G8" and c["gemm"] == "w2":
                c["overhead_median"] = float("nan")
        f.write_text(json.dumps(doc))
    else:
        f = _rows(pc, "instr-pc-sa", "A-G8-call1-w2.npy")
        a = np.load(f)
        if what == "zero stamp":
            a[0, 1] = 0
        elif what == "negative life":
            a[3, 7] = a[3, 1] - 10
        elif what == "over ceiling":
            a[2, 7] = a[2, 1] + 50
        else:
            a[5, 7] = a[5, 1] + (a[5, 7] - a[5, 1]) // 5   # 5x faster: over Q_B / 350 ns
        np.save(f, a)
    d = R.read(pc, _skip_commit_check_for_tests=True)
    row = next(r for r in d["PC-R1"]["A (D64, 1710)"] if r["cell"] == "A-G8")
    assert row["status"].startswith("NOT READ"), row
    assert d["H1"]["label"] == "NOT IDENTIFIED", d["H1"]
    assert d["PC-R5"]["A-G8-w8 / A-G8"]["omega"] is None and d["PC-R9"]["ratio flush off / on"] is None


@pytest.mark.parametrize("what, unit, why", [
    ("page", "sa", "the page is not the registered file"),
    ("versions", "sa", "versions"),
    ("label", "sa", "the page reads PERTURBED, the gate CLEAN"),
    ("sass", "sa", "fails C3's SASS legs"),
])
def test_a_unit_whose_page_versions_label_or_sass_is_wrong_is_invalid(supported, tmp_path, what, unit, why):
    pc = _copy(supported, tmp_path)
    f = _unit(pc, "instr-pc-sa") / "stamps.json"
    man = json.loads(f.read_text())
    if what == "page":
        man["source_sha256"] = "0" * 64
    elif what == "versions":
        man["versions"]["torch"] = "2.12.0"
    elif what == "label":
        man["page_label"] = "PERTURBED"
    else:
        man["sass"][0]["checks_c3"]["ok"] = False
    f.write_text(json.dumps(man))
    d = R.read(pc, _skip_commit_check_for_tests=True)
    assert not d["PC-R0"][unit]["valid"] and why in d["PC-R0"][unit]["why"], d["PC-R0"][unit]
    assert d["H1"]["label"] == "NOT IDENTIFIED"


def test_python_is_compared_on_major_minor_only(supported, tmp_path):
    pc = _copy(supported, tmp_path)
    f = _unit(pc, "instr-pc-sa") / "stamps.json"
    man = json.loads(f.read_text())
    man["versions"]["python"] = "3.12.9"
    f.write_text(json.dumps(man))
    assert R.read(pc, _skip_commit_check_for_tests=True)["PC-R0"]["sa"]["valid"]
    man["versions"]["python"] = "3.13.0"
    f.write_text(json.dumps(man))
    assert not R.read(pc, _skip_commit_check_for_tests=True)["PC-R0"]["sa"]["valid"]


def test_a_perturbed_page_is_read_under_c3s_rule_and_labelled(supported, tmp_path):
    pc = _copy(supported, tmp_path)
    vid = P.VARIANTS["sa"].replace("-", "_")
    g = _unit(pc, "perturb-ptp") / "gate.env"
    g.write_text(g.read_text().replace(f"GATE_{vid}=PASS", f"GATE_{vid}=FAIL"))
    f = _unit(pc, "instr-pc-sa") / "stamps.json"
    man = json.loads(f.read_text())
    man["page_label"] = "PERTURBED"
    f.write_text(json.dumps(man))
    d = R.read(pc, _skip_commit_check_for_tests=True)
    assert d["PC-R0"]["sa"]["valid"] and d["PC-R0"]["sa"]["page_label"] == "PERTURBED"
    assert all(r["page_label"] == "PERTURBED" for r in d["PC-R1"]["A (D64, 1710)"]) and d["H1"]["label"] == "SUPPORTED"


def test_a_failed_regcheck_or_hard_leg_or_a_lock_not_read_back_makes_the_unit_invalid(tmp_path):
    d = R.read(F.tree(tmp_path / "a", overrides={"regcheck": {"sa": "FAIL 2 against 3"}}),
               _skip_commit_check_for_tests=True)
    assert not d["PC-R0"]["sa"]["valid"] and "regcheck FAIL" in d["PC-R0"]["sa"]["why"]
    assert d["H1"]["label"] == "NOT IDENTIFIED"
    d = R.read(F.tree(tmp_path / "b", overrides={"after": {"sa": 1395}}), _skip_commit_check_for_tests=True)
    assert not d["PC-R0"]["sa"]["valid"] and "read back after the tool" in d["PC-R0"]["sa"]["why"]
    d = R.read(F.tree(tmp_path / "c", overrides={"after": {"ptp": 1395}}), _skip_commit_check_for_tests=True)
    assert not d["PC-R0"]["sd"]["valid"] and "its perturb unit's lock read back" in d["PC-R0"]["sd"]["why"]
    assert d["PC-R0"]["sc"]["valid"]


def test_the_board_gates_decode_units_on_both_locks_and_phi_units_on_g_band(tmp_path):
    d = R.read(F.tree(tmp_path / "a", overrides={"locks": "LOCK_PROBE_1410=held\n"}), _skip_commit_check_for_tests=True)
    assert not d["PC-R0"]["sa"]["valid"] and "LOCK_PROBE_1710=held" in d["PC-R0"]["sa"]["why"]
    assert d["PC-R0"]["se1"]["valid"] and d["PC-R0"]["se2"]["valid"]
    d = R.read(F.tree(tmp_path / "b", overrides={"gband": "FAIL"}), _skip_commit_check_for_tests=True)
    assert d["PC-R0"]["sa"]["valid"] and not d["PC-R0"]["se1"]["valid"] and not d["PC-R0"]["se2"]["valid"]
    assert d["PC-R10"]["E1 (128x128, W_c 264)"]["status"].startswith("NOT READ") and d["H1"]["label"] == "SUPPORTED"
    d = R.read(F.tree(tmp_path / "c", overrides={"power": "C2_POWER_W=900\n"}), _skip_commit_check_for_tests=True)
    assert not any(u["valid"] for u in d["PC-R0"].values())


def test_a_census_off_the_registered_occupancy_leaves_the_wave_readings_not_identified(tmp_path):
    d = R.read(F.tree(tmp_path, overrides={"census": {"64x128x64w4s3": 2, "64x128x64w8s3": 2}}),
               _skip_commit_check_for_tests=True)
    assert d["PC-R7"]["status_sat"].startswith("NOT IDENTIFIED") and d["PC-R7"]["X_sat"] is None
    assert d["PC-R10"]["E2 (64x128, W_c 396)"]["status"].startswith("NOT IDENTIFIED")
    assert d["PC-R10"]["E1 (128x128, W_c 264)"]["status"] == "READ"
    # the owner's triad fallback: X_sat unread, the low-load premise reads the triad, labelled FALLBACK
    low = d["H1"]["low_load_premise"]
    assert low["fallback"] is True and low["gbps"] == P.TRIAD_FALLBACK_GBPS and low["denominator"].startswith("FALLBACK")
    assert d["H1"]["label"] == "SUPPORTED" and any("FALLBACK" in w for w in d["H1"]["why"]), d["H1"]
    assert math.isclose(low["fraction"], low["X_G8"] / P.TRIAD_FALLBACK_GBPS)


def test_a_missing_unit_directory_is_not_read(supported, tmp_path):
    pc = _copy(supported, tmp_path)
    shutil.rmtree(_unit(pc, "instr-pc-sd"))
    d = R.read(pc, _skip_commit_check_for_tests=True)
    assert not d["PC-R0"]["sd"]["valid"] and "no stamp page" in d["PC-R0"]["sd"]["why"]
    assert d["H1"]["label"] == "SUPPORTED"   # D16 enters no H1 input


# ================================================================== the stand-in readout on C3's pages

#: the design's section 1.1 table (COMPUTED by the designer from the two published stamp pages)
DESIGN_11 = {"mixtral-8x7b": [(672, 22.3), (634, 16.4), (778, 16.6), (4828, 15.5)],
             "qwen2-57b-a14b": [(633, 17.3), (2917, 14.4), (768, 15.0), (71611, 14.1)]}


def test_the_reader_reproduces_the_designs_section_1_1_table_on_c3s_published_pages():
    d = R.standin(C3)
    for model, want in DESIGN_11.items():
        tbl = d["models"][model]["w2 rate by concurrency (raw, o 0: the design's section 1.1 table)"]
        for (n, r), got in zip(want, tbl.values(), strict=True):
            assert got["ctas"] == n and abs(got["r_median_gbps"] - r) <= 0.1, (model, got, r)
        assert 1325 <= d["models"][model]["f_mhz_median"] <= 1387   # the design's per-CTA clock range
    text = R.standin_txt(d)
    assert "STAND-IN READOUT" in text and chr(0x2014) not in text


def test_sigma_reads_an_h3_world_and_needs_a_read_cell():
    """Review MAJOR-4: CTAs sharing an SM at half the rate outlive the plateau; sigma reads every live CTA."""
    tile, N, K = P.D64, 128 * 160, 1024        # 160 CTAs at 10 to 20 GB/s: X under the ceiling
    b = K * 128 * 2
    sm = np.array([i % P.SMS for i in range(160)])
    shared = np.bincount(sm, minlength=P.SMS)[sm] >= 2
    s = 1000.0 + 2 * np.arange(160)
    e = s + np.where(shared, b / 10.0, b / 20.0)
    lr = R.launch(rows_of(s, e, sms=sm), [0], 64, N, K, tile, o=0.0, expected_live=160)
    got = R.sigma_of([lr, lr, lr])
    assert got["status"] == "READ" and math.isclose(got["sigma"], 0.5, rel_tol=1e-3), got
    assert got["n_median_shared"] < 160 and got["shared_ctas"] >= 20 and got["alone_ctas"] >= 20
    assert R.sigma_of([lr], "NOT READ: per-call spread")["status"].startswith("NOT IDENTIFIED: the cell is not READ")
    assert P.h1_label(1.0, 0.0, 1.0, 1.0, 1.0, 0.2, got["sigma"], 0.1)["label"] == "FALSIFIED"


def test_the_x_sat_pool_takes_long_k_flushed_w1_only(supported):
    _pc, d = supported
    pool = [lab for lab, _x in d["PC-R7"]["x_sat_pool"]]
    assert pool and all(" K " in lab for lab in pool)
    assert all(int(lab.split(" K ")[1]) >= P.X_SAT_W1_MIN_K for lab in pool if " w1 " in lab)
    assert not any("noflush" in lab or "A-G8 " in lab for lab in pool)


def test_a_few_cta_launch_takes_its_calls_w1_timer():
    """Review MAJOR-6: two CTAs in one globaltimer tick have no resolution of their own; the call's w1 lends it."""
    tile, N, K = P.D64, 128 * 2, 14336
    b = K * 128 * 2
    s = np.array([1000.0, 1000.0])
    a = rows_of(s, s + b / 40.0)
    assert not R.launch(a, [0], 64, N, K, tile, o=0.0, expected_live=2)["valid"]
    lr = R.launch(a, [0], 64, N, K, tile, o=0.0, expected_live=2, timer_ref={"res": 32, "f_long": 1400.0})
    assert lr["valid"] and lr["res_from"] == "the call's w1 launch" and np.allclose(lr["r"], 40.0, rtol=1e-3)


# ================================================================== the second review round

#: trap 11: every axis the plan sweeps, the cells that sweep it and the reading that reads them
AXES = {"G": ("PC-R7", ["A-G2", "A-G16", "A-G128", "A-G1024"]), "F": ("PC-R2", ["A-F1792-G8", "A-TP8-G128"]),
        "stages": ("PC-R3", ["A-G8-s2k64", "A-G8-s4k64", "A-G384-s4k64", "D-G16-s2"]),
        "BLOCK_K": ("PC-R3", ["A-G8-s2k128", "A-G128-s3k128"]), "warps": ("PC-R5", ["A-G8-w8", "A-G128-w8"]),
        "lock": ("PC-R4", ["C-G8", "C-G1024", "C-MX-T128-A4"]), "row fill": ("PC-R8", ["D-MX-T2-A2", "D-MX-T4-A2"]),
        "flush": ("PC-R9", None), "sharing": ("PC-R6", ["A-G160", "A-G192"]),
        "real shapes": ("PC-R8", ["A-MX-T128-A4", "A-JM-T128-A4", "A-TP2-G128"]),
        "replicate": ("replicate", ["R-G8", "R-G512"])}


def test_every_axis_the_plan_sweeps_appears_in_a_reading(supported):
    """Trap 11: G, F, stages, BLOCK_K, warps, lock, row fill, flush, sharing, the real shapes and the replicate."""
    _pc, d = supported
    swept = {c["label"] for pg in P.BLOCKS for c in P.page_doc(pg)["cells"]}
    for axis, (reading, cells) in AXES.items():
        text = json.dumps(d[reading])
        assert d[reading], axis
        for c in cells or []:
            assert c in swept and c in text, (axis, c, reading)
    assert d["PC-R9"]["ratio flush off / on"] is not None
    roles = {c["role"] for pg in P.BLOCKS for c in P.page_doc(pg)["cells"]}
    assert roles == {"ladder", "anchor CAL2/DS S-T128-A4", "anchor", "anchor half tiles", "F ladder G 8", "F ladder G 128",
                     "factorial", "factorial near saturation", "L2 control", "row fill", "replicate"}


def test_trap_13_the_per_expert_rates_and_their_flag():
    tile, N, K = P.D64, 128 * 2, 14336
    b = K * 128 * 2
    s = 1000.0 + 2 * np.arange(8)
    slow = np.array([1, 1, 0, 0, 0, 0, 0, 0], bool)       # expert 3's two CTAs (pids 0, 1) at 36 GB/s, the rest 40
    e = s + np.where(slow, b / 36.0, b / 40.0)
    lr = R.launch(rows_of(s, e), [3, 5, 6, 7], 4 * 64, N, K, tile, o=0.0, expected_live=8)
    assert list(lr["expert"]) == [3, 3, 5, 5, 6, 6, 7, 7]
    got = R.per_expert([lr, lr])
    assert got["expert_flag"] == "SPREAD OVER 5%" and math.isclose(got["expert_spread"], 4 / 40, rel_tol=1e-3)
    assert set(got["per_expert_r"]) == {3, 5, 6, 7}
    even = R.launch(rows_of(s, s + b / 40.0), [3, 5, 6, 7], 4 * 64, N, K, tile, o=0.0, expected_live=8)
    assert R.per_expert([even])["expert_flag"] == "within 5%"


def test_pc_r1_prints_the_expert_line_and_the_sharing_split(supported):
    _pc, d = supported
    row = next(r for r in d["PC-R1"]["A (D64, 1710)"] if r["cell"] == "A-G192")
    assert row["expert_flag"] == "within 5%" and len(row["per_expert_r"]) == 6
    assert row["alone_ctas"] > 0 and row["shared_ctas"] > 0 and row["r_alone"] == row["r_shared"]
    assert d["PC-R8"]["A-MX-T128-A4 / A-G128"]["anchor_expert_flag"] == "within 5%"
    assert d["H1"]["beta_Q_rates"] == {"A-G8-s2k64": "c0-corrected", "A-G8": "c0-corrected", "A-G8-s4k64": "c0-corrected"}
    assert "X_sat" in d["H1"]["X_sat_coupling"] and "FLUSHED bursts" in d["PC-R9"]["overhead_note"]
    low = d["H1"]["low_load_premise"]   # X_sat read: no fallback, nothing labelled
    assert low["fallback"] is False and low["denominator"] == "X_sat" and low["gbps"] == d["PC-R7"]["X_sat"]
    assert not any("FALLBACK" in w for w in d["H1"]["why"])


def test_the_triad_fallback_is_the_published_ruler_and_loosens_no_falsifier():
    import read_c2 as RC2
    assert P.TRIAD_FALLBACK_GBPS == RC2.TRIAD_A2_GBPS == 3726.2
    assert RC2.TRIAD_RANGE_GBPS[0] <= P.TRIAD_FALLBACK_GBPS <= RC2.TRIAD_RANGE_GBPS[1]
    # the plateau falsifier still needs X(G 16) / X_sat: with it unread a steep plateau cannot falsify
    got = P.h1_label(1.0, 0.0, 1.0, 1.6, 1.6, None, 1.0, 0.1)
    assert got["label"] == "NOT IDENTIFIED" or got["label"] == "MIXED"
    assert not any("no plateau" in w for w in got["why"])


def test_a_partial_page_is_refused(supported, tmp_path):
    pc = _copy(supported, tmp_path)
    f = _unit(pc, "instr-pc-sd") / "stamps.json"
    man = json.loads(f.read_text())
    man["complete"] = False
    f.write_text(json.dumps(man))
    d = R.read(pc, _skip_commit_check_for_tests=True)
    assert not d["PC-R0"]["sd"]["valid"] and "PARTIAL page" in d["PC-R0"]["sd"]["why"]


def test_txt_marks_a_truncated_reading():
    d = {"registration": "x", "status": "s", "pc": "p", "commit": {}, "board": {}, "census": {},
         "H1": {"label": "MIXED", "why": ["w"], "inputs": {}}}
    for k in ("PC-R0", "PC-R1", "PC-R2", "PC-R3", "PC-R4", "PC-R5", "PC-R6", "PC-R7", "PC-R8", "PC-R9", "PC-R10",
              "PC-R11", "PC-R12", "PC-R13", "PC-R14", "replicate"):
        d[k] = {}
    d["PC-R1"] = {"rows": ["x" * 30000]}
    t = R.txt(d)
    assert "[TRUNCATED: " in t and "the whole reading is in pc.read.json" in t
