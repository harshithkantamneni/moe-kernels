"""Tile rental A2's scoring code off the GPU (scripts/scoring/tileA2/, scripts/r3_timing_model_v4L.py,
scripts/tile_census_a2.py, scripts/tileA2_gate.py): v4 at a lock by P1's rule; the lock as data in the
rental-A scorers (a2common.scoring); the wrappers at rental A's own rules reproducing rental A's committed
scores byte for byte from the command line with no PYTHONPATH; every scorer's command line; a SYNTHETIC
lock1410 tree made from rental A's published pages (directory renamed, lock and clocks rewritten) scored
end to end, where the page glob, the band, G4 and the dips must all move with the lock (REVIEW-A2 6b);
the census merge and its rules; the probe rule; the gates; and finalise_a2."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
A2DIR = REPO / "scripts" / "scoring" / "tileA2"
TDIR = REPO / "scripts" / "scoring" / "tileA"
for p in (REPO, REPO / "scripts", A2DIR, TDIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import a2common as A2  # noqa: E402

TC = A2.TC
import census_a2 as CA  # noqa: E402
import probe_a2 as PR  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402
import r3_timing_model_v4L as V4L  # noqa: E402
import tile_census_a2 as TCA  # noqa: E402

SFA = TC._load("score_floor_a2", A2DIR / "score_floor_a2.py")
SDA = TC._load("score_dead_a2", A2DIR / "score_dead_a2.py")
SCA = TC._load("score_census_a2", A2DIR / "score_census_a2.py")
FNA = TC._load("finalise_a2", A2DIR / "finalise_a2.py")
GATE = TC._load("tileA2_gate", REPO / "scripts" / "tileA2_gate.py")

PY = sys.executable
PUB = "results/published/2026-10-20-nvidia_gh200_480gb-tileA-session"
TREE = REPO / PUB / "results"
CARD = TC.CARD
PARTS = ("census", "floor", "bytes", "rep", "dead", "e1cal")


def _env():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    return env


# --------------------------------------------------------------------------
# v4 at a lock
# --------------------------------------------------------------------------

def test_at_lock_multiplies_every_tiles_c_scale_by_1710_over_l_and_restores_it():
    prm = V4.Params(c128=900.0, F_wg={"w1": 520.0, "w2": 979.0})
    base = {t.key: V4.c_scale(t, "w1", prm) for t in (V4.R3_TILE, V4.CAL_TILE, V4.T64W8)}
    keep = V4.c_scale
    with V4L.at_lock(1410) as f:
        assert f == 1710.0 / 1410.0 and V4.CLOCK_MHZ == 1410.0 and V4L.current_lock() == 1410.0
        for t in (V4.R3_TILE, V4.CAL_TILE, V4.T64W8):
            assert V4.c_scale(t, "w1", prm) == base[t.key] * f
        with V4L.at_lock(1305):
            assert V4.c_scale(V4.R3_TILE, "w1", prm) == 1710.0 / 1305.0 and V4.CLOCK_MHZ == 1305.0
        assert V4.CLOCK_MHZ == 1410.0
    assert V4.c_scale is keep and V4.CLOCK_MHZ == 1710.0 and V4L.current_lock() == 1710.0
    with pytest.raises(ValueError):
        with V4L.at_lock(100):
            pass


def test_at_lock_1710_is_v4_bit_for_bit_on_registered_v3_predictions():
    got = V4L.check(limit=24)
    assert got["compared"] == 24 and not got["unequal_registered"] and not got["unequal_pricer"]
    assert got["fit_clock_is_1710"] and got["factor_at_1710"] == 1.0


# --------------------------------------------------------------------------
# the lock as data
# --------------------------------------------------------------------------

def test_scoring_swaps_the_one_r6common_object_the_names_and_the_glob_and_restores_them():
    C6 = TC.C6
    keep = (C6.LOCK_MHZ, C6.G4_MIN_MHZ, C6.BAND_MHZ, dict(TC.NAMES), TC.counter_page)
    r = A2.Rules.tileA2(1410)
    assert r.band == (1410 * 0.975, 1410.0) and r.g4 == 1405.0
    with A2.scoring(r):
        assert sys.modules["r6common"] is C6
        assert (C6.LOCK_MHZ, C6.G4_MIN_MHZ, C6.BAND_MHZ) == (1410.0, 1405.0, (1374.75, 1410.0))
        assert TC.NAMES["floor"] == "2026-10-20-tileA2-floor-gh200" and TC.NAMES["e1tile"] == "2026-10-20-tileB2-e1tile-gh200"
        assert TC.counter_page is not keep[4] and V4.CLOCK_MHZ == 1410.0
        assert C6.row_verdict({"ms_p50": 1.0, "clean_repeats": 9, "sm_clock_load_mhz": 1410.0}, 1.0) is None
    assert (C6.LOCK_MHZ, C6.G4_MIN_MHZ, C6.BAND_MHZ, dict(TC.NAMES), TC.counter_page) == keep
    assert C6.row_verdict({"ms_p50": 1.0, "clean_repeats": 9, "sm_clock_load_mhz": 1410.0}, 1.0) == "G4 clock 1410.0"
    fb = A2.Rules.tileA2(1410, "fallback")
    assert fb.band[0] == pytest.approx(1352.19) and A2.Rules.tileA().band == (1640.0, 1710.0)
    # rung 1710: rental A's own rules exactly (both bands), and v4L the identity
    for which in ("primary", "fallback"):
        r17 = A2.Rules.tileA2(1710, which)
        assert (r17.lock, r17.g4, r17.band) == A2.RULES_TILEA and r17.names == A2.NAMES_A2
    assert A2.LOCK_LADDER == (1710.0, 1410.0, 1305.0) and A2.LOCK_LOWEST == 1305.0
    with V4L.at_lock(1710) as f:
        assert f == 1.0


def test_the_lock_glob_reads_rental_as_pages_as_tcommon_does_at_1710():
    for model, label, G in (("olmoe-1b-7b", "f128", 64), ("mixtral-8x7b", "b128g16", 16)):
        a = TC.counter_page(TREE, model, label, G)
        b = A2.counter_page_at(TREE, model, label, G, lock=1710.0)
        assert a[1] == b[1] and a[2] == b[2] == "counted" and a[0] == b[0]
        assert A2.counter_page_at(TREE, model, label, G, lock=1410.0)[0] is None


def test_the_a2_wrappers_at_rental_as_rules_reproduce_its_committed_scores_from_the_command_line(tmp_path):
    """DESIGN-A2 6 risk 8: the wrappers change only the lock. No PYTHONPATH, PYTHONHASHSEED=0 (the committed
    census and floor JSON hold one set's order), rental A's published tree."""
    for p in PARTS:
        got = subprocess.run([PY, str(A2DIR / f"score_{p}_a2.py"), ".", f"{PUB}/results", str(tmp_path), "--rules", "tileA"],
                             cwd=REPO, env=_env(), capture_output=True, text=True, timeout=900)
        assert got.returncode == 0, (p, got.stderr[-2000:])
    for p in PARTS:
        for ext in ("json", "txt"):
            assert (tmp_path / f"{p}.score.{ext}").read_bytes() == (TDIR / f"{p}.score.{ext}").read_bytes(), (p, ext)


def test_score_floor_and_score_bytes_start_from_the_command_line_and_reproduce_rental_a(tmp_path):
    """The import-order fix (DESIGN-A2 5a, OD10): no PYTHONPATH; no verdict moves."""
    for p in ("census", "floor", "bytes"):
        got = subprocess.run([PY, str(TDIR / f"score_{p}.py"), ".", f"{PUB}/results", str(tmp_path)], cwd=REPO, env=_env(),
                             capture_output=True, text=True, timeout=900)
        assert got.returncode == 0, (p, got.stderr[-2000:])
    for p in ("floor", "bytes"):
        for ext in ("json", "txt"):
            assert (tmp_path / f"{p}.score.{ext}").read_bytes() == (TDIR / f"{p}.score.{ext}").read_bytes(), (p, ext)


SCORERS = ([TDIR / f"score_{p}.py" for p in (*PARTS, "tileB")] + [TDIR / "finalise.py"]
           + [A2DIR / f"score_{p}_a2.py" for p in (*PARTS, "convl", "probe", "tileB2", "session")]
           + [A2DIR / "finalise_a2.py", A2DIR / "decode_gate_a2.py"])


@pytest.mark.parametrize("script", SCORERS, ids=lambda p: p.name)
def test_every_scorer_prints_its_usage_and_exits_2_with_no_arguments(script):
    got = subprocess.run([PY, str(script)], cwd=REPO, env=_env(), capture_output=True, text=True, timeout=300)
    assert got.returncode == 2, got.stderr[-2000:]
    assert "python scripts/scoring/" in got.stdout and "Traceback" not in got.stderr


# --------------------------------------------------------------------------
# the synthetic lock1410 tree, from rental A's published pages (REVIEW-A2 6b)
# --------------------------------------------------------------------------

SYN_C, SYN_F = 900.0, {"w1": 520.0, "w2": 979.0}
OUT_OF_BAND_N = 9


def _counter_1410(src: Path, dst_root: Path):
    """Rental A's f128 page copied, its directory renamed lock1710 -> lock1410, the lock and the nvidia-smi
    bracket rewritten to 1410, and every GEMM's cycles a SYNTHETIC floor law (c 900, F 520 / 979: no rental-A
    cycle is fitted) with its duration at 1400 MHz (in the 1410 band), tread 9's at 1360 (out of it)."""
    import math

    from moe.spec import MODEL_CONFIGS
    p = json.loads(src.read_text())
    de = p["design"]
    cfg = MODEL_CONFIGS[de["model"]]
    bm, bk = int(de["block_m"]), int(de["block_k"])
    p["ncu"]["lock_mhz"] = 1410.0
    for k in ("smi_before", "smi_after"):
        for row in p["ncu"][k]["rows"]:
            row["clocks.sm"] = "1410"
            row["power.limit"] = "900.00"
    for c in p["cells"]:
        mhz = 1360.0 if int(c["n"]) == OUT_OF_BAND_N else 1400.0
        for g in ("w1", "w2"):
            pg = c["per_gemm"][g]
            D = int(c["declared"])
            R = math.ceil((cfg.num_experts * int(c["n"]) * bm + D * (bm - 1)) / bm)
            live = cfg.num_experts * int(c["n"]) * (int(pg["grid_size"]) // R)
            S = (cfg.hidden_size if g == "w1" else cfg.intermediate_size) // bk
            cyc = 5000.0 + math.ceil(live / 132) * (S * SYN_C + SYN_F[g])
            c["recorded"][g]["sm__cycles_elapsed.avg"] = cyc
            pg["gpu_time_ns"] = cyc / (mhz / 1e3)
            pg["dram_bytes_read"] = 0.1 * 4022.0 * pg["gpu_time_ns"]
    rel = src.relative_to(TREE)
    dst = dst_root / str(rel).replace("lock1710", "lock1410")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(p))


def _shift(v: str) -> str:
    return v if v in ("", None) else repr(float(v) - 300.0)


def _timed_1410(page_dir: Path, dst_root: Path):
    """A rental-A timed page copied: lock_mhz 1410, every clock (report rows, cells.csv repeat clocks and
    in-flight reads) 300 MHz lower, so each read sits as far from 1410 as it sat from 1710."""
    rep = json.loads((page_dir / "report.json").read_text())
    rep["lock_mhz"] = 1410.0
    rep["session_tag"] = str(rep.get("session_tag", "")).replace("-lock1710", "-lock1410")
    for row in rep["treads_table"]:
        if row.get("sm_clock_load_mhz") is not None:
            row["sm_clock_load_mhz"] = float(row["sm_clock_load_mhz"]) - 300.0
    rows = list(csv.DictReader((page_dir / "cells.csv").open(newline="")))
    for x in rows:
        x["sm_clock_load_mhz"] = _shift(x["sm_clock_load_mhz"])
        x["clock_samples_mhz"] = " ".join(_shift(v) for v in (x.get("clock_samples_mhz") or "").split())
    dst = dst_root / page_dir.relative_to(TREE)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "report.json").write_text(json.dumps(rep))
    with (dst / "cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def tree1410(tmp_path_factory):
    root = tmp_path_factory.mktemp("lock1410") / "results"
    for m in ("olmoe-1b-7b", "mixtral-8x7b"):
        _counter_1410(TREE / f"2026-10-20-{CARD}-{m}-f128-r3-counters" / "lock1710" / "r3c-g64.json", root)
    for lab in ("d9", "d15"):
        pages = sorted((TREE / f"gaps-{CARD}-olmoe-1b-7b-{lab}" / "private_weight_reference").glob("*/report.json"))
        assert len(pages) == 1
        _timed_1410(pages[0].parent, root)
    return root


def test_rental_as_rules_find_no_page_in_the_lock1410_tree(tree1410, tmp_path):
    res, _ = SFA.score(REPO, tree1410, tmp_path, A2.Rules.tileA())
    assert res["calibration"]["verdict"].startswith("NOT SCORED")
    assert all(str(v).startswith("NOT SCORED: 0 counter pages") for v in res["pages"]["f128"].values())


def test_at_1410_the_pages_are_found_the_band_moves_and_the_synthetic_floor_is_recovered(tree1410, tmp_path):
    res, lines = SFA.score(REPO, tree1410, tmp_path, A2.Rules.tileA2(1410))
    assert res["registration"] == "2026-10-20-tileA2-floor-gh200" and res["band_in_force"]["which"] == "primary"
    assert all(v.startswith("counted (") for v in res["pages"]["f128"].values()), res["pages"]
    cal = res["calibration"]
    assert cal["verdict"] == "CAL-A" and cal["c"] == pytest.approx(SYN_C, rel=1e-9)
    xc = res["cross-clock c_128 (DESCRIPTIVE, REVIEW-A2 7c)"]
    assert xc["a2_lock_mhz"] == 1410.0 and xc["difference"] == pytest.approx(SYN_C - 698.67, abs=1e-6) and "z" in xc
    assert any(x.startswith("CROSS-CLOCK c_128 (DESCRIPTIVE): A2 ") for x in lines)
    assert cal["F"]["w1"] == pytest.approx(SYN_F["w1"], abs=1e-6) and cal["F"]["w2"] == pytest.approx(SYN_F["w2"], abs=1e-6)
    # the band at 1410 keeps the 1400 MHz cells and drops tread 9's at 1360
    with A2.scoring(A2.Rules.tileA2(1410)):
        pages, _why = SFA.SF.pages_of(tree1410, "f128")
    assert pages and all(int(c["n"]) != OUT_OF_BAND_N for p in pages for c in p["cells"])
    assert any("BAND IN FORCE primary" in x for x in lines)
    # the fallback band would keep tread 9 too: it is applied only when the primary leaves no fit
    with A2.scoring(A2.Rules.tileA2(1410, "fallback")):
        fb, _ = SFA.SF.pages_of(tree1410, "f128")
    assert any(int(c["n"]) == OUT_OF_BAND_N for p in fb for c in p["cells"])
    fit, _ = SFA.calibration(tree1410, A2.Rules.tileA2(1410))
    assert fit is not None and SFA.fit_64(tree1410, A2.Rules.tileA2(1410))[0] is None


def test_at_1410_g4_and_the_dips_move_with_the_lock_and_the_dead_pages_score_end_to_end(tree1410, tmp_path):
    from_1710 = {}
    for lab in ("d9",):
        page = next((TREE / f"gaps-{CARD}-olmoe-1b-7b-{lab}" / "private_weight_reference").glob("*/report.json")).parent
        rep, raw = TC.C7.read_page(page)
        r, _ = TC.C7.rebuild(rep, raw)
        from_1710[lab] = r["g1_r7"]
    page = next((tree1410 / f"gaps-{CARD}-olmoe-1b-7b-d9" / "private_weight_reference").glob("*/report.json")).parent
    rep, raw = TC.C7.read_page(page)
    r, _ = TC.C7.rebuild(rep, raw)
    g = r["g1_r7"]
    assert g["lock_mhz"] == 1410.0 and g["non_clean_rows"] == from_1710["d9"]["non_clean_rows"]
    assert g["top_mhz"] == from_1710["d9"]["top_mhz"] - 300.0
    assert not TC.C7.dipped(1395.0, 1410.0) and TC.C7.dipped(1380.0, 1410.0) and TC.C7.dipped(1395.0, 1710.0)
    row = dict(next(t for t in r["treads_table"] if not t["lost"]), host_bound=False)   # G4 alone (d9's n 1 cells are host-bound)
    assert row["sm_clock_load_mhz"] <= 1410.0
    with A2.scoring(A2.Rules.tileA2(1410)):
        assert TC.C6.row_verdict(row, 1.0) is None
    with A2.scoring(A2.Rules.tileA()):
        assert str(TC.C6.row_verdict(row, 1.0)).startswith("G4 clock")
    out = tmp_path / "out"
    SFA.main([str(REPO), str(tree1410), str(out), "--lock", "1410"])
    res, _ = SDA.score(REPO, tree1410, out, A2.Rules.tileA2(1410))
    assert res["registration"] == "2026-10-20-tileA2-dead-gh200" and res["label"].startswith("SEEN-REPLICATE")
    assert res["pages"] == {"d9": "counted", "d15": "counted"}
    committed = json.loads((TDIR / "dead.score.json").read_text())["DEAD"]["delta_us"]
    assert res["DEAD"]["delta_us"] == committed and res["DEAD"]["c128"] == pytest.approx(SYN_C)
    with A2.scoring(A2.Rules.tileA()):
        none = TC.timed_page(tree1410, "olmoe-1b-7b", "d9", 1, TC.C3.TimedView("CLEAN"), TC.registration(REPO, "dead")["pages"]["files"]["olmoe-1b-7b-dead.json"])
    assert none[0] is not None   # the page is found (gaps directories carry no lock); G4 is per row, below


def test_the_band_and_f64w8_gates_read_the_scorers_rule(tree1410, tmp_path):
    class A:
        results, lock, session, repo, record = tree1410, 1410.0, None, REPO, None
    assert GATE._fit(A, "band")["verdict"] == "PASS"
    assert GATE._fit(A, "f64w8")["verdict"] == "FAIL"       # no f64w8 page in this tree
    A.lock = 1305.0
    assert GATE._fit(A, "band")["verdict"] == "FAIL"        # no lock1305 page


# --------------------------------------------------------------------------
# the census: tread, the n 1 / n 3 rule, equality, merge
# --------------------------------------------------------------------------

def test_the_census_treads_are_one_but_deepseeks_three():
    for t in V4.VLLM_TILES:
        assert TCA.tread_of("deepseek-v2-lite", t.bm) == 3
        for m in ("olmoe-1b-7b", "mixtral-8x7b", "qwen2-57b-a14b", "phi-3.5-moe", "qwen1.5-moe-a2.7b", "jetmoe-8b"):
            assert TCA.tread_of(m, t.bm) == 1


def test_at_tread_hands_compile_tile_the_tread_and_restores_arm_inputs():
    import private_weight_reference as PW
    keep = PW.arm_inputs
    seen = []
    PW.arm_inputs = lambda cfg, n, *a, **k: seen.append(n)
    try:
        inner = PW.arm_inputs
        with TCA.at_tread(3):
            PW.arm_inputs("cfg", 1, 128)
        assert seen == [3] and PW.arm_inputs is inner
    finally:
        PW.arm_inputs = keep


def test_the_census_dry_run_names_each_models_tread():
    got = subprocess.run([PY, str(REPO / "scripts" / "tile_census_a2.py"), "--dry-run", "--tiles", "128x128x64w8s3",
                          "--models", "olmoe-1b-7b,deepseek-v2-lite"], cwd=REPO, env=_env(), capture_output=True, text=True,
                         timeout=120)
    assert got.returncode == 2 and "deepseek-v2-lite: tread 128x128x64w8s3 n 3" in got.stdout, got.stdout + got.stderr


def _a2_census(ref: dict, *, drift=False, n3_differs=False, deepseek=True) -> dict:
    entries = [dict(e, tread=1) for e in ref["entries"]]
    if drift:
        entries[0] = dict(entries[0], regs=entries[0]["regs"] + 8)
    ds = []
    if deepseek:
        for e in ref["entries"]:
            if e["model"] == "qwen1.5-moe-a2.7b":
                ds.append(dict(e, model="deepseek-v2-lite", tread=3))
    n3 = [dict(e, tread=3) for e in ref["entries"] if e["model"] == "olmoe-1b-7b"]
    if n3_differs:
        n3[0] = dict(n3[0], shared=n3[0]["shared"] + 1024)
    return {"entries": entries + ds, "olmoe_n3_entries": n3, "errors": [], "path_mismatch": [], "verdict": "COMPLETE"}


def test_the_merge_admits_deepseek_only_on_the_n1_n3_rule_and_refuses_stack_drift():
    ref = CA.rental_a(REPO)
    doc, rec = CA.merge(ref, _a2_census(ref))
    assert rec["equality_gate"]["verdict"] == "EQUAL" and rec["deepseek_admitted"] and rec["deepseek_entries"] == 12
    assert doc["verdict"] == "COMPLETE" and len(doc["entries"]) == 72 + 12
    doc, rec = CA.merge(ref, _a2_census(ref, n3_differs=True))
    assert not rec["deepseek_admitted"] and len(doc["entries"]) == 72 and doc["verdict"] == "INCOMPLETE"
    assert "NOT SCORED" in doc["errors"][-1]
    doc, rec = CA.merge(ref, _a2_census(ref, drift=True))
    assert rec["equality_gate"]["verdict"] == "STACK DRIFT" and doc["verdict"].startswith("REFUSED")


def test_the_census_scorer_merges_and_scores_and_the_gate_reads_the_same_rule(tmp_path):
    ref = CA.rental_a(REPO)
    tree = tmp_path / "results"
    d = tree / f"2026-10-20-{CARD}-tile-census-mixtral-8x7b-ta2"
    d.mkdir(parents=True)
    (d / "tile-census.json").write_text(json.dumps(_a2_census(ref)))
    out = tmp_path / "out"
    res, lines = SCA.score(REPO, tree, out, A2.Rules.tileA2(1410))
    assert res["CENSUS"]["verdict"] == "VALID (COMPLETE)" and res["merge"]["deepseek_admitted"]
    assert "deepseek-v2-lite" in res["occupancy_by_model"] and res["registration"] == "2026-10-20-tileA2-census-gh200"
    assert (out / "census-merged").is_dir()

    class A:
        results, repo = tree, REPO
    assert GATE.g_census(A)["verdict"] == "PASS"
    (d / "tile-census.json").write_text(json.dumps(_a2_census(ref, drift=True)))
    assert GATE.g_census(A)["verdict"] == "FAIL"
    res, _ = SCA.score(REPO, tree, tmp_path / "out2", A2.Rules.tileA2(1410))
    assert res["CENSUS"]["verdict"].startswith("REFUSED")


# --------------------------------------------------------------------------
# the probe rule and the calibrate gate
# --------------------------------------------------------------------------

def _probe_tree(tmp_path: Path, lock: float, *, held=True, slip=False):
    """Rental A's d9 page standing in for a probe page (clocks shifted to `lock`), with locked_r3's
    summary.json for its tag."""
    results, session = tmp_path / "results", tmp_path / "session"
    src = next((TREE / f"gaps-{CARD}-olmoe-1b-7b-d9" / "private_weight_reference").glob("*/report.json")).parent
    shift = 1710.0 - lock
    for model, _n, _G, short in PR.PROBES:
        rep = json.loads((src / "report.json").read_text())
        rep["lock_mhz"] = lock
        base = f"probe-{short}"
        rep["session_tag"] = f"{base}-lock{int(lock)}"
        rows = list(csv.DictReader((src / "cells.csv").open(newline="")))
        for i, x in enumerate(rows):
            sl = slip and i % 2 == 0
            x["sm_clock_load_mhz"] = repr(float(x["sm_clock_load_mhz"]) - shift - (60.0 if sl else 0.0))
            x["clock_samples_mhz"] = " ".join(repr(float(v) - shift - (60.0 if sl else 0.0)) for v in x["clock_samples_mhz"].split())
        d = results / f"gaps-{CARD}-{model}-{PR.label_of(lock, short)}" / "private_weight_reference" / "run"
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps(rep))
        with (d / "cells.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        s = session / "locked-r3" / base
        s.mkdir(parents=True)
        (s / "summary.json").write_text(json.dumps({"outcome": {"held": int(lock) if held else None}}))
    return results, session


def test_a_held_probe_page_passes_and_slips_or_an_unheld_lock_fail_it(tmp_path):
    res, ses = _probe_tree(tmp_path / "a", 1410.0)
    r = PR.rung(res, ses, 1410.0, partial=True)          # the timed pages alone (no counter page written yet)
    assert r["verdict"] == "HOLDS", r["why"]
    full = PR.rung(res, ses, 1410.0)
    assert full["verdict"] == "FAILS" and [w for w in full["why"] if "P5" not in w] == []
    assert all("ms_p50" not in json.dumps(v) for v in r["pages"].values())
    res, ses = _probe_tree(tmp_path / "b", 1410.0, slip=True)
    r = PR.rung(res, ses, 1410.0, partial=True)
    assert r["verdict"] == "FAILS" and any("P4" in w for w in r["why"]) and any("P3" in w for w in r["why"])
    res, ses = _probe_tree(tmp_path / "c", 1305.0, held=False)
    r = PR.rung(res, ses, 1305.0)
    assert r["verdict"] == "FAILS" and any("P2" in w for w in r["why"])
    assert PR.rung(tmp_path / "nothing", tmp_path / "nothing", 1410.0)["verdict"] == "FAILS"


def test_the_calibrate_gate_reads_the_settled_gemm_clock_and_fails_with_no_reading(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    src = REPO / PUB / "session" / "logs" / "calibrate-mixtral-8x7b-ta.log"

    class A:
        session, lock = tmp_path, 1410.0
    (logs / "calibrate-mixtral-8x7b-ta2.log").write_text(src.read_text())
    got = GATE.g_calibrate(A)
    assert got["verdict"] == "PASS" and got["gemm_clock_mhz"] == 1440.0 and got["lowest_rung_mhz"] == 1305.0
    # build review fix 3: the gate reads the ladder's lowest rung, whatever the session's lock
    (logs / "calibrate-mixtral-8x7b-ta2.log").write_text(src.read_text().replace('"gemm_clock_mhz": 1440', '"gemm_clock_mhz": 1350'))
    A.lock = 1710.0
    assert GATE.g_calibrate(A)["verdict"] == "PASS"
    (logs / "calibrate-mixtral-8x7b-ta2.log").write_text(src.read_text().replace('"gemm_clock_mhz": 1440', '"gemm_clock_mhz": 1290'))
    assert GATE.g_calibrate(A)["verdict"] == "FAIL"
    (logs / "calibrate-mixtral-8x7b-ta2.log").write_text("RESULT: nothing\n")
    A.lock = 1410.0
    assert GATE.g_calibrate(A)["verdict"] == "FAIL"
    assert GATE.main(["calibrate", "--session", str(tmp_path)]) == 3
    assert GATE.main(["probe", "--lock", "1410"]) == 2


# --------------------------------------------------------------------------
# finalise_a2
# --------------------------------------------------------------------------

def _a2_dir(tmp_path, *, stop="GO: rental B may run", lock=1410.0, power=900.0, session=True):
    import test_tile_scorers as TS
    out = tmp_path / "out"
    TS.a_scores(out, sigma=0.0018)
    fl = json.loads((out / "floor.score.json").read_text())
    (out / "floor.score.json").write_text(json.dumps(dict(fl, lock_mhz=lock, calibration=dict(fl["calibration"], verdict="CAL-A"))))
    for part in ("census", "bytes", "rep"):
        d = json.loads((out / f"{part}.score.json").read_text())
        (out / f"{part}.score.json").write_text(json.dumps(dict(d, lock_mhz=lock)))
    (out / "e1cal.score.json").write_text(json.dumps({"OLS_fit": {"verdict": "NOT SCORED: synthetic"}, "lock_mhz": lock,
                                                      "STOP": {"verdict": stop, "why": []}}))
    if session:
        (out / "session.score.json").write_text(json.dumps({"verdict": "READ", "lock_mhz": 1410.0, "power_limit_w": power}))
    return out


def test_finalise_a2_takes_the_lock_and_power_limit_from_a2s_scores_and_refuses_any_disagreement(tmp_path):
    out = _a2_dir(tmp_path / "a")
    assert FNA.run_of(out) == (1410.0, 900)
    for kw, why in (({"session": False}, "no READ session.score.json"), ({"lock": 1710.0}, "names lock 1710"),
                    ({"power": 800.0}, "not a registered limit")):
        with pytest.raises(FNA.Refused, match=why):
            FNA.run_of(_a2_dir(tmp_path / str(len(why)), **kw))
    # re-review MINOR 4: census, bytes and rep are checked when present (and may be absent)
    (out / "rep.score.json").write_text(json.dumps({"REP": {"sigma_page": 0.0018}, "lock_mhz": 1305.0}))
    with pytest.raises(FNA.Refused, match="rep.score.json names lock 1305"):
        FNA.run_of(out)
    (out / "rep.score.json").unlink()
    assert FNA.run_of(out) == (1410.0, 900)
    dst = tmp_path / "p.json"
    assert FNA.main([str(REPO), str(_a2_dir(tmp_path / "s", stop="STOP: rental B does not run")), str(dst)]) == 2
    assert not dst.exists()
    for extra in (["--lock", "1410"], ["--power-limit-w", "900"]):
        assert FNA.main([str(REPO), str(out), str(dst), *extra]) == 2 and not dst.exists()
    sel = TC._load("finalise", TDIR / "finalise.py").selections(out)
    text = FNA.plan_b2(sel, 1410.0, 900)
    assert "- prelude lock-mhz=1410 power-limit-w=900 est=3 cap=15" in text and "gh200_tileA2_session.sh" in text
    assert "2026-10-20-tileB2-e1tile" in text and "tileA/finalise.py" not in text
    f = tmp_path / "rentalB2.plan"
    f.write_text(text)
    got = subprocess.run(["bash", str(REPO / "scripts" / "gh200_tileA2_session.sh"), "--dry-run", "--plan", str(f)],
                         capture_output=True, text=True, timeout=60, env={**os.environ, "HOME": "/nonexistent"})
    assert got.returncode == 2 and "a PLAN of 14 units" in got.stdout, got.stderr[-2000:]
    TC.use_census(None)


def test_the_session_record_reads_the_lock_and_the_power_limit_read_back(tmp_path):
    SS = TC._load("score_session_a2", A2DIR / "score_session_a2.py")
    tree = tmp_path / "results"
    ses = tmp_path / "session" / "gh200-driver"
    ses.mkdir(parents=True)
    tree.mkdir()
    pub = REPO / PUB / "session"
    (ses.parent / "clocks-at-start.txt").write_text((pub / "clocks-at-start.txt").read_text().replace("700.00 W", "900.00 W"))
    (ses.parent / "power-set.txt").write_text((pub / "power-set.txt").read_text().replace("700.00 W", "900.00 W", 1))
    res = SS.score(REPO, tree)
    assert res["verdict"].startswith("NOT READ") and "a2-lock.env" in res["verdict"]
    (ses / "a2-lock.env").write_text("A2_LOCK=1710\n")
    for L in (1710, 1410, 1305):
        (ses / f"gate-probe-{L}.json").write_text(json.dumps({"verdict": "FAIL"}))
    assert "no rung held" in SS.score(REPO, tree)["verdict"]          # the prelude's 1710 left by a failed ladder
    (ses / "gate-probe-1305.json").unlink()
    (ses / "gate-probe-1410.json").write_text(json.dumps({"verdict": "PASS"}))
    assert "the first rung that held is 1410" in SS.score(REPO, tree)["verdict"]
    (ses / "a2-lock.env").write_text("A2_LOCK=1410\n")
    res = SS.score(REPO, tree)
    assert res["verdict"] == "READ" and res["lock_mhz"] == 1410.0 and res["power_limit_w"] == 900.0
    (ses.parent / "power-set.txt").write_text((pub / "power-set.txt").read_text())
    assert "disagree" in SS.score(REPO, tree)["verdict"]
    (ses / "a2-lock.env").write_text("A2_LOCK=1500\n")
    assert "not a rung" in SS.score(REPO, tree)["verdict"]


def test_the_decode_gate_goes_only_with_a_scored_floor_calibration_whatever_the_stop_rule(tmp_path):
    DG = TC._load("decode_gate_a2", A2DIR / "decode_gate_a2.py")
    out = _a2_dir(tmp_path / "go", stop="STOP: rental B does not run")
    g = DG.gate(out)
    assert g["verdict"] == "GO" and "--tile-scores" in g["command"] and g["a2_stop_rule (printed, not read)"].startswith("STOP")
    assert DG.main([str(out)]) == 0
    (out / "floor.score.json").write_text(json.dumps({"calibration": {"verdict": "NOT SCORED: under 6 floor-bound cells"}}))
    assert DG.gate(out)["verdict"] == "WAIT" and DG.main([str(out)]) == 3
    TC.use_census(None)


def test_the_counter_clause_fails_rental_as_olmoe_f128_page_at_1710_and_passes_a_held_1410_page(tree1410, tmp_path):
    """The re-review's MAJOR 1: rental A's OLMoE timed pages held 1710 while its f128 counter page sagged."""
    res17 = tmp_path / "r17"
    src = TREE / f"2026-10-20-{CARD}-olmoe-1b-7b-f128-r3-counters"
    dst = res17 / f"2026-10-20-{CARD}-olmoe-1b-7b-lp1710f-r3-counters" / "lock1710"
    dst.mkdir(parents=True)
    (dst / "r3c-g64.json").write_text((src / "lock1710" / "r3c-g64.json").read_text())
    c = PR.counter_clause(res17, 1710.0)
    assert c["verdict"] == "FAIL" and c["primary"]["floor_bound_in_band"] == 0 and c["primary"]["cells_in_band"] == 6
    src14 = next(tree1410.rglob(f"*-{CARD}-olmoe-1b-7b-f128-r3-counters"))
    res14 = tmp_path / "r14"
    d14 = res14 / src14.name.replace("-f128-", "-lp1410f-") / "lock1410"
    d14.mkdir(parents=True)
    (d14 / "r3c-g64.json").write_text((src14 / "lock1410" / "r3c-g64.json").read_text())
    c = PR.counter_clause(res14, 1410.0)
    assert c["verdict"] == "PASS" and c["band_used"] == "primary" and c["primary"]["floor_bound_in_band"] >= 6
    assert PR.counter_clause(res14, 1305.0)["verdict"] == "FAIL"
    # partial: only the counter page written so far is read; full: the missing timed pages fail P1
    assert PR.rung(res14, tmp_path, 1410.0, partial=True)["verdict"] == "HOLDS"
    full = PR.rung(res14, tmp_path, 1410.0)
    assert full["verdict"] == "FAILS" and any(w.startswith("mixtral-8x7b-lp1410m: P1") for w in full["why"])


def test_conv_l_is_not_scored_when_a2_holds_1710(tmp_path):
    res = SCV.score(REPO, tmp_path, tmp_path / "out", 1710.0)
    assert res["CONV-L"].startswith("NOT SCORED: A2 holds rung 1710") and SCV.lines(res)


# --------------------------------------------------------------------------
# CONV-L end to end on pages made from rental 7's published Mixtral page (its cells are CONV-L's)
# --------------------------------------------------------------------------

SCV = TC._load("score_convl_a2", A2DIR / "score_convl_a2.py")
R7_SKA = sorted((REPO / "results/published/2026-10-14-nvidia_gh200_480gb-rental7-session/results"
                 / f"gaps-{CARD}-mixtral-8x7b-ska/private_weight_reference").glob("*/report.json"))[0].parent


def _convl_page(root: Path, reg: dict, unit: dict, lock: float, *, bias: float = 0.0):
    """CONV-L's page `unit` at `lock`: rental 7's 1710 rows of the page's cells (NATIVE), re-labelled to the
    CONV-L file and G; at L every time is the 1710 time times v3's predicted ratio times (1 + bias), and
    every clock 1710 - L lower."""
    page = unit["page"]
    f = reg["pages"]["files"][f"{page}.json"]
    want = {(c["label"], int(c["n"])) for c in f["cells"]}
    pred = {(p["label"], p["n"]): p for p in reg["predictions"] if p["page"] == page}
    scale = (lambda lab, n: 1.0) if lock == 1710.0 else (lambda lab, n: pred[(lab, n)]["ratio"][str(int(lock))] * (1 + bias))
    shift = 1710.0 - lock
    rep = json.loads((R7_SKA / "report.json").read_text())
    rows = [dict(t) for t in rep["treads_table"] if t["arm"] == "native" and (t["histogram"], int(t["tiles"])) in want]
    for t in rows:
        t["ms_p50"] *= scale(t["histogram"], int(t["tiles"]))
        t["sm_clock_load_mhz"] -= shift
    rep.update(treads_table=rows, lock_mhz=lock, repeat_rows=9 * len(rows), session_tag=f"cv-{unit['label']}-lock{int(lock)}")
    rep["pinned"] = dict(rep["pinned"], GROUP_SIZE_M=int(unit["G"]))
    rep["histogram"] = dict(rep["histogram"], file_sha256=f["sha256"], page=page)
    raw = []
    for x in csv.DictReader((R7_SKA / "cells.csv").open(newline="")):
        lab = x["detail"].split("histogram=", 1)[1].split(";", 1)[0]
        if x["arm"] != "native" or (lab, int(x["tiles"])) not in want:
            continue
        x["ms_p50"] = repr(float(x["ms_p50"]) * scale(lab, int(x["tiles"])))
        x["sm_clock_load_mhz"] = _shift(x["sm_clock_load_mhz"]) if shift else x["sm_clock_load_mhz"]
        if shift:
            x["clock_samples_mhz"] = " ".join(_shift(v) for v in x["clock_samples_mhz"].split())
        raw.append(x)
    d = root / f"gaps-{CARD}-{unit['model']}-{unit['label']}" / "private_weight_reference" / "run"
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps(rep))
    with (d / "cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)


@pytest.mark.parametrize("bias, verdict", [(0.0, "HOLDS"), (0.03, "FAILS")])
def test_conv_l_scores_the_per_cell_ratio_of_a_page_at_l_against_its_1710_twin(tmp_path, bias, verdict):
    reg = A2.registration(REPO, "convl")
    root = tmp_path / "results"
    for unit in reg["pages"]["units"]["at_L"]:
        if unit["model"] == "mixtral-8x7b":
            _convl_page(root, reg, unit, 1410.0, bias=bias if unit["class"] == "G8" else 0.0)
    for unit in reg["pages"]["units"]["at_1710"]:
        if unit["model"] == "mixtral-8x7b":
            _convl_page(root, reg, unit, 1710.0)
    res = SCV.score(REPO, root, tmp_path / "out", 1410.0)
    v = res["views"]["CLEAN"]
    assert v["classes"]["G1"]["verdict"] == "HOLDS" and v["classes"]["G1"]["stats"]["cells"] == 4, v
    assert v["classes"]["G8"]["verdict"] == verdict and v["classes"]["G8"]["stats"]["cells"] == 8
    assert res["CONV-L"] == verdict and v["pages"]["olmoe-1b-7b-cvo1"].startswith("NOT SCORED")
    if verdict == "HOLDS":
        assert max(abs(r["r"]) for r in v["rows"]) < 1e-9
    else:
        assert v["classes"]["G8"]["stats"]["mean"] == pytest.approx(0.03, abs=1e-9)
    assert res["reading (DESCRIPTIVE)"].startswith(f"CONV-L {verdict}, E1-CAL not scored")
    assert SCV.reading("HOLDS", "FAILS").startswith("CONV-L HOLDS and E1-CAL does not: this indicates")
