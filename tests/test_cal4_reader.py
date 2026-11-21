"""The calibration rental C4's reader (scripts/scoring/cal4/read_c4.py), histogram page and registration
(docs/registered/2026-10-27-cal4-gh200): the pure part on synthetic inputs with known answers (the five-model pass
estimate against the registered bounds and round 4's four-model estimate, the two readings' bands, Lambda, the table,
the census and triad files, the frozen-chain refusal); the tp2 page (vLLM-consistent tread, unique seeds, the
generator's bytes); the registration (its constants are the reader's, its pins hold, its text is its JSON's); then
(slow, where round 4's work folder and the exfil trees are on this machine) the reader's functions on C2's published
pages as stand-ins, reproducing round 4's Mixtral-8x22B held-out mean and C2's own readings, the reader on a stand-in
tree, and the generator reproducing the registration."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
C4 = REPO / "scripts" / "scoring" / "cal4"
for p in (REPO, REPO / "scripts", C4):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import read_c4 as R  # noqa: E402
import register_c4 as RG  # noqa: E402
import hist_c4 as HC  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / f"{RG.NAME}.json").read_text())
PUB = REPO / "results" / "published"
C2_PUB = PUB / "2026-10-21-nvidia_gh200_480gb-cal2-session"
EXFIL = Path.home() / "moe-kernels-exfil" / "runs"
HAVE_CHAIN = all((R.WORK / f).is_file() for f in R.FROZEN_FILES) and all(
    (EXFIL / d / "tree/vm/results").is_dir() for d in ("gh200-tile2-20261020T2242Z", "gh200-tile2-20261021T1842Z"))
chain_only = pytest.mark.skipif(not HAVE_CHAIN, reason="round 4's work folder or the A2 and C2 exfil trees are not here")
R1 = REG["readings"]["C4-R1"]
BOUNDS = R1["bounds (the five-model estimates at X, round 4's four means held fixed; 83 cells)"]


# --------------------------------------------------------------------------
# the pure part
# --------------------------------------------------------------------------

def test_the_registered_cells_are_b2s_less_round_4s_unpriced_cell():
    cells = R.RC.b2_cells()
    reg = R.registered_cells()
    assert len(cells) == 84 and len(reg) == 83
    gone = cells[R.UNPRICED_B2["index"]]
    assert (gone["model"], gone["cls"]) == ("jetmoe-8b", "128g1") and gone not in reg
    bad = copy.deepcopy(cells)
    bad[R.UNPRICED_B2["index"]]["model"] = "phi-3.5-moe"
    with pytest.raises(AssertionError):
        R.registered_cells(bad)


def test_the_four_model_estimate_reproduces_round_4_and_the_five_model_bounds_are_the_registrations():
    four = R.RC.pass_estimates(R.CAL_HELD_OUT[R.X22], cells=R.registered_cells(),
                               base=[R.CAL_HELD_OUT[m] for m in (R.O, R.Q, R.M)])
    assert four == R.ROUND4_PASS == {"A (4 models)": 0.3798, "B (4 models)": 0.4998, "D (O, Q, X; 8x7B G1 aside)": 0.4233}
    got = {f"X {X:+.2f}": R.pass_estimates(X) for X in R.BOUNDS_X}
    assert got == BOUNDS
    # the shape of the bounds: A peaks near X 0 and is under 0.5 there (0.4974): no fifth mean opens G2 by itself
    a = [BOUNDS[f"X {X:+.2f}"]["A (5 models)"] for X in R.BOUNDS_X]
    assert max(a) == BOUNDS["X +0.00"]["A (5 models)"] < 0.5 and a[0] < 0.1 and a[-1] < 0.1


@pytest.mark.parametrize("X, band", [(0.0, "CLEAN"), (1.0, "CLEAN"), (-1.0, "CLEAN"), (1.01, "HIGH"), (1.99, "HIGH"),
                                     (2.0, "HIGH-FAR"), (3.5, "HIGH-FAR"), (-1.01, "LOW"), (-1.99, "LOW"),
                                     (-2.0, "LOW-FAR"), (-4.0, "LOW-FAR")])
def test_the_r1_bands_are_the_registered_cuts(X, band):
    o = R.outcome(X, 6)
    assert o["band"] == band and R1["means_for_B3"][band] == o["b3"]


def test_too_few_cells_or_an_invalid_board_read_not_read():
    assert R.outcome(0.1, 3)["band"] == "NOT READ" and R.outcome(0.1, 4)["band"] == "CLEAN"
    assert R.outcome(None, 6)["band"] == "NOT READ"
    o = R.outcome(0.1, 6, valid=False, why="census: 0 tc4 census files")
    assert o["band"] == "NOT READ" and o["why"] == "census: 0 tc4 census files"
    assert R1["means_for_B3"]["NOT READ"] == o["b3"]


@pytest.mark.parametrize("lam, band", [(0.0, "W2-SIDE"), (1 / 3, "W2-SIDE"), (-1 / 3, "W2-SIDE"), (0.34, "BETWEEN"),
                                       (0.66, "BETWEEN"), (2 / 3, "M4-SIDE"), (1.0, "M4-SIDE"), (4 / 3, "M4-SIDE"),
                                       (-0.34, "BEYOND-W2"), (1.34, "BEYOND-M4")])
def test_the_r2_bands_are_the_registered_cuts(lam, band):
    o = R.bytes_outcome(lam, 0.14, 6)
    assert o["band"] == band and REG["readings"]["C4-R2"]["means"][band] == o["means"]


def test_r2_is_not_read_without_a_lever_cells_or_a_valid_board():
    assert R.bytes_outcome(0.0, 0.029, 6)["band"] == "NOT READ" and "lever" in R.bytes_outcome(0.0, 0.029, 6)["why"]
    assert R.bytes_outcome(0.0, 0.03, 6)["band"] == "W2-SIDE"
    assert R.bytes_outcome(0.0, 0.14, 3)["band"] == "NOT READ"
    assert R.bytes_outcome(None, None, 0)["band"] == "NOT READ"
    o = R.bytes_outcome(0.0, 0.14, 6, valid=False, why="G-BAND reads 'FAIL'")
    assert o["band"] == "NOT READ" and o["why"] == "G-BAND reads 'FAIL'"


def test_lambda_is_zero_on_w2s_bytes_one_on_m4s_and_skips_unpriced_cells():
    cells = [{"counted": 80.0, "w2": 80.0, "m4": 100.0}, {"counted": 90.0, "w2": 90.0, "m4": 100.0}]
    assert R.lambda_of(cells)["lambda"] == 0.0 and R.lambda_of(cells)["lever"] == 0.15
    m4 = [dict(c, counted=c["m4"]) for c in cells]
    assert R.lambda_of(m4)["lambda"] == 1.0 and R.lambda_of(m4)["m4_over_counted_pct"] == 0.0
    mid = [{"counted": 90.0, "w2": 80.0, "m4": 100.0}]
    assert R.lambda_of(mid)["lambda"] == 0.5 and R.lambda_of(mid)["w2_over_counted_pct"] == -11.11
    assert R.lambda_of(cells + [{"counted": 1.0, "w2": None, "m4": 2.0}])["n"] == 2
    assert R.lambda_of([]) == {"n": 0, "lambda": None, "lever": None}


def test_the_table_is_round_4s_four_means_and_x():
    t = R.table(0.5)
    assert [t[m]["held_out_128g1_mean_pct"] for m in R.CAL_MODELS] == [0.48, -1.44, 1.25, 0.24]
    assert t[R.MODEL]["held_out_128g1_mean_pct"] == 0.5 and t["max |mean| (five models)"] == 1.44
    assert t["spread"] == 2.69 and t["mixtral-8x7b, C2 board (beside)"]["held_out_128g1_mean_pct"] == 1.49
    assert t["round-4 gate G1 (every mean within 1.0, 8x7B on C2 too; DESCRIPTIVE)"] is False
    t = R.table(None)
    assert t[R.MODEL]["held_out_128g1_mean_pct"] is None and t["spread"] == 2.69


def test_the_census_and_triad_files_are_read(tmp_path):
    d = tmp_path / "2026-10-28-nvidia_gh200_480gb-tile-census-mixtral-8x7b-tc4"
    d.mkdir()
    ent = [{"model": R.MODEL, "tile": "128x128x64w8s3", "gemm": g, "occupancy": {"ctas_per_sm": 2}} for g in ("w1", "w2")]
    (d / "tile-census.json").write_text(json.dumps({"verdict": "COMPLETE", "entries": ent}))
    got, verdict, why = R.census_file(tmp_path)
    assert got == {"w1": 2, "w2": 2} and verdict == "COMPLETE" and why == "" and R.census_ok(got)
    assert not R.census_ok(got, "INCOMPLETE") and not R.census_ok(got, None)   # the verdict must read COMPLETE
    ent[1]["occupancy"]["ctas_per_sm"] = 1
    (d / "tile-census.json").write_text(json.dumps({"verdict": "COMPLETE", "entries": ent}))
    assert not R.census_ok(R.census_file(tmp_path)[0])
    assert R.census_file(tmp_path / "none")[2] == "0 tc4 census files" and not R.census_ok({})
    c = tmp_path / "calibration" / "nvidia_gh200_480gb-buffer_gb8.0-ceilingtriad-x"
    c.mkdir(parents=True)
    (c / "measured_nvidia_gh200_480gb.yaml").write_text("card: x\nachieved_bandwidth_gbps: 3726.0\n")
    t = R.triad(tmp_path)
    assert t["C4_gbps"] == 3726.0 and t["inside_range"] is True and t["A2_gbps"] == 3726.2
    assert "0 ceilingtriad pages" in R.triad(tmp_path / "none")["why"]


def test_gpu_times_reads_the_native_cells_of_one_page(tmp_path):
    d = tmp_path / f"2026-10-28-nvidia_gh200_480gb-{R.MODEL}-b128g1-r3-counters" / "lock1410"
    d.mkdir(parents=True)
    pg = {g: {"gpu_time_ns": 1000.0 + i, "sm__cycles_elapsed.avg": 1400.0 + i} for i, g in enumerate(("w1", "w2"))}
    cells = [{"arm": "native", "n": 2, "histogram": "PT", "per_gemm": pg}, {"arm": "shared", "n": 2, "histogram": "PT", "per_gemm": pg},
             {"arm": "native", "n": 2, "histogram": None, "per_gemm": pg}]
    (d / "r3c-g1.json").write_text(json.dumps({"cells": cells}))
    got = R.gpu_times(tmp_path, "b128g1")
    assert got == {(2, "PT", "w1"): (1000.0, 1400.0), (2, "PT", "w2"): (1001.0, 1401.0),
                   (2, "balanced", "w1"): (1000.0, 1400.0), (2, "balanced", "w2"): (1001.0, 1401.0)}
    assert R.gpu_times(tmp_path / "none", "b128g1") == {}


def test_a_moved_frozen_file_is_refused_before_anything_is_imported():
    bad = copy.deepcopy(REG)
    k = next(iter(bad["frozen"]["files"]))
    bad["frozen"]["files"][k] = "0" * 64
    if HAVE_CHAIN:
        with pytest.raises(SystemExit, match="REFUSED: the frozen chain is not the registered one"):
            R.check_frozen(bad)
        short = copy.deepcopy(REG)
        short["frozen"]["files"].pop(k)
        with pytest.raises(SystemExit, match="REFUSED"):
            R.check_frozen(short)
    assert not R._CHAIN


@chain_only
def test_the_frozen_files_are_the_registered_bytes():
    assert R.frozen_hashes() == REG["frozen"]["files"]
    R.check_frozen(REG)
    dec = R.WORK / "2026-10-27/round3/work"
    assert hashlib.sha256((dec / "DECLARED4.txt").read_bytes()).hexdigest().startswith("54c8e9a2")


def test_dumps_sorts_keys():
    assert R.dumps({"b": 1, "a": {"d": 2, "c": 3}}) == '{\n "a": {\n  "c": 3,\n  "d": 2\n },\n "b": 1\n}\n'


# --------------------------------------------------------------------------
# the tp2 page
# --------------------------------------------------------------------------

def test_the_tp2_page_is_vllm_consistent_whole_and_its_seeds_are_new():
    from moe.bench.tile_resolve import default_config, resolve_tile
    from moe.spec import MODEL_CONFIGS
    doc = json.loads((REPO / R.HIST / R.PAGE).read_text())
    cfg = MODEL_CONFIGS[R.MODEL]
    assert (cfg.hidden_size, cfg.intermediate_size, cfg.num_experts, cfg.top_k, cfg.tensor_parallel) == (4096, 7168, 8, 2, 2)
    assert (doc["model"], doc["block_m"], doc["group_m"], doc["shuffle_seed"]) == (R.MODEL, 128, 1, 20261450)
    t = resolve_tile(cfg.num_experts, cfg.intermediate_size, "bf16", "NVIDIA GH200 480GB", 2 * 128 * cfg.num_experts // cfg.top_k)
    assert (t.block_m_derived, t.block_n_derived, t.block_k_derived, t.num_warps_derived, t.num_stages_derived,
            t.group_m_derived, t.provenance) == (128, 128, 64, 8, 3, 1, "vllm_default_derived")
    for n in (1, 3):   # the only 128 G 1 tread
        c = default_config(n * 128 * cfg.num_experts // cfg.top_k, cfg.num_experts)
        assert (c["BLOCK_SIZE_M"], c["GROUP_SIZE_M"]) != (128, 1)
    assert [x["label"] for x in doc["cells"]] == ["PT", "PW", "PT2", "PW2", "uniform", "balanced"]
    assert {x["n"] for x in doc["cells"]} == {2} and all(x["arms"] == ["native", "shared"] for x in doc["cells"])
    for x in doc["cells"]:
        if x["counts"] is not None:
            assert sum(x["counts"]) == 2 * 128 * cfg.num_experts and len(x["counts"]) == 8
    seeds = [x["provenance"]["seed"] for x in doc["cells"] if x.get("provenance")]
    assert seeds == [20279470, 20279471, 20279472, 20279473]
    assert all(x["provenance"]["source_model"] == "mixtral-8x7b" for x in doc["cells"] if x.get("provenance"))
    others = set()
    for f in (REPO / "docs" / "registered").rglob("*.json"):
        if f.name == R.PAGE or "hist" not in f.parent.name:
            continue
        for x in json.loads(f.read_text()).get("cells") or []:
            if isinstance(x, dict) and x.get("provenance"):
                others.add(x["provenance"].get("seed"))
    assert others and not (set(seeds) & others)


def test_the_generator_restores_skew_synth_tiles_tables():
    import skew_synth_tile as ST
    keep = (ST.MODELS, dict(ST.SHAPE_SOURCE), dict(ST.BASE), dict(ST.LABELS))
    assert HC.build("mixtral-8x7b-tp2-128g1") == (REPO / R.HIST / R.PAGE).read_text()
    assert (ST.MODELS, ST.SHAPE_SOURCE, ST.BASE, ST.LABELS) == keep and R.MODEL not in ST.MODELS


# --------------------------------------------------------------------------
# the registration
# --------------------------------------------------------------------------

def test_the_registrations_constants_are_the_readers():
    f = REG["frozen"]
    assert set(f["files"]) == set(R.FROZEN_FILES) and f["chain_repo_modules"] == list(R.CHAIN_REPO_MODULES)
    c = f["constants"]
    assert c["A2 board F1 (the fingerprint's)"] == {"per_page": {m: list(v) for m, v in R.A2_PAGE_FITS.items()},
                                                    "par": list(R.A2_PAR_128)}
    assert "p_128 6.0, S0 2.75" in c["knee"] and "sigma_gap 0.0375" in c["byte law"] and "60 MiB" in c["byte law"]
    assert c["registered UNPRICED blind cell"]["index"] == R.UNPRICED_B2["index"] == 72
    assert f["round 4 selected table, re-priced by the frozen code at registration"] == {**R.CAL_HELD_OUT, "mixtral-8x7b C2": R.M_C2_BOARD}
    assert f["round 4 pass estimate on 83 cells, reproduced"] == R.ROUND4_PASS
    fp = f["fingerprint"]
    assert fp["cells"] == ["n2/PT", "n2/PW", "n2/PT2", "n2/PW2", "n2/uniform", "n2/balanced"] and None not in fp["ms"]
    assert fp["ms"][4] == fp["ms"][5]   # uniform and balanced have the same counts
    assert json.dumps(R.CAL_HELD_OUT) in R1["table"] and f"at least {R.MIN_CELLS} of its 6" in R1["validity"]
    assert R1["bands"] == {"CLEAN": "|X| <= 1", "HIGH": "1 < X < 2", "HIGH-FAR": "X >= 2", "LOW": "-2 < X < -1",
                           "LOW-FAR": "X <= -2", "NOT READ": "the validity clause fails"}
    assert "2 CTAs per SM" in R1["validity"] and "2 CTAs per SM" in REG["readings"]["C4-R2"]["validity"]
    assert REG["power_and_lock"]["power_limit_w"] == R.POWER_W == 700 and REG["power_and_lock"]["session_lock_mhz"] == R.LOCK
    assert "No fallback limit" in REG["power_and_lock"]["rule"]
    assert REG["status"].startswith("CALIBRATION") and "certifies nothing" in REG["status"]
    for shape in ("JetMoE", "Phi-3.5", "Qwen1.5", "deepseek-v2-lite"):
        assert shape in REG["seen"]
    assert "OD13 kept" in REG["seen"] and "TP shard" in REG["seen"]
    for k in ("C4-R1", "C4-R2", "C4-S1", "C4-S2", "C4-S3", "C4-S5"):
        fn = REG["readings"][k]["read_by"].split()[1].rstrip(",")
        assert REG["readings"][k]["read_by"].startswith("scripts/scoring/cal4/read_c4.py ") and hasattr(R, fn), k
    J = R1["joint (C4-R1 band x C4-R2 band: the registered reading of the two together, for the final declared round)"]
    assert J == R.JOINT and set(J) == set(R1["bands"]) and all(set(v) == set(REG["readings"]["C4-R2"]["bands"]) for v in J.values())
    assert "cancellation" in J["CLEAN"]["M4-SIDE"] and "not supported" in J["CLEAN"]["M4-SIDE"]
    assert "pass their first held-out test" in J["CLEAN"]["W2-SIDE"] and "only the time" in J["CLEAN"]["NOT READ"]
    assert "go blind with the round-4 model" not in json.dumps(REG)
    nr = R1["means_for_B3"]["NOT READ"]
    assert "four existing CAL models without tp2" in nr and "not rented again" in nr
    assert "final declared round" in R1["means_for_B3"]["LOW-FAR"] and all("round" in v for v in R1["means_for_B3"].values())
    assert "ONE final declared model round" in REG["after"] and "iteration stops" in REG["after"]
    assert "14.66% on the sum-based measure" in REG["readings"]["C4-R2"]["validity"] and "15.09% per-cell median" in REG["readings"]["C4-R2"]["validity"]
    assert "tread-8 cells" in REG["seen"] and "verdict COMPLETE" in R1["validity"]
    assert "PREFLIGHT" in R1["refusals"]
    sk = R1["stakes (frozen model against v5'' on tp2's cells at A2's par; NOT a prediction of record)"]
    assert sk["mean %"] == -2.93 and len(sk["W2 / M4 w1 bytes - 1, %"]) == 6


def test_the_registrations_pins_hold_and_its_text_is_its_json():
    for p, s in REG["code_pins"].items():
        assert hashlib.sha256((REPO / p).read_bytes()).hexdigest() == s, p
    assert set(REG["code_pins"]) == set(RG.PINNED) and set(R.CHAIN_REPO_MODULES) <= set(REG["code_pins"])
    txt = (REPO / "docs" / "registered" / f"{RG.NAME}.txt").read_text()
    assert txt == "\n".join(RG.txt(REG)) + "\n"
    dash = chr(0x2014)
    assert dash not in txt and dash not in json.dumps(REG, ensure_ascii=False)
    for name, f in REG["pages"]["files"].items():
        assert hashlib.sha256((REPO / f["path"]).read_bytes()).hexdigest() == f["sha256"], name
    # C2's driver, the one C4's copies, is pinned by C2 and here with the same bytes
    c2 = json.loads((REPO / "docs/registered/2026-10-22-cal2-gh200.json").read_text())["code_pins"]
    for p in set(c2) & set(REG["code_pins"]):
        assert c2[p] == REG["code_pins"][p], p


def test_the_preflight_refuses_a_moved_round_4_table_or_estimate_before_the_chain(monkeypatch):
    """MAJOR 3: the reader re-prices round 4's table and its four-model estimate before any page, and refuses on any
    difference (here with the re-pricing stubbed: the refusal comes before the chain is imported)."""
    good = {**R.CAL_HELD_OUT, "mixtral-8x7b C2": R.M_C2_BOARD}
    monkeypatch.setattr(R, "round4_table", lambda: {**good, R.X22: 0.25})
    with pytest.raises(SystemExit, match="does not reproduce round 4's table"):
        R.preflight(REG)
    monkeypatch.setattr(R, "round4_table", lambda: dict(good))
    monkeypatch.setattr(R, "round4_pass", lambda: {**R.ROUND4_PASS, "A (4 models)": 0.38})
    with pytest.raises(SystemExit, match="four-model estimate does not reproduce"):
        R.preflight(REG)
    assert not R._CHAIN
    assert R.round4_pass.__name__ == "<lambda>"   # the real one is checked in the slow tests through read


def test_the_commit_paths_hold_the_registration_the_page_the_plan_and_every_pin():
    paths = R.commit_paths(REG)
    for p in ("scripts/scoring/cal4/read_c4.py", "scripts/gh200_cal4_session.sh", f"docs/registered/{RG.NAME}.json",
              f"docs/registered/{RG.NAME}.txt", f"{R.HIST}/{R.PAGE}", R.PLAN, "moe/spec.py",
              "scripts/scoring/cal2/read_c2.py", "scripts/scoring/tileA/tcommon.py"):
        assert p in paths, p


def test_read_refuses_without_the_rentals_commit_before_any_chain_import(tmp_path):
    D = tmp_path / "c4" / "session" / "gh200-driver"
    D.mkdir(parents=True)
    with pytest.raises(SystemExit, match="REFUSED"):
        R.read(tmp_path / "c4")
    assert not R._CHAIN


def test_the_pure_reader_api_loads_no_chain_module_and_leaves_sys_path():
    """The in-process tests use the pure part only: it imports nothing of the frozen chain and adds no path."""
    keep = list(sys.path)
    R.outcome(0.5, 6), R.bytes_outcome(0.1, 0.1, 6), R.table(0.5), R.lambda_of([]), R.dumps({"a": 1}), R.census_ok({})
    R.frozen_hashes() if HAVE_CHAIN else None
    assert sys.path == keep
    assert not ({"r6", "r5", "r4", "k5", "d5", "v5price", "kneefit"} & set(sys.modules))
    assert not R._CHAIN


# --------------------------------------------------------------------------
# the frozen chain on real pages (stand-ins), each in its own interpreter
# --------------------------------------------------------------------------

def job(*args, cwd=None) -> dict:
    got = subprocess.run([sys.executable, "-B", str(REPO / "tests" / "cal4_chain_job.py"), *map(str, args)],
                         capture_output=True, text=True, timeout=900, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                         cwd=cwd)
    line = next((ln for ln in got.stdout.splitlines() if ln.startswith("JSON ")), None)
    assert got.returncode == 0 and line, got.stdout[-3000:] + got.stderr[-3000:]
    return json.loads(line[5:])


@pytest.mark.slow
@chain_only
def test_the_reader_on_c2s_pages_reproduces_round_4_and_c2s_readings():
    d = job("c2")
    c2 = json.loads((REPO / "scripts/scoring/cal2/cal2.read.json").read_text())
    # the board's sag: C2's per-page fits and par (cal2.read.json F1)
    assert d["per_page"] == c2["F1"]["per_page"] and d["par"] == c2["F1"]["par"] == [41.59, 412.5]
    # Mixtral-8x22B under the frozen model: K1's SEEN constants are its LOMO fold and F1 is par of both C2 pages,
    # so the reader's X on C2's page IS round 4's held-out +0.24 (and v5'''s M4 bytes put it back near +3)
    assert d["x22_rows"] == 6 and d["x22_why"] == "" and d["x22_drops"] == [None] * 6 and d["x22_unpriced"] == []
    assert d["x22_X"] == 0.24 == R.CAL_HELD_OUT[R.X22] and d["x22_X_m4"] > 2.0
    # the sag domain and the census on 8x22B: C2's own C2-S3 and C2-S5 numbers
    assert d["x22_sag"] == c2["C2-S3 8x22B sag"]
    assert d["x22_census"] == [{"w1": 2, "w2": 2}, "COMPLETE", ""]
    # Mixtral-8x7B b128g1 on C2's board: M4 over counted per cell as C2-S4 read it (w1 -0.21, w2 +22.28), W2's w1
    # lever too small to read (P 224: every law misses), and on w2 the counters sit on W2's side
    s4 = c2["C2-S4 Mixtral-8x7B b128g1"]["C2 board"]["bytes_m4_over_counted"]
    assert abs(d["m7_m4_cells_mean"]["w1"] - s4["w1"]["mean_pct"]) <= 0.05
    assert abs(d["m7_m4_cells_mean"]["w2"] - s4["w2"]["mean_pct"]) <= 0.05
    assert d["m7_bytes"]["w1"]["lever"] < R.MIN_LEVER and d["m7_w1_outcome"] == "NOT READ"
    assert d["m7_w2_outcome"] == "W2-SIDE" and -10 < d["m7_bytes"]["w2"]["w2_over_counted_pct"] < 0
    k = d["m7_knee"]
    assert k["rows"] == 8 and k["K1"] == {"p": 6.0, "S0": 2.75, "rate_gbps": pytest.approx(3726.2, abs=0.1)}
    assert k["w1"]["cycles"]["n"] == k["w1"]["wall"]["n"] == 4 and k["w1"]["x_median (counted bytes)"] == 1.151
    # the faster-than-priced Mixtral-8x7B w1 of every round, at K1's SEEN constants, in both domains
    assert k["w1"]["cycles"]["mean_pct"] > k["w1"]["wall"]["mean_pct"] > 1.0
    assert d["chain_repo_bad"] == []


@pytest.mark.slow
@chain_only
def test_read_on_a_stand_in_tree_without_tp2_pages_reads_not_read(tmp_path):
    (tmp_path / "c4").mkdir()
    (tmp_path / "c4" / "results").symlink_to(C2_PUB / "results", target_is_directory=True)
    (tmp_path / "c4" / "session").symlink_to(C2_PUB / "session", target_is_directory=True)
    d = job("standin", tmp_path / "c4")
    assert d["fingerprint"]["equals_registered"] is True and d["commit"] == {"sha": None, "checked": False}
    assert d["board"]["valid"] is True and d["F1"]["par"] == [41.59, 412.5]
    assert d["outcome"]["band"] == "NOT READ" and d["pass_estimates"] is None
    assert "NOT SCORED" in d["C4-R1"]["page_why"] and "census: 0 tc4 census files" in d["outcome"]["why"]
    assert d["table"][R.MODEL]["held_out_128g1_mean_pct"] is None
    assert d["bytes_outcome"]["band"] == "NOT READ" and "0 mixtral-8x7b-tp2 b128g1 counter pages" in d["bytes_outcome"]["why"]
    assert d["C4-S5 census"]["priced"] == {"w1": 2, "w2": 2} and d["C4-S5 census"]["two_per_sm"] is False
    assert d["C4-S5 triad"]["inside_range"] is True
    assert d["chain_repo"]["modules_checked"] == len(R.RC.CHAIN_REPO_MODULES)
    assert "OUTCOME: NOT READ" in R.txt(d) and "BYTES OUTCOME: NOT READ" in R.txt(d)


def _standin(tmp_path):
    (tmp_path / "c4").mkdir()
    (tmp_path / "c4" / "results").symlink_to(C2_PUB / "results", target_is_directory=True)
    (tmp_path / "c4" / "session").symlink_to(C2_PUB / "session", target_is_directory=True)
    return tmp_path / "c4"


@pytest.mark.slow
@chain_only
def test_read_down_its_valid_path_on_c2s_pages_as_tp2s(tmp_path):
    """MINOR 5: tp2's page functions replaced by C2's published pages (8x22B's timed rows, a COMPLETE 2-per-SM census,
    Mixtral-8x7B's byte page): X is round 4's 8x22B +0.24, CLEAN, the five-model table and pass estimate at X, the bytes
    band NOT READ on Mixtral's under-3% w1 lever, and the joint cell CLEAN x NOT READ."""
    d = job("valid", _standin(tmp_path))
    assert d["preflight"]["round4_table"] == {**R.CAL_HELD_OUT, "mixtral-8x7b C2": R.M_C2_BOARD}
    assert d["preflight"]["round4_pass"] == R.ROUND4_PASS and d["preflight"]["chain_repo_before_pages"] == "pinned"
    assert d["C4-R1"]["X_pct"] == 0.24 and len(d["C4-R1"]["cells"]) == 6 and d["outcome"]["band"] == "CLEAN"
    assert d["outcome"]["b3"] == R.MEANS_FOR_B3["CLEAN"]
    assert d["table"][R.MODEL]["held_out_128g1_mean_pct"] == 0.24 and d["table"]["max |mean| (five models)"] == 1.44
    assert d["pass_estimates"] == R.pass_estimates(0.24)
    assert d["bytes_outcome"]["band"] == "NOT READ" and "lever" in d["bytes_outcome"]["why"]
    assert d["joint"] == {"C4-R1": "CLEAN", "C4-R2": "NOT READ", "means": R.JOINT["CLEAN"]["NOT READ"], "sequence": R.SEQUENCE}
    b = d["C4-R1"]["beside (DESCRIPTIVE)"]
    assert b["X_with_M4_bytes_pct"] > 2.0 and "X_at_C4_triad_pct" in b and "X_with_counted_bytes_pct (oracle)" in b
    assert d["C4-S2 tp2 b128g1 knee"]["rows"] == 8 and d["C4-S3 tp2 sag"]["mean_err_us"] == -8.01
    assert "JOINT (CLEAN x NOT READ)" in R.txt(d)


@pytest.mark.slow
@chain_only
def test_read_refuses_a_moved_round_4_table_before_any_page_function(tmp_path):
    d = job("refuse", _standin(tmp_path))
    assert d["refused"] and "does not reproduce round 4's table" in d["refused"] and d["touched"] == []


@pytest.mark.slow
@chain_only
def test_the_generator_reproduces_the_registration():
    got = subprocess.run([sys.executable, "-B", str(C4 / "register_c4.py"), "--check"], capture_output=True, text=True,
                         timeout=900, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert got.returncode == 0 and got.stdout.count("same") == 2, got.stdout + got.stderr
