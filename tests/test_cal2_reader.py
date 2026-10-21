"""The calibration rental C2's reader (scripts/scoring/cal2/read_c2.py), histogram page and registration
(docs/registered/2026-10-22-cal2-gh200): the pure part on synthetic inputs with known answers (the ported power
against knee-v5 x22.json's twelve bounds, the bands, the table, the board preliminaries, the counter-page checks, the
frozen-chain refusal); the 8x22B page (vLLM-consistent tread, unique seeds, the generator's bytes); the registration
(its constants are the reader's, its pins hold, its text is its JSON's); then (slow, where the knee round's work
folder and A2's tree are on this machine) the reader on A2's and S1's published pages as stand-ins, reproducing the
knee round's numbers, and the generator reproducing the registration."""
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
C2 = REPO / "scripts" / "scoring" / "cal2"
for p in (REPO, REPO / "scripts", C2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import read_c2 as R  # noqa: E402
import register_c2 as RG  # noqa: E402
import hist_c2 as HC  # noqa: E402

REG = json.loads((REPO / "docs" / "registered" / f"{RG.NAME}.json").read_text())
PUB = REPO / "results" / "published"
A2_RES = PUB / "2026-10-21-nvidia_gh200_480gb-tileA2-session" / "results"
S1_RES = PUB / "2026-10-21-nvidia_gh200_480gb-settleS1-session" / "results"
HAVE_CHAIN = all((R.WORK / f).is_file() for f in R.FROZEN_FILES) and \
    (Path.home() / "moe-kernels-exfil/runs/gh200-tile2-20261020T2242Z/tree/vm/results").is_dir()
chain_only = pytest.mark.skipif(not HAVE_CHAIN, reason="the knee round's work folder or A2's exfil tree is not here")

#: knee-v5 x22.json's power_bounds (the pass-probability method's twelve numbers)
X22 = {"X +0.00": (0.0857, 0.1494, 0.8307), "X +1.00": (0.0701, 0.1262, 0.565),
       "X +1.60": (0.0517, 0.0978, 0.3026), "X +3.13": (0.0162, 0.0356, 0.0361)}


# --------------------------------------------------------------------------
# the pure part
# --------------------------------------------------------------------------

def test_the_ported_power_reproduces_x22s_twelve_bounds_bit_for_bit():
    for X, want in zip((0.0, 1.0, 1.6, 3.13), X22.values()):
        got = R.pass_estimates(X)
        assert tuple(got.values()) == want, (X, got)
    assert REG["readings"]["C2-R1"]["bounds (knee-v5 section 8; the estimates at X, v5'' + F1 par held fixed)"] == {
        k: dict(zip(("A (4 models)", "B (4 models)", "D (O, Q, X; 8x7B G1 aside)"), v)) for k, v in X22.items()}


def test_power_is_power5s_on_b2s_registered_grid_point():
    """power5's own sanity point: common 0, model-specific 0.01 (B2 registered 0.4987 with torch draws)."""
    got = R.power(R.b2_cells(), [0.0] * 3, 0.01, R=50_000)
    assert 0.45 < got["E1-TILE"] < 0.55 and set(got) >= {"E1-TILE", "pooled", *R.BLIND}


@pytest.mark.parametrize("X, band", [(0.0, "CLEAN"), (1.0, "CLEAN"), (-1.0, "CLEAN"), (1.01, "PARTIAL"), (1.99, "PARTIAL"),
                                     (2.0, "REGIME"), (3.13, "REGIME"), (-1.01, "OPPOSITE"), (-3.0, "OPPOSITE")])
def test_the_bands_are_the_registered_cuts(X, band):
    assert R.outcome(X, 6)["band"] == band
    assert REG["readings"]["C2-R1"]["means_for_B3"][band] == R.outcome(X, 6)["b3"]


def test_too_few_cells_or_an_invalid_board_read_not_read():
    assert R.outcome(0.1, 3)["band"] == "NOT READ" and R.outcome(0.1, 4)["band"] == "CLEAN"
    assert R.outcome(None, 6)["band"] == "NOT READ"
    o = R.outcome(0.1, 6, valid=False, why="G-BAND reads 'FAIL'")
    assert o["band"] == "NOT READ" and o["why"] == "G-BAND reads 'FAIL'"


def test_the_table_is_the_three_registered_means_and_x():
    t = R.heldout_table(0.5)
    assert [t[m]["held_out_128g1_mean_pct"] for m in R.CAL_MODELS] == [0.13, -0.82, 3.13]
    assert t["mixtral-8x22b"]["held_out_128g1_mean_pct"] == 0.5 and t["max |mean|"] == 3.13 and t["spread"] == 3.95
    t = R.heldout_table(None)
    assert t["mixtral-8x22b"]["held_out_128g1_mean_pct"] is None and t["spread"] == 3.95


def _session(tmp_path, power="C2_POWER_W=700\n", locks="LOCK_TIMED_OK=1\nLOCK_PROBE_1410=held\n", gate="PASS"):
    D = tmp_path / "session" / "gh200-driver"
    D.mkdir(parents=True)
    if power is not None:
        (D / "c2-power.env").write_text(power)
    (D / "locks.env").write_text(locks)
    if gate is not None:
        (D / "gate-band-1410.json").write_text(json.dumps({"gate": "band", "verdict": gate}))
    return tmp_path / "session"


@pytest.mark.parametrize("kw, why", [
    ({}, ""),
    ({"power": "C2_POWER_700=refused\n"}, "c2-power.env does not record C2_POWER_W=700"),
    ({"power": None}, "c2-power.env does not record C2_POWER_W=700"),
    ({"locks": "LOCK_PROBE_1410=refused\n"}, "locks.env has no LOCK_PROBE_1410=held"),
    ({"gate": "FAIL"}, "G-BAND reads 'FAIL'"),
    ({"gate": None}, "G-BAND reads None"),
])
def test_the_board_reads_the_power_the_lock_probe_and_g_band(tmp_path, kw, why):
    b = R.board(_session(tmp_path, **kw))
    assert b["valid"] == (why == "") and b["why"] == why, b
    assert R.board(None)["valid"] is False


def _counter_page(root, model, unit, *, bm=128, warps=8, G=1, sha="x", seed=1, cells=()):
    d = root / f"2026-10-22-nvidia_gh200_480gb-{model}-{unit}-r3-counters" / "lock1410"
    d.mkdir(parents=True)
    p = {"design": {"block_m": bm, "block_n": 128, "block_k": 64, "num_warps": warps, "num_stages": 3, "group_m": G,
                    "histogram": {"file_sha256": sha, "shuffle_seed": seed}}, "cells": list(cells)}
    (d / f"r3c-g{G}.json").write_text(json.dumps(p))
    return d


def test_a_counter_page_is_checked_against_its_tile_and_registered_file(tmp_path):
    f = REG["pages"]["files"]["mixtral-8x22b-128g1.json"]
    cells = [{"arm": "native", "n": c["n"], "histogram": None if c["label"] == "balanced" else c["label"],
              "counts_sha256": c["counts_sha256"]} for c in f["cells"]]
    _counter_page(tmp_path / "ok", R.MODEL, "b128g1", sha=f["sha256"], seed=f["shuffle_seed"], cells=cells)
    assert R.check_counter_page(tmp_path / "ok", R.MODEL, "b128g1", G=1, reg_file=f)[1] == ""
    _counter_page(tmp_path / "w4", R.MODEL, "b128g1", warps=4, sha=f["sha256"], seed=f["shuffle_seed"], cells=cells)
    assert "not 128x128x64 w8 s3 G 1" in R.check_counter_page(tmp_path / "w4", R.MODEL, "b128g1", G=1, reg_file=f)[1]
    _counter_page(tmp_path / "sha", R.MODEL, "b128g1", sha="0" * 64, seed=f["shuffle_seed"], cells=cells)
    assert "not the registered one" in R.check_counter_page(tmp_path / "sha", R.MODEL, "b128g1", G=1, reg_file=f)[1]
    bad = copy.deepcopy(cells)
    bad[0]["counts_sha256"] = "1" * 64
    _counter_page(tmp_path / "cnt", R.MODEL, "b128g1", sha=f["sha256"], seed=f["shuffle_seed"], cells=bad)
    assert "another counts sha256" in R.check_counter_page(tmp_path / "cnt", R.MODEL, "b128g1", G=1, reg_file=f)[1]
    assert "0 mixtral-8x22b b128g1 counter pages" in R.check_counter_page(tmp_path, R.MODEL, "b128g1", G=1, reg_file=f)[1]


def test_a_moved_frozen_file_is_refused_before_anything_is_imported():
    bad = copy.deepcopy(REG)
    k = next(iter(bad["frozen"]["files"]))
    bad["frozen"]["files"][k] = "0" * 64
    with pytest.raises(SystemExit, match="REFUSED: the frozen chain is not the registered one"):
        R.check_frozen(bad)
    short = copy.deepcopy(REG)
    short["frozen"]["files"].pop(k)
    if HAVE_CHAIN:
        with pytest.raises(SystemExit, match="REFUSED"):
            R.check_frozen(short)


@chain_only
def test_the_frozen_files_are_the_registered_bytes():
    assert R.frozen_hashes() == REG["frozen"]["files"]
    R.check_frozen(REG)


def test_dumps_sorts_keys():
    assert R.dumps({"b": 1, "a": {"d": 2, "c": 3}}) == '{\n "a": {\n  "c": 3,\n  "d": 2\n },\n "b": 1\n}\n'


# --------------------------------------------------------------------------
# the 8x22B page
# --------------------------------------------------------------------------

def test_the_8x22b_page_is_vllm_consistent_whole_and_its_seeds_are_new():
    from moe.bench.tile_resolve import default_config
    from moe.spec import MODEL_CONFIGS
    doc = json.loads((REPO / R.HIST / "mixtral-8x22b-128g1.json").read_text())
    cfg = MODEL_CONFIGS["mixtral-8x22b"]
    assert (doc["model"], doc["block_m"], doc["group_m"], doc["shuffle_seed"]) == ("mixtral-8x22b", 128, 1, 20261440)
    c = default_config(2 * 128 * cfg.num_experts // cfg.top_k, cfg.num_experts)
    assert (c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"], c["num_warps"], c["num_stages"], c["GROUP_SIZE_M"]) == (128, 128, 64, 8, 3, 1)
    for n in (1, 3):   # the only 128 G 1 tread
        c = default_config(n * 128 * cfg.num_experts // cfg.top_k, cfg.num_experts)
        assert (c["BLOCK_SIZE_M"], c["GROUP_SIZE_M"]) != (128, 1)
    assert [x["label"] for x in doc["cells"]] == ["PT", "PW", "PT2", "PW2", "uniform", "balanced"]
    assert {x["n"] for x in doc["cells"]} == {2} and all(x["arms"] == ["native", "shared"] for x in doc["cells"])
    for x in doc["cells"]:
        if x["counts"] is not None:
            assert sum(x["counts"]) == 2 * 128 * cfg.num_experts and len(x["counts"]) == 8
    seeds = [x["provenance"]["seed"] for x in doc["cells"] if x.get("provenance")]
    assert seeds == [20278460, 20278461, 20278462, 20278463]
    assert all(x["provenance"]["source_model"] == "mixtral-8x7b" for x in doc["cells"] if x.get("provenance"))
    others = set()
    for f in (REPO / "docs" / "registered").rglob("*.json"):
        if f.name == "mixtral-8x22b-128g1.json" or "hist" not in f.parent.name:
            continue
        for x in json.loads(f.read_text()).get("cells") or []:
            if isinstance(x, dict) and x.get("provenance"):
                others.add(x["provenance"].get("seed"))
    assert others and not (set(seeds) & others)


def test_the_generator_restores_skew_synth_tiles_tables():
    import skew_synth_tile as ST
    keep = (ST.MODELS, dict(ST.SHAPE_SOURCE), dict(ST.BASE), dict(ST.LABELS))
    assert HC.build("mixtral-8x22b-128g1") == (REPO / R.HIST / "mixtral-8x22b-128g1.json").read_text()
    assert (ST.MODELS, ST.SHAPE_SOURCE, ST.BASE, ST.LABELS) == keep and "mixtral-8x22b" not in ST.MODELS


# --------------------------------------------------------------------------
# the registration
# --------------------------------------------------------------------------

def test_the_registrations_constants_are_the_readers():
    f = REG["frozen"]["constants"]
    assert (f["p_128"], f["p_64"]) == (R.P128, R.P64) and f["R3's 1410 sag share in T0"] == [6.11, 50.0]
    assert f["A2 board F1 (the fingerprint's)"] == {"per_page": {m: list(v) for m, v in R.A2_PAGE_FITS.items()},
                                                    "par": list(R.A2_PAR_128)}
    assert set(REG["frozen"]["files"]) == set(R.FROZEN_FILES)
    assert len(REG["frozen"]["fingerprint_ms_at_A2_par"]) == 6 == len(REG["frozen"]["fingerprint_cells"])
    r1 = REG["readings"]["C2-R1"]
    assert json.dumps(R.CAL_HELD_OUT) in r1["table"] and f"at least {R.MIN_CELLS} of its 6" in r1["validity"]
    assert r1["bands"] == {"CLEAN": "|X| <= 1", "PARTIAL": "1 < X < 2", "REGIME": "X >= 2", "OPPOSITE": "X < -1",
                           "NOT READ": "the validity clause fails"}
    assert REG["power_and_lock"]["power_limit_w"] == R.POWER_W == 700 and REG["power_and_lock"]["session_lock_mhz"] == R.LOCK
    assert "No fallback limit" in REG["power_and_lock"]["rule"] and "power-set-unsettled-700.txt" in REG["power_and_lock"]["rule"]
    assert REG["status"].startswith("CALIBRATION") and "certifies nothing" in REG["status"]
    for shape in ("JetMoE", "Phi-3.5", "Qwen1.5"):
        assert shape in REG["seen"]
    assert "OD13 kept" in REG["seen"]
    for k in ("C2-R1", "C2-S1", "C2-S2", "C2-S3", "C2-S4", "C2-S5"):
        fn = REG["readings"][k]["read_by"].split()[1].rstrip(",")
        assert REG["readings"][k]["read_by"].startswith("scripts/scoring/cal2/read_c2.py ") and hasattr(R, fn), k


def test_the_registrations_pins_hold_and_its_text_is_its_json():
    for p, s in REG["code_pins"].items():
        assert hashlib.sha256((REPO / p).read_bytes()).hexdigest() == s, p
    assert set(REG["code_pins"]) == set(RG.PINNED)
    txt = (REPO / "docs" / "registered" / f"{RG.NAME}.txt").read_text()
    assert txt == "\n".join(RG.txt(REG)) + "\n"
    assert "—" not in txt and "—" not in json.dumps(REG)
    for name, f in REG["pages"]["files"].items():
        assert hashlib.sha256((REPO / f["path"]).read_bytes()).hexdigest() == f["sha256"], name


# --------------------------------------------------------------------------
# the frozen chain on real pages (stand-ins), each in its own interpreter
# --------------------------------------------------------------------------

def job(*args) -> dict:
    got = subprocess.run([sys.executable, "-B", str(REPO / "tests" / "cal2_chain_job.py"), *map(str, args)],
                         capture_output=True, text=True, timeout=900, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    line = next((ln for ln in got.stdout.splitlines() if ln.startswith("JSON ")), None)
    assert got.returncode == 0 and line, got.stdout[-3000:] + got.stderr[-3000:]
    return json.loads(line[5:])


@pytest.mark.slow
@chain_only
def test_the_reader_on_a2s_pages_reproduces_the_knee_rounds_numbers():
    d = job("a2")
    assert d["rows"] == 4 and d["why"] == "" and d["drops"] == [None] * 4
    # Mixtral-8x7B as the held-out model: its LOMO fold (p 5.25, F1 OLMoE's page alone) reads +3.13 (knee-v5)
    assert d["olmoe_fit"] == [29.65, 300.0] and d["mixtral_fold"] == 3.13
    # the C2 path on A2's board: both f128 pages, par, SEEN p: sagtau's SEEN par +2.56; the four A2 labels are all of them
    assert d["per_page"] == {m: list(v) for m, v in R.A2_PAGE_FITS.items()} and abs(d["par"][0] - 42.02) < 1e-9
    assert d["seen_par"] == 2.56 == d["x4"]
    assert (d["x"]["w1"]["median"], d["x"]["w2"]["median"]) == (1.151, 0.849)   # x22.py: 1.15 and 0.85
    assert d["olmoe_cycles"] == [-0.11, -2.41] and d["s1_cycles"] == [5.42, -0.94] and d["olmoe_sag"] == 2.93
    sp = d["split"]
    assert len(sp["cells"]) == 4
    for c in sp["cells"]:
        assert abs(c["time_err_ms"] - c["gemm_err_ms"] - c["per_call_err_ms"]) < 2e-6
    assert abs(sp["mean_time_pct"] - 3.13) < 0.01
    assert d["lock_refusal"] == "NOT SCORED: the timed page records lock_mhz 1710, not 1410"
    assert d["chain_repo_bad"] == []


@pytest.mark.slow
@chain_only
def test_read_on_a_stand_in_tree_without_8x22b_pages_reads_not_read(tmp_path):
    res = tmp_path / "c2" / "results"
    res.mkdir(parents=True)
    for d in A2_RES.glob("*-f128-r3-counters"):
        (res / d.name).symlink_to(d, target_is_directory=True)
    (res / "calibration").symlink_to(A2_RES / "calibration", target_is_directory=True)
    _session(tmp_path / "c2")
    d = job("standin", tmp_path / "c2")
    assert d["fingerprint"]["equals_registered"] is True and d["commit"] == {"sha": None, "checked": False}
    assert d["F1"]["per_page"] == {m: list(v) for m, v in R.A2_PAGE_FITS.items()}
    assert d["board"]["valid"] and d["outcome"]["band"] == "NOT READ" and d["pass_estimates"] is None
    assert "NOT SCORED" in d["C2-R1"]["page_why"] and d["table"]["mixtral-8x22b"]["held_out_128g1_mean_pct"] is None
    s4 = d["C2-S4 Mixtral-8x7B b128g1"]
    assert "0 mixtral-8x7b b128g1 counter pages" in s4["C2 board"]["why"]
    assert s4["reading"] == "NOT READ"
    assert (s4["S1 board (9da3c0)"]["w1"]["mean_pct"], s4["S1 board (9da3c0)"]["w2"]["mean_pct"]) == (5.42, -0.94)
    s6 = d["C2-S6 Mixtral-8x7B t128g1"]
    assert s6["M_pct"] is None and s6["reading"] == "NOT READ" and "NOT SCORED" in s6["why"]
    assert s6["F1 (C2 OLMoE f128 alone)"] == [29.65, 300.0]
    assert d["C2-S5 census"]["priced"] == {"w1": 2, "w2": 2} and "0 tc2 census files" in d["C2-S5 census"]["why"]
    t = d["C2-S5 triad"]
    assert t["C2_gbps"] == pytest.approx(3726.2, abs=0.1) and t["inside_range"] is True
    assert d["chain_repo"]["modules_checked"] == len(R.CHAIN_REPO_MODULES)
    assert "OUTCOME: NOT READ" in R.txt(d)


def test_the_pure_reader_api_loads_no_chain_module_and_leaves_sys_path():
    """The in-process tests use the pure part only: it imports nothing of the frozen chain and adds no path."""
    keep = list(sys.path)
    R.outcome(0.5, 6), R.heldout_table(0.5), R.s4_reading(1.0, 5.0), R.s6_reading(3.0), R.dumps({"a": 1})
    R.board(None), R.frozen_hashes() if HAVE_CHAIN else None
    assert sys.path == keep
    assert not ({"k5", "sagtau", "d5", "v5price", "kneefit"} & set(sys.modules))
    assert not R._CHAIN


@pytest.mark.slow
@chain_only
def test_the_generator_reproduces_the_registration():
    got = subprocess.run([sys.executable, "-B", str(C2 / "register_c2.py"), "--check"], capture_output=True, text=True,
                         timeout=900)
    assert got.returncode == 0 and got.stdout.count("same") == 2, got.stdout + got.stderr


@pytest.mark.parametrize("c2, s1, reading", [(5.0, 5.42, "SAME-AS-S1"), (0.3, 5.42, "BOARD-SPECIFIC"), (-1.9, 5.42, "BOARD-SPECIFIC"),
                                             (3.0, 5.42, "BETWEEN"), (2.1, 5.42, "BETWEEN"), (3.43, 5.42, "SAME-AS-S1"),
                                             (7.5, 5.42, "BEYOND-S1"), (-2.5, 5.42, "OPPOSITE"), (None, 5.42, "NOT READ"),
                                             (1.0, 2.5, "SAME-AS-S1"), (-3.0, -6.0, "BETWEEN"), (2.5, -6.0, "OPPOSITE")])
def test_s4_reads_the_registered_descriptive_cut(c2, s1, reading):
    assert R.s4_reading(c2, s1) == reading
    assert f"<= {R.S4_NEAR:g}" in REG["readings"]["C2-S4"]["reading"]


@pytest.mark.parametrize("m, reading", [(3.13, "REPRODUCES"), (2.2, "REPRODUCES"), (4.1, "REPRODUCES"), (0.0, "BOARD"),
                                        (-1.0, "BOARD"), (1.5, "BETWEEN"), (2.1, "BETWEEN"), (4.2, "ABOVE"),
                                        (-1.2, "OPPOSITE"), (None, "NOT READ")])
def test_s6_reads_the_registered_descriptive_cut(m, reading):
    assert R.s6_reading(m) == reading
    assert f"|M - 3.13| <= {R.S6_NEAR:g}" in REG["readings"]["C2-S6"]["reading"] and "changes no band" in REG["readings"]["C2-S6"]["reading"]


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def test_the_freeze_is_tied_to_the_rentals_commit(tmp_path):
    repo = tmp_path / "r"
    (repo / "a").mkdir(parents=True)
    (repo / "a" / "x.py").write_text("one\n")
    (repo / "y.json").write_text("{}\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c")
    sha = _git(repo, "rev-parse", "HEAD")
    assert R.check_commit(repo, sha, ["a/x.py", "y.json"]) == []
    (repo / "a" / "x.py").write_text("two\n")   # an edit after the rental's commit
    assert R.check_commit(repo, sha, ["a/x.py", "y.json"]) == [("a/x.py", f"differs from {sha[:12]}")]
    assert R.check_commit(repo, sha, ["z.py"]) == [("z.py", f"absent at {sha[:12]}")]
    assert R.check_commit(repo, "0" * 40, ["y.json"])[0][1].startswith("absent")
    # the commit comes from the ledger's driver start line, checked against commit.txt
    S = tmp_path / "session"
    (S / "gh200-driver").mkdir(parents=True)
    (S / "gh200-driver" / "status").write_text(f"2026-10-22T10:00:00Z driver start: commit {sha}, steps of the plan p\n")
    assert R.run_commit(S) == (sha, "")
    (S / "commit.txt").write_text("f" * 40 + "\n")
    assert R.run_commit(S)[0] is None and "commit.txt reads" in R.run_commit(S)[1]
    (S / "commit.txt").write_text(sha + "\n")
    assert R.run_commit(S) == (sha, "")
    assert R.run_commit(tmp_path)[0] is None
    paths = R.commit_paths(REG)
    for p in ("scripts/scoring/cal2/read_c2.py", "scripts/gh200_cal2_session.sh", f"docs/registered/{RG.NAME}.json",
              "scripts/scoring/tileA/tcommon.py", "moe/spec.py", "scripts/plans/rentalC2-cal-2026-10.plan"):
        assert p in paths, p
    assert set(R.CHAIN_REPO_MODULES) <= set(REG["code_pins"]) and REG["frozen"]["chain_repo_modules"] == list(R.CHAIN_REPO_MODULES)


def test_read_refuses_without_the_rentals_commit_before_any_chain_import(tmp_path):
    _session(tmp_path / "c2")
    with pytest.raises(SystemExit, match="REFUSED: the rental's commit is not on its ledger") if HAVE_CHAIN else pytest.raises(SystemExit):
        R.read(tmp_path / "c2")
    assert not R._CHAIN


def test_a_moved_pin_is_refused(tmp_path):
    bad = copy.deepcopy(REG)
    p = "scripts/scoring/tileA/tcommon.py"
    assert R.check_pins(REG, REPO) == []
    bad["code_pins"][p] = "0" * 64
    assert R.check_pins(bad, REPO) == [(p, f"sha256 under {REPO} is not the registered one")]


def test_the_triad_and_the_g_band_band_are_read(tmp_path):
    d = tmp_path / "calibration" / "nvidia_gh200_480gb-buffer_gb8.0-ceilingtriad-x"
    d.mkdir(parents=True)
    (d / "measured_nvidia_gh200_480gb.yaml").write_text("card: x\nachieved_bandwidth_gbps: 3721.4\n")
    t = R.triad(tmp_path)
    assert t["C2_gbps"] == 3721.4 and t["inside_range"] is False and t["A2_gbps"] == 3726.2
    assert "0 ceilingtriad pages" in R.triad(tmp_path / "none")["why"]
    S = _session(tmp_path / "s")
    (S / "gh200-driver" / "gate-band-1410.json").write_text(json.dumps(
        {"verdict": "PASS", "band_in_force": {"which": "fallback (registered: the primary band left joint_cF None)"}}))
    b = R.board(S)
    assert b["valid"] and b["g_band_which"].startswith("fallback")


def test_the_bands_wording_is_the_reviews():
    m = REG["readings"]["C2-R1"]["means_for_B3"]
    assert "stays at or under 0.09 at every X" in m["CLEAN"] and "not A" in m["CLEAN"]
    assert "regime or E 8 family property" in m["REGIME"]
    assert "X4" in REG["readings"]["C2-R1"]["beside X (DESCRIPTIVE)"] and "lock_mhz 1410" in REG["readings"]["C2-R1"]["refusals"]
