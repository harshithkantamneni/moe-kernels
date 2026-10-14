"""The tile stage's scaffolding off the GPU (design-tile 5, design-tile-review 6): scripts/r3_tile.py
(warps into R3's FIXED, the run id keyed by warps), scripts/dram_counter_route_tile.py (the page records
its own warps, the ncu child, the w4 / w8 run-id collision), scripts/tile_census.py (the PTX counters,
the path refusal), scripts/skew_synth_tile.py (the histogram drawer at any BLOCK_M), and the streamed
row-key rival (scripts/scoring/tileA/rowkeys_tile.py) against the pinned rowkeys.simulate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "tileA", REPO / "scripts" / "scoring" / "rental6"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import r3_tile as RT  # noqa: E402
import skew_synth_tile as ST  # noqa: E402

PY = sys.executable
HIST = "docs/registered/2026-10-14-tileA-hist/olmoe-1b-7b-128g1.json"


@pytest.fixture()
def fixed():
    """Every loaded FIXED back at R3's 8 warps after the test (the sweep module is shared state)."""
    import block_m_crossing_sweep as S1
    import private_weight_reference as R3

    import scripts.block_m_crossing_sweep as S2
    keep = (S1.FIXED["num_warps"], S2.FIXED["num_warps"], R3.SWEEP.FIXED["num_warps"])
    yield S1, S2
    S1.FIXED["num_warps"], S2.FIXED["num_warps"], R3.SWEEP.FIXED["num_warps"] = keep


# --------------------------------------------------------------------------
# r3_tile.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("argv, want, rest", [
    (["--num-warps", "4", "--model", "m"], 4, ["--model", "m"]),
    (["--model", "m", "--num-warps=8"], 8, ["--model", "m"]),
    (["--model", "m"], None, ["--model", "m"]),
])
def test_the_warps_flag_is_taken_out_of_r3s_argv(argv, want, rest, monkeypatch):
    monkeypatch.delenv(RT.ENV, raising=False)
    assert RT.take_warps(argv) == (want, rest)


def test_the_environment_carries_the_warps_to_a_child_and_bad_values_are_refused(monkeypatch):
    monkeypatch.setenv(RT.ENV, "4")
    assert RT.take_warps(["--x"]) == (4, ["--x"])
    for bad in (["--num-warps", "6"], ["--num-warps", "w"], ["--num-warps"]):
        with pytest.raises(RT.Refused):
            RT.take_warps(bad)


def test_set_warps_reaches_both_copies_of_the_sweep_module(fixed, monkeypatch):
    S1, S2 = fixed
    monkeypatch.delenv(RT.ENV, raising=False)
    assert S1.FIXED is not S2.FIXED
    RT.set_warps(4)
    assert S1.FIXED["num_warps"] == 4 and S2.FIXED["num_warps"] == 4
    import os
    assert os.environ[RT.ENV] == "4"
    import private_weight_reference as R3
    assert R3.pinned_config(128, 1, 3, 64)["num_warps"] == 4


def test_r3s_run_id_gains_warps_only_off_r3s_eight(fixed, monkeypatch):
    import private_weight_reference as R3
    keep = R3.default_run_id
    args = R3.build_parser().parse_args(["--model", "olmoe-1b-7b", "--block-m", "128", "--block-n", "128",
                                         "--group-m", "1", "--histogram", str(REPO / HIST), "--arms", "native,shared",
                                         "--declared-copies", "9"])
    try:
        base = R3.default_run_id(args, "NVIDIA GH200 480GB")
        RT.key_run_id(R3, 8)
        assert R3.default_run_id is keep
        RT.key_run_id(R3, 4)
        w4 = R3.default_run_id(args, "NVIDIA GH200 480GB")
        assert w4 != base          # the visible part is truncated; the hash covers every knob, warps included
    finally:
        R3.default_run_id = keep


@pytest.mark.parametrize("warps", [4, 8])
def test_r3_through_the_wrapper_pins_the_units_warps(warps):
    got = subprocess.run([PY, str(REPO / "scripts" / "r3_tile.py"), "--num-warps", str(warps), "--model", "olmoe-1b-7b",
                          "--block-m", "128", "--block-n", "128", "--block-k", "64", "--num-stages", "3", "--histogram",
                          HIST, "--arms", "native,shared", "--declared-copies", "9", "--repeats", "9", "--duty", "0.25",
                          "--seed", "0", "--group-m", "1", "--session-tag", "t-lock1710", "--slip-policy", "cell",
                          "--lock-mhz", "1710", "--min-clean-repeats", "0", "--ridge", "177.93", "--bandwidth-gbps",
                          "3725.1", "--dry-run"], capture_output=True, text=True, cwd=REPO, timeout=300)
    assert got.returncode == 2, got.stdout[-1500:] + got.stderr[-1500:]
    assert f"'num_warps': {warps}" in got.stdout and "'BLOCK_SIZE_N': 128" in got.stdout
    assert "reason: --dry-run was given" in got.stdout and "n_decl = 9 against n_max = 4" in got.stdout


# --------------------------------------------------------------------------
# dram_counter_route_tile.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("warps", [4, 8])
def test_the_counter_route_plan_records_its_own_warps_and_runs_the_wrapper_under_ncu(warps):
    got = subprocess.run([PY, str(REPO / "scripts" / "dram_counter_route_tile.py"), "--num-warps", str(warps), "--dry-run",
                          "--family", "r3-arms", "--model", "olmoe-1b-7b", "--block-m", "64", "--block-n", "128",
                          "--num-stages", "3", "--group-m", "64"], capture_output=True, text=True, cwd=REPO, timeout=300)
    assert got.returncode == 0, got.stdout[-1500:] + got.stderr[-1500:]
    assert f"num_warps={warps}" in got.stdout
    assert "scripts/r3_tile.py --counter-child" in got.stdout and "private_weight_reference.py --counter-child" not in got.stdout


def test_the_w4_and_w8_counter_run_ids_no_longer_collide(fixed):
    import dram_counter_route as DCR
    import dram_counter_route_tile as DT
    args = DCR.build_parser().parse_args(["--run", "--family", "r3-arms", "--model", "olmoe-1b-7b", "--block-m", "64",
                                          "--block-n", "128", "--num-stages", "3", "--group-m", "64"])
    keep = (DCR.run_id_for, DCR.r3_ncu_argv, DCR.R3_CHILD)
    try:
        w8 = DCR.run_id_for("r3-run", args, "NVIDIA GH200 480GB")
        DT.install(DCR, 4)
        w4 = DCR.run_id_for("r3-run", args, "NVIDIA GH200 480GB")
        assert w4 != w8
        assert DCR.R3_CHILD.name == "r3_tile.py"
        argv = DCR.r3_ncu_argv("ncu", Path("p.json"), Path("r.ncu-rep"), ["m"], launch_skip=1, launch_count=2)
        assert argv[-3].endswith("r3_tile.py") and argv[-2:] == ["--counter-child", "p.json"]
    finally:
        DCR.run_id_for, DCR.r3_ncu_argv, DCR.R3_CHILD = keep
    DT.install(DCR, 8)
    try:
        assert DCR.run_id_for("r3-run", args, "NVIDIA GH200 480GB") == w8
    finally:
        DCR.run_id_for, DCR.r3_ncu_argv, DCR.R3_CHILD = keep


# --------------------------------------------------------------------------
# tile_census.py
# --------------------------------------------------------------------------

def test_the_census_plan_refuses_nothing_on_a_dry_run_and_refuses_a_malformed_tile():
    ok = subprocess.run([PY, str(REPO / "scripts" / "tile_census.py"), "--dry-run", "--tiles", "128x128x64w8s3,32x128x64w4s3",
                         "--models", "olmoe-1b-7b,deepseek-v2-lite"], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert ok.returncode == 2 and "TILE CENSUS: compile only" in ok.stdout and "design occupancy 2" in ok.stdout
    bad = subprocess.run([PY, str(REPO / "scripts" / "tile_census.py"), "--tiles", "128x128", "--models", "olmoe-1b-7b"],
                         capture_output=True, text=True, cwd=REPO, timeout=120)
    assert bad.returncode == 2 and "REFUSED" in bad.stderr


def test_the_census_refuses_a_path_mismatch_and_summarises_occupancy():
    import tile_census as TCEN
    e = [{"model": "a", "tile": "128x128x64w8s3", "gemm": "w1", "mma": "wgmma", "occupancy": {"ctas_per_sm": 2}},
         {"model": "b", "tile": "128x128x64w8s3", "gemm": "w1", "mma": "wgmma", "occupancy": {"ctas_per_sm": 2}},
         {"model": "a", "tile": "32x128x64w4s3", "gemm": "w2", "mma": "wgmma", "occupancy": {"ctas_per_sm": 5}}]
    assert TCEN.path_check(e) == ["a 32x128x64w4s3 w2: compiled wgmma, the predicate says mma.sync"]
    s = TCEN.summary(e)
    assert s["128x128x64w8s3"]["one_value"] and s["128x128x64w8s3"]["values"] == [2]
    assert TCEN._gemm_of({"MUL_ROUTED_WEIGHT": True}) == "w2" and TCEN._gemm_of({}) == "w1"
    import r3_timing_model_v4 as V4
    from test_tile_v4 import R3_PTX
    with pytest.raises(ValueError):
        TCEN.wgmma_main_loop((REPO / R3_PTX).read_text())
    assert TCEN.ptx_counts((REPO / R3_PTX).read_text(), V4.R3_TILE)["wavefronts"] == 280.0


# --------------------------------------------------------------------------
# skew_synth_tile.py
# --------------------------------------------------------------------------

def test_the_committed_histogram_pages_are_the_drawers_byte_for_byte():
    got = subprocess.run([PY, str(REPO / "scripts" / "skew_synth_tile.py"), "--check"], capture_output=True, text=True,
                         cwd=REPO, timeout=300)
    assert got.returncode == 0 and got.stdout.count("same") == len(ST.PAGES), got.stdout


def test_every_tread_is_vllm_consistent_whole_and_its_seeds_unique():
    from moe.bench.tile_resolve import default_config
    from moe.spec import MODEL_CONFIGS
    seeds = set()
    for r, m, bm, G, ns, name in ST.PAGES:
        doc = json.loads(ST.path_of(r, name).read_text())
        assert (doc["block_m"], doc["group_m"]) == (bm, G)
        E = MODEL_CONFIGS[m].num_experts
        for n in ns:
            M = ST.tokens_at(m, n, bm)
            c = default_config(M, E)
            assert (c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"], c["num_warps"], c["num_stages"],
                    c["GROUP_SIZE_M"]) == (bm, 128, 64, 8, 3, G), (name, n)
            if G == 1:
                assert n < 6, (name, n)          # M4's Hopper G-1 rule (wave_split_bytes.HOPPER_G1_N)
        for c in doc["cells"]:
            k = MODEL_CONFIGS[m].top_k
            if c["counts"] is not None:
                assert sum(c["counts"]) == ST.tokens_at(m, c["n"], bm) * k
            if c.get("provenance"):
                assert c["provenance"]["seed"] not in seeds
                seeds.add(c["provenance"]["seed"])
        labs = {c["label"] for c in doc["cells"]}
        if name in ST.CONTROLS_ONLY:
            assert labs == {"uniform", "balanced"}
        else:
            assert labs == set(ST.LABELS[r]) | {"uniform", "balanced"}
    with pytest.raises(ValueError):
        ST.tokens_at("deepseek-v2-lite", 1, 128)


# --------------------------------------------------------------------------
# the streamed row-key rival
# --------------------------------------------------------------------------

def _small(bm, bn, bk, S=4):
    import rowkeys_tile as RKT
    c = next(x for x in RKT.cells("olmoe-1b-7b-128g1") if x["n"] == 1 and x["label"] == "PT")
    return dict(c, S=S)


@pytest.mark.parametrize("tile, occ", [((128, 128, 64), 2), ((32, 64, 64), 5)])
def test_the_stream_equals_the_pinned_rival_byte_for_byte(tile, occ):
    """rowkeys.simulate (pinned, its tile set in memory) and the stream on one launch cut to 4
    k-steps (half a million accesses: the pinned code's memory stays small), at two capacities."""
    import rowkeys as RK
    import rowkeys_tile as RKT
    import torch
    c = _small(*tile)
    bm, bn, bk = tile
    P = 2 * 1024 // bn
    R = -(-(sum(c["counts"]) + 64 * (bm - 1)) // bm)
    if bm != 128:
        from moe.spec import MODEL_CONFIGS
        cfg = MODEL_CONFIGS["olmoe-1b-7b"]
        counts = tuple([32] * 64)
        c = dict(c, counts=counts, tokens=256, shuffle_seed=None)
        R = -(-(sum(counts) + 64 * 31) // 32)
        assert cfg.top_k == 8
    got, _N = RKT.stream_misses("native", c["counts"], c["tokens"], 8, P, c["S"], R, c["shuffle_seed"],
                                bm=bm, bn=bn, bk=bk, occ=occ, G=1, caps_mib=(0.5, 2.0))
    old = (RK.BM, RK.BN, RK.BK)
    RK.BM, RK.BN, RK.BK = bm, bn, bk
    try:
        for C in (0.5, 2.0):
            ref, _s, _n = RK.simulate("native", c["counts"], c["tokens"], 8, P, c["S"], R, C, c["shuffle_seed"],
                                      device="cpu", dtype=torch.float64, occ=occ, G=1)
            assert got[C] == ref, (tile, C)
    finally:
        RK.BM, RK.BN, RK.BK = old


def test_native_and_shared_walk_the_same_live_ctas_at_g1():
    import numpy as np
    import rowkeys as RK
    a = RK.walk(40, 16, 1, 30)
    b = RK.walk(300, 16, 1, 30)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    # dead rows only ever extend the grid's tail, so the live order is the same at G 16 too
    g16a, g16b = RK.walk(40, 16, 16, 38), RK.walk(300, 16, 16, 38)
    assert all(np.array_equal(x, y) for x, y in zip(g16a, g16b, strict=True))


def test_the_memory_guard_stops_a_run(monkeypatch):
    import rowkeys_tile as RKT
    monkeypatch.setattr(RKT, "MAX_RSS_GB", 1e-6)
    with pytest.raises(RKT.TooBig):
        RKT.guard()


def test_the_rowkey_rows_are_the_registered_pages_cells():
    import rowkeys_tile as RKT
    doc = json.loads(RKT.OUT.read_text())
    want = {(c["page"], c["label"], c["n"]) for c in RKT.cells()}
    assert {(c["page"], c["label"], c["n"]) for c in doc["cells"]} == want and len(want) == 28
    assert all(c["peak_rss_gb"] < RKT.MAX_RSS_GB for c in doc["cells"])
    for c in doc["cells"]:
        if c["label"] in ("uniform", "balanced"):
            twin = next(x for x in doc["cells"] if x["page"] == c["page"] and x["n"] == c["n"]
                        and x["label"] == ("balanced" if c["label"] == "uniform" else "uniform"))
            assert twin["accesses"] == c["accesses"]
