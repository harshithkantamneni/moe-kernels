"""scripts/prod/frontier_op.py off the GPU (Frontier itself is never executed on the laptop; owner decision 2026-11-19:
its predictor, our profiles): Frontier's token grid, its "uniform" routing generator (the default, three seeded samples
per point) and the round robin (the opt-in "balanced" arm), the timer's plan (the FULL grid, every point, no budget),
the marker-based per-point segmentation of a synthetic timer trace, the clock flags, the points and their geometry, the
CSV in Frontier's MoE schema under both truth definitions, the partial arm, the trainer's command line and its one
shim, the batch predictions, the hold-out split and the dry run."""
from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "scripts" / "prod", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import frontier_op as FO  # noqa: E402
import moe_capture as MC  # noqa: E402
import prod_fakes as F  # noqa: E402
import prodlib as PL  # noqa: E402

#: frontier/moe_load_imbalance.py MOE_LOAD_IMBALANCE_FEATURES at db5a85b (the trainer's 14 grouped-GEMM features)
FRONTIER_FEATURES = ("total_routed_tokens", "num_experts_per_device", "hidden_dim", "expert_hidden_dim", "router_topk",
                     "model_expansion_ratio", "tokens_per_expert_avg", "tokens_to_experts_ratio",
                     "expert_utilization", "min_load_ratio", "load_imbalance_cv", "max_load_ratio", "load_entropy",
                     "load_gini_coefficient")


def test_the_config_carries_the_served_models_moe_operator(tmp_path):
    for key in ("ernie-4.5-21b-a3b", "olmoe-1b-7b"):
        p = FO.write_config(key, tmp_path)
        doc = json.loads(p.read_text())
        cfg = PL.MODELS[key].cfg
        assert (doc["hidden_size"], doc["moe_intermediate_size"], doc["num_experts"], doc["num_experts_per_tok"]) == \
            (cfg.hidden_size, cfg.intermediate_size, cfg.num_experts, cfg.top_k)
        assert p.parent == tmp_path / "data" / "config" / "models"
    assert str(FO.csv_relpath("ernie-4.5-21b-a3b")) == "compute/gh200/ernie45-21b-a3b-moe/moe_kernel_only.csv"


def test_the_token_grid_is_frontiers_get_num_tokens_to_profile():
    g = FO.token_grid(16384)
    assert len(g) == 387 and g == sorted(g, reverse=True) and g[0] == 16384 and g[-1] == 1
    for t in (1, 2, 4, 8, 1016, 1024, 1040, 2048, 2080, 4096, 4160, 8192, 8320, 16384):
        assert t in g, t
    for t in (3, 12, 1032, 2064, 4128, 8256, 16512):
        assert t not in g, t
    assert FO.token_grid(256) == sorted([1, 2, 4] + list(range(8, 257, 8)), reverse=True)
    assert FO.token_grid(20) == [20, 16, 8, 4, 2, 1]                     # max_tokens itself is appended


def test_the_balanced_arm_is_frontiers_round_robin_uniform_topk():
    ids = FO.round_robin_ids(5, 64, 6)
    assert ids.tolist()[0] == [0, 1, 2, 3, 4, 5] and ids.tolist()[1] == [6, 7, 8, 9, 10, 11]
    assert all(int(ids[t, j]) == (t * 6 + j) % 64 for t in range(5) for j in range(6))
    big = FO.round_robin_ids(1000, 64, 6)
    c = PL.counts_from_ids(big, 64)
    assert max(c) - min(c) <= 1 and sum(c) == 6000                         # balanced: Frontier's "balanced"
    assert all(len(set(r)) == 6 for r in big.tolist())


#: Frontier db5a85b's own generate_expert_routing(T, 64, 6, "uniform", seed) on the CPU (run once against the clone,
#: 2026-11-19): the ids' digest, the first token's ids and its float16 weights
FRONTIER_UNIFORM = {(136, 1): ("10c3db0e52e3e895", [54, 40, 31, 63, 36, 18],
                               [0.154175, 0.078247, 0.284912, 0.197998, 0.262207, 0.022537]),
                    (8, 0): ("44d1729b7e6ee8f3", [51, 26, 31, 37, 21, 7],
                             [0.130127, 0.112976, 0.225342, 0.243896, 0.092041, 0.196045])}


def test_the_default_routing_is_frontiers_uniform_generator_three_seeds_per_point():
    """Review 2026-11-19 M1: Frontier's grouped-GEMM profiling default is --load_distributions ["uniform"] with
    --num_samples_per_distribution 3 (main.py:282-295); the timer ports its generator verbatim."""
    assert FO.DEFAULT_ROUTING == "uniform" and FO.SAMPLES_PER_DISTRIBUTION == 3
    for (T, seed), (dig, first, w0) in FRONTIER_UNIFORM.items():
        ids, w = FO.uniform_routing(T, 64, 6, seed)
        assert PL.digest(ids) == dig and ids[0].tolist() == first and ids.dtype == np.int64
        assert w.dtype == np.float32 and np.allclose(w[0], w0, atol=1e-6)
        assert np.allclose(w.sum(axis=1), 1.0, atol=2e-3)                 # normalised in float16
        assert all(len(set(r)) == 6 for r in ids.tolist())
    src = os.environ.get("FRONTIER_SRC")
    if src and (Path(src) / "frontier" / "profiling" / "moe" / "load_distribution.py").is_file():
        sys.path.insert(0, src)
        try:
            import torch
            from frontier.profiling.moe.load_distribution import generate_expert_routing
            for T, seed in ((1, 0), (136, 1), (4096, 2)):
                wt, it = generate_expert_routing(T, 64, 6, "uniform", seed=seed)
                ids, w = FO.uniform_routing(T, 64, 6, seed)
                assert np.array_equal(it.numpy().astype(np.int64), ids)
                assert np.array_equal(wt.to(torch.float32).numpy(), w)
        finally:
            sys.path.remove(src)
    specs = FO.timer_specs("decode", 32)
    assert [s["label"] for s in specs[:4]] == ["f32s0", "f32s1", "f32s2", "f24s0"]   # main.py's order: T, then seed
    assert FO.samples_for("balanced") == [0] and FO.samples_for("uniform") == [0, 1, 2]
    ids, w = FO.routing_for("ernie-4.5-21b-a3b", 5, "balanced", 7)
    assert np.array_equal(ids, FO.round_robin_ids(5, 64, 6)) and np.allclose(w, 1 / 6)


def test_the_timer_plan_is_the_full_grid_every_point_once_at_its_branch_clock():
    """Owner decision 2026-11-19: Frontier's training data covers its FULL grid (387 points to 16384), no page, no
    budget; decode points (T <= 128) in the 1710 MHz pass, prefill in the 1410 pass, each in Frontier's order. The
    default routing samples each point three times (uniform); the opt-in balanced arm once."""
    assert len(FO.timer_specs("decode")) == 3 * 19 and len(FO.timer_specs("prefill")) == 3 * 368
    assert FO.timer_specs("prefill", points=[64, 1024, 8192]) == \
        [{"T": T, "seed": s, "label": f"f{T}s{s}"} for T in (1024, 8192) for s in range(3)]     # the probe's
    dec = FO.timer_plan("ernie-4.5-21b-a3b", "decode", routing="balanced")
    pre = FO.timer_plan("ernie-4.5-21b-a3b", "prefill", routing="balanced")
    Ts = [c["T"] for c, _i in dec + pre]
    assert len(dec) == 19 and len(pre) == 368 and len(Ts) == 387 == len(set(Ts))
    assert Ts == [T for T in FO.token_grid(16384) if T <= 128] + [T for T in FO.token_grid(16384) if T > 128] \
        and Ts[:19] == sorted(Ts[:19], reverse=True)                     # Frontier's order (descending) per pass
    assert sorted(Ts) == sorted(FO.token_grid(16384))
    assert all(c["branch"] == "decode" for c, _i in dec) and all(c["branch"] == "prefill" for c, _i in pre)
    assert FO.BRANCH_LOCK == {"decode": 1710, "prefill": 1410}
    assert (FO.WARMUP_STEPS, FO.ACTIVE_STEPS) == (2, 20)                   # Frontier's moe_wrapper.py:46-47
    assert MC.step_grid(FO.SINK_INDEX)[1] == 65535 and FO.SINK_INDEX > len(pre)
    c, ids = pre[-1]
    assert c["label"] == "f136" and ids.shape == (136, 6) and c["counts"] == PL.counts_from_ids(ids, 64)
    assert c["frontier"] == {"num_tokens": 136, "load_distribution": "balanced", "seed": 0}
    assert len(FO.timer_plan("ernie-4.5-21b-a3b", "decode", 256)) == 3 * 19
    u = FO.timer_plan("ernie-4.5-21b-a3b", "decode", 8)
    assert [c["label"] for c, _i in u] == ["f8s0", "f8s1", "f8s2", "f4s0", "f4s1", "f4s2", "f2s0", "f2s1", "f2s2",
                                           "f1s0", "f1s1", "f1s2"]
    assert u[0][0]["frontier"] == {"num_tokens": 8, "load_distribution": "uniform", "seed": 0}
    assert u[0][0]["ids_digest"] == FRONTIER_UNIFORM[(8, 0)][0]
    with pytest.raises(PL.KitError):
        FO.grid_cells("ernie-4.5-21b-a3b", [8], "zipf")


def test_clock_flags_mark_a_point_off_the_lock_by_more_than_one_step():
    assert FO.clock_flag(1710, [1710, 1710]) is None and FO.clock_flag(1410, [1395, 1425]) is None
    assert "under the 1710 MHz lock" in FO.clock_flag(1710, [1710, 1680])
    assert "over the 1410 MHz lock" in FO.clock_flag(1410, [1440, 1410])
    assert FO.clock_flag(1410, [None, 1410]) == "no clock reading"
    r = FO.point_record(3, {"label": "f8", "T": 8}, 1710, [1710, 1650])
    assert r["idx"] == 3 and r["sm_mhz"] == [1710, 1650] and "1650 MHz under" in r["off_lock"]


def _timer_pass(root, branch, *, max_tokens=16384, warmup=1, repeats=3, model="ernie-4.5-21b-a3b", readings=None,
                drop_marker=(), drop_sink=(), routing="balanced", points=None, sqlite_dir=None):
    """The timer.json and the nsys sqlite a timer pass leaves (run_timer's launch order, eager): per point the sink
    marker, the routing build and `warmup` calls, the point's marker (gridY = index + 1), `repeats` calls; every
    call through the functional API (9 fill kernels before align)."""
    cfg = PL.MODELS[model].cfg
    lock = FO.BRANCH_LOCK[branch]
    d = root / f"timer-{branch}"
    d.mkdir(parents=True)
    ev, expect, pts, t = [], {}, [], 1_000_000

    def mark(i):
        nonlocal t
        ev.append({"start": t, "end": t + 1000, "stream": F.STREAM, "name": F.NAMES["stepmark"], "gx": 1,
                   "gy": MC.step_grid(i)[1], "kind": "kernel", "role": "stepmark"})
        t += 1500
    for i, (c, ids) in enumerate(FO.timer_plan(model, branch, max_tokens, routing, points=points)):
        call = {"n": c["T"], "ids": ids, "layer": 0}
        if i not in drop_sink:
            mark(FO.SINK_INDEX)
        for _ in range(warmup):
            e, t, _x = F.call_events(cfg, call, t, drop_marker=True, api_fills=9)
            ev += [dict(x, start=x["start"], end=x["end"] + 3000) for x in e]   # warm-up: other durations
        if i not in drop_marker:
            mark(i)
        for _ in range(repeats):
            e, t, x = F.call_events(cfg, call, t, drop_marker=True, api_fills=9)
            ev += e
        expect[c["label"]] = x
        pts.append(FO.point_record(i, c, lock, (readings or {}).get(i, [lock, lock])))
    doc = {"schema": FO.TIMER_SCHEMA, "model": model, "branch": branch, "lock_mhz": lock, "routing": routing,
           "samples": len(FO.samples_for(routing)), "max_tokens": max_tokens, "warmup": warmup, "repeats": repeats,
           "seed": 0, "sink_index": FO.SINK_INDEX, "points": pts, "verdict": "PASS"}
    (d / "timer.json").write_text(json.dumps(doc))
    sq = d / "timer.sqlite"
    if sqlite_dir is not None:                      # main.sh: the trace outside the pushed results tree
        sq = Path(sqlite_dir) / f"timer-{branch}" / "timer.sqlite"
        sq.parent.mkdir(parents=True, exist_ok=True)
    F.write_sqlite(sq, ev, [], eager=True)
    return d, expect


def test_each_point_gets_exactly_its_own_timed_calls_by_its_marker(tmp_path):
    """The point's marker (gridY = index + 1) opens its segment, the next point's SINK marker closes it: the
    warm-up calls (longer here) and the routing build never count; the time is the core kernel-sum per call
    (align..moe_sum, the functional API's 9 fills stripped and reported), n = the timed calls."""
    d, expect = _timer_pass(tmp_path, "decode", max_tokens=64, repeats=4)
    r = FO.timer_pass_points("ernie-4.5-21b-a3b", d)
    assert r["problems"] == [] and len(r["points"]) == r["planned"] == 11
    for p in r["points"]:
        x = expect[p["label"]]
        assert p["n"] == 4 and p["source"] == "eager" and p["fills_n"] == 9
        assert round(p["kernel_sum_ms"]["median"] * 1e6) == x["kernel_sum_ns"]           # not the warm-up's
        assert p["kernel_sum_ms"]["max"] == p["kernel_sum_ms"]["min"]
        assert math.isclose(p["fills_ms"], x["fills_ns"] / 1e6) and p["off_lock"] is None and p["lock_mhz"] == 1710
    # a point whose marker is missing has no calls of its own; the point before it then holds two points' calls
    d2, _x = _timer_pass(tmp_path / "b", "decode", max_tokens=64, repeats=4, drop_marker=(3,))
    bad = FO.timer_pass_points("ernie-4.5-21b-a3b", d2)
    assert len(bad["points"]) <= 10 and any(s.startswith("f40:") for s in bad["problems"])
    # the sink marker keeps the next point's warm-up out of a point's segment (before any grid filter): with it
    # every segment holds exactly the 4 timed calls; without it the segment also holds the next warm-up call
    import nsys_extract as NX
    seg = NX.segment_marked(NX.read_sqlite(d / "timer.sqlite")["events"])
    assert sorted(k for k in seg if k != FO.SINK_INDEX) == list(range(11))
    assert all(len(seg[i]) == 4 for i in range(11))
    d3, _x = _timer_pass(tmp_path / "c", "decode", max_tokens=64, repeats=4, drop_sink=tuple(range(1, 11)))
    seg3 = NX.segment_marked(NX.read_sqlite(d3 / "timer.sqlite")["events"])
    assert [len(seg3[i]) for i in range(11)] == [5] * 10 + [4]


def test_off_lock_points_are_flagged_and_counted_not_dropped(tmp_path):
    d, _x = _timer_pass(tmp_path, "decode", max_tokens=32, readings={2: [1710, 1665], 4: [None, 1710]})
    doc = FO.build_points("ernie-4.5-21b-a3b", [d], 32)
    flagged = {p["label"]: p["off_lock"] for p in doc["points"] if p["off_lock"]}
    assert doc["off_lock_points"] == 2 and set(flagged) == {"f16", "f4"} and "1665 MHz under" in flagged["f16"]
    assert len(doc["points"]) == len(FO.token_grid(32)) and doc["complete"]
    assert all(p["sm_mhz"] == [1710, 1710] for p in doc["points"] if not p["off_lock"])


def test_the_full_grid_is_measured_and_every_point_is_in_the_csv(tmp_path, fake_frontier, capsys):
    """Grid completeness: two passes, 19 + 368 points, all 387 of get_num_tokens_to_profile(16384) in points.json
    and in moe_kernel_only.csv; fpoints refuses a missing pass (no budgeted subset) unless --allow-partial."""
    import pandas as pd
    dd, _x = _timer_pass(tmp_path, "decode", repeats=2)
    dp, _x = _timer_pass(tmp_path, "prefill", repeats=2)
    out = tmp_path / "points.json"
    assert FO.main(["fpoints", "--model", "ernie-4.5-21b-a3b", "--timer", str(dd), "--timer", str(dp),
                    "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["complete"] and doc["grid_points"] == doc["measured_points"] == 387 and doc["missing_points"] == []
    assert [p["measured"] for p in doc["passes"]] == [19, 368] and doc["problems"] == []
    assert FO.main(["_csv", "--model", "ernie-4.5-21b-a3b", "--points", str(out), "--out", str(tmp_path / "prof")]) == 0
    df = pd.read_csv(tmp_path / "prof" / FO.csv_relpath("ernie-4.5-21b-a3b"))
    assert len(df) == 387 and list(df["num_tokens"]) == FO.token_grid(16384)
    assert df[FO.TIME_COL].notna().all() and (df["time_stats.moe_grouped_gemm.count"] == 2).all()
    capsys.readouterr()
    assert FO.main(["fpoints", "--model", "ernie-4.5-21b-a3b", "--timer", str(dd), "--timer", str(tmp_path / "nope"),
                    "--out", str(out)]) == 3
    o = capsys.readouterr().out
    assert "FAIL frontier points: 19 rows (balanced, 1 per point), 19 of 387 grid points" in o and "must cover its full grid" in o
    doc = json.loads(out.read_text())
    assert not doc["complete"] and len(doc["missing_points"]) == 368 and "did not run" in doc["problems"][0]
    assert FO.main(["fpoints", "--model", "ernie-4.5-21b-a3b", "--timer", str(dd), "--timer", str(tmp_path / "nope"),
                    "--out", str(out), "--allow-partial"]) == 0


def test_points_carry_vllms_geometry(tmp_path):
    d, _x = _timer_pass(tmp_path, "decode", max_tokens=4096, repeats=1)
    dp, _x = _timer_pass(tmp_path, "prefill", max_tokens=1024, repeats=1)
    doc = FO.build_points("ernie-4.5-21b-a3b", [d, dp], 4096)
    assert not doc["complete"] and doc["grid_points"] == len(FO.token_grid(4096))
    r = doc["block_size_m_ranges"]
    assert r[0][0] == 1 and r[-1][1] == 4096 and all(a[1] + 1 == b[0] for a, b in zip(r, r[1:]))
    for p in doc["points"]:
        T = p["num_tokens"]
        bm = next(x[2] for x in r if x[0] <= T <= x[1])
        assert p["block_size_m"] == bm and int(PL.tile_bin(T).split("x")[0]) == bm
        if T > 2:                                                      # the aligned path: sum ceil(c/BM) BM
            assert p["num_tokens_post_padded"] == PL.post_padded(p["counts"], bm)
    with pytest.raises(PL.KitError):
        FO.build_points("ernie-4.5-21b-a3b", [tmp_path / "none"], 4096)
    bad = json.loads((d / "timer.json").read_text())
    bad["verdict"] = "FAIL"
    (d / "timer.json").write_text(json.dumps(bad))
    with pytest.raises(PL.KitError):
        FO.timer_pass_points("ernie-4.5-21b-a3b", d)


FAKE_LI = '''
class MoELoadImbalanceInput:
    def __init__(self, num_tokens, num_experts_per_device, hidden_dim, expert_hidden_dim, router_topk,
                 load_distribution="uniform", expert_token_counts=None, seed=None, tensor_parallel_size=1):
        self.c, self.ld, self.seed = expert_token_counts, load_distribution, seed
        self.k = (num_experts_per_device, hidden_dim, expert_hidden_dim, router_topk)

    def to_features_dict(self):
        f = {n: float(i) for i, n in enumerate(FEATURES)}
        f.update(total_routed_tokens=sum(self.c), num_experts_per_device=self.k[0], hidden_dim=self.k[1],
                 expert_hidden_dim=self.k[2], router_topk=self.k[3])
        return {**f, "load_distribution": self.ld, "seed": self.seed}


def block_size_m_for_tokens(ranges, T):
    return [r[2] for r in ranges if r[0] <= T <= r[1]][0]
'''


@pytest.fixture
def fake_frontier(tmp_path, monkeypatch):
    lib = tmp_path / "fakelib"
    (lib / "frontier" / "training").mkdir(parents=True)
    (lib / "frontier" / "__init__.py").write_text("")
    (lib / "frontier" / "moe_load_imbalance.py").write_text(f"FEATURES = {list(FRONTIER_FEATURES)!r}\n" + FAKE_LI)
    (lib / "frontier" / "training" / "__init__.py").write_text("")
    (lib / "frontier" / "training" / "moe_trainer.py").write_text(
        "class MoETrainer:\n    def _get_model_names(self):\n"
        "        return ['moe_gating_linear', 'moe_gating_routing_topk', 'moe_shuffling', 'moe_grouped_gemm']\n")
    (lib / "frontier" / "training" / "cli.py").write_text(
        "import sys, json\nfrom frontier.training.moe_trainer import MoETrainer\nSEEN = []\n"
        "def main():\n    SEEN.append((list(sys.argv), MoETrainer()._get_model_names()))\n    return 0\n")
    monkeypatch.syspath_prepend(str(lib))
    for m in [m for m in sys.modules if m == "frontier" or m.startswith("frontier.")]:
        monkeypatch.delitem(sys.modules, m)
    yield lib
    for m in [m for m in sys.modules if m == "frontier" or m.startswith("frontier.")]:
        sys.modules.pop(m, None)


def test_the_csv_is_frontiers_moe_schema_with_the_untimed_operators_empty(tmp_path, fake_frontier):
    import pandas as pd
    dd, _x = _timer_pass(tmp_path / "pts", "decode", max_tokens=16, repeats=2)
    dp, _x = _timer_pass(tmp_path / "pts", "prefill", max_tokens=144, repeats=2)
    doc = FO.build_points("ernie-4.5-21b-a3b", [dd, dp], 2048)
    (tmp_path / "points.json").write_text(json.dumps(doc))
    a = types.SimpleNamespace(points=str(tmp_path / "points.json"), model="ernie-4.5-21b-a3b", out=str(tmp_path / "prof"))
    with pytest.raises(PL.KitError, match="needs the full grid"):            # 7 of 2048's grid: not the primary arm
        FO._csv(a)
    a.allow_partial = True
    assert FO._csv(a) == 0
    df = pd.read_csv(tmp_path / "prof" / FO.csv_relpath("ernie-4.5-21b-a3b"))
    assert list(df["num_tokens"]) == [144, 136, 16, 8, 4, 2, 1]           # Frontier's order (descending)
    for op in FO.MOE_OPERATORS:                                              # the trainer's required targets
        assert f"time_stats.{op}.median" in df.columns
    for op in ("moe_gating_linear", "moe_gating_routing_topk", "moe_shuffling"):
        assert df[f"time_stats.{op}.median"].isna().all()                  # not measured: empty, never made up
    assert df[FO.TIME_COL].notna().all() and (df[FO.TIME_COL] > 0).all()
    for col in FRONTIER_FEATURES + ("load_distribution", "seed", "num_experts", "router_topk", "hidden_dim",
                                    "expert_hidden_dim", "num_tensor_parallel_workers", "expert_parallel_size",
                                    "time_stats.moe_grouped_gemm.num_tokens_post_padded",
                                    "time_stats.moe_grouped_gemm.block_size_m",
                                    "time_stats.moe_grouped_gemm.block_size_m_ranges", "routing_runtime_path"):
        assert col in df.columns, col
    assert set(df["measurement_type"]) == {"KERNEL_ONLY"} and set(df["profiling_precision"]) == {"BF16"}
    assert set(df["load_distribution"]) == {"balanced"} and set(df["routing_runtime_path"]) == {"uniform_topk"}
    assert (df["num_tensor_parallel_workers"] == 1).all() and (df["router_topk"] == 6).all()
    rng = json.loads(df["time_stats.moe_grouped_gemm.block_size_m_ranges"].iloc[0])
    assert rng[-1][1] == 2048


def test_the_trainer_is_frontiers_cli_with_one_shim(fake_frontier, tmp_path):
    argv = FO.cli_argv("ernie-4.5-21b-a3b", "/d/x.csv", "/m")
    assert argv[:2] == ["frontier.training.cli", "moe"]
    for k, v in (("--measurement_type", "KERNEL_ONLY"), ("--num_experts", "64"), ("--router_topk", "6"),
                 ("--hidden_dim", "2560"), ("--expert_hidden_dim", "1536"), ("--moe_tensor_parallel_size", "1"),
                 ("--expert_parallel_size", "1"), ("--device", "gh200")):
        assert argv[argv.index(k) + 1] == v, k
    saved = list(sys.argv)
    assert FO.run_frontier_cli("ernie-4.5-21b-a3b", tmp_path / "x.csv", tmp_path / "m") == 0
    import frontier.training.cli as C
    seen_argv, names = C.SEEN[-1]
    assert names == ["moe_grouped_gemm"]                                    # the shim: the timed operator only
    assert seen_argv[seen_argv.index("--measurement_type") + 1] == "KERNEL_ONLY" and sys.argv == saved


def test_holdout_split_is_every_fifth_point():
    assert FO.holdout_split([8, 1, 2, 4, 16, 24, 32, 40, 48, 56, 4]) == {4, 40}


def test_post_padded_matches_vllm_alignment():
    assert PL.post_padded([0, 3, 16, 17], 16) == 16 + 16 + 32


def test_dry_run_prints_commands_and_needs_no_frontier(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("FR_PY_PRIMARY", raising=False)
    assert FO.main(["train", "--model", "olmoe-1b-7b", "--src", str(tmp_path / "fr"), "--csv", "x.csv", "--out",
                    str(tmp_path / "m"), "--dry-run"]) == 0
    assert "_train" in capsys.readouterr().out and not (tmp_path / "m").exists()
    assert FO.main(["predict", "--model", "olmoe-1b-7b", "--src", str(tmp_path / "fr"), "--models", "cli=m",
                    "--calls", "A1710=/r", "--out", str(tmp_path / "p"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "_predict" in out and "--models cli=m" in out
    assert FO.main(["train", "--model", "olmoe-1b-7b", "--src", str(tmp_path / "fr"), "--csv", "x.csv", "--out",
                    str(tmp_path / "t")]) == 3                              # no Frontier venv: refused
    assert FO.main(["grid", "--max-tokens", "256"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["points"] == 35


FAKE_EST = '''
class Est:
    feature_names_in_ = ["total_routed_tokens", "max_load_ratio"]
    n_jobs = None

    def predict(self, X):
        import json, os
        with open(os.environ["FAKE_EST_LOG"], "a") as fh:
            fh.write(json.dumps(len(X)) + "\\n")
        return [0.001 * r["total_routed_tokens"] for _i, r in X.iterrows()]
'''


def test_predict_batches_every_distinct_call_into_one_dataframe_per_model(tmp_path):
    """Frontier's estimator is called ONCE per model, on one DataFrame of the distinct (T, counts) keys (on the VM,
    owner decision 2026-11-19), the output is compact (npy beside index.json) and a scikit-learn other than the
    trainer's refuses."""
    import pickle
    import subprocess
    lib = tmp_path / "lib"
    (lib / "frontier").mkdir(parents=True)
    (lib / "frontier" / "__init__.py").write_text("")
    (lib / "frontier" / "moe_load_imbalance.py").write_text(f"FEATURES = {list(FRONTIER_FEATURES)!r}\n" + FAKE_LI)
    (lib / "sklearn").mkdir()
    (lib / "sklearn" / "__init__.py").write_text("__version__ = '1.5.2'\n")
    (lib / "fake_est.py").write_text(FAKE_EST)
    sys.path.insert(0, str(lib))
    try:
        import fake_est
        models = tmp_path / "models"
        models.mkdir()
        with open(models / "moe_grouped_gemm_x.pkl", "wb") as fh:
            pickle.dump(fake_est.Est(), fh)
    finally:
        sys.path.remove(str(lib))
        sys.modules.pop("fake_est", None)
    run = tmp_path / "run"
    run.mkdir()
    rows = [{"g": i, "T_pad": T, "status": "ok"} for i, T in enumerate([8, 8, 600, 600, 8])]
    counts = np.zeros((5, 64), dtype=np.int32)
    counts[:, :8] = 8
    counts[2:4, :8] = 600
    (run / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.save(run / "counts.npy", counts)
    log = tmp_path / "predict_calls.log"
    env = dict(__import__("os").environ, PYTHONPATH=f"{lib}:{REPO}:{REPO / 'scripts'}", FAKE_EST_LOG=str(log))
    out = tmp_path / "pred"
    cmd = [sys.executable, str(REPO / "scripts/prod/frontier_op.py"), "_predict", "--model", "olmoe-1b-7b", "--models",
           f"cli={models}", "--models", f"simpath={models}", "--calls", f"A1710={run}", "--out", str(out)]
    (models / "versions.json").write_text(json.dumps({"sklearn": "1.4.0"}))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "scikit-learn 1.5.2 here, 1.4.0 at training" in r.stdout + r.stderr
    (models / "versions.json").write_text(json.dumps({"sklearn": "1.5.2"}))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert log.read_text().split() == ["2", "2"]                  # one predict call per model, two distinct keys
    got = FO.load_predictions(out)
    assert set(got) == {("A1710", g, arm) for g in range(5) for arm in ("cli", "simpath")}
    assert math.isclose(got[("A1710", 0, "cli")], 0.064, rel_tol=1e-6) and got[("A1710", 0, "cli")] == got[("A1710", 4, "cli")]
    assert "one batch" in r.stdout
    # the early probe: predictions on its own measured points (points.json, index = g)
    pts = tmp_path / "points.json"
    pts.write_text(json.dumps({"points": [{"num_tokens": 8, "counts": [8] * 8 + [0] * 56},
                                          {"num_tokens": 600, "counts": [600] * 8 + [0] * 56}]}))
    r = subprocess.run(cmd[:7] + ["--points", str(pts), "--out", str(tmp_path / "pp")], capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    got = FO.load_predictions(tmp_path / "pp")
    assert set(got) == {("points", 0, "cli"), ("points", 1, "cli")} and got[("points", 1, "cli")] > 0


def test_uniform_passes_need_all_three_samples_of_every_point_and_the_trace_may_live_elsewhere(tmp_path):
    """Grid completeness under the default routing: every point with ALL its samples (seeds 0..2); one missing
    sample makes the point missing (missing_samples says which). The sqlite may sit outside the results tree
    (main.sh: $MOE_HOME/traces/<run>/, review m6): fpoints --timer-sqlite names it."""
    tr = tmp_path / "traces"
    dd, _x = _timer_pass(tmp_path / "r", "decode", max_tokens=32, repeats=2, routing="uniform", sqlite_dir=tr)
    out = tmp_path / "points.json"
    assert FO.main(["fpoints", "--model", "ernie-4.5-21b-a3b", "--timer", str(dd), "--max-tokens", "32",
                    "--out", str(out)]) == 3                               # no sqlite beside timer.json
    assert FO.main(["fpoints", "--model", "ernie-4.5-21b-a3b", "--timer", str(dd), "--timer-sqlite",
                    str(tr / "timer-decode" / "timer.sqlite"), "--max-tokens", "32", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["routing"] == "uniform" and doc["samples_per_point"] == 3 and doc["complete"]
    assert len(doc["points"]) == 3 * len(FO.token_grid(32)) and {p["seed"] for p in doc["points"]} == {0, 1, 2}
    # one sample's marker lost: its point is missing (the point before holds two samples' calls: a problem)
    dd2, _x = _timer_pass(tmp_path / "r2", "decode", max_tokens=32, repeats=2, routing="uniform", drop_marker=(4,))
    doc2 = FO.build_points("ernie-4.5-21b-a3b", [dd2], 32)
    assert not doc2["complete"] and 24 in doc2["missing_points"] and 1 in doc2["missing_samples"]["24"]


def _points_doc(tmp_path, *, complete: bool):
    """Frontier's grid to 144: the decode pass (19 points), and the prefill pass (136, 144) when complete."""
    dirs = [_timer_pass(tmp_path / "pts", "decode", max_tokens=144, repeats=2)[0]]
    if complete:
        dirs.append(_timer_pass(tmp_path / "pts", "prefill", max_tokens=144, repeats=2)[0])
    doc = FO.build_points("ernie-4.5-21b-a3b", dirs, 144)
    f = tmp_path / "points.json"
    f.write_text(json.dumps(doc))
    return f, doc


def test_an_incomplete_grid_trains_only_the_labelled_partial_arm(tmp_path, fake_frontier, capsys):
    """Review 2026-11-19 M3: missing points still give points.json and a CSV, but only as the separately labelled
    PARTIAL arm (csv --partial, csv_meta.json lists the missing points; train --only-if-partial trains it); the
    primary CSV refuses an incomplete grid; on a complete grid the partial arm is not needed and trains nothing."""
    import pandas as pd
    f, doc = _points_doc(tmp_path / "inc", complete=False)
    assert not doc["complete"] and doc["missing_points"] == [136, 144]
    ns = dict(model="ernie-4.5-21b-a3b", points=str(f))
    with pytest.raises(PL.KitError, match="needs the full grid"):
        FO._csv(types.SimpleNamespace(**ns, out=str(tmp_path / "prim")))
    assert not (tmp_path / "prim" / FO.csv_relpath("ernie-4.5-21b-a3b")).exists()
    assert FO._csv(types.SimpleNamespace(**ns, out=str(tmp_path / "part"), partial=True)) == 0
    meta = json.loads((tmp_path / "part" / FO.CSV_META).read_text())
    assert meta["arm_kind"] == "partial" and meta["partial_needed"] and meta["missing_points"] == [136, 144]
    assert meta["truth"] == "kernel_sum" and meta["rows"] == 19 and FO.partial_needed(tmp_path / "part" / FO.CSV_META)
    assert len(pd.read_csv(tmp_path / "part" / FO.csv_relpath("ernie-4.5-21b-a3b"))) == 19
    # complete grid: no partial arm; train --only-if-partial is a PASS that trains nothing (no Frontier needed)
    fc, _d = _points_doc(tmp_path / "com", complete=True)
    assert FO._csv(types.SimpleNamespace(model="ernie-4.5-21b-a3b", points=str(fc), out=str(tmp_path / "p2"),
                                         partial=True)) == 0
    assert not (tmp_path / "p2" / FO.csv_relpath("ernie-4.5-21b-a3b")).exists()
    assert not FO.partial_needed(tmp_path / "p2" / FO.CSV_META) and not FO.partial_needed(tmp_path / "nope.json")
    capsys.readouterr()
    assert FO.main(["train", "--model", "ernie-4.5-21b-a3b", "--src", str(tmp_path / "fr"), "--csv", "x.csv",
                    "--out", str(tmp_path / "m"), "--arm-label", "cli-partial", "--only-if-partial",
                    str(tmp_path / "p2" / FO.CSV_META)]) == 0
    assert "not needed" in capsys.readouterr().out and not (tmp_path / "m").exists()


def test_two_truths_kernel_sum_without_fills_and_frontiers_native_with_fills(tmp_path, fake_frontier):
    """Review 2026-11-19 m4: the kernel_sum CSV (the fills stripped: the scorer's serving truth) and the with_fills
    CSV (every kernel of the functional call: Frontier's record_function definition) differ by exactly each call's
    fills; csv_meta.json records which."""
    import pandas as pd
    f, _doc = _points_doc(tmp_path, complete=True)
    for truth in FO.TRUTHS:
        assert FO._csv(types.SimpleNamespace(model="ernie-4.5-21b-a3b", points=str(f), out=str(tmp_path / truth),
                                             truth=truth)) == 0
        assert json.loads((tmp_path / truth / FO.CSV_META).read_text())["truth"] == truth
    a = pd.read_csv(tmp_path / "kernel_sum" / FO.csv_relpath("ernie-4.5-21b-a3b"))
    b = pd.read_csv(tmp_path / "with_fills" / FO.csv_relpath("ernie-4.5-21b-a3b"))
    doc = json.loads(f.read_text())
    fills = {p["num_tokens"]: p["fills_ms"] for p in doc["points"]}
    assert all(x > 0 for x in fills.values())
    for (_i, ra), (_j, rb) in zip(a.iterrows(), b.iterrows(), strict=True):
        assert math.isclose(rb[FO.TIME_COL] - ra[FO.TIME_COL], fills[int(ra["num_tokens"])], rel_tol=1e-9)
    with pytest.raises(PL.KitError):
        FO.point_time({"label": "x", "kernel_sum_ms": {"median": 1.0}}, "with_fills")


def test_the_probe_trains_with_a_narrow_grid_search_through_the_clis_own_arguments():
    argv = FO.cli_argv("ernie-4.5-21b-a3b", "/d/x.csv", "/m", narrow=True)
    assert argv[:len(FO.cli_argv("ernie-4.5-21b-a3b", "/d/x.csv", "/m"))] == FO.cli_argv("ernie-4.5-21b-a3b",
                                                                                        "/d/x.csv", "/m")
    for k, v in (("--k_fold_cv_splits", "3"), ("--num_estimators", "50"), ("--max_depth", "8"),
                 ("--min_samples_split", "2")):
        assert argv[argv.index(k) + 1] == v, k
    assert "--num_estimators" not in FO.cli_argv("ernie-4.5-21b-a3b", "/d/x.csv", "/m")
    assert [T for T in FO.PROBE_POINTS["ernie-4.5-21b-a3b"] if T <= 128] == [64]          # one decode, two prefill
    assert all(T in FO.token_grid(16384) for T in FO.PROBE_POINTS["ernie-4.5-21b-a3b"])
    assert all(T in FO.token_grid(256) for T in FO.PROBE_POINTS["olmoe-1b-7b"])
