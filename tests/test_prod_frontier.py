"""scripts/prod/frontier_op.py off the GPU (Frontier itself is never executed on the laptop; owner decision 2026-11-19:
its predictor, our profiles): Frontier's token grid and round-robin routing, the pages and their order, the samples the
replay times, the points and their geometry, the CSV in Frontier's MoE schema, the trainer's command line and its one
shim, the batch predictions, the hold-out split and the dry run."""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "scripts" / "prod"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import frontier_op as FO  # noqa: E402
import prodlib as PL  # noqa: E402
import replay as RP  # noqa: E402

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


def test_routing_is_frontiers_round_robin_uniform_topk():
    ids = FO.round_robin_ids(5, 64, 6)
    assert ids.tolist()[0] == [0, 1, 2, 3, 4, 5] and ids.tolist()[1] == [6, 7, 8, 9, 10, 11]
    assert all(int(ids[t, j]) == (t * 6 + j) % 64 for t in range(5) for j in range(6))
    big = FO.round_robin_ids(1000, 64, 6)
    c = PL.counts_from_ids(big, 64)
    assert max(c) - min(c) <= 1 and sum(c) == 6000                         # balanced: Frontier's "balanced"
    assert all(len(set(r)) == 6 for r in big.tolist())


def test_pages_partition_the_grid_and_their_order_is_bit_reversed():
    assert FO.page_order(16) == [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15] and FO.page_order(1) == [0]
    with pytest.raises(PL.KitError):
        FO.page_order(12)
    dec = FO.page_points("decode", 0, 16, 16384)
    assert dec == [1, 2, 4] + list(range(8, 129, 8)) and len(dec) == 19
    pre = [FO.page_points("prefill", j, 16, 16384) for j in range(16)]
    flat = sorted(t for p in pre for t in p)
    assert flat == FO.branch_points("prefill", 16384) and len(flat) + len(dec) == 387
    # the first four pages in order cover every 4th prefill point
    first4 = sorted(t for j in FO.page_order(16)[:4] for t in pre[j])
    assert first4 == FO.branch_points("prefill", 16384)[::4]
    with pytest.raises(PL.KitError):
        FO.page_points("decode", 1, 16, 16384)


def test_fsample_writes_a_sample_the_replay_loads(tmp_path):
    pairs = FO.fsample_cells("ernie-4.5-21b-a3b", FO.page_points("prefill", 3, 16, 16384), "balanced")
    p = RP.write_synthetic_sample(PL.MODELS["ernie-4.5-21b-a3b"], pairs, tmp_path, extra={"frontier": {"page": 3}})
    page, ids = RP.load_sample(p, PL.MODELS["ernie-4.5-21b-a3b"], "prefill")       # V-BIN on load passes
    assert [c.label for c in page.cells] == [f"f{T}" for T in FO.page_points("prefill", 3, 16, 16384)]
    doc = json.loads(p.read_text())
    assert all(c["synthetic"] and c["g"] is None and c["frontier"]["load_distribution"] == "balanced"
               for c in doc["cells"])
    assert ids["f160"].shape == (160, 6)
    with pytest.raises(PL.KitError):
        FO.fsample_cells("ernie-4.5-21b-a3b", [8], "zipf")


def _fake_page(root, name, branch, Ts, model="ernie-4.5-21b-a3b", measured=True):
    d = root / name
    pairs = FO.fsample_cells(model, Ts, "balanced")
    RP.write_synthetic_sample(PL.MODELS[model], pairs, d)
    if measured:
        cells = [{"label": c["label"], "T": c["T"], "n": 24, "kernel_count": 6, "fills_n": 9, "fills_ms": 0.01,
                  "source": "graph", "kernel_sum_ms": {"median": 0.01 * c["T"] ** 0.5, "mean": 0.01, "min": 0.009,
                                                       "max": 0.02, "std": 0.001}} for c, _i in pairs]
        (d / "cells.json").write_text(json.dumps({"cells": cells}))
    return d


def test_points_carry_vllms_geometry_and_skip_unmeasured_pages(tmp_path):
    _fake_page(tmp_path, "decode-00", "decode", [1, 2, 8, 128])
    _fake_page(tmp_path, "prefill-00", "prefill", [136, 2048])
    _fake_page(tmp_path, "prefill-08", "prefill", [264], measured=False)
    doc = FO.build_points("ernie-4.5-21b-a3b", tmp_path, 4096)
    assert doc["pages"] == ["decode-00", "prefill-00"] and doc["unmeasured_pages"] == ["prefill-08"]
    assert doc["measured_points"] == 6 and doc["grid_points"] == len(FO.token_grid(4096))
    r = doc["block_size_m_ranges"]
    assert r[0][0] == 1 and r[-1][1] == 4096 and all(a[1] + 1 == b[0] for a, b in zip(r, r[1:]))
    by = {p["num_tokens"]: p for p in doc["points"]}
    for T, p in by.items():
        bm = next(x[2] for x in r if x[0] <= T <= x[1])
        assert p["block_size_m"] == bm and int(PL.tile_bin(T).split("x")[0]) == bm
        if T > 2:                                                      # the aligned path: sum ceil(c/BM) BM
            assert p["num_tokens_post_padded"] == PL.post_padded(p["counts"], bm)
    with pytest.raises(PL.KitError):
        FO.build_points("ernie-4.5-21b-a3b", tmp_path / "prefill-08", 4096)


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
    _fake_page(tmp_path / "pts", "decode-00", "decode", [1, 8, 128])
    _fake_page(tmp_path / "pts", "prefill-00", "prefill", [256, 1024])
    doc = FO.build_points("ernie-4.5-21b-a3b", tmp_path / "pts", 2048)
    (tmp_path / "points.json").write_text(json.dumps(doc))
    a = types.SimpleNamespace(points=str(tmp_path / "points.json"), model="ernie-4.5-21b-a3b", out=str(tmp_path / "prof"))
    assert FO._csv(a) == 0
    df = pd.read_csv(tmp_path / "prof" / FO.csv_relpath("ernie-4.5-21b-a3b"))
    assert list(df["num_tokens"]) == [1024, 256, 128, 8, 1]               # Frontier's order (descending)
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
