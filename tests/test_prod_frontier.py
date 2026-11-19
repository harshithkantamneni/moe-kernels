"""scripts/prod/frontier_op.py off the GPU (Frontier itself is never executed on the laptop): the model config
written in Frontier's HF dialect, the documented profiler and trainer command lines, the API-branch reader, the
hold-out split, and the dry run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts", REPO / "scripts" / "prod"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import frontier_op as FO  # noqa: E402
import prodlib as PL  # noqa: E402


def test_the_config_carries_the_served_models_moe_operator(tmp_path):
    for key in ("ernie-4.5-21b-a3b", "olmoe-1b-7b"):
        p = FO.write_config(key, tmp_path)
        doc = json.loads(p.read_text())
        cfg = PL.MODELS[key].cfg
        assert (doc["hidden_size"], doc["moe_intermediate_size"], doc["num_experts"], doc["num_experts_per_tok"]) == \
            (cfg.hidden_size, cfg.intermediate_size, cfg.num_experts, cfg.top_k)
        assert doc["hidden_act"] == "silu" and doc["torch_dtype"] == "bfloat16"
        # Frontier's loader's required keys (frontier/config/model_config.py _create_from_hf_json)
        for k in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_size",
                  "max_position_embeddings", "vocab_size", "hidden_act"):
            assert k in doc
        assert p.parent == tmp_path / "data" / "config" / "models"


def test_the_profiler_runs_through_its_documented_wrapper_with_imbalance_on():
    cmd = FO.profile_cmd("/fr", "ernie-4.5-21b-a3b", ["uniform"], "/out")
    assert cmd[:2] == ["bash", "/fr/examples/profiling/profile_moe.sh"]
    i = cmd.index("--")
    mine, passed = cmd[:i], cmd[i + 1:]
    assert mine[mine.index("--profile-method") + 1] == "record_function"        # KERNEL_ONLY rows
    assert mine[mine.index("--max-tokens") + 1] == "16384"
    assert mine[mine.index("--device") + 1] == "gh200" and mine[mine.index("--model") + 1] == "ernie45-21b-a3b-moe"
    # the wrapper hard-codes --disable_load_imbalance; argparse's last flag wins
    assert passed[0] == "--enable_load_imbalance" and passed[passed.index("--load_distributions") + 1] == "uniform"
    all4 = FO.profile_cmd("/fr", "olmoe-1b-7b", list(FO.DISTS), "/out")
    assert all4[all4.index("--load_distributions") + 1:all4.index("--yes")] == list(FO.DISTS)
    try:
        FO.profile_cmd("/fr", "olmoe-1b-7b", ["zipf"], "/out")
        raise AssertionError("an unknown distribution must be refused")
    except PL.KitError:
        pass


def test_the_trainer_is_the_documented_cli():
    cmd = FO.train_cmd("/py", "ernie-4.5-21b-a3b", "/d/moe_kernel_only.csv", "/m")
    assert cmd[1:4] == ["-m", "frontier.training.cli", "moe"]
    for k, v in (("--model_name", "ernie45-21b-a3b-moe"), ("--device", "gh200"), ("--moe_tensor_parallel_size", "1"),
                 ("--expert_parallel_size", "1")):
        assert cmd[cmd.index(k) + 1] == v


def test_the_api_branch_is_read_off_frontiers_own_print():
    log = "x\nvLLM 0.27.1 loaded successfully (API: functional_fused_experts)\nFP8 ...\n"
    assert FO.parse_api(log) == {"vllm": "0.27.1", "api": "functional_fused_experts"}
    assert FO.parse_api("vLLM 0.10.2 loaded successfully (API: 0.10.x)") == {"vllm": "0.10.2", "api": "0.10.x"}
    assert FO.parse_api("nothing") == {"vllm": None, "api": None}


def test_holdout_split_is_every_fifth_point():
    held = FO.holdout_split([8, 1, 2, 4, 16, 24, 32, 40, 48, 56, 4])
    assert held == {4, 40}


def test_post_padded_matches_vllm_alignment():
    assert PL.post_padded([0, 3, 16, 17], 16) == 16 + 16 + 32


def test_dry_run_prints_commands_and_needs_no_frontier(tmp_path, capsys):
    assert FO.main(["profile", "--model", "olmoe-1b-7b", "--src", str(tmp_path / "fr"), "--out",
                    str(tmp_path / "o"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "profile_moe.sh" in out and "--enable_load_imbalance" in out and not (tmp_path / "o").exists()
    assert FO.main(["predict", "--model", "olmoe-1b-7b", "--src", str(tmp_path / "fr"), "--models", "m",
                    "--calls", "A1710=/r", "--out", str(tmp_path / "p.jsonl"), "--dry-run"]) == 0
    assert "_predict" in capsys.readouterr().out
    assert FO.main(["train", "--model", "olmoe-1b-7b", "--csv", "x.csv", "--out", str(tmp_path / "t")]) == 3


FAKE_FRONTIER = '''
class MoELoadImbalanceInput:
    def __init__(self, num_tokens, num_experts_per_device, hidden_dim, expert_hidden_dim, router_topk,
                 expert_token_counts):
        self.n, self.c = num_tokens, expert_token_counts

    def to_features_dict(self):
        return {"num_tokens": self.n, "active": sum(1 for x in self.c if x), "max_c": max(self.c),
                "load_distribution": "real", "seed": 0}


def block_size_m_for_tokens(ranges, T):
    return 64
'''
FAKE_EST = '''
CALLS = []


class Est:
    feature_names_in_ = ["num_tokens", "active", "max_c"]

    def predict(self, X):
        import json, os
        with open(os.environ["FAKE_EST_LOG"], "a") as fh:
            fh.write(json.dumps(len(X)) + "\\n")
        return [0.001 * r["num_tokens"] + r["active"] for _i, r in X.iterrows()]
'''


def test_predict_batches_every_distinct_call_into_one_dataframe_and_checks_versions(tmp_path):
    """MAJOR 7: Frontier's estimator is called ONCE, on one DataFrame of the distinct (T, counts) keys (run on the
    laptop after the rental), and a scikit-learn other than the trainer's refuses."""
    import pickle
    import subprocess

    import numpy as np
    lib = tmp_path / "lib"
    (lib / "frontier").mkdir(parents=True)
    (lib / "frontier" / "__init__.py").write_text("")
    (lib / "frontier" / "moe_load_imbalance.py").write_text(FAKE_FRONTIER)
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
    cmd = [sys.executable, str(REPO / "scripts/prod/frontier_op.py"), "_predict", "--model", "olmoe-1b-7b", "--models",
           str(models), "--calls", f"A1710={run}", "--arm-label", "primary-uniform", "--out", str(tmp_path / "p.jsonl")]
    (models / "versions.json").write_text(json.dumps({"sklearn": "1.4.0"}))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "scikit-learn 1.5.2 here, 1.4.0 at training" in r.stdout
    (models / "versions.json").write_text(json.dumps({"sklearn": "1.5.2"}))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert log.read_text().split() == ["2"]                   # one predict call, two distinct keys
    preds = [json.loads(x) for x in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert [p["g"] for p in preds] == [0, 1, 2, 3, 4] and preds[0]["pred_ms"] == preds[4]["pred_ms"]
    assert "one batch" in r.stdout
