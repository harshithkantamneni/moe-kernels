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
