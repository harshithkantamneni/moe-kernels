"""Frontier (github.com/NetX-lab/Frontier at db5a85b, MIT) driven UNMODIFIED, as its docs drive it (KIT-SPEC 6).

    FR=$MOE_HOME/frontier; . $FR/frontier.env                         # frontier_setup.sh wrote it
    python scripts/prod/frontier_op.py write-config --model ernie-4.5-21b-a3b --src $FRONTIER_SRC
    python scripts/prod/frontier_op.py profile --arm primary --model ernie-4.5-21b-a3b --dists uniform --out OUT
    python scripts/prod/frontier_op.py train   --arm primary --model ... --csv OUT/.../moe_kernel_only.csv --out MODELS
    python scripts/prod/frontier_op.py predict --arm primary --model ... --models MODELS --csv CSV \\
        --calls A1710=RUN_A1710 ... --sample SAMPLE_DIR/sample.json --out PRED.jsonl
    python scripts/prod/frontier_op.py holdout --arm fidelity --model ... --csv CSV --out DIR
    python scripts/prod/frontier_op.py profile ... --dry-run            # every command, nothing run

HOW IT IS DRIVEN (docs/profiling/README.md "MoE operators"; examples/profiling/profile_moe.sh; frontier/training/
MOE-README.md "Method 3: Command Line"):
  profile  bash examples/profiling/profile_moe.sh --model M --device gh200 --tp-sizes 1 --ep-sizes 1
           --profile-method record_function --max-tokens 16384 --output-root OUT -- --enable_load_imbalance
           --load_distributions <dists> --yes
           (the wrapper's documented pass-through after `--`; its own --disable_load_imbalance is overridden by the
           later --enable_load_imbalance, argparse's last-wins. record_function = KERNEL_ONLY rows, the
           kernel-sum truth's measurement type. --max-tokens 16384: the RF cannot extrapolate past the profile,
           CRITIQUE M1.) The API branch Frontier's moe_vllm_kernel.py selects is read off its own print line
           "vLLM <v> loaded successfully (API: <branch>)" and recorded: vLLM 0.27.1 has no low-level
           `invoke_fused_moe_kernel`, so the functional `fused_experts` branch is expected (align inside the
           timed step, like the in-serving call); 0.10.2 takes the low-level branch.
  train    python -m frontier.training.cli moe --dataset_path CSV --output_dir DIR --model_name M --device gh200
           --moe_tensor_parallel_size 1 --expert_parallel_size 1   (its RF, its grid search, its features)
  predict  the trained moe_grouped_gemm estimator on per-call features built by Frontier's OWN
           MoELoadImbalanceInput(expert_token_counts=<the call's real counts>).to_features_dict() (14 features;
           num_tokens = the padded T). When the table records num_tokens_post_padded (the 0.10.x low-level branch),
           the simulator's predictor trains on (total_routed_tokens, num_tokens_post_padded)
           (execution_time_predictor/moe_dataset_training.py select_moe_operator_features); that model is the
           `simpath` predictor, trained by Frontier's MoETrainer with its feature selection swapped for the
           simulator's (THE ONE SHIM, in this file, disclosed), fed (6T, sum ceil(c_e/BM) BM) with BM from
           Frontier's own block_size_m_for_tokens over the table's ranges (CRITIQUE M1).
  holdout  Frontier's fidelity on its OWN held-out profiled points: num_tokens points with index % 5 == 2 held
           out, the CLI trainer on the rest, its estimator on the held-out rows (MAPE, p50, p95).

DISCLOSURES (README "Frontier"): Frontier's source is not edited (frontier_setup.sh checks `git diff` empty); one
UNTRACKED data file is added, data/config/models/<name>.json, describing the served model's MoE operator in the
HF keys Frontier reads (model_type qwen3_moe so its architecture registry resolves; attention fields unused by the
MoE profiler); the simpath trainer above; predictions on real routing from a model trained on synthetic routing
are labelled by load distribution.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prodlib as PL  # noqa: E402

FRONTIER_URL = "https://github.com/NetX-lab/Frontier"
FRONTIER_COMMIT = "db5a85b"
DEVICE = "gh200"
MAX_TOKENS = 16384
DISTS = ("uniform", "balanced", "skewed", "extremely_skewed")
API_RE = re.compile(r"vLLM (\S+) loaded successfully \(API: ([^)]+)\)")
HOLDOUT_MOD, HOLDOUT_REM = 5, 2

#: the served models in Frontier's HF-config dialect. Only hidden_size, moe_intermediate_size, num_experts,
#: num_experts_per_tok, hidden_act and dtype reach the MoE operator; the rest satisfy the loader's required keys.
FRONTIER_MODELS = {
    "ernie-4.5-21b-a3b": ("ernie45-21b-a3b-moe", {
        "hidden_size": 2560, "moe_intermediate_size": 1536, "num_experts": 64, "num_experts_per_tok": 6,
        "intermediate_size": 12288, "num_hidden_layers": 28, "num_attention_heads": 20, "num_key_value_heads": 4,
        "head_dim": 128, "vocab_size": 103424, "max_position_embeddings": 131072, "rms_norm_eps": 1e-05,
        "norm_topk_prob": True, "use_qk_norm": False, "tie_word_embeddings": True}),
    "olmoe-1b-7b": ("olmoe-1b-7b-moe", {
        "hidden_size": 2048, "moe_intermediate_size": 1024, "num_experts": 64, "num_experts_per_tok": 8,
        "intermediate_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16, "num_key_value_heads": 16,
        "head_dim": 128, "vocab_size": 50304, "max_position_embeddings": 4096, "rms_norm_eps": 1e-05,
        "norm_topk_prob": False, "use_qk_norm": True, "tie_word_embeddings": False}),
}


def frontier_config(model_key: str) -> tuple[str, dict]:
    name, extra = FRONTIER_MODELS[model_key]
    cfg = PL.MODELS[model_key].cfg
    assert extra["hidden_size"] == cfg.hidden_size and extra["moe_intermediate_size"] == cfg.intermediate_size
    assert extra["num_experts"] == cfg.num_experts and extra["num_experts_per_tok"] == cfg.top_k
    doc = {"_comment": f"moe-kernels production check: {model_key}'s routed MoE operator in Frontier's HF-config "
                       "keys (scripts/prod/frontier_op.py). Not a Frontier file; untracked.",
           **extra, "hidden_act": "silu", "dtype": "bfloat16", "torch_dtype": "bfloat16", "model_type": "qwen3_moe",
           "architectures": ["Qwen3MoeForCausalLM"], "decoder_sparse_step": 1, "mlp_only_layers": [],
           "attention_bias": False, "rope_theta": 500000.0, "rope_scaling": None}
    return name, doc


def write_config(model_key: str, src) -> Path:
    name, doc = frontier_config(model_key)
    p = Path(src) / "data" / "config" / "models" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n")
    return p


def profile_cmd(src, model_key: str, dists, out, max_tokens: int = MAX_TOKENS) -> list[str]:
    name, _ = frontier_config(model_key)
    bad = [d for d in dists if d not in DISTS]
    if bad:
        raise PL.KitError(f"load distributions {bad}: Frontier's choices are {DISTS}")
    return ["bash", str(Path(src) / "examples" / "profiling" / "profile_moe.sh"), "--model", name, "--device", DEVICE,
            "--tp-sizes", "1", "--ep-sizes", "1", "--profile-method", "record_function", "--max-tokens",
            str(max_tokens), "--output-root", str(Path(out).resolve()), "--", "--enable_load_imbalance",
            "--load_distributions", *dists, "--yes"]


def train_cmd(py, model_key: str, csv, out) -> list[str]:
    name, _ = frontier_config(model_key)
    return [str(py), "-m", "frontier.training.cli", "moe", "--dataset_path", str(Path(csv).resolve()),
            "--output_dir", str(Path(out).resolve()), "--model_name", name, "--device", DEVICE,
            "--moe_tensor_parallel_size", "1", "--expert_parallel_size", "1"]


def parse_api(log_text: str) -> dict:
    m = API_RE.findall(log_text)
    if not m:
        return {"vllm": None, "api": None}
    v, api = m[-1]
    return {"vllm": v, "api": api}


def expected_csv(out, model_key: str) -> Path:
    name, _ = frontier_config(model_key)
    return Path(out) / "compute" / DEVICE / name / "moe_kernel_only.csv"


def holdout_split(num_tokens_values) -> set:
    """The held-out num_tokens points: every 5th distinct value (index % 5 == 2), ascending order."""
    vals = sorted(set(int(v) for v in num_tokens_values))
    return {v for i, v in enumerate(vals) if i % HOLDOUT_MOD == HOLDOUT_REM}


def env_for(src) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(src)) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_logged(cmd, *, cwd, env, log: Path, dry: bool) -> int:
    print("  $ " + shlex.join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    with open(log, "w") as fh:
        p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode


# ------------------------------------------------------------------------------------------------------------
# inside Frontier's interpreter (imports frontier, sklearn): _simpath, _predict, _holdout
# ------------------------------------------------------------------------------------------------------------

def _load_estimator(models_dir, op: str = "moe_grouped_gemm"):
    import pickle
    cands = sorted(Path(models_dir).glob(f"{op}_*.pkl"))
    if len(cands) != 1:
        raise PL.KitError(f"{models_dir}: {len(cands)} {op}_*.pkl files (want 1)")
    # Trusted: the pickle is the estimator Frontier's own trainer wrote minutes earlier in this run's directory
    # (base_trainer.py _store_model_in_cache); nothing downloaded is unpickled.
    with open(cands[0], "rb") as fh:
        return pickle.load(fh), cands[0].name


def _features(est, model_key: str, T: int, counts, ranges=None) -> dict:
    from frontier.moe_load_imbalance import MoELoadImbalanceInput, block_size_m_for_tokens
    cfg = PL.MODELS[model_key].cfg
    cols = list(getattr(est, "feature_names_in_", []))
    if cols == ["total_routed_tokens", "num_tokens_post_padded"]:
        if ranges is None:
            raise PL.KitError("a padded-feature model needs the table's block_size_m ranges")
        bm = block_size_m_for_tokens(ranges, int(T))
        return {"total_routed_tokens": int(sum(counts)), "num_tokens_post_padded": PL.post_padded(counts, bm),
                "_bm": bm}
    li = MoELoadImbalanceInput(num_tokens=int(T), num_experts_per_device=cfg.num_experts,
                               hidden_dim=cfg.hidden_size, expert_hidden_dim=cfg.intermediate_size,
                               router_topk=cfg.top_k, expert_token_counts=[int(c) for c in counts])
    f = li.to_features_dict()
    f.pop("load_distribution", None)
    f.pop("seed", None)
    if cols == ["num_tokens"]:
        return {"num_tokens": int(T)}
    return f


def _ranges(csv):
    import pandas as pd
    from frontier.execution_time_predictor.moe_dataset_training import (
        NUM_TOKENS_POST_PADDED_COLUMN,
        block_size_m_ranges,
    )
    df = pd.read_csv(csv)
    if NUM_TOKENS_POST_PADDED_COLUMN not in df.columns:
        return None
    return block_size_m_ranges(df)


def _simpath(a) -> int:
    """Frontier's MoETrainer with its feature selection swapped for the simulator's (the shim)."""
    import pandas as pd
    from frontier.execution_time_predictor.moe_dataset_training import select_moe_operator_features
    from frontier.training.moe_trainer import create_moe_trainer_from_model_config
    name, _ = frontier_config(a.model)
    tr = create_moe_trainer_from_model_config(dataset_path=str(Path(a.csv).resolve()),
                                              output_dir=str(Path(a.out).resolve()), model_name=name, device=DEVICE)
    df = pd.read_csv(a.csv)
    _df, feats = select_moe_operator_features("moe_grouped_gemm", df)
    orig = tr._get_feature_cols
    tr._get_feature_cols = lambda m: list(feats) if m == "moe_grouped_gemm" else orig(m)
    if "num_tokens_post_padded" in feats:
        orig_df = tr._get_training_df_for_model

        def _tdf(df_, m, fc, tc):
            if m == "moe_grouped_gemm":
                df_, _f = select_moe_operator_features(m, df_)
            return orig_df(df_, m, fc, tc)
        tr._get_training_df_for_model = _tdf
    tr.train()
    print(json.dumps({"simpath_features": list(feats)}))
    return 0


def _predict(a) -> int:
    import numpy as np
    import pandas as pd
    est, fname = _load_estimator(a.models)
    ranges = _ranges(a.csv) if a.csv else None
    out = open(a.out, "w")
    n = 0
    jobs = []
    for spec in a.calls or []:
        label, _, d = spec.partition("=")
        rows = PL.read_jsonl(Path(d) / "calls.jsonl")
        counts = np.load(Path(d) / "counts.npy")
        jobs += [(label, r["g"], r["T_pad"], [int(x) for x in c]) for r, c in zip(rows, counts, strict=True)
                 if r.get("status") == "ok"]
    if a.sample:
        doc = json.loads(Path(a.sample).read_text())
        jobs += [("sample", c["g"], c["T"], c["counts"]) for c in doc["cells"]]
    cache: dict = {}
    for label, g, T, counts in jobs:
        key = (T, tuple(counts))
        if key not in cache:
            f = _features(est, a.model, T, counts, ranges)
            X = pd.DataFrame([{k: v for k, v in f.items() if not k.startswith("_")}])
            cols = list(getattr(est, "feature_names_in_", X.columns))
            cache[key] = (float(est.predict(X[cols])[0]), f.get("_bm"))
        pred, bm = cache[key]
        out.write(json.dumps({"run": label, "g": g, "T_pad": T, "arm": a.arm_label, "pred_ms": pred,
                              "bm": bm, "model_file": fname}) + "\n")
        n += 1
    out.close()
    print(f"PASS frontier predict ({a.arm_label}): {n} calls, {len(cache)} distinct, estimator {fname}")
    return 0


def _holdout(a) -> int:
    import numpy as np
    import pandas as pd
    df = pd.read_csv(a.csv)
    held = holdout_split(df["num_tokens"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_csv = out / "train_rows.csv"
    df[~df["num_tokens"].astype(int).isin(held)].to_csv(train_csv, index=False)
    rc = subprocess.run(train_cmd(sys.executable, a.model, train_csv, out / "models"), cwd=str(a.src),
                        env=env_for(a.src)).returncode
    if rc:
        print(f"FAIL frontier holdout: the trainer exited {rc}")
        return 3
    est, fname = _load_estimator(out / "models")
    test = df[df["num_tokens"].astype(int).isin(held)].copy()
    cols = list(est.feature_names_in_)
    from frontier.execution_time_predictor.moe_dataset_training import select_moe_operator_features
    if cols == ["total_routed_tokens", "num_tokens_post_padded"]:
        test, _ = select_moe_operator_features("moe_grouped_gemm", test)
    y = test["time_stats.moe_grouped_gemm.median"].to_numpy(float)
    p = est.predict(test[cols])
    e = (p - y) / y
    res = {"arm": a.arm_label, "held_out_points": sorted(held), "rows": int(len(y)), "MAPE": float(100 * np.mean(abs(e))),
           "p50_abs_e": float(100 * np.percentile(abs(e), 50)), "p95_abs_e": float(100 * np.percentile(abs(e), 95)),
           "bias": float(100 * np.mean(e)), "features": cols, "estimator": fname}
    (out / "holdout.json").write_text(json.dumps(res, indent=2))
    print(f"PASS frontier holdout ({a.arm_label}): {res['rows']} rows MAPE {res['MAPE']:.2f}% "
          f"p95 {res['p95_abs_e']:.2f}%")
    return 0


# ------------------------------------------------------------------------------------------------------------
# CLI (runs under our interpreter; the profiler/trainer under Frontier's, from frontier.env)
# ------------------------------------------------------------------------------------------------------------

def _fr_python(arm: str) -> str:
    var = {"primary": "FR_PY_PRIMARY", "fidelity": "FR_PY_FIDELITY"}[arm]
    py = os.environ.get(var)
    if not py:
        raise PL.KitError(f"${var} is unset: source $MOE_HOME/frontier/frontier.env (frontier_setup.sh)")
    return py


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("write-config", "profile", "train", "simpath", "predict", "holdout", "_simpath", "_predict", "_holdout"):
        p = sub.add_parser(c)
        p.add_argument("--model", required=True, choices=sorted(PL.MODELS))
        p.add_argument("--src", default=os.environ.get("FRONTIER_SRC", ""))
        p.add_argument("--arm", choices=("primary", "fidelity"), default="primary")
        p.add_argument("--arm-label", default=None)
        p.add_argument("--dists", nargs="+", default=["uniform"])
        p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
        p.add_argument("--csv", default=None)
        p.add_argument("--models", default=None)
        p.add_argument("--calls", action="append", default=[])
        p.add_argument("--sample", default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    a.arm_label = a.arm_label or f"{a.arm}-{'+'.join(a.dists)}"
    try:
        if a.cmd == "_simpath":
            return _simpath(a)
        if a.cmd == "_predict":
            return _predict(a)
        if a.cmd == "_holdout":
            return _holdout(a)
        if not a.src:
            raise PL.KitError("--src (or $FRONTIER_SRC) names Frontier's checkout")
        src = Path(a.src)
        if a.cmd == "write-config":
            p = write_config(a.model, src)
            print(f"PASS frontier config: {p}")
            return 0
        py = "python3" if a.dry_run else _fr_python(a.arm)
        out = Path(a.out or ".")
        if not a.dry_run:
            out.mkdir(parents=True, exist_ok=True)
        if a.cmd == "profile":
            env = env_for(src)
            env["PYTHON_BIN"] = py
            log = out / "profile.log"
            rc = run_logged(profile_cmd(src, a.model, a.dists, out, a.max_tokens), cwd=src, env=env, log=log,
                            dry=a.dry_run)
            if a.dry_run:
                return 0
            api = parse_api(log.read_text(errors="replace"))
            (out / "api.json").write_text(json.dumps({**api, "arm": a.arm, "dists": a.dists, "exit": rc}, indent=2))
            csv = expected_csv(out, a.model)
            if rc or not csv.exists():
                print(f"FAIL frontier profile ({a.arm}): exit {rc}, csv {'present' if csv.exists() else 'missing'}; "
                      f"see {log}")
                return 3
            print(f"PASS frontier profile ({a.arm}): vLLM {api['vllm']} API branch {api['api']}; {csv}")
            return 0
        if a.cmd == "train":
            rc = run_logged(train_cmd(py, a.model, a.csv, out), cwd=src, env=env_for(src), log=out / "train.log",
                            dry=a.dry_run)
            print(f"{'PASS' if rc == 0 else 'FAIL'} frontier train ({a.arm}): exit {rc}")
            return 0 if rc == 0 else 3
        inner = {"simpath": "_simpath", "predict": "_predict", "holdout": "_holdout"}[a.cmd]
        cmd = [py, str(Path(__file__).resolve()), inner, "--model", a.model, "--src", str(src), "--arm", a.arm,
               "--arm-label", a.arm_label, "--out", str(out if a.cmd != "predict" else Path(a.out))]
        for k in ("csv", "models", "sample"):
            if getattr(a, k):
                cmd += [f"--{k}", str(getattr(a, k))]
        for c in a.calls:
            cmd += ["--calls", c]
        rc = run_logged(cmd, cwd=src, env=env_for(src), log=Path(str(a.out) + f".{a.cmd}.log") if a.cmd == "predict"
                        else out / f"{a.cmd}.log", dry=a.dry_run)
        if not a.dry_run:
            print(f"{'PASS' if rc == 0 else 'FAIL'} frontier {a.cmd} ({a.arm_label}): exit {rc}")
        return 0 if rc == 0 else 3
    except PL.KitError as exc:
        print(f"FAIL frontier_op: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
