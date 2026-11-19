"""Frontier (github.com/NetX-lab/Frontier at db5a85b, MIT): ITS PREDICTOR, OUR PROFILES (owner decision 2026-11-19).

    python scripts/prod/frontier_op.py grid --max-tokens 16384                      # Frontier's token grid (pure)
    python scripts/prod/frontier_op.py fsample --model M --branch prefill --page 0 --pages 16 --out PAGE_DIR
    (the replay harness times PAGE_DIR/sample.json under nsys at the branch clock; nsys_extract.py replay-cells)
    python scripts/prod/frontier_op.py fpoints --model M --pages-root POINTS_DIR --out POINTS.json
    FR=$MOE_HOME/frontier; . $FR/frontier.env                                        # frontier_setup.sh wrote it
    python scripts/prod/frontier_op.py csv     --model M --points POINTS.json --src $FRONTIER_SRC --out PROFILE_DIR
    python scripts/prod/frontier_op.py train   --model M --csv CSV --src ... --out MODELS        # its RF, its trainer
    python scripts/prod/frontier_op.py simpath --model M --csv CSV --src ... --out MODELS_SIMPATH
    python scripts/prod/frontier_op.py holdout --model M --csv CSV --src ... --out HOLDOUT
    python scripts/prod/frontier_op.py predict --model M --models cli=MODELS --models simpath=MODELS_SIMPATH \\
        --csv CSV --calls A1710=RUN ... --src ... --out PRED_DIR                         # on the VM, one batch per model
    python scripts/prod/frontier_op.py grid-check --src ...       # the grid above == Frontier's own function

WHY NOT FRONTIER'S PROFILER (smoke rental 2026-11-19, logs/frontier-*.log): on vLLM 0.27.1 its MoE profiler builds the
gate with vLLM's ReplicatedLinear, which needs vLLM's tensor-parallel group and set_current_vllm_config (AssertionError
"tensor model parallel group is not initialized"); its own pinned stack (torch 2.8 + vLLM 0.10.2) has no CUDA torch
for aarch64 (the cu128 index carries no aarch64 wheel; PyPI's aarch64 torch 2.8 is CPU-only). So on the GH200:

  PROFILER = OURS. The points are timed on the production kernel (vLLM 0.27.1's fused path, align .. moe_sum, the
             functional fused_experts entry as the replay calls it: the same functional-vs-modular caveat as replay C,
             SMOKE-FIXES 6) by the study's validated replay harness (scripts/prod/replay.py, r3_decode's GRR
             instrument under locked_r3) at the BRANCH CLOCK (T <= 128 at 1710 MHz, T > 128 at 1410), under nsys with
             cell marks; the time is the KIT'S KERNEL-SUM (the core align..moe_sum kernels' GPU durations summed, the
             API path's fill kernels stripped: nsys_extract.replay_cells) of the timed graph calls, in ms.
  SCHEMA, GRID, ROUTING, TRAINER, PREDICTOR = FRONTIER'S:
    grid     frontier/profiling/utils/__init__.py get_num_tokens_to_profile(16384) (profile_moe.sh --max-tokens
             16384): 387 points (token_grid; `grid-check` holds it to Frontier's own function on the VM).
    routing  round-robin top-k, Frontier's uniform_topk (frontier/profiling/moe/moe_impl.py:75-102: token t takes
             experts (t k + j) mod E, weights 1/k); in Frontier's load-distribution names that generator is
             "balanced" (load_distribution.py: "the round-robin assignment of the uniform_topk routing path"), and the
             rows say so. Frontier's "uniform" distribution (seeded random per-token top-k, torch RNG on the card) is
             available as --routing uniform (three seeds per point, Frontier's num_samples_per_distribution).
    columns  the row Frontier's MoE profiler writes for the grouped GEMM (moe_wrapper.py profile_grouped_gemm +
             _build_profile_result, main.py profile_model / _attach_moe_output_metadata, moe_vllm_kernel.py's
             KERNEL_ONLY stats): num_tokens, the 14 load-imbalance features + load_distribution + seed from Frontier's
             OWN MoELoadImbalanceInput(expert_token_counts=the point's counts).to_features_dict(), the model columns,
             the routing / gating-context metadata, profiling_precision BF16, measurement_type KERNEL_ONLY, and
             time_stats.moe_grouped_gemm.{median, mean, min, max, std, count, kernel_count} in ms, with
             num_tokens_post_padded, block_size_m and block_size_m_ranges (vLLM's default config, r3_decode, held to
             the kernel by GD5). The three MoE operators we do not time (moe_gating_linear, moe_gating_routing_topk,
             moe_shuffling) carry their target columns EMPTY (NaN: not measured, never a made-up number).
    trainer  `python -m frontier.training.cli moe` (its argument parser, its MoETrainer, its RandomForest grid search
             and K-fold CV, its pickled estimator), explicit-parameter mode, --measurement_type KERNEL_ONLY, in
             Frontier's venv. THE ONE SHIM (disclosed): MoETrainer._get_model_names is narrowed to
             ["moe_grouped_gemm"] for the run, since the other three operators have no measured rows (the CLI would
             stop on their empty targets). The `simpath` model is the same trainer with its feature selection swapped
             for the simulator's select_moe_operator_features (total_routed_tokens, num_tokens_post_padded).
    predict  every logged call of every run in ONE DataFrame per model, features by Frontier's own
             MoELoadImbalanceInput(expert_token_counts=real counts) (simpath: the padded count at Frontier's own
             block_size_m_for_tokens over the table's ranges), on the VM.
  ONE ROW PER GRID POINT under round-robin routing: Frontier writes num_samples_per_distribution (3) rows per point;
  round-robin ids do not depend on the seed, so those would be three timings of one input; the replay's repeats time
  it (the median is the row's median).
  BUDGET: the 387 points cost more than the main rental can give, so the decode points (19) are one page and the
  prefill points are interleaved into PAGES (16) pages (page j = every 16th point from j), measured in PAGE_ORDER
  (bit-reversed: 0, 8, 4, 12, ... so any prefix of pages is an even cover of the grid) until the Frontier budget
  (main.sh) is spent; the README states how many fit.
"""
from __future__ import annotations

import argparse
import json
import os
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
ROUTINGS = ("balanced", "uniform")
#: Frontier's main.py --num_samples_per_distribution default (the smoke's moe_config.yaml)
SAMPLES_PER_DISTRIBUTION = 3
PAGES = 16
HOLDOUT_MOD, HOLDOUT_REM = 5, 2
TIME_COL = "time_stats.moe_grouped_gemm.median"
#: Frontier's MoE family (frontier/operators/families.py MOE_FAMILY profiling_order); only the grouped GEMM is timed
MOE_OPERATORS = ("moe_gating_linear", "moe_gating_routing_topk", "moe_shuffling", "moe_grouped_gemm")
#: get_routing_runtime_metadata("uniform_topk") (moe_impl.py:236-258) and the prefill_hot gating context
#: (moe_gating_runtime.py:76-88): profile_moe.sh's defaults, so the rows carry what Frontier's profile would
ROUTING_METADATA = {"routing_runtime_path": "uniform_topk", "routing_assignment_policy": "round_robin_uniform",
                    "routing_weight_policy": "uniform_1_over_topk", "routing_uses_router_logits": False}
#: frontier/training/cli.py moe defaults (the trainer's grid search), mirrored for the simpath trainer
CLI_TRAINING = {"k_fold_cv_splits": 10, "num_training_job_threads": -1, "num_estimators": [250, 500, 750],
                "max_depth": [8, 16, 32], "min_samples_split": [2, 5, 10]}

#: the served models in Frontier's HF-config dialect (kept for Frontier's model registry; the explicit-parameter
#: trainer does not read it)
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


def csv_relpath(model_key: str) -> Path:
    """build_profile_method_output_path(..., op_name="moe", profile_method="record_function"): the KERNEL_ONLY file."""
    name, _ = frontier_config(model_key)
    return Path("compute") / DEVICE / name / "moe_kernel_only.csv"


# ------------------------------------------------------------------------------------------------------------
# Frontier's grid and routing (pure)
# ------------------------------------------------------------------------------------------------------------

def token_grid(max_tokens: int = MAX_TOKENS) -> list[int]:
    """frontier/profiling/utils/__init__.py get_num_tokens_to_profile(max_tokens) at db5a85b, verbatim in effect:
    NUM_TOKENS_SPACE up to max_tokens, plus max_tokens, deduplicated, sorted DESCENDING (its order)."""
    space = ([1, 2, 4] + list(range(8, 1024, 8)) + list(range(1024, 2 * 1024 + 1, 16))
             + list(range(2 * 1024, 4 * 1024 + 1, 32)) + list(range(4 * 1024, 8 * 1024 + 1, 64))
             + list(range(8 * 1024, 16 * 1024 + 1, 128)) + list(range(16 * 1024, 32 * 1024 + 1, 256))
             + list(range(32 * 1024, 64 * 1024 + 1, 512)) + list(range(64 * 1024, 128 * 1024 + 1, 1024)))
    out = []
    for n in space:
        if n <= max_tokens:
            out.append(n)
        else:
            break
    if max_tokens > 0:
        out.append(max_tokens)
    return sorted(set(out), reverse=True)


def round_robin_ids(T: int, E: int, k: int):
    """Frontier's uniform_topk (moe_impl.py:97-99): topk_ids[t, j] = (t k + j) mod E."""
    import numpy as np
    t = np.arange(int(T), dtype=np.int64)[:, None]
    return (t * int(k) + np.arange(int(k), dtype=np.int64)[None, :]) % int(E)


def uniform_ids(T: int, E: int, k: int, seed: int):
    """Frontier's "uniform" load distribution (load_distribution.py generate_expert_routing): torch.manual_seed(seed),
    np.random.seed(seed), rand(T, E) on the card (cpu without one), argsort descending, the first k."""
    import numpy as np
    import torch
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.rand(int(T), int(E), device=dev)
    return torch.argsort(m, dim=1, descending=True)[:, :int(k)].int().cpu().numpy().astype(np.int64)


def page_order(pages: int) -> list[int]:
    """Bit-reversed page order (pages a power of two): any prefix of it is an even cover of the grid."""
    if pages < 1 or pages & (pages - 1):
        raise PL.KitError(f"--pages {pages}: a power of two")
    bits = pages.bit_length() - 1
    return [int(format(i, f"0{bits}b")[::-1], 2) if bits else 0 for i in range(pages)]


def branch_points(branch: str, max_tokens: int) -> list[int]:
    return sorted(T for T in token_grid(max_tokens) if PL.branch(T) == branch)


def page_points(branch: str, page: int, pages: int, max_tokens: int) -> list[int]:
    """decode: every decode point (one page); prefill: page j of `pages` = every pages-th prefill point from j."""
    pts = branch_points(branch, max_tokens)
    if branch == "decode":
        if page != 0:
            raise PL.KitError("the decode points are one page (page 0)")
        return pts
    if not 0 <= page < pages:
        raise PL.KitError(f"page {page} of {pages}")
    return pts[page::pages]


def fsample_cells(model_key: str, Ts: list[int], routing: str, seeds: int = SAMPLES_PER_DISTRIBUTION) -> list:
    """(cell, ids) pairs for replay.write_synthetic_sample: label f<T> (balanced) or f<T>s<seed> (uniform)."""
    cfg = PL.MODELS[model_key].cfg
    if routing not in ROUTINGS:
        raise PL.KitError(f"--routing {routing}: one of {ROUTINGS}")
    out = []
    for T in Ts:
        for seed in ([0] if routing == "balanced" else range(int(seeds))):
            ids = round_robin_ids(T, cfg.num_experts, cfg.top_k) if routing == "balanced" else \
                uniform_ids(T, cfg.num_experts, cfg.top_k, seed)
            lab = f"f{T}" if routing == "balanced" else f"f{T}s{seed}"
            out.append(({"label": lab, "g": None, "T": int(T), "T_unpadded": int(T), "layer": None,
                         "tile": PL.tile_bin(T), "branch": PL.branch(T),
                         "counts": PL.counts_from_ids(ids, cfg.num_experts), "ids_digest": PL.digest(ids),
                         "step": None, "synthetic": True,
                         "frontier": {"num_tokens": int(T), "load_distribution": routing, "seed": int(seed)}}, ids))
    return out


# ------------------------------------------------------------------------------------------------------------
# the points (our interpreter: r3_decode's geometry = vLLM 0.27.1's default config, GD5-checked on the kernel)
# ------------------------------------------------------------------------------------------------------------

def bm_ranges(model_key: str, max_tokens: int) -> list[list[int]]:
    """[first, last, BLOCK_SIZE_M] ranges of vLLM's default config over 1..max_tokens (moe_vllm_kernel.py
    block_size_m_ranges' format)."""
    import r3_decode as RD
    E = PL.MODELS[model_key].cfg.num_experts
    out: list[list[int]] = []
    for T in range(1, int(max_tokens) + 1):
        bm = int(RD.default_config(T, E)["BLOCK_SIZE_M"])
        if out and out[-1][2] == bm:
            out[-1][1] = T
        else:
            out.append([T, T, bm])
    return out


def build_points(model_key: str, pages_root, max_tokens: int) -> dict:
    """Every measured point of every page under pages_root (each page dir: sample.json + cells.json from
    nsys_extract replay-cells), with its geometry. A page without cells.json was not measured (budget or failure)."""
    import r3_decode as RD
    cfg = PL.MODELS[model_key].cfg
    pts, pages, unmeasured = [], [], []
    for d in sorted(Path(pages_root).iterdir()):
        if not (d / "sample.json").is_file():
            continue
        if not (d / "cells.json").is_file():
            unmeasured.append(d.name)
            continue
        sample = json.loads((d / "sample.json").read_text())
        meas = {c["label"]: c for c in json.loads((d / "cells.json").read_text())["cells"]}
        pages.append(d.name)
        for c in sample["cells"]:
            m = meas.get(c["label"])
            if m is None:
                continue
            g = RD.geometry(cfg, int(c["T"]), c["counts"])
            pts.append({"label": c["label"], "page": d.name, "num_tokens": int(c["T"]), "counts": c["counts"],
                        "load_distribution": c["frontier"]["load_distribution"], "seed": c["frontier"]["seed"],
                        "block_size_m": int(g.bm), "num_tokens_post_padded": int(g.post_padded), "tile": g.tile,
                        "kernel_sum_ms": m["kernel_sum_ms"], "n": m["n"], "kernel_count": m["kernel_count"],
                        "fills_n": m.get("fills_n"), "fills_ms": m.get("fills_ms"), "source": m.get("source")})
    if not pts:
        raise PL.KitError(f"{pages_root}: no measured Frontier point")
    grid = token_grid(max_tokens)
    have = {p["num_tokens"] for p in pts}
    return {"schema": "moe-kernels/prod-frontier-points/1", "model": model_key, "max_tokens": int(max_tokens),
            "grid_points": len(grid), "measured_points": len(have), "pages": pages, "unmeasured_pages": unmeasured,
            "block_size_m_ranges": bm_ranges(model_key, max_tokens), "points": pts}


# ------------------------------------------------------------------------------------------------------------
# inside Frontier's interpreter (imports frontier, pandas, sklearn): _csv, _train, _simpath, _holdout, _predict
# ------------------------------------------------------------------------------------------------------------

def frontier_rows(model_key: str, doc: dict) -> list[dict]:
    """The CSV rows, Frontier's columns (module doc), features from Frontier's own MoELoadImbalanceInput."""
    from frontier.moe_load_imbalance import MoELoadImbalanceInput
    cfg = PL.MODELS[model_key].cfg
    ranges = json.dumps(doc["block_size_m_ranges"])
    rows = []
    for p in sorted(doc["points"], key=lambda p: (-p["num_tokens"], p["seed"])):
        ks = p["kernel_sum_ms"]
        li = MoELoadImbalanceInput(num_tokens=int(p["num_tokens"]), num_experts_per_device=cfg.num_experts,
                                   hidden_dim=cfg.hidden_size, expert_hidden_dim=cfg.intermediate_size,
                                   router_topk=cfg.top_k, load_distribution=p["load_distribution"],
                                   expert_token_counts=[int(c) for c in p["counts"]], seed=int(p["seed"]),
                                   tensor_parallel_size=1)
        row = {f"time_stats.{op}.median": float("nan") for op in MOE_OPERATORS if op != "moe_grouped_gemm"}
        row.update({"time_stats.moe_grouped_gemm.min": ks["min"], "time_stats.moe_grouped_gemm.max": ks["max"],
                    "time_stats.moe_grouped_gemm.mean": ks["mean"], TIME_COL: ks["median"],
                    "time_stats.moe_grouped_gemm.std": ks["std"], "time_stats.moe_grouped_gemm.count": int(p["n"]),
                    "time_stats.moe_grouped_gemm.kernel_count": int(p["kernel_count"]),
                    "time_stats.moe_grouped_gemm.block_size_m": int(p["block_size_m"]),
                    "time_stats.moe_grouped_gemm.num_tokens_post_padded": int(p["num_tokens_post_padded"]),
                    "time_stats.moe_grouped_gemm.block_size_m_ranges": ranges,
                    "num_tokens": int(p["num_tokens"]), "num_experts": cfg.num_experts,
                    "num_experts_per_device": cfg.num_experts, "expert_parallel_size": 1, **ROUTING_METADATA,
                    "gating_runtime_context": "prefill_hot", "router_topk": cfg.top_k,
                    "hidden_dim": cfg.hidden_size, "expert_hidden_dim": cfg.intermediate_size, "use_gated": True,
                    "num_tensor_parallel_workers": 1, **li.to_features_dict(),
                    "moe_grouped_gemm_backend": "vllm_fused", "moe_quantization_mode": "bf16",
                    "measurement_type": "KERNEL_ONLY", "profiling_precision": "BF16", "model_arch": "generic",
                    "model_architecture_profile": "generic", "quant_signature": "none"})
        rows.append(row)
    return rows


def _csv(a) -> int:
    import pandas as pd
    doc = json.loads(Path(a.points).read_text())
    if doc.get("model") != a.model:
        raise PL.KitError(f"{a.points}: points of {doc.get('model')!r}, not {a.model}")
    out = Path(a.out) / csv_relpath(a.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frontier_rows(a.model, doc)).to_csv(out, index=False)
    print(f"PASS frontier csv: {len(doc['points'])} rows ({doc['measured_points']} of {doc['grid_points']} grid "
          f"points) -> {out}")
    return 0


def cli_argv(model_key: str, csv, out) -> list[str]:
    """Frontier's CLI, explicit-parameter mode (cli.py "Configuration mode 2"), KERNEL_ONLY."""
    cfg = PL.MODELS[model_key].cfg
    return ["frontier.training.cli", "moe", "--dataset_path", str(Path(csv).resolve()), "--output_dir",
            str(Path(out).resolve()), "--measurement_type", "KERNEL_ONLY", "--device", DEVICE,
            "--num_experts", str(cfg.num_experts), "--router_topk", str(cfg.top_k), "--hidden_dim",
            str(cfg.hidden_size), "--expert_hidden_dim", str(cfg.intermediate_size), "--moe_tensor_parallel_size",
            "1", "--expert_parallel_size", "1", "--routing_runtime_path", "uniform_topk", "--gating_runtime_context",
            "prefill_hot"]


def _narrow_to_grouped_gemm():
    """THE ONE SHIM: MoETrainer trains only moe_grouped_gemm (the operators we do not time have no rows)."""
    from frontier.training import moe_trainer
    moe_trainer.MoETrainer._get_model_names = lambda self: ["moe_grouped_gemm"]


def run_frontier_cli(model_key: str, csv, out) -> int:
    from frontier.training import cli
    _narrow_to_grouped_gemm()
    saved = sys.argv
    sys.argv = cli_argv(model_key, csv, out)
    try:
        rc = cli.main()
    finally:
        sys.argv = saved
    return int(rc or 0)


def _train(a) -> int:
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rc = run_frontier_cli(a.model, a.csv, a.out)
    print(f"{'PASS' if rc == 0 else 'FAIL'} frontier train (cli): exit {rc}")
    return 0 if rc == 0 else 3


def _simpath(a) -> int:
    """Frontier's MoETrainer (the CLI's parameters) with its feature selection swapped for the simulator's."""
    import pandas as pd
    from frontier.execution_time_predictor.moe_dataset_training import select_moe_operator_features
    from frontier.training.moe_trainer import MoETrainer
    _narrow_to_grouped_gemm()
    cfg = PL.MODELS[a.model].cfg
    tr = MoETrainer(dataset_path=str(Path(a.csv).resolve()), output_dir=str(Path(a.out).resolve()),
                    num_experts=cfg.num_experts, router_topk=cfg.top_k, hidden_dim=cfg.hidden_size,
                    expert_hidden_dim=cfg.intermediate_size, moe_tensor_parallel_size=1, expert_parallel_size=1,
                    predictor_type="random_forest", measurement_type="KERNEL_ONLY",
                    routing_runtime_path="uniform_topk", gating_runtime_context="prefill_hot", **CLI_TRAINING)
    df = pd.read_csv(a.csv)
    _df, feats = select_moe_operator_features("moe_grouped_gemm", df)
    orig = tr._get_feature_cols
    tr._get_feature_cols = lambda m: list(feats) if m == "moe_grouped_gemm" else orig(m)
    if "num_tokens_post_padded" in feats:
        orig_df = tr._get_training_df_for_model

        def _tdf(df, model_name, feature_cols, target_col):     # MoETrainer.train calls it by keyword
            if model_name == "moe_grouped_gemm":
                df, _f = select_moe_operator_features(model_name, df)
            return orig_df(df=df, model_name=model_name, feature_cols=feature_cols, target_col=target_col)
        tr._get_training_df_for_model = _tdf
    tr.train()
    print(json.dumps({"simpath_features": list(feats)}))
    return 0


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


def _check_versions(models_dir) -> dict:
    """The estimator is unpickled under the library versions it was trained with (versions.json, written by train)."""
    import sklearn
    f = Path(models_dir) / "versions.json"
    if not f.is_file():
        raise PL.KitError(f"{f}: missing (frontier_op.py train records the trainer's library versions)")
    want = json.loads(f.read_text())
    if want.get("sklearn") != sklearn.__version__:
        raise PL.KitError(f"scikit-learn {sklearn.__version__} here, {want.get('sklearn')} at training: install the "
                          "trainer's version before predicting")
    return want


PRED_SCHEMA = "moe-kernels/prod-frontier-predictions/1"


def _predict(a) -> int:
    """Every call's prediction, per model ONE DataFrame and ONE est.predict over the distinct (T, counts) keys (a
    batch, not a row at a time); the calls are read once for every model. On the VM after the runs. Written compact
    (under the push's size cap): OUT/index.json, OUT/<run>.g.npy (the calls' g, int64) and OUT/<arm>.<run>.npy
    (pred_ms, float32, aligned with <run>.g.npy)."""
    import numpy as np
    import pandas as pd
    models = []
    for spec in a.models_list:
        arm, _, d = spec.partition("=")
        if not d:
            arm, d = a.arm_label, spec
        _check_versions(d)
        est, fname = _load_estimator(d)
        if hasattr(est, "n_jobs"):
            est.n_jobs = -1          # the trees in parallel: the same numbers, faster
        models.append((arm, est, fname))
    if not models:
        raise PL.KitError("no --models ARM=DIR")
    ranges = _ranges(a.csv) if a.csv else None
    jobs: dict = {}
    for spec in a.calls or []:
        label, _, d = spec.partition("=")
        rows = PL.read_calls(d)
        counts = PL.load_counts(d)
        jobs[label] = [(r["g"], r["T_pad"], [int(x) for x in c]) for r, c in zip(rows, counts, strict=True)
                       if r.get("status") == "ok"]
    if a.sample:
        doc = json.loads(Path(a.sample).read_text())
        jobs["sample"] = [(c["g"], c["T"], c["counts"]) for c in doc["cells"]]
    keys: dict = {}
    for lst in jobs.values():
        for _g, T, counts in lst:
            keys.setdefault((int(T), tuple(int(c) for c in counts)), len(keys))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {"schema": PRED_SCHEMA, "model": a.model, "runs": {}, "arms": {}}
    for label, lst in jobs.items():
        np.save(out / f"{label}.g.npy", np.asarray([-1 if g is None else int(g) for g, _T, _c in lst], dtype=np.int64))
        index["runs"][label] = {"g": f"{label}.g.npy", "n": len(lst)}
    for arm, est, fname in models:
        feats = []
        for (T, counts) in keys:
            f = _features(est, a.model, T, counts, ranges)
            feats.append({k: v for k, v in f.items() if not k.startswith("_")})
        y = np.zeros(0)
        if feats:
            X = pd.DataFrame(feats)
            cols = list(getattr(est, "feature_names_in_", X.columns))
            y = np.asarray(est.predict(X[cols]), dtype=float)
        index["arms"][arm] = {"model_file": fname, "features": list(getattr(est, "feature_names_in_", [])),
                              "distinct": len(keys), "files": {}}
        for label, lst in jobs.items():
            pred = np.asarray([y[keys[(int(T), tuple(int(c) for c in counts))]] for _g, T, counts in lst],
                              dtype=np.float32)
            np.save(out / f"{arm}.{label}.npy", pred)
            index["arms"][arm]["files"][label] = f"{arm}.{label}.npy"
        print(f"PASS frontier predict ({arm}): {sum(len(v) for v in jobs.values())} calls, {len(keys)} distinct in "
              f"one batch, estimator {fname}")
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return 0


def load_predictions(d) -> dict:
    """{(run, g, arm): pred_ms} from a _predict output directory."""
    import numpy as np
    d = Path(d)
    index = json.loads((d / "index.json").read_text())
    if index.get("schema") != PRED_SCHEMA:
        raise PL.KitError(f"{d}/index.json: schema {index.get('schema')!r}")
    out = {}
    for arm, spec in index["arms"].items():
        for run, fname in spec["files"].items():
            gs = np.load(d / index["runs"][run]["g"])
            ps = np.load(d / fname)
            if len(gs) != len(ps):
                raise PL.KitError(f"{d}: {fname} has {len(ps)} predictions for {len(gs)} calls")
            for g, p in zip(gs.tolist(), ps.tolist(), strict=True):
                out[(run, int(g), arm)] = float(p)
    return out


def holdout_split(num_tokens_values) -> set:
    """The held-out num_tokens points: every 5th distinct value (index % 5 == 2), ascending order."""
    vals = sorted(set(int(v) for v in num_tokens_values))
    return {v for i, v in enumerate(vals) if i % HOLDOUT_MOD == HOLDOUT_REM}


def _holdout(a) -> int:
    """Frontier's fidelity on its OWN profiled points (ours, in its schema): the CLI trainer on the points not held
    out, its estimator on the held-out rows."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(a.csv)
    held = holdout_split(df["num_tokens"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_csv = out / "train_rows.csv"
    df[~df["num_tokens"].astype(int).isin(held)].to_csv(train_csv, index=False)
    rc = run_frontier_cli(a.model, train_csv, out / "models")
    if rc:
        print(f"FAIL frontier holdout: the trainer exited {rc}")
        return 3
    est, fname = _load_estimator(out / "models")
    test = df[df["num_tokens"].astype(int).isin(held)].copy()
    cols = list(est.feature_names_in_)
    y = test[TIME_COL].to_numpy(float)
    p = est.predict(test[cols])
    e = (p - y) / y
    res = {"arm": a.arm_label, "held_out_points": sorted(held), "rows": int(len(y)),
           "MAPE": float(100 * np.mean(abs(e))), "p50_abs_e": float(100 * np.percentile(abs(e), 50)),
           "p95_abs_e": float(100 * np.percentile(abs(e), 95)), "bias": float(100 * np.mean(e)), "features": cols,
           "estimator": fname, "truth": "the kit's kernel-sum (align..moe_sum), ms"}
    (out / "holdout.json").write_text(json.dumps(res, indent=2))
    print(f"PASS frontier holdout ({a.arm_label}): {res['rows']} rows MAPE {res['MAPE']:.2f}% "
          f"p95 {res['p95_abs_e']:.2f}%")
    return 0


def _grid_check(a) -> int:
    from frontier.profiling.utils import get_num_tokens_to_profile
    theirs = list(get_num_tokens_to_profile(a.max_tokens))
    ours = token_grid(a.max_tokens)
    if theirs != ours:
        print(f"FAIL frontier grid-check: Frontier's grid ({len(theirs)} points) != ours ({len(ours)})")
        return 3
    print(f"PASS frontier grid-check: {len(ours)} points to {a.max_tokens}, identical to Frontier's own")
    return 0


# ------------------------------------------------------------------------------------------------------------
# CLI (our interpreter; train / simpath / holdout / predict / csv / grid-check re-run this file under Frontier's)
# ------------------------------------------------------------------------------------------------------------

VERSIONS_PY = ("import json, sklearn, pandas, numpy, sys; print(json.dumps({'sklearn': sklearn.__version__, "
               "'pandas': pandas.__version__, 'numpy': numpy.__version__, 'python': sys.version.split()[0]}))")
INNER = {"csv": "_csv", "train": "_train", "simpath": "_simpath", "holdout": "_holdout", "predict": "_predict",
         "grid-check": "_grid_check"}


def record_versions(py, out) -> int:
    """versions.json beside the trained models: the libraries the pickles need to load (predict checks them)."""
    p = subprocess.run([str(py), "-c", VERSIONS_PY], capture_output=True, text=True)
    if p.returncode:
        print(f"FAIL frontier train: the trainer's library versions could not be read: {p.stderr[-300:]}")
        return 3
    (Path(out) / "versions.json").write_text(p.stdout.strip().splitlines()[-1] + "\n")
    return 0


def env_for(src) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(src)) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _fr_python() -> str:
    py = os.environ.get("FR_PY_PRIMARY")
    if not py:
        raise PL.KitError("$FR_PY_PRIMARY is unset: source $MOE_HOME/frontier/frontier.env (frontier_setup.sh)")
    return py


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("grid", "fsample", "fpoints", "write-config", *INNER, *INNER.values()):
        p = sub.add_parser(c)
        p.add_argument("--model", default="ernie-4.5-21b-a3b", choices=sorted(PL.MODELS))
        p.add_argument("--src", default=os.environ.get("FRONTIER_SRC", ""))
        p.add_argument("--arm-label", default="cli")
        p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
        p.add_argument("--branch", choices=("decode", "prefill"), default=None)
        p.add_argument("--page", type=int, default=0)
        p.add_argument("--pages", type=int, default=PAGES)
        p.add_argument("--routing", choices=ROUTINGS, default="balanced")
        p.add_argument("--pages-root", default=None)
        p.add_argument("--points", default=None)
        p.add_argument("--csv", default=None)
        p.add_argument("--models", dest="models_list", action="append", default=[], help="ARM=DIR (predict)")
        p.add_argument("--calls", action="append", default=[])
        p.add_argument("--sample", default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "grid":
            g = token_grid(a.max_tokens)
            print(json.dumps({"max_tokens": a.max_tokens, "points": len(g), "grid": g}))
            return 0
        if a.cmd == "fsample":
            if not a.branch or not a.out:
                raise PL.KitError("fsample needs --branch and --out")
            import replay as RP
            Ts = page_points(a.branch, a.page, a.pages, a.max_tokens)
            pairs = fsample_cells(a.model, Ts, a.routing)
            p = RP.write_synthetic_sample(PL.MODELS[a.model], pairs, a.out, extra={
                "frontier": {"branch": a.branch, "page": a.page, "pages": a.pages, "routing": a.routing,
                             "max_tokens": a.max_tokens, "grid": "get_num_tokens_to_profile"}})
            print(f"PASS frontier fsample: {a.branch} page {a.page}/{a.pages}: {len(pairs)} cells "
                  f"(T {min(Ts)}..{max(Ts)}) -> {p}")
            return 0
        if a.cmd == "fpoints":
            doc = build_points(a.model, a.pages_root, a.max_tokens)
            Path(a.out).write_text(json.dumps(doc, indent=1))
            print(f"PASS frontier points: {len(doc['points'])} rows, {doc['measured_points']} of {doc['grid_points']} "
                  f"grid points, pages {doc['pages']} (unmeasured: {doc['unmeasured_pages']}) -> {a.out}")
            return 0
        if a.cmd == "write-config":
            print(f"PASS frontier config: {write_config(a.model, a.src)}")
            return 0
        inner = {v: v for v in INNER.values()}
        if a.cmd in inner:
            return {"_csv": _csv, "_train": _train, "_simpath": _simpath, "_holdout": _holdout,
                    "_predict": _predict, "_grid_check": _grid_check}[a.cmd](a)
        if not a.src:
            raise PL.KitError("--src (or $FRONTIER_SRC) names Frontier's checkout")
        src = Path(a.src)
        py = "python3" if a.dry_run else _fr_python()
        cmd = [py, str(Path(__file__).resolve()), INNER[a.cmd], "--model", a.model, "--src", str(src),
               "--arm-label", a.arm_label, "--max-tokens", str(a.max_tokens)]
        for k in ("csv", "points", "sample", "out"):
            if getattr(a, k):
                cmd += [f"--{k}", str(Path(getattr(a, k)).resolve())]
        for m in a.models_list:
            cmd += ["--models", m]
        for c in a.calls:
            cmd += ["--calls", c]
        print("  $ " + shlex.join(str(c) for c in cmd), flush=True)
        if a.dry_run:
            return 0
        if a.cmd in ("train", "simpath", "holdout", "csv", "predict") and a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(cmd, cwd=str(src), env=env_for(src)).returncode
        if rc == 0 and a.cmd in ("train", "simpath"):
            rc = record_versions(py, a.out)
        print(f"{'PASS' if rc == 0 else 'FAIL'} frontier {a.cmd} ({a.arm_label}): exit {rc}")
        return 0 if rc == 0 else 3
    except PL.KitError as exc:
        print(f"FAIL frontier_op: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
