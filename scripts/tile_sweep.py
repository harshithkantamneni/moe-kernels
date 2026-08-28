#!/usr/bin/env python
"""Does forcing a bigger tile buy anything, now that we know 16 gives up WGMMA?

    python scripts/tile_sweep.py
    python scripts/tile_sweep.py --model deepseek-v3 --tokens 16,64,256

WHY THIS EXISTS. `check_mma_path.sh` measured that vLLM's decode path emits zero
`wgmma` and 32 `mma.sync.aligned.m16n8k16`, because its tuned config picks
`BLOCK_SIZE_M = 16` and Hopper's warpgroup MMA has M fixed at 64. So the obvious
question is whether that costs anything. Force 64, reach the warpgroup
instruction, and see.

THE PREDICTION, stated before the run so it can fail. If the study's thesis is
right and this regime is memory-bound, then:

  * 16 -> 32   both stay on mma.sync. Padded MACs double. Time should NOT move,
               because those MACs hide under the weight read.
  * 32 -> 64   the instruction should switch to wgmma. Time should still NOT
               improve, because the tensor core was never the constraint, and it
               may get WORSE as the larger accumulator costs occupancy.
  * 64 -> 128  more of the same, more occupancy lost.

A FLAT curve confirms the thesis. A curve that improves at 64 refutes it and
says vLLM's autotuner left performance on the table, which would be the more
interesting result and is worth wanting.

CONFOUND, named rather than hidden. BLOCK_SIZE_M sizes the register accumulator,
so changing it changes occupancy as well as the instruction. A time change is
therefore ambiguous between the two. Only a NULL result is clean evidence, which
is exactly what the thesis predicts, and is why this experiment is worth running
in this direction rather than the other.

Routing is uniform and T is small on purpose: max rows per expert stays far below
16, so every expert is one tile at every setting and weight traffic is identical
across the sweep. Without that the hot expert spills and traffic stops being
flat, which is the objection raised in GPU MODE against the original design.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from moe.reference.torch_ref import make_inputs  # noqa: E402
from moe.routing.distributions import sample_topk_ids  # noqa: E402
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec  # noqa: E402

#: Held fixed so only M varies. Taken from vLLM's tuned H200 entry at batch 16.
FIXED = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
         "num_warps": 8, "num_stages": 4}


def find_override():
    """vLLM's own tuning hook: try_get_optimal_moe_config consults get_config()
    first, and a truthy value bypasses both the tuned file and the default.

    Probed rather than assumed, because the import path has moved between
    versions and a wrong guess would silently sweep nothing.
    """
    candidates = [
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe.config",
    ]
    import importlib
    for name in candidates:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(mod, "override_config", None)
        if fn is not None:
            return fn, name
    raise SystemExit(
        "could not find vllm's override_config in any of:\n  "
        + "\n  ".join(candidates)
        + "\nCheck the installed vLLM version; try_get_optimal_moe_config reads "
          "it via get_config(), so the hook exists under some name.")


def _make_call(fused_experts, x, weights, w, ids, kw):
    """Bind the arguments explicitly rather than closing over loop variables.

    The closure was safe here because it is invoked inside the same iteration,
    but a `def` inside a loop that captures the loop's variables is one refactor
    away from a silent late-binding bug, and ruff B023 is right to flag it.
    """
    def call():
        return fused_experts(hidden_states=x, w1=weights.w1, w2=weights.w2,
                             topk_weights=w, topk_ids=ids, **kw)
    return call


def time_call(fn, warmup: int, iters: int) -> tuple[float, float]:
    """Median and stdev milliseconds. No L2 flush: the comparison is between
    tile settings on identical data, and adding a flush adds its own variance."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out), statistics.pstdev(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v3", choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--tokens", default="16,64,256")
    ap.add_argument("--tiles", default="16,32,64,128")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs the GPU box")

    override_config, where = find_override()
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from moe.baselines._framework_config import vllm_call_kwargs

    cfg = MODEL_CONFIGS[args.model]
    tiles = [int(v) for v in args.tiles.split(",")]
    print(f"override hook: {where}.override_config")
    print(f"model {args.model}  E={cfg.num_experts} k={cfg.top_k}  "
          f"fixed {FIXED}\n")

    for tok in (int(v) for v in args.tokens.split(",")):
        spec = BenchSpec(cfg, num_tokens=tok, dtype="bf16",
                         routing=RoutingSpec("uniform", 0.0), seed=args.seed)
        x, weights = make_inputs(spec, device="cuda")
        ids = sample_topk_ids(spec.routing, tok, cfg.num_experts, cfg.top_k,
                              seed=args.seed, device="cuda")
        counts = torch.bincount(ids.flatten(), minlength=cfg.num_experts)
        active = int((counts > 0).sum())
        mx = int(counts.max())
        w = torch.full(ids.shape, 1.0 / cfg.top_k, dtype=torch.float32, device="cuda")

        kw = vllm_call_kwargs(spec)
        kw["activation"] = MoEActivation(kw["activation"])

        print(f"T={tok}: {active} active experts, max {mx} rows on one expert"
              + ("   <-- WARNING: max >= smallest tile, traffic will not be flat"
                 if mx >= min(tiles) else ""))
        print(f"  {'BLOCK_SIZE_M':>13} {'ms p50':>10} {'stdev':>8} {'vs first':>9}   note")
        base = None
        for bm in tiles:
            conf = dict(FIXED, BLOCK_SIZE_M=bm)
            call = _make_call(fused_experts, x, weights, w, ids, kw)
            with override_config(conf):
                try:
                    ms, sd = time_call(call, args.warmup, args.iters)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {bm:13d} {'FAILED':>10}   {type(exc).__name__}: {exc}")
                    continue
            base = base if base is not None else ms
            note = "mma.sync (M<64)" if bm < 64 else "wgmma reachable (M>=64)"
            print(f"  {bm:13d} {ms:10.4f} {sd:8.4f} {ms / base:7.3f}x   {note}")
        print()

    print("READING IT. Flat across all four confirms the thesis: neither the")
    print("padded arithmetic nor the tensor-core instruction is on the critical")
    print("path, because the weight read is. An improvement at 64 refutes it.")
    print("Confirm the instruction actually changed with check_mma_path.sh; this")
    print("script asserts nothing about what was emitted, only what it cost.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
