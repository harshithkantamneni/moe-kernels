#!/usr/bin/env python
"""Does forcing a bigger tile buy anything, now that we know 16 gives up WGMMA?

    python scripts/tile_sweep.py
    python scripts/tile_sweep.py --model deepseek-v3 --tokens 16,64,256
    python scripts/tile_sweep.py --dump-ptx /workspace/ptx-tiles   # verify the ISA

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
import os
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


def arm_ptx_dump(directory: Path) -> None:
    """Make Triton write its generated PTX, and guarantee it recompiles.

    Both halves matter. TRITON_KERNEL_DUMP/TRITON_DUMP_DIR ask for the dump, but
    Triton does not recompile a kernel it has already cached, and a cache hit
    dumps nothing. Pointing TRITON_CACHE_DIR at a fresh directory forces every
    specialisation to be built, so every tile setting produces a file.

    Called before vLLM is imported, since Triton reads these at compile time and
    the first compile happens on the first fused_experts call.
    """
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(directory)
    os.environ["TRITON_CACHE_DIR"] = str(directory / "_cache")


def scan_new_ptx(directory: Path, seen: set[Path]) -> tuple[int, int, list[str]]:
    """Instructions in PTX files that appeared since the last call.

    Returns (wgmma count, mma.sync count, distinct shapes). Each BLOCK_SIZE_M is
    a different Triton specialisation and therefore a different cache entry, so
    the files that appear after a setting ran belong to that setting.
    """
    import re
    fresh = [q for q in directory.rglob("*.ptx") if q not in seen]
    seen.update(fresh)
    w = m = 0
    shapes: set[str] = set()
    for q in fresh:
        try:
            text = q.read_text(errors="ignore")
        except OSError:
            continue
        w += len(re.findall(r"wgmma\.", text))
        m += len(re.findall(r"mma\.sync\.", text))
        # Not `wgmma\.aligned`: the real mnemonics are
        #   mma.sync.aligned.m16n8k16...      and
        #   wgmma.mma_async.sync.aligned.m64n128k16...
        # so the shape is reached through a variable middle section.
        shapes.update(re.findall(
            r"(?:wgmma|mma\.sync)[a-z0-9_.]*?\.m\d+n\d+k\d+", text))
    return w, m, sorted(shapes)


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
    ap.add_argument("--dump-ptx", type=Path, default=None,
                    help="dump and count the emitted ISA per tile setting, so "
                         "the wgmma claim is verified rather than labelled")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs the GPU box")

    # Before find_override(), which imports vLLM: Triton reads these at compile
    # time and the first compile happens on the first fused_experts call.
    if args.dump_ptx:
        arm_ptx_dump(args.dump_ptx)

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
        head = f"  {'BLOCK_SIZE_M':>13} {'ms p50':>10} {'stdev':>8} {'vs first':>9}   "
        head += "EMITTED" if args.dump_ptx else "note"
        print(head)
        seen_ptx: set[Path] = set()
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
            if args.dump_ptx:
                # Each BLOCK_SIZE_M is a distinct Triton specialisation and so a
                # distinct cache entry, which is why files appearing after this
                # setting ran belong to it.
                w_n, m_n, shapes = scan_new_ptx(args.dump_ptx, seen_ptx)
                note = f"wgmma={w_n} mma.sync={m_n}"
                if shapes:
                    note += "  " + ",".join(shapes)
                elif w_n == 0 and m_n == 0:
                    note += "  (no new PTX; cache hit or dump not armed)"
            else:
                note = "mma.sync (M<64)" if bm < 64 else "wgmma reachable (M>=64)"
            print(f"  {bm:13d} {ms:10.4f} {sd:8.4f} {ms / base:8.3f}x   {note}")
        print()

    print("READING IT. Flat across all four confirms the thesis: neither the")
    print("padded arithmetic nor the tensor-core instruction is on the critical")
    print("path, because the weight read is. An improvement at 64 refutes it.")
    if args.dump_ptx:
        print(f"ISA counted per setting from {args.dump_ptx}. If wgmma stays 0 at")
        print("M>=64 then Triton is not reaching the warpgroup instruction even when")
        print("the tile allows it, which is a finding in its own right.")
    else:
        print("Re-run with --dump-ptx to verify the instruction actually changed;")
        print("without it this asserts nothing about what was emitted, only cost.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
