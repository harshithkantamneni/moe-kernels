#!/usr/bin/env python
"""Which kernel does the baseline dispatch to, and what tile does it use?

This is Q2 of the two questions the 2026-08-22 sweep could not answer, and it
needs no profiler permissions at all. CUTLASS encodes its tile shape in the
kernel name, torch's own profiler records CUDA kernel names, and the recording
runs inside the same process that dispatched the kernel. So a rented pod where
`ncu` returns ERR_NVGPUCTRPERM can still settle it.

    python scripts/kernel_name.py                    # deepseek-v3, T=1, uniform
    python scripts/kernel_name.py --tokens 4096
    python scripts/kernel_name.py --impl torch_grouped_mm_down

T=1 is the default because there every active expert holds exactly one row, so
M-tiles and active experts are the same number and nothing else is moving. The
BLOCK_M=128 claim in FINDINGS section 2 was inferred from timing alone; a tile
shape in the kernel name confirms or kills it.

Why not nsys: an external tracer has to agree with you about its own command
line, and the version a pod happens to ship may not. Ubuntu's nsight-systems
2022.4 rejects `--` as an end-of-options separator. This has no such surface.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.profiler import ProfilerActivity, profile

import moe
from moe.bench.profiles import tiling_for
from moe.pipeline import build
from moe.reference.torch_ref import make_inputs
from moe.routing.distributions import sample_topk_ids
from moe.spec import MODEL_CONFIGS, BenchSpec, RoutingSpec
from moe.stages import get as get_span
from moe.state import MoEState

#: CUTLASS spells its tile as MxNxK somewhere in the mangled name.
_TILE = re.compile(r"\b(\d{2,4})x(\d{2,4})x(\d{1,4})\b")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v3", choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--impl", default="torch_grouped_mm_up")
    ap.add_argument("--routing", default="uniform")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; this has to run on the GPU box", file=sys.stderr)
        return 1

    moe.bootstrap("reference", "baselines")
    spec = BenchSpec(MODEL_CONFIGS[args.model], num_tokens=args.tokens,
                     routing=RoutingSpec(args.routing))
    cfg = spec.model
    span = get_span(args.impl)
    pipe = build(tiling_for(span), spec=spec)

    print(f"[cell]   {spec.label}")
    print(f"[impl]   {span.name} covers={'+'.join(span.covers)}")
    print(f"[torch]  {torch.__version__}  device={torch.cuda.get_device_properties(0).name}")

    x, weights = make_inputs(spec, device="cuda")
    forced = sample_topk_ids(spec.routing, spec.num_tokens, cfg.num_experts,
                             cfg.top_k, seed=spec.seed, device="cuda")
    st = MoEState(spec=spec, weights=weights, x=x)
    st.forced_topk_ids = forced

    # Warm first: the profiled pass should not be capturing autotune or the
    # lazy allocations of a first call.
    with torch.no_grad():
        pipe.run(st)
    torch.cuda.synchronize()

    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        pipe.run(st)
        torch.cuda.synchronize()

    events = [e for e in prof.key_averages()
              if getattr(e, "device_type", None) is not None
              and e.self_device_time_total > 0]
    events.sort(key=lambda e: -e.self_device_time_total)

    print(f"\n{'us':>10}  kernel")
    print("-" * 100)
    for e in events:
        print(f"{e.self_device_time_total:10.1f}  {e.key}")

    print("\n[tiles] shapes found in kernel names, hottest first:")
    hits = [(e.key, _TILE.findall(e.key)) for e in events]
    found = False
    for key, tiles in hits:
        for m, n, k in tiles:
            found = True
            print(f"  {m}x{n}x{k}   in  {key[:80]}")
    if not found:
        print("  none. The name may not carry a tile; check the full names above.")
    print("\nBLOCK_M is the M of the tile. FINDINGS section 2 infers 128 from timing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
