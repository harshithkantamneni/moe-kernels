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

#: CUTLASS 3.x carries the tile as cute::C<N> template parameters, and torch
#: hands us the MANGLED name, where those are Li<N>E. Demangling first is what
#: makes this readable; the fallbacks below cover a box without c++filt.
_MMA_ATOM = re.compile(r"MMA_(\d+)x(\d+)x(\d+)")
_TILE_TUPLE = re.compile(
    r"cute::tuple<\s*cute::C<(\d+)>\s*,\s*cute::C<(\d+)>\s*,\s*cute::C<(\d+)>")
#: Mangled form of tuple<C<a>, C<b>, ...>, for when demangling is unavailable.
_TILE_MANGLED = re.compile(r"ILi(\d+)EEEN\w\w?_ILi(\d+)EEE")


def demangle(name: str) -> str:
    """Best effort. Returns the input unchanged if nothing can demangle it."""
    if not name.startswith("_Z"):
        return name
    try:
        import subprocess
        out = subprocess.run(["c++filt", "-n", name], capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return name


def describe_tile(name: str) -> str:
    """What tile shape does this kernel name carry, if any.

    Order matters in the mangled form. `Li<N>E` pairs appear for the CLUSTER
    shape before the TILE shape, so taking the first match reports the cluster
    and calls it BLOCK_M. Anchor on the kernel-schedule token, which sits
    between them, and read the pair after it.
    """
    plain = demangle(name)
    bits = []

    m = _TILE_TUPLE.search(plain)
    if m:
        bits.append(f"TileShape M,N,K = {m.group(1)},{m.group(2)},{m.group(3)}")
    else:
        # Demangling unavailable or the tuple is spelled some other way.
        sched_at = max(plain.find("Pingpong"), plain.find("Cooperative"))
        tail = plain[sched_at:] if sched_at >= 0 else plain
        g = _TILE_MANGLED.search(tail)
        if g:
            bits.append(f"TileShape M,N = {g.group(1)},{g.group(2)}")

    # Literal in both the mangled and demangled name, so search the raw one.
    a = _MMA_ATOM.search(name) or _MMA_ATOM.search(plain)
    if a:
        bits.append(f"MMA atom = {a.group(1)}x{a.group(2)}x{a.group(3)}")

    for sched in ("Cooperative", "Pingpong"):
        if sched in plain:
            bits.append(f"schedule = {sched}")
    return "  |  ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v3", choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--tokens", default="1",
                    help="comma-separated. CUTLASS picks a tile per problem "
                         "shape, so one token count answers for one shape only")
    ap.add_argument("--impl", default="torch_grouped_mm_up")
    ap.add_argument("--routing", default="uniform")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; this has to run on the GPU box", file=sys.stderr)
        return 1

    moe.bootstrap("reference", "baselines")
    for tokens in (int(v) for v in str(args.tokens).split(",") if v.strip()):
        run_one(args, tokens)
    return 0


def run_one(args, tokens: int) -> None:
    spec = BenchSpec(MODEL_CONFIGS[args.model], num_tokens=tokens,
                     routing=RoutingSpec(args.routing))
    cfg = spec.model
    span = get_span(args.impl)
    pipe = build(tiling_for(span), spec=spec)

    print(f"\n[cell]   {spec.label}")
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

    # The GEMM answers the question; the rest of the tiling is the reference
    # path around it. Match on markers rather than on "cutlass" alone: vLLM and
    # SGLang dispatch Triton kernels, which carry none of CUTLASS's vocabulary.
    markers = ("cutlass", "fused_moe", "grouped_gemm", "gemm")
    gemms = [e for e in events
             if any(m in e.key.lower() for m in markers)
             and "prepare_grouped" not in e.key]
    if not gemms:
        print("  NO GEMM KERNEL MATCHED. Either the baseline fell back, or its")
        print("  kernel is named unlike any marker. Hottest kernels were:")
        for e in events[:5]:
            print(f"    {e.self_device_time_total:9.1f} us  {e.key[:110]}")
        return

    for e in gemms[:5]:
        print(f"  {e.self_device_time_total:9.1f} us  {e.key[:100]}")
        tile = describe_tile(e.key)
        if tile:
            print(f"  {'':9s}      {tile}")
        elif not e.key.startswith("_Z"):
            # A Triton kernel keeps its tile in a compile-time constexpr, not
            # in the symbol, so absence here is expected rather than a failure.
            print(f"  {'':9s}      no tile in the name (Triton keeps it in a constexpr)")
    top = [f"{e.self_device_time_total:.1f}us" for e in events[:3]]
    print(f"  (hottest kernels overall: {', '.join(top)})")


if __name__ == "__main__":
    sys.exit(main())
