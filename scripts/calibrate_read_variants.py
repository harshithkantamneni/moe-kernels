#!/usr/bin/env python
"""C4: is our read ceiling a ceiling, or just what one formulation achieved?

    python scripts/calibrate_read_variants.py
    python scripts/calibrate_read_variants.py --gib 8 --iters 50

THE PROBLEM. `calibrate_hardware.py` measures the `read` pattern with
`torch.sum(a, dim=0, out=sink)`, and its own docstring already flags this: "read
here is a torch.sum tree reduction". A reduction is not a pure streaming read.
It combines partial results across blocks, which costs traffic and
synchronisation that a weight stream does not pay.

That matters because 83 rows of the 2026-08-26 sweep imply a bandwidth ABOVE
that ceiling. Peak 4483.4 GB/s against a measured read of 4389.4. Nothing
exceeded the 4916.7 GB/s pin rate, so nothing impossible happened; the likely
reading is that the ceiling is too low, in which case every percent-of-ceiling
figure in the study is pessimistic by that margin, and so is anyone else's
computed the same way.

WHAT THIS DECIDES. If some formulation beats 4483.4, C4 is confirmed and the
83-row anomaly dissolves into a calibration artifact. If none does, the anomaly
survives and needs another explanation.

Every variant below reads the same buffer exactly once per iteration. They differ
only in what they do with the values, which is the part that must be cheap enough
not to matter and expensive enough that the compiler cannot delete the load.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from moe.bench.timing import L2Flusher, require_cuda  # noqa: E402

#: What the 2026-08-26 sweep's fastest row implied, in GB/s. A variant beating
#: this confirms C4.
ANOMALY_GBPS = 4483.4
#: What calibrate_hardware.py currently reports for `read` on that card.
CURRENT_READ_GBPS = 4389.4
#: 6144-bit bus at 3201 MHz. Nothing can exceed this.
PIN_RATE_GBPS = 4916.7


def time_it(fn, nbytes: int, warmup: int, iters: int, flusher) -> float:
    """Median GB/s over `iters`, L2 flushed before each timed call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if flusher is not None:
            flusher.flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        samples.append(nbytes / (ms * 1e-3) / 1e9)
    samples.sort()
    return samples[len(samples) // 2]


def variants(a: torch.Tensor, sink_row: torch.Tensor, sink_scalar: torch.Tensor):
    """Formulations that each read every element of `a` exactly once.

    None of these is a clever kernel. They are the obvious ways to express
    "touch all of it" in torch, and the point is how far apart they land.
    """
    return [
        ("torch.sum(dim=0)  [what calibrate.py uses]",
         lambda: torch.sum(a, dim=0, out=sink_row)),
        ("torch.sum(dim=1)  [reduces along the contiguous axis]",
         lambda: torch.sum(a, dim=1, out=sink_scalar)),
        ("a.sum()           [full reduction to one scalar]",
         lambda: torch.sum(a)),
        ("a.amax()          [reduction with no accumulation width]",
         lambda: torch.amax(a)),
        ("a.count_nonzero() [integer reduction]",
         lambda: torch.count_nonzero(a)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gib", type=float, default=8.0,
                    help="buffer size; must be far larger than L2 (default 8)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-flush", action="store_true")
    args = ap.parse_args()

    require_cuda()
    nbytes = int(args.gib * (1 << 30))
    rows = 4096
    cols = nbytes // (rows * 4)
    a = torch.empty((rows, cols), dtype=torch.float32, device="cuda")
    a.uniform_(0.0, 1.0)
    real_bytes = a.numel() * a.element_size()
    sink_row = torch.empty(cols, dtype=torch.float32, device="cuda")
    sink_scalar = torch.empty(rows, dtype=torch.float32, device="cuda")

    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", 0)
    print(f"device       {props.name}")
    print(f"buffer       {real_bytes / (1 << 30):.2f} GiB "
          f"({real_bytes / max(l2, 1):.0f}x L2)")
    print(f"flush        {'off' if args.no_flush else 'on'}   "
          f"iters {args.iters}, warmup {args.warmup}\n")

    flusher = None if args.no_flush else L2Flusher(device="cuda")

    print(f"{'variant':52} {'GB/s':>9} {'vs current':>11} {'vs anomaly':>11}")
    print("-" * 86)
    best_name, best = "", 0.0
    for name, fn in variants(a, sink_row, sink_scalar):
        gbps = time_it(fn, real_bytes, args.warmup, args.iters, flusher)
        if gbps > best:
            best_name, best = name, gbps
        print(f"{name:52} {gbps:9.1f} {gbps / CURRENT_READ_GBPS:10.3f}x "
              f"{gbps / ANOMALY_GBPS:10.3f}x")

    print("\n" + "=" * 86)
    print(f"best: {best_name.strip()}  at {best:.1f} GB/s")
    print(f"  calibrate.py currently reports  {CURRENT_READ_GBPS:.1f} GB/s")
    print(f"  the sweep's fastest row implied {ANOMALY_GBPS:.1f} GB/s")
    print(f"  bus pin rate                    {PIN_RATE_GBPS:.1f} GB/s")
    print()
    if best > PIN_RATE_GBPS:
        print("  MEASUREMENT IS BROKEN: nothing can exceed the pin rate. Check the")
        print("  byte count and that the compiler did not delete the read.")
    elif best > ANOMALY_GBPS:
        print("  -> C4 CONFIRMED. A plain formulation beats what the sweep's fastest")
        print("     row implied, so the 'ceiling' was one formulation's achieved rate.")
        print("     The 83 anomalous rows are a calibration artifact, and every")
        print("     percent-of-ceiling figure in the study is pessimistic by up to")
        print(f"     {100 * (best / CURRENT_READ_GBPS - 1):.1f}%.")
    elif best > CURRENT_READ_GBPS * 1.01:
        print("  -> PARTIAL. The ceiling is low by a real margin but not enough to")
        print("     explain the anomaly on its own. Both effects are in play.")
    else:
        print("  -> C4 NOT SUPPORTED. No formulation beats the current figure, so")
        print("     the 83 rows need a different explanation. Next suspect is the")
        print("     compulsory byte model for vLLM's span at 1-2 rows per expert.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
