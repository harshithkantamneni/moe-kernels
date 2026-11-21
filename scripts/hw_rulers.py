#!/usr/bin/env python
"""THE RULERS STEP (rental 4, reanalysis/REANALYSIS.md section 4): existing torch ops at the
lock, plus the device's own L2 size, so eta_mix = bw / ruler reads off this board.

    python scripts/hw_rulers.py --out DIR [--lock-mhz 1710] [--gib 4] [--dry-run]

No kernel source: every ruler is a torch op, timed by moe.bench.timing.time_kernel (its
warmup, trials, flush and clock), the instrument calibrate and C4 use.

  read2d   torch.sum(a, dim=1) over a [4096, cols] fp32 view (FINDINGS C4's read: a
           reduction along the contiguous axis, no global combine), bytes = the buffer
  copy     y.copy_(x), bytes = 2 x the buffer
  add      torch.add(a, b, out=c), a triad-like three streams, bytes = 3 x one buffer
  matmul   torch.matmul bf16 8192^3 (cuBLAS), FLOP = 2 n^3: the tensor-core ceiling the
           design's R2 names (an upper bound, not mma.sync's own)
  device   torch.cuda.get_device_properties(0): name, SMs, L2_cache_size, memory

Writes DIR/rulers.json. Exit 0 every ruler ran; 2 refused (no CUDA; and --dry-run); 3 a
ruler failed (the rest are written).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from moe.bench import exit_codes  # noqa: E402

RULERS = ("read2d", "copy", "add", "matmul")
MATMUL_N = 8192
ROWS = 4096


def plan(gib: float) -> dict:
    nbytes = int(gib * (1 << 30))
    cols = nbytes // (ROWS * 4)
    one = ROWS * cols * 4
    return {"read2d": {"bytes": one, "what": f"torch.sum(a, dim=1), a fp32 [{ROWS}, {cols}]"},
            "copy": {"bytes": 2 * one, "what": "y.copy_(x), fp32, each the buffer"},
            "add": {"bytes": 3 * one, "what": "torch.add(a, b, out=c), fp32, each the buffer"},
            "matmul": {"flop": 2 * MATMUL_N ** 3, "what": f"torch.matmul bf16 {MATMUL_N}^3"},
            "cols": cols, "buffer_bytes": one}


def rates(timings_ms: dict, p: dict) -> dict:
    """GB/s (TFLOP/s for matmul) from each ruler's median ms."""
    out = {}
    for k, ms in timings_ms.items():
        if ms is None:
            out[k] = None
        elif k == "matmul":
            out[k] = {"tflops": p[k]["flop"] / (ms * 1e-3) / 1e12, "ms": ms}
        else:
            out[k] = {"gbps": p[k]["bytes"] / (ms * 1e-3) / 1e9, "ms": ms}
    return out


def run(args, p: dict) -> int:
    import torch

    from moe.bench import timing
    props = torch.cuda.get_device_properties(0)
    dev = {"name": props.name, "sms": props.multi_processor_count,
           "l2_cache_size": int(getattr(props, "L2_cache_size", 0) or 0),
           "total_memory": int(props.total_memory),
           "capability": f"{props.major}.{props.minor}"}
    cols = p["cols"]
    a = torch.empty((ROWS, cols), dtype=torch.float32, device="cuda").uniform_()
    b = torch.empty_like(a).uniform_()
    c = torch.empty_like(a)
    sink = torch.empty(ROWS, dtype=torch.float32, device="cuda")
    m1 = torch.randn(MATMUL_N, MATMUL_N, dtype=torch.bfloat16, device="cuda")
    m2 = torch.randn(MATMUL_N, MATMUL_N, dtype=torch.bfloat16, device="cuda")
    fns = {"read2d": lambda: torch.sum(a, dim=1, out=sink), "copy": lambda: c.copy_(a),
           "add": lambda: torch.add(a, b, out=c), "matmul": lambda: torch.matmul(m1, m2)}
    ms, detail, bad = {}, {}, 0
    for k in RULERS:
        try:
            kt = timing.time_kernel(fns[k], warmup_ms=200, target_ms=200, trials=5,
                                    l2_flush=k != "matmul", reference_clock_mhz=args.lock_mhz)
            ms[k] = float(kt.ms_p50)
            detail[k] = {"ms_p50": kt.ms_p50, "ms_min": kt.ms_min, "ms_std": kt.ms_std,
                         "sm_clock_load_mhz": kt.sm_clock_load_mhz, "iters": kt.iters}
        except Exception as exc:                                  # noqa: BLE001
            ms[k] = None
            detail[k] = {"error": f"{type(exc).__name__}: {exc}"}
            bad += 1
    out = {"tool": "scripts/hw_rulers.py", "lock_mhz": args.lock_mhz, "device": dev, "plan": p,
           "timings": detail, "rates": rates(ms, p), "ended": time.time()}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "rulers.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"device": dev, "rates": out["rates"]}, default=str))
    return exit_codes.INVALID if bad else exit_codes.DONE


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lock-mhz", type=float, default=None)
    ap.add_argument("--gib", type=float, default=4.0, help="each streaming buffer (GiB)")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    p = plan(args.gib)
    print(f"RULERS at lock {args.lock_mhz} MHz, buffers {p['buffer_bytes'] / (1 << 30):.2f} GiB:")
    for k in RULERS:
        print(f"  {k:7s} {p[k]['what']}")
    if args.dry_run:
        print("DRY RUN: nothing timed (exit 2)")
        return exit_codes.REFUSED
    try:
        import torch
        ok = torch.cuda.is_available()
    except ImportError:
        ok = False
    if not ok:
        print("REFUSED: no CUDA device")
        return exit_codes.REFUSED
    return run(args, p)


if __name__ == "__main__":
    sys.exit(main())
