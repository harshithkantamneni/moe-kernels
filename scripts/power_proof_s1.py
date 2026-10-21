#!/usr/bin/env python
"""S1-M0, the power proof (docs/registered/2026-10-21-settleS1-gh200): a sustained dense load while the driver logs
nvidia-smi's power.draw, enforced.power.limit and clocks_event_reasons every 100 ms, so the draw itself shows which
GPU power limit held (A2 read 700 W from `nvidia-smi -q` and 900 W from power.limit for its whole session).

    python scripts/power_proof_s1.py --seconds 90 --out gemm.json [--dry-run]

The load is torch.matmul on two 8192 x 8192 bf16 operands, back to back, synchronised every 20 products; no clock
lock (the driver resets clocks first). It writes, per synchronised batch, the wall time and the achieved TFLOP/s, so
the reader can line the batches up with the nvidia-smi samples. It measures nothing itself and scores nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

N = 8192
BATCH = 20


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.seconds <= 0 or a.seconds > 600:
        print(f"REFUSED: --seconds {a.seconds}: 0 to 600")
        return 2
    print(f"POWER PROOF: bf16 matmul {N}^3, batches of {BATCH}, for {a.seconds:g} s; out {a.out}")
    if a.dry_run:
        return 2
    import torch
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device")
        return 2
    x = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
    z = torch.empty(N, N, device="cuda", dtype=torch.bfloat16)
    torch.matmul(x, y, out=z)
    torch.cuda.synchronize()
    flop = 2.0 * N ** 3 * BATCH
    batches, t_start = [], time.time()
    while time.time() - t_start < a.seconds:
        t0 = time.time()
        for _ in range(BATCH):
            torch.matmul(x, y, out=z)
        torch.cuda.synchronize()
        t1 = time.time()
        batches.append({"t0_unix": round(t0, 4), "t1_unix": round(t1, 4), "tflops": round(flop / (t1 - t0) / 1e12, 2)})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"n": N, "batch": BATCH, "seconds": a.seconds, "device": torch.cuda.get_device_name(),
                                 "batches": batches}, indent=1) + "\n")
    tf = sorted(b["tflops"] for b in batches)
    print(f"RESULT: {len(batches)} batches, median {tf[len(tf) // 2]} TFLOP/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
