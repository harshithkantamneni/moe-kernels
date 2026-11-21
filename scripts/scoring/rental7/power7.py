#!/usr/bin/env python
"""Every registered rental-7 power point, re-run on the laptop GPU (torch MPS, float32) and checked
against the CPU in float64 (the registered numbers: register.py computes them on the CPU).

    python scripts/scoring/rental7/power7.py [--out docs/registered/2026-10-13-rental7-skew-hist/power7_check.json]

The normals are drawn on the CPU in float64 from the same seeded generator in both runs and moved
to the device, so the comparison is arithmetic only. Writes, per point, both results, the largest
absolute difference over every probability, and the wall seconds on each device.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reg7 as R7  # noqa: E402

OUT = R7.HIST / "power7_check.json"


def _run(device: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    rows = R7.skew_rows()
    e1, q1 = R7.q1_rows()
    H = R7.q1h_pairs(e1, q1)
    t_price = time.perf_counter() - t0
    t1 = time.perf_counter()
    out = {"E1": R7.e1_power_block(rows, device), "Q1-H": R7.q1h_power_block(H, device), "Q1-C": R7.q1c_power_block(device)}
    return out, time.perf_counter() - t1, t_price


def _diff(a, b) -> float:
    if isinstance(a, dict):
        return max([_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, float):
        return abs(a - b)
    return 0.0


def main(argv=None) -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if not torch.backends.mps.is_available():
        print("no MPS device: nothing to check")
        return 2
    R7.seed_pricer()
    cpu, t_cpu, t_price = _run("cpu")
    mps, t_mps, _ = _run("mps")
    # the registered numbers are the CPU ones; MPS float32 must agree
    doc = {"what": "rental-7 power points: MPS float32 against CPU float64 (the registered numbers)",
           "torch": torch.__version__, "R": R7.POWER_R, "max_abs_diff": _diff(cpu, mps),
           "seconds": {"cpu_float64": round(t_cpu, 2), "mps_float32": round(t_mps, 2), "pricing_cpu_first_call": round(t_price, 1)},
           "cpu_float64": cpu, "mps_float32": mps}
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out}: max |cpu - mps| {doc['max_abs_diff']:.3g}; cpu {t_cpu:.2f} s, mps {t_mps:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
