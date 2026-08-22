#!/usr/bin/env python
"""Measure this machine's achievable ceilings. Run once per pod type.

Nsight Compute needs GPU performance counters, which need a host-level module
flag a container tenant cannot set; on a rented pod `ncu` fails with
ERR_NVGPUCTRPERM. So DRAM traffic cannot be read directly, and the roofline
would otherwise rest entirely on a datasheet peak.

This measures the ceilings with ordinary kernels and a clock instead, and writes
them beside the cited spec file. Efficiency can then be quoted against what the
machine actually delivers, which is both fairer to your kernel and more
defensible in public.

    python scripts/calibrate_hardware.py --out moe/bench/hardware/measured.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402

from moe.bench.calibrate import calibrate  # noqa: E402
from moe.bench.roofline import load_hardware  # noqa: E402
from moe.bench.schema import git_provenance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path("moe/bench/hardware/measured.yaml"))
    ap.add_argument("--buffer-gb", type=float, default=2.0)
    ap.add_argument("--gemm-n", type=int, default=8192)
    ap.add_argument("--compare-to", default="h200_nvl")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("calibration needs a GPU")

    print("[calibrate] measuring achievable bandwidth and dense BF16 ...")
    cal = calibrate(int(args.buffer_gb * (1 << 30)), args.gemm_n)

    print(f"\n  device            {cal.gpu_name}")
    for p in cal.bandwidth_patterns:
        print(f"  {p.pattern:<17} {p.gbps:8.1f} GB/s   "
              f"({p.bytes_moved / 1e9:.2f} GB in {p.ms_p50:.3f} ms)")
    print(f"  achieved BW       {cal.achieved_bandwidth_gbps:8.1f} GB/s")
    print(f"  achieved BF16     {cal.achieved_bf16_tflops:8.1f} TFLOP/s "
          f"(GEMM {args.gemm_n}^3)")

    spec_bw = spec_tf = None
    try:
        hw = load_hardware(args.compare_to)
        spec_bw = hw.bandwidth_bytes_s / 1e9
        spec_tf = hw.peak("bf16") / 1e12
        print(f"\n  vs {hw.name} datasheet:")
        print(f"    bandwidth       {100 * cal.achieved_bandwidth_gbps / spec_bw:5.1f}% "
              f"of {spec_bw:.0f} GB/s")
        print(f"    bf16 compute    {100 * cal.achieved_bf16_tflops / spec_tf:5.1f}% "
              f"of {spec_tf:.1f} TFLOP/s")
        measured_ridge = (cal.achieved_bf16_tflops * 1e12) / (
            cal.achieved_bandwidth_gbps * 1e9)
        print(f"    ridge point     {measured_ridge:.0f} FLOP/byte measured, "
              f"{hw.ridge_point('bf16'):.0f} from spec")
    except Exception as e:  # noqa: BLE001
        print(f"\n  (no spec comparison: {e})")

    # git_provenance runs with -C <repo root>, so it records the right SHA even
    # when the script is invoked from elsewhere, and it will not hang.
    sha, dirty = git_provenance()
    payload = {
        "name": f"{cal.gpu_name} (measured)",
        "verified": True,
        "source": "measured on this machine by scripts/calibrate_hardware.py",
        "source_note": (
            "Achievable ceilings, not datasheet peaks. Nsight Compute is "
            "unavailable on a rented pod (ERR_NVGPUCTRPERM), so DRAM traffic "
            "cannot be measured directly; these ceilings are what make the "
            "efficiency columns meaningful without counters."),
        "checked_by": "scripts/calibrate_hardware.py",
        "checked_on": time.strftime("%Y-%m-%d"),
        "measured_commit": sha,
        "memory": {"bandwidth_tb_s": cal.achieved_bandwidth_gbps / 1000.0},
        "compute_dense_tflops": {"bf16": cal.achieved_bf16_tflops,
                                 "fp16": cal.achieved_bf16_tflops},
        "detail": cal.as_dict(),
        "spec_comparison": {"bandwidth_gbps": spec_bw, "bf16_tflops": spec_tf},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"\n[calibrate] wrote {args.out}")
    print("  commit it, then plots and efficiency columns use measured ceilings")
    print(json.dumps({"achieved_bw_gbps": cal.achieved_bandwidth_gbps,
                      "achieved_bf16_tflops": cal.achieved_bf16_tflops}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
