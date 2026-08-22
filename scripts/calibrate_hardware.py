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

from moe.bench.calibrate import DEFAULT_CEILING, calibrate  # noqa: E402
from moe.bench.roofline import ambiguous_for_device, for_device, load_hardware  # noqa: E402
from moe.bench.schema import git_provenance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path("moe/bench/hardware/measured.yaml"))
    ap.add_argument("--buffer-gb", type=float, default=8.0,
                    help="STREAM buffer size; must dwarf L2 and run long enough "
                         "that launch and clock ramp do not matter")
    ap.add_argument("--gemm-n", type=int, default=8192)
    ap.add_argument("--ceiling", default=DEFAULT_CEILING,
                    choices=("read", "copy", "triad", "write"),
                    help="which pattern defines achieved bandwidth. triad is the "
                         "canonical STREAM metric; read is closest to streaming "
                         "expert weights at small batch")
    ap.add_argument("--compare-to", default=None,
                    help="datasheet profile to compare against; auto-detected "
                         "when the device name is unambiguous")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("calibration needs a GPU")

    print("[calibrate] measuring achievable bandwidth and dense BF16 ...")
    print(f"[calibrate] buffers {args.buffer_gb:g} GiB, L2 flushed between iterations")
    cal = calibrate(int(args.buffer_gb * (1 << 30)), args.gemm_n, args.ceiling)

    props = torch.cuda.get_device_properties(0)
    print(f"\n  device            {cal.gpu_name}")
    l2_mib = getattr(props, "L2_cache_size", 0) / 2**20
    ratio = args.buffer_gb * 1024 / max(l2_mib, 1)
    print(f"  L2                {l2_mib:.0f} MiB   (buffers are {ratio:.0f}x larger)")

    spec_bw = spec_tf = None
    profile = args.compare_to
    if profile is None:
        profile = for_device(cal.gpu_name)
        if profile is None:
            tied = ambiguous_for_device(cal.gpu_name)
            if tied:
                print(f"\n  device name is AMBIGUOUS between {tied}; pass "
                      "--compare-to to pick one. Measured values below stand "
                      "on their own regardless.")
    if profile:
        try:
            hw = load_hardware(profile)
            spec_bw, spec_tf = hw.bandwidth_bytes_s / 1e9, hw.peak("bf16") / 1e12
        except Exception as e:  # noqa: BLE001
            print(f"  (no spec comparison: {e})")

    print(f"\n  {'pattern':<8}{'GB/s':>10}{'of peak':>10}{'p50 ms':>10}"
          f"{'min ms':>9}   note")
    for pat in cal.bandwidth_patterns:
        pct = f"{100 * pat.gbps / spec_bw:>8.1f}%" if spec_bw else f"{'':>9}"
        mark = " <-- ceiling" if pat.pattern == cal.ceiling_pattern else ""
        print(f"  {pat.pattern:<8}{pat.gbps:>10.1f}{pct}{pat.ms_p50:>10.3f}"
              f"{pat.ms_min:>9.3f}   {pat.note}{mark}")

    print(f"\n  achieved BW       {cal.achieved_bandwidth_gbps:8.1f} GB/s "
          f"(pattern: {cal.ceiling_pattern})")
    print(f"  achieved BF16     {cal.achieved_bf16_tflops:8.1f} TFLOP/s "
          f"(GEMM {args.gemm_n}^3)")
    if spec_tf:
        print(f"                    {100 * cal.achieved_bf16_tflops / spec_tf:8.1f}% "
              f"of {spec_tf:.1f} datasheet")

    print("\n  ridge point by choice of denominator (FLOP/byte):")
    for pat in cal.bandwidth_patterns:
        mark = " <-- used" if pat.pattern == cal.ceiling_pattern else ""
        print(f"    {pat.pattern:<8}{cal.ridge_point(pat.gbps):>8.0f}{mark}")
    if spec_bw and spec_tf:
        print(f"    {'datasheet':<8}{spec_tf * 1e12 / (spec_bw * 1e9):>8.0f}")

    c = cal.clocks
    print(f"\n  clocks            {c['sm_start_mhz']} -> {c['sm_end_mhz']} MHz, "
          f"{c['temp_start_c']} -> {c['temp_end_c']} C, drift {c['drift_pct']}%"
          + ("  THROTTLED" if c["throttled"] else ""))
    if c["throttled"]:
        print("                    ceilings measured on a throttling GPU are low; "
              "let it cool and re-run")

    # A write figure at or above datasheet peak means the byte accounting is
    # wrong (a read-for-ownership would make real traffic 2N), not that the
    # hardware exceeded its specification.
    if spec_bw:
        w = cal.pattern("write")
        if w and w.gbps > spec_bw * 0.95:
            print(f"\n  NOTE: write measured at {100 * w.gbps / spec_bw:.0f}% of "
                  "datasheet peak. Either writes really are that efficient, or "
                  "the 1N accounting misses a read-for-ownership. This is why "
                  "write is not the default ceiling.")

    sha, dirty = git_provenance()
    payload = {
        "name": f"{cal.gpu_name} (measured)",
        "verified": True,
        "source": "measured on this machine by scripts/calibrate_hardware.py",
        "source_note": (
            "Achievable ceilings, not datasheet peaks. Nsight Compute is "
            "unavailable on a rented pod (ERR_NVGPUCTRPERM), so DRAM traffic "
            "cannot be measured directly; these ceilings are what make the "
            f"efficiency columns meaningful without counters. Bandwidth is the "
            f"'{cal.ceiling_pattern}' pattern; every pattern measured is under "
            "`detail` so a different denominator can be applied without "
            "re-running."),
        "checked_by": "scripts/calibrate_hardware.py",
        "checked_on": time.strftime("%Y-%m-%d"),
        "measured_commit": sha,
        "measured_dirty": dirty,
        "memory": {"bandwidth_tb_s": cal.achieved_bandwidth_gbps / 1000.0},
        "compute_dense_tflops": {"bf16": cal.achieved_bf16_tflops,
                                 "fp16": cal.achieved_bf16_tflops},
        "detail": cal.as_dict(),
        "spec_comparison": {"profile": profile, "bandwidth_gbps": spec_bw,
                            "bf16_tflops": spec_tf},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"\n[calibrate] wrote {args.out}")
    print("  commit it; plots and efficiency columns then use measured ceilings")
    print(json.dumps({"achieved_bw_gbps": cal.achieved_bandwidth_gbps,
                      "ceiling_pattern": cal.ceiling_pattern,
                      "achieved_bf16_tflops": cal.achieved_bf16_tflops,
                      "ridge": round(cal.ridge_point(), 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
