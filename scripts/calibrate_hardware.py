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
from moe.bench.roofline import (  # noqa: E402
    ambiguous_for_device,
    for_device,
    load_hardware,
    power_limit_w,
)
from moe.bench.schema import git_provenance  # noqa: E402


def _device_facts() -> dict:
    """Device facts that settle what a percentage is a percentage OF.

    The memory clock is the important one. HBM is DDR, so the pin rate is
    clk x 2 x bus_width / 8, and the doubling is mandatory: omitting it gives
    exactly half, the same class of silent 2x error as the read-for-ownership
    question. On this H200, 3201 MHz gives 4916.7 GB/s, which is 2.4% above
    NVIDIA's published 4.8 TB/s -- so the datasheet figure is already derated
    and is not a theoretical maximum.
    """
    import torch

    from moe.bench.timing import _nvidia_smi

    out: dict = {}
    props = torch.cuda.get_device_properties(0)
    out["l2_bytes"] = getattr(props, "L2_cache_size", 0)
    out["sm_count"] = props.multi_processor_count

    vals = _nvidia_smi("clocks.max.memory,clocks.current.memory,"
                       "clocks_throttle_reasons.active")
    if vals:
        parts = [v.strip() for v in vals[0].split(",")]
        try:
            out["clocks_max_memory_mhz"] = float(parts[0].split()[0])
            out["clocks_current_memory_mhz"] = float(parts[1].split()[0])
            out["throttle_reasons"] = parts[2] if len(parts) > 2 else ""
        except (ValueError, IndexError):
            pass
    clk = out.get("clocks_max_memory_mhz")
    if clk:
        # H200 is 6144-bit HBM3e. DDR doubles the transfer rate.
        out["memory_bus_bits"] = 6144
        out["pin_rate_gbps"] = round(clk * 2 * 6144 / 8 / 1000, 1)
    return out


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
    ap.add_argument("--no-settle", dest="settle", action="store_false",
                    help="skip the clock settle. Only for a quick smoke check: "
                         "measuring from idle walks the clock ramp across the "
                         "patterns and the ceilings are not comparable")
    ap.add_argument("--settle-seconds", type=float, default=30.0)
    ap.add_argument("--compare-to", default=None,
                    help="datasheet profile to compare against; auto-detected "
                         "when the device name is unambiguous")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("calibration needs a GPU")

    print("[calibrate] measuring achievable bandwidth and dense BF16 ...")
    print(f"[calibrate] buffers {args.buffer_gb:g} GiB, L2 flushed between iterations")
    if args.settle:
        print(f"[calibrate] settling clocks under load (up to "
              f"{args.settle_seconds:g}s) before measuring anything")
    cal = calibrate(int(args.buffer_gb * (1 << 30)), args.gemm_n, args.ceiling,
                    settle=args.settle, settle_seconds=args.settle_seconds)

    props = torch.cuda.get_device_properties(0)
    print(f"\n  device            {cal.gpu_name}")
    l2_mib = getattr(props, "L2_cache_size", 0) / 2**20
    ratio = args.buffer_gb * 1024 / max(l2_mib, 1)
    print(f"  L2                {l2_mib:.0f} MiB   (buffers are {ratio:.0f}x larger)")

    # The board power limit is what actually tells an H200 SXM (700 W) from an
    # H200 NVL (600 W); torch reports both as "NVIDIA H200".
    tdp = power_limit_w()
    observed = _device_facts()
    if tdp:
        print(f"  power limit       {tdp:.0f} W"
              + ("   (SXM)" if tdp > 650 else "   (NVL)"))
    pin = observed.get("pin_rate_gbps")
    if pin:
        print(f"  memory clock      {observed['clocks_max_memory_mhz']:.0f} MHz"
              f"  -> pin rate {pin:.1f} GB/s")
        print("                    (clk x 2 for DDR x 6144 bits / 8; a "
              "datasheet figure below this is already derated)")

    spec_bw = spec_tf = None
    profile = args.compare_to
    if profile is None:
        profile = for_device(cal.gpu_name, tdp_w=tdp)
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

    # Never print a bare "% of peak": the same figure is 97.0% of NVIDIA's
    # published 4.8 TB/s and 94.7% of the real 4916.7 GB/s pin rate, and a
    # reader cannot tell which unless the denominator is in the string.
    print(f"\n  {'pattern':<8}{'GB/s':>10}{'of spec':>9}{'of pin':>9}"
          f"{'p50 ms':>10}   note")
    for pat in cal.bandwidth_patterns:
        of_spec = f"{100 * pat.gbps / spec_bw:>7.1f}%" if spec_bw else f"{'':>8}"
        of_pin = f"{100 * pat.gbps / pin:>7.1f}%" if pin else f"{'':>8}"
        mark = " <-- ceiling" if pat.pattern == cal.ceiling_pattern else ""
        print(f"  {pat.pattern:<8}{pat.gbps:>10.1f}{of_spec}{of_pin}"
              f"{pat.ms_p50:>10.3f}   {pat.note}{mark}")
    if spec_bw and pin:
        print(f"  {'':8}{'':10}{'^ vs':>9}{'^ vs':>9}")
        print(f"  {'':8}{'':10}{spec_bw:>8.0f} {pin:>8.0f}  GB/s "
              "(published, derived)")

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
        print(f"    {pat.pattern:<8}{cal.ridge_point(pat.gbps):>8.0f}"
              f"  ({pat.pattern}){mark}")
    if spec_bw and spec_tf:
        print(f"    {'datasheet':<8}{spec_tf * 1e12 / (spec_bw * 1e9):>8.0f}")

    st = cal.settle
    if st.get("skipped"):
        print("\n  settle            SKIPPED: ceilings may be mid-ramp")
    else:
        hist = st.get("clock_history_mhz") or []
        print(f"\n  settle            "
              f"{'reached' if st.get('settled') else 'TIMED OUT'} at "
              f"{st.get('final_mhz', 0)} MHz   history {hist}")
        if not st.get("settled"):
            print("                    clocks still moving when the budget ran "
                  "out; raise --settle-seconds")

    if cal.warmup_pass:
        print(f"\n  warm-up pass      discarded; largest per-pattern change "
              f"{cal.warmup_drift_pct:.1f}%")
        if cal.warmup_drift_pct > 3.0:
            print("                    >3% means the settle did not reach the "
                  "state the measurement induces; raise --settle-seconds")

    if cal.clock_ramped:
        print("\n  WARNING: SM clock differed by >5% ACROSS the patterns, so "
              "they were measured in different states and are NOT comparable:")
        for pat in cal.bandwidth_patterns:
            print(f"    {pat.pattern:<8}{pat.sm_clock_start_mhz:>6} -> "
                  f"{pat.sm_clock_end_mhz:<6} MHz")

    c = cal.clocks
    print(f"\n  clocks            {c['sm_start_mhz']} -> {c['sm_end_mhz']} MHz, "
          f"{c['temp_start_c']} -> {c['temp_end_c']} C"
          + ("  THROTTLED" if c["throttled"] else ""))
    if c["throttled"]:
        print("                    ceilings measured on a throttling GPU are low; "
              "let it cool and re-run")

    # A write figure at or above datasheet peak means the byte accounting is
    # wrong (a read-for-ownership would make real traffic 2N), not that the
    # hardware exceeded its specification.
    if pin:
        w = cal.pattern("write")
        if w and w.gbps > pin * 0.93:
            print(f"\n  NOTE: write is {100 * w.gbps / pin:.1f}% of the derived "
                  "pin rate. That is a REAL store rate, not an accounting error: "
                  "a read-for-ownership would make it 2N, which would exceed the "
                  "pin rate by ~94% and is impossible. Write is not the ceiling "
                  "because it is the least representative pattern for a "
                  "read-dominated workload.")

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
        "observed": {**observed, "power_limit_w": tdp},
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
