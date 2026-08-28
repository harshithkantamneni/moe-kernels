#!/usr/bin/env python
"""Re-derive a published sweep's ceiling columns against a new calibration.

    python scripts/recompute_ceilings.py \
        --arm results/published/2026-08-26-nvidia_h200-full-three-way \
        --calibration moe/bench/hardware/measured_nvidia_h200.yaml

WHY, INSTEAD OF RE-SWEEPING. The calibration determines four columns and nothing
else (driver.py:262-289). `ms_p50`, `tflops`, `compulsory_gbps` and
`arith_intensity_compulsory` come from the measured time and the byte model. So
when calibrate.py was found to settle under a matmul and then measure bandwidth,
and to name a tree reduction as its read ceiling, the correction is four columns
wide, not three hours long. The timings were never wrong.

Writes a NEW arm rather than editing in place. The original is the record of
what was measured against the ruler of the day, and overwriting it would erase
the evidence that the ruler moved.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench.recompute import (  # noqa: E402
    load_calibration_hardware,
    rewrite_csv,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", type=Path, required=True,
                    help="published results directory to re-derive")
    ap.add_argument("--calibration", type=Path, required=True,
                    help="the NEW calibration yaml")
    ap.add_argument("--out", type=Path, default=None,
                    help="destination arm (default: <arm>-recalibrated)")
    args = ap.parse_args()

    if not args.arm.is_dir():
        raise SystemExit(f"no such arm: {args.arm}")
    hw = load_calibration_hardware(args.calibration)
    out = args.out or args.arm.parent / (args.arm.name + "-recalibrated")
    out.mkdir(parents=True, exist_ok=True)

    print(f"calibration : {hw.name}")
    print(f"  bandwidth : {hw.bandwidth_bytes_s / 1e9:.1f} GB/s "
          f"(pattern {hw.ceiling_pattern or 'unnamed'})")
    for dt, fl in sorted(hw.peak_flops.items()):
        print(f"  peak {dt:9}: {fl / 1e12:.1f} TFLOP/s")
    print(f"source arm  : {args.arm}")
    print(f"destination : {out}\n")

    csvs = sorted(args.arm.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs under {args.arm}")
    for src in csvs:
        info = rewrite_csv(src, out / src.name, hw)
        moved = {k: v for k, v in info["changed"].items() if v}
        print(f"  {src.name:44} {info['rows']:6d} rows  "
              + (", ".join(f"{k}:{v}" for k, v in moved.items()) or "unchanged"))

    # Everything that is not a CSV describes the run and travels with it.
    for extra in args.arm.iterdir():
        if extra.suffix != ".csv" and extra.is_file():
            shutil.copy2(extra, out / extra.name)
    shutil.copy2(args.calibration, out / "measured.yaml")
    print(f"\nwrote {out}")
    print("The original arm is untouched: it is what was measured against the")
    print("ruler of the day, and the pair is the evidence that the ruler moved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
