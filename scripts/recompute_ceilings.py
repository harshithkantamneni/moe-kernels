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

AND IT DECLARES ITSELF. The arm this produces is the one legitimate case of a
published result whose calibration comes from a later session than its rows --
that is the whole point of it -- and `publish_results.sh` now refuses exactly
that shape, because an undeclared instance of it cost claim C5 its target. So a
`DERIVED_FROM` marker goes in beside the rows, the same way `SUPERSEDED` does,
and `moe.bench.published.derived_from` reads it. The README keeps saying so too:
prose for a human, a marker for the check, and the marker is authoritative
because a README gets hand-edited and this one already has been.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench.published import DERIVED_MARKER  # noqa: E402
from moe.bench.recompute import (  # noqa: E402
    CEILING_COLUMNS,
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

    # Only the manifests travel. FINDINGS.md is a human analysis of the arm it
    # was written for and quotes ratios against the ruler of that day; copying
    # it here puts prose that disagrees with the rows right beside them.
    # SUMMARY.md is generated and would be equally stale.
    for extra in args.arm.iterdir():
        if extra.is_file() and extra.name.endswith(".manifest.jsonl"):
            shutil.copy2(extra, out / extra.name)
    shutil.copy2(args.calibration, out / "measured.yaml")

    # The declaration the publish gate reads. First line is the source arm, so
    # `derived_from` can name it without parsing prose.
    (out / DERIVED_MARKER).write_text(
        f"{args.arm.name}\n"
        f"recomputed by scripts/recompute_ceilings.py against "
        f"{args.calibration}\n"
        f"Its measured.yaml is deliberately from a different session than its\n"
        f"rows: only the ceiling columns were re-derived, and the timings are\n"
        f"the source arm's untouched.\n")

    (out / "README.md").write_text(
        f"# {out.name}\n\n"
        f"Derived from `{args.arm.name}` by `scripts/recompute_ceilings.py`.\n"
        f"The measurements are identical: `ms_p50`, `tflops`, `compulsory_gbps`\n"
        f"and `arith_intensity_compulsory` come from the timing and the byte\n"
        f"model and were never affected by the calibration. Only the four\n"
        f"calibration-derived columns differ:\n\n"
        + "".join(f"  - `{c}`\n" for c in CEILING_COLUMNS) +
        f"\nRecomputed against **{hw.name}**, {hw.bandwidth_bytes_s / 1e9:.1f} "
        f"GB/s (pattern `{hw.ceiling_pattern or 'unnamed'}`).\n\n"
        f"No FINDINGS.md here on purpose. The analysis in `{args.arm.name}` was\n"
        f"written against the ruler of that day, and the two arms together are\n"
        f"the evidence that the ruler moved. Read that one, and treat these rows\n"
        f"as the corrected numbers.\n")
    print(f"\nwrote {out}")
    print("The original arm is untouched: it is what was measured against the")
    print("ruler of the day, and the pair is the evidence that the ruler moved.")
    print(f"Dropped a {DERIVED_MARKER} marker, so the publish gate knows this")
    print("arm's calibration is from a later session on purpose.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
