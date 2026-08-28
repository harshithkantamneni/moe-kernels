#!/usr/bin/env python3
"""Does achieved-versus-peak explain why measured crossings land below `2R/b`?

Measured crossings sit at 0.63x (bf16) and 0.71x (fp8) of prediction across four
models. If that offset is the kernel reaching less of peak FLOPs than of peak
bandwidth, then

    measured_crossing / predicted_crossing  ==  effective_ridge / nominal_ridge

Both sides are computed here from the same CSVs, so the hypothesis can fail.

    python scripts/efficiency_report.py /workspace/results/run_*_vllm.csv \
        --ridge 160.3
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe.bench.crossing import crossing_from_points, timed_rows  # noqa: E402
from moe.bench.efficiency import efficiency_from_rows  # noqa: E402
from moe.bench.ridge import (  # noqa: E402
    crossing_batch,
    ridge_for_dtype,
    saturation_batch,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--ridge", type=float, required=True,
                    help="measured bf16 FLOP/byte from the calibration")
    ap.add_argument("--impl", default=None)
    ap.add_argument("--include-throttled", action="store_true")
    args = ap.parse_args()

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for path in args.csvs:
        with path.open(newline="") as fh:
            for r in timed_rows(list(csv.DictReader(fh))):
                if args.impl and r["impl"] != args.impl:
                    continue
                if r.get("correctness_passed") not in ("True", "true", "1", ""):
                    continue
                if not args.include_throttled and \
                        r.get("throttled") in ("True", "true", "1"):
                    continue
                groups.setdefault((r["model"], r["dtype"], r["impl"]), []).append(r)

    if not groups:
        print("nothing to report")
        return 1

    print(f"{'model / dtype / impl':<56} {'eff ridge':>10} {'eff/nom':>8} "
          f"{'meas/pred':>10} {'agree?':>8}")
    print("-" * 96)
    pairs = []
    for (model, dtype, impl), rows in sorted(groups.items()):
        eff = efficiency_from_rows(rows)
        if eff is None:
            continue
        by_t: dict[int, list[float]] = {}
        for r in rows:
            try:
                by_t.setdefault(int(r["num_tokens"]), []).append(float(r["ms_p50"]))
            except (KeyError, ValueError):
                continue
        points = [(t, statistics.median(v)) for t, v in sorted(by_t.items())]
        measured = crossing_from_points(points, min_tokens=saturation_batch(model))
        nominal = ridge_for_dtype(args.ridge, dtype)
        predicted = crossing_batch(model, nominal, dtype)

        eff_ratio = eff.ratio_against(nominal)
        label = f"{model} / {dtype} / {impl}"
        if measured is None:
            print(f"{label:<56} {eff.effective_ridge:>10.1f} {eff_ratio:>8.2f} "
                  f"{'no crossing':>10} {'':>8}")
            continue
        meas_ratio = measured / predicted
        # Within 20% is agreement at this resolution: the token grid is
        # log-spaced, so a crossing is only located to about a factor of 1.5.
        agree = "yes" if abs(eff_ratio - meas_ratio) / meas_ratio < 0.20 else "NO"
        pairs.append((eff_ratio, meas_ratio))
        print(f"{label:<56} {eff.effective_ridge:>10.1f} {eff_ratio:>8.2f} "
              f"{meas_ratio:>10.2f} {agree:>8}")

    if pairs:
        print()
        e = statistics.mean(p[0] for p in pairs)
        m = statistics.mean(p[1] for p in pairs)
        print(f"mean effective/nominal ridge   {e:.3f}")
        print(f"mean measured/predicted cross  {m:.3f}")
        print(f"gap                            {abs(e - m) / m * 100:.1f}%")
        print()
        if abs(e - m) / m < 0.20:
            print("The two agree: the offset IS the kernel reaching less of peak")
            print("FLOPs than of peak bandwidth. C2 needs no extra term, only the")
            print("honest ridge instead of the datasheet one.")
        else:
            print("The two DISAGREE, so achieved-versus-peak does not explain the")
            print("offset on its own. Activation traffic is the other candidate:")
            print("crossing_batch uses weights-only 2R/b, and real traffic adds")
            print("activations, which lowers AI and moves the crossing earlier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
