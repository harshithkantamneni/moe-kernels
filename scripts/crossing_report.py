#!/usr/bin/env python3
"""Where does a sweep actually cross its ridge, and does C2 predict it?

Reads one or more result CSVs, aggregates to one median per (model, dtype,
token count), recovers the crossing from measured TIME via moe.bench.crossing,
and prints it beside the `2R/b` prediction.

The measured side never consults the byte model, so the prediction can be wrong.

    python scripts/crossing_report.py /workspace/results/run_h200fp8b_vllm.csv \
        --ridge 160.3 --impl vllm_fused_experts
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe.bench.crossing import crossing_from_points, local_slopes  # noqa: E402
from moe.bench.ridge import crossing_batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--ridge", type=float, required=True,
                    help="measured FLOP/byte, from the calibration")
    ap.add_argument("--impl", default=None, help="restrict to one implementation")
    ap.add_argument("--routing", default=None, help="restrict to one routing kind")
    ap.add_argument("--include-throttled", action="store_true")
    args = ap.parse_args()

    cells: dict[tuple[str, str], dict[int, list[float]]] = {}
    kept = skipped = 0
    for path in args.csvs:
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if args.impl and r["impl"] != args.impl:
                    continue
                if args.routing and r["routing_kind"] != args.routing:
                    continue
                if r.get("correctness_passed") not in ("True", "true", "1", ""):
                    skipped += 1
                    continue
                if not args.include_throttled and r.get("throttled") in ("True", "true", "1"):
                    skipped += 1
                    continue
                try:
                    t, ms = int(r["num_tokens"]), float(r["ms_p50"])
                except (ValueError, KeyError):
                    continue
                cells.setdefault((r["model"], r["dtype"]), {}).setdefault(t, []).append(ms)
                kept += 1

    print(f"kept {kept} rows, skipped {skipped} (throttled or failed)\n")
    if not cells:
        print("nothing to report")
        return 1

    for (model, dtype), by_t in sorted(cells.items()):
        points = [(t, statistics.median(v)) for t, v in sorted(by_t.items())]
        print(f"=== {model} / {dtype} ===")
        slopes = dict(local_slopes(points))
        print(f"  {'T':>6} {'ms_p50':>9} {'slope':>7}   regime")
        prev = None
        for t, ms in points:
            s = next((v for k, v in slopes.items() if prev and prev < k < t), None)
            tag = "" if s is None else ("weight-bound" if s < 0.5 else "compute-bound")
            print(f"  {t:>6} {ms:>9.4f} {'' if s is None else f'{s:>7.3f}'}   {tag}")
            prev = t

        measured = crossing_from_points(points)
        predicted = crossing_batch(model, args.ridge, dtype)
        print(f"\n  predicted (2R/b, ridge {args.ridge}): {predicted:8.0f} tokens")
        if measured is None:
            print("  measured:                            not bracketed by this "
                  "token grid")
            print("  -> add token counts on both sides of the prediction")
        else:
            ratio = measured / predicted
            print(f"  measured (slope crosses 0.5):        {measured:8.0f} tokens"
                  f"   {ratio:.2f}x predicted")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
