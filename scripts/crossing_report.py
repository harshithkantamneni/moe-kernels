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
import collections
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe.bench.crossing import (  # noqa: E402
    crossing_from_points,
    local_slopes,
    timed_rows,
)
from moe.bench.ridge import (  # noqa: E402
    crossing_batch,
    ridge_for_dtype,
    saturation_batch,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--ridge", type=float, required=True,
                    help="measured bf16 FLOP/byte from the calibration; other "
                         "dtypes are scaled by their FLOP rate, since a format "
                         "changes the peak as well as the bytes")
    ap.add_argument("--impl", default=None, help="restrict to one implementation")
    ap.add_argument("--routing", default=None, help="restrict to one routing kind")
    ap.add_argument("--include-throttled", action="store_true")
    ap.add_argument("--l2-flush", choices=["true", "false"], default=None,
                    help="restrict to one L2 mode; default mixes both")
    ap.add_argument("--cuda-graph", choices=["true", "false"], default=None,
                    help="restrict to one capture mode; default mixes both")
    args = ap.parse_args()

    cells: dict[tuple[str, str], dict[int, list[float]]] = {}
    modes: collections.Counter = collections.Counter()
    kept = skipped = untimed = 0
    for path in args.csvs:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
            timed = timed_rows(rows)
            untimed += len(rows) - len(timed)
            for r in timed:
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
                if args.l2_flush is not None and \
                        str(r.get("l2_flush", "")).lower() != args.l2_flush:
                    continue
                if args.cuda_graph is not None and \
                        str(r.get("cuda_graph", "")).lower() != args.cuda_graph:
                    continue
                modes[(str(r.get("l2_flush")), str(r.get("cuda_graph")))] += 1
                try:
                    t, ms = int(r["num_tokens"]), float(r["ms_p50"])
                except (ValueError, KeyError):
                    continue
                # `impl` is in the key, not just an optional filter. Rows from
                # different implementations measure different SCOPES -- one
                # stage against five against a whole layer, 16.7x apart on the
                # published sweep -- so a median across them describes nothing.
                key = (r["model"], r["dtype"], r["impl"])
                cells.setdefault(key, {}).setdefault(t, []).append(ms)
                kept += 1

    print(f"kept {kept} rows, skipped {skipped} (throttled or failed), "
          f"{untimed} never timed (skipped graph mode: ms_p50 is 0.0, "
          f"which is not a measurement)")
    if len(modes) > 1:
        print("  timing modes mixed into each median (l2_flush, cuda_graph): "
              + ", ".join(f"{k}x{v}" for k, v in sorted(modes.items())))
        print("  pass --l2-flush/--cuda-graph to isolate one; the crossing is a "
              "slope, so mixing adds spread rather than bias")
    print()
    if not cells:
        print("nothing to report")
        return 1

    for (model, dtype, impl), by_t in sorted(cells.items()):
        points = [(t, statistics.median(v)) for t, v in sorted(by_t.items())]
        print(f"=== {model} / {dtype} / {impl} ===")
        slopes = dict(local_slopes(points))
        print(f"  {'T':>6} {'ms_p50':>9} {'slope':>7}   regime")
        prev = None
        for t, ms in points:
            s = next((v for k, v in slopes.items() if prev and prev < k < t), None)
            tag = "" if s is None else ("weight-bound" if s < 0.5 else "compute-bound")
            print(f"  {t:>6} {ms:>9.4f} {'' if s is None else f'{s:>7.3f}'}   {tag}")
            prev = t

        # Below E/k a batch misses experts, so weight traffic grows with the
        # batch and the slope crosses for a reason unrelated to the ridge.
        sat = saturation_batch(model)
        measured = crossing_from_points(points, min_tokens=sat)
        dtype_ridge = ridge_for_dtype(args.ridge, dtype)
        predicted = crossing_batch(model, dtype_ridge, dtype)
        print(f"\n  saturation (E/k, floor):             {sat:8.0f} tokens")
        print(f"  ridge for {dtype:<9}                  {dtype_ridge:8.1f} FLOP/byte")
        print(f"  predicted (2R/b at that ridge):      {predicted:8.0f} tokens")
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
