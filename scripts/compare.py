#!/usr/bin/env python
"""Compare implementations at matched conditions, with span extent made visible.

    python scripts/compare.py --results /workspace/results
    python scripts/compare.py --model deepseek-v3 --routing zipf:1.2
    python scripts/compare.py --metric compulsory_gbps

THE TRAP THIS EXISTS TO PREVENT. `torch_grouped_mm_up` covers ONE canonical
stage; vLLM's and SGLang's fused_experts cover FIVE; `__pipeline__` covers the
whole reference layer. Putting their ms side by side compares a fused block
against a single GEMM. `covers` is a column so the difference is recorded, but
nothing stops a chart from ignoring it, and `series_label()` keys on `impl`.

So this prints the extent under every table and says loudly when the columns are
not the same shape. Implementations sharing an extent are directly comparable;
the others need a normalised metric or a caveat.

Rows are filtered to ONE set of conditions rather than averaged over them,
because L2-cold and L2-warm are different experiments and so are eager and graph
replay. Averaging across those is how two methodologies become one number.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RIDGE_DEFAULT = 166.0


def load(results: Path) -> list[dict]:
    merged = results / "merged.csv"
    paths = [merged] if merged.exists() else sorted(results.glob("run_*.csv"))
    rows: list[dict] = []
    for p in paths:
        with p.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def rows_per_expert(row: dict) -> float:
    active = int(row.get("load_active_experts") or 0)
    total = int(row.get("load_total_rows") or 0)
    return total / active if active else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--model", default=None, help="default: every model present")
    ap.add_argument("--routing", default="uniform")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--l2-warm", action="store_true", help="default is L2-cold")
    ap.add_argument("--graph", action="store_true", help="default is eager")
    ap.add_argument("--metric", default="ms_p50",
                    choices=("ms_p50", "compulsory_gbps", "tflops",
                             "implied_traffic_ratio"))
    ap.add_argument("--ridge", type=float, default=RIDGE_DEFAULT,
                    help="FLOP/byte crossover, for the regime column")
    args = ap.parse_args()

    rows = load(args.results)
    if not rows:
        print(f"no rows under {args.results}", file=sys.stderr)
        return 1

    want_l2 = "False" if args.l2_warm else "True"
    want_graph = "True" if args.graph else "False"
    sel = [r for r in rows
           if r.get("l2_flush") == want_l2
           and r.get("cuda_graph") == want_graph
           and r.get("seed") == args.seed
           and r.get("correctness_passed") == "True"
           and r.get(args.metric) and float(r[args.metric]) > 0]
    if args.routing != "any":
        kind, _, param = args.routing.partition(":")
        sel = [r for r in sel if r.get("routing_kind") == kind
               and (not param or f"{float(r.get('routing_param') or 0):g}" == param)]
    if args.model:
        sel = [r for r in sel if r.get("model") == args.model]
    if not sel:
        print("no rows match those conditions", file=sys.stderr)
        return 1

    cond = (f"routing={args.routing} seed={args.seed} "
            f"L2={'warm' if args.l2_warm else 'cold'} "
            f"{'graph' if args.graph else 'eager'}  metric={args.metric}")
    print(cond)

    covers: dict[str, str] = {}
    for r in sel:
        covers.setdefault(r["impl"], r.get("covers", ""))

    for model in sorted({r["model"] for r in sel}):
        mrows = [r for r in sel if r["model"] == model]
        impls = sorted({r["impl"] for r in mrows})
        by: dict[tuple[int, str], float] = {}
        for r in mrows:
            by[(int(r["num_tokens"]), r["impl"])] = float(r[args.metric])
        tokens = sorted({int(r["num_tokens"]) for r in mrows})
        rpe = {int(r["num_tokens"]): rows_per_expert(r) for r in mrows}

        print(f"\n=== {model} ===")
        head = f"{'T':>6} {'rows/exp':>9} {'regime':>7}  "
        head += "  ".join(f"{i[:20]:>20}" for i in impls)
        print(head)
        for t in tokens:
            cells = []
            for i in impls:
                v = by.get((t, i))
                cells.append(f"{v:20.3f}" if v is not None else f"{'-':>20}")
            reg = "COMPUTE" if rpe.get(t, 0) >= args.ridge else "memory"
            print(f"{t:6d} {rpe.get(t, 0):9.0f} {reg:>7}  " + "  ".join(cells))

    print("\nspan extent per implementation:")
    shapes = defaultdict(list)
    for impl, cov in sorted(covers.items()):
        n = len(cov.split("+")) if cov and cov != "all" else 6
        shapes[cov].append(impl)
        print(f"  {impl:24s} {n} stage(s)  covers={cov}")
    if len(shapes) > 1:
        print("\n  WARNING: these columns do NOT cover the same stages, so their")
        print("  raw times are not comparable. Groups that ARE comparable:")
        for cov, group in shapes.items():
            if len(group) > 1:
                print(f"    {', '.join(group)}   ({cov})")
        print("  Across groups, use --metric compulsory_gbps or state the extent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
