#!/usr/bin/env python
"""Turn a results directory into figures. Reads CSVs only; never touches a GPU.

Three views, each answering a question the project actually asks:
  roofline     is this kernel memory bound or compute bound, and how close to
               the roof does it get at the intensity its tiling implies
  scaling      latency against token count, which is where the decode wall shows
  imbalance    latency against measured load skew at fixed token count, which is
               the load-imbalance story stated as a curve
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot; no display on a headless pod

import matplotlib.pyplot as plt  # noqa: E402

from moe.bench import roofline as RL  # noqa: E402
from moe.bench.schema import read_csv  # noqa: E402


def load_rows(results: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results.glob("*.csv")):
        if path.name == "merged.csv":
            continue
        try:
            rows.extend(read_csv(path))
        except (ValueError, OSError) as e:
            print(f"[plot] skipping {path.name}: {e}")
    merged = results / "merged.csv"
    if merged.exists() and not rows:
        rows = read_csv(merged)
    return [r for r in rows if r.get("correctness_passed") == "True"]


def facet(r) -> str:
    """Series label that never folds incomparable timing methodologies together.

    An L2-flushed measurement and an L2-warm one are different experiments. So
    are eager and graph-replay. Merging them into one series would hide the very
    effect the harness records these axes to expose.
    """
    bits = [r.get("impl", "")]
    bits.append("L2-flushed" if str(r.get("l2_flush")) in ("True", "true", "1")
                else "L2-warm")
    if str(r.get("cuda_graph")) in ("True", "true", "1"):
        bits.append("graph")
    return " / ".join(b for b in bits if b)


def _f(row, key, default=0.0):
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def plot_scaling(rows, out_dir: Path, dtype: str = "bf16"):
    """Latency against token count, one line per implementation, per model."""
    by_model = defaultdict(list)
    for r in rows:
        if r["dtype"] == dtype and r["scope"] == "span" and _f(r, "ms_p50") > 0:
            by_model[r["model"]].append(r)

    written = []
    for model, mrows in sorted(by_model.items()):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        by_impl = defaultdict(list)
        for r in mrows:
            by_impl[f"{facet(r)} ({r['routing_kind']})"].append(r)
        for label, series in sorted(by_impl.items()):
            series.sort(key=lambda r: int(float(r["num_tokens"])))
            xs = [int(float(r["num_tokens"])) for r in series]
            ys = [_f(r, "ms_p50") for r in series]
            ax.loglog(xs, ys, "o-", ms=4, lw=1.3, alpha=0.85, label=label)
        ax.set_xlabel("tokens entering the layer (1 = decode step)")
        ax.set_ylabel("p50 latency (ms)")
        ax.set_title(f"{model}: grouped-GEMM latency vs token count, {dtype}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / f"scaling_{model}_{dtype}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def plot_imbalance(rows, out_dir: Path, dtype: str = "bf16"):
    """Latency against measured load skew, at each token count."""
    pts = [r for r in rows
           if r["dtype"] == dtype and r["scope"] == "span"
           and _f(r, "ms_p50") > 0 and _f(r, "load_max_over_mean") > 0]
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(7.5, 5))
    by_impl = defaultdict(list)
    for r in pts:
        by_impl[facet(r)].append(r)
    for impl, series in sorted(by_impl.items()):
        xs = [_f(r, "load_max_over_mean") for r in series]
        ys = [_f(r, "ms_p50") for r in series]
        ax.loglog(xs, ys, "o", ms=5, alpha=0.8, label=impl)
    ax.set_xlabel("expert load imbalance (max rows / mean rows)")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title(f"Sensitivity to routing skew, {dtype}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / f"imbalance_{dtype}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--hardware", default="h200_nvl")
    ap.add_argument("--allow-unverified-roof", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.results)
    if not rows:
        print(f"[plot] no correctness-passing rows in {args.results}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    written = []

    try:
        written.append(RL.plot(rows, args.out / f"roofline_{args.dtype}.png",
                               hardware=args.hardware, dtype=args.dtype,
                               allow_unverified=args.allow_unverified_roof))
    except (RL.UnverifiedHardware, ValueError) as e:
        print(f"[plot] roofline skipped: {e}")

    written += plot_scaling(rows, args.out, args.dtype)
    written += plot_imbalance(rows, args.out, args.dtype)

    for p in written:
        print(f"[plot] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
