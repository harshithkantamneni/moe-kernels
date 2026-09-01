#!/usr/bin/env python3
"""Is the bandwidth headroom in production MoE kernels gated on the DTYPE?

THE RESULT THIS SCRIPT EXISTS TO MAKE CHECKABLE. Production fused MoE kernels,
uniform routing, T <= 64, the memory-bound regime a kernel would target: in bf16
the incumbent sits near its compulsory byte floor, and in fp8 at the SAME cells
it does not, because halving the weight bytes halves the floor WITHOUT halving
the fixed dispatch and tile-quantisation costs.

THE BUG IT CLOSES. That table was reconstructed at a prompt as
`achieved_bw_gbps / compulsory_gbps`, since the fp8 arm's calibration measured no
fp8 ceiling and so `driver.py` never wrote its `implied_traffic_ratio` column. The
arithmetic was right and the provenance was not, which is exactly the shape of
every number this project has had to retract. `efficiency.traffic_ratio` now names
the route it took and refuses when it has none, and this script is the analysis
built on it.

THE EAGER / GRAPH SPLIT IS LOAD-BEARING, not a facet. The eager-to-graph gap is
the per-call host DISPATCH component, which a CUDA graph replays away; what is
left in graph mode is tile quantisation and everything else. Reporting only the
pooled median hides that the fp8 eager figure is mostly dispatch.

AND THE TWO HALVES COME FROM DIFFERENT ARMS, so the script does not stop at the
pooled table. It re-runs the comparison on cells MATCHED across the two dtypes on
(model, tokens, routing, seed, impl, l2_flush), splits it per model, and
decomposes the gap. If the effect were an arm artefact rather than a dtype
effect, the matched panel is where it would fall apart.

    python scripts/dtype_headroom.py \
      --bf16 results/published/2026-08-26-nvidia_h200-full-three-way-recalibrated/run_*.csv \
      --bf16 results/published/2026-08-28-nvidia_h200-h200-v2lite/run_*.csv \
      --fp8  results/published/2026-08-28-nvidia_h200-h200-fp8-three-kernel/run_*.csv \
      --ridge 160.3
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe.bench.crossing import UNIFORM_ROUTING, routing_domain, timed_rows  # noqa: E402
from moe.bench.efficiency import (  # noqa: E402
    TrafficRatioUnavailable,
    summarise_traffic_ratios,
    traffic_ratio,
)
from moe.bench.published import (  # noqa: E402
    filter_superseded,
    superseded_impls,
    superseded_reason,
)
from moe.bench.ridge import ridge_for_dtype  # noqa: E402
from moe.bench.schema import row_bool, row_float  # noqa: E402

#: The two spans that are somebody's production kernel. `torch grouped_mm` covers
#: one canonical stage of six against these five, and the reference pipeline is a
#: correctness oracle at 12x the floor, so neither belongs in a headroom claim
#: about what a deployed kernel leaves on the table.
PRODUCTION_FUSED = ("vllm_fused_experts", "sglang_fused_experts")

#: What makes two rows the same cell across the two dtypes. `l2_flush` is in the
#: key because it is a swept axis that changes the measured time, and `seed` is in
#: it because uniform routing is SAMPLED per replicate, so two seeds are two
#: different expert histograms rather than two draws of one.
CELL_KEY = ("model", "num_tokens", "routing_kind", "seed", "impl", "l2_flush")

DERIVED_NOTICE = """\
NOTHING BELOW IS A TILE MEASUREMENT. Every published arm is schema v3 and records
no tile configuration, so any statement about BLOCK_SIZE_M on these rows is
DERIVED from vLLM 0.27.1's shipped configs plus the recorded gpu_name, never
OBSERVED. It matters here because vLLM resolves the fused-MoE config by
(E, N, dtype, device) and picks a DIFFERENT tile for the two dtypes on the same
shape -- fp8 takes BLOCK_SIZE_M 64 from M=1 where bf16 takes 16 to 32. So the
comparison below varies the tile along with the dtype, and no row can say by how
much."""


def mode_of(row) -> str:
    """"eager" or "graph". The split the whole analysis turns on, spelled once."""
    return "graph" if row_bool(row, "cuda_graph") else "eager"


def cell_of(row) -> tuple:
    """The identity of the measured cell, mode excluded."""
    return tuple(str(row.get(k, "")) for k in CELL_KEY)


def load_rows(paths: list[Path], *, impl=PRODUCTION_FUSED,
              routing: str = UNIFORM_ROUTING, max_tokens: int = 64,
              include_throttled: bool = False) -> tuple[list[dict], list[str]]:
    """Rows that survive supersession, the routing domain and the token cap.

    Returns the notes it wants printed rather than printing them, so the two
    dtype arms report their exclusions in one block instead of interleaved.

    The throttle filter is ON by default and it is not cosmetic: it is what makes
    the row counts 279 / 182 / 275 / 284 rather than 336 / 184 / 336 / 288, and
    it moves the fp8 eager median from 1.414 to 1.959. A published table that did
    not say which it was would be unreproducible.
    """
    notes: list[str] = []
    seen: set[str] = set()
    kept, dropped = filter_superseded(paths)
    for d in dropped:
        note = f"[skip] {d.parent.name}: {superseded_reason(d).splitlines()[0]}"
        if note not in seen:
            seen.add(note)
            notes.append(note)
    partial: dict[Path, set[str]] = {}
    for c in kept:
        names = superseded_impls(c)
        if names:
            partial[c] = names
            # Deduplicated by ARM, not by file. An arm holds one CSV per env, so
            # a per-file note prints the same retraction six times and buries the
            # rest of the header under it.
            note = (f"[skip] {c.parent.name}: {', '.join(sorted(names))} "
                    f"({superseded_reason(c).splitlines()[0]})")
            if note not in seen:
                seen.add(note)
                notes.append(note)

    rows: list[dict] = []
    for path in kept:
        with path.open(newline="") as fh:
            skip = partial.get(path, set())
            for r in timed_rows(list(csv.DictReader(fh))):
                if r["impl"] in skip or r["impl"] not in impl:
                    continue
                if r.get("correctness_passed") not in ("True", "true", "1", ""):
                    continue
                if r["routing_kind"] != routing:
                    continue
                if row_float(r, "num_tokens") > max_tokens:
                    continue
                if not include_throttled and row_bool(r, "throttled"):
                    continue
                rows.append(r)
    return rows, notes


def _panel(title: str, groups: dict[str, list[dict]], ridges: dict[str, float]) -> None:
    """One group per (dtype, mode), so the ridge is looked up per group.

    Per group rather than per panel because the ridge is dtype-dependent: fp8
    runs its tensor cores at twice the bf16 rate, so scoring an fp8 row against
    the bf16 ridge is the exact arithmetic the retracted 2x crossing prediction
    was built on.
    """
    if title:
        print(title)
    print(f"  {'dtype / mode':<22} {'n':>5} {'p10':>8} {'median':>8} {'p90':>8}  route")
    for label, rows in groups.items():
        ridge = ridges.get(rows[0]["dtype"]) if rows else None
        s = summarise_traffic_ratios(rows, ridge=ridge)
        if s is None:
            print(f"  {label:<22} {'0':>5}   nothing scored")
            continue
        refused = f"  ({s.refused} refused)" if s.refused else ""
        print(f"  {label:<22} {s.n:>5} {s.p10:>8.3f} {s.median:>8.3f} {s.p90:>8.3f}  "
              f"{s.route}{refused}")
    print()


def _value(row, ridge) -> float | None:
    try:
        return traffic_ratio(row, ridge=ridge).value
    except TrafficRatioUnavailable:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", action="append", type=Path, default=[], nargs="+",
                    help="CSVs for the bf16 side; repeatable")
    ap.add_argument("--fp8", action="append", type=Path, default=[], nargs="+",
                    help="CSVs for the fp8 side; repeatable")
    ap.add_argument("--ridge", type=float, default=None,
                    help="measured bf16 FLOP/byte. Given, every row is checked "
                         "against its own dtype's ridge and a compute-bound row "
                         "is refused rather than scored")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="the memory-bound regime the claim is about")
    ap.add_argument("--routing", default=UNIFORM_ROUTING)
    ap.add_argument("--impl", nargs="+", default=list(PRODUCTION_FUSED))
    ap.add_argument("--include-throttled", action="store_true")
    args = ap.parse_args()

    bf16_paths = [p for group in args.bf16 for p in group]
    fp8_paths = [p for group in args.fp8 for p in group]
    if not bf16_paths or not fp8_paths:
        print("both --bf16 and --fp8 are required: this script compares them")
        return 2

    load = dict(impl=args.impl, routing=args.routing, max_tokens=args.max_tokens,
                include_throttled=args.include_throttled)
    bf16, n1 = load_rows(bf16_paths, **load)
    fp8, n2 = load_rows(fp8_paths, **load)
    for line in n1 + n2:
        print(line)
    if n1 or n2:
        print()
    if not bf16 or not fp8:
        print(f"nothing to compare: {len(bf16)} bf16 rows, {len(fp8)} fp8 rows")
        return 1

    dtypes = {r["dtype"] for r in bf16} | {r["dtype"] for r in fp8}
    dom = routing_domain(bf16 + fp8)
    for line in dom.warning_lines():
        print(line)
    print(f"dtypes present: {', '.join(sorted(dtypes))}")
    print(f"rows: {len(bf16)} bf16, {len(fp8)} fp8, "
          f"routing {dom.census}, T <= {args.max_tokens}")
    print()
    print(DERIVED_NOTICE)
    print()

    ridges = {}
    if args.ridge is not None:
        for dt in sorted(dtypes):
            ridges[dt] = ridge_for_dtype(args.ridge, dt)
        print("ridge per dtype: " +
              ", ".join(f"{k} {v:.1f}" for k, v in sorted(ridges.items())))
        print()

    def ridge_of(row):
        return ridges.get(row["dtype"])

    # --- 1. the pooled table, which is the published one ---------------------
    pooled: dict[str, list[dict]] = {}
    for rows in (bf16, fp8):
        for r in rows:
            pooled.setdefault(f"{r['dtype']} {mode_of(r)}", []).append(r)
    _panel("POOLED, every row that passes the filters. This is the published table.",
           dict(sorted(pooled.items())), ridges)

    # --- 2. the same thing on cells matched across the two dtypes -------------
    #
    # The adversarial question. The bf16 and fp8 halves come from different arms
    # with different calibrations and different commits, so a pooled difference
    # could be an arm artefact carried by unequal cell coverage rather than a
    # dtype effect.
    by_cell_bf16 = {(cell_of(r), mode_of(r)): r for r in bf16}
    by_cell_fp8 = {(cell_of(r), mode_of(r)): r for r in fp8}
    shared = sorted(set(by_cell_bf16) & set(by_cell_fp8))
    print(f"MATCHED on {' / '.join(CELL_KEY)} + mode: {len(shared)} cells "
          f"({len(by_cell_bf16) - len(shared)} bf16-only, "
          f"{len(by_cell_fp8) - len(shared)} fp8-only)")
    matched: dict[str, list[dict]] = {}
    for key in shared:
        for src in (by_cell_bf16, by_cell_fp8):
            r = src[key]
            matched.setdefault(f"{r['dtype']} {key[1]}", []).append(r)
    _panel("", dict(sorted(matched.items())), ridges)

    # --- 3. paired, per model ------------------------------------------------
    print("PAIRED fp8/bf16 on the matched cells, per model. A pooled median can "
          "be carried\nby one model; this says whether every model moves the "
          "same way.")
    print(f"  {'model':<20} {'mode':<6} {'n':>5} {'bf16':>8} {'fp8':>8} "
          f"{'paired':>8} {'frac>1':>8}")
    for model in sorted({k[0][0] for k in shared}):
        for mode in ("eager", "graph"):
            keys = [k for k in shared if k[0][0] == model and k[1] == mode]
            pairs = []
            for k in keys:
                b = _value(by_cell_bf16[k], ridge_of(by_cell_bf16[k]))
                f = _value(by_cell_fp8[k], ridge_of(by_cell_fp8[k]))
                if b and f:
                    pairs.append((b, f))
            if not pairs:
                continue
            frac = sum(f > b for b, f in pairs) / len(pairs)
            print(f"  {model:<20} {mode:<6} {len(pairs):>5} "
                  f"{statistics.median(b for b, _ in pairs):>8.3f} "
                  f"{statistics.median(f for _, f in pairs):>8.3f} "
                  f"{statistics.median(f / b for b, f in pairs):>8.3f} {frac:>8.2f}")
    print()

    # --- 4. dispatch against everything else ---------------------------------
    #
    # On cells that carry all four of (bf16, fp8) x (eager, graph), the gap is
    # additive and separates exactly:
    #
    #     fp8_eager - bf16_eager
    #        = (fp8_graph - bf16_graph)                       <- residual
    #        + [(fp8_eager - fp8_graph) - (bf16_eager - bf16_graph)]  <- dispatch
    #
    # The residual is the part a CUDA graph cannot replay away, which is the
    # figure a serving system would feel.
    quad = [c for c in {k[0] for k in shared}
            if all((c, m) in by_cell_bf16 and (c, m) in by_cell_fp8
                   for m in ("eager", "graph"))]
    print(f"DECOMPOSITION on the {len(quad)} cells measured in all four of "
          f"(bf16, fp8) x (eager, graph).")
    if quad:
        parts: dict[str, list[float]] = {k: [] for k in
                                         ("bf16 dispatch", "fp8 dispatch",
                                          "residual (graphed fp8 - bf16)",
                                          "dispatch differential",
                                          "total eager gap")}
        for c in quad:
            v = {}
            for dt, src in (("bf16", by_cell_bf16), ("fp8", by_cell_fp8)):
                for m in ("eager", "graph"):
                    v[dt, m] = _value(src[c, m], ridge_of(src[c, m]))
            if any(x is None for x in v.values()):
                continue
            parts["bf16 dispatch"].append(v["bf16", "eager"] - v["bf16", "graph"])
            parts["fp8 dispatch"].append(v["fp8", "eager"] - v["fp8", "graph"])
            parts["residual (graphed fp8 - bf16)"].append(
                v["fp8", "graph"] - v["bf16", "graph"])
            parts["dispatch differential"].append(
                (v["fp8", "eager"] - v["fp8", "graph"])
                - (v["bf16", "eager"] - v["bf16", "graph"]))
            parts["total eager gap"].append(v["fp8", "eager"] - v["bf16", "eager"])
        for name, values in parts.items():
            if not values:
                continue
            frac = sum(x > 0 for x in values) / len(values)
            print(f"  {name:<32} median {statistics.median(values):+7.3f} "
                  f"  positive in {frac:.0%} of cells")
        # The same split in TIME, which is where an arm artefact would show: a
        # dispatch cost that is fixed in microseconds contributes twice as much
        # to an fp8 ratio as to a bf16 one, and no more than that.
        print()
        print("  and the same eager-to-graph gap in TIME, where a fixed cost is "
              "fixed:")
        for dt, src in (("bf16", by_cell_bf16), ("fp8", by_cell_fp8)):
            gaps = [row_float(src[c, "eager"], "ms_p50")
                    - row_float(src[c, "graph"], "ms_p50") for c in quad]
            graphed = [row_float(src[c, "graph"], "ms_p50") for c in quad]
            print(f"    {dt:<6} median graph {statistics.median(graphed) * 1e3:8.1f} us"
                  f"   median eager - graph {statistics.median(gaps) * 1e3:8.1f} us")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
