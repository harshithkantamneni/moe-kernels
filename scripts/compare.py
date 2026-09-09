#!/usr/bin/env python
"""Compare implementations at matched conditions, with span extent made visible.

    python scripts/compare.py --results /workspace/results
    python scripts/compare.py --model deepseek-v3 --routing zipf:1.2
    python scripts/compare.py --metric compulsory_gbps
    python scripts/compare.py --self-test

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

Routing is one of those conditions, which is why `--routing` defaults to uniform
and `--routing any` announces itself. See `crossing.routing_domain`.

THE RIDGE THE REGIME COLUMN IS SCORED AGAINST IS NOT A CONSTANT. Until
2026-09-08 this file carried `RIDGE_DEFAULT = 166.0` (retired) and `--ridge` defaulted
to it, so every table printed a per-row COMPUTE/memory verdict against a
figure belonging to no card: the committed calibrations put the H200 at 162.8
Op/B and the A100 at 145.8, so any grid with rows/expert between a card's own
ridge and 166 was labelled memory when the card's own ruler says compute, and
nothing on the page said where 166 came from. Retraction (e) of
`docs/FINDINGS.md` ("no --ridge default is a number", commit c42a20c) had
been applied to every other `--ridge` in scripts/ and missed this one: the
project's recurring defect, a fix applied at one of two call sites.

Now the ridge is one of three things, and the table says which:

  1. `--ridge <Op/B>`: the operator's assertion, printed as such, applied to
     every dtype in the selection. No file stands behind it and the line
     under the conditions says so.
  2. Otherwise THE CARD THE ROWS NAME: `roofline.hardware_for_rows` reads the
     `gpu_name` column and resolves that card's committed calibration
     (`moe/bench/hardware/measured_<card>.yaml`), and the ridge is
     `peak(dtype) / bandwidth` per dtype present. Figures get drawn on a
     laptop from a committed CSV, so the rows' device is the right one to
     ask, never the device running this process.
  3. Neither: the regime column is REFUSED by name (`no-ridge` in every cell)
     and the ms columns still print, because they never depended on a ridge
     and the one column that does must not carry a number nobody can trace
     to a file. The refusal names the flag that turns it into an assertion.

`--self-test` plants all three so the resolving and the refusing paths are
both exercised off-GPU.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe.bench import roofline  # noqa: E402
from moe.bench.crossing import routing_domain  # noqa: E402

#: What the regime column prints when no ridge is entitled to score it. A
#: word rather than a blank so a reader cannot mistake the column for one
#: that was scored, and one that no dtype or regime name collides with.
NO_RIDGE = "no-ridge"

#: What it prints when the rows at one batch carry two dtypes whose ridges
#: differ: there is no single ridge for that line and the tool will not pick.
MIXED_RIDGE = "mixed"


@dataclass(frozen=True)
class RidgeResolution:
    """Ridge per dtype for the regime column, and one line saying whence.

    `by_dtype` empty means REFUSED, and `source` then carries the reason and
    the fix; a caller that prints `source` prints something true either way,
    which is the same one-string discipline `roofline.ReferenceClock` uses.
    `asserted` records that the number came from the command line, so a
    report can be told apart from a measurement by more than its wording.
    """

    by_dtype: dict[str, float]
    source: str
    asserted: bool = False

    @property
    def refused(self) -> bool:
        return not self.by_dtype


def resolve_ridge(rows: list[dict], asserted: float | None) -> RidgeResolution:
    """The ridge these rows are entitled to be scored against, or a refusal.

    `asserted` is `--ridge`; None means resolve through `roofline` from the
    card the rows name. Never a module constant: the constant this replaced
    (166.0, retired) was quoted on both cards' tables without a source line.
    """
    dtypes = sorted({(r.get("dtype") or "bf16") for r in rows})
    if asserted is not None:
        if asserted <= 0:
            raise ValueError(f"--ridge must be positive Op/B, got {asserted!r}")
        return RidgeResolution(
            {d: asserted for d in dtypes},
            f"ridge {asserted:g} Op/B ASSERTED on the command line (--ridge) "
            f"for every dtype ({', '.join(dtypes)}); no calibration file stands "
            "behind it",
            asserted=True)
    try:
        hw = roofline.hardware_for_rows("measured", rows)
    except (FileNotFoundError, roofline.HardwareMismatch,
            roofline.UnverifiedHardware, ValueError, KeyError) as exc:
        return RidgeResolution(
            {},
            f"REFUSED, the rows name no card with a committed calibration "
            f"({exc}); every regime cell reads {NO_RIDGE!r}. To assert a ridge "
            "yourself pass --ridge <Op/B>, and it will be printed as your "
            "assertion")
    by_dtype: dict[str, float] = {}
    missing: list[str] = []
    for d in dtypes:
        try:
            by_dtype[d] = hw.ridge_point(d)
        except ValueError:
            missing.append(d)
    if not by_dtype:
        return RidgeResolution(
            {},
            f"REFUSED, {hw.name} carries no verified peak for "
            f"{', '.join(missing)}, so it cannot state a ridge; every regime "
            f"cell reads {NO_RIDGE!r}. Pass --ridge <Op/B> to assert one")
    parts = [f"{d} {v:.1f} Op/B ({hw.peak(d) / 1e12:.1f} TFLOP/s over "
             f"{hw.bandwidth_bytes_s / 1e9:.1f} GB/s)" for d, v in by_dtype.items()]
    source = f"ridge from the rows' own card, {hw.name}: " + "; ".join(parts)
    if missing:
        source += (f"; no verified peak for {', '.join(missing)}, those rows "
                   f"read {NO_RIDGE!r}")
    return RidgeResolution(by_dtype, source)


def regime(rows_per_expert: float, dtypes: set[str], ridge: RidgeResolution) -> str:
    """One line's COMPUTE/memory verdict, or the named refusal."""
    values = {ridge.by_dtype.get(d or "bf16") for d in dtypes}
    if None in values or not values:
        return NO_RIDGE
    if len(values) > 1:
        return MIXED_RIDGE
    return "COMPUTE" if rows_per_expert >= values.pop() else "memory"


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


def select(rows: list[dict], args) -> list[dict]:
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
    return sel


def render(sel: list[dict], args, out=None) -> None:
    """Print the tables for an already-filtered selection."""
    out = out or sys.stdout
    cond = (f"routing={args.routing} seed={args.seed} "
            f"L2={'warm' if args.l2_warm else 'cold'} "
            f"{'graph' if args.graph else 'eager'}  metric={args.metric}")
    print(cond, file=out)
    ridge = resolve_ridge(sel, args.ridge)
    print(f"regime column: {ridge.source}", file=out)

    # `--routing any` pools regimes into one table, and the `regime` column is a
    # roofline statement that only holds under uniform routing. Warned rather
    # than filtered: the default already IS uniform, so anything else was asked
    # for on purpose, and quietly overriding a flag the caller typed is worse
    # than printing a number next to what it is.
    domain = routing_domain(sel)
    banner = domain.warning_lines()
    if banner:
        print(file=out)
        for line in banner:
            print(line, file=out)
        print("  the `regime` column below scores each row's MEAN rows/expert "
              "against the", file=out)
        print("  ridge, so it is a uniform-routing statement as well, and under "
              "skew no", file=out)
        print("  expert is at the mean", file=out)

    covers: dict[str, str] = {}
    for r in sel:
        covers.setdefault(r["impl"], r.get("covers", ""))

    for model in sorted({r["model"] for r in sel}):
        mrows = [r for r in sel if r["model"] == model]
        impls = sorted({r["impl"] for r in mrows})
        by: dict[tuple[int, str], float] = {}
        dtypes_at: dict[int, set[str]] = defaultdict(set)
        for r in mrows:
            by[(int(r["num_tokens"]), r["impl"])] = float(r[args.metric])
            dtypes_at[int(r["num_tokens"])].add(r.get("dtype") or "bf16")
        tokens = sorted({int(r["num_tokens"]) for r in mrows})
        rpe = {int(r["num_tokens"]): rows_per_expert(r) for r in mrows}

        print(f"\n=== {model} ===", file=out)
        head = f"{'T':>6} {'rows/exp':>9} {'regime':>8}  "
        head += "  ".join(f"{i[:20]:>20}" for i in impls)
        print(head, file=out)
        for t in tokens:
            cells = []
            for i in impls:
                v = by.get((t, i))
                cells.append(f"{v:20.3f}" if v is not None else f"{'-':>20}")
            reg = regime(rpe.get(t, 0), dtypes_at[t], ridge)
            print(f"{t:6d} {rpe.get(t, 0):9.0f} {reg:>8}  " + "  ".join(cells),
                  file=out)

    print("\nspan extent per implementation:", file=out)
    shapes = defaultdict(list)
    for impl, cov in sorted(covers.items()):
        n = len(cov.split("+")) if cov and cov != "all" else 6
        shapes[cov].append(impl)
        print(f"  {impl:24s} {n} stage(s)  covers={cov}", file=out)
    if len(shapes) > 1:
        print("\n  WARNING: these columns do NOT cover the same stages, so their",
              file=out)
        print("  raw times are not comparable. Groups that ARE comparable:", file=out)
        for cov, group in shapes.items():
            if len(group) > 1:
                print(f"    {', '.join(group)}   ({cov})", file=out)
        print("  Across groups, use --metric compulsory_gbps or state the extent.",
              file=out)


def self_test() -> int:
    """Plant the three ridge paths and check each prints what it is.

    PASS side: rows naming the committed H200 resolve to that yaml's own
    bf16 ridge (a number with a file behind it, never the retired 166.0, never the
    withdrawn 160.3/176.2 pair), and rows/expert just above and just below
    it read COMPUTE and memory. FAIL side, planted so the refusal is seen to
    fire: rows naming no card get `no-ridge` in every cell with the flag that
    fixes it in the source line. And an asserted `--ridge` is printed as an
    assertion, scored against the asserted number, on the same rows.
    """
    import io

    def row(t: int, gpu: str, dtype: str = "bf16") -> dict:
        return {"impl": "vllm_fused_experts", "num_tokens": str(t),
                "ms_p50": "1.0", "model": "mixtral-8x7b", "dtype": dtype,
                "routing_kind": "uniform", "routing_param": "0", "seed": "0",
                "l2_flush": "True", "cuda_graph": "False",
                "correctness_passed": "True", "covers": "all",
                "load_active_experts": "8", "load_total_rows": str(2 * t),
                "gpu_name": gpu}

    h200 = roofline.load_hardware("measured_nvidia_h200")
    own = h200.ridge_point("bf16")
    failures: list[str] = []

    def check(cond: bool, what: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {what}")
        if not cond:
            failures.append(what)

    print("compare.py --self-test: the regime column's ridge")
    # rows/expert for mixtral = 2T/8 = T/4, so T = 4 * rows/expert.
    above, below = int(4 * own * 1.02), int(4 * own * 0.98)
    resolved = resolve_ridge([row(above, "NVIDIA H200"), row(below, "NVIDIA H200")], None)
    check(not resolved.refused and resolved.by_dtype.get("bf16") == own,
          f"rows naming the H200 resolve to its own yaml ridge {own:.1f} Op/B")
    check(abs(own - 166.0) > 1.0 and own not in (160.3, 176.2),  # retired; withdrawn
          "that ridge is neither the retired 166.0 default nor the withdrawn pair")
    check("measured_nvidia_h200" in resolved.source or "H200" in resolved.source,
          "the source line names the card the number came from")
    check(regime(own * 1.02, {"bf16"}, resolved) == "COMPUTE",
          "rows/expert 2% above the card's ridge reads COMPUTE")
    check(regime(own * 0.98, {"bf16"}, resolved) == "memory",
          "rows/expert 2% below the card's ridge reads memory")

    refused = resolve_ridge([row(above, "")], None)
    check(refused.refused and "--ridge" in refused.source,
          "rows naming no card REFUSE the column and name --ridge as the fix")
    check(regime(own * 1.02, {"bf16"}, refused) == NO_RIDGE,
          f"every regime cell then reads {NO_RIDGE!r}, never COMPUTE or memory")

    asserted = resolve_ridge([row(above, "")], 150.0)
    check(asserted.asserted and "ASSERTED" in asserted.source,
          "--ridge 150 is printed as an assertion, not a measurement")
    check(regime(151.0, {"bf16"}, asserted) == "COMPUTE"
          and regime(149.0, {"bf16"}, asserted) == "memory",
          "and the column is scored against 150, the asserted number")

    buf = io.StringIO()
    args = argparse.Namespace(routing="uniform", seed="0", l2_warm=False,
                              graph=False, metric="ms_p50", ridge=None, model=None)
    render([row(above, "NVIDIA H200"), row(below, "NVIDIA H200")], args, out=buf)
    text = buf.getvalue()
    check("regime column: ridge from the rows' own card" in text
          and " COMPUTE " in text and " memory " in text,
          "a rendered table carries the source line and both verdicts")
    buf = io.StringIO()
    render([row(above, ""), row(below, "")], args, out=buf)
    text = buf.getvalue()
    check("regime column: REFUSED" in text and text.count(NO_RIDGE) >= 3
          and "COMPUTE" not in text.split("regime column")[1].split("===")[1],
          "a rendered table with no card prints the refusal and no verdict")

    print("RESULT: compare --self-test " + ("PASS" if not failures else
                                            f"FAIL {len(failures)} check(s)"))
    return 0 if not failures else 1


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
    ap.add_argument("--ridge", type=float, default=None,
                    help="Op/B for the regime column, printed as YOUR assertion. "
                         "Without it the ridge is the committed calibration of "
                         "the card the rows name (peak(dtype)/bandwidth), and "
                         "with neither the column reads 'no-ridge'. It used to "
                         "default to 166.0 (retired), a figure belonging to "
                         "no card")
    ap.add_argument("--self-test", action="store_true",
                    help="plant the resolving, refusing and asserted ridge "
                         "paths off-GPU and exit non-zero if any misprints")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    rows = load(args.results)
    if not rows:
        print(f"no rows under {args.results}", file=sys.stderr)
        return 1
    sel = select(rows, args)
    if not sel:
        print("no rows match those conditions", file=sys.stderr)
        return 1
    render(sel, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
