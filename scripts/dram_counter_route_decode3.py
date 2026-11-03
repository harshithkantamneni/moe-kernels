#!/usr/bin/env python
"""The decode counter route with a wider kernel filter (registration 2026-10-29-decode-addendum3-gh200,
settling unit M2): scripts/dram_counter_route_decode.py and scripts/dram_counter_route.py, both byte-identical,
imported and called.

    python scripts/dram_counter_route_decode3.py --dry-run --family r3-decode --model M --histogram PAGE --kernels wide
    python scripts/dram_counter_route_decode3.py --run --family r3-decode --model M --histogram PAGE --kernels wide \\
        --page-lock-mhz 1710 --out DIR/r3d.json

--kernels wide  ncu's `-k` filter widened from ^fused_moe_kernel$ to the decode call's other kernels too:
                vLLM's align kernels (moe_align_block_size*, count_and_sort_expert_tokens*), the activation
                (act_and_mul*, silu*) and the reduction (moe_sum*). Same replay (kernel), same --cache-control
                all (every kernel starts at a cold L2), same metrics, same child, same lock.
The GEMM launches are attributed EXACTLY by the pinned route (attribute_launches: count and grid held to the
child's readback); every other launch is attributed to a (cell, call) by its place between them
(d3common.split_wide). The page (r3d.json) is the decode route's, plus per cell `per_class` (align,
silu_and_mul, moe_sum, other: the median over calls of each metric summed over that class's launches in a call)
and `kernels` (the names seen). No kernel code: this chooses what ncu profiles.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE), str(HERE / "scoring" / "decode")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dram_counter_route_decode as DRD  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

RD = DRD.RD
WIDE_FILTER = ("regex:^(fused_moe_kernel|moe_align_block_size\\w*|count_and_sort_expert_tokens\\w*|"
               "act_and_mul\\w*|silu\\w*|moe_sum\\w*)$")
KERNELS = ("gemm", "wide")
CLASSES = ("align", "silu_and_mul", "moe_sum", "other")


def kernel_class(name: str) -> str:
    import d3common as D3
    return D3.kernel_class(name)


def reduce_wide(others: list[dict], manifest: dict) -> dict:
    """{cell key: {"per_class": {class: {metric: median over calls of the per-call class sum}}, "kernels": [...]}}
    over the measured (non-warmup) calls."""
    by, names = {}, {}
    for o in others:
        if o["warmup"]:
            continue
        cl = kernel_class(o["kernel"])
        names.setdefault(o["key"], set()).add(o["kernel"])
        d = by.setdefault(o["key"], {}).setdefault(cl, {}).setdefault(o["call"], {})
        for k, v in o["metrics"].items():
            d[k] = d.get(k, 0.0) + float(v)
    out = {}
    for arm, T, lab in manifest["order"]:
        key = f"{arm}/{T}/{lab}"
        pc = {}
        for cl, calls in (by.get(key) or {}).items():
            ms = list(calls.values())
            keys = sorted({k for m in ms for k in m})
            pc[cl] = {k: statistics.median(m[k] for m in ms if k in m) for k in keys}
            pc[cl]["calls"] = len(ms)
        out[key] = {"per_class": pc, "kernels": sorted(names.get(key) or [])}
    return out


def run(args) -> int:
    import d3common as D3
    import dram_counter_route as DCR
    cfg = MODEL_CONFIGS[args.model]
    page = RD.load_page(args.histogram, cfg, args.model)
    out = Path(args.out)
    prof = out.parent / f"{out.stem}-profile"
    prof.mkdir(parents=True, exist_ok=True)
    rel = str(Path(args.histogram))
    plan = DRD.plan_of(args.model, page, rel, profile_dir=prof, stem=out.stem, calls=args.calls, warmups=args.warmups)
    plan_path = prof / f"{out.stem}.plan.json"
    plan_path.write_text(json.dumps(plan, indent=1))
    ncu = DCR.probe_ncu(DCR.R3_FAMILY, clock_control="none")
    if not DCR.counter_route_is_open(ncu):
        print(f"REFUSE: the counter route is not open on this box ({ncu.get('why') or ncu})")
        return exit_codes.REFUSED
    metrics = [m for m in DCR.R3_STRICT_METRICS] + [m for m in DCR.R3_CROSSCHECK_METRICS
                                                     if m in (ncu.get("metrics_proven") or ())]
    report = prof / f"{out.stem}.ncu-rep"
    argv = DCR.r3_ncu_argv(ncu["binary"], plan_path, report, metrics, launch_skip=0, launch_count=None,
                           child=DRD.CHILD, clock_control="none", kernel_filter=WIDE_FILTER)
    t0 = time.time()
    rc, tail = DCR._r3_capture(argv, prof / f"{out.stem}.ncu.log", timeout=args.timeout_s,
                               stale=(Path(plan["manifest"]), report, prof / f"{out.stem}.csv", out))
    if rc != 0 or not Path(plan["manifest"]).exists():
        print(f"REFUSE: the capture exited {rc}: {tail[-300:]}")
        return exit_codes.INVALID
    manifest = json.loads(Path(plan["manifest"]).read_text())
    manifest["launch_count"] = None
    csv_text = DCR._r3_reduce(ncu["binary"], report, prof / f"{out.stem}.csv")
    soft = frozenset(set(metrics) - {"gpu__time_duration.sum", "launch__grid_size"})
    parsed = DCR.parse_ncu_csv(csv_text, soft=soft)
    gemm = [ln for ln in parsed if DCR.GEMM_MARKER in ln.kernel]
    try:
        attributed = DCR.attribute_launches(gemm, manifest)
    except DCR.CounterRunRefused as exc:
        print(f"RESULT: VALIDITY grid FAIL {exc}")
        return exit_codes.INVALID
    others = D3.split_wide(parsed, attributed)
    cells = DRD.reduce_cells(attributed, manifest, page, args.model)
    wide = reduce_wide(others, manifest)
    for c in cells:
        c.update(wide.get(c["key"]) or {"per_class": {}, "kernels": []})
    payload = {"family": DRD.FAMILY, "kernels": "wide", "kernel_filter": WIDE_FILTER,
               "design": {"model": args.model, "page": rel, "page_sha256": page.file_sha256,
                          "calls_per_cell": plan["calls_per_cell"], "warmup_calls": plan["warmup_calls"],
                          "lock_mhz": args.page_lock_mhz, "clock_control": "none", "cache_control": "all",
                          "replay": "kernel"},
               "ncu": {"binary": ncu["binary"], "version": ncu.get("version"), "argv": argv},
               "manifest": manifest, "cells": cells, "seconds": round(time.time() - t0, 1),
               "unattributed": len(parsed) - len(gemm) - len(others),
               "gates": {"V_ALIGN_grid": "PASS (every GEMM launch's grid is the child's readback)",
                         "refused_cells": manifest.get("refused") or []}}
    out.write_text(json.dumps(payload, indent=1, default=str))
    print(f"RESULT: VALIDITY grid PASS {len(cells)} cells, {len(attributed)} GEMM launches attributed, "
          f"{len(others)} other launches by place; {out}")
    return exit_codes.DONE if not manifest.get("refused") else exit_codes.INVALID


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--family", required=True, choices=(DRD.FAMILY,))
    ap.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--histogram", type=Path, required=True)
    ap.add_argument("--kernels", required=True, choices=KERNELS)
    ap.add_argument("--calls", type=int, default=DRD.CALLS)
    ap.add_argument("--warmups", type=int, default=DRD.WARMUPS)
    ap.add_argument("--page-lock-mhz", type=int, default=None)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.kernels == "gemm":
        rest = [x for x in (argv if argv is not None else sys.argv[1:])]
        i = rest.index("--kernels")
        return DRD.main(rest[:i] + rest[i + 2:])
    try:
        cfg = MODEL_CONFIGS[args.model]
        page = RD.load_page(args.histogram, cfg, args.model)
        plan = DRD.plan_of(args.model, page, str(args.histogram), profile_dir=Path("/nonexistent"), stem="dry",
                           calls=args.calls, warmups=args.warmups)
    except RD.DecodeRefused as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    n = len(plan["cells"]) * (plan["calls_per_cell"] + plan["warmup_calls"]) * 6
    print(f"COUNTER ROUTE r3-decode (wide): {args.model}, page {args.histogram} (sha256 {page.file_sha256[:16]}...), "
          f"{len(plan['cells'])} (cell, arm), {plan['calls_per_cell']} calls + {plan['warmup_calls']} warmups each, "
          f"at most {n} profiled launches (the GEMMs, the align, activation and reduction kernels), about "
          f"{n * DRD.SEC_PER_LAUNCH / 60:.1f} min")
    print(f"  kernel filter: {WIDE_FILTER}")
    print("  child: scripts/r3_decode.py --counter-child (vLLM's own tile, the spy's grids in the manifest)")
    if args.dry_run:
        print("REFUSED. Nothing was measured: --dry-run was given")
        return exit_codes.REFUSED
    if args.out is None:
        print("REFUSED: --run needs --out")
        return exit_codes.REFUSED
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
