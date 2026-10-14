#!/usr/bin/env python
"""The counter route's r3-decode family, with scripts/dram_counter_route.py byte-identical (pinned at
rental 7's sha 130d5dc...; the decode stage, 2026-10-14; design-decode sections 5 and 7).

    python scripts/dram_counter_route_decode.py --dry-run --family r3-decode --model M --histogram PAGE
    python scripts/dram_counter_route_decode.py --run --family r3-decode --model M --histogram PAGE \\
        --page-lock-mhz 1710 --out DIR/r3d.json

A wrapper in the tile stage's manner (dram_counter_route_tile.py): every piece of the pinned route a
decode page needs is IMPORTED and called, nothing is copied or edited there:
  probe_ncu("r3-arms")        the ncu binary and the metrics this card proves readable
  r3_ncu_argv(child=...)      ncu over scripts/r3_decode.py --counter-child (cold L2 per launch, the
                              GEMM launches only, kernel replay); clock control `none` under the
                              driver's nvidia-smi lock (the route's own --page-clock none reading)
  r3_import_argv, parse_ncu_csv, attribute_launches
                              the reduction and the EXACT launch-by-launch attribution: every launch's
                              launch__grid_size must equal the grid the child READ BACK off vLLM's
                              own launch (the spy), else the page is refused (V-ALIGN on counter pages)
The child (r3_decode.counter_child) times nothing: per cell its warmups, then CALLS calls, and a
manifest. The page (r3d.json) holds per cell and GEMM the median over calls of each metric, the
child's readback grid, and v4d's bytes beside the counted ones (BYTES-D, DESCRIPTIVE).
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
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r3_decode as RD  # noqa: E402

from moe.bench import exit_codes  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

FAMILY = "r3-decode"
CHILD = HERE / "r3_decode.py"
CALLS = 3
WARMUPS = 2
#: seconds per profiled launch under kernel replay with --cache-control all (rental 7's pages)
SEC_PER_LAUNCH = 0.6


def plan_of(model: str, page: RD.DecodePage, page_rel: str, *, profile_dir: Path, stem: str, calls: int = CALLS,
            warmups: int = WARMUPS, seed: int = 0) -> dict:
    cells = [[c.label, a] for c in page.ordered() for a in c.arms]
    plan = {"family": FAMILY, "model": model, "dtype": "bf16", "page": page_rel, "page_sha256": page.file_sha256,
            "cells": cells, "calls_per_cell": int(calls), "warmup_calls": int(warmups),
            "gemms_per_call": RD.PWR.GEMMS_PER_CALL, "seed": int(seed), "decl_copies": page.decl_copies,
            "manifest": str(Path(profile_dir) / f"{stem}.manifest.json")}
    RD.validate_counter_plan(plan, page)
    return plan


def launches(plan: dict) -> int:
    return len(plan["cells"]) * plan["calls_per_cell"] * plan["gemms_per_call"]


def reduce_cells(attributed: list[dict], manifest: dict, page: RD.DecodePage, model: str) -> list[dict]:
    """Per cell and GEMM: the median over its calls of every metric, and v4d's read bytes beside."""
    import r3_timing_model_v4d as V4D
    by = {}
    for a in attributed:
        if a["warmup"]:
            continue
        by.setdefault((a["key"], a["gemm"]), []).append(a["launch"].metrics)
    cells_by = {c.label: c for c in page.cells}
    out = []
    for arm, T, lab in manifest["order"]:
        key = f"{arm}/{T}/{lab}"
        c = cells_by[lab]
        try:
            pr = V4D.price(model, int(T), c.counts, V4D.DecodeParams(),
                           declared=None if arm == RD.NATIVE else MODEL_CONFIGS[model].num_experts * int(page.decl_copies or 1),
                           with_constants=False)
            v4d_reads = pr["reads"]
        except Exception as exc:                                       # noqa: BLE001
            v4d_reads = {"error": f"{type(exc).__name__}: {exc}"}
        per = {}
        for g in RD.GEMMS:
            ms = by.get((key, g), [])
            names = sorted({k for m in ms for k in m})
            per[g] = {k: statistics.median(m[k] for m in ms if k in m) for k in names}
            per[g]["calls"] = len(ms)
            per[g]["grid_readback"] = manifest["grids"][key][g]
        out.append({"key": key, "arm": arm, "T": int(T), "label": lab, "counts_sha256": RD.counts_sha256(c.counts),
                    "per_gemm": per, "v4d_reads": v4d_reads})
    return out


def run(args) -> int:
    import dram_counter_route as DCR
    cfg = MODEL_CONFIGS[args.model]
    page = RD.load_page(args.histogram, cfg, args.model)
    out = Path(args.out)
    prof = out.parent / f"{out.stem}-profile"
    prof.mkdir(parents=True, exist_ok=True)
    rel = str(Path(args.histogram))
    plan = plan_of(args.model, page, rel, profile_dir=prof, stem=out.stem, calls=args.calls, warmups=args.warmups)
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
                           child=CHILD, clock_control="none")
    t0 = time.time()
    rc, tail = DCR._r3_capture(argv, prof / f"{out.stem}.ncu.log", timeout=args.timeout_s,
                               stale=(Path(plan["manifest"]), report, prof / f"{out.stem}.csv", out))
    if rc != 0 or not Path(plan["manifest"]).exists():
        print(f"REFUSE: the capture exited {rc}: {tail[-300:]}")
        return exit_codes.INVALID
    manifest = json.loads(Path(plan["manifest"]).read_text())
    manifest["launch_count"] = None
    csv_text = DCR._r3_reduce(ncu["binary"], report, prof / f"{out.stem}.csv")
    parsed = DCR.parse_ncu_csv(csv_text, soft=frozenset(DCR.R3_CROSSCHECK_METRICS))
    try:
        attributed = DCR.attribute_launches(parsed, manifest)
    except DCR.CounterRunRefused as exc:
        print(f"RESULT: VALIDITY grid FAIL {exc}")
        return exit_codes.INVALID
    cells = reduce_cells(attributed, manifest, page, args.model)
    payload = {"family": FAMILY, "design": {"model": args.model, "page": rel, "page_sha256": page.file_sha256,
                                            "calls_per_cell": plan["calls_per_cell"], "warmup_calls": plan["warmup_calls"],
                                            "lock_mhz": args.page_lock_mhz, "clock_control": "none"},
               "ncu": {"binary": ncu["binary"], "version": ncu.get("version"), "argv": argv},
               "manifest": manifest, "cells": cells, "seconds": round(time.time() - t0, 1),
               "gates": {"V_ALIGN_grid": "PASS (every launch's grid is the child's readback)",
                         "refused_cells": manifest.get("refused") or []}}
    out.write_text(json.dumps(payload, indent=1, default=str))
    print(f"RESULT: VALIDITY grid PASS {len(cells)} cells, {len(attributed)} launches attributed; {out}")
    return exit_codes.DONE if not manifest.get("refused") else exit_codes.INVALID


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--family", required=True, choices=(FAMILY,))
    ap.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--histogram", type=Path, required=True)
    ap.add_argument("--calls", type=int, default=CALLS)
    ap.add_argument("--warmups", type=int, default=WARMUPS)
    ap.add_argument("--page-lock-mhz", type=int, default=None)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        cfg = MODEL_CONFIGS[args.model]
        page = RD.load_page(args.histogram, cfg, args.model)
        plan = plan_of(args.model, page, str(args.histogram), profile_dir=Path("/nonexistent"), stem="dry",
                       calls=args.calls, warmups=args.warmups)
    except RD.DecodeRefused as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    n = launches(plan)
    print(f"COUNTER ROUTE r3-decode: {args.model}, page {args.histogram} (sha256 {page.file_sha256[:16]}...), "
          f"{len(plan['cells'])} (cell, arm), {plan['calls_per_cell']} calls + {plan['warmup_calls']} warmups each, "
          f"{n} profiled fused_moe_kernel launches (the warmups too: no launch skip), about "
          f"{(n + len(plan['cells']) * plan['warmup_calls'] * 2) * SEC_PER_LAUNCH / 60:.1f} min")
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
