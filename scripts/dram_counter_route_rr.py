#!/usr/bin/env python
"""The counter route on rental RR's pages (registration 2026-11-11-rr-gh200), with scripts/dram_counter_route.py
byte-identical: a wrapper in the decode route's manner (dram_counter_route_decode.py, dram_counter_route_decode3.py).

    python scripts/dram_counter_route_rr.py --dry-run --page docs/registered/2026-11-11-rr-pages/rr-128.json
    python scripts/dram_counter_route_rr.py --run --page PAGE --page-lock-mhz 1410 --out DIR/rr-counter.json
    python scripts/dram_counter_route_rr.py --counter-child PLAN      # ncu's child (the pinned r3_ncu_argv's contract)

WHY A WRAPPER. The pinned route's pages take a model by name and R3's arms; RR's cells are synthetic shapes (rrs-) at
vLLM's prefill tiles with several M-tiles per expert, the same cells RR's stamps unit stamps. Every piece of the
existing counter tooling RR needs is IMPORTED from the pinned route and called, nothing there is copied or edited:
  probe_ncu("r3-arms"), counter_route_is_open, query_metric_names
                              the ncu binary, the metrics this card proves readable, and the page clock's metric
  r3_ncu_argv(child=...)      ncu over this file's --counter-child: cold L2 per launch (--cache-control all), the
                              fused_moe_kernel launches only, kernel replay, clock control `none` under the driver's
                              nvidia-smi lock (the route's own --page-clock none reading), the warm-ups skipped by
                              --launch-skip and exactly the measured launches kept by --launch-count (R3's pages' rule)
  _r3_capture, _r3_reduce, parse_ncu_csv, attribute_launches
                              the capture (stale files deleted first), the reduction (the .ncu-rep kept), the parse, and
                              the EXACT launch-by-launch attribution: every launch's launch__grid_size must equal the grid
                              the child READ BACK off vLLM's own launch (the spy), else the page is refused
THE CHILD times nothing: it builds each shape's weights and inputs exactly as RR's probe does (scripts/instr_probe_rr.py's
Card, the same seed, the same ids), runs every cell's readback (V-BIN and V-ALIGN at the cell's config, r3_decode's
rules; the readback call is the first warm-up) and its warm-ups, then every cell's measured calls, the plain vLLM kernel
under the probe's config override, and writes a manifest with each cell's grids. No kernel code: this chooses what ncu
profiles and on which inputs.

THE PAGE (DIR/rr-counter.json, moe-kernels/rr-counter-page/1): per cell and GEMM the median over its calls of each
metric, the per-call values, the ncu clock (sm__cycles_elapsed.avg over gpu__time_duration.sum, MHz, DESCRIPTIVE), the
readback grid. The reader (scripts/scoring/rr/read_rr.py) reads the w1 DRAM bytes against the cell's compulsory bytes.

EXIT CODES (moe/bench/exit_codes.py): 0 done; 2 refused (and --dry-run); 3 a readback or attribution failed (INVALID).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE), str(HERE / "scoring" / "rr"), str(HERE / "scoring" / "pc2"), str(HERE / "scoring" / "pc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rrcommon as P  # noqa: E402

from moe.bench import exit_codes  # noqa: E402

FAMILY = "rr-counter"
CHILD = HERE / "dram_counter_route_rr.py"
GEMMS = ("w1", "w2")
CALLS = P.COUNT_CALLS
WARMUPS = P.COUNT_WARMUPS
#: seconds per profiled prefill launch under kernel replay with --cache-control all (PC's Phi b128g1 page on the GH200:
#: 144 profiled launches in 298 s with its child start and nine-copy weight build), and the child's fixed start
SEC_PER_LAUNCH = 1.7
SETUP_S = 90.0
#: the metrics a page cannot do without (a missing value refuses the parse); every other asked metric is parsed soft
HARD_METRICS = frozenset({"dram__bytes_read.sum", "launch__grid_size", "gpu__time_duration.sum"})
PLAN_KEYS = ("family", "page", "page_sha256", "shuffle_seed", "cells", "calls_per_cell", "warmup_calls",
             "gemms_per_call", "seed", "manifest")


class Refused(P.Refused):
    pass


def _rel(path) -> Path:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise Refused(f"{path}: a repo-relative path")
    return REPO / p


def load_page(page) -> tuple[dict, list[dict], str]:
    """(the page document, its checked cells, its sha256): rrcommon.check_page_doc, as the probe checks it."""
    path = _rel(page)
    try:
        data = path.read_bytes()
        doc = json.loads(data)
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read the page {page}: {exc}") from None
    cells = P.check_page_doc(doc, P.load_shapes())
    return doc, cells, hashlib.sha256(data).hexdigest()


def order(cells: list[dict]) -> list[dict]:
    """Cells grouped by shape in the order each shape first appears, the page's order within a shape (the probe's
    order: one weight build per shape)."""
    first = {}
    for i, c in enumerate(cells):
        first.setdefault(c["shape"], i)
    return sorted(cells, key=lambda c: (first[c["shape"]], cells.index(c)))


def plan_of(page_rel: str, doc: dict, cells: list[dict], sha: str, *, profile_dir: Path, stem: str,
            calls: int = CALLS, warmups: int = WARMUPS, seed: int = 0) -> dict:
    plan = {"family": FAMILY, "page": str(page_rel), "page_sha256": sha, "shuffle_seed": int(doc["shuffle_seed"]),
            "cells": [c["label"] for c in order(cells)], "calls_per_cell": int(calls), "warmup_calls": int(warmups),
            "gemms_per_call": len(GEMMS), "seed": int(seed),
            "manifest": str(Path(profile_dir) / f"{stem}.manifest.json")}
    validate_plan(plan, cells)
    return plan


def validate_plan(plan: dict, cells: list[dict]) -> None:
    missing = [k for k in PLAN_KEYS if k not in plan]
    if missing:
        raise Refused(f"the counter plan lacks {missing}")
    if plan["family"] != FAMILY:
        raise Refused(f"family {plan['family']!r}: this child runs {FAMILY} plans only")
    if int(plan["gemms_per_call"]) != len(GEMMS):
        raise Refused(f"gemms_per_call {plan['gemms_per_call']}: vLLM makes {len(GEMMS)}")
    if int(plan["warmup_calls"]) < 1:
        raise Refused("warmup_calls >= 1: the spy's readback call is the first warm-up")
    if int(plan["calls_per_cell"]) < 2:
        raise Refused("calls_per_cell >= 2: the per-call spread rule needs two calls")
    by = {c["label"] for c in cells}
    bad = [lab for lab in plan["cells"] if lab not in by]
    if bad or len(set(plan["cells"])) != len(plan["cells"]):
        raise Refused(f"plan cells {bad or plan['cells']} are not the page's (or repeat)")


def window(plan: dict) -> tuple[int, int]:
    """(launch skip, launch count): every cell's warm-ups run first (both GEMMs each), then every cell's measured calls;
    ncu skips the first and keeps exactly the second (R3's pages' rule, counted on fused_moe_kernel launches only)."""
    n, g = len(plan["cells"]), int(plan["gemms_per_call"])
    return n * int(plan["warmup_calls"]) * g, n * int(plan["calls_per_cell"]) * g


def key_of(label: str) -> str:
    return f"rr/{label}"


# --------------------------------------------------------------------------
# the child (under ncu, on the GPU)
# --------------------------------------------------------------------------

def counter_card(args):
    """RR's probe's Card, with one weight build per shape kept for the page (every warm-up runs before any measured
    call, so every shape's weights stay resident; the probe's own build, seed and inputs otherwise)."""
    import instr_probe_rr as PR

    class CounterCard(PR.Card):
        def __init__(self, a):
            super().__init__(a)
            self._kept = {}

        def load_shape(self, name: str):
            key = f"shape:{name}"
            if self.model == key:
                return
            if isinstance(self.model, str) and self.model.startswith("shape:"):
                self._kept[self.model] = (self.w1, self.w2, self.x_all, self.cfg, self.inputs)
            if key in self._kept:
                self.w1, self.w2, self.x_all, self.cfg, self.inputs = self._kept[key]
                self.model = key
                return
            super().load_shape(name)
    return CounterCard(args), PR


def counter_child(plan_path: Path) -> int:
    """The planned calls, nothing timed: every cell's readback and warm-ups (plan order), then every cell's measured
    calls, and a manifest with each cell's grids READ BACK by the spy."""
    import r3_decode as RD
    plan = json.loads(Path(plan_path).read_text())
    doc, cells, sha = load_page(plan["page"])
    if sha != plan["page_sha256"]:
        raise Refused("the plan's page sha256 is not the page's")
    validate_plan(plan, cells)
    by = {c["label"]: c for c in cells}
    card, PR = counter_card(argparse.Namespace(seed=int(plan["seed"])))
    torch = card.torch
    v = {"shuffle_seed": int(plan["shuffle_seed"])}
    order_, grids, refused, built = [], {}, [], {}
    for lab in plan["cells"]:
        c = dict(by[lab], kind="decode", stem=lab, model=by[lab]["shape"], arm="native")
        cf = PR.conf_of(c["tile"])
        fn, ids = card.cell_fn(v, c)
        with card.override_config(cf):
            with RD.Spy(card.FM) as spy:
                fn()
                torch.cuda.synchronize()
        exp = RD.expected_readback(card.cfg, int(c["T"]), list(c["counts"]), ids.reshape(-1).tolist(),
                                   declared=card.cfg.num_experts, copies=1, conf=cf)
        bad = RD.check_readback({"gemms": spy.records}, exp)
        if bad:
            refused.append({"cell": key_of(lab), "why": bad})
            break
        order_.append(["rr", lab])
        grids[key_of(lab)] = {g: int(r["grid"]) for g, r in zip(GEMMS, spy.records)}
        with card.override_config(cf):
            for _ in range(int(plan["warmup_calls"]) - 1):
                fn()
            torch.cuda.synchronize()
        built[key_of(lab)] = (fn, cf)
    if not refused:
        for _arm, lab in order_:
            fn, cf = built[key_of(lab)]
            with card.override_config(cf):
                for _ in range(int(plan["calls_per_cell"])):
                    torch.cuda.nvtx.range_push(f"rr/{lab}")
                    fn()
                    torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
    skip, count = window(plan)
    manifest = {"family": FAMILY, "order": order_, "grids": grids, "refused": refused,
                "calls_per_cell": int(plan["calls_per_cell"]), "warmup_calls": int(plan["warmup_calls"]),
                "gemms_per_call": int(plan["gemms_per_call"]), "launch_count": count, "launch_skip": skip,
                "versions": PR.PW._package_versions(), "page_sha256": sha, "upstream": card.upstream,
                "override_hook": card.where}
    tmp = Path(plan["manifest"] + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, default=str))
    os.replace(tmp, plan["manifest"])
    print(f"counter child: {len(order_)} cells, {len(refused)} refused, manifest {plan['manifest']}")
    return exit_codes.DONE if not refused else exit_codes.INVALID


# --------------------------------------------------------------------------
# the parent (ncu, the reduction, the page)
# --------------------------------------------------------------------------

def clock_mhz(m: dict):
    cyc, ns = m.get("sm__cycles_elapsed.avg"), m.get("gpu__time_duration.sum")
    if cyc is None or ns is None or not ns > 0:
        return None
    return float(cyc) / float(ns) * 1e3


def reduce_cells(attributed: list[dict], manifest: dict, by: dict) -> list[dict]:
    """Per cell (manifest order) and GEMM: every metric's per-call values (call order) and their median, and the ncu
    clock per call; the measured launches only (the manifest keeps no warm-up in the profile)."""
    per = {}
    for rec in attributed:
        if rec["warmup"]:
            continue
        per.setdefault(rec["key"], {}).setdefault(rec["gemm"], {})[int(rec["call"])] = rec["launch"].metrics
    out = []
    for entry in manifest["order"]:
        lab = entry[1]
        key = key_of(lab)
        c = by[lab]
        row = {"label": lab, "key": key, "shape": c["shape"], "T": c["T"], "counts": c["counts"],
               "tile": P.tile_key(c["tile"]), "series": c.get("series"), "P": c.get("P"), "layout": c.get("layout"),
               "grid": manifest["grids"].get(key), "per_gemm": {}, "per_call": {}, "clock_mhz": {}}
        for g in GEMMS:
            calls = per.get(key, {}).get(g, {})
            ms = [calls[k] for k in sorted(calls)]
            names = sorted({n for m in ms for n in m})
            row["per_call"][g] = {n: [m.get(n) for m in ms] for n in names}
            row["per_gemm"][g] = {n: (statistics.median([x for x in row["per_call"][g][n] if x is not None])
                                      if any(x is not None for x in row["per_call"][g][n]) else None) for n in names}
            row["per_gemm"][g]["calls"] = len(ms)
            row["clock_mhz"][g] = [clock_mhz(m) for m in ms]
        out.append(row)
    return out


def run(args) -> int:
    import dram_counter_route as DCR
    doc, cells, sha = load_page(args.page)
    by = {c["label"]: c for c in cells}
    out = Path(args.out)
    prof = out.parent / f"{out.stem}-profile"
    prof.mkdir(parents=True, exist_ok=True)
    plan = plan_of(str(args.page), doc, cells, sha, profile_dir=prof, stem=out.stem, calls=args.calls,
                   warmups=args.warmups)
    plan_path = prof / f"{out.stem}.plan.json"
    plan_path.write_text(json.dumps(plan, indent=1))
    ncu = DCR.probe_ncu(DCR.R3_FAMILY, clock_control="none")
    if not DCR.counter_route_is_open(ncu):
        print(f"REFUSE: the counter route is not open on this box ({ncu.get('why') or ncu})")
        return exit_codes.REFUSED
    metrics = tuple(ncu.get("metrics_proven") or DCR.R3_STRICT_METRICS)
    if args.page_lock_mhz is not None:
        names, query = DCR.query_metric_names(ncu["binary"])
        if names is None or DCR.metric_base(DCR.R3_PAGE_CLOCK_METRIC) not in names:
            print(f"REFUSE: --page-lock-mhz asks each launch's {DCR.R3_PAGE_CLOCK_METRIC} and this box's metric list "
                  f"does not show it ({query}); nothing was captured")
            return exit_codes.REFUSED
        metrics += tuple(m for m in (DCR.R3_PAGE_CLOCK_METRIC,) if m not in metrics)
    skip, count = window(plan)
    report = prof / f"{out.stem}.ncu-rep"
    argv = DCR.r3_ncu_argv(ncu["binary"], plan_path, report, metrics, launch_skip=skip, launch_count=count,
                           child=CHILD, clock_control="none", kernel_filter=DCR.R3_KERNEL_FILTER)
    t0 = time.time()
    rc, tail = DCR._r3_capture(argv, prof / f"{out.stem}.ncu.log", timeout=args.timeout_s,
                               stale=(Path(plan["manifest"]), report, prof / f"{out.stem}.csv", out))
    manifest = json.loads(Path(plan["manifest"]).read_text()) if Path(plan["manifest"]).exists() else None
    if manifest is not None and manifest.get("refused"):   # the child exits 3 (INVALID) on a failed readback
        r = manifest["refused"][0]
        print(f"RESULT: VALIDITY readback FAIL {r['cell']} (child exit {rc}, after {len(manifest.get('order') or [])} "
              f"cells read back; no measured call ran): {r['why'][:3]}; the cell's V-BIN / V-ALIGN mismatch is vLLM's "
              "readback against r3_decode's expected geometry at the cell's config (a page or a vLLM change, not a "
              "count)")
        return exit_codes.INVALID
    if rc != 0 or manifest is None:
        print(f"REFUSE: the capture exited {rc}: {tail[-300:]}")
        return exit_codes.INVALID
    csv_text = DCR._r3_reduce(ncu["binary"], report, prof / f"{out.stem}.csv")
    parsed = DCR.parse_ncu_csv(csv_text, soft=frozenset(set(metrics) - HARD_METRICS))
    gemm = [ln for ln in parsed if DCR.GEMM_MARKER in ln.kernel]
    try:
        attributed = DCR.attribute_launches(gemm, manifest)
    except DCR.CounterRunRefused as exc:
        print(f"RESULT: VALIDITY grid FAIL {exc}")
        return exit_codes.INVALID
    payload = {"schema": P.COUNTER_SCHEMA, "family": FAMILY, "registration": P.NAME,
               "design": {"page": str(args.page), "page_sha256": sha, "calls_per_cell": plan["calls_per_cell"],
                          "warmup_calls": plan["warmup_calls"], "lock_mhz": args.page_lock_mhz, "clock_control": "none",
                          "cache_control": "all", "replay": "kernel", "launch_skip": skip, "launch_count": count,
                          "kernel_filter": DCR.R3_KERNEL_FILTER, "metrics": list(metrics)},
               "ncu": {"binary": ncu["binary"], "version": ncu.get("version"), "argv": argv},
               "manifest": manifest, "versions": manifest.get("versions"),
               "cells": reduce_cells(attributed, manifest, by), "seconds": round(time.time() - t0, 1),
               "gates": {"V_ALIGN_grid": "PASS (every fused_moe_kernel launch's grid is the child's readback)",
                         "complete": len(manifest["order"]) == len(plan["cells"])}}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    os.replace(tmp, out)
    print(f"RESULT: VALIDITY grid PASS {len(payload['cells'])} cells, {len(attributed)} launches attributed; {out}")
    return exit_codes.DONE


def plan_lines(page, cells: list[dict], sha: str, calls: int, warmups: int) -> list[str]:
    n = len(cells)
    prof = n * calls * len(GEMMS)
    shapes = sorted({c["shape"] for c in cells})
    return [f"COUNTER ROUTE {FAMILY}: page {page} (sha256 {sha[:16]}...), {n} cells on "
            f"{sorted({P.tile_key(c['tile']) for c in cells})}, {len(shapes)} shapes, NATIVE, one weight copy each",
            f"  {warmups} warm-up calls per cell (the readback first, skipped by ncu), then {calls} measured calls: "
            f"{prof} profiled launches (w1 and w2), about {(SETUP_S + prof * SEC_PER_LAUNCH) / 60:.1f} min",
            "  ncu over the pinned route's r3_ncu_argv: --cache-control all, --clock-control none under the driver's "
            "lock, -k regex:^fused_moe_kernel$, kernel replay; the grids held to the child's readback "
            "(attribute_launches)",
            f"  child: {CHILD.relative_to(REPO)} --counter-child (RR's probe's Card and inputs, the plain kernel)"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--counter-child", type=Path, default=None)
    ap.add_argument("--page", default=None)
    ap.add_argument("--calls", type=int, default=CALLS)
    ap.add_argument("--warmups", type=int, default=WARMUPS)
    ap.add_argument("--page-lock-mhz", type=int, default=None)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.counter_child is not None:
        try:
            return counter_child(args.counter_child)
        except P.Refused as exc:
            print(f"REFUSED: {exc}")
            return exit_codes.REFUSED
    if not args.page:
        print("REFUSED: --page <an RR page> is required")
        return exit_codes.REFUSED
    try:
        doc, cells, sha = load_page(args.page)
        plan_of(str(args.page), doc, cells, sha, profile_dir=Path("/nonexistent"), stem="dry", calls=args.calls,
                warmups=args.warmups)
    except P.Refused as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    print("\n".join(plan_lines(args.page, cells, sha, args.calls, args.warmups)))
    if args.dry_run:
        print("REFUSED. Nothing was measured: --dry-run was given")
        return exit_codes.REFUSED
    if args.out is None:
        print("REFUSED: --run needs --out")
        return exit_codes.REFUSED
    try:
        return run(args)
    except P.Refused as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    except Exception as exc:                                      # noqa: BLE001  (never exit 1)
        print(f"CRASHED: {type(exc).__name__}: {exc}"[:400])
        return exit_codes.ERROR


if __name__ == "__main__":
    sys.exit(main())
