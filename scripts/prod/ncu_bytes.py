"""extras.sh X2: the study's counter route (ncu DRAM bytes per GEMM) on a stratified subsample of main's replay C sample,
real ids, at the branch clock. scripts/dram_counter_route.py is imported and called, never copied or edited, in the
manner of scripts/dram_counter_route_rr.py (rental RR's wrapper).

    python scripts/prod/ncu_bytes.py subsample --model M --sample decode=R/sample-decode/sample.json \\
        --sample prefill=R/sample-prefill/sample.json --n-per-branch 30 --out EX/ncu-sample
    python scripts/prod/ncu_bytes.py run --model M --sample EX/ncu-sample/sample.json --branch decode \\
        --lock-mhz 1710 --profile-dir TRACES/ncu-decode --out EX/ncu-decode/ncu_bytes.json     # under kit_lock 1710
    python scripts/prod/ncu_bytes.py plan --model M --sample EX/ncu-sample/sample.json --branch decode
    python scripts/prod/ncu_bytes.py --counter-child PLAN                # ncu's child (r3_ncu_argv's contract)

THE SUBSAMPLE (registered here; `subsample`): per branch an equal quota per tile present in main's replay sample
(n // tiles, the remainder to the tiles in prodlib.TILE_BINS order), inside a tile the sample's cells in (step, layer)
order taken systematically from a seeded start (seed 20261119); synthetic cells are never taken. Written in replay.py's
sample schema (sample.json + sample_ids.npz, each cell's ids copied from main's sample), so the scorer prices the same
calls. Each cell is relabelled <branch>_<label> (the two samples come from two runs whose g<index> labels can
coincide); `src_label` and `g` keep its run's own.

THE ROUTE (dram_counter_route.py, imported): probe_ncu("r3-arms", clock_control "none") and counter_route_is_open (the
ncu binary and the metrics this card proves readable), r3_ncu_argv(child=this file) (ncu over the child: --cache-control
all, every fused_moe_kernel launch starts at a cold L2; -k regex:^fused_moe_kernel$; kernel replay; --clock-control
none under the driver's nvidia-smi lock, the page clock metric asked; the warm-ups skipped by --launch-skip and exactly
the measured launches kept by --launch-count), _r3_capture, _r3_reduce, parse_ncu_csv, attribute_launches (EXACT
launch-by-launch attribution: every launch's launch__grid_size must equal the grid the child READ BACK off vLLM's own
launch with r3_decode's Spy, else the page is refused).

THE CHILD times nothing. It builds the model's weights once (private_weight_reference.build_private_weights, bf16, one
copy, seed 0) and random hidden states, and per cell calls vLLM 0.27.1's functional fused_experts (replay C's call:
align .. moe_sum) on the cell's REAL ids (weights 1/k): the first warm-up under the Spy (its two grids recorded; V-ALIGN
/ GD5 against r3_decode.expected_readback recorded as `readback`, a mismatch kept and flagged, never a silent drop: the
grids ncu is held to are the launched ones), the other warm-ups, then every cell's measured calls. Per cell and GEMM
the page (moe-kernels/prod-ncu-bytes/1) holds every metric's per-call values and median: dram__bytes_read.sum is the
counted w1 / w2 DRAM read bytes of the call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prodlib as PL  # noqa: E402  (puts the repo and scripts/ on sys.path)

FAMILY = "prod-ncu-bytes"
PAGE_SCHEMA = "moe-kernels/prod-ncu-bytes/1"
SAMPLE_SCHEMA = "moe-kernels/prod-replay-sample/1"          # replay.py's
SEED = 20261119
GEMMS = ("w1", "w2")
CALLS = 3
WARMUPS = 2
N_PER_BRANCH = 30
#: seconds per profiled launch under kernel replay with --cache-control all (dram_counter_route_rr.SEC_PER_LAUNCH, PC's
#: prefill page on the GH200) and the child's fixed start (vLLM import, the weight build, the Spy readbacks)
SEC_PER_LAUNCH = 1.7
SETUP_S = 120.0
HARD_METRICS = frozenset({"dram__bytes_read.sum", "launch__grid_size", "gpu__time_duration.sum"})
CHILD = Path(__file__).resolve()


class Refused(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------
# the subsample (pure)
# ------------------------------------------------------------------------------------------------------------

def pick(cells: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """The registered rule (module doc) on one branch's cells."""
    real = [c for c in cells if not c.get("synthetic") and c.get("g") is not None]
    by: dict = defaultdict(list)
    for c in real:
        by[c["tile"]].append(c)
    tiles = [lab for lab, _lo, _hi in PL.TILE_BINS if by.get(lab)]
    if not tiles:
        return []
    base, extra = divmod(int(n), len(tiles))
    rng = random.Random(int(seed))
    out = []
    for i, lab in enumerate(tiles):
        pool = sorted(by[lab], key=lambda c: (c.get("step") or 0, c.get("layer") or 0, int(c["g"])))
        q = min(len(pool), base + (1 if i < extra else 0))
        if q <= 0:
            continue
        stride = len(pool) / q
        start = rng.random() * stride
        seen = set()
        for j in range(q):
            c = pool[min(len(pool) - 1, int(start + j * stride))]
            if c["label"] not in seen:
                seen.add(c["label"])
                out.append(c)
    return out


def subsample(samples: dict, model_key: str, n_per_branch: int, out_dir, seed: int = SEED) -> Path:
    """samples: branch -> main's sample.json. Writes out_dir/{sample.json, sample_ids.npz} (replay.py's schema)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells, ids, src = [], {}, {}
    for b, path in sorted(samples.items()):
        p = Path(path)
        doc = json.loads(p.read_text())
        if doc.get("schema") != SAMPLE_SCHEMA or doc.get("model") != model_key:
            raise Refused(f"{p}: schema {doc.get('schema')!r} / model {doc.get('model')!r}")
        idf = p.parent / doc["ids_file"]
        if hashlib.sha256(idf.read_bytes()).hexdigest() != doc["ids_sha256"]:
            raise Refused(f"{idf}: sha256 differs from {p.name}'s record")
        got = pick([c for c in doc["cells"] if c.get("branch") == b], n_per_branch, seed)
        with np.load(idf) as z:
            for c in got:
                # main's two samples come from two runs (A1710, A1410), whose g<index> labels can coincide: each
                # cell is relabelled <branch>_<label>, its run's own label kept as src_label
                lab = f"{b}_{c['label']}"
                ids[lab] = np.asarray(z[c["label"]], dtype=np.int16)
                cells.append(dict(c, label=lab, src_label=c["label"], src_run=doc.get("run")))
        src[b] = {"sample": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "picked": len(got)}
    if not cells:
        raise Refused("no cell to profile")
    np.savez_compressed(out / "sample_ids.npz", **ids)
    doc = {"schema": SAMPLE_SCHEMA, "model": model_key, "seed": seed, "n_cells": len(cells), "run": None,
           "ids_file": "sample_ids.npz", "budget": None, "subsample_of": src, "rule": "ncu_bytes.pick",
           "ids_sha256": hashlib.sha256((out / "sample_ids.npz").read_bytes()).hexdigest(), "cells": cells}
    (out / "sample.json").write_text(json.dumps(doc, indent=1))
    return out / "sample.json"


def load_cells(sample, model_key: str, branch: str | None) -> tuple[list[dict], dict, str]:
    """(cells of the branch, {label: ids [T, k] int64}, the sample's sha256); V-BIN on load."""
    p = Path(sample)
    data = p.read_bytes()
    doc = json.loads(data)
    if doc.get("schema") != SAMPLE_SCHEMA or doc.get("model") != model_key:
        raise Refused(f"{p}: schema {doc.get('schema')!r} / model {doc.get('model')!r}")
    cfg = PL.MODELS[model_key].cfg
    idf = p.parent / doc["ids_file"]
    if hashlib.sha256(idf.read_bytes()).hexdigest() != doc["ids_sha256"]:
        raise Refused(f"{idf}: sha256 differs from the sample's record")
    cells, ids = [], {}
    with np.load(idf) as z:
        for c in doc["cells"]:
            if branch and c["branch"] != branch:
                continue
            a = np.asarray(z[c["label"]], dtype=np.int64)
            if a.shape != (int(c["T"]), cfg.top_k) or PL.counts_from_ids(a, cfg.num_experts) != list(c["counts"]):
                raise Refused(f"cell {c['label']} (V-BIN): ids {a.shape} do not give the cell's T and counts")
            cells.append(c)
            ids[c["label"]] = a
    return cells, ids, hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------------------------
# the plan and the window (pure)
# ------------------------------------------------------------------------------------------------------------

def key_of(label: str) -> str:
    return f"ncu/{label}"


def plan_of(sample, model_key: str, branch: str, *, profile_dir, stem: str, calls: int = CALLS,
            warmups: int = WARMUPS, seed: int = 0) -> dict:
    cells, _ids, sha = load_cells(sample, model_key, branch)
    if not cells:
        raise Refused(f"{sample}: no {branch} cell")
    if int(warmups) < 1 or int(calls) < 1:
        raise Refused("warmups >= 1 (the Spy's readback is the first) and calls >= 1")
    return {"family": FAMILY, "model": model_key, "branch": branch, "sample": str(Path(sample).resolve()),
            "sample_sha256": sha, "cells": [c["label"] for c in cells], "calls_per_cell": int(calls),
            "warmup_calls": int(warmups), "gemms_per_call": len(GEMMS), "seed": int(seed),
            "manifest": str(Path(profile_dir) / f"{stem}.manifest.json")}


def window(plan: dict) -> tuple[int, int]:
    """(launch skip, launch count): every cell's warm-ups first (both GEMMs each), then every cell's measured calls."""
    n, g = len(plan["cells"]), int(plan["gemms_per_call"])
    return n * int(plan["warmup_calls"]) * g, n * int(plan["calls_per_cell"]) * g


def expected_s(n_cells: int, calls: int = CALLS) -> float:
    return SETUP_S + n_cells * calls * len(GEMMS) * SEC_PER_LAUNCH


# ------------------------------------------------------------------------------------------------------------
# the child (under ncu, on the GPU)
# ------------------------------------------------------------------------------------------------------------

def counter_child(plan_path) -> int:
    import private_weight_reference as PWR
    import r3_decode as RD
    import torch
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("family") != FAMILY:
        raise Refused(f"family {plan.get('family')!r}")
    cells, ids_np, sha = load_cells(plan["sample"], plan["model"], plan["branch"])
    if sha != plan["sample_sha256"]:
        raise Refused("the plan's sample sha256 is not the sample's")
    by = {c["label"]: c for c in cells}
    cfg = PL.MODELS[plan["model"]].cfg
    w1, w2, _b = PWR.build_private_weights(cfg, "bf16", 1, int(plan["seed"]))
    gen = torch.Generator(device="cuda").manual_seed(int(plan["seed"]) + 1)
    x_all = torch.randn((max(int(by[lab]["T"]) for lab in plan["cells"]), cfg.hidden_size), device="cuda",
                        dtype=w1.dtype, generator=gen)
    act = MoEActivation("silu")
    order, grids, readback, built = [], {}, {}, {}
    with torch.inference_mode():
        for lab in plan["cells"]:
            c = by[lab]
            T = int(c["T"])
            ids = torch.as_tensor(ids_np[lab], dtype=torch.int32).to("cuda").contiguous()
            w = torch.full((T, cfg.top_k), 1.0 / cfg.top_k, dtype=torch.float32, device="cuda")
            x = x_all[:T].contiguous()

            def call(x=x, w=w, ids=ids):
                return fused_experts(hidden_states=x, w1=w1, w2=w2, topk_weights=w, topk_ids=ids, activation=act,
                                     apply_router_weight_on_input=False, global_num_experts=cfg.num_experts,
                                     expert_map=None)
            with RD.Spy() as spy:
                call()
                torch.cuda.synchronize()
            if len(spy.records) != len(GEMMS) or any("grid" not in r for r in spy.records):
                raise Refused(f"cell {lab}: the spy saw {len(spy.records)} fused_moe_kernel launches, not 2")
            exp = RD.expected_readback(cfg, T, list(c["counts"]), ids_np[lab].reshape(-1).tolist(),
                                       declared=cfg.num_experts, copies=1)
            bad = RD.check_readback({"gemms": spy.records}, exp)
            readback[key_of(lab)] = {"verdict": "PASS" if not bad else "MISMATCH", "why": bad[:4]}
            grids[key_of(lab)] = {g: int(r["grid"]) for g, r in zip(GEMMS, spy.records, strict=True)}
            order.append(["ncu", lab])
            for _ in range(int(plan["warmup_calls"]) - 1):
                call()
            torch.cuda.synchronize()
            built[lab] = call
        for lab in plan["cells"]:
            for _ in range(int(plan["calls_per_cell"])):
                torch.cuda.nvtx.range_push(key_of(lab))
                built[lab]()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
    skip, count = window(plan)
    manifest = {"family": FAMILY, "order": order, "grids": grids, "readback": readback, "refused": [],
                "calls_per_cell": int(plan["calls_per_cell"]), "warmup_calls": int(plan["warmup_calls"]),
                "gemms_per_call": int(plan["gemms_per_call"]), "launch_count": count, "launch_skip": skip,
                "sample_sha256": sha, "weights": "private_weight_reference.build_private_weights bf16, 1 copy"}
    tmp = Path(plan["manifest"] + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, default=str))
    os.replace(tmp, plan["manifest"])
    print(f"counter child: {len(order)} cells, readback mismatches "
          f"{sum(1 for v in readback.values() if v['verdict'] != 'PASS')}, manifest {plan['manifest']}")
    return 0


# ------------------------------------------------------------------------------------------------------------
# the parent (ncu, the reduction, the page)
# ------------------------------------------------------------------------------------------------------------

def clock_mhz(m: dict):
    cyc, ns = m.get("sm__cycles_elapsed.avg"), m.get("gpu__time_duration.sum")
    if cyc is None or ns is None or not ns > 0:
        return None
    return float(cyc) / float(ns) * 1e3


def reduce_cells(attributed: list[dict], manifest: dict, by: dict) -> list[dict]:
    """Per cell (manifest order) and GEMM: every metric's per-call values and their median, the ncu clock per call."""
    per: dict = {}
    for rec in attributed:
        if rec["warmup"]:
            continue
        per.setdefault(rec["key"], {}).setdefault(rec["gemm"], {})[int(rec["call"])] = rec["launch"].metrics
    out = []
    for entry in manifest["order"]:
        lab = entry[1]
        key = key_of(lab)
        c = by[lab]
        row = {"label": lab, "g": c.get("g"), "T": c["T"], "tile": c["tile"], "branch": c["branch"],
               "layer": c.get("layer"), "step": c.get("step"), "ids_digest": c.get("ids_digest"),
               "grid": manifest["grids"].get(key), "readback": (manifest.get("readback") or {}).get(key),
               "per_gemm": {}, "per_call": {}, "clock_mhz": {}}
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
    cells, _ids, sha = load_cells(args.sample, args.model, args.branch)
    by = {c["label"]: c for c in cells}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prof = Path(args.profile_dir or (out.parent / f"{out.stem}-profile"))
    prof.mkdir(parents=True, exist_ok=True)
    plan = plan_of(args.sample, args.model, args.branch, profile_dir=prof, stem=out.stem, calls=args.calls,
                   warmups=args.warmups)
    plan_path = prof / f"{out.stem}.plan.json"
    plan_path.write_text(json.dumps(plan, indent=1))
    ncu = DCR.probe_ncu(DCR.R3_FAMILY, clock_control="none")
    if not DCR.counter_route_is_open(ncu):
        print(f"FAIL ncu-bytes: the counter route is not open on this box ({ncu.get('why') or ncu})")
        return 3
    metrics = tuple(ncu.get("metrics_proven") or DCR.R3_STRICT_METRICS)
    if args.lock_mhz is not None:
        names, query = DCR.query_metric_names(ncu["binary"])
        if names is None or DCR.metric_base(DCR.R3_PAGE_CLOCK_METRIC) not in names:
            print(f"FAIL ncu-bytes: this box's metric list does not show {DCR.R3_PAGE_CLOCK_METRIC} ({query})")
            return 3
        metrics += tuple(m for m in (DCR.R3_PAGE_CLOCK_METRIC,) if m not in metrics)
    skip, count = window(plan)
    report = prof / f"{out.stem}.ncu-rep"
    argv = DCR.r3_ncu_argv(ncu["binary"], plan_path, report, metrics, launch_skip=skip, launch_count=count,
                           child=CHILD, clock_control="none", kernel_filter=DCR.R3_KERNEL_FILTER)
    t0 = time.time()
    rc, tail = DCR._r3_capture(argv, prof / f"{out.stem}.ncu.log", timeout=args.timeout_s,
                               stale=(Path(plan["manifest"]), report, prof / f"{out.stem}.csv", out))
    mpath = Path(plan["manifest"])
    manifest = json.loads(mpath.read_text()) if mpath.exists() else None
    if rc != 0 or manifest is None:
        print(f"FAIL ncu-bytes: the capture exited {rc}: {tail[-300:]}")
        return 3
    csv_text = DCR._r3_reduce(ncu["binary"], report, prof / f"{out.stem}.csv")
    parsed = DCR.parse_ncu_csv(csv_text, soft=frozenset(set(metrics) - HARD_METRICS))
    gemm = [ln for ln in parsed if DCR.GEMM_MARKER in ln.kernel]
    try:
        attributed = DCR.attribute_launches(gemm, manifest)
    except DCR.CounterRunRefused as exc:
        print(f"FAIL ncu-bytes: attribution: {exc}")
        return 3
    payload = {"schema": PAGE_SCHEMA, "family": FAMILY, "model": args.model, "branch": args.branch,
               "sample": os.path.relpath(Path(args.sample).resolve(), out.parent.resolve()), "sample_sha256": sha,
               "design": {"calls_per_cell": plan["calls_per_cell"], "warmup_calls": plan["warmup_calls"],
                          "lock_mhz": args.lock_mhz, "clock_control": "none", "cache_control": "all",
                          "replay": "kernel", "launch_skip": skip, "launch_count": count,
                          "kernel_filter": DCR.R3_KERNEL_FILTER, "metrics": list(metrics)},
               "ncu": {"binary": ncu["binary"], "version": ncu.get("version"), "argv": argv},
               "manifest": manifest, "cells": reduce_cells(attributed, manifest, by),
               "seconds": round(time.time() - t0, 1),
               "gates": {"grid": "PASS (every fused_moe_kernel launch's grid is the child's readback)",
                         "readback_mismatches": sum(1 for v in manifest.get("readback", {}).values()
                                                    if v.get("verdict") != "PASS")}}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    os.replace(tmp, out)
    print(f"PASS ncu-bytes {args.branch}: {len(payload['cells'])} cells, {len(attributed)} launches attributed -> {out}")
    return 0


def plan_lines(cells: list[dict], branch: str, calls: int, warmups: int) -> list[str]:
    n = len(cells)
    tiles = defaultdict(int)
    for c in cells:
        tiles[c["tile"]] += 1
    return [f"NCU BYTES {branch}: {n} cells {dict(tiles)}, {warmups} warm-ups (the Spy readback first) then {calls} "
            f"measured calls each: {n * calls * len(GEMMS)} profiled launches, about {expected_s(n, calls) / 60:.1f} min",
            "  ncu over dram_counter_route.r3_ncu_argv: --cache-control all, --clock-control none under the driver's "
            "lock, -k regex:^fused_moe_kernel$, kernel replay; the grids held to the child's readback "
            "(attribute_launches)"]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--counter-child"]:
        try:
            return counter_child(argv[1])
        except Refused as exc:
            print(f"REFUSED: {exc}")
            return 2
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subsample")
    s.add_argument("--model", required=True, choices=sorted(PL.MODELS))
    s.add_argument("--sample", action="append", required=True, help="BRANCH=main's sample.json")
    s.add_argument("--n-per-branch", type=int, default=N_PER_BRANCH)
    s.add_argument("--out", required=True)
    for name in ("run", "plan"):
        r = sub.add_parser(name)
        r.add_argument("--model", required=True, choices=sorted(PL.MODELS))
        r.add_argument("--sample", required=True)
        r.add_argument("--branch", required=True, choices=("decode", "prefill"))
        r.add_argument("--calls", type=int, default=CALLS)
        r.add_argument("--warmups", type=int, default=WARMUPS)
        r.add_argument("--lock-mhz", type=int, default=None)
        r.add_argument("--profile-dir", default=None, help="the .ncu-rep, its CSV and the plan (outside the pushed tree)")
        r.add_argument("--timeout-s", type=float, default=3000.0)
        r.add_argument("--out", required=name == "run")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "subsample":
            samples = {}
            for it in a.sample:
                b, _, p = it.partition("=")
                if b not in ("decode", "prefill") or not p:
                    raise Refused(f"--sample {it!r}: BRANCH=PATH")
                samples[b] = p
            p = subsample(samples, a.model, a.n_per_branch, a.out)
            doc = json.loads(p.read_text())
            print(f"PASS ncu subsample: {doc['n_cells']} cells -> {p} ({ {b: v['picked'] for b, v in doc['subsample_of'].items()} })")
            return 0
        cells, _ids, _sha = load_cells(a.sample, a.model, a.branch)
        if a.lock_mhz is not None and a.lock_mhz != PL.LOCKS[a.branch]:
            raise Refused(f"--branch {a.branch} is counted at its branch clock {PL.LOCKS[a.branch]}, not {a.lock_mhz}")
        print("\n".join(plan_lines(cells, a.branch, a.calls, a.warmups)))
        if a.cmd == "plan":
            return 0
        return run(a)
    except Refused as exc:
        print(f"FAIL ncu-bytes: {exc}")
        return 3
    except Exception as exc:                                          # noqa: BLE001  (never a bare traceback)
        print(f"FAIL ncu-bytes: {type(exc).__name__}: {exc}"[:400])
        return 3


if __name__ == "__main__":
    sys.exit(main())
