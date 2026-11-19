"""Replay control C (KIT-SPEC 5): re-time a pre-registered stratified subsample of the logged calls ALONE, with the
study's validated decode instrument, on each call's REAL ids.

    python scripts/prod/replay.py sample --run RUN_DIR --model ernie-4.5-21b-a3b --n 1500 --out SAMPLE_DIR
    python scripts/locked_r3.py --session-tag T --locks 1710 --groups 1 --slip-policy cell --min-clean-repeats 0 \\
        --r3-script scripts/prod/replay.py -- --model ernie-4.5-21b-a3b --histogram SAMPLE_DIR/sample.json \\
        --branch decode --inst GRR --repeats 9 --duty 0.25 --seed 0          # prefill: --locks 1410 --branch prefill
    python scripts/prod/replay.py --model ... --histogram SAMPLE_DIR/sample.json --branch decode --dry-run

THE INSTRUMENT IS scripts/r3_decode.py's, NOT A NEW TIMER. GRR: one CUDA graph of `calls` back-to-back calls of
vLLM's own `fused_experts` (align .. moe_sum, router excluded), call j on weight copy j mod R so every call is cold
(R x A_min x W_e >= 240 MiB), timed by launch_floor.mirror_time_cell at duty 0.25; per call = replay / calls. Its
untimed checks run unchanged per cell: V-ALIGN (the spy reads back sorted ids, expert ids, post-padded count,
grid), GD5 (vLLM's own default tile, no tuned file, no override), GD4 (graph output == eager output bit for bit),
COPIES, GD1 (host enqueue), and R3's repeat schedule with lock slips (--slip-policy cell under locked_r3).

THE ADAPTER (the smallest one; disclosed): r3_decode builds a cell's ids from its counts (`realize_counts` + a
token-row shuffle) and its page reader refuses T > 128. Real calls have real ids and prefill T, so this file
(1) builds the DecodePage directly from the sampled calls (label g<global index>, T = the call's PADDED T, counts =
bincount of its ids, arms native, shuffle off), skipping `page_from`'s decode-only T check, and (2) overrides
`GPUBackend.build` to hand the call's logged ids (int32, token-major as captured) to the same `_call` closures.
Nothing else of r3_decode changes: run_page, the spy, GD4/GD5, rotation, Store, write_report are imported.
Disclosed differences from in-serving: hidden states are random (not the layer's real activations); weights are
private copies (cold), where serving's are warm in L2 at small T (CRITIQUE MINOR).

THE SAMPLE (pre-registered rule, `select_sample`): calls with status ok, not warmup, joined to a step; per tile
an equal quota of n // tiles (the remainder to the tiles in TILE_BINS order); inside a tile the calls ordered by
(step, layer) and taken systematically from a seeded start (seed 20261119). Branch clock: decode tiles (T <= 128)
at the 1710 MHz lock, prefill tiles at 1410 (CRITIQUE B1): two locked_r3 runs, --branch decode / prefill.

THE TIME BUDGET (registered rule, review 1 MAJOR 6; `budget_quota`): every replay writes cell_wall.json, the wall time
each cell took (checks, graph captures and every repeat: the interval from its build to the next cell's). The smoke
rental's walls (merged by `replay.py wall-merge` into scripts/prod/replay_wall.json before the kit freezes) price
main's sample: a tile's cost per cell is WALL_SAFETY x the median smoke wall of that tile x (main's repeats + 1) /
(smoke's repeats + 1) (a tile the smoke did not time takes its branch's dearest timed tile); when the registered
quotas cost more than the branch's REPLAY_BUDGET_S, every tile's quota is scaled by budget / cost, floored, at least
1. The sample records the rule's inputs and the scaled quotas.

THE BIG CELL (smoke, `--min-big-T 8256`): the prefill sample always holds one cell at T >= 8256 (the G-16 tile and the
replay's worst memory): the run's largest such call when there is one, else a SYNTHETIC cell at T 16384 (seeded uniform
routing, label s16384, `synthetic: true`, never scored: the scorer reads g<index> labels only).

GRR MEMORY (bounded, review 1 MAJOR 6): a GRR graph keeps every call's output live (`holder`), so the plan adds
calls x T_max x H x 2 bytes to r3_decode's single-call memory plan and refuses a page over the device's free memory or
over GRR_OUT_CAP_BYTES of graph outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prodlib as PL  # noqa: E402  (puts scripts/ on sys.path for r3_decode)
import r3_decode as RD  # noqa: E402

SAMPLE_SCHEMA = "moe-kernels/prod-replay-sample/1"
SAMPLE_SEED = 20261119
DEFAULT_N = 1500
#: the registered time budget per branch replay on the main rental (seconds) and the smoke wall's safety factor
REPLAY_BUDGET_S = {"decode": 2400, "prefill": 2400}
WALL_SAFETY = 1.5
WALL_SCHEMA = "moe-kernels/prod-replay-wall/1"
#: the GRR graph's live outputs may not pass this (calls x T_max x H x 2 bytes)
GRR_OUT_CAP_BYTES = 8 * 2 ** 30
BIG_T_SYNTHETIC = 16384


# ------------------------------------------------------------------------------------------------------------
# the sample (pure)
# ------------------------------------------------------------------------------------------------------------

def select_sample(rows: list[dict], n: int = DEFAULT_N, seed: int = SAMPLE_SEED) -> list[dict]:
    """The registered stratified subsample (see the module doc). Deterministic in (rows, n, seed)."""
    ok = [r for r in rows if r.get("status") == "ok" and not r.get("warmup") and r.get("step") is not None]
    by_tile: dict = defaultdict(list)
    for r in ok:
        by_tile[r["tile"]].append(r)
    tiles = [lab for lab, _lo, _hi in PL.TILE_BINS if by_tile.get(lab)]
    if not tiles:
        return []
    base, extra = divmod(int(n), len(tiles))
    rng = random.Random(int(seed))
    out = []
    for i, lab in enumerate(tiles):
        pool = sorted(by_tile[lab], key=lambda r: (r["step"], r["layer"]))
        q = min(len(pool), base + (1 if i < extra else 0))
        if q <= 0:
            continue
        stride = len(pool) / q
        start = rng.random() * stride
        out += [pool[min(len(pool) - 1, int(start + j * stride))] for j in range(q)]
    return out


def budget_quota(quota: dict, wall: dict, branch: str, *, repeats: int, budget_s: float | None = None) -> dict:
    """The registered time-budget rule (module doc). quota: tile -> registered cells; wall: replay_wall.json's
    {"repeats": smoke repeats, "tiles": {tile: {"median_wall_s", "cells"}}}. Returns {"quota", "cost_s", "scale",
    "per_cell_s", "budget_s"}; raises KitError when the branch has no timed tile."""
    budget = float(REPLAY_BUDGET_S[branch] if budget_s is None else budget_s)
    timed = {t: float(v["median_wall_s"]) for t, v in (wall.get("tiles") or {}).items()
             if PL.branch(next(lo for lab, lo, _hi in PL.TILE_BINS if lab == t)) == branch}
    if not timed:
        raise PL.KitError(f"the smoke wall times no {branch} tile: the replay budget cannot be priced")
    r_smoke = int(wall["repeats"])
    per = {t: WALL_SAFETY * timed.get(t, max(timed.values())) * (int(repeats) + 1) / (r_smoke + 1) for t in quota}
    cost = sum(quota[t] * per[t] for t in quota)
    scale = min(1.0, budget / cost) if cost > 0 else 1.0
    q = {t: (quota[t] if scale >= 1.0 else max(1, int(quota[t] * scale))) for t in quota}
    return {"quota": q, "registered_quota": dict(quota), "per_cell_s": per, "cost_s": cost, "scale": scale,
            "budget_s": budget, "safety": WALL_SAFETY, "repeats": int(repeats), "smoke_repeats": r_smoke,
            "cost_after_s": sum(q[t] * per[t] for t in q)}


def tile_quota(rows: list[dict], n: int) -> dict:
    """The registered equal quota per tile present (select_sample's split)."""
    ok = [r for r in rows if r.get("status") == "ok" and not r.get("warmup") and r.get("step") is not None]
    have = {r["tile"] for r in ok}
    tiles = [lab for lab, _lo, _hi in PL.TILE_BINS if lab in have]
    if not tiles:
        return {}
    base, extra = divmod(int(n), len(tiles))
    return {lab: base + (1 if i < extra else 0) for i, lab in enumerate(tiles)}


def select_budgeted(rows: list[dict], quota: dict, seed: int = SAMPLE_SEED) -> list[dict]:
    """select_sample's systematic rule at per-tile quotas (the budget's)."""
    ok = [r for r in rows if r.get("status") == "ok" and not r.get("warmup") and r.get("step") is not None]
    by_tile: dict = defaultdict(list)
    for r in ok:
        by_tile[r["tile"]].append(r)
    rng = random.Random(int(seed))
    out = []
    for lab, _lo, _hi in PL.TILE_BINS:
        if lab not in quota or not by_tile.get(lab):
            continue
        pool = sorted(by_tile[lab], key=lambda r: (r["step"], r["layer"]))
        q = min(len(pool), quota[lab])
        stride = len(pool) / q
        start = rng.random() * stride
        out += [pool[min(len(pool) - 1, int(start + j * stride))] for j in range(q)]
    return out


def synthetic_cell(model: PL.ServedModel, T: int = BIG_T_SYNTHETIC, seed: int = SAMPLE_SEED) -> tuple[dict, np.ndarray]:
    cfg = model.cfg
    rng = np.random.default_rng(int(seed) + int(T))
    ids = np.argsort(rng.random((int(T), cfg.num_experts)), axis=1)[:, :cfg.top_k]
    return ({"label": f"s{int(T)}", "g": None, "T": int(T), "T_unpadded": int(T), "layer": None,
             "tile": PL.tile_bin(T), "branch": PL.branch(T), "counts": PL.counts_from_ids(ids, cfg.num_experts),
             "ids_digest": PL.digest(ids), "step": None, "synthetic": True}, ids)


def write_sample(run_dir, model: PL.ServedModel, picked: list[dict], out_dir, *, calls_file: str = "calls.jsonl",
                 synthetic: list | None = None, budget: dict | None = None) -> Path:
    """sample.json (cells) + sample_ids.npz (each cell's ids, [T_pad, k] int16) from the run's ring; `synthetic`:
    extra (cell, ids) pairs (the smoke's big cell when the run has none)."""
    cfg = model.cfg
    shards = sorted((Path(run_dir) / "ring").glob("shard_*.npz"))
    recs = {r["g"]: r for r in PL.decode_shards(shards, cfg.top_k)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells, ids = [], {}
    for r in picked:
        rec = recs[r["g"]]
        lab = f"g{r['g']}"
        counts = PL.counts_from_ids(rec["ids"], cfg.num_experts)
        cells.append({"label": lab, "g": r["g"], "T": rec["n"], "T_unpadded": r.get("T"), "layer": rec["layer"],
                      "tile": PL.tile_bin(rec["n"]), "branch": PL.branch(rec["n"]), "counts": counts,
                      "ids_digest": PL.digest(rec["ids"]), "step": r.get("step")})
        ids[lab] = np.asarray(rec["ids"], dtype=np.int16)
    for c, a in synthetic or []:
        cells.append(c)
        ids[c["label"]] = np.asarray(a, dtype=np.int16)
    np.savez_compressed(out / "sample_ids.npz", **ids)
    doc = {"schema": SAMPLE_SCHEMA, "model": model.key, "seed": SAMPLE_SEED, "n_cells": len(cells),
           "run": str(run_dir), "ids_file": "sample_ids.npz", "budget": budget,
           "ids_sha256": hashlib.sha256((out / "sample_ids.npz").read_bytes()).hexdigest(), "cells": cells}
    (out / "sample.json").write_text(json.dumps(doc, indent=1))
    return out / "sample.json"


def load_sample(path, model: PL.ServedModel, branch: str | None) -> tuple[RD.DecodePage, dict]:
    """The DecodePage of the sample's cells on one branch, and {label: ids [T, k] int32 np}. V-BIN on load: every
    cell's bincount equals its counts and no token repeats an expert."""
    p = Path(path)
    data = p.read_bytes()
    doc = json.loads(data)
    if doc.get("schema") != SAMPLE_SCHEMA or doc.get("model") != model.key:
        raise RD.DecodeRefused(f"{p}: schema {doc.get('schema')!r} / model {doc.get('model')!r}")
    idf = p.parent / doc["ids_file"]
    if hashlib.sha256(idf.read_bytes()).hexdigest() != doc["ids_sha256"]:
        raise RD.DecodeRefused(f"{idf}: sha256 differs from the sample's record")
    cfg = model.cfg
    cells, ids = [], {}
    with np.load(idf) as z:
        for c in doc["cells"]:
            if branch and c["branch"] != branch:
                continue
            a = np.asarray(z[c["label"]], dtype=np.int64)
            if a.shape != (c["T"], cfg.top_k):
                raise RD.DecodeRefused(f"cell {c['label']}: ids shape {a.shape}, T {c['T']} k {cfg.top_k}")
            if PL.counts_from_ids(a, cfg.num_experts) != list(c["counts"]):
                raise RD.DecodeRefused(f"cell {c['label']} (V-BIN): bincount is not the counts")
            srt = np.sort(a, axis=1)
            if (srt[:, 1:] == srt[:, :-1]).any():
                raise RD.DecodeRefused(f"cell {c['label']} (V-BIN): a token repeats an expert")
            cells.append(RD.DecodeCell(label=c["label"], T=int(c["T"]), counts=tuple(int(v) for v in c["counts"]),
                                       arms=(RD.NATIVE,), shuffle=False, role=c["tile"]))
            ids[c["label"]] = a.astype(np.int32)
    if not cells:
        raise RD.DecodeRefused(f"{p}: no cells on branch {branch}")
    page = RD.DecodePage(model=model.key, page=f"prod-replay-{branch or 'all'}", shuffle_seed=0, decl_copies=None,
                         cells=tuple(cells), file_sha256=hashlib.sha256(data).hexdigest(), label="prod replay")
    return page, ids


# ------------------------------------------------------------------------------------------------------------
# the adapter: r3_decode's backend, real ids
# ------------------------------------------------------------------------------------------------------------

class RealIdsMixin:
    """Overrides `build(cell, arm)` to return the call's logged ids instead of realize_counts' (the only change), and
    clocks each cell's wall time: the interval from a cell's build to the next build is that cell's (its checks,
    graph captures and repeats), summed per label (`wall()`)."""
    real_ids: dict = {}

    def _wall_mark(self, label) -> None:
        now = time.perf_counter()
        if not hasattr(self, "_wall"):
            self._wall = {}
        cur = getattr(self, "_wall_cur", None)
        if cur is not None and cur[0] is not None:
            self._wall[cur[0]] = self._wall.get(cur[0], 0.0) + (now - cur[1])
        self._wall_cur = (label, now)

    def wall(self) -> dict:
        self._wall_mark(None)
        return dict(getattr(self, "_wall", {}))

    def build(self, cell, arm):
        self._wall_mark(cell.label)
        if arm != RD.NATIVE:
            raise RD.DecodeRefused("the replay times the native arm only")
        a = self.real_ids[cell.label]
        try:
            import torch
            ids = torch.as_tensor(a, dtype=torch.int32)
            if getattr(self, "fused_experts", None) is not None:          # the GPU backend: ids on the card
                ids = ids.to("cuda")
        except ImportError:                                               # pragma: no cover
            ids = a
        if not hasattr(self, "_call"):                                    # a fake backend (tests)
            return ids, [(cell, arm)] * getattr(self, "calls", 1)
        calls = [self._call(cell, arm, ids, j % (self.R if self.inst == "GRR" else 1))
                 for j in range(self.calls if self.inst == "GRR" else 1)]
        return ids, calls


def prod_backend_cls():
    class ProdBackend(RealIdsMixin, RD.GPUBackend):
        pass
    return ProdBackend


# ------------------------------------------------------------------------------------------------------------
# CLI (locked_r3-compatible: --histogram, --session-tag, --lock-mhz, --slip-policy, --group-m, --out)
# ------------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(PL.MODELS))
    ap.add_argument("--histogram", type=Path, required=True, help="the replay sample (sample.json)")
    ap.add_argument("--branch", choices=("decode", "prefill"), required=True)
    ap.add_argument("--inst", default="GRR", choices=RD.INSTRUMENTS)
    ap.add_argument("--rot", default="auto")
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--duty", type=float, default=RD.DUTY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=RD.WARMUP_MS)
    ap.add_argument("--cell-budget-ms", type=float, default=RD.CELL_BUDGET_MS)
    ap.add_argument("--trials", type=int, default=RD.TRIALS)
    ap.add_argument("--group-m", type=int, default=1, help="a label only: vLLM's own GROUP_SIZE_M runs (GD5)")
    ap.add_argument("--session-tag", default="")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--slip-policy", choices=("page", "cell"), default="page")
    ap.add_argument("--lock-mhz", type=int, default=None)
    ap.add_argument("--min-clean-repeats", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def sample_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="replay.py sample")
    ap.add_argument("--run", required=True)
    ap.add_argument("--calls", default=None, help="calls.jsonl (default RUN/calls.jsonl)")
    ap.add_argument("--model", required=True, choices=sorted(PL.MODELS))
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--branch", choices=("decode", "prefill"), default=None,
                    help="only this branch's tiles (the decode sample from the 1710 run, prefill from 1410)")
    ap.add_argument("--wall", default=None, help="the smoke's replay_wall.json: scale the quotas to the time budget")
    ap.add_argument("--repeats", type=int, default=9, help="the replay's repeats (prices the budget)")
    ap.add_argument("--budget-s", type=float, default=None, help="override REPLAY_BUDGET_S[branch] (tests)")
    ap.add_argument("--min-big-T", type=int, default=None,
                    help="the sample holds a cell at T >= this (the run's largest, else a synthetic T 16384)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model = PL.MODELS[a.model]
    rows = PL.read_jsonl(a.calls or Path(a.run) / "calls.jsonl")
    if a.branch:
        rows = [r for r in rows if r.get("branch") == a.branch]
    budget = None
    if a.wall:
        if not a.branch:
            print("FAIL replay sample: --wall prices one branch (--branch)")
            return 3
        try:
            wall = json.loads(Path(a.wall).read_text())
            if wall.get("schema") != WALL_SCHEMA:
                raise PL.KitError(f"{a.wall}: schema {wall.get('schema')!r}")
            budget = budget_quota(tile_quota(rows, a.n), wall, a.branch, repeats=a.repeats, budget_s=a.budget_s)
        except (PL.KitError, OSError, ValueError, KeyError) as exc:
            print(f"FAIL replay sample: the budget: {exc}")
            return 3
        picked = select_budgeted(rows, budget["quota"])
    else:
        picked = select_sample(rows, a.n)
    if not picked:
        print("FAIL replay sample: no eligible calls")
        return 3
    synth = []
    if a.min_big_T:
        if not any(int(r["T_pad"]) >= a.min_big_T for r in picked):
            big = [r for r in rows if r.get("status") == "ok" and r.get("step") is not None
                   and int(r["T_pad"]) >= a.min_big_T]
            if big:
                picked.append(max(big, key=lambda r: (int(r["T_pad"]), -int(r["g"]))))
            else:
                synth.append(synthetic_cell(model))
    p = write_sample(a.run, model, picked, a.out, synthetic=synth, budget=budget)
    tiles = defaultdict(int)
    for r in picked:
        tiles[r["tile"]] += 1
    for c, _i in synth:
        tiles[c["tile"] + " (synthetic)"] += 1
    print(f"PASS replay sample: {len(picked) + len(synth)} cells -> {p}; per tile {dict(tiles)}"
          + (f"; budget scale {budget['scale']:.3f} ({budget['cost_after_s']:.0f} s of {budget['budget_s']:.0f})"
             if budget else ""))
    return 0


def wall_merge(argv) -> int:
    """replay_wall.json from replay runs' cell_wall.json files: per tile the median wall per cell (smoke)."""
    ap = argparse.ArgumentParser(prog="replay.py wall-merge")
    ap.add_argument("--from", dest="src", action="append", required=True, help="a directory holding cell_wall.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    per: dict = defaultdict(list)
    reps = set()
    for d in a.src:
        for f in sorted(Path(d).rglob("cell_wall.json")):
            doc = json.loads(f.read_text())
            reps.add(int(doc["repeats"]))
            for c in doc["cells"]:
                if not c.get("synthetic") or c["tile"].endswith("g16"):
                    per[c["tile"]].append(float(c["wall_s"]))
    if not per or len(reps) != 1:
        print(f"FAIL replay wall-merge: {sum(len(v) for v in per.values())} cells, repeats {sorted(reps)}")
        return 3
    doc = {"schema": WALL_SCHEMA, "repeats": reps.pop(),
           "tiles": {t: {"median_wall_s": float(np.median(v)), "cells": len(v)} for t, v in sorted(per.items())}}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"PASS replay wall-merge: {len(doc['tiles'])} tiles -> {a.out}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "sample":
        return sample_main(argv[1:])
    if argv and argv[0] == "wall-merge":
        return wall_merge(argv[1:])
    from moe.bench import exit_codes
    PWR, SWEEP = RD.PWR, RD.SWEEP
    args = build_parser().parse_args(argv)
    model = PL.MODELS[args.model]
    cfg = model.cfg
    try:
        page, real = load_sample(args.histogram, model, args.branch)
        if args.group_m != 1:
            raise RD.DecodeRefused("--group-m is a label here and must be 1 (locked_r3 --groups 1)")
        rot = None if args.rot == "auto" else int(args.rot)
        R, calls, why_R = RD.rotation(cfg, page, "bf16", rot)
        copies = RD.copies_for(page, args.inst, R)
        slip_lock = None
        if args.slip_policy == "cell":
            if not args.lock_mhz:
                raise RD.DecodeRefused("--slip-policy cell needs --lock-mhz (locked_r3 passes it)")
            slip_lock = int(args.lock_mhz)
            want = PL.LOCKS[args.branch]
            if slip_lock != want:
                raise RD.DecodeRefused(f"--branch {args.branch} is timed at its branch clock {want} MHz, not {slip_lock}")
        elif args.lock_mhz is not None or args.min_clean_repeats:
            raise RD.DecodeRefused("--lock-mhz and --min-clean-repeats ride with --slip-policy cell")
        if not 0 < args.duty < 1:
            raise RD.DecodeRefused(f"--duty {args.duty}")
    except (RD.DecodeRefused, ValueError, OSError, KeyError) as exc:
        print(f"REFUSED: {exc}")
        return exit_codes.REFUSED
    card = PWR.detect_card_slug()
    key = json.dumps({"model": args.model, "branch": args.branch, "inst": args.inst, "rot": args.rot,
                      "page": page.file_sha256, "session": args.session_tag, "repeats": args.repeats,
                      "duty": args.duty, "seed": args.seed}, sort_keys=True)
    run_id = args.run_id or f"{card}-prodreplay-{args.model}-{args.branch}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"
    out_dir = (args.out or SWEEP.results_root()) / "private_weight_reference" / run_id
    ns = argparse.Namespace(**{**vars(args), "dtype": "bf16"})
    header = RD.plan_lines(ns, cfg, page, R=R, calls=calls, why_R=why_R, copies=copies, out_dir=out_dir,
                           slip_lock=slip_lock)
    header[0] = f"experiment  prod replay ({args.branch}, real ids) / {out_dir.name}"
    print("PROD REPLAY PAGE (r3_decode's instrument; adapter: real ids, page built from the sample)")
    print("\n".join(header))
    free, src = PWR._device_memory(argparse.Namespace(device_memory_gb=0.0, dry_run=args.dry_run))
    mem = PWR.memory_plan(cfg, "bf16", 2, copies, max(page.Ts), free, src, copies_read=1)
    if mem.fits is False:
        print(f"\nREFUSED: the weight copies do not fit: {mem.predicted_peak_bytes / 1e9:.2f} GB predicted")
        return exit_codes.REFUSED
    bad_mem = grr_memory_refusal(cfg, page, args.inst, calls, mem.predicted_peak_bytes, free)
    if bad_mem:
        print(f"\nREFUSED (GRR memory): {bad_mem}")
        return exit_codes.REFUSED
    if args.dry_run:
        print("\nREFUSED. Nothing was measured and nothing was written.\n  reason: --dry-run was given")
        return exit_codes.REFUSED
    missing = SWEEP.missing_gpu_stack()
    if missing:
        print("\n" + missing)
        return exit_codes.REFUSED
    src_bad = RD.installed_source_check()
    if src_bad:
        print("\nREFUSED (V-SRC): " + "; ".join(src_bad))
        return exit_codes.REFUSED
    unreadable = PWR.clock_sampler_refusal()
    if unreadable:
        print(f"\nREFUSED: {unreadable}")
        return exit_codes.REFUSED
    out_dir.mkdir(parents=True, exist_ok=True)
    wrong = PWR.device_guard(out_dir, PWR.device_identity())
    if wrong:
        print(f"\nREFUSED: {wrong}")
        return exit_codes.REFUSED
    os.environ["TRITON_CACHE_DIR"] = str(out_dir / "triton-cache")
    store = RD.Store(out_dir / "cells.csv")
    t0 = time.time()
    Backend = prod_backend_cls()
    backend = Backend(cfg, page, inst=args.inst, R=R, calls=calls, copies=copies, seed=args.seed, dtype="bf16",
                      duty=args.duty, warmup_ms=args.warmup, cell_budget_ms=args.cell_budget_ms, trials=args.trials)
    backend.real_ids = real
    result = RD.run_page(cfg, page, backend, inst=args.inst, repeats=args.repeats, store=store, lock_mhz=slip_lock,
                         min_clean=args.min_clean_repeats if slip_lock is not None else 0,
                         say=lambda m: print(m, flush=True))
    store.close()
    write_wall(out_dir, page, backend.wall(), args.repeats, args.histogram)
    return RD.write_report(ns, cfg, page, result, out_dir=out_dir, run_id=run_id, card=card, R=R, calls=calls,
                           copies=copies, slip_lock=slip_lock, header=header, seconds=time.time() - t0)


def grr_memory_refusal(cfg, page, inst: str, calls: int, single_peak_bytes: int, free_bytes) -> str:
    """'' when the GRR graph fits, else why: the graph holds every call's output (calls x T_max x H x 2 bytes) on
    top of r3_decode's single-call plan."""
    if inst != "GRR":
        return ""
    out_b = int(calls) * max(page.Ts) * cfg.hidden_size * 2
    if out_b > GRR_OUT_CAP_BYTES:
        return f"{calls} calls x T {max(page.Ts)} outputs = {out_b / 2 ** 30:.2f} GiB, over {GRR_OUT_CAP_BYTES / 2 ** 30:.0f} GiB"
    if free_bytes and single_peak_bytes + out_b > free_bytes:
        return (f"{(single_peak_bytes + out_b) / 1e9:.2f} GB predicted (outputs {out_b / 1e9:.2f} GB) over "
                f"{free_bytes / 1e9:.2f} GB free")
    return ""


def write_wall(out_dir, page, wall: dict, repeats: int, sample_path) -> Path:
    """cell_wall.json: each cell's wall seconds (build to the next build: checks, captures, every repeat)."""
    doc = json.loads(Path(sample_path).read_text())
    syn = {c["label"] for c in doc["cells"] if c.get("synthetic")}
    cells = [{"label": c.label, "T": c.T, "tile": c.role, "wall_s": float(wall.get(c.label, 0.0)),
              "synthetic": c.label in syn} for c in page.cells]
    p = Path(out_dir) / "cell_wall.json"
    p.write_text(json.dumps({"schema": WALL_SCHEMA + "/cells", "repeats": int(repeats), "cells": cells}, indent=1))
    return p


if __name__ == "__main__":
    sys.exit(main())
