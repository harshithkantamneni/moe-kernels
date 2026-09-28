#!/usr/bin/env python
"""Register one card's predictions for ANOTHER MoE model, fitted on nothing of it.

    python scripts/cross_model_predict.py --target mixtral-8x22b \\
        --timed <run dir> [...] --counters <r3c-g*.json dir> --out predictions.json

WHAT IT TESTS. The two per-card models of docs/COUNTERS.md 6.7 were built and
fitted on Mixtral 8x7B's shapes only (w1 K 4096 x N 28672, w2 K 14336 x N
4096). A model of the kernel, not of those shapes, must predict another model's
bytes and times on the same card with every fitted number held: the wave-split
byte model's parameters (`scripts/wave_split_bytes.py`, the registered view)
and the timing model's (`scripts/r3_timing_model.py`: T0, c per CTA k-step, bw,
s_small, s_block, at the knee P_KNEE and k_w). Only the shapes change: the
N-tiles per M-row, the k-steps per CTA, the slab and A-row bytes, the weight
bytes, and the silu_and_mul and moe_sum bytes, all from the target's config in
`moe/spec.py`. The co-residency window (SMs x CTAs per SM) is carried over: the
tile, warps and stages are the same, so each CTA's registers and shared memory
are too, whatever K is. Nothing in the output is fitted on the target.

WHAT IT WRITES. Per (arm, G, n) at R3's ladder: the predicted weight-set reads
per call q of each GEMM (registered view, beside fill and ws), their status
(OUT-OF-DOMAIN, ILL-POSED), and the predicted call time T in ms at the source
pages' lock; and the source fit it came from (card, board, pages, parameters),
so a page measured later is scored against numbers fixed before it existed.
The time uses the predicted bytes (sigma from q), so a byte miss propagates
into T, as it would in the real kernel. NATIVE's bytes are SHARED's (the byte
model has no dead-CTA term), its path per n is the source's.

ANOTHER EXPERT COUNT (2026-09-28). A target with another E or top_k carries its
own R3 design: E experts, the counter pages' declaration (`counter_declaration`,
9 copies over the 9-deep ladder; NATIVE declares E), and NATIVE's alignment path
per tread from `path_census` on the target, all computed the way R3 computes
them on the card. The byte walk and the timing schedule take that design; every
fitted number is still the source's. Refused, as R3 refuses it: a target whose
tread cannot put whole tokens on every expert (`ladder_treads`), whose declaration
vLLM's alignment kernel refuses, or whose ladder switches alignment kernel inside
a ratio arm. The fitted parameters are indexed by GEMM name; at E = 64 w1 has
the window and N-tiles of Mixtral's w2, which the output states beside them.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dram_counter_route as DCR  # noqa: E402
import private_weight_reference as PW  # noqa: E402
import r3_timing_model as TM  # noqa: E402
import wave_split_bytes as W  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

PLAN_G = (1, 2, 3, 4, 8, 16, 32, 64)
PLAN_N = tuple(range(1, 10))
ARMS = ("native", "shared", "private")


class Refused(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Design:
    """R3's design for the target, as R3 computes it on the card."""

    experts: int
    copies: int
    declared: dict          # arm -> declared expert slots
    path: dict              # n -> NATIVE's alignment path
    treads: tuple


def target_design(target: str, block_m: int) -> Design:
    cfg = MODEL_CONFIGS[target]
    try:
        treads = PW.counter_ladder(cfg, block_m)
        copies, _why = PW.counter_declaration(cfg, block_m)
    except PW.PrivateWeightRefusal as exc:
        raise Refused(f"{target}: R3 refuses its ladder: {exc}") from None
    declared = {a: PW.declared_experts(a, cfg.num_experts, copies) for a in ARMS}
    census = PW.path_census(cfg, treads, block_m, declared)
    if census.refusals:
        raise Refused(f"{target}: R3 refuses its design: {census.refusals[0]}")
    path = {n: p for arm, n, _t, _ids, _d, p in census.rows if arm == "native"}
    return Design(cfg.num_experts, copies, declared, path, tuple(treads))


def target_geometry(geom: W.Geometry, target: str, design: Design | None = None) -> W.Geometry:
    """The source card's geometry with the target model's shapes and, given
    its design, the target's experts and declaration."""
    cfg = MODEL_CONFIGS[target]
    bm = DCR.r3_byte_model(cfg, geom.dtype, geom.block_m)
    P, K, Wg = {}, {}, {}
    for g in W.GEMMS:
        k, n_cols = DCR.r3_gemm_geometry(cfg, g)
        P[g] = DCR.pid_n_count(n_cols, geom.block_n)
        K[g] = int(k)
        Wg[g] = int(bm[f"W_{g}"])
    out = dataclasses.replace(geom, model=target, P=tuple(sorted(P.items())),
                              K=tuple(sorted(K.items())), W=tuple(sorted(Wg.items())))
    if design is not None:
        out = dataclasses.replace(out, experts=design.experts, copies_declared=design.copies,
                                  declared=tuple(sorted(design.declared.items())))
    return out


def predict(timed: list[Path], counters: Path, target: str) -> dict:
    card = W.load_card(counters)
    src_cfg, tgt_cfg = MODEL_CONFIGS[card.geom.model], MODEL_CONFIGS[target]
    same = (src_cfg.num_experts, src_cfg.top_k) == (tgt_cfg.num_experts, tgt_cfg.top_k)
    design = None if same else target_design(target, card.geom.block_m)
    plan_n = PLAN_N if same else tuple(n for n in design.treads if n in PLAN_N)
    res = W.analyse(card)
    view = W.REGISTERED_VIEW
    prm = {v: res.params(v) for v in W.VIEWS if v in res.fits}
    geom_t = target_geometry(card.geom, target, design)
    model_t = W.Model(geom_t)
    cells = [(a, G, n, g) for g in W.GEMMS for a in ("shared", "private")
             for G in PLAN_G for n in plan_n]
    ev = {v: dict(zip(cells, model_t.evaluate(p, cells), strict=True)) for v, p in prm.items()}

    R = TM.build(TM.build_parser().parse_args([*map(str, timed), "--counters", str(counters)]))
    fit, ctx = R["main"], R["ctx"]
    src_cells = R["cells"]
    path = {c.n: c.path for c in src_cells if c.arm == "native"}
    declared = {c.arm: c.declared for c in src_cells}
    if design is not None:
        path, declared = dict(design.path), dict(design.declared)
    old = TM.set_model(target)
    try:
        bm_t = TM.BYTE_MODEL
        out_cells = []
        for arm in TM.ARMS:
            for G in PLAN_G:
                for n in plan_n:
                    qa = "shared" if arm == "native" else arm
                    q = {g: ev[view][(qa, G, n, g)] for g in W.GEMMS}
                    shown = {g: W.shown(q[g]) for g in W.GEMMS}
                    row = {"arm": arm, "G": G, "n": n,
                           "q": shown, "status": {g: q[g]["rule"] for g in W.GEMMS},
                           "q_views": {v: {g: W.shown(ev[v][(qa, G, n, g)]) for g in W.GEMMS}
                                       for v in ev},
                           "ms": None}
                    if all(shown[g] is not None for g in W.GEMMS) and n in path:
                        reads = {g: shown[g] * bm_t[f"W_{g}"] + n * bm_t[f"operand_per_tile_{g}"]
                                 for g in W.GEMMS}
                        D = declared[arm]
                        sigma = {g: reads[g] / TM.window(arm, D, G, n, g, TM.window_width(
                            fit.k_w, ctx.sms, ctx.occupancy[g])).reads for g in W.GEMMS}
                        cell = TM.Cell(arm=arm, G=G, n=n, ms=float("nan"), mhz=None,
                                       path=path[n], declared=D, runs=(), source="PREDICTED",
                                       reads=reads, sigma=sigma, fit=False)
                        row["ms"] = TM.call_ms(fit.x, cell, ctx, fit.k_w)
                    out_cells.append(row)
    finally:
        TM.set_model(old)
    return {"tool": "scripts/cross_model_predict.py", "target": target,
            "source_model": card.geom.model, "card": card.geom.card,
            "board": DCR.board(card.geom.uuid),
            "byte_view": view, "p_knee": TM.P_KNEE, "k_w": fit.k_w,
            "design": None if design is None else {
                "experts": design.experts, "copies_declared": design.copies,
                "declared": design.declared, "native_path": design.path,
                "source_experts": src_cfg.num_experts},
            "clock_mhz": ctx.clock_mhz,
            "timing_params": fit.params, "timing_pages": [p.run for p in R["use"]],
            "counter_pages": {str(G): str(p) for G, p in sorted(card.paths.items())},
            "byte_params": {v: p.as_json() for v, p in prm.items()},
            "shapes": {g: {"K": dict(geom_t.K)[g], "npn": dict(geom_t.P)[g],
                           "W_bytes": dict(geom_t.W)[g]} for g in W.GEMMS},
            "window_m_rows": {g: {m: TM.window_width(fit.k_w, ctx.sms, ctx.occupancy[g])
                                  / dict(geo.P)[g]
                                  for m, geo in (("source", card.geom), ("target", geom_t))}
                              for g in W.GEMMS},
            "cells": out_cells}


def summary_lines(d: dict) -> list[str]:
    cells = {(c["arm"], c["G"], c["n"]): c for c in d["cells"]}
    plan_n = sorted({c["n"] for c in d["cells"]})
    L = [f"REGISTERED PREDICTIONS for {d['target']} on {d['card']}, from {d['source_model']}'s "
         f"fit (nothing fitted on the target): bytes in the {d['byte_view']} view, time at "
         f"{d['clock_mhz']:.0f} MHz, knee p = {d['p_knee']:g}, k_w = {d['k_w']:g}",
         "  shapes: " + "; ".join(f"{g} K {s['K']} x {s['npn']} N-tiles, "
                                  f"{s['W_bytes'] / 1e9:.3f} GB" for g, s in d["shapes"].items())]
    if d.get("design"):
        z = d["design"]
        L.append(f"  design: E {z['source_experts']} -> {z['experts']}, {z['copies_declared']} "
                 f"copies, declared {z['declared']}; NATIVE's path per n "
                 f"{sorted(set(z['native_path'].values()))}")
    if d.get("window_m_rows"):
        L.append("  co-residency window in M-rows (the fitted per-GEMM parameters were fitted at "
                 "the source's): " + "; ".join(
                     f"{g} {w['source']:.1f} -> {w['target']:.1f}"
                     for g, w in d["window_m_rows"].items()))
    for G in PLAN_G:
        for arm in ("shared", "private"):
            row = [cells[(arm, G, n)] for n in plan_n]
            T = " ".join("--" if r["ms"] is None else f"{r['ms']:.4f}" for r in row)
            q2 = " ".join("--" if r["q"]["w2"] is None else f"{r['q']['w2']:.3f}" for r in row)
            flag = " OOD" if any(r["status"]["w2"] == W.OUT_OF_DOMAIN for r in row) else ""
            L.append(f"  G={G:<3} {arm:<7} T ms {T}")
            L.append(f"  {'':<5} {'':<7} q_w2 {q2}{flag}")
    return L


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True, choices=sorted(MODEL_CONFIGS))
    p.add_argument("--timed", nargs="+", type=Path, required=True,
                   help="the source card's timed R3 run dirs (one lock, one card)")
    p.add_argument("--counters", type=Path, required=True,
                   help="the source card's r3c-g*.json directory (the same kernel)")
    p.add_argument("--out", type=Path, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        d = predict(args.timed, args.counters, args.target)
    except (Refused, TM.Refused, W.Refused) as exc:
        print(f"REFUSED: {exc}")
        return 2
    print("\n".join(summary_lines(d)))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(d, indent=1, default=str) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
