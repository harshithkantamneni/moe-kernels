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

It refuses a target whose expert count or top_k differs from the source's: R3's
declaration and ids per tread are then another experiment, not another shape.
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
import r3_timing_model as TM  # noqa: E402
import wave_split_bytes as W  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

PLAN_G = (1, 2, 3, 4, 8, 16, 32, 64)
PLAN_N = tuple(range(1, 10))


class Refused(Exception):
    pass


def target_geometry(geom: W.Geometry, target: str) -> W.Geometry:
    """The source card's geometry with the target model's shapes."""
    cfg = MODEL_CONFIGS[target]
    bm = DCR.r3_byte_model(cfg, geom.dtype, geom.block_m)
    P, K, Wg = {}, {}, {}
    for g in W.GEMMS:
        k, n_cols = DCR.r3_gemm_geometry(cfg, g)
        P[g] = DCR.pid_n_count(n_cols, geom.block_n)
        K[g] = int(k)
        Wg[g] = int(bm[f"W_{g}"])
    return dataclasses.replace(geom, model=target, P=tuple(sorted(P.items())),
                               K=tuple(sorted(K.items())), W=tuple(sorted(Wg.items())))


def predict(timed: list[Path], counters: Path, target: str) -> dict:
    card = W.load_card(counters)
    src_cfg, tgt_cfg = MODEL_CONFIGS[card.geom.model], MODEL_CONFIGS[target]
    if (src_cfg.num_experts, src_cfg.top_k) != (tgt_cfg.num_experts, tgt_cfg.top_k):
        raise Refused(f"{target} has E={tgt_cfg.num_experts}, top_k={tgt_cfg.top_k}; the "
                      f"source {card.geom.model} has E={src_cfg.num_experts}, "
                      f"top_k={src_cfg.top_k}: R3's declaration would be another experiment")
    res = W.analyse(card)
    view = W.REGISTERED_VIEW
    prm = {v: res.params(v) for v in W.VIEWS if v in res.fits}
    geom_t = target_geometry(card.geom, target)
    model_t = W.Model(geom_t)
    cells = [(a, G, n, g) for g in W.GEMMS for a in ("shared", "private")
             for G in PLAN_G for n in PLAN_N]
    ev = {v: dict(zip(cells, model_t.evaluate(p, cells), strict=True)) for v, p in prm.items()}

    R = TM.build(TM.build_parser().parse_args([*map(str, timed), "--counters", str(counters)]))
    fit, ctx = R["main"], R["ctx"]
    src_cells = R["cells"]
    path = {c.n: c.path for c in src_cells if c.arm == "native"}
    declared = {c.arm: c.declared for c in src_cells}
    old = TM.set_model(target)
    try:
        bm_t = TM.BYTE_MODEL
        out_cells = []
        for arm in TM.ARMS:
            for G in PLAN_G:
                for n in PLAN_N:
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
            "clock_mhz": ctx.clock_mhz,
            "timing_params": fit.params, "timing_pages": [p.run for p in R["use"]],
            "counter_pages": {str(G): str(p) for G, p in sorted(card.paths.items())},
            "byte_params": {v: p.as_json() for v, p in prm.items()},
            "shapes": {g: {"K": dict(geom_t.K)[g], "npn": dict(geom_t.P)[g],
                           "W_bytes": dict(geom_t.W)[g]} for g in W.GEMMS},
            "cells": out_cells}


def summary_lines(d: dict) -> list[str]:
    cells = {(c["arm"], c["G"], c["n"]): c for c in d["cells"]}
    L = [f"REGISTERED PREDICTIONS for {d['target']} on {d['card']}, from {d['source_model']}'s "
         f"fit (nothing fitted on the target): bytes in the {d['byte_view']} view, time at "
         f"{d['clock_mhz']:.0f} MHz, knee p = {d['p_knee']:g}, k_w = {d['k_w']:g}",
         "  shapes: " + "; ".join(f"{g} K {s['K']} x {s['npn']} N-tiles, "
                                  f"{s['W_bytes'] / 1e9:.3f} GB" for g, s in d["shapes"].items())]
    for G in PLAN_G:
        for arm in ("shared", "private"):
            row = [cells[(arm, G, n)] for n in PLAN_N]
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
