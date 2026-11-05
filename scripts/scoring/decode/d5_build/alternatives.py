#!/usr/bin/env python
"""Addendum 5's CAL2-only alternatives to v5d's open choices (D5-1 to D5-5, the regimes on): each wraps
r3_timing_model_v5d.price or swaps a term, CAL2 refitted each time; the SEEN blind is never re-scored here.
python alternatives.py [--write | --check]"""
import contextlib
import dataclasses
import sys

import d5b
import numpy as np

V5D, D4, RD = d5b.V5D, d5b.D4, d5b.D5.RD
B = d5b.BASE
ORIG = V5D.price


def patched(fn):
    @contextlib.contextmanager
    def ctx():
        V5D.price = fn
        try:
            yield
        finally:
            V5D.price = ORIG
    return ctx


def _adjust(p, g, new):
    d = new - p["gemm"][g]["gemm_ms"]
    p["gemm"][g]["gemm_ms"] = new
    p["body_ms"] += d
    if p["ms"] is not None:
        p["ms"] += d


def w1_whole(model, T, counts, prm, *, terms=V5D.NONE, **kw):
    """D5-3 alternative: s_t on the whole w1 price, ceiling and drain included."""
    p = ORIG(model, T, counts, prm, terms=terms, **kw)
    if "v5d" in p and p["v5d"]["w1_scale"] is not None:
        g = p["v5d"]["gemm"]["w1"]
        _adjust(p, "w1", (max(g["gemm_ms_v4dp"], g["ceiling_ms"]) + g["drain_ms"]) * p["v5d"]["w1_scale"])
    return p


def w2_scaled(model, T, counts, prm, *, terms=V5D.NONE, **kw):
    """D5-3 alternative: w2's bandwidth part at the counters' 3895.49 GB/s on every tile (DS assumed it, never read it)."""
    p = ORIG(model, T, counts, prm, terms=terms, **kw)
    if "v5d" in p and p["v5d"]["aligned"]:
        g = p["v5d"]["gemm"]["w2"]
        _adjust(p, "w2", max(g["bw_ms"] * D4.V4D_GBPS / D4.COUNTER_GBPS, g["ceiling_ms"]) + g["drain_ms"])
    return p


def _x(cfg, T, cls):
    reg = V5D.call_regressors(cfg, T, cls)
    return {"1": 1.0, "B_MB": reg["B_MB"], "x": float(cfg.intermediate_size if cls == "silu_and_mul"
                                                       else cfg.top_k * cfg.hidden_size)}


def _ds_line():
    from moe.spec import MODEL_CONFIGS as MC
    out = {}
    for cls in V5D.CALL_CLASSES:
        rs = [r for r in d5b.D5.call_rows(d5b.RD_) if r["class"] == cls]
        X = np.array([[_x(MC[r["model"]], r["T"], cls)[q] for q in ("1", "x", "B_MB")] for r in rs])
        out[cls] = np.linalg.lstsq(X, np.array([r["us"] for r in rs]), rcond=None)[0]
    return out


def call_ds_line(model, T, counts, prm, *, terms=V5D.NONE, **kw):
    """D5-4 alternative: DS's summary-line forms [1, x, B] (x = I or k H) refitted on the same rows."""
    from moe.spec import MODEL_CONFIGS as MC
    p = ORIG(model, T, counts, prm, terms=dataclasses.replace(terms, call={}), **kw)
    tot = sum(float(co @ np.array([_x(MC[model], T, c)[q] for q in ("1", "x", "B_MB")])) for c, co in LINE.items())
    d = tot * 1e-3 - p["other_ms"]
    p["body_ms"] += d
    if p["ms"] is not None:
        p["ms"] += d
    return p


LINE = _ds_line()


def shared_offset_regime():
    """D5-1 alternative: SUBWAVE re-read on DS's own ladders with one offset per (tile, GEMM) instead of one per
    ladder (the same rows, grid, weights and tie rule as d4common.fit_subwave); DRAIN re-read at it (d4common.fit_drain)."""
    lad = d5b.RD_["LADDERS"]["tiles"]
    reg, notes = {}, []
    for t in d5b.DC.DECODE_TILES:
        rows = [dict(x, ladder=k, gemm=k.split("/")[-1], model=k.split("/")[0], cross=x["wave"] != "sub-wave")
                for k, v in lad[t]["ladders"].items() for x in v]
        sub = {}
        for r in rows:
            if not r["cross"]:
                sub.setdefault(r["ladder"], []).append(r)
        use = [x for v in sub.values() if len(v) >= 2 for x in v]
        grp = sorted({x["gemm"] for x in use})
        X = np.zeros((len(use), len(grp)))
        for i, x in enumerate(use):
            X[i, grp.index(x["gemm"])] = 1.0
        L = np.array([x["L_us"] for x in use])
        best = None
        for r in D4.R_CTA_GRID:
            _b, res, _s = D4._lstsq(X, L - np.array([D4._ceil_us(x, r) for x in use]), 1 / L)
            obj = float(res @ res)
            if best is None or obj < best[0] - 1e-15:
                best = (obj, r, res)
        d = D4.fit_drain(rows, best[1])
        reg[t] = (best[1], d.get("r1_GBps"))
        notes.append(f"  {t}: shared-offset r_cta {best[1]:.2f} GB/s over {len(use)} rows, rms on DS's rows "
                     f"{float(np.sqrt(np.mean(best[2] ** 2))):.4f}; DRAIN at it r1 {d.get('r1_GBps')}")
    return dataclasses.replace(B, regime=reg), notes


if __name__ == "__main__":
    INT = d5b.RD_["ALIGN"]["SLOPE"]["intercepts_us"]
    sb = tuple((int(E), float(v)) for E, v in sorted(INT.items(), key=lambda kv: int(kv[0])))
    so, notes = shared_offset_regime()
    rows = [("v5d as registered", B, contextlib.nullcontext),
            ("D5-1 SUBWAVE one offset per tile/GEMM", so, contextlib.nullcontext),
            ("D5-3 W1 on the whole w1 price", B, patched(w1_whole)),
            ("D5-3 w2 at the counters' rate", B, patched(w2_scaled)),
            ("D5-4 CALL as DS's summary lines", B, patched(call_ds_line)),
            ("D5-5 SLOPE's per-E intercepts", dataclasses.replace(B, align=dict(B.align, **{RD.SMALL_BATCH: sb})),
             contextlib.nullcontext),
            ("D5-5 no slope", dataclasses.replace(B, slope=0.0), contextlib.nullcontext)]
    L = [d5b.line(d5b.run(t, k, blind=False, ctx=c)) for k, t, c in rows] + ["D5-1's shared-offset re-read:"] + notes
    sys.exit(d5b.emit(L, "alternatives", sys.argv[1:]))
