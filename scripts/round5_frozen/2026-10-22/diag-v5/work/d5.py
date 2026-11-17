"""Shared helpers for the v5 offline diagnosis (read-only).

Imports the design's prototype (design-v5/work/v5price.py) and DIAGNOSIS-A2's price_parts read-only. Nothing in the
repo, the exfil trees or any earlier work folder is written (run with PYTHONDONTWRITEBYTECODE=1).

Pricer first-build trap: the CAL engine's v2 fit is built on the first RG.pricer() call and reads TM.P_KNEE then.
This module builds it at import, at v4's own p 14, and asserts the carried fit (T0 43.23 us, bw 3597.8 GB/s).
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from contextlib import contextmanager
from pathlib import Path

sys.dont_write_bytecode = True
W5 = Path.home() / "code/moe-kernels-work/2026-10-21/design-v5/work"
sys.path.insert(0, str(W5))
import v5price as V  # noqa: E402
from v5price import A2, FN, RG, RULES, SCORE, TC, TM, V4, params_for  # noqa: E402
from common import price_parts  # noqa: E402  (diag-A2/work, on sys.path via v5price)

A2T = Path.home() / "moe-kernels-exfil/runs/gh200-tile2-20261020T2242Z/tree/vm/results"
S1T = Path.home() / "moe-kernels-exfil/runs/gh200-tile1-20261021T0704Z/tree/vm/results"

_PR = RG.pricer()
assert TM.P_KNEE == 14.0
_X = [float(v) for v in _PR.f2.x]
assert abs(_X[0] * 1e3 - 43.23) < 0.01 and abs(_X[2] - 3597.8) < 0.1, _X
SEL = FN.selections(SCORE)
F3P = V.f3_params()
P5 = V.knee_p()
assert P5 == {128: 5.75, 64: 3.75}, P5
R3_1410 = (6.11, 50.0)      # rental 3 R3 tile at 1410 (sag.json), MEASURED
R3_1710 = (16.66, 275.0)    # rental 3 R3 tile at 1710 (sag.json), MEASURED
BOARD_A2 = {128: V.SAG[1410][128], 64: V.SAG[1410][64]}


def sat(A_us, tau_us, t_ms):
    """A (1 - exp(-t / tau)) in ms."""
    return A_us * 1e-3 * (1 - math.exp(-t_ms / (tau_us * 1e-3)))


@contextmanager
def bw_set(bw):
    """the CAL engine fit's bw (x[2]) set for one block (b2risk.bw_scaled's mechanism), restored on exit."""
    fit = RG.pricer().f2
    keep = fit.x
    try:
        x = keep.copy()
        x[2] = bw
        fit.x = x
        yield
    finally:
        fit.x = keep


@contextmanager
def c_patch(fac):
    base = V4.c_scale
    V4.c_scale = lambda t_, g, prm_, _b=base, _f=fac: _b(t_, g, prm_) * _f[g]
    try:
        yield
    finally:
        V4.c_scale = base


def v5_parts(model, c, doc, prm, *, reads=None, F3=None, P=None, rho=True, sag="board", sag_const=None,
             own_off_us=None, r3=R3_1410, key="v3", sag_scale=1.0):
    """v5's price of one NATIVE cell with every term (price_parts under v5's swaps and F1 floor inflation).
    sag: 'board' (A, tau per tile, A2's pooled unless sag_const), 'own' (the cell's own measured off per GEMM, us),
    'none'. r3: the R3 share subtracted (T0 holds it); None subtracts nothing."""
    F3 = F3P if F3 is None else F3
    P = P5 if P is None else P
    pr = RG.pricer()
    bm = int(doc["block_m"])
    t = RG.PAGE_TILE[bm]
    n, counts, G = int(c["n"]), c["counts"], int(doc["group_m"])
    seed = RG.seed_of(doc, c)
    with V.swapped(P[bm], V.RHO_X[bm] if rho else None):
        if reads is None:
            reads, _q = V.f3_reads(pr, model, n, counts, seed, t, G, prm, F3[bm], key)
        base = price_parts(model, "native", c, doc, prm, key=key, reads=reads)
        if sag == "none":
            base["fac"] = {"w1": 1.0, "w2": 1.0}
            return base
        fac = {}
        for g in ("w1", "w2"):
            tf = base[g]["floor_only"]
            if sag == "board":
                A, tau = sag_const or BOARD_A2[bm]
                D = sat(A * sag_scale, tau, tf)
            else:
                D = own_off_us[g] * 1e-3
            if r3:
                D -= sat(r3[0], r3[1], tf)
            fac[g] = 1 + D / tf
        with c_patch(fac):
            out = price_parts(model, "native", c, doc, prm, key=key, reads=reads)
        out["fac"] = fac
        out["tf"] = {g: base[g]["floor_only"] for g in ("w1", "w2")}
        return out


def gemm_ms(p):
    return sum(p[g]["gemm"] + p[g]["dead"] + p[g]["const"] for g in ("w1", "w2"))


def percall_ms(p):
    return p["T0"] + p["other"] + p["s"] + p.get("b", 0.0)


def counter_page(root, model, unit, lock=1410):
    f = sorted(root.glob(f"*-{model}-{unit}-r3-counters/lock{lock}/r3c-g*.json"))
    assert len(f) == 1, (model, unit, f)
    p = json.loads(f[0].read_text())
    out = {}
    for c in p["cells"]:
        lab = c.get("histogram") or "balanced"
        d = {}
        for g in ("w1", "w2"):
            pg, rec = c["per_gemm"][g], (c.get("recorded") or {}).get(g) or {}
            cyc = float(pg.get("sm__cycles_elapsed.avg") or rec["sm__cycles_elapsed.avg"])
            tn = float(pg.get("gpu__time_duration.sum") or pg.get("gpu_time_ns"))
            d[g] = {"cyc": cyc, "t_us": tn / 1e3, "reads": float(pg["dram_bytes_read"]), "off_us": tn / 1e3 - cyc / lock}
        out[(int(c["n"]), lab, c["arm"])] = {"g": d, "bincount": c.get("bincount")}
    return p, out, f[0]


def e1cal_rows():
    """A2's E1-CAL NATIVE rows (the design's loader, the registered scorer's selections and drops)."""
    with A2.scoring(RULES):
        _sel, rows = V.e1cal_rows()
    return rows


def pct(x):
    return round(100 * x, 2)


def rms(v):
    v = list(v)
    return math.sqrt(st.fmean(x * x for x in v)) if v else float("nan")
