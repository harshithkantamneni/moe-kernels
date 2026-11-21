"""Addendum 6's build law with switches (build-D6's exploration law, committed): v6d's frozen-candidate form with every
open choice a switch, so the build records (d6_build/*.out) can recompute each alternative D6-x names. Pure numpy over
the committed readings: PC (pc.read.json, d6_build/pc_spans.json), PC2 (pc2.read.json), DS (ds.readings.json). At
CANDIDATE it prices every GEMM as r3_timing_model_v6d.LAW does (a test holds the two equal); the model file's numbers,
not this module, are what addendum 6 registers.

    life_t(m) = c0_t(m) + b L_t(N_eff) / Q_t          (duty load N_eff = m Q_t phi, phi = stream / life)
    life_t(m) = max(life, c0_t(m) + m b / X_t)        (the ceiling)
    wave_t(m) = kappa_t(m / S_t) x life_t(m)          (kappa by fill)
    multi-wave: blend+ as v6d
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SMS = 132
Q_A, B_A = 32768, 3670016
T64, T16, T32a, T32b = "64x128x64w4s3", "16x64x128w4s4", "32x64x128w4s3", "32x128x64w4s3"
TILES = (T16, T32a, T32b, T64)
OCC = {T16: 3, T64: 3, T32a: 4, T32b: 4}
N_IF = {T16: 3, T32a: 2, T32b: 2, T64: 2}
PC = json.loads((REPO / "scripts/scoring/pc/pc.read.json").read_text())
PC2 = json.loads((REPO / "scripts/scoring/pc2/pc2.read.json").read_text())
DS = json.loads((REPO / "scripts/scoring/decode/ds.readings.json").read_text())
SPANS = json.loads((HERE / "pc_spans.json").read_text())
DS_W1 = {t: v["counter_GBps"] for t, v in DS["W1RATE"]["tiles"].items()}
X_PC = float(PC["PC-R7"]["X_sat"])
C0_PC128 = float(PC["PC-R2"]["G128 (Mixtral tp8, tp4, tp2, 8x7B)"]["c0_us"])
C0_PC2_G8_64 = float(PC2["PC2-R2"]["R2a c0 per G (F 256 to 14336)"]["G 8"]["c0_us"])
C0_PC2_G8_16 = float(PC2["PC2-R1"]["R1b F ladders"]["G 8"]["c0_us"])
X_PC2_16 = float(PC2["PC2-R1"]["R1d X_sat"]["X_sat"])


def tile_dims(key):
    a, rest = key.split("w")
    bm, bn, bk = map(int, a.split("x"))
    w, s = rest.split("s")
    return bm, bn, bk, int(w), int(s)


def q_bytes(tile):
    _bm, bn, bk, _w, _s = tile_dims(tile)
    return N_IF[tile] * bk * bn * 2


def pc_rows():
    lam = {r["cell"]: r["lam_us"] for r in PC["PC-R1"]["A (D64, 1710)"] if str(r["status"]).startswith("READ")}
    return sorted([dict(r, lam_us=lam[r["label"]]) for r in PC["PC-R7"]["rows"] if r["label"] in lam], key=lambda r: r["n"])


def pc_kappa():
    """PC's span / median life on block A's w2 single-wave rungs (median over valid calls), by the rung's n."""
    by = {}
    for r in SPANS:
        if r["valid"] and r["unit"] == "sa" and r["gemm"] == "w2":
            by.setdefault(r["cell"], []).append(r)
    out = []
    for row in pc_rows():
        v = by.get(row["label"])
        if v and row["G"] <= 384:
            out.append((row["n"], float(np.median([x["span_us"] for x in v])) / float(np.median([x["life_med_us"] for x in v]))))
    return out


def pc2_kappa16():
    """PC2-R1a's READ kappa on 16x64x128 at the long slab (1.835 MB), single-wave rungs, by n."""
    return [(r["n"], r["kappa"]) for r in PC2["PC2-R1"]["R1a rate curve"]
            if r["cell"].startswith("D-G") and "-F" not in r["cell"] and str(r["status"]).startswith("READ")
            and r.get("kappa") is not None]


def pc2_kappa_r3(tile):
    """PC2-R3's READ kappa on a 4-per-SM tile at the long slab (b >= 1.8 MB), single-wave rungs, by n."""
    return [(r["n"], r["kappa"]) for r in PC2["PC2-R3"][tile]["rows"]
            if r.get("kappa") is not None and r["b"] >= 1.8e6 and str(r.get("status", "READ")).startswith("READ")]


def c0_pts(tile, form):
    if form == "const":
        return ((1, C0_PC128),)
    if form == "const_lo":
        return ((1, C0_PC2_G8_64),)
    lo = C0_PC2_G8_16 if tile == T16 else C0_PC2_G8_64
    if form == "interp":           # linear in G between PC2's G 8 and PC's G 128, held outside
        return ((8, lo), (128, C0_PC128))
    if form == "interp_logG":      # linear in ln G (handled by c0_at)
        return ((8, lo), (128, C0_PC128))
    raise ValueError(form)


@dataclass(frozen=True)
class Law:
    c0_form: str = "interp"           # const | const_lo | interp | interp_logG
    L16: str = "pc"                    # pc: PC's 64x128x64 table in bytes in flight | pc2: PC2-R1a's own rungs
    kap: str = "own+pcfill"           # v6d (PC by m, held) | pcfill | own16 | own32 | own+pcfill | fill_pooled
    x32: str = "ds_ratio"             # ds_ratio | pc2_r3
    x16: str = "pc2"                  # pc2 | ds_ratio
    duty: bool = True
    multi: str = "blend+"
    floor: bool = True
    c0_multi: str = "last"           # last: the remainder wave at c0(k'); full: every wave of a multi-wave GEMM at c0(S)
    naive: str = "v4dp"             # v4dp | twin | fetchers | all@fetch
    c0: dict = field(default_factory=dict)
    Ltab: dict = field(default_factory=dict)
    ktab: dict = field(default_factory=dict)
    xs: dict = field(default_factory=dict)

    def c0_at(self, tile, m):
        pts = self.c0[tile]
        if len(pts) == 1:
            return pts[0][1]
        (g0, a), (g1, b) = pts
        if m <= g0:
            return a
        if m >= g1:
            return b
        if self.c0_form == "interp_logG":
            return a + (b - a) * (math.log(m) - math.log(g0)) / (math.log(g1) - math.log(g0))
        return a + (b - a) * (m - g0) / (g1 - g0)

    def build(self):
        c0 = {t: c0_pts(t, self.c0_form) for t in TILES}
        tmp = replace(self, c0=c0)
        # the 64x128x64 L table off PC's rungs (G <= 384), c0 at each rung's G removed
        rows = [r for r in pc_rows() if r["G"] <= 384]
        Lt, Nt = [], []
        for r in rows:
            c = tmp.c0_at(T64, r["n"])
            L = Q_A * (r["lam_us"] - c) / B_A
            Lt.append(L)
            Nt.append(r["n"] * Q_A * (r["lam_us"] - c) / r["lam_us"] if self.duty else r["n"] * Q_A)
        Ltab = {t: (tuple(Nt), tuple(Lt)) for t in TILES}
        if self.L16 == "pc2":
            Q, b = q_bytes(T16), 1835008.0
            rs = [r for r in PC2["PC2-R1"]["R1a rate curve"] if r["cell"].startswith("D-G") and "-F" not in r["cell"]
                  and str(r["status"]).startswith("READ") and r["G"] <= 384]
            rs.sort(key=lambda r: r["n"])
            L2 = [Q * (r["lam_us"] - tmp.c0_at(T16, r["n"])) / b for r in rs]
            N2 = [r["n"] * Q * (r["lam_us"] - tmp.c0_at(T16, r["n"])) / r["lam_us"] for r in rs]
            Ltab[T16] = (tuple(N2), tuple(L2))
        pk = pc_kappa()
        S3, S4 = 396, 528
        ktab = {}
        if self.kap == "v6d":         # by m, absolute (v6d frozen)
            for t in TILES:
                ktab[t] = tuple((m / (SMS * OCC[t]), k) for m, k in pk)    # stored by fill of the tile ... see kappa_at
            ktab["_by_m"] = tuple(pk)
        elif self.kap in ("own16", "own32"):
            pcf = [(m / S3, k) for m, k in pk]
            for t in TILES:
                ktab[t] = tuple(pcf)
            if self.kap == "own16":
                own16 = [(m / S3, k) for m, k in pc2_kappa16()]
                ktab[T16] = tuple([p for p in pcf if p[0] < own16[0][0]] + own16)
            else:
                for t in (T32a, T32b):
                    own = sorted((m / S4, k) for m, k in pc2_kappa_r3(t))
                    ktab[t] = tuple([p for p in pcf if p[0] < own[0][0]] + own)
        elif self.kap == "pcfill":
            pcf = [(m / S3, k) for m, k in pk]
            for t in TILES:
                ktab[t] = tuple(pcf)
        elif self.kap in ("own+pcfill", "r3_by_m"):
            pcf = [(m / S3, k) for m, k in pk]
            ktab[T64] = tuple(pcf)
            own16 = [(m / S3, k) for m, k in pc2_kappa16()]
            ktab[T16] = tuple([p for p in pcf if p[0] < own16[0][0]] + own16)
            for t in (T32a, T32b):
                own = sorted((m / S4, k) for m, k in pc2_kappa_r3(t))
                ktab[t] = tuple([p for p in pcf if p[0] < own[0][0]] + own)
        elif self.kap == "fill_pooled":
            pts = [(m / S3, k) for m, k in pk] + [(m / S3, k) for m, k in pc2_kappa16()]
            for t in (T32a, T32b):
                pts += [(m / S4, k) for m, k in pc2_kappa_r3(t)]
            pts.sort()
            # median in fill bins of 0.1 (and the low-fill points as they are)
            edges = [0, .03, .06, .12, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
            tab = []
            for a, b in zip(edges[:-1], edges[1:]):
                s = [p for p in pts if a <= p[0] < b]
                if s:
                    tab.append((float(np.median([p[0] for p in s])), float(np.median([p[1] for p in s]))))
            for t in TILES:
                ktab[t] = tuple(tab)
        xs = {T64: X_PC}
        for t in (T32a, T32b):
            xs[t] = X_PC * DS_W1[t] / DS_W1[T64] if self.x32 == "ds_ratio" else float(PC2["PC2-R3"][t]["X_sat"]["X_sat"])
        xs[T16] = X_PC2_16 if self.x16 == "pc2" else X_PC * DS_W1[T16] / DS_W1[T64]
        return replace(tmp, Ltab=Ltab, ktab=ktab, xs=xs)

    def kappa_at(self, tile, m):
        if self.kap == "v6d":
            mm, kk = zip(*self.ktab["_by_m"])
            return float(np.interp(m, mm, kk))
        f, k = zip(*self.ktab[tile])
        return float(np.interp(m / (SMS * OCC[tile]), f, k))

    def L_at(self, tile, m, Q, b, c=None):
        Nt, Lt = self.Ltab[tile]
        X = self.xs[tile]

        def Lx(x):
            if x <= Nt[-1]:
                return float(np.interp(x, Nt, Lt))
            return max(Lt[-1], x / (X * 1e3))
        if not self.duty:
            return Lx(m * Q)
        c = self.c0_at(tile, m) if c is None else c
        L = Lt[0]
        for _ in range(80):
            s = b * L / Q
            phi = s / (c + s)
            L2 = Lx(m * Q * phi)
            if abs(L2 - L) < 1e-10:
                break
            L = L2
        return L

    def life(self, tile, m, b, floor_us=0.0, rho=1.0, cm=None):
        """rho: the fraction of the m CTAs that fetch their slab from DRAM (1 on the aligned paths; a naive GEMM's
        leaders over its live CTAs): the load and the ceiling's bytes are rho m, every CTA lives at that L."""
        Q = q_bytes(tile)
        c = self.c0_at(tile, m if cm is None else cm)
        L = self.L_at(tile, rho * m, Q, b, c)
        t = max(c + b * L / Q, c + rho * m * b / (self.xs[tile] * 1e3))
        if self.floor and floor_us:
            t = max(t, math.ceil(m / SMS) * floor_us)
        return t

    def wave(self, tile, m, b, floor_us=0.0, rho=1.0, cm=None):
        return self.kappa_at(tile, m) * self.life(tile, m, b, floor_us, rho, cm)

    def gemm_us(self, tile, n, b, floor_us=0.0, S=None, rho=1.0):
        if n <= 0:
            return 0.0
        S = S or SMS * OCC[tile]
        if n <= S:
            return self.wave(tile, n, b, floor_us, rho)
        W, k = divmod(n, S)
        last = k if k else S
        X = self.xs[tile] * 1e3
        if self.multi == "blend+":
            return (rho * (n - last) * b / X + (self.kappa_at(tile, S) - 1.0) * self.life(tile, S, b, floor_us, rho)
                    + self.wave(tile, last, b, floor_us, rho, S if self.c0_multi == "full" else None))
        if self.multi == "blend":
            return (n - last) * b / X + self.wave(tile, last, b, floor_us)
        raise ValueError(self.multi)


#: addendum 6's frozen candidate (r3_timing_model_v6d) in this module's switches
CANDIDATE = dict(c0_form="interp", c0_multi="full", x16="pc2", kap="own32", naive="twin-nofollow", L16="pc", x32="ds_ratio")
#: design-D6's frozen v6d in this module's switches (c0 4.70 constant, kappa by m held, X_sat x DS ratio, v4d' naive)
DESIGN_D6 = dict(c0_form="const", c0_multi="last", x16="ds_ratio", kap="v6d", naive="v4dp", L16="pc", x32="ds_ratio")


def variant(**kw) -> Law:
    return Law(**dict(CANDIDATE, **kw)).build()


# --------------------------------------------------------------------------
# the switches where the frozen scorers price a cell (CAL2 only in the build records)
# --------------------------------------------------------------------------

def price(law: Law, model, T, counts, prm, *, ids=None, fixes=None, terms=None, declared=None, with_constants=True):
    """r3_timing_model_v5d.price at `terms` with `law`'s GEMM prices on the aligned paths and on the naive path as its
    `naive` switch says (v4dp | twin | fetchers | all@fetch | twin-nofollow)."""
    import r3_timing_model_v6d as V6D
    V5D, TM, RD = V6D.V5D, V6D.TM, V6D.RD
    rec = {}
    orig = TM.gemm_ms

    def gemm_ms(win, gemm, sigma, c_ns, bw, sms, p=None, occ=None):
        geo = TM.GEOMETRY[gemm]
        n = int(win.live)
        rec[gemm] = {"n": n, "S": int(sms * occ) if occ else None, "slab": float(geo.slab),
                     "floor_us": TM.floor_ksteps(gemm) * c_ns * 1e-3, "fetch": V6D._fetches(win, geo, n) if n else 0.0}
        return orig(win, gemm, sigma, c_ns, bw, sms, p, occ)
    TM.gemm_ms = gemm_ms
    try:
        p = V5D.price(model, T, counts, prm, ids=ids, fixes=fixes if fixes is not None else V6D.V4DP.ALL,
                      terms=terms, declared=declared, with_constants=with_constants)
    finally:
        TM.gemm_ms = orig
    v = p.get("v5d")
    if v is None:
        return p
    naive = p["path"] == RD.NAIVE and law.naive != "v4dp"
    if naive and law.naive == "twin-nofollow" and any(abs(r["fetch"] - r["n"]) >= 0.5 for r in rec.values() if r["n"]):
        naive = False
    if not (v["aligned"] or naive):
        return p
    add = 0.0
    for g in TM.GEMMS:
        r = rec[g]
        if naive and law.naive == "fetchers":
            t = law.gemm_us(v["tile"], max(1, round(r["fetch"])), r["slab"], r["floor_us"], r["S"])
        elif naive and law.naive == "all@fetch":
            t = law.gemm_us(v["tile"], r["n"], r["slab"], r["floor_us"], r["S"], rho=min(1.0, r["fetch"] / r["n"]))
        else:
            t = law.gemm_us(v["tile"], r["n"], r["slab"], r["floor_us"], r["S"])
        add += t * 1e-3 - p["gemm"][g]["gemm_ms"]
        p["gemm"][g]["gemm_ms"] = t * 1e-3
    p["body_ms"] += add
    if p["ms"] is not None:
        p["ms"] = p["body_ms"] + p["const_ms"]
    return p


def cal2(law: Law) -> dict:
    """CAL2 under `law` (score_cal_d3's rules, the CAL constants refitted): the clause-1 reading."""
    import contextlib
    import sys as _s
    _s.path.insert(0, str(REPO / "scripts/scoring/decode"))
    import d3common as D3
    import d4common as D4
    import d6common as D6
    import dcommon as DC
    import score_cal_d3 as SC3
    terms = D6.registered_terms()
    orig = D3.price_cell_v4dp

    def price_cell_v4dp(model, cell, prm, arm="native", decl_copies=None, fixes=D6.V4DP.ALL):
        from moe.spec import MODEL_CONFIGS
        cfg = MODEL_CONFIGS[model]
        declared = None if arm == "native" else cfg.num_experts * int(decl_copies or 1)
        ids = None
        if fixes.naive_order and D6.RD.geometry(cfg, int(cell["T"]), cell["counts"], declared).path == D6.RD.NAIVE:
            ids = D6.V4DP.ids_of(model, cell)
        try:
            p = price(law, model, int(cell["T"]), cell["counts"], prm, ids=ids, fixes=fixes, terms=terms, declared=declared)
        except D6.TM.Refused:
            return None
        if prm.T0_GR_ms is not None and p["ms"] is None:
            return None
        return p
    tp, src = D4.tile_source()
    D3.price_cell_v4dp = price_cell_v4dp
    try:
        with contextlib.nullcontext():
            return SC3.score(REPO / D4.CAL2_PUBLISHED / "results", DC.registration(REPO, "cal"), tp, src, settle=False)
    finally:
        D3.price_cell_v4dp = orig


def cal2_line(label: str, c: dict) -> str:
    st, e = c["STOP"], c["E1_CAL_IN_SAMPLE"]["stats"]
    sel = c["FIT"]["selected"]
    f = c["FIT"]["forms"][sel] if sel else {}
    us = lambda k: (f.get(k) or 0.0) * 1e3  # noqa: E731
    rs = sorted(c["rows"], key=lambda r: -abs(r["e"]))[:2]
    return (f"{label:40s} CAL2 {st['verdict']:4s} rms {e['rms']:.4f} worst {e['worst']:+.4f} mean {e['mean']:+.4f} n {e['cells']} "
            f"T0 {us('T0_GR_ms'):.2f} sn {us('s_naive_GR_ms'):.2f} sb {us('s_block_GR_ms'):.2f} | worst: "
            + "; ".join(f"{r['key'].split('/native')[0]} {r['e']:+.2%}" for r in rs))


def emit(lines: list[str], name: str, argv) -> int:
    """Print the lines; --write saves <name>.out beside; --check compares (exit 1 on a difference)."""
    import sys as _s
    text = "\n".join(lines) + "\n"
    out = HERE / f"{name}.out"
    if "--write" in argv:
        out.write_text(text)
        return 0
    if "--check" in argv:
        same = out.exists() and out.read_text() == text
        print(f"{out.name}: {'same' if same else 'DIFFERS'}")
        return 0 if same else 1
    _s.stdout.write(text)
    return 0
