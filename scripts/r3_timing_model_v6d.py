#!/usr/bin/env python
"""MODEL v6d: model v5d (`r3_timing_model_v5d`, byte-identical, imported and called; it runs over v4d') with every
aligned GEMM, and every naive GEMM of a cell with no non-fetching follower, priced by one GEMM-time law built from
rentals PC and PC2 (registration 2026-11-10-decode-addendum6-gh200; design-D6 and build-D6). CALL, ALIGN and SLOPE
stay v5d's (DS's readings); v5d's regimes and W1 scale are emptied (the law's tile ceiling replaces them). With the
law off (`law=None`), `price` returns r3_timing_model_v5d.price's own value at the given terms, so v6d reduces to v5d.

Per GEMM of a decode call, n live CTAs on S = 132 x occ(tile) slots, each streaming a B slab of b = BLOCK_N x K x 2:

    c0(m)  = C0[tile] piecewise linear in m between its two points (G 8, G 128), held outside        (D6-18)
    L(m)   = interp of L_TABLE over N_eff = m Q phi, phi = (b L / Q) / (c0 + b L / Q) (duty, a fixed point);
             past the table's last point L = N_eff / X_SAT[tile] (Little's law at the ceiling)       (D6-2, D6-19)
    life(m)= max(c0 + b L / Q,  c0 + m b / X_SAT[tile],  ceil(m / 132) x floor)                      (D6-6, D6-20)
    wave(m)= kappa(m / S) x life(m),  kappa piecewise linear in the fill m / S over KAPPA[tile], held outside
                                                                                                     (D6-5, D6-21, D6-22)
    gemm   = wave(n)                                                          if n <= S
    gemm   = (n - k) b / X + (kappa(1) - 1) life(S) + wave(k)  every wave at c0(S)   if n > S, k = n mod S (or S)
                                                                                                     (D6-1, D6-18)
  Q = Q_B[tile] = n_if x BLOCK_K x BLOCK_N x 2 (the B bytes one CTA keeps in flight, off the compiled PTX)
  naive path: a cell whose naive GEMMs fetch every CTA's slab (no follower, or every follower re-fetches under v4d''s
  fix W) is priced GEMM by GEMM by the law as its aligned twin (PC2-R4); a cell with any non-fetching follower keeps
  v4d''s naive GEMM prices (D6-24)

Every constant below is a registered number. scripts/scoring/decode/d6common.terms_from recomputes each one
mechanically from the committed readings (PC, PC2, DS, PC's span reading and the PTX scan) and the registration's
--check holds them equal to 1e-9 relative.
"""
from __future__ import annotations

import bisect
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r3_timing_model_v5d as V5D  # noqa: E402

V4DP, TM, RD = V5D.V4DP, V5D.TM, V5D.RD
SMS = 132
T16, T32A, T32B, T64 = "16x64x128w4s4", "32x64x128w4s3", "32x128x64w4s3", "64x128x64w4s3"
TILES = (T16, T32A, T32B, T64)

#: B tiles in flight (PC's PTX rule, pccommon.inflight_stages): 16x64x128w4s4 3 (PC2-R1g on the stamped s4 PTX and
#: 160 published PTX files), 32x64x128w4s3 2 (46 files), 32x128x64w4s3 2 (50 files), 64x128x64w4s3 2 (PC-R0)
N_IF = {T16: 3, T32A: 2, T32B: 2, T64: 2}
#: c0 (us per CTA) against the live CTAs m: (G, c0) at G 8 (PC2-R1b on 16x64x128, PC2-R2a on 64x128x64; the 4-per-SM
#: tiles take 64x128x64's, never read on their own) and G 128 (PC-R2, read on 64x128x64; 16x64x128's G 128 is NOT
#: IDENTIFIED in PC2-R1b, so it keeps PC's)
C0 = {
    T16: ((8, 2.7208310295312446), (128, 4.704722441735574)),
    T32A: ((8, 2.4001601638571275), (128, 4.704722441735574)),
    T32B: ((8, 2.4001601638571275), (128, 4.704722441735574)),
    T64: ((8, 2.4001601638571275), (128, 4.704722441735574)),
}
#: the per-CTA streaming latency L (us per Q bytes) against the duty load N_eff (bytes in flight): PC-R7's rungs G 4
#: to G 384 (64x128x64w4s3, Q 32 KB, slab 3.67 MB) with PC-R1's median life, c0(n) removed at each rung:
#: L = Q (life - c0) / b, N_eff = n Q (life - c0) / life. One table for every tile (16x64x128's own, PC2-R1c, is NOT
#: IDENTIFIED)
L_TABLE = (
    (127969.59523710413, 255779.53524837442, 510664.08469548373, 1019059.4487052565, 2030052.499304114,
     4054203.3963742144, 5082196.801023465, 6077169.889670409, 8193944.004619163, 9239649.659211524,
     12106409.408297844),
    (0.8839557771282307, 0.8947048094385878, 0.9435207824643678, 0.9733047165459909, 1.0384092160845613,
     1.381130963833698, 1.5873115111097962, 1.8233470888617949, 2.286611264327609, 2.579574761930008,
     3.526293668752152),
)
#: the GPU ceiling per tile (GB/s): 64x128x64 PC-R7; 16x64x128 PC2-R1d; the 4-per-SM tiles PC-R7 x DS's W1RATE tile
#: ratio (as frozen in design-D6)
X_SAT = {T16: 3781.891121411795, T32A: 3703.847713349612, T32B: 3694.3528528800434, T64: 3612.4196138187617}
_PC_KAPPA = ((0.010101010101010102, 1.0), (0.02018367696333798, 1.0024213075060533),
             (0.04030489654046236, 1.0092165898617511), (0.0805934060967618, 1.0111358574610243),
             (0.1611153286518631, 1.0210084033613445), (0.3219179173437849, 1.0646687697160884),
             (0.402022201092449, 1.0537190082644627), (0.4791241524427666, 1.0650602409638557),
             (0.6430633065573189, 1.0398445092322643), (0.7236450241699585, 1.0461672473867596),
             (0.9440894244392145, 1.0319284802043422))
#: the span factor (the slowest CTA over the median) by fill m / S: PC's block-A w2 single-wave rungs (span over median
#: life, d6_build/pc_spans.py) at S 396 on 64x128x64 and 16x64x128; on the 4-per-SM tiles PC2-R3's READ rungs at K
#: 14336 (S 528) and, below the tile's lowest read fill, PC's
KAPPA = {
    T64: _PC_KAPPA,
    T16: _PC_KAPPA,
    T32A: _PC_KAPPA[:8] + ((0.4967664957768306, 1.079293049916408), (0.7399951323128958, 1.1003499633759257),
                           (0.8429986461454643, 1.0936806698633077), (0.9710788638131717, 1.0257858546168959)),
    T32B: _PC_KAPPA[:6] + ((0.3707799133913973, 1.0771679805942995), (0.49723462513452776, 1.0384155098133079),
                           (0.7343658836104111, 1.1048973524567112), (0.9624030643312559, 1.0193166691317852)),
}


def tile_dims(key: str) -> tuple[int, int, int, int, int]:
    a, rest = key.split("w")
    bm, bn, bk = map(int, a.split("x"))
    w, s = rest.split("s")
    return bm, bn, bk, int(w), int(s)


def q_bytes(tile: str) -> int:
    _bm, bn, bk, _w, _s = tile_dims(tile)
    return N_IF[tile] * bk * bn * 2


def _interp(x: float, xs, ys) -> float:
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    i = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))


class Law:
    """The registered law (constants above; `constants` overrides them, for the build's checks only)."""

    def __init__(self, constants: dict | None = None):
        c = constants or {}
        self.c0 = c.get("C0", C0)
        self.L = c.get("L_TABLE", L_TABLE)
        self.x = c.get("X_SAT", X_SAT)
        self.kap = c.get("KAPPA", KAPPA)
        self.n_if = c.get("N_IF", N_IF)

    def q(self, tile: str) -> int:
        _bm, bn, bk, _w, _s = tile_dims(tile)
        return self.n_if[tile] * bk * bn * 2

    def c0_at(self, tile: str, m: float) -> float:
        (g0, a), (g1, b) = self.c0[tile]
        return _interp(m, (g0, g1), (a, b))

    def kappa_at(self, tile: str, fill: float) -> float:
        pts = self.kap[tile]
        return _interp(fill, [p[0] for p in pts], [p[1] for p in pts])

    def _L_of_load(self, tile: str, x: float) -> float:
        N, L = self.L
        if x <= N[-1]:
            return _interp(x, N, L)
        return max(L[-1], x / (self.x[tile] * 1e3))

    def L_at(self, tile: str, m: float, b: float, c0: float) -> float:
        """us: the latency at m live CTAs streaming b bytes after c0 (the duty fixed point)."""
        Q = self.q(tile)
        L = self.L[1][0]
        for _ in range(80):
            s = b * L / Q
            L2 = self._L_of_load(tile, m * Q * s / (c0 + s))
            if abs(L2 - L) < 1e-10:
                return L2
            L = L2
        return L

    def life(self, tile: str, m: int, b: float, floor_us: float, c0_m: float) -> float:
        """us: one CTA's life (the median CTA) in a wave of m CTAs, c0 read at c0_m live CTAs."""
        c = self.c0_at(tile, c0_m)
        L = self.L_at(tile, m, b, c)
        t = max(c + b * L / self.q(tile), c + m * b / (self.x[tile] * 1e3))
        if floor_us:
            t = max(t, math.ceil(m / SMS) * floor_us)
        return t

    def gemm_us(self, tile: str, n: int, S: int, b: float, floor_us: float = 0.0) -> float:
        """us: one GEMM of n live CTAs on S slots."""
        if n <= 0:
            return 0.0
        if n <= S:
            return self.kappa_at(tile, n / S) * self.life(tile, n, b, floor_us, n)
        k = n % S or S
        X = self.x[tile] * 1e3
        return ((n - k) * b / X + (self.kappa_at(tile, 1.0) - 1.0) * self.life(tile, S, b, floor_us, S)
                + self.kappa_at(tile, k / S) * self.life(tile, k, b, floor_us, S))

    def as_dict(self) -> dict:
        return {"C0": {t: [list(p) for p in v] for t, v in self.c0.items()},
                "L_TABLE": [list(self.L[0]), list(self.L[1])], "X_SAT": dict(self.x),
                "KAPPA": {t: [list(p) for p in v] for t, v in self.kap.items()}, "N_IF": dict(self.n_if)}


LAW = Law()


def v6_terms(base: V5D.Terms) -> V5D.Terms:
    """v5d's terms with the regimes emptied on every decode tile (no ceiling, no drain: v5d removes the CORES tail and
    the law replaces the GEMM) and no W1 scale (the law's tile ceiling carries DS's W1RATE ratio); CALL, ALIGN and SLOPE
    as `base` (v5d's registered terms)."""
    return V5D.Terms(regime={t: (None, None) for t in TILES}, w1_scale={}, call=dict(base.call),
                     align=dict(base.align), slope=base.slope)


def _fetches(win, geo, n: int) -> float:
    """The slab fetches of a GEMM from v4d''s reads (fix W's on the naive path): (reads - n x act) / slab."""
    act = geo.arow / geo.npn / TM.BLOCK_M
    return (float(win.reads) - n * act) / geo.slab if geo.slab else float(n)


def price(model: str, T: int, counts, prm, *, ids=None, fixes=V4DP.ALL, terms: V5D.Terms = V5D.NONE,
          declared: int | None = None, with_constants: bool = True, law: Law | None = LAW) -> dict:
    """v6d's time of one decode cell (ms), its parts and its bytes: r3_timing_model_v5d.price at `terms` with the law's
    GEMM prices in place on the aligned paths and on a naive cell with no non-fetching follower. law None: v5d."""
    rec: dict = {}
    orig = TM.gemm_ms

    def gemm_ms(win, gemm, sigma, c_ns, bw, sms, p=None, occ=None):
        geo = TM.GEOMETRY[gemm]
        n = int(win.live)
        rec[gemm] = {"n": n, "S": int(sms * occ) if occ else None, "slab": float(geo.slab),
                     "floor_us": TM.floor_ksteps(gemm) * c_ns * 1e-3, "fetches": _fetches(win, geo, n) if n else 0.0}
        return orig(win, gemm, sigma, c_ns, bw, sms, p, occ)
    TM.gemm_ms = gemm_ms
    try:
        p = V5D.price(model, T, counts, prm, ids=ids, fixes=fixes, terms=terms, declared=declared,
                      with_constants=with_constants)
    finally:
        TM.gemm_ms = orig
    v = p.get("v5d")
    if law is None or v is None:
        return p
    naive = p["path"] == RD.NAIVE
    if naive:
        nofollow = all(r["n"] == 0 or abs(r["fetches"] - r["n"]) < 0.5 for r in rec.values())
        if not nofollow:
            p["v6d"] = {"tile": v["tile"], "naive": "a non-fetching follower: v4d''s naive GEMM prices kept"}
            return p
    elif not v["aligned"]:
        return p
    tile = v["tile"]
    add, out = 0.0, {}
    for g in TM.GEMMS:
        r = rec[g]
        t = law.gemm_us(tile, r["n"], r["S"], r["slab"], r["floor_us"]) * 1e-3
        old = p["gemm"][g]["gemm_ms"]
        p["gemm"][g]["gemm_ms"] = t
        add += t - old
        out[g] = dict(r, law_ms=t, v5d_ms=old)
    p["body_ms"] += add
    if p["ms"] is not None:
        p["ms"] = p["body_ms"] + p["const_ms"]
    p["v6d"] = {"tile": tile, "Q_B": law.q(tile), "gemm": out,
                "naive": "no non-fetching follower: priced by the law as its aligned twin" if naive else None}
    return p


def describe() -> dict:
    """The registered form of v6d (the constants are LAW.as_dict())."""
    return {
        "GEMM law": ("on the aligned paths, and on a naive cell with no non-fetching follower: per GEMM of n live CTAs "
                     "on S slots, life = max(c0(m) + b L / Q, c0(m) + m b / X_sat, ceil(m / 132) floor), L read off "
                     "L_TABLE at the duty load m Q phi (fixed point), wave = kappa(m / S) life; n <= S: wave(n); n > S: "
                     "(n - k) b / X_sat + (kappa(1) - 1) life(S) + wave(k), every wave at c0(S); b = BLOCK_N K 2, "
                     "Q = n_if BLOCK_K BLOCK_N 2"),
        "c0": "piecewise linear in the live CTAs between G 8 and G 128, held outside; a multi-wave GEMM at c0(S)",
        "naive": ("a naive cell whose every CTA fetches its slab (v4d''s fix W reads) is priced by the law as its aligned "
                  "twin; a cell with any non-fetching follower keeps v4d''s naive GEMM prices"),
        "kept from v5d": "CALL, ALIGN and SLOPE (DS's readings); v5d's regimes and W1 scale emptied",
        "CAL constants": "T0_GR, s_naive, s_block refitted on CAL2's kept cells under v6d (v4d's fit, rho selection)",
    }
