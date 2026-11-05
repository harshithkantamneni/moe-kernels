#!/usr/bin/env python
"""MODEL v5d: model v4d' (`r3_timing_model_v4dp`, byte-identical, imported and called) with its partial-wave tail
replaced by the two regimes DS measured per decode tile, the w1 rate DS measured per larger tile, and the call kernels
priced by registered rules (registration 2026-11-04-decode-addendum5-gh200; DS's readings, addendum 4; CAL2's pages).
With every term off (`NONE`), `price` returns v4d''s own price object unchanged, so v5d reduces to v4d' exactly.

    T = v4d'(T)  with, per GEMM on a tile in `Terms.regime` (its CORES rho tail removed: the GEMM at bandwidth, its
                 partial wave at its compute floor)
                     gemm = max(G_bw x s, b / r_cta,t) + d b / r1_t
                 - B_other + CALL(H, I, k, T)
                 + ALIGN(path, E) + SLOPE x (iterations - 1) on the small-batch path

  PATHS    SUBWAVE, DRAIN and W1 act on the aligned paths (small-batch and block-scan), the rows DS read: one M-tile
           per expert, each live CTA fetching its own slab. The naive path (one M-tile per (token, expert) row, a repeated
           expert's slab served from L2 to its followers; no DS row) keeps v4d''s GEMM price, fixes W and R included.
  SUBWAVE  r_cta,t: the per-CTA streaming rate of a GEMM whose few long-slab CTAs cannot fill the DRAM pipe (DS's
           SUBWAVE reading per tile, MEASURED; None: no ceiling). b is v4d''s bytes per CTA, (reads + the output
           tile x n) / n, n the GEMM's live CTAs, as DS read it. It binds only where b / r_cta exceeds the GEMM at
           bandwidth, which on DS's pages is a single wave of 64 to 128 CTAs at b 0.7 to 3.7 MB.
  DRAIN    r1_t: the rate of the one slab per CTA launched past a wave boundary (DS's DRAIN reading per tile,
           MEASURED; None: no drain); d = 1 when n > S, S = SMs x the census occupancy (396 or 528), else 0.
  W1       s: 1 on w2; on w1 fix R's 0.918488 on 16x64x128w4s4 (frozen, inside v4d''s G_bw) and s_t = 3577.96 GB/s
           / c_t on a larger tile, c_t DS's W1RATE counter rate there (MEASURED). s scales the GEMM's bandwidth part
           only: DS read b / r_cta and b / r1 in counter time, so neither is scaled.
  CALL     vLLM's silu_and_mul (between w1 and w2) and moe_sum (after w2), in place of v4d's B_other (their bytes at
           DRAM speed). Per class a registered form on the model's shape, its coefficients fitted on DS's NATIVE
           call rows by ordinary least squares (d5common.fit_call_rule): silu_and_mul a + g B / 1 MB with
           B = T k 3 I x 2 bytes; moe_sum a + h k + g B / 1 MB with B = (T k H + T H) x 2 bytes. An unseen model
           takes the same formula at its own H, I and k (never fitted on its pages).
  ALIGN    vLLM's align kernel(s) before w1: a_DS(path, E), DS's median align-class ncu time at E (small-batch: one
           loop iteration; block-scan: BLOCK), piecewise linear in E between the measured E and held at the nearest
           measured E outside them (never extrapolated); 0 on the naive path. On the small-batch path + SLOPE x
           (iterations - 1), iterations = ceil(numel / max(E, 32)), SLOPE DS's one slope (0.526921 us).
Every term enters the body; the CAL fit (T0_GR, s_naive, s_block) is refitted on CAL2's kept cells with them in
(v4d's registered fit and rho selection; with the CORES tail removed on every tile the two rho forms price alike).
"""
from __future__ import annotations

import bisect
import contextlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r3_decode as RD  # noqa: E402
import r3_timing_model_v4dp as V4DP  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

V4D, TM, V4 = V4DP.V4D, V4DP.TM, V4DP.V4
ALIGN_PATHS = (RD.SMALL_BATCH, RD.BLOCK_SCAN)
#: the paths SUBWAVE, DRAIN and W1 act on (registered: the aligned paths DS read; addendum 5, D5-2). The every-path
#: alternative D5-2 compares adds RD.NAIVE (scripts/scoring/decode/d5_build/every_path.py); never registered
GEMM_TERM_PATHS = (RD.SMALL_BATCH, RD.BLOCK_SCAN)
#: CALL's two kernel classes, their registered regressors, and the decode stage's activation bytes (bf16, as B_other)
CALL_CLASSES = ("silu_and_mul", "moe_sum")
CALL_FORMS = {"silu_and_mul": ("1", "B_MB"), "moe_sum": ("1", "k", "B_MB")}
DTYPE_BYTES = 2


@dataclass(frozen=True)
class Terms:
    """v5d's measured terms. `regime`: {tile: (r_cta_GBps or None, r1_GBps or None)}, the tiles whose CORES tail is
    replaced; `w1_scale`: {tile: s_t} on the larger tiles; `call`: {class: {regressor: coefficient}} on CALL_FORMS;
    `align`: {path: ((E, us), ...)} sorted by E; `slope`: us per align-loop iteration past the first (small-batch).
    NONE (all empty) is v4d' exactly."""
    regime: dict = field(default_factory=dict)
    w1_scale: dict = field(default_factory=dict)
    call: dict = field(default_factory=dict)
    align: dict = field(default_factory=dict)
    slope: float = 0.0

    def any(self) -> bool:
        return (bool(self.regime) or bool(self.w1_scale) or bool(self.call) or any(self.align.get(p) for p in ALIGN_PATHS)
                or bool(self.slope))

    def as_dict(self) -> dict:
        return {"regime": {t: {"r_cta_GBps": v[0], "r1_GBps": v[1]} for t, v in sorted(self.regime.items())},
                "w1_scale": {k: v for k, v in sorted(self.w1_scale.items())},
                "call": {c: dict(sorted(v.items())) for c, v in sorted(self.call.items())},
                "align_us": {p: [list(x) for x in self.align.get(p) or ()] for p in ALIGN_PATHS},
                "slope_us_per_iteration": self.slope}

    @classmethod
    def from_dict(cls, d: dict | None) -> Terms:
        d = d or {}
        f = lambda v: None if v is None else float(v)  # noqa: E731
        reg = {str(t): (f(v.get("r_cta_GBps")), f(v.get("r1_GBps"))) for t, v in (d.get("regime") or {}).items()}
        w1 = {str(k): float(v) for k, v in (d.get("w1_scale") or {}).items()}
        call = {str(c): {str(q): float(x) for q, x in v.items()} for c, v in (d.get("call") or {}).items()}
        al = {p: tuple(sorted((int(e), float(u)) for e, u in (d.get("align_us") or {}).get(p) or ())) for p in ALIGN_PATHS}
        return cls(regime=reg, w1_scale=w1, call=call, align={p: v for p, v in al.items() if v},
                   slope=float(d.get("slope_us_per_iteration") or 0.0))

    def only(self, *names: str) -> Terms:
        """These terms alone (the ablation rungs): any of regime, w1_scale, call, align (with slope)."""
        return Terms(regime=self.regime if "regime" in names else {}, w1_scale=self.w1_scale if "w1_scale" in names else {},
                     call=self.call if "call" in names else {}, align=self.align if "align" in names else {},
                     slope=self.slope if "align" in names else 0.0)


NONE = Terms()


def align_us(terms: Terms, path: str, E: int) -> float:
    """a_DS(path, E): piecewise linear between the measured E, held at the nearest measured E outside them; 0 off-path."""
    pts = terms.align.get(path) or ()
    if path not in ALIGN_PATHS or not pts:
        return 0.0
    es = [p[0] for p in pts]
    if E <= es[0]:
        return pts[0][1]
    if E >= es[-1]:
        return pts[-1][1]
    i = bisect.bisect_right(es, E)
    (e0, u0), (e1, u1) = pts[i - 1], pts[i]
    return u0 + (u1 - u0) * (E - e0) / (e1 - e0)


def iterations(numel: int, E: int) -> int:
    """ceil(numel / max(E, 32)): the small-batch align kernel's per-thread loop count."""
    return math.ceil(int(numel) / max(int(E), 32))


def call_regressors(cfg, T: int, cls: str) -> dict:
    """The registered regressors of one call class at T tokens, from the model's H, I and k (bf16 bytes)."""
    numel = int(T) * cfg.top_k
    if cls == "silu_and_mul":
        B = numel * 3 * cfg.intermediate_size * DTYPE_BYTES
    elif cls == "moe_sum":
        B = (numel * cfg.hidden_size + int(T) * cfg.hidden_size) * DTYPE_BYTES
    else:
        raise KeyError(cls)
    return {"1": 1.0, "k": float(cfg.top_k), "B_MB": B / 1e6}


def call_us(terms: Terms, cfg, T: int) -> float | None:
    """CALL(model, T) in us: the sum over the classes of their registered form; None when `terms.call` is empty."""
    if not terms.call:
        return None
    tot = 0.0
    for cls in CALL_CLASSES:
        x = call_regressors(cfg, T, cls)
        tot += sum(c * x[q] for q, c in terms.call[cls].items())
    return tot


@contextlib.contextmanager
def regime_hook(on: bool, store: dict):
    """For one price call: when `on`, r3_timing_model.cores_lifetime is the compute floor alone (the partial wave at
    bandwidth, the CORES rho tail removed). Always records each GEMM's n live CTAs, S and b into `store`."""
    orig_life, orig_gemm = TM.cores_lifetime, TM.gemm_ms

    def cores_lifetime(win, k, gemm, sigma, c_ns, bw, sms):
        return math.ceil(k / sms) * TM.floor_ksteps(gemm) * c_ns * 1e-6

    def gemm_ms(win, gemm, sigma, c_ns, bw, sms, p=None, occ=None):
        n = int(win.live)
        store[gemm] = {"n": n, "S": int(sms * occ) if occ else None,
                       "b": (sigma * float(win.reads) + TM.OUT_TILE * n) / n if n else 0.0}
        return orig_gemm(win, gemm, sigma, c_ns, bw, sms, p, occ)
    if on:
        TM.cores_lifetime = cores_lifetime
    TM.gemm_ms = gemm_ms
    try:
        yield
    finally:
        TM.cores_lifetime, TM.gemm_ms = orig_life, orig_gemm


def price(model: str, T: int, counts, prm: V4D.DecodeParams, *, ids=None, fixes: V4DP.Fixes = V4DP.ALL,
          terms: Terms = NONE, declared: int | None = None, with_constants: bool = True) -> dict:
    """v5d's time of one decode cell (ms), its parts and its bytes. With `terms` NONE this is
    r3_timing_model_v4dp.price's own return value."""
    if not terms.any():
        return V4DP.price(model, T, counts, prm, ids=ids, fixes=fixes, declared=declared, with_constants=with_constants)
    cfg = MODEL_CONFIGS[model]
    tile = RD.tile_key(RD.default_config(int(T), cfg.num_experts))
    aligned = RD.geometry(cfg, int(T), counts, declared).path in GEMM_TERM_PATHS
    reg = terms.regime.get(tile) if aligned else None
    rec: dict = {}
    with regime_hook(reg is not None, rec):
        p = V4DP.price(model, T, counts, prm, ids=ids, fixes=fixes, declared=declared, with_constants=with_constants)
    add, per = 0.0, {}
    s1 = terms.w1_scale.get(tile) if aligned and tile != V4DP.W1_TILE else None
    for g in TM.GEMMS:
        part, r = p["gemm"][g], rec.get(g) or {}
        base = part["gemm_ms"]
        bw = base * (s1 if (g == "w1" and s1 is not None) else 1.0)
        ceil = drain = 0.0
        if reg is not None and r.get("n"):
            r_cta, r1 = reg
            if r_cta is not None:
                ceil = r["b"] / (r_cta * 1e6)
            if r1 is not None and r["S"] and r["n"] > r["S"]:
                drain = r["b"] / (r1 * 1e6)
        t = max(bw, ceil) + drain
        per[g] = {"n": r.get("n"), "S": r.get("S"), "b_bytes": r.get("b"), "bw_ms": bw, "ceiling_ms": ceil,
                  "ceiling_binds": ceil > bw, "drain_ms": drain, "gemm_ms_v4dp": base}
        part["gemm_ms_v4dp"] = base
        part["gemm_ms"] = t
        add += t - base
    cu = call_us(terms, cfg, T)
    call_d = 0.0 if cu is None else cu * 1e-3 - p["other_ms"]
    add += call_d
    E = cfg.num_experts if declared is None else int(declared)
    a = align_us(terms, p["path"], E)
    it = iterations(p["numel"], E) if p["path"] == RD.SMALL_BATCH else 1
    s = terms.slope * (it - 1) if p["path"] == RD.SMALL_BATCH else 0.0
    add += (a + s) * 1e-3
    p["body_ms"] = p["body_ms"] + add
    if p["ms"] is not None:
        p["ms"] = p["body_ms"] + p["const_ms"]
    p["v5d"] = {"tile": tile, "aligned": aligned, "regime": None if reg is None else {"r_cta_GBps": reg[0], "r1_GBps": reg[1]},
                "w1_scale": s1, "gemm": per, "call_ms": None if cu is None else cu * 1e-3, "call_minus_other_ms": call_d,
                "align_ms": a * 1e-3, "slope_ms": s * 1e-3, "declared_E": E, "iterations": it}
    return p


def describe(terms: Terms) -> dict:
    """The registered form of v5d and its constants (DS's readings, filled mechanically by d5common.terms_from)."""
    d = terms.as_dict()
    return {"SUBWAVE, DRAIN": {"rule": "on the aligned paths (small-batch, block-scan; the naive path keeps v4d''s GEMM price), on a tile in regime: the CORES rho tail removed (the GEMM at bandwidth, its partial "
                                       "wave at its compute floor); gemm = max(G_bw x s, b / r_cta) + d b / r1, d = 1 when "
                                       "n > S; b v4d''s bytes per CTA", "fitted": 0, "regime": d["regime"]},
            "W1": {"rule": "on the aligned paths, w1's bandwidth part x s_t = 3577.96 / c_t on a larger tile (c_t DS's W1RATE); 16x64x128w4s4 "
                           "keeps fix R", "fitted": 0, "scale": d["w1_scale"]},
            "CALL": {"rule": "B_other replaced by silu_and_mul a + g B / 1 MB (B = T k 3 I x 2) plus moe_sum a + h k + "
                             "g B / 1 MB (B = (T k H + T H) x 2); an unseen model at its own H, I, k",
                     "forms": {c: list(v) for c, v in CALL_FORMS.items()}, "coefficients": d["call"]},
            "ALIGN": {"rule": "+ a_DS(path, E), piecewise linear in E, held at the nearest measured E outside them, 0 on the "
                              "naive path; + slope x (iterations - 1) on the small-batch path",
                      "align_us": d["align_us"], "slope_us_per_iteration": terms.slope}}
