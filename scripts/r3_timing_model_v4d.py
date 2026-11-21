#!/usr/bin/env python
"""MODEL v4d: the decode call's time and bytes (design-decode section 3; registration
2026-10-14-decodeC-cal-gh200). A separate module over model v4 (`r3_timing_model_v4`) and the pinned
v3 machinery (`r3_timing_model`), both byte-identical; the tile is swapped in memory by v4's
`at_tile` and restored. Nothing here writes tiling or scheduling logic: vLLM's tile is read from
`default_config(T, E)`, its assignment path and grid from r3_decode's cited v0.27.1 rules.

    T = T0_GR + s_path + sum_g [ gemm_g + dead_g ] + B_other(T) / bw

  gemm_g    v3's per-GEMM law (`r3_timing_model.gemm_ms`): sum over live CTAs of smax_14(floor,
            DRAM), the partial-wave tail under CORES (DRAM at min(bw, rho x CTAs)), at the decode
            tile's k-steps, c_time = c_card x c_scale(tile) and PHI(tile) from v4 (the tile stage's
            Params: census occupancy, c_128, F_wg, accounting, F form). The live-CTA schedule is
            r3_timing_model.schedule's at G 1 (every live CTA fetches its slab: one M-tile per
            expert unless c_e > BM; a hot expert's second tile re-reads, sigma 1). Naive path: one
            M-tile per (token, expert) row, LAUNCHED token-major (topk_ids.view(-1)) as vLLM does; the
            rows are listed expert-ascending here, an approximation that leaves the read total and
            the throughput sum unchanged (see _naive_reads); a repeated expert's slab is fetched once
            (registered; the rival NAIVE-NODEDUP prices every row's fetch).
  bytes     A W_e plus activations (the schedule's own read bytes), not a function of T.
  dead_g    v2's dead term: d (0.995 ns) per dead CTA less kappa (0.325) occ S' c, dead CTAs from
            r3_decode.geometry (the capped buffer and the EM cut). Zero on the naive path.
  B_other   silu_and_mul and moe_sum at DRAM speed over numel = T k rows.
  rho       RHO_FORMS: "fixed", v3's 11.48 GB/s per resident CTA; "scaled", 11.48 x the tile's
            weight bytes in flight per CTA over R3's tile's (16x64x128 s4 holds 2x). One discrete
            CAL selection (the registration's rule).
  T0_GR, s_naive_GR, s_block_GR   the three CAL constants, fitted by `fit_constants` (weighted least
            squares on the relative error, closed form). s_small_GR is folded into T0_GR: every cell
            has exactly one path, so T0_GR + s_path has three identifiable values, not four.

Padded and masked rows cost 0 (FILL0, tested by G3-D fill; a FILL-r rival was STRUCK before any page,
build-decode review 5: `fill_r_us` stays a test knob, never a registered prediction). Empty experts enter only
through A, L and the dead CTAs. The host is not a term: GD1 removes host-bound cells.
"""
from __future__ import annotations

import contextlib
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r3_decode as RD  # noqa: E402
import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

#: the card constants v4d carries (SEEN): rental 6's CAL 8x7B v2 fit (reg6.pricer().f2.x, recomputed
#: by scripts/scoring/decode/register.py --check), c per CTA k-step and the DRAM rate
CARD = {"c_ns": 202.54764415793517, "bw_gbps": 3597.800853079132, "T0_eager_ms": 0.04323024982488314,
        "s_small_eager_ms": 0.012986935170298423, "s_block_eager_ms": -0.0070686230486082914,
        "k_w": 0.5, "sms": 132, "source": "rental 6's CAL 8x7B v2 fit (scripts/scoring/rental6/reg6.pricer().f2.x)"}
RHO_FORMS = ("fixed", "scaled")
RHO_GBPS = TM.CORES_RHO_GBPS
PATHS = (RD.SMALL_BATCH, RD.NAIVE, RD.BLOCK_SCAN)
R3_TILE_INFLIGHT = 3 * 64 * 64 * 2


def tile_of(conf: dict) -> V4.Tile:
    return V4.Tile(int(conf["BLOCK_SIZE_M"]), int(conf["BLOCK_SIZE_N"]), int(conf["BLOCK_SIZE_K"]),
                   int(conf["num_warps"]), int(conf["num_stages"]))


def inflight_bytes(t: V4.Tile) -> int:
    """Weight bytes in flight per CTA: the B buffers the compiled cache allocates (wgmma num_stages,
    mma.sync num_stages - 1, v4's Tile.shared_bytes rule), BK x BN bf16 each."""
    bufs = t.stages if t.path == "wgmma" else t.stages - 1
    return bufs * t.bk * t.bn * 2


def rho_of(t: V4.Tile, form: str) -> float:
    if form == "fixed":
        return RHO_GBPS
    if form == "scaled":
        return RHO_GBPS * inflight_bytes(t) / R3_TILE_INFLIGHT
    raise ValueError(f"rho form {form!r}: one of {RHO_FORMS}")


@contextlib.contextmanager
def _rho(value: float):
    saved = TM.CORES_RHO_GBPS
    TM.CORES_RHO_GBPS = float(value)
    try:
        yield
    finally:
        TM.CORES_RHO_GBPS = saved


@dataclass
class DecodeParams:
    """v4d's constants. The CAL three are None until the CAL rental fits them; `tile` is the tile
    stage's v4 Params (its finalised selections), `occupancy` the census's per decode tile."""
    T0_GR_ms: float | None = None
    #: None: the path had no CAL row, so its cells are NOT PRICED (never an offset of 0)
    s_naive_GR_ms: float | None = 0.0
    s_block_GR_ms: float | None = 0.0
    rho_form: str = "fixed"
    tile: V4.Params = field(default_factory=V4.Params)
    c_ns: float = CARD["c_ns"]
    bw_gbps: float = CARD["bw_gbps"]
    k_w: float = CARD["k_w"]
    sms: int = CARD["sms"]
    naive_dedup: bool = True
    fill_r_us: float = 0.0          # a per-padded-row cost knob for tests (0: FILL0); no registered rival

    def s_path(self, path: str) -> float | None:
        return {RD.SMALL_BATCH: 0.0, RD.NAIVE: self.s_naive_GR_ms, RD.BLOCK_SCAN: self.s_block_GR_ms}[path]


def _naive_reads(cfg, counts, gemm: str) -> np.ndarray:
    """The naive path's per-CTA reads, from the counts alone. vLLM LAUNCHES the naive path in
    token-major top-k order (expert_ids = topk_ids.view(-1), one M-tile per (token, expert) row, the
    N-tiles of a row inner); this lists the rows expert-ascending instead, a REGISTERED APPROXIMATION:
    the read total and the throughput sum over CTAs are order-free, and only the box-window means and
    which CTAs form the partial-wave tail depend on the order. The dedup claim is the same in either
    order: one slab per (expert, N-tile), fetched by that expert's first row in launch order, the later
    rows of the same expert served from L2 (the NAIVE-NODEDUP rival fetches every row's)."""
    geo = TM.GEOMETRY[gemm]
    out = []
    for e, v in enumerate(counts):
        for j in range(int(v)):
            for _n in range(geo.npn):
                out.append((geo.slab if j == 0 else 0.0) + geo.arow / geo.npn / TM.BLOCK_M)
    return np.asarray(out, dtype=float)


def _window_of(r: np.ndarray, w: int) -> TM.Window:
    m = TM.box_mean(r, w)
    s = np.sort(m)
    return TM.Window(live=int(r.size), leads=int((r > 0).sum()), reads=float(r.sum()), sorted_mean=s,
                     cumsum=np.concatenate([[0.0], np.cumsum(s)]), mean=m, rcum=np.concatenate([[0.0], np.cumsum(r)]))


def price(model: str, T: int, counts, prm: DecodeParams, *, declared: int | None = None,
          with_constants: bool = True) -> dict:
    """v4d's time of one decode cell (ms), its parts and its bytes. `declared` E x C for DECL."""
    cfg = MODEL_CONFIGS[model]
    E = cfg.num_experts
    conf = RD.default_config(T, E)
    t = tile_of(conf)
    geo = RD.geometry(cfg, T, counts, declared, conf)
    ct = tuple(int(v) for v in counts)
    phis = {g: V4.phi(t, g, prm.tile) for g in TM.GEMMS}
    occ = {g: prm.tile.occ(t, g) for g in TM.GEMMS}
    parts, reads, live = {}, {}, {}
    with V4.at_tile(model, t, phis), _rho(rho_of(t, prm.rho_form)):
        for i, g in enumerate(TM.GEMMS):
            cg = prm.c_ns * V4.c_scale(t, g, prm.tile)
            w = TM.window_width(prm.k_w, prm.sms, occ[g])
            if geo.path == RD.NAIVE:
                r = _naive_reads(cfg, ct, g)
                if not prm.naive_dedup:
                    geo_g = TM.GEOMETRY[g]
                    r = np.full(r.size, geo_g.slab + geo_g.arow / geo_g.npn / TM.BLOCK_M)
                win = _window_of(r, w)
            else:
                win = TM.window("native", E, 1, 1, g, w, ct)
            gm = TM.gemm_ms(win, g, 1.0, cg, prm.bw_gbps, prm.sms, occ=occ[g])
            hidden = TM.V2_KAPPA * occ[g] * TM.floor_ksteps(g) * cg
            dm = max(0.0, geo.dead_ctas[i] * TM.V2_DEAD_NS - hidden) * 1e-6
            parts[g] = {"gemm_ms": gm, "dead_ms": dm, "c_ns": cg, "occ": occ[g], "phi": phis[g],
                        "ksteps": TM.GEOMETRY[g].ksteps, "live_ctas": win.live,
                        "waves": win.live / (prm.sms * occ[g])}
            reads[g] = win.reads
            live[g] = win.live
    numel = T * cfg.top_k
    b_other = (numel * 2 * cfg.intermediate_size + numel * cfg.intermediate_size + numel * cfg.hidden_size
               + T * cfg.hidden_size) * 2
    other_ms = b_other / (prm.bw_gbps * 1e6)
    fill_ms = prm.fill_r_us * 1e-3 * geo.P
    body = sum(p["gemm_ms"] + p["dead_ms"] for p in parts.values()) + other_ms + fill_ms
    const, not_priced = None, None
    if with_constants and prm.T0_GR_ms is not None:
        s = prm.s_path(geo.path)
        if s is None:
            not_priced = f"NOT PRICED: no CAL row fitted the {geo.path} offset"
        else:
            const = prm.T0_GR_ms + s
    return {"ms": None if const is None else body + const, "body_ms": body, "const_ms": const, "not_priced": not_priced,
            "path": geo.path, "tile": geo.tile, "A": geo.A, "L": geo.L, "P": geo.P, "numel": numel,
            "dead_ctas": list(geo.dead_ctas), "live_ctas": [live[g] for g in TM.GEMMS],
            "reads": reads, "bytes": sum(reads.values()) + b_other, "other_ms": other_ms, "gemm": parts,
            "sub_wave": any(p["waves"] < 1.0 for p in parts.values()), "rho_form": prm.rho_form,
            "weight_bytes": geo.A * RD.expert_bytes(cfg)}


def fit_constants(rows: list[dict]) -> dict:
    """The three CAL constants (T0_GR, s_naive_GR, s_block_GR) by weighted least squares on the
    relative error: minimise sum ((body + T0 + s_path - meas) / meas)^2, linear in the constants,
    closed form. rows: {meas_ms, body_ms, path}. A path with no row has NO offset (None, reported NOT
    FITTED): its cells are NOT PRICED, never priced at an offset of 0 (build-decode review 3)."""
    if not rows:
        return {"verdict": "NOT FITTED: no row"}
    paths = [p for p in (RD.NAIVE, RD.BLOCK_SCAN) if any(r["path"] == p for r in rows)]
    X, y = [], []
    for r in rows:
        w = 1.0 / float(r["meas_ms"])
        X.append([w] + [w * (r["path"] == p) for p in paths])
        y.append(w * (float(r["meas_ms"]) - float(r["body_ms"])))
    X, y = np.asarray(X), np.asarray(y)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return {"verdict": "NOT FITTED: the design matrix is rank-deficient"}
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = {"verdict": "FITTED", "cells": len(rows), "T0_GR_ms": float(beta[0]),
           "s_naive_GR_ms": None, "s_block_GR_ms": None, "fitted_paths": paths}
    for i, p in enumerate(paths):
        out["s_naive_GR_ms" if p == RD.NAIVE else "s_block_GR_ms"] = float(beta[1 + i])
    for p in (RD.NAIVE, RD.BLOCK_SCAN):
        if p not in paths:
            out.setdefault("not_fitted", []).append(p)
    return out


def rel_err(pred_ms: float, meas_ms: float, floor_us: float = 1.0) -> float:
    """E1-DEC's per-cell error: (pred - meas) / meas, with an absolute error at or under `floor_us`
    counted as 0 (the registration's reading of 'floored at 1 us')."""
    d = pred_ms - meas_ms
    if abs(d) * 1e3 <= floor_us:
        return 0.0
    return d / meas_ms


def roofline_ms(model: str, T: int, counts, T0_GR_ms: float, bw_gbps: float = CARD["bw_gbps"]) -> float:
    """The weight-stream roofline rival: T0_GR + A W_e / bw."""
    cfg = MODEL_CONFIGS[model]
    A = sum(1 for v in counts if v)
    return T0_GR_ms + A * RD.expert_bytes(cfg) / (bw_gbps * 1e6)


def eager_params(prm: DecodeParams) -> DecodeParams:
    """The 'v4 at eager constants' rival: v3's eager T0, s_small and s_block (naive at s_small)."""
    d = asdict(prm)
    d["tile"] = prm.tile
    d.update(T0_GR_ms=CARD["T0_eager_ms"] + CARD["s_small_eager_ms"], s_naive_GR_ms=0.0,
             s_block_GR_ms=CARD["s_block_eager_ms"] - CARD["s_small_eager_ms"])
    return DecodeParams(**d)


def params_from(fit: dict, *, rho_form: str, tile: V4.Params | None = None) -> DecodeParams:
    return DecodeParams(T0_GR_ms=fit.get("T0_GR_ms"), s_naive_GR_ms=fit.get("s_naive_GR_ms"),
                        s_block_GR_ms=fit.get("s_block_GR_ms"), rho_form=rho_form, tile=tile or V4.Params())


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--counts", required=True, help="comma-separated E counts")
    ap.add_argument("--rho", default="fixed", choices=RHO_FORMS)
    a = ap.parse_args(argv)
    prm = DecodeParams(rho_form=a.rho)
    p = price(a.model, a.T, [int(x) for x in a.counts.split(",")], prm)
    print(json.dumps({k: v for k, v in p.items() if k != "gemm"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
