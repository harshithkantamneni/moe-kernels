"""Shared loaders for the A2 diagnosis (read-only; nothing under the repo or the tree is written).

Every scorer path is the registered one: a2common.scoring(Rules.tileA2(1410)) around finalise.selections
and wave_split_bytes_v4.price, exactly as score_e1cal_a2.py runs them. `price_parts` re-walks
r3_timing_model_v4.call_ms term by term (same calls, same order) so the parts sum to v4's ms; `check`
asserts that on every cell.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path.home() / "code" / "moe-kernels"
TREE = Path.home() / "moe-kernels-exfil" / "runs" / "gh200-tile2-20261020T2242Z" / "tree"
SCORE = Path.home() / "code" / "moe-kernels-work" / "2026-10-20" / "scoreA2"
OUT = Path(__file__).resolve().parent
LOCK = 1410.0
for p in (REPO / "scripts" / "scoring" / "tileA2", REPO / "scripts" / "scoring" / "tileA", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import a2common as A2  # noqa: E402

TC = A2.TC
import finalise as FN  # noqa: E402
import regtile as RG  # noqa: E402
import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v3 as TV3  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402
import wave_split_bytes_v4 as WV4  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

RULES = A2.Rules.tileA2(LOCK)


def gemm_parts(win, g, sigma, c_ns, bw, sms, occ):
    """gemm_ms plus its pure-floor and pure-byte bounds (same window)."""
    full = TM.gemm_ms(win, g, sigma, c_ns, bw, sms, occ=occ)
    only_floor = TM.gemm_ms(win, g, sigma * 1e-9, c_ns, 1e12, sms, occ=occ)
    only_bytes = TM.gemm_ms(win, g, sigma, c_ns * 1e-9, bw, sms, occ=occ)
    keep = TM.TAIL
    try:
        TM.TAIL = False
        notail = TM.gemm_ms(win, g, sigma, c_ns, bw, sms, occ=occ)
    finally:
        TM.TAIL = keep
    return full, only_floor, only_bytes, notail


def price_parts(model, arm, c, doc, prm, *, key, reads=None, x_over=None):
    """v4's time of one histogram cell with every term (WV4.price's own steps)."""
    import r6model as M6  # noqa: F401
    pr = RG.pricer()
    t = RG.PAGE_TILE[int(doc["block_m"])]
    n, counts, G = int(c["n"]), c["counts"], int(doc["group_m"])
    seed = RG.seed_of(doc, c)
    base = WV4.price(pr, model, arm, n, counts, RG.COPIES, seed, t, G, prm, key=key, reads=reads)
    occ = {g: prm.occ(t, g) for g in TM.GEMMS}
    phis = {g: V4.phi(t, g, prm) for g in TM.GEMMS}
    scale = {g: V4.c_scale(t, g, prm) for g in TM.GEMMS}
    x = [float(v) for v in pr.f2.x]
    if x_over:
        for i, v in x_over.items():
            x[i] = v
    T0, c_ns, bw, s_small, s_block = x
    ct = None if counts is None else tuple(int(v) for v in counts)
    out = {"ms_v4": base["ms"], "reads": base["reads"], "q": base["q"], "occ": occ, "phi": phis, "scale": scale}
    with V4.at_tile(model, t, phis):
        ctx = pr.ctx if occ == dict(pr.ctx.occupancy) else dataclasses.replace(pr.ctx, occupancy=dict(occ))
        cell = V4.make_cell(pr.eng, model, arm, n, ct, RG.COPIES, G, t, base["reads"], occ, pr.f2.k_w, ctx.sms)
        treads = cell.n if ct is None else TM.numel_of(cell.n, ct) / TM.IDS_PER_TREAD
        out["T0"] = T0
        out["other"] = treads * TM.B_OTHER / (bw * 1e6)
        out["B_other_bytes"] = treads * TM.B_OTHER
        tot = T0 + out["other"]
        with TM.dead_model("v2"):
            for g in TM.GEMMS:
                cg = c_ns * scale[g]
                w = TM.window_width(pr.f2.k_w, ctx.sms, occ[g])
                win = TM.window(arm, cell.declared, G, n, g, w, ct)
                full, fl, by, nt = gemm_parts(win, g, cell.sigma[g], cg, bw, ctx.sms, occ[g])
                dead = TM.dead_ms(cell.declared, n, g, cg, pr.f2.k_w, occ[g], ct)
                const = TM.gemm_const_ms(g, cg)
                out[g] = {"gemm": full, "floor_only": fl, "bytes_only": by, "notail": nt, "dead": dead, "const": const,
                          "live": win.live, "dead_ctas": TM.dead_ctas(cell.declared, n, g, ct),
                          "ksteps": TM.GEOMETRY[g].ksteps, "S_prime": TM.floor_ksteps(g), "c_ns": cg,
                          "waves": win.live / (ctx.sms * occ[g]), "reads": base["reads"][g],
                          "W": TM.BYTE_MODEL[f"W_{g}"], "A_per_tread": TM.BYTE_MODEL[f"operand_per_tile_{g}"],
                          "out_tile": TM.OUT_TILE}
                tot += full + dead + const
        out["s"] = (s_small if cell.path == TM.SMALL_BATCH else s_block) if arm == "native" else 0.0
        out["path"] = cell.path
        tot += out["s"]
        out["b"] = TV3.v3_b_ms(arm, base["native_path"], None if base["A_us"] is None else base["A_us"] * 1e-3,
                               key=cell.key) if arm != "native" else 0.0
        tot += out["b"]
    out["ms_sum"] = tot
    return out


def counter_cells(model, label, G, lock=LOCK):
    hits = sorted(TREE.rglob(f"*-{TC.CARD}-{model}-{label}-r3-counters/lock{int(lock)}/r3c-g{G}.json"))
    if len(hits) != 1:
        return None, None
    p = json.loads(hits[0].read_text())
    return p, hits[0]


def r6(x):
    return None if x is None else round(float(x), 6)
