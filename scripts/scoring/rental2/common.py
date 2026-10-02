"""Rental 2's shared law and band arithmetic (docs/registered/2026-10-01-rental2-*).

One module for the registration generator (register.py) and the scorers, so a
band is computed by one rule on both sides. Pure functions; nothing here reads
a page. Every constant a band uses is passed in from the registered JSON.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

REG_DIR = REPO / "docs" / "registered"
DATE = "2026-10-01"
NAMES = {"knobs": f"{DATE}-rental2-knobs-gh200", "launch2": f"{DATE}-rental2-launch2-gh200",
         "const": f"{DATE}-rental2-const-gh200", "w1floor": f"{DATE}-rental2-w1floor-gh200"}
CARD = "nvidia_gh200_480gb"


def registration(repo: Path, part: str) -> dict:
    """The registered JSON of one part, read from `repo`'s docs/registered."""
    return json.loads((Path(repo) / "docs" / "registered" / f"{NAMES[part]}.json").read_text())


# --------------------------------------------------------------------------
# part 1: the lag law L2r, its rivals and their bands
# --------------------------------------------------------------------------

def ratio(dd: float, lam: float) -> float:
    """exp(-dd / lam), 0 when lam <= 0 (an infinitely steep law)."""
    if lam <= 0:
        return 0.0 if dd > 0 else 1.0
    return math.exp(-dd / lam)


def s_cap(x: float, knots: list) -> float:
    """Piecewise-linear s_cap through the registered knots [[x, s], ...]
    (ascending x), flat outside them."""
    if x <= knots[0][0]:
        return float(knots[0][1])
    if x >= knots[-1][0]:
        return float(knots[-1][1])
    for (x0, s0), (x1, s1) in zip(knots, knots[1:], strict=False):
        if x0 <= x <= x1:
            return float(s0 + (s1 - s0) * (x - x0) / (x1 - x0))
    raise AssertionError("unreachable")


def law_value(s_base: float, cap: float, dd: float, lam: float, past_edge: bool) -> float:
    """The L2r knob prediction as a ratio to the same-board base: past the edge
    s_base r, below it s_cap + (s_base - s_cap) r."""
    r = ratio(dd, lam)
    if past_edge:
        return s_base * r
    return cap + (s_base - cap) * r


def bands(s_base: float, cap: float, dd: float, *, lam: float, lam_sigma: float,
          lam_steep: float, past_edge: bool, s_noise: float, nl_half: float) -> dict:
    """{"L2r": [lo, hi], "NL": [lo, hi], "ST": [lo, hi]} for one cell and n."""
    lo_l = max(lam - lam_sigma, 0.0)
    hi_l = lam + lam_sigma
    a = law_value(s_base, cap, dd, lo_l, past_edge)
    b = law_value(s_base, cap, dd, hi_l, past_edge)
    st_top = law_value(s_base, cap, dd, lam_steep, past_edge)
    return {"L2r": [min(a, b) - s_noise, max(a, b) + s_noise],
            "NL": [s_base - nl_half, s_base + nl_half],
            "ST": [0.0, st_top + s_noise]}


def inside(v: float, band) -> bool:
    return band[0] <= v <= band[1]


def select(meas: dict, band_by_n: dict, hyps, ns, *, need: int) -> dict:
    """The 4-of-6 rule. For each hypothesis: the n inside its band and inside no
    other's (exclusive), the n outside. SELECTED: exclusive >= need; FALSIFIED:
    outside >= need. The verdict names the one SELECTED hypothesis, else
    INCONCLUSIVE."""
    per = {}
    for h in hyps:
        excl = [n for n in ns if inside(meas[n], band_by_n[n][h])
                and not any(inside(meas[n], band_by_n[n][o]) for o in hyps if o != h)]
        out = [n for n in ns if not inside(meas[n], band_by_n[n][h])]
        per[h] = {"exclusive": excl, "outside": out,
                  "status": ("SELECTED" if len(excl) >= need else
                             "FALSIFIED" if len(out) >= need else "neither")}
    sel = [h for h in hyps if per[h]["status"] == "SELECTED"]
    return {"per_hypothesis": per, "verdict": sel[0] if len(sel) == 1 else "INCONCLUSIVE"}


def lag_union(b: dict) -> list:
    """tp4 w1 s8's LAG hypothesis: the envelope of L2r and ST (the two cannot be
    told apart there; the review, section 1)."""
    return [min(b["L2r"][0], b["ST"][0]), max(b["L2r"][1], b["ST"][1])]


# --------------------------------------------------------------------------
# part 1, T5: the BLOCK_K separator
# --------------------------------------------------------------------------

def f_of_q(q: float, n: int) -> float:
    """PRIVATE w2's A-tile excess per tread, f = (q - n) 128 / (63 n): P_w2 = 64
    N-tiles, ATILE / SLAB = BLOCK_M / BLOCK_N = 1/2 on every Mixtral shard, at
    any BLOCK_K."""
    return (q - n) * 128.0 / (63.0 * n)


def sigma_rho(rho: float, f_num: float, f_den: float, sigma_q: float, n: int = 9) -> float:
    """sigma of rho = f_num / f_den, each f carrying sigma_q x 128 / (63 n) in
    quadrature (rental 1's T2 noise model)."""
    sf = sigma_q * 128.0 / (63.0 * n)
    return rho * math.sqrt((sf / f_num) ** 2 + (sf / f_den) ** 2)


def t5_candidates(reg_t5: dict):
    """{name: (wave_split_bytes.Params, content_a)} from the registered params."""
    import wave_split_bytes as W
    return {k: (W.Params.from_json(v["params"]), bool(v["content_a"]))
            for k, v in reg_t5["candidates"].items()}


def price_f(geom, params, content_a: bool, n: int = 9, G: int = 64) -> float:
    """One candidate's f at PRIVATE w2 (G, n) on a page's own geometry (its
    block_k and its W_c, from `wave_split_bytes.page_geometry`)."""
    import wave_split_bytes as W
    q = W.shown(W.Model(geom, content_a=content_a).evaluate(params, [("private", G, n, "w2")])[0])
    return f_of_q(q, n)


# --------------------------------------------------------------------------
# part 3 and 4 helpers
# --------------------------------------------------------------------------

def u_cycles(ksteps: int, gemm: str, c: float, F: dict) -> float:
    """The per-CTA floor u = S c + F_g (cycles)."""
    return ksteps * c + F[gemm]


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])
