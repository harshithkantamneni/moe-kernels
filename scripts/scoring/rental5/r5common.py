"""Rental 5's shared rules (docs/registered/2026-10-07-rental5-*).

One module for the registration generator (register.py) and every rental-5 scorer, so a
rule is computed one way on both sides. Pure functions over numbers, pages and the
registered JSON; nothing here fits a model at score time. The ALL / CLEAN views and the R3
page readers are rental 3's (r3common), the nearest-rival rule and the stamp readers rental
4's (r4common), imported and never copied.

THE SECONDARY METRICS (design-r5 part 5, corrected by design-r5-review section g):
  calibration  OLS of ln(meas) on ln(pred) over a test's scored cells (logs, so the largest
               cell does not set the slope). The CI is a WILD CLUSTER bootstrap on the
               cluster key the registration names (shape, for skew: pages A and B of one
               shape share its model error), Webb's six-point weights, 2000 resamples, seed
               20261007, 95% percentiles. CONSISTENT when 1 is in CI(slope) and 0 in
               CI(intercept); with fewer than 6 clusters the line is DESCRIPTIVE ONLY (point
               estimates printed, "NOT RESOLVED (C clusters)"), which applies to every
               rental-5 test (3 shapes at most).
  gamma        Goodman-Kruskal gamma, stratified: within each stratum every pair of cells
               whose MEASURED times differ by at least 2 sqrt2 sigma_page in ln (0.51%) and
               whose PREDICTIONS are not tied counts as concordant or discordant; pairs tied in
               prediction count in neither (that is gamma, not tau-b: the review's rename).
               Pooled (C - D) / (C + D) over strata, CI by resampling pages (clusters), seed
               20261007. CONSISTENT when the lower 95% bound is at least 0.5; NOT RESOLVED
               under 10 resolved pairs.
  regret       top-1 regret T(model argmin) / T(measured argmin) - 1: NOT APPLICABLE (one tile).
THE G3 RULE (design-r5-review owner item 1, owner decision 1 of 2026-10-07): TOST on the
  pooled difference with a cluster-t 90% interval (df = clusters - 1), margin 0.36%;
  Cochran's Q on the registered sigma gates (p < 0.01 sends the TOST to each n-stratum).
"""
from __future__ import annotations

import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C3 = _load("rental3_r3common", HERE.parent / "rental3" / "r3common.py")
C4 = _load("rental4_r4common", HERE.parent / "rental4" / "r4common.py")
CARD = C3.CARD
DATE = "2026-10-07"
#: tp2 (9 against 32 copies) was CUT by the owner on 2026-10-07; kappa is read from the stamps only
PARTS = ("v2", "skew", "c15", "stamps2", "secondk", "bk128")
NAMES = {p: f"{DATE}-rental5-{p}-gh200" for p in PARTS}
SMS = 132
LOCK_MHZ = 1710.0
#: sigma_page, rental 3 rep8 (SEEN): every rule is set at it and also printed at the alternative
SIGMA_PAGE = 0.0018
#: rental 4's NATIVE c9 / c15 same-call replicate (SEEN, FINDINGS rental 4 "Diagnosis")
SIGMA_PAGE_ALT = 0.0010
SEED = 20261007
BOOT = 2000
#: Webb's six-point weights for the wild cluster bootstrap
WEBB = (-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5))
MIN_CLUSTERS = 6
MIN_GAMMA_PAIRS = 10
GAMMA_FLOOR = 0.5
G3_MARGIN = 2 * SIGMA_PAGE          # 0.36%
G3_ALPHA = 0.10                      # a 90% interval: TOST at 5% each side
Q_ALPHA = 0.01

r6 = C4.r6
nearest = C4.nearest
median = C4.median
write_score = C4.write_score


def registration(repo: Path, part: str) -> dict:
    return json.loads((Path(repo) / "docs" / "registered" / f"{NAMES[part]}.json").read_text())


# --------------------------------------------------------------------------
# distributions with no scipy: Student t and chi-square tails
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def t_cdf(t: float, df: int) -> float:
    x = df / (df + t * t)
    tail = 0.5 * betai(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_ppf(q: float, df: int) -> float:
    """The q-quantile of Student t with df degrees of freedom, by bisection."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gammainc_upper(s: float, x: float) -> float:
    """Q(s, x), the regularized upper incomplete gamma."""
    if x <= 0:
        return 1.0
    if x < s + 1:
        term = total = 1.0 / s
        a = s
        for _ in range(1000):
            a += 1
            term *= x / a
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + s * math.log(x) - math.lgamma(s)))
    b, c = x + 1 - s, 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c if abs(c) > 1e-300 else 1e300
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return math.exp(-x + s * math.log(x) - math.lgamma(s)) * h


def chi2_sf(q: float, df: int) -> float:
    return gammainc_upper(df / 2.0, q / 2.0)


# --------------------------------------------------------------------------
# G3: TOST with a cluster-t interval, Cochran's Q
# --------------------------------------------------------------------------

def tost_cluster_t(d_by_cluster: dict, margin: float = G3_MARGIN, alpha: float = G3_ALPHA) -> dict:
    """The (1 - alpha) cluster-t interval of mean(d): the mean of the cluster means, se =
    sd(cluster means) / sqrt(P), t at df = P - 1. EQUIVALENT when the interval lies inside
    +-margin. Fewer than 2 clusters: NOT SCORED."""
    means = {k: st.fmean(v) for k, v in d_by_cluster.items() if v}
    P = len(means)
    cells = sum(len(v) for v in d_by_cluster.values())
    if P < 2:
        return {"verdict": f"NOT SCORED: {P} clusters", "clusters": P, "cells": cells}
    m = st.fmean(means.values())
    se = st.stdev(means.values()) / math.sqrt(P)
    tq = t_ppf(1 - alpha / 2, P - 1)
    lo, hi = m - tq * se, m + tq * se
    ok = lo >= -margin and hi <= margin
    return {"verdict": "EQUIVALENT" if ok else "NOT EQUIVALENT", "mean": r6(m), "se": r6(se),
            "ci": [r6(lo), r6(hi)], "df": P - 1, "t": r6(tq), "margin": margin, "clusters": P,
            "cells": cells, "cluster_means": {str(k): r6(v) for k, v in means.items()}}


def cochran_q(d: list, sigma: float = SIGMA_PAGE) -> dict:
    """Q = sum (d_i - mean)^2 / (2 sigma^2) against chi2(N - 1): each d_i is a ratio of two
    cells, each of noise sigma, so var(d_i) = 2 sigma^2."""
    d = [float(x) for x in d]
    if len(d) < 2:
        return {"verdict": "NOT SCORED", "cells": len(d)}
    m = st.fmean(d)
    Q = sum((x - m) ** 2 for x in d) / (2 * sigma * sigma)
    p = chi2_sf(Q, len(d) - 1)
    return {"Q": r6(Q), "df": len(d) - 1, "p": r6(p), "sigma": sigma, "cells": len(d),
            "heterogeneous": p < Q_ALPHA}


def g3_rule(rows: list, margin: float = G3_MARGIN, sigma: float = SIGMA_PAGE) -> dict:
    """The registered G3 on rows {cluster, stratum, d} (d = T_u/T_b - 1 - delta_M3): Q on all
    rows; homogeneous -> one TOST, a failure voids every histogram cell of the board;
    heterogeneous (p < 0.01) -> a TOST per stratum, only the failing strata void."""
    q = cochran_q([r["d"] for r in rows], sigma)
    by = {}
    for r in rows:
        by.setdefault(r["cluster"], []).append(r["d"])
    pooled = tost_cluster_t(by, margin)
    out = {"Q": q, "pooled": pooled, "rows": len(rows)}
    if not rows:
        out.update(verdict="NOT SCORED: no G3 cell", void_strata=None)
        return out
    if not q.get("heterogeneous"):
        ok = pooled["verdict"] == "EQUIVALENT"
        out.update(verdict="PASS" if ok else "FAIL (voids every histogram cell of the board)",
                   void_strata=[] if ok else "ALL")
        return out
    strata = {}
    for r in rows:
        strata.setdefault(r["stratum"], {}).setdefault(r["cluster"], []).append(r["d"])
    per = {str(k): tost_cluster_t(v, margin) for k, v in strata.items()}
    bad = [k for k, v in per.items() if v["verdict"] != "EQUIVALENT"]
    out.update(per_stratum=per, verdict="PASS (heterogeneous; every stratum equivalent)" if not bad
               else f"FAIL in strata {bad} (heterogeneous: only those strata's histogram cells void)",
               void_strata=bad)
    return out


# --------------------------------------------------------------------------
# the calibration line (logs, wild cluster bootstrap) and gamma
# --------------------------------------------------------------------------

def _ols(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return None
    b = sum((a - mx) * (c - my) for a, c in zip(x, y, strict=True)) / sxx
    return my - b * mx, b


def calibration(rows: list, *, seed: int = SEED, boot: int = BOOT) -> dict:
    """rows {pred, meas, cluster}: ln meas = a + b ln pred. The wild cluster bootstrap
    (Webb) keeps x, refits on y* = yhat + w_c e_i, one weight per cluster."""
    import numpy as np
    rows = [r for r in rows if r.get("pred") and r.get("meas") and r["pred"] > 0 and r["meas"] > 0]
    clusters = sorted({str(r["cluster"]) for r in rows})
    out = {"cells": len(rows), "clusters": len(clusters), "fit": "OLS ln(meas) on ln(pred)",
           "ci": "wild cluster bootstrap, Webb weights, 95% percentile", "seed": seed, "resamples": boot}
    if len(rows) < 3:
        out["verdict"] = f"NOT APPLICABLE ({len(rows)} cells)"
        return out
    x = [math.log(r["pred"]) for r in rows]
    y = [math.log(r["meas"]) for r in rows]
    f = _ols(x, y)
    if f is None:
        out["verdict"] = "NOT APPLICABLE (one predicted value)"
        return out
    a, b = f
    out.update(intercept=r6(a), slope=r6(b))
    if len(clusters) < MIN_CLUSTERS:
        out["verdict"] = f"NOT RESOLVED ({len(clusters)} clusters, under {MIN_CLUSTERS}): descriptive only"
        return out
    rng = np.random.default_rng(seed)
    idx = np.array([clusters.index(str(r["cluster"])) for r in rows])
    yhat = np.array([a + b * v for v in x])
    e = np.array(y) - yhat
    X = np.array(x)
    A = np.vstack([np.ones_like(X), X]).T
    w = np.array(WEBB)[rng.integers(0, 6, (boot, len(clusters)))]
    ys = yhat[None, :] + w[:, idx] * e[None, :]
    coef = np.linalg.lstsq(A, ys.T, rcond=None)[0]
    lo, hi = np.quantile(coef, [0.025, 0.975], axis=1)
    ok = lo[1] <= 1 <= hi[1] and lo[0] <= 0 <= hi[0]
    out.update(ci_intercept=[r6(lo[0]), r6(hi[0])], ci_slope=[r6(lo[1]), r6(hi[1])],
               verdict="CONSISTENT" if ok else "NOT CONSISTENT")
    return out


def _pairs(cells: list, sigma: float):
    thr = 2 * math.sqrt(2) * sigma
    C = D = 0
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            a, b = cells[i], cells[j]
            dm = math.log(a["meas"] / b["meas"])
            dp = a["pred"] - b["pred"]
            if abs(dm) < thr or dp == 0:
                continue
            if (dm > 0) == (dp > 0):
                C += 1
            else:
                D += 1
    return C, D


def gamma(rows: list, *, sigma: float = SIGMA_PAGE, seed: int = SEED, boot: int = BOOT) -> dict:
    """rows {pred, meas, stratum, cluster}: Goodman-Kruskal gamma pooled over strata (pairs
    within a stratum only), the CI by resampling clusters with replacement."""
    import numpy as np
    rows = [r for r in rows if r.get("pred") and r.get("meas")]

    def pooled(rs):
        by = {}
        for r in rs:
            by.setdefault(str(r["stratum"]), []).append(r)
        C = D = 0
        for v in by.values():
            c, d = _pairs(v, sigma)
            C += c
            D += d
        return C, D

    C, D = pooled(rows)
    out = {"name": "Goodman-Kruskal gamma (prediction ties dropped), stratified", "concordant": C,
           "discordant": D, "resolved_threshold_ln": r6(2 * math.sqrt(2) * sigma)}
    if C + D < MIN_GAMMA_PAIRS:
        out.update(gamma=r6((C - D) / (C + D)) if C + D else None,
                   verdict=f"NOT RESOLVED ({C + D} resolved pairs, under {MIN_GAMMA_PAIRS})")
        return out
    g = (C - D) / (C + D)
    clusters = sorted({str(r["cluster"]) for r in rows})
    rng = np.random.default_rng(seed)
    gs = []
    for _ in range(boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        rs = []
        for k, ci in enumerate(pick):
            # a resampled cluster keeps its cells, renamed so two draws of one page are two pages
            rs += [dict(r, stratum=f"{r['stratum']}") for r in rows if str(r["cluster"]) == clusters[ci]]
        c, d = pooled(rs)
        if c + d:
            gs.append((c - d) / (c + d))
    lo, hi = (np.quantile(gs, [0.025, 0.975]) if gs else (float("nan"), float("nan")))
    out.update(gamma=r6(g), ci=[r6(lo), r6(hi)], clusters=len(clusters),
               verdict="CONSISTENT" if lo >= GAMMA_FLOOR else "NOT CONSISTENT")
    return out


def secondary(rows: list, *, calib_key: str = "cluster", sigma: float = SIGMA_PAGE) -> dict:
    """The three secondary metrics every rental-5 registration prints."""
    cal = calibration([dict(r, cluster=r.get(calib_key)) for r in rows])
    return {"calibration": cal, "gamma": gamma(rows, sigma=sigma),
            "gamma_alt_sigma": gamma(rows, sigma=SIGMA_PAGE_ALT),
            "top1_regret": "NOT APPLICABLE (one tile on every rental-5 page; saved for the tile stage)"}


def stats(rel) -> dict:
    v = [float(x) for x in rel]
    if not v:
        return {"cells": 0, "rms": None, "worst": None, "mean": None}
    return {"cells": len(v), "rms": r6(math.sqrt(sum(x * x for x in v) / len(v))),
            "worst": r6(max(v, key=abs)), "mean": r6(st.fmean(v))}


def secondary_lines(sec: dict, indent: str = "  ") -> list[str]:
    c, g = sec["calibration"], sec["gamma"]
    out = [f"{indent}calibration (ln meas on ln pred): {c.get('verdict')}"
           + (f"; slope {c['slope']:.4f} intercept {c['intercept']:+.4f}" if c.get("slope") is not None else "")
           + (f"; CI slope {c['ci_slope']} intercept {c['ci_intercept']}" if c.get("ci_slope") else ""),
           f"{indent}gamma: {g.get('verdict')}; gamma {g.get('gamma')} (C {g['concordant']}, D {g['discordant']})"
           + (f" CI {g['ci']}" if g.get("ci") else ""),
           f"{indent}gamma at sigma {100 * SIGMA_PAGE_ALT:.2f}%: {sec['gamma_alt_sigma'].get('verdict')}",
           f"{indent}top-1 regret: {sec['top1_regret']}"]
    return out
