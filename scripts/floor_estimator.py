#!/usr/bin/env python
"""The floor falsifier's estimator and the quantities rental 2 derives from it
(docs/registered rental2-w1floor, rental2-const). Offline, no GPU, no fit
beyond the least-squares lines named here.

implied_slope   F4's estimator verbatim: the slope of a GEMM's cycles over
                grid x S / sms (CTA k-steps per SM) on a tread set, by
                np.polyfit(x, y, 1) as scripts/scoring/rental1/score_floor.py
                takes it. Applied to measured cycles it is the measured slope;
                applied to a model's per-cell cycles it is the slope that model
                implies for that tread set (H_EST).
theta_delta     per GEMM, from three or more sets: slope_k / asymptote - 1 =
                delta + theta x bias_k, bias_k the model's own estimator bias on
                set k (H_EST's slope / asymptote - 1). Unweighted OLS over the
                sets. theta 1 says the slopes follow the model's ceil
                quantisation, theta 0 says they do not (FLUID).
linear_weights  the cell-to-estimate weights of any of the above, so the
                registered noise model propagates EXACTLY through tread sets
                that share cells (sigma = sigma_cell x ||weights||).
intercept       OLS intercept and slope of y on x (part 3's Z: SM-active
                cycles regressed on q = N_live / 132, the slope free).
form_rms        part 4's co-primary per-cell test: the rms of y about
                c x X + a with a single free constant a, X = ceil(q) u or q u.
ols             rental 3 (2026-10-05): ordinary least squares of y on given
                columns, with the coefficients' covariance from the residuals.
joint_theta     rental 3's floor law per ruler: y / u = a + b q + theta frac,
                frac = ceil(q) - q the last wave's idle-SM share; theta 1 is the
                ceil law on that ruler, 0 the fluid law (docs/registered
                rental3-floorlaw).
rulers          the same cell read on six rulers: EL.avg, EL.max, ACT.max,
                ACT.avg, DUR (gpu__time_duration x f) and the two differences X
                = DUR - EL.avg (the per-kernel excess outside the SMs' elapsed
                windows) and GAP = EL.avg - ACT.max (both SM-clock counters).
"""
from __future__ import annotations

import math

import numpy as np

SMS = 132


def estimator_x(grid: int, ksteps: int, sms: int = SMS) -> float:
    """CTA k-steps per SM: the abscissa F4's slope is taken against."""
    return int(grid) * int(ksteps) / int(sms)


def implied_slope(cells: dict, treads, ksteps: int, sms: int = SMS) -> float:
    """`cells` maps tread n to (grid, cycles). The slope over `treads`."""
    ns = list(treads)
    if len(ns) < 2:
        raise ValueError(f"a slope needs two treads, got {ns}")
    missing = [n for n in ns if n not in cells]
    if missing:
        raise KeyError(f"treads {missing} have no cell")
    xs = [estimator_x(cells[n][0], ksteps, sms) for n in ns]
    ys = [float(cells[n][1]) for n in ns]
    return float(np.polyfit(xs, ys, 1)[0])


def slope_weights(grids: dict, treads, ksteps: int, sms: int = SMS) -> dict:
    """{n: w_n} with slope = sum w_n y_n (the OLS slope is linear in y)."""
    ns = list(treads)
    xs = np.array([estimator_x(grids[n], ksteps, sms) for n in ns])
    xc = xs - xs.mean()
    sxx = float((xc ** 2).sum())
    if sxx <= 0:
        raise ValueError(f"treads {ns} span no x")
    return {n: float(w) for n, w in zip(ns, xc / sxx, strict=True)}


def theta_delta(sets: list[dict]) -> dict:
    """Each set: {"slope": measured slope, "asymptote": a, "bias": model bias}
    (bias as a fraction, e.g. -0.110). Returns theta, delta and the OLS's own
    residual. Needs at least two sets with distinct biases (three to have a
    residual)."""
    if len(sets) < 2:
        raise ValueError("theta and delta need at least two tread sets")
    b = np.array([float(s["bias"]) for s in sets])
    r = np.array([float(s["slope"]) / float(s["asymptote"]) - 1 for s in sets])
    if float(np.ptp(b)) <= 0:
        raise ValueError("the sets' biases are equal: theta is not identified")
    theta, delta = np.polyfit(b, r, 1)
    res = r - (delta + theta * b)
    return {"theta": float(theta), "delta": float(delta),
            "residual_rms": float(math.sqrt((res ** 2).mean()))}


def theta_delta_weights(sets: list[dict], grids: dict, ksteps: int,
                        sms: int = SMS) -> tuple[dict, dict]:
    """({n: w} for theta, {n: w} for delta) over the cells the sets read, each
    set {"treads": (...), "asymptote": a, "bias": b}: theta and delta are linear
    in the cells' cycles, so their noise follows from these weights."""
    b = np.array([float(s["bias"]) for s in sets])
    bc = b - b.mean()
    sbb = float((bc ** 2).sum())
    k = len(sets)
    wt, wd = {}, {}
    for j, s in enumerate(sets):
        # theta = sum_j bc_j r_j / sbb ; delta = mean(r) - theta mean(b)
        a_theta = bc[j] / sbb
        a_delta = 1.0 / k - a_theta * b.mean()
        for n, w in slope_weights(grids, s["treads"], ksteps, sms).items():
            per = w / float(s["asymptote"])
            wt[n] = wt.get(n, 0.0) + a_theta * per
            wd[n] = wd.get(n, 0.0) + a_delta * per
    return wt, wd


def norm(weights: dict) -> float:
    return float(math.sqrt(sum(w * w for w in weights.values())))


def intercept(xs, ys) -> tuple[float, float]:
    """(intercept, slope) of the OLS line of ys on xs."""
    xs, ys = list(map(float, xs)), list(map(float, ys))
    if len(xs) < 2 or float(np.ptp(xs)) <= 0:
        raise ValueError("an intercept needs two distinct x")
    slope, icpt = np.polyfit(xs, ys, 1)
    return float(icpt), float(slope)


def intercept_sigma(xs, sigma: float) -> float:
    """The OLS intercept's standard error at a per-point noise `sigma`."""
    x = np.array(list(map(float, xs)))
    n = len(x)
    sxx = float(((x - x.mean()) ** 2).sum())
    return float(sigma * math.sqrt(1.0 / n + x.mean() ** 2 / sxx))


def form_rms(ys, X) -> float:
    """rms of ys about X + a, a the one free constant (its least-squares value
    is mean(ys - X))."""
    d = np.array(list(map(float, ys))) - np.array(list(map(float, X)))
    return float(math.sqrt(((d - d.mean()) ** 2).mean()))


def ols(y, columns) -> dict:
    """OLS of y on the given columns (each a sequence as long as y): the
    coefficients, their covariance s^2 (X'X)^-1 with s the residual rms over
    n - k degrees of freedom, s, and the condition number of X. Raises when
    there are no residual degrees of freedom."""
    Y = np.asarray(list(map(float, y)))
    X = np.column_stack([np.asarray(list(map(float, c))) for c in columns])
    n, k = X.shape
    if n <= k:
        raise ValueError(f"{n} points for {k} coefficients: no residual degree of freedom")
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ coef
    s = float(math.sqrt(float((res ** 2).sum()) / (n - k)))
    cov = (s * s) * np.linalg.pinv(X.T @ X)
    return {"coef": [float(v) for v in coef], "cov": cov.tolist(), "s": s,
            "cond": float(np.linalg.cond(X)), "n": int(n)}


def joint_theta(y_over_u, q, frac) -> dict:
    """y / u = a + b q + theta frac by OLS: a, b, theta, their standard errors, the
    residual rms (in u) and the condition number."""
    q = list(map(float, q))
    f = ols(y_over_u, [[1.0] * len(q), q, list(map(float, frac))])
    a, b, th = f["coef"]
    se = [float(math.sqrt(max(f["cov"][i][i], 0.0))) for i in range(3)]
    return {"a": a, "b": b, "theta": th, "se_a": se[0], "se_b": se[1], "se_theta": se[2],
            "sigma_u": f["s"], "cond": f["cond"], "cells": f["n"]}


#: the metrics `rulers` reads off a floor capture's per-GEMM block
RULER_METRICS = ("sm__cycles_elapsed.avg", "sm__cycles_elapsed.max", "sm__cycles_active.max",
                 "sm__cycles_active.avg", "gpu__time_duration.sum", "lts__cycles_elapsed.avg")


def rulers(per_gemm: dict, f_mhz: float | None) -> dict:
    """One (cell, GEMM) of a floor capture on every ruler, in SM cycles: EL.avg,
    EL.max, ACT.max, ACT.avg, DUR = gpu__time_duration (ns) x f_mhz / 1e3 (None
    without f), X = DUR - EL.avg, GAP = EL.avg - ACT.max, and the L2 clock
    lts__cycles_elapsed.avg / duration in MHz. A missing metric reads None."""
    g = {k: per_gemm.get(k) for k in RULER_METRICS}
    dur_ns = g["gpu__time_duration.sum"]
    el, elx = g["sm__cycles_elapsed.avg"], g["sm__cycles_elapsed.max"]
    acx, aca = g["sm__cycles_active.max"], g["sm__cycles_active.avg"]
    DUR = None if dur_ns is None or f_mhz is None else float(dur_ns) * float(f_mhz) / 1e3
    return {"EL.avg": el, "EL.max": elx, "ACT.max": acx, "ACT.avg": aca, "DUR": DUR,
            "X": None if DUR is None or el is None else DUR - el,
            "GAP": None if el is None or acx is None else el - acx,
            "L2_mhz": (None if not dur_ns or g["lts__cycles_elapsed.avg"] is None
                       else g["lts__cycles_elapsed.avg"] / float(dur_ns) * 1e3),
            "duration_ns": dur_ns}
