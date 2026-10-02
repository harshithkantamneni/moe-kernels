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
