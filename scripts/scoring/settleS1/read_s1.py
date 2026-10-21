#!/usr/bin/env python
"""Read the settle rental S1's pages into the registered quantities (docs/registered/2026-10-21-settleS1-gh200).
DESCRIPTIVE: every number here is a reading of a SEEN page; nothing is scored against a model and no verdict is
written. The reading bands are the registration's: each names the form of v5's F1, F2 or F3 a reading points to.

    python scripts/scoring/settleS1/read_s1.py <repo> <tree> <out>

Reads under <tree> (any depth): <date>-<card>-<model>-<label>-r3-counters/lock<L>/r3c-g<G>.json for the S1 units
(registration 'units'); writes <out>/settleS1.read.{json,txt}; exits 0 whatever it reads (2 on bad arguments).

The functions, one per measurement (each takes counter pages as the route writes them; M0 the power proof's CSV):
  power_proof(csv_text)          S1-M0: 1 s means of the GPU's power.draw under the sustained GEMM, the SW power cap's
                                 share of the busy samples, enforced.power.limit as read
  gemm_rows(page)                one row per (cell, GEMM): ncu duration, SM cycles, the offset off = t - cycles / lock
  sag_fit(rows)                  off = A (1 - exp(-t / tau)) by DIAGNOSIS-A2's grid (work/transient.py), and the
                                 linear t = a + cycles / f (work/clockfit.py)
  replay_contrast(k, a)          S1-M1: application against kernel replay, per GEMM and per cell (the call's sum)
  lock_ladder(pages)             S1-M2: sag_fit per lock, and the reading against the same board's 1410 page
  m3_bytes(page, predictions)    S1-M3: counted q per GEMM and v4's registered q against it, per key
  private_split(page)            S1-M4: PRIVATE's excess over n (dram_counter_route.r3_weight_bracket, pinned) against
                                 SHARED's re-read q - 1, per GEMM and tread
  stall_table(page, consts)      S1-M5: the stall and pipe metrics per GEMM beside x_s1 (DRAM time over floor time at
                                 the lock) and y_s1 (cycles over the larger of the two), with rank correlations
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

REG_NAME = "2026-10-21-settleS1-gh200"
CLOCK = "sm__cycles_elapsed.avg"
STALL_LS = "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
STALL_BAR = "smsp__warp_issue_stalled_barrier_per_warp_active.pct"
TENSOR = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
GEMMS = ("w1", "w2")
SMS = 132
#: DIAGNOSIS-A2 work/transient.py's grid and bins
TAU_GRID_US = tuple(range(50, 5000, 25))
BINS_US = ((0, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 1e9))
#: the registered reading bands (registration 'measurements' <M>.reading)
M1_ABS_US, M1_REL, M1_SLACK_US = 10.0, 0.25, 5.0
M2_GONE_US, M2_REDUCED = 8.0, 0.5
M3_OVER, M3_EXACT = 0.04, 0.02
M4_A_DOM, M4_B_DOM, M4_TINY = 0.5, 0.2, 0.02
M5_RHO = 0.6
#: M2's fallback reference: sag_fit on A2's own OLMoE f128 G 64 page at 1410 (54 cells, rms 4.89 us), the M2 page's twin
A2_REF_PAGE = "results/published/2026-10-21-nvidia_gh200_480gb-tileA2-session/results/2026-10-20-nvidia_gh200_480gb-olmoe-1b-7b-f128-r3-counters/lock1410/r3c-g64.json"
A2_REF_A_US, A2_REF_TAU_US = 29.652628, 300.0
M0_BUSY_W, M0_WINDOW, M0_ABOVE_W, M0_ABOVE_S, M0_AT = 200.0, 10, 720.0, 10.0, (680.0, 720.0)
M0_CAP_SHARE, SW_POWER_CAP = 0.5, 0x4
#: the power proof's CSV columns (the driver's --query-gpu list, noheader, nounits)
M0_COLUMNS = ("timestamp", "power.draw", "enforced.power.limit", "power.limit", "clocks.sm", "clocks.max.sm",
              "clocks_event_reasons.active", "temperature.gpu")


def r6(x):
    return None if x is None else round(float(x), 6)


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------

def page_lock(page: dict) -> float:
    lock = (page.get("ncu") or {}).get("lock_mhz")
    if lock is None:
        raise ValueError("the page names no nvidia-smi lock (ncu.lock_mhz): S1 reads locked pages only")
    return float(lock)


def cell_label(cell: dict) -> str:
    return str(cell.get("histogram") or "balanced")


def gemm_rows(page: dict, lock: float | None = None) -> list[dict]:
    """One row per (cell, GEMM) with a duration and a clock reading: t_us (gpu__time_duration.sum, the mean over the
    page's K calls), cyc (sm__cycles_elapsed.avg, recorded), off_us = t_us - cyc / lock (the time the kernel took that
    its SM cycles at the lock do not account for), f_eff_mhz = cyc / t_us, the counted DRAM reads and grid, and every
    recorded metric."""
    lock = page_lock(page) if lock is None else float(lock)
    out = []
    for c in page.get("cells") or []:
        for g in GEMMS:
            pg = ((c.get("per_gemm") or {}).get(g)) or {}
            rec = ((c.get("recorded") or {}).get(g)) or {}
            t, cyc = pg.get("gpu_time_ns"), rec.get(CLOCK)
            if not t or cyc is None:
                continue
            t_us = float(t) / 1e3
            out.append({"arm": str(c["arm"]), "n": int(c["n"]), "label": cell_label(c), "gemm": g,
                        "t_us": t_us, "cyc": float(cyc), "off_us": t_us - float(cyc) / lock,
                        "f_eff_mhz": float(cyc) / t_us, "dram_read": pg.get("dram_bytes_read"),
                        "grid": pg.get("grid_size"), "bincount": c.get("bincount"), "recorded": dict(rec)})
    return out


def sag_fit(rows: list[dict]) -> dict:
    """off = A (1 - exp(-t / tau)), tau on DIAGNOSIS-A2's grid (50 to 4975 us, step 25), A by least squares at each
    tau, the tau of least rms kept; the linear fit t = a + cyc / f; the median offset by duration bin."""
    pts = [(r["t_us"], r["off_us"], r["cyc"]) for r in rows]
    if len(pts) < 3:
        return {"cells": len(pts), "A_us": None, "tau_us": None, "rms_us": None, "linear": None, "bins": {}}
    best = None
    for tau in TAU_GRID_US:
        X = [1 - math.exp(-t / tau) for t, _o, _c in pts]
        xx = sum(x * x for x in X)
        A = sum(x * o for x, (_t, o, _c) in zip(X, pts)) / xx
        rms = math.sqrt(sum((o - A * x) ** 2 for x, (_t, o, _c) in zip(X, pts)) / len(pts))
        if best is None or rms < best[2]:
            best = (A, tau, rms)
    n = len(pts)
    mc = sum(c for _t, _o, c in pts) / n
    mt = sum(t for t, _o, _c in pts) / n
    sxx = sum((c - mc) ** 2 for _t, _o, c in pts)
    lin = None
    if sxx > 0:
        b = sum((c - mc) * (t - mt) for t, _o, c in pts) / sxx
        a = mt - b * mc
        lin = {"a_us": r6(a), "f_mhz": r6(1.0 / b) if b else None,
               "rms_us": r6(math.sqrt(sum((t - a - b * c) ** 2 for t, _o, c in pts) / n))}
    bins = {}
    for lo, hi in BINS_US:
        v = [o for t, o, _c in pts if lo <= t < hi]
        if v:
            bins[f"{lo:g}-{hi:g} us"] = {"median_off_us": r6(statistics.median(v)), "cells": len(v)}
    return {"cells": n, "A_us": r6(best[0]), "tau_us": float(best[1]), "rms_us": r6(best[2]), "linear": lin,
            "bins": bins, "f_eff_median_mhz": r6(statistics.median(r["f_eff_mhz"] for r in rows))}


# --------------------------------------------------------------------------
# the session's power limit (s1-power.env and power-set*.txt)
# --------------------------------------------------------------------------

def _gpu_current_limit(text: str):
    g = False
    for ln in text.splitlines():
        if "GPU Power Readings" in ln:
            g = True
        elif g and "Current Power Limit" in ln:
            return _num(ln.split(":", 1)[1].split()[0]) if ":" in ln else None
    return None


def session_power(tree) -> dict:
    """Which limit the session ran under: the driver's gh200-driver/s1-power.env lines (S1_POWER_900=refused,
    S1_POWER_W=...), and the -q GPU-scope Current Power Limit of power-set.txt and each power-set-unsettled*.txt.
    The registration's power_and_lock.outcomes are keyed on these."""
    env = sorted(Path(tree).rglob("gh200-driver/s1-power.env"))
    lines = env[0].read_text().split() if len(env) == 1 else []
    kv = dict(ln.split("=", 1) for ln in lines if "=" in ln)
    sets = {p.name: _gpu_current_limit(p.read_text()) for p in sorted(Path(tree).rglob("power-set*.txt"))}
    w = kv.get("S1_POWER_W")
    if w is None:
        outcome = "both refused" if kv.get("S1_POWER_700") == "refused" else None
    elif kv.get("S1_POWER_900") == "refused":
        outcome = "900 W refused, 700 W read back"
    else:
        outcome = "900 W read back"
    return {"env": lines, "limit_w": _num(w), "power_set_gpu_current_w": sets, "outcome": outcome}


# --------------------------------------------------------------------------
# S1-M0: the power proof
# --------------------------------------------------------------------------

def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def power_proof(csv_text: str) -> dict:
    """The samples nvidia-smi wrote every 100 ms during the GEMM (M0_COLUMNS). Busy samples: power.draw > 200 W.
    1 s means: every run of 10 consecutive busy samples. The reading (registration M0): ABOVE-700 when 1 s means over
    720 W cover >= 10 s; AT-700 when the largest 1 s mean lies in [680, 720] W, the SW power cap bit (0x4) is set on
    >= half the busy samples and the busy clock's median is under clocks.max.sm; CAP-BELOW-680 when the cap bit is
    set on >= half the busy samples and the largest 1 s mean is under 680 W; else UNDECIDED."""
    rows = []
    for ln in csv_text.splitlines():
        v = [x.strip() for x in ln.split(",")]
        if len(v) != len(M0_COLUMNS):
            continue
        d = dict(zip(M0_COLUMNS, v))
        w = _num(d["power.draw"])
        if w is None:
            continue
        try:
            reasons = int(d["clocks_event_reasons.active"], 16)
        except ValueError:
            reasons = None
        rows.append({"w": w, "enforced": _num(d["enforced.power.limit"]), "limit": _num(d["power.limit"]),
                     "sm": _num(d["clocks.sm"]), "max": _num(d["clocks.max.sm"]), "reasons": reasons})
    busy = [r for r in rows if r["w"] > M0_BUSY_W]
    means = [statistics.fmean(r["w"] for r in busy[i:i + M0_WINDOW]) for i in range(len(busy) - M0_WINDOW + 1)]
    cap = [r for r in busy if r["reasons"] is not None and r["reasons"] & SW_POWER_CAP]
    share = len(cap) / len(busy) if busy else 0.0
    peak = max(means) if means else None
    above_s = sum(1 for m in means if m > M0_ABOVE_W) * 0.1
    sm = [r["sm"] for r in busy if r["sm"] is not None]
    mx = [r["max"] for r in busy if r["max"] is not None]
    clock_under = bool(sm and mx and statistics.median(sm) < statistics.median(mx))
    if peak is None:
        reading = None
    elif above_s >= M0_ABOVE_S:
        reading = "ABOVE-700"
    elif M0_AT[0] <= peak <= M0_AT[1] and share >= M0_CAP_SHARE and clock_under:
        reading = "AT-700"
    elif share >= M0_CAP_SHARE and peak < M0_AT[0]:
        reading = "CAP-BELOW-680"
    else:
        reading = "UNDECIDED"
    enf = [r["enforced"] for r in rows if r["enforced"] is not None]
    lim = [r["limit"] for r in rows if r["limit"] is not None]
    return {"samples": len(rows), "busy_samples": len(busy), "peak_1s_mean_w": r6(peak), "seconds_over_720_w": r6(above_s),
            "sw_power_cap_share": r6(share), "busy_clock_median_mhz": r6(statistics.median(sm)) if sm else None,
            "enforced_limit_w": sorted(set(enf)), "power_limit_w": sorted(set(lim)), "reading": reading}


# --------------------------------------------------------------------------
# S1-M1: the replay contrast
# --------------------------------------------------------------------------

def _by_key(rows):
    return {(r["arm"], r["n"], r["label"], r["gemm"]): r for r in rows}


def replay_contrast(kernel_page: dict, app_page: dict) -> dict:
    """Per (cell, GEMM): off under kernel replay (A2's instrument) and under application replay, and their
    difference d = off_app - off_kernel; per cell the call's sum over w1 and w2. The reading (registration M1):
    INSTRUMENT-INDEPENDENT when |median d_cell| <= 10 us and every |d_cell| <= 0.25 x off_kernel_cell + 5 us;
    INSTRUMENT-INFLATED when median d_cell < -10 us (the kernel-replay sag is partly the instrument's);
    INSTRUMENT-DEFLATED when median d_cell > +10 us; else MIXED."""
    for name, p in (("kernel", kernel_page), ("application", app_page)):
        mode = (p.get("ncu") or {}).get("replay_mode")
        if mode != name:
            raise ValueError(f"the {name}-replay page records replay_mode {mode!r}")
    k, a = _by_key(gemm_rows(kernel_page)), _by_key(gemm_rows(app_page))
    gem, cells = [], {}
    for key in sorted(set(k) & set(a)):
        d = a[key]["off_us"] - k[key]["off_us"]
        gem.append({"cell": f"n{key[1]}/{key[2]}/{key[0]}", "gemm": key[3], "off_kernel_us": r6(k[key]["off_us"]),
                    "off_app_us": r6(a[key]["off_us"]), "d_us": r6(d), "t_kernel_us": r6(k[key]["t_us"]),
                    "t_app_us": r6(a[key]["t_us"]), "f_eff_kernel_mhz": r6(k[key]["f_eff_mhz"]),
                    "f_eff_app_mhz": r6(a[key]["f_eff_mhz"])})
        c = cells.setdefault(key[:3], {"off_kernel_us": 0.0, "off_app_us": 0.0, "gemms": 0})
        c["off_kernel_us"] += k[key]["off_us"]
        c["off_app_us"] += a[key]["off_us"]
        c["gemms"] += 1
    per_cell = []
    for key, c in sorted(cells.items()):
        if c["gemms"] != len(GEMMS):
            continue
        per_cell.append({"cell": f"n{key[1]}/{key[2]}/{key[0]}", "off_kernel_us": r6(c["off_kernel_us"]),
                         "off_app_us": r6(c["off_app_us"]), "d_us": r6(c["off_app_us"] - c["off_kernel_us"])})
    unmatched = sorted(f"{x}" for x in set(k) ^ set(a))
    if not per_cell:
        return {"gemms": gem, "cells": [], "unmatched": unmatched, "reading": None}
    ds = [c["d_us"] for c in per_cell]
    med = statistics.median(ds)
    within = all(abs(c["d_us"]) <= M1_REL * abs(c["off_kernel_us"]) + M1_SLACK_US for c in per_cell)
    if abs(med) <= M1_ABS_US and within:
        reading = "INSTRUMENT-INDEPENDENT"
    elif med < -M1_ABS_US:
        reading = "INSTRUMENT-INFLATED"
    elif med > M1_ABS_US:
        reading = "INSTRUMENT-DEFLATED"
    else:
        reading = "MIXED"
    return {"gemms": gem, "cells": per_cell, "unmatched": unmatched, "median_d_cell_us": r6(med),
            "max_abs_d_cell_us": r6(max(abs(x) for x in ds)), "reading": reading}


# --------------------------------------------------------------------------
# S1-M2: the sag against the lock
# --------------------------------------------------------------------------

def lock_ladder(pages: dict, ref_lock: float = 1410.0) -> dict:
    """sag_fit per lock ({lock MHz: page}, every cell and arm of the OLMoE f128 page), and per lower lock the
    reading against the same board's ref_lock page (registration M2): GONE when A(L) <= 8 us; REDUCED when A(L) <=
    0.5 x A(ref); else PRESENT. Without a ref page the reading compares with A2's fit of the same page, A2's OLMoE f128
    G 64 page at 1410 (A2_REF_A_US, another board), labelled so."""
    fits = {}
    for lock, page in sorted(pages.items()):
        if abs(page_lock(page) - float(lock)) > 0.5:
            raise ValueError(f"the page keyed {lock} MHz records lock {page_lock(page)}")
        fits[float(lock)] = sag_fit(gemm_rows(page))
    ref = fits.get(float(ref_lock))
    ref_A, ref_src = ((ref or {}).get("A_us"), f"this board's {ref_lock:g} MHz page") if ref and ref.get("A_us") is not None \
        else (A2_REF_A_US, f"A2's OLMoE f128 G 64 page at 1410 MHz, sag_fit {A2_REF_A_US} us / tau {A2_REF_TAU_US:g} us, another board ({A2_REF_PAGE})")
    readings = {}
    for lock, f in fits.items():
        if lock >= ref_lock or f.get("A_us") is None:
            continue
        A = f["A_us"]
        readings[f"{lock:g}"] = ("GONE" if A <= M2_GONE_US else "REDUCED" if A <= M2_REDUCED * ref_A else "PRESENT")
    return {"fits": {f"{k:g}": v for k, v in fits.items()}, "ref_A_us": r6(ref_A), "ref_source": ref_src,
            "readings": readings}


# --------------------------------------------------------------------------
# S1-M3: Mixtral 128 G 1 bytes
# --------------------------------------------------------------------------

def byte_model(page: dict) -> dict:
    import dram_counter_route as DCR
    from moe.spec import MODEL_CONFIGS
    d = page["design"]
    return DCR.r3_byte_model(MODEL_CONFIGS[d["model"]], d["dtype"], int(d["block_m"]))


def q_counted(cell: dict, bm: dict) -> dict:
    """(counted read bytes - n x the GEMM's A operand per tread) / the GEMM's weight set, as score_bytes.py."""
    n = int(cell["n"])
    return {g: (float(cell["per_gemm"][g]["dram_bytes_read"]) - n * bm[f"operand_per_tile_{g}"]) / bm[f"W_{g}"]
            for g in GEMMS}


def m3_bytes(page: dict, predictions: list[dict], bm: dict | None = None) -> dict:
    """Per NATIVE cell: q counted per GEMM and e = q_pred / q_counted - 1 per registered key (v3, CLAMP). The reading
    on w1 under CLAMP (v4's key at A2), the mean over the page's cells (registration M3): W1-OVER when >= +4%
    (DIAGNOSIS-A2 Q2's inference that v4 over-prices Mixtral 128 G 1's w1 by about 7%), W1-EXACT when |mean| < 2%,
    else W1-PARTIAL."""
    bm = bm or byte_model(page)
    pred = {(r["n"], r["label"], r["arm"]): r for r in predictions}
    rows = []
    for c in page.get("cells") or []:
        key = (int(c["n"]), cell_label(c), str(c["arm"]))
        pr = pred.get(key)
        if pr is None:
            continue
        q = q_counted(c, bm)
        e = {k: {g: r6(v[g] / q[g] - 1) for g in GEMMS if g in v} for k, v in pr["q_pred"].items() if v}
        rows.append({"cell": f"n{key[0]}/{key[1]}/{key[2]}", "q_counted": {g: r6(x) for g, x in q.items()}, "e": e})
    if not rows:
        return {"rows": [], "reading": None}
    keys = sorted({k for r in rows for k in r["e"]})
    mean = {k: {g: r6(statistics.fmean(r["e"][k][g] for r in rows if g in r["e"].get(k, {})))
                for g in GEMMS if any(g in r["e"].get(k, {}) for r in rows)} for k in keys}
    m = mean.get("CLAMP", {}).get("w1")
    reading = None if m is None else ("W1-OVER" if m >= M3_OVER else "W1-EXACT" if abs(m) < M3_EXACT else "W1-PARTIAL")
    return {"rows": rows, "mean_e": mean, "reading": reading}


# --------------------------------------------------------------------------
# S1-M4: PRIVATE's excess, A against B re-reads
# --------------------------------------------------------------------------

def private_split(page: dict) -> dict:
    """Per GEMM and tread of a ladder page with all three arms: q per arm (dram_counter_route.r3_q), PRIVATE's excess
    over n e_P (r3_weight_bracket at the K-call means, and with r3_call_edges at its highest call), SHARED's re-read
    q_S - 1 (one weight set compulsory at a balanced cell, every expert active), and the share of it PRIVATE's excess
    bounds as activation re-reads, a = min(e_P / (q_S - 1), 1). The reading per GEMM (registration M4): B-DOMINATED
    when at every tread e_P <= 0.2 x (q_S - 1) or e_P <= 0.02 weight sets; A-DOMINATED when e_P >= 0.5 x (q_S - 1) at
    most treads; else MIXED."""
    import dram_counter_route as DCR
    arms = set(page["design"]["arms"])
    if not {"shared", "private"} <= arms:
        raise ValueError(f"arms {sorted(arms)}: the split needs SHARED and PRIVATE")
    bm = byte_model(page)
    q = DCR.r3_q(page, bm)
    treads = sorted(q["shared"]["total"])
    e_mean, _lo, _hi = DCR.r3_weight_bracket(q, treads)
    e_edge, _lo2, _hi2 = DCR.r3_weight_bracket(q, treads, DCR.r3_call_edges(page, bm))
    out, readings = {}, {}
    for g in GEMMS:
        rows = []
        for n in treads:
            qs, qp = q["shared"][g][n], q["private"][g][n]
            ex = qs - 1.0
            ep = e_mean[g][n]
            rows.append({"n": n, "q_native": r6(q.get("native", {}).get(g, {}).get(n)), "q_shared": r6(qs),
                         "q_private": r6(qp), "e_P": r6(ep), "e_P_highest_call": r6(e_edge[g][n]),
                         "shared_reread": r6(ex), "a_share_max": r6(min(ep / ex, 1.0)) if ex > 0 else None})
        out[g] = rows
        small = [r["e_P"] <= M4_B_DOM * max(r["shared_reread"], 0.0) or r["e_P"] <= M4_TINY for r in rows]
        big = [r["shared_reread"] > 0 and r["e_P"] >= M4_A_DOM * r["shared_reread"] for r in rows]
        readings[g] = "B-DOMINATED" if all(small) else "A-DOMINATED" if sum(big) * 2 > len(rows) else "MIXED"
    return {"treads": treads, "group_m": int(page["design"]["group_m"]), "block_m": int(page["design"]["block_m"]),
            "gemms": out, "readings": readings}


# --------------------------------------------------------------------------
# S1-M5: stall reasons along x
# --------------------------------------------------------------------------

def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _ranks(list(x)), _ranks(list(y))
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def live_ctas(row: dict, design: dict, cfg) -> int:
    """Live CTAs of a GEMM: live M-tiles (sum over experts of ceil(rows / BLOCK_M), from the cell's bincount; E x n
    at a balanced cell without one) times the GEMM's N-tiles (DIAGNOSIS-A2 work/floor_cycles.py)."""
    bm, bn = int(design["block_m"]), int(design["block_n"])
    N = 2 * cfg.intermediate_size if row["gemm"] == "w1" else cfg.hidden_size
    counts = row.get("bincount")
    rows = sum(math.ceil(int(x) / bm) for x in counts) if counts else cfg.num_experts * row["n"]
    return rows * math.ceil(N / bn)


def stall_table(page: dict, consts: dict) -> dict:
    """Per NATIVE (cell, GEMM): the three S1 metrics, x_s1 and y_s1. floor_cyc = q (S c + F_g) with q = ceil(live
    CTAs / 132) and S = K / BLOCK_K (the tcommon.joint_cF form, at the registered CAL-A c and F of A2); bytes_cyc =
    counted read bytes / bw x lock (cycles at the lock, bw the registered v4 bandwidth); x_s1 = bytes_cyc / floor_cyc;
    y_s1 = cyc / max(floor_cyc, bytes_cyc). Rank correlations of x_s1 with each metric. The reading (registration
    M5): LATENCY when rho(x, long scoreboard) >= 0.6 and rho(x, tensor active) <= -0.6; BARRIER when rho(x, barrier)
    >= 0.6 and rho(x, long scoreboard) < 0.6; else NONE-SEEN."""
    from moe.spec import MODEL_CONFIGS
    d = page["design"]
    cfg = MODEL_CONFIGS[d["model"]]
    lock = page_lock(page)
    bk = int(d["block_k"])
    c, F, bw = float(consts["c_cycles"]), consts["F_cycles"], float(consts["bw_gbps"])
    rows = []
    for r in gemm_rows(page):
        if r["arm"] != "native" or r["dram_read"] is None:
            continue
        K = cfg.hidden_size if r["gemm"] == "w1" else cfg.intermediate_size
        q = math.ceil(live_ctas(r, d, cfg) / SMS)
        floor_cyc = q * ((K // bk) * c + float(F[r["gemm"]]))
        bytes_cyc = float(r["dram_read"]) / (bw * 1e9) * lock * 1e6
        rec = r["recorded"]
        rows.append({"cell": f"n{r['n']}/{r['label']}", "gemm": r["gemm"], "x_s1": r6(bytes_cyc / floor_cyc),
                     "y_s1": r6(r["cyc"] / max(floor_cyc, bytes_cyc)), "off_us": r6(r["off_us"]),
                     "long_scoreboard_pct": rec.get(STALL_LS), "barrier_pct": rec.get(STALL_BAR),
                     "tensor_active_pct": rec.get(TENSOR)})

    def rho(k):
        pts = [(x["x_s1"], x[k]) for x in rows if x[k] is not None]
        return r6(spearman([a for a, _b in pts], [b for _a, b in pts])) if pts else None
    rh = {"long_scoreboard": rho("long_scoreboard_pct"), "barrier": rho("barrier_pct"),
          "tensor_active": rho("tensor_active_pct"), "y_s1": rho("y_s1")}
    if rh["long_scoreboard"] is None and rh["barrier"] is None:
        reading = None
    elif (rh["long_scoreboard"] or 0) >= M5_RHO and (rh["tensor_active"] if rh["tensor_active"] is not None else 0) <= -M5_RHO:
        reading = "LATENCY"
    elif (rh["barrier"] or 0) >= M5_RHO and (rh["long_scoreboard"] or 0) < M5_RHO:
        reading = "BARRIER"
    else:
        reading = "NONE-SEEN"
    return {"rows": rows, "rho_x": rh, "reading": reading}


# --------------------------------------------------------------------------
# the tree
# --------------------------------------------------------------------------

def find_page(tree: Path, model: str, label: str, G: int) -> tuple[dict | None, Path | None, str]:
    hits = sorted(Path(tree).rglob(f"*-{model}-{label}-r3-counters/lock*/r3c-g{G}.json"))
    if len(hits) != 1:
        return None, None, f"{len(hits)} pages for {model} {label} G {G}"
    return json.loads(hits[0].read_text()), hits[0], "read"


def registration(repo: Path) -> dict:
    return json.loads((Path(repo) / "docs" / "registered" / f"{REG_NAME}.json").read_text())


def read_tree(repo: Path, tree: Path) -> dict:
    reg = registration(repo)
    res: dict = {"registration": REG_NAME, "status": "DESCRIPTIVE (SEEN for v5; no verdict)", "pages": {}}

    def get(model, label, G):
        p, path, why = find_page(tree, model, label, G)
        res["pages"][f"{model}-{label}"] = str(path) if p is not None else why
        return p
    res["power"] = session_power(tree)
    pp = sorted(Path(tree).rglob("*-powerproof-pp/smi.csv"))
    res["pages"]["powerproof"] = str(pp[0]) if len(pp) == 1 else f"{len(pp)} power-proof CSVs"
    res["M0"] = power_proof(pp[0].read_text()) if len(pp) == 1 else None
    m1 = {}
    for model in ("olmoe-1b-7b", "qwen2-57b-a14b"):
        k, a = get(model, "kr", 1), get(model, "ar", 1)
        m1[model] = replay_contrast(k, a) if (k is not None and a is not None) else None
    res["M1"] = m1
    lad, off = {}, []
    for lock, label in ((1410.0, "f1410"), (1305.0, "f1305"), (1005.0, "f1005")):
        p = get("olmoe-1b-7b", label, 64)
        if p is not None:
            lad[page_lock(p)] = p
            if abs(page_lock(p) - lock) > 0.5:
                off.append(f"{label} ran at {page_lock(p):g} MHz, not {lock:g}")
    res["M2"] = lock_ladder(lad) if lad else None
    if res["M2"] is not None and off:
        res["M2"]["lock_not_the_units"] = off
    p = get("mixtral-8x7b", "b128g1", 1)
    res["M3"] = m3_bytes(p, reg["byte_predictions_m3"]) if p is not None else None
    res["M4"] = {}
    for label, G in (("p128g16", 16), ("p64g1", 1)):
        p = get("mixtral-8x7b", label, G)
        res["M4"][label] = private_split(p) if p is not None else None
    res["M5"] = {}
    for model in ("qwen2-57b-a14b", "olmoe-1b-7b"):
        p = get(model, "st", 1)
        res["M5"][model] = stall_table(p, reg["constants"]["M5"]) if p is not None else None
    return res


def lines(res: dict) -> list[str]:
    out = [f"READ against {res['registration']} ({res['status']})"]
    pw = res.get("power") or {}
    out.append(f"power: {pw.get('outcome')} (limit {pw.get('limit_w')} W; {pw.get('env')})")
    m0 = res.get("M0")
    out.append("M0: " + ("no CSV" if not m0 else f"{m0['reading']} peak 1 s mean {m0['peak_1s_mean_w']} W, cap share "
                                                 f"{m0['sw_power_cap_share']}, enforced {m0['enforced_limit_w']} W"))
    for model, m in (res.get("M1") or {}).items():
        out.append(f"M1 {model}: " + ("no pair" if not m else f"{m['reading']} median d {m.get('median_d_cell_us')} us"))
    m2 = res.get("M2")
    out.append("M2: " + ("no page" if not m2 else
                         f"A by lock {({k: v['A_us'] for k, v in m2['fits'].items()})} us; {m2['readings']} (ref {m2['ref_source']})"))
    m3 = res.get("M3")
    out.append("M3: " + ("no page" if not m3 else f"{m3['reading']} mean e {m3.get('mean_e')}"))
    for k, v in (res.get("M4") or {}).items():
        out.append(f"M4 {k}: " + ("no page" if not v else f"{v['readings']}"))
    for k, v in (res.get("M5") or {}).items():
        out.append(f"M5 {k}: " + ("no page" if not v else f"{v['reading']} rho {v['rho_x']}"))
    return out


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    res = read_tree(Path(a[0]), Path(a[1]))
    out = Path(a[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / "settleS1.read.json").write_text(json.dumps(res, indent=1, default=float, sort_keys=True) + "\n")
    text = "\n".join(lines(res)) + "\n"
    (out / "settleS1.read.txt").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
