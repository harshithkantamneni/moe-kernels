"""The tile stage's shared rules (docs/registered/2026-10-14-tileA-* and 2026-10-14-tileB-*).

One module for the registration generator (register.py, regtile.py) and every tile-stage scorer, so
a rule is computed one way on both sides. It imports rental 7's r7common (which imports rentals 6,
5, 4 and 3); nothing is copied. Pure functions over numbers, pages and the registered JSON.

THE RULES THIS FILE OWNS (each stated in full in the registrations; the 14 scorer ambiguities of
design-tile-review section 9 are resolved in regtile.AMBIGUITIES and implemented here):
  joint_cF       c, F_w1, F_w2 of one tile from G-64 counter pages: sm__cycles_elapsed.avg =
                 a(page, GEMM, arm) + ceil(N_live / SMs) (S_g c + F_g) over the NATIVE and SHARED
                 cells at treads 4..9 whose GEMM reads DRAM at no more than half the 4022 GB/s peak
                 (the cell's own GEMM rate), one c for both GEMMs (the estimator that measured
                 344.1 / 520 / 979 at R3's tile).
  floor_test     each accounting's predicted u = S c + F per (page, GEMM) on the 64x128 pages
                 against the page's measured u, within 3%: DESCRIPTIVE (design-tile-review 1c).
  variant_rule   which accounting rental B's v4 uses at 64x128 (D3): the one whose u holds on every
                 scored (page, GEMM) when each other one fails somewhere; else none, and 64x128 takes
                 v4cal (its measured c and F, CAL-A).
  f_form_rule    F-step (v3's form) unless, on BOTH GEMMs, F-cyc is nearer the measured 64x128 F
                 AND F-step misses it by more than 3 se on at least one.
  named_key      the A key: the nearest of the four by rms over every scored w1 row (uniform,
                 balanced and skewed, design-tile-review 5) is NAMED when the next is at least 2x
                 farther and every row of the nearest is within 5%; else CLAMP.
  stop_rule      rental B runs only if every tA-e1cal tile class's mean NATIVE residual is within
                 0.75%, the two CAL models at 128 G 1 agree within 1%, every E1-CAL page ran and
                 was valid, and tA-e1cal's bar holds (D6).
  e1tile_rule    E1-TILE: pooled AND each BLIND-TILE model at the time bar; CERTIFIED in CLEAN with
                 2 of the model's pages there; deepseek-v2-lite in the pooled bar, its own bar
                 DESCRIPTIVE.
  rival_rule     rental 7's exclusion on lever cells (|T_R / T_v4 - 1| >= 2%, at least 6): R's rms
                 over 2% and over v4's rms + 3 se.
  dead_rule      D against SLOT at occupancy 2 (z at sigma_page); kappa only when c_128 >= 900.
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
for p in (str(REPO), str(REPO / "scripts"), str(HERE.parent / "rental7"), str(HERE.parent / "rental6")):
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


C7 = _load("r7common", HERE.parent / "rental7" / "r7common.py")
C6, C5, C4, C3 = C7.C6, C7.C5, C7.C4, C7.C3
CARD = C6.CARD
DATE = "2026-10-14"
PARTS_A = ("census", "floor", "bytes", "e1cal", "dead", "rep")
PARTS_B = ("e1tile", "stop")
PARTS = PARTS_A + PARTS_B
NAMES = {**{p: f"{DATE}-tileA-{p}-gh200" for p in PARTS_A}, **{p: f"{DATE}-tileB-{p}-gh200" for p in PARTS_B}}
HIST = {"A": f"docs/registered/{DATE}-tileA-hist", "B": f"docs/registered/{DATE}-tileB-hist"}
r6 = C6.r6
SMS = 132

# ---- tA-floor ------------------------------------------------------------------------------
FLOOR_NS = (4, 5, 6, 7, 8, 9)
DRAM_PEAK_GBPS = 4022.0
FLOOR_MAX_DRAM_FRAC = 0.5
FLOOR_TOL = 0.03
FCYC_K_SE = 3.0
# ---- tA-bytes ------------------------------------------------------------------------------
KEYS = ("v3", "CLAMP", "ROWKEY60", "ROWKEY30")
NAMED_FACTOR, NAMED_MAX_ABS = 2.0, 0.05
FALLBACK_KEY = "CLAMP"
# ---- time ----------------------------------------------------------------------------------
E1_BAR = dict(C6.E1_BAR)
CLASSES = ("64g1", "128g1", "128g16")
STOP_CLASS_MEAN = 0.0075
STOP_PAIR = 0.01
MIN_CLEAN_PAGES_PER_MODEL = 2
SIGMA_PAGE_R3 = C6.SIGMA_PAGE               # 0.18% (rental 3 rep8), the fallback when tA-rep is lost
LEVER_FRAC, LEVER_MIN, RIVAL_RMS_MIN, RIVAL_K_SE = 0.02, 6, 0.02, 3.0
# ---- tA-dead -------------------------------------------------------------------------------
DEAD_Z = 3.0
KAPPA_C128_MIN = 900.0
#: the host rule (rental 3's, registered since): a cell whose registered prediction + F240 is
#: under 0.40 ms is host-bound and cut
HOST_CUT_MS, F240_MS = C6.HOST_CUT_MS, C6.F240_MS


def registration(repo: Path, part: str) -> dict:
    return json.loads((Path(repo) / "docs" / "registered" / f"{NAMES[part]}.json").read_text())


def tile_class(bm: int, G: int) -> str:
    return f"{bm}g{G}"


# --------------------------------------------------------------------------
# counter pages: c and F of a tile
# --------------------------------------------------------------------------

def _kn(cfg, gemm: str) -> int:
    return cfg.hidden_size if gemm == "w1" else cfg.intermediate_size


def floor_rows(page: dict, ns=FLOOR_NS, max_frac: float | None = FLOOR_MAX_DRAM_FRAC) -> list[dict]:
    """The floor-bound NATIVE and SHARED GEMM cells of one counter page: (series, q, S, cycles)
    with q = ceil(N_live / SMs), N_live from the recorded grid (r4common.page_c's reading), at the
    page's own BLOCK_M and BLOCK_K. A cell enters when its GEMM reads DRAM at no more than
    max_frac of the 4022 GB/s peak (its own dram_bytes_read over its gpu_time_ns)."""
    from moe.spec import MODEL_CONFIGS
    de = page.get("design") or {}
    bm, bk = int(de.get("block_m") or 32), int(de.get("block_k") or 64)
    cfg = MODEL_CONFIGS[de["model"]]
    out = []
    for c in page.get("cells") or []:
        if c.get("arm") not in ("native", "shared") or int(c.get("n", 0)) not in ns:
            continue
        for g in ("w1", "w2"):
            r = (c.get("recorded") or {}).get(g) or {}
            pg = (c.get("per_gemm") or {}).get(g) or {}
            grid, cyc = pg.get("grid_size"), r.get("sm__cycles_elapsed.avg")
            if not grid or cyc is None:
                continue
            if max_frac is not None:
                b, t = pg.get("dram_bytes_read"), pg.get("gpu_time_ns")
                if not b or not t or float(b) / float(t) / DRAM_PEAK_GBPS > max_frac:
                    continue
            D = int(c["declared"])
            R = math.ceil((cfg.num_experts * int(c["n"]) * bm + D * (bm - 1)) / bm)
            npn = int(grid) // R
            live = cfg.num_experts * int(c["n"]) * npn
            occ = [v for kk, v in r.items() if kk.startswith("launch__occupancy_limit")]
            out.append({"series": f"{de['model']}|{g}|{c['arm']}", "gemm": g, "q": math.ceil(live / SMS),
                        "S": _kn(cfg, g) // bk, "cycles": float(cyc), "n": int(c["n"]), "arm": c["arm"],
                        "occ": int(min(occ)) if occ else None, "model": de["model"]})
    return out


def joint_cF(pages: list[dict], ns=FLOOR_NS, max_frac: float | None = FLOOR_MAX_DRAM_FRAC) -> dict | None:
    """c (one for both GEMMs) and F_w1, F_w2 by least squares over every page's floor-bound
    cells, one intercept per (page model, GEMM, arm) series: cycles = a_series + q (S c + F_g).
    None with fewer than 6 cells or rank deficiency. se from the residual variance."""
    import numpy as np
    rows = [r for p in pages for r in floor_rows(p, ns, max_frac)]
    if len(rows) < 6:
        return None
    series = sorted({r["series"] for r in rows})
    cols = series + ["c", "F_w1", "F_w2"]
    X = np.zeros((len(rows), len(cols)))
    y = np.array([r["cycles"] for r in rows])
    for i, r in enumerate(rows):
        X[i, series.index(r["series"])] = 1.0
        X[i, len(series)] = r["q"] * r["S"]
        X[i, len(series) + (1 if r["gemm"] == "w1" else 2)] = r["q"]
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    dof = max(len(rows) - X.shape[1], 1)
    cov = np.linalg.inv(X.T @ X) * float(res @ res) / dof
    se = np.sqrt(np.diag(cov))
    k = len(series)
    pred = X @ coef
    return {"c": float(coef[k]), "F": {"w1": float(coef[k + 1]), "w2": float(coef[k + 2])},
            "se": {"c": float(se[k]), "F_w1": float(se[k + 1]), "F_w2": float(se[k + 2])},
            "cells": len(rows), "series": len(series), "rel_rms": float(np.sqrt(np.mean((res / pred) ** 2))),
            "occupancy_recorded": sorted({r["occ"] for r in rows if r["occ"] is not None}),
            "models": sorted({r["model"] for r in rows})}


def u_measured(fit: dict, model: str, gemm: str, bk: int) -> float:
    from moe.spec import MODEL_CONFIGS
    S = _kn(MODEL_CONFIGS[model], gemm) // bk
    return S * fit["c"] + fit["F"][gemm]


def floor_test(pred_c: float, pred_F: dict, meas: dict, models, bk: int = 64, tol: float = FLOOR_TOL) -> dict:
    """Per (model, GEMM): e = u_pred / u_meas - 1 with u = S c + F (u_meas from the tile's joint fit
    over its pages); HOLDS iff every |e| <= tol."""
    from moe.spec import MODEL_CONFIGS
    rows = []
    for m in models:
        for g in ("w1", "w2"):
            S = _kn(MODEL_CONFIGS[m], g) // bk
            up, um = S * pred_c + pred_F[g], u_measured(meas, m, g, bk)
            rows.append({"model": m, "gemm": g, "S": S, "u_pred": r6(up), "u_meas": r6(um), "e": r6(up / um - 1)})
    ok = bool(rows) and all(abs(r["e"]) <= tol for r in rows)
    return {"verdict": "HOLDS" if ok else "FAILS", "rows": rows, "tol": tol}


def variant_rule(tests: dict) -> dict:
    """tests {variant: floor_test result or None (unreachable)}: SELECTED X iff X HOLDS and every
    other variant FAILS or is unreachable; else UNDECIDED (rental B takes v4cal at 64x128)."""
    holds = [v for v, t in tests.items() if t is not None and t["verdict"] == "HOLDS"]
    if len(holds) == 1:
        return {"verdict": f"SELECTED {holds[0]}", "selected": holds[0], "v4_at_64x128": f"derived ({holds[0]})"}
    return {"verdict": f"UNDECIDED ({len(holds)} accountings hold)", "selected": None,
            "v4_at_64x128": "v4cal: the 64x128 c and F measured on the CAL pages (CAL-A)"}


def f_form_rule(meas64: dict, phi_wg: dict, F_wg: dict) -> dict:
    """F-step (v3's form) predicts F_64,g = PHI_wg,g x c_64 (measured c); F-cyc predicts F_wg,g.
    F-cyc is selected iff on BOTH GEMMs it is nearer the measured F_64,g AND F-step misses by more
    than FCYC_K_SE se on at least one; else F-step."""
    rows, nearer, far = {}, 0, 0
    for g in ("w1", "w2"):
        Fm, se = meas64["F"][g], meas64["se"][f"F_{g}"]
        fs, fc = phi_wg[g] * meas64["c"], F_wg[g]
        rows[g] = {"F_meas": r6(Fm), "se": r6(se), "F_step": r6(fs), "F_cyc": r6(fc)}
        nearer += abs(fc - Fm) < abs(fs - Fm)
        far += abs(fs - Fm) > FCYC_K_SE * se
    sel = "F-cyc" if (nearer == 2 and far >= 1) else "F-step"
    return {"selected": sel, "rows": rows, "rule": "F-cyc iff nearer on both GEMMs and F-step off by > 3 se on one"}


# --------------------------------------------------------------------------
# the A key
# --------------------------------------------------------------------------

def named_key(rows: list[dict], keys=KEYS) -> dict:
    """rows {cell, label, e: {key: e}} (every scored w1 row; a key with no e on a row is not
    priced there and cannot be NAMED): the nearest key by rms is NAMED iff the next is at least
    NAMED_FACTOR farther and every row of the nearest is within NAMED_MAX_ABS; else FALLBACK_KEY."""
    if not rows:
        return {"verdict": f"NOT SCORED: no w1 row; v4 takes {FALLBACK_KEY}", "key": FALLBACK_KEY}
    priced = [k for k in keys if all(r["e"].get(k) is not None for r in rows)]
    if len(priced) < 2:
        return {"verdict": f"NOT NAMED (fewer than two keys priced on every row); v4 takes {FALLBACK_KEY}",
                "key": FALLBACK_KEY}
    rms = {k: math.sqrt(st.fmean(r["e"][k] ** 2 for r in rows)) for k in priced}
    order = sorted(rms, key=rms.get)
    near, nxt = order[0], order[1]
    worst = max(abs(r["e"][near]) for r in rows)
    named = rms[nxt] >= NAMED_FACTOR * rms[near] and worst <= NAMED_MAX_ABS
    by_label = {}
    for lab in sorted({r["label"] for r in rows}):
        by_label[lab] = {k: C5.stats(r["e"][k] for r in rows if r["label"] == lab) for k in priced}
    return {"verdict": f"NAMED {near}" if named else f"NOT NAMED (nearest {near}); v4 takes {FALLBACK_KEY}",
            "key": near if named else FALLBACK_KEY, "rows": len(rows),
            "rms_pct": {k: r6(100 * v) for k, v in rms.items()}, "nearest": near, "next": nxt,
            "next_over_nearest": r6(rms[nxt] / rms[near]) if rms[near] else None,
            "worst_nearest_pct": r6(100 * worst), "by_label": by_label}


# --------------------------------------------------------------------------
# time: the E1 bar, the class means, the stop rule, E1-TILE, the rivals
# --------------------------------------------------------------------------

def bar(v: list[float], rule: dict = E1_BAR) -> dict:
    s = C5.stats(v)
    if not s["cells"]:
        return {"verdict": "NOT SCORED: no valid cell", "stats": s}
    ok = s["rms"] <= rule["rms_max"] and abs(s["worst"]) <= rule["max_abs"] and abs(s["mean"]) <= rule["mean_max"]
    return {"verdict": "HOLDS" if ok else "FAILS", "stats": s}


def class_means(rows: list[dict]) -> dict:
    """rows {cls, model, e} (NATIVE E1 cells): per tile class the mean e, and per model in it."""
    out = {}
    for c in CLASSES:
        v = [r["e"] for r in rows if r["cls"] == c]
        out[c] = {"cells": len(v), "mean": r6(st.fmean(v)) if v else None,
                  "by_model": {m: r6(st.fmean(r["e"] for r in rows if r["cls"] == c and r["model"] == m))
                               for m in sorted({r["model"] for r in rows if r["cls"] == c})}}
    return out


def stop_rule(e1cal: dict, census: dict | None = None) -> dict:
    """D6 and design-tile-review 4 and 8, on tA-e1cal's score: {pages: {key: state}, rows: [...],
    bar: verdict}, and rental A's census score. STOP when any clause fails; every clause is printed.
    A census that refused (an MMA path mismatch) or is missing is a STOP (build-tile-review 9)."""
    why = []
    cv = str(((census or {}).get("CENSUS") or {}).get("verdict", "NOT SCORED: no census score"))
    if not cv.startswith("VALID"):
        why.append(f"the census reads {cv}: every v4 law is per MMA path (a STOP, never a fallback)")
    pages = e1cal.get("pages") or {}
    for k, v in sorted(pages.items()):
        if v != "counted":
            why.append(f"E1-CAL page {k} {v} (a SKIPPED, LOST or NOT SCORED E1-CAL page is a STOP)")
    cm = class_means(e1cal.get("rows") or [])
    for c, x in cm.items():
        if x["mean"] is None:
            why.append(f"class {c}: no valid NATIVE E1-CAL cell")
        elif abs(x["mean"]) > STOP_CLASS_MEAN:
            why.append(f"class {c}: mean residual {100 * x['mean']:+.3f}% beyond {100 * STOP_CLASS_MEAN:.2f}%")
    bm = cm["128g1"]["by_model"]
    if len(bm) >= 2:
        spread = max(bm.values()) - min(bm.values())
        if spread > STOP_PAIR:
            why.append(f"128 G 1: the CAL models' means differ by {100 * spread:.3f}% (over {100 * STOP_PAIR:.0f}%)")
    else:
        why.append(f"128 G 1: {len(bm)} CAL model(s) with valid cells, the model-specific check needs 2")
    b = (e1cal.get("bar") or {}).get("verdict")
    if b != "HOLDS":
        why.append(f"tA-e1cal's pooled bar reads {b}")
    if e1cal.get("g3_void"):
        why.append("G3 voided the E1-CAL cells")
    return {"verdict": "STOP: rental B does not run" if why else "GO: rental B may run", "why": why,
            "class_means": cm}


def e1tile_rule(rows: list[dict], pages_in_view: dict, view: str, blind_models, descriptive_models=()) -> dict:
    """rows {model, e}: the pooled bar over every row, and each BLIND-TILE model's; a model in
    `descriptive_models` is in the pooled bar and its own bar is printed only. CLEAN: a model is
    CERTIFIED with at least MIN_CLEAN_PAGES_PER_MODEL of its pages in the view."""
    pooled = bar([r["e"] for r in rows])
    per, ok = {}, pooled["verdict"] == "HOLDS"
    lost = [m for m in blind_models if not any(r["model"] == m for r in rows)]
    for m in list(blind_models) + list(descriptive_models):
        v = bar([r["e"] for r in rows if r["model"] == m])
        rec = {"verdict": v["verdict"], "stats": v["stats"], "pages_in_view": pages_in_view.get(m, [])}
        if m in descriptive_models:
            rec["label"] = "DESCRIPTIVE (one page; new model, seen shapes: never CERTIFIED)"
        else:
            ok = ok and v["verdict"] == "HOLDS"
            if view == "CLEAN":
                n = len(pages_in_view.get(m, []))
                rec["certification"] = ("CERTIFIED" if (v["verdict"] == "HOLDS" and n >= MIN_CLEAN_PAGES_PER_MODEL)
                                        else f"NOT CERTIFIED ({n} pages in CLEAN, bar {v['verdict']})")
        per[m] = rec
    if lost:
        # build-tile-review 9: a BLIND-TILE model with no valid cell leaves the test incomplete, not failed
        verdict = f"NOT SCORED (incomplete: no valid cell of {', '.join(lost)})"
        if pooled["verdict"] == "FAILS" or any(per[m]["verdict"] == "FAILS" for m in blind_models if m not in lost):
            verdict = "FAILS (and incomplete: no valid cell of " + ", ".join(lost) + ")"
    else:
        verdict = "HOLDS" if ok else ("FAILS" if pooled["verdict"] != "NOT SCORED: no valid cell" else pooled["verdict"])
    return {"verdict": verdict, "pooled": pooled, "per_model": per, "models_without_a_valid_cell": lost}


def rival_rule(rows: list[dict], R: str, *, sigma: float) -> dict:
    """rows {e (v4), pred: {R: T_R}, pred_v4, meas}: lever cells |T_R / T_v4 - 1| >= LEVER_FRAC;
    EXCLUDED iff at least LEVER_MIN lever cells and R's rms there over RIVAL_RMS_MIN and over v4's
    rms there + RIVAL_K_SE se (se = sigma / sqrt(cells)); UNDECIDED under LEVER_MIN."""
    lev = [r for r in rows if r["pred"].get(R) is not None and abs(r["pred"][R] / r["pred_v4"] - 1) >= LEVER_FRAC]
    if len(lev) < LEVER_MIN:
        return {"verdict": f"NOT SEPARATED: {len(lev)} lever cells, under {LEVER_MIN}", "lever_cells": len(lev)}
    eR = [r["pred"][R] / r["meas"] - 1 for r in lev]
    ev = [r["e"] for r in lev]
    rr, rv = math.sqrt(st.fmean(x * x for x in eR)), math.sqrt(st.fmean(x * x for x in ev))
    se = sigma / math.sqrt(len(lev))
    ex = rr > RIVAL_RMS_MIN and rr > rv + RIVAL_K_SE * se
    return {"verdict": "EXCLUDED" if ex else "NOT EXCLUDED", "lever_cells": len(lev), "rms_rival": r6(rr),
            "rms_v4_on_levers": r6(rv), "se": r6(se)}


def lever_count(preds_v4: list[float], preds_R: list[float]) -> int:
    return sum(1 for a, b in zip(preds_v4, preds_R, strict=True) if b is not None and abs(b / a - 1) >= LEVER_FRAC)


# --------------------------------------------------------------------------
# tA-dead and tA-rep
# --------------------------------------------------------------------------

def dead_rule(delta_us: float, se_us: float, laws_us: dict, c128: float | None) -> dict:
    """delta = (S - N at c15) - (S - N at c9), measured, us, minus nothing (the align growth is
    subtracted in the scorer from the pages' own probes). D against SLOT: each law's z =
    (delta - law) / se; a law is EXCLUDED iff |z| > DEAD_Z. The kappa rivals are scored only
    when c128 >= KAPPA_C128_MIN (below it they predict the same increment)."""
    out = {"delta_us": r6(delta_us), "se_us": r6(se_us), "laws": {}}
    for name, v in laws_us.items():
        if name.startswith("kappa") and (c128 is None or c128 < KAPPA_C128_MIN):
            out["laws"][name] = {"pred_us": r6(v), "verdict": f"NOT SCORED (c_128 {c128} under {KAPPA_C128_MIN:g})"}
            continue
        z = (delta_us - v) / se_us if se_us > 0 else float("inf")
        out["laws"][name] = {"pred_us": r6(v), "z": r6(z), "verdict": "EXCLUDED" if abs(z) > DEAD_Z else "NOT EXCLUDED"}
    d, s = out["laws"].get("D", {}), out["laws"].get("SLOT", {})
    out["D_vs_SLOT"] = ("D HOLDS, SLOT EXCLUDED" if d.get("verdict") == "NOT EXCLUDED" and s.get("verdict") == "EXCLUDED"
                        else "SLOT HOLDS, D EXCLUDED" if s.get("verdict") == "NOT EXCLUDED" and d.get("verdict") == "EXCLUDED"
                        else "NOT SEPARATED by this page")
    return out


def rep_sigma(pairs: list[tuple[float, float]]) -> dict:
    """sigma_page from a replicate pair of pages: the rms over shared cells of (t2/t1 - 1) / sqrt 2."""
    if len(pairs) < 3:
        return {"verdict": f"NOT SCORED: {len(pairs)} shared cells", "sigma_page": None}
    d = [b / a - 1 for a, b in pairs]
    s = math.sqrt(st.fmean(x * x for x in d)) / math.sqrt(2)
    return {"verdict": "MEASURED", "sigma_page": r6(s), "cells": len(d), "mean_shift": r6(st.fmean(d))}


def sigma_page_of(rep_score: dict | None) -> tuple[float, str]:
    """The sigma_page rental B and tA-dead use: tA-rep's when measured, else rental 3's 0.18%
    (design-tile-review 9.8)."""
    s = ((rep_score or {}).get("REP") or {}).get("sigma_page")
    if s:
        return float(s), "tA-rep (the 128x128 replicate pair)"
    return SIGMA_PAGE_R3, "rental 3's rep8 0.18% (tA-rep dropped, lost or not scored)"


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------

def timed_page(tree: Path, model: str, label: str, G: int, tview, reg_file: dict | None, view: str = "registered"):
    """rental 7's page_report at the unit's own G (1 or 16) instead of 8: G5 provenance, the
    r7-median-run3 rebuild from cells.csv, the view, and the page's recorded tile (BLOCK_M and
    num_warps, design-tile-review 6) against the registered one."""
    root = C3.find_gaps(tree, model, label)
    pages = [(p, r) for p, r in C3.timed_pages(root) if C3.group_of(r) == G]
    if len(pages) != 1:
        return None, f"NOT SCORED: {len(pages)} G = {G} pages under gaps-{model}-{label} (no report.json is NOT SCORED)"
    p, r = pages[0]
    if reg_file is not None and reg_file.get("tile"):
        pin = r.get("pinned") or {}
        t = reg_file["tile"]
        got = [int(r.get("block_m") or pin.get("BLOCK_SIZE_M") or 0), int(pin.get("BLOCK_SIZE_N") or 0),
               int(pin.get("BLOCK_SIZE_K") or 0), int(pin.get("num_warps") or 0), int(pin.get("num_stages") or 0)]
        if got != list(t):
            return None, f"NOT SCORED: the page ran tile {got}, the registration's is {t}"
    h = r.get("histogram") or {}
    if reg_file is None or h.get("file_sha256") != reg_file["sha256"]:
        return None, "NOT SCORED (G5): the page's histogram file sha256 is not the registered one"
    if h.get("shuffle_seed") is None or h.get("shuffle_seed") != reg_file.get("shuffle_seed"):
        return None, "NOT SCORED (G5): the page records no shuffle seed, or not the registered one"
    want = {(c["label"], c["n"]): c["counts_sha256"] for c in reg_file["cells"]}
    for row in r.get("treads_table") or []:
        k = (str(row.get("histogram")), int(row["tiles"]))
        if want.get(k) != row.get("counts_sha256") or not row.get("bincount_ok"):
            return None, f"NOT SCORED (G5): row {k} carries another counts sha256 or no exact bincount"
    if r.get("slip_policy") != "cell" or r.get("min_clean_repeats") != 0:
        return None, "NOT SCORED: the page did not run --slip-policy cell --min-clean-repeats 0"
    _rep, raw = C7.read_page(Path(p).parent)
    r, why = C7.rebuild(r, raw)
    if r is None:
        return None, f"NOT SCORED (rebuild): {why}"
    g = r["g1_r7"]
    state = g["verdict"]
    why = f"{g['non_clean_rows']} of {g['repeat_rows']} repeat rows non-clean ({100 * g['share']:.2f}%)"
    if state != "PASS":
        why = f"G1 FAIL (rebuilt): {why}"
    if view == "p20":
        state = r["g1_p20"]["verdict"]
        r = C7.p20_report(r)
    key = f"{model}-{label}"
    r = tview.admit(key, r)
    if r is None:
        return None, f"out of this view (a recorded VALIDITY gate failed; {why})"
    tview.pages[key]["g1_r7"] = why
    if state != "PASS" and tview.name == "CLEAN":
        tview.pages[key]["use"] = "excluded (G1: " + why + ")"
        return None, f"out of CLEAN (G1: {why})"
    return r, ("counted" if state == "PASS" else f"counted in ALL only (G1: {why})")


def counter_page(tree: Path, model: str, label: str, G: int, reg_file: dict | None = None, tile: list | None = None):
    """(page, path, why) for a unit's one G counter page (lock1710/r3c-g<G>.json), with G5 on a
    histogram page and the page's recorded tile (block_m, block_n, block_k, num_warps,
    num_stages) against the registered one."""
    hits = sorted(Path(tree).rglob(f"*-{CARD}-{model}-{label}-r3-counters/lock1710/r3c-g{G}.json"))
    if len(hits) != 1:
        return None, None, f"NOT SCORED: {len(hits)} counter pages for {model} {label} G {G}"
    p = json.loads(hits[0].read_text())
    de = p.get("design") or {}
    if tile is not None:
        got = [int(de.get(k) or 0) for k in ("block_m", "block_n", "block_k", "num_warps", "num_stages")]
        if got != list(tile):
            return None, None, f"NOT SCORED: the page ran tile {got}, the registration's is {list(tile)}"
    if reg_file is not None:
        h = de.get("histogram") or {}
        if h.get("file_sha256") != reg_file["sha256"] or h.get("shuffle_seed") != reg_file.get("shuffle_seed"):
            return None, None, "NOT SCORED (G5): the page's histogram sha256 or shuffle seed is not the registered one"
        want = {(c["label"], c["n"]): c["counts_sha256"] for c in reg_file["cells"]}
        bad = [c for c in p.get("cells") or [] if want.get((str(c.get("histogram")), int(c["n"]))) != c.get("counts_sha256")]
        if bad:
            return None, None, "NOT SCORED (G5): a cell carries another counts sha256"
    return p, hits[0], "counted"


def c_with_F_held(pages: list[dict], F: dict, ns=FLOOR_NS, max_frac: float | None = FLOOR_MAX_DRAM_FRAC) -> dict | None:
    """c of a one-model page (where c and two F's are not separable): cycles - q F_g = a_series +
    q S_g c, F held (rental 4's page_c, both GEMMs at once). None under 3 cells."""
    import numpy as np
    rows = [r for p in pages for r in floor_rows(p, ns, max_frac)]
    if len(rows) < 3:
        return None
    series = sorted({r["series"] for r in rows})
    X = np.zeros((len(rows), len(series) + 1))
    y = np.array([r["cycles"] - r["q"] * F[r["gemm"]] for r in rows])
    for i, r in enumerate(rows):
        X[i, series.index(r["series"])] = 1.0
        X[i, -1] = r["q"] * r["S"]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"c": float(coef[-1]), "F_held": dict(F), "cells": len(rows),
            "occupancy_recorded": sorted({r["occ"] for r in rows if r["occ"] is not None})}


def row_drop(row: dict, pred_ms: float | None) -> str | None:
    """rental 6's row_verdict (6 of 9 clean, the host rule on the prediction AND the timer flag, G4)."""
    return C6.row_verdict(row, pred_ms)


def valid_cells(report: dict, preds: dict, model: str, page: str) -> tuple[dict, list, dict | None]:
    """({(model, page, n, label, arm): ms}, dropped, the page's A) of a timed page's valid rows;
    preds {(model, page, n, label, arm): {"ms": v4 ms, ...}}."""
    out, dropped = {}, []
    A = C6.page_A_us(report)
    for row in report.get("treads_table") or []:
        k = (model, page, int(row["tiles"]), str(row["histogram"]), str(row["arm"]))
        pr = preds.get(k)
        why = "no registered prediction" if pr is None or pr.get("ms") is None else row_drop(row, pr["ms"])
        if why:
            dropped.append(["/".join(map(str, k)), why])
        else:
            out[k] = float(row["ms_p50"])
    return out, dropped, A


def carried(pr: dict, A: dict | None) -> float:
    """v4's prediction carried to the page's own A: term (b) is additive in A, so only block-scan
    SHARED cells move (r6model.reprice's v3 rule)."""
    import r6model as M6
    a_page = None if A is None else A.get((pr["arm"], pr["n"]))
    return M6.reprice(pr["ms"], pr["A_proxy_us"], a_page, pr["arm"], pr["native_path"], "v3")


def g3_rows(meas: dict, preds: dict, cluster_of) -> list[dict]:
    """rental 7's G3 rows: d = T_u / T_b - 1 - (pred_u / pred_b - 1) per (page, n, arm) with both cells
    valid; cluster = the page, stratum = the tread's rank on its page."""
    out = []
    for k, pr in preds.items():
        if k[3] != "uniform":
            continue
        b = k[:3] + ("balanced", k[4])
        if k in meas and b in meas and b in preds:
            delta = pr["ms"] / preds[b]["ms"] - 1
            ns = sorted({x[2] for x in preds if x[:2] == k[:2]})
            out.append({"cluster": f"{k[0]}-{k[1]}", "stratum": ns.index(k[2]), "d": meas[k] / meas[b] - 1 - delta,
                        "cell": "/".join(map(str, k))})
    return out


# --------------------------------------------------------------------------
# occupancy: the census's, never a typed-in number (build-tile-review 4a, ambiguity 11)
# --------------------------------------------------------------------------

#: rental A's scored census (score_census.py's census.score.json), when one exists; set by the
#: environment MOE_TILE_CENSUS_SCORE or `use_census`. With none, the occupancy is the shared-memory
#: rule's (r3_timing_model_v4.Tile.occ_shared, derived from the tile, labelled DESIGN), which the
#: census then replaces by this same code path (register.py --census-score, rowkeys_tile.py
#: --census-score): the rebuild is mechanical.
_CENSUS: list = []


class CensusRefused(Exception):
    pass


def use_census(path) -> None:
    _CENSUS[:] = [None if path is None else json.loads(Path(path).read_text())]


def census_score() -> dict | None:
    if not _CENSUS:
        import os
        p = os.environ.get("MOE_TILE_CENSUS_SCORE")
        _CENSUS.append(json.loads(Path(p).read_text()) if p else None)
    return _CENSUS[0]


def occupancy_of(tile, model: str | None = None) -> tuple[dict, str]:
    """({w1, w2} CTAs per SM of `tile` for `model`, source). A census that refused (an MMA path
    mismatch) raises CensusRefused: it is a STOP, never a fallback."""
    cs = census_score()
    if cs is not None:
        if str((cs.get("CENSUS") or {}).get("verdict", "")).startswith("REFUSED"):
            raise CensusRefused(f"the census refused: {cs['CENSUS']['verdict']}")
        by = ((cs.get("occupancy_by_model") or {}).get(model) or {}).get(tile.key) if model else None
        if by and all(by.get(g) for g in ("w1", "w2")):
            return {g: int(by[g]) for g in ("w1", "w2")}, "census (this model)"
        com = (cs.get("occupancy") or {}).get(tile.key)
        if com:
            return {g: int(com[g]) for g in ("w1", "w2")}, "census (every model)"
    o = tile.occ_shared()
    return {"w1": o, "w2": o}, "DESIGN (shared-memory rule; the census replaces it)"
