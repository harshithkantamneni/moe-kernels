"""Tile rental B3's pure rules (docs/registered/2026-11-04-tileB3-gh200): one module for the VM gates
(tileB3_gate.py), the reader and scorer (read_b3.py) and the registration (register_b3.py), so a rule is computed one
way on every side. It loads no frozen chain and no GPU library: numpy, the repo's census rules and, for the 64x128
clock clause, A2's scoring rules (a2common, score_floor_a2, r6common's clock rule). Every range clause is written to
fail on a NaN (`not lo <= x <= hi`, `not x <= max`, math.isfinite).

  bars        the working-model bar (owner, 2026-11-04) for every NEW blind test: rms <= 3%, every cell within 8%,
              |mean| <= 1.5%, per blind model and pooled, registered before any page. B2's E1-TILE bar (2%, 5%, 1%)
              is printed beside, DESCRIPTIVE, never a verdict.
  the sag     the on-board F1 estimator: design-v5 work/sag.py's pts_of and fit, copied verbatim (tau on a 50 to 4975 us
              grid step 25, A least squares, A rounded to 0.01 us), one fit per CAL estimator page (OLMoE f128 and
              Mixtral-8x7B f128, 128x128 G 64, lock 1410) and "par", the two fits' A and tau averaged (refit4 r4.par).
              The registration holds this copy to the frozen chain's r4.sag_page_fit bit for bit on A2's, C2's and C4's
              pages, and the reader recomputes it with the frozen chain and refuses on any difference.
  sag gate    the registered sag-fit range: a STOP before any timed unit (tileB3_gate.py sag).
  64x128 sag  the same estimator on the board's OLMoE f64w8 and Mixtral-8x7B f64w8 pages (64x128x64 w8 s3, G 64), par
              of the two fits (knee-v5's 'par' on A2: 30.37 us / 300 us); the 64x128 sag-fit range is a STOP before any
              timed unit (tileB3_gate.py sag64). The owner's choice: measured on B3's board, not CARRIED. Its clock
              clause (CLOCK64): each f64w8 page passes r6common.clock_rule (lock_gate.check_page PASS) under A2's
              scoring rules in the band in force (score_floor_a2.choose_band on the f128 pages, as G-BAND), and at
              least CLOCK64["cells_in_band_min"] of its cells sit in that band, the rule B3's byte pages are read under.
  par check   PAR_RTOL: the reader compares the gate record's par (computed on the GH200) with its own (the laptop) to a
              relative 1e-9, not exactly: the same estimator on the same page bytes, so the tolerance absorbs float
              summation order only.
  census gate B3's compile-only census of the blind models (label tb) equals A2's merged census on every compiled fact
              (tile_census_a2.FIELDS) at both tiles, with no MMA path mismatch: a STOP before any timed unit
              (tileB3_gate.py census). deepseek-v2-lite (SEEN-SHAPE, pooled only) drifting is no STOP: its page is NOT
              SCORED.
  power       read_c2.power (diag-v5 power5's, verbatim) with the bar as an argument: at B2's bar it equals
              read_c2.power bit for bit (asserted in the tests), at the working-model bar it is B3's estimate.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

NAME = "2026-11-04-tileB3-gh200"
PLAN = "scripts/plans/rentalB3-tile-2026-11.plan"
DRIVER = "scripts/gh200_tileB3_session.sh"
B2_NAME = "2026-10-20-tileB2-e1tile-gh200"
A2_MERGED_CENSUS = ("scripts/scoring/tileA2/census-merged/2026-10-20-nvidia_gh200_480gb-tile-census-merged-ta2m/"
                    "tile-census.json")
CARD = "nvidia_gh200_480gb"
LOCK = 1410
POWER_W = 700

#: the working-model bar (owner, 2026-11-04), fractions
BAR = {"rms_max": 0.03, "max_abs": 0.08, "mean_max": 0.015}
#: B2's registered E1-TILE bar (tileB2-e1tile), printed beside, DESCRIPTIVE
B2_BAR = {"rms_max": 0.02, "max_abs": 0.05, "mean_max": 0.01}
BLIND = ("phi-3.5-moe", "qwen1.5-moe-a2.7b", "jetmoe-8b")
DSV = "deepseek-v2-lite"
MODELS = BLIND + (DSV,)
MIN_CLEAN_PAGES = 2

#: the sag estimator's pages (the plan's labels) and the f128 page glob, r4.sag_page_fit's
EST_MODELS = ("olmoe-1b-7b", "mixtral-8x7b")
EST_UNIT = "f128"
#: the sag gate's registered range (a STOP before any timed unit when any clause fails)
SAG_RANGE = {
    "points_min": 48,              # of the 54 (27 cells x 2 GEMMs) each f128 page carries; every SEEN page has 54
    "A_us": (15.0, 120.0),         # per page; SEEN 29.65 to 67.55 (A2, C2, C4)
    "tau_us": (100.0, 2000.0),     # per page, off the fit grid's edges (50, 4975); SEEN 300 to 650
    "rms_us_max": 20.0,            # per page; SEEN 4.02 to 13.01 (Mixtral's pages 10.25 to 13.01)
    "ratio_1ms": (1.0, 2.5),       # Mixtral's sag at 1 ms over OLMoE's; SEEN 1.52 to 1.59
    "par_A_us": (25.0, 100.0),     # design-v5's A_128 range; SEEN par 41.59 to 54.25
}
#: the pars the frozen model's held-out means were read at (A2, C2, C4): DESCRIPTIVE, printed beside the gate
HELD_OUT_PAR_SPAN = {"A_us": (41.59, 54.25), "tau_us": (412.5, 587.5)}
#: the 64x128 sag estimator's pages (A2's f64w8 units) and their registered range (a STOP before any timed unit)
SAG64_UNIT = "f64w8"
SAG64_RANGE = {
    "points_min": 48,              # of the 54 each f64w8 page carries; A2's two pages have 54
    "A_us": (10.0, 80.0),          # per page; SEEN (A2 only) 24.79 (OLMoE), 35.94 (Mixtral-8x7B)
    "tau_us": (100.0, 2000.0),     # per page, off the fit grid's edges (50, 4975); SEEN 275, 325
    "rms_us_max": 20.0,            # per page; SEEN 3.32, 7.62
    "ratio_1ms": (1.0, 2.5),       # Mixtral's sag at 1 ms over OLMoE's; SEEN 1.42
    "par_A_us": (15.0, 70.0),      # SEEN par 30.37 (A2)
}
#: the 64x128 pages' clock clause: r6common.clock_rule in the band in force; SEEN (A2) 26 and 22 of 27 cells in band
#: (the f128 pages on A2, C2, C4: 17 to 24), so more than half the cells must be in band
CLOCK64 = {"cells_in_band_min": 14, "check_page": "PASS"}
#: the registered tolerance of the GH200-against-laptop par comparison (relative, per component)
PAR_RTOL = 1e-9
#: the 64x128 sag round 5's PART D priced Phi 64g1 at: A2's f64w8 par (knee-v5 KNEE.md 'par'), CARRIED there
SAG64_A2 = (30.37, 300.0)

CENSUS_LABEL = "tb"
CENSUS_TILES = ("128x128x64w8s3", "64x128x64w8s3")

PASS_RC, FAIL_RC, REFUSED_RC = 0, 3, 2


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dumps(d) -> str:
    return json.dumps(d, indent=1, sort_keys=True, default=str) + "\n"


# ================================================================== the on-board sag (design-v5 sag.py, verbatim)

def pts_of(files, lock):
    out = []
    for f in files:
        p = json.loads(Path(f).read_text())
        for c in p["cells"]:
            for g in ("w1", "w2"):
                r = (c.get("recorded") or {}).get(g) or {}
                pg = dict((c.get("per_gemm") or {}).get(g) or {})
                if "sm__cycles_elapsed.avg" in pg:
                    cyc, t = pg["sm__cycles_elapsed.avg"], pg["gpu__time_duration.sum"]
                else:
                    cyc, t = r.get("sm__cycles_elapsed.avg"), pg.get("gpu_time_ns")
                if cyc and t:
                    t = float(t) / 1e3
                    out.append((t, t - float(cyc) / lock))
    return out


def fit(p):
    t = np.array([x[0] for x in p])
    off = np.array([x[1] for x in p])
    best = None
    for tau in np.arange(50, 5000, 25):
        X = 1 - np.exp(-t / tau)
        A = float(X @ off / (X @ X))
        s = float(np.sqrt(np.mean((off - A * X) ** 2)))
        if best is None or s < best[2]:
            best = (A, float(tau), s)
    # linear t = a + cycles / f form, the CONV-type reading: slope of off against t
    Xl = np.vstack([np.ones_like(t), t]).T
    a, b = np.linalg.lstsq(Xl, off, rcond=None)[0]
    return {"cells": len(p), "A_us": round(best[0], 2), "tau_us": best[1], "rms_us": round(best[2], 2),
            "lin_a_us": round(float(a), 2), "lin_slope": round(float(b), 5),
            "t_range_us": [round(float(t.min()), 1), round(float(t.max()), 1)],
            "A_over_tau": round(best[0] / best[1], 4)}


def page_files(results: Path, model: str, unit: str = EST_UNIT, lock: int = LOCK) -> list:
    """r4.sag_page_fit's glob: the unit's counter page at the lock."""
    return sorted(Path(results).glob(f"*-{model}-{unit}-r3-counters/lock{lock}/r3c-g*.json"))


def page_fit(results: Path, model: str, unit: str = EST_UNIT) -> dict:
    """The estimator's fit of one f128 (or f64w8) page (r4.sag_page_fit: exactly one page, sag.fit over sag.pts_of at
    1410)."""
    f = page_files(results, model, unit)
    if len(f) != 1:
        raise LookupError(f"{len(f)} {model} {unit} pages under {results}")
    r = fit(pts_of(f, float(LOCK)))
    return {**r, "file": str(f[0]), "sha256": sha256(f[0])}


def par(fits) -> tuple:
    """refit4 r4.par: A and tau averaged over the pages' (A, tau)."""
    return (st.fmean(A for A, _ in fits), st.fmean(t for _, t in fits))


def sat_us(A: float, tau: float, t_us: float) -> float:
    return A * (1 - math.exp(-t_us / tau))


def sag_check(per: dict, R: dict = SAG_RANGE, unit: str = EST_UNIT) -> dict:
    """The registered sag-fit range on {model: fit}: (verdict, par, the clauses); the 64x128 range with R=SAG64_RANGE
    and unit=SAG64_UNIT (no held-out span there)."""
    why = []
    for m in EST_MODELS:
        f = per.get(m)
        if f is None:
            why.append(f"no {m} {unit} fit")
            continue
        if not f["cells"] >= R["points_min"]:
            why.append(f"{m}: {f['cells']} points, under {R['points_min']}")
        if not all(math.isfinite(float(f[k])) for k in ("A_us", "tau_us", "rms_us")):
            why.append(f"{m}: a non-finite fit (A {f['A_us']}, tau {f['tau_us']}, rms {f['rms_us']})")
        if not R["A_us"][0] <= f["A_us"] <= R["A_us"][1]:
            why.append(f"{m}: A {f['A_us']} us outside {list(R['A_us'])}")
        if not R["tau_us"][0] <= f["tau_us"] <= R["tau_us"][1]:
            why.append(f"{m}: tau {f['tau_us']} us outside {list(R['tau_us'])}")
        if not f["rms_us"] <= R["rms_us_max"]:
            why.append(f"{m}: rms {f['rms_us']} us over {R['rms_us_max']}")
    out = {"range": {k: (list(v) if isinstance(v, tuple) else v) for k, v in R.items()}}
    if why and any(per.get(m) is None for m in EST_MODELS):
        return {**out, "verdict": "FAIL", "why": "; ".join(why), "par": None, "ratio_1ms": None}
    o, m = per[EST_MODELS[0]], per[EST_MODELS[1]]
    so, sm = sat_us(o["A_us"], o["tau_us"], 1000.0), sat_us(m["A_us"], m["tau_us"], 1000.0)
    ratio = sm / so if (math.isfinite(so) and so > 0) else float("inf")
    if not R["ratio_1ms"][0] <= ratio <= R["ratio_1ms"][1]:
        why.append(f"sag at 1 ms, Mixtral over OLMoE, {ratio:.3f} outside {list(R['ratio_1ms'])}")
    p = par([(o["A_us"], o["tau_us"]), (m["A_us"], m["tau_us"])])
    if not R["par_A_us"][0] <= p[0] <= R["par_A_us"][1]:
        why.append(f"par A {p[0]:.3f} us outside {list(R['par_A_us'])}")
    if unit != EST_UNIT:
        return {**out, "verdict": "FAIL" if why else "PASS", "why": "; ".join(why), "par": [p[0], p[1]],
                "ratio_1ms": round(ratio, 4)}
    if not all(math.isfinite(x) for x in p):
        why.append(f"par {list(p)} not finite")
    span = HELD_OUT_PAR_SPAN
    inside = span["A_us"][0] <= p[0] <= span["A_us"][1] and span["tau_us"][0] <= p[1] <= span["tau_us"][1]
    return {**out, "verdict": "FAIL" if why else "PASS", "why": "; ".join(why), "par": [p[0], p[1]],
            "ratio_1ms": round(ratio, 4),
            "inside_held_out_par_span (DESCRIPTIVE)": inside, "held_out_par_span": {k: list(v) for k, v in span.items()}}


def sag_gate(results: Path, unit: str = EST_UNIT) -> dict:
    """tileB3_gate.py sag (f128) and sag64 (f64w8): the two estimator pages' fits, par and the registered range; any
    page missing is a FAIL."""
    R = SAG_RANGE if unit == EST_UNIT else SAG64_RANGE
    per, miss = {}, []
    for m in EST_MODELS:
        try:
            per[m] = page_fit(results, m, unit)
        except (LookupError, ValueError, KeyError, ZeroDivisionError) as e:
            miss.append(f"{m}: {e}")
    rec = sag_check(per, R, unit)
    if unit != EST_UNIT:
        try:
            rec["clock"] = clock64(results)
        except (LookupError, ValueError, KeyError, OSError) as e:
            rec["clock"] = {"verdict": "FAIL", "why": f"clock clause not read: {type(e).__name__} {e}"}
        if rec["clock"]["verdict"] != "PASS":
            rec["why"] = "; ".join(x for x in (rec.get("why"), rec["clock"]["why"]) if x)
            rec["verdict"] = "FAIL"
    if miss:
        rec["why"] = "; ".join(miss + ([rec["why"]] if rec.get("why") else []))
        rec["verdict"] = "FAIL"
    rec["gate"] = "sag" if unit == EST_UNIT else "sag64"
    rec["per_page"] = per
    return rec


def _a2_mods():
    for p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "tileA2")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import a2common as A2  # noqa: E402
    import score_floor_a2 as SFA  # noqa: E402
    return A2, SFA


def clock64(results: Path) -> dict:
    """The f64w8 pages' clock clause: the band in force (score_floor_a2.choose_band on the board's f128 pages, as
    G-BAND chose it), then r6common.clock_rule on each f64w8 page under A2's scoring rules at that band (as read_b3's
    byte legs read B3's byte pages): lock_gate.check_page PASS and at least CLOCK64 cells_in_band_min cells in band."""
    A2, SFA = _a2_mods()
    rules, rec = SFA.choose_band(Path(results), float(LOCK))
    out, why = {"band": [float(x) for x in rules.band], "band_which": rec.get("which"), "per_page": {}}, []
    if str(rec.get("which", "")).startswith("primary (neither"):
        why.append("no band in force (G-BAND reads neither band)")
    with A2.scoring(rules):
        for m in EST_MODELS:
            f = page_files(results, m, SAG64_UNIT)
            if len(f) != 1:
                why.append(f"{len(f)} {m} {SAG64_UNIT} pages")
                continue
            c = A2.TC.C6.clock_rule(json.loads(f[0].read_text()), f[0])
            n = int(c["cells_ok"])
            out["per_page"][m] = {"check_page": c["check_page"], "cells_in_band": n, "cells": len(c["cells"])}
            if c["check_page"] != CLOCK64["check_page"]:
                why.append(f"{m} {SAG64_UNIT}: lock_gate.check_page {c['check_page']}")
            if not n >= CLOCK64["cells_in_band_min"]:
                why.append(f"{m} {SAG64_UNIT}: {n} cells in the band in force, under {CLOCK64['cells_in_band_min']}")
    return {**out, "verdict": "FAIL" if why else "PASS", "why": "; ".join(why)}


def par_close(a, b, rtol: float = PAR_RTOL) -> bool:
    """The registered par comparison: same length, every component finite and within rtol (relative)."""
    if a is None or b is None or len(a) != len(b):
        return False
    return all(math.isfinite(float(x)) and math.isfinite(float(y)) and abs(float(x) - float(y)) <= rtol * max(abs(float(x)), abs(float(y)))
               for x, y in zip(a, b, strict=True))


# ================================================================== the census gate

def _census_mods():
    for p in (str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "tileA2")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tile_census_a2 as TCA  # noqa: E402
    return TCA


def find_census(results: Path, label: str = CENSUS_LABEL) -> Path | None:
    hits = sorted(Path(results).rglob(f"*-{CARD}-tile-census-*-{label}/tile-census.json"))
    return hits[0] if len(hits) == 1 else None


def census_check(ref: dict, got: dict, facts, fields) -> dict:
    """B3's census `got` against A2's merged census `ref` on the blind models and deepseek-v2-lite at both tiles."""
    by = {(e["model"], e["tile"], e["gemm"]): e for e in got.get("entries") or []}
    drift = {m: [] for m in MODELS}
    compared = 0
    for e in ref.get("entries") or []:
        if e["model"] not in MODELS or e["tile"] not in CENSUS_TILES:
            continue
        compared += 1
        k = (e["model"], e["tile"], e["gemm"])
        g = by.get(k)
        if g is None:
            drift[e["model"]].append(f"{'|'.join(k)}: not compiled on B3's board")
            continue
        d = [f for f in fields if facts(e)[f] != facts(g)[f]]
        if d:
            drift[e["model"]].append(f"{'|'.join(k)}: {', '.join(d)} differ")
    bad = list(got.get("path_mismatch") or [])
    want = 2 * len(CENSUS_TILES) * len(MODELS)
    stop = [x for m in BLIND for x in drift[m]]
    why = (["MMA path mismatch: " + "; ".join(bad[:4])] if bad else []) + stop[:8]
    if compared != want:
        why.append(f"A2's merged census holds {compared} of the {want} (model, tile, GEMM) entries")
    occ = {f"{e['model']}|{e['tile']}|{e['gemm']}": (e.get("occupancy") or {}).get("ctas_per_sm")
           for e in got.get("entries") or [] if e["model"] in MODELS and e["tile"] in CENSUS_TILES}
    return {"verdict": "FAIL" if why else "PASS", "why": "; ".join(why), "compared": compared, "drift": drift,
            "path_mismatch": bad, "deepseek": "NOT SCORED (census drift)" if drift[DSV] else "admitted",
            "ctas_per_sm": dict(sorted(occ.items()))}


def census_gate(results: Path, repo: Path = REPO) -> dict:
    TCA = _census_mods()
    p = find_census(results)
    if p is None:
        return {"gate": "census", "verdict": "FAIL", "why": f"no single B3 tile-census.json (label {CENSUS_LABEL})"}
    got = json.loads(p.read_text())
    ref = json.loads((Path(repo) / A2_MERGED_CENSUS).read_text())
    rec = census_check(ref, got, TCA.facts, TCA.FIELDS)
    return {"gate": "census", "file": str(p), "sha256": sha256(p), **rec}


# ================================================================== the bar and the verdict

def stats(v) -> dict:
    v = [float(x) for x in v]
    if not v:
        return {"cells": 0}
    return {"cells": len(v), "mean": round(st.fmean(v), 6), "rms": round(math.sqrt(st.fmean(x * x for x in v)), 6),
            "worst": round(max(v, key=abs), 6)}


def bar(v, rule: dict = BAR) -> dict:
    s = stats(v)
    if not s["cells"]:
        return {"verdict": "NOT SCORED: no valid cell", "stats": s}
    ok = s["rms"] <= rule["rms_max"] and abs(s["worst"]) <= rule["max_abs"] and abs(s["mean"]) <= rule["mean_max"]
    return {"verdict": "HOLDS" if ok else "FAILS", "stats": s}


def e1_rule(rows: list, pages_in_view: dict, view: str, rule: dict = BAR) -> dict:
    """tcommon.e1tile_rule with the bar as an argument: the pooled bar over every row and each blind model's;
    deepseek-v2-lite in the pooled bar with its own bar printed only; in CLEAN a blind model is CERTIFIED with its
    bar holding and at least MIN_CLEAN_PAGES of its pages in the view; a blind model with no valid cell leaves the
    test incomplete (NOT SCORED) unless the pool or another blind model fails."""
    pooled = bar([r["e"] for r in rows], rule)
    per, ok = {}, pooled["verdict"] == "HOLDS"
    lost = [m for m in BLIND if not any(r["model"] == m for r in rows)]
    for m in BLIND + (DSV,):
        v = bar([r["e"] for r in rows if r["model"] == m], rule)
        rec = {"verdict": v["verdict"], "stats": v["stats"], "pages_in_view": pages_in_view.get(m, [])}
        if m == DSV:
            rec["label"] = "DESCRIPTIVE (one page; new model, seen shapes: never CERTIFIED)"
        else:
            ok = ok and v["verdict"] == "HOLDS"
            if view == "CLEAN":
                n = len(pages_in_view.get(m, []))
                rec["certification"] = ("CERTIFIED" if (v["verdict"] == "HOLDS" and n >= MIN_CLEAN_PAGES)
                                        else f"NOT CERTIFIED ({n} pages in CLEAN, bar {v['verdict']})")
        per[m] = rec
    if lost:
        verdict = f"NOT SCORED (incomplete: no valid cell of {', '.join(lost)})"
        if pooled["verdict"] == "FAILS" or any(per[m]["verdict"] == "FAILS" for m in BLIND if m not in lost):
            verdict = "FAILS (and incomplete: no valid cell of " + ", ".join(lost) + ")"
    else:
        verdict = "HOLDS" if ok else ("FAILS" if pooled["verdict"] != "NOT SCORED: no valid cell" else pooled["verdict"])
    return {"verdict": verdict, "pooled": pooled, "per_model": per, "models_without_a_valid_cell": lost,
            "bar": dict(rule)}


def certification(e1_clean: dict, verdict: str) -> dict:
    """Per blind model: CERTIFIED only when the registered verdict (rental 2's rule 3 over ALL and CLEAN, as B2's
    scorer combines them: a disagreement reads INCONCLUSIVE) HOLDS and CLEAN certifies it."""
    out = {}
    for m in BLIND:
        c = ((e1_clean.get("per_model") or {}).get(m) or {}).get("certification", "NOT CERTIFIED (no CLEAN cell)")
        if verdict != "HOLDS" and c.startswith("CERTIFIED"):
            c = f"NOT CERTIFIED (E1-TILE reads {verdict})"
        out[m] = c
    return out


# ================================================================== the pass estimate (ESTIMATE)

def _rc():
    p = str(HERE.parent / "cal2")
    if p not in sys.path:
        sys.path.insert(0, p)
    import read_c2 as RC  # noqa: E402  (its pure part: NOISE, CLASSES, BLIND, b2_cells, to_b)
    return RC


def power(cells, bclass, sd_model, rule: dict = BAR, R=200_000, chunk=25_000, seed=20261022):
    """read_c2.power (diag-v5 power5's) with the bar as an argument; at B2_BAR it is read_c2.power exactly."""
    RC = _rc()
    rms_max, max_abs, mean_max = rule["rms_max"], rule["max_abs"], rule["mean_max"]
    models = sorted({c["model"] for c in cells})
    pages = sorted({c["page"] for c in cells})
    mi = np.array([models.index(c["model"]) for c in cells])
    ci = np.array([RC.CLASSES.index(c["cls"]) for c in cells])
    pi = np.array([pages.index(c["page"]) for c in cells])
    s = np.array([float(c["s"]) for c in cells])
    rng = np.random.default_rng(seed)
    ok_n = {"E1-TILE": 0, **{m: 0 for m in models}, "pooled": 0}
    done = 0
    N = RC.NOISE
    while done < R:
        B = min(chunk, R - done)
        sp = rng.standard_normal((B, len(models), 3)) * sd_model
        e = (np.asarray(bclass)[ci][None, :] + sp[:, mi, ci] + rng.standard_normal((B, len(cells))) * N["sig_cell"]
             + rng.standard_normal((B, len(cells))) * N["sig_byte"] * s[None, :]
             + rng.standard_normal((B, len(pages)))[:, pi] * N["sig_page"] + rng.standard_normal((B, 1)) * N["sig_board"])
        r = np.exp(-e) - 1          # pred / meas - 1

        def ok_bar(x):
            return (np.sqrt((x ** 2).mean(1)) <= rms_max) & (np.abs(x).max(1) <= max_abs) & (np.abs(x.mean(1)) <= mean_max)
        pooled = ok_bar(r)
        ok = pooled.copy()
        ok_n["pooled"] += int(pooled.sum())
        for j, m in enumerate(models):
            k = ok_bar(r[:, mi == j])
            ok_n[m] += int(k.sum())
            if m in RC.BLIND:
                ok &= k
        ok_n["E1-TILE"] += int(ok.sum())
        done += B
    return {k: round(v / R, 4) for k, v in ok_n.items()}


#: round 5's selected variant, five-model LOMO held-out 128 G 1 means in percent (ROUND5.md section 3, combo5.json;
#: Mixtral-8x7B at A2's board; C2's board +2.01 beside), the estimate's inputs
ROUND5_MEANS = {"olmoe-1b-7b": 0.45, "qwen2-57b-a14b": -1.46, "mixtral-8x7b": 1.77, "mixtral-8x22b": 0.24,
                "mixtral-8x7b-tp2": 2.39}
ROUND5_M_C2 = 2.01


def pass_estimates(means: dict = None, cells=None, rule: dict = BAR, R=200_000) -> dict:
    """The knee round's three estimates (read_c4's method) on five held-out means: A (common bias the mean of the five
    b = -ln(1 + e), model SD their SD), B (no common bias, SD their RMS), D (Mixtral-8x7B's G 1 set aside)."""
    RC = _rc()
    means = dict(ROUND5_MEANS if means is None else means)
    cells = RC.b2_cells() if cells is None else cells
    order = ("olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b", "mixtral-8x22b", "mixtral-8x7b-tp2")
    b = [RC.to_b(means[m]) for m in order]
    A = power(cells, [st.fmean(b)] * 3, st.stdev(b), rule, R=R)
    Bp = power(cells, [0.0] * 3, math.sqrt(st.fmean(x * x for x in b)), rule, R=R)
    four = [RC.to_b(means[m]) for m in order if m != "mixtral-8x7b"]
    D = power(cells, [st.fmean(four)] * 3, st.stdev(four), rule, R=R)
    return {"A (5 models)": A, "B (5 models)": Bp, "D (O, Q, 8x22B, tp2; 8x7B G1 aside)": D}
