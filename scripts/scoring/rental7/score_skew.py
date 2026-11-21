#!/usr/bin/env python
"""Score rental 7's skew stage (docs/registered/2026-10-13-rental7-skew-gh200.json): the gates (G1
under r7-median-run3, 6 of 9 clean, G2 host rule, G4, G5, G3 TOST), E1-SKEW at the time bar with
the per-shape clause and the CLEAN certification, E1-UNI, the E1 rivals and SKEW-RATIO. Rental 6's
scorer's statistics (score_skew.py there) on the rental-7 registration; the P20 view is printed
beside every number (DESCRIPTIVE).

    python scripts/scoring/rental7/score_skew.py <repo> <tree> <out>

Reads under <tree>: the R3 histogram pages gaps-<card>-<model>-ska, -skb, -skc (one G = 8 page
each; Mixtral and Phi). Writes <out>/skew.score.{json,txt}; exits 0 whatever the verdicts. Every
prediction, lever list, band and sha is the registration's; nothing is re-chosen on a page.
"""
from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7common as C7  # noqa: E402

C3, C5, C6 = C7.C3, C7.C5, C7.C6
PART = "skew"
RANK = {"mixtral-8x7b": (3, 6, 16), "phi-3.5-moe": (2, 4, 12)}
PAGES = ("A", "B", "C")
SKEWED = ("PT", "PW", "PW2", "DW")
r6 = C6.r6
DESCRIPTIVE_LABEL = "DESCRIPTIVE, NOT A VERDICT: E1 on the G3-voided histogram cells (rental 6's owner decision kept)"


def _verdict(v: list, rule: dict) -> dict:
    s = C5.stats(v)
    if not s["cells"]:
        return {"verdict": "NOT SCORED: no valid cell", "stats": s}
    ok = s["rms"] <= rule["rms_max"] and abs(s["worst"]) <= rule["max_abs"]
    if "mean_max" in rule:
        ok = ok and abs(s["mean"]) <= rule["mean_max"]
    return {"verdict": "HOLDS" if ok else "FALSIFIED", "stats": s}


def score_view(tree: Path, reg: dict, tview, view: str = "registered") -> dict:
    files = reg["histograms"]["files"]
    preds = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): r for r in reg["predictions"]}
    meas, A_of, use, dropped, in_view = {}, {}, {}, {}, {m: [] for m in RANK}
    for model in RANK:
        for page in PAGES:
            rep, why = C7.page_report(tree, model, f"sk{page.lower()}", tview, files.get(f"{model}-{page}.json"), view)
            use[f"{model}-{page}"] = why
            if rep is None:
                continue
            in_view[model].append(page)
            c, d, A = C6.valid_cells(rep, preds, model, page)
            dropped[f"{model}-{page}"] = d
            A_of[(model, page)] = A
            meas.update(c)

    def pred(k, v="v3"):
        return C6.carried(preds[k], v, A_of.get(k[:2]))

    # ---- G3 ----------------------------------------------------------------------------
    g3rows = []
    for k, r in preds.items():
        if r["label"] != "uniform":
            continue
        b = k[:3] + ("balanced", k[4])
        if k in meas and b in meas:
            g3rows.append({"cluster": f"{k[0]}-{k[1]}", "stratum": RANK[k[0]].index(k[2]),
                           "d": meas[k] / meas[b] - 1 - (r.get("delta_v3") or 0.0), "cell": "/".join(map(str, k)),
                           "arm": k[4], "shape": k[0]})
    g3 = C5.g3_rule(g3rows) if g3rows else {"verdict": "NOT SCORED: no G3 row"}
    g3["d"] = [{"cell": x["cell"], "d": r6(x["d"])} for x in g3rows]
    void = g3.get("void_strata")

    def voided(k) -> bool:
        if not g3rows or void == "ALL":
            return True
        return bool(void) and str(RANK[k[0]].index(k[2])) in [str(s) for s in void]

    # ---- E1-SKEW (pooled and per shape), E1-UNI, rivals ------------------------------------
    rows, uni, e1_void = [], [], []
    for k, r in preds.items():
        if k not in meas:
            continue
        e = pred(k) / meas[k] - 1
        if r["label"] == "balanced":
            uni.append(e)
            continue
        if voided(k):
            if void == "ALL" and g3rows:
                e1_void.append({"model": k[0], "arm": k[4], "e": e})
            continue
        rows.append({"cell": "/".join(map(str, k)), "model": k[0], "arm": k[4], "e": e,
                     "eR": {R: pred(k, R) / meas[k] - 1 for R in reg["tests"]["E1-rivals"]["rivals"]}, "lever": r["lever"]})
    bar = reg["tests"]["E1-SKEW"]
    E1 = _verdict([x["e"] for x in rows], bar)
    E1["per_shape"] = {}
    shapes_ok = True
    for m in RANK:
        v = _verdict([x["e"] for x in rows if x["model"] == m], bar)
        cert = None
        if tview.name == "CLEAN":
            n_pages = len(in_view[m])
            if n_pages < C7.MIN_CLEAN_PAGES_PER_SHAPE:
                cert = f"NOT CERTIFIED ({n_pages} of 3 pages in CLEAN, under {C7.MIN_CLEAN_PAGES_PER_SHAPE})"
            else:
                cert = "CERTIFIED" if v["verdict"] == "HOLDS" else "NOT CERTIFIED (its CLEAN cells miss the bar)"
        E1["per_shape"][m] = {"verdict": v["verdict"], "stats": v["stats"], "pages_in_view": in_view[m],
                              **({"certification": cert} if cert else {})}
        shapes_ok = shapes_ok and v["verdict"] == "HOLDS"
    if E1["verdict"] == "HOLDS" and not shapes_ok:
        E1["verdict"] = "FALSIFIED (a shape misses the bar)"
    E1["rental5_bands"] = _verdict([x["e"] for x in rows], bar["rental5_bands_printed"])["verdict"]
    E1["per_arm"] = {a: C5.stats(x["e"] for x in rows if x["arm"] == a) for a in ("native", "shared")}
    if void == "ALL" and g3rows:
        d = _verdict([x["e"] for x in e1_void], bar)
        E1 = {"verdict": "NOT SCORED: G3 FAILS and Q does not reject, so every histogram cell is voided (strict rule kept)",
              "descriptive": {"label": DESCRIPTIVE_LABEL, "reads_as": d["verdict"].replace("HOLDS", "would hold").replace(
                  "FALSIFIED", "would not hold"), "stats": d["stats"]}}
    T = reg["tests"]["E1-rivals"]
    riv = {}
    for R in T["rivals"]:
        lev = [x for x in rows if x["lever"].get(R)]
        if len(lev) < T["min_lever_cells"]:
            riv[R] = {"verdict": f"UNDECIDED: {len(lev)} lever cells survive, under {T['min_lever_cells']}"}
            continue
        rr = math.sqrt(st.fmean(x["eR"][R] ** 2 for x in lev))
        rv = math.sqrt(st.fmean(x["e"] ** 2 for x in lev))
        riv[R] = {"verdict": "EXCLUDED" if (rr >= T["rival_factor"] * rv and rr >= T["rival_floor"]) else "NOT EXCLUDED",
                  "lever_cells": len(lev), "rms_rival": r6(rr), "rms_v3_on_levers": r6(rv)}
    E1u = _verdict(uni, bar)

    # ---- SKEW-RATIO -------------------------------------------------------------------------
    TR = reg["tests"]["SKEW-RATIO"]
    rat = []
    for k, r in preds.items():
        if "ratio" not in r or voided(k):
            continue
        u = k[:3] + ("uniform", k[4])
        if k in meas and u in meas:
            rm = meas[k] / meas[u]
            rat.append({"cell": "/".join(map(str, k)), "e": r["ratio"]["v3"] / rm - 1, "e_LT3": r["ratio"]["LT3"] / rm - 1,
                        "e_U": 1.0 / rm - 1, "lever": r["lever_ratio"]})
    ratio = {"cells": len(rat)}
    if rat:
        s = C5.stats([x["e"] for x in rat])
        ratio["v3"] = {"verdict": "HOLDS" if (s["rms"] <= TR["rms_max"] and abs(s["worst"]) <= TR["max_abs"]) else "FAILS", "stats": s}
        for R, key in (("LT3", "e_LT3"), ("U", "e_U")):
            lev = [x for x in rat if x["lever"].get(R)]
            if len(lev) < C6.MIN_LEVER:
                ratio[R] = {"verdict": f"UNDECIDED: {len(lev)} lever cells"}
                continue
            rr = math.sqrt(st.fmean(x[key] ** 2 for x in lev))
            rv = math.sqrt(st.fmean(x["e"] ** 2 for x in lev))
            ratio[R] = {"verdict": "EXCLUDED" if (rr >= C6.RIVAL_FACTOR * rv and rr >= C6.RIVAL_FLOOR) else "NOT EXCLUDED",
                        "lever_cells": len(lev), "rms_rival": r6(rr), "rms_v3_on_levers": r6(rv)}
    else:
        ratio["v3"] = {"verdict": "NOT SCORED: no valid ratio"}
    out = {"pages": use, "dropped_rows": dropped, "cells_measured": len(meas), "G3": g3, "E1-SKEW": E1,
           "E1-rivals": riv, "E1-UNI": E1u, "SKEW-RATIO": ratio}
    if not g3rows:
        out["E1-SKEW"] = {"verdict": "NOT SCORED: no G3 row on the board, so the histogram cells have no control"}
    return out


def certification(res: dict) -> dict:
    """Per shape: CERTIFIED iff the CLEAN view certifies it (its CLEAN cells meet the bar, at least
    2 of its 3 pages in CLEAN) AND E1-SKEW's registered verdict (both views) HOLDS."""
    clean = (((res.get("addendum") or {}).get("CLEAN") or {}).get("E1-SKEW") or {}).get("per_shape") or {}
    holds = (res.get("E1-SKEW") or {}).get("verdict", "")
    out = {}
    for m in RANK:
        c = (clean.get(m) or {}).get("certification") or "NOT CERTIFIED (no CLEAN cell)"
        if c == "CERTIFIED" and holds != "HOLDS":
            c = f"NOT CERTIFIED (E1-SKEW's registered verdict reads {holds})"
        out[m] = c
    return out


def _p20(tree: Path, reg: dict, repo: Path) -> dict:
    p = C3.two_views(lambda fv, tv: score_view(tree, reg, tv, "p20"), repo=repo)
    clean = (p.get("addendum") or {}).get("CLEAN")
    return {"label": "DESCRIPTIVE, NOT A VERDICT: every test on the P20 side view (today's exclusions under a 20% ceiling; "
                     "rental7-g1rule)", "combined": {k: v for k, v in p.items() if k != "addendum"}, "CLEAN": clean,
            "pages": (p.get("addendum") or {}).get("pages")}


def score(repo: Path, tree: Path) -> dict:
    reg = C7.registration(repo, PART)
    res = C3.two_views(lambda fv, tv: score_view(tree, reg, tv), repo=repo)
    res["P20_view"] = _p20(tree, reg, repo)
    res["certification"] = certification(res)
    res.update(registration=reg["name"], part=PART)
    return res


def lines(res: dict) -> list[str]:
    out = [f"SCORED against {res['registration']} (ALL view; CLEAN and the P20 view beside it in the JSON)"]
    for k, v in res.get("pages", {}).items():
        out.append(f"  page {k}: {v}")
    g3 = res.get("G3", {})
    out.append(f"G3: {g3.get('verdict')}; pooled {(g3.get('pooled') or {}).get('verdict')} CI {(g3.get('pooled') or {}).get('ci')}")
    for t in ("E1-SKEW", "E1-UNI"):
        v = res.get(t, {})
        out.append(f"{t}: {v.get('verdict')} {v.get('stats', '')}")
        for m, s in (v.get("per_shape") or {}).items():
            out.append(f"  {m}: {s.get('verdict')} {s.get('stats')}")
    for R, v in (res.get("E1-rivals") or {}).items():
        out.append(f"  E1 rival {R}: {v.get('verdict')}")
    r = res.get("SKEW-RATIO", {})
    out.append(f"SKEW-RATIO v3: {(r.get('v3') or {}).get('verdict')} ({r.get('cells')} ratios); LT3 {(r.get('LT3') or {}).get('verdict')}; "
               f"U {(r.get('U') or {}).get('verdict')}")
    out.append(f"certification (CLEAN): {res.get('certification')}")
    p = (res.get("P20_view") or {}).get("combined") or {}
    out.append(f"P20 view (DESCRIPTIVE): E1-SKEW {(p.get('E1-SKEW') or {}).get('verdict')}; SKEW-RATIO "
               f"{((p.get('SKEW-RATIO') or {}).get('v3') or {}).get('verdict')}; G3 {(p.get('G3') or {}).get('verdict')}")
    out += C3.view_lines(res)
    return out


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    res = score(repo, tree)
    print(C3.write_score(out, PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
