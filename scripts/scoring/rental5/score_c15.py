#!/usr/bin/env python
"""Score rental 5's SHARED c15 lever (docs/registered/2026-10-07-rental5-c15-gh200.json): the
dead increment of 15 against 9 copies on OLMoE's skewed cells, D against M and FIXED, and
whether the increment moves with the histogram (SKEW-DEAD).

    python scripts/scoring/rental5/score_c15.py <repo> <tree> <out>

Reads gaps-<card>-olmoe-1b-7b-sk15 (c15) and -ska, -skb (c9) under <tree>. Writes
<out>/c15.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5common as C5  # noqa: E402
import score_skew as SK  # noqa: E402

C3 = C5.C3
PART = "c15"
MODEL = "olmoe-1b-7b"
E = 64


def _decl_ok(report: dict, copies: int) -> bool:
    return all(int(r["experts_declared"]) == E * copies for r in report.get("treads_table") or []
               if r["arm"] == "shared")


def _align(report: dict) -> dict:
    return C5.C4.align_ms(report)


def score_view(tree: Path, reg: dict, skreg: dict, tview) -> dict:
    files = skreg["histograms"]["files"]
    pages, why = {}, {}
    for lab, fname, copies in (("sk15", f"{MODEL}-c15.json", 15), ("ska", f"{MODEL}-A.json", 9),
                               ("skb", f"{MODEL}-B.json", 9)):
        rep, w = SK.page_report(tree, MODEL, lab, tview, files.get(fname))
        if rep is not None and not _decl_ok(rep, copies):
            rep, w = None, f"NOT SCORED: SHARED rows do not declare {E * copies} slots"
        pages[lab], why[lab] = rep, w
    out = {"pages": why}
    if any(v is None for v in pages.values()):
        out["verdict"] = "NOT SCORED: a page of the three is missing or out"
        return out
    m = {k: SK.cells_of(v)[0] for k, v in pages.items()}
    al = {k: _align(v) for k, v in pages.items()}
    rows = []
    for pr in reg["predictions"]:
        h, n = pr["label"], pr["n"]
        c9 = "skb" if h == "DW" else "ska"
        need = [("sk15", (h, n, "shared")), ("sk15", ("uniform", n, "native")),
                (c9, (h, n, "shared")), (c9, ("uniform", n, "native"))]
        if not all(k in m[p] for p, k in need):
            continue
        d = ((m["sk15"][(h, n, "shared")] - m["sk15"][("uniform", n, "native")])
             - (m[c9][(h, n, "shared")] - m[c9][("uniform", n, "native")])) * 1e3
        has_al = all(al[p] for p in ("sk15", c9))
        a = (((al["sk15"].get(("shared", n), 0) - al["sk15"].get(("native", n), 0))
              - (al[c9].get(("shared", n), 0) - al[c9].get(("native", n), 0))) * 1e3) if has_al else 0.0
        rows.append({"h": h, "n": n, "Delta_raw_us": C5.r6(d), "A_us": C5.r6(a) if has_al else None,
                     "Delta_us": C5.r6(d - a), "pred": pr["Delta_us"],
                     "y9": (m[c9][(h, n, "shared")] - m[c9][("uniform", n, "native")]) * 1e3,
                     "y15": (m["sk15"][(h, n, "shared")] - m["sk15"][("uniform", n, "native")]) * 1e3,
                     "dead9": pr.get("dead9_total"), "dead15": pr.get("dead15_total"), "live": pr.get("live_rows")})
    out["rows"] = [{k: v for k, v in r.items() if k not in ("y9", "y15")} for r in rows]
    if len(rows) < 4:
        out["verdict"] = f"NOT SCORED: {len(rows)} of the 8 cells"
        return out
    med = st.median(r["Delta_us"] for r in rows)
    sig = reg["noise"]["sigma_noise_us"]
    pred = reg["predicted_median_us"]
    z = {k: (med - v) / sig for k, v in pred.items()}
    res = {"Delta_median_us": C5.r6(med), "z": {k: C5.r6(v) for k, v in z.items()},
           "alignment": "corrected" if all(r["A_us"] is not None for r in rows) else "RAW (no align_probe on the pages)",
           "D": "FAILS" if abs(z["D"]) > 3 else "HOLDS",
           "M": "EXCLUDED" if abs(z["M"]) > 3 else "NOT EXCLUDED",
           "FIXED": "EXCLUDED" if med > 3 * sig else "NOT EXCLUDED"}
    sk = [r["Delta_us"] for r in rows if r["h"] != "uniform"]
    un = [r["Delta_us"] for r in rows if r["h"] == "uniform"]
    if sk and un:
        gap = st.median(sk) - st.median(un)
        res["SKEW-DEAD"] = {"gap_us": C5.r6(gap), "verdict": "SHOWN (the increment moves with the histogram)"
                            if abs(gap) > 3 * sig else "NOT SHOWN"}
    res["at_sigma_alt"] = {k: C5.r6((med - v) / (sig * C5.SIGMA_PAGE_ALT / C5.SIGMA_PAGE)) for k, v in pred.items()}
    # the two-regressor fit [S - N] = a + d N_dead + b live, with its VIF (printed)
    pts = [(r["dead9"], r["live"], r["y9"]) for r in rows if r["dead9"] is not None]
    pts += [(r["dead15"], r["live"], r["y15"]) for r in rows if r["dead15"] is not None]
    res["two_regressor"] = two_regressor(pts)
    out.update(res, verdict=res["D"])
    return out


def two_regressor(pts) -> dict:
    import numpy as np
    if len(pts) < 4:
        return {"verdict": "NOT APPLICABLE"}
    X = np.array([[1.0, p[0], p[1]] for p in pts])
    y = np.array([p[2] for p in pts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    x1, x2 = X[:, 1], X[:, 2]
    r = float(np.corrcoef(x1, x2)[0, 1]) if x1.std() and x2.std() else 0.0
    vif = 1 / (1 - r * r) if abs(r) < 1 else math.inf
    return {"a_us": C5.r6(coef[0]), "d_ns": C5.r6(coef[1] * 1e3), "b_us_per_row": C5.r6(coef[2]),
            "r_dead_live": C5.r6(r), "VIF": C5.r6(vif), "printed": "not a verdict"}


def score(repo: Path, tree: Path) -> dict:
    reg = C5.registration(repo, PART)
    skreg = C5.registration(repo, "skew")
    res = C3.two_views(lambda fv, tv: score_view(tree, reg, skreg, tv), repo=repo)
    res.update(registration=reg["name"], part=PART)
    return res


def lines(res: dict) -> list[str]:
    out = [f"SCORED against {res['registration']} (ALL view; CLEAN beside it in the JSON)"]
    for k, v in (res.get("pages") or {}).items():
        out.append(f"  page {k}: {v}")
    out.append(f"verdict: {res.get('verdict')}; Delta median {res.get('Delta_median_us')} us; z {res.get('z')}")
    for k in ("M", "FIXED", "SKEW-DEAD", "alignment", "two_regressor"):
        if k in res:
            out.append(f"  {k}: {res[k]}")
    out += C3.view_lines(res)
    return out


def main(argv=None) -> int:
    a = (argv if argv is not None else sys.argv[1:])
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    res = score(repo, tree)
    print(C5.write_score(out, PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
