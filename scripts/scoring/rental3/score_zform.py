#!/usr/bin/env python
"""Score rental 3's part A, the per-GEMM constant's form
(docs/registered/2026-10-05-rental3-zform-gh200.json): A1, A2, MID, the verdict.

    python scripts/scoring/rental3/score_zform.py <repo> <tree> <out>

Reads the qwen2-57b-a14b-tp8 and granite-3.0-1b-a400m floor and floor1005 captures
(and tp4's for the printed K3 refit) under <tree>. Writes <out>/zform.score.{json,txt};
exits 0 whatever the verdicts. Each capture FILE is its own series (base, lock1710,
lock1005 never pooled); every centre is priced at the capture's measured clock.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3common as C3  # noqa: E402

PART = "zform"
FORMS = {
    "PROP": lambda u, f: [u],
    "AFF": lambda u, f: [1.0, u],
    "MIX": lambda u, f: [f / 1710.0, u],
    "MIX3": lambda u, f: [f / 1710.0, 1.0, u],
}


def zpred(reg, form, u, f):
    return float(sum(c * x for c, x in zip(reg["fits_CAL"][form]["coef"], FORMS[form](u, f), strict=True)))


def z_of(rows: list, sigma_reg: float) -> dict | None:
    """Z by intercept of ACT.avg on q, with sigma = max(the registered sigma_int,
    the realised intercept error)."""
    import floor_estimator as FE
    pts = [(r["q"], r["ACT.avg"]) for r in rows if r["ACT.avg"] is not None]
    if len(pts) < 3:
        return None
    q, y = [p[0] for p in pts], [p[1] for p in pts]
    f = FE.ols(y, [[1.0] * len(q), q])
    se = math.sqrt(max(f["cov"][0][0], 0.0))
    return {"Z": f["coef"][0], "slope": f["coef"][1], "se_realised": se, "sigma": max(sigma_reg, se),
            "mhz": C3.median([r["mhz"] for r in rows if r["mhz"]]), "cells": [r["n"] for r in rows]}


def curvature_ok(rows: list, sep: dict) -> tuple[bool, dict]:
    lo = [r for r in rows if r["n"] <= 8]
    hi = [r for r in rows if r["n"] >= 8]
    sl, sh = sep["half_fits"]["sigma_lo"] or 0.0, sep["half_fits"]["sigma_hi"] or 0.0
    a, b = z_of(lo, sl), z_of(hi, sh)
    if a is None or b is None:
        return True, {"why": "a half has under 3 cells: not checked"}
    lim = 2 * math.sqrt(a["sigma"] ** 2 + b["sigma"] ** 2)
    return abs(a["Z"] - b["Z"]) <= lim, {"Z_lo": a["Z"], "Z_hi": b["Z"], "limit": lim}


def score_view(repo: Path, tree: Path, fview, tview) -> dict:
    reg = C3.registration(repo, PART)
    flr = C3.registration(repo, "floorlaw")
    res = {"registration": C3.NAMES[PART], "gemms": {}}
    pages = {}

    def page(model, label, stem):
        k = (model, label, stem)
        if k not in pages:
            pages[k] = fview.load(C3.find_floor(tree, model, label, stem))
        return pages[k]
    fails = {f: [] for f in ("PROP", "AFF", "MIX")}
    scored = {f: [] for f in ("PROP", "AFF", "MIX")}
    mids, mid5 = [], []
    for key, sep in reg["separating_gemms"].items():
        model, g = key.split()
        cells = flr["cells_registered"][model]
        caps = {"base": page(model, "floor", "r3f-g64"), "lock1710": page(model, "floor", "r3f-g64-lock1710"),
                "lock1005": page(model, "floor1005", "r3f-g64-lock1005")}
        Z, curv = {}, {}
        for c, p in caps.items():
            if p is None:
                continue
            rows = C3.floor_rows(p, cells, g, None)
            z = z_of(rows, sep["sigma_int"])
            if z is not None:
                ok, info = curvature_ok(rows, sep)
                z["curvature"] = info
                z["curvature_ok"] = ok
                Z[c] = z
        ent = {"Z": Z, "role": sep["role"], "u": sep["u"]}
        res["gemms"][key] = ent
        if "lock1710" not in Z:
            ent["verdict"] = "NOT SCORED: no 1710 capture"
            continue
        if not all(z["curvature_ok"] for z in Z.values()):
            ent["verdict"] = "NOT SCORED: the half-fits' intercepts disagree (curvature)"
            continue
        z7 = Z["lock1710"]
        u = sep["u"]
        s_tot = math.sqrt(z7["sigma"] ** 2 + reg["noise"]["sigma_board"] ** 2 + reg["noise"]["sigma_model"] ** 2)
        a1 = {}
        for form in ("PROP", "AFF", "MIX", "MIX3"):
            cen = zpred(reg, form, u, z7["mhz"])
            a1[form] = {"centre": cen, "band": [cen - 2 * s_tot, cen + 2 * s_tot],
                        "inside": abs(z7["Z"] - cen) <= 2 * s_tot,
                        "scored": form in sep["A1_forms"]}
        a2 = None
        if "base" in Z:
            zb = Z["base"]
            r = zb["Z"] / z7["Z"]
            sr = abs(r) * math.sqrt(zb["sigma"] ** 2 + z7["sigma"] ** 2) / abs(z7["Z"])
            rmix = zpred(reg, "MIX", u, zb["mhz"]) / zpred(reg, "MIX", u, z7["mhz"])
            a2 = {"r": r, "sigma_r": sr, "r_MIX": rmix, "AFF_PROP_fail": abs(r - 1) > 2 * sr,
                  "MIX_fail": abs(r - rmix) > 2 * sr, "MID": rmix + 2 * sr < r < 1 - 2 * sr}
            if a2["MID"]:
                mids.append(key)
        a2s = None
        if "lock1005" in Z:
            z5 = Z["lock1005"]
            r5 = z5["Z"] / z7["Z"]
            s5 = abs(r5) * math.sqrt(z5["sigma"] ** 2 + z7["sigma"] ** 2) / abs(z7["Z"])
            rm5 = zpred(reg, "MIX", u, z5["mhz"]) / zpred(reg, "MIX", u, z7["mhz"])
            a2s = {"r": r5, "sigma_r": s5, "r_MIX": rm5, "MID": rm5 + 2 * s5 < r5 < 1 - 2 * s5,
                   "label": "SM and L2 moved together (printed)"}
            if a2s["MID"] and sep["role"] == "scored":
                mid5.append(key)
        ent.update(A1=a1, A2=a2, A2_second_printed=a2s, sigma_tot=s_tot)
        ent["verdict"] = "RECORD"
        if sep["role"] != "scored":
            continue
        for form in fails:
            fail = (a1[form]["scored"] and not a1[form]["inside"])
            if a2 is not None:
                fail = fail or (a2["MIX_fail"] if form == "MIX" else a2["AFF_PROP_fail"])
            if a1[form]["scored"] or a2 is not None:
                scored[form].append(key)
                if fail:
                    fails[form].append(key)
    status = {f: ("FALSIFIED" if len(fails[f]) >= 2 else "no failure" if scored[f] and not fails[f]
                  else "not falsified") for f in fails}
    res["forms"] = {f: {"scored_on": scored[f], "fails_on": fails[f], "status": status[f]} for f in fails}
    if not any(scored.values()):
        res["verdict"] = "NOT SCORED: no separating GEMM scored"
    else:
        sel = [f for f in fails if status[f] == "no failure"
               and all(status[o] == "FALSIFIED" for o in fails if o != f)]
        res["verdict"] = ("MID" if len(mids) >= 2 else sel[0] if sel else "INCONCLUSIVE")
    res["MID_on"] = mids
    res["MID_1005_printed"] = {"on": mid5, "reads_MID": len(mid5) >= 2}
    # printed: K3's tp4 value refit under each form
    p7, p5 = page("mixtral-8x7b-tp4", "floor", "r3f-g64-lock1710"), page("mixtral-8x7b-tp4", "floor1005", "r3f-g64-lock1005")
    k3 = {}
    if p7 is not None and p5 is not None:
        cells = flr["cells_registered"]["mixtral-8x7b-tp4"]
        for g in ("w1", "w2"):
            a, b = z_of(C3.floor_rows(p5, cells, g, None), 0.0), z_of(C3.floor_rows(p7, cells, g, None), 0.0)
            if a and b:
                u = next(c["u"] for c in cells if c["gemm"] == g)
                k3[g] = {"r": a["Z"] / b["Z"], **{f: zpred(reg, f, u, a["mhz"]) / zpred(reg, f, u, b["mhz"])
                                                   for f in ("AFF", "MIX", "MIX3")}}
    res["K3_tp4_printed"] = k3
    return res


def score(repo: Path, tree: Path) -> dict:
    flr = C3.registration(repo, "floorlaw")
    return C3.two_views(lambda fv, tv: score_view(repo, tree, fv, tv),
                        floor_rules=flr["gates"]["floor"], repo=repo)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 3 PART A (the per-GEMM constant's form), {res['registration']}"]
    for k, e in res["gemms"].items():
        out.append(f"  {k} ({e['role']}): {e.get('verdict')}")
        for c, z in e.get("Z", {}).items():
            out.append(f"    {c}: Z {z['Z']:.0f} +- {z['sigma']:.0f} at {z['mhz']} MHz")
        if e.get("A2"):
            a = e["A2"]
            out.append(f"    A2 base/1710 r {a['r']:.3f} +- {a['sigma_r']:.3f} (MIX {a['r_MIX']:.3f}); MID {a['MID']}")
    for f, v in (res.get("forms") or {}).items():
        out.append(f"  {f}: {v['status']} (fails on {v['fails_on']})")
    out.append(f"  verdict: {res['verdict']}")
    return out + C3.view_lines(res)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    res = score(repo, tree)
    print(C3.write_score(out, PART, res, lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
