#!/usr/bin/env python
"""Score rental 3's part E, end-to-end blind time on qwen2-57b-a14b-tp8
(docs/registered/2026-10-05-rental3-e2e-gh200.json): E1 to E6.

    python scripts/scoring/rental3/score_e2e.py <repo> <tree> <out>

Reads under <tree>: the timed pages of the unit's `gaps-<card>-qwen2-57b-a14b-tp8-e2e`
directory (the timed step's -p2- and -p5- pages are the core, -deep- the deep
pages; rep8's own directory is never read here), the byte pages
`*-qwen2-57b-a14b-tp8-e2e-r3-counters/lock1710/r3c-g{3,8,32}.json` and the floor
capture `*-qwen2-57b-a14b-tp8-floor1005-r3-counters/r3f-g64-lock1005.json`.
Writes <out>/e2e.score.{json,txt}; exits 0 on any verdict. The cell population, the
predictions and every band are the registration's, never re-chosen on a page.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3common as C3  # noqa: E402
import score_replicate as SR  # noqa: E402

PART = "e2e"


def _stats(v):
    v = list(v)
    if not v:
        return None
    return {"cells": len(v), "rms": math.sqrt(sum(x * x for x in v) / len(v)),
            "max_abs": max(abs(x) for x in v)}


def e1_verdict(reg: dict, rows: list, need: int) -> dict:
    s = _stats(r["rel"] for r in rows)
    if s is None or s["cells"] < need:
        return {"verdict": f"NOT SCORED: {0 if s is None else s['cells']} of the population present, under {need}",
                "stats": s}
    ok = s["rms"] <= reg["E1"]["rms_max"] and s["max_abs"] <= reg["E1"]["max_abs"]
    return {"verdict": "HOLDS" if ok else "FALSIFIED", "stats": s}


def resolution(sp: dict, sb: float) -> float | None:
    p = sp.get("sigma_page")
    return None if p is None else math.sqrt(p * p + sb * sb)


def unresolved_label(reg: dict, s: dict | None, res: float | None, sb: float) -> str:
    if s is None:
        return ""
    r = res if res is not None else sb
    which = "page and board" if res is not None else "board only (no rep8)"
    near = (abs(s["rms"] - reg["E1"]["rms_max"]) < r or abs(s["max_abs"] - reg["E1"]["max_abs"]) < r)
    return f"UNRESOLVED (resolution {100 * r:.2f}%, {which})" if near else f"resolved (resolution {100 * r:.2f}%, {which})"


def default_e2(reg, repo, page_dirs, counters_dir):
    """E2 by cross_model_score.score at the registration's code (its sha256 checked)."""
    pin = reg.get("code_pins", {}).get("scripts/cross_model_score.py")
    have = hashlib.sha256((Path(repo) / "scripts" / "cross_model_score.py").read_bytes()).hexdigest()
    if pin and pin != have:
        return {"verdict": "NOT SCORED: scripts/cross_model_score.py differs from the registration's (sha256)"}
    import cores_heldout_predict as CP
    import cross_model_score as XS
    d = XS.score(CP.source_pages(), CP.C27, C3.TARGET, page_dirs, counters_dir)
    return {"cells": {(c["arm"], c["G"], c["n"]): c["resid"] for c in d["cells"]}}


def score_view(repo: Path, tree: Path, fview, tview, *, e2=None) -> dict:
    import floor_estimator as FE
    import numpy as np
    reg = C3.registration(repo, PART)
    rreg = C3.registration(repo, "replicate")
    flr = C3.registration(repo, "floorlaw")
    res = {"registration": C3.NAMES[PART], "scope": reg["scope"]}
    root = C3.find_gaps(tree, C3.TARGET, "e2e")
    core, deep, dirs = {}, {}, []
    for path, rep in C3.timed_pages(root):
        tag = str(rep.get("session_tag") or "")
        rep = tview.admit(f"e2e/{path.parent.name}", rep)
        if rep is None:
            continue
        G = C3.group_of(rep)
        acc = deep if "-deep-" in tag else core
        if acc is core:
            dirs.append(path.parent)
        for (arm, n), ms in C3.treads_ms(rep).items():
            acc.setdefault((arm, G, n), []).append(ms)
    core = {k: C3.median(v) for k, v in core.items()}
    deep = {k: C3.median(v) for k, v in deep.items()}

    def rows_of(pop, meas):
        out = []
        for c in pop:
            k = (c["arm"], c["G"], c["n"])
            if k in meas:
                out.append({"arm": c["arm"], "G": c["G"], "n": c["n"], "T_meas": meas[k], "T_M": c["T_M"],
                            "T_MZ": c["T_MZ"], "rel": meas[k] / c["T_M"] - 1, "rel_MZ": meas[k] / c["T_MZ"] - 1})
        return out
    pop = reg["population"]["core"]
    need = math.ceil(0.75 * len(pop))
    r1 = rows_of(pop, core)
    e1 = e1_verdict(reg, r1, need)
    sb = rreg["R"]["sigma_board"]["rms"]
    sp = SR.sigma_page(repo, tree, tview)
    rz = resolution(sp, sb)
    e1["resolution"] = rz
    e1["label"] = unresolved_label(reg, e1.get("stats"), rz, sb)
    e1["cells"] = r1
    res["E1"] = e1
    rd = rows_of(reg["population"]["deep"], deep)
    res["E1_deep_printed"] = {"cells": rd, "stats": _stats(r["rel"] for r in rd),
                              "reading": e1_verdict(reg, rd, math.ceil(0.75 * len(reg["population"]["deep"])))["verdict"]}
    # E2: counted bytes
    byte_paths = {G: C3.find_bytes(tree, C3.TARGET, "e2e", G) for G in (3, 8, 32)}
    if not dirs or any(p is None for p in byte_paths.values()):
        res["E2"] = {"verdict": "NOT SCORED: a timed or byte page is missing"}
    else:
        got = (e2 or default_e2)(reg, repo, dirs, next(iter(byte_paths.values())).parent)
        if "cells" not in got:
            res["E2"] = got
        else:
            rel = [got["cells"][(c["arm"], c["G"], c["n"])] for c in pop
                   if (c["arm"], c["G"], c["n"]) in got["cells"]]
            # cross_model_score's resid is predicted / measured - 1
            s = _stats(rel)
            if s is None or s["cells"] < need:
                res["E2"] = {"verdict": "NOT SCORED: too few population cells", "stats": s}
            else:
                ok = s["rms"] <= reg["E1"]["rms_max"] and s["max_abs"] <= reg["E1"]["max_abs"]
                res["E2"] = {"verdict": "HOLDS" if ok else "FALSIFIED", "stats": s}
    # E3 / E3s: bytes
    import dram_counter_route as DCR
    qrows = {"private": [], "shared": []}
    e6 = {"w1": [], "w2": []}
    for G, p in byte_paths.items():
        page = fview.load(p) if p is not None else None
        if page is None:
            continue
        q = DCR.r3_q(page)
        for arm in qrows:
            for n in range(1, 10):
                pred = reg["q_pred"].get(f"{arm} G={G} n={n}")
                for g in ("w1", "w2"):
                    v = (q.get(arm, {}).get(g) or {}).get(n)
                    if pred and pred[g] and v is not None:
                        qrows[arm].append({"G": G, "n": n, "gemm": g, "q": v, "q_pred": pred[g], "rel": v / pred[g] - 1})
        cells = {(c["arm"], int(c["n"])): c for c in page["cells"]}
        for n in range(2, 10):
            sh, na = cells.get(("shared", n)), cells.get(("native", n))
            if sh is None or na is None:
                continue
            for g in e6:
                a = ((sh.get("recorded") or {}).get(g) or {}).get("sm__cycles_elapsed.avg")
                b = ((na.get("recorded") or {}).get(g) or {}).get("sm__cycles_elapsed.avg")
                if a is not None and b is not None:
                    e6[g].append((a - b) / 1710.0)
    for arm, key in (("private", "E3"), ("shared", "E3s")):
        rs = qrows[arm]
        if not rs:
            res[key] = {"verdict": "NOT SCORED: no byte page"}
            continue
        rms = math.sqrt(sum(r["rel"] ** 2 for r in rs) / len(rs))
        share = sum(abs(r["rel"]) > reg["E3"]["beyond"] for r in rs) / len(rs)
        ok = rms <= reg["E3"]["rms_max"] and share <= reg["E3"]["beyond_share_max"]
        res[key] = {"verdict": "HOLDS" if ok else "FALSIFIED", "cells": len(rs), "rms": rms,
                    "share_beyond_5pct": share, "rows": rs}
    # E4: w2's per-CTA unit on the 1005 capture's DUR
    cap = fview.load(C3.find_floor(tree, C3.TARGET, "floor1005", "r3f-g64-lock1005"))
    if cap is None:
        res["E4"] = {"verdict": "NOT SCORED: the 1005 capture is missing or out of this view"}
    else:
        cr = flr["cells_registered"][C3.TARGET]
        rows = [r for r in C3.floor_rows(cap, cr, "w2", 1005.0) if r["DUR"] is not None]
        if len(rows) < 5:
            res["E4"] = {"verdict": "NOT SCORED: fewer than 5 w2 cells"}
        else:
            jt = FE.joint_theta([r["DUR"] / r["u"] for r in rows], [r["q"] for r in rows], [r["frac"] for r in rows])
            b = jt["b"]
            lo, hi = reg["E4"]["M_band"]
            flo, fhi = reg["E4"]["F0_band"]
            v = "M HOLDS (F0 FALSIFIED)" if lo <= b <= hi else ("F0 HOLDS (M FALSIFIED)" if flo <= b <= fhi else "NEITHER")
            w1 = [r for r in C3.floor_rows(cap, cr, "w1", 1005.0) if r["DUR"] is not None]
            w1s = (float(np.polyfit([r["q"] for r in w1], [r["DUR"] / r["u"] for r in w1], 1)[0])
                   if len(w1) >= 3 else None)
            res["E4"] = {"verdict": v, "fit_w2": jt, "w1_q_slope_printed": w1s}
    # E5: printed
    if r1:
        rm = math.sqrt(sum(r["rel"] ** 2 for r in r1) / len(r1))
        rmz = math.sqrt(sum(r["rel_MZ"] ** 2 for r in r1) / len(r1))
        p = sp.get("sigma_page") or 0.0
        res["E5_printed"] = {"rms_M": rm, "rms_MZ": rmz, "difference": rmz - rm,
                             "sigma_rms": math.sqrt(2 * p * p / len(r1) + sb * sb),
                             "status": reg["E5"]["status"]}
    # E6: the dead-CTA term
    w2 = e6["w2"]
    if len(w2) < 3:
        res["E6"] = {"verdict": "NOT SCORED: no SHARED / NATIVE byte cells"}
    else:
        m = C3.median(w2)
        bd = reg["E6"]["bands_w2"]
        dead = bd["DEAD"][0] <= m <= bd["DEAD"][1]
        fixed = bd["FIXED"][0] <= m <= bd["FIXED"][1]
        v = ("DEAD" if dead and not fixed else "FIXED" if fixed and not dead else "INCONCLUSIVE")
        res["E6"] = {"verdict": v, "median_w2_us": m, "cells_w2": len(w2),
                     "median_w1_us_printed": C3.median(e6["w1"]) if e6["w1"] else None,
                     "predicted_w2_us": reg["E6"]["predicted"]["w2"]["pred_us"]}
    return res


def score(repo: Path, tree: Path, *, e2=None) -> dict:
    flr = C3.registration(repo, "floorlaw")
    return C3.two_views(lambda fv, tv: score_view(repo, tree, fv, tv, e2=e2),
                        floor_rules=flr["gates"]["floor"], timed_ignore=("V5",), repo=repo)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 3 PART E (end-to-end blind time, {C3.TARGET}), {res['registration']}",
           f"  scope: {res['scope']}"]
    e1 = res["E1"]
    s = e1.get("stats") or {}
    out.append(f"  E1 (predicted bytes): {e1['verdict']} {e1.get('label', '')}"
               + (f" rms {100 * s['rms']:.2f}% max {100 * s['max_abs']:.2f}% over {s['cells']}" if s else ""))
    for k in ("E2", "E3", "E3s", "E4", "E6"):
        v = res.get(k) or {}
        out.append(f"  {k}: {v.get('verdict')}")
    if "E5_printed" in res:
        e5 = res["E5_printed"]
        out.append(f"  E5 (printed): rms M {100 * e5['rms_M']:.2f}%, M+Z {100 * e5['rms_MZ']:.2f}%, sigma_rms {100 * e5['sigma_rms']:.2f}%")
    out.append(f"  deep pages (printed): {res['E1_deep_printed']['reading']}")
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
