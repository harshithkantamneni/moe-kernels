#!/usr/bin/env python
"""Score rental 3's noise models (docs/registered/2026-10-05-rental3-replicate-gh200):
part R (sigma_page from the rep8 timed replicate, beside the SEEN sigma_board) and RK
(the tp4 floorrep capture: a same-board noise floor for rental 2's K1 and K4).

    python scripts/scoring/rental3/score_replicate.py <repo> <tree> <out>

Writes <out>/replicate.score.{json,txt}; exits 0 whatever it finds. No verdict of a
model is made here: R feeds E1's UNRESOLVED label, RK says whether K1 / K4 are
RESOLVED at their own noise.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3common as C3  # noqa: E402

PART = "replicate"


def timed_cells(tree: Path, label: str, view, *, tag_part: str | None = None, groups=None) -> dict:
    """{(arm, G, n): median ms_p50} over the unit's R3 pages `view` counts, restricted
    to session tags containing `tag_part` and to `groups` when given."""
    root = C3.find_gaps(tree, C3.TARGET, label)
    acc: dict = {}
    for path, rep in C3.timed_pages(root):
        tag = str(rep.get("session_tag") or "")
        if tag_part is not None and tag_part not in tag:
            continue
        G = C3.group_of(rep)
        if groups is not None and G not in groups:
            continue
        rep = view.admit(f"{label}/{path.parent.name}", rep)
        if rep is None:
            continue
        for (arm, n), ms in C3.treads_ms(rep).items():
            acc.setdefault((arm, G, n), []).append(ms)
    return {k: C3.median(v) for k, v in acc.items()}


def sigma_page(repo: Path, tree: Path, view) -> dict:
    """R's sigma_page: rep8 against the e2e timed step's G = 8 page (tag -p2-)."""
    reg = C3.registration(repo, PART)["R"]
    rep = timed_cells(tree, "rep8", view, groups={8})
    e2e = timed_cells(tree, "e2e", view, tag_part="-p2-", groups={8})
    rows = []
    for key in reg["sigma_page"]["cells"]:
        arm, g, n = key.split()
        k = (arm, int(g.split("=")[1]), int(n.split("=")[1]))
        if k in rep and k in e2e:
            rows.append({"cell": key, "rep8_ms": rep[k], "e2e_ms": e2e[k], "rel": rep[k] / e2e[k] - 1})
    if len(rows) < 3:
        return {"verdict": "NOT SCORED: fewer than 3 cells on both pages", "cells": rows}
    s = math.sqrt(sum(r["rel"] ** 2 for r in rows) / len(rows))
    return {"cells": rows, "sigma_page": s, "verdict": "RECORD"}


def k1_resolution(L, sig, thr):
    """L: [(L, gemm)]; sig: {gemm: sigma}. K1 FALSIFIED when > 20% of L exceed thr."""
    def verdict(flags):
        return "FALSIFIED" if sum(flags) / len(flags) > 0.20 else "HOLDS"
    lo = [(l > thr + 2 * sig[g]) for l, g in L]          # ambiguous cells counted under
    hi = [(l > thr - 2 * sig[g]) for l, g in L]          # ambiguous cells counted over
    a, b = verdict(lo), verdict(hi)
    amb = sum(abs(l - thr) <= 2 * sig[g] for l, g in L)
    return {"cells": len(L), "ambiguous": amb, "verdict_as_measured": verdict([l > thr for l, g in L]),
            "extremes": [a, b], "verdict": "RESOLVED" if a == b else "UNRESOLVED"}


def k4_resolution(D, sig):
    """D: [(D_imb, u, gemm)]. K4: FALSIFIED when >= 0.3 u on > 20%; HOLDS when
    within +-0.2 u on >= 80%; else INCONCLUSIVE."""
    def verdict(within, big):
        n = len(within)
        if sum(big) / n > 0.20:
            return "FALSIFIED (H_IMB)"
        return "HOLDS" if sum(within) / n >= 0.80 else "INCONCLUSIVE"
    best_w = [abs(d) <= 0.2 * u + 2 * sig[g] for d, u, g in D]
    best_b = [d >= 0.3 * u + 2 * sig[g] for d, u, g in D]
    worst_w = [abs(d) <= 0.2 * u - 2 * sig[g] for d, u, g in D]
    worst_b = [d >= 0.3 * u - 2 * sig[g] for d, u, g in D]
    a, b = verdict(best_w, best_b), verdict(worst_w, worst_b)
    meas = verdict([abs(d) <= 0.2 * u for d, u, g in D], [d >= 0.3 * u for d, u, g in D])
    return {"cells": len(D), "verdict_as_measured": meas, "extremes": [a, b],
            "verdict": "RESOLVED" if a == b else "UNRESOLVED"}


def rk(repo: Path, tree: Path, fview) -> dict:
    reg = C3.registration(repo, PART)["RK"]
    flr = C3.registration(repo, "floorlaw")
    cells = flr["cells_registered"]["mixtral-8x7b-tp4"]
    pa = fview.load(C3.find_floor(tree, "mixtral-8x7b-tp4", "floor", "r3f-g64-lock1710"))
    pb = fview.load(C3.find_floor(tree, "mixtral-8x7b-tp4", "floorrep", "r3f-g64-lock1710"))
    if pa is None or pb is None:
        return {"verdict": "NOT RUN: a tp4 1710 capture of the pair is missing or out of this view"}
    out = {"per_gemm": {}}
    sig = {"L": {}, "D_imb": {}}
    pool = {"L": [], "D": []}
    for g in ("w1", "w2"):
        ra = {r["n"]: r for r in C3.floor_rows(pa, cells, g, 1710.0)}
        rb = {r["n"]: r for r in C3.floor_rows(pb, cells, g, 1710.0)}
        ns = sorted(set(ra) & set(rb))
        dl, dd = [], []
        for n in ns:
            for src in (ra[n], rb[n]):
                if src["GAP"] is not None:
                    L = src["EL.avg"] - src["ACT.max"]
                    D = src["ACT.max"] - src["ACT.avg"] - src["frac"] * src["u"]
                    pool["L"].append((L, g))
                    pool["D"].append((D, src["u"], g))
            a, b = ra[n], rb[n]
            if a["ACT.max"] is None or b["ACT.max"] is None:
                continue
            dl.append((a["EL.avg"] - a["ACT.max"]) - (b["EL.avg"] - b["ACT.max"]))
            dd.append((a["ACT.max"] - a["ACT.avg"]) - (b["ACT.max"] - b["ACT.avg"]))
        if len(dl) < 3:
            out["per_gemm"][g] = {"verdict": "NOT SCORED: fewer than 3 common cells"}
            continue
        sig["L"][g] = math.sqrt(sum(x * x for x in dl) / len(dl)) / math.sqrt(2)
        sig["D_imb"][g] = math.sqrt(sum(x * x for x in dd) / len(dd)) / math.sqrt(2)
        out["per_gemm"][g] = {"cells": ns, "sigma_L": sig["L"][g], "sigma_D_imb": sig["D_imb"][g]}
    if len(sig["L"]) < 2:
        out["verdict"] = "NOT SCORED: a GEMM has no noise floor"
        return out
    thr = float(reg["K1_threshold"])
    out["rental3_tp4_K1"] = k1_resolution(pool["L"], sig["L"], thr)
    out["rental3_tp4_K4"] = k4_resolution(pool["D"], sig["D_imb"])
    r2 = Path(repo) / "scripts" / "scoring" / "rental2" / "const.score.json"
    if r2.exists():
        caps = json.loads(r2.read_text())["captures"]
        L2, D2 = [], []
        for m in ("mixtral-8x7b-tp8", "mixtral-8x7b-tp4", "mixtral-8x7b-tp2"):
            ent = caps.get(f"{m} lock1710") or {}
            for g in ("w1", "w2"):
                for r in (ent.get(g) or {}).get("rows", []):
                    if r.get("L") is not None:
                        L2.append((r["L"], g))
                    if r.get("D_imb") is not None:
                        D2.append((r["D_imb"], r["u"], g))
        if L2:
            out["rental2_K1_SEEN"] = k1_resolution(L2, sig["L"], thr)
        if D2:
            out["rental2_K4_SEEN"] = k4_resolution(D2, sig["D_imb"])
    out["verdict"] = "RECORD"
    return out


def score_view(repo: Path, tree: Path, fview, tview) -> dict:
    return {"registration": C3.NAMES[PART], "R": sigma_page(repo, tree, tview),
            "RK": rk(repo, tree, fview)}


def score(repo: Path, tree: Path) -> dict:
    flr = C3.registration(repo, "floorlaw")
    return C3.two_views(lambda fv, tv: score_view(repo, tree, fv, tv),
                        floor_rules=flr["gates"]["floor"], timed_ignore=("V5",), repo=repo)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 3, the noise models, {res['registration']}"]
    R = res["R"]
    out.append(f"  R sigma_page: {R.get('sigma_page', R.get('verdict'))}")
    for r in R.get("cells", []):
        out.append(f"    {r['cell']}: rep8 {r['rep8_ms']:.4f} e2e {r['e2e_ms']:.4f} ({100 * r['rel']:+.2f}%)")
    K = res["RK"]
    out.append(f"  RK: {K.get('verdict')}")
    for g, v in (K.get("per_gemm") or {}).items():
        out.append(f"    {g}: {v}")
    for k in ("rental3_tp4_K1", "rental3_tp4_K4", "rental2_K1_SEEN", "rental2_K4_SEEN"):
        if k in K:
            out.append(f"    {k}: {K[k]['verdict']} (as measured {K[k]['verdict_as_measured']}, extremes {K[k]['extremes']})")
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
