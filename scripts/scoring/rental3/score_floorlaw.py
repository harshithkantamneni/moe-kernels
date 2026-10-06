#!/usr/bin/env python
"""Score rental 3's part B, the floor law as CLOCK / CEIL / FLUID
(docs/registered/2026-10-05-rental3-floorlaw-gh200.json): B0 to B7 and the verdict.

    python scripts/scoring/rental3/score_floorlaw.py <repo> <tree> <out>

Reads the floor captures `*-<model>-<label>-r3-counters/r3f-g64[-lock<F>].json` and
the wall unit `*-launch-floor-wall/mixtral-8x7b-tp4/cells.csv` under <tree>. Writes
<out>/floorlaw.score.{json,txt}; exits 0 whatever the verdicts. The cells are the
registration's (`cells_registered`, floor-bound at sigma 1), never re-chosen.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3common as C3  # noqa: E402

PART = "floorlaw"
RULERS = ("ACT.max", "DUR", "EL.avg", "X", "GAP")
STEM = {"base": "r3f-g64", "lock1710": "r3f-g64-lock1710", "lock1005": "r3f-g64-lock1005",
        "lock1410": "r3f-g64-lock1410"}


def f_of(page: dict, rows: list) -> float | None:
    """The DUR ruler's f: the lock for a lock capture; a base capture's the median
    over its cells of n >= 10 of EL.max / duration (printed only)."""
    lock = C3.CM2.page_lock(page)
    if lock is not None:
        return lock
    xs = [r["EL.max"] / r["duration_ns"] * 1e3 for r in rows
          if r["n"] >= 10 and r["EL.max"] and r["duration_ns"]]
    return C3.median(xs) if xs else None


def fit_capture(page: dict, cells_reg: list, gemm: str, ident: dict) -> dict:
    import floor_estimator as FE
    rows0 = C3.floor_rows(page, cells_reg, gemm, None)
    f = f_of(page, rows0)
    rows = C3.floor_rows(page, cells_reg, gemm, f)
    ok, why = C3.identifiable([r["q"] for r in rows], [r["frac"] for r in rows], **ident)
    ent = {"f_ruler": f, "cells": [r["n"] for r in rows], "identifiable": ok, "why": why, "fits": {}}
    if not ok:
        return ent
    for name in RULERS:
        ys = [r[name] for r in rows]
        if any(v is None for v in ys):
            continue
        u = rows[0]["u"]
        ent["fits"][name] = FE.joint_theta([v / u for v in ys], [r["q"] for r in rows], [r["frac"] for r in rows])
    xs = [r["X"] for r in rows if r["X"] is not None]
    ent["X_median"] = C3.median(xs) if xs else None
    ent["u"] = rows[0]["u"] if rows else None
    ent["rows"] = rows
    return ent


def theta(ent, ruler, se_max):
    """(theta, counted?) of a capture's ruler; counted False when se exceeds se_max."""
    if not ent or not ent.get("identifiable") or ruler not in ent.get("fits", {}):
        return None, False
    f = ent["fits"][ruler]
    return f["theta"], f["se_theta"] <= se_max


def wall_fit(reg: dict, tree: Path) -> dict:
    import floor_estimator as FE
    hits = sorted(Path(tree).rglob("*-launch-floor-wall/mixtral-8x7b-tp4/cells.csv"))
    if len(hits) != 1:
        return {"verdict": "NOT RUN: the wall unit's cells.csv is missing"}
    d = hits[0].parent
    man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    if man.get("profiler_enabled_ever"):
        return {"verdict": "NOT SCORED: the profiler was on in the timed process"}
    W = {int(k): v for k, v in reg["tests"]["B6"]["W_ms"].items()}
    out = {}
    for mode in ("GR", "E240"):
        acc = {}
        for r in csv.DictReader(open(hits[0])):
            if r.get("status") == "ok" and r["mode"] == mode and r["arm"] == "native":
                acc.setdefault(int(r["tiles"]), []).append(float(r["ms_p50"]))
        ns = sorted(n for n in acc if n in W)
        if len(ns) < 5:
            out[mode] = {"verdict": f"NOT SCORED: {len(ns)} treads"}
            continue
        T = [C3.median(acc[n]) for n in ns]
        r = C3.corr(ns, [W[n] for n in ns])
        f = FE.ols(T, [[1.0] * len(ns), ns, [W[n] for n in ns]])
        th = f["coef"][2]
        se = math.sqrt(max(f["cov"][2][2], 0.0))
        out[mode] = {"theta_wall": th, "se": se, "corr_n_W": r, "treads": ns, "T_ms": T}
    g = out.get("GR") or {}
    if "theta_wall" not in g:
        out["verdict"] = g.get("verdict", "NOT SCORED")
    elif g["se"] > 0.20 or g["corr_n_W"] is None or abs(g["corr_n_W"]) > 0.95:
        out["verdict"] = "NOT SCORED: theta_wall not resolved (se or collinearity)"
    elif g["theta_wall"] >= 0.70:
        out["verdict"] = "NCU REPLAY ONLY"
    elif g["theta_wall"] <= 0.50:
        out["verdict"] = "IN WALL TIME"
    else:
        out["verdict"] = "INCONCLUSIVE"
    return out


def score_view(repo: Path, tree: Path, fview, tview) -> dict:
    import floor_estimator as FE
    reg = C3.registration(repo, PART)
    ident = {"max_corr": 0.95, "min_cells": 5}
    se_max = 0.15
    caps = {}
    for model, labels in reg["captures"].items():
        for label, spec in labels.items():
            for clk in spec["clocks"]:
                page = fview.load(C3.find_floor(tree, model, label, STEM[clk]))
                for g in ("w1", "w2"):
                    key = f"{model} {label} {clk} {g}"
                    caps[key] = (None if page is None else
                                 fit_capture(page, reg["cells_registered"][model], g, ident))
    res = {"registration": C3.NAMES[PART], "captures": {k: ({kk: vv for kk, vv in v.items() if kk != "rows"}
                                                             if v else {"verdict": "missing"})
                                                         for k, v in caps.items()}, "tests": {}}
    T = res["tests"]
    g1710 = [("granite-3.0-1b-a400m", "floor", "w1"), ("granite-3.0-1b-a400m", "floor", "w2"),
             ("qwen2-57b-a14b-tp8", "floor", "w1"), ("mixtral-8x7b-tp4", "floor", "w1"),
             ("mixtral-8x7b-tp4", "floor", "w2"), ("mixtral-8x7b-tp4", "floorrep", "w1"),
             ("mixtral-8x7b-tp4", "floorrep", "w2")]
    # B0
    b0 = []
    for m, lab, g in g1710:
        th, ok = theta(caps.get(f"{m} {lab} lock1710 {g}"), "X", se_max)
        if th is not None and ok:
            b0.append((f"{m} {lab} {g}", th))
    lo, hi = reg["tests"]["B0"]["band"]
    T["B0"] = ({"verdict": "NOT SCORED: no identifiable 1710 GEMM"} if not b0 else
               {"verdict": "HOLDS" if all(lo <= t <= hi for _, t in b0) else "FALSIFIED", "theta_X": dict(b0)})
    # B1: every lock capture of an identifiable GEMM
    b1 = []
    for key, ent in caps.items():
        if " lock" in key:
            th, ok = theta(ent, "ACT.max", se_max)
            if th is not None and ok:
                b1.append((key, th))
    lo, hi = reg["tests"]["B1"]["band"]
    if b1:
        share = sum(lo <= t <= hi for _, t in b1) / len(b1)
        T["B1"] = {"verdict": "HOLDS" if share >= 0.80 else "FAILS", "share": share, "theta_ACT_max": dict(b1)}
    else:
        T["B1"] = {"verdict": "NOT SCORED"}
    # B2: theta_DUR at 1005
    lo, hi = reg["tests"]["B2"]["band"]
    b2 = {}
    for key, ent in caps.items():
        if " lock1005 " in key:
            th, ok = theta(ent, "DUR", se_max)
            if th is not None and ok:
                b2[key] = {"theta": th, "holds": lo <= th <= hi}
    T["B2"] = {"per_gemm": b2, "verdict": ("NOT SCORED" if not b2 else
                                          "HOLDS" if all(v["holds"] for v in b2.values()) else
                                          f"FAILS on {sum(not v['holds'] for v in b2.values())}")}
    # B3: same-board pairs
    pairs = [("granite-3.0-1b-a400m", "w1"), ("granite-3.0-1b-a400m", "w2"),
             ("mixtral-8x7b-tp4", "w1"), ("mixtral-8x7b-tp4", "w2")]
    b3, clock_ok = {}, True
    for m, g in pairs:
        t5, ok5 = theta(caps.get(f"{m} floor1005 lock1005 {g}"), "DUR", se_max)
        t7, ok7 = theta(caps.get(f"{m} floor lock1710 {g}"), "DUR", se_max)
        if t5 is None or t7 is None or not (ok5 and ok7):
            continue
        dth = t5 - t7
        applies = t7 <= 0.70
        b3[f"{m} {g}"] = {"theta_1005": t5, "theta_1710": t7, "dtheta": dth, "applies": applies,
                          "CLOCK": dth >= 0.25 if applies else None, "CEIL_FLUID": abs(dth) <= 0.20}
        if applies and dth < 0.25:
            clock_ok = False
    printed = {}
    for m, g in (("qwen2-57b-a14b-tp8", "w2"),):
        t5, _ = theta(caps.get(f"{m} floor1005 lock1005 {g}"), "DUR", 9.0)
        t7, _ = theta(caps.get(f"{m} floor lock1710 {g}"), "DUR", 9.0)
        printed[f"{m} {g}"] = {"theta_1005": t5, "theta_1710": t7}
    T["B3"] = {"counted": b3, "printed": printed, "CLOCK_clause": clock_ok,
               "verdict": "RECORD" if b3 else "NOT SCORED"}
    # B4
    gem4 = [("granite-3.0-1b-a400m", "w1"), ("granite-3.0-1b-a400m", "w2"), ("mixtral-8x7b-tp4", "w1"),
            ("mixtral-8x7b-tp4", "w2"), ("qwen2-57b-a14b-tp8", "w1")]
    b4 = {}
    for m, g in gem4:
        ent = caps.get(f"{m} floor lock1710 {g}")
        td, okd = theta(ent, "DUR", se_max)
        ta, oka = theta(ent, "ACT.max", se_max)
        if td is None or ta is None or not (okd and oka):
            continue
        b4[f"{m} {g}"] = {"theta_DUR": td, "theta_ACT_max": ta,
                          "touch": ta > 0.90 and 0.70 <= td <= ta - 0.20}
    counted = {k: v for k, v in b4.items() if not v["touch"]}
    def _clk(v):
        return v["theta_DUR"] <= v["theta_ACT_max"] - 0.20
    # at least one BLIND shape must meet it (decided before any rental-3 page)
    clock4 = (sum(_clk(v) for v in counted.values()) >= 2
              and any(k.startswith(("granite-3.0-1b-a400m", "qwen2-57b-a14b-tp8")) and _clk(v)
                      for k, v in counted.items()))
    ceil4 = bool(counted) and all(v["theta_DUR"] >= 0.70 for v in counted.values())
    T["B4"] = {"per_gemm": b4, "CLOCK_clause": clock4, "CEIL_clause": ceil4,
               "verdict": "RECORD" if counted else "NOT SCORED"}
    # B5 printed
    T["B5_printed"] = {k: v.get("X_median") for k, v in caps.items() if v and " lock1005 " in k}
    # B6
    T["B6"] = wall_fit(reg, tree)
    # B7: part 4 re-asked on tp8 w1's 1005 DUR
    ent = caps.get("mixtral-8x7b-tp8 floor1005 lock1005 w1")
    sets = reg["tests"]["B7"]["sets"]
    if ent and ent.get("rows"):
        rows = {r["n"]: r for r in ent["rows"]}
        cells_reg = {c["n"]: c for c in reg["cells_registered"]["mixtral-8x7b-tp8"] if c["gemm"] == "w1"}
        pc = {n: (cells_reg[n]["grid"], rows[n]["DUR"]) for n in rows if rows[n]["DUR"] is not None}
        ks = cells_reg[next(iter(cells_reg))]["ksteps"]
        try:
            sl = [{"slope": FE.implied_slope(pc, v["treads"], ks), "asymptote": reg["tests"]["B7"]["asymptote"],
                   "bias": v["bias"]} for v in sets.values()]
            td = FE.theta_delta(sl)
            th = td["theta"]
            v = ("H_EST HOLDS" if th > 0.5 and abs(th - 1) <= 0.20 else "FLUID HOLDS" if abs(th) <= 0.20 else "INCONCLUSIVE")
            T["B7"] = {"verdict": v, **td}
        except (KeyError, ValueError) as exc:
            T["B7"] = {"verdict": f"NOT SCORED: {exc}"}
    else:
        T["B7"] = {"verdict": "NOT SCORED: the tp8 1005 capture is missing"}
    # X's clock law, printed
    xl = {}
    for g in ("w1", "w2"):
        x = {c: (caps.get(f"mixtral-8x7b-tp4 {lab} {c} {g}") or {}).get("X_median")
             for lab, c in (("floor", "lock1710"), ("floor1005", "lock1005"), ("floor1410", "lock1410"), ("floor", "base"))}
        if None not in (x["lock1710"], x["lock1005"], x["lock1410"]):
            lin = x["lock1005"] + (405 / 705) * (x["lock1710"] - x["lock1005"])
            x["LIN_centre"] = lin
            x["reads"] = ("LIN" if abs(x["lock1410"] - lin) <= 0.35 * abs(lin) else
                          "HIGH" if x["lock1410"] <= 0.25 * x["lock1710"] else
                          "FLAT" if x["lock1410"] >= 0.80 * x["lock1710"] else "none")
        xl[g] = x
    res["X_clock_law_printed"] = xl
    # the verdict
    b1v = T["B1"]["verdict"]
    b2v = [v["holds"] for v in b2.values()]
    b2_all = bool(b2v) and all(b2v)
    b2_fail = sum(not x for x in b2v)
    st = {}
    if b1v == "NOT SCORED" or not b2v or not counted:
        res["verdict"] = "NOT SCORED: B1, B2 or B4 has no data"
        res["hypotheses"] = st
        return res
    clock_holds = (b1v == "HOLDS" and b2_all and clock_ok and clock4 and T["B0"]["verdict"] == "HOLDS")
    st["CLOCK"] = "HOLDS" if clock_holds else ("FALSIFIED" if b2_fail >= 2 or ceil4 else "neither")
    st["CEIL"] = "HOLDS" if ceil4 and b2_all else ("FALSIFIED" if not ceil4 else "neither")
    dur_all = [v["theta"] for v in b2.values()] + [v["theta_DUR"] for v in counted.values()]
    if b1v == "HOLDS" or sum(b2v) >= 2:
        st["FLUID"] = "FALSIFIED"
    else:
        st["FLUID"] = "HOLDS" if all(t <= 0.30 for t in dur_all) else "neither"
    sel = [h for h, v in st.items() if v == "HOLDS" and all(st[o] == "FALSIFIED" for o in st if o != h)]
    res["hypotheses"] = st
    v = sel[0] if sel else "INCONCLUSIVE"
    b6 = T["B6"]["verdict"]
    if v == "CLOCK":
        v = ("CLOCK (in wall time)" if b6 == "IN WALL TIME" else "CLOCK (in the ncu replay only)" if b6 == "NCU REPLAY ONLY"
             else "CLOCK (ncu-measured)")
    res["verdict"] = v
    return res


def score(repo: Path, tree: Path) -> dict:
    reg = C3.registration(repo, PART)
    return C3.two_views(lambda fv, tv: score_view(repo, tree, fv, tv),
                        floor_rules=reg["gates"]["floor"], repo=repo)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 3 PART B (the floor law), {res['registration']}"]
    for k, v in res["tests"].items():
        out.append(f"  {k}: {v.get('verdict') if isinstance(v, dict) and 'verdict' in v else v}")
    out.append(f"  hypotheses: {res.get('hypotheses')}")
    out.append(f"  verdict: {res['verdict']}")
    for k, v in res["captures"].items():
        if v and v.get("fits"):
            out.append(f"  {k}: " + "  ".join(f"{r} {f['theta']:+.2f}+-{f['se_theta']:.2f}" for r, f in v["fits"].items()))
        elif v and "why" in v:
            out.append(f"  {k}: {v['why']}")
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
