#!/usr/bin/env python
"""Reads model v7's decode domain map (registration docs/registered/2026-11-18-v7-domain-map-gh200) on a results tree
and applies the registered threshold-selection rule mechanically. DESCRIPTIVE: a measurement, not a certification.

    python scripts/scoring/v7domain/read_map.py <results tree> <out dir>

It reads the registration at its committed path only (no option names another file), verifies the freeze (v7 as the
v7 gate froze it, and the map's own files: pages, code, launcher, driver) and the plan's registered sha256, and records
the registration's sha256 and the checkout's commit in its output. Then, as the registration's rule and its points say:
the power check (700 W) first; each page counted by dcommon.decode_page against the registered page and counts sha256;
each valid cell's e = dcommon.e_dec(T_v7, T_meas) and e_ub from its clean repeats; per path X_p by the rule with the
leave-one-shape-out maximum; the slab guard on family B; the domain D. Beside it, DESCRIPTIVE only: v7's error on the
map per path and family (out of sample on new shapes, not a certification), the aligned-step over-price by E and the
slab sweep. Writes domain.json and domain.txt.

Exit: 0 read (D written, possibly empty); 1 NOT READ (power check); 2 REFUSED (freeze, plan or registration)."""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7mcommon as M  # noqa: E402

import replicate_noise_floor as RNF  # noqa: E402

DC, D2 = M.DC, M.D2
C3, C7 = DC.C3, DC.C7
UNIT_LABEL, INST = "dm", "GRR"


# ================================================================== the rule (pure: lists of cell dicts)

def e_ub(pred_us: float, meas_us: float, s_us: float | None, n: int) -> float:
    """The registered bound: max |T_v7 / T - 1| over T = T_meas -/+ 2.306 x 1.2533 x s / sqrt(n)."""
    if s_us is None or n < 2:
        return math.inf
    h = M.T975_8 * M.MEDIAN_SE * s_us / math.sqrt(n)
    lo, hi = meas_us - h, meas_us + h
    if lo <= 0:
        return math.inf
    return max(abs(pred_us / lo - 1), abs(pred_us / hi - 1))


def summary(cells: list[dict]) -> dict:
    e = [c["e"] for c in cells]
    if not e:
        return {"cells": 0, "rms": None, "mean": None, "worst": None, "max_e_ub": None}
    return {"cells": len(e), "rms": math.sqrt(sum(x * x for x in e) / len(e)), "mean": st.fmean(e),
            "worst": max(e, key=abs), "max_e_ub": max(c["e_ub"] for c in cells)}


def passes(cells: list[dict]) -> bool:
    s = summary(cells)
    return (s["cells"] > 0 and s["max_e_ub"] <= M.E_UB_MAX and s["rms"] <= M.RMS_MAX
            and abs(s["mean"]) <= M.MEAN_MAX)


def threshold(cells: list[dict], key: str = "pred_us") -> dict:
    """X(set): the smallest counted candidate X (its set {key >= X} holds at least MIN_CELLS cells) such that every
    counted candidate X' >= X passes (i) to (iii). Returns {X, candidates: [[X, cells, passes]]}."""
    cands = sorted({c[key] for c in cells})
    rows = []
    for x in cands:
        sub = [c for c in cells if c[key] >= x]
        if len(sub) < M.MIN_CELLS:
            break
        rows.append([x, len(sub), passes(sub)])
    X = None
    for i in range(len(rows)):
        if all(r[2] for r in rows[i:]):
            X = rows[i][0]
            break
    return {"X": X, "candidates": rows}


def loso(cells: list[dict], shapes, key: str = "pred_us") -> dict:
    """(iv): the rule on the set and with each shape's cells removed in turn; the largest value, none when any is
    none."""
    full = threshold(cells, key)["X"]
    folds = {s: threshold([c for c in cells if c["model"] != s], key)["X"] for s in shapes}
    vals = [full, *folds.values()]
    X = None if any(v is None for v in vals) else max(vals)
    mover = sorted(s for s, v in folds.items() if v is not None and full is not None and v > full)
    return {"X": X, "full": full, "folds": folds, "shapes that raise X when removed": mover,
            "none in": sorted(s for s, v in folds.items() if v is None) + (["(full set)"] if full is None else [])}


def ols(y: list[float], cols: dict[str, list[float]]) -> dict:
    """Ordinary least squares with an intercept; a column constant over the rows is left out. b's 95% interval."""
    import numpy as np
    names = [k for k, v in cols.items() if len(set(v)) > 1]
    dropped = [k for k in cols if k not in names]
    n, p = len(y), 1 + len(names)
    if n - p < 1 or "W_e_MB" not in names:
        return {"verdict": "NOT FITTED", "n": n, "p": p, "dropped": dropped,
                "why": "fewer cells than coefficients plus one" if n - p < 1 else "W_e_MB constant over the cells"}
    X = np.column_stack([np.ones(n)] + [np.asarray(cols[k], float) for k in names])
    Y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    df = n - p
    s2 = float(res @ res) / df
    cov = s2 * np.linalg.inv(X.T @ X)
    j = 1 + names.index("W_e_MB")
    b, se = float(beta[j]), math.sqrt(max(float(cov[j, j]), 0.0))
    t = RNF.student_t_ppf(0.975, df)
    return {"verdict": "FITTED", "n": n, "p": p, "df": df, "dropped": dropped, "coef": dict(zip(["a", *names],
            [float(x) for x in beta])), "b_us_per_MB": b, "se_b": se, "t975": t, "b_ci95": [b - t * se, b + t * se]}


def slab_guard(cells: list[dict]) -> dict:
    """Family B's valid in-scope cells under a wave: resid_us = a + b W_e_MB + c 1[T=2]; triggers when b's 95%
    interval excludes 0 and |b| x 35 MB >= 1 us."""
    B = [c for c in cells if c["family"] == "B" and c["waves"] < 1]
    f = ols([c["pred_us"] - c["meas_us"] for c in B],
            {"W_e_MB": [c["W_e_MB"] for c in B], "T2": [1.0 if c["T"] == 2 else 0.0 for c in B]})
    trig = False
    if f["verdict"] == "FITTED":
        lo, hi = f["b_ci95"]
        excl = lo > 0 or hi < 0
        big = abs(f["b_us_per_MB"]) * M.SLAB_SPAN_MB >= M.SLAB_MIN_US
        trig = excl and big
        f.update({"interval excludes 0": excl, "|b| x 35 MB (us)": abs(f["b_us_per_MB"]) * M.SLAB_SPAN_MB,
                  "|b| x 35 MB >= 1 us": big})
    return {"fit": f, "cells": len(B), "triggered": trig,
            "verdict": "TRIGGERED: D adds W_e >= S_p" if trig else "NOT TRIGGERED: D has no slab condition"}


def domain(cells: list[dict], shapes=None) -> dict:
    """The registered rule on the valid in-scope cells: X_p per path, the slab guard, S_p when it triggers, D."""
    shapes = list(M.MS.NAMES if shapes is None else shapes)
    use = [c for c in cells if c["in_scope"]]
    per = {}
    for p in M.PATHS:
        cp = [c for c in use if c["path"] == p]
        r = loso(cp, shapes)
        per[p] = {"cells": len(cp), "X_p_us": r["X"], "rule": r,
                  "verdict": "IN THE DOMAIN" if r["X"] is not None else "OUTSIDE THE DOMAIN (no X_p)"}
    guard = slab_guard(use)
    for p, d in per.items():
        d["S_p_MB"] = None
        if guard["triggered"] and d["X_p_us"] is not None:
            sub = [c for c in use if c["path"] == p and c["pred_us"] >= d["X_p_us"]]
            s = loso(sub, shapes, key="W_e_MB")
            d["slab rule"] = s
            d["S_p_MB"] = s["X"]
            if s["X"] is None:
                d["verdict"] = "OUTSIDE THE DOMAIN (the slab guard triggered and no S_p)"
    D = []
    for p, d in per.items():
        if d["verdict"] != "IN THE DOMAIN":
            continue
        cond = f"path {p}, T_v7 >= {d['X_p_us']:.3f} us"
        if d["S_p_MB"] is not None:
            cond += f", W_e >= {d['S_p_MB']:.6f} MB"
        D.append(cond + ", every active expert in one M-tile")
    return {"per_path": per, "slab_guard": guard, "D": D,
            "statement": ("D = {decode calls with " + "; or ".join(D) + "}") if D else "D is EMPTY: no path has an X_p"}


# ================================================================== reading the tree

def cell_rows(tree: Path, reg: dict) -> tuple[list[dict], dict]:
    """(every registered cell with its reading, per-page status). A cell's 'valid' is the registration's rule."""
    pred = {(c["model"], c["label"]): c for c in reg["predictions"]["cells"]}
    files = reg["cells"]["pages"]
    out, pages = [], {}
    with M.MS.installed():
        for m in M.MS.NAMES:
            rf = files[m]
            rb, why = DC.decode_page(Path(tree), m, UNIT_LABEL, rf, INST)
            pages[m] = why
            valid, dropped, reps = {}, {}, {}
            if rb is not None:
                v, dr = DC.measured(rb, m, "main")
                valid = {k[2]: row for k, row in v.items() if k[4] == "native"}
                dropped = {x[0].split("/")[2]: x[1] for x in dr}
                root = C3.find_gaps(Path(tree), m, UNIT_LABEL)
                pg = [p for p, r in C3.timed_pages(root) if r.get("kind") == "decode-page"]
                _rep, raw = C7.read_page(Path(pg[0]).parent)
                lock = rb.get("lock_mhz")
                for (lab, _T, arm), rs in C7.repeat_rows(raw or []).items():
                    if arm == "native":
                        reps[lab] = [r["ms"] * 1e3 for r in rs if C7.clean_r7(r, lock)]
            doc = {x["label"]: x for x in json.loads((M.REPO / rf["path"]).read_text())["cells"]}
            for rc in rf["cells"]:
                c = dict(pred[(m, rc["label"])])
                c["in_scope"] = bool(c["in_scope"]) and D2.second_tile(m, doc[rc["label"]]) is None
                row = valid.get(rc["label"])
                if row is None:
                    c.update(valid=False, why=dropped.get(rc["label"]) or (why if rb is None else "no row"))
                    out.append(c)
                    continue
                r = reps.get(rc["label"]) or []
                meas = row["ms_p50"] * 1e3
                n = len(r)
                s = st.stdev(r) if n > 1 else None
                c.update(valid=True, why=None, meas_us=meas, e=DC.e_dec(c["pred_ms"], row["ms_p50"]), n=n, s_us=s,
                         e_ub=e_ub(c["pred_us"], meas, s, n), clock_mhz=row.get("sm_clock_load_mhz"),
                         median_check=(abs(st.median(r) - meas) <= 1e-9 * max(1.0, meas)) if r else False)
                out.append(c)
    return out, pages


# ================================================================== DESCRIPTIVE tables

def aligned_step(cells: list[dict]) -> list[dict]:
    """Per shape, each aligned twin (MIN-T*) against its naive base (L-T1-*, the same GEMMs): v7's step minus the
    measured step, in us (positive: v7 over-prices the path step)."""
    v = {(c["model"], c["label"]): c for c in cells if c.get("valid")}
    out = []
    for m in M.MS.NAMES:
        base = next((c for (mm, lab), c in v.items() if mm == m and lab.startswith("L-T1-")), None)
        if base is None:
            continue
        for (mm, lab), c in sorted(v.items(), key=lambda kv: kv[1]["T"]):
            if mm != m or not lab.startswith("MIN-T"):
                continue
            dp, dm = c["pred_us"] - base["pred_us"], c["meas_us"] - base["meas_us"]
            out.append({"model": m, "family": c["family"], "E": c["E"], "label": lab, "path": c["path"], "T": c["T"],
                        "step_pred_us": dp, "step_meas_us": dm, "over_us": dp - dm, "base_e": base["e"], "e": c["e"]})
    return out


def by_E(steps: list[dict]) -> list[dict]:
    g = {}
    for s in steps:
        g.setdefault((s["E"], s["path"]), []).append(s["over_us"])
    return [{"E": E, "path": p, "pairs": len(v), "median_over_us": st.median(v), "min": min(v), "max": max(v)}
            for (E, p), v in sorted(g.items())]


def slab_sweep(cells: list[dict]) -> list[dict]:
    return [{k: c.get(k) for k in ("model", "label", "T", "path", "tile", "W_e_MB", "waves", "pred_us", "meas_us", "e",
                                   "e_ub", "why")} | {"resid_us": (c["pred_us"] - c["meas_us"]) if c.get("valid") else None}
            for c in sorted((c for c in cells if c["family"] == "B"), key=lambda c: (c["W_e_MB"], c["pred_us"]))]


def descriptive(cells: list[dict]) -> dict:
    v = [c for c in cells if c.get("valid") and c["in_scope"]]
    st_ = aligned_step(cells)
    return {"label": "DESCRIPTIVE (v7 out of sample on new synthetic shapes; not a certification, no bar)",
            "pooled": summary(v),
            "per path": {p: summary([c for c in v if c["path"] == p]) for p in M.PATHS},
            "per family": {f: summary([c for c in v if c["family"] == f]) for f in "ABC"},
            "aligned step over-price by E (us)": by_E(st_), "aligned step pairs": st_, "slab sweep": slab_sweep(cells)}


# ================================================================== the readout

def refusals(reg: dict) -> list[str]:
    if reg.get("name") != M.NAME:
        return [f"the registration is not {M.NAME}"]
    bad = M.verify_freeze(reg)
    if bad:
        return ["the freeze does not hold: " + "; ".join(bad[:10])]
    plan = M.REPO / M.PLAN
    if not plan.is_file() or M.sha256(plan) != reg["rental"]["plan_sha256"]:
        return [f"{M.PLAN} is not the registered plan"]
    return []


def read(tree: Path, reg: dict) -> dict:
    power = D2.power_check(Path(tree))
    if power["verdict"] != "READ":
        return {"what": "model v7's decode domain map (DESCRIPTIVE)", "registration": M.NAME, "verdict": "NOT READ",
                "why": f"power: {power['verdict']}", "POWER": power}
    cells, pages = cell_rows(Path(tree), reg)
    valid = [c for c in cells if c["valid"]]
    dom = domain(valid)
    return {"what": "model v7's decode domain map (DESCRIPTIVE)", "registration": M.NAME, "verdict": "READ",
            "POWER": power, "pages": pages, "cells": {"registered": len(cells), "valid": len(valid),
                                                      "valid in scope": sum(c["in_scope"] for c in valid)},
            "rule": reg["rule"]["text"], "domain": dom, "descriptive": descriptive(cells), "rows": cells}


def _f(x, fmt):
    return "-" if x is None else format(x, fmt)


def lines(r: dict) -> list[str]:
    L = [f"MODEL v7 DECODE DOMAIN MAP ({M.NAME}): {r['verdict']}, DESCRIPTIVE (a measurement, not a certification)"]
    if r["verdict"] != "READ":
        return L + [r["why"]]
    L.append(f"power: {r['POWER']['verdict']} ({r['POWER'].get('gpu_current_power_limit_w')} W); cells "
             f"{r['cells']['valid']} valid of {r['cells']['registered']} ({r['cells']['valid in scope']} in scope)")
    for m, why in r["pages"].items():
        if why != "counted":
            L.append(f"  page {m}: {why}")
    d = r["domain"]
    L += ["", "DOMAIN (the registered rule, mechanically):", f"  {d['statement']}"]
    for p, x in d["per_path"].items():
        rr = x["rule"]
        L.append(f"  {p:11s} {x['cells']:3d} cells: X_p {_f(x['X_p_us'], '.3f')} us (full set {_f(rr['full'], '.3f')}; "
                 f"raised by removing {rr['shapes that raise X when removed'] or 'none'}; none in "
                 f"{rr['none in'] or 'no fold'}); S_p {_f(x['S_p_MB'], '.3f')} MB; {x['verdict']}")
    g, f = d["slab_guard"], d["slab_guard"]["fit"]
    L.append(f"  slab guard: {g['verdict']} ({g['cells']} family B cells under a wave; "
             + (f"b {f['b_us_per_MB']:+.4f} us/MB, 95% [{f['b_ci95'][0]:+.4f}, {f['b_ci95'][1]:+.4f}], |b| x 35 MB "
                f"{f['|b| x 35 MB (us)']:.2f} us" if f["verdict"] == "FITTED" else f"NOT FITTED: {f['why']}") + ")")
    s = r["descriptive"]
    L += ["", s["label"]]
    for name, x in [("pooled", s["pooled"]), *s["per path"].items(), *((f"family {k}", v) for k, v in s["per family"].items())]:
        L.append(f"  {name:13s} {x['cells']:3d} cells rms {_f(x['rms'], '.2%')} worst {_f(x['worst'], '+.2%')} "
                 f"mean {_f(x['mean'], '+.2%')} max e_ub {_f(x['max_e_ub'], '.2%')}")
    L.append("  aligned step over-price by E (v7's naive-to-aligned step minus the measured one, same GEMMs):")
    for x in s["aligned step over-price by E (us)"]:
        L.append(f"    E {x['E']:4d} {x['path']:11s} {x['pairs']} pairs: median {x['median_over_us']:+.2f} us "
                 f"({x['min']:+.2f} to {x['max']:+.2f})")
    L.append("  slab sweep (family B): W_e MB / label / T / path / waves / v7 us / measured us / resid us / e / e_ub")
    for x in s["slab sweep"]:
        L.append(f"    {x['W_e_MB']:6.2f} {x['label']:11s} T{x['T']:<4d} {x['path']:11s} {x['waves']:5.2f} "
                 f"{x['pred_us']:8.2f} {_f(x['meas_us'], '8.2f')} {_f(x['resid_us'], '+7.2f')} {_f(x['e'], '+.2%')} "
                 f"{_f(x['e_ub'], '.2%')}" + (f"  ({x['why']})" if x["why"] else ""))
    return L


def provenance(path: Path) -> dict:
    head = subprocess.run(["git", "-C", str(M.REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(M.REPO), "status", "--porcelain"], capture_output=True, text=True)
    return {"registration_path": M.REG_JSON, "registration_sha256": M.sha256(path),
            "git_head": head.stdout.strip() if head.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tree", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    path = M.REPO / M.REG_JSON
    reg = json.loads(path.read_text())
    why = refusals(reg)
    if why:
        print("REFUSED: " + "; ".join(why))
        return 2
    r = dict(read(a.tree, reg), provenance=provenance(path))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "domain.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    L = lines(r) + [f"registration sha256 {r['provenance']['registration_sha256']}, commit {r['provenance']['git_head']}"]
    (a.out / "domain.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if r["verdict"] == "READ" else 1


if __name__ == "__main__":
    sys.exit(main())
