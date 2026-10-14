#!/usr/bin/env python
"""Score rental B (docs/registered/2026-10-14-tileB-e1tile-gh200.json) against its finalised
predictions (finalise.py's file): E1-TILE (primary, NATIVE, pooled and each BLIND-TILE model, the
CLEAN certification), E1-TILE-S (SHARED, secondary), E2-TILE and B-BYTES (byte pages), G3, the
rivals and the split views.

    python scripts/scoring/tileA/score_tileB.py <repo> <tree> <out> <predictions.json>

Writes <out>/tileB.score.{json,txt}; exits 0 whatever the verdicts. Every prediction, band and sha
is the registration's or the finalised file's; nothing is re-chosen on a page.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import finalise as FN  # noqa: E402
import regtile as RG  # noqa: E402
import tcommon as TC  # noqa: E402

PART = "tileB"
r6 = TC.r6
E2_BAR = dict(TC.C6.E2_BAR)
B_BYTES = dict(TC.C6.B_SKEW)


def load_preds(path: Path) -> tuple[dict, dict]:
    doc = json.loads(Path(path).read_text())
    preds = {(r["model"], r["page"], int(r["n"]), r["label"], r["arm"]): dict(r, ms_all=r["ms"], ms=r["ms"]["v4"])
             for r in doc["cells"]}
    return doc, preds


def census_check(repo: Path, tree: Path, doc: dict) -> dict:
    """Ambiguity 11, rental B's half: rental B's own census (label tb, compile only) against rental A's
    occupancy that v4 read (the finalised selections). A (model, tile) whose CTAs per SM differ on
    either GEMM is NOT SCORED on every page of that (model, tile); a missing or refused rental-B census
    leaves the occupancy unconfirmed, and every page is NOT SCORED."""
    import score_census as SC
    cb = SC.score(repo, tree, "tb")
    v = str(cb["CENSUS"]["verdict"])
    if not v.startswith("VALID"):
        return {"verdict": f"NOT SCORED: rental B's census reads {v}", "excluded": "ALL"}
    sel = doc["selections"]
    excluded, rows = [], {}
    for u in RG.B_TIMED_PAGES + RG.B_BYTE_PAGES:
        model, name = u[1], u[2]
        t = RG.PAGE_TILE[int(RG.hist("B", name)["block_m"])]
        a = ((sel.get("occupancy_by_model") or {}).get(model) or {}).get(t.key) or (sel.get("occupancy") or {}).get(t.key) \
            or {"w1": t.occ_shared(), "w2": t.occ_shared()}
        b = (cb["occupancy_by_model"].get(model) or {}).get(t.key) or {}
        same = all(b.get(g) == a.get(g) for g in ("w1", "w2"))
        rows[f"{model}|{t.key}"] = {"rental_A": a, "rental_B": b, "same": same}
        if not same and (model, t.key) not in excluded:
            excluded.append((model, t.key))
    return {"verdict": "CHECKED", "rows": rows, "excluded": excluded}


def score_view(tree: Path, reg: dict, preds: dict, sigma: float, tview) -> dict:
    files = reg["pages"]["files"]
    meas, A_of, use, dropped, in_view = {}, {}, {}, {}, {}
    for u in RG.B_TIMED_PAGES:
        label, model, name = u[:3]
        f = files[f"{name}.json"]
        rep, why = TC.timed_page(tree, model, label, f["group_m"], tview, f)
        use[f"{model}-{label}"] = why
        if rep is None:
            continue
        sub = {k: v for k, v in preds.items() if k[0] == model and k[1] == name}
        c, d, A = TC.valid_cells(rep, sub, model, name)
        if c:
            in_view.setdefault(model, []).append(name)      # a page counts for certification with a scored cell
        dropped[f"{model}-{label}"] = d
        A_of[(model, name)] = A
        meas.update(c)
    g3 = TC.g3_rows(meas, preds, None)
    g3v = TC.C5.g3_rule(g3) if g3 else {"verdict": "NOT SCORED: no G3 row"}
    void = bool(g3) and g3v.get("void_strata") == "ALL"
    nat, sh = [], []
    for k, m in meas.items():
        pr = preds[k]
        row = {"cell": "/".join(map(str, k)), "model": k[0], "G": pr["G"], "n": k[2], "e": TC.carried(pr, A_of.get(k[:2])) / m - 1,
               "meas": m, "pred_v4": pr["ms"], "pred": {R: v for R, v in pr["ms_all"].items() if R != "v4"}}
        (nat if k[4] == "native" else sh).append(row)
    E1 = TC.e1tile_rule(nat, in_view, tview.name, RG.BLIND_TILE, (RG.DSV,))
    if void:
        E1 = {"verdict": "NOT SCORED: G3 FAILS and Q does not reject, so every histogram cell is voided (strict)",
              "descriptive": E1}
    E1S = TC.bar([r["e"] for r in sh])
    riv = {R: TC.rival_rule(nat, R, sigma=sigma) for R in sorted({R for r in nat for R in r["pred"]})}
    split = {"G-16 cells": TC.bar([r["e"] for r in nat if r["G"] == 16]),
             "G-1 n-1 cells": TC.bar([r["e"] for r in nat if r["G"] == 1 and r["n"] == 1])}
    return {"pages": use, "dropped_rows": dropped, "G3": {k: v for k, v in g3v.items() if k != "rows"}, "E1-TILE": E1,
            "E1-TILE-S (secondary)": E1S, "rivals": riv, "split views (DESCRIPTIVE)": split, "cells_measured": len(meas)}


def bytes_legs(repo: Path, tree: Path, reg: dict, doc: dict, preds: dict) -> dict:
    """B-BYTES (predicted against counted bytes per GEMM) and E2-TILE (v4's time from the counted
    bytes against the timed pages' valid NATIVE cells, ALL view) on the byte pages."""
    import wave_split_bytes_v4 as WV4
    files = reg["pages"]["files"]
    sel = doc["selections"]
    prm = FN.params(sel)
    rows, e2 = [], []
    tv = TC.C3.TimedView("ALL")
    timed = {}
    for u in RG.B_TIMED_PAGES:
        label, model, name = u[:3]
        f = files[f"{name}.json"]
        rep, _w = TC.timed_page(tree, model, label, f["group_m"], tv, f)
        if rep is not None:
            c, _d, _A = TC.valid_cells(rep, {k: v for k, v in preds.items() if k[0] == model and k[1] == name}, model, name)
            timed.update(c)
    pages = {}
    for u in RG.B_BYTE_PAGES:
        label, model, name = u[:3]
        f = files[f"{name}.json"]
        p, path, why = TC.counter_page(tree, model, label, f["group_m"], f, f["tile"])
        if p is not None:
            clk = TC.C6.clock_rule(p, path)
            if not clk["page_ok"]:
                p, why = None, f"NOT SCORED (clock rule): {clk['why']}"
        pages[f"{model}-{label}"] = why
        if p is None:
            continue
        hdoc = RG.hist("B", name)
        t = RG.PAGE_TILE[int(hdoc["block_m"])]
        for cell, cc in zip(p.get("cells") or [], clk["cells"], strict=True):
            k = (model, name, int(cell["n"]), str(cell.get("histogram")), str(cell["arm"]))
            if k not in preds or not cc["ok"]:
                continue
            counted = {g: float(cell["per_gemm"][g]["dram_bytes_read"]) for g in ("w1", "w2")}
            for g in ("w1", "w2"):
                rows.append({"cell": "/".join(map(str, k)), "gemm": g, "e": preds[k]["reads"][g] / counted[g] - 1})
            if k[4] == "native" and k in timed:
                hc = next(x for x in hdoc["cells"] if x["label"] == k[3] and int(x["n"]) == k[2])
                T = WV4.price(RG.pricer(), model, "native", k[2], hc["counts"], RG.COPIES, RG.seed_of(hdoc, hc), t,
                              int(hdoc["group_m"]), prm, key=sel["key"], reads=counted)["ms"]
                e2.append(T / timed[k] - 1)
    bb = {g: TC.C5.stats(r["e"] for r in rows if r["gemm"] == g) for g in ("w1", "w2")}
    ok = all(s["cells"] and s["rms"] <= B_BYTES["rms_max"] and abs(s["worst"]) <= B_BYTES["max_abs"] for s in bb.values())
    return {"pages": pages, "B-BYTES": {"verdict": ("HOLDS" if ok else "FAILS") if rows else "NOT SCORED: no byte row",
                                        "per_gemm": bb, "bar": B_BYTES},
            "E2-TILE": TC.bar(e2, E2_BAR) if e2 else {"verdict": "NOT SCORED: no cell on both a byte and a timed page"}}


def score(repo: Path, tree: Path, pred_path: Path) -> dict:
    reg = TC.registration(repo, "e1tile")
    doc, preds = load_preds(pred_path)
    sigma = float(doc["selections"]["sigma_page"])
    cc = census_check(repo, tree, doc)
    if cc["excluded"] == "ALL":
        preds = {}
    else:
        bad = set(cc["excluded"])
        preds = {k: v for k, v in preds.items() if (k[0], v["tile"]) not in bad}
    res = TC.C3.two_views(lambda fv, tv: score_view(tree, reg, preds, sigma, tv), repo=repo)
    clean = ((res.get("addendum") or {}).get("CLEAN") or {}).get("E1-TILE") or {}
    res["certification"] = {m: (clean.get("per_model") or {}).get(m, {}).get("certification", "NOT CERTIFIED (no CLEAN cell)")
                            for m in RG.BLIND_TILE}
    if (res.get("E1-TILE") or {}).get("verdict") != "HOLDS":
        res["certification"] = {m: (v if not v.startswith("CERTIFIED") else
                                    f"NOT CERTIFIED (E1-TILE reads {(res.get('E1-TILE') or {}).get('verdict')})")
                                for m, v in res["certification"].items()}
    res["bytes"] = bytes_legs(repo, tree, reg, doc, preds)
    res.update(registration=reg["name"], part=PART, predictions=str(pred_path), sigma_page=sigma, census=cc)
    return res


def lines(res: dict) -> list[str]:
    e = res.get("E1-TILE") or {}
    out = [f"SCORED against {res['registration']} (ALL view; CLEAN beside it in the JSON)",
           f"E1-TILE: {e.get('verdict')} pooled {(e.get('pooled') or {}).get('stats')}"]
    for m, v in (e.get("per_model") or {}).items():
        out.append(f"  {m}: {v.get('verdict')} {v.get('stats')}")
    out.append(f"certification: {res.get('certification')}")
    out.append(f"B-BYTES: {res['bytes']['B-BYTES']['verdict']}; E2-TILE: {res['bytes']['E2-TILE'].get('verdict')}")
    return out


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 4:
        print(__doc__)
        return 2
    res = score(Path(a[0]), Path(a[1]), Path(a[3]))
    print(TC.C3.write_score(Path(a[2]), PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
