#!/usr/bin/env python
"""Score rental A's E1-CAL pages (docs/registered/2026-10-14-tileA-e1cal-gh200.json) and read rental
B's stop rule (docs/registered/2026-10-14-tileB-stop-gh200.json).

    python scripts/scoring/tileA/score_e1cal.py <repo> <tree> <out>

Needs rental A's census, floor, bytes and rep scores in <out> (run score_census, score_floor,
score_bytes and score_rep first): v4's constants and selections are read there (finalise.selections)
and its predictions computed by finalise.predict_pages, the code that finalises rental B. Reads under
<tree> the five E1-CAL units' G-1 / G-16 pages (gaps-<card>-<model>-<label>), rebuilt under
r7-median-run3. Writes <out>/e1cal.score.{json,txt}: E1-CAL per view, the class means, G3, the OLS
rival's fit for rental B, and the STOP / GO verdict. Exits 0 whatever the verdicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import finalise as FN  # noqa: E402
import regtile as RG  # noqa: E402
import tcommon as TC  # noqa: E402

PART = "e1cal"
r6 = TC.r6


def score_view(tree: Path, reg: dict, preds: dict, tview) -> dict:
    files = reg["pages"]["files"]
    meas, A_of, use, dropped = {}, {}, {}, {}
    for label, model, name in RG.A_TIMED_PAGES:
        f = files[f"{name}.json"]
        rep, why = TC.timed_page(tree, model, label, f["group_m"], tview, f)
        key = f"{model}-{label}"
        use[key] = why if rep is not None else (why if why.startswith(("NOT SCORED", "out of")) else why)
        if rep is None:
            continue
        sub = {k: v for k, v in preds.items() if k[0] == model and k[1] == name}
        c, d, A = TC.valid_cells(rep, sub, model, name)
        dropped[key] = d
        A_of[(model, name)] = A
        meas.update(c)
    g3 = TC.g3_rows(meas, preds, None)
    g3v = TC.C5.g3_rule(g3) if g3 else {"verdict": "NOT SCORED: no G3 row"}
    void = g3v.get("void_strata") == "ALL"
    rows, shared = [], []
    for k, m in meas.items():
        pr = preds[k]
        e = TC.carried(pr, A_of.get(k[:2])) / m - 1
        r = {"cell": "/".join(map(str, k)), "model": k[0], "cls": pr["cls"], "e": e}
        (rows if k[4] == "native" else shared).append(r)
    bar = TC.bar([r["e"] for r in rows])
    if void:
        bar = {"verdict": "NOT SCORED: G3 voids every histogram cell", "descriptive": bar}
    pages = {k: ("counted" if v == "counted" or v.startswith("counted") else v) for k, v in use.items()}
    return {"pages": pages, "dropped_rows": dropped, "G3": {k: v for k, v in g3v.items() if k != "rows"},
            "g3_void": void, "bar": bar, "rows": rows, "class_means": TC.class_means(rows),
            "per_page": {f"{m}-{n}": TC.C5.stats(r["e"] for r in rows if r["cell"].startswith(f"{m}/{n}/"))
                         for _l, m, n in RG.A_TIMED_PAGES},
            "SHARED_printed": TC.C5.stats(r["e"] for r in shared), "meas": {"/".join(map(str, k)): v for k, v in meas.items()}}


def score(repo: Path, tree: Path, out: Path) -> dict:
    reg = TC.registration(repo, PART)
    res = {"registration": reg["name"], "part": PART}
    try:
        sel = FN.selections(out)
    except FN.Refused as exc:
        res["E1-CAL"] = {"verdict": f"NOT SCORED: {exc}"}
        res["STOP"] = {"verdict": "STOP: rental B does not run", "why": [str(exc)]}
        return res
    rows = FN.predict_pages(RG.A_TIMED_PAGES, "A", sel, rivals=False)
    preds = {(r["model"], r["page"], r["n"], r["label"], r["arm"]): dict(r, ms=r["ms"]["v4"]) for r in rows}
    views = {}
    for name in ("ALL", "CLEAN"):
        tv = TC.C3.TimedView(name)
        views[name] = score_view(tree, reg, preds, tv)
    res["selections"] = sel
    res["E1-CAL"] = {"ALL": views["ALL"]["bar"], "CLEAN": views["CLEAN"]["bar"],
                     "verdict": ("HOLDS" if all(views[v]["bar"].get("verdict") == "HOLDS" for v in views) else
                                 "FAILS" if any(views[v]["bar"].get("verdict") == "FAILS" for v in views) else
                                 views["CLEAN"]["bar"].get("verdict"))}
    res["views"] = {v: {k: x for k, x in d.items() if k not in ("rows", "meas")} for v, d in views.items()}
    res["STOP"] = TC.stop_rule(views["CLEAN"], FN._read(out, "census"))
    meas = {}
    for k, v in views["CLEAN"]["meas"].items():
        m, p, n, lab, arm = k.split("/")
        if arm == "native":
            meas[(m, p, int(n), lab)] = v
    res["OLS_fit"] = FN.ols_fit(rows, meas)
    return res


def lines(res: dict) -> list[str]:
    out = [f"SCORED against {res['registration']}", f"E1-CAL: {res['E1-CAL'].get('verdict')}"]
    for v in ("ALL", "CLEAN"):
        b = (res["E1-CAL"].get(v) or {})
        out.append(f"  {v}: {b.get('verdict')} {b.get('stats')}")
    s = res.get("STOP", {})
    out.append(f"STOP RULE: {s.get('verdict')}")
    out += [f"  {w}" for w in s.get("why") or []]
    return out


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 3:
        print(__doc__)
        return 2
    res = score(Path(a[0]), Path(a[1]), Path(a[2]))
    print(TC.C3.write_score(Path(a[2]), PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
