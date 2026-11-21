#!/usr/bin/env python
"""Score rental 7's OLMoE w1 byte page (docs/registered/2026-10-13-rental7-l2open-gh200.json):
OPEN-L2SKEW's rivals v3, ROWKEY39.5 and ROWKEY60 on the exact page priced before it ran. OPEN: no
model change follows from it, whatever it reads.

    python scripts/scoring/rental7/score_l2open.py <repo> <tree> <out>

Reads under <tree>: <date>-<card>-olmoe-1b-7b-l2open-r3-counters/lock1710/r3c-g8.json (one page).
G5: its histogram sha256, shuffle seed and every cell's counts sha256 must be the registered ones;
then rental 6's counter clock rule (r6common.clock_rule). Writes <out>/l2open.score.{json,txt};
exits 0 whatever the verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7common as C7  # noqa: E402

C3, C5, C6 = C7.C3, C7.C5, C7.C6
PART = "l2open"
MODEL = "olmoe-1b-7b"
r6 = C6.r6


def counter_page(tree: Path, reg_file: dict | None):
    hits = sorted(Path(tree).rglob(f"*-{C6.CARD}-{MODEL}-l2open-r3-counters/lock1710/r3c-g8.json"))
    if len(hits) != 1:
        return None, None, f"NOT SCORED: {len(hits)} l2open byte pages"
    p = json.loads(hits[0].read_text())
    h = (p.get("design") or {}).get("histogram") or {}
    if reg_file is None or h.get("file_sha256") != reg_file["sha256"]:
        return None, None, "NOT SCORED (G5): the page's histogram sha256 is not the registered one"
    if h.get("shuffle_seed") != reg_file.get("shuffle_seed"):
        return None, None, "NOT SCORED (G5): the page records another shuffle seed"
    want = {(c["label"], c["n"]): c["counts_sha256"] for c in reg_file["cells"]}
    bad = [f"{c.get('histogram')}/{c.get('arm')}/n{c.get('n')}" for c in p.get("cells") or []
           if want.get((str(c.get("histogram")), int(c["n"]))) != c.get("counts_sha256")]
    if bad:
        return None, None, f"NOT SCORED (G5): cells {bad[:4]} carry another counts sha256"
    return p, hits[0], "counted"


def score(repo: Path, tree: Path) -> dict:
    import r3_timing_model as TM
    reg = C7.registration(repo, PART)
    files = reg["page"]["file"]
    page, path, why = counter_page(tree, files.get(f"{MODEL}-l2open.json"))
    res = {"registration": reg["name"], "part": PART, "status": "OPEN (no model change follows)", "page": why}
    if page is None:
        res["L2OPEN"] = {"verdict": why}
        return res
    clk = C6.clock_rule(page, path)
    res["clock"] = {"check_page": clk["check_page"], "cells_in_band": clk["cells_ok"], "cells": len(clk["cells"])}
    if not clk["page_ok"]:
        res["L2OPEN"] = {"verdict": f"NOT SCORED (clock rule): {clk['why']}"}
        return res
    old = TM.set_model(MODEL)
    bm = dict(TM.BYTE_MODEL)
    TM.set_model(old)
    pred = {(r["n"], r["label"], r["arm"]): r for r in reg["byte_predictions"]}
    rows, w2, dropped = [], [], []
    for cell, cc in zip(page.get("cells") or [], clk["cells"], strict=True):
        lab, n, arm = str(cell.get("histogram")), int(cell["n"]), str(cell["arm"])
        pr = pred.get((n, lab, arm))
        if pr is None:
            dropped.append(f"{lab}/{arm}/n{n}: no registered prediction")
            continue
        if not cc["ok"]:
            dropped.append(f"{lab}/{arm}/n{n}: clock {cc['mhz']} outside {C6.BAND_MHZ}")
            continue
        q = {g: (float(cell["per_gemm"][g]["dram_bytes_read"]) - n * bm[f"operand_per_tile_{g}"]) / bm[f"W_{g}"] for g in ("w1", "w2")}
        rows.append({"cell": f"n{n}/{lab}/{arm}", "n": n, "label": lab, "arm": arm, "skewed": pr["skewed"], "q_counted_w1": r6(q["w1"]),
                     "e": {R: pr["q_pred"][R]["w1"] / q["w1"] - 1 for R in C7.L2_RIVALS}})
        w2.append({"cell": f"n{n}/{lab}/{arm}", "e_v3": r6(pr["q_pred"]["v3"]["w2"] / q["w2"] - 1)})
    res["L2OPEN"] = C7.l2open_rule(rows)
    res["rows"] = [dict(r, e={k: r6(v) for k, v in r["e"].items()}) for r in rows]
    res["w2_v3_printed"] = {"stats": C5.stats(x["e_v3"] for x in w2), "rows": w2}
    res["per_n_skewed_mean_pct"] = {n: {R: r6(100 * sum(r["e"][R] for r in rows if r["skewed"] and r["n"] == n) /
                                              max(1, sum(1 for r in rows if r["skewed"] and r["n"] == n))) for R in C7.L2_RIVALS}
                                    for n in sorted({r["n"] for r in rows})}
    res["dropped"] = dropped
    return res


def lines(res: dict) -> list[str]:
    out = [f"SCORED against {res['registration']} ({res['status']})", f"  page: {res['page']}"]
    if res.get("clock"):
        out.append(f"  clock: {res['clock']}")
    v = res.get("L2OPEN", {})
    out.append(f"L2OPEN: {v.get('verdict')} rms % {v.get('rms_pct')} worst nearest {v.get('worst_nearest_pct')}%")
    for n, x in (res.get("per_n_skewed_mean_pct") or {}).items():
        out.append(f"  n {n} skewed mean e %: {x}")
    if res.get("w2_v3_printed"):
        out.append(f"  w2 under v3 (printed): {res['w2_v3_printed']['stats']}")
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
