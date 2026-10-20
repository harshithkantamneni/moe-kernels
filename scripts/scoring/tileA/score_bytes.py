#!/usr/bin/env python
"""Score rental A's byte pages (docs/registered/2026-10-14-tileA-bytes-gh200.json): the A key rental
B's v4 takes (NAMED, else CLAMP) on every scored w1 row, the brackets, and the w2 and G-16 rows
(printed).

    python scripts/scoring/tileA/score_bytes.py <repo> <tree> <out>

Reads under <tree>: <date>-<card>-<model>-<label>-r3-counters/lock1710/r3c-g<G>.json for the units
b128g1 (OLMoE, Qwen2-57B), b64g1 and b128g16 (Mixtral 8x7B). G5, the tile and rental 6's counter
clock rule gate each page. Writes <out>/bytes.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# tcommon first: it puts the repo and scripts/ on sys.path, which r3_timing_model and
# r3_timing_model_v4 need from the command line (the tests import tcommon before this file)
import tcommon as TC  # noqa: E402
import r3_timing_model as TM  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402

PART = "bytes"
r6 = TC.r6
UNITS = (("b128g1", "olmoe-1b-7b", "olmoe-1b-7b-128g1"), ("b64g1", "mixtral-8x7b", "mixtral-8x7b-64g1"),
         ("b128g16", "mixtral-8x7b", "mixtral-8x7b-128g16"), ("b128g1", "qwen2-57b-a14b", "qwen2-57b-a14b-128g1"))


def byte_model(model: str, bm: int) -> dict:
    t = V4.T64W8 if bm == 64 else V4.CAL_TILE
    with V4.at_tile(model, t):
        return dict(TM.BYTE_MODEL)


def score(repo: Path, tree: Path) -> dict:
    reg = TC.registration(repo, PART)
    files = reg["pages"]["files"]
    pred = {(r["page"], r["n"], r["label"], r["arm"]): r for r in reg["byte_predictions"]}
    res = {"registration": reg["name"], "part": PART, "pages": {}}
    rows, printed, dropped = [], [], []
    for label, model, name in UNITS:
        f = files[f"{name}.json"]
        p, path, why = TC.counter_page(tree, model, label, f["group_m"], f, f["tile"])
        if p is not None:
            clk = TC.C6.clock_rule(p, path)
            if not clk["page_ok"]:
                p, why = None, f"NOT SCORED (clock rule): {clk['why']}"
        res["pages"][f"{model}-{label}"] = why
        if p is None:
            continue
        bm = byte_model(model, f["tile"][0])
        for cell, cc in zip(p.get("cells") or [], clk["cells"], strict=True):
            lab, n, arm = str(cell.get("histogram")), int(cell["n"]), str(cell["arm"])
            pr = pred.get((name, n, lab, arm))
            if pr is None or not cc["ok"]:
                dropped.append(f"{name}/{lab}/{arm}/n{n}: {'no prediction' if pr is None else 'clock'}")
                continue
            q = {g: (float(cell["per_gemm"][g]["dram_bytes_read"]) - n * bm[f"operand_per_tile_{g}"]) / bm[f"W_{g}"]
                 for g in ("w1", "w2")}
            e1 = {k: (v["w1"] / q["w1"] - 1 if v else None) for k, v in pr["q_pred"].items()}
            row = {"page": name, "cell": f"n{n}/{lab}/{arm}", "label": lab, "skewed": pr["skewed"], "q_counted": {g: r6(x) for g, x in q.items()},
                   "e": e1, "e_w2": {k: (v["w2"] / q["w2"] - 1 if v and "w2" in v else None) for k, v in pr["q_pred"].items()},
                   "below_ideal": q["w1"] < pr["q_ideal"]}
            (rows if pr["scored_for_key"] else printed).append(row)
    res["KEY"] = TC.named_key(rows)
    res["rows"] = [dict(r, e={k: r6(v) for k, v in r["e"].items()}) for r in rows]
    res["printed_G16_rows"] = [dict(r, e={k: r6(v) for k, v in r["e"].items()}) for r in printed]
    res["w2_printed"] = {k: TC.C5.stats(x["e_w2"][k] for x in rows + printed if x["e_w2"].get(k) is not None)
                         for k in ("v3", "CLAMP")}
    res["B_BYTES_bar_printed"] = {k: TC.C5.stats(x["e"][k] for x in rows if x["e"].get(k) is not None) for k in TC.KEYS}
    res["dropped"] = dropped
    return res


def lines(res: dict) -> list[str]:
    k = res.get("KEY", {})
    return [f"SCORED against {res['registration']}", f"  pages: {res['pages']}",
            f"KEY: {k.get('verdict')} rms % {k.get('rms_pct')} worst nearest {k.get('worst_nearest_pct')}%"]


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 3:
        print(__doc__)
        return 2
    res = score(Path(a[0]), Path(a[1]))
    print(TC.C3.write_score(Path(a[2]), PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
