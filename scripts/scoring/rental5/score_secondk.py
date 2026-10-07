#!/usr/bin/env python
"""Score rental 5's lever 4b, the second K (docs/registered/2026-10-07-rental5-secondk-gh200.json):
per-CTA fixed cost against main loop on qwen2-57b-a14b-tp8 counter pages k32s4 against k64s4.

    python scripts/scoring/rental5/score_secondk.py <repo> <tree> <out>

Reads `*-qwen2-57b-a14b-tp8-<label>-r3-counters/lock1710/r3c-g64.json` for labels k64s4 and
k32s4. Writes <out>/secondk.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5common as C5  # noqa: E402

C4, C3 = C5.C4, C5.C3
PART = "secondk"
NS = range(4, 10)
TOL = 0.03
OCCLAW = C5._load("rental4_score_occlaw", HERE.parent / "rental4" / "score_occlaw.py")


def score_view(repo: Path, tree: Path, view) -> dict:
    reg = C5.registration(repo, PART)
    model = reg["pages"]["path"].split("<card>-")[1].split("-<label>")[0]
    res = {"registration": C5.NAMES[PART], "pages": {}}
    pages = {}
    for label, (bk, stg) in reg["pages"]["labels"].items():
        page = view.load(C3.find_bytes(tree, model, label, 64))
        if page is None:
            res["pages"][label] = "NOT SCORED: missing or out of this view"
            continue
        de = page.get("design") or {}
        if (int(de.get("block_k") or 0), int(de.get("num_stages") or 0)) != (bk, stg):
            res["pages"][label] = f"NOT SCORED: the page's design is BK {de.get('block_k')} s{de.get('num_stages')}"
            continue
        pages[label] = page
        res["pages"][label] = "counted"
    if set(pages) != {"k64s4", "k32s4"}:
        res["verdict"] = "NOT SCORED: the k64s4 / k32s4 pair is incomplete"
        return res
    readings, rows = {}, {}
    for g in ("w1", "w2"):
        f64 = C4.page_c(pages["k64s4"], g, NS, C4.F_CTA[g], C4.FLOOR_MAX_DRAM_FRAC)
        f32 = C4.page_c(pages["k32s4"], g, NS, C4.F_CTA[g], C4.FLOOR_MAX_DRAM_FRAC)
        if f64 is None or f32 is None:
            rows[g] = {"verdict": "NOT SCORED: fewer than 3 floor-bound cells"}
            continue
        r = (f32["S"] * f32["c"] + C4.F_CTA[g]) / (f64["S"] * f64["c"] + C4.F_CTA[g])
        pr = reg["predicted_ratio"][g]
        e = {h: r / pr[h] - 1 for h in ("H_ITER", "H_CTA")}
        best = min(e, key=lambda h: abs(e[h]))
        reading = best if abs(e[best]) <= TOL else "NEITHER"
        readings[g] = reading
        rows[g] = {"c64": f64["c"], "c32": f32["c"], "S64": f64["S"], "S32": f32["S"], "r_meas": r, "e": e, "reading": reading}
    res["gemms"] = rows
    if len(readings) < 2:
        res["verdict"] = "NOT SCORED: a GEMM has no reading"
    elif len(set(readings.values())) == 1 and "NEITHER" not in readings.values():
        res["verdict"] = f"SELECTED {readings['w1']}"
    else:
        res["verdict"] = f"UNDECIDED ({readings})"
    sec_rows = [{"pred": reg["predicted_ratio"][g][v["reading"] if v["reading"] != "NEITHER" else "H_ITER"],
                 "meas": v["r_meas"], "cluster": "board", "stratum": g} for g, v in rows.items() if "r_meas" in v]
    res["secondary"] = C5.secondary(sec_rows)
    return res


def score(repo: Path, tree: Path) -> dict:
    return OCCLAW.bytes_two_views(lambda v: score_view(repo, tree, v), repo)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 5 SECOND K (per-CTA fixed vs main loop), {res['registration']}"]
    for lab, u in res["pages"].items():
        out.append(f"  page {lab}: {u}")
    for g, v in (res.get("gemms") or {}).items():
        if "r_meas" in v:
            out.append(f"  {g}: r {v['r_meas']:.4f} (c64 {v['c64']:.1f}, c32 {v['c32']:.1f}) | "
                       + "  ".join(f"{h} {100 * x:+.1f}%" for h, x in v["e"].items()) + f" -> {v['reading']}")
        else:
            out.append(f"  {g}: {v['verdict']}")
    out.append(f"  verdict: {res['verdict']}")
    if res.get("secondary"):
        out += C5.secondary_lines(res["secondary"])
    return out + C3.view_lines(res)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, tree, out = map(Path, a)
    res = score(repo, tree)
    print(C5.write_score(out, PART, res, lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
