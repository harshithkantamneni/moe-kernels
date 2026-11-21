#!/usr/bin/env python
"""Score rental 6's second K (docs/registered/2026-10-09-rental6-secondk-gh200.json): per-CTA
fixed cost against main loop on mixtral-8x7b-tp4 G = 64 counter pages k32s4 against k64s4,
both taken in rental 6 on one board.

    python scripts/scoring/rental6/score_secondk.py <repo> <tree> <out>

Gates (registered): the counter page's V-gates by rental 2's two views (V5 gates; V6, V7, V10
do not), r6common.clock_rule on each page (check_page PASS, cells in [1640, 1710] MHz), the two
pages' median cell clocks within 1%; a page whose V7 FAILS is read off its NATIVE cells only.
Writes <out>/secondk.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6common as C6  # noqa: E402

C3, C4, C5 = C6.C3, C6.C4, C6.C5
PART = "secondk"
OCCLAW = C6._load("rental4_score_occlaw", HERE.parent / "rental4" / "score_occlaw.py")


def _v7_failed(page: dict) -> bool:
    return any(str(g.get("number") or g.get("tag")) == "V7" and g.get("verdict") not in ("PASS", None)
               for g in page.get("gates") or [])


def score_view(repo: Path, tree: Path, view) -> dict:
    reg = C6.registration(repo, PART)
    model = reg["pages"]["path"].split("<card>-")[1].split("-<label>")[0]
    res = {"registration": C6.NAMES[PART], "pages": {}}
    pages, clocks, v7 = {}, {}, {}
    for label, (bk, stg) in reg["pages"]["labels"].items():
        path = C3.find_bytes(tree, model, label, 64)
        page = view.load(path)
        if page is None:
            res["pages"][label] = "NOT SCORED: missing or out of this view"
            continue
        de = page.get("design") or {}
        if (int(de.get("block_k") or 0), int(de.get("num_stages") or 0)) != (bk, stg):
            res["pages"][label] = f"NOT SCORED: the page's design is BK {de.get('block_k')} s{de.get('num_stages')}"
            continue
        clk = C6.clock_rule(page, path)
        if not clk["page_ok"]:
            res["pages"][label] = f"NOT SCORED (clock rule): {clk['why']}"
            continue
        keep = [c for c, cc in zip(page.get("cells") or [], clk["cells"], strict=True) if cc["ok"]]
        v7[label] = _v7_failed(page)
        if v7[label]:
            keep = [c for c in keep if c.get("arm") == "native"]
        pages[label] = dict(page, cells=keep)
        clocks[label] = st.median(cc["mhz"] for cc in clk["cells"] if cc["ok"]) if clk["cells_ok"] else None
        res["pages"][label] = "counted" + (" (V7 FAILED: NATIVE cells only)" if v7[label] else "")
    res["clocks_mhz"] = clocks
    if set(pages) != {"k64s4", "k32s4"}:
        res["verdict"] = "NOT SCORED: the k64s4 / k32s4 pair is incomplete"
        return res
    if None in clocks.values() or abs(clocks["k32s4"] / clocks["k64s4"] - 1) > C6.SECONDK_CLOCK_AGREE:
        res["verdict"] = f"NOT SCORED: the two pages' clocks {clocks} disagree by more than 1%"
        return res
    readings, rows = {}, {}
    ns = reg["inputs"]["ns"]
    frac = reg["inputs"]["max_dram_frac"]
    for g in ("w1", "w2"):
        F = reg["inputs"]["F_cal"][g]
        f64 = C4.page_c(pages["k64s4"], g, ns, F, frac)
        f32 = C4.page_c(pages["k32s4"], g, ns, F, frac)
        if f64 is None or f32 is None:
            rows[g] = {"verdict": "NOT SCORED: fewer than 3 floor-bound cells"}
            continue
        r = (f32["S"] * f32["c"] + F) / (f64["S"] * f64["c"] + F)
        pr = reg["predicted_ratio"][g]
        e = {h: r / pr[h] - 1 for h in ("H_ITER", "H_CTA")}
        best = min(e, key=lambda h: abs(e[h]))
        reading = best if abs(e[best]) <= C6.SECONDK_TOL else "NEITHER"
        readings[g] = reading
        rows[g] = {"c64": f64["c"], "c32": f32["c"], "S64": f64["S"], "S32": f32["S"], "r_meas": r, "e": e, "reading": reading}
    res["gemms"] = rows
    if len(readings) < 2:
        res["verdict"] = "NOT SCORED: a GEMM has no reading"
    elif len(set(readings.values())) == 1 and "NEITHER" not in readings.values():
        res["verdict"] = f"SELECTED {readings['w1']}" + (" (V7: native only)" if any(v7.values()) else "")
    else:
        res["verdict"] = f"UNDECIDED ({readings})"
    return res


def score(repo: Path, tree: Path) -> dict:
    return OCCLAW.bytes_two_views(lambda v: score_view(repo, tree, v), repo)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 6 SECOND K (per-CTA fixed vs main loop), {res['registration']}"]
    for lab, u in res["pages"].items():
        out.append(f"  page {lab}: {u}")
    out.append(f"  clocks: {res.get('clocks_mhz')}")
    for g, v in (res.get("gemms") or {}).items():
        if "r_meas" in v:
            out.append(f"  {g}: r {v['r_meas']:.4f} | " + "  ".join(f"{h} {100 * x:+.1f}%" for h, x in v["e"].items()) + f" -> {v['reading']}")
        else:
            out.append(f"  {g}: {v['verdict']}")
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
