#!/usr/bin/env python
"""Score CONV-L (docs/registered/2026-10-20-tileA2-convl-gh200.json; REVIEW-A2 fix 2, MAJOR 3a): the same R3-tile
anchor pages timed on A2's board at 1710 and at A2's lock L, scored as a PER-CELL RATIO against v3's
predicted ratio under P1's rule, so the board's level and v3's own level error cancel and only the
conversion (c x 1710 / L, every other constant carried) is tested.

    python scripts/scoring/tileA2/score_convl_a2.py <repo> <tree> <out> [--lock L]

Reads under <tree> the six CONV-L units' pages (gaps-<card>-<model>-<label>: cvm1, cvo1, cvm8 at L and
cvm1t, cvo1t, cvm8t at 1710), each rebuilt under r7-median-run3 at its own lock, G5 against the
registered files, and rental 6's row rule with G4 at the page's lock - 5 and the host rule on that
lock's registered prediction. Per NATIVE cell valid on both pages: r = (T_L / T_1710) / (P_L / P_1710) - 1.
Per class (G 1: the Mixtral DRAM-bound and OLMoE short pages; G 8: Mixtral c-heavy) the E1 bar
(rms <= 2%, worst <= 5%, |mean| <= 1%); CONV-L HOLDS iff both classes hold, in the CLEAN view (ALL
printed). The 2x2 reading with tileA2-e1cal is DESCRIPTIVE and worded "indicates". Writes
<out>/convl.score.{json,txt}.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402

TC = A2.TC
PART = "convl"
r6 = A2.r6


def rules_at(lock: float) -> A2.Rules:
    return A2.Rules(float(lock), A2.band_of(lock), float(lock) - A2.G4_BELOW_MHZ, dict(A2.NAMES_A2),
                    f"tileA2 CONV-L page at {lock:g} MHz")


def _page(tree: Path, reg: dict, unit: dict, lock: float, preds: dict, view: str):
    f = reg["pages"]["files"][f"{unit['page']}.json"]
    tv = TC.C3.TimedView(view)
    with A2.scoring(rules_at(lock)):
        rep, why = TC.timed_page(tree, unit["model"], unit["label"], int(unit["G"]), tv, f)
        if rep is None:
            return {}, why, []
        sub = {k: {"ms": v[str(int(lock))]} for k, v in preds.items() if k[0] == unit["model"] and k[1] == unit["page"]}
        meas, dropped, _A = TC.valid_cells(rep, sub, unit["model"], unit["page"])
    return meas, why, dropped


def score_view(tree: Path, reg: dict, lock: float, view: str) -> dict:
    preds = {(p["model"], p["page"], int(p["n"]), p["label"], "native"): p["ms"] for p in reg["predictions"]}
    lk = str(int(lock))
    rows, pages, dropped = [], {}, {}
    for unit in reg["pages"]["units"]["at_L"]:
        twin = next(u for u in reg["pages"]["units"]["at_1710"] if u["page"] == unit["page"])
        mL, wL, dL = _page(tree, reg, unit, lock, preds, view)
        mT, wT, dT = _page(tree, reg, twin, 1710.0, preds, view)
        pages[f"{unit['model']}-{unit['label']}"] = wL
        pages[f"{twin['model']}-{twin['label']}"] = wT
        dropped[unit["label"]], dropped[twin["label"]] = dL, dT
        for k, tl in mL.items():
            if k not in mT or k[4] != "native":
                continue
            p = preds[k]
            pr = p[lk] / p["1710"]
            r = (tl / mT[k]) / pr - 1
            rows.append({"cell": "/".join(map(str, k)), "page": unit["page"], "cls": unit["class"],
                         "ratio_meas": r6(tl / mT[k]), "ratio_pred": r6(pr), "r": r})
    classes = {}
    for c in reg["rule"]["classes"]:
        v = [x["r"] for x in rows if x["cls"] == c]
        classes[c] = TC.bar(v)
    ok = all(b["verdict"] == "HOLDS" for b in classes.values())
    if any(str(b["verdict"]).startswith("NOT SCORED") for b in classes.values()):
        verdict = "NOT SCORED: a class has no cell valid at both locks"
    else:
        verdict = "HOLDS" if ok else "FAILS"
    return {"verdict": verdict, "classes": classes, "pages": pages, "dropped": dropped,
            "rows": [dict(x, r=r6(x["r"])) for x in rows],
            "per_page (DESCRIPTIVE)": {pg: TC.C5.stats(x["r"] for x in rows if x["page"] == pg)
                                       for pg in sorted({x["page"] for x in rows})}}


def reading(convl: str, e1cal: str | None) -> str:
    """The 2x2 reading (DESCRIPTIVE, registered before any page)."""
    e = str(e1cal or "not scored")
    if convl not in ("HOLDS", "FAILS") or e not in ("HOLDS", "FAILS"):
        return f"CONV-L {convl}, E1-CAL {e}: no reading"
    if convl == "HOLDS" and e == "HOLDS":
        return "both hold: the conversion and the wgmma calibration both pass their bars"
    if convl == "HOLDS":
        return ("CONV-L HOLDS and E1-CAL does not: this indicates the wgmma calibration or v4's wgmma terms, "
                "not the conversion; a carried constant can still weigh more on 128x128 G 1 cells (more bytes, bw "
                "at the lower L2 clock) than on R3's tile, so it is not ruled out")
    if convl == "FAILS" and e == "HOLDS":
        return "CONV-L FAILS and E1-CAL holds: this indicates the conversion errs on R3's tile and not at the wgmma tiles' cells"
    return "both fail: this indicates the conversion (CONV-L) as at least one cause of E1-CAL's failure"


def score(repo: Path, tree: Path, out: Path, lock: float) -> dict:
    reg = A2.registration(repo, PART)
    if float(lock) == A2.FIT_CLOCK_MHZ:
        why = ("NOT SCORED: A2 holds rung 1710, so no clock conversion is used (v4L is v4) and the six CONV-L units are "
               "skipped by the driver (unless-lock=1710)")
        return {"registration": reg["name"], "part": PART, "lock_mhz": lock, "CONV-L": why, "views": {},
                "reading (DESCRIPTIVE)": "no conversion, no reading"}
    views = {v: score_view(tree, reg, lock, v) for v in ("ALL", "CLEAN")}
    e1 = (A2.read_score(out, "e1cal") or {}).get("E1-CAL", {}).get("verdict")
    return {"registration": reg["name"], "part": PART, "lock_mhz": lock, "CONV-L": views["CLEAN"]["verdict"],
            "views": views, "e1cal_verdict_read": e1,
            "reading (DESCRIPTIVE)": reading(views["CLEAN"]["verdict"], e1)}


def lines(res: dict) -> list[str]:
    out = [f"SCORED against {res['registration']} at {res['lock_mhz']:g} MHz and 1710",
           f"CONV-L: {res['CONV-L']}"]
    for c, b in (res["views"].get("CLEAN") or {}).get("classes", {}).items():
        out.append(f"  {c}: {b['verdict']} {b.get('stats')}")
    out.append(f"READING (DESCRIPTIVE): {res['reading (DESCRIPTIVE)']}")
    return out


def main(argv=None) -> int:
    a, rules = A2.parse_cli(argv if argv is not None else sys.argv[1:], __doc__)
    if a is None or not rules.label.startswith("tileA2"):
        return A2.usage(__doc__)
    repo, tree, out = (Path(x) for x in a)
    res = score(repo, tree, out, rules.lock)
    print(TC.C3.write_score(out, PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
