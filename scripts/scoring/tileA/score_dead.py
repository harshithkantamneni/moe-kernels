#!/usr/bin/env python
"""Score rental A's dead-CTA lever (docs/registered/2026-10-14-tileA-dead-gh200.json): D against SLOT
at occupancy 2; kappa only when c_128 >= 900.

    python scripts/scoring/tileA/score_dead.py <repo> <tree> <out>

Reads under <tree>: gaps-<card>-olmoe-1b-7b-d9 and -d15 (one G 1 page each, rebuilt under
r7-median-run3); <out>/floor.score.json (the measured c_128) and rep.score.json (sigma_page). Writes
<out>/dead.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regtile as RG  # noqa: E402
import tcommon as TC  # noqa: E402

PART = "dead"
MODEL = "olmoe-1b-7b"


def _read(out: Path, name: str) -> dict | None:
    p = Path(out) / f"{name}.score.json"
    return json.loads(p.read_text()) if p.exists() else None


def score(repo: Path, tree: Path, out: Path) -> dict:
    reg = TC.registration(repo, PART)
    f = reg["pages"]["files"]["olmoe-1b-7b-dead.json"]
    tv = TC.C3.TimedView("CLEAN")
    pages, why = {}, {}
    for label in ("d9", "d15"):
        r, w = TC.timed_page(tree, MODEL, label, 1, tv, f)
        why[label] = w
        if r is not None:
            pages[label] = r
    res = {"registration": reg["name"], "part": PART, "pages": why}
    if len(pages) != 2:
        res["DEAD"] = {"verdict": "NOT SCORED: both pages are needed"}
        return res
    ms = {lab: {(str(x["histogram"]), int(x["tiles"]), str(x["arm"])): float(x["ms_p50"])
                for x in p.get("treads_table") or [] if x.get("ms_p50") and not x.get("lost")} for lab, p in pages.items()}
    A = {lab: TC.C6.page_A_us(p) or {} for lab, p in pages.items()}
    deltas, Ts = [], []
    for (h, n, arm) in sorted(ms["d9"]):
        if arm != "native":
            continue
        k_s = (h, n, "shared")
        if k_s not in ms["d9"] or k_s not in ms["d15"] or (h, n, "native") not in ms["d15"]:
            continue
        d = 1e3 * ((ms["d15"][k_s] - ms["d15"][(h, n, "native")]) - (ms["d9"][k_s] - ms["d9"][(h, n, "native")]))
        a = (A["d15"].get(("shared", n), 0.0)) - (A["d9"].get(("shared", n), 0.0))
        deltas.append(d - a)
        Ts.append(1e3 * st.fmean([ms["d9"][k_s], ms["d15"][k_s], ms["d9"][(h, n, "native")], ms["d15"][(h, n, "native")]]))
    if not deltas:
        res["DEAD"] = {"verdict": "NOT SCORED: no cell valid on both pages"}
        return res
    sigma, src = TC.sigma_page_of(_read(out, "rep"))
    se = 2 * sigma * st.fmean(Ts) / math.sqrt(len(deltas))
    c128 = ((_read(out, "floor") or {}).get("calibration") or {}).get("c")
    laws = RG.dead_increments(float(c128)) if c128 else RG.dead_increments(900.0)
    res["DEAD"] = TC.dead_rule(st.median(deltas), se, laws, c128)
    res["DEAD"].update(cells=len(deltas), sigma_page=sigma, sigma_source=src,
                       c128=c128, laws_at=("the measured c_128" if c128 else "the nominal 900 (no calibration)"))
    return res


def lines(res: dict) -> list[str]:
    d = res.get("DEAD", {})
    return [f"SCORED against {res['registration']}", f"  pages: {res['pages']}",
            f"DEAD: {d.get('D_vs_SLOT', d.get('verdict'))}; delta {d.get('delta_us')} us se {d.get('se_us')}",
            *[f"  {k}: {v}" for k, v in (d.get('laws') or {}).items()]]


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
