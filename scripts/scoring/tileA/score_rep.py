#!/usr/bin/env python
"""Score rental A's replicate (docs/registered/2026-10-14-tileA-rep-gh200.json): sigma_page at the
128x128 tile from OLMoE's E1-CAL page and its replicate.

    python scripts/scoring/tileA/score_rep.py <repo> <tree> <out>

Reads under <tree>: gaps-<card>-olmoe-1b-7b-e128g1 and -e128g1r (one G 1 page each), rebuilt under
r7-median-run3. Writes <out>/rep.score.{json,txt}; exits 0 whatever it reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tcommon as TC  # noqa: E402

PART = "rep"
MODEL, FILE = "olmoe-1b-7b", "olmoe-1b-7b-128g1.json"


def score(repo: Path, tree: Path) -> dict:
    reg = TC.registration(repo, PART)
    reg_file = TC.registration(repo, "e1cal")["pages"]["files"][FILE]
    tv = TC.C3.TimedView("CLEAN")
    out, why = {}, {}
    for label in ("e128g1", "e128g1r"):
        r, w = TC.timed_page(tree, MODEL, label, 1, tv, reg_file)
        why[label] = w
        if r is not None:
            out[label] = {(str(x["histogram"]), int(x["tiles"]), str(x["arm"])): float(x["ms_p50"])
                          for x in r.get("treads_table") or [] if x.get("ms_p50") and not x.get("lost")}
    pairs = []
    if len(out) == 2:
        pairs = [(out["e128g1"][k], out["e128g1r"][k]) for k in sorted(out["e128g1"]) if k in out["e128g1r"]]
    s = TC.rep_sigma(pairs)
    sp, src = TC.sigma_page_of({"REP": s})
    return {"registration": reg["name"], "part": PART, "pages": why, "REP": s, "sigma_page_used": sp, "source": src}


def lines(res: dict) -> list[str]:
    return [f"SCORED against {res['registration']}", f"  pages: {res['pages']}",
            f"REP: {res['REP']['verdict']} sigma_page {res['REP'].get('sigma_page')}; used {res['sigma_page_used']} ({res['source']})"]


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
