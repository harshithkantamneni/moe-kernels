#!/usr/bin/env python
"""Score rental 5's lever 4a, BK 128 timed (docs/registered/2026-10-07-rental5-bk128-gh200.json):
a blind confirmation of the occupancy-3 floor on OLMoE G = 64 NATIVE.

    python scripts/scoring/rental5/score_bk128.py <repo> <tree> <out>

Reads gaps-<card>-olmoe-1b-7b-bk64 and -bk128 under <tree> (the G = 64 page of each). Writes
<out>/bk128.score.{json,txt}; exits 0 whatever the verdicts.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5common as C5  # noqa: E402

C3 = C5.C3
PART = "bk128"
MODEL = "olmoe-1b-7b"
MIN_CELLS = 4


def _page(tree: Path, label: str, want: tuple, tview):
    root = C3.find_gaps(tree, MODEL, label)
    pages = [(p, r) for p, r in C3.timed_pages(root) if C3.group_of(r) == 64]
    if len(pages) != 1:
        return None, f"NOT SCORED: {len(pages)} G = 64 pages under gaps-{MODEL}-{label}"
    _p, r = pages[0]
    pin = r.get("pinned") or {}
    got = (int(pin.get("BLOCK_SIZE_K") or 0), int(pin.get("num_stages") or 0))
    if got != tuple(want):
        return None, f"NOT SCORED: the page ran BK {got[0]} s{got[1]}, not BK {want[0]} s{want[1]}"
    r = tview.admit(f"{MODEL}-{label}", r)
    if r is None:
        return None, "NOT SCORED: the page is out of this view"
    return r, "counted"


def score_view(repo: Path, tree: Path, _fv, tview) -> dict:
    reg = C5.registration(repo, PART)
    res = {"registration": C5.NAMES[PART], "pages": {}}
    got = {}
    for label, want in reg["pages"]["labels"].items():
        r, use = _page(tree, label, want, tview)
        res["pages"][label] = use
        if r is not None:
            got[label] = C3.treads_ms(r)
    if set(got) != {"bk64", "bk128"}:
        res["verdict"] = "NOT SCORED: the bk64 / bk128 pair is incomplete"
        return res
    rows = []
    for n, pr in reg["predicted_q"].items():
        k = ("native", int(n))
        if k not in got["bk64"] or k not in got["bk128"]:
            continue
        q = got["bk128"][k] / got["bk64"][k]
        rows.append({"n": int(n), "q_meas": q, "e": {h: q / pr[h] - 1 for h in ("H_C", "SYNC", "PS")}})
    res["cells"] = rows
    if len(rows) < MIN_CELLS:
        res["verdict"] = f"NOT SCORED: {len(rows)} of n = 4..9 on both pages, under {MIN_CELLS}"
        return res
    rms = math.sqrt(sum(r["e"]["H_C"] ** 2 for r in rows) / len(rows))
    bound = 3 * math.sqrt(2) * C5.SIGMA_PAGE
    far = sum(abs(r["e"]["SYNC"]) > bound for r in rows)
    res["H_C"] = {"rms": rms, "verdict": "HOLDS" if rms <= 0.01 else "FAILS"}
    res["SYNC"] = {"cells_beyond": far, "bound": bound, "verdict": "EXCLUDED" if far >= MIN_CELLS else "NOT EXCLUDED"}
    res["PS"] = {"rms": math.sqrt(sum(r["e"]["PS"] ** 2 for r in rows) / len(rows)), "verdict": "PRINTED"}
    res["verdict"] = f"H_C {res['H_C']['verdict']}; SYNC {res['SYNC']['verdict']}"
    res["secondary"] = C5.secondary([{"pred": reg["predicted_q"][str(r["n"])]["H_C"], "meas": r["q_meas"],
                                      "cluster": "board", "stratum": "native"} for r in rows])
    return res


def score(repo: Path, tree: Path) -> dict:
    return C3.two_views(lambda fv, tv: score_view(repo, tree, fv, tv), timed_ignore=("V5",), repo=repo)


def lines(res: dict) -> list[str]:
    out = [f"RENTAL 5 BK 128 TIMED (blind confirmation of the occupancy-3 floor), {res['registration']}"]
    for k, u in res["pages"].items():
        out.append(f"  page {k}: {u}")
    for r in res.get("cells") or []:
        out.append(f"  n={r['n']}: q {r['q_meas']:.4f} | " + "  ".join(f"{h} {100 * x:+.2f}%" for h, x in r["e"].items()))
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
