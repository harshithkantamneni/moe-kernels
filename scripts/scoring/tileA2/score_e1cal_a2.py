#!/usr/bin/env python
"""Score tile rental A2's E1-CAL pages (docs/registered/2026-10-20-tileA2-e1cal-gh200.json) and read rental B2's
stop rule (2026-10-20-tileB2-stop-gh200): rental A's pinned scorer scripts/scoring/tileA/score_e1cal.py,
run unchanged inside a2common.scoring at the lock A2 ran. v4's predictions are finalise.predict_pages'
under r3_timing_model_v4L.at_lock(L) (P1's rule: c x 1710 / L at every tile), from A2's own selections.

    python scripts/scoring/tileA2/score_e1cal_a2.py <repo> <tree> <out> [--lock L] [--rules tileA2|tileA]

Needs A2's census, floor, bytes and rep scores in <out> (the A2 wrappers, in that order). Writes
<out>/e1cal.score.{json,txt}.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402
#: rental A's scorer by path: rental 1 and rental 4 also hold a score_floor.py and a score_dead.py
SE = A2.TC._load("tile_score_e1cal", A2.TILEA / "score_e1cal.py")

TC = A2.TC
PART = SE.PART


def score(repo: Path, tree: Path, out: Path, rules: A2.Rules) -> tuple[dict, list[str]]:
    with A2.scoring(rules):
        res = SE.score(repo, tree, out)
    if not rules.label.startswith("tileA2"):
        return res, SE.lines(res)
    res["lock_mhz"] = rules.lock
    res["conversion"] = ("none: A2 held rung 1710, v4 itself" if rules.lock == A2.FIT_CLOCK_MHZ else
                         f"v4 at {rules.lock:g} MHz by P1's rule (c x 1710 / {rules.lock:g}); T0, bw, s_small, s_block, h, "
                         "rho, d and MVA2's eta, z0, z1 carried unconverted (UNTESTED here; tileA2-convl)")
    return res, SE.lines(res) + [f"LOCK {rules.lock:g} MHz (v4 at_lock {rules.lock:g})"]


def main(argv=None) -> int:
    a, rules = A2.parse_cli(argv if argv is not None else sys.argv[1:], __doc__)
    if a is None:
        return A2.usage(__doc__)
    repo, tree, out = (Path(x) for x in a)
    res, lines = score(repo, tree, out, rules)
    print(TC.C3.write_score(out, PART, res, lines), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
