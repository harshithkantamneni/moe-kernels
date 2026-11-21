#!/usr/bin/env python
"""Score tile rental A2's replicate (docs/registered/2026-10-20-tileA2-rep-gh200.json): rental A's pinned scorer
scripts/scoring/tileA/score_rep.py, run unchanged inside a2common.scoring at the lock A2 ran (G4 at L - 5;
the r7-median-run3 rebuild reads each page's own lock).

    python scripts/scoring/tileA2/score_rep_a2.py <repo> <tree> <out> [--lock L] [--rules tileA2|tileA]

Reads under <tree>: gaps-<card>-olmoe-1b-7b-e128g1 and -e128g1r. A2's sigma_page when MEASURED; dropped,
lost or under 3 shared cells: the registered fallback, rental 3's 0.18% (tcommon.sigma_page_of; rental A's
SEEN 0.1275% is never copied in). Writes <out>/rep.score.{json,txt}.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402
#: rental A's scorer by path: rental 1 and rental 4 also hold a score_floor.py and a score_dead.py
SR = A2.TC._load("tile_score_rep", A2.TILEA / "score_rep.py")

TC = A2.TC
PART = SR.PART


def score(repo: Path, tree: Path, out: Path, rules: A2.Rules) -> tuple[dict, list[str]]:
    with A2.scoring(rules):
        res = SR.score(repo, tree)
    if not rules.label.startswith("tileA2"):
        return res, SR.lines(res)
    res["lock_mhz"] = rules.lock
    return res, SR.lines(res) + [f"LOCK {rules.lock:g} MHz"]


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
