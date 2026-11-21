#!/usr/bin/env python
"""Score tile rental A2's dead-CTA pages (docs/registered/2026-10-20-tileA2-dead-gh200.json): rental A's pinned
scorer scripts/scoring/tileA/score_dead.py, run unchanged inside a2common.scoring at the lock A2 ran (the
laws at the measured c_128 under r3_timing_model_v4L.at_lock(L): the dead window's c converted, d in ns as P1).
SEEN-REPLICATE, DESCRIPTIVE (REVIEW-A2 5 iv): rental A measured this lever (delta 20.006 us, se 0.612,
D, D2, SLOT and d 1.06 EXCLUDED), so A2's reading is a replicate, not a test.

    python scripts/scoring/tileA2/score_dead_a2.py <repo> <tree> <out> [--lock L] [--rules tileA2|tileA]

Reads under <tree>: gaps-<card>-olmoe-1b-7b-d9 and -d15; <out>/floor.score.json and rep.score.json.
Writes <out>/dead.score.{json,txt}.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402
#: rental A's scorer by path: rental 1 and rental 4 also hold a score_floor.py and a score_dead.py
SD = A2.TC._load("tile_score_dead", A2.TILEA / "score_dead.py")

TC = A2.TC
PART = SD.PART


def score(repo: Path, tree: Path, out: Path, rules: A2.Rules) -> tuple[dict, list[str]]:
    with A2.scoring(rules):
        res = SD.score(repo, tree, out)
    if not rules.label.startswith("tileA2"):
        return res, SD.lines(res)
    res["lock_mhz"] = rules.lock
    res["label"] = "SEEN-REPLICATE (DESCRIPTIVE): rental A measured this lever; A2 replicates it, no test"
    return res, SD.lines(res) + [f"LOCK {rules.lock:g} MHz; {res['label']}"]


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
