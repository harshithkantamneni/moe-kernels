#!/usr/bin/env python
"""Score rental B2 (docs/registered/2026-10-20-tileB2-e1tile-gh200.json) against its finalised predictions
(finalise_a2.py's file): scripts/scoring/tileA/score_tileB.py run unchanged inside a2common.scoring at the lock
and in the counter band the predictions file names (A2's), so the registration it reads is tileB2-e1tile, its
census check reads tileA2-census and rental B2's own census (label tb), and its byte pages are found under
lock<L>/.

    python scripts/scoring/tileA2/score_tileB2_a2.py <repo> <tree> <out> <predictions.json>

Writes <out>/tileB.score.{json,txt} (score_tileB's own stem); exits 0 whatever the verdicts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402
#: rental A's scorer by path: rental 1 and rental 4 also hold a score_floor.py and a score_dead.py
SB = A2.TC._load("tile_score_tileB", A2.TILEA / "score_tileB.py")

TC = A2.TC


def rules_of(pred_path: Path) -> A2.Rules:
    a2 = json.loads(Path(pred_path).read_text()).get("a2") or {}
    lock = float(a2["lock_mhz"])
    which = str((a2.get("band_in_force") or {}).get("which") or "primary")
    return A2.Rules.tileA2(lock, "fallback" if which.startswith("fallback") else "primary")


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 4:
        print(__doc__)
        return 2
    repo, tree, out, pred = (Path(x) for x in a)
    rules = rules_of(pred)
    with A2.scoring(rules):
        res = SB.score(repo, tree, pred)
        text = SB.lines(res)
    res["lock_mhz"] = rules.lock
    print(TC.C3.write_score(out, SB.PART, res, text + [f"LOCK {rules.lock:g} MHz; band {list(rules.band)}"]), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
