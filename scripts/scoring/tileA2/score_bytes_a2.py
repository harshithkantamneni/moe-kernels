#!/usr/bin/env python
"""Score tile rental A2's byte pages (docs/registered/2026-10-20-tileA2-bytes-gh200.json): rental A's pinned
scorer scripts/scoring/tileA/score_bytes.py, run unchanged inside a2common.scoring at the lock A2 ran and
in the counter band in force (the one <out>/floor.score.json recorded: run score_floor_a2 first).

    python scripts/scoring/tileA2/score_bytes_a2.py <repo> <tree> <out> [--lock L] [--rules tileA2|tileA]

Reads under <tree>: <date>-<card>-<model>-<label>-r3-counters/lock<L>/r3c-g<G>.json for b128g1 (OLMoE,
Qwen2-57B), b64g1 and b128g16 (Mixtral 8x7B). The key test runs on every scored w1 row the band keeps;
rental A's 53 clock-dropped rows stay unread (DESIGN-A2 7f). Writes <out>/bytes.score.{json,txt}.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402
#: rental A's scorer by path: rental 1 and rental 4 also hold a score_floor.py and a score_dead.py
SB = A2.TC._load("tile_score_bytes", A2.TILEA / "score_bytes.py")

TC = A2.TC
PART = SB.PART


def score(repo: Path, tree: Path, out: Path, rules: A2.Rules) -> tuple[dict, list[str]]:
    use = A2.band_in_force(out, rules)
    with A2.scoring(use):
        res = SB.score(repo, tree)
    if not rules.label.startswith("tileA2"):
        return res, SB.lines(res)
    res["lock_mhz"] = rules.lock
    res["band_in_force"] = {"band": [A2.r6(x) for x in use.band], "rules": use.label}
    return res, SB.lines(res) + [f"LOCK {rules.lock:g} MHz; band {list(use.band)}"]


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
