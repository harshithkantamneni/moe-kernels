#!/usr/bin/env python
"""Score tile rental A2's tile census (docs/registered/2026-10-20-tileA2-census-gh200.json): the NAMED MERGE STEP
(census_a2.merge, REVIEW-A2 6a) then rental A's pinned scorer scripts/scoring/tileA/score_census.py on the
merged file.

    python scripts/scoring/tileA2/score_census_a2.py <repo> <tree> <out> [--lock L] [--rules tileA2|tileA]

Reads under <tree> A2's <date>-<card>-tile-census-<model>-ta2/tile-census.json (scripts/tile_census_a2.py)
and, from the repo, rental A's census (docs/registered/2026-10-20-tileA2-hist/rentalA-tile-census.json,
its registered sha256). Writes the merged census to <out>/census-merged/<date>-<card>-tile-census-merged-ta2m/
tile-census.json (rental A's entries, held to A2's by the equality gate, plus A2's deepseek-v2-lite entries
at tread 3 when the OLMoE n 1 / n 3 rule reads SAME), then <out>/census.score.{json,txt} scored over it.
With no A2 census the score reads NOT SCORED (and the stop rule STOPs, as tileA-census). --rules tileA
scores rental A's own census (label ta), unchanged.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402
import census_a2 as CA  # noqa: E402
#: rental A's scorer by path: rental 1 and rental 4 also hold a score_floor.py and a score_dead.py
SC = A2.TC._load("tile_score_census", A2.TILEA / "score_census.py")

TC = A2.TC
PART = SC.PART


def score(repo: Path, tree: Path, out: Path, rules: A2.Rules) -> tuple[dict, list[str]]:
    if not rules.label.startswith("tileA2"):
        with A2.scoring(rules):
            res = SC.score(repo, tree, "ta")
        return res, SC.lines(res)
    reg = A2.registration(repo, "census")
    ref_path = Path(repo) / CA.RENTAL_A_CENSUS
    sha = hashlib.sha256(ref_path.read_bytes()).hexdigest()
    p = CA.find(tree)
    merged_root = Path(out) / "census-merged"
    if sha != reg["rental_A_census"]["sha256"]:
        rec = {"verdict": f"NOT SCORED: {CA.RENTAL_A_CENSUS} is not the registered file (sha256 {sha})"}
    elif p is None:
        rec = {"verdict": "NOT SCORED: no A2 tile-census.json (label ta2)"}
    else:
        doc, rec = CA.merge(CA.rental_a(repo), json.loads(p.read_text()))
        d = merged_root / f"{A2.DATE}-{A2.CARD}-tile-census-merged-{CA.MERGED_LABEL}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "tile-census.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        rec = dict(rec, verdict="MERGED", a2_census=str(p), merged=str(d / "tile-census.json"))
    with A2.scoring(rules):
        res = SC.score(repo, merged_root, CA.MERGED_LABEL)
    res["merge"] = rec
    res["lock_mhz"] = rules.lock
    lines = SC.lines(res) + [f"MERGE: {rec['verdict']}"]
    if rec.get("equality_gate"):
        lines.append(f"  equality gate {rec['equality_gate']['verdict']} over {rec['equality_gate']['compared']} rental-A entries")
        lines.append(f"  OLMoE n 1 / n 3: {rec['olmoe_n1_n3_rule']['verdict']}")
    return res, lines


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
