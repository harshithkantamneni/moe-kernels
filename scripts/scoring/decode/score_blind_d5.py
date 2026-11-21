#!/usr/bin/env python
"""Scores the new blind rental under the fifth addendum (registration 2026-11-04-decode-addendum5-gh200) against
finalise_decode_d5.py's output.

    python scripts/scoring/decode/score_blind_d5.py <results tree> <finalised.json> <out dir>

A wrapper over the addendum-3 blind wrapper (score_blind_d3.score, which runs score_blind_d2.score and the frozen
score_blind.score), unchanged, with three swaps: v5d at the finalised terms prices every cell the wrappers price
themselves (SIGMA-D, the knee rows); the frozen scorer's model list and page paths are the new blind models' and their
registered pages (d5common.blind5); and its registration is this addendum's blind block (units and pages in
decodeB-blind's form). The rules as frozen: the power check first, E1-DEC pooled and per model at the finalised (owner)
bars without addendum 2's DESCRIPTIVE second-tile cells, a model CERTIFIED when its own bar and the pooled bar hold,
GD2's void and G3-D shuffle's void, G3-D label DESCRIPTIVE (reported, never a void), the scope line. Writes
blind.score.json and blind.score.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d5common as D5  # noqa: E402
import dcommon as DC  # noqa: E402
import score_blind_d3 as SB3  # noqa: E402

V5D = D5.V5D
BLOCK = "blind (owner, 2026-11-04, D4-1; registered in addendum 5)"


def registration() -> dict:
    return json.loads((DC.REPO / "docs" / "registered" / f"{D5.NAME}.json").read_text())[BLOCK]


def score(tree: Path, reg: dict, fin: dict, excluded: set | None = None) -> dict:
    terms = V5D.Terms.from_dict((fin.get("frozen") or {}).get("model", {}).get("terms"))
    with D5.v5d(terms), D5.blind5():
        s = SB3.score(Path(tree), reg, fin, excluded)
    if s.get("verdict") == "NOT SCORED":
        return dict(s, addendum=D5.NAME)
    return dict(s, what="the decode stage's new blind score (E1-DEC) under the fifth addendum", addendum=D5.NAME,
                addenda=list(s.get("addenda") or []) + [D5.NAME], bars_source=fin["frozen"]["bars"].get("source"))


def lines(s: dict) -> list[str]:
    L = SB3.lines(s)
    L[0] = L[0].replace("(addendum 3, model v4d')", "(addendum 5, model v5d, new models)").replace(
        "(addendum 2)", "(addendum 5, model v5d, new models)")
    return L


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if len(a) != 3:
        print(__doc__)
        return 2
    fin = json.loads(Path(a[1]).read_text())
    if (fin.get("addendum") != D5.NAME or (fin.get("frozen") or {}).get("model", {}).get("name") != "v5d"
            or (fin.get("GATE") or {}).get("verdict") != "GO"):
        print(f"REFUSED: {a[1]} was not finalised under {D5.NAME} (finalise_decode_d5.py) with v5d on a gate GO")
        return 2
    s = score(Path(a[0]), registration(), fin)
    out = Path(a[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / "blind.score.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    (out / "blind.score.txt").write_text("\n".join(lines(s)) + "\n")
    print("\n".join(lines(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
