#!/usr/bin/env python
"""Scores the decode stage's blind rental under the third addendum (registration 2026-10-29-decode-addendum3-gh200)
against finalise_decode_d3.py's output.

    python scripts/scoring/decode/score_blind_d3.py <results tree> <finalised.json> <out dir>

A wrapper over the frozen addendum-2 blind wrapper (score_blind_d2.score, which runs the frozen score_blind.score),
unchanged, with v4d' pricing every cell it prices itself (SIGMA-D, the knee rows) at the finalised fixes: the
decode blind registration as frozen, v4d' in place of v4d, and one rule changed (owner, 2026-11-03): G3-D LABEL IS
DESCRIPTIVE. It is computed exactly as registered and reported per model (G3D_LABEL), and never voids a model
(d3common.label_descriptive hands the frozen scorer a NOT SCORED verdict for it). Everything else stays as
registered: the power check first, E1-DEC pooled and per model at the finalised bars without addendum 2's
DESCRIPTIVE cells, a model CERTIFIED when its own bar and the pooled bar hold, GD2's void and G3-D shuffle's void.
The scope line (d3common.SCOPE) says so beside every CERTIFIED. Writes blind.score.json and blind.score.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import d3common as D3  # noqa: E402
import dcommon as DC  # noqa: E402
import score_blind_d2 as SB2  # noqa: E402

V4DP = D3.V4DP
LABEL_NOTE = "DESCRIPTIVE (addendum 3, owner 2026-11-03): reported per model, never a void"


def score(tree: Path, reg: dict, fin: dict, excluded: set | None = None) -> dict:
    fixes = D3.fixes_from((fin.get("frozen") or {}).get("model", {}).get("fixes"))
    store = {}
    with D3.v4dp(fixes), D3.label_descriptive(store):
        s = SB2.score(Path(tree), reg, fin, excluded)
    if s.get("verdict") == "NOT SCORED":
        return dict(s, addendum=D3.NAME, scope=D3.SCOPE)
    lab = D3.label_blind(store)
    g3 = {m: dict(x, label=dict(x["label"], verdict=lab[m]["verdict"], descriptive=LABEL_NOTE)) if m in lab and "label" in x else x
          for m, x in (s.get("G3D") or {}).items()}
    return dict(s, what="the decode stage's blind score (E1-DEC) under the third addendum", addendum=D3.NAME,
                addenda=[D2.NAME, D3.NAME], model=(fin.get("frozen") or {}).get("model"), scope=D3.SCOPE,
                G3D=g3, G3D_LABEL=lab)


def lines(s: dict) -> list[str]:
    L = SB2.lines(s)
    L[0] = L[0].replace("(addendum 2)", "(addendum 3, model v4d')")
    for m, x in (s.get("G3D_LABEL") or {}).items():
        L.append(f"G3-D label {m} (DESCRIPTIVE, never a void): {x['verdict']} margin {x['margin_us']} us mean {x['mean_us']} us")
    return L


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if len(a) != 3:
        print(__doc__)
        return 2
    fin = json.loads(Path(a[1]).read_text())
    if fin.get("addendum") != D3.NAME or "frozen" not in fin or (fin["frozen"].get("model") or {}).get("fixes") != V4DP.ALL.as_dict():
        print(f"REFUSED: {a[1]} was not finalised under {D3.NAME} (finalise_decode_d3.py) with v4d' at the registered fixes")
        return 2
    s = score(Path(a[0]), DC.registration(DC.REPO, "blind"), fin)
    out = Path(a[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / "blind.score.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    (out / "blind.score.txt").write_text("\n".join(lines(s)) + "\n")
    print("\n".join(lines(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
