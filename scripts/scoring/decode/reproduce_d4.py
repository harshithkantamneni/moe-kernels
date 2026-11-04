#!/usr/bin/env python
"""The published decode trees reproduce their committed scores (registration 2026-11-04-decode-addendum4-gh200):
DESCRIPTIVE, changes no verdict.

    python scripts/scoring/decode/reproduce_d4.py <CAL2 results tree> <decode blind results tree>

The CAL2 tree under v4d' (score_cal_d3's rules, frozen) against scripts/scoring/decode/cal2.score.json (the selected
rho form, T0_GR, s_naive, s_block and the in-sample E1 statistics), and the decode blind tree under the committed
finalisation (score_blind_d3.score, frozen) against scripts/scoring/decode/blind.score.json (E1-DEC pooled and per
model), each to 1e-9 relative. Prints one line per tree and exits 0 when both reproduce, else 1. The next round's gate
is recorded as intent only and will name this check as its first clause.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d4common as D4  # noqa: E402


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if len(a) != 2:
        print(__doc__)
        return 2
    r = D4.reproduce(Path(a[0]), Path(a[1]))
    for k in ("CAL2", "decode blind"):
        print(f"{k}: {r[k]['verdict']}" + (": " + "; ".join(r[k]["why"]) if r[k]["why"] else ""))
    return 0 if r["verdict"] == "REPRODUCES" else 1


if __name__ == "__main__":
    sys.exit(main())
