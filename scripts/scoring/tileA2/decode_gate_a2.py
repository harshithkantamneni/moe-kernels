#!/usr/bin/env python
"""Whether the decode stage may use tile rental A2's constants (docs/registered/2026-10-20-decode-tilesource-gh200; the
owner's decision of 2026-10-20). New file: the frozen decode scorer (scripts/scoring/decode/score_cal.py) stays
byte-identical, and this gate is run before the decode CAL rental is launched.

    python scripts/scoring/tileA2/decode_gate_a2.py <A2 score dir>      # scripts/scoring/tileA2 once A2 is scored

GO (exit 0) iff A2's floor calibration is SCORED (floor.score.json: calibration CAL-A, a joint c_128 and F
fit) and tileA/finalise.selections reads the directory without refusing (the census not refused; a 64x128 fit
or a selected accounting), WHATEVER ELSE A2's stop rule reads (A2's STOP verdict is printed beside). Then it
prints the decode scorer's command: score_cal.py <tree> <out> --tile-scores <A2 score dir>.
WAIT (exit 3) otherwise: decode does not use A2's constants and the decode CAL rental waits. Exit 2 refused.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402

FN = A2.TC._load("finalise", A2.TILEA / "finalise.py")


def gate(score_dir: Path) -> dict:
    fl = A2.read_score(score_dir, "floor") or {}
    cal = fl.get("calibration") or {}
    e1 = A2.read_score(score_dir, "e1cal") or {}
    stop = str((e1.get("STOP") or {}).get("verdict", "no e1cal score"))
    out = {"score_dir": str(score_dir), "a2_stop_rule (printed, not read)": stop, "lock_mhz": fl.get("lock_mhz")}
    if cal.get("verdict") != "CAL-A" or not cal.get("c"):
        return dict(out, verdict="WAIT", why=f"A2's floor calibration is not scored ({cal.get('verdict', 'no floor score')})")
    try:
        sel = FN.selections(Path(score_dir))
    except FN.Refused as exc:
        return dict(out, verdict="WAIT", why=f"the calibration exists but finalise.selections refuses: {exc}")
    return dict(out, verdict="GO", c128=sel["c128"], variant=sel["variant"] or "v4cal", f_form=sel["f_form"],
                command=f"python scripts/scoring/decode/score_cal.py <tree> <out> --tile-scores {score_dir}")


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 1:
        print(__doc__)
        return 2
    g = gate(Path(a[0]))
    print(f"DECODE TILE SOURCE: {g['verdict']} ({g.get('why') or g.get('command')}); A2's stop rule: "
          f"{g['a2_stop_rule (printed, not read)']}")
    return 0 if g["verdict"] == "GO" else 3


if __name__ == "__main__":
    sys.exit(main())
