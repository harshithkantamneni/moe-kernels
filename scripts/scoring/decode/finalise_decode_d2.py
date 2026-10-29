#!/usr/bin/env python
"""The blind rental's predictions under the second addendum (registration 2026-10-28-decode-addendum2-gh200),
finalised MECHANICALLY by the frozen finalise_decode.finalise, and the addendum's CAL plan.

    python scripts/scoring/decode/finalise_decode_d2.py <CAL score dir> <out.json> [--plan-out <plan file>]
    python scripts/scoring/decode/finalise_decode_d2.py --cal-plan <plan file>     # the addendum's CAL plan

REFUSES (exit 2, nothing written) unless the CAL score directory's cal.score.json was written by
score_cal_d2.py under this addendum (its 'addendum' field), its power check reads READ and its excluded cells
are the registered set (review m6), then exactly as finalise_decode.py: unless
its STOP (the addendum's stop rule) reads GO and its fit selected a rho form. The predictions cover every
blind main cell, the DESCRIPTIVE ones included (score_blind_d2.py reads them as SIGMA-D); the blind plan
is the frozen one (finalise_decode.blind_plan).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import finalise_decode as FN  # noqa: E402

Refused = FN.Refused


def finalise(score_dir: Path) -> dict:
    cal = json.loads((Path(score_dir) / "cal.score.json").read_text())
    if cal.get("addendum") != D2.NAME:
        raise Refused(f"{score_dir}/cal.score.json was not written by score_cal_d2.py under {D2.NAME}")
    if (cal.get("POWER") or {}).get("verdict") != "READ":
        raise Refused(f"the CAL score's power check reads {(cal.get('POWER') or {}).get('verdict')}, not READ "
                      f"(the CAL stop rule reads {(cal.get('STOP') or {}).get('verdict')})")
    want = sorted("/".join(k) for k in D2.excluded_keys(FN.DC.CAL_MODELS))
    if sorted(cal.get("excluded_cells") or []) != want:
        raise Refused("the CAL score's excluded cells are not the registered ones")
    doc = FN.finalise(score_dir)
    doc["addendum"] = D2.NAME
    doc["what"] += f"; under {D2.NAME}"
    doc["scope"] = D2.SCOPE
    doc["descriptive_cells"] = sorted("/".join(k) for k in D2.excluded_keys(FN.DC.BLIND_MODELS))
    return doc


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if a[:1] == ["--cal-plan"] and len(a) == 2:
        Path(a[1]).write_text(D2.cal_plan_d2())
        print(f"wrote {a[1]}")
        return 0
    plan_out = None
    if "--plan-out" in a:
        i = a.index("--plan-out")
        plan_out = Path(a[i + 1])
        del a[i:i + 2]
    if len(a) != 2:
        print(__doc__)
        return 2
    try:
        doc = finalise(Path(a[0]))
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return 2
    Path(a[1]).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {a[1]}: {len(doc['cells'])} cells, {doc['not_priced']} not priced, rho {doc['frozen']['rho_form']}, "
          f"{len(doc['descriptive_cells'])} DESCRIPTIVE")
    if plan_out is not None:
        plan_out.write_text(FN.blind_plan(doc["frozen"]))
        print(f"wrote {plan_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
