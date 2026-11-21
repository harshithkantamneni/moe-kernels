#!/usr/bin/env python
"""The blind rental's predictions under the third addendum (registration 2026-10-29-decode-addendum3-gh200),
finalised MECHANICALLY from the CAL2 score by the frozen rules, and the addendum's CAL2 plan.

    python scripts/scoring/decode/finalise_decode_d3.py <CAL2 score dir> <out.json> [--plan-out <plan file>]
    python scripts/scoring/decode/finalise_decode_d3.py --cal-plan <plan file>     # the CAL2 plan

REFUSES (exit 2, nothing written) unless the score directory's cal.score.json was written by score_cal_d3.py under
this addendum (its 'addendum' field) with v4d' at the registered fixes, its power check reads READ, its excluded
cells are addendum 2's registered set, its STOP (the addendum-3 rule: G3-D label DESCRIPTIVE) reads GO and its fit
selected a rho form. Then exactly finalise_decode.finalise's rules (frozen: finalise_decode.frozen and .predict),
with v4d' pricing every cell: per blind main cell (the DESCRIPTIVE second-tile cells included) v4d''s time and
every rival's. WHAT IT FREEZES: the model (v4d', its three fixes and their constants), the three CAL2 constants
and the rho form, the tile stage's v4 Params as the CAL2 scorer used them, the blind bars (REP's tightened bars
from the kept cells' replicate), sigma_page, the GD2 band, the G3-D margin rule, the rivals' CAL2 fits (line,
OLS), the G3-D rule (shuffle voids as registered; label DESCRIPTIVE, reported per model and never a void: owner,
2026-11-03), the scope line (d3common.SCOPE), and the frozen blind plan (finalise_decode.blind_plan, the units of
decodeB-blind unchanged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import d3common as D3  # noqa: E402
import finalise_decode as FN  # noqa: E402

Refused = FN.Refused
V4DP = D3.V4DP


def finalise(score_dir: Path) -> dict:
    cal = json.loads((Path(score_dir) / "cal.score.json").read_text())
    if cal.get("addendum") != D3.NAME:
        raise Refused(f"{score_dir}/cal.score.json was not written by score_cal_d3.py under {D3.NAME}")
    if (cal.get("POWER") or {}).get("verdict") != "READ":
        raise Refused(f"the CAL2 score's power check reads {(cal.get('POWER') or {}).get('verdict')}, not READ")
    want = sorted("/".join(k) for k in D2.excluded_keys(FN.DC.CAL_MODELS))
    if sorted(cal.get("excluded_cells") or []) != want:
        raise Refused("the CAL2 score's excluded cells are not the registered ones")
    fixes = (cal.get("model") or {}).get("fixes")
    if fixes != V4DP.ALL.as_dict():
        raise Refused(f"the CAL2 score priced v4d' at fixes {fixes}, not the registered {V4DP.ALL.as_dict()}")
    fz = FN.frozen(cal)
    fz["model"] = {"name": "v4d'", "module": "scripts/r3_timing_model_v4dp.py", "fixes": V4DP.ALL.as_dict(),
                   "constants": V4DP.describe()}
    fz["g3d_label"] = ("DESCRIPTIVE (addendum 3, owner 2026-11-03): computed as registered and reported per model; "
                       "never voids a model's certification. G3-D shuffle voids as registered")
    with D3.v4dp(V4DP.ALL):
        rows = FN.predict(fz)
    return {"what": "the blind rental's predictions, finalised mechanically from the CAL2 score (decodeB-blind, "
                    f"with v4d' for v4d); under {D2.NAME} and {D3.NAME}",
            "addendum": D3.NAME, "frozen": fz, "cells": rows, "not_priced": sum(1 for r in rows if r["ms"]["v4d"] is None),
            "scope": D3.SCOPE, "descriptive_cells": sorted("/".join(k) for k in D2.excluded_keys(FN.DC.BLIND_MODELS)),
            "keys": "ms['v4d'] holds v4d''s time (the frozen scorers read that key); every rival priced by v4d' "
                    "with the rival's own change (rho other, NAIVE-NODEDUP, eager), roofline, line and OLS as frozen"}


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if a[:1] == ["--cal-plan"] and len(a) == 2:
        Path(a[1]).write_text(D3.cal2_plan())
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
          f"model v4d', {len(doc['descriptive_cells'])} DESCRIPTIVE")
    if plan_out is not None:
        plan_out.write_text(FN.blind_plan(doc["frozen"]))
        print(f"wrote {plan_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
