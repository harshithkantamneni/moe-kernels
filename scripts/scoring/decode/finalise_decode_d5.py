#!/usr/bin/env python
"""The new blind rental's predictions under the fifth addendum (registration 2026-11-04-decode-addendum5-gh200),
finalised MECHANICALLY from DS's readings and the published CAL2 tree by the registered rules.

    python scripts/scoring/decode/finalise_decode_d5.py <DS readings dir> <CAL2 results tree> \\
        <decode blind results tree> <out.json> [--plan-out scripts/plans/rentalD5-blind-2026-11.plan]

REFUSES (exit 2, nothing written) unless the launch gate (gate_d5.evaluate, the same code the operator runs) reads GO:
the DS readings give the registered v5d, both published trees reproduce their committed scores, CAL2 under v5d reads GO
by the addendum-3 stop rule, and the SEEN decode blind re-scored under v5d reads a pooled rms under 4.0% over its 220
cells (a weak sanity floor on seen pages, never a test). Then exactly finalise_decode's rules (frozen:
finalise_decode.frozen and .predict) with v5d pricing every cell, on the new blind models' registered pages: per main
cell v5d's time and every rival's. WHAT IT FREEZES: the model (v5d: v4d''s fixes W and R and the registered terms), the
three constants and the rho form of the CAL2 refit under v5d, the tile stage's v4 Params as the CAL2 scorer used them,
the OWNER's bars (3% rms, 8% worst, 1.5% |mean|; not tightened), sigma_page, the GD2 band, the G3-D margin rule, the
rivals' CAL2 fits under v5d, the G3-D rule (shuffle voids; label DESCRIPTIVE), the scope line (d3common.SCOPE) and the
blind plan (the registered unit lines under a finalised header).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import d3common as D3  # noqa: E402
import d5common as D5  # noqa: E402
import finalise_decode as FN  # noqa: E402

Refused = FN.Refused
V4DP, V5D = D3.V4DP, D5.V5D


def finalise(ds_dir: Path, cal2_tree: Path, blind_tree: Path, **kw) -> dict:
    import gate_d5 as G
    g, cal = G.evaluate(ds_dir, cal2_tree, blind_tree, **kw)
    if g["verdict"] != "GO":
        raise Refused("the launch gate reads NO-GO: " + "; ".join(g["why"]))
    terms = V5D.Terms.from_dict(g["terms"])
    fz = FN.frozen(cal)
    fz["bars"] = dict(D5.BARS_OWNER)
    fz["model"] = {"name": "v5d", "module": "scripts/r3_timing_model_v5d.py", "fixes": V4DP.ALL.as_dict(),
                   "terms": terms.as_dict(), "constants": V5D.describe(terms) | {"W, R": V4DP.describe()}}
    fz["g3d_label"] = ("DESCRIPTIVE (addendum 3, owner 2026-11-03; kept by addendum 5): computed as registered and "
                       "reported per model; never voids a model's certification. G3-D shuffle voids as registered")
    with D5.v5d(terms), D3.v4dp(V4DP.ALL), D5.blind5():
        rows = FN.predict(fz, models=D5.NEW_BLIND)
        desc = sorted("/".join(k) for k in D2.excluded_keys(D5.NEW_BLIND))
    return {"what": "the new blind rental's predictions, finalised mechanically from DS's readings and the CAL2 tree "
                    f"under {D5.NAME}",
            "addendum": D5.NAME, "frozen": fz, "cells": rows, "not_priced": sum(1 for r in rows if r["ms"]["v4d"] is None),
            "scope": D3.SCOPE, "descriptive_cells": desc, "GATE": g,
            "keys": "ms['v4d'] holds v5d's time (the frozen scorers read that key); every rival priced by v5d with the "
                    "rival's own change (rho other, NAIVE-NODEDUP, eager), roofline, line and OLS as frozen"}


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    plan_out = None
    if "--plan-out" in a:
        i = a.index("--plan-out")
        plan_out = Path(a[i + 1])
        del a[i:i + 2]
    if len(a) != 4:
        print(__doc__)
        return 2
    try:
        doc = finalise(Path(a[0]), Path(a[1]), Path(a[2]))
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return 2
    Path(a[3]).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {a[3]}: {len(doc['cells'])} cells, {doc['not_priced']} not priced, rho {doc['frozen']['rho_form']}, "
          f"model v5d, {len(doc['descriptive_cells'])} DESCRIPTIVE, gate {doc['GATE']['verdict']}")
    if plan_out is not None:
        plan_out.write_text(D5.blind_plan(doc["frozen"]))
        print(f"wrote {plan_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
