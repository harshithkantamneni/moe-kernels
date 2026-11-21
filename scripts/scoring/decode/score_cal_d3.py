#!/usr/bin/env python
"""Scores the decode stage's CAL2 rental under the third addendum (registration 2026-10-29-decode-addendum3-gh200,
over 2026-10-28-decode-addendum2-gh200, 2026-10-14-decodeC-cal-gh200 and 2026-10-20-decode-tilesource-gh200).

    python scripts/scoring/decode/score_cal_d3.py <results tree> <out dir> --tile-scores scripts/scoring/tileA2

A wrapper over the frozen addendum-2 wrapper (score_cal_d2.score, which runs the frozen score_cal.score), unchanged:
  model      v4d' (r3_timing_model_v4dp, fixes W and R; no align term) in place of v4d wherever a cell is priced
             (d3common.v4dp): the fit (v4d's registered T0_GR, s_naive, s_block and the rho selection, on the kept
             cells), E1-CAL-IN-SAMPLE, G3-D, the rivals' training rows, SIGMA-D and the knee rows
  stop       d3common.stop_rule_d3: the frozen stop rule with the G3-D label line DESCRIPTIVE (owner decision 2);
             power, exclusions, scope and SIGMA-D as addendum 2
  DESCRIPTIVE  the G3-D label reading (G3D_LABEL); v4d as registered on the same tree (V4D_READING: its fit, E1 and
             stop, beside); the settling units M1 to M4 (SETTLE), at the CAL2 fit's constants; fix R's ratio and fix
             A's slopes re-read on this tree (CONSTANTS_REPLICATE), never used to change v4d'
Writes cal.score.json (finalise_decode_d3.py reads it) and cal.score.txt.
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
import score_cal as SC  # noqa: E402
import score_cal_d2 as SC2  # noqa: E402

r6 = DC.r6
V4D, V4DP = DC.V4D, D3.V4DP


def score(tree: Path, reg: dict, tparams, tsource: str = "", fixes: "V4DP.Fixes" = V4DP.ALL,
          settle: bool = True) -> dict:
    tree = Path(tree)
    what = "the decode stage's CAL2 score under the third addendum"
    with D3.v4dp(fixes):
        s = SC2.score(tree, reg, tparams, tsource)
    if s.get("verdict") == "NOT SCORED":
        return dict(s, what=what, addendum=D3.NAME, addenda=[D2.NAME, D3.NAME])
    out = dict(s)
    out.update(what=what, addendum=D3.NAME, addenda=[D2.NAME, D3.NAME],
               model={"name": "v4d'", "module": "scripts/r3_timing_model_v4dp.py", "fixes": fixes.as_dict(),
                      "constants": V4DP.describe()})
    out["STOP_D2"] = dict(s["STOP"], label="DESCRIPTIVE: addendum 2's reading of the frozen stop rule (G3-D label a clause)")
    out["G3D_LABEL"] = D3.label_effect(s)
    out["STOP"] = D3.stop_rule_d3(out)
    out["scope"] = D3.SCOPE                     # review m4: addendum 3's scope line, the label rule in it
    v4d = SC2.score(tree, reg, tparams, tsource)
    out["V4D_READING"] = {"label": "DESCRIPTIVE: v4d as registered (no fix) on this tree",
                          "selected": v4d["FIT"]["selected"],
                          "forms": {f: {k: x.get(k) for k in ("verdict", "T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms", "in_sample")}
                                    for f, x in v4d["FIT"]["forms"].items()},
                          "E1_CAL_IN_SAMPLE": v4d["E1_CAL_IN_SAMPLE"], "STOP_D3_RULE": D3.stop_rule_d3(v4d)["verdict"]}
    sel = out["FIT"]["selected"]
    prm = V4D.params_from(out["FIT"]["forms"][sel], rho_form=sel, tile=tparams) if sel is not None else None
    if settle:
        out["SETTLE"] = D3.settle_readings(tree, prm)
        try:
            out["CONSTANTS_REPLICATE"] = {
                "label": "DESCRIPTIVE: fix R's ratio and the dropped fix A's slopes re-read on this tree by the registered rules; "
                         "v4d''s constants are the registered ones whatever these read",
                "R": {k: v for k, v in D3.w1_marginal(tree, prm).items() if k != "pairs"} if prm is not None else None,
                "A": D3.align_slopes(tree, reg)}
        except Exception as exc:                                         # noqa: BLE001
            out["CONSTANTS_REPLICATE"] = {"verdict": f"NOT READ: {type(exc).__name__}: {exc}"}
    return out


def lines(s: dict) -> list[str]:
    if s.get("verdict") == "NOT SCORED":
        return ["DECODE STAGE, CAL2 RENTAL (addendum 3): NOT SCORED", f"power: {s['POWER']['verdict']}",
                f"STOP: {s['STOP']['verdict']}" + "".join(f"\n  {w}" for w in s["STOP"]["why"])]
    L = SC2.lines(s)
    L[0] = "DECODE STAGE, CAL2 RENTAL (addendum 3): score, model v4d' (fixes W, R; 0 fitted constants added)"
    i = next(j for j, ln in enumerate(L) if ln.startswith("STOP:"))     # one element, its reasons on inner lines
    L[i] = (f"STOP (addendum 3, G3-D label DESCRIPTIVE): {s['STOP']['verdict']}"
            + "".join(f"\n  {w}" for w in s["STOP"]["why"])
            + "".join(f"\n  DESCRIPTIVE: {w}" for w in s["STOP"]["descriptive"]))
    g = s["G3D_LABEL"]
    L.append(f"G3-D label (DESCRIPTIVE): {g['verdict']} margin {g['margin_us']} us; "
             + "; ".join(f"{m} mean {x['mean_us']} us over {x['pairs']} pairs" for m, x in g["per_model"].items()))
    v = s["V4D_READING"]
    L.append(f"v4d as registered (DESCRIPTIVE): E1 {v['E1_CAL_IN_SAMPLE']['verdict']} {v['E1_CAL_IN_SAMPLE']['stats']}; "
             f"addendum-3 stop {v['STOP_D3_RULE']}")
    st_ = s.get("SETTLE") or {}
    for k in ("M1", "M2", "M3", "M4"):
        x = st_.get(k) or {}
        L.append(f"settle {k} (DESCRIPTIVE): {x.get('verdict')}" + (f"; {x.get('agree')}" if k == "M4" else "")
                 + (f"; q {x.get('q')}" if k == "M3" and "q" in x else ""))
    return L


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    ts = None
    if "--tile-scores" in a:
        i = a.index("--tile-scores")
        ts = Path(a[i + 1])
        del a[i:i + 2]
    if len(a) != 2:
        print(__doc__)
        return 2
    if ts is None:
        print("REFUSED: --tile-scores <A2 score dir> is required (the first addendum binds scripts/scoring/tileA2)")
        return 2
    tree, out = Path(a[0]), Path(a[1])
    reg = DC.registration(DC.REPO, "cal")
    tp, src = SC.tile_params(ts)
    s = score(tree, reg, tp, src)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cal.score.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    (out / "cal.score.txt").write_text("\n".join(lines(s)) + "\n")
    print("\n".join(lines(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
