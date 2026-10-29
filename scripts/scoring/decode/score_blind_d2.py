#!/usr/bin/env python
"""Scores the decode stage's blind rental under the second addendum (registration
2026-10-28-decode-addendum2-gh200, over 2026-10-14-decodeB-blind-gh200) against finalise_decode_d2.py's output.

    python scripts/scoring/decode/score_blind_d2.py <results tree> <finalised.json> <out dir>

A wrapper: the frozen scorer (score_blind.score) runs unchanged on every cell, GD2 and G3-D included (a
G3-D pair holds the same counts or multiset on both sides). Then E1-DEC (pooled and per model), the
certification and the rivals are re-read without the addendum's DESCRIPTIVE cells (d2common.excluded_keys
on the five blind main pages), with the frozen rules (dcommon.bar, tcommon.rival_rule). First the
session's power-set.txt must read a GPU Current Power Limit of 700 W, else nothing is scored.
DESCRIPTIVE beside: SIGMA-D (the second-tile cells against v4d at sigma 1 and the L2-served estimate) and
the knee rows (p 5.75 and 3.75), at the finalised constants. Writes blind.score.json and blind.score.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import dcommon as DC  # noqa: E402
import finalise_decode as FN  # noqa: E402
import score_blind as SB  # noqa: E402

r6 = DC.r6


def score(tree: Path, reg: dict, fin: dict, excluded: set | None = None) -> dict:
    what = "the decode stage's blind score (E1-DEC) under the second addendum"
    power = D2.power_check(Path(tree))
    if power["verdict"] != "READ":
        return D2.not_scored(what, power)
    excl = D2.excluded_keys(DC.BLIND_MODELS) if excluded is None else set(excluded)
    full = SB.score(Path(tree), reg, fin)
    fz = fin["frozen"]
    bars = full["bars"]
    rows = [dict(r, e=DC.e_dec(r["pred_v4"], r["meas"])) for r in full["rows"]]
    keep = [r for r in rows if (r["model"], r["label"]) not in excl]
    pooled = DC.bar([r["e"] for r in keep], bars)
    per = {m: DC.bar([r["e"] for r in keep if r["model"] == m], bars) for m in DC.BLIND_MODELS}
    voided = full["voided"]
    cert = {m: ("CERTIFIED" if per[m]["verdict"] == "HOLDS" and pooled["verdict"] == "HOLDS" else
                ("VOID: " + voided[m]) if m in voided else "NOT CERTIFIED") for m in DC.BLIND_MODELS}
    sigma = fz.get("sigma_page") or DC.SIGMA_PAGE_FALLBACK
    rivals = {R: DC.TC.rival_rule(keep, R, sigma=sigma) for R in SB.RIVALS}
    prm = FN.params_of(fz)
    docs = {m: {c["label"]: c for c in DC.page_doc(m, "main")["cells"]} for m in DC.BLIND_MODELS}
    sig = [D2.sigma_d_row(r["model"], docs[r["model"]][r["label"]], r["meas"], prm) for r in rows
           if (r["model"], r["label"]) in excl]
    kc = [(r["model"], docs[r["model"]][r["label"]], r["meas"], r["pred_v4"]) for r in keep]
    out = dict(full)
    out.update(what=what, addendum=D2.NAME, scope=D2.SCOPE, POWER=power,
               RIVALS_ALL_CELLS=dict(full["rivals"], label="DESCRIPTIVE: every rival on every cell, the DESCRIPTIVE cells included"), excluded_cells=sorted("/".join(k) for k in excl),
               pooled_over=sorted({r["model"] for r in keep}), E1_DEC={"pooled": pooled, "per_model": per},
               certified=cert, rivals=rivals, rows=[dict(r, e=r6(r["e"])) for r in keep],
               SIGMA_D={"verdict": "DESCRIPTIVE", "rows": sig, "per_model": D2.sigma_d_summary(sig),
                        "constants": "the finalised (frozen) constants"},
               KNEE_D=D2.knee_rows(kc, prm),
               REGISTERED_READING={"label": "DESCRIPTIVE: the frozen scorer on every cell (sigma 1 on the second-tile cells)",
                                   "E1_DEC": full["E1_DEC"], "certified": full["certified"]})
    return out


def lines(s: dict) -> list[str]:
    if s.get("verdict") == "NOT SCORED":
        return ["DECODE STAGE, BLIND RENTAL (addendum 2): NOT SCORED", f"power: {s['POWER']['verdict']}", s["scope"]]
    L = SB.lines(s)
    L[0] = "DECODE STAGE, BLIND RENTAL (addendum 2): E1-DEC"
    L.insert(1, f"power: {s['POWER']['verdict']} ({s['POWER'].get('gpu_current_power_limit_w')} W); "
                f"{len(s['excluded_cells'])} descriptive cells out of E1-DEC")
    L = [ln + ("\n    " + s["scope"] if "CERTIFIED" in ln and "->" in ln else "") for ln in L]
    for m, x in s["SIGMA_D"]["per_model"].items():
        L.append(f"SIGMA-D {m}: {x['band']} (median position {x['position_median']}, {x['cells_read']} of {x['cells']} "
                 f"cells read); v4d sigma 1 {x['sigma1']}; L2-served {x['served']}")
    for R, v in s["RIVALS_ALL_CELLS"].items():
        if R != "label":
            L.append(f"rival {R} on every cell (DESCRIPTIVE): {v['verdict']}")
    for k, x in s["KNEE_D"].items():
        L.append(f"knee {k} (DESCRIPTIVE): {x['stats']}")
    return L


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if len(a) != 3:
        print(__doc__)
        return 2
    fin = json.loads(Path(a[1]).read_text())
    if fin.get("addendum") != D2.NAME or "frozen" not in fin:
        print(f"REFUSED: {a[1]} was not finalised under {D2.NAME} (finalise_decode_d2.py)")
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
