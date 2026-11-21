#!/usr/bin/env python
"""Scores the decode stage's blind rental (registration 2026-10-14-decodeB-blind-gh200) against the
finalised predictions (finalise_decode.py's output).

    python scripts/scoring/decode/score_blind.py <results tree> <finalised.json> <out dir>

Per blind main page (dcommon.decode_page): the valid NATIVE cells; GD2 on the models with a GR1 page
(a page outside the frozen band VOIDS that model's GRR page); G3-D shuffle and label per page (a FAIL
voids the model's cells); E1-DEC pooled and per model at the finalised bars; a model CERTIFIED when
its own bar and the pooled bar hold; every rival under rental 7's exclusion rule. Writes
blind.score.json and blind.score.txt. Every published page is SEEN once scored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcommon as DC  # noqa: E402

r6 = DC.r6
RIVALS = ("rho other", "NAIVE-NODEDUP", "eager", "roofline", "line", "OLS")


def score(tree: Path, reg: dict, fin: dict) -> dict:
    fz = fin["frozen"]
    bars = {k: fz["bars"][k] for k in ("rms_max", "max_abs", "mean_max")}
    preds = {(r["model"], r["label"]): r for r in fin["cells"]}
    files = reg["pages"]["files"]
    units, valid, voided = {}, {}, {}
    for u in reg["units"]:
        rep, why = DC.decode_page(Path(tree), u["model"], u["label"], files[u["page"]], u["inst"])
        units[f"{u['model']}-{u['label']}"] = {"state": why, "nodrop": u["drop"] == "nodrop"}
        if rep is not None:
            v, d = DC.measured(rep, u["model"], u["page"].rsplit("-", 1)[1][:-5])
            valid[(u["model"], u["label"])] = v
    gd2 = {}
    for m in DC.BLIND_MODELS:
        ctl, main = valid.get((m, "dc")), valid.get((m, "dm"))
        if ctl is None or main is None or not fz.get("gd2_band_us"):
            continue
        diffs = [(row["ms_p50"] - main[(m, "main") + k[2:]]["ms_p50"]) * 1e3 for k, row in ctl.items()
                 if (m, "main") + k[2:] in main]
        gd2[m] = DC.gd2_blind(diffs, fz["gd2_band_us"])
        if gd2[m]["verdict"].startswith("VOID"):
            voided[m] = "GD2"
    g3 = {}
    for m in DC.BLIND_MODELS:
        main = valid.get((m, "dm"))
        if main is None:
            continue
        meas = {(m, k[2]): row["ms_p50"] for k, row in main.items() if k[4] == "native"}
        pr = {(m, lab): p["ms"]["v4d"] for (mm, lab), p in preds.items() if mm == m and p["ms"]["v4d"] is not None}
        doc = DC.page_doc(m, "main")
        g3[m] = {t: DC.g3d(meas, pr, {m: DC.g3d_pairs(doc, t)}, t) for t in ("shuffle", "label")}
        if any(x["verdict"] not in ("EQUIVALENT",) and not x["verdict"].startswith("NOT SCORED") for x in g3[m].values()):
            voided.setdefault(m, "G3-D")
    rows, not_priced = [], {}
    for m in DC.BLIND_MODELS:
        main = valid.get((m, "dm")) or {}
        if m in voided:
            continue
        for k, row in main.items():
            p = preds.get((m, k[2]))
            if k[4] != "native":
                continue
            if p is None or p["ms"]["v4d"] is None:
                not_priced.setdefault(m, []).append(k[2])
                continue
            rows.append({"model": m, "label": k[2], "T": k[3], "meas": row["ms_p50"], "pred_v4": p["ms"]["v4d"],
                         "pred": {r: p["ms"].get(r) for r in RIVALS},
                         "e": DC.e_dec(p["ms"]["v4d"], row["ms_p50"])})
    pooled = DC.bar([r["e"] for r in rows], bars)
    per = {m: DC.bar([r["e"] for r in rows if r["model"] == m], bars) for m in DC.BLIND_MODELS}
    cert = {m: ("CERTIFIED" if per[m]["verdict"] == "HOLDS" and pooled["verdict"] == "HOLDS" else
                ("VOID: " + voided[m]) if m in voided else "NOT CERTIFIED") for m in DC.BLIND_MODELS}
    sigma = fz.get("sigma_page") or DC.SIGMA_PAGE_FALLBACK
    rivals = {R: DC.TC.rival_rule(rows, R, sigma=sigma) for R in RIVALS}
    pool = sorted({r["model"] for r in rows})
    return {"what": "the decode stage's blind score (E1-DEC)", "bars": bars, "units": units, "GD2": gd2, "G3D": g3,
            "NOT_PRICED": not_priced, "pooled_over": pool,
            "voided": voided, "E1_DEC": {"pooled": pooled, "per_model": per}, "certified": cert, "rivals": rivals,
            "rows": [dict(r, e=r6(r["e"])) for r in rows], "label": "BLIND (SEEN once published)"}


def lines(s: dict) -> list[str]:
    L = ["DECODE STAGE, BLIND RENTAL: E1-DEC", f"bars {s['bars']}"]
    L += [f"  {k}: {v['state']}" for k, v in s["units"].items()]
    L.append(f"pooled over {', '.join(s['pooled_over'])}: {s['E1_DEC']['pooled']['verdict']} {s['E1_DEC']['pooled']['stats']}")
    for m, labs in s["NOT_PRICED"].items():
        L.append(f"  NOT PRICED on {m}: {len(labs)} valid cells ({', '.join(labs[:6])})")
    for m, b in s["E1_DEC"]["per_model"].items():
        L.append(f"  {m}: {b['verdict']} {b['stats']} -> {s['certified'][m]}")
    for R, v in s["rivals"].items():
        L.append(f"rival {R}: {v['verdict']}")
    return L


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if len(a) != 3:
        print(__doc__)
        return 2
    reg = DC.registration(DC.REPO, "blind")
    fin = json.loads(Path(a[1]).read_text())
    s = score(Path(a[0]), reg, fin)
    out = Path(a[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / "blind.score.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    (out / "blind.score.txt").write_text("\n".join(lines(s)) + "\n")
    print("\n".join(lines(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
