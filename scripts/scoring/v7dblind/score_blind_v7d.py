#!/usr/bin/env python
"""Scores model v7's decode domain blind (registration docs/registered/2026-11-18-v7-domain-blind-gh200) on a results
tree.

    python scripts/scoring/v7dblind/score_blind_v7d.py <results tree> <out dir>

It reads the registration at its committed path only (no option names another file) and records that file's sha256
and the checkout's commit in the score. It refuses (exit 2) a DRAFT (D unknown: nothing is IN-DOMAIN yet) and a
registration whose freeze does not hold (v7 as the gate froze it, this blind's own files, the map's registration,
plan and readout), or whose labels differ from the labels recomputed from the frozen readout.

The frozen blind scorer runs unchanged (score_blind_d3.score over score_blind_d2 and score_blind, as score_blind_v7.py
runs it) with v7 at the registration's frozen constants, the four models and this blind's pages: the power check first
(700 W), the page checks, row_drop, GD2's and G3-D shuffle's voids, addendum 2's second-tile cells out. Its E1-DEC over
every remaining cell is kept as DESCRIPTIVE. The registered reading is E1-DEC over the IN-DOMAIN cells only (the
registration's labels, frozen at --final): per model and pooled at rms 3%, worst 8%, |mean| 1.5% (owner, D4-1); a
model CERTIFIED when its own bar and the pooled bar hold and at least 10 of its cells are scored in-domain (the
minimum); OUT cells reported DESCRIPTIVE, and each model's in-domain cells split into within the map's measured
range and extrapolated past it, DESCRIPTIVE. The inherited all-cell blocks (rivals, RIVALS_ALL_CELLS, KNEE_D,
SIGMA_D, G3D_LABEL, the all-cell pooled_over) are renamed '... (DESCRIPTIVE, all cells)'; no verdict reads them. Writes blind.score.json and blind.score.txt. Exit: 0 scored; 1 NOT SCORED
(the power check); 2 REFUSED."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7dcommon as V  # noqa: E402

import score_blind_d3 as SB3  # noqa: E402

DC = V.DC


def registration_block(reg: dict) -> dict:
    return {"units": reg["cells"]["units"], "pages": reg["cells"]["pages"]}


def finalised(reg: dict) -> dict:
    p = reg["predictions"]
    return {"frozen": p["frozen"], "cells": p["cells"]}


def in_domain(rows: list[dict], reg: dict) -> dict:
    """The registered reading on the frozen scorer's rows: E1-DEC on the IN-DOMAIN cells, the minimum, the
    certification; the OUT cells' E1-DEC beside, DESCRIPTIVE."""
    bars = {k: reg["bars"][k] for k in ("rms_max", "max_abs", "mean_max")}
    lab = reg["domain"]["labels"]["cells"]
    e = {(r["model"], r["label"]): DC.e_dec(r["pred_v4"], r["meas"]) for r in rows}
    inn = [k for k in e if lab[f"{k[0]}/{k[1]}"]["in"]]
    out = [k for k in e if not lab[f"{k[0]}/{k[1]}"]["in"]]
    pooled = DC.bar([e[k] for k in inn], bars)
    per = {m: DC.bar([e[k] for k in inn if k[0] == m], bars) for m in V.MODELS}
    n = {m: sum(1 for k in inn if k[0] == m) for m in V.MODELS}
    geo = {(r["model"], r["label"]): r["geometry"] for r in reg["predictions"]["cells"]}
    ranges = reg["domain"]["map ranges (COMPUTED, the map registration's in-scope cells)"]
    wm = {k: V.within_map(geo[k], ranges) for k in inn}
    split = {"label": ("DESCRIPTIVE: each model's in-domain cells split into within the map's measured range on "
                       "their path (T_v7 and W_e) and extrapolated past it; no verdict reads it"),
             "per_model": {m: {"within the map's range": DC.bar([e[k] for k in inn if k[0] == m and wm[k]], bars),
                               "extrapolated": DC.bar([e[k] for k in inn if k[0] == m and not wm[k]], bars)}
                           for m in V.MODELS}}
    return {"label": "REGISTERED: E1-DEC on the IN-DOMAIN cells only", "bars": bars, "pooled": pooled,
            "pooled_over": sorted({k[0] for k in inn}), "per_model": per, "in-domain cells scored": n,
            "OUT cells (DESCRIPTIVE)": {"pooled": DC.bar([e[k] for k in out], bars),
                                        "per_model": {m: DC.bar([e[k] for k in out if k[0] == m], bars)
                                                      for m in V.MODELS}},
            "within the map's range or extrapolated (DESCRIPTIVE)": split}


def certify(r: dict, voided: dict) -> dict:
    out = {}
    for m in V.MODELS:
        if m in voided:
            out[m] = "VOID: " + voided[m]
        elif r["in-domain cells scored"][m] < V.MIN_IN_DOMAIN:
            out[m] = (f"NOT CERTIFIED: BELOW THE MINIMUM ({r['in-domain cells scored'][m]} in-domain cells scored, "
                      f"under {V.MIN_IN_DOMAIN})")
        elif r["per_model"][m]["verdict"] == "HOLDS" and r["pooled"]["verdict"] == "HOLDS":
            out[m] = "CERTIFIED"
        else:
            out[m] = "NOT CERTIFIED"
    return out


def score(tree: Path, reg: dict) -> dict:
    fin = finalised(reg)
    with V.v7_domain_blind(fin["frozen"]):
        s = SB3.score(Path(tree), registration_block(reg), fin)
    if s.get("verdict") == "NOT SCORED":
        return dict(s, registration=V.NAME)
    rows = [dict(r, e=DC.e_dec(r["pred_v4"], r["meas"])) for r in s["rows"]]
    r = in_domain(rows, reg)
    cert = certify(r, s.get("voided") or {})
    lab = reg["domain"]["labels"]["cells"]
    out = dict(s)
    for k in ("rivals", "RIVALS_ALL_CELLS", "KNEE_D", "SIGMA_D", "G3D_LABEL", "pooled_over"):
        if k in out:
            out[f"{k} (DESCRIPTIVE, all cells)"] = out.pop(k)
    out.update(what="model v7's decode domain blind score (E1-DEC inside D)", registration=V.NAME,
               addenda=list(s.get("addenda") or []) + [V.NAME], bars_source=reg["bars"]["source"],
               draft=reg.get("draft"), D=reg["domain"]["D"]["statement"], minimum=V.MIN_RULE,
               ALL_CELLS_DESCRIPTIVE={"label": "DESCRIPTIVE: the frozen scorer's E1-DEC over every scored cell, in or "
                                               "out of D", "E1_DEC": s["E1_DEC"], "certified": s["certified"]},
               E1_DEC_IN_DOMAIN=r, E1_DEC={"pooled": r["pooled"], "per_model": r["per_model"]}, certified=cert,
               pooled_over=r["pooled_over"],
               rows=[dict(x, domain="IN" if lab[f"{x['model']}/{x['label']}"]["in"] else "OUT") for x in s["rows"]])
    return out


def lines(s: dict) -> list[str]:
    if s.get("verdict") == "NOT SCORED":
        return ["MODEL v7 DECODE DOMAIN BLIND: NOT SCORED", f"power: {s['POWER']['verdict']}"]
    r = s["E1_DEC_IN_DOMAIN"]
    L = ["MODEL v7 DECODE DOMAIN BLIND (four models, IN-DOMAIN cells only): E1-DEC",
         f"power: {s['POWER']['verdict']} ({s['POWER'].get('gpu_current_power_limit_w')} W); bars {r['bars']}",
         f"D: {s['D']}"]
    L += [f"  {k}: {v['state']}" for k, v in s["units"].items()]
    L.append(f"pooled IN-DOMAIN over {', '.join(r['pooled_over']) or 'no model'}: {r['pooled']['verdict']} "
             f"{r['pooled']['stats']}")
    for m in V.MODELS:
        L.append(f"  {m}: {r['per_model'][m]['verdict']} {r['per_model'][m]['stats']} "
                 f"({r['in-domain cells scored'][m]} in-domain cells) -> {s['certified'][m]}")
    for m, x in r["within the map's range or extrapolated (DESCRIPTIVE)"]["per_model"].items():
        w = x["within the map's range"]["stats"]
        L.append(f"  {m} (DESCRIPTIVE): within the map's range {w}; extrapolated {x['extrapolated']['stats']}")
    o = r["OUT cells (DESCRIPTIVE)"]
    L.append(f"OUT cells (DESCRIPTIVE): {o['pooled']['stats']}")
    a = s["ALL_CELLS_DESCRIPTIVE"]["E1_DEC"]
    L.append(f"all cells, in and out of D (DESCRIPTIVE): {a['pooled']['verdict']} {a['pooled']['stats']}")
    for m, g in (s.get("GD2") or {}).items():
        L.append(f"GD2 {m}: {g['verdict']}")
    for m, v in (s.get("voided") or {}).items():
        L.append(f"VOID {m}: {v}")
    L.append(f"scope: {s.get('scope')}")
    if s.get("draft"):
        L.insert(0, "DRAFT REGISTRATION: a synthetic or test score, never a reading")
    return L


def refusals(reg: dict) -> list[str]:
    """Why this registration may not be scored (empty: it may)."""
    if reg.get("name") != V.NAME:
        return [f"the registration is not {V.NAME}"]
    if reg.get("draft"):
        return ["the registration is a DRAFT (D unknown: the map's readout is not committed and verified)"]
    bad = V.verify_freeze(reg)
    if bad:
        return ["the freeze does not hold: " + "; ".join(bad[:10])]
    return label_refusals(reg)


def label_refusals(reg: dict) -> list[str]:
    """The registered labels recomputed from the frozen readout (the file the freeze holds) must equal the
    registration's."""
    try:
        dom = json.loads((V.REPO / reg["domain"]["readout"]["path"]).read_text())
        lab = V.labels(reg["predictions"]["cells"], V.domain_of(dom))
    except (OSError, ValueError, KeyError, TypeError) as ex:
        return [f"the labels cannot be recomputed from the frozen readout ({type(ex).__name__}: {str(ex)[:100]})"]
    if lab != reg["domain"]["labels"]:
        diff = [k for k in lab["cells"] if lab["cells"][k] != reg["domain"]["labels"]["cells"].get(k)]
        return [f"the registered labels differ from the frozen readout's ({len(diff)} cells, e.g. {diff[:3]})"]
    return []


def provenance(path: Path) -> dict:
    import subprocess
    head = subprocess.run(["git", "-C", str(V.REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(V.REPO), "status", "--porcelain"], capture_output=True, text=True)
    return {"registration_path": V.REG_JSON, "registration_sha256": V.sha256(path),
            "git_head": head.stdout.strip() if head.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tree", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    path = V.REPO / V.REG_JSON
    reg = json.loads(path.read_text())
    why = refusals(reg)
    if why:
        print("REFUSED: " + "; ".join(why))
        return 2
    s = dict(score(a.tree, reg), provenance=provenance(path))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "blind.score.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    (a.out / "blind.score.txt").write_text("\n".join(lines(s) + [f"registration sha256 {s['provenance']['registration_sha256']}, "
                                                                 f"commit {s['provenance']['git_head']}"]) + "\n")
    print("\n".join(lines(s)))
    return 0 if s.get("verdict") != "NOT SCORED" else 1


if __name__ == "__main__":
    sys.exit(main())
