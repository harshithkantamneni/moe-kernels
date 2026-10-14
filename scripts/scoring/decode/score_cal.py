#!/usr/bin/env python
"""Scores the decode stage's CAL rental (registration 2026-10-14-decodeC-cal-gh200).

    python scripts/scoring/decode/score_cal.py <results tree> <out dir> --tile-scores <tile rental A score dir>

Reads every CAL unit's page (dcommon.decode_page: G5, the instrument, slip-policy cell, GD4, the
r7-median-run3 rebuild), then: the three CAL constants and the rho selection (fit_and_select) on the
two main pages' valid NATIVE cells; E1-CAL-IN-SAMPLE at the provisional bars; G3-D shuffle, label and
fill; the replicate's sigma_page and the tightened bars; GD2's band (GR1 against GRR) and GD3 (E240
against GR1); the DECL lever and the counter pages (DESCRIPTIVE); the two rivals the blind rental
fits on CAL (line, OLS); and the STOP rule. Writes cal.score.json and cal.score.txt.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcommon as DC  # noqa: E402

RD, V4, V4D = DC.RD, DC.V4, DC.V4D
r6 = DC.r6


def tile_params(tile_scores: Path | None) -> tuple[V4.Params, str]:
    """The tile stage's finalised v4 Params (its rental A selections), else the empty Params (the
    wgmma decode tile is then NOT PRICED)."""
    if tile_scores is None:
        return V4.Params(), "none: the tile stage's selections were not given (T 97..128 NOT PRICED)"
    sys.path.insert(0, str(HERE.parent / "tileA"))
    import finalise as TF
    sel = TF.selections(Path(tile_scores))
    return TF.params(sel), f"tile rental A's selections ({tile_scores})"


def counter_page(tree: Path, model: str, label: str):
    hits = sorted(Path(tree).rglob(f"*-{DC.CARD}-{model}-{label}-r3-counters/lock{DC.LOCK_MHZ}/r3d.json"))
    if len(hits) != 1:
        return None, f"NOT SCORED: {len(hits)} counter pages"
    return json.loads(hits[0].read_text()), "counted"


def ols_fit(rows: list[dict]) -> dict:
    """The OLS rival (the tile stage's rule): ridge on standardised features over CAL NATIVE cells."""
    import numpy as np
    X = np.array([r["features"] for r in rows], dtype=float)
    y = np.array([r["meas_ms"] for r in rows], dtype=float)
    if len(y) < 12:
        return {"verdict": f"NOT SCORED: {len(y)} cells, under 12"}
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    G = Z.T @ Z
    lam = 1e-3 * float(np.trace(G)) / Z.shape[1]
    Zc = Z - Z.mean(0)
    beta = np.linalg.solve(Zc.T @ Zc + lam * np.eye(Z.shape[1]), Zc.T @ (y - y.mean()))
    a = float(y.mean() - Z.mean(0) @ beta)
    return {"verdict": "FITTED", "cells": int(len(y)), "mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist(), "a": a}


def features(p: dict) -> list[float]:
    """OLS's features of one priced cell: weight stream ms, waves, ln(1 + dead CTAs), naive, block-scan."""
    w = p["weight_bytes"] / (V4D.CARD["bw_gbps"] * 1e6)
    waves = sum(g["waves"] for g in p["gemm"].values())
    return [w, waves, math.log1p(sum(p["dead_ctas"])), float(p["path"] == RD.NAIVE), float(p["path"] == RD.BLOCK_SCAN)]


def line_fit(rows: list[dict]) -> dict:
    """The per-token line rival a + b (T k), pooled over the CAL NATIVE cells."""
    xs = [r["numel"] for r in rows]
    ys = [r["meas_ms"] for r in rows]
    if len(xs) < 3:
        return {"verdict": "NOT SCORED"}
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return {"verdict": "FITTED", "a_ms": my - b * mx, "b_ms_per_row": b, "cells": len(xs)}


def score(tree: Path, reg: dict, tparams: V4.Params, tsource: str = "") -> dict:
    tree = Path(tree)
    files = reg["pages"]["files"]
    units, pages, refused = {}, {}, []
    for u in reg["units"]:
        key = f"{u['model']}-{u['label']}-{u['step']}"
        nod = u["drop"] == "nodrop"
        if u["step"] == "decodebytes":
            p, why = counter_page(tree, u["model"], u["label"])
            units[key] = {"state": why, "nodrop": nod}
            pages[key] = p
            continue
        rep, why = DC.decode_page(tree, u["model"], u["label"], files[u["page"]], u["inst"])
        units[key] = {"state": why, "nodrop": nod, "page": u["page"], "inst": u["inst"]}
        pages[key] = rep
        if rep is not None:
            refused += [f"{key}/{r['histogram']}/{r['arm']}" for r in rep["treads_table"] if r.get("refused")]
    docs = {(m, k): DC.page_doc(m, k) for m in DC.CAL_MODELS for k in ("main", "ctl", "e240", "decl")
            if (DC.REPO / DC.page_path(m, k)).exists()}

    def valid(key, model, kind):
        rep = pages.get(key)
        if rep is None:
            return {}, []
        return DC.measured(rep, model, kind)

    main = {}
    drops = {}
    for m in DC.CAL_MODELS:
        v, d = valid(f"{m}-dm-r3decode", m, "main")
        main.update(v)
        drops[f"{m}-dm"] = d
    cells_by = {(m, c["label"]): c for (m, k), doc in docs.items() if k == "main" for c in doc["cells"]}
    fit_rows = [{"model": k[0], "cell": cells_by[(k[0], k[2])], "meas_ms": row["ms_p50"], "key": "/".join(map(str, k))}
                for k, row in main.items() if k[4] == "native"]
    fit = DC.fit_and_select(fit_rows, tparams)
    import dataclasses
    out = {"what": "the decode stage's CAL score", "tile_params": tsource,
           "tile_params_dict": dataclasses.asdict(tparams), "units": units,
           "V_ALIGN_refused": refused, "dropped": drops, "FIT": fit}
    sel = fit["selected"]
    prm = None
    if sel is not None:
        prm = V4D.params_from(fit["forms"][sel], rho_form=sel, tile=tparams)
    # ---- E1 in sample, and the per-cell rows (the rivals' training set)
    e, rows, pred = [], [], {}
    not_priced = []
    if prm is not None:
        for k, row in main.items():
            if k[4] != "native":
                continue
            p = DC.price_cell(k[0], cells_by[(k[0], k[2])], prm)
            if p is None:
                not_priced.append("/".join(map(str, k)))
                continue
            pred[(k[0], k[2])] = p["ms"]
            e.append(DC.e_dec(p["ms"], row["ms_p50"]))
            rows.append({"key": "/".join(map(str, k)), "meas_ms": row["ms_p50"], "pred_ms": r6(p["ms"]),
                         "e": r6(e[-1]), "path": p["path"], "sub_wave": p["sub_wave"], "numel": p["numel"],
                         "features": features(p)})
    out["NOT_PRICED"] = not_priced
    out["tile_selections_missing"] = tparams.c128 is None and any(
        c["T"] >= 97 for (m, k), doc in docs.items() for c in doc["cells"])
    out["E1_CAL_IN_SAMPLE"] = dict(DC.bar(e, DC.BARS_PROVISIONAL), label="IN-SAMPLE")
    out["rows"] = rows
    out["RIVAL_FITS"] = {"line": line_fit(rows), "OLS": ols_fit(rows)}
    # ---- G3-D
    meas_m = {(k[0], k[2]): row["ms_p50"] for k, row in main.items() if k[4] == "native"}
    g3 = {}
    for test in ("shuffle", "label", "fill"):
        pairs = {m: DC.g3d_pairs(docs[(m, "main")], test) for m in DC.CAL_MODELS if (m, "main") in docs}
        g3[test] = DC.g3d(meas_m, pred, pairs, test) if prm is not None else {"verdict": "NOT SCORED: no fit"}
    out["G3D"] = g3
    # ---- the replicate: sigma_page and the blind bars
    repv, _ = valid("olmoe-1b-7b-dr-r3decode", "olmoe-1b-7b", "main")
    pairs = [(main[k]["ms_p50"], repv[k]["ms_p50"]) for k in repv if k in main and k[4] == "native"]
    sig = DC.sigma_rep(pairs)
    out["REP"] = {"pairs": len(pairs), "sigma_page": r6(sig) if sig is not None else None,
                  "bars_blind": DC.tighten(sig)}
    if prm is not None and repv:
        er = [DC.e_dec(pred[(k[0], k[2])], repv[k]["ms_p50"]) for k in repv
              if k[4] == "native" and (k[0], k[2]) in pred]
        out["E1_CAL_REP"] = dict(DC.bar(er, DC.BARS_PROVISIONAL), label="DESCRIPTIVE (the replicate, constants fitted on the main page)")
    # ---- GD2: GR1 against GRR
    diffs, gd2_rows = [], []
    for m in DC.CAL_MODELS:
        ctl, _ = valid(f"{m}-dc-r3decode", m, "ctl")
        for k, row in ctl.items():
            km = (m, "main", k[2], k[3], k[4])
            if km in main:
                d = (row["ms_p50"] - main[km]["ms_p50"]) * 1e3
                diffs.append(d)
                gd2_rows.append({"cell": f"{m}/{k[2]}", "gr1_us": r6(row["ms_p50"] * 1e3),
                                 "grr_us": r6(main[km]["ms_p50"] * 1e3), "d_us": r6(d)})
    out["GD2"] = dict(DC.gd2_band(diffs), rows=gd2_rows)
    # ---- GD3: E240 against GR1 (Mixtral)
    e240, _ = valid("mixtral-8x7b-de-r3decode", "mixtral-8x7b", "e240")
    ctl_mx, _ = valid("mixtral-8x7b-dc-r3decode", "mixtral-8x7b", "ctl")
    g3rows = []
    for k, row in e240.items():
        kc, km = ("mixtral-8x7b", "ctl", k[2], k[3], k[4]), ("mixtral-8x7b", "main", k[2], k[3], k[4])
        if kc in ctl_mx and km in main:
            g3rows.append({"e240_ms": row["ms_p50"], "gr1_ms": ctl_mx[kc]["ms_p50"], "grr_ms": main[km]["ms_p50"],
                           "host_ms": row.get("host_enqueue_ms")})
    out["GD3"] = DC.gd3(g3rows) if g3rows else {"verdict": "NOT SCORED: no E240 page or no matched cell"}
    # ---- DECL (DESCRIPTIVE)
    decl = []
    for m in DC.CAL_MODELS:
        v, _ = valid(f"{m}-dd-r3decode", m, "decl")
        doc = docs.get((m, "decl"))
        for k, row in v.items():
            if k[4] != "decl":
                continue
            kn = k[:4] + ("native",)
            if kn not in v or prm is None or doc is None:
                continue
            c = next(x for x in doc["cells"] if x["label"] == k[2])
            pn = DC.price_cell(m, c, prm)
            pd = DC.price_cell(m, c, prm, arm="decl", decl_copies=doc["decl_copies"])
            if pn is None or pd is None:
                continue
            decl.append({"cell": f"{m}/{k[2]}", "d_us": r6((row["ms_p50"] - v[kn]["ms_p50"]) * 1e3),
                         "pred_d_us": r6((pd["ms"] - pn["ms"]) * 1e3), "dead_extra": [a - b for a, b in zip(pd["dead_ctas"], pn["dead_ctas"])],
                         "path_decl": pd["path"]})
    out["DECL"] = {"verdict": "DESCRIPTIVE", "rows": decl}
    # ---- BYTES-D (DESCRIPTIVE)
    by = []
    for m in DC.CAL_MODELS:
        p = pages.get(f"{m}-dn-decodebytes")
        for c in (p or {}).get("cells") or []:
            for g in RD.GEMMS:
                got = (c["per_gemm"].get(g) or {}).get("dram__bytes_read.sum")
                want = (c.get("v4d_reads") or {}).get(g)
                if got and want:
                    by.append({"cell": f"{m}/{c['key']}/{g}", "ratio": r6(got / want)})
    out["BYTES_D"] = {"verdict": "DESCRIPTIVE", "rows": by,
                      "median_ratio": r6(st.median(x["ratio"] for x in by)) if by else None}
    out["STOP"] = DC.stop_rule(out)
    return out


def lines(s: dict) -> list[str]:
    L = ["DECODE STAGE, CAL RENTAL: score", f"tile params: {s['tile_params']}", "units:"]
    L += [f"  {k}: {v['state']}" for k, v in s["units"].items()]
    f = s["FIT"]
    L.append(f"fit: selected rho {f['selected']} ({f['rule']})")
    for form, x in f["forms"].items():
        L.append(f"  {form}: {x.get('verdict')} T0_GR {x.get('T0_GR_ms')} s_naive {x.get('s_naive_GR_ms')} "
                 f"s_block {x.get('s_block_GR_ms')} in-sample {x.get('in_sample')} sub-wave {x.get('sub_wave')}")
    L.append(f"E1-CAL-IN-SAMPLE: {s['E1_CAL_IN_SAMPLE']['verdict']} {s['E1_CAL_IN_SAMPLE']['stats']}")
    for t, g in s["G3D"].items():
        L.append(f"G3-D {t}: {g['verdict']} (margin {g.get('margin_us')} us)")
    L.append(f"REP: sigma_page {s['REP']['sigma_page']} over {s['REP']['pairs']} pairs; blind bars {s['REP']['bars_blind']}")
    L.append(f"GD2: {s['GD2']['verdict']} median {s['GD2'].get('median_us')} us band {s['GD2'].get('band_us')}")
    L.append(f"GD3: {s['GD3'].get('verdict')} {s['GD3'].get('in_band')}/{s['GD3'].get('cells')} in band")
    L.append(f"STOP: {s['STOP']['verdict']}" + ("".join(f"\n  {w}" for w in s["STOP"]["why"])))
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
    tree, out = Path(a[0]), Path(a[1])
    reg = DC.registration(DC.REPO, "cal")
    if ts is None:
        print("REFUSED: --tile-scores <tile rental A score dir> is required: every CAL main page holds T 97..128 "
              "cells on the wgmma decode tile, which v4d prices only with the tile stage's selections")
        return 2
    tp, src = tile_params(ts)
    s = score(tree, reg, tp, src)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cal.score.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    (out / "cal.score.txt").write_text("\n".join(lines(s)) + "\n")
    print("\n".join(lines(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
