#!/usr/bin/env python
"""Rental B's predictions, finalised MECHANICALLY from rental A's scored selections (owner decision
D3; registration 2026-10-14-tileB-e1tile), and rental A's own E1-CAL predictions from the same code.

    python scripts/scoring/tileA/finalise.py <repo> <rental-A score dir> <out.json> [--plan-out <plan file>]

REFUSES (exit 2, nothing written) unless tA-e1cal's stop rule reads GO, and when the census refused.

Reads rental A's score files (census.score.json, floor.score.json, bytes.score.json, rep.score.json,
e1cal.score.json for the OLS rival) and applies the registered rules, nothing else:
  occupancy   each tile's CTAs per SM from the census (the design occupancy where the census has none)
  c_128, F    tA-floor's calibration (CAL-A); none: REFUSED (no v4, no rental B)
  64x128      the SELECTED accounting's derived c, else v4cal (the 64x128 joint fit, CAL-A)
  F form      tA-floor's f_form_rule
  A key       tA-bytes' NAMED key, else CLAMP; a row key is priced on every rental-B w1 row here
              (rowkeys_tile.stream_misses at the census occupancy)
  sigma_page  tA-rep's, else rental 3's 0.18%
and writes, per rental-B cell and arm, v4's time and bytes and every registered rival's time. The
function `predict_pages` is what score_e1cal.py calls on rental A's own pages.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tcommon as TC  # noqa: E402

REPO = TC.REPO
import r3_timing_model_v4 as V4  # noqa: E402

OLS_MIN_CELLS, OLS_MAX_COND, OLS_LAMBDA = 12, 1e6, 1e-3


class Refused(Exception):
    pass


def _read(d: Path, name: str) -> dict | None:
    p = Path(d) / f"{name}.score.json"
    return json.loads(p.read_text()) if p.exists() else None


def selections(score_dir: Path) -> dict:
    """The registered selections, read off rental A's score files. A census that refused (an MMA path
    mismatch) refuses: it is a STOP, never a fallback to the design occupancy."""
    cen, flo, byt, rep = (_read(score_dir, n) for n in ("census", "floor", "bytes", "rep"))
    if cen is not None and str((cen.get("CENSUS") or {}).get("verdict", "")).startswith("REFUSED"):
        raise Refused(f"the census refused ({cen['CENSUS']['verdict']}): a STOP")
    if cen is not None:
        TC.use_census(Path(score_dir) / "census.score.json")
    if flo is None or not (flo.get("calibration") or {}).get("c"):
        raise Refused("tA-floor has no calibration (c_128, F_wg): v4 has no wgmma constants, rental B does not run")
    cal = flo["calibration"]
    sel = {"c128": float(cal["c"]), "F_wg": {g: float(cal["F"][g]) for g in ("w1", "w2")},
           "variant": (flo.get("selection") or {}).get("selected"),
           "v4cal": {}, "f_form": (flo.get("f_form") or {}).get("selected") or V4.F_FORM_REGISTERED,
           "key": ((byt or {}).get("KEY") or {}).get("key") or TC.FALLBACK_KEY,
           "occupancy": (cen or {}).get("occupancy") or {}, "counts": (cen or {}).get("counts") or {},
           "occupancy_by_model": (cen or {}).get("occupancy_by_model") or {}}
    s, why = TC.sigma_page_of(rep)
    sel["sigma_page"], sel["sigma_page_source"] = s, why
    if sel["variant"] is None:
        m64 = flo.get("fit_64x128_w8")
        if not m64:
            raise Refused("no accounting is SELECTED and the 64x128 w8 pages give no fit: no v4 at 64x128")
        sel["v4cal"][V4.T64W8.key] = {"c": float(m64["c"]), "F": {g: float(m64["F"][g]) for g in ("w1", "w2")}}
    return sel


def params(sel: dict, **over) -> V4.Params:
    d = dict(c128=sel["c128"], F_wg=dict(sel["F_wg"]), variant=sel["variant"] or "W", f_form=sel["f_form"],
             v4cal=dict(sel.get("v4cal") or {}), occupancy=dict(sel.get("occupancy") or {}), counts=dict(sel.get("counts") or {}))
    d.update(over)
    return V4.Params(**d)


def rowkey_w1(model: str, c: dict, doc: dict, key: str, occ: int) -> float | None:
    """A row key's w1 read bytes at one cell (streamed, bounded memory): NATIVE's walk, SHARED's at G 1."""
    import rowkeys_tile as RKT
    import skew_synth_tile as ST

    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    bm = int(doc["block_m"])
    t = V4.T64W8 if bm == 64 else V4.CAL_TILE
    n = int(c["n"])
    tokens = ST.tokens_at(model, n, bm)
    if c["counts"] is None:
        counts, seed = tuple([tokens * cfg.top_k // cfg.num_experts] * cfg.num_experts), None
    else:
        counts, seed = tuple(int(v) for v in c["counts"]), int(doc["shuffle_seed"])
    D = cfg.num_experts
    R = -(-(sum(counts) + D * (bm - 1)) // bm)
    C = {"ROWKEY60": 60.0, "ROWKEY30": 30.0}[key]
    got, _N = RKT.stream_misses("native", counts, tokens, cfg.top_k, 2 * cfg.intermediate_size // t.bn,
                                cfg.hidden_size // t.bk, R, seed, bm=t.bm, bn=t.bn, bk=t.bk, occ=occ,
                                G=int(doc["group_m"]), caps_mib=(C,))
    return got[C]


def predict_pages(pages, rental: str, sel: dict, *, rivals: bool = True) -> list[dict]:
    """Per cell and arm of `pages` ((label, model, page name, ...) tuples): v4 at the selections, its
    bytes and parts, and the registered rivals' times."""
    import regtile as RG
    out = []
    for item in pages:
        label, model, name = item[:3]
        # the census occupancy of THIS model where its compiled configs differ from the others'
        prm = params(sel, occupancy={**sel.get("occupancy", {}), **(sel.get("occupancy_by_model") or {}).get(model, {})})
        doc = RG.hist(rental, name)
        t = RG.PAGE_TILE[int(doc["block_m"])]
        for c in doc["cells"]:
            rk = None
            if sel["key"] in ("ROWKEY60", "ROWKEY30"):
                rk = rowkey_w1(model, c, doc, sel["key"], prm.occ(t, "w1"))
            for arm in c["arms"]:
                p = RG.price_cell(model, arm, c, doc, prm, key=sel["key"], rowkey_w1=rk)
                row = {"unit": label, "model": model, "page": name, "tile": t.key, "G": int(doc["group_m"]),
                       "cls": TC.tile_class(t.bm, int(doc["group_m"])), "n": int(c["n"]), "label": c["label"], "arm": arm,
                       "ms": {"v4": p["ms"]}, "A_proxy_us": p["A_us"], "native_path": p["native_path"],
                       "reads": p["reads"], "host_ok": TC.C6.host_ok(p["ms"]),
                       "live_ctas": sum(p["gemm"][g]["live_ctas"] for g in ("w1", "w2")),
                       "waves": sum(p["gemm"][g]["live_ctas"] / (V4.SMS * p["occ"][g]) for g in ("w1", "w2"))}
                if rivals and arm == "native":
                    o = prm.occupancy
                    alt = {"v3own": params(sel, law="v3own", occupancy=o),
                           "F-other": params(sel, f_form="F-cyc" if sel["f_form"] == "F-step" else "F-step", occupancy=o),
                           "eta_mma rival": params(sel, eta_rival=True, occupancy=o)}
                    for v in V4.VARIANTS:
                        if v != (sel["variant"] or "W") and V4.T64W8.key not in prm.v4cal:
                            alt[f"accounting {v}"] = params(sel, variant=v, occupancy=o)
                    for k, pp in alt.items():
                        try:
                            row["ms"][k] = RG.price_cell(model, arm, c, doc, pp, key=sel["key"], rowkey_w1=rk)["ms"]
                        except V4.TM.Refused:
                            row["ms"][k] = None
                    row["ms"]["roofline"] = RG.roofline_ms(model, int(c["n"]), c["counts"], doc, p["reads"], V4.eta_mma_of(t))
                out.append(row)
    return out


def ols_fit(rows: list[dict], meas: dict) -> dict:
    """The OLS rival (tileB-e1tile): ridge on standardised [waves, bytes / bw, ln(1 + CTAs)] over the
    valid NATIVE E1-CAL cells `meas` {(model, page, n, label): ms}; NOT SCORED under 12 cells or a
    condition number over 1e6."""
    import numpy as np
    import regtile as RG
    bw = float(RG.pricer().f2.x[2])
    X, y = [], []
    for r in rows:
        k = (r["model"], r["page"], r["n"], r["label"])
        if r["arm"] != "native" or k not in meas:
            continue
        X.append([r["waves"], sum(r["reads"].values()) / (bw * 1e6), math.log1p(r["live_ctas"])])
        y.append(meas[k])
    if len(y) < OLS_MIN_CELLS:
        return {"verdict": f"NOT SCORED: {len(y)} cells, under {OLS_MIN_CELLS}"}
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    G = Z.T @ Z
    cond = float(np.linalg.cond(G))
    if cond > OLS_MAX_COND:
        return {"verdict": f"NOT SCORED: condition number {cond:.3g} over {OLS_MAX_COND:g}"}
    lam = OLS_LAMBDA * float(np.trace(G)) / 3
    Zc = Z - Z.mean(0)
    beta = np.linalg.solve(Zc.T @ Zc + lam * np.eye(3), Zc.T @ (y - y.mean()))
    a = float(y.mean() - Z.mean(0) @ beta)
    return {"verdict": "FITTED", "cells": len(y), "mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist(), "a": a,
            "bw_gbps": bw, "condition": cond, "lambda": lam}


def ols_predict(fit: dict, row: dict) -> float | None:
    if fit.get("verdict") != "FITTED":
        return None
    x = [row["waves"], sum(row["reads"].values()) / (fit["bw_gbps"] * 1e6), math.log1p(row["live_ctas"])]
    z = [(x[i] - fit["mu"][i]) / fit["sd"][i] for i in range(3)]
    return fit["a"] + sum(z[i] * fit["beta"][i] for i in range(3))


def go_or_refuse(score_dir: Path) -> dict:
    """Rental B is finalised only on a GO: tA-e1cal's stop rule (tileB-stop) must read GO."""
    e1 = _read(score_dir, "e1cal")
    v = str(((e1 or {}).get("STOP") or {}).get("verdict", "no tA-e1cal score"))
    if not v.startswith("GO"):
        why = "; ".join(((e1 or {}).get("STOP") or {}).get("why") or []) or v
        raise Refused(f"the stop rule reads {v}: rental B does not run ({why})")
    return e1


def finalise(repo: Path, score_dir: Path) -> dict:
    import regtile as RG
    go_or_refuse(score_dir)
    sel = selections(score_dir)
    rows = predict_pages(RG.B_TIMED_PAGES, "B", sel)
    e1 = _read(score_dir, "e1cal") or {}
    fit = e1.get("OLS_fit") or {"verdict": "NOT SCORED: no tA-e1cal OLS fit"}
    for r in rows:
        if r["arm"] == "native":
            r["ms"]["OLS"] = ols_predict(fit, r)
    return {"what": "rental B's predictions, finalised mechanically from rental A's selections (tileB-e1tile)",
            "selections": {k: v for k, v in sel.items()}, "OLS_fit": fit,
            "params": dataclasses.asdict(params(sel)), "cells": rows,
            "host_cut": sum(1 for r in rows if not r["host_ok"])}


PLAN_B = "scripts/plans/rentalB-tile-2026-10.plan"


def _cap(est: int) -> int:
    return max(15, 3 * est)


def plan_b(sel: dict | None = None) -> str:
    """Rental B's plan file, generated from the frozen unit list of tileB-e1tile (regtile.B_TIMED_PAGES
    and B_BYTE_PAGES, in that order: the driver drops the last unit first, so byt goes, then dsv).
    Every value is the registration's; `sel` only names the finalised constants in the header."""
    import regtile as RG
    import skew_synth_tile as ST
    hdr = ["# RENTAL B OF THE TILE STAGE: the blind test at the tile vLLM picks (registration 2026-10-14-tileB-e1tile).",
           "# GENERATED by scripts/scoring/tileA/finalise.py from the frozen unit list; do not edit by hand.",
           "# Run by the tile driver: bash scripts/vm_run.sh start ... --plan " + PLAN_B + " --driver gh200_tile_session.sh",
           "# Drop order, last first: byt (the byte pages), dsv; every Phi, Qwen1.5 and JetMoE timed page is nodrop."]
    if sel:
        hdr.append(f"# Finalised constants: c_128 {sel['c128']:.3f}, accounting {sel['variant'] or 'v4cal'}, F form "
                   f"{sel['f_form']}, A key {sel['key']}, sigma_page {sel['sigma_page']}.")
    lines = hdr + ["- prelude est=3 cap=15", "mixtral-8x7b calibrate label=tb nodrop=1 est=2 cap=30",
                   "- tilecensus label=tb tiles=128x128x64w8s3,64x128x64w8s3 models=" +
                   ",".join(sorted({m for _l, m, *_r in RG.B_TIMED_PAGES})) + " nodrop=1 est=4 cap=15"]

    def tile_keys(doc):
        t = RG.PAGE_TILE[int(doc["block_m"])]
        return f"block-m={t.bm} block-n={t.bn} block-k={t.bk} num-warps={t.warps} num-stages={t.stages}"
    for label, model, name, drop, est in RG.B_TIMED_PAGES:
        doc = RG.hist("B", name)
        path = ST.path_of("B", name).relative_to(REPO)
        d = "nodrop=1" if drop == "nodrop" else f"drop-group={drop}"
        lines.append(f"{model} timed label={label} {tile_keys(doc)} histogram={path} arms=native,shared declared-copies=9 "
                     f"timed-groups={doc['group_m']} slip-policy=cell g1-rule=r7-median-run3 {d} est={est} cap={_cap(est)}")
    for label, model, name, drop, est in RG.B_BYTE_PAGES:
        doc = RG.hist("B", name)
        path = ST.path_of("B", name).relative_to(REPO)
        lines.append(f"{model} bytes label={label} {tile_keys(doc)} histogram={path} arms=native,shared "
                     f"byte-groups={doc['group_m']} drop-group={drop} est={est} cap={_cap(est)}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    plan_out = None
    if "--plan-out" in a:
        i = a.index("--plan-out")
        plan_out = Path(a[i + 1])
        del a[i:i + 2]
    if len(a) != 3:
        print(__doc__)
        return 2
    repo, scores, out = Path(a[0]), Path(a[1]), Path(a[2])
    try:
        doc = finalise(repo, scores)
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return 2
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {out}: {len(doc['cells'])} rows, key {doc['selections']['key']}, host cut {doc['host_cut']}")
    if plan_out is not None:
        plan_out.write_text(plan_b(doc["selections"]))
        print(f"wrote {plan_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
