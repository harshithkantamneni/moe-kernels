"""MODEL v5 (design, 2026-10-21): a prototype pricer over the pinned v4 / v4L / M4 modules (nothing pinned is touched;
every change is an in-memory swap restored on exit, as fixes.py and r3_timing_model_v4L do), the reduction checks,
the SEEN re-price of A2's E1-CAL cells, and v5's numbers on rental B2's frozen cells (levers only, never a prediction
of record: the registered predictions are written by the finalise step at the registered constants).

v5 = v4L(L) with, per wgmma tile t (each term is the identity at R3's tile and when its switch is off):
  F3  bytes: M4's walk with the tile family's own re-read constants (f3fit.json, SEEN-FITTED), the v3 content key
      (no CLAMP floor: DIAGNOSIS-A2 F5). Off: M4's registered constants and v4's key (CLAMP).
  F2  knee: r3_timing_model.P_KNEE = p_t (kneefit.json, SEEN-FITTED in the per-CTA form v4 uses). Off: 14.
  F4  partial-wave rho: CORES_RHO_GBPS x r_t, r_t = v4d's registered "scaled" form (weight bytes in flight per CTA
      over R3's): x 2.0 at 64x128 w8 s3 and at 128x128 w8 s3. Off: 1. (DIAGNOSIS-A2 F4 used operand bytes, x 2.67 at
      128x128; the two agree at 64x128, the only tile where a tail is measured.)
  F1  sag: each GEMM's c (time per counter cycle) inflated by (1 + D_t(t_f) / t_f), t_f the GEMM's floor-only time,
      D_t(t) = A_t(L) (1 - exp(-t / tau_t(L))) - A_R3(L) (1 - exp(-t / tau_R3(L))), MEASURED per lock (and board) on
      counter pages; 0 when A_t(L) <= 8 us (S1-M2's GONE). Off: D = 0.
Writes v5price.json.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from contextlib import contextmanager
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.home() / "code/moe-kernels-work/2026-10-21/diag-A2/work"))
from common import A2, FN, REPO, RG, RULES, SCORE, TC, TM, TREE, V4, WV4, price_parts  # noqa: E402
import wave_split_bytes as W  # noqa: E402
from moe.spec import MODEL_CONFIGS  # noqa: E402

H = Path(__file__).resolve().parent
BYTE_UNIT = {"olmoe-1b-7b-128g1": "b128g1", "mixtral-8x7b-64g1": "b64g1", "mixtral-8x7b-128g16": "b128g16",
             "qwen2-57b-a14b-128g1": "b128g1"}
#: F4: v4d's registered "scaled" rho (r3_timing_model_v4d.rho_of(t, "scaled") / RHO_GBPS: weight bytes in flight per CTA,
#: stages x BK x BN x 2, over R3's 3 x 64 x 64 x 2): x 2.0 at both wgmma tiles. One law for prefill and decode.
RHO_X = {128: 3 * 64 * 128 * 2 / (3 * 64 * 64 * 2), 64: 3 * 64 * 128 * 2 / (3 * 64 * 64 * 2)}
#: F1 constants (us), MEASURED on counter pages: sag.json (A2 board, 1410: the board E1-CAL ran on; R3: rental 3)
SAG = {1410: {128: (50.13, 750.0), 64: (34.52, 425.0), "R3": (6.11, 50.0)},
       1005: {128: (0.70, 75.0), 64: (0.0, 50.0), "R3": (0.33, 50.0)}}
GONE_US = 8.0


#: v5's F3 form per tile (f3fit.out, SEEN selection by pooled rms): 128x128 one constant set PER GEMM (native w1
#: 12.6 against 16.5 percent rms one-set, w2 2.4 against 3.7); 64x128 ONE set for both GEMMs (w1 3.9 against 6.0)
F3_FORM = {128: "refit_per_gemm", 64: "refit"}


def f3_params(form=None):
    """{tile: {gemm: W.Params}}: v5's registered forms (F3_FORM), or one form at both tiles."""
    d = json.loads((H / "f3fit.json").read_text())
    out = {}
    for k, v in d.items():
        fm = form or F3_FORM[int(k)]
        if fm == "refit":
            p = W.Params.from_json(v["refit"])
            out[int(k)] = {"w1": p, "w2": p}
        else:
            out[int(k)] = {g: W.Params.from_json(v["refit_per_gemm"][g]) for g in ("w1", "w2")}
    return out


def knee_p():
    """p_128 from kneefit.json (NATIVE G 1 + G 16, F4 off: no 128x128 E1 cell has a partial-wave tail), p_64 from
    kneefit64.json with F4 on (p is not identified at 64x128 without it)."""
    d = json.loads((H / "kneefit.json").read_text())["fits"]
    d64 = json.loads((H / "kneefit64.json").read_text())
    return {128: d["128 fit (NATIVE, G1+G16, 1410)"]["best_p"], 64: d64["rho x2.0 G1 pages (b64g1, p64g1)"]["best_p"]}


@contextmanager
def swapped(p_knee=None, rho_x=None):
    keep = (TM.P_KNEE, TM.CORES_RHO_GBPS)
    try:
        if p_knee is not None:
            TM.P_KNEE = p_knee
        if rho_x is not None:
            TM.CORES_RHO_GBPS = keep[1] * rho_x
        yield
    finally:
        TM.P_KNEE, TM.CORES_RHO_GBPS = keep


def f3_reads(pr, model, n, counts, seed, t, G, prm, m4prm, key):
    occ = {g: prm.occ(t, g) for g in TM.GEMMS}
    phis = {g: V4.phi(t, g, prm) for g in TM.GEMMS}
    ct = None if counts is None else tuple(int(v) for v in counts)
    with V4.at_tile(model, t, phis):
        if ct is None:
            E = MODEL_CONFIGS[model].num_experts
            ct, seed = tuple([n * t.bm] * E), None
        m = WV4.model_at(pr.eng, model, RG.COPIES, t, occ)
        q = {}
        for g in W.GEMMS:   # each (cell, GEMM) is its own fixed point: the GEMM's own constant set
            r = m.evaluate(m4prm[g], [("shared", G, n, g, ct, seed)])[0]
            if r["ill"]:
                return None, None
            q[g] = float(r["q"])
        reads = WV4.keyed_reads(key, q, n, WV4.q_ideal(model, n, ct, t.bm), None)
    return reads, q


def price(model, arm, c, doc, prm, *, lock, F1=False, F2=None, F3=None, F4=False, key="CLAMP", sag_scale=1.0,
          reads=None, sag_form="Sc"):
    """v5's ms of one cell. F2: p per tile dict or None; F3: M4 params per tile dict or None (then `key` CLAMP is v4's)."""
    pr = RG.pricer()
    bm = int(doc["block_m"])
    t = RG.PAGE_TILE[bm]
    n, counts, G = int(c["n"]), c["counts"], int(doc["group_m"])
    seed = RG.seed_of(doc, c)
    out = {}
    with swapped(F2[bm] if F2 else None, RHO_X[bm] if F4 else None):
        if reads is None and F3:
            reads, out["q"] = f3_reads(pr, model, n, counts, seed, t, G, prm, F3[bm], key)
            if reads is None:
                return None
        kw = {"reads": reads} if reads is not None else {}
        A, tau = SAG[lock][bm]
        A0, tau0 = SAG[lock]["R3"]
        if not F1 or A <= GONE_US:
            r = RG.price_cell(model, arm, c, doc, prm, key=key, **kw)
            out["ms"] = r["ms"]
            out["reads"] = r.get("reads")
            return out
        pp = price_parts(model, arm, c, doc, prm, key=key, **kw)
        A, A0 = A * 1e-3 * sag_scale, A0 * 1e-3
        tau, tau0 = tau * 1e-3, tau0 * 1e-3
        if sag_form == "S":
            ms = pp["ms_v4"]
            for g in ("w1", "w2"):
                tg = pp[g]["gemm"] + pp[g]["dead"]
                ms += A * (1 - math.exp(-tg / tau)) - A0 * (1 - math.exp(-tg / tau0))
            out["ms"], out["reads"] = ms, pp["reads"]
            return out
        fac = {}
        for g in ("w1", "w2"):
            tf = pp[g]["floor_only"]
            fac[g] = 1 + (A * (1 - math.exp(-tf / tau)) - A0 * (1 - math.exp(-tf / tau0))) / tf
        base_cs = V4.c_scale
        V4.c_scale = lambda t_, g, prm_, _b=base_cs, _f=fac: _b(t_, g, prm_) * _f[g]
        try:
            r = RG.price_cell(model, arm, c, doc, prm, key=key, **kw)
        finally:
            V4.c_scale = base_cs
        out["ms"], out["reads"] = r["ms"], r.get("reads")
        out["fac"] = fac
        return out


def params_for(sel, model):
    return FN.params(sel, occupancy={**sel.get("occupancy", {}), **(sel.get("occupancy_by_model") or {}).get(model, {})})


def e1cal_rows():
    """A2's E1-CAL NATIVE cells with their measured p50 (fixes.py's loop; the registered scorer's selections)."""
    sel = FN.selections(SCORE)
    reg = TC.registration(REPO, "e1cal")
    files = reg["pages"]["files"]
    tview = TC.C3.TimedView("ALL")
    rows = []
    for label, model, name in RG.A_TIMED_PAGES:
        f = files[f"{name}.json"]
        prm = params_for(sel, model)
        doc = RG.hist("A", name)
        rep, _why = TC.timed_page(TREE, model, label, f["group_m"], tview, f)
        tt = {(int(r["tiles"]), str(r["histogram"]), str(r["arm"])): r for r in rep["treads_table"]}
        for c in doc["cells"]:
            if "native" not in c["arms"]:
                continue
            k = (int(c["n"]), c["label"], "native")
            tr = tt[k]
            base = RG.price_cell(model, "native", c, doc, prm, key=sel["key"])["ms"]
            rows.append({"model": model, "page": name, "cls": TC.tile_class(int(doc["block_m"]), int(doc["group_m"])),
                         "n": k[0], "label": k[1], "meas": float(tr["ms_p50"]), "drop": TC.row_drop(tr, base),
                         "c": c, "doc": doc, "v4": base})
    return sel, rows


def summarise(rows, key):
    ok = [r for r in rows if not r["drop"] and r["ms"].get(key) is not None]
    e = [(r, r["ms"][key] / r["meas"] - 1) for r in ok]
    cm = {cl: 100 * st.fmean(x for r, x in e if r["cls"] == cl) for cl in ("64g1", "128g1", "128g16")}
    bym = {m: 100 * st.fmean(x for r, x in e if r["cls"] == "128g1" and r["model"] == m)
           for m in ("olmoe-1b-7b", "mixtral-8x7b", "qwen2-57b-a14b")}
    return {"cells": len(e), "rms_pct": 100 * math.sqrt(st.fmean(x * x for _, x in e)),
            "worst_pct": 100 * max((x for _, x in e), key=abs), "mean_pct": 100 * st.fmean(x for _, x in e),
            "class_mean_pct": cm, "g1_128_by_model_pct": bym, "g1_128_spread_pct": max(bym.values()) - min(bym.values())}


def main():
    RG.pricer()   # the CAL engine fit reads TM.P_KNEE on first build: build it before any swap
    assert TM.P_KNEE == 14.0
    F3P, P = f3_params(), knee_p()
    P_NOF4 = {128: P[128], 64: 14.0}   # without F4 no finite p fits 64x128 G 1 (kneefit64: best p "inf"): v4's 14
    print("F3 params", {k: {g: p.as_json() for g, p in v.items()} for k, v in F3P.items()})
    print("F2 p", P)
    variants = {
        "v4 (registered)": dict(),
        "v5 all-off (must equal v4)": dict(F1=False, F2={128: 14.0, 64: 14.0}, F3={128: None, 64: None}, F4=False),
        "F3 only": dict(F3=F3P, key="v3"),
        "F2 only": dict(F2=P),
        "F1 only (Sc)": dict(F1=True),
        "F1+F2": dict(F1=True, F2=P_NOF4),
        "F4 only": dict(F4=True),
        "F1+F2+F3 (no F4; p_64 14)": dict(F1=True, F2=P_NOF4, F3=F3P, key="v3"),
        "F1+F2+F3+F4 (v5)": dict(F1=True, F2=P, F3=F3P, key="v3", F4=True),
        "F1(S additive)+F2+F3+F4": dict(F1=True, F2=P, F3=F3P, key="v3", F4=True, sag_form="S"),
        "F1+F2+F3+F4, sag x0.8": dict(F1=True, F2=P, F3=F3P, key="v3", F4=True, sag_scale=0.8),
        "F1+F2+F3+F4, sag x1.2": dict(F1=True, F2=P, F3=F3P, key="v3", F4=True, sag_scale=1.2),
        "v5 with M4's one-set refit (F3 form rival)": dict(F1=True, F2=P, F3=f3_params("refit"), key="v3", F4=True),
        "v5, CLAMP key": dict(F1=True, F2=P, F3=F3P, key="CLAMP", F4=True),
    }
    out = {"F3": {k: {g: p.as_json() for g, p in v.items()} for k, v in F3P.items()}, "F2": P, "SAG": {str(k): v for k, v in SAG.items()}}
    with A2.scoring(RULES):
        sel, rows = e1cal_rows()
        for r in rows:
            prm = params_for(sel, r["model"])
            r["ms"] = {}
            for vn, kw in variants.items():
                if vn == "v4 (registered)":
                    r["ms"][vn] = r["v4"]
                    continue
                kw = dict(kw)
                if kw.get("F3") == {128: None, 64: None}:
                    kw.pop("F3")
                res = price(r["model"], "native", r["c"], r["doc"], prm, lock=1410, **kw)
                r["ms"][vn] = None if res is None else res["ms"]
        eq = [r for r in rows if r["ms"]["v5 all-off (must equal v4)"] != r["v4"]]
        print(f"REDUCTION v5(all off) == v4 bit for bit on {len(rows)} cells: {'YES' if not eq else 'NO ' + str(len(eq))}")
        out["reduction_v4_cells"] = {"cells": len(rows), "unequal": len(eq)}
        out["e1cal"] = {vn: summarise(rows, vn) for vn in variants}
        for vn, s in out["e1cal"].items():
            cm, bm_ = s["class_mean_pct"], s["g1_128_by_model_pct"]
            print(f"{vn:28s} 64g1 {cm['64g1']:+6.2f} 128g1 {cm['128g1']:+6.2f} 128g16 {cm['128g16']:+6.2f} | rms {s['rms_pct']:5.2f} worst {s['worst_pct']:+6.2f} "
                  f"| 128g1 OLMoE {bm_['olmoe-1b-7b']:+6.2f} Mixtral {bm_['mixtral-8x7b']:+6.2f} Qwen2 {bm_['qwen2-57b-a14b']:+6.2f} spread {s['g1_128_spread_pct']:.2f}")
        out["e1cal_cells"] = [{k: v for k, v in r.items() if k not in ("c", "doc")} for r in rows]
    (H / "v5price.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
