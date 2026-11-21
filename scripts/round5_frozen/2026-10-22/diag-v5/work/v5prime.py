"""Item 6: v5' re-priced on A2's E1-CAL NATIVE cells, SEEN and leave-one-model-out (LOMO), against v5 as designed.

Variants (each term labelled in DIAGNOSIS.md):
  v5        the design: F1 A2 pooled (50.13/750, 34.52/425), p_128 5.75, p_64 3.75, F3 per-GEMM refit at 128x128, F4 x2
            at both tiles
  v5'       F3 at 128x128: w1 M4 registered (CARRIED), w2 stage-2 refit ("w2b", bytes_lomo.py); p_64 6.0 (G 1 >= 2 waves,
            p64.py); F4 x 2 at 64x128 only (no 128x128 page tests it); F1 the B3 estimator (OLMoE f128 + Mixtral f128,
            OLMoE f64w8 + Mixtral f64w8 on the board); p_128 refitted in F2's estimator
  v5'M4     v5' with M4's registered constants for both GEMMs at 128x128 (no F3 refit; the LOMO byte winner)
  v5'K      v5' with knee2.py's per-CTA fixed DRAM-idle cost (R_inf 3799.7, S0 and p refitted) at 128x128
SEEN: every constant from all three CAL models. LOMO, held-out model H: p_128 (and S0) refitted on the knee rows
without H; F3 from bytes_lomo.json's fold without H; F1 from the B3 estimator's pages without H (H = OLMoE: Mixtral
f128 / f64w8 only; H = Mixtral: OLMoE's only; H = Qwen2-57B: both, the B3 estimator itself); p_64 cannot be held
out (Mixtral is the only 64x128 G 1 model; stated). Mixtral's 128g1 E1-CAL cells are scored in its fold too.
Also: v5 against v5' on B2's frozen cells (levers, model against model; never a prediction of record), and the
NATIVE path census of B2's cells. Writes v5prime.json."""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from contextlib import nullcontext

import d5
from d5 import A2, RULES, RG
import bw as BW
import knee2 as K2
import sag as S
import b2risk as B
import wave_split_bytes as W

A2T = d5.A2T
SAGP = {128: {"olmoe-1b-7b": "*olmoe-1b-7b-f128*/lock1410/r3c-g*.json", "mixtral-8x7b": "*mixtral-8x7b-f128*/lock1410/r3c-g*.json"},
        64: {"olmoe-1b-7b": "*olmoe-1b-7b-f64w8*/lock1410/r3c-g*.json", "mixtral-8x7b": "*mixtral-8x7b-f64w8*/lock1410/r3c-g*.json"}}
MODELS = ("olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b")


def sagfit(bm, models):
    files = []
    for m in models:
        files += sorted(A2T.glob(SAGP[bm][m]))
    r = S.fit(S.pts_of(files, 1410.0))
    return (r["A_us"], r["tau_us"])


def f3_from(blj, fold, form):
    d = blj[fold][form]["params"]
    return {128: {g: W.Params.from_json(d[g]) for g in ("w1", "w2")}, 64: d5.F3P[64]}


def price_rows(rows, models, *, F3, P, sag, f4_128, kf=None, rho64=2.0):
    out = []
    keep = d5.V.RHO_X[64]
    d5.V.RHO_X[64] = rho64
    try:
        return _price_rows(rows, models, F3=F3, P=P, sag=sag, f4_128=f4_128, kf=kf, out=out)
    finally:
        d5.V.RHO_X[64] = keep


def _price_rows(rows, models, *, F3, P, sag, f4_128, kf, out):
    with A2.scoring(RULES), (K2.kform(*kf) if kf else nullcontext()):
        for r in rows:
            if r["drop"] or r["model"] not in models:
                continue
            prm = d5.params_for(d5.SEL, r["model"])
            bm = int(r["doc"]["block_m"])
            ms = d5.v5_parts(r["model"], r["c"], r["doc"], prm, F3=F3, P=P, rho=(bm == 64 or f4_128), sag_const=sag[bm])["ms_sum"]
            out.append((r, ms / r["meas"] - 1))
    return out


def summ(res):
    e = [x for _, x in res]
    by_m, by_c, by_mc = {}, {}, {}
    for r, x in res:
        by_m.setdefault(r["model"], []).append(x)
        by_c.setdefault(r["cls"], []).append(x)
        by_mc.setdefault(f"{r['model']} {r['cls']}", []).append(x)
    f = lambda d: {k: round(100 * st.fmean(v), 2) for k, v in sorted(d.items())}  # noqa: E731
    g1 = {k.split()[0]: v for k, v in f(by_mc).items() if k.endswith("128g1")}
    return {"n": len(e), "rms_pct": round(100 * math.sqrt(st.fmean(x * x for x in e)), 2), "worst_pct": round(100 * max(e, key=abs), 2),
            "mean_pct": round(100 * st.fmean(e), 2), "class_means": f(by_c), "model_means": f(by_m),
            "model_rms": {k: round(100 * math.sqrt(st.fmean(x * x for x in v)), 2) for k, v in sorted(by_m.items())},
            "model_class_means": f(by_mc), "spread_128g1": round(max(g1.values()) - min(g1.values()), 2) if len(g1) > 1 else None}


def main():
    blj = json.load(open("bytes_lomo.json"))
    e1 = d5.e1cal_rows()
    knee = BW.knee_rows(1.0)                       # carried bw, F2's estimator rows
    kneeK = {}
    s0s = K2.S0S
    for s0 in s0s:
        with K2.kform(3799.726028703809, s0):
            kneeK[s0] = BW.knee_rows(1.0)
    p_seen = BW.best(knee)[0]

    def kbest(models, use_k):
        if not use_k:
            return None, BW.best([r for r in knee if r["model"] in models])[0]
        b = None
        for s0 in s0s:
            p, s, _ = BW.best([r for r in kneeK[s0] if r["model"] in models])
            if b is None or s < b[2]:
                b = (s0, p, s)
        return b[0], b[1]
    sag_seen_design = {128: d5.BOARD_A2[128], 64: d5.BOARD_A2[64]}
    sag_b3 = {bm: sagfit(bm, ("olmoe-1b-7b", "mixtral-8x7b")) for bm in (128, 64)}
    print("F1 B3 estimator (A2 board):", sag_b3, "design pooled:", sag_seen_design)
    variants = {
        "v5 (design)": dict(form="v5", p64=3.75, f4_128=True, use_k=False, sag="design"),
        "v5 (design), B3 F1": dict(form="v5", p64=3.75, f4_128=True, use_k=False, sag="b3"),
        "v5'": dict(form="w2b", p64=6.0, f4_128=False, use_k=False, sag="b3"),
        "v5' with p_64 3.75": dict(form="w2b", p64=3.75, f4_128=False, use_k=False, sag="b3"),
        "v5' with F4 at 128": dict(form="w2b", p64=6.0, f4_128=True, use_k=False, sag="b3"),
        "v5'K": dict(form="w2b", p64=6.0, f4_128=False, use_k=True, sag="b3"),
        "v5'M4": dict(form="M4", p64=6.0, f4_128=False, use_k=False, sag="b3"),
        "v5'M4K": dict(form="M4", p64=6.0, f4_128=False, use_k=True, sag="b3"),
        "v5'M4 rho_64 1.5": dict(form="M4", p64=6.0, f4_128=False, use_k=False, sag="b3", rho64=1.5),
        "v5'M4 p_64 3.75": dict(form="M4", p64=3.75, f4_128=False, use_k=False, sag="b3"),
    }
    out = {"sag_b3": sag_b3, "p_seen": p_seen, "variants": {}}
    for vn, v in variants.items():
        res = {}
        s0, p128 = kbest(MODELS, v["use_k"])
        kf = (3799.726028703809, s0) if v["use_k"] else None
        sag = sag_seen_design if v["sag"] == "design" else sag_b3
        F3 = d5.F3P if v["form"] == "v5" else f3_from(blj, "none (SEEN)", v["form"])
        seen = price_rows(e1, MODELS, F3=F3, P={128: p128, 64: v["p64"]}, sag=sag, f4_128=v["f4_128"], kf=kf, rho64=v.get("rho64", 2.0))
        res["SEEN"] = summ(seen)
        res["SEEN"]["constants"] = {"p_128": p128, "S0": s0, "p_64": v["p64"], "sag": sag}
        lomo_all = []
        res["LOMO"] = {}
        for H in MODELS:
            others = tuple(m for m in MODELS if m != H)
            s0H, pH = kbest(others, v["use_k"])
            kfH = (3799.726028703809, s0H) if v["use_k"] else None
            sagH = {bm: (sag[bm] if v["sag"] == "design" else sagfit(bm, tuple(m for m in ("olmoe-1b-7b", "mixtral-8x7b") if m != H))) for bm in (128, 64)}
            F3H = f3_from(blj, H, v["form"])
            rr = price_rows(e1, (H,), F3=F3H, P={128: pH, 64: v["p64"]}, sag=sagH, f4_128=v["f4_128"], kf=kfH, rho64=v.get("rho64", 2.0))
            lomo_all += rr
            res["LOMO"][H] = {**summ(rr), "constants": {"p_128": pH, "S0": s0H, "sag": sagH}}
        res["LOMO pooled"] = summ(lomo_all)
        out["variants"][vn] = res
        s, L = res["SEEN"], res["LOMO pooled"]
        print(f"== {vn}: SEEN rms {s['rms_pct']} worst {s['worst_pct']} class {s['class_means']} models {s['model_means']} 128g1 {s['model_class_means']} spread {s['spread_128g1']}")
        print(f"   LOMO pooled rms {L['rms_pct']} worst {L['worst_pct']} class {L['class_means']} models {L['model_means']} model rms {L['model_rms']} 128g1 spread {L['spread_128g1']}")
        for H in MODELS:
            print(f"   held out {H}: {res['LOMO'][H]['model_class_means']} rms {res['LOMO'][H]['rms_pct']} consts {res['LOMO'][H]['constants']}")
        sys.stdout.flush()
    # B2's frozen cells: each v5' variant against v5 (SEEN constants), levers only; and the path census
    cells = B.b2_cells()
    paths = {}
    sv = out["variants"]
    LV = {"v5'": ("w2b", 2.0), "v5'M4": ("M4", 2.0), "v5'M4 p_64 3.75": ("M4", 2.0), "v5'M4 rho_64 1.5": ("M4", 1.5)}
    lev = {k: {} for k in LV}
    with A2.scoring(RULES):
        for r in cells:
            prm = d5.params_for(d5.SEL, r["model"])
            bm = int(r["doc"]["block_m"])
            a = d5.v5_parts(r["model"], r["c"], r["doc"], prm)
            paths.setdefault(f"{r['model']} {r['cls']}", {}).setdefault(a["path"], 0)
            paths[f"{r['model']} {r['cls']}"][a["path"]] += 1
            for vn, (form, rho64) in LV.items():
                c5 = sv[vn]["SEEN"]["constants"]
                keep = d5.V.RHO_X[64]
                d5.V.RHO_X[64] = rho64
                try:
                    b_ = d5.v5_parts(r["model"], r["c"], r["doc"], prm, F3=f3_from(blj, "none (SEEN)", form), P={128: c5["p_128"], 64: c5["p_64"]},
                                     rho=(bm == 64), sag_const=c5["sag"][bm])
                finally:
                    d5.V.RHO_X[64] = keep
                lev[vn].setdefault(f"{r['model']} {r['cls']}", []).append(b_["ms_sum"] / a["ms_sum"] - 1)
    out["b2_levers_vs_v5"] = {vn: {k: {"mean_pct": round(100 * st.fmean(v), 2), "max_pct": round(100 * max(v, key=abs), 2), "n": len(v)}
                                   for k, v in sorted(d.items())} for vn, d in lev.items()}
    out["b2_paths"] = paths
    for vn, d in out["b2_levers_vs_v5"].items():
        print(f"B2 levers {vn} against v5 (SEEN constants):", {k: v["mean_pct"] for k, v in d.items()})
    print("B2 NATIVE path census:", paths)
    json.dump(out, open("v5prime.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
