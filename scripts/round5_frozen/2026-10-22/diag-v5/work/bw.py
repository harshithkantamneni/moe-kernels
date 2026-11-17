"""Item 2: the board-bandwidth hypothesis. The GEMM bw (v4's carried 3597.8 GB/s, the CAL engine's v2 fit) replaced in
the GEMM term only (TM.gemm_ms's bw argument scaled; B_other / bw untouched) by measured rates, p_128 refitted in F2's
own estimator (kneefit.py: 128x128 NATIVE G 1 + G 16 byte pages, A2 and S1, counted bytes, per-CTA smax, against
counted cycles x K), then:
  (a) the knee fit by page (cycles domain), Mixtral S1 b128g1 w1 and w2;
  (b) the time-domain GEMM part on A2's E1-CAL matched cells (own sag, R3 share in the GEMM: split.py's gemmR);
  (c) v5's SEEN E1-CAL re-price (design constants otherwise).
Rates (MEASURED, published calibration pages): A2 triad 3726.2 (calibration ceilingtriad, 1980 MHz), A2 read_stream
3884.9 (same page), A2 rulers read2d at the 1410 lock 3799.7 (rulers-lp1410). Writes bw.json."""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from contextlib import contextmanager

import d5
from d5 import A2, RULES, TM, V4, RG, price_parts
import kneefit as KF

BWS = {"carried 3597.8": 3597.8, "A2 triad 3726.2": 3726.2173727263944, "A2 read2d@1410 3799.7": 3799.726028703809,
       "A2 read_stream 3884.9": 3884.935749661172}
PS = [x / 4 for x in range(12, 49)]   # 3.0 .. 12.0
FIT_PAGES = [p for p in KF.PAGES if p[4] == "fit" and p[2] not in ("b64g1", "p64g1")]   # kneefit's 128 fit set


@contextmanager
def gemm_bw(f):
    orig = TM.gemm_ms
    TM.gemm_ms = lambda win, g, sigma, c_ns, bw, sms, occ=None, _o=orig: _o(win, g, sigma, c_ns, bw * f, sms, occ=occ)
    try:
        yield
    finally:
        TM.gemm_ms = orig


def knee_rows(f):
    rows = []
    for root, model, unit, lock, role in FIT_PAGES:
        cells = KF.cells_of(root, model, unit, lock)
        with A2.scoring(A2.Rules.tileA2(lock)):
            prm = d5.params_for(d5.SEL, model)
            for r in cells:
                if r["arm"] != "native":
                    continue
                t = RG.PAGE_TILE[r["doc"]["block_m"]]
                pred = {g: {} for g in ("w1", "w2")}
                keep = TM.P_KNEE
                try:
                    with gemm_bw(f):
                        for p in PS:
                            TM.P_KNEE = p
                            pp = price_parts(model, "native", r["c"], r["doc"], prm, key="CLAMP", reads=r["reads"])
                            for g in ("w1", "w2"):
                                pred[g][p] = pp[g]["gemm"] + pp[g]["dead"]
                finally:
                    TM.P_KNEE = keep
                for g in ("w1", "w2"):
                    K = pp[g]["c_ns"] / V4.c_counter(t, g, prm)
                    rows.append({"board": "A2" if root == KF.A2T else "S1", "model": model, "unit": unit, "G": r["doc"]["group_m"],
                                 "n": r["c"]["n"], "label": r["c"]["label"], "gemm": g, "x": pp[g]["bytes_only"] / pp[g]["floor_only"],
                                 "meas": r["cyc"][g] * K * 1e-6, "pred": pred[g]})
    return rows


def best(rows):
    b = None
    for p in PS:
        e = [r["pred"][p] / r["meas"] - 1 for r in rows]
        s = math.sqrt(st.fmean(x * x for x in e))
        if b is None or s < b[1]:
            b = (p, s, st.fmean(e))
    return b


def main():
    out = {}
    e1 = d5.e1cal_rows()
    split = {(r["page"], r["n"], r["label"]): r for r in json.load(open("split.json"))}
    for name, bw in BWS.items():
        f = bw / 3597.8
        rows = knee_rows(f)
        p, s, m = best(rows)
        res = {"bw": bw, "p_128": p, "knee_rms_pct": round(100 * s, 2), "knee_mean_pct": round(100 * m, 2), "by_page": {}, "lomo": {}}
        for key in sorted({(r["board"], r["model"], r["unit"], r["gemm"]) for r in rows}):
            sub = [r for r in rows if (r["board"], r["model"], r["unit"], r["gemm"]) == key]
            e = [r["pred"][p] / r["meas"] - 1 for r in sub]
            res["by_page"][" ".join(key)] = {"n": len(e), "mean_pct": round(100 * st.fmean(e), 2), "x": [round(min(r["x"] for r in sub), 2), round(max(r["x"] for r in sub), 2)]}
        for H in ("olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b"):
            pH = best([r for r in rows if r["model"] != H])[0]
            e = [r["pred"][pH] / r["meas"] - 1 for r in rows if r["model"] == H]
            res["lomo"][H] = {"p": pH, "held_out_mean_pct": round(100 * st.fmean(e), 2), "rms_pct": round(100 * math.sqrt(st.fmean(x * x for x in e)), 2)}
        # (b) time domain GEMM part on the matched E1-CAL cells, own sag, R3 share in the GEMM (no subtraction)
        P = {128: p, 64: d5.P5[64]}
        td, pgemm = {}, {}
        with A2.scoring(RULES), gemm_bw(f):
            for r in e1:
                if r["drop"] or r["cls"] == "64g1":
                    continue
                sp = split.get((r["page"], r["n"], r["label"]))
                if sp is None:
                    continue
                prm = d5.params_for(d5.SEL, r["model"])
                root, unit = __import__("split").BYTE[r["page"]]
                cc = d5.counter_page(root, r["model"], unit)[1][(r["n"], r["label"], "native")]
                for nm, sc in (("own", 1.0), ("own x0.67", 0.67)):
                    if nm not in sp["split"]:
                        continue
                    own = {g: cc["g"][g]["off_us"] * sc for g in ("w1", "w2")}
                    po = d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads={g: cc["g"][g]["reads"] for g in ("w1", "w2")},
                                     P=P, sag="own", own_off_us=own, r3=None)
                    for g in ("w1", "w2"):
                        meas = (cc["g"][g]["cyc"] / 1410.0 + own[g]) * 1e-3
                        mod = po[g]["gemm"] + po[g]["dead"] + po[g]["const"]
                        k = f"{r['model']} {r['cls']} [{nm}]"
                        td.setdefault(k, [0.0, 0.0])
                        td[k][0] += mod
                        td[k][1] += meas
                        pgemm.setdefault(f"{k} {g}", []).append(mod / meas - 1)
        res["time_domain_gemm_pct"] = {k: round(100 * (a / b - 1), 2) for k, (a, b) in td.items()}
        res["time_domain_per_gemm_mean_pct"] = {k: round(100 * st.fmean(v), 2) for k, v in pgemm.items()}
        # (c) SEEN E1-CAL re-price
        es = {}
        with A2.scoring(RULES), gemm_bw(f):
            for r in e1:
                if r["drop"]:
                    continue
                prm = d5.params_for(d5.SEL, r["model"])
                ms = d5.v5_parts(r["model"], r["c"], r["doc"], prm, P=P)["ms_sum"]
                es.setdefault((r["model"], r["cls"]), []).append(ms / r["meas"] - 1)
        allv = [x for v in es.values() for x in v]
        res["seen"] = {"rms_pct": round(100 * math.sqrt(st.fmean(x * x for x in allv)), 2), "worst_pct": round(100 * max(allv, key=abs), 2),
                       "by": {f"{m} {c}": round(100 * st.fmean(v), 2) for (m, c), v in sorted(es.items())}}
        out[name] = res
        print(f"== {name}: p_128 {p} knee rms {res['knee_rms_pct']} mean {res['knee_mean_pct']}; LOMO {res['lomo']}")
        print("   knee by page:", {k: v["mean_pct"] for k, v in res["by_page"].items()})
        print("   time-domain GEMM (own sag, R3 in GEMM):", res["time_domain_gemm_pct"])
        print("   per GEMM:", res["time_domain_per_gemm_mean_pct"])
        print("   SEEN E1-CAL:", res["seen"])
        sys.stdout.flush()
    json.dump(out, open("bw.json", "w"), indent=1)


if __name__ == "__main__":
    main()
