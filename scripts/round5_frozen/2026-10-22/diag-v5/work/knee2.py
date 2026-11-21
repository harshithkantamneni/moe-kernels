"""Item 2 follow-up: a per-CTA fixed DRAM-idle cost on the byte branch, priced from shape, at 128x128 only.

rate_g = R_inf x S_g / (S_g + S0), S_g the GEMM's k-steps (K / 64, the model config), applied to the GEMM term's bw
when TM.BLOCK_M == 128 (64x128 untouched: carried bw, p_64 3.75). Motivation (rate.py, rate_waves.py, MEASURED): at
x 1.08 to 1.17 the steady-cycle read rate rises with K: K 1024 2.75-2.80, 2048 3.20-3.27, 2560 3.26, 3584 3.32-3.37,
4096 3.40-3.48 TB/s. R_inf: the carried 3597.8 or A2's ruler read2d at the 1410 lock, 3799.7 (MEASURED; a ruler
runs on every rental board at the lock). S0 on a grid, p refitted per S0 in F2's estimator (kneefit's 128 NATIVE fit
rows, counted bytes, cycles x K). Then LOMO (p and S0 refitted without each model), the time-domain GEMM part and the
SEEN E1-CAL re-price. Writes knee2.json."""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from contextlib import contextmanager

import d5
from d5 import A2, RULES, TM
import bw as BW

S0S = [0.0, 3.0, 6.0, 9.0, 12.0]
RINF = {"carried 3597.8": 3597.8, "read2d@1410 3799.7": 3799.726028703809}


@contextmanager
def kform(rinf, s0):
    orig = TM.gemm_ms

    def w(win, g, sigma, c_ns, bw, sms, occ=None, _o=orig):
        if TM.BLOCK_M == 128 and bw < 1e9:
            ks = TM.GEOMETRY[g].ksteps
            bw = bw * (rinf / 3597.8) * ks / (ks + s0)
        return _o(win, g, sigma, c_ns, bw, sms, occ=occ)
    TM.gemm_ms = w
    try:
        yield
    finally:
        TM.gemm_ms = orig


def main():
    grid = {}
    for rn, rinf in RINF.items():
        for s0 in S0S:
            # BW.knee_rows applies a uniform factor through BW.gemm_bw; nest kform inside a factor of 1
            with kform(rinf, s0):
                rows = BW.knee_rows(1.0)
            grid[(rn, s0)] = rows
            p, s, m = BW.best(rows)
            print(f"{rn} S0 {s0:4.1f}: p {p} knee rms {100 * s:.2f} mean {100 * m:+.2f}")
            sys.stdout.flush()
    # pick (R_inf, S0, p) by SEEN rms, and by LOMO (fit S0 and p without the model, score the model)
    out = {"grid": {f"{k[0]} S0 {k[1]}": dict(zip(("p", "rms", "mean"), BW.best(v))) for k, v in grid.items()}, "lomo": {}}
    for rn in RINF:
        lo = {}
        for H in ("olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b"):
            best = None
            for s0 in S0S:
                tr = [r for r in grid[(rn, s0)] if r["model"] != H]
                p, s, _m = BW.best(tr)
                if best is None or s < best[2]:
                    best = (s0, p, s)
            te = [r for r in grid[(rn, best[0])] if r["model"] == H]
            e = [r["pred"][best[1]] / r["meas"] - 1 for r in te]
            lo[H] = {"S0": best[0], "p": best[1], "held_out_mean_pct": round(100 * st.fmean(e), 2), "held_out_rms_pct": round(100 * math.sqrt(st.fmean(x * x for x in e)), 2)}
        out["lomo"][rn] = lo
        print("LOMO", rn, lo)
    # by page at the SEEN-best S0 per R_inf
    for rn in RINF:
        s0b = min(S0S, key=lambda s0: BW.best(grid[(rn, s0)])[1])
        p = BW.best(grid[(rn, s0b)])[0]
        rows = grid[(rn, s0b)]
        bp = {}
        for key in sorted({(r["board"], r["model"], r["unit"], r["gemm"]) for r in rows}):
            e = [r["pred"][p] / r["meas"] - 1 for r in rows if (r["board"], r["model"], r["unit"], r["gemm"]) == key]
            bp[" ".join(key)] = round(100 * st.fmean(e), 2)
        out[f"by_page {rn}"] = {"S0": s0b, "p": p, "by_page": bp}
        print(f"by page {rn} S0 {s0b} p {p}:", bp)
        # time-domain GEMM part and SEEN E1-CAL at (rinf, s0b, p)
        e1 = d5.e1cal_rows()
        sp = {(r["page"], r["n"], r["label"]): r for r in json.load(open("split.json"))}
        import split as SPL
        td, es = {}, {}
        with A2.scoring(RULES), kform(RINF[rn], s0b):
            for r in e1:
                if r["drop"]:
                    continue
                prm = d5.params_for(d5.SEL, r["model"])
                P = {128: p, 64: d5.P5[64]}
                ms = d5.v5_parts(r["model"], r["c"], r["doc"], prm, P=P)["ms_sum"]
                es.setdefault((r["model"], r["cls"]), []).append(ms / r["meas"] - 1)
                s = sp.get((r["page"], r["n"], r["label"]))
                if s is None or r["cls"] == "64g1":
                    continue
                root, unit = SPL.BYTE[r["page"]]
                cc = d5.counter_page(root, r["model"], unit)[1][(r["n"], r["label"], "native")]
                for nm, sc in (("own", 1.0), ("own x0.67", 0.67)):
                    if nm not in s["split"]:
                        continue
                    own = {g: cc["g"][g]["off_us"] * sc for g in ("w1", "w2")}
                    po = d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads={g: cc["g"][g]["reads"] for g in ("w1", "w2")}, P=P,
                                     sag="own", own_off_us=own, r3=None)
                    for g in ("w1", "w2"):
                        meas = (cc["g"][g]["cyc"] / 1410.0 + own[g]) * 1e-3
                        k = f"{r['model']} {r['cls']} [{nm}] {g}"
                        td.setdefault(k, [0.0, 0.0])
                        td[k][0] += po[g]["gemm"] + po[g]["dead"] + po[g]["const"]
                        td[k][1] += meas
        allv = [x for v in es.values() for x in v]
        out[f"seen {rn}"] = {"rms_pct": round(100 * math.sqrt(st.fmean(x * x for x in allv)), 2), "worst_pct": round(100 * max(allv, key=abs), 2),
                             "by": {f"{m} {c}": round(100 * st.fmean(v), 2) for (m, c), v in sorted(es.items())},
                             "time_domain_gemm_pct": {k: round(100 * (a / b - 1), 2) for k, (a, b) in td.items()}}
        print(f"SEEN {rn}:", out[f"seen {rn}"])
        sys.stdout.flush()
    json.dump(out, open("knee2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
