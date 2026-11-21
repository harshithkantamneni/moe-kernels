"""Score the 2026-10-01 atile-ksteps-gh200 registration on rental 1's pages.

Predictions are the registered numbers (docs/registered/2026-10-01-atile-ksteps-gh200.json,
cells_private and tests), unchanged. Measured q is wave_split_bytes.load_card (DCR.r3_q) on
each lock1710 page directory of the run tree. A cross-check recomputes R1/R2 on the MEASURED
pages' geometry with the registered params (W.Model content_a=True) to confirm the registered
numbers are those of the pages that landed.
Run: PYTHONPATH=REPO:REPO/scripts python score_atile.py REPO TREE_RESULTS OUTDIR
"""
import json, math, sys
from pathlib import Path
import numpy as np
REPO, RES, OUT = map(Path, sys.argv[1:4])
import wave_split_bytes as W

reg = json.loads((REPO / "docs/registered/2026-10-01-atile-ksteps-gh200.json").read_text())
pred = reg["cells_private"]
CANDS = ["R0", "R1-cal", "R2-cal", "R1-all8", "R2-all8"]
DIRS = {
    "mixtral-8x7b-tp8": ["mixtral-8x7b-tp8-atile", "mixtral-8x7b-tp8-l2"],
    "mixtral-8x7b-tp4": ["mixtral-8x7b-tp4-atile", "mixtral-8x7b-tp4-l2"],
    "mixtral-8x7b-tp2": ["mixtral-8x7b-tp2-atile", "mixtral-8x7b-tp2-l2"],
    "mixtral-8x7b": ["mixtral-8x7b-sameboard"],
    "qwen2-57b-a14b": ["qwen2-57b-a14b-g128"],
    "olmoe-1b-7b": ["olmoe-1b-7b-g128"],
}
meas, cards, src = {}, {}, {}
for model, ds in DIRS.items():
    for d in ds:
        p = next(RES.glob(f"*-{d}-r3-counters/lock1710"))
        c = W.load_card(p)
        assert c.geom.model == model, (c.geom.model, model)
        cards[(model, d)] = c
        for (arm, G, n, g), v in c.measured.items():
            if arm == "private":
                meas[f"{model}:{G}:{n}:{g}"] = v
                src[f"{model}:{G}:{n}:{g}"] = str(c.paths[G].relative_to(RES))
out = {"registration": "2026-10-01-atile-ksteps-gh200", "pred_minus": "residual = pred/meas - 1"}

# cross-check: recompute R1/R2 on the measured geometry with the registered params
fits = reg["fits"]
prm = {}
for tier, tag in (("primary", "cal"), ("secondary", "all8")):
    fk = next(k for k in fits if k.startswith(tier))
    for r in ("R1", "R2"):
        P = dict(fits[fk][r]["params"]); P.pop("view", None)
        prm[f"{r}-{tag}"] = W.Params(view=W.MIX, **P)
xc = []
for (model, d), c in cards.items():
    cells = [("private", G, n, g) for g in W.GEMMS for G in sorted(c.pages) for n in c.treads]
    b = W.Model(c.geom, content_a=True).batch(cells)
    for name, p in prm.items():
        v = b.solve(p)
        for (arm, G, n, g), x in zip(cells, v):
            k = f"{model}:{G}:{n}:{g}"
            if k in pred:
                xc.append(abs(x / pred[k][name] - 1))
out["crosscheck_max_rel_dev_R1R2_vs_registered"] = float(max(xc))
out["crosscheck_cells"] = len(xc)
missing = sorted(set(pred) - set(meas)); extra = sorted(set(meas) - set(pred))
out["registered_cells_without_page"] = missing
out["measured_cells_without_registration"] = extra

def r(k, c): return pred[k][c] / meas[k] - 1

# T1
t1 = {}
for model in ("qwen2-57b-a14b", "olmoe-1b-7b"):
    rows = []
    for n in range(1, 10):
        k = f"{model}:128:{n}:w1"
        m = meas[k]; p0, p1 = pred[k]["R0"], pred[k]["R1-cal"]
        rows.append({"n": n, "q": round(m, 4), "R0": p0, "R1": p1, "R2": pred[k]["R2-cal"],
                     "dev_R1_pct": round(100 * (m / p1 - 1), 2), "dev_R0_pct": round(100 * (m / p0 - 1), 2),
                     "nearer": "R1" if abs(m - p1) < abs(m - p0) else "R0"})
    within = all(abs(x["dev_R1_pct"]) <= 2 for x in rows if x["n"] >= 2)
    near_r0 = [x["n"] for x in rows if x["n"] >= 4 and x["nearer"] == "R0"]
    t1[model] = {"rows": rows, "within2pct_R1_all_n>=2": within,
                 "worst_dev_R1_n>=2_pct": max((x["dev_R1_pct"] for x in rows if x["n"] >= 2), key=abs),
                 "nearer_R0_at_n>=4": near_r0,
                 "verdict": "FALSIFIED" if near_r0 else ("HELD" if within else "NOT HELD (inside neither pass nor falsifier: within-2% pass fails, nearer R1 everywhere)")}
out["T1"] = t1

# T2
def f(k):
    n = int(k.split(":")[2]); return (meas[k] - n) * 128 / (63 * n)
fv = {"tp4 G64": f("mixtral-8x7b-tp4:64:9:w2"), "8x7B G16": f("mixtral-8x7b:16:9:w2"),
      "tp2 G64": f("mixtral-8x7b-tp2:64:9:w2"), "8x7B G32": f("mixtral-8x7b:32:9:w2")}
rho = {"rho42": fv["tp4 G64"] / fv["8x7B G16"], "rho84": fv["tp2 G64"] / fv["8x7B G32"]}
bands = reg["tests"]["T2 duration term (separates R2 from every byte law, R0 and R1)"]["rho"]
t2 = {"f_n9": {k: round(v, 4) for k, v in fv.items()}, "rho": {k: round(v, 3) for k, v in rho.items()}, "cands": {}}
sig_q = 0.03641213879116405
sf = sig_q * 128 / (63 * 9)
t2["sigma_rho_measured_approx"] = {k: round(rho[k] * math.hypot(sf / fv[a], sf / fv[b]), 3)
                                    for k, a, b in (("rho42", "tp4 G64", "8x7B G16"), ("rho84", "tp2 G64", "8x7B G32"))}
for c in CANDS:
    ins = {k: bands[k][c]["band"][0] <= rho[k] <= bands[k][c]["band"][1] for k in rho}
    t2["cands"][c] = {"bands": {k: bands[k][c]["band"] for k in rho}, "pred": {k: bands[k][c]["rho"] for k in rho},
                      "inside": ins, "f_pred": {k: (bands[k][c]["f_num"], bands[k][c]["f_den"]) for k in rho},
                      "verdict": "FALSIFIED" if not any(ins.values()) else ("HELD" if all(ins.values()) else "INCONCLUSIVE")}
# 21 MiB row, supporting only
iso = reg["tests"]["T2 duration term (separates R2 from every byte law, R0 and R1)"]["iso_f_n9"]
rowm = {}
for size, d in iso.items():
    for key, pr in d.items():
        mdl, G = key.split(":G"); k = f"{mdl}:{G}:9:w2"
        rowm[f"{size} {key}"] = {"measured_f": round(f(k), 4), **pr}
t2["iso_rows_n9"] = rowm
out["T2"] = t2

# T3
cr = reg["tests"]["T3 G=1 credit (separates content-keyed sharing from no sharing at G=1)"]["credit"]
t3 = {}
for m, c in cr.items():
    q1 = meas[f"{m}:1:1:w1"] - 1
    inR1 = c["band_R1_cal"][0] <= q1 <= c["band_R1_cal"][1]
    inR0 = c["band_R0"][0] <= q1 <= c["band_R0"][1]
    v = ("PASS" if inR1 and not inR0 else "FAIL") if c["discriminating"] else "RECORD"
    t3[m] = {"q_minus_1": round(q1, 5), "R1": c["R1_cal_q_minus_1"], "R0": c["R0_q_minus_1"],
             "full_credit": c["full_credit_q_minus_1"], "inside_R1_band": inR1, "inside_R0_band": inR0,
             "verdict": v, "page": src[f"{m}:1:1:w1"]}
# board check: same-board 8x7B G=1 n=1 w1 against the eight published values is not here; record it
t3["mixtral-8x7b (board check)"] = {"q_minus_1": round(meas["mixtral-8x7b:1:1:w1"] - 1, 5),
                                    "R1": round(pred["mixtral-8x7b:1:1:w1"]["R1-cal"] - 1, 5),
                                    "R0": round(pred["mixtral-8x7b:1:1:w1"]["R0"] - 1, 5)}
out["T3"] = t3

# T4
sc = [k for k in pred if int(k.split(":")[1]) != 1]
t4 = {}
for c in CANDS:
    res = np.array([r(k, c) for k in sc])
    big = [(k, round(100 * r(k, c), 2)) for k in sc if int(k.split(":")[1]) >= 32 and int(k.split(":")[2]) >= 6
           and abs(r(k, c)) > 0.04]
    per = {}
    for mdl in DIRS:
        ks = [k for k in sc if k.split(":")[0] == mdl]
        a = np.array([r(k, c) for k in ks])
        per[mdl] = {"rms_pct": round(100 * float(np.sqrt(np.mean(a ** 2))), 2),
                    "worst_pct": round(100 * float(a[np.argmax(abs(a))]), 2)}
    pg = {}
    for g in ("w1", "w2"):
        a = np.array([r(k, c) for k in sc if k.endswith(g)])
        pg[g] = round(100 * float(np.sqrt(np.mean(a ** 2))), 2)
    rms = 100 * float(np.sqrt(np.mean(res ** 2)))
    t4[c] = {"cells": len(sc), "rms_pct": round(rms, 2), "worst_pct": round(100 * float(res[np.argmax(abs(res))]), 2),
             "worst_cell": sc[int(np.argmax(abs(res)))], "G>=32_n>=6_beyond_4pct": big, "per_model": per, "per_gemm_rms": pg,
             "verdict": "HELD" if rms <= 2 and not big else "FALSIFIED"}
out["T4"] = t4
# per-cell table
out["cells"] = {k: {"q": round(meas[k], 5), **{c: pred[k][c] for c in CANDS},
                    **{f"res_{c}_pct": round(100 * r(k, c), 2) for c in CANDS}, "page": src[k]} for k in sorted(pred)}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "score.json").write_text(json.dumps(out, indent=1))

L = []
P = L.append
P("atile-ksteps-gh200 (registered 2026-10-01 at 436e41c) scored on rental 1, run gh200-rental1-20261001T2017Z")
P(f"cross-check: R1/R2 recomputed on the measured pages' geometry with the registered params: max |dev| {100*out['crosscheck_max_rel_dev_R1R2_vs_registered']:.4f}% over {len(xc)} cell-values")
P(f"registered cells without a page: {missing}; measured PRIVATE cells without a registration: {len(extra)} (G=2 l2 pages, not scored here)")
P("")
P("T1 PRIVATE w1 G=128 (pass: within 2% of R1-cal at every n>=2; falsified: nearer R0 than R1 at any n>=4)")
for m, t in t1.items():
    P(f"  {m}: {t['verdict']}; worst dev from R1 (n>=2) {t['worst_dev_R1_n>=2_pct']:+.2f}%; nearer R0 at n>=4: {t['nearer_R0_at_n>=4']}")
    for x in t["rows"]:
        P(f"    n={x['n']} q {x['q']:.4f}  R0 {x['R0']:.4f} ({x['dev_R0_pct']:+.2f}%)  R1 {x['R1']:.4f} ({x['dev_R1_pct']:+.2f}%)  R2 {x['R2']:.4f}  nearer {x['nearer']}")
P("")
P("T2 w2 n=9, f = (q - n) 128 / (63 n)")
P("  measured f: " + ", ".join(f"{k} {v:.4f}" for k, v in t2["f_n9"].items()))
P(f"  rho42 {rho['rho42']:.3f}, rho84 {rho['rho84']:.3f} (approx 1 sigma from sigma_q: {t2['sigma_rho_measured_approx']})")
for c, t in t2["cands"].items():
    P(f"  {c:8} rho42 {t['pred']['rho42']:.3f} {t['bands']['rho42']} in={t['inside']['rho42']}; rho84 {t['pred']['rho84']:.3f} {t['bands']['rho84']} in={t['inside']['rho84']}: {t['verdict']}")
P("  iso rows (supporting only), measured f vs predictions:")
for k, v in rowm.items():
    P(f"    {k:28} meas {v['measured_f']:.4f}  R0 {v['R0']:.4f} R1 {v['R1-cal']:.4f} R2 {v['R2-cal']:.4f} R1sec {v['R1-all8']:.4f} R2sec {v['R2-all8']:.4f}")
P("")
P("T3 PRIVATE w1 G=1 n=1, q - 1 (pass inside R1's +-0.0010 band and outside R0's)")
for m, t in t3.items():
    P(f"  {m}: " + ", ".join(f"{a} {b}" for a, b in t.items() if a != "page"))
P("")
P("T4 every PRIVATE cell of the scoring pages, G=1 aside (pass: rms <= 2% and no G>=32 n>=6 cell beyond 4%)")
for c, t in t4.items():
    P(f"  {c:8} {t['verdict']}: rms {t['rms_pct']:.2f}% over {t['cells']} cells, worst {t['worst_pct']:+.2f}% ({t['worst_cell']}), w1 rms {t['per_gemm_rms']['w1']}%, w2 rms {t['per_gemm_rms']['w2']}%; G>=32 n>=6 beyond 4%: {len(t['G>=32_n>=6_beyond_4pct'])} {t['G>=32_n>=6_beyond_4pct'][:6]}")
    P("           per model: " + "; ".join(f"{m} {v['rms_pct']}/{v['worst_pct']:+}" for m, v in t["per_model"].items()))
P("")
P("cells (q measured; residual pred/meas - 1 in %): key q | R0 R1 R2 R1sec R2sec")
for k, v in out["cells"].items():
    P(f"  {k:28} {v['q']:.4f} | " + " ".join(f"{v['res_'+c+'_pct']:+6.2f}" for c in CANDS))
(OUT / "score.txt").write_text("\n".join(L) + "\n")
print("\n".join(L[:80]))
