"""Score the 2026-10-01 L2-survival TP-shard registration (docs/registered/2026-10-01-l2-survival-tp-gh200.*)
on rental 1's pages. s per design-l2/work/surv_g1.py (same formula, same far-share decomposition);
G=2 secondary via wave_split_bytes.load_card(...).measured (the q the R0 numbers are in).
Run: PYTHONPATH=<scratch repo>:<scratch repo>/scripts python score_l2.py TREE_RESULTS REG_JSON OUT_DIR"""
import json, sys, glob, math
from pathlib import Path
import moe
from moe.spec import MODEL_CONFIGS as M
import wave_split_bytes as W
R, REG, OUT = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])
assert "score-r1/repo" in moe.__file__, moe.__file__
L2 = 62914560
def page(t, G): return R / f"2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-{t}-r3-counters/lock1710/r3c-g{G}.json"
res = {"pages": {}, "s": {}, "far_share": {}, "fabric": {}, "rules": {}}
def gates(p):
    d = json.load(open(p)); return d, [(g["number"], g["kind"]) for g in d["gates"] if g["verdict"] != "PASS"]
def surv(d):
    m = d["design"]["model"]; c = M[m]; H, I, E = c.hidden_size, c.intermediate_size, c.num_experts
    Wg = {"w1": E * 2 * I * H * 2, "w2": E * H * I * 2}
    cs = {(x["arm"], x["n"]): x for x in d["cells"]}
    out = {}
    for g in ("w1", "w2"):
        rec = cs[("private", 2)]["recorded"][g]
        occ = rec["launch__occupancy_limit_registers"]; Wc = occ * 132
        P = (2 * I if g == "w1" else H) // 64; K = H if g == "w1" else I
        x = (P - 1) * 64 * K * 2 / L2
        ss, fs = [], []
        for n in range(2, 10):
            a, b = cs[("shared", n)]["per_gemm"][g], cs[("private", n)]["per_gemm"][g]
            ss.append((b["dram_bytes_read"] - a["dram_bytes_read"]) / ((n - 1) * Wg[g]))
            def parts(x_, r):
                T = x_["l2_tex_read_sectors"]; Hh = x_["l2_tex_read_hit_sectors"]; Mm = x_["l2_read_miss_sectors"]
                F = r["lts__t_sectors_srcunit_ltcfabric.sum"]; Fm = Mm - (T - Hh); return Hh, F - Fm
            hs, fsh = parts(a, cs[("shared", n)]["recorded"][g]); hp, fp = parts(b, cs[("private", n)]["recorded"][g])
            dh, df = hs - hp, fsh - fp
            fs.append(df / (dh + df) if dh + df > 0 else float("nan"))
        out[g] = dict(x=x, d=P / Wc, P=P, Wc=Wc, s=ss, far=fs)
    return out
def fabric(d):
    v = []
    for c in d["cells"]:
        if c["arm"] != "private": continue
        for g in ("w1", "w2"):
            pg = c["per_gemm"][g]; F = c["recorded"][g]["lts__t_sectors_srcunit_ltcfabric.sum"]
            Mn = pg["l2_tex_read_sectors"] - pg["l2_tex_read_hit_sectors"]; v.append((c["n"], g, F / Mn))
    return v
S1 = {}
for t in ("tp2-l2", "tp4-l2", "tp8-l2", "sameboard"):
    for G in ((1, 2) if t != "sameboard" else (1,)):
        p = page(t, G); d, fl = gates(p); res["pages"][f"{t} G={G}"] = fl
        fab = fabric(d); res["fabric"][f"{t} G={G}"] = [min(f[2] for f in fab), max(f[2] for f in fab), len(fab)]
        if G == 1:
            S1[t] = surv(d)
pred = {(c["model"], c["gemm"]): c for c in REG["predictions_G1"]}
for t in ("tp2", "tp4", "tp8"):
    for g in ("w1", "w2"):
        m = f"mixtral-8x7b-{t}"; r = S1[f"{t}-l2"][g]; pr = pred[(m, g)]
        res["s"][f"{t} {g}"] = dict(x=r["x"], d=r["d"], Wc=r["Wc"], meas=r["s"], H1=pr["H1_H2a"], H0=pr["H0_H2a"], far=r["far"])
anc = S1["sameboard"]["w2"]["s"]; res["s"]["sameboard 8x7B w2"] = dict(meas=anc, far=S1["sameboard"]["w2"]["far"], x=S1["sameboard"]["w2"]["x"], d=S1["sameboard"]["w2"]["d"])
res["s"]["sameboard 8x7B w1"] = dict(meas=S1["sameboard"]["w1"]["s"], x=S1["sameboard"]["w1"]["x"], d=S1["sameboard"]["w1"]["d"])
N = list(range(2, 10)); ix = lambda n: n - 2
s = lambda k: res["s"][k]["meas"]
# rule 1: tp2 w2
t2 = s("tp2 w2"); dev = [abs(t2[ix(n)] - anc[ix(n)]) for n in range(4, 10)]
h1 = max(dev) <= 0.08 and all(t2[ix(n)] < 0.65 for n in range(4, 10)); h0 = all(t2[ix(n)] >= 0.74 for n in range(4, 10))
res["rules"]["H1_vs_H0_tp2_w2"] = dict(verdict="H1" if h1 else "H0" if h0 else "INCONCLUSIVE", max_dev_from_anchor_n4_9=max(dev), max_s_n4_9=max(t2[2:]), min_s_n4_9=min(t2[2:]))
# rule 2: tp4 w1
t4 = s("tp4 w1"); P1 = res["s"]["tp4 w1"]["H1"]; P0 = res["s"]["tp4 w1"]["H0"]
r6 = range(6, 10)
v1 = all(t4[ix(n)] < 0.40 for n in r6) and all(abs(t4[ix(n)] - P1[ix(n)]) <= 0.08 for n in r6)
v0 = all(t4[ix(n)] > 0.40 for n in r6) and all(abs(t4[ix(n)] - P0[ix(n)]) <= 0.08 for n in r6)
res["rules"]["H1_vs_H0_tp4_w1"] = dict(verdict="H1" if v1 else "H0" if v0 else "INCONCLUSIVE",
    dev_H1_n6_9=[t4[ix(n)] - P1[ix(n)] for n in r6], dev_H0_n6_9=[t4[ix(n)] - P0[ix(n)] for n in r6])
# rule 3: x-only rival
diffs = [t2[ix(n)] - t4[ix(n)] for n in range(4, 10)]; k = sum(abs(x) > 0.10 for x in diffs)
res["rules"]["x_only_rival"] = dict(verdict="FALSIFIED" if k >= 3 else "HELD", n_over_0p10=k, diffs_tp2w2_minus_tp4w1_n4_9=diffs)
# controls
c = {}
for key in ("tp8 w2", "tp8 w1"): c[key] = dict(min=min(s(key)), ok=min(s(key)) >= 0.95)
row = res["s"]["tp4 w2"]["H1"]; dv = [m_ - p for m_, p in zip(s("tp4 w2"), row)]
c["tp4 w2"] = dict(dev=dv, ok=max(abs(x) for x in dv) <= 0.05)
c["tp2 w1"] = dict(max=max(s("tp2 w1")), ok=max(s("tp2 w1")) <= 0.08)
res["rules"]["controls"] = c
fb = res["fabric"]; res["rules"]["fabric_control"] = dict(ok=all(0.50 <= v[0] and v[1] <= 0.56 for v in fb.values()), ranges=fb)
rec = {}
for kk, v in res["s"].items():
    if "far" in v: rec[kk] = [(n, round(v["meas"][ix(n)], 3), round(v["far"][ix(n)], 3)) for n in N if v["meas"][ix(n)] >= 0.1]
res["rules"]["far_share_record"] = rec
# secondary G=2 vs R0 (q via load_card)
R0 = {(c_["target"], c_["arm"], c_["gemm"]): c_["q"] for c_ in REG["secondary_G2"]["cells"] if c_["G"] == 2}
sec = []
for t in ("tp2", "tp4", "tp8"):
    import tempfile, shutil, os
    td = Path(tempfile.mkdtemp()); shutil.copy(page(f"{t}-l2", 2), td / "r3c-g2.json")
    card = W.load_card(td)
    for (arm, G, n, g), v in card.measured.items():
        if arm == "native": continue
        p = R0[(f"mixtral-8x7b-{t}", arm, g)][n - 1]
        sec.append(dict(t=t, arm=arm, n=n, gemm=g, meas=v, R0=p, err=None if p is None else (p / v - 1) * 100))
res["secondary_G2"] = sec
bad = [x for x in sec if x["err"] is not None and abs(x["err"]) > 5]
res["rules"]["secondary_G2"] = dict(scored=sum(x["err"] is not None for x in sec), beyond5=len(bad),
    worst=max((x for x in sec if x["err"] is not None), key=lambda x: abs(x["err"])))
OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "score.json", "w"), indent=1)
f = lambda v: " ".join(f"{x:.2f}" for x in v)
L = ["L2 survival TP-shard registration, rental 1 scoring (pages lock1710, G=1 n=2..9)", ""]
L.append("page gates not PASS: " + "; ".join(f"{k}: {v}" for k, v in res["pages"].items()))
L.append("")
for kk, v in res["s"].items():
    L.append(f"{kk:20} x {v['x']:.3f} d {v['d']:.3f}  meas {f(v['meas'])}" + (f"\n{'':20}            H1 {f(v['H1'])}\n{'':20}            H0 {f(v['H0'])}" if "H1" in v else ""))
    if "far" in v: L.append(f"{'':20}         far {f(v['far'])}")
L.append("")
for kk, v in res["rules"].items():
    if kk == "far_share_record": continue
    L.append(f"{kk}: {json.dumps(v, default=lambda o: round(o, 4))}")
L.append("")
L.append("secondary G=2 (R0 / measured - 1, %):")
for t in ("tp2", "tp4", "tp8"):
    for arm in ("shared", "private"):
        for g in ("w1", "w2"):
            xs = sorted((x for x in sec if x["t"] == t and x["arm"] == arm and x["gemm"] == g), key=lambda x: x["n"])
            L.append(f"  {t} {arm:7} {g} meas " + " ".join(f"{x['meas']:.3f}" for x in xs) + " | err " + " ".join("--" if x["err"] is None else f"{x['err']:+.1f}" for x in xs))
(OUT / "score.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
