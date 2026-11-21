"""Split the T0 fit's own prediction on its Mixtral-8x7B fit cells into GEMM / non-GEMM, beside the pages'
event-timed call and ncu's GEMM time. Read-only."""
import sys, json, glob, statistics as st
from pathlib import Path
R = Path.home() / "code/moe-kernels"
sys.path[:0] = [str(R), str(R / "scripts"), str(R / "scripts/scoring/rental5"), str(R / "scripts/scoring/rental6")]
import skewmodel as SM
import r3_timing_model as TM
eng = SM.Engine()
x, k_w, ctx = eng.fit2.x, eng.fit2.k_w, eng.ctx
T0, c_ns, bw, s_small, s_block = (float(v) for v in x)
print(f"fit2: T0 {T0*1e3:.2f} us, c {c_ns:.1f} ns, bw {bw:.1f} GB/s, s_small {s_small*1e3:.2f} us, s_block {s_block*1e3:.2f} us, k_w {k_w}")
cnt = {}
base = R / "results/published/2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710"
for f in glob.glob(str(base / "r3c-g*.json")):
    j = json.load(open(f)); G = j["design"]["group_m"]
    for c in j["cells"]:
        cnt[(G, c["arm"], c["n"])] = c["per_call"]["gpu_time_ns"] / 1e6
rows = []
with TM.dead_model("v2"):
    for c in eng.src2["cells"]:
        if not c.fit:
            continue
        counts = getattr(c, "counts", None)
        treads = c.n if counts is None else TM.numel_of(c.n, counts) / TM.IDS_PER_TREAD
        other = treads * TM.B_OTHER / (bw * 1e6)
        g_ms = 0.0
        for g in TM.GEMMS:
            win = TM.window(c.arm, c.declared, c.G, c.n, g, TM.window_width(k_w, ctx.sms, ctx.occupancy[g]), counts)
            g_ms += TM.gemm_ms(win, g, c.sigma[g], c_ns, bw, ctx.sms, occ=ctx.occupancy[g])
            g_ms += TM.dead_ms(c.declared, c.n, g, c_ns, k_w, ctx.occupancy[g], counts)
            g_ms += TM.gemm_const_ms(g, c_ns)
        s = (s_small if c.path == TM.SMALL_BATCH else s_block) if c.arm == "native" else 0.0
        nong = T0 + other + s
        tot = TM.call_ms(x, c, ctx, k_w)
        ncu = cnt.get((c.G, c.arm, c.n))
        rows.append(dict(G=c.G, arm=c.arm, n=c.n, meas=c.ms, pred=tot, gemm_pred=g_ms, nong_pred=nong, ncu_gemm=ncu))
print(f"{len(rows)} fit cells")
for arm in ("native", "shared", "private"):
    r = [d for d in rows if d["arm"] == arm and d["ncu_gemm"]]
    if not r: continue
    f = lambda k: st.median(k(d) for d in r) * 1e3
    print(f"{arm:8s} n={len(r)}: fit total - measured {f(lambda d: d['pred']-d['meas']):+.1f} us | "
          f"GEMMs: fit {f(lambda d: d['gemm_pred']):.0f} vs ncu {f(lambda d: d['ncu_gemm']):.0f} -> fit - ncu {f(lambda d: d['gemm_pred']-d['ncu_gemm']):+.1f} us | "
          f"non-GEMM: fit {f(lambda d: d['nong_pred']):.1f} vs measured-minus-ncu {f(lambda d: d['meas']-d['ncu_gemm']):.1f} us")
json.dump(rows, open(sys.argv[1], "w"))
