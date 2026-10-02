"""Score the 2026-09-30 mixtral-8x7b-tp8 floor registration on rental 1's captures.
Run with PYTHONPATH=<scratch repo>:<scratch repo>/scripts. Reads only."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(sys.argv[1]); TREE = Path(sys.argv[2]); OUT = Path(sys.argv[3])
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import r3_timing_model as TM
assert Path(TM.__file__).resolve().is_relative_to(REPO.resolve()), TM.__file__
REG = json.loads((REPO / "docs/registered/2026-09-30-mixtral-8x7b-tp8-floor-gh200.json").read_text())
P = REG["predictions"]; c, bw = P["source_fit"]["c"], P["source_fit"]["bw"]; k_w = P["k_w"]
occ = P["occupancy"]; SMS, MHZ = 132, 1710.0
LAWS = {"cores": dict(TAIL=True, CORES=True, CORES_PER_LEAD=False),
        "old_lifetime": dict(TAIL=True, CORES=False, CORES_PER_LEAD=False),
        "cores_per_lead": dict(TAIL=True, CORES=True, CORES_PER_LEAD=True),
        "throughput": dict(TAIL=False, CORES=True, CORES_PER_LEAD=False)}
regcell = {(r["G"], r["n"], r["gemm"]): r for r in P["cells"]}
UNIT = REG["unit_cycles"]["w1"]
D_REG = REG["discriminant_D_w1_cycles"]["G=64"]
FLOOR_REG = P["floor_per_cta_kstep_cycles"]
TM.set_model("mixtral-8x7b-tp8")

def price(G, n, g, sigma):
    win = TM.window("native", TM.E, G, n, g, TM.window_width(k_w, SMS, occ[g]))
    out = {}
    for law, flags in LAWS.items():
        for k, v in flags.items(): setattr(TM, k, v)
        ms = TM.gemm_ms(win, g, sigma, c, bw, SMS, occ=occ[g]) + TM.dead_ms(TM.E, n, g, c, k_w, occ[g])
        out[law] = ms * MHZ * 1e3
    for k, v in LAWS["cores"].items(): setattr(TM, k, v)
    return out, win.reads

res = {"registration": "docs/registered/README.md 2026-09-30 tp8 floor; 2026-09-30-mixtral-8x7b-tp8-floor-gh200.json",
       "captures": {}}
lines = []
d = TREE / "vm/results/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor-r3-counters"
for name in ["r3f-g64", "r3f-g8", "r3f-g64-lock1710", "r3f-g8-lock1710"]:
    j = json.loads((d / f"{name}.json").read_text())
    G = j["plan"]["group_m"]
    gates = [(x.get("number") or x.get("id") or x.get("name"), x.get("verdict") or x.get("status")) for x in j.get("gates", [])]
    cap = {"file": f"{d.name}/{name}.json", "G": G, "clock": j["clock"]["control"], "gates": gates, "cells": []}
    meas = {}
    for cl in j["cells"]:
        n = int(cl["n"])
        for g in ("w1", "w2"):
            pg = cl["per_gemm"][g]
            cyc = float(pg["sm__cycles_elapsed.avg"]); byt = float(pg["dram__bytes_read.sum"])
            win = TM.window("native", TM.E, G, n, g, TM.window_width(k_w, SMS, occ[g]))
            sigma = byt / win.reads
            rp, _ = price(G, n, g, sigma)
            rg = regcell[(G, n, g)]
            row = dict(n=n, gemm=g, cycles=cyc, mhz=pg.get("sm_clock_mhz"), dram_read=byt, sigma=sigma,
                       grid=int(pg["launch__grid_size"]), waves=rg["waves"],
                       reg_cores=rg["cores"], repriced={k: round(v) for k, v in rp.items()},
                       dev_reg=cyc / rg["cores"] - 1, dev_repriced=cyc / rp["cores"] - 1)
            cap["cells"].append(row); meas[(n, g)] = row
    # F1, F2
    Dm = meas[(4, "w1")]["cycles"] - meas[(2, "w1")]["cycles"]
    Drep = {law: meas[(4, "w1")]["repriced"][law] - meas[(2, "w1")]["repriced"][law] for law in LAWS}
    cap["F1"] = dict(D=Dm, D_units=Dm / UNIT, registered=D_REG, repriced=Drep,
                     dev_vs_cores=Dm - D_REG["cores"], dev_units=(Dm - D_REG["cores"]) / UNIT,
                     dev_vs_cores_repriced_units=(Dm - Drep["cores"]) / UNIT,
                     inside={law: abs(Dm - v) <= 0.5 * UNIT for law, v in D_REG.items()},
                     inside_repriced={law: abs(Dm - v) <= 0.5 * UNIT for law, v in Drep.items()},
                     verdict="HELD" if abs(Dm - D_REG["cores"]) <= 0.5 * UNIT else "FALSIFIED")
    m2 = meas[(2, "w1")]
    cap["F2"] = dict(cycles=m2["cycles"], reg=162474, dev=m2["cycles"] / 162474 - 1,
                     repriced=m2["repriced"], dev_repriced=m2["cycles"] / m2["repriced"]["cores"] - 1,
                     rivals_dev={law: m2["cycles"] / v - 1 for law, v in m2["repriced"].items()},
                     verdict="HELD" if -0.03 <= m2["cycles"] / 162474 - 1 <= 0.08 else "FALSIFIED")
    others = [r for r in cap["cells"] if not (r["n"] == 2 and r["gemm"] == "w1")]
    cap["F3"] = dict(outside_reg=[(r["n"], r["gemm"], round(r["dev_reg"] * 100, 2)) for r in others if abs(r["dev_reg"]) > 0.08],
                     outside_repriced=[(r["n"], r["gemm"], round(r["dev_repriced"] * 100, 2)) for r in others if abs(r["dev_repriced"]) > 0.08],
                     all_repriced=[(r["n"], r["gemm"], round(r["dev_repriced"] * 100, 2)) for r in others])
    cap["F3"]["verdict"] = "FALSIFIED" if cap["F3"]["outside_repriced"] else "HELD"
    f4 = {}
    for g, ns in (("w1", [6, 8, 10]), ("w2", [5, 6, 8, 10])):
        S = TM.GEOMETRY[g].ksteps
        xs = [meas[(n, g)]["grid"] * S / SMS for n in ns]; ys = [meas[(n, g)]["cycles"] for n in ns]
        s = float(np.polyfit(xs, ys, 1)[0])
        xa = [meas[(n, g)]["grid"] * S / SMS for n in sorted({r["n"] for r in cap["cells"]})]
        ya = [meas[(n, g)]["cycles"] for n in sorted({r["n"] for r in cap["cells"]})]
        f4[g] = dict(cells=ns, slope=s, reg=FLOOR_REG[g], dev=s / FLOOR_REG[g] - 1,
                     all_cell_slope=float(np.polyfit(xa, ya, 1)[0]),
                     verdict="HELD" if abs(s / FLOOR_REG[g] - 1) <= 0.02 else "FALSIFIED")
    cap["F4"] = f4
    res["captures"][name] = cap
    lines.append(f"== {name} (G={G}, clock {cap['clock']}, gates {gates})")
    for r in cap["cells"]:
        lines.append(f"  n={r['n']:2d} {r['gemm']} waves {r['waves']:5.2f} cycles {r['cycles']:9.0f} reg {r['reg_cores']:7d} ({r['dev_reg']*100:+6.2f}%)"
                     f" sigma {r['sigma']:.3f} repriced {r['repriced']['cores']:7d} ({r['dev_repriced']*100:+6.2f}%)"
                     f" | old {r['repriced']['old_lifetime']} lead {r['repriced']['cores_per_lead']} thr {r['repriced']['throughput']}")
    F1 = cap["F1"]
    lines.append(f"  F1 D {F1['D']:.0f} ({F1['D_units']:.2f} u) vs {D_REG['cores']} dev {F1['dev_units']:+.2f} u -> {F1['verdict']};"
                 f" repriced D: " + ", ".join(f"{k} {v}" for k, v in Drep.items()) + f"; inside(reg) {F1['inside']}")
    F2 = cap["F2"]
    lines.append(f"  F2 w1 n=2 {F2['cycles']:.0f} vs 162474 {F2['dev']*100:+.2f}% (repriced cores {F2['repriced']['cores']}, {F2['dev_repriced']*100:+.2f}%) -> {F2['verdict']};"
                 f" rivals repriced " + ", ".join(f"{k} {v*100:+.1f}%" for k, v in F2["rivals_dev"].items()))
    lines.append(f"  F3 outside 8% (repriced): {cap['F3']['outside_repriced']} -> {cap['F3']['verdict']}; (registered sigma=1: {cap['F3']['outside_reg']})")
    for g, v in f4.items():
        lines.append(f"  F4 {g} n={v['cells']} slope {v['slope']:.1f} vs {v['reg']} {v['dev']*100:+.2f}% -> {v['verdict']} (all-cell {v['all_cell_slope']:.1f})")
OUT.joinpath("score.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
OUT.joinpath("score.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
