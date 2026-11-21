"""F2: the floor/DRAM soft-max exponent p per wgmma tile, fitted in v4's OWN per-CTA form (TM.gemm_ms with TM.P_KNEE
= p, every other term v4's) at COUNTED bytes against the counted SM cycles x K (SEEN-FITTED, 1410 MHz).

DIAGNOSIS-A2 Q5.2 fitted p per GEMM on sums (smax of the GEMM's floor-only and bytes-only totals: p 5.25 at 128x128);
v4 applies smax per CTA (r3_timing_model.gemm_ms), so v5's p is refitted in the form v5 uses. Domain: NATIVE cells of
the G 1 and G 16 byte pages (the classes E1-CAL and E1-TILE score), A2 and S1, 1410 MHz. SHARED and the G 64 floor pages
printed beside (DESCRIPTIVE), and S1-M2's 1305 and 1005 pages read at their own lock (the lock dependence of p).
measured = sm__cycles_elapsed.avg x K, K = the GEMM's c_ns / c_counter(tile) (ns per counter cycle at the lock);
model = v4-oracle GEMM + dead at counted bytes (DIAGNOSIS-A2 Q1's gemm_cyc). Writes kneefit.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.home() / "code/moe-kernels-work/2026-10-21/diag-A2/work"))
import numpy as np  # noqa: E402

from common import A2, FN, RG, SCORE, TM, V4, price_parts  # noqa: E402
import r3_timing_model_v4L as V4L  # noqa: E402

H = Path(__file__).resolve().parent
A2T = Path.home() / "moe-kernels-exfil/runs/gh200-tile2-20261020T2242Z/tree/vm/results"
S1T = Path.home() / "moe-kernels-exfil/runs/gh200-tile1-20261021T0704Z/tree/vm/results"
PAGES = [  # (root, model, unit, lock, role)
    (A2T, "olmoe-1b-7b", "b128g1", 1410, "fit"), (A2T, "qwen2-57b-a14b", "b128g1", 1410, "fit"),
    (A2T, "mixtral-8x7b", "b128g16", 1410, "fit"), (S1T, "mixtral-8x7b", "b128g1", 1410, "fit"),
    (S1T, "mixtral-8x7b", "p128g16", 1410, "fit"), (S1T, "olmoe-1b-7b", "st", 1410, "fit"),
    (S1T, "qwen2-57b-a14b", "st", 1410, "fit"),
    (A2T, "mixtral-8x7b", "b64g1", 1410, "fit"), (S1T, "mixtral-8x7b", "p64g1", 1410, "fit"),
    (A2T, "olmoe-1b-7b", "f128", 1410, "g64"), (A2T, "mixtral-8x7b", "f128", 1410, "g64"),
    (S1T, "olmoe-1b-7b", "f1410", 1410, "g64"), (S1T, "olmoe-1b-7b", "f1305", 1305, "g64"),
    (S1T, "olmoe-1b-7b", "f1005", 1005, "g64"),
    (A2T, "olmoe-1b-7b", "f64w8", 1410, "g64"), (A2T, "mixtral-8x7b", "f64w8", 1410, "g64"),
]
PGRID = [float(x) for x in np.arange(2.0, 16.01, 0.25)] + [1e9]


def cells_of(root, model, unit, lock):
    f = sorted(root.glob(f"*-{model}-{unit}-r3-counters/lock{lock}/r3c-g*.json"))
    if len(f) != 1:
        return []
    p = json.loads(f[0].read_text())
    de = p["design"]
    seed = (de.get("histogram") or {}).get("shuffle_seed")
    doc = {"block_m": int(de["block_m"]), "group_m": int(de["group_m"]), "shuffle_seed": seed}
    out = []
    for c in p["cells"]:
        if c["arm"] == "private":
            continue
        lab = c.get("histogram") or "balanced"
        counts = None if lab == "balanced" else c.get("bincount")
        cyc = {}
        for g in ("w1", "w2"):
            pg, rec = c["per_gemm"][g], c.get("recorded", {}).get(g, {})
            cyc[g] = float(pg.get("sm__cycles_elapsed.avg") or rec["sm__cycles_elapsed.avg"])
        out.append({"model": model, "unit": unit, "lock": lock, "doc": doc, "arm": c["arm"],
                    "c": {"n": int(c["n"]), "counts": counts, "label": lab},
                    "reads": {g: float(c["per_gemm"][g]["dram_bytes_read"]) for g in ("w1", "w2")}, "cyc": cyc})
    return out


def main():
    rows = []
    sel = FN.selections(SCORE)
    # the CAL engine's v2 fit is built on the FIRST pricer() call and reads TM.P_KNEE (and TAIL, CORES) then:
    # build it here at v4's own constants, before any swap (a first call inside the p loop refits c and bw at that p)
    RG.pricer()
    assert TM.P_KNEE == 14.0
    for root, model, unit, lock, role in PAGES:
        cells = cells_of(root, model, unit, lock)
        if not cells:
            print("missing", model, unit, lock)
            continue
        with A2.scoring(A2.Rules.tileA2(lock)):
            prm = FN.params(sel, occupancy={**sel.get("occupancy", {}), **(sel.get("occupancy_by_model") or {}).get(model, {})})
            for r in cells:
                t = RG.PAGE_TILE[r["doc"]["block_m"]]
                out = {g: {} for g in ("w1", "w2")}
                keep = TM.P_KNEE
                try:
                    for p in PGRID:
                        TM.P_KNEE = p
                        pp = price_parts(model, r["arm"], r["c"], r["doc"], prm, key="CLAMP", reads=r["reads"])
                        for g in ("w1", "w2"):
                            out[g][p] = pp[g]["gemm"] + pp[g]["dead"]
                            K = pp[g]["c_ns"] / V4.c_counter(t, g, prm)       # ns per counter cycle at the lock
                            fl, by = pp[g]["floor_only"], pp[g]["bytes_only"]
                            out[g]["meas"] = r["cyc"][g] * K * 1e-6       # ms
                            out[g]["x"] = by / fl
                finally:
                    TM.P_KNEE = keep
                for g in ("w1", "w2"):
                    rows.append({"board": "A2" if root == A2T else "S1", "model": model, "unit": unit, "lock": lock, "role": role, "bm": r["doc"]["block_m"],
                                 "G": r["doc"]["group_m"], "arm": r["arm"], "n": r["c"]["n"], "label": r["c"]["label"],
                                 "gemm": g, "x": out[g]["x"], "meas_ms": out[g]["meas"],
                                 "pred": {str(p): out[g][p] for p in PGRID}})
    res = {}

    def best(sub):
        b = None
        for p in PGRID:
            e = [r["pred"][str(p)] / r["meas_ms"] - 1 for r in sub]
            rms = math.sqrt(sum(x * x for x in e) / len(e))
            if b is None or rms < b[1]:
                b = (p, rms, sum(e) / len(e))
        e14 = [r["pred"]["14.0"] / r["meas_ms"] - 1 for r in sub]
        return {"best_p": b[0], "rms_pct": 100 * b[1], "mean_pct": 100 * b[2], "gemms": len(sub),
                "p14_rms_pct": 100 * math.sqrt(sum(x * x for x in e14) / len(e14)), "p14_mean_pct": 100 * sum(e14) / len(e14),
                "x": [round(min(r["x"] for r in sub), 3), round(max(r["x"] for r in sub), 3)]}
    sets = {
        "128 fit (NATIVE, G1+G16, 1410)": lambda r: r["role"] == "fit" and r["bm"] == 128 and r["arm"] == "native",
        "128 fit, A2 pages only": lambda r: r["role"] == "fit" and r["bm"] == 128 and r["arm"] == "native" and r["board"] == "A2",
        "128 fit, S1 pages only": lambda r: r["role"] == "fit" and r["bm"] == 128 and r["arm"] == "native" and r["board"] == "S1",
        "128 SHARED (G1+G16, 1410)": lambda r: r["role"] == "fit" and r["bm"] == 128 and r["arm"] == "shared",
        "128 G1 only NATIVE": lambda r: r["role"] == "fit" and r["bm"] == 128 and r["arm"] == "native" and r["G"] == 1,
        "128 G16 only NATIVE": lambda r: r["role"] == "fit" and r["bm"] == 128 and r["arm"] == "native" and r["G"] == 16,
        "64 fit (NATIVE, G1, 1410)": lambda r: r["role"] == "fit" and r["bm"] == 64 and r["arm"] == "native",
        "128 G64 NATIVE A2 f128 1410": lambda r: r["role"] == "g64" and r["bm"] == 128 and r["arm"] == "native" and r["unit"] == "f128",
        "128 G64 NATIVE S1 f1410": lambda r: r["unit"] == "f1410" and r["arm"] == "native",
        "128 G64 NATIVE S1 f1305": lambda r: r["unit"] == "f1305" and r["arm"] == "native",
        "128 G64 NATIVE S1 f1005": lambda r: r["unit"] == "f1005" and r["arm"] == "native",
        "64 G64 NATIVE A2 f64w8": lambda r: r["unit"] == "f64w8" and r["arm"] == "native",
    }
    for name, f in sets.items():
        sub = [r for r in rows if f(r)]
        if sub:
            res[name] = best(sub)
            print(f"{name:34s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()})
    # the 128 fit's residual by x band at the best p
    p128 = res["128 fit (NATIVE, G1+G16, 1410)"]["best_p"]
    bands = ((0, 0.45), (0.45, 0.7), (0.7, 0.9), (0.9, 1.3))
    res["128 fit by x band at best p"] = {}
    for lo, hi in bands:
        sub = [r for r in rows if sets["128 fit (NATIVE, G1+G16, 1410)"](r) and lo <= r["x"] < hi]
        if sub:
            e = [r["pred"][str(p128)] / r["meas_ms"] - 1 for r in sub]
            e14 = [r["pred"]["14.0"] / r["meas_ms"] - 1 for r in sub]
            res["128 fit by x band at best p"][f"{lo}-{hi}"] = {"n": len(sub), "mean_pct": 100 * sum(e) / len(e), "p14_mean_pct": 100 * sum(e14) / len(e14)}
    print(json.dumps(res["128 fit by x band at best p"], indent=0))
    (H / "kneefit.json").write_text(json.dumps({"fits": res, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
