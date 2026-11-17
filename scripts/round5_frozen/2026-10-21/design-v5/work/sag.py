"""F1 evidence: the per-kernel clock sag off = t_ncu - cycles / L, fitted as A (1 - exp(-t / tau)) per page,
per tile, per lock and per board (A2 board 0f62f9aa, S1 board), plus R3's tile (rental 3). DESCRIPTIVE (SEEN).

Same estimator as diag-A2 work/transient.py and read_s1.sag_fit (tau on a 50..4975 us grid step 25, A least
squares), so A2's 128x128 pooled 50.1 us / 750 us and S1's OLMoE f128 numbers reproduce. Writes sag.json.
"""
import json
import math
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
A2 = Path.home() / "moe-kernels-exfil/runs/gh200-tile2-20261020T2242Z/tree/vm/results"
S1 = Path.home() / "moe-kernels-exfil/runs/gh200-tile1-20261021T0704Z/tree/vm/results"
R3 = Path.home() / "code/moe-kernels/results/published/2026-10-06-nvidia_gh200_480gb-rental3-session/results"
TA = Path.home() / "code/moe-kernels/results/published/2026-10-20-nvidia_gh200_480gb-tileA-session/results"


def pts_of(files, lock):
    out = []
    for f in files:
        p = json.loads(Path(f).read_text())
        for c in p["cells"]:
            for g in ("w1", "w2"):
                r = (c.get("recorded") or {}).get(g) or {}
                pg = dict((c.get("per_gemm") or {}).get(g) or {})
                if "sm__cycles_elapsed.avg" in pg:
                    cyc, t = pg["sm__cycles_elapsed.avg"], pg["gpu__time_duration.sum"]
                else:
                    cyc, t = r.get("sm__cycles_elapsed.avg"), pg.get("gpu_time_ns")
                if cyc and t:
                    t = float(t) / 1e3
                    out.append((t, t - float(cyc) / lock))
    return out


def fit(p):
    t = np.array([x[0] for x in p])
    off = np.array([x[1] for x in p])
    best = None
    for tau in np.arange(50, 5000, 25):
        X = 1 - np.exp(-t / tau)
        A = float(X @ off / (X @ X))
        s = float(np.sqrt(np.mean((off - A * X) ** 2)))
        if best is None or s < best[2]:
            best = (A, float(tau), s)
    # linear t = a + cycles / f form, the CONV-type reading: slope of off against t
    Xl = np.vstack([np.ones_like(t), t]).T
    a, b = np.linalg.lstsq(Xl, off, rcond=None)[0]
    return {"cells": len(p), "A_us": round(best[0], 2), "tau_us": best[1], "rms_us": round(best[2], 2),
            "lin_a_us": round(float(a), 2), "lin_slope": round(float(b), 5),
            "t_range_us": [round(float(t.min()), 1), round(float(t.max()), 1)],
            "A_over_tau": round(best[0] / best[1], 4)}


def g(root, pat):
    return sorted(root.glob(pat))


SRC = {
    # A2 board, 1410 (A2's own E1-CAL board)
    "A2 128x128 all pages 1410": (g(A2, "*f128*/lock1410/r3c-g*.json") + g(A2, "*lp1410f*/lock1410/r3c-g*.json") + g(A2, "*b128*/lock1410/r3c-g*.json"), 1410.0),
    "A2 128x128 OLMoE f128 1410": (g(A2, "*olmoe-1b-7b-f128*/lock1410/r3c-g*.json"), 1410.0),
    "A2 128x128 Mixtral f128 1410": (g(A2, "*mixtral-8x7b-f128*/lock1410/r3c-g*.json"), 1410.0),
    "A2 128x128 byte pages 1410": (g(A2, "*b128*/lock1410/r3c-g*.json"), 1410.0),
    "A2 64x128w8 all 1410": (g(A2, "*f64w8*/lock1410/r3c-g*.json") + g(A2, "*b64g1*/lock1410/r3c-g*.json"), 1410.0),
    "A2 64x128w8 OLMoE f64w8 1410": (g(A2, "*olmoe-1b-7b-f64w8*/lock1410/r3c-g*.json"), 1410.0),
    # S1 board
    "S1 128x128 OLMoE f128 1410": (g(S1, "*olmoe-1b-7b-f1410*/lock1410/r3c-g*.json"), 1410.0),
    "S1 128x128 OLMoE f128 1305": (g(S1, "*olmoe-1b-7b-f1305*/lock1305/r3c-g*.json"), 1305.0),
    "S1 128x128 OLMoE f128 1005": (g(S1, "*olmoe-1b-7b-f1005*/lock1005/r3c-g*.json"), 1005.0),
    "S1 128x128 G1/G16 byte pages 1410 (kr, st, b128g1, p128g16)": (
        g(S1, "*-kr-r3-counters/lock1410/r3c-g*.json") + g(S1, "*-st-r3-counters/lock1410/r3c-g*.json")
        + g(S1, "*b128g1*/lock1410/r3c-g*.json") + g(S1, "*p128g16*/lock1410/r3c-g*.json"), 1410.0),
    "S1 64x128w8 p64g1 1410": (g(S1, "*p64g1*/lock1410/r3c-g*.json"), 1410.0),
    # R3's tile, rental 3
    "R3 32x64 rental 3 1005": (g(R3, "*floor1005*/r3f-*.json"), 1005.0),
    "R3 32x64 rental 3 1410": (g(R3, "*tp4-floor1410*/r3f-*.json"), 1410.0),
    "R3 32x64 rental 3 1710": (g(R3, "*tp4-floorrep*/r3f-*.json"), 1710.0),
}


def main():
    res = {}
    for name, (files, lock) in SRC.items():
        p = pts_of(files, lock)
        if not p:
            res[name] = {"cells": 0, "files": len(files)}
            print(f"{name:62s} no cells ({len(files)} files)")
            continue
        r = fit(p)
        r["files"] = [str(f).split("/results/")[-1] for f in files]
        res[name] = r
        print(f"{name:62s} n {r['cells']:4d} A {r['A_us']:6.2f} us tau {r['tau_us']:6.0f} us rms {r['rms_us']:5.2f} "
              f"A/tau {r['A_over_tau']:.4f} | lin a {r['lin_a_us']:6.2f} slope {r['lin_slope']:+.5f} | t {r['t_range_us']}")
    (H / "sag.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
