#!/usr/bin/env python
"""Registered per-GEMM cycles for a NATIVE floor capture of a model no page has
measured, under the partial wave's co-residency law and its rivals.

    python scripts/cores_heldout_predict.py --model mixtral-8x7b-tp8 \\
        --treads 1 2 3 4 5 6 8 10 --group-m 64 8 [--out reg.json]

The timing parameters are 8x7B's 2026-09-27 GH200 fit on the cross-model
source pages (tests/test_cross_model_score.py's SRC), nothing fitted on the
target. Each GEMM of each cell is priced at sigma = 1 (the launch order's own
reads) with the dead CTAs, in sm cycles at the 1710 MHz lock, under four
laws: CORES (the default), the old lifetime (`--no-cores`), the per-fetching-
CTA form (`--cores-per-lead`) and throughput (`--no-tail`). The score reprices
every cell with the capture's own bytes; this registers the numbers before it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import r3_timing_model as TM  # noqa: E402

PUB = ROOT / "results" / "published"
S27 = PUB / "2026-09-27-nvidia_gh200_480gb-session"
C27 = S27 / "results" / "2026-09-27-nvidia_gh200_480gb-r3-counters" / "lock1710"
SRC = [("deep", 2), ("p5", 3), ("p2", 32), ("deep", 4), ("p2", 8)]
SMS, MHZ = 132, 1710.0
LAWS = {"cores": dict(TAIL=True, CORES=True, CORES_PER_LEAD=False),
        "old_lifetime": dict(TAIL=True, CORES=False, CORES_PER_LEAD=False),
        "cores_per_lead": dict(TAIL=True, CORES=True, CORES_PER_LEAD=True),
        "throughput": dict(TAIL=False, CORES=True, CORES_PER_LEAD=False)}


def source_pages():
    out = []
    for rep in sorted(S27.rglob("private_weight_reference/*/report.json")):
        j = json.loads(rep.read_text())
        if any(j["session_tag"].endswith(f"-{t}-lock1710") and j["pinned"]["GROUP_SIZE_M"] == g
               for t, g in SRC):
            out.append(rep.parent)
    return out


def predict(model: str, treads, gs, params: dict, k_w: float) -> dict:
    c, bw = params["c"], params["bw"]
    occ = dict(TM.ASSUMED_OCCUPANCY["9.0"])
    old = TM.set_model(model)
    saved = {k: getattr(TM, k) for k in ("TAIL", "CORES", "CORES_PER_LEAD")}
    cells = []
    try:
        for G in gs:
            for n in treads:
                for g in TM.GEMMS:
                    geo = TM.GEOMETRY[g]
                    N = TM.live_rows(n) * geo.npn
                    slots = SMS * occ[g]
                    rem = N % slots if N >= slots else N
                    k = -(-rem // SMS) if rem else 0
                    r, _ = TM.schedule("native", TM.E, G, n, g)
                    row = {"G": G, "n": n, "gemm": g, "live_ctas": N,
                           "dead_ctas": TM.dead_ctas(TM.E, n, g), "waves": round(N / slots, 3),
                           "partial_wave_ctas": rem,
                           "partial_wave_k_sm": [k, (rem - (k - 1) * SMS) if rem else 0],
                           "partial_wave_fetching_ctas": int((r[N - rem:] > geo.slab / 2).sum()),
                           "in_domain": N < 2 * slots,
                           "unit_cycles": round(TM.floor_ksteps(g) * c * 1e-3 * MHZ, 1)}
                    win = TM.window("native", TM.E, G, n, g, TM.window_width(k_w, SMS, occ[g]))
                    for law, flags in LAWS.items():
                        for key, v in flags.items():
                            setattr(TM, key, v)
                        ms = (TM.gemm_ms(win, g, 1.0, c, bw, SMS, occ=occ[g])
                              + TM.dead_ms(TM.E, n, g, c, k_w, occ[g]))
                        row[law] = round(ms * MHZ * 1e3)
                    cells.append(row)
        geometry = {g: {"K": TM.GEOMETRY[g].K, "N": TM.GEOMETRY[g].N, "npn": TM.GEOMETRY[g].npn,
                        "ksteps": TM.GEOMETRY[g].ksteps} for g in TM.GEMMS}
        floor = {g: round(344.1 + TM.CTA_FIXED_KSTEPS[g] * 344.1 / TM.GEOMETRY[g].ksteps, 1)
                 for g in TM.GEMMS}
    finally:
        for key, v in saved.items():
            setattr(TM, key, v)
        TM.set_model(old)
    return {"model": model, "arm": "native", "clock_mhz": MHZ, "sms": SMS, "occupancy": occ,
            "source_fit": params, "k_w": k_w, "p_knee": TM.P_KNEE,
            "cores_rho_gbps": TM.CORES_RHO_GBPS, "cta_fixed_ksteps": dict(TM.CTA_FIXED_KSTEPS),
            "bytes": "sigma = 1: the launch order's own reads",
            "floor_per_cta_kstep_cycles": floor, "geometry": geometry, "cells": cells}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--treads", type=int, nargs="+", required=True)
    ap.add_argument("--group-m", type=int, nargs="+", default=[64])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    src = TM.build(TM.build_parser().parse_args(
        [*map(str, source_pages()), "--counters", str(C27)]))
    d = predict(a.model, a.treads, a.group_m, src["main"].params, src["main"].k_w)
    for r in d["cells"]:
        print(f"G={r['G']:2d} n={r['n']:2d} {r['gemm']} live {r['live_ctas']:5d} "
              f"waves {r['waves']:5.2f} "
              f"partial {r['partial_wave_ctas']:4d} k_SM {r['partial_wave_k_sm']} | "
              + " ".join(f"{law} {r[law]:7d} ({r[law] / r['unit_cycles']:5.2f} u)" for law in LAWS))
    if a.out:
        a.out.write_text(json.dumps(d, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
