#!/usr/bin/env python
"""DS's ladders under the law (SEEN DS pages; ncu frame, so one offset): design-D6's v6d against the candidate, the
pooled offset, each GEMM at its own offset, and each tile at the pooled offset.

    python scripts/scoring/decode/d6_build/ds_check.py [--write | --check]"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import d6v as L2  # noqa: E402

sys.path.insert(0, str(L2.REPO))
from moe.spec import MODEL_CONFIGS as MC  # noqa: E402


def lines() -> list[str]:
    LAD = L2.DS["LADDERS"]["tiles"]
    rows = []
    for t, v in LAD.items():
        for k, rs in v["ladders"].items():
            m, _T, g = k.split("/")
            _bm, bn, _bk, _w, _s = L2.tile_dims(t)
            K = MC[m].hidden_size if g == "w1" else MC[m].intermediate_size
            rows += [dict(x, tile=t, gemm=g, slab=bn * K * 2) for x in rs]
    L = np.array([r["L_us"] for r in rows])
    out = [f"{len(rows)} DS ladder rows (L_us, ncu)"]
    for name, lw in (("design-D6 v6d", L2.Law(**L2.DESIGN_D6).build()), ("CANDIDATE", L2.variant())):
        p = np.array([lw.gemm_us(r["tile"], r["n"], r["slab"], 0.0, r["S"]) for r in rows])
        off = float(np.median(L - p))
        e = (p + off - L) / L
        s = f"{name}: pooled offset {off:.2f} us, rms {np.sqrt(np.mean(e ** 2)):.2%}"
        for g in ("w1", "w2"):
            k = np.array([r["gemm"] == g for r in rows])
            o2 = float(np.median((L - p)[k]))
            s += f" | {g} at its own offset {o2:.2f}: {np.sqrt(np.mean(((p[k] + o2 - L[k]) / L[k]) ** 2)):.2%}"
        for t in LAD:
            k = np.array([r["tile"] == t for r in rows])
            s += f" | {t} {np.sqrt(np.mean(e[k] ** 2)):.1%}"
        out.append(s)
    return out


if __name__ == "__main__":
    sys.exit(L2.emit(lines(), "ds_check", sys.argv[1:]))
