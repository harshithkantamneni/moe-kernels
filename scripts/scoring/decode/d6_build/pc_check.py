#!/usr/bin/env python
"""The law against PC's own block-A launches on 64x128x64w4s3 (SEEN; the L table and kappa come from these pages, so
the single-wave class is a consistency check; the multi-wave and the short-K w1 launches set no constant): predicted span
over the stamped span - 1 (d6_build/pc_spans.json, the median over valid calls), design-D6's v6d against the candidate
and the D6-18 alternatives.

    python scripts/scoring/decode/d6_build/pc_check.py [--write | --check]"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import d6v as L2  # noqa: E402

VAR = {"design-D6 v6d": L2.Law(**L2.DESIGN_D6).build(), "CANDIDATE": L2.variant(),
       "c0 const 4.70": L2.variant(c0_form="const"), "c0 multi last": L2.variant(c0_multi="last")}


def spans() -> dict:
    by = {}
    for r in L2.SPANS:
        if r["valid"] and r["unit"] == "sa" and r["tile"] == L2.T64:
            by.setdefault((r["cell"], r["gemm"]), []).append(r)
    return {k: dict(v[0], span_us=float(np.median([x["span_us"] for x in v]))) for k, v in by.items()}


def lines() -> list[str]:
    rs = []
    for (cell, g), r in sorted(spans().items()):
        n, b = r["live"], r["K"] * 128 * 2
        cls = ("single w2" if g == "w2" else "single w1 (short K)") if n <= 396 else f"multi {g}"
        rs.append(dict(cls=cls, cell=cell, g=g, n=n, K=r["K"], span=r["span_us"],
                       **{nm: lw.gemm_us(L2.T64, n, b) / r["span_us"] - 1 for nm, lw in VAR.items()}))
    L = ["PC block A on 64x128x64w4s3: rms / mean of predicted span over stamped span - 1; columns: " + "; ".join(VAR)]
    for c in sorted({r["cls"] for r in rs}):
        s = [r for r in rs if r["cls"] == c]
        L.append(f"  {c:20s} ({len(s):2d}) " + " | ".join(
            f"{np.sqrt(np.mean(np.square([r[n] for r in s]))):.2%}/{np.mean([r[n] for r in s]):+.2%}" for n in VAR))
    for r in rs:
        L.append(f"  {r['cls']:20s} {r['cell']:16s} {r['g']} n {r['n']:5d} K {r['K']:5d} span {r['span']:8.1f} | "
                 + " ".join(f"{r[n]:+.1%}" for n in VAR))
    return L


if __name__ == "__main__":
    sys.exit(L2.emit(lines(), "pc_check", sys.argv[1:]))
