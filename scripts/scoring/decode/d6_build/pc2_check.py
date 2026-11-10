#!/usr/bin/env python
"""The law against PC2's READ launches (SEEN measurement pages; computed before the freeze): predicted span over PC2's
t - 1, per launch class, for the candidate and the alternatives D6-18 to D6-22 name. Kappa (R3) and c0 at G 8 come from
these pages, so the classes holding them are consistency checks; the G 32 and G 256 short rungs (single rungs READ,
their ladders NOT IDENTIFIED), the multi-wave launches and the w1 launches set no constant.

    python scripts/scoring/decode/d6_build/pc2_check.py [--write | --check]"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import d6v as L2  # noqa: E402

d = L2.PC2
VAR = {"design-D6 v6d": L2.Law(**L2.DESIGN_D6).build(), "CANDIDATE": L2.variant(),
       "c0 const 4.70": L2.variant(c0_form="const"), "c0 const 2.40": L2.variant(c0_form="const_lo"),
       "c0 interp ln G": L2.variant(c0_form="interp_logG"), "c0 multi last": L2.variant(c0_multi="last"),
       "X16 DS ratio": L2.variant(x16="ds_ratio"), "kappa R1e on 16": L2.variant(kap="own+pcfill"),
       "kappa v6d (by m, held)": L2.variant(kap="v6d"), "kappa fill pooled": L2.variant(kap="fill_pooled"),
       "L16 PC2 own": L2.variant(L16="pc2"), "X32 PC2-R3": L2.variant(x32="pc2_r3")}


def rows() -> list[dict]:
    out = []
    for r in d["PC2-R1"]["R1a rate curve"]:
        if str(r["status"]).startswith("READ") and r.get("t_us") is not None:
            out.append(dict(cls="D16 " + ("short" if r["b"] < 1e6 else "long") + (" single" if r["G"] <= 396 else " multi"),
                            cell=r["cell"], tile=L2.T16, n=r["G"], b=r["b"], t=r["t_us"]))
    for c, v in d["PC2-R2"]["R2c SPAN at short slabs, 64x128x64"].items():
        if str(v["status"]).startswith("READ"):
            F = int(c.split("-F")[1]) if "-F" in c else None
            out.append(dict(cls=f"S64 {'short' if F else 'long'} G{v['live']}", cell=c, tile=L2.T64, n=v["live"],
                            b=128 * (F or 14336) * 2, t=v["t_us"]))
    for t in (L2.T32a, L2.T32b):
        for r in d["PC2-R3"][t]["rows"]:
            if str(r["status"]).startswith("READ") and r.get("t_us") is not None:
                out.append(dict(cls=f"W {'short' if r['b'] < 1e6 else 'long'} " + ("single" if r["G"] <= 528 else "multi"),
                                cell=r["cell"], tile=t, n=r["G"], b=r["b"], t=r["t_us"]))
    for fam, tile in (("S", L2.T64), ("D", L2.T16)):
        for r in d["PC2-R2"]["R2e w1 launches (short K = H slabs)"][fam]:
            if str(r["status"]).startswith("READ") and r.get("t_us") is not None:
                out.append(dict(cls=f"w1 {fam} " + ("single" if r["n"] <= 396 else "multi"), cell=r["cell"] + " w1",
                                tile=tile, n=r["n"], b=r["b"], t=r["t_us"]))
    for r in out:
        for nm, lw in VAR.items():
            r[nm] = lw.gemm_us(r["tile"], int(r["n"]), r["b"]) / r["t"] - 1
    return out


def lines() -> list[str]:
    rs = rows()
    names = list(VAR)
    L = ["PC2's READ launches: rms / mean of predicted span over t - 1, by class; columns: "
         + "; ".join(f"[{i}] {n}" for i, n in enumerate(names))]
    for c in sorted({r["cls"] for r in rs}) + ["ALL"]:
        s = [r for r in rs if c in ("ALL", r["cls"])]
        L.append(f"  {c:18s} ({len(s):2d}) " + " | ".join(
            f"{np.sqrt(np.mean(np.square([r[n] for r in s]))):.1%}/{np.mean([r[n] for r in s]):+.1%}" for n in names))
    L.append("per launch: [0] design-D6, [1] CANDIDATE")
    for r in rs:
        L.append(f"  {r['cls']:18s} {r['cell']:18s} n {int(r['n']):5d} b {r['b'] / 1e3:7.0f} KB t {r['t']:8.2f} us | "
                 f"{r[names[0]]:+.1%} {r[names[1]]:+.1%}")
    return L


if __name__ == "__main__":
    sys.exit(L2.emit(lines(), "pc2_check", sys.argv[1:]))
