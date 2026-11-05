#!/usr/bin/env python
"""D5-7: DS's three weak readings. (a) CAL2 under each weak term removed or moved (CAL2 refitted; CAL2 only). (b) How
much of each new model's v5d price those terms carry, from their configs and registered page lists only (no page of
theirs exists; prices at the CAL2 refit's constants under v5d; nothing fitted).  python weak.py [--write | --check]"""
import dataclasses
import math
import sys

import d5b

D4, D5, V5D, RD = d5b.D4, d5b.D5, d5b.V5D, d5b.D5.RD
B, W = d5b.BASE, d5b.RD_
q32, w128 = W["DRAIN"]["tiles"]["32x64x128w4s3"], W["W1RATE"]["tiles"]["32x128x64w4s3"]


def regime(**edit):
    r = dict(B.regime)
    r.update(edit)
    return dataclasses.replace(B, regime=r)


def w1(c):
    s = dict(B.w1_scale)
    s["32x128x64w4s3"] = D4.V4D_GBPS / c
    return dataclasses.replace(B, w1_scale=s)


VAR = {"no ceiling anywhere": regime(**{t: (None, B.regime[t][1]) for t in B.regime}),
       "W1 32x128x64 at c - SE": w1(w128["counter_GBps"] - w128["se_counter_GBps"]),
       "W1 32x128x64 at c + SE": w1(w128["counter_GBps"] + w128["se_counter_GBps"]),
       "W1 32x128x64 off": dataclasses.replace(B, w1_scale={k: v for k, v in B.w1_scale.items() if k != "32x128x64w4s3"}),
       "no drain on 32x64x128": regime(**{"32x64x128w4s3": (B.regime["32x64x128w4s3"][0], None)}),
       "drain 32x64x128 at q + 2 SE": regime(**{"32x64x128w4s3": (B.regime["32x64x128w4s3"][0],
                                                                  1e-3 / (q32["q_us_per_byte"] + 2 * q32["se_q_us_per_byte"]))})}

if __name__ == "__main__":
    import decode_synth as SY

    from moe.spec import MODEL_CONFIGS as MC
    L = ["(a) CAL2 only, refitted each time"] + [d5b.line(d5b.run(t, k, blind=False)) for k, t in VAR.items()]
    c = d5b.cal(B)
    prm = D5.V4DP.V4D.params_from(c["FIT"]["forms"][c["FIT"]["selected"]], rho_form=c["FIT"]["selected"],
                                  tile=D4.V4.Params(**c["tile_params_dict"]))
    L.append("(b) new models, main cells: v5d's price change at the CAL2 refit's constants (no page of theirs)")

    def ms(m, cell, t):
        ids = D5.V4DP.ids_of(m, cell) if RD.geometry(MC[m], cell["T"], cell["counts"]).path == RD.NAIVE else None
        return V5D.price(m, cell["T"], cell["counts"], prm, ids=ids, terms=t)
    for m in D5.NEW_BLIND:
        cells = SY.page(m, "main")["cells"]
        base = [ms(m, x, B) for x in cells]
        al = [p for p in base if p["v5d"]["aligned"]]
        binds = sum(any(g["ceiling_binds"] for g in p["v5d"]["gemm"].values()) for p in al)
        sub = [g for p in al for g in p["v5d"]["gemm"].values() if g["n"] <= g["S"]]
        ratio = max(g["ceiling_ms"] / g["bw_ms"] for p in al for g in p["v5d"]["gemm"].values())
        L.append(f"  {m}: {len(cells)} cells; the ceiling binds on {binds}; single-wave GEMMs n "
                 + (f"{min(g['n'] for g in sub)} to {max(g['n'] for g in sub)}" if sub else "none")
                 + f"; largest b / r_cta over the GEMM at bandwidth {ratio:.2f}")
        for k, t in VAR.items():
            d = [(ms(m, x, t)["ms"] - b["ms"]) / b["ms"] for x, b in zip(cells, base, strict=True)]
            moved = sum(abs(v) > 1e-12 for v in d)
            L.append(f"    {k:30s} cells moved {moved:2d}  max |change| {max(map(abs, d)):.2%}  "
                     f"rms {math.sqrt(sum(v * v for v in d) / len(d)):.2%}")
    sys.exit(d5b.emit(L, "weak", sys.argv[1:]))
