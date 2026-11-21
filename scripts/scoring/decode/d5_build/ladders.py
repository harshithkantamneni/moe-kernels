#!/usr/bin/env python
"""The CAL clause's cause, read off DS's committed readings alone (LADDERS on 64x128x64w4s3): the excess of each
single-wave w2 row at A 4 over its A 8 row after bytes (X = L - max(Dc, F)), and the per-CTA rate b / (L - c0) at a
counter offset c0 equal to the median of the offsets DS's DRAIN fits read per mixed ladder (INFERENCE: those ladders'
offsets stand in for these, whose own offsets SUBWAVE's fit leaves free).
python ladders.py [--write | --check]"""
import statistics as st
import sys

import d5b

if __name__ == "__main__":
    W = d5b.RD_
    lad = W["LADDERS"]["tiles"]["64x128x64w4s3"]["ladders"]
    offs = [v for t in W["DRAIN"]["tiles"].values() for v in (t.get("offsets_us") or {}).values()]
    C0 = st.median(offs)
    L = [f"DRAIN ladder offsets over the four tiles: {len(offs)} ladders, median {st.median(offs):.2f} us, "
         f"range {min(offs):.2f} to {max(offs):.2f} us; c0 taken as that median"]
    for k in ("mixtral-8x7b/T128/w2", "phi-3.5-moe/T128/w2", "jetmoe-8b/T128/w2"):
        rows = {x["A"]: x for x in lad[k]}
        a4, a8 = rows[4], rows[8]
        L.append(f"{k}: A 4 X {a4['X_us']:.2f} us, A 8 X {a8['X_us']:.2f} us, A 4 over A 8 after bytes "
                 f"{a4['X_us'] - a8['X_us']:+.2f} us")
        for A in (4, 8):
            x = rows[A]
            L.append(f"  A {A}: n {x['n']} ({x['wave']}), b {x['b_bytes'] / 1e6:.3f} MB, L {x['L_us']:.2f} us, per-CTA rate "
                     f"b / (L - c0) {x['b_bytes'] / (x['L_us'] - C0) / 1e3:.1f} GB/s, aggregate n b / (L - c0) "
                     f"{x['n'] * x['b_bytes'] / (x['L_us'] - C0) / 1e6:.2f} TB/s")
    sw = W["SUBWAVE"]["tiles"]["64x128x64w4s3"]
    L.append("offset sensitivity: the same rows at SUBWAVE's own fitted per-ladder offsets on this tile")
    for k in ("mixtral-8x7b/T128/w2", "phi-3.5-moe/T128/w2", "jetmoe-8b/T128/w2"):
        c = sw["offsets_us"][k]
        for x in sorted(lad[k], key=lambda x: x["A"]):
            L.append(f"  {k} A {x['A']}: n {x['n']}, offset {c:.2f} us, per-CTA rate b / (L - c0) "
                     f"{x['b_bytes'] / (x['L_us'] - c) / 1e3:.1f} GB/s")
    L.append(f"SUBWAVE r_cta on this tile {sw['r_cta_GBps']:.1f} GB/s, binding rows {sw['binding_rows']}")
    sys.exit(d5b.emit(L, "ladders", sys.argv[1:]))
