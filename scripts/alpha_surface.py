"""Read every block_m_crossing report under a root and print the alpha surface.

WHY THIS EXISTS. The 2026-09-01 session produced alpha from four routes and
reconciled the wrong two: it scored the PUBLISHED 0.558 against an ablation whose
own gates had failed, and never looked at the sweep's per-BLOCK_M fits -- the only
alpha values measured on that card. This reads the reports themselves, so the
comparison is over what was measured rather than over what was quoted.

The surface is alpha against the two things that set REUSE DISTANCE: the swizzle
width, which decides how many M-tiles share one weight read, and the per-expert
footprint, which decides how much else evicts it in between. Neither is "does the
expert fit in L2": qwen2's 37 MB expert fits in 60 MiB and still pays alpha 0.71.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

#: Below this many memory-bound treads the fit is not a measurement. The sweep
#: itself refuses a verdict under 3 and says so; this refuses to TABLE it, so a
#: blank cell means "not identifiable here" rather than a number nobody checked.
MIN_TREADS = 3


def reports(root: Path):
    for p in sorted(root.rglob("report.json")):
        try:
            yield p, json.loads(p.read_text())
        except Exception as exc:                      # noqa: BLE001
            print(f"  UNREADABLE {p}: {type(exc).__name__}", file=sys.stderr)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows = []
    for path, r in reports(root):
        fixed = r.get("fixed", {})
        for bm, lad in sorted(r.get("ladder", {}).items(), key=lambda kv: int(kv[0])):
            rows.append({
                "model": r.get("model", "?"),
                "g": fixed.get("GROUP_SIZE_M"),
                "bn": fixed.get("BLOCK_SIZE_N"),
                "bm": int(bm),
                "treads": lad.get("memory_points") or 0,
                "alpha": lad.get("alpha"),
                "corr": lad.get("alpha_corrected"),
                "hi": lad.get("alpha_upper"),
                "err": lad.get("mean_rel_err"),
                "path": path,
            })
    if not rows:
        print(f"no report.json under {root}")
        return 1

    print("ALPHA SURFACE -- every fit found under", root)
    print()
    print("alpha is the fraction of a weight re-read that MISSES L2. 1.0 means an")
    print("extra M-tile costs a full re-read; 0.0 means L2 absorbs it entirely.")
    print("A blank alpha means the fit was not identifiable at that BLOCK_M --")
    print(f"fewer than {MIN_TREADS} memory-bound treads -- not that it is zero.")
    print()
    print(f"  {'model':<18} {'G':>4} {'BN':>4} {'BM':>4} {'treads':>7} "
          f"{'alpha':>7} {'corrected':>10} {'upper':>7} {'fit err':>8}")
    print(f"  {'-'*18} {'-'*4} {'-'*4} {'-'*4} {'-'*7} {'-'*7} {'-'*10} "
          f"{'-'*7} {'-'*8}")
    for x in rows:
        ident = (x["treads"] or 0) >= MIN_TREADS and x["alpha"] is not None
        a = f"{x['alpha']:.3f}" if ident else "--"
        c = f"{x['corr']:.3f}" if ident and x["corr"] is not None else "--"
        h = f"{x['hi']:.3f}" if ident and x["hi"] is not None else "--"
        e = f"{x['err']*100:.2f}%" if ident and x["err"] is not None else "--"
        print(f"  {x['model']:<18} {str(x['g']):>4} {str(x['bn']):>4} "
              f"{x['bm']:>4} {x['treads']:>7} {a:>7} {c:>10} {h:>7} {e:>8}")

    ident = [x for x in rows if (x["treads"] or 0) >= MIN_TREADS
             and x["alpha"] is not None]
    print()
    print(f"{len(ident)} identifiable fit(s) of {len(rows)} block sizes swept.")
    if not ident:
        return 0

    # The two levers, each collapsed over the other, so a trend is visible even
    # when the grid is ragged. Medians rather than means: a single unidentified
    # BLOCK_M dropping out should not move the summary.
    def med(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

    # PAIRED AT MATCHED LEVELS, never pooled. Averaging one lever over the others
    # is only valid on a balanced grid and this one is ragged: the 2026-09-01
    # H200 run had ten identifiable mixtral fits against seven for qwen2, spread
    # differently over GROUP_SIZE_M, and the pooled medians came out 0.731 and
    # 0.713 -- a footprint effect of 0.018, which is nothing. At matched
    # (G=1, BN=64) the same data give 0.98 and 0.72. The pooling did not measure
    # a small effect, it averaged a large one away, and it would have been read
    # as evidence AGAINST the footprint mechanism.
    others = {"g": ("model", "bn", "bm"), "model": ("g", "bn", "bm"),
              "bn": ("model", "g", "bm")}
    for lever, label in (("g", "GROUP_SIZE_M, the swizzle width"),
                         ("model", "model, i.e. per-expert footprint"),
                         ("bn", "BLOCK_SIZE_N, the activation-confound control")):
        vals = sorted({x[lever] for x in ident}, key=str)
        if len(vals) < 2:
            continue
        keys = others[lever]
        cells: dict = {}
        for x in ident:
            cells.setdefault(tuple(x[k] for k in keys), {})[x[lever]] = x
        # A cell only counts when it holds EVERY level of the lever, so the
        # comparison is within one shape at one tile rather than across shapes.
        full = [c for c in cells.values() if set(c) == set(vals)]
        print()
        print(f"alpha against {label}:")
        if not full:
            print(f"  no {'/'.join(keys)} cell holds every level of this lever, "
                  "so it cannot be compared without pooling. Not reported.")
            for v in vals:
                sel = [x["alpha"] for x in ident if x[lever] == v]
                print(f"    (unpaired: {v} n={len(sel)} median "
                      f"{med(sel):.3f} -- NOT a comparison)")
            continue
        print(f"  {len(full)} cell(s) of ({', '.join(keys)}) hold every level")
        for v in vals:
            sel = [c[v]["alpha"] for c in full]
            cor = [c[v]["corr"] for c in full if c[v]["corr"] is not None]
            print(f"  {str(v):<20} n={len(sel):<3} median alpha {med(sel):.3f}"
                  + (f"   corrected {med(cor):.3f}" if cor else ""))
        lo, hi = vals[0], vals[-1]
        deltas = [c[hi]["alpha"] - c[lo]["alpha"] for c in full]
        print(f"  paired change {lo} -> {hi}: median {med(deltas):+.3f}"
              f"   (every cell moves the same way: "
              f"{'yes' if len({d > 0 for d in deltas}) == 1 else 'NO'})")

    print()
    print("READ IT AGAINST THE THREE CANDIDATES:")
    for name, cand in (("this repo, retracted", 0.100),
                       ("TEMPO arXiv:2608.13057", 0.330),
                       ("the 2026-09-01 pooled refit", 0.558)):
        inside = [x for x in ident if abs(x["alpha"] - cand) < 0.05]
        print(f"  alpha = {cand:.3f} ({name}): "
              f"{len(inside)} of {len(ident)} fits within 0.05")
    print()
    print("A pooled scalar cannot describe this surface if the levers move it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
