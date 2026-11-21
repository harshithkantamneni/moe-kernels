#!/usr/bin/env python
"""The gate audit's refit of FL1 / V10's clock fit (moved from the session
scratchpad's gate-audit/refit.py into the repo, logic unchanged): for every
published lock page (r3c-g*.json byte pages and r3f-*lock*.json floor captures,
2026-10-02 excluded as in the audit unless --include-rental2; sessions published after
the audit, rental 3's of 2026-10-06 on, are outside its scope and excluded too), per GEMM the least-
squares fit duration = t0 + cycles / f over the page's cells, the fitted f and t0,
and the cells whose t0-corrected clock 1e3 cycles / (t - t0) reads above the lock
by more than one 15 MHz step.

    python scripts/paper/lock_refit.py [--out refit.json] [--include-rental2]

Prints one block per page; --out writes the rows as JSON (the audit's refit.json).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEP = 15.0
TMAX = 50000.0
TMIN = -1000.0
K = 3.0


def lockfit(points, lock):
    pts = [(lab, float(c), float(t)) for lab, c, t in points if c and t]
    mx = statistics.fmean(c for _, c, _ in pts)
    my = statistics.fmean(t for _, _, t in pts)
    b = sum((c - mx) * (t - my) for _, c, t in pts) / sum((c - mx) ** 2 for _, c, _ in pts)
    t0 = my - b * mx
    f = 1e3 / b
    off, cells = [], []
    if abs(f - lock) > STEP:
        off.append(f"fitted {f:.0f}")
    if not TMIN <= t0 <= TMAX:
        off.append(f"offset {t0 / 1e3:.1f}us")
    for lab, c, t in pts:
        cs = 1e3 * c / (lock - STEP) - 1e3 * c / lock
        r = (t - (t0 + 1e3 * c / lock)) / cs
        mhz = 1e3 * c / (t - t0) if t > t0 else float("nan")
        cells.append((lab, c, t, mhz, r))
        if abs(r) > K:
            off.append(f"{lab} {mhz:.0f}({r:+.1f}st)")
    return f, t0, off, cells


def rows(include_rental2: bool = False) -> list[dict]:
    root = str(ROOT / "results/published")
    out = []
    for p in sorted(glob.glob(root + "/*/results/**/r3c-g*.json", recursive=True)
                    + glob.glob(root + "/*/results/**/r3f-*lock*.json", recursive=True)):
        if ("2026-10-02" in p and not include_rental2) or "profiles" in p \
                or Path(p).relative_to(root).parts[0][:10] > "2026-10-02":
            continue
        d = json.load(open(p))
        lock = (d.get("ncu") or {}).get("lock_mhz") or (d.get("clock") or {}).get("lock_mhz")
        if not lock:
            continue
        floor = d.get("kind") == "floor"
        stored = [g["verdict"] for g in d["gates"] if (g.get("number") or g.get("name")) in ("V10", "FL1")]
        res = []
        for g in ("w1", "w2"):
            if floor:
                pts = [(f"n={c['n']}", c["per_gemm"][g].get("sm__cycles_elapsed.avg"),
                        c["per_gemm"][g].get("gpu__time_duration.sum"))
                       for c in d["cells"] if g in c["per_gemm"]]
            else:
                pts = [(f"{c['arm']}/{c['n']}",
                        ((c.get("recorded") or {}).get(g) or {}).get("sm__cycles_elapsed.avg"),
                        ((c.get("per_gemm") or {}).get(g) or {}).get("gpu_time_ns"))
                       for c in d["cells"]]
            f, t0, off, cells = lockfit(pts, lock)
            hi = [x for x in cells if x[3] > lock + STEP]
            res.append(dict(g=g, f=f, t0=t0 / 1e3, off=off, ncells=len(cells),
                            above=[(x[0], round(x[3]), round(x[2] / 1e3, 1), round(x[4], 1)) for x in hi],
                            minlen=min(x[2] for x in cells) / 1e3))
        out.append(dict(file=os.path.relpath(p, root), lock=lock, stored=stored,
                        refit_pass=not any(o["off"] for o in res), gemms=res))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--include-rental2", action="store_true")
    a = ap.parse_args(argv)
    rs = rows(a.include_rental2)
    if a.out:
        a.out.write_text(json.dumps(rs, indent=1) + "\n")
    for r in rs:
        s = r["file"].split("/")
        name = s[0][:10] + " " + s[0][30:] + " " + s[-3][-22:] + "/" + s[-1]
        print(f"{name:75s} stored={r['stored']} refit={'PASS' if r['refit_pass'] else 'FAIL'}")
        for o in r["gemms"]:
            print(f"    {o['g']} f={o['f']:.1f} t0={o['t0']:.1f}us shortest={o['minlen']:.1f}us "
                  f"off={o['off'][:5]}{'...' if len(o['off']) > 5 else ''} above_lock={o['above'][:4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
