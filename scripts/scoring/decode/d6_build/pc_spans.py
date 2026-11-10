#!/usr/bin/env python
"""PC's span reading (addendum 6; design-D6's work/pc_spans.py, committed): per (unit, cell, GEMM, call) of rental PC's
published stamp pages, the launch as it ran: the live CTAs, the span from the first start to the last end, the median,
max and min CTA life, all overhead-corrected the reader's way (life / (1 + o)). It reuses read_pc.stamp_unit, pinned
and unchanged, so every validity rule is PC's own. v6d's kappa on 64x128x64 (d6common.pc_kappa) is the span over the
median life on block A's w2 single-wave rungs. A NEW reading of PC's SEEN pages: PC's registered reader does not print
it (design-D6 section 10).

    python scripts/scoring/decode/d6_build/pc_spans.py [--write | --check]

--write writes pc_spans.json beside; --check recomputes it and compares (exit 1 on a difference). About 0.3 GB."""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for p in ("scripts/scoring/pc", "scripts/scoring/cal3", "scripts/scoring/cal2"):
    sys.path.insert(0, str(REPO / p))
import read_pc as RP  # noqa: E402

PC = REPO / "results/published/2026-11-06-nvidia_gh200_480gb-pc-session"
OUT = HERE / "pc_spans.json"
SLOTS = 396


def read() -> list[dict]:
    results, session = PC / "results", PC / "session"
    R = RP.reg()
    perturbs, regchecks = {}, {}
    for k in ("ptp", "ptc", "pt3"):
        d, w = RP.unit_dir(results, RP.OTHER_UNITS[k])
        perturbs[k] = {"dir": d, "why": w, "json": json.loads((d / "perturb.json").read_text()) if d else {}}
    for k in ("rcp", "rc3"):
        d, w = RP.unit_dir(results, RP.OTHER_UNITS[k])
        regchecks[k] = {"dir": d, "why": w}
    census, _ = RP.census_occupancy(results)
    out = []
    for label in ("sa", "sc", "sr"):
        u = RP.stamp_unit(results, session, label, R, perturbs, regchecks, census)
        for stem, c in sorted(u["cells"].items()):
            o_by = {g: ((c.get("over") or {}).get(g) or {}).get("median") for g in ("w1", "w2")}
            for g in ("w1", "w2"):
                for lr in c[g]:
                    if not lr.get("valid") and "s" not in lr:
                        continue
                    o = o_by[g] or 0.0
                    s, e, life = lr["s"], lr["e"], lr["lam"]
                    t0 = s.min()
                    out.append({"unit": label, "cell": stem, "gemm": g, "call": lr.get("call"),
                                "tile": "x".join(map(str, lr["tile"][:3])) + f"w{lr['tile'][3]}s{lr['tile'][4]}",
                                "K": lr["K"], "N": lr["N"], "live": int(lr["live"]), "valid": bool(lr.get("valid")),
                                "o": float(o), "span_us": float((e.max() - t0) / (1 + o) / 1e3),
                                "life_med_us": float(np.median(life) / 1e3), "life_max_us": float(life.max() / 1e3),
                                "life_min_us": float(life.min() / 1e3)})
    return out


def text(rows: list[dict]) -> str:
    return json.dumps(rows, indent=0, sort_keys=True) + "\n"


def main(argv) -> int:
    t = text(read())
    if "--write" in argv:
        OUT.write_text(t)
        print(f"wrote {OUT.name}")
        return 0
    if "--check" in argv:
        same = OUT.exists() and OUT.read_text() == t
        print(f"{OUT.name}: {'same' if same else 'DIFFERS'}")
        return 0 if same else 1
    sys.stdout.write(t)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
