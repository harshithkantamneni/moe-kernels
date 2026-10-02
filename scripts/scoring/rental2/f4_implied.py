#!/usr/bin/env python
"""The floor falsifier F4's estimator applied to the MODEL's own per-cell
predictions, beside the measured slope and the registered asymptote 344.1 + F / S
(docs/FINDINGS.md, rental 1, the correction note on F4). Diagnosis, offline.

    python scripts/scoring/rental2/f4_implied.py [--out FILE]

For each published GEMM F4 scored, the slope of cycles over grid x S / 132 on
the scored treads: of the CORES model's per-cell cycles (8x7B 2026-09-27 fit,
sigma 1), of the pure ceil(N / 132) floor, and of the base-clock G = 64 capture's
measured `sm__cycles_elapsed.avg`. Fixes design-r2's work/f4_implied.py, whose
session-tag match found no rental-1 capture and printed `nan` for tp8: the
measured cells are now read straight off the named session's base capture.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

PUB = ROOT / "results" / "published"
CASES = [("mixtral-8x7b", "2026-09-30-nvidia_gh200_480gb-mixtral8x7b-floor-session", {"w1": [2, 3, 4, 6, 7, 8], "w2": [6, 7, 8]}),
         ("jetmoe-8b", "2026-09-30-nvidia_gh200_480gb-jetmoe-floor-session", {"w1": [2, 3, 4, 6, 9, 10, 11], "w2": [9, 10, 11]}),
         ("granite-3.0-3b-a800m", "2026-09-30-nvidia_gh200_480gb-granite-session", {"w2": [3, 4, 6]}),
         ("phi-3.5-moe", "2026-09-29-nvidia_gh200_480gb-phi3.5-session", {"w1": [2, 3, 4, 6], "w2": [2, 3, 4, 6]}),
         ("qwen1.5-moe-a2.7b", "2026-09-29-nvidia_gh200_480gb-qwen1.5-session", {"w1": [2, 3, 4, 6], "w2": [2, 3, 4, 6]}),
         ("qwen2-57b-a14b", "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session", {"w1": [2, 3, 4, 6], "w2": [2, 3, 4, 6]}),
         ("olmoe-1b-7b", "2026-09-29-nvidia_gh200_480gb-olmoe-session", {"w1": [2, 3, 4, 6], "w2": [2, 3, 4, 6]}),
         ("mixtral-8x22b", "2026-09-28-nvidia_gh200_480gb-8x22b-session", {"w1": [2, 3, 4, 6], "w2": [3, 4, 6]}),
         ("mixtral-8x7b-tp8", "2026-10-01-nvidia_gh200_480gb-rental1-session", {"w1": [6, 8, 10], "w2": [5, 6, 8, 10]})]


def measured_cells(session_dir: Path, model: str, G: int = 64, stem: str = "r3f-g64") -> dict:
    """{(n, gemm): (grid, cycles)} off the session's one base-clock G capture of
    `model`, {} when there is none (two are refused: the match must be exact)."""
    hits = []
    for p in sorted(Path(session_dir).rglob(f"{stem}.json")):
        d = json.loads(p.read_text())
        if d.get("plan", {}).get("model") == model and int(d["plan"]["group_m"]) == G:
            hits.append(d)
    if len(hits) != 1:
        return {}
    out = {}
    for c in hits[0]["cells"]:
        for g in ("w1", "w2"):
            pg = c["per_gemm"][g]
            grid = (c.get("grid") or {}).get(g) or pg.get("launch__grid_size")
            out[(int(c["n"]), g)] = (int(grid), float(pg["sm__cycles_elapsed.avg"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    import cores_heldout_predict as CP
    import floor_estimator as FE
    import r3_timing_model as TM
    src = TM.build(TM.build_parser().parse_args([*map(str, CP.source_pages()), "--counters", str(CP.C27)]))
    params, kw = src["main"].params, src["main"].k_w
    rows, lines = [], []
    for model, sess, sel in CASES:
        ns_all = sorted({n for v in sel.values() for n in v})
        pr = CP.predict(model, ns_all, [64], params, kw)
        P = {(c["n"], c["gemm"]): c for c in pr["cells"]}
        M = measured_cells(PUB / sess, model)
        for g, ns in sel.items():
            S = pr["geometry"][g]["ksteps"]
            asym = pr["floor_per_cta_kstep_cycles"][g]
            model_cells = {n: (P[(n, g)]["live_ctas"] + P[(n, g)]["dead_ctas"], P[(n, g)]["cores"]) for n in ns}
            u = (S + TM.CTA_FIXED_KSTEPS[g]) * 344.1
            ceil_cells = {n: (model_cells[n][0], math.ceil(P[(n, g)]["live_ctas"] / 132) * u) for n in ns}
            imp = FE.implied_slope(model_cells, ns, S)
            flo = FE.implied_slope(ceil_cells, ns, S)
            ms = (FE.implied_slope({n: M[(n, g)] for n in ns}, ns, S)
                  if all((n, g) in M for n in ns) else None)
            row = {"model": model, "session": sess, "gemm": g, "treads": ns, "asymptote": asym,
                   "model_through_estimator": imp, "pure_ceil": flo, "measured": ms,
                   "model_vs_asymptote": imp / asym - 1,
                   "measured_vs_asymptote": None if ms is None else ms / asym - 1,
                   "measured_vs_model": None if ms is None else ms / imp - 1}
            rows.append(row)
            lines.append(f"{model:22s} {g} S{S:<4d} n={ns}: asymptote {asym:6.1f} | model through "
                         f"estimator {imp:6.1f} ({imp / asym - 1:+.2%}) pure-ceil {flo:6.1f} | measured "
                         + ("none" if ms is None else
                            f"{ms:6.1f}: vs asymptote {ms / asym - 1:+.2%}, vs model {ms / imp - 1:+.2%}"))
    print("\n".join(lines))
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
