#!/usr/bin/env python
"""Score another model's timed pages with one card's fitted timing parameters,
the target's OWN counted bytes standing in for the byte model.

    python scripts/cross_model_score.py --target olmoe-1b-7b \\
        --source-timed <run dir> [...] --source-counters <r3c-g*.json dir> \\
        --target-timed <run dir> [...] --target-counters <r3c-g*.json dir> [--out scores.json]

WHAT IT SEPARATES (2026-09-29). `scripts/cross_model_predict.py` registers a
target's call times from PREDICTED bytes, so a byte-model miss propagates into
time. This scores the timing model's structure alone: the source fit's
parameters (T0, c per CTA k-step, bw, s_small, s_block, at the knee P_KNEE,
k_w, with the partial-wave co-residency law (CORES), dead-CTA and per-CTA
fixed-cost terms) are held, and every target cell is priced from its own page's
counted DRAM bytes (sigma from the target's counter pages). Nothing is fitted
on the target: `r3_timing_model.build` is run on the target only to read its
cells, context and counted bytes; its fit is discarded.

OUTPUT. Per cell (arm, G, n): measured and predicted ms and the residual
(predicted / measured - 1); per set: SHARED and PRIVATE together (the
registered set), and each arm, G and tread alone: rms, worst, cells beyond 5%.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import r3_timing_model as TM  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

SCORED = ("shared", "private")
LIMIT = 0.05


def _flags(no_cta_fixed=False, no_cores=False, cores_per_lead=False):
    return [*(["--no-cta-fixed"] if no_cta_fixed else []),
            *(["--no-cores"] if no_cores else []),
            *(["--cores-per-lead"] if cores_per_lead else [])]


def _build(timed, counters, no_cta_fixed=False, no_cores=False, cores_per_lead=False):
    return TM.build(TM.build_parser().parse_args(
        [*map(str, timed), "--counters", str(counters),
         *_flags(no_cta_fixed, no_cores, cores_per_lead)]))


def _stats(v):
    v = list(v)
    if not v:
        return {"cells": 0, "rms": None, "worst": None, "beyond_5pct": 0}
    return {"cells": len(v), "rms": math.sqrt(sum(x * x for x in v) / len(v)),
            "worst": max(v, key=abs), "beyond_5pct": sum(abs(x) > LIMIT for x in v)}


def score(source_timed, source_counters, target, target_timed, target_counters,
          no_cta_fixed=False, no_cores=False, cores_per_lead=False, source=None) -> dict:
    """`source`, when given, is a `build` of the source pages under the same
    flags, reused instead of refitting them."""
    src = source if source is not None else _build(source_timed, source_counters, no_cta_fixed,
                                                   no_cores, cores_per_lead)
    fit = src["main"]
    old = TM.set_model(target)
    saved_lead = TM.CORES_PER_LEAD
    try:
        tgt = _build(target_timed, target_counters, no_cta_fixed, no_cores, cores_per_lead)
        ctx = tgt["ctx"]
        rows = []
        TM.CORES_PER_LEAD = bool(cores_per_lead)
        with TM.cta_fixed(not no_cta_fixed), TM.cores(not no_cores):
            for c in tgt["cells"]:
                p = TM.call_ms(fit.x, c, ctx, fit.k_w)
                rows.append({"arm": c.arm, "G": c.G, "n": c.n, "measured_ms": c.ms,
                             "predicted_ms": p, "resid": p / c.ms - 1.0})
    finally:
        TM.CORES_PER_LEAD = saved_lead
        TM.set_model(old)
    scored = [r for r in rows if r["arm"] in SCORED]
    sets = {"shared+private": _stats(r["resid"] for r in scored)}
    for key in ("arm", "G", "n"):
        for v in sorted({r[key] for r in rows}):
            sets[f"{key}={v}"] = _stats(r["resid"] for r in rows if r[key] == v)
    return {"tool": "scripts/cross_model_score.py", "target": target,
            "source_model": TM.MODEL, "clock_mhz": src["ctx"].clock_mhz,
            "timing_params": fit.params, "p_knee": TM.P_KNEE, "k_w": fit.k_w,
            "cta_fixed_ksteps": None if no_cta_fixed else dict(TM.CTA_FIXED_KSTEPS),
            "cores": None if no_cores else {"rho_gbps": TM.CORES_RHO_GBPS,
                                            "per_lead": bool(cores_per_lead)},
            "source_pages": [p.run for p in src["use"]],
            "target_pages": [p.run for p in tgt["use"]], "sets": sets, "cells": rows}


def lines(d: dict) -> list[str]:
    s = d["sets"]["shared+private"]
    out = [f"{d['target']} timed with {d['source_model']}'s fit and its own counted bytes "
           f"(nothing fitted on it), {d['clock_mhz']:.0f} MHz: SHARED+PRIVATE "
           f"{s['cells']} cells, rms {100 * s['rms']:.2f}%, worst {100 * s['worst']:+.2f}%, "
           f"{s['beyond_5pct']} beyond 5%"]
    for k, v in d["sets"].items():
        if k != "shared+private" and v["cells"]:
            out.append(f"  {k:<12} {v['cells']:>3} cells rms {100 * v['rms']:.2f}% "
                       f"worst {100 * v['worst']:+.2f}%")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True, choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--source-timed", nargs="+", type=Path, required=True)
    ap.add_argument("--source-counters", type=Path, required=True)
    ap.add_argument("--target-timed", nargs="+", type=Path, required=True)
    ap.add_argument("--target-counters", type=Path, required=True)
    ap.add_argument("--no-cta-fixed", action="store_true",
                    help="drop the per-CTA fixed cost (r3_timing_model's --no-cta-fixed)")
    ap.add_argument("--no-cores", action="store_true",
                    help="the old partial-wave lifetime (r3_timing_model's --no-cores)")
    ap.add_argument("--cores-per-lead", action="store_true",
                    help="the co-residency law's DRAM term per fetching CTA (--cores-per-lead)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    try:
        d = score(a.source_timed, a.source_counters, a.target, a.target_timed, a.target_counters,
                  a.no_cta_fixed, a.no_cores, a.cores_per_lead)
    except TM.Refused as exc:
        print(f"REFUSED: {exc}")
        return 2
    print("\n".join(lines(d)))
    if a.out:
        a.out.write_text(json.dumps(d, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
