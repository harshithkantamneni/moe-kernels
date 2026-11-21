#!/usr/bin/env python
"""Score the cross-model registrations (docs/registered 2026-09-27 to 2026-09-30)
on their published pages: the quantities every session README printed after
"Scored on the laptop against the registered JSON, cell by cell", from a
committed script.

    python scripts/paper/score_registered.py <case> [--out-dir DIR]
    python scripts/paper/score_registered.py all --out-dir scripts/scoring/crossmodel

Cases: 8x22b, qwen2, olmoe, qwen1.5, phi3.5, jetmoe, granite, jetmoe-floor,
mixtral-floor. Reads only results/published/** and docs/registered/**; fits
nothing; writes <case>.registered.{json,txt}.

WHAT EACH QUANTITY IS (the registrations' own definitions, docs/registered/README.md):

time_predicted_bytes  the registered JSON's per-cell `ms` (priced from PREDICTED
    bytes) against the measured call time: each (arm, G, n) cell is the median
    ms_p50 over the session's VALID R3 pages at that G whose every cell reads the
    1710 MHz lock (`r3_timing_model.load_timed_page`, the label its loader gives).
    Set: SHARED and PRIVATE. rms and worst of predicted / measured - 1, cells
    beyond 5%. The 8x22B and Qwen2-57B time tests are this; for OLMoE and the four
    held-out models it is printed only (their time test is cross_model_score.py).
slope_g8  the G >= 8 SHARED slope over treads 2 to 6: np.polyfit of the measured
    SHARED cells' ms against n at n = 2..6, at G = 8 and G = 32.
floor  cycles per CTA k-step: `floor_estimator.implied_slope` (the slope of
    `sm__cycles_elapsed.avg` over grid x S / 132) on the floor captures
    r3f-g64.json (base clock) and r3f-g64-lock1710.json, over each GEMM's
    registered tread set.
bytes  measured q = `dram_counter_route.r3_q` on the lock-1710 counter pages
    against the JSON's registered q (MIX view; ILL-POSED cells carry none),
    rel error q_pred / q_meas - 1, per set.

The registered floor values and bands are typed below with the README line that
states them (the JSON files do not carry them).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dram_counter_route as DCR  # noqa: E402
import floor_estimator as FE  # noqa: E402
import r3_timing_model as TM  # noqa: E402

PUB = ROOT / "results/published"
REG = ROOT / "docs/registered"
LOCK = 1710.0
LIMIT = 0.05

# floor: {gemm: (registered value or None, band lo, band hi, treads)}; source cited per case.
CASES = {
    "8x22b": dict(reg="2026-09-27-mixtral-8x22b-gh200.json",
                  session="2026-09-28-nvidia_gh200_480gb-8x22b-session",
                  # D/README.md:24-28: w1 inside 340 to 365; slope 0.9270, band 0.908 to 0.946
                  floor={"w1": (None, 340.0, 365.0, (2, 3, 4, 6)), "w2": (None, None, None, (2, 3, 4, 6))},
                  slope=(0.9270, 0.908, 0.946), time_scored=True, rows=[10, 11, 12, 13, 14]),
    "qwen2": dict(reg="2026-09-28-qwen2-57b-a14b-gh200.json",
                  session="2026-09-28-nvidia_gh200_480gb-qwen2-57b-session",
                  # D/README.md:71-74
                  floor={"w1": (None, 340.0, 365.0, (2, 3, 4, 6)), "w2": (None, None, None, (2, 3, 4, 6))},
                  slope=(0.6837, 0.670, 0.697), time_scored=True, rows=[15, 16, 17, 18, 19]),
    "olmoe": dict(reg="2026-09-29-olmoe-1b-7b-gh200.json",
                  session="2026-09-29-nvidia_gh200_480gb-olmoe-session",
                  # D/README.md:136-140
                  floor={"w1": (None, 340.0, 365.0, (2, 3, 4, 6)), "w2": (None, None, None, (2, 3, 4, 6))},
                  slope=(0.1597, 0.1565, 0.1629), time_scored=False, rows=[21, 22, 23]),
    # D/README.md:187-192 (registered floor and slope), :199-203 (within 2%)
    "qwen1.5": dict(reg="2026-09-29-qwen1.5-moe-a2.7b-gh200.json",
                    session="2026-09-29-nvidia_gh200_480gb-qwen1.5-session",
                    floor={"w1": (360.4, None, None, (2, 3, 4, 6)), "w2": (388.6, None, None, (2, 3, 4, 6))},
                    slope=(0.2159, None, None), time_scored=False, rows=[26, 27, 28]),
    "phi3.5": dict(reg="2026-09-29-phi-3.5-moe-gh200.json",
                   session="2026-09-29-nvidia_gh200_480gb-phi3.5-session",
                   floor={"w1": (352.2, None, None, (2, 3, 4, 6)), "w2": (353.9, None, None, (2, 3, 4, 6))},
                   slope=((0.4898, 0.4896), None, None), time_scored=False, rows=[30, 31, 32]),
    "jetmoe": dict(reg="2026-09-29-jetmoe-8b-gh200.json",
                   session="2026-09-29-nvidia_gh200_480gb-jetmoe-session",
                   floor={"w1": (360.4, None, None, (2, 3, 4, 6)), "w2": (355.2, None, None, (2, 3, 4, 6))},
                   slope=(0.1076, None, None), time_scored=False, rows=[34, 35, 36]),
    # D/README.md:292-297: Granite w2 on n = 3, 4, 6 (the 4+-wave estimator); w1 NOT SCORABLE
    "granite": dict(reg="2026-09-29-granite-3.0-3b-a800m-gh200.json",
                    session="2026-09-30-nvidia_gh200_480gb-granite-session",
                    floor={"w1": (365.8, None, None, (2, 3, 4, 6)), "w2": (466.5, None, None, (3, 4, 6))},
                    floor_not_scorable=("w1",),
                    slope=(0.0440, None, None), time_scored=False, rows=[37, 38, 39, 40]),
    # D/README.md:311-316 (the floor-only sessions, 79c5034)
    "jetmoe-floor": dict(reg=None, model="jetmoe-8b",
                         session="2026-09-30-nvidia_gh200_480gb-jetmoe-floor-session",
                         floor={"w1": (360.4, None, None, (2, 3, 4, 6, 9, 10, 11)),
                                "w2": (355.2, None, None, (9, 10, 11))},
                         rows=[42]),
    "mixtral-floor": dict(reg=None, model="mixtral-8x7b",
                          session="2026-09-30-nvidia_gh200_480gb-mixtral8x7b-floor-session",
                          floor={"w1": (352.2, None, None, (2, 3, 4, 6, 7, 8)),
                                 "w2": (348.5, None, None, (6, 7, 8))},
                          rows=[43]),
}
FLOOR_TOL = 0.02  # D/README.md:199-201, :309


def _stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return {"cells": 0, "rms": None, "worst": None, "beyond_5pct": 0}
    return {"cells": len(v), "rms": math.sqrt(sum(x * x for x in v) / len(v)),
            "worst": max(v, key=abs), "beyond_5pct": sum(abs(x) > LIMIT for x in v)}


def _counters_dir(session: Path) -> Path:
    hits = sorted(session.glob("results/*-r3-counters"))
    if len(hits) != 1:
        raise SystemExit(f"{session}: expected one r3-counters directory, found {hits}")
    return hits[0]


def measured_time(session: Path, model: str) -> tuple[dict, list[dict]]:
    old = TM.set_model(model)
    try:
        pages = TM.discover_timed([session / "results"])
    finally:
        TM.set_model(old)
    use = [p for p in pages if p.label == "VALID" and p.locked and next(iter(p.clocks)) == LOCK]
    cells: dict = {}
    for p in use:
        for (arm, n), (ms, _clk, _dec) in p.rows.items():
            cells.setdefault((arm, p.G, n), []).append(ms)
    listing = [{"run": p.run, "G": p.G, "label": p.label,
                "clock": sorted(c for c in p.clocks if c is not None), "used": p in use}
               for p in pages]
    return {k: statistics.median(v) for k, v in cells.items()}, listing


def floor_slopes(session: Path, ksteps: dict, spec: dict) -> dict:
    out = {}
    cd = _counters_dir(session)
    for cap in ("r3f-g64.json", "r3f-g64-lock1710.json"):
        p = cd / cap
        if not p.exists():
            out[cap] = None
            continue
        page = json.loads(p.read_text())
        by_n = {int(c["n"]): c for c in page["cells"]}
        res = {}
        for g, (reg, lo, hi, treads) in spec.items():
            cells = {n: (by_n[n]["per_gemm"][g]["launch__grid_size"],
                         by_n[n]["per_gemm"][g]["sm__cycles_elapsed.avg"]) for n in by_n}
            s = FE.implied_slope(cells, treads, ksteps[g])
            allcell = FE.implied_slope(cells, sorted(by_n), ksteps[g])
            res[g] = {"treads": list(treads), "slope": s, "all_cell_slope": allcell,
                      "registered": reg, "band": [lo, hi] if lo is not None else None,
                      "rel": (s / reg - 1) if reg else None}
        out[cap] = res
    return out


def verdict_floor(r: dict, not_scorable: bool) -> str:
    if not_scorable:
        return "NOT SCORABLE"
    if r["band"]:
        return "HELD" if r["band"][0] <= r["slope"] <= r["band"][1] else "FALSIFIED"
    if r["registered"]:
        return "HELD" if abs(r["rel"]) <= FLOOR_TOL else "FALSIFIED"
    return "printed"


def score_case(name: str) -> dict:
    case = CASES[name]
    session = PUB / case["session"]
    reg = json.loads((REG / case["reg"]).read_text()) if case["reg"] else None
    model = reg["target"] if reg else case["model"]
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    ksteps = {"w1": cfg.hidden_size // 64, "w2": cfg.intermediate_size // 64}
    out = {"tool": "scripts/paper/score_registered.py", "case": name, "model": model,
           "registration": f"docs/registered/{case['reg']}" if case["reg"] else
           "docs/registered/README.md (2026-09-30 floor-only sessions)",
           "session": f"results/published/{case['session']}", "ledger_rows": case["rows"],
           "ksteps": ksteps}
    fl = floor_slopes(session, ksteps, case["floor"])
    ns = case.get("floor_not_scorable", ())
    for cap, res in fl.items():
        if res:
            for g, r in res.items():
                r["verdict"] = verdict_floor(r, g in ns)
    out["floor"] = fl
    if reg is None:
        return out
    meas, listing = measured_time(session, model)
    out["timed_pages"] = listing
    rows = []
    for c in reg["cells"]:
        k = (c["arm"], c["G"], c["n"])
        if k in meas and c.get("ms") is not None:
            rows.append({"arm": c["arm"], "G": c["G"], "n": c["n"], "measured_ms": meas[k],
                         "registered_ms": c["ms"], "resid": c["ms"] / meas[k] - 1})
    sp = [r["resid"] for r in rows if r["arm"] in ("shared", "private")]
    out["time_predicted_bytes"] = {"scored": case["time_scored"], "shared+private": _stats(sp),
                                   "cells": rows}
    slopes = {}
    for G in (8, 32):
        pts = [(n, meas[("shared", G, n)]) for n in range(2, 7) if ("shared", G, n) in meas]
        if len(pts) == 5:
            import numpy as np
            slopes[f"G={G}"] = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0])
    reg_slope, lo, hi = case["slope"]
    out["slope_g8"] = {"measured": slopes, "registered": reg_slope,
                       "band": [lo, hi] if lo is not None else None}
    cd = _counters_dir(session) / "lock1710"
    qrows = []
    for page in sorted(cd.glob("r3c-g*.json")):
        payload = json.loads(page.read_text())
        G = int(payload["design"]["group_m"])
        q = DCR.r3_q(payload)
        for c in reg["cells"]:
            if c["G"] != G:
                continue
            for g in ("w1", "w2"):
                qp = (c.get("q") or {}).get(g)
                qm = q.get(c["arm"], {}).get(g, {}).get(c["n"])
                if qp is None or qm is None:
                    continue
                qrows.append({"arm": c["arm"], "G": G, "n": c["n"], "gemm": g, "q_meas": qm,
                              "q_reg": qp, "rel": qp / qm - 1})

    def sel(f):
        return [r["rel"] for r in qrows if f(r)]
    named = sel(lambda r: r["arm"] in ("shared", "private") and (
        r["gemm"] == "w1" or r["arm"] == "private" or (r["G"] <= 16 and r["n"] <= 4)))
    out["bytes"] = {
        "w1 all arms": _stats(sel(lambda r: r["gemm"] == "w1")),
        "w1 shared+private": _stats(sel(lambda r: r["arm"] != "native" and r["gemm"] == "w1")),
        "private w1": _stats(sel(lambda r: r["arm"] == "private" and r["gemm"] == "w1")),
        "private w2": _stats(sel(lambda r: r["arm"] == "private" and r["gemm"] == "w2")),
        "shared w1": _stats(sel(lambda r: r["arm"] == "shared" and r["gemm"] == "w1")),
        "shared w2": _stats(sel(lambda r: r["arm"] == "shared" and r["gemm"] == "w2")),
        "shared+native w2 G<=16 n<=4": _stats(sel(lambda r: r["arm"] != "private"
                                                  and r["gemm"] == "w2" and r["G"] <= 16
                                                  and r["n"] <= 4)),
        "named set (w1 S+P, PRIVATE w2, SHARED w2 G<=16 n<=4)": _stats(named),
        "cells": qrows}
    return out


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def lines(d: dict) -> list[str]:
    out = [f"{d['case']}: {d['model']}, {d['registration']}, {d['session']} "
           f"(ledger rows {d['ledger_rows']})"]
    for cap, res in d["floor"].items():
        if not res:
            out.append(f"  floor {cap}: missing")
            continue
        for g, r in res.items():
            ref = (f"registered {r['registered']} ({100 * r['rel']:+.2f}%)" if r["registered"]
                   else f"band {r['band']}" if r["band"] else "printed")
            out.append(f"  floor {cap:24s} {g} n={r['treads']} {r['slope']:.1f} {ref} "
                       f"all-cell {r['all_cell_slope']:.1f} {r['verdict']}")
    if "time_predicted_bytes" in d:
        t = d["time_predicted_bytes"]["shared+private"]
        out.append(f"  time from predicted bytes ({'scored' if d['time_predicted_bytes']['scored'] else 'printed'}): "
                   f"{t['cells']} cells rms {_pct(t['rms'])} worst {_pct(t['worst'])} "
                   f"beyond 5% {t['beyond_5pct']}")
        s = d["slope_g8"]
        out.append("  G>=8 SHARED slope n=2..6: " + ", ".join(
            f"{k} {v:.4f}" for k, v in s["measured"].items())
            + f" (registered {s['registered']}, band {s['band']})")
        for k, v in d["bytes"].items():
            if k != "cells":
                out.append(f"  bytes {k:<48s} {v['cells']:>4} cells rms {_pct(v['rms'])} "
                           f"worst {_pct(v['worst'])} beyond 5% {v['beyond_5pct']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case", choices=[*CASES, "all"])
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    for name in (CASES if a.case == "all" else [a.case]):
        d = score_case(name)
        txt = "\n".join(lines(d)) + "\n"
        print(txt, end="")
        if a.out_dir:
            a.out_dir.mkdir(parents=True, exist_ok=True)
            (a.out_dir / f"{name}.registered.json").write_text(
                json.dumps(d, indent=1, default=str) + "\n")
            (a.out_dir / f"{name}.registered.txt").write_text(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
