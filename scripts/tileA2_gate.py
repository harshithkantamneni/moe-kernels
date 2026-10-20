#!/usr/bin/env python
"""Tile rental A2's early abort gates, read on the VM between units (docs/registered/2026-10-20-tileA2-gates-gh200).

    python scripts/tileA2_gate.py calibrate --session $S [--record FILE]
    python scripts/tileA2_gate.py census    --results $RESULTS_ROOT --repo $REPO [--record FILE]
    python scripts/tileA2_gate.py probe     --results $RESULTS_ROOT --session $S --lock L [--record FILE]
    python scripts/tileA2_gate.py band      --results $RESULTS_ROOT --lock L [--record FILE]
    python scripts/tileA2_gate.py f64w8     --results $RESULTS_ROOT --lock L [--record FILE]

Exit 0: the gate PASSES and the rental goes on. Exit 3: it FAILS, and A2's driver ends the session at once
(scripts/gh200_tileA2_session.sh: the card reset, the results pushed, DRIVER-DONE). Exit 2: refused (bad
arguments); the driver treats ANY exit but 0 as a FAIL, so a gate that cannot read its input never passes.
Each gate is the registered rule, computed by the code the scorers run (a2common, score_floor_a2,
census_a2, probe_a2), and prints one RESULT line:

  calibrate  G-CAL: calibrate's settled dense bf16 GEMM clock (its summary line's gemm_clock_mhz, the
             median under load) is at least the ladder's LOWEST rung, 1305 MHz (build review fix 3: a board
             the lowest rung could hold is never ended here; the probe decides the rung). No reading is a FAIL.
  census     G-CEN: A2's tile census (label ta2) equals rental A's on every (model, tile, GEMM) rental A
             compiled, every compiled fact (census_a2.equality), and has no MMA path mismatch.
  probe      G-PROBE, one rung: probe_a2.rung at L (P1 to P5; --partial after each page for the early stop).
             The driver walks the registered ladder
             1710, 1410, 1305: the first rung that HOLDS is A2's lock; none holds: the rental ends.
  band       G-BAND, after the f128 Mixtral page: tcommon.joint_cF over the f128 pages in the counter band
             in force is not None (the primary [0.975 L, L], else the registered fallback [0.959 L, L]).
  f64w8      G-64, after the f64w8 Mixtral page: tcommon.joint_cF over the f64w8 pages in the band in force
             is not None (without it finalise refuses when no accounting is SELECTED, and the stop rule STOPs).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE), str(HERE / "scoring" / "tileA2")):
    if p not in sys.path:
        sys.path.insert(0, p)

PASS_RC, FAIL_RC, REFUSED_RC = 0, 3, 2


def gemm_clock(session: Path) -> tuple[float | None, str]:
    """calibrate's settled bf16 GEMM clock: the JSON summary line's gemm_clock_mhz in the one
    calibrate log under $S/logs."""
    logs = sorted((Path(session) / "logs").glob("calibrate-*.log"))
    if len(logs) != 1:
        return None, f"{len(logs)} calibrate logs under {session}/logs"
    for line in logs[0].read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{") and '"gemm_clock_mhz"' in line:
            try:
                v = json.loads(line).get("gemm_clock_mhz")
            except ValueError:
                continue
            if isinstance(v, (int, float)):
                return float(v), str(logs[0])
    m = re.findall(r"clock, bf16 GEMM\s+(\d+) MHz median", logs[0].read_text(errors="replace"))
    if m:
        return float(m[-1]), str(logs[0]) + " (the 'clock, bf16 GEMM' line)"
    return None, f"no gemm_clock_mhz in {logs[0]}"


def g_calibrate(a) -> dict:
    import a2common as A2
    v, src = gemm_clock(a.session)
    low = A2.LOCK_LOWEST
    ok = v is not None and v >= low
    return {"gate": "calibrate", "verdict": "PASS" if ok else "FAIL", "gemm_clock_mhz": v, "lowest_rung_mhz": low,
            "source": src, "why": "" if ok else (f"settled bf16 GEMM clock {v} MHz under the ladder's lowest rung {low:g}"
                                                  if v is not None else f"no settled GEMM clock read ({src})")}


def g_census(a) -> dict:
    import census_a2 as CA
    p = CA.find(a.results)
    if p is None:
        return {"gate": "census", "verdict": "FAIL", "why": "no A2 tile-census.json (label ta2)"}
    got = json.loads(p.read_text())
    eq = CA.equality(CA.rental_a(a.repo), got)
    bad = got.get("path_mismatch") or []
    ok = eq["verdict"] == "EQUAL" and not bad
    why = "" if ok else "; ".join((["MMA path mismatch: " + "; ".join(bad[:4])] if bad else []) + eq["drift"][:6])
    return {"gate": "census", "verdict": "PASS" if ok else "FAIL", "file": str(p), "equality": eq,
            "path_mismatch": bad, "why": why}


def g_probe(a) -> dict:
    import probe_a2 as PR
    r = PR.rung(a.results, a.session, a.lock, partial=a.partial)
    return {"gate": "probe", "verdict": "PASS" if r["verdict"] == "HOLDS" else "FAIL", "rung": r,
            "why": "; ".join(r["why"][:8])}


def _fit(a, which: str) -> dict:
    import score_floor_a2 as SFA
    rules, rec = SFA.choose_band(a.results, a.lock)
    if which == "band":
        ok = not rec["which"].startswith("primary (neither")
        return {"gate": "band", "verdict": "PASS" if ok else "FAIL", "band_in_force": rec,
                "why": "" if ok else "joint_cF is None on the f128 pages in the primary and the fallback band"}
    fit, why = SFA.fit_64(a.results, rules)
    ok = fit is not None
    return {"gate": "f64w8", "verdict": "PASS" if ok else "FAIL", "band": list(rules.band), "pages": why,
            "cells": (fit or {}).get("cells"),
            "why": "" if ok else f"joint_cF is None on the f64w8 pages in the band in force {list(rules.band)}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gate", choices=("calibrate", "census", "probe", "band", "f64w8"))
    ap.add_argument("--session", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--lock", type=float, default=None)
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--partial", action="store_true", help="probe: read only the pages written so far (the early stop)")
    a = ap.parse_args(argv)
    need = {"calibrate": ("session",), "census": ("results",), "probe": ("results", "session", "lock"),
            "band": ("results", "lock"), "f64w8": ("results", "lock")}[a.gate]
    miss = [k for k in need if getattr(a, k) is None]
    if miss:
        print(f"REFUSED: the {a.gate} gate needs --{' --'.join(miss)}", file=sys.stderr)
        return REFUSED_RC
    fn = {"calibrate": g_calibrate, "census": g_census, "probe": g_probe,
          "band": lambda x: _fit(x, "band"), "f64w8": lambda x: _fit(x, "f64w8")}[a.gate]
    rec = fn(a)
    if a.record is not None:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        a.record.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"RESULT: GATE {a.gate} {rec['verdict']}{(' ' + rec['why'][:300]) if rec.get('why') else ''}")
    return PASS_RC if rec["verdict"] == "PASS" else FAIL_RC


if __name__ == "__main__":
    sys.exit(main())
