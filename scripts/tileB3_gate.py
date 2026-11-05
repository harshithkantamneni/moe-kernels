#!/usr/bin/env python
"""Tile rental B3's three early STOP gates, read on the VM between units (docs/registered/2026-11-04-tileB3-gh200).

    python scripts/tileB3_gate.py census --results $RESULTS_ROOT [--repo $REPO] [--record FILE]
    python scripts/tileB3_gate.py sag    --results $RESULTS_ROOT [--record FILE]
    python scripts/tileB3_gate.py sag64  --results $RESULTS_ROOT [--record FILE]

B3's driver (scripts/gh200_tileB3_session.sh) runs a plan gate named b3census, b3sag or b3sag64 through this file and
every other gate through scripts/tileA2_gate.py as before. Exit 0: the gate PASSES and the rental goes on. Exit 3: it
FAILS and the driver ends the session at once (the card reset, the results pushed, DRIVER-DONE), before any timed unit.
Exit 2: refused (bad arguments); the driver treats ANY exit but 0 as a FAIL. Each gate prints one RESULT line, which
the driver copies into the ledger, and writes its record (the driver's $D/gate-b3<name>-1410.json):

  census  B3's compile-only census (label tb) of Phi-3.5-MoE, Qwen1.5-MoE-A2.7B, JetMoE-8B and deepseek-v2-lite at
          128x128x64w8s3 and 64x128x64w8s3 equals A2's merged census on every compiled fact the frozen pricing reads
          (tile_census_a2.FIELDS: registers, shared, warps, the MMA path, occupancy, PTX counts, config), with no MMA path
          mismatch. A blind model's drift is a FAIL; deepseek-v2-lite's is not (its page is NOT SCORED).
  sag     the on-board F1 estimator on this board's OLMoE f128 and Mixtral-8x7B f128 pages (b3common.page_fit, the
          frozen chain's r4.sag_page_fit copied verbatim) and the registered sag-fit range (b3common.SAG_RANGE):
          points, A, tau, rms per page, the 1 ms ratio, par's A. A missing page is a FAIL. The RESULT line names par
          and both pages' sha256 prefixes, so the ledger holds the finalisation's inputs before the first timed unit.
  sag64   the same estimator on this board's OLMoE f64w8 and Mixtral-8x7B f64w8 pages and the registered 64x128 range
          (b3common.SAG64_RANGE), plus the pages' clock clause (b3common.clock64: r6common.clock_rule in the band in
          force, CLOCK64); par is the 64x128 cells' MEASURED-ON-BOARD sag. Its RESULT line as sag's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
B3 = HERE / "scoring" / "tileB3"
if str(B3) not in sys.path:
    sys.path.insert(0, str(B3))
import b3common as B  # noqa: E402


def line(rec: dict) -> str:
    if rec["gate"] in ("sag", "sag64") and rec.get("par"):
        pp = rec.get("per_page") or {}
        pages = " ".join(f"{m.split('-')[0]} {f['A_us']}/{f['tau_us']:g} rms {f['rms_us']} sha {f['sha256'][:12]}"
                         for m, f in pp.items())
        head = (f"RESULT: GATE b3{rec['gate']} {rec['verdict']} par A {rec['par'][0]:.4f} us tau {rec['par'][1]:g} us "
                f"ratio {rec['ratio_1ms']} {pages}")
    else:
        head = f"RESULT: GATE b3{rec['gate']} {rec['verdict']}"
    return head + (f" WHY {rec['why'][:300]}" if rec.get("why") else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gate", choices=("census", "sag", "sag64"))
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=B.REPO)
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--session", type=Path, default=None, help="accepted and unused (the driver passes it to every gate)")
    ap.add_argument("--lock", type=float, default=None, help="accepted; must be the registered lock 1410 when given")
    a = ap.parse_args(argv)
    if a.results is None:
        print(f"REFUSED: the {a.gate} gate needs --results", file=sys.stderr)
        return B.REFUSED_RC
    if a.lock is not None and float(a.lock) != float(B.LOCK):
        print(f"REFUSED: B3's gates run at the registered lock {B.LOCK}, not {a.lock:g}", file=sys.stderr)
        return B.REFUSED_RC
    rec = {"census": lambda: B.census_gate(a.results, a.repo), "sag": lambda: B.sag_gate(a.results),
           "sag64": lambda: B.sag_gate(a.results, B.SAG64_UNIT)}[a.gate]()
    if a.record is not None:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        a.record.write_text(B.dumps(rec))
    print(line(rec))
    return B.PASS_RC if rec["verdict"] == "PASS" else B.FAIL_RC


if __name__ == "__main__":
    sys.exit(main())
