#!/usr/bin/env python
"""Model v7's prefill blind's census gate, read on the VM between units (docs/registered/2026-11-18-v7-prefill-blind-gh200).

    python scripts/v7p_gate.py census --results $RESULTS_ROOT [--repo $REPO] [--record FILE] [--lock 1410]

The blind's driver (scripts/gh200_v7p_session.sh, B3's driver plus lines marked V7P) runs a plan gate named v7pcensus
through this file; b3sag stays B3's (scripts/tileB3_gate.py sag, unchanged) and every other gate A2's. Exit 0: PASS
and the rental goes on. Exit 3: FAIL, and the driver ends the session at once (the card reset, the results pushed,
DRIVER-DONE), before any timed unit. Exit 2: refused (bad arguments); the driver treats ANY exit but 0 as a FAIL.

  census  this blind's compile-only census (label tp) of Granite-3.0-1B-A400M, Granite-3.0-3B-A800M and
          Qwen2-57B-A14B-tp8 at 128x128x64w8s3: each model's w1 and w2 entries equal OLMoE-1B-7B's in A2's merged census
          (the same top_k 8, which the compiled config carries) on every compiled fact the frozen pricing reads
          (tile_census_a2.FIELDS: registers, shared, warps, the MMA path, occupancy (2 CTAs per SM, RR's W_c 264), PTX
          counts, config), with no MMA path mismatch (v7pcommon.census_check). Any drift is a FAIL.

It prints one RESULT line, which the driver copies into the ledger, and writes its record (the driver's
$D/gate-v7pcensus-1410.json)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
P = HERE / "scoring" / "v7pblind"
if str(P) not in sys.path:
    sys.path.insert(0, str(P))
import v7pcommon as V  # noqa: E402

B = V.B


def line(rec: dict) -> str:
    head = f"RESULT: GATE v7p{rec['gate']} {rec['verdict']}"
    if rec.get("ctas_per_sm"):
        head += " ctas_per_sm " + ",".join(sorted({str(v) for v in rec["ctas_per_sm"].values()}))
    return head + (f" WHY {rec['why'][:300]}" if rec.get("why") else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gate", choices=("census",))
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=V.REPO)
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--session", type=Path, default=None, help="accepted and unused (the driver passes it to every gate)")
    ap.add_argument("--lock", type=float, default=None, help="accepted; must be the registered lock 1410 when given")
    a = ap.parse_args(argv)
    if a.results is None:
        print(f"REFUSED: the {a.gate} gate needs --results", file=sys.stderr)
        return B.REFUSED_RC
    if a.lock is not None and float(a.lock) != float(V.LOCK):
        print(f"REFUSED: the gate runs at the registered lock {V.LOCK}, not {a.lock:g}", file=sys.stderr)
        return B.REFUSED_RC
    rec = V.census_gate(a.results, a.repo)
    if a.record is not None:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        a.record.write_text(B.dumps(rec))
    print(line(rec))
    return B.PASS_RC if rec["verdict"] == "PASS" else B.FAIL_RC


if __name__ == "__main__":
    sys.exit(main())
