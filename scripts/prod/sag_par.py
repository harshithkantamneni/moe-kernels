"""The board's 128x128 sag par for the production check (review 1, BLOCKER 2), finalised by the v7 prefill blind's OWN
finaliser: scripts/scoring/v7pblind/v7pcommon.py finalise_pars, unchanged, on the two f128 bytes pages and the b3sag
gate record that scripts/prod/sag1410.sh's driver run (scripts/gh200_v7p_session.sh --plan scripts/prod/sag1410.plan)
left on the VM.

    python scripts/prod/sag_par.py --results $RESULTS_ROOT --record $SESSION_ROOT/gh200-driver/gate-b3sag-1410.json \\
        --driver-exit 0 --started <epoch s the sag1410 step began> --out <results>/sag/par.json

A PASS needs the driver's exit 0 AND a gate record written after the step began (review 2, MINOR 4: a record left by
an earlier run of the driver, or a driver that stopped after its gate, is a FAIL, never a par).

finalise_pars holds the repo's sag estimator (b3common.page_fit) and the frozen chain's (r4.sag_page_fit) equal to
each other and to the gate's record (its par and both pages' sha256), and returns r4.par of the two fits: (A us,
tau us). par.json carries that par with verdict PASS; anything else is verdict FAIL with the reason and exit 3 (the
scorer refuses a FAIL par: prodlib.load_board_par). HOW THE PAR FLOWS: par.json -> score_prod.py --sag ->
prodlib.load_board_par -> prodlib.price_v8(sag128=par) -> v8.price({... "sag128": par, "sag64": A2_PAR_64, "knee":
K1_SEEN}) -> v7's prefill branch (round5_chain.price_128 / k5.cell_ms) at the board's sag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "scoring" / "v7pblind")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import prodlib as PL  # noqa: E402


def finalise(results: Path, record_path: Path, driver_exit: int, started: float) -> dict:
    """par.json's document; never raises on a bad board (verdict FAIL with the reason)."""
    doc = {"schema": PL.PAR_SCHEMA, "verdict": "FAIL", "par": None, "driver_exit": int(driver_exit),
           "started": float(started), "results": str(results), "record": str(record_path),
           "finaliser": "scripts/scoring/v7pblind/v7pcommon.py finalise_pars (unchanged)",
           "plan": "scripts/prod/sag1410.plan"}
    if not record_path.is_file():
        doc["why"] = f"no sag gate record {record_path} (the driver exited {driver_exit} before the gate)"
        return doc
    if int(driver_exit) != 0:
        doc["why"] = f"the driver exited {driver_exit} (a par needs its exit 0)"
        return doc
    mtime = record_path.stat().st_mtime
    doc["record_mtime"] = mtime
    if mtime < float(started):
        doc["why"] = (f"the gate record {record_path} was written at {mtime:.0f}, before this step began at "
                      f"{float(started):.0f} (a stale record of an earlier run)")
        return doc
    raw = record_path.read_bytes()
    doc["record_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        import v7pcommon as VP
        rec = json.loads(raw)
        fin = VP.finalise_pars(Path(results), rec)
    except (ValueError, LookupError, OSError, TypeError, ZeroDivisionError) as exc:
        doc["why"] = f"{type(exc).__name__}: {exc}"
        return doc
    doc.update(verdict="PASS", par=[float(x) for x in fin["par"]], per_page=fin["per_page"],
               gate_verdict=rec.get("verdict"),
               inside_held_out_par_span=fin.get("inside_held_out_par_span (DESCRIPTIVE)"))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--record", required=True, type=Path)
    ap.add_argument("--driver-exit", type=int, required=True)
    ap.add_argument("--started", type=float, required=True, help="epoch seconds the sag1410 step began")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    doc = finalise(a.results, a.record, a.driver_exit, a.started)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    if doc["verdict"] != "PASS":
        print(f"FAIL sag par: {doc['why']}")
        return 3
    print(f"PASS sag par: A {doc['par'][0]:.3f} us, tau {doc['par'][1]:.1f} us -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
