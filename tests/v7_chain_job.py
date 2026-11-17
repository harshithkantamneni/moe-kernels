"""The job for tests/test_timing_model_v7_prefill.py: v7's prefill branch over round 5's chain, in its own interpreter
(the chain loads generic module names and takes about a minute), once: `python tests/v7_chain_job.py` prints one JSON
document with every reading the tests check.

  vendored  scripts/round5_frozen holds B3's 32 frozen work files at their registered sha256 (round5_chain.verify)
  b3_off    v7 with RR off on B2's 84 NATIVE cells at B3's board pars (b3.predictions.json par, par64): each cell's ms
            to 6 decimals and its reads to the byte, against B3's committed predictions
  r5_off    round 5's held-out 128 G 1 table re-priced by v7 with RR off (r4.summ means, 2 decimals), against
            tileB3.score.json preflight.round5_table
  b3_on     v7 (RR on) on the same 84 cells: which cells moved, by tile and G; with D6FIX also on and off (no change);
            the RR term's parts per moved cell. B3's 84 cells are SEEN information, not a gate clause: no statistic
            against measurement is computed here
  gate_trees  gate_v7.prefill_trees() on this tree (the prefill regime's TREES refusal): must be no difference
  trace     every file the job opened outside the repository and the interpreter (an audit hook from before the chain
            loads): must be none
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
OPENED: set = set()


def _hook(ev, args):
    if ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            OPENED.add(os.path.realpath(os.fsdecode(args[0])))
        except Exception:
            pass


sys.addaudithook(_hook)
for p in (REPO, REPO / "scripts", REPO / "scripts" / "scoring" / "v7"):
    sys.path.insert(0, str(p))
import r3_timing_model_v7 as V7  # noqa: E402
import round5_chain as R5  # noqa: E402
import gate_v7 as GV  # noqa: E402


def outside(paths) -> list:
    home = str(Path.home())
    keep = (str(REPO.resolve()), sys.prefix, sys.base_prefix, os.path.realpath(sys.prefix),
            os.path.realpath(sys.base_prefix))
    return sorted(p for p in paths if p.startswith(home) and not p.startswith(keep) and "/.venv/" not in p)


def main() -> dict:
    out: dict = {}
    R5.verify()
    out["vendored"] = {f: R5.sha256(R5.FROZEN_ROOT / f) for f in R5.FROZEN_FILES}
    fin = json.loads((REPO / "scripts/scoring/tileB3/b3.predictions.json").read_text())
    par, par64 = tuple(fin["par"]), tuple(fin["par64"])
    rows = R5.b2_rows()
    off, on, on_nofix, moved = [], [], [], []
    for r, c in zip(rows, fin["cells"]):
        a = V7.price_prefill(r, sag128=par, sag64=par64, switches=V7.OFF)
        b = V7.price_prefill(r, sag128=par, sag64=par64, switches=V7.V7)
        b2 = V7.price_prefill(r, sag128=par, sag64=par64, switches=V7.Switches(RR=True, D6FIX=False))
        off.append({"ms": round(a["ms"], 6), "reads": {g: int(round(v)) for g, v in a["reads"].items()},
                    "want_ms": c["ms"], "want_reads": c["reads"]})
        on.append(b["ms"])
        on_nofix.append(b2["ms"])
        if b["ms"] != a["ms"] or b["reads"]["w1"] != a["reads"]["w1"] or b["reads"]["w2"] != a["reads"]["w2"]:
            moved.append({"index": c["index"], "model": c["model"], "cls": c["cls"], "tile": b["tile"], "G": b["G"],
                          "w2_same": b["reads"]["w2"] == a["reads"]["w2"], "rr": b["terms"].get("RR")})
    out["b3_off"] = off
    out["b3_on"] = {"ms": on, "ms_rr_without_d6fix": on_nofix, "moved": moved,
                    "g1_cells": sum(1 for c in fin["cells"] if int(c["G"]) == 1)}
    C = R5.chain()
    r5 = {}
    for k, (cells, hm) in R5.held_out_sets().items():
        knee = R5.held_out_knee(hm)
        e = [V7.price_prefill(r, sag128=R5.held_out_sag(r, hm), knee=knee, switches=V7.OFF)["ms"] / r["meas"] - 1
             for r in cells]
        r5[k] = C["r4"].summ(e)["mean"]
    out["r5_off"] = r5
    out["r5_want"] = json.loads((REPO / "scripts/scoring/tileB3/tileB3.score.json").read_text())["preflight"]["round5_table"]
    out["gate_trees"] = GV.prefill_trees()
    out["trace"] = outside(OPENED)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), default=str))
