#!/usr/bin/env python
"""DESCRIPTIVE, post-registration, NOT A VERDICT: model v7's PREFILL clauses re-run with one fix, after the registered
gate refused the regime.

    python scripts/scoring/v7/prefill_rerun_v7.py [--out <dir>]

The registered gate (gate_v7.py, run 2026-11-18 at 6ed4c57d after the registration was public; committed output
gate_v7.out.txt and gate_v7.json) read PREFILL: NO-GO: REFUSED: ModuleNotFoundError: No module named 'refit4'. G2
calls round 5's r7.pass_estimates5, which calls r7._rc() (scripts/round5_frozen/2026-10-28/round5/work/r7.py:203-208):
os.chdir(r5.REFIT4); import refit4. That import resolves only while REFIT4 or the current directory is on sys.path.
round5_chain.chain() imports refit4 once (RF = r7._rc(), while r5's own sys.path entry is live), then pops it from
sys.modules and restores sys.path, so the later call finds no refit4. That registered reading stands: PREFILL is
NO-GO, and nothing this script prints can change it.

This script runs exactly gate_v7's PREFILL computation and clauses (gate_v7.compute_prefill: the TREES refusal, the six
held-out means, G2's estimate, B3's 84 cells as information; gate_v7.prefill_clauses: G1, G2; the every-switch-off
rung beside, as the gate's evaluate() would have), after the freeze check, with the single fix that r7._rc returns
the refit4 module round5_chain.chain() already imported (the module the chain's held-out sets and B2 cells already
use). No other line of the gate, the chain or round 5 is changed; the fix is applied in memory only. It writes
prefill_rerun_v7.json and prefill_rerun_v7.txt, labelled DESCRIPTIVE. Exit 0 when it ran, 2 when it refused (freeze,
or the committed gate output not the registered run's)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gate_v7 as G  # noqa: E402

V7 = G.V7
LABEL = ("DESCRIPTIVE, post-registration, NOT A VERDICT: gate_v7's PREFILL clauses with one fix (r7._rc returns "
         "the refit4 module round5_chain.chain() imported); the registered PREFILL verdict stands")
#: the registered gate's committed outputs (the official run, 2026-11-18) and their sha256
GATE_TXT = HERE / "gate_v7.out.txt"
GATE_JSON = HERE / "gate_v7.json"
GATE_TXT_SHA = "882c749163b1b622b4a077ea0e3db2750cb014a4cf8cfc5b3a8a1fae68f56df7"
GATE_JSON_SHA = "db84baf428bb1a23f1d27170d79a8a97ce95d2faf564ec0450994a8b42364670"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def registered_prefill_line() -> str:
    """The registered gate's PREFILL line, from its committed output (held to the recorded sha256)."""
    for p, want in ((GATE_TXT, GATE_TXT_SHA), (GATE_JSON, GATE_JSON_SHA)):
        if not p.is_file() or sha256(p) != want:
            raise G.V7.Refused(f"{p.name} is not the registered gate run's output (sha256 {want[:12]})")
    return next(x for x in GATE_TXT.read_text().splitlines() if x.startswith("PREFILL:"))


def apply_fix() -> None:
    """The one fix: r7._rc returns the refit4 module the chain imported, instead of importing it again."""
    import round5_chain as R5
    C = R5.chain()
    RF = C["RF"]
    assert RF.__name__ == "refit4", RF
    C["r7"]._rc = lambda: RF


def judge(x: dict) -> dict:
    """gate_v7's PREFILL judgement (evaluate()'s 'prefill' entry), unchanged."""
    return dict(G.prefill_clauses(x["means"], x["P_A"]),
                information={"B3 84 cells (SEEN, never a clause)": x.get("b3_information"),
                             "estimate": x.get("estimate"), "estimate_cells": x.get("estimate_cells")})


def describe(d: dict) -> dict:
    """The clause document with its top-level verdict renamed: a DESCRIPTIVE reading, not a verdict."""
    d = dict(d)
    d["reads (DESCRIPTIVE, not a verdict)"] = d.pop("verdict")
    return d


def run() -> dict:
    reg = json.loads(G.REG_PATH.read_text())
    doc = {"what": "model v7's PREFILL clauses re-run after the registered gate refused the regime", "label": LABEL,
           "registration": G.NAME, "registration_sha256": G.sha256(G.REG_PATH),
           "registered_verdict (stands)": registered_prefill_line()}
    bad = G.verify_freeze(reg)
    if bad:
        raise V7.Refused("the freeze does not hold: " + "; ".join(bad[:10]))
    doc["freeze"] = {"verdict": "HOLDS", "files": len(reg["freeze"]["files"])}
    apply_fix()
    doc["fix"] = "r7._rc returns round5_chain.chain()['RF'] (in memory); nothing else changed"
    doc["switches"] = V7.V7.as_dict()
    doc["prefill"] = describe(judge(G.compute_prefill(V7.V7)))
    doc["prefill"]["ablation (DESCRIPTIVE)"] = {lab: describe(judge(G.compute_prefill(sw)))
                                                for lab, sw in G.ABLATION["prefill"].items()}
    return doc


def lines(doc: dict) -> list[str]:
    def block(name: str, p: dict) -> list[str]:
        g1, g2, inf = p["G1"], p["G2"], p["information"]
        est = inf["estimate"]
        L = [f"{name}: reads {p['reads (DESCRIPTIVE, not a verdict)']}"
             + (": " + "; ".join(p["why"]) if p["why"] else "")]
        L.append(f"  G1 {g1['verdict']}: held-out 128 G 1 means (%, |mean| <= {g1['max_abs_mean_pct']}): "
                 + ", ".join(f"{k} {v:+.2f}" for k, v in g1["means_pct"].items()))
        L.append(f"  G2 {g2['verdict']}: estimated P(A) {g2['P_A']} (bar {g2['min']}); "
                 + ", ".join(f"{k} {v}" for k, v in est.items()) + f"; on {inf['estimate_cells']} cells")
        L.append("  B3 84 cells (SEEN information, never a clause), % mean / rms / worst: "
                 + ", ".join(f"{m} {s['mean']:+.2f} / {s['rms']:.2f} / {s['worst']:+.2f} (n {s['n']})"
                             for m, s in inf["B3 84 cells (SEEN, never a clause)"].items()))
        return L
    L = [LABEL, f"REGISTERED (stands): {doc['registered_verdict (stands)']}",
         f"freeze {doc['freeze']['verdict']} ({doc['freeze']['files']} files); fix: {doc['fix']}"]
    L += block("PREFILL re-run, v7 every switch on (DESCRIPTIVE)", doc["prefill"])
    for lab, p in doc["prefill"]["ablation (DESCRIPTIVE)"].items():
        L += block(f"PREFILL re-run, {lab} (DESCRIPTIVE)", p)
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE, help="write prefill_rerun_v7.{json,txt} here (default: beside)")
    a = ap.parse_args(argv)
    try:
        doc = run()
    except V7.Refused as exc:
        print(f"REFUSED: {exc}")
        return 2
    L = lines(doc)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "prefill_rerun_v7.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    (a.out / "prefill_rerun_v7.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
