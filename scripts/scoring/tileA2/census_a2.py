"""Tile rental A2's census rules (docs/registered/2026-10-20-tileA2-census-gh200): the equality gate and the
named merge step that writes A2's census.score.json input. One module for the VM gate
(scripts/tileA2_gate.py census) and the scorer (score_census_a2.py), so both read one rule.

  equality gate  every (model, tile, GEMM) rental A compiled (its tile-census.json, copied verbatim into
                 docs/registered/2026-10-20-tileA2-hist/rentalA-tile-census.json, sha256 registered) must
                 be in A2's census with every compiled fact equal (tile_census_a2.FIELDS); anything else is
                 STACK DRIFT: the gate aborts the rental, and a census that drifted scores REFUSED (a STOP
                 for rental B2 and finalise_a2 refuses, as a path mismatch does).
  merge          A2's census.score.json is scored (score_census.py's own code) over ONE merged census:
                 rental A's entries (held to A2's by the gate) plus A2's deepseek-v2-lite entries at tread
                 3 when the OLMoE n 1 / n 3 rule reads SAME (tile_census_a2.check_rule); otherwise rental
                 A's entries alone (deepseek NOT SCORED, as rental A). A2's path mismatches carry into the
                 merged file (REFUSED).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(HERE), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import a2common as A2  # noqa: E402
import tile_census_a2 as TCA  # noqa: E402

RENTAL_A_CENSUS = f"{A2.HIST}/rentalA-tile-census.json"
LABEL = "ta2"
MERGED_LABEL = "ta2m"


def find(tree: Path, label: str = LABEL) -> Path | None:
    hits = sorted(Path(tree).rglob(f"*-{A2.CARD}-tile-census-*-{label}/tile-census.json"))
    return hits[0] if len(hits) == 1 else None


def rental_a(repo: Path) -> dict:
    return json.loads((Path(repo) / RENTAL_A_CENSUS).read_text())


def equality(ref: dict, got: dict) -> dict:
    """The equality gate: every rental-A (model, tile, GEMM) in `got` with every compiled fact equal."""
    by = {(e["model"], e["tile"], e["gemm"]): e for e in got.get("entries") or []}
    rows, drift = 0, []
    for e in ref.get("entries") or []:
        k = (e["model"], e["tile"], e["gemm"])
        rows += 1
        g = by.get(k)
        if g is None:
            drift.append(f"{'|'.join(k)}: not compiled on A2's board")
            continue
        d = [f for f in TCA.FIELDS if TCA.facts(e)[f] != TCA.facts(g)[f]]
        if d:
            drift.append(f"{'|'.join(k)}: {', '.join(d)} differ")
    return {"verdict": "EQUAL" if not drift else "STACK DRIFT", "compared": rows, "drift": drift}


def merge(ref: dict, got: dict) -> tuple[dict, dict]:
    """(the merged tile-census document, the merge record)."""
    eq = equality(ref, got)
    # recomputed from the entries (the VM's own reading of the rule is printed, never trusted)
    rule = TCA.check_rule(got.get("entries") or [], got.get("olmoe_n3_entries") or [])
    gated = [e for e in got.get("entries") or [] if e["model"] == TCA.GATED_MODEL]
    keep = [e for e in ref.get("entries") or [] if e["model"] != TCA.GATED_MODEL]
    admitted = bool(rule.get("admit")) and bool(gated)
    entries = keep + (gated if admitted else [])
    ref_err = [x for x in ref.get("errors") or [] if not x.startswith(TCA.GATED_MODEL)]
    got_err = [x for x in got.get("errors") or [] if x.startswith(TCA.GATED_MODEL)] if admitted else []
    errors = ref_err + got_err + ([] if admitted else [f"{TCA.GATED_MODEL}: NOT SCORED ({rule.get('verdict')})"])
    bad = list(got.get("path_mismatch") or [])
    if eq["verdict"] != "EQUAL":
        bad.append("STACK DRIFT against rental A's census: " + "; ".join(eq["drift"][:6]))
    doc = {"tool": "scripts/scoring/tileA2/census_a2.py merge (rental A's entries + A2's deepseek-v2-lite entries)",
           "tiles": ref.get("tiles"), "models": ref.get("models"), "entries": entries,
           "summary": TCA.TCEN.summary(entries), "errors": errors, "path_mismatch": bad,
           "verdict": ("REFUSED (path mismatch or stack drift)" if bad else ("INCOMPLETE" if errors else "COMPLETE"))}
    rec = {"equality_gate": eq, "olmoe_n1_n3_rule": {k: v for k, v in rule.items() if k != "rows"},
           "olmoe_n1_n3_rows": rule.get("rows"), "deepseek_admitted": admitted, "deepseek_entries": len(gated),
           "rental_A_entries": len(keep), "merged_entries": len(entries)}
    return doc, rec
