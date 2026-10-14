#!/usr/bin/env python
"""Score rental A's tile census (docs/registered/2026-10-14-tileA-census-gh200.json): no verdict but
VALIDITY (the MMA path), and the occupancies and PTX counts v4 reads.

    python scripts/scoring/tileA/score_census.py <repo> <tree> <out> [label]

Reads under <tree>: <date>-<card>-tile-census-<model>-<label>/tile-census.json (label ta). Writes
<out>/census.score.{json,txt}; exits 0 whatever it reads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tcommon as TC  # noqa: E402

PART = "census"


def find(tree: Path, label: str) -> Path | None:
    hits = sorted(Path(tree).rglob(f"*-{TC.CARD}-tile-census-*-{label}/tile-census.json"))
    return hits[0] if len(hits) == 1 else None


def score(repo: Path, tree: Path, label: str = "ta") -> dict:
    reg = TC.registration(repo, PART)
    res = {"registration": reg["name"], "part": PART}
    p = find(tree, label)
    if p is None:
        res["CENSUS"] = {"verdict": "NOT SCORED: no tile-census.json (v4 takes the design occupancy; a mechanical step "
                                    "re-reads it on rental B's board)"}
        res["occupancy"], res["occupancy_by_model"], res["counts"] = {}, {}, {}
        return res
    doc = json.loads(p.read_text())
    bad = doc.get("path_mismatch") or []
    by_model: dict = {}
    counts: dict = {}
    for e in doc.get("entries") or []:
        occ = (e.get("occupancy") or {}).get("ctas_per_sm")
        by_model.setdefault(e["model"], {}).setdefault(e["tile"], {})[e["gemm"]] = occ
        pc = e.get("ptx_counts") or {}
        if "error" not in pc and e["gemm"] == "w1":
            counts.setdefault(e["tile"], pc)
    common = {}
    for t in {t for m in by_model.values() for t in m}:
        vals = {g: {by_model[m][t].get(g) for m in by_model if t in by_model[m]} for g in ("w1", "w2")}
        if all(len(v) == 1 and None not in v for v in vals.values()):
            common[t] = {g: next(iter(v)) for g, v in vals.items()}
    design = {t["tile"]: t["design_occupancy"] for t in reg["tiles"]}
    differs = {t: {"design": design.get(t), "census": v} for t, v in common.items()
               if design.get(t) is not None and any(x != design[t] for x in v.values())}
    res["CENSUS"] = {"verdict": ("REFUSED (path mismatch): " + "; ".join(bad)) if bad else f"VALID ({doc.get('verdict')})",
                     "file": str(p), "entries": len(doc.get("entries") or []), "errors": doc.get("errors") or []}
    res["occupancy"] = {} if bad else common
    res["occupancy_by_model"] = {} if bad else by_model
    res["counts"] = {} if bad else counts
    res["differs_from_design"] = differs
    res["mechanical_steps"] = (["re-price rowkeys_tile.py at the census occupancy and rebuild tileA-bytes (tileA-census)"]
                               if differs else [])
    return res


def lines(res: dict) -> list[str]:
    out = [f"SCORED against {res['registration']}", f"CENSUS: {res['CENSUS']['verdict']}"]
    out += [f"  {t}: {v}" for t, v in sorted(res.get("occupancy", {}).items())]
    if res.get("differs_from_design"):
        out.append(f"  differs from the design occupancy: {res['differs_from_design']}")
    return out


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if len(a) not in (3, 4):
        print(__doc__)
        return 2
    res = score(Path(a[0]), Path(a[1]), *(a[3:4] or ["ta"]))
    print(TC.C3.write_score(Path(a[2]), PART, res, lines(res)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
