#!/usr/bin/env python
"""Writes the tile stage's eight registered JSON files: rental A's six (docs/registered/2026-10-14-tileA-*)
and rental B's two, frozen now and finalised mechanically from rental A's selections
(docs/registered/2026-10-14-tileB-*).

    python scripts/scoring/tileA/register.py [--out-dir docs/registered] [--check] [--only PART ...]
                                             [--census-score <rental A's census.score.json>]

Every number is computed here from committed files (regtile.py: model v4 over the pinned v3 modules,
the CAL pricer of rental 6 with its A proxy as of 2026-10-14, the synthetic histogram pages of
scripts/skew_synth_tile.py, the row-key rows of rowkeys_tile.py) or is a typed-in input with its
provenance and label. The power numbers are computed on the CPU in float64; power_tile.py re-runs
every point on the laptop GPU (torch MPS, float32) into the tileB hist directory's
power_tile_check.json. `--check` recomputes and compares with the committed JSON and text (exit 1 on
any difference) instead of writing; the `code_pins` block is excluded from the comparison, as in
rentals 3 to 7.

LEAKAGE, said once for all eight: every published page is SEEN, rental 7's included; the published
set is read as of 2026-10-14 (sessions whose name carries 'tile' excluded), so a check after the tile
pages are published recomputes the same files. No tile-stage page exists when this runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regtile as RT  # noqa: E402
import tcommon as TC  # noqa: E402

REPO = TC.REPO
REG = REPO / "docs" / "registered"
#: every registered code pin of rental 7 (byte-identical, tests/test_pinned_files.py), and the tile stage's own code
PINNED = ("scripts/r3_timing_model.py", "scripts/wave_split_bytes.py", "scripts/r3_timing_model_v3.py",
          "scripts/wave_split_bytes_v3.py", "scripts/private_weight_reference.py", "scripts/locked_r3.py",
          "scripts/gh200_model_session.sh", "scripts/dram_counter_route.py", "scripts/lock_gate.py",
          "scripts/skew_synth.py", "scripts/skew_synth7.py", "scripts/scoring/rental5/skewmodel.py",
          "scripts/scoring/rental6/r6common.py", "scripts/scoring/rental6/r6model.py", "scripts/scoring/rental6/reg6.py",
          "scripts/scoring/rental6/rowkeys.py", "scripts/scoring/rental7/r7common.py", "scripts/scoring/rental7/reg7.py",
          "scripts/scoring/rental7/rowkeys7.py",
          "scripts/r3_timing_model_v4.py", "scripts/wave_split_bytes_v4.py", "scripts/r3_tile.py",
          "scripts/dram_counter_route_tile.py", "scripts/tile_census.py", "scripts/skew_synth_tile.py",
          "scripts/gh200_tile_session.sh", "scripts/scoring/tileA/tcommon.py", "scripts/scoring/tileA/regtile.py",
          "scripts/scoring/tileA/rowkeys_tile.py", "scripts/scoring/tileA/power_tile.py",
          "scripts/scoring/tileA/finalise.py")
SKIP_TXT = {"code_pins", "byte_predictions", "nominal_cells", "files", "cells", "points", "scorer_ambiguities",
            "design_tables (derived, DESCRIPTIVE)", "tiles", "seen_ptx"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _flat(prefix, v, out, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
            if k in SKIP_TXT:
                out.append(f"{prefix}{k}: (in the JSON)")
                continue
            if isinstance(x, (dict, list)) and depth < 2:
                out.append(f"{prefix}{k}:")
                _flat(prefix + "  ", x, out, depth + 1)
            else:
                out.append(f"{prefix}{k}: {x if isinstance(x, str) else json.dumps(x, default=str)}")
    elif isinstance(v, list):
        for x in v:
            out.append(f"{prefix}- {x if isinstance(x, str) else json.dumps(x, default=str)}")
    else:
        out.append(f"{prefix}{v}")


def txt(part: str, d: dict) -> list[str]:
    rental = "rental A" if part in TC.PARTS_A else "rental B (frozen; finalised mechanically)"
    L = [f"REGISTERED {d['registered']}: the tile stage, {rental}, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d.get('tool', '-')}", ""]
    if part in RT.TEXT:
        L += RT.TEXT[part](d) + [""]
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool"):
            continue
        _flat("", {k: v}, L)
    return L


def build(only=None) -> dict:
    parts = [p for p in TC.PARTS if only is None or p in only]
    docs = {p: RT.BUILDERS[p]() for p in parts}
    pins = {p: sha256(REPO / p) for p in PINNED}
    for d in docs.values():
        d["code_pins"] = pins
    return docs


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "code_pins"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=REG)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", nargs="+", choices=TC.PARTS, default=None)
    ap.add_argument("--census-score", type=Path, default=None,
                    help="rental A's census.score.json: every occupancy is read from it (the mechanical rebuild)")
    a = ap.parse_args(argv)
    if a.census_score is not None:
        TC.use_census(a.census_score)
    docs = build(a.only)
    bad = 0
    for part, doc in docs.items():
        jpath = a.out_dir / f"{TC.NAMES[part]}.json"
        tpath = a.out_dir / f"{TC.NAMES[part]}.txt"
        jtext = json.dumps(doc, indent=1, default=str) + "\n"
        ttext = "\n".join(txt(part, doc)) + "\n"
        if a.check:
            same_j = jpath.exists() and _strip(json.loads(jpath.read_text())) == _strip(json.loads(jtext))
            same_t = tpath.exists() and tpath.read_text() == ttext
            for p, same in ((jpath, same_j), (tpath, same_t)):
                print(f"{p.name}: {'same' if same else 'DIFFERS'}")
                bad += not same
        else:
            jpath.write_text(jtext)
            tpath.write_text(ttext)
            print(f"wrote {jpath}\nwrote {tpath}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
