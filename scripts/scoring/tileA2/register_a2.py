#!/usr/bin/env python
"""Writes tile rental A2's registrations: docs/registered/2026-10-20-tileA2-{census,floor,bytes,e1cal,dead,rep,convl,gates}-gh200,
rental B2's 2026-10-20-tileB2-{e1tile,stop}-gh200 (generated over the frozen tileB builders) and the decode stage's
2026-10-20-decode-tilesource-gh200 addendum, each a .json and a .txt.

    python scripts/scoring/tileA2/register_a2.py [--out-dir docs/registered] [--check] [--only PART ...]
                                                 [--census-score <A2's census.score.json>]

Every number is computed here from committed files (regtile_a2.py over the pinned regtile, v4 at L by
r3_timing_model_v4L) or is a typed-in input with its provenance and label. `--check` recomputes and compares
with the committed JSON (the `code_pins` block excluded, as rentals 3 to 7 and the tile stage) and text
(exit 1 on any difference) instead of writing. `--census-score` rebuilds every registration at A2's merged
census occupancy, the tile stage's mechanical step (tileA-census mechanical_steps), before B2 is finalised.

LEAKAGE: every published page is SEEN, rental A's tile session included (each registration's SEEN block);
the published set the tileA builders read is regtile's as of 2026-10-14 (sessions whose name carries
'tile' excluded), so a check after A2's pages are published recomputes the same files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a2common as A2  # noqa: E402
import regtile_a2 as RT  # noqa: E402

#: the tile stage's register.py (its PINNED list and renderer), loaded by path: several scoring
#: directories hold a register.py, so a bare import would take whichever comes first on sys.path
RA = A2.TC._load("tileA_register", A2.TILEA / "register.py")

TC = A2.TC
REPO = A2.REPO
REG = REPO / "docs" / "registered"
PARTS = A2.PARTS_A2 + A2.PARTS_B2 + ("tilesource",)
#: the tile stage's pins, and A2's own code (every file A2's driver, gates, scorers and generator run)
PINNED = tuple(RA.PINNED) + (
    "scripts/r3_timing_model_v4L.py", "scripts/tile_census_a2.py", "scripts/gh200_tileA2_session.sh", "scripts/tileA2_gate.py",
    "scripts/scoring/tileA/score_floor.py", "scripts/scoring/tileA/score_bytes.py", "scripts/scoring/tileA/score_census.py",
    "scripts/scoring/tileA/score_e1cal.py", "scripts/scoring/tileA/score_rep.py", "scripts/scoring/tileA/score_dead.py",
    "scripts/scoring/tileA/score_tileB.py",
    "scripts/scoring/tileA2/a2common.py", "scripts/scoring/tileA2/census_a2.py", "scripts/scoring/tileA2/probe_a2.py",
    "scripts/scoring/tileA2/hist_a2.py", "scripts/scoring/tileA2/conv_offline.py", "scripts/scoring/tileA2/regtile_a2.py",
    "scripts/scoring/tileA2/register_a2.py", "scripts/scoring/tileA2/finalise_a2.py",
    "scripts/scoring/tileA2/score_census_a2.py", "scripts/scoring/tileA2/score_floor_a2.py",
    "scripts/scoring/tileA2/score_bytes_a2.py", "scripts/scoring/tileA2/score_rep_a2.py",
    "scripts/scoring/tileA2/score_e1cal_a2.py", "scripts/scoring/tileA2/score_dead_a2.py",
    "scripts/scoring/tileA2/score_convl_a2.py", "scripts/scoring/tileA2/score_probe_a2.py",
    "scripts/scoring/tileA2/score_tileB2_a2.py", "scripts/scoring/tileA2/score_session_a2.py",
    "scripts/scoring/tileA2/decode_gate_a2.py")
SKIP_TXT = set(RA.SKIP_TXT) | {"predictions", "rental_A_census", "nominal_ms at the fallback rung 1305 (page, label, n, arm, ms)",
                               "at the fallback rung 1305", "units", "pages"}


def name_of(part: str) -> str:
    return A2.TILESOURCE if part == "tilesource" else A2.NAMES_A2[part]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def txt(part: str, d: dict) -> list[str]:
    who = ("tile rental A2" if part in A2.PARTS_A2 else "tile rental B2 (frozen; finalised mechanically)" if part in A2.PARTS_B2
           else "the decode stage (an addendum)")
    L = [f"REGISTERED {d['registered']}: {who}, {d['name']}", f"status: {d['status']}", f"design: {d['design']}",
         f"tool: {d.get('tool', '-')}", ""]
    if part in RT.TEXT:
        L += RT.TEXT[part](d) + [""]
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool"):
            continue
        RA._flat("", {k: v}, L)
    return L


def build(only=None) -> dict:
    parts = [p for p in PARTS if only is None or p in only]
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
    ap.add_argument("--only", nargs="+", choices=PARTS, default=None)
    ap.add_argument("--census-score", type=Path, default=None,
                    help="A2's census.score.json: every occupancy is read from it (the mechanical rebuild)")
    a = ap.parse_args(argv)
    if a.census_score is not None:
        TC.use_census(a.census_score)
    docs = build(a.only)
    bad = 0
    for part, doc in docs.items():
        jpath = a.out_dir / f"{name_of(part)}.json"
        tpath = a.out_dir / f"{name_of(part)}.txt"
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
