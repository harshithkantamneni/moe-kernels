#!/usr/bin/env python
"""Tile rental A2's histogram pages (docs/registered/2026-10-20-tileA2-hist/): cells COPIED, never drawn.

    python scripts/scoring/tileA2/hist_a2.py --write     # write the five pages
    python scripts/scoring/tileA2/hist_a2.py --check     # recompute them, compare byte for byte (exit 1 on a difference)

  mixtral-8x7b-128g16-probe.json   the lock-hold probe (tileA2-gates; REVIEW-A2 1c): Mixtral 128x128 G 16,
                                   treads 3 and 8, every label, NATIVE and SHARED, from
                                   2026-10-14-tileA-hist/mixtral-8x7b-128g16.json (rental A's worst page:
                                   n 8 the most non-clean repeats, 34 of 72; n 3 the deepest read, 1365)
  qwen2-57b-a14b-128g1-probe.json  Qwen2-57B 128x128 G 1, tread 2, from 2026-10-14-tileA-hist/qwen2-57b-a14b-128g1.json
  convl-mixtral-8x7b-g1.json       CONV-L's G-1 page (tileA2-convl): R3's tile (BLOCK_M 32), tread 3, the four
                                   labels, NATIVE only, from 2026-10-13-rental7-skew-hist/mixtral-8x7b-A.json
                                   (at G 1 v3's bytes are ILL-POSED from tread 6 on Hopper: n 3 is the page's
                                   only G-1 tread)
  convl-olmoe-1b-7b-g1.json        CONV-L's short G-1 page: OLMoE at R3's tile, tread 3 (about 0.64 ms, T0 about
                                   7% of it), PT, PW and uniform, NATIVE only, from 2026-10-09-rental6-skew-hist/
                                   olmoe-1b-7b-A.json (v3 prices no G-1 balanced OLMoE cell there)
  convl-mixtral-8x7b-g8.json       CONV-L's G-8 page: the Mixtral source, treads 6 and 16, NATIVE only

Every copied cell keeps its label, n and counts (so its counts sha256) and its page's shuffle seed (R3's
token-row shuffle is a function of the seed and the token count only, so a copied cell realises the
same ids as on its source page); a CONV-L cell's arms are narrowed to ["native"] (R3 refuses a cell
asking an arm its run does not run). Each page names its source and the source's sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "docs" / "registered" / "2026-10-20-tileA2-hist"
SRC_A = "docs/registered/2026-10-14-tileA-hist"
SRC_R7 = "docs/registered/2026-10-13-rental7-skew-hist"
SRC_R6 = "docs/registered/2026-10-09-rental6-skew-hist"
ALL = ("PT", "PW", "uniform", "balanced")
#: page name: (source file, treads kept, labels kept, arms override or None, block_m, group_m, role)
PAGES = {
    "mixtral-8x7b-128g16-probe": (f"{SRC_A}/mixtral-8x7b-128g16.json", (3, 8), ALL, None, 128, 16,
                                  "the lock-hold probe (tileA2-gates)"),
    "qwen2-57b-a14b-128g1-probe": (f"{SRC_A}/qwen2-57b-a14b-128g1.json", (2,), ALL, None, 128, 1,
                                   "the lock-hold probe (tileA2-gates)"),
    "convl-mixtral-8x7b-g1": (f"{SRC_R7}/mixtral-8x7b-A.json", (3,), ALL, ["native"], 32, 1,
                              "CONV-L's G-1 DRAM-bound page (tileA2-convl)"),
    "convl-olmoe-1b-7b-g1": (f"{SRC_R6}/olmoe-1b-7b-A.json", (3,), ("PT", "PW", "uniform"), ["native"], 32, 1,
                             "CONV-L's G-1 short page, T0 about 7% of the cell (tileA2-convl)"),
    "convl-mixtral-8x7b-g8": (f"{SRC_R7}/mixtral-8x7b-A.json", (6, 16), ALL, ["native"], 32, 8,
                              "CONV-L's G-8 c-heavy page (tileA2-convl)"),
}
PROBE_PAGES = ("mixtral-8x7b-128g16-probe", "qwen2-57b-a14b-128g1-probe")
CONVL_PAGES = ("convl-mixtral-8x7b-g1", "convl-olmoe-1b-7b-g1", "convl-mixtral-8x7b-g8")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def path_of(name: str) -> Path:
    return OUT / f"{name}.json"


def build(name: str) -> str:
    src, treads, labels, arms, bm, G, role = PAGES[name]
    doc = json.loads((REPO / src).read_text())
    cells = []
    for c in doc["cells"]:
        if int(c["n"]) in treads and c["label"] in labels:
            c = dict(c)
            if arms is not None:
                c["arms"] = list(arms)
            cells.append(c)
    out = {"schema": doc["schema"], "model": doc["model"], "page": name, "shuffle_seed": doc["shuffle_seed"],
           "block_m": bm, "group_m": G, "cells": cells,
           "synthetic": f"COPIED cells (treads {', '.join(map(str, treads))}; labels {', '.join(labels)}) of {src}, sha256 {sha256(REPO / src)}: "
                        f"{role}; tile rental A2, scripts/scoring/tileA2/hist_a2.py"}
    return json.dumps(out, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    bad = 0
    for name in PAGES:
        text = build(name)
        p = path_of(name)
        if a.write:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            print(f"wrote {p.relative_to(REPO)}")
        else:
            same = p.exists() and p.read_text() == text
            bad += not same
            print(f"{p.name}: {'same' if same else 'DIFFERS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
