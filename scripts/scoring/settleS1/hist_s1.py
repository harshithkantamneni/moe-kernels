#!/usr/bin/env python
"""The settle rental S1's histogram pages (docs/registered/2026-10-21-settleS1-hist/): cells COPIED, never drawn.

    python scripts/scoring/settleS1/hist_s1.py --write     # write the five pages
    python scripts/scoring/settleS1/hist_s1.py --check     # recompute them, compare byte for byte (exit 1 on a difference)

Every page copies cells of rental A's registered E1-CAL histogram pages (docs/registered/2026-10-14-tileA-hist), the
pages A2's E1-CAL and byte pages ran, and narrows each cell's arms to ["native"]: R3 refuses a cell that asks an arm
its run does not run (private_weight_reference.cell_arms), and every S1 histogram unit runs NATIVE only.

  olmoe-1b-7b-128g1-rp.json      S1-M1 (replay contrast): OLMoE 128x128 G 1, n 1 PT and n 4 PT
  qwen2-57b-a14b-128g1-rp.json   S1-M1: Qwen2-57B 128x128 G 1, n 2 PT
  mixtral-8x7b-128g1-n.json      S1-M3 (Mixtral 128 G 1 bytes): every cell (n 2 PT, PW, uniform, balanced)
  qwen2-57b-a14b-128g1-n.json    S1-M5 (stall reasons): every cell (n 1 and 2, four labels)
  olmoe-1b-7b-128g1-n.json       S1-M5: every cell (n 1, 2 and 4, four labels)

Each copied cell keeps its label, n and counts (so its counts sha256) and its page's shuffle seed: R3's token-row
shuffle is seeded per cell by the page's seed and the token count only (histogram_ids), so a copied cell realises the
same ids as on its source page and on A2's pages. Each page names its source and the source's sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "docs" / "registered" / "2026-10-21-settleS1-hist"
SRC = "docs/registered/2026-10-14-tileA-hist"
ALL = ("PT", "PW", "uniform", "balanced")
#: page name: (source file, treads kept, labels kept, block_m, group_m, role)
PAGES = {
    "olmoe-1b-7b-128g1-rp": (f"{SRC}/olmoe-1b-7b-128g1.json", (1, 4), ("PT",), 128, 1,
                             "S1-M1, the replay contrast (application against kernel replay)"),
    "qwen2-57b-a14b-128g1-rp": (f"{SRC}/qwen2-57b-a14b-128g1.json", (2,), ("PT",), 128, 1,
                                "S1-M1, the replay contrast (application against kernel replay)"),
    "mixtral-8x7b-128g1-n": (f"{SRC}/mixtral-8x7b-128g1.json", (2,), ALL, 128, 1,
                             "S1-M3, Mixtral 128 G 1 counted bytes"),
    "qwen2-57b-a14b-128g1-n": (f"{SRC}/qwen2-57b-a14b-128g1.json", (1, 2), ALL, 128, 1,
                               "S1-M5, stall reasons along x"),
    "olmoe-1b-7b-128g1-n": (f"{SRC}/olmoe-1b-7b-128g1.json", (1, 2, 4), ALL, 128, 1,
                            "S1-M5, stall reasons along x"),
}
ARMS = ["native"]


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def path_of(name: str) -> Path:
    return OUT / f"{name}.json"


def build(name: str) -> str:
    src, treads, labels, bm, G, role = PAGES[name]
    doc = json.loads((REPO / src).read_text())
    if int(doc["block_m"]) != bm or int(doc["group_m"]) != G:
        raise ValueError(f"{src}: block_m {doc['block_m']} G {doc['group_m']}, not {bm} / {G}")
    cells = []
    for c in doc["cells"]:
        if int(c["n"]) in treads and c["label"] in labels:
            c = dict(c)
            c["arms"] = list(ARMS)
            cells.append(c)
    want = len(treads) * len(labels)
    if len(cells) != want:
        raise ValueError(f"{name}: {len(cells)} cells copied from {src}, {want} asked")
    out = {"schema": doc["schema"], "model": doc["model"], "page": name, "shuffle_seed": doc["shuffle_seed"],
           "block_m": bm, "group_m": G, "cells": cells,
           "synthetic": f"COPIED cells (treads {', '.join(map(str, treads))}; labels {', '.join(labels)}; arms narrowed to "
                        f"native) of {src}, sha256 {sha256(REPO / src)}: {role}; settle rental S1, "
                        "scripts/scoring/settleS1/hist_s1.py"}
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
