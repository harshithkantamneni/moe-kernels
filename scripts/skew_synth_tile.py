#!/usr/bin/env python
"""The tile stage's SYNTHETIC histogram pages at any BLOCK_M (registrations 2026-10-14-tileA-* and
2026-10-14-tileB-*; design-tile 5.6, design-tile-review 4 and 9.14). Nothing here reads or writes
a trace-derived count.

    python scripts/skew_synth_tile.py --draw     # write docs/registered/2026-10-14-tile{A,B}-hist
    python scripts/skew_synth_tile.py --check    # recompute every page, compare byte for byte

`scripts/skew_synth.py` is pinned by rental 6's registrations and is imported, never edited. Its
`tokens_for` is tied to BLOCK_M 32; here the tokens of tread n at tile BLOCK_M are
n x BLOCK_M x E / k (R3's histogram_page_from rule at the run's --block-m, which it validates).

THE RULE. One deterministic draw per (page, n, label) from rental 5's fitted shapes
(`synth_fit.json`), the parameter log-B interpolated between the bracketing fitted batches and
clamped at the ends (`skew_synth.shape_param6`, rental 6's rule unchanged). A model with no fitted
shapes carries the shapes of a fitted model with the same E (SEEN-INFORMED choices, stated per
model in SHAPE_SOURCE). seed = BASE + 10000 x tile index (BLOCK_M 64: 0, 128: 1) + 1000 x model
index + 10 n + label index; shuffle seed = BASE. Rental A's base is 20261410 and rental B's
20261420, so no draw of either repeats rental 5 to 7's.

UNIFORM AND BALANCED. A uniform cell gives every expert n x BLOCK_M rows; a balanced cell's counts
are null (R3's balanced_ids). Both are the same at every seed. A uniform or balanced cell at
BLOCK_M 64 or 128 has the same counts as an R3-tile cell at 2x or 4x the tread, so its WORKLOAD
may be SEEN (the registrations key leakage by counts, not by tread).

ONE FILE PER (model, tile, G). A byte page and a timed page of the same (model, tile, G) read the
SAME file, so their counts are identical by construction (design-tile-review 9.14).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import skew_synth as SS  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

DATE = "2026-10-14"
OUT = {"A": REPO / "docs" / "registered" / f"{DATE}-tileA-hist",
       "B": REPO / "docs" / "registered" / f"{DATE}-tileB-hist"}
BASE = {"A": 20261410, "B": 20261420}
#: the model order the seed's model index reads
MODELS = ("mixtral-8x7b", "olmoe-1b-7b", "qwen2-57b-a14b", "phi-3.5-moe", "qwen1.5-moe-a2.7b",
          "jetmoe-8b", "deepseek-v2-lite")
#: whose fitted shapes each model draws (synth_fit.json fits Mixtral, OLMoE and Qwen1.5 only)
SHAPE_SOURCE = {
    "mixtral-8x7b": ("mixtral-8x7b", "its own fitted shapes"),
    "olmoe-1b-7b": ("olmoe-1b-7b", "its own fitted shapes"),
    "qwen2-57b-a14b": ("olmoe-1b-7b", "OLMoE's (the same E 64 and top_k 8; SEEN-INFORMED)"),
    "phi-3.5-moe": ("mixtral-8x7b", "Mixtral's (rental 6 owner decision 4, carried)"),
    "qwen1.5-moe-a2.7b": ("qwen1.5-moe-a2.7b", "its own fitted shapes"),
    "jetmoe-8b": ("mixtral-8x7b", "Mixtral's (the same E 8 and top_k 2; SEEN-INFORMED)"),
    "deepseek-v2-lite": ("olmoe-1b-7b", "OLMoE's (the same E 64; SEEN-INFORMED)"),
}
TILE_INDEX = {64: 0, 128: 1}
#: rental A draws one PT and one PW per tread; rental B two of each (design-tile-review 4)
LABELS = {"A": ("PT", "PW"), "B": ("PT", "PW", "PT2", "PW2")}
LABEL_INDEX = {"PT": 0, "PW": 1, "PT2": 2, "PW2": 3}
ARMS = ("native", "shared")

#: (rental, model, BLOCK_M, G, treads, page name). Every tread is vLLM-consistent: vLLM's
#: default config at M = n x BLOCK_M x E / k is this tile and this G (tests hold it). No G-1
#: tread reaches n 6: the byte model (M4) rules w2 at G 1 and n >= 6 ILL-POSED on Hopper
#: (wave_split_bytes.HOPPER_G1_N), so such a cell would have no v4 prediction (the design's
#: OLMoE n 8 is dropped for it; n 1, 2 and 4 remain).
PAGES = (
    ("A", "olmoe-1b-7b", 128, 1, (1, 2, 4), "olmoe-1b-7b-128g1"),
    ("A", "mixtral-8x7b", 64, 1, (1, 2), "mixtral-8x7b-64g1"),
    ("A", "mixtral-8x7b", 128, 16, (3, 4, 8), "mixtral-8x7b-128g16"),
    ("A", "mixtral-8x7b", 128, 1, (2,), "mixtral-8x7b-128g1"),
    ("A", "qwen2-57b-a14b", 128, 1, (1, 2), "qwen2-57b-a14b-128g1"),
    ("A", "olmoe-1b-7b", 128, 1, (1,), "olmoe-1b-7b-dead"),
    ("B", "phi-3.5-moe", 64, 1, (1,), "phi-3.5-moe-64g1"),
    ("B", "phi-3.5-moe", 128, 1, (1, 2), "phi-3.5-moe-128g1"),
    ("B", "phi-3.5-moe", 128, 16, (3, 4), "phi-3.5-moe-128g16"),
    ("B", "qwen1.5-moe-a2.7b", 128, 1, (1, 2, 4), "qwen1.5-moe-a2.7b-128g1"),
    ("B", "qwen1.5-moe-a2.7b", 128, 16, (5, 6), "qwen1.5-moe-a2.7b-128g16"),
    ("B", "jetmoe-8b", 128, 16, (3, 4), "jetmoe-8b-128g16"),
    ("B", "jetmoe-8b", 128, 1, (2,), "jetmoe-8b-128g1"),
    ("B", "deepseek-v2-lite", 128, 1, (3,), "deepseek-v2-lite-128g1"),
)
#: the dead-lever page has the controls only (its increment does not depend on the draw)
CONTROLS_ONLY = {"olmoe-1b-7b-dead"}


def tokens_at(model: str, n: int, block_m: int) -> int:
    """n x BLOCK_M x E / k, refused when not whole (R3 refuses it too)."""
    cfg = MODEL_CONFIGS[model]
    rows = n * block_m * cfg.num_experts
    if rows % cfg.top_k:
        raise ValueError(f"{model} n={n} at BLOCK_M {block_m}: {rows} / {cfg.top_k} is not a whole token count")
    return rows // cfg.top_k


def seed_of(rental: str, model: str, block_m: int, n: int, label: str) -> int:
    return BASE[rental] + 10000 * TILE_INDEX[block_m] + 1000 * MODELS.index(model) + 10 * n + LABEL_INDEX[label]


def page(rental: str, model: str, block_m: int, G: int, ns, name: str, fitd: dict) -> dict:
    E = MODEL_CONFIGS[model].num_experts
    src, why = SHAPE_SOURCE[model]
    cells = []
    for n in ns:
        B = tokens_at(model, n, block_m)
        if name not in CONTROLS_ONLY:
            for lab in LABELS[rental]:
                shape = lab.rstrip("2")
                kind, prm, rule = SS.shape_param6(fitd, src, shape, B)
                seed = seed_of(rental, model, block_m, n, lab)
                c = [int(v) for v in SS._draw(kind, prm, model, B, seed)]
                st = SS.hist_stats(c)
                cells.append(SS._cell(lab, n, c, ARMS, kind=kind, param=prm, seed=seed, B=B, source_model=src,
                                      source_shape=shape, rule=rule, stats={s: round(v, 6) for s, v in st.items()}))
        cells += [SS._cell("uniform", n, [n * block_m] * E, ARMS), SS._cell("balanced", n, None, ARMS)]
    return {"schema": SS.SCHEMA, "model": model, "page": name, "shuffle_seed": BASE[rental],
            "block_m": block_m, "group_m": G, "cells": cells,
            "synthetic": f"fitted Zipf/Dirichlet shapes of {src} ({why}); scripts/skew_synth_tile.py, tile stage "
                         f"rental {rental}; no trace count"}


def pages(fitd: dict | None = None, rental: str | None = None) -> dict:
    fitd = fitd if fitd is not None else json.loads(SS.FIT.read_text())
    return {(r, name): page(r, m, bm, G, ns, name, fitd) for r, m, bm, G, ns, name in PAGES
            if rental is None or r == rental}


def path_of(rental: str, name: str) -> Path:
    return OUT[rental] / f"{name}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--draw", action="store_true", help="write docs/registered/2026-10-14-tile{A,B}-hist")
    g.add_argument("--check", action="store_true", help="recompute every page, compare byte for byte")
    a = ap.parse_args(argv)
    bad = 0
    for (r, name), doc in pages().items():
        p = path_of(r, name)
        if a.draw:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(SS.dump(doc))
            print(f"wrote {p}")
        else:
            same = p.exists() and p.read_text() == SS.dump(doc)
            print(f"{p.name}: {'same' if same else 'DIFFERS'}")
            bad += not same
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
