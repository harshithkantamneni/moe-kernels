#!/usr/bin/env python
"""Rental 7's SYNTHETIC histogram pages (registration 2026-10-13-rental7-skew and
2026-10-13-rental7-l2open; design-r7 S10, design-r7-review (c) and (e), owner decisions 3 and 5
of 2026-10-13). Nothing here reads or writes a trace-derived count.

    python scripts/skew_synth7.py --draw7     # write docs/registered/2026-10-13-rental7-skew-hist
    python scripts/skew_synth7.py --check7    # recompute every page, compare byte for byte

scripts/skew_synth.py is pinned by rental 6's registrations and is imported, never edited.

THE SKEW PAGES (owner decision 5: fresh seeds on the same treads). mixtral-8x7b-{A,B,C} and
phi-3.5-moe-{A,B,C} are rental 6's rule exactly (skew_synth.pages6: the same fitted shapes, the
same log-B interpolation, the treads Mixtral 3 6 16 and Phi 2 4 12, Phi carrying Mixtral's
shapes, the same pages and arms), drawn at DRAW7_SEED_BASE instead of rental 6's 20261010:
seed = DRAW7_SEED_BASE + 1000 x shape index + 10 n + label index, the shape index rental 6's
(Mixtral 0, OLMoE 1, Phi 2), so seed(n) here is rental 6's seed(n + 1), a tread rental 6 never
drew; shuffle seed SHUFFLE_SEED7. The uniform and balanced cells do not depend on the seed: their
counts are rental 6's (SEEN replicates); only the token-row shuffle of the uniform cells moves.

THE L2OPEN PAGE (owner decision 3: in). olmoe-1b-7b-l2open is the OLMoE w1 byte page: PT, PW,
uniform and balanced at n 4, 8 and 16, NATIVE and SHARED at n <= 9 and NATIVE only at n 16 (the
byte pages' rule since rental 5). Its draws are the ones design-r7's olmoe_w1.py priced (the
review's (e): the page the build emits must be the page that was priced): seed =
L2OPEN_SEED_BASE + 10 n + label index (PT 0, PW 1; no shape index), shuffle seed
L2OPEN_SEED_BASE. The build prices EVERY row of this exact page before any page runs
(scripts/scoring/rental7/reg7.py).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import skew_synth as SS  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

OUT7 = REPO / "docs" / "registered" / "2026-10-13-rental7-skew-hist"
DRAW7_SEED_BASE = 20261020
SHUFFLE_SEED7 = 20261020
MODELS7 = ("mixtral-8x7b", "phi-3.5-moe")
PAGES7 = ("A", "B", "C")
L2OPEN_MODEL = "olmoe-1b-7b"
L2OPEN_SEED_BASE = 20261030
L2OPEN_TREADS = (4, 8, 16)
L2OPEN_LABELS = ("PT", "PW")


def draw7_seed(model: str, n: int, label: str) -> int:
    """The registered seed of one rental-7 skew draw (rental 6's rule at DRAW7_SEED_BASE)."""
    return DRAW7_SEED_BASE + 1000 * list(SS.SHAPES6).index(model) + 10 * n + SS.LABELS6.index(label)


def l2open_seed(n: int, label: str) -> int:
    """The registered seed of one l2open draw: L2OPEN_SEED_BASE + 10 n + label index."""
    return L2OPEN_SEED_BASE + 10 * n + SS.LABELS6.index(label)


@contextlib.contextmanager
def _rental6_rule_at(base: int, shuffle: int):
    """skew_synth.pages6 with its two registered seeds moved, restored on exit: the rule is
    pages6's own code, so nothing about the draw is restated here."""
    old = (SS.DRAW6_SEED_BASE, SS.SHUFFLE_SEED6)
    SS.DRAW6_SEED_BASE, SS.SHUFFLE_SEED6 = base, shuffle
    try:
        yield
    finally:
        SS.DRAW6_SEED_BASE, SS.SHUFFLE_SEED6 = old


def skew_pages(fitd: dict) -> dict:
    with _rental6_rule_at(DRAW7_SEED_BASE, SHUFFLE_SEED7):
        p6 = SS.pages6(fitd)
    out = {f"{m}-{p}": p6[f"{m}-{p}"] for m in MODELS7 for p in PAGES7}
    for name, doc in out.items():
        for c in doc["cells"]:
            prov = c.get("provenance")
            if prov:
                assert prov["seed"] == draw7_seed(doc["model"], c["n"], c["label"]), (name, c["label"], c["n"])
    return out


def l2open_page(fitd: dict) -> dict:
    model = L2OPEN_MODEL
    E = MODEL_CONFIGS[model].num_experts
    both = ("native", "shared")
    cells = []
    for n in L2OPEN_TREADS:
        B = SS.tokens_for(model, n)
        arms = both if n <= 9 else ("native",)
        for lab in L2OPEN_LABELS:
            kind, prm, rule = SS.shape_param6(fitd, model, lab, B)
            seed = l2open_seed(n, lab)
            c = [int(v) for v in SS._draw(kind, prm, model, B, seed)]
            st = SS.hist_stats(c)
            cells.append(SS._cell(lab, n, c, arms, kind=kind, param=prm, seed=seed, B=B, source_model=model,
                                  source_shape=lab, rule=rule, stats={s: round(v, 6) for s, v in st.items()}))
        cells += [SS._cell("uniform", n, [n * SS.BLOCK_M] * E, arms), SS._cell("balanced", n, None, arms)]
    return {"schema": SS.SCHEMA, "model": model, "page": "l2open", "shuffle_seed": L2OPEN_SEED_BASE,
            "cells": cells, "synthetic": "fitted Zipf/Dirichlet shapes (scripts/skew_synth7.py, rental 7 l2open "
                                          "draws, the design's priced draws); no trace count"}


def pages7(fitd: dict | None = None) -> dict:
    fitd = fitd if fitd is not None else json.loads(SS.FIT.read_text())
    out = skew_pages(fitd)
    out[f"{L2OPEN_MODEL}-l2open"] = l2open_page(fitd)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--draw7", action="store_true", help=f"write {OUT7.relative_to(REPO)}")
    g.add_argument("--check7", action="store_true", help="recompute every rental-7 page, compare")
    a = ap.parse_args(argv)
    OUT7.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, doc in pages7().items():
        p = OUT7 / f"{name}.json"
        if a.draw7:
            p.write_text(SS.dump(doc))
            print(f"wrote {p}")
        else:
            same = p.exists() and p.read_text() == SS.dump(doc)
            print(f"{p.name}: {'same' if same else 'DIFFERS'}")
            bad += not same
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
