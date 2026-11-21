#!/usr/bin/env python
"""The calibration rental C4's histogram page (docs/registered/2026-10-27-cal4-hist/): DRAWN by the tile stage's
registered generator, never from a trace count.

    python scripts/scoring/cal4/hist_c4.py --write     # write the page
    python scripts/scoring/cal4/hist_c4.py --check     # recompute it, compare byte for byte (exit 1 on a difference)

  mixtral-8x7b-tp2-128g1.json   C4-T (tp2 t128g1) and C4-B (tp2 b128g1): Mixtral-8x7B's tensor-parallel-2 shard
                                (moe/spec.py MODEL_CONFIGS "mixtral-8x7b-tp2": H 4096, I 7168, E 8, top_k 2) at
                                128x128 G 1, tread 2, labels PT, PW, PT2, PW2 (one deterministic draw each), uniform and
                                balanced; arms NATIVE and SHARED. A timed page and a byte page of one (model, tile, G)
                                read the SAME file, so their counts are identical by construction (design-tile-review
                                9.14).

THE GENERATOR. scripts/skew_synth_tile.py (pinned by the tileA and tileB registrations) is imported, never edited;
its `page` builds the cells with rental 6's rule unchanged (skew_synth.shape_param6 on rental 5's fitted shapes,
synth_fit.json, the parameter log-B interpolated between the bracketing fitted batches; skew_synth._draw). Four of
its tables are extended IN MEMORY for this one call, as hist_c2 did, and restored after it:
  MODELS        + mixtral-8x22b (index 7, C2's in-memory index) + mixtral-8x7b-tp2 (index 8): no model index repeats
                an earlier rental's
  SHAPE_SOURCE  mixtral-8x7b-tp2 draws Mixtral-8x7B's own fitted shapes: a tensor-parallel shard splits each expert's
                intermediate width and leaves the router and E untouched, so its routing IS Mixtral-8x7B's
  BASE, LABELS  rental "C4": base 20261450 (A 20261410, B 20261420, C2 20261440), rental B's four skewed labels
seed = 20261450 + 10000 (BLOCK_M 128) + 8000 (model index 8) + 10 n + label index: 20279470 to 20279473 at n 2.

THE TREAD. n 2 is the only tread at which vLLM's default config (moe.bench.tile_resolve.default_config, v0.27.1; no
tuned file ships for E 8, N 7168) for tp2 (E 8, top_k 2) is 128x128x64 w8 s3 G 1: M = n x 128 x 8 / 2 = 512 n tokens;
n 1 (M 512) is BLOCK_M 64, n 3 (M / E 192 > 128) is G 16. It is Mixtral-8x7B's E1-CAL tread and 8x22B's C2 tread.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import skew_synth_tile as ST  # noqa: E402  (pinned; imported, never edited)

OUT = REPO / "docs" / "registered" / "2026-10-27-cal4-hist"
RENTAL = "C4"
BASE = 20261450
MODEL = "mixtral-8x7b-tp2"
#: C2's in-memory model at index 7, kept so tp2's index (8) is new
C2_MODEL = "mixtral-8x22b"
INDEX = 8
SHAPE = ("mixtral-8x7b", "Mixtral-8x7B's own fitted shapes: a TP shard keeps the model's router and E")
LABELS = ("PT", "PW", "PT2", "PW2")
#: page name: (model, block_m, group_m, treads, role)
PAGES = {
    "mixtral-8x7b-tp2-128g1": (MODEL, 128, 1, (2,), "C4-T and C4-B, Mixtral-8x7B-tp2 128x128 G 1 (the fifth CAL model)"),
}


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def path_of(name: str) -> Path:
    return OUT / f"{name}.json"


@contextlib.contextmanager
def c4_tables():
    """skew_synth_tile's MODELS, SHAPE_SOURCE, BASE and LABELS extended for rental C4, restored on exit."""
    keep = (ST.MODELS, dict(ST.SHAPE_SOURCE), dict(ST.BASE), dict(ST.LABELS))
    try:
        add = tuple(m for m in (C2_MODEL, MODEL) if m not in ST.MODELS)
        ST.MODELS = tuple(ST.MODELS) + add
        ST.SHAPE_SOURCE[MODEL] = SHAPE
        ST.BASE[RENTAL] = BASE
        ST.LABELS[RENTAL] = LABELS
        yield
    finally:
        ST.MODELS = keep[0]
        ST.SHAPE_SOURCE.clear(); ST.SHAPE_SOURCE.update(keep[1])
        ST.BASE.clear(); ST.BASE.update(keep[2])
        ST.LABELS.clear(); ST.LABELS.update(keep[3])


def build(name: str) -> str:
    model, bm, G, treads, role = PAGES[name]
    fitd = json.loads(ST.SS.FIT.read_text())
    with c4_tables():
        assert ST.MODELS.index(C2_MODEL) == 7 and ST.MODELS.index(MODEL) == INDEX, ST.MODELS
        doc = ST.page(RENTAL, model, bm, G, treads, name, fitd)
    doc["synthetic"] = (f"fitted Zipf/Dirichlet shapes of {SHAPE[0]} ({SHAPE[1]}); scripts/skew_synth_tile.py's page "
                        f"(sha256 {sha256(REPO / 'scripts' / 'skew_synth_tile.py')}) with rental C4's base {BASE} and "
                        f"model index {INDEX}, scripts/scoring/cal4/hist_c4.py: {role}; no trace count")
    return ST.SS.dump(doc)


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
