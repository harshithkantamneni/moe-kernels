#!/usr/bin/env python
"""Writes rental PC2's synthetic shapes and pages (registration 2026-11-10-pc2-gh200), before any PC2 page.

    python scripts/scoring/pc2/gen_pc2_pages.py --write     # docs/registered/2026-11-10-pc2-shapes.json, -pc2-pages/*.json
    python scripts/scoring/pc2/gen_pc2_pages.py --check     # regenerate and compare byte for byte (exit 1 on a difference)

The cells are pc2common's blocks (D, D3, S, W, N, R: moe-kernels/pc2-page/1). Every page is checked as the probe will
check it (pc2common.check_page_doc: every cell, and block N's twins and pairs) before it is written; every write goes
through a temporary file and os.replace. Nothing here is a kernel: routing, token counts and configs only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(HERE), str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import pc2common as P  # noqa: E402


def files() -> dict:
    """{repo-relative path: text} of every file this generator owns."""
    out = {P.SHAPES_FILE: P.shapes_text()}
    out.update({f"{P.PAGES_DIR}/{n}": t for n, t in P.page_files().items()})
    return out


def validate(texts: dict) -> list[str]:
    """Every page parsed as the probe will (pc2common.check_page_doc)."""
    shapes = P.check_shapes_doc(json.loads(texts[P.SHAPES_FILE]))
    out = []
    for rel, t in sorted(texts.items()):
        if rel == P.SHAPES_FILE:
            continue
        cells = P.check_page_doc(json.loads(t), shapes)
        out.append(f"{rel}: {len(cells)} cells")
    return out


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    want = files()
    for line in validate(want):
        print(line)
    if a.check:
        bad = [rel for rel, t in want.items() if not (REPO / rel).is_file() or (REPO / rel).read_text() != t]
        pdir = REPO / P.PAGES_DIR
        extra = sorted(str(p.relative_to(REPO)) for p in pdir.glob("*.json")
                       if str(p.relative_to(REPO)) not in want) if pdir.is_dir() else []
        for rel in sorted(want):
            print(f"{rel}: {'DIFFERS' if rel in bad else 'same'}")
        for rel in extra:
            print(f"{rel}: not generated here")
        return 1 if bad or extra else 0
    for rel, t in want.items():
        atomic_write(REPO / rel, t)
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
