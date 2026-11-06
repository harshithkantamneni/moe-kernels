#!/usr/bin/env python
"""Writes rental PC's synthetic shapes and pages (registration 2026-11-06-pc-gh200), before any PC page.

    python scripts/scoring/pc/gen_pc_pages.py --write     # docs/registered/2026-11-06-pc-shapes.json, -pc-pages/*.json
    python scripts/scoring/pc/gen_pc_pages.py --check     # regenerate and compare byte for byte (exit 1 on a difference)

The cells are pccommon's blocks (A, C, D, R: moe-kernels/pc-page/1) and unit 16's Mixtral counter page (decode-page/1,
the pinned decode counter child's schema). Every page is checked as the probe will check it (pccommon.check_page_doc;
r3_decode.page_from for the counter page) before it is written. Nothing here is a kernel: routing, token counts and
configs only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(HERE), str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import pccommon as P  # noqa: E402


def files() -> dict:
    """{repo-relative path: text} of every file this generator owns."""
    out = {P.SHAPES_FILE: P.shapes_text()}
    out.update({f"{P.PAGES_DIR}/{n}": t for n, t in P.page_files().items()})
    return out


def validate(texts: dict) -> list[str]:
    """Every page parsed as its reader will: PC pages by pccommon.check_page_doc, the counter page by r3_decode."""
    import r3_decode as RD
    from moe.spec import MODEL_CONFIGS
    shapes = P.check_shapes_doc(json.loads(texts[P.SHAPES_FILE]))
    out = []
    for rel, t in sorted(texts.items()):
        if rel == P.SHAPES_FILE:
            continue
        doc = json.loads(t)
        if doc["schema"] == P.SCHEMA:
            cells = P.check_page_doc(doc, shapes)
            out.append(f"{rel}: {len(cells)} cells")
        else:
            page = RD.page_from(doc, MODEL_CONFIGS[doc["model"]], doc["model"])
            out.append(f"{rel}: {len(page.cells)} decode cells")
    return out


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
        (REPO / rel).parent.mkdir(parents=True, exist_ok=True)
        (REPO / rel).write_text(t)
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
