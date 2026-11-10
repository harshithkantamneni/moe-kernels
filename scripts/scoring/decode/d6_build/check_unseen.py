#!/usr/bin/env python
"""The new blind models stay unseen for decode (addendum 6): (1) d4common.no_decode_page for each; (2) every published
session dated 2026-11-04 or later scanned for the models' names in file names and text; (3) PC's and PC2's registered
synthetic shapes against the reserved (E, H, F, k).

    python scripts/scoring/decode/d6_build/check_unseen.py [--write | --check]"""
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import d4common as D4  # noqa: E402
import d6v as L2  # noqa: E402

from moe.spec import MODEL_CONFIGS as MC  # noqa: E402

REPO = L2.REPO
M = D4.NEW_BLIND
PAT = re.compile(r"8x22b|qwen2-57b|qwen2_57b|granite-3\.0-1b|granite_3_0_1b|1b-a400m", re.I)
UNTIL = "2026-11-10-nvidia_gh200_480gb-pc2-session"


def lines() -> list[str]:
    L = []
    for m in M:
        d = D4.no_decode_page(m)
        L.append(f"{m}: no_decode_page -> {d['verdict']} {d['registered_pages'] or ''}{d['published_dirs'] or ''}")
    pub = REPO / "results/published"
    later = sorted(p for p in pub.iterdir() if p.is_dir() and "2026-11-04" <= p.name[:10] and p.name <= UNTIL)
    L.append("published sessions dated 2026-11-04 to PC2's: " + ", ".join(p.name for p in later))
    for s in later:
        names, text = [], {}
        for f in sorted(s.rglob("*")):
            rel = str(f.relative_to(s))
            if PAT.search(rel):
                names.append(rel)
            if f.is_file() and f.suffix in (".json", ".txt", ".log", ".env", ".plan", ".yaml", ".csv") and f.stat().st_size < 50e6:
                for mm in PAT.finditer(f.read_text(errors="replace")):
                    text.setdefault(rel, set()).add(mm.group(0).lower())
        L.append(f"  {s.name}: file names naming a reserved model: {len(names)}; text files with a hit: {len(text)}")
        L += [f"     {k}: {sorted(v)}" for k, v in sorted(text.items())[:8]]
    shapes = {(MC[m].num_experts, MC[m].hidden_size, MC[m].intermediate_size, MC[m].top_k) for m in M}
    for f in ("2026-11-06-pc-shapes.json", "2026-11-10-pc2-shapes.json"):
        pcs = json.loads((REPO / "docs/registered" / f).read_text())["shapes"]
        hits = [x["name"] for x in pcs if (x["E"], x["H"], x["F"], x["k"]) in shapes]
        L.append(f"{f}: registered synthetic shapes equal to a reserved model's (E, H, F, k): {hits or 'none'}")
    return L


if __name__ == "__main__":
    sys.exit(L2.emit(lines(), "check_unseen", sys.argv[1:]))
