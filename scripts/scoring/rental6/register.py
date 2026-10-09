#!/usr/bin/env python
"""Writes rental 6's four registered JSON files (docs/registered/2026-10-09-rental6-*).

    python scripts/scoring/rental6/register.py [--out-dir docs/registered] [--check] [--only PART ...]

Every number is computed here from committed files (reg6.py: the timing model under v3's flag,
the byte model with MODEL M4, the published pages, the synthetic histogram pages, the row-key
rival's file) or is a typed-in input with its provenance and label. `--check` recomputes and
compares with the committed JSON and text (exit 1 on any difference) instead of writing; the
`code_pins` block is excluded from the comparison, as in rentals 3 to 5.

LEAKAGE, said once for all four: every published page is SEEN, and every constant typed in was
fitted or chosen after seeing published pages (labelled per entry). No rental-6 page exists
when this runs. A file's `seen_data` names what of it is SEEN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6common as C6  # noqa: E402
import reg6 as R6  # noqa: E402

REPO = C6.REPO
REG = REPO / "docs" / "registered"
PINNED = ("scripts/r3_timing_model.py", "scripts/wave_split_bytes.py", "scripts/r3_timing_model_v3.py",
          "scripts/wave_split_bytes_v3.py", "scripts/private_weight_reference.py",
          "scripts/dram_counter_route.py", "scripts/skew_synth.py", "scripts/lock_gate.py", "scripts/locked_r3.py",
          "scripts/scoring/rental5/skewmodel.py", "scripts/scoring/rental6/r6common.py",
          "scripts/scoring/rental6/r6model.py", "scripts/scoring/rental6/reg6.py", "scripts/scoring/rental6/rowkeys.py")
#: blocks too long for the text file (they are in the JSON)
SKIP_TXT = {"code_pins", "predictions", "byte_predictions", "align_proxy", "seen_reproduction", "power", "files"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _flat(prefix, v, out, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
            if k in SKIP_TXT or k == "pairs":
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
    L = [f"REGISTERED {d['registered']}: rental 6, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d.get('tool', '-')}", ""]
    if part in R6.TEXT:
        L += R6.TEXT[part](d) + [""]
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool"):
            continue
        _flat("", {k: v}, L)
    return L


def build(only=None) -> dict:
    parts = [p for p in C6.PARTS if only is None or p in only]
    docs = {p: R6.BUILDERS[p]() for p in parts}
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
    ap.add_argument("--only", nargs="+", choices=C6.PARTS, default=None)
    a = ap.parse_args(argv)
    docs = build(a.only)
    bad = 0
    for part, doc in docs.items():
        jpath = a.out_dir / f"{C6.NAMES[part]}.json"
        tpath = a.out_dir / f"{C6.NAMES[part]}.txt"
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
