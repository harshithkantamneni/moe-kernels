#!/usr/bin/env python
"""Writes rental 7's four registered JSON files (docs/registered/2026-10-13-rental7-*).

    python scripts/scoring/rental7/register.py [--out-dir docs/registered] [--check] [--only PART ...]

Every number is computed here from committed files (reg7.py: rental 6's pricer with model v3
unchanged, the published pages read through R3's own functions, the synthetic histogram pages of
scripts/skew_synth7.py, the row-key rival's file of rowkeys7.py) or is a typed-in input with its
provenance and label. The power numbers are computed on the CPU in float64; the build re-ran every
point on the laptop GPU (torch MPS, float32) into power7_check.json. `--check` recomputes and
compares with the committed JSON and text (exit 1 on any difference) instead of writing; the
`code_pins` block is excluded from the comparison, as in rentals 3 to 6.

LEAKAGE, said once for all four: every published page is SEEN, rental 6's included, and every
constant typed in was fitted or chosen after seeing published pages (labelled per entry). No
rental-7 page exists when this runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7common as C7  # noqa: E402
import reg7 as R7  # noqa: E402

REPO = C7.REPO
REG = REPO / "docs" / "registered"
PINNED = ("scripts/r3_timing_model.py", "scripts/wave_split_bytes.py", "scripts/r3_timing_model_v3.py",
          "scripts/wave_split_bytes_v3.py", "scripts/private_weight_reference.py", "scripts/locked_r3.py",
          "scripts/gh200_model_session.sh", "scripts/dram_counter_route.py", "scripts/lock_gate.py",
          "scripts/skew_synth.py", "scripts/skew_synth7.py", "scripts/scoring/rental5/skewmodel.py",
          "scripts/scoring/rental6/r6common.py", "scripts/scoring/rental6/r6model.py", "scripts/scoring/rental6/reg6.py",
          "scripts/scoring/rental6/rowkeys.py", "scripts/scoring/rental7/r7common.py", "scripts/scoring/rental7/reg7.py",
          "scripts/scoring/rental7/rowkeys7.py")
#: blocks too long for the text file (they are in the JSON)
SKIP_TXT = {"code_pins", "predictions", "byte_predictions", "evidence_seen", "positive_control", "power", "pairs", "files",
            "file"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _flat(prefix, v, out, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
            if k in SKIP_TXT:
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
    L = [f"REGISTERED {d['registered']}: rental 7, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d.get('tool', '-')}", ""]
    if part in R7.TEXT:
        L += R7.TEXT[part](d) + [""]
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool"):
            continue
        _flat("", {k: v}, L)
    return L


def _seed_leakage_as_of_registration() -> None:
    """The leakage keys (`reg7.published_counts_sha`: every (model, n, counts sha256) a published
    page carries) glob the whole published tree, so rental 7's own session, published after the
    registration, would turn every rental-7 draw SEEN. `--check` recomputes the registered files, so
    the set is read as of the registration, as reg7.seed_pricer reads the align proxy: a directory
    of links to the sessions dated on or before C7.DATE whose name does not carry 'rental7'.
    published_counts_sha() keeps the first set it computes."""
    import atexit
    import shutil
    import tempfile
    if R7._PUBSHA:
        return
    tmp = Path(tempfile.mkdtemp(prefix="r7-leak-asof-"))
    atexit.register(shutil.rmtree, tmp, True)
    for d in sorted(R7.PUB.iterdir()):
        if d.is_dir() and d.name[:10] <= C7.DATE and "rental7" not in d.name:
            (tmp / d.name).symlink_to(d.resolve(), target_is_directory=True)
    pub = R7.PUB
    R7.PUB = tmp
    try:
        R7.published_counts_sha()
    finally:
        R7.PUB = pub


def _held_pages_as_of_registration() -> None:
    """reg7._held_pages (the g1rule builder's input) globs every published histogram page and leaves
    out rental 7's by name, so a session published later (the tile stage's of 2026-10-20) would join
    the 17 pages the rule was registered on. `--check` recomputes the registered files, so the pages
    are read as of the registration: those of sessions dated on or before C7.DATE (each entry's name
    starts with its session's date). reg7.py (pinned) is unchanged; the wrapper is installed once."""
    if getattr(R7._held_pages, "_as_of", False):
        return
    held = R7._held_pages

    def as_of():
        return [x for x in held() if x[0][:10] <= C7.DATE]
    as_of._as_of = True
    R7._held_pages = as_of


def build(only=None) -> dict:
    parts = [p for p in C7.PARTS if only is None or p in only]
    R7.seed_pricer()
    _seed_leakage_as_of_registration()
    _held_pages_as_of_registration()
    docs = {p: R7.BUILDERS[p]() for p in parts}
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
    ap.add_argument("--only", nargs="+", choices=C7.PARTS, default=None)
    a = ap.parse_args(argv)
    docs = build(a.only)
    bad = 0
    for part, doc in docs.items():
        jpath = a.out_dir / f"{C7.NAMES[part]}.json"
        tpath = a.out_dir / f"{C7.NAMES[part]}.txt"
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
