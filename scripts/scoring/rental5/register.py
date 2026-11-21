#!/usr/bin/env python
"""Writes rental 5's six registered JSON files (docs/registered/2026-10-07-rental5-*).

    python scripts/scoring/rental5/register.py [--out-dir docs/registered] [--check] [--only PART ...]

Every number is computed here from committed files (the timing model with v2, M2 and M3, the
byte model, the published pages, the synthetic histogram pages, the instrumented copy's and
the tools' own constants) or is a typed-in input with its provenance and its label (CAL,
CAL-counters, SEEN, SEEN-fitted, BLIND, BLIND-CALMODEL). `--check` recomputes and compares
with the committed JSON and text (exit 1 on any difference) instead of writing; the
`code_pins` block is excluded from the comparison, as in rentals 3 and 4.

Design: scratchpad design-r5/DESIGN.md, corrected by design-r5-review/REVIEW.md (the later
document wins), with the owner's decisions of 2026-10-07: 1 G3 a TOST on the pooled
shuffled-minus-balanced difference with a cluster-t interval, margin 0.36%, Cochran's Q
gating, every per-cell difference kept, the model's shuffle effect per cell, the permutation
cells; 2 dead- and live-CTA stamp sampling at 1 in 17 under the CTAs/SM-only gate, regcheck
right after calibrate; 3 SYNTHETIC skew histograms (Zipf / Dirichlet fitted to the traces'
statistics, deterministic draws, published openly, no trace-derived count in the repo or on
the VM).

LEAKAGE, said once for all six: every published page is SEEN, and every constant typed in
was fitted or chosen after seeing published pages (labelled per entry). No rental-5 page
exists when this runs. A file's `seen_data` names what of it is SEEN.

The parts are built by reg_levers (v2, secondk, bk128; tp2 CUT by the owner, 2026-10-07), reg_skew (skew, c15) and
reg_stamps2 (stamps2); this file renders, pins and checks them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5common as C5  # noqa: E402
import reg_levers as RL  # noqa: E402
import reg_skew as RS  # noqa: E402
import reg_stamps2 as RT  # noqa: E402

REPO = C5.REPO
REG = REPO / "docs" / "registered"
BUILDERS = {**RL.BUILDERS, **RS.BUILDERS, **RT.BUILDERS}
TEXT = {**getattr(RL, "TEXT", {}), **getattr(RS, "TEXT", {}), **getattr(RT, "TEXT", {})}
PINNED = ("scripts/r3_timing_model.py", "scripts/wave_split_bytes.py", "scripts/private_weight_reference.py",
          "scripts/dram_counter_route.py", "scripts/instr_probe.py", "scripts/skew_synth.py",
          "scripts/cross_model_score.py", "scripts/cross_model_predict.py",
          "moe/instrumented/__init__.py", "moe/instrumented/fused_moe_instr.py", "moe/instrumented/cubin.py",
          "scripts/scoring/rental5/r5common.py", "scripts/scoring/rental5/skewmodel.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _flat(prefix, v, out, depth=0):
    if isinstance(v, dict):
        for k, x in v.items():
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


#: blocks too long for the text file (they are in the JSON)
SKIP_TXT = {"code_pins", "predictions", "byte_predictions", "laws_on_seen_points", "rows_expected"}


def txt(part: str, d: dict) -> list[str]:
    L = [f"REGISTERED {d['registered']}: rental 5, {d['name']}", f"status: {d['status']}",
         f"design: {d['design']}", f"tool: {d.get('tool', '-')}", ""]
    if part in TEXT:
        L += TEXT[part](d) + [""]
    for k, v in d.items():
        if k in SKIP_TXT or k in ("registered", "name", "status", "design", "tool"):
            continue
        _flat("", {k: v}, L)
    return L


def build(only=None) -> dict:
    parts = [p for p in C5.PARTS if only is None or p in only]
    docs = {p: BUILDERS[p]() for p in parts}
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
    ap.add_argument("--only", nargs="+", choices=C5.PARTS, default=None)
    a = ap.parse_args(argv)
    docs = build(a.only)
    bad = 0
    for part, doc in docs.items():
        jpath = a.out_dir / f"{C5.NAMES[part]}.json"
        tpath = a.out_dir / f"{C5.NAMES[part]}.txt"
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
