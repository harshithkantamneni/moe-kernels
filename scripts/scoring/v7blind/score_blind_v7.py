#!/usr/bin/env python
"""Scores model v7's decode blind (registration docs/registered/2026-11-17-v7-decode-blind-gh200) on a results tree.

    python scripts/scoring/v7blind/score_blind_v7.py <results tree> <out dir> [--allow-draft]

It reads the registration at its committed path only (docs/registered/2026-11-17-v7-decode-blind-gh200.json; no option
names another file, as gate_v7.py) and records that file's sha256 and the checkout's commit in the score.

A wrapper over the addendum-3 blind wrapper (score_blind_d3.score, which runs score_blind_d2.score and the frozen
score_blind.score), unchanged, as score_blind_d6.py wraps it, with three swaps: v7 at the registration's frozen terms,
law and D6FIX constants prices every cell the wrappers price themselves (SIGMA-D, the knee rows); the frozen scorer's
model list and page paths are the three blind models' and addendum 5's registered pages (d5common.blind5); and its
registration is this one (addendum 5's units and pages, the owner's D4-1 bars). The predictions are the
registration's own, frozen before any page. The rules as frozen: the power check first (700 W), E1-DEC pooled and per
model at rms 3%, worst 8%, |mean| 1.5% (owner, D4-1) without addendum 2's DESCRIPTIVE second-tile cells, a model CERTIFIED when its
own bar and the pooled bar hold, GD2's void and G3-D shuffle's void, G3-D label DESCRIPTIVE, the scope line.

REFUSES (exit 2) when the freeze does not hold (v7 as the gate froze it and this blind's own files), or the registration
is a DRAFT (--allow-draft scores a DRAFT for the tests, on synthetic trees only). Writes blind.score.json and
blind.score.txt."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7bcommon as V  # noqa: E402

import score_blind_d3 as SB3  # noqa: E402


def registration_block(reg: dict) -> dict:
    """score_blind.score's registration: the units and the pages."""
    return {"units": reg["cells"]["units"], "pages": reg["cells"]["pages"]}


def finalised(reg: dict) -> dict:
    p = reg["predictions"]
    return {"frozen": p["frozen"], "cells": p["cells"]}


def score(tree: Path, reg: dict) -> dict:
    fin = finalised(reg)
    with V.v7_blind(fin["frozen"]):
        s = SB3.score(Path(tree), registration_block(reg), fin)
    if s.get("verdict") == "NOT SCORED":
        return dict(s, registration=V.NAME)
    return dict(s, what="model v7's decode blind score (E1-DEC)", registration=V.NAME,
                addenda=list(s.get("addenda") or []) + [V.NAME], bars_source=reg["bars"]["source"],
                draft=reg.get("draft"))


def lines(s: dict) -> list[str]:
    L = SB3.lines(s)
    L[0] = L[0].replace("(addendum 3, model v4d')", "(model v7, three reserved models)").replace(
        "(addendum 2)", "(model v7, three reserved models)")
    if s.get("draft"):
        L.insert(0, "DRAFT REGISTRATION: a synthetic or test score, never a reading")
    return L


def refusals(reg: dict, allow_draft: bool = False) -> list[str]:
    """Why this registration may not be scored (empty: it may)."""
    if reg.get("name") != V.NAME:
        return [f"the registration is not {V.NAME}"]
    if reg.get("draft") and not allow_draft:
        return ["the registration is a DRAFT (the v7 gate has not read DECODE: GO on record)"]
    bad = V.verify_freeze(reg)
    return ["the freeze does not hold: " + "; ".join(bad[:10])] if bad else []


def provenance(path: Path) -> dict:
    import subprocess
    head = subprocess.run(["git", "-C", str(V.REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(V.REPO), "status", "--porcelain"], capture_output=True, text=True)
    return {"registration_path": V.REG_JSON, "registration_sha256": V.sha256(path),
            "git_head": head.stdout.strip() if head.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tree", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--allow-draft", action="store_true", help="score a DRAFT (tests, synthetic trees only)")
    a = ap.parse_args(argv)
    path = V.REPO / V.REG_JSON
    reg = json.loads(path.read_text())
    why = refusals(reg, a.allow_draft)
    if why:
        print("REFUSED: " + "; ".join(why))
        return 2
    s = dict(score(a.tree, reg), provenance=provenance(path))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "blind.score.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    (a.out / "blind.score.txt").write_text("\n".join(lines(s) + [f"registration sha256 {s['provenance']['registration_sha256']}, "
                                                                 f"commit {s['provenance']['git_head']}"]) + "\n")
    print("\n".join(lines(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
