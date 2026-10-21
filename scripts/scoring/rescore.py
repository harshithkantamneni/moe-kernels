#!/usr/bin/env python
"""Re-score a published stage from its published tree and compare with the committed score files.

    python scripts/scoring/rescore.py <stage> [--only NAME ...] [--keep DIR]

<stage> is tileA or tileA2. Each registered scorer runs in a subprocess, in the stage's registered order,
from the repo root, with PYTHONHASHSEED=0: scripts/scoring/tileA/score_census.py builds its occupancy dict
from a set, so its key order (and the floor score's, which reads it) follows the hash seed. That scorer is
pinned by the tileA2 registration (scripts/scoring/tileA2/register_a2.py PINNED), so the seed is fixed here
instead of in the file; the committed tileA and tileA2 score files were written with PYTHONHASHSEED=0.

The scorers write into a scratch directory; every *.score.json and *.score.txt there is compared with the
committed file of the same name after the scratch directory's path is replaced by the committed one (the
tileA2 census score records the merged census under its output directory). Exits 0 when every file is
equal, 1 otherwise, printing each file's reading.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PUB = "results/published"

#: stage -> (published session, committed score directory, scorers in their registered order)
STAGES = {
    "tileA": ("2026-10-20-nvidia_gh200_480gb-tileA-session", "scripts/scoring/tileA",
              [f"scripts/scoring/tileA/score_{s}.py" for s in ("census", "floor", "bytes", "rep", "dead", "e1cal")]),
    "tileA2": ("2026-10-21-nvidia_gh200_480gb-tileA2-session", "scripts/scoring/tileA2",
               [f"scripts/scoring/tileA2/score_{s}_a2.py"
                for s in ("session", "probe", "census", "floor", "bytes", "rep", "dead", "e1cal", "convl")]),
}


def scorer_env() -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    return env


def rescore(stage: str, out: Path, only: list[str] | None = None) -> list[tuple[str, int]]:
    """Run the stage's scorers into `out`; [(scorer, exit code)]."""
    session, _committed, scorers = STAGES[stage]
    tree = f"{PUB}/{session}/results"
    ran = []
    for s in scorers:
        if only and not any(o in Path(s).name for o in only):
            continue
        rc = subprocess.run([sys.executable, s, ".", tree, str(out)], cwd=REPO, env=scorer_env(),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode
        ran.append((s, rc))
    return ran


def compare(stage: str, out: Path) -> list[tuple[str, str]]:
    """[(file name, EQUAL | DIFFERS | MISSING)] over every score file the scratch run wrote."""
    committed = REPO / STAGES[stage][1]
    rows = []
    for p in sorted(out.glob("*.score.*")):
        if p.suffix not in (".json", ".txt"):
            continue
        c = committed / p.name
        if not c.exists():
            rows.append((p.name, "MISSING"))
            continue
        got = p.read_text().replace(str(out), STAGES[stage][1])
        rows.append((p.name, "EQUAL" if got == c.read_text() else "DIFFERS"))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--only", nargs="*", help="run only scorers whose file name contains one of these")
    ap.add_argument("--keep", help="write into this directory and keep it")
    a = ap.parse_args(argv)
    out = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix=f"rescore-{a.stage}-"))
    out.mkdir(parents=True, exist_ok=True)
    try:
        ran = rescore(a.stage, out, a.only)
        for s, rc in ran:
            print(f"{s}: exit {rc}")
        rows = compare(a.stage, out)
        for name, state in rows:
            print(f"  {name}: {state}")
        ok = all(rc == 0 for _s, rc in ran) and rows and all(state == "EQUAL" for _n, state in rows)
        print("RESCORE", a.stage, "EQUAL" if ok else "NOT EQUAL")
        return 0 if ok else 1
    finally:
        if not a.keep:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
