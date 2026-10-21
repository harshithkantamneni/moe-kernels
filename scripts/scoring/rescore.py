#!/usr/bin/env python
"""Re-score a published stage from its published tree and compare with the committed score files.

    python scripts/scoring/rescore.py <stage> [--only NAME ...] [--keep DIR]

<stage> is tileA, tileA2, settleS1 or cal2. Each registered scorer runs in a subprocess, in the stage's
registered order, from the repo root, with PYTHONHASHSEED=0: scripts/scoring/tileA/score_census.py builds its occupancy dict
from a set, so its key order (and the floor score's, which reads it) follows the hash seed. That scorer is
pinned by the tileA2 registration (scripts/scoring/tileA2/register_a2.py PINNED), so the seed is fixed here
instead of in the file; the committed tileA and tileA2 score files were written with PYTHONHASHSEED=0.

settleS1 is descriptive: its one registered reader (scripts/scoring/settleS1/read_s1.py, pinned by the
2026-10-21-settleS1-gh200 registration) reads the whole published session (results/ and session/, for the
power read-backs), and its files are settleS1.read.json and settleS1.read.txt.

cal2 is the calibration rental C2: its one registered reader (scripts/scoring/cal2/read_c2.py, pinned by the
2026-10-22-cal2-gh200 registration) reads the whole published session too, takes it as --c2 and its output
directory as --out-dir, and writes cal2.read.json and cal2.read.txt. It imports the knee round's frozen chain
from ~/code/moe-kernels-work and refuses unless every frozen file and the run's commit check (its docstring).

The scorers write into a scratch directory; every score file there (*.score.json and *.score.txt, or the
stage's own pattern) is compared with the committed file of the same name after the scratch directory's path
is replaced by the committed one (the tileA2 census score records the merged census under its output
directory). Exits 0 when every file is equal, 1 otherwise, printing each file's reading.
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
    "settleS1": ("2026-10-21-nvidia_gh200_480gb-settleS1-session", "scripts/scoring/settleS1",
                 ["scripts/scoring/settleS1/read_s1.py"]),
    "cal2": ("2026-10-21-nvidia_gh200_480gb-cal2-session", "scripts/scoring/cal2",
             ["scripts/scoring/cal2/read_c2.py"]),
}
#: the tree each stage's scorers read, under the published session (default: results)
TREE = {"settleS1": "", "cal2": ""}
#: the files each stage's scorers write (default: *.score.*)
OUTPUTS = {"settleS1": "settleS1.read.*", "cal2": "cal2.read.*"}
#: the command-line each stage's scorers take after the script (default: the repo, the tree, the output directory)
ARGV = {"cal2": lambda tree, out: ["--c2", tree, "--out-dir", out]}


def tree_of(stage: str) -> str:
    sub = TREE.get(stage, "results")
    return f"{PUB}/{STAGES[stage][0]}" + (f"/{sub}" if sub else "")


def scorer_env() -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    return env


def rescore(stage: str, out: Path, only: list[str] | None = None) -> list[tuple[str, int]]:
    """Run the stage's scorers into `out`; [(scorer, exit code)]."""
    _session, _committed, scorers = STAGES[stage]
    tree = tree_of(stage)
    ran = []
    for s in scorers:
        if only and not any(o in Path(s).name for o in only):
            continue
        args = ARGV[stage](tree, str(out)) if stage in ARGV else [".", tree, str(out)]
        rc = subprocess.run([sys.executable, s, *args], cwd=REPO, env=scorer_env(),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True).returncode
        ran.append((s, rc))
    return ran


def compare(stage: str, out: Path) -> list[tuple[str, str]]:
    """[(file name, EQUAL | DIFFERS | MISSING)] over every score file the scratch run wrote."""
    committed = REPO / STAGES[stage][1]
    rows = []
    for p in sorted(out.glob(OUTPUTS.get(stage, "*.score.*"))):
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
