#!/usr/bin/env python
"""Re-score a published stage from its published tree and compare with the committed score files.

    python scripts/scoring/rescore.py <stage> [--only NAME ...] [--keep DIR]

<stage> is tileA, tileA2, settleS1, cal2, cal3, cal4, decodeCAL, decodeCAL2, decodeBlind, decodeDS, tileB3, pc or pc2. Each registered scorer runs in a subprocess, in the stage's
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

cal3 is the calibration rental C3: its one registered reader (scripts/scoring/cal3/read_c3.py, pinned by the
2026-10-26-cal3-gh200 registration) reads the whole published session, takes it as --c3 and its output directory
as --out-dir, and writes cal3.read.json and cal3.read.txt. It needs no work folder: it refuses unless the run's
commit (e10fda1b, on the ledger's driver start line) holds the registration, the priced table, the plan, the
histograms and every pinned file byte for byte, and every repo module it loads is pinned (its docstring).

cal4 is the calibration rental C4: its one registered reader (scripts/scoring/cal4/read_c4.py, pinned by the
2026-10-27-cal4-gh200 registration) reads the whole published session, takes it as --c4 and its output directory
as --out-dir, and writes cal4.read.json and cal4.read.txt. Like cal2 it imports a frozen chain from
~/code/moe-kernels-work (round 4's, 2026-10-27/round3/work, and what it imports) and refuses unless every frozen file,
the run's commit (aa5e1b7c, on the ledger's driver start line) and every repo module it loads check (its docstring).

decodeCAL is the decode stage's CAL rental: its registered scorer under the second addendum
(scripts/scoring/decode/score_cal_d2.py, 2026-10-28-decode-addendum2-gh200, which runs the frozen score_cal.py
inside) reads the published session's results/ (and its session/power-set.txt, for the 700 W check), takes
--tile-scores scripts/scoring/tileA2 after the tree and the output directory, and writes cal.score.json and
cal.score.txt, committed in scripts/scoring/decode/ (the directory finalise_decode_d2.py reads).

decodeCAL2 is the decode stage's CAL2 rental: its registered scorer under the third addendum
(scripts/scoring/decode/score_cal_d3.py, 2026-10-29-decode-addendum3-gh200, which runs the frozen d2 wrapper and
score_cal.py inside) takes the same arguments as decodeCAL's and writes cal.score.json and cal.score.txt. Those were
scored on the fetched run tree before the blind finalisation and committed as scripts/scoring/decode/cal2.score.json
and cal2.score.txt (the finalisation's input, kept byte for byte), so the scratch files are compared under those names,
and the one path the committed JSON records from the fetched tree (the power file's) is read as the published copy's.

decodeBlind is the decode stage's blind rental: its registered scorer under the third addendum
(scripts/scoring/decode/score_blind_d3.py, which runs the frozen d2 wrapper and score_blind.py inside) reads the
published session's results/ (and session/power-set.txt) against scripts/scoring/decode/decodeB.finalised.json and
writes blind.score.json and blind.score.txt, committed in scripts/scoring/decode/.

decodeDS is the decode stage's settling rental DS, descriptive: its one registered reader under the fourth addendum
(scripts/scoring/decode/read_ds.py, 2026-11-04-decode-addendum4-gh200) reads the published session's results/ (and
its session/power-set.txt, for the 700 W check), takes the tree and the output directory, and writes ds.readings.json
and ds.readings.txt, committed in scripts/scoring/decode/ as read from the published tree.

tileB3 is tile rental B3, the blind test of round 5's frozen large-tile model (2026-11-04-tileB3-gh200): its registered
scorer (scripts/scoring/tileB3/read_b3.py score) takes the published session directory itself (results/ and session/,
`--b3`) and an output directory (`--out-dir`), re-finalises the predictions from the four sag pages, refuses unless
they equal the committed scripts/scoring/tileB3/b3.predictions.json, and writes tileB3.score.json and
tileB3.score.txt, committed in scripts/scoring/tileB3/ as scored on the published tree.

pc is rental PC, a measurement rental (2026-11-06-pc-gh200), descriptive: its one registered reader
(scripts/scoring/pc/read_pc.py) takes the published session directory itself (results/ and session/, `--pc`) and an
output directory (`--out-dir`), refuses unless the run's commit (c9c92f1e, on the ledger's driver start line) holds
the registration, the pages, the shapes, the priced file, the plan and every code_pins path, and writes pc.read.json
and pc.read.txt, committed in scripts/scoring/pc/ as read from the published tree.

pc2 is rental PC2, a measurement rental (2026-11-10-pc2-gh200), descriptive: its one registered reader
(scripts/scoring/pc2/read_pc2.py, over read_pc.py's pinned functions) takes the published session directory itself
(`--pc2`) and an output directory (`--out-dir`), refuses unless the run's commit (e7786f4e, on the ledger's driver start
line) holds the registration, the pages, the shapes, the plan and every code_pins path, and writes pc2.read.json and
pc2.read.txt, committed in scripts/scoring/pc2/ as read from the published tree.

The scorers write into a scratch directory; every score file there (*.score.json and *.score.txt, or the
stage's own pattern) is compared with the committed file of the same name after the scratch directory's path
is replaced by the committed one (the tileA2 census score records the merged census under its output
directory). Exits 0 when every file is equal, 1 otherwise, printing each file's reading.
"""
from __future__ import annotations

import argparse
import os
import re
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
    "cal3": ("2026-10-27-nvidia_gh200_480gb-cal3-session", "scripts/scoring/cal3",
             ["scripts/scoring/cal3/read_c3.py"]),
    "cal4": ("2026-10-27-nvidia_gh200_480gb-cal4-session", "scripts/scoring/cal4",
             ["scripts/scoring/cal4/read_c4.py"]),
    "decodeCAL": ("2026-10-29-nvidia_gh200_480gb-decodeCAL-session", "scripts/scoring/decode",
                  ["scripts/scoring/decode/score_cal_d2.py"]),
    "decodeCAL2": ("2026-11-03-nvidia_gh200_480gb-decodeCAL2-session", "scripts/scoring/decode",
                   ["scripts/scoring/decode/score_cal_d3.py"]),
    "decodeBlind": ("2026-11-04-nvidia_gh200_480gb-decodeBlind-session", "scripts/scoring/decode",
                    ["scripts/scoring/decode/score_blind_d3.py"]),
    "decodeDS": ("2026-11-04-nvidia_gh200_480gb-decodeDS-session", "scripts/scoring/decode",
                 ["scripts/scoring/decode/read_ds.py"]),
    "tileB3": ("2026-11-05-nvidia_gh200_480gb-tileB3-session", "scripts/scoring/tileB3",
               ["scripts/scoring/tileB3/read_b3.py"]),
    "pc": ("2026-11-06-nvidia_gh200_480gb-pc-session", "scripts/scoring/pc", ["scripts/scoring/pc/read_pc.py"]),
    "pc2": ("2026-11-10-nvidia_gh200_480gb-pc2-session", "scripts/scoring/pc2", ["scripts/scoring/pc2/read_pc2.py"]),
}
#: the tree each stage's scorers read, under the published session (default: results)
TREE = {"settleS1": "", "cal2": "", "cal3": "", "cal4": "", "tileB3": "", "pc": "", "pc2": ""}
#: the files each stage's scorers write (default: *.score.*)
OUTPUTS = {"settleS1": "settleS1.read.*", "cal2": "cal2.read.*", "cal3": "cal3.read.*", "cal4": "cal4.read.*",
           "decodeDS": "ds.readings.*", "pc": "pc.read.*", "pc2": "pc2.read.*"}
#: the command-line each stage's scorers take after the script (default: the repo, the tree, the output directory)
ARGV = {"cal2": lambda tree, out: ["--c2", tree, "--out-dir", out],
        "cal3": lambda tree, out: ["--c3", tree, "--out-dir", out],
        "cal4": lambda tree, out: ["--c4", tree, "--out-dir", out],
        "decodeCAL": lambda tree, out: [tree, out, "--tile-scores", "scripts/scoring/tileA2"],
        "decodeCAL2": lambda tree, out: [tree, out, "--tile-scores", "scripts/scoring/tileA2"],
        "decodeBlind": lambda tree, out: [tree, "scripts/scoring/decode/decodeB.finalised.json", out],
        "decodeDS": lambda tree, out: [tree, out],
        "tileB3": lambda tree, out: ["score", "--b3", tree, "--out-dir", out],
        "pc": lambda tree, out: ["--pc", tree, "--out-dir", out],
        "pc2": lambda tree, out: ["--pc2", tree, "--out-dir", out]}
#: the committed name of a scratch score file, where the stage committed it under another name
COMMITTED_AS = {"decodeCAL2": {"cal.score.json": "cal2.score.json", "cal.score.txt": "cal2.score.txt"}}
#: the run-tree root a committed score file records, scored on the fetched tree before publication, matched on its
#: run-relative tail (the part before it is whichever machine fetched it); it reads as the published session (the
#: fetched tree's vm/ is the session's top level)
FETCHED = {"decodeCAL2": "/runs/gh200-tile2-20261103T2203Z/tree/vm"}


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
        c = committed / COMMITTED_AS.get(stage, {}).get(p.name, p.name)
        if not c.exists():
            rows.append((p.name, "MISSING"))
            continue
        got = p.read_text().replace(str(out), STAGES[stage][1])
        want = c.read_text()
        if stage in FETCHED:
            want = re.sub(r'(?<=")[^"]*' + re.escape(FETCHED[stage]), f"{PUB}/{STAGES[stage][0]}", want)
        rows.append((p.name, "EQUAL" if got == want else "DIFFERS"))
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
