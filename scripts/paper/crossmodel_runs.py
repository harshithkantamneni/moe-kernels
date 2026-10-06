#!/usr/bin/env python
"""The committed cross-model score outputs (scripts/scoring/crossmodel/) and the
exact runs that made them.

    python scripts/paper/crossmodel_runs.py manifest            # write MANIFEST.json
    python scripts/paper/crossmodel_runs.py commands            # print every command
    python scripts/paper/crossmodel_runs.py run --tree DIR --commit C   # rerun C's runs

Two kinds of output, each a JSON and the console text beside it:

1. `<target>.score.{json,txt}`: `scripts/cross_model_score.py` AT THE
   REGISTRATION'S COMMIT (0f77622 for the four held-out models, 7d9a1a1 for
   OLMoE-1B-7B; docs/registered/README.md:130 and :196 name "at this commit"),
   source = 8x7B's five lock-1710 timed pages of 2026-09-27 (the registration
   JSONs' `timing_pages`) and that session's lock-1710 counter pages. To rerun
   one, check out its commit with `results/` linked to this tree:
       git archive <commit> -- . ':!results' | tar -x -C DIR; ln -s $PWD/results DIR/results
       python scripts/paper/crossmodel_runs.py run --tree DIR --commit <commit>
2. `<case>.registered.{json,txt}`: `scripts/paper/score_registered.py` at this
   tree (time from predicted bytes, the G >= 8 slope, the floor, the bytes).

Granite-3.0-3B's time test is NOT SCORABLE (no VALID page); its run is kept and
its output is the scorer's refusal line.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts/scoring/crossmodel"
PUB = "results/published"
SRC_DIR = f"{PUB}/2026-09-27-nvidia_gh200_480gb-session/results/gaps-nvidia_gh200_480gb/private_weight_reference"
SOURCE_TIMED = [f"{SRC_DIR}/nvidia_gh200_480gb-bm32-budget200.0-declauto-dtypebf16-duty0.25-{x}" for x in (
    "g2-l2flushtrue-modelmixtral_8x7b-04d9320b", "g3-l2flushtrue-modelmixtral_8x7b-e20972ef",
    "g32-l2flushtrue-modelmixtral_8x7-8ab16728", "g4-l2flushtrue-modelmixtral_8x7b-848523de",
    "g8-l2flushtrue-modelmixtral_8x7b-2085264b")]
SOURCE_COUNTERS = f"{PUB}/2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters/lock1710"


JET = f"{PUB}/2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/gaps-nvidia_gh200_480gb/private_weight_reference"
# JetMoE: the four lock-1710 pages; its G=2 page held 1605 (D/README.md:233-234) and the
# scorer refuses mixed locks, so the 1710 run dirs are named.
JET_1710 = [f"{JET}/{d}" for d in sorted(os.listdir(ROOT / JET))
            if any(k in d for k in ("593be327", "1f286b22", "9b196085", "d9bd231b"))] \
    if (ROOT / JET).exists() else []

XM = [  # (target, session, commit, ledger rows, target timed dirs or None for the whole dir)
    ("olmoe-1b-7b", "2026-09-29-nvidia_gh200_480gb-olmoe-session", "7d9a1a1", [20], None),
    ("qwen1.5-moe-a2.7b", "2026-09-29-nvidia_gh200_480gb-qwen1.5-session", "0f77622", [25], None),
    ("phi-3.5-moe", "2026-09-29-nvidia_gh200_480gb-phi3.5-session", "0f77622", [29], None),
    ("jetmoe-8b", "2026-09-29-nvidia_gh200_480gb-jetmoe-session", "0f77622", [33], JET_1710),
    ("granite-3.0-3b-a800m", "2026-09-30-nvidia_gh200_480gb-granite-session", "0f77622", [37], None),
]
HEAD_FLAGS = {"0f77622": ["--no-cores"], "7d9a1a1": ["--no-cta-fixed", "--no-cores"]}
REGISTERED = ["8x22b", "qwen2", "olmoe", "qwen1.5", "phi3.5", "jetmoe", "granite",
              "jetmoe-floor", "mixtral-floor"]


def xm_argv(target, session, timed) -> list[str]:
    sess = f"{PUB}/{session}/results"
    counters = next(p.name for p in sorted((ROOT / sess).glob("*-r3-counters")))
    return ["scripts/cross_model_score.py", "--target", target,
            "--source-timed", *SOURCE_TIMED, "--source-counters", SOURCE_COUNTERS,
            "--target-timed", *(timed or [f"{sess}/gaps-nvidia_gh200_480gb/private_weight_reference"]),
            "--target-counters", f"{sess}/{counters}/lock1710",
            "--out", f"scripts/scoring/crossmodel/{target}.score.json"]


def manifest() -> dict:
    sys.path.insert(0, str(ROOT / "scripts" / "paper"))
    import score_registered as SR
    runs = []
    for target, session, commit, rows, timed in XM:
        runs.append({"output": [f"scripts/scoring/crossmodel/{target}.score.json",
                                f"scripts/scoring/crossmodel/{target}.score.txt"],
                     "scorer": "scripts/cross_model_score.py", "scorer_commit": commit,
                     "argv": xm_argv(target, session, timed),
                     "stdout_to": f"scripts/scoring/crossmodel/{target}.score.txt",
                     "registration": f"docs/registered/2026-09-29-{target}-gh200.json",
                     "ledger_rows": rows, "quantity": "time from the target's own counted bytes",
                     # HEAD's scorer reproduces the registered model version with these flags
                     # (CORES came 2026-09-30; the per-CTA F at 0f77622, after 7d9a1a1):
                     "head_flags": HEAD_FLAGS[commit]})
    for case in REGISTERED:
        c = SR.CASES[case]
        runs.append({"output": [f"scripts/scoring/crossmodel/{case}.registered.json",
                                f"scripts/scoring/crossmodel/{case}.registered.txt"],
                     "scorer": "scripts/paper/score_registered.py", "scorer_commit": "this tree",
                     "argv": ["scripts/paper/score_registered.py", case, "--out-dir",
                              "scripts/scoring/crossmodel"],
                     "registration": (f"docs/registered/{c['reg']}" if c["reg"]
                                      else "docs/registered/README.md 2026-09-30 floor-only"),
                     "session": f"{PUB}/{c['session']}", "ledger_rows": c["rows"],
                     "quantity": "time from predicted bytes, G>=8 slope, floor, bytes"})
    return {"what": "committed scorer outputs for OUTLINE ledger rows 10 to 43 (gap 6.8)",
            "m5_note": "grading/checks/m5_scores.py reads no manifest; it counts any committed JSON "
                       "with \"tool\": \"scripts/cross_model_score.py\" as MISMATCH (rerun not "
                       "implemented there). tests/test_paper_crossmodel.py is the rerun check.",
            "runs": runs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["manifest", "commands", "run"])
    ap.add_argument("--tree", type=Path)
    ap.add_argument("--commit")
    a = ap.parse_args(argv)
    m = manifest()
    if a.what == "manifest":
        (OUT / "MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
        return 0
    if a.what == "commands":
        for r in m["runs"]:
            tail = f" > {r['stdout_to']}" if r.get("stdout_to") else ""
            print(f"# at {r['scorer_commit']}: python " + " ".join(r["argv"]) + tail)
        return 0
    tree = a.tree.resolve()
    env = dict(os.environ, PYTHONPATH=f"{tree}:{tree}/scripts", PYTHONDONTWRITEBYTECODE="1")
    for r in m["runs"]:
        if r["scorer_commit"] != a.commit:
            continue
        argv = [x if not x.startswith("scripts/scoring/crossmodel/") else str(ROOT / x)
                for x in r["argv"]]
        p = subprocess.run([sys.executable, *argv], cwd=tree, env=env, capture_output=True,
                           text=True)
        (ROOT / r["stdout_to"]).write_text(p.stdout)
        print(f"{r['output'][0]}: exit {p.returncode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
