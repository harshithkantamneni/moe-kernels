#!/usr/bin/env python
# ruff: noqa: E501  (published page paths and scorer keys are quoted whole)
"""What separates the held from the failed registered time tests (OUTLINE 2.11):
per test, the three regime conditions the conclusion names (waves a GEMM runs,
k-steps a CTA runs, GPU-bound calls), the dead CTAs a SHARED call dispatches, the
model version that wrote the prediction and the bytes it was priced from.

    python scripts/paper/regime.py [--out DIR]     # writes DIR/regime.md and DIR/regime.csv

Reads only committed files: the scorer outputs in scripts/scoring/crossmodel/ (the
scored cells, their verdict numbers, the pages each used), MANIFEST.json (the
target's timed-page directory), the pages' cells.csv (`host_bound`), and the
shapes in moe/spec.py through `r3_timing_model.set_model`. Fits nothing.

DEFINITIONS.
waves     live CTAs of one GEMM / (132 SMs x occupancy), occupancy 5 (w1) and 4 (w2),
          `r3_timing_model.ASSUMED_OCCUPANCY["9.0"]`; live CTAs = E x n x N-tiles.
          Printed at the smallest tread n of the scored set.
k-steps   K / BLOCK_K per CTA, per GEMM.
host      scored cells (SHARED and PRIVATE) with `host_bound` True in any repeat on
          a page the scorer used, over all scored cells.
dead      `r3_timing_model.dead_ctas` of a SHARED call at n = 1 (E x 9 declared slots),
          both GEMMs; and their dispatch at DEAD_CTA_NS = 1.333 ns before any hiding,
          as a fraction of the measured SHARED n = 1 call at the smallest scored G.
          An upper bound on the term's share; the term itself hides one lifetime.
version   the commit that wrote the registered prediction and which of the terms
          added after 2026-09-27 it carried (docs/registered/README.md; T3).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "paper"))

XM = ROOT / "scripts/scoring/crossmodel"
SMS = 132
SCORED = ("shared", "private")

#: (label, model, scorer output, kind, writing commit, terms after 2026-09-27 it carried).
#: Commits: docs/registered/README.md (8x22B 9813d6b; Qwen2-57B b0312ea; OLMoE 7d9a1a1;
#: the four held-out models 0f77622) and scripts/scoring/crossmodel/MANIFEST.json.
TESTS = [
    ("Mixtral 8x22B", "mixtral-8x22b", "8x22b.registered.json", "predicted bytes", "9813d6b",
     "knee, MIX; no partial-wave, dead-CTA or F term"),
    ("Qwen2-57B-A14B", "qwen2-57b-a14b", "qwen2.registered.json", "predicted bytes", "b0312ea",
     "knee, MIX, TAIL, stage 3; no dead-CTA or F term"),
    ("OLMoE-1B-7B", "olmoe-1b-7b", "olmoe-1b-7b.score.json", "own counted bytes", "7d9a1a1",
     "+ dead-CTA term, LATER_MISS; no F term"),
    ("Qwen1.5-MoE-A2.7B", "qwen1.5-moe-a2.7b", "qwen1.5-moe-a2.7b.score.json",
     "own counted bytes", "0f77622", "+ per-CTA F"),
    ("Phi-3.5-MoE", "phi-3.5-moe", "phi-3.5-moe.score.json", "own counted bytes", "0f77622",
     "+ per-CTA F"),
    ("JetMoE-8B", "jetmoe-8b", "jetmoe-8b.score.json", "own counted bytes", "0f77622",
     "+ per-CTA F"),
]
#: The registered bar (docs/registered/README.md): SHARED and PRIVATE rms <= 2%, none beyond 5%.
BAR_RMS, BAR_CELL = 0.02, 0.05


def _scored(d: dict) -> tuple[list[dict], dict, set[str]]:
    """(scored cells, stats, run ids of the pages used) from one committed output."""
    if "time_predicted_bytes" in d:
        cells = [c for c in d["time_predicted_bytes"]["cells"] if c["arm"] in SCORED]
        return cells, d["time_predicted_bytes"]["shared+private"], \
            {p["run"] for p in d["timed_pages"] if p["used"]}
    cells = [c for c in d["cells"] if c["arm"] in SCORED]
    return cells, d["sets"]["shared+private"], set(d["target_pages"])


def _timed_dirs(model_file: str, d: dict) -> list[Path]:
    if "session" in d:
        base = ROOT / d["session"] / "results"
        return sorted(base.glob("gaps-*/private_weight_reference/*/"))
    man = json.loads((XM / "MANIFEST.json").read_text())
    run = next(r for r in man["runs"] if f"scripts/scoring/crossmodel/{model_file}" in r["output"])
    argv = run["argv"]
    i = argv.index("--target-timed") + 1
    dirs = []
    while i < len(argv) and not argv[i].startswith("--"):
        p = ROOT / argv[i]
        dirs += [p] if (p / "cells.csv").exists() else sorted(p.glob("*/"))
        i += 1
    return dirs


def _G(dirname: str) -> int:
    return int(dirname.split("-g")[1].split("-")[0])


def host_bound(dirs: list[Path], runs: set[str], cells: list[dict]) -> tuple[int, int]:
    want = {(c["arm"], int(c["G"]), int(c["n"])) for c in cells}
    hb = set()
    for p in dirs:
        if p.name.split("-")[-1] not in runs or not (p / "cells.csv").exists():
            continue
        G = _G(p.name)
        for r in csv.DictReader((p / "cells.csv").open()):
            k = (r["arm"], G, int(r["tiles"]))
            if k in want and r["host_bound"] == "True":
                hb.add(k)
    return len(hb), len(want)


def build() -> list[dict]:
    import r3_timing_model as TM
    occ = TM.ASSUMED_OCCUPANCY["9.0"]
    out = []
    for label, model, f, kind, commit, terms in TESTS:
        d = json.loads((XM / f).read_text())
        cells, st, runs = _scored(d)
        old = TM.set_model(model)
        try:
            E = TM.E
            geo = {g: TM.GEOMETRY[g] for g in TM.GEMMS}
            n0 = min(int(c["n"]) for c in cells)
            waves = {g: E * n0 * geo[g].npn / (SMS * occ[g]) for g in TM.GEMMS}
            declared = 9 * E
            dead = {g: TM.dead_ctas(declared, 1, g) for g in TM.GEMMS}
        finally:
            TM.set_model(old)
        Gmin = min(int(c["G"]) for c in cells)
        ms1 = [c["measured_ms"] for c in cells if c["arm"] == "shared" and int(c["n"]) == 1
               and int(c["G"]) == Gmin]
        dead_us = sum(dead.values()) * TM.DEAD_CTA_NS / 1e3
        nh, nc = host_bound(_timed_dirs(f, d), runs, cells)
        held = st["rms"] <= BAR_RMS and st["beyond_5pct"] == 0
        cond = {"waves >= 4": min(waves.values()) >= 4.0,
                "k-steps >= 22": min(g.ksteps for g in geo.values()) >= 22,
                "GPU-bound": nh == 0}
        out.append({"test": label, "bytes": kind, "version": commit, "terms": terms,
                    "verdict": "HELD" if held else "FALSIFIED", "rms_pct": 100 * st["rms"],
                    "beyond_5pct": st["beyond_5pct"], "cells": st["cells"], "E": E,
                    "ksteps_w1": geo["w1"].ksteps, "ksteps_w2": geo["w2"].ksteps,
                    "n_min": n0, "waves_w1": waves["w1"], "waves_w2": waves["w2"],
                    "host_bound": f"{nh} of {nc}", "dead_ctas_n1": dead["w1"] + dead["w2"],
                    "dead_us_n1": dead_us,
                    "dead_share_n1_pct": 100 * dead_us / 1e3 / ms1[0] if ms1 else None,
                    "meets_all_three": all(cond.values()),
                    "fails": ", ".join(k for k, v in cond.items() if not v) or "none"})
    return out


COLS = ["test", "bytes", "version", "verdict", "rms_pct", "beyond_5pct", "cells", "E",
        "ksteps_w1", "ksteps_w2", "n_min", "waves_w1", "waves_w2", "host_bound",
        "dead_ctas_n1", "dead_us_n1", "dead_share_n1_pct", "meets_all_three", "fails", "terms"]


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.2f}"
    return "" if x is None else str(x)


def write(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "regime.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLS)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in COLS])
    md = ["# Regime conditions against the registered time verdicts",
          "", "Generated by `python scripts/paper/regime.py`. Sources: "
          "`scripts/scoring/crossmodel/*.registered.json`, `scripts/scoring/crossmodel/*.score.json`, "
          "`scripts/scoring/crossmodel/MANIFEST.json`, the used timed pages' `cells.csv`, `moe/spec.py`. "
          "Definitions in the script's docstring. Read after every page: a description, not a test.",
          "", "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        md.append("| " + " | ".join(_fmt(r[c]) for c in COLS) + " |")
    held = [r for r in rows if r["verdict"] == "HELD"]
    failed = [r for r in rows if r["verdict"] != "HELD"]
    md += ["", f"Held: {', '.join(r['test'] for r in held)}. Failed: "
           f"{', '.join(r['test'] for r in failed)}.",
           "Meets all three conditions: " + ", ".join(
               f"{r['test']} ({r['verdict']})" for r in rows if r["meets_all_three"]) + "."]
    (out / "regime.md").write_text("\n".join(md) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs/paper")
    a = ap.parse_args(argv)
    write(build(), a.out)
    print(f"regime: written under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
