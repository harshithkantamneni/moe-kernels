#!/usr/bin/env python
# ruff: noqa: E501  (published page paths are quoted whole)
"""Noise floors of the paper's error metrics, from published pages only.

    python scripts/paper/noise_floors.py [--out-dir docs/paper]

Reads results/published/** and committed files; fits nothing, measures nothing
new. Writes docs/paper/noise_floors.csv (one row per metric and method) and
docs/paper/noise_floors.md (the same table, with the method notes).

THREE METHODS, each used only where the published pages hold it.

  (a) repeat: the same cell (model, arm, G, n, clock, board) measured more than
      once inside one session: the repeat rows of a timed page's cells.csv
      (R3 times each cell over its `repeat` rows; the cell value is a median
      over them), the per-call values of a counter page (`per_gemm_values`,
      3 calls), the 3 repeats of a launch-floor cell, and the base, unlocked
      and 1710 captures of one floor capture's cells (cycles; the clock
      differs, which is said on the row).
  (c) retake: a timed page at the same (model, G, clock) taken twice on one
      board in one session (the second after a gate failure). A retake pair
      is a replicate of the cell value, whatever the gate verdict; the gate is
      named on the row.
  (b) board: the same cell on two GH200 boards. Mixtral 8x7B is the only
      model with full R3 data on two boards (2026-09-25 board 1310e2 and
      2026-09-27 board 9b6d01, timed and counter pages); its counter cells
      also exist on rental 1 (7269a7, `sameboard`) and rental 2 (4da056,
      `l2base`), and its floor capture on board 594c0f (2026-09-30).

A difference d between two single measurements of one cell has sd sqrt(2)
sigma, so a single-measurement sigma is rms(d) / sqrt(2); the rows say which
of the two they print. A cell value that is a median of R repeats is printed
with its standard error sd / sqrt(R) (the median's is about 1.25 x that; the
rows print the mean's, labelled).

Where a metric has no replicate of any kind the row reads
"no replicate exists; rental 3 part R does not yet include one" (part R, as designed,
measures sigma_page on qwen2-57b-a14b-tp8 and sigma_board on Mixtral 8x7B only).

Added 2026-10-05 (gradefix4): NATIVE bytes, (a) and (b) as for PRIVATE and SHARED;
and the launch offsets P2 (E240 - GR), P3 (E0 - E240), P4 (E480 - E240), P8
(H - I_E240), each the SE of the offset from repeat-paired differences over the
3 repeats of a cell, and P6's eager cell value (relative SE), with rental 1
against rental 2 for the same (model, arm, n) as the board-and-host term.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dram_counter_route as DCR  # noqa: E402
import floor_estimator as FE  # noqa: E402
import l2_survival as L2  # noqa: E402

from moe.spec import MODEL_CONFIGS  # noqa: E402

PUB = ROOT / "results" / "published"
NONE = "no replicate exists; rental 3 part R does not yet include one"
SCORED_ARMS = ("shared", "private")


# ----------------------------------------------------------------- helpers
def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def rms(v) -> float:
    v = list(v)
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")


def pct(v, q) -> float:
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def timed_dir(session: str, run: str) -> Path:
    hits = sorted((PUB / session).glob(f"results/gaps-*/private_weight_reference/*-{run}"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{session} run {run}: {hits}")
    return hits[0]


def read_csv(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def cell_values(page_dir: Path) -> dict:
    """(arm, n) -> ms_p50 of the page's treads_table (the cell value)."""
    rep = json.loads((page_dir / "report.json").read_text())
    return {(r["arm"], int(r["tiles"])): float(r["ms_p50"])
            for r in rep["treads_table"] if r.get("ms_p50") is not None}


def cell_repeats(page_dir: Path) -> dict:
    """(arm, n) -> [ms_p50 of each ok repeat row] from cells.csv."""
    out: dict = {}
    for r in read_csv(page_dir / "cells.csv"):
        if r.get("status") != "ok" or not r.get("ms_p50"):
            continue
        out.setdefault((r["arm"], int(r["tiles"])), []).append(float(r["ms_p50"]))
    return out


def counter(path: Path) -> dict:
    return json.loads(path.read_text())


def ccells(page: dict) -> dict:
    return {(c["arm"], int(c["n"])): c for c in page["cells"]}


def ksteps(model: str, gemm: str) -> int:
    cfg = MODEL_CONFIGS[model]
    k = cfg.hidden_size if gemm == "w1" else cfg.intermediate_size
    return k // 64


# ------------------------------------------------------------ the inputs
#: Every VALID lock-1710 timed page the cross-model time tests scored, by model:
#: (session, [runs]). Read off each session README and the page labels
#: (r3_timing_model.load_timed_page); 8x22B's and JetMoE's G=2 pages ran at 1605
#: and are not here.
SCORED_TIMED = {
    "mixtral-8x7b": ("2026-09-27-nvidia_gh200_480gb-session",
                     ["04d9320b", "e20972ef", "848523de", "2085264b", "8ab16728"]),
    "mixtral-8x22b": ("2026-09-28-nvidia_gh200_480gb-8x22b-session",
                      ["aa86fa35", "a4bcaba2", "ded3fdb7"]),
    "qwen2-57b-a14b": ("2026-09-28-nvidia_gh200_480gb-qwen2-57b-session",
                       ["46a3d596", "e44afbb4", "1a7ed59e", "089cc6ab"]),
    "olmoe-1b-7b": ("2026-09-29-nvidia_gh200_480gb-olmoe-session",
                    ["e5a28ceb", "72f810fa", "7316af55", "c9dcad2e", "87953518"]),
    "qwen1.5-moe-a2.7b": ("2026-09-29-nvidia_gh200_480gb-qwen1.5-session",
                          ["8b67e946", "cd42c1ed", "89bd3066", "00804a3a", "176ddae1"]),
    "phi-3.5-moe": ("2026-09-29-nvidia_gh200_480gb-phi3.5-session",
                    ["d4af2011", "29a83ec7", "2b704b65", "7b62501d", "dbd2c524"]),
    "jetmoe-8b": ("2026-09-29-nvidia_gh200_480gb-jetmoe-session",
                  ["593be327", "1f286b22", "9b196085", "d9bd231b"]),
}

#: Retake pairs: same session, model, G and 1710 lock, two timed pages.
#: (label, session, run_a, run_b, gate note)
RETAKES = [
    ("8x22B G=8", "2026-09-28-nvidia_gh200_480gb-8x22b-session", "a4bcaba2", "b03360ac",
     "VALID / INVALID (V0)"),
    ("Qwen2-57B G=2", "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session", "54d173bf",
     "cf8fe7b5", "INVALID (V0) / INVALID (V0)"),
    ("Qwen2-57B G=32", "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session", "72217a81",
     "e44afbb4", "INVALID (V0) / VALID"),
    ("Granite-3B G=2", "2026-09-30-nvidia_gh200_480gb-granite-session", "44033b12", "b617b2e5",
     "INVALID (V5) both"),
    ("Granite-3B G=3", "2026-09-30-nvidia_gh200_480gb-granite-session", "335f71aa", "d2096702",
     "INVALID (V5) both"),
    ("Granite-3B G=4", "2026-09-30-nvidia_gh200_480gb-granite-session", "5302d6ff", "7a37fd2c",
     "INVALID (V5) both"),
    ("Granite-3B G=8", "2026-09-30-nvidia_gh200_480gb-granite-session", "12535485", "d753ab2c",
     "INVALID (V5) both"),
    ("Granite-3B G=32", "2026-09-30-nvidia_gh200_480gb-granite-session", "16608de2", "e0fddc2d",
     "INVALID (V5) both"),
]

#: Mixtral 8x7B on two boards, VALID lock-1710 timed pages at the same G.
BOARD_TIMED = [
    ("G=2", ("2026-09-25-nvidia_gh200_480gb-session", "df37ea07"),
     ("2026-09-27-nvidia_gh200_480gb-session", "04d9320b")),
    ("G=4", ("2026-09-25-nvidia_gh200_480gb-session", "01c08abd"),
     ("2026-09-27-nvidia_gh200_480gb-session", "848523de")),
]

S25 = "2026-09-25-nvidia_gh200_480gb-session/results/2026-09-25-nvidia_gh200_480gb-r3-counters"
S27 = "2026-09-27-nvidia_gh200_480gb-session/results/2026-09-27-nvidia_gh200_480gb-r3-counters"
R1 = "2026-10-01-nvidia_gh200_480gb-rental1-session/results"
R2 = "2026-10-02-nvidia_gh200_480gb-rental2-session/results"
SB = f"{R1}/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-sameboard-r3-counters/lock1710"
L2B = f"{R2}/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-l2base-r3-counters/lock1710"

#: Mixtral 8x7B counter pages of one G on two boards: (G, page a, page b, boards).
BOARD_BYTES = [
    (1, f"{S25}/r3c-g1.json", f"{S27}/lock1710/r3c-g1.json", "1310e2 / 9b6d01"),
    (2, f"{S25}/r3c-g2.json", f"{S27}/lock1710/r3c-g2.json", "1310e2 / 9b6d01"),
    (4, f"{S25}/r3c-g4.json", f"{S27}/lock1710/r3c-g4.json", "1310e2 / 9b6d01"),
    (16, f"{S25}/r3c-g16.json", f"{S27}/lock1710/r3c-g16.json", "1310e2 / 9b6d01"),
    (64, f"{S25}/r3c-g64.json", f"{S27}/lock1710/r3c-g64.json", "1310e2 / 9b6d01"),
    (1, f"{SB}/r3c-g1.json", f"{S27}/lock1710/r3c-g1.json", "7269a7 / 9b6d01"),
    (8, f"{SB}/r3c-g8.json", f"{S27}/lock1710/r3c-g8.json", "7269a7 / 9b6d01"),
    (16, f"{SB}/r3c-g16.json", f"{S27}/lock1710/r3c-g16.json", "7269a7 / 9b6d01"),
    (32, f"{SB}/r3c-g32.json", f"{S27}/lock1710/r3c-g32.json", "7269a7 / 9b6d01"),
    (1, f"{L2B}/r3c-g1.json", f"{SB}/r3c-g1.json", "4da056 / 7269a7"),
]

#: Floor captures (base, lock and unlocked where published), by session dir.
FLOOR_SESSIONS = [
    ("mixtral-8x7b 2026-09-27 (9b6d01)", S27, "mixtral-8x7b", (2, 3, 4, 6)),
    ("mixtral-8x7b 2026-09-30 (594c0f)",
     "2026-09-30-nvidia_gh200_480gb-mixtral8x7b-floor-session/results/"
     "2026-09-30-nvidia_gh200_480gb-r3-counters", "mixtral-8x7b", (2, 3, 4, 6)),
    ("jetmoe-8b 2026-09-29 (d67185)",
     "2026-09-29-nvidia_gh200_480gb-jetmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters",
     "jetmoe-8b", (2, 3, 4, 6)),
    ("jetmoe-8b 2026-09-30 (594c0f)",
     "2026-09-30-nvidia_gh200_480gb-jetmoe-floor-session/results/"
     "2026-09-30-nvidia_gh200_480gb-r3-counters", "jetmoe-8b", (2, 3, 4, 6)),
    ("mixtral-8x22b (435984)",
     "2026-09-28-nvidia_gh200_480gb-8x22b-session/results/2026-09-28-nvidia_gh200_480gb-r3-counters",
     "mixtral-8x22b", (2, 3, 4, 6)),
    ("qwen2-57b-a14b (50e61f)",
     "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session/results/"
     "2026-09-28-nvidia_gh200_480gb-r3-counters", "qwen2-57b-a14b", (2, 3, 4, 6)),
    ("olmoe-1b-7b (d663f7)",
     "2026-09-29-nvidia_gh200_480gb-olmoe-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters",
     "olmoe-1b-7b", (2, 3, 4, 6)),
    ("qwen1.5-moe-a2.7b (d67185)",
     "2026-09-29-nvidia_gh200_480gb-qwen1.5-session/results/"
     "2026-09-29-nvidia_gh200_480gb-r3-counters", "qwen1.5-moe-a2.7b", (2, 3, 4, 6)),
    ("phi-3.5-moe (d67185)",
     "2026-09-29-nvidia_gh200_480gb-phi3.5-session/results/2026-09-29-nvidia_gh200_480gb-r3-counters",
     "phi-3.5-moe", (2, 3, 4, 6)),
]

TP_FLOOR_R2 = [
    ("tp8", f"{R2}/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor2-r3-counters"),
    ("tp4", f"{R2}/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp4-floor-r3-counters"),
    ("tp2", f"{R2}/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-floor-r3-counters"),
]
TP8_R1 = f"{R1}/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp8-floor-r3-counters"

#: G = 1 counter pages carrying s, by (model and GEMM, board): the L2-survival cells.
SURVIVAL = [
    ("8x7B w2", "mixtral-8x7b", "w2", [
        ("1310e2", f"{S25}/r3c-g1.json"), ("9b6d01", f"{S27}/lock1710/r3c-g1.json"),
        ("7269a7", f"{SB}/r3c-g1.json"), ("4da056", f"{L2B}/r3c-g1.json")]),
    ("tp2 w2", "mixtral-8x7b-tp2", "w2", [
        ("7269a7", f"{R1}/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp2-l2-r3-counters/lock1710/r3c-g1.json"),
        ("4da056", f"{R2}/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-l2base-r3-counters/lock1710/r3c-g1.json")]),
    ("tp4 w1", "mixtral-8x7b-tp4", "w1", [
        ("7269a7", f"{R1}/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp4-l2-r3-counters/lock1710/r3c-g1.json"),
        ("4da056", f"{R2}/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp4-l2base-r3-counters/lock1710/r3c-g1.json")]),
]

LAUNCH = {
    "rental1": f"{R1}/2026-10-01-nvidia_gh200_480gb-launch-floor-rental1",
    "rental2": f"{R2}/2026-10-02-nvidia_gh200_480gb-launch-floor-r2",
}


# ------------------------------------------------------------ the metrics
def time_repeat() -> dict:
    """(a) the standard error of each scored cell's value from its repeat rows."""
    se, srcs, per_model = [], [], {}
    for model, (sess, runs) in SCORED_TIMED.items():
        m = []
        for run in runs:
            d = timed_dir(sess, run)
            srcs.append(rel(d / "cells.csv"))
            for (arm, _n), v in cell_repeats(d).items():
                if arm in SCORED_ARMS and len(v) >= 2:
                    m.append(st.stdev(v) / st.mean(v) / math.sqrt(len(v)))
        per_model[model] = rms(m)
        se += m
    return {"cells": len(se), "median": st.median(se), "p90": pct(se, 0.9), "rms": rms(se),
            "per_model_rms": per_model, "sources": srcs}


def page_pair(sa, ra, sb, rb, arms=SCORED_ARMS) -> list[float]:
    a, b = cell_values(timed_dir(sa, ra)), cell_values(timed_dir(sb, rb))
    return [b[k] / a[k] - 1 for k in sorted(set(a) & set(b)) if k[0] in arms]


def host_bound_cells(sess, run) -> set:
    hb = set()
    for r in read_csv(timed_dir(sess, run) / "cells.csv"):
        if r.get("host_bound") == "True":
            hb.add((r["arm"], int(r["tiles"])))
    return hb


def time_retake() -> dict:
    rows, gpu, srcs = [], [], []
    for label, sess, ra, rb, gate in RETAKES:
        a, b = cell_values(timed_dir(sess, ra)), cell_values(timed_dir(sess, rb))
        hb = host_bound_cells(sess, ra) | host_bound_cells(sess, rb)
        keys = [k for k in sorted(set(a) & set(b)) if k[0] in SCORED_ARMS]
        d = [b[k] / a[k] - 1 for k in keys]
        dg = [b[k] / a[k] - 1 for k in keys if k not in hb]
        pair_src = f"{rel(timed_dir(sess, ra) / 'report.json')}; {rel(timed_dir(sess, rb) / 'report.json')}"
        rows.append((label, gate, len(d), rms(d) / math.sqrt(2), len(dg),
                     rms(dg) / math.sqrt(2) if dg else float("nan"), pair_src))
        if not label.startswith("Granite"):
            gpu += dg
        srcs.append(pair_src)
    return {"pairs": rows, "gpu_bound_nongranite_sigma": rms(gpu) / math.sqrt(2),
            "gpu_bound_nongranite_cells": len(gpu), "sources": srcs}


def time_board() -> dict:
    d, srcs = [], []
    for _label, (sa, ra), (sb, rb) in BOARD_TIMED:
        d += page_pair(sa, ra, sb, rb)
        srcs += [rel(timed_dir(sa, ra) / "report.json"), rel(timed_dir(sb, rb) / "report.json")]
    return {"cells": len(d), "rms_diff": rms(d), "sigma_single": rms(d) / math.sqrt(2),
            "worst": max(d, key=abs), "mean": st.mean(d), "sources": srcs}


def bytes_repeat(arm: str = "private") -> dict:
    """(a) per-call spread of one arm's per-GEMM DRAM bytes, every published GH200
    lock-1710 counter page of the scored models and Mixtral (3 calls a cell)."""
    se, srcs = [], []
    pages = sorted(PUB.glob("2026-09-2*-nvidia_gh200*/results/*-r3-counters/lock1710/r3c-g*.json"))
    for p in pages:
        pg = counter(p)
        srcs.append(rel(p))
        for c in pg["cells"]:
            if c["arm"] != arm:
                continue
            for _g, vals in (c.get("per_gemm_values") or {}).items():
                if len(vals) >= 2:
                    se.append(st.stdev(vals) / st.mean(vals) / math.sqrt(len(vals)))
    return {"pages": len(pages), "cell_gemms": len(se), "median": st.median(se),
            "p90": pct(se, 0.9), "max": max(se), "sources": srcs}


def bytes_board(arms=("private", "shared")) -> dict:
    out = {**{a: [] for a in arms}, "rows": [], "sources": []}
    for G, pa, pb, boards in BOARD_BYTES:
        a, b = ccells(counter(PUB / pa)), ccells(counter(PUB / pb))
        out["sources"] += [f"results/published/{pa}", f"results/published/{pb}"]
        for arm in arms:
            d = []
            for key in sorted(set(a) & set(b)):
                if key[0] != arm:
                    continue
                for g in ("w1", "w2"):
                    d.append(b[key]["per_gemm"][g]["dram_bytes_read"]
                             / a[key]["per_gemm"][g]["dram_bytes_read"] - 1)
            out[arm] += d
            if d:
                out["rows"].append((G, boards, arm, len(d), rms(d), max(d, key=abs)))
    return out


def _board_diffs(pa: str, pb: str, arm: str) -> list[float]:
    a, b = ccells(counter(PUB / pa)), ccells(counter(PUB / pb))
    return [b[k]["per_gemm"][g]["dram_bytes_read"] / a[k]["per_gemm"][g]["dram_bytes_read"] - 1
            for k in sorted(set(a) & set(b)) if k[0] == arm for g in ("w1", "w2")]


def floor_slopes(sdir: str, model: str, treads) -> dict:
    """{capture: {gemm: slope}} over `treads`, the registered estimator."""
    out = {}
    for name in ("r3f-g64.json", "r3f-g64-lock1710.json", "r3f-g64-unlocked.json"):
        p = PUB / sdir / name
        if not p.exists():
            continue
        cells = ccells(counter(p))
        out[name] = {}
        for g in ("w1", "w2"):
            pts = {n: (cells[("native", n)]["grid"][g],
                       cells[("native", n)]["per_gemm"][g]["sm__cycles_elapsed.avg"])
                   for n in treads if ("native", n) in cells}
            if len(pts) == len(treads):
                out[name][g] = FE.implied_slope(pts, treads, ksteps(model, g))
    return out


def floor_metric() -> dict:
    caps, srcs, d_clock = {}, [], []
    for label, sdir, model, treads in FLOOR_SESSIONS:
        caps[label] = floor_slopes(sdir, model, treads)
        srcs += [f"results/published/{sdir}/{k}" for k in caps[label]]
        base = caps[label].get("r3f-g64.json", {})
        for other in ("r3f-g64-lock1710.json", "r3f-g64-unlocked.json"):
            for g, v in caps[label].get(other, {}).items():
                if g in base:
                    d_clock.append(v / base[g] - 1)
    board = []
    for a, b in (("mixtral-8x7b 2026-09-27 (9b6d01)", "mixtral-8x7b 2026-09-30 (594c0f)"),
                 ("jetmoe-8b 2026-09-29 (d67185)", "jetmoe-8b 2026-09-30 (594c0f)")):
        for cap in ("r3f-g64.json", "r3f-g64-lock1710.json"):
            for g in ("w1", "w2"):
                va, vb = caps[a].get(cap, {}).get(g), caps[b].get(cap, {}).get(g)
                if va and vb:
                    board.append((a.split()[0], cap, g, va, vb, vb / va - 1))
    return {"captures": caps, "clock_diffs": d_clock, "board": board, "sources": srcs}


def slope_metric() -> dict:
    """(a) the G >= 8 SHARED slope over treads 2..6 per repeat index, every scored
    VALID G >= 8 page; (c) the retake pairs' page slopes."""
    rel_se, srcs = [], []
    for _model, (sess, runs) in SCORED_TIMED.items():
        for run in runs:
            d = timed_dir(sess, run)
            rep = json.loads((d / "report.json").read_text())
            if int(rep["pinned"]["GROUP_SIZE_M"]) < 8:
                continue
            by_rep: dict = {}
            for r in read_csv(d / "cells.csv"):
                if r["arm"] == "shared" and r.get("status") == "ok" and 2 <= int(r["tiles"]) <= 6:
                    by_rep.setdefault(int(r["repeat"]), {})[int(r["tiles"])] = float(r["ms_p50"])
            slopes = []
            for pts in by_rep.values():
                if len(pts) == 5:
                    xs = sorted(pts)
                    mx = st.mean(xs)
                    my = st.mean(pts[x] for x in xs)
                    slopes.append(sum((x - mx) * (pts[x] - my) for x in xs)
                                  / sum((x - mx) ** 2 for x in xs))
            if len(slopes) >= 2:
                rel_se.append(st.stdev(slopes) / st.mean(slopes) / math.sqrt(len(slopes)))
                srcs.append(rel(d / "cells.csv"))
    retake = []
    for label, sess, ra, rb, gate in RETAKES:
        if not any(t in label for t in ("G=8", "G=32")):
            continue
        sl = []
        for run in (ra, rb):
            v = cell_values(timed_dir(sess, run))
            xs = [2, 3, 4, 5, 6]
            mx, my = st.mean(xs), st.mean(v[("shared", x)] for x in xs)
            sl.append(sum((x - mx) * (v[("shared", x)] - my) for x in xs)
                      / sum((x - mx) ** 2 for x in xs))
        retake.append((label, gate, sl[0], sl[1], sl[1] / sl[0] - 1))
    return {"pages": len(rel_se), "median": st.median(rel_se), "max": max(rel_se),
            "retake": retake, "sources": srcs}


def tp8_D() -> dict:
    out = {}
    for name in ("r3f-g64.json", "r3f-g64-lock1710.json", "r3f-g8.json", "r3f-g8-lock1710.json"):
        cells = ccells(counter(PUB / TP8_R1 / name))
        y = {n: cells[("native", n)]["per_gemm"]["w1"]["sm__cycles_elapsed.avg"] for n in (2, 4)}
        out[name] = y[4] - y[2]
    return out


def part4_cells() -> dict:
    """Per-cell w1 cycles, 1710 capture minus base capture, rental 2's tp floors."""
    d, srcs = [], []
    for _tp, sdir in TP_FLOOR_R2:
        a = ccells(counter(PUB / sdir / "r3f-g64.json"))
        b = ccells(counter(PUB / sdir / "r3f-g64-lock1710.json"))
        srcs += [f"results/published/{sdir}/r3f-g64.json",
                 f"results/published/{sdir}/r3f-g64-lock1710.json"]
        for k in sorted(set(a) & set(b)):
            if k[1] >= 2:
                d.append(b[k]["per_gemm"]["w1"]["sm__cycles_elapsed.avg"]
                         - a[k]["per_gemm"]["w1"]["sm__cycles_elapsed.avg"])
    return {"cells": len(d), "mean": st.mean(d), "sd": st.stdev(d),
            "sigma_single": st.stdev(d) / math.sqrt(2), "sources": srcs}


def survival_metric() -> dict:
    rows, srcs, call_sd = [], [], []
    for label, _model, g, pages in SURVIVAL:
        s = {}
        for bd, p in pages:
            pg = counter(PUB / p)
            srcs.append(f"results/published/{p}")
            s[bd] = L2.survival(pg)[g]["s"]
            cells = ccells(pg)
            wg = L2.weight_bytes(MODEL_CONFIGS[pg["design"]["model"]], g, 2)
            for n in range(2, 10):
                sh, pr = cells.get(("shared", n)), cells.get(("private", n))
                if sh and pr and sh.get("per_gemm_values") and pr.get("per_gemm_values"):
                    vs, vp = sh["per_gemm_values"][g], pr["per_gemm_values"][g]
                    var = st.variance(vs) / len(vs) + st.variance(vp) / len(vp)
                    call_sd.append(math.sqrt(var) / ((n - 1) * wg))
        bds = list(s)
        for i in range(len(bds)):
            for j in range(i + 1, len(bds)):
                common = sorted(set(s[bds[i]]) & set(s[bds[j]]))
                diffs = [s[bds[j]][n] - s[bds[i]][n] for n in common if n >= 4]
                if diffs:
                    rows.append((label, f"{bds[i]} / {bds[j]}", len(diffs), rms(diffs),
                                 max(diffs, key=abs)))
    return {"rows": rows, "call_se_median": st.median(call_sd), "call_se_max": max(call_sd),
            "sources": srcs}


def t2_metric() -> dict:
    """f = (q - 9) 128 / (63 x 9), PRIVATE w2 at n = 9, on the same cell on two boards."""
    def f(path: str) -> float:
        q = DCR.r3_q(counter(PUB / path))["private"]["w2"][9]
        return (q - 9) * 128 / (63 * 9), q
    pairs = [
        ("tp2 G=64", f"{R1}/2026-10-01-nvidia_gh200_480gb-mixtral-8x7b-tp2-atile-r3-counters/lock1710/r3c-g64.json",
         f"{R2}/2026-10-02-nvidia_gh200_480gb-mixtral-8x7b-tp2-atbk64-r3-counters/lock1710/r3c-g64.json",
         "7269a7 / 4da056"),
        ("8x7B G=16", f"{S27}/lock1710/r3c-g16.json", f"{SB}/r3c-g16.json", "9b6d01 / 7269a7"),
        ("8x7B G=32", f"{S27}/lock1710/r3c-g32.json", f"{SB}/r3c-g32.json", "9b6d01 / 7269a7"),
    ]
    rows, srcs = [], []
    for label, pa, pb, bds in pairs:
        (fa, qa), (fb, qb) = f(pa), f(pb)
        rows.append((label, bds, qa, qb, fa, fb, fb / fa - 1))
        srcs += [f"results/published/{pa}", f"results/published/{pb}"]
    return {"rows": rows, "sources": srcs}


def launch_metric() -> dict:
    """(a) GR repeats per cell; (b) Granite-3B and JetMoE GR cells, rental 1 against 2."""
    rep_sd, srcs, vals = [], [], {}
    for rental, d in LAUNCH.items():
        for model in ("granite-3.0-3b-a800m", "jetmoe-8b", "granite-3.0-1b-a400m",
                      "mixtral-8x7b-tp8"):
            p = PUB / d / model / "cells.csv"
            if not p.exists():
                continue
            srcs.append(rel(p))
            cell: dict = {}
            for r in read_csv(p):
                if r["mode"] == "GR" and r["status"] == "ok":
                    cell.setdefault((r["arm"], int(r["tiles"])), []).append(float(r["ms_p50"]))
            for k, v in cell.items():
                if len(v) >= 2:
                    rep_sd.append(st.stdev(v) / st.mean(v))
                vals[(rental, model, k)] = st.median(v)
    board = []
    for model in ("granite-3.0-3b-a800m", "jetmoe-8b", "granite-3.0-1b-a400m"):
        ks = {k for (r, m, k) in vals if m == model and r == "rental1"} & \
             {k for (r, m, k) in vals if m == model and r == "rental2"}
        d = [vals[("rental2", model, k)] / vals[("rental1", model, k)] - 1 for k in sorted(ks)]
        if d:
            board.append((model, len(d), rms(d), max(d, key=abs)))
    return {"cells": len(rep_sd), "rep_sd_median": st.median(rep_sd), "rep_sd_p90": pct(rep_sd, 0.9),
            "board": board, "sources": srcs}


#: The launch-floor offsets (D/2026-10-01-launch-floor-gh200.txt; rental 2's rerun,
#: D/2026-10-01-rental2-launch2-gh200.txt): (metric, quoted error, mode a, mode b,
#: cell rule). Each offset is a difference of two cell values in us; P6 is the
#: eager cell value itself, against MP. The cell rules are the registrations' sets
#: on Granite-3B and JetMoE; P8's rule (cells host-bound in every repeat at E240)
#: is applied to every model on the page.
LAUNCH_OFFSETS = [
    ("P2 E240 - GR (us)", "rental 1 1 of 15 out; rental 2 E240 - GR 3.6 to 6.7 us; band max(2%, 6 us)",
     "E240", "GR", lambda m, n: (m == "granite-3.0-3b-a800m" and n >= 7) or (m == "jetmoe-8b" and n >= 3)),
    ("P3 E0 - E240 (us)", "+45 to +65 us; rental 2 pooled 10 of 15 out; band +-12 us",
     "E0", "E240", lambda m, n: m == "granite-3.0-3b-a800m" and n <= 2),
    ("P4 E480 - E240 (us)", "shift -54 to -73 us against -67.4; band +-12 us",
     "E480", "E240", lambda m, n: m == "granite-3.0-3b-a800m"),
    ("P8 H - I_E240 (us)", "H - I 75 to 79 us; plateau band [0.23, 0.27] ms",
     "H", "E240", lambda m, n: True),
]


def _launch_cells(d: str, model: str) -> dict:
    """(mode, arm, n) -> {repeat: (ms_p50, H ms per call, host_bound)} of one launch page."""
    out: dict = {}
    p = PUB / d / model / "cells.csv"
    if not p.exists():
        return out
    for r in read_csv(p):
        if r["status"] != "ok" or not r.get("ms_p50"):
            continue
        H = (float(r["host_enqueue_ms"]) / float(r["calls_per_burst"])
             if r.get("host_enqueue_ms") and r.get("calls_per_burst") else None)
        out.setdefault((r["mode"], r["arm"], int(r["tiles"])), {})[int(r["repeat"])] = (
            float(r["ms_p50"]), H, r["host_bound"] == "True")
    return out


def launch_offsets() -> dict:
    """(a) the SE of each launch offset from the 3 repeats of its cells (repeat-paired
    differences, sd / sqrt 3, the mean's SE), and the relative SE of each eager cell
    value (P6); (b) the same offset's cell median, rental 1 against rental 2."""
    models = ("granite-3.0-3b-a800m", "jetmoe-8b", "granite-3.0-1b-a400m", "mixtral-8x7b-tp8")
    pages = {(rental, m): _launch_cells(d, m) for rental, d in LAUNCH.items() for m in models}
    srcs = sorted(rel(PUB / LAUNCH[r] / m / "cells.csv") for (r, m), c in pages.items() if c)
    res = {}
    for metric, quoted, ma, mb, rule in LAUNCH_OFFSETS:
        se, med = [], {}
        for (rental, m), cells in pages.items():
            for (mode, arm, n), reps in cells.items():
                if mode != mb or not rule(m, n):
                    continue
                if ma == "H":
                    if not all(v[2] for v in reps.values()):
                        continue
                    d = [1e3 * (v[1] - v[0]) for v in reps.values() if v[1] is not None]
                else:
                    other = cells.get((ma, arm, n))
                    if not other:
                        continue
                    common = sorted(set(reps) & set(other))
                    d = [1e3 * (other[k][0] - reps[k][0]) for k in common]
                if len(d) >= 2:
                    se.append(st.stdev(d) / math.sqrt(len(d)))
                    med[(rental, m, arm, n)] = st.median(d)
        board = [med[("rental2", m, a, n)] - med[("rental1", m, a, n)]
                 for (r, m, a, n) in med if r == "rental1" and ("rental2", m, a, n) in med]
        res[metric] = {"quoted": quoted, "cells": len(se),
                       "se_median": st.median(se) if se else None,
                       "se_max": max(se) if se else None, "board_cells": len(board),
                       "board_rms": rms(board) if board else None,
                       "board_worst": max(board, key=abs) if board else None}
    rel_se, board6 = [], []
    for (rental, m), cells in pages.items():
        for (mode, arm, n), reps in cells.items():
            if mode in ("E0", "E240", "E480") and len(reps) >= 2:
                v = [x[0] for x in reps.values()]
                rel_se.append(st.stdev(v) / st.mean(v) / math.sqrt(len(v)))
                if rental == "rental1" and (mode, arm, n) in pages.get(("rental2", m), {}):
                    v2 = [x[0] for x in pages[("rental2", m)][(mode, arm, n)].values()]
                    board6.append(st.median(v2) / st.median(v) - 1)
    res["P6 eager I (relative)"] = {
        "quoted": "rental 1 55 of 57 out; rental 2 tp8 5 of 24 out, n=1 7 to 9% over MP; band max(5%, 12 us)",
        "cells": len(rel_se), "se_median": st.median(rel_se), "se_max": max(rel_se),
        "board_cells": len(board6), "board_rms": rms(board6) if board6 else None,
        "board_worst": max(board6, key=abs) if board6 else None}
    return {"metrics": res, "sources": srcs}


# ------------------------------------------------------------ the table
def build() -> list[dict]:
    rows: list[dict] = []

    def add(metric, quoted, method, floor, cells, sources, note):
        rows.append({"metric": metric, "quoted_error": quoted, "method": method,
                     "noise_floor": floor, "cells": cells, "sources": sources, "note": note})

    tr = time_repeat()
    add("time rms % (per cell, predicted/measured - 1)",
        "0.57 to 7.17% rms; bar 2% rms, 5% per cell",
        "a: repeat rows of each scored VALID lock-1710 timed page",
        f"SE of a cell value {100 * tr['median']:.3f}% median, {100 * tr['p90']:.3f}% p90, "
        f"{100 * tr['rms']:.3f}% rms",
        tr["cells"], "; ".join(sorted({s.rsplit('/', 2)[0] for s in tr["sources"]})),
        f"the 2% bar is {0.02 / tr['rms']:.0f}x the pooled rms; per model rms "
        + ", ".join(f"{m} {100 * v:.3f}%" for m, v in tr["per_model_rms"].items()))
    rt = time_retake()
    for label, gate, n, sig, ng, sg, src in rt["pairs"]:
        add(f"time per cell, retake {label}", "as above", f"c: retake pair ({gate})",
            f"single-page sigma {100 * sig:.2f}% (all {n} cells); GPU-bound cells {100 * sg:.2f}% ({ng})",
            n, src,
            "page-to-page on one board; gate-failed pages are replicates of the cell value "
            "only, never scored" + ("; the host-bound plateau cells move with the host, "
                                    "not the GPU" if label.startswith("Granite") else ""))
    add("time per cell, retakes pooled (8x22B, Qwen2-57B; GPU-bound cells)", "as above",
        "c: the three non-Granite retake pairs", f"single-page sigma "
        f"{100 * rt['gpu_bound_nongranite_sigma']:.2f}%", rt["gpu_bound_nongranite_cells"],
        "; ".join(rt["sources"][:3]), "")
    tb = time_board()
    add("time per cell, Mixtral 8x7B board to board", "as above",
        "b: 2026-09-25 (1310e2) against 2026-09-27 (9b6d01), G=2 and 4, n=1..6, SHARED+PRIVATE",
        f"rms difference {100 * tb['rms_diff']:.2f}% (single-board sigma "
        f"{100 * tb['sigma_single']:.2f}%), mean {100 * tb['mean']:+.2f}%, worst "
        f"{100 * tb['worst']:+.2f}%", tb["cells"], "; ".join(tb["sources"]),
        "one model, two G: the only board-to-board time replicate published")
    add("time per cell, between pages on one board: OLMoE, Qwen1.5, Phi-3.5, JetMoE",
        "as above", "none", NONE, 0, "",
        "none of these four has two pages at one G and clock; the only retakes are 8x22B, "
        "Qwen2-57B and Granite-3B (above), and the only board-to-board time pair is 8x7B")

    br = bytes_repeat()
    add("byte error % (PRIVATE per GEMM)", "0.17 to 3.02% rms; bar 5% per cell",
        "a: per-call values of every lock-1710 GH200 counter page 2026-09-2x (3 calls)",
        f"SE of a cell value {100 * br['median']:.4f}% median, {100 * br['p90']:.4f}% p90, "
        f"{100 * br['max']:.3f}% max", br["cell_gemms"],
        f"{br['pages']} pages under results/published/2026-09-2*-nvidia_gh200*/results/*-r3-counters/lock1710/",
        "registered noise model beside it: sigma_q 0.036 (90th pct of per-call se of PRIVATE w2 q, "
        "D/README.md T2)")
    bb = bytes_board()
    for G, boards, arm, n, r, w in bb["rows"]:
        add(f"bytes per cell, Mixtral 8x7B G={G} {arm}, board to board", "as above",
            f"b: {boards}", f"rms difference {100 * r:.2f}%, worst {100 * w:+.2f}%", n,
            "see BOARD_BYTES in scripts/paper/noise_floors.py",
            "bytes are clock-free; 2026-09-25's pages are base clock, the others 1710")
    add("bytes, PRIVATE, pooled board to board (8x7B)", "as above", "b: every pair above",
        f"rms difference {100 * rms(bb['private']):.2f}%, worst "
        f"{100 * max(bb['private'], key=abs):+.2f}%", len(bb["private"]),
        "; ".join(sorted(set(bb["sources"]))), "single-board sigma = rms / sqrt 2")
    nr = bytes_repeat("native")
    add("byte error % (NATIVE per GEMM)",
        "printed in the w1 all-arms and SHARED+NATIVE w2 G<=16 n<=4 sets (8x22B, Qwen2-57B); bar 5% per cell",
        "a: per-call values of every lock-1710 GH200 counter page 2026-09-2x (3 calls)",
        f"SE of a cell value {100 * nr['median']:.4f}% median, {100 * nr['p90']:.4f}% p90, "
        f"{100 * nr['max']:.3f}% max", nr["cell_gemms"],
        f"{nr['pages']} pages under results/published/2026-09-2*-nvidia_gh200*/results/*-r3-counters/lock1710/",
        "NATIVE added 2026-10-05 (gradefix4)")
    nb = bytes_board(("native",))
    nb2 = [x for G, pa, pb, _b in BOARD_BYTES if G >= 2
           for x in _board_diffs(pa, pb, "native")]
    add("bytes, NATIVE, pooled board to board (8x7B), G >= 2", "as above",
        "b: every BOARD_BYTES pair at G >= 2",
        f"rms difference {100 * rms(nb2):.2f}%, worst {100 * max(nb2, key=abs):+.2f}%",
        len(nb2), "; ".join(sorted(set(nb["sources"]))),
        "single-board sigma = rms / sqrt 2; NATIVE added 2026-10-05 (gradefix4)")
    nb1d = [x for G, pa, pb, _b in BOARD_BYTES if G == 1 for x in _board_diffs(pa, pb, "native")]
    add("bytes, NATIVE, pooled board to board (8x7B), G = 1", "as above",
        "b: every BOARD_BYTES pair at G = 1",
        f"rms difference {100 * rms(nb1d):.2f}%, worst {100 * max(nb1d, key=abs):+.2f}%",
        len(nb1d), "; ".join(sorted(set(nb["sources"]))),
        "single-board sigma = rms / sqrt 2; NATIVE added 2026-10-05 (gradefix4)")

    fm = floor_metric()
    add("floor cycles per CTA k-step (slope over n = 2, 3, 4, 6)",
        "-3.6 to +3.6% against registration; bar 2%",
        "a: base, unlocked and 1710 captures of the same cells, one board each",
        f"capture-to-capture rms {100 * rms(fm['clock_diffs']):.2f}%, worst "
        f"{100 * max(fm['clock_diffs'], key=abs):+.2f}%", len(fm["clock_diffs"]),
        "; ".join(fm["sources"]),
        "the clock differs between the captures; the slope is in SM cycles")
    for model, cap, g, va, vb, d in fm["board"]:
        add(f"floor, {model} {g} {cap}, board to board", "as above",
            "b: 9b6d01 or d67185 against 594c0f",
            f"{va:.1f} / {vb:.1f} cycles, {100 * d:+.2f}%", 4,
            "see FLOOR_SESSIONS in scripts/paper/noise_floors.py", "same treads n = 2, 3, 4, 6")

    sm = slope_metric()
    add("G >= 8 SHARED slope (ms per tread, n = 2..6)", "-1.4 to +6.8%; bar 2%",
        "a: the slope per repeat index on each scored VALID G >= 8 page",
        f"SE of a page slope {100 * sm['median']:.3f}% median, {100 * sm['max']:.3f}% max",
        sm["pages"], "; ".join(sm["sources"]), "")
    for label, gate, a, b, d in sm["retake"]:
        add(f"G >= 8 slope, retake {label}", "as above", f"c: retake pair ({gate})",
            f"{a:.4f} / {b:.4f} ms, {100 * d:+.2f}%", 5,
            "see RETAKES in scripts/paper/noise_floors.py",
            "Granite's treads 2..5 are host-bound" if "Granite" in label else "")

    D = tp8_D()
    add("tp8 F1 D (w1 cycles n=4 minus n=2)", "156,941 / 155,820 vs 155,193; tolerance 11,345",
        "a: base against 1710 capture of the same cells, G=64 and G=8 (rental 1, 7269a7)",
        f"G=64 {D['r3f-g64.json']:,.0f} / {D['r3f-g64-lock1710.json']:,.0f}; G=8 "
        f"{D['r3f-g8.json']:,.0f} / {D['r3f-g8-lock1710.json']:,.0f} (base / 1710)", 4,
        f"results/published/{TP8_R1}/r3f-g{{64,8}}{{,-lock1710}}.json",
        "spread across the four readings "
        f"{max(D.values()) - min(D.values()):,.0f} cycles, against the 11,345 tolerance")

    sv = survival_metric()
    add("L2 survival s (G=1, per n)", "bands +-0.05 / 0.08; rental-2 lag band +-0.03 per s",
        "a: per-call values propagated (SHARED and PRIVATE, 3 calls each)",
        f"SE of s {sv['call_se_median']:.4f} median, {sv['call_se_max']:.4f} max",
        "", "; ".join(sv["sources"]), "")
    for label, bds, n, r, w in sv["rows"]:
        add(f"L2 survival s, {label}, board to board", "as above", f"b: {bds}, n = 4..9",
            f"rms difference {r:.3f}, worst {w:+.3f}", n,
            "see SURVIVAL in scripts/paper/noise_floors.py", "")

    t2 = t2_metric()
    for label, bds, qa, qb, fa, fb, d in t2["rows"]:
        add(f"T2 f (PRIVATE w2 n=9 re-read fraction), {label}, board to board",
            "rho42 1.199, rho84 1.184 against bands R1 ~1.0, R2 1.4 to 1.9",
            f"b: {bds}", f"q {qa:.3f} / {qb:.3f}; f {fa:.4f} / {fb:.4f} ({100 * d:+.1f}%)", 1,
            "; ".join(s for s in t2["sources"]),
            "rho is a same-board ratio, so this board term enters rho only through a "
            "denominator taken on another board; registered noise model sigma_q 0.036")

    lm = launch_metric()
    add("launch GR / C_reg (graph replay)", "P1 band [0.80, 1.10] x C_reg",
        "a: the 3 repeats of each GR cell, rentals 1 and 2",
        f"repeat sd {100 * lm['rep_sd_median']:.2f}% median, {100 * lm['rep_sd_p90']:.2f}% p90",
        lm["cells"], "; ".join(lm["sources"]), "")
    for model, n, r, w in lm["board"]:
        add(f"launch GR, {model}, rental 1 against rental 2", "as above",
            "b: 7269a7 / 4da056, same (arm, n)", f"rms difference {100 * r:.2f}%, worst "
            f"{100 * w:+.2f}%", n, f"results/published/{LAUNCH['rental1']}/{model}/cells.csv; "
            f"results/published/{LAUNCH['rental2']}/{model}/cells.csv",
            "two boards and two host states (rental 1's post-trace host)")
    lo = launch_offsets()
    for metric, m in lo["metrics"].items():
        unit = "%" if "relative" in metric else " us"
        sc = 100 if unit == "%" else 1
        f = (f"SE {sc * m['se_median']:.2f}{unit} median, {sc * m['se_max']:.2f}{unit} max"
             if m["se_median"] is not None else NONE)
        add(f"launch {metric}", m["quoted"], "a: the 3 repeats of each cell, rentals 1 and 2",
            f, m["cells"], "; ".join(lo["sources"]),
            "repeat-paired differences, sd / sqrt 3; added 2026-10-05 (gradefix4)")
        if m["board_cells"]:
            add(f"launch {metric}, rental 1 against rental 2", "as above",
                "b: 7269a7 / 4da056, same (model, arm, n)",
                f"rms difference {sc * m['board_rms']:.2f}{unit}, worst "
                f"{sc * m['board_worst']:+.2f}{unit}", m["board_cells"], "; ".join(lo["sources"]),
                "two boards and two host states (rental 1's post-trace host); added 2026-10-05 (gradefix4)")

    p4 = part4_cells()
    add("part-4 per-cell rms (cycles)", "tp8 4248 / 4104, tp4 4028 / 4423; sigma_cell 2678 (CAL)",
        "a: rental 2 tp8, tp4, tp2 w1 cells, 1710 capture minus base capture, n >= 2",
        f"sd of difference {p4['sd']:,.0f} cycles (single-capture sigma {p4['sigma_single']:,.0f}), "
        f"mean {p4['mean']:+,.0f}", p4["cells"], "; ".join(p4["sources"]),
        "between clocks; a same-clock repeat capture does not exist: " + NONE)

    add("clock elasticity eta", "0.988 at G=4, 0.312 at G=1", "interval of the fit",
        "[0.985, 0.990] and [0.311, 0.313]", "",
        "results/published/2026-09-27-nvidia_gh200_480gb-session/session/README.md; docs/FINDINGS.md 2026-09-27",
        "cited, not recomputed")
    add("K3 intercept ratio", "0.921 against ns form 0.597", "registered noise model",
        "+-0.075 (2 sigma 0.150)", "", "scripts/scoring/rental2/SCORES.md part 3, K3",
        "cited, not recomputed")
    add("part-4 theta", "+0.54 tp8, +0.80 tp4", "registered noise model",
        "sigma 0.276 tp8, 0.200 tp4", "", "scripts/scoring/rental2/SCORES.md part 4",
        "cited, not recomputed")
    return rows


def fmt_cell(v) -> str:
    return str(v).replace("|", "/")


def write(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["metric", "quoted_error", "method", "noise_floor", "cells", "sources", "note"]
    with (out_dir / "noise_floors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    doc = __doc__.split("\n\n", 1)[1].rsplit('"""', 1)[0].strip()
    lines = ["# Noise floors of the paper's error metrics", "",
             "Generated by `python scripts/paper/noise_floors.py` from published pages only "
             "(pinned by `tests/test_paper_noise_floors.py`). Methods: (a) repeat, (b) board, "
             "(c) retake.", "", doc, "",
             "| metric | quoted error | method | noise floor | cells | note |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt_cell(r[k]) for k in
                                         ("metric", "quoted_error", "method", "noise_floor",
                                          "cells", "note")) + " |")
    lines += ["", "## Sources per row", ""]
    for r in rows:
        lines.append(f"- {r['metric']}: {r['sources'] or 'none'}")
    (out_dir / "noise_floors.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "paper")
    a = ap.parse_args(argv)
    rows = build()
    write(rows, a.out_dir)
    for r in rows:
        print(f"{r['metric']}: {r['noise_floor']}  [{r['method']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
