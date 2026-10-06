#!/usr/bin/env python
# ruff: noqa: E501  (commit ids, page paths and quoted docstring text are kept whole)
"""Ledger rows 1 to 9 (OUTLINE section 3.1), scored by the procedure the code
registered, on the 2026-09-27 GH200 pages. WRITTEN 2026-10-05, AFTER THE PAGES.

    python scripts/scoring/session0927/score.py [--repo GIT_REPO] [--out DIR] [--python EXE]

writes DIR/score.json, DIR/score.txt and DIR/prerun/ (default DIR: this directory).
--repo is a clone holding the commits below (default: $MOE_HISTORY_REPO, else this
checkout); it is only read (`git archive`, `git show`, `git grep`, `git merge-base`,
`git rev-parse`). --python runs the pre-run code (default: this interpreter).

WHAT IT DOES. Rows 1 to 9 were registered in two module docstrings, not in
docs/registered/: `scripts/r3_timing_model.py` at fe73508 ("THE REGISTERED
PREDICTIONS", P1 to P8; P1's third clock re-registered as 1605 MHz at f0a831b) and
`scripts/wave_split_bytes.py` at bb979d4 ("WHAT WOULD FALSIFY IT"). Some numbers are
literal in those docstrings (group a). The rest are what the registered procedure
computes from the 2026-09-25 GH200 pages (published at a49a2a4) and were never
printed in a docstring before the pages (group b). This scorer

  1. checks the M3 facts: commit times, `git merge-base --is-ancestor` of each
     registering commit against the run commit 161f9ec and the publishing commit
     c7fd3db, and the git tree ids of both sessions' page directories at every
     commit involved (the 2026-09-25 pages are one tree from a49a2a4 to 161f9ec);
  2. extracts f0a831b, fe73508 and bb979d4 with `git archive` into a temporary
     directory and runs THAT commit's tool on the 2026-09-25 pages, through
     `prerun.py` (glue only), recomputing every group-(b) value; the captured
     printouts are prerun/*.txt;
  3. checks every group-(a) quote at its file and line in the registering
     commit's docstring, and searches each group-(b) value's printed form in the
     registering commit's tool and test files (a hit in a test file is a test pin,
     committed before the run but not in the docstring: reported, still group b);
  4. reads the 2026-09-27 pages from c7fd3db (timed pages through f0a831b's own
     loader, byte pages through bb979d4's, floor captures as JSON) and scores rows
     1 to 9 by the registered rules;
  5. compares every number and verdict with the hand scoring in the session
     README (typed in HAND below, for the comparison only; no verdict here is
     read from it) and prints DIFFERS where they part.

RULES. A rule marked REGISTERED is the registration's own threshold. Where the
registered words carry no number ("agree", "flat", "like the others", "far",
"near"), the rule is marked READING and states the number used. The one tolerance
the timing registration states for a timed quantity is P2's band, [0.98, 1.01] x
the model's value (fe73508 docstring lines 152 to 156); readings R0 use it. A
READING verdict is the scorer's, not the registration's.

Nothing is fitted on the 2026-09-27 pages. The only computation beyond the pre-run
tools' is arithmetic on page values (OLS slopes by the pre-run `_slope`,
increments, medians, ratios) and the eta inversion of row 1 (bisection on the
pre-run model's own prediction function, in prerun.py).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

C = {"pages_0925": "a49a2a46e7af2bb66d5a9f75b5b7dd34d92c16a8",
     "timing": "fe73508997c0a413e05b63ba8265ad3ad6aae5b9",
     "wsc": "bb979d4a604b89e513924a37f4a9c53579e982a2",
     "p1_1605": "f0a831b66fa381c1363f52621effd54026ff1ece",
     "run": "161f9ec19347efb7aa64ac4c645c527e0d2b46d6",
     "pages_0927": "c7fd3db259ef1081f1d3419a736b47523f971252"}
S0925 = "results/published/2026-09-25-nvidia_gh200_480gb-session"
S0927 = "results/published/2026-09-27-nvidia_gh200_480gb-session"
CNT0925 = f"{S0925}/results/2026-09-25-nvidia_gh200_480gb-r3-counters"
CNT0927 = f"{S0927}/results/2026-09-27-nvidia_gh200_480gb-r3-counters"
GH200_RUNS = ("d9f1f37c", "df37ea07", "01c08abd", "1b285de2", "6ff34777")
TIMING_FILES = ("scripts/r3_timing_model.py", "tests/test_r3_timing_model.py")
WSC_FILES = ("scripts/wave_split_bytes.py", "tests/test_wave_split_bytes.py")
FLOOR_PAGES = ("r3f-g2.json", "r3f-g64.json", "r3f-g64-unlocked.json", "r3f-g64-lock1710.json")
#: Warps an sm_90 SM holds (the 100% of sm__warps_active.avg.pct_of_peak_sustained_active).
MAX_WARPS_SM90 = 64
NUM_WARPS = 8
K_STEPS = {"w1": 64, "w2": 224}
NPN = {"w1": 448, "w2": 64}
#: Reading R0: P2's registered band ratio, the one tolerance the timing registration states.
R0 = (0.98, 1.01)
#: P2's band width, 0.556 - 0.538: the registration's flatness tolerance in ms.
P2_WIDTH = 0.018
#: Reading for P3's "flat in n": the OLS change over n = 3..9 below half the 0.10 ms floor.
P3_TREND_MAX = 0.05
#: Readings for P6's "near the limit" and "far from 354", beside R0.
P6_NEAR_LOOSE, P6_FAR_LOOSE = 0.90, 0.05

#: The hand scoring, typed from the session README "What it answered (registered
#: 2026-09-25, scored on these pages)" (results/published/2026-09-27-nvidia_gh200_480gb-session/session/README.md
#: lines 74 to 99) FOR THE COMPARISON ONLY. Nothing below reads a verdict from it.
HAND = {
    1: {"verdict": "HELD", "G4_slopes": [0.6598, 0.6222, 0.5812],
        "reg_eta1": [0.6654, 0.6259, 0.5854], "reg_eta035": [0.5878, 0.5754, 0.5621],
        "G2_inc_1410": [0.541, 0.732, 0.591, 0.731, 0.595],
        "reg_G2_eta1": [0.550, 0.731, 0.600, 0.733, 0.598]},
    2: {"verdict": "HELD", "claim": "G=8 and G=32 SHARED steps 2 to 6 all inside [0.538, 0.556]"},
    3: {"verdict": "HELD", "excess": [0.154, 0.153, 0.162, 0.160], "reg": [0.143, 0.145, 0.144, 0.151]},
    4: {"verdict": "HELD", "inc_8_9": 0.508, "reg": 0.504},
    5: {"verdict": "FALSIFIED", "inc_2_8": [0.491, 0.576, 0.550, 0.516, 0.580, 0.560]},
    6: {"verdict": "HELD", "w1": [350, 352], "w2": [354, 367], "reg": 353.8},
    7: {"verdict": "HELD", "q": [1.078, 1.100], "reg": [1.082, 1.118]},
    8: {"verdict": "FALSIFIED", "pct": {"G16": [7.5, 12.6, 16.3], "G4 n8": [5.9], "G2": [6.0, 6.2, 13.2]},
        "claim": "w2 SHARED at n >= 5 under-predicted by more than 5% at G=16 n=5,7,8, G=4 n=8, G=2; every w1 and PRIVATE cell within 1.5% except w1 SHARED G=64"},
    9: {"verdict": "not answered"},
}

#: Group (a): text literal in a registering commit's module docstring. Each quote is
#: checked at its line there; `value` ties it to the recomputed number where one exists.
QUOTES = [
    (1, "P1 clocks and etas", "p1_1605", TIMING_FILES[0], "P1  a held lock at 1410, 1500 and 1605 MHz: the G=4 SHARED slope 2-6, and"),
    (1, "P1 etas", "timing", TIMING_FILES[0], "the G=2 increments at 1410, at eta 1 (a fixed cycle count) and at eta"),
    (1, "P1 agreement rule", "timing", TIMING_FILES[0], "and the eta the G=2 low steps give must agree."),
    (2, "P2 n=1 to 2 step 0.315 ms", "timing", TIMING_FILES[0], "point prediction and is not held to the band: on the GH200 it is 0.315"),
    (2, "P2 band [0.538, 0.556]", "timing", TIMING_FILES[0], "ms against a band of [0.538, 0.556]. The judge stated that band and"),
    (2, "P2 band rule [0.98, 1.01]", "timing", TIMING_FILES[0], "for it. [0.98, 1.01] x the model's slope, rounded outward to 3"),
    (3, "P3 at least 0.10 ms, flat", "timing", TIMING_FILES[0], "P3  G=2 at n=7 to 10, and its odd-n excess over G=4: at least 0.10 ms and"),
    (4, "P4 q_w2 1.031 to 1.003", "timing", TIMING_FILES[0], "P4  G=4 n=8 to 9, where q_w2 drops (1.031 to 1.003 at 132 SMs)."),
    (5, "P5 flat above 2/3", "timing", TIMING_FILES[0], "G=3 group has (2/3), a period-3 ripple when it is below."),
    (6, "P6 353.8 cycles", "timing", TIMING_FILES[0], "WHAT IT CANNOT TELL. Which unit sets the floor: 353.8 cycles per CTA k-step"),
    (6, "P6 falsifier (AND form)", "timing", TIMING_FILES[0], "- P6 showing achieved occupancy near the occupancy limit AND cycles per CTA"),
    (6, "P6 falsifier, 354", "timing", TIMING_FILES[0], "k-step far from 354;"),
    (7, "row 7 ceiling 1.12, WSC 1.082", "wsc", WSC_FILES[0], "- GH200 w2 SHARED at G=8, n = 2 or 4, above 1.12 (it predicts 1.082 and"),
    (7, "row 7 WSC 1.118, LRU 1.139 and 1.159", "wsc", WSC_FILES[0], "1.118; the LRU rival predicts 1.139 and 1.159);"),
    (8, "row 8 rule", "wsc", WSC_FILES[0], "- n=5 or n=8 at G = 2, 4 or 16 outside +-5% of its registered predictions;"),
    (9, "P7 1782 and 1787 MHz", "timing", TIMING_FILES[0], "two such pages: 1782 and 1787 MHz against NVML's 1935). That reading"),
    (9, "P8 H100 bytes", "timing", TIMING_FILES[0], "tread, against the byte source's (the H100 at G=1: 1.32 to 4.84 W_w2"),
]
#: Text the pre-run tool PRINTS (not docstring): the falsifier lines rows 1, 4, 5, 6 read.
#: Digit strings that match a group-(b) value but name another quantity.
COINCIDENT = {("bb979d4", "scripts/wave_split_bytes.py", 173):
              "the LRU rival's G=8 n=4 prediction (1.159), not WSC's G=16 n=8 number"}
PRINTED = [
    (1, "timing", "falsified if: a G=4 slope that neither eta gives (between them, near "),
    (4, "timing", "falsified if: n=8->9 reads like the others (then the q term is wrong)"),
    (5, "timing", "falsified if: the ladder does the other"),
    (6, "timing", "falsified if: cycles far from this, or achieved occupancy near the limit "),
    (6, "timing", "\"P6: achieved occupancy near the limit and cycles per CTA k-step far from 354\","),
]


# ----------------------------------------------------------------------- git

class Refused(SystemExit):
    pass


def git(repo: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                       env={**os.environ, "TZ": "UTC"})
    if check and r.returncode:
        raise Refused(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def has(repo: Path, sha: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True).returncode == 0


def ancestor(repo: Path, a: str, b: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", a, b],
                          capture_output=True).returncode == 0


def tree_id(repo: Path, sha: str, path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "-q", "--verify", f"{sha}:{path}"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def extract(repo: Path, sha: str, dest: Path, *paths: str) -> Path:
    dest.mkdir(parents=True)
    arc = subprocess.run(["git", "-C", str(repo), "archive", sha, *paths], capture_output=True,
                         check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arc, check=True)
    return dest


def facts(repo: Path) -> dict:
    when = {k: git(repo, "show", "-s", "--format=%cI", v).strip() for k, v in C.items()}
    utc = {k: git(repo, "show", "-s", "--format=%cd", "--date=format-local:%Y-%m-%d %H:%M:%S",
                  v).strip() for k, v in C.items()}
    anc = {}
    for a in ("pages_0925", "timing", "wsc", "p1_1605"):
        for b in ("timing", "wsc", "p1_1605", "run", "pages_0927"):
            if a != b:
                anc[f"{a} -> {b}"] = ancestor(repo, C[a], C[b])
    trees = {k: tree_id(repo, C[k], S0925) for k in ("pages_0925", "timing", "wsc", "p1_1605", "run", "pages_0927")}
    t27 = {k: tree_id(repo, C[k], S0927) for k in ("run", "pages_0927")}
    return {"commits": dict(C), "committer_time": when, "committer_time_utc": utc,
            "ancestry": anc, "tree_0925_pages": trees,
            "tree_0925_pages_one_tree": len({v for v in trees.values()}) == 1,
            "tree_0927_pages": t27}


# ------------------------------------------------------------- literal search

def docstring_span(text: str) -> tuple[int, int]:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines, 1) if ln.lstrip().startswith('"""'))
    end = next(i for i, ln in enumerate(lines, 1) if i > start and '"""' in ln)
    return start, end


def quotes(repo: Path) -> list[dict]:
    out = []
    for row, what, key, path, text in QUOTES:
        body = git(repo, "show", f"{C[key]}:{path}")
        lo, hi = docstring_span(body)
        hits = [i for i, ln in enumerate(body.splitlines(), 1) if text in ln]
        out.append({"row": row, "what": what, "commit": C[key][:7], "file": path,
                    "lines": hits, "in_docstring": bool(hits) and all(lo <= i <= hi for i in hits),
                    "docstring_lines": [lo, hi], "text": text})
    for row, key, text in PRINTED:
        body = git(repo, "show", f"{C[key]}:{TIMING_FILES[0]}")
        lo, hi = docstring_span(body)
        hits = [i for i, ln in enumerate(body.splitlines(), 1) if text in ln]
        out.append({"row": row, "what": "printed by the tool (not docstring)", "commit": C[key][:7],
                    "file": TIMING_FILES[0], "lines": hits,
                    "in_docstring": bool(hits) and all(lo <= i <= hi for i in hits),
                    "docstring_lines": [lo, hi], "text": text})
    return out


def search(repo: Path, key: str, files, needle: str) -> list[dict]:
    hits = []
    for path in files:
        body = git(repo, "show", f"{C[key]}:{path}", check=False)
        if not body:
            continue
        lo, hi = docstring_span(body)
        for i, ln in enumerate(body.splitlines(), 1):
            if needle in ln:
                hits.append({"commit": C[key][:7], "file": path, "line": i,
                             "in_docstring": lo <= i <= hi, "text": ln.strip()})
    return hits


# ------------------------------------------------------------------ pre-run

def run_prerun(py: str, tree: Path, mode: str, out: Path, txt: Path, pages: Path, *rest: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    r = subprocess.run([py, "-I", str(HERE / "prerun.py"), mode, str(tree), str(out), str(txt),
                        str(pages), *rest], cwd=tree, capture_output=True, text=True, env=env)
    if r.returncode:
        raise Refused(f"pre-run {mode} at {tree.name} failed:\n{r.stderr[-3000:]}")
    return json.loads(out.read_text())


# --------------------------------------------------------------------- score

def ols(ys, x0=2):
    xs = list(range(x0, x0 + len(ys)))
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum((x - mx) ** 2 for x in xs)


def r0(v, ref):
    return R0[0] <= v / ref <= R0[1]


def ladder(meas, G, mhz):
    rows = [d for d in meas if d["G"] == G and d["clocks"] == [mhz] and d["label"] == "VALID"]
    if not rows:
        return None, []
    ns = sorted({int(n) for d in rows for n in d["shared_ms"]})
    return ({n: statistics.median(d["shared_ms"][str(n)] for d in rows) for n in ns},
            [d["run"] for d in rows])


def incs(lad):
    ns = sorted(lad)
    return {n: lad[n + 1] - lad[n] for n in ns if n + 1 in lad}


def same(a, b, nd):
    return round(a, nd) == round(b, nd)


def row1(T, meas):
    P = T["predictions"]
    clocks = P["P1_clocks"]
    out = {"rule": "READING R0 for 'gives' and 'agree' (no number registered); falsifier as printed: "
                   "'a G=4 slope that neither eta gives ... or the eta from the G=4 slope and the eta "
                   "from the G=2 low steps disagreeing'", "G4": {}, "G2": {}}
    g4_etas = []
    for f in clocks:
        lad, runs = ladder(meas, 4, f)
        s = ols([lad[n] for n in range(2, 7)])
        pe = {e: P["P1"][f"G4/f{f:.0f}/eta{e}"]["slope_2_6"] for e in (1.0, 0.35)}
        gives = [e for e, v in pe.items() if r0(s, v)]
        g4_etas.append(set(gives))
        out["G4"][f"{f:.0f}"] = {"runs": runs, "measured_slope_2_6": s,
                                 "endpoint_slope_(T6-T2)/4": (lad[6] - lad[2]) / 4,
                                 "registered": pe, "ratio_eta1": s / pe[1.0],
                                 "ratio_eta0.35": s / pe[0.35], "etas_giving_it_R0": gives,
                                 "eta_inverted": T["predictions"]["eta_inversion"][f"G4/f{f:.0f}"]["eta"]}
    lad, runs = ladder(meas, 2, clocks[0])
    inc = [lad[n + 1] - lad[n] for n in range(1, 6)]
    low = [inc[i] for i in (0, 2, 4)]
    g2 = {}
    for e in (1.0, 0.35):
        reg = P["P1"][f"G2/f{clocks[0]:.0f}/eta{e}"]["inc"]
        g2[str(e)] = {"registered_inc": reg, "low_steps_all_R0": all(r0(m, r) for m, r in zip(low, [reg[i] for i in (0, 2, 4)], strict=True))}
    g2_etas = {float(e) for e, v in g2.items() if v["low_steps_all_R0"]}
    out["G2"] = {"runs": runs, "measured_inc": inc, "low_steps": low, "by_eta": g2,
                 "etas_giving_low_steps_R0": sorted(g2_etas),
                 "eta_inverted": P["eta_inversion"][f"G2/f{clocks[0]:.0f}/low_steps"]["eta"]}
    every_given = all(g4_etas)
    common = set.intersection(*g4_etas, g2_etas) if every_given else set()
    out["neither_eta_gives_a_G4_slope"] = not every_given
    out["common_eta_R0"] = sorted(common)
    out["verdict"] = "HELD" if every_given and common else "FALSIFIED"
    out["verdict_eta"] = sorted(common)
    return out


def row2(T, meas):
    P = T["predictions"]["P2"]
    out = {"rule": "REGISTERED: every SHARED per-tread increment from n=2 to n=6 and the slope 2-6 inside "
                   "the band, at G=8 and G=32 (fe73508 docstring P2; the printed falsifier)", "G": {}}
    bad = []
    for G in (8, 32):
        lo, hi = P[str(G)]["band"]
        lad, runs = ladder(meas, G, 1710.0)
        inc = incs(lad)
        vals = {f"n{n}->{n + 1}": inc[n] for n in range(2, 6)}
        vals["slope_2_6"] = ols([lad[n] for n in range(2, 7)])
        outside = {k: v for k, v in vals.items() if not (lo <= v <= hi)}
        bad += [f"G={G} {k} {v:.5f}" for k, v in outside.items()]
        out["G"][str(G)] = {"runs": runs, "band": [lo, hi], "values": vals, "outside": outside,
                            "inside_if_rounded_to_3_decimals": all(lo <= round(v, 3) <= hi for v in vals.values()),
                            "registered_step_1_2": P[str(G)]["step_1_2"], "measured_step_1_2": inc[1]}
    out["outside"] = bad
    out["verdict"] = "FALSIFIED" if bad else "HELD"
    return out


def row3(T, meas):
    P = T["predictions"]["P3"]
    g2, r2 = ladder(meas, 2, 1710.0)
    g4, r4 = ladder(meas, 4, 1710.0)
    ns = (3, 5, 7, 9)
    ex = {n: g2[n] - g4[n] for n in ns}
    xs = list(ns)
    mx, my = statistics.fmean(xs), statistics.fmean(ex.values())
    slope = sum((x - mx) * (ex[x] - my) for x in xs) / sum((x - mx) ** 2 for x in xs)
    change = slope * (ns[-1] - ns[0])
    floor_ok = min(ex.values()) >= P["min_excess_ms"]
    flat = abs(change) < P3_TREND_MAX
    return {"rule": f"REGISTERED: every odd-n excess >= {P['min_excess_ms']} ms; READING for 'flat in n': "
                    f"|OLS change over n=3..9| < {P3_TREND_MAX} ms (half the registered floor)",
            "runs": {"G2": r2, "G4": r4}, "measured_excess": {str(n): v for n, v in ex.items()},
            "registered_excess": P["odd_n_excess"], "ols_change_3_to_9": change,
            "at_least_floor": floor_ok, "flat": flat,
            "verdict": "HELD" if floor_ok and flat else "FALSIFIED"}


def row4(T, meas):
    P = T["predictions"]["P4"]
    g4, runs = ladder(meas, 4, 1710.0)
    inc = incs(g4)
    i89 = inc[8]
    others = [inc[n] for n in range(2, 8)]
    med = statistics.median(others)
    like = r0(i89, med)
    return {"rule": "READING R0 for 'reads like the others': n=8->9 within [0.98, 1.01] x the median of the "
                    "other measured increments (n=2->3 to 7->8; the registered median skips n=1->2 and n=8->9)",
            "runs": runs, "measured_inc_8_9": i89, "registered_inc_8_9": P["inc_8_9"],
            "ratio_to_registered": i89 / P["inc_8_9"], "measured_others_median": med,
            "registered_others_median": P["inc_others_median"], "ratio_to_others": i89 / med,
            "reads_like_the_others": like, "verdict": "FALSIFIED" if like else "HELD"}


def phase_amplitude(inc: dict) -> tuple[float, dict]:
    ph = {}
    for n, v in inc.items():
        ph.setdefault(n % 3, []).append(v)
    means = {k: statistics.fmean(v) for k, v in sorted(ph.items())}
    return max(means.values()) - min(means.values()), means


def row5(T, meas):
    P = T["predictions"]["P5"]
    g3, runs = ladder(meas, 3, 1710.0)
    inc = incs(g3)
    m = {n: inc[n] for n in range(2, 8)}
    reg = {n: P["G3_inc"][n - 1] for n in range(2, 8)}
    am, pm = phase_amplitude(m)
    ar, pr = phase_amplitude(reg)
    ripple = am > ar + P2_WIDTH
    return {"rule": "REGISTERED: the model calls G=3 flat (rho*_w1 above 2/3); falsified if the ladder does the "
                    f"other. READING for 'a period-3 ripple': the increments n=2->3 to 7->8 grouped by n mod 3; "
                    f"ripple if the measured phase-mean range exceeds the registered ladder's own by more than "
                    f"{P2_WIDTH} ms (P2's band width)",
            "runs": runs, "registered_flat": P["flat"], "rho_star_w1": P["rho_star_w1"],
            "slab_fraction_max_G3": P["slab_fraction_max_G3"],
            "measured_inc": {str(n): v for n, v in m.items()}, "registered_inc": {str(n): v for n, v in reg.items()},
            "measured_phase_means": {str(k): v for k, v in pm.items()}, "measured_amplitude": am,
            "registered_phase_means": {str(k): v for k, v in pr.items()}, "registered_amplitude": ar,
            "ripple": ripple, "verdict": "FALSIFIED" if (ripple and P["flat"]) else "HELD"}


def floor_cells(tree: Path) -> list[dict]:
    out = []
    for name in FLOOR_PAGES:
        d = json.loads((tree / CNT0927 / name).read_text())
        for c in d["cells"]:
            for g in ("w1", "w2"):
                m = c["per_gemm"][g]
                live = 8 * c["n"] * NPN[g]
                out.append({"page": name, "clock": d["clock"].get("control"),
                            "lock_mhz": d["clock"].get("lock_mhz"),
                            "stored_gates": [(x["number"], x["verdict"]) for x in d["gates"]],
                            "arm": c["arm"], "n": c["n"], "gemm": g, "grid": m["launch__grid_size"],
                            "live_ctas": live,
                            "cycles_per_cta_kstep": m["sm__cycles_active.avg"] * 132 / (live * K_STEPS[g]),
                            "warps_active_pct": m["sm__warps_active.avg.pct_of_peak_sustained_active"]})
    return out


def recorded_limit(tree: Path) -> dict:
    lim = {}
    for p in sorted((tree / CNT0927 / "lock1710").glob("r3c-g*.json")):
        for c in json.loads(p.read_text())["cells"]:
            for g, rec in c["recorded"].items():
                v = min(x for k, x in rec.items() if k.startswith("launch__occupancy_limit_"))
                lim.setdefault(g, set()).add(v)
    if any(len(v) != 1 for v in lim.values()):
        raise Refused(f"occupancy limits differ across the lock pages: {lim}")
    return {g: next(iter(v)) for g, v in lim.items()}


def row6(T, tree):
    P = T["predictions"]["P6"]
    reg = P["cycles_per_cta_kstep"]
    lim = recorded_limit(tree)
    cells = floor_cells(tree)
    for c in cells:
        c["achieved_ctas_per_sm"] = c["warps_active_pct"] / 100 * MAX_WARPS_SM90 / NUM_WARPS
        c["achieved_over_limit"] = c["achieved_ctas_per_sm"] / lim[c["gemm"]]
        c["dev"] = c["cycles_per_cta_kstep"] / reg - 1
    rng = {g: [min(c["cycles_per_cta_kstep"] for c in cells if c["gemm"] == g),
               max(c["cycles_per_cta_kstep"] for c in cells if c["gemm"] == g)] for g in ("w1", "w2")}
    occ = {g: [min(c["achieved_over_limit"] for c in cells if c["gemm"] == g),
               max(c["achieved_over_limit"] for c in cells if c["gemm"] == g)] for g in ("w1", "w2")}

    def verdicts(near_fn, far_fn):
        both = [c for c in cells if near_fn(c) and far_fn(c)]
        either = [c for c in cells if near_fn(c) or far_fn(c)]
        return {"AND form (docstring falsifier list, FALSIFIERS)": "FALSIFIED" if both else "HELD",
                "OR form (printed falsified-if line)": "FALSIFIED" if either else "HELD",
                "cells_near": sum(map(near_fn, cells)), "cells_far": sum(map(far_fn, cells)),
                "cells_both": [f"{c['page']} n={c['n']} {c['gemm']}" for c in both]}
    loose = verdicts(lambda c: c["achieved_over_limit"] >= P6_NEAR_LOOSE,
                     lambda c: abs(c["dev"]) >= P6_FAR_LOOSE)
    strict = verdicts(lambda c: R0[0] <= c["achieved_over_limit"] <= R0[1],
                      lambda c: not (R0[0] <= 1 + c["dev"] <= R0[1]))
    vs = {v for d in (loose, strict) for k, v in d.items() if k.endswith("form)") or "form (" in k}
    verdict = vs.pop() if len(vs) == 1 else "UNDECIDED"
    return {"rule": "REGISTERED words, no number: falsified by achieved occupancy near the limit AND cycles far "
                    "from 354 (docstring falsifier list and the FALSIFIERS tuple) or, as the tool prints it, cycles "
                    f"far OR occupancy near. READINGS: loose (near >= {P6_NEAR_LOOSE:.0%} of the limit, far >= "
                    f"{P6_FAR_LOOSE:.0%}) and R0 ([0.98, 1.01] for both). UNDECIDED when the forms or readings part.",
            "registered_cycles": reg, "ptx_smem_cycles": P["ptx_smem_cycles"],
            "occupancy_limit_ctas_per_sm": lim, "max_warps_per_sm": MAX_WARPS_SM90, "num_warps": NUM_WARPS,
            "cycles_range": rng, "max_abs_dev": max(abs(c["dev"]) for c in cells),
            "achieved_over_limit_range": occ, "cells": cells,
            "reading_loose": loose, "reading_R0": strict, "verdict": verdict}


def wsc_tables(W):
    pred = {(r["arm"], r["G"], r["n"], r["gemm"]): r for r in W["predicted"]}
    meas = {(r["arm"], r["G"], r["n"], r["gemm"]): r["q"] for r in W["measured_0927"]}
    return pred, meas


def page_gates(tree: Path, G: int) -> list:
    d = json.loads((tree / CNT0927 / "lock1710" / f"r3c-g{G}.json").read_text())
    return [(x["number"], x["verdict"]) for x in d["gates"] if x.get("kind") == "VALIDITY" and x["verdict"] != "PASS"]


def row7(W, tree):
    pred, meas = wsc_tables(W)
    vals = {n: meas[("shared", 8, n, "w2")] for n in (2, 4)}
    return {"rule": "REGISTERED: falsified if GH200 w2 SHARED at G=8, n = 2 or 4 reads above 1.12",
            "measured_q": {str(n): v for n, v in vals.items()},
            "registered_fill": {str(n): pred[("shared", 8, n, "w2")]["q_fill"] for n in (2, 4)},
            "lru_rival_quoted": {"2": 1.139, "4": 1.159},
            "page_stored_validity_not_pass": page_gates(tree, 8),
            "verdict": "FALSIFIED" if any(v > 1.12 for v in vals.values()) else "HELD"}


def row8(W, tree):
    pred, meas = wsc_tables(W)
    cells, outside = [], []
    for G in (2, 4, 16):
        for n in (5, 7, 8):
            for arm in ("shared", "native", "private"):
                for g in ("w1", "w2"):
                    p = pred[("shared" if arm == "native" else arm, G, n, g)]
                    if p["status"] != "PREDICTED" or p["q_fill"] is None:
                        continue
                    q = meas[(arm, G, n, g)]
                    rel = q / p["q_fill"] - 1
                    c = {"arm": arm, "G": G, "n": n, "gemm": g, "q_meas": q, "q_reg": p["q_fill"],
                         "rel_to_registered": rel, "rel_to_measured": 1 - p["q_fill"] / q,
                         "in_falsifier": n in (5, 8),
                         "native_takes_shared": arm == "native"}
                    cells.append(c)
                    if c["in_falsifier"] and abs(rel) > 0.05:
                        outside.append(c)
    by_page = {}
    for c in outside:
        by_page.setdefault(c["G"], []).append(f"{c['arm']} {c['gemm']} n={c['n']} {100 * c['rel_to_registered']:+.1f}%")
    w1p = [c for c in cells if c["gemm"] == "w1" or c["arm"] == "private"]
    return {"rule": "REGISTERED: falsified if n=5 or n=8 at G = 2, 4 or 16 is outside +-5% of the registered "
                    "prediction (relative to the prediction). NATIVE takes SHARED's number, as the pre-run "
                    "printout states; n=7 is printed by the tool but is not in the falsifier",
            "cells": cells, "outside": {str(k): v for k, v in by_page.items()},
            "stored_validity_not_pass": {str(G): page_gates(tree, G) for G in (2, 4, 16)},
            "outside_without_G2": sum(1 for c in outside if c["G"] != 2),
            "w1_and_private_max_abs_rel": max(abs(c["rel_to_registered"]) for c in w1p),
            "w1_and_private_worst": max(w1p, key=lambda c: abs(c["rel_to_registered"])),
            "verdict": "FALSIFIED" if outside else "HELD"}


def row9(T, meas):
    P = T["predictions"]
    unlocked = [d["run"] for d in meas if not d["locked"]]
    return {"rule": "P7 needs an unlocked page of the card among the inputs beside a locked fit; P8 needs "
                    "T3 to trip (the H100 at G=1)",
            "unlocked_0927_timed_pages": unlocked, "P7_registered_pages": P["P7"]["pages"],
            "P8_registered": P["P8"], "verdict": "not answered"}


# ------------------------------------------------------------------- compare

def compare(rows: dict, T: dict, W: dict) -> list[dict]:
    out = []

    def add(row, what, mine, hand, nd=None, note=""):
        if nd is None:
            ok = mine == hand
        elif isinstance(mine, list):
            ok = len(mine) == len(hand) and all(same(a, b, nd) for a, b in zip(mine, hand, strict=True))
        else:
            ok = same(mine, hand, nd)
        out.append({"row": row, "what": what, "scorer": mine, "hand": hand,
                    "agree": ok, "note": note})

    for r in range(1, 10):
        add(r, "verdict", rows[r]["verdict"], HAND[r]["verdict"])
    P = T["predictions"]
    r1 = rows[1]
    add(1, "G=4 slopes 2-6 at 1410/1500/1605", [r1["G4"][f]["measured_slope_2_6"] for f in ("1410", "1500", "1605")],
        HAND[1]["G4_slopes"], 4, "the registered slope 2-6 is the OLS slope (`_slope`); the README's values are (T6 - T2) / 4")
    add(1, "registered eta-1 slopes (recomputed)", [P["P1"][f"G4/f{f}/eta1.0"]["slope_2_6"] for f in ("1410", "1500", "1605")],
        HAND[1]["reg_eta1"], 4)
    add(1, "registered eta-0.35 slopes (recomputed)", [P["P1"][f"G4/f{f}/eta0.35"]["slope_2_6"] for f in ("1410", "1500", "1605")],
        HAND[1]["reg_eta035"], 4)
    add(1, "G=2 increments at 1410", r1["G2"]["measured_inc"], HAND[1]["G2_inc_1410"], 3)
    add(1, "registered G=2 eta-1 increments (recomputed)", P["P1"]["G2/f1410/eta1.0"]["inc"], HAND[1]["reg_G2_eta1"], 3)
    add(3, "odd-n excess", list(rows[3]["measured_excess"].values()), HAND[3]["excess"], 3)
    add(3, "registered excess (recomputed)", list(P["P3"]["odd_n_excess"].values()), HAND[3]["reg"], 3)
    add(4, "G=4 n=8->9", rows[4]["measured_inc_8_9"], HAND[4]["inc_8_9"], 3)
    add(4, "registered n=8->9 (recomputed)", P["P4"]["inc_8_9"], HAND[4]["reg"], 3)
    add(5, "G=3 increments n=2->3..7->8", list(rows[5]["measured_inc"].values()), HAND[5]["inc_2_8"], 3)
    rng = rows[6]["cycles_range"]
    add(6, "w1 cycles range", [round(rng["w1"][0]), round(rng["w1"][1])], HAND[6]["w1"], None,
        "all four floor captures; the README range leaves out r3f-g2 n=3 w1")
    add(6, "w2 cycles range", [round(rng["w2"][0]), round(rng["w2"][1])], HAND[6]["w2"], None)
    add(6, "registered cycles (recomputed)", P["P6"]["cycles_per_cta_kstep"], HAND[6]["reg"], 1)
    add(7, "w2 SHARED G=8 q at n=2, 4", list(rows[7]["measured_q"].values()), HAND[7]["q"], 3)
    add(7, "registered q (recomputed)", list(rows[7]["registered_fill"].values()), HAND[7]["reg"], 3)
    cs = {(c["arm"], c["G"], c["n"], c["gemm"]): c for c in rows[8]["cells"]}
    g16 = [100 * cs[("shared", 16, n, "w2")]["rel_to_registered"] for n in (5, 7, 8)]
    add(8, "w2 SHARED G=16 n=5, 7, 8 (% vs registered)", g16, HAND[8]["pct"]["G16"], 1,
        "the README divides by the measured q: (meas - reg) / meas reads "
        + ", ".join(f"{100 * cs[('shared', 16, n, 'w2')]['rel_to_measured']:.1f}" for n in (5, 7, 8))
        + "; n=7 is not in the registered falsifier")
    add(8, "w2 SHARED G=4 n=8 (%)", [100 * cs[("shared", 4, 8, "w2")]["rel_to_registered"]], HAND[8]["pct"]["G4 n8"], 1)
    add(8, "w2 SHARED G=2 n=5, 7, 8 (%)", [100 * cs[("shared", 2, n, "w2")]["rel_to_registered"] for n in (5, 7, 8)],
        HAND[8]["pct"]["G2"], 1)
    worst = rows[8]["w1_and_private_worst"]
    add(8, "every w1 and PRIVATE cell within 1.5% (README, beside the test)",
        rows[8]["w1_and_private_max_abs_rel"] <= 0.015, True, None,
        f"over the cells the pre-run printout predicts at G = 2, 4, 16 and n = 5, 7, 8 the worst is {worst['arm']} "
        f"{worst['gemm']} G={worst['G']} n={worst['n']} {100 * worst['rel_to_registered']:+.1f}%")
    return out


# --------------------------------------------------------------------- print

def fmt(v, nd=4):
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, list):
        return " ".join(fmt(x, nd) for x in v)
    return str(v)


def lines(doc: dict) -> list[str]:
    F, rows = doc["facts"], doc["rows"]
    L = ["LEDGER ROWS 1 TO 9 (OUTLINE 3.1), SCORED BY THE PROCEDURE REGISTERED IN CODE DOCSTRINGS, ON THE 2026-09-27 GH200 PAGES",
         "scripts/scoring/session0927/score.py, WRITTEN 2026-10-05 AFTER THE PAGES. Readings marked READING are this scorer's.",
         "", "M3 FACTS (git, read only)"]
    for k, v in F["commits"].items():
        L.append(f"  {k:<11} {v[:7]}  committed {F['committer_time_utc'][k]} UTC")
    for k, v in F["ancestry"].items():
        L.append(f"  is-ancestor {k:<26} {v}")
    L.append(f"  2026-09-25 page tree identical at a49a2a4, fe73508, bb979d4, f0a831b, 161f9ec, c7fd3db: {F['tree_0925_pages_one_tree']} ({F['tree_0925_pages']['pages_0925'][:12]})")
    L.append(f"  2026-09-27 page tree at c7fd3db: {(F['tree_0927_pages']['pages_0927'] or '')[:12]} (absent at 161f9ec: {F['tree_0927_pages']['run'] is None})")
    L += ["", "GROUP (a): LITERAL IN A REGISTERING COMMIT'S DOCSTRING (checked at file:line)"]
    for q in doc["quotes"]:
        tag = "docstring" if q["in_docstring"] else "NOT DOCSTRING"
        L.append(f"  row {q['row']} {q['commit']} {q['file']}:{','.join(map(str, q['lines'])) or 'MISSING'} [{tag}] {q['what']}: \"{q['text'].strip()}\"")
    L += ["", "GROUP (b): RECOMPUTED FROM THE PRE-RUN CODE ON THE 2026-09-25 PAGES (printouts: prerun/*.txt)"]
    for b in doc["group_b"]:
        hits = "; ".join(f"{h['commit']} {h['file']}:{h['line']}{' DOCSTRING' if h['in_docstring'] else ''}"
                         + (f" (coincident: {h['coincident']})" if h.get("coincident") else "")
                         for h in b["literal_hits"]) or "none"
        L.append(f"  row {b['row']} [{b['group']}] {b['what']}: {b['printed']}  | literal before the run in: {hits}")
    L.append(f"  fe73508 against f0a831b: P1 at 1410 and 1500 identical, 1600 -> 1605 only; everything else identical: {doc['fe73508_vs_f0a831b']['only_P1_third_clock_differs']}")
    L += ["", "VERDICTS"]
    for r in range(1, 10):
        d = rows[str(r)]
        L.append(f"  row {r}: {d['verdict']}   (hand: {HAND[r]['verdict']})")
        L.append(f"     rule: {d['rule']}")
    r1 = rows["1"]
    for f, d in r1["G4"].items():
        L.append(f"     row 1 G=4 {f}: slope {d['measured_slope_2_6']:.4f} (endpoint {d['endpoint_slope_(T6-T2)/4']:.4f}) vs eta1 {d['registered']['1.0']:.4f}; ratio {d['ratio_eta1']:.4f}; etas giving it {d['etas_giving_it_R0']}; inverted eta {d['eta_inverted']:.3f}")
    L.append(f"     row 1 G=2 1410 increments {fmt(r1['G2']['measured_inc'], 3)}; low steps given by etas {r1['G2']['etas_giving_low_steps_R0']}; inverted eta {r1['G2']['eta_inverted']:.3f}; common eta {r1['common_eta_R0']}")
    for G, d in rows["2"]["G"].items():
        L.append(f"     row 2 G={G}: " + " ".join(f"{k} {v:.5f}" for k, v in d["values"].items()) + f" | band {d['band']} | outside: {', '.join(d['outside']) or 'none'} | inside if rounded to 3 decimals: {d['inside_if_rounded_to_3_decimals']}")
    d = rows["3"]
    L.append(f"     row 3 excess {fmt(list(d['measured_excess'].values()), 4)} vs {fmt(list(d['registered_excess'].values()), 3)}; OLS change n=3..9 {d['ols_change_3_to_9']:+.4f} ms")
    d = rows["4"]
    L.append(f"     row 4 n=8->9 {d['measured_inc_8_9']:.4f} vs {d['registered_inc_8_9']:.4f} (ratio {d['ratio_to_registered']:.4f}); others' median {d['measured_others_median']:.4f} (ratio {d['ratio_to_others']:.4f})")
    d = rows["5"]
    L.append(f"     row 5 increments {fmt(list(d['measured_inc'].values()), 3)} vs {fmt(list(d['registered_inc'].values()), 3)}; phase-mean range {d['measured_amplitude']:.4f} vs registered ladder's {d['registered_amplitude']:.4f}")
    d = rows["6"]
    L.append(f"     row 6 cycles w1 {fmt(d['cycles_range']['w1'], 1)}, w2 {fmt(d['cycles_range']['w2'], 1)} vs {d['registered_cycles']:.1f} (max |dev| {100 * d['max_abs_dev']:.2f}%); achieved occupancy / limit w1 {fmt(d['achieved_over_limit_range']['w1'], 3)}, w2 {fmt(d['achieved_over_limit_range']['w2'], 3)} (limit {d['occupancy_limit_ctas_per_sm']} CTAs per SM)")
    for name in ("reading_loose", "reading_R0"):
        r = d[name]
        L.append(f"     row 6 {name}: AND {r['AND form (docstring falsifier list, FALSIFIERS)']}, OR {r['OR form (printed falsified-if line)']} (near {r['cells_near']}, far {r['cells_far']} of {len(d['cells'])} cells; both: {', '.join(r['cells_both']) or 'none'})")
    for c in d["cells"]:
        L.append(f"        {c['page']:<24} n={c['n']} {c['gemm']} {c['cycles_per_cta_kstep']:7.1f} cycles ({100 * c['dev']:+.2f}%), occupancy {c['achieved_over_limit']:.3f} of the limit")
    d = rows["7"]
    L.append(f"     row 7 q {fmt(list(d['measured_q'].values()), 4)} vs WSC {fmt(list(d['registered_fill'].values()), 3)}, LRU {d['lru_rival_quoted']}, ceiling 1.12; G=8 page stored VALIDITY not PASS: {d['page_stored_validity_not_pass']}")
    d = rows["8"]
    for c in d["cells"]:
        if c["gemm"] == "w2" and c["arm"] != "native":
            mark = "OUTSIDE" if c["in_falsifier"] and abs(c["rel_to_registered"]) > 0.05 else ("(n=7, not in the falsifier)" if not c["in_falsifier"] else "")
            L.append(f"     row 8 {c['arm']:<7} w2 G={c['G']:<2} n={c['n']} meas {c['q_meas']:.3f} reg {c['q_reg']:.3f} {100 * c['rel_to_registered']:+6.1f}% {mark}")
    L.append(f"     row 8 outside (all arms, both GEMMs): {d['outside']}; without the G=2 page (V7 FAIL even re-scored): {d['outside_without_G2']} cells outside; stored VALIDITY not PASS: {d['stored_validity_not_pass']}")
    d = rows["9"]
    L.append(f"     row 9 unlocked 2026-09-27 timed pages: {d['unlocked_0927_timed_pages'] or 'none'}")
    L += ["", "AGAINST THE HAND SCORING (session README)"]
    for c in doc["compare"]:
        L.append(f"  row {c['row']} {'agree' if c['agree'] else 'DIFFERS'}: {c['what']}: scorer {fmt(c['scorer'], 4)} | hand {fmt(c['hand'], 4)}" + (f" | {c['note']}" if c["note"] else ""))
    return L


# ---------------------------------------------------------------------- main

def clean(o, prefix: str):
    if isinstance(o, dict):
        return {str(k): clean(v, prefix) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, prefix) for v in o]
    if isinstance(o, str):
        return o.replace(prefix, "<tmp>")
    return o


def build(repo: Path, py: str, work: Path, out: Path) -> dict:
    for k, v in C.items():
        if not has(repo, v):
            raise Refused(f"{repo} lacks {k} {v[:7]}; pass --repo (or MOE_HISTORY_REPO) with the history")
    doc = {"what": "ledger rows 1 to 9, scored by the procedure registered in the fe73508/f0a831b and bb979d4 "
                   "docstrings, on the 2026-09-27 pages; written 2026-10-05, after the pages",
           "facts": facts(repo), "quotes": quotes(repo)}
    pre = out / "prerun"
    pre.mkdir(parents=True, exist_ok=True)
    p27 = extract(repo, C["pages_0927"], work / "pages0927", S0927)
    trees = {}
    for key in ("p1_1605", "timing", "wsc"):
        trees[key] = extract(repo, C[key], work / C[key][:7], "scripts", "moe", S0925)
    runs = []
    for r in GH200_RUNS:
        hit = sorted((trees["p1_1605"] / S0925).glob(f"results/gaps-*/private_weight_reference/*{r}"))
        runs.append(str(hit[0].relative_to(trees["p1_1605"])))
    T = {}
    for key in ("p1_1605", "timing"):
        sha = C[key][:7]
        T[key] = run_prerun(py, trees[key], "timing", work / f"{sha}.json",
                            pre / f"r3_timing_model.{sha}.txt", p27 / S0927, *runs, "--counters", CNT0925)
    Wd = run_prerun(py, trees["wsc"], "wsc", work / "wsc.json", pre / f"wave_split_bytes.{C['wsc'][:7]}.txt",
                    p27 / CNT0927 / "lock1710", CNT0925)
    T0, T1 = T["p1_1605"], T["timing"]
    a, b = T0["predictions"], T1["predictions"]
    same_rest = all(a[k] == b[k] for k in a if k not in ("P1", "P1_clocks", "eta_inversion", "falsifiers"))
    p1_same = all(a["P1"][k] == b["P1"][k] for k in a["P1"] if "f1605" not in k and k in b["P1"])
    doc["fe73508_vs_f0a831b"] = {"P1_clocks": {"fe73508": b["P1_clocks"], "f0a831b": a["P1_clocks"]},
                                 "fe73508_third_clock_slopes": {e: b["P1"][f"G4/f1600/eta{e}"]["slope_2_6"] for e in (1.0, 0.35)},
                                 "only_P1_third_clock_differs": same_rest and p1_same}
    meas = T0["measured_0927"]
    rows = {1: row1(T0, meas), 2: row2(T0, meas), 3: row3(T0, meas), 4: row4(T0, meas),
            5: row5(T0, meas), 6: row6(T0, p27), 7: row7(Wd, p27), 8: row8(Wd, p27), 9: row9(T0, meas)}
    doc["group_b"] = group_b(repo, T0, Wd)
    doc["compare"] = compare(rows, T0, Wd)
    doc["timed_pages_0927"] = [{k: d[k] for k in ("run", "G", "label", "failed", "clocks", "locked")} for d in meas]
    doc["rows"] = {str(k): v for k, v in rows.items()}
    doc["prerun"] = {"f0a831b": {k: v for k, v in T0["predictions"].items()},
                     "fe73508_P1": T1["predictions"]["P1"], "bb979d4_predicted": Wd["predicted"],
                     "bb979d4_registered_view": Wd["registered_view"]}
    doc["verdicts"] = {str(k): {"scorer": v["verdict"], "hand": HAND[k]["verdict"],
                                "agree": v["verdict"] == HAND[k]["verdict"]} for k, v in rows.items()}
    return clean(doc, str(work))


def group_b(repo: Path, T: dict, W: dict) -> list[dict]:
    P = T["predictions"]
    out = []

    def add(row, what, printed, needles, key, files):
        hits = []
        for nd in needles:
            hits += search(repo, key, files, nd)
        uniq = {(h["commit"], h["file"], h["line"]): h for h in hits}
        for h in uniq.values():
            why = COINCIDENT.get((h["commit"], h["file"], h["line"]))
            if why and any(nd in h["text"] for nd in needles):
                h["coincident"] = why
        real = [h for h in uniq.values() if "coincident" not in h]
        out.append({"row": row, "what": what, "printed": printed, "needles": needles,
                    "literal_hits": list(uniq.values()),
                    "group": "a" if any(h["in_docstring"] for h in real) else "b"})

    for e in (1.0, 0.35):
        for f in (1410, 1500, 1605):
            s = f"{P['P1'][f'G4/f{f}/eta{e}']['slope_2_6']:.4f}"
            for key in ("timing", "p1_1605"):
                add(1, f"P1 G=4 slope at {f} MHz, eta {e} (searched at {C[key][:7]})", s, [s], key, TIMING_FILES)
        inc = P["P1"][f"G2/f1410/eta{e}"]["inc"]
        sp, sc = " ".join(f"{v:.3f}" for v in inc), ", ".join(f"{v:.3f}" for v in inc)
        add(1, f"P1 G=2 increments at 1410, eta {e}", sp, [sp, sc], "timing", TIMING_FILES)
    s = f"{P['P2']['8']['slope_2_6']:.4f}"
    add(2, "P2 model slope 2-6 (the band's centre)", s, [s], "timing", TIMING_FILES)
    ex = list(P["P3"]["odd_n_excess"].values())
    sp, sc = " ".join(f"{v:.3f}" for v in ex), ", ".join(f"{v:.3f}" for v in ex)
    add(3, "P3 odd-n excess at n=3,5,7,9", sp, [sp, sc], "timing", TIMING_FILES)
    s = f"{P['P3']['G2_T'][6]:.4f}"
    add(3, "P3 G=2 n=7 time", s, [s], "timing", TIMING_FILES)
    s = f"{P['P4']['inc_8_9']:.3f}"
    add(4, "P4 G=4 n=8->9 increment", s, [f"{s}, abs", f"{s} against", f"{s} ms"], "timing", TIMING_FILES)
    s = f"{P['P4']['inc_others_median']:.3f}"
    add(4, "P4 median of the other increments", s, [f"against {s} elsewhere"], "timing", TIMING_FILES)
    inc = P["P5"]["G3_inc"]
    sp, sc = " ".join(f"{v:.3f}" for v in inc), ", ".join(f"{v:.3f}" for v in inc)
    add(5, "P5 G=3 increments", sp, [sp, sc], "timing", TIMING_FILES)
    s = f"{P['P5']['rho_star_w1']:.3f}"
    add(5, "P5 rho*_w1", s, [f"{s}, abs", f"rho*_w1 {s}"], "timing", TIMING_FILES)
    pred = {(r["arm"], r["G"], r["n"], r["gemm"]): r for r in W["predicted"]}
    for G in (2, 4, 16):
        for arm in ("shared", "private"):
            for g in ("w1", "w2"):
                for n in (5, 8):
                    s = f"{pred[(arm, G, n, g)]['q_fill']:.3f}"
                    add(8, f"WSC {arm} {g} G={G} n={n}", s, [s], "wsc", WSC_FILES)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=Path(os.environ.get("MOE_HISTORY_REPO", ROOT)))
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory(prefix="session0927-") as td:
            doc = build(a.repo.resolve(), a.python, Path(td), a.out)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "score.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    (a.out / "score.txt").write_text("\n".join(lines(doc)) + "\n")
    print(f"session0927: written under {a.out}")
    for k, v in doc["verdicts"].items():
        print(f"  row {k}: {v['scorer']}  (hand {v['hand']}){'' if v['agree'] else '  DIFFERS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
