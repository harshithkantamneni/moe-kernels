#!/usr/bin/env python
"""The calibration rental C4's reader (docs/registered/2026-10-27-cal4-gh200): the five-model held-out table.

    python scripts/scoring/cal4/read_c4.py --c4 results/published/<date>-nvidia_gh200_480gb-cal4-session \\
        [--out-dir scripts/scoring/cal4]          # writes cal4.read.json (sorted keys) and cal4.read.txt

C4 is a CALIBRATION rental: it certifies nothing, and every C4 page becomes SEEN once read. This reader prices C4's
Mixtral-8x7B-tp2 pages (the fifth CAL model, in blind Phi-3.5's w1 regime: K 4096, x near 1, P 112) with the ROUND-4
MODEL EXACTLY AS FROZEN at the registration (moe-kernels-work 2026-10-27 round3/ROUND4.md's selected variant: the W2
byte law on both GEMMs, the K1 knee fitted in the wall domain at its SEEN constants, the clock form of F1, F1 'par'
from this board's two estimator pages; the code is round3/work r6.py and what it imports, every file pinned by sha256
in the registration and checked here before anything is imported).

THE PRIMARY READING (C4-R1). X = tp2's held-out 128 G 1 mean, in percent, e = T_model / T_meas - 1:
  cells    C4-T's NATIVE cells (docs/registered/2026-10-27-cal4-hist/mixtral-8x7b-tp2-128g1.json, n 2, six labels), the
           page admitted by tcommon.timed_page (G5 provenance, the r7-median-run3 rebuild, the ALL view) and each row
           kept unless tcommon.row_drop drops it against v4's price (read_c2.timed_rows' loop, A2's selections)
  model    r6.cell_ms4(r, K1 at the A2 triad with S0 2.75, p 6.0, F1 par, W2 reads, clock=True): tp2 is in no fit, so
           the SEEN constants ARE its held-out fold (knee4.json K1 SEEN; W2 has no fitted constant)
  F1       one (A, tau) per C4 estimator page (refit4 r4.sag_page_fit: design-v5 sag.fit over sag.pts_of at 1410) for
           OLMoE f128 and Mixtral-8x7B f128 on C4's board, A and tau averaged; tp2 has no estimator page to drop
  UNPRICED a kept cell whose W2 fixed point the guard refuses (r6.reads_of None) is UNPRICED: listed, outside X
  X        r4.summ's mean of e over the kept, priced cells (percent, rounded to 0.01)
THE TABLE: round 4's four LOMO held-out means (combo4.json's selected variant: OLMoE +0.48, Qwen2-57B -1.44,
Mixtral-8x7B +1.25, Mixtral-8x22B +0.24; Mixtral-8x7B on C2's board +1.49 beside) and X. THE PASS ESTIMATE: read_c2.power
(diag-v5 power5's, verbatim) on B2's registered cells less round 4's registered UNPRICED cell (83 cells), by the knee
round's method on five models: A (b_class the mean of the five b = -ln(1 + e), sd their SD), B (no common bias, sd
their RMS), D (Mixtral-8x7B set aside: OLMoE, Qwen2-57B, Mixtral-8x22B and X). THE BAND (registered): CLEAN, HIGH,
HIGH-FAR, LOW, LOW-FAR or NOT READ (`outcome`).

THE SECOND READING (C4-R2), the w1 bytes: on C4-B's NATIVE cells, Lambda = sum(counted - W2) / sum(M4 - W2) over the
cells' w1 dram_bytes_read (0 when the counters are W2's, 1 when they are M4's): Phi-3.5's bytes lever, W2 against M4
at P near 100, held out for the first time. Bands W2-SIDE, BETWEEN, M4-SIDE, BEYOND-W2, BEYOND-M4 or NOT READ
(`bytes_outcome`).

SECONDARY READINGS (DESCRIPTIVE; they explain X, they change no band):
  C4-S1  tp2's x per GEMM on its kept cells (bytes-only over floor-only time with M4's bytes, and with W2's)
  C4-S2  tp2 b128g1: K1 at its SEEN constants on counted bytes per GEMM in the wall domain round 4 fitted it in
         (knee4.py's target and kappa_row) and in the cycles domain beside; counted bytes against M4 and W2 per GEMM
  C4-S3  tp2's own sag on its b128g1 page against C4's par estimator (sag domain, us)
  C4-S5  the board: C4's per-page (A, tau) beside A2's and C2's, the power read-back, the lock probe, G-BAND; tp2's
         compiled CTAs per SM at 128x128 (the tc4 census) beside the 2 per SM W2's W_c 264 assumes; calibrate's triad
Beside X (DESCRIPTIVE): X at A2's par; X with M4's bytes (the bytes lever's share in time); X with C4-B's counted
bytes (oracle); X with K1 at C4's own triad; v5'''s held-out mean on the same cells.
THE FREEZE IS TIED TO THE RENTAL'S COMMIT, as C2's: `read` takes the commit from the ledger's driver start line and
refuses unless that commit holds the working copies of the registration, the page, the plan and every code_pins path,
the code_pins hold here, and every repo module the chain loads from its repo (git-tracked, outside tests/ and any
environment directory: read_c2.repo_modules) is pinned with its registered bytes; and the frozen model reproduces the
registered fingerprint on tp2's six cells to 6 decimals.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE.parent / "cal2") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "cal2"))
import read_c2 as RC  # noqa: E402  (the pure part: power, board, the commit and pin checks; its chain() is never called)

WORK = Path.home() / "code" / "moe-kernels-work"
ROUND4 = WORK / "2026-10-27" / "round3" / "work"
REG_NAME = "2026-10-27-cal4-gh200"
REG_PATH = REPO / "docs" / "registered" / f"{REG_NAME}.json"
HIST = "docs/registered/2026-10-27-cal4-hist"
PAGE = "mixtral-8x7b-tp2-128g1.json"
PLAN = "scripts/plans/rentalC4-cal-2026-10.plan"

#: every work file the frozen chain imports or reads (traced with an audit hook over r6's import, a W2 and an M4 read,
#: a clock-form price, a sag fit, a counter page and a timed page), and the round-4 outputs this reader takes numbers
#: from (knee4.json's K1 constants; bytes7.json's and combo4.json's selections and table; DECLARED4.txt, the declared
#: forms); the registration pins each by sha256
FROZEN_FILES = (
    "2026-10-20/scoreA2/bytes.score.json", "2026-10-20/scoreA2/census.score.json",
    "2026-10-20/scoreA2/floor.score.json", "2026-10-20/scoreA2/rep.score.json",
    "2026-10-21/diag-A2/work/common.py",
    "2026-10-21/design-v5/work/v5price.py", "2026-10-21/design-v5/work/kneefit.py",
    "2026-10-21/design-v5/work/sag.py", "2026-10-21/design-v5/work/b2risk.py",
    "2026-10-21/design-v5/work/f3fit.json", "2026-10-21/design-v5/work/kneefit.json",
    "2026-10-21/design-v5/work/kneefit64.json",
    "2026-10-22/diag-v5/work/d5.py", "2026-10-22/diag-v5/work/bw.py", "2026-10-22/diag-v5/work/knee2.py",
    "2026-10-22/diag-v5/work/v5prime.py", "2026-10-22/diag-v5/work/bytes_lomo.json",
    "2026-10-22/knee-v5/work/k5.py",
    "2026-10-22/refit4/work/r4.py",
    "2026-10-27/round3/work/r5.py", "2026-10-27/round3/work/r6.py", "2026-10-27/round3/work/tailmap.json",
    "2026-10-27/round3/work/knee4.json", "2026-10-27/round3/work/bytes7.json", "2026-10-27/round3/work/combo4.json",
    "2026-10-27/round3/work/DECLARED4.txt",
)

# ------------------------------------------------------------------ frozen constants (registered by value)
MODEL = "mixtral-8x7b-tp2"
O, Q, M, X22 = "olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b", "mixtral-8x22b"
CAL_MODELS = (O, Q, M, X22)
#: round 4's selected variant (W2 + K1 wall knee + clock form + F1 par), LOMO held-out 128 G 1 means in percent
#: (combo4.json tables["SELECTED: W2 + K1 wall knee + clock form"]); Mixtral-8x7B on C2's board beside
CAL_HELD_OUT = {O: 0.48, Q: -1.44, M: 1.25, X22: 0.24}
M_C2_BOARD = 1.49
#: round 4's estimate on its 83 registered cells (combo4.json pass, selected), reproduced at registration
ROUND4_PASS = {"A (4 models)": 0.3798, "B (4 models)": 0.4998, "D (O, Q, X; 8x7B G1 aside)": 0.4233}
SELECTED = "SELECTED: W2 + K1 wall knee + clock form"
LAW = "W2"
#: K1 in the wall domain, SEEN (knee4.json candidates K1 SEEN): (p_128, S0 k-steps)
K1_SEEN = (6.0, 2.75)
TRIAD_A2_GBPS, TRIAD_RANGE_GBPS = 3726.2, (3725.6, 3727.6)
LOCK = 1410
POWER_W = 700
A2_PAGE_FITS = {O: (29.65, 300.0), M: (54.39, 550.0)}
A2_PAR_128 = (42.02, 425.0)
C2_PAGE_FITS = {O: (30.6, 325.0), M: (52.58, 500.0)}
C2_PAR_128 = (41.59, 412.5)
EST_UNITS = {O: "f128", M: "f128"}
T_LABEL, B_LABEL, CENSUS_LABEL = "t128g1", "b128g1", "tc4"
#: round 4's registered UNPRICED blind cell (combo4.json unpriced_b2): its index in B2's registered power cells and
#: its identity in b2risk.b2_cells, excluded from E1-TILE's statistics
UNPRICED_B2 = {"index": 72, "model": "jetmoe-8b", "cls": "128g1", "n": 2, "label": "PT"}
#: C4-R1's band (percent)
BAND = {"clean": 1.0, "far": 2.0}
MIN_CELLS = 4
#: C4-R2's band on Lambda, and the smallest W2 lever (sum(M4 - W2) / sum(M4), w1) it is read at
LAM = {"w2_side": 1 / 3, "m4_side": 2 / 3, "beyond_w2": -1 / 3, "beyond_m4": 4 / 3}
MIN_LEVER = 0.03
CTAS_PER_SM = 2
#: the X grid the registration's pass-estimate bounds are printed at (percent)
BOUNDS_X = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
#: every repo module the frozen chain and this reader import from the repo they run (diag-A2 common.REPO for the
#: chain, this file's repo for the reader), traced over a full stand-in read; the registration pins each
CHAIN_REPO_MODULES = RC.CHAIN_REPO_MODULES + ("scripts/scoring/cal2/read_c2.py", "scripts/scoring/cal4/read_c4.py")
COMMIT_FILES = (f"docs/registered/{REG_NAME}.json", f"docs/registered/{REG_NAME}.txt", f"{HIST}/{PAGE}", PLAN)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dumps(d) -> str:
    return json.dumps(d, indent=1, sort_keys=True, default=str) + "\n"


# ================================================================== the pure part (no frozen chain)

def to_b(e_pct: float) -> float:
    return -math.log(1 + e_pct / 100)


def registered_cells(cells=None) -> list:
    """B2's registered power cells less round 4's registered UNPRICED cell (83 cells)."""
    cells = RC.b2_cells() if cells is None else cells
    i = UNPRICED_B2["index"]
    assert cells[i]["model"] == UNPRICED_B2["model"] and cells[i]["cls"] == UNPRICED_B2["cls"], cells[i]
    return [c for k, c in enumerate(cells) if k != i]


def pass_estimates(X: float, cells=None, base=None, R=200_000) -> dict:
    """The knee round's three estimates on five held-out means (round 4's four and X, percent)."""
    cells = registered_cells() if cells is None else cells
    base = [CAL_HELD_OUT[m] for m in CAL_MODELS] if base is None else list(base)
    b = [to_b(x) for x in base + [X]]
    A = RC.power(cells, [st.fmean(b)] * 3, st.stdev(b), R=R)["E1-TILE"]
    Bp = RC.power(cells, [0.0] * 3, math.sqrt(st.fmean(x * x for x in b)), R=R)["E1-TILE"]
    four = [to_b(x) for x in (base[0], base[1], base[3], X)]
    D = RC.power(cells, [st.fmean(four)] * 3, st.stdev(four), R=R)["E1-TILE"]
    return {"A (5 models)": A, "B (5 models)": Bp, "D (O, Q, 8x22B, X; 8x7B G1 aside)": D}


#: the owner's sequence (2026-10-27): C4, then ONE final declared model round; if that round fails the B3 gate,
#: iteration stops and the owner chooses between narrowing scope and going blind; B3 is registered only after a
#: declared round passes. If C4 reads NOT READ, the final round runs on the four existing CAL models without tp2 and
#: C4 is not rented again. Each band's sentence says what C4-R1 means for that final round; the joint table below
#: (C4-R1 band x C4-R2 band) is the registered reading of the two together.
SEQUENCE = ("C4, then ONE final declared model round (gate: G1 every held-out 128 G 1 mean within 1.0%, Mixtral-8x7B on "
            "C2's board as well; G2 estimated P(A) >= 0.5 on the registered cells). If that round fails the B3 gate, "
            "iteration stops and the owner chooses between narrowing scope and going blind. B3 is registered only after "
            "a declared round passes. If C4 reads NOT READ, the final round runs on the four existing CAL models without "
            "tp2, and C4 is not rented again (owner, 2026-10-27)")
MEANS_FOR_B3 = {
    "CLEAN": ("the frozen round-4 model prices tp2 (Phi-3.5's w1 geometry: K 4096, x near 1, P 112) within the gate's 1% "
              "bar. What that supports depends on C4-R2 (the joint table): with W2-SIDE both Phi levers pass their first "
              "held-out test; with BETWEEN or M4-SIDE the 1% agreement is a cancellation of a bytes error against a knee "
              "error and Phi's stake is not supported; with NOT READ only the time is tested. The final declared round "
              "refits on five models with tp2 SEEN and keeps the round-4 forms that the joint cell supports; G1 covers "
              "the five"),
    "HIGH": ("tp2 runs faster than the frozen model prices it, as Mixtral-8x7B's 128 G 1 does (+1.25 on A2, +1.49 on C2's "
             "board): the Mixtral-type miss partly reproduces at P 112. The final declared round must carry a knee "
             "variable that separates K 4096, x near 1 (Mixtral-8x7B and tp2) from Qwen2-57B at K 3584, with C4-S2's "
             "wall-domain w1 error as its evidence, and the bytes term the joint cell names"),
    "HIGH-FAR": ("as HIGH, past Mixtral-8x7B's own residual: the miss is a regime of K 4096, x near 1, G 1 (or of "
                 "Mixtral-8x7B's E 8 routing, which tp2 shares; disclosed), and Phi-3.5 (K 4096, x 0.94 to 1.07) most "
                 "likely sits in it. The final declared round must model that regime on five models; Phi-3.5's stake is "
                 "reported as at risk until it does"),
    "LOW": ("the frozen model prices tp2 too fast, the direction of Phi-3.5's stake (-2.3% against v5''). The final declared "
            "round must remove the term the joint cell names: W2's w1 service at P near 100 (M4-SIDE, BETWEEN) or the "
            "knee's sharpness at P 112 (W2-SIDE, BEYOND-W2)"),
    "LOW-FAR": ("as LOW, past Qwen2-57B's -1.44: the round-4 model's Phi-3.5 stake is most likely an error large enough to "
                "fail E1-TILE's 1% bar on Phi alone. The final declared round must remove the term the joint cell names, "
                "and Phi-3.5's stake is reported as at risk in that round's registration"),
    "NOT READ": ("no fifth held-out model: the final declared round runs on the four existing CAL models without tp2 (G1 on "
                 "the four, Mixtral-8x7B on C2's board as well), and C4 is not rented again (owner, 2026-10-27)"),
}
#: C4-R1 band x C4-R2 band: one sentence for the final declared round per cell (registered, MAJOR 1 of the review)
_R2 = ("W2-SIDE", "BETWEEN", "M4-SIDE", "BEYOND-W2", "BEYOND-M4", "NOT READ")
_J = {
    "CLEAN": ("both Phi levers pass their first held-out test: the round keeps W2 on both GEMMs and the wall knee",
              "the 1% agreement is partly a cancellation: W2 over-serves w1 at P 112 and the knee offsets it; the round "
              "must reduce W2's w1 service and re-check the knee on five models; Phi's stake is not supported",
              "the 1% agreement is a cancellation: W2's w1 service at P near 100 is wrong and the knee error offsets it; "
              "the round keeps M4 on w1 at this distance and refits the knee; Phi's stake is not supported",
              "time agrees but the counters show more service than W2: the round keeps W2's direction and discloses its "
              "size; Phi's bytes lever is supported in sign only",
              "time agrees but neither byte law describes w1 at P 112: the round has no measured w1 law at Phi's distance "
              "and Phi's stake is not supported",
              "only the time is tested: the round keeps the round-4 forms; Phi's bytes lever stays untested"),
    "HIGH": ("the bytes are right, so the excess is the knee: the round must sharpen the knee at K 4096, x near 1",
             "the knee is sharper than K1's and W2 partly over-serves w1: the round changes both terms",
             "the knee is sharper still, partly hidden by W2's over-service: the round keeps M4 on w1 at P near 100 "
             "and sharpens the knee",
             "the knee carries the excess and W2 under-serves: the round sharpens the knee and discloses the bytes",
             "neither byte law fits and tp2 still runs fast: the round has no measured w1 law at Phi's distance",
             "the knee is the only tested term: the round sharpens the knee at K 4096, x near 1"),
    "HIGH-FAR": ("a knee regime at K 4096, x near 1 with the bytes right: the round must model it; Phi's stake at risk",
                 "a knee regime plus W2 over-service: the round changes both; Phi's stake at risk",
                 "a knee regime hidden in part by W2's error: the round keeps M4 on w1 at P near 100 and models the "
                 "regime; Phi's stake at risk",
                 "a knee regime with W2 under-serving: the round models the regime; Phi's stake at risk",
                 "a knee regime and no byte law at P 112: the round models the regime; Phi's stake at risk",
                 "a knee regime, bytes untested: the round models the regime; Phi's stake at risk"),
    "LOW": ("the bytes are right, so the knee is too sharp at P 112 (Qwen2-like): the round softens the knee there",
            "W2 over-serves w1 at P 112: the round reduces W2's w1 service at P near 100",
            "W2's w1 service at P near 100 is the error (Phi's bytes lever is the law's error): the round keeps M4 on w1 "
            "at this distance",
            "the knee is too sharp and W2 under-serves: the round softens the knee",
            "neither byte law fits: the round has no measured w1 law at Phi's distance",
            "the term is not identified: the round treats the knee and W2's w1 service as suspect"),
    "LOW-FAR": ("the knee is too sharp at P 112: the round softens it; Phi's stake at risk",
                "W2 over-serves w1: the round reduces W2's w1 service; Phi's stake at risk",
                "Phi's bytes lever is the law's error: the round keeps M4 on w1 at P near 100; Phi's stake at risk",
                "the knee is too sharp: the round softens it; Phi's stake at risk",
                "neither byte law fits at P 112: Phi's stake at risk",
                "the term is not identified: the round treats the knee and W2's w1 service as suspect; Phi's stake at risk"),
    "NOT READ": tuple("the time is not read: the final round runs on the four CAL models without tp2" +
                      ("" if b == "NOT READ" else f"; the bytes reading ({b}) is DESCRIPTIVE beside it") for b in _R2),
}
JOINT = {r1: dict(zip(_R2, cells)) for r1, cells in _J.items()}


def joint(r1: str, r2: str) -> str:
    return JOINT[r1][r2]


BYTES_MEANS = {
    "W2-SIDE": ("W2 prices tp2's w1 bytes at P 112: same-wave w1 re-reads about 0.4 of a wave apart are served, as the "
                "law says. Phi-3.5's bytes lever (W2's w1 bytes 9% under M4's, ROUND3.md section 4) passes its first "
                "held-out test"),
    "BETWEEN": ("the counted w1 bytes sit between W2's and M4's: the law serves too many of the re-reads it moves. "
                "Phi-3.5's bytes lever is partly the law's error; the declared round may not keep W2's w1 service as it is"),
    "M4-SIDE": ("the re-reads W2 serves at P 112 miss, as M4 prices them: W2's w1 service at P near 100 is the law's "
                "error, and Phi-3.5's w1 bytes under W2 are suspect. The declared round keeps M4's w1 at this distance "
                "(ROUND3.md section 3's 'W2 on w2 only' row moves the four-model table by 0.2 points or less)"),
    "BEYOND-W2": ("fewer w1 bytes than even W2 prices: more re-reads are served than the launch law places under the "
                  "cut. W2's direction holds at P 112; its size does not"),
    "BEYOND-M4": ("more w1 bytes than M4 prices: neither law describes w1 at P 112; the declared round has no measured w1 "
                  "law at Phi-3.5's distance"),
    "NOT READ": "no held-out test of the w1 bytes lever: Phi-3.5's lever stays untested",
}


def outcome(X: float | None, n_cells: int, valid: bool = True, why: str = "") -> dict:
    """C4-R1's registered band and what it means for the B3 decision."""
    if not valid or X is None or n_cells < MIN_CELLS:
        return {"band": "NOT READ", "why": why or f"{n_cells} kept, priced tp2 cells (at least {MIN_CELLS} needed)",
                "b3": MEANS_FOR_B3["NOT READ"]}
    if abs(X) <= BAND["clean"]:
        band = "CLEAN"
    elif X >= BAND["far"]:
        band = "HIGH-FAR"
    elif X > 0:
        band = "HIGH"
    elif X <= -BAND["far"]:
        band = "LOW-FAR"
    else:
        band = "LOW"
    return {"band": band, "b3": MEANS_FOR_B3[band]}


def bytes_outcome(lam: float | None, lever: float | None, n_cells: int, valid: bool = True, why: str = "") -> dict:
    """C4-R2's registered band on Lambda."""
    if not valid or lam is None or lever is None or n_cells < MIN_CELLS or lever < MIN_LEVER:
        w = why or (f"W2's w1 lever {lever:.3f} under {MIN_LEVER}" if lever is not None and lever < MIN_LEVER
                    else f"{n_cells} NATIVE cells priced under both laws (at least {MIN_CELLS} needed)")
        return {"band": "NOT READ", "why": w, "means": BYTES_MEANS["NOT READ"]}
    if lam < LAM["beyond_w2"]:
        band = "BEYOND-W2"
    elif lam <= LAM["w2_side"]:
        band = "W2-SIDE"
    elif lam < LAM["m4_side"]:
        band = "BETWEEN"
    elif lam <= LAM["beyond_m4"]:
        band = "M4-SIDE"
    else:
        band = "BEYOND-M4"
    return {"band": band, "means": BYTES_MEANS[band]}


def lambda_of(cells: list) -> dict:
    """C4-R2 over [{counted, w2, m4}] (w1 bytes per NATIVE cell): Lambda, the W2 lever and per-cell numbers."""
    cells = [c for c in cells if c.get("w2") is not None and c.get("m4") is not None]
    if not cells:
        return {"n": 0, "lambda": None, "lever": None}
    num = sum(c["counted"] - c["w2"] for c in cells)
    den = sum(c["m4"] - c["w2"] for c in cells)
    out = {"n": len(cells), "lambda": round(num / den, 4) if den else None,
           "lever": round(den / sum(c["m4"] for c in cells), 4),
           "w2_over_counted_pct": round(100 * (sum(c["w2"] for c in cells) / sum(c["counted"] for c in cells) - 1), 2),
           "m4_over_counted_pct": round(100 * (sum(c["m4"] for c in cells) / sum(c["counted"] for c in cells) - 1), 2)}
    return out


def table(X: float | None) -> dict:
    t = {m: {"held_out_128g1_mean_pct": CAL_HELD_OUT[m], "source": "round 4 selected (combo4.json, LOMO)"} for m in CAL_MODELS}
    t[MODEL] = {"held_out_128g1_mean_pct": X, "source": "C4-R1 (this reader)"}
    t["mixtral-8x7b, C2 board (beside)"] = {"held_out_128g1_mean_pct": M_C2_BOARD, "source": "round 4 selected (combo4.json)"}
    v = [x["held_out_128g1_mean_pct"] for k, x in t.items() if k in (*CAL_MODELS, MODEL) and x["held_out_128g1_mean_pct"] is not None]
    t["max |mean| (five models)"] = round(max(abs(x) for x in v), 2)
    t["spread"] = round(max(v) - min(v), 2)
    t["rms of means"] = round(math.sqrt(st.fmean(x * x for x in v)), 2)
    vg = v + [M_C2_BOARD]
    t["round-4 gate G1 (every mean within 1.0, 8x7B on C2 too; DESCRIPTIVE)"] = all(abs(x) <= 1.0 for x in vg)
    return t


def cell_summary(es: list) -> dict:
    e = [100 * x for x in es]
    if not e:
        return {"n": 0}
    return {"n": len(e), "mean_pct": round(st.fmean(e), 2), "rms_pct": round(math.sqrt(st.fmean(x * x for x in e)), 2),
            "worst_pct": round(max(e, key=abs), 2)}


def triad(results: Path) -> dict:
    """C4-S5: calibrate's triad ruler on C4's board beside A2's 3726.2 GB/s (15 rentals 3725.6 to 3727.6)."""
    f = sorted(Path(results).glob("calibration/*ceilingtriad*/measured_nvidia_gh200_480gb.yaml"))
    out = {"A2_gbps": TRIAD_A2_GBPS, "range_15_rentals_gbps": list(TRIAD_RANGE_GBPS)}
    if len(f) != 1:
        return {**out, "why": f"{len(f)} ceilingtriad pages"}
    for line in f[0].read_text().splitlines():
        if line.strip().startswith("achieved_bandwidth_gbps:"):
            v = float(line.split(":")[1])
            return {**out, "C4_gbps": v, "inside_range": TRIAD_RANGE_GBPS[0] <= v <= TRIAD_RANGE_GBPS[1]}
    return {**out, "why": "no achieved_bandwidth_gbps"}


def census_file(results: Path, model: str = MODEL, label: str = CENSUS_LABEL) -> tuple:
    """({gemm: compiled CTAs per SM}, verdict, why) for one model at 128x128x64 w8 s3 from a census unit (tp2, tc4)."""
    f = sorted(Path(results).glob(f"*-tile-census-*-{label}/tile-census.json"))
    if len(f) != 1:
        return {}, None, f"{len(f)} {label} census files"
    p = json.loads(f[0].read_text())
    got = {e["gemm"]: int(e["occupancy"]["ctas_per_sm"]) for e in p.get("entries") or []
           if e.get("model") == model and e.get("tile") == "128x128x64w8s3"}
    return got, p.get("verdict"), ""


def census_ok(compiled: dict, verdict: str | None = "COMPLETE") -> bool:
    """2 CTAs per SM on both GEMMs and a COMPLETE census."""
    return compiled == {"w1": CTAS_PER_SM, "w2": CTAS_PER_SM} and verdict == "COMPLETE"


def gpu_times(results: Path, unit: str, model: str = MODEL) -> dict:
    """{(n, label, gemm): (ncu gpu_time ns, cycles)} on one counter page, NATIVE (round 3 w1boards.gpu_times)."""
    f = RC.page_files(results, model, unit)
    if len(f) != 1:
        return {}
    p = json.loads(f[0].read_text())
    out = {}
    for c in p["cells"]:
        if c["arm"] != "native":
            continue
        lab = c.get("histogram") or "balanced"
        for g in ("w1", "w2"):
            pg = c["per_gemm"][g]
            out[(int(c["n"]), lab, g)] = (float(pg["gpu_time_ns"]),
                                          float(pg.get("sm__cycles_elapsed.avg") or c["recorded"][g]["sm__cycles_elapsed.avg"]))
    return out


# ================================================================== the frozen chain

def frozen_hashes() -> dict:
    return {f: sha256(WORK / f) for f in FROZEN_FILES}


def check_frozen(reg: dict) -> None:
    want = reg["frozen"]["files"]
    got = frozen_hashes()
    bad = [f for f in FROZEN_FILES if want.get(f) != got[f]]
    if bad or set(want) != set(FROZEN_FILES):
        raise SystemExit(f"REFUSED: the frozen chain is not the registered one: {bad or sorted(set(want) ^ set(FROZEN_FILES))}")


_CHAIN: dict = {}
#: the module names the frozen chain imports from the work folders (generic names other repo modules share)
CHAIN_MODULES = ("r6", "r5", "r4", "k5", "d5", "common", "v5price", "kneefit", "sag", "b2risk", "bw", "knee2", "v5prime")


def chain():
    """round 4's r6 (round3/work), imported once, with r5, r4, k5, d5 and b2risk; k5 builds the CAL pricer at p 14 and
    T0 43.23 us and bw 3597.8 GB/s are asserted (the first-build trap, in d5, k5, r4 and r5)."""
    if not _CHAIN:
        keep_path = list(sys.path)
        keep_mods = {m: sys.modules.pop(m) for m in CHAIN_MODULES if m in sys.modules}
        try:
            sys.path.insert(0, str(ROUND4))
            import r6  # noqa: E402
            import b2risk  # noqa: E402  (imported by the chain; b2_cells for the UNPRICED cell's identity)
        finally:
            for m in CHAIN_MODULES:
                sys.modules.pop(m, None)
            sys.modules.update(keep_mods)
            sys.path[:] = keep_path
        r5, r4, k5, d5 = r6.r5, r6.r4, r6.k5, r6.d5
        assert d5.TM.P_KNEE == 14.0
        x = [float(v) for v in d5.RG.pricer().f2.x]
        assert abs(x[0] * 1e3 - 43.23) < 0.01 and abs(x[2] - 3597.8) < 0.1, x
        assert abs(k5.RULERS["triad"] - TRIAD_A2_GBPS) < 0.1 and r5.SIGMA_GAP == 0.0375 and r5.L2_MIB == 60.0
        kn = json.loads((ROUND4 / "knee4.json").read_text())
        assert kn["selection"]["chosen"] == "K1" and tuple(kn["candidates"]["K1"]["SEEN"]) == K1_SEEN, kn["selection"]
        assert json.loads((ROUND4 / "bytes7.json").read_text())["selection"]["chosen"] == LAW
        _CHAIN.update(r6=r6, r5=r5, r4=r4, k5=k5, d5=d5, B=b2risk)
    return _CHAIN


def knee(rate: float | None = None):
    """K1 at its SEEN S0 and the A2 triad (or `rate`)."""
    C = chain()
    return C["k5"].Form("K1", C["k5"].RULERS["triad"] if rate is None else float(rate), K1_SEEN[1])


def reads(r: dict, law: str | None = LAW):
    """{w1, w2} DRAM reads of one cell under the law (None: M4), at p 6.0; None when the fixed-point guard refuses."""
    return chain()["r6"].reads_of(r, K1_SEEN[0], law)


def price(r: dict, sag128: tuple, rd=None, rate: float | None = None) -> float | None:
    """The frozen round-4 model's ms of one 128x128 NATIVE cell (r6.cell_ms4, the clock form)."""
    rd = reads(r) if rd is None else rd
    if rd is None:
        return None
    return chain()["r6"].cell_ms4(r, knee(rate), K1_SEEN[0], tuple(sag128), rd, clock=True)


def v5pp(r: dict, sag128: tuple) -> float:
    """v5'' (null knee, p 5.75, M4, v5'''s own sag) on one cell, beside (DESCRIPTIVE)."""
    C = chain()
    with C["r5"].law3():
        return C["k5"].cell_ms(r, None, 5.75, tuple(sag128), (30.37, 300.0))


def page_doc() -> dict:
    return json.loads((REPO / HIST / PAGE).read_text())


def native_rows(doc: dict) -> list:
    return [{"model": MODEL, "page": doc["page"], "n": c["n"], "label": c["label"], "doc": doc, "c": c}
            for c in doc["cells"] if "native" in c["arms"]]


def fingerprint(sag128: tuple = A2_PAR_128) -> dict:
    """The frozen model's ms on tp2's six NATIVE cells at A2's par, and W2's and M4's reads, rounded."""
    rows = native_rows(page_doc())
    out = {"cells": [f"n{r['n']}/{r['label']}" for r in rows], "ms": [], "w2_reads": [], "m4_reads": []}
    for r in rows:
        w2, m4 = reads(r), reads(r, None)
        out["w2_reads"].append(None if w2 is None else [int(round(float(w2[g]))) for g in ("w1", "w2")])
        out["m4_reads"].append([int(round(float(m4[g]))) for g in ("w1", "w2")])
        p = price(r, sag128, w2) if w2 is not None else None
        out["ms"].append(None if p is None else round(p, 6))
    return out


def sag_fits(results: Path) -> dict:
    """{model: (A us, tau us)} per C4 estimator page (refit4 r4.sag_page_fit) and their average ("par")."""
    r4 = chain()["r4"]
    per = {}
    for m, unit in EST_UNITS.items():
        files = RC.page_files(results, m, unit)
        if len(files) != 1:
            raise LookupError(f"{len(files)} {m} {unit} pages")
        per[m] = r4.sag_page_fit(Path(results), m)
    return {"per_page": per, "par": r4.par(list(per.values()))}


def timed_rows(results: Path, reg_file: dict, doc: dict, model: str = MODEL, label: str = T_LABEL) -> tuple:
    """(rows, why): a timed page's NATIVE cells as read_c2.timed_rows builds them (A2's selections, the ALL view,
    tcommon.row_drop against v4's price); tp2's t128g1 by default."""
    d5 = chain()["d5"]
    TC, RG = d5.TC, d5.RG
    prm = d5.params_for(d5.SEL, model)
    with d5.A2.scoring(d5.RULES):
        rep, why = TC.timed_page(Path(results), model, label, int(reg_file["group_m"]), TC.C3.TimedView("ALL"), reg_file)
        if rep is None:
            return [], why
        if rep.get("lock_mhz") is None or int(rep["lock_mhz"]) != LOCK:
            return [], f"NOT SCORED: the timed page records lock_mhz {rep.get('lock_mhz')!r}, not {LOCK}"
        tt = {(int(r["tiles"]), str(r["histogram"]), str(r["arm"])): r for r in rep["treads_table"]}
        rows = []
        for c in doc["cells"]:
            if "native" not in c["arms"]:
                continue
            k = (int(c["n"]), c["label"], "native")
            tr = tt.get(k)
            base = RG.price_cell(model, "native", c, doc, prm, key=d5.SEL["key"])["ms"]
            row = {"model": model, "page": doc["page"], "cls": "128g1", "n": k[0], "label": k[1], "c": c, "doc": doc, "v4": base}
            if tr is None:
                rows.append({**row, "meas": None, "drop": "no row on the page"})
                continue
            rows.append({**row, "cls": TC.tile_class(int(doc["block_m"]), int(doc["group_m"])), "meas": float(tr["ms_p50"]),
                         "drop": TC.row_drop(tr, base)})
    return rows, ""


def held_out(rows: list, sag128: tuple, *, rd_of=None, rate: float | None = None) -> dict:
    """X over the kept rows: e = price / meas - 1 (r4.summ's mean). rd_of(row) gives the reads (default W2's)."""
    r4 = chain()["r4"]
    keep = [r for r in rows if not r["drop"] and r["meas"]]
    res, unpriced = [], []
    for r in keep:
        rd = reads(r) if rd_of is None else rd_of(r)
        p = price(r, sag128, rd, rate) if rd is not None else None
        if p is None:
            unpriced.append({"n": r["n"], "label": r["label"]})
            continue
        res.append((r, p / r["meas"] - 1))
    out = {"cells": [{"n": r["n"], "label": r["label"], "meas_ms": round(r["meas"], 6), "e_pct": round(100 * e, 3)} for r, e in res],
           "dropped": [{"n": r["n"], "label": r["label"], "why": r["drop"] or "no measurement"} for r in rows if r["drop"] or not r["meas"]],
           "unpriced": unpriced}
    s = r4.summ([e for _, e in res])
    out["X_pct"] = s.get("mean")
    out["summ"] = s
    return out


def v5pp_held_out(rows: list, sag128: tuple) -> float | None:
    keep = [r for r in rows if not r["drop"] and r["meas"]]
    e = [v5pp(r, sag128) / r["meas"] - 1 for r in keep]
    return chain()["r4"].summ(e).get("mean") if e else None


def x_of_rows(rows: list) -> dict:
    """C4-S1: x = bytes-only / floor-only per GEMM (v5'' terms, F1 off) with M4's bytes and with W2's."""
    C = chain()
    k5, d5 = C["k5"], C["d5"]
    xs = {f"{g} {b}": [] for g in ("w1", "w2") for b in ("M4", "W2")}
    for r in rows:
        prm = d5.params_for(d5.SEL, r["model"])
        w2 = reads(r)
        for b, rd in (("M4", None), ("W2", w2)):
            if b == "W2" and rd is None:
                continue
            with d5.A2.scoring(d5.RULES):
                p = d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads=rd, F3=k5.F3_M4, P={128: K1_SEEN[0], 64: 3.75}, sag="none")
            for g in ("w1", "w2"):
                xs[f"{g} {b}"].append(p[g]["bytes_only"] / p[g]["floor_only"])
    return {k: {"min": round(min(v), 3), "median": round(st.median(v), 3), "max": round(max(v), 3)} for k, v in xs.items() if v}


def counter_cells(results: Path, model: str = MODEL, unit: str = B_LABEL) -> list:
    """A byte page's NATIVE 128x128 counter cells (kneefit.cells_of via r4.counter_cells; tp2's b128g1 by default) with
    W2's and M4's reads beside."""
    r4 = chain()["r4"]
    out = []
    for c in r4.counter_cells(Path(results), model, unit):
        w2, m4 = reads(c), reads(c, None)
        out.append({"n": c["c"]["n"], "label": c["c"]["label"], "counted": c["reads"],
                    "w2": None if w2 is None else {g: float(w2[g]) for g in ("w1", "w2")},
                    "m4": {g: float(m4[g]) for g in ("w1", "w2")}})
    return out


def bytes_reading(cells: list) -> dict:
    """C4-R2 (w1) and, beside, the same quantities on w2, from counter_cells' output."""
    out = {}
    for g in ("w1", "w2"):
        rows = [{"counted": c["counted"][g], "w2": None if c["w2"] is None else c["w2"][g], "m4": c["m4"][g]} for c in cells]
        out[g] = lambda_of(rows)
        out[g]["per_cell"] = [{"n": c["n"], "label": c["label"], "counted": int(c["counted"][g]),
                               "w2_over_counted_pct": None if c["w2"] is None else round(100 * (c["w2"][g] / c["counted"][g] - 1), 2),
                               "m4_over_counted_pct": round(100 * (c["m4"][g] / c["counted"][g] - 1), 2)} for c in cells]
    return out


@contextlib.contextmanager
def pages_as(root: Path, unit: str, model: str = MODEL):
    """kneefit.PAGES set to one counter page for k5.capture_rows, restored on exit."""
    k5 = chain()["k5"]
    keep = k5.KF.PAGES
    k5.KF.PAGES = [(Path(root), model, unit, LOCK, "c4")]
    try:
        yield
    finally:
        k5.KF.PAGES = keep


def knee_reading(results: Path, model: str = MODEL, unit: str = B_LABEL) -> dict:
    """C4-S2: K1 at its SEEN constants on tp2's counted bytes, per GEMM, in the wall domain (knee4.py: target t_row -
    R3share(t_row), the row's kappa = 1 + (off - R3share) / (cycles / 1410 MHz) on c and dead) and the cycles domain."""
    C = chain()
    k5, d5 = C["k5"], C["d5"]
    with pages_as(results, unit, model):
        rows = k5.capture_rows(("c4",))
    if not rows:
        return {"rows": 0}
    gt = gpu_times(results, unit, model)
    f = knee()
    p = K1_SEEN[0]
    wall, cyc = {"w1": [], "w2": []}, {"w1": [], "w2": []}
    cell = {}
    for r in rows:
        win, state, sigma, c_ns, bw, sms, occ = r["args"]
        pc = k5.gemm_form(win, *state, sigma, c_ns, sms, p, occ, f) + r["dead"]
        cyc[r["gemm"]].append(pc / r["meas"] - 1)
        k = (r["n"], r["label"], r["gemm"])
        if k not in gt:
            continue
        gns, cy = gt[k]
        t = gns * 1e-6
        r3s = d5.sat(d5.R3_1410[0], d5.R3_1410[1], t)
        kappa = 1 + (t - cy / 1410e3 - r3s) / (cy / 1410e3)
        pw = k5.gemm_form(win, *state, sigma, c_ns * kappa, sms, p, occ, f) + r["dead"] * kappa
        wall[r["gemm"]].append(pw / (t - r3s) - 1)
        a = cell.setdefault((r["n"], r["label"]), [0.0, 0.0])
        a[0] += pw
        a[1] += t - r3s
    out = {"rows": len(rows), "K1": {"p": p, "S0": K1_SEEN[1], "rate_gbps": k5.RULERS["triad"]}}
    for g in ("w1", "w2"):
        out[g] = {"wall": cell_summary(wall[g]), "cycles": cell_summary(cyc[g]),
                  "x_median (counted bytes)": round(st.median(r["x"] for r in rows if r["gemm"] == g), 3)}
    out["cell sum (w1 + w2), wall"] = cell_summary([a / b - 1 for a, b in cell.values()])
    return out


def sag_domain(results: Path, est: tuple, model: str = MODEL, unit: str = B_LABEL) -> dict:
    """C4-S3: C4's par estimator minus the measured off (us) on one counter page (tp2's b128g1; sag.pts_of at 1410)."""
    k5 = chain()["k5"]
    p = k5.S.pts_of(RC.page_files(results, model, unit), float(LOCK))
    if not p:
        return {"n": 0}
    A, tau = est
    e = [A * (1 - math.exp(-t / tau)) - off for t, off in p]
    own = k5.S.fit(p)
    return {"n": len(e), "mean_err_us": round(st.fmean(e), 2), "rms_us": round(math.sqrt(st.fmean(x * x for x in e)), 2),
            "own_fit_A_us": own["A_us"], "own_fit_tau_us": own["tau_us"]}


def census(results: Path) -> dict:
    """C4-S5: tp2's compiled CTAs per SM (tc4) beside the occupancy the frozen pricing uses (A2's selections)."""
    d5 = chain()["d5"]
    prm = d5.params_for(d5.SEL, MODEL)
    t = d5.RG.PAGE_TILE[128]
    got, verdict, why = census_file(results)
    out = {"priced": {g: int(prm.occ(t, g)) for g in ("w1", "w2")}, "W_c_assumed": 132 * CTAS_PER_SM}
    if why:
        return {**out, "why": why, "two_per_sm": False}
    return {**out, "compiled": got, "verdict": verdict, "two_per_sm": census_ok(got, verdict)}


def b2_unpriced_identity() -> dict:
    """The registered UNPRICED cell as b2risk.b2_cells lists it (the row behind B2's power cell at that index)."""
    r = chain()["B"].b2_cells()[UNPRICED_B2["index"]]
    return {"model": r["model"], "cls": r["cls"], "n": int(r["n"]), "label": r["label"]}


def round4_table() -> dict:
    """Round 4's selected variant re-priced by the frozen code (combo4.py's 'SELECTED' loop: W2 reads at each fold's p,
    K1 at the fold's LOMO (p, S0), the clock form, F1 par of the fold's estimator pages on the timed page's board, via
    r4.sag_page_fit on A2's and C2's f128 pages: the sag-fit path the C4 reading itself uses)."""
    C = chain()
    r6, r4, k5 = C["r6"], C["r4"], C["k5"]
    kw = json.loads((ROUND4 / "knee4.json").read_text())["candidates"]["K1"]["LOMO"]
    fits = {b: {m: r4.sag_page_fit(root, m) for m in (O, M)} for b, root in (("A2", r4.A2T), ("C2", r4.C2T))}
    a2 = [dict(r, board="A2") for r in r4.a2_rows() if r["cls"] == "128g1"]
    sets = {O: [r for r in a2 if r["model"] == O], Q: [r for r in a2 if r["model"] == Q],
            M: [r for r in a2 if r["model"] == M], X22: [dict(r, board="C2") for r in r4.c2_8x22b_rows()],
            "mixtral-8x7b C2": [dict(r, board="C2") for r in r4.c2_8x7b_rows()]}
    out = {}
    for k, cells in sets.items():
        hm = M if k == "mixtral-8x7b C2" else k
        p, s0 = kw[hm]
        f = k5.Form("K1", k5.RULERS["triad"], s0)
        sag = r4.par([fits[cells[0]["board"]][m] for m in (O, M) if m != hm])
        e = [r6.cell_ms4(r, f, p, tuple(sag), r6.reads_of(r, p, LAW), clock=True) / r["meas"] - 1 for r in cells]
        out[k] = r4.summ(e)["mean"]
    return out


def round4_pass() -> dict:
    """Round 4's four-model estimate on the 83 registered cells (read_c2.pass_estimates)."""
    return RC.pass_estimates(CAL_HELD_OUT[X22], cells=registered_cells(), base=[CAL_HELD_OUT[m] for m in (O, Q, M)])


def preflight(R: dict) -> dict:
    """Before any page: the frozen model reproduces round 4's selected table and its four-model estimate (refuse on any
    difference), and every repo module loaded so far is pinned (check_chain_repo; it runs again at the end)."""
    t4, want = round4_table(), {**CAL_HELD_OUT, "mixtral-8x7b C2": M_C2_BOARD}
    if t4 != want or t4 != R["frozen"]["round 4 selected table, re-priced by the frozen code at registration"]:
        raise SystemExit(f"REFUSED: the frozen model does not reproduce round 4's table: {t4} against {want}")
    pa = round4_pass()
    if pa != ROUND4_PASS or pa != R["frozen"]["round 4 pass estimate on 83 cells, reproduced"]:
        raise SystemExit(f"REFUSED: round 4's four-model estimate does not reproduce: {pa} against {ROUND4_PASS}")
    bad = check_chain_repo(R)
    if bad:
        raise SystemExit(f"REFUSED: a repo module loaded before any page is not its registered bytes: {bad[:5]}")
    return {"round4_table": t4, "round4_pass": pa, "chain_repo_before_pages": "pinned"}


# ================================================================== the whole reading

def reg() -> dict:
    return json.loads(REG_PATH.read_text())


def commit_paths(R: dict) -> list:
    return sorted(set(COMMIT_FILES) | set(R["code_pins"]))


def chain_repo() -> Path:
    return Path(chain()["d5"].V.REPO)


def check_chain_repo(R: dict) -> list:
    """[(path, why)]: every repo module loaded from the chain's repo and from this reader's repo (read_c2.repo_modules:
    absolute, git-tracked, outside tests/ and any environment directory) is a registered pin with its registered bytes,
    and every CHAIN_REPO_MODULES path in the chain's repo has its registered sha256."""
    bad = []
    for root in sorted({chain_repo().resolve(), REPO.resolve()}):
        loaded = RC.repo_modules(list(sys.modules.values()), root, RC.tracked_files(root))
        bad += [(rel, f"loaded from {root} but not pinned") for rel in loaded if rel not in R["code_pins"]]
        bad += [(rel, f"under {root}: not the registered sha256") for rel in loaded
                if rel in R["code_pins"] and sha256(root / rel) != R["code_pins"][rel]]
    root = chain_repo().resolve()
    bad += [(p, f"under the chain's repo {root}: not the registered sha256") for p in RC.CHAIN_REPO_MODULES
            if p not in R["code_pins"] or not (root / p).is_file() or sha256(root / p) != R["code_pins"][p]]
    return sorted(set(bad))


def read(c4: Path, *, registration: dict | None = None, _skip_commit_check_for_tests: bool = False) -> dict:
    """The C4 reading from a published C4 session directory (results/ and session/ under it). Refuses (SystemExit)
    unless the work files are the registered bytes, the rental's commit holds the working copies of the registration,
    the page, the plan and every code_pins path, the code_pins hold here, every repo module loaded is pinned, and the
    frozen model reproduces the registered fingerprint."""
    R = registration or reg()
    check_frozen(R)
    c4 = Path(c4)
    results, session = c4 / "results", c4 / "session"
    out = {"registration": REG_NAME, "c4": str(c4), "status": "CALIBRATION: certifies nothing; every C4 page is SEEN once read"}
    if _skip_commit_check_for_tests:
        out["commit"] = {"sha": None, "checked": False}
    else:
        sha, why = RC.run_commit(session)
        if sha is None:
            raise SystemExit(f"REFUSED: the rental's commit is not on its ledger: {why}")
        bad = RC.check_commit(REPO, sha, commit_paths(R))
        if bad:
            raise SystemExit(f"REFUSED: the working tree is not the rental's commit {sha}: {bad[:5]}")
        out["commit"] = {"sha": sha, "checked": True, "files": len(commit_paths(R))}
    bad = RC.check_pins(R, REPO)
    if bad:
        raise SystemExit(f"REFUSED: a pinned file is not its registered bytes: {bad[:5]}")
    out["board"] = brd = RC.board(session)
    fp = fingerprint()
    want = R["frozen"]["fingerprint"]
    out["fingerprint"] = {"equals_registered": fp == want}
    if fp != want:
        raise SystemExit("REFUSED: the frozen round-4 model does not reproduce the registered fingerprint: the chain moved")
    if b2_unpriced_identity() != {k: UNPRICED_B2[k] for k in ("model", "cls", "n", "label")}:
        raise SystemExit("REFUSED: B2's cell at the registered UNPRICED index is not round 4's UNPRICED cell")
    out["preflight"] = preflight(R)
    doc = page_doc()
    reg_file = R["pages"]["files"][PAGE]
    cen = census(results)
    out["C4-S5 census"] = cen
    out["C4-S5 triad"] = tri = triad(results)
    try:
        sf = sag_fits(results)
    except (LookupError, AssertionError) as e:
        sf, out["F1"] = None, {"why": str(e)}
    rows, why = [], ""
    if sf is not None:
        out["F1"] = {"per_page": {m: list(v) for m, v in sf["per_page"].items()}, "par": [round(sf["par"][0], 4), sf["par"][1]],
                     "A2 per_page": {m: list(v) for m, v in A2_PAGE_FITS.items()}, "A2 par": list(A2_PAR_128),
                     "C2 per_page": {m: list(v) for m, v in C2_PAGE_FITS.items()}, "C2 par": list(C2_PAR_128)}
        rows, why = timed_rows(results, reg_file, doc)
        r1 = held_out(rows, sf["par"]) if rows else {"X_pct": None, "cells": [], "dropped": [], "unpriced": []}
    else:
        r1 = {"X_pct": None, "cells": [], "dropped": [], "unpriced": []}
    X = r1.get("X_pct")
    valid_why = [w for w in (brd.get("why") if not brd["valid"] else "", "" if sf is not None else "no F1",
                             why, "" if cen.get("two_per_sm") else f"census: {cen.get('why') or (cen.get('compiled'), cen.get('verdict'))}") if w]
    valid = not valid_why
    out["C4-R1"] = {**r1, "page_why": why}
    # the byte page (C4-R2, C4-S2, C4-S3)
    f, w = RC.check_counter_page(results, MODEL, B_LABEL, G=1, reg_file=reg_file)
    cc = counter_cells(results) if f else []
    if rows:   # DESCRIPTIVE beside X
        bes = {"X_at_A2_par_pct": held_out(rows, A2_PAR_128)["X_pct"],
               "X_with_M4_bytes_pct": held_out(rows, sf["par"], rd_of=lambda r: reads(r, None))["X_pct"],
               "v5''_held_out_pct": v5pp_held_out(rows, sf["par"])}
        if cc:
            tw = {(c["n"], c["label"]): c["counted"] for c in cc}
            bes["X_with_counted_bytes_pct (oracle)"] = held_out(rows, sf["par"], rd_of=lambda r: tw.get((r["n"], r["label"])))["X_pct"]
        if tri.get("C4_gbps"):
            bes["X_at_C4_triad_pct"] = held_out(rows, sf["par"], rate=tri["C4_gbps"])["X_pct"]
        out["C4-R1"]["beside (DESCRIPTIVE)"] = bes
        out["C4-S1 x"] = x_of_rows([r for r in rows if not r["drop"]])
    out["table"] = table(X if valid else None)
    out["outcome"] = outcome(X, len(r1["cells"]), valid, "; ".join(valid_why))
    out["pass_estimates"] = pass_estimates(X) if out["outcome"]["band"] != "NOT READ" else None
    br = bytes_reading(cc) if cc else {}
    w1 = br.get("w1") or {}
    bwhy = [x for x in (brd.get("why") if not brd["valid"] else "", w,
                        "" if cen.get("two_per_sm") else f"census: {cen.get('why') or (cen.get('compiled'), cen.get('verdict'))}") if x]
    out["C4-R2 w1 bytes"] = {**br, "page_why": w}
    out["bytes_outcome"] = bytes_outcome(w1.get("lambda"), w1.get("lever"), w1.get("n", 0), not bwhy, "; ".join(bwhy))
    out["C4-S2 tp2 b128g1 knee"] = knee_reading(results) if f else {"why": w}
    out["C4-S3 tp2 sag"] = sag_domain(results, sf["par"]) if (f and sf) else {"why": w or "no F1"}
    out["joint"] = {"C4-R1": out["outcome"]["band"], "C4-R2": out["bytes_outcome"]["band"],
                    "means": joint(out["outcome"]["band"], out["bytes_outcome"]["band"]), "sequence": SEQUENCE}
    bad = check_chain_repo(R)
    if bad:
        raise SystemExit(f"REFUSED: the reading ran a repo module that is not its registered bytes: {bad[:5]}")
    out["chain_repo"] = {"path": str(chain_repo()), "modules_checked": len(RC.CHAIN_REPO_MODULES)}
    return out


def txt(d: dict) -> str:
    L = [f"C4 READING ({d['registration']}): {d['status']}", f"  pages: {d['c4']}", f"  commit: {json.dumps(d['commit'], sort_keys=True)}",
         f"  board: {json.dumps(d['board'], sort_keys=True)}",
         f"  fingerprint (the frozen round-4 model equals the registration): {d['fingerprint']['equals_registered']}",
         f"  F1 (C4 board): {json.dumps(d.get('F1'), sort_keys=True)}", "", "FIVE-MODEL HELD-OUT TABLE (128 G 1 means, %)"]
    for m, v in d["table"].items():
        L.append(f"  {m}: {json.dumps(v, sort_keys=True)}")
    r = d["C4-R1"]
    L += ["", f"C4-R1: X = {r.get('X_pct')} over {len(r['cells'])} cells; dropped {r['dropped']}; unpriced {r['unpriced']}; {r.get('page_why', '')}"]
    if "beside (DESCRIPTIVE)" in r:
        L.append(f"  beside (DESCRIPTIVE): {json.dumps(r['beside (DESCRIPTIVE)'], sort_keys=True)}")
    for c in r["cells"]:
        L.append(f"  n {c['n']} {c['label']}: meas {c['meas_ms']} ms, e {c['e_pct']:+.3f}%")
    L += [f"OUTCOME: {d['outcome']['band']}: {d['outcome']['b3']}", f"PASS ESTIMATES: {d['pass_estimates']}", ""]
    b = d["C4-R2 w1 bytes"]
    L.append(f"C4-R2: Lambda {(b.get('w1') or {}).get('lambda')} lever {(b.get('w1') or {}).get('lever')}; {b.get('page_why', '')}")
    L.append(f"BYTES OUTCOME: {d['bytes_outcome']['band']}: {d['bytes_outcome']['means']}")
    if "joint" in d:
        L.append(f"JOINT ({d['joint']['C4-R1']} x {d['joint']['C4-R2']}): {d['joint']['means']}")
    L.append(f"C4-R2 detail: {json.dumps(b, sort_keys=True)}")
    for k in ("C4-S1 x", "C4-S2 tp2 b128g1 knee", "C4-S3 tp2 sag", "C4-S5 census", "C4-S5 triad", "chain_repo"):
        if k in d:
            L.append(f"{k}: {json.dumps(d[k], sort_keys=True)}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c4", type=Path, required=True, help="the published C4 session directory (results/, session/)")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    a = ap.parse_args(argv)
    d = read(a.c4)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "cal4.read.json").write_text(dumps(d))
    (a.out_dir / "cal4.read.txt").write_text(txt(d))
    print(txt(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
