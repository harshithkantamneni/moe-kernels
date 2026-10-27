#!/usr/bin/env python
"""The calibration rental C3's reader (docs/registered/2026-10-26-cal3-gh200): per-CTA launch and finish stamps.

    python scripts/scoring/cal3/read_c3.py --c3 results/published/<date>-nvidia_gh200_480gb-cal3-session \\
        [--out-dir scripts/scoring/cal3]          # writes cal3.read.json (sorted keys) and cal3.read.txt

C3 is DESCRIPTIVE: it certifies nothing, no registered verdict moves, and every C3 page is SEEN once read. Each reading
below has a registered band that says what it means for the next model round; the round itself is the owner's.

THE PAGES. scripts/instr_probe_c3.py's stamp pages (moe/instrumented's copy of vLLM's kernel at STAMPS 3, "ends": per
CTA its SM, the start and the end in %globaltimer ns and %clock64 cycles, and its kind, live or dead), one per launch
(w1, w2) of each stamped call on A2's E1-CAL cells at 128x128x64 w8 s3 G 1, NATIVE; each beside the expert_ids and
num_tokens_post_padded vLLM handed the launch. The perturb unit's plain-kernel medians on the same cells are the
uninstrumented reference. Geometry at G 1: pid = pid_m x P + pid_n, P = N / 128; the live CTAs are the pids under
(num_tokens_post_padded / 128) x P, and live rank r = pid. W_c = 264 (132 SMs x 2 CTAs per SM: the A2 and C2 census).

THE READINGS (registered; functions named):
  C3-R1 theta_L (C5's constant, refit4 DECLARED.txt AM2: "a CTA at in-wave position u launches theta_L u / W_c
        lifetimes after its wave's first CTA"). Per launch (`launch_reading` via `wave_theta`): each later wave
        k >= 1 (live ranks k W_c .. (k+1) W_c - 1) with at least MIN_WAVE CTAs, least squares start_ns = a + s u / W_c,
        u = r - k W_c; theta_k = s / L, L the launch's median live lifetime (ns); the launch's theta is the CTA-weighted
        mean of its waves'. The model's theta_L is the median over its w2 launches. Band (`theta_band`) on Mixtral-8x7B
        w2: IN [0.25, 0.30], NEAR [0.20, 0.35] outside IN, OUT otherwise.
  C3-R2 same-wave re-reads. A slab re-read is CTA r = (pid_m, pid_n) whose pid_m - 1 holds the same expert (`pairs`):
        it re-reads the B slab (expert, pid_n) its partner r - P read. Its launch gap in units of M4's staggered gap
        is g = (start(r) - start(r - P)) / (P L / W_c) (`gap_ratio`): M4's stream gives g 1, C5's same-wave pair g
        theta_L, C3's synchronized wave g 0. Schedule share: floor(r / W_c) equals floor((r - P) / W_c) (refit4
        REFIT.md's 82 to 88%); measured share: g < GAP_TOGETHER (0.5), the re-reader launched within half of M4's
        gap of its partner, over the re-reads in waves k >= 1 that the schedule puts in their partner's wave (review
        MAJOR-1: wave 0 launches together under every picture; C5 and C3 read 1, M4 0), pooled over the model's w2
        launches whose timer resolves a quarter of P L / W_c (MINOR-4). Band (`samewave_band`) on Mixtral-8x7B w2:
        LAUNCH-TOGETHER >= 0.75, STAGGERED <= 0.25, MIXED between. The all-pairs share is printed beside.
  C3-R3 w1 against priced. Per cell, T_pred = refit4 ROUND2's selected knee K1 at its LOMO constants (N0 beside),
        registered by value (docs/registered/2026-10-26-cal3-priced.json); L_pred = T_pred W_c / live; T_stamp the
        stamped launch's event-pair time, L_med its median live lifetime. Exactly
        ln(T_pred / T_stamp) = ln(L_pred / L_med) + ln(live L_med / (W_c T_stamp)): the LIFE term (the CTAs run
        faster or slower than priced) and the PACKING term (the GEMM is shorter or longer than its waves of median
        lives). LIFE and the total are corrected for the cell's measured overhead o (L and T_stamp over 1 + o; owner,
        2026-10-26). Per model the means over cells; life share = mean LIFE / mean total (`r3_model`). Band
        (`w1_band`) on Mixtral-8x7B w1 under K1, read only when this board's plain kernel reproduces the miss (mean
        T_pred / T_plain - 1 >= +2%) and the w1 overhead |o| (median over cells) is at most a quarter of that miss:
        LIFE >= 2/3, PACKING <= 1/3, BOTH between; NOT REPRODUCED, or DESCRIPTIVE ONLY, otherwise.
  C3-R4 validity (`launch_reading`, `read`): the board (C2's power read-back, the 1410 lock probe, G-BAND:
        read_c2.board), the variant's regcheck line and the gate's HARD line PASS (occupancy, all-off SASS, upstream,
        C3's SASS legs), the stamped copy's C3 SASS legs, the page's label equal to the gate's, and
        per launch: every row stamped, the live CTAs exactly the pids under (ntpp / 128) x P, USE_TD False, 132
        distinct SMs, the globaltimer resolution (rental 4's rule: the least positive difference of the launch's
        distinct start stamps) at most L / 10. A reading on an invalid page is NOT READ.
PERTURBED PAGES (owner OD-1, 2026-10-26). A gate that fails on its timing legs alone (median over 1%, worst over 2%)
keeps the stamp pages, labelled PERTURBED; every reading carries the page label and the read GEMM's overhead (per
cell overhead_median and overhead_worst from the perturb unit; `overhead_summary`). C3-R1 and C3-R2 are ordering
quantities and are read on PERTURBED pages, unless the read GEMM's median overhead exceeds 10% or its worst burst pair
20% (`overhead_ok`): then NOT READ. C3-R3 on PERTURBED pages is DESCRIPTIVE ONLY: never used to price or select a model
form. The hard legs (occupancy, SASS, wgmma path, upstream file, regcheck) still refuse.
DESCRIPTIVE beside them (no band): theta on w1 and on Qwen2-57B, the pair-gap theta (the median g over
schedule-same-wave pairs in waves >= 1, C5's theta_L by its pairs), the median g over all pairs, the per-CTA SM clock (life cycles over life ns),
first-wave against later-wave lifetimes, life spread, the stamped against the plain time (the instrument's cost).
THE FREEZE IS TIED TO THE RENTAL'S COMMIT (read_c2's rule): the commit on the ledger's driver start line must hold
the working copies of the registration, the priced table, the plan, the two histogram files and every code_pins path,
each code_pins path must have its registered sha256, and every repo module this reader loads (read_c2.repo_modules: a
git-tracked file under the repo, outside tests/ and any venv or site-packages) must be pinned.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE.parent / "cal2") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "cal2"))
import read_c2 as RC  # noqa: E402

REG_NAME = "2026-10-26-cal3-gh200"
REG_PATH = REPO / "docs" / "registered" / f"{REG_NAME}.json"
PRICED = "docs/registered/2026-10-26-cal3-priced.json"
PLAN = "scripts/plans/rentalC3-cal-2026-10.plan"
HISTS = {"mixtral-8x7b": "docs/registered/2026-10-14-tileA-hist/mixtral-8x7b-128g1.json",
         "qwen2-57b-a14b": "docs/registered/2026-10-14-tileA-hist/qwen2-57b-a14b-128g1.json"}
#: the plan's stamps units: model -> (label, variant id)
UNITS = {"mixtral-8x7b": ("st7", "u7-st7"), "qwen2-57b-a14b": ("stq", "u8-stq")}
GATE_LABEL, REGCHECK_LABEL = "pt3", "rc3"
COMMIT_FILES = (f"docs/registered/{REG_NAME}.json", f"docs/registered/{REG_NAME}.txt", PRICED, PLAN, *HISTS.values())

# ------------------------------------------------------------------ registered constants (by value)
COLUMNS = ("smid", "start_ns", "start_clk", "prologue_ns", "prologue_clk", "epi_ns", "epi_clk", "end_ns", "end_clk",
           "kind")
HDR = 10
BLOCK_M, BLOCK_N = 128, 128
SMS, CTAS_PER_SM = 132, 2
W_C = SMS * CTAS_PER_SM
MIN_WAVE = W_C // 2
LOCK = 1410
THETA_IN, THETA_NEAR = (0.25, 0.30), (0.20, 0.35)
TOGETHER, STAGGERED = 0.75, 0.25
GAP_TOGETHER = 0.5
LIFE_HI, LIFE_LO = 2 / 3, 1 / 3
REPRO_MIN_PCT = 2.0
TIMER_MAX_FRAC = 1 / 10
#: owner OD-1 (2026-10-26): the overhead above which even the ordering readings (C3-R1, C3-R2) are NOT READ, on the
#: read GEMM's cells: the median over cells of each cell's median overhead (copy / plain - 1), and the worst burst
#: pair of any cell (`overhead_ok`; the justification is the registration's)
OVERHEAD_MEDIAN_MAX, OVERHEAD_WORST_MAX = 0.10, 0.20
#: review MINOR-4: a launch's C3-R2 is admitted only when its globaltimer resolution is at most this fraction of M4's
#: staggered gap P L / W_c
R2_TIMER_FRAC = 1 / 4
#: review MAJOR-3 (owner, 2026-10-26): C3-R3's band is read only when the w1 overhead (absolute, the median over cells
#: of |overhead_median|) is at most this fraction of this board's plain-kernel w1 miss (e_plain)
R3_OVERHEAD_FRAC = 1 / 4
PERTURBED_USE = ("DESCRIPTIVE ONLY: read on PERTURBED pages (the overhead gate failed; owner OD-1), never used to price "
                 "or select a model form")
FORM, FORM_BESIDE = "K1", "N0"
#: refit4 REFIT.md section 2a's schedule figure the same-wave share is set beside
SCHEDULE_QUOTE = (0.82, 0.88)
#: the smallest |ln(T_pred / T_stamp)| a life share is read at (0.5%)
SHARE_MIN_TOTAL = 0.005


def dumps(d) -> str:
    return json.dumps(d, indent=1, sort_keys=True, default=str) + "\n"


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), k)


# ================================================================== per launch (pure)

def decode(rows) -> dict:
    """A [ctas, 10] int64 stamp array (no iteration marks at the ends level) as named columns."""
    a = np.asarray(rows, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != HDR:
        raise ValueError(f"an ends-level stamp array is [ctas, {HDR}]; got {a.shape}")
    return {c: a[:, i] for i, c in enumerate(COLUMNS)}


def timer_resolution(ns) -> int | None:
    """Rental 4's rule: the least positive difference between the launch's distinct globaltimer values."""
    u = np.unique(np.asarray(ns, dtype=np.int64))
    d = np.diff(u)
    d = d[d > 0]
    return int(d.min()) if d.size else None


def wave_theta(start_ns, L: float, w_c: int = W_C, min_wave: int = MIN_WAVE) -> dict:
    """C3-R1 on one launch. start_ns[r] is live rank r's start; per later wave k >= 1 with at least min_wave CTAs,
    least squares start = a + s u / w_c (u = r - k w_c), theta_k = s / L; theta = the CTA-weighted mean."""
    s = np.asarray(start_ns, dtype=np.float64)
    waves = []
    for k in range(1, int(math.ceil(len(s) / w_c))):
        seg = s[k * w_c:(k + 1) * w_c]
        if len(seg) < min_wave:
            continue
        u = np.arange(len(seg), dtype=np.float64) / w_c
        A = np.vstack([np.ones_like(u), u]).T
        (a, slope), *_ = np.linalg.lstsq(A, seg, rcond=None)
        waves.append({"k": k, "ctas": int(len(seg)), "slope_ns": float(slope), "theta": float(slope / L)})
    th = (sum(w["theta"] * w["ctas"] for w in waves) / sum(w["ctas"] for w in waves)) if waves else None
    return {"waves": waves, "theta": th}


def pairs(experts_live, P: int) -> np.ndarray:
    """C3-R2's slab re-reads: the live ranks r = pid_m P + pid_n whose M-tile pid_m - 1 holds the same expert."""
    e = np.asarray(experts_live).reshape(-1)
    m = np.nonzero(e[1:] == e[:-1])[0] + 1
    if not m.size:
        return np.zeros(0, dtype=np.int64)
    return (m[:, None] * P + np.arange(P)[None, :]).reshape(-1).astype(np.int64)


def same_wave(r, P: int, w_c: int = W_C) -> np.ndarray:
    r = np.asarray(r, dtype=np.int64)
    return (r // w_c) == ((r - P) // w_c)


def gap_ratio(start_ns, r, P: int, L: float, w_c: int = W_C) -> np.ndarray:
    """Each re-read's launch gap behind its partner in units of M4's staggered gap P L / w_c."""
    r = np.asarray(r, dtype=np.int64)
    s = np.asarray(start_ns, dtype=np.float64)
    return (s[r] - s[r - P]) / (P * L / w_c)


def launch_reading(rows, experts, ntpp: int, N: int, *, use_td: bool = False, launch_ms: float | None = None,
                   w_c: int = W_C) -> dict:
    """Every C3 number of one stamped launch, and its validity (C3-R4's per-launch legs)."""
    c = decode(rows)
    why = []
    P = -(-int(N) // BLOCK_N)
    kind = c["kind"]
    if (kind == -1).any() or not np.isin(kind, (1, 2)).all():
        why.append(f"{int((~np.isin(kind, (1, 2))).sum())} rows not stamped live or dead")
    live_tiles = int(ntpp) // BLOCK_M
    live = live_tiles * P
    pid = np.arange(len(kind))
    if live > len(kind) or not np.array_equal(kind == 1, pid < live):
        why.append(f"the live CTAs are not the pids under (ntpp {ntpp} / {BLOCK_M}) x P {P} = {live}")
    if use_td:
        why.append("USE_TD True")
    out = {"P": P, "live": live, "dead": int((kind == 2).sum()), "grid": int(len(kind)), "launch_ms": launch_ms}
    if live < 2 or live > len(kind):
        out.update(valid=False, why="; ".join(why) or "no live CTA")
        return out
    s_ns, e_ns = c["start_ns"][:live].astype(np.float64), c["end_ns"][:live].astype(np.float64)
    s_ck, e_ck = c["start_clk"][:live].astype(np.float64), c["end_clk"][:live].astype(np.float64)
    life = e_ns - s_ns
    lclk = e_ck - s_ck
    L = float(np.median(life))
    sms = int(np.unique(c["smid"][:live]).size)
    if sms != SMS:
        why.append(f"{sms} distinct SMs, not {SMS}")
    res = timer_resolution(c["start_ns"][:live])
    if res is None or L <= 0 or res > TIMER_MAX_FRAC * L:
        why.append(f"globaltimer resolution {res} ns over L / 10 ({L / 10:.0f} ns)")
    f = lclk / np.where(life > 0, life, np.nan) * 1e3
    th = wave_theta(s_ns, L, w_c)
    r = pairs(np.asarray(experts).reshape(-1)[:live_tiles], P)
    sw = same_wave(r, P, w_c) if r.size else np.zeros(0, bool)
    g = gap_ratio(s_ns, r, P, L, w_c) if r.size else np.zeros(0)
    tog = g < GAP_TOGETHER
    later = (r // w_c) >= 1 if r.size else np.zeros(0, bool)
    swl = sw & later
    gaps = g[swl]
    # review MINOR-4: R2's cut is half of M4's gap P L / W_c; the timer must resolve a quarter of it
    r2_ok = res is not None and L > 0 and res <= R2_TIMER_FRAC * P * L / w_c
    w0 = life[: min(w_c, live)]
    wl = life[w_c:]
    out.update({
        "waves": live / w_c, "L_ns": L, "L_clk": float(np.median(lclk)),
        "life_iqr_rel": float((np.percentile(life, 75) - np.percentile(life, 25)) / L) if L else None,
        "f_mhz_median": float(np.nanmedian(f)), "span_ns": float(e_ns.max() - s_ns.min()),
        "timer_resolution_ns": res, "sms": sms, "theta": th["theta"], "theta_waves": th["waves"],
        "pairs": int(r.size), "same_wave_sched": int(sw.sum()), "together_meas": int(tog.sum()),
        "together_meas_later": int((tog & later).sum()), "pairs_later": int(later.sum()),
        "r2_admitted": bool(r2_ok), "pairs_sw_later": int(swl.sum()) if r2_ok else 0,
        "together_sw_later": int((tog & swl).sum()) if r2_ok else 0,
        "event_over_span": (launch_ms * 1e6 / float(e_ns.max() - s_ns.min())) if launch_ms else None,
        "gap_ratio_median": float(np.median(g)) if g.size else None,
        "theta_pair": float(np.median(gaps)) if gaps.size else None,
        "life_wave0_over_later": float(np.median(w0) / np.median(wl)) if wl.size and w0.size else None})
    out["valid"] = not why
    out["why"] = "; ".join(why)
    return out


# ================================================================== bands (pure)

def theta_band(x: float | None) -> str:
    if x is None:
        return "NOT READ"
    if THETA_IN[0] <= x <= THETA_IN[1]:
        return "IN"
    if THETA_NEAR[0] <= x <= THETA_NEAR[1]:
        return "NEAR"
    return "OUT"


def samewave_band(f: float | None) -> str:
    if f is None:
        return "NOT READ"
    if f >= TOGETHER:
        return "LAUNCH-TOGETHER"
    if f <= STAGGERED:
        return "STAGGERED"
    return "MIXED"


def w1_band(e_plain_pct: float | None, share: float | None, overhead_abs: float | None = 0.0) -> str:
    """C3-R3's band. Review MAJOR-3 (owner): read only when the w1 overhead (absolute) is at most R3_OVERHEAD_FRAC
    of the plain-kernel miss; otherwise DESCRIPTIVE ONLY."""
    if e_plain_pct is None:
        return "NOT READ"
    if e_plain_pct < REPRO_MIN_PCT:
        return "NOT REPRODUCED"
    if overhead_abs is None:
        return "NOT READ"
    if 100 * overhead_abs > R3_OVERHEAD_FRAC * e_plain_pct:
        return "DESCRIPTIVE ONLY"
    if share is None:
        return "NOT READ"
    if share >= LIFE_HI:
        return "LIFE"
    if share <= LIFE_LO:
        return "PACKING"
    return "BOTH"


MEANING = {
    "theta": {
        "IN": ("C5 MEASURED: the launch spread C5 fitted (0.25 to 0.30) is the measured one. A next byte-law round may "
               "carry C5's distance with theta_L at the measured value as a MEASURED constant (MEASURED-PERTURBED, and "
               "usable only as such, when read on PERTURBED pages); it is IN-SAMPLE for Mixtral-8x7B in any later "
               "held-out round (read off that model's own w2); its w1 guard still applies (C5 failed it through M4's "
               "soft slab survival, which this does not touch)"),
        "NEAR": ("C5's mechanism holds and its constant moves: the measured value replaces the SEEN fit in any next "
                 "round (MEASURED; MEASURED-PERTURBED on PERTURBED pages, usable only as such; in-sample for "
                 "Mixtral-8x7B), and C5's held-out w2 means are re-priced at it before anything is selected"),
        "OUT": ("the jitter picture is not the launch: C5's 0.25 to 0.30 absorbs something else. C5 stays SEEN-FITTED and "
                "cannot be argued from launch physics; a next byte law starts from the measured launch times"),
        "NOT READ": "no valid Mixtral-8x7B w2 launch: C5 stays SEEN-FITTED"},
    "samewave": {
        "LAUNCH-TOGETHER": ("most re-reads launch within half of M4's staggered gap of their partner, as C3 and C5 "
                            "assume and the schedule's 82 to 88% says: M4's staggered distance is the wrong picture for "
                            "Mixtral w2 at 128x128, and a next byte law prices the launch gap, not the rank distance"),
        "STAGGERED": ("most re-reads launch about one staggered gap behind their partner, M4's picture: M4's w2 "
                      "over-prediction is then its survival law's, not its distance's"),
        "MIXED": "neither picture: a next byte law prices the measured share directly",
        "NOT READ": "no valid Mixtral-8x7B w2 launch"},
    "w1": {
        "LIFE": ("Mixtral-8x7B's w1 CTAs themselves run faster than priced: the next knee form needs a per-CTA rate term "
                 "that Mixtral w1 carries and Qwen2 w1 does not (cycles domain), not a wave or launch term"),
        "PACKING": ("the CTAs live as priced and the GEMM is shorter than its waves of median lives: the next round "
                    "declares a time-domain launch-overlap or tail-packing term (C5's jitter physics applied to time)"),
        "BOTH": "both terms carry the miss: a next form needs a rate term and a packing term, declared separately",
        "NOT REPRODUCED": ("this board's plain kernel does not reproduce the w1 miss (under +2%): the decomposition is "
                           "printed, nothing is attributed"),
        "DESCRIPTIVE ONLY": ("the w1 overhead (absolute) is over a quarter of this board's plain-kernel miss: the "
                             "overhead-corrected split is printed and attributes nothing (owner, 2026-10-26)"),
        "NOT READ": "no valid Mixtral-8x7B w1 launch"},
}


# ================================================================== aggregation (pure)

def _med(xs):
    xs = [x for x in xs if x is not None]
    return float(st.median(xs)) if xs else None


def cell_reading(launches: list[dict]) -> dict:
    """The medians over a cell's stamped calls of one GEMM's per-launch numbers (valid launches only)."""
    ok = [x for x in launches if x.get("valid")]
    keys = ("L_ns", "L_clk", "launch_ms", "theta", "theta_pair", "gap_ratio_median", "f_mhz_median", "span_ns",
            "life_iqr_rel", "life_wave0_over_later", "waves", "event_over_span")
    out = {k: _med([x.get(k) for x in ok]) for k in keys}
    out.update(calls=len(launches), valid_calls=len(ok), live=ok[0]["live"] if ok else None,
               why=sorted({x["why"] for x in launches if not x.get("valid")}))
    for k in ("pairs", "same_wave_sched", "together_meas", "together_meas_later", "pairs_later", "pairs_sw_later",
              "together_sw_later"):
        out[k] = int(sum(x.get(k, 0) for x in ok))
    return out


def r3_cell(cell: dict, priced: dict, plain_ms: float | None, form: str = FORM, o: float | None = 0.0) -> dict:
    """C3-R3 on one (cell, GEMM): the LIFE and PACKING terms of ln(T_pred / T_stamp), and e_plain. Review MAJOR-3
    (owner, 2026-10-26): the terms are corrected for the cell's measured overhead o (the perturb unit's
    overhead_median): L / (1 + o) and T_stamp / (1 + o), so LIFE and the total each gain ln(1 + o) and PACKING is
    unchanged; the uncorrected terms are kept beside (`*_raw`). o None: unread, nothing corrected, the band unread."""
    Tp = priced[f"pred_ms_{form}"]
    out = {"T_pred_ms": Tp, "T_plain_ms": plain_ms, "T_stamp_ms": cell.get("launch_ms"), "live_page": cell.get("live"),
           "live_priced": priced["live"]}
    if cell.get("live") != priced["live"] or not cell.get("L_ns") or not cell.get("launch_ms"):
        out["why"] = "no valid launch, or the page's live CTAs are not the priced cell's"
        return out
    live, L, Ts = priced["live"], cell["L_ns"], cell["launch_ms"] * 1e6
    Lp = Tp * 1e6 * W_C / live
    life = math.log(Lp / L)
    pack = math.log(live * L / (W_C * Ts))
    total = math.log(Tp * 1e6 / Ts)
    c = math.log(1 + o) if o is not None else 0.0
    out.update(L_pred_ns=Lp, L_med_ns=L, life_term_raw=life, total_raw=total, overhead=o,
               life_term=life + c, pack_term=pack, total=total + c,
               identity_residual=(total + c) - (life + c) - pack, e_stamp_pct=100 * (Tp * 1e6 / Ts - 1),
               e_plain_pct=None if not plain_ms else 100 * (Tp / plain_ms - 1),
               stamp_over_plain_pct=None if not plain_ms else 100 * (cell["launch_ms"] / plain_ms - 1))
    return out


def r3_model(rows: list[dict]) -> dict:
    ok = [r for r in rows if "total" in r]
    if not ok:
        return {"cells": 0, "e_plain_mean_pct": None, "life_share": None}
    life, pack, tot = (st.fmean(r[k] for r in ok) for k in ("life_term", "pack_term", "total"))
    ep = [r["e_plain_pct"] for r in ok if r["e_plain_pct"] is not None]
    ov = [abs(r["overhead"]) for r in ok if r.get("overhead") is not None]
    return {"cells": len(ok), "life_term_mean": life, "pack_term_mean": pack, "total_mean": tot,
            "overhead_abs_median": float(st.median(ov)) if len(ov) == len(ok) else None,
            "life_share_raw": (st.fmean(r["life_term_raw"] for r in ok) / st.fmean(r["total_raw"] for r in ok)
                               if abs(st.fmean(r["total_raw"] for r in ok)) >= SHARE_MIN_TOTAL else None),
            "life_share": (life / tot) if abs(tot) >= SHARE_MIN_TOTAL else None,
            "e_plain_mean_pct": st.fmean(ep) if ep else None, "e_stamp_mean_pct": st.fmean(r["e_stamp_pct"] for r in ok),
            "stamp_over_plain_mean_pct": st.fmean(r["stamp_over_plain_pct"] for r in ok
                                                  if r["stamp_over_plain_pct"] is not None) if ep else None}


# ================================================================== the files

def env_line(path: Path | None, prefix: str, vid: str) -> str:
    """The verdict word of `<prefix>_<vid with non-alphanumerics as _>=WORD ...` in an env file, or ''."""
    if path is None or not Path(path).is_file():
        return ""
    key = f"{prefix}_" + "".join(ch if ch.isalnum() else "_" for ch in vid)
    for line in Path(path).read_text().splitlines():
        k, _, v = line.partition("=")
        if k == key:
            return v.split()[0] if v.split() else ""
    return ""


def unit_dir(results: Path, what: str) -> Path | None:
    d = sorted(Path(results).glob(f"*-{what}"))
    return d[-1] if d else None


def priced_rows(R: dict | None = None) -> dict:
    doc = json.loads((REPO / PRICED).read_text())
    return {(r["model"], r["label"], int(r["n"]), r["gemm"]): r for r in doc["rows"]}


def overhead_of(perturb: dict, vid: str) -> dict:
    """(label, n, gemm) -> the perturb unit's overhead_median and overhead_worst on that NATIVE cell."""
    v = (perturb.get("variants") or {}).get(vid) or {}
    return {(c["label"], int(c["n"]), c["gemm"]): {"median": c.get("overhead_median"), "worst": c.get("overhead_worst")}
            for c in v.get("cells", []) if c.get("arm") == "native"}


def overhead_summary(cells: list) -> dict:
    """A GEMM's overhead over its cells: the median of the cells' |median| and the largest |worst pair| (review MINOR-1:
    a copy faster than the plain kernel is as much a compile effect as a slower one)."""
    med = [abs(c["median"]) for c in cells if c and c.get("median") is not None]   # review MINOR-1: two-sided
    wst = [abs(c["worst"]) for c in cells if c and c.get("worst") is not None]
    return {"median": float(st.median(med)) if med else None, "worst": max(wst) if wst else None, "cells": len(med)}


def overhead_ok(s: dict) -> tuple[bool, str]:
    """OD-1's NOT-READ bound for the ordering readings: median <= OVERHEAD_MEDIAN_MAX and worst <= OVERHEAD_WORST_MAX
    (an unread overhead is NOT READ)."""
    if s.get("median") is None or s.get("worst") is None:
        return False, "the overhead is unread"
    if s["median"] > OVERHEAD_MEDIAN_MAX or s["worst"] > OVERHEAD_WORST_MAX:
        return False, (f"overhead median {100 * s['median']:.2f}%, worst {100 * s['worst']:.2f}% over the bound "
                       f"({100 * OVERHEAD_MEDIAN_MAX:.0f}%, {100 * OVERHEAD_WORST_MAX:.0f}%)")
    return True, ""


def plain_ms_of(perturb: dict, vid: str) -> dict:
    v = (perturb.get("variants") or {}).get(vid) or {}
    return {(c["label"], int(c["n"]), c["gemm"]): c["plain_ms"] for c in v.get("cells", []) if c.get("arm") == "native"}


def stamp_page(d: Path) -> tuple[dict, list]:
    """(manifest, [(launch index entry, launch_reading)]) of one stamps unit directory."""
    man = json.loads((d / "stamps.json").read_text())
    out = []
    for e in man.get("launches", []):
        if "error" in e:
            out.append((e, {"valid": False, "why": e["error"]}))
            continue
        rows = np.load(d / "stamps" / e["file"])
        ex = np.load(d / "stamps" / e["experts_file"])
        lr = launch_reading(rows, ex, e["num_tokens_post_padded"], e["N"], use_td=e.get("use_td", False),
                            launch_ms=e.get("launch_ms"))
        if e.get("late_buffers"):   # review MINOR-7: a buffer made inside the call dirties the L2 (rental 5 item 4)
            lr.update(valid=False, why="; ".join([w for w in [lr.get("why")] if w] + [f"late_buffers {e['late_buffers']}"]))
        out.append((e, lr))
    return man, out


def model_reading(model: str, page: list, priced: dict, plain: dict, over: dict | None = None) -> dict:
    """Per GEMM: the cells, the theta and same-wave pools, and C3-R3 under K1 and N0."""
    out = {}
    for gemm in ("w1", "w2"):
        cells = {}
        for e, lr in page:
            if e.get("gemm") == gemm:
                cells.setdefault((e["label"], int(e["n"])), []).append(lr)
        rows, r3 = {}, {FORM: [], FORM_BESIDE: []}
        th = []
        pool = {"pairs": 0, "same_wave_sched": 0, "together_meas": 0, "together_meas_later": 0, "pairs_later": 0,
                "pairs_sw_later": 0, "together_sw_later": 0}
        for (label, n), lrs in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            c = cell_reading(lrs)
            pr = priced.get((model, label, n, gemm))
            pm = plain.get((label, n, gemm))
            c["overhead"] = (over or {}).get((label, n, gemm))
            o = (c["overhead"] or {}).get("median")
            c["r3"] = {f: r3_cell(c, pr, pm, f, o) for f in (FORM, FORM_BESIDE)} if pr else {"why": "no priced row"}
            for f in (FORM, FORM_BESIDE):
                if pr:
                    r3[f].append(c["r3"][f])
            th += [x["theta"] for x in lrs if x.get("valid") and x.get("theta") is not None]
            for k in pool:
                pool[k] += c[k]
            rows[f"{label} n{n}"] = c
        out[gemm] = {"cells": rows, "theta_L": _med(th), "theta_launches": len(th),
                     "same_wave_sched_share": pool["same_wave_sched"] / pool["pairs"] if pool["pairs"] else None,
                     "together_meas_share": pool["together_meas"] / pool["pairs"] if pool["pairs"] else None,
                     "together_meas_share_later": (pool["together_meas_later"] / pool["pairs_later"]
                                                   if pool["pairs_later"] else None),
                     "together_sw_later_share": (pool["together_sw_later"] / pool["pairs_sw_later"]
                                                 if pool["pairs_sw_later"] else None),
                     "pairs_sw_later": pool["pairs_sw_later"],
                     "pairs": pool["pairs"], "r3": {f: r3_model(v) for f, v in r3.items()},
                     "overhead": overhead_summary([c.get("overhead") for c in rows.values()])}
    return out


# ================================================================== the whole reading

def reg() -> dict:
    return json.loads(REG_PATH.read_text())


def commit_paths(R: dict) -> list:
    return sorted(set(COMMIT_FILES) | set(R["code_pins"]))


def check_repo_modules(R: dict, root: Path = REPO, modules=None) -> list:
    """[(path, why)]: every repo module this process has loaded (read_c2.repo_modules: absolute __file__, under the
    repo, outside tests/ and any venv or site-packages, git-tracked) is a code_pins path with its registered bytes."""
    mods = list(sys.modules.values()) if modules is None else list(modules)
    loaded = RC.repo_modules(mods, root, RC.tracked_files(root))
    bad = [(p, "loaded by the reader but not pinned") for p in loaded if p not in R["code_pins"]]
    bad += [(p, "not its registered sha256") for p in loaded
            if p in R["code_pins"] and RC.sha256(Path(root) / p) != R["code_pins"][p]]
    return sorted(set(bad))


def read(c3: Path, *, registration: dict | None = None, _skip_commit_check_for_tests: bool = False) -> dict:
    R = registration or reg()
    c3 = Path(c3)
    results, session = c3 / "results", c3 / "session"
    out = {"registration": REG_NAME, "c3": str(c3),
           "status": "CALIBRATION, DESCRIPTIVE: certifies nothing; every C3 page is SEEN once read"}
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
    gd, rd = unit_dir(results, f"perturb-{GATE_LABEL}"), unit_dir(results, f"regcheck-{REGCHECK_LABEL}")
    perturb = json.loads((gd / "perturb.json").read_text()) if gd and (gd / "perturb.json").is_file() else {}
    priced = priced_rows(R)
    models, valid_all = {}, {}
    for model, (label, vid) in UNITS.items():
        g = env_line(gd / "gate.env" if gd else None, "GATE", vid)
        h = env_line(gd / "gate.env" if gd else None, "HARD", vid)
        rc = env_line(rd / "regcheck.env" if rd else None, "REGCHECK", vid)
        pv = (perturb.get("variants") or {}).get(vid) or {}
        m = {"variant": vid, "gate": g or None, "hard": h or None, "regcheck": rc or None,
             "page_label": "CLEAN" if g == "PASS" else ("PERTURBED" if g else None),
             "gate_median_dev": pv.get("median_dev"), "gate_worst_dev": pv.get("worst_dev"),
             "sass_checks_c3": pv.get("sass_checks_c3"), "gate_why": pv.get("why")}
        d = unit_dir(results, f"instr-{model}-{label}")
        why = []
        if not brd.get("valid"):
            why.append(f"board: {brd.get('why')}")
        if h != "PASS" or not g:   # owner OD-1: the hard legs refuse; a timing FAIL alone reads PERTURBED
            why.append(f"gate hard legs {h or 'absent'}")
        if rc != "PASS":
            why.append(f"regcheck {rc or 'absent'}")
        if d is None or not (d / "stamps.json").is_file():
            why.append("no stamp page")
            page = []
        else:
            man, page = stamp_page(d)
            if man.get("page_label") != m["page_label"]:
                why.append(f"the page reads {man.get('page_label')}, the gate {m['page_label']}")
            m["page"] = {"dir": d.name, "launches": len(page), "tile": man.get("tile"), "spec": man.get("spec_text"),
                         "histogram_sha256": man.get("histogram_sha256"), "sass": man.get("sass")}
            bad_l = [lr["why"] for _e, lr in page if not lr.get("valid")]
            if bad_l:
                why.append(f"{len(bad_l)} invalid launches: {sorted(set(bad_l))[:3]}")
            if man.get("histogram_sha256") != R["pages"][model]["sha256"]:
                why.append("the page's histogram is not the registered file")
            if any(not (s.get("checks_c3") or {}).get("ok") for s in man.get("sass") or [{}]):
                why.append("a stamped config fails C3's SASS legs")
        m["valid"] = not why
        m["why"] = "; ".join(why)
        m["gemms"] = model_reading(model, page, priced, plain_ms_of(perturb, vid), overhead_of(perturb, vid)) if page else {}
        models[model] = m
        valid_all[model] = m["valid"]
    out["models"] = models
    mx = models["mixtral-8x7b"]
    w2 = (mx.get("gemms") or {}).get("w2") or {}
    w1 = (mx.get("gemms") or {}).get("w1") or {}
    label = mx["page_label"]
    ow2 = w2.get("overhead") or {}
    ord_ok, ord_why = overhead_ok(ow2) if mx["valid"] else (False, mx["why"])
    th = w2.get("theta_L") if mx["valid"] and ord_ok else None
    sw = w2.get("together_sw_later_share") if mx["valid"] and ord_ok else None
    r3m = ((w1.get("r3") or {}).get(FORM) or {}) if mx["valid"] else {}
    disclose = {"page_label": label, "overhead_w2": ow2, "overhead_bound": [OVERHEAD_MEDIAN_MAX, OVERHEAD_WORST_MAX],
                "not_read_why": ord_why}
    tb, tmean = theta_band(th), None
    th_hi = None
    if th is not None and label == "PERTURBED" and ow2.get("median") is not None:
        # review MINOR-2: at overhead o theta reads low by at most 1 / (1 + o); the band stands only when theta and
        # theta (1 + o) fall in one band
        th_hi = th * (1 + ow2["median"])
        b1, b2 = theta_band(th), theta_band(th_hi)
        if b1 != b2:
            tb = f"{b1} or {b2} (PERTURBED)"
            tmean = (f"not resolved at the measured overhead: theta {th:.3f} and theta (1 + o) {th_hi:.3f} fall in {b1} "
                     f"and {b2}; neither band's meaning applies")
    out["C3-R1"] = {"theta_L_mixtral_w2": th, "theta_times_1_plus_o": th_hi, "band": tb,
                    "meaning": tmean or MEANING["theta"][tb],
                    "constant_label": (None if th is None else ("MEASURED-PERTURBED" if label == "PERTURBED" else "MEASURED")),
                    "in_sample_for": "mixtral-8x7b",
                    "fitted_folds": list(THETA_IN), "near": list(THETA_NEAR), **disclose}
    out["C3-R2"] = {"together_sw_later_share_mixtral_w2": sw, "pairs_banded": w2.get("pairs_sw_later"),
                    "together_meas_share_all_pairs_mixtral_w2": w2.get("together_meas_share"),
                    "gap_together_below": GAP_TOGETHER, "same_wave_sched_share_mixtral_w2": w2.get("same_wave_sched_share"),
                    "schedule_quote": list(SCHEDULE_QUOTE), "band": samewave_band(sw),
                    "meaning": MEANING["samewave"][samewave_band(sw)], **disclose}
    b = w1_band(r3m.get("e_plain_mean_pct"), r3m.get("life_share"), r3m.get("overhead_abs_median"))
    out["C3-R3"] = {"form": FORM, "mixtral_w1": r3m, "band": b, "meaning": MEANING["w1"][b],
                    "page_label": label, "overhead_w1": w1.get("overhead"),
                    "use": PERTURBED_USE if label == "PERTURBED" else "registered reading (CLEAN pages)",
                    "qwen2_w1": (((models["qwen2-57b-a14b"].get("gemms") or {}).get("w1") or {}).get("r3") or {}).get(FORM)}
    out["C3-R4"] = {"valid": valid_all, "why": {k: v["why"] for k, v in models.items()}}
    out["next_round_gate"] = R["next_round_gate"]
    bad = check_repo_modules(R)
    if bad:
        raise SystemExit(f"REFUSED: the reader loaded a repo module that is not pinned with its registered bytes: {bad[:5]}")
    return out


def txt(d: dict) -> str:
    L = [f"C3 READING ({d['registration']}): {d['status']}", f"  pages: {d['c3']}",
         f"  commit: {json.dumps(d['commit'], sort_keys=True)}", f"  board: {json.dumps(d['board'], sort_keys=True)}", ""]
    for k in ("C3-R1", "C3-R2", "C3-R3"):
        r = d[k]
        L.append(f"{k}: {r['band']} [{r.get('page_label')} pages]: {r['meaning']}"
                 + (f" ({r['use']})" if k == "C3-R3" else ""))
        L.append("  " + json.dumps({a: b for a, b in r.items() if a not in ("band", "meaning")}, sort_keys=True, default=str))
    L += [f"C3-R4: {json.dumps(d['C3-R4'], sort_keys=True)}", ""]
    for model, m in d["models"].items():
        L.append(f"{model}: variant {m['variant']}, {m['page_label']} pages, gate {m['gate']}, hard legs {m['hard']} "
                 f"(median {m['gate_median_dev']}, worst "
                 f"{m['gate_worst_dev']}), regcheck {m['regcheck']}, valid {m['valid']} {m['why']}")
        for gemm, g in (m.get("gemms") or {}).items():
            L.append(f"  {gemm}: overhead {json.dumps(g.get('overhead'), sort_keys=True)}")
            L.append(f"  {gemm}: theta_L {_r(g['theta_L'])} over {g['theta_launches']} launches; re-read pairs {g['pairs']}, "
                     f"schedule same-wave {_r(g['same_wave_sched_share'])}, banded together (waves >= 1, schedule "
                     f"same-wave) {_r(g['together_sw_later_share'])} over {g['pairs_sw_later']}, all pairs "
                     f"{_r(g['together_meas_share'])} "
                     f"(waves >= 1: {_r(g['together_meas_share_later'])})")
            for f, x in g["r3"].items():
                L.append(f"    R3 {f}: {json.dumps({k: _r(v) if isinstance(v, float) else v for k, v in x.items()}, sort_keys=True)}")
            for cname, c in g["cells"].items():
                L.append(f"    {cname}: L {_r(c['L_ns'], 1)} ns, theta {_r(c['theta'])}, pair theta {_r(c['theta_pair'])}, "
                         f"median gap {_r(c['gap_ratio_median'])}, "
                         f"f {_r(c['f_mhz_median'], 1)} MHz, waves {_r(c['waves'], 2)}, wave0/later {_r(c['life_wave0_over_later'])}, "
                         f"event / stamp span {_r(c['event_over_span'])}, calls {c['valid_calls']}/{c['calls']}, overhead {json.dumps(c.get('overhead'))}")
    L += ["", f"NEXT ROUND GATE: {d['next_round_gate']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c3", type=Path, required=True, help="the published C3 session directory (results/, session/)")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    a = ap.parse_args(argv)
    d = read(a.c3)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "cal3.read.json").write_text(dumps(d))
    (a.out_dir / "cal3.read.txt").write_text(txt(d))
    print(txt(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
