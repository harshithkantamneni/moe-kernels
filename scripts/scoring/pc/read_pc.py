#!/usr/bin/env python
"""Rental PC's reader (docs/registered/2026-11-06-pc-gh200): per-CTA DRAM rate against loads in flight.

    python scripts/scoring/pc/read_pc.py --pc <published PC session dir> [--out-dir DIR]     # pc.read.{json,txt}
    python scripts/scoring/pc/read_pc.py --c3-standin <published C3 session dir> [--out-dir DIR]   # standin.read.*

PC is a MEASUREMENT rental: every reading is DESCRIPTIVE. It registers no model form and no constant, and moves no
registered verdict. The one label (owner decision 4) is H1's SUPPORTED / FALSIFIED / MIXED (pccommon.h1_label), printed
beside the numbers, DESCRIPTIVE. A reading that does not clear its rule reads NOT IDENTIFIED with the reason, never a
fallback, never a zero.

THE PAGES. The stamp pages of scripts/instr_probe_pc.py (blocks A, C, D, R and Phi's E2) and of scripts/instr_probe_c3.py
(Phi's E1): per launch the int64 rows of the instrumented copy at the "ends" level (per CTA its SM, its start and end in
%globaltimer ns and %clock64 cycles, its kind), the expert_ids and num_tokens_post_padded vLLM handed the launch, the
event-pair ms; the perturb units' plain medians and per-(cell, GEMM) overheads (the uninstrumented reference); the
regcheck lines; each compiled config's PTX loads in flight.

PER LAUNCH (`launch`): the live CTAs (kind 1) are exactly the pids under (ntpp / BLOCK_M) x ceil(N / BLOCK_N) and equal
the cell's sum over experts of ceil(c_e / BLOCK_M) x ceil(N / BLOCK_N); every row stamped live or dead; USE_TD False;
no late buffer. ROW SANITY: every live start and end positive and finite, end after start, clock64 life positive, the
per-CTA clock in (0, 1980] MHz, every per-CTA rate and the launch's X at most 4,022 GB/s (the HBM3 spec peak), and (on a
flushed cell) no CTA faster than Q_B / 0.35 us (an L2-served CTA voids its launch; PC-R9's flush-off control aside).
Any failure: the launch is NOT READ, never zero. Per live CTA c: life lambda_c = (end - start) / (1 + o), o the cell's
GEMM overhead (the clock64 path k_c / f_l / (1 + o) where the globaltimer resolution exceeds a tenth of the median life,
f_l the median clock of the CTAs living at least 10 resolutions); f_c = k_c / (end - start); b_c = K x the tile's BLOCK_N
columns x 2 (the B slab); r_c = b_c / lambda_c; n_c and m_c the time averages over the CTA's life of the launch's live
CTAs GPU-wide and on its SM; plateau CTAs n_c >= 0.9 x the launch's peak live count; X_l the B bytes streamed in the
steady window (the live count at least 0.9 x its peak) over its length, times (1 + o).

PER CELL AND GEMM (`cell_gemm`): pooled over the stamped calls; r = the median r_c over plateau CTAs (NOT READ under 8);
the per-call medians' spread at most 3% (5% at G <= 8) else NOT READ; a cell whose GEMM overhead median exceeds 10% or
worst pair 20% (C3's OD-1 bound, owner decision 5) is NOT READ; PERTURBED pages are read under that rule and labelled.

THE READINGS (registered): PC-R0 validity, R1 the rate curve, R2 the F ladders (c0, r_inf), R3 beta_Q (against the PTX
in-flight ratio), R4 beta_f, R5 omega, R6 sigma, R7 the saturation transition (X_sat, r0, G_90, d(G), L_gpu), R8 transfer
to real shapes (and whether the curve carries the CAL2 cell), R9 the L2 control, R10 Phi's w1 re-read pairs (E1 at
128x128 and E2 at 64x128, against B3's counted bytes), R11 Phi 64g1 life against packing (a printed extra), R12 counter
against time (w2's ncu times confounded: L2 flushed before each kernel, a lower clock), R13 the clock at kernel start,
R14 placement; H1's label; the replicate.

THE FREEZE IS TIED TO THE RENTAL'S COMMIT (read_c2's rule): the commit on the ledger's driver start line holds the
registration, the pages, the shapes, the priced file, the plan and every code_pins path; each code_pins path has its
registered sha256; every repo module this reader loads is pinned. PYTHONHASHSEED=0.
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
for _p in (str(HERE), str(HERE.parent / "cal3"), str(HERE.parent / "cal2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pccommon as P  # noqa: E402
import read_c3 as RC3  # noqa: E402

RC = RC3.RC
REG_NAME = P.NAME
REG_PATH = REPO / "docs" / "registered" / f"{REG_NAME}.json"
NI = "NOT IDENTIFIED"
NR = "NOT READ"

#: the plan's units by label: (result directory suffix, variant id, the pc-locks uid, the page or histogram, the lock,
#: the perturb unit, the regcheck unit). A frozen literal (the reader never scans for a unit it was not registered with)
STAMP_UNITS = {
    "sa": ("instr-pc-sa", P.VARIANTS["sa"], "mixtral-8x7b-sa", f"{P.PAGES_DIR}/pc-a.json", P.DECODE_LOCK, "ptp", "rcp"),
    "sc": ("instr-pc-sc", P.VARIANTS["sc"], "mixtral-8x7b-sc", f"{P.PAGES_DIR}/pc-c.json", P.SESSION_LOCK, "ptc", "rcp"),
    "sd": ("instr-pc-sd", P.VARIANTS["sd"], "mixtral-8x7b-sd", f"{P.PAGES_DIR}/pc-d.json", P.DECODE_LOCK, "ptp", "rcp"),
    "sr": ("instr-pc-sr", P.VARIANTS["sr"], "mixtral-8x7b-sr", f"{P.PAGES_DIR}/pc-r.json", P.DECODE_LOCK, "ptp", "rcp"),
    "se2": ("instr-phi-3.5-moe-se2", P.VARIANTS["se2"], "phi-3.5-moe-se2", P.PHI_64, P.SESSION_LOCK, "ptc", "rcp"),
    "se1": ("instr-phi-3.5-moe-se1", P.VARIANTS["se1"], None, P.PHI_128, P.SESSION_LOCK, "pt3", "rc3"),
}
OTHER_UNITS = {"ptp": "perturb-ptp", "ptc": "perturb-ptc", "pt3": "perturb-pt3", "rcp": "regcheck-rcp",
               "rc3": "regcheck-rc3", "census": "tile-census-mixtral-8x7b-pc", "cnt": "mixtral-8x7b-cnt-r3-counters",
               "b128g1": "phi-3.5-moe-b128g1-r3-counters", "b64g1": "phi-3.5-moe-b64g1-r3-counters",
               "b128g16": "phi-3.5-moe-b128g16-r3-counters"}
PAGE_OF = {"sa": "pc-a", "sc": "pc-c", "sd": "pc-d", "sr": "pc-r"}
#: the probe=pc perturb units' pc-locks uids and locks
PERTURB_UIDS = {"ptp": ("mixtral-8x7b-ptp", P.DECODE_LOCK), "ptc": ("mixtral-8x7b-ptc", P.SESSION_LOCK)}
COMMIT_FILES = (f"docs/registered/{REG_NAME}.json", f"docs/registered/{REG_NAME}.txt", P.SHAPES_FILE, P.PRICED, P.PLAN,
                *(f"{P.PAGES_DIR}/{n}" for n in P.page_files()), P.PHI_128, P.PHI_64, P.PHI_128G16)
DS_READINGS = "scripts/scoring/decode/ds.readings.json"


def dumps(d) -> str:
    return json.dumps(d, indent=1, sort_keys=True, default=_default) + "\n"


def _default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not math.isfinite(float(x)) else float(x)
    return str(x)


def _r(x, k=4):
    return None if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)) else round(float(x), k)


def _fin(*xs) -> bool:
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in xs)


def _med(xs):
    xs = [float(x) for x in xs if _fin(x)]
    return float(st.median(xs)) if xs else None


# ================================================================== time-averaged counts (pure)

def step_integral(s, e, w=None):
    """The step function sum_c w_c 1[s_c <= t < e_c] as sorted breakpoints t, its value after each breakpoint and
    its integral up to each breakpoint; F(x) evaluates the integral (`integral_at`)."""
    s, e = np.asarray(s, np.float64), np.asarray(e, np.float64)
    w = np.ones_like(s) if w is None else np.asarray(w, np.float64)
    t = np.concatenate([s, e])
    d = np.concatenate([w, -w])
    o = np.argsort(t, kind="stable")
    t, d = t[o], d[o]
    val = np.cumsum(d)
    cum = np.concatenate([[0.0], np.cumsum(val[:-1] * np.diff(t))])
    return t, val, cum


def integral_at(t, val, cum, x):
    x = np.asarray(x, np.float64)
    j = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)
    return cum[j] + val[j] * (x - t[j])


def concurrency(s, e) -> tuple[np.ndarray, float]:
    """(n_c, peak): each CTA's time-averaged count of live CTAs (itself included) over [s_c, e_c], and the peak count."""
    s, e = np.asarray(s, np.float64), np.asarray(e, np.float64)
    t, val, cum = step_integral(s, e)
    life = e - s
    n = (integral_at(t, val, cum, e) - integral_at(t, val, cum, s)) / np.where(life > 0, life, np.nan)
    return n, float(val.max()) if val.size else 0.0


def sharing(s, e, sm) -> np.ndarray:
    """m_c: each CTA's time-averaged count of live CTAs on its own SM (itself included)."""
    s, e, sm = np.asarray(s, np.float64), np.asarray(e, np.float64), np.asarray(sm)
    m = np.full(len(s), np.nan)
    for x in np.unique(sm):
        k = sm == x
        m[k] = concurrency(s[k], e[k])[0]
    return m


def steady_x(s, e, b, frac: float = P.PLATEAU_FRAC) -> tuple[float | None, float]:
    """(X, |W|): the B bytes streamed in the steady window W (the live count at least frac x its peak; each CTA streams
    its b_c uniformly over its life) over |W| (bytes per ns = GB/s)."""
    s, e, b = (np.asarray(x, np.float64) for x in (s, e, b))
    t, val, _ = step_integral(s, e)
    if not val.size:
        return None, 0.0
    on = val >= frac * val.max()
    seg = np.diff(t)
    width = float(np.sum(seg[on[:-1]]))
    if width <= 0:
        return None, 0.0
    ind = np.where(on[:-1], 1.0, 0.0)
    cumw = np.concatenate([[0.0], np.cumsum(ind * seg)])
    tv = np.concatenate([ind, [0.0]])

    def W(x):
        return integral_at(t, tv, cumw, x)
    life = e - s
    inw = (W(e) - W(s)) / np.where(life > 0, life, np.nan)
    return float(np.nansum(b * inw) / width), width


def rank_corr(a, b) -> float | None:
    """Spearman's rank correlation (average ranks for ties)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3:
        return None

    def rk(x):
        o = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[o] = np.arange(len(x), dtype=np.float64)
        for v in np.unique(x):
            k = x == v
            if k.sum() > 1:
                r[k] = r[k].mean()
        return r
    ra, rb = rk(a), rk(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ================================================================== one launch (pure)

def launch(rows, experts, ntpp: int, N: int, K: int, tile, *, o, expected_live: int | None = None,
           launch_ms=None, flush: bool = True, use_td: bool = False, late: int = 0, q_guard: int | None = None,
           slab_once: bool = True, timer_ref: dict | None = None) -> dict:
    """Every PC number of one stamped launch and its validity (PC-R0's per-launch legs and ROW SANITY). slab_once: every
    B slab is read by one CTA (PC's decode cells), so X is a DRAM rate and takes the 4,022 GB/s ceiling; a launch with
    co-launched re-reads (Phi's and C3's tread cells) has X a demand rate, L2-served in part, and no ceiling on it."""
    c = RC3.decode(rows)
    why = []
    bm, bn = int(tile[0]), int(tile[1])
    Pn = -(-int(N) // bn)
    kind = c["kind"]
    if (~np.isin(kind, (1, 2))).any():
        why.append(f"{int((~np.isin(kind, (1, 2))).sum())} rows not stamped live or dead")
    live_tiles = int(ntpp) // bm
    live = live_tiles * Pn
    pid = np.arange(len(kind))
    if live > len(kind) or not np.array_equal(kind == 1, pid < live):
        why.append(f"the live CTAs are not the pids under (ntpp {ntpp} / {bm}) x {Pn} = {live}")
    if expected_live is not None and live != int(expected_live):
        why.append(f"{live} live CTAs, the cell's geometry gives {expected_live}")
    if use_td:
        why.append("USE_TD True")
    if late:
        why.append(f"late_buffers {late}")
    if not _fin(o):
        why.append("the cell's overhead is unread")
    out = {"live": live, "dead": int((kind == 2).sum()), "grid": int(len(kind)), "launch_ms": launch_ms, "P": Pn}
    if launch_ms is not None and not (_fin(launch_ms) and launch_ms > 0):
        why.append(f"launch_ms {launch_ms!r} is not finite and positive")
    if live < 1 or live > len(kind) or not _fin(o):
        out.update(valid=False, why="; ".join(why) or "no live CTA")
        return out
    s = c["start_ns"][:live].astype(np.float64)
    e = c["end_ns"][:live].astype(np.float64)
    k = (c["end_clk"][:live] - c["start_clk"][:live]).astype(np.float64)
    sm = c["smid"][:live]
    life = e - s
    if (s <= 0).any() or (e <= 0).any():
        why.append("a live CTA's start or end stamp is not positive")
    if (life <= 0).any():
        why.append(f"{int((life <= 0).sum())} live CTAs end at or before they start")
    if (k <= 0).any():
        why.append(f"{int((k <= 0).sum())} live CTAs with a clock64 life not positive")
    if why:
        out.update(valid=False, why="; ".join(why))
        return out
    f = k / life * 1e3
    if (f > P.F_CEILING_MHZ).any() or not np.isfinite(f).all():
        why.append(f"per-CTA clock up to {float(np.nanmax(f)):.0f} MHz, over the {P.F_CEILING_MHZ:g} MHz ceiling")
    L = float(np.median(life))
    own_res = RC3.timer_resolution(s)
    ref = timer_ref or {}
    res, res_from = own_res, "own starts"
    if res is None and ref.get("res"):   # review MAJOR-6: a launch of a few CTAs in one tick takes its call's w1 timer
        res, res_from = ref["res"], "the call's w1 launch"
    path = "globaltimer"
    lam = life.copy()
    f_long = None
    if res is not None:
        lg = life >= P.CLOCK_PATH_MIN_RES * res
        f_long = float(np.median(f[lg])) if lg.any() else None
    if res is None or res > P.TIMER_MAX_FRAC * L:
        if f_long is not None:
            lam, path = k / (f_long / 1e3), f"clock64 at {f_long:.1f} MHz (own long CTAs)"
        elif _fin(ref.get("f_long")):
            lam, path = k / (ref["f_long"] / 1e3), f"clock64 at {ref['f_long']:.1f} MHz (the call's w1 launch)"
        else:
            why.append(f"globaltimer resolution {res} ns over L / 10 and no CTA lives 10 resolutions here or in the "
                       "call's w1 launch (no clock64 path)")
    lam = lam / (1.0 + float(o))
    bcols = np.minimum(bn, int(N) - (pid[:live] % Pn) * bn)
    b = int(K) * bcols * 2.0
    r = b / lam
    if (r > P.RATE_CEILING_GBPS).any():
        why.append(f"a per-CTA rate {float(r.max()):.1f} GB/s over the {P.RATE_CEILING_GBPS:g} GB/s ceiling")
    if flush and q_guard and (r > q_guard / P.L2_LATENCY_NS).any():
        why.append(f"{int((r > q_guard / P.L2_LATENCY_NS).sum())} CTAs faster than Q_B / {P.L2_LATENCY_NS:g} ns "
                   f"({q_guard / P.L2_LATENCY_NS:.1f} GB/s): L2-served")
    n, peak = concurrency(s, e)
    m = sharing(s, e, sm)
    X, width = steady_x(s, e, b)
    X = None if X is None else X * (1.0 + float(o))
    if flush and slab_once and X is not None and X > P.RATE_CEILING_GBPS:
        why.append(f"X {X:.1f} GB/s over the {P.RATE_CEILING_GBPS:g} GB/s ceiling")
    plateau = n >= P.PLATEAU_FRAC * peak
    t0 = float(s.min())
    early = (s - t0) <= P.START_WINDOW_NS
    r13 = None
    if early.sum() >= P.START_MIN_CTAS and (~early).sum() >= P.START_MIN_CTAS:
        r13 = float(np.median(f[early]) / np.median(f[~early]))
    order = np.argsort(s, kind="stable")
    ex = np.asarray(experts).reshape(-1)
    cta_expert = ex[pid[:live] // Pn] if ex.size >= live_tiles else None   # trap 13: each CTA's expert, expert_ids[pid // Pn]
    out.update(f_long=f_long, res_from=res_from, expert=cta_expert)
    out.update(valid=not why, why="; ".join(why), s=s, e=e, sm=sm, f=f, lam=lam, b=b, r=r, n=n, m=m,
               plateau=plateau, peak=peak, X=X, window_ns=width, L_ns=L, timer_res_ns=res, path=path,
               span_ns=float(e.max() - s.min()), t_first=t0, t_last=float(e.max()), r13=r13,
               sms=int(np.unique(sm).size), placement_rank=rank_corr(np.arange(live), np.argsort(order)))
    return out


# ================================================================== a cell's GEMM over its calls (pure)

def cell_gemm(lrs: list, G_live: int, over: dict | None) -> dict:
    """One (cell, GEMM) pooled over its stamped calls: r over plateau CTAs, the per-call spread rule, the overhead rule."""
    ov = over or {}
    om, ow = ov.get("median"), ov.get("worst")
    out = {"launches": len(lrs), "valid_launches": sum(1 for x in lrs if x.get("valid")), "G": G_live,
           "overhead": {"median": om, "worst": ow}}
    why = []
    if not _fin(om, ow):
        why.append("the overhead is unread")
    elif abs(om) > P.OVERHEAD_MEDIAN_MAX or abs(ow) > P.OVERHEAD_WORST_MAX:
        why.append(f"overhead median {100 * om:.2f}%, worst {100 * ow:.2f}% over C3's NOT-READ bound "
                   f"({100 * P.OVERHEAD_MEDIAN_MAX:.0f}%, {100 * P.OVERHEAD_WORST_MAX:.0f}%)")
    bad = [x.get("why") for x in lrs if not x.get("valid")]
    if bad:
        why.append(f"{len(bad)} invalid launches: {sorted(set(bad))[:2]}")
    valid = [x for x in lrs if x.get("valid")]
    if not valid:
        out.update(status=f"{NR}: " + ("; ".join(why) or "no launch"))
        return out
    rs = np.concatenate([x["r"][x["plateau"]] for x in valid])
    per_call = [float(np.median(x["r"][x["plateau"]])) for x in valid if x["plateau"].any()]
    out.update(r=float(np.median(rs)) if rs.size else None, plateau_ctas=int(rs.size),
               r_per_call=per_call,
               lam_ns=float(np.median(np.concatenate([x["lam"][x["plateau"]] for x in valid]))),
               b=float(np.median(np.concatenate([x["b"] for x in valid]))),
               n=float(np.median(np.concatenate([x["n"][x["plateau"]] for x in valid]))),
               m=float(np.median(np.concatenate([x["m"][x["plateau"]] for x in valid]))),
               f_mhz=float(np.median(np.concatenate([x["f"][x["plateau"]] for x in valid]))),
               X=_med([x["X"] for x in valid]), peak=_med([x["peak"] for x in valid]),
               L_ns=_med([x["L_ns"] for x in valid]), launch_ms=_med([x["launch_ms"] for x in valid]),
               span_ns=_med([x["span_ns"] for x in valid]), paths=sorted({x["path"] for x in valid}),
               r13=_med([x["r13"] for x in valid]), sms=_med([x["sms"] for x in valid]))
    out.update(per_expert(valid))
    out.update(by_sharing(valid))
    if rs.size < P.MIN_PLATEAU_CTAS:
        why.append(f"{rs.size} plateau CTAs, under {P.MIN_PLATEAU_CTAS}")
    if len(per_call) >= 2:
        spread = (max(per_call) - min(per_call)) / st.median(per_call)
        out["spread"] = spread
        lim = P.SPREAD_MAX_SMALL if G_live <= P.SPREAD_SMALL_G else P.SPREAD_MAX
        if not spread <= lim:
            why.append(f"per-call spread {100 * spread:.2f}% over {100 * lim:.0f}%")
    else:
        why.append("fewer than 2 calls with plateau CTAs")
    out["status"] = "READ" if not why else f"{NR}: " + "; ".join(why)
    return out


def per_expert(valid: list) -> dict:
    """Trap 13 (DRAM addresses; CAL2's M3 FOLLOWS-ADDRESS, the G3-D label): the median r_c over plateau CTAs per expert
    (each CTA's expert read off the expert_ids vLLM handed the launch, expert_ids[pid // ceil(N / BLOCK_N)]), and a flag
    when their spread (max - min over the median of the expert medians) exceeds 5%. DESCRIPTIVE: printed, never a
    refusal."""
    if any(x.get("expert") is None for x in valid):
        return {"per_expert_r": None, "expert_spread": None, "expert_flag": "UNREAD: no expert_ids for a launch"}
    r = np.concatenate([x["r"][x["plateau"]] for x in valid])
    e = np.concatenate([x["expert"][x["plateau"]] for x in valid])
    med = {int(k): float(np.median(r[e == k])) for k in np.unique(e)}
    if len(med) < 2:
        return {"per_expert_r": med, "expert_spread": None, "expert_flag": "one expert"}
    v = list(med.values())
    spread = (max(v) - min(v)) / st.median(v)
    return {"per_expert_r": med, "expert_spread": spread,
            "expert_flag": "SPREAD OVER 5%" if spread > P.EXPERT_SPREAD_FLAG else "within 5%"}


def by_sharing(valid: list) -> dict:
    """PC-R1's split by SM sharing: the median r_c over every live CTA alone on its SM (m_c <= 1.2) and sharing it
    (m_c >= 1.8), with their counts. DESCRIPTIVE."""
    r = np.concatenate([x["r"] for x in valid])
    m = np.concatenate([x["m"] for x in valid])
    al, sh = r[m <= P.ALONE_M], r[m >= P.SHARED_M]
    return {"r_alone": float(np.median(al)) if al.size else None, "alone_ctas": int(al.size),
            "r_shared": float(np.median(sh)) if sh.size else None, "shared_ctas": int(sh.size)}


def r_of(cg: dict | None):
    """A cell GEMM's rate when READ, else None."""
    return cg.get("r") if cg and cg.get("status") == "READ" else None


# ================================================================== the readings (pure, on cell readings)

def ols(x, y) -> dict | None:
    """y = a + s x by ordinary least squares: a, s, the standard error of s, n."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    n = len(x)
    if n < 2 or np.ptp(x) == 0:
        return None
    A = np.vstack([np.ones(n), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    se = None
    if n > 2:
        s2 = float(res @ res) / (n - 2)
        cov = s2 * np.linalg.inv(A.T @ A)
        se = float(math.sqrt(cov[1, 1]))
    return {"a": float(coef[0]), "s": float(coef[1]), "se_s": se, "n": n}


def f_ladder(rungs: list) -> dict:
    """PC-R2: lambda_med(b) = c0 + b / r_inf over the ladder's READ rungs [(label, b bytes, lambda ns)]; READ with all
    F_RUNGS rungs and the standard error of 1 / r_inf at most 5% of it."""
    ok = [(lab, b, lam) for lab, b, lam in rungs if _fin(b, lam)]
    out = {"rungs": [[lab, b, lam] for lab, b, lam in rungs]}
    if len(ok) < P.F_RUNGS:
        return {**out, "status": f"{NI}: {len(ok)} of {P.F_RUNGS} rungs READ"}
    fit = ols([b for _l, b, _x in ok], [lam for _l, _b, lam in ok])
    if fit is None or not _fin(fit["s"]) or fit["s"] <= 0:
        return {**out, "status": f"{NI}: the fit is not finite or its slope not positive"}
    rel = fit["se_s"] / fit["s"] if _fin(fit["se_s"]) else None
    out.update(c0_us=fit["a"] / 1e3, r_inf_gbps=1.0 / fit["s"], se_inv_rinf_rel=rel)
    if rel is None or not rel <= P.SE_INV_RINF_MAX:
        return {**out, "status": f"{NI}: the standard error of 1 / r_inf is {rel} of it, over {P.SE_INV_RINF_MAX}"}
    return {**out, "status": "READ"}


def beta_q(points: list, need: int) -> dict:
    """PC-R3: d ln r / d ln Q_B by least squares over [(label, r, Q_B)]; READ with `need` points."""
    ok = [(lab, r, q) for lab, r, q in points if _fin(r, q) and r > 0 and q > 0]
    out = {"points": [[lab, r, q] for lab, r, q in points]}
    if len(ok) < need:
        return {**out, "status": f"{NI}: {len(ok)} of {need} configs READ with a PTX in-flight count"}
    fit = ols([math.log(q) for _l, _r, q in ok], [math.log(r) for _l, r, _q in ok])
    if fit is None:
        return {**out, "status": f"{NI}: the configs share one Q_B"}
    qs = [q for _l, _r, q in ok]
    return {**out, "beta_Q": fit["s"], "se": fit["se_s"], "q_ratio": max(qs) / min(qs), "status": "READ",
            "null": "scored against the PTX in-flight ratio (trap 13): H1 predicts beta_Q near 1 at low load"}


def beta_f(r_lo, f_lo, r_hi, f_hi) -> float | None:
    """PC-R4: ln(r_1410 / r_1710) / ln(f_1410 / f_1710), f the rungs' measured median clocks."""
    if not _fin(r_lo, f_lo, r_hi, f_hi) or min(r_lo, f_lo, r_hi, f_hi) <= 0 or f_lo == f_hi:
        return None
    return math.log(r_lo / r_hi) / math.log(f_lo / f_hi)


def sigma_of(lrs: list, cell_status: str = "READ") -> dict:
    """PC-R6 inside one cell's launches, read only when the cell reads READ under the common rule (review MAJOR-4):
    every live CTA (not the plateau: a shared CTA at half the rate outlives the plateau, so the plateau rule would hide
    H3), shared (m >= 1.8) against alone (m <= 1.2); each group's median n_c is printed beside."""
    if cell_status != "READ":
        return {"status": f"{NI}: the cell is not READ ({cell_status})"}
    valid = [x for x in lrs if x.get("valid")]
    if not valid:
        return {"status": f"{NI}: no valid launch"}
    r = np.concatenate([x["r"] for x in valid])
    m = np.concatenate([x["m"] for x in valid])
    n = np.concatenate([x["n"] for x in valid])
    sh, al = r[m >= P.SHARED_M], r[m <= P.ALONE_M]
    out = {"shared_ctas": int(sh.size), "alone_ctas": int(al.size),
           "n_median_shared": float(np.median(n[m >= P.SHARED_M])) if sh.size else None,
           "n_median_alone": float(np.median(n[m <= P.ALONE_M])) if al.size else None}
    if sh.size < P.MIN_SHARE_CTAS or al.size < P.MIN_SHARE_CTAS:
        return {**out, "status": f"{NI}: under {P.MIN_SHARE_CTAS} CTAs in a group (the scheduler did not mix them)"}
    return {**out, "sigma": float(np.median(sh) / np.median(al)), "status": "READ"}


def transition(rungs: list, x_sat_pool: list) -> dict:
    """PC-R7 over the D64 ladder's READ rungs [(label, G, r, X, n, Q_B)] and the X_sat pool [(label, X)]."""
    out = {"x_sat_pool": x_sat_pool}
    xs = [x for _l, x in x_sat_pool if _fin(x)]
    x_sat = float(st.median(xs)) if xs else None
    out["X_sat"] = x_sat
    out["triad_gbps"] = P.TRIAD_GBPS
    plat = [(lab, G, r) for lab, G, r, _X, _n, _q in rungs if G in P.PLATEAU_GS and _fin(r)]
    r0 = float(st.median([r for _l, _g, r in plat])) if plat else None
    out["r0"] = r0
    within = [lab for lab, _g, r in plat if r0 and abs(r / r0 - 1) <= P.PLATEAU_WITHIN]
    out["plateau_rungs_within"] = within
    sat = [lab for lab, _G, _r, X, _n, _q in rungs if x_sat and _fin(X) and abs(X / x_sat - 1) <= P.SAT_WITHIN]
    out["sat_rungs_within"] = sat
    out["status_plateau"] = "READ" if r0 and len(within) >= 3 else f"{NI}: {len(within)} rungs within 10% of r0"
    out["status_sat"] = "READ" if x_sat and len(sat) >= P.SAT_RUNGS else f"{NI}: {len(sat)} rungs within 3% of X_sat"
    rows = []
    for lab, G, r, X, n, q in sorted(rungs, key=lambda z: z[1]):
        row = {"label": lab, "G": G, "r": r, "X": X, "n": n, "Q_B": q,
               "N_if": (n * q) if _fin(n, q) else None,
               "L_gpu_us": (q * n / X / 1e3) if _fin(n, q, X) and X > 0 else None}
        if out["status_plateau"] == "READ" and out["status_sat"] == "READ" and _fin(X):
            row["deficit"] = X / min(G * r0, x_sat) - 1
        rows.append(row)
    out["rows"] = rows
    g90 = [rw["G"] for rw in rows if out["status_sat"] == "READ" and _fin(rw["X"]) and rw["X"] >= P.KNEE_FRAC * x_sat]
    out["G_90"] = min(g90) if g90 else None
    return out


def l_gpu_at(rows: list, X: float) -> float | None:
    """L_gpu read off the ladder at X: piecewise linear in X over the READ rows, NOT IDENTIFIED outside their span."""
    pts = sorted((rw["X"], rw["L_gpu_us"]) for rw in rows if _fin(rw.get("X"), rw.get("L_gpu_us")))
    if len(pts) < 2 or not _fin(X) or not pts[0][0] <= X <= pts[-1][0]:
        return None
    xs, ys = zip(*pts)
    return float(np.interp(X, xs, ys))


# ================================================================== Phi: PC-R10, PC-R11

def reread(rows, experts, ntpp: int, N: int, K: int, tile, *, w_c: int, counts, H: int, D: float | None) -> dict:
    """PC-R10 on one w1 launch: the re-read pairs (C3-R2's pairs: same expert, same pid_n, pid_m - 1), their launch gap
    in units of M4's gap P L / W_c, life overlap, the B bytes streamed GPU-wide between partner starts, the compulsory
    bytes C (each active expert's w1 once plus the gathered A rows once: T k rows x H x 2, phi-split's C, so the miss
    fraction is comparable with REPORT.md's 0.73 and 0.38), the re-read demand R (pairs x the slab), the measured
    miss fraction (D - C) / R with D B3's counted w1 bytes for the cell, and the zero-constant rule's bytes (a re-read
    misses iff its life does not overlap its partner's and more than 60 MiB stream between their starts)."""
    c = RC3.decode(rows)
    bm, bn = int(tile[0]), int(tile[1])
    Pn = -(-int(N) // bn)
    lt = int(ntpp) // bm
    live = lt * Pn
    s = c["start_ns"][:live].astype(np.float64)
    e = c["end_ns"][:live].astype(np.float64)
    life = e - s
    if live < 2 or (life <= 0).any():
        return {"status": f"{NR}: no live CTA, or a life not positive"}
    L = float(np.median(life))
    r = RC3.pairs(np.asarray(experts).reshape(-1)[:lt], Pn)
    slab = int(K) * bn * 2
    act = sum(1 for x in counts if int(x))
    C = act * int(N) * int(K) * 2 + sum(int(x) for x in counts) * int(H) * 2
    Rd = int(r.size) * slab
    out = {"pairs": int(r.size), "C_bytes": C, "R_bytes": Rd, "D_bytes": D, "L_ns": L, "slab_bytes": slab}
    if not r.size:
        out.update(status="READ (no re-read)", phi_meas=None, rule_bytes=C)
        return out
    g = RC3.gap_ratio(s, r, Pn, L, w_c)
    ov = np.maximum(0.0, np.minimum(e[r], e[r - Pn]) - np.maximum(s[r], s[r - Pn]))
    b = np.full(live, float(slab))
    t, val, cum = step_integral(s, e, b / life)
    dB = integral_at(t, val, cum, s[r]) - integral_at(t, val, cum, s[r - Pn])
    miss = (ov <= 0) & (dB > P.REREAD_L2_MIB * 2 ** 20)
    out.update(gap_median=float(np.median(g)), together_share=float(np.mean(g < RC3.GAP_TOGETHER)),
               overlap_share=float(np.mean(ov > 0)), dB_median_mib=float(np.median(dB) / 2 ** 20),
               rule_miss_share=float(np.mean(miss)), rule_bytes=float(C + miss.sum() * slab),
               phi_meas=(float((D - C) / Rd) if _fin(D) and Rd else None), status="READ")
    out["rule_over_counted"] = (out["rule_bytes"] / D - 1) if _fin(D) and D > 0 else None
    return out


def life_packing(T_pred: float, T_stamp: float, L_med: float, live: int, w_c: int, o: float) -> dict | None:
    """PC-R11 (C3-R3's exact split, overhead-corrected): ln(T_pred / T_stamp) = LIFE + PACKING."""
    if not _fin(T_pred, T_stamp, L_med, o) or min(T_pred, T_stamp, L_med) <= 0 or live < 1:
        return None
    Ts, Lm = T_stamp / (1 + o), L_med / (1 + o)
    L_pred = T_pred * w_c / live
    life = math.log(L_pred / Lm)
    pack = math.log(live * Lm / (w_c * Ts))
    return {"total": math.log(T_pred / Ts), "life": life, "packing": pack}


# ================================================================== loading the tree

def unit_dir(results: Path, suffix: str) -> tuple[Path | None, str]:
    d = sorted(p for p in Path(results).glob(f"*-{suffix}") if p.is_dir())
    if len(d) != 1:
        return None, f"{len(d)} directories *-{suffix}"
    return d[0], ""


def env_word(path: Path | None, prefix: str, vid: str) -> str:
    return RC3.env_line(path, prefix, vid)


def overheads(perturb: dict, vid: str) -> dict:
    """(stem, GEMM) -> {median, worst, plain_ms} of a variant's cells in a perturb unit's perturb.json."""
    v = (perturb.get("variants") or {}).get(vid) or {}
    out = {}
    for c in v.get("cells", []):
        stem = c.get("stem") or RC3_stem(c)
        out[(stem, c["gemm"])] = {"median": c.get("overhead_median"), "worst": c.get("overhead_worst"),
                                  "plain_ms": c.get("plain_ms")}
    return out


def RC3_stem(c: dict) -> str:
    return f"{c['label']}-n{c['n']}-{c.get('arm', 'native')}"


def versions_ok(v: dict | None) -> bool:
    v = v or {}
    for k, want in P.VERSIONS.items():
        got = v.get(k)
        if k in P.VERSIONS_MAJOR_MINOR:
            if got is None or str(got).split(".")[:2] != want.split(".")[:2]:
                return False
        elif got != want:
            return False
    return True


def inflight_of(man: dict) -> dict:
    """tile key -> n_if from a stamps manifest's compiled configs (every config of the tile agreeing), else None."""
    by = {}
    for x in man.get("sass") or []:
        cfgk = x.get("config") or []
        if len(cfgk) < 6:
            continue
        key = P.tile_key((cfgk[0], cfgk[1], cfgk[2], cfgk[5], cfgk[4]))
        inf = x.get("inflight")
        if inf is None and x.get("ptx_file"):
            inf = {"n_if": None}
        by.setdefault(key, []).append((inf or {}).get("n_if"))
    return {k: (v[0] if len(set(v)) == 1 else None) for k, v in by.items()}


def lock_after(session: Path, uid: str | None, lock: int) -> tuple[bool, str]:
    """The unit's lock read back after its tool (driver pc-locks.env) equals its lock."""
    if uid is None:
        return True, "C3's probe unit: locked_run's read-back before the tool"
    f = Path(session) / "gh200-driver" / "pc-locks.env"
    key = "PC_LOCK_" + "".join(ch if ch.isalnum() else "_" for ch in uid)
    lines = [ln for ln in (f.read_text().splitlines() if f.is_file() else []) if ln.startswith(key + "=")]
    if not lines:
        return False, f"no {key} line in pc-locks.env"
    last = lines[-1]
    ok = f"after={lock}" in last.split() and last.split("=", 1)[1].split()[0] == str(lock)
    return ok, last


def board(session: Path) -> dict:
    """The board's preliminaries: C2's power read-back (700 W), both lock probes held, G-BAND (for the Phi readings)."""
    out = RC.board(session)
    D = Path(session) / "gh200-driver"
    locks = (D / "locks.env").read_text().split() if (D / "locks.env").is_file() else []
    out["lock_probe_1710"] = f"LOCK_PROBE_{P.DECODE_LOCK}=held" in locks
    why = []
    pw = out.get("c2_power_env") or ""
    if f"C2_POWER_W={P.POWER_W}" not in pw.split():
        why.append(f"c2-power.env does not record C2_POWER_W={P.POWER_W}")
    if f"LOCK_PROBE_{P.SESSION_LOCK}=held" not in locks:
        why.append(f"no LOCK_PROBE_{P.SESSION_LOCK}=held")
    if not out["lock_probe_1710"]:
        why.append(f"no LOCK_PROBE_{P.DECODE_LOCK}=held")
    out["valid_decode"] = not why
    out["why_decode"] = "; ".join(why)
    out["valid_phi"] = bool(out.get("valid"))
    return out


def stamp_unit(results: Path, session: Path, label: str, R: dict, perturbs: dict, regchecks: dict,
               census_occ: dict) -> dict:
    """One stamps unit: its validity (PC-R0) and every launch read, grouped by (cell stem, GEMM)."""
    suffix, vid, uid, src, lock, pt, rc = STAMP_UNITS[label]
    d, dwhy = unit_dir(results, suffix)
    gate_env = (perturbs.get(pt) or {}).get("dir")
    g = env_word(gate_env / "gate.env" if gate_env else None, "GATE", vid)
    h = env_word(gate_env / "gate.env" if gate_env else None, "HARD", vid)
    rcd = (regchecks.get(rc) or {}).get("dir")
    rcw = env_word(rcd / "regcheck.env" if rcd else None, "REGCHECK", vid)
    u = {"variant": vid, "gate": g or None, "hard": h or None, "regcheck": rcw or None, "lock": lock,
         "page_label": "CLEAN" if g == "PASS" else ("PERTURBED" if g else None)}
    why = []
    if h != "PASS" or not g:
        why.append(f"gate hard legs {h or 'absent'}")
    if rcw != "PASS":
        why.append(f"regcheck {rcw or 'absent'}")
    ok, lw = lock_after(session, uid, lock)
    u["lock_after"] = lw
    if not ok:
        why.append(f"the unit's lock read back after the tool: {lw}")
    if pt in PERTURB_UIDS:   # the perturb unit sets o and the plain reference (review MINOR-11)
        ok2, lw2 = lock_after(session, PERTURB_UIDS[pt][0], PERTURB_UIDS[pt][1])
        u["perturb_lock_after"] = lw2
        if not ok2:
            why.append(f"its perturb unit's lock read back after the tool: {lw2}")
    if d is None:
        u.update(valid=False, why="; ".join(why + [f"no stamp page ({dwhy})"]), cells={}, launches=0)
        return u
    man = json.loads((d / "stamps.json").read_text())
    if man.get("page_label") != u["page_label"]:
        why.append(f"the page reads {man.get('page_label')}, the gate {u['page_label']}")
    want_sha = R["pages"][src]["sha256"]
    got_sha = man.get("source_sha256") or man.get("histogram_sha256")
    if got_sha != want_sha:
        why.append("the page is not the registered file")
    if not versions_ok(man.get("versions")):
        why.append(f"versions {man.get('versions')} are not {P.VERSIONS} (python on major.minor)")
    u["partial"] = man.get("complete") is False
    if u["partial"]:   # refused, not read: the stamped compiles' SASS legs are written only at the unit's end
        why.append("PARTIAL page (the unit stopped before its end): its stamped configs' SASS legs are unread, so it "
                   "is refused (the cells stay on disk, DESCRIPTIVE only)")
    elif any(not (s.get("checks_c3") or {}).get("ok") for s in man.get("sass") or [{}]):
        why.append("a stamped config fails C3's SASS legs")
    u["inflight"] = inflight_of(man)
    over = overheads(perturbs.get(pt, {}).get("json") or {}, vid)
    cells = {}
    if "cells" in man:
        cells_meta = man["cells"]
    else:   # C3's manifest (E1): the histogram's cells
        cells_meta = {}
    timer_refs = {}
    for e in man.get("launches", []):
        stem = e.get("stem") or (RC3_stem(e) if "label" in e and "n" in e else e.get("label"))
        meta = cells_meta.get(stem) or {}
        if "error" in e:
            cells.setdefault(stem, {"meta": meta, "w1": [], "w2": [], "errors": []})["errors"].append(e["error"])
            continue
        tile = tuple(int(e[k]) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"))
        q_if = u["inflight"].get(P.tile_key(tile))
        decode = bool(meta.get("live"))
        q_guard = P.q_bytes(tile, P.L2_GUARD_STAGES) if decode else None   # review MAJOR-5: s4's Q, not the cell's
        rows = np.load(d / "stamps" / e["file"])
        ex = np.load(d / "stamps" / e["experts_file"])
        o = (over.get((stem, e["gemm"])) or {}).get("median")
        expl = (meta.get("live") or {}).get(e["gemm"])
        tref = timer_refs.get((stem, e.get("call"))) if e["gemm"] == "w2" else None
        lr = launch(rows, ex, e["num_tokens_post_padded"], e["N"], e["K"], tile, o=o, expected_live=expl, timer_ref=tref,
                    launch_ms=e.get("launch_ms"), flush=bool(e.get("flush", True)), use_td=bool(e.get("use_td")),
                    late=int(e.get("late_buffers") or 0), q_guard=q_guard, slab_once=decode)
        if e["gemm"] == "w1":
            timer_refs[(stem, e.get("call"))] = {"res": lr.get("timer_res_ns"), "f_long": lr.get("f_long")}
        lr.update(call=e.get("call"), gemm=e["gemm"], N=e["N"], K=e["K"], ntpp=e["num_tokens_post_padded"],
                  tile=tile, q_if=q_if, rows_file=e["file"], experts_file=e["experts_file"])
        if label in ("se1", "se2") and e["gemm"] == "w1":
            lr.update(_rows=rows, _experts=ex)   # PC-R10 reads the pairs off the raw rows
        c = cells.setdefault(stem, {"meta": meta, "w1": [], "w2": [], "errors": []})
        c[e["gemm"]].append(lr)
    # PC-R0: first w2 start after the last w1 end, per call
    for stem, c in cells.items():
        bad = []
        for call in sorted({x.get("call") for x in c["w1"]} & {x.get("call") for x in c["w2"]}):
            a = [x for x in c["w1"] if x.get("call") == call and "t_last" in x]
            b = [x for x in c["w2"] if x.get("call") == call and "t_first" in x]
            if a and b and not b[0]["t_first"] > a[0]["t_last"]:
                bad.append(call)
        if bad:
            for g_ in ("w1", "w2"):
                for x in c[g_]:
                    if x.get("call") in bad:
                        x.update(valid=False, why=(x.get("why") or "") + "; w2 started before w1 ended")
        c["over"] = {g_: over.get((stem, g_)) for g_ in ("w1", "w2")}
    u.update(valid=not why, why="; ".join(why), cells=cells, dir=d.name, launches=len(man.get("launches", [])),
             source=src, kind=man.get("kind"))
    return u


def gemm_reading(u: dict, stem: str, gemm: str = "w2") -> dict | None:
    """A cell GEMM's reading in a unit; None when the cell is absent; NOT READ when the unit is invalid."""
    c = (u.get("cells") or {}).get(stem)
    if c is None:
        return None
    G = ((c["meta"].get("live") or {}).get(gemm)) or (c[gemm][0]["live"] if c[gemm] else 0)
    cg = cell_gemm(c[gemm], int(G or 0), (c.get("over") or {}).get(gemm))
    if not u.get("valid"):
        cg["status"] = f"{NR}: the unit is invalid ({u.get('why')})"
    cg["page_label"] = u.get("page_label")
    return cg


# ================================================================== the whole reading

def reg() -> dict:
    return json.loads(REG_PATH.read_text())


def commit_paths(R: dict) -> list:
    return sorted(set(COMMIT_FILES) | set(R["code_pins"]))


def check_repo_modules(R: dict, root: Path = REPO, modules=None) -> list:
    mods = list(sys.modules.values()) if modules is None else list(modules)
    loaded = RC.repo_modules(mods, root, RC.tracked_files(root))
    bad = [(p, "loaded by the reader but not pinned") for p in loaded if p not in R["code_pins"]]
    bad += [(p, "not its registered sha256") for p in loaded
            if p in R["code_pins"] and RC.sha256(Path(root) / p) != R["code_pins"][p]]
    return sorted(set(bad))


def census_occupancy(results: Path) -> tuple[dict, str]:
    d, why = unit_dir(results, OTHER_UNITS["census"])
    f = d / "tile-census.json" if d else None
    if f is None or not f.is_file():
        return {}, why or "no tile-census.json"
    doc = json.loads(f.read_text())
    if doc.get("path_mismatch"):
        return {}, f"census path mismatch {doc['path_mismatch']}"
    out = {}
    for t, x in (doc.get("summary") or {}).items():
        vals = set((x.get("by") or {}).values())
        out[t] = vals.pop() if len(vals) == 1 else None
    return out, ""


def occupancy_check(census: dict, regcheck_json: dict) -> dict:
    """The registered occupancy (pccommon.OCCUPANCY) on the board: the census unit's reading of each registered tile;
    the wave-dependent readings (R7's X_sat pool, R10, R11) read only where it equals the registered value."""
    out = {}
    for t, want in P.OCCUPANCY.items():
        got = census.get(t)
        out[t] = {"registered": want, "census": got, "ok": got == want}
    return out


def read(pc: Path, *, registration: dict | None = None, _skip_commit_check_for_tests: bool = False) -> dict:
    R = registration or reg()
    pc = Path(pc)
    results, session = pc / "results", pc / "session"
    out = {"registration": REG_NAME, "pc": str(pc),
           "status": "MEASUREMENT, DESCRIPTIVE: certifies nothing, registers no model form and no constant"}
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
    brd = board(session)
    out["board"] = brd
    census, cwhy = census_occupancy(results)
    occ = occupancy_check(census, {})
    out["census"] = {"occupancy": census, "why": cwhy, "registered_check": occ,
                     "design_occupancy_printed": P.DESIGN_OCCUPANCY}
    perturbs, regchecks = {}, {}
    for k in ("ptp", "ptc", "pt3"):
        d, w = unit_dir(results, OTHER_UNITS[k])
        perturbs[k] = {"dir": d, "why": w, "json": json.loads((d / "perturb.json").read_text())
                       if d and (d / "perturb.json").is_file() else {}}
    for k in ("rcp", "rc3"):
        d, w = unit_dir(results, OTHER_UNITS[k])
        regchecks[k] = {"dir": d, "why": w}
    units = {}
    for label in STAMP_UNITS:
        u = stamp_unit(results, session, label, R, perturbs, regchecks, census)
        if label != "se1" and label != "se2" and not brd["valid_decode"]:
            u["valid"], u["why"] = False, "; ".join(x for x in (u["why"], f"board: {brd['why_decode']}") if x)
        if label in ("se1", "se2") and not brd["valid_phi"]:
            u["valid"], u["why"] = False, "; ".join(x for x in (u["why"], f"board (G-BAND): {brd.get('why')}") if x)
        units[label] = u
    out["PC-R0"] = {lab: {k: u.get(k) for k in ("variant", "valid", "why", "page_label", "gate", "hard", "regcheck",
                                                "lock", "lock_after", "dir", "launches", "inflight")}
                    for lab, u in units.items()}
    A, Cu, Dd, Rr = units["sa"], units["sc"], units["sd"], units["sr"]

    def rd(u, stem, gemm="w2"):
        return gemm_reading(u, stem, gemm)

    def q_of(u, stem):
        c = (u.get("cells") or {}).get(stem)
        if not c or not (c["w2"] or c["w1"]):
            return None
        t = (c["w2"] or c["w1"])[0]["tile"]
        n_if = (u.get("inflight") or {}).get(P.tile_key(t))
        return P.q_bytes(t, n_if) if n_if == int(t[4]) - 1 else None

    # ---- PC-R1: the rate curves
    r1 = {}
    for lab, u in (("A (D64, 1710)", A), ("C (D64, 1410)", Cu), ("D (D16, 1710)", Dd), ("R (replicate)", Rr)):
        rows = []
        for stem, c in sorted((u.get("cells") or {}).items()):
            g = rd(u, stem)
            if g is None:
                continue
            rows.append({"cell": stem, "tile": c["meta"].get("tile"), "shape": c["meta"].get("shape"),
                         "G": g.get("G"), "r": g.get("r"), "n": g.get("n"), "m": g.get("m"), "f_mhz": g.get("f_mhz"),
                         "X": g.get("X"), "lam_us": (g.get("lam_ns") or 0) / 1e3 if g.get("lam_ns") else None,
                         "status": g.get("status"), "overhead": g.get("overhead"), "page_label": g.get("page_label"),
                         "f_in_band_printed": (P.F_BAND[u["lock"]][0] <= g["f_mhz"] <= P.F_BAND[u["lock"]][1])
                         if _fin(g.get("f_mhz")) else None,
                         "per_expert_r": g.get("per_expert_r"), "expert_spread": g.get("expert_spread"),
                         "expert_flag": g.get("expert_flag"), "r_alone": g.get("r_alone"),
                         "alone_ctas": g.get("alone_ctas"), "r_shared": g.get("r_shared"),
                         "shared_ctas": g.get("shared_ctas")})
        r1[lab] = rows
    out["PC-R1"] = r1

    # ---- PC-R2: the F ladders
    def ladder(rungs):
        pts = []
        for u, stem in rungs:
            g = rd(u, stem)
            pts.append((stem, g.get("b") if g else None, g.get("lam_ns") if g and g.get("status") == "READ" else None))
        return f_ladder(pts)
    lad8 = ladder([(A, "A-F1792-G8"), (A, "A-F3584-G8"), (A, "A-F7168-G8"), (A, "A-G8")])
    lad128 = ladder([(A, "A-TP8-G128"), (A, "A-TP4-G128"), (A, "A-TP2-G128"), (A, "A-MX-T128-A4")])
    ds = {}
    try:
        dsr = json.loads((REPO / DS_READINGS).read_text())
        ds = ((((dsr.get("SUBWAVE") or {}).get("tiles") or {}).get("64x128x64w4s3") or {}).get("offsets_us") or {})
    except (OSError, ValueError):
        ds = {}
    out["PC-R2"] = {"G8 (synthetic F ladder)": lad8, "G128 (Mixtral tp8, tp4, tp2, 8x7B)": lad128,
                    "DS offsets beside (64x128x64w4s3, board cce89f, cross-board)": ds}

    # ---- PC-R3: stage and BLOCK_K elasticity
    def corr_r(u, stem, lad):
        g = rd(u, stem)
        if not g or g.get("status") != "READ":
            return None, "NOT READ"
        if lad.get("status") == "READ":
            lam = g["lam_ns"] - lad["c0_us"] * 1e3
            return (g["b"] / lam if lam > 0 else None), "c0-corrected"
        return g["r"], "uncorrected (PC-R2 not READ at this G)"
    r3 = {}
    for G, lad, base, pre in ((8, lad8, "A-G8", "A-G8"), (128, lad128, "A-G128", "A-G128"),
                              (384, {"status": "none"}, "A-G384", "A-G384")):
        pts64, labs = [], {}
        for stem in (f"{pre}-s2k64", base, f"{pre}-s4k64"):
            r, how = corr_r(A, stem, lad)
            pts64.append((stem, r, q_of(A, stem)))
            labs[stem] = how
        fit64 = beta_q(pts64, 3)
        fit64["rates"] = labs
        res = {"BK64 (s2, s3, s4)": fit64}
        if G != 384:
            pts128 = []
            for stem in (f"{pre}-s2k128", f"{pre}-s3k128"):
                r, how = corr_r(A, stem, lad)
                pts128.append((stem, r, q_of(A, stem)))
            res["BK128 (s2, s3)"] = beta_q(pts128, 2)
            ra, rb = corr_r(A, f"{pre}-s2k128", lad)[0], corr_r(A, base, lad)[0]
            res["rho_seg r(s2, BK128) / r(s3, BK64)"] = (ra / rb) if _fin(ra, rb) and rb > 0 else None
        r3[f"G {G}"] = res
    for G, base in ((16, "D-G16"), (128, "D-G128")):
        pts = [(stem, r_of(rd(Dd, stem)), q_of(Dd, stem)) for stem in (f"{base}-s2", f"{base}-s3", base)]
        r3[f"D16 G {G} (BK128: s2, s3, s4), DESCRIPTIVE"] = beta_q(pts, 3)
    out["PC-R3"] = r3

    # ---- PC-R4: clock elasticity
    r4 = {}
    for cstem, astem in (("C-G8", "A-G8"), ("C-G32", "A-G32"), ("C-G128", "A-G128"), ("C-G256", "A-G256"),
                         ("C-G512", "A-G512"), ("C-G1024", "A-G1024"), ("C-G8-s2k64", "A-G8-s2k64"),
                         ("C-G8-s4k64", "A-G8-s4k64"), ("C-MX-T128-A4", "A-MX-T128-A4")):
        lo, hi = rd(Cu, cstem), rd(A, astem)
        rl, rh = r_of(lo), r_of(hi)
        bf = beta_f(rl, (lo or {}).get("f_mhz"), rh, (hi or {}).get("f_mhz"))
        r4[f"{cstem} / {astem}"] = {"beta_f": bf, "r_1410": rl, "r_1710": rh, "f_1410": (lo or {}).get("f_mhz"),
                                    "f_1710": (hi or {}).get("f_mhz"),
                                    "status": "READ" if bf is not None else f"{NI}: a twin not READ",
                                    "nulls": "compute-bound predicts 1, latency-bound 0"}
    out["PC-R4"] = r4

    # ---- PC-R5: warps
    r5 = {}
    for w8, w4 in (("A-G8-w8", "A-G8"), ("A-G128-w8", "A-G128")):
        a, b = r_of(rd(A, w8)), r_of(rd(A, w4))
        r5[f"{w8} / {w4}"] = {"omega": a / b if _fin(a, b) and b > 0 else None,
                              "status": "READ" if _fin(a, b) else f"{NI}: a twin not READ"}
    out["PC-R5"] = r5

    # ---- PC-R6: SM sharing inside one launch
    r6 = {}
    for G in P.SIGMA_GS:
        c = (A.get("cells") or {}).get(f"A-G{G}")
        g6 = rd(A, f"A-G{G}")
        r6[f"A-G{G}"] = sigma_of(c["w2"], (g6 or {}).get("status", "absent")) if c and A.get("valid") else \
            {"status": f"{NI}: the cell is absent or invalid"}
    out["PC-R6"] = r6

    # ---- PC-R7: the transition into saturation
    occ_ok = occ.get(P.tile_key(P.D64), {}).get("ok")
    ladder_rows, pool = [], []
    for stem, c in sorted((A.get("cells") or {}).items()):
        meta = c["meta"]
        if meta.get("tile") != P.tile_key(P.D64):
            continue
        g2 = rd(A, stem)
        if stem.startswith("A-G") and stem[3:].isdigit() and g2 and g2.get("status") == "READ":
            ladder_rows.append((stem, g2["G"], g2["r"], g2["X"], g2["n"], q_of(A, stem)))
            if occ_ok and g2["G"] >= P.SMS * P.OCCUPANCY[P.tile_key(P.D64)]:
                pool.append((f"{stem} w2 K {c['w2'][0]['K']}", g2["X"]))
        g1 = rd(A, stem, "w1")
        if occ_ok and g1 and g1.get("status") == "READ" and (meta.get("live") or {}).get("w1", 0) >= \
                P.SMS * P.OCCUPANCY[P.tile_key(P.D64)] and meta.get("flush", True) and c["w1"] and \
                int(c["w1"][0]["K"]) >= P.X_SAT_W1_MIN_K:
            pool.append((f"{stem} w1 K {c['w1'][0]['K']}", g1["X"]))
    r7 = transition(ladder_rows, pool if occ_ok else [])
    if not occ_ok:
        r7["status_sat"] = f"{NI}: the census does not read the registered D64 occupancy ({occ.get(P.tile_key(P.D64))})"
    out["PC-R7"] = r7

    # ---- PC-R8: transfer to real shapes, and the CAL2 cell
    r8 = {}
    for anchor, syn, note in (("A-MX-T128-A4", "A-G128", "same tile, config, G and F"),
                              ("A-JM-T128-A4", "A-G64", "F differs (5632 against 14336)"),
                              ("A-TP8-G128", "A-G128", "F differs (1792 against 14336)"),
                              ("A-TP4-G128", "A-G128", "F differs (3584 against 14336)"),
                              ("A-TP2-G128", "A-G128", "F differs (7168 against 14336)")):
        ga = rd(A, anchor)
        a, s = r_of(ga), r_of(rd(A, syn))
        r8[f"{anchor} / {syn}"] = {"t": a / s - 1 if _fin(a, s) and s > 0 else None, "note": note,
                                   "anchor_expert_spread": (ga or {}).get("expert_spread"),
                                   "anchor_expert_flag": (ga or {}).get("expert_flag")}
    for u, anchor, syn, note in ((Cu, "C-MX-T128-A4", "C-G128", "the 1410 twin"),
                                 (Dd, "D-MX-T16-A2", "D-G128", "D16, E 8 against 32"),
                                 (Dd, "D-MX-T4-A2", "D-G128", "D16 row fill: 4 of 16 rows per M-tile"),
                                 (Dd, "D-MX-T2-A2", "D-G128", "D16 row fill: 2 of 16 rows per M-tile")):
        ga = rd(u, anchor)
        a, s = r_of(ga), r_of(rd(u, syn))
        r8[f"{anchor} / {syn}"] = {"t": a / s - 1 if _fin(a, s) and s > 0 else None, "note": note,
                                   "anchor_expert_spread": (ga or {}).get("expert_spread"),
                                   "anchor_expert_flag": (ga or {}).get("expert_flag")}
    anc = rd(A, "A-MX-T128-A4")
    lg = l_gpu_at(r7.get("rows") or [], (anc or {}).get("X"))
    qb = q_of(A, "A-MX-T128-A4")
    cal2 = {"L_gpu_us_at_anchor_X": lg, "Q_B": qb, "r_anchor": r_of(anc)}
    if lg and qb and _fin(r_of(anc)):
        pred = qb / (lg * 1e3)
        cal2.update(r_pred=pred, ratio_minus_1=r_of(anc) / pred - 1,
                    reading=("CARRIES: the curve prices the CAL2 cell within 10%" if abs(r_of(anc) / pred - 1)
                             <= P.CARRY_WITHIN else "SHAPE-SPECIFIC: over 10% off the curve (A bytes, E, addresses)"))
    else:
        cal2["reading"] = f"{NI}: the anchor or L_gpu at its X is not read"
    r8["does the curve price the CAL2 cell (Mixtral T128 A4)"] = cal2
    out["PC-R8"] = r8

    # ---- PC-R9: the L2 control
    a, b = r_of(rd(A, "A-G8-noflush")), r_of(rd(A, "A-G8"))
    out["PC-R9"] = {"ratio flush off / on": a / b if _fin(a, b) and b > 0 else None,
                    "status": "READ" if _fin(a, b) else f"{NI}: a twin not READ",
                    "guard": f"a flushed launch with a CTA over Q_max / {P.L2_LATENCY_NS:g} ns is void (ROW SANITY)",
                    "overhead_note": ("the flush-off cell's overhead o is the perturb unit's, measured on FLUSHED bursts "
                                      "(the burst timer always flushes); its rate is corrected by that o")}

    # ---- PC-R10 and PC-R11: Phi
    priced = {(r["cls"], r["n"], r["label"]): r for r in json.loads((REPO / P.PRICED).read_text())["rows"]}
    out["PC-R10"] = {k: phi_pairs(units[lab], cls, w_c, occ.get(t, {}).get("ok"), priced, board_counted(results, bu))
                     for k, lab, cls, w_c, t, bu in (
                         ("E1 (128x128, W_c 264)", "se1", "128g1", 264, "128x128x64w8s3", "b128g1"),
                         ("E2 (64x128, W_c 396)", "se2", "64g1", 396, "64x128x64w8s3", "b64g1"))}
    out["PC-R11"] = phi_life_packing(units["se2"], priced, occ.get("64x128x64w8s3", {}).get("ok"))

    # ---- PC-R12: counter against time
    out["PC-R12"] = counter_vs_time(results, R, A, Dd)

    # ---- PC-R13, PC-R14
    r13, r14 = {}, []
    for lab, u in units.items():
        vals = [x["r13"] for c in (u.get("cells") or {}).values() for g_ in ("w1", "w2") for x in c[g_]
                if x.get("valid") and _fin(x.get("r13"))]
        r13[lab] = {"launches": len(vals), "median": _med(vals),
                    "flagged_under_0.97": sum(1 for v in vals if v < P.START_FLAG)}
        for stem, c in sorted((u.get("cells") or {}).items()):
            for x in c["w2"]:
                if x.get("valid") and x["live"] <= P.SMS and x.get("call") == 0:
                    r14.append({"unit": lab, "cell": stem, "G_live": x["live"], "distinct_sms": x["sms"],
                                "per_sm_max": int(np.bincount(np.unique(x["sm"], return_inverse=True)[1]).max()),
                                "start_rank_vs_pid": x["placement_rank"]})
    out["PC-R13"] = r13
    out["PC-R14"] = r14

    # ---- the replicate
    rep = {}
    for rstem, astem in (("R-G8", "A-G8"), ("R-G128", "A-G128"), ("R-G512", "A-G512")):
        a, b = r_of(rd(Rr, rstem)), r_of(rd(A, astem))
        rep[f"{rstem} / {astem}"] = a / b - 1 if _fin(a, b) and b > 0 else None
    out["replicate"] = rep

    # ---- H1's label
    x_sat = r7.get("X_sat") if r7.get("status_sat") == "READ" else None
    g16, g8 = rd(A, "A-G16"), rd(A, "A-G8")
    plat = [r_of(rd(A, f"A-G{G}")) for G in P.PLATEAU_GS]
    pr = (max(plat) / min(plat)) if all(_fin(x) and x > 0 for x in plat) else None
    p216 = (plat[0] / plat[-1]) if _fin(plat[0], plat[-1]) and plat[-1] > 0 else None
    bq = (r3["G 8"]["BK64 (s2, s3, s4)"]).get("beta_Q") if r3["G 8"]["BK64 (s2, s3, s4)"].get("status") == "READ" else None
    # the low-load premise: against X_sat when PC-R7 reads it, else against the published triad (owner decision, the
    # FALLBACK, labelled); the plateau falsifier's load condition keeps X_sat alone (no falsifier is loosened)
    low = {"denominator": "X_sat", "gbps": x_sat, "fallback": False}
    if not x_sat:
        low = {"denominator": f"FALLBACK: the triad, {P.TRIAD_FALLBACK_GBPS:g} GB/s ({P.TRIAD_FALLBACK_SOURCE})",
               "gbps": P.TRIAD_FALLBACK_GBPS, "fallback": True}
    x8 = (g8 or {}).get("X") / low["gbps"] if _fin((g8 or {}).get("X")) else None
    low["X_G8"], low["fraction"] = (g8 or {}).get("X"), x8
    lab = P.h1_label(bq, r4["C-G8 / A-G8"]["beta_f"], r5["A-G8-w8 / A-G8"]["omega"], pr, p216,
                     (g16 or {}).get("X") / x_sat if x_sat and _fin((g16 or {}).get("X")) else None,
                     r6["A-G192"].get("sigma"), x8)
    if low["fallback"]:
        lab["why"] = lab["why"] + ["the low-load premise read against the triad FALLBACK (X_sat not read)"]
    out["H1"] = {**lab, "status": "DESCRIPTIVE (owner decision 4): printed beside the numbers, never a verdict",
                 "bands": P.H1, "beta_Q_rates": r3["G 8"]["BK64 (s2, s3, s4)"].get("rates"),
                 "low_load_premise": low,
                 "X_sat_coupling": ("the plateau falsifier's load condition X(G 16) <= 0.25 X_sat reads X_sat only, so "
                                    "with X_sat not read it cannot fire; the low-load premise falls back to the triad, "
                                    "labelled")}
    bad = check_repo_modules(R) if not _skip_commit_check_for_tests else []
    if bad:
        raise SystemExit(f"REFUSED: the reader loaded a repo module that is not pinned with its registered bytes: {bad[:5]}")
    return out


def phi_pairs(u: dict, cls: str, w_c: int, occ_ok, priced: dict, board: dict | None = None) -> dict:
    """PC-R10 over a Phi stamps unit's w1 launches, per cell (the median over calls), against B3's counted w1 bytes."""
    if not u.get("valid"):
        return {"status": f"{NR}: {u.get('why')}"}
    if not occ_ok:
        return {"status": f"{NI}: the census does not read the registered occupancy for W_c {w_c}"}
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["phi-3.5-moe"]
    hist = json.loads((REPO / (P.PHI_128 if cls == "128g1" else P.PHI_64)).read_text())
    bm = int(hist["block_m"])
    counts_of = {(c["label"], int(c["n"])): c.get("counts") for c in hist["cells"]}
    cells = {}
    keys = ("pairs", "gap_median", "together_share", "overlap_share", "dB_median_mib", "rule_miss_share", "phi_meas",
            "rule_bytes", "rule_over_counted", "C_bytes", "R_bytes")
    for stem, c in sorted((u.get("cells") or {}).items()):
        label, n = stem.split("-n")[0], int(stem.split("-n")[1].split("-")[0])
        counts = counts_of.get((label, n))
        if counts is None and label == "balanced":
            counts = [n * bm] * cfg.num_experts   # balanced: n x BLOCK_M rows per expert
        if counts is None:
            cells[stem] = {"status": f"{NR}: the histogram gives no counts for {label} n {n}"}
            continue
        D = ((priced.get((cls, n, label)) or {}).get("counted_bytes") or {}).get("w1")
        per = [reread(x["_rows"], x["_experts"], x["ntpp"], x["N"], x["K"], x["tile"], w_c=w_c, counts=counts,
                      H=cfg.hidden_size, D=D) for x in c["w1"] if x.get("valid") and "_rows" in x]
        per = [q for q in per if str(q.get("status", "")).startswith("READ")]
        if not per:
            cells[stem] = {"status": f"{NR}: no readable w1 launch"}
            continue
        cells[stem] = {k: _med([q.get(k) for q in per]) for k in keys}
        cells[stem].update(D_bytes=D, calls=len(per), status="READ",
                           board_counted_w1_printed=(board or {}).get((label, n), {}).get("w1"))
    return {"cells": cells, "D_source": "B3's counted w1 bytes (board ebedab), docs/registered/2026-11-06-pc-priced.json",
            "status": "READ" if any(x.get("status") == "READ" for x in cells.values()) else f"{NI}: no cell"}


def phi_life_packing(u: dict, priced: dict, occ_ok) -> dict:
    """PC-R11 (a printed extra; C3-R3's rule, review MINOR-12): Phi 64g1 LIFE against PACKING per cell and GEMM at W_c
    396, overhead-corrected; T_stamp the stamped launch's event-pair time (median over calls); the miss is read off
    the uninstrumented reference: e_plain = mean T_pred / T_plain - 1 over the skewed cells (T_plain the perturb unit's
    plain median); READ only when e_plain is at least 2% in size and the median |o| at most a quarter of it, the
    stamped launch's live CTAs equal to the priced live; per model the MEANS of the terms."""
    if not u.get("valid"):
        return {"status": f"{NR}: {u.get('why')}"}
    if not occ_ok:
        return {"status": f"{NI}: the census does not read 3 CTAs per SM at 64x128x64 w8 s3"}
    rows, e_plain = [], []
    for stem, c in sorted((u.get("cells") or {}).items()):
        label, n = stem.split("-n")[0], int(stem.split("-n")[1].split("-")[0])
        pr = priced.get(("64g1", n, label))
        if not pr:
            continue
        for g in ("w1", "w2"):
            v = [x for x in c[g] if x.get("valid")]
            ov = (c.get("over") or {}).get(g) or {}
            o = ov.get("median")
            if not v or not _fin(o):
                continue
            live = v[0]["live"]
            row = {"cell": stem, "gemm": g, "skewed": label not in ("uniform", "balanced"), "o": o, "live": live,
                   "priced_live": pr["parts"][g]["live"]}
            if live != pr["parts"][g]["live"]:
                row["why"] = f"the stamped launch has {live} live CTAs, the priced cell {pr['parts'][g]['live']}"
                rows.append(row)
                continue
            lp = life_packing(pr["T_pred_ms"][g], _med([x["launch_ms"] for x in v]), _med([x["L_ns"] for x in v]) / 1e6,
                              live, 396, o)
            if lp:
                row.update(lp)
                if row["skewed"] and _fin(ov.get("plain_ms")) and ov["plain_ms"] > 0:
                    e_plain.append(pr["T_pred_ms"][g] / ov["plain_ms"] - 1)
                rows.append(row)
    sk = [r for r in rows if r["skewed"] and "total" in r]
    ep = float(np.mean(e_plain)) if e_plain else None
    om = _med([abs(r["o"]) for r in rows])
    if ep is None:
        status = f"{NI}: no skewed cell with a plain reference"
    elif abs(ep) < P.REPRO_MIN:
        status = f"NOT REPRODUCED: e_plain {100 * ep:.2f}% is under 2% in size on this board"
    elif om is not None and om > P.OVERHEAD_FRAC * abs(ep):
        status = "DESCRIPTIVE ONLY: the median |o| exceeds a quarter of the miss"
    else:
        status = "READ (a printed extra; the Phi miss is the byte law, phi-split REPORT.md)"

    def mean(sel, k):
        v = [r[k] for r in rows if sel(r) and k in r]
        return float(np.mean(v)) if v else None
    contrast = {k: (mean(lambda r: r["skewed"], k) - mean(lambda r: not r["skewed"], k))
                if mean(lambda r: r["skewed"], k) is not None and mean(lambda r: not r["skewed"], k) is not None else None
                for k in ("life", "packing")}
    return {"rows": rows, "e_plain_mean": ep, "means_skewed": {k: mean(lambda r: r["skewed"], k)
                                                                for k in ("total", "life", "packing")},
            "skew_contrast": contrast, "status": status, "skewed_cells_read": len(sk)}


def board_counted(results: Path, unit: str) -> dict:
    """This board's own counted Phi bytes (units b128g1, b64g1: ncu, L2 flushed per kernel), per NATIVE (label, n):
    w1 and w2 dram_bytes_read, DESCRIPTIVE (they will likely fail B3's per-cell clock rule again)."""
    d, _ = unit_dir(results, OTHER_UNITS[unit])
    f = sorted(d.glob(f"lock{P.SESSION_LOCK}/r3c-g1.json")) if d else []
    if len(f) != 1:
        return {}
    out = {}
    for c in json.loads(f[0].read_text()).get("cells", []):
        if c.get("arm") == "native":
            out[(str(c.get("histogram") or "balanced"), int(c["n"]))] = {
                g: (c.get("per_gemm") or {}).get(g, {}).get("dram_bytes_read") for g in ("w1", "w2")}
    return out


def counter_vs_time(results: Path, R: dict, A: dict, Dd: dict) -> dict:
    """PC-R12: per matching cell, the stamped launch's event-pair time against its stamps' span; with unit 16's
    counter page, gpu__time_duration against the span and the counted DRAM bytes against the construction's."""
    d, why = unit_dir(results, OTHER_UNITS["cnt"])
    page = None
    if d:
        f = sorted(d.glob(f"lock{P.DECODE_LOCK}/r3d.json"))
        page = json.loads(f[0].read_text()) if len(f) == 1 else None
    pcheck = "READ"
    if page is None:
        pcheck = f"{NR}: no counter page ({why or 'no r3d.json'})"
    else:
        man = page.get("manifest") or {}
        bad = []
        if man.get("page_sha256") != R["pages"][f"{P.PAGES_DIR}/{P.CNT_PAGE}.json"]["sha256"]:
            bad.append("page_sha256 is not the registered page's")
        if not str((page.get("gates") or {}).get("V_ALIGN_grid", "")).startswith("PASS"):
            bad.append(f"V_ALIGN_grid {(page.get('gates') or {}).get('V_ALIGN_grid')!r}")
        if page.get("kernels") != "wide":
            bad.append(f"kernels {page.get('kernels')!r}")
        if not versions_ok(man.get("versions")):
            bad.append("versions")
        if bad:
            pcheck = f"{NR}: " + "; ".join(bad)
    by = {c["label"]: c for c in (page or {}).get("cells", []) if c.get("arm") == "native"} if pcheck == "READ" else {}
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["mixtral-8x7b"]
    rows = {}
    for lab, (pg, stem) in P.CNT_MATCH.items():
        u = A if pg == "pc-a" else Dd
        c = (u.get("cells") or {}).get(stem)
        for g in ("w1", "w2"):
            v = [x for x in (c or {}).get(g, []) if x.get("valid")]
            row = {"stamped_cell": stem}
            if v:
                ev = _med([x["launch_ms"] * 1e6 for x in v])
                span = _med([x["span_ns"] for x in v])
                row.update(event_minus_span_us=(ev - span) / 1e3 if _fin(ev, span) else None)
                cc = by.get(lab)
                if cc:
                    pg_ = cc["per_gemm"][g]
                    ncu = pg_.get("gpu__time_duration.sum")
                    dram = pg_.get("dram__bytes_read.sum")
                    counts = (c or {}).get("meta", {}).get("counts") or []
                    N, K = P.gemm_dims(cfg, g)
                    build = sum(1 for x in counts if x) * N * K * 2
                    T = int((c or {}).get("meta", {}).get("T") or sum(counts) // cfg.top_k)
                    row["A_bytes_printed"] = (T * cfg.hidden_size * 2 if g == "w1"
                                              else sum(counts) * cfg.intermediate_size * 2)
                    row.update(ncu_minus_span_us=(ncu - span) / 1e3 if _fin(ncu, span) else None,
                               dram_over_construction=(dram / build) if _fin(dram) and build else None)
            rows[f"{lab} {g}"] = row
    return {"page": pcheck, "rows": rows,
            "confound": ("w2's ncu times are confounded: ncu flushes L2 before each kernel (cache-control all), so w2's "
                         "A operand, L2-hot in vLLM's call order and in the stamped calls, is a DRAM read under ncu; and "
                         "ncu's replays run at a lower clock (phi-split REPORT.md: 3 to 7% under the timed run's)")}


# ================================================================== the C3 stand-in readout

def standin(c3: Path) -> dict:
    """PC's per-launch functions on C3's published stamp pages (128x128x64 w8 s3, 1410 MHz, PERTURBED): the design's
    section 1.1 table (per-CTA rate against concurrency), X and L_gpu at saturation, the clock at kernel start and
    placement. A STAND-IN: C3's pages hold no few-CTA launch; nothing here is a PC reading."""
    res = Path(c3) / "results"
    bins = ((140, 180), (180, 220), (220, 250), (250, 265))
    out = {"what": ("PC's per-launch functions on C3's published stamp pages (128x128x64 w8 s3, 1410 MHz, PERTURBED): "
                    "a STAND-IN, not a PC reading (C3 holds no few-CTA launch; its w2 slabs are re-read, so X is a demand "
                    "rate)"), "source": str(c3), "models": {}}
    for model, unit, K in (("mixtral-8x7b", "instr-mixtral-8x7b-st7", 14336),
                           ("qwen2-57b-a14b", "instr-qwen2-57b-a14b-stq", 2560)):
        d, _w = unit_dir(res, unit)
        man = json.loads((d / "stamps.json").read_text())
        perturb = json.loads(next(res.glob("*-perturb-pt3")).joinpath("perturb.json").read_text())
        vid = man["variant"]
        over = {(c["label"], int(c["n"]), c["gemm"]): c.get("overhead_median")
                for c in perturb["variants"][vid]["cells"] if c.get("arm") == "native"}
        rr, nn, X, Lg, r13, f = [], [], [], [], [], []
        for e in man["launches"]:
            if e.get("gemm") != "w2" or "error" in e:
                continue
            rows = np.load(d / "stamps" / e["file"])
            ex = np.load(d / "stamps" / e["experts_file"])
            o = over.get((e["label"], int(e["n"]), "w2"))
            lr = launch(rows, ex, e["num_tokens_post_padded"], e["N"], e["K"], (128, 128, 64, 8, 3), o=0.0,
                        launch_ms=e.get("launch_ms"), slab_once=False)
            if not lr.get("valid"):
                continue
            rr.append(lr["r"])
            nn.append(lr["n"])
            if _fin(lr["X"]):
                X.append(lr["X"] * (1 + (o or 0.0)))
                n_pl = float(np.median(lr["n"][lr["plateau"]]))
                Lg.append(2 * 64 * 128 * 2 * n_pl / (lr["X"] * (1 + (o or 0.0))) / 1e3)
            if lr["r13"] is not None:
                r13.append(lr["r13"])
            f.append(float(np.median(lr["f"])))
        r, n = np.concatenate(rr), np.concatenate(nn)
        tbl = {}
        for lo, hi in bins:
            k = (n >= lo) & (n < hi)
            tbl[f"{lo} to {hi}"] = {"ctas": int(k.sum()), "r_median_gbps": float(np.median(r[k])) if k.any() else None}
        out["models"][model] = {"w2 rate by concurrency (raw, o 0: the design's section 1.1 table)": tbl,
                                "X_w2_median_gbps (overhead-corrected)": _med(X),
                                "L_gpu_us_median (Q_B 2 x 32 KiB at s3 x n / X)": _med(Lg),
                                "clock_at_start_median": _med(r13), "f_mhz_median": _med(f), "launches": len(rr)}
    return out


# ================================================================== text

def txt(d: dict) -> str:
    L = [f"PC READING ({d['registration']}): {d['status']}", f"  pages: {d['pc']}",
         f"  commit: {json.dumps(d['commit'], sort_keys=True)}", f"  board: {json.dumps(d['board'], sort_keys=True, default=str)}",
         f"  census: {json.dumps(d['census'], sort_keys=True, default=str)}", ""]
    h = d["H1"]
    L.append(f"H1 (per-CTA memory-level parallelism): {h['label']} [DESCRIPTIVE] {'; '.join(h['why'])}")
    L.append("  inputs " + json.dumps({k: _r(v) for k, v in h["inputs"].items()}, sort_keys=True))
    L.append("")
    for k in ("PC-R0", "PC-R1", "PC-R2", "PC-R3", "PC-R4", "PC-R5", "PC-R6", "PC-R7", "PC-R8", "PC-R9", "PC-R10",
              "PC-R11", "PC-R12", "PC-R13", "PC-R14", "replicate"):
        L.append(f"{k}:")
        t = json.dumps(d[k], sort_keys=True, default=_default)
        L.append("  " + (t if len(t) <= 20000 else t[:20000] + f" ... [TRUNCATED: {len(t) - 20000} more characters; "
                                                                "the whole reading is in pc.read.json]"))
    return "\n".join(L) + "\n"


def standin_txt(d: dict) -> str:
    L = [f"STAND-IN READOUT on C3's published stamp pages ({d['source']}): {d['what']}"]
    for m, x in d["models"].items():
        L.append(f"{m}:")
        for k, v in x.items():
            L.append(f"  {k}: {json.dumps(v, sort_keys=True, default=_default)}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pc", type=Path, help="the published PC session directory (results/, session/)")
    g.add_argument("--c3-standin", type=Path, help="the published C3 session directory: the stand-in readout")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.c3_standin:
        d = standin(a.c3_standin)
        (a.out_dir / "standin.read.json").write_text(dumps(d))
        (a.out_dir / "standin.read.txt").write_text(standin_txt(d))
        print(standin_txt(d))
        return 0
    d = read(a.pc)
    (a.out_dir / "pc.read.json").write_text(dumps(d))
    (a.out_dir / "pc.read.txt").write_text(txt(d))
    print(txt(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
