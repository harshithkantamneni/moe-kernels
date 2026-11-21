"""knee-v5 shared module (read-only offline round): K-dependent DRAM-side knee forms at 128x128, priced inside v5''.

Imports diag-v5/work/d5.py read-only (it builds the CAL pricer at import at P_KNEE 14 and asserts T0 43.23 us, bw
3597.8 GB/s: the first-build trap). Nothing in the repo or any earlier work folder is written (run with
PYTHONDONTWRITEBYTECODE=1).

A form replaces r3_timing_model.gemm_ms ONLY when TM.BLOCK_M == 128 and the call is not the floor-only bound
(bw < 1e9). At R3's tile (BLOCK_M 32) and at 64x128 the carried function runs unchanged. Per live CTA i of GEMM g:

    t_i = smax_p(floor, m_i)          floor = wave_q S'_g c / SMs  (carried)
    null  m_i = b_i / bw                                  bw = carried 3597.8 (v5'') or a ruler R
    K1    m_i = b_i / R x (S + S0) / S                    rate R S / (S + S0): S0 k-steps of the CTA's own stream lost
    K2    m_i = b_i / R + t0 / SMs                        a fixed per-CTA DRAM-idle time t0 (us of one SM's time)
    K3    m_i = b_i / R, p = ln 2 / ln(1 + c / S)        knee sharpness set by the fill/drain share c / S
b_i = sigma x box-mean reads + OUT_TILE, S = the GEMM's k-steps (K / 64). The partial-wave tail keeps v4's CORES law with
the form's rate (K1, K2) and t0 added once to the tail's DRAM time (K2).
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
DIAG = Path.home() / "code/moe-kernels-work/2026-10-22/diag-v5/work"
sys.path.insert(0, str(DIAG))
import d5  # noqa: E402  (builds the pricer at p 14 and asserts the carried fit)
from d5 import A2, RG, RULES, TM  # noqa: E402
import kneefit as KF  # noqa: E402
import v5prime as VP  # noqa: E402
import sag as S  # noqa: E402

assert TM.P_KNEE == 14.0
_X = [float(v) for v in RG.pricer().f2.x]
assert abs(_X[0] * 1e3 - 43.23) < 0.01 and abs(_X[2] - 3597.8) < 0.1, _X
BW_CARRIED = _X[2]
assert not TM.CORES_PER_LEAD

A2RES = Path.home() / "moe-kernels-exfil/runs/gh200-tile2-20261020T2242Z/tree/vm/results"
TRIAD_PAGE = next(A2RES.glob("calibration/*ceilingtriad*/measured_nvidia_gh200_480gb.yaml"))
RULER_PAGE = A2RES / "2026-10-20-nvidia_gh200_480gb-rulers-lp1410/rulers.json"


def _triad():
    for line in TRIAD_PAGE.read_text().splitlines():
        if line.strip().startswith("achieved_bandwidth_gbps:"):
            return float(line.split(":")[1])
    raise RuntimeError("no triad")


RULERS = {"triad": _triad(), "read2d@1410": float(json.loads(RULER_PAGE.read_text())["rates"]["read2d"]["gbps"])}
assert abs(RULERS["triad"] - 3726.2) < 0.1 and abs(RULERS["read2d@1410"] - 3799.7) < 0.1, RULERS

BLJ = json.loads((DIAG / "bytes_lomo.json").read_text())
MODELS = ("olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b")
F3_M4 = VP.f3_from(BLJ, "none (SEEN)", "M4")          # M4 registered at 128x128 (nothing refit), F3 at 64x128
P64 = 3.75


@dataclass(frozen=True)
class Form:
    kind: str            # null, K1, K2, K3
    R: float             # GB/s in the GEMM term at 128x128
    k: float = 0.0       # S0 (K1), t0 us (K2), c (K3)

    def label(self):
        return f"{self.kind} R {self.R:.1f} k {self.k:g}"


def p_of_S(c, S):
    return math.log(2.0) / math.log1p(c / S) if c > 0 else float("inf")


def gemm_form(win, S_, Sp, out_tile, rho, tail, cores, sigma, c_ns, sms, p, occ, f: Form):
    """TM.gemm_ms with the form's DRAM side (finite p). S_ = k-steps, Sp = floor k-steps (S + CTA_FIXED), out_tile,
    rho (CORES_RHO_GBPS), tail / cores (TM.TAIL / TM.CORES) as TM holds them at the call. Equal to TM.gemm_ms bit for bit
    for null with R = bw (asserted in check_reduction.py)."""
    R = f.R
    extra = 0.0
    tail_extra = 0.0
    if f.kind == "K1" and f.k:
        R = f.R * S_ / (S_ + f.k)
    elif f.kind == "K2" and f.k:
        extra = f.k * 1e-3 / sms
        tail_extra = f.k * 1e-3
    elif f.kind == "K3" and f.k:
        p = p_of_S(f.k, S_)
    floor = TM.wave_q(win.live, sms) * Sp * c_ns * 1e-6 / sms
    rate = R * 1e6
    if tail and occ and win.live < 2 * sms * occ:
        slots = sms * occ
        k = win.live - slots if win.live > slots else win.live
        m = (sigma * win.mean + out_tile) / rate
        if extra:
            m = m + extra
        t = TM.smax(floor, m, p)
        if cores:
            fl = math.ceil(k / sms) * Sp * c_ns * 1e-6
            lo = win.live - k
            raw = win.rcum[win.live] - win.rcum[lo] if win.rcum is not None else float(win.mean[lo:].sum())
            dr = (sigma * raw + out_tile * k) / (min(R, rho * k) * 1e6) + tail_extra
            lifetime = max(fl, dr)
        else:
            lifetime = Sp * c_ns * 1e-6 * occ
        return float(t[:win.live - k].sum() + max(float(t[win.live - k:].sum()), lifetime))
    m = (sigma * win.sorted_mean + out_tile) / rate
    if extra:
        m = m + extra
    return float(TM.smax(floor, m, p).sum())


def _tm_state(g):
    return (TM.GEOMETRY[g].ksteps, TM.floor_ksteps(g), TM.OUT_TILE, TM.CORES_RHO_GBPS, TM.TAIL, TM.CORES)


@contextmanager
def form_ctx(f: Form | None):
    """TM.gemm_ms replaced at 128x128 (bw < 1e9 calls) by the form; identity elsewhere and when f is None."""
    if f is None:
        yield
        return
    orig = TM.gemm_ms

    def w(win, g, sigma, c_ns, bw, sms, p=None, occ=None, _o=orig):
        if TM.BLOCK_M == 128 and bw < 1e9:
            return gemm_form(win, *_tm_state(g), sigma, c_ns, sms, TM.P_KNEE if p is None else p, occ, f)
        return _o(win, g, sigma, c_ns, bw, sms, p=p, occ=occ)
    TM.gemm_ms = w
    try:
        yield
    finally:
        TM.gemm_ms = orig


# ---------------------------------------------------------------- counter rows (cycles domain), captured once
def capture_rows(roles=("fit",)):
    """F2's estimator rows (bw.knee_rows' set: 128x128 NATIVE of kneefit's PAGES with the given roles, 1410), with the
    full gemm_ms call's arguments captured so any form and p can be evaluated without re-pricing."""
    rows = []
    pages = [p for p in KF.PAGES if p[4] in roles and p[2] not in ("b64g1", "p64g1", "f64w8") and p[3] == 1410]
    for root, model, unit, lock, role in pages:
        cells = KF.cells_of(root, model, unit, lock)
        with A2.scoring(A2.Rules.tileA2(lock)):
            prm = d5.params_for(d5.SEL, model)
            for r in cells:
                if r["arm"] != "native" or int(r["doc"]["block_m"]) != 128:
                    continue
                cap = {}
                orig = TM.gemm_ms

                def w(win, g, sigma, c_ns, bw, sms, p=None, occ=None, _o=orig, _cap=cap):
                    if bw < 1e9 and c_ns > 1e-3 and TM.TAIL and g not in _cap:
                        _cap[g] = (win, sigma, c_ns, bw, sms, occ, _tm_state(g), TM.GEOMETRY[g].K)
                    return _o(win, g, sigma, c_ns, bw, sms, p=p, occ=occ)
                TM.gemm_ms = w
                try:
                    pp = d5.price_parts(model, "native", r["c"], r["doc"], prm, key="CLAMP", reads=r["reads"])
                finally:
                    TM.gemm_ms = orig
                t = RG.PAGE_TILE[128]
                for g in ("w1", "w2"):
                    K = pp[g]["c_ns"] / d5.V4.c_counter(t, g, prm)
                    win, sigma, c_ns, bw, sms, occ, state, Kdim = cap[g]
                    rows.append({"board": "A2" if root == KF.A2T else "S1", "model": model, "unit": unit, "G": int(r["doc"]["group_m"]),
                                 "n": r["c"]["n"], "label": r["c"]["label"], "gemm": g, "role": role,
                                 "x": pp[g]["bytes_only"] / pp[g]["floor_only"], "meas": r["cyc"][g] * K * 1e-6,
                                 "dead": pp[g]["dead"], "S": state[0], "K": Kdim, "args": (win, state, sigma, c_ns, bw, sms, occ),
                                 "gemm14": pp[g]["gemm"], "cyc": r["cyc"][g], "reads": r["reads"][g]})
    return rows


def eval_rows(rows, f: Form, p):
    """pred / meas - 1 per row at form f and exponent p (p ignored for K3)."""
    out = []
    for r in rows:
        win, state, sigma, c_ns, bw, sms, occ = r["args"]
        pred = gemm_form(win, *state, sigma, c_ns, sms, p, occ, f) + r["dead"]
        out.append(pred / r["meas"] - 1)
    return out


def rms(v):
    v = list(v)
    return math.sqrt(st.fmean(x * x for x in v)) if v else float("nan")


# ---------------------------------------------------------------- E1-CAL (time domain) inside v5''
_READS = {}


def e1_rows():
    rows = d5.e1cal_rows()
    return [r for r in rows if not r["drop"]]


def cell_ms(r, f: Form | None, p128, sag128, sag64, sag_curve=None):
    """v5'' ms of one E1-CAL / B2 NATIVE cell at A2's selections, M4 bytes (v3 key) at 128x128, F3 at 64x128, p_64 3.75,
    F4 x 2 at both tiles, F1 Sc with (A, tau) per tile (or sag_curve: tile -> function of floor-only ms giving us)."""
    prm = d5.params_for(d5.SEL, r["model"])
    bm = int(r["doc"]["block_m"])
    key = (r["model"], r.get("page"), r["n"], r["label"], bm, int(r["doc"]["group_m"]))
    with A2.scoring(RULES):
        if key not in _READS:
            with d5.V.swapped(p128 if bm == 128 else P64, d5.V.RHO_X[bm]):
                _READS[key] = d5.V.f3_reads(RG.pricer(), r["model"], int(r["n"]), r["c"]["counts"], RG.seed_of(r["doc"], r["c"]),
                                            RG.PAGE_TILE[bm], int(r["doc"]["group_m"]), prm, F3_M4[bm], "v3")[0]
        reads = _READS[key]
        P = {128: p128, 64: P64}
        with form_ctx(f if bm == 128 else None):
            if sag_curve is None:
                return d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads=reads, F3=F3_M4, P=P, rho=True,
                                   sag_const={128: sag128, 64: sag64}[bm])["ms_sum"]
            base = d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads=reads, F3=F3_M4, P=P, rho=True, sag="none")
            own = {g: sag_curve[bm](base[g]["floor_only"]) for g in ("w1", "w2")}
            return d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads=reads, F3=F3_M4, P=P, rho=True, sag="own",
                               own_off_us=own, r3=d5.R3_1410)["ms_sum"]


def sagfit(bm, models):
    return VP.sagfit(bm, models)


def summ(res):
    """res: list of (row, e). Same layout as v5prime.summ."""
    return VP.summ(res)
