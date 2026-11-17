"""refit4 shared module (offline, read-only): the four-model refit of v5'' after rental C2.

Imports knee-v5/work/k5.py (which imports diag-v5/work/d5.py: the CAL pricer is built at import at P_KNEE 14 and the
carried fit is asserted, T0 43.23 us and bw 3597.8 GB/s, the first-build trap; re-asserted here). Nothing in the repo
or any earlier work folder is written (run with PYTHONDONTWRITEBYTECODE=1).

The w2 byte law candidates (DECLARED.txt A) are in-memory replacements of wave_split_bytes.walk and .cell_events, active
only at BLOCK_M 128 on histogram walks (every 128x128 cell M4 prices goes through f3_reads with counts) and only while
LAW is not M4. With k0 = 0 they reproduce M4 bit for bit (checked in bytes4.py).
"""
from __future__ import annotations

import dataclasses
import json
import math
import statistics as st
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
KNEE = Path.home() / "code/moe-kernels-work/2026-10-22/knee-v5/work"
sys.path.insert(0, str(KNEE))
import k5  # noqa: E402
from k5 import d5  # noqa: E402

TM, RG, A2, RULES = d5.TM, d5.RG, d5.A2, d5.RULES
assert TM.P_KNEE == 14.0
_X = [float(v) for v in RG.pricer().f2.x]
assert abs(_X[0] * 1e3 - 43.23) < 0.01 and abs(_X[2] - 3597.8) < 0.1, _X
import wave_split_bytes as W  # noqa: E402
import wave_split_bytes_v3 as WV3  # noqa: E402
from common import WV4  # noqa: E402  (diag-A2 common: wave_split_bytes_v4)

REPO = Path.home() / "code/moe-kernels"
A2T = d5.A2T
S1T = d5.S1T
C2T = Path.home() / "moe-kernels-exfil/runs/gh200-tile2-20261021T1842Z/tree/vm/results"
LOCK = 1410
MODELS4 = ("olmoe-1b-7b", "qwen2-57b-a14b", "mixtral-8x7b", "mixtral-8x22b")
SHORT = {"olmoe-1b-7b": "OLMoE", "qwen2-57b-a14b": "Qwen2-57B", "mixtral-8x7b": "Mixtral-8x7B", "mixtral-8x22b": "Mixtral-8x22B"}
E_FAMILY = {"olmoe-1b-7b": 64, "qwen2-57b-a14b": 64, "mixtral-8x7b": 8, "mixtral-8x22b": 8}

#: every 128x128 counter page at 1410 (root, model, unit, board)
PAGES128 = [
    (A2T, "olmoe-1b-7b", "b128g1", "A2"), (S1T, "olmoe-1b-7b", "st", "S1"),
    (A2T, "qwen2-57b-a14b", "b128g1", "A2"), (S1T, "qwen2-57b-a14b", "st", "S1"),
    (S1T, "mixtral-8x7b", "b128g1", "S1"), (C2T, "mixtral-8x7b", "b128g1", "C2"),
    (C2T, "mixtral-8x22b", "b128g1", "C2"),
    (A2T, "mixtral-8x7b", "b128g16", "A2"), (S1T, "mixtral-8x7b", "p128g16", "S1"),
    (A2T, "olmoe-1b-7b", "f128", "A2"), (A2T, "mixtral-8x7b", "f128", "A2"), (S1T, "olmoe-1b-7b", "f1410", "S1"),
    (C2T, "olmoe-1b-7b", "f128", "C2"), (C2T, "mixtral-8x7b", "f128", "C2"),
]

# ------------------------------------------------------------------ the byte law (DECLARED.txt A)
LAW = {"kind": "M4", "k0": 0.0, "theta": 1.0}
_ORIG_WALK = W.walk
_ORIG_CELL_EVENTS = W.cell_events


def share(k: np.ndarray) -> np.ndarray:
    """the share of a later-wave (k >= 1) re-read that follows the law's synchronized-wave rule (1 = always)."""
    k = np.asarray(k, dtype=float)
    if LAW["kind"] in ("WAVE", "SYNC"):
        return np.ones_like(k)
    k0 = LAW["k0"]
    if k0 <= 0:
        return np.zeros_like(k)
    if math.isinf(k0):
        return np.ones_like(k)
    return np.exp(-k / k0)


def walk_law(arm, G, n, P, W_c, num_pid_m, experts, *, slab_bytes=0, atile_bytes=0, counts=None, block_m=0):
    if counts is None or block_m != 128 or LAW["kind"] == "M4" or arm == "private":
        return _ORIG_WALK(arm, G, n, P, W_c, num_pid_m, experts, slab_bytes=slab_bytes, atile_bytes=atile_bytes,
                          counts=counts, block_m=block_m)
    sync = LAW["kind"] in ("SYNC", "SYNCDECAY")
    jit = LAW["kind"] == "JITTER"
    # the histogram branch of wave_split_bytes.walk; slab events split by wave (DECLARED.txt A and AM1)
    t_owner, t_rows = W.tile_owners(counts, block_m)
    live_m = int(t_owner.size)
    if num_pid_m < live_m:
        raise ValueError("num_pid_m below live tiles")
    pid = np.arange(num_pid_m * P, dtype=np.int64)
    gid, pid_m, pid_n = W.pid_map(pid, num_pid_m, P, G)
    keep = pid_m < live_m
    gid, pid_m, pid_n = gid[keep], pid_m[keep], pid_n[keep]
    live = int(pid_m.size)
    owner = t_owner[pid_m]
    key = owner * P + pid_n
    cur, prev = W._previous(key)
    slabs = live - int(cur.size)
    D = cur - prev
    inside = gid[cur] == gid[prev]
    kc, kp = cur // W_c, prev // W_c
    same = kc == kp
    first = kc == 0                       # M4's first window (then prev is in it too)
    if jit:
        sh = np.zeros_like(kc, dtype=float)
    elif sync:
        sh = np.where(first, 1.0, share(kc))
    else:
        sh = np.where(first, 1.0, np.where(same, share(kc), 0.0))
    dk = (kc - kp).astype(float)
    xs = ~inside
    xc = cur[xs]
    S = (W.stack_distance(G, n, P, live_m, slab_bytes, atile_bytes, gid[xc], pid_n[xc], t_owner)
         if xc.size else np.zeros(0))
    x = Counter()
    if jit:   # AM2 C5: D_eff = dk W_c (1 - theta_L) + theta_L D for a reader in wave k >= 1; first window M4's
        th = LAW["theta"]
        for d, kcv, dkv, Sv in zip(D[xs].tolist(), kc[xs].tolist(), dk[xs].tolist(), S.tolist(), strict=True):
            if kcv == 0:
                x[(d, True, Sv)] += 1.0
            else:
                x[(dkv * W_c * (1.0 - th) + th * d, False, Sv)] += 1.0
        ins_ = np.nonzero(inside)[0]
        in_win1 = float(np.sum(kc[ins_] == 0))
        in_later = float(np.sum(kc[ins_] != 0))
    for d, sv, sm, dkv, Sv in ([] if jit else zip(D[xs].tolist(), sh[xs].tolist(), same[xs].tolist(), dk[xs].tolist(), S.tolist(), strict=True)):
        if sv > 0:
            if sm:
                x[(d, True, Sv)] += sv                       # same wave: a first-window event
            else:                                            # SYNC only (sh is 0 off the same wave otherwise)
                x[(dkv * W_c, False, Sv)] += sv              # cross-wave: (k_cur - k_prev) waves of fill
        if sv < 1:
            x[(d, False, Sv)] += 1.0 - sv                    # M4's staggered event
    ins = np.nonzero(inside)[0]
    if not jit:
        in_win1 = float(np.sum(np.where(same[ins], sh[ins], 0.0)))
        in_later = float(np.sum(1.0 - sh[ins]))
    for sv, dkv in zip(sh[ins][~same[ins]].tolist(), dk[ins][~same[ins]].tolist(), strict=True):
        if sv > 0:
            x[(dkv * W_c, False, 0.0)] += sv                 # SYNC: an in-group cross-wave re-read at that distance
    groups = -(-num_pid_m // G)
    ws = np.zeros(groups)
    for g in range(groups):
        first = g * G
        tiles = range(first, min(first + G, live_m)) if first < live_m else range(0)
        owners = len({int(t_owner[m]) for m in tiles})
        ws[g] = len(tiles) * atile_bytes + owners * slab_bytes
    cur_a, prev_a = W._previous(pid_m)       # A events: M4's law (first window only), unchanged
    da = cur_a - prev_a
    wa = cur_a < W_c
    a = Counter()
    frac = (t_rows[pid_m[cur_a]] / block_m).tolist()
    for key_a, f in zip(zip(da.tolist(), wa.tolist(), ws[gid[cur_a]].tolist(), strict=True), frac, strict=True):
        a[key_a] += f

    def arrays(counter, width):
        if not counter:
            return [np.zeros(0) for _ in range(width + 1)]
        keys = sorted(counter)
        cols = [np.array([k_[i] for k_ in keys], dtype=float) for i in range(width)]
        return cols + [np.array([counter[k_] for k_ in keys], dtype=float)]

    x_D, x_w, x_S, x_k = arrays(x, 3)
    a_D, a_w, a_ws, a_k = arrays(a, 3)
    return W.Events(arm=arm, G=G, n=n, gemm="", P=P, W_c=W_c, num_pid_m=num_pid_m, live=live, slabs=slabs,
                    x_D=x_D, x_win1=x_w.astype(bool), x_k=x_k, x_S=x_S, in_win1=in_win1, in_later=in_later,
                    a_D=a_D, a_win1=a_w.astype(bool), a_ws=a_ws, a_k=a_k)


def cell_events_law(geom, arm, G, n, gemm, counts=None):
    if counts is None or LAW["kind"] == "M4" or geom.block_m != 128:
        return _ORIG_CELL_EVENTS(geom, arm, G, n, gemm, counts)
    P = geom.get("P", gemm)
    Wc = geom.get("W_c", gemm)
    npm = geom.num_pid_m(arm, n)
    ev = walk_law(arm, G, n, P, Wc, npm, geom.experts, slab_bytes=geom.slab(gemm), atile_bytes=geom.atile(gemm),
                  counts=counts, block_m=geom.block_m)
    if W.LATER_MISS and G >= 2 and ev.x_D.size:
        later = ~ev.x_win1 & (ev.x_S > geom.l2_bytes)
        ev = dataclasses.replace(ev, slabs=ev.slabs + float(ev.x_k[later].sum()), x_D=ev.x_D[~later],
                                 x_win1=ev.x_win1[~later], x_k=ev.x_k[~later], x_S=ev.x_S[~later])
    return dataclasses.replace(ev, gemm=gemm, per_set=geom.per_set(gemm), weight_bytes=geom.get("W", gemm),
                               atile=geom.atile(gemm), ksteps=geom.get("K", gemm) // max(1, geom.block_k))


W.walk = walk_law
W.cell_events = cell_events_law


@contextmanager
def law(kind="M4", k0=0.0, theta=1.0):
    keep = dict(LAW)
    LAW.update(kind=kind, k0=float(k0), theta=float(theta))
    WV4._MODELS.clear()
    k5._READS.clear()
    try:
        yield
    finally:
        LAW.clear()
        LAW.update(keep)
        WV4._MODELS.clear()
        k5._READS.clear()


# ------------------------------------------------------------------ predicted reads at 128x128
def m4_reads(model, c, doc, p128=5.75):
    """M4's (or the active law's) v3-key reads {w1, w2} and q for one cell (c: n, counts, label)."""
    prm = d5.params_for(d5.SEL, model)
    bm = int(doc["block_m"])
    with A2.scoring(RULES):
        with d5.V.swapped(p128 if bm == 128 else k5.P64, d5.V.RHO_X[bm]):
            reads, q = d5.V.f3_reads(RG.pricer(), model, int(c["n"]), c["counts"], RG.seed_of(doc, c), RG.PAGE_TILE[bm],
                                     int(doc["group_m"]), prm, k5.F3_M4[bm], "v3")
    return reads, q


def counter_cells(root, model, unit):
    """NATIVE 128x128 cells of one counter page (kneefit.cells_of), with counted reads and cycles."""
    return [r for r in k5.KF.cells_of(Path(root), model, unit, LOCK) if r["arm"] == "native" and int(r["doc"]["block_m"]) == 128]


# ------------------------------------------------------------------ time-domain pricing with a reads override
def cell_ms(r, p128, sag128, reads=None, sag64=None):
    """v5'' ms of one NATIVE cell (k5.cell_ms, null form) at p128 and F1 (A, tau) = sag128, with `reads` (dict w1,w2)
    replacing the active law's reads when given."""
    prm = d5.params_for(d5.SEL, r["model"])
    bm = int(r["doc"]["block_m"])
    if reads is None:
        reads, _ = m4_reads(r["model"], r["c"], r["doc"], p128)
    with A2.scoring(RULES):
        P = {128: p128, 64: k5.P64}
        return d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads=reads, F3=k5.F3_M4, P=P, rho=True,
                           sag_const={128: sag128, 64: sag64}[bm])["ms_sum"]


def a2_rows():
    """A2's E1-CAL NATIVE rows, drops removed (the registered set)."""
    return k5.e1_rows()


def c2_rows(model, label, reg_key, hist_rel):
    """C2's timed NATIVE rows (read_c2.timed_rows' logic, the ALL view, A2's selections, row_drop vs v4)."""
    R = json.loads((REPO / "docs/registered/2026-10-22-cal2-gh200.json").read_text())
    reg_file = R["pages"]["files"][reg_key]
    doc = json.loads((REPO / hist_rel).read_text())
    TC = d5.TC
    prm = d5.params_for(d5.SEL, model)
    with A2.scoring(RULES):
        rep, why = TC.timed_page(C2T, model, label, int(reg_file["group_m"]), TC.C3.TimedView("ALL"), reg_file)
        assert rep is not None, why
        assert int(rep["lock_mhz"]) == LOCK
        tt = {(int(r["tiles"]), str(r["histogram"]), str(r["arm"])): r for r in rep["treads_table"]}
        rows = []
        for c in doc["cells"]:
            if "native" not in c["arms"]:
                continue
            k = (int(c["n"]), c["label"], "native")
            tr = tt.get(k)
            base = RG.price_cell(model, "native", c, doc, prm, key=d5.SEL["key"])["ms"]
            drop = TC.row_drop(tr, base) if tr else "no row"
            rows.append({"model": model, "page": doc["page"], "cls": TC.tile_class(int(doc["block_m"]), int(doc["group_m"])),
                         "n": k[0], "label": k[1], "meas": float(tr["ms_p50"]) if tr else None, "drop": drop, "c": c, "doc": doc})
    return [r for r in rows if not r["drop"] and r["meas"]]


def c2_8x22b_rows():
    return c2_rows("mixtral-8x22b", "t128g1", "mixtral-8x22b-128g1.json", "docs/registered/2026-10-22-cal2-hist/mixtral-8x22b-128g1.json")


def c2_8x7b_rows():
    return c2_rows("mixtral-8x7b", "t128g1m", "mixtral-8x7b-128g1.json", "docs/registered/2026-10-14-tileA-hist/mixtral-8x7b-128g1.json")


# ------------------------------------------------------------------ sag estimator pages
EST_UNIT = {"olmoe-1b-7b": "f128", "mixtral-8x7b": "f128"}


def sag_page_fit(root, model):
    f = sorted(Path(root).glob(f"*-{model}-{EST_UNIT[model]}-r3-counters/lock{LOCK}/r3c-g*.json"))
    assert len(f) == 1, (root, model, f)
    r = k5.S.fit(k5.S.pts_of(f, float(LOCK)))
    return (r["A_us"], r["tau_us"])


def par(fits):
    return (st.fmean(A for A, _ in fits), st.fmean(t for _, t in fits))


def pct(x):
    return round(100 * x, 2)


def summ(e):
    e = [100 * x for x in e]
    if not e:
        return {"n": 0}
    return {"n": len(e), "mean": round(st.fmean(e), 2), "rms": round(math.sqrt(st.fmean(x * x for x in e)), 2),
            "worst": round(max(e, key=abs), 2)}


# ------------------------------------------------------------------ ROUND 2 (DECLARED2.txt R2-B): slab survival
_ORIG_SIGMA = W.sigma
_CB = float(k5.F3_M4[128]["w2"].C_B)
_BB = float(k5.F3_M4[128]["w2"].beta_B)
assert _CB == float(k5.F3_M4[128]["w1"].C_B) and _BB == float(k5.F3_M4[128]["w1"].beta_B)


def sigma_law(F, C, beta):
    """slab x events (called with M4's C_B, beta_B) take a step at LAW['step_mib'] when it is set; all else M4's."""
    st_ = LAW.get("step_mib")
    if st_ and np.ndim(C) == 0 and float(C) == _CB * W.MIB and np.ndim(beta) == 0 and float(beta) == _BB:
        return np.where(np.asarray(F) < st_ * W.MIB, 1.0, 0.0)
    return _ORIG_SIGMA(F, C, beta)


W.sigma = sigma_law


@contextmanager
def law2(theta=1.0, step_mib=None, kind="JITTER"):
    keep = dict(LAW)
    LAW.update(kind=kind, k0=0.0, theta=float(theta), step_mib=step_mib)
    WV4._MODELS.clear()
    k5._READS.clear()
    try:
        yield
    finally:
        LAW.clear()
        LAW.update(keep)
        WV4._MODELS.clear()
        k5._READS.clear()
