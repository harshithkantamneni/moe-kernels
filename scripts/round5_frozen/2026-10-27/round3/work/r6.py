"""ROUND 4 shared module (DECLARED4.txt), layered over r5 (-> r4 -> k5 -> d5: pricer built at p 14, T0 43.23 us and
bw 3597.8 GB/s asserted, the first-build trap; re-asserted in r5). Adds: W2m (the cut tested on the compulsory fill),
W2T (the measured tail map on later-wave same-wave launch gaps), and the CLOCK FORM of F1 (fac = 1 + D_net(t0) / t0 on
the floor side). Nothing in the repo or any earlier folder is written."""
from __future__ import annotations

import json
import math
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import r5
from r5 import r4, k5, d5, W, WV4

_TM = json.loads((Path(__file__).resolve().parent / "tailmap.json").read_text())
TAIL_U, TAIL_R = np.array(_TM["u"]), np.array(_TM["r"])
R3 = d5.R3_1410
r5.LAW3.setdefault("tail", False)
r5.LAW3.setdefault("cutmin", False)


def tail_factor(u):
    return np.interp(u, TAIL_U, TAIL_R)          # held at the end values outside [0.005, 0.995]


_R5_WALK = r5.walk_ls


def walk_ls4(arm, G, n, P, W_c, num_pid_m, experts, *, slab_bytes=0, atile_bytes=0, counts=None, block_m=0):
    """r5.walk_ls; with LAW3['tail'] every later-wave same-wave re-read's D_eff is multiplied by r(u), u its quantile
    among the cell's later-wave same-wave pairs within each replica ((rank - 0.5) / m)."""
    if not r5.LAW3.get("tail"):
        return _R5_WALK(arm, G, n, P, W_c, num_pid_m, experts, slab_bytes=slab_bytes, atile_bytes=atile_bytes,
                        counts=counts, block_m=block_m)
    # same walk as r5.walk_ls, the later-wave block rebuilt with the tail map
    t_owner, t_rows = W.tile_owners(counts, block_m)
    live_m = int(t_owner.size)
    pid = np.arange(num_pid_m * P, dtype=np.int64)
    gid, pid_m, pid_n = W.pid_map(pid, num_pid_m, P, G)
    keep = pid_m < live_m
    gid, pid_m, pid_n = gid[keep], pid_m[keep], pid_n[keep]
    live = int(pid_m.size)
    key = t_owner[pid_m] * P + pid_n
    cur, prev = W._previous(key)
    inside = gid[cur] == gid[prev]
    xs = ~inside
    ev = _R5_WALK(arm, G, n, P, W_c, num_pid_m, experts, slab_bytes=slab_bytes, atile_bytes=atile_bytes,
                  counts=counts, block_m=block_m)
    cx, px = cur[xs], prev[xs]
    kx, kpx = cx // W_c, px // W_c
    lat = kx >= 1
    if not lat.any():
        return ev
    S = (W.stack_distance(G, n, P, live_m, slab_bytes, atile_bytes, gid[cx], pid_n[cx], t_owner)
         if cx.size else np.zeros(0))
    st = r5.ls_starts(live, W_c, r5.LAW3["sigma"])
    dt = st[:, cx[lat]] - st[:, px[lat]]                               # (R, m)
    same = (kx[lat] == kpx[lat])
    if same.any():
        sub = dt[:, same]
        m = sub.shape[1]
        rank = np.argsort(np.argsort(sub, axis=1, kind="stable"), axis=1)
        sub = sub * tail_factor((rank + 0.5) / m)
        dt[:, same] = sub
    deff = np.round(W_c * dt * 8.0) / 8.0
    Sl = np.broadcast_to(S[lat], deff.shape)
    u, cnt = np.unique(np.stack([deff.ravel(), Sl.ravel()], axis=1), axis=0, return_counts=True)
    x = Counter()
    for d_, w_, s_, k_ in zip(ev.x_D.tolist(), ev.x_win1.tolist(), ev.x_S.tolist(), ev.x_k.tolist(), strict=True):
        if w_:
            x[(d_, True, s_)] += k_                      # the first window, unchanged
    for (dv, Sv), c in zip(u.tolist(), cnt.tolist(), strict=True):
        x[(dv, False, Sv)] += c / r5.R_REP
    keys = sorted(x)
    return W.Events(arm=ev.arm, G=ev.G, n=ev.n, gemm="", P=ev.P, W_c=ev.W_c, num_pid_m=ev.num_pid_m, live=ev.live,
                    slabs=ev.slabs, x_D=np.array([k_[0] for k_ in keys], dtype=float),
                    x_win1=np.array([k_[1] for k_ in keys], dtype=bool),
                    x_k=np.array([x[k_] for k_ in keys], dtype=float), x_S=np.array([k_[2] for k_ in keys], dtype=float),
                    in_win1=ev.in_win1, in_later=ev.in_later, a_D=ev.a_D, a_win1=ev.a_win1, a_ws=ev.a_ws, a_k=ev.a_k)


r5.walk_ls = walk_ls4
_ORIG_F = r5._ORIG_F


def F_cut4(self, prm, q):
    """r5.F_cut; with LAW3['cutmin'] the cut is tested on F_min = th D fbar(q_min) (q-independent), survival still at
    the full fill F(q)."""
    out = _ORIG_F(self, prm, q)
    cut = r5.LAW3.get("cut_mib")
    if cut is None or math.isinf(cut) or not self.x_D.size:
        return out
    q = np.asarray(q, dtype=float)
    two = q.ndim == 2
    per = self.Wg / self.live
    fbar = q * (per[:, None] if two else per)
    th = np.where(self.x_w1, prm.theta1, 1.0) * self.x_D
    F = th[:, None] * fbar[self.x_cell] if two else th * fbar[self.x_cell]
    later = ~self.x_w1
    if r5.LAW3.get("cutmin"):
        Fm = th * (self.q_min * per)[self.x_cell]
        hit = later & (Fm >= cut * W.MIB)
        mask = np.broadcast_to(hit[:, None], F.shape) if two else hit
    else:
        mask = (later[:, None] if two else later) & (F >= cut * W.MIB)
    surv = np.exp(-np.power(np.maximum(F, 0.0) / (prm.C_B * W.MIB), prm.beta_B))
    add = (self.x_k[:, None] if two else self.x_k) * surv * mask
    from wave_split_bytes import _segsum
    return out + _segsum(add, self.x_cell, self.count) / (self.S[:, None] if two else self.S)


W.Batch.F = F_cut4

LAWS4 = {"W2": {"tail": False, "cutmin": False}, "W2m": {"tail": False, "cutmin": True},
         "W2T": {"tail": True, "cutmin": False}, "W2mT": {"tail": True, "cutmin": True}}


@contextmanager
def law4(name=None):
    """name None: M4; else one of LAWS4 (all on W2's LS law at sigma_gap with the 60 MiB cut)."""
    if name is None:
        with r5.law3():
            yield
        return
    keep = {k: r5.LAW3.get(k) for k in ("tail", "cutmin")}
    with r5.law3(kind="LS", sigma=r5.SIGMA_GAP, cut_mib=r5.L2_MIB):
        r5.LAW3.update(LAWS4[name])
        WV4._MODELS.clear()
        k5._READS.clear()
        try:
            yield
        finally:
            r5.LAW3.update(keep)
    WV4._MODELS.clear()
    k5._READS.clear()


def reads_of(r, p, name):
    with law4(name):
        pr, _ = r4.m4_reads(r["model"], r["c"], r["doc"], p)
    return pr


# ------------------------------------------------------------------ the clock form of F1 (DECLARED4.txt PART B)
def v5_parts_clock(model, c, doc, prm, *, reads, F3, P, sag_const, rho=True, key="v3"):
    """d5.v5_parts with fac_g = 1 + D_net(t0) / t0, t0 the GEMM's sag-free time (price_parts' gemm at fac 1),
    D_net(t) = sat(A, tau, t) - sat(R3 share, t). 128x128 cells only."""
    pr = RG = d5.RG
    pr = RG.pricer()
    bm = int(doc["block_m"])
    assert bm == 128
    with d5.V.swapped(P[bm], d5.V.RHO_X[bm] if rho else None):
        base = d5.price_parts(model, "native", c, doc, prm, key=key, reads=reads)
        fac = {}
        A, tau = sag_const
        for g in ("w1", "w2"):
            t0 = base[g]["gemm"]
            D = d5.sat(A, tau, t0) - d5.sat(R3[0], R3[1], t0)
            fac[g] = 1 + D / t0
        with d5.c_patch(fac):
            out = d5.price_parts(model, "native", c, doc, prm, key=key, reads=reads)
        out["fac"] = fac
        return out


def cell_ms4(r, form, p128, sag128, reads, clock=True, sag64=(30.37, 300.0)):
    """the v5'' ms of one NATIVE cell with the knee form (k5.Form or None) at p128, the given reads, and F1 as the
    clock form (clock True) or v5'''s own (False). 64x128 cells: v5'' unchanged (k5.cell_ms)."""
    bm = int(r["doc"]["block_m"])
    if bm != 128:
        return k5.cell_ms(r, None, p128, sag128, sag64)
    prm = d5.params_for(d5.SEL, r["model"])
    with r4.A2.scoring(r4.RULES):
        P = {128: p128, 64: k5.P64}
        with k5.form_ctx(form):
            if clock:
                return v5_parts_clock(r["model"], r["c"], r["doc"], prm, reads=reads, F3=k5.F3_M4, P=P,
                                      sag_const=sag128)["ms_sum"]
            return d5.v5_parts(r["model"], r["c"], r["doc"], prm, reads=reads, F3=k5.F3_M4, P=P, rho=True,
                               sag_const={128: sag128, 64: sag64}[bm])["ms_sum"]
