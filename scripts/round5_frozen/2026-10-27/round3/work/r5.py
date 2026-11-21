"""ROUND 3 shared module (offline, read-only): the list-scheduling (LS) launch law and the capacity cut of
DECLARED3.txt PART A, as in-memory replacements layered over refit4's r4 (which imports knee-v5 k5 -> diag-v5 d5: the
CAL pricer is built at P_KNEE 14 and T0 43.23 us, bw 3597.8 GB/s are asserted, the first-build trap; re-asserted here).
Nothing in the repo or any earlier work folder is written (run with PYTHONDONTWRITEBYTECODE=1 from this folder).

LAW3 kinds: None (delegate to r4: M4 or r4's JITTER), "LS" (list scheduling at sigma), "STAG" (the staggered start
model start(r) = r / W_c through the same machinery: the reduction check against M4). LAW3["cut_mib"]: the capacity cut
(a later-wave slab x event, win1 False, with fill F >= cut is a certain miss), active under any kind.
"""
from __future__ import annotations

import dataclasses
import heapq
import math
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REFIT4 = Path.home() / "code/moe-kernels-work/2026-10-22/refit4/work"
sys.path.insert(0, str(REFIT4))
import r4  # noqa: E402
from r4 import k5, d5, W, WV4  # noqa: E402

TM, RG = d5.TM, d5.RG
assert TM.P_KNEE == 14.0
_X = [float(v) for v in RG.pricer().f2.x]
assert abs(_X[0] * 1e3 - 43.23) < 0.01 and abs(_X[2] - 3597.8) < 0.1, _X

SIGMA_GAP = 0.0375          # work/sigma_gap.json (MEASURED-PERTURBED, in-sample for Mixtral-8x7B)
THETA_C3 = 0.3513490841673576   # C3-R1, Mixtral-8x7B w2 (MEASURED-PERTURBED), descriptive laws D1 / D2 only
L2_MIB = 62914560 / 2 ** 20  # 60 MiB, page card
R_REP = 16
SEED = 20261027
LAW3 = {"kind": None, "sigma": SIGMA_GAP, "cut_mib": None}
_STARTS: dict = {}


def ls_starts(live: int, W_c: int, sigma: float) -> np.ndarray:
    """(R, live) start times in lifetimes: list scheduling onto W_c slots, wave 0 at t 0, iid N(1, sigma) lifetimes
    clipped at 0.05, replica i seeded SEED x 1000 + i (the same draws for every cell of equal live count)."""
    key = (live, W_c, round(float(sigma), 6))
    if key not in _STARTS:
        out = np.zeros((R_REP, live))
        for i in range(R_REP):
            rng = np.random.default_rng(SEED * 1000 + i)
            life = np.maximum(0.05, rng.normal(1.0, sigma, live))
            s = out[i]
            h = [float(life[r]) for r in range(min(W_c, live))]
            heapq.heapify(h)
            for r in range(W_c, live):
                t = heapq.heappop(h)
                s[r] = t
                heapq.heappush(h, t + float(life[r]))
        _STARTS[key] = out
    return _STARTS[key]


def stag_starts(live: int, W_c: int) -> np.ndarray:
    return np.tile(np.arange(live, dtype=float) / W_c, (R_REP, 1))


def walk_ls(arm, G, n, P, W_c, num_pid_m, experts, *, slab_bytes=0, atile_bytes=0, counts=None, block_m=0):
    """r4.walk_law's histogram branch with every later-wave cross-group slab re-read at D_eff = W_c (start(cur) -
    start(prev)) / L from the launch law (R replicas, weight 1 / R, rounded to 1/8 rank); the first window keeps M4's
    theta1 D; in-group and A events M4's."""
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
    kc = cur // W_c
    xs = ~inside
    xc = cur[xs]
    S = (W.stack_distance(G, n, P, live_m, slab_bytes, atile_bytes, gid[xc], pid_n[xc], t_owner)
         if xc.size else np.zeros(0))
    x = Counter()
    Dx, kx, cx, px = D[xs], kc[xs], cur[xs], prev[xs]
    first = kx == 0
    for d, Sv in zip(Dx[first].tolist(), S[first].tolist(), strict=True):
        x[(d, True, Sv)] += 1.0
    lat = ~first
    if lat.any():
        st = stag_starts(live, W_c) if LAW3["kind"] == "STAG" else ls_starts(live, W_c, LAW3["sigma"])
        deff = np.round(W_c * (st[:, cx[lat]] - st[:, px[lat]]) * 8.0) / 8.0     # (R, m)
        Sl = np.broadcast_to(S[lat], deff.shape)
        kk = np.stack([deff.ravel(), Sl.ravel()], axis=1)
        u, cnt = np.unique(kk, axis=0, return_counts=True)
        for (dv, Sv), c in zip(u.tolist(), cnt.tolist(), strict=True):
            x[(dv, False, Sv)] += c / R_REP
    ins = np.nonzero(inside)[0]
    in_win1 = float(np.sum(kc[ins] == 0))
    in_later = float(np.sum(kc[ins] != 0))
    groups = -(-num_pid_m // G)
    ws = np.zeros(groups)
    for g in range(groups):
        f0 = g * G
        tiles = range(f0, min(f0 + G, live_m)) if f0 < live_m else range(0)
        owners = len({int(t_owner[m]) for m in tiles})
        ws[g] = len(tiles) * atile_bytes + owners * slab_bytes
    cur_a, prev_a = W._previous(pid_m)
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


_R4_CELL_EVENTS = r4.cell_events_law


def cell_events_ls(geom, arm, G, n, gemm, counts=None):
    if counts is None or LAW3["kind"] is None or geom.block_m != 128 or arm == "private":
        return _R4_CELL_EVENTS(geom, arm, G, n, gemm, counts)
    P = geom.get("P", gemm)
    Wc = geom.get("W_c", gemm)
    npm = geom.num_pid_m(arm, n)
    ev = walk_ls(arm, G, n, P, Wc, npm, geom.experts, slab_bytes=geom.slab(gemm), atile_bytes=geom.atile(gemm),
                 counts=counts, block_m=geom.block_m)
    if W.LATER_MISS and G >= 2 and ev.x_D.size:
        later = ~ev.x_win1 & (ev.x_S > geom.l2_bytes)
        ev = dataclasses.replace(ev, slabs=ev.slabs + float(ev.x_k[later].sum()), x_D=ev.x_D[~later],
                                 x_win1=ev.x_win1[~later], x_k=ev.x_k[~later], x_S=ev.x_S[~later])
    return dataclasses.replace(ev, gemm=gemm, per_set=geom.per_set(gemm), weight_bytes=geom.get("W", gemm),
                               atile=geom.atile(gemm), ksteps=geom.get("K", gemm) // max(1, geom.block_k))


W.cell_events = cell_events_ls

_ORIG_F = W.Batch.F


def F_cut(self, prm, q):
    """Batch.F plus the capacity cut: a later-wave (win1 False) slab x event with fill F >= cut misses for certain
    (its carried survival is added back as a miss). Identity when no cut is set."""
    out = _ORIG_F(self, prm, q)
    cut = LAW3.get("cut_mib")
    if cut is None or math.isinf(cut) or not self.x_D.size:
        return out
    q = np.asarray(q, dtype=float)
    two = q.ndim == 2
    per = self.Wg / self.live
    fbar = q * (per[:, None] if two else per)
    th = np.where(self.x_w1, prm.theta1, 1.0) * self.x_D
    f = fbar[self.x_cell]
    F = th[:, None] * f if two else th * f
    later = ~self.x_w1
    mask = (later[:, None] if two else later) & (F >= cut * W.MIB)
    surv = np.exp(-np.power(np.maximum(F, 0.0) / (prm.C_B * W.MIB), prm.beta_B))
    add = (self.x_k[:, None] if two else self.x_k) * surv * mask
    from wave_split_bytes import _segsum
    return out + _segsum(add, self.x_cell, self.count) / (self.S[:, None] if two else self.S)


W.Batch.F = F_cut


@contextmanager
def law3(kind=None, sigma=SIGMA_GAP, cut_mib=None, r4kind="M4", theta=1.0):
    keep, keep4 = dict(LAW3), dict(r4.LAW)
    LAW3.update(kind=kind, sigma=float(sigma), cut_mib=cut_mib)
    r4.LAW.update(kind=r4kind, k0=0.0, theta=float(theta), step_mib=None)
    WV4._MODELS.clear()
    k5._READS.clear()
    try:
        yield
    finally:
        LAW3.clear(); LAW3.update(keep)
        r4.LAW.clear(); r4.LAW.update(keep4)
        WV4._MODELS.clear()
        k5._READS.clear()


# the declared candidates: name -> list of (constant, law3 kwargs)
SIG_GRID = [0.0125, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
CUT_GRID = [30, 40, 50, 60, 70, 80, 100, 120, math.inf]


def law_kwargs(name, const=None):
    if name == "W0 M4":
        return {}
    if name == "W1 LS-SOFT":
        return {"kind": "LS", "sigma": SIGMA_GAP, "cut_mib": None}
    if name == "W2 LS-CUT":
        return {"kind": "LS", "sigma": SIGMA_GAP, "cut_mib": L2_MIB}
    if name == "W3 LS-CUT-s":
        return {"kind": "LS", "sigma": const, "cut_mib": L2_MIB}
    if name == "W4 LS-CUT-C":
        return {"kind": "LS", "sigma": SIGMA_GAP, "cut_mib": None if math.isinf(const) else float(const)}
    if name == "D1 C5@0.351 soft":
        return {"kind": None, "r4kind": "JITTER", "theta": THETA_C3}
    if name == "D2 C5@0.351 cut":
        return {"kind": None, "r4kind": "JITTER", "theta": THETA_C3, "cut_mib": L2_MIB}
    raise KeyError(name)
