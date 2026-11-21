"""ROUND 5 shared module (DECLARED5.txt), layered over round 4's r6 (-> r5 -> r4 -> k5 -> d5: the CAL pricer is built
at p 14 and T0 43.23 us and bw 3597.8 GB/s are asserted, the first-build trap; re-asserted here). Adds the declared w1
byte laws at 128x128 (w1 slab x events only; w2 stays W2 in every candidate, in-group and A events unchanged), the C4
board's pages (tp2's counter and timed cells, C4's estimator pages), and the five-model pass estimate.
Nothing in the repo or any earlier work folder is written (run with PYTHONDONTWRITEBYTECODE=1 from this folder)."""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROUND4 = Path.home() / "code/moe-kernels-work/2026-10-27/round3/work"
sys.path.insert(0, str(ROUND4))
import r6  # noqa: E402
from r6 import r5, r4, k5, d5, W, WV4  # noqa: E402

assert d5.TM.P_KNEE == 14.0
_X = [float(v) for v in d5.RG.pricer().f2.x]
assert abs(_X[0] * 1e3 - 43.23) < 0.01 and abs(_X[2] - 3597.8) < 0.1, _X
assert r5.SIGMA_GAP == 0.0375 and r5.L2_MIB == 60.0

REPO = r4.REPO
C4T = Path.home() / "moe-kernels-exfil/runs/gh200-tile4-20261027T2238Z/tree/vm/results"
TP2 = "mixtral-8x7b-tp2"
O, Q, M, X22 = r4.MODELS4
MODELS5 = (O, Q, M, X22, TP2)
SHORT = {**r4.SHORT, TP2: "tp2"}
E_FAMILY = {**r4.E_FAMILY, TP2: 8}
ROUTER = {O: O, Q: Q, M: M, X22: X22, TP2: M}          # tp2 is a TP shard of Mixtral-8x7B: same router
L2 = r5.L2_MIB                                          # 60 MiB, the page card (MEASURED)
C_GRID = [30 + 5 * i for i in range(19)]                # STAG-RR-C: 30 .. 120 MiB

# ------------------------------------------------------------------ the declared w1 laws (DECLARED5.txt PART A)
#: name -> (walk, survival, cut, constants); walk STAG = M4's rank distance, LS = W2's launch-law D_eff
W1_LAWS = {
    "W2": None,                                                   # round 4's law on w1 (r6.law4('W2') as is)
    "M4w1": {"walk": "STAG", "surv": "SOFT", "cut": False, "nconst": 0},
    "STAG-CUT": {"walk": "STAG", "surv": "SOFT", "cut": True, "nconst": 0},
    "STAG-RR": {"walk": "STAG", "surv": "RR", "cut": False, "C": L2, "nconst": 0},
    "LS-RR": {"walk": "LS", "surv": "RR", "cut": False, "C": L2, "nconst": 0},
    "STAG-RR-C": {"walk": "STAG", "surv": "RR", "cut": False, "C": None, "nconst": 1},
}
LAW5 = {"w1": None}

_R5_CE = r5.cell_events_ls
assert W.cell_events is _R5_CE


def cell_events5(geom, arm, G, n, gemm, counts=None):
    L = LAW5["w1"]
    if L is None or gemm != "w1" or counts is None or geom.block_m != 128 or arm == "private":
        return _R5_CE(geom, arm, G, n, gemm, counts)
    if L["walk"] == "STAG":
        assert r4.LAW["kind"] == "M4"
        return r4._ORIG_CELL_EVENTS(geom, arm, G, n, gemm, counts)      # M4's walk (W.walk -> r4.walk_law at M4)
    keep = r5.LAW3["kind"]
    r5.LAW3["kind"] = "LS"
    try:
        return _R5_CE(geom, arm, G, n, gemm, counts)
    finally:
        r5.LAW3["kind"] = keep


W.cell_events = cell_events5
_F4 = W.Batch.F
assert _F4 is r6.F_cut4


def F5(self, prm, q):
    """The w1 law on w1 slab x events (survival replaced; first window at theta1 D, later at D); w2 events keep W2's
    soft survival and 60 MiB cut exactly as r6.F_cut4. Identity (r6.F_cut4) when no w1 law is set."""
    L = LAW5["w1"]
    if L is None or not self.x_D.size:
        return _F4(self, prm, q)
    assert not r5.LAW3.get("tail") and not r5.LAW3.get("cutmin")
    out = r5._ORIG_F(self, prm, q)
    q = np.asarray(q, dtype=float)
    two = q.ndim == 2

    def bc(a):
        return a[:, None] if two else a

    per = self.Wg / self.live
    fbar = q * (per[:, None] if two else per)
    th = np.where(self.x_w1, prm.theta1, 1.0) * self.x_D
    F = th[:, None] * fbar[self.x_cell] if two else th * fbar[self.x_cell]
    Fp = np.maximum(F, 0.0)
    w1e = self.is_w1[self.x_cell]
    later = ~self.x_w1
    soft = np.exp(-np.power(Fp / (prm.C_B * W.MIB), prm.beta_B))
    k = bc(self.x_k)
    add = np.zeros_like(F)
    cut = r5.LAW3.get("cut_mib")
    if cut is not None and not math.isinf(cut):
        add = add + k * soft * (bc(later & ~w1e) & (F >= cut * W.MIB))
    if L["surv"] == "RR":
        surv = np.exp(-Fp / (float(L["C"]) * W.MIB))
    else:
        surv = soft
    if L["cut"]:
        surv = np.where(bc(later) & (F >= L2 * W.MIB), 0.0, surv)
    add = add + k * (soft - surv) * bc(w1e)
    from wave_split_bytes import _segsum
    return out + _segsum(add, self.x_cell, self.count) / (self.S[:, None] if two else self.S)


W.Batch.F = F5


@contextmanager
def law5(name, C=None):
    """W2 on w2 (r6.law4('W2')) and the named w1 law (C: STAG-RR-C's capacity, MiB)."""
    spec = W1_LAWS[name]
    if spec is not None:
        spec = dict(spec)
        if spec["surv"] == "RR" and spec["C"] is None:
            assert C is not None
            spec["C"] = float(C)
    keep = LAW5["w1"]
    with r6.law4("W2"):
        LAW5["w1"] = spec
        WV4._MODELS.clear()
        k5._READS.clear()
        try:
            yield
        finally:
            LAW5["w1"] = keep
            WV4._MODELS.clear()
            k5._READS.clear()


def reads5(r, p, name, C=None):
    with law5(name, C):
        pr, _ = r4.m4_reads(r["model"], r["c"], r["doc"], p)
    return pr


def reads_m4(r, p):
    with r6.law4(None):
        pr, _ = r4.m4_reads(r["model"], r["c"], r["doc"], p)
    return pr


# ------------------------------------------------------------------ cells
def counter_cells_g1():
    """every NATIVE 128 G 1 counter cell of the five CAL models: refit4 bytes4.load_cells' four-model pages and C4's
    tp2 b128g1 page (kneefit.cells_of via r4.counter_cells)."""
    import os
    h = os.getcwd(); os.chdir(r5.REFIT4)
    import bytes4 as B4
    os.chdir(h)
    cells = [x for x in B4.load_cells() if x["G"] == 1]
    for r in r4.counter_cells(C4T, TP2, "b128g1"):
        cells.append({"model": TP2, "unit": "b128g1", "board": "C4", "G": int(r["doc"]["group_m"]), "n": r["c"]["n"],
                      "label": r["c"]["label"], "c": r["c"], "doc": r["doc"], "counted": r["reads"]})
    assert all(x["G"] == 1 for x in cells) and len(cells) == 60, len(cells)
    return cells


def c4_timed_rows():
    """tp2's t128g1 NATIVE rows as read_c4.timed_rows builds them (tcommon.timed_page, the ALL view, A2's selections,
    row_drop against v4), the registered page entry."""
    R = json.loads((REPO / "docs/registered/2026-10-27-cal4-gh200.json").read_text())
    reg_file = R["pages"]["files"]["mixtral-8x7b-tp2-128g1.json"]
    doc = json.loads((REPO / "docs/registered/2026-10-27-cal4-hist/mixtral-8x7b-tp2-128g1.json").read_text())
    TC, RG = d5.TC, d5.RG
    prm = d5.params_for(d5.SEL, TP2)
    with d5.A2.scoring(d5.RULES):
        rep, why = TC.timed_page(C4T, TP2, "t128g1", int(reg_file["group_m"]), TC.C3.TimedView("ALL"), reg_file)
        assert rep is not None, why
        assert int(rep["lock_mhz"]) == 1410
        tt = {(int(r["tiles"]), str(r["histogram"]), str(r["arm"])): r for r in rep["treads_table"]}
        rows = []
        for c in doc["cells"]:
            if "native" not in c["arms"]:
                continue
            k = (int(c["n"]), c["label"], "native")
            tr = tt.get(k)
            base = RG.price_cell(TP2, "native", c, doc, prm, key=d5.SEL["key"])["ms"]
            drop = TC.row_drop(tr, base) if tr else "no row"
            rows.append({"model": TP2, "page": doc["page"], "cls": TC.tile_class(int(doc["block_m"]), int(doc["group_m"])),
                         "n": k[0], "label": k[1], "meas": float(tr["ms_p50"]) if tr else None, "drop": drop, "c": c,
                         "doc": doc, "board": "C4"})
    return [r for r in rows if not r["drop"] and r["meas"]]


def c4_sag_par():
    per = {m: r4.sag_page_fit(C4T, m) for m in (O, M)}
    return per, r4.par(list(per.values()))


def c4_twins():
    return {(TP2, "C4", c["c"]["n"], c["c"]["label"]): c["reads"] for c in r4.counter_cells(C4T, TP2, "b128g1")}


# ------------------------------------------------------------------ the five-model pass estimate (read_c4's method)
def _rc():
    import os
    h = os.getcwd(); os.chdir(r5.REFIT4)
    import refit4 as RF
    os.chdir(h)
    return RF


def pass_estimates5(means: dict, cells, R=200_000) -> dict:
    RC = _rc().RC
    to_b = RC.to_b
    b = [to_b(means[m]) for m in (O, Q, M, X22, TP2)]
    A = RC.power(cells, [st.fmean(b)] * 3, st.stdev(b), R=R)["E1-TILE"]
    Bp = RC.power(cells, [0.0] * 3, math.sqrt(st.fmean(x * x for x in b)), R=R)["E1-TILE"]
    four = [to_b(means[m]) for m in (O, Q, X22, TP2)]
    D = RC.power(cells, [st.fmean(four)] * 3, st.stdev(four), R=R)["E1-TILE"]
    return {"A (5 models)": A, "B (5 models)": Bp, "D (O, Q, 8x22B, tp2; 8x7B G1 aside)": D}
