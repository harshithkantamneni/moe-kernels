#!/usr/bin/env python
"""The decode stage's sixth addendum (docs/registered/2026-11-10-decode-addendum6-gh200): model v6d, its gate and the
new blind (addendum 5's pages, units and bars, reused), in one module for the registration builder, the gate, the
finalisation and the blind scorer. It imports the frozen decode modules (dcommon, d2common, d3common, d4common,
d5common, finalise_decode) and v6d (r3_timing_model_v6d, over v5d over v4d') and changes none of them: every rule here
acts by swapping v6d in where the frozen code prices a cell, or by computing v6d's constants from the readings.

THE RULES THIS FILE OWNS:
  v6d        `v6d(terms, law)`: d3common.price_cell_v4dp (the one function every addendum-3 hook prices a page cell
             with) priced by r3_timing_model_v6d.price at `terms` and `law`; law None is v5d at `terms`
  constants  `law_from(pc, pc2, spans)`: the law's numbers from the committed readings, mechanically (no choice after
             the readings): c0 at G 8 (PC2-R1b, PC2-R2a) and G 128 (PC-R2), the L table (PC-R7 and PC-R1 with c0(G)
             removed), X_sat (PC-R7, PC2-R1d, PC-R7 x DS's W1RATE ratio), kappa by fill (PC's spans, PC2-R3), n_if
             (PC-R0, PC2-R0). A reading that is not READ is BLOCKING (`blocking`), never a fallback
  terms      v5d's registered terms (addendum 5) with the regimes and the W1 scale emptied (r3_timing_model_v6d.v6_terms)
  gate       the owner's two clauses as addendum 5 coded them: CAL2 under v6d reads GO by the addendum-3 stop rule;
             the SEEN decode blind re-scored under v6d reads a pooled rms under 4.0% over the committed 220 cells (a
             WEAK SANITY FLOOR on seen pages, never a test); plus READINGS and TREES
  blind      addendum 5's new blind (d5common: models, pages, units, bars, plan form), reused unchanged
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d3common as D3  # noqa: E402
import d4common as D4  # noqa: E402
import d5common as D5  # noqa: E402
import dcommon as DC  # noqa: E402

RD, TM = DC.RD, DC.V4.TM
REPO = DC.REPO
V4DP, V5D = D3.V4DP, D5.V5D
import r3_timing_model_v6d as V6D  # noqa: E402

r6 = DC.r6
DATE = "2026-11-10"
NAME = f"{DATE}-decode-addendum6-gh200"
PC_READ = "scripts/scoring/pc/pc.read.json"
PC2_READ = "scripts/scoring/pc2/pc2.read.json"
SPANS = "scripts/scoring/decode/d6_build/pc_spans.json"
BLIND6_PLAN = "scripts/plans/rentalD6-blind-2026-11.plan"
NEW_BLIND, NEW_CTL, PAGES5, PAGE_KEYS = D5.NEW_BLIND, D5.NEW_CTL, D5.PAGES5, D5.PAGE_KEYS
BARS_OWNER, RESCORE_RMS_MAX, RESCORE_CELLS = D5.BARS_OWNER, D5.RESCORE_RMS_MAX, D5.RESCORE_CELLS
SEEN_DISCLOSURE = D4.SEEN_DISCLOSURE
WEAK_FLOOR = ("a WEAK SANITY FLOOR on SEEN pages (" + SEEN_DISCLOSURE + "; v6d's form and constants were chosen on PC's, "
              "PC2's, DS's and CAL2's pages after that diagnosis, and design-D6 re-scored those pages under v6d variants "
              "before this addendum); never a test and never a certification")
STOPPING_RULE = ("the owner's stopping rule (2026-11-10): decode gets PC2 plus at most one decode blind, run only on a "
                 "gate GO; decode then closes whatever the result. Addendum 6 is the last decode round")

# --------------------------------------------------------------------------
# v6d where the frozen scorers price a cell
# --------------------------------------------------------------------------


def price_cell_v6d(model: str, cell: dict, prm, arm: str = "native", decl_copies: int | None = None,
                   fixes: V4DP.Fixes = V4DP.ALL, terms: V5D.Terms = V5D.NONE, law: V6D.Law | None = V6D.LAW) -> dict | None:
    """dcommon.price_cell's contract (None: NOT PRICED) with v6d at `terms` and `law` over v5d over v4d' at `fixes`."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    declared = None if arm == "native" else cfg.num_experts * int(decl_copies or 1)
    ids = None
    if fixes.naive_order and RD.geometry(cfg, int(cell["T"]), cell["counts"], declared).path == RD.NAIVE:
        ids = V4DP.ids_of(model, cell)
    try:
        p = V6D.price(model, int(cell["T"]), cell["counts"], prm, ids=ids, fixes=fixes, terms=terms, declared=declared,
                      law=law)
    except V4DP.TM.Refused:
        return None
    if prm.T0_GR_ms is not None and p["ms"] is None:
        return None
    return p


@contextlib.contextmanager
def v6d(terms: V5D.Terms, law: V6D.Law | None = V6D.LAW):
    """Inside this block d3common's cell price is v6d's at `terms` and `law` (use it around d3common.v4dp or a d3
    wrapper, which enters v4dp itself)."""
    orig = D3.price_cell_v4dp

    def price_cell_v4dp(model, cell, prm, arm="native", decl_copies=None, fixes=V4DP.ALL):
        return price_cell_v6d(model, cell, prm, arm, decl_copies, fixes, terms, law)
    D3.price_cell_v4dp = price_cell_v4dp
    try:
        yield
    finally:
        D3.price_cell_v4dp = orig


def registered_terms() -> V5D.Terms:
    """v6d's v5d terms: addendum 5's registered terms with the regimes and the W1 scale emptied."""
    return V6D.v6_terms(D5.registered_terms())


# --------------------------------------------------------------------------
# the law's constants from the readings (mechanical)
# --------------------------------------------------------------------------

T16, T32A, T32B, T64 = V6D.T16, V6D.T32A, V6D.T32B, V6D.T64
Q_A, B_A, S3, S4 = 32768, 3670016.0, 396, 528
G_LO, G_HI = 8, 128
R1A, R1B, R1D = "R1a rate curve", "R1b F ladders", "R1d X_sat"
R2A = "R2a c0 per G (F 256 to 14336)"
PC_R2 = "G128 (Mixtral tp8, tp4, tp2, 8x7B)"


def committed() -> tuple[dict, dict, list]:
    return (json.loads((REPO / PC_READ).read_text()), json.loads((REPO / PC2_READ).read_text()),
            json.loads((REPO / SPANS).read_text()))


def _read(x) -> bool:
    return str((x or {}).get("status", "")).startswith("READ")


def pc_rungs(pc: dict) -> list[dict]:
    """PC-R7's rungs G 4 to G 384 (single-wave, 64x128x64w4s3, 1710 MHz) with PC-R1's median life, READ only, by n."""
    lam = {r["cell"]: r["lam_us"] for r in pc["PC-R1"]["A (D64, 1710)"] if str(r["status"]).startswith("READ")}
    rows = [dict(r, lam_us=lam[r["label"]]) for r in pc["PC-R7"]["rows"] if r["label"] in lam and r["G"] <= 384]
    return sorted(rows, key=lambda r: r["n"])


def pc_kappa(pc: dict, spans: list) -> list[tuple]:
    """(fill at S 396, kappa) on block A's w2 single-wave rungs: the median over valid calls of the span over the
    median over valid calls of the median life (d6_build/pc_spans.json)."""
    by: dict = {}
    for r in spans:
        if r["valid"] and r["unit"] == "sa" and r["gemm"] == "w2":
            by.setdefault(r["cell"], []).append(r)
    out = []
    for row in pc_rungs(pc):
        v = by.get(row["label"])
        if v:
            k = float(np.median([x["span_us"] for x in v])) / float(np.median([x["life_med_us"] for x in v]))
            out.append((row["n"] / S3, k))
    return out


def r3_kappa(pc2: dict, tile: str) -> list[tuple]:
    """(fill at S 528, kappa) on PC2-R3's READ single-wave rungs of a 4-per-SM tile at K 14336 (no F suffix)."""
    return sorted((r["n"] / S4, r["kappa"]) for r in pc2["PC2-R3"][tile]["rows"]
                  if "-F" not in r["cell"] and _read(r) and r.get("kappa") is not None and r["G"] <= S4)


def c0_points(pc: dict, pc2: dict) -> dict:
    hi = float(pc["PC-R2"][PC_R2]["c0_us"])
    lo64 = float(pc2["PC2-R2"][R2A]["G 8"]["c0_us"])
    lo16 = float(pc2["PC2-R1"][R1B]["G 8"]["c0_us"])
    return {T16: ((G_LO, lo16), (G_HI, hi)), T32A: ((G_LO, lo64), (G_HI, hi)), T32B: ((G_LO, lo64), (G_HI, hi)),
            T64: ((G_LO, lo64), (G_HI, hi))}


def law_from(pc: dict, pc2: dict, spans: list, ds: dict | None = None) -> dict:
    """The law's constants (r3_timing_model_v6d.Law's dict form) from the readings, mechanically. Call `blocking`
    first; a missing reading raises KeyError."""
    ds = ds if ds is not None else D5.committed_readings()
    c0 = c0_points(pc, pc2)
    tmp = V6D.Law({"C0": c0})
    N, L = [], []
    for r in pc_rungs(pc):
        c = tmp.c0_at(T64, r["n"])
        L.append(Q_A * (r["lam_us"] - c) / B_A)
        N.append(r["n"] * Q_A * (r["lam_us"] - c) / r["lam_us"])
    x_pc = float(pc["PC-R7"]["X_sat"])
    w1 = {t: float(v["counter_GBps"]) for t, v in ds["W1RATE"]["tiles"].items()}
    xs = {T64: x_pc, T16: float(pc2["PC2-R1"][R1D]["X_sat"]),
          T32A: x_pc * w1[T32A] / w1[T64], T32B: x_pc * w1[T32B] / w1[T64]}
    pk = pc_kappa(pc, spans)
    kap = {T64: tuple(pk), T16: tuple(pk)}
    for t in (T32A, T32B):
        own = r3_kappa(pc2, t)
        kap[t] = tuple([p for p in pk if p[0] < own[0][0]] + own)
    nif = {}
    for src in (pc["PC-R0"].get("sa") or {}, *(pc2["PC2-R0"].get(u) or {} for u in ("sd", "ss", "sw"))):
        for t, v in (src.get("inflight") or {}).items():
            if t in V6D.TILES:
                nif.setdefault(t, v)
    return {"C0": {t: [list(p) for p in v] for t, v in c0.items()}, "L_TABLE": [N, L], "X_SAT": xs,
            "KAPPA": {t: [list(p) for p in v] for t, v in kap.items()}, "N_IF": nif}


def blocking(pc: dict, pc2: dict, spans: list) -> list[str]:
    """Every reading the law needs that is not READ (each BLOCKING: v6d never falls back)."""
    why = []
    if pc["PC-R7"].get("status_sat") != "READ":
        why.append(f"PC-R7 X_sat: {pc['PC-R7'].get('status_sat')}")
    if (pc["PC-R2"].get(PC_R2) or {}).get("status") != "READ":
        why.append("PC-R2 G 128 c0: not READ")
    if len(pc_rungs(pc)) < 8:
        why.append(f"PC-R7/R1: {len(pc_rungs(pc))} READ rungs (needs 8)")
    if len(pc_kappa(pc, spans)) < 8:
        why.append(f"PC spans: kappa at {len(pc_kappa(pc, spans))} rungs (needs 8)")
    if (pc2["PC2-R1"][R1D] or {}).get("status") != "READ":
        why.append("PC2-R1d X_sat: not READ")
    if (pc2["PC2-R1"][R1B].get("G 8") or {}).get("status") != "READ":
        why.append("PC2-R1b G 8 c0: not READ")
    if (pc2["PC2-R2"][R2A].get("G 8") or {}).get("status") != "READ":
        why.append("PC2-R2a G 8 c0: not READ")
    for t in (T32A, T32B):
        if len(r3_kappa(pc2, t)) < 3:
            why.append(f"PC2-R3 {t}: {len(r3_kappa(pc2, t))} READ kappa rungs (needs 3)")
    nif = law_from(pc, pc2, spans)["N_IF"] if not why else {}
    for t in V6D.TILES:
        if not why and nif.get(t) is None:
            why.append(f"n_if {t}: not read")
    return why


REL = 1e-9


def same_law(a: dict, b: dict) -> list[str]:
    """Where two law constant sets differ (each number to 1e-9 relative)."""
    why = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                why.append(f"{path}: keys {sorted(x)} against {sorted(y)}")
                return
            for k in x:
                walk(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                why.append(f"{path}: {len(x)} entries against {len(y)}")
                return
            for i, (p, q) in enumerate(zip(x, y, strict=True)):
                walk(p, q, f"{path}[{i}]")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if abs(x - y) > REL * max(abs(x), abs(y), 1e-12):
                why.append(f"{path}: {x} against {y}")
        elif x != y:
            why.append(f"{path}: {x!r} against {y!r}")
    walk(json.loads(json.dumps(a)), json.loads(json.dumps(b)), "law")
    return why


def registered_law() -> dict:
    return V6D.LAW.as_dict()


# --------------------------------------------------------------------------
# checks the registration computes
# --------------------------------------------------------------------------

def reduction() -> dict:
    """COMPUTED: v6d with the law off is v5d at the same terms on every registered CAL and blind main-page NATIVE cell
    (geometry prices, no constants); and with the law on, every naive cell with a non-fetching follower keeps v4d''s
    naive GEMM prices."""
    from moe.spec import MODEL_CONFIGS
    prm = D4.ds_prm()
    terms = registered_terms()
    n = same = kept = kept_same = lawed = 0
    for m in DC.CAL_MODELS + DC.BLIND_MODELS:
        for c in DC.page_doc(m, "main")["cells"]:
            if "native" not in c.get("arms", ["native"]):
                continue
            ids = None
            if RD.geometry(MODEL_CONFIGS[m], int(c["T"]), c["counts"]).path == RD.NAIVE:
                ids = V4DP.ids_of(m, c)
            try:
                a = V5D.price(m, int(c["T"]), c["counts"], prm, ids=ids, terms=terms)
            except TM.Refused:
                continue
            b = V6D.price(m, int(c["T"]), c["counts"], prm, ids=ids, terms=terms, law=None)
            n += 1
            same += int(a["body_ms"] == b["body_ms"] and a["bytes"] == b["bytes"])
            if a["path"] == RD.NAIVE:
                t = V6D.price(m, int(c["T"]), c["counts"], prm, ids=ids, terms=terms)
                if (t.get("v6d") or {}).get("gemm") is None:
                    kept += 1
                    kept_same += int(all(t["gemm"][g]["gemm_ms"] == a["gemm"][g]["gemm_ms"] for g in TM.GEMMS))
                else:
                    lawed += 1
    return {"cells": n, "identical to v5d with the law off": same, "naive cells with a non-fetching follower": kept,
            "of them keeping v4d''s naive GEMM prices": kept_same, "naive cells priced by the law": lawed,
            "verdict": "REDUCES" if n and same == n and kept_same == kept else "DIFFERS"}


def new_model_reach(law: V6D.Law = V6D.LAW) -> dict:
    """COMPUTED from the new models' configs and their registered page lists (no page of theirs is read): per model the
    main cells by path, the aligned GEMMs by tile and fill class, how many single-wave GEMMs sit at fill over 0.73 on a
    4-per-SM tile (where term 3's kappa acts), the naive cells (with and without a non-fetching follower), and the
    short-slab single-wave GEMMs (slab under 459 KB, PC2's shortest READ single-wave w2 class)."""
    from moe.spec import MODEL_CONFIGS
    prm = D4.ds_prm()
    terms = registered_terms()
    out = {}
    for m in NEW_BLIND:
        cfg = MODEL_CONFIGS[m]
        paths, cls, naive, short, big4 = {}, {}, {"no non-fetching follower": 0, "a non-fetching follower": 0}, 0, 0
        for c in D5.page5(m, "main")["cells"]:
            g = RD.geometry(cfg, int(c["T"]), c["counts"])
            paths[g.path] = paths.get(g.path, 0) + 1
            ids = V4DP.ids_of(m, c) if g.path == RD.NAIVE else None
            p = V6D.price(m, int(c["T"]), c["counts"], prm, ids=ids, terms=terms, law=law)
            v = p.get("v6d") or {}
            if g.path == RD.NAIVE:
                naive["a non-fetching follower" if v.get("gemm") is None else "no non-fetching follower"] += 1
                continue
            for gm, r in (v.get("gemm") or {}).items():
                f = r["n"] / r["S"]
                k = f"{v['tile']} {gm} " + ("multi-wave" if f > 1 else "fill <= 0.33" if f <= 1 / 3 else
                                            "fill <= 0.67" if f <= 2 / 3 else "fill <= 1")
                cls[k] = cls.get(k, 0) + 1
                if f <= 1 and r["slab"] < 459e3:
                    short += 1
                if f <= 1 and r["S"] == S4 and f > 0.73:
                    big4 += 1
        out[m] = {"E": cfg.num_experts, "k": cfg.top_k, "H": cfg.hidden_size, "I": cfg.intermediate_size,
                  "main cells by path": paths, "naive cells": naive, "aligned GEMMs by tile and fill": dict(sorted(cls.items())),
                  "single-wave GEMMs with a slab under 459 KB": short,
                  "single-wave GEMMs on a 4-per-SM tile at fill over 0.73 (term 3)": big4}
    return out
