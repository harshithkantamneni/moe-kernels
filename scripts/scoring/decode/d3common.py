#!/usr/bin/env python
"""The decode stage's third addendum (docs/registered/2026-10-29-decode-addendum3-gh200): the rules the owner's
decisions of 2026-10-29 add, in one module for the registration builder, the three scorer wrappers and the page
writer. It imports the frozen decode modules (dcommon, d2common, finalise_decode) and v4d' (r3_timing_model_v4dp)
and changes none of them: every rule here acts by swapping v4d' in for v4d where the frozen code prices a cell,
by re-reading the frozen stop rule with the G3-D label line set DESCRIPTIVE, or by reading a DESCRIPTIVE row.

THE RULES THIS FILE OWNS:
  v4d'        `v4dp(fixes)`: dcommon.price_cell (the one place every frozen decode scorer prices a page cell)
              priced by r3_timing_model_v4dp.price with the cell's own ids (the naive path's launch order).
              With fixes NONE it is v4d exactly (the registration reproduces cal.score.json byte for byte)
  stop        `stop_rule_d3`: dcommon.stop_rule (unchanged) read on the score, the 'G3-D label' line moved
              to DESCRIPTIVE; every other clause as registered (owner decision 2)
  constants   `w1_marginal`: fix R's ratio recomputed from a decode CAL results tree (the published one for the
              registration; a CAL2 tree as a DESCRIPTIVE replicate). `align_slopes`: the dropped fix A's slopes, a
              DESCRIPTIVE record only (A is not in v4d': owner, 2026-11-03)
  pages       the CAL2 settling pages (M1 tiles, M2 wide, M3 label, M4 gaps), written and checked here
  plan        the CAL2 plan: addendum 2's CAL plan unchanged, then the settling units (drop groups m3, m2,
              m4, m1 dropped first, in that order)
  readings    M1 to M4, DESCRIPTIVE, registered before any page: what each reads, how, and its bands
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2common as D2  # noqa: E402
import dcommon as DC  # noqa: E402

RD, V4, V4D, TM = DC.RD, DC.V4, DC.V4D, DC.V4.TM
REPO = DC.REPO
import r3_timing_model_v4dp as V4DP  # noqa: E402

#: the diagnosis's align-loop slope (fix A, DROPPED from v4d' by the owner on 2026-11-03): printed beside M2's
#: reading for scale, never priced
DIAG_ALIGN_US = 0.96


def align_iterations(numel: int, declared: int) -> int:
    """ceil(numel / max(E declared, 32)): vLLM's small-batch align kernel's per-thread loop count (descriptive)."""
    return math.ceil(int(numel) / max(int(declared), 32))

r6 = DC.r6
DATE = "2026-10-29"
NAME = f"{DATE}-decode-addendum3-gh200"
PAGES3 = f"docs/registered/{DATE}-decode3-pages"
CAL2_PLAN = "scripts/plans/rentalD-cal2-d3-2026-10.plan"
DRIVER3 = "gh200_decode3_session.sh"
PUBLISHED = "results/published/2026-10-29-nvidia_gh200_480gb-decodeCAL-session"
PUBLISHED_SCORE = "scripts/scoring/decode/cal.score.json"

# --------------------------------------------------------------------------
# v4d' where the frozen scorers price a cell
# --------------------------------------------------------------------------


def price_cell_v4dp(model: str, cell: dict, prm, arm: str = "native", decl_copies: int | None = None,
                    fixes: V4DP.Fixes = V4DP.ALL) -> dict | None:
    """dcommon.price_cell's contract (None: NOT PRICED) with v4d' at `fixes`; the naive ids from the cell."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    E = cfg.num_experts
    declared = None if arm == "native" else E * int(decl_copies or 1)
    ids = None
    if fixes.naive_order and RD.geometry(cfg, int(cell["T"]), cell["counts"], declared).path == RD.NAIVE:
        ids = V4DP.ids_of(model, cell)
    try:
        p = V4DP.price(model, int(cell["T"]), cell["counts"], prm, ids=ids, fixes=fixes, declared=declared)
    except V4.TM.Refused:
        return None
    if prm.T0_GR_ms is not None and p["ms"] is None:
        return None
    return p


@contextlib.contextmanager
def v4dp(fixes: V4DP.Fixes = V4DP.ALL):
    """Every frozen decode scorer's cell price (dcommon.price_cell) is v4d''s inside this block."""
    orig = DC.price_cell

    def price_cell(model, cell, prm, arm="native", decl_copies=None):
        return price_cell_v4dp(model, cell, prm, arm, decl_copies, fixes)
    DC.price_cell = price_cell
    try:
        yield
    finally:
        DC.price_cell = orig


def fixes_from(d: dict | None) -> V4DP.Fixes:
    return V4DP.Fixes(**(d or V4DP.ALL.as_dict()))


# --------------------------------------------------------------------------
# the stop rule with G3-D label DESCRIPTIVE (owner decision 2)
# --------------------------------------------------------------------------

LABEL_LINE = "G3-D label "


def stop_rule_d3(score: dict) -> dict:
    """dcommon.stop_rule unchanged, its 'G3-D label' line reported DESCRIPTIVE instead of read."""
    s = DC.stop_rule(score)
    keep = [w for w in s["why"] if not w.startswith(LABEL_LINE)]
    out = {"verdict": "GO" if not keep else "STOP", "why": keep,
           "descriptive": [w for w in s["why"] if w.startswith(LABEL_LINE)]}
    out["frozen_rule"] = {"verdict": s["verdict"], "label": "DESCRIPTIVE: dcommon.stop_rule as registered, the G3-D "
                                                           "label line read as a stop clause"}
    return out


def label_effect(score: dict) -> dict:
    """The G3-D label reading (DESCRIPTIVE): its verdict, margin and the per-model pair differences."""
    g = (score.get("G3D") or {}).get("label") or {}
    rows = g.get("rows") or []
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r["d_us"])
    return {"verdict": g.get("verdict"), "margin_us": g.get("margin_us"),
            "mean_us": (g.get("tost") or {}).get("mean"),
            "per_model": {m: {"pairs": len(v), "d_us": v, "mean_us": r6(st.fmean(v))} for m, v in sorted(by.items())},
            "label": "DESCRIPTIVE (addendum 3, owner decision 2): reported, never a stop clause"}


#: the certified domain under this addendum: addendum 2's scope, then the label rule (owner, 2026-11-03)
SCOPE = (D2.SCOPE + " Certification covers time as a function of the expert counts only: it does not cover a "
         "dependence on which experts are active (Mixtral's published expert-set effect, up to about 2.5 us at R 1). "
         "G3-D label, the test of that dependence, is DESCRIPTIVE (addendum 3, owner 2026-10-29 and 2026-11-03): it is "
         "reported per model and neither stops the CAL2 reading nor voids a blind model's certification; G3-D shuffle "
         "and GD2 void as registered.")
LABEL_DESCRIPTIVE = "NOT SCORED: DESCRIPTIVE (addendum 3, G3-D label never voids)"


@contextlib.contextmanager
def label_descriptive(store: dict):
    """dcommon.g3d inside the frozen blind scorer: the 'label' test is computed exactly as registered and kept in
    `store` per call (keyed by the model of its pairs), but returned to the frozen scorer with a NOT SCORED verdict,
    which score_blind.score reads as no void. Shuffle is untouched."""
    orig = DC.g3d

    def g3d(meas, preds, pairs_by_model, test):
        got = orig(meas, preds, pairs_by_model, test)
        if test != "label":
            return got
        for m in pairs_by_model:
            store[m] = got
        return dict(got, verdict=LABEL_DESCRIPTIVE, registered_verdict=got["verdict"])
    DC.g3d = g3d
    try:
        yield store
    finally:
        DC.g3d = orig


def label_blind(store: dict) -> dict:
    """The blind G3-D label reading per model (DESCRIPTIVE): verdict, margin, mean and the pair differences."""
    out = {}
    for m, g in sorted(store.items()):
        out[m] = {"verdict": g.get("verdict"), "margin_us": g.get("margin_us"), "mean_us": (g.get("tost") or {}).get("mean"),
                  "d_us": [r["d_us"] for r in g.get("rows") or []],
                  "label": "DESCRIPTIVE (addendum 3, owner 2026-11-03): reported, never a void"}
    return out


# --------------------------------------------------------------------------
# the measured constants, recomputed from a decode CAL tree
# --------------------------------------------------------------------------

def _counter_rows(tree: Path, model: str, label: str, page_kind: str, prm, fixes=V4DP.NONE) -> list[dict]:
    """One counter page's NATIVE cells: per GEMM the counted bytes, ncu time and the model's GEMM price."""
    import score_cal as SC
    page, _why = SC.counter_page(tree, model, label)
    cells = {c["label"]: c for c in page_doc_any(model, page_kind)["cells"]}
    out = []
    for c in (page or {}).get("cells") or []:
        if c.get("arm") != RD.NATIVE or c["label"] not in cells:
            continue
        pc = cells[c["label"]]
        p = price_cell_v4dp(model, pc, prm, fixes=fixes)
        if p is None:
            continue
        from moe.spec import MODEL_CONFIGS
        g = RD.geometry(MODEL_CONFIGS[model], int(pc["T"]), pc["counts"])
        row = {"model": model, "label": c["label"], "T": int(pc["T"]), "A": g.A, "path": g.path, "tile": g.tile,
               "multi": g.multi_tile_experts}
        for gm in RD.GEMMS:
            pg = c["per_gemm"].get(gm) or {}
            row[gm] = {"counted": pg.get("dram__bytes_read.sum"), "ncu_ns": pg.get("gpu__time_duration.sum"),
                       "model_ms": p["gemm"][gm]["gemm_ms"]}
        out.append(row)
    return out


def marginal_pairs(rows: list[dict], *, min_mb: float = 100.0) -> list[dict]:
    """The diagnosis's rule (marginal_rates.py): kept cells (no second M-tile) of one (T, path), the lowest and
    highest w1 counted bytes, at least `min_mb` MB apart; per GEMM the counters' and the model's marginal rate."""
    by = {}
    for r in rows:
        if r["multi"] == 0 and all(r[g]["counted"] and r[g]["ncu_ns"] for g in RD.GEMMS):
            by.setdefault((r["model"], r["T"], r["path"]), []).append(r)
    out = []
    for (m, T, path), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["w1"]["counted"])
        lo, hi = rs[0], rs[-1]
        if (hi["w1"]["counted"] - lo["w1"]["counted"]) / 1e6 < min_mb:
            continue
        rec = {"model": m, "T": T, "path": path, "tile": lo["tile"], "lo": lo["label"], "hi": hi["label"]}
        for g in RD.GEMMS:
            dB = hi[g]["counted"] - lo[g]["counted"]
            rec[g] = {"counter_GBps": dB / (hi[g]["ncu_ns"] - lo[g]["ncu_ns"]),
                      "model_GBps": dB / ((hi[g]["model_ms"] - lo[g]["model_ms"]) * 1e6)}
        out.append(rec)
    return out


def w1_marginal(tree: Path, prm, *, label: str = "dn", page_kind: str = "ctl", tile: str = V4DP.W1_TILE,
                models=DC.CAL_MODELS, fixes=V4DP.NONE) -> dict:
    """Fix R's ratio: v4d's pooled w1 marginal rate over the counters' on the tile's kept-cell pairs."""
    pairs = [p for m in models for p in marginal_pairs(_counter_rows(tree, m, label, page_kind, prm, fixes))
             if p["tile"] == tile]
    if not pairs:
        return {"verdict": "NOT READ: no kept-cell pair", "pairs": []}
    c = st.fmean(p["w1"]["counter_GBps"] for p in pairs)
    v = st.fmean(p["w1"]["model_GBps"] for p in pairs)
    return {"verdict": "READ", "pairs": [{k: x[k] for k in ("model", "T", "lo", "hi")} |
                                         {"counter_GBps": r6(x["w1"]["counter_GBps"]), "model_GBps": r6(x["w1"]["model_GBps"])}
                                         for x in pairs],
            "counter_GBps": c, "model_GBps": v, "scale": v / c, "tile": tile}


def numel_other_us(cfg, numel: int, T: int, prm) -> float:
    """v4d's B_other (silu_and_mul and moe_sum at DRAM speed, r3_timing_model_v4d.price), in us."""
    b = (numel * 2 * cfg.intermediate_size + numel * cfg.intermediate_size + numel * cfg.hidden_size
         + T * cfg.hidden_size) * 2
    return b / (prm.bw_gbps * 1e6) * 1e3


def align_slopes(tree: Path, reg: dict, *, label: str = "dn", models=DC.CAL_MODELS) -> dict:
    """Fix A's slopes: per model, X = GRR main-page time - ncu w1 - ncu w2 (us) on the kept small-batch cells
    with a counter row (matched by label), against ceil(numel / max(E, 32)); Mixtral's A 2 cells at T >= 4 left
    out (the diagnosis's checks.py (a): ncu's w2 there reads +14 us over T 1-2, a step GRR does not show)."""
    import score_cal as SC
    from moe.spec import MODEL_CONFIGS
    files = reg["pages"]["files"]
    excl = D2.excluded_keys(models)
    out = {}
    for m in models:
        rep, _why = DC.decode_page(tree, m, "dm", files[f"{m}-main.json"], "GRR")
        page, _w = SC.counter_page(tree, m, label)
        if rep is None or page is None:
            out[m] = {"verdict": "NOT READ: no main or counter page"}
            continue
        meas, _d = DC.measured(rep, m, "main")
        grr = {k[2]: row["ms_p50"] for k, row in meas.items() if k[4] == "native" and (m, k[2]) not in excl}
        cells = {c["label"]: c for c in DC.page_doc(m, "main")["cells"]}
        cfg = MODEL_CONFIGS[m]
        pts, held = [], []
        bare = V4D.DecodeParams()
        for c in page.get("cells") or []:
            lab = c["label"]
            if c.get("arm") != RD.NATIVE or lab not in grr or lab not in cells:
                continue
            g = RD.geometry(cfg, int(cells[lab]["T"]), cells[lab]["counts"])
            if g.path != RD.SMALL_BATCH:
                continue
            ncu = sum(c["per_gemm"][gm]["gpu__time_duration.sum"] for gm in RD.GEMMS) / 1e3
            other = numel_other_us(cfg, g.numel, g.T, bare)
            pt = (align_iterations(g.numel, cfg.num_experts), grr[lab] * 1e3 - ncu, lab, other)
            (held if (m == "mixtral-8x7b" and g.A == 2 and g.T >= 4) else pts).append(pt)
        if len(pts) < 3:
            out[m] = {"verdict": f"NOT READ: {len(pts)} cells"}
            continue
        it, x = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
        b, a = np.polyfit(it, x, 1)
        bo = np.polyfit(it, np.array([p[3] for p in pts], float), 1)[0]
        bn = np.polyfit(it, x - np.array([p[3] for p in pts], float), 1)[0]
        rec = {"verdict": "READ", "cells": len(pts), "iterations": [int(it.min()), int(it.max())],
               "slope_us": float(b), "intercept_us": float(a), "labels": [p[2] for p in pts],
               "B_other_slope_us": float(bo), "slope_net_of_B_other_us": float(bn)}
        if held:
            allp = pts + held
            rec["slope_with_held_out_cells_us"] = float(np.polyfit([p[0] for p in allp], [p[1] for p in allp], 1)[0])
            rec["held_out"] = [p[2] for p in held]
        out[m] = rec
    read = [x for x in out.values() if x.get("verdict") == "READ"]
    out["mean_slope_us"] = st.fmean(x["slope_us"] for x in read) if read else None
    if read and all("slope_with_held_out_cells_us" in x or k == "olmoe-1b-7b" for k, x in out.items() if isinstance(x, dict)
                    and x.get("verdict") == "READ"):
        out["mean_slope_with_held_out_us"] = st.fmean(x.get("slope_with_held_out_cells_us", x["slope_us"]) for x in read)
    return out


# --------------------------------------------------------------------------
# the CAL2 settling pages (M1 to M4)
# --------------------------------------------------------------------------

#: M1: main-page kept cells at two A on each of the three tiles without a counter pair (diagnosis M1)
M1_CELLS = {"mixtral-8x7b": ("L-T64-A4", "L-T64-A8", "L-T96-A6", "L-T96-A8", "L-T128-A4", "L-T128-A8"),
            "olmoe-1b-7b": ("L-T64-A27", "L-T64-A64", "L-T96-A27", "L-T96-A64", "L-T128-A27", "L-T128-A64")}
#: M2: ctl-page cells spanning the align loop's iteration count (and OLMoE's block-scan T 128)
M2_CELLS = {"olmoe-1b-7b": ("L-T4-A8", "L-T16-A64", "L-T32-A64", "L-T64-A64", "L-T96-A64", "L-T128-A64"),
            "mixtral-8x7b": ("L-T4-A8", "L-T16-A8", "L-T32-A8", "L-T64-A8", "L-T96-A8", "L-T128-A8")}
#: M3: Mixtral T 4, two experts at 4 rows each, on six expert sets; S01 the reference (MIN-T4's set)
M3_SETS = (("S01", (0, 1)), ("S25", (2, 5)), ("S46", (4, 6)), ("S15", (1, 5)), ("S37", (3, 7)), ("S26", (2, 6)))
M3_T = 4
#: M3's pads (owner, 2026-11-03, review m2): 0 (the control) and the smallest pad that moves expert 5's w1 slab past
#: a 2 MiB page boundary: a pad row adds H x 2 bytes to w1's slot stride (Mixtral H 4096: 8192 B), slot 5 starts
#: 5 x P x 8192 B later, and 5 x P x 8192 >= 2 MiB (2097152 B) needs P >= 51.2, so P = 52 (2129920 B, 2.03 MiB;
#: w2's slot 5 moves 5 x 52 x 14336 x 2 = 7454720 B, 7.11 MiB)
M3_PAGE_BYTES = 2 * 2 ** 20
M3_EXPERT = 5


def m3_pad(model: str = "mixtral-8x7b", expert: int = M3_EXPERT, page: int = M3_PAGE_BYTES) -> int:
    from moe.spec import MODEL_CONFIGS
    row = MODEL_CONFIGS[model].hidden_size * 2
    return math.ceil(page / (expert * row))


M3_PADS = (0, m3_pad())
#: M4: OLMoE T 2, three repeated experts at row gap 8 + s (s token-1 singles indexed below them), one at 15,
#: and the no-repeat control (L-T2-A16's counts)
M4_GAPS3 = (0, 1, 2, 3, 4, 5)
M4_SLOTS = 132 * 3


def _cell(label, T, counts, role, shuffle=True, **prov) -> dict:
    d = {"label": label, "T": int(T), "counts": [int(v) for v in counts], "arms": ["native"], "role": role}
    if not shuffle:
        d["shuffle"] = False
    d.update(prov)
    return d


def _copy_cells(model: str, kind: str, labels) -> list[dict]:
    cells = {c["label"]: c for c in DC.page_doc(model, kind)["cells"]}
    out = []
    for lab in labels:
        c = cells[lab]
        out.append(_cell(lab, c["T"], c["counts"], c["role"], c.get("shuffle", True), source=f"{model}-{kind}"))
    return out


def m4_counts(s: int, repeats: int = 3) -> list[int]:
    """OLMoE T 2: `repeats` experts at 2 rows, the rest of the 16 rows singles; token 0 takes the repeated
    experts and the lowest-index singles (realize_counts' greedy order), token 1 the rest, so s singles indexed
    below the repeated experts put each follower s rows past 8 from its leader."""
    E, k = 64, 8
    n1 = 2 * k - 2 * repeats
    low = list(range(k - repeats))                       # token 0's singles
    rest = list(range(len(low), len(low) + n1 - len(low) + repeats))
    mid, rep_ids, high = rest[:s], rest[s:s + repeats], rest[s + repeats:]
    c = [0] * E
    for e in low + mid + high:
        c[e] = 1
    for e in rep_ids:
        c[e] = 2
    assert sum(c) == 2 * k and sum(1 for v in c if v == 1) == n1
    return c


def rows_of_ids(ids) -> list[list[int]]:
    first, out = {}, []
    for r, e in enumerate(ids):
        if e in first:
            out.append([first[e], r])
        else:
            first[e] = r
    return out


def m4_cells() -> list[dict]:
    m = "olmoe-1b-7b"
    out = []
    for s in M4_GAPS3:
        c = _cell(f"G{8 + s:02d}-R3", 2, m4_counts(s), "GAP", shuffle=False)
        c["d3_rows"] = rows_of_ids(V4DP.ids_of(m, c))
        out.append(c)
    c = _cell("G15-R1", 2, m4_counts(7, repeats=1), "GAP", shuffle=False)
    c["d3_rows"] = rows_of_ids(V4DP.ids_of(m, c))
    out.append(c)
    base = {x["label"]: x for x in DC.page_doc(m, "main")["cells"]}["L-T2-A16"]
    out.append(_cell("BASE-A16", 2, base["counts"], "GAP-CONTROL", shuffle=False, d3_rows=[]))
    return out


def m3_cells() -> list[dict]:
    out = []
    for lab, (a, b) in M3_SETS:
        c = [0] * 8
        c[a] = c[b] = M3_T
        out.append(_cell(f"{lab}-T{M3_T}", M3_T, c, "LABEL-SET", experts=[a, b]))
    return out


PAGE_SPECS = {
    ("olmoe-1b-7b", "tiles"): lambda: _copy_cells("olmoe-1b-7b", "main", M1_CELLS["olmoe-1b-7b"]),
    ("mixtral-8x7b", "tiles"): lambda: _copy_cells("mixtral-8x7b", "main", M1_CELLS["mixtral-8x7b"]),
    ("olmoe-1b-7b", "wide"): lambda: _copy_cells("olmoe-1b-7b", "ctl", M2_CELLS["olmoe-1b-7b"]),
    ("mixtral-8x7b", "wide"): lambda: _copy_cells("mixtral-8x7b", "ctl", M2_CELLS["mixtral-8x7b"]),
    ("mixtral-8x7b", "label"): m3_cells,
    ("olmoe-1b-7b", "gaps"): m4_cells,
}
PAGE_WHAT = {"tiles": "M1: counter pages at two A on each of the T 64, T 96 and T 128 tiles (main-page kept cells)",
             "wide": "M2: the wider ncu run (align, silu_and_mul, moe_sum and the GEMMs) on ctl-page cells",
             "label": "M3: Mixtral label sets at T 4, run at pad rows 0 and 52 (rental 2's address control; 52 moves expert 5's w1 slab past a 2 MiB page)",
             "gaps": "M4: OLMoE T 2 cells with controlled leader-follower row gaps (fix W's same-wave / later-wave split)"}


def page3_path(model: str, kind: str) -> str:
    return f"{PAGES3}/{model}-{kind}.json"


def page3(model: str, kind: str) -> dict:
    return {"schema": RD.SCHEMA, "model": model, "page": f"{model}-{kind}", "shuffle_seed": V4DP.PAGE_SEED,
            "decl_copies": None, "label": f"SYNTHETIC (scripts/scoring/decode/d3common.py, {NAME}); {PAGE_WHAT[kind]}",
            "cells": PAGE_SPECS[(model, kind)]()}


def page3_files() -> dict[str, str]:
    return {f"{m}-{k}.json": json.dumps(page3(m, k), indent=1) + "\n" for m, k in PAGE_SPECS}


def page_doc_any(model: str, kind: str) -> dict:
    if (model, kind) in PAGE_SPECS:
        return json.loads((REPO / page3_path(model, kind)).read_text())
    return DC.page_doc(model, kind)


def reg_file(model: str, kind: str) -> dict:
    """The registered record of one settling page (decode_page's G5 check: file sha256, cell counts sha256)."""
    p = REPO / page3_path(model, kind)
    doc = json.loads(p.read_text())
    return {"path": page3_path(model, kind), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "shuffle_seed": doc["shuffle_seed"],
            "cells": [{"label": c["label"], "T": c["T"], "counts_sha256": RD.counts_sha256(c["counts"]),
                       "role": c["role"], "arms": c["arms"], "shuffle": c.get("shuffle", True)} for c in doc["cells"]]}


# --------------------------------------------------------------------------
# the CAL2 settling units and the plan
# --------------------------------------------------------------------------

def _rd_est(model: str, kind: str) -> int:
    from moe.spec import MODEL_CONFIGS
    page = RD.page_from(page3(model, kind), MODEL_CONFIGS[model], model)
    return math.ceil(RD.estimated_seconds(page, 9) / 60)


def _cr_est(model: str, kind: str, kernels_per_call: int = 2) -> int:
    """The counter route's own rule (dram_counter_route_decode's dry run: 0.6 s per profiled launch, warmups
    profiled) at `kernels_per_call` launches per call, rounded up, plus 2 min of setup (the registered dn
    units: OLMoE's 26 cells read 2.6 min by the rule and were registered at 5)."""
    n = len(page3(model, kind)["cells"])
    launches = n * (3 + 2) * kernels_per_call
    return math.ceil(launches * 0.6 / 60) + 2


#: (group, model, step, label, page kind, extra keys) in plan order; the driver drops the last first
SETTLE = (
    ("m1", "olmoe-1b-7b", "decodebytes", "dt", "tiles", ""),
    ("m1", "mixtral-8x7b", "decodebytes", "dt", "tiles", ""),
    ("m4", "olmoe-1b-7b", "r3decode", "dg", "gaps", "inst=GRR slip-policy=cell g1-rule=r7-median-run3"),
    ("m4", "olmoe-1b-7b", "decodebytes", "dgn", "gaps", ""),
    ("m2", "olmoe-1b-7b", "decodebytes", "dw", "wide", "kernels=wide"),
    ("m2", "mixtral-8x7b", "decodebytes", "dw", "wide", "kernels=wide"),
    ("m3", "mixtral-8x7b", "r3decode", "dp0", "label", "inst=GRR pad-rows=0 slip-policy=cell g1-rule=r7-median-run3"),
    ("m3", "mixtral-8x7b", "r3decode", f"dp{M3_PADS[1]}", "label",
     f"inst=GRR pad-rows={M3_PADS[1]} slip-policy=cell g1-rule=r7-median-run3"),
)
#: the wide route's launches per call at most (block-scan: two align kernels, w1, silu_and_mul, w2, moe_sum)
WIDE_KERNELS_PER_CALL = 6


def settle_units() -> list[dict]:
    out = []
    for grp, m, step, lab, kind, keys in SETTLE:
        if step == "r3decode":
            est = _rd_est(m, kind)
        else:
            est = _cr_est(m, kind, WIDE_KERNELS_PER_CALL if "kernels=wide" in keys else 2)
        out.append({"unit": grp, "model": m, "step": step, "label": lab, "page": page3_path(m, kind), "kind": kind,
                    "keys": keys, "drop": grp, "est": est, "cap": max(15, 3 * est)})
    return out


def settle_line(u: dict) -> str:
    k = f" {u['keys']}" if u["keys"] else ""
    return f"{u['model']} {u['step']} label={u['label']} histogram={u['page']}{k} drop-group={u['drop']} est={u['est']} cap={u['cap']}"


def cal2_plan() -> str:
    """Addendum 2's CAL plan, its unit lines byte-identical, then the settling units, under this header."""
    base = D2.cal_plan_d2()
    units = [ln for ln in base.splitlines() if ln and not ln.startswith("#")]
    hdr = ["# THE DECODE STAGE'S CAL2 RENTAL under the third addendum (2026-10-29): the registered CAL plan (addendum 2's",
           f"# {D2.CAL_PLAN_D2}, every unit line byte-identical) and the diagnosis's settling units M1 to M4,",
           f"# DESCRIPTIVE. GENERATED by scripts/scoring/decode/finalise_decode_d3.py --cal-plan from registration {NAME};",
           "# do not edit by hand. One gpu_1x_gh200 at $2.29/h, the decode driver's D3 copy:",
           "#   bash scripts/vm_run.sh start --ip <ip> --run-id <id> --deadline <epoch s> \\",
           f"#     --plan {CAL2_PLAN} --driver {DRIVER3}",
           "# Drop order, last first: m3, m2, m4, m1 (the settling units), then rep, decl, e240. A settling unit's page is",
           "# never a STOP and never enters the CAL fit."]
    return "\n".join(hdr + units + [settle_line(u) for u in settle_units()]) + "\n"


def cal2_minutes() -> dict:
    base = json.loads((REPO / "docs" / "registered" / f"{D2.NAME}.json").read_text())["rentals (owner, 2026-10-28, D2-5)"]
    settle = sum(u["est"] for u in settle_units())
    return {"cal_units_min": base["cal"]["units_min"], "settle_units_min": settle,
            "cal2": D2.rental(base["cal"]["units_min"] + settle), "blind": D2.rental(base["blind"]["units_min"])}


# --------------------------------------------------------------------------
# the settling readings (DESCRIPTIVE)
# --------------------------------------------------------------------------

M3_CUTS = (1.0 / 3.0, 2.0 / 3.0)
M3_MIN_US = 0.5
M4_CUTS = (1.0 / 3.0, 2.0 / 3.0)


def read_m1(tree: Path, prm) -> dict:
    """M1: per model and decode tile, the kept-cell pair of one T at two A on the tiles page (label dt): per GEMM
    the counters' and v4d''s marginal DRAM rate and their ratio; and fix R's own pairs re-read on the CAL2 dn pages
    (a replicate of the registered ratio, never used to change it)."""
    import score_cal as SC
    rows, seen = [], []
    for m in DC.CAL_MODELS:
        if SC.counter_page(tree, m, "dt")[0] is not None:
            seen.append(m)
        rows += marginal_pairs(_counter_rows(tree, m, "dt", "tiles", prm, V4DP.ALL), min_mb=1.0)
    out = {"verdict": "DESCRIPTIVE" if seen else "NOT READ: no tiles counter page (dt)", "pages": seen,
           "pairs": [], "by_tile": {}}
    for p in rows:
        rec = {k: p[k] for k in ("model", "T", "tile", "lo", "hi")}
        for g in RD.GEMMS:
            rec[g] = {"counter_GBps": r6(p[g]["counter_GBps"]), "v4dp_GBps": r6(p[g]["model_GBps"]),
                      "v4dp_over_counter": r6(p[g]["model_GBps"] / p[g]["counter_GBps"])}
        out["pairs"].append(rec)
        out["by_tile"].setdefault(p["tile"], []).append(rec)
    out["by_tile"] = {t: {g: {"pairs": len(v), "mean_ratio": r6(st.fmean(x[g]["v4dp_over_counter"] for x in v))}
                          for g in RD.GEMMS} for t, v in out["by_tile"].items()}
    out["R_replicate"] = w1_marginal(tree, prm)
    out["reads"] = ("the marginal per-byte cost of each tile's GEMMs against v4d''s: a ratio under 1 means v4d' prices "
                    "that tile's bytes slow (as R found on 16x64x128). DESCRIPTIVE; it never refits v4d' or the CAL2 fit")
    return out


def split_wide(launches, attributed) -> list[dict]:
    """M2's attribution of the non-GEMM launches: each to a (cell, call) by its place between the attributed GEMM
    launches (launch ID order). Between a call's w1 and w2: that call (silu_and_mul's slot). After a call's w2:
    an align-class kernel (moe_align*, count_and_sort*) belongs to the next GEMM's call, any other to the
    previous call (moe_sum's slot). Before the first GEMM: the first call's."""
    order = sorted(launches, key=lambda ln: int(ln.launch_id))
    pos = {id(a["launch"]): a for a in attributed}
    gem = [(i, pos[id(ln)]) for i, ln in enumerate(order) if id(ln) in pos]
    out = []
    for i, ln in enumerate(order):
        if id(ln) in pos:
            continue
        prev = next((a for j, a in reversed(gem) if j < i), None)
        nxt = next((a for j, a in gem if j > i), None)
        alignish = any(s in ln.kernel for s in ("moe_align", "count_and_sort"))
        if prev is None:
            owner, slot = nxt, "before"
        elif prev["gemm"] == "w1":
            owner, slot = prev, "between"
        elif alignish and nxt is not None:
            owner, slot = nxt, "align"
        else:
            owner, slot = prev, "after"
        if owner is None:
            continue
        out.append({"key": owner["key"], "call": owner["call"], "warmup": owner["warmup"], "kernel": ln.kernel,
                    "slot": slot, "metrics": dict(ln.metrics)})
    return out


def kernel_class(name: str) -> str:
    if "moe_align" in name or "count_and_sort" in name:
        return "align"
    if "act_and_mul" in name or "silu" in name:
        return "silu_and_mul"
    if "moe_sum" in name:
        return "moe_sum"
    return "other"


def read_m2(tree: Path, prm) -> dict:
    """M2: the wide counter pages (label dw): per cell the median per call of each kernel class's
    gpu__time_duration and dram bytes; per model the OLS slope and intercept of the align class's time against
    ceil(numel / max(E, 32)) (fix A's 0.96 us beside), and silu_and_mul + moe_sum against v4d's B_other price
    (DRAM speed). Under kernel replay with --cache-control all every kernel starts at a cold L2, so whether the
    intermediates are L2-resident in the real call is NOT READ here."""
    import score_cal as SC
    from moe.spec import MODEL_CONFIGS
    out = {"verdict": "DESCRIPTIVE", "models": {}}
    for m in DC.CAL_MODELS:
        page, why = SC.counter_page(tree, m, "dw")
        if page is None:
            out["models"][m] = {"verdict": f"NOT READ: {why}"}
            continue
        cfg = MODEL_CONFIGS[m]
        cells = {c["label"]: c for c in page_doc_any(m, "wide")["cells"]}
        rows, pts = [], []
        for c in page.get("cells") or []:
            pc = cells.get(c["label"])
            if pc is None or c.get("arm") != RD.NATIVE:
                continue
            g = RD.geometry(cfg, int(pc["T"]), pc["counts"])
            p = price_cell_v4dp(m, pc, prm, fixes=V4DP.ALL)
            k = c.get("per_class") or {}
            us = {cl: (k.get(cl) or {}).get("gpu__time_duration.sum") for cl in ("align", "silu_and_mul", "moe_sum", "other")}
            us = {cl: (None if v is None else v / 1e3) for cl, v in us.items()}
            it = align_iterations(g.numel, cfg.num_experts)
            row = {"label": c["label"], "T": g.T, "path": g.path, "iterations": it, "us": us,
                   "kernels": sorted(c.get("kernels") or []),
                   "B_other_us": None if p is None else r6(p["other_ms"] * 1e3)}
            so = [v for v in (us["silu_and_mul"], us["moe_sum"]) if v is not None]
            if p is not None and len(so) == 2:
                row["silu_moe_sum_over_B_other"] = r6(sum(so) / (p["other_ms"] * 1e3))
            rows.append(row)
            if g.path == RD.SMALL_BATCH and us["align"] is not None:
                pts.append((it, us["align"]))
        rec = {"cells": rows}
        if len(pts) >= 3 and len({p[0] for p in pts}) >= 2:
            b, a = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
            rec["align_fit"] = {"slope_us": r6(float(b)), "intercept_us": r6(float(a)), "cells": len(pts),
                                "diagnosis_us_per_iteration (not in v4d')": DIAG_ALIGN_US}
        else:
            rec["align_fit"] = {"verdict": f"NOT READ: {len(pts)} small-batch cells with an align-class launch"}
        out["models"][m] = rec
    if all(str(x.get("verdict", "")).startswith("NOT READ") for x in out["models"].values()):
        out["verdict"] = "NOT READ: no wide counter page (dw)"
    return out


def read_m3(tree: Path) -> dict:
    """M3: the label page at pad rows 0 (dp0) and M3_PADS[1] (52: expert 5's w1 slab moved past a 2 MiB page
    boundary), GRR: per set d = T(set) - T(S01) at each pad; the expert-5 sets' mean d (S25, S15) at both pads and
    q = d_pad / d0. FOLLOWS-EXPERT when q >= 2/3 (the effect does not follow a page-level shift of the slab: identity
    or data), FOLLOWS-ADDRESS when q <= 1/3 (it moved with placement), MIXED between; NOT READ when d0 is under
    0.5 us (the effect did not reproduce at pad 0), a page is missing, or a page's report does not record the pad
    rows its unit registers (review m1: r3_decode3.py writes report.json["d3"]["pad_rows"])."""
    rf = reg_file("mixtral-8x7b", "label")
    t = {}
    for pad in M3_PADS:
        lab = f"dp{pad}"
        root = DC.C3.find_gaps(tree, "mixtral-8x7b", lab)
        raw = [r for _p, r in DC.C3.timed_pages(root) if r.get("kind") == "decode-page"]
        got = [(r.get("d3") or {}).get("pad_rows") for r in raw]
        if got != [pad]:
            return {"verdict": f"NOT READ: {lab} records pad rows {got}, its unit registers [{pad}]"}
        rep, why = DC.decode_page(tree, "mixtral-8x7b", lab, rf, "GRR")
        if rep is None:
            return {"verdict": f"NOT READ: {lab} {why}"}
        meas, _d = DC.measured(rep, "mixtral-8x7b", "label")
        t[pad] = {k[2]: row["ms_p50"] * 1e3 for k, row in meas.items() if k[4] == "native"}
    ref = f"S01-T{M3_T}"
    sets = [f"{s}-T{M3_T}" for s, _e in M3_SETS]
    d = {p: {s: r6(t[p][s] - t[p][ref]) for s in sets if s in t[p] and ref in t[p]} for p in M3_PADS}
    e5 = [f"{s}-T{M3_T}" for s, e in M3_SETS if M3_EXPERT in e]
    P = M3_PADS[1]
    d0 = [d[0][s] for s in e5 if s in d[0]]
    d7 = [d[P][s] for s in e5 if s in d[P]]
    out = {"d_us": {str(p): v for p, v in d.items()}, "expert5_sets": e5, "pads": list(M3_PADS)}
    if not d0 or not d7:
        return dict(out, verdict="NOT READ: an expert-5 set has no valid cell")
    m0, m7 = st.fmean(d0), st.fmean(d7)
    out.update(d0_us=r6(m0), d_pad_us=r6(m7))
    if m0 < M3_MIN_US:
        return dict(out, verdict=f"NOT READ: the expert-5 sets read {m0:.2f} us over S01 at pad 0, under {M3_MIN_US} us")
    q = m7 / m0
    out["q"] = r6(q)
    out["verdict"] = "FOLLOWS-EXPERT" if q >= M3_CUTS[1] else "FOLLOWS-ADDRESS" if q <= M3_CUTS[0] else "MIXED"
    return out


def read_m4(tree: Path, prm) -> dict:
    """M4: the gaps page. GRR (dg): each cell's time against v4d' (fix W) and v4d (dedup) at the CAL2 fit. Counters
    (dgn): per GEMM f = (counted - dedup) / (nodedup - dedup), the share of the followers' slabs re-fetched, against
    the rule's f_W = (W's reads - dedup) / (nodedup - dedup); each band SERVED under 1/3, RE-FETCHED over 2/3,
    MIXED between; a GEMM row AGREES when the two bands match."""
    import dataclasses

    import score_cal as SC
    m = "olmoe-1b-7b"
    cells = {c["label"]: c for c in page_doc_any(m, "gaps")["cells"]}
    out = {"verdict": "DESCRIPTIVE", "time": [], "bytes": []}
    rep, why = DC.decode_page(tree, m, "dg", reg_file(m, "gaps"), "GRR")
    if rep is not None and prm is not None and prm.T0_GR_ms is not None:
        meas, _d = DC.measured(rep, m, "gaps")
        for k, row in meas.items():
            c = cells.get(k[2])
            if c is None or k[4] != "native":
                continue
            a = price_cell_v4dp(m, c, prm, fixes=V4DP.ALL)
            b = price_cell_v4dp(m, c, prm, fixes=V4DP.Fixes(naive_order=False))
            out["time"].append({"label": k[2], "meas_us": r6(row["ms_p50"] * 1e3),
                                "v4dp_us": None if a is None else r6(a["ms"] * 1e3),
                                "dedup_us": None if b is None else r6(b["ms"] * 1e3),
                                "e_v4dp": None if a is None else r6(DC.e_dec(a["ms"], row["ms_p50"])),
                                "e_dedup": None if b is None else r6(DC.e_dec(b["ms"], row["ms_p50"]))})
    else:
        out["time_verdict"] = f"NOT READ: {why if rep is None else 'no CAL2 fit'}"
    page, why = SC.counter_page(tree, m, "dgn")
    bare = V4D.DecodeParams(tile=prm.tile) if prm is not None else V4D.DecodeParams()
    nod = dataclasses.replace(bare, naive_dedup=False)
    band = lambda f: "SERVED" if f < M4_CUTS[0] else "RE-FETCHED" if f > M4_CUTS[1] else "MIXED"  # noqa: E731
    for c in (page or {}).get("cells") or []:
        pc = cells.get(c["label"])
        if pc is None or c.get("arm") != RD.NATIVE:
            continue
        w = price_cell_v4dp(m, pc, bare, fixes=V4DP.ALL)
        d0 = price_cell_v4dp(m, pc, bare, fixes=V4DP.NONE)
        d1 = price_cell_v4dp(m, pc, nod, fixes=V4DP.NONE)
        for g in RD.GEMMS:
            got = (c["per_gemm"].get(g) or {}).get("dram__bytes_read.sum")
            lo, hi, wr = d0["reads"][g], d1["reads"][g], w["reads"][g]
            row = {"cell": f"{c['label']}/{g}", "counted": got, "dedup": r6(lo), "nodedup": r6(hi), "v4dp": r6(wr)}
            if got and hi > lo:
                f, fw = (got - lo) / (hi - lo), (wr - lo) / (hi - lo)
                row.update(f=r6(f), f_W=r6(fw), band=band(f), band_W=band(fw), agrees=band(f) == band(fw))
            out["bytes"].append(row)
    if page is None:
        out["bytes_verdict"] = f"NOT READ: {why}"
    rows = [r for r in out["bytes"] if "agrees" in r]
    out["agree"] = f"{sum(r['agrees'] for r in rows)} of {len(rows)} GEMM rows" if rows else "NOT READ"
    return out


def settle_readings(tree: Path, prm) -> dict:
    out = {"label": "DESCRIPTIVE (addendum 3, owner decision 4): registered before any page; never a STOP, never in "
                    "the CAL2 fit, never used to refit v4d' before the CAL2 reading"}
    for k, fn in (("M1", lambda: read_m1(tree, prm)), ("M2", lambda: read_m2(tree, prm)),
                  ("M3", lambda: read_m3(tree)), ("M4", lambda: read_m4(tree, prm))):
        try:
            out[k] = fn() if prm is not None or k == "M3" else {"verdict": "NOT READ: no CAL2 fit"}
        except Exception as exc:                                         # noqa: BLE001 (a reading never stops a score)
            out[k] = {"verdict": f"NOT READ: {type(exc).__name__}: {exc}"}
    return out
