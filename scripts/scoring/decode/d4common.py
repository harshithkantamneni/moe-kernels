#!/usr/bin/env python
"""The decode stage's fourth addendum (docs/registered/2026-11-04-decode-addendum4-gh200): the settling rental DS, a
MEASUREMENT registration only (owner, 2026-11-04, D4-7). One module for the registration builder, the DS reader and
the published-tree reproduction. It imports the frozen decode modules (dcommon, d2common, d3common) and v4d'
(r3_timing_model_v4dp, through d3common) and changes none of them. v4d' enters only as geometry: each GEMM's live CTAs
n, its bytes per CTA b and its compute floor F, as addendum 3 registered them. No timing model is fitted, priced or
frozen here.

THE RULES THIS FILE OWNS:
  DS units   DS's own tile census (the CAL2 plan's tilecensus unit on the six DS models) and six SYNTHETIC counter
             pages on SEEN models (no new blind model, no new blind cell): GEMM ladders on both sides of one wave and
             within a single wave per decode tile, few-CTA long-slab cells, w1 byte-rate pairs, the call kernels on six
             (H, I, k), and the align cells (one loop iteration at E 8, 16, 32 by DECL, 40, 60, 64; 1 to 24 loop
             iterations for the slope; the block-scan points at E 40 and 64)
  stop rules power first, then per page its integrity (registered page, the grid gate, the wide filter, the CAL2
             session's versions); then DS's own tile census: the census gate (missing or a path mismatch is a STOP for
             the per-CTA readings), per tile the occupancy rule, and the cross-check against CAL2's published census
             (reported; a difference is a STOP for the per-CTA readings)
  readings   LADDERS, SUBWAVE, DRAIN, W1RATE, CALLS, ALIGN with BLOCK, and SLOPE: every one DESCRIPTIVE, each computed
             from the pages by a rule registered before any DS page; none is a model term, none is a STOP
  trees      the published CAL2 and decode blind trees reproduce their committed scores (DESCRIPTIVE check)
The decode model form for the new blind is chosen after DS, from DS's and CAL2's pages, and frozen and registered
before any page of the three new blind models; none of that is here.
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
import d3common as D3  # noqa: E402
import dcommon as DC  # noqa: E402

RD, V4, V4D, TM = DC.RD, DC.V4, DC.V4D, DC.V4.TM
REPO = DC.REPO
V4DP = D3.V4DP

#: v4d''s cell price as addendum 3 registered it (its per-GEMM geometry is what the readings take from it)
_PRICE_V4DP = D3.price_cell_v4dp

r6 = DC.r6
DATE = "2026-11-04"
NAME = f"{DATE}-decode-addendum4-gh200"
PAGES4 = f"docs/registered/{DATE}-decode4-pages"
DS_PLAN = "scripts/plans/rentalDS-settle-d4-2026-11.plan"
DS_DRIVER = D3.DRIVER3
CAL2_SCORE = "scripts/scoring/decode/cal2.score.json"
CAL2_PUBLISHED = "results/published/2026-11-03-nvidia_gh200_480gb-decodeCAL2-session"
CAL2_CENSUS = f"{CAL2_PUBLISHED}/results/2026-11-03-nvidia_gh200_480gb-tile-census-mixtral-8x7b-dec/tile-census.json"
BLIND_SCORE = "scripts/scoring/decode/blind.score.json"
BLIND_PUBLISHED = "results/published/2026-11-04-nvidia_gh200_480gb-decodeBlind-session"
FINALISED_D3 = "scripts/scoring/decode/decodeB.finalised.json"
DIAG = "~/code/moe-kernels-work/2026-11-04/diag-blind/DIAGNOSIS.md (offline, read-only against 82364213)"
DIAG_RESIDUAL = "~/code/moe-kernels-work/2026-11-04/diag-residual/DIAGNOSIS.md (offline, read-only against b2fb3402)"

#: the three new blind models (the diagnosis's section 5): named here so DS can be shown not to touch them; no page of
#: theirs is written, generated into the registration or read
NEW_BLIND = ("mixtral-8x22b", "qwen2-57b-a14b", "granite-3.0-1b-a400m")
SEEN_MODELS = DC.CAL_MODELS + DC.BLIND_MODELS

#: the plain statements the registration and the reader carry (owner, 2026-11-04, D4-7)
MODEL_AFTER_DS = ("the decode model form for the new blind will be chosen after DS, from DS's and CAL2's pages; it will "
                  "be frozen and registered before any page of the three new blind models (Mixtral-8x22B, Qwen2-57B-A14B, "
                  "Granite-3.0-1B-A400M), which remain unseen")
SEEN_DISCLOSURE = ("the earlier decode blind pages (Phi-3.5, Qwen1.5, JetMoE, deepseek-v2-lite, Granite-3B; published "
                   "2026-11-04) were seen during the diagnosis: the two offline diagnoses of 2026-11-04 read them, and what "
                   "they showed (a drain at wave boundaries, slow few-CTA GEMMs, per-model call kernels, a per-tile w1 "
                   "rate) chose what DS measures; any later re-score of those pages is a weak sanity floor, never a test")

# --------------------------------------------------------------------------
# the DS pages (SEEN models only)
# --------------------------------------------------------------------------

N, ND = ("native",), ("native", "decl")
DS_LABEL = "ds"
#: per SEEN model: (decl_copies, ((T, A, arms), ...)); A experts 0..A-1 balanced (decode_synth.balanced_over)
DS_SPECS = {
    "jetmoe-8b": (4, ((4, 2, N), (4, 3, N), (4, 4, N), (4, 8, N), (8, 8, ND), (16, 8, ND), (64, 4, N), (64, 8, ND),
                      (96, 6, N), (96, 8, N), (128, 4, N), (128, 8, N))),
    "phi-3.5-moe": (2, ((4, 2, N), (4, 4, N), (8, 8, ND), (16, 16, ND), (64, 4, N), (64, 16, ND), (80, 5, N), (80, 6, N),
                        (96, 6, N), (96, 8, N), (96, 16, N), (128, 4, N), (128, 8, N))),
    "mixtral-8x7b": (2, ((4, 2, ND), (4, 3, N), (4, 4, ND), (16, 8, ND), (64, 4, N), (64, 8, N), (96, 6, N), (96, 7, N),
                         (96, 8, N), (128, 4, N), (128, 8, N))),
    "granite-3.0-3b-a800m": (None, ((2, 8, N), (4, 8, N), (4, 16, N), (4, 32, N), (8, 8, N), (8, 40, N), (64, 16, N),
                                    (64, 40, N), (96, 32, N), (96, 40, N), (120, 40, N), (128, 40, N))),
    "qwen1.5-moe-a2.7b": (None, ((4, 4, N), (4, 16, N), (8, 4, N), (8, 16, N), (8, 32, N), (64, 8, N), (64, 60, N),
                                 (96, 12, N), (96, 16, N), (96, 24, N), (96, 32, N), (128, 8, N), (128, 32, N))),
    "olmoe-1b-7b": (None, ((4, 8, N), (4, 32, N), (8, 8, N), (8, 64, N), (64, 64, N), (128, 16, N), (128, 32, N), (128, 64, N))),
}
#: the cells the v4 build adds (owner, 2026-11-04, D4-7 and D4-8): Phi T 80 A 5 and A 6 and OLMoE T 128 A 16 and A 32 (the
#: owner's four, D4-8, in place of Phi T 96 A 5 and T 128 A 3, whose experts are all past BLOCK_M: a third DRAIN model on
#: 32x128x64w4s3 and 64x128x64w4s3; A 32 gives OLMoE's ladders there a crossing row past one boundary only); Phi T 96 A 16 and Qwen1.5 T 96 A 12 and A 24 (the cheapest SEEN cells that give
#: 32x128x64w4s3 a second and third few-CTA long-slab ladder, which it lacked); Mixtral T 96 A 7 (v4f, owner: a third
#: Mixtral w1 row past the wave on 32x128x64w4s3, so one refused row cannot leave W1RATE there with one model)
DS_ADDED_V4 = (("phi-3.5-moe", 80, 5, "DRAIN"), ("phi-3.5-moe", 80, 6, "DRAIN"), ("olmoe-1b-7b", 128, 16, "DRAIN"),
               ("olmoe-1b-7b", 128, 32, "DRAIN"),
               ("phi-3.5-moe", 96, 16, "SUBWAVE"), ("qwen1.5-moe-a2.7b", 96, 12, "SUBWAVE"),
               ("qwen1.5-moe-a2.7b", 96, 24, "SUBWAVE"), ("mixtral-8x7b", 96, 7, "W1RATE"))
#: the owner's two cells of D4-7 that D4-8 replaced (every expert past BLOCK_M: they entered CALLS and ALIGN only)
DS_REPLACED = (("phi-3.5-moe", 96, 5), ("phi-3.5-moe", 128, 3))
#: plan order; every unit is nodrop: a unit that does not run leaves readings NOT IDENTIFIED (DS is DESCRIPTIVE; the
#: next round chooses the model form from what DS and CAL2 hold)
DS_ORDER = (("jetmoe-8b", "nodrop"), ("phi-3.5-moe", "nodrop"), ("mixtral-8x7b", "nodrop"),
            ("granite-3.0-3b-a800m", "nodrop"), ("qwen1.5-moe-a2.7b", "nodrop"), ("olmoe-1b-7b", "nodrop"))
DS_WHAT = ("DS: the settling rental's counter page (kernels=wide: align, silu_and_mul, moe_sum and the GEMMs) on a SEEN "
           "model, DESCRIPTIVE: GEMM ladders on both sides of one wave and within a single wave, few-CTA long-slab "
           "cells, w1 byte-rate pairs, the call kernels, one-iteration and block-scan align cells (DECL arms declare "
           "E x C)")


def ds_counts(model: str, T: int, A: int) -> list[int]:
    import decode_synth as SY
    return SY.balanced_over(model, int(T), list(range(int(A))))


def ds_cells(model: str) -> list[dict]:
    _dc, spec = DS_SPECS[model]
    return [{"label": f"S-T{T}-A{A}", "T": int(T), "counts": ds_counts(model, T, A), "arms": list(arms),
             "role": "DS-ALIGN" if "decl" in arms else "DS"} for T, A, arms in spec]


def page4_path(model: str, kind: str = DS_LABEL) -> str:
    return f"{PAGES4}/{model}-{kind}.json"


def page4(model: str, kind: str = DS_LABEL) -> dict:
    import decode_synth as SY
    if kind != DS_LABEL:
        raise KeyError(f"addendum 4 registers DS pages only, not {kind}")
    return {"schema": RD.SCHEMA, "model": model, "page": f"{model}-{kind}", "shuffle_seed": SY.SHUFFLE_SEED,
            "decl_copies": DS_SPECS[model][0], "label": f"SYNTHETIC (scripts/scoring/decode/d4common.py, {NAME}); {DS_WHAT}",
            "cells": ds_cells(model)}


PAGE_KEYS = tuple((m, DS_LABEL) for m, _d in DS_ORDER)


def page4_files() -> dict[str, str]:
    return {f"{m}-{k}.json": json.dumps(page4(m, k), indent=1) + "\n" for m, k in PAGE_KEYS}


def page_doc4(model: str, kind: str = DS_LABEL) -> dict:
    return json.loads((REPO / page4_path(model, kind)).read_text())


def reg_file4(model: str, kind: str = DS_LABEL) -> dict:
    p = REPO / page4_path(model, kind)
    doc = json.loads(p.read_text())
    return {"path": page4_path(model, kind), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "shuffle_seed": doc["shuffle_seed"], "decl_copies": doc["decl_copies"],
            "cells": [{"label": c["label"], "T": c["T"], "counts_sha256": RD.counts_sha256(c["counts"]), "role": c["role"],
                       "arms": c["arms"], "shuffle": c.get("shuffle", True)} for c in doc["cells"]]}


# --------------------------------------------------------------------------
# DS units, the DS plan and the rental
# --------------------------------------------------------------------------

def _pairs(model: str) -> int:
    return sum(len(c["arms"]) for c in page4(model)["cells"])


def ds_est(model: str, pairs: int | None = None) -> int:
    """The wide counter route's rule (d3common._cr_est): 0.6 s per profiled launch, warmups profiled (3 calls + 2
    warmups), 6 launches per call at most, per (cell, arm), rounded up, plus 2 min of setup."""
    launches = (_pairs(model) if pairs is None else pairs) * (3 + 2) * D3.WIDE_KERNELS_PER_CALL
    return math.ceil(launches * 0.6 / 60) + 2


#: profiled seconds per added (cell, arm) under the wide route's rule (3 calls + 2 warmups, 6 launches, 0.6 s each)
PAIR_SECONDS = (3 + 2) * D3.WIDE_KERNELS_PER_CALL * 0.6


def ds_units() -> list[dict]:
    out = []
    for m, drop in DS_ORDER:
        est = ds_est(m)
        out.append({"model": m, "step": "decodebytes", "label": DS_LABEL, "page": page4_path(m),
                    "keys": "kernels=wide", "drop": drop, "est": est, "cap": max(15, 3 * est)})
    return out


def ds_line(u: dict) -> str:
    d = "nodrop=1" if u["drop"] == "nodrop" else f"drop-group={u['drop']}"
    return f"{u['model']} decodebytes label={u['label']} histogram={u['page']} {u['keys']} {d} est={u['est']} cap={u['cap']}"


PRELUDE = ("- prelude est=3 cap=15", "mixtral-8x7b calibrate label=dcal nodrop=1 est=2 cap=30")
#: DS's own tile census (owner, 2026-11-04, D4-8): the CAL2 plan's tilecensus unit (compile only, the pinned
#: tile_census.py, est 6 cap 25, nodrop) on the four decode tiles and the six DS models, after calibrate as in CAL2
CENSUS_EST, CENSUS_CAP = 6, 25


def census_line() -> str:
    return (f"- tilecensus label={DS_LABEL} tiles={','.join(DC.DECODE_TILES)} models={','.join(m for m, _d in DS_ORDER)} "
            f"nodrop=1 est={CENSUS_EST} cap={CENSUS_CAP}")


def ds_plan() -> str:
    hdr = ["# THE DECODE SETTLING RENTAL DS under the fourth addendum (2026-11-04): a MEASUREMENT rental, DESCRIPTIVE",
           "# counter pages on SEEN models only (GEMM ladders on both sides of one wave and within a single wave per tile;",
           "# w1 byte-rate pairs; the call kernels; the align kernel across E). It registers no model form and no constant.",
           f"# GENERATED by scripts/scoring/decode/register_decode_d4.py from registration {NAME}; do not edit by hand.",
           "# One gpu_1x_gh200 at $2.29/h, the decode driver's D3 copy (its kernels=wide key):",
           "#   bash scripts/vm_run.sh start --ip <ip> --run-id <id> --deadline <epoch s> \\",
           f"#     --plan {DS_PLAN} --driver {DS_DRIVER}",
           "# Every unit is nodrop. No new blind model is loaded or timed. The tilecensus unit is DS's own census (the",
           "# reader's occupancy stop rule reads it; CAL2's census is a cross-check).",
           "# After DS: read_ds.py (DESCRIPTIVE readings). The model form is chosen in the next round, from DS and CAL2."]
    return "\n".join(hdr + list(PRELUDE) + [census_line()] + [ds_line(u) for u in ds_units()]) + "\n"


def rentals() -> dict:
    ds = (sum(int(x.split("est=")[1].split()[0]) for x in PRELUDE) + CENSUS_EST
          + sum(u["est"] for u in ds_units()))
    return {"ds": D2.rental(ds)}


def cell_check(model: str, T: int, A: int, prm=None) -> dict:
    """What one cell does (COMPUTED from the routing and v4d''s geometry): rows per expert against the tile's BLOCK_M, the
    align path, and per GEMM n live CTAs against S with its wave class and bytes per CTA; whether it enters the GEMM
    readings (no expert past BLOCK_M, not the naive path)."""
    from moe.spec import MODEL_CONFIGS
    counts = ds_counts(model, T, A)
    live = [int(v) for v in counts if int(v)]
    g = RD.geometry(MODEL_CONFIGS[model], int(T), counts)
    recs = gemm_records(model, {"label": f"S-T{T}-A{A}", "T": int(T), "counts": counts, "arms": [RD.NATIVE]},
                        prm or ds_prm()) or {}
    return {"model": model, "cell": f"S-T{T}-A{A}", "tile": g.tile, "BLOCK_M": g.bm, "path": g.path,
            "rows_per_expert": [min(live), max(live)], "experts_over_BLOCK_M": g.multi_tile_experts,
            "enters_gemm_readings": g.path != RD.NAIVE and not g.multi_tile_experts,
            "gemms": {gm: {"n": r["n"], "S": r["S"], "wave": wave_class(r["n"], r["S"]) if r["S"] else None,
                           "b_MB": r6(r["b"] / 1e6)} for gm, r in recs.items()}}


def ds_added() -> dict:
    """What the v4 cells and DS's own census add (COMPUTED): per added cell its pairs, profiled seconds and check
    (cell_check); per unit the estimate before and after; the units' minutes, wall and cost before (v3's cells, no DS
    census), with the cells, and with the cells and the census, at $2.29/h."""
    cells, per_unit = [], {}
    for m, T, A, why in DS_ADDED_V4:
        cells.append(dict(cell_check(m, T, A), why=why, pairs=1, profiled_s=PAIR_SECONDS))
        per_unit[m] = per_unit.get(m, 0) + 1
    units = {m: {"est (before, after)": [ds_est(m, _pairs(m) - k), ds_est(m)]} for m, k in per_unit.items()}
    after = rentals()["ds"]
    cells_only = D2.rental(after["units_min"] - CENSUS_EST)
    before = D2.rental(cells_only["units_min"] - sum(x["est (before, after)"][1] - x["est (before, after)"][0]
                                                     for x in units.values()))
    added_s = PAIR_SECONDS * len(DS_ADDED_V4)
    return {"cells": cells, "units": units,
            "units min (before, with the cells, with the cells and the census)": [
                before["units_min"], cells_only["units_min"], after["units_min"]],
            "wall min (same)": [before["wall_min"], cells_only["wall_min"], after["wall_min"]],
            "cost at $2.29/h (same)": [before["cost_usd"], cells_only["cost_usd"], after["cost_usd"]],
            "unit min added (cells, census)": [cells_only["units_min"] - before["units_min"], CENSUS_EST],
            "cost added at $2.29/h": r6(after["cost_usd"] - before["cost_usd"]),
            "profiled time added by the cells": f"{added_s:.0f} s (about ${added_s / 0.84 / 3600 * D2.RATE:.2f} at "
                                                "$2.29/h before the estimator's rounding)"}


# --------------------------------------------------------------------------
# small-batch align iterations and the call kernels' regressors (geometry, not a model)
# --------------------------------------------------------------------------

#: the call kernel classes the wide filter attributes (the align class is ALIGN's)
CALL_CLASSES = ("silu_and_mul", "moe_sum")
#: the decode stage's activation bytes (bf16)
DTYPE_BYTES = 2


def iterations(numel: int, E: int) -> int:
    """ceil(numel / max(E, 32)): the small-batch align kernel's per-thread loop count."""
    return math.ceil(int(numel) / max(int(E), 32))


def call_x(cfg, T: int, cls: str) -> tuple[float, float]:
    """(x, B) of one call class at T tokens from the model's H, I, k: silu_and_mul x = I, B = T k 3 I x 2 bytes;
    moe_sum x = k H, B = (T k H + T H) x 2 bytes (the CALLS reading's DESCRIPTIVE summary line)."""
    numel = int(T) * cfg.top_k
    if cls == "silu_and_mul":
        return float(cfg.intermediate_size), float(numel * 3 * cfg.intermediate_size * DTYPE_BYTES)
    if cls == "moe_sum":
        return float(cfg.top_k * cfg.hidden_size), float((numel * cfg.hidden_size + int(T) * cfg.hidden_size) * DTYPE_BYTES)
    raise KeyError(cls)


def align_iterations(model: str, page: dict | None = None, arms: tuple = ("native", "decl")) -> dict[int, list[int]]:
    """{declared E: sorted small-batch align-loop iteration counts} over a page's (cell, arm) (COMPUTED)."""
    from moe.spec import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model]
    doc = page or page4(model)
    out: dict[int, set] = {}
    for c in doc["cells"]:
        for arm in c.get("arms") or ["native"]:
            if arm not in arms:
                continue
            E = cfg.num_experts * (int(doc.get("decl_copies") or 1) if arm == "decl" else 1)
            g = RD.geometry(cfg, int(c["T"]), c["counts"], None if arm == "native" else E)
            if g.path == RD.SMALL_BATCH:
                out.setdefault(E, set()).add(iterations(g.numel, E))
    return {E: sorted(v) for E, v in sorted(out.items())}


def ds_iterations() -> dict[int, list[int]]:
    """{declared E: the iteration counts DS's small-batch (cell, arm) reach}, over the six DS pages (COMPUTED)."""
    out: dict[int, set] = {}
    for m, _d in DS_ORDER:
        for E, its in align_iterations(m).items():
            out.setdefault(E, set()).update(its)
    return {E: sorted(v) for E, v in sorted(out.items())}


def blind_reach() -> dict:
    """COMPUTED from the new blind models' CONFIGS and decode_synth's frozen main-page generator (in memory, never written
    or registered as a page): the most small-batch align iterations a new blind main page would reach, and each call
    class's x against DS's range. Neither is a page of those models; the next round reads them when it chooses a form."""
    import decode_synth as SY

    from moe.spec import MODEL_CONFIGS
    its = {m: max((i for v in align_iterations(m, SY.page(m, "main"), ("native",)).values() for i in v), default=None)
           for m in NEW_BLIND}
    calls = {}
    for c in CALL_CLASSES:
        ds = [call_x(MODEL_CONFIGS[m], 1, c)[0] for m, _d in DS_ORDER]
        calls[c] = {"DS x range": [min(ds), max(ds)], "new": {m: call_x(MODEL_CONFIGS[m], 1, c)[0] for m in NEW_BLIND}}
    return {"small-batch align iterations (most, main page)": its, "call class x": calls}


# --------------------------------------------------------------------------
# per-GEMM records of a cell (the geometry the readings take from v4d')
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _recording(store: list):
    orig = TM.gemm_ms

    def gemm_ms(win, gemm, sigma, c_ns, bw, sms, p=None, occ=None):
        n = int(win.live)
        S = int(sms * occ) if occ else None
        b = (sigma * float(win.reads) + TM.OUT_TILE * n) / n if n else 0.0
        store.append({"gemm": gemm, "n": n, "S": S, "b": b, "sms": sms,
                      "F_us": math.ceil(n / sms) * TM.floor_ksteps(gemm) * c_ns * 1e-3 if n else 0.0})
        return orig(win, gemm, sigma, c_ns, bw, sms, p, occ)
    TM.gemm_ms = gemm_ms
    try:
        yield
    finally:
        TM.gemm_ms = orig


def gemm_records(model: str, cell: dict, prm) -> dict | None:
    """{gemm: record} of one NATIVE cell's geometry under v4d' (n live CTAs, S, b bytes per CTA, F floor us), or None
    when v4d' does not price it."""
    store: list = []
    with _recording(store):
        p = _PRICE_V4DP(model, cell, prm, fixes=V4DP.ALL)
    if p is None:
        return None
    return {r["gemm"]: r for r in store}


# --------------------------------------------------------------------------
# the stop rules (registered before any DS page)
# --------------------------------------------------------------------------

#: the CAL2 session's software (MEASURED: the manifests of its counter pages, 2026-11-03), which its pages and its tile
#: census ran under; a DS page under other versions is NOT READ (DS's pages, its census and CAL2's cross-check would
#: not be the same kernels). python is compared on major.minor (a patch release builds no kernel); its patch level is
#: recorded and printed, DESCRIPTIVE. torch, triton and vllm are compared exactly
CAL2_VERSIONS = {"python": "3.12.16", "torch": "2.13.0", "triton": "3.7.1", "vllm": "0.27.1"}
#: the keys PAGE compares on major.minor only
VERSIONS_MAJOR_MINOR = ("python",)


def _major_minor(v) -> str | None:
    return None if v is None else ".".join(str(v).split(".")[:2])


def versions_match(got: dict) -> bool:
    """PAGE's version clause: python on major.minor, every other key exact."""
    return all((_major_minor(got.get(k)) == _major_minor(w)) if k in VERSIONS_MAJOR_MINOR else got.get(k) == w
               for k, w in CAL2_VERSIONS.items())


def page_check(model: str, page: dict | None) -> dict:
    """PAGE (stop rule, per DS page): READ when the counter page's manifest names the registered DS page's sha256, its
    V_ALIGN_grid gate reads PASS, its kernel filter is wide, and its versions are CAL2_VERSIONS (python on major.minor,
    its patch level recorded as python_patch, DESCRIPTIVE); else NOT READ (no row of the page enters any reading).
    Refused cells (gates.refused_cells) are left out either way."""
    if page is None:
        return {"verdict": "NOT READ: no page"}
    why = []
    want = hashlib.sha256((REPO / page4_path(model)).read_bytes()).hexdigest()
    got = (page.get("manifest") or {}).get("page_sha256")
    if got != want:
        why.append(f"page_sha256 {got} is not the registered page's {want}")
    if not str((page.get("gates") or {}).get("V_ALIGN_grid", "")).startswith("PASS"):
        why.append(f"V_ALIGN_grid reads {(page.get('gates') or {}).get('V_ALIGN_grid')!r}")
    if page.get("kernels") != "wide":
        why.append(f"kernels reads {page.get('kernels')!r}, not wide")
    v = {k: ((page.get("manifest") or {}).get("versions") or {}).get(k) for k in CAL2_VERSIONS}
    if not versions_match(v):
        why.append(f"versions {v} are not the CAL2 session's {CAL2_VERSIONS} (python on major.minor)")
    refused = sorted((page.get("gates") or {}).get("refused_cells") or [])
    return {"verdict": "READ" if not why else "NOT READ: " + "; ".join(why), "refused_cells": refused,
            "python_patch": {"label": "DESCRIPTIVE", "page": v.get("python"), "CAL2": CAL2_VERSIONS["python"]}}


def census_check(prm, census: dict) -> dict:
    """CENSUS (stop rule, per decode tile; the occupancy rule, read on DS's own census): the census must read, for every
    DS model's w1 and w2, the occupancy the readings' S uses (the registered census occupancy: 3 CTAs per SM on
    16x64x128w4s4 and 64x128x64w4s3, 4 on the two s3 mma.sync tiles); else every S-dependent reading on that tile
    (LADDERS, SUBWAVE, DRAIN, W1RATE) is NOT READ there. The same rule reads CAL2's census as the cross-check."""
    out = {}
    for t in DC.DECODE_TILES:
        by = ((census.get("summary") or {}).get(t) or {}).get("by") or {}
        occ = (prm.tile.occupancy or {}).get(t) or {}
        bad = [f"{m}|{g}: census {by.get(f'{m}|{g}')} against {occ.get(g)}" for m, _d in DS_ORDER for g in RD.GEMMS
               if by.get(f"{m}|{g}") is None or int(by[f"{m}|{g}"]) != int(occ.get(g, -1))]
        out[t] = {"verdict": "READ" if not bad else "NOT READ: " + "; ".join(bad), "occupancy": dict(occ)}
    return out


def ds_census(tree: Path) -> tuple[dict | None, str]:
    """DS's own tile census: <date>-<card>-tile-census-<model>-ds/tile-census.json under the tree (exactly one)."""
    hits = sorted(Path(tree).rglob(f"*-{DC.CARD}-tile-census-*-{DS_LABEL}/tile-census.json"))
    if len(hits) != 1:
        return None, f"{len(hits)} DS census files"
    return json.loads(hits[0].read_text()), str(hits[0])


def census_gate(census: dict | None, why: str = "") -> dict:
    """The census gate CAL used (decodeC-cal's census rule, the tile stage's build-tile-review 9), on DS's own census: a
    census that is missing or refused (an MMA path mismatch) is a STOP for the per-CTA readings (LADDERS, SUBWAVE,
    DRAIN, W1RATE), never a fallback to another census."""
    if census is None:
        return {"verdict": f"STOP: no DS census ({why}); a missing census is a STOP"}
    bad = list(census.get("path_mismatch") or [])
    if bad:
        return {"verdict": "STOP: REFUSED (path mismatch): " + "; ".join(map(str, bad))}
    return {"verdict": f"READ ({census.get('verdict')})", "errors": list(census.get("errors") or [])}


#: the compile record a census entry carries, compared entry by entry in the cross-check
CENSUS_FIELDS = ("regs", "shared", "num_warps", "mma", "occupancy", "ptx_counts")


def census_cross(prm, ds: dict | None, cal2: dict) -> dict:
    """The cross-check (reported, not gating; owner, 2026-11-04, D4-8): CAL2's published census under the occupancy rule
    (census_check), and DS's census against CAL2's entry by entry for every DS model, decode tile and GEMM (the config,
    and CENSUS_FIELDS). SAME, or DIFFERS with every difference listed; DIFFERS is a STOP for the per-CTA readings."""
    out = {"CAL2 census (occupancy rule, reported)": {t: x["verdict"] for t, x in census_check(prm, cal2).items()}}
    if ds is None:
        return dict(out, verdict="NOT READ: no DS census", differences=[])
    models = {m for m, _d in DS_ORDER}

    def keyed(doc):
        return {(e["model"], e["tile"], e["gemm"], json.dumps(e.get("config"))): e for e in doc.get("entries") or []
                if e.get("model") in models and e.get("tile") in DC.DECODE_TILES}
    a, b = keyed(ds), keyed(cal2)
    diff = []
    for k in sorted(set(a) | set(b)):
        name = f"{k[0]} {k[1]} {k[2]} config {k[3]}"
        if k not in a or k not in b:
            diff.append(f"{name}: only in {'CAL2' if k not in a else 'DS'}'s census")
            continue
        for f in CENSUS_FIELDS:
            if a[k].get(f) != b[k].get(f):
                diff.append(f"{name}: {f} DS {json.dumps(a[k].get(f))} against CAL2 {json.dumps(b[k].get(f))}")
    return dict(out, verdict="SAME" if not diff else "DIFFERS", entries_compared=len(set(a) & set(b)), differences=diff)


# --------------------------------------------------------------------------
# the DS readings (registered before any DS page; every one DESCRIPTIVE)
# --------------------------------------------------------------------------

#: the counters' DRAM rate (fix R's MEASURED rate, the published CAL dn pages): Dc = B / this, except w1 on a tile whose
#: W1RATE is read (its own c_t)
COUNTER_GBPS = V4DP.W1_COUNTER_GBPS
#: v4d's bandwidth (the card's 3577.96 GB/s): W1RATE reports its rate against this as a ratio, DESCRIPTIVE
V4D_GBPS = V4DP.W1_V4D_GBPS
#: IDENTIFIED rules (geometric, on the pages' rows): a reading on a tile is IDENTIFIED with at least this many qualifying
#: ladders from at least this many models; else NOT IDENTIFIED (reported, never a STOP, never a fallback)
MIN_LADDERS, MIN_MODELS = 2, 2
#: SUBWAVE: a qualifying ladder holds two or more sub-wave rows (n <= S) whose n span at least this factor, with bytes
#: per CTA b of at least LONG_SLAB_BYTES (long slab); its smallest n is its few-CTA row
SUBWAVE_SPAN = 2.0
LONG_SLAB_BYTES = 0.5e6
#: SUBWAVE's registered grid of per-CTA rate ceilings: no ceiling, then 10 x 1.02^j GB/s for j = 0 .. 232 (10 to 990
#: GB/s); ties go to the first minimum in that order
R_CTA_GRID = (None,) + tuple(10.0 * 1.02 ** j for j in range(233))
#: W1RATE: a ladder (model, T) is usable with two or more crossing w1 rows whose counted bytes span at least this
W1RATE_MIN_MB = 100.0
#: CALLS: the DESCRIPTIVE summary line per class is fitted with rows from at least this many models and distinct x
CALL_MIN_MODELS = 3
#: BLOCK: block-scan points at at least this many declared E
BLOCK_MIN_E = 2
#: ALIGN: the DECL points (E 32) enter only when DECL transfers: Mixtral's DECL E 16 median within this of Phi's native E 16
DECL_TRANSFER_US = 1.0
DECL_CHECK_E = 16
#: SLOPE: READ when at least this many declared E each hold two or more iteration counts; the check against 0 calls
#: |s| / SE >= 2 DIFFERS FROM 0
SLOPE_MIN_E = 2
SLOPE_T_CUT = 2.0


#: ROW SANITY (registered before any DS page): a GEMM row is refused unless its L and B are finite and positive and B
#: lies within B_BAND x n b (n live CTAs x v4d''s bytes per CTA: 0.5 to 2, a factor of 2 either way). The band is a
#: garbled-row guard, not a fit: on CAL2's SEEN counter pages B / n b reads 0.94 to 1.00 (POST-HOC, the stand-in)
B_BAND = (0.5, 2.0)
#: ROW SANITY's rate ceiling (registered before any DS page): a GEMM row whose implied DRAM rate B / L is above this is
#: refused, and W1RATE's c_t above it reads NOT IDENTIFIED. The ceiling is the GH200's HBM3 spec peak, 4,022 GB/s (owner,
#: 2026-11-04): no DRAM read can run faster, so a row past it holds garbled bytes or a garbled time. The published decode
#: counter pages never exceed 3,768 GB/s (MEASURED: 3767.6 GB/s, Mixtral dn L-T4-A8 w1, CAL2 2026-11-03, over 272 GEMM
#: rows), 6.3% under the spec peak
RATE_CEILING_GBPS = 4022.0


def _finite_pos(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x)) and float(x) > 0.0
    except (TypeError, ValueError):
        return False


def row_sanity(L_ns, B, n: int, b: float) -> str | None:
    """Why a GEMM row is refused (ROW SANITY), or None: L or B missing, non-finite or not positive, B outside
    B_BAND x n b, or an implied DRAM rate B / L above RATE_CEILING_GBPS (L in ns, so B / L is GB/s)."""
    if not _finite_pos(L_ns):
        return f"time {L_ns!r} is not finite and positive"
    if not _finite_pos(B):
        return f"bytes {B!r} are not finite and positive"
    ratio = float(B) / (n * b)
    if not B_BAND[0] <= ratio <= B_BAND[1]:
        return f"bytes {float(B):.6g} are {ratio:.4g} x n b, outside {B_BAND[0]:g} to {B_BAND[1]:g}"
    rate = float(B) / float(L_ns)
    if rate > RATE_CEILING_GBPS:
        return f"implied DRAM rate {rate:.6g} GB/s is above the {RATE_CEILING_GBPS:g} GB/s ceiling"
    return None


def _ds_page(tree: Path, model: str):
    import score_cal as SC
    return SC.counter_page(tree, model, DS_LABEL)


def wave_class(n: int, S: int) -> str:
    return "sub-wave" if n <= S else "crossing" if n < 2 * S else "multi-wave"


def gemm_rows(prm, pages: dict | None = None, tiles_read: set | None = None,
              refused_rows: list | None = None) -> list[dict]:
    """Per GEMM of every DS NATIVE cell on the small-batch or block-scan path with no expert past BLOCK_M: n live CTAs,
    S = SMs x the census occupancy, b = v4d''s bytes per CTA, F = the GEMM's compute floor (us). With `pages`
    ({model: (counter page, why)}), only cells the page holds and does not refuse, with B = the counted dram__bytes_read
    and L = the ncu gpu__time_duration in us; without, the registered pages' geometry (COMPUTED, for the identification
    check). `tiles_read`: the tiles whose CENSUS reads READ (None: every tile). Rows that fail ROW SANITY are left out
    and appended to `refused_rows` (model, label, GEMM, why)."""
    from moe.spec import MODEL_CONFIGS
    rows = []
    for m, _drop in DS_ORDER:
        cfg = MODEL_CONFIGS[m]
        cells = {c["label"]: c for c in page4(m)["cells"]}
        if pages is None:
            got = [{"label": lab, "arm": RD.NATIVE, "per_gemm": {}} for lab, c in cells.items() if RD.NATIVE in c["arms"]]
        else:
            page = (pages.get(m) or (None, None))[0]
            refused = set(((page or {}).get("gates") or {}).get("refused_cells") or [])
            got = [c for c in (page or {}).get("cells") or [] if c.get("arm") == RD.NATIVE and c.get("label") in cells
                   and c.get("key") not in refused]
        for c in got:
            pc = cells[c["label"]]
            g = RD.geometry(cfg, int(pc["T"]), pc["counts"])
            if g.path == RD.NAIVE or g.multi_tile_experts or (tiles_read is not None and g.tile not in tiles_read):
                continue
            recs = gemm_records(m, pc, prm)
            if recs is None:
                continue
            for gm in RD.GEMMS:
                r = recs[gm]
                if r["S"] is None or not r["n"]:
                    continue
                row = {"model": m, "label": c["label"], "gemm": gm, "tile": g.tile, "T": g.T, "A": g.A, "path": g.path,
                       "n": r["n"], "S": r["S"], "cross": r["n"] > r["S"], "wave": wave_class(r["n"], r["S"]),
                       "b_bytes": r6(r["b"]), "F_us": r6(r["F_us"]), "ladder": f"{m}/T{g.T}/{gm}"}
                if pages is not None:
                    pg = (c.get("per_gemm") or {}).get(gm) or {}
                    L, B = pg.get("gpu__time_duration.sum"), pg.get("dram__bytes_read.sum")
                    why = row_sanity(L, B, r["n"], r["b"])
                    if why is not None:
                        if refused_rows is not None:
                            refused_rows.append({"model": m, "label": c["label"], "gemm": gm, "why": why})
                        continue
                    row.update(L_us=r6(L / 1e3), B_bytes=float(B))
                rows.append(row)
    return rows


def _lstsq(X, y, w=None):
    """Weighted least squares; non-finite inputs or a solver failure give NaN (the fits then read NOT IDENTIFIED)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if not (np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(w).all()):
        return np.full(X.shape[1], np.nan), np.full(len(y), np.nan), None
    try:
        beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan), np.full(len(y), np.nan), None
    res = (y - X @ beta) * w
    dof = len(y) - X.shape[1]
    se = None
    if dof > 0:
        cov = np.linalg.pinv((X * w[:, None]).T @ (X * w[:, None])) * float(res @ res) / dof
        se = [math.sqrt(max(0.0, float(v))) for v in np.diag(cov)]
    return beta, res, se


def _finite(*xs) -> bool:
    """Every value finite (scalars, arrays, lists)."""
    return all(np.isfinite(np.asarray(x, float)).all() for x in xs if x is not None)


NONFINITE = "NOT IDENTIFIED: the fit is not finite"


def _ladders(rows: list[dict]) -> dict:
    lad = {}
    for r in rows:
        lad.setdefault(r["ladder"], []).append(r)
    return lad


def mixed_ladders(rows: list[dict]) -> list[str]:
    """Ladders (model, T, GEMM) on both sides of one wave: a row with n <= S and a row with n > S."""
    return sorted(k for k, v in _ladders(rows).items() if any(not x["cross"] for x in v) and any(x["cross"] for x in v))


def subwave_ladders(rows: list[dict]) -> list[str]:
    """Ladders within a single wave that qualify for SUBWAVE: two or more sub-wave rows, n spanning SUBWAVE_SPAN or more,
    b at least LONG_SLAB_BYTES on every sub-wave row (long slab)."""
    out = []
    for k, v in _ladders(rows).items():
        sub = [x for x in v if not x["cross"]]
        if len(sub) >= 2 and max(x["n"] for x in sub) >= SUBWAVE_SPAN * min(x["n"] for x in sub) \
                and min(x["b_bytes"] for x in sub) >= LONG_SLAB_BYTES:
            out.append(k)
    return sorted(out)


def _models_of(rows: list[dict], ladders: list[str]) -> list[str]:
    lad = _ladders(rows)
    return sorted({lad[k][0]["model"] for k in ladders})


def read_ladders(rows: list[dict], w1rate: dict) -> dict:
    """LADDERS (DESCRIPTIVE): per decode tile, every ladder (model, T, GEMM) with its rows by n: n, S, the wave class
    (sub-wave n <= S, crossing S < n < 2S, multi-wave n >= 2S), b, B, L, Dc = B / c (c as dc_rows) and the excess
    X = L - max(Dc, F); and which ladders are MIXED (both sides of one wave) and SUBWAVE (within one wave, few-CTA
    long-slab). NO FALLBACK: on a tile where W1RATE reads no rate, w1 rows show Dc and X none, and the tile says so."""
    rows = dc_rows(rows, w1rate)
    out = {}
    for t in DC.DECODE_TILES:
        tr = [r for r in rows if r["tile"] == t]
        lad = _ladders(tr)
        out[t] = {"ladders": {k: [{q: x[q] for q in ("label", "A", "n", "S", "wave", "b_bytes", "B_bytes", "L_us", "Dc_us",
                                                    "F_us", "X_us")} for x in sorted(v, key=lambda x: x["n"])]
                              for k, v in sorted(lad.items())},
                  "mixed": mixed_ladders(tr), "subwave": subwave_ladders(tr), "verdict": "READ",
                  "w1_unpriced": (f"w1 rows not priced (Dc, X none): W1RATE reads {_w1_rate(w1rate, t)[1]!r}; {NO_FALLBACK}"
                                  if any(x.get("w1_unpriced") for x in tr) else None)}
    return {"label": "DESCRIPTIVE: the GEMM ladders as measured; no model", "tiles": out}


def fit_w1rate(rows: list[dict]) -> dict:
    """W1RATE on one tile (see read_w1rate): OLS of L on the counted bytes with an offset per usable ladder."""
    lad = {}
    for r in rows:
        if r["gemm"] == "w1" and r["cross"]:
            lad.setdefault(f"{r['model']}/T{r['T']}", []).append(r)
    use = {k: v for k, v in sorted(lad.items())
           if len(v) >= 2 and (max(x["B_bytes"] for x in v) - min(x["B_bytes"] for x in v)) / 1e6 >= W1RATE_MIN_MB}
    models = sorted({x["model"] for v in use.values() for x in v})
    if len(use) < MIN_LADDERS or len(models) < MIN_MODELS:
        return {"verdict": f"NOT IDENTIFIED: {len(use)} usable w1 ladders from {len(models)} models (needs {MIN_LADDERS} "
                           f"from {MIN_MODELS})", "ladders": sorted(use), "models": models}
    names = sorted(use)
    rs = [x for k in names for x in use[k]]
    X = [[1.0 if j == names.index(f"{x['model']}/T{x['T']}") else 0.0 for j in range(len(names))] + [x["B_bytes"]]
         for x in rs]
    beta, res, se = _lstsq(X, [x["L_us"] for x in rs])
    if not _finite(beta, res):
        return {"verdict": NONFINITE, "ladders": names, "models": models, "counter_GBps": None, "rows": len(rs)}
    u = float(beta[-1])
    if u <= 0:
        return {"verdict": f"NOT IDENTIFIED: the marginal time per byte reads {u:.3e} us, not positive (no rate)",
                "ladders": names, "models": models, "counter_GBps": None, "ratio_to_v4d": None, "rows": len(rs)}
    c = 1e-3 / u
    if c > RATE_CEILING_GBPS:
        return {"verdict": f"NOT IDENTIFIED: c {c:.6g} GB/s is above the {RATE_CEILING_GBPS:g} GB/s ceiling",
                "ladders": names, "models": models, "counter_GBps": None, "ratio_to_v4d": None, "rows": len(rs)}
    return {"verdict": "READ: OLS with one offset per ladder", "counter_GBps": r6(c), "ratio_to_v4d": r6(V4D_GBPS / c),
            "ladders": names, "models": models, "rows": len(rs), "offsets_us": {k: r6(float(beta[q])) for q, k in enumerate(names)},
            "se_counter_GBps": None if se is None else r6(c * se[-1] / u),
            "rms_us": r6(float(math.sqrt(sum(float(v) ** 2 for v in res) / len(rs))))}


def read_w1rate(rows: list[dict]) -> dict:
    """W1RATE (DESCRIPTIVE; the per-tile w1 byte-rate pairs). Rows: DS's w1 GEMM rows (gemm_rows) on each decode tile
    whose n live CTAs cross at least one wave boundary (n > S; a multi-wave row, n >= 2S, crosses more than one, so the
    ladder's offset holds a boundary cost only as far as its rows cross the same number of boundaries, and any
    difference falls into u_t). A ladder is one (model, T); it is usable with two or more rows whose counted bytes
    span at least 100 MB (fix R's cut). FORMULA per tile t: L = c0_ladder + B u_t by ordinary least squares over the
    usable ladders' rows, c_t = 1 / u_t the counters' marginal w1 rate (GB/s); beside, c_t's ratio to v4d's 3577.96
    GB/s. IDENTIFIED with at least 2 usable ladders from at least 2 models. On 16x64x128w4s4 it re-reads fix R."""
    tiles = {t: fit_w1rate([r for r in rows if r["tile"] == t]) for t in DC.DECODE_TILES}
    return {"tiles": tiles, "label": "DESCRIPTIVE: w1's counter marginal rate per decode tile; no model term"}


#: NO FALLBACK (registered before any DS page, every reading): w1's bytes are priced only at W1RATE's c_t on the row's
#: tile. On a tile where W1RATE reads no rate, a w1 row is not priced (Dc and X None, w1_unpriced), and a reading whose
#: rows hold such a row reads NOT IDENTIFIED (SUBWAVE, DRAIN) or shows it unpriced (LADDERS). 3895.49 GB/s is never a
#: fallback for w1; it is the counters' MEASURED rate (fix R) for w2, the only rate w2 has
NO_FALLBACK = "w1's bytes are never priced at a fallback rate (NO FALLBACK)"


def _w1_rate(w1rate: dict, tile: str) -> tuple[float | None, str]:
    x = (w1rate.get("tiles") or {}).get(tile) or {}
    return (float(x["counter_GBps"]) if x.get("counter_GBps") else None), str(x.get("verdict", "absent"))


def dc_rows(rows: list[dict], w1rate: dict) -> list[dict]:
    """Rows with Dc = B / c and the excess X = L - max(Dc, F): c is W1RATE's c_t on the row's tile for w1 and 3895.49
    GB/s (the counters' MEASURED rate) for w2. NO FALLBACK: a w1 row on a tile where W1RATE reads no rate gets Dc, X
    and c None and w1_unpriced True."""
    out = []
    for r in rows:
        c = COUNTER_GBPS
        if r["gemm"] == "w1":
            c, _why = _w1_rate(w1rate, r["tile"])
        dc = r["B_bytes"] / c / 1e3 if "B_bytes" in r and c else None
        out.append(dict(r, Dc_us=None if dc is None else r6(dc), c_GBps=None if c is None else r6(c),
                        X_us=None if dc is None else r6(r["L_us"] - max(dc, r["F_us"])),
                        w1_unpriced=r["gemm"] == "w1" and c is None))
    return out


def _unpriced(rows: list[dict], w1_why: str, which: str) -> str | None:
    """The NO FALLBACK verdict when a fit's rows hold an unpriced w1 row, else None."""
    n = sorted({x["ladder"] for x in rows if x.get("w1_unpriced")})
    if not n:
        return None
    return (f"NOT IDENTIFIED: the {which} ladders hold w1 rows ({len(n)} ladders) and W1RATE on this tile reads "
            f"{w1_why!r}; {NO_FALLBACK}")


def _ceil_us(x: dict, r) -> float:
    return max(x["Dc_us"], x["F_us"], 0.0 if r is None else x["b_bytes"] / (r * 1e3))


def fit_subwave(rows: list[dict], w1_why: str = "") -> dict:
    """SUBWAVE on one tile (see read_subwave). NO FALLBACK: NOT IDENTIFIED when the fitted ladders hold an unpriced w1
    row."""
    q = subwave_ladders(rows)
    models = _models_of(rows, q)
    ident = len(q) >= MIN_LADDERS and len(models) >= MIN_MODELS
    lad = _ladders([x for x in rows if not x["cross"]])
    fit = sorted(k for k, v in lad.items() if len(v) >= 2)
    base = {"qualifying_ladders": q, "models": models, "fit_ladders": fit}
    if not ident:
        return dict(base, verdict=f"NOT IDENTIFIED: {len(q)} few-CTA long-slab ladders from {len(models)} models (needs "
                                  f"{MIN_LADDERS} from {MIN_MODELS})", r_cta_GBps=None)
    use = [x for k in fit for x in lad[k]]
    nf = _unpriced(use, w1_why, "fitted")
    if nf:
        return dict(base, verdict=nf, r_cta_GBps=None)
    X = np.zeros((len(use), len(fit)))
    for i, x in enumerate(use):
        X[i, fit.index(x["ladder"])] = 1.0
    L = np.array([x["L_us"] for x in use], float)
    w = 1.0 / L
    best = None
    for r in R_CTA_GRID:
        y = L - np.array([_ceil_us(x, r) for x in use])
        _b, res, _s = _lstsq(X, y, w)
        obj = float(res @ res)
        if not math.isfinite(obj):
            continue
        if best is None or obj < best[0] - 1e-15:
            best = (obj, r, res, _b)
    y0 = L - np.array([_ceil_us(x, None) for x in use])
    _b0, res0, _s0 = _lstsq(X, y0, w)
    if best is None or not _finite(best[2], best[3], res0):
        return dict(base, verdict=NONFINITE, r_cta_GBps=None)
    obj, r, res, beta = best
    binding = [f"{x['ladder']} n {x['n']}" for x in use if r is not None and x["b_bytes"] / (r * 1e3) > max(x["Dc_us"], x["F_us"])]
    return dict(base, verdict="READ: grid over the per-CTA ceiling, weighted least squares, one offset per ladder"
                + ("; no ceiling binds" if r is None else ""),
                r_cta_GBps=None if r is None else r6(r), rows=len(use), binding_rows=binding,
                offsets_us={k: r6(float(beta[i])) for i, k in enumerate(fit)},
                rms_rel=r6(float(np.sqrt(np.mean(res ** 2)))), rms_rel_no_ceiling=r6(float(np.sqrt(np.mean(res0 ** 2)))))


def read_subwave(rows: list[dict], w1rate: dict) -> dict:
    """SUBWAVE (DESCRIPTIVE; few-CTA long-slab cells within a single wave). Rows: dc_rows, sub-wave rows only (n <= S).
    A ladder (model, T, GEMM) enters the fit with two or more sub-wave rows. FORMULA per decode tile t: r_cta,t minimises
    sum ((c0 + max(Dc, F, b / r) - L) / L)^2 over those rows, every c0 free (weighted least squares at each r), r over
    the registered grid (no ceiling, then 10 x 1.02^j GB/s, 10 to 990), ties to the first minimum; r_cta,t is the
    per-CTA streaming rate a few-CTA GEMM runs at when its CTAs cannot fill the DRAM pipe. IDENTIFIED with at least 2
    qualifying ladders (sub-wave rows spanning n by 2 or more, every b at least 0.5 MB) from at least 2 models; else NOT
    IDENTIFIED, no fit. Beside: the binding rows, the offsets, the relative rms at the fit and with no ceiling."""
    rows = dc_rows(rows, w1rate)
    return {"tiles": {t: fit_subwave([r for r in rows if r["tile"] == t], _w1_rate(w1rate, t)[1]) for t in DC.DECODE_TILES},
            "label": "DESCRIPTIVE: the per-CTA rate a few-CTA long-slab GEMM runs at within one wave; no model term"}


def fit_drain(rows: list[dict], r_cta=None, w1_why: str = "") -> dict:
    """DRAIN on one tile (see read_drain): weighted least squares, one offset per MIXED ladder, q >= 0. NO FALLBACK:
    NOT IDENTIFIED when the mixed ladders hold an unpriced w1 row."""
    mixed = mixed_ladders(rows)
    models = _models_of(rows, mixed)
    if len(mixed) < MIN_LADDERS or len(models) < MIN_MODELS:
        return {"verdict": f"NOT IDENTIFIED: {len(mixed)} mixed ladders from {len(models)} models (needs {MIN_LADDERS} from "
                           f"{MIN_MODELS})", "ladders": mixed, "models": models, "r1_GBps": None}
    lad = _ladders(rows)
    nf = _unpriced([x for k in mixed for x in lad[k]], w1_why, "mixed")
    if nf:
        return {"verdict": nf, "ladders": mixed, "models": models, "r1_GBps": None}
    use = [x for k in mixed for x in lad[k]]
    X = np.zeros((len(use), len(mixed) + 1))
    for i, x in enumerate(use):
        X[i, mixed.index(x["ladder"])] = 1.0
        X[i, -1] = float(x["cross"]) * x["b_bytes"]
    L = np.array([x["L_us"] for x in use], float)
    y = L - np.array([_ceil_us(x, r_cta) for x in use])
    w = 1.0 / L
    beta, res, se = _lstsq(X, y, w)
    if not _finite(beta, res):
        return {"verdict": NONFINITE, "ladders": mixed, "models": models, "r1_GBps": None}
    q = float(beta[-1])
    clamped = q <= 0.0
    if clamped:
        beta, res, _se0 = _lstsq(X[:, :-1], y, w)
        beta, q = np.append(beta, 0.0), 0.0
    _b0, res0, _s = _lstsq(X[:, :-1], y, w)
    return {"verdict": "READ: weighted least squares, one offset per mixed ladder" + ("; q read <= 0, set to 0: no drain"
                                                                                      if clamped else ""),
            "q_us_per_byte": q, "r1_GBps": None if q == 0.0 else r6(1e-3 / q),
            "se_q_us_per_byte": None if se is None else float(se[-1]), "r_cta_GBps_used": None if r_cta is None else r6(r_cta),
            "ladders": mixed, "models": models, "rows": len(use), "crossing_rows": int(sum(x["cross"] for x in use)),
            "offsets_us": {k: r6(float(beta[i])) for i, k in enumerate(mixed)},
            "rms_rel": r6(float(np.sqrt(np.mean(res ** 2)))), "rms_rel_no_drain": r6(float(np.sqrt(np.mean(res0 ** 2))))}


def read_drain(rows: list[dict], w1rate: dict, subwave: dict) -> dict:
    """DRAIN (DESCRIPTIVE; the wave-boundary drain). Rows: dc_rows; d = 1 when the GEMM's n live CTAs cross a wave
    boundary (n > S), else 0. A LADDER is one (model, T, GEMM), its rows sharing one offset c0 (the counter page's
    per-kernel time outside the stream); it is MIXED when it holds a row with n <= S and a row with n > S. FORMULA per
    decode tile t: q_t minimises sum ((c0 + max(Dc, F, b / r_cta,t) + d b q - L) / L)^2 over the rows of the tile's
    mixed ladders, every c0 free (closed-form weighted least squares), r_cta,t SUBWAVE's reading on the tile (no
    ceiling when SUBWAVE reads none or is NOT IDENTIFIED); q_t read at or under 0 is set to 0 (no drain) with the
    offsets refitted; r1_t = 1 / q_t GB/s, the rate of the one slab per CTA past the boundary. IDENTIFIED with at least 2
    mixed ladders from at least 2 models; NO FALLBACK (an unpriced w1 row in the mixed ladders reads NOT IDENTIFIED).
    Beside: q's standard error, the offsets, the relative rms at the fit and with no drain."""
    rows = dc_rows(rows, w1rate)
    tiles = {}
    for t in DC.DECODE_TILES:
        sw = (subwave.get("tiles") or {}).get(t) or {}
        tiles[t] = fit_drain([r for r in rows if r["tile"] == t], sw.get("r_cta_GBps"), w1_why=_w1_rate(w1rate, t)[1])
    return {"tiles": tiles, "label": "DESCRIPTIVE: the one-slab drain past a wave boundary per tile; no model term"}


def call_rows(pages: dict, refused_rows: list | None = None) -> list[dict]:
    """CALLS' rows: every DS (cell, arm) with a class time (per_class, the median over calls) for align, silu_and_mul or
    moe_sum, at the declared E. A class time that is not finite and positive is left out and appended to
    `refused_rows` (ROW SANITY)."""
    from moe.spec import MODEL_CONFIGS
    rows = []
    for m, _drop in DS_ORDER:
        page = (pages.get(m) or (None, None))[0]
        if page is None:
            continue
        cfg = MODEL_CONFIGS[m]
        doc = page4(m)
        cells = {c["label"]: c for c in doc["cells"]}
        refused = set((page.get("gates") or {}).get("refused_cells") or [])
        for c in page.get("cells") or []:
            pc = cells.get(c.get("label"))
            if pc is None or c.get("key") in refused:
                continue
            arm = c.get("arm")
            E = cfg.num_experts * (int(doc["decl_copies"] or 1) if arm != RD.NATIVE else 1)
            for cls in ("align",) + CALL_CLASSES:
                us = ((c.get("per_class") or {}).get(cls) or {}).get("gpu__time_duration.sum")
                if us is None:
                    continue
                if not _finite_pos(us):
                    if refused_rows is not None:
                        refused_rows.append({"model": m, "label": c["label"], "arm": arm, "class": cls,
                                             "why": f"time {us!r} is not finite and positive"})
                    continue
                row = {"model": m, "label": c["label"], "arm": arm, "class": cls, "T": int(pc["T"]), "E": E,
                       "us": r6(us / 1e3)}
                if cls in CALL_CLASSES:
                    row["x"], row["B_bytes"] = call_x(cfg, int(pc["T"]), cls)
                rows.append(row)
    return rows


def fit_call(rows: list[dict]) -> dict:
    """The DESCRIPTIVE summary line of one call class (see read_calls)."""
    models = sorted({r["model"] for r in rows})
    xs = sorted({r["x"] for r in rows})
    if len(models) < CALL_MIN_MODELS or len(xs) < CALL_MIN_MODELS:
        return {"verdict": f"NOT IDENTIFIED: rows from {len(models)} models with {len(xs)} distinct x (needs "
                           f"{CALL_MIN_MODELS})", "models": models}
    beta, res, se = _lstsq([[1.0, r["x"], r["B_bytes"] / 1e6] for r in rows], [r["us"] for r in rows])
    if not _finite(beta, res):
        return {"verdict": NONFINITE, "models": models}
    return {"verdict": "READ: OLS on [1, x, B / 1 MB]", "a_us": r6(float(beta[0])), "b_us_per_x": float(beta[1]),
            "g_us_per_MB": r6(float(beta[2])), "se": se, "models": models, "x_range": [xs[0], xs[-1]], "rows": len(rows),
            "rms_us": r6(float(math.sqrt(sum(float(v) ** 2 for v in res) / len(rows))))}


def read_calls(pages: dict, refused_rows: list | None = None) -> dict:
    """CALLS (DESCRIPTIVE; the per-call kernel timings per model). Rows: call_rows, NATIVE arms. Per model and class
    (align, silu_and_mul, moe_sum): the median, minimum and maximum us over its cells, and the median per T. Beside, per
    call class, a summary line us = a + b x + g B / 1 MB by ordinary least squares over every NATIVE row (silu_and_mul
    x = I, B = T k 3 I x 2 bytes; moe_sum x = k H, B = (T k H + T H) x 2 bytes; H, I, k from the model's config),
    computed with rows from at least 3 models holding at least 3 distinct x. The line is a summary of DS's pages, not a
    model term and not a rule for any other model."""
    rows = [r for r in call_rows(pages, refused_rows) if r["arm"] == RD.NATIVE]
    per = {}
    for m in sorted({r["model"] for r in rows}):
        per[m] = {}
        for cls in ("align",) + CALL_CLASSES:
            v = [r for r in rows if r["model"] == m and r["class"] == cls]
            if not v:
                continue
            us = [r["us"] for r in v]
            per[m][cls] = {"cells": len(v), "median_us": r6(st.median(us)), "min_us": r6(min(us)), "max_us": r6(max(us)),
                           "median_by_T": {str(T): r6(st.median([r["us"] for r in v if r["T"] == T]))
                                           for T in sorted({r["T"] for r in v})}}
    lines = {c: fit_call([r for r in rows if r["class"] == c]) for c in CALL_CLASSES}
    return {"per_model": per, "summary_lines": lines, "rows": rows,
            "label": "DESCRIPTIVE: the call kernels per model as measured; the summary lines are not a model term"}


def read_block(align: dict) -> dict:
    """BLOCK (DESCRIPTIVE; the block-scan constant per E): ALIGN's block-scan points (the median align-pair ncu time on
    NATIVE block-scan rows per declared E: Granite-3B T 128 A 40 at E 40, OLMoE T 128 A 64 at E 64). IDENTIFIED with
    points at 2 or more E."""
    pts = (align.get("points") or {}).get(RD.BLOCK_SCAN) or []
    if len(pts) < BLOCK_MIN_E:
        return {"verdict": f"NOT IDENTIFIED: block-scan points at {len(pts)} E (needs {BLOCK_MIN_E})", "points": pts}
    return {"verdict": "READ: ALIGN's block-scan points", "points": pts}


def read_align(tree: Path, pages: dict | None = None) -> dict:
    """ALIGN (DESCRIPTIVE; the align kernel across E). Rows: every DS (cell, arm) with an align-class time (per_class
    'align': moe_align_block_size* and count_and_sort_expert_tokens* summed per call, median over calls), at the
    declared E (E native, E x C on a DECL arm) and its path at that E. Small-batch points: per E the median over rows at
    one loop iteration (ceil(numel / max(E, 32)) == 1), NATIVE rows; DECL rows add an E no NATIVE row has (32) only
    when DECL transfers (Mixtral's DECL E 16 median within 1.0 us of Phi's NATIVE E 16 median; else they are dropped).
    Block-scan points: per E the median over NATIVE block-scan rows. SLOPE: fit_slope on the small-batch rows the
    points use (NATIVE, and DECL when DECL transfers). Beside: per E the OLS slope of the small-batch align time against
    the iterations, every arm."""
    from moe.spec import MODEL_CONFIGS
    rows, seen = [], {}
    for m, _drop in DS_ORDER:
        page, why = (pages or {}).get(m) or _ds_page(tree, m)
        seen[m] = why
        if page is None:
            continue
        cfg = MODEL_CONFIGS[m]
        doc = page4(m)
        cells = {c["label"]: c for c in doc["cells"]}
        refused = set((page.get("gates") or {}).get("refused_cells") or [])
        for c in page.get("cells") or []:
            pc = cells.get(c["label"])
            us = ((c.get("per_class") or {}).get("align") or {}).get("gpu__time_duration.sum")
            if pc is None or not _finite_pos(us) or c.get("key") in refused:      # a bad time: listed by call_rows
                continue
            arm = c.get("arm")
            E = cfg.num_experts if arm == RD.NATIVE else cfg.num_experts * int(doc["decl_copies"] or 1)
            g = RD.geometry(cfg, int(pc["T"]), pc["counts"], None if arm == RD.NATIVE else E)
            rows.append({"model": m, "label": c["label"], "arm": arm, "E": E, "T": int(pc["T"]), "numel": g.numel,
                         "path": g.path, "iterations": iterations(g.numel, E), "align_us": r6(us / 1e3)})

    def med(sel):
        v = [r["align_us"] for r in sel]
        return r6(st.median(v)) if v else None
    sb = [r for r in rows if r["path"] == RD.SMALL_BATCH and r["iterations"] == 1]
    nat = {E: med([r for r in sb if r["arm"] == RD.NATIVE and r["E"] == E])
           for E in sorted({r["E"] for r in sb if r["arm"] == RD.NATIVE})}
    dec = {E: med([r for r in sb if r["arm"] != RD.NATIVE and r["E"] == E])
           for E in sorted({r["E"] for r in sb if r["arm"] != RD.NATIVE})}
    chk_n = med([r for r in sb if r["arm"] == RD.NATIVE and r["E"] == DECL_CHECK_E and r["model"] == "phi-3.5-moe"])
    chk_d = med([r for r in sb if r["arm"] != RD.NATIVE and r["E"] == DECL_CHECK_E and r["model"] == "mixtral-8x7b"])
    if chk_n is None or chk_d is None:
        transfer = {"verdict": "NOT READ: a DECL E 16 or a NATIVE E 16 row is missing; the DECL points are dropped",
                    "used": False}
    else:
        ok = abs(chk_d - chk_n) <= DECL_TRANSFER_US
        transfer = {"verdict": "TRANSFERS" if ok else "DOES NOT TRANSFER: the DECL points are dropped", "used": ok,
                    "decl_E16_us": chk_d, "native_E16_us": chk_n, "diff_us": r6(chk_d - chk_n), "cut_us": DECL_TRANSFER_US}
    sb_pts = dict(nat)
    if transfer["used"]:
        for E, v in dec.items():
            if E not in sb_pts:
                sb_pts[E] = v
    bs = [r for r in rows if r["path"] == RD.BLOCK_SCAN and r["arm"] == RD.NATIVE]
    bs_pts = {E: med([r for r in bs if r["E"] == E]) for E in sorted({r["E"] for r in bs})}
    slopes = {}
    for E in sorted({r["E"] for r in rows if r["path"] == RD.SMALL_BATCH}):
        sel = [r for r in rows if r["path"] == RD.SMALL_BATCH and r["E"] == E]
        its = {r["iterations"] for r in sel}
        if len(its) >= 2:
            s, a = np.polyfit([r["iterations"] for r in sel], [r["align_us"] for r in sel], 1)
            slopes[str(E)] = {"slope_us": r6(float(s)), "intercept_us": r6(float(a)), "rows": len(sel)}
    pooled = r6(st.median(x["slope_us"] for x in slopes.values())) if slopes else None
    use = [r for r in rows if r["path"] == RD.SMALL_BATCH and (r["arm"] == RD.NATIVE or transfer["used"])]
    slope = dict(fit_slope(use), per_E=slopes, median_per_E_us=pooled)
    return {"verdict": "READ" if sb_pts or bs_pts else "NOT READ",
            "pages": seen, "rows": rows,
            "points": {RD.SMALL_BATCH: [[E, v] for E, v in sorted(sb_pts.items()) if v is not None],
                       RD.BLOCK_SCAN: [[E, v] for E, v in sorted(bs_pts.items()) if v is not None]},
            "native_E": nat, "decl_E": dec, "DECL_TRANSFER": transfer, "SLOPE": slope,
            "label": "DESCRIPTIVE: the align kernel per path and E, and its cost per loop iteration; no model term"}


def fit_slope(rows: list[dict]) -> dict:
    """SLOPE (DESCRIPTIVE; the align slope). Over the small-batch align rows (align_us, iterations, declared E):
    align_us = a_E + s (iterations - 1), one intercept a_E per declared E and one slope s, by ordinary least squares
    over every row of an E that holds two or more iteration counts. READ when at least SLOPE_MIN_E such E; else NOT
    IDENTIFIED (s None). Beside: the check of s against 0 (s / its OLS standard error, DIFFERS FROM 0 at |t| >= 2) and
    the iteration range read."""
    by = {}
    for r in rows:
        by.setdefault(int(r["E"]), []).append(r)
    es = sorted(E for E, v in by.items() if len({x["iterations"] for x in v}) >= 2)
    use = [r for E in es for r in by[E]]
    if len(es) < SLOPE_MIN_E:
        return {"verdict": f"NOT IDENTIFIED: {len(es)} declared E with two or more iteration counts (needs {SLOPE_MIN_E})",
                "slope_us": None, "E": es, "rows": len(use)}
    X = np.zeros((len(use), len(es) + 1))
    for i, r in enumerate(use):
        X[i, es.index(int(r["E"]))] = 1.0
        X[i, -1] = r["iterations"] - 1
    y = np.array([r["align_us"] for r in use], float)
    beta, _res, _se = _lstsq(X, y)
    if not _finite(beta):
        return {"verdict": NONFINITE, "slope_us": None, "E": es, "rows": len(use)}
    s = float(beta[-1])
    dof = len(use) - X.shape[1]
    res = y - X @ beta
    var = float(res @ res) / dof if dof > 0 else float("nan")
    cov = np.linalg.pinv(X.T @ X)
    se = math.sqrt(var * cov[-1, -1]) if dof > 0 else float("nan")
    t = s / se if se > 0 else None
    if t is None:
        check = "EXACT FIT: no residual to test against 0" if dof > 0 else "NOT TESTED: no degrees of freedom"
    else:
        check = "DIFFERS FROM 0" if abs(t) >= SLOPE_T_CUT else "NOT DISTINGUISHABLE FROM 0"
    its = [r["iterations"] for r in use]
    return {"verdict": "READ: one slope, one intercept per declared E, OLS", "slope_us": r6(s),
            "intercepts_us": {str(E): r6(float(beta[q])) for q, E in enumerate(es)}, "E": es, "rows": len(use),
            "iterations_read": [min(its), max(its)],
            "check_against_0": {"label": "DESCRIPTIVE", "verdict": check, "se_us": None if se != se else r6(se),
                                "t": None if t is None else r6(t), "dof": dof}}


def _results_root(tree: Path) -> Path:
    return tree / "results" if (tree / "results").is_dir() else tree


#: the per-CTA readings: every reading that takes S (the census occupancy) per CTA
PER_CTA = ("LADDERS", "SUBWAVE", "DRAIN", "W1RATE")


def read_ds(tree: Path, prm, census: dict | None = None, cal2_census: dict | None = None) -> dict:
    """Every DS reading, DESCRIPTIVE. The stop rules first: POWER (addendum 2's rule; not READ, nothing is read), then
    PAGE per model; then DS's own census: the census gate (missing or a path mismatch: a STOP for the per-CTA readings),
    CENSUS per tile (the occupancy rule), and the cross-check against CAL2's census (reported; DIFFERS is a STOP for the
    per-CTA readings). Then W1RATE, SUBWAVE, DRAIN, LADDERS, CALLS, ALIGN with BLOCK and SLOPE. `census` and
    `cal2_census` stand in for the files (tests)."""
    power = D2.power_check(Path(tree))
    out = {"what": "the decode settling rental DS's readings under the fourth addendum", "addendum": NAME, "POWER": power,
           "label": "DESCRIPTIVE rental: every reading describes DS's pages; none is a model term, none is a STOP; "
                    + MODEL_AFTER_DS}
    if power["verdict"] != "READ":
        return dict(out, verdict="NOT READ")
    root = _results_root(Path(tree))
    raw = {m: _ds_page(root, m) for m, _d in DS_ORDER}
    out["PAGE"] = {m: page_check(m, p) for m, (p, _w) in raw.items()}
    pages = {m: (p if out["PAGE"][m]["verdict"] == "READ" else None, why if out["PAGE"][m]["verdict"] == "READ"
                 else out["PAGE"][m]["verdict"]) for m, (p, why) in raw.items()}
    ds_doc, ds_why = (census, "given") if census is not None else ds_census(root)
    cal2 = cal2_census if cal2_census is not None else json.loads((REPO / CAL2_CENSUS).read_text())
    out["CENSUS GATE"] = dict(census_gate(ds_doc, ds_why), file=ds_why)
    gate_ok = out["CENSUS GATE"]["verdict"].startswith("READ")
    out["CENSUS"] = (census_check(prm, ds_doc) if gate_ok
                     else {t: {"verdict": f"NOT READ: {out['CENSUS GATE']['verdict']}"} for t in DC.DECODE_TILES})
    out["CENSUS CROSS-CHECK"] = census_cross(prm, ds_doc if gate_ok else None, cal2)
    stop = (out["CENSUS GATE"]["verdict"] if not gate_ok else
            "STOP: DS's census differs from CAL2's: " + "; ".join(out["CENSUS CROSS-CHECK"]["differences"])
            if out["CENSUS CROSS-CHECK"]["verdict"] == "DIFFERS" else None)
    out["PER-CTA STOP"] = {"readings": list(PER_CTA), "verdict": stop or "none"}
    tiles_read = set() if stop else {t for t, x in out["CENSUS"].items() if x["verdict"] == "READ"}
    refused: list = []
    rows = gemm_rows(prm, pages, tiles_read, refused)
    out["pages"] = {m: why for m, (_p, why) in pages.items()}
    out["W1RATE"] = read_w1rate(rows)
    out["SUBWAVE"] = read_subwave(rows, out["W1RATE"])
    out["DRAIN"] = read_drain(rows, out["W1RATE"], out["SUBWAVE"])
    out["LADDERS"] = read_ladders(rows, out["W1RATE"])
    out["CALLS"] = read_calls(pages, refused)
    out["REFUSED ROWS"] = {"rule": f"ROW SANITY: L and B finite and positive, B within {B_BAND[0]:g} to {B_BAND[1]:g} "
                                   "x n b (GEMM rows); class times finite and positive", "rows": refused}
    out["ALIGN"] = read_align(root, pages)
    out["ALIGN"]["BLOCK"] = read_block(out["ALIGN"])
    for k in ("W1RATE", "SUBWAVE", "DRAIN", "LADDERS"):
        for t in DC.DECODE_TILES:
            v = stop if stop else (None if t in tiles_read else f"NOT READ: CENSUS {out['CENSUS'][t]['verdict']}")
            if v is not None:
                out[k]["tiles"][t] = ({"verdict": v, "ladders": {}, "mixed": [], "subwave": []} if k == "LADDERS"
                                      else {"verdict": v})
    if stop:
        out["LADDERS"]["verdict"] = stop
    out["NOT_IDENTIFIED"] = not_identified(out)
    out["verdict"] = "READ" if any(p is not None for p, _w in pages.values()) else "NOT READ"
    return out


def not_identified(r: dict) -> list[str]:
    """Every reading that did not identify (DESCRIPTIVE: reported so the next round knows what DS holds; never a STOP)."""
    why = []
    for k in ("SUBWAVE", "DRAIN", "W1RATE"):
        for t in DC.DECODE_TILES:
            x = ((r.get(k) or {}).get("tiles") or {}).get(t) or {}
            if not str(x.get("verdict", "")).startswith("READ"):
                why.append(f"{k} {t}: {x.get('verdict', 'absent')}")
    for c, x in (((r.get("CALLS") or {}).get("summary_lines")) or {}).items():
        if not str(x.get("verdict", "")).startswith("READ"):
            why.append(f"CALLS line {c}: {x.get('verdict')}")
    a = r.get("ALIGN") or {}
    for k in ("BLOCK", "SLOPE"):
        x = a.get(k) or {}
        if not str(x.get("verdict", "")).startswith("READ"):
            why.append(f"{k}: {x.get('verdict', 'absent')}")
    return why


def ds_identification(prm=None) -> dict:
    """COMPUTED from the registered DS pages alone (geometry, no counter): per decode tile both regimes (SUBWAVE's
    few-CTA long-slab ladders within one wave, DRAIN's mixed ladders across one wave) and their models; W1RATE's usable
    w1 ladders per tile (crossing rows, counted bytes estimated by n b, at least 100 MB apart); the call classes' models
    and distinct x; BLOCK's block-scan E; and whether each clears its IDENTIFIED rule if every unit runs."""
    from moe.spec import MODEL_CONFIGS
    prm = prm or ds_prm()
    rows = gemm_rows(prm)
    out = {"SUBWAVE": {}, "DRAIN": {}, "W1RATE": {}}
    for t in DC.DECODE_TILES:
        tr = [r for r in rows if r["tile"] == t]
        sw, mx = subwave_ladders(tr), mixed_ladders(tr)
        out["SUBWAVE"][t] = {"ladders": sw, "models": _models_of(tr, sw),
                             "few_cta_n": {k: min(x["n"] for x in tr if x["ladder"] == k and not x["cross"]) for k in sw},
                             "identified": len(sw) >= MIN_LADDERS and len(_models_of(tr, sw)) >= MIN_MODELS}
        out["DRAIN"][t] = {"ladders": mx, "models": _models_of(tr, mx),
                           "identified": len(mx) >= MIN_LADDERS and len(_models_of(tr, mx)) >= MIN_MODELS}
        lad = {}
        for r in tr:
            if r["gemm"] == "w1" and r["cross"]:
                lad.setdefault(f"{r['model']}/T{r['T']}", []).append(r["n"] * r["b_bytes"])
        use = sorted(k for k, v in lad.items() if len(v) >= 2 and (max(v) - min(v)) / 1e6 >= W1RATE_MIN_MB)
        mods = sorted({k.split("/")[0] for k in use})
        out["W1RATE"][t] = {"ladders": use, "models": mods,
                            "identified": len(use) >= MIN_LADDERS and len(mods) >= MIN_MODELS}
    out["CALLS"] = {}
    for c in CALL_CLASSES:
        xs = {m: call_x(MODEL_CONFIGS[m], 1, c)[0] for m, _d in DS_ORDER}
        out["CALLS"][c] = {"x_by_model": xs, "identified": len(set(xs.values())) >= CALL_MIN_MODELS}
    bs = sorted({E for m, _d in DS_ORDER for c in page4(m)["cells"]
                 for E in [MODEL_CONFIGS[m].num_experts]
                 if RD.geometry(MODEL_CONFIGS[m], int(c["T"]), c["counts"]).path == RD.BLOCK_SCAN})
    out["BLOCK"] = {"block_scan_E": bs, "identified": len(bs) >= BLOCK_MIN_E}
    its = ds_iterations()
    out["SLOPE"] = {"iterations_by_E": {str(E): v for E, v in its.items()},
                    "identified": sum(len(v) >= 2 for v in its.values()) >= SLOPE_MIN_E}
    out["both_regimes_on_every_tile"] = all(out["SUBWAVE"][t]["identified"] and out["DRAIN"][t]["identified"]
                                            for t in DC.DECODE_TILES)
    out["all_identified"] = (out["both_regimes_on_every_tile"] and all(x["identified"] for x in out["W1RATE"].values())
                             and all(x["identified"] for x in out["CALLS"].values()) and out["BLOCK"]["identified"]
                             and out["SLOPE"]["identified"])
    return out


def tile_params_cal2():
    """The tile stage's v4 Params as the CAL2 scorer used them (decodeB.finalised.json's frozen tile_params)."""
    fz = json.loads((REPO / FINALISED_D3).read_text())["frozen"]
    return V4.Params(**fz["tile_params"])


def ds_prm() -> V4D.DecodeParams:
    """The readings' v4d' parameters (geometry only): CAL2's tile parameters (the census occupancy) and rho form."""
    fz = json.loads((REPO / FINALISED_D3).read_text())["frozen"]
    return V4D.DecodeParams(tile=V4.Params(**fz["tile_params"]), rho_form=fz["rho_form"])


# --------------------------------------------------------------------------
# the published trees reproduce their committed scores (DESCRIPTIVE check)
# --------------------------------------------------------------------------

REL = 1e-9


def tile_source():
    import score_cal as SC
    with contextlib.chdir(DC.REPO):           # the CAL2 score names the tile source by its repo-relative path
        return SC.tile_params(Path("scripts/scoring/tileA2"))


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= REL * max(abs(a), abs(b), 1e-12)


def cal2_check(got: dict, want: dict) -> list[str]:
    """Where a v4d' re-score of the CAL2 tree differs from the committed CAL2 score (empty: the same tree)."""
    why = []
    gs, ws = (got.get("FIT") or {}).get("selected"), (want.get("FIT") or {}).get("selected")
    if gs != ws:
        return [f"rho form {gs} against the committed {ws}"]
    for k in ("T0_GR_ms", "s_naive_GR_ms", "s_block_GR_ms"):
        if not _close(got["FIT"]["forms"][gs].get(k), want["FIT"]["forms"][ws].get(k)):
            why.append(f"{k} {got['FIT']['forms'][gs].get(k)} against {want['FIT']['forms'][ws].get(k)}")
    a, b = got["E1_CAL_IN_SAMPLE"]["stats"], want["E1_CAL_IN_SAMPLE"]["stats"]
    if a.get("cells") != b.get("cells") or not all(_close(a.get(k), b.get(k)) for k in ("rms", "worst", "mean")):
        why.append(f"E1 in-sample {a} against {b}")
    return why


def blind_check(got: dict, want: dict) -> list[str]:
    """Where a v4d' re-score of the decode blind tree differs from the committed blind score (empty: the same tree)."""
    why = []
    a, b = got.get("E1_DEC") or {}, want.get("E1_DEC") or {}
    pairs = [("pooled", (a.get("pooled") or {}).get("stats") or {}, (b.get("pooled") or {}).get("stats") or {})]
    pairs += [(m, ((a.get("per_model") or {}).get(m) or {}).get("stats") or {}, x.get("stats") or {})
              for m, x in sorted((b.get("per_model") or {}).items())]
    for name, g, w in pairs:
        if g.get("cells") != w.get("cells") or not all(_close(g.get(k), w.get(k)) for k in ("rms", "worst", "mean")):
            why.append(f"E1-DEC {name} {g} against {w}")
    return why


def reproduce(cal2_tree: Path, blind_tree: Path, *, want: dict | None = None, want_blind: dict | None = None,
              tile=None) -> dict:
    """The CAL2 tree under v4d' (score_cal_d3's rules, frozen) against the committed CAL2 score, and the decode blind
    tree under the committed finalisation (score_blind_d3.score, frozen) against the committed blind score, to 1e-9
    relative. DESCRIPTIVE: it changes no verdict; the next round's gate will name it as its first clause."""
    import score_blind_d3 as SB3
    import score_cal_d3 as SC3
    want = want if want is not None else json.loads((REPO / CAL2_SCORE).read_text())
    want_blind = want_blind if want_blind is not None else json.loads((REPO / BLIND_SCORE).read_text())
    tp, src = tile if tile is not None else tile_source()
    c = cal2_check(SC3.score(Path(cal2_tree), DC.registration(REPO, "cal"), tp, src, settle=False), want)
    fin3 = json.loads((REPO / FINALISED_D3).read_text())
    b = blind_check(SB3.score(Path(blind_tree), DC.registration(REPO, "blind"), fin3), want_blind)
    return {"CAL2": {"verdict": "REPRODUCES" if not c else "DIFFERS", "why": c},
            "decode blind": {"verdict": "REPRODUCES" if not b else "DIFFERS", "why": b},
            "verdict": "REPRODUCES" if not c and not b else "DIFFERS"}


# --------------------------------------------------------------------------
# what DS touches, and the new models' standing
# --------------------------------------------------------------------------

def no_decode_page(model: str) -> dict:
    """Whether any registered decode page or published decode session names the model (COMPUTED)."""
    regs = sorted(str(p.relative_to(REPO)) for d in (REPO / "docs" / "registered").glob("*decode*pages")
                  for p in d.glob(f"{model}-*.json"))
    pubs = sorted(str(p.relative_to(REPO)) for s in (REPO / "results" / "published").glob("*decode*")
                  for p in (s / "results").glob(f"*-{model}-*") if (s / "results").is_dir())
    return {"registered_pages": regs, "published_dirs": pubs, "verdict": "NO DECODE PAGE" if not regs and not pubs else "HAS A PAGE"}


#: the new blind models' earlier non-decode sessions, FROZEN as earlier_pages computed them on the published tree at
#: b2fb3402 (2026-11-04); the registration carries this list, so a later publish does not change its --check
EARLIER_PAGES_B2FB3402 = {
 "mixtral-8x22b": [
  "2026-09-28-nvidia_gh200_480gb-8x22b-session",
  "2026-10-21-nvidia_gh200_480gb-cal2-session"
 ],
 "qwen2-57b-a14b": [
  "2026-08-22-standard-sweep",
  "2026-08-26-nvidia_h200-full-three-way",
  "2026-08-28-nvidia_h200-h200-fp8-refixed",
  "2026-08-28-nvidia_h200-h200-fp8-three-kernel",
  "2026-08-28-nvidia_h200-h200-v2lite",
  "2026-08-28-nvidia_h200-h200-whole-layer",
  "2026-08-28-nvidia_h200-ridge-resolution",
  "2026-09-01-nvidia_h200-alpha-0558",
  "2026-09-01-nvidia_h200-alpha-surface-s4",
  "2026-09-01-nvidia_h200-cross-card-s3",
  "2026-09-02-nvidia_a100_sxm4_80gb-alpha-surface-s3",
  "2026-09-28-nvidia_gh200_480gb-qwen2-57b-session",
  "2026-10-01-nvidia_gh200_480gb-rental1-session",
  "2026-10-06-nvidia_gh200_480gb-rental3-session",
  "2026-10-07-nvidia_gh200_480gb-rental4-session",
  "2026-10-09-nvidia_gh200_480gb-rental5-session",
  "2026-10-20-nvidia_gh200_480gb-tileA-session",
  "2026-10-21-nvidia_gh200_480gb-settleS1-session",
  "2026-10-21-nvidia_gh200_480gb-tileA2-session",
  "2026-10-27-nvidia_gh200_480gb-cal3-session"
 ],
 "granite-3.0-1b-a400m": [
  "2026-10-01-nvidia_gh200_480gb-rental1-session",
  "2026-10-02-nvidia_gh200_480gb-rental2-session",
  "2026-10-06-nvidia_gh200_480gb-rental3-session"
 ]
}


def earlier_pages(model: str) -> list[str]:
    """Every published session directory (results/published/<session>, three levels down) holding a file or directory
    named for the model, outside the decode sessions (COMPUTED, a live scan): the model's earlier non-decode pages, none
    read here. The registration carries EARLIER_PAGES_B2FB3402, not this scan."""
    import os
    root = REPO / "results" / "published"
    out = set()
    for s in sorted(p for p in root.iterdir() if p.is_dir() and "decode" not in p.name):
        base = len(s.parts)
        for d, dirs, files in os.walk(s):
            if len(Path(d).parts) - base >= 3:
                dirs[:] = []
            if any(model in x for x in dirs + files):
                out.add(s.name)
                break
    return sorted(out)


def ds_touches() -> dict:
    """What DS loads and times: SEEN models only; the E values it declares; nothing of a new blind model."""
    from moe.spec import MODEL_CONFIGS
    models = sorted(m for m, _d in DS_ORDER)
    declared = sorted({MODEL_CONFIGS[m].num_experts * (DS_SPECS[m][0] or 1) for m in models if DS_SPECS[m][0]}
                      | {MODEL_CONFIGS[m].num_experts for m in models})
    shapes = sorted({(MODEL_CONFIGS[m].hidden_size, MODEL_CONFIGS[m].intermediate_size) for m in models})
    new_shapes = {(MODEL_CONFIGS[m].hidden_size, MODEL_CONFIGS[m].intermediate_size) for m in NEW_BLIND}
    return {"models": models, "all_seen": all(m in SEEN_MODELS for m in models),
            "new_blind_models_loaded": sorted(set(models) & set(NEW_BLIND)),
            "declared_E": declared, "gemm_shapes_HI": [list(s) for s in shapes],
            "new_blind_shapes_measured": sorted(list(s) for s in set(shapes) & new_shapes)}
