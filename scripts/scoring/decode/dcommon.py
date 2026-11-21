"""The decode stage's shared rules (docs/registered/2026-10-14-decodeC-cal-gh200 and
2026-10-14-decodeB-blind-gh200). One module for the registration builder, the CAL scorer, the
mechanical finalisation and the blind scorer, so a rule is computed one way everywhere. It imports
the tile stage's tcommon (and through it rentals 3 to 7); nothing is copied.

THE RULES THIS FILE OWNS:
  row_drop      a decode row is valid when it has a time, at least MIN_CLEAN (6 of 9) clean repeats
                under r7-median-run3 (rebuilt from cells.csv), was not refused (V-ALIGN, GD5, V-BIN),
                passes GD1 (G2: host-bound excluded), carries no timer host flag and its clock is at
                least G4_MIN_MHZ
  e_dec         E1-DEC's per-cell error: (pred - meas) / meas, an absolute error at or under 1 us = 0
  bar           E1-DEC: rms, worst and |mean| against the bars (provisional DD5: 3%, 8%, 1.5%)
  tighten       the bars from the CAL replicate's sigma_page: rms = min(3%, max(1.5%, 5 sigma)),
                worst = min(8%, max(4%, 8/3 rms)), mean = min(1.5%, max(0.75%, rms / 2)); never looser
  fit + rho     the three CAL constants by v4d.fit_constants under each rho form; the form with the
                lower CAL rms on SUB-WAVE cells (some GEMM under one wave) is selected, "fixed" unless
                "scaled"'s rms is lower by at least RHO_MARGIN ABSOLUTE (0.1 percentage points of rms)
  g3d           G3-D's three TOSTs (shuffle, label, fill) on d_us = (T_a - T_b) - (P_a - P_b),
                clusters (model, T) (fill: (model, tile band)), margin max(0.3 us, 0.36% x the median
                T_b); shuffle and label failing is a STOP; fill is a model test (FILL0)
  gd2           GR1 - GRR per call on matched cells: the CAL band [p05 - 1 us, p95 + 1 us], frozen;
                CAL's median outside [-1, 10] us, a band wider than 8 us, or a band leaving [-2, 11] us is
                a STOP; a blind GR1 page with over GD2_VOID_FRAC of
                its cells outside the band voids that model's GRR page
  gd3           E240 - GR1 in [0, 10] us on cells GPU-bound eager (C + F240 > 1.25 H): DESCRIPTIVE
  stop          the CAL STOP rule (see `stop_rule`): every nodrop CAL unit scored and valid, the fit
                FITTED, in-sample E1-DEC within the provisional bars, G3-D shuffle and label
                EQUIVALENT, no V-ALIGN refusal on a CAL cell, GD2's median in [-1, 10] us
"""
from __future__ import annotations

import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# the tile stage's and rental 7's directories go AFTER this one: their register.py and finalise.py
# must never shadow a decode module by name
for _p in (str(HERE.parent / "tileA"), str(HERE.parent / "rental7"), str(HERE.parent / "rental6")):
    if _p not in sys.path:
        sys.path.append(_p)


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TC = _load("tcommon", HERE.parent / "tileA" / "tcommon.py")
C7, C6, C5, C3 = TC.C7, TC.C6, TC.C5, TC.C3
CARD = TC.CARD
r6 = TC.r6

import r3_decode as RD  # noqa: E402
import r3_timing_model_v4 as V4  # noqa: E402
import r3_timing_model_v4d as V4D  # noqa: E402

DATE = "2026-10-14"
PARTS = ("cal", "blind")
NAMES = {"cal": f"{DATE}-decodeC-cal-gh200", "blind": f"{DATE}-decodeB-blind-gh200"}
PAGES = f"docs/registered/{DATE}-decode-pages"
CAL_MODELS = ("olmoe-1b-7b", "mixtral-8x7b")
BLIND_MODELS = ("phi-3.5-moe", "qwen1.5-moe-a2.7b", "jetmoe-8b", "deepseek-v2-lite", "granite-3.0-3b-a800m")
LOCK_MHZ = 1710
MIN_CLEAN = C6.MIN_CLEAN
G4_MIN_MHZ = 1705.0
FLOOR_US = 1.0
BARS_PROVISIONAL = {"rms_max": 0.03, "max_abs": 0.08, "mean_max": 0.015}
BARS_FLOOR = {"rms_max": 0.015, "max_abs": 0.04, "mean_max": 0.0075}
RHO_MARGIN = 0.001
G3D_REL, G3D_US = 0.0036, 0.3
GD2_EXPECTED_US = (0.0, 5.0)
GD2_STOP_US = (-1.0, 10.0)
GD2_PAD_US = 1.0
#: the frozen GR1 band's width cap (build-decode review 4): a wider CAL band is a STOP
GD2_MAX_WIDTH_US = 8.0
GD2_VOID_FRAC = 0.20
GD3_US = (0.0, 10.0)
GD3_HOST_FACTOR = 1.25
F240_MS = TC.F240_MS
SIGMA_PAGE_FALLBACK = TC.SIGMA_PAGE_R3

#: the CAL rental's units in plan order: (label, model, page kind, step, inst, drop group or nodrop, est min)
CAL_UNITS = (
    ("dm", "olmoe-1b-7b", "main", "r3decode", "GRR", "nodrop", 17),
    ("dm", "mixtral-8x7b", "main", "r3decode", "GRR", "nodrop", 16),
    ("dc", "olmoe-1b-7b", "ctl", "r3decode", "GR1", "nodrop", 9),
    ("dc", "mixtral-8x7b", "ctl", "r3decode", "GR1", "nodrop", 9),
    ("de", "mixtral-8x7b", "e240", "r3decode", "E240", "e240", 6),
    ("dn", "olmoe-1b-7b", "ctl", "decodebytes", None, "ncu", 5),
    ("dn", "mixtral-8x7b", "ctl", "decodebytes", None, "ncu", 6),
    ("dd", "olmoe-1b-7b", "decl", "r3decode", "GRR", "decl", 5),
    ("dd", "mixtral-8x7b", "decl", "r3decode", "GRR", "decl", 5),
    ("dr", "olmoe-1b-7b", "main", "r3decode", "GRR", "rep", 17),
)
#: the blind rental's units (frozen now; the plan is generated at finalisation)
BLIND_UNITS = tuple(("dm", m, "main", "r3decode", "GRR", "nodrop", est) for m, est in (
    ("phi-3.5-moe", 15), ("qwen1.5-moe-a2.7b", 18), ("jetmoe-8b", 15), ("deepseek-v2-lite", 15),
    ("granite-3.0-3b-a800m", 15))) + (
    ("dc", "qwen1.5-moe-a2.7b", "ctl", "r3decode", "GR1", "gr1", 9),
    ("dc", "granite-3.0-3b-a800m", "ctl", "r3decode", "GR1", "gr1", 9))
#: the four decode tiles (tile stage census format)
DECODE_TILES = ("16x64x128w4s4", "32x64x128w4s3", "32x128x64w4s3", "64x128x64w4s3")
#: DESIGN-NOMINAL, for scale, levers and power only (never a registered prediction): the tile stage's
#: nominal wgmma constants and a T0_GR of 25 us
NOMINAL = {"T0_GR_ms": 0.025, "s_naive_GR_ms": 0.0, "s_block_GR_ms": 0.0,
           "tile": {"c128": 900.0, "F_wg": {"w1": 520.0, "w2": 979.0}}}


def registration(repo: Path, part: str) -> dict:
    return json.loads((Path(repo) / "docs" / "registered" / f"{NAMES[part]}.json").read_text())


def page_path(model: str, kind: str) -> str:
    return f"{PAGES}/{model}-{kind}.json"


def page_doc(model: str, kind: str, repo: Path = REPO) -> dict:
    return json.loads((Path(repo) / page_path(model, kind)).read_text())


def nominal_params(rho_form: str = "fixed") -> V4D.DecodeParams:
    t = V4.Params(c128=NOMINAL["tile"]["c128"], F_wg=dict(NOMINAL["tile"]["F_wg"]))
    return V4D.DecodeParams(T0_GR_ms=NOMINAL["T0_GR_ms"], rho_form=rho_form, tile=t)


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------

def row_drop(row: dict) -> str | None:
    """None when a rebuilt decode row is a valid cell, else why it is dropped (in rule order)."""
    if row.get("refused"):
        return "refused (V-ALIGN, GD5 or V-BIN)"
    if row.get("failed"):
        return "failed (no repeat ran)"
    if row.get("ms_p50") is None:
        return "no time"
    if int(row.get("clean_repeats") or 0) < MIN_CLEAN:
        return f"{row.get('clean_repeats')} clean repeats, under {MIN_CLEAN}"
    if (row.get("gd1") or {}).get("verdict") != "PASS":
        return "G2 (GD1): host-bound by the probe"
    if row.get("host_bound"):
        return "G2: the timer's host flag"
    clk = row.get("sm_clock_load_mhz")
    if clk is None or float(clk) < G4_MIN_MHZ:
        return f"G4 clock {clk}"
    return None


def e_dec(pred_ms: float, meas_ms: float) -> float:
    return V4D.rel_err(pred_ms, meas_ms, FLOOR_US)


def bar(v: list[float], rule: dict) -> dict:
    s = C5.stats(v)
    if not s["cells"]:
        return {"verdict": "NOT SCORED: no valid cell", "stats": s, "bars": rule}
    ok = s["rms"] <= rule["rms_max"] and abs(s["worst"]) <= rule["max_abs"] and abs(s["mean"]) <= rule["mean_max"]
    return {"verdict": "HOLDS" if ok else "FAILS", "stats": s, "bars": rule}


def tighten(sigma: float | None) -> dict:
    """The blind bars from the CAL replicate's sigma_page (None: the provisional bars)."""
    if sigma is None:
        return dict(BARS_PROVISIONAL, source="provisional (no CAL replicate)")
    rms = min(BARS_PROVISIONAL["rms_max"], max(BARS_FLOOR["rms_max"], 5.0 * sigma))
    worst = min(BARS_PROVISIONAL["max_abs"], max(BARS_FLOOR["max_abs"], rms * 8.0 / 3.0))
    mean = min(BARS_PROVISIONAL["mean_max"], max(BARS_FLOOR["mean_max"], rms / 2.0))
    return {"rms_max": r6(rms), "max_abs": r6(worst), "mean_max": r6(mean),
            "source": f"tightened from the CAL replicate's sigma_page {sigma:.5f}"}


def sigma_rep(pairs: list[tuple[float, float]]) -> float | None:
    """sigma_page from (main, replicate) NATIVE cell pairs: rms(rep / main - 1) / sqrt(2)."""
    d = [b / a - 1 for a, b in pairs if a and b]
    if len(d) < 6:
        return None
    return math.sqrt(st.fmean(x * x for x in d)) / math.sqrt(2.0)


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------

def decode_page(tree: Path, model: str, label: str, reg_file: dict, inst: str):
    """(rebuilt report, why) for one r3decode unit's page: one report under gaps-<card>-<model>-<label>,
    its page file sha256 the registered one, its instrument the unit's, every row's counts sha256
    the registered cell's, then rental 7's r7-median-run3 rebuild from cells.csv. GD4 failing voids."""
    root = C3.find_gaps(tree, model, label)
    pages = [(p, r) for p, r in C3.timed_pages(root) if r.get("kind") == "decode-page"]
    if len(pages) != 1:
        return None, f"NOT SCORED: {len(pages)} decode pages under gaps-{model}-{label}"
    p, r = pages[0]
    h = r.get("histogram") or {}
    if h.get("file_sha256") != reg_file["sha256"]:
        return None, "NOT SCORED (G5): the page's decode-page sha256 is not the registered one"
    if r.get("inst") != inst:
        return None, f"NOT SCORED: the page ran inst {r.get('inst')}, the unit's is {inst}"
    want = {c["label"]: c["counts_sha256"] for c in reg_file["cells"]}
    for row in r.get("treads_table") or []:
        if want.get(str(row.get("histogram"))) != row.get("counts_sha256"):
            return None, f"NOT SCORED (G5): row {row.get('histogram')} carries another counts sha256"
    if r.get("slip_policy") != "cell" or r.get("min_clean_repeats") != 0:
        return None, "NOT SCORED: the page did not run --slip-policy cell --min-clean-repeats 0"
    gd4 = (r.get("histogram_gates") or {}).get("GD4_graph_fidelity", {})
    if gd4.get("verdict") == "FAIL":
        return None, f"VOID (GD4): captured calls differ from the eager output ({gd4.get('failed')})"
    _rep, raw = C7.read_page(Path(p).parent)
    r, raw, out_cells = drop_dead_cells(r, raw)
    rb, why = C7.rebuild(r, raw)
    if rb is None:
        return None, f"NOT SCORED (rebuild): {why}"
    rb["treads_table"] = rb["treads_table"] + out_cells
    rb["g1_r7"]["refused_or_failed_cells_excluded"] = [f"{x['histogram']}/{x['arm']}/T{x['tiles']}" for x in out_cells]
    g = rb["g1_r7"]
    if g["verdict"] != "PASS":
        return None, f"G1 FAIL (rebuilt): {g['non_clean_rows']} of {g['repeat_rows']} rows non-clean"
    return rb, "counted"


def drop_dead_cells(report: dict, raw: list | None):
    """(report, raw, removed rows): a REFUSED cell (V-ALIGN, GD5, V-BIN) or a FAILED one (no repeat with
    status ok) is taken out of the r7-median-run3 rebuild, rows and cells.csv repeats both, before G1's
    non-clean share is counted, so a refusal cannot fail the page the registration says stands
    (build-decode review section 8). The removed rows return as lost, refused, untimed rows."""
    import copy
    if raw is None:
        return report, raw, []
    ok = {}
    for x in raw:
        m = __import__("re").search(r"histogram=([^;]+)", x.get("detail") or "")
        if m:
            k = (m.group(1), int(x["tiles"]), x["arm"])
            ok[k] = ok.get(k, False) or (x.get("status", "ok") == "ok")
    r = copy.deepcopy(report)
    dead, keep = [], []
    for t in r.get("treads_table") or []:
        k = (str(t["histogram"]), int(t["tiles"]), str(t["arm"]))
        if t.get("refused") or not ok.get(k, False):
            dead.append(dict(t, ms_p50=None, lost=True, refused=True if t.get("refused") else t.get("refused", False),
                             failed=not t.get("refused"), clean_repeats=0))
        else:
            keep.append(t)
    if not dead:
        return report, raw, []
    gone = {(str(t["histogram"]), int(t["tiles"]), str(t["arm"])) for t in dead}

    def key(x):
        m = __import__("re").search(r"histogram=([^;]+)", x.get("detail") or "")
        return (m.group(1), int(x["tiles"]), x["arm"]) if m else None
    raw2 = [x for x in raw if key(x) not in gone]
    r["treads_table"] = keep
    if r.get("repeat_rows") is not None:
        r["repeat_rows"] = len(raw2)
    return r, raw2, dead


def measured(report: dict, model: str, kind: str) -> tuple[dict, list]:
    """({(model, kind, label, T, arm): row}, dropped) of a rebuilt page's valid rows."""
    out, dropped = {}, []
    for row in report.get("treads_table") or []:
        k = (model, kind, str(row["histogram"]), int(row["tiles"]), str(row["arm"]))
        why = row_drop(row)
        if why:
            dropped.append(["/".join(map(str, k)), why])
        else:
            out[k] = row
    return out, dropped


# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------

def price_cell(model: str, cell: dict, prm: V4D.DecodeParams, arm: str = "native", decl_copies: int | None = None) -> dict | None:
    """v4d at one page cell; None when v4d cannot price it: NOT PRICED, out of E1-DEC and LISTED by
    every scorer (the wgmma decode tile with no tile-stage constants, or a path whose CAL offset was
    not fitted). The CAL scorer refuses without the tile-stage selections and STOPs on any NOT PRICED
    CAL cell; the blind scorer lists them per model."""
    from moe.spec import MODEL_CONFIGS
    E = MODEL_CONFIGS[model].num_experts
    try:
        p = V4D.price(model, int(cell["T"]), cell["counts"], prm,
                      declared=None if arm == "native" else E * int(decl_copies or 1))
    except V4.TM.Refused:
        return None
    if prm.T0_GR_ms is not None and p["ms"] is None:
        return None                     # the path's offset was not fitted: NOT PRICED, never s = 0
    return p


def fit_and_select(rows: list[dict], tile: V4.Params) -> dict:
    """rows {model, cell, meas_ms}: the three constants under each rho form, and the selection."""
    out = {}
    for form in V4D.RHO_FORMS:
        prm = V4D.DecodeParams(rho_form=form, tile=tile)
        fr, sub = [], []
        for r in rows:
            p = price_cell(r["model"], r["cell"], prm)
            if p is None:
                continue
            fr.append({"meas_ms": r["meas_ms"], "body_ms": p["body_ms"], "path": p["path"], "sub_wave": p["sub_wave"],
                       "key": r["key"]})
        fit = V4D.fit_constants(fr)
        if fit.get("verdict") == "FITTED":
            e_all, e_sub = [], []
            for x in fr:
                const = fit["T0_GR_ms"] + {RD.SMALL_BATCH: 0.0, RD.NAIVE: fit["s_naive_GR_ms"],
                                           RD.BLOCK_SCAN: fit["s_block_GR_ms"]}[x["path"]]
                e = e_dec(x["body_ms"] + const, x["meas_ms"])
                e_all.append(e)
                if x["sub_wave"]:
                    e_sub.append(e)
            fit["in_sample"] = C5.stats(e_all)
            fit["sub_wave"] = C5.stats(e_sub)
        out[form] = fit
    f, s = out["fixed"], out["scaled"]
    if f.get("verdict") != "FITTED" and s.get("verdict") != "FITTED":
        sel = None
    elif s.get("verdict") != "FITTED":
        sel = "fixed"
    elif f.get("verdict") != "FITTED":
        sel = "scaled"
    else:
        fr_, sr_ = (f["sub_wave"]["rms"] or 0.0), (s["sub_wave"]["rms"] or 0.0)
        sel = "scaled" if sr_ < fr_ - RHO_MARGIN else "fixed"
    return {"forms": out, "selected": sel,
            "rule": f"the lower CAL rms on sub-wave cells (sub-wave: some GEMM under one wave at the census "
                    f"occupancy); fixed unless scaled's rms is lower by {RHO_MARGIN * 100:.1f} percentage points "
                    "(absolute)"}


# --------------------------------------------------------------------------
# G3-D, GD2, GD3
# --------------------------------------------------------------------------

def g3d_pairs(page: dict, kind: str) -> list[tuple[str, str, str]]:
    """(cluster id, label a, label b) for one G3-D test on a main page (b the reference)."""
    cells = {c["label"]: c for c in page["cells"]}
    out = []
    for c in page["cells"]:
        if kind == "shuffle" and c["role"] == "UNIFORM-UNSHUFFLED":
            out.append((f"T{c['T']}", c["pair"], c["label"]))
        if kind == "label" and c["role"] == "MINCOVER" and c.get("pair") in cells:
            out.append((f"T{c['T']}", c["label"], c["pair"]))
    if kind == "fill":
        from r3_decode import tile_key, default_config
        from moe.spec import MODEL_CONFIGS
        E = MODEL_CONFIGS[page["model"]].num_experts
        tops = [c for c in page["cells"] if c["role"] == "BALANCED"]
        for a in tops:
            for b in tops:
                A_a = sum(1 for v in a["counts"] if v)
                A_b = sum(1 for v in b["counts"] if v)
                if (b["T"] < a["T"] and A_a == A_b
                        and tile_key(default_config(a["T"], E)) == tile_key(default_config(b["T"], E))):
                    out.append((tile_key(default_config(a["T"], E)), a["label"], b["label"]))
    return out


def g3d(meas: dict, preds: dict, pairs_by_model: dict, test: str) -> dict:
    """One G3-D TOST over every model's pairs: meas {(model, label): ms}, preds {(model, label): ms}."""
    by, rows, ref = {}, [], []
    for model, pairs in pairs_by_model.items():
        for cl, a, b in pairs:
            ka, kb = (model, a), (model, b)
            if ka in meas and kb in meas and ka in preds and kb in preds:
                d_us = ((meas[ka] - meas[kb]) - (preds[ka] - preds[kb])) * 1e3
                by.setdefault(f"{model}/{cl}", []).append(d_us)
                rows.append({"model": model, "a": a, "b": b, "d_us": r6(d_us)})
                ref.append(meas[kb] * 1e3)
    if not rows:
        return {"verdict": "NOT SCORED: no pair with both cells valid", "rows": []}
    margin = max(G3D_US, G3D_REL * st.median(ref))
    t = C5.tost_cluster_t(by, margin)
    return {"verdict": "EQUIVALENT" if t["verdict"] == "EQUIVALENT" else t["verdict"], "tost": t,
            "margin_us": r6(margin), "rows": rows, "test": test}


def gd2_band(diffs_us: list[float]) -> dict:
    """GR1 - GRR per call on the CAL's matched cells: the frozen band and the STOP clause."""
    if len(diffs_us) < 6:
        return {"verdict": f"NOT SCORED: {len(diffs_us)} matched cells", "band_us": None}
    s = sorted(diffs_us)
    q = lambda p: s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]
    med = st.median(s)
    band = [r6(q(0.05) - GD2_PAD_US), r6(q(0.95) + GD2_PAD_US)]
    why = []
    if not GD2_STOP_US[0] <= med <= GD2_STOP_US[1]:
        why.append(f"median {med:.2f} us outside {list(GD2_STOP_US)}")
    if band[1] - band[0] > GD2_MAX_WIDTH_US:
        why.append(f"band width {band[1] - band[0]:.2f} us over the {GD2_MAX_WIDTH_US:g} us cap")
    if band[0] < GD2_STOP_US[0] - GD2_PAD_US or band[1] > GD2_STOP_US[1] + GD2_PAD_US:
        why.append(f"band {band} leaves [{GD2_STOP_US[0] - GD2_PAD_US:g}, {GD2_STOP_US[1] + GD2_PAD_US:g}] us")
    return {"verdict": "PASS" if not why else "FAIL (a STOP): " + "; ".join(why), "median_us": r6(med),
            "band_us": band, "cells": len(s), "max_width_us": GD2_MAX_WIDTH_US,
            "expected_us": list(GD2_EXPECTED_US), "stop_us": list(GD2_STOP_US),
            "in_expected": sum(GD2_EXPECTED_US[0] <= d <= GD2_EXPECTED_US[1] for d in s)}


def gd2_blind(diffs_us: list[float], band: list[float]) -> dict:
    out = [d for d in diffs_us if not band[0] <= d <= band[1]]
    frac = len(out) / len(diffs_us) if diffs_us else 0.0
    return {"verdict": "VOID (the model's GRR page)" if frac > GD2_VOID_FRAC else "PASS", "outside": len(out),
            "cells": len(diffs_us), "fraction": r6(frac), "band_us": band}


def gd3(rows: list[dict]) -> dict:
    """rows {e240_ms, gr1_ms, grr_ms, host_ms}: E240 - GR1 in [0, 10] us on cells GPU-bound eager."""
    sel = [r for r in rows if r["grr_ms"] + F240_MS > GD3_HOST_FACTOR * (r["host_ms"] or 0.0)]
    d = [(r["e240_ms"] - r["gr1_ms"]) * 1e3 for r in sel]
    inb = sum(GD3_US[0] <= x <= GD3_US[1] for x in d)
    return {"verdict": "DESCRIPTIVE", "cells": len(d), "in_band": inb, "band_us": list(GD3_US),
            "median_us": r6(st.median(d)) if d else None}


# --------------------------------------------------------------------------
# the STOP rule (blind runs only on a GO)
# --------------------------------------------------------------------------

def stop_rule(score: dict) -> dict:
    why = []
    for key, state in (score.get("units") or {}).items():
        if state.get("nodrop") and state.get("state") != "counted":
            why.append(f"nodrop unit {key}: {state.get('state')}")
    fit = score.get("FIT") or {}
    sel = fit.get("selected")
    if sel is None:
        why.append("the CAL fit is NOT FITTED under either rho form")
    e1 = (score.get("E1_CAL_IN_SAMPLE") or {}).get("verdict", "NOT SCORED")
    if e1 != "HOLDS":
        why.append(f"in-sample E1-DEC on the CAL main pages {e1} at the provisional bars")
    for t in ("shuffle", "label"):
        v = (score.get("G3D") or {}).get(t, {}).get("verdict", "NOT SCORED")
        if v != "EQUIVALENT":
            why.append(f"G3-D {t} {v}: time is not shown to be a function of counts alone")
    if score.get("tile_selections_missing"):
        why.append("the tile stage's selections (tile rental A) were not given: T 97..128 cannot be priced")
    np_ = score.get("NOT_PRICED") or []
    if np_:
        why.append(f"{len(np_)} valid CAL NATIVE cells NOT PRICED (no tile-stage constants, or an unfitted offset)")
    ref = score.get("V_ALIGN_refused") or []
    if ref:
        why.append(f"{len(ref)} CAL cells refused by V-ALIGN or GD5: the decode rules are not vLLM's")
    g2 = (score.get("GD2") or {}).get("verdict", "NOT SCORED")
    if not str(g2).startswith("PASS"):
        why.append(f"GD2 {g2}")
    return {"verdict": "GO" if not why else "STOP", "why": why}
